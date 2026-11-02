"""Live CS2 plugin ingest.

Accepts length-delimited msgpack frames from a Source2 server plugin
(see ``PLUGIN_PROTOCOL.md``), assembles a ``Session``, and either:

  * writes a ``.aegis`` capture for offline training / review, and/or
  * scores the match through the loaded ensemble + SPRT adjudicator.

Sequence gaps invalidate the affected window rather than inventing motion —
a dropped tick looks like a teleport to every kinematic feature.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..adapters.base import effective_capabilities
from ..adapters.cs2.adapter import CS2Adapter
from ..adapters.cs2.capture import decode_frames_from_buffer, encode_frame, write_capture
from ..core.config import AegisConfig
from ..core.types import Session
from ..features.pipeline import extract_engagement_tensors, extract_session_features
from ..fusion.policy import Adjudicator
from ..fusion.prior import trust_from_player_info
from ..fusion.sprt import EvidenceState, SPRT
from ..integrity.audit import AuditLog, make_audit_record
from ..models.calibrate import prob_to_llr
from ..models.ensemble import CalibratedEnsemble

log = logging.getLogger("aegis.ingest")


@dataclass
class StreamState:
    """One TCP connection = one match stream."""

    peer: str
    frames: list[dict] = field(default_factory=list)
    last_seq: int | None = None
    gaps: list[tuple[int, int]] = field(default_factory=list)
    buf: bytearray = field(default_factory=bytearray)
    started: float = field(default_factory=time.time)
    session_id: str = ""


@dataclass
class IngestResult:
    session_id: str
    capture_path: str | None
    gaps: list[tuple[int, int]]
    n_frames: int
    players: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


ScoreHook = Callable[[Session, list[tuple[int, int]]], IngestResult | None]


class IngestHub:
    """Shared scoring / capture logic used by the TCP server and lab clients."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        capture_dir: str | Path | None = None,
        config: AegisConfig | None = None,
        ensemble: CalibratedEnsemble | None = None,
    ):
        self.cfg = config or AegisConfig(model_dir=str(model_dir or "models"))
        self.capture_dir = Path(capture_dir) if capture_dir else None
        self.adapter = CS2Adapter()
        self.ensemble = ensemble
        if self.ensemble is None and model_dir and (Path(model_dir) / "ensemble.json").exists():
            self.ensemble = CalibratedEnsemble.load(model_dir, device=self.cfg.model.device)
        self.states: dict[str, EvidenceState] = {}
        self.adjudicator = Adjudicator(policy=self.cfg.policy, sprt=SPRT(self.cfg.fusion))
        self.audit = (
            AuditLog(self.cfg.policy.audit_log_path)
            if self.cfg.policy.audit_log_path else None
        )

    def ingest_frames(
        self,
        frames: list[dict],
        *,
        gaps: list[tuple[int, int]] | None = None,
        write: bool = True,
        score: bool = True,
    ) -> IngestResult:
        gaps = gaps or []
        session = self.adapter.parse_frames(frames)
        if gaps:
            # Mark invalidated windows; features that care can refuse them.
            session.meta["seq_gaps"] = [{"from": a, "to": b} for a, b in gaps]
            session.meta["partial"] = True

        capture_path = None
        if write and self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = str(
                write_capture(self.capture_dir / f"{session.session_id}.aegis", frames)
            )

        players: list[dict] = []
        if score and self.ensemble is not None:
            players = self._score_session(session)

        return IngestResult(
            session_id=session.session_id,
            capture_path=capture_path,
            gaps=gaps,
            n_frames=len(frames),
            players=players,
            meta={"partial": bool(gaps), "game": session.game},
        )

    def _score_session(self, session: Session) -> list[dict]:
        assert self.ensemble is not None
        caps = effective_capabilities(self.adapter, session)

        feats = extract_session_features(
            session, caps, cfg=self.cfg.features, spec=self.ensemble.spec
        )
        tensors = extract_engagement_tensors(
            session, caps, cfg=self.cfg.features, seq_len=self.cfg.model.seq_len
        )
        by_slot: dict[int, list] = {}
        for t in tensors:
            by_slot.setdefault(t.slot, []).append(t)

        import numpy as np

        out: list[dict] = []
        for f in feats:
            info = session.players[f.slot]
            prior = trust_from_player_info(info)
            seqs = scals = None
            if f.slot in by_slot:
                seqs = np.stack([t.data for t in by_slot[f.slot]])
                scals = np.stack([t.scalars for t in by_slot[f.slot]])
            score = self.ensemble.score_player(
                session_id=session.session_id, slot=f.slot, player_id=info.player_id,
                features=f.values, engagement_seqs=seqs, engagement_scalars=scals,
                prior=prior.p,
            )
            st = self.states.get(info.player_id) or EvidenceState(player_id=info.player_id)
            nov_llr = float(prob_to_llr(np.array([score.p_novelty]), prior=prior.p)[0]) * 0.4
            self.adjudicator.sprt.observe_session(
                st,
                session_id=session.session_id,
                session_llr=score.llr_session,
                engagement_llrs=score.engagement_llrs,
                novelty_llr=nov_llr,
                t=time.time(),
            )
            self.states[info.player_id] = st
            verd = self.adjudicator.decide(
                st, score, prior,
                n_engagements_lifetime=int(st.n_units) + score.meta.get("n_engagements", 0),
                title=session.game,
            )
            out.append({
                "player_id": info.player_id,
                "slot": f.slot,
                "score": score.to_dict(),
                "verdict": verd.to_dict(),
                "evidence": {
                    "log_odds": st.log_odds,
                    "n_units": st.n_units,
                    "posterior": st.posterior(prior.p),
                },
            })
            if self.audit is not None:
                self.audit.append(make_audit_record(
                    session_id=session.session_id,
                    player_id=info.player_id,
                    verdict=verd,
                    score=score,
                    shadow=self.cfg.policy.shadow_mode,
                    model_digest=self.ensemble.digest,
                ))
        return out


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    hub: IngestHub,
    token: str = "",
) -> None:
    addr = writer.get_extra_info("peername")
    peer = f"{addr[0]}:{addr[1]}" if addr else "unknown"
    st = StreamState(peer=peer)
    log.info("ingest connect %s", peer)
    # With a token configured, the first frame on every connection must be
    # {"k": "auth", "token": ...}; anything else is dropped unscored, so a host
    # that can merely reach the port cannot inject telemetry or verdicts.
    authed = not token
    rejected = False
    try:
        while not rejected:
            try:
                chunk = await reader.read(65536)
            except (ConnectionError, OSError) as e:
                # A crashed or force-killed game server resets the connection
                # instead of closing it. Keep whatever we already decoded.
                log.warning("connection lost %s: %s", peer, e)
                break
            if not chunk:
                break
            st.buf.extend(chunk)
            try:
                frames = decode_frames_from_buffer(st.buf)
            except Exception as e:
                log.warning("bad frame from %s: %s", peer, e)
                break
            for fr in frames:
                if fr.get("k") == "auth":
                    if not authed:
                        authed = hmac.compare_digest(
                            str(fr.get("token", "")).encode(), token.encode()
                        )
                        if not authed:
                            log.warning("ingest from %s: wrong token; closing", peer)
                            rejected = True
                            break
                    continue
                if not authed:
                    log.warning("ingest from %s: no auth frame; closing", peer)
                    rejected = True
                    break
                seq = fr.get("seq")
                if seq is not None:
                    seq = int(seq)
                    if st.last_seq is not None and seq > st.last_seq + 1:
                        st.gaps.append((st.last_seq, seq))
                        log.warning("seq gap %s: %s -> %s", peer, st.last_seq, seq)
                    st.last_seq = seq
                if fr.get("k") == "hdr":
                    st.session_id = str(fr.get("session_id", ""))
                st.frames.append(fr)
        if rejected:
            pass
        elif st.frames:
            result = hub.ingest_frames(st.frames, gaps=st.gaps)
            log.info(
                "ingest closed %s session=%s frames=%d gaps=%d capture=%s players=%d",
                peer, result.session_id, result.n_frames, len(result.gaps),
                result.capture_path, len(result.players),
            )
            try:
                writer.write(encode_frame({
                    "k": "ack",
                    "session_id": result.session_id,
                    "n_frames": result.n_frames,
                    "gaps": len(result.gaps),
                }))
                await writer.drain()
            except Exception:
                pass
        else:
            log.warning("ingest closed %s with no frames", peer)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def run_ingest_server(
    host: str = "0.0.0.0",
    port: int = 9100,
    *,
    model_dir: str | Path = "models",
    capture_dir: str | Path = "data/captures",
    config: AegisConfig | None = None,
    token: str = "",
) -> None:
    hub = IngestHub(model_dir=model_dir, capture_dir=capture_dir, config=config)
    server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, hub, token), host, port
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    log.info("aegis ingest listening on %s (model=%s capture=%s)", addrs, model_dir, capture_dir)
    async with server:
        await server.serve_forever()


def serve_ingest(
    host: str = "0.0.0.0",
    port: int = 9100,
    *,
    model_dir: str | Path = "models",
    capture_dir: str | Path = "data/captures",
    token: str = "",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run_ingest_server(
        host, port, model_dir=model_dir, capture_dir=capture_dir, token=token
    ))


def push_frames(
    host: str,
    port: int,
    frames: list[dict],
    *,
    compress: bool = False,
    timeout: float = 30.0,
    token: str = "",
) -> None:
    """Lab helper: push a synthetic / recorded frame list to a running ingest."""
    import socket

    if token:
        frames = [{"k": "auth", "token": token}, *frames]
    payload = b"".join(encode_frame(fr, compress=compress) for fr in frames)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(payload)


def result_to_json(result: IngestResult) -> str:
    return json.dumps({
        "session_id": result.session_id,
        "capture_path": result.capture_path,
        "gaps": result.gaps,
        "n_frames": result.n_frames,
        "players": result.players,
        "meta": result.meta,
    }, indent=2, default=str)


__all__ = [
    "IngestHub",
    "IngestResult",
    "StreamState",
    "run_ingest_server",
    "serve_ingest",
    "push_frames",
    "result_to_json",
]
