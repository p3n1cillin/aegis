"""Serve-time HTTP API.

Thin FastAPI surface over the ensemble + fusion + integrity stack. Designed
for a game-server sidecar or a central adjudication service to call after
each match (or continuously for live plugin feeds).

Shadow mode is the default: endpoints compute verdicts but ``issued`` is
false until an operator flips ``policy.shadow_mode`` in config.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.config import AegisConfig
from ..core.registry import ADAPTERS, Capability
from ..features.pipeline import extract_engagement_tensors, extract_session_features
from ..fusion.explain import explain_score
from ..fusion.policy import Adjudicator
from ..fusion.prior import trust_from_player_info
from ..fusion.sprt import EvidenceState, SPRT
from ..integrity.attest import Attestor
from ..integrity.hid import HIDAnalyzer
from ..models.calibrate import prob_to_llr
from ..models.ensemble import CalibratedEnsemble

# Import adapters for registration side effects.
import aegis.adapters  # noqa: F401


class AnalyzeRequest(BaseModel):
    adapter: str = "cs2"
    source: Any = Field(
        description="Adapter-specific payload: path string, frame list, or generic_fps JSON"
    )
    player_states: dict[str, dict] = Field(
        default_factory=dict,
        description="Optional map of player_id -> {log_odds, n_units, last_t} for SPRT continuity",
    )


class AttestIssueRequest(BaseModel):
    session_id: str
    player_id: str = ""


class AttestVerifyRequest(BaseModel):
    session_id: str
    challenge_id: str
    measurement_hex: str
    mac_hex: str
    probe_ok: dict[str, bool] = Field(default_factory=dict)


def create_app(
    model_dir: str | Path = "models",
    config: AegisConfig | None = None,
) -> FastAPI:
    cfg = config or AegisConfig(model_dir=str(model_dir))
    app = FastAPI(title="Aegis Anti-Cheat", version="0.1.0")
    ensemble: CalibratedEnsemble | None = None
    states: dict[str, EvidenceState] = {}
    attestor = Attestor(cfg.integrity)
    hid = HIDAnalyzer()
    adjudicator = Adjudicator(policy=cfg.policy, sprt=SPRT(cfg.fusion))

    @app.on_event("startup")
    def _load() -> None:
        nonlocal ensemble
        path = Path(cfg.model_dir)
        if (path / "ensemble.json").exists():
            ensemble = CalibratedEnsemble.load(path, device=cfg.model.device)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "model_loaded": ensemble is not None,
            "shadow_mode": cfg.policy.shadow_mode,
            "spec_digest": ensemble.digest if ensemble else None,
        }

    @app.post("/v1/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        if ensemble is None:
            raise HTTPException(503, "model not loaded; run `aegis train` first")
        try:
            adapter_cls = ADAPTERS.get_or_raise(req.adapter)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
        adapter = adapter_cls() if isinstance(adapter_cls, type) else adapter_cls
        try:
            session = adapter.parse(req.source)
        except Exception as e:
            raise HTTPException(400, f"adapter parse failed: {e}") from e

        caps = set(adapter.capabilities)
        if session.meta.get("no_vis") or not session.visibility:
            caps.discard(Capability.VISIBILITY)
        if not session.meta.get("attested", False):
            caps.discard(Capability.HID_INPUT)

        feats = extract_session_features(session, caps, cfg=cfg.features, spec=ensemble.spec)
        tensors = extract_engagement_tensors(
            session, caps, cfg=cfg.features, seq_len=cfg.model.seq_len
        )
        # Index tensors by slot.
        by_slot: dict[int, list] = {}
        for t in tensors:
            by_slot.setdefault(t.slot, []).append(t)

        results = []
        for f in feats:
            info = session.players[f.slot]
            prior = trust_from_player_info(info)
            seqs = scals = None
            if f.slot in by_slot:
                seqs = np.stack([t.data for t in by_slot[f.slot]])
                scals = np.stack([t.scalars for t in by_slot[f.slot]])
            score = ensemble.score_player(
                session_id=session.session_id, slot=f.slot, player_id=info.player_id,
                features=f.values, engagement_seqs=seqs, engagement_scalars=scals,
                prior=prior.p,
            )
            st = states.get(info.player_id) or EvidenceState(player_id=info.player_id)
            # Restore continuity if the caller supplied it.
            saved = req.player_states.get(info.player_id)
            if saved and st.n_units == 0:
                st.log_odds = float(saved.get("log_odds", 0.0))
                st.n_units = float(saved.get("n_units", 0.0))
                st.last_t = float(saved.get("last_t", 0.0))

            nov_llr = float(prob_to_llr(np.array([score.p_novelty]), prior=prior.p)[0]) * 0.4
            adjudicator.sprt.observe_session(
                st,
                session_id=session.session_id,
                session_llr=score.llr_session,
                engagement_llrs=score.engagement_llrs,
                novelty_llr=nov_llr,
                t=time.time(),
            )
            states[info.player_id] = st
            integ = attestor.state(session.session_id, info.player_id)
            verd = adjudicator.decide(
                st, score, prior,
                integrity_confirmed_tamper=integ.confirmed_tamper,
                n_engagements_lifetime=int(st.n_units) + score.meta.get("n_engagements", 0),
            )
            expl = explain_score(score, features=f, tabular=ensemble.bundle.tabular, spec=ensemble.spec)
            hid_report = None
            if session.meta.get("attested") and info.player_id and session.inputs.get(f.slot):
                hid_report = hid.analyze(session.inputs[f.slot]).__dict__

            results.append({
                "score": score.to_dict(),
                "verdict": verd.to_dict(),
                "explanation": expl.to_dict(),
                "evidence": {
                    "log_odds": st.log_odds,
                    "n_units": st.n_units,
                    "last_t": st.last_t,
                    "posterior": st.posterior(prior.p),
                },
                "hid": hid_report,
            })
        return {
            "session_id": session.session_id,
            "game": session.game,
            "shadow_mode": cfg.policy.shadow_mode,
            "players": results,
        }

    @app.post("/v1/attest/issue")
    def attest_issue(req: AttestIssueRequest) -> dict:
        ch = attestor.issue(req.session_id, req.player_id)
        return {
            "challenge_id": ch.challenge_id,
            "nonce_hex": ch.nonce.hex(),
            "deadline": ch.deadline,
            "probes": list(ch.probes),
        }

    @app.post("/v1/attest/verify")
    def attest_verify(req: AttestVerifyRequest) -> dict:
        res = attestor.verify(
            req.session_id,
            challenge_id=req.challenge_id,
            client_measurement=bytes.fromhex(req.measurement_hex),
            client_mac=bytes.fromhex(req.mac_hex),
            probe_ok=req.probe_ok,
        )
        st = attestor.state(req.session_id)
        return {
            "ok": res.ok,
            "reason": res.reason,
            "latency_ms": res.latency_ms,
            "attested": st.attested,
            "untrusted": st.untrusted,
            "confirmed_tamper": st.confirmed_tamper,
        }

    return app
