"""Serve-time HTTP API.

Thin FastAPI surface over the ensemble + fusion + integrity stack. Designed
for a game-server sidecar or a central adjudication service to call after
each match (or continuously for live plugin feeds).

Shadow mode is the default: endpoints compute verdicts but ``issued`` is
false until an operator flips ``policy.shadow_mode`` in config.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..adapters.base import effective_capabilities
from ..core.config import AegisConfig
from ..core.registry import ADAPTERS
from ..features.pipeline import extract_engagement_tensors, extract_session_features
from ..fusion.explain import explain_score
from ..fusion.policy import Adjudicator
from ..fusion.prior import trust_from_player_info
from ..fusion.sprt import EvidenceState, SPRT
from ..integrity.attest import Attestor
from ..integrity.audit import AuditLog, make_audit_record
from ..integrity.hid import HIDAnalyzer
from ..models.calibrate import prob_to_llr
from ..models.ensemble import CalibratedEnsemble

# Import adapters for registration side effects.
import aegis.adapters  # noqa: F401

log = logging.getLogger("aegis.service")


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


class ReviewRequest(BaseModel):
    """One human review verdict -- HANDOFF backlog item 5's queue input."""

    player_id: str
    session_id: str
    y: int = Field(ge=0, le=1, description="0 = clean, 1 = cheat")
    family: str = ""
    reviewer: str = ""
    notes: str = ""
    p_fused: float | None = Field(default=None, ge=0.0, le=1.0)


def _resolve_source_path(raw: str, allowed_dir: str) -> Path:
    """Only honour a filesystem-path ``source`` if it resolves inside allowed_dir.

    Without this, any caller who can reach ``/v1/analyze`` can pass an
    arbitrary path and have the server read it back through the adapter
    parse error / capture output -- an unauthenticated file-read primitive.
    """
    base = Path(allowed_dir).resolve()
    candidate = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if base != candidate and base not in candidate.parents:
        raise HTTPException(400, "source path must resolve inside the configured capture directory")
    return candidate


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
    # Every verdict (shadow included) lands here -- see integrity/audit.py.
    audit = AuditLog(cfg.policy.audit_log_path) if cfg.policy.audit_log_path else None

    if not cfg.api_key:
        log.warning(
            "AEGIS_API_KEY not set -- /v1/* routes are unauthenticated. Fine for "
            "127.0.0.1-only dev use, not for anything reachable off-box."
        )

    def require_api_key(x_aegis_api_key: str = Header(default="")) -> None:
        if cfg.api_key and x_aegis_api_key != cfg.api_key:
            raise HTTPException(401, "missing or invalid X-Aegis-Api-Key")

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

    @app.post("/v1/analyze", dependencies=[Depends(require_api_key)])
    def analyze(req: AnalyzeRequest) -> dict:
        if ensemble is None:
            raise HTTPException(503, "model not loaded; run `aegis train` first")
        try:
            adapter_cls = ADAPTERS.get_or_raise(req.adapter)
        except KeyError as e:
            raise HTTPException(400, str(e)) from e
        adapter = adapter_cls() if isinstance(adapter_cls, type) else adapter_cls
        source = req.source
        if isinstance(source, str):
            source = _resolve_source_path(source, cfg.allowed_source_dir)
        try:
            session = adapter.parse(source)
        except Exception as e:
            raise HTTPException(400, f"adapter parse failed: {e}") from e

        caps = effective_capabilities(adapter, session)

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
                title=session.game,
            )
            expl = explain_score(score, features=f, tabular=ensemble.bundle.tabular, spec=ensemble.spec)
            if audit is not None:
                audit.append(make_audit_record(
                    session_id=session.session_id,
                    player_id=info.player_id,
                    verdict=verd,
                    score=score,
                    shadow=cfg.policy.shadow_mode,
                    model_digest=ensemble.digest,
                ))
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

    @app.post("/v1/attest/issue", dependencies=[Depends(require_api_key)])
    def attest_issue(req: AttestIssueRequest) -> dict:
        ch = attestor.issue(req.session_id, req.player_id)
        return {
            "challenge_id": ch.challenge_id,
            "nonce_hex": ch.nonce.hex(),
            "deadline": ch.deadline,
            "probes": list(ch.probes),
        }

    @app.post("/v1/attest/verify", dependencies=[Depends(require_api_key)])
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

    @app.post("/v1/reviews", dependencies=[Depends(require_api_key)])
    def submit_review(req: ReviewRequest) -> dict:
        """Append one human review label to the online store (backlog item 5).

        Append-only by design: labels are evidence, and refreshes are a
        separate explicit action so a queue worker can never silently retrain
        the serving model as a side effect of filing paperwork.
        """
        from ..train.online import OnlineStore, ReviewLabel

        store = OnlineStore.load(cfg.review_store_path)
        label = ReviewLabel(
            player_id=req.player_id,
            session_id=req.session_id,
            y=int(req.y),
            family=req.family,
            p_fused=req.p_fused,
            reviewer=req.reviewer,
            notes=req.notes,
        )
        store.append(label)
        log.info("review label appended player=%s y=%d (%d total)",
                 req.player_id, req.y, len(store.labels))
        return {"ok": True, "n_labels": len(store.labels)}

    @app.post("/v1/reviews/refresh", dependencies=[Depends(require_api_key)])
    def refresh_from_review_queue() -> dict:
        """Operator-triggered online refresh from accumulated review labels.

        Refuses below a minimum label count: refitting calibrators on a
        handful of rows produces operating points worse than the synthetic
        bootstrap they would replace. Returns the RefreshReport on success.
        """
        nonlocal ensemble
        from ..train.online import MIN_LABELS_FOR_REFRESH, OnlineStore, refresh_from_reviews

        if ensemble is None:
            raise HTTPException(503, "no ensemble loaded")
        store = OnlineStore.load(cfg.review_store_path)
        if len(store.labels) < MIN_LABELS_FOR_REFRESH:
            raise HTTPException(
                409,
                f"only {len(store.labels)} review labels; need >= "
                f"{MIN_LABELS_FOR_REFRESH} before a refresh is meaningful",
            )
        new_ensemble, report = refresh_from_reviews(
            ensemble, store, out_dir=cfg.model_dir, cfg=cfg,
        )
        # Serve the refreshed artefacts immediately; save() already ran
        # inside refresh_from_reviews (and re-signed if AEGIS_MODEL_KEY set).
        ensemble = new_ensemble
        out = report.to_dict()
        log.info("online refresh complete: %s", json.dumps(out)[:200])
        return out

    return app
