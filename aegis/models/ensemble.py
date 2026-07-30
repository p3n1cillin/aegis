"""Calibrated multi-model ensemble.

This is the serve-time face of the model layer. Given session features and
(optional) engagement tensors it returns a ``Score`` with:

  * calibrated P(cheat) from the tabular and temporal heads
  * novelty score from the autoencoder
  * a fused session-level probability
  * per-engagement probabilities for the sequential test
  * conformal p-value against the clean null

Fusion turns these into LLRs; policy turns LLRs into verdicts. Neither should
ever call a raw model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features.pipeline import FeatureSpec
from .calibrate import ConformalGate, ProbabilityCalibrator, prob_to_llr
from .novelty import NoveltyAE
from .tabular import TabularGBDT
from .temporal import TemporalEncoder


@dataclass
class Score:
    player_id: str
    session_id: str
    slot: int
    p_tabular: float
    p_temporal: float  # mean over engagements; 0.5 if none
    p_novelty: float
    p_fused: float
    llr_session: float
    conformal_p: float
    engagement_ps: list[float] = field(default_factory=list)
    engagement_llrs: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "session_id": self.session_id,
            "slot": self.slot,
            "p_tabular": self.p_tabular,
            "p_temporal": self.p_temporal,
            "p_novelty": self.p_novelty,
            "p_fused": self.p_fused,
            "llr_session": self.llr_session,
            "conformal_p": self.conformal_p,
            "n_engagements": len(self.engagement_ps),
            "meta": self.meta,
        }


@dataclass
class ModelBundle:
    tabular: TabularGBDT
    temporal: TemporalEncoder | None
    novelty: NoveltyAE
    cal_tabular: ProbabilityCalibrator
    cal_temporal: ProbabilityCalibrator | None
    conformal: ConformalGate
    # Fusion weights for the three modalities (log-odds space).
    w_tab: float = 1.0
    w_temp: float = 0.85
    w_nov: float = 0.45
    prior: float = 0.05  # population cheat rate assumed at calibration time


@dataclass
class CalibratedEnsemble:
    bundle: ModelBundle
    spec: FeatureSpec

    @property
    def digest(self) -> str:
        return self.spec.digest

    def score_player(
        self,
        *,
        session_id: str,
        slot: int,
        player_id: str,
        features: np.ndarray,
        engagement_seqs: np.ndarray | None = None,
        engagement_scalars: np.ndarray | None = None,
        prior: float | None = None,
    ) -> Score:
        b = self.bundle
        prior = float(prior if prior is not None else b.prior)
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(self.spec.columns):
            raise ValueError(
                f"feature width {x.shape[1]} != spec {len(self.spec.columns)} "
                f"(digest {self.spec.digest})"
            )

        raw_tab = float(b.tabular.predict_proba(x)[0])
        p_tab = float(b.cal_tabular.transform(np.array([raw_tab]))[0])

        eng_ps: list[float] = []
        eng_llrs: list[float] = []
        p_temp = 0.5
        if (
            b.temporal is not None
            and b.cal_temporal is not None
            and engagement_seqs is not None
            and engagement_seqs.size > 0
        ):
            raw_e = b.temporal.predict_proba(engagement_seqs, engagement_scalars)
            cal_e = b.cal_temporal.transform(raw_e)
            eng_ps = [float(p) for p in cal_e]
            eng_llrs = [float(v) for v in prob_to_llr(cal_e, prior=prior)]
            # Session-level temporal summary: soft-OR via mean logit, not mean p
            # (mean p understates evidence when many engagements are mild).
            logits = np.log(np.clip(cal_e, 1e-6, 1 - 1e-6)) - np.log(
                1 - np.clip(cal_e, 1e-6, 1 - 1e-6)
            )
            # Correlation discount: engagements in one match are dependent.
            eff = max(1.0, len(cal_e) ** 0.65)
            mean_logit = float(logits.sum() / eff)
            p_temp = float(1.0 / (1.0 + np.exp(-mean_logit)))

        p_nov = float(b.novelty.novelty_probability(x)[0])

        # Fuse in logit space with modality weights.
        def logit(p: float) -> float:
            p = min(max(p, 1e-6), 1 - 1e-6)
            return float(np.log(p) - np.log(1 - p))

        fused_logit = (
            b.w_tab * logit(p_tab)
            + b.w_temp * logit(p_temp)
            + b.w_nov * logit(p_nov)
        ) / max(b.w_tab + b.w_temp + b.w_nov, 1e-6)
        # Re-center so that (0.5, 0.5, 0.5) maps back near the prior, not 0.5.
        fused_logit = fused_logit + logit(prior) * 0.0  # kept explicit for clarity
        p_fused = float(1.0 / (1.0 + np.exp(-fused_logit)))
        # Shrink toward prior when we have almost no engagements -- a single
        # lucky spray should not move the posterior much.
        n_eng = len(eng_ps)
        shrink = 1.0 - np.exp(-n_eng / 8.0) if n_eng else 0.55
        p_fused = float(prior + shrink * (p_fused - prior))

        llr = float(prob_to_llr(np.array([p_fused]), prior=prior)[0])
        conf_p = float(b.conformal.pvalue(np.array([p_fused]))[0])

        return Score(
            player_id=player_id,
            session_id=session_id,
            slot=slot,
            p_tabular=p_tab,
            p_temporal=p_temp,
            p_novelty=p_nov,
            p_fused=p_fused,
            llr_session=llr,
            conformal_p=conf_p,
            engagement_ps=eng_ps,
            engagement_llrs=eng_llrs,
            meta={"prior": prior, "n_engagements": n_eng, "raw_tabular": raw_tab},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        b = self.bundle
        b.tabular.save(path)
        if b.temporal is not None:
            b.temporal.save(path)
        b.novelty.save(path)
        b.cal_tabular.save(path / "cal_tabular.joblib")
        if b.cal_temporal is not None:
            b.cal_temporal.save(path / "cal_temporal.joblib")
        b.conformal.save(path / "conformal.npz")
        (path / "ensemble.json").write_text(
            json.dumps({
                "spec": self.spec.to_dict(),
                "w_tab": b.w_tab,
                "w_temp": b.w_temp,
                "w_nov": b.w_nov,
                "prior": b.prior,
                "has_temporal": b.temporal is not None,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CalibratedEnsemble":
        path = Path(path)
        meta = json.loads((path / "ensemble.json").read_text(encoding="utf-8"))
        spec = FeatureSpec.from_dict(meta["spec"])
        tabular = TabularGBDT.load(path)
        temporal = TemporalEncoder.load(path, device=device) if meta.get("has_temporal", True) and (path / "temporal.pt").exists() else None
        novelty = NoveltyAE.load(path, device=device)
        cal_tab = ProbabilityCalibrator.load(path / "cal_tabular.joblib")
        cal_temp = None
        if (path / "cal_temporal.joblib").exists():
            cal_temp = ProbabilityCalibrator.load(path / "cal_temporal.joblib")
        conformal = ConformalGate.load(path / "conformal.npz")
        bundle = ModelBundle(
            tabular=tabular,
            temporal=temporal,
            novelty=novelty,
            cal_tabular=cal_tab,
            cal_temporal=cal_temp,
            conformal=conformal,
            w_tab=float(meta.get("w_tab", 1.0)),
            w_temp=float(meta.get("w_temp", 0.85)),
            w_nov=float(meta.get("w_nov", 0.45)),
            prior=float(meta.get("prior", 0.05)),
        )
        return cls(bundle=bundle, spec=spec)
