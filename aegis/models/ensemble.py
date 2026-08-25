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
import os
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


def coverage_shrink(coverage: float, floor: float = 0.5) -> float:
    """Extra shrink multiplier from how much of the feature vector is real.

    A no-op at and above ``floor`` on purpose -- a fully-capable synthetic
    session still routinely sits around 70% coverage (hitgroups/visibility
    columns that stay NaN unless that specific event happened), and that
    ordinary variance must not get dampened. Below the floor, scales down
    linearly to 0. See ``CalibratedEnsemble.score_player`` for why this
    exists: a session with ~11% coverage produced a tabular score that was
    provably insensitive to its own populated values (see COMPARISON.md).
    """
    return min(1.0, coverage / floor)


# Sparse-session policy constants (closes HANDOFF entry 34's open item).
# These are *policy* constants, not learned parameters, so they live here in
# code rather than in ensemble.json -- changing them is a reviewed decision,
# not something a retrain can silently move underneath an operator.
SPARSE_ZERO_ENGAGEMENTS = 0  # sessions with no engagements at all
SPARSE_COVERAGE_CUTOFF = 0.25  # below this even engagements can't save the vector
SPARSE_ZERO_ENG_SHRINK = 0.10  # hard shrink toward prior for zero-engagement


def sparse_session_policy(n_engagements: int, coverage: float) -> dict:
    """Single decision point for how a low-information session is scored.

    Entry 34 diagnosed that the tabular GBDT emits a near-constant score on
    rare missingness patterns (e.g. ~11% coverage spectral-only profiles):
    independently rescaling every populated feature value by 0.2x-3x left the
    raw output unchanged to 15 decimal places. ``min_samples_leaf=40`` makes
    this structural, not a sampling problem -- and ``coverage_shrink`` alone
    (entry 35) mitigated the *impact* without addressing the cause. This
    policy implements entry 34's option (b): a dedicated path that stops
    asking the tabular channel to vote when its input pattern is one it was
    never able to learn from.

    Rules:
      * Zero engagements -> the tabular channel does not participate at all
        and the fused posterior shrinks hard (0.10) toward the prior. A
        session with no duel evidence has no honest behavioural signal;
        emitting anything confident from it is manufacturing evidence.
      * Engagements present but coverage under ``SPARSE_COVERAGE_CUTOFF`` ->
        tabular excluded again (its output is untrustworthy on these NaN
        patterns by direct measurement); engagement/temporal evidence keeps
        its normal weight and ``coverage_shrink`` still scales the result.
      * Otherwise -> unchanged behaviour, including the documented no-op at
        >=50% coverage so genuine detections are never dented (the regression
        made once before, see entry 35).

    Returns ``{"use_tabular": bool, "shrink": float}``.
    """
    cov = max(0.0, min(1.0, coverage))
    if n_engagements <= SPARSE_ZERO_ENGAGEMENTS:
        return {"use_tabular": False, "shrink": SPARSE_ZERO_ENG_SHRINK}
    if cov < SPARSE_COVERAGE_CUTOFF:
        return {"use_tabular": False, "shrink": coverage_shrink(cov)}
    return {"use_tabular": True, "shrink": coverage_shrink(cov)}


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
    # cal_temporal is fit against the *engagement*-level cheat rate (cheaters
    # tend to contribute more engagements per session than clean players, so
    # this is not the same number as `prior`, which is session-level). Using
    # `prior` to convert engagement probabilities to LLRs when the calibrator
    # was fit against a different population than `prior` describes produces
    # a systematic per-engagement bias -- confirmed live: mean calibrated P
    # for known-clean engagements sat at ~0.30 against prior=0.20, which
    # SPRT's cross-session accumulation (no discount for a *repeating*
    # per-session bias, only for *within*-session correlation) turns into an
    # eventual false-ban for every clean player, not just noisy ones. See
    # COMPARISON.md.
    eng_prior: float = 0.05


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
            eng_llrs = [float(v) for v in prob_to_llr(cal_e, prior=b.eng_prior)]
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
        # Sparse-session policy: decide whether the tabular channel is allowed
        # to vote at all, and how hard to shrink toward the prior (entry 34).
        n_eng = len(eng_ps)
        coverage = float(np.mean(np.isfinite(x[0])))
        policy = sparse_session_policy(n_eng, coverage)
        if not policy["use_tabular"]:
            # Re-fuse without the tabular channel rather than post-hoc
            # overwriting p_fused: the remaining modalities renormalize among
            # themselves, then the shrink pulls the whole thing to the prior.
            fused_logit = (
                b.w_temp * logit(p_temp)
                + b.w_nov * logit(p_nov)
            ) / max(b.w_temp + b.w_nov, 1e-6)
            p_fused = float(1.0 / (1.0 + np.exp(-fused_logit)))
        shrink = policy["shrink"]
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
            meta={
                "prior": prior,
                "n_engagements": n_eng,
                "raw_tabular": raw_tab,
                "coverage": coverage,
                "tabular_used": policy["use_tabular"],
                "shrink": shrink,
            },
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
                "eng_prior": b.eng_prior,
                "has_temporal": b.temporal is not None,
            }, indent=2),
            encoding="utf-8",
        )
        # Fail-closed signing: when an operator holds a model key, every save
        # is signed so downstream loads can verify (see models/signing.py).
        key = os.environ.get("AEGIS_MODEL_KEY", "")
        if key:
            from .signing import sign_model_dir

            sign_model_dir(path, key.encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CalibratedEnsemble":
        path = Path(path)
        # Verify BEFORE touching any artefact -- a tampered tabular.joblib is
        # the cleanest attack on this platform (see models/signing.py).
        key = os.environ.get("AEGIS_MODEL_KEY", "")
        if key:
            from .signing import verify_model_dir

            verify_model_dir(path, key.encode("utf-8"))
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
            # Old bundles predate this field; falling back to `prior` keeps
            # their (known-miscalibrated) behaviour rather than silently
            # changing it for an artefact that hasn't been retrained.
            eng_prior=float(meta.get("eng_prior", meta.get("prior", 0.05))),
        )
        return cls(bundle=bundle, spec=spec)
