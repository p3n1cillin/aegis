"""Evaluation: ROC, PR, operating-point FP budget, per-family recall.

The number that matters for an anti-cheat is not AUC. It is:

  * false-positive rate at the policy ban threshold
  * detection rate on *legit* (humanised) cheats at that same threshold
  * whether the conformal gate actually keeps FP under ``alpha``

Everything else is diagnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ..core.config import AegisConfig
from ..fusion.policy import Adjudicator, VerdictBand
from ..fusion.prior import TrustPrior
from ..fusion.sprt import EvidenceState, EvidenceUnit, SPRT
from ..models.calibrate import expected_calibration_error, prob_to_llr
from ..models.ensemble import CalibratedEnsemble
from .dataset import Dataset


@dataclass
class OperatingPoint:
    name: str
    threshold: float
    fpr: float
    tpr: float
    precision: float
    n_fp: int
    n_tp: int
    n_clean: int
    n_cheat: int


@dataclass
class EvaluationResult:
    auc: float
    ap: float
    ece: float
    n_clean: int
    n_cheat: int
    operating_points: list[OperatingPoint] = field(default_factory=list)
    per_family_recall: dict[str, float] = field(default_factory=dict)
    conformal_fpr: float = 0.0
    policy_shadow_bans: int = 0
    policy_reviews: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def evaluate_scores(
    y: np.ndarray,
    p: np.ndarray,
    families: list[str] | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> EvaluationResult:
    y = np.asarray(y, dtype=np.int32).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    thresholds = thresholds or {
        "watch": 0.60, "review": 0.90, "ban": 0.995, "ban_loose": 0.95,
    }
    n_clean = int((y == 0).sum())
    n_cheat = int((y == 1).sum())
    ops = []
    for name, thr in thresholds.items():
        pred = (p >= thr).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        ops.append(OperatingPoint(
            name=name, threshold=thr,
            fpr=fp / max(n_clean, 1), tpr=tp / max(n_cheat, 1),
            precision=tp / max(tp + fp, 1),
            n_fp=fp, n_tp=tp, n_clean=n_clean, n_cheat=n_cheat,
        ))

    per_fam: dict[str, float] = {}
    if families is not None:
        for fam in sorted(set(families)):
            if fam == "clean":
                continue
            m = np.array([f == fam for f in families])
            if m.sum() == 0:
                continue
            # Recall at the "review" threshold -- the practical catch rate.
            per_fam[fam] = float((p[m] >= thresholds.get("review", 0.90)).mean())

    return EvaluationResult(
        auc=_safe_auc(y, p),
        ap=_safe_ap(y, p),
        ece=expected_calibration_error(p, y),
        n_clean=n_clean,
        n_cheat=n_cheat,
        operating_points=ops,
        per_family_recall=per_fam,
    )


def evaluate_ensemble(
    ensemble: CalibratedEnsemble,
    ds: Dataset,
    cfg: AegisConfig | None = None,
) -> EvaluationResult:
    """Score every player-session in ``ds`` and summarise."""
    cfg = cfg or AegisConfig()
    # Group engagement tensors by player_id for this dataset.
    eng_by_pid: dict[str, list[int]] = {}
    for i, pid in enumerate(ds.eng_player_ids):
        eng_by_pid.setdefault(pid, []).append(i)

    ps = []
    for i in range(len(ds.y)):
        pid = ds.player_ids[i]
        idxs = eng_by_pid.get(pid, [])
        if idxs:
            seqs = ds.eng_X[idxs]
            scals = ds.eng_scalars[idxs]
        else:
            seqs, scals = None, None
        score = ensemble.score_player(
            session_id=ds.session_ids[i],
            slot=0,
            player_id=pid,
            features=ds.X[i],
            engagement_seqs=seqs,
            engagement_scalars=scals,
        )
        ps.append(score.p_fused)
    p = np.asarray(ps, dtype=np.float64)
    ev = evaluate_scores(ds.y, p, ds.family)

    # Conformal empirical FPR on clean.
    clean_p = p[ds.y == 0]
    if clean_p.size:
        admits = ensemble.bundle.conformal.admits_ban(clean_p)
        ev.conformal_fpr = float(np.mean(admits))

    # Policy dry-run (single-session SPRT -- understates real multi-session power).
    adj = Adjudicator(policy=cfg.policy, sprt=SPRT(cfg.fusion))
    bans = reviews = 0
    for i, pid in enumerate(ds.player_ids):
        st = EvidenceState(player_id=pid)
        llrs = list(prob_to_llr(
            np.array([p[i]]), prior=ensemble.bundle.prior
        ))
        # Fabricate a Score-like path via observe_session.
        from ..models.ensemble import Score
        sc = Score(
            player_id=pid, session_id=ds.session_ids[i], slot=0,
            p_tabular=float(p[i]), p_temporal=0.5, p_novelty=0.0,
            p_fused=float(p[i]), llr_session=float(llrs[0]),
            conformal_p=float(ensemble.bundle.conformal.pvalue(np.array([p[i]]))[0]),
            meta={"n_engagements": 20},
        )
        adj.sprt.observe_session(
            st, session_id=ds.session_ids[i],
            session_llr=sc.llr_session, engagement_llrs=[], t=float(i),
        )
        verd = adj.decide(st, sc, TrustPrior(p=ensemble.bundle.prior, reason="eval"))
        if verd.band in (VerdictBand.BAN, VerdictBand.INTEGRITY_BAN):
            bans += 1
        elif verd.band == VerdictBand.REVIEW:
            reviews += 1
    ev.policy_shadow_bans = bans
    ev.policy_reviews = reviews
    ev.notes.append(
        "Single-session SPRT understates detection power; production accumulates "
        "across matches. Synthetic FP rates are optimistic vs real telemetry."
    )
    return ev


def detection_vs_stealth(
    ensemble: CalibratedEnsemble,
    *,
    n_per_level: int = 8,
    duration_s: float = 60.0,
    seed: int = 99,
) -> list[dict]:
    """Sweep aimbot humaniser strength and report catch rate at review threshold.

    Returns a list of {smooth_ticks, fov, strength, recall_at_review}.
    """
    from .simulate import CheatConfig, simulate_session
    from ..features.pipeline import extract_session_features
    from .dataset import SIM_CAPS

    rng = np.random.default_rng(seed)
    rows = []
    for smooth in (1, 4, 8, 14, 22):
        for fov in (8.0, 18.0, 40.0):
            ys, ps = [], []
            for k in range(n_per_level):
                cheat = CheatConfig(
                    aimbot=True, aimbot_fov=fov, aimbot_smooth_ticks=smooth,
                    aimbot_strength=0.7, aimbot_retain_tremor=True,
                    aimbot_jitter_deg=0.3, aimbot_delay_s=0.12,
                )
                # 1 cheater + 3 clean
                families = ["aimbot_legit", "clean", "clean", "clean"]
                # Override by running simulate then... easier: use family label
                sess, labels = simulate_session(
                    session_id=f"stealth_{smooth}_{fov}_{k}",
                    cheat_families=families,
                    duration_s=duration_s,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                # Manually retarget slot 0's cheat config for the sweep knobs.
                sess.meta["cheat_configs"][0] = cheat
                feats = extract_session_features(sess, set(SIM_CAPS), spec=ensemble.spec)
                for f in feats:
                    lab = labels.get(f.slot, "clean")
                    if lab == "clean" and f.slot != 0:
                        continue
                    if f.slot != 0:
                        continue
                    sc = ensemble.score_player(
                        session_id=f.session_id, slot=f.slot, player_id=f.player_id,
                        features=f.values,
                    )
                    ys.append(1)
                    ps.append(sc.p_fused)
            if ps:
                rows.append({
                    "smooth_ticks": smooth,
                    "fov": fov,
                    "n": len(ps),
                    "mean_p": float(np.mean(ps)),
                    "recall_at_review": float(np.mean(np.array(ps) >= 0.90)),
                })
    return rows
