"""Evaluation: ROC, PR, operating-point FP budget, per-family recall.

The number that matters for an anti-cheat is not AUC. It is:

  * false-positive rate at the policy ban threshold
  * detection rate on *legit* (humanised) cheats at that same threshold
  * whether the conformal gate actually keeps FP under ``alpha``

Everything else is diagnostic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

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


def multi_session_accumulation(
    ensemble: CalibratedEnsemble,
    *,
    cfg: AegisConfig | None = None,
    cheat_family: str = "wallhack",
    n_cheat_identities: int = 15,
    n_clean_identities: int = 15,
    n_sessions: int = 20,
    duration_s: float = 60.0,
    seed: int = 7,
) -> dict:
    """Evidence accumulation across *matches*, not one score per match.

    ``evaluate_ensemble``'s single-session numbers understate the system by
    design -- SPRT exists because one match should not decide a verdict. This
    replays each of several persistent identities across ``n_sessions``
    separate matches (one simulated day apart, well inside the 45-day evidence
    half-life) through the same ``EvidenceState``/SPRT path the live service
    uses, and reports how many matches it actually takes to flag/ban a
    persistent cheater, and — the number that actually matters for a
    real deployment — the false-flag rate among clean identities over the
    same number of matches. ``cheat_family`` defaults to "wallhack" because
    that is Aegis's own named differentiator against VACnet's blind spot
    (README's failure-mode table); a single low single-session recall number
    for it is not evidence the claim fails, since the claim was never "one
    match is enough."
    """
    from ..features.pipeline import extract_engagement_tensors, extract_session_features
    from .dataset import SIM_CAPS
    from .simulate import simulate_session

    cfg = cfg or AegisConfig()
    caps = set(SIM_CAPS)
    prior = TrustPrior(p=ensemble.bundle.prior, reason="multi_session_eval")
    sprt = SPRT(cfg.fusion)
    # Real decide() gates on posterior>=ban_threshold AND crossed_ban AND
    # lifetime engagements>=min_engagements_for_ban AND conformal admission --
    # strictly more conservative than sprt.crossed_ban() alone. shadow_mode is
    # forced off here so `issued` reflects what decide() actually decided, not
    # the deployment default; this function measures the policy, it does not
    # enact it.
    adjudicator = Adjudicator(policy=replace(cfg.policy, shadow_mode=False), sprt=sprt)
    rng = np.random.default_rng(seed)

    def run_identity(tag: str, family: str) -> dict:
        st = EvidenceState(player_id=tag)
        sessions_to_flag: int | None = None
        sessions_to_ban: int | None = None
        sessions_to_policy_review: int | None = None
        sessions_to_policy_ban: int | None = None
        conformal_ever_blocked = False
        lifetime_engagements = 0
        t0 = time.time()
        for m in range(n_sessions):
            sess, _ = simulate_session(
                session_id=f"{tag}_m{m}", cheat_families=[family, "clean", "clean", "clean"],
                duration_s=duration_s, seed=int(rng.integers(0, 2**31 - 1)),
            )
            feats = extract_session_features(sess, caps, spec=ensemble.spec)
            f0 = next(f for f in feats if f.slot == 0)
            tensors = extract_engagement_tensors(sess, caps, cfg=cfg.features, seq_len=cfg.model.seq_len)
            eng0 = [tt for tt in tensors if tt.slot == 0]
            seqs = np.stack([tt.data for tt in eng0]) if eng0 else None
            scals = np.stack([tt.scalars for tt in eng0]) if eng0 else None
            score = ensemble.score_player(
                session_id=sess.session_id, slot=0, player_id=tag,
                features=f0.values, engagement_seqs=seqs, engagement_scalars=scals,
            )
            lifetime_engagements += len(eng0)
            nov_llr = float(prob_to_llr(np.array([score.p_novelty]), prior=prior.p)[0]) * 0.4
            t = t0 + m * 86400.0  # one match per simulated day
            sprt.observe_session(
                st, session_id=sess.session_id, session_llr=score.llr_session,
                engagement_llrs=score.engagement_llrs, novelty_llr=nov_llr, t=t,
            )
            if sessions_to_flag is None and sprt.crossed_flag(st):
                sessions_to_flag = m + 1
            if sessions_to_ban is None and sprt.crossed_ban(st):
                sessions_to_ban = m + 1
                if score.conformal_p >= sprt.cfg.alpha:
                    conformal_ever_blocked = True

            verd = adjudicator.decide(
                st, score, prior, n_engagements_lifetime=lifetime_engagements,
            )
            if sessions_to_policy_review is None and verd.band.value == "review":
                sessions_to_policy_review = m + 1
            if sessions_to_policy_ban is None and verd.band.value == "ban":
                sessions_to_policy_ban = m + 1
        return {
            "sessions_to_flag": sessions_to_flag,
            "sessions_to_ban": sessions_to_ban,
            "sessions_to_policy_review": sessions_to_policy_review,
            "sessions_to_policy_ban": sessions_to_policy_ban,
            "conformal_ever_blocked_a_raw_ban": conformal_ever_blocked,
            "final_posterior": st.posterior(prior.p),
            "final_n_units": st.n_units,
            "lifetime_engagements": lifetime_engagements,
        }

    cheat_results = [run_identity(f"cheat_{i}", cheat_family) for i in range(n_cheat_identities)]
    clean_results = [run_identity(f"clean_{i}", "clean") for i in range(n_clean_identities)]

    def rate(results: list[dict], key: str) -> float:
        return sum(1 for r in results if r[key] is not None) / max(len(results), 1)

    def median_of(results: list[dict], key: str) -> float | None:
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.median(vals)) if vals else None

    return {
        "cheat_family": cheat_family,
        "n_sessions": n_sessions,
        "n_cheat_identities": n_cheat_identities,
        "n_clean_identities": n_clean_identities,
        # Raw SPRT boundary crossing -- ignores conformal + min_engagements_for_ban.
        "cheat_flag_rate": rate(cheat_results, "sessions_to_flag"),
        "cheat_ban_rate_raw_sprt": rate(cheat_results, "sessions_to_ban"),
        "clean_false_flag_rate": rate(clean_results, "sessions_to_flag"),
        "clean_false_ban_rate_raw_sprt": rate(clean_results, "sessions_to_ban"),
        # Full production Adjudicator.decide() path -- what would actually happen.
        "cheat_policy_review_rate": rate(cheat_results, "sessions_to_policy_review"),
        "cheat_policy_ban_rate": rate(cheat_results, "sessions_to_policy_ban"),
        "clean_false_policy_review_rate": rate(clean_results, "sessions_to_policy_review"),
        "clean_false_policy_ban_rate": rate(clean_results, "sessions_to_policy_ban"),
        "conformal_ever_blocked_a_raw_ban_cheat": sum(r["conformal_ever_blocked_a_raw_ban"] for r in cheat_results),
        "conformal_ever_blocked_a_raw_ban_clean": sum(r["conformal_ever_blocked_a_raw_ban"] for r in clean_results),
        "median_sessions_to_flag": median_of(cheat_results, "sessions_to_flag"),
        "median_sessions_to_ban_raw_sprt": median_of(cheat_results, "sessions_to_ban"),
        "median_sessions_to_policy_ban": median_of(cheat_results, "sessions_to_policy_ban"),
        "cheat_final_posteriors": [r["final_posterior"] for r in cheat_results],
        "clean_final_posteriors": [r["final_posterior"] for r in clean_results],
    }
