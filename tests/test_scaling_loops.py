"""Regression tests for the improvement-loop work (see implementation_plan.md).

Kept separate from tests/test_invariants.py so the invariant suite stays
stable; everything here must also stay fast (< a few seconds each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis.models.ensemble import (
    SPARSE_COVERAGE_CUTOFF,
    SPARSE_ZERO_ENG_SHRINK,
    CalibratedEnsemble,
    coverage_shrink,
    sparse_session_policy,
)
from aegis.train.dataset import SIM_CAPS
from aegis.train.simulate import simulate_session


MODEL_DIR = Path("models")


def _load_ensemble() -> CalibratedEnsemble:
    if not (MODEL_DIR / "ensemble.json").exists():
        pytest.skip("models/ not trained yet")
    return CalibratedEnsemble.load(MODEL_DIR)


# ---------------------------------------------------------------------------
# Loop 1 -- sparse-session policy (HANDOFF entry 34 closure)
# ---------------------------------------------------------------------------


def test_sparse_policy_unit_rules():
    """Zero engagements -> tabular out + hard shrink; low coverage with
    engagements -> tabular out + coverage-proportional shrink; healthy
    sessions -> exactly the old coverage_shrink behaviour."""
    # Zero-engagement branch.
    pol = sparse_session_policy(0, coverage=0.95)
    assert pol["use_tabular"] is False
    assert pol["shrink"] == SPARSE_ZERO_ENG_SHRINK

    # Sparse-but-engaged branch.
    pol = sparse_session_policy(30, coverage=SPARSE_COVERAGE_CUTOFF / 2)
    assert pol["use_tabular"] is False
    assert pol["shrink"] == pytest.approx(coverage_shrink(SPARSE_COVERAGE_CUTOFF / 2))

    # Healthy branch is a pure passthrough to coverage_shrink.
    for cov in (0.5, 0.7, 1.0):
        pol = sparse_session_policy(50, coverage=cov)
        assert pol["use_tabular"] is True
        assert pol["shrink"] == pytest.approx(coverage_shrink(cov))


def test_zero_engagement_session_never_produces_confident_posterior():
    """The core entry-34 regression: a zero-engagement session whose feature
    vector is stuffed with extreme values must not be able to reach even the
    watch band. Previously the tabular channel voted on such sessions with a
    provably insensitive near-constant score (~0.62 'watch'); now it does not
    vote at all and the posterior is pinned within shrink of the prior."""
    ens = _load_ensemble()
    prior = ens.bundle.prior
    sess, _ = simulate_session(
        session_id="t_sparse",
        cheat_families=["blatant", "clean", "clean", "clean"],
        duration_s=12.0,
        seed=11,
    )
    from aegis.features.pipeline import build_spec, extract_session_features

    feats = extract_session_features(sess, set(SIM_CAPS), spec=ens.spec)
    f0 = next(f for f in feats if f.slot == 0)

    x = np.asarray(f0.values, dtype=np.float32)
    x_extreme = np.where(np.isfinite(x), x * 5.0, x)  # scream into every column

    for label, vec in (("as-is", x), ("x5", x_extreme)):
        sc = ens.score_player(
            session_id=sess.session_id,
            slot=0,
            player_id=f"p_{label}",
            features=vec,
            engagement_seqs=None,
            engagement_scalars=None,
        )
        assert sc.meta["tabular_used"] is False, label
        ceiling = prior + SPARSE_ZERO_ENG_SHRINK * (1.0 - prior) + 1e-6
        assert sc.p_fused <= ceiling, (
            f"{label}: p_fused={sc.p_fused:.4f} exceeded zero-information "
            f"ceiling {ceiling:.4f} -- a session with no engagements must "
            "never approach the watch band on tabular evidence alone"
        )


def test_full_capability_aimbot_still_scored_by_tabular():
    """Guard against the regression made once before (HANDOFF entry 35): the
    sparse path must be invisible to genuine, fully-observed detections."""
    ens = _load_ensemble()
    sess, _ = simulate_session(
        session_id="t_full",
        cheat_families=["blatant", "clean", "clean", "clean"],
        duration_s=45.0,
        seed=12,
    )
    from aegis.core.config import AegisConfig
    from aegis.features.pipeline import (
        extract_engagement_tensors,
        extract_session_features,
    )

    cfg = AegisConfig()
    caps = set(SIM_CAPS)
    feats = extract_session_features(sess, caps, spec=ens.spec)
    f0 = next(f for f in feats if f.slot == 0)
    tensors = [t for t in extract_engagement_tensors(
        sess, caps, cfg=cfg.features, seq_len=cfg.model.seq_len) if t.slot == 0]
    seqs = np.stack([t.data for t in tensors]) if tensors else None
    scals = np.stack([t.scalars for t in tensors]) if tensors else None

    sc = ens.score_player(
        session_id=sess.session_id,
        slot=0,
        player_id="aimbot",
        features=f0.values,
        engagement_seqs=seqs,
        engagement_scalars=scals,
    )
    assert sc.meta["n_engagements"] > 0
    assert sc.meta["tabular_used"] is True, (
        "a full-capability, engaged session must keep the tabular channel -- "
        "the sparse path leaked onto legitimate detections"
    )
    assert sc.meta["shrink"] == pytest.approx(1.0), (
        f"full-capability coverage {sc.meta['coverage']:.2f} should sit above "
        "the 0.5 no-op floor"
    )
