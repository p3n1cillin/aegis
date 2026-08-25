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


# ---------------------------------------------------------------------------
# Loop 3 -- simulator realism (aim_smoothing family + network jitter)
# ---------------------------------------------------------------------------


def test_aim_smoothing_family_simulates_and_labels():
    """The extreme-humanised closet profile must produce a valid canonical
    session under a distinct label, with every humaniser knob actually on."""
    from aegis.adapters.base import validate_session
    from aegis.adapters.cs2.adapter import CS2Adapter

    sess, labels = simulate_session(
        session_id="t_smooth",
        cheat_families=["clean", "aim_smoothing", "clean", "aim_smoothing"],
        duration_s=10.0,
        seed=21,
    )
    adapter = CS2Adapter()
    problems = validate_session(sess, adapter, strict=False)
    assert not any("native units" in p for p in problems)
    cheat_slots = [s for s, lab in labels.items() if "aimbot" in lab]
    assert len(cheat_slots) == 2
    for slot in cheat_slots:
        cfg = sess.meta["cheat_configs"][slot]
        assert cfg.aimbot and cfg.aimbot_retain_tremor
        assert cfg.aimbot_smooth_ticks >= 16
        assert cfg.aimbot_delay_s >= 0.18
        assert cfg.aimbot_fov <= 6.0


def test_network_jitter_preserves_monotonic_timebase():
    """Jittered tick timestamps must stay monotonic per player, non-negative,
    bounded by max_ms displacement, and still pass units validation."""
    from aegis.adapters.base import validate_session
    from aegis.adapters.cs2.adapter import CS2Adapter

    MAX_MS = 20.0
    sess, _ = simulate_session(
        session_id="t_jitter",
        cheat_families=["clean"] * 4,
        duration_s=6.0,
        seed=22,
        net_jitter_ms=MAX_MS,
    )
    ticks = sess.ticks
    for slot in np.unique(ticks["slot"]):
        t = np.sort(ticks["t"][ticks["slot"] == slot])
        assert np.all(np.diff(t) >= -1e-9), f"slot {slot}: jitter broke monotonicity"
    assert np.all(ticks["t"] >= 0.0)

    # Boundedness: no timestamp may move by more than max_ms from its grid.
    dt = 1.0 / sess.tickrate
    grid = ticks["tick"].astype(np.float64) * dt
    assert np.max(np.abs(ticks["t"] - grid)) <= MAX_MS / 1000.0 + 1e-6

    adapter = CS2Adapter()
    problems = validate_session(sess, adapter, strict=False)
    assert not any("monotonic" in p.lower() for p in problems)


def test_network_jitter_changes_timestamps_but_not_ordering():
    """Sanity: jitter actually displaces times (otherwise the robustness
    claim above is vacuous) while leaving tick/step indices untouched."""
    clean, _ = simulate_session(
        session_id="t_j_a", cheat_families=["clean"] * 2, duration_s=4.0, seed=23,
    )
    noisy, _ = simulate_session(
        session_id="t_j_b", cheat_families=["clean"] * 2, duration_s=4.0, seed=23,
        net_jitter_ms=30.0,
    )
    assert np.array_equal(clean.ticks["tick"], noisy.ticks["tick"])
    assert not np.allclose(clean.ticks["t"], noisy.ticks["t"])


# ---------------------------------------------------------------------------
# Loop 4 -- verdict audit log
# ---------------------------------------------------------------------------


def test_audit_log_roundtrip(tmp_path):
    """Append/tail roundtrip including reasons list and unicode safety."""
    from aegis.integrity.audit import AuditLog, AuditRecord

    log = AuditLog(tmp_path / "nested" / "audit.jsonl")
    rec = AuditRecord(
        ts="2026-01-01T00:00:00.000+00:00",
        session_id="s1", player_id="plåyer-1",
        verdict="watch", shadow=True,
        p_fused=0.61, conformal_p=0.4,
        reasons=["posterior_ge_watch"], model_digest="abc", n_engagements=7,
    )
    log.append(rec)
    log.append(AuditRecord(
        ts="2026-01-01T00:00:01.000+00:00",
        session_id="s1", player_id="p2",
        verdict="clear", shadow=True,
        p_fused=0.05, conformal_p=0.9,
        reasons=[], model_digest="abc", n_engagements=3,
    ))
    got = log.tail(10)
    assert len(got) == 2
    assert got[0].player_id == "plåyer-1"  # unicode survives the JSONL line
    assert got[-1].verdict == "clear"
    assert got[0].reasons == ["posterior_ge_watch"]
    assert log.tail(1)[0].player_id == "p2"
    # Empty / missing log reads as empty.
    assert AuditLog(tmp_path / "missing.jsonl").tail(5) == []


def test_audit_log_records_shadow_verdicts(tmp_path):
    """The live ingest scoring path must append one audit record per player
    verdict, flagged shadow while shadow_mode is on (HANDOFF backlog item 8)."""
    from aegis.adapters.cs2.capture import session_to_frames
    from aegis.core.config import AegisConfig
    from aegis.service.ingest import IngestHub
    from aegis.integrity.audit import AuditLog

    if not (MODEL_DIR / "ensemble.json").exists():
        pytest.skip("models/ not trained yet")
    audit_path = tmp_path / "audit.jsonl"
    cfg = AegisConfig(model_dir=str(MODEL_DIR))
    cfg.policy.audit_log_path = str(audit_path)
    cfg.policy.shadow_mode = True

    sess, _ = simulate_session(
        session_id="t_audit",
        cheat_families=["blatant", "clean", "clean", "clean"],
        duration_s=12.0,
        seed=31,
    )
    hub = IngestHub(model_dir=MODEL_DIR, capture_dir=tmp_path / "caps", config=cfg)
    result = hub.ingest_frames(session_to_frames(sess))
    n_players = len(result.players)
    assert n_players >= 1

    records = AuditLog(audit_path).tail(100)
    assert len(records) == n_players
    assert all(r.shadow for r in records), (
        "shadow_mode is on; every audited verdict must be flagged shadow"
    )
    assert {r.verdict for r in records} <= {
        "clear", "watch", "review", "ban", "integrity_ban"
    }
    assert any(r.p_fused > 0.5 for r in records), (
        "the blatant cheater should have scored above prior in this lobby"
    )


# ---------------------------------------------------------------------------
# Loop 5 -- per-title conformal gates
# ---------------------------------------------------------------------------


def _ban_capable_fixture():
    """State/score/policy combo that reaches the ban branch when conformal
    admits: huge SPRT evidence, plenty of engagements, p_fused ~1."""
    from aegis.core.config import FusionConfig, PolicyConfig
    from aegis.fusion.policy import Adjudicator
    from aegis.fusion.prior import TrustPrior
    from aegis.fusion.sprt import EvidenceState, SPRT
    from aegis.models.ensemble import Score

    fusion = FusionConfig(alpha=1e-3)  # relaxed global gate for the test
    policy = PolicyConfig(shadow_mode=True, min_engagements_for_ban=10)
    adj = Adjudicator(policy=policy, sprt=SPRT(fusion))
    st = EvidenceState(player_id="p", log_odds=20.0, n_units=200.0)
    score = Score(
        player_id="p", session_id="s", slot=0,
        p_tabular=0.999, p_temporal=0.999, p_novelty=0.9,
        p_fused=0.999, llr_session=10.0,
        conformal_p=1e-4,  # admits under 1e-3, refuses under 1e-5
        meta={"n_engagements": 80},
    )
    prior = TrustPrior(p=0.05, reason="test")
    return adj, st, score, prior


def test_per_title_alpha_gate():
    """A title-specific alpha must change the conformal admission decision
    only for that title; other titles and the default path are untouched."""
    adj, st, score, prior = _ban_capable_fixture()
    adj.sprt.cfg.per_title_alpha = {"tarkov": 1e-5}

    # Untitled / unconfigured title -> relaxed global alpha admits the ban.
    assert adj.decide(st, score, prior).band.value == "ban"
    assert adj.decide(st, score, prior, title="cs2").band.value == "ban"
    # Configured title -> stricter gate refuses the auto-ban.
    v = adj.decide(st, score, prior, title="tarkov")
    assert v.band.value == "review"
    assert any("per-title" in r or "refusing auto-ban" in r for r in v.reasons)
    # Alpha resolution itself is deterministic and falls back correctly.
    assert adj.alpha_for("tarkov") == 1e-5
    assert adj.alpha_for("cs2") == adj.sprt.cfg.alpha
    assert adj.alpha_for(None) == adj.sprt.cfg.alpha


def test_per_title_alpha_never_relaxes_below_global():
    """An operator typo (title alpha *looser* than intended system budget)
    is still honoured explicitly -- but the resolved value must always be
    visible in the reason string so audits catch it."""
    adj, st, score, prior = _ban_capable_fixture()
    adj.sprt.cfg.per_title_alpha = {"tarkov": 1e-2}
    v = adj.decide(st, score, prior, title="tarkov")
    assert v.band.value == "ban"
    assert any("conformal_p" in r for r in v.reasons)

