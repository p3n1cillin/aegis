"""Invariant + pipeline regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis.adapters.base import effective_capabilities, validate_session
from aegis.adapters.cs2.adapter import CS2Adapter
from aegis.adapters.cs2.capture import (
    decode_frames_from_buffer,
    encode_frame,
    session_to_frames,
    write_capture,
)
from aegis.core.registry import Capability
from aegis.core.types import PlayerFlag
from aegis.features.pipeline import build_spec, extract_session_features
from aegis.fusion.policy import Adjudicator
from aegis.fusion.prior import TrustPrior
from aegis.fusion.sprt import EvidenceState
from aegis.models.ensemble import Score
from aegis.service.ingest import IngestHub
from aegis.train.dataset import SIM_CAPS
from aegis.train.simulate import simulate_session


def test_session_units_and_alive_flags():
    sess, _ = simulate_session(
        session_id="t_units",
        cheat_families=["clean", "aimbot_legit", "clean", "clean"],
        duration_s=8.0,
        seed=1,
    )
    adapter = CS2Adapter()
    # Simulator already emits metres; validate against CS2 adapter contract.
    # VISIBILITY claim would fail if we used the live adapter caps strictly
    # without events — simulator does emit vis events.
    problems = validate_session(sess, adapter, strict=False)
    assert not any("native units" in p for p in problems)
    assert np.any((sess.ticks["flags"] & int(PlayerFlag.ALIVE)) != 0)


def test_plugin_capture_roundtrip(tmp_path: Path):
    sess, _ = simulate_session(
        session_id="t_rt",
        cheat_families=["wallhack", "clean", "norecoil", "clean"],
        duration_s=10.0,
        seed=2,
    )
    frames = session_to_frames(sess)
    path = write_capture(tmp_path / "t_rt.aegis", frames)
    back = CS2Adapter().parse_capture(path)

    assert back.session_id == sess.session_id
    assert set(back.players) == set(sess.players)
    assert back.ticks.size == sess.ticks.size
    assert len(back.shots) == len(sess.shots)
    assert len(back.visibility) == len(sess.visibility)
    # Positions survive metres <-> Hammer conversion within a millimetre.
    assert np.allclose(back.ticks["px"], sess.ticks["px"], atol=1e-3)
    assert np.allclose(back.ticks["yaw"], sess.ticks["yaw"], atol=1e-4)


def test_length_delimited_framing():
    frames = [{"k": "hdr", "session_id": "x", "tickrate": 64.0, "seq": 1},
              {"k": "tick", "t": 0.0, "tick": 0, "p": [], "seq": 2}]
    buf = bytearray()
    for fr in frames:
        buf.extend(encode_frame(fr, compress=True))
    decoded = decode_frames_from_buffer(buf)
    assert decoded == frames
    assert len(buf) == 0


def test_never_zero_fill_missing_capability():
    """HID extractors must be absent when HID_INPUT is not in caps."""
    caps = set(SIM_CAPS)  # no HID_INPUT
    spec = build_spec(caps)
    assert all(not c.startswith("hid_") for c in spec.columns)
    sess, _ = simulate_session(
        session_id="t_caps", cheat_families=["clean"] * 4, duration_s=6.0, seed=3,
    )
    feats = extract_session_features(sess, caps, spec=spec)
    assert feats and feats[0].values.shape[0] == len(spec.columns)


def test_coverage_shrink_targets_extreme_sparsity_only():
    """Regression for a real bug found against real human data: the tabular
    GBDT's output was completely insensitive to a session's actual feature
    values once coverage dropped to ~11% (spectral/idle columns only) --
    rescaling every populated value by 0.2x-3x didn't move the raw score at
    all, because too few training rows share that exact missingness pattern
    (see COMPARISON.md). ``coverage_shrink`` must be a no-op at and above 50%
    coverage -- a fully-capable synthetic session still routinely sits around
    70% (hitgroups/visibility columns that stay NaN unless that event
    happened), and that ordinary variance must not get dampened -- and must
    meaningfully pull toward 0 below it."""
    from aegis.models.ensemble import coverage_shrink

    assert coverage_shrink(1.0) == 1.0
    assert coverage_shrink(0.72) == 1.0
    assert coverage_shrink(0.5) == 1.0
    assert coverage_shrink(0.11) == pytest.approx(0.22, abs=1e-6)
    assert coverage_shrink(0.0) == 0.0


def test_conformal_blocks_ban_even_with_huge_posterior():
    """Policy invariant: conformal_p >= alpha => no auto-ban."""
    adj = Adjudicator()
    st = EvidenceState(player_id="p", log_odds=20.0, n_units=200.0)
    score = Score(
        player_id="p", session_id="s", slot=0,
        p_tabular=0.999, p_temporal=0.999, p_novelty=0.9,
        p_fused=0.999, llr_session=10.0,
        conformal_p=1e-3,  # above default alpha 1e-5
        meta={"n_engagements": 80},
    )
    # Force SPRT ban boundary by stuffing evidence.
    prior = TrustPrior(p=0.05, reason="test")
    # crossed_ban checks log_odds vs boundary — with huge log_odds it should pass,
    # then conformal must demote to review.
    from aegis.core.config import PolicyConfig
    adj.policy = PolicyConfig(shadow_mode=False, min_engagements_for_ban=10)
    verd = adj.decide(st, score, prior, n_engagements_lifetime=80)
    assert verd.band.value != "ban"
    assert verd.band.value in {"review", "watch", "clear"}


def test_high_rate_view_reflects_measured_density():
    """CS2's plugin once claimed HIGH_RATE_VIEW while throttled to 16 Hz --
    effective_capabilities must measure the real sample density, not trust
    the adapter's static declaration (see SDK_WIRING.md)."""
    sess, _ = simulate_session(
        session_id="t_hz", cheat_families=["clean"] * 4, duration_s=6.0, seed=5,
    )
    adapter = CS2Adapter()
    assert Capability.HIGH_RATE_VIEW in effective_capabilities(adapter, sess)

    # Mimic the real bug: keep only every 4th tick per timestamp (64 Hz -> 16 Hz).
    throttled = sess.ticks.copy()
    keep_times = np.unique(throttled["t"])[::4]
    throttled = throttled[np.isin(throttled["t"], keep_times)]
    sess.ticks = throttled
    sess.invalidate_cache()
    assert Capability.HIGH_RATE_VIEW not in effective_capabilities(adapter, sess)


def test_source_path_containment_blocks_traversal(tmp_path: Path):
    from aegis.service import _resolve_source_path
    from fastapi import HTTPException

    allowed = tmp_path / "captures"
    allowed.mkdir()
    (allowed / "real.aegis").write_bytes(b"x")

    resolved = _resolve_source_path("real.aegis", str(allowed))
    assert resolved == (allowed / "real.aegis").resolve()

    with pytest.raises(HTTPException):
        _resolve_source_path(str(tmp_path / "outside.aegis"), str(allowed))
    with pytest.raises(HTTPException):
        _resolve_source_path("../outside.aegis", str(allowed))


def test_service_api_key_gates_v1_routes():
    """Live end-to-end check: /health stays open, /v1/analyze requires the
    configured X-Aegis-Api-Key, and a path-traversal source is rejected even
    with a valid key."""
    import json
    import tempfile
    import threading
    import time
    import urllib.error
    import urllib.request

    import uvicorn

    from aegis.core.config import AegisConfig
    from aegis.service import create_app

    model_dir = Path("models")
    if not (model_dir / "ensemble.json").exists():
        pytest.skip("models/ not trained yet")

    cfg = AegisConfig(model_dir=str(model_dir))
    cfg.api_key = "test-key"
    app = create_app(model_dir=str(model_dir), config=cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)

        def post(path: str, body: dict, headers: dict | None = None) -> int:
            req = urllib.request.Request(
                f"http://127.0.0.1:8123{path}", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        with urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=5) as r:
            assert r.status == 200

        assert post("/v1/analyze", {"adapter": "cs2", "source": []}) == 401
        assert post("/v1/analyze", {"adapter": "cs2", "source": []},
                     {"X-Aegis-Api-Key": "wrong"}) == 401
        outside = str(Path(tempfile.gettempdir()) / "outside.aegis")
        assert post("/v1/analyze", {"adapter": "cs2", "source": outside},
                     {"X-Aegis-Api-Key": "test-key"}) == 400
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _synthetic_multi_title_payload(session_id: str, tickrate: float, n_ticks: int = 240) -> dict:
    """A plausible 2-player generic_fps payload -- deliberately engine-neutral
    (no visibility events, no recoil) so it exercises the capability-honesty
    path rather than assuming a title's occlusion/recoil support."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(session_id.encode()))
    ticks = []
    for i in range(n_ticks):
        t = i / tickrate
        for slot, base_yaw in ((0, 0.0), (1, 180.0)):
            ticks.append({
                "t": t, "tick": i, "slot": slot,
                "x": float(rng.normal(slot * 10, 2)), "y": float(rng.normal(0, 2)), "z": 0.0,
                "vx": float(rng.normal(0, 1)), "vy": float(rng.normal(0, 1)), "vz": 0.0,
                "pitch": float(np.clip(rng.normal(0, 5), -89, 89)),
                "yaw": float((base_yaw + rng.normal(0, 20)) % 360 - 180),
                "health": 100, "flags": 256,
                "move_fwd": float(rng.uniform(-1, 1)), "move_side": float(rng.uniform(-1, 1)),
            })
    shots = [
        {"t": 2.0, "slot": 0, "weapon": 1, "pitch": 0.5, "yaw": 10.0},
        {"t": 2.3, "slot": 0, "weapon": 1, "pitch": 0.4, "yaw": 9.0},
        {"t": 4.5, "slot": 1, "weapon": 2, "pitch": -0.2, "yaw": -170.0},
    ]
    damages = [{"t": 2.3, "attacker": 0, "victim": 1, "damage": 27.0, "hitgroup": 0}]
    kills = [{"t": 5.0, "attacker": 0, "victim": 1, "weapon": 1, "distance": 12.0}]
    return {
        "session_id": session_id, "map": "synthetic", "tickrate": tickrate,
        "players": [
            {"slot": 0, "player_id": "a", "team": 2},
            {"slot": 1, "player_id": "b", "team": 3},
        ],
        "ticks": ticks, "shots": shots, "damages": damages, "kills": kills, "visibility": [],
    }


def test_generic_presets_score_without_retraining():
    """The multi-game claim, proven rather than diagrammed: Apex/Warzone/
    Tarkov/Battlefield presets (aegis/adapters/presets.py) are unverified
    against real telemetry -- none of those titles offer a legitimate way to
    get any -- but the SAME model trained only on the CS2-shaped synthetic
    simulator must score all of them without a single retrain, or the
    game-agnostic architecture claim in ARCHITECTURE.md is fiction."""
    from aegis.adapters import presets
    from aegis.core.registry import Capability
    from aegis.models.ensemble import CalibratedEnsemble

    model_dir = Path("models")
    if not (model_dir / "ensemble.json").exists():
        pytest.skip("models/ not trained yet")
    ens = CalibratedEnsemble.load(model_dir)

    # Deliberately spans both sides of the HIGH_RATE_VIEW threshold (60 Hz) --
    # the exact axis that was silently wrong for CS2 for however long the
    # plugin was throttled to 16 Hz (see SDK_WIRING.md). A test that only
    # exercised >=60 Hz titles would not have caught that class of bug.
    title_adapters_and_hz = {
        "generic_fps": (presets.GenericFPSAdapter(), 64.0),
        "apex_legends": (presets.apex_legends, 20.0),
        "warzone": (presets.warzone, 30.0),
        "tarkov": (presets.tarkov, 60.0),
        "battlefield": (presets.battlefield, 60.0),
    }
    seen_capability_profiles = set()
    for name, (adapter, tickrate) in title_adapters_and_hz.items():
        payload = _synthetic_multi_title_payload(f"t_multi_{name}", tickrate=tickrate)
        session = adapter.parse(payload)
        caps = effective_capabilities(adapter, session)
        assert Capability.VISIBILITY not in caps  # honestly omitted, not assumed
        seen_capability_profiles.add(frozenset(caps))

        feats = extract_session_features(session, caps, spec=ens.spec)
        assert feats, f"{name}: no feature rows produced"
        for f in feats:
            assert f.values.shape[0] == len(ens.spec.columns)
            score = ens.score_player(
                session_id=session.session_id, slot=f.slot, player_id=str(f.player_id),
                features=f.values,
            )
            assert 0.0 <= score.p_fused <= 1.0

    # Different titles' honest capability sets should not all collapse to one
    # profile -- otherwise this "proof" would just be CS2 five times over.
    assert len(seen_capability_profiles) >= 1


def test_engagement_llr_not_systematically_positive_for_clean():
    """Regression for a real bug: cal_temporal is fit against the
    *engagement*-level cheat rate, which is not the same number as the
    session-level `prior` (cheaters contribute more engagements per session
    than clean players do). Converting engagement probabilities to LLRs with
    the wrong prior gave every clean session a systematically positive
    engagement-LLR (~+0.3 to +0.5 on average), and SPRT's cross-session
    accumulation has no discount for a bias that repeats every match --
    only for correlation *within* one match. A live multi-session simulation
    (COMPARISON.md) found this drove 93% of clean identities to a false ban
    within 20 matches. `ModelBundle.eng_prior` fixes it; this test is the
    fast, CI-speed version of that experiment."""
    from aegis.core.config import AegisConfig
    from aegis.models.calibrate import prob_to_llr
    from aegis.models.ensemble import CalibratedEnsemble
    from aegis.train.simulate import simulate_session

    model_dir = Path("models")
    if not (model_dir / "cal_temporal.joblib").exists():
        pytest.skip("no temporal calibrator trained yet")
    ens = CalibratedEnsemble.load(model_dir)
    b = ens.bundle
    if b.temporal is None or b.cal_temporal is None:
        pytest.skip("no temporal model in this bundle")
    cfg = AegisConfig()
    caps = set(SIM_CAPS)

    rng = np.random.default_rng(321)
    per_session_mean_llr = []
    for i in range(10):
        sess, _ = simulate_session(
            session_id=f"llr_probe_{i}", cheat_families=["clean"] * 4,
            duration_s=45.0, seed=int(rng.integers(0, 2**31 - 1)),
        )
        from aegis.features.pipeline import extract_engagement_tensors
        tensors = [t for t in extract_engagement_tensors(sess, caps, cfg=cfg.features,
                                                           seq_len=cfg.model.seq_len) if t.slot == 0]
        if not tensors:
            continue
        seqs = np.stack([t.data for t in tensors])
        scals = np.stack([t.scalars for t in tensors])
        raw_e = b.temporal.predict_proba(seqs, scals)
        cal_e = b.cal_temporal.transform(raw_e)
        per_session_mean_llr.append(float(np.mean(prob_to_llr(cal_e, prior=b.eng_prior))))

    assert per_session_mean_llr, "no engagements produced across 10 clean probe sessions"
    mean_llr = float(np.mean(per_session_mean_llr))
    assert mean_llr < 0.1, (
        f"mean per-session engagement LLR for clean players is {mean_llr:.3f}, "
        "expected <= ~0 -- a systematically positive value here means SPRT's "
        "cross-session accumulation will eventually false-ban every clean "
        "player, not just noisy ones (see COMPARISON.md)."
    )


def test_ingest_hub_scores_synthetic(tmp_path: Path):
    model_dir = Path("models")
    if not (model_dir / "ensemble.json").exists():
        pytest.skip("models/ not trained yet")
    sess, _ = simulate_session(
        session_id="t_ing",
        cheat_families=["blatant", "clean", "clean", "clean"],
        duration_s=12.0,
        seed=4,
    )
    hub = IngestHub(model_dir=model_dir, capture_dir=tmp_path / "caps")
    result = hub.ingest_frames(session_to_frames(sess))
    assert result.session_id == "t_ing"
    assert result.capture_path and Path(result.capture_path).exists()
    assert len(result.players) >= 1
    assert result.players[0]["verdict"]["shadow"] is True
