"""Invariant + pipeline regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis.adapters.base import validate_session
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
