"""Dataset construction from synthetic (or real) sessions.

Produces the matrices the training loop consumes:

  * session feature matrix X_sess, labels y_sess, player ids
  * engagement tensors X_eng (N, C, T), scalars, labels

Also writes a small manifest so a training run is reproducible from seed +
config alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.config import FeatureConfig
from ..core.registry import Capability
from ..features.pipeline import (
    FeatureSpec,
    build_spec,
    extract_engagement_tensors,
    extract_session_features,
    stack_features,
)
from .simulate import CHEAT_FAMILIES, make_lobby, simulate_session

# Capabilities the simulator actually provides.
SIM_CAPS = frozenset({
    Capability.VIEW_ANGLES, Capability.HIGH_RATE_VIEW, Capability.POSITIONS,
    Capability.VELOCITIES, Capability.SHOTS, Capability.DAMAGE,
    Capability.HITGROUPS, Capability.VISIBILITY, Capability.RECOIL_STATE,
    Capability.MOVEMENT_INPUT, Capability.BUTTONS, Capability.AIM_TARGET,
})

# What basically any integration provides from tick one -- dropping these
# would not be a "partial capability" session, it would not be a usable one.
CORE_CAPS = frozenset({
    Capability.VIEW_ANGLES, Capability.HIGH_RATE_VIEW, Capability.POSITIONS,
    Capability.SHOTS,
})
OPTIONAL_CAPS = tuple(SIM_CAPS - CORE_CAPS)


def sample_session_profile(rng: np.random.Generator, base_duration_s: float,
                             dropout: float, short_session_prob: float,
                             ) -> tuple[set[Capability], float]:
    """Capability set and duration for one session, drawn jointly.

    Sampling these two independently (as an earlier version of this function
    did) makes the joint "short clip AND minimal capabilities" case --
    exactly what a real highlight clip, attacker-only replay, or kill-window
    export actually looks like -- only `short_session_prob * dropout * 0.5`
    of sessions, compounding two already-small probabilities into one that's
    too rare to matter (at the defaults, roughly 2 of 64 training sessions).
    Verified directly: even after both the capability and duration fixes
    shipped independently, the trained GBDT's output for real held-out data
    matching that joint profile was *completely insensitive* to the actual
    feature values (randomly rescaling every populated real feature by
    0.2x-3x, independently, left the output unchanged to 4 decimal places) --
    conclusive evidence that the joint profile still doesn't get enough
    distinct training examples to learn real splits, only enough to end up
    on some single, coarse, undertrained leaf. Explicitly guaranteeing the
    joint case its own slice of sessions is the fix that's actually implied
    by that result; see COMPARISON.md.
    """
    if rng.random() < short_session_prob * 0.5:
        return set(CORE_CAPS), float(rng.uniform(3.0, 15.0))
    return sample_session_caps(rng, dropout), sample_session_duration(rng, base_duration_s, short_session_prob)


def sample_session_duration(rng: np.random.Generator, base_duration_s: float,
                              short_session_prob: float) -> float:
    """Occasionally sample a short-clip duration instead of a full match.

    Found the hard way: every training session used to be a 45-90s "full
    match," so a genuinely short session (a single ~4.7s kill-window clip --
    exactly what a highlight reel, a demo snippet, or a live feed's first
    few seconds after (re)connecting looks like) was out of distribution on
    *duration* alone, independent of capability sparsity. Confirmed directly:
    a synthetic session simulated at the real clip's actual duration
    reproduced the identical saturated score multiple *different* labelled
    families landed on, using the exact same trained model that had already
    been fixed for capability sparsity. Spectral features are the most
    exposed -- a Welch PSD estimated from ~300 ticks is a different
    statistical object than one from ~5000, and the model had only ever
    seen the latter.
    """
    if rng.random() < short_session_prob:
        return float(rng.uniform(3.0, 15.0))
    return base_duration_s


def sample_session_caps(rng: np.random.Generator, dropout: float) -> set[Capability]:
    """One session's honest capability set for training.

    Every synthetic session used to be extracted with the full ``SIM_CAPS``
    set, every time -- the model never once saw the partial-capability,
    heavily-NaN feature vector that any non-CS2 generic_fps integration, or
    even a CS2 session with ``no_vis`` or an unattested client, actually
    produces. Confirmed against real (non-synthetic) human data: fed such a
    vector, the tabular model saturated to ~1.0 "cheat" confidence for two
    distinct real accounts regardless of their true label, because the
    missingness *pattern* itself was out of distribution, not because of
    anything in the populated columns.

    Real integrations drop capabilities in *correlated blocks*, not
    independently -- a demo-only feed lacks visibility, damage, hitgroups,
    recoil, movement input, buttons, and aim_target all at once, because
    they all come from the same missing live-plugin channel, not from eight
    independent coin flips. An earlier version of this function dropped each
    optional capability independently, which made the all-core-only profile
    a 0.3**8 =~ 0.00007 event -- meaning it still, for practical purposes,
    never appeared in a training run, and the saturation this function
    exists to fix was still measured after that version shipped. Explicitly
    weighting the minimal profile fixes that.
    """
    r = rng.random()
    if r < dropout * 0.5:
        return set(CORE_CAPS)  # minimal profile, explicitly guaranteed representation
    if r < dropout:
        caps = set(CORE_CAPS)
        for cap in OPTIONAL_CAPS:
            if rng.random() < 0.5:
                caps.add(cap)
        return caps
    return set(SIM_CAPS)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray  # 1 = any cheat, 0 = clean
    family: list[str]
    player_ids: list[str]
    session_ids: list[str]
    spec: FeatureSpec
    # Engagement-level (may be empty if build_engagements=False)
    eng_X: np.ndarray = field(default_factory=lambda: np.zeros((0, 14, 128), dtype=np.float32))
    eng_scalars: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    eng_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    eng_player_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "session.npz",
            X=self.X, y=self.y,
        )
        np.savez_compressed(
            path / "engagement.npz",
            X=self.eng_X, scalars=self.eng_scalars, y=self.eng_y,
        )
        (path / "meta.json").write_text(
            json.dumps({
                "spec": self.spec.to_dict(),
                "family": self.family,
                "player_ids": self.player_ids,
                "session_ids": self.session_ids,
                "eng_player_ids": self.eng_player_ids,
                "meta": self.meta,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        sess = np.load(path / "session.npz")
        eng = np.load(path / "engagement.npz")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(
            X=sess["X"], y=sess["y"],
            family=meta["family"], player_ids=meta["player_ids"],
            session_ids=meta["session_ids"],
            spec=FeatureSpec.from_dict(meta["spec"]),
            eng_X=eng["X"], eng_scalars=eng["scalars"], eng_y=eng["y"],
            eng_player_ids=meta.get("eng_player_ids", []),
            meta=meta.get("meta", {}),
        )


def generate_dataset(
    n_sessions: int = 40,
    duration_s: float = 90.0,
    seed: int = 1337,
    cheat_rate: float = 0.30,
    lobby_size: int = 6,
    seq_len: int = 128,
    max_eng_per_player: int = 24,
    feature_cfg: FeatureConfig | None = None,
    build_engagements: bool = True,
    families: tuple[str, ...] | None = None,
    capability_dropout: float = 0.3,
    short_session_prob: float = 0.2,
) -> Dataset:
    """Simulate ``n_sessions`` matches and extract features.

    ``capability_dropout`` controls what fraction of *optional* capabilities
    (everything outside ``CORE_CAPS``) each session independently drops,
    training the model on a realistic mix of full- and partial-capability
    inputs instead of only ever seeing every channel populated.
    ``short_session_prob`` is the fraction of sessions simulated as a short
    (3-15s) clip instead of a full ``duration_s`` match -- see
    ``sample_session_duration`` for why this matters as much as capability
    sparsity does. Set either to 0.0 to reproduce the old
    (validated-to-be-out-of-distribution-on-real-data) behaviour.
    """
    rng = np.random.default_rng(seed)
    feature_cfg = feature_cfg or FeatureConfig()
    families = families or tuple(f for f in CHEAT_FAMILIES if f != "clean")
    # spec is built from the full capability set so the column layout stays
    # fixed regardless of which extractors a given session's caps satisfy --
    # per-session extraction below still writes NaN for anything gated out.
    spec = build_spec(set(SIM_CAPS))

    rows = []
    fam_out: list[str] = []
    pids: list[str] = []
    sids: list[str] = []
    eng_X: list[np.ndarray] = []
    eng_S: list[np.ndarray] = []
    eng_y: list[int] = []
    eng_pids: list[str] = []

    for i in range(n_sessions):
        lobby = make_lobby(rng, cheat_rate=cheat_rate, families=families, size=lobby_size)
        sess_caps, sess_duration = sample_session_profile(
            rng, duration_s, capability_dropout, short_session_prob,
        )
        # Guarantee at least some clean + some cheats across the run.
        sess, labels = simulate_session(
            session_id=f"sim_{seed}_{i:04d}",
            cheat_families=lobby,
            duration_s=sess_duration,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        feats = extract_session_features(sess, sess_caps, cfg=feature_cfg, spec=spec)
        for f in feats:
            lab = labels.get(f.slot, "clean")
            # Record the family *name* (e.g. "aim_smoothing"), not the
            # collapsed behaviour label ("aimbot"): per-family recall must be
            # able to see the extreme-humanised closet profile separately
            # from mid-range aimbots, or evaluation silently averages the
            # hardest case away.
            fam = lobby[f.slot] if f.slot < len(lobby) else lab
            rows.append(f)
            fam_out.append(fam)
            pids.append(f.player_id)
            sids.append(f.session_id)

        if build_engagements:
            tensors = extract_engagement_tensors(
                sess, sess_caps, cfg=feature_cfg, seq_len=seq_len,
                max_per_player=max_eng_per_player,
            )
            for t in tensors:
                lab = labels.get(t.slot, "clean")
                eng_X.append(t.data)
                eng_S.append(t.scalars)
                eng_y.append(0 if lab == "clean" else 1)
                eng_pids.append(t.player_id)

    X, spec = stack_features(rows)
    y = np.array([0 if f == "clean" else 1 for f in fam_out], dtype=np.int32)

    if eng_X:
        eng_Xa = np.stack(eng_X).astype(np.float32)
        # Pad scalars to common width.
        width = max(s.shape[0] for s in eng_S)
        eng_Sa = np.zeros((len(eng_S), width), dtype=np.float32)
        for i, s in enumerate(eng_S):
            eng_Sa[i, : s.shape[0]] = s
        eng_ya = np.array(eng_y, dtype=np.int32)
    else:
        eng_Xa = np.zeros((0, len(spec.seq_channels), seq_len), dtype=np.float32)
        eng_Sa = np.zeros((0, 1), dtype=np.float32)
        eng_ya = np.zeros(0, dtype=np.int32)

    return Dataset(
        X=X, y=y, family=fam_out, player_ids=pids, session_ids=sids, spec=spec,
        eng_X=eng_Xa, eng_scalars=eng_Sa, eng_y=eng_ya, eng_player_ids=eng_pids,
        meta={
            "n_sessions": n_sessions, "duration_s": duration_s, "seed": seed,
            "cheat_rate": cheat_rate, "lobby_size": lobby_size,
            "families": list(families),
            "n_clean": int((y == 0).sum()), "n_cheat": int((y == 1).sum()),
            "n_engagements": int(eng_ya.size),
        },
    )


def split_by_player(
    ds: Dataset, val_frac: float = 0.2, cal_frac: float = 0.15, seed: int = 0
) -> tuple[Dataset, Dataset, Dataset]:
    """Player-level split so the same account never appears in train and val.

    Returns (train, calibrate, test).
    """
    rng = np.random.default_rng(seed)
    players = np.array(sorted(set(ds.player_ids)))
    rng.shuffle(players)
    n = len(players)
    n_test = max(1, int(n * val_frac))
    n_cal = max(1, int(n * cal_frac))
    test_p = set(players[:n_test])
    cal_p = set(players[n_test:n_test + n_cal])
    train_p = set(players[n_test + n_cal:])

    def subset(keep: set[str]) -> Dataset:
        idx = [i for i, p in enumerate(ds.player_ids) if p in keep]
        eidx = [i for i, p in enumerate(ds.eng_player_ids) if p in keep]
        return Dataset(
            X=ds.X[idx], y=ds.y[idx],
            family=[ds.family[i] for i in idx],
            player_ids=[ds.player_ids[i] for i in idx],
            session_ids=[ds.session_ids[i] for i in idx],
            spec=ds.spec,
            eng_X=ds.eng_X[eidx] if eidx else ds.eng_X[:0],
            eng_scalars=ds.eng_scalars[eidx] if eidx else ds.eng_scalars[:0],
            eng_y=ds.eng_y[eidx] if eidx else ds.eng_y[:0],
            eng_player_ids=[ds.eng_player_ids[i] for i in eidx],
            meta={**ds.meta, "split_players": len(keep)},
        )

    return subset(train_p), subset(cal_p), subset(test_p)
