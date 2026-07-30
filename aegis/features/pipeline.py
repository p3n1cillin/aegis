"""Feature pipeline: session -> (feature matrix, sequence tensors).

Two products come out of here and both feed the model layer:

  ``SessionFeatures``     a fixed-order float vector per player per session,
                          consumed by the gradient-boosted model and the
                          anomaly autoencoder.
  ``EngagementTensor``    a (C, T) float array per engagement, consumed by the
                          temporal encoder.

Column order is derived from the registry and frozen into
``FeatureSpec.columns``, which is persisted alongside every trained model.
Loading a model with a mismatched spec is a hard error -- silently reordered
features are the classic way an ML pipeline starts convicting innocent
players.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..core.config import FeatureConfig
from ..core.registry import EXTRACTORS, Capability
from ..core.types import PlayerFlag, Session
from .base import Engagement, PlayerContext, build_context

# Import for registration side effects. Order does not matter; column order is
# derived from sorted extractor names.
from . import (  # noqa: F401
    aim,
    input_dynamics,
    movement,
    outcome,
    reaction,
    recoil,
    spectral,
    visibility,
)


@dataclass(frozen=True)
class FeatureSpec:
    """Frozen description of the feature space a model was trained on."""

    columns: tuple[str, ...]
    seq_channels: tuple[str, ...]
    caps: tuple[str, ...]

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        h.update("|".join(self.columns).encode())
        h.update(b"##")
        h.update("|".join(self.seq_channels).encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "seq_channels": list(self.seq_channels),
                "caps": list(self.caps), "digest": self.digest}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(tuple(d["columns"]), tuple(d["seq_channels"]), tuple(d.get("caps", ())))


SEQ_CHANNELS = (
    "yaw_vel",
    "pitch_vel",
    "ang_speed",
    "ang_accel",
    "ang_jerk",
    "target_err",
    "target_err_rate",
    "shot",
    "target_visible",
    "log_distance",
    "airborne",
    "player_speed",
    "scoped",
    "weapon_class",
)


def build_spec(caps: set[Capability]) -> FeatureSpec:
    cols: list[str] = []
    for name in sorted(EXTRACTORS):
        ex = EXTRACTORS[name]
        if getattr(ex, "scope", "session") != "session":
            continue
        if set(getattr(ex, "requires", ())) <= caps:
            cols.extend(ex.names)
    return FeatureSpec(tuple(cols), SEQ_CHANNELS, tuple(sorted(c.value for c in caps)))


@dataclass
class SessionFeatures:
    session_id: str
    game: str
    slot: int
    player_id: str
    values: np.ndarray  # float32, aligned to FeatureSpec.columns
    spec: FeatureSpec
    n_engagements: int
    duration_s: float
    attested: bool = False
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {c: float(v) for c, v in zip(self.spec.columns, self.values)}


@dataclass
class EngagementTensor:
    session_id: str
    slot: int
    player_id: str
    engagement_index: int
    data: np.ndarray  # (C, T) float32
    scalars: np.ndarray  # per-engagement scalar features from aim extractor
    scalar_names: tuple[str, ...]
    killed: bool


def extract_session_features(
    session: Session,
    caps: set[Capability],
    cfg: FeatureConfig | None = None,
    spec: FeatureSpec | None = None,
) -> list[SessionFeatures]:
    cfg = cfg or FeatureConfig()
    spec = spec or build_spec(caps)
    out: list[SessionFeatures] = []

    extractors = [
        EXTRACTORS[n] for n in sorted(EXTRACTORS)
        if getattr(EXTRACTORS[n], "scope", "session") == "session"
        and set(getattr(EXTRACTORS[n], "requires", ())) <= caps
    ]

    for slot in session.slots():
        info = session.players[slot]
        if info.is_bot:
            continue
        ctx = build_context(session, slot, cfg, caps)
        if len(ctx.view) < cfg.min_window_samples:
            continue
        values: dict[str, float] = {}
        for ex_cls in extractors:
            ex = ex_cls() if isinstance(ex_cls, type) else ex_cls
            try:
                values.update(ex.extract(ctx))
            except Exception as exc:  # a broken extractor must not kill the run
                values.update(ex.empty())
                values["_err_" + ex.name] = 1.0
                session.meta.setdefault("extractor_errors", []).append(f"{ex.name}:{exc!r}")
        vec = np.array([values.get(c, np.nan) for c in spec.columns], dtype=np.float32)
        out.append(
            SessionFeatures(
                session_id=session.session_id,
                game=session.game,
                slot=slot,
                player_id=info.player_id,
                values=vec,
                spec=spec,
                n_engagements=len(ctx.engagements),
                duration_s=session.duration(),
                attested=bool(session.meta.get("attested", False)),
            )
        )
    return out


def _engagement_tensor(ctx: PlayerContext, eng: Engagement, seq_len: int) -> np.ndarray:
    """Resample one engagement window into a (C, seq_len) tensor.

    Channels are scaled to roughly unit range here rather than in the model, so
    that a saved tensor is inspectable and the same normalisation applies at
    train and serve time without a separate scaler artefact.
    """
    v = ctx.view
    sl = v.window(eng.t_window_start, eng.t_window_end)
    n = sl.stop - sl.start
    if n < 8:
        return np.zeros((len(SEQ_CHANNELS), seq_len), dtype=np.float32)

    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, seq_len)

    def rs(x: np.ndarray) -> np.ndarray:
        return np.interp(dst, src, np.nan_to_num(x[sl], nan=0.0, posinf=0.0, neginf=0.0))

    g, err = ctx.target_angle_error(eng.target)
    err_w = np.nan_to_num(err[sl], nan=180.0)
    err_rate = np.gradient(err_w, v.dt) if err_w.size > 2 else np.zeros_like(err_w)

    t_win = v.t[sl]
    shot = np.zeros(n, dtype=np.float64)
    for s in eng.shots:
        i = int(np.searchsorted(t_win, s.t))
        if 0 <= i < n:
            shot[i] = 1.0

    vis = np.zeros(n, dtype=np.float64)
    if np.isfinite(eng.t_target_visible):
        vis[t_win >= eng.t_target_visible] = 1.0

    tk = ctx.ticks
    if tk.size:
        pspeed = np.interp(t_win, tk["t"], np.hypot(tk["vx"], tk["vy"]))
        flags = tk["flags"].astype(np.int64)
        idx = np.clip(np.searchsorted(tk["t"], t_win, side="right") - 1, 0, tk.size - 1)
        airborne = ((flags[idx] & int(PlayerFlag.ON_GROUND)) == 0).astype(np.float64)
        scoped = ((flags[idx] & int(PlayerFlag.SCOPED)) != 0).astype(np.float64)
    else:
        pspeed = np.zeros(n)
        airborne = np.zeros(n)
        scoped = np.zeros(n)

    log_d = np.full(n, np.log1p(max(eng.distance, 0.0)) / 4.0)
    wclass = np.full(n, eng.weapon / 32.0)

    chans = [
        rs(v.yaw_v) / 500.0,
        rs(v.pitch_v) / 500.0,
        rs(v.speed) / 500.0,
        rs(v.accel) / 5000.0,
        rs(v.jerk) / 50000.0,
        np.interp(dst, src, np.clip(err_w, 0, 180)) / 30.0,
        np.interp(dst, src, np.clip(err_rate, -2000, 2000)) / 500.0,
        np.interp(dst, src, shot),
        np.interp(dst, src, vis),
        np.interp(dst, src, log_d),
        np.interp(dst, src, airborne),
        np.interp(dst, src, pspeed) / 8.0,
        np.interp(dst, src, scoped),
        np.interp(dst, src, wclass),
    ]
    arr = np.asarray(chans, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_engagement_tensors(
    session: Session,
    caps: set[Capability],
    cfg: FeatureConfig | None = None,
    seq_len: int = 128,
    max_per_player: int | None = None,
) -> list[EngagementTensor]:
    cfg = cfg or FeatureConfig()
    aim_ex = aim.AimEngagementFeatures()
    out: list[EngagementTensor] = []
    for slot in session.slots():
        info = session.players[slot]
        if info.is_bot:
            continue
        ctx = build_context(session, slot, cfg, caps)
        engs = ctx.engagements
        if max_per_player is not None and len(engs) > max_per_player:
            step = len(engs) / max_per_player
            engs = [engs[int(i * step)] for i in range(max_per_player)]
        for i, e in enumerate(engs):
            try:
                data = _engagement_tensor(ctx, e, seq_len)
                sc = aim_ex.extract_one(ctx, e)
            except Exception:
                continue
            out.append(
                EngagementTensor(
                    session_id=session.session_id,
                    slot=slot,
                    player_id=info.player_id,
                    engagement_index=i,
                    data=data,
                    scalars=np.array([sc[k] for k in aim_ex.names], dtype=np.float32),
                    scalar_names=aim_ex.names,
                    killed=e.killed,
                )
            )
    return out


def stack_features(rows: list[SessionFeatures]) -> tuple[np.ndarray, FeatureSpec]:
    if not rows:
        raise ValueError("no feature rows")
    spec = rows[0].spec
    if any(r.spec.digest != spec.digest for r in rows):
        raise ValueError("feature spec mismatch across rows -- refusing to stack")
    return np.vstack([r.values for r in rows]).astype(np.float32), spec
