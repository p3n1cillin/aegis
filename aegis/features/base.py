"""Feature extraction framework.

Two granularities, because they answer different questions:

  * **Engagement** -- one duel: the ~1.3 s around a player shooting at another
    player. This is where aim-assistance shows up, and it is the unit of
    evidence for the sequential test.
  * **Session** -- aggregates over a whole match. This is where *consistency*
    shows up: humans have heavy-tailed, autocorrelated performance; cheats
    (especially "legit" ones with humaniser settings) are suspiciously
    stationary.

VACnet-style systems score engagements. Scoring only engagements makes you
blind to the cheats that are individually unremarkable but collectively
impossible, so Aegis carries both and fuses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from ..core.config import FeatureConfig
from ..core.registry import Capability
from ..core.timebase import ViewSignal
from ..core.types import (
    DamageEvent,
    KillEvent,
    PlayerFlag,
    Session,
    ShotEvent,
    angular_distance,
)


@dataclass(slots=True)
class Engagement:
    """One offensive encounter by ``slot`` against ``target``."""

    session_id: str
    slot: int
    target: int
    t_first_shot: float
    t_last_shot: float
    t_window_start: float
    t_window_end: float
    shots: list[ShotEvent] = field(default_factory=list)
    damages: list[DamageEvent] = field(default_factory=list)
    kill: KillEvent | None = None
    # Time the target became visible to the shooter (-inf if never / unknown).
    t_target_visible: float = float("-inf")
    weapon: int = 0
    distance: float = 0.0

    @property
    def killed(self) -> bool:
        return self.kill is not None

    @property
    def n_shots(self) -> int:
        return len(self.shots)


@dataclass
class PlayerContext:
    """Everything an extractor may look at for one player in one session.

    Built once per player; the expensive resampling/filtering is shared.
    """

    session: Session
    slot: int
    cfg: FeatureConfig
    view: ViewSignal
    ticks: np.ndarray
    engagements: list[Engagement]
    caps: set[Capability]
    # Angular distance (deg) from crosshair to each *currently engaged*
    # target, on the resampled grid. Filled lazily by aim extractors.
    _cache: dict = field(default_factory=dict)

    def cached(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def alive_mask(self) -> np.ndarray:
        """Alive mask on the resampled grid."""

        def _build():
            if self.ticks.size == 0:
                return np.zeros(len(self.view), dtype=bool)
            flags = self.ticks["flags"].astype(np.int64)
            alive = (flags & int(PlayerFlag.ALIVE)) != 0
            idx = np.searchsorted(self.ticks["t"], self.view.t, side="right") - 1
            idx = np.clip(idx, 0, self.ticks.size - 1)
            return alive[idx]

        return self.cached("alive_mask", _build)

    def flag_mask(self, flag: PlayerFlag) -> np.ndarray:
        def _build():
            if self.ticks.size == 0:
                return np.zeros(len(self.view), dtype=bool)
            on = (self.ticks["flags"].astype(np.int64) & int(flag)) != 0
            idx = np.searchsorted(self.ticks["t"], self.view.t, side="right") - 1
            idx = np.clip(idx, 0, self.ticks.size - 1)
            return on[idx]

        return self.cached(f"flag_{int(flag)}", _build)

    def target_angle_error(self, target: int) -> tuple[np.ndarray, np.ndarray]:
        """(grid_times, angular error in deg) from our crosshair to ``target``.

        Uses the target's *head* position, approximated as origin + eye height,
        which is what every aimbot actually aims at.
        """

        def _build():
            tt = self.session.ticks_for(target)
            if tt.size == 0 or self.ticks.size == 0:
                return self.view.t, np.full(len(self.view), np.nan)
            g = self.view.t
            # Interpolate both players' positions onto the grid.
            tx = np.interp(g, tt["t"], tt["px"])
            ty = np.interp(g, tt["t"], tt["py"])
            tz = np.interp(g, tt["t"], tt["pz"]) + EYE_HEIGHT_M
            sx = np.interp(g, self.ticks["t"], self.ticks["px"])
            sy = np.interp(g, self.ticks["t"], self.ticks["py"])
            sz = np.interp(g, self.ticks["t"], self.ticks["pz"]) + EYE_HEIGHT_M
            dx, dy, dz = tx - sx, ty - sy, tz - sz
            want_yaw = np.degrees(np.arctan2(dy, dx))
            horiz = np.hypot(dx, dy)
            want_pitch = -np.degrees(np.arctan2(dz, np.maximum(horiz, 1e-6)))
            err = angular_distance(self.view.pitch, self.view.yaw, want_pitch, want_yaw)
            return g, err

        return self.cached(f"tae_{target}", _build)


EYE_HEIGHT_M = 1.60  # standing eye height; ducking handled by adapters if needed


@runtime_checkable
class FeatureExtractor(Protocol):
    """Contract for a feature extractor.

    ``requires`` gates the extractor on adapter capabilities; ``names``
    declares the exact output keys so the feature matrix has a stable column
    order across processes and model versions.
    """

    name: str
    requires: tuple[Capability, ...]
    names: tuple[str, ...]
    scope: str  # "session" or "engagement"

    def extract(self, ctx: PlayerContext) -> dict[str, float]: ...


class SessionExtractor:
    """Base class for session-scope extractors."""

    name: str = "unnamed"
    requires: tuple[Capability, ...] = ()
    names: tuple[str, ...] = ()
    scope = "session"

    def extract(self, ctx: PlayerContext) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def empty(self) -> dict[str, float]:
        return {n: float("nan") for n in self.names}


class EngagementExtractor:
    """Base class for engagement-scope extractors."""

    name: str = "unnamed"
    requires: tuple[Capability, ...] = ()
    names: tuple[str, ...] = ()
    scope = "engagement"

    def extract_one(self, ctx: PlayerContext, eng: Engagement) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def empty(self) -> dict[str, float]:
        return {n: float("nan") for n in self.names}


# ---------------------------------------------------------------------------
# Engagement segmentation
# ---------------------------------------------------------------------------


def build_engagements(session: Session, slot: int, cfg: FeatureConfig) -> list[Engagement]:
    """Group a player's shots into engagements against a single target.

    Rules:
      * A shot belongs to an engagement if it is within ``gap`` seconds of the
        previous shot in that engagement.
      * The target is whoever the player damaged during the burst; if nobody
        was damaged we fall back to the crosshair raycast target, and failing
        that the engagement is dropped (spray-into-a-wall carries no signal).
    """
    shots = sorted(session.shots_for(slot), key=lambda s: s.t)
    if not shots:
        return []

    dmg_by_time = sorted((d for d in session.damages if d.attacker == slot), key=lambda d: d.t)
    kills = sorted(session.kills_by(slot), key=lambda k: k.t)
    vis = sorted((v for v in session.visibility if v.observer == slot), key=lambda v: v.t)

    gap = 0.60  # seconds of silence that ends a burst
    groups: list[list[ShotEvent]] = [[shots[0]]]
    for s in shots[1:]:
        if s.t - groups[-1][-1].t <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])

    ticks = session.ticks_for(slot)
    out: list[Engagement] = []
    for grp in groups:
        t0, t1 = grp[0].t, grp[-1].t
        dmgs = [d for d in dmg_by_time if t0 - 0.05 <= d.t <= t1 + 0.35]
        target = -1
        if dmgs:
            # Most-damaged victim in the burst.
            tally: dict[int, float] = {}
            for d in dmgs:
                tally[d.victim] = tally.get(d.victim, 0.0) + d.damage
            target = max(tally, key=tally.get)
        elif Capability.AIM_TARGET.value and ticks.size:
            sel = ticks[(ticks["t"] >= t0 - 0.1) & (ticks["t"] <= t1 + 0.1)]
            cand = sel["aim_target"][sel["aim_target"] >= 0]
            if cand.size:
                vals, counts = np.unique(cand, return_counts=True)
                target = int(vals[np.argmax(counts)])
        if target < 0:
            continue

        kill = next((k for k in kills if k.victim == target and t0 - 0.1 <= k.t <= t1 + 0.5), None)

        # Last visibility-open transition for this target before the burst.
        t_vis = float("-inf")
        for v in vis:
            if v.subject != target:
                continue
            if v.t > t1:
                break
            if v.visible:
                t_vis = v.t
        dist = dmgs[0].distance if dmgs else (kill.distance if kill else 0.0)
        out.append(
            Engagement(
                session_id=session.session_id,
                slot=slot,
                target=target,
                t_first_shot=t0,
                t_last_shot=t1,
                t_window_start=t0 - cfg.engagement_pre_s,
                t_window_end=t1 + cfg.engagement_post_s,
                shots=grp,
                damages=dmgs,
                kill=kill,
                t_target_visible=t_vis,
                weapon=grp[0].weapon,
                distance=float(dist),
            )
        )
    return out


def build_context(session: Session, slot: int, cfg: FeatureConfig, caps: Iterable[Capability]) -> PlayerContext:
    ticks = session.ticks_for(slot)
    view = ViewSignal(
        ticks["t"].astype(np.float64),
        ticks["pitch"].astype(np.float64),
        ticks["yaw"].astype(np.float64),
        cfg.resample_hz,
        cfg.sg_window,
        cfg.sg_poly,
    )
    engs = build_engagements(session, slot, cfg)
    return PlayerContext(
        session=session, slot=slot, cfg=cfg, view=view, ticks=ticks,
        engagements=engs, caps=set(caps),
    )


# ---------------------------------------------------------------------------
# Small numeric helpers shared by extractors
# ---------------------------------------------------------------------------


def safe_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Robust summary of a 1-D array. NaN-safe, empty-safe."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "p50", "p90", "max", "skew")}
    mean = float(x.mean())
    std = float(x.std())
    skew = float(((x - mean) ** 3).mean() / (std**3 + 1e-12)) if x.size > 2 else 0.0
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_p50": float(np.median(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_skew": skew,
    }


def stat_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{k}" for k in ("mean", "std", "p50", "p90", "max", "skew"))
