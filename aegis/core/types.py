"""Canonical, game-agnostic telemetry types.

Every game adapter lowers its native events into these structures. The whole
feature/model stack downstream is written against *only* this schema, which is
what lets one trained detector generalise across CS2, other FPS titles, and
(with a different adapter) non-FPS games.

Storage is columnar (numpy structured arrays) rather than object-per-tick:
a single 40-minute CS2 match at 64 tick with 10 players is ~1.5M player-ticks,
and object overhead would dominate both memory and feature-extraction time.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Units contract (enforced by adapters, assumed everywhere downstream)
#   time      : seconds, float64, monotonic from session start
#   position  : metres, float32
#   velocity  : metres/second, float32
#   view      : degrees, float32; pitch in [-90, 90] (negative = up),
#               yaw in [-180, 180), increasing counter-clockwise
#   angles are ALWAYS wrapped by the adapter before entering the core
# --------------------------------------------------------------------------

DEG = np.float32


class PlayerFlag(enum.IntFlag):
    """Bit flags carried on every player tick."""

    NONE = 0
    ON_GROUND = 1 << 0
    DUCKING = 1 << 1
    SCOPED = 1 << 2
    RELOADING = 1 << 3
    DEFUSING = 1 << 4
    PLANTING = 1 << 5
    FLASHED = 1 << 6
    SMOKED = 1 << 7  # view path intersects a smoke volume
    ALIVE = 1 << 8
    WALKING = 1 << 9  # shift-walk
    AIRBORNE = 1 << 10
    LADDER = 1 << 11
    FROZEN = 1 << 12  # freezetime / round transition
    BLIND_PARTIAL = 1 << 13


class Team(enum.IntEnum):
    UNKNOWN = 0
    SPECTATOR = 1
    SIDE_A = 2  # CS2: T
    SIDE_B = 3  # CS2: CT


class HitGroup(enum.IntEnum):
    GENERIC = 0
    HEAD = 1
    CHEST = 2
    STOMACH = 3
    LEFT_ARM = 4
    RIGHT_ARM = 5
    LEFT_LEG = 6
    RIGHT_LEG = 7
    NECK = 8
    GEAR = 9


# ---------------------------------------------------------------------------
# Columnar per-tick player state
# ---------------------------------------------------------------------------

# One row per (player, tick). Kept deliberately narrow; anything rarer than
# per-tick lives in the sparse event lists below.
TICK_DTYPE = np.dtype(
    [
        ("t", "f8"),  # seconds from session start
        ("tick", "i4"),  # native server tick
        ("slot", "i2"),  # player slot within the session
        ("px", "f4"),
        ("py", "f4"),
        ("pz", "f4"),
        ("vx", "f4"),
        ("vy", "f4"),
        ("vz", "f4"),
        ("pitch", "f4"),
        ("yaw", "f4"),
        ("health", "i2"),
        ("armor", "i2"),
        ("weapon", "i2"),  # WeaponClass id from the adapter's table
        ("ammo", "i2"),
        ("flags", "i4"),
        ("shots_fired", "i2"),  # cumulative within the current spray
        ("aim_target", "i2"),  # slot the crosshair ray hits, -1 if none
        ("aim_target_vis", "i1"),  # 1 if that target is actually visible
        ("move_fwd", "f4"),  # normalised movement input, -1..1
        ("move_side", "f4"),
        ("buttons", "i4"),  # raw button bitmask (attack, jump, duck, use...)
    ]
)


@dataclass(slots=True)
class ShotEvent:
    t: float
    tick: int
    slot: int
    weapon: int
    pitch: float
    yaw: float
    # Recoil/spread state the *server* believes the weapon is in. Used to test
    # whether the player's counter-movement matches the true recoil vector.
    recoil_pitch: float = 0.0
    recoil_yaw: float = 0.0
    spread: float = 0.0
    is_first_of_spray: bool = False
    spray_index: int = 0


@dataclass(slots=True)
class DamageEvent:
    t: float
    tick: int
    attacker: int
    victim: int
    damage: float
    hitgroup: int
    weapon: int
    distance: float
    # Was the victim rendered visible to the attacker's client on this tick?
    victim_visible: bool = True


@dataclass(slots=True)
class KillEvent:
    t: float
    tick: int
    attacker: int
    victim: int
    weapon: int
    headshot: bool
    distance: float
    through_smoke: bool = False
    through_wall: bool = False
    attacker_flashed: bool = False
    noscope: bool = False
    # Milliseconds the victim had been *continuously visible* to the attacker
    # before the fatal shot. Negative => the victim was never visible.
    victim_exposure_ms: float = 0.0


@dataclass(slots=True)
class VisibilityEvent:
    """Server-authoritative occlusion transition.

    Emitted when the line of sight between two players opens or closes. This is
    the backbone of wallhack detection: a player who consistently reacts
    *before* the transition to visible has information they should not have.
    """

    t: float
    tick: int
    observer: int
    subject: int
    visible: bool
    # Angular distance from the observer's crosshair to the subject at the
    # moment of transition, in degrees.
    angle_error_deg: float = 0.0
    distance: float = 0.0


@dataclass(slots=True)
class InputSample:
    """Client-reported HID-level input, sampled above the tick rate.

    Optional (requires the Aegis client module). When present it is by far the
    highest-signal channel, because synthetic mouse input has a fundamentally
    different timing distribution than a real sensor. Treated as *untrusted*
    until the attestation layer vouches for the session.
    """

    t: float
    dx: int  # raw counts, not pixels
    dy: int
    buttons: int
    device_seq: int  # sequence number from the HID report
    hw_timestamp_us: int  # device-side timestamp when available


@dataclass(slots=True)
class RoundInfo:
    index: int
    t_start: float
    t_end: float
    t_freeze_end: float
    winner: int
    reason: int = 0


@dataclass(slots=True)
class PlayerInfo:
    slot: int
    player_id: str  # stable cross-session account id (hashed upstream)
    name: str = ""
    team: int = int(Team.UNKNOWN)
    is_bot: bool = False
    # Prior trust, supplied by the platform. Feeds the Bayesian prior, never
    # the feature vector -- we do not want the model learning "new account =
    # cheater", only the decision policy may use it.
    trust_prior: float = 0.5
    rank: float | None = None
    hours_played: float | None = None


@dataclass
class Session:
    """One match / server session, fully lowered to the canonical schema."""

    session_id: str
    game: str  # adapter key, e.g. "cs2"
    map_name: str = ""
    tickrate: float = 64.0
    players: dict[int, PlayerInfo] = field(default_factory=dict)
    ticks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=TICK_DTYPE))
    shots: list[ShotEvent] = field(default_factory=list)
    damages: list[DamageEvent] = field(default_factory=list)
    kills: list[KillEvent] = field(default_factory=list)
    visibility: list[VisibilityEvent] = field(default_factory=list)
    inputs: dict[int, list[InputSample]] = field(default_factory=dict)
    rounds: list[RoundInfo] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def slots(self) -> list[int]:
        return sorted(self.players)

    def ticks_for(self, slot: int) -> np.ndarray:
        """Per-player tick view, sorted by time.

        Cached because every feature extractor asks for it.
        """
        cache = self.meta.setdefault("_slot_index", {})
        if slot not in cache:
            sel = self.ticks[self.ticks["slot"] == slot]
            cache[slot] = sel[np.argsort(sel["t"], kind="stable")]
        return cache[slot]

    def invalidate_cache(self) -> None:
        self.meta.pop("_slot_index", None)

    def duration(self) -> float:
        if self.ticks.size == 0:
            return 0.0
        return float(self.ticks["t"].max() - self.ticks["t"].min())

    def shots_for(self, slot: int) -> list[ShotEvent]:
        return [s for s in self.shots if s.slot == slot]

    def kills_by(self, slot: int) -> list[KillEvent]:
        return [k for k in self.kills if k.attacker == slot]


# ---------------------------------------------------------------------------
# Angle helpers -- used everywhere, correctness matters more than elegance
# ---------------------------------------------------------------------------


def wrap_deg(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles into [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed difference a - b, wrapped."""
    return wrap_deg(a - b)


def angular_distance(pitch_a, yaw_a, pitch_b, yaw_b) -> np.ndarray:
    """Great-circle angle between two view directions, in degrees.

    Using the haversine form rather than a naive sqrt(dp^2 + dy^2) because the
    latter overestimates badly near the poles, and pitch near +-89 is exactly
    where spinbots and anti-aim live.
    """
    p1 = np.radians(np.asarray(pitch_a, dtype=np.float64))
    p2 = np.radians(np.asarray(pitch_b, dtype=np.float64))
    dy = np.radians(wrap_deg(np.asarray(yaw_a, dtype=np.float64) - np.asarray(yaw_b, dtype=np.float64)))
    dp = p2 - p1
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dy / 2) ** 2
    return np.degrees(2 * np.arcsin(np.clip(np.sqrt(h), 0.0, 1.0)))


def view_to_vector(pitch, yaw) -> np.ndarray:
    """Unit forward vector(s) from pitch/yaw in degrees. Shape (..., 3)."""
    p = np.radians(np.asarray(pitch, dtype=np.float64))
    y = np.radians(np.asarray(yaw, dtype=np.float64))
    cp = np.cos(p)
    return np.stack([cp * np.cos(y), cp * np.sin(y), -np.sin(p)], axis=-1)
