"""Plugin registries.

Two extension points carry the multi-game story:

  * ``ADAPTERS``   -- lower a game's native telemetry into ``core.types``
  * ``EXTRACTORS`` -- produce named features from a canonical ``Session``

Adding a game means writing one adapter. Adding a detection idea means writing
one extractor. Neither requires touching the model, fusion or policy layers,
and extractors declare which capabilities they need so a game that cannot
supply (say) server-side visibility simply skips those features instead of
producing garbage.
"""

from __future__ import annotations

import enum
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


class Capability(enum.Enum):
    """What a game adapter is able to provide.

    Extractors gate on these. A title with no server-side occlusion queries
    still gets aim/movement/input detection; it just loses the wallhack
    channel, and fusion accounts for the missing modality rather than
    imputing zeros.
    """

    VIEW_ANGLES = "view_angles"  # per-tick pitch/yaw
    HIGH_RATE_VIEW = "high_rate_view"  # >= 64 Hz view sampling
    POSITIONS = "positions"
    VELOCITIES = "velocities"
    SHOTS = "shots"
    DAMAGE = "damage"
    HITGROUPS = "hitgroups"
    VISIBILITY = "visibility"  # server-authoritative LOS
    RECOIL_STATE = "recoil_state"  # authoritative recoil vector per shot
    MOVEMENT_INPUT = "movement_input"  # forward/side input axes
    BUTTONS = "buttons"
    HID_INPUT = "hid_input"  # raw mouse counts from client module
    AIM_TARGET = "aim_target"  # crosshair raycast result


class Registry(dict[str, T]):
    def register(self, key: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if key in self:
                raise KeyError(f"{key!r} already registered in {self._name}")
            self[key] = obj
            return obj

        return deco

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def get_or_raise(self, key: str) -> T:
        if key not in self:
            raise KeyError(f"unknown {self._name}: {key!r}. known: {sorted(self)}")
        return self[key]


ADAPTERS: Registry = Registry("adapter")
EXTRACTORS: Registry = Registry("extractor")


def extractors_for(caps: Iterable[Capability]) -> list:
    """All registered extractors whose requirements are satisfied by ``caps``."""
    have = set(caps)
    out = []
    for name in sorted(EXTRACTORS):
        ex = EXTRACTORS[name]
        if set(getattr(ex, "requires", ())) <= have:
            out.append(ex)
    return out
