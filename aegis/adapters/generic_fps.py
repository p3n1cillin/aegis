"""Generic FPS adapter — the multi-game on-ramp.

This exists to prove and to enforce the claim that the detection stack is
game-agnostic. A new title integrates by emitting JSON/msgpack frames in the
canonical schema below; no Aegis code changes at all. Once a title outgrows
the generic path (needs custom weapon tables, engine-specific flags, a native
demo format) it graduates to its own adapter subclass, and the CS2 adapter is
the worked example of what that looks like.

The schema here is deliberately the canonical one from ``core.types`` rather
than a compressed wire format, because the point is legibility for an
integrating studio, not bytes on the wire.

Declared capabilities are computed *from the data actually present*, not
assumed. A title that ships positions and view angles but no occlusion queries
gets aim, spectral, movement and outcome detection immediately, and can add
the wallhack channel later by emitting `visibility` events — no retraining of
the other channels required, because each channel is fused independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.registry import ADAPTERS, Capability
from ..core.types import (
    DamageEvent,
    KillEvent,
    PlayerInfo,
    Session,
    ShotEvent,
    Team,
    VisibilityEvent,
    wrap_deg,
)
from .base import GameAdapter

# Every capability the generic path *can* express. The adapter narrows this
# per-session based on what the payload actually contains.
_ALL = frozenset({
    Capability.VIEW_ANGLES, Capability.HIGH_RATE_VIEW, Capability.POSITIONS,
    Capability.VELOCITIES, Capability.SHOTS, Capability.DAMAGE,
    Capability.HITGROUPS, Capability.VISIBILITY, Capability.RECOIL_STATE,
    Capability.MOVEMENT_INPUT, Capability.BUTTONS, Capability.AIM_TARGET,
})


@ADAPTERS.register("generic_fps")
class GenericFPSAdapter(GameAdapter):
    key = "generic_fps"
    display_name = "Generic FPS"
    unit_scale = 1.0  # caller supplies metres
    max_run_speed = 6.0
    capabilities = _ALL

    def __init__(self, game_key: str = "generic_fps", unit_scale: float = 1.0,
                 max_run_speed: float = 6.0, pitch_up_is_positive: bool = False):
        self.key = game_key
        self.unit_scale = unit_scale
        self.max_run_speed = max_run_speed
        # Half the engines in existence disagree about this sign. Getting it
        # wrong flips every pitch feature, so it is an explicit constructor
        # argument rather than something to discover in production.
        self.pitch_sign = -1.0 if pitch_up_is_positive else 1.0

    def parse(self, source: Any) -> Session:
        if isinstance(source, (str, Path)):
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        elif isinstance(source, (bytes, bytearray)):
            payload = json.loads(source.decode("utf-8"))
        else:
            payload = source
        return self._lower(payload)

    def _lower(self, d: dict) -> Session:
        s = self.new_session(
            str(d["session_id"]),
            map_name=str(d.get("map", "")),
            tickrate=float(d.get("tickrate", 60.0)),
        )
        us = self.unit_scale

        for p in d.get("players", []):
            s.players[int(p["slot"])] = PlayerInfo(
                slot=int(p["slot"]),
                player_id=str(p["player_id"]),
                name=str(p.get("name", "")),
                team=int(p.get("team", Team.UNKNOWN)),
                is_bot=bool(p.get("is_bot", False)),
                trust_prior=float(p.get("trust_prior", 0.5)),
            )

        rows = []
        for r in d.get("ticks", []):
            rows.append({
                "t": float(r["t"]),
                "tick": int(r.get("tick", 0)),
                "slot": int(r["slot"]),
                "px": float(r.get("x", 0.0)) * us,
                "py": float(r.get("y", 0.0)) * us,
                "pz": float(r.get("z", 0.0)) * us,
                "vx": float(r.get("vx", 0.0)) * us,
                "vy": float(r.get("vy", 0.0)) * us,
                "vz": float(r.get("vz", 0.0)) * us,
                "pitch": float(np.clip(self.pitch_sign * float(r.get("pitch", 0.0)), -90.0, 90.0)),
                "yaw": float(wrap_deg(r.get("yaw", 0.0))),
                "health": int(r.get("health", 100)),
                "armor": int(r.get("armor", 0)),
                "weapon": int(r.get("weapon", 0)),
                "ammo": int(r.get("ammo", 0)),
                "flags": int(r.get("flags", 0)),
                "shots_fired": int(r.get("shots_fired", 0)),
                "aim_target": int(r.get("aim_target", -1)),
                "aim_target_vis": int(r.get("aim_target_vis", 0)),
                "move_fwd": float(r.get("move_fwd", 0.0)),
                "move_side": float(r.get("move_side", 0.0)),
                "buttons": int(r.get("buttons", 0)),
            })
        s.ticks = self.pack_ticks(rows)

        for e in d.get("shots", []):
            s.shots.append(ShotEvent(
                t=float(e["t"]), tick=int(e.get("tick", 0)), slot=int(e["slot"]),
                weapon=int(e.get("weapon", 0)),
                pitch=self.pitch_sign * float(e.get("pitch", 0.0)),
                yaw=float(wrap_deg(e.get("yaw", 0.0))),
                recoil_pitch=self.pitch_sign * float(e.get("recoil_pitch", 0.0)),
                recoil_yaw=float(e.get("recoil_yaw", 0.0)),
                spread=float(e.get("spread", 0.0)),
                is_first_of_spray=bool(e.get("first_of_spray", False)),
                spray_index=int(e.get("spray_index", 0)),
            ))
        for e in d.get("damages", []):
            s.damages.append(DamageEvent(
                t=float(e["t"]), tick=int(e.get("tick", 0)),
                attacker=int(e["attacker"]), victim=int(e["victim"]),
                damage=float(e.get("damage", 0.0)), hitgroup=int(e.get("hitgroup", 0)),
                weapon=int(e.get("weapon", 0)), distance=float(e.get("distance", 0.0)) * us,
                victim_visible=bool(e.get("victim_visible", True)),
            ))
        for e in d.get("kills", []):
            s.kills.append(KillEvent(
                t=float(e["t"]), tick=int(e.get("tick", 0)),
                attacker=int(e["attacker"]), victim=int(e["victim"]),
                weapon=int(e.get("weapon", 0)), headshot=bool(e.get("headshot", False)),
                distance=float(e.get("distance", 0.0)) * us,
                through_smoke=bool(e.get("through_smoke", False)),
                through_wall=bool(e.get("through_wall", False)),
                attacker_flashed=bool(e.get("attacker_flashed", False)),
                victim_exposure_ms=float(e.get("victim_exposure_ms", 0.0)),
            ))
        for e in d.get("visibility", []):
            s.visibility.append(VisibilityEvent(
                t=float(e["t"]), tick=int(e.get("tick", 0)),
                observer=int(e["observer"]), subject=int(e["subject"]),
                visible=bool(e["visible"]),
                angle_error_deg=float(e.get("angle_error_deg", 0.0)),
                distance=float(e.get("distance", 0.0)) * us,
            ))

        s.meta["attested"] = bool(d.get("attested", False))
        s.meta["capabilities"] = sorted(c.value for c in self.session_capabilities(s))
        s.invalidate_cache()
        return s

    def session_capabilities(self, s: Session) -> frozenset[Capability]:
        """Capabilities actually backed by data in this session.

        Declaring a capability the data does not support is the most common
        integration bug, and it produces NaN-filled features that the model
        cannot distinguish from a quiet player. So we derive rather than trust.
        """
        caps = {Capability.VIEW_ANGLES, Capability.POSITIONS}
        t = s.ticks
        if t.size == 0:
            return frozenset()
        if s.tickrate >= 60.0:
            caps.add(Capability.HIGH_RATE_VIEW)
        if np.any(np.hypot(t["vx"], t["vy"]) > 0):
            caps.add(Capability.VELOCITIES)
        if s.shots:
            caps.add(Capability.SHOTS)
            if any(sh.recoil_pitch or sh.recoil_yaw for sh in s.shots):
                caps.add(Capability.RECOIL_STATE)
        if s.damages:
            caps.add(Capability.DAMAGE)
            if any(d.hitgroup for d in s.damages):
                caps.add(Capability.HITGROUPS)
        if s.visibility:
            caps.add(Capability.VISIBILITY)
        if np.any(t["buttons"] != 0):
            caps.add(Capability.BUTTONS)
        if np.any(t["move_fwd"] != 0) or np.any(t["move_side"] != 0):
            caps.add(Capability.MOVEMENT_INPUT)
        if np.any(t["aim_target"] >= 0):
            caps.add(Capability.AIM_TARGET)
        if s.inputs and s.meta.get("attested"):
            caps.add(Capability.HID_INPUT)
        return frozenset(caps)


def example_payload() -> dict:
    """Minimal well-formed payload, used by tests and by integration docs."""
    return {
        "session_id": "example-1",
        "map": "example_map",
        "tickrate": 64.0,
        "players": [
            {"slot": 0, "player_id": "hash_a", "team": 2},
            {"slot": 1, "player_id": "hash_b", "team": 3},
        ],
        "ticks": [
            {"t": 0.0, "slot": 0, "x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "flags": 256},
            {"t": 0.0, "slot": 1, "x": 10, "y": 0, "z": 0, "pitch": 0, "yaw": 180, "flags": 256},
        ],
        "shots": [], "damages": [], "kills": [], "visibility": [],
    }
