"""CS2 adapter.

Two ingest paths, both producing the same canonical ``Session``:

  1. **Live plugin feed** (production path). A Source2 server plugin streams
     msgpack frames over a socket. This is what a deployment actually uses:
     it gives per-tick view angles at full server rate, server-authoritative
     recoil state, and -- the part no demo can provide -- occlusion queries,
     which are what makes the wallhack permutation test possible. The frame
     schema lives in ``aegis/adapters/cs2/PLUGIN_PROTOCOL.md``.

  2. **Demo replay** (backfill / offline evaluation). CS2 demos are Source2
     protobuf with entity delta encoding. Rather than reimplement that here,
     ``parse_demo`` accepts any callable that yields decoded frames, so an
     existing parser (demoparser2, demoinfocs-golang via a pipe) can be
     plugged in. What this module owns is the *lowering* -- unit conversion,
     angle convention, occlusion reconstruction -- which is the part that has
     to be identical between the two paths or the model sees two different
     worlds.

Coordinate conventions handled here, once:
  * Hammer units -> metres (1 unit = 0.01905 m).
  * CS2 pitch is already negative-is-up; yaw is already [-180, 180).
  * Velocities arrive in units/s.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import msgpack
import numpy as np

from ...core.registry import ADAPTERS, Capability
from ...core.types import (
    DamageEvent,
    InputSample,
    KillEvent,
    PlayerFlag,
    PlayerInfo,
    RoundInfo,
    Session,
    ShotEvent,
    Team,
    VisibilityEvent,
    wrap_deg,
)
from ..base import GameAdapter
from .weapons import WEAPONS, weapon_by_defindex, weapon_by_name

UNITS_TO_M = 0.01905
MAX_RUN_SPEED_M_S = 250.0 * UNITS_TO_M  # 4.7625

# CS2 button bitmask -> canonical bitmask used by movement features.
_BTN_MAP = {
    1 << 0: 1 << 0,   # IN_ATTACK
    1 << 1: 1 << 1,   # IN_JUMP
    1 << 2: 1 << 2,   # IN_DUCK
    1 << 3: 1 << 3,   # IN_FORWARD
    1 << 4: 1 << 4,   # IN_BACK
    1 << 5: 1 << 5,   # IN_USE
    1 << 9: 1 << 9,   # IN_MOVELEFT
    1 << 10: 1 << 10,  # IN_MOVERIGHT
}


def _map_flags(native: int, alive: bool) -> int:
    """CS2 player flags/pflags -> canonical PlayerFlag bits."""
    f = PlayerFlag.NONE
    if alive:
        f |= PlayerFlag.ALIVE
    if native & (1 << 0):
        f |= PlayerFlag.ON_GROUND
    else:
        f |= PlayerFlag.AIRBORNE
    if native & (1 << 1):
        f |= PlayerFlag.DUCKING
    if native & (1 << 2):
        f |= PlayerFlag.SCOPED
    if native & (1 << 3):
        f |= PlayerFlag.RELOADING
    if native & (1 << 4):
        f |= PlayerFlag.DEFUSING
    if native & (1 << 5):
        f |= PlayerFlag.PLANTING
    if native & (1 << 6):
        f |= PlayerFlag.FLASHED
    if native & (1 << 7):
        f |= PlayerFlag.SMOKED
    if native & (1 << 8):
        f |= PlayerFlag.WALKING
    if native & (1 << 9):
        f |= PlayerFlag.LADDER
    if native & (1 << 10):
        f |= PlayerFlag.FROZEN
    if native & (1 << 11):
        f |= PlayerFlag.BLIND_PARTIAL
    return int(f)


@ADAPTERS.register("cs2")
class CS2Adapter(GameAdapter):
    key = "cs2"
    display_name = "Counter-Strike 2"
    unit_scale = UNITS_TO_M
    max_run_speed = MAX_RUN_SPEED_M_S
    capabilities = frozenset({
        Capability.VIEW_ANGLES,
        Capability.HIGH_RATE_VIEW,
        Capability.POSITIONS,
        Capability.VELOCITIES,
        Capability.SHOTS,
        Capability.DAMAGE,
        Capability.HITGROUPS,
        Capability.VISIBILITY,
        Capability.RECOIL_STATE,
        Capability.MOVEMENT_INPUT,
        Capability.BUTTONS,
        Capability.AIM_TARGET,
        Capability.HID_INPUT,
    })

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def parse(self, source: Any) -> Session:
        """Dispatch on source type.

        ``str``/``Path`` ending in .aegis -> plugin capture file
        ``str``/``Path`` ending in .dem   -> demo (requires a frame provider)
        iterable of dicts                 -> already-decoded plugin frames
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.suffix == ".dem":
                raise NotImplementedError(
                    "CS2 demo parsing needs a frame provider; call "
                    "parse_demo(path, provider=...) with a decoder such as "
                    "demoparser2. See module docstring."
                )
            return self.parse_capture(p)
        return self.parse_frames(source)

    def parse_capture(self, path: str | Path) -> Session:
        """Read a length-prefixed msgpack capture written by the plugin."""
        data = Path(path).read_bytes()
        return self.parse_frames(_iter_msgpack(io.BytesIO(data)))

    def parse_demo(self, path: str | Path, provider: Callable[[Path], Iterable[dict]]) -> Session:
        """Lower a demo using an external frame ``provider``.

        The provider must yield the same frame dicts documented in
        PLUGIN_PROTOCOL.md. Demo sources cannot supply ``vis`` frames, so the
        resulting session will lack VISIBILITY and the wallhack channel is
        skipped rather than faked -- see the capability gating in
        ``features.pipeline``.
        """
        return self.parse_frames(provider(Path(path)))

    def demo_capabilities(self) -> frozenset[Capability]:
        """Capabilities available when the source is a demo rather than live."""
        return self.capabilities - {Capability.VISIBILITY, Capability.HID_INPUT}

    # ------------------------------------------------------------------
    # Frame lowering
    # ------------------------------------------------------------------

    def parse_frames(self, frames: Iterable[dict]) -> Session:
        session: Session | None = None
        tick_rows: list[dict] = []
        t0: float | None = None
        tickrate = 64.0
        # Per-slot spray tracking, needed because the plugin reports absolute
        # recoil state but not spray boundaries.
        last_shot_t: dict[int, float] = {}
        spray_idx: dict[int, int] = {}

        for fr in frames:
            kind = fr.get("k")

            if kind == "hdr":
                tickrate = float(fr.get("tickrate", 64.0))
                session = self.new_session(
                    str(fr["session_id"]),
                    map_name=str(fr.get("map", "")),
                    tickrate=tickrate,
                )
                session.meta["server"] = fr.get("server", "")
                session.meta["plugin_version"] = fr.get("ver", "")
                continue

            if session is None:
                # Tolerate a stream that starts mid-flight.
                session = self.new_session(str(fr.get("session_id", "unknown")), tickrate=tickrate)

            if kind == "players":
                for p in fr["players"]:
                    session.players[int(p["slot"])] = PlayerInfo(
                        slot=int(p["slot"]),
                        player_id=str(p["pid"]),
                        name=str(p.get("name", "")),
                        team=_map_team(p.get("team", 0)),
                        is_bot=bool(p.get("bot", False)),
                        trust_prior=float(p.get("trust", 0.5)),
                        rank=p.get("rank"),
                        hours_played=p.get("hours"),
                    )

            elif kind == "tick":
                t = float(fr["t"])
                if t0 is None:
                    t0 = t
                rel = t - t0
                for p in fr["p"]:
                    alive = bool(p.get("hp", 0) > 0)
                    tick_rows.append({
                        "t": rel,
                        "tick": int(fr.get("tick", 0)),
                        "slot": int(p["s"]),
                        "px": float(p["x"]) * UNITS_TO_M,
                        "py": float(p["y"]) * UNITS_TO_M,
                        "pz": float(p["z"]) * UNITS_TO_M,
                        "vx": float(p.get("vx", 0.0)) * UNITS_TO_M,
                        "vy": float(p.get("vy", 0.0)) * UNITS_TO_M,
                        "vz": float(p.get("vz", 0.0)) * UNITS_TO_M,
                        "pitch": float(np.clip(p["pi"], -90.0, 90.0)),
                        "yaw": float(wrap_deg(p["ya"])),
                        "health": int(p.get("hp", 0)),
                        "armor": int(p.get("ar", 0)),
                        "weapon": weapon_by_defindex(int(p.get("w", 0))).wid,
                        "ammo": int(p.get("am", 0)),
                        "flags": _map_flags(int(p.get("f", 0)), alive),
                        "shots_fired": int(p.get("sf", 0)),
                        "aim_target": int(p.get("at", -1)),
                        "aim_target_vis": int(p.get("atv", 0)),
                        "move_fwd": float(p.get("mf", 0.0)),
                        "move_side": float(p.get("ms", 0.0)),
                        "buttons": _map_buttons(int(p.get("b", 0))),
                    })

            elif kind == "shot":
                t = float(fr["t"]) - (t0 or 0.0)
                slot = int(fr["s"])
                prev = last_shot_t.get(slot, -99.0)
                first = (t - prev) > 0.25
                spray_idx[slot] = 0 if first else spray_idx.get(slot, 0) + 1
                last_shot_t[slot] = t
                session.shots.append(ShotEvent(
                    t=t, tick=int(fr.get("tick", 0)), slot=slot,
                    weapon=weapon_by_defindex(int(fr.get("w", 0))).wid,
                    pitch=float(fr.get("pi", 0.0)), yaw=float(wrap_deg(fr.get("ya", 0.0))),
                    recoil_pitch=float(fr.get("rpi", 0.0)),
                    recoil_yaw=float(fr.get("rya", 0.0)),
                    spread=float(fr.get("sp", 0.0)),
                    is_first_of_spray=first,
                    spray_index=spray_idx[slot],
                ))

            elif kind == "dmg":
                session.damages.append(DamageEvent(
                    t=float(fr["t"]) - (t0 or 0.0), tick=int(fr.get("tick", 0)),
                    attacker=int(fr["a"]), victim=int(fr["v"]),
                    damage=float(fr.get("d", 0.0)), hitgroup=int(fr.get("hg", 0)),
                    weapon=weapon_by_defindex(int(fr.get("w", 0))).wid,
                    distance=float(fr.get("dist", 0.0)) * UNITS_TO_M,
                    victim_visible=bool(fr.get("vis", True)),
                ))

            elif kind == "kill":
                session.kills.append(KillEvent(
                    t=float(fr["t"]) - (t0 or 0.0), tick=int(fr.get("tick", 0)),
                    attacker=int(fr["a"]), victim=int(fr["v"]),
                    weapon=weapon_by_defindex(int(fr.get("w", 0))).wid,
                    headshot=bool(fr.get("hs", False)),
                    distance=float(fr.get("dist", 0.0)) * UNITS_TO_M,
                    through_smoke=bool(fr.get("smoke", False)),
                    through_wall=bool(fr.get("wall", False)),
                    attacker_flashed=bool(fr.get("flash", False)),
                    noscope=bool(fr.get("noscope", False)),
                    victim_exposure_ms=float(fr.get("exp_ms", 0.0)),
                ))

            elif kind == "vis":
                session.visibility.append(VisibilityEvent(
                    t=float(fr["t"]) - (t0 or 0.0), tick=int(fr.get("tick", 0)),
                    observer=int(fr["o"]), subject=int(fr["u"]),
                    visible=bool(fr["v"]),
                    angle_error_deg=float(fr.get("ae", 0.0)),
                    distance=float(fr.get("dist", 0.0)) * UNITS_TO_M,
                ))

            elif kind == "hid":
                slot = int(fr["s"])
                lst = session.inputs.setdefault(slot, [])
                for s in fr["samples"]:
                    lst.append(InputSample(
                        t=float(s[0]) - (t0 or 0.0), dx=int(s[1]), dy=int(s[2]),
                        buttons=int(s[3]), device_seq=int(s[4]), hw_timestamp_us=int(s[5]),
                    ))

            elif kind == "round":
                session.rounds.append(RoundInfo(
                    index=int(fr.get("i", len(session.rounds))),
                    t_start=float(fr.get("ts", 0.0)) - (t0 or 0.0),
                    t_end=float(fr.get("te", 0.0)) - (t0 or 0.0),
                    t_freeze_end=float(fr.get("tf", 0.0)) - (t0 or 0.0),
                    winner=int(fr.get("w", 0)),
                    reason=int(fr.get("r", 0)),
                ))

            elif kind == "attest":
                # Set by the integrity layer; features that read client data
                # refuse to run unless this is true.
                session.meta["attested"] = bool(fr.get("ok", False))
                session.meta["attest_detail"] = fr.get("detail", {})

        if session is None:
            raise ValueError("empty frame stream")
        session.ticks = self.pack_ticks(tick_rows)
        session.invalidate_cache()
        return session


def _map_team(native: int) -> int:
    return {2: int(Team.SIDE_A), 3: int(Team.SIDE_B), 1: int(Team.SPECTATOR)}.get(int(native), int(Team.UNKNOWN))


def _map_buttons(native: int) -> int:
    out = 0
    for src, dst in _BTN_MAP.items():
        if native & src:
            out |= dst
    return out


def _iter_msgpack(fh) -> Iterator[dict]:
    unpacker = msgpack.Unpacker(fh, raw=False, strict_map_key=False)
    yield from unpacker


__all__ = ["CS2Adapter", "WEAPONS", "weapon_by_name", "weapon_by_defindex", "UNITS_TO_M"]
