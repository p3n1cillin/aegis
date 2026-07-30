"""CS2 plugin capture encode / decode helpers.

The live plugin speaks the frame schema in ``PLUGIN_PROTOCOL.md``. Files on
disk are concatenated msgpack maps (what ``CS2Adapter.parse_capture`` already
reads). The TCP path uses length-delimited frames with an optional zstd
envelope so a 128-tick 10-player stream stays cheap.

Encoding a ``Session`` back to plugin frames is lossy for a few fields the
simulator never carries in native form (exact CS2 button layout is recovered
via the reverse map). It is enough for:

  * round-trip tests of the adapter
  * generating synthetic ``.aegis`` captures before a real plugin exists
  * feeding the live ingest server in lab mode
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Iterable, Iterator

import msgpack
import numpy as np
import zstandard as zstd

from ...core.types import PlayerFlag, Session, Team
from .weapons import defindex_for_weapon

# Metres -> Hammer units (inverse of adapter.UNITS_TO_M).
M_TO_UNITS = 1.0 / 0.01905

# Canonical button bits -> CS2 IN_* bits (inverse of adapter._BTN_MAP).
_BTN_TO_NATIVE = {
    1 << 0: 1 << 0,
    1 << 1: 1 << 1,
    1 << 2: 1 << 2,
    1 << 3: 1 << 3,
    1 << 4: 1 << 4,
    1 << 5: 1 << 5,
    1 << 9: 1 << 9,
    1 << 10: 1 << 10,
}

# PlayerFlag bits that map onto the plugin's native ``f`` bitmask.
_FLAG_TO_NATIVE = (
    (PlayerFlag.ON_GROUND, 1 << 0),
    (PlayerFlag.DUCKING, 1 << 1),
    (PlayerFlag.SCOPED, 1 << 2),
    (PlayerFlag.RELOADING, 1 << 3),
    (PlayerFlag.DEFUSING, 1 << 4),
    (PlayerFlag.PLANTING, 1 << 5),
    (PlayerFlag.FLASHED, 1 << 6),
    (PlayerFlag.SMOKED, 1 << 7),
    (PlayerFlag.WALKING, 1 << 8),
    (PlayerFlag.LADDER, 1 << 9),
    (PlayerFlag.FROZEN, 1 << 10),
    (PlayerFlag.BLIND_PARTIAL, 1 << 11),
)


def _flags_to_native(flags: int) -> int:
    out = 0
    for pf, bit in _FLAG_TO_NATIVE:
        if flags & int(pf):
            out |= bit
    return out


def _buttons_to_native(buttons: int) -> int:
    out = 0
    for src, dst in _BTN_TO_NATIVE.items():
        if buttons & src:
            out |= dst
    return out


def _team_to_native(team: int) -> int:
    return {int(Team.SIDE_A): 2, int(Team.SIDE_B): 3, int(Team.SPECTATOR): 1}.get(int(team), 0)


def session_to_frames(session: Session) -> list[dict]:
    """Lower a canonical Session into plugin wire frames (Hammer units)."""
    frames: list[dict] = [
        {
            "k": "hdr",
            "session_id": session.session_id,
            "map": session.map_name,
            "tickrate": float(session.tickrate or 64.0),
            "server": str(session.meta.get("server", "synthetic")),
            "ver": str(session.meta.get("plugin_version", "synth-0")),
        }
    ]
    if session.meta.get("no_vis"):
        frames[0]["no_vis"] = True

    if session.players:
        frames.append({
            "k": "players",
            "players": [
                {
                    "slot": int(p.slot),
                    "pid": str(p.player_id),
                    "name": str(p.name),
                    "team": _team_to_native(p.team),
                    "bot": bool(p.is_bot),
                    "trust": float(p.trust_prior),
                    "rank": p.rank,
                    "hours": p.hours_played,
                }
                for p in sorted(session.players.values(), key=lambda x: x.slot)
            ],
        })

    # Group ticks by (t, tick) so one wire frame holds the whole lobby.
    if session.ticks.size:
        order = np.argsort(session.ticks, order=("t", "tick", "slot"))
        ticks = session.ticks[order]
        i = 0
        n = ticks.size
        while i < n:
            t = float(ticks[i]["t"])
            tick = int(ticks[i]["tick"])
            j = i + 1
            while j < n and float(ticks[j]["t"]) == t and int(ticks[j]["tick"]) == tick:
                j += 1
            players = []
            for row in ticks[i:j]:
                players.append({
                    "s": int(row["slot"]),
                    "x": float(row["px"]) * M_TO_UNITS,
                    "y": float(row["py"]) * M_TO_UNITS,
                    "z": float(row["pz"]) * M_TO_UNITS,
                    "vx": float(row["vx"]) * M_TO_UNITS,
                    "vy": float(row["vy"]) * M_TO_UNITS,
                    "vz": float(row["vz"]) * M_TO_UNITS,
                    "pi": float(row["pitch"]),
                    "ya": float(row["yaw"]),
                    "hp": int(row["health"]),
                    "ar": int(row["armor"]),
                    "w": defindex_for_weapon(int(row["weapon"])),
                    "am": int(row["ammo"]),
                    "f": _flags_to_native(int(row["flags"])),
                    "sf": int(row["shots_fired"]),
                    "at": int(row["aim_target"]),
                    "atv": int(row["aim_target_vis"]),
                    "mf": float(row["move_fwd"]),
                    "ms": float(row["move_side"]),
                    "b": _buttons_to_native(int(row["buttons"])),
                })
            frames.append({"k": "tick", "t": t, "tick": tick, "p": players})
            i = j

    for s in session.shots:
        frames.append({
            "k": "shot", "t": float(s.t), "tick": int(s.tick), "s": int(s.slot),
            "w": defindex_for_weapon(s.weapon),
            "pi": float(s.pitch), "ya": float(s.yaw),
            "rpi": float(s.recoil_pitch), "rya": float(s.recoil_yaw),
            "sp": float(s.spread),
        })

    for d in session.damages:
        frames.append({
            "k": "dmg", "t": float(d.t), "tick": int(d.tick),
            "a": int(d.attacker), "v": int(d.victim),
            "d": float(d.damage), "hg": int(d.hitgroup),
            "w": defindex_for_weapon(d.weapon),
            "dist": float(d.distance) * M_TO_UNITS,
            "vis": bool(d.victim_visible),
        })

    for k in session.kills:
        frames.append({
            "k": "kill", "t": float(k.t), "tick": int(k.tick),
            "a": int(k.attacker), "v": int(k.victim),
            "w": defindex_for_weapon(k.weapon),
            "hs": bool(k.headshot),
            "dist": float(k.distance) * M_TO_UNITS,
            "smoke": bool(k.through_smoke),
            "wall": bool(k.through_wall),
            "flash": bool(k.attacker_flashed),
            "noscope": bool(k.noscope),
            "exp_ms": float(k.victim_exposure_ms),
        })

    for v in session.visibility:
        frames.append({
            "k": "vis", "t": float(v.t), "tick": int(v.tick),
            "o": int(v.observer), "u": int(v.subject),
            "v": bool(v.visible),
            "ae": float(v.angle_error_deg),
            "dist": float(v.distance) * M_TO_UNITS,
        })

    for slot, samples in session.inputs.items():
        if not samples:
            continue
        frames.append({
            "k": "hid",
            "s": int(slot),
            "samples": [
                [float(s.t), int(s.dx), int(s.dy), int(s.buttons),
                 int(s.device_seq), int(s.hw_timestamp_us)]
                for s in samples
            ],
        })

    for r in session.rounds:
        frames.append({
            "k": "round",
            "i": int(r.index),
            "ts": float(r.t_start),
            "te": float(r.t_end),
            "tf": float(r.t_freeze_end),
            "w": int(r.winner),
            "r": int(r.reason),
        })

    if "attested" in session.meta:
        frames.append({
            "k": "attest",
            "ok": bool(session.meta.get("attested")),
            "detail": session.meta.get("attest_detail", {}),
        })

    return frames


def pack_frames(frames: Iterable[dict]) -> bytes:
    """Concatenate msgpack maps — the on-disk ``.aegis`` format."""
    out = io.BytesIO()
    for fr in frames:
        out.write(msgpack.packb(fr, use_bin_type=True))
    return out.getvalue()


def write_capture(path: str | Path, frames: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack_frames(frames))
    return path


def iter_capture(path: str | Path) -> Iterator[dict]:
    data = Path(path).read_bytes()
    yield from msgpack.Unpacker(io.BytesIO(data), raw=False, strict_map_key=False)


# ---------------------------------------------------------------------------
# Length-delimited TCP framing (PLUGIN_PROTOCOL.md)
# ---------------------------------------------------------------------------

_HDR = struct.Struct(">I")  # unsigned big-endian length


def encode_frame(frame: dict, *, compress: bool = False) -> bytes:
    payload = msgpack.packb(frame, use_bin_type=True)
    flags = 0
    if compress:
        payload = zstd.ZstdCompressor(level=1).compress(payload)
        flags = 1
    # wire: [flags:u8][len:u32][payload]
    return bytes([flags]) + _HDR.pack(len(payload)) + payload


def decode_frames_from_buffer(buf: bytearray) -> list[dict]:
    """Consume complete length-delimited frames from ``buf`` in-place."""
    out: list[dict] = []
    while True:
        if len(buf) < 5:
            break
        flags = buf[0]
        (n,) = _HDR.unpack_from(buf, 1)
        if len(buf) < 5 + n:
            break
        payload = bytes(buf[5:5 + n])
        del buf[:5 + n]
        if flags & 1:
            payload = zstd.ZstdDecompressor().decompress(payload)
        fr = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        if not isinstance(fr, dict):
            raise ValueError(f"expected map frame, got {type(fr)}")
        out.append(fr)
    return out


__all__ = [
    "session_to_frames",
    "pack_frames",
    "write_capture",
    "iter_capture",
    "encode_frame",
    "decode_frames_from_buffer",
    "M_TO_UNITS",
]
