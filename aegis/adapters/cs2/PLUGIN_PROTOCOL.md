# Aegis CS2 plugin wire protocol

The server-side plugin streams length-delimited **msgpack maps** over a TCP
socket (or writes them to a `.aegis` capture file, same encoding). Every frame
is a map with a `k` key naming the frame kind. Keys are short because a 10-slot
128-tick server emits ~1280 player states per second and the framing cost
matters.

All positions and velocities are in **Hammer units** and unit-per-second; the
adapter converts to metres. All angles are degrees in CS2's native convention
(pitch negative = up, yaw wrapped). Timestamps `t` are float seconds from
server start; the adapter rebases them to session-relative.

## Frames

### `hdr` — once, first
| key | type | meaning |
|---|---|---|
| `session_id` | str | unique match id |
| `map` | str | map name |
| `tickrate` | float | server tickrate |
| `server` | str | server identifier |
| `ver` | str | plugin version |

### `players` — on join/leave/team change
`players`: list of `{slot, pid, name, team, bot, trust, rank, hours}`.
`pid` must be a **salted hash** of the account id, not the raw id — the
detection stack never needs to know who someone is.

### `tick` — every server tick
| key | meaning |
|---|---|
| `t`, `tick` | time, tick number |
| `p` | list of per-player states |

Per-player state keys: `s` slot, `x`/`y`/`z` origin, `vx`/`vy`/`vz` velocity,
`pi`/`ya` pitch/yaw, `hp`, `ar` armor, `w` item definition index, `am` ammo,
`f` flag bitmask, `sf` shots fired in current spray, `at` crosshair raycast
target slot (`-1` none), `atv` whether that target is visible, `mf`/`ms`
forward/side movement input in `[-1, 1]`, `b` button bitmask.

Flag bits (`f`): 0 on-ground, 1 ducking, 2 scoped, 3 reloading, 4 defusing,
5 planting, 6 flashed, 7 view-through-smoke, 8 walking, 9 ladder,
10 freezetime, 11 partially blind.

> **Send the view angle the server used for the shot**, not the post-shot
> angle. Getting this backwards inverts the recoil-compensation lag feature.

### `shot` — every bullet
`t`, `tick`, `s` slot, `w` defindex, `pi`/`ya` view at fire, `rpi`/`rya` the
**server's authoritative accumulated recoil vector** at that shot, `sp` spread.

The recoil vector is the single most valuable thing the plugin provides and it
cannot be recovered from a demo. Report `m_aimPunchAngle` (accumulated), not
the per-shot delta.

### `dmg`
`t`, `tick`, `a` attacker, `v` victim, `d` damage, `hg` hitgroup, `w`, `dist`,
`vis` whether the victim was visible to the attacker at the moment of impact.

### `kill`
`t`, `tick`, `a`, `v`, `w`, `hs`, `dist`, `smoke`, `wall`, `flash`, `noscope`,
`exp_ms` — milliseconds the victim had been *continuously visible* to the
attacker before the fatal shot. `-1` if never visible.

### `vis` — occlusion transitions
`t`, `tick`, `o` observer, `u` subject, `v` visible bool, `ae` angular error
from observer crosshair to subject, `dist`.

Emit a transition whenever the line of sight between any live pair opens or
closes. Rate-limit to at most one transition per pair per 100 ms to avoid
flicker at cover edges — the permutation test in `features/visibility.py`
assumes transitions are real, not jitter.

**Cost note:** a naive all-pairs trace is 90 traces/tick. Do it at 16 Hz on a
worker thread with a bounding-volume early-out; the detector resamples anyway.

### `hid` — client input batches (requires the Aegis client module)
`s` slot, `samples`: list of `[t, dx, dy, buttons, device_seq, hw_timestamp_us]`.

Only forwarded when the client is attested. See `aegis/integrity/`.

### `round`
`i` index, `ts` start, `te` end, `tf` freezetime end, `w` winner, `r` reason.

### `attest`
`ok` bool, `detail` map. Emitted by the integrity service, not the game server.
Gates every client-sourced feature.

## Implementation notes for the plugin author

* Buffer and flush on a background thread. Never block the game tick on I/O.
* Compress with zstd at level 1 for the socket path; captures are already
  small enough that compression is optional.
* Ship a monotonic `seq` on the socket path so the ingest service can detect
  dropped frames; a gap invalidates the affected window rather than silently
  producing a fake "player stopped moving" interval.
* If the plugin cannot supply `vis` frames, say so in `hdr` (`"no_vis": true`)
  and the ingest service will drop the VISIBILITY capability for that session.
  Do **not** emit synthetic visibility.
