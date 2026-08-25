# SDK wiring checklist

Status (lab build): **Position, view angle, health, weapon, shots-fired,
recoil/punch angle, native 64 Hz tick rate, and hurt/kill telemetry all
confirmed live against a real CS2 dedicated server** (bots, de_dust2; one
soak ran `events fire=5858 hurt=1467 kill=251` over ~50 min with sane moving
coordinates and real weapon defindices; separate runs confirmed real
accumulating recoil patterns and a measured 64.0 Hz sample rate —
`validate_session` returns zero problems on a live capture). Visibility
stays off (`hdr.no_vis`) until TraceShape is reverse-engineered.

## Origin/angle chain — the part that actually broke, and how it was found

`CBaseEntity` has **no** `m_pGameSceneNode` field on this engine build (that
was a guess from stale reference material — the actual live schema dump said
`off=-1`). The real, live-confirmed chain is two hops through the body
component, not one:

```
CBaseEntity::m_CBodyComponent   (CBodyComponent*)
  -> CBodyComponent::m_pSceneNode  (CGameSceneNode*, off=8)
       -> CGameSceneNode::m_vecAbsOrigin    (off=200)
       -> CGameSceneNode::m_angAbsRotation  (off=212)
```

`CGameSceneNode` itself is registered in the schema system's **global** type
scope, not `server.dll`'s module-local scope — `schema_offset.cpp::find_class`
now falls back to `ISchemaSystem::GlobalTypeScope()` when the module scope
misses, which is what makes this (and any other engine-level class) resolvable
at all.

Do not hand-guess a schema field name and ship it. `aegis::schema::dump_class`
prints a live class's real field names/offsets via `AEGIS_LOG_FILE` — use it
against a running server before trusting any new field lookup, the same way
this chain was actually derived.

Recoil/punch angle needed the identical treatment: `m_aimPunchAngle` does not
exist on `CCSPlayerPawn` on this build either. It moved into a services
sub-object, same pattern as the body component:

```
CCSPlayerPawn::m_pAimPunchServices        (CCSPlayer_AimPunchServices*)
  -> m_predictableBaseAngle    (client-reproducible recoil pattern)
  -> m_unpredictableBaseAngle  (small server-only correction)
```

The effective punch used for `shot.recoil_pitch/yaw` is the sum of both,
matching Valve's own view code. Live-verified: a deagle spray shows the
expected accumulate-then-reset shape (-0.1° → -0.6° → -1.4° → -2.3° → -3.1° →
-3.8°, next burst starts back near zero), and 221/299 shots in that capture
carry nonzero recoil — the remaining zeros are first-shots-after-a-pause,
where punch has fully decayed, which is the expected shape, not a bug.

## Tick rate — 16 Hz quietly broke tremor detection

`Hook_GameFrame` throttled emission to every 4th engine tick (~16 Hz on a
64-tick server) to "keep ingest light" (`config.h`/`aegis_mm.cpp`). Bandwidth
math for a server-side sidecar feed says otherwise: 10 players x ~15 fields x
4 bytes x 64 Hz is under 40 KB/s, trivial for any deployment topology this
runs in. Meanwhile physiological hand tremor sits at 4-12 Hz, and 16 Hz
sampling has a Nyquist ceiling of 8 Hz -- most of the tremor band was being
thrown away at the *plugin*, before Python ever saw a sample, so no amount of
downstream resampling could recover it. `SpectralFeatures` (the tremor
channel) requires `Capability.HIGH_RATE_VIEW`, and `CS2Adapter` claimed it
unconditionally regardless of what was actually emitted -- the channel looked
fine on full-rate synthetic training data and was quietly running on
insufficient real data.

Fixed on both sides:
- `config.h`/`aegis_mm.cpp`: default `tick_hz` raised to the engine's native
  rate; throttle divisor is now computed from `engine_hz / cfg.tick_hz`
  (override with `AEGIS_TICK_HZ` if an operator has a real reason to reduce
  it). Live-verified: `np.diff` on unique tick times in a fresh capture gives
  exactly `0.015625s` (64.0 Hz), where the old captures gave `0.0625s`
  (16.0 Hz).
- `aegis/adapters/base.py::effective_capabilities`: `HIGH_RATE_VIEW` is now
  derived from the session's *measured* sample density (median dt across
  unique tick times, >= 60 Hz), not trusted from the adapter's static
  declaration. Verified against both an old 16 Hz capture (capability
  correctly dropped) and a fresh 64 Hz one (capability correctly kept).

A red herring worth recording: `start-server-debug.bat` used to launch with
`+game_type 1 +game_mode 2 +mp_ignore_round_win_conditions 1` (arms race).
On this build that leaves every bot unspawned — all ten report the identical
falling coordinate from world origin, forever. That looked exactly like a
shared-pointer bug (identical values across distinct pawns) and cost real
debugging time before a pointer-identity check (each pawn/body/node address is
in fact distinct) and a same-config A/B against `start-lab.bat` isolated it as
a server config issue, not a plugin bug. The script now matches `start-lab.bat`
(`game_type 0 game_mode 0`).

## 1. Environment

```bat
set HL2SDKCS2=...\lab\sdk\hl2sdk-cs2
set MMSOURCE112=...\lab\sdk\metamod-source
set AEGIS_ACCOUNT_SALT=your-long-random-server-secret
lab\build-plugin.bat
lab\start-lab.bat
```

## 2. Schema fields (`sdk_bridge.cpp` / `schema_offset.cpp`)

Resolved at runtime via `ISchemaSystem` → `server.dll` type scope (no hardcoded offsets).

| Accessor | Schema class | Field | Notes |
|---|---|---|---|
| eye | `CCSPlayerPawn` / `CCSPlayerPawnBase` | `m_angEyeAngles` | degrees; live-confirmed |
| punch | `CCSPlayerPawn`→`CCSPlayer_AimPunchServices` | `m_pAimPunchServices`→`m_predictableBaseAngle`+`m_unpredictableBaseAngle` | live-confirmed real recoil pattern; `m_aimPunchAngle` does not exist on this build |
| origin | `CBaseEntity`→`CBodyComponent`→`CGameSceneNode` | `m_CBodyComponent`→`m_pSceneNode`→`m_vecAbsOrigin` | two-hop chain, live-confirmed; `m_vOldOrigin` does not exist on this build (kept as a harmless fallback) |
| abs angle | same chain | `m_angAbsRotation` | fallback when eye angles read zero |
| velocity | `CBaseEntity` | `m_vecAbsVelocity` | |
| health / armor | pawn | `m_iHealth` / `m_ArmorValue` | live-confirmed, hp drops on real damage |
| shots fired | pawn | `m_iShotsFired` | |
| weapon | event name → defindex table | — | tick uses last fire; live-confirmed real defindices |

## 3. Hooks (`aegis_mm.cpp`)

1. `SH_ADD_HOOK` `IServerGameDLL::GameFrame` → tick frames (~16 Hz).
2. `OnLevelInit` / `OnLevelShutdown` → `hdr` session.
3. `ClientActive` / `ClientDisconnect` / `OnClientConnected` → roster.
4. `CGameEventManager::LoadEventsFromFile` DVPHOOK → capture `IGameEventManager2*` (resolved via game `server.dll` address, not MetaMod's stub).
5. Listeners: `weapon_fire`, `player_hurt`, `player_death`, `player_spawn`, …

## 4. Visibility

- Still **disabled** — public SDK has no TraceShape.
- Plugin sets `hdr.no_vis=true`; Python adapter drops VISIBILITY capability.

## 5. Smoke without full SDK

```bat
cmake -S plugins/cs2 -B plugins/cs2/build-tools -DAEGIS_TOOLS_ONLY=ON
cmake --build plugins/cs2/build-tools --config Release
```
