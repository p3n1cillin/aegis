# SDK wiring checklist

Fill these in order against your `hl2sdk-cs2` + MetaMod:Source2 trees. Do not
guess offsets — use the schema system so Valve patches do not silently invert
recoil features.

## 1. Environment

```bat
git clone --branch cs2 https://github.com/alliedmodders/hl2sdk hl2sdk-cs2
git clone --recursive https://github.com/alliedmodders/metamod-source
set HL2SDKCS2=C:\path\to\hl2sdk-cs2
set MMSOURCE112=C:\path\to\metamod-source
set AEGIS_ACCOUNT_SALT=your-long-random-server-secret
```

```bat
cmake -S plugins/cs2 -B plugins/cs2/build -G "Visual Studio 17 2022" -A x64 ^
  -DAEGIS_WITH_METAMOD=ON
cmake --build plugins/cs2/build --config Release
```

## 2. Schema fields (`sdk_bridge.cpp`)

| Accessor | Schema class | Field | Notes |
|---|---|---|---|
| `read_eye_angles` | `CCSPlayerPawn` | `m_angEyeAngles` | degrees; negative pitch = up |
| `read_aim_punch` | `CCSPlayerPawn` | `m_aimPunchAngle` | **accumulated**, not delta |
| `read_origin` | `CBasePlayerPawn` / pawn | `m_vOldOrigin` or abs origin | Hammer units |
| `read_velocity` | pawn | `m_vecAbsVelocity` | units/s |
| `read_health` / armor | pawn | `m_iHealth` / `m_ArmorValue` | |
| `read_weapon_defindex` | active weapon | item def index | |
| `read_buttons` | pawn services | button state | map to `btn::*` |
| `read_shots_fired` | pawn | spray counter | |

## 3. Hooks (`aegis_mm.cpp`)

1. `SH_ADD_HOOK` `IServerGameDLL::GameFrame` → `Hook_GameFrame` → `runtime_.tick_from_sdk`.
2. `IGameEventManager2::AddListener` for `round_start`, `weapon_fire`, `player_hurt`, `player_death`, `player_connect_full`, `player_disconnect`.
3. On `weapon_fire`: call `sdk::snapshot_fire(pawn)` **before** any post-shot mutation, then `runtime_.on_weapon_fire(...)`.
4. On map start: run one TraceShape self-test; only then `sdk::mark_vis_ready(true)`.

## 4. Visibility

- Rate ≤ 16 Hz, pair debounce ≥ 100 ms (already in `VisWorker`).
- If TraceShape is unavailable: leave `vis_ready=false` so `hdr.no_vis=true`.
- **Never** synthesise LOS.

## 5. Smoke without full SDK

```bat
cmake -S plugins/cs2 -B plugins/cs2/build-tools -DAEGIS_TOOLS_ONLY=ON
cmake --build plugins/cs2/build-tools --config Release
```

Then against a running Python ingest:

```bat
aegis ingest --port 9100 --capture-dir data/captures
aegis simulate --aegis-capture --out data/sample_session
```
