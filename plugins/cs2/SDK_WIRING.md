# SDK wiring checklist

Status (lab build): **GameFrame + roster + schema readers + game-event path are wired.**
Visibility stays off (`hdr.no_vis`) until TraceShape is reverse-engineered.

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
| eye | `CCSPlayerPawn` / `CCSPlayerPawnBase` | `m_angEyeAngles` | degrees |
| punch | `CCSPlayerPawn` | `m_aimPunchAngle` | accumulated |
| origin | `CBasePlayerPawn` | `m_vOldOrigin` | Hammer units |
| velocity | `CBaseEntity` | `m_vecAbsVelocity` | |
| health / armor | pawn | `m_iHealth` / `m_ArmorValue` | |
| shots fired | pawn | `m_iShotsFired` | |
| weapon | event name → defindex table | — | tick uses last fire |

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
