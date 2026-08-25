# Aegis CS2 Source2 plugin

Server-side MetaMod:Source2 plugin that streams the wire protocol in
[`../../aegis/adapters/cs2/PLUGIN_PROTOCOL.md`](../../aegis/adapters/cs2/PLUGIN_PROTOCOL.md)
to `aegis ingest`.

This is what makes Aegis beat demo-only VACnet Live paths: **server-authoritative
recoil**, **occlusion transitions**, and **salted account ids**.

## Layout

```
plugins/cs2/
  README.md
  CMakeLists.txt
  configure.py              # thin wrapper → AMBuild/CMake with SDK paths
  addons/metamod/aegis.vdf  # metamod loader stub
  src/
    aegis_mm.h / .cpp       # ISmmPlugin entry, hooks
    encode.h / .cpp         # msgpack frame builders (no SDK dep)
    net_client.h / .cpp     # async TCP + length-delimited framing
    vis_worker.h / .cpp     # 16 Hz LOS worker (SDK hooks stubbed)
    salt.h / .cpp           # account-id salting
    config.h                # ingest host/port, feature flags
  tools/
    encode_smoke.cpp        # builds WITHOUT hl2sdk — protocol self-test
```

## Build (when SDKs are present)

```bat
set HL2SDKCS2=C:\path\to\hl2sdk-cs2
set MMSOURCE112=C:\path\to\metamod-source
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DHL2SDKCS2=%HL2SDKCS2% -DMMSOURCE112=%MMSOURCE112%
cmake --build build --config Release
```

Without SDKs you can still compile the protocol encoder smoke test:

```bat
cmake -S . -B build-tools -DAEGIS_TOOLS_ONLY=ON
cmake --build build-tools
build-tools\encode_smoke.exe
```

## Runtime

1. Install MetaMod:Source2 on the CS2 dedicated server.
2. Drop `aegis.dll` / `aegis.so` under `csgo/addons/aegis/` and the `.vdf` under `addons/metamod/`.
3. Set `aegis_ingest_host` / `aegis_ingest_port` (default `127.0.0.1:9100`).
4. Run `aegis ingest --port 9100 --capture-dir data/captures` on the adjudication host.

## Implementation status

| Module | Status |
|---|---|
| msgpack encode + TCP framing | Done |
| Net client (non-blocking queue) | Done |
| Account-id salt (HMAC-SHA256 on Windows) | Done |
| SDK bridge + MetaMod hooks | **Live** — GameFrame, roster, weapon_fire / hurt / death |
| Schema field readers | **Verified live** against a real dedicated server — position/angle/health/weapon/recoil all confirmed sane, `validate_session` returns zero problems on a live capture (see SDK_WIRING.md) |
| Tick rate | **Native 64 Hz, verified** — was silently throttled to 16 Hz (below the Nyquist floor for tremor detection); measured 64.0 Hz exactly on a fresh capture |
| Vis worker | Ready; gated off (`hdr.no_vis`) until TraceShape exists |
| Weapon fire / punch angle | **Live and verified** — real accumulating recoil pattern confirmed against a live spray |
| Full MetaMod load | Needs `HL2SDKCS2` + `MMSOURCE112` (lab scripts set these) |

See [SDK_WIRING.md](SDK_WIRING.md) for the exact field/hook checklist.

Do **not** invent visibility. If traces are unavailable, set `hdr.no_vis = true`.
