# Aegis attestation client (reference implementation)

Speaks the challenge-response protocol `aegis/integrity/attest.py` implements
server-side. This is what HANDOFF.md's backlog called "client attestation
module (separate binary) speaking `integrity/attest.py`" — that item is done
as a reference implementation, not a production client.

## What this proves, verified live against a running `aegis serve`

- **Real challenge-response round trip**: `POST /v1/attest/issue` → run the
  exact probes the server asked for → `HMAC-SHA256(device_key, nonce ||
  measurement)` → `POST /v1/attest/verify`. Verified: `ok=1 attested=1
  untrusted=0 confirmed_tamper=0`.
- **Real tamper detection**: a deliberately wrong MAC is rejected by the
  server with `confirmed_tamper=True, untrusted=True` — the security property
  actually holds, checked by sending a bad response, not just by reading the
  code.
- **Real probes, not stubs**: `module_hash` is a genuine SHA-256 of this
  client's own executable (BCrypt, the same primitive
  `plugins/cs2/src/salt.cpp` already uses server-side); `input_stack` is a
  genuine `IsDebuggerPresent`/`CheckRemoteDebuggerPresent` check.

## What this does not prove, stated plainly

- **`module_hash` does not detect in-memory-only tampering** after launch —
  only that the file on disk matches at hash time. A real deployment needs
  either periodic re-verification with a server-held baseline hash (not
  implemented — this client sends the hash, but nothing here validates it
  against a known-good baseline; that registry doesn't exist yet) or a
  kernel-level integrity channel, which is explicitly the category of
  approach this project's own architecture argues has other blind spots
  (see `ARCHITECTURE.md`, `COMPARISON.md`) — not a reason to skip file
  integrity, but a reason not to oversell it as sufficient alone.
- **`input_stack` is a single, well-known, easily-defeated signal.**
  `IsDebuggerPresent` is trivially bypassed by anyone who knows to look for
  it. It is real and functioning, not decoration, but reading a `true` here
  as "definitely not tampered with" would be wrong. A production version
  needs more probes (hooked-API detection, unexpected loaded modules,
  timing-based anti-debug) — this reference implementation exists to prove
  the *protocol* works end-to-end, not to be a complete anti-tamper suite.
- **The device key is a lab-only shared secret.** Defaults to
  `SHA-256("aegis-dev-key")` — byte-identical to `Attestor`'s own fallback
  server-side — purely so the protocol is testable without standing up key
  provisioning. Set `AEGIS_DEVICE_KEY` to override. A real deployment needs
  a per-device secret provisioned through a secure, authenticated channel at
  install/first-launch time (an HSM-backed issuance flow, not a shared
  passphrase) — this client does not implement that, and says so rather than
  pretending a shared dev key is production-ready.

## Build

```bat
cmake -S client -B client\build -G "Visual Studio 18 2026" -A x64
cmake --build client\build --config Release
```

No third-party dependencies — WinHTTP and BCrypt are both native Windows
APIs. JSON is a small hand-written encoder/extractor for the known, flat
`AttestIssueRequest`/`AttestVerifyRequest` schema in
`aegis/service/__init__.py`, not a general-purpose parser; see
`src/json_lite.h` for why that scope is deliberate.

## Run

```bat
:: against a local aegis serve instance (aegis serve --model-dir models)
client\build\Release\aegis_attest_client.exe --host 127.0.0.1 --port 8080 ^
  --session-id my-session --player-id my-player

:: continuous, matching how a real client would poll
client\build\Release\aegis_attest_client.exe --loop --interval 30
```

Set `AEGIS_API_KEY` in the environment if the target server has
`AEGIS_API_KEY` configured (see `aegis/cli.py`'s `serve` command) — the
client sends it as `X-Aegis-Api-Key` automatically when set.

## Layout

```
client/
  CMakeLists.txt
  src/
    main.cpp         entry point, the issue -> probe -> verify loop
    http_client.h/cpp WinHTTP POST wrapper
    crypto.h/cpp      BCrypt HMAC-SHA256 + SHA-256 (file and buffer)
    probes.h/cpp      module_hash, input_stack -- see limitations above
    json_lite.h/cpp   minimal encoder/extractor for the known wire schema
```
