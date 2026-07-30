# Handoff

## Goal

Sophisticated, learning, multi-game anti-cheat starting with CS2 (beat VACnet Live failure modes) and expandable past EAC-style single-channel designs.

Repo: https://github.com/p3nicillin/aegis

## Status

| Area | Status |
|---|---|
| Core schema, config, registries, timebase | Done |
| Feature extractors (aim, spectral, recoil, reaction, visibility, movement, input, outcome) | Done |
| Adapters: base, CS2 (+ weapons + plugin protocol), generic FPS | Done |
| Synthetic human + cheat simulator | Done |
| Models: GBDT, temporal encoder, novelty AE, calibration, conformal, ensemble | Done |
| Fusion: trust prior, SPRT, policy, explainability | Done |
| Integrity: attestation server + HID analyzer | Done |
| Train / evaluate / CLI / FastAPI service | Done |
| Plugin capture encode + TCP ingest (`aegis ingest`) | Done |
| Online learning refresh from review labels | Done |
| Invariant regression tests | Done |
| Real CS2 plugin binary / client module | Scaffold + SDK bridge (`plugins/cs2/`) — schema/hook fill-ins left |
| Training on real labelled matches | Not yet — synthetic bootstrap only |
| Production deployment / non-shadow bans | Explicitly gated off |

## How to run what exists

```bash
pip install -e ".[dev]"   # or: pip install -e . && pip install pytest
aegis train --sessions 48 --duration 75 --out models
aegis evaluate --model-dir models --stealth-curve
aegis simulate --aegis-capture --out data/sample_session
aegis ingest --model-dir models --capture-dir data/captures --port 9100
aegis serve --model-dir models
pytest -q
```

Artefacts land in `models/`: `ensemble.json`, `tabular.*`, `temporal.pt`, `novelty.pt`, calibrators, `conformal.npz`, `train_report.json`, `eval.json`, optional `dataset/`.

## What landed after the initial handoff

14. Novelty AE robust to all-NaN capability columns
15. Session ↔ plugin-frame capture encode (`adapters/cs2/capture.py`)
16. Live TCP ingest with seq-gap handling + capture write (`service/ingest.py`)
17. Online calibrator / novelty refresh from review JSONL (`train/online.py`)
18. CLI: `ingest`, `online-refresh`, `simulate --aegis-capture`
19. Invariant tests + CS2 plugin scaffold (encode/net/salt/vis stub + MetaMod entry)
20. `plugins/cs2` CMake tools build (`encode_smoke`) without hl2sdk

## Next work (priority order)

1. **Fill schema + hooks** per `plugins/cs2/SDK_WIRING.md` (eye angles, punch, TraceShape, GameFrame / game events).
2. **Fit on real negatives** from plugin captures before touching ban thresholds.
3. Client attestation module (separate binary) speaking `integrity/attest.py`.
4. Review UI / queue that appends `ReviewLabel` rows and calls `aegis online-refresh`.
5. Per-title conformal gates (do not share α operating points across games).
6. Expand simulator maps / network jitter once real telemetry shows residual error modes.
7. Hardening: auth on the API, signed model artefacts, audit log for every issued verdict.

## Invariants — do not break

- Never zero-fill a missing capability.
- Never put trust/rank into the feature vector (prior only).
- Never auto-ban when conformal p ≥ α, even if posterior looks huge.
- Never ban from a single engagement; SPRT + `min_units_ban` + `min_engagements_for_ban`.
- Keep `FeatureSpec.digest` glued to every model artefact.
- Shadow mode on until an explicit operator decision.

## Key files to read first

- `aegis/core/types.py` — schema
- `aegis/features/aim.py` + `visibility.py` — detection thesis
- `aegis/train/simulate.py` — how labels are born
- `aegis/models/ensemble.py` — serve-time scoring
- `aegis/fusion/sprt.py` + `policy.py` — how bans are (not) issued
- `aegis/adapters/cs2/PLUGIN_PROTOCOL.md` — production ingest
- `aegis/service/ingest.py` — live capture path
- `plugins/cs2/README.md` — plugin implementation notes
