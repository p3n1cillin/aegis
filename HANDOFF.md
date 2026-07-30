# Handoff

## Goal

Sophisticated, learning, multi-game anti-cheat starting with CS2 (beat VACnet Live failure modes) and expandable past EAC-style single-channel designs.

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
| Docs (README, ARCHITECTURE, HANDOFF) | Done |
| Real CS2 plugin binary / client module | Not in-repo (protocol specified) |
| Training on real labelled matches | Not yet — synthetic bootstrap only |
| Production deployment / non-shadow bans | Explicitly gated off |

## How to run what exists

```bash
pip install -e .
aegis train --sessions 48 --duration 75 --out models
aegis evaluate --model-dir models --stealth-curve
aegis serve --model-dir models
```

Artefacts land in `models/`: `ensemble.json`, `tabular.*`, `temporal.pt`, `novelty.pt`, calibrators, `conformal.npz`, `train_report.json`, `eval.json`, optional `dataset/`.

## What Claude completed before handoff

1. Canonical `Session` / columnar ticks / event types / angle helpers
2. Config knobs for features, models, fusion, policy, integrity
3. Plugin registries (`ADAPTERS`, `EXTRACTORS`, `Capability`)
4. Full feature layer with causal / statistical rationale in module docstrings
5. CS2 adapter + plugin wire protocol + weapon table
6. Generic FPS adapter
7. Physically-grounded simulator (submovement human aim, mechanism-based cheats)

## What this handoff completed

8. Model stack + calibrated ensemble
9. Bayesian SPRT fusion + adjudication + explanations
10. Attestation + HID analysis
11. Dataset / train / eval pipeline
12. CLI + HTTP API
13. README / ARCHITECTURE / HANDOFF

## Next work (priority order)

1. **Fit on real negatives** from a CS2 server plugin capture before touching ban thresholds.
2. Implement / integrate the Source2 plugin per `PLUGIN_PROTOCOL.md` (vis at 16 Hz, authoritative recoil, salted account ids).
3. Client attestation module (separate binary) speaking the challenge protocol in `integrity/attest.py`.
4. Online learning loop: human review labels → calibration refresh → novelty fine-tune on new cleans.
5. Per-title conformal gates (do not share α operating points across games).
6. Expand simulator maps / network jitter once real telemetry shows the residual error modes.
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
