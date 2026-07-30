# Aegis architecture

## Design thesis

VACnet-class systems learn a mapping from labelled engagement snippets to a cheat score. That works until the cheat distribution shifts, and it never explains itself. EAC-class systems focus on client integrity; they miss purely informational cheats and anything that looks like a human at the input layer.

Aegis splits the problem into **channels that fail independently** and fuses them with an explicit error budget:

1. **Causal behavioural features** (motor control, spectral tremor, recoil compensation lag, reaction-time ex-Gaussian, wallhack permutation test, movement, HID dynamics).
2. **Supervised models** on those features (session GBDT + engagement temporal encoder).
3. **Open-set novelty** (autoencoder trained only on clean).
4. **Sequential evidence** (SPRT with half-life and within-session correlation discount).
5. **Conformal gate** so auto-ban is inadmissible unless the clean-null p-value beats system α.
6. **Integrity** (attestation + HID) as a separate deterministic path.

## Package map

| Package | Responsibility |
|---|---|
| `aegis/core` | Canonical types, units contract, config, registries, timebase |
| `aegis/adapters` | Game → `Session`. CS2 plugin/demo + generic FPS JSON |
| `aegis/features` | Capability-gated extractors + pipeline (`FeatureSpec` digest) |
| `aegis/models` | Tabular GBDT, temporal encoder, novelty AE, calibration, ensemble |
| `aegis/fusion` | Trust prior, SPRT, adjudication policy, explainability |
| `aegis/integrity` | Challenge-response attestation, HID injection scoring |
| `aegis/train` | Synthetic humans/cheats, dataset, train, evaluate |
| `aegis/service` | FastAPI sidecar |
| `aegis/cli.py` | `aegis train \| evaluate \| simulate \| serve \| explain` |

## Units contract (non-negotiable)

Adapters convert **before** anything else sees the data:

- time: seconds, session-relative, monotonic per player
- position: metres
- velocity: m/s
- view: degrees; pitch ∈ [-90, 90] (negative = up); yaw wrapped to [-180, 180)

`validate_session` catches the checkable violations. A units bug looks like cheating to every layer above.

## Feature → model → fusion data flow

```
Session
  ├─ extract_session_features → float vector aligned to FeatureSpec.columns
  │     └─ TabularGBDT → raw p → isotonic → p_tabular
  │     └─ NoveltyAE (clean-trained) → novelty score → p_novelty
  └─ extract_engagement_tensors → (C,T) per duel
        └─ TemporalEncoder → raw p → isotonic → p_eng[]
              └─ soft-OR / corr-discount → p_temporal

CalibratedEnsemble.score_player
  → p_fused, llr_session, engagement_llrs, conformal_p

SPRT.observe_session(engagement_llrs, session_llr, novelty_llr)
  → EvidenceState.log_odds, n_units

Adjudicator.decide(posterior, conformal_p, integrity, shadow_mode)
  → clear | watch | review | ban | integrity_ban
```

Loading a model with a mismatched `FeatureSpec.digest` is a hard error.

## Capability gating

Extractors declare `requires: set[Capability]`. Adapters declare what they can honestly provide. Demo paths drop `VISIBILITY` and `HID_INPUT` rather than synthesising them. Fusion simply receives fewer modalities.

## CS2 production ingest

See `aegis/adapters/cs2/PLUGIN_PROTOCOL.md`. Live plugin feeds are required for the wallhack channel (occlusion transitions). Demos are for backfill/eval only.

## Policy defaults

```python
watch_threshold = 0.60
review_threshold = 0.90
ban_threshold = 0.995
min_engagements_for_ban = 40
shadow_mode = True
fusion.alpha = 1e-5   # system-wide P(ban | innocent) target
```

Auto-ban also requires SPRT upper boundary, minimum effective units, and conformal admission.

## Extending to another game

1. Start with `GenericFPSAdapter` JSON.
2. When you need weapon tables / native demos / engine flags, copy the CS2 adapter pattern.
3. Register with `@ADAPTERS.register("your_game")`.
4. Run `validate_session` in CI.
5. Recalibrate conformal + policy thresholds on that title's real negatives — do not reuse CS2 operating points blindly.
