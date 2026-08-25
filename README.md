# Aegis

Adaptive, multi-game, learning anti-cheat. Built to beat the failure modes of VACnet-style supervised CNNs and Easy Anti-Cheat-style kernel-only approaches: novel cheat families, humanised aim assist, wallhacks that never aim for you, and the inability to expand cleanly past a single title.

## Why this is different

| Failure mode | Typical AC | Aegis |
|---|---|---|
| Novel humaniser / zero-day cheat | Blind until labelled | Self-supervised novelty AE + mechanism-based synthetic red-team |
| Wallhack without aimbot | Weak heuristics | Permutation test on occluded aim error vs time-shifted null |
| One freak lucky spray | Instant ban / instant miss | SPRT across engagements & sessions with correlation discount + half-life |
| FP budget is a vibe | Threshold on a raw score | Calibrated probs + split-conformal gate bound system-wide α |
| New game = rewrite | Per-title detector | Game adapters lower to one canonical schema; features/models/fusion stay |
| Ban you cannot defend | Black-box score | Causal features + structured explanations for review/appeals |
| Client-only or server-only | One side | Behavioural stack + attestation/HID integrity as complementary channels |

Shadow mode defaults **ON**. A fresh deployment cannot ban anyone until an operator opts in.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
aegis train --sessions 48 --duration 75 --out models
aegis evaluate --model-dir models --sessions 24
aegis simulate --aegis-capture --out data/sample_session
aegis ingest --model-dir models --capture-dir data/captures --port 9100
aegis explain --model-dir models
aegis serve --model-dir models --port 8080
pytest -q
```
## Architecture (one screen)

```
Game plugin / demo / generic JSON
        │
        ▼
   adapters/          → canonical Session (metres, seconds, degrees)
        │
        ▼
   features/          → session vector + engagement tensors (capability-gated)
        │
        ▼
   models/            → GBDT + temporal Transformer + novelty AE
                        → isotonic calibration + conformal gate
        │
        ▼
   fusion/            → trust prior + SPRT evidence + adjudication policy
        │
   integrity/         → attestation challenges + HID injection analysis
        │
        ▼
   service/ + cli     → verdicts (clear / watch / review / ban)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and [HANDOFF.md](HANDOFF.md) for depth, and [COMPARISON.md](COMPARISON.md) for an honest read on where this stands against VACnet/EAC/BattlEye/Ricochet/FairFight — verified evidence vs. argued hypothesis, kept separate on purpose.

## Multi-game path

1. Emit the generic FPS schema (`aegis.adapters.generic_fps`) — detection works immediately on whatever capabilities you actually send.
2. Graduate to a first-class adapter (CS2 is the reference: recoil tables, plugin wire protocol, demo lowering).
3. Do **not** retrain the world when you add visibility later — channels fuse independently; missing capabilities are omitted, never zero-filled.

Named presets for `apex_legends`, `warzone`, `tarkov`, and `battlefield` live in `aegis/adapters/presets.py` — the same on-ramp CS2 started from, registered so an integration is a config change, not a code change. None of those titles offer a legitimate server-plugin API the way CS2's MetaMod:Source2 does, so **these presets are architecturally ready and unverified against real telemetry from those games, not "Aegis supports X."** `tests/test_invariants.py::test_generic_presets_score_without_retraining` is the actual claim being made: the one model trained on the CS2-shaped synthetic simulator scores all of them without a retrain — proof that the architecture is game-agnostic, not proof that detection quality transfers, which cannot be known until a studio integration produces real telemetry to calibrate against.

## Honest limitations

Synthetic bootstrap metrics are for ranking detectors and proving the pipeline. Operating thresholds must be re-derived on real negatives before any verdict is issued. See the simulator docstring in `aegis/train/simulate.py`.

Current single-session synthetic numbers (`models/eval.json`, `aegis evaluate --stealth-curve`) show weak recall at the `review`/`ban` bands for most individual cheat families in isolation — by design: the SPRT fusion layer is built to accumulate evidence *across* matches, and a single session understates it (see the `notes` field in `train_report.json`). Do not read a strong AUC as a claim that Aegis outperforms VACnet or EAC today; no such head-to-head has been run, and the closed, proprietary nature of both makes one largely impossible to run honestly. What is verified: the CS2 telemetry pipeline itself, against a real dedicated server (`plugins/cs2/SDK_WIRING.md`), and the architecture's cross-title portability (above). What is not: comparative accuracy against any commercial anti-cheat.

## License

Proprietary — all rights reserved unless otherwise noted.
