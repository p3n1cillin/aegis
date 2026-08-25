# Implementation Plan

## Overview

Run a prioritized, iterative improvement loop ("loop engineering") over the Aegis anti-cheat platform: each loop follows **improve → retrain → re-evaluate → regression-test → commit**, with the highest-impact items first — (A) model quality & scale-up including closing the diagnosed sparse-session GBDT insensitivity (HANDOFF.md entry 34's open item), (B) simulator realism so detection metrics measure something harder than today's cheat distribution, and (C) operational backlog items (audit log, signed model artefacts, per-title conformal gates). Work proceeds in repeated passes until time budget is exhausted; every pass must leave `pytest -q` green and end in a git commit so progress is never lost. Note honestly: an agentic session cannot guarantee a wall-clock 5 hours, so the plan is structured as self-contained loops that can be cut off at any boundary without leaving the tree broken.

Current baseline (verified this session): all 12 invariant tests pass in ~12s; latest recorded synthetic metrics AUC=0.880 / AP=0.757 / ECE=0.103 with known weak single-session recall for isolated cheat families; real sparse-data sessions collapse to near-constant GBDT output (mitigated by `coverage_shrink`, root cause still open).

## Types

- `aegis/models/ensemble.py`
  - Extend `Score.meta` usage (no schema change needed) to carry `coverage` and `tabular_used: bool`.
  - New module-level function `sparse_session_policy(n_engagements: int, coverage: float) -> dict` returning e.g. `{"use_tabular": bool, "shrink": float}` — single decision point for how a low-information session is scored.
- `aegis/train/simulate.py`
  - Add `"triggerbot"` and `"aim_smoothing"` (humanised aim-assist) to the family registry inside `sample_cheat(rng, family)` and `CheatConfig.label()`. No new top-level types required beyond config fields on `CheatConfig` (e.g. `trigger_reaction_ms`, `smooth_rate`).
  - New function `apply_network_jitter(tick_t: np.ndarray, rng, jitter_ms_range: tuple[float,float]) -> np.ndarray`.
- `aegis/integrity/audit.py` (new)
  - `@dataclass AuditRecord`: `ts`, `session_id`, `player_id`, `verdict`, `shadow: bool`, `p_fused`, `conformal_p`, `reasons: list[str]`, `model_digest`.
  - `class AuditLog`: `append(record)`, `tail(n)` — JSONL, append-only.
- `aegis/core/config.py`
  - Fields only: `PolicyConfig.audit_log_path = "audit/verdicts.jsonl"`, new `SecurityConfig(model_signature_key_path: str = "")`, `FusionConfig.per_title_alpha: dict[str, float]`.

## Files

New:
- `aegis/integrity/audit.py` — verdict audit log (backlog item 8).
- `tests/test_scaling_loops.py` — fast regression tests for everything added below (kept separate from the stable invariant suite).
- `lab/sweep_gbdt.py` — TabularGBDT hyperparameter sweep script writing `models/sweep_report.json`.
- `implementation_plan.md` — this document.

Modified:
- `aegis/models/ensemble.py` — sparse-session policy wired into `CalibratedEnsemble.score_player`; policy params persisted in artefact JSON.
- `aegis/train/simulate.py` — triggerbot + aim-smoothing families; network jitter option on `simulate_session(..., net_jitter_ms=None)`; wider human skill variance.
- `aegis/train/dataset.py` — include new families in the family mix (`SIM_CAPS` unchanged).
- `aegis/train/train.py` — record family list + generator version in `train_report.json`; accept expanded knobs.
- `aegis/train/evaluate.py` — verify per-family breakdown covers new families automatically; adjust only if hardcoded.
- `aegis/service/__init__.py` (+ `service/ingest.py`) — append an `AuditRecord` on every adjudication path including shadow verdicts; auth/containment untouched.
- `aegis/cli.py` — `aegis audit --tail N` subcommand; optional signing plumbing.
- `aegis/core/config.py` — fields listed under Types.
- `HANDOFF.md`, `README.md` — numbered loop-result entries per repo convention.

Out of scope: `plugins/cs2/`, `client/`, `lab/sdk/` (native/live work needing hardware/game servers).

## Functions

New:
- `sparse_session_policy(n_engagements, coverage) -> dict` in `aegis/models/ensemble.py` — decides whether the tabular channel participates at all for zero-engagement / ultra-sparse sessions (entry 34's option b) instead of asking one GBDT to handle every NaN pattern.
- `apply_network_jitter(tick_t, rng, jitter_ms_range)` in `aegis/train/simulate.py` — bounded random walk on tick timestamps, monotonic-preserving.
- `_simulate_triggerbot(...)`, `_apply_aim_smoothing(...)` helpers in `simulate.py` behind `sample_cheat` dispatch.
- `AuditLog.append/tail`, `make_audit_record(verdict, score, cfg)` in `aegis/integrity/audit.py`.

Modified:
- `CalibratedEnsemble.score_player` (`aegis/models/ensemble.py`) — call `sparse_session_policy`; when `use_tabular=False`, shrink toward prior rather than emitting a near-constant tabular probability; record `coverage` + `tabular_used` in `Score.meta`.
- `coverage_shrink` — unchanged semantics; composed by the policy function (regression test keeps its contract).
- `sample_cheat`, `CheatConfig.sample`, `CheatConfig.label`, `simulate_session` (`simulate.py`) — register new families and the jitter hook.
- `bootstrap_train` (`train.py`) — expanded session counts, pass-through GBDT params.
- Verdict construction sites in `service/__init__.py` and `service/ingest.py` — emit audit records.

Removed functions: none.

## Classes

New:
- `AuditLog`, `AuditRecord` (`aegis/integrity/audit.py`) — plain dataclasses, no inheritance, matching repo style.
- `SecurityConfig` (`aegis/core/config.py`) — dataclass embedded in `AegisConfig`, default-constructed in `load`/`save`.

Modified:
- `CheatConfig` (`simulate.py`) — new triggerbot/smoothing fields; `label()` updated.
- `AegisConfig` (`config.py`) — gains `security`, audit path, per-title alpha; `load`/`save` symmetric.
- `ModelBundle` / `CalibratedEnsemble` (`ensemble.py`) — carry sparse-policy constants in saved artefact JSON (backward-compatible defaults).

Removed classes: none.

## Dependencies

None added. Artefact signing (Loop 6) uses `cryptography.hazmat` already shipped with `cryptography>=42`; if Ed25519 is unavailable in the installed build, fall back to HMAC-SHA256 with a key file.

## Testing

All work validated by `.venv\Scripts\python.exe -m pytest -q` after every change (currently 12 passing, ~12s). New tests in `tests/test_scaling_loops.py`:

- `test_sparse_zero_engagement_session_not_scored_by_tabular` — zero-engagement session never produces a confident tabular posterior regardless of feature values (entry 34 regression).
- `test_sparse_policy_noop_at_full_capability` — full-capability aimbot score unaffected vs. baseline (guards against denting genuine detections; the mistake made once before per HANDOFF entry 35).
- `test_triggerbot_and_aim_smoothing_families_simulate_and_label` — valid sessions, distinct labels, `validate_session` clean.
- `test_network_jitter_preserves_monotonic_timebase`.
- `test_audit_log_records_shadow_verdicts` — one JSONL line per player verdict, readable via `tail`.
- `test_per_title_alpha_gate` — override changes conformal admission only for that title.
- Signing test (if Loop 6 lands): tampered artefact fails verification.

Retrain validation: `aegis train --sessions <N> --out models`, then `aegis evaluate --model-dir models --stealth-curve`; compare AUC/AP/ECE and per-family recall against `models/eval.json` baseline and record deltas. Multi-session false-ban check via `evaluate.multi_session_accumulation` after each retrain.

## Implementation Order

1. **Loop 1 — Sparse-session fix (entry 34 closure)**: `sparse_session_policy` in `ensemble.py`, wire into `score_player`, tests, pytest green, commit.
2. **Loop 2 — Scale-up train**: raise session count well past 220, run `lab/sweep_gbdt.py` sweep over GBDT knobs, pick best by held-out AP + per-family recall, retrain, evaluate, record delta, commit.
3. **Loop 3 — Simulator realism**: triggerbot + aim-smoothing families, network jitter, wider skill variance, dataset mix update, tests, retrain + evaluate against the harder simulator (report honestly, no metric spinning), commit.
4. **Loop 4 — Audit log**: `integrity/audit.py`, service wiring, `aegis audit --tail`, config field, tests, commit.
5. **Loop 5 — Per-title conformal gates**: `per_title_alpha`, gate lookup by title at scoring time, tests, commit.
6. **Loop 6 — Signed model artefacts** (time permitting): signature over artefact bytes, verification on `CalibratedEnsemble.load`, tamper test, commit.
7. **Loops 7+ — refine/repeat**: revisit whichever loop showed the weakest measured gain; further scale-up runs, more families, docs. Each iteration ends in a HANDOFF.md numbered entry + commit.

Guardrails honored throughout (HANDOFF "Invariants — do not break"): never zero-fill missing capabilities, never put trust into features, never auto-ban past conformal p ≥ α, never ban from a single engagement, keep `FeatureSpec.digest` glued to artefacts, shadow mode stays on, any new calibrator gets its own prior.

## Loop discipline ("5 hours")

Each iteration: pick next item → read affected code → implement smallest complete change → pytest green → train/evaluate where applicable → HANDOFF.md entry → `git commit`. Loops are atomic; the session stops cleanly at any loop boundary.
