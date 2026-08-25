# Where Aegis actually stands vs. commercial anti-cheat

This document exists because "make it better than VACnet / EAC / BattlEye /
Ricochet / FairFight" is the standing goal for this project, and that claim
needs to be checked against what has actually been verified rather than
asserted. Read this alongside `HANDOFF.md` (what's built) and
`plugins/cs2/SDK_WIRING.md` (what's been proven live).

## The most important finding in this document is a bug, not a benchmark

Trying to produce real comparative evidence — multi-session SPRT
accumulation instead of the single-session numbers below, because that is
the system's actual designed operating mode — surfaced a critical,
previously-undetected false-positive bug: **93% of clean synthetic
identities were falsely banned within 20 simulated matches.** Not noisy
edge cases — 14 of 15. That is the finding that matters most here, and it is
reported before anything favourable-looking because a system that eventually
bans everyone is not "better than VACnet or EAC," it is unusable, and
burying that under the more flattering numbers elsewhere in this document
would defeat the entire point of writing it.

**Root cause**: `ModelBundle.prior` (the assumed population cheat rate used
to convert a calibrated probability into a log-likelihood ratio) is computed
from the *session*-level cheat rate in training data (`train.y.mean()`,
capped at 0.2) and was being reused for *both* the tabular channel (correct
— that is the population it was calibrated against) and the temporal
engagement channel (wrong — `cal_temporal` is fit against the engagement-
level cheat rate, `cal.eng_y.mean()`, which is a different and, empirically,
notably higher number: 0.356 vs 0.200 in the run that surfaced this, because
cheaters generate more/longer engagements per session than clean players
do). Using the wrong prior gave every engagement-level LLR a systematic
positive offset — confirmed live: mean calibrated P(cheat) for known-clean
engagements sat at ~0.30 against the 0.20 prior actually used. `SPRT.observe`
discounts *within*-session correlation (`n**0.65`) but has no mechanism for a
bias that repeats identically every session, so it does not fade — it
accumulates, and with 5-13 such units per match, it inevitably crosses any
finite ban boundary given enough matches, for every player, innocent or not.

**Fix**: `ModelBundle.eng_prior`, a separate constant derived from
`cal.eng_y.mean()` at train time, used only for the engagement-LLR
conversion (`aegis/models/ensemble.py`). `test_engagement_llr_not_
systematically_positive_for_clean` is the fast regression test.

The first pass at measuring this used raw `sprt.crossed_ban()` as a proxy for
"banned," which **skips two real safety gates** the production path actually
has: `min_engagements_for_ban` and, more importantly, the conformal gate
(`score.conformal_p < alpha`). That proxy overstated both the false-ban rate
and the true-ban rate. Corrected numbers, run through the actual
`Adjudicator.decide()` path, same 15-cheater/15-clean/20-match setup:

| Metric | Before fix (raw SPRT proxy) | After fix (raw SPRT proxy) | After fix (full policy path) |
|---|---|---|---|
| Clean identities falsely flagged within 20 matches | 100% | 0% | 0% |
| Clean identities falsely **banned** within 20 matches | 93.3% | 0% | 0% |
| Wallhack identities flagged within 20 matches | — | 93.3% | 100% |
| Wallhack identities **banned** within 20 matches | 100% | 66.7% | **0%** |
| Median matches to flag a wallhacker for review | — | 7.5 | 7.5 |
| Median matches to auto-ban a wallhacker | 13 | 16 | no ban occurred |

The corrected picture: **the conformal gate is not dead weight** — of the 15
wallhack identities, 10 reached a raw SPRT ban-boundary crossing where
`conformal_p >= alpha`, and the gate correctly downgraded all 10 to REVIEW
instead of BAN (`conformal_ever_blocked_a_raw_ban_cheat=10` in the report
dict). Combined with `min_engagements_for_ban`, the full policy never
auto-banned a single wallhack identity in this run — it flagged every one of
them for human review instead, which is the conservative behaviour a
1e-5-alpha "1 in 100k" false-ban budget is supposed to produce when the
calibration set is small and synthetic. That is a defensible design decision,
not obviously a flaw — but it means the earlier "66.7% banned" framing in
this document was wrong and has been corrected here rather than left to
stand.

Read the false-ban numbers and the ban-rate numbers with different weight:
the false-ban corrections (100%→0%, 93.3%→0%) are the important, credible
result of a real bug fix. The 0% true-ban rate is not evidence the system
doesn't work — it is evidence that 20 matches of synthetic wallhack behaviour
plus a deliberately strict alpha is not enough to clear an intentionally high
bar, which single-session numbers already implied and this now confirms
end-to-end.

This was found and fixed with synthetic data — the same caveat as everywhere
else in this document applies: it says nothing about real players. What it
does establish is that the claim "SPRT accumulation lets alpha be a real
operating guarantee, not a vibe" (`fusion/sprt.py`'s own docstring) was not
actually true before the `eng_prior` fix, for any population, synthetic or
real — and that the conformal gate, once the false-ban bug stopped masking
its effect, is measurably doing the job it was built for.

## Tested against real human data — a second real bug, found but only partly fixed

Everything above this point is synthetic. To get past that, this session
used a small, legitimately public research dataset — a "demonstration"
subset of [yviler/cs2-cheat-detection](https://github.com/yviler/cs2-cheat-detection),
33 real per-tick CS2 kill-window CSVs (~4.7s each, real 64 Hz server ticks)
from two distinct, independently-labelled real Steam accounts (one flagged
cheater, one legit). This is *not* a live deployment and *not* a VACnet
head-to-head — VACnet's verdict on these accounts isn't observable — but it
is real human aim data, which nothing else in this document is.

**What happened**: every one of the 33 real segments, cheater and legit
alike, scored `p_tabular ≈ 1.0000` — a saturated, completely
non-discriminating "cheat" score regardless of the true label.

**First diagnosis (confirmed, and fixed)**: the real session only has 4 of
12 simulator capabilities (view angles, positions, shots, high-rate view —
no visibility, recoil, damage, hitgroups, movement input, buttons, or
aim_target, because a single-player kill-window CSV cannot honestly provide
them). Every synthetic training session, without exception, had all 12.
That exact sparse profile had never once appeared in training — confirmed by
checking `SIM_CAPS` in `aegis/train/dataset.py`, which was a single hardcoded
constant used for every session. Fixed with `sample_session_caps` (same
file): sessions now drop capabilities in correlated blocks (not independent
per-capability coin flips — the first attempt at this used independent
dropout, made the fully-minimal profile a `0.3**8 ≈ 0.00007` event, and
measurably did not fix anything on a retrain; the corrected version
explicitly weights the minimal profile so it actually appears in training).
**Confirmed fixed for synthetic data**: a fresh minimal-capability synthetic
session now scores a sane, varied, non-saturated range (0.09-0.60 across
clean/aimbot/wallhack probes) instead of saturating.

**First fix confirmed insufficient on its own**: after retraining, all 33 real
segments *still* scored `p_tabular ≈ 1.0000`.

**Second diagnosis (confirmed, and partially fixed)**: the real segments are
~4.7s kill-window clips (300 ticks); every synthetic training session was a
full 45-90s match. Simulating a *synthetic* session at the real clip's exact
duration reproduced the same collapse — three different labelled families
(clean, aimbot, wallhack) landed on the identical raw score using the
already-capability-fixed model, proving duration was a second, independent
missing axis, not a symptom of the first. Added `sample_session_duration`
(same file): a fraction of training sessions now simulate a short 3-15s clip
instead of a full match. **Partially confirmed on synthetic data**: raw
tabular scores for fresh short synthetic sessions now vary meaningfully
(0.36-0.85 across clean/aimbot/wallhack/triggerbot/bhop probes, versus one
repeated constant before) — real progress. But most of that range collapsed
onto a single calibrated output (five of seven probes calibrated to exactly
the same value) — the isotonic calibrator's resolution in this
still-undersampled input region is coarse, a third, narrower issue sitting
on top of the first two.

**Still not fixed after both fixes**: the real cheater and real legit segment
produced a **bit-for-bit identical raw score** (0.8805) — not merely the
same calibration bucket, the same leaf output across the whole GBDT, for two
genuinely different real accounts.

**Placeholder-opponent artefact, conclusively ruled out.** One candidate
explanation was that the test harness's hand-built session gives slot 1 zero
ticks, a pattern the simulator never produces. Tested directly: rebuilt the
same real cheater session with slot 1 populated by real, non-zero,
independently-varying tick data from *eight different other real accounts*
(cheater-vs-legit, legit-vs-legit as a control). The raw score was identical
to 15 decimal places in every single case, including the original
zero-tick placeholder. The opponent's data — real or absent — has zero
effect on slot 0's score. Whatever is happening is entirely a property of
slot 0's own feature vector.

**Mechanism, fully diagnosed.** Perturbation test: took the real session's
actual 15 populated spectral/idle feature values and independently rescaled
each one by a random factor between 0.2x and 3x, five times. The raw score
did not move at all (`std == 0.0` across trials). The GBDT is not reading
these values for this exact combination of populated/NaN columns — it
outputs a near-constant based purely on which columns are missing, because
too few training rows ever land in this *exact* combination for the tree to
have learned real splits there.

**Two more targeted fixes attempted, both partial.** (a) The capability and
duration fixes above were sampled *independently*, making the joint "short
clip AND minimal capabilities" case — exactly the real profile — only
`short_session_prob * capability_dropout * 0.5` of sessions, compounding two
small probabilities into roughly 2 of 64 training sessions. Added
`sample_session_profile` to draw both jointly with an explicit guaranteed
slice for the minimal case. Retrained: AUC rose to 0.850, wallhack
recall@review rose from 0.0 to 1.0 (real, useful side effects) — but the
real-data perturbation test still showed zero sensitivity, just at a
different constant (0.6541 instead of 0.8805). (b) Checked
`TabularGBDT`'s `min_samples_leaf=40`: at ~10% joint-profile probability and
64 training sessions with ~5 players each, that exact profile gets roughly
30 rows total — structurally below the minimum needed to form its own leaf.
Retrained with `--sessions 220` instead of 64 to clear that floor. Result:
still not resolved. Full-precision raw scores across all 34 real segments
collapse to exactly **two** values (0.948308 and 0.942081), and neither
correlates with the true cheater/legit label — both appear in both groups.
More training data changed which two constants exist, not whether the model
discriminates.

**Four escalating, precisely-diagnosed fix attempts, one root cause fully
understood, discrimination for this exact real-data profile still
unresolved.** This is reported as open, not fixed. Closing it properly
needs either a real labelled dataset large enough to actually calibrate
against (33 segments from 2 accounts was enough to find and mechanistically
explain the problem in full, nowhere near enough to fix it with confidence
via more synthetic sampling), or an architectural change — e.g. a
dedicated path for zero-engagement sessions rather than asking one GBDT to
handle every capability/duration profile at once. Listed first in
`HANDOFF.md`'s next-work priorities.

**Mitigating context worth stating plainly**: this specific defect is real,
but the system's *other*, independent safety gates currently prevent it from
producing a wrongful high-severity verdict on data shaped like this. Every
real test session here has zero valid engagements (no matched target), and
`PolicyConfig.min_engagements_for_ban=40` means `Adjudicator.decide()` cannot
reach a BAN verdict for such a session regardless of how the tabular channel
scores it — that gate fails before the tabular score is even relevant to the
ban path. The measured `p_fused` (~0.62) sits below both the review (0.90)
and ban (0.995) thresholds, landing at most in the low-severity "watch" band.
This is not a defence of the bug — a `p_fused` this insensitive to real
signal is still a real problem for anyone reading "watch" as meaningful, and
the same GBDT insensitivity would presumably also affect sessions that *do*
have enough engagements to clear `min_engagements_for_ban`, which was not
tested here. It is stated because reporting the bug without this context
would overstate the practical severity in the current architecture, and
understating a real bug's severity is exactly as dishonest as overstating
one.

The capability-dropout, duration-variation, and joint-sampling fixes are all
kept — each is a genuine, independently-verified improvement on synthetic
data. AUC across the sequence of retrains in this investigation: 0.878
(pre-fix baseline) → 0.819 (capability dropout) → 0.789 (+ duration
variation) → 0.850 (+ joint-profile sampling) → 0.768 (+ 220 sessions instead
of 64, a different train/test split as much as a different fix). Movement in
either direction reflects a harder, more varied eval set and a different
random split each time, not a clean monotonic trend to read too much into —
wallhack recall@review going from 0.0 to 1.0 partway through is the more
meaningful single data point. None of these changes, individually or
combined, resolved discrimination for the exact narrow real-data profile
that started this investigation — that took a real dataset to find, and will
most likely take a larger one to fix.

## Fifth fix: stop trusting an unreliable score instead of trying to make it reliable

The mitigating context above (`min_engagements_for_ban` blocks a BAN on
zero-engagement sessions) left an explicit gap: it does not cover sessions
with *enough* engagements to clear that gate but still capability-sparse,
which was flagged as untested. Rather than leave that gap open, added
`coverage_shrink` (`aegis/models/ensemble.py`) — a small, directly-tested
pure function that pulls `p_fused` toward the prior in proportion to how
much of the feature vector is actually populated, engaged separately from
(and in addition to) the existing engagement-count shrink. It does not fix
the GBDT's insensitivity in the sparse region — that's still the layer-4
open item above — it stops that unreliable-by-construction score from
moving the posterior as if it were trustworthy, which is a defensible
position to ship even without a full fix: an anti-cheat system should treat
"I don't have enough signal to say" and "I have enough signal and it says
clean" differently, and it previously did not.

Deliberately a no-op at and above 50% feature coverage — a fully-capable
synthetic session still routinely sits around 70% coverage on its own
(hitgroups/visibility-derived columns that stay NaN unless that specific
event happened in that session), and an early version of this fix that
shrank proportionally to *raw* coverage measurably pulled down a genuine
aimbot session's score that had nothing wrong with it. The threshold
version doesn't. Verified: the real sparse-data test's `p_fused` dropped
from ~0.62 (in the low-severity "watch" band) to ~0.20-0.29 (below "watch"
entirely, into "clear"), while a full-capability synthetic aimbot session's
score was confirmed mathematically unaffected (`coverage_shrink(0.72) ==
1.0`, exactly). Fast regression test:
`test_coverage_shrink_targets_extreme_sparsity_only`.

## What has not been done, and cannot honestly be done here

No comparative benchmark against VACnet, Easy Anti-Cheat, BattlEye, Ricochet,
or FairFight has been run, and there is no legitimate way to run one from
this environment:

- All five are closed, proprietary systems. None publish their models,
  heuristics, or ban-rate statistics in a form that supports a real
  head-to-head.
- Four of the five target titles this goal named (Call of Duty: Warzone,
  Escape from Tarkov, Apex Legends, Battlefield 2042/V) have no public
  server-side plugin API, no dedicated-server binary Aegis can attach to, and
  no SDK — unlike CS2's MetaMod:Source2, which is what made the CS2 work in
  this repo possible at all. There is no legitimate path to live telemetry
  from any of them.
- Getting telemetry from a live, anti-cheat-protected commercial game without
  the operator's authorization would mean reverse-engineering client access
  into a service that is actively trying to prevent exactly that. That is not
  something this project will do, regardless of the stated goal — it is the
  same category of action as writing a cheat, aimed at someone else's
  infrastructure instead of a game.
- Aegis has never scored a real human population, cheating or clean. Every
  number below comes from a synthetic simulator or from bot-vs-bot CS2
  telemetry (real pipeline, not real players).

Any claim of the form "Aegis is more accurate than VACnet" or "Aegis catches
X% more Tarkov cheaters than BattlEye" would be fabricated. None of that
exists yet. What follows is the honest version: what's actually been built
and verified, what's architecturally different and why that might matter, and
what would actually be required to make the comparison real.

## What has been verified, concretely

**CS2 telemetry pipeline**, against a real MetaMod:Source2 dedicated server,
not simulated:
- Position, view angle, health, weapon, shots-fired, recoil/punch angle, and
  native 64 Hz sample rate — all confirmed live (`plugins/cs2/SDK_WIRING.md`).
  Two of these (origin, recoil) were wrong in ways that looked plausible
  until checked against the running server's own schema reflection; the
  fixes are in this repo now, not just planned.
- `validate_session` returns zero problems on a live capture.

**Capability honesty**, the mechanism the "never zero-fill a missing
capability" invariant depends on: `HIGH_RATE_VIEW` and `VISIBILITY` are now
derived from what a session actually measured, not trusted from an adapter's
default declaration. This was a real, live bug — CS2's plugin was throttled
to 16 Hz (Nyquist 8 Hz) while still claiming `HIGH_RATE_VIEW`, which the
`SpectralFeatures` tremor extractor requires and which physiological tremor
(4-12 Hz) needs well above 8 Hz to resolve. Fixed and covered by
`test_high_rate_view_reflects_measured_density`.

**Cross-title architecture**, proven rather than diagrammed: the one model
trained on the CS2-shaped synthetic simulator scores `apex_legends`,
`warzone`, `tarkov`, `battlefield`, and plain `generic_fps` sessions without
retraining (`test_generic_presets_score_without_retraining`). This is real
evidence that the feature/model/fusion stack is game-agnostic. It is **not**
evidence that Aegis detects cheating well in any of those titles — see the
caveat in `aegis/adapters/presets.py`: no real telemetry from any of them
exists, and the preset constants (unit scale, run speed) are labelled
placeholders, not measured engine facts.

**API hardening**: every `/v1/*` route now requires `X-Aegis-Api-Key` when
`AEGIS_API_KEY` is set, and an arbitrary-file-read via `AnalyzeRequest.source`
accepting any filesystem path with zero containment check has been closed.
Verified live: unauthenticated → 401, wrong key → 401, valid key + path
traversal → 400, valid key + in-bounds path → 200
(`test_service_api_key_gates_v1_routes`).

**Current synthetic detection numbers** (`aegis train --sessions 220
--duration 75`, `aegis evaluate --stealth-curve`, this session, this
codebase — reproducible via those two commands; the session count changed
from the earlier 64 during the real-data investigation above):

| Metric | Value |
|---|---|
| AUC | 0.768 |
| AP | 0.697 |
| ECE | 0.109 |

(Numbers moved several times over the course of this session — 0.880 → 0.878
after the `eng_prior` fix, then down through 0.819 / 0.789 / 0.850 / 0.768
across the capability-dropout, duration-variation, joint-sampling, and
larger-training-set changes above, each also changing the train/test split.
Movement in either direction reflects a harder or differently-shaped eval
set, not a clean trend — see the real-data investigation section for why
chasing this number further wasn't the point of any of those changes.)

| Band | threshold | TPR | FPR | precision |
|---|---|---|---|---|
| watch | 0.60 | 0.402 | 0.070 | 0.755 |
| review | 0.90 | 0.130 | 0.000 | 1.000 |
| ban | 0.995 | 0.000 | 0.000 | — |

Per-family recall @ review, single session: 0/8 individual cheat families
recall above zero except the multi-symptom combo
(`aimbot+triggerbot+wallhack+norecoil`, recall 1.0). This is weak, and it is
supposed to be read that way, not spun: single-session scoring is not the
system's real operating mode. `fusion/sprt.py` exists specifically because
one match's evidence is meant to accumulate with the next one, and the
`train_report.json` notes say so explicitly ("Single-session SPRT
understates detection power; production accumulates across matches").
Nothing here has been re-run in a multi-session accumulation scenario against
synthetic cheats yet, and nothing here has been run against a real cheat at
all.

## Structural differences worth naming — argued, not measured

These are architectural claims grounded in what's publicly documented about
each system's known design, not results from running against them. Treat
them as hypotheses this codebase is built to test once real data exists, not
conclusions.

- **VACnet** (Valve, CS2): reported to be a supervised model over labelled
  engagement snippets. That architecture is blind between retraining cycles
  to cheat patterns it wasn't trained on, and — being a single score — has no
  natural way to explain a ban. Aegis's causal-feature + explicit-fusion
  design (`fusion/explain.py`) is built to produce a human-readable reason
  for every verdict, and the novelty autoencoder (`models/novelty.py`) is
  built to flag distributional outliers a supervised model would score as
  confidently clean. Whether either actually holds up against a real novel
  cheat is untested.
- **Easy Anti-Cheat / BattlEye / Ricochet** (kernel-level, client-resident):
  built to catch tampering, not behaviour. A wallhack that reads memory
  without ever moving the crosshair, or a human "humaniser" cheat tuned to
  look like a person, is structurally outside what a pure integrity layer can
  see. Aegis pairs `integrity/attest.py` (comparable in spirit — a
  challenge-response tamper channel) with a behavioural layer that does not
  need the cheat to touch memory in a detectable way. This is the same
  argument the README's failure-mode table already makes; it is not new
  evidence, and it has not been tested against an actual kernel-mode cheat or
  an actual purely-informational wallhack.
- **FairFight / EA Anti-Cheat** (server-side heuristics, Battlefield):
  closer in spirit to Aegis's approach than the kernel-mode systems, and
  publicly reported to rely significantly on stat-outlier heuristics (K/D,
  accuracy) rather than causal motor-control features. Aimbot detection via
  raw accuracy is easy to smooth away with humaniser tuning; Aegis's spectral
  tremor and recoil-compensation-lag channels target the motor-control signal
  a stat-outlier check does not see. Also unverified against a real cheat.

## What it would actually take to make any of this real

1. Real, labelled negatives (clean human matches) and, ideally, real cheat
   samples for at least one title — CS2 is closest, given the live plugin
   now works, but bots are not humans and this repo has none yet.
2. Multi-session SPRT evaluation, not single-session — the current numbers
   above are a lower bound on what the system claims to actually do.
3. An authorized data-sharing relationship with a studio for any title
   beyond CS2. Nothing in this repo should be read as a plan to get that data
   any other way.
4. Re-derived, per-title conformal + policy thresholds before any verdict
   from a non-CS2 title is trusted (`ARCHITECTURE.md`, "Extending to another
   game," step 5) — sharing CS2's operating point across titles would be a
   silent, undetectable miscalibration.

Until (1) and (2) happen for at least one title, "better than VACnet" and
"better than EAC" are hypotheses this architecture is designed to eventually
support, not results.
