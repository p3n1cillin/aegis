"""Sequential probability ratio test over behavioural evidence.

Why SPRT and not "ban if score > 0.99"
-------------------------------------
A single match score, no matter how well calibrated, cannot carry a 1-in-100k
false-positive budget. Real cheating is a *persistent* behaviour: it produces
many mildly-positive evidence units across matches. Innocent variance produces
occasional spikes. SPRT is the optimal sequential test for deciding between
two hypotheses with controlled error rates (Wald 1947), and it is what lets
``FusionConfig.alpha`` be an actual operating guarantee rather than a vibe.

Mechanics
---------
Each evidence unit contributes a clipped log-likelihood ratio. Within a
session, units are correlated (same player, same day, same mood), so we
discount by ``n ** corr_exponent``. Across sessions we apply an exponential
half-life so evidence from months ago fades. The running sum is compared to
the classical SPRT boundaries derived from alpha/beta.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..core.config import FusionConfig


@dataclass(slots=True)
class EvidenceUnit:
    """One atomic contribution to the SPRT accumulator."""

    llr: float
    t: float  # unix seconds (or any monotonic clock shared by the state)
    session_id: str
    kind: str = "engagement"  # engagement | session | integrity | novelty
    weight: float = 1.0
    meta: dict = field(default_factory=dict)


@dataclass
class EvidenceState:
    player_id: str
    log_odds: float = 0.0  # current accumulated LLR (after decay)
    n_units: float = 0.0  # effective (correlation-discounted) count
    n_sessions: int = 0
    last_t: float = 0.0
    units: list[EvidenceUnit] = field(default_factory=list)
    # Per-session running totals, used for the correlation discount.
    _session_llr: dict[str, float] = field(default_factory=dict)
    _session_n: dict[str, int] = field(default_factory=dict)

    def posterior(self, prior: float) -> float:
        """P(cheat | evidence) given an account prior."""
        prior = min(max(prior, 1e-9), 1 - 1e-9)
        logit_prior = math.log(prior) - math.log(1 - prior)
        z = logit_prior + self.log_odds
        # Stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)


@dataclass
class SPRT:
    cfg: FusionConfig = field(default_factory=FusionConfig)

    @property
    def upper(self) -> float:
        """Ban boundary on the LLR scale."""
        a, b = self.cfg.alpha, self.cfg.beta
        return math.log((1 - b) / a)

    @property
    def lower(self) -> float:
        """Clear / continue boundary."""
        a, b = self.cfg.alpha, self.cfg.beta
        return math.log(b / (1 - a))

    def decay_factor(self, dt_seconds: float) -> float:
        if dt_seconds <= 0 or self.cfg.half_life_days <= 0:
            return 1.0
        half = self.cfg.half_life_days * 86400.0
        return 0.5 ** (dt_seconds / half)

    def observe(self, state: EvidenceState, unit: EvidenceUnit) -> EvidenceState:
        """Incorporate one evidence unit; returns the mutated state."""
        # Decay existing evidence toward 0.
        if state.last_t > 0 and unit.t >= state.last_t:
            f = self.decay_factor(unit.t - state.last_t)
            state.log_odds *= f
            # Effective counts also fade -- old evidence is less "independent".
            state.n_units *= f

        llr = max(-self.cfg.llr_clip, min(self.cfg.llr_clip, unit.llr)) * unit.weight

        # Correlation discount within a session: the marginal value of the
        # n-th unit in a session is (n**c - (n-1)**c), not 1.
        n_prev = state._session_n.get(unit.session_id, 0)
        n_new = n_prev + 1
        c = self.cfg.corr_exponent
        marginal = (n_new ** c) - (n_prev ** c)
        state._session_n[unit.session_id] = n_new
        state._session_llr[unit.session_id] = state._session_llr.get(unit.session_id, 0.0) + llr
        if n_prev == 0:
            state.n_sessions += 1

        state.log_odds += llr * marginal
        state.n_units += marginal
        state.last_t = unit.t
        state.units.append(unit)
        # Bound memory: keep the most recent 500 units for explainability.
        if len(state.units) > 500:
            state.units = state.units[-500:]
        return state

    def observe_session(
        self,
        state: EvidenceState,
        *,
        session_id: str,
        session_llr: float,
        engagement_llrs: list[float],
        novelty_llr: float = 0.0,
        t: float | None = None,
    ) -> EvidenceState:
        """Convenience: ingest a whole match's worth of scores."""
        t = time.time() if t is None else t
        for i, llr in enumerate(engagement_llrs):
            self.observe(
                state,
                EvidenceUnit(llr=llr, t=t + i * 1e-3, session_id=session_id, kind="engagement"),
            )
        if session_llr:
            self.observe(
                state,
                EvidenceUnit(llr=session_llr, t=t + 0.5, session_id=session_id, kind="session"),
            )
        if novelty_llr:
            self.observe(
                state,
                EvidenceUnit(llr=novelty_llr, t=t + 0.6, session_id=session_id, kind="novelty"),
            )
        return state

    def crossed_ban(self, state: EvidenceState) -> bool:
        return (
            state.log_odds >= self.upper
            and state.n_units >= self.cfg.min_units_ban
        )

    def crossed_flag(self, state: EvidenceState) -> bool:
        # Soft boundary at half the ban threshold, with a lower unit count.
        return (
            state.log_odds >= 0.5 * self.upper
            and state.n_units >= self.cfg.min_units_flag
        )
