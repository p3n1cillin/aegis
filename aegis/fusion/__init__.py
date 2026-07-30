"""Bayesian evidence fusion and adjudication.

Models emit per-session and per-engagement scores. This package turns them
into a running posterior over "this account is cheating" and a verdict:

  * ``TrustPrior``     -- account-level prior from platform trust signals
  * ``EvidenceState``  -- SPRT accumulator with half-life and correlation discount
  * ``Adjudicator``    -- maps posterior + conformal gate -> watch/review/ban
  * ``explain``        -- human-readable rationale for appeals / reviewers
"""

from .explain import Explanation, explain_score
from .policy import Adjudicator, Verdict, VerdictBand
from .prior import TrustPrior, trust_from_player_info
from .sprt import EvidenceState, EvidenceUnit, SPRT

__all__ = [
    "Adjudicator",
    "EvidenceState",
    "EvidenceUnit",
    "Explanation",
    "SPRT",
    "TrustPrior",
    "Verdict",
    "VerdictBand",
    "explain_score",
    "trust_from_player_info",
]
