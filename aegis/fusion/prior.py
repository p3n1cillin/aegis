"""Account-level trust priors.

The behavioural model must not see trust features -- otherwise it learns
"new account = cheater" and launders a reputation heuristic into a supposedly
behavioural score. The prior is applied *only* at the fusion layer, where it
belongs: as P(cheat) before any match evidence arrives.

Platforms that already have a trust system (VAC bans on record, Steam age,
prime status, rank) should map them into ``TrustPrior`` here. Defaults are
conservative: a brand-new account is mildly elevated, not presumed guilty.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import PlayerInfo


@dataclass(frozen=True)
class TrustPrior:
    """Prior P(cheat) for an account before behavioural evidence."""

    p: float
    reason: str = ""

    def clamped(self, lo: float = 1e-4, hi: float = 0.5) -> "TrustPrior":
        return TrustPrior(p=min(max(self.p, lo), hi), reason=self.reason)


def trust_from_player_info(
    info: PlayerInfo,
    *,
    base_rate: float = 0.04,
    new_account_hours: float = 50.0,
) -> TrustPrior:
    """Derive a prior from the fields the adapter is allowed to supply.

    ``trust_prior`` from the platform is respected if present and in (0,1),
    then gently adjusted by hours played. Rank is intentionally unused: skill
    is correlated with both honesty and cheating sophistication, and using it
    here would create a perverse incentive.
    """
    p = float(info.trust_prior) if info.trust_prior is not None else base_rate
    reasons = []
    if info.trust_prior is not None and abs(info.trust_prior - 0.5) > 1e-6:
        reasons.append(f"platform_trust={info.trust_prior:.3f}")
    else:
        p = base_rate
        reasons.append(f"base_rate={base_rate:.3f}")

    hours = info.hours_played
    if hours is not None and hours < new_account_hours:
        # Lift toward 2x base rate, tapering as hours approach the threshold.
        lift = (1.0 - hours / new_account_hours) * base_rate
        p = min(0.5, p + lift)
        reasons.append(f"new_account_hours={hours:.0f}")

    if info.is_bot:
        p = 1e-6
        reasons = ["bot"]

    return TrustPrior(p=p, reason=";".join(reasons)).clamped()
