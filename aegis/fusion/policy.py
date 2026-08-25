"""Adjudication policy.

Turns a posterior + conformal p-value + integrity status into a verdict.
Deliberately boring and explicit: every branch is a named threshold from
``PolicyConfig``, shadow mode defaults ON, and integrity bans are a separate
deterministic path so a behavioural false positive can never be "rescued" by
a clean integrity check or vice versa.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from ..core.config import PolicyConfig
from ..models.ensemble import Score
from .prior import TrustPrior
from .sprt import EvidenceState, SPRT


class VerdictBand(enum.Enum):
    CLEAR = "clear"
    WATCH = "watch"
    REVIEW = "review"
    BAN = "ban"
    INTEGRITY_BAN = "integrity_ban"


@dataclass
class Verdict:
    player_id: str
    band: VerdictBand
    posterior: float
    conformal_p: float
    log_odds: float
    n_units: float
    n_engagements: int
    shadow: bool
    reasons: list[str] = field(default_factory=list)
    issued: bool = False  # False when shadow_mode suppresses the action
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "band": self.band.value,
            "posterior": self.posterior,
            "conformal_p": self.conformal_p,
            "log_odds": self.log_odds,
            "n_units": self.n_units,
            "n_engagements": self.n_engagements,
            "shadow": self.shadow,
            "issued": self.issued,
            "reasons": self.reasons,
            "ts": self.ts,
        }


@dataclass
class Adjudicator:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sprt: SPRT = field(default_factory=SPRT)

    def alpha_for(self, title: str | None) -> float:
        """Effective conformal alpha for a title (per-title override else global).

        Per-title gates exist because each game's clean population is
        different -- sharing one operating point across titles silently
        re-targets the system-wide false-positive budget (HANDOFF item 6).
        """
        overrides = getattr(self.sprt.cfg, "per_title_alpha", None) or {}
        if title and title in overrides:
            return float(overrides[title])
        return float(self.sprt.cfg.alpha)

    def decide(
        self,
        state: EvidenceState,
        score: Score,
        prior: TrustPrior,
        *,
        integrity_confirmed_tamper: bool = False,
        n_engagements_lifetime: int | None = None,
        title: str | None = None,
    ) -> Verdict:
        reasons: list[str] = []
        posterior = state.posterior(prior.p)
        n_eng = n_engagements_lifetime if n_engagements_lifetime is not None else score.meta.get("n_engagements", 0)
        alpha_eff = self.alpha_for(title)

        if integrity_confirmed_tamper and self.policy.integrity_ban_on_confirmed:
            band = VerdictBand.INTEGRITY_BAN
            reasons.append("confirmed_integrity_tamper")
            issued = not self.policy.shadow_mode
            return Verdict(
                player_id=state.player_id, band=band, posterior=1.0,
                conformal_p=0.0, log_odds=state.log_odds, n_units=state.n_units,
                n_engagements=int(n_eng), shadow=self.policy.shadow_mode,
                reasons=reasons, issued=issued,
            )

        band = VerdictBand.CLEAR
        if posterior >= self.policy.ban_threshold and self.sprt.crossed_ban(state):
            if n_eng < self.policy.min_engagements_for_ban:
                band = VerdictBand.REVIEW
                reasons.append(
                    f"posterior={posterior:.4f} crosses ban but only {n_eng} engagements "
                    f"(need {self.policy.min_engagements_for_ban})"
                )
            elif score.conformal_p >= alpha_eff:
                band = VerdictBand.REVIEW
                reason_src = (
                    f"per-title alpha={alpha_eff:.2e} [{title}]"
                    if title and title in (getattr(self.sprt.cfg, "per_title_alpha", None) or {})
                    else f"alpha={alpha_eff:.2e}"
                )
                reasons.append(
                    f"posterior={posterior:.4f} but conformal_p={score.conformal_p:.2e} "
                    f">= {reason_src}; refusing auto-ban"
                )
            else:
                band = VerdictBand.BAN
                reasons.append(
                    f"posterior={posterior:.4f} llr={state.log_odds:.2f} "
                    f"conformal_p={score.conformal_p:.2e} units={state.n_units:.1f}"
                )
        elif posterior >= self.policy.review_threshold or self.sprt.crossed_flag(state):
            band = VerdictBand.REVIEW
            reasons.append(f"posterior={posterior:.4f} flagged for human review")
        elif posterior >= self.policy.watch_threshold:
            band = VerdictBand.WATCH
            reasons.append(f"posterior={posterior:.4f} elevated; continue observing")
        else:
            reasons.append(f"posterior={posterior:.4f} within clear band")

        reasons.append(f"prior={prior.p:.4f} ({prior.reason})")
        reasons.append(
            f"scores tab={score.p_tabular:.3f} temp={score.p_temporal:.3f} "
            f"nov={score.p_novelty:.3f} fused={score.p_fused:.3f}"
        )

        issued = (not self.policy.shadow_mode) and band in (
            VerdictBand.BAN, VerdictBand.INTEGRITY_BAN,
        )
        return Verdict(
            player_id=state.player_id,
            band=band,
            posterior=posterior,
            conformal_p=score.conformal_p,
            log_odds=state.log_odds,
            n_units=state.n_units,
            n_engagements=int(n_eng),
            shadow=self.policy.shadow_mode,
            reasons=reasons,
            issued=issued,
        )
