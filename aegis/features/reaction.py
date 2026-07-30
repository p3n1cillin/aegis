"""Reaction-time analysis.

Human simple reaction time to a visual stimulus has a hard physiological
floor. Retinal transduction plus cortical processing plus motor execution
cannot complete in under roughly 100 ms, and choice reaction -- which is what
"an enemy appeared, decide it is hostile, pull the trigger" actually is -- is
150-250 ms with an ex-Gaussian right tail.

Crucially the *floor* is what matters, not the mean. A very good player has a
mean of 180 ms; nobody has a mean of 180 ms and a floor of 40 ms. Triggerbots
live below the floor, and no amount of added delay in the cheat's config fixes
the shape of the distribution: a constant offset shifts the whole thing, which
shows up as an implausibly small variance for the given mean.

We model the distribution as ex-Gaussian (Gaussian + exponential tail), which
is the standard descriptive model for human RT, and score the likelihood of
the observed sample under a human prior.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from ..core.registry import EXTRACTORS, Capability
from .base import PlayerContext, SessionExtractor, safe_stats, stat_names

# Physiological floor for a visually-triggered aimed shot, seconds.
HUMAN_FLOOR_S = 0.100
# Population priors for choice RT in an FPS duel, fitted from open aim-trainer
# datasets; used as the null model, not as a hard rule.
PRIOR_MU, PRIOR_SIGMA, PRIOR_TAU = 0.190, 0.040, 0.075


def _exgauss_logpdf(x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    """log pdf of an ex-Gaussian, numerically stable for small tau."""
    sigma = max(sigma, 1e-4)
    tau = max(tau, 1e-4)
    z = (x - mu) / sigma - sigma / tau
    # log(1/tau) - (x-mu)/tau + sigma^2/(2 tau^2) + log Phi(z)
    log_phi = special.log_ndtr(z)
    return -np.log(tau) - (x - mu) / tau + (sigma**2) / (2 * tau**2) + log_phi


def _fit_exgauss(x: np.ndarray) -> tuple[float, float, float]:
    if x.size < 8:
        return float("nan"), float("nan"), float("nan")

    def nll(p):
        mu, log_s, log_t = p
        v = _exgauss_logpdf(x, mu, np.exp(log_s), np.exp(log_t))
        if not np.all(np.isfinite(v)):
            return 1e9
        return -float(v.sum())

    x0 = np.array([float(np.median(x)), np.log(max(x.std(), 1e-3)), np.log(max(x.std(), 1e-3))])
    try:
        res = optimize.minimize(nll, x0, method="Nelder-Mead",
                                options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4})
        mu, s, t = res.x[0], float(np.exp(res.x[1])), float(np.exp(res.x[2]))
        return float(mu), s, t
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _reaction_times(ctx: PlayerContext) -> tuple[np.ndarray, np.ndarray]:
    """(reaction times, pre-fire margins) for this player's engagements.

    Reaction time  = first shot - moment the target became visible.
    Pre-fire margin = positive when the player fired *before* visibility, which
    is legitimate only when pre-aiming a known position; we keep it separate
    rather than treating it as a negative reaction time.
    """
    rts, pre = [], []
    for e in ctx.engagements:
        if not np.isfinite(e.t_target_visible):
            continue
        d = e.t_first_shot - e.t_target_visible
        if d >= 0:
            rts.append(d)
        else:
            pre.append(-d)
    return np.asarray(rts, dtype=np.float64), np.asarray(pre, dtype=np.float64)


@EXTRACTORS.register("reaction")
class ReactionFeatures(SessionExtractor):
    name = "reaction"
    requires = (Capability.VISIBILITY, Capability.SHOTS)
    names = (
        *stat_names("rt"),
        "rt_min",
        "rt_p05",
        "rt_frac_under_floor",
        "rt_frac_under_150ms",
        "rt_n",
        "rt_exg_mu",
        "rt_exg_sigma",
        "rt_exg_tau",
        "rt_loglik_vs_prior",
        "rt_cv",
        "rt_tail_ratio",
        "prefire_rate",
        "prefire_mean_margin",
        "rt_kill_vs_miss_delta",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        rts, pre = _reaction_times(ctx)
        n_eng = max(len(ctx.engagements), 1)
        out["prefire_rate"] = float(pre.size / n_eng)
        out["prefire_mean_margin"] = float(pre.mean()) if pre.size else 0.0
        out["rt_n"] = float(rts.size)
        if rts.size == 0:
            return out

        out.update(safe_stats(rts, "rt"))
        out["rt_min"] = float(rts.min())
        out["rt_p05"] = float(np.percentile(rts, 5))
        out["rt_frac_under_floor"] = float((rts < HUMAN_FLOOR_S).mean())
        out["rt_frac_under_150ms"] = float((rts < 0.150).mean())
        out["rt_cv"] = float(rts.std() / (rts.mean() + 1e-9))
        # Right-tail mass relative to the median: humans have a long tail
        # (distraction, target reacquisition). A triggerbot has almost none.
        med = float(np.median(rts))
        out["rt_tail_ratio"] = float((rts > 2 * med).mean()) if med > 1e-6 else float("nan")

        if rts.size >= 8:
            mu, sig, tau = _fit_exgauss(rts)
            out["rt_exg_mu"], out["rt_exg_sigma"], out["rt_exg_tau"] = mu, sig, tau
            if np.isfinite(mu):
                ll_fit = float(_exgauss_logpdf(rts, mu, sig, tau).sum())
                ll_prior = float(_exgauss_logpdf(rts, PRIOR_MU, PRIOR_SIGMA, PRIOR_TAU).sum())
                # How much better the player's own distribution fits than the
                # human population prior, per observation. Large positive =
                # this player's RT distribution is not a human one.
                out["rt_loglik_vs_prior"] = (ll_fit - ll_prior) / rts.size

        # Do reaction times differ between engagements that killed and those
        # that did not? For a human, faster reactions convert more often. For a
        # triggerbot, everything converts and the delta collapses.
        kill_rt, miss_rt = [], []
        for e in ctx.engagements:
            if not np.isfinite(e.t_target_visible):
                continue
            d = e.t_first_shot - e.t_target_visible
            if d < 0:
                continue
            (kill_rt if e.killed else miss_rt).append(d)
        if len(kill_rt) >= 4 and len(miss_rt) >= 4:
            out["rt_kill_vs_miss_delta"] = float(np.mean(miss_rt) - np.mean(kill_rt))
        return out
