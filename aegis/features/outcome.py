"""Outcome and performance features.

These are the weakest evidence in the system and are deliberately kept
separate from the mechanistic channels, because they are the ones that
generate false positives: a genuinely excellent player has an excellent
headshot rate.

They earn their place for two reasons. First, as a *conditioning* signal --
extreme mechanical outcomes raise the prior that something is worth looking
at, without ever being sufficient on their own (see ``detect.fusion``, which
caps the LLR contribution of this block). Second, for *inconsistency*
detection: a player whose outcome statistics are far better than their
mechanical statistics predict is either playing below their ability or being
helped, and the residual between the two is more informative than either.

All rates are shrunk toward the population mean with an empirical-Bayes prior
so that a 3-for-3 opening does not read as 100% headshot rate.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import EXTRACTORS, Capability
from ..core.types import HitGroup
from .base import PlayerContext, SessionExtractor, safe_stats, stat_names

# Empirical-Bayes pseudo-counts, roughly matched to open CS2 match data.
HS_PRIOR, HS_STRENGTH = 0.42, 20.0
ACC_PRIOR, ACC_STRENGTH = 0.22, 40.0


def _shrink(successes: float, trials: float, prior: float, strength: float) -> float:
    return float((successes + prior * strength) / (trials + strength))


@EXTRACTORS.register("outcome")
class OutcomeFeatures(SessionExtractor):
    name = "outcome"
    requires = (Capability.SHOTS, Capability.DAMAGE)
    names = (
        "oc_headshot_rate",
        "oc_headshot_rate_raw",
        "oc_accuracy",
        "oc_first_bullet_accuracy",
        "oc_damage_per_shot",
        "oc_kills_per_min",
        "oc_kd",
        "oc_multi_kill_rate",
        "oc_time_between_kills_min",
        "oc_kill_dist_p90",
        "oc_long_range_hs_rate",
        "oc_spray_hit_ratio",
        "oc_hitgroup_entropy",
        *stat_names("oc_ttk"),
        "oc_n_kills",
        "oc_n_shots",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        shots = ctx.session.shots_for(ctx.slot)
        dmgs = [d for d in ctx.session.damages if d.attacker == ctx.slot]
        kills = ctx.session.kills_by(ctx.slot)
        deaths = sum(1 for k in ctx.session.kills if k.victim == ctx.slot)

        out["oc_n_shots"] = float(len(shots))
        out["oc_n_kills"] = float(len(kills))
        if not shots:
            return out

        n_shots = float(len(shots))
        n_hits = float(len(dmgs))
        out["oc_accuracy"] = _shrink(n_hits, n_shots, ACC_PRIOR, ACC_STRENGTH)
        out["oc_damage_per_shot"] = float(sum(d.damage for d in dmgs) / n_shots)

        hs = sum(1 for d in dmgs if d.hitgroup == int(HitGroup.HEAD))
        out["oc_headshot_rate"] = _shrink(hs, max(n_hits, 1.0), HS_PRIOR, HS_STRENGTH)
        out["oc_headshot_rate_raw"] = float(hs / n_hits) if n_hits else float("nan")

        # First-bullet accuracy: the shot that starts an engagement. Immune to
        # spray-transfer luck, and the number an aimbot inflates most.
        firsts = [e for e in ctx.engagements if e.n_shots]
        if firsts:
            hit_first = 0
            for e in firsts:
                t0 = e.shots[0].t
                if any(abs(d.t - t0) < 0.12 for d in e.damages):
                    hit_first += 1
            out["oc_first_bullet_accuracy"] = _shrink(hit_first, len(firsts), ACC_PRIOR, ACC_STRENGTH)

        dur_min = max(ctx.session.duration() / 60.0, 1e-3)
        out["oc_kills_per_min"] = float(len(kills) / dur_min)
        out["oc_kd"] = float(len(kills) / max(deaths, 1))

        if kills:
            kt = np.array(sorted(k.t for k in kills))
            if kt.size >= 2:
                gaps = np.diff(kt)
                out["oc_time_between_kills_min"] = float(gaps.min())
                out["oc_multi_kill_rate"] = float((gaps < 2.0).mean())
            dist = np.array([k.distance for k in kills], dtype=np.float64)
            out["oc_kill_dist_p90"] = float(np.percentile(dist, 90))
            long_k = [k for k in kills if k.distance > 25.0]
            if len(long_k) >= 3:
                out["oc_long_range_hs_rate"] = _shrink(
                    sum(1 for k in long_k if k.headshot), len(long_k), HS_PRIOR, HS_STRENGTH)

        # Spray hit ratio: of shots after the third in a burst, how many land?
        # Bloom makes this low for humans and high for anyone with no-spread.
        late_shots, late_hits = 0, 0
        for e in ctx.engagements:
            for i, s in enumerate(e.shots):
                if i < 3:
                    continue
                late_shots += 1
                if any(abs(d.t - s.t) < 0.08 for d in e.damages):
                    late_hits += 1
        if late_shots >= 10:
            out["oc_spray_hit_ratio"] = float(late_hits / late_shots)

        # Hitgroup entropy: a human's damage spreads across the body; an
        # aimbot locked to one bone concentrates it.
        if dmgs:
            groups = np.array([d.hitgroup for d in dmgs])
            _, counts = np.unique(groups, return_counts=True)
            p = counts / counts.sum()
            out["oc_hitgroup_entropy"] = float(-(p * np.log(p)).sum() / np.log(max(len(p), 2)))

        # Time-to-kill within an engagement.
        ttk = [e.kill.t - e.t_first_shot for e in ctx.engagements if e.killed]
        out.update(safe_stats(np.asarray(ttk), "oc_ttk"))
        return out
