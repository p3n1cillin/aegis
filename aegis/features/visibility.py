"""Wallhack / ESP detection via information-availability testing.

The framing that makes this tractable: cheating of this kind is not a motor
anomaly, it is an *information* anomaly. The player acts on state they could
not have observed. So instead of hand-tuning "suspicious tracking" thresholds,
we run a permutation test.

For every moment where an enemy is occluded from the player, we measure the
angular error between the player's crosshair and that enemy. Then we build a
null distribution by re-running the identical measurement against the *same
enemy's trajectory time-shifted by a large offset*. The shift destroys the
causal link while preserving everything else that could bias the statistic:
map geometry, the player's habitual angles, common holding positions, the
enemy's own movement habits, the duration of the round.

If a player has no wall information, their crosshair error against the real
occluded enemy is distributed identically to their error against the shifted
one. If they have information, the real distribution is shifted left, and the
size of that shift is a calibrated effect size rather than a magic number.

This is materially stronger than the "did they aim at someone through a wall"
heuristics that most systems use, because it is robust to pre-aiming common
angles -- pre-aiming shows up in the null too, and cancels.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import EXTRACTORS, Capability
from ..core.types import Team, angular_distance
from .base import EYE_HEIGHT_M, PlayerContext, SessionExtractor

# Angular window counted as "on" an enemy for the occupancy statistic.
NEAR_DEG = 12.0
# Time offsets (seconds) used to build the null. Multiple offsets reduce the
# variance of the null estimate.
NULL_SHIFTS = (37.0, 71.0, 113.0, -53.0, -89.0)
MIN_OCCLUDED_SAMPLES = 200


def _visibility_timeline(ctx: PlayerContext, subject: int) -> np.ndarray | None:
    """Boolean 'subject visible to ctx.slot' on the view grid.

    Built by replaying VisibilityEvent transitions. Returns None if the
    adapter never emitted transitions for this pair, which is different from
    "never visible" and must not be conflated with it.
    """
    evs = [v for v in ctx.session.visibility if v.observer == ctx.slot and v.subject == subject]
    if not evs:
        return None
    evs.sort(key=lambda v: v.t)
    g = ctx.view.t
    out = np.zeros(g.size, dtype=bool)
    state = False
    idx = 0
    for ev in evs:
        j = int(np.searchsorted(g, ev.t, side="left"))
        if j > idx:
            out[idx:j] = state
        state = ev.visible
        idx = max(idx, j)
    out[idx:] = state
    return out


def _bearing_error(ctx: PlayerContext, subject: int, shift_s: float = 0.0) -> np.ndarray:
    """Angular error from our crosshair to ``subject``, optionally time-shifted.

    A positive shift means we compare our view at time t against where the
    subject was at t + shift, wrapped within the subject's own trajectory so
    the marginal spatial distribution is preserved exactly.
    """
    tt = ctx.session.ticks_for(subject)
    if tt.size < 4 or ctx.ticks.size < 4:
        return np.full(len(ctx.view), np.nan)
    g = ctx.view.t
    st = tt["t"]
    span = float(st[-1] - st[0])
    if span <= 1.0:
        return np.full(g.size, np.nan)
    # Circular shift within the subject's own timeline.
    q = st[0] + np.mod((g + shift_s) - st[0], span)
    tx = np.interp(q, st, tt["px"])
    ty = np.interp(q, st, tt["py"])
    tz = np.interp(q, st, tt["pz"]) + EYE_HEIGHT_M
    sx = np.interp(g, ctx.ticks["t"], ctx.ticks["px"])
    sy = np.interp(g, ctx.ticks["t"], ctx.ticks["py"])
    sz = np.interp(g, ctx.ticks["t"], ctx.ticks["pz"]) + EYE_HEIGHT_M
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    want_yaw = np.degrees(np.arctan2(dy, dx))
    want_pitch = -np.degrees(np.arctan2(dz, np.maximum(np.hypot(dx, dy), 1e-6)))
    return angular_distance(ctx.view.pitch, ctx.view.yaw, want_pitch, want_yaw)


def _enemies(ctx: PlayerContext) -> list[int]:
    me = ctx.session.players.get(ctx.slot)
    if me is None:
        return []
    my_team = me.team
    if my_team in (int(Team.UNKNOWN), int(Team.SPECTATOR)):
        return []
    return [s for s, p in ctx.session.players.items()
            if p.team not in (my_team, int(Team.UNKNOWN), int(Team.SPECTATOR))]


@EXTRACTORS.register("visibility")
class VisibilityFeatures(SessionExtractor):
    name = "visibility"
    requires = (Capability.VIEW_ANGLES, Capability.POSITIONS, Capability.VISIBILITY)
    names = (
        "wh_occluded_near_rate",
        "wh_occluded_near_rate_null",
        "wh_occupancy_lift",
        "wh_median_err_delta",
        "wh_ks_statistic",
        "wh_effect_size",
        "wh_track_corr",
        "wh_track_corr_null",
        "wh_reveal_err_median",
        "wh_reveal_err_p10",
        "wh_reveal_prealigned_frac",
        "wh_n_occluded_samples",
        "wh_n_reveals",
        "wh_smoke_kill_rate",
        "wh_flashed_kill_rate",
        "wh_wallbang_rate",
        "wh_first_contact_advantage",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        enemies = _enemies(ctx)
        if not enemies or len(ctx.view) < 64:
            return out

        alive = ctx.alive_mask()
        real_err_parts: list[np.ndarray] = []
        null_err_parts: list[np.ndarray] = []
        track_real: list[float] = []
        track_null: list[float] = []
        reveal_errs: list[float] = []

        for sub in enemies:
            vis = _visibility_timeline(ctx, sub)
            if vis is None:
                continue
            sub_alive_ticks = ctx.session.ticks_for(sub)
            if sub_alive_ticks.size < 4:
                continue
            from ..core.types import PlayerFlag

            sub_alive_flags = (sub_alive_ticks["flags"].astype(np.int64) & int(PlayerFlag.ALIVE)) != 0
            idx = np.clip(np.searchsorted(sub_alive_ticks["t"], ctx.view.t, side="right") - 1,
                          0, sub_alive_ticks.size - 1)
            sub_alive = sub_alive_flags[idx]

            occluded = (~vis) & alive & sub_alive
            if occluded.sum() < 32:
                continue

            err = _bearing_error(ctx, sub)
            m = occluded & np.isfinite(err)
            if m.sum() < 32:
                continue
            real_err_parts.append(err[m])

            for sh in NULL_SHIFTS:
                nerr = _bearing_error(ctx, sub, shift_s=sh)
                nm = occluded & np.isfinite(nerr)
                if nm.sum() >= 32:
                    null_err_parts.append(nerr[nm])

            # Tracking correlation: does our yaw rate follow the occluded
            # enemy's bearing rate? Computed on occluded stretches only.
            e_smooth = err.copy()
            de = np.gradient(np.where(np.isfinite(e_smooth), e_smooth, 0.0), ctx.view.dt)
            yv = ctx.view.speed
            mm = occluded & np.isfinite(de) & np.isfinite(yv)
            if mm.sum() >= 64 and np.std(de[mm]) > 1e-9 and np.std(yv[mm]) > 1e-9:
                track_real.append(float(abs(np.corrcoef(de[mm], yv[mm])[0, 1])))
            for sh in NULL_SHIFTS[:2]:
                nerr = _bearing_error(ctx, sub, shift_s=sh)
                nde = np.gradient(np.where(np.isfinite(nerr), nerr, 0.0), ctx.view.dt)
                nm = occluded & np.isfinite(nde)
                if nm.sum() >= 64 and np.std(nde[nm]) > 1e-9 and np.std(yv[nm]) > 1e-9:
                    track_null.append(float(abs(np.corrcoef(nde[nm], yv[nm])[0, 1])))

            # Error at the instant visibility opened.
            trans = np.where(np.diff(vis.astype(np.int8)) == 1)[0]
            for i in trans:
                if alive[i] and sub_alive[i] and np.isfinite(err[i]):
                    reveal_errs.append(float(err[i]))

        if not real_err_parts:
            return out

        real = np.concatenate(real_err_parts)
        out["wh_n_occluded_samples"] = float(real.size)
        out["wh_occluded_near_rate"] = float((real < NEAR_DEG).mean())

        if null_err_parts and real.size >= MIN_OCCLUDED_SAMPLES:
            null = np.concatenate(null_err_parts)
            out["wh_occluded_near_rate_null"] = float((null < NEAR_DEG).mean())
            base = out["wh_occluded_near_rate_null"]
            # Lift: how many times more often than chance the crosshair sits on
            # an enemy it cannot see. 1.0 = no information. This is the headline
            # number a reviewer looks at.
            out["wh_occupancy_lift"] = float(out["wh_occluded_near_rate"] / max(base, 1e-4))
            out["wh_median_err_delta"] = float(np.median(null) - np.median(real))
            from scipy import stats

            try:
                sub_r = real if real.size <= 20000 else np.random.default_rng(0).choice(real, 20000, replace=False)
                sub_n = null if null.size <= 20000 else np.random.default_rng(1).choice(null, 20000, replace=False)
                out["wh_ks_statistic"] = float(stats.ks_2samp(sub_r, sub_n).statistic)
            except Exception:
                pass
            pooled = np.sqrt((real.var() + null.var()) / 2.0) + 1e-9
            out["wh_effect_size"] = float((null.mean() - real.mean()) / pooled)

        if track_real:
            out["wh_track_corr"] = float(np.mean(track_real))
        if track_null:
            out["wh_track_corr_null"] = float(np.mean(track_null))

        if reveal_errs:
            re_arr = np.asarray(reveal_errs)
            out["wh_n_reveals"] = float(re_arr.size)
            out["wh_reveal_err_median"] = float(np.median(re_arr))
            out["wh_reveal_err_p10"] = float(np.percentile(re_arr, 10))
            # Fraction of reveals where the crosshair was already within 5 deg.
            out["wh_reveal_prealigned_frac"] = float((re_arr < 5.0).mean())

        kills = ctx.session.kills_by(ctx.slot)
        if kills:
            n = float(len(kills))
            out["wh_smoke_kill_rate"] = sum(k.through_smoke for k in kills) / n
            out["wh_flashed_kill_rate"] = sum(k.attacker_flashed for k in kills) / n
            out["wh_wallbang_rate"] = sum(k.through_wall for k in kills) / n
            # First-contact advantage: fraction of kills where the victim had
            # been visible to the attacker for under 250 ms. High values mean
            # the attacker consistently won the information race.
            out["wh_first_contact_advantage"] = float(
                np.mean([1.0 if 0 <= k.victim_exposure_ms < 250 else 0.0 for k in kills])
            )
        return out
