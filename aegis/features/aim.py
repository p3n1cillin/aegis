"""Aim kinematics.

The core claim of this module: human aiming is *motor control*, and motor
control has structure that is hard to fake and easy to measure.

Specifically, human target acquisition follows the optimised-submovement model
(Meyer et al.): a fast, ballistic primary movement that deliberately
undershoots or overshoots, followed by one or more visually-guided corrective
submovements. That produces a characteristic signature -- multi-peaked speed
profile, curved path, asymmetric velocity profile with faster acceleration
than deceleration, and residual tremor at 8-12 Hz.

An aimbot solves for the target angle and interpolates. Even good humanisers
produce a *single* corrective structure, near-perfect path efficiency, and a
suspiciously tight distribution of "error at trigger pull".

VACnet looks mostly at the raw angle deltas around a kill and lets a CNN find
whatever it finds. That works, but it is a black box that has to be retrained
on new labels for every new cheat family. The features here are causal: they
describe *why* a movement is non-human, which means (a) they transfer to new
cheats without new labels, (b) they transfer to new games, and (c) they can be
shown to a human reviewer during an appeal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..core.registry import EXTRACTORS, Capability
from .base import Engagement, EngagementExtractor, PlayerContext, SessionExtractor, safe_stats, stat_names

# Angular error (deg) below which we consider the crosshair "on target".
ON_TARGET_DEG = 2.5
# Speed (deg/s) above which the crosshair is considered to be actively moving.
MOTION_ONSET_DEG_S = 45.0


def _approach_window(ctx: PlayerContext, eng: Engagement) -> tuple[slice, np.ndarray, np.ndarray]:
    """Locate the movement that brought the crosshair onto the target.

    Returns (slice into the view grid, angular error over that slice, times).
    The window ends at the first shot and starts at the last moment the
    crosshair was both far from the target and slow -- i.e. the start of the
    ballistic phase, not an arbitrary fixed lookback.
    """
    g, err = ctx.target_angle_error(eng.target)
    sl = ctx.view.window(eng.t_window_start, eng.t_first_shot)
    if sl.stop - sl.start < 4:
        return sl, np.empty(0), np.empty(0)

    e = err[sl]
    spd = ctx.view.speed[sl]
    t = g[sl]
    # Walk back from the shot to the last sample that was off-target and slow.
    off = np.where((e > ON_TARGET_DEG * 2.5) & (spd < MOTION_ONSET_DEG_S))[0]
    start = int(off[-1]) if off.size else 0
    sl2 = slice(sl.start + start, sl.stop)
    return sl2, err[sl2], g[sl2]


def _path_efficiency(pitch: np.ndarray, yaw: np.ndarray, cosp: np.ndarray) -> float:
    """Net angular displacement divided by total path length.

    1.0 means the crosshair travelled in a perfect geodesic to the target --
    which is what a solver does and what a hand does not. Humans land in
    0.75-0.95 for fast flicks and lower for tracking.
    """
    if pitch.size < 3:
        return float("nan")
    dp = np.diff(pitch)
    dy = np.diff(yaw) * cosp[:-1]
    steps = np.hypot(dp, dy)
    path = float(steps.sum())
    if path < 1e-6:
        return float("nan")
    net = float(np.hypot(pitch[-1] - pitch[0], (yaw[-1] - yaw[0]) * float(np.mean(cosp))))
    return min(net / path, 1.0)


def _submovements(speed: np.ndarray, dt: float) -> tuple[int, float]:
    """Count corrective submovements and return the primary/secondary ratio.

    A submovement is a peak in the angular speed profile. Human acquisitions
    show 2-4; a single-peak profile with nothing after it is the signature of
    an interpolated solve.
    """
    if speed.size < 6:
        return 0, float("nan")
    peak_v = float(speed.max())
    if peak_v < MOTION_ONSET_DEG_S:
        return 0, float("nan")
    idx, props = find_peaks(speed, prominence=peak_v * 0.12, distance=max(2, int(0.02 / dt)))
    if idx.size == 0:
        return 1, float("nan")
    heights = speed[idx]
    order = np.argsort(heights)[::-1]
    ratio = float(heights[order[1]] / heights[order[0]]) if idx.size > 1 else 0.0
    return int(idx.size), ratio


def _velocity_asymmetry(speed: np.ndarray) -> float:
    """Acceleration-time over total-time for the dominant speed peak.

    Human rapid movements are asymmetric: roughly 40% of the movement duration
    is spent accelerating and 60% decelerating (visually-guided braking).
    A linear interpolation or a symmetric ease produces ~0.5.
    """
    if speed.size < 5:
        return float("nan")
    k = int(np.argmax(speed))
    peak = speed[k]
    if peak <= 1e-6:
        return float("nan")
    thresh = peak * 0.10
    i0 = k
    while i0 > 0 and speed[i0 - 1] > thresh:
        i0 -= 1
    i1 = k
    while i1 < speed.size - 1 and speed[i1 + 1] > thresh:
        i1 += 1
    total = i1 - i0
    if total <= 1:
        return float("nan")
    return float((k - i0) / total)


def _quantisation_score(pitch: np.ndarray, yaw: np.ndarray) -> float:
    """Detect angle deltas that live on a coarse lattice.

    Many cheats compute a correction, divide it by N smoothing steps, and
    apply identical increments. Real mouse input is quantised by the sensor,
    but at a much finer scale and with dither from sensitivity scaling. We
    measure the fraction of consecutive deltas that repeat to 5 decimals.
    """
    if pitch.size < 8:
        return float("nan")
    d = np.round(np.hypot(np.diff(pitch), np.diff(yaw)), 5)
    d = d[d > 1e-5]
    if d.size < 6:
        return float("nan")
    _, counts = np.unique(d, return_counts=True)
    return float(counts.max() / d.size)


@EXTRACTORS.register("aim_engagement")
class AimEngagementFeatures(EngagementExtractor):
    name = "aim_engagement"
    requires = (Capability.VIEW_ANGLES, Capability.POSITIONS, Capability.SHOTS)
    names = (
        "aim_snap_duration",
        "aim_snap_peak_speed",
        "aim_snap_peak_accel",
        "aim_snap_peak_jerk",
        "aim_path_efficiency",
        "aim_n_submovements",
        "aim_submovement_ratio",
        "aim_velocity_asymmetry",
        "aim_overshoot_deg",
        "aim_overshoot_ratio",
        "aim_error_at_first_shot",
        "aim_error_min_pre_shot",
        "aim_speed_at_first_shot",
        "aim_settle_time",
        "aim_quantisation",
        "aim_lock_time_after_visible",
        "aim_fire_delay_after_lock",
        "aim_approach_angle_deg",
        "aim_jerk_entropy",
        "aim_post_shot_drift",
    )

    def extract_one(self, ctx: PlayerContext, eng: Engagement) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        sl, err, times = _approach_window(ctx, eng)
        v = ctx.view
        if err.size < 4:
            return out

        speed = v.speed[sl]
        accel = v.accel[sl]
        jerk = v.jerk[sl]
        pitch = v.pitch_u[sl]
        yaw = v.yaw_u[sl]
        cosp = np.cos(np.radians(np.clip(v.pitch[sl], -89.5, 89.5)))
        dt = v.dt

        out["aim_snap_duration"] = float(times[-1] - times[0])
        out["aim_snap_peak_speed"] = float(speed.max())
        out["aim_snap_peak_accel"] = float(accel.max())
        out["aim_snap_peak_jerk"] = float(jerk.max())
        out["aim_path_efficiency"] = _path_efficiency(pitch, yaw, cosp)

        n_sub, ratio = _submovements(speed, dt)
        out["aim_n_submovements"] = float(n_sub)
        out["aim_submovement_ratio"] = ratio
        out["aim_velocity_asymmetry"] = _velocity_asymmetry(speed)
        out["aim_quantisation"] = _quantisation_score(pitch, yaw)

        # Overshoot: did the crosshair cross the target and come back?
        i_min = int(np.argmin(err))
        out["aim_error_min_pre_shot"] = float(err[i_min])
        if i_min < err.size - 1:
            after = err[i_min:]
            out["aim_overshoot_deg"] = float(after.max() - err[i_min])
            span = float(err[0] - err[i_min])
            out["aim_overshoot_ratio"] = float((after.max() - err[i_min]) / span) if span > 1e-6 else 0.0
        else:
            out["aim_overshoot_deg"] = 0.0
            out["aim_overshoot_ratio"] = 0.0

        out["aim_error_at_first_shot"] = float(err[-1])
        out["aim_speed_at_first_shot"] = float(speed[-1])
        out["aim_approach_angle_deg"] = float(err[0])

        # Settle time: from first crossing of ON_TARGET to the shot. An aimbot
        # that snaps-and-fires has near-zero settle; a human has 40-200 ms of
        # micro-correction.
        on = np.where(err <= ON_TARGET_DEG)[0]
        out["aim_settle_time"] = float(times[-1] - times[on[0]]) if on.size else float("nan")

        # Jerk entropy: humans produce broadband jerk (tremor + correction);
        # interpolated motion produces a few discrete jerk impulses.
        j = np.abs(jerk)
        s = j.sum()
        if s > 1e-9 and j.size > 3:
            p = j / s
            p = p[p > 0]
            out["aim_jerk_entropy"] = float(-(p * np.log(p)).sum() / np.log(p.size))

        # Time from the target becoming visible to the crosshair locking on.
        if np.isfinite(eng.t_target_visible) and on.size:
            out["aim_lock_time_after_visible"] = float(times[on[0]] - eng.t_target_visible)
        if on.size:
            out["aim_fire_delay_after_lock"] = float(eng.t_first_shot - times[on[0]])

        # Post-shot drift: after firing, does the crosshair keep tracking the
        # target with no input (aimbot still engaged) or fall off (human)?
        g, full_err = ctx.target_angle_error(eng.target)
        post = ctx.view.window(eng.t_last_shot, eng.t_last_shot + 0.30)
        pe = full_err[post]
        pe = pe[np.isfinite(pe)]
        if pe.size >= 3:
            out["aim_post_shot_drift"] = float(pe.max() - pe.min())
        return out


@EXTRACTORS.register("aim_session")
class AimSessionFeatures(SessionExtractor):
    """Session-level aim consistency.

    The individually-unremarkable-but-collectively-impossible channel. A human
    has a wide, heavy-tailed distribution of aim error at trigger pull. A
    "legit" aimbot with a 30% FOV and humaniser has a *narrower* one, and
    crucially its width barely changes with target distance, target speed or
    the player's own movement state -- because the solver does not care about
    any of those, and a hand does.
    """

    name = "aim_session"
    requires = (Capability.VIEW_ANGLES, Capability.POSITIONS, Capability.SHOTS)
    names = (
        *stat_names("s_aim_err"),
        *stat_names("s_path_eff"),
        *stat_names("s_snap_speed"),
        "s_aim_err_cv",
        "s_aim_err_dist_corr",
        "s_aim_err_speed_corr",
        "s_path_eff_frac_over_097",
        "s_submov_mean",
        "s_submov_frac_single",
        "s_settle_frac_under_30ms",
        "s_vel_asym_mean",
        "s_vel_asym_std",
        "s_quantisation_max",
        "s_n_engagements",
        "s_flick_rate_per_min",
        "s_err_stationarity",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        engs = ctx.engagements
        out["s_n_engagements"] = float(len(engs))
        if not engs:
            return out

        eng_feat = AimEngagementFeatures()
        rows = [eng_feat.extract_one(ctx, e) for e in engs]

        errs = np.array([r["aim_error_at_first_shot"] for r in rows], dtype=np.float64)
        effs = np.array([r["aim_path_efficiency"] for r in rows], dtype=np.float64)
        spds = np.array([r["aim_snap_peak_speed"] for r in rows], dtype=np.float64)
        subs = np.array([r["aim_n_submovements"] for r in rows], dtype=np.float64)
        setl = np.array([r["aim_settle_time"] for r in rows], dtype=np.float64)
        asym = np.array([r["aim_velocity_asymmetry"] for r in rows], dtype=np.float64)
        quant = np.array([r["aim_quantisation"] for r in rows], dtype=np.float64)
        dists = np.array([e.distance for e in engs], dtype=np.float64)

        out.update(safe_stats(errs, "s_aim_err"))
        out.update(safe_stats(effs, "s_path_eff"))
        out.update(safe_stats(spds, "s_snap_speed"))

        fe = errs[np.isfinite(errs)]
        if fe.size >= 3:
            out["s_aim_err_cv"] = float(fe.std() / (fe.mean() + 1e-9))
            # Stationarity: split the match in half; a human's aim error
            # distribution drifts (warm-up, tilt, fatigue). A solver's does not.
            h = fe.size // 2
            if h >= 3:
                a, b = fe[:h], fe[h:]
                pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
                out["s_err_stationarity"] = float(abs(a.mean() - b.mean()) / pooled)

        m = np.isfinite(errs) & np.isfinite(dists) & (dists > 0)
        if m.sum() >= 5 and np.std(dists[m]) > 1e-6 and np.std(errs[m]) > 1e-9:
            out["s_aim_err_dist_corr"] = float(np.corrcoef(errs[m], dists[m])[0, 1])
        m2 = np.isfinite(errs) & np.isfinite(spds)
        if m2.sum() >= 5 and np.std(spds[m2]) > 1e-6 and np.std(errs[m2]) > 1e-9:
            out["s_aim_err_speed_corr"] = float(np.corrcoef(errs[m2], spds[m2])[0, 1])

        fef = effs[np.isfinite(effs)]
        if fef.size:
            out["s_path_eff_frac_over_097"] = float((fef > 0.97).mean())
        fs = subs[np.isfinite(subs)]
        if fs.size:
            out["s_submov_mean"] = float(fs.mean())
            out["s_submov_frac_single"] = float((fs <= 1).mean())
        fst = setl[np.isfinite(setl)]
        if fst.size:
            out["s_settle_frac_under_30ms"] = float((fst < 0.030).mean())
        fa = asym[np.isfinite(asym)]
        if fa.size:
            out["s_vel_asym_mean"] = float(fa.mean())
            out["s_vel_asym_std"] = float(fa.std())
        fq = quant[np.isfinite(quant)]
        if fq.size:
            out["s_quantisation_max"] = float(fq.max())

        dur_min = max(ctx.session.duration() / 60.0, 1e-3)
        flicks = np.isfinite(spds) & (spds > 600.0)
        out["s_flick_rate_per_min"] = float(flicks.sum() / dur_min)
        return out
