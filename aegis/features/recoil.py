"""Recoil compensation analysis (no-recoil / RCS detection).

Every automatic weapon applies a deterministic recoil vector per shot, known
authoritatively to the server. A player fighting that recoil produces
counter-movement in their view angles. The question is *how* they produce it.

  Human:  learned motor program, applied open-loop, with 30-80 ms of visual
          correction on top. Gain is close to but not equal to 1, the residual
          is large and structured, the lag is non-zero, and -- critically --
          the trace differs from spray to spray because a hand cannot replay
          the same motion twice.

  Cheat:  the recoil vector is read from memory and negated. Gain is ~1.000,
          residual is at the noise floor, lag is 0 or negative (applied on the
          same tick as the shot), and consecutive sprays are near-identical.

The cross-spray self-similarity feature is the strongest one here, and it is
what defeats "randomised RCS" humanisers: adding noise to a perfect solution
still leaves the *mean* trace inhumanly stable, and it drives the residual
distribution toward Gaussian white, which a hand never produces.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import EXTRACTORS, Capability
from ..core.types import wrap_deg
from .base import PlayerContext, SessionExtractor, safe_stats, stat_names

MIN_SPRAY = 5  # shots
TRACE_LEN = 12  # resampled shot-index positions per spray


def _spray_groups(shots: list) -> list[list]:
    """Split a player's shots into sprays (continuous automatic fire)."""
    if not shots:
        return []
    out = [[shots[0]]]
    for s in shots[1:]:
        prev = out[-1][-1]
        same_weapon = s.weapon == prev.weapon
        continuous = (s.t - prev.t) < 0.25  # slower than this is tap-firing
        increasing = s.spray_index > prev.spray_index or s.is_first_of_spray is False
        if same_weapon and continuous and increasing:
            out[-1].append(s)
        else:
            out.append([s])
    return [g for g in out if len(g) >= MIN_SPRAY]


def _view_at(ctx: PlayerContext, t: float) -> tuple[float, float]:
    v = ctx.view
    if len(v) == 0:
        return float("nan"), float("nan")
    i = int(np.clip(np.searchsorted(v.t, t), 0, len(v) - 1))
    return float(v.pitch_u[i]), float(v.yaw_u[i])


def _spray_traces(ctx: PlayerContext, spray: list) -> tuple[np.ndarray, np.ndarray]:
    """Per-shot (recoil delta, view delta) in the pitch/yaw plane.

    Returns arrays of shape (n_shots-1, 2): the recoil the weapon applied
    between consecutive shots, and the view change the player made over the
    same interval.
    """
    rec, vw = [], []
    for a, b in zip(spray, spray[1:]):
        dr_p = b.recoil_pitch - a.recoil_pitch
        dr_y = wrap_deg(b.recoil_yaw - a.recoil_yaw)
        pa, ya = _view_at(ctx, a.t)
        pb, yb = _view_at(ctx, b.t)
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        rec.append((dr_p, float(dr_y)))
        vw.append((pb - pa, yb - ya))
    return np.asarray(rec, dtype=np.float64), np.asarray(vw, dtype=np.float64)


def _resample_trace(x: np.ndarray, n: int = TRACE_LEN) -> np.ndarray:
    """Resample a per-shot trace to a fixed length so sprays are comparable."""
    if x.size == 0:
        return np.full(n, np.nan)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


@EXTRACTORS.register("recoil")
class RecoilFeatures(SessionExtractor):
    name = "recoil"
    requires = (Capability.VIEW_ANGLES, Capability.SHOTS, Capability.RECOIL_STATE)
    names = (
        "rc_gain_pitch",
        "rc_gain_yaw",
        "rc_r2",
        "rc_residual_rms",
        "rc_residual_norm",
        "rc_lag_shots",
        "rc_cross_spray_corr",
        "rc_cross_spray_corr_std",
        "rc_trace_std_norm",
        "rc_gain_stability",
        "rc_n_sprays",
        "rc_residual_gauss_p",
        *stat_names("rc_spray_len"),
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        shots = sorted(ctx.session.shots_for(ctx.slot), key=lambda s: s.t)
        sprays = _spray_groups(shots)
        out["rc_n_sprays"] = float(len(sprays))
        out.update(safe_stats(np.array([len(s) for s in sprays], dtype=float), "rc_spray_len"))
        if len(sprays) < 2:
            return out

        all_rec, all_vw = [], []
        traces_p, traces_y = [], []
        gains = []
        for sp in sprays:
            rec, vw = _spray_traces(ctx, sp)
            if rec.shape[0] < MIN_SPRAY - 1:
                continue
            all_rec.append(rec)
            all_vw.append(vw)
            traces_p.append(_resample_trace(vw[:, 0]))
            traces_y.append(_resample_trace(vw[:, 1]))
            # Per-spray gain, for the stability statistic.
            denom = float((rec[:, 0] ** 2).sum())
            if denom > 1e-9:
                gains.append(float(-(rec[:, 0] * vw[:, 0]).sum() / denom))

        if len(all_rec) < 2:
            return out

        R = np.vstack(all_rec)
        V = np.vstack(all_vw)

        # Compensation gain: view delta ~ -gain * recoil delta, per axis.
        for axis, key in ((0, "rc_gain_pitch"), (1, "rc_gain_yaw")):
            d = float((R[:, axis] ** 2).sum())
            if d > 1e-9:
                out[key] = float(-(R[:, axis] * V[:, axis]).sum() / d)

        # Joint fit quality. r2 near 1.0 with tiny residual is the tell.
        gp = out["rc_gain_pitch"] if np.isfinite(out["rc_gain_pitch"]) else 0.0
        gy = out["rc_gain_yaw"] if np.isfinite(out["rc_gain_yaw"]) else 0.0
        pred = np.stack([-gp * R[:, 0], -gy * R[:, 1]], axis=1)
        resid = V - pred
        ss_res = float((resid**2).sum())
        ss_tot = float(((V - V.mean(axis=0)) ** 2).sum())
        if ss_tot > 1e-12:
            out["rc_r2"] = float(1.0 - ss_res / ss_tot)
        out["rc_residual_rms"] = float(np.sqrt((resid**2).sum(axis=1).mean()))
        scale = float(np.sqrt((R**2).sum(axis=1).mean()))
        if scale > 1e-9:
            out["rc_residual_norm"] = out["rc_residual_rms"] / scale

        # Lag: cross-correlate the pitch recoil sequence against the pitch view
        # sequence over shot index. A human lags by ~1 shot; a memory-read RCS
        # compensates on the same shot or one ahead.
        a = R[:, 0] - R[:, 0].mean()
        b = -(V[:, 0] - V[:, 0].mean())
        if a.size >= 6 and a.std() > 1e-9 and b.std() > 1e-9:
            xc = np.correlate(b, a, mode="full")
            lags = np.arange(-a.size + 1, a.size)
            m = np.abs(lags) <= 4
            out["rc_lag_shots"] = float(lags[m][int(np.argmax(xc[m]))])

        # Cross-spray self-similarity.
        TP = np.vstack(traces_p)
        TY = np.vstack(traces_y)
        T = np.hstack([TP, TY])
        T = T[np.isfinite(T).all(axis=1)]
        if T.shape[0] >= 2:
            Tc = T - T.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(Tc, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1e-12
            U = Tc / norms
            C = U @ U.T
            iu = np.triu_indices(C.shape[0], k=1)
            vals = C[iu]
            out["rc_cross_spray_corr"] = float(vals.mean())
            out["rc_cross_spray_corr_std"] = float(vals.std())
            mag = float(np.abs(T).mean())
            out["rc_trace_std_norm"] = float(T.std(axis=0).mean() / (mag + 1e-9))

        if len(gains) >= 3:
            g = np.asarray(gains)
            out["rc_gain_stability"] = float(g.std() / (abs(g.mean()) + 1e-9))

        # Residual normality. A hand leaves autocorrelated, skewed residuals;
        # an RCS with added jitter leaves textbook Gaussian white noise.
        r = resid.ravel()
        r = r[np.isfinite(r)]
        if r.size >= 30:
            from scipy import stats

            try:
                out["rc_residual_gauss_p"] = float(stats.normaltest(r).pvalue)
            except Exception:
                pass
        return out
