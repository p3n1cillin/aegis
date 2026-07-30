"""HID-level input dynamics.

Available only when the Aegis client module is present and the session is
attested (see ``aegis.integrity``). When it is available it is the single
highest-signal channel in the system, because it observes the input *before*
the game applies sensitivity, smoothing, or any of the transforms that a
server-side view-angle stream has already baked in.

What a real mouse produces:
  * reports at a fixed polling interval (125/500/1000 Hz) with hardware jitter
    of tens of microseconds -- a *characteristic, device-specific* jitter, not
    a clean quantisation;
  * integer counts whose magnitude distribution has a heavy tail and a
    pronounced mode at +-1 (sensor noise while "still");
  * diagonal motion that follows the sensor's own axis coupling;
  * a strict relationship between the reported counts and the resulting
    in-game view delta, fixed by sensitivity and DPI.

An injected view change breaks the last property outright: the view moves
without a corresponding HID report. That is the check that matters, and it is
close to a hard tell rather than a statistical one -- which is why this module
computes an *unexplained motion budget* instead of only a similarity score.

Anti-spoofing note: the client cannot be trusted to report honestly, so this
module never accepts HID data unless ``session.meta['attested']`` is set by the
integrity layer, and the unexplained-motion check is deliberately one-sided --
a cheat that suppresses its own HID reports makes the residual *larger*, not
smaller.
"""

from __future__ import annotations

import numpy as np

from ..core.registry import EXTRACTORS, Capability
from .base import PlayerContext, SessionExtractor


def _poll_interval_stats(ts: np.ndarray) -> tuple[float, float, float]:
    """(modal polling interval, jitter std in us, fraction at the mode)."""
    if ts.size < 64:
        return float("nan"), float("nan"), float("nan")
    dt = np.diff(ts)
    dt = dt[(dt > 0) & (dt < 0.05)]
    if dt.size < 32:
        return float("nan"), float("nan"), float("nan")
    # Modal interval via a fine histogram rather than a mean, because dropped
    # reports create integer multiples that would drag a mean upward.
    hist, edges = np.histogram(dt, bins=400)
    mode = float((edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2)
    near = np.abs(dt - mode) < mode * 0.4
    jitter_us = float(dt[near].std() * 1e6) if near.sum() > 16 else float("nan")
    return mode, jitter_us, float(near.mean())


@EXTRACTORS.register("input_dynamics")
class InputDynamicsFeatures(SessionExtractor):
    name = "input_dynamics"
    requires = (Capability.HID_INPUT, Capability.VIEW_ANGLES)
    names = (
        "hid_poll_interval_ms",
        "hid_poll_jitter_us",
        "hid_poll_mode_frac",
        "hid_count_mode_one_frac",
        "hid_count_kurtosis",
        "hid_zero_report_frac",
        "hid_axis_coupling",
        "hid_seq_gap_rate",
        "hid_unexplained_motion_frac",
        "hid_unexplained_motion_deg",
        "hid_sensitivity_r2",
        "hid_sensitivity_drift",
        "hid_hw_timestamp_skew",
        "hid_n_samples",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        if not ctx.session.meta.get("attested", False):
            # Untrusted client: refuse to produce features rather than produce
            # features an attacker controls.
            out["hid_n_samples"] = 0.0
            return out

        samples = ctx.session.inputs.get(ctx.slot, [])
        out["hid_n_samples"] = float(len(samples))
        if len(samples) < 256:
            return out

        ts = np.array([s.t for s in samples], dtype=np.float64)
        dx = np.array([s.dx for s in samples], dtype=np.float64)
        dy = np.array([s.dy for s in samples], dtype=np.float64)
        seq = np.array([s.device_seq for s in samples], dtype=np.int64)
        hw = np.array([s.hw_timestamp_us for s in samples], dtype=np.float64)

        order = np.argsort(ts)
        ts, dx, dy, seq, hw = ts[order], dx[order], dy[order], seq[order], hw[order]

        mode, jitter, mode_frac = _poll_interval_stats(ts)
        out["hid_poll_interval_ms"] = mode * 1000.0 if np.isfinite(mode) else float("nan")
        out["hid_poll_jitter_us"] = jitter
        out["hid_poll_mode_frac"] = mode_frac

        mag = np.hypot(dx, dy)
        nz = mag > 0
        out["hid_zero_report_frac"] = float((~nz).mean())
        if nz.sum() > 64:
            m = mag[nz]
            out["hid_count_mode_one_frac"] = float((m <= 1.5).mean())
            mu, sd = m.mean(), m.std()
            out["hid_count_kurtosis"] = float(((m - mu) ** 4).mean() / (sd**4 + 1e-12)) if sd > 1e-9 else float("nan")
            if np.std(np.abs(dx[nz])) > 1e-9 and np.std(np.abs(dy[nz])) > 1e-9:
                out["hid_axis_coupling"] = float(abs(np.corrcoef(np.abs(dx[nz]), np.abs(dy[nz]))[0, 1]))

        if seq.size > 2:
            d = np.diff(seq)
            d = d[d > 0]
            out["hid_seq_gap_rate"] = float((d > 1).mean()) if d.size else float("nan")

        if hw.size > 64 and np.ptp(hw) > 0:
            # Drift between the host clock and the device clock. A synthetic
            # HID stream generated on the host has zero independent skew.
            host_us = (ts - ts[0]) * 1e6
            dev_us = hw - hw[0]
            if np.std(host_us) > 1e-6:
                slope = float(np.polyfit(host_us, dev_us, 1)[0])
                out["hid_hw_timestamp_skew"] = abs(slope - 1.0)

        # --- the important one: does the view move without HID input? ------
        v = ctx.view
        if len(v) > 64:
            # Accumulate HID counts onto the view grid.
            edges = np.concatenate([v.t, [v.t[-1] + v.dt]])
            acc_x, _ = np.histogram(ts, bins=edges, weights=dx)
            acc_y, _ = np.histogram(ts, bins=edges, weights=dy)
            dyaw = np.diff(v.yaw_u, prepend=v.yaw_u[0])
            dpitch = np.diff(v.pitch_u, prepend=v.pitch_u[0])

            # Fit degrees-per-count on samples where the player is clearly
            # moving the mouse, then look at the residual everywhere.
            m = (np.abs(acc_x) > 2) | (np.abs(acc_y) > 2)
            if m.sum() >= 128:
                A = np.stack([acc_x[m], acc_y[m]], axis=1)
                B = np.stack([dyaw[m], dpitch[m]], axis=1)
                sol, *_ = np.linalg.lstsq(A, B, rcond=None)
                pred_all = np.stack([acc_x, acc_y], axis=1) @ sol
                resid = np.stack([dyaw, dpitch], axis=1) - pred_all
                rmag = np.hypot(resid[:, 0], resid[:, 1])
                total = np.hypot(dyaw, dpitch)
                moved = total > 0.02
                if moved.sum() > 64:
                    # Fraction of frames where >50% of the view change is
                    # unaccounted for by any HID report.
                    out["hid_unexplained_motion_frac"] = float(
                        (rmag[moved] > 0.5 * total[moved]).mean()
                    )
                    out["hid_unexplained_motion_deg"] = float(rmag[moved].sum())
                pred = A @ sol
                ss_res = float(((B - pred) ** 2).sum())
                ss_tot = float(((B - B.mean(axis=0)) ** 2).sum())
                if ss_tot > 1e-12:
                    out["hid_sensitivity_r2"] = float(1.0 - ss_res / ss_tot)

                # Sensitivity should be constant. A cheat that scales its
                # injected movement to match the player's sensitivity still
                # drifts, because it does not know the pitch-dependent scaling.
                half = m.sum() // 2
                if half >= 64:
                    idx = np.where(m)[0]
                    s1, *_ = np.linalg.lstsq(
                        np.stack([acc_x[idx[:half]], acc_y[idx[:half]]], axis=1),
                        np.stack([dyaw[idx[:half]], dpitch[idx[:half]]], axis=1), rcond=None)
                    s2, *_ = np.linalg.lstsq(
                        np.stack([acc_x[idx[half:]], acc_y[idx[half:]]], axis=1),
                        np.stack([dyaw[idx[half:]], dpitch[idx[half:]]], axis=1), rcond=None)
                    denom = np.abs(s1).mean() + 1e-9
                    out["hid_sensitivity_drift"] = float(np.abs(s1 - s2).mean() / denom)
        return out
