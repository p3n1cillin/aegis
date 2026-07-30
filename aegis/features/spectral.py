"""Spectral and fractal signature of the view-angle signal.

This is the channel that catches cheats no behavioural heuristic will:
the *texture* of the signal itself.

A human holding a mouse produces:
  * physiological tremor, a broad peak at 8-12 Hz, always present, that
    survives even when the hand is nominally still;
  * 1/f^beta pink noise from the neuromuscular system, beta ~ 1.0-1.6;
  * long-range temporal correlation, DFA alpha ~ 0.7-1.0 (persistent);
  * cross-axis coupling -- pitch and yaw are not independent, because they
    come from the same wrist and forearm.

Synthetic input produces none of these unless the cheat author deliberately
models them, and the humanisers that ship in real cheats add *white* jitter,
which pushes spectral flatness toward 1.0 and DFA alpha toward 0.5 -- the
opposite direction from a real hand. That makes a naive humaniser more
detectable here than no humaniser at all.

Robustness note: these features are computed on *idle and tracking* segments,
never on flicks, because a large ballistic movement dominates any PSD.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..core.registry import EXTRACTORS, Capability
from .base import PlayerContext, SessionExtractor

TREMOR_BAND = (7.0, 13.0)
MIN_SEGMENT = 96  # samples


def _idle_segments(ctx: PlayerContext, min_len: int = MIN_SEGMENT) -> list[slice]:
    """Contiguous stretches where the player is alive and not flicking.

    We want the micro-motion floor, so we exclude anything above a modest
    angular speed, plus freezetime and death.
    """
    v = ctx.view
    if len(v) < min_len:
        return []
    ok = (v.speed < 120.0) & ctx.alive_mask()
    from ..core.types import PlayerFlag

    ok &= ~ctx.flag_mask(PlayerFlag.FROZEN)

    segs: list[slice] = []
    i = 0
    n = ok.size
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= min_len:
            segs.append(slice(i, j))
        i = j
    return segs


def _welch_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nper = min(256, x.size)
    if nper < 32:
        return np.empty(0), np.empty(0)
    f, p = sps.welch(x - x.mean(), fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear")
    return f, p


def _spectral_slope(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    """beta in P(f) ~ 1/f^beta, fitted by least squares in log-log."""
    m = (f >= lo) & (f <= hi) & (p > 0)
    if m.sum() < 6:
        return float("nan")
    lf = np.log10(f[m])
    lp = np.log10(p[m])
    slope = np.polyfit(lf, lp, 1)[0]
    return float(-slope)


def _spectral_flatness(p: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean of the PSD.

    1.0 = white noise, ~0 = a pure tone. Injected uniform jitter drives this up.
    """
    p = p[p > 0]
    if p.size < 8:
        return float("nan")
    return float(np.exp(np.log(p).mean()) / (p.mean() + 1e-30))


def _band_ratio(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    if f.size == 0:
        return float("nan")
    tot = np.trapezoid(p, f)
    if tot <= 0:
        return float("nan")
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapezoid(p[m], f[m]) / tot)


def _dfa_alpha(x: np.ndarray, scales: np.ndarray | None = None) -> float:
    """Detrended fluctuation analysis exponent.

    alpha ~ 0.5 uncorrelated (white), ~1.0 pink / long-range correlated,
    ~1.5 brownian. Human motor output sits near 0.75-1.05. This is one of the
    few statistics that is stable across sensitivity settings, DPI, and
    framerate, which makes it excellent for cross-game transfer.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 128:
        return float("nan")
    y = np.cumsum(x - x.mean())
    if scales is None:
        scales = np.unique(np.round(np.logspace(np.log10(8), np.log10(max(16, n // 8)), 12)).astype(int))
    fluct = []
    used = []
    for s in scales:
        if s < 4 or s > n // 4:
            continue
        nseg = n // s
        seg = y[: nseg * s].reshape(nseg, s)
        idx = np.arange(s)
        # Least-squares detrend each window (order 1).
        coeffs = np.polyfit(idx, seg.T, 1)
        trend = np.outer(coeffs[0], idx) + coeffs[1][:, None]
        rms = np.sqrt(((seg - trend) ** 2).mean(axis=1))
        f = np.sqrt((rms**2).mean())
        if f > 0:
            fluct.append(f)
            used.append(s)
    if len(used) < 4:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


def _sample_entropy(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """Sample entropy -- regularity of the series.

    Deterministic interpolation is highly regular (low SampEn); a real hand is
    not. Subsampled to keep this O(n^2) statistic affordable.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size > 600:
        x = x[:: x.size // 600 + 1]
    n = x.size
    if n < 64:
        return float("nan")
    r = r_mult * x.std()
    if r <= 0:
        return float("nan")

    def _count(mm: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        # Chebyshev distance between all pairs.
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        np.fill_diagonal(d, np.inf)
        return int((d <= r).sum())

    a = _count(m + 1)
    b = _count(m)
    if a == 0 or b == 0:
        return float("nan")
    return float(-np.log(a / b))


@EXTRACTORS.register("spectral")
class SpectralFeatures(SessionExtractor):
    name = "spectral"
    requires = (Capability.VIEW_ANGLES, Capability.HIGH_RATE_VIEW)
    names = (
        "sp_tremor_ratio_yaw",
        "sp_tremor_ratio_pitch",
        "sp_slope_yaw",
        "sp_slope_pitch",
        "sp_flatness_yaw",
        "sp_flatness_pitch",
        "sp_peak_freq_yaw",
        "sp_dfa_yaw",
        "sp_dfa_pitch",
        "sp_axis_coupling",
        "sp_sample_entropy_yaw",
        "sp_micro_motion_rms",
        "sp_zero_motion_fraction",
        "sp_idle_coverage",
        "sp_hf_power_ratio",
    )

    def extract(self, ctx: PlayerContext) -> dict[str, float]:
        out = dict.fromkeys(self.names, float("nan"))
        v = ctx.view
        segs = _idle_segments(ctx)
        if not segs:
            out["sp_idle_coverage"] = 0.0
            return out

        out["sp_idle_coverage"] = float(sum(s.stop - s.start for s in segs) / max(len(v), 1))
        fs = v.hz

        yaw_parts, pitch_parts = [], []
        psd_y, psd_p, freqs = [], [], None
        for s in segs:
            yv = v.yaw_v[s]
            pv = v.pitch_v[s]
            yaw_parts.append(yv)
            pitch_parts.append(pv)
            f, py = _welch_psd(yv, fs)
            _, pp = _welch_psd(pv, fs)
            if f.size:
                freqs = f
                psd_y.append(py)
                psd_p.append(pp)

        if freqs is None or not psd_y:
            return out

        py = np.mean(psd_y, axis=0)
        pp = np.mean(psd_p, axis=0)
        nyq = fs / 2

        out["sp_tremor_ratio_yaw"] = _band_ratio(freqs, py, *TREMOR_BAND)
        out["sp_tremor_ratio_pitch"] = _band_ratio(freqs, pp, *TREMOR_BAND)
        out["sp_slope_yaw"] = _spectral_slope(freqs, py, 1.0, min(40.0, nyq * 0.8))
        out["sp_slope_pitch"] = _spectral_slope(freqs, pp, 1.0, min(40.0, nyq * 0.8))
        out["sp_flatness_yaw"] = _spectral_flatness(py)
        out["sp_flatness_pitch"] = _spectral_flatness(pp)
        band = (freqs >= 2.0) & (freqs <= min(40.0, nyq * 0.9))
        if band.sum() > 2:
            out["sp_peak_freq_yaw"] = float(freqs[band][np.argmax(py[band])])
        out["sp_hf_power_ratio"] = _band_ratio(freqs, py, min(30.0, nyq * 0.6), nyq * 0.95)

        yv_all = np.concatenate(yaw_parts)
        pv_all = np.concatenate(pitch_parts)
        out["sp_dfa_yaw"] = _dfa_alpha(yv_all)
        out["sp_dfa_pitch"] = _dfa_alpha(pv_all)
        out["sp_sample_entropy_yaw"] = _sample_entropy(yv_all)

        # Cross-axis coupling: |corr| between |pitch velocity| and |yaw
        # velocity|. A wrist produces coupled motion; two independent noise
        # generators produce ~0.
        if yv_all.size > 64 and np.std(yv_all) > 1e-9 and np.std(pv_all) > 1e-9:
            out["sp_axis_coupling"] = float(abs(np.corrcoef(np.abs(yv_all), np.abs(pv_all))[0, 1]))

        mag = np.hypot(yv_all, pv_all)
        out["sp_micro_motion_rms"] = float(np.sqrt((mag**2).mean()))
        # Exactly-zero motion is the giveaway for a client that only writes the
        # view angle when the cheat updates it.
        out["sp_zero_motion_fraction"] = float((mag < 1e-4).mean())
        return out
