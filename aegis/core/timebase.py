"""Tick/time normalisation and signal resampling.

Games disagree about tickrate (CS2 subtick, 64/128 tick servers, 30 Hz
snapshots in other titles) and about whether the recorded view angle is the
one used for the shot or the one after it. Everything downstream assumes a
uniform grid at ``FeatureConfig.resample_hz``, so this module is where the
mess is contained.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .types import wrap_deg


def uniform_grid(t: np.ndarray, hz: float) -> np.ndarray:
    """Uniform time grid covering ``t``."""
    if t.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.arange(t[0], t[-1] + 0.5 / hz, 1.0 / hz, dtype=np.float64)


def resample_linear(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear resample of a non-angular signal onto ``grid``."""
    if t.size == 0 or grid.size == 0:
        return np.zeros_like(grid)
    return np.interp(grid, t, y)


def resample_angle(t: np.ndarray, deg: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample an angular signal without wrap artefacts.

    Interpolating yaw directly across the +-180 seam invents a 360 deg/frame
    spike, which reads as a spinbot. Unwrap first, interpolate, re-wrap.
    """
    if t.size == 0 or grid.size == 0:
        return np.zeros_like(grid)
    unwrapped = np.degrees(np.unwrap(np.radians(np.asarray(deg, dtype=np.float64))))
    return wrap_deg(np.interp(grid, t, unwrapped))


def unwrap_deg(deg: np.ndarray) -> np.ndarray:
    """Continuous (non-wrapped) version of an angular series."""
    return np.degrees(np.unwrap(np.radians(np.asarray(deg, dtype=np.float64))))


def derivatives(y: np.ndarray, dt: float, window: int = 9, poly: int = 3) -> tuple[np.ndarray, ...]:
    """Smoothed 1st/2nd/3rd derivatives via Savitzky-Golay.

    Finite differences on tick data are dominated by quantisation noise (the
    view angle is transmitted at limited precision), and the discriminative
    content for aim analysis lives in *jerk* -- the third derivative -- which
    naive differencing destroys entirely. SG fits a local polynomial and
    differentiates that, which is both smoother and unbiased for polynomial
    motion.
    """
    n = y.size
    if n < 3:
        z = np.zeros(n)
        return z, z.copy(), z.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        w = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
        if w > n:
            # Too short for SG; fall back to gradients.
            d1 = np.gradient(y, dt)
            d2 = np.gradient(d1, dt)
            d3 = np.gradient(d2, dt)
            return d1, d2, d3
    d1 = savgol_filter(y, w, poly, deriv=1, delta=dt, mode="interp")
    d2 = savgol_filter(y, w, poly, deriv=2, delta=dt, mode="interp")
    d3 = savgol_filter(y, w, min(poly, w - 2), deriv=3, delta=dt, mode="interp") if w >= poly + 2 else np.gradient(d2, dt)
    return d1, d2, d3


class ViewSignal:
    """Resampled, differentiated view-angle signal for one player.

    This object is built once per player per session and shared by every
    kinematic extractor, because SG filtering the whole match is the single
    most expensive step in the pipeline.
    """

    __slots__ = (
        "t", "pitch", "yaw", "hz", "dt",
        "pitch_u", "yaw_u",
        "pitch_v", "yaw_v", "speed",
        "pitch_a", "yaw_a", "accel",
        "pitch_j", "yaw_j", "jerk",
    )

    def __init__(self, t: np.ndarray, pitch: np.ndarray, yaw: np.ndarray, hz: float,
                 sg_window: int = 9, sg_poly: int = 3):
        grid = uniform_grid(np.asarray(t, dtype=np.float64), hz)
        self.t = grid
        self.hz = hz
        self.dt = 1.0 / hz
        self.pitch = resample_angle(t, pitch, grid)
        self.yaw = resample_angle(t, yaw, grid)
        # Unwrapped versions are what derivatives must be taken on.
        self.pitch_u = np.interp(grid, np.asarray(t, dtype=np.float64), unwrap_deg(pitch)) if len(t) else grid * 0
        self.yaw_u = np.interp(grid, np.asarray(t, dtype=np.float64), unwrap_deg(yaw)) if len(t) else grid * 0

        pv, pa, pj = derivatives(self.pitch_u, self.dt, sg_window, sg_poly)
        yv, ya, yj = derivatives(self.yaw_u, self.dt, sg_window, sg_poly)
        # Yaw degrees are not isotropic with pitch: near the poles a degree of
        # yaw is a smaller angular move. Scale by cos(pitch) so "angular speed"
        # is a true on-sphere speed and a spinbot at pitch=-89 is not inflated.
        cp = np.cos(np.radians(np.clip(self.pitch, -89.5, 89.5)))
        self.pitch_v, self.yaw_v = pv, yv * cp
        self.pitch_a, self.yaw_a = pa, ya * cp
        self.pitch_j, self.yaw_j = pj, yj * cp
        self.speed = np.hypot(self.pitch_v, self.yaw_v)
        self.accel = np.hypot(self.pitch_a, self.yaw_a)
        self.jerk = np.hypot(self.pitch_j, self.yaw_j)

    def __len__(self) -> int:
        return self.t.size

    def window(self, t0: float, t1: float) -> slice:
        """Index slice covering [t0, t1]."""
        i0 = int(np.searchsorted(self.t, t0, side="left"))
        i1 = int(np.searchsorted(self.t, t1, side="right"))
        return slice(max(0, i0), min(self.t.size, i1))
