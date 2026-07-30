"""HID-level mouse input analysis.

Physical USB/HID mice emit reports on a hardware schedule (125/250/500/1000 Hz
polling) with characteristic jitter from the sensor and the host controller.
Injected input -- SendInput, raw device driver hooks, external hardware
spoofers that are not careful -- produces one of:

  * Exact period with near-zero jitter (software loop on a multimedia timer)
  * Quantised micro-jitter at the OS scheduler quantum (~15.6 ms on older
    Windows, ~1 ms with high-res timers) that real sensors do not show
  * dx/dy sequences with impossible acceleration profiles (instantaneous
    direction changes at constant speed -- a solver, not a hand)
  * Device sequence numbers that skip or reset

These features are only trusted when the attestation layer says the client
module is intact; otherwise a cheater could forge "human-looking" HID.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import InputSample


@dataclass
class HIDReport:
    n_samples: int
    poll_hz_est: float
    period_jitter_cv: float  # coef of variation of inter-report times
    quantisation_score: float  # 0..1, higher = more scheduler-quantised
    accel_spike_rate: float
    seq_gap_rate: float
    synthetic_score: float  # calibrated-ish 0..1 P(injected)
    reasons: list[str]


class HIDAnalyzer:
    """Score a player's HID stream for injection artefacts."""

    def analyze(self, samples: list[InputSample]) -> HIDReport:
        if len(samples) < 64:
            return HIDReport(
                n_samples=len(samples), poll_hz_est=0.0, period_jitter_cv=0.0,
                quantisation_score=0.0, accel_spike_rate=0.0, seq_gap_rate=0.0,
                synthetic_score=0.0, reasons=["insufficient_samples"],
            )

        t = np.array([s.t for s in samples], dtype=np.float64)
        dx = np.array([s.dx for s in samples], dtype=np.float64)
        dy = np.array([s.dy for s in samples], dtype=np.float64)
        seq = np.array([s.device_seq for s in samples], dtype=np.int64)

        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size < 32:
            return HIDReport(
                n_samples=len(samples), poll_hz_est=0.0, period_jitter_cv=0.0,
                quantisation_score=0.0, accel_spike_rate=0.0, seq_gap_rate=0.0,
                synthetic_score=0.0, reasons=["bad_timestamps"],
            )

        med = float(np.median(dt))
        poll = 1.0 / med if med > 0 else 0.0
        jitter_cv = float(np.std(dt) / med) if med > 0 else 0.0

        # Quantisation: fraction of inter-report times that land near multiples
        # of 1 ms or 15.625 ms (legacy Windows quantum).
        def near_mult(x, base, tol=0.08):
            q = np.round(x / base) * base
            return np.mean(np.abs(x - q) < tol * base)

        q1 = float(near_mult(dt, 0.001))
        q15 = float(near_mult(dt, 0.015625))
        quant = max(q1, q15)

        # Acceleration spikes: angle of velocity vector changes by >90 deg
        # between consecutive non-zero reports while speed stays high.
        vx, vy = dx[1:], dy[1:]
        speed = np.hypot(vx, vy)
        # Dot product of consecutive velocity vectors.
        dots = vx[1:] * vx[:-1] + vy[1:] * vy[:-1]
        mags = speed[1:] * speed[:-1]
        cos = np.divide(dots, mags, out=np.ones_like(dots), where=mags > 1e-6)
        sharp = (cos < 0) & (speed[1:] > np.quantile(speed, 0.7))
        accel_spike_rate = float(np.mean(sharp)) if sharp.size else 0.0

        dseq = np.diff(seq)
        # Expected +1 each report; gaps or zeros are suspicious.
        seq_gap_rate = float(np.mean(dseq != 1)) if dseq.size else 0.0

        reasons = []
        score = 0.0
        # Near-zero jitter at a round poll rate is the classic software inject.
        if jitter_cv < 0.02 and 100 <= poll <= 1100:
            score += 0.45
            reasons.append(f"near_zero_jitter cv={jitter_cv:.4f} @ {poll:.0f}Hz")
        if quant > 0.85 and jitter_cv < 0.08:
            score += 0.25
            reasons.append(f"scheduler_quantisation={quant:.2f}")
        if accel_spike_rate > 0.12:
            score += 0.20
            reasons.append(f"accel_spike_rate={accel_spike_rate:.2f}")
        if seq_gap_rate > 0.05:
            score += 0.15
            reasons.append(f"seq_gap_rate={seq_gap_rate:.2f}")
        # Real sensors at 1000 Hz still have CV ~0.03-0.10; very high CV can
        # indicate timestamp forgery rather than honesty.
        if jitter_cv > 0.5:
            score += 0.10
            reasons.append(f"excessive_jitter cv={jitter_cv:.2f}")

        return HIDReport(
            n_samples=len(samples),
            poll_hz_est=poll,
            period_jitter_cv=jitter_cv,
            quantisation_score=quant,
            accel_spike_rate=accel_spike_rate,
            seq_gap_rate=seq_gap_rate,
            synthetic_score=float(min(score, 1.0)),
            reasons=reasons or ["looks_physical"],
        )
