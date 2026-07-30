"""Probability calibration and conformal decision gates.

Raw model scores are not probabilities. HistGBDT on imbalanced data and a
BCE-trained temporal net both systematically mis-scale. Before fusion turns a
score into a log-likelihood ratio we need:

  1. **Calibration** -- isotonic regression fitted on a held-out calibration
     set so that ``P(cheat | score)`` is honest in the large-sample sense.
  2. **Conformal gate** -- a distribution-free upper bound on the false
     positive rate at a chosen operating point. We only allow the policy to
     auto-ban when the conformal p-value for "this player is clean" falls
     below the system-wide alpha. That is what makes the ``FusionConfig.alpha``
     budget *mean something* rather than being a hopeful threshold.

Both are fit on real (or synthetic-but-honest) negatives. Never fit the
calibrator on the training set of the model it calibrates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilityCalibrator:
    """Maps raw scores in [0,1] (or any monotone) to calibrated probabilities."""

    iso: IsotonicRegression
    method: str = "isotonic"

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)
        iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        iso.fit(scores, labels)
        return cls(iso=iso)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        return self.iso.predict(scores).astype(np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        joblib.dump({"iso": self.iso, "method": self.method}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        blob = joblib.load(path)
        return cls(iso=blob["iso"], method=blob.get("method", "isotonic"))


def prob_to_llr(p: np.ndarray, prior: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    """Convert a calibrated P(cheat | evidence) into a log-likelihood ratio.

    LLR = log P(e|cheat) / P(e|clean). With a calibrated posterior and a known
    prior, Bayes gives LLR = logit(p) - logit(prior). Clipped for numerical
    safety; the fusion layer applies its own per-unit clip on top.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    prior = float(np.clip(prior, eps, 1.0 - eps))
    logit_p = np.log(p) - np.log(1.0 - p)
    logit_prior = np.log(prior) - np.log(1.0 - prior)
    return logit_p - logit_prior


@dataclass
class ConformalGate:
    """Split-conformal p-values for the null "this player is clean".

    Fit on a calibration set of *confirmed clean* scores. At serve time,
    ``pvalue(score)`` is the fraction of calibration clean scores that are
    >= the observed score (higher score = more cheat-like). An auto-ban is
    only admissible when pvalue < alpha.
    """

    clean_scores: np.ndarray
    alpha: float = 1e-5

    @classmethod
    def fit(cls, clean_scores: np.ndarray, alpha: float = 1e-5) -> "ConformalGate":
        s = np.sort(np.asarray(clean_scores, dtype=np.float64).ravel())
        if s.size < 50:
            raise ValueError("conformal gate needs >= 50 clean calibration scores")
        return cls(clean_scores=s, alpha=alpha)

    def pvalue(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        # (#{cal >= score} + 1) / (n + 1) -- the standard smoothed p-value.
        n = self.clean_scores.size
        # searchsorted on ascending array: index of first element > score,
        # so n - idx = count of elements >= roughly... use >= via side.
        idx = np.searchsorted(self.clean_scores, scores, side="left")
        ge = n - idx
        return (ge + 1.0) / (n + 1.0)

    def admits_ban(self, scores: np.ndarray) -> np.ndarray:
        return self.pvalue(scores) < self.alpha

    def threshold(self) -> float:
        """Score threshold corresponding roughly to alpha (finite-sample)."""
        n = self.clean_scores.size
        # Want (ge+1)/(n+1) < alpha => ge < alpha*(n+1) - 1
        k = max(0, int(np.floor(self.alpha * (n + 1) - 1)))
        # k clean scores are allowed above the threshold; take the (n-k)th.
        return float(self.clean_scores[max(0, n - k - 1)])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path, clean_scores=self.clean_scores, alpha=np.array([self.alpha]))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalGate":
        blob = np.load(path)
        return cls(clean_scores=blob["clean_scores"], alpha=float(blob["alpha"][0]))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not np.any(m):
            continue
        ece += float(np.abs(y[m].mean() - p[m].mean()) * m.mean())
    return float(ece)
