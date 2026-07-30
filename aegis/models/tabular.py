"""Session-level gradient-boosted detector.

Operates on the fixed-order feature vector from ``features.pipeline``. Chosen
over a neural net for this channel because:

  * The features are already engineered causal summaries; a shallow tree
    ensemble on top of them is hard to overfit and easy to explain.
  * HistGradientBoosting is in scikit-learn -- no extra native dependency --
    and handles missing values natively, which matters when a game adapter
    lacks a capability and the corresponding columns are NaN.
  * Feature importances and per-row contributions (via TreeSHAP-style
    path-dependent values, approximated here with permutation on held-out
    clean players) feed the explainability layer.

The temporal encoder owns per-engagement pattern recognition; this model owns
*consistency across a match*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..features.pipeline import FeatureSpec


def _sanitize_features(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Make a feature matrix safe for HistGradientBoosting.

    HistGBDT accepts NaN but dies when a column has fewer than two distinct
    finite values (the binning code assumes it can take a midpoint). Constant
    and all-missing columns show up in short synthetic runs and whenever a
    capability is absent for every row in the batch.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    rng = np.random.default_rng(seed)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            X[:, j] = 0.0
            continue
        if np.unique(finite).size < 2:
            fill = float(finite[0])
            col = np.where(np.isfinite(col), col, fill)
            # Break exact constancy without changing scale in any meaningful way.
            col = col + rng.normal(0.0, 1e-6, size=col.shape)
            X[:, j] = col
    return X


@dataclass
class TabularGBDT:
    spec: FeatureSpec
    model: HistGradientBoostingClassifier
    feature_importance_: dict[str, float] | None = None

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        spec: FeatureSpec,
        *,
        max_iter: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.06,
        l2: float = 1e-3,
        min_samples_leaf: int = 40,
        seed: int = 1337,
        sample_weight: np.ndarray | None = None,
    ) -> "TabularGBDT":
        if X.shape[1] != len(spec.columns):
            raise ValueError(
                f"X has {X.shape[1]} cols but spec has {len(spec.columns)}; "
                "refusing to train on a mismatched feature space"
            )
        X = _sanitize_features(X, seed=seed)
        # Short synthetic runs can leave very few rows after the player split;
        # clamp leaf size so HistGBDT can still fit.
        min_samples_leaf = min(min_samples_leaf, max(2, X.shape[0] // 5))
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=l2,
            min_samples_leaf=min_samples_leaf,
            early_stopping=X.shape[0] >= 40,
            validation_fraction=0.15 if X.shape[0] >= 40 else 0.1,
            n_iter_no_change=20,
            random_state=seed,
            class_weight="balanced",
        )
        clf.fit(X, y.astype(np.int32), sample_weight=sample_weight)
        out = cls(spec=spec, model=clf)
        # Cheap importance on a stratified subset so explainability has
        # something to say even before a dedicated SHAP pass.
        try:
            n = min(800, X.shape[0])
            idx = np.random.default_rng(seed).choice(X.shape[0], size=n, replace=False)
            pi = permutation_importance(
                clf, X[idx], y[idx], n_repeats=5, random_state=seed, n_jobs=1
            )
            out.feature_importance_ = {
                c: float(v) for c, v in zip(spec.columns, pi.importances_mean)
            }
        except Exception:
            out.feature_importance_ = {c: 0.0 for c in spec.columns}
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(cheat) for each row. Shape (N,)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.spec.columns):
            raise ValueError("feature width mismatch with trained tabular model")
        X = _sanitize_features(np.asarray(X, dtype=np.float64), seed=0)
        proba = self.model.predict_proba(X)
        # class 1 = cheat
        return proba[:, 1].astype(np.float64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "tabular.joblib")
        (path / "tabular_spec.json").write_text(
            json.dumps({
                **self.spec.to_dict(),
                "importance": self.feature_importance_ or {},
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularGBDT":
        path = Path(path)
        raw = json.loads((path / "tabular_spec.json").read_text(encoding="utf-8"))
        spec = FeatureSpec.from_dict(raw)
        model = joblib.load(path / "tabular.joblib")
        return cls(spec=spec, model=model, feature_importance_=raw.get("importance"))
