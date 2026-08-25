"""Online learning from human review labels.

Production loop (shadow → review → learn):

  1. Ingest scores a session; policy emits ``review`` (or shadow ``ban``).
  2. A human labels the player clean / cheat (+ optional family).
  3. This module folds the label into:
       * isotonic calibrators (tabular ± temporal) on an expanding cal set
       * novelty AE fine-tune when the label is **clean**
       * optional conformal gate refresh on clean scores
  4. Artefacts are rewritten under ``model_dir`` with the same FeatureSpec
     digest — a digest change is a hard error (retrain from scratch instead).

Never put trust/rank into the feature vector here. Never auto-lower ban
thresholds from synthetic-only refreshes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.config import AegisConfig
from ..models.calibrate import ConformalGate, ProbabilityCalibrator, expected_calibration_error
from ..models.ensemble import CalibratedEnsemble
from ..models.novelty import NoveltyAE

# Minimum review labels before an online refresh may replace the
# bootstrap-fitted calibrators -- same floor as refresh_from_reviews'
# min_cal_rows. The /v1/reviews/refresh endpoint refuses below this rather
# than fitting an operating point on a handful of rows.
MIN_LABELS_FOR_REFRESH = 30


@dataclass
class ReviewLabel:
    player_id: str
    session_id: str
    y: int  # 0 clean, 1 cheat
    family: str = ""
    features: list[float] = field(default_factory=list)
    raw_tabular: float | None = None
    raw_temporal: float | None = None
    p_fused: float | None = None
    reviewer: str = ""
    t: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class OnlineStore:
    """Append-only review log + feature cache used for refreshes."""

    path: Path
    labels: list[ReviewLabel] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "OnlineStore":
        path = Path(path)
        obj = cls(path=path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                obj.labels.append(ReviewLabel(**raw))
        return obj

    def append(self, label: ReviewLabel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(label)) + "\n")
        self.labels.append(label)


@dataclass
class RefreshReport:
    n_labels: int
    n_clean: int
    n_cheat: int
    tabular_ece: float | None = None
    novelty_threshold_p99: float | None = None
    conformal_threshold: float | None = None
    digest: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def refresh_from_reviews(
    ensemble: CalibratedEnsemble,
    store: OnlineStore,
    *,
    out_dir: str | Path | None = None,
    cfg: AegisConfig | None = None,
    min_clean_for_novelty: int = 20,
    min_cal_rows: int = 30,
) -> tuple[CalibratedEnsemble, RefreshReport]:
    """Refit calibrators + novelty (clean-only) from accumulated reviews.

    The GBDT / temporal encoder weights are left untouched — those need a full
    ``aegis train`` when the feature space or cheat mixture shifts materially.
    """
    cfg = cfg or AegisConfig()
    out_dir = Path(out_dir or cfg.model_dir)
    labels = [lb for lb in store.labels if lb.features]
    if not labels:
        raise RuntimeError("no review labels with feature vectors")

    X = np.asarray([lb.features for lb in labels], dtype=np.float32)
    y = np.asarray([lb.y for lb in labels], dtype=np.int32)
    if X.shape[1] != len(ensemble.spec.columns):
        raise ValueError(
            f"feature width {X.shape[1]} != spec {len(ensemble.spec.columns)}; "
            "refusing to refresh across a FeatureSpec change"
        )

    report = RefreshReport(
        n_labels=int(y.size),
        n_clean=int((y == 0).sum()),
        n_cheat=int((y == 1).sum()),
        digest=ensemble.digest,
    )

    # ---- tabular calibrator --------------------------------------------
    raw_tab = np.asarray(
        [
            lb.raw_tabular if lb.raw_tabular is not None
            else float(ensemble.bundle.tabular.predict_proba(X[i:i + 1])[0])
            for i, lb in enumerate(labels)
        ],
        dtype=np.float64,
    )
    if y.size >= min_cal_rows and y.min() != y.max():
        cal_tab = ProbabilityCalibrator.fit(raw_tab, y)
        ensemble.bundle.cal_tabular = cal_tab
        report.tabular_ece = expected_calibration_error(cal_tab.transform(raw_tab), y)

    # ---- temporal calibrator (only if raws were logged) ----------------
    temp_pairs = [
        (lb.raw_temporal, lb.y) for lb in labels if lb.raw_temporal is not None
    ]
    if ensemble.bundle.cal_temporal is not None and len(temp_pairs) >= min_cal_rows:
        rt = np.asarray([p[0] for p in temp_pairs], dtype=np.float64)
        yt = np.asarray([p[1] for p in temp_pairs], dtype=np.int32)
        if yt.min() != yt.max():
            ensemble.bundle.cal_temporal = ProbabilityCalibrator.fit(rt, yt)

    # ---- novelty fine-tune on cleans -----------------------------------
    clean = X[y == 0]
    if clean.shape[0] >= min_clean_for_novelty:
        nov = NoveltyAE.create(ensemble.spec, ensemble.bundle.novelty.cfg)
        hist = nov.fit(clean, epochs=min(ensemble.bundle.novelty.cfg.epochs, 12))
        ensemble.bundle.novelty = nov
        report.novelty_threshold_p99 = float(hist["threshold_p99"])

    # ---- conformal on calibrated clean scores --------------------------
    cal_scores = ensemble.bundle.cal_tabular.transform(
        ensemble.bundle.tabular.predict_proba(X)
    )
    clean_scores = cal_scores[y == 0]
    if clean_scores.size >= 8:
        # Pad tiny review sets the same way bootstrap train does.
        if clean_scores.size < 50:
            rng = np.random.default_rng(cfg.model.seed)
            clean_scores = np.concatenate([
                clean_scores,
                np.clip(
                    rng.normal(
                        float(clean_scores.mean()),
                        max(float(clean_scores.std()), 1e-3),
                        size=50 - clean_scores.size,
                    ),
                    0.0,
                    1.0,
                ),
            ])
        ensemble.bundle.conformal = ConformalGate.fit(
            clean_scores, alpha=cfg.fusion.alpha
        )
        report.conformal_threshold = ensemble.bundle.conformal.threshold()

    ensemble.save(out_dir)
    (out_dir / "online_refresh.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return ensemble, report


__all__ = [
    "ReviewLabel",
    "OnlineStore",
    "RefreshReport",
    "refresh_from_reviews",
]
