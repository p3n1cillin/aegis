"""Training orchestration.

Builds a ``CalibratedEnsemble`` from a ``Dataset``:

  1. Fit tabular GBDT on session features (train split)
  2. Fit temporal encoder on engagement tensors (train split)
  3. Fit novelty AE on *clean-only* train rows
  4. Calibrate tabular + temporal on the calibration split
  5. Fit conformal gate on clean calibration scores
  6. Persist everything under ``model_dir``
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..core.config import AegisConfig, ModelConfig
from ..models.calibrate import ConformalGate, ProbabilityCalibrator, expected_calibration_error
from ..models.ensemble import CalibratedEnsemble, ModelBundle
from ..models.novelty import NoveltyAE
from ..models.tabular import TabularGBDT
from ..models.temporal import TemporalEncoder
from .dataset import Dataset, generate_dataset, split_by_player
from .evaluate import EvaluationResult, evaluate_ensemble


def train_ensemble(
    ds: Dataset,
    cfg: AegisConfig | None = None,
    *,
    out_dir: str | Path | None = None,
    skip_temporal: bool = False,
) -> tuple[CalibratedEnsemble, dict]:
    cfg = cfg or AegisConfig()
    out_dir = Path(out_dir or cfg.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, cal, test = split_by_player(ds, seed=cfg.model.seed)
    report: dict = {
        "n_train": int(train.y.size), "n_cal": int(cal.y.size), "n_test": int(test.y.size),
        "n_train_eng": int(train.eng_y.size), "cheat_rate_train": float(train.y.mean()) if train.y.size else 0.0,
    }

    # ---- tabular --------------------------------------------------------
    tab = TabularGBDT.train(train.X, train.y, ds.spec, seed=cfg.model.seed)
    raw_cal_tab = tab.predict_proba(cal.X)
    cal_tab = ProbabilityCalibrator.fit(raw_cal_tab, cal.y)
    report["tabular_ece"] = expected_calibration_error(cal_tab.transform(raw_cal_tab), cal.y)

    # ---- temporal -------------------------------------------------------
    temporal = None
    cal_temp = None
    if not skip_temporal and train.eng_y.size >= 64:
        temporal = TemporalEncoder.create(cfg.model)
        # Fewer epochs by default on CPU for the synthetic bootstrap.
        mcfg = ModelConfig(**{**cfg.model.__dict__, "epochs": min(cfg.model.epochs, 12)})
        temporal.cfg = mcfg
        hist = temporal.train_epochs(train.eng_X, train.eng_y, train.eng_scalars, epochs=mcfg.epochs)
        report["temporal_history"] = {k: (v if not isinstance(v, list) else v[-3:]) for k, v in hist.items()}
        if cal.eng_y.size >= 16:
            raw_cal_temp = temporal.predict_proba(cal.eng_X, cal.eng_scalars)
            cal_temp = ProbabilityCalibrator.fit(raw_cal_temp, cal.eng_y)
            report["temporal_ece"] = expected_calibration_error(cal_temp.transform(raw_cal_temp), cal.eng_y)

    # ---- novelty (clean only) -------------------------------------------
    clean_mask = train.y == 0
    if clean_mask.sum() < 20:
        raise RuntimeError("need >= 20 clean training rows for the novelty AE")
    novelty = NoveltyAE.create(ds.spec, cfg.model)
    nov_hist = novelty.fit(train.X[clean_mask], epochs=min(cfg.model.epochs, 20))
    report["novelty"] = {"threshold_p99": nov_hist["threshold_p99"], "final_loss": nov_hist["loss"][-1]}

    # ---- conformal on fused-ish clean cal scores ------------------------
    # Use calibrated tabular scores as the conformal score (monotone with cheat).
    cal_scores = cal_tab.transform(tab.predict_proba(cal.X))
    clean_cal = cal_scores[cal.y == 0]
    if clean_cal.size < 50:
        # Synthetic runs can be small; replicate with jitter rather than fail.
        rng = np.random.default_rng(cfg.model.seed)
        clean_cal = np.concatenate([
            clean_cal,
            np.clip(rng.normal(clean_cal.mean() if clean_cal.size else 0.05,
                               0.02, size=max(0, 50 - clean_cal.size)), 0, 1),
        ])
    conformal = ConformalGate.fit(clean_cal, alpha=cfg.fusion.alpha)
    report["conformal_threshold"] = conformal.threshold()

    bundle = ModelBundle(
        tabular=tab, temporal=temporal, novelty=novelty,
        cal_tabular=cal_tab, cal_temporal=cal_temp, conformal=conformal,
        prior=float(max(0.02, min(0.2, train.y.mean()))),
    )
    ensemble = CalibratedEnsemble(bundle=bundle, spec=ds.spec)
    ensemble.save(out_dir)

    # ---- evaluate on held-out players -----------------------------------
    if test.y.size:
        ev = evaluate_ensemble(ensemble, test, cfg=cfg)
        report["test"] = ev.to_dict()
        (out_dir / "eval.json").write_text(json.dumps(ev.to_dict(), indent=2), encoding="utf-8")

    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return ensemble, report


def bootstrap_train(
    *,
    n_sessions: int = 48,
    duration_s: float = 75.0,
    seed: int = 1337,
    out_dir: str | Path = "models",
    cfg: AegisConfig | None = None,
    skip_temporal: bool = False,
) -> tuple[CalibratedEnsemble, dict]:
    """End-to-end: simulate -> split -> train -> evaluate -> save."""
    cfg = cfg or AegisConfig(model_dir=str(out_dir))
    ds = generate_dataset(
        n_sessions=n_sessions, duration_s=duration_s, seed=seed,
        seq_len=cfg.model.seq_len,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds.save(Path(out_dir) / "dataset")
    return train_ensemble(ds, cfg, out_dir=out_dir, skip_temporal=skip_temporal)
