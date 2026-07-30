"""Self-supervised novelty detector.

Supervised models are only as good as the cheat families in the training set.
A novel technique -- a new humaniser, a new injection method -- looks like
"clean" to a classifier trained yesterday. The autoencoder closes that gap:

  * Trained **only on clean players**.
  * At serve time, reconstruction error is a novelty score.
  * High novelty does not by itself ban anyone (that would make every skilled
    player a suspect); it raises the prior and routes the session to review,
    and it is the channel that catches zero-days before labels exist.

Architecture is a small MLP AE over the session feature vector with a
contractive penalty, so the latent is forced to be smooth in input space and
cannot memorise individual clean players.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.config import ModelConfig
from ..features.pipeline import FeatureSpec


class _AE(nn.Module):
    def __init__(self, n_in: int, latent: int, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), z


@dataclass
class NoveltyAE:
    spec: FeatureSpec
    cfg: ModelConfig
    net: _AE
    device: torch.device
    # Robust scale fitted on clean training data so scores are comparable.
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    score_mu_: float = 0.0
    score_sd_: float = 1.0
    # Conformal / empirical threshold at a target false-alarm rate on clean.
    threshold_p99_: float = 3.0

    @classmethod
    def create(cls, spec: FeatureSpec, cfg: ModelConfig | None = None) -> "NoveltyAE":
        cfg = cfg or ModelConfig()
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        net = _AE(len(spec.columns), cfg.latent, hidden=max(64, cfg.hidden)).to(device)
        return cls(spec=spec, cfg=cfg, net=net, device=device)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.center_ is None or self.scale_ is None:
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Z = (X - self.center_) / self.scale_
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X_clean: np.ndarray, *, epochs: int | None = None) -> dict:
        """Fit on clean-only rows. Cheaters must not enter this set."""
        X = np.asarray(X_clean, dtype=np.float32)
        if X.shape[1] != len(self.spec.columns):
            raise ValueError("feature width mismatch")
        # Robust column scale from clean population. Some capability-gated
        # columns are all-NaN on short / partial sessions (e.g. bhop metrics
        # with no jump chains) — treat those as zero-centered unit scale so
        # the AE does not poison itself with nanmedian warnings.
        with np.errstate(all="ignore"):
            center = np.nanmedian(X, axis=0)
            mad = np.nanmedian(np.abs(X - center), axis=0)
        self.center_ = np.nan_to_num(center, nan=0.0).astype(np.float32)
        self.scale_ = np.maximum(
            1.4826 * np.nan_to_num(mad, nan=0.0), 1e-3
        ).astype(np.float32)
        Z = self._prep(X)

        epochs = epochs or self.cfg.epochs
        ds = TensorDataset(torch.from_numpy(Z))
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        history = {"loss": []}
        self.net.train()
        for _ in range(epochs):
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                xb = xb.requires_grad_(True)
                opt.zero_grad(set_to_none=True)
                recon, z = self.net(xb)
                rec = ((recon - xb) ** 2).mean()
                # Contractive: penalise Jacobian of encoder via grad of ||z||.
                # Cheap approximation: penalise sensitivity of latent norm.
                z_norm = (z ** 2).sum(dim=1).mean()
                grad = torch.autograd.grad(z_norm, xb, create_graph=True, retain_graph=True)[0]
                contract = (grad ** 2).mean()
                loss = rec + 1e-4 * contract
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
            history["loss"].append(total / max(len(ds), 1))

        scores = self.raw_score(X)
        self.score_mu_ = float(np.median(scores))
        self.score_sd_ = float(np.maximum(1.4826 * np.median(np.abs(scores - self.score_mu_)), 1e-6))
        self.threshold_p99_ = float(np.quantile(self.normalized_score(X), 0.99))
        history["threshold_p99"] = self.threshold_p99_
        return history

    @torch.no_grad()
    def raw_score(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Z = self._prep(X)
        t = torch.from_numpy(Z).to(self.device)
        out = []
        for i in range(0, t.shape[0], 512):
            recon, _ = self.net(t[i:i + 512])
            err = ((recon - t[i:i + 512]) ** 2).mean(dim=1)
            out.append(err.cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def normalized_score(self, X: np.ndarray) -> np.ndarray:
        """Robust z-score of reconstruction error relative to clean training."""
        raw = self.raw_score(X)
        return (raw - self.score_mu_) / self.score_sd_

    def novelty_probability(self, X: np.ndarray) -> np.ndarray:
        """Map novelty to a soft [0,1] score via a logistic on the z-score.

        This is NOT a calibrated P(cheat); fusion treats it as a separate
        modality with its own likelihood ratio table.
        """
        z = self.normalized_score(X)
        return 1.0 / (1.0 + np.exp(-(z - self.threshold_p99_ * 0.5)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.net.state_dict(),
            "cfg": self.cfg.__dict__,
            "center": self.center_,
            "scale": self.scale_,
            "score_mu": self.score_mu_,
            "score_sd": self.score_sd_,
            "threshold_p99": self.threshold_p99_,
            "n_in": len(self.spec.columns),
            "latent": self.cfg.latent,
        }, path / "novelty.pt")
        (path / "novelty_spec.json").write_text(json.dumps(self.spec.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "NoveltyAE":
        path = Path(path)
        blob = torch.load(path / "novelty.pt", map_location="cpu", weights_only=False)
        spec = FeatureSpec.from_dict(json.loads((path / "novelty_spec.json").read_text(encoding="utf-8")))
        cfg = ModelConfig(**blob["cfg"])
        if device:
            cfg.device = device
        obj = cls.create(spec, cfg)
        obj.net.load_state_dict(blob["state"])
        obj.center_ = blob["center"]
        obj.scale_ = blob["scale"]
        obj.score_mu_ = float(blob["score_mu"])
        obj.score_sd_ = float(blob["score_sd"])
        obj.threshold_p99_ = float(blob["threshold_p99"])
        return obj
