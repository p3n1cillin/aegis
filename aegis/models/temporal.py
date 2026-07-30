"""Temporal encoder for engagement sequences.

Consumes the (C, T) tensors from ``features.pipeline.extract_engagement_tensors``.
Architecture is a small Transformer encoder over a 1-D convolutional stem:

  * The stem (depthwise-separable convs) captures local kinematic motifs --
    snap shape, correction timing, recoil compensation rhythm -- at low cost.
  * The Transformer lets distant timesteps interact, which is what distinguishes
    a planned human acquisition (correction after a ballistic miss) from an
    aimbot that re-solves every few ticks.

Kept deliberately small (``ModelConfig.hidden`` default 96) so inference on a
match of ~200 engagements is milliseconds on CPU. Capacity is in the features
and the fusion layer, not in a giant net.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..core.config import ModelConfig
from ..features.pipeline import SEQ_CHANNELS


class _ConvStem(nn.Module):
    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.net(x)


class _TemporalNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = _ConvStem(cfg.seq_channels, cfg.hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_blocks)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
        nn.init.normal_(self.cls, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )
        # Auxiliary scalar head: engagement-level engineered features from aim.
        self.scalar_proj = nn.Linear(16, cfg.hidden)  # padded; real width set at train

    def forward(self, seq: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        # seq: (B, C, T) -> stem -> (B, H, T) -> (B, T, H)
        h = self.stem(seq).transpose(1, 2)
        b = h.size(0)
        cls = self.cls.expand(b, -1, -1)
        if scalars is not None and scalars.numel() > 0:
            # Pad / truncate scalars to the projection width.
            s = scalars
            if s.size(-1) < 16:
                s = F.pad(s, (0, 16 - s.size(-1)))
            elif s.size(-1) > 16:
                s = s[..., :16]
            cls = cls + self.scalar_proj(s).unsqueeze(1)
        tok = torch.cat([cls, h], dim=1)
        out = self.encoder(tok)
        return self.head(out[:, 0]).squeeze(-1)


@dataclass
class TemporalEncoder:
    cfg: ModelConfig
    net: _TemporalNet
    device: torch.device
    scalar_names: tuple[str, ...] = ()

    @classmethod
    def create(cls, cfg: ModelConfig | None = None) -> "TemporalEncoder":
        cfg = cfg or ModelConfig()
        # Keep channel count honest against the pipeline constant.
        cfg = ModelConfig(**{**cfg.__dict__, "seq_channels": len(SEQ_CHANNELS)})
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        net = _TemporalNet(cfg).to(device)
        return cls(cfg=cfg, net=net, device=device)

    def train_epochs(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        scalars: np.ndarray | None = None,
        *,
        epochs: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """Supervised training. ``sequences`` shape (N, C, T), labels in {0,1}."""
        cfg = self.cfg
        epochs = epochs or cfg.epochs
        self.net.train()
        sequences = np.nan_to_num(np.asarray(sequences, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = torch.from_numpy(sequences)
        y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        if scalars is None:
            scalars = np.zeros((X.shape[0], 1), dtype=np.float32)
        scalars = np.nan_to_num(np.asarray(scalars, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        S = torch.from_numpy(scalars)
        if sample_weight is None:
            # Balance classes.
            pos = float((y == 1).sum().clamp(min=1))
            neg = float((y == 0).sum().clamp(min=1))
            w = torch.where(y > 0.5, neg / (pos + neg), pos / (pos + neg))
            w = w * (len(y) / w.sum())
        else:
            w = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))

        ds = TensorDataset(X, S, y, w)
        # Hold out 15% for early stopping.
        n = len(ds)
        n_val = max(1, int(0.15 * n))
        n_train = n - n_val
        g = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val, best_state, patience, bad = float("inf"), None, 8, 0
        history = {"train_loss": [], "val_loss": []}

        for ep in range(epochs):
            self.net.train()
            tr = 0.0
            for xb, sb, yb, wb in train_loader:
                xb, sb, yb, wb = xb.to(self.device), sb.to(self.device), yb.to(self.device), wb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logit = self.net(xb, sb)
                loss = (F.binary_cross_entropy_with_logits(logit, yb, reduction="none") * wb).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tr += float(loss.item()) * len(xb)
            sched.step()
            tr /= max(n_train, 1)

            self.net.eval()
            va = 0.0
            with torch.no_grad():
                for xb, sb, yb, wb in val_loader:
                    xb, sb, yb = xb.to(self.device), sb.to(self.device), yb.to(self.device)
                    logit = self.net(xb, sb)
                    va += float(F.binary_cross_entropy_with_logits(logit, yb).item()) * len(xb)
            va /= max(n_val, 1)
            history["train_loss"].append(tr)
            history["val_loss"].append(va)
            if va < best_val - 1e-4:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        history["best_val"] = best_val
        history["epochs_run"] = len(history["train_loss"])
        return history

    @torch.no_grad()
    def predict_proba(self, sequences: np.ndarray, scalars: np.ndarray | None = None) -> np.ndarray:
        self.net.eval()
        sequences = np.nan_to_num(np.asarray(sequences, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X = torch.from_numpy(sequences).to(self.device)
        if scalars is None:
            S = torch.zeros((X.shape[0], 1), dtype=torch.float32, device=self.device)
        else:
            scalars = np.nan_to_num(np.asarray(scalars, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            S = torch.from_numpy(scalars).to(self.device)
        out = []
        for i in range(0, X.shape[0], 256):
            logit = self.net(X[i:i + 256], S[i:i + 256])
            out.append(torch.sigmoid(logit).cpu().numpy())
        arr = np.concatenate(out).astype(np.float64) if out else np.zeros(0)
        return np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.net.state_dict(),
            "cfg": self.cfg.__dict__,
            "scalar_names": list(self.scalar_names),
            "seq_channels": list(SEQ_CHANNELS),
        }, path / "temporal.pt")
        (path / "temporal_meta.json").write_text(
            json.dumps({"cfg": self.cfg.__dict__, "seq_channels": list(SEQ_CHANNELS)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TemporalEncoder":
        path = Path(path)
        blob = torch.load(path / "temporal.pt", map_location="cpu", weights_only=False)
        cfg = ModelConfig(**blob["cfg"])
        if device:
            cfg.device = device
        enc = cls.create(cfg)
        enc.net.load_state_dict(blob["state"])
        enc.scalar_names = tuple(blob.get("scalar_names", ()))
        return enc
