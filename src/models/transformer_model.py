"""Causal Transformer encoder for sequential sepsis prediction.

The encoder is causally masked: at hour t the model can only attend to hours
0..t. This matches the streaming-prediction protocol of the challenge --
without it, AUROC inflates because attention over future rows leaks the
ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .lstm_model import _SepsisSeqDataset, _collate
from ..features import feature_columns
from ..data_loader import TARGET


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_len: int = 336
    pos_weight: float | None = None
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _SinPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, in_features: int, cfg: TransformerConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, cfg.d_model)
        self.pos = _SinPosEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.pos(self.proj(x))

        # Causal mask: at position t, can only see <= t.
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Padding mask: True where padded.
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pad_mask = idx >= lengths.to(x.device).unsqueeze(1)

        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(h).squeeze(-1)


class TransformerSepsisModel:
    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        self.cfg = cfg or TransformerConfig()
        self.net: _TransformerNet | None = None
        self.feature_names_: list[str] | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None

    # Normalisation logic mirrors the LSTM model -- duplicated rather than
    # inherited because the two model classes are otherwise independent and
    # may diverge (e.g. the Transformer might want learnable feature embeddings).
    def _fit_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        feats = self.feature_names_
        assert feats is not None
        std_cols = [c for c in feats if not (c.endswith("_mask") or c.endswith("_dt"))]
        means = frame[std_cols].mean().to_numpy(dtype=np.float32)
        stds = frame[std_cols].std().replace(0, 1).to_numpy(dtype=np.float32)
        full_means = np.zeros(len(feats), dtype=np.float32)
        full_stds = np.ones(len(feats), dtype=np.float32)
        idx = {c: i for i, c in enumerate(feats)}
        for c, m, s in zip(std_cols, means, stds):
            full_means[idx[c]] = m
            full_stds[idx[c]] = s
        self.feature_means_ = full_means
        self.feature_stds_ = full_stds
        return self._apply_norm(frame)

    def _apply_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        feats = self.feature_names_
        assert feats is not None and self.feature_means_ is not None
        out = frame.copy()
        out[feats] = (out[feats].to_numpy(dtype=np.float32) - self.feature_means_) / self.feature_stds_
        out[feats] = out[feats].fillna(0.0)
        return out

    def fit(self, train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None = None) -> "TransformerSepsisModel":
        cfg = self.cfg
        feats = feature_columns(train_frame)
        self.feature_names_ = feats
        train_norm = self._fit_norm(train_frame)
        train_ds = _SepsisSeqDataset(train_norm, feats, cfg.max_len)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate, num_workers=0
        )

        valid_loader = None
        if valid_frame is not None and len(valid_frame):
            valid_norm = self._apply_norm(valid_frame)
            valid_ds = _SepsisSeqDataset(valid_norm, feats, cfg.max_len)
            valid_loader = DataLoader(
                valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate, num_workers=0
            )

        device = torch.device(cfg.device)
        self.net = _TransformerNet(len(feats), cfg).to(device)

        pos_weight = cfg.pos_weight
        if pos_weight is None:
            y = train_frame[TARGET].to_numpy()
            pos = max(int(y.sum()), 1)
            neg = max(len(y) - pos, 1)
            pos_weight = neg / pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        best_val = float("inf")
        best_state: dict | None = None
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            total, denom = 0.0, 0.0
            for X, y, m, lens in train_loader:
                X, y, m = X.to(device), y.to(device), m.to(device)
                logits = self.net(X, lens)
                loss = loss_fn(logits, y) * m
                loss = loss.sum() / m.sum().clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                opt.step()
                total += loss.item() * m.sum().item()
                denom += m.sum().item()
            sched.step()
            train_loss = total / max(denom, 1.0)

            val_msg = ""
            if valid_loader is not None:
                self.net.eval()
                v_total, v_denom = 0.0, 0.0
                with torch.no_grad():
                    for X, y, m, lens in valid_loader:
                        X, y, m = X.to(device), y.to(device), m.to(device)
                        logits = self.net(X, lens)
                        loss = loss_fn(logits, y) * m
                        v_total += loss.sum().item()
                        v_denom += m.sum().item()
                val = v_total / max(v_denom, 1.0)
                val_msg = f" val_loss={val:.4f}"
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            print(f"[tx] epoch {epoch:02d} train_loss={train_loss:.4f}{val_msg}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_per_patient(self, frame: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self.net is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit")
        cfg = self.cfg
        device = torch.device(cfg.device)
        feats = self.feature_names_
        norm = self._apply_norm(frame)
        ds = _SepsisSeqDataset(norm, feats, cfg.max_len)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
        pids = [p for p, _ in norm.groupby("pid", sort=False)]
        labels_by_pid = {p: g.sort_values("hour")[TARGET].to_numpy() for p, g in norm.groupby("pid", sort=False)}

        self.net.eval()
        score_seqs: list[np.ndarray] = []
        for X, _y, _m, lens in loader:
            X = X.to(device)
            logits = self.net(X, lens)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, L in enumerate(lens.tolist()):
                score_seqs.append(probs[i, :L])

        label_seqs = []
        for pid, scr in zip(pids, score_seqs):
            lab = labels_by_pid[pid]
            if len(lab) > cfg.max_len:
                lab = lab[-cfg.max_len:]
            label_seqs.append(lab)
        return label_seqs, score_seqs
