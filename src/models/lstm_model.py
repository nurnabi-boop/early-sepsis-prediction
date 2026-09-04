"""LSTM with mask-aware inputs for sequential sepsis prediction.

The model consumes the engineered per-hour feature frame as a (T, F) tensor
per patient. Variable-length sequences are padded and masked. The forward-fill
mask channels are already part of the feature set so the LSTM can learn to
discount imputed values, but we additionally pass a sequence-length mask to
the loss so padded steps don't contribute gradients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..features import feature_columns
from ..data_loader import TARGET


@dataclass
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False  # forbidden for streaming -- left as a knob, off by default
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_len: int = 336  # 14 days; longer stays are tail-clipped at the input
    pos_weight: float | None = None  # BCE pos_weight, auto if None
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _SepsisSeqDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feats: list[str], max_len: int) -> None:
        self.feats = feats
        self.max_len = max_len
        self.groups: list[tuple[np.ndarray, np.ndarray]] = []
        for _, g in frame.groupby("pid", sort=False):
            g = g.sort_values("hour")
            X = g[feats].to_numpy(dtype=np.float32)
            y = g[TARGET].to_numpy(dtype=np.float32)
            if len(X) > max_len:
                X = X[-max_len:]
                y = y[-max_len:]
            self.groups.append((X, y))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.groups[idx]


def _collate(batch: list[tuple[np.ndarray, np.ndarray]]):
    lengths = [len(x) for x, _ in batch]
    T = max(lengths)
    F = batch[0][0].shape[1]
    Xp = np.zeros((len(batch), T, F), dtype=np.float32)
    yp = np.zeros((len(batch), T), dtype=np.float32)
    mp = np.zeros((len(batch), T), dtype=np.float32)
    for i, (X, y) in enumerate(batch):
        L = len(X)
        Xp[i, :L] = X
        yp[i, :L] = y
        mp[i, :L] = 1.0
    return (
        torch.from_numpy(Xp),
        torch.from_numpy(yp),
        torch.from_numpy(mp),
        torch.tensor(lengths, dtype=torch.long),
    )


class _LSTMNet(nn.Module):
    def __init__(self, in_features: int, cfg: LSTMConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Sort by length descending for pack_padded; keep an inverse permutation.
        sorted_lens, sort_idx = lengths.sort(descending=True)
        unsort_idx = sort_idx.argsort()
        x_sorted = x[sort_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lens.cpu(), batch_first=True, enforce_sorted=True
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = out[unsort_idx]
        logits = self.head(out).squeeze(-1)
        return logits


class LSTMSepsisModel:
    def __init__(self, cfg: LSTMConfig | None = None) -> None:
        self.cfg = cfg or LSTMConfig()
        self.net: _LSTMNet | None = None
        self.feature_names_: list[str] | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None

    def _normalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Standardise non-mask features. Mask / dt channels are left alone."""
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

        out = frame.copy()
        out[feats] = (out[feats].to_numpy(dtype=np.float32) - full_means) / full_stds
        out[feats] = out[feats].fillna(0.0)
        return out

    def _apply_norm(self, frame: pd.DataFrame) -> pd.DataFrame:
        feats = self.feature_names_
        assert feats is not None and self.feature_means_ is not None
        out = frame.copy()
        out[feats] = (out[feats].to_numpy(dtype=np.float32) - self.feature_means_) / self.feature_stds_
        out[feats] = out[feats].fillna(0.0)
        return out

    def fit(self, train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None = None) -> "LSTMSepsisModel":
        cfg = self.cfg
        feats = feature_columns(train_frame)
        self.feature_names_ = feats

        train_norm = self._normalise(train_frame)
        train_ds = _SepsisSeqDataset(train_norm, feats, cfg.max_len)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=_collate, num_workers=0,
        )
        valid_loader = None
        if valid_frame is not None and len(valid_frame):
            valid_norm = self._apply_norm(valid_frame)
            valid_ds = _SepsisSeqDataset(valid_norm, feats, cfg.max_len)
            valid_loader = DataLoader(
                valid_ds, batch_size=cfg.batch_size, shuffle=False,
                collate_fn=_collate, num_workers=0,
            )

        device = torch.device(cfg.device)
        self.net = _LSTMNet(len(feats), cfg).to(device)

        pos_weight = cfg.pos_weight
        if pos_weight is None:
            y = train_frame[TARGET].to_numpy()
            pos = max(int(y.sum()), 1)
            neg = max(len(y) - pos, 1)
            pos_weight = neg / pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val = float("inf")
        best_state: dict | None = None

        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            total = 0.0
            denom = 0.0
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
            print(f"[lstm] epoch {epoch:02d} train_loss={train_loss:.4f}{val_msg}")

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

        # Re-derive per-patient ordering (groupby preserves first-seen order).
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

        # Truncate labels to match model max_len trimming.
        label_seqs = []
        for pid, scr in zip(pids, score_seqs):
            lab = labels_by_pid[pid]
            if len(lab) > cfg.max_len:
                lab = lab[-cfg.max_len:]
            label_seqs.append(lab)
        return label_seqs, score_seqs
