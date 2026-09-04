"""Per-row XGBoost classifier on engineered features.

Treats every (patient, hour) row as an independent example for training, but
predictions are kept grouped by patient so the utility score and trajectory
plots can re-assemble the time series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features import feature_columns
from ..data_loader import TARGET


@dataclass
class XGBConfig:
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    scale_pos_weight: float | None = None  # auto from class freq if None
    tree_method: str = "hist"
    early_stopping_rounds: int | None = 30
    random_state: int = 0


class XGBSepsisModel:
    def __init__(self, cfg: XGBConfig | None = None) -> None:
        self.cfg = cfg or XGBConfig()
        self.model: xgb.XGBClassifier | None = None
        self.feature_names_: list[str] | None = None

    def _build(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        c = self.cfg
        return xgb.XGBClassifier(
            n_estimators=c.n_estimators,
            max_depth=c.max_depth,
            learning_rate=c.learning_rate,
            subsample=c.subsample,
            colsample_bytree=c.colsample_bytree,
            min_child_weight=c.min_child_weight,
            reg_lambda=c.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            tree_method=c.tree_method,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=c.random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        train_frame: pd.DataFrame,
        valid_frame: pd.DataFrame | None = None,
    ) -> "XGBSepsisModel":
        feats = feature_columns(train_frame)
        self.feature_names_ = feats
        X_tr = train_frame[feats].to_numpy(dtype=np.float32)
        y_tr = train_frame[TARGET].to_numpy(dtype=np.int8)

        spw = self.cfg.scale_pos_weight
        if spw is None:
            pos = max(int(y_tr.sum()), 1)
            neg = max(len(y_tr) - pos, 1)
            spw = neg / pos

        self.model = self._build(spw)

        eval_set = None
        if valid_frame is not None and len(valid_frame):
            X_v = valid_frame[feats].to_numpy(dtype=np.float32)
            y_v = valid_frame[TARGET].to_numpy(dtype=np.int8)
            eval_set = [(X_v, y_v)]

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
            if self.cfg.early_stopping_rounds:
                # XGBoost >=2.0 accepts callbacks; keep the simple kwarg path for portability.
                fit_kwargs["early_stopping_rounds"] = self.cfg.early_stopping_rounds
        self.model.fit(X_tr, y_tr, **fit_kwargs)
        return self

    def predict_proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit")
        X = frame[self.feature_names_].to_numpy(dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def predict_per_patient(
        self, frame: pd.DataFrame
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (label_seqs, score_seqs) grouped by `pid`, preserving hour order."""
        scores = self.predict_proba_frame(frame)
        out_lab, out_scr = [], []
        for _, group in frame.groupby("pid", sort=False):
            order = np.argsort(group["hour"].to_numpy())
            out_lab.append(group[TARGET].to_numpy()[order])
            out_scr.append(scores[group.index.to_numpy()][order])
        return out_lab, out_scr

    def feature_importance(self) -> pd.Series:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit")
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)
