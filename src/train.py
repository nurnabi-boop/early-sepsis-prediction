"""End-to-end training driver.

Usage:
    python -m src.train --model xgb --subset 5000
    python -m src.train --model lstm --train-hospital A --test-hospital B
    python -m src.train --model transformer --subset 5000

Outputs metrics + best threshold + a risk-trajectory figure into `results/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_loader import (
    load_dataset,
    patient_train_test_split,
    split_by_hospital,
    PatientRecord,
)
from .features import featurize_many
from .evaluate import (
    discrimination_metrics,
    normalised_utility,
    sweep_threshold_for_utility,
    threshold_predictions,
)
from .visualize import save_trajectory_grid

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def _build_model(name: str):
    if name == "xgb":
        from .models.xgb_model import XGBSepsisModel
        return XGBSepsisModel()
    if name == "lstm":
        from .models.lstm_model import LSTMSepsisModel
        return LSTMSepsisModel()
    if name == "transformer":
        from .models.transformer_model import TransformerSepsisModel
        return TransformerSepsisModel()
    raise ValueError(f"Unknown model: {name}")


def _split_for_protocol(
    records: list[PatientRecord],
    train_hosp: str | None,
    test_hosp: str | None,
    seed: int,
) -> tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]]:
    """Train / valid / test split.

    If train/test hospitals are specified, the model is trained on patients
    from `train_hosp` only (with a held-out validation slice) and tested on
    patients from `test_hosp`. Otherwise patients from all loaded hospitals
    are pooled and split 70/10/20.
    """
    if train_hosp and test_hosp:
        groups = split_by_hospital(records)
        if train_hosp not in groups:
            raise ValueError(f"No patients loaded from hospital {train_hosp}")
        if test_hosp not in groups:
            raise ValueError(f"No patients loaded from hospital {test_hosp}")
        train_pool = groups[train_hosp]
        test_pool = groups[test_hosp]
        train, valid = patient_train_test_split(train_pool, test_frac=0.15, seed=seed)
        return train, valid, test_pool

    train_pool, test = patient_train_test_split(records, test_frac=0.20, seed=seed)
    train, valid = patient_train_test_split(train_pool, test_frac=0.125, seed=seed + 1)
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "lstm", "transformer"], required=True)
    parser.add_argument("--subset", type=int, default=None,
                        help="Cap number of patients loaded for fast iteration.")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--train-hospital", type=str, default=None,
                        help="If set with --test-hospital, do cross-hospital training.")
    parser.add_argument("--test-hospital", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix added to artefact filenames.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hospitals = ("A", "B")
    if args.train_hospital and args.test_hospital:
        hospitals = (args.train_hospital, args.test_hospital)

    print(f"[train] loading data from {args.data_root}")
    t0 = time.time()
    records = load_dataset(args.data_root, hospitals=hospitals, subset=args.subset, seed=args.seed)
    if not records:
        raise SystemExit(
            f"No PSV files found under {args.data_root}. Place training_setA/ "
            "and training_setB/ from the PhysioNet 2019 challenge there."
        )
    print(f"[train] loaded {len(records)} patients in {time.time()-t0:.1f}s")

    train_recs, valid_recs, test_recs = _split_for_protocol(
        records, args.train_hospital, args.test_hospital, args.seed,
    )
    print(f"[train] split: train={len(train_recs)} valid={len(valid_recs)} test={len(test_recs)}")

    print("[train] featurising...")
    t0 = time.time()
    train_frame = featurize_many(train_recs)
    valid_frame = featurize_many(valid_recs)
    test_frame = featurize_many(test_recs)
    print(f"[train] featurised in {time.time()-t0:.1f}s "
          f"(train rows={len(train_frame):,}, "
          f"sepsis prevalence={train_frame['SepsisLabel'].mean():.3%})")

    model = _build_model(args.model)
    print(f"[train] fitting {args.model}...")
    t0 = time.time()
    model.fit(train_frame, valid_frame=valid_frame)
    print(f"[train] fit in {time.time()-t0:.1f}s")

    # Predictions on validation: tune the alarm threshold against utility.
    print("[train] tuning threshold on validation...")
    valid_labels, valid_scores = model.predict_per_patient(valid_frame)
    best_thr, best_val = sweep_threshold_for_utility(valid_labels, valid_scores)
    print(f"[train] best validation utility = {best_val['normalised_utility']:.4f} @ thr={best_thr:.3f}")

    # Test set
    print("[train] evaluating on test...")
    test_labels, test_scores = model.predict_per_patient(test_frame)
    test_preds = threshold_predictions(test_scores, best_thr)
    util = normalised_utility(test_labels, test_preds)
    disc = discrimination_metrics(test_labels, test_scores)

    summary = {
        "model": args.model,
        "subset": args.subset,
        "hospitals_loaded": list(hospitals),
        "train_hospital": args.train_hospital,
        "test_hospital": args.test_hospital,
        "n_patients": {
            "train": len(train_recs),
            "valid": len(valid_recs),
            "test": len(test_recs),
        },
        "threshold": float(best_thr),
        "validation_utility": best_val["normalised_utility"],
        "test_normalised_utility": util["normalised_utility"],
        "test_observed_utility": util["observed_utility"],
        "test_best_utility": util["best_utility"],
        "test_inaction_utility": util["inaction_utility"],
        "test_auroc_row": disc.auroc_row,
        "test_auprc_row": disc.auprc_row,
        "test_auroc_patient": disc.auroc_patient,
        "test_auprc_patient": disc.auprc_patient,
        "test_brier": disc.brier,
    }
    print("[train] " + json.dumps(summary, indent=2))

    tag = args.tag or f"{args.model}"
    if args.train_hospital and args.test_hospital:
        tag += f"_train{args.train_hospital}_test{args.test_hospital}"
    elif args.subset:
        tag += f"_n{args.subset}"
    out_metrics = RESULTS_DIR / f"metrics_{tag}.json"
    out_metrics.write_text(json.dumps(summary, indent=2))

    # Save artefacts -- joblib for sklearn-style models, torch state dict for nets.
    if args.model == "xgb":
        joblib.dump({"model": model, "threshold": best_thr}, MODELS_DIR / f"{tag}.joblib")
    else:
        import torch
        state = {
            "state_dict": model.net.state_dict(),
            "feature_names": model.feature_names_,
            "feature_means": model.feature_means_,
            "feature_stds": model.feature_stds_,
            "config": vars(model.cfg),
            "threshold": float(best_thr),
        }
        torch.save(state, MODELS_DIR / f"{tag}.pt")

    # Risk trajectories
    test_pids = [p for p, _ in test_frame.groupby("pid", sort=False)]
    save_trajectory_grid(
        test_labels, test_scores, test_pids,
        RESULTS_DIR / f"trajectories_{tag}.png",
        threshold=best_thr,
    )
    print(f"[train] wrote {out_metrics} and trajectories_{tag}.png")


if __name__ == "__main__":  # pragma: no cover
    main()
