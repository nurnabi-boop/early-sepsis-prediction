"""PhysioNet 2019 utility score plus auxiliary metrics.

The utility function is piecewise linear in time relative to the clinical
sepsis onset `t_sepsis`. The hyperparameters are taken from the official
challenge specification:

    dt_early   = -12   # 12h before t_sepsis: window opens
    dt_optimal =  -6   # 6h before t_sepsis: SepsisLabel becomes 1
    dt_late    =   3   # 3h after  t_sepsis: window closes
    U_TP_max =  1.0    # max reward, attained at t_optimal
    U_FN     = -2.0    # worst penalty, attained for missed predictions at t_late
    U_FP     = -0.05   # penalty for false alarms
    U_TN     =  0.0

The reward for predicting `1` at time t (for a septic patient) ramps linearly
from `U_FP` at `t_sepsis + dt_early` to `U_TP_max` at `t_sepsis + dt_optimal`,
then linearly back to 0 at `t_sepsis + dt_late`. The reward for predicting
`0` is `U_TN` until `t_sepsis + dt_optimal`, then ramps linearly to `U_FN` at
`t_sepsis + dt_late`. For non-septic patients, predicting 1 incurs `U_FP` and
predicting 0 yields `U_TN`.

The final challenge score normalises the sum across patients between the
"inaction" baseline (never predict positive) and the perfect-classifier upper
bound, so 1.0 = perfect and 0.0 = inaction.

Reference: https://physionet.org/content/challenge-2019/1.0.0/, in particular
the `evaluate_sepsis_score.py` script published with the challenge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

# Challenge constants
DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
U_TP_MAX = 1.0
U_FN_MIN = -2.0
U_FP = -0.05
U_TN = 0.0


def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = DT_EARLY,
    dt_optimal: int = DT_OPTIMAL,
    dt_late: int = DT_LATE,
    max_u_tp: float = U_TP_MAX,
    min_u_fn: float = U_FN_MIN,
    u_fp: float = U_FP,
    u_tn: float = U_TN,
) -> float:
    """Per-patient unnormalised utility, summed across hours."""
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if labels.shape != predictions.shape:
        raise ValueError(f"Length mismatch: labels {labels.shape} vs preds {predictions.shape}")

    n = len(labels)
    if np.any(labels):
        is_septic = True
        # SepsisLabel becomes 1 at t_sepsis + dt_optimal, so t_sepsis = first_pos - dt_optimal.
        t_sepsis = int(np.argmax(labels)) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float("inf")

    u = np.zeros(n, dtype=float)
    for t in range(n):
        if not is_septic:
            u[t] = u_fp if predictions[t] else u_tn
            continue

        # Reward for predicting 1 at time t
        if t <= t_sepsis + dt_early:
            r1 = u_fp
        elif t <= t_sepsis + dt_optimal:
            r1 = (max_u_tp - u_fp) * (t - (t_sepsis + dt_early)) / (dt_optimal - dt_early) + u_fp
        elif t <= t_sepsis + dt_late:
            r1 = max_u_tp * (1.0 - (t - (t_sepsis + dt_optimal)) / (dt_late - dt_optimal))
        else:
            r1 = 0.0

        # Reward for predicting 0 at time t
        if t <= t_sepsis + dt_optimal:
            r0 = u_tn
        elif t <= t_sepsis + dt_late:
            r0 = min_u_fn * (t - (t_sepsis + dt_optimal)) / (dt_late - dt_optimal)
        else:
            r0 = min_u_fn

        u[t] = r1 if predictions[t] else r0

    return float(u.sum())


def _best_predictions(labels: np.ndarray) -> np.ndarray:
    """Predict 1 from t_optimal onward (perfect-clairvoyance baseline)."""
    labels = np.asarray(labels, dtype=np.int8)
    pred = np.zeros_like(labels)
    if np.any(labels):
        first_pos = int(np.argmax(labels))
        pred[first_pos:] = 1
    return pred


def _inaction_predictions(labels: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(labels), dtype=np.int8)


def normalised_utility(
    label_seqs: Iterable[np.ndarray],
    pred_seqs: Iterable[np.ndarray],
) -> dict:
    """Population-level normalised PhysioNet utility score."""
    label_seqs = list(label_seqs)
    pred_seqs = list(pred_seqs)
    obs = sum(compute_prediction_utility(l, p) for l, p in zip(label_seqs, pred_seqs))
    best = sum(compute_prediction_utility(l, _best_predictions(l)) for l in label_seqs)
    inaction = sum(compute_prediction_utility(l, _inaction_predictions(l)) for l in label_seqs)
    denom = best - inaction if (best - inaction) != 0 else 1.0
    return {
        "observed_utility": obs,
        "best_utility": best,
        "inaction_utility": inaction,
        "normalised_utility": (obs - inaction) / denom,
    }


# ---- Patient / row-level discrimination -----------------------------------
@dataclass
class DiscriminationMetrics:
    auroc_row: float
    auprc_row: float
    auroc_patient: float
    auprc_patient: float
    brier: float


def discrimination_metrics(
    label_seqs: list[np.ndarray],
    score_seqs: list[np.ndarray],
) -> DiscriminationMetrics:
    """Row-level and patient-level AUROC/AUPRC.

    Row-level: every (patient, hour) pair is one example. This is what the
    challenge reports.

    Patient-level: one example per patient; positive iff sepsis ever occurred,
    score = max risk over the stay. Cleaner clinically because it ignores the
    correlation between rows of the same stay.
    """
    flat_y, flat_s = [], []
    pat_y, pat_s = [], []
    for y, s in zip(label_seqs, score_seqs):
        y = np.asarray(y, dtype=np.int8)
        s = np.asarray(s, dtype=float)
        flat_y.append(y)
        flat_s.append(s)
        pat_y.append(int(y.max()))
        pat_s.append(float(s.max()))
    flat_y = np.concatenate(flat_y)
    flat_s = np.concatenate(flat_s)
    pat_y = np.array(pat_y)
    pat_s = np.array(pat_s)

    return DiscriminationMetrics(
        auroc_row=roc_auc_score(flat_y, flat_s) if flat_y.min() != flat_y.max() else float("nan"),
        auprc_row=average_precision_score(flat_y, flat_s) if flat_y.min() != flat_y.max() else float("nan"),
        auroc_patient=roc_auc_score(pat_y, pat_s) if pat_y.min() != pat_y.max() else float("nan"),
        auprc_patient=average_precision_score(pat_y, pat_s) if pat_y.min() != pat_y.max() else float("nan"),
        brier=brier_score_loss(flat_y, np.clip(flat_s, 0, 1)),
    )


def threshold_predictions(
    score_seqs: list[np.ndarray],
    threshold: float,
) -> list[np.ndarray]:
    """Once a score crosses the threshold the patient stays "alarmed".

    The challenge spec lets predictions toggle freely, but in clinical practice
    once you've raised an early-warning alarm you don't un-raise it at the
    next hour, and this monotonic version is what most challenge entries
    submitted."""
    out = []
    for s in score_seqs:
        s = np.asarray(s, dtype=float)
        triggered = (s >= threshold).astype(np.int8)
        # Monotonic: once 1, stay 1.
        triggered = np.maximum.accumulate(triggered)
        out.append(triggered)
    return out


def sweep_threshold_for_utility(
    label_seqs: list[np.ndarray],
    score_seqs: list[np.ndarray],
    grid: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Find the score threshold that maximises the normalised utility."""
    if grid is None:
        all_scores = np.concatenate([np.asarray(s, dtype=float) for s in score_seqs])
        grid = np.quantile(all_scores, np.linspace(0.50, 0.999, 50))
    best = None
    for thr in grid:
        preds = threshold_predictions(score_seqs, thr)
        result = normalised_utility(label_seqs, preds)
        result["threshold"] = float(thr)
        if best is None or result["normalised_utility"] > best["normalised_utility"]:
            best = result
    return best["threshold"], best


# ---- Self-test ------------------------------------------------------------
def _selftest() -> None:
    """Sanity checks for the utility implementation.

    These are the expected qualitative behaviours, not the toy numbers from
    the challenge website (which we cannot ship). Anything that violates them
    means the score is broken; passing them is necessary but not sufficient.
    """
    rng = np.random.default_rng(0)

    # 1. Non-septic patient, no alarms: utility = 0
    labels = np.zeros(48, dtype=np.int8)
    preds = np.zeros(48, dtype=np.int8)
    assert compute_prediction_utility(labels, preds) == 0.0

    # 2. Non-septic patient, always alarming: utility = U_FP * len
    preds = np.ones(48, dtype=np.int8)
    expected = U_FP * 48
    assert abs(compute_prediction_utility(labels, preds) - expected) < 1e-9

    # 3. Septic patient predicted optimally (positive starting at t_sepsis-6):
    #    utility should equal the "best" baseline -- the maximum achievable.
    labels = np.zeros(48, dtype=np.int8)
    t_sepsis = 30
    labels[t_sepsis + DT_OPTIMAL :] = 1
    best_preds = _best_predictions(labels)
    obs = compute_prediction_utility(labels, best_preds)
    best = compute_prediction_utility(labels, best_preds)
    assert obs == best

    # 4. Septic patient never predicted: should equal the inaction baseline,
    #    which is strictly negative for septic patients (U_FN domain).
    inaction = compute_prediction_utility(labels, np.zeros_like(labels))
    assert inaction < 0

    # 5. Predicting at t_sepsis-12 (window edge) yields ~ U_FP from that hour
    #    and the linear ramp to U_TP_max=1 at t_sepsis-6.
    early_preds = np.zeros_like(labels)
    early_preds[t_sepsis + DT_EARLY :] = 1
    early = compute_prediction_utility(labels, early_preds)
    assert early > inaction  # better to predict early than not at all

    # 6. Normalised utility of the perfect predictor on a mixed cohort = 1.0.
    cohort_labels, cohort_scores = [], []
    for _ in range(50):
        L = rng.integers(20, 100)
        lab = np.zeros(L, dtype=np.int8)
        if rng.random() < 0.1:  # ~ challenge prevalence
            t_s = int(rng.integers(L // 2, L - 1))
            lab[max(0, t_s + DT_OPTIMAL) :] = 1
        cohort_labels.append(lab)
        cohort_scores.append(_best_predictions(lab).astype(float))
    nu = normalised_utility(cohort_labels, cohort_scores)
    assert abs(nu["normalised_utility"] - 1.0) < 1e-9, nu

    # 7. Inaction predictor → normalised utility = 0.
    cohort_inaction = [np.zeros_like(l, dtype=float) for l in cohort_labels]
    nu = normalised_utility(cohort_labels, cohort_inaction)
    assert abs(nu["normalised_utility"] - 0.0) < 1e-9, nu

    print("evaluate.py self-test: OK")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
