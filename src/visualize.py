"""Plotting utilities -- risk trajectories, missingness, calibration.

All plots accept matplotlib Axes so they can be composed inside notebooks. The
trajectory plotter is the headline output of the project: it shows how the
predicted risk evolves over a stay, with the clinical onset and the model's
alarm time both marked.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .evaluate import DT_OPTIMAL


def plot_risk_trajectory(
    label_seq: np.ndarray,
    score_seq: np.ndarray,
    *,
    threshold: float | None = None,
    pid: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Risk score over time for a single patient, with onset markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    hours = np.arange(len(score_seq))
    ax.plot(hours, score_seq, color="#2563eb", lw=1.6, label="Predicted risk")

    label_seq = np.asarray(label_seq, dtype=np.int8)
    if label_seq.any():
        first_pos = int(np.argmax(label_seq))
        t_sepsis = first_pos - DT_OPTIMAL  # = first_pos + 6
        ax.axvline(t_sepsis, color="#dc2626", lw=1.5, ls="--", label="Clinical onset")
        ax.axvline(first_pos, color="#dc2626", lw=1.0, ls=":", label="Label-positive (t_sepsis-6)")
        ax.axvspan(max(0, first_pos - 6), first_pos + 3, color="#dc2626", alpha=0.07)

    if threshold is not None:
        ax.axhline(threshold, color="#6b7280", lw=1.0, ls=":", label="Alarm threshold")
        crossing = np.flatnonzero(score_seq >= threshold)
        if crossing.size:
            ax.axvline(int(crossing[0]), color="#16a34a", lw=1.5, ls="--", label="Model alarm")

    ax.set_xlabel("ICU hour")
    ax.set_ylabel("Risk")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Risk trajectory{' for ' + pid if pid else ''}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return ax


def plot_missingness_heatmap(frame: pd.DataFrame, columns: list[str], ax: plt.Axes | None = None) -> plt.Axes:
    """Mean missingness rate by hour-since-admission for each variable."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    by_hour = frame.groupby("hour")[columns].apply(lambda g: g.isna().mean())
    sns.heatmap(by_hour.T, cmap="rocket_r", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Fraction missing"})
    ax.set_xlabel("Hour since admission")
    ax.set_ylabel("Variable")
    ax.set_title("Missingness over time")
    return ax


def plot_time_to_sepsis(records: list, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    onsets = [r.sepsis_onset for r in records if r.is_sepsis]
    onsets_hours = [o for o in onsets if o is not None]
    sns.histplot(onsets_hours, bins=40, ax=ax, color="#2563eb")
    ax.set_xlabel("Hour of label-positive onset (t_sepsis - 6)")
    ax.set_ylabel("Patients")
    ax.set_title(f"Time to first label-positive (n={len(onsets_hours)} septic)")
    return ax


def plot_calibration(
    label_seqs: list[np.ndarray],
    score_seqs: list[np.ndarray],
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    label: str = "model",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    y = np.concatenate(label_seqs).astype(np.int8)
    s = np.concatenate(score_seqs).astype(float)
    s = np.clip(s, 0, 1)
    prob_true, prob_pred = calibration_curve(y, s, n_bins=n_bins, strategy="quantile")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, ls="--")
    ax.plot(prob_pred, prob_true, "o-", lw=1.5, label=label)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False)
    return ax


def save_trajectory_grid(
    label_seqs: list[np.ndarray],
    score_seqs: list[np.ndarray],
    pids: list[str],
    out_path: str | Path,
    *,
    threshold: float | None = None,
    n_septic: int = 6,
    n_nonseptic: int = 3,
) -> None:
    """Save a multi-panel figure: a few interesting septic patients on top, a
    few non-septic patients below as controls."""
    septic_idx = [i for i, l in enumerate(label_seqs) if np.any(l)]
    nonseptic_idx = [i for i, l in enumerate(label_seqs) if not np.any(l)]
    rng = np.random.default_rng(0)
    sep_pick = rng.choice(septic_idx, size=min(n_septic, len(septic_idx)), replace=False) if septic_idx else []
    non_pick = rng.choice(nonseptic_idx, size=min(n_nonseptic, len(nonseptic_idx)), replace=False) if nonseptic_idx else []

    chosen = list(sep_pick) + list(non_pick)
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.6 * len(chosen)), squeeze=False)
    for ax, idx in zip(axes.ravel(), chosen):
        plot_risk_trajectory(label_seqs[idx], score_seqs[idx], threshold=threshold, pid=pids[idx], ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
