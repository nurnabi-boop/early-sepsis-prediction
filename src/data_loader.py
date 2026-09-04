"""PhysioNet 2019 Sepsis Challenge data loader.

Each patient is a single PSV file with one row per ICU-hour. The file lives in
either `training_setA/` (Hospital A: Beth Israel) or `training_setB/` (Hospital
B: Emory). Columns are 40 fixed features plus the `SepsisLabel` target.

The label convention from the challenge: `SepsisLabel = 1` for every row at
`t_sepsis - 6` and later, where `t_sepsis` is the clinical onset time. Models
are therefore implicitly trained to predict 6 hours ahead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

VITALS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
DEMOGRAPHICS = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
FEATURES = VITALS + LABS + DEMOGRAPHICS
TARGET = "SepsisLabel"
ALL_COLUMNS = FEATURES + [TARGET]


@dataclass
class PatientRecord:
    """One patient's full ICU stay."""
    pid: str
    hospital: str           # "A" or "B"
    df: pd.DataFrame        # rows = hours, columns = ALL_COLUMNS
    sepsis_onset: int | None  # earliest hour with SepsisLabel == 1, or None

    @property
    def is_sepsis(self) -> bool:
        return self.sepsis_onset is not None

    @property
    def length(self) -> int:
        return len(self.df)


def _hospital_for_dir(dirname: str) -> str:
    name = dirname.lower()
    if "seta" in name or name.endswith("a"):
        return "A"
    if "setb" in name or name.endswith("b"):
        return "B"
    return "?"


def _onset_from_labels(labels: np.ndarray) -> int | None:
    where = np.flatnonzero(labels.astype(np.int8) == 1)
    return int(where[0]) if where.size else None


def load_patient(path: str | os.PathLike) -> PatientRecord:
    path = Path(path)
    df = pd.read_csv(path, sep="|")
    # Tolerate datasets that ship without some columns by reindexing.
    df = df.reindex(columns=ALL_COLUMNS)
    pid = path.stem
    hospital = _hospital_for_dir(path.parent.name)
    onset = _onset_from_labels(df[TARGET].fillna(0).to_numpy())
    return PatientRecord(pid=pid, hospital=hospital, df=df, sepsis_onset=onset)


def iter_patient_paths(root: str | os.PathLike) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return iter(())
    return sorted(root.glob("*.psv"))


def load_dataset(
    data_root: str | os.PathLike,
    hospitals: tuple[str, ...] = ("A", "B"),
    subset: int | None = None,
    seed: int = 0,
) -> list[PatientRecord]:
    """Load patient records from `data_root/training_setX/`.

    `subset` caps the number of patients sampled (per hospital, proportionally),
    which is the right knob for the "validate the pipeline on 5k" workflow.
    """
    data_root = Path(data_root)
    rng = np.random.default_rng(seed)

    paths_by_hosp: dict[str, list[Path]] = {}
    for h in hospitals:
        candidates = [
            data_root / f"training_set{h}",
            data_root / f"training_{h}",
            data_root / h,
        ]
        for d in candidates:
            paths = list(iter_patient_paths(d))
            if paths:
                paths_by_hosp[h] = paths
                break

    if subset is not None and paths_by_hosp:
        total = sum(len(v) for v in paths_by_hosp.values())
        sampled: dict[str, list[Path]] = {}
        for h, paths in paths_by_hosp.items():
            quota = max(1, round(subset * len(paths) / total))
            idx = rng.choice(len(paths), size=min(quota, len(paths)), replace=False)
            sampled[h] = [paths[i] for i in sorted(idx)]
        paths_by_hosp = sampled

    records: list[PatientRecord] = []
    for h, paths in paths_by_hosp.items():
        for p in tqdm(paths, desc=f"Hospital {h}", unit="pt"):
            try:
                records.append(load_patient(p))
            except Exception as e:  # corrupted file — skip with a warning
                print(f"[load_dataset] skipping {p}: {e}")
    return records


def split_by_hospital(
    records: list[PatientRecord],
) -> dict[str, list[PatientRecord]]:
    out: dict[str, list[PatientRecord]] = {}
    for r in records:
        out.setdefault(r.hospital, []).append(r)
    return out


def patient_train_test_split(
    records: list[PatientRecord],
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[PatientRecord], list[PatientRecord]]:
    """Split *patients*, never rows — leakage across rows of the same stay
    inflates AUROC dramatically on this dataset."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    cut = int(len(records) * (1 - test_frac))
    train = [records[i] for i in idx[:cut]]
    test = [records[i] for i in idx[cut:]]
    return train, test
