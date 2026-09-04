"""Feature engineering for the PhysioNet 2019 sepsis pipeline.

The challenge is dominated by missingness: lab values may be charted only every
12-24 hours while vitals come hourly. The mask channel (was-this-observed) and
time-since-last-measurement carry as much signal as the imputed value itself,
so we expose all three explicitly.

Rolling statistics are computed on the *forward-filled* series (so the value at
hour t is the most recent observation by hour t -- no lookahead), with separate
4 / 8 / 24-hour windows. SOFA components are scored from the standard cutoffs
in Singer et al. 2016 (the Sepsis-3 definition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import VITALS, LABS, DEMOGRAPHICS, FEATURES, TARGET, PatientRecord

ROLL_WINDOWS = (4, 8, 24)
ROLL_FEATURES = VITALS + ["Lactate", "Creatinine", "WBC", "Platelets", "Bilirubin_total"]


def _forward_fill_with_mask(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (filled, mask, time_since_last) over the FEATURES columns.

    `mask[col][t] = 1` iff the raw value at hour t was observed (not NaN).
    `time_since_last[col][t]` is hours since the most recent observation; 0 on
    rows where the value was just observed, growing by 1 each subsequent hour.
    """
    sub = df[FEATURES]
    mask = sub.notna().astype(np.int8)
    filled = sub.ffill()

    # Hours since the last observation per column, computed from the mask.
    delta = pd.DataFrame(index=sub.index, columns=FEATURES, dtype=float)
    for col in FEATURES:
        m = mask[col].to_numpy()
        out = np.empty(len(m), dtype=float)
        last = np.nan
        for i, observed in enumerate(m):
            if observed:
                last = i
                out[i] = 0.0
            else:
                out[i] = (i - last) if not np.isnan(last) else np.nan
        delta[col] = out
    return filled, mask, delta


def _rolling_stats(filled: pd.DataFrame) -> pd.DataFrame:
    """Per-window mean/std/min/max for a curated subset of columns.

    Vitals + a few labs only — including all 40 features at three windows blows
    the column count up to several hundred and adds little signal for the rare
    labs.
    """
    out = {}
    for col in ROLL_FEATURES:
        if col not in filled.columns:
            continue
        s = filled[col]
        for w in ROLL_WINDOWS:
            r = s.rolling(window=w, min_periods=1)
            out[f"{col}_mean_{w}h"] = r.mean()
            out[f"{col}_std_{w}h"] = r.std().fillna(0.0)
            out[f"{col}_min_{w}h"] = r.min()
            out[f"{col}_max_{w}h"] = r.max()
    return pd.DataFrame(out, index=filled.index)


# ---- SOFA components (Sepsis-3 cutoffs) -----------------------------------
def _score_pao2_fio2(pao2: float, fio2: float) -> int:
    """Respiratory SOFA from PaO2/FiO2 ratio. FiO2 may be on either 0-1 or
    0-100 scales depending on the unit's charting convention -- normalise."""
    if np.isnan(pao2) or np.isnan(fio2) or fio2 <= 0:
        return 0
    if fio2 > 1.0:
        fio2 = fio2 / 100.0
    ratio = pao2 / fio2
    if ratio < 100: return 4
    if ratio < 200: return 3
    if ratio < 300: return 2
    if ratio < 400: return 1
    return 0


def _score_platelets(plt: float) -> int:
    if np.isnan(plt): return 0
    if plt < 20:  return 4
    if plt < 50:  return 3
    if plt < 100: return 2
    if plt < 150: return 1
    return 0


def _score_bilirubin(bili: float) -> int:
    if np.isnan(bili): return 0
    if bili >= 12.0: return 4
    if bili >= 6.0:  return 3
    if bili >= 2.0:  return 2
    if bili >= 1.2:  return 1
    return 0


def _score_map(mapv: float) -> int:
    """Cardiovascular SOFA without vasopressor doses (which the dataset does
    not chart). MAP < 70 mmHg as a single-point proxy for early hypotension."""
    if np.isnan(mapv): return 0
    if mapv < 70: return 1
    return 0


def _score_creatinine(cr: float) -> int:
    if np.isnan(cr): return 0
    if cr >= 5.0: return 4
    if cr >= 3.5: return 3
    if cr >= 2.0: return 2
    if cr >= 1.2: return 1
    return 0


def compute_sofa_components(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row SOFA component scores plus a coarse total.

    The dataset is missing GCS and vasopressor dose, so the neuro and full
    cardiovascular components are approximated."""
    pao2 = df.get("PaCO2")  # PaO2 isn't charted; SaO2 is the closest proxy
    sao2 = df.get("SaO2")
    fio2 = df.get("FiO2")
    plt = df.get("Platelets")
    bili = df.get("Bilirubin_total")
    mapv = df.get("MAP")
    cr = df.get("Creatinine")

    n = len(df)
    resp = np.zeros(n, dtype=np.int8)
    if sao2 is not None and fio2 is not None:
        # Use SaO2/FiO2 (S/F) instead of PaO2/FiO2 (P/F) when PaO2 is absent —
        # standard ICU substitute when ABGs aren't drawn that hour.
        s = sao2.to_numpy()
        f = fio2.to_numpy()
        for i in range(n):
            sv, fv = s[i], f[i]
            if np.isnan(sv) or np.isnan(fv) or fv <= 0:
                continue
            if fv > 1.0: fv = fv / 100.0
            sf = sv / fv
            if sf < 89:   resp[i] = 4
            elif sf < 235: resp[i] = 3
            elif sf < 315: resp[i] = 2
            elif sf < 400: resp[i] = 1

    coag = np.array([_score_platelets(v) for v in (plt if plt is not None else [np.nan]*n)], dtype=np.int8)
    liver = np.array([_score_bilirubin(v) for v in (bili if bili is not None else [np.nan]*n)], dtype=np.int8)
    cardio = np.array([_score_map(v) for v in (mapv if mapv is not None else [np.nan]*n)], dtype=np.int8)
    renal = np.array([_score_creatinine(v) for v in (cr if cr is not None else [np.nan]*n)], dtype=np.int8)

    total = resp + coag + liver + cardio + renal
    return pd.DataFrame({
        "sofa_resp": resp,
        "sofa_coag": coag,
        "sofa_liver": liver,
        "sofa_cardio": cardio,
        "sofa_renal": renal,
        "sofa_total": total,
    }, index=df.index)


# ---- Public API -----------------------------------------------------------
def featurize_patient(record: PatientRecord) -> pd.DataFrame:
    """Build the full per-row feature frame for one patient.

    Columns: forward-filled values, mask channels, time-since-last channels,
    rolling stats, SOFA components. Plus the pid/hour index and `SepsisLabel`
    so downstream code can split / score without re-joining.
    """
    df = record.df
    filled, mask, delta = _forward_fill_with_mask(df)
    rolled = _rolling_stats(filled)
    sofa = compute_sofa_components(filled)

    mask = mask.add_suffix("_mask")
    delta = delta.add_suffix("_dt")

    out = pd.concat([filled, mask, delta, rolled, sofa], axis=1)
    out["pid"] = record.pid
    out["hospital"] = record.hospital
    out["hour"] = np.arange(len(out))
    out[TARGET] = df[TARGET].fillna(0).astype(np.int8).to_numpy()
    return out


def featurize_many(records: list[PatientRecord]) -> pd.DataFrame:
    frames = [featurize_patient(r) for r in records]
    return pd.concat(frames, axis=0, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All learnable feature columns -- everything except identifiers and
    the target."""
    drop = {"pid", "hospital", "hour", TARGET}
    return [c for c in frame.columns if c not in drop]
