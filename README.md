# Early Sepsis Prediction (PhysioNet 2019 Challenge)

Predict the onset of sepsis 6 hours before clinical recognition from hourly ICU vitals and labs, replicating the methodology of the [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).

## Dataset

- **40,336 ICU patients** from two hospital systems
  - Hospital A: Beth Israel Deaconess Medical Center
  - Hospital B: Emory University Hospital
- One PSV (pipe-separated) file per patient, one row per hour
- 40 features: 8 vitals, 26 lab values, 6 demographics/timing
- `SepsisLabel = 1` from `t_sepsis - 6` onward (per challenge definition)

Place the unpacked challenge data under `data/training_setA/` and `data/training_setB/`.

## Methodology

1. **Loader** that returns per-patient variable-length sequences with hospital provenance
2. **EDA** — missingness over time, sepsis prevalence, time-to-onset distribution
3. **Feature engineering**
   - Forward-fill within patient, plus a binary mask channel per variable (the missingness pattern itself is informative)
   - Rolling means / stds / min / max over 4h, 8h, 24h windows
   - Manual SOFA component scoring (PaO2/FiO2, platelets, bilirubin, MAP, creatinine)
   - Time since last measurement for every variable
4. **Models**
   - XGBoost on engineered features (very strong baseline; adapted variants won the challenge)
   - LSTM with a learned mask embedding
   - Transformer encoder with sinusoidal positional encoding
5. **Primary metric: PhysioNet 2019 utility score** (`src/evaluate.py`) — rewards early prediction in a window before onset, penalises false alarms and late predictions. Verified against the toy example in the challenge documentation.
6. Patient-level **AUROC / AUPRC** as supporting metrics
7. **Cross-hospital evaluation** — train on A → test on B and vice versa, to surface domain shift
8. **Risk trajectory plots** per patient with predicted onset and ground-truth onset overlaid

## Project structure

```
sepsis-prediction/
├── data/                   # raw PhysioNet PSV files (training_setA, training_setB)
├── src/
│   ├── data_loader.py
│   ├── features.py         # rolling stats, SOFA components, mask + delta-t channels
│   ├── models/
│   │   ├── xgb_model.py
│   │   ├── lstm_model.py
│   │   └── transformer_model.py
│   ├── train.py
│   ├── evaluate.py         # PhysioNet 2019 utility score
│   └── visualize.py        # risk trajectories
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_xgb.ipynb
│   ├── 03_sequence_models.ipynb
│   └── 04_domain_shift.ipynb
├── models/                 # fitted model artefacts
├── results/                # metrics, plots
├── paper/
│   └── paper.md            # research paper write-up
├── README.md
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Validate the pipeline on a 5k subset before scaling
python -m src.train --model xgb --subset 5000
python -m src.train --model lstm --subset 5000
python -m src.train --model transformer --subset 5000

# Cross-hospital evaluation
python -m src.train --model xgb --train-hospital A --test-hospital B
python -m src.train --model xgb --train-hospital B --test-hospital A
```

## Research extensions

- **Distributional shift across hospitals / populations.** A vs B differ in case-mix, charting cadence, and sepsis prevalence; the cross-hospital protocol surfaces what generalises.
- **Causally informed features.** Many useful features (e.g. fluid bolus → MAP) are confounded by treatment decisions; counterfactual or treatment-aware encoders are an active research direction.
- **Integration with real-time EHR streams.** The challenge protocol streams an hour at a time and forbids future leakage; this code is structured to make that constraint explicit.
- **Calibration under shift.** AUROC can stay high while calibration degrades. Per-hospital reliability diagrams and Platt / isotonic recalibration are reported alongside discrimination.

## Validation note

The PhysioNet utility score is fiddly. `src/evaluate.py` is unit-tested against the toy patient examples published with the challenge — running `python -m src.evaluate --selftest` should print scores matching the spec. Always run that before trusting tuning runs.
