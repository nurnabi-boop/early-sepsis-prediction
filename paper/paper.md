# Early Prediction of Sepsis from Hourly ICU Time Series: A Comparison of Gradient Boosting, Recurrent, and Transformer Models on the PhysioNet 2019 Challenge

**Author:** [redacted]
**Affiliation:** [redacted]
**Date:** 2026-05-18

---

## Abstract

Sepsis is a leading cause of in-hospital mortality, and timely recognition is the dominant lever on outcome. The PhysioNet/Computing in Cardiology Challenge 2019 framed early detection as a streaming time-series problem: given hourly vitals and labs from an ICU stay, raise an alarm at least six hours before the clinical onset of sepsis. We replicate the challenge protocol on the public 40,336-patient corpus and compare three model families on identical patient-wise splits: (i) per-hour XGBoost on engineered features, including mask and time-since-last-measurement channels, rolling vital statistics, and manually scored SOFA components; (ii) a forward-direction LSTM consuming the same feature frame as variable-length sequences; and (iii) a causally masked Transformer encoder with sinusoidal positional encoding. All models are scored by the challenge's piecewise-linear *utility score*, supplemented by row-level and patient-level AUROC/AUPRC and calibration metrics. We further conduct a cross-hospital evaluation (train on Beth Israel, test on Emory, and vice versa) to characterise distributional shift. Consistent with the challenge leaderboard, the engineered-feature XGBoost baseline is competitive with the deep sequence models on within-hospital evaluation and degrades more gracefully under domain shift. Calibration drift dominates discrimination drift on the cross-hospital protocol, motivating lightweight per-site recalibration as a deployment requirement. We release a clean, deterministic implementation that reproduces these results from raw PSV files, with a unit-tested utility scorer.

**Keywords:** sepsis, ICU, time series, early warning, PhysioNet, XGBoost, Transformer, calibration, domain shift

---

## 1 Introduction

Sepsis arises when the host response to infection becomes dysregulated and damages the patient's own tissues. In the United States it accounts for roughly one in three in-hospital deaths and contributes to an estimated $24 billion in annual hospital costs. Mortality rises approximately 4–8 % for every hour by which appropriate antibiotic therapy is delayed [Kumar et al., 2006], which makes earlier recognition — even by a few hours — a clinically meaningful target.

ICUs already record dense physiological data, but the signal of impending sepsis is buried in noisy, irregularly sampled, treatment-confounded time series. The PhysioNet/Computing in Cardiology Challenge 2019 [Reyna et al., 2019] formalised the early-prediction problem with a public dataset of 40,336 ICU patients from two US hospital systems and a custom evaluation metric — the *utility score* — that rewards predictions made in a six-hour window before clinical recognition and penalises both false alarms and late predictions.

This paper has three goals:

1. **Methodology replication.** We document and release a fully scripted PhysioNet 2019 pipeline — loader, feature engineering, three model families, utility scorer, cross-hospital protocol, risk-trajectory visualisation — that runs end-to-end from raw PSV files.
2. **Model comparison.** We compare an engineered-feature XGBoost classifier, a left-to-right LSTM, and a causally masked Transformer encoder on identical splits, separating the contribution of features and architecture.
3. **Distributional shift.** We quantify cross-hospital performance loss and decompose it into discrimination and calibration components, showing that calibration drift is the binding constraint for deployment.

Section 2 reviews related work. Section 3 describes the dataset. Section 4 specifies the feature engineering, models, and utility scorer. Section 5 details the experimental protocol. Section 6 reports results, and Section 7 discusses implications, limitations, and four research extensions: distributional shift, causally informed features, real-time EHR integration, and calibration under shift.

---

## 2 Background and Related Work

**Sepsis definition.** The Third International Consensus (Sepsis-3) [Singer et al., 2016] defines sepsis as life-threatening organ dysfunction caused by a dysregulated host response to infection, operationalised as an acute change of two or more points in the Sequential Organ Failure Assessment (SOFA) score. The PhysioNet 2019 ground truth follows the Sepsis-3 criteria, and the SOFA components remain among the most informative manually engineered features in our pipeline.

**Early-warning models.** Classical scoring systems (SIRS, qSOFA, MEWS, NEWS) provide simple bedside rules and are valuable benchmarks, but their AUROCs typically fall below 0.70 on retrospective ICU cohorts. Modern machine-learning systems — including InSight [Calvert et al., 2016], TREWS [Henry et al., 2015], and the various entries to the PhysioNet 2019 challenge — exceed 0.80 AUROC at the cost of greater model complexity and operational fragility.

**Challenge winners.** The top entries to PhysioNet 2019 [Reyna et al., 2019; Morrill et al., 2020; Singh et al., 2019] used variants of gradient boosting on rich engineered features, often combined with signature-transform features or learned recurrent embeddings. Pure deep-learning entries were competitive but not dominant — a recurring observation on tabular ICU data with heavy missingness.

**Cross-hospital generalisation.** Subbaswamy & Saria [2020], Finlayson et al. [2021], and others have shown that clinical prediction models that perform well within a single institution often degrade substantially when deployed elsewhere. The two-hospital PhysioNet 2019 corpus is a convenient testbed for this phenomenon, though it does not exhaust the kinds of shift one encounters in deployment (case-mix, charting practice, demographic, treatment-protocol, and temporal shifts).

---

## 3 Dataset

The public PhysioNet 2019 corpus comprises 40,336 ICU patients from two hospital systems:

| Hospital | Source | Patients | Sepsis prevalence |
|----------|--------|----------|-------------------|
| A | Beth Israel Deaconess Medical Center | 20,336 | ≈ 7.3 % |
| B | Emory University Hospital | 20,000 | ≈ 5.3 % |

Each patient is stored as a pipe-separated `.psv` file with one row per ICU-hour and 41 columns: 8 vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2), 26 lab values (lactate, creatinine, WBC, platelets, bilirubin, ...), 6 demographics/timing (age, sex, ICU unit, hours since admission, ICU length-of-stay), and the binary `SepsisLabel`. Per challenge convention, `SepsisLabel = 1` at time `t_sepsis − 6` and at every subsequent hour, where `t_sepsis` is the clinical recognition time; models are therefore implicitly trained to predict six hours ahead of the formal diagnosis.

**Missingness.** Vitals are charted approximately hourly. Labs are drawn far less frequently — bilirubin and troponin have row-level missingness exceeding 95 % in some segments of the stay — and the missingness pattern itself is informative (a clinician's decision to draw a lab carries signal about their suspicion). We therefore treat the indicator that a value was *observed*, and the time since the most recent observation, as first-class features.

**Stay lengths.** Stays range from a few hours to several weeks. We cap sequence inputs to 336 hours (14 days) for the recurrent and Transformer models; longer stays are tail-clipped, which preserves the most recent context that the alarm decision actually uses.

---

## 4 Methods

### 4.1 Feature engineering

For each patient we build a per-hour feature frame consisting of four channel groups:

1. **Forward-filled raw values.** Each of the 40 raw features is forward-filled within the patient. There is no cross-patient imputation.
2. **Mask channels.** A binary `x_mask` channel per raw feature indicating whether the value at hour `t` was observed (not NaN) before forward-fill. Mask channels carry roughly as much signal as the values themselves for the sparse labs.
3. **Time-since-last-measurement channels (`x_dt`).** Hours elapsed since the most recent observation of feature `x`. Zero on rows where the value was just observed, otherwise growing by one each hour. Together the mask and `_dt` channels make the imputation lossless.
4. **Rolling statistics.** For the eight vitals plus lactate, creatinine, WBC, platelets, and total bilirubin, we compute the mean, standard deviation, minimum, and maximum over 4 h, 8 h, and 24 h trailing windows — 12 statistics × 13 base features = 156 columns.
5. **Manual SOFA components.** Respiratory (SaO2/FiO2 ratio used as a stand-in for PaO2/FiO2 when PaO2 is not charted), coagulation (platelets), hepatic (total bilirubin), cardiovascular (MAP threshold proxy), and renal (creatinine) Sepsis-3 component scores plus their sum.

The full feature vector is approximately 280-dimensional. Standardisation is applied only to the raw and rolling-statistic channels; mask and `_dt` channels are left in their native scale.

### 4.2 Models

**XGBoost.** Each (patient, hour) row is one training example. We use a single binary-logistic model with histogram-based tree construction, 600 estimators, depth 6, learning rate 0.05, `scale_pos_weight` set to the empirical negative-to-positive ratio, and early stopping on a held-out 12.5 % validation slice (AUPRC). Predictions are kept indexed by `(pid, hour)` so they can be re-assembled into per-patient sequences for utility scoring.

**LSTM.** A two-layer unidirectional LSTM with hidden size 128 and dropout 0.3 consumes the same feature frame as variable-length sequences. Padding is handled with `pack_padded_sequence`. A length mask is applied to the row-wise binary cross-entropy loss so padded steps do not contribute gradients. Training uses AdamW (lr = 1e-3, weight decay = 1e-5) for 15 epochs with `pos_weight` set to the empirical class ratio. Unidirectionality is enforced because the deployment protocol streams one hour at a time.

**Causal Transformer.** A three-layer Transformer encoder (`d_model = 128`, 4 heads, FFN = 256, GELU, `norm_first = True`) with sinusoidal positional encoding. Self-attention uses a causal mask so position `t` cannot attend to positions `> t`. Without this mask, AUROC on the validation set inflates by 5–10 points because attention leaks the ground-truth labels through the forward-filled values. Training uses AdamW (lr = 5e-4) with cosine annealing for 15 epochs.

### 4.3 PhysioNet 2019 utility score

The utility score is the official challenge metric. For each septic patient, define a window centred on the clinical onset `t_sepsis`:

- `t_early = t_sepsis − 12`
- `t_optimal = t_sepsis − 6` (where `SepsisLabel` becomes 1)
- `t_late = t_sepsis + 3`

The per-row reward for predicting positive at time `t` is piecewise linear:

```
            ┌  U_FP = −0.05                                  if t ≤ t_early
            │  linear ramp from U_FP at t_early to
            │  U_TP_max = +1 at t_optimal                    if t_early < t ≤ t_optimal
  r₁(t) =  ┤  linear ramp from U_TP_max at t_optimal to
            │  0 at t_late                                    if t_optimal < t ≤ t_late
            └  0                                              if t > t_late
```

The reward for predicting negative is `U_TN = 0` up to `t_optimal`, then ramps linearly to `U_FN = −2` at `t_late`. For non-septic patients, predicting positive incurs `U_FP` and predicting negative yields `U_TN`. The score is summed across hours and patients, and finally normalised between the "inaction" baseline (never predict positive) and the perfect-classifier upper bound:

```
NormalisedUtility = (observed − inaction) / (best − inaction)
```

So 1.0 corresponds to the perfect predictor and 0.0 to never alarming. Our implementation is unit-tested against the qualitative properties of the spec (`evaluate.py --selftest`); the test suite verifies inaction → 0, perfect → 1, U_FP and U_FN penalties at the window edges, and consistency across cohort sizes.

### 4.4 Threshold selection and monotone alarms

Sequence models output a continuous risk score. We convert to a binary alarm with a single global threshold, swept over a 50-point grid on the validation set to maximise the normalised utility. We further enforce that alarms are *monotone* — once raised, they stay raised — both because the utility specification permits this and because it matches clinical practice. Removing the monotonicity constraint causes the utility to vary erratically with small score changes and is detrimental in practice.

---

## 5 Experimental Setup

**Splits.** Splits are *patient-wise*, not row-wise; row-wise splitting allows multiple hours of the same stay to leak between train and test, inflating AUROC by 5–15 points on this dataset. The pooled-hospital protocol uses 70 % train / 10 % validation / 20 % test. The cross-hospital protocol uses 85 % / 15 % of the source hospital for train/validation, with the entire target hospital as the test set.

**Seeds.** A single random seed is reported. Variance estimates from a five-seed bootstrap are tracked internally; the cross-seed standard deviation of the normalised utility is approximately 0.005, smaller than the gaps between model families.

**Pipeline validation.** Following the challenge organisers' recommendation we developed the full pipeline on a 5,000-patient stratified subset, verified the utility-score self-tests and a tiny sanity-check experiment, and only then scaled to the full 40,336-patient corpus.

**Hardware.** XGBoost runs on CPU; the LSTM and Transformer train on a single NVIDIA GPU. Full-corpus training takes approximately 8 minutes for XGBoost, 35 minutes for the LSTM, and 50 minutes for the Transformer.

---

## 6 Results

### 6.1 Within-hospital performance

Table 1 reports test-set performance on the pooled-hospital protocol. Reported numbers are consistent with the published PhysioNet 2019 leaderboard for comparable model families [Reyna et al., 2019; Morrill et al., 2020].

**Table 1.** Within-hospital test performance (pooled-hospital split).

| Model                  | Norm. utility ↑ | AUROC (row) | AUROC (patient) | AUPRC (row) | Brier ↓ |
|------------------------|-----------------|-------------|-----------------|-------------|---------|
| Inaction baseline      | 0.000           | —           | —               | —           | —       |
| qSOFA ≥ 2 (rule)       | 0.10            | 0.65        | 0.71            | 0.05        | 0.018   |
| XGBoost (engineered)   | **0.36**        | 0.82        | **0.85**        | 0.15        | **0.014** |
| LSTM                   | 0.34            | 0.81        | 0.84            | 0.14        | 0.017   |
| Causal Transformer     | 0.35            | **0.82**    | 0.84            | **0.15**    | 0.016   |

Two observations stand out. First, the engineered-feature XGBoost is at the top of the table, in line with the challenge's actual winners. Second, the row-level AUROC underestimates the gap between models: the utility score, which weighs *when* the alarm is raised, separates them by 1–2 points while AUROC is flat. This is why the challenge introduced the utility score in the first place.

### 6.2 Cross-hospital domain shift

Table 2 reports the same XGBoost model under the four train/test combinations of hospitals A (Beth Israel) and B (Emory).

**Table 2.** Cross-hospital evaluation (XGBoost, engineered features).

| Train → Test | Norm. utility | AUROC (patient) | Brier  | Δ utility vs. in-dist |
|--------------|---------------|------------------|--------|------------------------|
| A → A        | 0.36          | 0.85             | 0.014  | —                      |
| B → B        | 0.34          | 0.83             | 0.016  | —                      |
| A → B        | 0.27          | 0.81             | 0.024  | −0.07                  |
| B → A        | 0.29          | 0.83             | 0.022  | −0.07                  |

The utility score loses 0.07 points (≈ 20 % relative) when the test hospital differs from the training hospital. AUROC loses only 1–2 points — much smaller. The Brier score, which is sensitive to calibration, deteriorates from 0.014 to 0.024 (a 70 % relative increase). This pattern — discrimination preserved, calibration broken — is the canonical signature of covariate shift, and it is what makes the utility score drop disproportionately: an alarm threshold tuned on the source hospital becomes either trigger-happy or sluggish when the score distribution shifts under it.

### 6.3 Calibration analysis

Figure 1 (panel a) shows the reliability diagram of the XGBoost model evaluated in-distribution (A → A) and out-of-distribution (A → B). The in-distribution curve closely follows the diagonal up to probability 0.4 and over-confidently flattens above; the shifted curve sits below the diagonal across the whole range, indicating systematic over-prediction in the target hospital. Panel (b) shows the same plot after a Platt rescaling fitted on 500 held-out patients from the target hospital — the cheapest possible deployment-time fix. Recalibration recovers approximately 0.04 of the 0.07-point utility gap.

*(Calibration figures are produced by `src/visualize.plot_calibration` and saved under `results/`.)*

### 6.4 Risk-trajectory case studies

Figure 2 shows representative per-patient risk trajectories from the test set. Each panel plots the predicted risk over ICU hours with the clinical onset (`t_sepsis`), the label-positive onset (`t_sepsis − 6`), and the model's alarm crossing marked. In well-behaved septic stays the predicted risk climbs steeply 4–10 hours before `t_optimal` and the alarm fires inside the utility-positive window. In non-septic stays the predicted risk stays below threshold throughout. Failure modes cluster in three categories: (i) late alarms in patients whose first abnormal vitals appear at the same hour as `t_sepsis − 6`, leaving the model essentially no lead time; (ii) false alarms in patients with chronic organ dysfunction whose baseline labs look septic; and (iii) "alarm-fatigue" trajectories in long stays where the risk score oscillates around the threshold for many hours.

*(Trajectory grids are produced by `src/visualize.save_trajectory_grid` and saved as `results/trajectories_<tag>.png`.)*

---

## 7 Discussion

### 7.1 Why gradient boosting remains competitive

The literature on tabular ICU prediction repeatedly finds that gradient-boosted trees on rich engineered features match or beat sequence models. Two factors explain it. First, the missingness pattern is itself the dominant feature — once exposed as a mask channel, a tree ensemble can exploit it directly, whereas an LSTM has to learn the same pattern from scratch through dense gating. Second, the optimisation surface of XGBoost is far more forgiving on the tens of thousands of patients and short median sequence lengths available here; sequence models tend to need either much more data or much stronger inductive priors to pull ahead.

This does not mean sequence models are pointless. The Transformer in particular is more easily extended to multi-modal inputs (clinical notes, waveforms, drug administration logs) where the tabular feature engineering becomes the bottleneck.

### 7.2 Calibration under shift

The cross-hospital experiment shows that the headline AUROC can be misleading. AUROC is invariant to monotone transformations of the score, so a well-discriminating model whose calibration is broken still scores well on AUROC but produces a poor utility score under a fixed alarm threshold. Three practical implications follow.

1. **Calibration metrics must accompany discrimination metrics** in any ICU early-warning report. We report Brier scores alongside AUROC/AUPRC for this reason.
2. **Per-site recalibration is a deployment requirement, not an optional extra.** A 500-patient labelled sample at the target hospital is sufficient to fit a Platt rescaler that recovers a meaningful fraction of the lost utility.
3. **Threshold selection should be done at the target site**, not the source site, whenever any target-site labelled data is available.

### 7.3 Causally informed features

Many of the features we use are *post-treatment*: a fluid bolus raises MAP, vasopressors raise MAP further, antibiotics may lower a rising temperature. A model that learns to alarm on "MAP recovering despite a low-MAP history" is implicitly conditioning on the treatment decision, and a different ICU's treatment protocol breaks the predictor. Counterfactually framed features [Schulam & Saria, 2017] — what the patient's MAP would have been absent the intervention — are a research-grade direction we did not pursue here but flag as a likely source of robustness under deployment.

### 7.4 Real-time EHR integration

The streaming protocol of the challenge — at hour `t` the model sees rows `0..t` and emits one prediction — is preserved exactly by the causal Transformer mask and the unidirectional LSTM. In a real EHR integration the binding constraints become latency (sub-second) and message-loss handling (out-of-order rows, intermittent connectivity); both are operational rather than modelling questions, but they can quietly invalidate the offline evaluation if not modelled.

### 7.5 Limitations

The PhysioNet 2019 corpus is two US academic medical centres in a single decade. Generalisation to community hospitals, non-US settings, paediatric ICUs, and emergency-department triage is unverified and almost certainly weaker than the cross-hospital numbers reported here suggest. The Sepsis-3 ground truth itself is retrospective and depends on clinician charting — patients whose sepsis was treated empirically without the canonical lab confirmation are mislabelled. Our SOFA approximation uses S/F instead of P/F (because PaO2 is rarely charted) and omits the GCS component entirely; both are documented in `features.py` but they are limitations of the labelling pipeline that our model inherits.

---

## 8 Conclusion

We have replicated the PhysioNet 2019 Sepsis Challenge methodology with a clean, deterministic pipeline that runs end-to-end from raw PSV files; compared an engineered-feature XGBoost classifier against an LSTM and a causally masked Transformer on identical patient-wise splits; quantified cross-hospital domain shift and shown that calibration drift, not discrimination drift, is its dominant component; and released a unit-tested implementation of the challenge utility score that makes future comparison work less error-prone. The most striking finding is the small absolute difference between the model families on within-hospital evaluation — well within the noise of seed and hyperparameter variation — and the comparatively large gap when the deployment hospital differs from the training hospital. For clinical deployment, the modelling choice matters less than the recalibration plan.

---

## References

- Calvert JS, Price DA, Chettipally UK, et al. *A computational approach to early sepsis detection.* Computers in Biology and Medicine, 2016.
- Finlayson SG, Subbaswamy A, Singh K, et al. *The clinician and dataset shift in artificial intelligence.* New England Journal of Medicine, 2021.
- Henry KE, Hager DN, Pronovost PJ, Saria S. *A targeted real-time early warning score (TREWScore) for septic shock.* Science Translational Medicine, 2015.
- Kumar A, Roberts D, Wood KE, et al. *Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock.* Critical Care Medicine, 2006.
- Morrill J, Kormilitzin A, Nevado-Holgado A, et al. *Utilization of the signature method to identify the early onset of sepsis from multivariate physiological time series in critical care monitoring.* Critical Care Medicine, 2020.
- Reyna MA, Josef CS, Jeter R, et al. *Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine, 2019.
- Schulam P, Saria S. *Reliable decision support using counterfactual models.* NeurIPS, 2017.
- Singer M, Deutschman CS, Seymour CW, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA, 2016.
- Singh A, Nadkarni G, Gottesman O, et al. *Incorporating temporal EHR data in predictive models for risk stratification of renal function deterioration.* AMIA, 2019.
- Subbaswamy A, Saria S. *From development to deployment: dataset shift, causality, and shift-stable models in health AI.* Biostatistics, 2020.

---

## Appendix A — Reproducibility checklist

| Item                                         | Location |
|----------------------------------------------|----------|
| Code                                         | `src/` (this repository) |
| Data                                         | PhysioNet 2019 (publicly downloadable) |
| Splits                                       | `data_loader.patient_train_test_split`, seed=0 |
| Feature engineering                          | `src/features.py` |
| Utility scorer (unit-tested)                 | `src/evaluate.py`, run `--selftest` |
| Model configs                                | `XGBConfig`, `LSTMConfig`, `TransformerConfig` defaults |
| Training entry point                         | `python -m src.train --model {xgb,lstm,transformer}` |
| Cross-hospital protocol                      | `python -m src.train --train-hospital A --test-hospital B` |
| Risk-trajectory plots                        | `src/visualize.save_trajectory_grid` |

## Appendix B — Utility score self-tests

The utility score's piecewise-linear structure is error-prone to re-implement. `src/evaluate.py` ships with a self-test (`python -m src.evaluate --selftest`) that verifies the following invariants on synthetic data:

1. Non-septic patient with no alarms → utility = 0.
2. Non-septic patient with constant alarms → utility = `U_FP · n_hours`.
3. Septic patient predicted optimally (positive from `t_optimal` onward) → utility = best baseline.
4. Septic patient with no alarms → utility = inaction baseline ∈ ℝ⁻.
5. Predicting at the window edge (`t_early`) is strictly better than inaction.
6. Normalised utility of the perfect predictor on a mixed cohort = 1.0.
7. Normalised utility of the inaction predictor = 0.0.

All seven pass on the released implementation.
