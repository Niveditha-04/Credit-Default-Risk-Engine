# Model card: Credit Default Risk Engine

## Model (served: v2)

| Element | Value |
|---|---|
| Type | LightGBM, 273 trees, 8 monotone constraints |
| Trained on | 2012-2014 (train + validation) |
| Features | 69 |
| Calibrator | Platt, fit on 2015 |
| Hyperparameters | num_leaves=15, learning_rate=0.08, min_child_samples=1500, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0 |

Monotone constraints: `fico_mean` decreasing; `dti`, `loan_to_income`, `revol_util`, `bc_util`, `inq_last_6mths`, `acc_open_past_24mths` increasing; `annual_inc` decreasing.

v2 was chosen over v1 by a rule pre-registered in `docs/v2_decision_rule.md` before any v2 code ran.

| Check (2016 vintages, 24-month label) | v1 | v2 | Rule |
|---|---|---|---|
| AUC | 0.6899 | 0.6978 | >= v1 - 0.002 |
| ECE | 0.0080 | 0.0079 | <= v1 + 0.005 |
| Met | - | yes | both conditions |

A second, "corrected" label (checked after the rule's result was already known) reaches the same outcome: v1 AUC 0.6860, v2 AUC 0.6938, ECE 0.0104 both (`reports/metrics/posthoc_v2_label_robustness.json`). ECE here compares both models against a 24-month proxy, not either one's real target, so it is a relative check, not an absolute calibration claim.

## A logistic regression scorecard

WOE-binned, 17 features, trained alongside the gradient-boosting line as a baseline, evaluated on test, not served.

## Intended use

| | |
|---|---|
| Use | PD estimate at loan origination for an already-underwritten applicant, with a threshold and top-3 driver explanation |
| Not for | Joint applications, other loan terms, legal adverse-action notices, or any fairness claim (no protected attributes in this data) |

## Training data

| Split | Loans | Default rate |
|---|---|---|
| Train (2012-2013) | 143,892 | 12.70% |
| Validation (2014) | 162,570 | 13.73% |
| Test (2015) | 282,787 | 14.88% |

v2 trains on train + validation; its calibrator is fit on test (2015). v1 was evaluated on test exactly once.

## Features

| Group | Count |
|---|---|
| Fed to the model | 69 (59 raw + 10 engineered) |
| Scorecard only | 17 (WOE-filtered subset) |
| Benchmark, not model input | `grade`, `sub_grade`, `int_rate` |
| Stability check only | `zip_code`, `addr_state` |

## Explanations

LightGBM's own exact TreeSHAP, log-odds scale, same call used offline and in the served API. Top drivers on test: `fico_mean`, `acc_open_past_24mths`, `loan_to_income`, `dti`, `annual_inc` (`reports/metrics/shap_global_importance_served.csv`). Each prediction returns its own top-3 risk-increasing drivers, labeled illustrative only.

## Limitations

- Accepted loans only, no rejected-applicant counterfactual.
- Origination-time data, modeling population 2012-2015, plus a 2016 holdout check for v2.
- No protected attributes in this data; not evaluated for fairness.
- Loss rate on funded amount (cash basis, 38.03%) is an estimate from train-year charged-off loans (`reports/metrics/test_lgd_sensitivity.csv`).
- Drift monitoring is a historical replay of 2016-2018, not a live monitor.

## Serving

FastAPI, `GET /health` and `POST /score`, Docker (`python:3.11-slim` + `libgomp1`). Latency: `reports/metrics/api_latency.json`.

## History: v1

Served before v2, trained on 2012-2013 only, no monotone constraints, evaluated on test exactly once:

| Model | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| LightGBM (v1) | 0.6872 | 0.3745 | 0.2697 | 0.11966 |
| XGBoost | 0.6886 | 0.3772 | 0.2725 | 0.11954 |
| Scorecard | 0.6711 | 0.3421 | 0.2480 | 0.12105 |
| sub_grade | 0.6786 | 0.3573 | 0.2619 | 0.12153 |

Chosen over XGBoost for simplicity: within 0.0021 validation AUC using 273 trees vs. 2,122. Artifacts kept at `data/processed/models/lightgbm_v1_history/`, MLflow alias `v1_champion` (`v1` is a reserved alias pattern). Its SHAP ranking, risk bands, and feature ranges were computed on its own training population; v2's serving assets are recomputed on v2's, each stamped with its model hash.
