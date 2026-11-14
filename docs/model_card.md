# Model card: Credit Default Risk Engine

## Model (served)

LightGBM gradient-boosted trees, 273 trees, with monotone constraints on 8 features (`fico_mean` decreasing; `dti`, `loan_to_income`, `revol_util`, `bc_util`, `inq_last_6mths`, `acc_open_past_24mths` increasing; `annual_inc` decreasing), plus a Platt calibrator fit on 2015. 69 features feed it. Trained on 2012 to 2014 (train and validation concatenated), with hyperparameters `num_leaves=15, learning_rate=0.08, min_child_samples=1500, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0` and tree count 273, the same recipe v1 (below) already used.

v2 is selected over v1 by a champion/challenger evaluation with a decision rule pre-registered in `docs/v2_decision_rule.md` before any of v2's code ran: adopt the challenger (v2) if, on 2016 vintages (issue through October 2016, 24-month default label), its AUC is no more than 0.002 below the champion's (v1) and its Expected Calibration Error (ECE) is no more than 0.005 above the champion's.

| | v1 (frozen, unchanged) | v2 |
|---|---|---|
| AUC, 2016 vintages, 24-month label | 0.6899 | 0.6978 |
| ECE, 2016 vintages, 24-month label | 0.0080 | 0.0079 |
| Rule met | - | yes, both conditions |

A second, corrected 24-month label (chosen after the rule's own result was already known, so it only checks robustness, not the decision itself) reaches the same conclusion: v1 AUC 0.6860 vs. v2 AUC 0.6938, ECE 0.0104 vs. 0.0104, rule met either way (`reports/metrics/posthoc_v2_label_robustness.json`). The ECE values above compare each model's lifetime-horizon PD against a 24-month outcome, a mismatch by construction, and v2 has no calibrator fit for a 24-month horizon at all, so the ECE comparison is read as relative (is v2 worse than v1 on the same mismatched basis), not as an absolute calibration claim.

## A logistic regression scorecard

A logistic regression scorecard on WOE-binned features (17 features) is trained alongside the gradient-boosting line as a baseline, evaluated on test (2015), not served.

## Intended use

Estimate the probability that a 36-month LendingClub loan, issued to an individual applicant, ends in lifetime default, using only information available at origination. Intended for an approval decision at origination, with an expected-loss-based threshold, plus a top-3 driver explanation per prediction.

Not intended for: applicants outside this population (joint applications, other loan terms), legal adverse-action notices (the reason codes here are illustrative only, not a compliant adverse-action letter), or any claim about fairness across protected classes, since none are in this data.

## Training data

LendingClub accepted loans, 2007 to 2018 (public Kaggle copy, Nathan George, CC0). Modeling population: 36-month, individual-application loans issued 2012 to 2015.

- Train: 2012 to 2013, 143,892 loans, 12.70% default rate.
- Validation: 2014, 162,570 loans, 13.73% default rate. This model is trained on train and validation together.
- Test: 2015, 282,787 loans, 14.88% default rate. This model's calibrator is fit on test (2015); the prior model below was evaluated on test exactly once.

## Features

59 raw features plus 10 engineered (fico_mean, credit_history_months, loan_to_income, revol_bal_to_income, and 6 "never happened yet" flags for `mths_since_*` columns) feed the LightGBM line, 69 in total. The scorecard uses a filtered, WOE-binned subset of 17. `grade`, `sub_grade`, and `int_rate` are excluded from both models (they are the lender's own pricing decision) and used only as a benchmark. Geographic fields (`zip_code`, `addr_state`) are excluded from modeling and used only for stability checks, not as model inputs.

## How I explain a prediction

The served model's own exact TreeSHAP (`booster.predict(pred_contrib=True)`), on the raw model's log-odds scale, the same call used both offline (full test set, no sampling) and in the served API. Top global drivers on test (2015): `fico_mean`, `acc_open_past_24mths`, `loan_to_income`, `dti`, `annual_inc` (`reports/metrics/shap_global_importance_served.csv`). Each served prediction returns its own top 3 risk-increasing drivers, labeled "illustrative only," not a legally compliant adverse-action notice.

## Limitations

- Accepted loans only, no rejected-applicant counterfactual.
- Origination-time data, not necessarily the exact instant of application.
- US, 2012 to 2015 only.
- No protected attributes in this data, and none were constructed as proxies. This model has not been evaluated for fairness across protected groups. A zip3-level stability check is ecological inference (group-level patterns, not individual ones) and cannot stand in for a fairness analysis.
- LGD is an estimate from train-year charged-off loans, cash-recovered basis. A sensitivity table over a range of LGD values is in `reports/metrics/test_lgd_sensitivity.csv`.
- Drift monitoring is a historical replay of 2016 to 2018, not a live production monitor.
- The 2016-vintage result above is post-hoc and exploratory, not a touched-once test evaluation the way the prior model's 2015 result was.

## Serving

FastAPI app, `GET /health` and `POST /score`, baked into a Docker image (`python:3.11-slim` plus `libgomp1`). Local-laptop latency, 1,000 sequential requests: see `reports/metrics/api_latency.json` for current numbers.

## History: the prior model

Before this model, the served champion was a LightGBM model trained on 2012 to 2013 only (no monotone constraints), evaluated on test (2015) exactly once:

| Model | AUC | 95% CI | Gini | KS | Brier |
|---|---|---|---|---|---|
| LightGBM (prior champion) | 0.6872 | 0.6847 to 0.6898 | 0.3745 | 0.2697 | 0.11966 |
| XGBoost | 0.6886 | 0.6859 to 0.6912 | 0.3772 | 0.2725 | 0.11954 |
| Scorecard | 0.6711 | 0.6686 to 0.6735 | 0.3421 | 0.2480 | 0.12105 |
| LendingClub sub_grade | 0.6786 | 0.6760 to 0.6812 | 0.3573 | 0.2619 | 0.12153 |

It was chosen over XGBoost because it was within 0.0021 validation AUC of XGBoost (the best alternative) while using 273 trees instead of 2122, a much simpler model for a negligible performance difference. Its artifacts are kept at `data/processed/models/lightgbm_v1_history/` and its MLflow registry alias is `v1_champion` (not `v1`, which MLflow reserves). Its SHAP ranking, risk bands, and feature ranges were computed on validation (2014) and train (2012-2013) respectively; the current model's serving assets are recomputed on its own training and calibration populations, each stamped with its model hash, so the two are never mixed.
