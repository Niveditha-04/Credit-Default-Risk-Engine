# Model card: Credit Default Risk Engine

## Model

LightGBM gradient-boosted trees (the champion), with a Platt calibrator on top. A logistic regression scorecard on WOE-binned features is trained alongside it as a baseline, and XGBoost as a comparison model. 69 features feed LightGBM; 17 feed the scorecard.

## Intended use

Estimate the probability that a 36-month LendingClub loan, issued to an individual applicant, ends in lifetime default, using only information available at origination. Intended for an approval decision at origination, with an expected-loss-based threshold, plus a top-3 driver explanation per prediction.

Not intended for: applicants outside this population (joint applications, other loan terms), legal adverse-action notices (the reason codes here are illustrative only, not a compliant adverse-action letter), or any claim about fairness across protected classes, since none are in this data.

## Training data

LendingClub accepted loans, 2007 to 2018 (public Kaggle copy, Nathan George, CC0). Modeling population: 36-month, individual-application loans issued 2012 to 2015.

- Train: 2012 to 2013, 143,892 loans, 12.70% default rate.
- Validation: 2014, 162,570 loans, 13.73% default rate.
- Test: 2015, 282,787 loans, 14.88% default rate, touched once.

## Features

59 raw features plus 10 engineered (fico_mean, credit_history_months, loan_to_income, revol_bal_to_income, and 6 "never happened yet" flags for `mths_since_*` columns) feed LightGBM, 69 in total. The scorecard uses a filtered, WOE-binned subset of 17. `grade`, `sub_grade`, and `int_rate` are excluded from both models (they are the lender's own pricing decision) and used only as a benchmark. Geographic fields (`zip_code`, `addr_state`) are excluded from modeling and used only for stability checks, not as model inputs.

## Performance (test, 2015, touched once)

| Model | AUC | 95% CI | Gini | KS | Brier |
|---|---|---|---|---|---|
| LightGBM (champion) | 0.6872 | 0.6847 to 0.6898 | 0.3745 | 0.2697 | 0.11966 |
| XGBoost | 0.6886 | 0.6859 to 0.6912 | 0.3772 | 0.2725 | 0.11954 |
| Scorecard | 0.6711 | 0.6686 to 0.6735 | 0.3421 | 0.2480 | 0.12105 |
| LendingClub sub_grade | 0.6786 | 0.6760 to 0.6812 | 0.3573 | 0.2619 | 0.12153 |

## How I explain a prediction

LightGBM's own exact TreeSHAP (`booster.predict(pred_contrib=True)`), on the raw model's log-odds scale, the same call used both offline (full test set, no sampling) and in the served API. Top global drivers: `fico_mean`, `annual_inc`, `acc_open_past_24mths`, `loan_to_income`, `dti`. Each served prediction returns its own top 3 risk-increasing drivers, labeled "illustrative only," not a legally compliant adverse-action notice.

## Limitations

- Accepted loans only, no rejected-applicant counterfactual.
- Origination-time data, not necessarily the exact instant of application.
- US, 2012 to 2015 only.
- No protected attributes in this data, and none were constructed as proxies. This model has not been evaluated for fairness across protected groups.
- LGD is an estimate from train-year charged-off loans, cash-recovered basis. A sensitivity table over a range of LGD values is in `reports/metrics/test_lgd_sensitivity.csv`.
- Drift monitoring is a historical replay of 2016 to 2018, not a live production monitor.

## Serving

FastAPI app, `GET /health` and `POST /score`, baked into a Docker image (`python:3.11-slim` plus `libgomp1`). Local-laptop latency, 1,000 sequential requests: p50 16.29 ms / p95 18.07 ms with reason codes, p50 13.62 ms / p95 14.86 ms without.
