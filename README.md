# Credit Default Risk Engine

[![CI](https://github.com/Niveditha-04/Credit-Default-Risk-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Niveditha-04/Credit-Default-Risk-Engine/actions/workflows/ci.yml)

I built an origination-time credit scoring model for LendingClub loans: given only what a lender knows at the moment a 36-month loan is issued, predict the probability it ends in default over its lifetime. I wanted to build it the way a lender actually would, not just a notebook with one accuracy number: a time-based split, a scorecard baseline, an expected-loss approval policy, explanations for every prediction, and a served model monitored for drift after launch.

## Results at a glance

- LightGBM (AUC 0.6872) beats LendingClub's own sub_grade (AUC 0.6786) on test, 2015: +0.0086 AUC, 95% CI 0.0063 to 0.0109, significant.
- LightGBM also beats the scorecard baseline (AUC 0.6711) by +0.0162 AUC, 95% CI 0.0146 to 0.0178, significant.
- `fico_mean` is the strongest driver on average (mean |SHAP|).
- Served through FastAPI in Docker at 16.29 ms p50 latency; 244 tests pass, 241 of them in CI.

## Data

I used LendingClub's accepted-loan data, 2007 to 2018, from a public Kaggle copy uploaded by Nathan George (listed as CC0). The original data was published under LendingClub's own terms at the time. I do not redistribute the raw file; it stays local and gitignored.

- Raw file: 2,260,701 rows, 151 columns, 33 footer/summary rows removed, 2,260,668 real loan rows left. My modeling population: 36-month, individual-application loans only. Time-based split, so I never shuffle across time:

| Split | Issue years | Loans | Default rate | Use |
|---|---|---|---|---|
| Train | 2012 to 2013 | 143,892 | 12.70% | Fit models, bin variables, estimate LGD |
| Validation | 2014 | 162,570 | 13.73% | Tune, calibrate, choose the approval threshold |
| Test | 2015 | 282,787 | 14.88% | Touched once, at the end, for the final numbers |
| Replay | 2016 to 2018 | 919,053 | not used | Drift checks only, no labels read |

In the 2015 test population, 147 loans were still unresolved (72 Current, 61 Late 31-120 days, 9 In Grace Period, 5 Late 16-30 days) and are excluded from the labeled population, not counted as either outcome.

## How I defined default

Target is 1 if `loan_status` is Charged Off or Default. Target is 0 if `loan_status` is Fully Paid (a loan that pays off early cannot default afterward, so I count it as 0, not excluded). I exclude Current, In Grace Period, both Late buckets, and any "does not meet the credit policy" loan, because none of those have a resolved outcome yet. This is lifetime default of a 36-month loan, not default within some fixed window after issuance.

## Approach

- Built an explicit allow-list for all 151 raw columns: 59 kept as model features, 12 kept as metadata only (dates, state, zip, grade, for filtering and stability checks, never fed to a model), 1 is the target source (`loan_status`), 79 dropped.
- Engineered 10 features from origination-time data only: `fico_mean`, `credit_history_months`, `loan_to_income`, `revol_bal_to_income`, and 6 "never happened yet" flags for the `mths_since_*` columns.
- Built a WOE-binned logistic scorecard as an honest baseline, then trained LightGBM as the main model and XGBoost as a comparison, both tuned and calibrated on validation only and logged to MLflow.
- Picked one approval cutoff on validation by maximizing total expected net value, then applied that same fixed cutoff to test, once.
- Explained every prediction with LightGBM's own exact TreeSHAP on the full test set, and served the champion through FastAPI with reason codes, in a Docker image.
- Checked drift over 2016 to 2018 with PSI, a historical replay not live monitoring, and tested the target logic, population filters, PSI math, and the API in CI.

## Results on test (2015), each model touched test exactly once

| Model | AUC (95% CI) | Gini | KS | Brier | AUC diff vs. LightGBM (95% CI) |
|---|---|---|---|---|---|
| LightGBM (champion) | 0.6872 (0.6847 to 0.6898) | 0.3745 | 0.2697 | 0.11966 | n/a |
| XGBoost (comparison) | 0.6886 (0.6859 to 0.6912) | 0.3772 | 0.2725 | 0.11954 | -0.0014 (-0.0020 to -0.0008) |
| Scorecard | 0.6711 (0.6686 to 0.6735) | 0.3421 | 0.2480 | 0.12105 | +0.0162 (0.0146 to 0.0178) |
| LendingClub sub_grade | 0.6786 (0.6760 to 0.6812) | 0.3573 | 0.2619 | 0.12153 | +0.0086 (0.0063 to 0.0109) |

All three differences are statistically significant (95% bootstrap CI, 1,000 resamples). XGBoost edges out LightGBM by a small but significant margin; both gradient-boosted models beat the scorecard and sub_grade by a larger margin. I deployed LightGBM anyway: it is within 0.0021 AUC of XGBoost on validation, using 273 trees instead of 2,122, and retraining noise alone moves validation AUC by about 0.0007 (`reports/metrics/training_order_sensitivity.json`), so the two are effectively tied.

## Fixed approval-rate comparison (test, cutoffs set on validation, $ in millions)

| Model | Target approval rate | Actual approval rate on test | Actual default rate approved | Ex-ante expected net value | Ex-post realized net cash |
|---|---|---|---|---|---|
| LightGBM | 70% | 69.91% | 10.38% | $218.4M | $196.8M |
| LightGBM | 80% | 79.59% | 11.59% | $242.8M | $213.7M |
| LightGBM | 90% | 89.47% | 12.91% | $264.2M | $228.3M |
| Scorecard | 70% | 72.02% | 11.00% | $235.7M | $198.9M |
| Scorecard | 80% | 81.35% | 12.08% | $259.4M | $213.7M |
| Scorecard | 90% | 90.54% | 13.26% | $279.5M | $225.9M |
| sub_grade map | 70% | 75.57% | 11.33% | $209.5M | $192.3M |
| sub_grade map | 80% | 84.59% | 12.46% | $237.9M | $210.4M |
| sub_grade map | 90% | 92.88% | 13.59% | $266.6M | $224.6M |

Only XGBoost's test predictions were saved, so it appears in the main results but not in this table. sub_grade can't hit a target rate exactly either: it has only 35 discrete levels, so its actual rate on test misses the target by a point or more.

## Key findings

- My ex-ante expected net value ($276.2M) overstates the actual realized net cash ($231.6M) by $44.6M (16.2% of ex-ante).
- About half of that gap (49.1%) is calibration timing, not over-approval: fit on 2014 (13.73% default rate), but test (2015) ran at 14.88%, so predicted PD on approved loans (13.56%) sits below their actual rate (14.70%). The rest is two train-year assumptions missing on test: interest income ran lower (80.24% vs. 82.62% of scheduled interest, 57.0% of the remainder) and loss ran higher (39.47% vs. 38.03%, 32.3%); together 89.3% explained, 10.7% unexplained.
- A stricter policy means less total cash, not more: LightGBM realizes $196.8M at 70% approval and $228.3M at 90%, against $231.6M for the chosen policy (99.41%), since riskier loans also carry higher interest and approving more of them raises total cash even as more default. No cost of capital, operating cost, or time value of money here, and accepted-loans-only data, so I cannot say what a rejected applicant would have done.
- A stricter robustness check (default within 24 months of issue, from payment dates, labeling only) gives a different default rate (11.93% vs. 14.88%) but the same model ranking: XGBoost, LightGBM, sub_grade, scorecard, in that order, on both labels.

## Drift findings (historical replay, 2015 to 2018, not live monitoring)

I checked monthly PSI for 69 features over 48 months, train as the baseline. By end of 2018, 23 of 69 crossed the 0.10 "watch" threshold at least once, 6 crossed the 0.25 "alert" threshold: `verification_status` (0.52), `bc_util` (0.44), `revol_util` (0.43), `percent_bc_gt_75` (0.43). None of these 6 are top-10 SHAP drivers; they rank 11 to 62 of 69 by SHAP importance. Prediction drift, the model's own score distribution vs. train, stayed low: max monthly PSI 0.0549, never crossing 0.10. I only checked performance drift for 2015, since 2015 is the last year with labels: monthly test AUC ranged from 0.679 to 0.695, no clear trend down, and I make no performance-drift claim for 2016 to 2018.

## What drives the model (SHAP, full test set, no sampling)

LightGBM's own exact TreeSHAP (`pred_contrib=True`), additivity checked to within 1.24e-14 of the raw model output.

| Rank | Feature | Mean \|SHAP\| (log-odds) |
|---|---|---|
| 1 | fico_mean | 0.2166 |
| 2 | annual_inc | 0.1330 |
| 3 | acc_open_past_24mths | 0.1309 |
| 4 | loan_to_income | 0.1306 |
| 5 | dti | 0.0994 |
| 6 | purpose | 0.0903 |
| 7 | num_tl_op_past_12m | 0.0771 |
| 8 | mths_since_recent_inq | 0.0724 |
| 9 | mths_since_recent_bc | 0.0702 |
| 10 | home_ownership | 0.0690 |

Every direction matches intuition: higher FICO lowers risk, higher DTI and loan-to-income raise it. `annual_inc` alone is most protective around $90K-110K (mean SHAP -0.24), then weaker above, settling near -0.14 to -0.15 from $150K up. Grouped with the other two income-scaled features (`loan_to_income`, `revol_bal_to_income`), the combined effect stays in a -0.22 to -0.30 band above $90K rather than trending back toward zero like `annual_inc` alone (`reports/metrics/shap_income_group_reversal_check.csv`). All 15 shared numeric features agree in direction between SHAP and the scorecard's WOE.

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, no load balancer, on an Apple M4:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 16.29 ms | 18.07 ms | 22.13 ms |
| Without reason codes | 13.62 ms | 14.86 ms | 17.11 ms |

## Tests and coverage

244 tests total: 241 run in CI against small synthetic fixtures, 3 need the real trained model and run locally only. Coverage, for the core library modules (`src/features`, `src/data/target.py`, `src/monitoring/psi.py`, `src/api`):

| File | Coverage |
|---|---|
| src/api (main, schemas, scoring, reason_codes); src/features/feature_engineering.py; src/monitoring/psi.py | 100% |
| src/features/build_features.py | 98.51% |
| src/features/allow_list.py | 83.93% |
| src/data/target.py | 81.82% |

The rest of those four modules are one-time pipeline scripts I run by hand against real data, not pytest tests, so they show 0% here.

## How to run

```bash
make setup                                    # Python 3.11 venv, pinned deps
make data                                     # raw csv.gz -> cleaned, typed parquet
make features                                 # allow-list + origination-time features
make train                                    # scorecard + LightGBM/XGBoost, as a candidate only
.venv/bin/python -m src.models.gbm --promote  # promote the candidate to the frozen path
.venv/bin/python -m src.models.calibrate_comparison_models
.venv/bin/python -m src.models.freeze_model   # write frozen_model_config.json
make evaluate                                 # test metrics, benchmark, economics
make serve                                    # FastAPI dev server
make test                                     # pytest, synthetic fixtures only
make drift                                    # historical PSI replay, 2015 to 2018
docker build -t credit-default-risk-engine:latest .
docker run -d --rm -p 8000:8000 --name cdre credit-default-risk-engine:latest
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "loan_amnt": 15000.0, "annual_inc": 65000.0, "dti": 18.5, "fico_range_low": 680.0, "fico_range_high": 684.0, "earliest_cr_line": "2005-03-01",
  "home_ownership": "RENT", "purpose": "debt_consolidation", "verification_status": "Verified", "emp_length": 5.0, "revol_util": 45.0, "revol_bal": 8000.0,
  "open_acc": 10.0, "total_acc": 20.0, "pub_rec": 0.0, "pub_rec_bankruptcies": 0.0, "inq_last_6mths": 1.0, "mort_acc": 1.0, "acc_open_past_24mths": 3.0
}'
docker stop cdre
```

`make data` needs `accepted_2007_to_2018Q4.csv.gz` in `data/raw/` (not included, see Data above). On macOS, LightGBM needs `brew install libomp` first; the Docker image installs `libgomp1` on Linux. See `docs/validation_report.md` for the calibrator selection rule.

## Project structure

```
src/
  data/        load, clean, define the target and the time split
  features/    allow-list and origination-time feature engineering
  models/      scorecard, LightGBM/XGBoost, calibration, MLflow
  evaluation/  test metrics, approval economics, benchmarks
  explain/     SHAP
  monitoring/  PSI drift and historical replay
  api/         FastAPI serving app
tests/         pytest, synthetic fixtures only
reports/       metrics/ (every number here traces to a file), figures/ (saved plots)
docs/          model card, validation report, data audit
notebooks/     EDA viewing only
```

## Limitations

- Accepted loans only. I never see loans LendingClub rejected, so I cannot evaluate the model on applicants who were turned down.
- This uses origination-time data, what LendingClub recorded when the loan was issued, not necessarily the exact instant someone applied.
- US data, 2012 to 2015 only. I make no claim this generalizes to other lenders, other countries, or other years.
- No protected attributes are in this data, and I did not create proxies for them. I cannot evaluate this model for fairness, only for stability across issue quarter and state.
- Drift monitoring here is a historical replay against 2016 to 2018 data, not a live monitoring system watching a production feed.
- LGD is estimated from train-year charged-off loans on a cash-recovered basis, an approximation; a sensitivity table over a range of LGD values is in `reports/metrics/test_lgd_sensitivity.csv`.
- Training uses seeded bagging, which depends on row order. Retraining reproduces the champion to within about 0.001 validation AUC, not bit for bit (`reports/metrics/training_order_sensitivity.json`).

## License and data credit

The code in this repository is released under the MIT license (see `LICENSE`). The data is not mine to redistribute: I used LendingClub's accepted-loan data, 2007 to 2018, from a public Kaggle copy uploaded by Nathan George, and `data/` is gitignored so it never enters this repository. Anyone who wants to run this project needs to get the data themselves, under LendingClub's and Kaggle's own terms.
