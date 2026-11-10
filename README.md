# Credit Default Risk Engine

I built an origination-time credit scoring model for LendingClub loans: given only what a lender knows at the moment a 36-month loan is issued, predict the probability it ends in default over its lifetime.

## Why I built this

I wanted to build a credit model the way a lender actually would, not just a notebook with one accuracy number. That means a time-based train/validation/test split, a traditional scorecard as an honest baseline, an approval policy backed by expected loss math, explanations for every prediction, and a served model that gets monitored for drift after launch.

## Data

I used LendingClub's accepted-loan data, 2007 to 2018, from a public Kaggle copy uploaded by Nathan George (listed as CC0). The original data was published under LendingClub's own terms at the time. I do not redistribute the raw file; it stays local and gitignored.

- Raw file: 2,260,701 rows, 151 columns. 33 are footer/summary rows, not real loans. After removing them: 2,260,668 real loan rows.
- My modeling population: 36-month, individual-application loans only.
- Time-based split, so I never shuffle across time:

| Split | Issue years | Loans | Default rate | Use |
|---|---|---|---|---|
| Train | 2012 to 2013 | 143,892 | 12.70% | Fit models, bin variables, estimate LGD |
| Validation | 2014 | 162,570 | 13.73% | Tune, calibrate, choose the approval threshold |
| Test | 2015 | 282,787 | 14.88% | Touched once, at the end, for the final numbers |
| Replay | 2016 to 2018 | 919,053 | not used | Drift checks only, no labels read |

In the 2015 test population, 147 loans were still unresolved (72 Current, 61 Late 31-120 days, 9 In Grace Period, 5 Late 16-30 days) and are excluded from the labeled population, not counted as either outcome.

## How I defined default

Target is 1 if `loan_status` is Charged Off or Default. Target is 0 if `loan_status` is Fully Paid (a loan that pays off early cannot default afterward, so I count it as 0, not excluded). I exclude Current, In Grace Period, both Late buckets, and any "does not meet the credit policy" loan, because none of those have a resolved outcome yet.

This is lifetime default of a 36-month loan, not default within some fixed window after issuance.

## Approach

- Built an explicit allow-list for all 151 raw columns: 59 kept as model features, 12 kept as metadata only (id, dates, state, zip, grade, for filtering and stability checks, never fed to a model), 1 is the target source (`loan_status`), 79 dropped (anything recorded after origination, investor-side fields, free text, joint-applicant fields, and fields mostly missing in the training years).
- Engineered 10 features from origination-time data only: `fico_mean`, `credit_history_months`, `loan_to_income`, `revol_bal_to_income`, and 6 "never happened yet" flags for the `mths_since_*` columns, gated so a missing value before a field existed in LendingClub's schema is not read as "never happened."
- Built a scorecard: WOE-binned features (optbinning) and logistic regression on 17 final features, after filtering 63 candidates down by IV, correlation, and VIF. This is my honest bank-style baseline.
- Trained LightGBM as the main model and XGBoost as a comparison, both tuned and calibrated on validation only, both logged to MLflow with their raw data hash and git commit.
- Picked one approval cutoff on validation by maximizing total expected net value (expected interest income minus expected loss), then applied that same fixed cutoff to test, once.
- Explained every prediction with LightGBM's own exact TreeSHAP (`pred_contrib=True`), on the full test set, no sampling.
- Served the champion model through FastAPI, with reason codes, in a Docker image.
- Ran a historical replay of 2016 to 2018 with PSI drift checks, kept separate from live monitoring.
- Tested the target logic, population filters, PSI math, and the API in CI.

## Results on test (2015), each model touched test exactly once

| Model | AUC (95% CI) | Gini | KS | Brier |
|---|---|---|---|---|
| LightGBM (champion) | 0.6872 (0.6847 to 0.6898) | 0.3745 | 0.2697 | 0.11966 |
| XGBoost (comparison) | 0.6886 (0.6859 to 0.6912) | 0.3772 | 0.2725 | 0.11954 |
| Scorecard | 0.6711 (0.6686 to 0.6735) | 0.3421 | 0.2480 | 0.12105 |
| LendingClub sub_grade | 0.6786 (0.6760 to 0.6812) | 0.3573 | 0.2619 | 0.12153 |

Paired AUC differences against the LightGBM champion, 95% bootstrap CI, 1,000 resamples:

| Comparison | Mean AUC diff | 95% CI | Significant |
|---|---|---|---|
| LightGBM vs scorecard | +0.0162 | 0.0146 to 0.0178 | Yes |
| LightGBM vs XGBoost | -0.0014 | -0.0020 to -0.0008 | Yes |
| LightGBM vs sub_grade | +0.0086 | 0.0063 to 0.0109 | Yes |

XGBoost edges out LightGBM by a small but statistically significant margin. Both gradient-boosted models beat the scorecard and LendingClub's own sub_grade by a larger, clearly significant margin. LightGBM is the one I deployed, per the project's own design (main model vs. comparison model), not because it wins on raw AUC.

## Fixed approval-rate comparison (test, cutoffs set on validation)

XGBoost is not in this table: I never saved its validation-set predictions, and its frozen model file is quarantined after an unrelated incident (see Limitations), so there is no way to re-score validation with it.

| Model | Target approval rate | PD cutoff (from validation) | Actual approval rate on test | Actual default rate approved | Ex-ante expected net value | Ex-post realized net cash |
|---|---|---|---|---|---|---|
| LightGBM | 70% | 0.1677 | 69.91% | 10.38% | $218,424,849.97 | $196,767,654.56 |
| LightGBM | 80% | 0.1988 | 79.59% | 11.59% | $242,802,780.16 | $213,673,508.30 |
| LightGBM | 90% | 0.2467 | 89.47% | 12.91% | $264,179,665.94 | $228,304,417.89 |
| Scorecard | 70% | 0.1668 | 72.02% | 11.00% | $235,729,814.94 | $198,858,365.93 |
| Scorecard | 80% | 0.1939 | 81.35% | 12.08% | $259,427,311.94 | $213,747,497.58 |
| Scorecard | 90% | 0.2342 | 90.54% | 13.26% | $279,507,172.89 | $225,879,610.22 |
| sub_grade map | 70% | 0.1701 | 75.57% | 11.33% | $209,544,424.67 | $192,252,839.60 |
| sub_grade map | 80% | 0.1796 | 84.59% | 12.46% | $237,882,097.24 | $210,354,942.94 |
| sub_grade map | 90% | 0.2026 | 92.88% | 13.59% | $266,615,434.41 | $224,586,336.86 |

## Key findings

- LightGBM beats the scorecard by 0.0162 AUC and LendingClub's own sub_grade by 0.0086 AUC on test, both statistically significant. The gap between LightGBM and XGBoost is tiny (0.0014 AUC) but still significant.
- The policy I chose on validation is very permissive: it approves 99.41% of test loans (281,109 of 282,787) at a PD cutoff of 0.3998. A stricter cutoff is available (see the fixed-rate table above) if a lender wants fewer, better loans instead.
- At that permissive cutoff, my ex-ante expected net value ($276.19M) overstates the actual realized net cash ($231.56M) by $44.62M (16.16% of ex-ante). About half of this gap (49.1%) is just the model's mean predicted PD (13.56%) running a bit below the actual default rate among the loans it approved (14.70%), which is expected when a policy approves nearly everyone and therefore barely screens anyone out. Most of the rest comes from two assumptions estimated on train not holding exactly on test: realized interest income on fully paid loans ran lower than assumed (80.24% vs. 82.62% of scheduled interest, 57.0% of the remaining gap), and realized loss on charged-off loans ran higher than assumed (39.47% vs. 38.03% cash-basis loss, 32.3% of the remaining gap). Together these explain 89.3% of the remaining gap; 10.7% stays unexplained.
- This expected-net-value model has no cost of capital, no operating cost, and no time value of money in it. It is comparing raw cash in against raw cash out, nothing more.
- This is accepted-loans-only data. I only ever see loans LendingClub approved, so I cannot say anything about applicants LendingClub rejected.
- A stricter robustness check (default within 24 months of issue, from payment dates, labeling only) gives a different default rate (11.93% vs. 14.88%) but the same model ranking: XGBoost, LightGBM, sub_grade, scorecard, in that order, on both the original and the stricter label.

## Drift findings (historical replay, 2015 to 2018, not live monitoring)

I checked monthly PSI (population stability index) for 69 features over 48 months, train as the baseline. By the end of 2018, 23 of 69 features had crossed the conventional 0.10 "watch" threshold at least once, and 6 had crossed the 0.25 "alert" threshold. The worst: `verification_status` (PSI 0.52, crossed watch in January 2015, alert by June 2015), `bc_util` (0.44), `revol_util` (0.43), `percent_bc_gt_75` (0.43).

I only checked performance drift (does the model stay accurate) for 2015, since that is the only year with real labels under the project's own rules. Monthly test AUC in 2015 ranged from 0.679 to 0.695, no clear trend down. I do not claim anything about model performance in 2016 to 2018: no labels were used there, by design.

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

Every direction here matches intuition: higher FICO lowers risk, higher DTI raises it, higher loan-to-income raises it, more recent credit inquiries raise it. Where the scorecard's WOE and SHAP's direction can both be checked (15 shared numeric features), all 15 agree on direction.

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, no load balancer, on an Apple M4:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 16.29 ms | 18.07 ms | 22.13 ms |
| Without reason codes | 13.62 ms | 14.86 ms | 17.11 ms |

## Tests and coverage

233 tests pass, all against small synthetic fixtures (no real data in the repo). Core-module coverage:

| Module | Coverage |
|---|---|
| src/api/main.py, schemas.py, scoring.py, reason_codes.py | 100% |
| src/features/feature_engineering.py | 100% |
| src/monitoring/psi.py | 100% |
| src/features/build_features.py | 99% |
| src/features/allow_list.py | 84% |
| src/data/target.py | 82% |
| src/api/build_assets.py, checks.py, parity_test.py | 0% |
| src/features/eda.py, field_rollout.py, missing_by_year.py | 0% |

The 0% files are one-time scripts that need the real trained model or the real raw data to do anything meaningful. Since real data never goes in this repo, they are exercised by hand against real data, not by the automated suite.

## How to run

```bash
make setup
```

```bash
make data
```

```bash
make features
```

```bash
make train
```

```bash
make evaluate
```

```bash
make serve
```

```bash
make test
```

```bash
make drift
```

`make data` needs `accepted_2007_to_2018Q4.csv.gz` in `data/raw/` (not included, see Data above). On macOS, LightGBM needs `brew install libomp` first. The Docker image installs `libgomp1` for the same reason on Linux.

## Project structure

```
src/
  data/        load, clean, define the target and the time split
  features/    the allow-list and origination-time feature engineering
  models/      scorecard, LightGBM/XGBoost, calibration, MLflow
  evaluation/  test metrics, approval economics, benchmarks
  explain/     SHAP
  monitoring/  PSI drift and historical replay
  api/         FastAPI serving app
tests/         pytest, synthetic fixtures only
reports/
  metrics/     every number in this README traces back to a file here
  figures/     saved plots
docs/          model card, validation report, data audit
notebooks/     EDA viewing only
```

## Limitations

- Accepted loans only. I never see loans LendingClub rejected, so I cannot evaluate the model on applicants who were turned down.
- This uses origination-time data, what LendingClub recorded when the loan was issued, not necessarily the exact instant someone applied.
- US data, 2012 to 2015 only. I make no claim this generalizes to other lenders, other countries, or other years.
- No protected attributes are in this data. I did not create proxies for them. I cannot evaluate this model for fairness across protected groups, only for stability across issue quarter and state.
- Drift monitoring here is a historical replay against 2016 to 2018 data, not a live monitoring system watching a production feed.
- LGD (loss given default) is estimated from train-year charged-off loans on a cash-recovered basis. It is an approximation, and I show a sensitivity table over a range of LGD values rather than relying on one point estimate.
- An unrelated row-order sensitivity in LightGBM/XGBoost's seeded bagging (confirmed and documented in `data/processed/models/quarantine/README_note.txt`) means the original XGBoost model artifact cannot be regenerated bit for bit. The frozen LightGBM champion is unaffected and was restored and verified by SHA-256 against three independent copies.

## License and data credit

The code in this repository is released under the MIT license (see `LICENSE`). The data is not mine to redistribute: I used LendingClub's accepted-loan data, 2007 to 2018, from a public Kaggle copy uploaded by Nathan George, and `data/` is gitignored so it never enters this repository. Anyone who wants to run this project needs to get the data themselves, under LendingClub's and Kaggle's own terms.
