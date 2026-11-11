# Validation report: Credit Default Risk Engine

## Methodology

Time-based split, no shuffling across time: train on 2012 to 2013, tune and calibrate on 2014, read test (2015) exactly once at the end. Every modeling choice (features, hyperparameters, calibration method, approval threshold) is fixed on train and validation before test is read. The test read is logged (`reports/metrics/test_access_log.json`).

## Population

36-month, individual-application loans. Train 143,892 loans (12.70% default), validation 162,570 (13.73%), test 282,787 (14.88%). 147 test-year loans were still unresolved at data-pull time and are excluded from the labeled population.

## Results (test, 2015)

| Model | AUC | 95% CI | Gini | KS | Brier | Brier skill score |
|---|---|---|---|---|---|---|
| LightGBM | 0.6872 | 0.6847 to 0.6898 | 0.3745 | 0.2697 | 0.11966 | 0.0555 |
| XGBoost | 0.6886 | 0.6859 to 0.6912 | 0.3772 | 0.2725 | 0.11954 | 0.0564 |
| Scorecard | 0.6711 | 0.6686 to 0.6735 | 0.3421 | 0.2480 | 0.12105 | 0.0445 |
| sub_grade | 0.6786 | 0.6760 to 0.6812 | 0.3573 | 0.2619 | 0.12153 | 0.0407 |

Paired AUC differences, 95% bootstrap CI, 1,000 resamples:

| Comparison | Mean diff | 95% CI | Significant |
|---|---|---|---|
| LightGBM vs scorecard | +0.0162 | 0.0146 to 0.0178 | Yes |
| LightGBM vs XGBoost | -0.0014 | -0.0020 to -0.0008 | Yes |
| LightGBM vs sub_grade | +0.0086 | 0.0063 to 0.0109 | Yes |

## Benchmark against LendingClub's own sub_grade

Two checks, both test-set, both using only train-year information for the mapping: (a) sub_grade's ordinal rank directly as a score (AUC 0.6786), (b) a sub_grade-to-PD map learned from train years only, then applied to test for Brier/calibration (Brier 0.12153). LightGBM beats sub_grade on both.

## Robustness: stricter label

A stricter label, default within 24 months of issue (from payment dates, labeling only, no new filters), was tested as a sensitivity check, not a redefinition of the primary target. It gives a different default rate (11.93% vs. 14.88%) on 273,313 of the 282,787 test loans (9,474 excluded as slow charge-offs past the 24-month window). The model ranking by AUC is unchanged: XGBoost (0.6984), LightGBM (0.6967), sub_grade (0.6922), scorecard (0.6786), same order as the primary label.

## Ablations (post-hoc)

Four ablations, all fit with the frozen champion's own hyperparameters (num_leaves=15, learning_rate=0.08, min_child_samples=1500, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0) and no tuning search. Validation (2014) is the primary number for every ablation, matching how the champion itself was selected; test (2015) AUC is reported for context only and is explicitly post-hoc (`reports/metrics/posthoc_ablations_summary.json`, `src/models/posthoc_ablations.py`):

| Ablation | Validation AUC | vs. champion | Test AUC (post-hoc) | vs. champion (post-hoc) |
|---|---|---|---|---|
| Champion (69 features) | 0.6834 | n/a | 0.6872 | n/a |
| a. LightGBM, scorecard's 17 features only | 0.6738 | -0.0096 | 0.6787 | -0.0085 |
| b. WOE logistic regression, all 63 scorecard candidates, no correlation/VIF/sign-correction reduction | 0.6684 | -0.0150 | 0.6735 | -0.0137 |
| c. LightGBM, champion features + sub_grade | 0.6891 | +0.0057 | 0.6967 | +0.0095 |
| d. LightGBM, champion features, 8 monotone constraints (fico_mean -, dti +, loan_to_income +, revol_util +, bc_util +, inq_last_6mths +, acc_open_past_24mths +, annual_inc -) | 0.6833 | -0.0001 | 0.6871 | -0.0001 |

Ablation (b) produces 15 positive (sign-mismatched) WOE coefficients out of 63 - exactly the instability the scorecard's correlation and VIF reduction steps exist to catch.

For (d), the monotone constraints cost essentially nothing in AUC and mostly remove the income non-monotonicity flagged in the SHAP section above: binning test by income and taking the mean SHAP contribution of `annual_inc`, the champion's curve turns protective, peaks around $90K-110K, then partially reverses (swinging back up by 0.0926 log-odds after the peak); under the monotone constraint the same curve still peaks and then only drifts by 0.0041 log-odds afterward, a 95.6% reduction in that reversal (`reports/metrics/posthoc_ablations_summary.json`'s `income_monotonicity_check`). The reversal shrinks sharply rather than disappearing to exactly zero, since the constraint only forces the tree splits to be non-increasing in `annual_inc`, which allows a flat (not strictly decreasing) segment.

### Paired monthly lift (post-hoc)

Per-month AUC on test (2015), LightGBM vs. sub_grade's ordinal rank and vs. the scorecard, all 12 months (`reports/metrics/posthoc_monthly_lift.csv`, `src/evaluation/posthoc_monthly_lift.py`):

| Comparison | Mean lift | Min | Max | Months positive |
|---|---|---|---|---|
| LightGBM minus sub_grade | +0.0082 | +0.0043 | +0.0165 | 12 / 12 |
| LightGBM minus scorecard | +0.0161 | +0.0107 | +0.0213 | 12 / 12 |

### Economics at matched funding volume (post-hoc)

For each model, funds exactly the top 70/80/90% of test loans by that model's own score (an evaluation device, not a policy: the deployed policy's cutoff is fixed on validation, see Selection economics above). Realized return per dollar funded = realized net cash / funded amount, summed over the funded loans (`reports/metrics/posthoc_matched_volume_economics.csv`, `reports/figures/posthoc_return_per_dollar_vs_selection_rate.png`, `src/evaluation/posthoc_matched_volume_economics.py`):

| Model | 70% | 80% | 90% |
|---|---|---|---|
| LightGBM | 7.57% | 7.30% | 6.99% |
| XGBoost | 7.55% | 7.34% | 6.99% |
| Scorecard | 7.28% | 7.04% | 6.80% |
| sub_grade map | 6.97% | 6.89% | 6.74% |

Cost-of-funds sensitivity (same selection rates, LightGBM): cost of funds = annual rate x funded_amnt x (months from issue to last payment / 12) x 0.5, where 0.5 approximates the average outstanding balance of an amortizing loan over its observed life (starts near funded_amnt, ends near 0 if fully paid off). At 0% annual cost of funds the returns above are unchanged; at 4% they fall to 3.40% / 3.16% / 2.88%; at 8% they go negative, -0.76% / -0.97% / -1.23%, since 8% approaches the gross realized return itself over a typical holding period. `months_to_last_pymnt` is set to 0 for the handful of test loans with no recorded payment date.

## Stability

Checked by issue quarter and by `addr_state` grouping, labeled stability, not fairness, since there are no protected attributes in this data to check fairness against. See `reports/metrics/test_stability_by_quarter.csv` and `test_stability_by_region.csv`.

## Calibration

LightGBM, XGBoost, and the scorecard are each calibrated out-of-sample on a held-out slice of validation (fit on 2014-01 to 2014-09, evaluated on 2014-10 to 2014-12), then refit on all of 2014 for deployment. The champion's calibrator deploys isotonic only if it beats Platt by more than 0.0005 holdout Brier score; otherwise it deploys Platt. The live champion picked Platt. Every scoring path (`score_models.py`, `decision_policy.py`, the served MLflow model, the API) applies whichever calibrator it got through one shared function, `apply_calibrator()`, so either choice scores correctly. LightGBM's champion mean predicted PD on test (13.74%) sits close to, but below, the actual test default rate (14.88%). See `reports/metrics/test_calibration_lightgbm.csv` for the full calibration curve.

## Selection economics

Threshold chosen entirely on validation (2014) and train (2012 to 2013), by maximizing total expected net value: PD x exposure x loss rate against expected interest income. Chosen cutoff: PD <= 0.3998, a 99.5% validation selection rate. Applied unchanged to test:

| Model | PD cutoff | Selection rate on test | Default rate funded | Ex-ante EV | Ex-post realized cash | Gap |
|---|---|---|---|---|---|---|
| LightGBM | 0.3998 | 99.41% | 14.70% | $276,185,609.46 | $231,563,567.21 | $44,622,042.25 (16.16%) |
| XGBoost | 0.3998 | 99.12% | 14.63% | $274,940,133.99 | $232,196,032.38 | $42,744,101.61 (15.55%) |
| Scorecard | 0.3998 | 99.82% | 14.83% | $292,623,250.94 | $230,207,167.68 | $62,416,083.26 (21.33%) |

Loss rate used in the EV model: loss rate on funded amount, cash basis (`1 - (total payment received - collection/recovery fee) / funded amount`), 0.3803 mean on train charged-off loans. The standard LGD definition on outstanding principal at charge-off averages 0.8906 on the same loans; the cash-basis version is used in the EV model because it matches dollars actually paid, not an accounting write-off. Sensitivity table over the cash-basis loss rate, 0.25 to 0.6, is in `reports/metrics/test_lgd_sensitivity.csv`. This EV model has no cost of capital, no operating cost, and no time value of money.

Gap decomposition for LightGBM, in the order attributed (each step's dollar amount comes out of what's left after the previous step, so the prepayment/loss-severity split specifically depends on this order): $21,903,123.37 (49.1% of the $44,622,042.25 gap) is default-rate drift, the model's mean predicted PD on funded loans (13.56%) being below their actual default rate (14.70%), since calibration was fit on 2014's 13.73% default rate and test ran at 14.88%. Of the $22,718,918.88 left after that, $12,945,771.83 (57.0% of the remainder, 29.0% of the original gap) is prepayment: realized interest income on test's fully paid loans came in lower than train's assumption (80.24% vs. 82.62% of scheduled interest). $7,336,260.28 (32.3% of the remainder, 16.4% of the original gap) is loss severity: realized loss on test's charged-off loans came in higher than train's assumption (39.47% vs. 38.03%). Combined, default-rate drift, prepayment, and loss severity explain $42,185,155.48 of the $44,622,042.25 gap (94.5%); $2,436,886.77 (5.5%) is unexplained.

## Drift (historical replay)

PSI computed monthly, 2015 to 2018, bins fixed on train, against the conventional thresholds (<0.10 stable, 0.10 to 0.25 watch, >0.25 alert). 23 of 69 features crossed watch by end of 2018, 6 crossed alert. Performance drift is reported for 2015 only (the only year with real labels); no performance-drift claim is made for 2016 to 2018.

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, no load balancer, on an Apple M4:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 16.29 ms | 18.07 ms | 22.13 ms |
| Without reason codes | 13.62 ms | 14.86 ms | 17.11 ms |

## Test-set access log

Every read of `test_scored.parquet` after the initial one-time scoring is logged in `reports/metrics/test_access_log.json`, with what was read and why. No test read has ever fed back into a model, feature, or threshold choice.
