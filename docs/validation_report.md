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

## Stability

Checked by issue quarter and by `addr_state` grouping, labeled stability, not fairness, since there are no protected attributes in this data to check fairness against. See `reports/metrics/test_stability_by_quarter.csv` and `test_stability_by_region.csv`.

## Calibration

Both LightGBM and XGBoost are calibrated on validation (Platt). LightGBM's champion mean predicted PD on test (13.74%) sits close to, but below, the actual test default rate (14.88%). See `reports/metrics/test_calibration_lightgbm.csv` for the full calibration curve.

## Approval economics

Threshold chosen entirely on validation (2014) and train (2012 to 2013), by maximizing total expected net value: PD x exposure x LGD against expected interest income. Chosen cutoff: PD <= 0.3998, a 99.5% validation approval rate. Applied unchanged to test:

| Model | PD cutoff | Approval rate on test | Default rate approved | Ex-ante EV | Ex-post realized cash | Gap |
|---|---|---|---|---|---|---|
| LightGBM | 0.3998 | 99.41% | 14.70% | $276,185,609.46 | $231,563,567.21 | $44,622,042.25 (16.16%) |
| XGBoost | 0.3998 | 99.12% | 14.63% | $274,940,133.99 | $232,196,032.38 | $42,744,101.61 (15.55%) |
| Scorecard | 0.3998 | 99.82% | 14.83% | $292,623,250.94 | $230,207,167.68 | $62,416,083.26 (21.33%) |

LGD used in the EV model: 0.3803 (cash basis, train charged-off loans). Sensitivity table over LGD 0.25 to 0.6 is in `reports/metrics/test_lgd_sensitivity.csv`. This EV model has no cost of capital, no operating cost, and no time value of money.

Gap decomposition for LightGBM: 49.09% of the $44.62M gap is explained by the model's mean predicted PD on approved loans (13.56%) being below their actual default rate (14.70%). Of the remaining $22.72M, 57.0% is explained by realized interest income on test's fully paid loans coming in lower than train's assumption (80.24% vs. 82.62% of scheduled interest), and 32.3% by realized loss on test's charged-off loans coming in higher than train's LGD assumption (39.47% vs. 38.03%). Combined, these explain 89.3% of the remainder; 10.7% is unexplained.

## Drift (historical replay)

PSI computed monthly, 2015 to 2018, bins fixed on train, against the conventional thresholds (<0.10 stable, 0.10 to 0.25 watch, >0.25 alert). 23 of 69 features crossed watch by end of 2018, 6 crossed alert. Performance drift is reported for 2015 only (the only year with real labels); no performance-drift claim is made for 2016 to 2018.

## Test-set access log

Every read of `test_scored.parquet` after the initial one-time scoring is logged in `reports/metrics/test_access_log.json`, with what was read and why. No test read has ever fed back into a model, feature, or threshold choice.
