# Validation report: Credit Default Risk Engine

## Methodology

- Time-based split, no shuffling across time.
- All models, features, hyperparameters, calibration method, and the approval threshold were frozen on train and validation (2012-2014) before test (2015) was read.
- Test was read once for the primary evaluation. Every read after that is post-hoc, and every read is logged (`reports/metrics/test_access_log.json`).

## Population

| Split | n | Default rate |
|---|---|---|
| Train (2012-2013) | 143,892 | 12.70% |
| Validation (2014) | 162,570 | 13.73% |
| Test (2015) | 282,787 | 14.88% |

147 test-year loans were unresolved at data-pull time and excluded from the labeled population.

### Eligibility caps by issue year

FICO floor holds at 660 every year; the DTI ceiling does not (`reports/metrics/eligibility_caps_by_year.csv`):

| Issue year | n | Min FICO | Max DTI | Share DTI > 35 |
|---|---|---|---|---|
| 2012 | 43,470 | 660.0 | 34.99 | 0.00% |
| 2013 | 100,422 | 660.0 | 34.99 | 0.00% |
| 2014 | 162,570 | 660.0 | 39.99 | 1.38% |
| 2015 | 282,934 | 660.0 | 39.99 | 3.42% |
| 2016 | 318,277 | 660.0 | 49.96 | 2.49% |
| 2017 | 296,484 | 660.0 | 39.99 | 0.98% |
| 2018 | 304,292 | 660.0 | 39.99 | 3.37% |

### Extrapolation past the training DTI cap (post-hoc)

| Model | Population | In-range gap | Above-cap gap |
|---|---|---|---|
| v1 (cap 34.99) | Test (2015) | +1.01 pp (n=273,086) | +5.07 pp (n=9,701) |
| v2 (cap 39.99) | 2016 holdout, relative to in-range | - | +0.19 pp (35-40), +1.58 pp (>40, n=1,746), neither significant |

v1 under-predicts loans above its own cap by about 5x the in-range gap. v2's absolute gaps on the 2016 holdout are negative throughout (its calibrator targets lifetime default against a 24-month label); the relative comparison above removes that bias and shows the same direction as v1, smaller and not conclusive.

## Results (test, 2015)

| Model | AUC | 95% CI | Gini | KS | Brier | Brier skill |
|---|---|---|---|---|---|---|
| LightGBM (v1) | 0.6872 | 0.6847-0.6898 | 0.3745 | 0.2697 | 0.11966 | 0.0555 |
| XGBoost | 0.6886 | 0.6859-0.6912 | 0.3772 | 0.2725 | 0.11954 | 0.0564 |
| Scorecard | 0.6711 | 0.6686-0.6735 | 0.3421 | 0.2480 | 0.12105 | 0.0445 |
| sub_grade | 0.6786 | 0.6760-0.6812 | 0.3573 | 0.2619 | 0.12153 | 0.0407 |

Paired AUC differences, 95% bootstrap CI, 1,000 resamples:

| Comparison | Mean diff | 95% CI | Significant |
|---|---|---|---|
| LightGBM vs. scorecard | +0.0162 | 0.0146-0.0178 | Yes |
| LightGBM vs. XGBoost | -0.0014 | -0.0020 to -0.0008 | Yes |
| LightGBM vs. sub_grade | +0.0086 | 0.0063-0.0109 | Yes |

## Robustness: stricter label

Default within 24 months of issue (labeling only, no new filters), 273,313 of 282,787 test loans (9,474 excluded as slow charge-offs).

| Model | AUC (24-month label) |
|---|---|
| XGBoost | 0.6984 |
| LightGBM | 0.6967 |
| sub_grade | 0.6922 |
| Scorecard | 0.6786 |

Same ranking as the primary label. Default rate under this label: 11.93% vs. 14.88%.

## Walk-forward (post-hoc)

Fold 1 reuses the champion's own train-validation numbers (in-sample for tuning, not a holdout). Fold 2 trains fresh on 2012-2014 and evaluates on test, genuinely out-of-sample (`reports/metrics/posthoc_walkforward.csv`, `src/evaluation/posthoc_validation.py`).

| Fold | Train | Evaluate | LightGBM AUC | Scorecard AUC | sub_grade AUC | Lift vs. scorecard | Lift vs. sub_grade |
|---|---|---|---|---|---|---|---|
| 1 (in-sample) | 2012-2013 | 2014 | 0.6834 | 0.6650 | 0.6596 | +0.0184 | +0.0238 |
| 2 (out-of-sample) | 2012-2014 | 2015 | 0.6962 | 0.6739 | 0.6786 | +0.0223 | +0.0175 |

Fold 2's LightGBM AUC (0.6962) beats the frozen champion's own test AUC (0.6872): one more year of data helps. Not a basis for touching the frozen champion.

## Partial performance on replay (post-hoc, exploratory)

One window, January-October 2016, has enough seasoning (29+ months by the March 2019 snapshot) for a fixed 24-month label (`reports/metrics/posthoc_replay_24m_performance.csv`). 5,929 of 265,967 loans (2.2%) excluded as unlabelable.

| Population | AUC range (24-month label) |
|---|---|
| Test (2015), same label | 0.6877-0.7009 |
| Replay 2016 window | 0.6725-0.7045 |

Replay's range fully contains test's: no large discrimination drop under this label on more recent originations. A narrow, single-window, exploratory check, not a monitoring signal.

## v2 decision rule: results (post-hoc)

Pre-registered and committed in `docs/v2_decision_rule.md` before any v2 code ran. Population: 2016 vintages, issue <= 2016-10-31, 260,038 loans, 12.70% label rate (`reports/metrics/posthoc_v2_evaluation.json`).

| Model | AUC | ECE |
|---|---|---|
| v1 | 0.6899 | 0.0080 |
| v2 | 0.6978 | 0.0079 |

Rule: adopt v2 if AUC(v2) >= AUC(v1) - 0.002 AND ECE(v2) <= ECE(v1) + 0.005. Both met (0.6978 >= 0.6879; 0.0079 <= 0.0130). **Outcome: adopt v2.** ECE here compares both models against a 24-month proxy, not either one's actual lifetime-default target, so it is a relative v1-vs-v2 check, not an absolute calibration claim.

### Label robustness check (post-hoc)

Second, "corrected" label, chosen after the result above was recorded (`reports/metrics/posthoc_v2_label_robustness.json`):

| Label | n | Label rate | v1 AUC | v1 ECE | v2 AUC | v2 ECE | Outcome |
|---|---|---|---|---|---|---|---|
| Original | 260,038 | 12.70% | 0.6899 | 0.0080 | 0.6978 | 0.0079 | Adopt v2 |
| Corrected | 265,967 | 12.55% | 0.6860 | 0.0104 | 0.6938 | 0.0104 | Adopt v2 |

Decision unchanged.

### Promotion

v2 promoted via `src.models.promote_v2 --promote` (`reports/metrics/v2_promotion_summary.json`). v1 kept at `data/processed/models/lightgbm_v1_history/`, MLflow alias `v1_champion`. Parity check 1,000/1,000 exact; `verify_champion_alias` max abs diff 0.0; Docker build and smoke test passing. Model hashes: v1 `b8437eb84d6a...`, v2 `d7dda06ef209...` (served). Full detail: `docs/model_card.md`.

### v2 on test (2015), post-hoc

v2's AUC on the full 2015 population, never computed before this check. 2015 is not a clean holdout for v2 (its calibrator was fit on it, though calibration does not change AUC/Gini/KS) (`reports/metrics/posthoc_v2_test_2015.json`).

| Metric | v2 on test (2015) |
|---|---|
| AUC (95% CI) | 0.6959 (0.6934-0.6985) |
| Gini | 0.3919 |
| KS | 0.2850 |
| AUC diff vs. v1 (95% CI) | +0.0087 (0.0079-0.0097), significant |
| AUC diff vs. sub_grade (95% CI) | +0.0173 (0.0151-0.0194), significant |

## Ablations (post-hoc)

Four ablations, champion's frozen hyperparameters, no tuning search. Validation is primary; test AUC is post-hoc context (`reports/metrics/posthoc_ablations_summary.json`).

| Model | Validation AUC | Test AUC (post-hoc) |
|---|---|---|
| Scorecard (deployed) | 0.6650 | 0.6711 |
| a. LightGBM, scorecard's 17 features | 0.6738 | 0.6787 |
| b. WOE logreg, 63 candidates, no reduction | 0.6684 | 0.6735 |
| Champion (LightGBM, 69 features) | 0.6834 | 0.6872 |
| c. Champion features + sub_grade | 0.6891 | 0.6967 |
| d. Champion features, 8 monotone constraints | 0.6833 | 0.6871 |

Ablation (b) scores above the deployed scorecard but has 15 sign-mismatched WOE coefficients out of 63: the correlation/VIF/sign-correction steps buy stability, not AUC. Ablation (d) costs no detectable AUC (-0.0001, within the 0.0007 retraining-noise spread, `reports/metrics/training_order_sensitivity.json`) and led to v2.

Income monotonicity (mean SHAP for `annual_inc`, binned): the champion's curve reverses by 0.0926 log-odds after its peak; ablation (d)'s constrained curve reverses by only 0.0041 (95.6% smaller).

## Paired monthly lift (post-hoc)

Per-month AUC on test, all 12 months (`reports/metrics/posthoc_monthly_lift.csv`):

| Comparison | Mean lift | Min | Max | Months positive |
|---|---|---|---|---|
| LightGBM minus sub_grade | +0.0082 | +0.0043 | +0.0165 | 12 / 12 |
| LightGBM minus scorecard | +0.0161 | +0.0107 | +0.0213 | 12 / 12 |

Months in one year share macro conditions: 12 of 12 positive is a consistency check, not 12 independent confirmations.

## Economics at matched funding volume (post-hoc)

Funds the top 70/80/90% of test loans by each model's own score. Return per dollar funded = realized net cash / funded amount (`reports/metrics/posthoc_matched_volume_economics.csv`).

| Model | 70% | 80% | 90% |
|---|---|---|---|
| LightGBM | 7.57% | 7.30% | 6.99% |
| XGBoost | 7.55% | 7.34% | 6.99% |
| Scorecard | 7.28% | 7.04% | 6.80% |
| sub_grade map | 6.97% | 6.88% | 6.74% |

At 4% annual cost of funds: 3.40% / 3.16% / 2.88% (LightGBM). At 8%: -0.76% / -0.97% / -1.23%. Not annualized: measured over each loan's observed life.

Paired bootstrap 95% CI on LightGBM's edge, 1,000 resamples (`reports/metrics/posthoc_matched_volume_comparison_table.csv`):

| Selection rate | Cost of funds | LightGBM minus scorecard | LightGBM minus sub_grade |
|---|---|---|---|
| 70% | 0% | +0.28 pp (0.22-0.34) | +0.59 pp (0.52-0.67) |
| 80% | 0% | +0.26 pp (0.21-0.30) | +0.42 pp (0.36-0.49) |
| 90% | 0% | +0.19 pp (0.15-0.22) | +0.25 pp (0.20-0.30) |

All 18 differences checked (3 rates x 3 cost levels) are positive and significant.

## Funding-fraction optimization (post-hoc)

LightGBM's fraction maximizing total net cash after cost, per cost of funds (`reports/metrics/posthoc_funding_optimization_optimal.csv`):

| Cost of funds | Optimal rate | Net cash after cost |
|---|---|---|
| 0% | 95% | $232.8M |
| 4% | 85% | $94.4M |
| 8% | 5% (grid floor) | -$2.0M |

At 8% cost of funds every model's best fraction still loses money: not a volume problem at this approximation.

## sub_grade alone vs. sub_grade + LightGBM (post-hoc)

Champion features plus `sub_grade` (ablation c) vs. sub_grade's own PD map (`reports/metrics/posthoc_subgrade_lightgbm_matched_volume.csv`):

| Year | Model | 70% | 90% |
|---|---|---|---|
| 2014 (in-sample) | sub_grade map | 8.86% | 8.89% |
| 2014 (in-sample) | sub_grade + LightGBM | 9.58% | 9.25% |
| 2015 (post-hoc) | sub_grade map | 6.98% | 6.73% |
| 2015 (post-hoc) | sub_grade + LightGBM | 7.49% | 6.97% |

sub_grade + LightGBM beats sub_grade alone at every rate, both years. 2016 not used: unresolved by the March 2019 snapshot.

## Calibration

Out-of-sample on validation (fit 2014-01 to 2014-09, holdout 2014-10 to 2014-12), refit on all of 2014 for deployment. Isotonic deploys only if it beats Platt by > 0.0005 holdout Brier; the champion picked Platt.

| Check | Value |
|---|---|
| Mean predicted PD, test | 13.74% |
| Actual default rate, test | 14.88% |
| Expected Calibration Error, test, 10 bins | 0.0121 |
| Largest monthly gap | 1.89 pp |
| Largest FICO-band gap | 1.90 pp |

Reliability plot: `reports/figures/posthoc_reliability_plot.png`.

### Recalibration policy (intent, not running code)

| Element | Value |
|---|---|
| Trigger | Calibration-in-the-large gap > ~2 pp on a cohort seasoned 24+ months |
| First response | Refit the calibrator only (Platt/isotonic) |
| If gap persists | Re-open full model selection, pre-registered rule |
| Exercised | No, no seasoned post-test cohort yet |

## Segments and stability (post-hoc)

LightGBM AUC and default rate by segment, test (`reports/metrics/posthoc_segment_table.csv`); also checked by issue quarter and `addr_state` (`test_stability_by_quarter.csv`, `test_stability_by_region.csv`), labeled stability not fairness since this data has no protected attributes:

| Dimension | AUC range | Default-rate range |
|---|---|---|
| FICO band | 0.6481-0.7417 | 4.77%-19.32% |
| Purpose (top 6 + other) | 0.6677-0.6945 | 12.23%-18.48% |
| Loan size band | 0.6596-0.7124 | - |
| Issue quarter | 0.6812-0.6914 | - |

The model discriminates better among safer borrowers and larger loans.

## Selection economics

Threshold set on validation and train, maximizing total expected net value. Cutoff: PD <= 0.3998 (99.5% validation selection rate), applied unchanged to test.

| Model | Selection rate | Default rate funded | Ex-ante EV | Realized cash | Gap |
|---|---|---|---|---|---|
| LightGBM | 99.41% | 14.70% | $276.2M | $231.6M | $44.6M (16.2%) |
| XGBoost | 99.12% | 14.63% | $274.9M | $232.2M | $42.7M (15.6%) |
| Scorecard | 99.82% | 14.83% | $292.6M | $230.2M | $62.4M (21.3%) |

Loss rate: cash basis, `1 - (total payment - fees) / funded amount`, 0.3803 mean on train charged-off loans. Sensitivity table: `reports/metrics/test_lgd_sensitivity.csv`. No cost of capital, operating cost, or time value of money in this EV model.

### Gap decomposition, LightGBM

| Component | Amount | % of $44.6M gap |
|---|---|---|
| Default-rate drift | $21.9M | 49.1% |
| Prepayment (interest shortfall) | $12.9M | 29.0% |
| Loss severity | $7.3M | 16.4% |
| Unexplained | $2.4M | 5.5% |

### Fixed selection-rate sweep (cutoff set on validation, applied to test)

| Model | Target | Actual rate | Realized cash |
|---|---|---|---|
| LightGBM | 70% | 69.91% | $196.8M |
| LightGBM | 80% | 79.59% | $213.7M |
| LightGBM | 90% | 89.47% | $228.3M |

A different "70%" from decision quality below (fixed PD cutoff vs. exact top-N% rank): both correct under their own definition.

### Decision quality (post-hoc, exact rank on test)

| Approval rate | Default rate approved | Defaults caught | Good declined per default avoided | Realized cash |
|---|---|---|---|---|
| 70.00% | 10.40% | 51.10% | 2.94 | $196.9M |
| 90.00% | 12.99% | 21.44% | 2.13 | $228.9M |
| 99.41% (chosen) | 14.70% | 1.79% | 1.23 | $231.6M |

See `docs/decision_memo.md` for this table read as a lending decision.

### KPI movement (post-hoc)

95% CI on each 2015 month's default rate and month-over-month change: 7 of 11 changes exceed sampling noise (`reports/metrics/kpi_movement_monthly.csv`). Symmetric mix/rate decomposition by FICO band and purpose, for each flagged change:

| Measure | Value |
|---|---|
| Mean \|mix effect\| | 0.082 pp |
| Mean \|rate effect\| | 0.769 pp |
| Mix dominant | 0 of 14 splits |

The model's own predicted PD moves the right direction every time but explains only a small share. Read as vintage effects the model's features do not capture, not a trend in applicant mix.

## Drift (historical replay, 2015-2018)

PSI monthly, bins fixed on train, conventional thresholds (<0.10 stable, 0.10-0.25 watch, >0.25 alert). 23 of 69 features crossed watch, 6 crossed alert (`reports/metrics/drift_combined_psi_table.csv`).

| Feature | Peak PSI | First crossed alert | SHAP rank (of 69) |
|---|---|---|---|
| verification_status | 0.5223 | 2015-06 | 38 |
| bc_util | 0.4377 | 2017-10 | 18 |
| revol_util | 0.4332 | 2017-09 | 13 |
| percent_bc_gt_75 | 0.4318 | 2017-09 | 14 |
| bc_open_to_buy | 0.3134 | 2017-12 | 11 |
| num_tl_120dpd_2m | 0.2880 | 2016-11 | 62 |

3 of the top-10 SHAP features reached watch: fico_mean (0.1513), purpose (0.1421), loan_to_income (0.1202). None of the top-10 reached alert. Performance drift reported for 2015 only (the only year with real labels); no performance-drift claim for 2016-2018.

## API

`/score` accepts 63 fields (`issue_d` never required, defaults to today). 19 of the remaining 62 are required (< 1% missing in train + validation).

| Field group | Count | Rule |
|---|---|---|
| Required | 19 | < 1% missing, 2012-2014 |
| Optional | 43 | >= 1% missing, several missing-not-at-random |

Example request (62 fields): `docs/api_example.json`. The response lists what it did not get (`missing_fields`) and warns when a missing field is one of the top-10 SHAP drivers (`reports/metrics/shap_global_importance_served.csv`).

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, Apple M4:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 17.25 ms | 21.61 ms | 35.14 ms |
| Without reason codes | 14.62 ms | 20.68 ms | 29.54 ms |

