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

## Walk-forward (post-hoc)

Two folds, both with the champion's frozen hyperparameters (no tuning search); fold 1 reuses the already-committed train (2012-2013) -> validation (2014) numbers, fold 2 trains fresh on 2012-2014 and evaluates on test (2015), post-hoc (`reports/metrics/posthoc_walkforward.csv`, `src/evaluation/posthoc_walkforward.py`). LightGBM reuses the champion's exact tree count (273) in fold 2 rather than early-stopping against anything inside the fold, since there is no non-test holdout left to early-stop against without using test for a design decision. The scorecard reruns its full pipeline (WOE binning, IV filter, correlation reduction, VIF reduction, sign-correction logistic regression) fresh on each fold's own train data; fold 2 selects a slightly different 16-feature set than the frozen scorecard's 17: it drops `mo_sin_old_il_acct` and `num_rev_tl_bal_gt_0` and adds `mort_acc` (`reports/metrics/posthoc_walkforward_summary.json` has the exact list).

| Fold | Train years | Evaluate | LightGBM AUC | Scorecard AUC | sub_grade AUC | Lift vs. scorecard | Lift vs. sub_grade |
|---|---|---|---|---|---|---|---|
| 1 | 2012-2013 | 2014 (primary) | 0.6834 | 0.6650 | 0.6596 | +0.0184 | +0.0238 |
| 2 | 2012-2014 | 2015 (post-hoc) | 0.6962 | 0.6739 | 0.6786 | +0.0223 | +0.0175 |

Lift vs. scorecard: mean +0.0204, range +0.0184 to +0.0223. Lift vs. sub_grade: mean +0.0207, range +0.0175 to +0.0238. Both lifts are positive in both folds - the advantage holds walking forward a year with a year more training data, not just within the single frozen validation/test split. Fold 2's LightGBM AUC (0.6962) is higher than the frozen champion's own test AUC (0.6872); with a year more training data and the champion's exact hyperparameters and tree count carried over unchanged, this is not surprising, and it is not a basis for touching the frozen champion.

## Partial performance on replay (post-hoc, exploratory)

Replay (2016-2018) carries no official target because most of it has not had time to resolve by this data's March 2019 snapshot (see Limitations). One narrow exception: loans issued January 2016 through March 2017 have all had at least 24 months to season by March 2019, so a fixed 24-month-horizon default label (same idea as the stricter robustness check above, re-derived for this window) is computable for them (`reports/metrics/posthoc_replay_24m_performance.csv`, `src/evaluation/posthoc_replay_performance.py`).

Label: 1 = charged off with `last_pymnt_d` within 24 months of `issue_d`; 0 = everything else that has been observed past the 24-month mark without charging off (including loans still Current or Late at the snapshot date - the question is only "did it charge off within 24 months", not "is it fully resolved good"). Excluded entirely (cannot be labelled): charged-off loans with `last_pymnt_d` more than 24 months after issue, or missing altogether - 6,315 of 388,055 loans in this window (1.6%). `last_pymnt_d` is only an approximation of when a loan stopped performing; LendingClub does not record an exact default date.

Applying the identical label to test (2015) by quarter, for comparison: AUC ranges 0.6877 to 0.7009 across the 4 quarters of 2015. On the replay window (2016-Q1 through 2017-Q1, 5 quarters): AUC ranges 0.6747 to 0.7045. The two ranges overlap; nothing here suggests the model's discrimination under this stricter label has degraded on more recent originations, though this remains a narrow, single-window, exploratory check, not a repeatable monitoring signal.

## Ablations (post-hoc)

Four ablations, all fit with the frozen champion's own hyperparameters (num_leaves=15, learning_rate=0.08, min_child_samples=1500, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0) and no tuning search. Validation (2014) is the primary number for every ablation, matching how the champion itself was selected; test (2015) AUC is reported for context only and is explicitly post-hoc (`reports/metrics/posthoc_ablations_summary.json`, `src/models/posthoc_ablations.py`):

Absolute validation AUC for every ablation, with the deployed scorecard and the champion as reference rows, ordered so each row builds on the one above it:

| Model | Validation AUC | Test AUC (post-hoc) |
|---|---|---|
| Scorecard (deployed: 17 features, WOE, correlation/VIF/sign-correction reduction) | 0.6650 | 0.6711 |
| a. LightGBM on the scorecard's same 17 (reduced) features | 0.6738 | 0.6787 |
| b. WOE logistic regression on all 63 scorecard candidates, no correlation/VIF/sign-correction reduction | 0.6684 | 0.6735 |
| Champion (LightGBM, 69 features) | 0.6834 | 0.6872 |
| c. LightGBM, champion features + sub_grade | 0.6891 | 0.6967 |
| d. LightGBM, champion features, 8 monotone constraints (fico_mean -, dti +, loan_to_income +, revol_util +, bc_util +, inq_last_6mths +, acc_open_past_24mths +, annual_inc -) | 0.6833 | 0.6871 |

Ablation (b)'s own validation AUC (0.6684) is slightly above the deployed scorecard's (0.6650), not below it - skipping the correlation/VIF/sign-correction reduction does not cost ranking power on its own. What it produces instead is 15 positive (sign-mismatched) WOE coefficients out of 63, exactly the instability those reduction steps exist to catch; a scorecard with 15 business-sense-violating coefficients is not one I would ship, regardless of its AUC.

A candidate for the next model-selection round: ablation (d)'s monotone-constrained LightGBM costs about -0.0001 validation AUC relative to the champion and removes most of the income non-monotonicity (see below). It is not swapped in as the champion here, since test (2015) has already been read for the frozen champion, and choosing a replacement with test numbers in hand would not be a legitimate model-selection decision - that choice belongs to a new round that freezes a model before test is touched again.

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
| sub_grade map | 6.97% | 6.88% | 6.74% |

Cost-of-funds sensitivity (same selection rates, LightGBM): cost of funds = annual rate x funded_amnt x (months from issue to last payment / 12) x 0.5, where 0.5 approximates the average outstanding balance of an amortizing loan over its observed life (starts near funded_amnt, ends near 0 if fully paid off). At 0% annual cost of funds the returns above are unchanged; at 4% they fall to 3.40% / 3.16% / 2.88%; at 8% they go negative, -0.76% / -0.97% / -1.23%, since 8% approaches the gross realized return itself over a typical holding period. `months_to_last_pymnt` is set to 0 for the handful of test loans with no recorded payment date.

Return per dollar funded is measured over the loan's observed life (issue to last payment on record), not annualized - a 36-month loan's realized return per dollar is not directly comparable to an annual interest rate without dividing by its holding period.

#### LightGBM vs. scorecard vs. sub_grade map, with paired bootstrap 95% CI on each difference (1,000 resamples, seeded)

Each row resamples loans with replacement, then recomputes every model's "fund the top N%" return on that SAME resample, so the CI reflects the two models' outcomes on identical resampled loans, not two independently resampled populations (`reports/metrics/posthoc_matched_volume_comparison_table.csv`, `reports/metrics/posthoc_matched_volume_bootstrap.csv`).

| Selection rate | Cost of funds | LightGBM | Scorecard | sub_grade map | LightGBM minus scorecard (95% CI) | LightGBM minus sub_grade (95% CI) |
|---|---|---|---|---|---|---|
| 70% | 0% | 7.57% | 7.28% | 6.97% | +0.28 pp (0.22 to 0.34), significant | +0.59 pp (0.52 to 0.67), significant |
| 70% | 4% | 3.40% | 3.13% | 2.74% | +0.27 pp (0.21 to 0.33), significant | +0.66 pp (0.59 to 0.73), significant |
| 70% | 8% | -0.76% | -1.01% | -1.49% | +0.26 pp (0.20 to 0.31), significant | +0.73 pp (0.66 to 0.80), significant |
| 80% | 0% | 7.30% | 7.04% | 6.88% | +0.26 pp (0.21 to 0.30), significant | +0.42 pp (0.36 to 0.49), significant |
| 80% | 4% | 3.16% | 2.92% | 2.70% | +0.24 pp (0.19 to 0.29), significant | +0.47 pp (0.41 to 0.53), significant |
| 80% | 8% | -0.97% | -1.20% | -1.49% | +0.23 pp (0.18 to 0.27), significant | +0.51 pp (0.46 to 0.58), significant |
| 90% | 0% | 6.99% | 6.80% | 6.74% | +0.19 pp (0.15 to 0.22), significant | +0.25 pp (0.20 to 0.30), significant |
| 90% | 4% | 2.88% | 2.70% | 2.60% | +0.18 pp (0.14 to 0.21), significant | +0.28 pp (0.23 to 0.33), significant |
| 90% | 8% | -1.23% | -1.40% | -1.54% | +0.16 pp (0.13 to 0.20), significant | +0.31 pp (0.25 to 0.35), significant |

Every one of the 18 differences is positive and significant at the 95% level: LightGBM's advantage over both the scorecard and the sub_grade map at matched funding volume holds at every selection rate and every cost-of-funds level checked here.

## Stability

Checked by issue quarter and by `addr_state` grouping, labeled stability, not fairness, since there are no protected attributes in this data to check fairness against. See `reports/metrics/test_stability_by_quarter.csv` and `test_stability_by_region.csv`.

## Calibration

LightGBM, XGBoost, and the scorecard are each calibrated out-of-sample on a held-out slice of validation (fit on 2014-01 to 2014-09, evaluated on 2014-10 to 2014-12), then refit on all of 2014 for deployment. The champion's calibrator deploys isotonic only if it beats Platt by more than 0.0005 holdout Brier score; otherwise it deploys Platt. The live champion picked Platt. Every scoring path (`score_models.py`, `decision_policy.py`, the served MLflow model, the API) applies whichever calibrator it got through one shared function, `apply_calibrator()`, so either choice scores correctly. LightGBM's champion mean predicted PD on test (13.74%) sits close to, but below, the actual test default rate (14.88%). See `reports/metrics/test_calibration_lightgbm.csv` for the full calibration curve.

Expected Calibration Error (10 bins, weighted by bin size): **0.0121**. Reliability plot: `reports/figures/posthoc_reliability_plot.png` (`src/evaluation/posthoc_calibration.py`, post-hoc for the by-month and by-FICO-band breakdowns below; the ECE and reliability plot themselves use only the already-committed 10-bin calibration table, no new test read). Calibration holds up similarly across both cuts:

- By issue month (2015): mean predicted PD tracks the actual default rate in every month; the largest gap is 1.89 percentage points (`reports/metrics/posthoc_calibration_by_month.csv`).
- By FICO band: predicted and actual default rates track closely in every band from (660, 680] down to (760, 850]; the largest gap is 1.90 percentage points, and both predicted and actual rates fall monotonically as FICO rises (19.32% actual at the lowest band to 4.77% at the highest) (`reports/metrics/posthoc_calibration_by_fico_band.csv`).

## Segments (post-hoc)

LightGBM AUC and default rate on test, by segment (`reports/metrics/posthoc_segment_table.csv`, `src/evaluation/posthoc_segments.py`):

- **FICO band:** AUC rises from 0.6481 at (660, 680] to 0.7417 at (760, 850] - the model discriminates better among safer borrowers, where there is more room between a typical good loan and a typical bad one. Default rate falls from 19.32% to 4.77% across the same bands.
- **Purpose (top 6 by volume, rest combined):** AUC ranges 0.6677 to 0.6945, narrower than the FICO-band spread - `credit_card` is both the lowest-default (12.23%) and highest-AUC (0.6945) purpose; `all_other_purposes_combined` is the highest-default (18.48%) and one of the lower-AUC segments (0.6764).
- **Loan size band:** AUC rises from 0.6596 at (0, 5000] to 0.7124 at (25000, 35000] - similar to the FICO pattern, the model discriminates better among larger loans.
- **Issue quarter:** AUC ranges 0.6812 to 0.6914 across the 4 quarters of 2015, matching `reports/metrics/test_stability_by_quarter.csv`'s existing per-quarter numbers exactly (a cross-check, not a new finding).

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

PSI is also computed against a second, closer-in-time reference: validation (2014) vs. train, the nearest in-sample-adjacent comparison available without touching test or replay (`reports/metrics/drift_baseline_psi_validation_vs_train.csv`). Two features are already at "watch" between validation and train alone: `verification_status` (0.1678) and `mths_since_last_record` (0.1285) - both already flagged as drifters by the main 2015-2018 scan, so this is not a new finding, just an earlier confirmation of the same two features. No feature reaches "alert" against this closer reference.

The full 69-feature table, both references side by side with each feature's SHAP rank (`reports/metrics/drift_combined_psi_table.csv`, built by `combined_psi_table()` in `src/monitoring/report.py`), shows that every one of the top-10 SHAP features is "stable" against the validation-vs-train reference (max 0.0326) and none reaches "alert" against the full 2015-2018 reference either (the closest is `purpose` at 0.1421, "watch"). The 6 features that do reach "alert" against the full reference (`verification_status`, `bc_util`, `revol_util`, `percent_bc_gt_75`, `bc_open_to_buy`, `num_tl_120dpd_2m`) rank 11 to 62 of 69 by SHAP importance - see the README's drift section for that list with PSI values.

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, no load balancer, on an Apple M4:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 16.29 ms | 18.07 ms | 22.13 ms |
| Without reason codes | 13.62 ms | 14.86 ms | 17.11 ms |

## Test-set access log

Every read of `test_scored.parquet` after the initial one-time scoring is logged in `reports/metrics/test_access_log.json`, with what was read and why. No test read has ever fed back into a model, feature, or threshold choice.
