# Validation report: Credit Default Risk Engine

## Methodology

Time-based split, no shuffling across time: train on 2012 to 2013, tune and calibrate on 2014, read test (2015) exactly once at the end. Every modeling choice (features, hyperparameters, calibration method, approval threshold) is fixed on train and validation before test is read. The test read is logged (`reports/metrics/test_access_log.json`).

## Population

36-month, individual-application loans. Train 143,892 loans (12.70% default), validation 162,570 (13.73%), test 282,787 (14.88%). 147 test-year loans were still unresolved at data-pull time and are excluded from the labeled population.

### Eligibility caps by issue year (2012 to 2018)

LendingClub's own FICO floor holds at 660 in every year used here, but its DTI ceiling does not (`reports/metrics/eligibility_caps_by_year.csv`):

| Issue year | n | Min FICO | Max DTI | Share with DTI > 35 |
|---|---|---|---|---|
| 2012 | 43,470 | 660.0 | 34.99 | 0.00% |
| 2013 | 100,422 | 660.0 | 34.99 | 0.00% |
| 2014 | 162,570 | 660.0 | 39.99 | 1.38% |
| 2015 | 282,934 | 660.0 | 39.99 | 3.42% |
| 2016 | 318,277 | 660.0 | 49.96 | 2.49% |
| 2017 | 296,484 | 660.0 | 39.99 | 0.98% |
| 2018 | 304,292 | 660.0 | 39.99 | 3.37% |

So "every loan has DTI <= 34.99%" is only true for train (2012-2013). The cap rose to 39.99 by 2014 (already inside validation), briefly to 49.96 in 2016, and stayed near 39.99 for the rest of replay. `src/api/feature_ranges.json`'s own DTI max (39.99, computed on train plus validation, 2012-2014) reflects this. See the README's drift section for how this interacts with `dti` and `loan_to_income`'s own PSI.

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

**Fold 1 is in-sample for model selection, not a clean holdout**: 2014 was validation's own early-stopping target (it set the tree count, 273) and the primary number the champion itself was picked on - it is not an independent test of this configuration. Only fold 2 (train 2012-2014, evaluate test 2015) is genuinely out-of-sample for this walk-forward check, since test was never used to tune or early-stop anything here or in the frozen champion.

| Fold | Train years | Evaluate | Status | LightGBM AUC | Scorecard AUC | sub_grade AUC | Lift vs. scorecard | Lift vs. sub_grade |
|---|---|---|---|---|---|---|---|---|
| 1 | 2012-2013 | 2014 | in-sample for tuning/calibration | 0.6834 | 0.6650 | 0.6596 | +0.0184 | +0.0238 |
| 2 | 2012-2014 | 2015 | out-of-sample, post-hoc | 0.6962 | 0.6739 | 0.6786 | +0.0223 | +0.0175 |

Per-fold lifts, not a rounded mean: fold 1 (in-sample) +0.0184 vs. scorecard and +0.0238 vs. sub_grade; fold 2 (the only genuinely out-of-sample fold) +0.0223 vs. scorecard and +0.0175 vs. sub_grade. Both lifts are positive in both folds, and in particular the lift holds in fold 2, the one fold this check can actually speak to. Fold 2's LightGBM AUC (0.6962) is higher than the frozen champion's own test AUC (0.6872); with a year more training data and the champion's exact hyperparameters and tree count carried over unchanged, this is not surprising, and it is not a basis for touching the frozen champion.

## Partial performance on replay (post-hoc, exploratory)

Replay (2016-2018) carries no official target because most of it has not had time to resolve by this data's March 2019 snapshot (see Limitations). One narrow exception: loans issued January through October 2016 have all had at least 24 + 5 = 29 months to season by March 2019 (October 2016 + 29 months = March 2019 exactly), a 5-month buffer past the 24-month label horizon itself, so a fixed 24-month-horizon default label (same idea as the stricter robustness check above, re-derived for this window) is computable for them without relying on the bare minimum (`reports/metrics/posthoc_replay_24m_performance.csv`, `src/evaluation/posthoc_replay_performance.py`). Loans issued after October 2016 are excluded from this window entirely, since they would not have that buffer.

Label: 1 = charged off with `last_pymnt_d` within 24 months of `issue_d`; 0 = everything else that has been observed past the 24-month mark without charging off (including loans still Current or Late at the snapshot date - the question is only "did it charge off within 24 months", not "is it fully resolved good"). Excluded entirely (cannot be labelled): charged-off loans with `last_pymnt_d` more than 24 months after issue, or missing altogether - 5,929 of 265,967 loans in this restricted window (2.2%). `last_pymnt_d` is only an approximation of when a loan stopped performing; LendingClub does not record an exact default date.

As a censoring check, loans in this window that are still open and not charged off at the March 2019 snapshot, by issue quarter (`reports/metrics/posthoc_replay_still_open_by_vintage.csv`):

| Issue quarter | n | Current | Late (16-30d) | Late (31-120d) | In Grace Period |
|---|---|---|---|---|---|
| 2016Q1 | 94,592 | 5,413 | 82 | 523 | 177 |
| 2016Q2 | 73,666 | 22,971 | 146 | 555 | 243 |
| 2016Q3 | 73,579 | 25,387 | 143 | 786 | 287 |
| 2016Q4 (Oct only) | 24,130 | 9,255 | 50 | 260 | 101 |

A meaningful share of the later quarters is still "Current" even at 29+ months (these are legitimately counted as 0 - survived without charging off); the "Late" and "In Grace Period" buckets, which carry more risk of a late-arriving charge-off this check would miss, are small (a few hundred loans per quarter out of 70,000+).

Applying the identical label to test (2015) by quarter, for comparison: AUC ranges 0.6877 to 0.7009 across the 4 quarters of 2015. On the restricted replay window (2016-Q1 through 2016-Q4, the last partial): AUC ranges 0.6725 to 0.7045 - a wider range that fully contains test's. This is consistent with no large degradation in the model's discrimination under this stricter label on more recent originations, though this remains a narrow, single-window, exploratory check, not a repeatable monitoring signal.

## v2 decision rule: results (post-hoc, exploratory)

The adoption rule (v2 definition, evaluation population, label, and the rule itself) was pre-registered and committed in `docs/v2_decision_rule.md` before this evaluation was run or any v2 code was written - the git history shows that commit before this one. This section reports the outcome exactly as the rule specifies, computed by `src/models/posthoc_v2_evaluation.py` (`reports/metrics/posthoc_v2_evaluation.json`).

Population: 2016 vintages, issue_d <= 2016-10-31, the same restricted, censoring-safe window as the replay check above (260,038 loans, 12.70% 24-month-label rate).

| Model | AUC | ECE |
|---|---|---|
| v1 (frozen champion, scored fresh on this population) | 0.6899 | 0.0080 |
| v2 (fresh-fit: LightGBM, 2012-2014, frozen hyperparameters and tree count, 8 monotone constraints) | 0.6978 | 0.0079 |

ECE caveat: both models' predicted PDs here are compared against a 24-month outcome, but v1's PD is calibrated for the project's actual, lifetime-default target (any time in a 36-month loan's life), not a 24-month one - the two definitions do not match, so neither ECE number is a calibration-quality statement in its own right. v2 additionally has no calibrator at all at this point in the process (these are v2's raw, uncalibrated predictions) - so this ECE comparison is a relative v1-vs-v2 check only (does v2's shape look worse than v1's against the same proxy), not an absolute claim about either model's calibration.

Rule: adopt v2 only if both `AUC(v2) >= AUC(v1) - 0.002` and `ECE(v2) <= ECE(v1) + 0.005`.

- AUC condition: 0.6978 >= 0.6879 - **met** (v2 is actually higher, not just within the slack).
- ECE condition: 0.0079 <= 0.0130 - **met**, with the relative-only caveat above.
- **Outcome: adopt v2**, per the pre-registered rule.

### Label robustness check (post-hoc, exploratory - does not change the result above)

Rerun on the same 2016-vintage population with a second, "corrected" 24-month label, chosen only after the pre-registered result above was already recorded: a charge-off more than 24 months after issue now counts as 0 (survived the 24-month window before eventually failing, rather than being excluded), and a charge-off with no recorded `last_pymnt_d` at all now counts as 1 (no payment ever made reads as an immediate failure, rather than being excluded) (`reports/metrics/posthoc_v2_label_robustness.json`, `src/models/posthoc_v2_label_robustness.py`).

| Label | n scored | Label rate | v1 AUC | v1 ECE | v2 AUC | v2 ECE | Rule outcome |
|---|---|---|---|---|---|---|---|
| Original (pre-registered) | 260,038 | 12.70% | 0.6899 | 0.0080 | 0.6978 | 0.0079 | Adopt v2 |
| Corrected (robustness check) | 265,967 | 12.55% | 0.6860 | 0.0104 | 0.6938 | 0.0104 | Adopt v2 |

**The decision does not change.** Under the corrected label, v1's AUC drops a little more than v2's (0.6899 to 0.6860 vs. 0.6978 to 0.6938), so v2's margin over v1 is essentially unchanged; ECE moves to a tie (0.0104 vs. 0.0104) rather than v2 edging ahead, but the rule's ECE condition (within 0.005) is still comfortably met.

### Promotion

v2 has been promoted to the served model (`src/models/promote_v2.py`, `reports/metrics/v2_promotion_summary.json`). v1's artifacts are kept, not deleted:

1. **Calibration (calibration only, not evaluation):** v2's own Platt calibrator, fit out-of-sample within 2015 itself (fit Jan-Sep, holdout-check Oct-Dec, refit on the full year for deployment) - the same discipline already used for v1's own calibration on 2014. This is a new, explicit, logged read of test; it happens after v2 was already selected against v1 on 2016 vintages, so it is not an evaluation read. Holdout Brier: 0.11801 uncalibrated vs. 0.11779 Platt-calibrated (a small improvement; mean predicted PD moves from 13.45% to 14.51% against a 14.81% actual holdout rate).
2. **Artifact promotion:** v1's `model_chosen.joblib`, `calibrator.joblib`, and `features.json` were copied to `data/processed/models/lightgbm_v1_history/` (same read-only locking as the frozen path) before v2's artifacts were written over the frozen `data/processed/models/lightgbm/` path.
3. **Frozen config:** `reports/metrics/frozen_model_config.json`'s `lightgbm_chosen` now describes v2; the previous `lightgbm_chosen` content is preserved verbatim under a new key, `lightgbm_v1_history`. `scorecard` and `xgboost` are untouched.
4. **MLflow:** the model version that was `champion` (version 3, v1) now also carries the alias `v1_champion` (MLflow reserves alias names matching `v<number>`, so a literal `v1` alias is not allowed); v2 is registered as a new version (4) with alias `champion`.
5. **Verification, all passing against v2:** parity check 1,000/1,000 exact matches; `verify_champion_alias` max abs diff 0.0 against validation; Docker build and a `/health` + `/score` smoke test.

Model hashes (SHA-256 of `model_chosen.joblib`): v1 `b8437eb84d6a...` (history), v2 `d7dda06ef209...` (now served). The served model's `model_version` field in every `/score` response is self-describing and independently verifiable from these: `lightgbm_chosen@git:<commit>+sha256:<first 12 hex chars of the served booster's hash>`.

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

A candidate for the next model-selection round: ablation (d)'s monotone-constrained LightGBM has no detectable cost in validation AUC relative to the champion (-0.0001, well within the 0.0007 retraining-noise spread already measured from row order alone, `reports/metrics/training_order_sensitivity.json`) and removes most of the income non-monotonicity (see below). It is not swapped in as the champion here, since test (2015) has already been read for the frozen champion, and choosing a replacement with test numbers in hand would not be a legitimate model-selection decision - that choice belongs to a new round that freezes a model before test is touched again.

For (d), the monotone constraints cost essentially nothing in AUC (no detectable cost, within retraining noise) and mostly remove the income non-monotonicity flagged in the SHAP section above: binning test by income and taking the mean SHAP contribution of `annual_inc`, the champion's curve turns protective, peaks around $90K-110K, then partially reverses (swinging back up by 0.0926 log-odds after the peak); under the monotone constraint the same curve still peaks and then only drifts by 0.0041 log-odds afterward, a 95.6% reduction in that reversal (`reports/metrics/posthoc_ablations_summary.json`'s `income_monotonicity_check`). The reversal shrinks sharply rather than disappearing to exactly zero, since the constraint only forces the tree splits to be non-increasing in `annual_inc`, which allows a flat (not strictly decreasing) segment.

### Paired monthly lift (post-hoc)

Per-month AUC on test (2015), LightGBM vs. sub_grade's ordinal rank and vs. the scorecard, all 12 months (`reports/metrics/posthoc_monthly_lift.csv`, `src/evaluation/posthoc_monthly_lift.py`):

| Comparison | Mean lift | Min | Max | Months positive |
|---|---|---|---|---|
| LightGBM minus sub_grade | +0.0082 | +0.0043 | +0.0165 | 12 / 12 |
| LightGBM minus scorecard | +0.0161 | +0.0107 | +0.0213 | 12 / 12 |

Months within the same year share macro conditions (the same broad credit cycle, seasonality, and underwriting environment) - they are not 12 independent draws, so "12 of 12 positive" is a consistency check across the months this specific year happened to have, not 12 separate confirmations of the lift in a statistical sense.

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

### Funding-fraction optimization (post-hoc)

For each model and each cost of funds (0/4/8% annual), the funding fraction on 2015 test that maximizes TOTAL realized net cash after cost - a dollar amount, not a per-dollar rate - searched over a 5%-step grid (`reports/metrics/posthoc_funding_optimization_curve.csv`, `reports/metrics/posthoc_funding_optimization_optimal.csv`, `reports/figures/posthoc_net_cash_vs_funding_volume.png`, `src/evaluation/posthoc_funding_optimization.py`):

| Model | Cost of funds | Optimal selection rate | Net cash after cost | Annualized return on average capital |
|---|---|---|---|---|
| LightGBM | 0% | 95% | $232.8M | 6.60% |
| XGBoost | 0% | 95% | $232.5M | 6.60% |
| Scorecard | 0% | 100% | $229.9M | 6.24% |
| sub_grade map | 0% | 100% | $229.9M | 6.24% |
| LightGBM | 4% | 85% | $94.4M | 2.94% |
| XGBoost | 4% | 85% | $94.7M | 2.96% |
| Scorecard | 4% | 90% | $89.6M | 2.64% |
| sub_grade map | 4% | 95% | $85.5M | 2.42% |
| LightGBM | 8% | 5% | -$2.0M | -0.86% |
| XGBoost | 8% | 5% | -$2.1M | -0.89% |
| Scorecard | 8% | 5% | -$2.5M | -0.99% |
| sub_grade map | 8% | 5% | -$5.6M | -2.44% |

At 0% cost of funds, LightGBM and XGBoost's better ranking lets them profitably exclude a small risky tail (peaking at 95% funded, just above the scorecard and sub_grade map's corner solution of funding everyone); at 4% the gap widens and the optimal fraction drops further for all 4. At 8%, **every model's best achievable fraction still loses money** - the optimal selection rate collapses to the grid's lowest tested value (5%) and net cash is still negative there, so this is not a volume problem solvable by selecting more conservatively; 8% annual cost of funds makes this book unprofitable under this approximation regardless of selection volume. Annualized return on average capital = net cash after cost / (funded amount x years to last payment x 0.5) - the same average-balance approximation (0.5) already used for the cost-of-funds formula, applied here to annualize the return itself; it is an approximation, not a true duration-weighted yield, since years-to-last-payment is realized, not contractual, duration.

### sub_grade alone vs. sub_grade + LightGBM (post-hoc)

LightGBM with the champion's 69 features plus `sub_grade` as a feature (ablation c) vs. sub_grade's own train-year PD map, at matched funding volume, on validation (2014) and test (2015) (`reports/metrics/posthoc_subgrade_lightgbm_matched_volume.csv`, `src/evaluation/posthoc_subgrade_lightgbm_matched_volume.py`):

| Year | Model | 70% | 80% | 90% |
|---|---|---|---|---|
| 2014 (in-sample for this model's tree count) | sub_grade map | 8.86% | 8.89% | 8.89% |
| 2014 (in-sample for this model's tree count) | sub_grade + LightGBM | 9.58% | 9.47% | 9.25% |
| 2015 (post-hoc) | sub_grade map | 6.98% | 6.88% | 6.73% |
| 2015 (post-hoc) | sub_grade + LightGBM | 7.49% | 7.28% | 6.97% |

sub_grade + LightGBM beats sub_grade alone at every rate in both years. 2016 is not used as a cleaner, fully-resolved holdout year instead of validation: 2016 loans are not fully resolved by this data's March 2019 snapshot (see Partial performance on replay, above), so `realized_net_cash` (built from `total_pymnt`, which keeps growing until a loan resolves) would understate cash for loans still being paid down. There is no year in this data that is both unused by this model's own tree-count selection and fully resolved - validation (2014) is reused instead, with that caveat stated plainly: 2014 was this model's early-stopping target, so its economics here are in-sample for tuning, unlike test (2015), which is genuinely out-of-sample for this ablation.

## Stability

Checked by issue quarter and by `addr_state` grouping, labeled stability, not fairness, since there are no protected attributes in this data to check fairness against. See `reports/metrics/test_stability_by_quarter.csv` and `test_stability_by_region.csv`.

## Calibration

LightGBM, XGBoost, and the scorecard are each calibrated out-of-sample on a held-out slice of validation (fit on 2014-01 to 2014-09, evaluated on 2014-10 to 2014-12), then refit on all of 2014 for deployment. The champion's calibrator deploys isotonic only if it beats Platt by more than 0.0005 holdout Brier score; otherwise it deploys Platt. The live champion picked Platt. Every scoring path (`score_models.py`, `decision_policy.py`, the served MLflow model, the API) applies whichever calibrator it got through one shared function, `apply_calibrator()`, so either choice scores correctly. LightGBM's champion mean predicted PD on test (13.74%) sits close to, but below, the actual test default rate (14.88%). See `reports/metrics/test_calibration_lightgbm.csv` for the full calibration curve.

Expected Calibration Error, **on test (2015)**, 10 bins weighted by bin size: **0.0121**. Reliability plot: `reports/figures/posthoc_reliability_plot.png` (`src/evaluation/posthoc_calibration.py`, post-hoc for the by-month and by-FICO-band breakdowns below; the ECE and reliability plot themselves use only the already-committed 10-bin calibration table, no new test read). Calibration holds up similarly across both cuts, still on test (2015), not on any later data:

- By issue month (2015): mean predicted PD tracks the actual default rate in every month; the largest gap is 1.89 percentage points (`reports/metrics/posthoc_calibration_by_month.csv`).
- By FICO band: predicted and actual default rates track closely in every band from (660, 680] down to (760, 850]; the largest gap is 1.90 percentage points, and both predicted and actual rates fall monotonically as FICO rises (19.32% actual at the lowest band to 4.77% at the highest) (`reports/metrics/posthoc_calibration_by_fico_band.csv`).

### Recalibration policy

Not yet implemented as running code - a statement of intent for operating this model past test (2015).

- **Trigger:** once a cohort of funded loans has seasoned past a fixed horizon (24 months, matching the horizon used throughout this project's post-hoc replay checks) and the calibration-in-the-large gap on that seasoned cohort - mean predicted PD minus actual default rate, the same quantity reported by month and by FICO band above - exceeds a threshold (a starting point consistent with what is already seen here: roughly 2 percentage points, since the largest gap observed on test in-sample is 1.89-1.90 points; this threshold itself would need validating against more seasoned cohorts before being relied on operationally).
- **What gets refit:** first, only the calibrator (Platt or isotonic, `apply_calibrator()`) is refit on the newly-seasoned cohort - the same out-of-sample fit-then-holdout methodology already used for the champion's own calibration, not the underlying LightGBM model. If the gap persists or grows after a calibrator refit, that is a signal to re-open full model selection (new candidate models, a pre-registered decision rule committed before evaluation, as in `docs/v2_decision_rule.md`), not to keep refitting the calibrator indefinitely.
- This policy has not been exercised: there is no seasoned post-test cohort with a real (non-post-hoc, non-exploratory) label available yet to trigger it.

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

Gap decomposition for LightGBM, in the order attributed (each step's dollar amount comes out of what's left after the previous step, so the prepayment/loss-severity split specifically depends on this order): $21,903,123.37 (49.1% of the $44,622,042.25 gap) is default-rate drift, the model's mean predicted PD on funded loans (13.56%) being below their actual default rate (14.70%), since calibration was fit on 2014's 13.73% default rate and test ran at 14.88%. Of the $22,718,918.88 left after that, $12,945,771.83 (57.0% of the remainder, 29.0% of the original gap) is prepayment: realized interest income on test's fully paid loans came in lower than train's assumption (80.24% vs. 82.62% of scheduled interest). $7,336,260.28 (32.3% of the remainder, 16.4% of the original gap) is loss severity: realized loss on test's charged-off loans came in higher than train's assumption (39.47% vs. 38.03%). Combined, default-rate drift, prepayment, and loss severity explain $42,185,155.48 of the $44,622,042.25 gap (94.5%); $2,436,886.77 (5.5%) is unexplained. Taken together rather than split, prepayment ($12,945,771.83) and loss severity ($7,336,260.28) are jointly $20,282,032.11 (45.4% of the original gap) - both train-year assumptions about test's realized cash flows that did not hold, independent of the attribution order used to split them from each other above.

### Fixed selection-rate sweep (test, cutoffs set on validation, $ in millions)

Rather than the single EV-maximizing cutoff above, this sweeps 3 fixed target selection rates (70/80/90%) per model, cutoff chosen on validation and applied unchanged to test:

| Model | Target selection rate | Actual selection rate on test | Actual default rate funded | Ex-ante expected net value | Ex-post realized net cash |
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

At these validation-set cutoffs, LightGBM and the scorecard realize nearly the same cash (0 to 1.1% apart) - a different methodology from the matched-funding-volume comparison below, which ranks loans by each model's own test-set score instead of a cutoff fixed on validation, and does show a significant LightGBM edge. Only XGBoost's test predictions were saved, so it appears in the main results table but not here. sub_grade can't hit a target rate exactly either: it has only 35 discrete levels.

### Decision quality (post-hoc, descriptive only)

A second, different "70%" and "90%" from the table above: the champion's own exact-rank selection (the lowest-risk 70% or 90% of test loans by its own score, not a PD value learned on validation) (`reports/metrics/test_decision_quality.csv`):

| Approval rate (exact rank on test) | Default rate approved | % of defaults caught by declining | Good loans declined per default avoided | Realized net cash |
|---|---|---|---|---|
| 70.00% | 10.40% | 51.10% | 2.94 | $196.9M |
| 90.00% | 12.99% | 21.44% | 2.13 | $228.9M |
| 99.41% (chosen policy) | 14.70% | 1.79% | 1.23 | $231.6M |

This is not the same cutoff as the "Fixed selection-rate sweep" above (a PD value learned on validation, applied unchanged to test, landing at 69.91% actual for a 70% target), so the two tables' dollar figures ($196.8M vs. $196.9M at "70%") are each correct under their own definition, not a disagreement. See `docs/decision_memo.md` for the same table read as a lending decision.

### KPI movement (post-hoc, descriptive only)

Why the 2015 monthly default rate moved the way it did, computed by `src/evaluation/kpi_movement.py`. A 95% CI (normal approximation) on each month's default rate and on each month-over-month change: 7 of 11 changes exceed 95% sampling noise, so most moves are real, not sampling noise (`reports/metrics/kpi_movement_monthly.csv`):

| Month | n | Default rate | 95% CI |
|---|---|---|---|
| 2015-01 | 23,631 | 14.879% | 14.425 to 15.333 |
| 2015-02 | 16,023 | 14.167% | 13.627 to 14.707 |
| 2015-03 | 16,914 | 15.401% | 14.857 to 15.945 |
| 2015-04 | 23,619 | 15.538% | 15.076 to 16.000 |
| 2015-05 | 21,539 | 15.228% | 14.748 to 15.708 |
| 2015-06 | 19,064 | 15.359% | 14.847 to 15.871 |
| 2015-07 | 30,864 | 14.619% | 14.225 to 15.013 |
| 2015-08 | 23,938 | 14.245% | 13.802 to 14.688 |
| 2015-09 | 18,765 | 14.937% | 14.427 to 15.447 |
| 2015-10 | 33,150 | 14.145% | 13.770 to 14.520 |
| 2015-11 | 25,157 | 14.763% | 14.325 to 15.202 |
| 2015-12 | 30,123 | 15.576% | 15.167 to 15.986 |

For each of the 7 flagged changes, a symmetric (midpoint) mix/rate decomposition by FICO band and by purpose (`reports/metrics/kpi_movement_decomposition.csv`): across all 14 flagged-transition x dimension splits, the compositional (mix) effect is small and never the larger half (mean |mix effect| 0.082 pp vs. mean |rate effect| 0.769 pp; mix dominant in 0 of 14). A model-based split tells the same story from a different angle: the champion's own mean predicted PD moves in the same direction as the actual change in every one of the 7 flagged months, but only accounts for a small fraction of the size of the move; most of each flagged change is residual (`reports/metrics/kpi_movement_summary.json`). No single compositional driver explains these real month-to-month moves. They read as vintage effects the model's features do not capture, something to monitor through the calibration-in-the-large recalibration trigger below, not a trend in applicant mix.

## Drift (historical replay)

PSI computed monthly, 2015 to 2018, bins fixed on train, against the conventional thresholds (<0.10 stable, 0.10 to 0.25 watch, >0.25 alert). 23 of 69 features crossed watch by end of 2018, 6 crossed alert. Performance drift is reported for 2015 only (the only year with real labels); no performance-drift claim is made for 2016 to 2018.

PSI is also computed against a second, closer-in-time reference: validation (2014) vs. train, the nearest in-sample-adjacent comparison available without touching test or replay (`reports/metrics/drift_baseline_psi_validation_vs_train.csv`). Two features are already at "watch" between validation and train alone: `verification_status` (0.1678) and `mths_since_last_record` (0.1285) - both already flagged as drifters by the main 2015-2018 scan, so this is not a new finding, just an earlier confirmation of the same two features. No feature reaches "alert" against this closer reference.

The full 69-feature table, both references side by side with each feature's SHAP rank (`reports/metrics/drift_combined_psi_table.csv`, built by `combined_psi_table()` in `src/monitoring/report.py`), shows that every one of the top-10 SHAP features is "stable" against the validation-vs-train reference (max 0.0326) and none reaches "alert" against the full 2015-2018 reference either (the closest is `purpose` at 0.1421, "watch"). The 6 features that do reach "alert" against the full reference (`verification_status`, `bc_util`, `revol_util`, `percent_bc_gt_75`, `bc_open_to_buy`, `num_tl_120dpd_2m`) rank 11 to 62 of 69 by SHAP importance - see the README's drift section for that list with PSI values.

## API

`/score` accepts 63 fields: `issue_d` (never required - not something a live applicant supplies, defaults to today) plus 62 real-applicant fields. 19 of those 62 are required (missing returns 422): fields with under 1% missing in 2012-2014 (train + validation, the served model's training population), on the reasoning that a live caller omitting one is a bad request, not a natively-missing value. The other 43 stay optional - several are legitimately missing-not-at-random in training (an event, like a delinquency, that never happened). The split is computed and asserted against this same under-1% rule by `src/api/field_requirements.py` (`reports/metrics/api_field_missing_rates.csv`, `reports/metrics/api_field_requirements.json`), and did not change when the population moved from train alone to train + validation.

**Required** (< 1% missing in 2012-2014): `loan_amnt`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `earliest_cr_line`, `home_ownership`, `purpose`, `verification_status`, `acc_now_delinq`, `chargeoff_within_12_mths`, `collections_12_mths_ex_med`, `delinq_2yrs`, `delinq_amnt`, `inq_last_6mths`, `open_acc`, `pub_rec`, `pub_rec_bankruptcies`, `revol_bal`, `revol_util`, `tax_liens`, `total_acc`.

**Optional** (>= 1% missing in 2012-2014, with a clean gap between the lowest optional field and the highest required field): `emp_length`, `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mort_acc`, `mths_since_last_delinq`, `mths_since_last_major_derog`, `mths_since_last_record`, `mths_since_recent_bc`, `mths_since_recent_bc_dlq`, `mths_since_recent_inq`, `mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`, `tot_coll_amt`, `tot_cur_bal`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`, `total_rev_hi_lim`.

The response still lists what it didn't get (`missing_fields`) and warns when a missing field is one of the model's top-10 global SHAP drivers (`warnings`, `reason: missing_top10_shap_feature_lowers_fidelity`), since the served score then runs on less information than the model saw on most of its training rows. This top-10 SHAP ranking (`reports/metrics/shap_global_importance_served.csv`) is recomputed for the served model on test (2015): the ranked order shifts slightly from the prior model's ranking, but the set of 10 features is identical, so the warning logic's behavior is unchanged. A trimmed response for a request sending the 19 required fields plus `emp_length` (20 of the 62 fields):

```json
{
  "request_id": "f3a1...",
  "calibrated_pd": 0.1553,
  "risk_band": "High",
  "top_3_risk_increasing_drivers_illustrative_only": [
    {"feature": "fico_mean", "plain_english": "credit score (FICO)", "shap_value_log_odds": 0.167837},
    {"feature": "bc_open_to_buy", "plain_english": "available bankcard credit", "shap_value_log_odds": 0.101559},
    {"feature": "home_ownership", "plain_english": "home ownership status", "shap_value_log_odds": 0.056849}
  ],
  "model_name": "lightgbm_chosen",
  "model_version": "lightgbm_chosen@git:53597a48f88c+sha256:d7dda06ef209",
  "missing_fields": ["acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "..."],
  "warnings": [
    {"field": "acc_open_past_24mths", "reason": "missing_top10_shap_feature_lowers_fidelity", "value": "missing"},
    {"field": "mths_since_recent_bc", "reason": "missing_top10_shap_feature_lowers_fidelity", "value": "missing"},
    {"field": "mths_since_recent_inq", "reason": "missing_top10_shap_feature_lowers_fidelity", "value": "missing"},
    {"field": "num_tl_op_past_12m", "reason": "missing_top10_shap_feature_lowers_fidelity", "value": "missing"}
  ]
}
```

## API latency (local laptop, not a production benchmark)

1,000 sequential requests, one container, no load balancer, on an Apple M4, served model:

| Condition | p50 | p95 | p99 |
|---|---|---|---|
| With reason codes | 17.25 ms | 21.61 ms | 35.14 ms |
| Without reason codes | 14.62 ms | 20.68 ms | 29.54 ms |

Numbers in `reports/metrics/api_latency.json`. Re-measured on the current served model: the request/response shapes and the SHAP call behind reason codes are unchanged, but a fresh measurement on the running image is cheap and keeps this table honest rather than assumed.

## Test-set access log

Every read of `test_scored.parquet` after the initial one-time scoring is logged in `reports/metrics/test_access_log.json`, with what was read and why. No test read has ever fed back into a model, feature, or threshold choice.
