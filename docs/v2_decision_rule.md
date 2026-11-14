# v2 decision rule

This file is committed before any v2 evaluation is run. The git history shows this commit before the commit that reports v1-vs-v2 results, so the rule below was fixed before anyone saw how v2 performs against it.

## v2 definition

LightGBM, refit on 2012-2014 (train + validation concatenated), with:

- The frozen v1 champion's exact hyperparameters, no tuning search: `num_leaves=15, learning_rate=0.08, min_child_samples=1500, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0`.
- The frozen champion's exact tree count, 273, used directly rather than re-run through early stopping. There is no non-test, non-2016-vintage holdout left inside 2012-2014 to early-stop against without using evaluation data for a design decision.
- Monotone constraints on 8 features: `fico_mean` decreasing, `dti` increasing, `loan_to_income` increasing, `revol_util` increasing, `bc_util` increasing, `inq_last_6mths` increasing, `acc_open_past_24mths` increasing, `annual_inc` decreasing. All other features unconstrained.
- The same 69-feature set as the frozen v1 champion.

v2 is fit once, in memory, for this evaluation. It is never written to `data/processed/models/` and never registered in MLflow as a champion; only v1 (the already-frozen, already-deployed model) holds that status regardless of this rule's outcome.

## Evaluation population

2016 vintages: loans issued January 2016 through October 2016 (issue_d <= 2016-10-31), term 36 months, individual application - the same population filter used throughout this project. This window is restricted to issue <= October 2016 specifically so that every loan in it has at least 24 + 5 = 29 months of observation by this data's March 2019 snapshot (October 2016 + 29 months = March 2019), a 5-month buffer past the 24-month label horizon itself, not just the bare minimum.

## Label

The same fixed 24-month-horizon label already used elsewhere in this project for test and for the 2016-2017 replay check: 1 = charged off (loan_status Charged Off or Default) with `last_pymnt_d` within 24 months of `issue_d`; 0 = everything else that has survived past the 24-month mark without charging off; excluded (cannot be labelled) = charged off with `last_pymnt_d` more than 24 months after issue, or with `last_pymnt_d` missing.

## Metrics

- **AUC**: on the 2016-vintage population above, this label, v1's existing predictions (already-frozen model, scored fresh on this population - a new, post-hoc read) and v2's predictions (freshly fit, scored on this same population).
- **ECE**: Expected Calibration Error, 10 bins weighted by bin size, same definition already used for v1 on test (`src/evaluation/posthoc_calibration.py`), applied to both v1 and v2 on this 2016-vintage population and label.

## Adoption rule

Adopt v2 in place of v1 only if **both** hold:

1. `AUC(v2) >= AUC(v1) - 0.002`
2. `ECE(v2) <= ECE(v1) + 0.005`

On a tie (both deltas exactly at the threshold), prefer v2.

If adopted, "adoption" means recording the decision and the numbers it was made on; it does not mean v1's frozen artifacts, MLflow champion alias, or served model are changed by this file or by the evaluation that follows it. Promoting an actual v2 champion is a separate, later action, not an automatic consequence of this evaluation passing.

## Process

1. This file is committed and pushed, alone, before any v2 code is written or run.
2. v1 is scored fresh on the 2016-vintage population (post-hoc, logged as a new test-adjacent data access).
3. v2 is fit fresh, once, with no tuning, and scored on the same population.
4. AUC and ECE are computed for both under the rule above, and the pass/fail outcome is reported exactly as the rule specifies - the rule is not re-interpreted or adjusted after seeing the numbers.
