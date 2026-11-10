# Data audit: Credit Default Risk Engine

## Source

LendingClub accepted loans, 2007 to 2018, public Kaggle copy uploaded by Nathan George, listed CC0. LendingClub's own data terms applied to the original download. Not redistributed here; `data/` is gitignored.

## Raw file

- 2,260,701 rows, 151 columns, SHA-256 hashed and recorded at `reports/metrics/raw_data_profile.json`.
- 33 footer/summary rows removed (null `issue_d`, text rows like "Total amount funded in policy code..."). 2,260,668 real loan rows remain.
- `id` is unique across all 2,260,668 rows: 0 nulls, 0 duplicates.
- `term` parses to exactly two values after cleaning: 36 and 60.
- 2 rows had a negative `dti` and were set to null; 2,561 rows had `dti` over 100 and were set to null.

## Population filters

Modeling population is 36-month, individual-application loans, issued 2012 to 2015. Of 1,539,983 36-month individual-application loans across all years, 3.4% (52,413) have a last recorded payment after month 36 (the term stretches when a loan runs late), which is why `last_pymnt_d` only approximates, rather than pins down, when a charged-off loan actually stopped paying.

## Target

1 if `loan_status` is Charged Off or Default. 0 if Fully Paid. Excluded (unresolved, no target assigned): Current, In Grace Period, Late (16-30 days), Late (31-120 days), and any "does not meet the credit policy" row.

| Split | Fully Paid | Default/Charged Off | Unresolved (excluded) | Default rate |
|---|---|---|---|---|
| Train (2012-2013) | 125,611 | 18,281 | 0 | 12.70% |
| Validation (2014) | 140,255 | 22,315 | 0 | 13.73% |
| Test (2015) | 240,698 | 42,089 | 147 | 14.88% |

Test's 147 unresolved loans: 72 Current, 61 Late (31-120 days), 9 In Grace Period, 5 Late (16-30 days).

## Feature allow-list

Every one of the 151 raw columns has an explicit decision in `reports/metrics/feature_audit.csv`, an allow-list (not a block-list): a column only reaches the model if marked `keep_feature`.

| Decision | Count |
|---|---|
| keep_feature (fed to the model) | 59 |
| keep_not_feature (kept for filtering/splitting/stability checks only) | 12 |
| drop | 79 |
| target_source (`loan_status`) | 1 |

Always dropped: everything recorded after origination (`total_pymnt`, `recoveries`, `last_pymnt_d`, and similar), investor-side fields (`funded_amnt_inv`, `initial_list_status`, `policy_code`), free text (`emp_title`, `title`, `desc`, `url`), and joint/secondary-applicant fields (individual applications only). `grade`, `sub_grade`, and `int_rate` are kept as a benchmark, not as model features, since they are the lender's own pricing decision. `installment` is dropped: it is derived from `int_rate`, so it carries the same pricing-decision information. `zip_code` and `addr_state` are kept only for stability analysis, not as model features, since they are geographic proxies.

## Missingness by issue year

Columns mostly missing in the train years (2012 to 2013) are dropped, recorded in `reports/metrics/missing_by_year.csv`. The rollout timing of four `mths_since_*` fields is documented in `reports/metrics/field_rollout.csv`: `mths_since_recent_bc_dlq` and `mths_since_last_major_derog` are gated on `tot_cur_bal` being null for that row (same bureau-data refresh); `mths_since_recent_inq` and `mths_since_recent_revol_delinq` are gated on `issue_d` before April 2012. Before its gate, a missing value in these four means "not recorded yet," not "never happened," so the engineered "never happened" flag is left null there rather than set to 1.

## Engineered features

`fico_mean` (average of `fico_range_low`/`fico_range_high`), `credit_history_months` (`issue_d` minus `earliest_cr_line`, in months), `loan_to_income`, `revol_bal_to_income`, and 6 "never happened yet" flags, one per `mths_since_*` source column, gated as above. All computed from origination-time data only, by the same function used in training and in the served API.
