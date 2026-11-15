# Data audit: Credit Default Risk Engine

## Source

LendingClub accepted loans, 2007-2018, public Kaggle copy (Nathan George, CC0). Not redistributed; `data/` is gitignored.

## Raw file

| Check | Value |
|---|---|
| Rows x columns | 2,260,701 x 151 |
| Footer/summary rows removed | 33 |
| Real loan rows remaining | 2,260,668 |
| `id` uniqueness | 0 nulls, 0 duplicates across 2,260,668 rows |
| `term` values after cleaning | 36, 60 |
| `dti` cleaned | 2 negative, 2,561 over 100, both set to null |

SHA-256 hash recorded at `reports/metrics/data/raw_data_profile.json`; the cleaning counts above are from `reports/metrics/data/cleaning_summary.json`.

## Population filters

Modeling population: 36-month, individual-application loans, 2012-2015, chosen on resolved-share by issue year and term (`reports/metrics/data/status_by_year.csv`, `reports/metrics/data/maturity_audit.csv`): 36-month loans reach 100% resolved by 2014, while 60-month and later 36-month vintages have not. Of 1,539,983 36-month individual loans across all years, 3.4% (52,413) have a last payment after month 36 (`reports/metrics/data/event_timing_audit_summary.json`), which is why `last_pymnt_d` only approximates when a charged-off loan stopped paying.

## Target

1 if `loan_status` is Charged Off or Default, 0 if Fully Paid. Excluded (no target): Current, In Grace Period, both Late buckets, "does not meet the credit policy."

| Split | Fully Paid | Charged Off/Default | Unresolved (excluded) | Default rate |
|---|---|---|---|---|
| Train (2012-2013) | 125,611 | 18,281 | 0 | 12.70% |
| Validation (2014) | 140,255 | 22,315 | 0 | 13.73% |
| Test (2015) | 240,698 | 42,089 | 147 | 14.88% |

Test's 147 unresolved: 72 Current, 61 Late (31-120d), 9 In Grace Period, 5 Late (16-30d).

## Feature allow-list

Every one of 151 raw columns has an explicit decision in `reports/metrics/data/feature_audit.csv`.

| Decision | Count |
|---|---|
| keep_feature (fed to the model) | 59 |
| keep_not_feature (filtering/stability only) | 12 |
| drop | 79 |
| target_source (`loan_status`) | 1 |

| Rule | Example |
|---|---|
| Always dropped | Anything recorded after origination, investor-side fields, free text, joint-applicant fields |
| Benchmark only, not a feature | `grade`, `sub_grade`, `int_rate` |
| Dropped: derived from `int_rate` | `installment` |
| Stability check only | `zip_code`, `addr_state` |

## Missingness by issue year

Columns mostly missing in train years are dropped (`reports/metrics/data/missing_by_year.csv`). Four `mths_since_*` fields roll out mid-dataset (`reports/metrics/data/field_rollout.csv`): `mths_since_recent_bc_dlq` and `mths_since_last_major_derog` gate on `tot_cur_bal` being null; `mths_since_recent_inq` and `mths_since_recent_revol_delinq` gate on `issue_d` before April 2012. Before its gate, missing means "not recorded yet," so the engineered "never happened" flag stays null there.

## Engineered features

| Feature | Source |
|---|---|
| `fico_mean` | Average of `fico_range_low`/`fico_range_high` |
| `credit_history_months` | `issue_d` minus `earliest_cr_line`, months |
| `loan_to_income` | `loan_amnt` / `annual_inc` |
| `revol_bal_to_income` | `revol_bal` / `annual_inc` |
| 6 "never happened" flags | One per `mths_since_*` source, gated as above |

All computed from origination-time data only, by the same function used in training and in the served API.
