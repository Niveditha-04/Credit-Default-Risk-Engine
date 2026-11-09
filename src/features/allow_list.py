"""Explicit allow-list for all 151 raw columns: decision, reason, and
whether the column is known at loan origination.

This is a positive list (ALLOW), not a block-list: a column only becomes a
model feature if it is explicitly marked decision="keep_feature" here.
Everything else is excluded by construction, including any column we
simply forgot to think about (the completeness assertion below would fail
first).

Decisions:
    keep_feature       -> goes into the model feature table
    keep_not_feature   -> kept in the dataset for filtering/splitting/joins/
                          stability analysis, never fed to a model as-is
    drop                -> excluded entirely
    target_source       -> this is where the target comes from, not a feature
"""

from dataclasses import dataclass

from src.data.columns import ALL_GROUPED_COLUMNS


@dataclass(frozen=True)
class ColumnDecision:
    decision: str  # keep_feature | keep_not_feature | drop | target_source
    reason: str
    known_at_origination: bool | None  # None = not applicable (not a feature)


ALLOW_LIST: dict[str, ColumnDecision] = {}


def _add(cols: list[str], decision: str, reason: str, known_at_origination: bool | None) -> None:
    for c in cols:
        if c in ALLOW_LIST:
            raise ValueError(f"Column {c} already has a decision ({ALLOW_LIST[c]})")
        ALLOW_LIST[c] = ColumnDecision(decision, reason, known_at_origination)


# ---------------------------------------------------------------------------
# Identifiers / target source / population-filter-only columns
# ---------------------------------------------------------------------------

_add(
    ["id"],
    "keep_not_feature",
    "Row identifier, no predictive meaning (sequential key). Kept for joins and audit trails only.",
    True,
)
_add(
    ["member_id"],
    "drop",
    "Always null in every row of this public release (privacy redaction). Verified: 100% missing in all years.",
    None,
)
_add(
    ["loan_status"],
    "target_source",
    "This is the raw column the target is derived from. Not usable as a feature by definition: it is the outcome.",
    False,
)
_add(
    ["issue_d"],
    "keep_not_feature",
    "Needed to assign the time-based split and to compute credit_history_months. Excluded as a raw model feature "
    "to avoid encoding a calendar-vintage shortcut that would not generalize to new issue dates.",
    True,
)
_add(
    ["term"],
    "keep_not_feature",
    "Used to filter the modelling population to 36-month loans. Constant after filtering, so zero predictive "
    "value as a feature.",
    True,
)
_add(
    ["application_type"],
    "keep_not_feature",
    "Used to filter the modelling population to individual applications. Constant ('Individual') after "
    "filtering, so zero predictive value as a feature.",
    True,
)

# ---------------------------------------------------------------------------
# Post-origination / outcome leakage (brief's always-drop list)
# ---------------------------------------------------------------------------

_add(
    [
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "out_prncp",
        "out_prncp_inv",
        "pymnt_plan",
        "debt_settlement_flag",
    ],
    "drop",
    "Post-origination: only known once the loan is already being repaid or has stopped paying. "
    "Using these would leak the outcome into the features.",
    False,
)

_add(
    [
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
    ],
    "drop",
    "Hardship-plan fields: only exist for loans that already hit financial trouble after origination. "
    "Pure post-origination leakage.",
    False,
)

_add(
    [
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    ],
    "drop",
    "Debt-settlement fields: only exist for loans already in settlement after default. Post-origination leakage.",
    False,
)

# ---------------------------------------------------------------------------
# Benchmark-only (LendingClub's own risk pricing, not used as model input)
# ---------------------------------------------------------------------------

_add(
    ["grade", "sub_grade", "int_rate"],
    "keep_not_feature",
    "LendingClub's own risk grade and priced rate, known at origination. Kept out of the model so we can "
    "honestly benchmark our model against LendingClub's grading in Phase 5, not reuse their answer as an input.",
    True,
)

_add(
    ["installment"],
    "drop",
    "installment is a near-deterministic function of loan_amnt, int_rate and term (the monthly payment formula). "
    "It mostly re-encodes int_rate, which is itself LendingClub's own pricing decision, held out as benchmark-only. "
    "Flagged to Nivi: dropped rather than justified as independent signal.",
    True,
)

# ---------------------------------------------------------------------------
# Lender/investor-side, zero-variance, or geography (brief's explicit rules)
# ---------------------------------------------------------------------------

_add(
    ["funded_amnt_inv"],
    "drop",
    "Investor-side funding amount, not a borrower-risk signal known at the underwriting decision.",
    True,
)
_add(
    ["initial_list_status"],
    "drop",
    "Lender-side listing mechanics (whole vs fractional loan listing), not a borrower attribute.",
    True,
)
_add(
    ["policy_code"],
    "drop",
    "Constant = 1.0 in the modelling population (verified). The only other value (2) marks "
    "'does not meet the credit policy' loans, already excluded by the population filter. Zero variance.",
    True,
)
_add(
    ["disbursement_method"],
    "drop",
    "Constant = 'Cash' in the modelling population (verified). Zero variance, no predictive value.",
    True,
)
_add(
    ["funded_amnt"],
    "drop",
    "Lender's funding decision, not a borrower attribute. Nearly identical to loan_amnt (differs in only 23 of "
    "589,396 modelling-population rows, r approx 1.0 in the train correlation matrix, verified), so it adds no "
    "signal loan_amnt does not already have. Dropped rather than kept-and-later-pruned in Phase 3.",
    True,
)

_add(
    ["zip_code", "addr_state"],
    "keep_not_feature",
    "Geographic proxies: dropped from the model to avoid using location as a risk driver. Kept in the dataset "
    "only for the Phase 5 stability-by-region check (explicitly labelled stability, not fairness).",
    True,
)

# ---------------------------------------------------------------------------
# Free text
# ---------------------------------------------------------------------------

_add(
    ["emp_title", "title", "desc", "url"],
    "drop",
    "Free text, not a structured feature. Would need NLP to use and risks encoding leakage-prone or "
    "identifying information.",
    True,
)

# ---------------------------------------------------------------------------
# Joint-application and secondary-applicant columns (individual-only population)
# ---------------------------------------------------------------------------

_add(
    ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint"],
    "drop",
    "Joint-application fields. 100% missing in the modelling population because we filter to individual "
    "applications only (verified).",
    None,
)

_add(
    [
        "sec_app_fico_range_low",
        "sec_app_fico_range_high",
        "sec_app_earliest_cr_line",
        "sec_app_inq_last_6mths",
        "sec_app_mort_acc",
        "sec_app_open_acc",
        "sec_app_revol_util",
        "sec_app_open_act_il",
        "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",
        "sec_app_mths_since_last_major_derog",
    ],
    "drop",
    "Secondary-applicant fields. 100% missing in the modelling population because we filter to individual "
    "applications only (verified).",
    None,
)

# ---------------------------------------------------------------------------
# Fields not collected yet in the train years (2012-2013): 100% missing both
# years, verified against reports/metrics/missing_by_year.csv. These are
# genuinely unavailable at that time, not just rare events.
# ---------------------------------------------------------------------------

_add(
    [
        "open_acc_6m",
        "open_act_il",
        "open_il_12m",
        "open_il_24m",
        "mths_since_rcnt_il",
        "total_bal_il",
        "il_util",
        "open_rv_12m",
        "open_rv_24m",
        "max_bal_bc",
        "all_util",
        "inq_fi",
        "total_cu_tl",
        "inq_last_12m",
    ],
    "drop",
    "100% missing in both train years 2012 and 2013 (verified in missing_by_year.csv): LendingClub had not "
    "started collecting this field yet. Cannot be fit on train, so dropped.",
    False,
)

# ---------------------------------------------------------------------------
# Fields that rolled out mid-2012 (partially missing in 2012 only, ~0%
# missing from 2013 onward). Average missingness across both train years is
# well under 50%, so kept per the brief's "mostly missing" threshold -
# but documented as a known data-availability quirk, not a leakage concern
# (it is a collection-process artifact, not future information).
# ---------------------------------------------------------------------------

_add(
    [
        "mo_sin_old_il_acct",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
        "mo_sin_rcnt_tl",
        "mort_acc",
        "num_accts_ever_120_pd",
        "num_actv_bc_tl",
        "num_actv_rev_tl",
        "num_bc_sats",
        "num_bc_tl",
        "num_il_tl",
        "num_op_rev_tl",
        "num_rev_accts",
        "num_rev_tl_bal_gt_0",
        "num_sats",
        "num_tl_120dpd_2m",
        "num_tl_30dpd",
        "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m",
        "pct_tl_nvr_dlq",
        "total_rev_hi_lim",
        "avg_cur_bal",
        "tot_coll_amt",
        "tot_cur_bal",
        "tot_hi_cred_lim",
        "total_bal_ex_mort",
        "total_bc_limit",
        "total_il_high_credit_limit",
        "acc_open_past_24mths",
        "bc_open_to_buy",
        "bc_util",
        "percent_bc_gt_75",
        "mths_since_recent_bc",
    ],
    "keep_feature",
    "Rolled out partway through 2012: ~52% missing in 2012 but ~0-4% missing from 2013 onward (verified in "
    "missing_by_year.csv). Average missingness across both train years is well under 50%, so kept. Missing "
    "values are handled natively by LightGBM and binned as their own category by optbinning.",
    True,
)

# ---------------------------------------------------------------------------
# Delinquency/derogatory "months since" columns: high missingness is
# meaningful (missing = event never happened), not data unavailability.
# Brief explicitly asks for a "never happened" flag rather than a drop.
# ---------------------------------------------------------------------------

_add(
    [
        "mths_since_last_delinq",
        "mths_since_last_major_derog",
        "mths_since_last_record",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq",
        "mths_since_recent_inq",
    ],
    "keep_feature",
    "High missingness (48%-92% in train, verified in missing_by_year.csv) is meaningful, not unavailability: "
    "a missing value means that delinquency/derogatory/record/inquiry event never happened for this borrower. "
    "Kept raw (missing preserved) plus an engineered never_* flag built in build_features.py.",
    True,
)

# ---------------------------------------------------------------------------
# Core, clean, origination-time columns
# ---------------------------------------------------------------------------

_add(
    [
        "loan_amnt",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "purpose",
        "dti",
        "delinq_2yrs",
        "fico_range_low",
        "fico_range_high",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "collections_12_mths_ex_med",
        "acc_now_delinq",
        "chargeoff_within_12_mths",
        "delinq_amnt",
        "pub_rec_bankruptcies",
        "tax_liens",
        "emp_length",
    ],
    "keep_feature",
    "Core applicant/credit-bureau attribute known at the time of underwriting. Verified ~0% missing in all "
    "modelling years.",
    True,
)

_add(
    ["earliest_cr_line"],
    "keep_not_feature",
    "Used only to derive credit_history_months (issue_d minus earliest_cr_line) in build_features.py. "
    "The raw date itself is not fed to the model.",
    True,
)


def check_completeness() -> None:
    allow_set = set(ALLOW_LIST.keys())
    raw_set = set(ALL_GROUPED_COLUMNS)
    missing = raw_set - allow_set
    extra = allow_set - raw_set
    assert not missing, f"Columns with no allow-list decision: {sorted(missing)}"
    assert not extra, f"Allow-list references columns not in the raw file: {sorted(extra)}"


def banned_columns() -> set[str]:
    """Anything not explicitly marked keep_feature must never reach the model."""
    return {c for c, d in ALLOW_LIST.items() if d.decision != "keep_feature"}


def feature_columns() -> list[str]:
    return sorted(c for c, d in ALLOW_LIST.items() if d.decision == "keep_feature")


def write_feature_audit_csv(path: str = "reports/metrics/feature_audit.csv") -> None:
    import csv

    check_completeness()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "decision", "known_at_origination", "reason"])
        for col in sorted(ALLOW_LIST):
            d = ALLOW_LIST[col]
            writer.writerow([col, d.decision, d.known_at_origination, d.reason])
    print(f"Wrote {path} ({len(ALLOW_LIST)} rows)")


if __name__ == "__main__":
    check_completeness()
    print(f"All {len(ALLOW_LIST)} columns have a decision.")
    feats = feature_columns()
    print(f"{len(feats)} columns marked keep_feature:")
    for c in feats:
        print(f"  {c}")
    write_feature_audit_csv()
