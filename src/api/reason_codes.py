"""Plain-English labels for every one of the 69 champion model features,
for illustrative reason codes in the API response. The first block below is
copied verbatim from the Phase 6 SHAP report's labels
(src/explain/shap_analysis.py) rather than imported from it: that module
pulls in matplotlib at the top (it draws and saves dependence plots), which
has no reason to be a serving dependency just to read a dict of strings.
The second block fills in the rest - Phase 6 only needed labels for the
handful of features that showed up as top global drivers there, but any of
the 69 can be a top-3 driver for some individual applicant.
"""

_PHASE_6_LABELS = {
    "fico_mean": "credit score (FICO)",
    "dti": "debt-to-income ratio",
    "loan_to_income": "loan amount relative to income",
    "inq_last_6mths": "number of recent credit inquiries",
    "annual_inc": "annual income",
    "revol_util": "revolving credit utilization",
    "purpose": "stated loan purpose",
    "bc_open_to_buy": "available bankcard credit",
    "acc_open_past_24mths": "accounts opened in the last 24 months",
    "mths_since_recent_bc": "months since most recent bankcard account opened",
    "mths_since_recent_inq": "months since most recent credit inquiry",
    "total_bc_limit": "total bankcard credit limit",
    "num_tl_op_past_12m": "accounts opened in the last 12 months",
    "bc_util": "bankcard utilization",
    "home_ownership": "home ownership status",
    "revol_bal_to_income": "revolving balance relative to income",
    "loan_amnt": "loan amount",
    "mo_sin_old_il_acct": "age of oldest installment account",
    "mo_sin_old_rev_tl_op": "age of oldest revolving account",
    "tot_hi_cred_lim": "total high credit limit across accounts",
    "num_rev_tl_bal_gt_0": "number of revolving accounts with a balance",
    "credit_history_months": "length of credit history (months)",
    "emp_length": "years employed",
}

_ADDITIONAL_LABELS = {
    "acc_now_delinq": "accounts currently delinquent",
    "avg_cur_bal": "average current balance across accounts",
    "chargeoff_within_12_mths": "charge-offs in the last 12 months",
    "collections_12_mths_ex_med": "collections in the last 12 months (excluding medical)",
    "delinq_2yrs": "delinquencies in the last 2 years",
    "delinq_amnt": "past-due amount on currently delinquent accounts",
    "mo_sin_rcnt_rev_tl_op": "months since most recent revolving account opened",
    "mo_sin_rcnt_tl": "months since most recent account opened",
    "mort_acc": "number of mortgage accounts",
    "mths_since_last_delinq": "months since last delinquency",
    "mths_since_last_major_derog": "months since last major derogatory mark",
    "mths_since_last_record": "months since last public record",
    "mths_since_recent_bc_dlq": "months since most recent bankcard delinquency",
    "mths_since_recent_revol_delinq": "months since most recent revolving delinquency",
    "never_flag__mths_since_last_delinq": "never delinquent flag",
    "never_flag__mths_since_last_major_derog": "no major derogatory mark on file flag",
    "never_flag__mths_since_last_record": "no public record on file flag",
    "never_flag__mths_since_recent_bc_dlq": "never had a bankcard delinquency flag",
    "never_flag__mths_since_recent_inq": "no recent credit inquiry flag",
    "never_flag__mths_since_recent_revol_delinq": "never had a revolving delinquency flag",
    "num_accts_ever_120_pd": "accounts ever 120+ days past due",
    "num_actv_bc_tl": "number of active bankcard accounts",
    "num_actv_rev_tl": "number of active revolving accounts",
    "num_bc_sats": "number of satisfactory bankcard accounts",
    "num_bc_tl": "number of bankcard accounts",
    "num_il_tl": "number of installment accounts",
    "num_op_rev_tl": "number of open revolving accounts",
    "num_rev_accts": "number of revolving accounts",
    "num_sats": "number of satisfactory accounts",
    "num_tl_120dpd_2m": "accounts 120+ days past due in the last 2 months",
    "num_tl_30dpd": "accounts currently 30 days past due",
    "num_tl_90g_dpd_24m": "accounts 90+ days past due in the last 24 months",
    "open_acc": "number of open credit accounts",
    "pct_tl_nvr_dlq": "percent of accounts never delinquent",
    "percent_bc_gt_75": "percent of bankcard accounts over 75% of limit",
    "pub_rec": "number of derogatory public records",
    "pub_rec_bankruptcies": "number of bankruptcies on record",
    "revol_bal": "revolving credit balance",
    "tax_liens": "number of tax liens",
    "tot_coll_amt": "total amount ever sent to collections",
    "tot_cur_bal": "total current balance across accounts",
    "total_acc": "total number of credit accounts ever opened",
    "total_bal_ex_mort": "total balance excluding mortgage",
    "total_il_high_credit_limit": "total installment high credit limit",
    "total_rev_hi_lim": "total revolving high credit limit",
    "verification_status": "income verification status",
}

PLAIN_ENGLISH = {**_PHASE_6_LABELS, **_ADDITIONAL_LABELS}


def describe(feature: str) -> str:
    return PLAIN_ENGLISH.get(feature, feature.replace("_", " "))
