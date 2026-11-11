"""Pydantic schemas. Input fields are RAW application fields only - the
same 59 allow-listed raw features plus the 3 fields needed purely to
derive engineered features (fico_range_low, fico_range_high,
earliest_cr_line). Engineered features are never accepted as input; they
are always computed server-side by src.features.feature_engineering,
the same module the training pipeline imports.
"""

from datetime import date

from pydantic import BaseModel, Field, field_validator

CATEGORICAL_FIELDS = ["home_ownership", "purpose", "verification_status"]


class ScoreRequest(BaseModel):
    # --- Required, hard-validated fields ---
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount, dollars")
    annual_inc: float = Field(..., ge=0, description="Self-reported annual income, dollars")
    dti: float = Field(..., ge=0, description="Debt-to-income ratio")
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)
    earliest_cr_line: date = Field(..., description="Date the earliest credit line was opened")
    issue_d: date | None = Field(
        None,
        description="Origination date. Not a field a live applicant supplies; "
        "defaults to today if omitted. Only meant to be set explicitly when "
        "scoring a historical record (e.g. a parity or backtest check), so "
        "credit_history_months is computed against the real origination date.",
    )
    home_ownership: str
    purpose: str
    verification_status: str
    emp_length: float | None = Field(None, ge=0, le=10)

    # --- Bureau/trade-line fields: non-negative where provided, missing
    # allowed (optbinning/LightGBM both handle missing natively, and
    # several of these are legitimately missing-not-at-random in training). ---
    acc_now_delinq: float | None = Field(None, ge=0)
    acc_open_past_24mths: float | None = Field(None, ge=0)
    avg_cur_bal: float | None = Field(None, ge=0)
    bc_open_to_buy: float | None = Field(None, ge=0)
    bc_util: float | None = Field(None, ge=0)
    chargeoff_within_12_mths: float | None = Field(None, ge=0)
    collections_12_mths_ex_med: float | None = Field(None, ge=0)
    delinq_2yrs: float | None = Field(None, ge=0)
    delinq_amnt: float | None = Field(None, ge=0)
    inq_last_6mths: float | None = Field(None, ge=0)
    mo_sin_old_il_acct: float | None = Field(None, ge=0)
    mo_sin_old_rev_tl_op: float | None = Field(None, ge=0)
    mo_sin_rcnt_rev_tl_op: float | None = Field(None, ge=0)
    mo_sin_rcnt_tl: float | None = Field(None, ge=0)
    mort_acc: float | None = Field(None, ge=0)
    mths_since_last_delinq: float | None = Field(None, ge=0)
    mths_since_last_major_derog: float | None = Field(None, ge=0)
    mths_since_last_record: float | None = Field(None, ge=0)
    mths_since_recent_bc: float | None = Field(None, ge=0)
    mths_since_recent_bc_dlq: float | None = Field(None, ge=0)
    mths_since_recent_inq: float | None = Field(None, ge=0)
    mths_since_recent_revol_delinq: float | None = Field(None, ge=0)
    num_accts_ever_120_pd: float | None = Field(None, ge=0)
    num_actv_bc_tl: float | None = Field(None, ge=0)
    num_actv_rev_tl: float | None = Field(None, ge=0)
    num_bc_sats: float | None = Field(None, ge=0)
    num_bc_tl: float | None = Field(None, ge=0)
    num_il_tl: float | None = Field(None, ge=0)
    num_op_rev_tl: float | None = Field(None, ge=0)
    num_rev_accts: float | None = Field(None, ge=0)
    num_rev_tl_bal_gt_0: float | None = Field(None, ge=0)
    num_sats: float | None = Field(None, ge=0)
    num_tl_120dpd_2m: float | None = Field(None, ge=0)
    num_tl_30dpd: float | None = Field(None, ge=0)
    num_tl_90g_dpd_24m: float | None = Field(None, ge=0)
    num_tl_op_past_12m: float | None = Field(None, ge=0)
    open_acc: float | None = Field(None, ge=0)
    pct_tl_nvr_dlq: float | None = Field(None, ge=0)
    percent_bc_gt_75: float | None = Field(None, ge=0)
    pub_rec: float | None = Field(None, ge=0)
    pub_rec_bankruptcies: float | None = Field(None, ge=0)
    revol_bal: float | None = Field(None, ge=0)
    revol_util: float | None = Field(None, ge=0)
    tax_liens: float | None = Field(None, ge=0)
    tot_coll_amt: float | None = Field(None, ge=0)
    tot_cur_bal: float | None = Field(None, ge=0)
    tot_hi_cred_lim: float | None = Field(None, ge=0)
    total_acc: float | None = Field(None, ge=0)
    total_bal_ex_mort: float | None = Field(None, ge=0)
    total_bc_limit: float | None = Field(None, ge=0)
    total_il_high_credit_limit: float | None = Field(None, ge=0)
    total_rev_hi_lim: float | None = Field(None, ge=0)

    @field_validator("fico_range_high")
    @classmethod
    def high_not_below_low(cls, v, info):
        low = info.data.get("fico_range_low")
        if low is not None and v < low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return v


class Driver(BaseModel):
    feature: str
    plain_english: str
    shap_value_log_odds: float


class RangeWarning(BaseModel):
    field: str
    reason: str
    value: str


class ScoreResponse(BaseModel):
    request_id: str
    calibrated_pd: float
    risk_band: str
    top_3_risk_increasing_drivers_illustrative_only: list[Driver]
    model_name: str
    model_version: str
    missing_fields: list[str] = []
    warnings: list[RangeWarning] = []


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
