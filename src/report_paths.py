"""Where each reports/metrics/ file lives, grouped by topic subfolder:
data/ (raw data, cleaning, population, feature audit), models/ (scorecard,
LightGBM, XGBoost, served-model checks), evaluation/ (test-set metrics,
decision economics, post-hoc checks), drift/ (PSI replay), explain/ (SHAP).
"""

from pathlib import Path

METRICS_DIR = Path("reports/metrics")

_METRICS_TOPIC = {
    # data
    "raw_data_profile.json": "data",
    "cleaning_summary.json": "data",
    "status_by_year.csv": "data",
    "maturity_audit.csv": "data",
    "feature_audit.csv": "data",
    "missing_by_year.csv": "data",
    "field_rollout.csv": "data",
    "event_timing_audit_summary.json": "data",
    "population_and_target.json": "data",
    "eligibility_caps_by_year.csv": "data",
    "build_features_summary.json": "data",
    # models
    "frozen_model_config.json": "models",
    "gbm_calibration.json": "models",
    "gbm_feature_importance.csv": "models",
    "gbm_feature_list.json": "models",
    "gbm_tuning_trials.json": "models",
    "gbm_validation_metrics.json": "models",
    "logistic_regression_coefficients.csv": "models",
    "iv_table.csv": "models",
    "iv_suspicious_investigation.json": "models",
    "feature_selection_log.json": "models",
    "scorecard_calibration.json": "models",
    "scorecard_points.json": "models",
    "scorecard_score_summary.json": "models",
    "scorecard_validation_metrics.json": "models",
    "xgboost_calibration.json": "models",
    "xgboost_order_sensitivity_check.json": "models",
    "training_order_sensitivity.json": "models",
    "risk_bands.json": "models",
    "v2_promotion_summary.json": "models",
    "served_model_version_check.json": "models",
    "docker_image.json": "models",
    "api_latency.json": "models",
    "api_field_requirements.json": "models",
    # evaluation
    "decision_policy.json": "evaluation",
    "dti_cap_extrapolation_summary.json": "evaluation",
    "dti_cap_extrapolation_v1_test.csv": "evaluation",
    "dti_cap_extrapolation_v2_2016.csv": "evaluation",
    "dti_cap_extrapolation_v2_2016_relative.csv": "evaluation",
    "kpi_movement_monthly.csv": "evaluation",
    "kpi_movement_summary.json": "evaluation",
    "posthoc_ablations_summary.json": "evaluation",
    "posthoc_calibration_summary.json": "evaluation",
    "posthoc_funding_optimization_optimal.csv": "evaluation",
    "posthoc_funding_optimization_summary.json": "evaluation",
    "posthoc_matched_volume_comparison_table.csv": "evaluation",
    "posthoc_matched_volume_economics.csv": "evaluation",
    "posthoc_matched_volume_economics_summary.json": "evaluation",
    "posthoc_monthly_lift.csv": "evaluation",
    "posthoc_monthly_lift_summary.json": "evaluation",
    "posthoc_replay_24m_performance.csv": "evaluation",
    "posthoc_replay_24m_performance_summary.json": "evaluation",
    "posthoc_segment_summary.json": "evaluation",
    "posthoc_segment_table.csv": "evaluation",
    "posthoc_subgrade_lightgbm_matched_volume.csv": "evaluation",
    "posthoc_subgrade_lightgbm_matched_volume_summary.json": "evaluation",
    "posthoc_v2_evaluation.json": "evaluation",
    "posthoc_v2_label_robustness.json": "evaluation",
    "posthoc_v2_test_2015.json": "evaluation",
    "posthoc_walkforward.csv": "evaluation",
    "posthoc_walkforward_summary.json": "evaluation",
    "test_bootstrap_ci.json": "evaluation",
    "test_calibration_lightgbm.csv": "evaluation",
    "test_core_metrics.json": "evaluation",
    "test_decision_quality.csv": "evaluation",
    "test_economics_at_fixed_cutoff.csv": "evaluation",
    "test_fixed_approval_rate_comparison.csv": "evaluation",
    "test_gap_decomposition_lightgbm.json": "evaluation",
    "test_gap_remainder_diagnostic.json": "evaluation",
    "test_lgd_sensitivity.csv": "evaluation",
    "test_robustness.json": "evaluation",
    "test_scorecard_gap_explanation.json": "evaluation",
    "test_scored.parquet": "evaluation",
    "test_stability_by_quarter.csv": "evaluation",
    "test_stability_by_region.csv": "evaluation",
    "test_sub_grade_level_count.json": "evaluation",
    # drift
    "drift_baseline_psi_validation_vs_train.csv": "drift",
    "drift_combined_psi_table.csv": "drift",
    "drift_data_first_crossing.csv": "drift",
    "drift_data_monthly_psi.csv": "drift",
    "drift_evidently_file_sizes.json": "drift",
    "drift_loan_to_income_cap_signal.json": "drift",
    "drift_loan_to_income_psi_explanation.json": "drift",
    "drift_performance_2015_monthly.csv": "drift",
    "drift_prediction_monthly.csv": "drift",
    "drift_summary.json": "drift",
    # explain
    "shap_global_importance.csv": "explain",
    "shap_global_importance_served.csv": "explain",
    "shap_global_importance_served_summary.json": "explain",
    "shap_reason_codes_examples.json": "explain",
    "shap_rollout_missing_check.json": "explain",
    "shap_summary.json": "explain",
}


def metrics_path(filename: str) -> Path:
    """Full path for a reports/metrics/ file, creating its topic subfolder
    if needed. filename is the bare name, e.g. "cleaning_summary.json".
    """
    path = METRICS_DIR / _METRICS_TOPIC[filename] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
