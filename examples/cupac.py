import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from increment import Analysis, Method, MetricSpec
    from increment.errors import CodedError

    return Analysis, CodedError, Method, MetricSpec, alt, mo, np, pd, warnings


@app.cell
def _(mo):
    mo.md(r"""
    # CUPAC: CUPED with external predictions

    CUPAC here uses the ordinary scalar CUPED adjustment. A model is trained on
    historical units, frozen, and scored before assignment; its prediction is the
    single per-unit covariate. The experiment never trains or refits the predictor.

    The historical training window, feature timing, and unit separation below are
    explicit analyst declarations. Interval calibration is checked by repeated-draw
    tests, not inferred from this one synthetic experiment.
    """)
    return


@app.cell
def _(np):
    MU0 = 20.0
    TAU = 0.6
    TRUE_LIFT = TAU / MU0
    TRAINING_CUTOFF = "2025-01-31"
    EXPERIMENT_START = "2025-02-01"
    N_HIST = 5000
    N_PER_ARM = 1500
    MODEL_VERSION = "cupac-demo-1"

    def basis(features):
        f1, f2 = features[:, 0], features[:, 1]
        return np.column_stack((np.ones(len(features)), f1, f2, f1**2, f1 * f2))

    def fit_frozen(features_hist, y_hist):
        return np.linalg.lstsq(features_hist, y_hist, rcond=None)[0]

    historical_rng = np.random.default_rng(4101)
    historical_features = historical_rng.normal(size=(N_HIST, 3))
    h1, h2 = historical_features[:, 0], historical_features[:, 1]
    historical_signal = 3 * h1 + 2 * h2 + 1.5 * (h1**2 - 1)
    historical_outcome = MU0 + historical_signal + historical_rng.normal(0, 2, N_HIST)
    coefficients = fit_frozen(basis(historical_features[:, :2]), historical_outcome)
    historical_unit_ids = {f"history-{i}" for i in range(N_HIST)}
    return (
        EXPERIMENT_START,
        MODEL_VERSION,
        MU0,
        N_PER_ARM,
        TAU,
        TRAINING_CUTOFF,
        TRUE_LIFT,
        basis,
        coefficients,
        historical_unit_ids,
    )


@app.cell
def _(
    EXPERIMENT_START,
    MODEL_VERSION,
    TRAINING_CUTOFF,
    experiment_unit_ids,
    historical_unit_ids,
    pd,
):
    provenance = {
        "training_cutoff": TRAINING_CUTOFF,
        "feature_as_of": "before assignment",
        "model_version": MODEL_VERSION,
        "trained_on": "historical units, disjoint from experiment",
    }
    assert TRAINING_CUTOFF < EXPERIMENT_START
    assert historical_unit_ids.isdisjoint(experiment_unit_ids)
    provenance_frame = pd.DataFrame([provenance])
    return provenance, provenance_frame


@app.cell
def _(MU0, N_PER_ARM, TAU, basis, coefficients, historical_unit_ids, np, pd):
    experiment_rng = np.random.default_rng(4102)
    experiment_features = experiment_rng.normal(size=(2 * N_PER_ARM, 3))
    prediction = basis(experiment_features[:, :2]) @ coefficients
    assignment = np.repeat(["control", "treatment"], N_PER_ARM)
    treatment = (assignment == "treatment").astype(float)
    f1, f2 = experiment_features[:, 0], experiment_features[:, 1]
    control_signal = 3 * f1 + 2 * f2 + 1.5 * (f1**2 - 1)
    revenue = MU0 + control_signal + TAU * treatment + experiment_rng.normal(0, 2, 2 * N_PER_ARM)
    experiment = pd.DataFrame(
        {
            "user_id": [f"experiment-{i}" for i in range(2 * N_PER_ARM)],
            "variant": assignment,
            "revenue": revenue,
            "prediction": prediction,
        }
    )
    experiment_unit_ids = set(experiment["user_id"])
    assert experiment_unit_ids.isdisjoint(historical_unit_ids)
    return experiment, experiment_unit_ids


@app.cell
def _(alt, experiment, mo):
    coverage = experiment.groupby("variant", as_index=False).agg(
        units=("user_id", "size"), scored=("prediction", "count")
    )
    coverage["prediction_coverage"] = coverage["scored"] / coverage["units"]
    mo.md("## Prediction coverage by randomized arm")
    mo.ui.table(coverage)
    coverage_plot = (
        alt.Chart(coverage)
        .mark_bar()
        .encode(x="variant:N", y="prediction_coverage:Q", color="variant:N")
        .properties(title="Prediction coverage before adjustment", width=420)
    )
    coverage_plot
    return coverage, coverage_plot


@app.cell
def _(alt, experiment, mo, np):
    control = experiment.loc[experiment["variant"] == "control"]
    calibration_slope, calibration_intercept = np.polyfit(
        control["prediction"], control["revenue"], 1
    )
    fitted_control = calibration_intercept + calibration_slope * control["prediction"]
    calibration_r2 = 1 - np.sum((control["revenue"] - fitted_control) ** 2) / np.sum(
        (control["revenue"] - control["revenue"].mean()) ** 2
    )
    calibration = {
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibration_r2": calibration_r2,
    }
    calibration_plot = (
        alt.Chart(control.sample(n=min(800, len(control)), random_state=17))
        .mark_circle(opacity=0.35, size=18)
        .encode(x="prediction:Q", y="revenue:Q")
        .properties(title="Predictor calibration on control units", width=420)
    )
    mo.md(
        f"Predictor calibration (not interval coverage): control-arm slope "
        f"**{calibration_slope:.3f}**, intercept **{calibration_intercept:.3f}**, "
        f"$R^2$ **{calibration_r2:.3f}**."
    )
    calibration_plot
    return calibration, calibration_intercept, calibration_plot, calibration_r2, calibration_slope


@app.cell
def _(Analysis, Method, MetricSpec, TRUE_LIFT, alt, experiment, mo, pd):
    UNADJUSTED = Method(name="unadjusted")
    CUPED = Method(name="cuped", variance_reduction="cuped")
    analysis = Analysis.from_unit_summary(
        experiment,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                type="mean",
                value_column="revenue",
                covariate="prediction",
                covariate_missing="error",
            )
        ],
    )
    estimates = analysis.run(
        decision_method=UNADJUSTED,
        sensitivity_methods=(CUPED,),
    )
    treatment_rows = [row for row in estimates if row.group_id == "treatment"]
    rows = []
    for row in treatment_rows:
        lift = row.require_lift()
        rows.append(
            {
                "method": row.method,
                "lift": lift.value,
                "lower": lift.lb,
                "upper": lift.ub,
                "interval_width": lift.ub - lift.lb,
            }
        )
    method_table = pd.DataFrame(rows)
    widths = {row["method"]: row["interval_width"] for row in rows}
    width_ratio = widths["cuped"] / widths["unadjusted"]
    truth_summary = pd.DataFrame(
        [{"known_relative_lift": TRUE_LIFT, "cuped_to_unadjusted_width_ratio": width_ratio}]
    )
    interval_chart = (
        alt.Chart(method_table)
        .mark_errorbar(ticks=True)
        .encode(x="lower:Q", x2="upper:Q", y="method:N", color="method:N")
        .properties(title="Relative-lift intervals", width=420)
    )
    mo.md(
        f"The known relative-lift truth is **{TRUE_LIFT:.3%}**; the observed "
        f"CUPED/unadjusted interval-width ratio is **{width_ratio:.3f}**."
    )
    mo.ui.table(method_table)
    mo.ui.table(truth_summary)
    interval_chart
    return (
        CUPED,
        UNADJUSTED,
        analysis,
        estimates,
        interval_chart,
        method_table,
        truth_summary,
        width_ratio,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Shifted-population example

    The frozen model is now scored on a population whose outcome relationship has
    shifted. This single draw illustrates the estimate and interval; it does not
    establish coverage. Interval calibration comes from the repeated-draw tests
    `test_cupac_coverage_and_bias` and `test_cupac_coverage_smoke` in
    `tests/test_cupac.py`.
    """)
    return


@app.cell
def _(
    Analysis, CUPED, Method, MetricSpec, MU0, N_PER_ARM, TAU, TRUE_LIFT, basis, coefficients, np, pd
):
    shifted_rng = np.random.default_rng(4103)
    shifted_features = shifted_rng.normal(size=(2 * N_PER_ARM, 3))
    shifted_prediction = basis(shifted_features[:, :2]) @ coefficients
    shifted_assignment = np.repeat(["control", "treatment"], N_PER_ARM)
    shifted_treatment = (shifted_assignment == "treatment").astype(float)
    shifted_f1, shifted_f2 = shifted_features[:, 0], shifted_features[:, 1]
    shifted_signal = 1.5 * shifted_f1 + 2 * shifted_f2 + 3.0 * shifted_f1 * shifted_f2
    shifted_revenue = (
        MU0 + shifted_signal + TAU * shifted_treatment + shifted_rng.normal(0, 2, 2 * N_PER_ARM)
    )
    shifted_frame = pd.DataFrame(
        {
            "user_id": [f"shifted-{i}" for i in range(2 * N_PER_ARM)],
            "variant": shifted_assignment,
            "revenue": shifted_revenue,
            "prediction": shifted_prediction,
        }
    )
    shifted_analysis = Analysis.from_unit_summary(
        shifted_frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="shifted_revenue",
                type="mean",
                value_column="revenue",
                covariate="prediction",
                covariate_missing="error",
            )
        ],
    )
    shifted_estimates = shifted_analysis.run(
        decision_method=Method(name="unadjusted"),
        sensitivity_methods=(CUPED,),
    )
    shifted_treatment_row = next(
        row for row in shifted_estimates if row.group_id == "treatment" and row.method == "cuped"
    )
    shifted_lift = shifted_treatment_row.require_lift()
    shifted_lift_table = pd.DataFrame(
        [
            {
                "shifted_lift": shifted_lift.value,
                "lower": shifted_lift.lb,
                "upper": shifted_lift.ub,
                "known_relative_lift": TRUE_LIFT,
            }
        ]
    )
    return shifted_frame, shifted_lift, shifted_lift_table


@app.cell
def _(mo, shifted_lift_table):
    mo.md("The shifted-population draw is a single illustration against the known truth.")
    mo.ui.table(shifted_lift_table)
    return


@app.cell
def _(
    Analysis, CodedError, Method, MetricSpec, MU0, TAU, TRUE_LIFT, experiment, mo, np, pd, warnings
):
    missing_frame = experiment.copy()
    treatment_mask = missing_frame["variant"] == "treatment"
    treated_prediction_cutoff = missing_frame.loc[treatment_mask, "prediction"].quantile(0.55)
    arm_dependent_missing = treatment_mask & (
        missing_frame["prediction"] >= treated_prediction_cutoff
    )
    missing_frame.loc[arm_dependent_missing, "prediction"] = np.nan
    missingness = missing_frame.groupby("variant", as_index=False).agg(
        units=("user_id", "size"), predictions=("prediction", "count")
    )
    missingness["missing_fraction"] = 1 - missingness["predictions"] / missingness["units"]
    try:
        error_analysis = Analysis.from_unit_summary(
            missing_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    value_column="revenue",
                    covariate="prediction",
                    covariate_missing="error",
                )
            ],
        )
        error_analysis.run()
    except CodedError as exc:
        missing_error_code = exc.code
        assert missing_error_code == "frame.validation.metric_covariate_column"
    else:
        raise AssertionError("missing predictions under error policy must be refused")

    with warnings.catch_warnings(record=True) as imputation_warnings:
        warnings.simplefilter("always")
        imputed_analysis = Analysis.from_unit_summary(
            missing_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    type="mean",
                    value_column="revenue",
                    covariate="prediction",
                    covariate_missing="impute",
                )
            ],
        )
        imputed_rows = imputed_analysis.run(
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
        )
    imputed_cuped_row = next(
        row for row in imputed_rows if row.group_id == "treatment" and row.method == "cuped"
    )
    imputed_lift = imputed_cuped_row.require_lift()
    imputation_result = pd.DataFrame(
        [
            {
                "imputed_cuped_lift": imputed_lift.value,
                "known_relative_lift": TRUE_LIFT,
                "absolute_error": abs(imputed_lift.value - TRUE_LIFT),
            }
        ]
    )
    assert imputation_warnings
    imputation_diagnostic = str(imputation_warnings[0].message)
    assert abs(imputed_lift.value - TRUE_LIFT) > 0.005
    mo.md(
        f"The `error` policy refuses with **{missing_error_code}**. In this fixed-seed "
        "example, predictions are missing selectively in the treatment arm; pooled-mean "
        "imputation then produces a biased CUPED lift. This is why the recipe recommends "
        '`covariate_missing="error"`. Use `impute` only when scoring is pre-assignment '
        "and missingness cannot depend on assignment."
    )
    mo.md(f"Observed imputation diagnostic: `{imputation_diagnostic}`")
    mo.ui.table(missingness)
    mo.ui.table(imputation_result)
    return imputation_result, missing_error_code, missingness, imputation_diagnostic


@app.cell
def _(mo, provenance_frame):
    mo.md("## CUPAC provenance record")
    mo.ui.table(provenance_frame)
    return


if __name__ == "__main__":
    app.run()
