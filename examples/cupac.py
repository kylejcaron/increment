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

    import increment
    from increment.errors import IncrementWarning

    return IncrementWarning, alt, increment, mo, np, pd, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CUPAC: CUPED with external predictions

    Train a model on historical units, freeze it, and score experiment units
    before assignment. Increment treats the stored prediction as one numeric
    covariate. It estimates the adjustment from experiment data but never
    refits the prediction model or substitutes its scores for actual outcomes.
    """)
    return


@app.cell
def _(np, pd):
    mu = 20.0
    effect = 0.6
    known_lift = effect / mu
    n_history = 5000
    n_units = 3000
    training_cutoff = pd.Timestamp("2025-01-31T23:59:59Z")
    feature_as_of = pd.Timestamp("2025-02-01T08:00:00Z")
    assignment_at = pd.Timestamp("2025-02-01T09:00:00Z")
    history_ids = {f"history-{i}" for i in range(n_history)}
    unit_ids = [f"experiment-{i}" for i in range(n_units)]
    assert history_ids.isdisjoint(unit_ids)
    assert training_cutoff < feature_as_of < assignment_at

    def features(values):
        f1, f2 = values[:, 0], values[:, 1]
        return np.column_stack((np.ones(len(values)), f1, f2, f1**2, f1 * f2))

    def irrelevant_feature(values):
        return np.column_stack((np.ones(len(values)), values[:, 2]))

    def signal(values):
        f1, f2 = values[:, 0], values[:, 1]
        return 3 * f1 + 2 * f2 + 1.5 * (f1**2 - 1)

    history_rng = np.random.default_rng(4101)
    history = history_rng.normal(size=(n_history, 3))
    historical_revenue = mu + signal(history) + history_rng.normal(0, 2, n_history)
    frozen = np.linalg.lstsq(features(history), historical_revenue, rcond=None)[0]
    weak = np.linalg.lstsq(irrelevant_feature(history), historical_revenue, rcond=None)[0]

    experiment_rng = np.random.default_rng(4102)
    inputs = experiment_rng.normal(size=(n_units, 3))
    prediction = features(inputs) @ frozen
    weak_prediction = irrelevant_feature(inputs) @ weak
    assert np.array_equal(features(inputs) @ frozen, prediction)

    variant = np.repeat(["control", "treatment"], n_units // 2)
    noise = experiment_rng.normal(0, 2, n_units)
    mean = mu + signal(inputs) + effect * (variant == "treatment")
    sessions = 1 + experiment_rng.poisson(3, n_units)
    units = pd.DataFrame(
        {
            "user_id": unit_ids,
            "variant": variant,
            "revenue": mean + noise,
            "total_revenue": sessions * mean + noise * np.sqrt(sessions),
            "sessions": sessions,
            "prediction": prediction,
        }
    )
    f1, f2 = inputs[:, 0], inputs[:, 1]
    shifted_signal = 1.5 * f1 + 2 * f2 + 3 * f1 * f2
    shifted = units.assign(revenue=units["revenue"] - signal(inputs) + shifted_signal)
    provenance = pd.DataFrame(
        [
            {
                "training_cutoff": training_cutoff.isoformat(),
                "feature_as_of": feature_as_of.isoformat(),
                "assignment_at": assignment_at.isoformat(),
                "model_version": "cupac-demo-1",
                "trained_on": "simulated historical units, disjoint from experiment",
                "scoring_check": "the stored prediction reproduced from the same inputs",
            }
        ]
    )
    return known_lift, provenance, shifted, units, weak_prediction


@app.cell(hide_code=True)
def _(mo, provenance, units):
    coverage = units.groupby("variant", as_index=False).agg(
        units=("user_id", "size"), scored=("prediction", "count")
    )
    coverage["scored_fraction"] = coverage["scored"] / coverage["units"]
    mo.vstack(
        [
            mo.md("## Check timing, model identity, and scoring coverage by arm"),
            mo.ui.table(provenance),
            mo.ui.table(coverage),
            mo.md("The score column itself cannot prove the timing or rule out leakage."),
        ]
    )
    return


@app.cell
def _(alt, mo, np, units):
    control = units.loc[units["variant"] == "control"]
    slope, intercept = np.polyfit(control["prediction"], control["revenue"], 1)
    calibration = (
        alt.Chart(control.sample(n=600, random_state=17))
        .mark_circle(opacity=0.35, size=20)
        .encode(
            x=alt.X("prediction:Q", title="Frozen prediction", scale=alt.Scale(zero=False)),
            y=alt.Y("revenue:Q", title="Observed revenue", scale=alt.Scale(zero=False)),
        )
        .properties(width="container", height=240)
    )
    mo.vstack(
        [
            mo.md("## Check prediction quality"),
            calibration,
            mo.md(
                f"The fitted line on control units has slope **{slope:.2f}** and "
                f"intercept **{intercept:.2f}**. This checks the relationship between "
                "the score and the outcome in this sample, not interval coverage."
            ),
        ]
    )
    return


@app.cell
def _(increment):
    def analyze(frame, *, missing="refuse", include_ratio=False):
        covariate = increment.CupedCovariate(name="prediction", kind="continuous", missing=missing)
        methods = {
            "decision_method": increment.Method.unadjusted(name="unadjusted"),
            "sensitivity_methods": (increment.Method.cuped(covariates=covariate),),
        }
        metrics = [increment.MetricSpec(name="revenue", preferred_direction="increase", **methods)]
        if include_ratio:
            metrics.append(
                increment.MetricSpec(
                    name="revenue_per_session",
                    type="ratio",
                    numerator="total_revenue",
                    denominator="sessions",
                    **methods,
                )
            )
        experiment, reader = increment.from_frame(
            frame,
            shape="summary",
            unit="user_id",
            group="variant",
            design=increment.Randomized(
                control_group="control",
                allocation={"control": 0.5, "treatment": 0.5},
            ),
            metrics=metrics,
            plan=increment.AnalysisPlan(
                primary="revenue", secondaries=("revenue_per_session",) if include_ratio else ()
            ),
        )
        return increment.run(experiment, reader).readout()

    def method_record(row):
        lift = row.require_lift()
        return {
            "metric": row.metric,
            "method": row.method,
            "role": row.method_role,
            "lift": lift.value,
            "lower": lift.lb,
            "upper": lift.ub,
        }

    return analyze, method_record


@app.cell
def _(alt, analyze, known_lift, method_record, mo, pd, units):
    estimates = analyze(units, include_ratio=True)
    method_table = pd.DataFrame([method_record(row) for row in estimates])
    mean_rows = method_table.loc[method_table["metric"] == "revenue"]
    interval_chart = (
        alt.Chart(mean_rows)
        .mark_errorbar(ticks=True)
        .encode(
            x=alt.X("lower:Q", title="Relative lift (revenue)", axis=alt.Axis(format=".1%")),
            x2="upper:Q",
            y=alt.Y("role:N", title=None),
        )
        .properties(width="container", height=100)
    )
    mo.vstack(
        [
            mo.md("## Read the decision and sensitivity results"),
            mo.ui.table(method_table),
            interval_chart,
            mo.md(
                f"The synthetic relative effect is **{known_lift:.1%}** for both metrics. "
                "Use the unadjusted row for the declared decision even if the CUPED interval "
                "is narrower. Revenue per session is a ratio of arm means; CUPED adjusts "
                "the numerator and denominator separately."
            ),
        ]
    )
    return (estimates,)


@app.cell(hide_code=True)
def _(analyze, estimates, mo, pd, shifted, units, weak_prediction):
    def width_ratio(rows):
        lifts = {row.method: row.require_lift() for row in rows if row.metric == "revenue"}
        return (lifts["cuped"].ub - lifts["cuped"].lb) / (
            lifts["unadjusted"].ub - lifts["unadjusted"].lb
        )

    comparison = pd.DataFrame(
        [
            {"score": "useful", "CUPED / unadjusted interval width": width_ratio(estimates)},
            {
                "score": "irrelevant feature",
                "CUPED / unadjusted interval width": width_ratio(
                    analyze(units.assign(prediction=weak_prediction))
                ),
            },
            {
                "score": "shifted population",
                "CUPED / unadjusted interval width": width_ratio(analyze(shifted)),
            },
        ]
    )
    mo.vstack(
        [
            mo.md("## Compare useful, weak, and shifted predictions"),
            mo.ui.table(comparison),
            mo.md("Below one means narrower. These are results from one draw, not guarantees."),
        ]
    )
    return


@app.cell
def _(analyze, mo, units):
    treated = units["variant"] == "treatment"
    high = units["prediction"] >= units.loc[treated, "prediction"].quantile(0.55)
    missing_units = units.assign(prediction=units["prediction"].mask(treated & high))
    missingness = missing_units.groupby("variant", as_index=False).agg(
        units=("user_id", "size"), scored=("prediction", "count")
    )
    results = {row.method: row for row in analyze(missing_units)}
    decision, sensitivity = results["unadjusted"], results["cuped"]
    mo.vstack(
        [
            mo.md("## Refuse selectively missing scores"),
            mo.ui.table(missingness),
            mo.md(
                f"The unadjusted decision still reports **{decision.require_lift().value:.2%}**. "
                f"The CUPED sensitivity is unavailable: `{sensitivity.failure_code}`. "
                "Fix and recheck scoring before interpreting an adjusted result."
            ),
        ]
    )
    return (missing_units,)


@app.cell
def _(IncrementWarning, analyze, known_lift, missing_units, mo, pd, warnings):
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always", IncrementWarning)
        imputed = analyze(missing_units, missing="impute")
    imputed_lift = next(row for row in imputed if row.method == "cuped").require_lift().value
    mo.vstack(
        [
            mo.md("## Deliberately impute in a separate analysis"),
            mo.ui.table(
                pd.DataFrame(
                    [{"imputed_cuped_lift": imputed_lift, "known_relative_lift": known_lift}]
                )
            ),
            mo.md(
                "Filling missing scores with the pooled mean shifts the adjusted result "
                "in this example because high treatment scores were removed. "
                "Imputation is not a fallback for failed scoring. "
                + " ".join(f"`{item.message}`" for item in diagnostics)
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
