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

    holdout_rng = np.random.default_rng(4103)
    holdout = holdout_rng.normal(size=(2000, 3))
    holdout_revenue = mu + signal(holdout) + holdout_rng.normal(0, 2, len(holdout))
    holdout_quality = pd.DataFrame(
        [
            {
                "score": name,
                "holdout_units": len(holdout),
                "holdout_correlation": np.corrcoef(score, holdout_revenue)[0, 1],
            }
            for name, score in (
                ("useful", features(holdout) @ frozen),
                ("irrelevant feature", irrelevant_feature(holdout) @ weak),
            )
        ]
    )

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
    return holdout_quality, known_lift, provenance, shifted, units, weak_prediction


@app.cell(hide_code=True)
def _(mo, provenance, units):
    def score_summary(frame):
        summary = frame.groupby("variant", as_index=False).agg(
            units=("user_id", "size"),
            scored=("prediction", "count"),
            score_p10=("prediction", lambda score: score.quantile(0.10)),
            score_median=("prediction", "median"),
            score_p90=("prediction", lambda score: score.quantile(0.90)),
        )
        return summary.assign(missing=summary["units"] - summary["scored"])[
            ["variant", "units", "scored", "missing", "score_p10", "score_median", "score_p90"]
        ].round(2)

    mo.vstack(
        [
            mo.md("## Can you trust how the scores were made?"),
            mo.ui.table(provenance),
            mo.md(
                "Here we apply one frozen model before assigning shoppers to either arm. "
                "For your data, verify the training cutoff, feature timestamps, model version, "
                "and that the same scoring code and join apply to both arms."
            ),
            mo.ui.table(score_summary(units)),
            mo.md(
                "Counts include every assigned shopper. The 10th, 50th, and 90th percentiles "
                "describe only the scores you have. Similar distributions cannot rule out "
                "late features or a bad join."
            ),
        ]
    )
    return (score_summary,)


@app.cell
def _(holdout_quality, mo):
    mo.vstack(
        [
            mo.md("## Does the score predict revenue for shoppers the model hasn’t seen?"),
            mo.ui.table(holdout_quality.round({"holdout_correlation": 3})),
            mo.md(
                "These historical shoppers were used for neither training nor the experiment. "
                "Correlation measures whether higher scores tend to go with higher revenue. "
                "A score can be off in dollars and still be useful. This check does not "
                "establish that an experiment interval covers the true effect."
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
    comparison = []
    for scenario, rows in (
        ("useful", estimates),
        ("irrelevant feature", analyze(units.assign(prediction=weak_prediction))),
        ("shifted population", analyze(shifted)),
    ):
        revenue = {row.method: row for row in rows if row.metric == "revenue"}
        comparison.append(
            {
                "scenario": scenario,
                "unadjusted_abs_se": revenue["unadjusted"].abs_se,
                "cuped_abs_se": revenue["cuped"].abs_se,
            }
        )
    mo.vstack(
        [
            mo.md("## Did CUPED make the revenue comparison more precise?"),
            mo.ui.table(pd.DataFrame(comparison).round(4)),
            mo.md(
                "Both columns give the standard error of the treatment-minus-control difference "
                "in mean revenue, in the same units as revenue. Compare the columns within each "
                "row. Lower means less estimated uncertainty in this draw, not a guarantee. "
                "A weak score may add little. If scores used late data or were made differently "
                "by arm, do not interpret the adjusted result even if its standard error is smaller."
            ),
        ]
    )
    return


@app.cell
def _(analyze, mo, score_summary, units):
    treated = units["variant"] == "treatment"
    high = units["prediction"] >= units.loc[treated, "prediction"].quantile(0.55)
    missing_units = units.assign(prediction=units["prediction"].mask(treated & high))
    results = {row.method: row for row in analyze(missing_units)}
    decision, sensitivity = results["unadjusted"], results["cuped"]
    mo.vstack(
        [
            mo.md("## What happens when scores are missing"),
            mo.ui.table(score_summary(missing_units)),
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
            mo.md("## Why filling missing scores can mislead"),
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
