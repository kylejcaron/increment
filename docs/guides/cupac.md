# CUPAC: use a frozen prediction with CUPED

If you already have one numeric prediction per unit, scored before assignment
by a model trained independently of this experiment, use it as a [CUPED](cuped.md)
covariate. This recipe is often called *control using predictions as covariates*
(CUPAC). Increment does not train or refit your predictor. It uses the experiment
outcomes to estimate the usual scalar CUPED adjustment, not to replace the
observed outcomes with predictions.

## Prepare the prediction

Train on a separate historical sample, freeze the model and transformations,
and score each experiment unit before assignment. Join the stored scores to
experiment units by ID. The model can use many features; the analysis sees one
column. For a panel, repeat the same score on every row for that unit.

Before looking at results, record and check:

- **Training cutoff and cohort.** Record the last time included in training
  (and its timezone), and check training units against experiment IDs. Do not
  train on the experiment's assignments, features measured after assignment,
  or outcomes.
- **Model and feature timing.** Record the model version and verify that every
  feature was available before assignment. Use the same model, scoring code,
  and join for both arms. A frozen model does not make a late feature safe.
- **Scoring coverage.** Compare scored and missing counts and score percentiles
  by arm. Count missing scores among all assigned units; compute percentiles
  only from scores you have. Similar distributions do not prove a safe join.
- **Reproducibility.** Retain the model, scoring code, input snapshot, and scored
  outputs. Check that the same inputs reproduce the stored scores.

A numeric column alone cannot prove independence, correct timing, or absence of
leakage from experiment outcomes. Do not substitute a variable measured after
assignment just because it predicts the outcome well. To adjust with several raw
covariates instead of one score, see [multicovariate CUPED](../reference/variance-reduction.md#mixed-numeric-and-categorical-covariates).
Fitting the prediction model on this experiment would need a different design.

## Keep the decision and sensitivity separate

Prespecify the unadjusted decision method and CUPED sensitivity before seeing the
results. Do not pick the narrower interval afterward. The rows below are example
inputs with scores already stored; the [notebook](../examples/cupac.md) shows how
a separate historical sample can produce them.

```python
import increment
import pandas as pd

units = pd.DataFrame(
    {
        "user_id": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
        "variant": ["control"] * 4 + ["treatment"] * 4,
        "revenue": [13.0, 18.2, 21.5, 24.3, 15.4, 19.6, 23.4, 26.2],
        "sessions": [2, 3, 4, 4, 2, 4, 3, 5],
        "prediction": [13.8, 17.9, 20.1, 24.5, 14.8, 19.3, 21.8, 25.3],
    }
)

prediction = increment.CupedCovariate(name="prediction", kind="continuous", missing="refuse")
methods = {
    "decision_method": increment.Method.unadjusted(name="unadjusted"),
    "sensitivity_methods": (increment.Method.cuped(covariates=prediction),),
}
experiment, reader = increment.from_frame(
    units,
    shape="summary",
    unit="user_id",
    group="variant",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[
        increment.MetricSpec(
            name="revenue", type="mean", preferred_direction="increase", **methods
        ),
        increment.MetricSpec(
            name="revenue_per_session",
            type="ratio",
            numerator="revenue",
            denominator="sessions",
            **methods,
        ),
    ],
    plan=increment.AnalysisPlan(primary="revenue", secondaries=("revenue_per_session",)),
)
result = increment.run(experiment, reader)
for row in result.readout():
    lift = row.lift
    if lift is None:
        print(row.metric, row.method_role, row.method, row.failure_code)
    else:
        print(row.metric, row.method_role, row.method, lift.value, (lift.lb, lift.ub))
```

You should see four rows: an unadjusted result and a CUPED result for each
metric. Use the unadjusted result for your planned decision. Use the CUPED
result to see how adjusting for the prediction changes the estimate and interval.

With a complete score column, `Method.cuped(covariates="prediction")` is the
short form. The explicit declaration above lets you choose what happens when a
score is missing. For revenue per session, the analysis still estimates the
ratio of arm means (mean revenue divided by mean sessions). CUPED adjusts the
numerator and denominator with separate slopes; it does not average predicted
per-unit ratios. See [CUPED for ratio metrics](cuped.md#ratio-metrics).

## When predictions are missing

With `missing="refuse"`, a null or NaN score leaves the unadjusted decision row
available and reports `frame.validation.metric_covariate_column` on the
unavailable CUPED sensitivity row. Fix scoring and reassess coverage by arm
before interpreting the adjusted result. A prediction that changes across rows
for the same unit is a different problem: the panel builder refuses with
`frame.frame_panel.unit_covariate_varies`. Fix the join or supply one stored
score per unit; filling missing scores does not resolve conflicting scores.

Pooled-mean imputation (`missing="impute"`) is a separate choice, not a fallback.
If high scores vanish selectively from one arm, replacing them with the observed
mean can change the adjusted estimate even if the model was safely trained.
The notebook demonstrates that failure mode with synthetic data. It does not
measure the bias you would get in another experiment.

## Where to supply the score

This table shows what to supply for a mean measured over the full experiment
at a fixed end date.

| Entry point | What to supply |
|---|---|
| `from_frame(..., shape="summary")` | One score per unit, named on the CUPED sensitivity. |
| `from_frame(..., shape="panel")` | The same score repeated across the unit's rows. |
| `from_definitions` | A continuous `as_of: pre_exposure` property on the prediction source, bound on the sensitivity with `n_pre_periods: 0`. |
| `from_unit_day_artifact` | An artifact with the saved score and the matching definition that names it. |
| `from_moments` | Replay the moments produced by `workflows.export_moments`; the export must carry the statistics for that score. |
| `from_switchback_frame` | Refuses CUPED with `source.frame.switchback.metric`. |

For other artifacts, breakouts, daily or as-of results, or sequential
inference, check the [CUPED compatibility rules](compatibility.md). Freezing
the prediction model does not set the CUPED adjustment in advance.

## Interpret the comparison

On historical data the model did not train on, check whether higher scores go
with higher observed outcomes. A model that overpredicts everyone by the same
amount can still help. In the experiment, compare the [standard errors of the
same absolute difference](cuped.md#measuring-variance-reduction) with and
without adjustment. A weak or shifted score may add little precision.

An [unusual gap in average scores](cuped.md#when-cuped-misleads-a-covariate-that-differs-by-arm)
triggers a warning, not a refusal. Investigate scoring, the join, and assignment.
If the score used late information or the scoring process differed by arm, do
not interpret the adjusted result, even if it looks more precise. Neither
check changes the planned decision or establishes interval coverage. See the
[validation record](../validation.md#scalar-cuped-with-a-frozen-external-prediction).

For the original recipe, see [DoorDash's introduction to CUPAC](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac).
