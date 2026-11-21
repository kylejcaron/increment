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
- **Model and feature timing.** Record the model version, input transformations,
  and timestamps showing that every feature was available before each unit's
  assignment. A frozen model does not make a late feature safe.
- **Scoring coverage.** Count scored and missing units separately by assigned
  arm; an overall percentage can hide a gap. Investigate the cause of missing
  scores, even when coverage looks balanced.
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

The case `cuped_external_prediction_fixed_horizon` compares the decision and
sensitivity roles, estimates, and intervals on the same units and scores in a
fixed-horizon, whole-window mean example. It checks source agreement, not
statistical coverage.

| Entry point | How the case uses the score |
|---|---|
| `from_frame(..., shape="summary")` | One score per unit, named on the CUPED sensitivity. |
| `from_frame(..., shape="panel")` | The same score repeated across the unit's rows. |
| `from_definitions` | A continuous `as_of: pre_exposure` property on the prediction source, bound on the sensitivity with `n_pre_periods: 0`. |
| `from_unit_day_artifact` | Reopen the published artifact with its saved named covariate and matching definition binding. |
| `from_moments` | Replay the moments produced by `workflows.export_moments`; the export must carry the statistics for that score. |
| `from_switchback_frame` | Refuses CUPED with `source.frame.switchback.metric`. |

This case does not establish support for arbitrary artifacts, breakouts, daily
or as-of results, or sequential inference. In particular, freezing the predictor
does not fix the CUPED adjustment coefficient before experiment observations
arrive. Use the existing [CUPED compatibility rules](compatibility.md).

## Interpret the comparison

A useful score can narrow an interval. A score based on an irrelevant feature
may change little, and a score trained on a different population can lose
precision when the relationship between features and outcomes shifts. The
notebook compares all three on one synthetic experiment. Its control-arm
calibration is about prediction quality; neither it nor a narrower interval
establishes interval coverage. The [validation record](../validation.md) describes
the separate repeated-draw checks and their limits.

For the original recipe, see [DoorDash's introduction to CUPAC](https://careersatdoordash.com/blog/improving-experimental-power-through-control-using-predictions-as-covariate-cupac).
