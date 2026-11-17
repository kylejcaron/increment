# A/B testing in Python: quickstart

Compare treatment with control from a dataframe. You need one row per analysis
unit, its assigned group, and the metric you want to compare. No YAML or
warehouse connection is needed.

```bash
pip install --pre increment polars
```

This example uses Polars. Eager pandas and PyArrow frames work too.

## Build and run an experiment

These eight illustrative observations come from a two-group experiment.
Declare the assignment procedure that was actually used; here each unit had
an independent 50% chance of treatment. Revenue is the primary metric, and
increasing it is favorable.

```python
import increment
import polars as pl

df = pl.DataFrame(
    {
        "user_id": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
        "variant": ["control", "treatment"] * 4,
        "revenue": [12.0, 14.5, 9.0, 13.0, 11.5, 15.0, 10.0, 12.5],
    }
)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[increment.MetricSpec(name="revenue", type="mean", preferred_direction="increase")],
    plan=increment.AnalysisPlan(primary="revenue"),
)
result = increment.run(experiment, reader)
print(increment.render(result, "brief"))
```

The report starts with this result, wrapped here for readability:

```text
revenue / arm=treatment,
comparison=treatment vs control:
effect=29.4118% [6.20238%, 57.6933%]
(relative, 95.0% confidence);
status=available; 95.0% confidence;
1-member family
```

Treatment's average revenue is 29.4% higher. The 95% confidence interval runs
from 6.2% to 57.7% higher: the estimate is positive but imprecise. The report
also shows assignment checks and whether every required result is available.
Keep those qualifications when sharing it; this tiny example demonstrates
the API, not a real launch decision.

Keep `experiment` with the `reader` returned alongside it. The dataframe
reader captures the frame at construction; rebuild after correcting data.
See [experiment specifications](experiment-specifications.md) for saving
declarations, reader identity, offline checks and multiple views.

If units share a randomization cluster, pass `cluster=` naming that column.

## Missing data

Missing group labels and metric values are errors by default. Choose their
meaning explicitly.

### Missing group labels

`on_unassigned="exclude"` excludes unassigned rows.
`increment.workflows.unit_counts(experiment, reader)` and the sample-ratio
mismatch check account for them separately as `"(unassigned)"`, never as a
treatment group.

### Missing metric values

| Policy | Meaning |
|---|---|
| `MetricSpec(missing="zero")` | Missing means no events. |
| `MetricSpec(missing="skip")` | Missing means unobserved; omit it for this metric only. |

Skipping is unbiased only when missingness is unrelated to values. A
whole-window row's `population` records observed group counts; breakout
rows state that segment counts were not recorded.
`frame.validation.metric_missing_differs_by_arm` warns when missing shares
differ. Clustered designs do not run this check: compare missingness across
clusters yourself.

`increment.estimators.impute.zeros` and `drop_null` return
`(frame, affected_count)`. The former assumes no events; the latter drops
whole rows and needs the same missingness assumption.

!!! warning "Do not mean-fill outcomes"
    Mean-filling shrinks variance and can bias estimates. `pooled_mean` is
    for appropriate pre-period covariates, not outcomes.

Declare each column's role:

```python
import pyarrow as pa
from increment.estimators import impute

units = pa.table({"baseline": [4.0, None, 8.0]})
units, n_filled = impute.pooled_mean(units, "baseline", roles={"baseline": "covariate"})
print(n_filled, units["baseline"].to_pylist())
```

An `"outcome"` role raises `impute.pooled_mean_outcome`; no role emits
`impute.pooled_mean_role_undeclared` with per-column counts.
`MetricSpec(missing="impute")` raises `frame.metric.missing_impute`.
For a CUPED covariate, use its method-owned declaration:
`Method.cuped(covariates=CupedCovariate(name="baseline", kind="continuous", missing="impute"))`.

## Group identities

Labels become strings: integer `0` accepts `control=0` or `"0"`.
Boolean labels become `"true"`/`"false"`; Boolean columns also accept
`True`/`False` and `"True"`/`"False"` as controls. String columns preserve
their exact spelling. Boolean `from_moments` reloads accept Python Booleans
or canonical lowercase labels, not `"True"`/`"False"`.

Colliding labels such as `1` and `"1"` raise
`frame.validation.group_label_collision`; rename them.
`"(unassigned)"` and `"(mixed assignment)"` raise
`frame.validation.group_label_reserved`. Missing labels remain missing;
the literal `"nan"` is valid.

Old sequential checkpoints keyed by `"True"`/`"False"` are replay-only:
register a new protocol, rather than relabeling them.

## One row per unit per day

Use `shape="panel"` with `date=`; see the complete
[panel example](data-model.md#one-row-per-unit-per-day).
Daily/as-of views require `exposure_date=`. Units' values are combined across
days, so daily 0/1 indicators become counts, not “ever occurred” flags:
prepare the intended aggregation first.

Daily fixed-horizon summaries do **not** permit repeated decisions.
Configure [sequential inference](sequential-inference.md) before reading
outcomes.

## Report a guardrail that cannot be checked

A missing guardrail is not evidence of safety. See the runnable
[unavailable-guardrail example](../reference/result-fields.md#report-a-guardrail-that-cannot-be-checked).

## Recover the unavailable guardrail

Correct the source, rebuild and rerun:
[guardrail recovery](../reference/result-fields.md#recover-the-unavailable-guardrail).

## Read experiment definitions

For warehouse data, install `increment[demo]` for the DuckDB example and
follow [warehouse setup](data-model.md).

## Next step

- Interpret the report: [Reading results](reading-results.md).
- Reduce variance: [CUPED](cuped.md#on-the-dataframe-path).
- Plan repeated monitoring: [sequential inference](sequential-inference.md).
- Choose switchback, observational or another design:
  [Which method should I use?](choose-a-method.md).
