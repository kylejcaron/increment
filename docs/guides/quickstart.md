# A/B testing in Python: quickstart

Analyze a randomized experiment from a dataframe. Start with one row per
user, store, or other analysis unit. No YAML, warehouse connection, or
DuckDB installation is needed.

```bash
pip install increment polars
```

The examples use Polars. Any eager dataframe supported by
[Narwhals](https://narwhals-dev.github.io/narwhals/) works, including
pandas and PyArrow.

## One row per analysis unit

Your dataframe needs:

- One row per analysis unit.
- A column identifying the control or treatment group.
- One column per metric.

If several units share a randomization cluster, pass its column as
`cluster=`. For daily observations, use
[`Analysis.from_unit_panel`](#one-row-per-unit-per-day) instead.

This example creates synthetic data for 500 users per group:

```python
import random

import polars as pl

random.seed(7)
n = 500
df = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "revenue": [max(0.0, random.gauss(10, 4)) for _ in range(n)]
        + [max(0.0, random.gauss(11, 4)) for _ in range(n)],
        "converted": [int(random.random() < 0.30) for _ in range(n)]
        + [int(random.random() < 0.34) for _ in range(n)],
    }
)
```

## Run the analysis

Pass the dataframe and column names to `Analysis.from_unit_summary`.
The `{"column": "type"}` shorthand handles common metrics. For ratios
or CUPED covariates, use `MetricSpec` objects; see
[Metric types](metric-types.md).

Always set `control=`. Increment does not infer the control group from
the order of its labels.

```python
from increment import Analysis

analysis = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
)
results = analysis.run()

for r in results:
    if r.lift is None or r.lift.lb is None or r.lift.ub is None:
        # Missing numeric bounds: print a summary without formatting them.
        print(repr(r))
        continue
    print(
        f"{r.metric} / {r.group_id}: "
        f"lift={r.lift.value:+.2%} [{r.lift.lb:+.2%}, {r.lift.ub:+.2%}] "
        f"significant={r.lift.excludes(0.0)}"
    )
```

```text
revenue / treatment: lift=+11.33% [+6.27%, +16.64%] significant=True
converted / treatment: lift=+12.90% [-5.44%, +34.80%] significant=False
```

## Reading a `LiftEstimate`

`run()` returns a list of `LiftEstimate` objects, one per metric and
treatment group in this example:

| Field | Meaning |
|---|---|
| `r.metric` | Metric name. |
| `r.group_id` | Treatment group compared with control. |
| `r.lift.value` | Relative lift: `0.05` means 5% above control. |
| `r.lift.lb`, `r.lift.ub` | Confidence interval bounds; 95% by default. |

For the default two-sided test, `r.lift.excludes(0.0)` checks whether the
interval excludes zero. A row can have no `r.lift`, as the guard in the loop
above handles; see [Reading results](reading-results.md) for missing points,
open and sequential intervals, switchback contrasts, and flat readout rows.
A row summary does not enumerate a retained sequential set; inspect
`r.sequential_result.bounds` for its `status` and `components`.

The readout's `stat_sig` column also handles one-sided tests and nonzero
null values. For other decision statistics, see `r.chance_to_beat()`,
`r.prob_beyond(...)`, `r.risk_if_shipped()`, and `r.p_value()` in the
[API reference](../api.md).

## Results as a dataframe

Convert the results to a dataframe in your preferred library:

```python
table = results.to_frame(backend="polars")
print(table.select("metric", "group_id", "lift", "lb", "ub"))
```

For a plain list assembled separately, such as by a list comprehension,
use `to_frame(estimates, model=LiftEstimate, backend=...)`.

## Group identities

Both dataframe constructors convert nonmissing group labels to strings
using the dataframe library's representation. An integer group column
containing `0` and `1`, for example, accepts `control="0"`.

Labels must remain distinct after conversion. Values such as `1` and
`"1"` in a pandas object column raise `frame.validation.group_label_collision`
rather than merge groups. Rename them to distinct strings before analysis.

`"(unassigned)"` and `"(mixed assignment)"` are reserved for accounting.
Using either as a group raises `frame.validation.group_label_reserved`.
Missing labels stay missing; the literal string `"nan"` is a valid group.

## Missing data

Both constructors reject missing group labels and metric values by default.
Choose how to handle them explicitly.

### Missing group labels

Pass `on_unassigned="exclude"` to exclude rows with no assignment.
`unit_counts()` reports the excluded units under `"(unassigned)"`, and
the sample-ratio mismatch (SRM) check reports them separately. They never
become a treatment group.

### Missing metric values

A null or NaN needs a meaning before it can be included in an average:

| Setting | Use when |
|---|---|
| `MetricSpec(missing="zero")` | Missing means no events. |
| `MetricSpec(missing="drop")` | Missing means not observed. |

!!! warning "Dropping missing values can bias the result"
    Dropping values uses only complete cases. This is unbiased only when
    missingness is unrelated to the values.

You can also prepare the dataframe with `increment.impute.zeros` or
`drop_null`. Each returns `(frame, affected_count)` so you can record how many
values or rows changed. `zeros` assumes a missing value means no events, and
`drop_null` removes whole rows, which is valid only when missingness is unrelated
to the values.

!!! warning "Do not fill missing outcomes with their mean"
    `increment.impute.pooled_mean` is for appropriate pre-period covariates,
    such as a CUPED baseline, not outcomes. Filling missing outcomes with their
    mean shrinks variance and can bias estimates.

Declare the role of each column you pass to `pooled_mean`:

```python
import pyarrow as pa

from increment import impute

units = pa.table({"baseline": [4.0, None, 8.0]})
units, n_filled = impute.pooled_mean(units, "baseline", roles={"baseline": "covariate"})
print(n_filled, units["baseline"].to_pylist())
```

A column declared `"outcome"` is refused (`impute.pooled_mean_outcome`) before
any fill. Filling a column with no declared role still works but emits the
`impute.pooled_mean_role_undeclared` warning with per-column counts.
`MetricSpec(missing="impute")` is refused (`frame.metric.missing_impute`); use
`missing="zero"` or `"drop"` for an outcome, or `covariate_missing="impute"` for
a covariate.

## One row per unit per day

Use `Analysis.from_unit_panel` for one row per unit per day.
`run()` sums each unit's values across days and estimates lift for the
whole panel. `run_daily()` returns daily group summaries:

```python
import datetime as dt

panel = pl.DataFrame(
    {
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u4"],
        "variant": ["control", "control", "control", "control", "treatment", "treatment"],
        "day": [
            dt.date(2026, 2, 1),
            dt.date(2026, 2, 2),
            dt.date(2026, 2, 1),
            dt.date(2026, 2, 3),
            dt.date(2026, 2, 1),
            dt.date(2026, 2, 2),
        ],
        "revenue": [3.0, 0.0, 5.0, 2.5, 6.0, 4.0],
    }
)

panel_analysis = Analysis.from_unit_panel(
    panel,
    unit="user_id",
    group="variant",
    date="day",
    control="control",
    metrics={"revenue": "mean"},
)
panel_analysis.run()  # whole-panel lift, per-unit totals
panel_analysis.run_daily()  # per-day group summaries
```

!!! warning "Daily indicators become counts"
    Summing a daily 0/1 indicator counts days with an event, not whether
    the event ever occurred. Prepare the metric's intended aggregation
    before analysis.

This is the same unit-by-day structure used by the warehouse path; see
[The data model](data-model.md).

!!! warning "Daily summaries are not sequential inference"
    To make decisions while monitoring an experiment, configure
    [sequential inference](sequential-inference.md) before reading outcomes.
    Repeated fixed-horizon intervals are not a substitute.

??? note "Registered sequential inference details"
    Declare the sampling model, priors, retained roster, and joint reveal
    contract before reading outcomes.
    `InferenceSpec(kind="always_valid", registration=registration)` uses
    raw finalized checkpoints, not a standard-error-only cube.
    Current or explicitly frozen e-values select the full registered family;
    selected intervals invert the same stopped likelihood at
    `min(q * R / m, nominal alpha)`. Unsupported quantiles and adjusted scores
    retain their fixed-horizon routes.

## Next step

Read [Which experiment analysis method should I use?](choose-a-method.md)
to choose a method for your design and data.
