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

## One row per randomization unit

Your dataframe needs one row per unit, a group column, and one column per
metric. If multiple units share a randomization cluster, pass `cluster=`.
For unit-by-day data, use [`Analysis.from_unit_panel`](#one-row-per-unit-per-day).

The example below samples each unit's assignment independently with
probability 0.5, then declares that same assignment law. Revenue is the
primary metric; complaints are a guardrail. The first dataframe deliberately
has no usable treatment-arm complaint observations, simulating a source-data
gap rather than treating missing outcomes as zero.

```python
from random import Random

import polars as pl

from increment import Analysis, AnalysisPlan, MetricSpec, Randomized

rng = Random(7)
n = 500
arms = ["treatment" if rng.random() < 0.5 else "control" for _ in range(2 * n)]
revenue = [max(0.0, rng.gauss(10 + (arm == "treatment"), 4)) for arm in arms]
complaints = [int(rng.random() < 0.04) for _ in arms]
df = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(2 * n)],
        "variant": arms,
        "revenue": revenue,
        "complaints": [
            None if arm == "treatment" else value
            for arm, value in zip(arms, complaints, strict=True)
        ],
    }
)
```

## Declare the design and metrics

Always declare the control explicitly. `allocation` names the actual
assignment probabilities; `allocation_scheme="independent"` is correct here
because the generator used independent Bernoulli assignments. Do not declare
this law for blocked or otherwise constrained assignment.

```python
import warnings

from increment import Analysis, AnalysisPlan, MetricSpec, Randomized

design = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
plan = AnalysisPlan(primary="revenue", guardrails=["complaints"])
metrics = [
    MetricSpec(name="revenue", type="mean"),
    MetricSpec(
        name="complaints",
        type="conversion",
        missing="drop",
        preferred_direction="decrease",
    ),
]
with warnings.catch_warnings(record=True) as analysis_warnings:
    warnings.simplefilter("always")
    analysis = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        design=design,
        metrics=metrics,
        plan=plan,
    )
    results = analysis.run()
for captured_warning in analysis_warnings:
    print(f"Warning: {captured_warning.message}")
print(results)
```

```text
Warning: metric 'complaints': missing='drop' excludes rows with 537 null/NaN value(s) from this metric's moments (complete-case). This is unbiased only when missingness is unrelated to the values; if null means "no events", missing='zero' keeps every unit.
assigned — treatment vs control
Integrity: not_rejected — no detected mismatch; not proof of valid randomization.
Decision scope completeness: incomplete.
Population roster completeness: complete (declared_allocation).
Required cells: 2; available: 1.
Unavailable complaints: Metric complaints has no usable observations for arm treatment (arms with observations: control); check the metric's missing-value policy and the values available in that arm, then correct the data or policy and rerun
Multiplicity (primary): declared_plan; family primary.
Multiplicity (guardrail): declared_plan; family guardrail.
revenue (primary, relative lift): 0.115788, set [0.0621892, 0.172092] at 0.95 confidence.
revenue interpretation: excludes zero at 95.0% confidence; effect direction supported (increase).
revenue additive effect: 1.15169, set [0.638286, 1.6651].
revenue test alternative: two-sided; null (relative lift): 0.
revenue result identity: method unadjusted; method role decision; estimand itt.
complaints test alternative: less; null (relative lift): 0.
complaints result identity: method unadjusted; method role decision; estimand itt.
```

This is a complete-case warning: dropping rows is unbiased only when
missingness is unrelated to outcomes; use `missing="zero"` only when null
means no events.

The relative lift is about 11.6%; the additive effect is about 1.15 revenue
units. The interval describes uncertainty about the effect, not the range
containing 95% of users. `not_rejected` means this check found no mismatch;
it does not prove valid randomization. The guardrail has no usable treatment
observations, so the required decision scope is incomplete; absence is not
evidence of no increase. Both declared metrics retain their roles: revenue is
primary, and complaints is a guardrail. Only a metric outside the declared
plan is labelled `exploratory_unadjusted`.

## Recover the unavailable guardrail

Repair the source so the treatment-arm observations are present, then rerun.
Here the full generated values stand in for the corrected source; in practice,
resolve why observations were missing and choose a missing-value policy that
matches the metric's meaning.

```python
recovered_df = df.with_columns(pl.Series("complaints", complaints))
recovered = Analysis.from_unit_summary(
    recovered_df,
    unit="user_id",
    group="variant",
    design=design,
    metrics=metrics,
    plan=plan,
).run()
print(recovered)
print(recovered.to_frame(backend="polars"))
```

```text
assigned — treatment vs control
Integrity: not_rejected — no detected mismatch; not proof of valid randomization.
Decision scope completeness: complete.
Population roster completeness: complete (declared_allocation).
Required cells: 2; available: 2.
Multiplicity (primary): declared_plan; family primary.
Multiplicity (guardrail): declared_plan; family guardrail.
revenue (primary, relative lift): 0.115788, set [0.0621892, 0.172092] at 0.95 confidence.
revenue test alternative: two-sided; null (relative lift): 0.
revenue result identity: method unadjusted; method role decision; estimand itt.
complaints (guardrail, relative lift): -0.137803, set [-1, 0.666805] at 0.9 confidence.
complaints interpretation: does not reject; inconclusive, not evidence of no effect.
complaints additive effect: -0.00535734, set [-0.0249109, 0.0141962].
complaints binomial set upper_bound: [-1, 0.666805] at 0.9 confidence.
complaints test alternative: less; null (relative lift): 0.
complaints result identity: method unadjusted; method role decision; estimand itt.
```

The repaired collection is complete. The guardrail point estimate is a 13.8%
decrease, but its 90% interval includes zero, so it does not reject the
one-sided null; that is inconclusive, not proof of safety or no effect.

`repr(results)` and `print(results)` are the ordinary collection summary.
`results.to_frame()` keeps numeric result columns and qualification columns;
selecting only effect columns is a deliberate loss of evidence. For an
optional rendered table install `increment[tables]`, then keep the consumer
import distinct from authoring imports:

```python
from increment.tables import readout_table

readout_table(recovered)
```

If you only receive serialized result rows, the result/table import does not
configure an analysis. Rows alone do not carry the collection's scope, so the
table reports scope and integrity as unknown; receive the whole collection
when those qualifications matter.

```python
from increment.results import LiftEstimate
from increment.tables import estimates_to_readout, readout_table


def received_result_table(result: LiftEstimate):
    return readout_table(estimates_to_readout([result]))


received_result_table(recovered[0])
```

See [Reading results](reading-results.md) for scale, uncertainty, availability,
integrity, completeness, population and multiplicity qualifications.

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

- Reduce variance: [CUPED](cuped.md#on-the-dataframe-path).
- Monitor while the experiment runs: [sequential monitoring of a conversion
  metric](sequential-inference.md#sequential-monitoring-of-a-conversion-metric),
  or [ordinary continuous monitoring](sequential-inference.md#ordinary-continuous-monitoring)
  for mean and ratio metrics.
- Run a switchback: [supported contract](switchback.md#supported-contract).
- Estimate an opt-in or self-selected treatment: [observational
  inference](observational.md).
- Define warehouse data and experiments: [data model](data-model.md).
- Anything else: [Which experiment analysis method should I use?](choose-a-method.md)
