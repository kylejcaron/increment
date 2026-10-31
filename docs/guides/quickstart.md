# A/B testing in Python: quickstart

Analyze a randomized experiment from a dataframe. Start with one row per
user, store, or other analysis unit. The dataframe examples need no YAML,
warehouse connection, or DuckDB installation.

```bash
pip install increment polars
```

The examples use Polars. Any eager dataframe supported by
[Narwhals](https://narwhals-dev.github.io/narwhals/) works, including pandas
and PyArrow.

## Build and run an experiment

Your dataframe needs one row per unit, a group column, and one column per
metric. If multiple units share a randomization cluster, pass `cluster=`.
For unit-by-day data, use the panel shape described below.

This example samples assignment independently with probability 0.5, then
declares the same assignment law. Revenue is the primary metric; complaints
are a guardrail. Treatment-arm complaint values are missing to show how the
result reports a guardrail that cannot be checked.

```python
from random import Random
import warnings

import polars as pl

from increment import AnalysisPlan, MetricSpec, Randomized, from_frame, render, run, runnable
from increment.errors import IncrementWarning


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

with warnings.catch_warnings(record=True) as data_warnings:
    warnings.simplefilter("always", IncrementWarning)
    exp, resolver = from_frame(
        df,
        shape="summary",
        unit="user_id",
        group="variant",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=[
            "revenue",
            MetricSpec(
                name="complaints",
                type="conversion",
                missing="drop",
                preferred_direction="decrease",
            ),
        ],
        plan=AnalysisPlan(primary="revenue", guardrails=["complaints"]),
    )
    runnable(exp, "revenue")  # Static check only; it does not read the dataframe.
    result = run(exp, resolver)
for warning in data_warnings:
    print(warning.message)
print(render(result, "brief"))
```

The missing-data warning is surfaced above the rendered report, not discarded:
it explains that the selected missing-value policy changes the metric's
population. The report separately keeps the guardrail unavailable and states
what the remaining evidence does and does not establish.

`exp` describes the experiment and can be saved with `exp.to_json()` or
identified with `exp.hash()`. It does not hold a live dataframe. `resolver`
binds the dataframe when `run` reads evidence; if its contents no longer match
the declared source, the run is refused. `runnable` checks known
incompatibilities across every declared view without reading data. It is
sound, not complete: no refusal does not guarantee the run will succeed,
because some checks need data. `run` currently accepts one declared view at a
time; separate calls can read changed data.

The rendered result keeps unavailable guardrails, checks that were not
completed, and limits on what the data can establish visible. A missing
guardrail is not evidence of safety or no effect. `not_rejected` means no
mismatch was detected; it does not prove the assignment was valid.

## Recover the unavailable guardrail

Repair the source so treatment-arm observations are present, then rebuild and
run the experiment. Here the generated values stand in for the corrected
source; in practice, resolve why observations were missing and choose a
missing-value policy that matches the metric's meaning.

```python
recovered_df = df.with_columns(pl.Series("complaints", complaints))
recovered_exp, recovered_resolver = from_frame(
    recovered_df,
    shape="summary",
    unit="user_id",
    group="variant",
    design=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[
        "revenue",
        MetricSpec(
            name="complaints",
            type="conversion",
            missing="drop",
            preferred_direction="decrease",
        ),
    ],
    plan=AnalysisPlan(primary="revenue", guardrails=["complaints"]),
)
recovered = run(recovered_exp, recovered_resolver)
print(render(recovered, "brief"))
```

The repaired result has a complaint estimate, but its uncertainty still
matters: an interval including zero is inconclusive, not proof of safety or
no effect. Keep the rendered qualifications when sharing the result.

## Read experiment definitions

This complete example creates a small event table and definition file in a
temporary directory. Replace them with your warehouse connection and
version-controlled definitions when applying the same workflow:

Install the optional `demo` extra for the DuckDB backend used in this definitions example:

```bash
pip install "increment[demo]" polars
```

```python
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import ibis
import polars as pl

from increment import from_definitions, render, run

demo_rows = []
for i, revenue in enumerate([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]):
    for event in ["page_view", "purchase"]:
        demo_rows.append(
            {
                "user_id": f"u{i}",
                "group_id": "control" if i < 4 else "treatment",
                "experiment_id": "demo",
                "ts": dt.datetime(2025, 1, 1, 12),
                "event": event,
                "revenue": revenue if event == "purchase" else None,
            }
        )
demo_con = ibis.duckdb.connect()
demo_con.create_table("demo_events", pl.DataFrame(demo_rows).to_arrow())

with TemporaryDirectory() as definitions_directory:
    Path(definitions_directory, "demo.yaml").write_text(
        """\
dialect: duckdb
fact_sources:
  - name: demo_events
    sql: SELECT * FROM demo_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {name: page_view, column: null}
      - {name: purchase, column: revenue}
exposures:
  - {name: enrolled, fact: page_view}
metrics:
  - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 1}
experiments:
  - name: demo
    exposure: enrolled
    unit: user_id
    start: 2025-01-01
    end: 2025-01-01
    observation_end: 2025-01-02
    control_group: control
    plan: {primary: revenue}
""",
        encoding="utf-8",
    )
    exp, resolver = from_definitions("demo", definitions_directory, demo_con)
    result = run(exp, resolver)
    print(render(result, "brief"))
```

The warehouse connection is bound at run time by the resolver. See
[The data model](data-model.md) for definitions, query inspection, and
portable unit-day inputs.

See [Reading results](reading-results.md) for scale, uncertainty, availability,
integrity, completeness, population and multiplicity qualifications.
## Group identities

Nonmissing group values are rendered as strings consistently across
dataframe libraries. Boolean columns use the canonical labels `"true"` and
`"false"`; on a Boolean column, `control` accepts `True`, `"True"`, or
`"true"` for the true arm (and the corresponding false spellings). Other
column types keep their string labels, so a string column containing
`"True"` and `"False"` retains those exact identities. For example, an
integer group column containing `0` and `1` accepts `control=0` or
`control="0"`.

For `from_moments` reloads of a Boolean-group export, use a Python Boolean
control (`True` or `False`) or the exact canonical string label (`"true"` or
`"false"`). Moment-cube string controls remain case-sensitive; `"True"` does
not alias the exported `"true"` group.

Labels must remain distinct after conversion. Values such as `1` and
`"1"` in a pandas object column raise `frame.validation.group_label_collision`
rather than merge groups. Rename them to distinct strings before analysis.

`"(unassigned)"` and `"(mixed assignment)"` are reserved for accounting.
Using either as a group raises `frame.validation.group_label_reserved`.
Missing labels stay missing; the literal string `"nan"` is a valid group.
Saved sequential state keyed by pandas Boolean labels (`"True"`/`"False"`)
cannot be continued under the canonical labels: retain it as a replay-only
record, or register a new monitoring protocol rather than relabelling the
checkpoint.

## Missing data

Both constructors reject missing group labels and metric values by default.
Choose how to handle them explicitly.

### Missing group labels

Pass `on_unassigned="exclude"` to exclude rows with no assignment.
`analysis.unit_counts()` reports the excluded units under `"(unassigned)"`, and
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

You can also prepare the dataframe with `increment.estimators.impute.zeros` or
`drop_null`. Each returns `(frame, affected_count)` so you can record how many
values or rows changed. `zeros` assumes a missing value means no events, and
`drop_null` removes whole rows, which is valid only when missingness is unrelated
to the values.

!!! warning "Do not fill missing outcomes with their mean"
    `increment.estimators.impute.pooled_mean` is for appropriate pre-period covariates,
    such as a CUPED baseline, not outcomes. Filling missing outcomes with their
    mean shrinks variance and can bias estimates.

Declare the role of each column you pass to `pooled_mean`:

```python
import pyarrow as pa

from increment.estimators import impute

units = pa.table({"baseline": [4.0, None, 8.0]})
units, n_filled = impute.pooled_mean(units, "baseline", roles={"baseline": "covariate"})
print(n_filled, units["baseline"].to_pylist())
```

A column declared `"outcome"` is refused (`impute.pooled_mean_outcome`) before
any fill. Filling a column with no declared role still works but emits the
`impute.pooled_mean_role_undeclared` warning with per-column counts.
`MetricSpec(missing="impute")` is refused (`frame.metric.missing_impute`); use
`missing="zero"` or `"drop"` for an outcome. Set a CUPED covariate's policy on
its method-owned declaration, for example
`Method.cuped(covariates=CupedCovariate(name="baseline", kind="continuous", missing="impute"))`.

## One row per unit per day

Use `from_frame(..., shape="panel")` for one row per unit per day. The
result combines each unit's values across days before estimating the
experiment-wide effect:

```python
import datetime as dt

from increment import AnalysisPlan, MetricSpec, Randomized, from_frame, render, run

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
exp, resolver = from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="day",
    design=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[MetricSpec(name="revenue", type="mean")],
    plan=AnalysisPlan(primary="revenue"),
)
print(render(run(exp, resolver), "brief"))
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
