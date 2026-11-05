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
    df, shape="summary", unit="user_id", group="variant", control="control", metrics=["revenue"]
)
result = increment.run(experiment, reader)
print(increment.render(result, "brief"))
```

`experiment` describes the analysis and holds no data; save it with
`experiment.to_json()` and identify it with `experiment.hash()`. `reader`
reads the data for it. A dataframe reader keeps the rows the frame held when
`from_frame` built it, so later changes to `df` do not reach the run; a
warehouse reader reads live on every call, so separate calls can read changed
data. The experiment's hash and its source digests cover declarations, such as
source names, columns, metrics and design, not rows. Readers built with
identical declarations are therefore interchangeable: `increment.run(experiment, other_reader)`
runs on whatever rows `other_reader` holds, so keep each experiment with the
reader its builder returned. A reader built for a different experiment
refuses with `kernel.run.execution_unavailable`.

`increment.runnable(experiment, "revenue")` checks known incompatibilities
across every declared view without reading data. It is sound, not complete: no
refusal does not guarantee the run will succeed, because some checks need
data. `increment.run(experiment, reader)` executes all declared supported views
in one shared capture; `views=(view, ...)` selects declared views, which run in
declaration order. Every selected view is preflighted before capture. The
result keeps one run ID and each cell's original view; monitoring views do not
become whole-experiment decisions. See
[experiment specifications](experiment-specifications.md) for the selection
contract and existing source and persistence limitations.

## Report a guardrail that cannot be checked

This example samples assignment independently with probability 0.5, then
declares the same assignment law. Revenue is the primary metric; complaints
are a guardrail. Treatment-arm complaint values are missing to show how the
result reports a guardrail that cannot be checked.

```python
from random import Random
import warnings

import polars as pl

import increment
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
        metrics=[
            "revenue",
            increment.MetricSpec(
                name="complaints",
                type="conversion",
                missing="skip",
                preferred_direction="decrease",
            ),
        ],
        plan=increment.AnalysisPlan(primary="revenue", guardrails=["complaints"]),
    )
    result = increment.run(experiment, reader)
for warning in data_warnings:
    print(warning.message)
print(increment.render(result, "brief"))
```

The missing-data warning is surfaced above the rendered report, not discarded:
complaint values are missing far more often in one group, so the units that
have a value may not be comparable across groups. The complaints row names the
units it compares in its `population`, and the report keeps the guardrail
unavailable and states what the remaining evidence does and does not establish.

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
recovered_experiment, recovered_reader = increment.from_frame(
    recovered_df,
    shape="summary",
    unit="user_id",
    group="variant",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[
        "revenue",
        increment.MetricSpec(
            name="complaints",
            type="conversion",
            missing="skip",
            preferred_direction="decrease",
        ),
    ],
    plan=increment.AnalysisPlan(primary="revenue", guardrails=["complaints"]),
)
recovered = increment.run(recovered_experiment, recovered_reader)
print(increment.render(recovered, "brief"))
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

import increment

demo_rows = []
for i, revenue in enumerate([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]):
    for event in ["page_view", "purchase"]:
        demo_rows.append(
            {
                "user_id": f"u{i}",
                "group_id": "control" if i < 4 else "treatment",
                "experiment_id": "demo",
                "ts": dt.datetime(2025, 1, 1, 12 if event == "page_view" else 13),
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
    experiment, reader = increment.from_definitions(
        definitions_directory, demo_con, experiment_name="demo"
    )
    with reader:
        result = increment.run(experiment, reader)
    print(increment.render(result, "brief"))
```

The reader reads the warehouse through `demo_con` when `run` needs data.
Leaving the `with` block releases any temporary tables it created; a closed
reader refuses every read, and closing it never closes `demo_con`. Building
with a connection also writes a catalog, `.increment-kernel-catalog.json`, and
its `.lock` file into the definitions directory. See
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
`increment.workflows.unit_counts(experiment, reader)` reports the excluded
units under `"(unassigned)"`, and the sample-ratio mismatch (SRM) check reports
them separately. They never become a treatment group.

### Missing metric values

A null or NaN needs a meaning before it can be included in an average:

| Setting | Use when |
|---|---|
| `MetricSpec(missing="zero")` | Missing means no events. |
| `MetricSpec(missing="skip")` | Missing means not observed. |

!!! warning "Skipping missing values can bias the result"
    The metric uses only the units with a value; they stay in every other
    metric, and the whole-window row's `population` states how many units each
    group observed (a breakout row's says the segment's counts were not
    recorded). This is unbiased only when missingness is unrelated to the
    values. When the share of missing values differs between groups, the
    `frame.validation.metric_missing_differs_by_arm` warning reports it.
    On clustered designs the check is not run; compare missingness across
    clusters yourself.

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
`missing="zero"` or `"skip"` for an outcome. Set a CUPED covariate's policy on
its method-owned declaration, for example
`Method.cuped(covariates=CupedCovariate(name="baseline", kind="continuous", missing="impute"))`.

## One row per unit per day

Use `from_frame(..., shape="panel")` for one row per unit per day. The
result combines each unit's values across days before estimating the
experiment-wide effect:

```python
import datetime as dt

import increment

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
experiment, reader = increment.from_frame(
    panel,
    shape="panel",
    unit="user_id",
    group="variant",
    date="day",
    design=increment.Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics=[increment.MetricSpec(name="revenue", type="mean")],
    plan=increment.AnalysisPlan(primary="revenue"),
)
print(increment.render(increment.run(experiment, reader), "brief"))
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
