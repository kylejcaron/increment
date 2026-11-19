# Analyze warehouse data

<span id="warehouse-native-experimentation-data-model"></span>
<span id="running-it"></span>

Declare experiments and metrics once, in version-controlled YAML, and run them
against warehouse data. Use this path when your events, enrollment records,
and metric definitions need to stay in the warehouse.

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
                "ts": dt.datetime(2025, 1, 1, 12, 1 if event == "purchase" else 0),
                "event": event,
                "revenue": revenue if event == "purchase" else None,
            }
        )
demo_rows.append(
    {
        "user_id": "background",
        "group_id": None,
        "experiment_id": None,
        "ts": dt.datetime(2025, 1, 3, 12),
        "event": "purchase",
        "revenue": 1.0,
    }
)
demo_con = ibis.duckdb.connect()
demo_con.create_table("demo_events", pl.DataFrame(demo_rows).to_arrow())

with TemporaryDirectory() as definitions_directory:
    Path(definitions_directory, "demo.yaml").write_text("""\
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
""")
    experiment, reader = increment.from_definitions(
        definitions_directory, demo_con, experiment_name="demo"
    )
    demo_results = increment.run(experiment, reader).readout()
for row in demo_results:
    print(row.metric, row.group_id, row.lift)
```

This self-contained example uses the `increment[demo]` extra, which supplies a
compatible Ibis/DuckDB combination, plus `polars` to create a small event table.

Purchases occur one minute after enrollment. Outcome windows exclude events
at the exact enrollment timestamp.

Increment builds the unit-by-day panel and exposes every query it runs. If your
data is already one row per unit or per unit-day, use
`from_frame(..., shape="summary")` or `shape="panel"` instead
([Quickstart](quickstart.md)).

Building with a connection writes a catalog of the bound experiments,
`.increment-kernel-catalog.json`, and its `.lock` file into the definitions
directory; `increment.from_definitions(definitions_directory)` reads that
catalog to inspect an experiment without a connection. The reader reads the
warehouse through `demo_con`; closing it, or leaving its `with` block, never
closes `demo_con`, and a closed reader refuses every read.

The [warehouse example](../examples/analysis_from_a_warehouse.md) walks through
the full flow; [Binding a normalized warehouse](#binding-a-normalized-warehouse)
shows how to join dimension tables.

!!! warning "Check data maturity before interpreting results"
    Missing or late-arriving facts can leave a unit's measurement window
    incomplete. Increment censors unfinished windows rather than scoring
    partial data. A missing estimate can also have a row-specific reason; see
    [Reading results](reading-results.md). Assignment audit buckets are not
    analysis arms; see [Multiplicity](multiplicity.md) for populations and
    statuses.

## Metrics

Install `increment[metric-definitions]` alongside `increment[warehouse]` to
import portable metric definitions with `increment.from_definitions`. Add a
`metric_definitions:` block beside native sections such as `metrics:`.
Here is the checkout demo's binding:

```yaml
metric_definitions:
  file: shop.metrics.yaml
  use: [average_order_value]
  sources: {orders: orders}
  measures: {revenue: order, order_count: order}
  metrics:
    average_order_value:
      entity: user
      preferred_direction: increase
      window_days: null
```

`file` is relative to this YAML file and must stay inside the loaded
definitions directory. `use` names the shared metrics to import. Refer to
those names in the experiment plan exactly as you would refer to a native
metric. Do not also declare an imported name in `metrics`.

For each selected measure, `sources` maps the shared source name to a native
fact source, and `measures` maps the measure name to an event in that source.
The demo maps both operands to the same `order` event on the same
completed-order rows; its existing native SQL still excludes other orders.
Bind sources that represent the same rows as the shared table or view. Native
SQL must expose the shared row key, entity column, event time and any value or
filter columns. In `metrics`, `entity` names the shared entity and must map
to the experiment's native unit. Keep enrollment, methods, windows and other
experiment settings in the native YAML.

Increment supports ratios of two named sums or row counts, means over a
declared eligible roster, and conversions that ask whether a unit had at least
one qualifying event. A ratio divides the arm's total numerator by its total
denominator, not the average of each unit's ratio. Filters are applied to each
operand separately. Bind a filtered event's string, boolean or numeric
dimension to a native property with the same column and type; filters requiring
a shared join are not supported. Roster-based metrics require the roster to
contain exactly the assigned units, with no duplicate or null keys; they cannot
be read against a separate trigger population.

`MetricTrends` reads native metric definitions only. To read an imported metric,
use `increment.from_definitions` for experiments or Terrine or Malloy for
descriptive reports.

Experiment reads of shared metrics require a warehouse snapshot so the row keys
and eligible roster are checked against the same rows used for the result.
Use the default captured reads, not `coherence="live"`.
Offline `runnable()` also flags a saved live request for a shared metric.

Use other formulas in descriptive reports with Terrine or Malloy, not experiment inference.
An unselected formula does not have to be supported by Increment, but the whole
shared file must still be valid. Changing a selected metric or its dependencies
requires rebuilding the offline catalog with a connection. Comments, formatting,
and unrelated valid formulas do not. Other experiment builders accept their
usual prepared inputs; they do not read shared YAML.

## One row per unit per day

Each unit's clock starts at its first qualifying exposure inside the experiment
window. That exposure fixes the unit's arm and its day 0, and every metric
window counts days from there, so units enrolled on different dates get the
same-length window. Dates follow the experiment's `day_boundary` (default
`"UTC"`); see [day-boundary rules](metric-types.md#how-days-are-counted-day-0-and-the-day-boundary).

| Column | Meaning |
|---|---|
| `unit_id` | Analysis unit |
| `group_id` | Assigned arm |
| `first_exposure_ts` / `first_exposure_date` | Start of the unit's clock |
| `day_offset` | Days since exposure, starting at 0 |
| `ds` | Local calendar date, `first_exposure_date + day_offset` |

Two rules hold on every path: an eligible unit with no events on a day
contributes zero, not a missing value, and a unit whose window has not finished
is censored, not scored from partial data.

**Warehouse.** The panel is dense through the observation horizon: `observation_end`,
else `end`, else the latest observed event date for a running experiment.
Each fact also carries `data_as_of`, the latest date actually loaded. A
window ending after that bound is censored; a warning reports material
drops. A fact with no loaded rows has no such date, so it bounds nothing and
every enrolled unit scores zero on it; the run warns with
`source.native.fact_no_rows`, listing up to five event labels the fact source
does carry, so a fact name that matches no label is visible.
An outcome event counts only when its timestamp is strictly after the unit's
`first_exposure_ts`, and the CUPED window ends just before that instant. An
event logged at the exact exposure timestamp, such as the order that enrolls
a unit under a first-purchase exposure or a line item logged with it, therefore
counts in neither the outcome nor the CUPED window. Encouragement uptake is the
one exception: uptake at the exposure instant qualifies.

**Dataframe.** A sparse unit panel is not a raw event log: every assigned unit
must appear at least once with its assignment and exposure identity. Missing
unit-days for a represented unit are treated as zero, but zero filling cannot
invent a wholly absent unit or recover its assignment. Start with the assignment
table, aggregate activity to one row per unit/day, then left-join activity so
inactive assigned units remain as zero-valued anchors.
`from_frame(..., shape="panel")` accepts sparse input and fills absent eligible
unit-days with zero. A unit becomes eligible on its `exposure_date`. Without
that column, the rows cannot distinguish days before exposure from days with no
activity, so only whole-window results of unwindowed metrics are available.
Daily and as-of views, windowed and retention metrics, a windowed uptake,
as-of compliance and sequential inference need `exposure_date=`; if every
unit was exposed on the same day, add a column holding that date. `observation_end=`
names the last observed day in the date column's own form: a date or an ISO-8601
string such as `"2025-01-31"` for a calendar column, a number for a numeric day
index, or a label spelled like the column's own for a string column (`"d9"` after
`"d1"` to `"d8"`, though no row is dated `"d9"`). Values dated after it enter no
result, units exposed after it are not analysed, and a value that names no day of
the column's kind, or one before its first day, refuses with
`source.panel.observation_end` when the panel is built. Small inputs may be
densified temporarily; larger ones use a sparse day-at-a-time
reduction. `densified_cells` counts the cells that needed zero fill inside each
unit's eligible days, from its exposure date (or the first panel day) through the
observation end (or the last panel day), not physical rows.

For example, if assignments contain 100 units per arm but activity contains
only 50 control and 60 treatment units, joining activity to assignments retains
the absent units as zero-valued panel rows:

Because every active unit has value 10, the activity-only means are
descriptive and constant arms may not have a finite uncertainty interval.
The actual inference below runs on the complete panel.

```python
from math import isclose

import polars as pl

import increment

control_ids = [f"c{i}" for i in range(100)]
treatment_ids = [f"t{i}" for i in range(100)]
assignments = pl.DataFrame(
    {
        "user_id": control_ids + treatment_ids,
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "exposure_date": ["2025-01-01"] * 200,
    }
)
events = pl.DataFrame(
    {
        "user_id": control_ids[:50] + treatment_ids[:60],
        "date": ["2025-01-01"] * 110,
        "spend": [10.0] * 110,
    }
)
activity = events.group_by("user_id", "date").agg(pl.col("spend").sum())
assert activity.select(pl.struct("user_id", "date").n_unique()).item() == activity.height

activity_only = activity.join(assignments, on="user_id", how="inner")
complete = assignments.join(activity, on="user_id", how="left").with_columns(
    pl.col("date").fill_null("2025-01-01"),
    pl.col("spend").fill_null(0.0),
)


def analyze(panel):
    experiment, reader = increment.from_frame(
        panel,
        shape="panel",
        unit="user_id",
        group="variant",
        control="control",
        date="date",
        exposure_date="exposure_date",
        observation_end="2025-01-01",
        metrics={"spend": "mean"},
    )
    return next(iter(increment.run(experiment, reader).readout()))


descriptive = activity_only.group_by("variant").agg(
    pl.col("user_id").n_unique().alias("n"),
    pl.col("spend").mean().alias("mean"),
)
control = descriptive.filter(pl.col("variant") == "control").row(0, named=True)
treatment = descriptive.filter(pl.col("variant") == "treatment").row(0, named=True)
assert (control["n"], treatment["n"]) == (50, 60)
assert isclose(control["mean"], 10.0, rel_tol=1e-12, abs_tol=1e-12)
assert isclose(treatment["mean"], 10.0, rel_tol=1e-12, abs_tol=1e-12)

result = analyze(complete)
assert (result.n_control, result.n_treat) == (100, 100)
assert isclose(result.mean_control, 5.0, rel_tol=1e-12, abs_tol=1e-12)
assert isclose(result.mean_treat, 6.0, rel_tol=1e-12, abs_tol=1e-12)
assert isclose(result.require_lift().value, 0.2, rel_tol=1e-12, abs_tol=1e-12)
```

The activity-only means describe observed units only; they do not account for
assigned units absent from the activity source. The complete panel retains all
assignments, including zero-activity anchors.

??? info "Dataframe zero-fill route and memory limits"
    For small inputs, zero filling may use a temporary date spine when a scratch
    estimate based on the number of cells, value columns and identity-buffer size
    falls below the 64 MiB selection threshold. This is a route-selection
    heuristic, not a guarantee on peak memory: dataframe backend overhead and
    temporary operations are not fully captured. Larger estimates select a sparse
    day-at-a-time reduction that avoids retaining the full unit-by-date product.
    `densified_cells` counts observed or implicit cells that require zero filling,
    not physical rows written or retained.

    Metric windows (`window_days`, `threshold_days`) count in
    these offsets, so every unit gets the same-length observation window
    regardless of when it enrolled.

Units with conflicting or missing assignments are reported separately and never
analyzed as arms; the default refuses them.

??? info "Assignment diagnostics"
    Assignment integrity is checked before each warehouse readout. A unit seen in
    more than one real arm is counted as `(mixed assignment)`; a unit seen only
    with NULL, or with both a real arm and NULL, is counted as `(unassigned)`.
    These are accounting entries, not arms: NULL never receives an estimate or an
    SRM degree of freedom. The `on_mixed_assignment` policy governs both
    conditions: the default `"error"` refuses, while `"warn"` and `"exclude"`
    remove them and report each count separately in diagnostics.


## Definition details

The example uses one file. Larger projects can split declarations across YAML files without changing how they run.

- <span id="the-definitions-directory"></span>[The definitions directory](../reference/warehouse-definitions.md#the-definitions-directory).
- <span id="fact_sourcesyaml-where-events-live"></span>[fact_sources.yaml: where events live](../reference/warehouse-definitions.md#fact_sourcesyaml-where-events-live).
- <span id="exposuresyaml-what-enrolls-a-unit"></span>[exposures.yaml: what enrolls a unit](../reference/warehouse-definitions.md#exposuresyaml-what-enrolls-a-unit).
- <span id="metricsyaml-what-to-measure"></span>[metrics.yaml: what to measure](../reference/warehouse-definitions.md#metricsyaml-what-to-measure).
- <span id="experimentsyaml-tying-it-together"></span>[experiments.yaml: tying it together](../reference/warehouse-definitions.md#experimentsyaml-tying-it-together).
- <span id="window-edges-and-the-day-boundary"></span>[Window edges and the day boundary](../reference/warehouse-definitions.md#window-edges-and-the-day-boundary).
- <span id="per-metric-method-and-prior-bindings"></span>[Per-metric method and prior bindings](../reference/warehouse-definitions.md#per-metric-method-and-prior-bindings).
- <span id="randomization-grain-vs-analysis-grain"></span>[Randomization grain vs analysis grain](../reference/warehouse-definitions.md#randomization-grain-vs-analysis-grain).

- <span id="binding-a-normalized-warehouse"></span>[Normalized warehouses](../reference/warehouse-definitions.md#binding-a-normalized-warehouse):
    - <span id="fact-sources-are-queries"></span>[SQL joins](../reference/warehouse-definitions.md#fact-sources-are-queries).
    - <span id="four-rules-that-bite"></span>[Four join pitfalls](../reference/warehouse-definitions.md#four-rules-that-bite).
    - <span id="declarative-dim-sources"></span>[Declarative dimensions](../reference/warehouse-definitions.md#declarative-dim-sources).
    - <span id="end-to-end"></span>[Complete example](../reference/warehouse-definitions.md#end-to-end).

Use the reference’s [complete normalized example](../reference/warehouse-definitions.md#end-to-end) to try those joins. Generated dimension joins keep unmatched fact rows; handwritten inner joins do not. Check row counts when migrating. Pre-exposure attributes need events before exposure, not just a dimension table with historical values.

To inspect a calculation, use `increment.query.panel_sql(experiment, reader)` or `summary_sql(experiment, reader)`. They return SQL, not effect estimates. Under the default fact-NULL policy, constructing a preview can read NULL counts. [SQL preview details](../reference/warehouse-definitions.md#inspect-generated-sql) explain the query stages and lifetime.

## Reuse an analysis or its data

[Save an experiment specification](experiment-specifications.md) to reuse settings. It does not save data. Publish an immutable unit-day artifact to reuse warehouse data without rebuilding facts and exposures, or export moments when reduced summaries are enough.

- <span id="versioned-assignment-evidence"></span>[Versioned assignment evidence](../reference/artifacts-and-replay.md#versioned-assignment-evidence).
- <span id="assignment-evidence-and-replay"></span>[Assignment evidence and replay](../reference/artifacts-and-replay.md#assignment-evidence-and-replay).
- <span id="canonical-unit-day-artifacts"></span>[Canonical unit-day artifacts](../reference/artifacts-and-replay.md#canonical-unit-day-artifacts).
- <span id="operation-and-extension-matrix"></span>[Operation and extension matrix](../reference/artifacts-and-replay.md#operation-and-extension-matrix).
- <span id="trust-namespace-and-read-consistency"></span>[Trust, namespace, and read consistency](../reference/artifacts-and-replay.md#trust-namespace-and-read-consistency).
- <span id="moments-wire-migration"></span>[Moments wire migration](../reference/artifacts-and-replay.md#moments-wire-migration).

Never adopt a generation from a failed publication. Follow the [recovery instructions](../reference/artifacts-and-replay.md#recovering-from-a-failed-publication); do not erase a relation unless its ownership is established. Timestamps alone do not prove a feed complete.

## Other warehouses

Use the same `from_definitions` call with any SQL backend Ibis supports. DuckDB, PostgreSQL, Snowflake and BigQuery have execution verification. Other backends are unverified; compare their results with a verified backend or dataframe before relying on them. Warehouse artifact verification is available only on those four verified backends.

<span id="temp-tables"></span>

The default run needs warehouse TEMP tables for its captured data. If only source capture requires them, `run(experiment, reader, coherence="live")` can read live instead, including with scalar CUPED. Live reads give up a shared captured copy. Multicovariate CUPED fitting and categorical-cell acquisition still require TEMP tables; use `from_frame` when the warehouse cannot create them.

Backend-specific details:

- <span id="the-unverified-warehouse-notice"></span>[Unverified-backend notice](../reference/warehouse-definitions.md#the-unverified-warehouse-notice).
- <span id="when-ibis-cannot-translate-an-operation"></span>[Missing Ibis translations](../reference/warehouse-definitions.md#when-ibis-cannot-translate-an-operation). Earlier queries may have read data when a translation fails; no result is produced.
- <span id="event-times-with-a-time-zone"></span>[Time-zone conversion](../reference/warehouse-definitions.md#event-times-with-a-time-zone).

## Next step

Choose an estimator with [Which experiment analysis method should I use?](choose-a-method.md), or continue to the [warehouse analysis example](../examples/analysis_from_a_warehouse.md).
