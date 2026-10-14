# Warehouse-native experimentation data model

Declare experiments and metrics once, in version-controlled YAML, and run them
against warehouse data. Use this path when your events, enrollment records,
and metric definitions need to stay in the warehouse.

```python
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import ibis
import polars as pl

from increment import Analysis

demo_rows = []
for i, revenue in enumerate([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]):
    for event in ["page_view", "purchase"]:
        demo_rows.append({
            "user_id": f"u{i}",
            "group_id": "control" if i < 4 else "treatment",
            "experiment_id": "demo",
            "ts": dt.datetime(2025, 1, 1, 12),
            "event": event,
            "revenue": revenue if event == "purchase" else None,
        })
demo_rows.append({
    "user_id": "background", "group_id": None, "experiment_id": None,
    "ts": dt.datetime(2025, 1, 3, 12), "event": "purchase", "revenue": 1.0,
})
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
    demo_results = Analysis.from_definitions(
        "demo", definitions_directory, demo_con
    ).run()
for row in demo_results:
    print(row.metric, row.group_id, row.lift)
```

This self-contained example uses `ibis-framework[duckdb]` and `polars` to
create a small event table. Increment builds the unit-by-day panel and exposes
every query it runs. If your data is already one row per unit or per unit-day,
use `Analysis.from_unit_summary` / `from_unit_panel` instead
([Quickstart](quickstart.md)).

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
drops.

**Dataframe.** `Analysis.from_unit_panel` accepts sparse input and treats absent eligible
unit-days as zero. Small inputs may be densified temporarily; larger ones
use a sparse day-at-a-time reduction. `densified_cells` counts cells that
needed zero fill, not physical rows.

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

## The definitions directory

Definitions are YAML files; the loader merges every `.yaml` and `.yml` file in
the directory. The split below is a convention, not a requirement. Only
`dim_sources.yaml` is optional; use it when attributes live in separate
dimension tables ([Declarative dim sources](#declarative-dim-sources)).

```
definitions/
  fact_sources.yaml   # SQL relations that expose event tables
  dim_sources.yaml    # dimension tables joined into fact sources on demand
  exposures.yaml      # what marks a unit as enrolled
  metrics.yaml        # metric derivations (mean, conversion, ...)
  experiments.yaml    # experiment schedules and metric lists
```

Loading validates syntax, known fields, duplicate names, SQL, and the references
between experiments, exposures, metrics, and units. Errors name the offending
file.

??? info "Definition validation and errors"
    Loading validates everything up front: YAML syntax, model fields (unknown
    fields are rejected at every level), fact-source SQL syntax, duplicate names
    across files, and every cross-reference. An experiment's exposure must be a
    defined exposure, every metric it lists must be a defined metric, and every
    metric's entity must match the experiment's unit. Failures raise a
    `DefinitionError` naming the offending file.

    Fact-source and dim-source SQL must not be blank: an empty or whitespace-only
    `sql` refuses at load with `definition.source.sql_empty` (context
    `source_kind`, `source_name`), before any warehouse access.

!!! warning "Definitions are trusted code"
    Source SQL must be one non-empty, read-only `SELECT` (read-only CTEs and
    set operations included). Batches, DML, DDL, `SELECT ... INTO`, and locking
    reads are rejected. The loader checks syntax only: review every query and
    use read-only warehouse credentials. Report SQL is checked again before
    execution.

## `fact_sources.yaml`: where events live

A fact source is a SQL relation (table, view, or query) whose rows are events.
It declares which columns mean what:

```yaml
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
        description: A user viewed a page (occurrence-only event)
      - name: purchase
        column: revenue
        description: A purchase event with its revenue amount
    properties:
      - name: country
        column: country_code
        dtype: string
        description: Country at event time
```

| Field | What to declare |
|---|---|
| `dialect` | The SQL dialect. Prefer the warehouse's native dialect: Ibis translation is best-effort and can change meaning silently. If omitted, SQL is checked against the connection's dialect. |
| `timestamp_column` | The event time used for every window. |
| `entities` | Unit identifier columns; the experiment's `unit` must be one of them. |
| `facts` | Event types and their values. The SQL must expose `event`, the timestamp, and every entity under the declared names. `column` holds a fact's value; `column: null` marks an occurrence-only fact, usable for counts, conversion, and retention but not value aggregations. |
| `properties` | Columns for filters and breakouts. `as_of` states when a value applies: `event_time` (default), `pre_exposure` (latest value strictly before exposure), or `static` (asserted unchanged). |

!!! warning "Event-time properties cannot be breakouts"
    Segmenting outcomes by a value observed after exposure can bias results,
    so only `pre_exposure` and `static` properties are eligible.

??? info "Source SQL translation and fact values"
    `dialect` identifies the dialect in which the source SQL is written. If the
    connection uses another dialect, Ibis transpiles the SQL as a best-effort
    syntax translation, not a semantic guarantee. Some constructs can silently
    change meaning rather than fail loudly: an epoch-milliseconds function can
    map to a target function with the inverse meaning, integer floor division
    can become a differently rounded cast, and a file-reading table function can
    pass through unchanged to the connected backend. Match `dialect:` to the
    connection you actually use, and prefer SQL written natively for that
    backend except for ordinary arithmetic and standard SQL. Leaving `dialect`
    undeclared is safe for admission because source SQL is validated against the
    connection's dialect. The same guidance applies when you declare a dialect
    different from the backend you run against.

    A fact's `column` is its value column (`revenue` for a purchase); `column: null`
    declares an occurrence-only event: counting, conversion, and retention
    work on it, but value aggregations (`sum`, `avg_event`,
    `avg_calendar_day`, ...) do not.

## `exposures.yaml`: what enrolls a unit

An exposure marks enrollment. Define exactly one of `fact` (the first matching
event in the experiment window, optionally filtered) or `sql` (a query
returning `unit_id`, `ts`, and `group_id`, plus optional `experiment_id`).

```yaml
exposures:
  - name: first_page_view
    fact: page_view
    description: >
      The first page view inside the experiment window marks the unit
      as enrolled.
```

For a fact-based exposure, the fact source's rows also supply the arm: the
enrolling row carries `experiment_id` and `group_id`, and the unit's first
qualifying row decides both its enrollment time and its arm.

!!! warning "SRM needs an assignment-valid population"
    `srm()` compares enrolled counts with the planned allocation. That is an
    allocation check only when enrollment is recorded at assignment, or from a
    pre-treatment signal the treatment cannot change. If the treatment can
    affect whether a unit reaches or logs the enrolling event, imbalance may be
    a selection effect rather than a randomizer failure: run the allocation
    check on all targeted units and treat the subset ratio as a separate
    selection/telemetry diagnostic.

!!! warning "Allocation shares alone do not justify anytime-valid SRM"
    Anytime-valid SRM requires cumulative prefixes with the same known
    conditional arm probabilities at every assignment; independent categorical
    randomization qualifies. Blocked, adaptive, dependent, quota, exact-balance,
    and without-replacement protocols need fixed or protocol-specific inference.
    Use `inference="fixed"` for one predeclared look.

??? info "SRM result fields and inference"
    `SRMResult.p_value` has been removed. Pearson fixed-look evidence is
    `SRMResult.fixed_p_value`, and `is_srm` uses the evidence selected by
    `inference`. Anytime-valid adapters require `expected` or
    `design.allocation`; declared arms absent from a cumulative prefix keep
    zero counts. Fixed inference without expected/design allocation infers
    equal shares from observed arms and returns `log_e_value=None`, with
    Pearson as the selected evidence. Anytime-valid inference requires
    cumulative prefixes whose assignment units have the same known conditional
    arm probabilities at every assignment; independent categorical
    randomization suffices. Static marginal shares alone do not: blocked,
    adaptive, dependent, quota, exact-balance, and without-replacement
    protocols need fixed or protocol-specific inference. `inference="fixed"`
    is for one predeclared look.

## `metrics.yaml`: what to measure

A metric reduces each unit's events to one number. The four types (`mean`,
`conversion`, `retention`, `ratio`) are covered in
[Metric types](metric-types.md); two are enough here:

```yaml
metrics:
  - type: conversion
    name: purchase_rate
    description: "Did the user purchase within 14 days of exposure?"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14

  - type: mean
    name: revenue_per_user
    description: "Total revenue per user over the first 14 days"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14
```

`window_days` counts from each unit's first exposure. Names are unique across
the directory, so experiments reference metrics by name. For heavy-tailed
means, declare `winsorization`; its typed `inference` settings persist with
the metric through results and artifacts
([Winsorization](metric-types.md#winsorization)).

!!! warning "Percentile winsorization cannot support a decision"
    Its current intervals are experimental and withhold decision evidence:
    `stat_sig` is unavailable, not "not significant."

!!! warning "Prespecify the winsorization method"
    Method choice and seed/stream must be set before examining inferential results.

??? info "Winsorization inference contract"
    For heavy-tailed mean outcomes, add `winsorization` to the mean metric. Its
    typed `inference` specification persists through frame construction, native
    definitions, artifact context, and result JSON. Percentile inference defaults
    to `pooled-size-route-v1`, which runs `influence-normal-v1` on pools of at
    least 20,000 units with an expected `N (1 - q)` of at least 100 above the cutoff
    and `positive-log-kernel-bootstrap-t-v1` otherwise; the executed method is
    recorded on the confidence set and its reference. Explicit
    `joint-rank-projection-v1` retains the uniform rank method. Method choice and
    seed/stream must be set before examining inferential results. Every method's
    typed public status is `experimental`. The bootstrap's persisted qualification
    is `pointwise_asymptotic_model_conditioned_v1`, the influence interval's is
    `pointwise_asymptotic_influence_v1`, and a routed request carries
    `pointwise_asymptotic_size_routed_v1`: pointwise asymptotic candidates under an
    iid nonnegative smooth-positive-density model, not universal finite-sample or
    heterogeneous-effect claims.
    The full historical calibration manifest remains unresolved. It records 12
    known alternative variance failures plus contamination width-failure evidence;
    bounded normal/binomial diagnostics do not turn these into passes. An experimental
    inference specification reports its descriptive confidence set and withholds decision
    evidence (`evidence.experimental_reference`). The rank
    method's qualification is `uniform_support_conditioned_v1` and it requires
    externally justified support. Zero outcomes are accepted by every method; the
    bootstrap and influence constructions require the percentile to sit above the
    pooled zero share and refuse otherwise (`estimation.winsor.cutoff_in_zero_atom`).
    Percentile cutoffs are pooled across all eligible arms before the
    transformed per-unit values are aggregated. Fixed values use the declared cap
    directly. The same cutoff is applied to every arm, and the estimate concerns the
    capped outcome.

## `experiments.yaml`: tying it together

An experiment connects an exposure, an analysis unit, metrics, a schedule, and
a decision plan:

```yaml
experiments:
  - name: new_onboarding_v2
    description: "Redesigned onboarding flow vs the current flow"
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    plan:
      secondaries:
        - purchase_rate
        - revenue_per_user
    control_group: C
```

| Field | What to declare |
|---|---|
| `start`, `end` | Enrollment window in local days. Omit `end` for a running experiment. |
| `observation_end` | Extends follow-up past enrollment. For a one-time intervention, set it to `end` plus the longest metric window so late enrollees mature; omit it when treatment stops with the experiment. |
| `control_group` | Required. |
| `plan` | `primary`, `secondaries` (one multiplicity family), and `guardrails` (tested one-sided on the adverse side), plus `alpha`, `q`, and `inference`. |
| `design` | An `Encouragement` or `Observational` design; see the [encouragement](encouragement.md#declare-on-from_definitions) and [observational](observational.md) guides. |
| `n_pre_periods` | CUPED lookback. The window ends at first exposure and includes earlier events on the exposure date. |
| `breakouts` | Lift by declared property. |
| `cluster` | Randomization grain when it is coarser than `unit`; see [below](#randomization-grain-vs-analysis-grain). |

??? info "Decision roles and the CUPED window"
    The `plan` block declares the decision rule: `primary` (the
    pre-registered outcome), `secondaries` (reported alongside, sharing one
    multiplicity family), and `guardrails` (tracked separately, tested
    one-sided on their adverse side). It also carries `alpha`, `q`, and
    `inference`.

    An experiment can also declare `n_pre_periods` (a CUPED lookback in
    days, see [CUPED](cuped.md)) and `breakouts` (per-segment lift
    breakdowns on a declared property).
    The CUPED window is `[first exposure date − n_pre_periods days, first exposure timestamp)`, including exposure-day events before exposure.

    `observation_end` optionally extends data collection past `end`. Set it
    to `end + <longest metric window>` when the intervention is delivered
    once at enrollment (a new onboarding flow, a welcome email), so units
    enrolled near the end still get a complete window instead of being
    censored. Leave it unset for a continuous intervention that reverts
    when the experiment stops.

    `control_group` is required: the engine never guesses which arm is
    control.

### Window edges and the day boundary

`start`, `end`, and `observation_end` name days at the declared
`day_boundary`, however they are spelled: a value without a UTC offset
(`2025-01-15`, `2025-01-15T09:00:00`) is wall-clock time at the boundary, so
its written date is the day; a value with an offset (`Z`, `-05:00`) is
converted to the boundary first. Two spellings of one instant always declare
the same window. At `day_boundary: UTC-05:00`:

| Declared `start` | Kind | Window opens on local day |
|---|---|---|
| `2025-01-15` | naive (wall clock at the boundary) | 2025-01-15 |
| `2025-01-15T00:00:00-05:00` | aware | 2025-01-15 |
| `2025-01-15T05:00:00Z` | aware, same instant | 2025-01-15 |
| `2025-01-15T00:00:00Z` | aware | 2025-01-14 |

```python
from datetime import date
from increment.semantics.models import Experiment

exp = Experiment.model_validate(
    {
        "name": "demo",
        "exposure": "enrolled",
        "unit": "user_id",
        "start": "2025-01-15T00:00:00Z",
        "end": "2025-01-20T00:00:00Z",
        "control_group": "control",
        "day_boundary": "UTC-05:00",
        "plan": {"primary": "conversion"},
    }
)
assert exp.start_day == date(2025, 1, 14)
```

Published artifacts and sequential registrations bind these derived days, so
state created before this rule existed must be republished or re-registered.

### Per-metric method and prior bindings

Plan entries can name a metric directly or bind `decision_method`,
`sensitivity_methods`, or `prior` to that metric:

```yaml
experiments:
  - name: new_onboarding_v2
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    plan:
      secondaries:
        - metric: purchase_rate
          decision_method:
            name: unadjusted
          sensitivity_methods:
            - name: cuped
              variance_reduction: cuped
          prior:
            mu: 0.0
            sigma: 0.02
        - avg_session_duration  # shorthand -- unaffected by the binding above
    n_pre_periods: 14
    control_group: C
```

A call-wide `run(...)` setting overrides a plan binding for the fields it
specifies; otherwise each metric uses its own binding, so one call can mix
methods and priors across metrics. [CUPED](cuped.md#per-metric-method-bindings)
covers precedence and the dataframe equivalent,
`MetricSpec(decision_method=..., sensitivity_methods=..., prior=...)`.

??? info "Metric binding defaults and override scope"
    Every entry in a `plan` role (`primary`/`secondaries`/`guardrails`) is either a bare metric NAME
    (shorthand -- reports under the call's own defaults) or a BINDING that
    overrides `decision_method`/`sensitivity_methods`/`prior` for that one metric
    only:

    A call-wide role-aware `run(decision_method=..., sensitivity_methods=...)`/`run(prior=...)` still wins over a
    declared binding for every metric it names -- the binding only fills in
    when the call leaves that field unset. This precedence is per-metric,
    independently: one `run()` call may dispatch several metrics under
    different adjustment sets and different priors at once.

## Randomization grain vs analysis grain

`unit` is the analysis grain: one row per unit feeds the estimate. Many
experiments randomize something coarser than they measure: stores are flipped
while users are measured, geos while sessions are measured. Units in one
randomization cluster share its assignment and usually its quirks, so they are
not independent. Treating them as independent understates the standard error
by roughly `sqrt(1 + (m - 1) * rho)` (`m` = average units per cluster, `rho` =
intra-cluster correlation): at 20 users per store and `rho = 0.5`, the
reported interval is ~3x too narrow.

Declare it one of two ways:

- **The cluster is what you measure** (revenue per store): declare
  `unit: store_id` and stop. Randomization and analysis share a grain, so the
  plain per-arm variance is valid.
- **Coarser randomization, finer measurement** (revenue per user, randomized
  by store): keep `unit: user_id` and declare the grain.

```yaml
experiments:
  - name: store_rollout
    exposure: enrolled
    unit: user_id
    cluster: store_id   # column on the exposure source
    plan: {secondaries: [revenue_per_user]}
    control_group: control
```

**Warehouse.** `cluster` names a column on the exposure source. Uncertainty is then
computed from independent clusters, and `srm()` tests the distinct cluster
count per arm (`SRMResult.grain == "cluster"`). Per-arm unit counts ride
along on `SRMResult.unit_counts` as context: unequal cluster sizes make
unit counts drift for reasons the randomizer never controlled, so that
imbalance is reported, never tested.

**Dataframe.** Pass `cluster="store_id"` to `Analysis.from_unit_summary`.
First aggregate panel data to one row per unit, keeping its cluster label.
Uncertainty is computed from independent clusters, and `srm()` reports the
same two grains.

What a clustered experiment supports:

| | Supported | Refuses |
|---|---|---|
| Metrics | `mean`, `conversion`, `retention`, and `ratio` (over the metric's own per-cluster totals) | Quantile metrics: they do not decompose over cluster moments. |
| Views | Total grain: `run()`. | Daily, as-of, and cohort views, breakouts, and factors: none carries a cluster-robust variance. |
| Designs | Randomized; observational on unadjusted, IPTW, DML, and AIPW; encouragement ITT and the additive LATE. | Under `Encouragement`: CUPED, ratio metrics, and the complier-relative `late` row ([why](cuped.md#clustered-encouragement-designs)). |
| Decision stats | Read the interval directly. | `chance_to_beat` and `prob_favorable` on every clustered row: sampling intervals do not provide a Normal posterior. |
| Exports | Encouragement exports, through the versioned compliance payload. | Ordinary clustered exports from `from_definitions`, `from_unit_summary`, and `from_unit_day_artifact`: `source.moments.cluster_grain`. Analyze the source directly. |
| Labels | One arm per cluster in randomized and encouragement designs; observational dependence clusters may span arms. | A null cluster label; a label in both arms of a randomized or encouragement design. |

!!! warning "Independent information comes from clusters, not rows"
    Below 40 clusters the readout warns that the variance estimate is noisy;
    each enrolled arm needs at least two clusters. No cluster should dominate
    an arm's variance: a store carrying half its traffic undermines the
    approximation even when the count clears the threshold. Cluster count sets
    the working reference; it does not establish calibration.

!!! warning "Clustered SRM also depends on the assignment protocol"
    Independently assigned clusters with known, constant conditional arm
    probabilities qualify for anytime-valid SRM; blocked or exact-balance
    cluster protocols do not. Fixed-look Pearson inference remains available
    as an explicit option.

??? info "Statistical details"
    With `cluster` declared (or `from_unit_summary(..., cluster="store_id")` on
    the dataframe path), uncertainty is computed from independent clusters.
    The dataframe path (`from_unit_summary(..., cluster=...)`) reports the same two grains.

    - Randomized moments and observational `Method(name="unadjusted")` with
      disjoint arms use a joint relative confidence set with a working t
      reference at `min(K_T - 1, K_C - 1)` degrees of freedom; the additive
      interval uses its own Welch degrees of freedom. Observational clusters
      spanning both arms keep the cross-arm covariance with an asymptotic
      Normal reference. IPTW/DML/AIPW group per-unit influence contributions
      by cluster and take the variance over cluster totals (Liang-Zeger)
      against an asymptotic Normal score reference, with two clusters per
      pure arm and the below-forty advisory. Each path counts distinct
      cluster labels across the contrast once; for disjoint arms this equals
      `K_T + K_C`.
    - If the estimated variance of the relative contrast at its observed point
      is zero, the row keeps that point but withholds relative bounds and
      decisions with `relative_unavailable_reason="zero_relative_variance"`;
      an available additive interval stays on the row. Zero sample
      variability is not a certificate of zero population uncertainty; other
      singular Fieller geometries remain supported. Such a secondary stays a
      non-rejection in its family's declared roster, so it neither inflates
      sibling discoveries nor blocks usable siblings.
    - Estimators that do not refit nuisance models (randomized moments, IPTW)
      change only the SE and reference when a cluster is declared. DML/AIPW
      cross-fit on cluster-atomic folds, so a non-singleton cluster
      declaration can change nuisance predictions and the point estimate;
      singleton clusters deliberately fall back to unit-ID folds and keep the
      unclustered point estimate.
    - A ratio metric needs no new wire fields: a clustered row is already a
      ratio over clusters, and the denominator family carries the metric's
      own per-cluster denominator total `den_j` instead of the cluster size
      `m_j`, so the estimand is `sum(num_j) / sum(den_j)`, exactly the ratio
      the unclustered readout reports. Which reading applies is driven by the
      metric's declared `type`, never guessed from the row. Revenue per
      session in a market test is the motivating case.
    - Under `Encouragement`, the ITT row and the additive `late` row use the
      same cluster-robust ratio delta method with component-specific Welch
      references and the structural two-clusters-per-arm requirement; the
      `compliance` (first-stage) row reports on the absolute scale directly.
      CUPED refuses because the clustered collapse carries no per-unit
      covariate moments; a clustered ratio metric refuses because the LATE
      first stage already reads cluster sizes out of the denominator family
      the metric would need for its own denominator; the relative LATE needs
      a moment family this reduction does not build.
    - Encouragement exports preserve cluster grain through the versioned
      compliance payload, including member counts and centered uptake/size
      moments. Ordinary clustered exports refuse before writing because the
      moments wire format has no cluster marker; artifact-backed direct
      analysis retains its cluster evidence.
    - CATE and targeting consume unit rows with cluster metadata (the
      source's `unit_frame`, not the collapsed arm moments): fitting, honest
      holdouts, selection folds, and uncertainty keep the declared dependence
      grain. Choose `cluster_weight="member_count"` or `"equal"` consistently
      across the workflow, and declare `intervention_grain="cluster"` for
      whole-cluster deployment; see
      [conditional effects](metric-types.md#conditional-effects-cate).
    - Anytime-valid `srm()` holds for independently assigned clusters with
      known, constant conditional arm probabilities; blocked or exact-balance
      cluster protocols do not qualify. Fixed-look Pearson inference remains
      available as an explicit option.

## Canonical unit-day artifacts

Definitions-backed analysis has two persistence boundaries:

| Call | What it does |
|---|---|
| `analysis.materialize()` | Rebuilds a session-local copy of the unit-day core from current warehouse data (TEMP relations on the current connection). Every call rescans the source unless `store="none"` or no metrics are declared; qualifying readouts rebuild per call (`"always"` from the first call, `"auto"` from the second), and nested reductions reuse that operation's copy, not an earlier call's. Not a portable artifact, and not an atomic snapshot across several source tables. |
| `analysis.publish_unit_day_artifact(store, extensions=(), refresh_of=None)` | Builds the canonical core once and publishes an immutable format-1 artifact generation through the store, returning a `UnitDayArtifactRef`. A later process adopts it with `Analysis.from_unit_day_artifact(store, ref, expected_context=...)` without rebuilding facts or exposures. |

A refresh keeps `artifact_id` but creates a new `generation_id`, and must use
the same compiled context as the pinned prior generation. Completeness is only
ever proven by certifications you supply: observed timestamps, `data_as_of`,
publication time, and experiment end are not completeness proofs.

!!! warning "Do not adopt a generation from a failed publication"
    An exception after the manifest is published invalidates the generation:
    reads are refused with `artifact.generation.dropped`. If cleanup itself
    fails, the exception names the `artifact_id`, `generation_id`, and any
    retained relations; finish erasing them from a fresh process as described
    in the recovery fold below.

??? info "What publication captures"
    Publication captures the required fact, exposure, dimension, trigger, and
    extension streams in one source execution; assignment and cluster
    validation, measure rows, freshness, event horizons, and coverage all
    derive from those captured inputs. The captured data is held in a
    warehouse-side `TEMP` table, so the backend must support temporary-table
    materialization. Raw relations are not collected into client memory, and
    later reductions never fall back to a live source. Relation writes persist
    the same captured rows used for their digest.

    Definitions-backed callers may supply `SourceSnapshotEvidence` with an
    explicit aware event-time cutoff and per-feed certified `complete_through`
    instants. Missing certifications remain unknown. Trigger membership uses
    only the trigger feed's certification; each metric's trigger outcomes use
    the certifications for that metric's contributing fact-source feeds, all
    bound to the same pinned cutoff.

    A percentile-winsorized metric's own snapshot, every artifact publish's
    snapshot, and `dashboard_snapshot`/sequential capture's own pinned read
    scope each captured fact/dimension source to the experiment's enrolled
    units and, for sources whose rows are bounded by the analysis window, to
    `start - (n_pre_periods + 1)` days onward. No upper time bound is pushed,
    because the freshness watermark and breakout/factor/covariate property
    lookups need rows that bound would clip.

    The manifest is written last. It contains the artifact/generation
    identity, experiment and local day boundary, date coverage, freshness, the
    mandatory `exposures` and `measure_stats` relations, metric-to-measure
    bindings, the caller-trusted `ArtifactContext`, and any extension
    descriptors. Every relation ref is bound to the same artifact generation
    and semantic role.

    The warehouse store persists each relation before hashing the persisted
    rows. Digesting uses bounded Arrow batches and a disk-backed canonical
    merge sort, so hashes do not depend on warehouse collation and remain
    byte-identical to buffered digests. Streaming `RelationDigests.rows_bytes`
    is `None`; buffered `digest_relation()` still includes the canonical row
    bytes. These bounds apply to Python digest batches, not total process
    memory: exposure metadata still uses one row per unit, warehouse query
    buffers may grow, and `unit_frame()` materializes the requested row-level
    data.

    Adoption verifies a private temporary copy of each relation and keeps
    reductions warehouse-backed. Later external writes to a published table
    cannot alter that verified copy; closing the read handle removes its
    private tables.

??? info "Recovering from a failed publication"
    If an exception is raised after the manifest is published, the publication
    context invalidates the generation through `drop_generation` (tombstone
    first): a fresh store lists nothing for it, and reads (`open_snapshot`,
    `verify_relation`) are refused with `artifact.generation.dropped`. If
    invalidation itself fails, the original exception is re-raised carrying a
    note with the `artifact_id` and `generation_id`; from a fresh process, run
    `store.drop_generation(artifact_id, generation_id)` to finish erasing the
    relations. That call is idempotent.

    An exception before the manifest-index insert is submitted never made the
    generation visible, and the store only drops its relations, attempting
    every one, including a relation whose write failed and whose immediate
    cleanup drop also failed. If a drop fails there, an aborting exception
    carries a note naming the retained relations, and a publication that ends
    normally raises
    `query.session.warehouse_artifact.publication_cleanup_incomplete` with
    `relations`, `artifact_id`, `generation_id` and `route` in its context;
    drop them by name.

    Once the insert has been submitted the store cannot know whether it
    committed, and a remote insert can still commit after the call has raised,
    so the abort writes the tombstone for the generation first: whether the
    manifest row already exists or lands later, it is hidden and refused with
    `artifact.generation.dropped`, and only then are the relations erased;
    erasure attempts every relation even if one drop fails. Two failures keep
    relations in the warehouse. If the tombstone cannot be written, every
    relation is preserved. From a fresh process, call
    `store.abandon_generation(artifact_id, generation_id)` first: it writes
    only the tombstone, is durable and safe to repeat, and hides the generation
    whether its manifest row exists now or lands later. Then drop the listed
    relations by name. If a manifest row exists, `drop_generation` also works.
    If only a drop fails after the tombstone, the generation is already hidden
    and the failed relations are removed by name. An aborting exception
    carries a note with the `artifact_id`, `generation_id` and the retained
    relation names, qualified by catalog and schema. A publication that ends
    normally without a manifest after an ambiguous insert raises
    `query.session.warehouse_artifact.publication_state_unknown` with
    `relations` (a tuple of qualified names), `artifact_id`, `generation_id`,
    `tombstoned` and `route` in its context, chained to the underlying failure.

??? info "Dropping a published generation"
    `drop_generation()` on `WarehouseArtifactStore` durably invalidates a
    published generation: it appends a tombstone row to a store-owned
    control-plane table before touching any relation, and every store sharing
    that namespace refreshes visibility on each access, so the generation is
    hidden and refused everywhere as soon as the tombstone commits. The
    manifest-index row is kept afterward as a cleanup receipt, not erased, so
    a fresh process can recover the exact relation names and retry physical
    erasure if a prior attempt failed partway. A successful return
    additionally means every named relation was erased; a failure during
    erasure still leaves the generation durably invalid and safe to retry. A
    never-published pair returns immediately with no effect;
    `abandon_generation` hides such a pair. An already-dropped pair is not a
    true no-op: it revalidates the retained receipt and retries erasing every
    named relation, idempotently (an already-erased relation is left alone).
    Rows already exported, cached, or copied elsewhere before invalidation are
    not revoked by this call.

    Enabling deletion in an existing deployment requires namespace migration
    privileges once, to create the tombstone table, before any process
    constructs a `WarehouseArtifactStore` against that namespace again:
    construction itself fails closed with a coded refusal if the tombstone
    table is missing and cannot be created, so a reader can never silently
    miss a tombstone. Every reader and writer sharing that namespace must be
    upgraded first, since older library versions ignore tombstones and are
    unsafe during incomplete cleanup. Generations dropped by an older
    in-memory-only release left no durable record; drop them again explicitly
    if they should be retired. `Analysis.close()` only releases an adopted
    snapshot handle and never deletes durable artifact data; any backend
    cleanup policy must be coordinated by the caller outside these handles.

### Operation and extension matrix

Every adopted artifact exposes `moments_source`, `day_source`, and
`export_moments` (which exports a moments cube; it does not re-publish the
artifact). On an adopted artifact, `materialize()` and `panel_sql()` are
refused with `facade.analysis.operation`, and `summary_sql()` with
`artifact.operation.unsupported`. Optional operations are enabled only by
their published evidence:

| Published extension | Enabled operation(s) |
|---|---|
| `breakout_dimension` | `breakout_source`, `breakout_sources`, `breakout_summaries` |
| `factor_dimension` | `factor_summaries` |
| `site_volume` | `sitewide_evidence` |
| `trigger_population` plus `assignment_counts` | `triggered_counts` |
| `trigger_population` plus `assignment_counts` and the metric's `trigger_measure_stats` | metric-specific `triggered_source` outcomes |

`cluster_identity`, `cuped_preperiod`, and `encouragement_uptake` provide
relation evidence consumed by the corresponding total/day reductions; they do
not add a generic operation. The extension catalog is closed and immutable: a
request must match exactly one catalog entry, including its canonical
definition and source-provenance hashes. Supported requests are
`breakout_dimension`, `factor_dimension`, `cluster_identity`,
`cuped_preperiod`, `assignment_counts`, `trigger_population`,
`trigger_measure_stats`, `encouragement_uptake`, and `site_volume`.

??? info "Triggered outcomes and site volume"
    Triggered outcome summaries are raw-event-filtered from the unit's first
    eligible trigger, then aggregated by that trigger-relative local day. The
    exclusive window edge is the first local day after the declared number of
    days. Trigger-population evidence before version 3 and outcome artifacts
    without the selected metric's summary cannot serve a triggered outcome;
    use definitions-backed input or rebuild and republish with version-3
    trigger membership and the metric's `trigger_measure_stats` request.

    The publisher filters trigger-relative raw events at the explicit pinned
    event-time cutoff before making daily aggregates. The reader rejects
    aggregate dates later than the cutoff's local day, while permitting a
    partial cutoff-day aggregate. Daily rows contain no raw timestamps, so the
    reader cannot independently prove which within-day events were admitted;
    exact cutoff filtering on that day is trusted publisher behavior bound to
    the artifact's source provenance and cutoff metadata.

    Logical metrics may share one physical site-volume recipe. Publication
    writes that recipe once; each metric retains its binding to the shared
    evidence when the artifact is adopted. Site volume covers the enrollment
    window `[start, end]` in days at the declared `day_boundary`, even when
    `observation_end` extends past `end`; an experiment without an `end`
    publishes no site-volume coverage.

??? info "Artifact refusal codes"
    - `artifact.format.unsupported`: the context format is not version 2 or the
      artifact format is not version 1; republish from trusted definitions.
    - `artifact.manifest.invalid`: the manifest body, identity, ordering, or
      self-digest fails validation.
    - `artifact.identifier.unsafe`: a locator, namespace identifier, or digest
      identifier is malformed or outside the permitted identifier grammar.
    - `artifact.context.mismatch`: canonical context JSON, its hash, or a supplied
      context payload does not agree.
    - `artifact.refresh.invalid_ref`: a pinned ref is malformed, unpublished,
      generation-bound incorrectly, or has a locator/digest mismatch.
    - `artifact.refresh.context_mismatch`: a refresh or open uses a context different
      from the caller-pinned/previous generation context.
    - `artifact.relation.schema_mismatch` and `artifact.relation.digest_mismatch`:
      a stored relation's schema or content digest differs from its manifest ref.
    - `artifact.snapshot.mixed`: a relation, manifest, or generation changes or is
      bound to a different snapshot while a read is in progress.
    - `artifact.digest.type`, `artifact.digest.nullable`, and
      `artifact.digest.primary_key`: a digest cell has the wrong physical type,
      violates nullability, or has a missing/duplicate/non-unique primary key;
      multi-row relations must provide a primary key.
    - `artifact.digest.row`: a digest row is not a mapping/model or is missing a
      field declared by the relation schema.
    - `artifact.digest.rows`: the relation rows container is not a supported
      sequence.
    - `artifact.digest.nonfinite`, `artifact.digest.schema`,
      `artifact.digest.timestamp`, `artifact.digest.range`, `artifact.digest.role`,
      `artifact.digest.json`, and `artifact.digest.recanonicalize`: digest input
      violates the corresponding finite-number, schema, UTC timestamp,
      integer-range, ASCII-role, JSON, or canonical-representation rule.
    - `artifact.aggregate.impossible`: publication encounters an aggregate with
      missing sufficient statistics, impossible min/max or event counts, or
      non-finite/contradictory single-event values.
    - `artifact.extension.missing`, `artifact.extension.invalid`, and
      `artifact.metric.binding_mismatch`: required evidence is absent,
      malformed/tampered, or inconsistent with the manifest metric binding.
    - `artifact.operation.unsupported`: the SOURCE cannot perform the requested
      operation at all — an unsupported grain, or raw SQL a snapshot does not
      expose.
    - `artifact.evidence.unavailable`: the requested evidence or shape is not
      available from the selected source or published artifact, even though the
      broader operation is supported — a source lacking a required operation or
      unit-grain evidence shape during extension publication raises this too.
    - `artifact.store.namespace`: the store's own catalog/schema namespace does
      not exist or is invalid, or the deletion tombstone table specifically is
      missing and cannot be created, or has an unexpected schema. The
      manifest-index table is validated at store construction too: it must
      contain the columns `artifact_id`, `generation_id`, `manifest_name`,
      `manifest_sha256`, and `manifest_json`, all of string type, with no
      missing, extra, or wrong-type columns (physical column order is free),
      and an absent index that cannot be created is refused as well.
    - `artifact.manifest.unreferenced_relation`: a publication wrote a relation
      that its manifest does not reference (in base relations or extensions);
      publication is sealed against this before the manifest is persisted.
    - `artifact.generation.dropped`: the requested generation has a durable
      deletion tombstone; every read (`open_snapshot`, `verify_relation`) is
      refused with this code. Dropping it again is not refused: `drop_generation`
      revalidates the retained receipt and retries erasing every named relation
      idempotently.

### Trust, namespace, and read consistency

The caller-pinned `UnitDayArtifactRef` (artifact UUID, generation UUID,
namespace-qualified manifest locator, and manifest digest) is the trust root.
The store, not Increment, owns authorization for the catalog/schema namespace;
locators outside that namespace or outside the pinned generation are refused.
Opening validates the pinned manifest and then holds one immutable
`ArtifactSnapshot` for verification and every reduction, so later writes or
rebinding cannot mix generations. If both the artifact and the supplied ref
are replaced together, Increment cannot detect that replacement: the caller's
ref distribution and store authorization are the authenticity boundary.

!!! warning "Digests are integrity evidence, not secrecy"
    Do not put credentials in SQL literals. Hashing does not conceal a secret
    that someone can guess, and copies of a format-1 manifest that were
    already stored or shared still contain the SQL until an operator handles
    them.

??? info "Context and digest construction"
    `ArtifactContext` is compiled from an already-loaded `Definitions` model,
    not from a path or hidden warehouse I/O. Its `context_format` is `2`, and
    the same `context_format` appears inside `canonical_json` so the digest
    binds it; the two must agree. Its `canonical_json` is compact RFC 8785
    canonical JSON: finite values, no whitespace, and object keys ordered by
    UTF-16BE code units. Its `sha256` is
    `SHA256(b"increment.unit-day-artifact\x00v1\x00context\x00" +
    canonical_json_bytes)`, where `\x00` denotes the actual NUL separator byte.

    A format-2 context carries the typed experiment (including the effective
    encouragement design), the complete selected metric roster, the dialect
    and day boundary, and the extension catalog. It never carries fact,
    dimension or exposure SQL: the native source recipe and every extension
    source recipe are represented by domain-tagged SHA-256 digests, and a
    receipt is never executed. The context also carries `window_days`, the
    boundary-local day of each declared window edge (`start`, `end`,
    `observation_horizon`), and readers use those stored days rather than
    deriving an edge from a timestamp. A format-2 context without
    `window_days` is refused by `artifact.context.mismatch` (route: republish
    from trusted definitions), before any manifest, relation or pin is used.

    Context format 1 is retired. Opening or publishing with a format-1 context
    is refused by `artifact.format.unsupported`, naming the received and
    supported versions. Republish from trusted definitions to obtain a
    format-2 artifact; the refusal never upgrades an old context. Existing
    artifacts are not deleted and credentials are not rotated automatically.
    Relation digest versions are independent of the context format and remain
    readable.

    Relation schema/content and manifest digests use the same domain root,
    role-specific markers, schema bytes, row count, and rows sorted by primary
    key. The manifest digest excludes its own `manifest_sha256` field. All
    digest strings are lowercase 64-character hexadecimal SHA-256.

### Moments wire migration

`Analysis.from_moments` reads centered, complete `moments_format=12`
fixed-horizon rows, including a complete embedded `decision_plan`, a source
identity record and, on every denominator row, the ratio denominator's
centered third moment `cden3`. It also reads the earlier fixed-horizon formats
10 and 11 and sequential checkpoint formats 9 and 10. Formats above 12 are
refused with `moments.format.unsupported_future`; fixed-horizon formats 1–8
with `moments.format.unsupported_legacy`. `from_moments` is separate from the
unit-day artifact boundary.

!!! warning "An old cube cannot be upgraded in place"
    Re-export from the original observations, or pin the Increment revision
    that wrote the old cube. A cube alone cannot recover exact binary counts
    from rounded moments, and a format-10 or format-11 cube cannot recover the
    third moment: its ratio rows load, and every lift row estimated from them
    reports the denominator-skew coverage check as unavailable rather than
    passing silently. Rounded fixed-horizon moments and legacy format-8
    sequential envelopes cannot resume a sequential process.

??? info "Row contracts, hand-authored summaries, and envelopes"
    Fixed-horizon format 10 and sequential format 9 lack a source identity
    field, and a format 11 cube may be loaded without one. Their replay
    identity includes a legacy-format namespace and study ID; the original
    provenance cannot be reconstructed. A `source_identity` embedded in a
    moments cube is self-asserted, not authenticated provenance.

    For hand-authored per-arm summaries, `moments_from_summaries(...)` builds
    format-12 rows for fixed-horizon mean and conversion metrics. Mean rows
    require `n >= 2`, `mean`, and unbiased `var` (ddof=1), then use
    `ref_y=mean`, `cy1=0`, and `cy2=(n-1)·var`; conversion rows require exact
    integer `successes` and retain that count. Rows have a strict schema;
    extra fields are refused with `moments.summaries.invalid`. Optional
    `assignment_counts` records source-level enrollment counts independently
    of metric counts. When omitted, `unit_counts()`, `srm()`, and the health
    allocation check are unavailable with `source.moments.assignment_counts`.
    The helper does not infer ratio denominators, covariates, cluster
    structure, or sequential registration; use `from_unit_summary` for those
    richer summaries.

    Each ordinary row carries integer `n >= 1` and nullable integer
    `successes` between zero and `n`. Eligible unit-grain conversion and
    retention outcomes retain their original 0/1 success counts;
    cluster-aggregate rows and other outcomes carry null. Partition merges add
    counts as integers, never via floating totals. Missing counts are not
    reconstructed from a mean. Import and export validate count ranges on
    every row, including unselected imported metrics, with
    `moments.count_out_of_range` for invalid ranges. Exports from definitions,
    unit-day artifacts, and dataframe sources refuse invalid counts before
    creating the Parquet file. Parquet exports use nullable `int64` for
    `successes`; preserve that type when loading through a dataframe,
    including pandas' nullable `Int64` dtype. Warehouse daily conversion
    summaries also retain integer successes: each unit contributes at most one
    success per day, regardless of its event count.

    The `metrics` declaration selects which metrics to replay. Winsorization
    declarations must match selected metrics; an unselected clipped metric
    does not require its configuration to replay an ordinary sibling. Format
    stamps and wire-schema validation still cover every row of the imported
    cube.

    An empty-metric fixed-horizon Encouragement export carries one current
    format-12 `design_summary` envelope instead of outcome rows. It includes
    the experiment identity, complete assignment counts and compliance state,
    source identity, and empty-metric fixed-horizon decision plan. Reload with
    `metrics=[]` and the same design. Missing state, mixed rows, outcome
    fields, a different identity, or a nonempty metric catalog are refused;
    overriding the plan cannot bypass envelope validation. This envelope is
    not a sequential checkpoint.

    Registered sequential exports use one current format-10
    `sequential_checkpoint` envelope carrying the exact snapshot, retained
    roster, registration (schema version 2, including source mapping identity
    and compliance policy), parent-linked record proof, and source identity.
    Their embedded compiled plan uses wire version 3. The preceding sequential
    format 9 remains readable. See
    [Sequential inference](sequential-inference.md) for explicit finalization,
    idempotent replay and append-only continuation.

## Running it

With the four files above saved under `definitions/`, the whole pipeline is
one constructor and one call. `events` here is a stand-in for the table your
pipelines already load, one row per `page_view` and `purchase`:

<!-- invisible-code-block: python
from pathlib import Path

Path("definitions").mkdir(exist_ok=True)
Path("definitions/fact_sources.yaml").write_text("""\
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
        description: A user viewed a page (occurrence-only event)
      - name: purchase
        column: revenue
        description: A purchase event with its revenue amount
    properties:
      - name: country
        column: country_code
        dtype: string
        description: Country at event time
""")
Path("definitions/exposures.yaml").write_text("""\
exposures:
  - name: first_page_view
    fact: page_view
    description: >
      The first page view inside the experiment window marks the unit
      as enrolled.
""")
Path("definitions/metrics.yaml").write_text("""\
metrics:
  - type: conversion
    name: purchase_rate
    description: "Did the user purchase within 14 days of exposure?"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14

  - type: mean
    name: revenue_per_user
    description: "Total revenue per user over the first 14 days"
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14
""")
Path("definitions/experiments.yaml").write_text("""\
experiments:
  - name: new_onboarding_v2
    description: "Redesigned onboarding flow vs the current flow"
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    plan:
      secondaries:
        - purchase_rate
        - revenue_per_user
    control_group: C
""")

import datetime as dt
import random

import polars as pl

# A tiny synthetic event log; in production this is a table your
# pipelines already load.
random.seed(7)
start = dt.datetime(2025, 1, 15, 9, 0)
rows = []
for i in range(600):
    user = f"u{i:04d}"
    group = "T" if i % 2 else "C"
    enrolled = start + dt.timedelta(days=i % 10)
    rows.append(
        {
            "event_at": enrolled,
            "user_id": user,
            "event": "page_view",
            "revenue": None,
            "country_code": random.choice(["US", "DE", "BR"]),
            "experiment_id": "new_onboarding_v2",
            "group_id": group,
        }
    )
    if random.random() < (0.32 if group == "T" else 0.25):
        rows.append(
            {
                "event_at": enrolled + dt.timedelta(days=random.randint(0, 13), hours=2),
                "user_id": user,
                "event": "purchase",
                "revenue": round(random.uniform(5.0, 60.0), 2),
                "country_code": None,
                "experiment_id": None,
                "group_id": None,
            }
        )

# Background purchases from users outside the experiment: the warehouse
# keeps loading events past the last enrollee's window close.
for d in range(30):
    rows.append(
        {
            "event_at": dt.datetime(2025, 1, 16, 12, 0) + dt.timedelta(days=d),
            "user_id": f"bg{d:03d}",
            "event": "purchase",
            "revenue": round(random.uniform(5.0, 60.0), 2),
            "country_code": None,
            "experiment_id": None,
            "group_id": None,
        }
    )
events = pl.DataFrame(rows)
-->

```python
import ibis

from increment import Analysis

con = ibis.duckdb.connect()  # in-memory; swap for Snowflake/BigQuery/Postgres
con.create_database("analytics")
con.create_table("event_log", events.to_arrow(), database="analytics")

analysis = Analysis.from_definitions("new_onboarding_v2", "definitions/", con)
results = analysis.run()
for r in results:
    print(f"{r.metric} / {r.group_id}: lift={r.lift.value:+.2%}")
```

<!-- output -->
```text
purchase_rate / T: lift=+12.37%
revenue_per_user / T: lift=+10.64%
```

The connection is bound once at construction; every readout (`run()`,
`run_daily()`, `run_breakout()`, and the rest) reads it from the instance. To
see the SQL a readout would execute without running it, `analysis.panel_sql()`
and `analysis.summary_sql()` return the per-metric queries as strings. SQL
previews and `build_panel_for_metric()` return live warehouse expressions, not
operation-owned TEMP table references, so they remain executable after another
readout refreshes materialization.

!!! note "Why the feed must load past the window"
    Each metric is bounded by its fact's `data_as_of`: the latest date the
    fact has actually loaded. If the purchase feed stopped before the last
    enrollee's 14-day window closed, those units would be censored out (with
    a warning) rather than scored on partial data. The censoring resolves
    once more data arrives, not by changing the experiment's schedule.

## Binding a normalized warehouse

The example above binds one flat event log. Most warehouses are not shaped
that way: attributes live in dimension tables, a subscription plan has history
rather than a current value, and the arm arrives from a platform export rather
than riding on every event row. None of that requires reshaping the warehouse
first: a fact source is a **query**, so the join lives in the definition and
the warehouse runs it. `examples/realistic_demo/` is the complete worked
example (a seven-table generator, definitions, and a notebook). This section
shows the three join shapes it needs and the four rules that bite, first as
hand-written SQL every backend understands, then as the declarative
`dims:`/`dim_sources:` form the shipped YAML uses: same joins, same numbers,
generated instead of hand-maintained.

### Fact sources are queries

An order fact joined to a static dimension (equi-join) *and* to a
slowly-changing one (temporal range join):

```yaml
fact_sources:
  - name: orders
    sql: |
      SELECT
        'order' AS event,
        o.user_id,
        o.ordered_at,
        o.total_amount,
        u.country,
        p.plan
      FROM analytics.fact_orders o
      JOIN analytics.dim_user u
        ON u.user_id = o.user_id
      JOIN analytics.snap_user_plan p
        ON p.user_id = o.user_id
       AND o.ordered_at >= p.valid_from
       AND o.ordered_at <  p.valid_to
      WHERE o.status = 'completed'
    timestamp_column: ordered_at
    entities:
      - user_id
    facts:
      - name: order
        column: total_amount
        description: A completed order, valued at its total amount
    properties:
      - name: country
        column: country
        dtype: string
        as_of: static
        description: Signup country, fixed for the unit's lifetime
      - name: plan
        column: plan
        dtype: string
        as_of: pre_exposure
        description: Plan in force at the row's own timestamp
```

`snap_user_plan` is a *snapshot* table (one row per plan version, bounded by
`valid_from`/`valid_to`), not a Kimball dimension carrying a surrogate key per
version. The fact rows hold only `user_id` and a timestamp, so the range join
resolves the version at query time and `as_of` controls which timestamp it
resolves against. If your ETL instead stamps a version key onto the fact rows,
that resolution already happened upstream: the column arrives pre-joined, and
`as_of: static` is the honest declaration for it.

The `event` column is synthesized (`'order' AS event`) because an orders table
has no natural discriminator. A behavioural event stream aliases its own
column instead (`event_name AS event`) and declares one fact per value; the
full example below carries such a source (`web_events`), joined to the same
two dimensions for the reason the third rule explains.

The arm is its own join. Assignment (every bucketed unit) and exposure (the
subset that actually reached the surface) are separate tables, so the exposure
source joins them to pick up `group_id`:

```yaml
  - name: experiment_exposure
    sql: |
      SELECT
        'exposure' AS event,
        x.user_id,
        x.experiment_id,
        a.variant AS group_id,
        x.exposed_at
      FROM analytics.fact_exposure x
      JOIN analytics.fact_assignment a
        ON a.user_id = x.user_id
       AND a.experiment_id = x.experiment_id
    timestamp_column: exposed_at
    entities:
      - user_id
    facts:
      - name: exposure
        column: null
        description: The unit reached the surface the experiment changed
```

Exposure rows may repeat per unit (platform exports usually log every
evaluation); the earliest qualifying one wins. Units that never trigger never
appear, which makes this a triggered comparison rather than an
intent-to-treat one.

### Four rules that bite

!!! warning "`as_of: pre_exposure` needs rows that predate exposure"
    A pre-exposure property resolves to the latest value strictly *before*
    the unit's first exposure. A source whose rows are all post-exposure (an
    orders table, say) has nothing to resolve, and every unit lands in the
    `__null__` bin rather than erroring. Resolve such properties from a source
    that carries pre-experiment history.

!!! warning "Declare `source:` on every breakout"
    With `source:` omitted, a breakout resolves to the *first* fact source
    that lists the property. When the same property is on several sources,
    which the next rule forces, that silently picks one for you. Name it.

!!! warning "Every metric's own source must carry the breakout property"
    A breakout applies to each metric through *that metric's* source, so a
    property declared on only one source fails validation for metrics bound
    to the others (`skip_missing: true` opts a breakout out instead). This is
    why the behavioural-event source in the full example repeats the two
    dimension joins the `orders` source already makes.

!!! warning "Close slowly-changing rows with a sentinel, not `NULL`"
    Give a current SCD2 row a far-future `valid_to` (`9999-12-31`) so the
    join predicate stays a clean range. The nullable alternative forces
    `valid_to IS NULL OR ts < valid_to`, and that `OR` defeats range-join
    optimization: measured on this schema it made the whole pipeline
    super-linear and ~23x slower at 50k units while returning identical
    numbers. Correctness tests cannot catch it; only wall-clock can.

### Declarative dim sources

The joins above are hand-written SQL: correct, but boilerplate a definitions
author re-derives (and can mis-derive) every time a new fact source needs
`country` or `plan`. `dim_sources:` declares each dimension table once; any
fact source references it by name in `dims:` and the engine generates the
join. It is its own top-level key, so it gets its own `dim_sources.yaml`; the
loader merges every file in the directory either way.

Two logical dim kinds cover three physical shapes:

| kind | physical shape | `validity:` | join | allowed `as_of` |
|---|---|---|---|---|
| plain dim | one row per key | absent | equi (LEFT) | `static` only |
| versioned history | validity ranges | `valid_from` + `valid_to` | equi + half-open range (LEFT) | `pre_exposure` / `event_time` |
| versioned history | changelog | `changed_at` | same range join, ranges windowed from the changelog | `pre_exposure` / `event_time` |

A plain dim, equivalent to the `dim_user` equi-join above:

```yaml
dim_sources:
  - name: users
    sql: SELECT user_id, country FROM analytics.dim_user
    entity: user_id
    properties:
      - name: country
        column: country
        dtype: string
        as_of: static
        description: Signup country, fixed for the unit's lifetime
```

A versioned dim declared as validity ranges, equivalent to the
`snap_user_plan` range join above:

```yaml
dim_sources:
  - name: user_plan
    sql: SELECT user_id, plan, valid_from, valid_to FROM analytics.snap_user_plan
    entity: user_id
    validity:
      valid_from: valid_from
      valid_to: valid_to
    properties:
      - name: plan
        column: plan
        dtype: string
        as_of: pre_exposure
        description: Plan in force as of the row's own timestamp
```

The same history as a changelog (one row per *change*, not per version) is
common when there is no dbt snapshot: a plain
`plan_changes(user_id, plan, changed_at)` log. Declare `validity: {changed_at:
changed_at}` instead of `valid_from`/`valid_to`, and the engine windows it into
ranges itself: `valid_from` is each row's `changed_at`, `valid_to` is the next
change for that key, and the last row is closed with the far-future sentinel.

```yaml
dim_sources:
  - name: user_plan
    sql: SELECT user_id, plan, changed_at FROM analytics.plan_changes
    entity: user_id
    validity:
      changed_at: changed_at
    properties:
      - name: plan
        column: plan
        dtype: string
        as_of: pre_exposure
        description: Plan in force as of the row's own timestamp
```

`orders` referencing both dims shrinks to the bare event query:

```yaml
fact_sources:
  - name: orders
    sql: |
      SELECT 'order' AS event, user_id, ordered_at, total_amount
      FROM analytics.fact_orders
      WHERE status = 'completed'
    timestamp_column: ordered_at
    entities:
      - user_id
    dims: [users, user_plan]
    facts:
      - name: order
        column: total_amount
        description: A completed order, valued at its total amount
```

`country`/`plan` behave exactly as if they were still projected inline:
resolution, `as_of` semantics, and breakout/filter eligibility are unchanged.

!!! warning "Generated joins are LEFT, not INNER"
    A fact row with no matching dim row (a `dim_user` gap) or no in-range
    version (a `snap_user_plan` gap) keeps NULL properties instead of being
    dropped from the metric: it lands in the `__null__` breakout bin, as the
    "declare `source:` on every breakout" rule already documents for a
    missing property. The hand-written `JOIN`s above are INNER, so migrating
    a source to `dims:` can surface units the inline version was quietly
    excluding. That is usually the fix, not a regression, but check the row
    counts, not just the numbers.

The changelog shape closes every open row with the far-future sentinel
automatically. A *declared* validity-ranges dim with a `NULL` `valid_to` on an
open row is also safe (the engine coalesces it to the sentinel before the
join, not inside the join predicate), but writing the explicit sentinel in the
warehouse table remains the recommended convention: it is what every other
reader and tool sees.

!!! warning "A dim's fan-out is checked at query time; upstream constraints are still recommended"
    Both cardinality contracts are verified before the join runs: a plain
    dim's one-row-per-key uniqueness, and a versioned dim's non-overlapping
    validity per key (tied boundaries included). Either violation fans the
    fact table out and doubles every additive metric downstream, so the
    engine refuses rather than joining. Enforcing the constraint upstream is
    still worth doing, because it is what every other reader and tool sees.
    A window-based check on a versioned dim's own table (a pure `max`/`min`
    aggregate cannot express this — it flags history depth, not overlap):

    ```sql
    SELECT user_id FROM (
      SELECT user_id, valid_to,
             lead(valid_from) OVER (PARTITION BY user_id ORDER BY valid_from) AS next_from
      FROM analytics.snap_user_plan
    ) t WHERE next_from < valid_to;  -- next version starts before this one ends
    ```

    A changelog with two changes at the same `(entity, changed_at)` instant
    has no defined ordering — which value the window function picks is
    backend-dependent. Dedupe upstream rather than relying on one backend's
    tiebreak.

??? info "Scope boundaries and naming"
    - **A versioned dim refuses `as_of: static` properties**, even when a
      column in the history table happens to be immutable in practice (an
      original signup country stored alongside a plan history, say). Declare
      that column on a separate plain dim instead, or leave it in raw SQL. One
      dim declares one time axis, so its properties' `as_of` semantics stay
      unambiguous from the declaration alone.
    - **`as_of: pre_exposure` resolves from the *fact source's own events*,
      not from the dim table directly.** Joining `user_plan` onto `orders`
      means a unit's pre-exposure plan is resolved from that unit's own
      pre-exposure *order* rows: the same "latest value strictly before first
      exposure" rule as an inline column, sourced through the dim join. A unit
      with no qualifying event on that fact source has no pre-exposure value
      and lands in `__null__`, even if `user_plan` itself has a well-known
      value at that time. This is the failure mode the "needs rows that
      predate exposure" warning describes; dims do not change it.
    - **Property names share one flat namespace per fact source.** A
      dim-contributed property behaves exactly like an inline one, so it can
      collide like one: two properties named `country` on the same dim, on two
      dims joined to one fact source, or on a dim and the fact source's own
      inline `properties:`, are all refused at load. `name` and `column` are
      independent, so two dims wrapping the same physical `country` column
      declare `name: buyer_country` and `name: seller_country`. That rename is
      global (a dim is declared once and joined into every fact source that
      references it), so aliasing the *same* dim differently per fact source
      is a real, currently unsupported extension, not something to route
      around with clever naming.
    - **When not to reach for `dims:`**: a join that brings in `group_id`
      (assignment-to-exposure, above) needs both sides' filters and columns
      threaded through by hand; a fan-out join (one fact row matching several
      dim rows on purpose) has no LEFT-join-with-a-key-contract shape to
      declare; a multi-column key is not supported (`entity:` is one column).
      All three stay in `fact_source.sql`, which is unconditionally supported.
      `dims:` is sugar on that substrate, not a replacement for it.

### End to end

The calling code does not change: the same constructor and the same `run()`
as [Running it](#running-it), only the definitions differ, and the joins they
carry stay in the warehouse. Against a connection to the six-table warehouse
these definitions describe:

<!-- invisible-code-block: python
from pathlib import Path

Path("definitions").mkdir(exist_ok=True)
Path("definitions/fact_sources.yaml").write_text("""\
dialect: duckdb

fact_sources:
  - name: orders
    sql: |
      SELECT
        'order' AS event,
        o.user_id,
        o.ordered_at,
        o.total_amount,
        u.country,
        p.plan
      FROM analytics.fact_orders o
      JOIN analytics.dim_user u
        ON u.user_id = o.user_id
      JOIN analytics.snap_user_plan p
        ON p.user_id = o.user_id
       AND o.ordered_at >= p.valid_from
       AND o.ordered_at <  p.valid_to
      WHERE o.status = 'completed'
    timestamp_column: ordered_at
    entities:
      - user_id
    facts:
      - name: order
        column: total_amount
        description: A completed order, valued at its total amount
    properties:
      - name: country
        column: country
        dtype: string
        as_of: static
        description: Signup country, fixed for the unit's lifetime
      - name: plan
        column: plan
        dtype: string
        as_of: pre_exposure
        description: Plan in force at the row's own timestamp

  - name: web_events
    sql: |
      SELECT
        e.event_name AS event,
        e.user_id,
        e.event_ts,
        u.country,
        p.plan
      FROM analytics.events e
      JOIN analytics.dim_user u
        ON u.user_id = e.user_id
      JOIN analytics.snap_user_plan p
        ON p.user_id = e.user_id
       AND e.event_ts >= p.valid_from
       AND e.event_ts <  p.valid_to
    timestamp_column: event_ts
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
        description: A page view (occurrence-only)
    properties:
      - name: country
        column: country
        dtype: string
        as_of: static
        description: Signup country, fixed for the unit's lifetime
      - name: plan
        column: plan
        dtype: string
        as_of: pre_exposure
        description: Plan in force at the row's own timestamp

  - name: experiment_exposure
    sql: |
      SELECT
        'exposure' AS event,
        x.user_id,
        x.experiment_id,
        a.variant AS group_id,
        x.exposed_at
      FROM analytics.fact_exposure x
      JOIN analytics.fact_assignment a
        ON a.user_id = x.user_id
       AND a.experiment_id = x.experiment_id
    timestamp_column: exposed_at
    entities:
      - user_id
    facts:
      - name: exposure
        column: null
        description: The unit reached the surface the experiment changed
""")
Path("definitions/exposures.yaml").write_text("""\
exposures:
  - name: checkout_trigger
    fact: exposure
    description: The first trigger inside the window enrolls the unit.
""")
Path("definitions/metrics.yaml").write_text("""\
metrics:
  - type: conversion
    name: conversion_rate
    description: "Did the unit order within 14 days of exposure?"
    entity: user_id
    preferred_direction: increase
    fact: order
    window_days: 14
""")
Path("definitions/experiments.yaml").write_text("""\
experiments:
  - name: checkout_redesign
    description: "One-page checkout vs the current flow"
    exposure: checkout_trigger
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    control_group: control
    plan:
      secondaries:
        - conversion_rate
    breakouts:
      - property: country
        source: web_events
      - property: plan
        source: web_events
""")

import datetime as dt
import random

import ibis
import pyarrow as pa

random.seed(11)
START = dt.datetime(2025, 1, 15, 9, 0)
FOREVER = dt.datetime(9999, 12, 31)  # SCD2 sentinel, never NULL
UPGRADE_AT = dt.datetime(2025, 3, 1)  # every upgrade lands after the window

dim_user, snap_user_plan, assignment, exposure, orders, events = [], [], [], [], [], []
for i in range(400):
    user = f"u{i:04d}"
    variant = random.choice(["control", "treatment"])
    dim_user.append({"user_id": user, "country": random.choice(["US", "GB", "DE"])})

    # Plan history. Everyone starts free; ~30% upgrade, but only ever after
    # the experiment window has closed.
    upgrades = random.random() < 0.30
    snap_user_plan.append(
        {
            "user_id": user,
            "plan": "free",
            "valid_from": START - dt.timedelta(days=200),
            "valid_to": UPGRADE_AT if upgrades else FOREVER,
        }
    )
    if upgrades:
        snap_user_plan.append(
            {"user_id": user, "plan": "pro", "valid_from": UPGRADE_AT, "valid_to": FOREVER}
        )

    assigned_at = START + dt.timedelta(days=i % 5)
    assignment.append(
        {
            "user_id": user,
            "experiment_id": "checkout_redesign",
            "variant": variant,
            "assigned_at": assigned_at,
        }
    )
    # Pre-experiment browsing: an as_of=pre_exposure lookup needs rows that
    # predate first exposure, or it has nothing to resolve.
    events.append(
        {
            "user_id": user,
            "event_ts": START - dt.timedelta(days=random.randint(30, 60)),
            "event_name": "page_view",
        }
    )
    # Only ~70% of assigned units trigger, and triggering never depends on arm.
    if random.random() < 0.70:
        exposed_at = assigned_at + dt.timedelta(hours=1)
        exposure.append(
            {"user_id": user, "experiment_id": "checkout_redesign", "exposed_at": exposed_at}
        )
        if random.random() < (0.34 if variant == "treatment" else 0.25):
            orders.append(
                {
                    "user_id": user,
                    "ordered_at": exposed_at + dt.timedelta(days=random.randint(0, 13)),
                    "total_amount": round(random.uniform(5.0, 60.0), 2),
                    "status": "completed",
                }
            )

# The warehouse keeps loading long after the experiment ends: past every
# unit's window close (so nothing is censored for want of observable data),
# and past the plan upgrades above. Those post-upgrade rows are the point:
# they are in the source, and a pre-exposure lookup must ignore them.
AFTER = dt.datetime(2025, 3, 10)
for i in range(400):
    events.append({"user_id": f"u{i:04d}", "event_ts": AFTER, "event_name": "page_view"})
for d in range(5):
    orders.append(
        {"user_id": f"u{d:04d}", "ordered_at": AFTER, "total_amount": 9.99, "status": "completed"}
    )

con = ibis.duckdb.connect()
con.create_database("analytics")
for name, rows in [
    ("dim_user", dim_user),
    ("snap_user_plan", snap_user_plan),
    ("fact_assignment", assignment),
    ("fact_exposure", exposure),
    ("fact_orders", orders),
    ("events", events),
]:
    con.create_table(name, pa.Table.from_pylist(rows), database="analytics")

import pyarrow.parquet as pq

from increment import Analysis

analysis = Analysis.from_definitions("checkout_redesign", "definitions/", con)

for r in analysis.run():
    print(f"{r.metric} / {r.group_id}: lift={r.lift.value:+.2%}")

segments = analysis.run_breakout()
plan_segments = sorted({b.dimension_value for b in segments if b.dimension == "plan"})
country_segments = sorted({b.dimension_value for b in segments if b.dimension == "country"})
print(f"plan segments:    {plan_segments}")
print(f"country segments: {country_segments}")

# `plan` resolves as of exposure, so the later upgrade to "pro" never reaches
# the breakout, even though those rows are right there in the source.
# Declaring it `static` instead would pull them in and split the segment.
assert plan_segments == ["free"]

# A property missing from a breakout's source lands every unit in the null
# segment instead of failing, so the absence of that bucket is the check.
assert "__null__" not in country_segments

analysis.export("moments.parquet")
scanned = sum(
    int(con.table(t, database="analytics").count().execute()) for t in ("fact_orders", "events")
)
print(f"warehouse rows scanned: {scanned:,}")
print(f"moment rows returned:   {pq.read_metadata('moments.parquet').num_rows}")
-->

<!-- output -->
```text
conversion_rate / treatment: lift=+53.33%
plan segments:    ['free']
country segments: ['DE', 'GB', 'US']
warehouse rows scanned: 894
moment rows returned:   2
```

The cube is one row per metric per arm (2 rows here) no matter how much the
warehouse scanned to produce it (894 rows in this fixture), and that shape is
the same at a billion. `examples/realistic_demo/scale_sweep.py` measures that
directly across partition counts, and `examples/data_model.py` renders the
whole readout, including the sample-ratio check on the triggered population.
That population is assignment-valid in this fixture because triggering is
counterfactual and arm-independent, and assignments are independent with a
constant 50/50 probability.

!!! note "Why the `plan` breakout is single-valued here"
    Every upgrade in this fixture lands after the experiment window, so a
    correct pre-exposure resolution can only ever return `free`. This page
    runs as part of the test suite and asserts exactly that; re-declaring the
    property as `as_of: static` makes it fail with `['free', 'pro']`, which is
    the failure mode the annotation exists to prevent.

## Next step

Choose an estimator with [Which experiment analysis method should I use?](choose-a-method.md), or continue to the [warehouse analysis example](../examples/analysis_from_a_warehouse.md).
