# Warehouse definitions

Use the [warehouse guide](../guides/data-model.md) for a complete first analysis. This reference covers YAML fields, assignment grain, joins and backend-specific behavior.

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

For a [two-stage offer experiment](../guides/spillover.md), select the experiment by name as usual. Its exposure records each eligible person's assignment, including no offer. `design.mechanism: spillover` adds three table locators: `roster` (eligible people, groups and strata), `offer_counts` (each group's planned counts under every possible policy), and `group_policy` (each group's actual assignment). Only the first two are needed before launch. The primary metric still names an ordinary fixed-window conversion.

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

### `dialect`

The SQL dialect. Prefer the warehouse's native dialect: Ibis translation is best-effort and can change meaning silently. If omitted, SQL is checked against the connection's dialect.

### `timestamp_column`

The event time used for every window. A timestamp without a time zone is read as UTC. When an experiment's exposure, metric, trigger, uptake and CUPED sources, and every source of a unit property taken before exposure (`as_of: pre_exposure`, whether or not the experiment declares it), mix timestamps with and without a time zone, those with one are converted to UTC before any comparison, so results never depend on the connection's session time zone.

### `entities`

Unit identifier columns; the experiment's `unit` must be one of them.

### `facts`

Event types and their values. When facts are declared, the SQL must expose `event`, the timestamp, every entity, and every fact's value column under the declared names. Property-only sources need no `event` column. `column` holds a fact's value; `column: null` marks an occurrence-only fact, usable for counts, conversion, and retention but not value aggregations. A declared column the SQL does not expose refuses with `definition.fact_source.missing_columns`, and SQL the connection cannot resolve refuses with `definition.fact_source.sql_unresolved`.

### `missing` (on a fact)

The one rule for a NULL in the fact's `column`, applied to every metric on the fact, counts, conversion and retention included. `error` (the default) refuses a read whose rows hold one, with `source.native.fact_null_values` naming the fact, the column and the NULL rows: an enrolled unit's rows inside a metric window or retention band (counted from its trigger in a triggered read) or its CUPED window, whole-site volume rows inside the enrollment window, and the rows with a unit a `MetricTrends` trend reads; a read that returns no row refuses all the same. A row the read leaves out, such as one after the date a pinned snapshot's feed is complete through, never refuses. `skip` leaves those rows out of every aggregation; `zero` reads the value as 0 and still counts the row. SQL previews (`panel_sql`, `summary_sql`) and a trend's `to_table()` and `sql()` refuse the same way before they return: under `error` each first runs one read that counts the NULL values among the rows its SQL reads. The SQL they return is not checked again when you run it, and under `error` it reads only the rows with a value, as under `skip`. The values are spelled as on a dataframe `MetricSpec(missing=...)`.

### `properties`

Columns for filters and breakouts; each declared `column` must be in the SQL. `as_of` states when a value applies: `event_time` (default), `pre_exposure` (latest value strictly before exposure), or `static` (asserted unchanged).


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

A row enrolls in the experiment whose `name` equals its `experiment_id`; a
NULL key marks a row that belongs to no experiment, such as a pre-period event.
When the column is absent, or NULL on every row, every row enrolls, on either
kind of exposure. When the rows carry keys but none equals the experiment's
`name`, nothing enrolls and the run refuses with
`source.native.experiment_key_unmatched`, listing up to five keys the rows do
carry.

An experiment whose enrollment counts no unit at all refuses with
`readout.arms.none_enrolled`; one whose units are all in the control arm
refuses with `readout.arms.no_treatment`. An enrolled arm whose units have no
usable outcome for a metric, for example because every unit was censored, is
an unavailable row with `readout.cell.missing_metric_observations`.

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
    `inference`. Both modes require `expected` or `design.allocation`; arms absent
    from a cumulative prefix keep zero counts. No SRM path infers its null
    from observed arms. Anytime-valid inference requires cumulative prefixes
    whose assignment units have the same known conditional arm probabilities
    at every assignment; independent categorical randomization suffices. Static
    marginal shares alone do not: blocked, adaptive, dependent, quota, exact-balance,
    protocols need fixed or protocol-specific inference. `inference="fixed"`
    is for one predeclared look.

## `metrics.yaml`: what to measure

A metric reduces each unit's events to one number. The four types (`mean`,
`conversion`, `retention`, `ratio`) are covered in
[Metric types](../guides/metric-types.md); two are enough here:

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
([Winsorization](../guides/metric-types.md#winsorization)).

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
| `design` | An `Encouragement` or `Observational` design; see the [encouragement](../guides/encouragement.md#declare-on-from_definitions) and [observational](../guides/observational.md) guides. |
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
    days, see [CUPED](../guides/cuped.md)) and `breakouts` (per-segment lift
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
import increment

definition = increment.ExperimentDefinition.model_validate(
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
assert definition.start_day == date(2025, 1, 14)
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
methods and priors across metrics. [CUPED](../guides/cuped.md#per-metric-method-bindings)
covers precedence and the dataframe equivalent,
`MetricSpec(decision_method=..., sensitivity_methods=..., prior=...)`.

??? info "Metric binding defaults and override scope"
    Every entry in a `plan` role (`primary`/`secondaries`/`guardrails`) is either a bare metric NAME
    (shorthand -- reports under the call's own defaults) or a BINDING that
    overrides `decision_method`/`sensitivity_methods`/`prior` for that one metric
    only:

    A call-wide role-aware `run(decision_method=..., sensitivity_methods=...)`/`run(prior=...)` takes precedence over a
    declared binding for every metric it names -- the binding only fills in
    when the call leaves that field unset. A differing override of a metric's
    non-default declared value emits one warning for that setting per `run()`;
    matching overrides and overrides of undeclared defaults do not warn.
    This precedence is per-metric, independently: one `run()` call may
    dispatch several metrics under different adjustment sets and different
    priors at once.

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

Declare the cluster on either input path; the method-specific references are below.

**Warehouse.** `cluster` names a column on the exposure source. Uncertainty is then
computed from independent clusters, and `workflows.srm()` tests the distinct cluster
count per arm (`SRMResult.grain == "cluster"`). Per-arm unit counts ride
along on `SRMResult.unit_counts` as context: unequal cluster sizes make
unit counts drift for reasons the randomizer never controlled, so that
imbalance is reported, never tested.

**Dataframe.** Pass `cluster="store_id"` to `from_frame(..., shape="summary")`.
First aggregate panel data to one row per unit, keeping its cluster label.
Uncertainty is computed from independent clusters, and `workflows.srm()` reports the
same two grains.

What a clustered experiment supports:

### Metrics

Supports `mean`, `conversion`, `retention` and `ratio`, using the metric's own per-cluster totals. Quantiles do not decompose over cluster moments and are refused.

### Views

Supports the whole-window `run` view. Daily, as-of and cohort views, breakouts and factors do not carry a cluster-robust variance and are refused.

### Designs

Supports randomized experiments; observational unadjusted, IPTW, DML and AIPW; encouragement ITT and additive LATE. Under `Encouragement`, CUPED, ratio metrics and complier-relative `late` refuse ([why](../guides/cuped.md#clustered-encouragement-designs)).

### Decision statistics

`chance_to_beat` and `prob_favorable` are unavailable on clustered rows.
A declared prior, including Normal, refuses with `arm.adjustment.cluster_prior`.
This is a cluster restriction, not a general restriction on t references:
eligible unclustered delta-method results can retain a separate working-model
posterior. See [prior support](bayesian-results.md#mutual-exclusions-prior-vs-sequential-inference-prior-vs-cluster).

### Exports

Encouragement exports preserve clustering through the versioned compliance payload. Ordinary clustered exports from `from_definitions`, `from_frame` summaries and `from_unit_day_artifact` refuse with `evidence.cluster_unavailable`; analyze the source directly.

### Labels

Cluster labels must be non-null. Each cluster must occupy one arm in randomized and encouragement designs; observational dependence clusters may span arms.

### Cluster sample and assignment checks

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


### Cluster-specific inference

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
- A mixed-CUPED row whose relative projection is refused for a representability
  or finiteness reason carries `relative_unavailable_reason="relative_unrepresentable"`
  and the exact code in `relative_error_code`; its additive projection and `t`
  interval remain on the row, so an absolute-margin decision is unaffected. A
  relative interval alone that a level cannot represent keeps the row's point,
  `log_mean` and `log_se` and withholds only the bounds, with the refusal in
  `relative_error_code` and no `relative_unavailable_reason`.
- Randomized moments and IPTW do not refit nuisance models: clustering changes only the SE and reference. DML/AIPW use cluster-atomic cross-fitting, so non-singleton clusters can change nuisance predictions and the point estimate. Singleton clusters use unit-ID folds and preserve the unclustered point.
- Clustered ratio metrics reuse the existing fields: the denominator family contains the metric’s per-cluster `den_j`, not cluster size `m_j`. The estimand remains `sum(num_j) / sum(den_j)`, as without clustering. Interpret it from the declared metric `type`, never by guessing from the row.
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
- CATE and targeting use unit rows (`unit_frame`) with cluster metadata, not collapsed arm moments. Fitting, honest holdouts, selection folds and uncertainty preserve that dependence grain. Keep `cluster_weight="member_count"` or `"equal"` consistent throughout; declare `intervention_grain="cluster"` for whole-cluster deployment. See [conditional effects](../guides/metric-types.md#conditional-effects-cate).

## Binding a normalized warehouse

Fact-source SQL can join static dimensions, changing attributes and assignment tables inside the warehouse. `examples/realistic_demo/` supplies a complete generator, definitions and notebook.

The handwritten inner joins below drop unmatched fact rows; generated `dims:`/`dim_sources:` joins keep them. Results agree when every fact has a matching dimension row.

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

#### Plain dimension

- **Shape:** one row per key.
- **Validity:** absent.
- **Join:** LEFT equi-join.
- **Property time:** `as_of: static` only.

#### Versioned history with validity ranges

- **Validity:** `valid_from` and `valid_to`.
- **Join:** LEFT equi-join with a half-open time range.
- **Property time:** `pre_exposure` or `event_time`.

#### Versioned history as a changelog

- **Validity:** `changed_at`.
- **Join:** the same range join, with ranges derived from the changelog.
- **Property time:** `pre_exposure` or `event_time`.

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
    - **One time axis per dimension.** Versioned dimensions refuse `as_of: static`, even for an immutable column. Put that column on a separate plain dimension or in raw SQL.
    - **Pre-exposure lookup uses the fact source’s events, not dimension rows directly.** With `user_plan` joined to `orders`, it selects the latest qualifying order strictly before exposure. No such order means `__null__`, even if the dimension has a historical value.
    - **Property names must be unique within each fact source.** Duplicates within a dimension, across joined dimensions, or against inline `properties:` refuse at load. Logical `name` and physical `column` are independent: use `buyer_country` and `seller_country` for two dimensions exposing `country`. Names belong to the dimension globally; per-source aliases of the same dimension are unsupported.
    - **Use `fact_source.sql` for other joins.** Assignment-to-exposure joins carrying `group_id` need both sides’ filters and columns; deliberate fan-out does not satisfy the dimension key contract; multi-column keys cannot use the single-column `entity:`. All remain expressible in source SQL.

### End to end

The calling code does not change: the same constructor and the same `run()`
as [the warehouse guide](../guides/data-model.md#running-it), only the definitions differ, and the joins they
carry stay in the warehouse. Against a connection to the six-table warehouse
these definitions describe:

??? info "Complete normalized-warehouse example"

    ```python
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
    
    import increment
    from increment.workflows import export_moments
    
    experiment, reader = increment.from_definitions(
        "definitions/", con, experiment_name="checkout_redesign"
    )
    whole_window = increment.View.overall()
    breakouts = increment.View.breakout(())
    result = increment.run(experiment.with_views(breakouts), reader)
    
    for r in result.readout(whole_window):
        print(f"{r.metric} / {r.group_id}: lift={r.lift.value:+.2%}")
    
    segments = result.readout(breakouts)
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
    
    export_moments(experiment, reader, "moments.parquet")
    scanned = sum(
        int(con.table(t, database="analytics").count().execute()) for t in ("fact_orders", "events")
    )
    print(f"warehouse rows scanned: {scanned:,}")
    print(f"moment rows returned:   {pq.read_metadata('moments.parquet').num_rows}")
    ```

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


## Other warehouses

The [warehouse guide](../guides/data-model.md#other-warehouses) summarizes backend coverage. Unit-day artifact verification is restricted to DuckDB, PostgreSQL, Snowflake and BigQuery; publishing or opening an artifact elsewhere refuses with `query.session.warehouse_artifact.verification_unavailable`.

### The unverified-warehouse notice

The first time Increment uses a connection to an unverified backend, it shows
one `UnverifiedWarehouseWarning` naming the backend. The notice changes nothing
in the results and is not recorded in them. It means that operations can behave
differently across backends, so check results against a verified warehouse or a
dataframe before relying on them. A team that has checked its backend can
silence it:

```python
import warnings

from increment import UnverifiedWarehouseWarning

warnings.filterwarnings("ignore", category=UnverifiedWarehouseWarning)
```

### TEMP tables

By default a run captures its source rows in a warehouse TEMP table, so every
read in the run sees the same data. On a connection that cannot create TEMP
tables, the default captured run refuses with
`query.session.snapshot.materialization_failed`. When only source capture needs
TEMP tables, `run(experiment, reader, coherence="live")` reads the sources
directly, and its results are marked live
([Reading results](../guides/reading-results.md) explains what that gives up). Scalar
CUPED can use this live route. Multicovariate CUPED fitting and
categorical-cell acquisition still require TEMP tables; if the warehouse cannot
create them, use `from_frame` for those analyses.

### When Ibis cannot translate an operation

When Ibis has no translation of an operation for the backend, the analysis
stops with `query.backend.operation_unsupported` where that query is reached.
Queries earlier in the run may already have read data; no result is produced.
The refusal's `backend` names the SQL dialect and its `detail` repeats Ibis's
message, which names the operation. Ibis 12, for example, has no Trino
translation of `Hash`, which Increment uses to check assignments; the `detail`
then reads `Compilation rule for 'Hash' operation is not defined`.

Ibis translates each operation with a method named `visit_<Operation>` on the
dialect's compiler class, so you can add a missing one in your own code without
changing Increment. This rule translates `Hash` for Trino with Trino's 64-bit
xxHash; `to_utf8` takes text, which is all Increment hashes:

```python
import ibis
from ibis.backends.sql.compilers.trino import TrinoCompiler


def visit_hash(self, op, *, arg):
    return self.f.from_big_endian_64(self.f.xxhash64(self.f.to_utf8(arg)))


TrinoCompiler.visit_Hash = visit_hash

units = ibis.table({"user_id": "string"}, name="units")
print(ibis.to_sql(units.select(bucket=units.user_id.hash()), dialect="trino"))
```

<!-- output -->
```text
SELECT
  FROM_BIG_ENDIAN_64(XXHASH64(TO_UTF8("t0"."user_id"))) AS "bucket"
FROM "units" AS "t0"
```

<!-- invisible-code-block: python
del TrinoCompiler.visit_Hash
-->

Register the rule in the process that runs the analysis, before `run`;
`ibis.to_sql`, as above, shows whether the dialect now translates an
expression. You own the rule's correctness: Increment does not check what it
computes, and results on that backend stay unverified. Consider contributing
the rule to [Ibis](https://github.com/ibis-project/ibis) so it ships with the
backend. TEMP-table support is the backend's own behavior and cannot be added
this way.

### Event times with a time zone

Increment reads event times that carry a time zone in UTC. On the four verified
backends it converts them to UTC timestamps without a time zone with the
backend's own time-zone conversion, both to bucket events into days and to
compare them with sources whose event times have no time zone. Ibis has no
conversion that keeps the instant and its full precision on every backend, so
on any other backend an analysis whose event times carry a time zone stops with
`query.backend.operation_unsupported`, and the refusal's `detail` reads
`converting event times that carry a time zone to UTC (convert them to UTC
timestamps without a time zone in the source SQL)`. Ibis reads a plain
Databricks `TIMESTAMP` column as carrying a time zone, so ordinary Databricks
tables stop here too; a `TIMESTAMP_NTZ` column has none.

To analyze such sources on that backend, convert the `timestamp_column` in the
source SQL to a timestamp without a time zone that holds the UTC time, using the
warehouse's own conversion. Increment then reads it as UTC, like any timestamp
without a time zone. A `visit_<Operation>` rule, as in the previous section,
does not help here: the conversion is Increment's own, not an Ibis operation.

## Inspect generated SQL

The connection is bound once, by `from_definitions`; every view and workflow
operation reads it from the reader. To inspect the warehouse calculation, call
`increment.query.panel_sql(experiment, reader)` or
`increment.query.summary_sql(experiment, reader)`. They return per-metric SQL
strings. Preview construction does not execute the returned query, but may read
fact NULL counts to enforce the declared missing-value policy.

`summary_sql` organizes the existing calculation into named query stages:

- `inc_eligible_outcome_days`: observed days admitted by the metric's window,
  when this boundary can be named independently.
- `inc_unit_measurements`: one metric value per eligible unit, including the
  existing zero-event and denominator rules.
- `inc_analysis_rows`: outcomes and applicable covariates, uptake or cluster
  totals prepared for aggregation.
- `inc_group_references` and `inc_group_moments`: group reference values,
  followed by counts and centered sums.

Follow these stages to inspect the calculation without tracing the whole query
from its final projection. Shared input branches can retain generated aliases:
naming them separately must not change whether a source is evaluated once or
independently. Stage names and formatting are presentation details, not stable
identifiers to parse in application code.

The output remains a summary for the estimators, not a table of effect
estimates. For example, `n * ref_y + cy1` reconstructs the outcome sum;
`successes` is the exact binary count for conversion and retention metrics.
Optional moment fields remain NULL when unavailable.

`build_panel_for_metric(experiment, reader, metric)` returns the assigned
population's panel, spine and event relations; pass `population="triggered"` for
the triggered population. Native SQL previews use live warehouse expressions,
not operation-owned TEMP tables, so they remain executable after another
readout refreshes materialization. They express the same calculation but are
not a transcript of the runtime's materialized statement sequence. Named query
stages do not promise materialization or faster execution.
