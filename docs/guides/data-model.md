# Warehouse-native experimentation data model

Increment’s warehouse path turns declared fact sources, dimensions, exposures, metrics, and experiments into a canonical unit-day panel and inspectable Ibis queries. Definitions live in version-controlled YAML and execute against the warehouse backend you configure.

## One row per unit per day

A unit's clock starts at its **first exposure**. Raw exposure events are
deduplicated to one row per unit (the earliest qualifying event inside the
experiment window), carrying the unit's single group assignment and
`first_exposure_ts`. Units that appear in more than one real group (mixed-group
assignment), or that have a NULL assignment, are dropped.

Assignment integrity is checked before each warehouse readout. A unit seen in
more than one real arm is counted as `(mixed assignment)`; a unit seen only
with NULL, or with both a real arm and NULL, is counted as `(unassigned)`.
These are accounting entries, not arms: NULL never receives an estimate or an
SRM degree of freedom. The `on_mixed_assignment` policy governs both
conditions: the default `"error"` refuses, while `"warn"` and `"exclude"`
remove them and report each count separately in diagnostics.

From each unit's first exposure, the panel lays out one row per day:

| column | meaning |
|---|---|
| `unit_id` | the randomisation unit |
| `group_id` | the unit's arm |
| `first_exposure_ts` / `first_exposure_date` | when the unit's clock started |
| `day_offset` | days since first exposure: 0, 1, 2, ... |
| `ds` | the calendar date, `first_exposure_date + day_offset` |

Two properties of this spine matter:

- **Day offsets are unit-relative.** Day 3 for a unit enrolled on
  January 15 is January 18; for a unit enrolled on January 20 it is
  January 23. Metric windows (`window_days`, `threshold_days`) count in
  these offsets, so every unit gets the same-length observation window
  regardless of when it enrolled.
- **The spine is dense.** Every day from exposure to the panel's end
  exists as a row, whether or not the unit did anything that day. A day
  with no events is a zero, not a missing row; absence is data, and it
  has to participate in means and variances.
- **Dates are local to the declared day boundary.**
  `first_exposure_date` is the exposure timestamp's calendar day under
  the experiment's `day_boundary` — a fixed UTC offset, default
  `"UTC"` — and every `ds` is date arithmetic on that local day. See
  [Metric types](metric-types.md#how-days-are-counted-day-0-and-the-day-boundary)
  for how to choose and declare the boundary.

The spine runs to the experiment's observation horizon: `observation_end` if
declared, otherwise `end`, or, for a running experiment, the latest observed
event date. Independently, each metric is bounded by its own fact's
**`data_as_of`**: the latest date that fact has actually loaded.
A unit whose window would close beyond the observable bound is censored rather
than scored on partial data. When the drop is material, a warning reports the
count.

The dataframe path (`Analysis.from_unit_panel`, see the
[Quickstart](quickstart.md)) accepts this same one-row-per-unit-per-day
shape directly and zero-fills it to a dense spine internally.

## The definitions directory

The warehouse path is driven by declarative YAML. A definitions directory
is any directory of `.yaml`/`.yml` files; the loader reads and merges all of
them, so how you split content across files is up to you. The
conventional split:

```
definitions/
  fact_sources.yaml   # SQL relations that expose event tables
  dim_sources.yaml    # dimension tables joined into fact sources on demand
  exposures.yaml      # what marks a unit as enrolled
  metrics.yaml        # metric derivations (mean, conversion, ...)
  experiments.yaml    # experiment schedules and metric lists
```

Loading validates everything up front: YAML syntax, model fields (unknown
fields are rejected at every level), fact-source SQL syntax, duplicate names
across files, and every cross-reference. An experiment's exposure must be a
defined exposure, every metric it lists must be a defined metric, and every
metric's entity must match the experiment's unit. Failures raise a
`DefinitionError` naming the offending file.

Fact-source and dim-source SQL must not be blank: an empty or whitespace-only
`sql` refuses at load with `definition.source.sql_empty` (context
`source_kind`, `source_name`), before any warehouse access. Every SQL value
must be exactly one read-only `SELECT` query. Plain
selects, read-only CTEs, and set operations are allowed; statement batches,
DML, DDL, `SELECT ... INTO`, and locking reads are rejected. The loader parses
definitions as defense in depth, but definitions remain trusted, reviewed
code: review every query before deployment and use read-only production
warehouse credentials. Report population SQL is checked again immediately
before it reaches the backend.

The rest of this guide builds a minimal working directory, file by file. Only
`dim_sources.yaml` is optional; it earns its place once attributes live in
dimension tables ([Declarative dim sources](#declarative-dim-sources)).

## `fact_sources.yaml`: where events live

A fact source is a SQL relation (table, view, or query) whose rows are
events. It declares which columns mean what:

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

- `dialect` identifies the dialect in which the source SQL is written. If the
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
- `timestamp_column` is the event timestamp every window is measured
  against.
- `entities` lists the unit-identifier columns the relation carries. An
  experiment's `unit` must be among them.
- **Facts** are the events you aggregate into metrics. The relation must
  carry an `event` column whose value names each row's fact. A fact's
  `column` is its value column (`revenue` for a purchase); `column: null`
  declares an occurrence-only event: counting, conversion, and retention
  work on it, but value aggregations (`sum`, `avg_event`,
  `avg_calendar_day`, ...) do not.
- **Properties** are unit attributes you filter and break out on. Each
  carries an `as_of` annotation declaring *when* the value is measured:
  `event_time` (the default, free to change during the experiment, and
  therefore rejected as a breakout dimension), `pre_exposure` (resolved to
  the latest value strictly before the unit's first exposure), or
  `static` (asserted constant for the unit's lifetime, e.g. signup
  country). This is a correctness annotation, not metadata; segmenting
  on a value measured after exposure biases the result.

## `exposures.yaml`: what enrolls a unit

An exposure defines what marks a unit as enrolled. It is either
**fact-based** (the first occurrence of a fact within the experiment
window, optionally restricted by filters) or **query-based**: explicit
SQL returning `unit_id`, `ts`, and `group_id` rows (plus an optional
`experiment_id`), for teams with a dedicated enrollment table. Exactly one of `fact` or `sql` must be set.

```yaml
exposures:
  - name: first_page_view
    fact: page_view
    description: >
      The first page view inside the experiment window marks the unit
      as enrolled.
```

For a fact-based exposure, the fact source's rows also supply the arm
assignment: the enrolling event row carries `experiment_id` and
`group_id` columns, and the unit's first qualifying row decides both its
enrollment time and its arm.

!!! warning "SRM requires an assignment-valid population"
    `srm()` compares these enrolled counts with the planned allocation. That is
    an allocation diagnostic only when enrollment is recorded at assignment or
    from a pre-treatment, arm-invariant eligibility/exposure signal. If the
    treatment can change whether a unit reaches or logs this event, imbalance in
    the enrolled subset may be a treatment-induced selection effect rather than
    a randomizer failure. Run the allocation check on all targeted units and
    treat the subset ratio as a separate selection/telemetry diagnostic.

!!! note "SRM result fields"
    `SRMResult.p_value` has been removed. Pearson fixed-look evidence is now
    available as `SRMResult.fixed_p_value`; `is_srm` uses the evidence selected
    by `inference`. Anytime-valid adapters require `expected` or
    `design.allocation`; declared arms absent from a cumulative prefix retain
    zero counts. Fixed inference that omits expected/design allocation infers
    equal shares from observed arms and returns `log_e_value=None`; Pearson
    remains the selected evidence. Anytime-valid inference requires cumulative
    prefixes whose assignment units have the same known conditional arm
    probabilities at every assignment; independent categorical randomization
    suffices. Static marginal shares alone do not: blocked, adaptive, dependent,
    quota, exact-balance, and without-replacement protocols need fixed or
    protocol-specific inference. `inference="fixed"` is for one predeclared look.

## `metrics.yaml`: what to measure

Metrics reference facts and reduce each unit's events to one number. Four
types (`mean`, `conversion`, `retention`, `ratio`) are covered in depth in
[Metric types](metric-types.md). Two are enough here:

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

For heavy-tailed mean outcomes, add `winsorization` to the mean metric. Its
typed `inference` specification persists through frame construction, native
definitions, artifact context, and result JSON. Positive percentile inference
defaults to `positive-log-kernel-bootstrap-t-v1`; explicit
`joint-rank-projection-v1` retains the uniform rank method. Method choice and
seed/stream must be set before examining inferential results. The bootstrap's
typed public status is `experimental`, and its persisted qualification is
`pointwise_asymptotic_model_conditioned_v1`: a pointwise asymptotic candidate
under its iid positive smooth-density model, not a universal finite-sample or
heterogeneous-effect claim.
The full historical calibration manifest remains unresolved. It records 12
known alternative variance failures plus contamination width-failure evidence;
bounded normal/binomial diagnostics do not turn these into passes. Adaptive-winsor
variants remain experimental and excluded from stable admission. The rank
method's qualification is `uniform_support_conditioned_v1`; it permits zeros
and requires externally justified support. Bootstrap requires strictly positive
outcomes. Percentile cutoffs are pooled across all eligible arms before the
transformed per-unit values are aggregated. Fixed values use the declared cap
directly. The same cutoff is applied to every arm, and the estimate concerns the
capped outcome.

`window_days` counts in day offsets from each unit's own first exposure;
the panel concept above is what makes a "14-day window" mean the same
thing for every unit. Metric names are globally unique across the whole
directory, so an experiment can reference any metric by name.

## `experiments.yaml`: tying it together

An experiment binds an exposure, a unit, a metric list, and a schedule:

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

- `start` / `end` bound **enrollment**, as whole days at the experiment's
  `day_boundary` (see [Window edges and the day boundary](#window-edges-and-the-day-boundary)).
  Leave `end` unset for a running experiment.
- `observation_end` optionally extends data collection past `end`. Set it
  to `end + <longest metric window>` when the intervention is delivered
  once at enrollment (a new onboarding flow, a welcome email), so units
  enrolled near the end still get a complete window instead of being
  censored. Leave it unset for a continuous intervention that reverts
  when the experiment stops.
- `control_group` is required: the engine never guesses which arm is
  control.
- `design` declares a non-default identification mechanism
  (`Encouragement`, `Observational`) alongside `control_group`; see
  [encouragement.md](encouragement.md#declare-on-from_definitions) and
  [observational.md](observational.md).
- The `plan` block declares the decision rule: `primary` (the
  pre-registered outcome), `secondaries` (reported alongside, sharing one
  multiplicity family), and `guardrails` (tracked separately, tested
  one-sided on their adverse side). It also carries `alpha`, `q`, and
  `inference`.
- An experiment can also declare `n_pre_periods` (a CUPED lookback in
  days, see [CUPED](cuped.md)) and `breakouts` (per-segment lift
  breakdowns on a declared property).

### Window edges and the day boundary

`start`, `end` and `observation_end` name **days at the declared
`day_boundary`**, whatever spelling declares them:

- A value without a UTC offset (`2025-01-15`, `2025-01-15T09:00:00`) is
  wall-clock time at the boundary, so its written date is the day.
- A value with an offset (`Z`, `-05:00`) is converted to the boundary first,
  and then its date is the day. Two spellings of one instant always declare
  the same window.

Worked example at `day_boundary: UTC-05:00`:

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
state created before this rule was introduced must be republished or
re-registered.

### Per-metric method and prior bindings

Every entry in a `plan` role (`primary`/`secondaries`/`guardrails`) is either a bare metric NAME
(shorthand -- reports under the call's own defaults) or a BINDING that
overrides `decision_method`/`sensitivity_methods`/`prior` for that one metric
only:

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

A call-wide role-aware `run(decision_method=..., sensitivity_methods=...)`/`run(prior=...)` still wins over a
declared binding for every metric it names -- the binding only fills in
when the call leaves that field unset. This precedence is per-metric,
independently: one `run()` call may dispatch several metrics under
different adjustment sets and different priors at once. See
[CUPED](cuped.md#per-metric-method-bindings) for the full precedence
rules and the dataframe path's
`MetricSpec(decision_method=..., sensitivity_methods=..., prior=...)`
counterpart.

## Randomization grain vs analysis grain

`unit` is the **analysis** grain: one row per unit feeds the moments. When
assignment happened at that same grain, nothing more to declare. But many
experiments randomize something *coarser* than what they measure -- stores
are flipped while users are measured, geos are flipped while sessions are
measured. Units inside one randomization cluster share its assignment (and
usually its idiosyncrasies), so they are **not independent**: treating them
as such understates the SE by roughly `sqrt(1 + (m - 1) * rho)` (`m` =
average units per cluster, `rho` = intra-cluster correlation). At 20 users
per store and `rho = 0.5` the reported interval is ~3x too narrow.

Two honest ways to declare this:

- **The cluster IS what you measure** -- revenue *per store*: declare
  `unit: store_id` and stop. Randomization and analysis share a grain, the
  plain per-arm variance is valid.
- **Coarser randomization, finer measurement** -- revenue *per user*,
  randomized by store: keep `unit: user_id` and declare the grain:

```yaml
experiments:
  - name: store_rollout
    exposure: enrolled
    unit: user_id
    cluster: store_id   # column on the exposure source
    plan: {secondaries: [revenue_per_user]}
    control_group: control
```

With `cluster` declared (or `from_unit_summary(..., cluster="store_id")` on
the dataframe path), uncertainty is computed from independent clusters.
Randomized moments and observational `Method(name="unadjusted")` with disjoint
arms use a joint relative confidence set with a working t reference at
`min(K_T - 1, K_C - 1)` degrees of freedom. The additive interval uses its own
Welch degrees of freedom. For observational clusters spanning both arms,
unadjusted inference retains the cross-arm covariance with an asymptotic Normal
reference. IPTW/DML/AIPW also use an asymptotic Normal score reference. Each
path counts distinct cluster labels across the contrast once; for disjoint
arms this equals `K_T + K_C`.
If the estimated variance of the relative contrast at its observed point is zero,
the result keeps that point but withholds relative bounds and decisions with
`relative_unavailable_reason="zero_relative_variance"`. Any available additive
interval remains on the row. Zero sample variability is not a certificate of
zero population uncertainty; other singular Fieller geometries remain supported.
Such a secondary remains a non-rejection in its family's declared roster, so it
does not inflate sibling discoveries or prevent usable siblings from reporting.

For estimators that do not refit nuisance models (including randomized
moments and IPTW), declaring a cluster changes only the SE and reference.
DML/AIPW cross-fitting instead uses cluster-atomic folds, so a non-singleton
cluster declaration can change
nuisance predictions and the point estimate; singleton clusters deliberately
fall back to the unit-ID folds and preserve the unclustered point estimate.

What to expect, and what refuses:
- **Independent information comes from clusters, not rows.** Cluster count
  sets the working reference but does not establish calibration. Below 40
  clusters the readout warns that variance is noisy; each enrolled arm needs
  at least two clusters. Asymptotic justification also requires that no cluster
  dominate the arm's variance: a store carrying half its traffic can undermine
  the approximation even when the total count exceeds the warning threshold.
- **Ratio metrics are supported.** A clustered row already IS a ratio row
  over clusters, so a ratio metric needs no new wire fields: its
  denominator family carries that metric's OWN per-cluster denominator
  total `den_j` instead of the cluster size `m_j`, and the estimand is
  `sum(num_j) / sum(den_j)` -- exactly the ratio the unclustered readout
  reports. The cluster size is not an input to that delta method, so
  nothing is lost by the rebinding. Which reading applies is driven by
  the metric's declared `type`, never guessed from the row. Revenue per
  session in a market test is the motivating case. **Quantile metrics
  still refuse**: they do not decompose over cluster moments at all.
- **Total grain only.** Daily/as-of/cohort views, breakouts, and factors
  refuse: none of them carries a cluster-robust variance, and silently
  reporting unit-grain uncertainty there is exactly the bug this feature
  removes.
- **Observational designs are supported on the adjusted estimators.** A
  phased geo rollout is observational AND clustered: declare both, and
  IPTW/DML/AIPW group per-unit influence contributions by cluster, taking
  the variance over cluster totals (Liang-Zeger) against an asymptotic Normal
  reference, with two clusters per pure arm and a below-forty advisory. IPTW's point estimate is unchanged;
  DML/AIPW can change theirs when cluster-atomic folds alter nuisance
  predictions. Singleton clusters use the unit-ID fold fallback, preserving
  their unclustered point estimates. Decision stats (`chance_to_beat`,
  `prob_favorable`) refuse on every clustered row: sampling intervals do not
  provide a Normal posterior; read the interval directly. See the
  [observational guide](observational.md).
- **Encouragement designs are supported too (ITT and the additive LATE).**
  Declaring both `cluster` and an `Encouragement` design switches the ITT
  row and the ADDITIVE `late` row to the same cluster-robust ratio delta
  method, with component-specific Welch references and the structural
  two-clusters-per-arm requirement; the
  `compliance` (first-stage) row reports on the absolute scale directly.
  **CUPED, ratio metrics, and the complier-RELATIVE `late` row still
  refuse** -- the clustered collapse carries no per-unit covariate
  moments, the LATE first stage already reads cluster SIZES out of the
  denominator family a clustered ratio metric would need for its own
  denominator, and the relative LATE needs a moment family this
  reduction does not build. See the
  [encouragement + CUPED guide](cuped.md#clustered-encouragement-designs).
- **Encouragement exports preserve cluster grain** through the versioned
  compliance payload, including member counts and centered uptake/size moments.
  Ordinary clustered exports from `from_definitions`, `from_unit_summary`, and
  `from_unit_day_artifact` refuse with `source.moments.cluster_grain` before
  writing: their moments wire format has no cluster marker. Analyze those sources
  directly instead; artifact-backed direct analysis retains its cluster evidence.
- **Labels must be clean**: a null cluster label refuses (no
  silent-absorption knob). Arm purity is required only for randomized and
  encouragement designs: a label appearing in both arms refuses as a bucketing
  bug. Observational dependence clusters may span treatment values.
- **CATE and targeting consume unit rows with cluster metadata.** Their fitting,
  honest holdouts, selection folds and uncertainty retain the declared dependence
  grain. Choose `cluster_weight="member_count"` or `"equal"` consistently across
  the workflow, and declare `intervention_grain="cluster"` for whole-cluster
  deployment. These APIs use the source's `unit_frame`, rather than the collapsed
  arm moments described above. See [conditional effects](metric-types.md#conditional-effects-cate).
- **The SRM check moves to the cluster grain.** Randomization happened over
  clusters, so `srm()` tests the DISTINCT CLUSTER count per arm
  (`SRMResult.grain == "cluster"`). Independently assigned clusters with known,
  constant conditional arm probabilities satisfy the anytime-valid contract;
  blocked or exact-balance cluster protocols do not. Fixed-look Pearson
  inference remains available as an explicit option. Per-arm unit counts still
  ride along on `SRMResult.unit_counts` as context -- unequal cluster sizes make unit counts
  drift for reasons the randomizer never controlled, so that imbalance is
  reported, never tested. The dataframe path
  (`from_unit_summary(..., cluster=...)`) reports the same two grains.

## Canonical unit-day artifacts

Definitions-backed analysis has two distinct persistence boundaries:

- `analysis.materialize()` rebuilds a session-local realization of the unit-day
  core from current warehouse data (TEMP relations under the current connection).
  Every call rescans the source unless `store="none"` or no metrics are declared.
  Likewise, qualifying readouts rebuild per call: `"always"` from the first call,
  `"auto"` from the second. Nested reductions reuse that operation's realization,
  not an earlier call's snapshot. This is not portable artifact publication or
  a guarantee of an atomic snapshot across multiple source tables.
- `analysis.publish_unit_day_artifact(store, extensions=(), refresh_of=None)`
  builds the canonical core once and publishes an immutable format-1 artifact
  generation through the store. It returns a `UnitDayArtifactRef`. A later
  process enters at the same boundary with
  `Analysis.from_unit_day_artifact(store, ref, expected_context=...)`; adoption
  performs no upstream fact/exposure build.

Publication captures the required fact, exposure, dimension, trigger, and
extension streams in one source execution. Assignment and cluster validation,
measure rows, freshness, event horizons, and coverage all derive from those
captured inputs. The captured data is held in a warehouse-side `TEMP` table,
so the backend must support temporary-table materialization. Raw relations are
not collected into client memory, and later reductions never fall back to a
live source. Relation writes persist the same captured rows used for their
digest.
A percentile-winsorized metric's own snapshot, every artifact publish's
snapshot, and `dashboard_snapshot`/sequential capture's own pinned read all
scope each captured fact/dimension source to the experiment's enrolled units
and, for the sources whose row is bounded by the analysis window, to
`start - (n_pre_periods + 1)` days onward -- no upper time bound is pushed,
since the freshness watermark and a breakout/factor/covariate property
lookup both need to see rows this bound would otherwise clip -- rather than
capturing the whole source table.

The manifest is written last and contains the artifact/generation identity,
experiment and local day boundary, date coverage, freshness, the mandatory
`exposures` and `measure_stats` relations, metric-to-measure bindings, the
caller-trusted `ArtifactContext`, and any extension descriptors. Every relation
ref is bound to the same artifact generation and semantic role. A refresh keeps
`artifact_id` but creates a new `generation_id`; it must use the
same compiled context as the pinned prior generation.

If an exception is raised after the manifest is published, the publication
context invalidates the generation through `drop_generation` (tombstone first,
below): a fresh store lists nothing for it, and reads
(`open_snapshot`, `verify_relation`) are refused with
`artifact.generation.dropped`. If invalidation itself fails, the original
exception is re-raised carrying a note with the `artifact_id` and
`generation_id`; from a fresh process, run
`store.drop_generation(artifact_id, generation_id)` to finish erasing the
relations. That call is idempotent. An exception before the manifest-index insert
is submitted never made the generation visible, and the store only drops its
relations, attempting every one. That includes a relation whose write failed and
whose immediate cleanup drop also failed. If a drop fails there, an aborting exception
carries a note naming the retained relations, and a publication that ends normally
raises `query.session.warehouse_artifact.publication_cleanup_incomplete` with
`relations`, `artifact_id`, `generation_id` and `route` in its context; drop them
by name. Once the insert has been submitted the store cannot know whether it
committed, and a remote insert can still commit after the call has raised, so the
abort writes the tombstone for the generation first: whether the manifest row
already exists or lands later, it is hidden and refused with
`artifact.generation.dropped`, and only then are the relations erased; erasure
attempts every relation even if one drop fails. Two failures keep relations in the
warehouse. If the tombstone cannot be written, every relation is preserved. From a
fresh process, call `store.abandon_generation(artifact_id, generation_id)` first:
it writes only the tombstone, is durable and safe to repeat, and hides the
generation whether its manifest row exists now or lands later. Then drop the
listed relations by name. If a manifest row exists, `drop_generation` also works.
If only a drop fails after the tombstone, the generation is already hidden and the
failed relations are removed by name. An aborting exception carries a note with the
`artifact_id`, `generation_id` and the retained relation names, qualified by
catalog and schema. A publication that ends normally without a manifest after an
ambiguous insert raises `query.session.warehouse_artifact.publication_state_unknown`
with `relations` (a tuple of qualified names), `artifact_id`, `generation_id`,
`tombstoned` and `route` in its context, chained to the underlying failure.
Do not adopt a generation from a failed publication.

The warehouse store persists each relation before hashing those persisted rows.
Digesting uses bounded Arrow batches and a disk-backed canonical merge sort, so
hashes do not depend on warehouse collation and remain byte-identical to buffered
digests. Streaming `RelationDigests.rows_bytes` is `None`; buffered
`digest_relation()` still includes the canonical row bytes.

Adoption verifies a private temporary copy of each relation and keeps reductions
warehouse-backed. Later external writes to a published table cannot alter that
verified copy; closing the read handle removes its private tables. These bounds
apply to Python digest batches, not total process memory: exposure metadata still
uses one row per unit, warehouse query buffers may grow, and `unit_frame()`
materializes the requested row-level data.

`drop_generation()` on `WarehouseArtifactStore` durably invalidates a
published generation: it appends a tombstone row to a store-owned
control-plane table before touching any relation, and every store sharing
that namespace refreshes visibility on each access, so the generation is
hidden and refused everywhere as soon as the tombstone commits. The
manifest-index row is kept afterward as a cleanup receipt, not erased, so a
fresh process can recover the exact relation names and retry physical
erasure if a prior attempt failed partway. A successful return additionally
means every named relation was erased; a failure during erasure still
leaves the generation durably invalid and safe to retry. A never-published
pair returns immediately with no effect; `abandon_generation` hides such a pair. An already-dropped pair is not a
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

Every adopted artifact exposes these base operations:
`moments_source`, `day_source`, and `export_moments`. The latter exports a
moments cube; it does not re-publish the unit-day artifact. On an adopted
artifact, `materialize()` and `panel_sql()` are refused with
`facade.analysis.operation`, while `summary_sql()` is refused with
`artifact.operation.unsupported`. Optional operations are enabled only by their
published evidence:

| Published extension | Enabled operation(s) |
|---|---|
| `breakout_dimension` | `breakout_source`, `breakout_sources`, `breakout_summaries` |
| `factor_dimension` | `factor_summaries` |
| `site_volume` | `sitewide_evidence` |
| `trigger_population` plus `assignment_counts` | `triggered_source`, `triggered_counts` |

`cluster_identity`, `cuped_preperiod`, and `encouragement_uptake` provide
relation evidence consumed by the corresponding total/day reductions; they do
not invent a generic operation. The extension catalog is closed and immutable:
the request must match exactly one catalog entry, including its canonical
definition and source-provenance hashes. Supported requests are
`breakout_dimension`, `factor_dimension`, `cluster_identity`,
`cuped_preperiod`, `assignment_counts`, `trigger_population`,
`encouragement_uptake`, and `site_volume`.

Logical metrics may share one physical site-volume recipe. Publication writes
that recipe once; each metric retains its binding to the shared evidence when
the artifact is adopted.

Site volume covers the enrollment window `[start, end]` in days at the declared
`day_boundary`, even when `observation_end` extends past `end`; an experiment
without an `end` publishes no site-volume coverage.


The complete artifact refusal taxonomy is:

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

`ArtifactContext` is compiled from an already-loaded `Definitions` model, not
from a path or hidden warehouse I/O. Its `context_format` is `2`, and the same
`context_format` appears inside `canonical_json` so the digest binds it; the two
must agree. Its `canonical_json` is compact RFC 8785 canonical JSON: finite values, no
whitespace, and object keys ordered by UTF-16BE code units. Its `sha256` is
`SHA256(b"increment.unit-day-artifact\x00v1\x00context\x00" +
canonical_json_bytes)`, where `\x00` denotes the actual NUL separator byte.

A format-2 context carries the typed experiment (including the effective
encouragement design), the complete selected metric roster, the dialect and day
boundary, and the extension catalog. It never carries fact, dimension or
exposure SQL: the native source recipe and every extension source recipe are
represented by domain-tagged SHA-256 digests, and a receipt is never executed.
The digests are integrity evidence, not encryption and not a credential vault.
Do not put credentials in SQL literals; hashing does not conceal a secret that
someone can guess, and copies of a format-1 manifest that were already stored
or shared still contain the SQL until an operator handles them.

The context also carries `window_days`, the boundary-local day of each declared
window edge (`start`, `end`, `observation_horizon`), and readers use those stored
days rather than deriving an edge from a timestamp. A format-2 context without
`window_days` is refused by `artifact.context.mismatch` (route: republish from
trusted definitions), before any manifest, relation or pin is used.

Context format 1 is retired. Opening or publishing with a format-1 context is
refused by `artifact.format.unsupported`, naming the received and supported
versions. Republish from trusted definitions to obtain a format-2 artifact; the
refusal never upgrades an old context. Existing artifacts are not deleted and
credentials are not rotated automatically. Relation digest versions are
independent of the context format and remain readable.

Relation schema/content and manifest digests use the same domain root,
role-specific markers, schema bytes, row count, and rows sorted by primary
key. The manifest digest excludes its own `manifest_sha256` field. All digest
strings are lowercase 64-character hexadecimal SHA-256.

The caller-pinned `UnitDayArtifactRef` (artifact UUID, generation UUID,
namespace-qualified manifest locator, and manifest digest) is the trust root.
The store, not Increment, owns authorization for the catalog/schema namespace;
locators outside that namespace or outside the pinned generation are refused.
Opening validates the pinned manifest and then holds one immutable
`ArtifactSnapshot` for verification and every reduction, so later writes or
rebinding cannot mix generations. If both the artifact and the supplied ref
are replaced together, Increment cannot detect that replacement—the caller's
ref distribution and store authorization are the authenticity boundary.

### Moments wire migration

`Analysis.from_moments` accepts centered, complete `moments_format=8`
rows, including a complete embedded `decision_plan`. Fixed-horizon format 7
remains readable; formats 1–6 are refused
with `moments.format.unsupported_legacy`, and a newer format with
`moments.format.unsupported_future`. Re-exporting is the only safe upgrade for a
cube alone: reconstruct the raw metric/experiment definitions and re-export
format 8, or pin the Increment revision that wrote the old cube. `from_moments`
is separate from the unit-day artifact boundary.

The `metrics` declaration selects which metrics to replay. Winsorization
declarations must match selected metrics; an unselected clipped metric does
not require its configuration to replay an ordinary sibling. Format stamps
and wire-schema validation still cover every row of the imported cube.

A fixed-horizon Encouragement source with an empty metric catalog exports one
format-8 `design_summary` envelope instead of outcome rows. It carries the
experiment identity, complete assignment counts and compliance state, and an
empty-metric fixed-horizon decision plan. Reload with `metrics=[]` and the same
design. Missing state, mixed rows, outcome fields, a different identity, or a
nonempty metric catalog are refused; overriding the plan cannot bypass envelope
validation. This envelope is not a sequential checkpoint.

Registered sequential exports use one format-9 `sequential_checkpoint` envelope
carrying the exact snapshot, retained roster, registration (schema version 2,
including the source mapping identity and compliance policy) and parent-linked
record proof. Their embedded compiled plan uses wire version 3. Rounded
fixed-horizon moments and legacy format-8 sequential envelopes cannot resume a
sequential process. See [Sequential inference](sequential-inference.md) for
explicit finalization, idempotent replay and append-only continuation.

## Running it

With the four files above saved under `definitions/`, the whole pipeline is
one constructor and one call. `events` here is a stand-in for the table your
pipelines already load — one row per `page_view` and `purchase`:

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

```text
purchase_rate / T: lift=+12.37%
revenue_per_user / T: lift=+10.64%
```

The connection is bound once at construction; every readout (`run()`,
`run_daily()`, `run_breakout()`, and the rest) reads it from the
instance. To see the SQL a readout would execute without running it, use
`analysis.panel_sql()` and `analysis.summary_sql()`, which return the
per-metric queries as strings. SQL previews and `build_panel_for_metric()`
return live warehouse expressions, not operation-owned TEMP table references;
they remain executable after another readout refreshes materialization.

!!! note "Why the feed must load past the window"
    Each metric is bounded by its fact's `data_as_of`: the latest date
    the fact has actually loaded. If the purchase feed stopped before the
    last enrollee's 14-day window closed, those units would be censored
    out (with a warning) rather than scored on partial data. The
    censoring resolves once more data arrives, not by changing the
    experiment's schedule.

## Binding a normalized warehouse

The example above binds a single flat event log. Most warehouses aren't
shaped that way: attributes live in dimension tables, a subscription plan
has history rather than a current value, and the experiment's arm arrives
from a platform export rather than riding on every event row.

Nothing about that requires reshaping the warehouse first. A fact source is
a **query**, not a table — so the join lives in the definition, and the
warehouse does it. `examples/realistic_demo/` is a complete worked example
(a seven-table generator, definitions, and a notebook); this section walks
the three join shapes it needs and the four rules that bite, first as the
hand-written SQL every backend understands, then as the declarative
`dims:`/`dim_sources:` sugar the shipped YAML actually uses (below, under
"Declarative dim sources") — same joins, same numbers, generated instead
of hand-maintained.

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

`snap_user_plan` is a *snapshot* table — one row per plan version, bounded by
`valid_from`/`valid_to` — rather than a Kimball dimension carrying a surrogate
key per version. That distinction decides where point-in-time resolution
happens. The fact rows here hold only `user_id` and a timestamp, so the range
join above resolves the version at query time and `as_of` controls which
timestamp it resolves against. If your ETL instead stamps a version key onto
the fact rows, that resolution already happened upstream: the column arrives
pre-joined, and `as_of: static` is the honest declaration for it.

The `event` column is synthesized (`'order' AS event`) because an orders
table has no natural discriminator — every row is the same kind of fact. A
behavioural event stream aliases its own column instead
(`event_name AS event`) and declares one fact per value. The full example
below carries such a source (`web_events`), joined to the same two
dimensions as `orders` for the reason the third rule explains.

The arm is its own join. Assignment (every bucketed unit) and exposure (the
subset that actually reached the surface) are separate tables, so the
exposure source joins them to pick up `group_id`:

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

Exposure rows may repeat per unit — platform exports usually log every
evaluation — and the layer deduplicates to the earliest qualifying one.
Units that never trigger simply never appear, which is what makes the
analysis a triggered comparison rather than an intent-to-treat one.

### Four rules that bite

!!! warning "`as_of: pre_exposure` needs rows that predate exposure"
    A pre-exposure property resolves to the latest value strictly *before*
    the unit's first exposure. A source whose rows are all
    post-exposure — an orders table, say — has nothing to resolve, and
    every unit lands in the `__null__` bin rather than erroring. Resolve
    such properties from a source that carries pre-experiment history.

!!! warning "Declare `source:` on every breakout"
    With `source:` omitted, a breakout resolves to the *first* fact source
    that happens to list the property. With the same property on several
    sources — which the next rule forces — that silently picks one for you.
    Name it.

!!! warning "Every metric's own source must carry the breakout property"
    A breakout applies to each metric through *that metric's* source, so a
    property declared on only one source fails validation for metrics
    bound to the others (`skip_missing: true` opts a breakout out instead).
    In the full example below, this is why the behavioural-event source
    repeats the same two dimension joins the `orders` source already
    makes — it isn't redundancy.

!!! warning "Close slowly-changing rows with a sentinel, not `NULL`"
    Give a current SCD2 row a far-future `valid_to` (`9999-12-31`) so the
    join predicate stays a clean range. The nullable alternative forces
    `valid_to IS NULL OR ts < valid_to`, and that `OR` defeats range-join
    optimization: measured on this schema it made the whole pipeline
    super-linear and ~23x slower at 50k units, while returning identical
    numbers. Correctness tests cannot catch it; only wall-clock can.

### Declarative dim sources

The joins on the previous page are hand-written SQL: correct, but the
equi-join and the temporal range join are both boilerplate a definitions
author re-derives (and can mis-derive) every time a new fact source needs
`country` or `plan`. `dim_sources:` declares each dimension table once and
references it by name from any fact source's `dims:` list; the engine
generates the join. It is its own top-level key, so it gets its own
`dim_sources.yaml` under the same conventional split - the loader merges
every file in the directory either way.

There are two logical dim kinds, covering three physical shapes:

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

The same history as a changelog — one row per *change*, not one row per
version — is common when there's no dbt snapshot: a plain
`plan_changes(user_id, plan, changed_at)` log. Declare `validity:
{changed_at: changed_at}` instead of `valid_from`/`valid_to`, and the
engine windows it into ranges itself (`valid_from` = each row's
`changed_at`, `valid_to` = the next change for that key, closed with the
far-future sentinel on the last row):

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

`country`/`plan` behave exactly as if they were still projected inline —
resolution, `as_of` semantics, and breakout/filter eligibility are
unchanged. The one visible difference: **generated joins are LEFT, not
INNER**. A fact row with no matching dim row (a `dim_user` gap) or no
in-range version (a `snap_user_plan` gap) keeps NULL properties instead of
being silently dropped from the metric — it lands in the `__null__`
breakout bin, same as the "declare `source:` on every breakout" rule
above already documents for a missing property. The hand-written `JOIN`s
on the previous page are INNER, so migrating a source to `dims:` can
surface units the inline version was quietly excluding; that is usually
the fix, not a regression, but check the row counts, not just the numbers.

The changelog shape's generated ranges get the "close slowly-changing rows
with a sentinel, not `NULL`" warning above for free — the engine closes
every open row with the same far-future sentinel automatically. A
*declared* validity-ranges dim with a `NULL` `valid_to` on an open row is
also safe (the engine coalesces it to the sentinel before the join, not
inside the join predicate), but writing the explicit sentinel in the
warehouse table remains the recommended convention: it is what every
other reader and tool sees, not just this engine.

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

Two v1 scope boundaries, both deliberate:

- **A versioned dim refuses `as_of: static` properties**, even when a
  column in the history table happens to be immutable in practice (an
  original signup country stored alongside a plan history, say). Declare
  that column on a separate plain dim instead, or leave it in raw SQL. One
  dim declares one time axis, so its properties' `as_of` semantics stay
  unambiguous from the declaration alone.
- **`as_of: pre_exposure` resolves from the *fact source's own events*,
  not from the dim table directly.** Joining `user_plan` onto `orders`
  means a unit's pre-exposure plan is resolved from that unit's own
  pre-exposure *order* rows — the same "latest value strictly before
  first exposure" rule the `as_of: pre_exposure` warning above already
  states, just now sourced through the dim join instead of an inline
  column. A unit with no qualifying event on that fact source has no
  pre-exposure value and lands in `__null__`, even if `user_plan` itself
  has a well-known value at that time. This is the same failure mode the
  "needs rows that predate exposure" warning above describes; dims do not
  change it.

**Property names share one flat namespace per fact source** — a
dim-contributed property behaves exactly like an inline one, which means
it can also collide like one. Two properties named `country` on the same
dim, on two different dims joined to one fact source, or on a dim and the
fact source's own inline `properties:`, are all refused at load. The
escape hatch is the same one hand-written SQL forces via aliasing:
`name` and `column` are independent, so two dims wrapping the same
physical `country` column declare `name: buyer_country` and `name:
seller_country`. That rename is global — a dim is declared once and
joined into every fact source that references it — so a future need to
alias the *same* dim differently per fact source is a real, currently
unsupported extension, not something to route around with clever naming.

**When not to reach for `dims:`**: a join that brings in `group_id`
(assignment-to-exposure, above) needs both sides' filters and columns
threaded through by hand; a fan-out join (one fact row matching several
dim rows on purpose) has no LEFT-join-with-a-key-contract shape to
declare; a multi-column key isn't supported (`entity:` is one column).
All three stay in `fact_source.sql`, which is unconditionally supported —
`dims:` is sugar on that substrate, not a replacement for it.


### End to end

The calling code does not change. It is the same constructor and the same
`run()` as [Running it](#running-it) above — only the definitions differ, and
the joins they carry stay in the warehouse. Against a connection to the
six-table warehouse these definitions describe:

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

```text
conversion_rate / treatment: lift=+53.33%
plan segments:    ['free']
country segments: ['DE', 'GB', 'US']
warehouse rows scanned: 894
moment rows returned:   2
```

The cube is one row per metric per arm — 2 rows here — no matter how much
the warehouse scanned to produce it (894 rows in this fixture), and that
shape is the same at a billion.
`examples/realistic_demo/scale_sweep.py` measures that directly across
partition counts, and `examples/data_model.py` renders the whole
readout, including the sample-ratio check on the triggered population. That
population is assignment-valid in this fixture because triggering is
counterfactual and arm-independent; assignments are independent with a constant
50/50 probability.

!!! note "Why the `plan` breakout is single-valued here"
    Every upgrade in this fixture lands after the experiment window, so a
    correct pre-exposure resolution can only ever return `free`. This page
    runs as part of the test suite and asserts exactly that; re-declaring the
    property as `as_of: static` makes it fail with `['free', 'pro']`, which is
    the failure mode the annotation exists to prevent.

## Next step

Choose an estimator with [Which experiment analysis method should I use?](choose-a-method.md), or continue to the [warehouse analysis example](../examples/analysis_from_a_warehouse.md).
