# Metric types

Choose the quantity you want to compare, then declare how each analysis unit
contributes to it. Warehouse definitions reduce events to those values;
dataframes supply them directly.

| What you measure | Type | Units and interpretation |
|---|---|---|
| Revenue per user, sessions per user, average duration | `mean` | Dollars, counts or seconds per unit |
| Whether a user purchased | `conversion` | Proportion; a difference of `0.01` is one percentage point |
| Whether a user returned in a specified day band | `retention` | Proportion; a difference of `0.01` is one percentage point |
| Revenue per order, clicks per impression | `ratio` | Numerator units per denominator unit |
| Median or p99 latency | `quantile` | Recorded outcome units; relative-lift results only |

Relative lift is different: `0.10` means 10% above control, whatever the
metric. Check each result's `value_scale`; some designs report additive
effects instead. Start with explicit types:

```python
import polars as pl

import increment

metric_demo = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(8)],
        "variant": ["control"] * 4 + ["treatment"] * 4,
        "revenue": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        "converted": [0, 1, 0, 1, 1, 0, 1, 1],
    }
)
experiment, reader = increment.from_frame(
    metric_demo,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
)
metric_results = increment.run(experiment, reader).readout()
for row in metric_results:
    print(row.metric, row.group_id, row.lift)
```

Control revenue averages 2.5 and treatment 3.5: an additive increase of 1
per user, or 40% relative lift. Conversion rises from 50% to 75%: **25
percentage points**, or 50% relative lift. These small samples demonstrate
units, not a launch decision.

The example uses Polars; any eager dataframe supported by Narwhals works.
See [Reading results](reading-results.md) for uncertainty and population
qualifications, and [method compatibility](compatibility.md) for source limits.

## Declarations, defaults and missing values

`MetricSpec(name="revenue")` defaults to `type="mean"`. A **bare column name**
instead uses storage dtype: Boolean means `conversion`, floating point means
`mean` even for 0.0/1.0 values, and integers require an explicit type. Declare
the type before seeing results if storage can change; the
[dtype example](#the-dataframe-path-metricspec) shows why.

`value_column` lets a metric name differ from its input column. For a ratio,
name both `numerator` and `denominator`; for a quantile, set `quantile` in
`(0, 1)`. Use the [quantile guide](quantile-metrics.md) for its raw-data needs.
An omitted `preferred_direction` on `MetricSpec` means undeclared, not
“increase”; declare increase, decrease or neutral deliberately.

Missing outcomes default to `missing="error"`. Choose `zero` only when a
missing value means no activity, or `skip` to omit that unit for this metric
only. Skipping needs missingness unrelated to outcome values; it changes
which population is observed. Outcomes cannot be mean-filled
(`frame.metric.missing_impute`). Covariate imputation is a separate
[CUPED policy](cuped.md#missing-covariates).

The sections below give warehouse declarations and type-specific conditions.
Use `shape="summary"` only for already-aggregated, unwindowed unit values;
dated windows and retention need a panel or warehouse events.

## Mean

Aggregate a numeric fact per unit, then average across units. `aggregation`
selects the per-unit reduction: `sum`, `count`, `count_distinct`, `avg_event`,
`avg_calendar_day`, `min`, or `max`. `avg_event` divides the qualifying
event-value sum by the qualifying event count; `avg_calendar_day` divides the
sum by the inclusive admitted calendar-day count, treating eventless days as
zero. Bare `avg` is rejected.

```yaml
- type: mean
  name: avg_session_duration
  description: "Mean session duration (seconds) per user"
  entity: user_id
  preferred_direction: increase
  fact: session_end
  aggregation: avg_event
  window_days: 7
  filters:
    - property: platform
      op: equals
      values: [web]
```

`filters` restrict which events qualify and reference properties declared on
the fact source. `preferred_direction` (`increase`, `decrease`, or `neutral`)
declares which direction is favorable (a latency metric prefers `decrease`)
and drives favorability readouts.

??? info "Inference reference for ordinary means"
    For an absolute contrast of ordinary independent iid means,
    `increment.estimation.infer_independent_mean(control_arm, treatment_arm)`
    accepts centered `ArmStats` from the existing moments producers. It retains
    the component counts and centered sums, uses ddof=1 within each arm and a
    Welch-Satterthwaite reference, and persists its df and components on the result.
    This operation does not apply to adjusted scores, clusters, or estimated-cutoff
    moments; those require their own declared inference contracts.

### Winsorization

Use winsorization when a few extreme values would dominate a mean, and report
the capped outcome it estimates. Definitions-backed sources can winsorize any
mean aggregation; a dataframe `MetricSpec` treats each supplied row as the
unit outcome and sums it before winsorizing.

```yaml
- type: mean
  name: revenue_per_user
  entity: user_id
  fact: purchase
  aggregation: sum
  winsorization:
    upper_percentile: 0.99
```

```python
import increment

increment.MetricSpec(
    name="revenue_per_user",
    type="mean",
    winsorization={"upper_value": 500.0},
)
```

Percentile bounds use the pooled eligible units across arms, with exact linear
interpolation; fixed bounds use the values you declare. Zero is a regular
observation. Results expose each
arm's resolved cap and capped count. With fixed caps, these remain available
when an arm's mean is zero or negative and relative lift cannot be reported.
A difference in cap rates may signal a treatment-driven change in the tail;
it is diagnostic, not an adjustment.

!!! warning "Percentile winsorization has no decision evidence"
    Percentile cutoffs are estimated from the data, so their uncertainty must
    be included in inference. Current percentile-winsorized intervals are
    experimental: `stat_sig` is unavailable, not "not significant." Do not use
    these results to make a significance or family-selection decision.

Prespecify the winsorization method and seed/stream before examining
inferential results.

??? info "Statistical details and limits"
    Winsorization is not supported for conversion, retention, ratio, quantile,
    active, or sitewide metrics. Daily, as-of, and cohort readouts are refused.
    For a single upper percentile, randomized fixed-horizon two-sided unit
    inference defaults to `pooled-size-route-v1`, a provisional size route between
    two constructions of the same estimand. Pools of at least 20,000 units whose
    expected count above the cutoff, `N (1 - q)`, is at least 100 run
    `influence-normal-v1`; every smaller request runs
    `positive-log-kernel-bootstrap-t-v1`. The method that ran is recorded as
    `confidence_set.method` (and on `confidence_set.reference.method`); the request
    keeps the route literal in `confidence_set.raw.inference.method`. The thresholds
    are provisional pending the zero-inflated calibration campaign across the route
    boundary (tracked as `1e0h`); either construction can also be requested
    explicitly through `inference: {method: ...}`.

    Outcomes may be zero. The log-kernel pilot acts on each arm's positive
    outcomes and resamples the arm's zeros with the same probability as any
    positive value, so the estimated zero share, the estimated cutoff and their
    cross term are part of every interval. On strictly positive outcomes the
    construction is identical to the earlier positive-only method, root for root.
    The pooled cutoff must sit above the zero atom: with pooled zero share `pi`,
    `upper_percentile` must exceed `pi`, and the p`q` of all units is then the
    `(q - pi) / (1 - pi)` quantile of the positive outcomes (p99 at 90 % zeros is p90
    of purchasers). A request whose type-7 cutoff has a zero as its lower neighbour
    (the pooled quantile sits in the atom, or interpolates between zero and the
    smallest positive outcome) refuses with
    `estimation.winsor.cutoff_in_zero_atom`; raise `upper_percentile` or use a fixed
    `upper_value`. Negative outcomes refuse with
    `estimation.winsor.pilot_negative_outcome`; an arm with fewer than two positive
    outcomes (including an all-zero arm) refuses with
    `estimation.winsor.pilot_degenerate`. Close to the atom (zero share just below
    `upper_percentile`) bootstrap replicates whose resampled cutoff lands on a zero
    are failed roots, and the interval is reported unavailable with reason
    `bootstrap_replicate_failure` rather than approximated.

    `positive-log-kernel-bootstrap-t-v1` is a full-procedure bootstrap-t: 1,999
    fixed-count resamples from the arm-specific pilot, recomputing the pooled
    cutoff, means, density and full influence studentization in every replicate.
    It requires nondegenerate positive log variances, independent iid arms, and a
    smooth positive population cutoff. Its public qualification is
    `pointwise_asymptotic_model_conditioned_v1`: it is an experimental candidate
    for specified populations, not a universal finite-sample or heterogeneous-
    effects guarantee. Calibration remains unresolved for the preserved stress
    grid, including 12 alternative rows with known clipped-variance failures and
    the contamination width failure evidence; these rows remain unavailable
    evidence, not passes. No external upper-tail bound is required.

    `influence-normal-v1` is the analytic influence-function interval: one pooled
    cutoff, clipped means, the same positive-part log-kernel density at the cutoff
    scaled by each arm's positive share, and the centred empirical influence
    variance over every pool arm (estimated cutoff, estimated zero shares and the
    cross term included), closed with normal quantiles on the log-ratio and
    difference scales. It costs one partition and one density pass: the estimate
    takes 0.002 s for two arms of 10,000 and 0.2 s for two arms of 1,000,000 with
    90 % zeros (0.5 s with 50 % zeros), against 1.1 s and minutes for the bootstrap;
    a public readout at that size spends a further few seconds on raw-outcome
    capture and the per-unit evidence digest. Its qualification is
    `pointwise_asymptotic_influence_v1`: a first-order normal approximation that
    drops the bootstrap-t refinement, which is why the route reserves it for large
    pools with a populated upper tail. A size-routed request carries
    `pointwise_asymptotic_size_routed_v1` until the pool resolves it.

    The typed public status of every method is `experimental`: the descriptive
    confidence set is reported and decision evidence is withheld
    (`evidence.experimental_reference`), as described below.
    Select `inference: {method: joint-rank-projection-v1}` for the optional
    uniform rank confidence set. That method requires an explicit finite lower
    support declaration justified independently of the sample. Certified Decimal
    rank calibration supports at most 64 units in each pool arm; larger arms refuse
    before calibration with `estimation.winsor.rank_size_unsupported`. This limit
    binds computation and is not a statistical sample-size requirement. It does not
    apply to the bootstrap or influence methods.
    All arms contribute to the allocation-weighted cutoff, including arms outside
    the reported contrast. Clustered, CUPED-adjusted, prior-weighted, sequential,
    observational, breakout, factor, one-sided and lower-percentile inference refuse
    before estimation with `readout.metric.percentile_winsorization` (sequential
    requests with `sequential.transform.unpredictable` at source construction),
    whichever winsor method the request names; ratio metrics cannot declare
    winsorization. Multiplicity roles are unaffected: experimental rows carry no
    decision evidence into a family. `run()` supports the fixed-horizon, two-sided
    independent-unit path when the source preserves raw outcomes; transformed
    moments alone cannot reconstruct cutoff uncertainty.
    Every persisted inference method carries `status="experimental"`. Their
    descriptive confidence sets remain available, but none supplies decision or
    family-selection evidence.

    `source.unit_frame(metric, outcome_stage="raw")` returns the exact outcomes
    after missingness handling and before winsorization. The default remains
    `"transformed"`. Dataframe percentile snapshots preserve both views after
    caller mutation. A native mixed readout captures every selected metric in one
    warehouse-side snapshot before data validation and reduction, including ordinary
    metrics alongside percentile metrics. Assigned and triggered populations retain
    their own scope; unused metric sources are not loaded. Adopted unit-day artifacts recover
    raw totals from verified pre-transform measure statistics on their pinned
    snapshot; moments-only artifacts cannot supply this state. The direct
    estimator accepts a typed `WinsorRawState` through
    `raw_outcomes={metric.name: state}`; `raw_state_from_source` constructs it using
    the same source protocol as the public readout.

    When a direct `estimate_lift` call mixes percentile metrics (via
    `raw_outcomes`) with ordinary summary metrics, pass
    `summary_population="assigned"` or `"triggered"` explicitly. Summary rows carry
    `experiment_id` but do not carry trustworthy population metadata, so the
    engine never infers it; the supplied population must match every raw state's
    population, and all rows/states must share one study. This requirement applies
    only to mixed calls; raw-only and ordinary-only calls retain their existing
    contracts. A one-use iterable of summary rows is retained for the ordinary
    portion after this identity check.

    `LiftEstimate.reference_kind == "confidence_set"` identifies the new reference.
    `confidence_set.relative` and `.additive` carry endpoints with
    `status` equal to `finite`, `unbounded`, or `undefined`. Nonfinite endpoints
    have numeric `value=None` and an explicit reason. Use these endpoints even
    when the empirical point is undefined. The discriminated `.reference` stores
    rank error allocations, the bootstrap pilot (positive log centres and the
    arm's zero count), explicit RNG seed/stream, centering targets, SEs and all
    roots (including failed roots), or the influence reference's cutoff,
    cutoff-scaled pooled density and both standard errors. `.cutoff` is a confidence region only for
    rank inference; bootstrap cutoff diagnostics live in the bootstrap reference.
    `reintervalize(alpha)` uses the stored roots or standard errors without
    reading or rerandomizing. Bootstrap tails finer than `2/2000` are unresolved,
    and any failed root makes the interval unavailable. `p_value()` inverts the
    stored bootstrap effect test or the normal pivot of the influence reference;
    rank inference retains its coarse single-level test.

    `WinsorInferenceSpec(seed=1729, stream=0)` records the default PCG64DXSM stream.
    Prespecify distinct stream ordinals for repeated datasets, independently of
    the outcome-sampling generator. Changing the pilot or budget requires a new
    method version. `conditional_permutation_test` in
    `increment.estimation.winsor` separately tests
    `raw_distribution_exchangeability`; its p-value is never an effect interval.

    Optional `support.upper` is a hard outcome bound and observed violations
    refuse. `support.quantile_upper` is an externally justified population cutoff
    bound; observations above it are allowed. `support.control_mean_lower` declares
    a positive lower bound for the control winsorized mean. These declarations
    cannot be chosen from the current sample. Without sufficient tail information,
    small p99 designs can require unbounded rank intervals. This uniform-coverage
    limit does not imply that bootstrap calibration on specified distributions is
    impossible. Finite width or useful power is not guaranteed by either method.

## Conversion

Binary outcome: did the event occur at least once within the window? Each unit
scores 0 or 1, and the metric is the share of units that converted.

```yaml
- type: conversion
  name: purchase_rate
  description: "Did the user purchase within the analysis window?"
  entity: user_id
  preferred_direction: increase
  fact: purchase
  window_days: 14
```

A conversion metric works on occurrence-only facts (`column: null` in the
fact source); it needs the event to exist, not to carry a value.

## Retention

Did the unit return on or after day `threshold_days`? One field declares
either band shape. A bare integer `N` is the **unbounded** band `[N, inf)`:
returned on or after day N, ever. A two-element `[a, b]` is the **bounded**,
half-open band `[a, b)`: returned during days a..b-1. Day 0 is the exposure
day, so a bounded return counts when

$$
\text{first exposure date} + a \le ds <
\text{first exposure date} + b
$$

and the unbounded case drops the upper bound.

```yaml
- type: retention
  name: d7_retention
  description: "User returned at least once during days 7-13 post-exposure"
  entity: user_id
  preferred_direction: increase
  fact: page_view
  threshold_days: [7, 14]  # half-open band [7, 14): a return counts on days 7-13
```

An unbounded band is the same field with a bare integer:

```yaml
- type: retention
  name: ever_returned_d7
  description: "User returned at least once on or after day 7"
  entity: user_id
  preferred_direction: increase
  fact: page_view
  threshold_days: 7  # unbounded band [7, inf)
```

A bounded band closes on the right, so a unit's outcome is **final** at
`first_exposure_date + b` (its *maturity*). An unbounded outcome has no
completion date; it can only ratchet 0 → 1 as more data arrives, so it is
reportable only on the calendar axis (below), not on the cohort
(exposure-indexed) day axis. `window_days` is not a retention field: declaring
it on a `RetentionMetric` raises at construction; both band edges live in
`threshold_days`.

### How days are counted: day 0 and the day boundary

Two conventions decide which day an event lands on, and both are common
sources of "your number doesn't match mine" against a warehouse query or a
product-analytics tool.

**The exposure day is day 0.** The band test is
`ds >= first_exposure_date + threshold_days`, so "d7 retention" means the unit
returned on or after the 7th calendar day *after* the exposure day. A
warehouse query that counts the exposure day as day 1 measures a band shifted
by one day and produces a genuinely different number. When reconciling, check
the indexing convention first. On day 0, warehouse definitions count an
outcome event only when its timestamp is strictly after `first_exposure_ts`,
so an event at that exact instant, such as the order that enrolls a unit
under a first-purchase exposure, counts in neither the outcome nor the CUPED
window (see [Data model](data-model.md#one-row-per-unit-per-day)).

**Days are calendar days at the declared `day_boundary` (default UTC).** The
exposure timestamp is cast to a calendar date at the boundary (plain `"UTC"`
by default, or a fixed UTC offset), and the band is date arithmetic on that
date. Declare it on the experiment, or once as a definitions-level default:

```yaml
day_boundary: "UTC-05:00"
```

The experiment window follows the same rule: `start`, `end` and
`observation_end` name days at the boundary. A value without a UTC offset is
wall-clock time at the boundary, so its written date is the day; a value with
an offset is converted to the boundary first, so two spellings of one instant
open the same window (worked example in
[Data model](data-model.md#window-edges-and-the-day-boundary)).

Only fixed offsets are accepted. Named IANA zones (DST-aware) are not
supported yet: a fixed offset translates portably to every warehouse backend;
a DST-aware zone does not.

Declare `day_boundary` before launch and freeze it. Changing it moves
exposure days and window edges, makes the daily series discontinuous and
invalidates pre/post comparisons across the change. This changes what is
measured, not just how it is displayed.

Choose a boundary near the product's activity trough. For example,
`UTC-09:00` cuts days near 4 AM for an EST-centered organization. A fixed
offset drifts by an hour against local time across DST, but its buckets
stay deterministic.

A user exposed ten minutes before the boundary gets a ten-minute day 0;
day 1 is the first full observation day. Calendar windows therefore differ
by up to a day across users, regardless of the boundary. This matters greatly
for d1 retention; an elapsed-24h window would be needed to remove it and is
not supported. Tools using a different day boundary can report different
numbers.

Each segment's own lift remains internally valid under an arm-symmetric
convention. Comparing effects across timezone-correlated segments can change
meaning when their local days are sliced differently. With a trough-anchored
boundary this is second-order at d7 and beyond.

### Two series, one metric

Retention has two readouts, and they answer different questions:

- Whole-window and breakout views compare each unit's **mature, whole-window**
  outcome, arm to arm: the windowed decision estimand. Unbiased, and the
  number a ship/no-ship call should be made on.
- As-of values and as-of lift views return a **cumulative monitoring series**
  indexed by calendar date: each date's value blends whatever tenure every
  unit has reached by that date. It starts earlier (as soon as a unit clears
  `threshold_days`, rather than once its whole band closes) and carries much
  less sampling noise per point, which is what makes it useful for watching a
  running experiment.

!!! warning "The monitoring series is not a decision estimate"
    The tenure mix changes every day, so consecutive points are not comparable
    as a time series, and the series can diverge from `run()`'s decision
    estimate whenever the arms return at different *times* rather than
    different *rates*; even a true whole-window lift of zero can show up as a
    large early divergence. Pass `completed_windows_only=True` to gate on full
    maturity instead: a maturing decision estimate that only counts units
    whose band has fully closed.

### Peeking at the monitoring series

Looking at the series repeatedly and acting on it requires a registered
sequential process declared before any data is read; see
[Sequential inference](sequential-inference.md) for the construction and
replay API.

Reveal all relevant outcomes for a unit together after the longest required
window, in an outcome-independent order. A mature band or
`completed_windows_only=True` alone does not prove finalization: native and
panel sources require `capture_sequential(as_of=..., finalized=True)`.
Continuation checks earlier units, assignments, values and definitions.

??? info "What sequential retention requires"
    Declare the sampling law, proper predictive priors, the metric × arm ×
    segment roster, and the common joint-unit reveal assumptions before
    supplying data. Bernoulli outcomes use Beta prediction and are the only
    sampling law admitted by the public sequential replay/readout views.
    Scalar Gaussian outcomes use NIG prediction, and ratios require actual
    joint numerator/denominator observations under a declared bivariate
    Gaussian law and positive population denominator means; those Gaussian
    and ratio paths remain private raw diagnostic kernels and cannot produce
    public sequential significance or readouts. These assumptions do not
    follow from a metric type, a positive observed mean, or a finite standard
    error.


    Raw encouragement ITT and registered relative Bernoulli uptake compliance
    are supported. Binary-uptake LATE, refitted CUPED, generic adjusted scores
    and quantiles have no matching likelihood proof and refuse sequential
    requests; their fixed-horizon analyses remain available. Fixed-horizon
    as-of LATE is a descriptive series and does not carry an
    optional-stopping guarantee.

### The dead zone

Neither series has data before `first_exposure_date + threshold_days`.
For a `[7, 14)` band, cumulative monitoring can start at day 7 instead of
waiting for day 14, but it cannot remove that first seven-day gap.

### Units whose band never closes

The cumulative gate admits a unit as soon as it clears `threshold_days`,
before its band has closed. Without a declared `observation_end`, a unit
enrolled near the end can have its band close after data collection stops;
the engine can no longer tell whether it would have returned, so that unit
sits at a hard 0 that never resolves.

Declare `observation_end = end + b`, where `b` is the band's right edge:
`end + 14` for `[7, 14)`, not `end + 7`. Every enrolled unit's band then
closes inside the observation horizon rather than remaining an unresolved
zero.

### Unbounded retention doesn't keep getting more precise

Longer follow-up does not necessarily sharpen an unbounded retention effect.
In a simulation with a constant +30% return-hazard effect, the z-statistic
peaked around day 14 and collapsed by day 120 as later, weakly informative
tenures diluted the estimate. A real constant effect can become undetectable
without changing. Prefer a bounded band for a completed-window comparison;
an unbounded band has no maturity date, so `completed_windows_only=True`
cannot make it complete.

## Ratio

Numerator and denominator are both random quantities, each finer-grained than
the randomisation unit: revenue per session when the unit is a user, clicks
per impression when the unit is a visitor. Because both parts vary per unit,
the arm-level ratio is analysed with the delta method rather than as a mean of
per-unit ratios.

```yaml
- type: ratio
  name: revenue_per_session
  description: "Revenue per session (ratio of sums)"
  entity: user_id
  preferred_direction: increase
  numerator:
    fact: purchase
    aggregation: sum
    window_days: 7
  denominator:
    fact: session_end
    aggregation: count
    window_days: 7
```

Each part is a full measure: its own `fact`, `aggregation`, `window_days`, and
optional `filters`.

A fact's `missing` policy applies to every metric using its value column,
including counts. Thus purchase `sum` divided by purchase `count` equals mean
order value: `skip` omits a NULL-revenue order from both parts; `zero` counts
it as a zero-revenue order. The default, `error`, refuses a read whose window
contains a NULL value with `source.native.fact_null_values`.
See [fact definitions](../reference/warehouse-definitions.md).

### Heavy-tailed denominators at small samples

The delta method approximates the joint behavior of the numerator and
denominator. Sparse or strongly uneven event counts can make that approximation
unreliable, but denominator skew alone does not determine interval coverage.

!!! warning "`estimation.engine.ratio_denominator_skew`"
    When either arm's sample skewness divided by `sqrt(n)` exceeds 0.30, the
    readout raises this warning naming the metric, arm, `n`, skewness and
    threshold, and the result row's `note` records it so tables and the
    dashboard can retain the caution. The estimate and interval are reported
    unchanged. Review the affected slice's event counts and sample size; this
    heuristic does not establish that its interval undercovers or is well calibrated.

??? info "Measured coverage"
    With an independent Normal numerator in the calibration simulations, a
    lognormal(1.5) denominator covers 90.2% at 50 units per arm and 93.7% at 400,
    and a lognormal(2.0) denominator 85.1% and 91.7%, at nominal 95% confidence.
    Those measurements do not transfer automatically to a ratio whose numerator
    and denominator come from the same events. The moments carry the denominator's
    third moment, so its sample skewness can be checked. Older moments without
    that value report the check as unavailable rather than passing silently. See
    [limitations](../limitations.md#uncertainty-is-estimated-on-every-path)
    for the measured table and the cases (clustered, sequential) the check
    does not claim.

When a ratio metric uses mixed-covariate `cuped_linear`, it has the
additional fixed-horizon M1 support check described in the
[mixed-CUPED reference](../reference/variance-reduction.md#mixed-numeric-and-categorical-covariates): each raw
arm denominator must have relative standard error at most 0.15. This
rejects some heavy-denominator designs around 100 units per arm; it is a
support restriction, not a coverage certificate. Scalar ratio CUPED remains
a separate method and does not use this M1 check.


### Conditional effects (CATE)

Conditional effects support mean and conversion metrics only. `estimate_cate`,
`validate_cate`, `targeting_rule`, and `select_targeting_rule` refuse ratio
metrics before reading unit-grain data: conditional ratio effects require
modeling denominator variation, which is outside the initial-release contract.
To study conditional ratio effects, model the numerator and denominator as
separate mean metrics. The arm-level ratio lift remains supported:
`readouts.run()` computes the ratio of summed numerators to summed
denominators for each arm and reports their difference on the ratio's own
scale.

#### Cluster identities and intervention grain

Declare `cluster="store_id"` on dataframe sources, or `cluster: store_id` in
an experiment definition, to retain cluster identities through CATE and
targeting. These identities are metadata, not numeric predictors; they cannot
also be requested through `interact` or `adjust`.

Declare `intervention_grain="cluster"` when treatment is deployed by cluster.
The default remains `"unit"`: dependence clusters alone do not imply cluster-level
deployment. Randomized clusters must belong to one arm; observational dependence
clusters may contain both arms. Fitted CATE results report `n_clusters`, while
targeting policies retain the intervention grain and required scoring columns.

All four operations take the `(experiment, reader)` pair from a builder such
as `from_frame` or `from_definitions`. `n_train` and `n_holdout`
count members; validation's `n_clusters` counts independent heldout clusters.
Selection's count describes the inner population, while `selection.rule.n_clusters`
describes the untouched outer holdout. Repeating members under the same cluster
IDs adds no independent randomizations.

Covariates must be available from the source's `unit_frame`. The definitions
source serves any numeric `pre_exposure` or `static` property by name, resolved
like a breakout. Unit-day artifacts automatically publish observational
adjustment properties and per-metric CUPED property and measure bindings, so
those covariates are also available on replay, including for `cuped_linear`
estimation (see [CUPED](cuped.md)).
Intercept-only calls (`interact=[]`) can use native and adopted artifact sources.
They have constant rankings, so clustered AUTOC/Qini uncertainty is unavailable
with `estimation.targeting.degenerate_rank_distribution`, and the targeting gate
stays closed. An available holdout-average interval remains useful independently
of that ranking result.

`deploy_grain=None` resolves from that declaration. Requesting `"unit"` for a
cluster intervention raises `estimation.targeting.unsupported_unit_deployment`.
For cluster policies, member scores are averaged and clusters ordered by score,
with canonical IDs breaking ties. The policy takes the longest whole-cluster
prefix within the requested fraction, without splitting or skipping clusters.
Member-count weighting budgets members; equal-cluster weighting budgets clusters
and uses the same target for fitting and evaluation. An oversized leading cluster
leaves the policy empty. Fractions zero and one select nobody and everybody.

<!-- skip: next "requires a clustered experiment with a spend covariate" -->

```python
from increment.estimators import ClusterBootstrap, targeting_rule
from increment.results import TargetingRule

rule = targeting_rule(
    experiment,
    reader,
    "revenue",
    control="control",
    interact=["spend"],
    fraction=0.4,
    deploy_grain="cluster",
    cluster_weight="member_count",
    bootstrap=ClusterBootstrap(seed=7, repetitions=999),
)
restored = TargetingRule.model_validate_json(rule.model_dump_json())
# Supply every member of each new deployment cluster in the same batch.
if restored.recommendation == "target":
    actions = restored.predict({"spend": new_spend}, cluster_ids=new_store_ids)
```

`rule.fraction` and `rule.achieved_fraction` record requested and realized
holdout budget shares. Its fitted scoring state survives JSON reload.
`fit.score(..., cluster_ids=...)` returns keyed `ClusterScore` records for
cluster interventions and aligned unit scores otherwise.

Mixed-treatment dependence clusters keep individual actions and
cluster-aware uncertainty under their default unit intervention declaration.
Adopted unit-day artifacts preserve cluster provenance when published with
the `cluster_identity` extension.

## Margins: non-inferiority guardrails

A guardrail margin declares a tolerated adverse move, shifting the null from
"no change" to "no worse than this." Quantiles do not support guardrails;
see [quantile restrictions](quantile-metrics.md#unsupported-combinations).
A supported metric carries at most one of two forms:

- **`margin` (relative):** a fraction of the control mean.
  `margin: 0.01` with `preferred_direction: increase` means “we did not lose
  more than 1% *of the current value*.”
- **`margin_abs` (absolute):** the metric's own units.
  `margin_abs: 0.01` on conversion means “no more than 1 *percentage point*”;
  `margin_abs: 0.10` on revenue per unit means “no more than $0.10 per unit.”

```yaml
- type: conversion
  name: purchase_rate
  description: "Guardrail: hold purchase rate within 1pp"
  entity: user_id
  preferred_direction: increase
  fact: purchase
  window_days: 14
  margin_abs: 0.01  # tolerate at most a 1-percentage-point drop
```

State the margin the way the requirement is stated: if the tolerance scales
with the baseline ("we can afford to lose 1% of conversion, whatever it
currently is"), use `margin`; if it is fixed in the metric's units, use
`margin_abs`. An absolute margin is tested as a genuine additive-scale interval
(`abs_diff` against the absolute boundary), not by dividing the absolute margin
by the observed control mean, which would make the null hypothesis
data-dependent.

Declaring both on one metric is a validation error. Both forms require an
explicit, non-neutral `preferred_direction`: the direction picks which side of
the boundary is adverse, and a defaulted direction must never silently pick a
guardrail's adverse side. A plan's guardrail-binding `margin`/`margin_abs`
(declared per experiment, in the plan's `guardrails` entry) overrides the
metric's own catalog-declared value; there is no call-time override on
`run()`.

!!! warning "When the absolute interval is unavailable"
    A Wald-based additive interval needs the absolute standard error
    (`abs_se`). If it cannot be computed (for example, because of numerical
    underflow), the absolute-margin decision is unavailable *loudly*:
    `stat_sig` is `False`, and there is no silent fallback to the relative
    interval. Binomial score-reference intervals are different: they use the
    retained arm counts and can remain available when `abs_se` is `None`.

## The dataframe path: `MetricSpec`

The [declaration rules above](#declarations-defaults-and-missing-values) cover
ordinary inputs. A bare name cannot be used with `from_moments`, which has
no dataframe schema. This example shows how explicit types preserve the
procedure when storage dtype changes.

This deterministic example has 100 successes among 1,000 control units and 130
among 1,000 treatment units. The inferred Boolean and floating-point columns
select different procedures; each explicit declaration gives matching results
for the two storage types:

```python
from math import isclose

import polars as pl

import increment

n = 1000
control_successes = 100
treatment_successes = 130
successes = (
    [True] * control_successes
    + [False] * (n - control_successes)
    + [True] * treatment_successes
    + [False] * (n - treatment_successes)
)
df = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "success_bool": successes,
        "success_float": [float(value) for value in successes],
    },
    schema_overrides={"success_float": pl.Float64},
)


def analyze(column, metric_type=None):
    metric = column if metric_type is None else increment.MetricSpec(name=column, type=metric_type)
    experiment, reader = increment.from_frame(
        df,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics=[metric],
    )
    return next(iter(increment.run(experiment, reader).readout()))


boolean_inferred = analyze("success_bool")
float_inferred = analyze("success_float")
assert boolean_inferred.metric_type == "conversion"
assert float_inferred.metric_type == "mean"
assert boolean_inferred.reference_kind != float_inferred.reference_kind

for row in (boolean_inferred, float_inferred):
    print(
        f"{row.metric_type} ({row.reference_kind}): "
        f"p={row.p_value():.6g}; stat_sig={row.stat_sig()}"
    )


def assert_parity(left, right, expected_type):
    assert left.metric_type == right.metric_type == expected_type
    assert left.reference_kind == right.reference_kind
    assert isclose(left.mean_control, right.mean_control, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(left.mean_treat, right.mean_treat, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(
        left.require_lift().value,
        right.require_lift().value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert isclose(left.p_value(), right.p_value(), rel_tol=1e-12, abs_tol=1e-12)
    assert left.stat_sig() == right.stat_sig()


assert_parity(
    analyze("success_bool", "conversion"),
    analyze("success_float", "conversion"),
    "conversion",
)
assert_parity(
    analyze("success_bool", "mean"),
    analyze("success_float", "mean"),
    "mean",
)
```

Here the inferred Boolean result is `conversion` with a binomial reference
(`p=0.06715`, `stat_sig=False`), while the inferred `Float64` result is `mean`
with a t reference (`p=0.0364455`, `stat_sig=True`). The changed decision
reflects dtype-selected procedures, not an arithmetic or backend failure.


```python
import random

import polars as pl

import increment

random.seed(11)
n = 400
df = pl.DataFrame(
    {
        "user_id": [f"u{i:04d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "revenue": [max(0.0, random.gauss(10, 4)) for _ in range(n)]
        + [max(0.0, random.gauss(11, 4)) for _ in range(n)],
        "converted": [int(random.random() < 0.30) for _ in range(n)]
        + [int(random.random() < 0.34) for _ in range(n)],
        "orders": [random.randint(1, 5) for _ in range(2 * n)],
    }
)

experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        increment.MetricSpec(name="revenue", type="mean"),
        increment.MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
        increment.MetricSpec(
            name="revenue_per_order",
            type="ratio",
            numerator="revenue",
            denominator="orders",
        ),
    ],
)
results = increment.run(experiment, reader).readout()

for r in results:
    print(f"{r.metric}: lift={r.lift.value:+.2%}")
```

For dataframe inputs, a `mean` column already holds each unit's aggregated
value; `conversion` holds a 0/1 indicator, not raw events; `ratio` uses two
per-unit total columns. A `quantile` compares the distribution's chosen
p50/p90/p99 value.

`value_column` reads from a column different from the metric's display name.
For CUPED, declare the pre-experiment covariate on the method, not on
`MetricSpec` itself; see [CUPED](cuped.md).

!!! warning "Conversion columns are binary"
    A `type="conversion"` column must contain only `0`/`1` or boolean values.
    Conversion metrics estimate the share of units converting; they do not
    interpret arbitrary nonzero magnitudes as occurrences. If the source
    contains magnitudes, declare a `mean` metric or pre-compute a 0/1
    indicator at the input analysis grain. Retention still dispatches on "any
    occurrence" for pre-collapsed per-day values, so a legitimate `0.0`
    remains indistinguishable from no event on that day.

!!! info "Windows and retention on the panel path"
    `MetricSpec` accepts `window_days` (mean/conversion/ratio) and
    `threshold_days`/`type="retention"` directly. On the panel path these are
    computed for real, given an explicit `exposure_date=` column naming each
    unit's own day 0 (a derived anchor would silently define the window). The
    same column is the panel's only clock for daily and as-of views, an
    encouragement design's uptake window and as-of compliance; without it these
    refuse and only whole-window results of unwindowed metrics remain.
    `observation_end=` bounds late-enrollee censoring; omitted, it falls back
    to each metric's own observed extent. Both parameters are accepted by
    `from_frame(..., shape="panel")` and forwarded to
    `increment.frame.from_unit_panel`, the lower-level factory it wraps.
    A summary refuses `window_days`/`threshold_days` outright (a
    one-row-per-unit summary carries no dates to window or band against), and
    `window_days` on `type="quantile"` is refused on either constructor. See
    [the dataframe analysis example](../examples/analysis_from_a_dataframe.md)
    for a worked windowed-mean and bounded-retention example, and
    [The data model](data-model.md) for `exposure_date`/`observation_end`.

Quantile order-statistic brackets are distribution-free; their two-arm
relative-lift intervals remain asymptotic. Ties can widen those intervals.
See [quantile interpretation](quantile-metrics.md#the-woodruff-order-statistic-bracket-and-tied-values)
for the width rule, its limits and the interval–p-value relationship.
