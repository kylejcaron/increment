# Metric types

Choose the metric type that matches what one unit contributes: a numeric
outcome, an event indicator, a return window, or a ratio of totals. Declare it
in warehouse YAML or in a dataframe `MetricSpec`.

```python
import polars as pl

from increment import Analysis

metric_demo = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(8)],
        "variant": ["control"] * 4 + ["treatment"] * 4,
        "revenue": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        "converted": [0, 1, 0, 1, 1, 0, 1, 1],
    }
)
metric_results = Analysis.from_unit_summary(
    metric_demo,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean", "converted": "conversion"},
).run()
for row in metric_results:
    print(row.metric, row.group_id, row.lift)
```

The example uses Polars; any eager dataframe supported by Narwhals works.
See [Reading results](reading-results.md) and the
[population/status table](multiplicity.md) when interpreting results.

Warehouse definitions reduce events to unit outcomes; dataframe inputs already
hold per-unit values. Support differs by metric and source; see
[method compatibility](compatibility.md) for the full limits.

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
from increment import MetricSpec

MetricSpec(
    name="revenue_per_user",
    type="mean",
    winsorization={"upper_value": 500.0},
)
```

Percentile bounds use the pooled eligible units across arms; fixed bounds use
the values you declare. Zero is a regular observation. Results expose each
arm's resolved cap and capped count. A difference in cap rates may signal a
treatment-driven change in the tail; it is diagnostic, not an adjustment.

??? info "Percentile bound resolution"
    Percentile bounds are resolved once from the pooled eligible unit population
    across all arms, with exact linear interpolation; fixed bounds use the supplied
    values.

!!! warning "Percentile winsorization has no decision evidence"
    Percentile cutoffs are estimated from the data, so their uncertainty must
    be included in inference. Current percentile-winsorized intervals are
    experimental: `stat_sig` is unavailable, not "not significant." Do not use
    these results to make a significance or family-selection decision.

!!! warning "Prespecify the winsorization method"
    Method choice and seed/stream must be set before examining inferential results.

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
the indexing convention first.

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

!!! warning "The boundary is part of the estimand: declare it before launch and freeze it"
    Changing `day_boundary` mid-flight reshuffles every day bucket: exposure
    days move, window and band edges move with them, the daily series is
    discontinuous at the change, and any pre/post comparison across it is
    invalid. It is an experiment-design declaration, not a display preference.

!!! tip "Anchor the boundary in the activity trough"
    Choose the offset that rolls the day where the product is quietest (an
    EST-centred org might declare `UTC-09:00`, cutting days at roughly 4 AM
    local) so as little event mass as possible sits at the bucket edge. A
    fixed offset drifts ±1 hour against the wall clock across DST transitions,
    but deterministically; the buckets themselves stay stable.

!!! warning "Short retention windows and timezone comparisons need care"
    A unit's calendar-day window can differ by up to a day; for d1 retention,
    that is as large as the window itself. Changing `day_boundary` cannot fix
    this: an elapsed-24h window would be needed and is not supported here.
    A boundary that does not match a segment's local day also slices traffic
    differently across timezone-correlated segments, so comparing their effect
    sizes can change meaning. Each segment's own lift remains internally
    valid under an arm-symmetric convention.

??? info "Short day 0, short thresholds, and cross-segment reads"
    Under any boundary, a unit exposed ten minutes before it (23:50 UTC under
    the default) gets a ten-minute "day 0"; its first full day of observation
    is day 1. Day buckets also shift relative to any analytics tooling that
    buckets events at a different boundary.

    Calendar-day bucketing therefore leaves each unit's effective observation
    window heterogeneous by up to a day, and no choice of boundary removes
    that. For d1-class retention the heterogeneity is the same order as the
    window itself, so the estimand is not uniform across units; the remedy is
    a per-metric elapsed-24h window (a tracked follow-up), not a different
    `day_boundary`.

    Each segment's own lift is internally valid under any arm-symmetric
    convention. What a boundary not aligned with a segment's local day
    degrades is comparing effect sizes *across* timezone-correlated segments
    (a US vs. EU breakout, say), whose traffic is sliced into days
    differently. With a trough-anchored boundary this is second-order at d7
    and beyond.

### Two series, one metric

Retention has two readouts, and they answer different questions:

- `run()` / `run_breakout` compare each unit's **mature, whole-window**
  outcome, arm to arm: the windowed decision estimand. Unbiased, and the
  number a ship/no-ship call should be made on.
- `run_asof` / `run_asof_lift` return a **cumulative monitoring series**
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

!!! warning "A mature window alone does not prove sequential finalization"
    Reveal every relevant outcome for a unit together after the longest
    required window, in an outcome-independent order. Bounded windows and
    `completed_windows_only=True` do not establish finalization: native and
    panel sources require `capture_sequential(as_of=..., finalized=True)`.
    Configure the registered process before supplying data; continuation
    checks earlier units, assignments, values, and definitions.

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

    Every relevant outcome for a unit must be revealed together after the
    longest required window, in an outcome-independent order. Bounded windows
    and `completed_windows_only=True` alone do not prove finalization. Native
    and panel sources require `capture_sequential(as_of=..., finalized=True)`;
    continuation checks prior unit identities, assignments, values and
    definitions.

    Raw encouragement ITT and registered relative Bernoulli uptake compliance
    are supported. Binary-uptake LATE, refitted CUPED, generic adjusted scores
    and quantiles have no matching likelihood proof and refuse sequential
    requests; their fixed-horizon analyses remain available. Fixed-horizon
    as-of LATE is a descriptive series and does not carry an
    optional-stopping guarantee.

### The dead zone

Neither series has any data before `first_exposure_date + threshold_days`: a
unit has no tenure to score before then, on any view. The as-of gate moved the
*start* of the cumulative series from `window_days` days after enrollment to
`threshold_days` days (roughly halving the gap for a `[7, 14)` band), but
`threshold_days` itself is the irreducible remainder.

### Units whose band never closes

The cumulative gate admits a unit as soon as it clears `threshold_days`,
before its band has closed. Without a declared `observation_end`, a unit
enrolled near the end can have its band close after data collection stops;
the engine can no longer tell whether it would have returned, so that unit
sits at a hard 0 that never resolves.

!!! tip "Declare `observation_end = end + b`"
    `b` is the band's right edge (see [The data model](data-model.md)): for a
    `[7, 14)` band, `observation_end = end + 14`, not `end + 7`. Every
    enrolled unit's band then closes inside the observation horizon, so no
    unit is left as a permanent, unresolved zero.

### Unbounded retention doesn't keep getting more precise

Watching an unbounded metric longer does not monotonically sharpen it. Under a
constant per-tenure effect the z-statistic peaks early and then declines as
later, weakly-informative tenures dilute the running estimate: measured on a
constant +30% return-hazard effect, it peaks around day 14 and has collapsed
by day 120, so a real, constant effect can become statistically undetectable
purely by observing longer. If you need an estimate that keeps improving the
longer you watch, prefer a bounded band (or `completed_windows_only=True` on
the unbounded one) over leaving the band open indefinitely.

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

### Heavy-tailed denominators at small samples

The delta-method interval assumes the denominator mean is close to normally
distributed. A heavily right-skewed denominator (a few units with very many
sessions) breaks that at small samples.

!!! warning "`estimation.engine.ratio_denominator_skew`"
    When either arm's sample skewness divided by `sqrt(n)` exceeds 0.30, the
    readout raises this warning naming the metric, arm, `n`, skewness and
    threshold, and the result row's `note` records it so tables and the
    dashboard show it. The estimate and interval are reported unchanged: treat
    the interval as optimistic, and add units or examine the denominator's
    tail before deciding.

??? info "Measured coverage"
    At nominal 95%, a lognormal(1.5) denominator covers 90.2% at 50 units per
    arm and 93.7% at 400, and a lognormal(2.0) denominator 85.1% and 91.7%,
    while a lognormal(0.5) denominator is nominal at every size. The moments
    carry the denominator's third moment, so each arm's sample skewness is
    known when the interval is built. A row estimated from a moments cube
    written before the third moment existed says the check was unavailable
    rather than passing silently. See
    [limitations](../limitations.md#uncertainty-is-estimated-on-every-path)
    for the measured table and the cases (clustered, sequential) the check
    does not claim.

When a ratio metric uses mixed-covariate `cuped_linear`, it has the
additional fixed-horizon M1 support check described in the
[CUPED guide](cuped.md#mixed-numeric-and-categorical-covariates): each raw
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

All four entry points also work through `Analysis`. `n_train` and `n_holdout`
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

<!-- skip: next "requires a clustered source with a spend covariate" -->

```python
from increment import ClusterBootstrap, targeting_rule
from increment.results import TargetingRule

rule = targeting_rule(
    source,
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

`rule.fraction` and `rule.achieved_fraction` record the requested and realized
holdout budget shares. The frozen fitted scoring state survives JSON reload;
Covariates come from the source's `unit_frame`. Native warehouse definitions can
resolve requested `pre_exposure` or `static` properties, and adopted unit-day
artifacts can use covariates declared in their observational design. Dataframe
unit-summary sources can carry covariate columns directly.

??? info "Counts, covariates, and cluster policies"
    Fitted CATE results report `n_clusters`, while targeting policies retain
    the intervention grain and required scoring columns. All four entry points
    also work through `Analysis`. `n_train` and `n_holdout` count members;
    validation's `n_clusters` counts independent heldout clusters. Selection's
    count describes the inner population, while `selection.rule.n_clusters`
    describes the untouched outer holdout. Repeating members under the same
    cluster IDs adds no independent randomizations.

    Covariates must be available from the source's `unit_frame`. The
    definitions source serves any numeric `pre_exposure` or `static` property
    by name, resolved like a breakout; a unit-day artifact serves only the
    covariates its experiment declares under an observational
    `design.covariates` (see
    [observational.md](observational.md#from-definitions-the-warehouse-path)).
    Intercept-only calls (`interact=[]`) can use native and adopted artifact
    sources. They have constant rankings, so clustered AUTOC/Qini uncertainty
    is unavailable with `estimation.targeting.degenerate_rank_distribution`,
    and the targeting gate stays closed. An available holdout-average interval
    remains useful independently of that ranking result.

    `deploy_grain=None` resolves from the intervention-grain declaration.
    Requesting `"unit"` for a cluster intervention raises
    `estimation.targeting.unsupported_unit_deployment`. For cluster policies,
    member scores are averaged and clusters ordered by score, with canonical
    IDs breaking ties. The policy takes the longest whole-cluster prefix
    within the requested fraction, without splitting or skipping clusters.
    Member-count weighting budgets members; equal-cluster weighting budgets
    clusters and uses the same target for fitting and evaluation. An oversized
    leading cluster leaves the policy empty. Fractions zero and one select
    nobody and everybody.

    `fit.score(..., cluster_ids=...)` defaults to keyed `ClusterScore` records
    for cluster interventions and aligned unit scores otherwise.
    Mixed-treatment dependence clusters keep individual actions and
    cluster-aware uncertainty under their default unit intervention
    declaration. Adopted unit-day artifacts preserve this provenance when
    published with the `cluster_identity` extension. Native warehouse and
    adopted artifact `unit_frame` implementations can attach requested covariates:
    definitions resolve eligible properties by name, while artifacts serve
    covariates declared under `observational.design.covariates`. See
    [observational.md](observational.md#from-definitions-the-warehouse-path).

## Margins: non-inferiority guardrails

Any metric can declare a guardrail margin: a tolerated adverse move that
shifts the null boundary from "no change" to "no worse than this." A metric
carries at most one of two forms:

| Form | Tolerance | Reads as |
|---|---|---|
| `margin` (relative) | A fraction of the control mean. | `margin: 0.01` with `preferred_direction: increase`: "we did not lose more than 1% *of the current value*." |
| `margin_abs` (absolute) | The metric's own units. | `margin_abs: 0.01` on a conversion metric: "no more than 1 *percentage point*"; `margin_abs: 0.10` on revenue per unit: "no more than $0.10 per unit." |

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
    The additive test needs the absolute standard error (`abs_se`). In the
    rare case it cannot be computed (numerical underflow), the absolute-margin
    decision is unavailable *loudly*: `stat_sig` is `False`, and there is no
    silent fallback to the relative interval.

## The dataframe path: `MetricSpec`

On the dataframe path (`Analysis.from_unit_summary` / `from_unit_panel`), your
columns already hold per-unit values, so a metric is a column plus a type. The
mapping form `{"revenue": "mean", "converted": "conversion"}` covers the
simple cases; explicit `MetricSpec` objects unlock the rest:

```python
import random

import polars as pl

from increment import Analysis, MetricSpec

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

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        MetricSpec(name="revenue", type="mean"),
        MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
        MetricSpec(
            name="revenue_per_order",
            type="ratio",
            numerator="revenue",
            denominator="orders",
        ),
    ],
).run()

for r in results:
    print(f"{r.metric}: lift={r.lift.value:+.2%}")
```

| Type | The column holds |
|---|---|
| `mean` | Each unit's already-aggregated value. |
| `conversion` | A pre-computed 0/1 indicator, not raw events. |
| `ratio` | Two per-unit total columns, named by `numerator` and `denominator`. |
| `quantile` | Values whose distributional quantile (p50/p90/p99, set by `quantile=`) is compared. |

`value_column` reads values from a different column than the metric's display
name; `covariate` attaches a pre-experiment column for CUPED (see
[CUPED](cuped.md)).

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
    unit's own day 0 (a derived anchor would silently define the window).
    `observation_end=` bounds late-enrollee censoring; omitted, it falls back
    to each metric's own observed extent. Both parameters are accepted by
    `Analysis.from_unit_panel` and forwarded to
    `increment.frame.from_unit_panel`, the lower-level factory it wraps.
    `from_unit_summary` refuses `window_days`/`threshold_days` outright (a
    one-row-per-unit summary carries no dates to window or band against), and
    `window_days` on `type="quantile"` is refused on either constructor. See
    [the dataframe analysis example](../examples/analysis_from_a_dataframe.md)
    for a worked windowed-mean and bounded-retention example, and
    [The data model](data-model.md) for `exposure_date`/`observation_end`.

??? info "Quantile intervals on coarsely recorded outcomes"
    A quantile's interval stays valid on tied or coarsely recorded outcomes
    (whole milliseconds or seconds, cents, counts, prices ending in both .99
    and .00, or a grid with some values recorded off it) instead of refusing
    on tied order statistics. While the order-statistic bracket holds a tie
    and does not yet span a dozen repeated values, the interval is widened
    just enough to contain that bracket, whose own coverage holds for any
    distribution; otherwise it is the classical order-statistic interval. The
    cost is nil on continuous data and well-resolved grids and about 1.1 to 3
    times the classical width on coarse grids at large sample sizes and on
    counts. It has no fixed upper bound (a bracket collapsed onto one value
    has zero classical width), so every widened row states its own ratio in
    its `note`. The row's p-value is dual to its interval:
    `p_value() <= alpha` exactly when the interval at `alpha` excludes zero.
