# Metric types

Four metric types cover most experiment readouts: **mean**,
**conversion**, **retention**, and **ratio**. Each reduces a unit's events to
one number, then compares the arms. This guide explains what each type
measures, its YAML shape on the definitions path, and its `MetricSpec`
equivalent on the dataframe path.

All windowed fields (`window_days`, `threshold_days`) count in day
offsets from each unit's own first exposure: see
[The data model](data-model.md).

## Mean

Aggregate a numeric fact per unit, then average across units. The
`aggregation` field selects the per-unit reduction: `sum`, `count`,
`count_distinct`, `avg_event`, `avg_calendar_day`, `min`, or `max`.
`avg_event` divides the qualifying event-value sum by the qualifying event
count. `avg_calendar_day` divides the sum by the inclusive admitted calendar-
day count, treating eventless days as zero. Bare `avg` is rejected.

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
and drives downstream favorability readouts.

### Winsorization

Definitions-backed metrics support every mean aggregation (`sum`, `count`,
`count_distinct`, `avg_event`, `avg_calendar_day`, `min`, and `max`).
Dataframe inputs supply the per-row outcome directly; `MetricSpec` aggregates
those values with sum semantics before winsorization.

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

Percentile bounds are resolved once from the pooled eligible unit population
across all arms, with exact linear interpolation; fixed bounds use the supplied
values. Zeros are ordinary observations and remain eligible. Reported effects
describe the transformed outcome, not the raw tail, and results expose the
resolved bounds and capped counts for each arm. A cap-rate difference can
indicate treatment-driven tail movement; it is diagnostic, not a separate
adjustment.

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

For an absolute contrast of ordinary independent iid means,
`increment.estimation.infer_independent_mean(control_arm, treatment_arm)`
accepts centered `ArmStats` from the existing moments producers. It retains
the component counts and centered sums, uses ddof=1 within each arm and a
Welch-Satterthwaite reference, and persists its df and components on the result.
This operation does not apply to adjusted scores, clusters, or estimated-cutoff
moments; those require their own declared inference contracts.

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

A survival outcome: did the unit return on or after day `threshold_days`?
This single field supports both band shapes. A bare integer `N` declares the
**unbounded** band `[N, inf)`: "returned on or after day N, ever." A
two-element `[a, b]` declares the **bounded**, half-open band `[a, b)`:
"returned during days a..b-1." The band is day-0-indexed: an event counts
when

$$
\text{first exposure date} + a \le ds <
\text{first exposure date} + b
$$

for the bounded case; the unbounded case drops the upper bound.

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

Because a bounded band closes on the right, a unit's outcome is
**final** at `first_exposure_date + b` (its *maturity*). An unbounded
outcome has no completion date; it can only ratchet 0 → 1 as more data
arrives, so it is not reportable on the cohort (exposure-indexed) day
axis; only on the calendar axis, below. `window_days` is not a
retention field: declaring it on a `RetentionMetric` raises at
construction rather than being silently ignored; both band edges live
in `threshold_days`.

### How days are counted: day 0 and the day boundary

Two conventions decide which day an event lands on, and both are common
sources of "your number doesn't match mine" when comparing against a
warehouse query or a product-analytics tool.

**The exposure day is day 0.** The band test is
`ds >= first_exposure_date + threshold_days`, so "d7 retention" means
the unit returned on or after the 7th calendar day *after* the exposure
day — the exposure day itself is day 0, not day 1. A warehouse query
that counts the exposure day as day 1 is measuring a band shifted by
one day and will produce a genuinely different number, not a rounding
difference. When reconciling, check the indexing convention first.

**Day boundaries are calendar days at the declared `day_boundary`
(default UTC).** The exposure timestamp is cast to a calendar date at
the experiment's declared day boundary -- plain `"UTC"` by default, or
a fixed UTC offset -- and the band is date-interval arithmetic on that
date. Declare it on the experiment, or once as a definitions-level
default:

```yaml
day_boundary: "UTC-05:00"
```

The declared experiment window follows the same rule: `start`, `end` and
`observation_end` name days at the boundary. A value without a UTC offset is
wall-clock time at the boundary, so its written date is the day; a value with an
offset is converted to the boundary first, so two spellings of one instant open
the same window (worked example in
[Data model](data-model.md#window-edges-and-the-day-boundary)).

Only fixed offsets are accepted. Named IANA zones (DST-aware) are not
supported yet: a fixed offset has a portable translation on every
warehouse backend, a DST-aware zone does not. Two consequences worth
knowing under any boundary: a unit exposed ten minutes before the
boundary (23:50 UTC under the default) gets a ten-minute "day 0" — its
first full day of observation is day 1 — and day buckets shift
relative to any analytics tooling that buckets events at a different
boundary.

**The boundary is part of the estimand — declare it before launch and
treat it as frozen.** Changing `day_boundary` mid-flight reshuffles
every day bucket: exposure days move, window and band edges move with
them, the daily series is discontinuous at the change, and any
pre/post comparison across it is invalid. It is an experiment-design
declaration, not a display preference.

**Anchor the boundary in the activity trough.** Choose the offset that
rolls the day where the product is quietest — an EST-centred org might
declare `UTC-09:00`, cutting days at roughly 4 AM local — so as
little event mass as possible sits at the bucket edge. A fixed offset
drifts ±1 hour against the wall clock across DST transitions, but it
does so deterministically; the buckets themselves stay stable.

**Short thresholds stay heterogeneous.** Calendar-day bucketing leaves
each unit's effective observation window heterogeneous by up to a day
(the short-day-0 artifact above), and no choice of boundary — no
calendar convention at all — removes that. For d1-class retention the
heterogeneity is the same order as the window itself, so the estimand
is not uniform across units; the remedy is a per-metric elapsed-24h
window (a tracked follow-up), not a different `day_boundary`.

**Cross-segment reads.** Each segment's own lift is internally valid
under any arm-symmetric convention. What a boundary that is not
aligned with a segment's local day degrades is comparing effect sizes
*across* timezone-correlated segments (a US vs. EU breakout, say),
whose traffic gets sliced into days differently. With a
trough-anchored boundary this is second-order at d7 and beyond.

### Two series, one metric

Retention has two readouts, and they answer different questions:

- `run()` / `run_breakout` compare each unit's **mature, whole-window**
  outcome, arm to arm: the windowed decision estimand. Unbiased, and
  the number a ship/no-ship call should be made on.
- `run_asof` / `run_asof_lift` return a **cumulative monitoring
  series** indexed by calendar date: each date's value blends whatever
  It starts earlier (as soon as a unit clears `threshold_days`, rather
  than once its whole band closes) and carries much less sampling noise
  per point, which
  is what makes it useful for watching a running experiment. It is not,
  however, a stable reading of one parameter: the tenure mix changes
  every day, so consecutive points are not comparable to one another as
  a time series, and the series can diverge from `run()`'s decision
  estimate whenever the arms return at different *times* rather than
  different *rates*; even a true whole-window lift of zero can show up
  as a large early divergence. Pass `completed_windows_only=True` to
  gate on full maturity instead, which restores the old behaviour: a
  maturing decision estimate that only ever counts units whose band has
  fully closed.

### Peeking at the monitoring series

Repeated inspection requires a registered raw-likelihood process. Declare the
sampling law, proper predictive priors, metric × arm × segment roster and common
joint-unit reveal assumptions before supplying data. See [Sequential
inference](sequential-inference.md) for the complete construction and replay API.

Bernoulli outcomes use Beta prediction and are the only sampling law admitted
by the public sequential replay/readout views. Scalar Gaussian outcomes use NIG
prediction, and ratios require actual joint numerator/denominator observations
under a declared bivariate Gaussian law and positive population denominator
means; those Gaussian and ratio paths remain private raw diagnostic kernels and
cannot produce public sequential significance or readouts. These assumptions
do not follow from a metric type, positive observed mean, or finite standard
error.

Every relevant outcome for a unit must be revealed together after the longest
required window, in an outcome-independent order. Bounded windows and
`completed_windows_only=True` alone do not prove finalization. Native and panel
sources require `capture_sequential(as_of=..., finalized=True)`; continuation
checks prior unit identities, assignments, values and definitions.

Raw encouragement ITT and registered relative Bernoulli uptake compliance are
supported. Binary-uptake LATE, refitted CUPED, generic adjusted scores and
quantiles have no matching likelihood proof and refuse sequential requests.
Their fixed-horizon analyses remain available. Fixed-horizon as-of LATE is a
descriptive series and does not carry an optional-stopping guarantee.

### The dead zone

Neither series has any data before
`first_exposure_date + threshold_days`: a unit has no tenure to score
before then, on any view. This floor cannot be removed: the as-of gate
moved the *start* of the cumulative series from `window_days` days
after enrollment to `threshold_days` days (roughly halving the gap for
a `[7, 14)` band), but `threshold_days` itself is the irreducible
remainder.

### Units whose band never closes

`run_asof`/`run_asof_lift`'s cumulative gate admits a unit as soon as
it clears `threshold_days`, before its band has closed. If the
experiment has no declared `observation_end`, a unit enrolled near the
end can have its band close after data collection stops; the engine
can no longer tell whether it would have returned, so that unit sits at
a hard 0 that never resolves. Declaring
`observation_end = end + b` (see
[The data model](data-model.md)), where `b` is the band's right edge:
for a `[7, 14)` band, `observation_end = end + 14`, not `end + 7`.
This removes the problem: every enrolled unit's band then closes
inside the observation horizon, so no unit is ever left stuck as a
permanent, unresolved zero.

### Unbounded retention doesn't keep getting more precise

Watching an unbounded metric longer does not monotonically sharpen it.
Under a constant per-tenure effect the z-statistic peaks early and then
declines as later, weakly-informative tenures dilute the running
estimate: measured on a constant +30% return-hazard effect, it peaks
around day 14 and has collapsed by day 120, so a real, constant effect
can become statistically undetectable purely by observing longer. If
you need an estimate that keeps improving the longer you watch, prefer
a bounded band (or `completed_windows_only=True` on the unbounded one)
over leaving the band open indefinitely.

## Ratio

Numerator and denominator are both random quantities, each finer-grained
than the randomisation unit: revenue per session when the unit is a
user, clicks per impression when the unit is a visitor. Because both
parts vary per unit, the arm-level ratio is analysed via the delta
method rather than as a simple mean of per-unit ratios.

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

Each part is a full measure: its own `fact`, `aggregation`,
`window_days`, and optional `filters`.

### Heavy-tailed denominators at small samples

The delta-method interval assumes the denominator mean is close to normally
distributed. A heavily right-skewed denominator (a few units with very many
sessions) breaks that at small samples: measured at nominal 95%, a
lognormal(1.5) denominator covers 90.2% at 50 units per arm and 93.7% at 400,
and a lognormal(2.0) denominator 85.1% and 91.7%, while a lognormal(0.5)
denominator is nominal at every size. The moments carry the denominator's
third moment, so each arm's sample skewness is known when the interval is
built. When either arm's skewness divided by `sqrt(n)` exceeds 0.30, the
readout raises the `estimation.engine.ratio_denominator_skew` warning naming
the metric, arm, `n`, skewness and threshold, and the result row's `note`
records it so tables and the dashboard show it. The estimate and interval are
reported unchanged; treat the interval as optimistic, add units or examine the
denominator's tail before deciding. A row estimated from a moments cube written
before the third moment existed says the check was unavailable rather than
passing silently. See [limitations](../limitations.md#uncertainty-is-estimated-on-every-path)
for the measured table and the cases (clustered, sequential) the check does
not claim.

### Conditional effects (CATE)

Conditional effects currently support mean and conversion metrics only.
`estimate_cate`, `validate_cate`, `targeting_rule`, and
`select_targeting_rule` refuse ratio metrics before reading unit-grain data:
conditional ratio effects require modeling denominator variation, which is
outside the initial-release contract. To study conditional ratio effects,
model the numerator and denominator as separate mean metrics.

The arm-level ratio lift remains supported: `readouts.run()` computes the
ratio of summed numerators to summed denominators for each arm and reports
their difference on the ratio's own scale.

#### Cluster identities and intervention grain

Declare `cluster="store_id"` on dataframe sources, or `cluster: store_id` in an
experiment definition, to retain cluster identities through CATE and targeting.
These identities are metadata, not numeric predictors; they cannot also be
requested through `interact` or `adjust`.

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
like a breakout; a unit-day artifact serves only the covariates its experiment
declares under an observational `design.covariates` (see
[observational.md](observational.md#from-definitions-the-warehouse-path)).
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
prediction does not fit on the new data. `fit.score(..., cluster_ids=...)` defaults
to keyed `ClusterScore` records for cluster interventions and aligned unit scores
otherwise. Mixed-treatment dependence clusters keep individual actions and
cluster-aware uncertainty under their default unit intervention declaration.

Adopted unit-day artifacts preserve this provenance when published with the
`cluster_identity` extension. Native and adopted warehouse unit frames contain
the metric outcome and identifiers, not requested model covariates; use a
dataframe unit-summary source for covariate-adjusted fits.

## Margins: non-inferiority guardrails

Any metric can declare a guardrail margin: a tolerated adverse move that
shifts the null boundary from "no change" to "no worse than this."
There are two forms, and a metric carries at most one -- one guardrail,
one scale:

- **`margin`** (relative): the tolerance is a fraction of the control
  mean. `margin: 0.01` with `preferred_direction: increase` reads
  "confident we did not lose more than 1% *of the current value*."
- **`margin_abs`** (absolute): the tolerance is in the metric's own
  units. `margin_abs: 0.01` on a conversion metric reads "confident we
  did not lose more than 1 *percentage point*"; `margin_abs: 0.10` on a
  revenue-per-unit mean reads "no more than $0.10 of revenue per unit."

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

Declaring both on one metric is a validation error. Both forms require
`preferred_direction` to be explicitly declared and non-neutral: the
direction picks which side of the boundary is adverse, and a defaulted
direction must never silently pick a guardrail's adverse side. A plan's
guardrail-binding `margin`/`margin_abs` (declared per experiment, in the
plan's `guardrails` entry) overrides the metric's own catalog-declared
value; there is no call-time override on `run()`.

Which form to reach for: state the margin the way the requirement is
stated. If the tolerance scales with the baseline ("we can afford to
lose 1% of conversion, whatever conversion currently is"), that is
`margin`. If it is fixed in the metric's units ("no more than 1
percentage point", "no more than $0.10 per unit"), that is
`margin_abs` -- and it is tested as a genuine additive-scale interval
(`abs_diff` against the absolute boundary), not by dividing the
absolute margin by the observed control mean, which would make the
null hypothesis data-dependent.

!!! warning "When the absolute interval is unavailable"
    The additive test needs the absolute standard error (`abs_se`). In
    the rare case it cannot be computed (numerical underflow), the
    absolute-margin decision is unavailable *loudly*: `stat_sig` is
    `False`, and there is no silent fallback to the relative interval.

## The dataframe path: `MetricSpec`

On the dataframe path (`Analysis.from_unit_summary` /
`from_unit_panel`), your columns already hold per-unit values, so a
metric is just a column plus a type. The terse mapping form
`{"revenue": "mean", "converted": "conversion"}` covers the simple
cases; explicit `MetricSpec` objects unlock the rest:

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

How the types map:

- **mean**: the column holds each unit's already-aggregated value.
- **conversion**: the column must be a pre-computed 0/1 indicator, not
  raw events.
- **ratio**: `numerator` and `denominator` name two per-unit total
  columns.
- `value_column` reads values from a different column than the metric's
  display name; `covariate` attaches a pre-experiment column for CUPED
  (see [CUPED](cuped.md)); `type="quantile"` with `quantile=` estimates
  the lift of a distributional quantile (p50/p90/p99).

!!! note "Quantile intervals on coarsely recorded outcomes"
    A quantile's interval stays valid on tied or coarsely recorded outcomes (whole
    milliseconds or seconds, cents, counts, prices ending in both .99 and .00, or a grid
    with some values recorded off it) instead of refusing on tied order statistics. While
    the order-statistic bracket holds a tie and does not yet span a dozen repeated
    values, the interval is widened just enough to contain that bracket, whose own
    coverage holds for any distribution; otherwise it is the classical order-statistic
    interval. The cost is nil on continuous data and well-resolved grids and about 1.1 to
    3 times the classical width on coarse grids at large sample sizes and on counts. It has
    no fixed upper bound (a bracket collapsed onto one value has zero classical width), so
    every widened row states its own ratio in its `note`. The row's p-value is dual to its
    interval: `p_value() <= alpha` exactly when the interval at `alpha` excludes zero.

!!! info "Windows and retention on the panel path"
    `MetricSpec` accepts `window_days` (mean/conversion/ratio) and
    `threshold_days`/`type="retention"` directly -- construction no longer
    rejects them. On the panel path these are computed for real, given an
    explicit `exposure_date=` column naming each unit's own day-0 (a
    derived anchor would silently define the window). `observation_end=`
    bounds late-enrollee censoring; omitted, it falls back to each
    metric's own observed extent. Both parameters are accepted by
    `Analysis.from_unit_panel` and forwarded to
    `increment.frame.from_unit_panel`, the lower-level factory it
    wraps. `from_unit_summary` still refuses
    `window_days`/`threshold_days` outright -- a one-row-per-unit summary
    carries no dates to window or band against -- and `window_days` on
    `type="quantile"` is refused on either constructor. See
    [the dataframe analysis example](../examples/analysis_from_a_dataframe.md) for a
    worked windowed-mean and bounded-retention example, and
    [The data model](data-model.md) for `exposure_date`/`observation_end`.

!!! note "Panel conversion columns are binary"
    A `type="conversion"` column must contain only `0`/`1` or boolean
    values. Conversion metrics estimate the share of units converting;
    they do not interpret arbitrary nonzero magnitudes as occurrences.
    If the source contains magnitudes, declare a `mean` metric or
    pre-compute a 0/1 indicator at the input analysis grain. Retention
    still dispatches on "any occurrence" for pre-collapsed per-day values,
    so a legitimate `0.0` remains indistinguishable from no event on that
    day.
