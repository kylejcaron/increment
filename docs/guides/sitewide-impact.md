# Sitewide impact

Use `increment.estimators.sitewide(experiment, reader, metric)` to estimate
how much the whole site would have changed **during the observed experiment
window** if the measured effect had applied to every enrolled unit.
It is a rollout estimate for that window, **not a forecast**.

For example, the [complete revenue example](../reference/sitewide-calculation.md#worked-example)
estimates a per-unit gain of 5.92 across 200 enrolled units. Rolling that
effect out to all 200 gives an estimated site gain of **1,184.44**, with
interval **[385.90, 1,982.98]**. That is 78.81% of the estimated no-treatment
site baseline—not 78.81% per-user lift, and not annual revenue.

## Run it

1. Use an experiment built with `from_definitions`, or adopt a
   `from_unit_day_artifact` artifact published with the metric's
   `site_volume` extension. Frame summaries, panels and moments do not carry
   whole-site raw-event totals.
2. Choose an unwindowed mean metric aggregated by sum or count, or a ratio
   whose two components use those aggregations. A converter indicator is
   not additive; use a count on its event fact instead.
3. Call `sitewide(experiment, reader, "revenue_per_user")`. For multiple
   treatment arms, also pass `arm="treatment_b"` to choose one.

Start with the [complete native example](../reference/sitewide-calculation.md#worked-example):
it visibly creates every YAML definition, event row and DuckDB table before
the call. For a saved artifact and both population choices, use the
[complete assigned/triggered example](../reference/sitewide-calculation.md#complete-assigned-and-triggered-artifact-example).
Neither requires a prepared variable from another page.

!!! warning "Do not extend this estimate to a later window"
    The calculation assumes the observed per-unit effect applies to all
    enrolled units and uses only `experiment.start .. experiment.end`.
    It is not annualized. A measured effect does not establish its size or
    detectability later; even constant retention effects can become less
    detectable as the window grows
    ([why](metric-types.md#unbounded-retention-doesnt-keep-getting-more-precise)).

## Read the result

For a mean metric, `SitewideImpact.absolute_impact` is the estimated gain
in the metric's own units. The baseline subtracts treatment lift already
present in observed site volume; the gain then applies that lift to all
enrolled units, including control. Background activity from people never
enrolled still belongs in site volume.

For a ratio metric, `SitewideRatioImpact.absolute_impact` is
`shipped_ratio - baseline_ratio`: a **difference of site ratios of totals**,
not a ratio of differences or an average of individual ratios.
For either model, `relative_impact` divides the absolute gain by its
baseline. A non-positive baseline makes relative impact undefined and the
call refuses rather than returning a misleading fraction.

Use `absolute_impact_lb` / `absolute_impact_ub` for the absolute interval,
or `relative_impact_lb` / `relative_impact_ub` for the relative interval.
Both results are `exploratory_unadjusted`: no family adjustment is applied,
and the fixed-horizon intervals are not anytime-valid. The
[field reference](../reference/sitewide-calculation.md#result-fields) and
[calculation](../reference/sitewide-calculation.md#ship-to-all-calculation)
give the exact definitions and uncertainty formulas.

## Assigned or triggered

With a declared `trigger:`, choose the population that answers your question:

| Choice | Question |
|---|---|
| `population="assigned"` (default) | What is the intent-to-treat impact of assignment? |
| `population="triggered"` | What if the effect among finalized trigger members applied to all triggered enrolled units? |

Both use the experiment calendar window and **full-site** volume, not only
triggered users' activity. This differs from `run()`'s trigger-anchored
windows: outcomes after experiment end are excluded even if
`observation_end` is later. A late trigger can still qualify for membership.

The triggered interpretation assumes membership is independent of
treatment assignment, never-triggered units are unaffected, and enrolled
units exhaust exposure. Its allocation-balance diagnostic cannot establish
those causal assumptions.

Outcomes must be complete through experiment end. Triggered calls require
a feed watermark in `SourceSnapshotEvidence` or the artifact proving this.
Assigned calls refuse if a declared snapshot cutoff or watermark ends too
early; a native call without snapshot evidence has no cutoff to check.
See [population and completeness contracts](../reference/sitewide-calculation.md#assigned-or-triggered),
including the limits of snapshot identity and simulation evidence.

## Multi-arm experiments

Pass `arm=` to ask what would happen if **one selected arm alone** shipped
to everyone, compared with no treatment. Arms are not combined into an
unshippable scenario. The baseline removes every enrolled non-control
arm's lift, not just the selected arm's; those shared-baseline estimates
also contribute uncertainty. See
[multi-arm covariance](../reference/sitewide-calculation.md#multi-arm-covariance).

## Cluster-randomized experiments

Declare `Experiment.cluster` when stores, markets or other clusters—not
individual measured units—were assigned. Supported mean/ratio metrics
still need unwindowed sum/count components. Clustering leaves the point
estimate unchanged but changes its standard error and uses a t reference;
`absolute_dof` and `relative_dof` can differ.

Each enrolled arm needs at least two clusters. Below 40 total clusters,
a `RuntimeWarning` flags over-rejection risk; borderline significance is
fragile. This is a warning threshold, not an admission floor. Adding units
within a cluster is not a substitute for more independent clusters.

!!! warning "Enrollment must include everyone exposed to treatment"
    A store-wide treatment can affect customers who never logged enrollment.
    Their affected volume then enters the baseline as if untouched; the
    documented cluster-population bias understates the rollout impact.
    Increment does not correct spillovers or restrict the site total to
    enrolled clusters. Check this assumption before interpreting the result.

The [cluster reference](../reference/sitewide-calculation.md#cluster-randomized-experiments)
explains the variance calculation and arm-specific degrees of freedom.

## When it refuses

Use randomized experiments with additive, calendar-window-aligned metrics.
Retention, quantiles, conversion indicators, per-unit windows and
non-additive aggregations cannot supply aligned site totals. Observational
and encouragement designs are not supported by this unadjusted contrast;
use `run(experiment, reader)` for those designs.

If outcomes are incomplete, wait for feeds through experiment end. If an
artifact lacks trigger membership, republish it with `trigger_population`.
Find exact reasons and recovery steps in
[when it refuses](../reference/sitewide-calculation.md#when-it-refuses).

## Worked example

The [complete native example](../reference/sitewide-calculation.md#worked-example)
creates definitions and raw events, including off-experiment activity,
then computes the mean and ratio results interpreted above.

### A cluster-randomized rollout

Continue that example with the
[store rollout](../reference/sitewide-calculation.md#a-cluster-randomized-rollout)
to see how 50 clusters change interval uncertainty without changing the
point-estimate calculation.
