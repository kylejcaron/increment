# Heterogeneity and rollout

After reading a breakout view, use `segment_heterogeneity` to ask whether effects differ across segments, and `segment_rollout_recommendation` to rank segments against a rollout cost. Both require estimates from the same breakout readout; combining runs silently merges the analysis.

For trigger-declared experiments, breakout output includes assigned and triggered populations. The functions keep these populations separate. Filter to one population when that is the question you intend to answer.

??? info "Population and summary fields"
    Both consumers include `analysis_population` in their grouping keys and preserve it on summary and segment outputs, so the two populations are never pooled.

## `segment_heterogeneity`: are segment effects different?

```python
import numpy as np
import polars as pl
from increment import MetricSpec, Scope, from_frame, run
from increment.estimators import segment_heterogeneity, segment_rollout_recommendation

rng = np.random.default_rng(21)
n = 4000
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
segment = rng.choice(["us", "eu", "apac"], size=n)
seg_lift = {"us": 0.10, "eu": 0.02, "apac": -0.01}
base = 20.0
revenue = (
    base
    + np.array([seg_lift[s] for s in segment]) * base * (variant == "treatment")
    + rng.normal(0, 6, n)
)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": variant,
        "segment": segment,
        "revenue": revenue,
    }
)

experiment, resolver = from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="revenue", type="mean")],
    breakouts=["segment"],
)
breakout_view = Scope(view="breakout", grain="total")
breakout_estimates = run(experiment.with_views(breakout_view), resolver).readout(breakout_view)
summary, segments = segment_heterogeneity(breakout_estimates)
for row in summary:
    print(f"{row.metric} dim={row.dimension} Q={row.q:.2f} tau2={row.tau2} i2={row.i2}")
```

<!-- output -->
```text
revenue dim=segment Q=26.15 tau2=0.0031985694898157... i2=None
revenue dim=segment Q=27.14 tau2=1.36529161872605... i2=None
```

The summary has separate relative (log-RR) and absolute-scale rows; they can disagree. Outcome-based exclusions suppress `tau2` and all `i2` fields because excluding outcomes can bias heterogeneity estimates. The `i2` point estimate is unavailable below five live segments; `i2` bounds are available from three, but not two.

??? info "Heterogeneity statistics and exclusions"
    The function returns two rows: one for the relative (log-RR) heterogeneity pass and one for the absolute pass. They can disagree because a between-segment spread that looks large on one scale can look small on the other. `tau2`, `i2`, `i2_lb`, and `i2_ub` all return `None` when an outcome-based exclusion affects that (key, scale); an outcome exclusion would bias `tau^2`, so none of the four is trustworthy. Independently, `i2`'s point estimate is also `None` below `k=5` live segments on a scale, even with zero exclusions (as in this example's `k=3`). Under a true null, P(I² > 25%) is still ~25--26% with so few segments, so the point estimate produces false positives too often to report. `i2_lb`/`i2_ub` remain available down to `k=3`; they are undefined (`None`) only at `k=2`, where the Higgins-Thompson interval has no closed form.

## Shrinkage and numerical integration

For each estimable segment and scale, `segments` includes a raw estimate and a shrunken estimate. Shrinkage uses a random-effects model; its reported interval summarizes posterior moments, not posterior quantiles.

??? info "Statistical details"
    `segments` carries two rows per estimable `(segment, scale)`: `estimator="raw"` (the segment's own estimate) and `estimator="shrunken"` (a posterior estimate integrated by adaptive quadrature: a REML marginal likelihood for tau, times a `HalfNormal(tau_prior_scale)` prior, mixed against Morris's (1983) conditional posterior for each segment -- not the HKSJ pooled mean, which feeds only the *pooled* row in `HeterogeneitySummary` above). The reported interval is a Wald summary of the posterior moments, not a posterior-quantile interval.

    Integration adapts both support and resolution, with posterior-tail and moment convergence checks. A posterior extending beyond eight prior scales is no longer clipped or withheld merely for crossing that boundary. The same integration supplies shrinkage and rollout pricing.


!!! warning "Prior scale and unresolved integration"
    `tau_prior_scale` sets the prior width and changes the model; it is not an integration ceiling. The default `0.30` is on the log-RR scale. For absolute effects, choose a value in the metric's units; a narrow prior can strongly shrink large absolute effects even when it is integrated accurately.

    If the numerical budget cannot resolve a posterior, heterogeneity keeps the raw rows but withholds that scale's shrunken values with `excluded="estimation_failed"` and warning code `breakout.heterogeneity.posterior_integration_unresolved`. The rollout functions instead propagate `InvalidRequestError` with code `estimation.meta.posterior_integration_unresolved`; this numerical failure is not an ordinary noise-offset refusal or a recommendation against rollout.

## `segment_rollout_recommendation`: pricing the rollout

```python
recommendations, rollout_segments = segment_rollout_recommendation(breakout_estimates)
for row in recommendations:
    print(f"{row.metric} dim={row.dimension}")
for row in rollout_segments:
    print(f"  {row.dimension_value:<6} selected={row.selected} excluded={row.excluded}")
```

<!-- output -->
```text
revenue dim=segment
  apac   selected=False excluded=None
  eu     selected=True excluded=None
  us     selected=True excluded=None
```

Both APIs group by `estimand` and `value_scale`. `segment_heterogeneity` reports relative and absolute results for relative-primary families, but only absolute results for absolute-primary families. `segment_rollout_recommendation` prices only relative-primary families and skips absolute-scale families, including encouragement LATE.

??? info "Grouping and computed scale"
    Both APIs group by `estimand` and upstream `value_scale`. `segment_heterogeneity` identifies its computed scale with `scale`: relative-primary families can produce relative and absolute results; absolute-primary families produce only absolute results, using their additive moments. `segment_rollout_recommendation` prices only relative-primary families. It skips every `value_scale="absolute"` family, including encouragement LATE, even when the shared `lift.log_mean`/`lift.log_se` fields contain additive moments.

Cost precedence is: explicit `rollout_cost=`; the matching metric's declared `rollout_cost`; then `0.0` (select any segment estimated favorable). A mapping with an unknown metric is refused. Frame-path `MetricSpec` has no `rollout_cost` field, so passing its `increment.workflows.metrics(experiment, resolver)` resolves to `0.0`. Costs are not weighted by exposure; small and large segments count equally.

??? info "Supplying declared costs and reading recommendations"
    Cost resolution is highest-precedence-first: an explicit `rollout_cost=` argument (a mapping's unknown metric name is refused), then the metric's own declared `rollout_cost` -- read from whichever `Metric` objects are passed via `metrics=`, matched to each estimate's metric name -- then `0.0` (ship any segment estimated favorable at all). This guide's own `increment.workflows.metrics(experiment, resolver)` (synthesized from the frame-path `MetricSpec` the example declares) can never carry a declared cost: `MetricSpec` has no `rollout_cost` field to set, so passing `metrics=increment.workflows.metrics(experiment, resolver)` here always resolves to `0.0` too. An arbitrary matching `Metric` object built with its own `rollout_cost` set is honored the same way regardless of how the estimates themselves were produced. Values are unweighted by exposure: a segment with few units and a wide interval is treated the same as a well-powered one on this pass. Read `RolloutRecommendation`'s field docs before quoting a rollout number in a launch review.

## Scale dependence

For relative-primary families, `segment_heterogeneity` runs independent relative
and absolute passes over the same segments. They can disagree about which
segments look heterogeneous because percentage and dollar effects measure
different quantities with different natural spread. Absolute-primary families
have only the absolute pass. Its `tau_prior_scale` is one
setting shared by both passes, so choose a value in the units of the axis
you are interpreting (relative log-RR or the metric's absolute units).

`segment_rollout_recommendation` has no absolute pass to share that setting.
Its `tau_prior_scale` is always interpreted on the log-RR scale (see the
pricing section above), never in a metric's absolute units, even though its
result rows carry the same `value_scale` label as
`segment_heterogeneity` rows.

## Caveats

DerSimonian-Laird followed by an unmodified Hartung-Knapp-Sidik-Jonkman adjustment, with no variance floor, can be anticonservative with few segments. Read [Meta-analysis can be anticonservative at small K](../limitations.md#meta-analysis-can-be-anticonservative-at-small-k) before trusting a pooled interval from three or four segments.

## Categorical adjustment in individual targeting

`validate_cate`, `targeting_rule` and `select_targeting_rule` take the
`(experiment, resolver)` pair from a builder and accept
categorical string adjustment columns on observational designs. Their
doubly robust nuisance fits learn modal-reference indicators within each
training split; held-out outcomes never choose the encoding. These targeting
routes retain `missing="refuse"`; categorical support does not relax their
identification, overlap, or honest-validation requirements. See
[observational adjustment](observational.md) for source-specific declarations
and missing-value policies.

The lower-level array functions also accept custom five-argument `psi_fn`
callbacks, typed as `PsiFn`. Import `PsiFn` and `ScoreDesign` from
`increment.estimation`; annotate the callback's third argument as
`X: ScoreDesign`. Its numeric `X` exposes `X.columns` (indicator names such as
`region=west`) and `X.sources` (the original adjustment column for each).
One basis is fitted on the training half and reused across scoring calls;
nested selection fits it on the inner half, never on the outer test.
Row-only indexing and copies preserve the names; views changing column axes
clear them rather than attach incorrect labels. Custom callbacks remain
responsible for their own honest or frozen nuisance predictions.

When a nuisance fit or custom-score basis encounters a held-out level it
never trained on, it uses its reference-level encoding and emits
`estimation.targeting.unseen_level_advisory`.
The warning identifies the covariate, level, distinct scored rows and affected
fits. Encoding is deterministic; the advisory is not evidence of overlap and
does not silently exclude those rows.
