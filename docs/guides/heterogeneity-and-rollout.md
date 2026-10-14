# Heterogeneity and rollout

After `run_breakout()`, use `segment_heterogeneity` to ask whether effects differ across segments, and `segment_rollout_recommendation` to rank segments against a rollout cost. Both require estimates from the same `run_breakout()` call; combining calls silently merges the analysis.

For trigger-declared experiments, breakout output includes assigned and triggered populations. The functions keep these populations separate. Filter to one population when that is the question you intend to answer.

## `segment_heterogeneity`: are segment effects different?

```python
import numpy as np
import polars as pl
from increment import Analysis, MetricSpec, segment_heterogeneity, segment_rollout_recommendation

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

analysis = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="revenue", type="mean")],
    breakouts=["segment"],
)

breakout_estimates = analysis.run_breakout()
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

## Shrinkage and numerical integration

For each estimable segment and scale, `segments` includes a raw estimate and a shrunken estimate. Shrinkage uses a random-effects model; its reported interval summarizes posterior moments, not posterior quantiles.

??? info "Statistical details"
    The shrunken estimate integrates a REML marginal likelihood for `tau` times a `HalfNormal(tau_prior_scale)` prior, mixed with Morris's (1983) conditional posterior for each segment. It is not the HKSJ pooled mean, which appears only in the pooled `HeterogeneitySummary` row. Integration adapts support and resolution and checks posterior tails and moment convergence; it does not clip or withhold a posterior merely because it extends beyond eight prior scales. The same integration supports rollout pricing.


`tau_prior_scale` sets the prior width and changes the model; it is not an integration ceiling. The default `0.30` is on the log-RR scale. For absolute effects, choose a value in the metric's units. If numerical integration cannot resolve the posterior, heterogeneity retains raw rows but withholds shrunken values (`breakout.heterogeneity.posterior_integration_unresolved`). Rollout instead raises `estimation.meta.posterior_integration_unresolved`; this is a numerical failure, not a recommendation against rollout.

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

Cost precedence is: explicit `rollout_cost=`; the matching metric's declared `rollout_cost`; then `0.0` (select any segment estimated favorable). A mapping with an unknown metric is refused. Frame-path `MetricSpec` has no `rollout_cost` field, so passing its `analysis.metrics` resolves to `0.0`. Costs are not weighted by exposure; small and large segments count equally.

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

`validate_cate`, `targeting_rule` and `select_targeting_rule` accept
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
