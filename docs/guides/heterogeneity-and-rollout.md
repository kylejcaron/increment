# Heterogeneity and rollout

Use one breakout readout to ask whether segment effects differ and whether
segment-level rollout looks worthwhile. Do not combine estimates from
different runs: the consumers would silently merge their analyses.

For individual targeting rather than segment summaries, use the
[separate targeting route](#categorical-adjustment-in-individual-targeting).
Assigned and triggered populations stay separate through `analysis_population`;
filter to the population your question concerns.

## `segment_heterogeneity`: are segment effects different?

```python
import numpy as np
import polars as pl
import increment
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

experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[increment.MetricSpec(name="revenue", type="mean")],
    breakouts=["segment"],
)
breakout_view = increment.View.breakout(())
breakout_estimates = increment.run(
    experiment.with_views(breakout_view), reader, views=(breakout_view,)
).readout(breakout_view)
summary, segments = segment_heterogeneity(breakout_estimates)
for row in summary:
    print(
        f"{row.metric} dim={row.dimension} scale={row.scale} Q={row.q:.2f} tau2={row.tau2} i2={row.i2}"
    )
```

<!-- output -->
```text
revenue dim=segment scale=relative Q=26.15 tau2=0.0031985694898157... i2=None
revenue dim=segment scale=absolute Q=27.14 tau2=1.36529161872605... i2=None
```

Relative-primary families get independent relative (log-RR) and absolute
passes; absolute-primary families get only the absolute pass. They can
disagree because percentage and additive effects have different spread.
`value_scale` identifies the input family; `scale` identifies the computed
pass. Both consumers also group by `estimand` and population.

`tau2` measures between-segment spread. An outcome-based exclusion withholds
`tau2`, `i2`, `i2_lb` and `i2_ub` for that group/scale: selecting outcomes
would bias those summaries. Independently, `i2` is unavailable below five
live segments; its bounds remain available at three or four, but not two.
This example has three, so `i2=None` does not mean zero heterogeneity.
Under a true null, few-segment checks still gave roughly 25–26% probability
of `I² > 25%`, motivating withholding the point estimate.

## Shrinkage and numerical integration

Each estimable segment/scale gets `estimator="raw"` and `"shrunken"` rows.
The shrunken interval is a Wald summary of posterior moments, **not posterior
quantiles**. It mixes Morris’s conditional segment posterior over a REML
likelihood for tau times a `HalfNormal(tau_prior_scale)` prior. The HKSJ
pooled mean belongs to `HeterogeneitySummary`, not this segment posterior.

Integration adapts support and resolution, checking posterior tails and
moments. Crossing eight prior scales is not itself a clipping or refusal
rule. The same integration supplies shrinkage and rollout pricing.

!!! warning "Choose the prior scale in the units you interpret"
    `tau_prior_scale` changes the model, not the integration ceiling. The
    default 0.30 is on the log-RR scale. Heterogeneity shares this one setting
    across its passes; choose deliberately for the relative or absolute
    units you interpret. A narrow absolute prior can strongly shrink large
    effects despite accurate integration.

An unresolved posterior preserves raw heterogeneity rows but withholds
shrunken values on that scale with `excluded="estimation_failed"` and warning
`breakout.heterogeneity.posterior_integration_unresolved`. Rollout instead
raises `InvalidRequestError` with `estimation.meta.posterior_integration_unresolved`.
That is a numerical failure, not a noise-offset refusal or a recommendation
against rollout.

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

This toy example selects eu and us at the default zero rollout cost. It is
not a launch rule independent of its prior, metric and cost assumptions.
Rollout prices **relative-primary families only**, skipping every
`value_scale="absolute"` family, including encouragement LATE—even if shared
`lift.log_mean`/`lift.log_se` fields hold additive moments.

Cost precedence is:

1. Explicit `rollout_cost=`; a mapping with an unknown metric is refused.
2. The matching `Metric` object’s declared `rollout_cost`, supplied through `metrics=`.
3. Zero: select any segment estimated favorable.

Frame `MetricSpec` has no `rollout_cost` field, so this example’s
`increment.workflows.metrics(experiment, reader)` supplies no declared cost
and also resolves to zero. A separately supplied matching `Metric` with a
cost is honored regardless of the estimates’ source. Costs are unweighted
by exposure: small and large segments count equally. Read
`RolloutRecommendation`’s field definitions before quoting a rollout value.

## Scale dependence

Rollout has no absolute pass: its `tau_prior_scale` always uses log-RR units,
never the metric’s absolute units. Do not infer otherwise from its
`value_scale` label shared with heterogeneity rows.

## Caveats

DerSimonian-Laird followed by an unmodified Hartung-Knapp-Sidik-Jonkman adjustment, with no variance floor, can be anticonservative with few segments. Read [Meta-analysis can be anticonservative at small K](../limitations.md#meta-analysis-can-be-anticonservative-at-small-k) before trusting a pooled interval from three or four segments.

## Categorical adjustment in individual targeting

`validate_cate`, `targeting_rule` and `select_targeting_rule` take the
`(experiment, reader)` pair from a builder and accept
categorical string adjustment columns on observational designs. Their
doubly robust nuisance fits learn modal-reference indicators within each
training split; held-out outcomes never choose the encoding. These targeting
routes retain `missing="refuse"`; categorical support does not relax their
identification, overlap, or honest-validation requirements. See
[observational adjustment](observational.md) for source-specific declarations
and missing-value policies.

Like `estimate_cate`, they contrast the experiment's declared control arm
unless `control=` names another. Covariates must be measured before exposure,
so on a dataframe experiment all four refuse a column the metric's value is
read from (its `value_column`, or a ratio's numerator and denominator) as an
`interact` or `adjust` covariate with `cate.covariate.is_outcome`, before any
data is read. A column that only shares the metric's name is accepted.

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
