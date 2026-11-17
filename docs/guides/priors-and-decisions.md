# Priors and Bayesian decisions

A sampling result describes the observed effect and its uncertainty without
a prior. A posterior combines a prior with a working model for the data.
Use `run(prior=...)` when you want posterior probabilities as well as the
sampling result; it never changes `lift`, its confidence interval,
`p_value()` or `stat_sig()`.

## Run and read a posterior

This complete example declares a skeptical prior directly on relative lift:
mean zero and standard deviation 10%. Revenue is measured per user, and an
increase is favorable.

```python
import polars as pl
import increment
from increment.estimators import Normal, RelativeLiftPrior

df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(40)],
        "variant": ["control"] * 20 + ["treatment"] * 20,
        "revenue": [8.0, 12.0] * 10 + [10.0, 14.0] * 10,
    }
)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[increment.MetricSpec(name="revenue", type="mean", preferred_direction="increase")],
)
prior = RelativeLiftPrior(distribution=Normal(mu=0.0, sigma=0.10))
row = next(iter(increment.run(experiment, reader, prior=prior).readout()))
print(f"sampling lift: {row.require_lift().value:+.1%}")
if row.posterior_available:
    print(f"posterior median: {row.posterior_estimate:+.1%}")
    print(f"posterior interval: [{row.posterior_lb:+.1%}, {row.posterior_ub:+.1%}]")
    print(f"probability of an increase: {row.chance_to_beat():.1%}")
else:
    print(row.posterior_reason_code, row.posterior_reason_context)
```

<!-- output -->
```text
sampling lift: +20.0%
posterior median: +13.2%
posterior interval: [+1.8%, +24.6%]
probability of an increase: 98.8%
```

Control averages 10 and treatment 12, so the sampling lift is +20%.
The skeptical prior pulls the posterior median to +13.2%, with a 98.8%
probability of an increase **under that prior and working model**.
The posterior interval is not another confidence interval, and 98.8% is not
a sampling significance verdict. The published posterior estimate is its
median, which need not equal its mean for an asymmetric mixture.

!!! warning "An available posterior is not a calibration guarantee"
    This uses a Normal working likelihood with estimated uncertainty.
    Skew, sparse events, a noisy baseline and dependence between the estimate
    and its SE can affect posterior calibration. Informative-prior intervals
    need not have 95% repeated-sampling coverage at every fixed true effect.
    See [the approximation and its measured limits](../reference/bayesian-results.md#availability-replay-and-approximation).

## `RelativeLiftPrior`: specify fractional relative lift

`RelativeLiftPrior(distribution=Normal(mu=0.0, sigma=0.10))` means a prior on
the relative change itself. `0.10` is 10%, **not** ten percentage points of a
conversion rate, and **not** a Normal prior on log lift.

The wrapper also accepts a zero-centered `StudentTPrior` or a `MixturePrior`.
Student-t `scale` is not its standard deviation; its `k` does not choose an
approximation for the wrapped posterior. Use the
[family declarations and formulas](../reference/bayesian-results.md#relativeliftprior-specify-fractional-relative-lift)
to choose them deliberately.

Neither the relative prior nor its posterior is truncated at -100%, and
losses are not capped at one. Mass below -100% is a limitation of this
approximation for nonnegative outcomes with a positive baseline. Signed
outcomes can have such effects, but the wrapper does not broaden admission
rules.

### What is updated

The likelihood describes the arithmetic treatment/control relative change,
not the mean of logged outcomes. The
[working-model formula](../reference/bayesian-results.md#what-is-updated)
defines its uncertainty and how each prior family is updated.

### Availability, replay and approximation

Check `posterior_available` and `posterior_reason_code` separately from
`sampling_available`. Sampling inference can be usable when the posterior
is unavailable. A failed probability or loss query can likewise leave the
posterior and other queries intact; unavailable is never zero probability.

Saved results replay their retained update, not a second update of the prior.
See [saved-state and query contracts](../reference/bayesian-results.md#availability-replay-and-approximation)
for JSON, frames, exports and trusted pickle.

## Historical evidence: predict the next true effect

Use a prior for the **next experiment's true effect**, not uncertainty in the
historical pooled mean or a future noisy estimate. More studies can locate
the pooled mean precisely without removing variation among experiments.

For example, suppose the historical mean's posterior is exactly
$N(0.02, 0.01^2)$, between-experiment SD is known to be 0.10, and a new
exchangeable true effect varies independently and Normally around that mean:

```python
import math
from increment.estimators import Normal, RelativeLiftPrior

next_effect_prior = RelativeLiftPrior(distribution=Normal(mu=0.02, sigma=math.hypot(0.01, 0.10)))
```

Its SD is about 10.05%, not 1%. Do not add the next experiment's sampling SE
to this prior: that noise belongs in the likelihood. Unknown heterogeneity
requires integrating over its uncertainty; the result is generally neither
a single Normal nor a Student-t. Increment does not fit or automatically
convert a historical model for you.

Check that historical studies match the metric, denominator, analysis unit,
scale, follow-up window, population and intervention. Preserve treatment/control
orientation: reversing a relative ratio is not generally a sign flip.
Account for shared data and selection, reporting or stopping biases.
Never reuse current outcomes in both the historical prior and likelihood.
See the [predictive-distribution comparison and full checklist](../reference/bayesian-results.md#historical-evidence-predict-the-next-true-effect).

### Student-t spread and transformed historical models

For a Student-t with $\nu>2$, SD is `scale` times $\sqrt{\nu/(\nu-2)}$;
its available family is centered at zero. A historical nonzero center must
not be discarded. A Normal prediction on $\log(1+r)$ is not the same as
a relative Normal with identical parameters. Use the
[spread and transformation formulas](../reference/bayesian-results.md#student-t-spread-and-transformed-historical-models).

## Existing unwrapped priors: `Normal`, `StudentTPrior`, `MixturePrior`

Unwrapped priors keep their existing route-specific meanings: on randomized
relative-lift routes they describe **log lift**; supported absolute-native
routes use their own units. Adding `RelativeLiftPrior` selects a different
model, not an alias or automatic migration.

The legacy Student-t uses a finite Normal-mixture approximation; the wrapped
Student-t integrates its continuous density. See
[legacy construction and support](../reference/bayesian-results.md#existing-unwrapped-priors-normal-studenttprior-mixtureprior)
before reusing an old declaration.

### Binary metrics and path coverage

`conversion_inference="auto"` selects its sampling procedure from the event
counts, independently of the prior. A sparse exact-binomial sampling result
can survive while its approximate posterior is unavailable. Explicit
`conversion_inference="finite_sample"` with a prior is refused.
See [binary availability](../reference/bayesian-results.md#binary-metrics-and-path-coverage).

## `prob_favorable`, `p_value`, and `stat_sig`

| Question | Read |
|---|---|
| Does the row meet its sampling test? | `stat_sig()` |
| What is its sampling p-value? | `p_value()` |
| What is the posterior probability of positive lift? | `chance_to_beat()` |
| What is the posterior probability on the favorable side of zero? | `chance_to_beat_favorable()` |

`prob_favorable()` uses the favorable side of the row's declared null, which
can differ from zero. Favorability comes from `preferred_direction`, never
the test `alternative`. A neutral metric has no favorable side; those
favorable-probability accessors return `None`.

A failed sampling cell gives `stat_sig() is None`, not a non-rejection.
Posterior accessors never infer a posterior from a p-value or interval.
Read [exact probability and decision rules](../reference/bayesian-results.md#prob_favorable-p_value-and-stat_sig)
for absolute margins, sequential refusals and query thresholds.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

Priors do not combine with sequential inference or declared clusters:
requests refuse before estimation with `sequential.route.unsupported` or
`arm.adjustment.cluster_prior`. Switchback contrasts do not accept priors
either. The relative wrapper does not add observational or encouragement
support. Some unwrapped Normal routes differ; consult the
[prior support rules](../reference/bayesian-results.md#supported-prior-routes),
not the distribution's name.

## Per-metric priors

Use `MetricSpec(prior=...)` or a YAML `ExperimentMetric` binding for a default.
Call-time `run(prior=...)` overrides every metric for that run only;
`run(prior=None)` clears their priors, while omission inherits them.
A later run that omits the argument uses the declaration again.
Exact [encoding and override rules](../reference/bayesian-results.md#per-metric-priors)
also cover saved moment exports and multiplicity.
