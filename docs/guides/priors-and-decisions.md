# Priors and Bayesian decisions

`run(prior=...)` adds a posterior for probability-based decisions without changing the sampling estimate, interval, reference, p-value, or significance verdict. Use the posterior fields and accessors for Bayesian summaries; ordinary `lift`, `stat_sig()`, and `p_value()` remain prior-free sampling results.

## `RelativeLiftPrior`: specify fractional relative lift

Use an explicit wrapper when beliefs describe the relative change itself:

```python
from increment.estimators import MixturePrior, Normal, RelativeLiftPrior, StudentTPrior

relative_normal = RelativeLiftPrior(distribution=Normal(mu=0.0, sigma=0.10))
relative_heavy_tail = RelativeLiftPrior(distribution=StudentTPrior(nu=4, scale=0.10))
relative_mixture = RelativeLiftPrior(
    distribution=MixturePrior(weights=(0.25, 0.75), means=(-0.10, 0.20), sigmas=(0.05, 0.10))
)
```

The Normal declaration means mean zero and standard deviation **10% relative
lift**, not ten percentage points in a conversion rate and not a Normal prior
on log lift. Fractions are literal: `0.10` means 10%, not `10`.
Student-t `scale` is not its standard deviation; the distribution remains
centered at zero. Its `k` parameter is retained for saved-input compatibility
but does not select an approximation for the wrapped Student-t posterior.

Pass a wrapper as the call-time `prior`, or bind it with
`MetricSpec(name="revenue", prior=relative_normal)`. The call-time value wins;
an explicit `prior=None` clears the declared prior for that run. Omission
continues to inherit the declaration.

Per-metric experiment definitions use explicit scale and family tags:

```yaml
metric: revenue
prior:
  type: relative_lift
  distribution:
    type: student_t
    nu: 4
    scale: 0.10
    k: 80
```

For a Normal, use `type: normal` with `mu` and `sigma`; for a mixture, use
`type: normal_mixture` with ordered `weights`, `means`, and `sigmas`.
New tags do not change the encoding or meaning of existing unwrapped priors.

The new posterior uses a Normal working likelihood directly on relative lift.
It is an approximation for eligible fixed-horizon randomized contrasts, not a
promise that posterior probabilities are calibrated for every outcome
distribution. Existing positivity, uncertainty and method checks still apply.
It does not add support for sequential inference, clustering, switchbacks,
observational adjustment or encouragement designs.

Support is unrestricted: neither prior nor posterior is truncated at -100%,
and losses are not capped at one. With nonnegative outcomes and a positive
baseline, mass below -100% is a limitation of this working approximation.
Signed outcomes can have such relative effects, but this declaration does not
expand the existing data-admission rules.

### What is updated

For an eligible contrast, write the observed relative change as
$x=\bar Y_T/\bar Y_C-1$. The working observation model is
$x\mid r\sim N(r,s_r^2)$, with $s_r=(1+x)s_{\log}$ from the retained
sampling uncertainty. The prior describes $r$ itself. A log coordinate used
to integrate a Student-t density is not a log-lift observation model.

For example, $x=0.20$, $s_r=0.10$ and a relative Normal prior with mean zero
and SD $0.10$ give posterior mean and median $0.10$ and SD
$\sqrt{0.005}$. The published posterior estimate is the **median** for every
family, including asymmetric mixtures; it need not equal the posterior mean.
Normal and mixture updates retain their updated parameters. Wrapped Student-t
updates integrate its continuous density rather than a fixed component grid.

This changes a distribution on an effect, not the metric's estimand. Repeated
control outcomes `[4, 16]` and treatment outcomes `[1, 29]` have arithmetic
means 10 and 15: sampling lift is +50%, regardless of the prior. Taking logs
of individual outcomes would answer a different question.

### Availability, replay and approximation

Check `posterior_available` and `posterior_reason_code` separately from
`sampling_available`. A numerical inability to construct a posterior preserves
the sampling result and valid sibling metrics. A later probability or loss
query can also be unavailable without invalidating the retained posterior.
Batch exports retain each query's value or its exact reason independently;
one failed query does not erase successful queries.
A finite median alone is not sufficient: if the saved center and spread
cannot retain accurate posterior probabilities, the posterior is unavailable
instead of silently replaying a different density.

Saved rows retain the relative model, original declaration and either updated
Normal/mixture parameters or the Student-t defining state. JSON and frame
replay use that state once, not a second likelihood update. Existing public
probability signatures and threshold validation remain unchanged: unrestricted
parameter support does not introduce a new percentile-input or query API.

The working model uses estimated uncertainty. Skew, low event counts, a noisy
baseline and dependence between an estimate and its estimated SE can affect
posterior calibration even when existing admission checks pass. Frozen
joint-model simulations check the numerical posterior under its assumed
Normal likelihood. Separate raw-outcome simulations found six distributional
discrepancies among thirteen measurable settings and one sparse setting with
no admitted experiments; seven fixed-truth raw/proxy contrasts were flagged.
Those diagnostics do not establish a universal calibration guarantee or a
new sample-size rule. In particular, informative-prior intervals need not
have 95% repeated-sampling coverage at every fixed true effect.

From a source checkout, reproduce the frozen raw-outcome diagnostics with
`python -m calibration.relative_prior_validation --phase all --output results.json`.
The runner preserves both historical registrations, seeds, excluded
populations and diagnostic flags while evaluating the production relative
Normal posterior. The separate slow working-model tests report their PIT
distance, inclusion count, exact interval and p-value when run with `-s`.

## Historical evidence: predict the next true effect

Before importing a historical model, distinguish uncertainty in its pooled
mean from variation among experiments. If the historical posterior for the
mean is exactly $N(0.02,0.01^2)$, between-experiment SD is known to be $0.10$,
and a new exchangeable true effect has independent Normal variation around
that mean, its predictive distribution is exactly
$N(0.02,0.01^2+0.10^2)$ under those assumptions:

```python
import math
from increment.estimators import Normal, RelativeLiftPrior

next_effect_prior = RelativeLiftPrior(distribution=Normal(mu=0.02, sigma=math.hypot(0.01, 0.10)))
```

| Target under this hypothetical model | SD | Equal-tail 95% interval | Probability above 10% |
| --- | ---: | --- | ---: |
| Pooled mean | 0.0100000000 | [0.0004003602, 0.0395996398] | 6.2209605743e-16 |
| Next true effect | 0.1004987562 | [-0.1769739427, 0.2169739427] | 0.2130073713 |
| Future noisy estimate, additionally assuming SE 0.05 | 0.1122497216 | [-0.2000054116, 0.2400054116] | 0.2380167465 |

Use the next-true-effect distribution as the effect prior, not the first or
third row. Future observation noise belongs in the likelihood; adding it to
the prior counts it twice. More historical studies can reduce uncertainty in
the pooled mean without eliminating between-experiment variation.

Known heterogeneity, a plug-in estimate and uncertainty about heterogeneity
are different models. With uncertain historical parameters, prediction
integrates over their joint posterior. It is generally neither a single
Normal nor a Student-t. A finite Normal mixture can describe a genuinely
discrete model or a separately assessed approximation; a family name does
not validate that approximation. A frequentist pooled estimate and SE are not
automatically a Bayesian predictive distribution. No fitting or automatic
conversion is performed here.

### Student-t spread and transformed historical models

For a zero-centered Student-t with degrees of freedom $\nu>2$, SD is
$a\sqrt{\nu/(\nu-2)}$, where $a$ is `scale`. For SD 10% and $\nu=4$, use
`StudentTPrior(nu=4, scale=0.10 * math.sqrt((4 - 2) / 4))` inside the wrapper.
Its scale is approximately 0.0707106781, not 0.10. There is no finite SD for
$\nu\leq2$, and no mean for $\nu\leq1$, although central quantiles exist.
The available Student-t family is centered at zero; a nonzero historical
center must not be silently discarded. `k` is not historical sample size or
uncertainty in heterogeneity.

A Normal prediction for $L=\log(1+r)$ instead gives
$r=\exp(L)-1$: a shifted lognormal distribution with support $r>-1$.
If $L\sim N(a,b^2)$, its relative median is $\operatorname{expm1}(a)$,
mean is $\operatorname{expm1}(a+b^2/2)$ and quantiles are
$\operatorname{expm1}(a+b\Phi^{-1}(p))$. Setting the same Normal parameters
inside `RelativeLiftPrior` does not reproduce that distribution. Even a
moment-matched relative Normal is an approximation with different shape and
support, not an exact scale translation.

Before treating historical experiments as exchangeable with this one, check:

- The same estimand, metric denominator, analysis unit, arithmetic-versus-log
  outcome definition, and fractional-versus-percentage-point scale.
- Compatible exposure windows, follow-up horizons, populations and
  interventions, including assignment versus triggering.
- The same treatment/control orientation. Reversing a relative ratio is not
  generally a sign flip; favorable direction does not redefine the ratio.
- Defensible study-specific sampling SEs and any dependence from shared
  controls, users, windows or repeated views.
- Selection, reporting and stopping biases; heterogeneity does not repair
  them automatically.
- A historical evidence set fixed independently of the current likelihood.
  Reusing current outcomes in both prior and update double-counts evidence.

These checks do not broaden source or estimator eligibility.


## Existing unwrapped priors: `Normal`, `StudentTPrior`, `MixturePrior`

Unwrapped declarations retain their historical, route-specific meanings.
On the randomized relative-lift routes below they describe log lift;
supported absolute-native routes retain their own units. The explicit
relative wrapper selects a different statistical model, not a compatibility
alias or an automatic migration of these inputs.

```python
import numpy as np
import polars as pl
import increment
from increment.estimators import MixturePrior, Normal, StudentTPrior

rng = np.random.default_rng(6)
n = 2500
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
revenue = 20.0 + np.where(variant == "treatment", 0.4, 0.0) + rng.normal(0, 9, n)
df = pl.DataFrame({"user_id": [f"u{i}" for i in range(n)], "variant": variant, "revenue": revenue})

# A skeptical prior: log-lift centered at 0, sigma=0.05 (~5% typical move).
skeptical = Normal(mu=0.0, sigma=0.05)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[increment.MetricSpec(name="revenue", type="mean", preferred_direction="increase")],
)
results = increment.run(experiment, reader, prior=skeptical).readout()
for r in results:
    print(f"sampling_lift={r.lift.value:+.2%} posterior_estimate={r.posterior_estimate!r}")
```

Pass `prior=` to `run(...)`, not to a builder such as `from_frame`. A call-time prior takes precedence over a metric's declared prior for that run; `prior=None` resets to flat. A differing override of a non-default declared prior emits one warning for that run. If omitted, `AnalysisPlan` or `ExperimentMetric` declarations supply the default.

Choose `Normal` for a single bell-shaped prior, `StudentTPrior` for heavier tails, or `MixturePrior` for a mix of shapes. The declared prior and updated posterior are stored separately.

!!! warning "Supported prior routes"
    `StudentTPrior` and `MixturePrior` support only unclustered randomized parallel-arm reads on log-relative lift for mean, conversion, ratio, retention, and quantile metrics. They refuse observational designs with `readout.observational.prior`, and adjustment/encouragement paths with `estimation.adjust.prior.type`. Use `Normal` only where the support matrix permits it.

    Fixed-horizon encouragement compliance and LATE support a `Normal` prior only when the row records a supported update. Sequential inference, declared clusters (including with `Normal`), and switchback contrasts do not accept priors. `StudentTPrior.nu` is capped at 200; use `Normal` for larger values.

??? info "Posterior construction and representation"
    `run(prior=...)` leaves the sampling estimate, interval, reference, p-value, and significance verdict equal to the same request with `prior=None`. It stores the posterior separately in `posterior_*` fields; use `posterior_estimate`, `posterior_lb`/`posterior_ub`, and the posterior probability accessors for Bayesian decisions. The row's ordinary `lift` remains the prior-free sampling construction. `stat_sig()` and `p_value()` are sampling decisions, not posterior summaries.

    `StudentTPrior(nu=4, scale=0.10)` and `MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.02, 0.20))` declare heavier-tailed or bimodal shapes instead of a single Normal.

    `Normal` uses a closed-form Normal-Normal conjugate update (a precision-weighted mean, with no expansion). `StudentTPrior` and `MixturePrior` expand through `.components()` into a `MixturePrior` on `k` Gauss-Laguerre nodes (`MixturePrior.components()` returns itself) and use a separate K-component mixture posterior. The declared input remains in `prior_spec`. The separately stored mixture posterior uses `posterior_components`, an exact ordered carrier of equal-length `weights`, `means`, and `sigmas` tuples. Component positions are retained even when a posterior weight is zero; weights are validated to sum to one within `1e-9` and are never normalized. Accessors replay these updated values directly, not the declaration or the prior-free sampling lift.

    `run(prior=...)` accepts all three, but `StudentTPrior` and `MixturePrior` support only unclustered randomized parallel-arm reads on the log-relative-lift scale produced by `mean`/`conversion`/`ratio`/`retention`/`quantile` rows (`infer_lift`/`estimate_lift`). Cluster-robust rows therefore never reach the Student-t/mixture checks. Their supported posterior construction is the randomized log-relative working likelihood; those routes do not gain mixture support from separating sampling and posterior payloads.

    Fixed-horizon encouragement compliance and LATE rows with a supported Normal prior likewise retain their prior-free `lift` and store the Normal posterior separately. The posterior is available only when the row records a supported update.

    No prior, including `Normal`, is settable on a switchback contrast call. `run()` accepts only `UNSET` for `prior`/`decision_method`/`sensitivity_methods` on a switchback source. `StudentTPrior.nu` is capped at 200: larger values are numerically unstable to expand and are within 0.6% of `Normal(mu=0, sigma=scale)`. Use `Normal` directly instead.

### Binary metrics and path coverage

A valid sampling result does not guarantee an available posterior. Check `posterior_available` and its reason before using a Bayesian summary.

!!! warning "Exact binary inference and declarations"
    An explicit `conversion_inference="finite_sample"` request with a prior remains refused; exact binomial inversion has no posterior for a prior to update. Per-metric/YAML declarations accept an unwrapped Normal or an explicitly tagged relative Normal, Student-t or mixture. Unwrapped Student-t and mixtures remain call-time options.

??? info "Binary construction and path coverage"
    Informative priors also work on ordinary conversion data. `conversion_inference="auto"` chooses from the four success/failure counts whether or not a prior is declared. Dense counts use the same prior-free delta-method construction (including its t reference) as an unadjusted mean; sparse counts use exact binomial test inversion. A supported posterior is stored separately for dense counts. For sparse counts, valid exact sampling inference remains available even when the existing approximate-posterior guard reports `posterior_available=False` and its exact reason.

    Zero-success, all-success and sufficiently imprecise binary samples can fail the approximate posterior's log-mean/SE requirements even when the prior-free binomial route can report a confidence set.

    The Normal, Student-t and mixture cases are compared across `from_definitions`, `from_unit_day_artifact`, `from_frame` summaries and panels, and `from_moments` on identical mean, ratio, conversion and CUPED data, including posterior probabilities and prior metadata. The warehouse cases run on DuckDB and PostgreSQL. This is fixed-horizon parallel-arm coverage, not a switchback or sequential posterior claim.

## `prob_favorable`, `p_value`, and `stat_sig`

Use `stat_sig()` for the row's sampling decision, `p_value()` for a sampling p-value, and `prob_favorable()` for a stored posterior probability. An unavailable verdict is not a non-rejection.
For a neutral metric, neither accessor assigns a favorable side: `prob_favorable()` and `chance_to_beat_favorable()` return `None`, and when a posterior is available the readout displays its estimate with the direction-free `chance_to_beat()` probability. For a metric that declares `preferred_direction`, the readout, `render(result, "full")` and the table column `posterior_chance_to_beat` show `chance_to_beat_favorable()`, the posterior probability that the lift lies on its favorable side of zero; the test `alternative` never chooses that side. A decrease metric reads the posterior's lower tail directly, so a vanishing probability is reported at its size rather than as 0.

!!! warning "Probability access is not sequential inference"
    `prob_favorable()` and `p_value()` require fixed-horizon inference; sequential rows refuse with `CapabilityError` code `sequential.route.unsupported`. `prob_favorable()` also needs the metric's `preferred_direction`, not its test `alternative`. Posterior accessors return `None` when no supported posterior is stored; they never reconstruct one from sampling evidence. A selected sequential interval also refuses `prob_beyond`.

??? info "Decision semantics and refusal details"
    `stat_sig()` reads the row's own interval directly -- `lift.lb`/`lift.ub` against `null_lift` (or `abs_lb`/`abs_ub` against `null_abs` on an absolute-margin row) -- and works on every row's own decision: fixed-horizon or sequential, Normal-referenced or cluster-robust. It is tri-state: `None` when the row carries no usable sampling evidence (a failed cell, or `sampling_available` False), the same value the readout frame reports, so an unavailable verdict is never mistaken for a non-rejection.

    An exact-binomial row (`reference_kind="binomial"`) instead recomputes the test its interval was inverted from out of its persisted counts, so the two agree, and it refuses exactly where a fresh estimate of those counts at that null would (an extreme alpha at arm sizes whose float margin dominates the tail level, unless the null is one the control arm's own bound rejects; see the limitations page); a row persisted under an earlier construction of that test, or without naming its construction, is refused when read rather than shown beside a verdict its endpoints can contradict. `BreakoutEstimate` and `DailyLiftEstimate` carry the same two methods, computed by the same rule.

    `prob_favorable()` reads the stored posterior only when it is available. `p_value()` instead reads the prior-free sampling construction and remains a sampling p-value whether or not a prior was declared. Both require `inference == "fixed"`: a sequential row (`AlwaysValid`/`AsymptoticMean`) refuses with `CapabilityError` code `sequential.route.unsupported`, since a confidence sequence has no fixed endpoint distribution or posterior to summarize. `p_value()` additionally handles a cluster-robust (`reference_kind="t"`) row directly against its own t reference. Posterior access follows `posterior_available`: declared cluster-prior requests remain refused, and an unavailable posterior is never reconstructed from cluster sampling fields. `prob_favorable()` also needs `preferred_direction` declared on the metric (`"increase"`/`"decrease"`/`"neutral")` -- it is not derived from `alternative`, which can point the opposite way on a harm/futility test.

    The posterior probability accessors (`chance_to_beat`, `prob_beyond`, `prob_within`, and `risk_if_shipped`) never reconstruct a posterior from sampling interval endpoints and never infer one from a p-value. They return `None` when no supported posterior is stored. A selected sequential interval whose alpha was widened by FCR selection also refuses `prob_beyond` with `sequential.route.unsupported`; selection does not define a posterior probability.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

!!! warning "Prior exclusions are checked before estimation"
    Priors do not combine with sequential inference or declared clusters. Sequential coverage is a frequentist guarantee; cluster rows use a joint reference. These requests refuse before estimation with `sequential.route.unsupported` or `arm.adjustment.cluster_prior`. See [Sequential inference](sequential-inference.md).

??? info "How exclusions apply"
    A prior and sequential inference (`AlwaysValid`/`AsymptoticMean`) cannot compose: sequential coverage is a frequentist martingale guarantee. Declaring both -- an `AnalysisPlan(inference=...)` combined with a `run(prior=...)` override or a per-metric declared prior -- raises `sequential.route.unsupported`; a per-metric prior also names its `metric` and `inference` in the error context. A prior also cannot compose with a declared cluster: clustered rows use a joint reference built from cluster totals, and separating payloads does not authorize a cluster prior. Combining `cluster=` with a prior raises `arm.adjustment.cluster_prior`. In both cases the request is refused before estimation.

## Per-metric priors

`MetricSpec(prior=...)` and YAML `ExperimentMetric` bindings declare a metric's default prior. A call-time `run(prior=...)`, including `run(prior=None)`, overrides every metric for that call only.

??? info "Prior declaration and override guarantees"
    `prior=` belongs on `run(...)`, not on `from_frame`/`from_definitions`/etc. The builder assembles the source and its declared metrics. `run(prior=...)` overrides the epistemic policy for that call; `AnalysisPlan`/`ExperimentMetric` declarations provide the default when `prior` is `UNSET`, and `prior=None` resets it to flat.

    `MetricSpec(prior=...)` declares one metric's own prior on the dataframe path. It applies only when the call itself leaves `prior` at its default `UNSET`: an *explicit* `run(prior=...)` call -- including `run(prior=None)` -- overrides every metric's prior for that call, including metrics that declared their own `MetricSpec.prior`, the same call-wide-overrides-per-metric-declaration precedence `decision_method`/`sensitivity_methods` use (see [The data model](data-model.md)).

    The same override applies to priors declared on YAML `ExperimentMetric` bindings. An informative prior changes the separately stored posterior, not the prior-free sampling evidence used for an otherwise eligible secondary's configured multiplicity family; method-compatibility checks still apply. Overrides do not change the declaration: a later call that omits `prior` uses the declared prior again for posterior inference. This also works after reloading older moment exports with a declared prior. An explicit no-family policy remains excluded.
