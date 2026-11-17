# Bayesian results: prior models, availability and replay

Use this reference to identify the model behind a posterior, check its availability
and replay saved results. For a complete analysis, start with
[Priors and Bayesian decisions](../guides/priors-and-decisions.md).

A declared prior changes only the stored posterior. Sampling estimates,
intervals, reference distributions, p-values and significance remain prior-free.

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
Malformed saved query parameters raise `estimation.posterior.invalid_request`;
they are not treated as numerical unavailability.
A finite median alone is not sufficient: if the saved center and spread
cannot retain accurate posterior probabilities, the posterior is unavailable
instead of silently replaying a different density.

Saved rows retain the relative model, original declaration and either updated
Normal/mixture parameters or the Student-t defining state. JSON and frame
replay use that state once, not a second likelihood update. Existing public
probability signatures and threshold validation remain unchanged: unrestricted
parameter support does not introduce a new percentile-input or query API.
Dates with no observations remain unavailable even when another date has a
posterior. Loading a previously saved readout preserves its recorded prior
meaning; newly added optional metadata does not create a posterior.
Saved result collections and envelopes preserve recorded posterior queries in
tables and CSV exports, including unavailable values and their exact codes and
contexts. Exporting does not replace those records with default queries.
Trusted pickle restoration fills missing optional fields without recalculating
saved inference. Pickle is not an untrusted-input format; JSON and Arrow have
separate validation boundaries.

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
distance, inclusion count, exact interval and p-value with:

```bash
make test TESTS=tests/estimation/test_relative_posterior.py \
  PYTEST_XDIST_ARGS= PYTEST_ARGS=-s
```

This runs without parallel workers so successful-test reports remain visible.

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
from increment.estimators import MixturePrior, Normal, StudentTPrior

log_normal = Normal(mu=0.0, sigma=0.05)
log_heavy_tail = StudentTPrior(nu=4, scale=0.10)
log_mixture = MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.02, 0.20))
```

### Posterior construction and representation

An unwrapped `Normal` uses a closed-form Normal–Normal conjugate update,
a precision-weighted mean with no expansion. An unwrapped `StudentTPrior`
uses `.components()` to approximate its mixing density on `k` Gauss–Laguerre
nodes; `MixturePrior.components()` returns the declared mixture itself.
Both use a K-component Normal-mixture update. This finite Student-t
approximation is not the continuous update used by `RelativeLiftPrior`.

The declaration remains separate from the update: `prior_spec` retains the
Student-t/mixture input; `posterior_components` stores ordered equal-length
`weights`, `means` and `sigmas` tuples. Positions survive even when an updated
weight is zero. Weights must sum to one within `1e-9` and are never normalized
on restoration. Accessors use the updated values, not another update of the
declaration or the prior-free sampling estimate.

### Supported prior routes

- Unwrapped Student-t and mixture priors support unclustered randomized
  parallel-arm log-relative mean, conversion, ratio, retention and quantile
  results (`infer_lift`/`estimate_lift`). Observational requests refuse with
  `readout.observational.prior`; adjustment/encouragement requests refuse with
  `estimation.adjust.prior.type`. Use an unwrapped Normal only where the
  [support matrix](capabilities-by-entry-point.md) permits it.
- Fixed-horizon encouragement compliance and LATE can store a Normal update
  while retaining their prior-free sampling result. A posterior is available
  only when that row records a supported update.
- A multicovariate CUPED prior does not change its sampling result. Its
  posterior is unavailable with
  `estimation.results.lift.t_posterior_sufficient_statistics`: the retained
  HC2/BM statistics do not support composing the declared prior. The context
  retains `metric`, `group_id`, `reference_df` and the missing-statistics reason.
- No prior, including Normal, is settable on a switchback contrast call.
  `run()` accepts only `UNSET` for `prior`, `decision_method` and
  `sensitivity_methods` on a switchback source.
- `StudentTPrior.nu` is capped at 200. The legacy expansion becomes numerically
  unstable beyond that, where the shape is within 0.6% of
  `Normal(mu=0, sigma=scale)`; use Normal instead. The same declaration range
  applies inside the relative wrapper.

### Binary metrics and path coverage

An explicit `conversion_inference="finite_sample"` request with a prior
refuses: exact binomial inversion has no posterior for a prior to update.
Per-metric/YAML declarations accept an unwrapped Normal or an explicitly
tagged relative Normal, Student-t or mixture. Unwrapped Student-t and mixtures
remain call-time options.

Informative priors also work on ordinary conversion data. `conversion_inference="auto"` chooses from the four success/failure counts whether or not a prior is declared. Dense counts use the same prior-free delta-method construction (including its t reference) as an unadjusted mean; sparse counts use exact binomial test inversion. A supported posterior is stored separately for dense counts. For sparse counts, valid exact sampling inference remains available even when the existing approximate-posterior guard reports `posterior_available=False` and its exact reason.

Zero-success, all-success and sufficiently imprecise binary samples can fail the approximate posterior's log-mean/SE requirements even when the prior-free binomial route can report a confidence set.

The Normal, Student-t and mixture cases are compared across `from_definitions`, `from_unit_day_artifact`, `from_frame` summaries and panels, and `from_moments` on identical mean, ratio, conversion and CUPED data, including posterior probabilities and prior metadata. The warehouse cases run on DuckDB and PostgreSQL. This is fixed-horizon parallel-arm coverage, not a switchback or sequential posterior claim.

## `prob_favorable`, `p_value`, and `stat_sig`

`stat_sig()` is the row's own sampling decision, not its multiplicity-family
selection. A failed cell or `sampling_available=False` gives `None`, not a
non-rejection. For fixed-horizon results, the decision compares the persisted
sampling test with the row's `decision_alpha`; sequential results read their
registered confidence set. See [sampling decisions](result-fields.md#significance-p-values-and-selection).

An exact-binomial row (`reference_kind="binomial"`) recomputes the test its
interval was inverted from, using persisted counts. It refuses where a fresh
estimate of those counts at that null would: an extreme alpha whose tail is
dominated by the arm-size float margin, unless the control arm's own bound
rejects that null. Saved rows naming an earlier test construction, or no
construction, refuse rather than display a verdict contradicting their
endpoints. `BreakoutEstimate` and `DailyLiftEstimate` use the same rule.

`p_value()` uses sampling evidence whether or not a prior was declared,
including a cluster-robust row's own t reference. `prob_favorable()` uses only
the retained posterior, at the row's `null_lift`. Both require fixed-horizon
inference; sequential rows raise `CapabilityError` with
`sequential.route.unsupported`. Neither a confidence sequence nor widening
one for FCR selection creates a posterior; a selected sequential interval
also refuses `prob_beyond` with that code.

`chance_to_beat()` instead asks whether lift exceeds zero.
`chance_to_beat_favorable()` uses the declared favorable side of zero.
For `preferred_direction="decrease"`, the lower tail is evaluated directly,
so very small probabilities do not disappear through subtraction from one.
`prob_favorable()` and `chance_to_beat_favorable()` return `None` for a
neutral metric or one without a declared direction. The test `alternative`
never supplies that direction; a harm/futility test may point the other way.
An absolute-margin null does not supply a relative `prob_favorable()` answer.

The readout, `render(result, "full")` and the table's
`posterior_chance_to_beat` show the probability on the favorable side of zero
when a direction is declared; neutral or undeclared directions use the
direction-free `chance_to_beat()` alongside the posterior estimate.
`posterior_prob_favorable` is a different summary: it refers to the declared
null. Its reason distinguishes an absolute-margin null, missing direction and
neutral direction.

`chance_to_beat`, `prob_beyond`, `prob_within` and `risk_if_shipped` never
reconstruct a posterior from sampling endpoints or a p-value. If no supported
posterior is stored they return `None`. Relative `prob_within(threshold)` still
requires `0 < threshold < 1`; unrestricted relative-prior support does not
relax this API. For an absolute result, that threshold must be positive.
Log-scale `prob_beyond` requires a threshold greater than -1.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

Increment does not accept an effect prior together with registered sequential
inference (`AlwaysValid`/`AsymptoticMean`). An `AnalysisPlan(inference=...)`
combined with a `run(prior=...)` override or a per-metric prior raises
`sequential.route.unsupported` before estimation. A per-metric prior also
names its `metric` and `inference` in the error context. The predictive priors
declared inside a [sequential model](../guides/sequential-inference.md) serve
a different purpose; they are not these effect priors.

Declared cluster-prior requests, including Normal, refuse with
`arm.adjustment.cluster_prior` before estimation. Posterior accessors do not
reconstruct a posterior from cluster sampling statistics.
This is a prior-admission rule, not a claim that every t-referenced row lacks
a posterior: ordinary unclustered delta-method rows can retain a supported
working-model update separately from their sampling t reference.

## Per-metric priors

`prior=` belongs on `run(...)`, not a builder. `MetricSpec(prior=...)` and YAML
`ExperimentMetric` bindings supply defaults when the call leaves `prior=UNSET`.
An explicit `run(prior=...)` overrides every metric for that call;
`run(prior=None)` clears the declaration for that run, without fabricating a
flat-prior posterior. A differing override of a non-default declared prior
emits one warning for the run.

Overrides never change the saved declaration. A later call omitting `prior`
uses it again, including after reloading older moment exports. A prior does
not change the prior-free sampling evidence used by an otherwise eligible
secondary's configured multiplicity family; method-compatibility checks still
apply, and an explicit no-family policy remains excluded.
