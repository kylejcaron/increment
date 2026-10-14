# Priors and Bayesian decisions

`run(prior=...)` adds a posterior for probability-based decisions without changing the sampling estimate, interval, reference, p-value, or significance verdict. Use the posterior fields and accessors for Bayesian summaries; ordinary `lift`, `stat_sig()`, and `p_value()` remain prior-free sampling results.

## Declaring a prior: `Normal`, `StudentTPrior`, `MixturePrior`

```python
import numpy as np
import polars as pl
from increment import Analysis, MetricSpec, Normal
from increment.estimation.priors import MixturePrior, StudentTPrior

rng = np.random.default_rng(6)
n = 2500
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
revenue = 20.0 + np.where(variant == "treatment", 0.4, 0.0) + rng.normal(0, 9, n)
df = pl.DataFrame({"user_id": [f"u{i}" for i in range(n)], "variant": variant, "revenue": revenue})

# A skeptical prior: log-lift centered at 0, sigma=0.05 (~5% typical move).
skeptical = Normal(mu=0.0, sigma=0.05)
analysis = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="revenue", type="mean", preferred_direction="increase")],
)
results = analysis.run(prior=skeptical)
for r in results:
    print(f"sampling_lift={r.lift.value:+.2%} posterior_estimate={r.posterior_estimate!r}")
```

Pass `prior=` to `run(...)`, not to an `Analysis` constructor. A call-time prior overrides the declared prior for that call; `prior=None` resets to flat. If omitted, `AnalysisPlan` or `ExperimentMetric` declarations supply the default.

`StudentTPrior(nu=4, scale=0.10)` and
`MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.02, 0.20))`
declare heavier-tailed or bimodal shapes instead of a single Normal.
`Normal` has a closed-form update. `StudentTPrior` and `MixturePrior` use a finite mixture representation and a separate mixture posterior. The original declaration remains in `prior_spec`; `posterior_components` stores the updated weights, means, and scales. Components keep their order, including zero-weight components, and weights must sum to one (they are not normalized).


`StudentTPrior` and `MixturePrior` support only unclustered randomized parallel-arm reads on log-relative lift for mean, conversion, ratio, retention, and quantile metrics. Any declared `cluster=` refuses all priors, including `Normal`. Student-t and mixture priors also refuse observational, adjustment, and encouragement paths. Use `Normal` only where the support matrix permits it.

Fixed-horizon encouragement compliance and LATE support a `Normal` prior while retaining prior-free `lift`. Sequential inference, clusters, and switchback contrasts do not accept priors. `StudentTPrior.nu` is capped at 200; larger values are numerically unstable and within 0.6% of a Normal with the same scale, so use `Normal` instead.

### Binary metrics and path coverage

Informative priors also work on ordinary conversion data. `conversion_inference="auto"`
chooses from the four success/failure counts whether or not a prior is
declared. Dense counts use the same prior-free delta-method construction
(including its t reference) as an unadjusted mean; sparse counts use exact
binomial test inversion. A supported posterior is stored separately for
dense counts. For sparse counts, valid exact sampling inference remains
available even when the existing approximate-posterior guard reports
`posterior_available=False` and its exact reason. An explicit
`conversion_inference="finite_sample"` request with a prior remains refused;
the exact binomial inversion has no posterior for a prior to update.

Zero-success, all-success and sufficiently imprecise binary samples can
fail the approximate posterior's log-mean/SE requirements even when the
prior-free binomial route can report a confidence set.

The Normal, Student-t and mixture cases are compared across
`from_definitions`, `from_unit_day_artifact`, `from_unit_summary`,
`from_unit_panel` and `from_moments` on identical mean, ratio, conversion
and CUPED data, including posterior probabilities and prior metadata.
The warehouse cases run on DuckDB and PostgreSQL. This is fixed-horizon
parallel-arm coverage, not a switchback or sequential posterior claim.
Per-metric/YAML prior declarations currently accept Normal priors;
Student-t and mixtures are call-time `run(prior=...)` options.

`stat_sig()` reads the row's sampling interval against its declared null. It returns `None` when sampling evidence is unavailable; an unavailable result is not a non-rejection. Exact-binomial rows recompute the test from persisted counts, and refuse when the interval's original construction cannot be recovered. `BreakoutEstimate` and `DailyLiftEstimate` follow the same rule.

`prob_favorable()` uses an available posterior; `p_value()` uses prior-free sampling evidence. Both require fixed-horizon inference. Sequential rows refuse with `sequential.route.unsupported`; a confidence sequence has no fixed endpoint distribution or posterior summary. `p_value()` supports cluster-robust t references, but posterior access does not reconstruct a posterior from cluster sampling fields. `prob_favorable()` also requires `preferred_direction` on the metric.

Posterior accessors (`chance_to_beat`, `prob_beyond`, `prob_within`, and `risk_if_shipped`) never infer a posterior from intervals or p-values. They return `None` when no supported posterior is stored. FCR-selected sequential intervals also refuse `prob_beyond`; selection does not create a posterior probability.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

Priors do not combine with sequential inference or declared clusters. Sequential coverage is a frequentist guarantee; cluster rows use a joint reference. These requests refuse before estimation with `sequential.route.unsupported` or `arm.adjustment.cluster_prior`. See [Sequential inference](sequential-inference.md).

## Per-metric priors

`MetricSpec(prior=...)` declares a dataframe metric's prior; YAML `ExperimentMetric` bindings can also declare priors. A call-time `run(prior=...)` overrides declarations for every metric, including `run(prior=None)`. Omitting `prior` on a later call uses the declared value again. Priors change the posterior, not prior-free sampling evidence or multiplicity selection. Per-metric/YAML declarations currently accept `Normal`; Student-t and mixtures are call-time options.
