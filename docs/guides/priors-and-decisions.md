# Priors and Bayesian decisions

`run(prior=...)` leaves the sampling estimate, interval, reference, p-value,
and significance verdict equal to the same request with `prior=None`. It
stores the posterior separately in `posterior_*` fields; use
`posterior_estimate`, `posterior_lb`/`posterior_ub`, and the posterior
probability accessors for Bayesian decisions. The row's ordinary `lift`
remains the prior-free sampling construction. `stat_sig()` and `p_value()`
are sampling decisions, not posterior summaries.

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
        f"sampling_lift={r.lift.value:+.2%} posterior={r.posterior_estimate:+.2%} "
        f"prob_favorable={r.prob_favorable():.3f}"
```

```text
sampling_lift=+0.00% posterior=+0.16% prob_favorable=0.538

`prior=` belongs on `run(...)`, not on `from_unit_summary`/
`from_definitions`/etc. The constructor assembles the source and its
declared metrics. `run(prior=...)` overrides the epistemic policy for that
call; `AnalysisPlan`/`ExperimentMetric` declarations provide the default
when `prior` is `UNSET`, and `prior=None` resets it to flat.

`StudentTPrior(nu=4, scale=0.10)` and
`MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.02, 0.20))`
declare heavier-tailed or bimodal shapes instead of a single Normal.
`Normal` uses a closed-form Normal-Normal conjugate update (a
precision-weighted mean, with no expansion). `StudentTPrior` and
`MixturePrior` expand through `.components()` into a `MixturePrior` on
`k` Gauss-Laguerre nodes (`MixturePrior.components()` returns itself) and
use a separate K-component mixture posterior.

`run(prior=...)` accepts all three, but `StudentTPrior` and `MixturePrior`
support only unclustered randomized parallel-arm reads on the
log-relative-lift scale produced by `mean`/`conversion`/`ratio`/
`retention`/`quantile` rows (`infer_lift`/`estimate_lift`). A declared
`cluster=` refuses every prior, including `Normal`; see
["Mutual exclusions"](#mutual-exclusions-prior-vs-sequential-inference-prior-vs-cluster).
Cluster-robust rows therefore never reach the Student-t/mixture checks.

`StudentTPrior` and `MixturePrior` also refuse observational designs with
`readout.observational.prior`. Their closed-form update uses the log-RR
scale reported by `infer_lift`/`estimate_lift`, while observational
estimators (`iptw`, `dml`, `aipw`) use a linear-relative parameterization.
This is a scale mismatch, not missing statistics: the observational path
persists the raw point estimate and standard error just as the randomized
path does. These priors also refuse adjustment and encouragement paths with
`estimation.adjust.prior.type`, where the raw pre-prior statistics are not
persisted in a form that can recompute mixture decision statistics. Pass
`Normal` in either case.

No prior, including `Normal`, is settable on a switchback contrast call.
`run()` accepts only `UNSET` for `prior`/`decision_method`/
`sensitivity_methods` on a switchback source. `StudentTPrior.nu` is capped
at 200: larger values are numerically unstable to expand and are within
0.6% of `Normal(mu=0, sigma=scale)`. Use `Normal` directly instead.

### Binary metrics and path coverage

Informative priors also work on ordinary conversion data. This uses a
Normal likelihood approximation for the observed **log risk ratio**, not
two binomial likelihoods. Without a prior, eligible conversion/retention
rows are routed by their counts (`Method.conversion_inference`, default
`"auto"` chooses from the four success/failure counts whether or not a
prior is declared. Dense counts use the same prior-free delta-method
construction (including its t reference) as an unadjusted mean; sparse
counts use exact binomial test inversion. A supported posterior is stored
separately for dense counts. For sparse counts, valid exact sampling
inference remains available even when the existing approximate-posterior
guard reports `posterior_available=False` and its exact reason. An explicit
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

## `prob_favorable`, `p_value`, and `stat_sig`

`stat_sig()` reads the row's own interval directly -- `lift.lb`/`lift.ub`
against `null_lift` (or `abs_lb`/`abs_ub` against `null_abs` on an
absolute-margin row) -- and works on every row's own decision: fixed-
horizon or sequential, Normal-referenced or cluster-robust. An exact-binomial
row (`reference_kind="binomial"`) instead recomputes the test its interval was
inverted from out of its persisted counts, so the two agree, and it refuses
exactly where a fresh estimate of those counts at that null would (an extreme
alpha at arm sizes whose float margin dominates the tail level, unless the null
is one the control arm's own bound rejects; see the limitations page); a row
persisted under an earlier construction of that test, or without naming its
construction, is refused when read rather than shown beside a verdict its
endpoints can contradict.

`prob_favorable()` reads the stored posterior only when it is available.
`p_value()` instead reads the prior-free sampling construction and remains
a sampling p-value whether or not a prior was declared. Both require
`inference == "fixed"`: a sequential row (`AlwaysValid`/
`AsymptoticMean`) refuses on both, since a confidence sequence has no
fixed endpoint distribution to summarize. `p_value()` additionally
handles a cluster-robust (`reference_kind="t"`) row directly against its
own t reference, while posterior access remains unavailable for unsupported
cluster-prior requests. `prob_favorable()` also needs `preferred_direction`
declared on the metric (`"increase"`/`"decrease"`/`"neutral"`) -- it is
not derived from `alternative`, which can point the opposite way on a
harm/futility test.

The posterior probability accessors (`chance_to_beat`, `prob_beyond`,
`prob_within`, and `risk_if_shipped`) never reconstruct a posterior from
sampling interval endpoints and never infer one from a p-value. They return
`None` when no supported posterior is stored.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

A prior and sequential inference (`AlwaysValid`/`AsymptoticMean`) cannot
compose: sequential coverage is a frequentist martingale guarantee. Declaring
both -- an `AnalysisPlan(inference=...)` combined with a `run(prior=...)`
override or a per-metric declared prior -- raises `sequential.route.unsupported`.
A prior also cannot compose with a declared cluster: clustered rows use a
joint reference built from cluster totals, and separating payloads does not
authorize a cluster prior. Combining `cluster=` with a prior raises
`arm.adjustment.cluster_prior`. In both cases the request is refused before
estimation.

The same override applies to priors declared on YAML `ExperimentMetric`
bindings. An informative prior changes the separately stored posterior, not
the prior-free sampling evidence used for an otherwise eligible secondary's
configured multiplicity family; method-compatibility checks still apply.
Overrides do not change the declaration: a later call that omits `prior`
uses the declared prior again for posterior inference. This also works after
reloading older moment exports with a declared prior. An explicit no-family
policy remains excluded.
per-metric-declaration precedence `decision_method`/`sensitivity_methods`
use (see [The data model](data-model.md)).

The same override applies to priors declared on YAML `ExperimentMetric`
bindings. Clearing a prior lets an otherwise eligible secondary re-enter its
configured multiplicity family; method-compatibility checks still apply.
Overrides do not change the declaration: a later call that omits `prior`
uses the declared prior again and excludes that posterior from the family.
This also works after reloading older moment exports with a declared prior.
An explicit no-family policy remains excluded.
