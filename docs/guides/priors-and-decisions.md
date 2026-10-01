# Priors and Bayesian decisions

`run(prior=...)` shifts a metric from a plain frequentist interval to
a Normal-conjugate (or discretized heavier-tailed) posterior. The
reported `value`/`lb`/`ub` become prior-informed posterior values; the
row's raw `log_mean`/`log_se` remain the unshrunk statistics underneath.
Each `LiftEstimate` exposes decision summaries from that row:
`stat_sig()`, `prob_favorable()`, and `p_value()`. They are not equally
universal: `stat_sig()` reads the row's own interval and works on any
row, while `prob_favorable()` and `p_value()` reconstruct the actual
posterior or sampling distribution and refuse on some row shapes (see
["`prob_favorable`, `p_value`, and `stat_sig`"](#prob_favorable-p_value-and-stat_sig)).

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
    print(
        f"prior_shrunk={r.prior_shrunk} lift={r.lift.value:+.2%} prob_favorable={r.prob_favorable():.3f}"
    )
```

```text
prior_shrunk=True lift=+0.16% prob_favorable=0.538
```

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
rows instead use exact binomial test inversion. Adding a prior therefore
changes the reported reference as well as the estimate; it does not add a
posterior to an otherwise unchanged sampling report.

For example, with 20 conversions among 60 control units and 40 among 60
treatment units, `Normal(mu=0, sigma=0.1)` gives posterior median lift
about 14.15% and `prob_favorable()` about 0.9294. `run(prior=None)` clears
the prior and restores the exact-binomial row with observed lift 100%.
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
horizon or sequential, Normal-referenced or cluster-robust.

`prob_favorable()` and `p_value()` instead reconstruct the actual
posterior or sampling distribution the row's interval was cut from, so
they need `inference == "fixed"`: a sequential row (`AlwaysValid`/
`AsymptoticMean`) refuses on both, since a confidence sequence has no
fixed endpoint distribution to summarize. `p_value()` additionally
handles a cluster-robust (`reference_kind="t"`) row directly against its
own t reference, so it still returns a number there; `prob_favorable()`'s
plain relative-null branch always needs the recovered Normal/mixture
posterior and so refuses on a cluster-robust row too -- the clustered
path bypasses the Normal-Normal tail that branch reconstructs. (An
absolute-margin `null_abs` row that is also cluster-robust refuses on
both `prob_favorable()` and `p_value()`, since neither reads a clustered
additive tail.) `prob_favorable()` also needs `preferred_direction`
declared on the metric (`"increase"`/`"decrease"`/`"neutral"`) -- it is
not derived from `alternative`, which can point the opposite way on a
harm/futility test.

Without an informative prior, `p_value()`'s fixed-horizon Normal-reference
branch is a Normal-tail sampling approximation. The estimated log-delta
standard error does not make it an exact finite-sample p-value. On a
`prior_shrunk=True` row, `p_value()` is instead a property of the posterior
that produced the row, not a guarantee of frequentist calibration.
Do not use that posterior-tail helper as input to BH or e-BH.

## Mutual exclusions: prior vs sequential inference, prior vs cluster

A prior and sequential inference (`AlwaysValid`/`AsymptoticMean`) cannot
compose: sequential coverage is a frequentist martingale guarantee, and a
prior-shifted posterior center would void it. Declaring both -- an
`AnalysisPlan(inference=...)` combined with a `run(prior=...)` override
or a per-metric declared prior -- raises
`sequential.route.unsupported` ("registered runtime supports predictive
priors, not posterior effect priors"). A prior also cannot compose with a
declared cluster: clustered rows use a t reference built from cluster
totals, which bypasses the Normal-Normal conjugate update a prior needs,
so combining `cluster=` with a prior raises
`arm.adjustment.cluster_prior`. In both cases the raw statistics
become the inference inputs directly, with no prior layered on top. See
[Sequential inference](sequential-inference.md) for the always-valid/
asymptotic-mean side of this exclusion.

## Per-metric priors

`MetricSpec(prior=...)` declares one metric's own prior on the dataframe
path. It applies only when the call itself leaves `prior` at its default
`UNSET`: an *explicit* `run(prior=...)` call -- including `run(prior=None)`
-- overrides every metric's prior for that call, including metrics that
declared their own `MetricSpec.prior`, the same call-wide-overrides-
per-metric-declaration precedence `decision_method`/`sensitivity_methods`
use (see [The data model](data-model.md)).

The same override applies to priors declared on YAML `ExperimentMetric`
bindings. Clearing a prior lets an otherwise eligible secondary re-enter its
configured multiplicity family; method-compatibility checks still apply.
Overrides do not change the declaration: a later call that omits `prior`
uses the declared prior again and excludes that posterior from the family.
This also works after reloading older moment exports with a declared prior.
An explicit no-family policy remains excluded.
