# Causal inference in Python: IPTW, DML, and AIPW

Use observational analysis to compare self-selected, phased, or opt-in
treatment. You need each unit's treatment, outcome, and pre-treatment
covariates that explain both treatment choice and outcome.

The causal assumption is **conditional ignorability**: after accounting for
those covariates, treatment is as-good-as-random. You must justify it;
overlap and balance checks cannot establish it.

## Declaring an adjustment set

For a mean or conversion outcome, declare the confounders in `AdjustmentSet`.
The default inverse propensity of treatment weighting (IPTW) estimates the
average treatment effect over the eligible population, reported as a relative
lift against its control mean. It relies on a correctly specified treatment
probability model.

This complete example simulates opt-in treatment and revenue:

```python
import numpy as np
import polars as pl
import increment

rng = np.random.default_rng(7)
n = 4_000
tenure_days = rng.gamma(shape=4.0, scale=90.0, size=n)
plan_tier_rank = rng.integers(0, 3, size=n).astype(float)
logit = -1.2 + 0.004 * (tenure_days - 360) + 0.5 * (plan_tier_rank - 1)
treated = rng.random(n) < 1 / (1 + np.exp(-logit))
base = 20 + 0.02 * tenure_days + 4.0 * plan_tier_rank
revenue = base * np.where(treated, 1.05, 1.0) + rng.normal(0, 4, size=n)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": np.where(treated, "opted_in", "control"),
        "revenue": revenue,
        "tenure_days": tenure_days,
        "plan_tier_rank": plan_tier_rank,
    }
)
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=increment.Observational(
        control_group="control",
        adjustment=increment.AdjustmentSet(covariates=("tenure_days", "plan_tier_rank")),
    ),
)
results = increment.run(experiment, reader).readout()
for row in results:
    print(
        f"{row.metric} / {row.group_id} [{row.method}]: "
        f"lift={row.lift.value:+.2%} "
        f"CI=({row.lift.lb:+.2%}, {row.lift.ub:+.2%})"
    )
```

<!-- output -->
```text
revenue / opted_in [iptw]: lift=+5.67% CI=(+4.48%, +6.87%)
```

The adjusted estimate is close to the simulated +5% effect; its interval
excludes zero. This is a large-sample working approximation, not a small-sample
guarantee or proof that adjustment removes confounding in your data.
Here the rank has a deliberately linear effect. For real categorical covariates,
pass the original strings, not arbitrary integer codes.

Each fit learns its categorical encoding. Unseen levels map to that fit's
reference level and produce an advisory, not automatic trimming or refusal.
See [encoding and advisory details](../reference/causal-methods.md#covariate-encoding).

## The identification gate

`IdentificationGate` checks whether treatment groups have comparable units.
By default, a fitted propensity outside `[0.01, 0.99]` raises
`IdentificationError`; inspect the reported arm counts and propensity deciles.

You can explicitly choose
`IdentificationGate(overlap="trim", min_propensity=0.05)`, but trimming changes
the population being estimated. `row.population` names the retained set.
That set depends on fitted probabilities; its interval conditions on the
retained units and omits uncertainty in the estimated trim boundary.
Do not describe it as the effect on everyone.

[Overlap rules](../reference/causal-methods.md#overlap-and-retained-population)
explain the shared mask used with multiple treatments.

### Weight concentration and effective sample size

Inspect IPTW/AIPW weight concentration alongside the estimate. Effective
sample size (ESS) describes how many units—or independent clusters—carry the
weights. It is not a causal check, an interval correction, or a choice of
degrees of freedom. See [diagnostic definitions](../reference/causal-methods.md#weight-concentration-and-effective-sample-size).

### Balance checks

The gate compares weighted covariate means using standardized mean differences
(SMDs). By default, `|SMD| > 0.1` warns; `gate.max_smd` makes your threshold
a hard refusal. Balance only checks the columns you supplied. A misspecified
model can balance those columns while missing important transformations or
interactions; passing is not evidence that no confounder was omitted.
See [balance calculations and measured failure cases](../reference/causal-methods.md#balance-checks).

### Missing covariate values

`AdjustmentSet(missing="refuse")` is the default. Do not fill a confounder with
`impute.pooled_mean` alone: without a missingness indicator, it can hide
residual confounding.

- `impute-indicator` keeps units, fills numeric/categorical values using pooled
  means/modes, and adds missingness indicators to fitting and balance checks.
- `pattern` fits separately within missingness patterns; IPTW only.
- `allow` needs explicit NaN-native learners capable of interactions. Set
  `gate.max_smd`: finite predictions and marginal balance do not establish
  that the learner can model missingness.
- `complete-case` drops incomplete units and labels the changed population.
  It can be biased when missingness is informative.

For `pattern` and `allow`, you must justify ignorability given observed values
and the missingness pattern: an unobserved value must not still confound within
that pattern. Read the [missingness requirements](../reference/causal-methods.md#missing-covariate-values)
before choosing either.

## Near-zero baselines: reporting the additive effect

Near a zero control mean, a relative effect can have an unbounded or disconnected
confidence set, or no finite point. That does not erase the additive effect.
Request metric units explicitly with
`value_scale={"revenue": "absolute"}` on `run`; scale never changes automatically.

Ordinary relative rows also retain `abs_diff`, `abs_se`, `abs_lb`, and `abs_ub`.
Read `relative_confidence_set` and `relative_unavailable_reason`, not just two
displayed endpoints. An unavailable interval is not a zero-width result.
See the [complete additive example and interval rules](../reference/causal-methods.md#relative-and-additive-results).
Supported priors affect separate posterior fields, never the sampling result;
the [prior restrictions](../reference/causal-methods.md#relative-and-additive-results)
still apply.

## Choosing `iptw` vs `dml` vs `aipw`

Set `decision_method=increment.Method(name="aipw")` or `"dml"` on `run`.

- **IPTW** fits treatment probabilities and compares weighted group means.
  It targets the unit-weighted average treatment effect (ATE).
- **AIPW** adds an outcome model per arm and cross-fits: each prediction uses
  training data that exclude that unit. It targets the same ATE. Its point is
  consistent if either model is right, but its interval additionally requires
  both model errors to shrink fast enough. One correct model alone does not
  guarantee calibrated uncertainty.
- **DML** estimates a partially linear slope, weighted toward units with
  more treatment-probability variation. Under heterogeneous effects this is
  not generally the ATE. It is insensitive to small model errors, not doubly
  robust: a correct outcome model cannot rescue a badly wrong propensity model.

Prefer cross-fitted AIPW/DML for flexible learners. IPTW fits and evaluates
on the same units; memorizing treatment can pull its answer toward the naive
comparison without triggering either gate. Explicit `Method(name="unadjusted")`
is allowed but labelled as confounded.

[Method equations and learner configuration](../reference/causal-methods.md#methods-and-learners)
cover defaults, factories, folds, and relative DML's denominator.

## Multiple treatment arms

IPTW/AIPW compare each treatment with one shared control mean over the same
eligible population, not separate treatment/control subsets. Every arm needs
support and conditional ignorability over that population. DML retains a
comparison-specific weighted slope.
[Multiple-arm definitions](../reference/causal-methods.md#multiple-treatment-arms)
explain the weights and what the population label means.

## Multiplicity across roles

Primaries split alpha across treatments; secondaries use their declared family;
guardrails must all clear. Filtering rows does not change the tested family.
See [role and snapshot rules](../reference/causal-methods.md#multiplicity-and-consistent-data).

## Clustered rollouts

Declare `cluster="geo_id"` when units share geo-level shocks. Adjusted intervals
use an asymptotic Normal reference over independent clusters, not a small-sample
guarantee. Fewer than 40 clusters in a comparison warns; a large cohort-wide
`n_clusters` can conceal much smaller support for one comparison.
Informative priors and posterior decision probabilities remain unsupported here.
See the [complete clustered example and covariance rules](../reference/causal-methods.md#clustered-rollouts).

## From definitions (the warehouse path)

Declare numeric or categorical covariates as `pre_exposure` or `static`
properties. Definitions and artifacts use `missing="refuse"`; other missingness
or gate settings require a full design on a dataframe summary/panel.
See [warehouse covariate declarations](../reference/causal-methods.md#warehouse-covariates).

## How strong would an omitted confounder need to be?

For a fixed-horizon IPTW/AIPW conversion risk ratio, `confounding_sensitivity`
reports how strong omitted confounding would need to be to cross a chosen
threshold. It is a conservative marginal bound, not proof of identification
or protection against selection/measurement bias. It inherits the row's
interval qualifications.
See the [complete sensitivity example and supported rows](../reference/causal-methods.md#omitted-confounder-sensitivity).

## Current limits

Use per-unit outcomes and covariates. `from_moments` cannot reconstruct them.
Dataframe panels support unwindowed mean/conversion outcomes with covariates
constant within each unit; prepare windowed values upstream as summary rows.
Adjusted ratio and quantile effects, observational CUPED, and relative margins
are unsupported. Absolute margins use the additive fields on relative rows;
a shifted null on an absolute-scale row is also unsupported.
SRM is not applicable without a randomized allocation.

See [exact input limits, alternatives and refusal order](../reference/causal-methods.md#supported-inputs-and-refusal-order)
before combining methods or metric types.

## Next step

Use [Reading results](reading-results.md) to interpret result availability, or
[Encouragement designs](encouragement.md) when randomized assignment changes
uptake without forcing treatment.

## References and assumptions

Conditional ignorability, consistency and positivity are required for the
causal interpretation. DML follows Chernozhukov et al.,
[“Double/Debiased Machine Learning for Treatment and Causal Parameters”](https://doi.org/10.1111/ectj.12097).
Diagnostics cannot validate these assumptions.
