# Causal methods reference

Use this page for estimator definitions, input contracts, and saved-result details.
Start with the practical guides for [observational comparisons](../guides/observational.md),
[encouragement designs](../guides/encouragement.md), or
[logged-policy evaluation](../guides/logged-policy.md).

## Observational adjustment

### Covariate encoding

Covariates may be numeric (`int`, `float`, `bool`) or categorical strings.
Declare the original categorical column, not integer ranks or hand-built
dummies. Each nuisance fit learns its own modal reference level and one
indicator per other training level; a level unseen in that fit maps to
the reference. Numeric columns keep their existing meaning. Encoding
does not establish overlap or conditional ignorability.

A level absent from a fit's training rows is disclosed in each adjusted
result's `note` and in `estimation.adjust_common.unseen_level_advisory`.
Its structured context lists each covariate and level, the distinct rows
scored as the reference, and the number of affected fits. This advisory
does not itself trim rows or refuse the analysis; it identifies predictions
without training support for that level.

### Overlap and retained population

For a binary comparison, `IdentificationGate(overlap="refuse")` rejects any
fitted propensity outside `[min_propensity, 1 - min_propensity]` (default
`min_propensity=0.01`). `IdentificationError` reports out-of-window counts
per arm and propensity deciles. `overlap="trim"` instead drops those units.

`LiftEstimate.population` is `None` without a population restriction; a
trimmed result names its overlap window and retained count. The retained
population is defined by fitted probabilities and is sample-dependent.
The influence-function (IF) interval conditions on it, omitting trim-boundary
uncertainty; untrimmed conservative-direction claims do not transfer.

With M arms, one mask retains a unit only if **every marginal arm propensity**,
including control, is at least `min_propensity`. All comparisons then report
`overlap with every marginal arm propensity >= g (kept of total units)`.
The gate refuses when M × `min_propensity` > 1 (probabilities sum to one),
or when trimming empties any arm, naming that arm.

For a binary trim, the population label reads
`overlap e in [g, 1-g] (kept of total units)`. On the
[guide's seeded data](../guides/observational.md#declaring-an-adjustment-set),
`min_propensity=0.05` retained 3,968 of 4,000 units and gave +5.46% lift.
Those numbers describe that fitted subpopulation only.

### Weight concentration and effective sample size

IPTW/AIPW use their actual inverse marginal propensity weights after the shared
overlap decision. For arm weights $w_i$:

- Kish ESS: $(\sum_i w_i)^2 / \sum_i w_i^2$.
- Maximum share: $\max_i(w_i)/\sum_i w_i$.
- Positive-weight count: the number of contributing units.

ESS and maximum share are invariant to self-normalization. The accompanying
definition identifies the weights and retained population; no clipping is added.
With `cluster=`, compute them over each cluster's total retained arm weight;
otherwise the grain is unit. Each treatment and the shared control have one
summary over the same retained cohort.

These describe concentration, not score support, identification, interval
correction or degrees of freedom. Unwindowed mean/conversion inputs from
`from_frame` summaries/panels, `from_definitions` and `from_unit_day_artifact`
have matched-input row/interval parity checks. Moments lack per-unit
propensities/covariates; switchbacks use scheduled contrasts instead.
Other methods return null diagnostics with an explicit not-applicable reason.

### Balance checks

SMD is the weighted covariate mean difference divided by the **unweighted**
pooled SD (Austin & Stuart 2015). Thus inflating weights cannot inflate the
denominator to pass the gate. `|SMD| > 0.1` gives a `UserWarning` by default;
exceeding an explicit `gate.max_smd` raises `IdentificationError`.

The default linear propensity model approximately balances its input columns
by construction, even when misspecified. In one simulation, integer-coding a
three-level category produced bias +0.59 on a true lift 0.10, with no SMD warning
in 60 replications; dummy encoding recovered the effect in that setup.
This is a measured failure case, not a universal calibration result. Use
categorical strings and consider checking transformations: marginal balance
does not establish ignorability.

In the guide's opt-in simulation, the naive comparison gave +19.1% against a
generating effect of +5%: tenure and tier raised both uptake and spending.
This single example illustrates confounding; it does not certify adjustment
for arbitrary data.

### Missing covariate values

Missing-value handling runs before learners and gates: NaN comparisons could
otherwise bypass thresholds. Choose `AdjustmentSet(missing=...)`:

- **`refuse` (default):** `IdentificationError` names incomplete columns and
  counts, with data-keeping alternatives. `impute.pooled_mean` alone is not a
  confounder repair; it is intended for randomized pre-period covariates such
  as CUPED baselines.
- **`impute-indicator`:** fill numeric values with pooled means and categories
  with pooled modes across all analysed arms jointly, never per arm. Add each
  incomplete column's missingness indicator to propensity fitting and SMDs;
  record the repair in the result `note`.
- **`pattern` (IPTW only):** fit `e(X_observed, pattern)` separately within
  missingness patterns. It does not assume a particular missing-data mechanism,
  but requires ignorability given observed values and pattern: unobserved
  values must not still confound within a pattern. More than 20 patterns or
  fewer than 30 units in a pattern refuses, recommending `impute-indicator`
  or coarser upstream patterns. With multiple treatments, fit each
  treatment/control comparison per pattern, predict every unit in that pattern,
  and count that comparison's own units toward 30. DML/AIPW lack per-pattern
  outcome fits and refuse, pointing to `impute-indicator` or `allow`.
- **`allow`:** pass NaN to explicit NaN-native learners without appending input
  missingness indicators. It asserts the same identification condition as
  `pattern`, without enumerating patterns. Set `gate.max_smd` and use an
  interaction-capable learner; finite output is not informative output.
- **`complete-case`:** retain fully observed units and relabel
  `LiftEstimate.population`. Informative missingness can bias it, so refusal
  messages do not recommend this option.

#### NaN-native inputs and diagnostics

Under `allow`, numeric columns stay unchanged. Categorical columns become
fit-specific level indicators, with NaN across the block for a null level.
A one-level fit keeps an otherwise-zero column carrying those NaNs instead
of dropping an empty indicator block.

Gates/SMDs use pooled mean/mode imputation plus indicators, adding
`{c}__missing*{b}` SMDs for every indicator crossed with every **other**
imputed covariate. Self-crosses are excluded because they duplicate the
indicator. These diagnostic inputs are distinct from learner inputs.

Every learner consuming X must be explicit: `propensity_learner` for IPTW,
both factories for DML/AIPW. Any defaulted learner refuses unconditionally;
the defaults can return non-finite predictions rather than raising on NaN.
When NaN is present, a small NaN-planted fit/predict probe rejects exceptions
or non-finite output; every fitted nuisance also gets a finiteness check.
Learner exceptions become `IdentificationError` naming learner, fold and arm.
Only NaN passes through: ±inf refuses by column name before probing.
The result `note` names incomplete columns and missing counts.

#### NaN tolerance is not capacity

Pattern-specific confounding needs missingness-indicator × covariate structure.
Depth-≥2 trees with learned NaN-routing directions can represent it; additive
stumps/GAM-like models and CART-style surrogate routing do not provide that
capacity merely by accepting NaN.

In the studied data-generating process, an additive NaN-native learner passed
all refusal layers and marginal balance checks while producing bias +0.4.
Interaction SMDs flagged it in 30/30 negative-control replications. They remain
advisory unless `gate.max_smd` is set; setting it is a hard balance check,
not a guarantee of adequate modelling. The 0.1 interaction warning fired in
about 7% of clean n=4,000 runs even with oracle propensities, reflecting the
higher kurtosis of interaction columns.

Use `pattern` for a few well-populated structural patterns with the parametric
default; use `allow` for scattered/high-dimensional missingness with suitable
learners. Rare patterns need no separate fit under `allow`, but behavior at an
all-NaN training column remains learner-dependent; a raising learner produces
a named refusal.

### Relative and additive results

IPTW, AIPW, and DML sampling results retain the joint covariance of the additive
contrast `tau` (for DML, its slope `theta`) and the control mean `mu0`. Their
relative result is a Fieller set in `relative_confidence_set`, not a
symmetric interval for `tau / mu0`.
Near zero controls can produce disconnected, one-sided, or full-real sets;
at exactly zero control mean the set can remain available without a finite
point (`lift=None`). Negative control means do not trigger a log-scale refusal.

The additive fields remain available independently. If the joint covariance
is genuinely indefinite or cannot be represented, `relative_unavailable_reason`
explains the missing relative inference; it does not discard the additive
effect. An exactly zero estimated variance for the relative contrast at its
observed point likewise withholds its bounds and decisions, with reason
`zero_relative_variance`, rather than certifying a zero-width effect.
These are working Normal approximations, not small-sample guarantees.
An explicitly informative IID Normal prior adds a separate scalar posterior and retains its
near-zero denominator guard: it refuses `abs(mu0) <= 4 * SE(mu0)`. A declared
cluster excludes that prior path.

#### How directional intervals and additive fields are constructed
Directional joint sets use the full `alpha` tail; two-sided sets use `alpha/2`.
That closure shapes the set the decision reads -- `stat_sig` and the p-value
test the null against it -- while the row *displays* the pre-closure central
interval, both endpoints finite, at the doubled `alpha_eff` that every other
inference path in the library reports a one-sided read at. A `None` endpoint
on a displayed relative interval therefore means the inversion could not bound
that side, never that a computable bound was dropped; `Estimate.open_side` and
the set's own `geometry` record which. Only FCR-selected re-estimation opens
the far endpoint, and it says so with `open_side`.
Directional p-values use one reference tail over the composite null, retaining
non-rejection when disconnected geometry extends into that null. The additive
point and SE are canonical projections of the persisted joint reference.
Additive bounds are reconstructed using their own persisted reference and the
same alpha and alternative, with exact binary64 equality on serialization.
Zero additive variance carries `abs_se=None` and `(abs_lb, abs_ub)=(None, None)`;
it does not fabricate a zero-width additive interval.

The additive effect can still be estimated near a zero baseline when the
causal assumptions hold. This complete example requests metric units explicitly:

```python
import numpy as np
import polars as pl
import increment

rng2 = np.random.default_rng(21)
m = 800
x = rng2.normal(0.0, 1.0, m)
opted = rng2.random(m) < 1 / (1 + np.exp(-0.5 * x))
net_delta = 0.05 * x + 0.02 * opted + rng2.normal(0, 1, m)
experiment, reader = increment.from_frame(
    pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(m)],
            "variant": np.where(opted, "opted_in", "control"),
            "net_delta": net_delta,
            "x": x,
        }
    ),
    shape="summary",
    unit="user_id",
    group="variant",
    metrics={"net_delta": "mean"},
    design=increment.Observational(
        control_group="control",
        adjustment=increment.AdjustmentSet(covariates=("x",)),
    ),
)
results = increment.run(experiment, reader, value_scale={"net_delta": "absolute"}).readout()
assert all(row.value_scale == "absolute" for row in results)
```

The mapping keys METRICS, not methods, so one call serves a mixed source:
only the named metric switches, every sibling metric keeps its relative
row, and a metric's rows stay on one scale across every requested method.
Through `run()`, an unknown metric name is refused at entry
(`readout.value_scale.unknown_metric`), and an absolute scale requested on a
quantile or ratio metric is refused at entry (`readout.value_scale.invalid`)
rather than silently dropped. The lower-level `estimate_ate()` path applies its
own check, refusing such a request with `estimation.adjust.value_scale_names`.

On an absolute row:

- `lift.value` and its interval are in the metric's own units, and
  `value_scale == "absolute"` marks them as such;

- `abs_diff`/`abs_se` stay `None` — they would only re-represent `lift`;
- `prior=None` requests no informative update; it does not turn sampling
  inference into a posterior. An explicit supported Normal prior is read in
  the metric's own units and affects only the separate posterior fields, not
  the sampling estimate, interval, p-value or discovery decision. One global
  prior across mixed relative/absolute scales refuses; it cannot mean both
  percentages and dollars. Even a wide `Normal(0, 1e6)` is not scale-free:
  a conjugate posterior with likelihood SE `1e5` shrinks its mean by about 1%,
  and one with SE `1e6` by 50%.

The reporting scale is never inferred from the data. Auto-switching a row
to absolute when the guard fires would make the units of a metric column
depend on sampling noise — in an as-of series, early dates could report
dollars and later dates percentages under one heading.

**Every relative row also carries the additive pair.** `abs_diff`,
`abs_se`, `abs_lb`, and `abs_ub` are populated on ordinary relative
observational rows too, computed from the same influence functions — which
is what makes a declared absolute margin (`Metric.margin_abs`/
`ExperimentMetric.margin_abs`, absolute-unit non-inferiority) work on this
path. For correctly specified, untrimmed parametric IPTW on this generic
plug-in path, the SE ignores propensity estimation and is deliberately
conservative (Lunceford & Davidian 2004): under strongly prognostic covariates,
measured intervals run ~1.5–1.8× wider than the true sampling spread (99.5%
observed coverage at a nominal 95%), so coverage is above nominal and power is
below what a refit-aware (e.g. bootstrap) interval would give. This claim does
not extend to the fitted pooled-logistic covariance correction, fitted overlap
trimming, or AIPW/DML cross-fitted orthogonal-score intervals. The untrimmed
default logistic path includes fitted-propensity estimating equations instead.
The measured AIPW/DML calibration was within a few percent in the studied
setups, not a guarantee for arbitrary learners or populations. The quoted
IPTW coverage and width ratios likewise describe that simulation only.

`RelativeLiftPrior` does not extend observational support; unwrapped
Student-t and mixture priors are also unsupported. See
[prior support rules](bayesian-results.md#supported-prior-routes) for the
separate posterior contracts.

### Methods and learners

IPTW uses self-normalized weighted arm means and relies on correct propensity
specification. AIPW adds one cross-fitted outcome model per arm and targets
the same unit-weighted ATE. AIPW's point is consistent if either nuisance model
is correct; its IF interval additionally needs the product of nuisance errors
to be `o(1/√N)`. One badly misspecified model can therefore leave a consistent
point with miscalibrated uncertainty.

DML partials out one pooled outcome model per treatment, not one per arm.
Its slope is first-order insensitive to small nuisance errors, not doubly
robust: propensity misspecification attenuates it by the propensity model's
squared error even with a correct pooled outcome model.

#### The DML estimand and relative-scale calculation
```text
theta = E[(D - e)(Y - l)] / E[(D - e)^2] = E[e(1-e)·tau(X)] / E[e(1-e)]
```

Here `e(X)` is propensity, `l(X)=E[Y|X]`, and `tau(X)` the conditional effect;
the result uses `estimand="plr_slope"`. This equals the unit-weighted ATE when
effects are homogeneous, propensity is constant, or `e(1-e)` is uncorrelated
with `tau(X)`. Otherwise it is a different target: one heterogeneous-effect
simulation separated them by 26 Monte Carlo standard errors. Choose DML for
that weighted slope, not as another estimate of the ATE.

A relative `dml` row reports `theta / mu0`, **not** `mu1 / mu0 - 1`. Its
denominator is the augmented control-arm mean of the whole analysed
population,

```text
mu0 = mean(m0(X)) + sum_i w_i (Y_i - m0(X_i)) / sum_i w_i,   w_i = 1{A_i = 0} / p0(X_i)
```

with `m0` a control-only outcome regression cross-fitted on the same folds
(fitted only when a relative row is requested, after the slope's own fits)
and `p0` the control propensity. Neither the pooled regression `l` nor a
constant-effect shift of it is a control mean under heterogeneous effects.
The denominator is consistent if either `m0` or `p0` is right; the slope is
not, and the joint Fieller set uses the covariance of both influences. The
absolute slope is identical whether or not the relative row is requested.

#### Default learners and custom learner configuration
All methods share the identification gate. Defaults need no extra installs:
ridge-stabilized logistic propensity regression and closed-form ridge linear
outcome regression. Custom models implement `Learner.fit(X, d)`/`predict(X)`;
pass zero-argument factories via `Method`'s `propensity_learner` and
`outcome_learner`, with `folds` for AIPW/DML.

Factory calls:

- IPTW: one propensity instance per treatment, no cross-fitting. Specifying
  an outcome model or folds refuses.
- AIPW: per populated fold, one propensity model per treatment and one
  outcome model per arm.
- DML: per populated fold, one propensity and one pooled outcome model per
  treatment; relative rows additionally fit a control-only outcome model
  after the slope fits.

When `missing="allow"` actually encounters NaN, IPTW reuses its fit instance
for the capability probe; AIPW/DML create one extra discarded probe instance
per factory. Finite inputs skip probing; infinities refuse before it.

#### Flexible IPTW learners can overfit

Memorizing treatment makes fitted propensities approach treatment indicators
and weights approach 1, pulling IPTW toward the naive contrast. The gate detects
saturation, not overfitting inside its bounds. No runtime flag diagnoses this
from fitted probabilities alone; prefer cross-fitted AIPW/DML for flexible models.

#### Cross-fitting and cluster-fold details

AIPW/DML deterministically balance arms across folds. Clusters stay intact:
no member of a validation cluster enters training. Stable content hashes of
cluster labels make folds row-order invariant; singleton clusters rank unit
IDs instead to match unclustered folds.

Every arm needs at least `folds` distinct clusters, counting pure and shared
clusters. Observational dependence clusters may span arms; only randomized
and encouragement designs require arm purity. One all-arm-stratified fold
vector serves every model of a metric. Each training fold must contain
each treatment and control or refuse, naming the comparison.

Explicit `Method(name="unadjusted")` remains a labelled confounded comparison.

### Multiple treatment arms

With a control and treatments `1..J`, every comparison is evaluated over the
same eligible population: all units of the metric's cohort, whichever arm they
are in. For IPTW and AIPW each arm mean is
`mu_a = E[m_a(X)]` with `m_a(X) = E[Y | A = a, X]`, averaged over the covariates
of every cohort unit — so the rows share one control mean `mu_0`, and
`tau_a = mu_a - mu_0` and `tau_a / mu_0` are comparable across treatments.
Identification needs, for every arm, conditional ignorability of that arm
given the covariates and positive arm support over the whole population. A
comparison is not re-estimated on its own treatment/control rows, which would
average over a different, comparison-specific covariate mix.

#### How multiple-arm propensities are combined
Increment keeps the ordinary binary learners. For each treatment `a` it fits
the propensity learner on that treatment's and the control's rows only, giving
`q_a(X) = P(A = a | A in {0, a}, X)`, predicts it for every cohort unit, and
couples the conditional odds into marginal arm propensities:

```text
q_a / (1 - q_a) = p_a / p_0
p_0 = 1 / (1 + sum_a q_a / (1 - q_a)),   p_a = p_0 q_a / (1 - q_a)
```

With one treatment this is the usual `e` and `1 - e`. With the default
linear logistic learner, the pairwise fits are consistent for a multinomial
logit propensity, though they are not its joint maximum-likelihood fit. Other
arms' outcomes never train an arm's outcome model: other arms contribute their
covariates to the averaged population, not their outcomes. IPTW then weights
each arm by `1 / p_a`; AIPW fits one outcome model per arm and adds the
self-normalized residual correction. The untrimmed default logistic IPTW
covariance includes the estimating equations of every treatment-versus-control
model, whose scores share control units.

DML keeps each comparison's own partially-linear slope. On the comparison's
rows it solves the pooled partialling-out equation, which as a functional of
the whole population is

```text
theta_a = E[h_a(X) tau_a(X)] / E[h_a(X)],   h_a(X) = p_a(X) p_0(X) / (p_a(X) + p_0(X))
```

a comparison-specific weighted effect, not the population ATE (and not a
treatment-versus-everyone-else slope). Its score is zero outside the
comparison's rows, so it is reduced over that comparison alone; a relative row
divides it by the common augmented control mean. For example, with
`(mu_0, mu_1, mu_2) = (4, 7, 8)` over the population, AIPW and IPTW report
relative lifts `0.75` and `1.0`; a DML slope that is `1.8` for the first
treatment reports `0.45` against the same `mu_0 = 4`.

Balance diagnostics stay comparison-specific: SMDs compare each treatment's
own rows with the control's under the marginal arm weights, never relabelling
other treatments as control. They do not prove balance to the whole population
or the causal assumptions. `LiftEstimate.population` stays `None` unless
explicit trimming or complete-case analysis restricted the shared population,
in which case every row carries the same label; for DML, `None` still means a
weighted slope over the full population, not an unweighted ATE.

### Multiplicity and consistent data

A primary's alpha is Bonferroni-split across its treatment arms, exactly
like the randomized path. Secondaries join one Benjamini-Hochberg family at
`plan.q`: every secondary is estimated once at the nominal `plan.alpha`,
the family is selected at `q`, and selected cells are re-estimated at the
Benjamini-Yekutieli FCR level -- `LiftEstimate.discovery`, `.family_q`, and
`.family_threshold` are populated on every decision-role secondary row, not
left `None`. A guardrail keeps its full declared alpha and never joins a
family: it is an intersection-union test (ship only if every guardrail
clears), whose own family-wise error rate is already bounded by construction
regardless of how many guardrails are declared.

One readout reduces each metric's whole-window moments once. That evidence
supplies the arm gate, a primary's Bonferroni split, the family's size `m`, and
every estimate and FCR re-estimate, so every one of them sees the same arms. A warehouse-backed
source also pins its inputs for the readout (`readout_snapshot`), so the unit
frames IPTW, AIPW, DML and clustered unadjusted estimates read are the same
execution as those moments, and data that lands during the readout joins
none of it. A dataframe source is immutable; a custom provider with no snapshot
scope is trusted not to change during the call. The lower-level
`estimate_ate()` likewise reduces each metric's evidence through the declared
provider acquisition boundary, which validates the request before reading.
Custom sources backing either a builder's reader or the lower-level estimator facade
must declare `EvidenceCapabilities` or implement the typed provider requests;
see the [custom-provider example](https://github.com/kylejcaron/increment/blob/main/examples/custom_evidence_provider.py)
and the [migration map](../development/adding-an-estimator.md#custom-evidence-providers).

### Clustered rollouts

```python
import numpy as np
import polars as pl
import increment

geo_rng = np.random.default_rng(7)
k, m = 40, 25  # half the total geos, users per geo
geo_z = geo_rng.normal(0.0, 1.0, 2 * k)  # geo covariate drives the rollout
rolled_out = geo_rng.random(2 * k) < 1 / (1 + np.exp(-0.8 * geo_z))
geo_effect = geo_rng.normal(0.0, 2.0, 2 * k)  # shared shocks: ICC > 0

geo = np.repeat(np.arange(2 * k), m)
geo_df = pl.DataFrame(
    {
        "user_id": [f"gu{i}" for i in range(2 * k * m)],
        "variant": np.where(rolled_out[geo], "rollout", "control"),
        "revenue": 20 + 2.0 * geo_z[geo] + geo_effect[geo] + geo_rng.normal(0, 4, 2 * k * m),
        "geo_z": geo_z[geo],
        "geo_id": [f"geo{j}" for j in geo],
    }
)

experiment, reader = increment.from_frame(
    geo_df,
    shape="summary",
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    cluster="geo_id",  # the independent-cluster column
    design=increment.Observational(
        control_group="control",
        adjustment=increment.AdjustmentSet(covariates=("geo_z",)),
    ),
)
(geo_result,) = increment.run(experiment, reader).readout()
print(f"K={geo_result.n_clusters}, reference={geo_result.reference_kind}, dof={geo_result.dof}")
```

<!-- output -->
```text
K=80, reference=normal, dof=None
```

Adjusted clustered inference sums centered member influences within each cluster
and retains their complete covariance. Arm-pure clusters do not justify dropping
between-arm variation for a superpopulation target. The reference is asymptotic
Normal (`reference_kind="normal"`, `dof=None`); Bessel scaling does not correct
fitted-response leverage or establish small-K coverage.

#### Cluster support and covariance details

After trimming, each comparison needs at least two distinct clusters among
its treatment/control units; if all are arm-pure, at least two in each arm.
Other treatments' clusters do not satisfy these admission requirements.

The variance support depends on the influence, not just the requested arm:

- DML slopes and fixed-propensity IPTW influences (generic learners, pattern
  and trimmed fits) vanish outside the comparison. Use its `K_a` clusters
  and Bessel factor `c_a=K_a/(K_a-1)`. The fixed-propensity control mean also
  uses that comparison's factor. These IPTW and absolute DML rows report `K_a`.
- AIPW arm means, default untrimmed logistic IPTW's fitted-propensity correction,
  and relative DML's control mean include cohort-wide terms. Use all retained
  `K` clusters, factor `c=K/(K-1)`, and report `n_clusters=K`, including clusters
  containing only another treatment.

This support is structural. Even constant outcome predictions, whose influences
vanish outside the comparison, keep the cohort factor/count on AIPW and relative
DML rows. As population terms shrink, the comparison's own clusters can carry
most information; a large reported `n_clusters` does not guarantee reliable
inference for a small `K_a`.

Relative DML keeps each influence's own clustered variance: its additive SE
equals the absolute row's SE, and all comparisons share one control-mean
variance. The cross-covariance of complete cluster totals uses `sqrt(c_a*c)`,
rounded down to a multiple of `2**-64` to keep the matrix positive semidefinite.
When supports coincide, as in a binary cohort, the factor is exactly
`K/(K-1)`. Small-K calibration of mixed support is unmeasured.

Untrimmed pooled-logistic IPTW includes every treatment/control model's
estimating equations and covariance with arm means. Generic, pattern-specific
and trimmed fits omit that correction. AIPW/DML need cluster-level nuisance
rates; cross-fitted linear residuals have no finite-sample correction.
Result notes retain these limits; target weighting is unchanged.

Fewer than 40 clusters in the comparison produces a `RuntimeWarning` about
over-rejection, but structurally valid comparisons still run. Under clustering:

- `chance_to_beat()`, `prob_beyond()` and `prob_favorable()` refuse, including
  the absolute-margin branch. Sampling intervals do not establish a posterior.
  Informative priors also refuse.
- IPTW/AIPW/DML ratio metrics remain unsupported.
- Encouragement ITT and additive LATE have component-based Welch references
  and the same small-K admission policy. Clustered encouragement CUPED and
  complier-relative LATE still refuse; see
  [encouragement and CUPED](../guides/cuped.md#clustered-encouragement-designs).

#### Explicit unadjusted clustered comparisons

`Method(name="unadjusted")` jointly aggregates both arms from the retained unit
frame, counting shared clusters once and retaining signed cross-arm covariance
for the additive difference and relative Fieller set. It supports ratio metrics,
but remains a **confounded** observed comparison, weighted by members (or
declared denominator totals), not an equal-cluster or causal effect.

Disjoint arms use per-arm Bessel covariance, additive Welch degrees of freedom,
and relative t reference `min(K_T-1, K_C-1)`. Mixed arms correct the response to
estimating both means using each cluster's member/denominator share. Additive
variance is unbiased under independent clusters, fixed masses and common arm
expectations, allowing unequal variances and arbitrary within-cluster covariance.
Random ratio denominators still need a delta approximation before joint inversion.

The mixed-arm Normal reference is asymptotic; random-composition bias and
small-K calibration remain unresolved. Singular response geometry retains a
noted CR1 approximation. Negative corrected variance is unavailable, not clipped;
zero additive variance can leave null absolute uncertainty alongside available
relative inference. Each arm needs two clusters; below-40 warnings and scale
restrictions remain. Mixed-cluster unit-frame sources work, but dataframe input
still rejects spanning labels.

### Warehouse covariates

Declare the covariate as a `pre_exposure` or `static` `Property` on the
fact source that carries it. For example, a `users` source can declare
`name: tenure_days`, `column: tenure_days`, `dtype: float`, and
`as_of: pre_exposure` in its `properties` list. On the experiment, set
`design.mechanism: observational` and include
`{property: tenure_days, source: users}` in `design.covariates`.
The ordinary exposure, unit, control group, dates, metrics and plan declarations
are still required; see [warehouse setup](../guides/data-model.md).

`run(*from_definitions("definitions/", con, experiment_name="rollout")).readout()`
returns the same rows as
`run(*from_frame(..., shape="summary", design=Observational(...))).readout()` over the same
per-unit data. Each unit's covariate is its latest value strictly before its
first exposure (`pre_exposure`) or its latest value overall (`static`); a unit
with no such value gets a missing covariate. Numeric (`dtype: int`, `float`,
or `bool`) and categorical (`dtype: string`) properties are accepted; date
properties and `as_of: event_time` refuse when definitions load, naming the
property. `source:` is optional when exactly one fact source carries the
property for the experiment's unit, and required to disambiguate otherwise,
exactly like a breakout.

Definitions and artifact constructors retain `missing="refuse"`, including
for a null categorical level. YAML `design:` rejects `missing` and `gate`
with `definition.experiment.design_tuning_key_not_yaml`; these are not
warehouse configuration keys. Use a `from_frame` summary or panel
with a full `Observational` design for another missing-value policy.

A unit-day artifact automatically publishes every declared covariate:
numeric values use the unchanged `unit_covariate` relation, while strings
use `unit_covariate_level`, preserving nulls separately from literal labels.
Their typed requests are `UnitCovariateRequest` and
`UnitCovariateLevelRequest`; neither needs to be supplied to include a
declared covariate. `from_unit_day_artifact` reopens with the same
`Observational` design, and `run(*from_unit_day_artifact(...)).readout()`
returns the same rows. Undeclared or absent
extensions refuse with `artifact.extension.missing`.

`estimate_cate`, `validate_cate`, `targeting_rule`, and `select_targeting_rule`
read numeric or categorical properties passed to `interact=`/`adjust=` from
`from_definitions`, resolved the same way; no `design:` block is needed. From an
artifact they read only declared covariates, so `estimate_cate` (randomized
designs only) cannot read covariates from an artifact; use `from_definitions`.

### Omitted-confounder sensitivity

For a fixed-horizon IPTW/AIPW conversion RR, apply `confounding_sensitivity`. Eligibility
comes from the row's declared `metric_type`: `conversion` or `retention` rows are accepted,
other types are refused, and a row carrying no metric type gives `metric_type_unavailable`
(after the estimand check: an ITT or compliance row gets `estimand_unsupported` first).

```python
import numpy as np
import polars as pl
import increment
from increment.estimators import confounding_sensitivity

conversion_rng = np.random.default_rng(20261013)
conversion_n = 4_000
segments = conversion_rng.integers(0, 2, size=conversion_n)
converted_treatment = conversion_rng.random(conversion_n) < np.where(segments == 0, 0.3, 0.7)
risk = np.array([[0.08, 0.20], [0.30, 0.45]])[converted_treatment.astype(int), segments]
conversion_units = pl.DataFrame(
    {
        "unit": np.arange(conversion_n),
        "arm": np.where(converted_treatment, "treatment", "control"),
        "segment": segments,
        "converted": (conversion_rng.random(conversion_n) < risk).astype(int),
    }
)
experiment, reader = increment.from_frame(
    conversion_units,
    shape="summary",
    unit="unit",
    group="arm",
    metrics=[increment.MetricSpec(name="converted", type="conversion")],
    design=increment.Observational(
        control_group="control", adjustment=increment.AdjustmentSet(covariates=("segment",))
    ),
)
with reader:
    conversion_rows = increment.run(
        experiment,
        reader,
        decision_method=increment.Method(name="iptw"),
        sensitivity_methods=[increment.Method(name="aipw")],
    ).readout()
    for conversion_row in conversion_rows:
        sensitivity = confounding_sensitivity(conversion_row, threshold_risk_ratio=1.05)
        print(sensitivity.model_dump())
```

The threshold above is a 5% increase, not the no-effect RR of one. `bias_factor_point` is the
smallest B that could move the causal RR to that threshold; `e_value_point` is the equal
strength m with B(m, m) = m²/(2m − 1). Interval fields use the displayed threshold-side limit
at `interval_level`, even if the directional test set is `all_real`; one means a crossing, a
point at the threshold with any displayed limit, or `all_real` without that limit.
Unavailable/empty sets or other missing limits give null and `interval_limit_unavailable`.

This is a **conservative marginal** bound: RR_true is between RR_obs/B² and
RR_obs·B², not the usual conditional E-value applied directly to RR_obs.
Heterogeneous covariate strata can have true marginal RR one but standardized
observed RR about 2.446 with actual strengths three: the naive conditional
inversion exceeds four, whereas this marginal tipping strength is below three.
The construction aggregates the per-arm bounds of Ding & VanderWeele (2016),
*Epidemiology* 27:368–377, eAppendix Prop. A.8 and §5.2; the equal-strength
inversion follows VanderWeele & Ding (2017), *Ann Intern Med* 167:268–274.

Assume consistency, positivity and appropriate nuisance-model correctness in
the reported (possibly trimmed) population, and an omitted U such that (C, U)
suffices for exchangeability. RR_EU is the maximum confounder prevalence ratio
over strata and **every pair of cohort arms, in both directions**; RR_UD is the
maximum outcome-risk ratio over U within strata and arms. This addresses only
confounding, not selection or measurement bias. **Robustness is not identification**:
reaching the tipping strength means confounding could explain the effect, not
that it does. The deterministic bound inherits the row's asymptotic interval,
including its working-approximation qualifications, rather than adding coverage.

Conversion-row parity is exercised for `from_frame` summaries and panels,
`from_definitions` and `from_unit_day_artifact`. Observational adjusted reads from
`from_moments` refuse upstream with `evidence.covariate_unavailable`;
`from_switchback_frame` refuses with `source.frame.switchback.identification`.
The accepted estimands are `ate` and `overlap_subpopulation_ate`, on relative scale.

| Interaction | Disposition |
|---|---|
| CUPED | Not applicable: observational adjustment refuses CUPED. |
| Ratio metrics | Refused from the row's metric type (`outcome_unsupported`); observational adjustment also refuses ratio metrics upstream. |
| Clustering | Inherits the row's reported interval; public unit-summary IPTW/AIPW conversion rows with declared dependence clusters are exercised. |
| Sequential inference | Refused; only `inference="fixed"` is accepted. |
| Winsorization | Not applicable to the binary metric types. |
| Breakouts | `LiftEstimate` only; `BreakoutEstimate` is not its subclass and is refused. |
| Multiplicity roles | Uses the reported, possibly family-adjusted, interval, without recomputing it. |

DML's slope and continuous outcomes are not supported. A tighter estimator-scoped
bound is not supplied by this construction.

### Supported inputs and refusal order

#### Panels and moments

`from_frame` panels collapse unwindowed mean/conversion outcomes to per-unit
totals, as `moments(grain="total")` does. Covariates must be constant within
unit; otherwise `frame.frame_panel.unit_covariate_varies` names offending units.
Windowed/retention metrics refuse with `source.frame.unit_frame_panel`.
Prepare windowed values upstream as unwindowed summary metrics; no frame
source supplies unit-grain retention estimates.

`from_moments` raises `CapabilityError` with `evidence.covariate_unavailable`,
naming covariates: scalar moments lack unit-level weights and attached covariates.

#### Quantiles

No observational quantile estimator exists, under any inference kind.
Randomized order-statistic intervals would mislabel confounded effects;
adjusted-mean machinery would mislabel means as quantiles. Use a randomized
design for quantile effects.

For two-sided zero-null requests, `run()`, `estimate_ate` and
`estimate_quantile_lift` raise `UnsupportedRequestError`,
`readout.observational.quantile`, with `metric` context **before outcomes**.
This is identical on definitions, artifacts, and both dataframe shapes.
An absolute `margin_abs` first raises `readout.metric.quantile_alternative`
(one-sided tail); a relative margin raises `plan.observational.relative_margin`
at construction.

`export_moments()` on those sources raises the observational-quantile code
before even earlier catalog metrics are read. A quantile declared over scalar
moments can construct, but cannot turn them into genuine quantile data:
the same `run()`/margin precedence applies. Breakout views instead get
`evidence.breakout_unavailable`; day-axis views get `evidence.grain_unavailable`.
See [quantiles and portable moments](../guides/quantile-metrics.md#quantiles-and-portable-moments).
`workflows.planning_baseline` can still read control per-unit quantile data
because it estimates no effect.

#### Ratios

IPTW/AIPW/DML adjust one outcome, not a numerator/denominator pair with joint
covariance. Their `UnsupportedRequestError` uses
`estimation.adjust_common.supported_ratio_metric` with `method`, `metric`,
`role`, `family`, and `correction` context:

- All metrics ratios, all decision methods adjusted: `run()` refuses at entry.
  Sensitivity methods such as `unadjusted` do not prevent it.
- Any ratio in a `bh`/`e_bh` family: refusal at entry names the family, even with
  other supported metrics and regardless of informative priors.
- Other mixed catalogs: unsupported ratio/method pairings warn with
  `estimation.adjust.skip_unsupported_metric`; supported pairings still return.
  A skipped decision pairing records an unavailable hypothesis, not a result row.

Randomized [ratio CUPED](../guides/cuped.md#ratio-metrics) adjusts both components
jointly but is not confounding adjustment. Observational `unadjusted` supports
ratios only as confounded comparisons. Adjusting numerator and denominator as
separate means answers two different questions; their intervals cannot be
recombined as independent intervals for a ratio effect.

#### Assignment checks and margins

`srm()` returns `NotApplicable`: no randomized target allocation exists.
Absolute margins (`Metric.margin_abs`/`ExperimentMetric.margin_abs`) use the
additive interval on relative rows. Relative margins (`Metric.margin`/
`ExperimentMetric.margin`) refuse. Shifted nulls of either kind on
`value_scale="absolute"` rows also refuse; support for a metric-unit
`null_lift` is not supplied by this path.

The DML discussion follows Chernozhukov et al., [“Double/Debiased Machine Learning for Treatment and Causal Parameters”](https://doi.org/10.1111/ectj.12097). Conditional ignorability, consistency, and positivity identify the causal estimand; overlap and post-adjustment balance are diagnostics for support and model behavior, not validation of conditional ignorability.

## Encouragement designs

### Definitions and supported inputs

The experiment's `control_group`/`allocation` declare assignment.
`design` adds `mechanism: encouragement`, `uptake.fact`, optional
`uptake.window_days`, and `one_sided`. LATE also needs
`exclusion_restriction.justification`, a mechanism statement such as
“the button can only move deflection by being clicked.”

`uptake.fact` is validated like exposure/trigger: its source's `entities` must
include the experiment's unit. Statistical tuning (weak-instrument threshold,
overlap policy) is not YAML configuration; override it with a full design on
a dataframe summary/panel. Observational YAML uses the
[warehouse covariate rules](#warehouse-covariates).

Encouragement works on dataframe summaries/panels, definitions, artifacts and
moments. Artifact/moments round trips retain ITT, compliance, LATE, guardrails
and BH/e-BH-selected secondaries. Switchback inputs accept only randomized
identification, not encouragement or observational designs.

The default requests all three estimands. Without exclusion, a request including
LATE raises `identification.encouragement.exclusion_required` before outcomes,
even if its first stage would later fail. Omission is supported in YAML and
artifact/moments round trips for ITT/compliance, including supported sequential
monitoring. Exclusion does not enable sequential LATE. The exclusion object needs
only `justification`; artifacts containing the retired field refuse with a
re-materialization remedy.

### Estimands, selection and margins

LATE divides additive ITT by the compliance first stage. Relative LATE uses
the complier control mean; its `p_value()` reads its own log-scale reference
and agrees with its interval. It adds no decision evidence beyond additive LATE.

Secondary discovery and BH/e-BH interval correction are ITT-based. Selected
LATE rows are re-estimated at ITT's corrected level (same null), but only ITT
reports `discovery`. Fixed-horizon compliance has `discovery is None`;
sequential compliance enters selection only through explicit policy membership.

Unadjusted independent-unit binary ITT uses ordinary conversion inference
even with uptake moments: dense counts use delta-method `reference_kind="t"`,
sparse counts use `"binomial"`. `conversion_inference="finite_sample"` on the
unadjusted method selects exact binomial inference at every size. This includes
conversion breakouts from definitions, artifacts and both dataframe shapes;
summary sources can declare a breakout dimension. Zero control conversions can
remove the point while leaving a confidence set.

A guardrail returns the requested estimands like any other metric; `run` does
not add estimands. A margin (`Metric.margin`/`margin_abs` or plan-bound margin)
uses the randomized non-inferiority construction on fixed-horizon whole-window
ITT. Missing guardrail margins refuse with `plan.guardrail.margin_required`.
LATE has a plain interval, no margin verdict; compliance does not apply an
outcome margin. Refused combinations:

- LATE without ITT for a margined metric: `readout.encouragement.margin`.
- As-of margins: `readout.metric_declare_non`.
- Breakout margins: `readout.margin.breakout`.
- Sequential margins: `plan.encouragement.margin`.
- Design-level compliance-only `run(..., by=...)`:
  `readout.run.segment_unsupported`. Per-segment breakout-view compliance works.

The [notebook](../examples/encouragement.md)'s registered uptake sequence uses
**ratio coordinates**, not lift. The display shows bounds, confidence level
and geometry; `sequential_result.bounds` retains exact rational endpoints
and endpoint certificates.

### Outcome cohorts and first-stage failures

Design-level compliance uses enrolled units and `uptake.window_days`, independently
of missing outcomes, metric order or count. Fixed-horizon compliance-only calls
skip outcome checks/reductions, including for a declared triggered population;
metric names and source/design state are still validated. Definitions, artifacts,
dataframe summaries and moments agree even when all treatment outcomes are missing.

Each outcome's LATE uses its own outcome cohort. Weak first stages retain a
planned row with `lift=None` and
`failure_code="estimation.encouragement.late.weak_first_stage"`.
A z at or below minus the gate gives
`estimation.encouragement.late.negative_first_stage` instead.
Context contains `first_stage_estimate`, `first_stage_se`, `first_stage_z`,
`min_first_stage_z`, `metric`, `group_id` and `method`.
With no sampling variance (e.g. everyone in control takes up, nobody encouraged
does), z is null and `first_stage_z_reason="first_stage_without_sampling_variance"`.
Every requested method's LATE row carries the same cohort failure; it adds no
compliance row. Other cohorts can still pass, and design-level compliance can
have a different weak/strong classification.

### Uptake-only and sequential requests

All five encouragement constructors support no outcome catalog: only assignment
and uptake are needed. Pass `metrics=[]` to dataframe/portable constructors and
request `estimands=("compliance",)`. Fixed-horizon calls need no sequential policy;
exports retain design-level state without inventing outcome rows.

Panels still refuse `missing="skip"` with `frame.missing_policy.panel_skip`.
Do not substitute zeros for undefined outcomes; mixed ITT/LATE requests need them.

Uptake-only monitoring needs `InferenceSpec(kind="always_valid")`, an explicit
`SequentialCompliancePolicy`, and uptake registration, not a scalar outcome law.
An empty catalog alone is insufficient: ITT has no outcome rows to scope and
the default includes unsupported LATE. Match policy alpha, direction, null and
family membership to registration. Registered ITT and relative Bernoulli
compliance also require joint finalization and immutable-prefix proofs.
Compliance is excluded from selection by default. Sequential as-of reads need
`completed_windows_only=True`; see [sequential inference](../guides/sequential-inference.md).

A nonempty catalog may retain unbounded retention only with **explicit uptake-only**
registration: automatic registration or outcome-consuming requests raise
`sequential.route.unsupported`. Supported sources are definitions, artifacts
(metric declared on the experiment), and moments (metric in `metrics=`), not
dataframe constructors. This is distinct from the empty-catalog case.
Moments replay the retained checkpoint. Registered dataframe panels retain one
capture identity across a combined whole-window/completed-as-of uptake-only run,
even without outcome metrics.

### Daily warehouse example

Daily uptake is a first-stage rate; per-day LATE is unsupported. This example
creates its own definitions and DuckDB data and asks only for compliance.
It demonstrates the request shape, not precision from two units.

```python
import datetime as dt
from pathlib import Path

import ibis
import pyarrow as pa

import increment
from increment.artifacts import SourceSnapshotEvidence

definitions = Path("encouragement.yml")
definitions.write_text(
    """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
      - name: clicked
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    window_days: 3
    aggregation: sum
experiments:
  - name: encouragement_example
    exposure: assignment
    trigger: saw_surface
    unit: user_id
    start: 2026-01-01T00:00:00
    end: 2026-01-03
    observation_end: 2026-01-04
    control_group: control
    plan:
      secondaries: [revenue]
    design:
      mechanism: encouragement
      uptake:
        fact: clicked
    """
)
con = ibis.duckdb.connect()
con.create_table(
    "events",
    obj=pa.table(
        {
            "experiment_id": ["encouragement_example"] * 8,
            "user_id": ["c1", "t1", "c1", "t1", "c1", "t1", "c1", "t1"],
            "group_id": ["control", "encouraged"] * 4,
            "ts": [
                dt.datetime(2026, 1, 1),
                dt.datetime(2026, 1, 1),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 2),
                dt.datetime(2026, 1, 3),
                dt.datetime(2026, 1, 3),
            ],
            "event": [
                "enrolled",
                "enrolled",
                "saw_surface",
                "saw_surface",
                "clicked",
                "clicked",
                "revenue",
                "revenue",
            ],
            "value": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
        }
    ),
)
experiment, reader = increment.from_definitions(
    definitions,
    con,
    experiment_name="encouragement_example",
    source_snapshot_evidence=SourceSnapshotEvidence(
        dt.datetime(2026, 1, 4, tzinfo=dt.UTC),
        {"events": dt.datetime(2026, 1, 4, tzinfo=dt.UTC)},
    ),
)
triggered_daily = increment.View.daily(population="triggered")
with reader:
    increment.run(
        experiment.with_views(triggered_daily),
        reader,
        views=(triggered_daily,),
        estimands=("compliance",),
    ).readout(triggered_daily)
con.disconnect()
```

Triggered sequential compliance is unavailable (`triggered_uptake_unsupported`), and clustered daily outcome histories are unavailable. A weak first stage leaves a typed LATE failure (`estimation.encouragement.late.weak_first_stage`, or `estimation.encouragement.late.negative_first_stage` when encouragement lowered uptake); it is never silently dropped.

### Windows, timelines and retained state

Definitions/artifacts apply uptake windows to raw timestamps **before** day
bucketing. For a one-day window starting at noon, uptake at exposure or 21
hours later qualifies, but exactly 24 hours later does not. Frame panels
contain day-reduced flags; use elapsed exposure days when reducing the same
events for cross-source comparisons.

Qualifying uptake after the last outcome day remains visible as-of. It does not
add outcome-observation days, change calendar averages or admit newer outcome
cohorts; completed outcomes still need their last required observation day.
With `completed_windows_only=True`, compliance requires closed uptake windows.
Definitions/artifacts additionally wait for **both** outcome and uptake windows
on completed cumulative outcome rows, including ITT-only requests. A separate
uptake feed shares the data snapshot; without the filter, ITT-only requests
need no uptake moments.

As-of compliance admits units assigned by the day and uptake through that day.
Triggered compliance instead admits by first eligible trigger day, while keeping
assignment-anchored uptake windows. Triggered sequential compliance is unavailable
(`triggered_uptake_unsupported`); clustered daily outcome histories are unavailable.

Continuous ITT uses the randomized asymptotic mean confidence sequence: assignment
is randomized, so its law depends on allocation, not the mechanism's label.
Binary-uptake LATE is fixed-horizon, not a bivariate Gaussian observation model.
When compliance joins a sequential family, e-BH selects the joint continuous-ITT/
Bernoulli-compliance family. Compliance is exact under its conditions, continuous
ITT asymptotic; every row shares `family_guarantee="asymptotic_sequential"` when
a continuous member is present.

Frame as-of summaries preserve calendar, numeric and structured-string day labels
and the metric moments' chronological order. Open-ended native compliance timelines
extend through the latest enrollment or post-enrollment uptake event, including
repeats outside the uptake window. Outcome filters/missingness do not shorten
them; an explicit observation horizon takes precedence.

Custom stores supply `compliance_series(ComplianceSeriesRequest)` independently
of outcome metrics, or receive the shared capability refusal. Outcome dates
cannot substitute. This typed contract covers builder-backed and standalone
custom providers; separate source facades retain their own contracts. See
[custom providers](../development/adding-an-estimator.md#custom-evidence-providers).

Clustered compliance is **total uptake / total members**, not a mean of cluster
rates. Uncertainty uses centered cluster uptake-total/size moments.
`ComplianceSummary` stores immutable sufficient state, distinguishing member
counts from independent cluster counts.

Native/frame/artifact `export_moments()` preserves compliance points and
uncertainty. Reload with the same design; omit `experiment_id` to recover the
exported identity or explicitly supply that same identity.

Saved formats:

- Every current cube row has version-1 `compliance_summary`.
- New encouragement artifacts contain assignment counts, cluster IDs when needed,
  and version-2 uptake with first qualifying timestamp and enrollment/uptake
  observation edge from the publication snapshot.
- Timestamp-bearing artifacts missing that edge use enrollment/first qualifying
  uptake dates as a conservative coverage bound.
- Boolean-only uptake artifacts and cubes missing the payload raise
  `source.compliance_summary.legacy_uptake_state`, with a re-export remedy.
- Malformed current state is a separate error, including disagreement between
  compliance member counts and cube assignment-unit counts. Mixed/unassigned
  accounting entries are excluded from that comparison.
- Standalone summaries preserve ISO calendar dates, numeric values and tagged
  strings/datetimes; as-of lift JSON also preserves day labels.

LATE's assumptions follow Imbens and Angrist,
[“Identification and Estimation of Local Average Treatment Effects”](https://doi.org/10.2307/2951620);
diagnostics do not validate random assignment, relevance, exclusion or monotonicity.

## Logged-policy evaluation

### Policy value and uncertainty

A trace holds, for each unit, `T` decisions in order. At decision `t` the logging policy saw a history `H_t`, chose an action `A_t` with probability `p_t = P(A_t | H_t)` (the logged **propensity**), and a reward `Y_t` closed later. A candidate policy `pi` assigns its own probability `pi(A_t | H_t)` to the same action at the same history.

The estimand is the **finite-horizon dynamic policy value contrast**

```text
Delta_T = V_T(target) - V_T(reference),    V_T(pi) = (1/T) sum_t E_pi[Y_t]
```

the difference in mean reward per decision, averaged over the horizon, that the target policy would have produced against the reference policy. Each `V_T(pi)` is estimated by a self-normalized importance-weighted mean at every decision index with the cumulative likelihood ratio `W_t = prod_{s <= t} pi(A_s | H_s) / p_s`, so later contexts that depend on earlier actions are re-weighted through the whole trajectory, not only the current step. The interval is a two-sided Wald interval from a unit-clustered sandwich standard error on a `t` reference with `n_units - 1` degrees of freedom.

`contrast.estimate` is `Delta_T` on the reward scale, with `lb`, `ub`, `se` and `p_value` for the null of no difference. `target_value` and `reference_value` are the two policy values, `ess_by_time` the smaller of the two policies' effective sample sizes at each decision index, and `logging_policy_versions` the single version the trace was logged under.

This API accepts a `LoggedTrace`, not an experiment reader. Experiment builders
and `run()` do not carry decision histories: their capability record is
“not expressible,” not a refusal with a `.code`.

### Fixed logger and support

* The logger must be one fixed `(policy_id, version)` throughout the trace. An updating bandit is refused with `logged_policy.inference.adaptive_logging_unsupported`; a logger that did not refit during collection is allowed. This interval is asymptotic in the number of independent units, not a finite-sample guarantee.
* Propensities must be the exact values recorded at decision time. The trace checks them against the registered logger or its complete recorded action distribution; it does not infer, normalize, clip, or trim them.
* Every positive candidate-action logging probability must be at least `PROPENSITY_FLOOR = 0.05`, including for unchosen actions. Zero is allowed only when both evaluated policies assign zero probability. Set this floor in the logger: filtering traces afterward can select on outcomes and make the survivors unrepresentative.

Two measured floors prevent use where the interval is poorly calibrated:

- At least 20 independent units (`logged_policy.inference.cluster_floor`).
- Effective sample size (ESS) of at least 20 for both policies at every decision (`logged_policy.support.ess_floor`).

Refusals include the ESS, maximum cumulative weight, and minimum propensity for every decision index, so you can diagnose the shortfall.

IDs and versions are opaque: slashes are allowed, and identical display labels do not merge distinct pairs. The construction that makes the interval valid under adaptive collection (stabilized or adaptively weighted estimators with a martingale variance, as in Hadad et al. 2021 and Zhang, Janson and Murphy 2021) is not implemented; the route forward is one policy version per trace.

Admission checks the whole supplied distribution, not just the observed action. A smaller positive probability violates the declared weight bound, not the existence of a finite likelihood ratio. The effective sample size of a policy can never exceed the unit count, so fewer units could never pass the ESS floor either; the unit count is named first because that is the actual shortfall.

For both policies and at every decision index, `ESS_t = (sum_i W_it)^2 / sum_i W_it^2` must be at least `ESS_FLOOR = 20` (`logged_policy.support.ess_floor`).

Recorded distributions are claims about the original logger, not proof that
those claims are true. Do not relabel adaptive collection as one fixed version.

### ESS calibration evidence

In a fixed-logger simulation spanning target divergence, horizon, and unit count (26 cells; 9,188 intervals), 95% interval coverage by smallest per-index ESS was 0.862 for `[2, 5)`, 0.892 for `[5, 10)`, 0.933 for `[10, 20)`, 0.933 for `[20, 40)`, 0.943 for `[40, 80)`, 0.947 for `[80, 160)`, and 0.949 above 160. Each band held 700–2,190 intervals. `[20, 40)` was the first band where every band stayed within three Monte Carlo standard errors of nominal. Below the floor, the problem was underestimated variance: the sandwich SE was about 20% too small below ESS 5 and 7% too small at ESS 10–20; estimate bias stayed within Monte Carlo error.

Those are measured results for the stated simulation, not a universal coverage
guarantee. Passing the floor does not turn this asymptotic interval into a
finite-sample or anytime-valid procedure.

### Importing decision records

`LoggedTrace.from_records` accepts `DecisionRecord` objects or mappings.
`LoggedTrace.from_frame` accepts pandas, polars or pyarrow frames with record
fields and the declared context columns. Supply exactly one logging-law source:
a `PolicyRegistry`, or the complete action distribution recorded per decision.
A chosen-action propensity alone is insufficient.

This self-contained import example uses a simulated fixed logger to provide
representative records. For production, retain the actual decision-time values,
not a model fitted afterward.

```python
import pandas as pd
from increment.estimators import LoggedTrace
from increment.simulate import simulate_logged_run

logged = simulate_logged_run("fixed_random", n_units=200, horizon=3, seed=7)
records = [record.model_dump(mode="python") for record in logged.trace.records]
recorded_laws = [
    dict(zip(record.candidate_actions, probabilities, strict=True))
    for record, probabilities in zip(
        logged.trace.records, logged.trace.logging_distributions, strict=True
    )
]

# Reproduce the original logger with its registered implementation.
registry_trace = LoggedTrace.from_records(records, registry=logged.registry)

# Or use the original complete distributions instead of executable policies.
recorded_trace = LoggedTrace.from_records(records, logging_distributions=recorded_laws)

# Frame inputs put the pre-decision context in named columns.
frame_records = []
for record, law in zip(records, recorded_laws, strict=True):
    row = dict(record)
    row.update(row.pop("pre_decision_context"))
    row["logging_distribution"] = law
    frame_records.append(row)
frame = pd.DataFrame(frame_records, dtype=object)
frame_trace = LoggedTrace.from_frame(
    frame,
    logging_distribution_column="logging_distribution",
    context_columns=("x", "prior"),
)
registry_frame_trace = LoggedTrace.from_frame(
    frame.drop(columns="logging_distribution"),
    registry=logged.registry,
    context_columns=("x", "prior"),
)
assert all(
    trace.records == logged.trace.records
    for trace in (registry_trace, recorded_trace, frame_trace, registry_frame_trace)
)
```

`recorded_laws` contains one mapping per input record, for example
`{"A": 0.2, "B": 0.3, "C": 0.5}` for candidates `("A", "B", "C")`.
The record and its mapping are paired before ordering by unit and decision
index, then copied into immutable snapshots. Every mapping must cover exactly
the candidate actions, contain finite nonnegative probabilities summing to one
within the existing tolerance, meet the positive-probability floor, and agree
with the chosen-action propensity. A chosen propensity alone is insufficient.

Both alternatives retain support, fixed-logger identity/version, unit-count and
ESS checks. Recorded distributions are claims about the original logger, not
proof of their truthfulness. Do not relabel adaptive collection as one fixed
policy version; this admission route does not change the estimator or its guarantee.

Frame unit IDs may be strings or integers. Distinct native IDs that stringify
identically, such as `1` and `"1"`, refuse with
`estimation.crossfit.identity_collision` rather than becoming one trajectory.
The records constructor continues to require string IDs.

A `TabularPolicy` is an action distribution over one context key; any object
with `policy_id`, `version` and `probability(action, context)` satisfies the
`StochasticPolicy` protocol and can stand in for a hand-written rule.

### Persisting a TabularPolicy

`probabilities` and `default` are immutable copies of what you pass in.

- **Python persistence** (`model_dump(mode="python")` then `model_validate`,
  `copy.deepcopy`, and pickle) keeps every supported context key type,
  including distinct `1` and `"1"`, tuples and datetimes, and rebuilds a
  policy with the same action probabilities and the same `default` (or none).
  Pickle is for trusted data only.
- **JSON persistence** (`model_dump(mode="json")`, `model_dump_json`,
  `model_validate_json`) roundtrips only when every table key is a string,
  including default-only and no-`default` policies. Any other key type refuses
  at dump time with `logged_policy.policy.json_context_key`, because JSON
  would silently turn `1` into `"1"` or merge the two. Construction accepts
  such keys; use Python persistence or string-keyed tables. This includes the
  built-in integer-keyed `TARGET_POLICY_V1`.

### Package interactions

- No multiplicity: one `alpha` per call. Comparing several target policies
  against one reference is several calls, and no family correction is
  applied across them.
- No clusters beyond the unit: the trajectory is the independent replicate.
  Units nested in a shared cluster are not modelled.
- Fixed horizon only: every admitted unit must have all `T` rewards closed.
  Re-running on a growing trace is repeated fixed-horizon inference and has
  no anytime validity.
- No CUPED, winsorization, ratio rewards or breakouts.
