# Causal inference in Python: IPTW, DML, and AIPW

When treatment is self-selected, phased, or opt-in, a raw arm difference is not automatically causal. Declare and justify the adjustment set. Increment checks overlap and post-adjustment balance, but cannot validate conditional ignorability.

Choose IPTW, DML, or AIPW only when the causal assumptions are defensible.

IPTW, AIPW, and DML run identically from `Analysis.from_unit_summary`, `Analysis.from_definitions`, `Analysis.from_unit_day_artifact`, and `Analysis.from_unit_panel` (unwindowed mean/conversion metrics only): identical per-unit data returns the same rows and intervals on each. Ratio metrics are not covered by any of the three (see [Current limits](#current-limits)). `from_moments` refuses by name (see [Current limits](#current-limits)).

## Declaring an adjustment set

An `Observational` design has three parts:

- `control_group`: which group plays the role of control.
- `adjustment`: an `AdjustmentSet` naming the covariate columns that,
  conditioned on, make treatment as-good-as-random.
- `gate`: an `IdentificationGate` controlling what happens when the data
  cannot support that claim (see below).

`AdjustmentSet` is a declaration, not a validation: the library cannot check
that your covariates satisfy conditional ignorability. That argument is
yours to make. What the library *does* check is whether the declared set
achieves overlap, and whether the declared columns — as given — are
balanced after weighting. The balance check is necessary evidence, not
sufficient (see [Balance checks](#balance-checks) below).

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(7)
n = 4_000

tenure_days = rng.gamma(shape=4.0, scale=90.0, size=n)
plan_tier_rank = rng.integers(0, 3, size=n).astype(float)

# Opt-in probability rises with tenure and plan tier: confounding.
logit = -1.2 + 0.004 * (tenure_days - 360) + 0.5 * (plan_tier_rank - 1)
treated = rng.random(n) < 1 / (1 + np.exp(-logit))

# Revenue depends on the same covariates, plus a true +5% treatment effect.
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

from increment import (
    AdjustmentSet,
    Analysis,
    IdentificationError,
    IdentificationGate,
    Method,
    Observational,
)

results = Analysis.from_unit_summary(
    df,  # one row per unit; must also carry the adjustment covariate columns
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("tenure_days", "plan_tier_rank")),
    ),
).run()

for r in results:
    print(
        f"{r.metric} / {r.group_id} [{r.method}]: "
        f"lift={r.lift.value:+.2%} "
        f"CI=({r.lift.lb:+.2%}, {r.lift.ub:+.2%})"
    )
```

```text
revenue / opted_in [iptw]: lift=+5.67% CI=(+3.89%, +7.45%)
```

On this data a naive randomized-style analysis reports a lift of **+19.1%**
(nearly four times the true +5% effect) because long-tenured,
higher-tier users both opt in more and spend more. The adjusted estimate
recovers the truth.

Under an `Observational` design, `run()` defaults to inverse propensity of
treatment weighting (IPTW): a model of who gets treated is fit on the
declared covariates, each unit is weighted by the inverse of its propensity,
and the weighted (Hájek, self-normalized) arm means are compared. With more
than one treatment arm, every arm mean targets the same eligible population
(see [Multiple treatment arms](#multiple-treatment-arms)). The
influence-function standard error is a plug-in calculation that treats the
fitted propensity as known for generic, pattern-specific, and trimmed fits.
The untrimmed pooled-logistic native path additionally includes the fitted
propensity estimating equations (of every treatment-versus-control model when
several treatments share the control) and their covariance with the arm means.
For correctly specified, untrimmed parametric IPTW on the generic plug-in
path, this is deliberately conservative (Lunceford & Davidian 2004) —
when the covariates strongly drive uptake, measured intervals run ~1.5–1.8×
wider than the true sampling spread (99.5% observed coverage at a nominal
95%), so coverage is above nominal and power is below what a refit-aware (e.g.
bootstrap) interval would give. This conservative-direction claim does not
extend to the fitted pooled-logistic covariance correction, fitted overlap
trimming, or AIPW/DML cross-fitted orthogonal-score intervals.
With `gate.overlap="trim"`, the IF interval conditions on the fitted retained
set and omits estimated trim-boundary variability.

!!! note
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

## The identification gate

Weighting only works where the arms actually overlap: a unit whose fitted
propensity is near 0 or 1 has essentially no counterpart in the other arm,
and its inverse weight explodes. `IdentificationGate` decides what happens
then, and it **defaults to refusing**:

- `overlap="refuse"` (default): if any unit's fitted propensity falls
  outside `[min_propensity, 1 - min_propensity]` (default `0.01`), `run()`
  raises `IdentificationError` instead of returning a number.
- `overlap="trim"`: an explicit opt-in to drop the poorly-overlapping
  units and estimate on the rest.

```python
# A covariate that near-deterministically separates the arms: overlap fails.
df2 = df.with_columns(
    pl.Series("signup_score", np.where(treated, 5.0, -5.0) + rng.normal(0, 0.5, n))
)

try:
    Analysis.from_unit_summary(
        df2,
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("signup_score",)),
        ),
    ).run()
except IdentificationError as err:
    print(err)
```

The refusal message reports how many units in each arm fall outside the
overlap window and the propensity deciles, so you can see whether the
problem is a few extreme units or a wholesale separation of the arms.

Trimming is not a free fix: it **changes the estimand**. You are no longer
estimating the effect on the full population, only on the subpopulation
where both arms are represented. The result records this honestly:

```python
design = Observational(
    control_group="control",
    adjustment=AdjustmentSet(covariates=("tenure_days", "plan_tier_rank")),
    gate=IdentificationGate(overlap="trim", min_propensity=0.05),
)
results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=design,
).run()

for r in results:
    print(f"{r.group_id}: lift={r.lift.value:+.2%} population={r.population}")
```

```text
opted_in: lift=+5.46% population=overlap e in [0.05, 0.95] (3968 of 4000 units)
```

`LiftEstimate.population` is `None` for an untrimmed estimate; when set, it
names the overlap window and how many units survived. Note the overlap
subpopulation is defined by the *fitted* propensity, not the true one: the
trim boundary is itself estimated from this sample, so the trimmed estimand
is a model-defined (sample-dependent) subpopulation. Its IF interval
conditions on the fitted retained set and omits estimated trim-boundary
variability; the untrimmed conservative-direction claim does not apply.

With several treatment arms the gate is one mask over every arm's
*marginal* propensity, control included: a unit is kept only when each of
them is at least `min_propensity`, so every row of the metric reports the
same retained population (`overlap with every marginal arm propensity >= g
(kept of total units)`), never a separately trimmed comparison. The
propensities of M arms sum to one, so the band is empty when
M × `min_propensity` > 1; the refusal names that condition. A trim that
leaves an arm with no units refuses, naming the arm.

### Balance checks

After weighting, the gate also inspects covariate balance via the
standardized mean difference (SMD) of every declared covariate -- the
weighted mean difference divided by an unweighted pooled SD (Austin &
Stuart 2015), so an inflating weight scheme cannot shrink the denominator
to slip past the gate:

- By default, any covariate with |SMD| > 0.1 after weighting produces an
  advisory `UserWarning`.
- Set `gate.max_smd` to turn that into a hard refusal: any covariate
  exceeding the threshold raises `IdentificationError`, on the grounds that
  the declared adjustment set does not achieve balance for this comparison.

!!! warning
    The SMD check measures balance of the declared columns as given. The
    default propensity model is linear in those same columns, so it will
    approximately balance them **by construction** even when it is badly
    misspecified — a categorical encoded as an integer rank can pass every
    balance check while the estimate is several times the truth (measured:
    a 3-level categorical smuggled in as an integer code produced a bias of
    +0.59 on a true lift of 0.10 — 6× the truth — with the SMD advisory
    firing in 0 of 60 replications; the same data dummy-encoded recovered
    the truth). Balance on the declared covariates is necessary, not
    sufficient; pass categorical strings for internal level encoding and
    consider checking balance on transformations yourself.

### Missing covariate values

A null/NaN in a declared covariate column defeats both gates
arithmetically: NaN compares False against every threshold, so an
explicitly-set overlap or balance gate would be silently disabled.
`AdjustmentSet(missing=...)` declares what a missing covariate value
means, and the policy is enforced before any learner sees the matrix and
before either gate runs:

- `missing="refuse"` (default): `run()` raises `IdentificationError`,
  naming each incomplete covariate and its missing count, together with
  the data-keeping alternatives below. Do not repair a confounder with
  `increment.impute.pooled_mean`: filling it without a missingness
  indicator can hide residual confounding. That helper is for randomized
  pre-period covariates such as a CUPED baseline.
- `missing="impute-indicator"`: impute incomplete numeric covariates with
  their pooled mean and categorical covariates with their pooled modal level,
  over every analysed arm jointly, never per arm. Append each incomplete
  column's missingness indicator, which enters propensity fitting and SMD
  diagnostics. The estimate carries a `note` recording the repair.
- `missing="pattern"`: fit the propensity separately per missingness
  pattern — the generalized propensity e(X_observed, pattern), which
  balances the observed covariates and the pattern itself with no
  assumption on the missing-data mechanism. Refused when the distinct
  patterns are too many (above a 20-pattern cap) or too thin (below a
  30-unit floor), with the refusal recommending `impute-indicator` or
  coarsening the missingness upstream: scattered per-covariate
  missingness makes every per-pattern fit too thin to trust. Structural
  missingness — a few fat patterns, e.g. per-source join gaps — is its
  home turf. IPTW only: DML and AIPW refuse it (per-pattern outcome models
  are not implemented for them) and point to `impute-indicator` or `allow`
  (raw NaN to explicit NaN-native learners, below). With several treatments,
  each treatment-versus-control propensity is fit per pattern and predicted
  for every unit sharing that pattern, and the 30-unit floor counts that
  comparison's own units.
- `missing="allow"`: pass NaN through to explicitly supplied NaN-native
  learner(s), without appending missingness indicators to their inputs.
  Numeric columns remain unchanged; categorical columns become fitted level
  indicators, with NaN across their block for a null level. If a fit sees
  only one level, an otherwise-zero column retains those NaNs rather than
  dropping them with an empty indicator block. Gates and SMD diagnostics use
  pooled-mean (numeric) or pooled-mode (categorical) imputation plus indicators,
  extended with `{c}__missing*{b}` indicator×covariate
  interaction SMD rows (each missingness indicator crossed with every
  *other* imputed covariate; self-crosses are collinear with the
  indicator row and excluded). Identification is the same
  missingness-pattern condition `pattern` asserts — ignorability given
  the observed values and the pattern itself — realized without
  enumerating patterns. Requires explicit learners: declaring `allow`
  with any defaulted learner that consumes X refuses unconditionally
  (`propensity_learner` for IPTW, BOTH factories for DML/AIPW), because
  the package defaults do not raise on NaN — they silently return
  non-finite predictions, which the NaN-blind gates cannot catch. A
  NaN-capability probe (fit/predict on a small NaN-planted matrix,
  refusing on raise or non-finite output) and a post-fit finiteness
  guard on every cross-fitted nuisance close the remaining gaps, and
  any allow-path learner raise surfaces as an `IdentificationError`
  naming the learner, fold, and arm. Only NaN passes through: ±inf
  refuses by column name. The estimate carries a `note` naming each
  incomplete covariate with its missing count.
- `missing="complete-case"`: analyse only fully-observed units,
  relabelling the estimand via `LiftEstimate.population` — the same
  honesty mechanism overlap trimming uses. Discouraged, and deliberately
  never suggested by the refusal messages: it is biased whenever
  missingness is informative. Prefer the options above, which keep every
  unit.

!!! warning "NaN tolerance is not capacity"
    `allow` realizes its identification claim only up to the supplied
    learner's capacity to represent pattern-specific (indicator ×
    covariate) structure — splits on the NaN-routing direction FOLLOWED
    by splits on other covariates. A depth-≥2 learned-split-direction
    tree learner (LightGBM/HistGradientBoosting-style missing-value
    routing) has that capacity. Additive/GAM-like NaN-native learners
    (depth-1 stumps with learned NaN directions) and surrogate-split NaN
    handling (CART-style: route NaN by a correlated observed covariate)
    do NOT deliver it: measured on the design DGP, an additive
    NaN-native learner passes every refusal layer and every marginal
    balance check while shipping bias +0.4 — only the
    indicator×covariate interaction SMD rows see it (flagged in 30/30
    negative-control replications). Those rows are advisory-only unless
    `gate.max_smd` is set, so **set `gate.max_smd` whenever you declare
    `missing="allow"`** — it is the only hard line between a
    finite-but-uninformative learner (the refusal layers guarantee
    finite nuisances, not informative ones) and a confident wrong
    number. Interaction columns are higher-kurtosis than marginal ones,
    so the 0.1 advisory can chatter (~7% of clean runs at n=4000 even
    under an oracle propensity).

Choosing between `pattern` and `allow`: both assert the same
missingness-pattern identification, and both inherit its caveat — a
covariate that confounds when observed must not confound through its
unobserved value within a pattern, which needs a domain argument.
Choose `pattern` for a few fat structural patterns with the parametric
default learner (IPTW only); choose `allow` for scattered or
high-dimensional missingness with an interaction-capable NaN-native
learner — it needs no pattern enumeration, works on all three methods,
and degrades gracefully when a pattern is rare within a training fold
(learner-dependent at the all-NaN-column edge, where a raising learner
surfaces as a named refusal rather than a traceback).

## Near-zero baselines: reporting the additive effect

Prior-free IPTW, AIPW, and DML retain the joint covariance of the additive
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
An explicitly informative IID prior retains the separate scalar path and its
near-zero denominator guard: it refuses `abs(mu0) <= 4 * SE(mu0)`. A declared
cluster excludes that prior path.

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

The additive effect `tau = mu1 - mu0` is perfectly well identified there.
Ask for it per metric:

<!-- invisible-code-block: python
rng2 = np.random.default_rng(21)
m = 800
x = rng2.normal(0.0, 1.0, m)
opted = rng2.random(m) < 1 / (1 + np.exp(-0.5 * x))
net_delta = 0.05 * x + 0.02 * opted + rng2.normal(0, 1, m)
analysis = Analysis.from_unit_summary(
    pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(m)],
            "variant": np.where(opted, "opted_in", "control"),
            "net_delta": net_delta,
            "x": x,
        }
    ),
    unit="user_id",
    group="variant",
    metrics={"net_delta": "mean"},
    design=Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("x",)),
    ),
)
-->

```python
results = analysis.run(value_scale={"net_delta": "absolute"})
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
- `prior=None` means *exactly* flat. The default `Normal(0, 1e6)` prior is
  "approximately flat" only in unitless terms; on the additive scale it
  shrinks a point estimate with `se(tau) = 1e5` by 1% and one with
  `se(tau) = 1e6` by 50%. An explicit `prior=` is honored and read in the
  metric's own units, and is refused outright on a call that mixes scales
  (one `Normal` cannot be a percentage on one row and dollars on another).

The reporting scale is never inferred from the data. Auto-switching a row
to absolute when the guard fires would make the units of a metric column
depend on sampling noise — in an as-of series, early dates could report
dollars and later dates percentages under one heading.

**Every relative row also carries the additive pair.** `abs_diff`,
`abs_se`, `abs_lb`, and `abs_ub` are populated on ordinary relative
observational rows too, computed from the same influence functions — which
is what makes a declared absolute margin (`Metric.margin_abs`/
`ExperimentMetric.margin_abs`, absolute-unit non-inferiority) work on this
path. On the generic plug-in IPTW path the additive SE ignores propensity
estimation and is conservative (see the ~1.5–1.8× inflation measured under
strongly prognostic covariates above), so its coverage runs at or above
nominal; the untrimmed default logistic path includes the fitted-propensity
estimating equations instead. `dml` and `aipw` are calibrated to within a
few percent.

## Choosing `iptw` vs `dml` vs `aipw`

Three adjustment methods are available under the same design, selected via
role-aware `run(...)`:

```python
analysis = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=design,
)
results = analysis.run(decision_method=Method(name="dml"))  # default: iptw
```

- `Method(name="iptw")` (the default) fits one propensity model per
  treatment and compares self-normalized weighted arm means. It relies
  entirely on the propensity model being right.
- `Method(name="aipw")`: augmented IPW (doubly robust). Fits the propensity
  model(s) AND one outcome model per arm, cross-fit. Targets the SAME
  unit-weighted ATE as `iptw`. Its point estimate is consistent if EITHER
  nuisance model is correctly specified, not just attenuated toward zero the
  way DML is; its influence-function interval additionally assumes both
  nuisances converge fast enough (their error product vanishing faster than
  1/√N), so one badly misspecified model can leave a consistent point with a
  miscalibrated interval. Prefer `aipw` over `iptw` whenever you are willing
  to fit an outcome model too.
- `Method(name="dml")`: double machine learning, cross-fit
  partialling-out, fits a propensity model and ONE pooled outcome model
  (not per-arm) for the slope. The slope is first-order insensitive to
  *small* estimation error in either model. This is not double robustness:
  if the propensity model is badly misspecified, a correct outcome model does
  not rescue the slope — it is attenuated toward zero by the propensity
  model's squared error.

**`dml` and `aipw`/`iptw` do not target the same estimand under
heterogeneous treatment effects.** `dml`'s partially-linear model solves
for a propensity-variance-weighted average slope

```text
theta = E[(D - e)(Y - l)] / E[(D - e)^2] = E[e(1-e)·tau(X)] / E[e(1-e)]
```

with `e(X)` the propensity, `l(X) = E[Y | X]` and `tau(X)` the conditional
effect (`estimand="plr_slope"`). It coincides with the unit-weighted ATE
(`mu1 - mu0`, what `iptw` and `aipw` both target) when the effect is
homogeneous, when the propensity is constant, or whenever `e(1-e)` happens to
be uncorrelated with `tau(X)`; not in general. On a heterogeneous-effect DGP
the two numbers are genuinely different quantities, not two estimates of the
same one — measured 26 Monte Carlo standard errors apart in one worked design
review. If you need the ATE proper under effect heterogeneity, use `iptw` or
`aipw`, not `dml`. Choose `dml` when you specifically want the
propensity-weighted PLR quantity and expect a flexible outcome model to earn
its keep alongside the propensity model.

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

All three pass through the same identification gate and default to the
same built-in learners: a ridge-stabilized logistic regression for the
propensity model, plus a closed-form ridge linear regression for the
outcome model(s) (`dml` fits one pooled model per treatment, plus the
control-arm model on relative rows; `aipw` fits one per arm). No extra
installs are required, and you can bring your own model: anything with
`fit(X, d)` and `predict(X)` satisfying the `Learner` protocol works,
including a thin wrapper over an sklearn or LightGBM estimator, passed via
`Method(name=..., propensity_learner=..., outcome_learner=..., folds=...)`
(all three take zero-argument factories via `propensity_learner`; each
factory is called afresh for every populated fold of every model: the
propensity factory once per treatment, AIPW's outcome factory once per arm,
DML's once per treatment plus once more per fold for a relative row's
control-arm model after every slope fit). `iptw` fits its propensity
models with no cross-fitting, so it calls its factory once per treatment. When
a contrast's covariates actually contain NaN values and `missing="allow"`
is declared, `iptw` reuses that same instance for both the fit and a
capability probe, while `dml`/`aipw` construct one extra (discarded)
probe instance per factory. Infinite covariate values are refused
outright under `missing="allow"`, before any probe runs. Only NaN reaches
the probe; a contrast with finite covariates skips it regardless of the
declared `missing=` policy.

`iptw` has no outcome model or folds. Setting either option on an `iptw`
`Method` raises a refusal.

!!! warning "Flexible IPTW learners can overfit"
    IPTW fits and evaluates propensities on the same units; only `dml` and
    `aipw` cross-fit. A learner that memorizes treatment assignments pushes
    fitted propensities toward each unit's treatment indicator. Weights then
    approach 1, biasing the estimate toward the naive contrast. The overlap
    gate catches saturated propensities, not overfitting within its bounds.

    Prefer `dml` or `aipw` for flexible learners. Unlike overlap and balance
    failures, this modeling risk has no runtime flag: fitted propensities
    alone cannot establish whether the learner overfit.

For `dml` and `aipw`, cross-fitting is deterministic and balances treatment arms
across folds. With a declared cluster, the cluster is the split unit: every unit
in one cluster receives the same fold, so no validation cluster (including its
other units) can enter that fold's training data. The implementation ranks unique
cluster labels with a stable content hash, making folds invariant to source row
order. When every cluster contains exactly one unit, it ranks unit IDs instead;
this preserves the same fold assignments as an unclustered analysis. Each arm
must appear in at least `folds` distinct clusters, counting both clusters confined
to that arm and clusters shared with other arms. Observational dependence clusters
may span treatment values; shared clusters remain intact while the split balances
per-arm counts across folds. Arm purity is required only for randomized and
encouragement designs. With several treatments one fold vector, stratified by
every arm, serves every model of the metric, and each fold's training rows must
still contain each treatment and the control; otherwise the fold refuses, naming
the comparison.


`Method(name="unadjusted")` is allowed under an observational design, but
only when named explicitly; the result is confounded and labelled
accordingly, so it cannot be mistaken for an adjusted estimate.

## Multiple treatment arms

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

## Multiplicity across roles

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


## Clustered rollouts

A phased geo rollout is observational AND clustered: geos carry the
assignment, units are measured. Declare both and the adjusted estimators
handle the pairing:

```python
geo_rng = np.random.default_rng(7)
k, m = 40, 25  # geos per arm (2*k total), users per geo
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

(geo_result,) = Analysis.from_unit_summary(
    geo_df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    cluster="geo_id",  # the randomization-grain column
    design=Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("geo_z",)),
    ),
).run()
print(f"K={geo_result.n_clusters}, reference={geo_result.reference_kind}, dof={geo_result.dof}")
```

```text
K=80, reference=normal, dof=None
```

With `cluster` declared, IPTW/DML/AIPW sum centered member influence
contributions within each cluster and use their complete covariance. Pure
clusters do not justify removing between-arm variation for a superpopulation
target. The reference is asymptotic Normal (`reference_kind="normal"`,
`dof=None`), with the cluster count retained in `n_clusters`. The Bessel
multiplier does not correct fitted-response leverage or establish small-K
coverage. This path does not certify small-sample coverage.

With several treatment arms, support is checked per comparison on its own
rows, after any trimming: every comparison needs at least two distinct
clusters among its treatment and control units, and at least two treatment
and two control clusters when all of them are arm-pure. Clusters of other
treatments never count toward either requirement. An influence that vanishes
outside a comparison's treatment and control units is reduced over that
comparison's `K_a` clusters with `K_a/(K_a - 1)`; every other influence is
reduced over all `K` retained cohort clusters with `K/(K - 1)`. The DML slope
and the fixed-propensity IPTW influences (generic learners, pattern and
trimmed fits) are of the first kind. The fixed-propensity control mean, which
only control units carry, is reduced the same way, so its variance takes each
comparison's own factor. Those IPTW rows and absolute DML rows report `K_a` as
`n_clusters`. AIPW arm means, the untrimmed default logistic IPTW correction
and a relative DML row's control mean have terms on every cohort unit
(outcome-model population terms, every treatment-versus-control model's
estimating equations), so their totals, factor and `n_clusters` cover every
retained cluster, including clusters that hold only another treatment.

That cohort support is structural, set by the estimator rather than detected
from the data. When outcome predictions carry no population variation
(constant predictions, for example), AIPW influences and a relative DML row's
control mean vanish outside the comparison exactly as fixed-propensity IPTW
influences do, yet the row keeps `K/(K - 1)` and the cohort `n_clusters`. As
population terms shrink relative to residual terms, the comparison's own
clusters carry a growing share of the information. For AIPW, default logistic
IPTW and relative DML rows, `n_clusters` is therefore the widest support among
the row's influences, not the comparison's count. The small-cluster advisory
below counts the comparison's own `K_a`, and `K_a` can govern how reliable the
interval is even when `n_clusters` is much larger.

A relative DML row divides the slope by the cohort-wide control mean. The
joint covariance keeps each influence's own clustered variance (the additive
sidecar equals the absolute row's standard error; the shared control mean has
one variance on every row). It scales the covariance of the complete cluster
totals by `sqrt(c_a * c)`, with `c_a = K_a/(K_a - 1)` and `c = K/(K - 1)`,
rounded down to a multiple of `2**-64` so the matrix stays positive
semidefinite; the factor is exactly `K/(K - 1)` when both supports count the
same clusters, as in a binary cohort. Small-K calibration of this
mixed-support convention is unmeasured.

Untrimmed pooled logistic IPTW includes the fitted propensity estimating
equations and their covariance with every weighted arm mean. Generic,
pattern-specific and trimmed propensity fits still lack that correction.
Cross-fitted AIPW/DML require cluster-level nuisance-rate assumptions; linear
cross-fit residual geometry has not yet received a finite-sample correction.
The result note records these limitations. Target weighting is unchanged.

The randomized path's small-K policy applies as an advisory: when a
comparison's own treatment and control units span fewer than 40 clusters, a
`RuntimeWarning` names the over-rejection risk, while structurally valid
contrasts with fewer clusters still run on their qualified working
Normal/t reference. What else changes or refuses under a declared cluster:

- **Decision stats refuse.** `chance_to_beat()`, `prob_beyond()`, and
  `prob_favorable()` (both the relative and the absolute-margin branch)
  retain their clustered-row refusals. An asymptotic sampling interval does
  not establish the posterior decision model. Read the interval directly.
- **An informative `prior` refuses** — the clustered path bypasses the
  conjugate Normal-Normal update.
- **Ratio metrics still refuse** for IPTW/DML/AIPW. **Encouragement
  designs compose too** (ITT and additive LATE have their own component-based
  Welch references and the same small-K admission policy) -- CUPED and the complier-relative LATE row
  still refuse there; see the
  [encouragement + CUPED guide](cuped.md#clustered-encouragement-designs).
- The explicit `Method(name="unadjusted")` comparison reads the metric's
  retained unit frame and aggregates both arms jointly by distinct cluster.
  It counts clusters touching either arm once and retains signed cross-arm
  covariance for the absolute difference and relative Fieller set, including
  supported ratio metrics. The point remains the confounded observed
  group-mean comparison, weighted by members (ratio metrics use their declared
  denominator totals), not an equal-cluster or causal effect. Disjoint arms
  use per-arm Bessel covariance, additive Welch degrees of freedom, and a
  relative t reference at `min(K_T - 1, K_C - 1)`. Mixed arms correct the
  covariance for the response to estimating both arm means, using each
  cluster's member or denominator share. The absolute variance is unbiased
  under independent clusters, fixed masses, and common arm expectations,
  allowing unequal variances and arbitrary covariance within shared clusters.
  Ratio metrics with random denominators retain a delta approximation for
  each arm mean before the joint relative inversion. The Normal reference is asymptotic; random
  composition bias and small-K calibration remain unresolved. Singular
  response geometry retains an explicitly noted CR1 approximation. A negative
  corrected variance is unavailable rather than clipped; an available relative
  result can retain numeric-null absolute uncertainty when its additive
  variance is zero. Each arm must touch at least two clusters; the below-forty
  warning and existing value-scale restrictions remain.
  Mixed-cluster unit-frame sources can use this estimator route; dataframe
  ingress still rejects spanning labels.

## From definitions (the warehouse path)

Declare the covariate as a `pre_exposure` or `static` `Property` on the fact
source that carries it, then declare the design on the experiment:

```yaml
fact_sources:
  - name: users
    sql: SELECT * FROM users
    timestamp_column: updated_at
    entities: [user_id]
    facts: [...]
    properties:
      - {name: tenure_days, column: tenure_days, dtype: float, as_of: pre_exposure}

experiments:
  - name: rollout
    exposure: saw_feature
    unit: user_id
    control_group: control
    start: 2026-01-01
    plan: {...}
    design:
      mechanism: observational
      covariates:
        - {property: tenure_days, source: users}
```

`Analysis.from_definitions("rollout", "definitions/", con).run()` returns the
same rows as `from_unit_summary(..., design=Observational(...))` over the same
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
warehouse configuration keys. Use `from_unit_summary` or `from_unit_panel`
with a full `Observational` design for another missing-value policy.

A unit-day artifact automatically publishes every declared covariate:
numeric values use the unchanged `unit_covariate` relation, while strings
use `unit_covariate_level`, preserving nulls separately from literal labels.
Their typed requests are `UnitCovariateRequest` and
`UnitCovariateLevelRequest`; neither needs to be supplied to include a
declared covariate. `from_unit_day_artifact` reopens with the same
`Observational` design and returns the same rows. Undeclared or absent
extensions refuse with `artifact.extension.missing`.

`estimate_cate`, `validate_cate`, `targeting_rule`, and `select_targeting_rule`
read numeric or categorical properties passed to `interact=`/`adjust=` from
`from_definitions`, resolved the same way; no `design:` block is needed. From an
artifact they read only declared covariates, so `estimate_cate` (randomized
designs only) cannot read covariates from an artifact; use `from_definitions`.

## Current limits

- **`from_unit_panel` covers unwindowed metrics only.** It collapses to a
  per-unit total for an unwindowed mean/conversion metric (the same
  collapse `moments(grain="total")` already uses) and reads a covariate
  that is constant across each unit's own rows; a windowed or retention
  metric still refuses by name (`source.frame.unit_frame_panel`). For a
  windowed metric, compute each unit's windowed value upstream and declare it
  as an unwindowed metric on `from_unit_summary`; no frame source serves
  unit-grain estimators for a retention metric. A
  covariate that genuinely varies within a unit refuses by name too
  (`frame.frame_panel.unit_covariate_varies`), naming the offending units.
  A moments-only source (`from_moments`) raises `CapabilityError`
  (`source.moments.covariate_unavailable`) naming the covariates: a
  moments cube has no unit grain to attach weights to.
- **Quantile metrics refuse.** No observational quantile estimator exists:
  the distribution-free order-statistic interval assumes independently
  randomized arms, so a confounded contrast would be reported as a causal
  quantile lift, and the adjusted-mean machinery would report a mean effect
  under the quantile metric's name. The refusal is an
  `UnsupportedRequestError` with code `readout.observational.quantile` and
  the `metric` context, raised by `run()`, `estimate_ate` and
  `estimate_quantile_lift` before any outcome is read, and identically from
  every path that supports an observational design (`from_definitions`,
  `from_unit_day_artifact`, `from_unit_summary`, `from_unit_panel`) for a
  two-sided, zero-null request. A request that carries an absolute
  `margin_abs` is refused first with `readout.metric.quantile_alternative`
  (a one-sided tail), and a relative margin never reaches a readout: the
  constructor refuses it with `plan.observational.relative_margin`. Run a
  quantile metric under a randomized design. Fixed-horizon inference does not
  change this: no inference kind has an observational quantile estimator.
  No observational ingress can export a quantile cube either (`export()`
  raises the same code on definitions, a reopened artifact, and both frames,
  before any earlier metric in the catalog is read). `from_moments` therefore
  never receives a genuine
  quantile: a quantile *declared* over exported scalar moments constructs, and
  a two-sided, zero-null `run()` raises `readout.observational.quantile` before
  those moments can be used as quantile data (an absolute `margin_abs` raises
  `readout.metric.quantile_alternative` first; a relative margin raises
  `plan.observational.relative_margin` at construction), whereas
  `run_breakout()` raises
  `facade.analysis.operation` and the day-axis methods
  `facade.analysis.no_definitions` (source limits). See
  [Quantiles and portable moments](quantile-metrics.md#quantiles-and-portable-moments).
  `Analysis.planning_baseline` of a quantile metric still reads the control
  arm's per-unit values, since a planning baseline estimates no effect.
- **Ratio metrics refuse.** IPTW, DML, and AIPW reweight or residualize a
  single per-unit outcome; a ratio's numerator and denominator would need
  the adjustment applied jointly with their covariance retained, which is not
  implemented. The refusal is an `UnsupportedRequestError` with code
  `estimation.adjust_common.supported_ratio_metric`, carrying `method`,
  `metric`, `role`, `family`, and `correction` context. How it surfaces
  depends on the request:
  - When every declared metric is a ratio and each metric's decision method is
    IPTW, DML, or AIPW, `run()` raises it at entry, before any estimation. Only
    decision methods are examined: a sensitivity method such as `unadjusted`
    does not prevent the refusal.
  - A ratio metric that is a member of a Benjamini-Hochberg (`bh`) or e-BH
    (`e_bh`) family without an informative prior raises it at entry with
    that family named in the context, even when other metrics in the request
    are supported.
  - In other mixed metric lists, each unsupported pairing of a ratio metric
    with an IPTW, DML, or AIPW method is skipped with a warning
    (`estimation.adjust.skip_unsupported_metric`) while every supported
    metric/method pairing still returns its estimate (a ratio metric's
    `unadjusted` rows, for example). A skipped decision-role pairing records
    the hypothesis as unavailable rather than as a result row.

  Routes forward, each answering a different question. Under a randomized
  design, `Method(name="cuped", variance_reduction="cuped")` with a
  `covariate=` column adjusts a ratio metric's numerator and denominator
  jointly (see [CUPED](cuped.md#ratio-metrics)); it is not a confounding
  adjustment. Under an `Observational` design, the explicit
  `Method(name="unadjusted")` comparison supports ratio metrics but is the
  confounded observed comparison. Adjusting the numerator and denominator
  as separate mean metrics estimates two different estimands (an adjusted
  numerator mean and an adjusted denominator mean); it is not a
  ratio-effect analysis, and the two intervals cannot be recombined as
  independent intervals into an interval for the ratio.
- **No SRM check.** `srm()` returns `NotApplicable` under an observational
  design: a sample-ratio test presumes a target randomized allocation,
  which does not exist here.
- **No RELATIVE shifted nulls.** A declared absolute margin
  (`Metric.margin_abs`, or a plan-bound `ExperimentMetric.margin_abs`)
  works here — the additive interval every row now carries is exactly
  what that decision reads. A declared relative margin (`Metric.margin`,
  or a plan-bound `ExperimentMetric.margin`) is still refused by name:
  relative shifted-null plumbing is not built for this path yet. A
  shifted null of either kind targeting a `value_scale="absolute"` row
  is refused too — that boundary is a `null_lift` in the metric's own
  units, which lands with the relative pass.

See the [API reference](../api.md) for `Observational`, `AdjustmentSet`,
`IdentificationGate`, and `IdentificationError`.

## Next step

Compare method choices in [Which experiment analysis method should I use?](choose-a-method.md), or read [Encouragement designs](encouragement.md) when assignment changes uptake without forcing treatment.

## References and assumptions

The DML discussion follows Chernozhukov et al., [“Double/Debiased Machine Learning for Treatment and Causal Parameters”](https://doi.org/10.1111/ectj.12097). Conditional ignorability, consistency, and positivity identify the causal estimand; overlap and post-adjustment balance are diagnostics for support and model behavior, not validation of conditional ignorability.
