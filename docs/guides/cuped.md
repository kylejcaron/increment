# CUPED variance reduction for A/B tests

CUPED narrows an experiment's confidence interval by adjusting each unit's
outcome with a measurement taken before the experiment, such as last month's
revenue. Use it when that pre-period value predicts the outcome; the reported
effect still targets the same comparison.

!!! warning "A covariate does not promise a narrower interval"
    CUPED can widen an interval. A coefficient fitted from experiment
    outcomes has asymptotic, not finite-sample, guarantees. Check the reported
    estimate and interval; see [When CUPED does nothing](#when-cuped-does-nothing).

## On the dataframe path

Declare each covariate on the method that uses it. For the usual scalar
procedure, use `Method.cuped` with one continuous covariate. Declaring a
covariate alone changes nothing: `run()` keeps its declared or default
decision method.

```python
import numpy as np
import polars as pl
from increment import Analysis, CupedCovariate, Method, MetricSpec

rng = np.random.default_rng(11)
n = 20_000

# Pre-period revenue: a stable per-user spending habit.
pre_revenue = rng.gamma(shape=2.0, scale=15.0, size=n)
variant = np.where(rng.random(n) < 0.5, "treatment", "control")

# In-experiment revenue correlates strongly with the pre-period value.
noise = rng.normal(0, 10, size=n)
revenue = 5.0 + 0.9 * pre_revenue + np.where(variant == "treatment", 0.6, 0.0) + noise

df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": variant,
        "revenue": revenue,
        "pre_revenue": pre_revenue,
    }
)

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        MetricSpec(
            name="revenue",
            type="mean",
            sensitivity_methods=(
                Method.cuped(
                    covariates=CupedCovariate(
                        name="pre_revenue", kind="continuous", missing="impute"
                    )
                ),
            ),
        )
    ],
).run(decision_method=Method.unadjusted(name="unadjusted"))

for r in results:
    width = r.lift.ub - r.lift.lb
    print(f"{r.method:>10}: lift={r.lift.value:+.2%}  CI width={width:.2%}")
```

<!-- output -->
```text
unadjusted: lift=+2.01%  CI width=3.79%
     cuped: lift=+2.15%  CI width=1.77%
```

The adjusted estimate is similar, with an interval less than half as wide.

??? info "Statistical details: the example's precision gain"
    Here $\rho \approx 0.89$, so the predicted width ratio,
    $\sqrt{1 - 0.89^2} \approx 0.47$, matches the observed ratio,
    $1.77\% / 3.79\% \approx 0.47$.

Choose scalar CUPED for one continuous covariate when the ordinary
pairwise procedure is the goal. Choose multicovariate for multiple,
categorical, or indicator inputs, or when an intentional fixed-horizon HC2/BM
regression fit is desired. A backend does not select the procedure, and
adding covariates does not itself justify a different inference method.

### Comparing scalar and multicovariate procedures

The example below uses a single continuous covariate with three arms. The
decision method is predeclared scalar CUPED; multicovariate HC2/BM is reported
as a sensitivity. Scalar CUPED fits each treatment/control pair separately.
The multicovariate procedure fits all eligible arms together, so even with the
same covariate its coefficients and interval can differ.
Under a randomized design, adding a multicovariate sensitivity does not change
the scalar decision fit.

```python
import numpy as np
import polars as pl
from increment import Analysis, CupedCovariate, Method, MetricSpec

rng = np.random.default_rng(21)
n_per_arm = 80
arm = np.repeat(["control", "variant_a", "variant_b"], n_per_arm)
pre_period = rng.normal(size=3 * n_per_arm)
effect = np.repeat([0.0, 0.4, 0.8], n_per_arm)
outcome = 10 + effect + 0.9 * pre_period + rng.normal(0, 1, size=3 * n_per_arm)
covariate = CupedCovariate(name="pre_period", kind="continuous")

analysis = Analysis.from_unit_summary(
    pl.DataFrame(
        {
            "unit": [f"u{i}" for i in range(len(arm))],
            "arm": arm,
            "outcome": outcome,
            "pre_period": pre_period,
        }
    ),
    unit="unit",
    group="arm",
    control="control",
    metrics=[
        MetricSpec(
            name="outcome",
            decision_method=Method.cuped(covariates=covariate),
            sensitivity_methods=(Method.cuped_multicovariate(name="hc2_bm", covariates=covariate),),
        )
    ],
)

for row in analysis.run():
    lift = row.require_lift()
    print(
        f"{row.method:>6} {row.group_id}: "
        f"lift={lift.value:.4f}, interval=({lift.lb:.4f}, {lift.ub:.4f})"
    )
```

<!-- output -->
```text
 cuped variant_a: lift=0.0627, interval=(0.0345, 0.0916)
hc2_bm variant_a: lift=0.0641, interval=(0.0359, 0.0929)
 cuped variant_b: lift=0.0969, interval=(0.0658, 0.1289)
hc2_bm variant_b: lift=0.0955, interval=(0.0641, 0.1278)
```

This is a fixed-horizon procedure comparison, not an automatic choice:
ordinary scalar CUPED is the one-continuous-covariate default, while the
multicovariate fit is an explicit alternative.

!!! warning "Panel covariates must be constant per unit"
    Under fixed-horizon inference, `from_unit_panel` supports CUPED for
    unwindowed mean and ratio metrics when each method-owned CUPED covariate
    is constant across every unit's rows; varying values raise
    `frame.frame_panel.unit_covariate_varies`. Windowed and retention metrics
    reject CUPED covariate declarations on a panel (`frame.validation.from_unit_panel`):
    compute a non-retention windowed metric per unit upstream and declare its
    covariate on the method. No frame source serves a per-unit retention value;
    remove the method covariate to run retention without CUPED. For sequential
    CUPED, use the paths [below](#cuped-under-sequential-inference).

### Missing covariates

For scalar CUPED, choose the missing-value policy on its covariate:
`CupedCovariate(name="pre_revenue", kind="continuous", missing="impute")`
fills missing values with the pooled observed mean and warns with the count;
`missing="zero"` treats null as no pre-period activity, and `missing="refuse"`
rejects missing values. The policy belongs to the method, so decision and
sensitivity methods can make different choices without changing the raw
covariate binding.

### Ratio metrics

For a ratio metric, one covariate column adjusts the numerator and the
denominator separately, then the ratio is formed. The target remains
$E[N]/E[D]$:

```python
MetricSpec(
    name="revenue_per_order",
    type="ratio",
    numerator="revenue",
    denominator="orders",
    decision_method=Method.cuped(
        covariates=CupedCovariate(name="pre_revenue_per_order", kind="continuous", missing="impute")
    ),
)
```
The scalar `cuped` procedure uses one continuous covariate for both
components, with a separate slope for numerator and denominator. For multiple
or categorical covariates on a fixed-horizon mean or ratio, use
`Method.cuped_multicovariate`; it retains the joint component covariance.

On `from_definitions` and `from_unit_day_artifact`, the covariate is the
**pre-period numerator**: its total over the `n_pre_periods` days before
exposure, computed like a mean metric's covariate (see
[In YAML definitions](#in-yaml-definitions)). The numerator is usually the
outcome of interest and correlates with the denominator, so one covariate can
improve both components. Under fixed-horizon inference, `from_unit_panel`
accepts the same constant-per-unit covariate for an unwindowed ratio metric,
keeping it once per unit rather than summing it across days.

??? info "Statistical details: adjusting a ratio"
    Both components use the same covariate column but have separate slopes:
    $\theta_N$ from $\mathrm{Cov}(N, X)/\mathrm{Var}(X)$ and $\theta_D$ from
    $\mathrm{Cov}(D, X)/\mathrm{Var}(X)$, so the covariate can predict each
    component differently. Both adjusted components use the pooled covariate
    mean as their anchor, preserving the target ratio.

    Sharing a covariate can correlate the adjusted components even when the
    raw components are uncorrelated. The interval therefore uses all three
    adjusted second moments:

    $$
    \begin{aligned}
    \mathrm{Var}(N - \theta_N X) &= \mathrm{Var}(N) - 2\theta_N\mathrm{Cov}(N,X) + \theta_N^2\mathrm{Var}(X) \\
    \mathrm{Var}(D - \theta_D X) &= \mathrm{Var}(D) - 2\theta_D\mathrm{Cov}(D,X) + \theta_D^2\mathrm{Var}(X) \\
    \mathrm{Cov}(N - \theta_N X,\, D - \theta_D X) &= \mathrm{Cov}(N,D) - \theta_D\mathrm{Cov}(N,X) - \theta_N\mathrm{Cov}(D,X) + \theta_N\theta_D\mathrm{Var}(X)
    \end{aligned}
    $$

    These moments feed the same delta method, including the
    $-2\,\mathrm{Cov}$ term, as the unadjusted ratio. The interval retains its
    Welch-Satterthwaite $t$ reference and degrees of freedom. Keeping the raw
    $\mathrm{Cov}(N,D)$ instead would understate uncertainty.

## In YAML definitions

With `Analysis.from_definitions`, the experiment's `n_pre_periods` setting
supplies the covariate: each metric's total over
`[first exposure date − n_pre_periods days, first exposure timestamp)`,
including exposure-day events before the unit's first exposure. For a ratio
metric, it totals the numerator:

```yaml
experiments:
  - name: new_onboarding_v2
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    plan:
      secondaries:
        - purchase_rate
        - avg_session_duration
    n_pre_periods: 14  # CUPED lookback in days; 0 disables
    control_group: C
```

A unit with no pre-period activity gets zero in both groups. Under
randomization, this preserves independence from assignment; such units
contribute less precision.

`n_pre_periods` supplies the covariate but does not request the adjustment,
and the covariate is not queried until a CUPED method is requested. Add CUPED
to the decision or sensitivity method. The unused-covariate advisory follows
the same eligibility and defaulted-decision conditions as on the dataframe
path. Implicit scalar CUPED requires `n_pre_periods > 0`; an explicit
pre-exposure property binding does not. A covariate declaration with only
`name`, `kind`, and its missing policy resolves the property with that name.
Use `property` and `source` when an alias or source disambiguation is needed.
Call-wide methods override per-metric methods while retaining a declared
alias's property or measure binding. Each scalar family is acquired separately,
including when the decision and sensitivity families differ. A scalar breakout
override acquires the effective requested family; requests retaining several
distinct scalar families are refused before reading data.

Artifact publication retains raw named property and measure covariates, with
numeric NaNs represented as missing values. Replay applies each method's own
missing policy. Implicit scalar CUPED additionally needs its `cuped_preperiod`
extension selected at publication. Named scalar breakouts are not supported
by the artifact route: `from_unit_day_artifact.run_breakout` refuses with
`artifact.cuped.scalar_breakout` before reading evidence. Use definitions or
frame input for those breakouts, or run the artifact's scalar headline.

### Per-metric method bindings

Methods passed to `run()` apply to every reported metric. To request CUPED
for one metric, bind its methods in the experiment:

```yaml
experiments:
  - name: new_onboarding_v2
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-15
    plan:
      secondaries:
        - metric: purchase_rate
          decision_method:
            name: unadjusted
          sensitivity_methods:
            - name: cuped
              variance_reduction: cuped
        - avg_session_duration  # unbound -- reports unadjusted only
    n_pre_periods: 14  # CUPED lookback in days; 0 disables
    control_group: C
```

Each `analysis.run()` now reports unadjusted and CUPED rows for
`purchase_rate`; `avg_session_duration` stays unadjusted. Call-wide
`decision_method` and `sensitivity_methods` overrides take precedence. The
dataframe path uses the same precedence through
`MetricSpec(decision_method=..., sensitivity_methods=...)`; see
[The data model](data-model.md).

## CUPED under sequential inference

Support depends on the sequential method. The exact `AlwaysValid` e-process
refuses CUPED with `arm.adjustment.sequential_cuped` /
`plan.metric_cuped_methods`: refitting the coefficient at each look would
reweight past observations with information that was not available when they
arrived. `InferenceSpec(kind="asymptotic_mean")` supports two forms, below.

??? info "Statistical details: predictability"
    An exact e-process requires both choices made before each observation
    arrives (**predictability**) and observations that follow the registered
    sampling law. Predictability alone is not enough. The fixed-horizon fit
    uses experiment outcomes, so it is not predictable.

### The standard coefficient on the asymptotic route

`InferenceSpec(kind="asymptotic_mean")` fits the coefficient from experiment
data at each look. A method declares the source covariate and registers
`adjusted_mean`, or `adjusted_ratio_mean` for a ratio metric.

<!-- skip: next "requires a prepared unit frame" -->
```python
from increment import Analysis, AnalysisPlan, CupedCovariate, InferenceSpec, Method, MetricSpec

plan = AnalysisPlan(primary="revenue", inference=InferenceSpec(kind="asymptotic_mean"))
spec = MetricSpec(
    name="revenue",
    decision_method=Method.cuped(covariates=CupedCovariate(name="revenue_pre", kind="continuous")),
)
analysis = Analysis.from_unit_summary(
    units,
    unit="user_id",
    group="variant",
    metrics=[spec],
    plan=plan,
    experiment_id="exp",
    exposure_date="enrolled_on",
)
```

The method needs positive within-group covariate variance (a constant
covariate reports `zero_covariate_variance`) and per-unit covariate data:

- `from_unit_summary`: declare the covariate on a CUPED method.
- `from_definitions`: set `n_pre_periods > 0`. A plan binding with a CUPED
  method registers the adjusted law automatically.
- `from_unit_day_artifact`: publish each adjusted method's `cuped_preperiod`
  extension. Select its `entry.request` from
  `unit_day_artifact_extension_catalog(context)` and pass it through
  `extensions=` to `analysis.publish_unit_day_artifact`. Default publication
  (`extensions=()`) omits CUPED covariates. See the
  [artifact extension matrix](data-model.md#operation-and-extension-matrix).

Capture retains the same zero-filled pre-period total as fixed-horizon CUPED.
Unit panels and warehouse experiments with `n_pre_periods: 0` reject these
laws because they cannot supply the covariate.

??? info "Statistical details: the asymptotic adjusted law"
    Capture retains the joint per-unit vector $(Y, X)$ or $(N, D, X)$ and its
    full within-group scatter. At each look, Increment fits the same
    inverse-$n$-weighted coefficient as `fit_cuped`, centers both groups on
    the pooled covariate mean, and inverts a delta-method approximation at the
    scalar method's count-clock boundary. The gradients account for
    dependence between groups through their shared covariate mean.

    This uses the heteroskedasticity-robust asymptotic regression-adjusted
    confidence sequence of Lindon, Ham, Tingley and Bojinov (2022)
    ([arXiv:2210.08589](https://arxiv.org/abs/2210.08589)), with the
    relative-lift construction of Schmit and Miller (2022) for ratio laws
    ([paper](https://svenschmit.com/assets/pdf/code_2022_ci.pdf)).

    Although $\theta$ is not predictable, its estimation error multiplies the
    covariate imbalance. Both vanish at the law-of-the-iterated-logarithm
    rate, so their product is smaller in order than the boundary width.
    Retained state and boundary coefficients are exact rationals; the
    linearization and boundary are asymptotic.

### A pre-period coefficient on the asymptotic scalar-mean route

The asymptotic scalar-mean method also accepts a coefficient and covariate
center fixed from pre-period data before outcomes are read.
`fit_predeclared_adjustment` applies `fit_cuped` to a pre-period frame;
capture then retains the single adjusted value $Y_i - \theta (X_i - c)$
per unit. The exact Bernoulli method does not accept this adjustment.

<!-- skip: next "requires a prepared pre-period frame" -->
```python
from increment import (
    AnalysisPlan,
    CupedCovariate,
    InferenceSpec,
    Method,
    MetricSpec,
    fit_predeclared_adjustment,
)

adjustment = fit_predeclared_adjustment(
    pre_period,
    unit="user_id",
    group="variant",
    control="control",
    outcome="revenue_pre_window",
    covariate="revenue_earlier_window",
)
plan = AnalysisPlan(
    primary="revenue",
    inference=InferenceSpec(kind="asymptotic_mean", adjustments={"revenue": adjustment}),
)
spec = MetricSpec(
    name="revenue",
    decision_method=Method.cuped(covariates=CupedCovariate(name="revenue_pre", kind="continuous")),
)
```

An explicit registration uses `ScalarMeanModel.adjustment`.

!!! warning "A pre-period coefficient changes the trade-offs"
    The coefficient is not optimized for the experiment's observed contrast,
    so it reduces variance less than the in-experiment fit on the same data.
    Covariate drift between the pre-period and the experiment also creates
    first-order bias in relative lift that does not vanish with sample size:
    each adjusted group mean shifts by the same constant
    $\theta\,(\mathbb{E}[X] - c)$, which cancels in a difference but not in
    the ratio the scalar-mean method reports. It cancels at a zero-effect
    null, not at nonzero guardrail or non-inferiority margins. A nearby
    pre-period center may reduce drift; it does not guarantee its absence.

Winsorization on this route must also be predictable: a fixed threshold, or
one chosen from pre-period data, is; a percentile estimated from accumulated
data is not and raises `sequential.transform.unpredictable`. Clipped
observations must still follow the registered sampling law.

## Encouragement designs: CUPED on LATE

Low compliance makes encouragement tests costly: required sample size grows
by $1/\text{compliance}^2$. CUPED can help by adjusting the
intention-to-treat (ITT) outcome and the additive local average treatment
effect (LATE). Under an `Encouragement` design,
`variance_reduction="cuped"` computes:

$$
\hat\tau_{\text{CUPED}}
= \frac{\tilde{A}}{B}
= \frac{(\bar y_t - \theta(\bar x_t - \mu_X)) - (\bar y_c - \theta(\bar x_c - \mu_X))}
       {\bar d_t - \bar d_c}
$$

Decision and sensitivity scalar methods may name different covariates. Each
ITT and LATE fit uses that method's own prepared covariate and its joint
moments with outcome and uptake. Selected secondary intervals reuse the same
method-specific evidence at the family's corrected level. Multicovariate CUPED
is not supported under encouragement, including as a sensitivity method; such
a request refuses with `estimation.engine.cuped_linear_encouragement`.

<!-- invisible-code-block: python
rng = np.random.default_rng(3)
m = 2_000
revenue_pre = rng.gamma(shape=2.0, scale=15.0, size=m)
arm = np.where(rng.random(m) < 0.5, "treatment", "control")
clicked = np.where(arm == "treatment", rng.random(m) < 0.6, rng.random(m) < 0.1).astype(int)
df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(m)],
        "variant": arm,
        "revenue": 5.0 + 0.8 * revenue_pre + 2.0 * clicked + rng.normal(0, 8, m),
        "revenue_pre": revenue_pre,
        "clicked": clicked,
    }
)
-->

```python
import increment as inc

analysis = inc.Analysis.from_unit_summary(
    df,  # one row per unit, carrying the uptake column and the pre-period covariate
    unit="user_id",
    group="variant",
    metrics=[
        inc.MetricSpec(
            name="revenue",
            decision_method=inc.Method.cuped(
                covariates=inc.CupedCovariate(name="revenue_pre", kind="continuous")
            ),
        )
    ],
    design=inc.Encouragement(
        control_group="control",
        uptake=inc.UptakeSpec(fact="clicked"),
        exclusion_restriction=inc.ExclusionRestriction(
            acknowledged=True,
            justification="The prompt only changes behavior through the click.",
        ),
    ),
)
results = analysis.run(
    estimands=("itt", "compliance", "late"),
)
```

Three details matter:

- **Only the numerator is adjusted.** The first stage $B$ stays unadjusted,
  so the weak-instrument check (`min_first_stage_z`) and which groups
  receive a LATE row are unchanged; compliance rows keep `method="unadjusted"`.
- **LATE and ITT share one adjustment:** the same pairwise pooled $\theta$
  and adjusted group means, so
  `cuped LATE == cuped ITT abs_diff / compliance lift` exactly.
- **Complier-relative LATE is withheld.** It needs $\sum x y d$ moments
  that are not stored; the additive row's `note` says so, and no unadjusted
  result is reported under a CUPED label.

### CUPED can *cost* power on a LATE

CUPED can widen a LATE interval when the covariate strongly predicts uptake
but predicts the outcome weakly once uptake is accounted for, and the effect
is large. The standard error includes that inflation, so the loss is in power,
not in coverage. Increment compares both variances and adds a note when
adjustment increases variance:

> `CUPED INFLATED this LATE's variance versus no adjustment (SE 1.59x the
> unadjusted SE): the covariate drives uptake more than it predicts the
> outcome net of uptake -- pass variance_reduction='none' for metric 'rev'
> to recover the tighter interval`

!!! warning "The note is not an automatic fallback"
    Choosing the narrower interval after seeing the data is data-dependent
    method selection; the fixed procedure's coverage guarantee would no
    longer apply.

??? info "Statistical details: why a LATE can lose precision"
    For a Wald ratio, the variance-optimal coefficient is
    $c^\star = \theta_Y - \tau\theta_D$, where
    $\theta_D = \text{Cov}(D, X)/\text{Var}(X)$. Adjusting only the numerator
    uses $\theta_Y$ instead. The difference adds
    $(\tau\theta_D)^2\text{Var}(X)$ to the influence-function variance, which
    is largest when the covariate predicts uptake, predicts the outcome weakly
    net of uptake, and $\tau$ is large.

## Clustered encouragement designs

An encouragement experiment can also be clustered: stores receive the
assignment, while users choose whether to click a banner. Declare both
`cluster` and `design=Encouragement(...)`:

<!-- invisible-code-block: python
rng = np.random.default_rng(5)
k, m = 30, 20  # stores per arm, users per store
store_effect = rng.normal(0.0, 1.5, 2 * k)  # shared per-store shock (ICC > 0)
store = np.repeat(np.arange(2 * k), m)
arm = np.where(store % 2 == 0, "control", "treatment")[store]
took = np.where(arm == "treatment", rng.random(k * m * 2) < 0.55, 0).astype(int)
revenue = 20 + store_effect[store] + 6.0 * took + rng.normal(0, 5, 2 * k * m)

df = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(2 * k * m)],
        "variant": arm,
        "store_id": [f"s{s}" for s in store],
        "revenue": revenue,
        "took": took,
    }
)
-->

```python
import increment as inc

analysis = inc.Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    metrics={"revenue": "mean"},
    design=inc.Encouragement(
        control_group="control",
        uptake=inc.UptakeSpec(fact="took"),
        exclusion_restriction=inc.ExclusionRestriction(
            acknowledged=True,
            justification="The banner only changes behavior through the click.",
        ),
        one_sided=True,
    ),
    cluster="store_id",  # the randomization-grain column
)
results = analysis.run(estimands=("itt", "compliance", "late"))
(late,) = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
print(f"LATE={late.lift.value:+.2f}  CI=({late.lift.lb:+.2f}, {late.lift.ub:+.2f})")
print(f"K={late.n_clusters}, dof={late.dof:g}")
```

<!-- output -->
```text
LATE=+5.63  CI=(+3.90, +7.36)
K=60, dof=56.6809
```

Clustering leaves point estimates unchanged and adjusts their standard
errors and reference distribution. Fewer than 40 total clusters triggers a
`RuntimeWarning` about over-rejection risk; contrasts with at least two
clusters per enrolled group still run with a qualified working $t$
reference, each pairwise contrast counting only its two groups' clusters.

The clustered encouragement path rejects or withholds:

- **CUPED:** the clustered reduction does not retain the per-unit covariate
  moments. Combining `cluster=` with `variance_reduction="cuped"` raises
  `CapabilityError`, as it does in `estimate_lift`.
- **Complier-relative LATE:** the reduction lacks the cluster-level
  `sum(y**2*d)` / `sum(x*y*d)` moments; the additive row's `note` says so.
- **Ratio metrics:** encouragement needs cluster sizes in the same
  denominator fields ratio metrics use, so they cannot share a row.
  Clustered ratio metrics without an encouragement design are supported.
- **Informative `prior` or sequential `inference`:** this path uses a $t$
  reference, not the conjugate Normal-Normal calculation or a sequential
  boundary.

??? info "Statistical details: the clustered LATE"
    ITT and additive LATE use a cluster-robust ratio delta method. `mean_y`
    and `mean_d` are ratios of cluster totals, `sum(g_j)/sum(m_j)` and
    `sum(d_j)/sum(m_j)`, and the Wald ratio uses the matching cluster-level
    covariance between these numerators. Compliance reports the first-stage
    effect on the absolute scale, without the log-relative-risk calculation
    that assumes unit-level binary uptake.

## When CUPED does nothing

CUPED helps most when the covariate predicts the outcome and most units have
pre-period history. Watch for:

- **Little correlation:** with $\rho \approx 0$, $\theta \approx 0$, so the
  adjustment does little. This is common in new-user experiments, where
  zero-filled history predicts little.
- **Constant covariate:** zero variance makes $\theta$ undefined. The
  estimator raises a `ValueError` rather than reporting an unadjusted
  estimate under a CUPED label.
- **Quantile metrics:** these reject CUPED covariates because a quantile is not
  a mean of per-unit values. [Ratio metrics](#ratio-metrics) adjust both
  components with one covariate.
- **Covariates that mostly predict uptake:** an adjusted LATE can have a
  wider interval; see [CUPED can cost power on a LATE](#cuped-can-cost-power-on-a-late).

Use the metric's pre-period value when most units have history, but do not
assume every adjustment will improve precision: relative-lift intervals can
widen even with a fixed coefficient, and the LATE adjustment can lose
precision.

??? info "Statistical details: the CUPED coefficient"
    For outcome $Y$ and pre-period covariate $X$, CUPED adjusts each group's
    mean:

    $$
    \hat\mu_a = \bar{Y}_a - \theta\,(\bar{X}_a - \bar{X}), \qquad
    \theta = \frac{\operatorname{Cov}(Y_t, X_t)/n_t + \operatorname{Cov}(Y_c, X_c)/n_c}
                  {\operatorname{Var}(X_t)/n_t + \operatorname{Var}(X_c)/n_c}
    $$

## Mixed numeric and categorical covariates

`Method.cuped_multicovariate` fits a fixed-horizon linear adjustment. Mean,
conversion, and ratio metrics use HC2 uncertainty, including fitted-slope and
pooled-covariate-anchor uncertainty, with a Bell–McCaffrey Student-t
reference. This is an explicit procedure, not a finite-sample or sequential
guarantee.

### Combining scalar and mixed CUPED

Both procedures can be emitted for the same metric. Declare the covariate on
each method; they may share a source binding while choosing independent
missing-value policies:

```python
history = CupedCovariate(name="pre_revenue", kind="continuous")
scalar = Method.cuped(covariates=history)
mixed = Method.cuped_multicovariate(name="linear", covariates=history)
```

Assign one estimator as the decision method and the other as a sensitivity
method. `from_definitions` can use `n_pre_periods > 0` for scalar CUPED's
implicit pre-period binding, while the multicovariate method uses its explicit
`covariates` bindings. To adjust the same historical signal, those per-unit
values must match. Publication for `from_unit_day_artifact` must include the
`cuped_preperiod` extension required by scalar CUPED. The four-ingress witness
in `test_warehouse_cuped_parity.py` exercises both methods on the same
per-unit historical revenue data; this is runtime compatibility, not coverage
certification.

### A mixed-covariate dataframe recipe

The following complete example uses a per-user revenue mean and a revenue/orders
ratio. It supplies country, device, and new-versus-returning-user categories plus
numeric pre-period revenue. Twelve percent of the synthetic users have no
pre-period revenue; `missing="impute-indicator"` fills those values with the
pooled observed mean and adds a missingness indicator. Categorical covariates
must be strings (encode a Boolean as labels such as `"new"` and `"returning"`).

```python
import numpy as np
import polars as pl
from increment import Analysis, Method, MetricSpec
from increment.semantics.models import CupedCovariate

rng = np.random.default_rng(20261013)
n = 4_000
country = rng.choice(["US", "CA", "GB"], n, p=[0.55, 0.25, 0.20])
device = rng.choice(["web", "ios", "android"], n, p=[0.45, 0.35, 0.20])
new_user = np.where(rng.random(n) < 0.30, "new", "returning")
pre_revenue = rng.gamma(2.2, 12.0, n)
pre_revenue[rng.random(n) < 0.12] = np.nan
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
pre_for_outcomes = np.nan_to_num(pre_revenue, nan=np.nanmean(pre_revenue))

country_effect = np.select([country == "CA", country == "GB"], [2.0, -1.0], default=0.0)
device_effect = np.select([device == "ios", device == "android"], [1.5, -0.5], default=0.0)
revenue = (
    4.0
    + 0.65 * pre_for_outcomes
    + country_effect
    + device_effect
    + 3.0 * (new_user == "new")
    + 1.8 * (variant == "treatment")
    + rng.normal(0, 9, n)
)
orders = np.maximum(
    2.0
    + 0.015 * pre_for_outcomes
    + 0.12 * (device == "ios")
    + 0.08 * (variant == "treatment")
    + rng.normal(0, 0.65, n),
    0.1,
)

units = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(n)],
        "variant": variant,
        "revenue": revenue,
        "orders": orders,
        "country": country,
        "device": device,
        "new_user": new_user,
        "pre_revenue": pre_revenue,
    }
)
covariates = [
    CupedCovariate(name="country", kind="categorical"),
    CupedCovariate(name="device", kind="categorical"),
    CupedCovariate(name="new_user", kind="categorical"),
    CupedCovariate(
        name="pre_revenue",
        kind="continuous",
        missing="impute-indicator",
    ),
]
methods = [
    MetricSpec(name="revenue", type="mean"),
    MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
    ),
]
analysis = Analysis.from_unit_summary(
    units,
    unit="user_id",
    group="variant",
    control="control",
    metrics=methods,
)
results = analysis.run(
    decision_method=Method.unadjusted(name="unadjusted"),
    sensitivity_methods=(Method.cuped_multicovariate(name="linear", covariates=covariates),),
)

for result in results:
    lift = result.lift
    print(
        result.metric,
        result.method,
        "lift=",
        None if lift is None else round(lift.value, 6),
        "CI=",
        None if lift is None else (round(lift.lb, 6), round(lift.ub, 6)),
        "failure=",
        result.failure_code,
    )
```

Observed output from a fresh run:

```text
revenue unadjusted lift= 0.082946 CI= (0.042652, 0.124797) failure= None
revenue linear lift= 0.068045 CI= (0.042767, 0.093936) failure= None
revenue_per_order unadjusted lift= 0.049966 CI= (0.011863, 0.089504) failure= None
revenue_per_order linear lift= 0.039378 CI= (0.009775, 0.069849) failure= None
```

The lift is relative: for example, `0.068045` is a 6.8045% estimated increase
in mean revenue, and its confidence interval is a relative-lift interval.
Here the adjusted revenue interval is narrower than the unadjusted one, but
that is an observed property of this synthetic draw, not a guarantee. The ratio
rows target the ratio of arm-level mean revenues to arm-level mean orders, not
the mean of users' individual revenue-per-order values. Both metrics share the
same covariate set; the ratio adjustment jointly accounts for its numerator and
denominator.

This is intentionally not scalar CUPED: `cuped_multicovariate` fits one
multivariable adjustment to the per-unit outcome vector and uses HC2
uncertainty (including fitted-slope and pooled-anchor uncertainty) with a
Bell–McCaffrey Student-t approximation. Scalar `cuped` keeps its
Welch–Satterthwaite route. Neither route inherits the other's statistical
properties.

The fixed-horizon M1 support region below is a hard boundary, not a warning or
an automatic fallback. For a concrete small-sample failure, a balanced
40-unit synthetic design with these four covariates returned
`estimation.cuped_linear.support_region` with
`violations=(('leverage_ratio', 0.25, 0.2),)`: five encoded columns divided
by 20 units in the smaller arm exceeds the 0.20 cap. Duplicating that
balanced design to 80 units gave `leverage_ratio=0.125` and a usable estimate
in a fresh run (`lift=0.120882`, 95% CI `[0.100753, 0.141379]`; values are
specific to this synthetic witness, not coverage evidence). In real
experiments the remedy is more independent units
with adequate category support; do not duplicate observations as if they
were new users. A failed decision fit stays unavailable rather than silently
switching to unadjusted inference.

For ratio CUPED, M1 additionally requires each raw arm denominator's
`sqrt(var_den / n) / mean_den` to be at most 0.15. Heavy-tailed denominators
at about 100 units per arm are unsupported when this screen is exceeded; a
larger sample may pass the screen, but passing it is not evidence of
calibrated coverage.

This same-experiment estimator does not train an external prediction model:
external-prediction training (tracker #72) and longitudinal dependence
(tracker #80) are not implemented. A preregistered coverage campaign exists
but has not been run, so coverage is not established. Its current commands
and status are recorded in [External validation coverage](../validation.md#mixed-covariate-linear-cuped).


Fits must lie in support region **M1**: at most 203 encoded covariates,
at most 200 levels in any categorical covariate, encoded width divided by
the smallest arm size at most 0.20, and maximum unit leverage at most 0.45.
Ratio fits additionally require each raw arm denominator's relative
standard error, `sqrt(var_den / n) / mean_den`, at most 0.15; adjustment
does not substitute its residual variance for this check. An arm whose raw
denominator mean is not positive cannot form the ratio estimand at all: it
is withheld from the fit and reported as its own refused cell under the
same `estimation.variance.ratio_moments_nonpositive_denominator_mean`
code an unadjusted row carries, while its siblings keep their adjusted
evidence; a control arm in that state refuses every contrast. Full rank,
positive residual degrees of freedom, and at least two observations per
categorical level in every arm are also required.

An out-of-region fit emits an unavailable cell with
`estimation.cuped_linear.support_region` and its retained observable
violations, rather than an unadjusted substitute. Each violation is
`(observable, value, bound)`; an observable that is not finite is kept as
its text form (`"inf"`, `"nan"`) so the refusal survives JSON replay. The
decision cell keeps this refusal whatever sensitivity methods accompany it,
and a call-time `cuped_multicovariate` override on a metric without declared
cells fit separately and may therefore be unavailable even when the
whole-population fit is supported. Assigned and triggered results retain
their own populations; aggregate and per-unit evidence share one source
snapshot, including the resolved fact source of each breakout.

Relative and absolute projections have separate availability. A usable
relative estimate survives an absolute-projection refusal, whose exact
code and context are retained in `absolute_error_code` and
`absolute_error_context`. Absolute-margin guardrails may retain sampling
inference when a nonpositive adjusted mean prevents relative lift.
The relative projection's exact refusal is likewise retained in
`relative_error_code` and `relative_error_context`, without labeling the
usable absolute projection as failed. `estimates_to_readout` forwards all
four projection fields alongside the usable estimate.
Declared priors do not change sampling inference; a posterior unsupported
by the retained HC2/BM statistics is explicitly unavailable.

## The math

    The coefficient minimizes the variance of the additive contrast
    $\hat\mu_t - \hat\mu_c$ when the group moments are treated as known. For
    independent groups, that variance is

    $$
    \sum_a [\operatorname{Var}(Y_a) - 2\theta\operatorname{Cov}(Y_a, X_a) + \theta^2\operatorname{Var}(X_a)]/n_a
    $$

    A pooled regression slope weights the same moments by $n_a - 1$,
    targeting a different quantity. With unequal allocation and different
    group slopes, it can be worse than no adjustment; this allocation-aware
    weighting addresses that case. The two coefficients agree with equal
    allocation or matching within-group slopes. Fitting a separate $\theta$
    for each group would reintroduce bias.

    Because $X$ is independent of assignment, adjustment preserves the
    target difference. When both groups share a slope, the approximate
    variance-reduction factor is:

    $$
    1 - \rho^2, \qquad \rho = \operatorname{Corr}(Y, X)
    $$

    A correlation of $\rho = 0.7$ roughly halves the variance.
    The corresponding interval-width factor is $\sqrt{1 - \rho^2}$.

    !!! note "A planning approximation, not the interval calculation"
        The $(1 - \rho^2)$ factor helps plan an experiment. Reported intervals
        use each group's adjusted mean and full quadratic variance calculation,
        not this correlation shortcut.

    !!! note "What the interval assumes"
        $\theta$ targets the additive contrast, not relative lift.
        Relative lift is nonlinear, and its log-scale standard error accounts
        for the shared pooled mean $\bar{X}$. That mean cancels in a
        difference but not in a ratio, so the adjusted group means cannot
        be treated as independent; its variance optimum differs, and
        different group slopes or means make widening more likely.

        The interval calculation treats the fitted $\theta$ as fixed.
        Its guarantee is therefore asymptotic; there is no finite-sample
        guarantee that fitting the adjustment cannot widen an interval. A
        fixed slope with a valid pre-period covariate is unbiased; a slope
        fitted from the same outcomes can correlate with the sample's
        covariate imbalance, a finite-sample bias that vanishes
        asymptotically under the usual regularity conditions.

    See the [API reference](../api.md) for `MetricSpec` and `Method`. Planning
    an experiment around an expected $\rho$ is covered in the
    [power analysis guide](power-analysis.md).

## Next step

Plan the expected precision gain with [Power analysis](power-analysis.md).

## References and assumptions

CUPED follows Deng, Xu, Kohavi, and Walker,
[“Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data”](https://doi.org/10.1145/2433396.2433413).
Use a covariate measured before treatment. Correlation with the outcome
provides the potential precision gain; the adjustment preserves treatment
assignment and the estimand.
