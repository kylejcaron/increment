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

Declare the pre-period column with `MetricSpec(covariate=...)`, then request
the adjustment with `variance_reduction="cuped"` on a decision or sensitivity
`Method`. Declaring the covariate alone changes nothing: `run()` keeps its
declared or default decision method. When the effective design, metric, and
inference support CUPED, `run()` emits `analysis.cuped.covariate_unused` only
if an omitted decision method defaults to unadjusted and no CUPED sensitivity
method is selected. An explicitly chosen unadjusted decision does not warn.

```python
import numpy as np
import polars as pl
from increment import Analysis, Method, MetricSpec

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
    metrics=[MetricSpec(name="revenue", type="mean", covariate="pre_revenue")],
).run(
    decision_method=Method(name="unadjusted"),
    sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
)

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

`variance_reduction="cuped"` enables the adjustment; `name` only labels
the result. This example keeps the unadjusted decision method and reports
CUPED alongside it as a sensitivity analysis. See
[Reading results](reading-results.md) for row interpretation and
[Multiplicity](multiplicity.md) for the result population and status table.

!!! warning "Panel covariates must be constant per unit"
    Under fixed-horizon inference, `from_unit_panel` supports CUPED for
    unwindowed mean and ratio metrics when `covariate=` is constant across
    each unit's rows; varying values raise
    `frame.frame_panel.unit_covariate_varies`. Windowed and retention metrics
    reject panel covariates (`frame.validation.from_unit_panel`): compute a
    non-retention windowed metric per unit upstream and declare it on
    `from_unit_summary`. No frame source serves a per-unit retention value;
    remove the covariate to run retention without CUPED. For sequential
    CUPED, use the paths [below](#cuped-under-sequential-inference).

### Missing covariates

By default, a missing covariate is filled with the pooled mean of the
observed covariates, and a `UserWarning` reports how many values were filled.
The fill uses neither group assignment nor outcomes; under randomization,
this preserves the covariate's independence from assignment. Filled values
add no precision.
Use `MetricSpec(covariate_missing="error")` to reject missing covariates, or
`covariate_missing="zero"` when missing means no pre-period activity.

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
    covariate="pre_revenue_per_order",
)
```

!!! warning "One covariate column, two slopes"
    Separate covariates for the numerator and denominator are not supported
    on any path: the stored moments contain only one covariate and its cross
    moments. If each component needs its own covariate, analyze them as two
    mean metrics instead. That reports component effects, not the ratio effect.

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
path. A CUPED request with `n_pre_periods == 0` raises `ValueError`.
Call-wide methods override per-metric bindings.

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
data at each look. A metric with `covariate=` and a CUPED method registers
`adjusted_mean`, or `adjusted_ratio_mean` for a ratio metric.

<!-- skip: next "requires a prepared unit frame" -->
```python
from increment import Analysis, AnalysisPlan, InferenceSpec, Method, MetricSpec

plan = AnalysisPlan(primary="revenue", inference=InferenceSpec(kind="asymptotic_mean"))
spec = MetricSpec(
    name="revenue",
    covariate="revenue_pre",
    decision_method=Method(name="cuped", variance_reduction="cuped"),
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

- `from_unit_summary`: supply `MetricSpec(covariate=...)`.
- `from_definitions`: set `n_pre_periods > 0`. A plan binding with a CUPED
  method registers the adjusted law automatically.
- `from_unit_day_artifact`: publish each adjusted metric's `cuped_preperiod`
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
    Analysis,
    AnalysisPlan,
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
    covariate="revenue_pre",
    decision_method=Method(name="cuped", variance_reduction="cuped"),
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
    metrics=[inc.MetricSpec(name="revenue", covariate="revenue_pre")],
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
    decision_method=inc.Method(name="cuped", variance_reduction="cuped"),
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
- **Quantile metrics:** these reject `covariate=` because a quantile is not
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

    Here $\bar{X}$ is the covariate mean pooled across both groups. All other
    moments are sample moments computed within each group.

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
