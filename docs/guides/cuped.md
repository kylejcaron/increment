# CUPED variance reduction for A/B tests

CUPED uses information measured before an experiment to remove some of the
noise in its results. Try it when that information predicts the outcome—for
example, last month's spending predicts this month's revenue.

!!! warning "A covariate does not promise a narrower interval"
    CUPED can widen an interval. Fitting its coefficient from experiment
    outcomes gives asymptotic, not finite-sample, guarantees. Declare the
    decision method before seeing results; do not choose whichever interval
    turns out narrower.

## On the dataframe path

Supply one row per unit with its outcome and pre-experiment measurement
(the **covariate**). This complete example compares unadjusted inference with
scalar CUPED as a sensitivity analysis:

```python
import numpy as np
import polars as pl
import increment

rng = np.random.default_rng(11)
n = 20_000
pre_revenue = rng.gamma(shape=2.0, scale=15.0, size=n)
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
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
experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        increment.MetricSpec(
            name="revenue",
            type="mean",
            sensitivity_methods=(
                increment.Method.cuped(
                    covariates=increment.CupedCovariate(
                        name="pre_revenue", kind="continuous", missing="impute"
                    )
                ),
            ),
        )
    ],
)
results = increment.run(
    experiment, reader, decision_method=increment.Method.unadjusted(name="unadjusted")
).readout()
for r in results:
    width = r.lift.ub - r.lift.lb
    print(f"{r.method:>10}: lift={r.lift.value:+.2%}  CI width={width:.2%}")
```

<!-- output -->
```text
unadjusted: lift=+2.01%  CI width=3.79%
     cuped: lift=+2.15%  CI width=1.77%
```

Both estimates suggest about 2% higher revenue. CUPED's interval is less
than half as wide in this synthetic example; real gains depend on the
covariate. These are sampling confidence intervals, not credible intervals.

Bind `Method.cuped(...)` as `decision_method` to make it the planned decision
procedure. Merely supplying a covariate does not change the method.
`covariates="pre_revenue"` is shorthand for a continuous covariate with
`missing="refuse"`; text or date values raise
`frame.validation.covariate_not_numeric`.

### Measuring variance reduction

Compare each metric and treatment arm on the **same absolute
treatment-minus-control difference**, using its reported `abs_se`:

$$
\text{variance reduction}
= 1 - \left(\frac{\mathrm{SE}_{\mathrm{CUPED,absolute}}}
                  {\mathrm{SE}_{\mathrm{unadjusted,absolute}}}\right)^2.
$$

Positive means lower estimated variance; negative means higher. Missing SEs
or a zero unadjusted SE make the comparison unavailable, not zero gain.
A zero adjusted SE with a positive unadjusted SE means 100% estimated reduction.

Relative-lift interval widths are not a substitute: they also depend on the
confidence level, reference distribution and scale transformation. A one-sided
guardrail may have an open endpoint and no finite width. The
[CUPED notebook](../examples/cuped.md) compares absolute variance for continuous,
conversion and retention outcomes; a striking change in interval appearance
need not mean a large variance reduction.

### Missing covariates

Choose a policy on each method's covariate:

| Policy | Preparation |
|---|---|
| `refuse` (default) | Withhold that method's cell if a value is missing. |
| `zero` | Treat missing as no pre-period activity. |
| `impute` | Fill with the pooled observed mean and warn with counts. |

Use zero only when it has that meaning. Either fill needs the filled
covariate to have the same expected mean across arms. Equal missing shares
alone do not establish this; a join that loses high values in one arm and
low values in the other can bias the adjustment. Mean filling can tolerate
different missing shares when missingness is unrelated to value size.

A missing value under `refuse` gives
`frame.validation.metric_covariate_column`; an all-missing `impute` input gives
`frame.validation.metric_covariate_impute_all_null`. Healthy methods and
metrics still return. An absent column instead rejects the request before
reading data. See the [full missingness rules](../reference/variance-reduction.md#missing-covariates),
including the population used for imputation.

## Choose the adjustment

| Situation | Method or next step |
|---|---|
| One continuous covariate | `Method.cuped` |
| One prediction scored by a frozen external model | Follow the [CUPAC recipe](cupac.md) |
| Several covariates, categories or missingness indicators | `Method.cuped_multicovariate` |
| Warehouse pre-period outcome | Set `n_pre_periods` and request CUPED |
| Repeated decisions | Check the [sequential restrictions](#cuped-under-sequential-inference) first |

Multicovariate CUPED is an explicit fixed-horizon regression procedure using
HC2 uncertainty and a Bell–McCaffrey t approximation. It is not scalar CUPED
with more columns, and has a hard [support region](../reference/variance-reduction.md#mixed-numeric-and-categorical-covariates),
not a finite-sample guarantee. Backend choice does not choose the procedure.

### Ratio metrics

Scalar CUPED adjusts numerator and denominator separately with one covariate,
then forms the ratio. The target remains $E[N]/E[D]$, not a mean of unit
ratios. Warehouse and artifact inputs use the pre-period **numerator**.
See the [ratio formulas](../reference/variance-reduction.md#adjusting-a-ratio)
and [source rules](../reference/variance-reduction.md#ratio-covariate-sources).

## In YAML definitions

`n_pre_periods` supplies each metric's pre-period total; it does not request
adjustment. For a ratio it totals the numerator. The window ends at each
unit's first exposure timestamp, and no pre-period activity means zero.
Exposure timestamps must be recorded the same way in every arm.

### Per-metric method bindings

Bind CUPED to a metric's decision or sensitivity method:

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
        - avg_session_duration
    n_pre_periods: 14
    control_group: C
```

Here only `purchase_rate` gets both methods. Call-wide methods passed to
`run()` override per-metric bindings. For named properties, measure bindings,
artifacts and moments, use the [source contracts](../reference/variance-reduction.md#in-yaml-definitions).
Fixed-horizon panels require covariates constant within each unit and support
only unwindowed mean/ratio CUPED; see [panel restrictions](../reference/variance-reduction.md#dataframe-covariates).

## When CUPED does nothing

- Little correlation, common for new users with no history, gives little gain.
- A covariate constant in every arm gives an unavailable CUPED cell with
  `estimation.cuped.covariate_zero_variance`, not a relabeled unadjusted result.
  If only one arm is constant, the other supplies the slope; investigate why.
- Quantiles reject CUPED covariates: a quantile is not a mean of unit values.
- An encouragement covariate that mostly predicts uptake can widen LATE's interval.

## When CUPED misleads: a covariate that differs by arm

Measure the covariate before treatment and identically across arms. Lost
values, arm-specific rescaling or a window overlapping treatment can bias
the adjusted effect. This bias does not shrink with sample size.

`estimation.cuped.covariate_imbalance` asks you to investigate how the
covariate was built. It changes neither the estimate nor the decision.
Chance imbalance can trigger it, and a small systematic shift can escape
it: passing the check does **not** certify the covariate. Check assignment
with [SRM diagnostics](srm-diagnostics.md) too.
The [balance-test reference](../reference/variance-reduction.md#when-cuped-misleads-a-covariate-that-differs-by-arm)
defines its thresholds, reported shifts and sequential qualifications.

## CUPED under sequential inference

Exact `AlwaysValid` inference rejects CUPED. The `asymptotic_mean` method
supports specific refitted and predeclared adjustments, with asymptotic
guarantees only. A fixed pre-period coefficient can bias nonzero relative
effects if its center drifts. Use the [registration and adjustment recipes](../reference/variance-reduction.md#cuped-under-sequential-inference),
not repeated fixed-horizon runs.

<a id="the-standard-coefficient-on-the-asymptotic-route"></a>

- [Refitting the coefficient at each look](../reference/variance-reduction.md#the-standard-coefficient-on-the-asymptotic-route).

<a id="a-pre-period-coefficient-on-the-asymptotic-scalar-mean-route"></a>

- [Pre-period coefficients and drift bias](../reference/variance-reduction.md#a-pre-period-coefficient-on-the-asymptotic-scalar-mean-route).

## Encouragement designs: CUPED on LATE

CUPED adjusts ITT and additive LATE's numerator, not compliance. It withholds
complier-relative LATE; multicovariate CUPED is unsupported. Use the complete
[encouragement recipe](../reference/variance-reduction.md#encouragement-designs-cuped-on-late).

### CUPED can *cost* power on a LATE

A covariate that predicts uptake more than the outcome net of uptake can
increase LATE's variance. The reported SE includes this inflation. Do not
switch to the narrower method after seeing results; see
[the explanation and advisory](../reference/variance-reduction.md#cuped-can-cost-power-on-a-late).

## Clustered encouragement designs

Clustered encouragement rejects CUPED, ratio metrics, informative priors and
sequential inference, and withholds complier-relative LATE. Its unadjusted
additive LATE uses cluster-robust uncertainty; see
[the complete clustered example and limits](../reference/variance-reduction.md#clustered-encouragement-designs).

## Mixed numeric and categorical covariates

For a complete mean/ratio example, category encoding and support conditions,
use the [mixed-covariate reference](../reference/variance-reduction.md#mixed-numeric-and-categorical-covariates).
Unsupported fits stay unavailable; they never silently switch to unadjusted.

<a id="comparing-scalar-and-multicovariate-procedures"></a>

- [Compare scalar and multicovariate fits](../reference/variance-reduction.md#comparing-scalar-and-multicovariate-procedures).

<a id="warehouse-magnitude-range"></a>

- [Warehouse magnitude limits](../reference/variance-reduction.md#warehouse-magnitude-range).

<a id="combining-scalar-and-mixed-cuped"></a>

- [Declare both methods](../reference/variance-reduction.md#combining-scalar-and-mixed-cuped).

<a id="a-mixed-covariate-dataframe-recipe"></a>

- [Run the mixed-covariate example](../reference/variance-reduction.md#a-mixed-covariate-dataframe-recipe).

<a id="joint-categorical-evidence"></a>

- [Joint categorical counts and warehouse disclosure guards](../reference/variance-reduction.md#joint-categorical-evidence).

<a id="portable-cuped-evidence"></a>

- [Export and replay CUPED evidence](../reference/variance-reduction.md#portable-cuped-evidence).

## The math

The [coefficient and variance derivation](../reference/variance-reduction.md#the-math)
explains allocation weighting, shared-mean uncertainty and why relative
intervals can widen.

## Next step

Plan an expected precision gain with [Power analysis](power-analysis.md).

## References and assumptions

CUPED follows Deng, Xu, Kohavi, and Walker,
[“Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data”](https://doi.org/10.1145/2433396.2433413).
Pre-treatment measurement independent of assignment preserves the target
comparison; correlation with the outcome supplies the potential gain.
