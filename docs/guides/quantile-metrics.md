# Quantile metrics

Use a quantile for a distributional question such as p90 latency, not a mean
effect. Supply real per-unit values through a supported dataframe, warehouse
or reopened artifact source. Additive moments cannot reconstruct them.
Quantile results report relative lift only.

## Declaring a quantile metric

```python
import numpy as np
import polars as pl
import increment

rng = np.random.default_rng(12)
n = 2000
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
latency = np.exp(rng.normal(4.5, 0.5, n)) + np.where(variant == "treatment", -15.0, 0.0)
latency = np.clip(latency, 1.0, None)
df = pl.DataFrame(
    {"user_id": [f"u{i}" for i in range(n)], "variant": variant, "latency_ms": latency}
)

experiment, reader = increment.from_frame(
    df,
    shape="summary",
    unit="user_id",
    group="variant",
    control="control",
    metrics=[increment.MetricSpec(name="latency_ms", type="quantile", quantile=0.9)],
)
results = increment.run(experiment, reader).readout()
for r in results:
    print(f"p90 relative lift: {r.lift.value:+.2%} [{r.lift.lb:+.2%}, {r.lift.ub:+.2%}]")
```

<!-- output -->
```text
p90 relative lift: -7.94% [-13.63%, -1.87%]
```

This synthetic p90 estimate is about 8% lower under treatment. The reported
interval is for the **relative contrast**, not an interval containing 90% of
individual latencies. Set `0 < q < 1`. Quantiles do not support `window_days`
or CUPED covariates; the YAML type is `QuantileMetric`.

## The Woodruff order-statistic bracket, and tied values

Each arm starts with a distribution-free order-statistic bracket
`[Y_(a), Y_(b)]` at the requested alpha (Woodruff 1952). Its derived standard
error feeds a two-arm relative-lift interval. **That combined interval is
asymptotic, not distribution-free exact.** Continuous observations do not
guarantee the sample quantile is the bracket’s log midpoint.

Ties are accepted. An arm whose 95% bracket spans fewer than 12 distinct
repeated values can need a wider symmetric interval to contain its bracket.
Untied brackets and resolved arms retain the classical Woodruff half-width.
Every widened row records its width ratio in `note`.

!!! warning "Rounding can leave wide uncertainty"
    Coarse rounding and counts produced widths 1.1–3 times the classical
    width in measured examples, versus about 1.00–1.05 for incidental ties.
    There is no upper bound: a bracket collapsed onto one value has zero
    classical width, while the reported width stays positive. If positive
    spread cannot be established, the numerical backstop is
    `estimation.quantile.degenerate_spread_order`, not an ordinary ties refusal.

??? info "How bracket ranks are certified"
    Certified logarithms enclose candidate-rank mass; bounded summation and
    a remainder bound enclose its binomial tail. Unresolved comparisons
    round outward. This does not assume a fixed SciPy CDF accuracy allowance
    or change the sampling assumptions. Classical symmetric standard errors
    are an asymptotic approximation, not an exact recentering of every bracket.

## Feasibility floor per (q, alpha)

Each `(q, alpha)` pair has a minimum per-arm sample size. The floor rises as
`q` approaches 0 or 1 and as `alpha` shrinks. If the sample is too small,
Increment reports `n`, `q`, `level`, and the required `n_min`; collect more
units or choose a less extreme quantile.

## Alpha-dependent interval, alpha-independent p-value

The bracket uses the caller's `alpha`, so the same quantile's standard error
and interval change with its multiplicity budget, even before BH selection.
A fixed-reference-alpha standard error rescaled to other alphas failed
coverage. The p-value is different: it inverts the same construction (the
smallest alpha whose interval excludes the null) and depends only on the data.
Do not compare intervals or standard errors across analyses with different
alpha budgets. See [the alpha-dependent interval limitation](../limitations.md#a-quantiles-reported-interval-depends-on-the-alpha-you-asked-for).
The row’s p-value inverts this same interval: `p_value() <= alpha` exactly
when the interval at that alpha excludes zero.

## Sequential inference is refused

Fixed-horizon quantile brackets do not supply a time-uniform process.
Sequential requests refuse before outcomes with `sequential.route.unsupported`;
a quantile reaching estimation under explicit registration refuses with
`arm.metric.quantile_sequential`. Use one planned fixed-horizon analysis,
not repeated looks.

## Unsupported combinations

Quantiles support a two-sided, zero-null relative contrast only:

- No declared clusters: `arm.metric.quantile_cluster`.
- No breakouts: `readout.metric.quantile_breakout`.
- No CUPED: `arm.metric.quantile_cuped`; request `variance_reduction="none"`.
- No observational estimator: `readout.observational.quantile`, including
  fixed-horizon definitions, artifacts, summaries and panels.
- No one-sided alternatives, shifted nulls or guardrail margins:
  `readout.metric.quantile_alternative`. Every non-`None` absolute null is
  unsupported, including zero; `value_scale="absolute"` is refused outright.

Use the [capability matrix](../reference/capabilities-by-entry-point.md) and
[error-code reference](../reference/error-codes.md) for the applicable route
and structured context, not a different source as a workaround for an absent
estimator. An observational relative margin refuses earlier at construction
with `plan.observational.relative_margin`; a guardrail missing its margin
refuses with `plan.guardrail.margin_required`. On an otherwise admissible
read request, the one-sided/absolute-margin check precedes the observational
quantile refusal.

## Quantiles and portable moments

Exporting a fixed-horizon quantile refuses with `source.frame.quantile_no_moments`
from the metric catalog, before reading counts or writing a file—even when
additive metrics accompany it. Definitions, reopened artifacts, summaries and
panels share that rule. Observational quantile export instead refuses with
`readout.observational.quantile`; a clustered quantile cannot be exported
(`evidence.cluster_unavailable`).

A declaration over an existing scalar-moments cube can construct but cannot
recover per-unit outcomes. Its ordinary two-sided randomized read refuses
with `evidence.unit_rows_unavailable`. One-sided or shifted-null requests
still refuse first with `readout.metric.quantile_alternative`; an admissible
observational request instead reaches `readout.observational.quantile`.
Construction-time margin checks above still apply.

On that cube, breakout reads refuse with `evidence.breakout_unavailable`;
daily/as-of values or lift refuse with `evidence.grain_unavailable`;
`planning_baseline` refuses with `analysis.planning_baseline.quantile_source_unavailable`.
These view/source checks precede estimator reads under either design.

A registered sequential export is different: it can checkpoint modeled
unit-record proofs while retaining an **unmodeled, unwindowed** catalog
quantile. It writes no quantile moments and still cannot estimate that
quantile sequentially. The `sequential-mixed-quantile-catalog-checkpoint`
scenario verifies matching modeled results and retained state across
definitions, artifacts, summaries and panels. `from_moments` can replay
that checkpoint/catalog but cannot re-export (`evidence.operation_unavailable`);
switchback cannot declare that catalog. See [sequential contracts](../reference/sequential-contracts.md).
