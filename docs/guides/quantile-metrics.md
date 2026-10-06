# Quantile metrics

A mean summarizes a distribution's center; some questions need a tail,
such as p90 latency or p99 payload size. A quantile is not additive
across units, unlike a sum or mean. Increment therefore estimates it with
a distribution-free order-statistic method instead of the moments cube
used by other metric types, and only `run()` can serve it.

## Declaring a quantile metric

```python
import numpy as np
import polars as pl
from increment import Analysis, MetricSpec

rng = np.random.default_rng(12)
n = 2000
variant = np.where(rng.random(n) < 0.5, "treatment", "control")
latency = np.exp(rng.normal(4.5, 0.5, n)) + np.where(variant == "treatment", -15.0, 0.0)
latency = np.clip(latency, 1.0, None)
df = pl.DataFrame(
    {"user_id": [f"u{i}" for i in range(n)], "variant": variant, "latency_ms": latency}
)

results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="latency_ms", type="quantile", quantile=0.9)],
).run()
for r in results:
    print(f"p90 relative lift: {r.lift.value:+.2%} [{r.lift.lb:+.2%}, {r.lift.ub:+.2%}]")
```

```text
p90 relative lift: -7.94% [-13.63%, -1.87%]
```

`MetricSpec(type="quantile", quantile=q)` requires `0 < q < 1` and
refuses `window_days` (quantiles have no windowed decomposition) and
`covariate` (a quantile has no mean to CUPED-adjust). On the definitions
(YAML) path the equivalent declaration is a `QuantileMetric` with the
same `quantile` field.

## The Woodruff order-statistic bracket, and tied values

The reported standard error comes from a distribution-free bracket
`[Y_(a), Y_(b)]` around the sample quantile, evaluated at the request's
own `alpha`, not from a plug-in delta-method SE (Woodruff 1952). The
bracket covers the population quantile at the nominal rate for every
distribution, continuous or not. The classical Woodruff SE (a symmetric
width around the sample quantile) inherits that coverage only when the
quantile sits at the bracket's log midpoint. Continuous data guarantee
that condition; tied recorded values do not. Prices, counts, and rounded
latencies can pin the bracket to one recorded value on one side and place
it a full recording cell away on the other.

The bracket's ranks are checked against independently enclosed binomial tails,
not a fixed accuracy allowance for SciPy's CDF. Certified logarithms enclose the
mass at a candidate rank; bounded summation and a remainder bound enclose its
tail. If a numerical comparison cannot be resolved, the bracket rounds outward.
No additional configuration is required, and the sampling assumptions are
unchanged.

Increment does not refuse a tie. When a bracket contains a tie and the
arm is not "resolved" (its 95% bracket spans at least
`RESOLVED_REPEATED_VALUES` = 12 distinct repeated values), the reported
half-width expands to the smallest symmetric interval around the sample
quantile that contains the whole bracket. This preserves the bracket's
coverage on a recording grid. A resolved arm, or any untied bracket,
keeps the classical Woodruff half-width bit-identically.

Under coincidental ties, the widened SE is never worse than about
1.00–1.05x the classical one. Coarse rounding or count data can produce
1.1–3x widths when the bracket spans only two or three recorded values;
there is no upper bound. A bracket collapsed onto one recorded value has
zero classical width, but the reported width remains positive. If values
collapse so far that the widened bracket cannot resolve a strictly
positive spread, Increment raises `estimation.quantile.degenerate_spread_order`.
That is a numerical backstop, not an ordinary tie response.

## Feasibility floor per (q, alpha)

The order-statistic bracket needs a minimum per-arm sample size for its
requested coverage. That floor grows as `q` approaches 0 or 1 and as
`alpha` shrinks. Too few units for a `(quantile, alpha)` pair raises an
error naming both values:
`n=<n> is too small to bound the q=<q> quantile at level <level> -- ...
needs n >= <n_min> per arm; collect more units or target a less extreme
quantile`.

## Alpha-dependent interval, alpha-independent p-value

The bracket uses the caller's `alpha`, so the SAME quantile metric's
reported standard error and interval change with its multiplicity budget,
even before BH selection. A fixed-reference-alpha SE rescaled to other
alphas was measured to fail coverage. The p-value differs: Increment
computes it once by inverting the same construction (the smallest alpha
whose interval excludes the null), so it depends only on the data.
Do not compare quantile standard errors or intervals across analyses with
different alpha budgets. Read
[A quantile's reported interval depends on the alpha you asked for](../limitations.md#a-quantiles-reported-interval-depends-on-the-alpha-you-asked-for)
before comparing plans; the p-value remains comparable.

## Sequential inference is refused

A quantile metric cannot use sequential inference. The Woodruff bracket is
fixed-horizon, and no sequential (time-uniform) boundary exists for a
per-arm order statistic. `AnalysisPlan(inference=InferenceSpec(
kind="always_valid"|"asymptotic_mean"))` refuses before reading outcomes
with `sequential.route.unsupported`. A metric that reaches estimation under
an explicit sequential registration refuses again with
`arm.metric.quantile_sequential`. Use fixed-horizon quantile inference (valid for one planned analysis, not repeated looks).

## Unsupported combinations

A quantile metric cannot be estimated with a declared `cluster`
(`arm.metric.quantile_cluster` -- quantiles do not decompose over
cluster moments), under a `dimension=`/breakout request
(`readout.metric.quantile_breakout` -- quantiles do not decompose over
segment moments), with CUPED (`arm.metric.quantile_cuped` -- a
quantile has no mean to adjust; pass `variance_reduction="none"`), or under
an `Observational` design (`readout.observational.quantile` -- the
order-statistic interval assumes independently randomized arms, and no
observational quantile estimator exists; run the metric under a randomized
design). Where `readout.observational.quantile` is reached, it is the same on
`from_definitions`, `from_unit_day_artifact`, `from_unit_summary`, and
`from_unit_panel`, and fixed-horizon inference does not lift it. It is the
refusal for a two-sided, zero-null request. Any one-sided or shifted-null
request is refused first by `readout.metric.quantile_alternative` -- a plan
`alternative="greater"`/`"less"`, a guardrail without a margin (its adverse
tail is one-sided), or an absolute `margin_abs`. A relative margin never
reaches a readout: the observational constructors refuse it at construction
with `plan.observational.relative_margin`.

A quantile metric also refuses a one-sided `alternative` and a shifted
null under the single code `readout.metric.quantile_alternative`
("one-sided alternative is not supported for quantile metrics"):
only a two-sided test against `null_lift=0` on the relative axis is
supported. A non-zero `null_lift` override or a non-inferiority/
superiority `margin` refuses this way (both resolve to a non-zero
`null_lift` and an implied one-sided tail). The absolute axis is not
a second supported mode -- every non-`None` `null_abs`, including a
hypothetical zero, is refused: a declared `margin_abs` always resolves
to a non-zero `null_abs` with an implied one-sided tail (so it refuses
the same way), and a quantile metric can never report on the absolute
value scale in the first place (`value_scale="absolute"` is refused
outright for `type="quantile"`). Quantile metrics output relative lift
only.

## Quantiles and portable moments

A portable moments cube holds additive moments, never the per-unit values an
order statistic needs, and two different requests reach it:

- **Exporting a quantile** refuses at `export()` with
  `source.frame.quantile_no_moments` (a source limit: a quantile has no moments
  representation), so no quantile cube ever exists to replay. Every
  export-capable ingress (`from_definitions`, a reopened unit-day artifact,
  `from_unit_summary`, `from_unit_panel`) refuses a fixed-horizon moments
  export the same way, from the metric catalog alone, before reading any
  count or moment or writing a file, even when additive metrics accompany the
  quantile. An observational design refuses with
  `readout.observational.quantile` instead, because no estimator could use the
  cube. A registered sequential plan is the one exception: it exports a
  checkpoint of unit-record proofs and the declaration, with no moments rows,
  so a catalog quantile no registered model covers does not block it and never
  appears as mean moments (replay still refuses to estimate that quantile
  sequentially).
- **Declaring a quantile over a cube that already exists** (for example a
  `MetricSpec(type="quantile")` over exported scalar moments) constructs, then
  refuses when read, and the code depends on the request, measured on a real
  scalar-moments source:

  | Design and request | `run()` raises |
  |---|---|
  | randomized, two-sided zero null | `source.moments.unit_grain` (those moments are not quantile data) |
  | randomized, one-sided `alternative`, marginless guardrail, or relative or absolute margin | `readout.metric.quantile_alternative` (refused first) |
  | `Observational`, two-sided zero null | `readout.observational.quantile` (the estimator refusal takes precedence over the source limit, because no source could supply it an input) |
  | `Observational`, one-sided `alternative`, marginless guardrail, or absolute margin (`margin_abs`) | `readout.metric.quantile_alternative` (the one-sided check precedes the observational refusal) |
  | `Observational`, relative margin | `plan.observational.relative_margin`, at construction, before any read |

  `run_breakout()` raises `facade.analysis.operation`, the day-axis methods
  (`run_daily`, `run_daily_lift`, `run_asof`, `run_asof_lift`) raise
  `facade.analysis.no_definitions`, and `planning_baseline` raises
  `analysis.planning_baseline.quantile_source_unavailable`; these source limits
  fire before any estimator reads the cube, under either design. A clustered
  cube cannot be exported for a quantile in the first place
  (`source.moments.cluster_grain`).
