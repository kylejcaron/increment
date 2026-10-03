# Which experiment analysis method should I use?

Choose the estimator from the treatment-assignment process, the monitoring schedule, and the question you need to answer. Do not use a randomized-experiment estimator for a treatment that users selected themselves.

## I have a randomized A/B test

Use the randomized design and compare treatment arms against the declared control. Start with the [Quickstart](quickstart.md) for one-row-per-unit or unit-by-day data.

## My arm sizes look imbalanced

Use [SRM diagnostics](srm-diagnostics.md). It detects an observed-count
imbalance only; it cannot determine whether a broken assignment mechanism or
an asymmetric downstream selection or telemetry gap caused it. Check both the
full assigned-population counts and the downstream-population counts used by
your metric. A mismatch that appears only downstream points to selection or
telemetry, not assignment.

## I have pre-treatment data correlated with the outcome

Use [CUPED variance reduction](cuped.md). CUPED uses pre-experiment information to reduce outcome variance; it does not change the treatment assignment or estimand.

## I need sample size, power, or MDE before launch

Use [Power analysis](power-analysis.md). Solve for required sample size, achieved power, or minimum detectable effect from control-arm assumptions.

## I need a percentile, not a mean

Use [Quantile metrics](quantile-metrics.md). A quantile is not additive across units, so it
needs a distribution-free order-statistic interval instead of the moments-based method every
other metric type shares, and only `run()` serves it.

## I will inspect the experiment repeatedly

Use sequential or always-valid inference in the declared analysis plan. These methods are for repeated looks; do not replace them with a fixed-horizon interval while continuing to peek. Declare the sampling model, predictive priors and joint reveal law before data access as described in [Registered sequential inference](sequential-inference.md) and the sequential planning section in [Power analysis](power-analysis.md).

## I want a decision framed with probabilities, not just a p-value

Use [Priors and Bayesian decisions](priors-and-decisions.md). Declaring a prior shifts a
metric's reported interval to a posterior and adds probability-of-superiority and
expected-loss decision summaries, without changing which estimator ran underneath.

## Treatment was self-selected, phased, or opt-in

Use an `Observational` design with unit-level dataframe data and declare and justify the adjustment set. Increment checks overlap and post-adjustment balance, but cannot validate conditional ignorability. Choose IPTW, DML, or AIPW only when the causal assumptions are defensible. Read [Observational comparisons](observational.md).

## Assignment changed treatment uptake but did not force treatment

Use an encouragement design. Report ITT, compliance, and LATE only when the relevant
first-stage, exclusion, monotonicity, and random-assignment assumptions are defensible. Read
[Encouragement designs](encouragement.md), and see [LATE over time](../examples/late_over_time.md)
for how the estimate matures under staggered enrollment.

## Treatment alternates within units or a shared roster over time

Use a [fixed-horizon switchback contrast](switchback.md) when treatment order
is randomized independently within each unit-cycle, or a shared schedule when
one order is drawn per two-period block for a fixed roster. Declare a
two-period `SwitchbackAssignment`, exclude the declared washout (plus any
declared `carryover_order`, supported at `0`, `1`, or `2` when enough
observation steps remain), and report the mean-unit additive retained-window
difference. This path assumes no residual carryover after the discarded
steps. Unit-cycle orders use a prospective variance envelope for calibrated
inference; otherwise the qualified unit-t approximation
(`UnitCycleTApproximation()`, the default) gives approximate inference.
Shared schedules use a block-level Student-t interval. It does not provide sequential/always-valid
inference, CUPED, multiplicity adjustment, or arm-method overrides.

## Choosing decision and sensitivity methods

For parallel arm evidence, choose one `decision_method` and any reporting-only
`sensitivity_methods` for each metric. Bind them in the analysis plan (or
`MetricSpec` for dataframe input), or override them call-wide with
`run(decision_method=..., sensitivity_methods=...)`. Sensitivity rows do not
create additional decision evidence.

When the decision method is omitted, observational designs use IPTW; randomized
and encouragement designs retain their unadjusted default. Adding a sensitivity
method does not change that decision: for example,
`run(sensitivity_methods=[Method(name="unadjusted")])` on an observational
analysis reports an IPTW decision and an unadjusted sensitivity. An explicitly
chosen unadjusted decision stays unadjusted.

An omitted sensitivity override inherits the declaration; `sensitivity_methods=[]`
clears it. The legacy method-list keyword is not part of the public `run()` API.

For observational designs, `run_daily_lift()` and `run_asof_lift()` refuse with
`facade.analysis.observational_day_axis`; these views do not provide adjusted
day-axis contrasts. Use `run_daily()` or `run_asof()` for descriptive values,
or `run()` for adjusted whole-experiment comparisons.

For metric selection plus always-valid inference, secondary family role,
enabled multiplicity, and evidence interactions, use
the [method compatibility reference](compatibility.md) and [Multiplicity](multiplicity.md)
for how a primary's alpha share divides again across its own arms, a guardrail tests unsplit
at the full alpha, and a secondaries family is controlled by false-discovery rate at its own
q instead of an alpha split.

## I need effects for different user segments

Use heterogeneous-treatment-effect tools after estimating the overall effect. Read the
[Heterogeneous effects example](../examples/hte.md) and
[Heterogeneity and rollout](heterogeneity-and-rollout.md) for segment-level heterogeneity
testing and rollout recommendations, and treat targeting validation as a separate decision
from effect estimation.

## I need warehouse-backed analysis

For randomized or observational experiments, declare fact sources, dimensions,
exposures, metrics, and experiments in YAML and use `Analysis.from_definitions`.
Increment compiles inspectable Ibis expressions to the configured warehouse
backend. `Analysis.from_unit_day_artifact` reuses published unit-day data.
Observational methods require the declared pre-exposure covariates; portable
moments cannot reconstruct them. Read [The data model](data-model.md) and
[Analysis from a warehouse](../examples/analysis_from_a_warehouse.md).

## What Increment does not do

Increment analyzes data that already exists. It does not allocate traffic, manage feature flags, ingest events, or maintain a separate event store.

## What the method you picked assumes

Every estimator here is approximate in some direction, and a few target a narrower estimand than their name suggests. Before you act on a number, read [Statistical limitations](../limitations.md) — it states what each method assumes, and when the assumption stops holding.
