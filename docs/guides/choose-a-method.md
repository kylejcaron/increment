# Which experiment analysis method should I use?

Choose a method from how treatment was assigned, how often you will look, and what decision you need to make. Start with the route below; check its assumptions before acting on the result.

## I have a randomized A/B test

Use the randomized design and compare treatment arms against the declared control. Start with the [Quickstart](quickstart.md) for one-row-per-unit or unit-by-day data.

## My arm sizes look imbalanced

Use [SRM diagnostics](srm-diagnostics.md) to check for a sample-ratio mismatch. Compare counts in the full assigned population with the population used by your metric: a mismatch that appears only downstream points to selection or telemetry, not necessarily broken assignment.

!!! warning "SRM cannot identify the cause"
    It detects an observed-count imbalance only; it cannot determine whether a broken assignment mechanism or an asymmetric downstream selection or telemetry gap caused it.

## I have pre-treatment data correlated with the outcome

Use [CUPED variance reduction](cuped.md). CUPED uses pre-experiment information to reduce outcome variance; it does not change the treatment assignment or estimand.

## I need sample size, power, or MDE before launch

Use [Power analysis](power-analysis.md). Solve for required sample size, achieved power, or minimum detectable effect from control-arm assumptions.

## I need a percentile, not a mean

Use [Quantile metrics](quantile-metrics.md). Quantiles are not additive across units and use a different interval method; only `run()` supports them.

??? info "Quantile interval construction"
    A quantile is not additive across units, so it needs a distribution-free order-statistic interval instead of the moments-based method every other metric type shares, and only `run()` serves it.

## I will inspect the experiment repeatedly

Use sequential or always-valid inference declared in the analysis plan. These methods are designed for repeated looks; repeatedly checking a fixed-horizon interval is not a substitute. Register the sampling model, priors, and joint reveal law before reading outcomes. See [Registered sequential inference](sequential-inference.md) and [Power analysis](power-analysis.md).

## I want a decision framed with probabilities, not just a p-value

Use [Priors and Bayesian decisions](priors-and-decisions.md) when you want posterior probabilities or expected-loss summaries. A prior does not change the sampling estimator or its interval; those remain available separately.

## Treatment was self-selected, phased, or opt-in

Use an `Observational` design for self-selected, phased, or opt-in treatment. Declare and justify the adjustment set. Increment checks overlap and post-adjustment balance, but cannot validate conditional ignorability. Choose IPTW, DML, or AIPW only when their causal assumptions are defensible; see [Observational comparisons](observational.md).

## Assignment changed treatment uptake but did not force treatment

Use an encouragement design when random assignment changes treatment uptake but does not force it. Report ITT, compliance, and LATE only when random assignment, a relevant first stage, exclusion, and monotonicity are defensible. See [Encouragement designs](encouragement.md) and [LATE over time](../examples/late_over_time.md).

## Treatment alternates within units or a shared roster over time

Use a [fixed-horizon switchback contrast](switchback.md) when treatment order is randomized independently within each unit-cycle, or a shared schedule when one order is drawn per two-period block for a fixed roster. Declare the two-period `SwitchbackAssignment` and washout (and optional `carryover_order`, 0–2, when enough observation steps remain). The estimate is the mean-unit additive difference over retained windows; it assumes no residual carryover after washout. Unit-cycle inference uses a prospective variance envelope or the default approximate `UnitCycleTApproximation()`; shared schedules use a block-level Student-t interval. This route does not support sequential inference, CUPED, multiplicity adjustment, or arm-method overrides.

## Choosing decision and sensitivity methods
For parallel-arm evidence, choose one `decision_method` and optional reporting-only `sensitivity_methods` per metric. Declare them in the plan (or `MetricSpec` for dataframe input), or override them call-wide with `run(decision_method=..., sensitivity_methods=...)`. A differing call-time override replaces a non-default metric declaration for that run and emits one warning per overridden setting. Sensitivity rows are not additional decision evidence.

Without an explicit choice, observational designs use IPTW; randomized and encouragement designs use their unadjusted default. A sensitivity method does not change the decision method. For example, requesting unadjusted sensitivity on an observational analysis still reports an IPTW decision. An explicit unadjusted decision remains unadjusted.

An omitted sensitivity override inherits the declaration; `sensitivity_methods=[]` clears it. Matching overrides and overrides of undeclared defaults do not warn. The legacy method-list keyword is not part of the public `run()` API.

??? info "Decision defaults and sensitivity overrides"
    When the decision method is omitted, observational designs use IPTW; randomized and encouragement designs retain their unadjusted default. Adding a sensitivity method does not change that decision: for example, `run(sensitivity_methods=[Method(name="unadjusted")])` on an observational analysis reports an IPTW decision and an unadjusted sensitivity. An explicitly chosen unadjusted decision stays unadjusted.


For observational designs, daily and as-of lift views refuse with
`facade.analysis.observational_day_axis`; these views do not provide adjusted
day-axis contrasts. Use daily or as-of values views
(`Scope(..., values=True)`) for descriptive values, or the whole-window `run`
view for adjusted whole-experiment comparisons.

For metric selection, always-valid inference, secondary family roles, and interactions between methods, see the [method compatibility reference](compatibility.md) and [Multiplicity](multiplicity.md).

??? info "Allocation across metrics and arms"
    For metric selection plus always-valid inference, secondary family role, enabled multiplicity, and evidence interactions, use the [method compatibility reference](compatibility.md) and [Multiplicity](multiplicity.md) for how a primary's alpha share divides again across its own arms, a guardrail tests unsplit at the full alpha, and a secondaries family is controlled by false-discovery rate at its own q instead of an alpha split.

## I need effects for different user segments

Use heterogeneous-treatment-effect tools after estimating the overall effect. See the [Heterogeneous effects example](../examples/hte.md) and [Heterogeneity and rollout](heterogeneity-and-rollout.md). Validate targeting separately from estimating effects.

## I need warehouse-backed analysis

For randomized or observational warehouse analysis, declare sources, dimensions, exposures, metrics, and experiments in YAML, then use `from_definitions`. Increment compiles inspectable Ibis expressions for the configured backend. `from_unit_day_artifact` reuses published unit-day data. Observational methods need declared pre-exposure covariates; portable moments cannot reconstruct them. See [The data model](data-model.md) and [Analysis from a warehouse](../examples/analysis_from_a_warehouse.md).

## What Increment does not do

Increment analyzes existing data; it does not allocate traffic, manage feature flags, ingest events, or maintain an event store.

## What the method you picked assumes

Every method has assumptions, and some estimate narrower effects than their names suggest. Read [Statistical limitations](../limitations.md) before acting on a result.
