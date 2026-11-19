# Which experiment analysis method should I use?

Start with how treatment was assigned, when you need to decide, and what you
want to measure. Follow the matching guide; check its assumptions before acting.

## I have a randomized A/B test

Use the randomized design and compare treatment arms against the declared control. Start with the [Quickstart](quickstart.md) for one-row-per-unit or unit-by-day data.

## My arm sizes look imbalanced

Use [SRM diagnostics](srm-diagnostics.md) to check whether group counts differ
from the planned allocation. Compare all assigned users with the users included
in the metric. An imbalance only in the latter may come from selection or
missing events. **The check cannot identify the cause** or prove assignment
was correct.

## I have pre-treatment data correlated with the outcome

Use [CUPED](cuped.md) to reduce variance with pre-experiment information.
It does not change treatment assignment or the quantity being estimated.

## I need sample size, power, or MDE before launch

Use [Power analysis](power-analysis.md). Solve for required sample size, achieved power, or minimum detectable effect from control-arm assumptions.

## I need a percentile, not a mean

Use [quantile metrics](quantile-metrics.md) through `run()`. They need the
outcome distribution, not just sums and counts, and use a distribution-free
order-statistic interval.

## I will inspect the experiment repeatedly

Use [sequential inference](sequential-inference.md), declared before you inspect
outcomes. Register the sampling model, priors and how observations arrive
together. Repeatedly checking a fixed-horizon interval is not a substitute.
See [power analysis](power-analysis.md) for planning the experiment.

## I want a decision framed with probabilities, not just a p-value

Use [Priors and Bayesian decisions](priors-and-decisions.md) when you want posterior probabilities or expected-loss summaries. A prior does not change the sampling estimator or its interval; those remain available separately.

## Treatment was self-selected, phased, or opt-in

Use an `Observational` design only if the measured adjustment variables account
for confounding. Overlap and balance checks cannot establish that assumption.
The [observational guide](observational.md) explains when IPTW, AIPW or DML
can support a causal comparison.

## Assignment changed treatment uptake but did not force treatment

Use an [encouragement design](encouragement.md). Check random assignment,
a real effect on uptake, exclusion (assignment affects outcomes only through
treatment), and monotonicity (no defiers). The guide distinguishes the effect
of assignment (ITT), uptake (compliance), and treatment among compliers (LATE).
The [notebook](../examples/encouragement.md) also monitors LATE over time.

## Treatment alternates within units or a shared roster over time

Use a [fixed-horizon switchback](switchback.md) when randomized treatment order
alternates over time. Declare a two-period `SwitchbackAssignment` and washout;
the result is an additive difference, not relative lift. Inference assumes no
residual carryover after washout.

Two schedule types are supported: independent orders within each unit-cycle,
or one shared order per two-period block for a fixed roster. The
[switchback guide](switchback.md#supported-contract) covers their different
intervals, optional carryover adjustment, and restrictions. This route does
not support sequential inference, CUPED, multiplicity adjustment or arm-method
overrides.

## Choosing decision and sensitivity methods
Choose one `decision_method` per metric. Optional `sensitivity_methods` show
how estimates change under other methods; they are **not extra decision
evidence**.

Declare these in the plan or, for dataframe input, in `MetricSpec`. Without an
explicit choice, observational designs use IPTW; randomized and encouragement
designs use their unadjusted default. Adding a sensitivity method never changes
that choice; an explicit unadjusted decision also stays unadjusted.

Call-wide `run(decision_method=..., sensitivity_methods=...)` overrides the
declaration. A different override of a non-default metric setting warns once
per setting. Matching overrides and overrides of undeclared defaults do not
warn. Omit the sensitivity override to inherit it; pass `[]` to clear it.
The legacy method-list keyword is not part of the public `run()` API.

For observational designs, daily and as-of lift views refuse with
`facade.analysis.observational_day_axis`; these views do not provide adjusted
day-axis contrasts. Use daily or as-of values views
(`View.daily(values=True)`, `View.asof(values=True)`) for descriptive values,
or the whole-window `View.overall()` view for adjusted whole-experiment
comparisons.

Check [method compatibility](compatibility.md) before combining options.
[Multiplicity](multiplicity.md) explains the error budgets: the primary's
alpha share is divided across its arms, guardrails use the full unsplit alpha,
and secondary families use false-discovery control at their own q. These
roles are not interchangeable.

## I need effects for different user segments

Use heterogeneous-treatment-effect tools after estimating the overall effect. See the [Heterogeneous effects example](../examples/hte.md) and [Heterogeneity and rollout](heterogeneity-and-rollout.md). Validate targeting separately from estimating effects.

## I need warehouse-backed analysis

Follow [warehouse setup](data-model.md) to declare sources, dimensions,
assignment, metrics and experiments in YAML. `from_definitions` builds queries
you can inspect; `from_unit_day_artifact` reuses published unit-day data.
Observational methods need declared pre-exposure covariates; portable moments
cannot reconstruct them. Try the [local warehouse notebook](../examples/analysis_from_a_warehouse.md).

## I need to assign traffic

Use the optional [assignment SDKs](assign-experiments.md) for deterministic
enrollment, variants, layers and holdouts. You can also keep using your existing
assignment system. Increment does not host feature flags or collect events.

## What the method you picked assumes

Every method has assumptions, and some estimate narrower effects than their names suggest. Read [Statistical limitations](../limitations.md) before acting on a result.
