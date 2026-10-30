# Power analysis and MDE for A/B tests

Power analysis answers three questions before an experiment starts: how many
units you need, what power a given size gives, and the smallest lift you can
reliably detect. Each answer comes from a control-arm `Baseline` and the same
decision rule the analysis will use.

The standard conversion plan, a 10% baseline rate with a 10% relative lift (to
11%), tested two-sided at `alpha=0.05` for 80% power, needs 15,597 units per
arm:

<!-- skip: next "enumerating the conversion count lattice takes about a minute" -->
```python
from increment.estimators import Baseline
from increment.power import ArmPlanningProcedure, required_sample_size

plan = required_sample_size(
    relative_lift=0.10,
    baseline=Baseline.from_proportion(0.10),
    procedure=ArmPlanningProcedure.standard("conversion"),
)
print(plan.n_per_arm, plan.n_total, plan.power_basis)
print(f"power={plan.power:.4f} in [{plan.power_lower:.4f}, {plan.power_upper:.4f}]")
```

```text
15597 31194 lower_bound
power=0.8000 in [0.8000, 0.8003]
```

## What `power` and `power_lower` mean

An unadjusted, unclustered, fixed-horizon conversion or retention plan without
CUPED, absorbed factors, winsorization, or priors can replay the runtime's own
accept/reject decisions over plausible control and treatment counts. Adding
up the rejecting counts' probabilities gives a narrow range, `power_lower`
to `power_upper`, rather than just a single number:

- When the computed point is within `1e-6` of both enclosure endpoints,
  `power` is that point and `power_basis="exact"`.
- When the computation budget leaves it wider, as in the plan above (about
  `3e-4`), `power` is the bottom of the range and `power_basis="lower_bound"`.
  The runtime's rejection probability is at least `power`, under the planner's
  stated numerical qualification, so the plan errs on the side of a few more
  units, never fewer.
- For an enumerated plan, a target is reached only when `power_lower`
  reaches it. A size the range straddles is not reached.
- Mean metrics and supported binary plans outside the conditions above use
  a model-based calculation: `power_basis="asymptotic"`, with `power_lower`
  and `power_upper` set to `None`. Dense eligible binary plans can also
  use a closed-form model when their count lattice is too large to enumerate.

!!! warning "Power is conditional on the decision rule and planning model"
    Every result depends on the selected decision rule and planning
    assumptions. A model-based result estimates power; it does not bound the
    runtime's rejection probability. An enumerated result's numerical range
    addresses evaluation accuracy, not model misspecification or calibration.

## Size a mean metric

Every parallel-arm solver takes a validated
`increment.estimation.arm_contract.ArmPlanningProcedure`: the assignment,
analysis axes, estimand, metric capabilities, decision rule, compiled
inference, decision method, sensitivity methods, family expansion, and whether
a prior is present. Planning therefore sits on the same contract as the
analysis that will consume the result. `ArmPlanningProcedure.standard(metric_type)`
builds the ordinary fixed-horizon one:

```python
from increment.estimators import Baseline, PowerDesign
from increment.power import (
    ArmPlanningProcedure,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)

baseline = Baseline(mean=20.0, var=400.0)
# The default procedure is a fixed-horizon standard for this baseline's metric.
size = required_sample_size(
    relative_lift=0.02,
    baseline=baseline,
    design=PowerDesign(power=0.80, allocation=0.5),
)
power = achieved_power(25_000, 0.02, baseline)
mde = minimum_detectable_effect(25_000, baseline)
```

The solver signatures are:

```text
required_sample_size(relative_lift, baseline, procedure=None, design=None, *, planned_looks=None)
achieved_power(n_per_arm, relative_lift, baseline, procedure=None, design=None, *, planned_looks=None)
minimum_detectable_effect(n_per_arm, baseline, procedure=None, design=None, *, planned_looks=None)
```

When `procedure` is omitted, the baseline's declared metric type determines
the standard procedure; clustered baselines also select the clustered standard.
An explicit procedure always takes precedence. Pass an explicit procedure to
change the decision method, inference regime, or other planning choices.
`inference=`, `alpha=`, `alternative=`, and method-list arguments are not solver
parameters; put those decisions in the procedure. `PowerDesign` supplies only
the target `power` (default `0.80`) and treatment `allocation` (default `0.5`).
`n_per_arm` is the treatment-arm size; the control-arm size follows from
`allocation`, and `PowerResult.n_total` is the assigned total across the
contrast. Each solver returns a complete `PowerResult` at the actual integer
size, including `power`, `mde_relative`, and `effective_var`.

## Build a baseline

| Constructor | Use for |
|---|---|
| `Baseline(mean=, var=)` | A control-arm mean and per-unit variance. |
| `Baseline.from_proportion(p)` | Conversion and retention rates. |
| `Baseline.from_summary(summary_stats)` | A pilot variance with its confidence-bound inflation. |
| `Baseline.from_ratio(...)` | A ratio metric's linearized numerator, denominator, and covariance moments. |
| `Baseline.from_absorption(mean, var, icc)` | A measured one-way factor absorption reduction. |
| `analysis.planning_baseline(metric)` | An `Analysis` over a prior or pilot experiment (or an in-flight experiment's pre-treatment data): reads that data and its declared design directly. |

`Baseline.metric_type` defaults to `"mean"` and is omitted from dumps at that
default, so existing mean baselines keep their serialized form. Conversion,
ratio, and quantile constructors declare their respective metric types;
`planning_baseline()` carries the declared metric type through the selected
pilot route so the standard solver procedure matches the metric.

With no prior experiment or pilot data, there is nothing for
`planning_baseline` to read; use the other constructors with an assumed or
externally sourced value. On a switchback analysis the same call returns the
pilot-fitted `SwitchbackBaseline` the `switchback_*` solvers take (see
[switchback contrasts](#switchback-contrasts)); for a quantile metric it
returns a `QuantileBaseline`.

??? info "Statistical details: what `planning_baseline` reads"
    Outcome moments describe the analyzed population: under a declared
    `trigger` that is the triggered population, with `trigger_rate` its
    observed share of the assigned control arm; with no trigger declared, the
    assigned population at `trigger_rate=1.0`. A declared trigger whose
    evidence the source does not carry (a unit-day artifact published without
    the `trigger_population` and `assignment_counts` extensions) refuses with
    `analysis.planning_baseline.trigger_evidence_unavailable`.

    From that population's control arm: per-unit mean/var (or a ratio's
    linearized moments); `cuped_rho` when this metric declares CUPED, as the
    control arm's variance reduction under the runtime's own CUPED fit;
    `compliance` from the source's design-level uptake when the design is
    `Encouragement`; and `cluster_icc`/`cluster_size_cv` from contributing
    clusters when a `cluster` is declared, with ICC on the analyzed score
    (for a ratio, `y - R*y_den`). Recruitment metadata stays on the assigned
    population: `avg_cluster_size` is assigned control units per randomized
    cluster, and `cluster_participation` is contributing control clusters
    divided by assigned control clusters. The artifact route also needs
    `cluster_identity` for a clustered pilot.

    Both clustered trigger routes require the metric's outcomes for every
    trigger-eligible control unit in the pilot. Missing metric rows raise
    `analysis.planning_baseline.trigger_metric_population_unavailable`, with
    observed and eligible counts. Complete unfinished windows. If an outcome
    is structurally undefined, such as `avg_event` for a unit with no events,
    choose a metric defined for every eligible unit; waiting or republishing
    cannot define that outcome.

### Quantile metrics

`QuantileBaseline` carries the pilot's real per-unit control values, not a
single variance: at each candidate size the solver re-evaluates the readout's
own order-statistic construction over them, and its `var` is the per-unit
variance the pilot's standard error implies. The baseline declares its metric
type, so the solver can infer the matching standard procedure; an explicit
procedure can still be supplied when different planning choices are needed.
Results echo `PowerResult.planned_metric_name`/`planned_quantile`, so reusing
one metric's baseline for another is visible in the answer. There is no CUPED
route for quantiles.

!!! warning "Quantile power has a ceiling and is not monotone"
    On rounded data the readout's interval never narrows below the gap from
    the quantile to the next recorded value, so a target above the ceiling is
    refused with `power.quantile_size_search_unreachable` (context
    `maximum_power`, `limiting_condition="recording_grid"`). Power can dip
    past the pilot's size, and a planned size within about a quarter of the
    pilot's size measured 0.71-0.76 power against a planned 0.80 on rounded
    data.

??? info "Statistical details: quantile planning"
    Quantile planning follows the quantile readout: at each candidate size it
    applies the readout's own half-width rule (`quantile_half_width`) to a
    bracket projected from the pilot, and at the pilot's own size the planned
    standard error is the readout's on the pilot. A pilot whose bracket holds
    no tied values (continuous data) plans the classical standard error,
    shrinking as `1/sqrt(n)`. On rounded data the readout's point and bracket
    ends are recorded values, so planning reads them from the pilot's
    recorded distribution. At any other size than the pilot's, where the
    quantile falls within its recording cell is known only to the pilot's
    precision, and that position decides whether a narrow bracket spans one
    cell or two, so the planned standard error averages over it, integrating
    exactly over the recorded values the bracket can reach: between adjacent
    sizes the average moves by no more than the mass a bracket end can carry
    across one recorded value. The averaged shift is that of a sample sharing
    its units with the pilot, so it vanishes at the pilot's size and is
    smallest near it.

    Power is not monotone in the sample size: it passes through the pilot's
    own power at the pilot's size, an arm whose bracket stops spanning twelve
    repeated values switches to the readout's wider tied interval so power
    dips past that size, and on a coarse grid it wavers slightly with the
    readout's own bracket ranks. The search evaluates a fine grid of sizes
    in order and settles the crossing between the first grid size reaching
    the target and the one before, so the planned size reaches the target
    while the size below it does not; a smaller size can reach the target
    only within that rank-level waver. A size at which the projected bracket
    reaches a non-positive recorded value has no log-scale interval, as at
    the readout, and `achieved_power` refuses it with the readout's own
    code; the size search passes over such sizes. The quantile readout tests
    two-sided against a zero null and has no breakouts, so planning refuses
    a one-sided alternative, a shifted null or a guardrail role with
    `readout.metric.quantile_alternative`, and the `segment_pairwise_*`
    solvers refuse a quantile metric with `readout.metric.quantile_breakout`.

??? info "Statistical details: the alternative-arm variance model"
    The three arm solvers evaluate each arm's log-scale variance at its own
    mean under the alternative. For mean-like metrics, planning assumes the
    treatment arm keeps the control arm's absolute effective variance. For
    conversion and retention metrics the runtime does not decide with this
    model (below), it rescales that variance by the Bernoulli shape at the
    implied treatment rate. For quantile metrics, the treatment arm's own
    order-statistic construction is evaluated at its own analyzed count,
    reusing the pilot's recording grid (a property of the instrument, not of
    the hypothesized shift), an exact identity under this module's
    multiplicative-effect convention rather than the "same absolute
    variance" approximation. These are planning assumptions, not claims
    about every future data-generating process. Bounded baselines, nulls,
    and alternatives must imply rates in `(0, 1]`; a requested rate above 1
    is refused. Every such result reports `power_basis="asymptotic"`.

## Conversion and retention: planning follows the runtime's route

For the unadjusted, unclustered fixed-horizon conversion and retention plans
above (no CUPED, absorbed factors, winsorization, or priors), planning follows
the runtime's count-based decision. With the default
`conversion_inference="auto"`, the runtime uses a delta-method test for dense
arm success/failure counts and the exact Berger–Boos risk-ratio test
otherwise. The planner sums the combined decision's rejection probability
over the count law. This route uses the baseline rate, not `Baseline.var`;
other supported binary plans retain the model-based calculation.

| `power_basis` | `power` is | Reported range |
|---|---|---|
| `exact` | The enumerated rejection mass, with the computed point within `1e-6` of each endpoint. | `power_lower`, `power_upper` |
| `lower_bound` | `power_lower`, the bottom of a range where the computed point is farther than `1e-6` from an endpoint. Nothing inside the range is claimed: no midpoint, no substitute approximation. | `power_lower`, `power_upper` |
| `asymptotic` | The closed-form model's own probability, used where counts route to the delta method with near certainty (unrouted mass at most `1e-6`) and the lattice has more than 100,000 cells. | `None` |

For an enumerated plan, `required_sample_size` returns the smallest size of
its bracket whose lower end reaches the target; `minimum_detectable_effect`
and a companion `mde_relative` certify the returned effect the same way.
Enumerating the count lattice is deterministic work and can take about a
minute per call.

`cuped_rho` reduces planning variance by `(1 - cuped_rho**2)` only when the
effective decision method uses CUPED. A CUPED sensitivity readout does not
reduce an unadjusted decision's sample requirement. Mixed power/MDE curves
apply this rule independently to each procedure.

Power planning does not construct the mixed `cuped_linear` estimator and
refuses procedures that request it with `power.cuped_linear_unavailable`.
Plan without variance reduction, or use scalar CUPED on one covariate from the
set as an approximation. The scalar plan is not a `cuped_linear` power claim:
with many covariates or small arms, the extra fitted columns cost degrees of
freedom, so it is not guaranteed to be conservative.
`compliance` rescales a diluted encouragement effect, and `trigger_rate`
accounts for the fraction of assigned units entering a triggered analysis. For clustered
randomization, `avg_cluster_size` is the assigned mean size; `cluster_icc`
and `cluster_size_cv` describe analyzed scores and contributing cluster sizes.
For a triggered cluster design, also supply `cluster_participation`: the
fraction of recruited clusters that contributed at least one analyzed unit
in a comparable pilot. `planning_baseline()` derives it automatically on the
definitions and sufficiently published artifact routes.

    An enumerated result carries the computed enclosure of the rejection
    probability, `power_lower` and `power_upper` (both `None` for a
    closed-form or sequential plan). Internal bounds account for summation
    error, omitted support and unresolved count pairs. The point is published
    when the enclosure resolves to absolute error at most `1e-6`, conditional
    on the deployed SciPy/Boost special-function error model; this is not a
    cross-build floating-point proof. `"exact"` describes the decision-law
    calculation, not error-free arithmetic or a finite-sample guarantee for
    the hybrid delta-method route. Under `"lower_bound"`, the runtime's
    rejection probability is at least `power` conditional on the same error
    model, and `power_upper` bounds it from above. The dense validation
    tolerance remains `0.005`; it is a comparison criterion, not a universal
    error bound for every design or data-generating process.

    The explicit point request,
    `increment.power.core.planned_enclosure(...).reported`, refuses an
    unresolved enclosure with `power.binomial_probability_unresolved` (context
    `point`, `lower`, `upper`, `error`, `tolerance`). `required_sample_size`
    refuses with the same code, adding `n_per_arm`, when its search reaches
    the ceiling with an enclosure holding the target; a ceiling whose upper
    end falls short refuses with the replay-bound or size-limit code naming
    the certified lower end. No accuracy tuning parameter is exposed.

??? info "Statistical details: enumeration budgets and search limits"
    Enumeration has deterministic work and memory limits, not
    timing-dependent answers. Delta-decision enumeration is bounded at
    100,000 count pairs, and finite-sample replay at 150,000 directional
    replays per evaluation. A solve stores at most 10,000,000
    control/treatment count cells across its null and evaluated alternative
    windows. A supplied effect, an MDE search and each `power_curve` row are
    separate solves: each is bounded by the cells of its own requests, and an
    MDE search's interval bounds read only what that search decided.
    A count pair's decision is the runtime's own, the same whichever request
    decides it, so the solves of one design share those decisions: a companion
    MDE search copies the cells the supplied effect or the sizing already
    decided instead of replaying them, and replays of each interval's far
    endpoint only the cells its bound depends on; an endpoint whose power and
    cells no bound reads (the far end of the admissible effects, the far
    endpoints of wide intervals above the answer) is never replayed, while one
    a curvature term reads is. Exceeding the cell bound raises
    `power.binomial_replay_bound_exceeded`; an MDE search may return
    `mde_relative=None` with `mde_unavailable_reason="numerical_resolution"`.
    These are planner limits; the runtime's finite-sample ceiling remains
    1,000,000,000 units per arm. Dense `auto` plans using the closed form do
    not enumerate the count lattice.

    The runtime's bounded finite-sample search can produce nonmonotone
    decisions. Planning does not infer one count pair from its neighbour, and
    power need not be monotone in sample size or effect. A required size
    meets the target under the selected calculation; it does not prove no
    smaller size could. `minimum_detectable_effect` searches for the earliest
    detectable region to an effect tolerance of `1e-8` absolute plus `1e-8`
    relative, measured on the relative-effect scale, not the first
    IEEE-representable effect. The result's `power` is evaluated at the
    reported effect, not copied from the target. A band whose evaluated ends
    fall short with resolved enclosures refuses with `numerical_resolution`.
    A band whose evaluated ends both have unresolved enclosures straddling the
    target is passed over without interior probes; at tolerance width, either
    straddling end suffices.
    Thus `mde_relative` certifies its own target but may lie above an unprobed
    effect that would already certify it under the same evaluation budgets.
    This is a conservative bounded search, not a global minimum certificate.
    A target excluded over every admissible effect is `unattainable`. If no
    evaluated effect certifies the target and any skipped band remains
    unexcluded, or an earlier region remains unresolved at numerical tolerance,
    or the far end's enclosure straddles the target, a companion MDE is
    unavailable with `numerical_resolution`; a standalone call refuses with
    `power.minimum_detectable_effect.numerical_resolution`. A supplied
    effect's own answer still stands without a companion MDE.

    With a shifted null and partial compliance, early candidate effects can
    imply unrepresentable absolute lifts even when larger effects are
    representable. The planner bounds those candidates before excluding
    them; unresolved candidates yield `numerical_resolution`, not an
    assumption that a detectable effect exists outside the representable
    range. Calibration is a separate diagnostic, not a prerequisite for
    planning.

### When it refuses

A design the runtime refuses in full decides no count pair, so it has no
power to plan and is never replayed:

| Code | When |
|---|---|
| `power.binomial_arm_ceiling_exceeded` | An arm is beyond the runtime's one-billion-unit ceiling. |
| `power.binomial_arm_ceiling_below_smallest_design` | The allocation is so lopsided that the smallest design already has an arm above the ceiling. |
| `power.binomial_tail_level_unrepresentable` | `alpha` is too small for the decision: the nuisance budget (`alpha / 32`) is below the endpoint solver's floor (an alpha under `3.2e-8`), or the float margin dominates the tail level (a two-sided alpha below about `9.5e-7` at a billion units per arm). |
| `power.binomial_replay_bound_exceeded` | A size search reached its enumeration ceiling before reaching the target. |
| `power.binomial_size_search_unreachable` | A size search reached the arm ceiling, or the size where the float margin starts dominating, without reaching the target. |

??? info "Statistical details: refusal contexts"
    `achieved_power`, `minimum_detectable_effect` and every `power_curve` row
    that reaches a refused design raise `power.binomial_arm_ceiling_exceeded`
    (context `n_c`, `n_t`, `max_arm_size`) or
    `power.binomial_tail_level_unrepresentable` (`scope="requested"`). Under
    a dominating margin the runtime still decides a count pair whose control
    count alone rejects a shifted null (its Clopper-Pearson lower bound above
    `1 / (1 + null_lift)`, for a two-sided or "less" test); the context's
    `decided_from` is the smallest such count (`None` when there is none). A
    `finite_sample` plan whose control window at the baseline rate holds
    smaller counts, which the runtime refuses, is refused rather than planned
    with them as non-rejections; an `auto` plan is refused when the counts
    the count rule keeps on the finite-sample route below `decided_from`
    carry more than half of `1e-6`, and otherwise carries them as undecided
    mass. The margin grows with the arm, so a smaller size of the same alpha
    can be planned; the solver floor refuses every size.

    `required_sample_size` can end a replayed search with:

    | Code | When | Context |
    |---|---|---|
    | `power.binomial_replay_bound_exceeded` | the search reached its ceiling: about 1/128 under the crossing of a bisection for the largest size whose null and supplied-effect rectangles fit the replay bound (the cell count is not monotone in the size, so a size above the ceiling may fit and may reach more) | `power`, `power_reached`, `n_c`, `n_t`, `p_c`, `p_t` (when the alternative exceeds), `cells`, `max_cells`, `max_arm_size` |
    | `power.binomial_size_search_unreachable` | the search reached the runtime's arm ceiling, or the size where the float margin starts dominating the tail level, without reaching the target | `power`, `maximum_power`, `n_per_arm`, `max_arm_size` |
    | `power.binomial_tail_level_unrepresentable` | the decision is refused even at the smallest design: the nuisance budget is below the solver floor, or the float margin already dominates the tail level there, so no size reads the treatment arm; raised before any search | `alpha`, `beta`, `tail_alpha`, `margin`, `n_c`, `n_t`, `p_c`, `decided_from`, `solver_floor`, `cause` (`solver_floor` or `float_margin`), `scope` (`smallest`) |
    | `power.binomial_arm_ceiling_below_smallest_design` | the allocation is so lopsided that the smallest design already has an arm above the runtime's ceiling; raised before any search | `n_t`, `n_c`, `allocation`, `max_arm_size` |

### Comparing with statsmodels

[`TTestIndPower.power`](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.power.html)
and [`NormalIndPower.power`](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.power.html)
return model-based point rejection probabilities for independent-sample t-
and z-tests with standardized mean differences. Increment's `power` is the
same kind of quantity, but a relative-lift binomial risk-ratio test can
legitimately have different power. Compare only after aligning the decision
rule, effect scale, baseline/variance assumptions, alpha, sidedness and arm
allocation.

??? info "Statistical details: planning model versions"
    The current binomial planning model is `hybrid_finite_plus_delta_v3`.
    Persisted results labelled `hybrid_finite_plus_delta_v2` or its v1
    predecessor must be recomputed; they do not acquire point-power
    semantics by relabelling.

## Adjust the baseline for your design

Build the procedure with `ArmPlanningProcedure.standard(metric_type)` and
pass the same `Baseline` to the solver. Compliance, triggering, and factor
absorption are derived from non-default baseline fields; CUPED credit follows
the effective decision method. A procedure explicitly built with a
conflicting axis raises `CapabilityError` naming the field and the conflict.

| `Baseline` field | Effect on the plan |
|---|---|
| `cuped_rho` | Reduces planning variance by `(1 - cuped_rho**2)` only when the effective decision method uses CUPED. A CUPED sensitivity readout does not reduce an unadjusted decision's requirement. CUPED composes with `inference=InferenceSpec(kind='asymptotic_mean')`. |
| `compliance` | Rescales a diluted encouragement effect. |
| `trigger_rate` | The fraction of assigned units entering a triggered analysis. |
| `avg_cluster_size`, `cluster_icc`, `cluster_size_cv` | Clustered randomization: assigned mean size, ICC on analyzed scores, and the size CV across contributing clusters. |
| `cluster_participation` | For a triggered cluster design, the fraction of recruited clusters that contributed at least one analyzed unit in a comparable pilot. `planning_baseline()` derives it on the definitions and sufficiently published artifact routes. |

!!! warning "Measure the cluster size CV across contributing clusters only"
    Do not substitute the variation in assigned sizes, and set
    `cluster_participation` to the contributing fraction, not the unit
    trigger fraction.

??? info "Statistical details: the cluster design effect and reference"
    Writing assigned mean size as $m$, unit trigger fraction as $q$, contributing
    cluster fraction as $p$, and analyzed size CV as $c$, the analyzed mean size is
    $m q/p$ and the design effect is $1 + ((1+c^2)m q/p - 1)\rho$.
    For example, 100 assigned units in ten clusters with 20 triggered units have
    $m=10$, $q=0.2$. If five clusters contribute, set $p=0.5$, not $0.2$:
    the analyzed mean size is four. At ICC 0.2 and equal analyzed sizes, the
    design effect is 1.6; if all ten clusters contribute, it is 1.2 instead.

    Results report recruited cluster counts in `n_clusters_per_arm` and
    `n_clusters_total`, using assigned mean size. Inference uses the floored
    expected represented counts $K'_a=\lfloor K_a p\rfloor$, with roundoff
    allowance at integer boundaries, and requires at least two per arm.
    The relative t reference uses $\min(K'_T-1,K'_C-1)$ degrees of freedom.
    Analysis retains separate Welch degrees of freedom for its additive
    interval. This is a pilot-conditioned planning approximation, not a
    guarantee that a future trial will realize the pilot participation rate
    or achieve finite-sample coverage. There is no ten-cluster admission
    floor; small-cluster calibration remains a limitation. The noncentral-t
    log-scale power calculation is a planning approximation, not an exact
    forecast of Fieller-set rejection under weak denominators.

    The `segment_pairwise_*` solvers are separate: they retain a
    baseline-only four-arm Normal approximation with cluster design effects
    and the same two-per-arm structural minimum within each segment; they do
    not share the arm solver's t reference or its numerical power model. Do
    not use the three arm solvers as numerical oracles for that model. Mixed
    power/MDE curves apply the CUPED rule independently to each procedure.

### Secondary metrics and guardrails

A secondary metric plans at `role='secondary', secondaries=<count>,
q=<FDR target>`, sized as a conservative Bonferroni-at-`q/m` bound. A
guardrail plans at `role='guardrail', preferred_direction='increase'/'decrease'`
(or an explicit one-sided `alternative`), always at full alpha, never
arm-split.

??? info "Statistical details: secondary members"
    `m` is secondaries × comparisons under fixed-horizon inference, and
    secondaries alone under `inference=InferenceSpec(kind='asymptotic_mean')`,
    which supports one treatment arm. For a conversion or retention
    secondary, the runtime member is routed at its own level and read at
    `q/m` through its composite p-value, whose rejection set is contained in
    the count-routed decision at `q/m` the planner enumerates, so the planned
    power of such a member is an upper bound on its rejection probability
    (measured equal on the calibrated designs). A replay-bound refusal names
    both counts: the runtime's `dense_min_count` at the member's own tail and
    the `planned_dense_min_count` the closed form reads from.

## Curves

`power_curve` takes a scalar procedure, a sequence of procedures, or no
procedure (which selects the baseline's standard procedure). Provide exactly
one of `relative_lift` and `target_power`; the omitted quantity is solved for
each Cartesian-grid row, using the procedure's compiled decision alpha and
inference:

```python
from increment.power import power_curve

curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    relative_lift=[0.01, 0.02, 0.05],
    baseline=baseline,
    units_per_week=50_000,
)
print(curve.to_frame())

mde_curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    target_power=[0.80, 0.90],
    baseline=baseline,
)
```

The full curve signature is:

```text
power_curve(*, n_per_arm, baseline, procedure=None, relative_lift=None,
            target_power=None, design=None, units_per_week=None,
            planned_looks=None, max_workers=None)
```

Rows preserve input product order, including duplicates. `units_per_week`
adds `duration_days` based on assigned units; omit it to leave duration
unset. For sequential procedures, `expected_n_total` and
`expected_duration_days` describe expected early stopping alongside the
worst-case `n_total`/`duration_days`.

## Fixed and sequential planning

A fixed procedure uses the fixed-horizon reference. Declare sequential
planning the same way the runtime reads it,
`ArmPlanningProcedure.standard(..., inference=InferenceSpec(kind="asymptotic_mean"))`,
and the planner builds the always-valid boundary the runtime executes, tuned
identically at every information fraction, including one-sided decisions.
`planned_looks` sets the equal-look schedule (omitted: 14 looks). The result's
`inference_to_declare` is the exact `InferenceSpec` to pass at runtime
registration, with `expected_decision_sample_size` set to the solved
`n_total`, so planning and the runtime never silently disagree on N:

```python
from increment import InferenceSpec
from increment.power import ArmPlanningProcedure

procedure = ArmPlanningProcedure.standard("mean", inference=InferenceSpec(kind="asymptotic_mean"))
sized = required_sample_size(0.02, baseline, procedure, planned_looks=14)
# Declare the same boundary at runtime registration:
runtime_inference = sized.inference_to_declare
```

!!! warning "Power planning does not make unplanned peeking valid"
    Use a matching sequential procedure when the analysis plan commits to
    repeated looks; a fixed-horizon plan is valid for one planned analysis.

Refused on this route:

- `AlwaysValid`, which carries a registered raw-likelihood runtime policy, and
  continuous sequential `AsymptoticMean` inference, which uses actual
  finalized arm counts and a shifted-contrast variance: both refuse with
  `sequential.route.unsupported` (see
  [continuous sequential means](sequential-inference.md#continuous-scalar-means)).
  This prospective route does not size registered likelihood policies.
- A CUPED method with non-fixed inference: `arm.adjustment.sequential_cuped`.
- Clustered sequential planning; use fixed-horizon clustered planning or
  unit-randomized sequential planning.

??? info "Statistical details: the sequential crossing calculation"
    The `asymptotic_mean` boundary used for planning is the boundary the
    runtime executes (see [sequential inference](sequential-inference.md));
    planning does not use a different group-sequential Gaussian boundary.
    Sequential inversion recomputes the alternative-arm standard error and
    boundary at every candidate. The crossing evaluator uses composite
    Gauss–Legendre integration with propagated derivative-remainder,
    floating arithmetic, and normal-tail bounds; expected information is
    bounded independently. Internal arm-planning paths carry the log
    variance into both the boundary and standardized drift, avoiding a
    zero/infinite intermediate at extreme scales. An MDE is returned only
    when those bounds certify the first crossing at the solver's declared
    distance resolution. If quadrature, subdivision, float resolution, or
    the work or evaluation ceiling cannot distinguish the target, direct MDE
    calculation refuses with
    `power.minimum_detectable_effect.numerical_resolution`; an otherwise
    valid sample-size or achieved-power result remains available and records
    the missing companion MDE through `mde_unavailable_reason`.

    The Gaussian recursion is bounded to 256 planned looks, 300,000,000
    estimated kernel-work units, and 4,096 nodes. A design beyond these
    limits is refused with `power.boundary_crossing_quadrature`,
    `power.sequential_sample_size`, or
    `power.minimum_detectable_effect.numerical_resolution`, as appropriate;
    it is not assigned a midpoint estimate or called statistically
    unattainable. Use fewer planned looks or group looks only if that changed
    schedule matches the design. The Gaussian log-SE planning approximations
    do not estimate power for the continuous `AsymptoticMean` construction.

## Reading `PowerResult`

| Field | Meaning |
|---|---|
| `n_per_arm` | Assigned units in the treatment arm |
| `n_total` | Assigned units across the two-arm contrast |
| `power` | Planned power at the returned integer size, read with `power_basis` |
| `power_lower`, `power_upper` | The computed range holding an enumerated binomial power; `None` for a closed-form or sequential plan |
| `power_basis` | `exact`, `lower_bound`, or `asymptotic`; see [above](#what-power-and-power_lower-mean) |
| `numerical_qualification` | `closed_form_model_only_v1` for the asymptotic model; `scipy_special_function_error_model_conditional_v1` for either enumerated basis (conditional on the deployed SciPy/Boost special-function error model, not a cross-build proof) |
| `mde_relative` | Detectable relative effect at the target power, rescaled for `compliance`; `None` when no admissible numeric answer is certified |
| `mde_unavailable_reason` | `unattainable`, `unrepresentable`, or `numerical_resolution` when `mde_relative` is `None`; otherwise `None` |
| `effective_var` | Per-unit control-arm variance for asymptotic planning, after decision-method reductions and cluster design effect; sensitivity-only CUPED receives no credit, so this can differ from the caller's `Baseline.effective_var`. For a `QuantileBaseline`, its pilot-implied variance. Enumerated binomial power uses event rates and counts, not this variance. |
| `n_clusters_per_arm`, `n_clusters_total` | Ceiled randomization-cluster counts, or `None` for unit randomization |
| `n_triggered_per_arm`, `n_triggered_total` | Analyzed units when `trigger_rate < 1`, otherwise `None` |
| `expected_n_total` | Expected sequential stopping size, or `None` for fixed horizon |
| `inference_to_declare` | The `InferenceSpec` to declare at runtime registration for the same boundary, or `None` for fixed horizon |

Read the result at its actual integer size and check `power_basis` and
`numerical_qualification` before quoting it as a runtime-power claim. See
[Reading results](reading-results.md) and [Multiplicity](multiplicity.md)
for interpreting the analysis the plan feeds.

## Switchback contrasts

Switchback planning uses a `ContrastDecisionProcedure` and a frozen
`SwitchbackBaseline`, usually `analysis.planning_baseline(metric)` on a pilot
built by `Analysis.from_switchback_panel(...)`, or
`source.planning_baseline(source.metrics[0])` on a source built by
`increment.frame.from_switchback_panel(...)`; both return the same baseline.
These approximate functions are exported from both `increment` and
`increment.power`:

```text
switchback_achieved_power(n, delta, baseline, procedure)
switchback_minimum_detectable_effect(n, baseline, procedure, *, target_power)
switchback_required_blocks_or_units(delta, baseline, procedure, *, target_power)
```

`n` counts independent **units** for independent Bernoulli orders and
independent **blocks** for shared schedules, with `df = n - 1` and
`2 <= n <= 2**53`. Neither cycles nor roster size multiply `n`; the summary
already averages within the independent replicate, and one cycle per unit is
valid. `delta` and the MDE are **absolute** effects; there is no relative or
standardized MDE. `power` already multiplies by `admission_probability`, the
exact probability a shared-schedule design at that block count realizes both
cycle orders (`1.0` for an independent-order baseline), so every solver
reports or targets *available* power.

!!! warning "A pilot fit is an approximation, not a certificate"
    The noncentral-t reference is a planning approximation, not an exact
    finite-branch randomization law. Predictions condition on the fitted
    moments and reference effect (`moment_source="pilot_estimated"`) and do
    not integrate uncertainty from estimating the pilot; a pilot sample
    variance is not a prospective variance envelope, and the fit neither
    establishes population moments nor validates no residual carryover.
    Conversion aggregation (`"any"`) is refused with
    `power.switchback.bounded_model_required`.

For a two-sided MDE query, `procedure.preferred_direction="decrease"`
searches below the null, including decrease-only `effect_bounds`;
`"increase"`, `"neutral"`, or an unspecified direction searches above it.
This does not change two-sided alpha; one-sided queries follow their declared
`alternative`.

??? info "Statistical details: the switchback baseline"
    A producer supplies the original `SwitchbackAssignment` (including its
    window, probability, law and assignment grain), `metric`, distinct
    `control_group` and `treatment_group`, matching `aggregation`/`estimand`,
    and the identifying assumption
    `no_residual_carryover_after_discarded_steps`. Independent orders require
    positive integer `cycles_per_unit` and no roster; shared schedules
    require a nonempty, duplicate-free `shared_roster: tuple[str, ...]` and
    no cycle count. The metric must match the decision procedure. The
    procedure has no group label fields; the producer must preserve the
    control-to-treatment orientation.

    The numerical fields are `delta_ref`, `sd_a`, `sd_g`, and `rho`. They
    encode centered `VarA = sd_a²`, `VarG = sd_g²`, `CovAG = rho*sd_a*sd_g`
    at the independent grain. `A` is the reference-effect contribution and
    `G` its slope under a constant additive retained-window shift, with
    `E[G] = 1`. Thus for `d = delta - delta_ref`, variance is
    `VarA + 2*d*CovAG + d²*VarG`, and `SE = sqrt(variance/n)`. All four
    inputs are finite; SDs are nonnegative and `-1 <= rho <= 1`. Singular
    PSD covariance is valid. Set `rho=0` if either SD is zero. A true zero
    variance at the requested effect is unavailable. Changing the
    assignment, window, cycles, roster or shift model requires a newly
    derived summary. The power layer takes no raw observations and does no
    source reduction.

    Ordinary planning fits the reference effect from the pilot mean. On the
    source, a caller may provide `delta_ref=reference_effect` for a
    deliberately matched scenario, but need not provide specialist reference
    moments or a population variance certificate. For ordinary unit-cycle
    inference, source construction chooses the qualified
    `UnitCycleTApproximation` from the declared assignment. The fit reduces
    the same retained-window contributions as inference and preserves their
    joint covariance with the assignment-dependent shift slopes; at least two
    independent units or blocks are required.

    The `effect_model="constant_additive"` contract describes additive total
    effects: bounds on the effect do not establish this variance model for
    binary outcomes. Optional finite `effect_bounds` may restrict the
    additive total-effect domain and must contain `delta_ref`.

??? info "Statistical details: `SwitchbackPowerResult` and MDE inversion"
    `SwitchbackPowerResult` retains `baseline`, `procedure`, `n`, `df`,
    `delta`, `standard_error`, `power`, `mde_abs`, and
    `admission_probability`. Every missing number has its own
    `*_unavailable_reason` (the MDE reason is `mde_unavailable_reason`).
    Achieved-power and required-N queries leave `mde_abs=None` with reason
    `not_requested`; neither uses an implicit target power. An MDE query
    reports the first representable favorable absolute effect, including the
    null itself when its variance is positive and its power meets the
    explicit target. The result and its nested metadata are immutable and
    serialize to JSON with numeric nulls.

    MDE inversion classifies the variance quadratic analytically and solves
    its noncentrality equation with a cancellation-safe quadratic formula.
    It checks both the returned public float and its predecessor through
    achieved power. `unattained` means the target cannot be attained in the
    declared model/domain (including equality with an excluded asymptote);
    `unrepresentable` means a mathematical solution lies outside the
    representable planning domain; `numerical_resolution` means the
    numerical reference could not resolve the answer (for a runtime-binomial
    plan, also that the effect search would have stored more cells than the
    replay bound allows). `degenerate_zero_variance` identifies deterministic
    effects, and `non_favorable_effect` refuses required-N queries at or
    inside the null.

For calibrated unit-cycle inference, a prospective `UnitCycleVarianceEnvelope`
supplies a certified power lower bound, and `UnitCycleJointLaw` adds a
complete population model for `unit_cycle_model_power`,
`unit_cycle_model_mde`, and `unit_cycle_model_required_units`. Their Monte
Carlo bounds and search statuses are distinct from moment-t approximations;
see the [switchback guide](switchback.md) for the model, admission budget,
numerical availability and certification limits.
