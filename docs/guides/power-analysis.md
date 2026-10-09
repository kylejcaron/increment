# Power analysis and MDE for A/B tests

Increment's arm power solvers calculate required sample size, achieved power, or minimum detectable relative effect from a control-arm baseline and declared alternative-arm assumptions. Mean-like plans use the estimator's log-ratio variance model; eligible fixed-horizon conversion and retention plans follow the runtime's count-based decision route or its dense closed-form planning model.

## Procedure-first API

Every parallel-arm scalar solver requires a validated
`increment.estimation.arm_contract.ArmPlanningProcedure`. The procedure carries
the assignment, analysis axes, estimand, metric capabilities, decision rule,
compiled inference, decision method, sensitivity methods, family expansion, and
whether a prior is present. This keeps planning on the same contract as the
analysis that will consume the result.

```python
from increment import Baseline, PowerDesign
from increment.power import (
    ArmPlanningProcedure,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)

baseline = Baseline(mean=20.0, var=400.0)
# One fixed-horizon plan for an ordinary parallel test. Pass `comparisons=`
# to split alpha across arms, or `clustered=True` for a clustered design;
# build ArmPlanningProcedure directly for anything outside that.
procedure = ArmPlanningProcedure.standard("mean")
size = required_sample_size(
    relative_lift=0.02,
    baseline=baseline,
    procedure=procedure,
    design=PowerDesign(power=0.80, allocation=0.5),
)
power = achieved_power(25_000, 0.02, baseline, procedure)
mde = minimum_detectable_effect(25_000, baseline, procedure)
```

The solver signatures are:

```text
required_sample_size(relative_lift, baseline, procedure, design=None, *, planned_looks=None)
achieved_power(n_per_arm, relative_lift, baseline, procedure, design=None, *, planned_looks=None)
minimum_detectable_effect(n_per_arm, baseline, procedure, design=None, *, planned_looks=None)
```

`procedure` is required; `inference=`, `alpha=`, `alternative=`, and method-list
arguments are not solver parameters. Put those decisions in the procedure. A
`PowerDesign` supplies only numeric solver controls: target `power` (default
`0.80`) and treatment `allocation` (default `0.5`).

`n_per_arm` is the treatment-arm size. The control-arm size is derived from
`PowerDesign.allocation`; `PowerResult.n_total` is the assigned total across the
contrast. Each solver returns a complete `PowerResult` at the actual integer
sample size, including `power`, `mde_relative`, and `effective_var`.

The `Baseline` constructors cover common pilot inputs:

- `Baseline(mean=, var=)` for control-arm mean and per-unit variance;
- `Baseline.from_proportion(p)` for conversion/retention assumptions;
- `Baseline.from_summary(summary_stats)` for a pilot variance with its
  confidence-bound inflation;
- `Baseline.from_ratio(...)` for ratio metrics' linearized numerator,
  denominator, and covariance moments;
- `Baseline.from_absorption(mean, var, icc)` when planning for a measured
  one-way factor absorption reduction.

If you already have an `Analysis` over a prior or pilot experiment's data
(or an in-flight experiment's pre-treatment data) -- not the future test
you are about to plan -- `analysis.planning_baseline(metric)` reads that
prior experiment's own data and declared design, and builds the `Baseline`
directly from it. Outcome moments describe the analyzed population: under a
declared `trigger` that is the triggered population, with `trigger_rate` its
observed share of the assigned control arm; with no trigger declared, the
assigned population at `trigger_rate=1.0`. A declared trigger whose evidence
the source does not carry (a unit-day artifact published without the
`trigger_population` and `assignment_counts` extensions) refuses with
`analysis.planning_baseline.trigger_evidence_unavailable`. From that
population's control arm: per-unit mean/var (or a ratio's linearized
moments); `cuped_rho` when this metric declares CUPED, as the control arm's
variance reduction under the runtime's own CUPED fit; `compliance` from the
source's design-level uptake when the design is `Encouragement`; and
`cluster_icc`/`cluster_size_cv` from contributing clusters when a `cluster` is
declared, with ICC on the analyzed score (for a ratio, `y - R*y_den`).
Recruitment metadata stays on the assigned population: `avg_cluster_size` is
assigned control units per randomized cluster, and `cluster_participation`
is contributing control clusters divided by assigned control clusters.
The artifact route also needs `cluster_identity` for a clustered pilot.
Both clustered trigger routes require the metric's outcomes for every
trigger-eligible control unit in the pilot. Missing metric rows raise
`analysis.planning_baseline.trigger_metric_population_unavailable`, with
observed and eligible counts. Complete unfinished windows. If an outcome is
structurally undefined, such as `avg_event` for a unit with no events, choose
a metric defined for every eligible unit; waiting or republishing cannot
define that outcome.
If you have no prior experiment or pilot
data at all, there is nothing for `planning_baseline` to read; use
`Baseline`'s other constructors directly with an assumed or
externally-sourced value. On a switchback analysis the same call returns the
pilot-fitted `SwitchbackBaseline` the `switchback_*` solvers take (see
[switchback contrasts](#switchback-contrasts)).

For a quantile metric, `planning_baseline(metric)` returns a
`QuantileBaseline` instead. There is no internal statistical quantity to
set -- at each candidate sample size the solver re-evaluates the runtime's
own order-statistic construction over the pilot's real per-unit control
values, not a single number you could otherwise be asked to supply; its
`var` is the per-unit variance the pilot's standard error implies. It has
no CUPED route (quantile metrics have no runtime CUPED
construction to credit), and `required_sample_size`/`achieved_power`/
`minimum_detectable_effect` echo `PowerResult.planned_metric_name`/
`planned_quantile` whenever the baseline is one, so reusing one metric's
baseline to plan a different metric is visible in the answer rather than
silently mis-sizing a design. A `QuantileBaseline` passed with
`ArmPlanningProcedure.standard()` plans a quantile metric; with another
declared metric type it is refused.

Quantile planning follows the quantile readout: at each candidate size it
applies the readout's own half-width rule (`quantile_half_width`) to a
bracket projected from the pilot, and at the pilot's own size the planned
standard error is the readout's on the pilot. A pilot whose bracket holds no
tied values (continuous data) plans the classical standard error, shrinking
as `1/sqrt(n)`. On rounded data the readout's point and bracket ends are
recorded values, so planning reads them from the pilot's recorded
distribution. At any other size than the pilot's, where the quantile falls
within its recording cell is known only to the pilot's precision, and that
position decides whether a narrow bracket spans one cell or two, so the
planned standard error averages over it, integrating exactly over the
recorded values the bracket can reach: between adjacent sizes the average
moves by no more than the mass a bracket end can carry across one recorded
value. The averaged shift is that of a sample sharing its units with the
pilot, so it vanishes at the pilot's size and is smallest near it: a planned
size within about a quarter of the pilot's size leans most on the pilot's
own bracket, and on rounded data measured power there ran 0.71-0.76 against
a planned 0.80. The readout's interval never narrows below the gap from the quantile
to the next recorded value, so power has a ceiling below one: a target above
it is refused with `power.quantile_size_search_unreachable`, whose context
names the largest power the search found at any size or in the large-sample
limit (`maximum_power`, `limiting_condition="recording_grid"`). Power is not
monotone in the sample size: it passes through the pilot's own power at the
pilot's size, an arm whose bracket stops spanning twelve repeated values
switches to the readout's wider tied interval so power dips past that size,
and on a coarse grid it wavers slightly with the readout's own bracket
ranks. The search evaluates a fine grid of sizes in order and settles the
crossing between the first grid size reaching the target and the one
before, so the planned size reaches the target while the size below it does
not; a smaller size can reach the target only within that rank-level waver.
A size at which the projected bracket reaches a non-positive recorded value
has no log-scale interval, as at the readout, and `achieved_power` refuses
it with the readout's own code; the size search passes over such sizes. The
quantile readout tests two-sided against a zero null and has no breakouts,
so planning refuses a one-sided alternative, a shifted null or a guardrail
role with `readout.metric.quantile_alternative`, and the
`segment_pairwise_*` solvers refuse a quantile metric with
`readout.metric.quantile_breakout` -- the readout's own codes.

The three arm solvers evaluate each arm's log-scale variance at its own mean
under the alternative. For mean-like metrics, planning assumes the treatment
arm keeps the control arm's absolute effective variance. For conversion and
retention metrics the runtime does not decide with this model (below), it
rescales that variance by the Bernoulli shape at the implied treatment rate.
For quantile metrics, the treatment arm's own
order-statistic construction is evaluated at its own analyzed count,
reusing the pilot's recording grid (a property of the instrument, not of
the hypothesized shift) -- an exact identity under this module's
multiplicative-effect convention, not the "same absolute variance"
approximation mean-like metrics use. These are planning assumptions, not
claims about every future data-generating process. Bounded baselines,
nulls, and alternatives must imply rates in `(0, 1]`; a requested rate
above 1 is refused. Every such result reports `power_basis="asymptotic"`.

### Conversion and retention: planning follows the runtime's route

An unadjusted, unclustered, fixed-horizon conversion or retention plan (no CUPED, no
absorbed factor, no winsorization, no prior) is routed at runtime by its counts
(`conversion_inference`, default `"auto"`): a count pair whose four per-arm success and failure
counts are all dense for the tail allocation is decided by the delta-method test, every other
pair by the exact Berger-Boos risk-ratio test on the raw counts. Planning reports the
probability that this union rejects, summed over the binomial count law at the analyzed integer
counts: the delta-method decision on the routed rectangle of count pairs (the production
decision itself, each pair decided by the runtime's own calculation) plus the replayed finite-sample decision on the rest. It is neither route's power
and not a function of the two: a plan whose counts straddle the threshold is not "the smaller
of the two". The count law is integrated over windows that leave out at most about `1e-12` of
mass. `ArmPlanningProcedure.standard(..., conversion_inference="finite_sample")` plans the
replay at every size (refused for a mean, clustered or sequential plan, which the finite-sample
route does not serve). Power depends on the baseline rate alone; `var` does not enter.

Public `power` is a model-based point probability, not the lower endpoint of a
numerical enclosure:

- `power_basis="exact"`: the computed rejection mass from the count-law
  enumeration. Internal bounds account for summation error, omitted support
  and unresolved count pairs. The point is published only when the computed
  enclosure resolves to absolute error at most `1e-6`, conditional on the
  deployed SciPy/Boost special-function error model; this is not a cross-build
  floating-point proof. `"exact"` describes the decision-law calculation, not
  error-free arithmetic or a finite-sample guarantee for the hybrid delta-method
  route.
- `power_basis="approximate"`: an unresolved diagnostic with no claim on runtime
  power. Materially unresolved results are refused rather than published.
- `power_basis="asymptotic"`: the closed-form model above, where counts route to
  the delta method with near certainty (unrouted mass at most `1e-6`) and the
  lattice has more than 100,000 cells. This is the model's own probability,
  not an enumeration of the runtime's decision. The dense validation tolerance
  remains `0.005`; it is a comparison criterion, not a universal error bound
  for every design or data-generating process.
- If the enumeration cannot resolve the probability to `1e-6`, the public
  request refuses with `power.binomial_probability_unresolved` (context
  `lower`, `upper`, `error`, `tolerance`). It does not publish a lower endpoint,
  midpoint or substitute approximation. Internal diagnostic enclosures may remain
  wide; they are not public `PowerResult` fields or user tuning parameters.

Enumeration has deterministic work and memory limits, not a timing-dependent
answer. Delta-decision enumeration is bounded at 100,000 count pairs, and
finite-sample replay at 150,000 directional replays per evaluation. A solve
stores at most 10,000,000 control/treatment count cells across its null and
evaluated alternative windows. A supplied effect, an MDE search and each
`power_curve` row are separate solves: each is bounded by the cells of its own
requests, and an MDE search's interval bounds read only what that search decided.
A count pair's decision is the runtime's own, the same whichever request decides
it, so the solves of one design share those decisions: a companion MDE search
copies the cells the supplied effect or the sizing already decided instead of
replaying them, and replays of each interval's far endpoint only the cells its
bound depends on; an endpoint whose power and cells no bound reads (the far end
of the admissible effects, the far endpoints of wide intervals above the answer)
is never replayed, while one a curvature term reads is. Exceeding the cell bound raises
`power.binomial_replay_bound_exceeded`; an MDE search that reaches it can leave
a companion `mde_relative=None` with `mde_unavailable_reason="numerical_resolution"`.
These are planner limits: the finite-sample runtime ceiling remains
1,000,000,000 units per arm. A dense `auto` plan using the closed form does
not enumerate that lattice.

The runtime's bounded finite-sample search can produce nonmonotone rejection
decisions. Planning does not infer a count pair's decision from its neighbour.
Likewise, power need not be monotone in sample size or effect:

- `required_sample_size` returns a size meeting the target under the selected
  planning calculation, not a proof that no smaller size can reach it.
- `minimum_detectable_effect` searches for the earliest detectable region to
  an effect tolerance of `1e-8` absolute plus `1e-8` relative, measured on the
  relative-effect scale, not the first IEEE-representable effect. It
  cannot silently skip a materially earlier unresolved band. The result's
  `power` is evaluated at the reported effect, not copied from the target.
- A target excluded over every admissible effect is `unattainable`. If the
  search cannot distinguish detectability or exclude an earlier region at its
  numerical resolution, a companion MDE is unavailable with
  `numerical_resolution`; a standalone call refuses with
  `power.minimum_detectable_effect.numerical_resolution`. A supplied effect
  whose point power is resolved can still be returned without a companion MDE.

With a shifted null and partial compliance, an initial band can imply
unrepresentable absolute lifts even though farther effects are representable.
The planner bounds that band's power before excluding it. An unresolved band
yields `numerical_resolution`, not a claim that a detectable effect exists
outside the representable domain.

All conclusions are conditional on the selected decision rule and planning
model. Numerical bounds address evaluation accuracy, not model misspecification
or calibration. Calibration remains a separate, deferred diagnostic rather
than a prerequisite on the planning path.

#### Comparing with statsmodels

The official [`TTestIndPower.power`](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.power.html)
and [`NormalIndPower.power`](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.power.html)
APIs return model-based point rejection probabilities, not lower numerical
bounds. Increment uses the same point-probability interpretation, but that does
not imply numerical parity: those APIs plan independent-sample t- and z-tests,
respectively, with standardized mean differences. Compare only after aligning
the decision rule, effect scale, baseline/variance assumptions, alpha, sidedness
and arm allocation. A relative-lift binomial risk-ratio test can legitimately
have different power from either test.

The current binomial planning model is `hybrid_finite_plus_delta_v3`. Persisted
results labelled `hybrid_finite_plus_delta_v2` or its v1 predecessor must be
recomputed; they do not acquire point-power semantics by relabelling.

A design the runtime refuses in full decides no count pair, so it has no power to plan, and it
is never replayed: `achieved_power`, `minimum_detectable_effect` and every `power_curve` row
that reaches one refuse it, with `power.binomial_arm_ceiling_exceeded` (context `n_c`, `n_t`,
`max_arm_size`) for an arm beyond the runtime's one-billion unit ceiling and
`power.binomial_tail_level_unrepresentable` (the context below, `scope="requested"`) for a
nuisance budget (`alpha / 32`) below the endpoint solver's floor (an alpha under `3.2e-8`) or a
tail level its float margin dominates (a two-sided alpha below about `9.5e-7` at a billion
units per arm; see the limitations page). Under a dominating margin the runtime still decides
a count pair whose control count alone rejects a shifted null (its Clopper-Pearson lower bound
above `1 / (1 + null_lift)`, for a two-sided or "less" test); the context's `decided_from` is
the smallest such count (`None` when there is none). A `finite_sample` plan whose control
window at the baseline rate holds smaller counts, which the runtime refuses, is refused rather
than planned with them as non-rejections; an `auto` plan is refused when the counts the count
rule keeps on the finite-sample route below `decided_from` carry more than half of `1e-6`, and
otherwise carries them as undecided mass. The margin grows with the arm, so a smaller size of
the same alpha can be planned; the solver floor refuses every size.

In addition to numerical-resolution refusals, `required_sample_size` can end a
replayed search with the following resource or decision-domain refusals:

| Code | When | Context |
|---|---|---|
| `power.binomial_replay_bound_exceeded` | the search reached its ceiling: about 1/128 under the crossing of a bisection for the largest size whose null and supplied-effect rectangles fit the replay bound (the cell count is not monotone in the size, so a size above the ceiling may fit and may reach more) | `power`, `power_reached`, `n_c`, `n_t`, `p_c`, `p_t` (when the alternative exceeds), `cells`, `max_cells`, `max_arm_size` |
| `power.binomial_size_search_unreachable` | the search reached the runtime's arm ceiling, or the size where the float margin starts dominating the tail level, without reaching the target | `power`, `maximum_power`, `n_per_arm`, `max_arm_size` |
| `power.binomial_tail_level_unrepresentable` | the decision is refused even at the smallest design: the nuisance budget is below the solver floor, or the float margin already dominates the tail level there, so no size reads the treatment arm; raised before any search | `alpha`, `beta`, `tail_alpha`, `margin`, `n_c`, `n_t`, `p_c`, `decided_from`, `solver_floor`, `cause` (`solver_floor` or `float_margin`), `scope` (`smallest`) |
| `power.binomial_arm_ceiling_below_smallest_design` | the allocation is so lopsided that the smallest design already has an arm above the runtime's ceiling; raised before any search | `n_t`, `n_c`, `allocation`, `max_arm_size` |

The `segment_pairwise_*` solvers are separate: they retain a baseline-only
four-arm variance approximation. Do not use the three arm solvers as numerical
oracles for that model.

`cuped_rho` reduces planning variance by `(1 - cuped_rho**2)` only when the
effective decision method uses CUPED. A CUPED sensitivity readout does not
reduce an unadjusted decision's sample requirement. Mixed power/MDE curves
apply this rule independently to each procedure.
`compliance` rescales a diluted encouragement effect, and `trigger_rate`
accounts for the fraction of assigned units entering a triggered analysis. For clustered
randomization, `avg_cluster_size` is the assigned mean size; `cluster_icc`
and `cluster_size_cv` describe analyzed scores and contributing cluster sizes.
For a triggered cluster design, also supply `cluster_participation`: the
fraction of recruited clusters that contributed at least one analyzed unit
in a comparable pilot. `planning_baseline()` derives it automatically on the
definitions and sufficiently published artifact routes.

Writing assigned mean size as $m$, unit trigger fraction as $q$, contributing
cluster fraction as $p$, and analyzed size CV as $c$, the analyzed mean size is
$m q/p$ and the design effect is $1 + ((1+c^2)m q/p - 1)\rho$.
For example, 100 assigned units in ten clusters with 20 triggered units have
$m=10$, $q=0.2$. If five clusters contribute, set $p=0.5$, not $0.2$:
the analyzed mean size is four. At ICC 0.2 and equal analyzed sizes, the
design effect is 1.6; if all ten clusters contribute, it is 1.2 instead.
Measure the CV only across contributing clusters. Do not substitute the
variation in assigned sizes.

Results report recruited cluster counts in `n_clusters_per_arm` and
`n_clusters_total`, using assigned mean size. Inference uses the floored
expected represented counts $K'_a=\lfloor K_a p\rfloor$, with roundoff
allowance at integer boundaries, and requires at least two per arm.
The relative t reference uses $\min(K'_T-1,K'_C-1)$ degrees of freedom.
Analysis retains separate Welch degrees of freedom for its additive interval.
This is a pilot-conditioned planning approximation, not a guarantee that a
future trial will realize the pilot participation rate or achieve finite-sample
coverage. There is no ten-cluster admission floor; small-cluster calibration
remains a limitation.
The noncentral-t log-scale power calculation is a planning approximation,
not an exact forecast of Fieller-set rejection under weak denominators.
Build the procedure with `ArmPlanningProcedure.standard(metric_type)` and pass
the same `Baseline` to the solver call. CUPED credit follows the effective
decision method; compliance, triggering, and factor absorption are derived
from non-default baseline fields. A procedure explicitly built with a
conflicting axis raises `CapabilityError` naming the field and the conflict. CUPED composes with
`inference=InferenceSpec(kind='asymptotic_mean')`.

A secondary metric plans at `role='secondary', secondaries=<count>,
q=<FDR target>` -- sized as a conservative Bonferroni-at-`q/m` bound
(`m` = secondaries x comparisons under fixed-horizon inference, secondaries
alone under `inference=InferenceSpec(kind='asymptotic_mean')`, which supports
one treatment arm). A guardrail plans at `role='guardrail',
preferred_direction='increase'/'decrease'` (or an explicit one-sided
`alternative`) -- always full alpha, never arm-split.

Segment-pairwise planning retains its separate four-arm, baseline-only Normal
approximation with cluster design effects. It enforces the same two-per-arm
structural minimum within each segment; it does not share the arm solver's
t reference or its numerical power model.

## Curves

`power_curve` also requires one procedure or a sequence of procedures. Provide
exactly one of `relative_lift` and `target_power`; the omitted quantity is
solved for each Cartesian-grid row. The procedure's compiled decision alpha and
inference are used for every row.

```python
from increment.power import power_curve

curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    relative_lift=[0.01, 0.02, 0.05],
    baseline=baseline,
    procedure=procedure,
    units_per_week=50_000,
)
print(curve.to_frame())

mde_curve = power_curve(
    n_per_arm=[5_000, 10_000, 20_000],
    target_power=[0.80, 0.90],
    baseline=baseline,
    procedure=procedure,
)
```

The full curve signature is:

```text
power_curve(*, n_per_arm, baseline, procedure, relative_lift=None,
            target_power=None, design=None, units_per_week=None,
            planned_looks=None, max_workers=None)
```

Rows preserve input product order, including duplicate values. `units_per_week`
adds `duration_days` based on assigned units; omit it to leave duration unset.
For sequential procedures, `expected_n_total` and
`expected_duration_days` describe expected early stopping in addition to the
worst-case `n_total`/`duration_days`.

## Fixed and sequential planning

A fixed procedure uses the fixed-horizon reference. Sequential planning is
declared the same way the runtime reads it:
`ArmPlanningProcedure.standard(..., inference=InferenceSpec(kind="asymptotic_mean"))`
builds the always-valid boundary internally, tuned identically to the
runtime's own boundary at every information fraction, including one-sided
decisions. `planned_looks` controls the equal-look repeated-look calculation
(omitted: 14 looks), and `required_sample_size` solves for the
boundary-crossing power at those looks. The result's
`PowerResult.inference_to_declare` is the exact `InferenceSpec` to pass at
runtime registration, with `expected_decision_sample_size` set to the solved
`n_total`, so planning and the runtime never silently disagree on N:

```python
from increment import InferenceSpec
from increment.power import ArmPlanningProcedure

procedure = ArmPlanningProcedure.standard("mean", inference=InferenceSpec(kind="asymptotic_mean"))
sized = required_sample_size(0.02, baseline, procedure, planned_looks=14)
# Declare the same boundary at runtime registration:
runtime_inference = sized.inference_to_declare
```

`AlwaysValid` carries a registered raw-likelihood runtime policy. Planning APIs
refuse it with `sequential.route.unsupported`. The `asymptotic_mean` boundary
used for planning is the boundary the runtime executes (see
[sequential inference](sequential-inference.md)). Planning does not use a
different group-sequential Gaussian boundary.

Sequential inversion recomputes the alternative-arm standard error and
boundary at every candidate. The crossing evaluator uses composite
Gauss–Legendre integration with propagated derivative-remainder, floating
arithmetic, and normal-tail bounds; expected information is bounded
independently. Internal arm-planning paths carry the log variance into both
the boundary and standardized drift, avoiding a zero/infinite intermediate at
extreme scales. An MDE is returned only when those bounds certify the first
crossing at the solver's declared distance resolution. If quadrature,
subdivision, float resolution, or the work or evaluation ceiling cannot
distinguish the target, direct MDE calculation refuses with
`power.minimum_detectable_effect.numerical_resolution`. An otherwise valid
sample-size or achieved-power result remains available and records the missing
companion MDE through `mde_unavailable_reason`.

The Gaussian recursion is bounded to 256 planned looks, 300,000,000 estimated
kernel-work units, and 4,096 nodes. A design beyond these limits is refused
with `power.boundary_crossing_quadrature`,
`power.sequential_sample_size`, or
`power.minimum_detectable_effect.numerical_resolution`, as appropriate; it is
not assigned a midpoint estimate or called statistically unattainable. Use
fewer planned looks or group looks only if that changed schedule matches the
design. This prospective route does not size registered likelihood policies.

Combining a CUPED method with non-fixed inference is not supported. Such a
procedure is refused with `arm.adjustment.sequential_cuped`. Clustered
sequential planning is also refused; use fixed-horizon clustered planning (valid for one planned analysis, not repeated looks) or
unit-randomized sequential planning.

## Reading `PowerResult`

| Field | Meaning |
|---|---|
| `n_per_arm` | Assigned units in the treatment arm |
| `n_total` | Assigned units across the two-arm contrast |
| `power` | Planned power at the returned integer size under `power_basis`; an admitted finite-sample point is published only when its computed enclosure resolves to absolute error at most `1e-6`, conditional on the deployed SciPy/Boost special-function error model. An unresolved diagnostic is not runtime-power evidence; an asymptotic result is the model's own probability. |
| `power_basis` | `asymptotic` (log-ratio model, for counts the runtime takes the delta-method route with near certainty in a lattice too large to enumerate), `exact` (the runtime's decision summed over the count lattice), or `approximate` (an unresolved diagnostic, making no runtime-power claim) |
| `numerical_qualification` | `closed_form_model_only_v1`, `scipy_special_function_error_model_conditional_v1`, or `unclaimed_approximation_diagnostic_v1`, respectively; the finite-sample label is conditional on the deployed SciPy/Boost special-function error model, not a cross-build proof |
| `mde_relative` | Detectable relative effect at the target power, rescaled for `compliance`; `None` when no admissible numeric answer is certified |
| `mde_unavailable_reason` | `unattainable`, `unrepresentable`, or `numerical_resolution` when `mde_relative` is `None`; otherwise `None` |
| `effective_var` | Per-unit control-arm variance for asymptotic planning, after decision-method reductions and cluster design effect; sensitivity-only CUPED receives no credit, so this can differ from the caller's `Baseline.effective_var`. For a `QuantileBaseline`, its pilot-implied variance. Exact/approximate binomial power uses event rates and counts, not this reported variance. |
| `n_clusters_per_arm`, `n_clusters_total` | Ceiled randomization-cluster counts, or `None` for unit randomization |
| `n_triggered_per_arm`, `n_triggered_total` | Analyzed units when `trigger_rate < 1`, otherwise `None` |
| `expected_n_total` | Expected sequential stopping size, or `None` for fixed horizon |
| `inference_to_declare` | The `InferenceSpec` to declare at runtime registration for the same boundary, or `None` for fixed horizon |

Power planning does not make an analysis interval valid under unplanned
continued peeking. Use a matching sequential/always-valid procedure when the
analysis plan commits to repeated looks.

Continuous sequential `AsymptoticMean` inference is not supported by the power
solvers. It uses actual finalized arm counts and a shifted-contrast variance;
passing it refuses with `sequential.route.unsupported`. The Gaussian log-SE
planning approximations do not estimate power for this construction. See
[continuous sequential means](sequential-inference.md#continuous-scalar-means).


## Switchback contrasts

Moment-t switchback planning uses `ContrastDecisionProcedure` and a frozen
`SwitchbackBaseline`. These approximate functions are exported from both
`increment` and `increment.power`:

```text
switchback_achieved_power(n, delta, baseline, procedure)
switchback_minimum_detectable_effect(n, baseline, procedure, *, target_power)
switchback_required_blocks_or_units(delta, baseline, procedure, *, target_power)
```

`n` counts independent **units** for independent Bernoulli orders and independent
**blocks** for shared schedules. Each candidate uses `df = n - 1`, with `n >= 2`.
One cycle per unit is valid. Neither cycles nor roster size multiply `n`; the
summary already includes averaging within the independent replicate. The
noncentral-t reference is a planning approximation, not an exact finite-branch
randomization law. The integer reference range is `2 <= n <= 2**53`.

A producer supplies the original `SwitchbackAssignment` (including its window,
probability, law and assignment grain), `metric`, distinct `control_group` and
`treatment_group`, matching `aggregation`/`estimand`, and the identifying
assumption `no_residual_carryover_after_discarded_steps`. Independent orders
require positive integer `cycles_per_unit` and no roster; shared schedules
require a nonempty, duplicate-free `shared_roster: tuple[str, ...]` and no cycle
count. The metric must match the decision procedure. The procedure has no group
label fields; the producer must preserve the control-to-treatment orientation.

The numerical fields are `delta_ref`, `sd_a`, `sd_g`, and `rho`. They encode
centered `VarA = sd_a²`, `VarG = sd_g²`, `CovAG = rho*sd_a*sd_g` at the independent
grain. `A` is the reference-effect contribution and `G` its slope under a
constant additive retained-window shift, with `E[G] = 1`. Thus for
`d = delta - delta_ref`, variance is `VarA + 2*d*CovAG + d²*VarG`, and
`SE = sqrt(variance/n)`. All four inputs are finite; SDs are nonnegative and
`-1 <= rho <= 1`. Singular PSD covariance is valid. Set `rho=0` if either SD is
zero. A true zero variance at the requested effect is unavailable. Changing the
assignment, window, cycles, roster or shift model requires a newly derived
summary. The power layer takes no raw observations and does no source reduction.

The centered moments are obtained from a validated pilot created by
`increment.frame.from_switchback_panel(...)` or
`Analysis.from_switchback_panel(...)`.  Ordinary planning fits the
reference effect from the pilot mean: `analysis.planning_baseline(metric)`
on the analysis, or `source.planning_baseline(source.metrics[0])` on the
source; both return the same `SwitchbackBaseline`.  On the source, a caller may provide
`delta_ref=reference_effect` for a deliberately matched scenario, but need not
provide specialist reference moments or a population variance certificate.
For ordinary unit-cycle inference, source construction chooses the qualified
`UnitCycleTApproximation` from the declared assignment.  A pilot sample
variance is not a prospective variance envelope.
This reduces the same retained-window contributions as inference and preserves
their joint covariance with the assignment-dependent shift slopes. At least two
independent units or blocks are required. The baseline records
`moment_source="pilot_estimated"`. Predictions condition on the fitted moments
and reference effect; they do not integrate uncertainty from estimating the pilot.
The fit neither establishes population moments nor validates no residual carryover.

The `effect_model="constant_additive"` contract describes additive total effects.
Conversion aggregation (`"any"`) is refused with
`power.switchback.bounded_model_required`: bounds on the effect do not establish
this variance model for binary outcomes. Optional finite `effect_bounds` may
restrict the additive total-effect domain and must contain `delta_ref`.
For a two-sided MDE query, `procedure.preferred_direction="decrease"` searches
below the null, including decrease-only `effect_bounds`. `"increase"`,
`"neutral"`, or an unspecified direction searches above the null.
This selection does not change two-sided alpha. One-sided queries follow
their declared `alternative`.

`SwitchbackPowerResult` retains `baseline`, `procedure`, `n`, `df`, `delta`,
`standard_error`, `power`, `mde_abs`, and `admission_probability`. Every missing
number has its own `*_unavailable_reason` (the MDE reason is `mde_unavailable_reason`).
Achieved-power and required-N queries leave `mde_abs=None` with reason `not_requested`;
neither uses an implicit target power. An MDE query reports the first representable
favorable **absolute effect**, including the null itself when its variance is
positive and its power meets the explicit target. There is no relative or
standardized MDE. `admission_probability` is the exact probability a shared-schedule
design at that block count realizes both cycle orders (`1.0` for an independent-order
baseline); `power` already multiplies by it, so achieved power, sizing and MDE all
report/target *available* power, not power conditional on the design admitting at
all. The result and its nested metadata are immutable and serialize
to JSON with numeric nulls.

MDE inversion classifies the variance quadratic analytically and solves its
noncentrality equation with a cancellation-safe quadratic formula. It checks
both the returned public float and its predecessor through achieved power.
`unattained` means the target cannot be attained in the declared model/domain
(including equality with an excluded asymptote); `unrepresentable` means a
mathematical solution lies outside the representable planning domain;
`numerical_resolution` means the numerical reference could not resolve the
answer (for a runtime-binomial plan, also that the effect search would have stored
more cells than the replay bound allows). `degenerate_zero_variance` identifies deterministic effects, and
`non_favorable_effect` refuses required-N queries at or inside the null.


For calibrated unit-cycle inference, a prospective `UnitCycleVarianceEnvelope`
supplies a certified power lower bound. `UnitCycleJointLaw` adds a complete
population model for `unit_cycle_model_power`, `unit_cycle_model_mde`, and
`unit_cycle_model_required_units`. Their Monte Carlo bounds and search statuses
are distinct from moment-t approximations; see the [switchback guide](switchback.md)
for the model, admission budget, numerical availability and certification limits.
