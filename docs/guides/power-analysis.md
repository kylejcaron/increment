# Power analysis and MDE for A/B tests

Increment's arm power solvers invert the same log-ratio-of-means variance model used for parallel-arm readouts. They calculate required sample size, achieved power, or minimum detectable relative effect from a control-arm baseline and declared alternative-arm variance assumptions.

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

### Conversion and retention: the runtime's exact binomial decision

An unadjusted, unclustered, fixed-horizon conversion or retention plan (no
CUPED, no absorbed factor, no winsorization, no prior) is analyzed at runtime
by the exact Berger-Boos risk-ratio test on the raw counts. For these plans
`achieved_power`, `minimum_detectable_effect`, `required_sample_size` and
`power_curve` report the probability that this unchanged decision rejects at
the analyzed integer counts, integrating the binomial count law over windows
that leave out at most about `1e-12` of mass. At 701 units per arm, a 10%
baseline and a 50% lift this is 0.7144 (the log-ratio model said 0.8004), and
the planned size for 80% power is 831 per arm. Power depends on the baseline
rate alone; `var` does not enter.

- `power_basis="exact"`: the decision set is built by replaying the runtime's
  own nuisance search for every count pair that matters; the reported power is
  the runtime's rejection probability. This route is used when the geometry at
  the null rate has at most 16,000 (control, treatment) count cells -- a
  complete call then takes about two seconds or less.
- `power_basis="approximate"`: larger designs replay the same search with a
  continuity-corrected Normal tail for the conditional binomial sum. Measured
  against the runtime it agreed on dense designs to about 0.03 percentage
  points but understated power by up to 0.8 points with an unequal allocation
  and a shifted null, and it can misclassify count pairs whose runtime p-value
  sits near the tail allocation in rare-event designs.
- Replay bound: the replay's work and memory follow the number of (control,
  treatment) count cells the geometry stores, not the arm size: the control
  window by the treatment windows at the null rate and at every alternative
  evaluated (their union, for an effect search or a power curve). A design
  whose null rectangle exceeds 10,000,000 cells, or an alternative whose window
  would take the geometry past it, is refused with
  `power.binomial_replay_bound_exceeded` before any replay or allocation (the
  error's context names the analyzed counts, the control rate, the
  alternative treatment rate `p_t` when it is the alternative that exceeds,
  `cells` and `max_cells`). An effect search that would exceed it ends
  unresolved: a companion `mde_relative` is `None` with `numerical_resolution`,
  and `minimum_detectable_effect` is refused with the bound, after the work
  that preceded it (about one to two minutes of CPU in the designs measured). An effect
  search stores more than the null rectangle, so it reaches the bound at
  smaller arms than a supplied effect: at a 5% baseline `achieved_power` is
  answered at 1,000,000 units per arm (but its companion effect is not) and
  `minimum_detectable_effect` at 500,000 but not at 1,000,000.
  `required_sample_size` searches only sizes at most 1/128 under the largest
  whose null and supplied-effect rectangles fit (or under the size where the
  runtime starts refusing the tail level) and refuses with the power it reached
  at that ceiling (`power_reached`), which skipped sizes just above it may
  exceed. A 5% baseline with equal arms reaches the bound at about one million
  units per arm and a 50% baseline at about 190,000; a rare baseline stays far
  inside it at any arm size (100 million units per arm at a rate of 2e-7 expect
  twenty events and replay about four thousand cells). Measured on an Apple M3
  Pro, a call within the bound takes up to a minute of CPU for `achieved_power`
  or `minimum_detectable_effect` and about 4.5 minutes for `required_sample_size`
  near a million units per arm at 5%, with a peak under 2 GiB (the
  [limitations page](../limitations.md) lists the cells measured). The bound
  limits the planner only: the runtime decides arms of up to a billion units.

The route depends only on the design, never on timing. Power is not monotone
in the sample size under this decision, so `required_sample_size` returns a
verified size whose power reaches the target while the size one below does
not -- not a proof that no smaller size reaches it. `minimum_detectable_effect`
returns the first admissible effect whose power reaches the target; earlier
effects are excluded by their own power or by a bound on the rejection set,
to within `2e-12` of the target. A design the runtime refuses in full has power
0 and is never replayed: an arm beyond its one-billion unit ceiling, a nuisance
budget (`alpha / 32`) below the endpoint solver's floor (an alpha under `3.2e-8`),
or a tail level its float margin dominates (a two-sided alpha below about `9.5e-7`
at a billion units per arm; see the limitations page). `required_sample_size`
refuses a decision refused even at the smallest arms with
`power.binomial_tail_level_unrepresentable` and asks for a larger alpha.

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

Combining a CUPED method with non-fixed inference is not supported. Such a
procedure is refused with `arm.adjustment.sequential_cuped`. Clustered
sequential planning is also refused; use fixed-horizon clustered planning (valid for one planned analysis, not repeated looks) or
unit-randomized sequential planning.

## Reading `PowerResult`

| Field | Meaning |
|---|---|
| `n_per_arm` | Assigned units in the treatment arm |
| `n_total` | Assigned units across the two-arm contrast |
| `power` | Planned power at the returned integer size, under `power_basis` |
| `power_basis` | `asymptotic` (log-ratio model), `exact` (the runtime's binomial decision), or `approximate` (that decision with Normal conditional tails) |
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
