# Fixed-horizon switchback contrasts

A switchback alternates a unit such as a store, a city, or a whole market
between treatment and control over time instead of splitting users into two
groups. Use it when units cannot be split and time trends affect the outcome;
`Analysis.from_switchback_panel` reports an additive difference over the
retained windows, not a per-user relative lift.

## Supported contract

The input is an eager DataFrame with one row per **unit × cycle × period ×
step** cell. Pass the column names for `unit`, `cycle`, `period`, `step`, and
`group`, plus a mapping or sequence of mean or conversion metrics, and declare:

| Argument | Requirement |
|---|---|
| `identification` | A `Randomized` design with exactly one treatment arm and one control arm, allocation exactly `0.5`/`0.5`. |
| `assignment` | A `SwitchbackAssignment` with `periods_per_cycle=2` and one of the two order laws below. |
| `SwitchbackWindow` | `washout_steps`, `observation_steps`, and `carryover_order` (default `0`): extra observation steps discarded after washout. Orders `0`, `1`, and `2` are supported under both laws whenever enough steps remain. |

<!-- skip: next "requires a prepared switchback panel" -->
```python
# The frame has columns: store_id, cycle, period, step, arm, orders.
# `panel` is an eager pandas, polars, or pyarrow frame prepared by the caller.
from increment import Analysis, Randomized
from increment.estimators import (
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)

assignment = SwitchbackAssignment(
    sequence=IndependentBernoulliOrder(probability_ct=0.5),
    window=SwitchbackWindow(washout_steps=2, observation_steps=6),
)
# Ordinary switchback inference selects the qualified unit-t approximation
# from this design; no population envelope or reference declaration is needed.
analysis = Analysis.from_switchback_panel(
    panel,
    unit="store_id",
    cycle="cycle",
    period="period",
    step="step",
    group="arm",
    metrics={"orders": "mean"},
    identification=Randomized(
        control_group="control",
        allocation={
            "control": 0.5,
            "treatment": 0.5,
        },
    ),
    assignment=assignment,
)
results = analysis.run()
diagnostics = analysis.assignment_diagnostic()
```

### Choose the order law

| Law | Who draws the CT/TC order | Independent replicate | Reference |
|---|---|---|---|
| `IndependentBernoulliOrder(independence_unit="unit_cycle")` | Each unit-cycle, independently | The unit | `UnitCycleTApproximation`, `dof=n_units-1`; at least two units |
| `SharedScheduleOrder(independence_unit="shared_block")` | One draw per two-period block, shared by a fixed, complete roster | The block | Block-t, `dof=n_blocks-1`; at least two blocks, roster of one or more units |

For a common schedule across the roster, pass
`sequence=SharedScheduleOrder(probability_ct=0.5)` in the same constructor.
The order probability describes independent Bernoulli draws; it does not
impose fixed CT/TC counts or a finite-population correction. Under a shared
schedule, uniformly duplicating the roster changes neither the target scale
nor the number of independent blocks.

### Washout and carryover

The first `washout_steps + carryover_order` rows of each period are excluded,
and the remaining `observation_steps - carryover_order` steps are summed for
mean metrics or combined with `any` semantics for conversion metrics. With
`SwitchbackWindow(washout_steps=2, observation_steps=6, carryover_order=2)`,
four steps per period are retained; orders zero and one retain six and five.

!!! warning "Carryover is assumed away, not tested"
    The identifying assumption is
    `no_residual_carryover_after_discarded_steps`: the declared washout plus
    any `carryover_order` steps must remove treatment carryover from the
    retained steps. Increment validates the logged schedule and applies the
    window; it cannot prove the scheduler was random or that carryover has
    vanished. If residual carryover remains, the contrast can be biased and
    the interval need not cover.

??? info "Statistical details: validation and the contrast calculation"
    The panel validator requires non-empty labels, exact integer (not
    boolean) `cycle`, `period`, and `step` values, contiguous cycles `0..C-1`
    for every unit, periods exactly `{0, 1}`, and every period's complete
    step range `0..washout_steps + observation_steps - 1`. Every unit-cycle
    must contain one control and one treatment period. Missing cells,
    duplicate cells, invalid labels, non-finite outcomes, and an incomplete
    schedule are coded input refusals. A declared unit-cycle variance
    envelope supports one unit; its descriptive sample SE is then null with
    reason `insufficient_replicates`.

    For independent unit-cycle orders, each cycle contribution is
    inverse-probability weighted using the declared CT/TC order probability,
    then averaged over a unit's cycles; the final contrast averages those
    unit contributions. For shared schedules, each period's retained outcomes
    are first averaged across the fixed roster, then inverse-probability
    weighted into a block contribution; the final contrast averages those
    block contributions. Independent unit-cycle inference automatically
    records `UnitCycleTApproximation`; an explicit, justified prospective
    envelope remains optional. Neither reference is a sequential boundary.
    Integer outcomes retain their exact signed differences before division
    or floating-point conversion, so large common offsets cannot erase a
    representable treatment contrast.

    For shared block `b`, write the roster means in its two retained periods
    as `M1_b` and `M2_b`. Its contribution is `(M2_b - M1_b)/(2p)` for CT and
    `(M1_b - M2_b)/(2(1-p))` for TC, where `p=probability_ct`. The estimate
    is the mean of the `B` block contributions, with standard error
    `s_X/sqrt(B)` and `B-1` degrees of freedom. This is an approximate
    HT/Neyman average-effect interval; it is not Fisher sharp-null inversion
    or an exact finite-sample coverage guarantee. Heterogeneous fixed-block
    effects can make its variance conservative. Generalizing to future
    blocks additionally requires a superpopulation interpretation.

    Under a positive prospective envelope, zero observed between-unit spread
    does not remove the interval: its uncertainty comes from the declared
    residual variance bound. A zero envelope identifies a singleton under its
    declared model; it must never be inferred from a zero sample standard
    error. Legitimate all-CT or all-TC observations remain admissible when
    the split check accepts them.

To size a switchback, see
[switchback power planning](power-analysis.md#switchback-contrasts). Ordinary
planning uses a moment-t approximation with pilot-estimated moments; two-sided
MDE planning retains the signed additive effect, searching below the null
when `preferred_direction="decrease"` (within any declared `effect_bounds`)
and above it otherwise.

## Randomizing at a market or geo grain

For independent unit-cycle orders, `unit` names whatever entity drew its own
treatment order. For a market or geo switchback, pass the geo column as
`unit` and pre-aggregate the metric to `geo x cycle x period x step` first;
the validator's shape requirements then apply per geo. `n_units` is the number
of geos, and the default unit-t approximation uses `dof=n_units-1` (an
optional prospective envelope uses that `N` in its variance bound and reports
`dof=None`). Each geo must draw its order independently, as
`independent_bernoulli_order` asserts.

When the whole market flips together, use
`SharedScheduleOrder(probability_ct=...)`: each two-period cycle is an
independent block with one common order across the complete geo roster. This
supports even one geo given at least two blocks, with `dof = n_blocks - 1`
regardless of roster size.

!!! warning "Independent orders do not make geos independent"
    Independent per-geo orders make the contrast design-unbiased, but shared
    time shocks, correlated schedules, and interference can still violate
    independence. See the limitations page's entry on switchback unit
    independence.

## Results and diagnostics

`Analysis.run()` returns `ContrastResults`, a typed list of `ContrastResult`
objects, one per selected metric, each with `method_role="decision"` and
`assignment="switchback"` (see [Reading results](reading-results.md) for
contrast rows and [Multiplicity](multiplicity.md) for the status table):

| Field | Meaning |
|---|---|
| `estimate`, `lb`, `ub`, `standard_error` | The additive effect and its interval, with the declared `alpha`, `alternative`, and `null_abs`. |
| `method` | `switchback_unit_t_approximation` (default unit-cycle), `switchback_unit_variance_envelope` (prospectively bounded unit-cycle), or `switchback_block_t` (shared schedule). |
| `estimand` | `mean_unit_retained_window_difference` for a unit-cycle envelope; `retained_window_total_difference` (mean) or `retained_window_conversion_difference` (conversion) for shared schedules. |
| `n_units`, `n_cycles`, `n_blocks`, `dof` | Roster and replicate counts. For a shared schedule, `n_units` is the roster size and `n_cycles=n_units*n_blocks`; roster size is not the inferential sample size. |
| `ct_cycles`, `tc_cycles` | Realized order draws: unit-cycle draws under `independent_bernoulli_order`, block draws under `shared_schedule`. |
| `aggregation`, `probability_ct`, `randomization_law`, `independence_grain`, `carryover_order`, `observation_steps`, `retained_steps`, `identifying_assumption` | The declared design, echoed; `retained_steps=observation_steps-carryover_order`. |

`increment.workflows.assignment_diagnostic(experiment, resolver)` returns the
validated schedule evidence for an experiment built with
`from_switchback_frame` (`analysis.assignment_diagnostic()` returns the same
evidence on the compatibility route):
`probability_ct`, `randomization_law`, `independence_grain`,
`schedule_complete`, `n_units`, `n_cycles`, `n_blocks` for shared schedules,
`ct_count` and `tc_count` (unit-cycle draws under the unit-cycle law, block
draws under the shared law), washout/observation/retained step and row
counts, `carryover_order`, `identifying_assumption`, and any integrity
failures. It is not an SRM test and contains no p-value or anytime-valid
guarantee. Unit-arm checks do not apply to switchback order randomization:
`health_check` reports assignment balance as unavailable
(`integrity.switchback_assignment_law`) and `srm` refuses with
`facade.analysis.contrast_unavailable`.

Unit-cycle envelopes support additive `sum` aggregation only; they reject
`any`/conversion shifts, whose binary support does not permit an unrestricted
additive effect model. Shared conversion contrasts keep their
probability-difference meaning.

??? info "Statistical details: the portable schema"
    The schema distinguishes unit-cycle envelope results (`n_blocks=null`,
    `reference="residual_chebyshev"` or `"residual_cantelli"`, `dof=null`,
    `dof_unavailable_reason="not_applicable"`), sample-based unit-cycle
    approximations (`method="switchback_unit_t_approximation"`,
    `reference="unit_t_approximation"`, `dof=n_units-1`), and shared-block
    results (`method="switchback_block_t"`, `reference="block_t"`,
    `dof=n_blocks-1`). Law/grain mismatches and inconsistent counts or
    reference metadata fail coded. The panel entry point selects the unit or
    block reduction from the declared sequence, and the estimator uses the
    corresponding reference.

    `ContrastStats` and `ContrastResult` evidence carries `ct_cycles` and
    `tc_cycles` in JSON, result frames, and readout rows; they sum to
    `n_cycles` under `independent_bernoulli_order` and to `n_blocks` under
    `shared_schedule`, regardless of roster size. Invalid sums raise
    `estimation.contrast.order_counts`. Custom `ContrastPartition` producers
    must provide a copied, immutable `ct_counts_by_unit` mapping with the
    same keys as `unit_deltas` and `cycles_by_unit`, counting CT cycles for
    independent units or holding 0 (TC) or 1 (CT) for shared blocks.
    Reduction sums these counts once per disjoint key; counts cannot be
    inferred from `probability_ct`. Independent-order evidence may omit both
    counts, which stay numeric nulls rather than being reconstructed;
    supplying only one count is invalid, and shared-schedule evidence always
    requires both. `observation_steps` and `retained_steps` are required in
    both JSON forms; order zero retains the full observation window.

## A nondegenerate shared example

This hand-constructed schedule illustrates the calculation for four
independent block draws and three roster members. Each observed period has
one washout step and three observation steps; order one discards the first
observation step. The fixed roster has unit effects 4, 6, and 8 per retained
step. The realized block contributions are 10, 14, 8, and 16, giving an
effect of 12 and `SE=sqrt(40/(4*3))`, with three degrees of freedom.

```python
import math

import pandas as pd

from increment import Analysis, Randomized
from increment.estimators import (
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)

orders = [("control", "treatment")] * 2 + [("treatment", "control")] * 2
shifts = [-1, 1, -2, 2]
panel = pd.DataFrame(
    [
        {
            "unit": f"u{unit}",
            "cycle": block,
            "period": period,
            "step": step,
            "group": group,
            "value": 999 if step < 2 else 10 + (effect + shifts[block]) * (group == "treatment"),
        }
        for unit, effect in enumerate([4, 6, 8])
        for block, order in enumerate(orders)
        for period, group in enumerate(order)
        for step in range(4)
    ]
)
analysis = Analysis.from_switchback_panel(
    panel,
    unit="unit",
    cycle="cycle",
    period="period",
    step="step",
    group="group",
    metrics={"value": "mean"},
    identification=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
    ),
    assignment=SwitchbackAssignment(
        sequence=SharedScheduleOrder(probability_ct=0.5),
        window=SwitchbackWindow(washout_steps=1, observation_steps=3, carryover_order=1),
    ),
)
result = analysis.run()[0]
assert math.isclose(result.estimate.value, 12)
assert math.isclose(result.standard_error, math.sqrt(40 / 12))
assert result.n_blocks == 4 and result.n_units == 3 and result.dof == 3
assert result.observation_steps == 3 and result.retained_steps == 2
```

## When it refuses

Construction refuses before any statistics are built:

- A degenerate split (for example `ct=320`, `tc=0`, with a binomial check
  below `1e-6`) or an incompatible unit-cycle declaration. A plausible split
  cannot prove the declaration is truthful.
- A shared schedule whose blocks never realize one of the two orders at all,
  even when the split looks plausible: 6 blocks at `probability_ct=0.8` land
  all-CT about 26% of the time (`0.8**6`), and block-t's between-block
  variance would then measure only observation noise, never the treatment
  contrast. An implausible split under the shared law also refuses, using
  block draw counts.

!!! warning "Plan skewed shared schedules with more blocks"
    `switchback_achieved_power` and the solvers built on it already multiply
    by the probability that a shared-schedule design of that block count and
    `probability_ct` is admissible at all
    (`1 - probability_ct**n_blocks - (1 - probability_ct)**n_blocks`), so a
    skewed `probability_ct` with few blocks is sized with more blocks, not
    silently under-powered.

Relabeling an independent-order panel as shared does not create uncertainty:
zero block spread yields zero SE, unavailable decision evidence, and
numeric-null bounds and confidence level, never a fabricated interval.

## Deliberate refusals

Switchback construction failures expose these stable codes:

| Code | Refused condition |
|---|---|
| `source.frame.switchback.assignment` | The assignment is not a two-period `SwitchbackAssignment` with a declared `IndependentBernoulliOrder` or `SharedScheduleOrder` sequence. |
| `source.frame.switchback.identification` | Identification is not `Randomized`, does not contain exactly one control and one treatment, or has invalid allocation weights. A non-50/50 allocation raises this same code when a `SwitchbackStudyEnvelope` is constructed. |
| `source.frame.switchback.metric` | A metric type or option is outside mean/conversion switchback support (ratio, quantile, retention, winsorization, non-default missingness, CUPED, method/prior declarations, or a window option). |
| `source.frame.switchback.columns` | A required column is missing, or the group labels are empty/non-string. |
| `source.frame.switchback.domain` | Unit labels or cycle/period/step domains are invalid, including non-contiguous cycles or periods outside `{0, 1}`. |
| `source.frame.switchback.schedule` | Schedule cells are missing/duplicated, steps are incomplete, a unit-cycle does not contain one control and one treatment period, a shared schedule's realized order disagrees across the roster within a block, the realized CT/TC split is implausible under the declared `probability_ct`, or a shared schedule realizes only one cycle order across all declared blocks (a positivity failure, even when the split is plausible). |
| `source.frame.switchback.missingness` | A required schedule or metric value is null. |
| `source.frame.switchback.numeric` | A metric, aggregate, inverse-probability contribution, or reduction is non-finite or overflows. |
| `source.frame.switchback.units` | Fewer than two units are present for the default unit-cycle t approximation or pilot variance. A declared variance envelope and a shared schedule permit a one-unit roster. |
| `source.frame.switchback.plan` | A plan has non-default `q`, `view_multiplicity`, secondary-family inference, non-fixed inference, unsupported plan-bound `decision_method`/`sensitivity_methods`/`prior`, relative margins, or a nonzero relative null. |

- Shared-block inference with fewer than two independent blocks refuses with
  `InvalidRequestError`, code `estimation.contrast.switchback_block_inference`;
  a larger roster does not satisfy this minimum.
- The validator raises `InvalidRequestError` for assignment, identification,
  columns, domain, schedule, missingness, numeric, and units codes;
  unsupported metric options use `CapabilityError`.
- Multiple primary metrics are supported and receive the compiler's
  conservative alpha splitting. Duplicate metric specifications are rejected
  by metric coercion with an uncoded `ValueError`, and duplicate plan entries
  by `AnalysisPlan` validation, not `source.frame.switchback.plan`.
- `Analysis.run()` accepts only its default role/prior arguments (`UNSET`).
  The fixed `ContrastDecisionProcedure` owns the one decision method, fixed
  inference, family (`none`), alternative, null, and alpha. Passing
  `decision_method`, `sensitivity_methods`, or `prior` raises
  `CapabilityError` with code `readout.contrast.override` before source access.

For parallel A/B analyses, use `Analysis.from_unit_summary`,
`Analysis.from_unit_panel`, or `Analysis.from_definitions`; their arm-moment
results and role-aware method controls are separate from this contrast family.

## Optional finite-sample references and research laws

If you can justify a prospective upper bound on the residual variance, supply
`contrast_references={metric_name: envelope}` to `from_switchback_panel` or
`Analysis.from_switchback_panel`. A `UnitCycleVarianceEnvelope` binds the
assignment and window, metric and arm labels, response meaning, cycles per
unit, the residual variance bound, and the provenance of the assumption: who
declared it and why. Software does not prove it true. Ordinary construction
without a reference selects the qualified unit-t approximation; no certificate
is required for inference or pilot planning.

!!! warning "A pilot variance is not a prospective bound"
    A pilot variance or observed standard error is not a prospective upper
    bound, and the default unit-t reference is not a finite-sample envelope
    guarantee; its small-sample calibration failures below remain
    unresolved.

Complete-law declarations live in `increment.semantics.unit_cycle`, and their
oracle solvers in `increment.power.unit_cycle`; they are not root or ordinary
power-planning exports.

??? info "Statistical details: the envelope interval"
    For CT probability `p`, the unchanged HT point is `mean(A_i)`, where
    `A_i=mean_c(w_order D_order)`, `w_CT=1/(2p)`, `w_TC=1/(2(1-p))`. Let
    `G_i=mean_c(w_order)`. The declaration asserts `E[A_i-delta G_i]=0` and
    `Var(mean(A_i-delta G_i))<=V0/N` for independent units. Heterogeneous
    unit, cycle, and period effects remain in this residual model.

    After reserving the split-refusal bound `r_N`, the two-sided residual
    cutoff is `c=sqrt(V0/(N*alpha_effective))`, with
    `alpha_effective<=alpha-r_N`. The interval is
    `[(mean(A)-c)/mean(G), (mean(A)+c)/mean(G)]`. Its center is an inversion
    device; the reported point remains HT and need not lie inside the
    interval. Coverage is unconditional over sampling and assignment,
    including refused intervals as misses. It is not conditional on the
    observed orders or slope. Directional intervals use the Cantelli cutoff
    `sqrt((V0/N)*(1-alpha_effective)/alpha_effective)` and an open opposite
    side.

    Results name the method `switchback_unit_variance_envelope` and reference
    `residual_chebyshev` or `residual_cantelli`. They retain the observed
    mean slope, CT/TC counts, requested/effective alpha, refusal bound,
    cutoff and provenance. Degrees of freedom are numeric null with reason
    `not_applicable`; sample SE remains descriptive. Evidence tests
    `mean(A)-null_abs*mean(G)` with the matching Markov/Cantelli probability
    and admission allocation, rounded upward to binary64. Rejection means
    this public p-value is strictly below `alpha`; the separately
    outward-displayed interval and cutoff do not define that event. Frames
    and wire serialization preserve these fields and numeric nulls. Readout
    `stat_sig` uses that same p-value decision, not interval exclusion.

    Extreme `alpha`, including the smallest positive binary64 value, is
    supported only when the admission reservation leaves positive effective
    alpha and the cutoff and required endpoints remain representable. A
    positive refusal bound can exhaust this budget
    (`unit_cycle.error_budget_exhausted`). The displayed confidence level may
    round to `1.0`; the stored alpha and upward-rounded p-value remain
    authoritative. The default unit-t reference is labeled
    `switchback_unit_t_approximation` / `unit_t_approximation`, with
    `dof=n_units-1`; source construction records it automatically, so callers
    need not duplicate it in a source mapping and procedure.

??? info "Statistical details: UnitCycleJointLaw and Monte Carlo planning"
    `UnitCycleJointLaw` declares a portable full population model for actual
    power: positive relative type weights, ordered CT/TC cycle means and noise
    loads, innovation multiplicities, a normal/centered-lognormal/centered-gamma
    innovation, and a per-unit reuse probability. A reuse unit shares one
    innovation across its cycles; a non-reuse unit averages independent
    micro-innovations at each selected cycle's declared multiplicity. Types,
    orders and reuse draws are independent across units. Type weights
    describe population probabilities, not analysis weights. For a nonzero
    reference effect, the required bound is `Var(A_ref-reference_effect*G)`,
    not `Var(A_ref)`.

    - `moment_t_approximation`: an effect-dependent moment-t calculation,
      retaining its nonmonotone cases. It does not establish finite-law power.
    - `certified_power_lower_bound`: `unit_cycle_power_lower_bound` gives a
      guarantee under an envelope for the affine-unit sufficient-state
      adapter described below. It subtracts bounds on reporting and
      descriptive unavailability. A zero bound with
      `runtime_availability_not_certified` does not prove unattainability.
    - `model_mc_power`: `unit_cycle_model_power`, `unit_cycle_model_mde` and
      `unit_cycle_model_required_units` bound successful deployed rejection
      for that adapter under `UnitCycleJointLaw`. Supply repetitions, seed and
      MC error prospectively. MDE output is an earliest plausible/feasible
      bracket with uncertainty, not an exact scalar root. Availability can
      make even one-sided power decrease with effect or N.

    `unit_cycle_law_moments` derives population moments from that full law;
    `unit_cycle_variance_envelope` returns its matching prospective envelope.
    The law is itself a scientific assertion. Neither function validates it
    from data.

    Actual-model planning uses the exact residual
    `T=U+(effect_delta-null_abs)*G`, including exact subtraction of the two
    supplied effects. Write `v=V0/N` and `cap=nextafter(alpha,0)-r_N`,
    evaluated as rationals. For positive variance and positive `cap`,
    two-sided rejection is `T²>=v/cap`; directional rejection additionally
    requires a favorable residual and uses `T²>=v*(1-cap)/cap`. Equality at
    these deployed boundaries rejects. If positive variance has `cap<=0`, no
    finite residual rejects. Zero variance instead excludes the residual-zero
    atom and, for directional tests, the unfavorable side. All probabilities
    retain refused and failed draws in the unconditional denominator.
    Effect-size planning searches a prespecified finite grid and returns a
    selection-safe power certificate for the first qualifying grid candidate.

    Planning results identify `evaluation_scope="affine_unit_sufficient_state"`
    and `availability_method="exact_reporting_certified_descriptive_v1"`. The
    adapter rounds exact modeled unit values `U_i + effect_delta*G_i` to
    binary64, constructs scalar centered descriptive moments in unit
    generation order, and supplies their separate exact total and order
    counts to envelope inference. The law does not specify raw period levels
    or their rounding, so this scope does not promise availability for
    arbitrary raw-panel encodings.

    Point and required confidence-endpoint availability are exact. A
    sufficient rounded-value-window certificate establishes finite
    descriptive construction; outside it, availability remains unresolved.
    `rejected` counts certified available rejections, while
    `possible_rejected` includes unresolved outcomes that could reject.
    Proven reporting refusals enter neither count. Sampling failures remain
    possible rejections when admitted. All counts use attempted repetitions
    as their denominator. `known_runtime_refused`, `availability_uncertified`,
    and `sampling_failed` distinguish admitted failures; an incomplete
    admitted evaluation makes `power` null with an explicit reason.

    Seeded records use `unit-prefix-v2` and `finite_type_reuse_exact_micro_v2`:
    orders are drawn before innovations, and innovations with no effective
    load are not sampled. Records with the v1 sampler require that
    implementation and cannot be replayed with the v2 sampler or relabeled
    during deserialization. Missing `rng` or `sampler` fields retain their v1
    defaults. New results explicitly record v2, including when serialized
    with default-valued fields omitted.

    For a prospectively fixed effect, `unit_cycle_model_power` uses one-sided
    Hoeffding bounds on the certified and possible rejection proportions.
    Each tail receives half the MC error; the outward-rounded margin is
    `sqrt(log(1 / tail_error) / (2 * repetitions))`. The method is
    `fixed_query_bernoulli_hoeffding`, with `cdf_bands=2` and
    `cdf_terms_per_bound=1`. Unknown outcomes widen the bounds, not their
    denominator.

    For unrestricted adaptive effect selection from the same simulated data,
    `unit_cycle_model_power` requires `simultaneous_effects=True`. Effect
    arguments are normalized to binary64 before calculation and recorded at
    that precision. This optional full-domain mode chooses the tighter of two
    simultaneous bounds:

    - `simultaneous_binary64_hoeffding` allocates the MC error across both
      outcome bounds and at most \(2^{64}\) effect encodings. Its margin is
      `sqrt(log(1 / tail_error) / (2 * repetitions))`, with `cdf_bands=2` and
      `cdf_terms_per_bound=1`. The allocation covers adaptive selection, not
      just the effect eventually reported.
    - `simultaneous_dkw_massart` covers endpoint CDFs: four processes and two
      terms per bound for one-sided curves; eight processes and four terms
      for two-sided curves. It remains available when tighter or when
      finite-domain allocation underflows.

    The choice depends on the declared budget and alternative, never
    simulated outcomes.

    `unit_cycle_model_mde` instead requires `effect_grid`: a nonempty
    sequence of finite effects, strictly ordered in the favorable direction
    from the null. The null itself is allowed. The result copies the grid
    into an immutable tuple, which records the exact search range and
    nonuniform resolution. Before sampling, the planner divides `mc_error`
    conservatively across every declared candidate. It evaluates all
    candidates using common draws and the same deployed
    rejection/availability rules as fixed-effect power. A union bound
    protects selection even though those draws are shared. The method is
    `simultaneous_effect_grid_bernoulli_hoeffding`, with
    `search_domain="declared_effect_grid"`.

    `plausible_effect` is the first grid point whose power upper bound
    reaches the target; `feasible_effect` is the first whose lower bound
    reaches it. A certified result has `search_status="certified_grid_effect"`.
    Otherwise `grid_exhausted` means only that this grid and Monte Carlo
    precision supplied no certificate. Neither result establishes the global
    minimum, excludes effects between grid points, or proves the target
    unattainable. No monotonicity is assumed. Top-level failure counts
    describe `diagnostic_effect` (the null); `power_at_feasible` describes
    the selected effect. Replay that certificate with default fixed-query
    power and `mc_error=result.mc_error_per_effect`, using its recorded seed,
    repetitions, law, procedure and NumPy version.

    Required-N planning holds the effect fixed and divides the MC error
    across every size from 1 through `max_n` before sampling. Each size uses
    the cheaper fixed-query bound; the union bound protects size selection
    even though prefixes are shared. This method is
    `simultaneous_n_bernoulli_hoeffding`. Availability is recomputed at every
    prefix without assuming monotonicity. Replay the selected certificate
    with default fixed-query power and `mc_error=mc_error_per_n`.

## Response meaning and independent certification

Switchback contrasts use inverse-probability CT/TC weighting, then average
equally over cycles within each unit and over independent units. Supplied
responses may be totals or pre-normalized means from equal or unequal
underlying micro-observation counts; the runtime receives neither
micro-counts nor a second weighting selector, and unequal retained
schedule-row counts remain outside the complete-schedule contract.

!!! warning "Totals and means define different responses"
    Each response meaning needs its own potential outcomes and effect units.
    In the calibration fixtures, the treatment increment is defined on the
    supplied aggregate scale. Changing from totals to means therefore changes
    the scientific response model; it is not a lossless conversion.

??? info "Statistical details: the simulation scenario and manifest"
    `SwitchbackScenario` adds explicit `noise_distribution`, `noise_sd` and
    `within_unit_correlation` inputs. The named standardized innovations are
    normal, `(exp(Z)-exp(.5))/sqrt(e*(e-1))`, and `(Gamma(2,1)-2)/sqrt(2)`.
    A per-unit Bernoulli selector chooses one common treatment innovation or
    independent micro-errors. Control errors are zero, so paired contrasts
    retain noise. `period_effect_heterogeneity` and
    `treatment_period_effect` add centered heterogeneity. Noise uses
    `SeedSequence([scenario.seed,1414])`; `noise_sd=0` makes no extra draws
    and preserves the default generator's random stream.

    `tests/unit_cycle_prospective.json` preserves all 4,320 cells: N
    `4/10/20/40/200`, C `1/5/20`, p `.25/.5/.75/.9`, three innovations, reuse
    `0/.5/.9`, equal/unequal micro-counts, effects `0/1`, and two response
    meanings. The base cells have true=declared carryover zero. A separate
    full true-by-declared `0/1/2` matrix under both assignment laws remains
    required for certification.

    Independent rational checks cover expectations, covariance, residual
    variance, positive slopes, coverage/width, actual split-refusal mass and
    admitted HT-point bias. Public source/estimate/evidence/frame/wire parity
    is a separate check. The corrected N4/C1/p.9 counterexample retains
    miscoverage lower bounds `.05653644209396722` and `.05635913367335912`; a
    unit-t pivot cannot certify it.

### Representative release checks

!!! warning "The unit-cycle campaign has not been executed"
    The thresholds below are acceptance bands the campaign is *written to
    enforce*, not results it has returned. No coverage, availability or power
    figure here has been measured on the manifest it describes; read them as
    the specification of a pending check.

The ordinary suite uses deterministic representatives covering every manifest
axis value, plus explicit small-N/skew/dependence intersections. It does not
claim that a representative run executed the full matrix.

??? info "Statistical details: representative cells and acceptance bands"
    Public-runtime calibration uses 1,536 attempts per checkpoint for these
    cases:

    | N | C | CT probability | Innovation | Reuse | Counts | Response |
    |---|---|---|---|---|---|---|
    | 4 | 1 | .9 | Normal | 0 | Unequal | Retained total |
    | 10 | 5 | .75 | Centered lognormal | .9 | Unequal | Pre-normalized mean |
    | 40 | 1 | .25 | Centered Gamma | .5 | Equal | Retained total |

    Null, effect-one and certified-grid-effect lanes retain every attempted
    draw, including failures. Coverage lower bounds must reach `.945`;
    interval availability lower bounds must reach `.99`; null rejection upper
    bounds must be at most `.055`. The selected grid effect also needs a
    public-runtime power lower bound of `.80`. The separate
    N40/C1/p.5/V0=1/effect1 Normal witness compares model and public-runtime
    power against its analytic reference, whose power exceeds `.90`.

    Returned-point bias and its Monte Carlo standard error remain explicit
    diagnostics alongside the independent analytic bias check.
    Runtime-minus-model power intervals are uncertainty diagnostics, not
    `.005` equivalence certificates. Every observed public decision is
    checked against the independent scalar oracle. There is no fixed tiny
    Monte Carlo margin requirement or globally smallest-MDE claim.

    The `.001` family reservation is split conservatively: half for the four
    representative cases, half for the full research campaign. Case and
    checkpoint allocations are declared before outcomes; checkpoint j spends
    `case_error/[j*(j+1)]`. Planning and public-runtime streams are
    independent. Clopper–Pearson decisions use SciPy beta inversions with
    outward binary64 rounding; these are not validated-arithmetic enclosures
    of the special functions.

### Bounded full research campaign

The complete 4,320-cell manifest is unchanged; its two effect rows share each
of 2,160 model/MDE designs. Run the full campaign explicitly:

```bash
uv run python -m calibration.unit_cycle \
  --output /tmp/unit-cycle-campaign \
  --budget-seconds 3600 --case-budget-seconds 60
```

For a short software smoke, add `--smoke --budget-seconds 30`; its small draw
count normally exits with status 2 (`inconclusive`), not a calibration
certificate. `full_matrix_certified` is true only if the entire requested
matrix finished with certified cases; `inconclusive`, `failed` and `timeout`
are not scientific passes.

??? info "Statistical details: campaign execution and records"
    Each design executes both analytic/public-parity cases, then evaluates
    declared effect grids and public-runtime calibration at geometric
    checkpoints. Only unresolved decisions receive further simulation. Every
    record retains seeds, allocations, counts, failures and uncertainty.
    Output also records source hashes and dependency versions.

    The overall execution budget cannot exceed one hour; per-design guards
    prevent one case from consuming it all. Completed checkpoints and partial
    progress remain in the output directory after interruption or timeout.
    `--start-design` and `--stop-design` select an explicit restart range
    without changing case seeds or full-family error allocation. Use a new
    output directory: existing evidence is never overwritten.

    The frozen-sizing program and manifest criteria remain research records,
    not mandatory release dependencies. Representative checks retain their
    evidence through pytest's `evidence_path` property; the standalone
    campaign writes its own portable JSON/JSONL bundle. Neither replaces the
    separate carryover matrix or unrelated release gates.
