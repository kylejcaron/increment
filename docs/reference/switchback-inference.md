# Switchback inference reference

Use the [switchback guide](../guides/switchback.md) to choose a schedule and
run a complete example. This reference defines the input, calculations,
result fields and limits of each statistical reference.

## Input and schedule validation

`from_switchback_frame` takes an eager dataframe, column names (`unit`,
`cycle`, `period`, `step`, `group`), metric declarations, a two-arm
`Randomized` design and a `SwitchbackAssignment`. Match the allocation to
the scheduler, not to the realized data:

| Sequence | Periods in each cycle | Control / treatment allocation |
|---|---|---|
| `IndependentBernoulliOrder` or `SharedScheduleOrder` | Two; one period per arm. CT probability is an **order** probability. | `0.5`, `0.5` |
| `FixedCountOrder(treated_periods=m, independence_unit=...)` | `K >= 3`; draw `m` of `K` equal-duration positions uniformly without replacement, `0 < m < K`. | `(K-m)/K`, `m/K` |

Cycles, periods and steps are contiguous, zero-based integer axes, not
booleans. In a dense frame, supply one row with its arm and response at
every unit × cycle × period × step, including washout. Duplicate keys,
gaps and nulls do not establish a schedule. Every shared block must have
the same schedule across its roster.

**Separate assignment log.** Pass `schedule=SwitchbackObservationSchedule(
assignments=..., units=..., cycles=...)` when the log, not the response
table, records who was assigned. The log has one `group` per unit, cycle
and period, no step axis. Declare the complete roster and expected cycle
count independently of observed outcomes: a missing last cycle or entire
unit is an error. An observation may omit `group`; if included, it must
match. A repeated or off-schedule observation key invalidates the study
even if it is in washout.

Under a valid log, a missing discarded step is harmless. An unmeasured
retained step makes its **metric** unavailable, not the whole study.
For sparse event feeds, an absent event value or numerator counts as zero
only when `MetricSpec(missing="zero", ...)` and `complete_event_metrics`
name that same metric. A recorded null is unknown; denominators and forecasts
are never supplied by this rule. There is no complete-case comparison.

**Separate forecast table.** For scalar mean CUPED, `pre_assignment_forecasts`
needs exactly one row per unit × cycle × period, using the builder's key
column names. `forecast_at` and `assigned_at` must both be dates, or both
datetimes with the same time zone. The latest forecast must strictly precede
the first assignment. A missing key, repeated key or late forecast invalidates
the table. A missing **value** at a valid key makes only metrics using that
forecast unavailable (`missing_forecast`). Other metrics keep the full roster
and their original share of alpha. Use finite numeric values; declare the
predictor with `MetricSpec(..., decision_method=Method.cuped(covariates="forecast"))`.

The retained window is defined in the guide's
[washout and carryover section](../guides/switchback.md#washout-and-carryover).
The declared carryover assumption is not a test that residual effects have
vanished.

## Contrast calculation

### Unadjusted means and conversions

Reduce each period to its retained total (mean metric) or an indicator for
any event (conversion). For a unit-cycle draw, compute a contrast per
cycle and average a unit's cycles; for a shared draw, average over the
fixed roster within the block. Average the resulting independent
replicates. Under either schedule use their sample SD divided by the
square root of their number, with approximate t inference on `n-1`
degrees of freedom. Neither t reference is a sequential boundary.
Integer outcomes retain their exact signed differences before division
or floating-point conversion, so large common offsets cannot erase a
representable treatment contrast.

For a **paired** shared block `b`, write the roster means in its two periods
as `M1_b` and `M2_b`. Its contribution is `(M2_b - M1_b)/(2p)` for CT and
`(M1_b - M2_b)/(2(1-p))` for TC, where `p=probability_ct`. The estimate
is the mean of the `B` block contributions, with standard error
`s_X/sqrt(B)` and `B-1` degrees of freedom. This is an approximate
HT/Neyman average-effect interval; it is not Fisher sharp-null inversion
or an exact finite-sample coverage guarantee. Heterogeneous fixed-block
effects can make its variance conservative. Generalizing to future
blocks additionally requires a superpopulation interpretation.

Under a positive prospective **paired independent-unit** envelope, zero observed between-unit spread
does not remove the interval: its uncertainty comes from the declared
residual variance bound. A zero envelope identifies a singleton under its
declared model; it must never be inferred from a zero sample standard
error. Legitimate all-CT or all-TC observations remain admissible when
the split check accepts them.

For a **fixed-count** draw let `Y[j]` be the retained response at position
`j` and `Z[j]` its treatment indicator. Every `m`-subset is equally likely,
so each treatment probability is `p=m/K`. The draw contributes

$$
\frac{1}{K}\sum_j \left[\frac{Z_jY_j}{p}-
\frac{(1-Z_j)Y_j}{1-p}\right]
= \operatorname{mean}_{T}(Y)-\operatorname{mean}_{C}(Y).
$$

The fixed count alone does not justify those weights. For instance, with
untreated outcomes `(1, 0, 0)`, one treated slot and no treatment effect,
the contrast is 1 when the first slot is treated and −1/2 otherwise. It
averages to zero under uniform drawing but to 1/4 if the first slot is
treated half the time. Seeing both arms at every position is required to
analyze the schedule; it does not establish uniform randomization.

**Why a block minimum isn't a coverage guarantee.** Consider one store and
48 independent five-period blocks, each with one control slot drawn uniformly.
Suppose orders are always 100, 100, 100, 100 and 120 by position, regardless
of treatment. The block contrast is −20 if the busy slot gets control, +5
otherwise. At two-sided alpha .05, **7.14% of all assignments** falsely
report a difference. An interval is available in 99.989% of assignments;
of those, 92.86% include the true zero. This is an exact calculation for
one calendar, not a simulated rate or a prediction for another store.

To reproduce: for each number `h` of blocks with the busy slot in control,
the average is `5 − 25h/48` and its standard error is
`25 × sqrt((h/48)(1−h/48)/47)`. Give each admitted schedule weight `1/5^48`.
There are `choose(48,h) × Σ[j=0..4] (−1)^j choose(4,j)(4−j)^(48−h)` such
schedules for `1 ≤ h ≤ 44`: every ordinary slot must also have appeared in
control. Count a false call when the absolute average exceeds the standard
error times the 97.5th percentile of a t distribution with 47 degrees of freedom.

### Ratios and forecast-adjusted means

A ratio metric compares **two ratios of totals**, not the mean of period
rates. Within each independent replicate, compute arm-specific weighted
numerator and denominator totals using each position's arm probabilities.
For arm \(a\), write those contributions as \(Y_{ai},D_{ai}\) and the
rate as \(r_a=\bar Y_a/\bar D_a\). The point is \(r_T-r_C\).

- A zero denominator in either arm gives `zero_denominator` and no point.
  If a nonzero denominator's two-sided t interval at the metric's allocated
  alpha includes zero, `weak_denominator` retains the point only. The
  reason identifies the affected arm.
- Otherwise use the sample SD of
  \((Y_{Ti}-r_T D_{Ti})/\bar D_T-(Y_{Ci}-r_C D_{Ci})/\bar D_C\)
  divided by \(\sqrt n\), with `n-1` degrees of freedom. Keeping the two
  arms together preserves their covariance. This first-order approximation
  needs stable denominators; the separate denominator checks are not a
  joint coverage guarantee or a Fieller interval.

For **mean CUPED**, form a signed contrast of the frozen forecast with the
same order weights as the outcome. Regress the outcome contrasts on the
forecast contrasts with an intercept. The adjusted point is the intercept
at forecast contrast zero, its *randomization* expectation; subtracting
the observed forecast mean would change the answer. Refit without each
independent unit or block in turn. The jackknife standard error uses the
variation in those intercepts, with a t reference on `n-2` degrees of
freedom. A constant predictor gives no adjusted point (`rank_deficient`);
if deleting one replicate makes the slope unidentified, the full-fit point
is descriptive only (`high_leverage`). Neither fit falls back to an
unadjusted decision.

### Market and geo units

Set `unit` to the entity that drew its own schedule. For a geo experiment,
aggregate raw measurements to geo × cycle × period × step first.
When all geos follow one common draw, use `SharedScheduleOrder` for a
pair or `FixedCountOrder(..., independence_unit="shared_block")` for
multiple periods. Even one geo can be a roster; the independent sample
size is the number of blocks, not the number of geo-block rows.


## Result fields and diagnostics

`run(experiment, reader).readout().to_frame(...)` gives a row per selected
metric. Each has `method_role="decision"` and `assignment="switchback"`.
See [Reading results](result-fields.md) for contrast rows and
[Multiplicity](../guides/multiplicity.md) for status definitions.

- **`estimate`, `lb`, `ub`, `standard_error`:** the additive effect and its
  interval, with the declared `alpha`, `alternative`, and `null_abs`.
- **`method`:** unadjusted rows use `switchback_unit_t_approximation`,
  `switchback_unit_variance_envelope` or `switchback_block_t`. Ratios use
  `switchback_ratio_delta_t`, adjusted means `switchback_cuped_jackknife_t`.
  `forecast_column` identifies the predictor; changing it changes the
  method identity even if the numerical estimates coincide.
- **`estimand`:** for example, `retained_window_total_difference` (a
  difference in totals), `retained_window_conversion_difference` (a
  probability difference), or `retained_window_ratio_difference` (a
  difference in rates). None is relative lift.
- **`n_units`, `n_cycles`, `n_blocks`, `dof`:** roster and replicate counts.
  For a shared schedule, `n_units` is the roster size and
  `n_cycles=n_units*n_blocks`; roster size is not the inferential sample size.
- **`ct_cycles`, `tc_cycles`:** realized order draws: unit-cycle draws under
  `independent_bernoulli_order`, block draws under `shared_schedule`.
- **`periods_per_cycle`, `treated_periods`, `treatment_probability`:**
  fixed-count rows record `K`, `m` and `m/K` instead. `probability_ct`,
  `ct_cycles` and `tc_cycles` are null; CT and TC are not cycle types here.
- **`aggregation`, `probability_ct`, `randomization_law`, `independence_grain`,
  `carryover_order`, `observation_steps`, `retained_steps`,
  `identifying_assumption`:** the declared design, echoed;
  `retained_steps=observation_steps-carryover_order`.
- **`point_available`, `sampling_available`, `failure_code`, `failure_context`:**
  distinguish no answer from a point without a test. A missing retained
  response records `unobserved_response`, `missing_count`, `first_missing_key`
  in unit/cycle/period/step order, and the arm at that key. Other metrics
  keep their own results; `decision_scope_complete` is false when a required
  decision is missing. No zero estimate or non-rejection is substituted.

`increment.workflows.assignment_diagnostic(experiment, reader)` returns the
validated schedule evidence for an experiment built with
`from_switchback_frame`:
`probability_ct`, `randomization_law`, `independence_grain`,
`schedule_complete`, `n_units`, `n_cycles`, `n_blocks` for shared schedules,
`ct_count` and `tc_count` for paired draws or the declared fixed-count
exposure, washout/observation/retained step and row counts, `carryover_order`,
`identifying_assumption`, and any integrity failures. For fixed count it
checks the number of treated positions and realized support, not equality
of pattern frequencies. It is not an SRM test and contains no p-value or anytime-valid
guarantee. Unit-arm checks do not apply to switchback order randomization:
`health_check` reports assignment balance as unavailable
(`integrity.switchback_assignment_law`) and `srm` refuses with
`facade.analysis.contrast_unavailable`.

Unit-cycle envelopes support additive `sum` aggregation only; they reject
`any`/conversion shifts, whose binary support does not permit an unrestricted
additive effect model. Shared conversion contrasts keep their
probability-difference meaning.

### The portable schema

For unadjusted metrics the schema distinguishes unit-cycle envelope results (`n_blocks=null`,
`reference="residual_chebyshev"` or `"residual_cantelli"`, `dof=null`,
`dof_unavailable_reason="not_applicable"`), sample-based unit-cycle
approximations (`method="switchback_unit_t_approximation"`,
`reference="unit_t_approximation"`, `dof=n_units-1`), and shared-block
results (`method="switchback_block_t"`, `reference="block_t"`,
`dof=n_blocks-1`). Law/grain mismatches and inconsistent counts or
reference metadata fail coded. The panel entry point selects the unit or
block reduction from the declared sequence, and the estimator uses the
corresponding reference.

Under the paired laws, `ContrastStats` and `ContrastResult` carry `ct_cycles` and
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
supplying only one count is invalid, and paired shared-schedule evidence
requires both. Paired payloads do not need the new fixed-count fields.
`observation_steps` and `retained_steps` identify the window; order zero
retains the full observation window.

Metric failures keep the original metric, treatment arm, method and reason
through the readout, JSON and saved run, including a run in which every
decision is unavailable. Read the numeric null as “not estimated”, not a
zero-length interval.

## When it refuses

Construction refuses before any statistics are built:

- A degenerate paired split (for example `ct=320`, `tc=0`, with a binomial check
  below `1e-6`) or an incompatible unit-cycle declaration. A plausible split
  cannot prove the declaration is truthful.
- A shared paired schedule with just one realized order. Six blocks at
  `probability_ct=0.8` land all-CT about 26% of the time (`0.8**6`).
  This refuses even when the split is plausible; the split check counts
  block draws, not roster members.
- A fixed-count schedule with the wrong `m`, with conflicting positions
  across a shared block, or with a position never observed in one arm
  (`missing_position_support`). More rows from an existing block do not
  supply a missing draw.

The paired mean moment-t solvers multiply by the probability of seeing
both orders in a shared schedule, `1 - p**n_blocks - (1-p)**n_blocks`.
This factor alone is not a replay of every split or outcome-based failure.
The [profile grid planner](planning-methods.md#switchback-contrasts)
counts admission and rejection jointly for its supported models.

Relabeling an independent-order panel as shared does not create uncertainty:
zero block spread yields zero SE, unavailable decision evidence, and
numeric-null bounds and confidence level, never a fabricated interval.

## Deliberate refusals

Switchback construction failures expose these stable codes:

- **`source.frame.switchback.assignment`:** not a supported switchback
  assignment. The paired sequences require two periods.
- **`definition.assignment.fixed_count`:** with a positive treated count,
  the cycle must have at least three periods and at least one control.
  There is no position-probability override; unknown fields such as
  `position_probabilities` refuse with `model.field.unknown`.
- **`source.frame.switchback.identification`:** identification is not
  `Randomized`, does not name exactly one control and one treatment, or
  the declared allocation is invalid for the schedule.
- **`source.frame.switchback.metric`:** an unsupported metric or option,
  such as quantile, retention, winsorization, `missing="skip"`, CUPED on
  a ratio or conversion, a sensitivity method or a prior.
- **`source.frame.switchback.columns`:** a required column is missing, or the
  group labels are empty/non-string.
- **`source.frame.switchback.domain`:** unit labels or cycle/period/step domains
  are invalid, including non-contiguous cycles or out-of-range positions.
- **`source.frame.switchback.schedule`:** examples include duplicate cells,
  a gap in a dense schedule, the wrong treated count, a shared schedule
  that differs between units, insufficient position support or a paired
  schedule that fails its realized-order check.
- **`source.frame.switchback.missingness`:** a required value is null on
  the dense route.
- **`source.frame.switchback.assignment_unrecorded`:** a declared unit,
  cycle and period lacks a logged arm. Restore the assignment log, not
  an outcome row.
- **`source.frame.switchback.event_coverage`:** `missing="zero"` and the
  declaration of a complete event feed do not match.
- **`source.frame.switchback.response_unobserved`:** with a valid log, a
  required retained measurement is unavailable. The readout keeps the
  metric with its reason; requesting its pilot refuses.
- **`source.frame.switchback.numeric`:** a metric, aggregate, inverse-probability
  contribution, or reduction is non-finite or overflows.
- **`source.frame.switchback.units`:** fewer than two units are present for the
  default unit-cycle t approximation or pilot variance. A declared variance
  envelope and a shared schedule permit a one-unit roster.
- **`source.frame.switchback.plan`:** a plan has non-default `q`,
  `view_multiplicity`, secondary-family inference, non-fixed inference,
  unsupported plan-bound `decision_method`/`sensitivity_methods`/`prior`,
  relative margins, or a nonzero relative null.

- A shared schedule declared over one block refuses with
  `source.frame.switchback.schedule`: one block realizes only one cycle order,
  a positivity failure, and a larger roster does not lift it. Block-t inference
  over combined shared-schedule partitions that hold fewer than two independent
  blocks refuses with `InvalidRequestError`, code
  `estimation.contrast.switchback_block_inference`.
- The validator raises `InvalidRequestError` for assignment, identification,
  columns, domain, schedule, missingness, numeric, and units codes;
  unsupported metric options use `CapabilityError`.
- Multiple primary metrics are supported and receive the compiler's
  conservative alpha splitting. Duplicate metric specifications refuse with
  `InvalidRequestError`, code `frame.duplicate_metric_name`. Duplicate plan
  entries are rejected by `AnalysisPlan` validation, not
  `source.frame.switchback.plan`.
- `run(experiment, reader)` accepts only its default role/prior arguments (`UNSET`).
  The fixed `ContrastDecisionProcedure` owns the one decision method, fixed
  inference, family (`none`), alternative, null, and alpha. Passing
  `decision_method`, `sensitivity_methods`, or `prior` raises
  `CapabilityError` with code `readout.contrast.override` before source access.

For parallel A/B analyses, use `from_frame` (summary or panel shape) or
`from_definitions`; their arm-moment
results and role-aware method controls are separate from this contrast family.

## Optional finite-sample references and research laws

For a paired unadjusted mean under independent unit-cycle orders, a
prospective residual-variance upper bound can replace the sample-based
reference. Supply `contrast_references={metric_name: envelope}` to
`from_switchback_frame`. A `UnitCycleVarianceEnvelope` binds the assignment,
window, metric and arms, response meaning, cycles per unit, variance bound
and its provenance: who declared it and why. Software does not prove it
true. Without it the default paired unit-t approximation needs no
certificate for inference or pilot planning.

!!! warning "A pilot variance is not a prospective bound"
    A pilot variance or observed standard error is not a prospective upper
    bound, and the default unit-t reference is not a finite-sample envelope
    guarantee; its small-sample calibration failures below remain
    unresolved.

Complete-law declarations live in `increment.semantics.unit_cycle`, and their
oracle solvers in `increment.power.unit_cycle`; they are not root or ordinary
power-planning exports.

### The envelope interval

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
cutoff, provenance, and the side of the null the tested residual lies on
(`residual_side`: `greater`, `less`, or `split` for a residual of exactly
zero). Degrees of freedom are numeric null with reason
`not_applicable`; sample SE remains descriptive. Evidence tests
`mean(A)-null_abs*mean(G)` with the matching Markov/Cantelli probability
and admission allocation, rounded upward to binary64. Rejection means
this public p-value is strictly below `alpha`; the separately
outward-displayed interval and cutoff do not define that event. The
direction of a rejection is `residual_side`, taken from the exact
residual; a residual rebuilt from the rounded point and slope can have
the other sign. Frames, readout rows and wire serialization preserve
these fields and numeric nulls. Readout `stat_sig` and the interval
interpretation use that same p-value decision and side, not interval
exclusion: a rejection whose displayed interval still includes the null
reads as a rejection and says the interval is rounded outward, never that
it excludes the null. An envelope result saved before
`residual_side` existed refuses to load with
`estimation.contrast.legacy_envelope`; recompute it from the source data.

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

## Complete-law power planning

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

The paired independent-unit comparisons below weight by CT/TC probability,
then average equally over cycles within each unit and over independent
units. Supplied responses may be totals or pre-normalized means from equal or unequal
underlying micro-observation counts; the runtime receives neither
micro-counts nor a second weighting selector, and unequal retained
schedule-row counts remain outside the complete-schedule contract.

!!! warning "Totals and means define different responses"
    Each response meaning needs its own potential outcomes and effect units.
    In the calibration fixtures, the treatment increment is defined on the
    supplied aggregate scale. Changing from totals to means therefore changes
    the scientific response model; it is not a lossless conversion.

### The simulation scenario and manifest

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

### Coverage with few units or blocks

A nominal 95% interval need not cover in 95% of small switchbacks. Each
row below represents 512 attempted experiments at zero treatment effect.
`True/discarded` counts contaminated post-washout steps and the steps the
analyst discarded. The point and interval were available on the same draws.

| Independent replicates | CT probability | True/discarded | Returned | Covered | Coverage given interval | Coverage per attempt | Mean returned estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 units | .5 | 0/0 | 512 | 445 | .869 ± .015 | .869 ± .015 | −.043 ± .140 |
| 4 units | .5 | 1/0 | 512 | 445 | .869 ± .015 | .869 ± .015 | −1.548 ± .172 |
| 4 units | .5 | 1/1 | 512 | 447 | .873 ± .015 | .873 ± .015 | −.029 ± .094 |
| 4 units | .8 | 0/0 | 512 | 302 | .590 ± .022 | .590 ± .022 | .053 ± .167 |
| 4 units | .8 | 1/0 | 512 | 302 | .590 ± .022 | .590 ± .022 | −1.423 ± .230 |
| 4 units | .8 | 1/1 | 512 | 309 | .604 ± .022 | .604 ± .022 | .028 ± .114 |
| 6 shared blocks | .5 | 0/0 | 498 | 496 | .996 ± .003 | .969 ± .008 | −.043 ± .101 |
| 6 shared blocks | .5 | 1/0 | 498 | 485 | .974 ± .007 | .947 ± .010 | −1.538 ± .108 |
| 6 shared blocks | .5 | 1/1 | 498 | 496 | .996 ± .003 | .969 ± .008 | −.014 ± .068 |
| 6 shared blocks | .8 | 0/0 | 382 | 382 | 1.000 ± .000 | .746 ± .019 | −1.241 ± .116 |
| 6 shared blocks | .8 | 1/0 | 382 | 382 | 1.000 ± .000 | .746 ± .019 | −2.684 ± .137 |
| 6 shared blocks | .8 | 1/1 | 382 | 382 | 1.000 ± .000 | .746 ± .019 | −.846 ± .079 |

`±` is one estimated Monte Carlo standard error, not an interval for an
individual study. A zero estimated MC error when every returned interval
covers is not proof of perfect coverage. Returned-estimate means condition
on schedule admission and need not have the unconditional design expectation.

In the four-unit skewed design the nominal 95% interval covered only 59%
of attempts (Monte Carlo standard error 2.2 percentage points). In the
skewed shared design every *returned* interval covered, but 130 attempts
had no interval. Even before observing outcomes, the probability of both
orders occurring there is only `1 - 0.8**6 - 0.2**6`, about 74%.

To reproduce, use `run_switchback_end_to_end` at revision `77ab667b`
with `SwitchbackScenario(seed=20261119, treatment_effect=0, unit_effect_sd=0,
temporal_effect=2, noise_sd=1, washout_steps=1, observation_steps=3,
carryover_amplitude=3)`. Use one cycle per unit in the four-unit design,
or `shared_schedule=True`, three roster members and six cycles for shared
blocks. Set `true_carryover_order` and `declared_carryover_order` as shown.
Discarding the genuinely contaminated step reduced bias in the balanced
unit design; this checks a specified contamination model, not whether
carryover has ended in an observed experiment.

### Representative release checks

!!! warning "The unit-cycle campaign has not been executed"
    The thresholds below are acceptance bands the campaign is *written to
    enforce*, not results it has returned. No coverage, availability or power
    figure here has been measured on the manifest it describes; read them as
    the specification of a pending check.

The ordinary suite uses deterministic representatives covering every manifest
axis value, plus explicit small-N/skew/dependence intersections. It does not
claim that a representative run executed the full matrix.

### Representative cells and acceptance bands

Public-runtime calibration uses 1,536 attempts per checkpoint for these
cases:

- **Small, skewed design:** N=4, C=1, CT probability=.9, Normal
  innovation, reuse=0, unequal micro-counts, retained-total response.
- **Reused, skewed noise:** N=10, C=5, CT probability=.75, centered
  lognormal innovation, reuse=.9, unequal micro-counts, pre-normalized mean.
- **Larger design:** N=40, C=1, CT probability=.25, centered Gamma
  innovation, reuse=.5, equal micro-counts, retained-total response.

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

### Campaign execution and records

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
