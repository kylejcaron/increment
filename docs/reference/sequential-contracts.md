# Sequential registration and capture contracts

Use this reference when integrating a source, saving or continuing a checkpoint,
or interpreting the construction behind a sequential result. For the ordinary
plan-derived workflow, start with [Sequential inference](../guides/sequential-inference.md).
Family selection and freezing are documented in [Family inference](family-inference.md).

## Manual registration

A hand-written registration is needed only when the plan-derived declaration
cannot express your integration. A posterior effect prior, estimated log standard
error or freshness timestamp cannot replace a sampling declaration.

The exact public observation model is Bernoulli with a proper Beta prior.
Scalar Gaussian normal–inverse-gamma and paired Gaussian normal–inverse-Wishart
kernels are private research diagnostics: they do not produce certified
`EValueEvidence` or public sequential decisions. Those Gaussian models allow
unequal unknown arm variances and require positive population control means
(and positive denominator means for ratios); positive observed means do not
establish those assumptions.

```python
from fractions import Fraction

from increment.estimators import (
    AlwaysValid,
    JointReveal,
    PredictivePrior,
    SequentialCell,
    SequentialModel,
    SequentialRegistration,
    capture_sequential_snapshot,
    estimate_sequential,
)

prior = PredictivePrior(kind="beta", a=1, b=1)
registration = SequentialRegistration(
    source_id="checkout-experiment",
    definitions_id="immutable-observation-definition-v1",
    control_group="control",
    committed_before_data=True,
    reveal=JointReveal(
        filtration_id="finalized-randomization-units-v1",
        independent_unit_vectors=True,
        simultaneous_metrics=True,
        outcome_independent_order=True,
        immutable_finalized_outcomes=True,
        longest_window_days=14,
    ),
    models=(
        SequentialModel(
            metric="conversion",
            law="bernoulli",
            control_prior=prior,
            treatment_prior=prior,
            positive_population_control=True,
        ),
    ),
    roster=(SequentialCell(metric="conversion", group_id="treatment", alpha=Fraction(1, 20)),),
)
policy = AlwaysValid(registration=registration)
# Read outcomes only after committing the declarations above.
finalized_joint_records = [
    {"unit_id": "u-control", "group_id": "control", "values": {"conversion": 0}, "segments": {}},
    {
        "unit_id": "u-treatment",
        "group_id": "treatment",
        "values": {"conversion": 1},
        "segments": {},
    },
]
snapshot = capture_sequential_snapshot(
    registration,
    finalized_joint_records,
    source_id=registration.source_id,
    definitions_id=registration.definitions_id,
    finalized=True,
)
decision = estimate_sequential(snapshot, policy)
```

Each record has canonical string `unit_id` and `group_id`, every registered
metric in `values`, and any registered `segments`. Scalar-mean observations
are scalars; ratios and adjusted laws retain the vectors listed
[below](#adjusted-means-and-ratio-metrics-the-joint-laws). Reveal all relevant
metrics together after the longest outcome or uptake window, in
outcome-independent order, then keep them immutable. These are assumptions,
not conclusions drawn from timestamps.

### Source and definition identity

For frames, compute `sequential_definition_id` from the synthesized metrics,
design, `MetricSpec` transformations and required `source_mapping` before
constructing the source. A panel mapping is
`{"unit": "user_id", "group": "variant", "uptake": None, "date": "ds", "exposure_date": "enrolled_on"}`.
Summary frames also require an `exposure_date` role. Uptake must name its
resolved column. Reveal order is ascending exposure date, then canonical
string unit ID; null exposure follows `on_unassigned`.

Native registrations bind the full exposure, fact, dimension, timestamp,
entity and experiment recipes. Artifacts retain the same digest in saved
context, not the recipes themselves. A mismatch refuses **initial capture**,
not just continuation. Bind a manual registration with
`AnalysisPlan(inference=InferenceSpec(kind="always_valid", registration=registration))`.

## Capture and continuation

Unit-summary frames supply finalized outcomes. Panels, native warehouse
sources and artifacts need
`increment.workflows.capture_sequential(experiment, reader, finalized=True, as_of=...)`
before a decision. Reading first raises `sequential.capture.required`, whose
`route` names that call. `as_of` is required on panel and relational sources;
the common reveal window determines which units can enter. The reader's source
keeps the capture, so later `run(experiment, reader)` reads it. Native and
artifact capture streams bounded Arrow batches through source relations.

Retrieve the exact prefix with `increment.workflows.sequential_snapshot`.
Pass it as `previous=` when continuing from a new source object. This complete
example saves four units, then proves that the eight-unit source extends them:

```python
import increment
import polars as pl
from increment.estimators import snapshot_from_json
from increment.workflows import capture_sequential, sequential_snapshot

design = increment.Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
    allocation_scheme="independent",
)
plan = increment.AnalysisPlan(
    primary="purchase", inference=increment.InferenceSpec(kind="always_valid")
)
frame = pl.DataFrame(
    {
        "user_id": [f"u{i}" for i in range(8)],
        "variant": ["control", "treatment"] * 4,
        "enrolled_on": list(range(8)),
        "purchase": [0, 1, 1, 0, 0, 0, 1, 1],
    }
)


def bind(frame):
    return increment.from_frame(
        frame,
        shape="summary",
        unit="user_id",
        group="variant",
        design=design,
        metrics=[increment.MetricSpec(name="purchase", type="conversion")],
        experiment_id="continued-checkout",
        plan=plan,
        exposure_date="enrolled_on",
    )


experiment, reader = bind(frame.head(4))
with reader:
    saved = sequential_snapshot(experiment, reader).model_dump_json()

experiment, reader = bind(frame)
with reader:
    parent = snapshot_from_json(saved)
    continued = capture_sequential(experiment, reader, finalized=True, previous=parent)
    result = increment.run(experiment, reader)
print(continued.parent_id == parent.prefix_id)
```

<!-- output -->
```text
True
```

The same snapshot is idempotent. Appending must preserve every earlier unit's
identity, assignment, value, definition, transformation and reveal position.
A changed generation hash alone proves nothing about an append.

Corrections raise `sequential.continuation.rewrite`. Context identifies the
first differing `position` (zero-based), the earlier `unit_id`, and `changed`:

- `missing`: the unit is no longer revealed.
- `order`: it appears later, for example after a late-arriving unit.
- `assignment`: its assigned group changed.
- `values_or_definitions`: its values or definitions changed.

If records are identical but retained states differ, `position` and `unit_id`
are `None`. Never treat a correction as a fresh continuation of the old process.
Raw-record producers may use `capture_sequential_snapshot(..., previous=parent,
append=True)` for **new units only**. Full-prefix capture verifies every old
record and reuses the parent's exact sufficient state.

Native capture and artifact publication pin assignment, outcomes, uptake,
property streams, freshness and coverage in one backend execution. Reductions
use those pinned inputs. Cluster identity is checked on raw admitted assignments;
cluster-grain inference remains fixed-horizon.

## Checkpoint encoding

`increment.workflows.export_moments(experiment, reader, path)` writes a format-10
sequential envelope containing version-2 registration/snapshot/checkpoint,
version-3 compiled plan and originating source identity. Use the reader that
captured or adopted the checkpoint; a fresh `from_definitions(...)` pair does
not own it. Reload with `from_moments`.

Sequential format 9 remains readable. Fixed-horizon format 12 carries the same
source identity; fixed-horizon format 11 remains readable. See the central
[saved-format index](saved-formats.md) for version/migration lookup.

### Rational grammar and limits

Every rational—prior hyperparameters, `rho`, `treatment_probability`, cell
`alpha` and `null_lift`, `q`, adjustment coefficients/centers, retained arm
means and scatter entries—uses a signed integer or integer quotient with a
positive denominator, as printed by `Fraction`. Fixed decimal strings such
as `"0.05"` are also read.

JSON accepts these strings or bounded integer literals, never Booleans or
floating literals. Write `"0.05"` and `"0.00001"`, not `0.05` or `1e-5`:
the latter have already been rounded by the JSON parser. Each integer
component is limited to 4,096 decimal digits; scientific notation is not a
portable spelling.

Invalid grammar raises `sequential.wire.rational_invalid` (`InvalidRequestError`),
with `field`, `reason`, `limit_digits` and `route_forward`, **before arithmetic**.
Thus `"1e5000000"` cannot allocate a five-million-digit integer. Non-JSON text
raises `sequential.source.invalid`. Non-finite rational inputs use the rational
error too. Export beyond the ceiling refuses with `field="export"` instead of
writing an unreadable checkpoint.

The limit covers supported captures: a binary64 observation is an exact dyadic
rational with at most 309 numerator digits and a 1,075-bit denominator; retained
means and scatters stay near 1,300 digits even when the largest and smallest
finite values alternate. In-process `Fraction` and finite `Decimal` values
keep their exact meaning. Python floats normally bind their exact binary64
value; compliance `alpha`/`null_lift` and `InferenceSpec.baseline_rate` instead
bind their shortest decimal spelling (`0.05` becomes `1/20`).

## Exact conversion route details

Automatic `always_valid` registration creates a Bernoulli model per conversion
or retention metric before reading outcomes. Both arms use
`Beta(w * p0, w * (1 - p0))` when `baseline_rate=p0`, with
`DEFAULT_BERNOULLI_PRIOR_WEIGHT = 10`; omitting the rate uses `Beta(1, 1)`.
Use pre-experiment control rates, not observed experiment outcomes.
Every proper prior is valid; misspecifying the rate can cost power.

For uptake-only encouragement with no outcome metrics, `baseline_rate` describes
control uptake. With outcomes present it tunes only their priors; compliance
keeps its flat uptake prior. Explicit registrations retain their own priors.

Each declared metric/treatment-arm pair gets a cell; segmented frames add a
cell per declared level, including absent levels. Primary allocation divides
across treatment arms; secondary metric/arm cells share `q`, capped at their
procedure allocations; segmented e-BH uses the breakout view's `q`.

Identity binds metrics, design, source mapping, baseline rate and declared
cells. Unchanged declarations continue an existing process; changed membership
or prefixes refuse. Existing two-arm identities are retained. Multi-arm
asymptotic monitoring and multi-arm sequential-secondary power planning remain
unsupported.

YAML can write `inference: {kind: always_valid, baseline_rate: 0.03}`; the float
binds the decimal as typed. A mean or ratio under registration-less
`always_valid` raises `definition.inference.always_valid_metric_type`, naming
the metric and `asymptotic_mean` route. `baseline_rate` with another kind or an
explicit registration raises `definition.inference.baseline_rate_route`.

### Choosing the Beta prior weight for a declared baseline rate

The default weight is a robustness choice, not the weight maximizing measured
power when the baseline is correct. Low power for rare conversions does not
establish the asymptotic route's calibration at those rates.

??? info "Prior-weight measurements"
    `calibration/bernoulli_prior.py` used 200 replications per cell, looks every
    250 units per arm through 4000, and `alpha=0.05`. Type I is the fraction of
    null paths that ever crossed; power is the fraction crossing by the stated
    look. These are measurements, not a new validity theorem.

    In the tables, **power** lists lifts `.05/.10/.20` at 2000 per arm, then
    lift `.10` at 4000. **True rate doubled** lists type I at 4000 and power
    for lift `.10` at 2000. A dash means not measured.

    **Declared control rate 0.02**

    | Prior | Type I @4000 | Power | True rate doubled |
    |---|---|---|---|
    | Beta(1,1) | .000 | .000/.000/.000/.000 | — |
    | w=2 | .000 | .000/.000/.000/.000 | .000/.000 |
    | w=10 | .000 | .000/.000/.000/.000 | .000/.000 |
    | w=50 | .015 | .000/.000/.000/.000 | .000/.005 |
    | w=200 | .005 | .000/.005/.010/.000 | .000/.000 |

    **Declared control rate 0.1**

    | Prior | Type I @4000 | Power | True rate doubled |
    |---|---|---|---|
    | Beta(1,1) | .000 | .000/.000/.000/.000 | — |
    | w=2 | .000 | .000/.000/.000/.000 | .000/.000 |
    | w=10 | .000 | .000/.005/.015/.005 | .000/.000 |
    | w=50 | .005 | .010/.010/.080/.010 | .000/.005 |
    | w=200 | .010 | .005/.015/.105/.030 | .000/.000 |

    **Declared control rate 0.3**

    | Prior | Type I @4000 | Power | True rate doubled |
    |---|---|---|---|
    | Beta(1,1) | .000 | .000/.000/.335/.065 | — |
    | w=2 | .000 | .000/.010/.405/.035 | .000/.280 |
    | w=10 | .000 | .010/.020/.520/.140 | .000/.165 |
    | w=50 | .000 | .010/.015/.685/.185 | .000/.000 |
    | w=200 | .005 | .025/.100/.685/.250 | .000/.000 |

    The largest displayed null-crossing rate is `.015`, below `.05`. The probe's
    power-only rule—the smallest weight within five percentage points of the
    best at lift `.10` and 2000 units per arm for every declared rate—selects
    `w=200`. The default remains `w=10`: it is the largest tested weight that
    retains power when a declared `.3` baseline is actually `.6`. A confident
    caller may choose a heavier prior in `SequentialModel.control_prior` and
    `treatment_prior` on an explicit registration.

    Even the best measured power at lift `.20` and 2000 per arm is `.105` for
    rate `.1`, and `.010` for rate `.02`. This grid and the bounded release
    audit do not verify asymptotic rare-conversion calibration at rates `.10`
    or below.

## Asymptotic mean construction

The [guide's assumptions](../guides/sequential-inference.md#ordinary-continuous-monitoring)
apply to every law here. A 0/1 scalar's centered scatter is its Bernoulli
variance; no separate binary law is needed on `AsymptoticMean`. This is not a
reinterpretation of NIG/NIW priors, `AlwaysValid` or Gaussian log-SE planning.

Registration freezes source/definition identity, fixed allocation, models,
member list and allocated nominal alphas before capture. Tuning uses the
planned `expected_decision_sample_size` (`5000` by default): with
`t=N*rho^2`, solve `t-log1p(t)=-2*log(alpha)`. The default operational start
is two observations per arm, not a finite-sample or heavy-tail safeguard.
Changed mappings, assignments, membership or prefixes refuse continuation.
Definitions use `Experiment.allocation`; artifacts retain the bound registration
and never retune from outcomes. `ScalarMeanModel` remains available for
inspection and replay.

### Boundary and confidence-set geometry

The clock is `N=nc+nt` finalized independent units. For arm centered sums of
squares `M2a`, use `sa²=M2a/na` and `Va=sa²/na`. For null lift `ell0` and
ratio `r=1+ell0`:

```text
Dhat_r = meanT - r * meanC
Vr = VT + r² * VC
lambda = rho^-2
K_N = (1 + lambda/N) * (log1p(N/lambda) - 2*log(alpha))
boundary² = Vr * K_N
```

Invert `(meanT-r*meanC)² <= K_N*(VT+r²*VC)` over **all real** `r`.
`result.bounds.components` preserves bounded intervals, rays, disconnected,
full and empty sets in ratio coordinates. Subtract one from finite endpoints
for lift; there is no −1 lift floor. Rational centered moments, outward
logarithms and rational root enclosures avoid raw-sum cancellation, overflow
and arbitrary near-singularity tolerances. A point may be unavailable while
the confidence set remains available.

Before both arm starts, with fewer than two observations in either arm, or
with zero observed variance, the set is full with a precise availability
reason. This is not evidence of zero population variance. Empty geometry
withholds decisions. Fixed `rho` and start are part of model, registration and
checkpoint identity. The claim is an asymptotic confidence-sequence
approximation for each fixed supported distribution, following the
arm-mean time-uniform linearization and
[Waudby-Smith et al.'s AsympCS framework](https://arxiv.org/html/2103.06476v7).
No delayed-start coverage-limit claim or universal finite-start guarantee
is made.

### One-sided boundaries

`greater` requires `Dhat_r>sqrt(Vr*K_N)`; `less` requires
`Dhat_r<-sqrt(Vr*K_N)`. Outside a family, a one-sided cell uses `K_N` at
`2*alpha` and one tail of that symmetric boundary. By symmetry, that tail's
crossing probability is at most `alpha` plus half the probability of crossing
both tails at different looks. A Gaussian-null simulation at `alpha=.05`
over 20,000 paths through 20,000 units measured upper-tail crossing `.033`
and no both-tail crossings; this is not a finite-sample theorem. The reported
ray's endpoint equals the two-sided endpoint at `2*alpha`.

A family member instead uses `K_N` at `alpha`: its set contains exactly the
ratios whose sign-gated plug-in mixture value stays below `1/alpha`, matching
the evidence used in family selection. At the same reporting level,
`row.stat_sig()` and family evidence agree; selection and individual
significance still [differ when the nominal cap binds](family-inference.md#fcr-re-estimation-of-discovered-cells).
A one-sided non-family cell with `alpha >= 1/2` raises
`sequential.asymptotic_mean.invalid`.

`ArmPlanningProcedure.standard(inference=InferenceSpec(kind="asymptotic_mean"))`
uses this boundary: `alpha` for one-sided secondaries, `2*alpha` for one-sided
primaries or guardrails. It sizes unsegmented monitoring and rejects
`InferenceSpec.segments`; whole-experiment power does not establish segment
power. Low-level log-SE Gaussian power solvers instead reject `AsymptoticMean`
with `sequential.route.unsupported`: those approximations do not size the
count boundary.

### Evidence and exports

`AsymptoticSequentialEvidence` retains checkpoint, actual arm counts, model,
null, direction, allocation, start, tuning, geometry and decision.
`result.log_e` is sign-gated **asymptotic** log evidence, not a finite-sample
e-value or p-value. Ratio evidence is capped by denominator stability.
Frame exports include `sequential_validity_regime`, float `sequential_alpha`,
`sequential_components` and a **numeric-null** `sequential_log_e`, rather than
mislabeling that plug-in value as exact evidence. JSON replay recomputes
geometry and rejects changed evidence. Identical capture is idempotent;
frozen results require a verified ancestor prefix.

On exact Bernoulli rows, `sequential_log_e` is a float log e-value, usable even
when exponentiation would overflow. `sequential_alpha` is a float; exact
fractions, the certificate, checkpoint and ratio-coordinate bounds remain on
`row.require_sequential_result()`. `sequential_lower`/`sequential_upper` are
lift-scale endpoints rounded outward to floats or nulls. No equivalent-normal
p-value is reported. Both tails of an exact two-sided sequence use `1/alpha`,
not a halved central level: at the true ratio they are bounded by one test
martingale, so Ville's inequality bounds their union by `alpha`.
`row.stat_sig()` tests whether the registered null lies outside that sequence.
See [result fields](result-fields.md#sequential-inference-two-coordinate-systems).

## Adjusted means and ratio metrics: the joint laws

| Law | Per-unit vector | Compared quantity |
|---|---|---|
| `scalar_mean` | `(Y)` | Arm means; mean, conversion or retention |
| `adjusted_mean` | `(Y, X)` | CUPED-adjusted arm means |
| `ratio_mean` | `(N, D)` | Arm ratios `E[N]/E[D]` |
| `adjusted_ratio_mean` | `(N, D, X)` | Adjusted arm ratios |

The whole vector needs finite `2+delta` moments and positive limiting variance
of its linearized contrast. Adjusted laws also need a positive within-arm
covariate variance and a pre-assignment covariate `X`. Ratio laws require
population denominator means bounded away from zero and
`positive_population_denominators=True`.

A summary frame supplies `X` through
`decision_method=Method.cuped(covariates=CupedCovariate(...))`; native definitions
derive it from `n_pre_periods > 0`. Artifacts require the adjusted metric's
published `cuped_preperiod` extension, absent from default publication.
Panels support bounded-window, unadjusted `ratio_mean`, but refuse
`adjusted_mean` and `adjusted_ratio_mean`: they lack a per-unit pre-period
covariate. See [CUPED source selection](../guides/cuped.md#the-standard-coefficient-on-the-asymptotic-route).

The joint laws linearize `f_t-r*f_c` at observed means. Gradients are affine in
`r`; `V(r)` is their quadratic form over each arm's scatter divided by `n_a²`.
The scalar solver inverts `(f_t-r*f_c)² <= K_N*V(r)`.
Adjusted laws fit the standard inverse-n-weighted within-arm CUPED slope at
every look and anchor both arms on the pooled covariate mean; gradients retain
that coupling. Ratio laws retain the `-2 Cov(N,D)` cross term, and adjusted
ratios also retain shared-covariate covariance.

Retained state, coefficients, anchors, gradients, quadratic coefficients and
`K_N` are exact. The boundary and nuisance-plug-in linearization are asymptotic;
under the contract, the linearization error is lower order than boundary width
([Lindon, Ham, Tingley and Bojinov 2022](https://arxiv.org/abs/2210.08589);
[Schmit and Miller 2022](https://svenschmit.com/assets/pdf/code_2022_ci.pdf)).
No snapshot version change or migration of existing `scalar_mean` state was
needed for these new vector shapes.

`MetricSpec(missing="zero")` replaces null/NaN outcome coordinates with exact
zero before capture, including a ratio denominator when declared.
`CupedCovariate.missing` separately controls covariate preparation and never
changes outcomes. Infinities are not silently clipped.

### Denominator availability and selection

Zero or insufficiently separated observed denominators leave the **cell**, not
the whole family, unavailable with `zero_denominator_mean` or
`denominator_near_zero`. Ratio-family log evidence depends on the stopped
state, not inversion alpha, and is capped by denominator stability. This cap
is dominated by the underlying mixture value; it is not itself a martingale.

Every member remains counted. A ratio unresolved at the nominal reporting
ceiling contributes no selection evidence; selected reporting levels must
also resolve its denominators. A ratio unavailable at a tighter registered
allocation can become reportable at a later, wider selected level, even when
frozen: reinversion uses the same stopped state and refreshes the reason.
Before-start/missing-arm states, nonpositive denominators and degenerate
variances still abstain regardless of alpha.

## Supported sources and transformations

The scalar model follows summary, panel, native, artifact, moments replay,
current as-of and registered breakout paths. Current labeled checkpoints
round-trip through moments and as-of readouts. Frame day labels may be dates,
datetimes, numeric days, ISO dates or structured labels such as `d14`; their
representation is preserved.

Binary-uptake LATE, quantiles, clustered sequential inference, repeated units,
switchback, observational causal inference and additive sequential decisions
are unsupported. These requests require matching constructions and refuse
before source reads.

Transforms must be predictable at each reveal **and** preserve the sampling
law. A fixed winsorization cap meets predictability, not automatically the law.
An accumulated-data percentile cap raises `sequential.transform.unpredictable`.

A coefficient fixed from pre-period data (`InferenceSpec.adjustments`,
`ScalarMeanModel.adjustment`) applies at capture on the scalar asymptotic
route, not exact Bernoulli. Its common shift cancels at zero effect, but changes
relative lift and nonzero-null margins if the experiment covariate mean differs
from the declared center; that bias does not vanish with sample size. Fitted
in-experiment coefficients require the adjusted laws above. `AlwaysValid`
refuses fitted or predeclared CUPED; refitting would reweight past increments
and break the exact martingale. See [CUPED](../guides/cuped.md).

### Encouragement and uptake

Raw randomized Bernoulli intent-to-treat (ITT), asymptotic scalar-mean ITT
under Randomized or Encouragement designs, and two-sided-design Bernoulli
relative uptake are supported. Uptake needs an explicit
`SequentialCompliancePolicy(alpha=Fraction(1, 20), alternative="two-sided", null_lift=0, family=False)`
in `AnalysisPlan(compliance=..., ...)`. Cells must match it; alpha divides
across treatment arms. Request `estimands=("itt",)` or
`estimands=("compliance",)` explicitly. Encouragement as-of requests require
`completed_windows_only=True`, even before capture.

These requests may omit `ExclusionRestriction`; it is required for LATE.
Omitting `estimands` still requests LATE and refuses without the declaration.
Declaring exclusion does not enable sequential LATE. Structural-zero control
uptake is a rate target: use fixed-horizon compliance.

Compliance-only capture uses the uptake declaration and finalized enrollment
cohort, not the outcome table. Its date series follows enrollment and uptake
coverage; an outcome timestamp cannot extend it. Whole-window encouragement
may combine asymptotic ITT with exact Bernoulli uptake; `family=True` joins
uptake to the same e-BH family, labeled `asymptotic_sequential`. Other mixtures
of exact and continuous models require separate registrations.

### Predeclared segments

On summary/panel frames, declare every categorical level before outcomes, for
example `InferenceSpec(kind="always_valid", segments={"country": ("US", "CA", "GB")})`.
Only declared levels enter; unseen declared levels stay empty members.
Asymptotic segment membership must be fixed before assignment.

Read a `View.breakout(...)` over the declared dimension. Whole-window and as-of
lift views raise `sequential.source.invalid` with that route. Exact breakouts
use declared correction (e-BH by default); asymptotic breakouts use fixed-roster
Bonferroni by default and refuse BH. Both retain the explicit uncorrected
option. Manual registration cannot be combined with `InferenceSpec.segments`.
See [family construction](family-inference.md#segmented-views).

Boolean labels are `"true"`/`"false"`, missing labels `"__null__"`; strings keep
case/content (`"True"` differs from `"true"`), and numerics use existing string
conversion. A Boolean column with no matching declared canonical level raises
`sequential.registration.invalid`, naming `dimension`, `declared` and
`canonical`. String columns allow unseen/undeclared levels. Summary, panel and
fixed-horizon breakouts use these same labels on pandas, Polars and Arrow.

Older summary checkpoints recorded backend spellings such as `True` or `None`.
They still replay unchanged via `from_moments`, but continuing with canonically
relabeled units raises `sequential.continuation.rewrite`. Do not spend alpha
again merely because continuation refused. Keep the old checkpoint, continue
with the old spellings as strings, or design and register a new protocol whose
validity you can defend.

Segments cannot combine with compliance `family=True`: use `family=False` for
separately allocated uptake, or omit segments for a joint whole-window family.
Existing encouragement breakout restrictions remain. `from_definitions` and
`from_unit_day_artifact` refuse segmented capture pending an immutable
property-relation contract. `from_moments` replays segmented state but has no
breakout catalog (`evidence.breakout_unavailable`); as-of output does not infer
segmented historical checkpoints from a moment cube. Segment heterogeneity
and contrasts need a separate covariance construction: use registered per-cell
sets or fixed-horizon heterogeneity/contrasts instead.

## Triggered populations

A trigger-declared experiment has an assigned chain and a separately identified
triggered chain in `snapshot.triggered`. A unit enters the latter after its
first eligible trigger is observed and its longest joint reveal window closes.
That window can include longer assignment-anchored uptake even though uptake
is not in the triggered chain. Reveal order is `(first trigger, unit)`;
outcomes are trigger-anchored and pre-period covariates assignment-anchored.

`capture_sequential(finalized=True, as_of=d)` captures both in one pinned
execution on native/artifact sources. Artifacts need `trigger_population`,
`trigger_measure_stats` and triggered `assignment_counts` extensions.
`from_moments` can replay both chains, not start one; frames have no trigger
declaration. `View.overall(population="triggered")` or
`View.asof(population="triggered")` reads that chain; add an assigned view to
read both populations.

Distinguish an incomplete trigger feed/watermark, an observed trigger whose
outcome window has not matured, and a captured row with an unavailable point.
Wait for completeness and maturity, then read its persisted availability
reason: a null point may coexist with interval evidence. None of these states
establishes treatment-independent triggering. Counterfactual logging must
record triggers without regard to treatment; equal observed rates cannot show
which units would have triggered in either arm.

`InferenceSpec.triggered_registration` is derived before observations, never
hand-declared. It has the assigned outcome models, cells and tuning, but its
own population, definitions and reveal identity, and no uptake model. Retaining
assigned tuning makes the count boundary wider than optimal by the trigger
rate, without changing its validity regime.

### What the triggered claim assumes

- **A1:** Bernoulli randomization independent of everything else.
- **A2:** Neither trigger membership nor timing depends on assignment. The
  contrast is then ITT on the triggered subpopulation. Equal trigger rates do
  not establish this: treatment could change which units trigger.
- **A3:** Trigger timing carries no information about the complete retained
  vector—outcome, ratio numerator/denominator and pre-period covariate. Thus
  the sequence is stationary in trigger order.

Under A1–A3, calendar looks reveal prefixes of one fixed iid sequence.
`srm(..., population="triggered")` and `trigger_rates(...)` from
`increment.workflows` can reject some violations, not establish A2.
Assignment-window uptake is mechanically tied to trigger timing, outside A3.
Triggered compliance remains unavailable with `readout.cell.unsupported_request`,
reason `triggered_uptake_unsupported`; uptake-only plans derive no triggered
registration.

### A practical triggered-health review

Call `increment.workflows.health_check(experiment, reader)`. Review
`assigned_allocation`, then `triggered_balance` and descriptive `trigger_rates`
separately. Check metric maturity and every feed's certified completeness
watermark. Unavailable/not-applicable is not a pass; observed event times do
not certify a feed.

Read `trigger_assumptions`: rates prove neither counterfactual invariance,
absence of pre-trigger treatment effects nor stationarity.
`triggered_look_predictability` is also assumed, not checked: family looks,
stops and freezes may use only already-revealed joint triggered records.
The report is a sanity check, not a causal-validity or ship verdict.
See the [health-check API](../api.md#diagnostics-and-robustness).

### Per-cell coverage and the family verdict

Per-cell path guarantees hold for any look schedule under A1–A3, in the
registered exact/asymptotic regime. Family FDR and selected intervals need
stopped evidence to remain valid: looks, stops and freezes must be stopping
times for the **joint triggered filtration**, meaning they use only information
already revealed by that chain. A predeclared schedule qualifies.

Correct triggering and no pre-trigger treatment effect do not establish this
condition. With three-day windows, triggers one day after assignment and
outcomes only on trigger day, assigned and triggered outcomes coincide—but
the assigned window closes a day earlier. Freezing before an unfavorable
triggered block based on assigned results uses future information.

Do not use **any** assigned result revealing future triggered outcomes,
including the same metric, to time triggered family decisions. Such lookahead
can inflate stopped e-value expectation without changing the per-cell crossing
bound (`tests/estimation/test_triggered_sequential_validity.py`). Under A1–A3,
correct triggering and the stopping condition, the family retains
`MultiplicityFamily.guarantee="fdr"` and the assigned family's
`multiplicity_status`/`family_guarantee` labels. These are conditional claims,
not verification of the analyst's rule. There is no joint correction across
the two populations.

### Certified feeds, not evidence presence

Every contributing trigger, outcome, denominator and uptake feed needs a
`SourceSnapshotEvidence` watermark whose certified whole day reaches `as_of`.
Otherwise capture raises `sequential.source.invalid`, naming the feed and
certified day. An inclusive cutoff during `as_of` certifies only the previous
day. A SQL-declared trigger has no certifiable feed and refuses; declare it
through a fact.

Later looks must only append. A late trigger or outcome inside certified days
changes an earlier record and raises `sequential.continuation.rewrite`; capture
cannot retract a decision already reported.

### Processes begun without a triggered commitment

An old checkpoint without a triggered chain continues assigned-only, with
triggered cells marked `triggered_chain_uncommitted`. Attaching a chain raises
`sequential.continuation.legacy`: the monitoring choice would follow exposure
to assigned outcomes. Triggered monitoring needs a new registered process,
not a retroactive addition.

The derivation is kept out of assigned identity, so assigned records and
checkpoints have the same identities with or without triggered commitment.
An artifact published before the derivation existed can derive the same
triggered registration from its stored context.

## Explicit research campaign

The bounded sequential research campaign is separate from ordinary tests:

```bash
uv run python -m calibration.sequential \
  --output /tmp/sequential-campaign \
  --profile smoke \
  --budget-seconds 3600 \
  --case-seconds 60
```

Use a new output directory. The shared hard-budget runner writes an immutable
manifest and per-case checkpoints labeled completed, failed, timeout or
unresolved. Failures and unavailable numerical cases are not successes, and
the campaign does not extend supported models or transformations.

Replication journals record selection, availability, certified-interval and
point-reason counters, not interim finite-look confidence coverage.
`--case 0 --repetitions 1` runs a one-draw diagnostic: a completed draw is not
certification; the CLI exits 2 while the campaign remains incomplete even if
its controller/workers exit 0. Deterministic stopping/replay tests verify
implementation contracts, not empirical calibration or a theorem.

## Primary references

The public Bernoulli process assumes a predeclared common joint-unit reveal:
all relevant outcomes arrive together, in committed order, and stopping may
occur at any observed prefix. This is the setting of
[Ville's maximal inequality](https://doi.org/10.1007/BF01503646) and
[Howard et al.'s time-uniform concentration framework](https://arxiv.org/abs/1810.08240).
Private Gaussian diagnostics are not covered by that public anytime-valid
contract. Asymptotic construction sources are linked with the boundary and
joint laws above; family-selection sources are in [Family inference](family-inference.md).
