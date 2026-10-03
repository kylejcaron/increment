# Registered sequential inference

`AlwaysValid` consumes finalized raw-observation checkpoints. It requires
a `SequentialRegistration` committed before any data access. For a
conversion or retention metric, the source builds that registration from
the plan alone (see [sequential monitoring of a conversion metric](#sequential-monitoring-of-a-conversion-metric));
the hand-written form below covers every other exact declaration. A
posterior effect prior, an estimated log standard error, or a freshness
timestamp cannot replace this registration.

The exact public anytime-valid observation model is Bernoulli with a proper Beta
prior. Scalar Gaussian normal–inverse-gamma and paired Gaussian
normal–inverse-Wishart kernels remain available only through the private
research-diagnostic route; they do not produce certified `EValueEvidence` or
public sequential decisions. Gaussian arms may have different unknown
variances. Declare positive population control means, and positive denominator
means for ratios; positive observed averages do not establish these assumptions.

```python
from fractions import Fraction

from increment import (
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
    roster=(
        SequentialCell(
            metric="conversion",
            group_id="treatment",
            alpha=Fraction(1, 20),
        ),
    ),
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

Each record supplies canonical string `unit_id` and `group_id`, a `values`
mapping with every registered metric, and any registered `segments`. The
continuous public route accepts scalar values, not `(numerator, denominator)`
pairs. Reveal order must be independent of outcomes. Reveal all relevant
metrics together after the longest required outcome or uptake window, then
keep outcomes immutable. These are sampling declarations, not conclusions
drawn from dates.

For frame analyses, compute `sequential_definition_id` from the synthesized
metrics, design, `MetricSpec` transformations, and required `source_mapping`
before constructing the source. For a panel, the mapping is:
`{"unit": "user_id", "group": "variant", "uptake": None, "date": "ds",
"exposure_date": "enrolled_on"}`. Summary frames must also provide an
`exposure_date` role. Reveal records in ascending exposure-date order, with
ties broken by canonical string unit id, never caller row order or outcomes.
Null exposure follows the constructor's `on_unassigned` policy. Uptake must
name its resolved column. Native registrations bind a digest of the full exposure,
fact, dimension, timestamp, entity, and experiment recipes; artifacts carry that
same digest in their saved context and never store the recipes themselves. A
mismatch refuses initial capture, not only continuation.

Bind the registration through `AnalysisPlan(inference=InferenceSpec(
kind="always_valid", registration=registration))`. Unit-summary frames
supply finalized outcomes. Panel, native, and artifact sources require
`analysis.capture_sequential(finalized=True, as_of=...)` before a decision.
The common window determines which units can enter. Native and artifact
capture streams bounded Arrow batches through existing source relations.

Use `analysis.sequential_snapshot()` to retain the exact prefix. Supply it
as `previous=` when continuing from a new source object. The same snapshot
is idempotent. An append must preserve every earlier unit identity,
assignment, value, definition, transformation, and reveal position. A
changed generation hash alone proves nothing about an append. Corrections
refuse continuation; never treat them as a fresh continuation of the old
process.

Raw-record producers may use `capture_sequential_snapshot(...,
previous=parent, append=True)` for a batch of new units. Existing unit
identities cannot appear in that batch. Full-prefix capture checks every
old record proof and reuses the parent's exact sufficient state.

`analysis.export(path)` writes a format-9 sequential checkpoint envelope,
containing a version-2 registration/snapshot/checkpoint and a version-3
compiled plan. `Analysis.from_moments` can replay this exact state.
Legacy format-8 sequential moment files cannot resume the process; fixed-horizon
format-8 (and the older format-7) moment behavior remains available.

Every rational in that envelope -- prior hyperparameters, `rho`,
`treatment_probability`, cell `alpha` and `null_lift`, `q`, adjustment
coefficients and centres, and each retained arm mean and scatter entry -- is
spelled as `Fraction` prints it: a signed integer or an integer quotient with
a positive denominator (a fixed decimal string such as `"0.05"` is also read).
JSON carries a rational as that string or a bounded integer literal, never a
boolean or floating literal: `0.05` and `1e-5` have already been rounded by the
JSON parser, so spell them `"0.05"` and `"0.00001"` instead. Each integer
component may carry at most 4,096 decimal digits, and scientific notation is
never a portable spelling. A payload outside that grammar refuses
with `sequential.wire.rational_invalid` (an `InvalidRequestError` whose context
names the `field`, `reason`, `limit_digits` and `route_forward`) before any
arithmetic, so a nine-byte `"1e5000000"` cannot make the reader allocate a
five-million-digit integer; text that is not JSON refuses with
`sequential.source.invalid`. The ceiling holds every supported capture: a
binary64 observation is an exact dyadic rational with at most 309 numerator
digits and a 1,075-bit denominator, and retained means and scatters stay near
1,300 digits even when the largest and smallest finite values alternate.
In-process `Fraction` and finite `Decimal` values keep their exact meaning.
A finite Python float normally binds its exact binary64 value; compliance
`alpha`/`null_lift` and `InferenceSpec.baseline_rate` instead bind its shortest
decimal spelling (`0.05` is `1/20`). Non-finite inputs refuse with the same code.
Exporting a state outside the ceiling refuses with `field="export"` instead
of writing a checkpoint no reader admits.

The retained family is fixed before data: metric × arm × segment cells
remain members even when no units enroll. Missing cells contribute log
evidence of minus infinity. Singular or zero-event prefixes retain their
observations and may become informative later.

`Analysis.capture_sequential(freeze=["metric_name"])` stops monitoring a
metric at that capture. Every registered cell for that metric—each
treatment arm and segment—keeps that capture's evidence at later looks.
A cell with no observations on either arm has nothing to freeze and stays
monitored; name the metric again at a later capture to freeze it.
Freezing is recorded in the snapshot's hashed prefix identity and carries
forward on later appends and parent proofs. A caller cannot retroactively
apply an earlier look's evidence. Raw-record producers call
`declare_sequential_freeze(snapshot, ["metric_name"])` on a captured
snapshot for the same effect. `Analysis.run()` reads frozen cells from the
snapshot; it has no run-time override.

Freezing fixes a cell's evidence, not its family verdict. e-BH uses current
or frozen evidence, never a running maximum, and recomputes it at every
look over the whole family. A frozen secondary's `discovery` and selected
interval can therefore change as other members' evidence moves. Reusing an
earlier verdict would not control FDR: e-BH requires every selected e-value
to be at least `m / (q R)` for the reported `R` (Wang and Ramdas 2022), and
a remembered selection can fail that threshold when `R` shrinks. Selected
intervals are reinverted at `min(q * R / m, nominal_alpha)` using the same
stopped likelihood state, which keeps the false coverage rate at `q` for
the selected set (Xu, Wang and Ramdas 2022).

Read `row.require_sequential_result()` for the authoritative confidence set,
certificate, and checkpoint. Confidence bounds use ratio coordinates; subtract
one to express lift. Endpoints can be unbounded, and sets can be full or empty.
A missing relative point estimate does not erase the confidence set. The
`sequential_lower` and `sequential_upper` frame columns project endpoints
outward to numeric floats or nulls. `sequential_log_e` remains available when
exponentiating evidence would overflow. `row.stat_sig()` uses the registered
null and exact allocated alpha. No equivalent-normal-score p-value is reported.
A two-sided confidence sequence compares each tail's e-process with the same
`1 / alpha` the decision uses; there is no central halving. At the true ratio
both tail e-processes are bounded by one test martingale, so Ville's
inequality already bounds the union of the two tails by `alpha`, and
`row.stat_sig()` is exactly "the registered null lies outside the sequence".
Sequential planning is declared with the same object the runtime reads:
`ArmPlanningProcedure.standard(inference=InferenceSpec(kind="asymptotic_mean"))`.
This convenience constructor sizes unsegmented monitoring. It rejects
`InferenceSpec.segments`: a whole-experiment sample size does not establish
power for individual segments.

Raw randomized ITT (Bernoulli law), asymptotic scalar-mean ITT
(continuous outcomes under a Randomized or Encouragement design), and
two-sided-design Bernoulli relative uptake are supported. Uptake also
requires `AnalysisPlan(compliance=SequentialCompliancePolicy(
alpha=Fraction(1, 20), alternative="two-sided", null_lift=0, family=False), ...)`.
Registered compliance cells must match this policy; its alpha is divided
across retained treatment arms. Request `estimands=("itt",)` or
`estimands=("compliance",)` explicitly. Encouragement as-of calls require
`completed_windows_only=True`, even before a checkpoint exists.

These explicit ITT/compliance requests may omit `ExclusionRestriction`; it is
required only when requesting LATE. Omitting `estimands` still requests LATE
and refuses without that declaration. Declaring it does not enable sequential
LATE.

Binary-uptake LATE, quantiles, clustered sequential inference, and
observational sequential causal identification require matching proofs and
refuse before source reads. Reusing a registered construction requires
choices predictable when each unit is revealed and observations that obey
the registered sampling law; predictability alone is insufficient. A fixed
winsorization threshold meets the predictability requirement, not
automatically the law requirement. A percentile threshold computed from
accumulated data refuses with `sequential.transform.unpredictable`.

A CUPED coefficient fixed from pre-period data
(`InferenceSpec.adjustments`, `ScalarMeanModel.adjustment`) applies at
capture on the asymptotic scalar-mean route, not the exact Bernoulli route.
Its common centering shift cancels at a zero-effect null, but changes
relative lift and nonzero-null margins when the experiment covariate mean
differs from the declared center. That bias does not vanish with sample
size. A coefficient fitted from in-experiment outcomes is admitted on
`AsymptoticMean` through the `adjusted_mean` and `adjusted_ratio_mean` laws
below; their confidence sequence accounts for the estimated coefficient.
The exact e-process route (`AlwaysValid`) refuses it because refitting
re-weights past increments and breaks its martingale (see
[`cuped.md`](cuped.md)).

Structural-zero control uptake is a rate target; use the fixed-horizon
compliance path. On `from_unit_summary` and `from_unit_panel`, declare a
randomized segmented family before observing outcomes with ordinary frame metadata:

```python
from increment import InferenceSpec

InferenceSpec(kind="always_valid", segments={"country": ("US", "CA", "GB")})
```

Use a categorical column and list every level to monitor, including levels
with no observations yet. Only the declared levels enter this family; other
values join no cell. Membership must be determined before assignment for
asymptotic monitoring, not chosen from outcomes. The package derives the
registration and retains empty cells; read it with `run_breakout()`, not
whole-window `run()` or `asof_lift()`. Exact monitoring uses the declared
breakout correction (e-BH by default); asymptotic monitoring uses fixed-roster
Bonferroni by default and still refuses BH. Both retain the existing explicit
uncorrected option. A manually built registration remains available but cannot
be combined with `InferenceSpec.segments`.

Segment labels are canonical strings on both frame routes: Boolean values are
`true`/`false`, missing values are `__null__`, string labels keep their exact
case and content (`"True"` is not `"true"`), and numeric labels follow the
existing string conversion. Declare Boolean levels as `"true"` and `"false"`
and a missing level as `"__null__"`. Unit-summary, unit-panel and
fixed-horizon breakouts read the same labels, so a Boolean column and its
string equivalent give identical cells, counts and readouts on pandas, Polars
and Arrow. A declared level with no observations stays a retained empty cell.

This corrects earlier releases, which recorded unit-summary Boolean and null
labels under backend-specific spellings such as `True` or `None` and left the
declared `true`/`false`/`__null__` cells empty. A checkpoint captured under
those spellings still replays unchanged through `from_moments`, but a source
that labels the same units canonically is refused with
`sequential.continuation.rewrite`: the recorded prefix is not silently
relabelled. A refusal is not permission to restart monitoring and spend alpha
again on the same experiment. Keep the earlier checkpoint as the experiment's
evidence, or design and register a new monitoring protocol whose validity you
can defend. Continuing with the earlier spellings as string labels remains
valid.

Predeclared segments cannot be combined with a compliance policy's
`family=True`. Use `family=False` to retain separately allocated uptake cells,
or omit segments for the existing joint whole-window family. This does not
remove the existing encouragement breakout-readout restrictions.

`from_definitions` and `from_unit_day_artifact` still refuse segmented capture
because relational segment capture needs an immutable property-relation
contract. `from_moments` can replay retained segmented state but carries no
breakout catalog, so `run_breakout` refuses with `readout.source.dimension`.
As-of output does not infer segmented historical checkpoints from a moment
cube. Automatic registration does not change these source boundaries.

Native capture and artifact publication pin assignment, outcomes, uptake,
property streams, freshness, and coverage in one backend execution. Subsequent
reductions read those pinned inputs. Cluster identity checks run on the raw
admitted assignments before reduction; cluster-grain inference remains fixed-horizon.

Current labeled checkpoints round-trip through moments and as-of readouts.
Frame day labels may be dates, datetimes, numeric days, ISO date strings, or
structured labels such as `d14`; the reveal label's representation is preserved.
Compliance-only monitoring uses the uptake declaration and finalized enrollment
cohort and does not need the outcome table. Compliance date series use enrollment
and uptake coverage; an outcome timestamp cannot extend that series.

Sequential segment heterogeneity and contrasts require a separate covariance
proof and are refused. Use the registered per-segment likelihood confidence sets
or the existing fixed-horizon heterogeneity and contrast routes.

## Error control by plan role

A sequentially monitored plan reports every declared role. Nothing must be
dropped to monitor a primary. Each role makes a different claim, so each
gets its own budget and procedure. The table below is the complete promise;
no implicit correction occurs downstream.

| Role | Level per metric | Guarantee | Procedure |
| --- | --- | --- | --- |
| primary | `alpha / n_primaries`, split again across the metric's own non-control arms | FWER `<= alpha` over every look for exact laws; asymptotic approximation otherwise, with the both-tails excess below for one-sided cells | Exact e-process or asymptotic confidence sequence |
| guardrail | `alpha`, never divided across guardrails | Per-guardrail type-I `<= alpha` over every look for exact laws; asymptotic approximation with the both-tails excess otherwise | One-sided non-inferiority using the registered law |
| secondary | nominal level, then e-BH at `q` for every law | FDR `<= q` at a stopping time for exact families; an asymptotic approximation otherwise, not a union-over-dates guarantee | e-BH selection, selected intervals reinverted at `min(q * R / m, nominal_alpha)` |
| unassigned | `alpha` | Per-metric only, in the registered law's validity regime; no multiplicity claim | Exact e-process or asymptotic confidence sequence |

Register each roster cell at or below its role's level.
`validate_sequential_plan` refuses a cell registered above that level and
names the allowance. The compiled procedure is the ceiling: a cell may be
registered tighter, never looser.

### Guardrails: one-sided non-inferiority, uncorrected across guardrails

A guardrail asks, "did this get materially worse?" It is a one-sided
non-inferiority test on the adverse side implied by the metric's
`preferred_direction` and margin, never the plan's `alternative`, which is
scoped to primaries and secondaries. A margin-less guardrail still tests
the declared adverse tail. Rejecting the null (harm at least as large as
the margin) certifies the metric SAFE.

`alpha` is **not** divided across guardrails. Each guardrail's type-I error
(falsely certifying SAFE) is bounded by `alpha` regardless of guardrail
count because the allocation is per test. The compound decision "ship only
if every declared guardrail is SAFE" is an intersection-union test (Berger
1982): the probability of shipping while any guardrail is truly harmful is
bounded by that same `alpha`. Dividing `alpha` by the guardrail count would
tighten the compound bound to `alpha / k`, but would make each individual
test harder to reject, reducing power without improving the IUT bound at
plain `alpha`.

Those finite-sample bounds require exact-law guardrails. Asymptotic guardrails
inherit the approximation and one-sided both-tails qualification below; the
intersection-union argument does not remove either qualification.

Increment reports each guardrail's own SAFE/HARM label; it does not compute
or display an aggregate ship/no-ship verdict. The `alpha` guarantee above
applies to the compound "ship only if every guardrail is SAFE" rule —
reading several guardrail labels without applying that rule (for example,
shipping when most, but not all, guardrails are SAFE) falls back to the
weaker union bound (`k * alpha` across `k` guardrails) on the joint
probability that at least one label is a false SAFE, not the tighter
single-`alpha` IUT bound.

### Secondaries: false discovery rate control at `q` for every in-family secondary

A secondary is tested at its nominal level, then its family controls FDR at
`q`: Benjamini-Hochberg over p-values for fixed-horizon inference, and e-BH
over exact e-values under `AlwaysValid` or plug-in mixture values under
`AsymptoticMean`. A selected interval is reinverted at
`min(q * R / m, nominal_alpha)`. Bernoulli families have a finite-sample
stopping-time guarantee; continuous or mixed families have an asymptotic
approximation. Estimated variance does not make a plug-in value an exact
e-value, so the oracle martingale theorem cannot provide a finite-sample
bound. There is no separate toggle or fixed-roster split.

Every family row reports `family_guarantee`: `finite_sample` only when all
members are exact, and `asymptotic_sequential` when any member is
asymptotic. A member without usable evidence—a missing arm, fewer than two
observations in an arm, or zero observed variance—contributes log evidence
of minus infinity, counts in the family size, and is never selected.
Guardrails share neither this budget nor its selection step; they test at
the full plan `alpha`. A prior-bound secondary stays outside the family
under every inference kind, as it does at fixed horizon.

This is the metric/arm secondary family. A registered breakout family is a
separate axis whose correction is declared per registration: a continuous
(scalar-mean) breakout uses fixed per-cell Bonferroni with no reselection,
while a discrete (Bernoulli) breakout uses this e-BH-at-`q` family.

Asymptotic metric/arm families were Bonferroni-corrected (familywise error)
before they moved to e-BH. A registration records which rule its asymptotic
family commits to, so a snapshot, moments export or compiled plan stored under
the Bonferroni rule refuses with `sequential.continuation.legacy` instead of
continuing under the weaker false-discovery guarantee; start a new
registration by capturing without `previous=`.

### Choosing the Beta prior weight for a declared baseline rate

An automatic exact Bernoulli registration built from
`InferenceSpec(kind="always_valid", baseline_rate=p0)` commits `Beta(w * p0, w * (1 - p0))`
on both arms. The e-process is valid for every proper prior, so `w` only moves power;
`calibration/bernoulli_prior.py` measured it on the grid below (200 replications
per cell, one look every 250 units per arm to 4000, `alpha = 0.05`; "type I" is the fraction
of null replications that ever crossed, "power" the fraction that crossed by the stated look).

| p0 | prior | type I @4000 | power @2000, lift .05 | lift .10 | lift .20 | power @4000, lift .10 | true p0 = 2x: type I @4000 | power @2000, lift .10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | Beta(1,1) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - |
| 0.02 | w=2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.02 | w=10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.02 | w=50 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 |
| 0.02 | w=200 | 0.005 | 0.005 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 |
| 0.1 | Beta(1,1) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - |
| 0.1 | w=2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.1 | w=10 | 0.000 | 0.000 | 0.005 | 0.015 | 0.005 | 0.000 | 0.000 |
| 0.1 | w=50 | 0.005 | 0.010 | 0.010 | 0.080 | 0.010 | 0.000 | 0.005 |
| 0.1 | w=200 | 0.010 | 0.005 | 0.015 | 0.105 | 0.030 | 0.000 | 0.000 |
| 0.3 | Beta(1,1) | 0.000 | 0.000 | 0.015 | 0.335 | 0.065 | - | - |
| 0.3 | w=2 | 0.000 | 0.000 | 0.010 | 0.405 | 0.035 | 0.000 | 0.280 |
| 0.3 | w=10 | 0.000 | 0.010 | 0.020 | 0.520 | 0.140 | 0.000 | 0.165 |
| 0.3 | w=50 | 0.000 | 0.010 | 0.015 | 0.685 | 0.185 | 0.000 | 0.000 |
| 0.3 | w=200 | 0.005 | 0.025 | 0.100 | 0.685 | 0.250 | 0.000 | 0.000 |

Every weight is valid: the largest null crossing rate anywhere on the grid, including the
misdeclared cells, is `0.015` against `alpha = 0.05`, so the weight is a power choice only.
The probe's own rule reads power alone (the smallest weight within five points of the best
at lift 0.10 after 2000 units per arm at every declared rate) and selects `w=200`, which at
`p0 = 0.3` and 4000 units per arm reaches `0.250` at lift 0.10 against `0.140` for `w=10`
and `0.065` for the flat prior. The default is not that weight. It is
`DEFAULT_BERNOULLI_PRIOR_WEIGHT = 10`, a robustness choice read off the last two columns:
with the true control rate at twice the declared one (`0.3` declared, `0.6` true), power at
lift 0.10 after 2000 units per arm is `0.280` for `w=2`, `0.165` for `w=10` and `0.000` for
both `w=50` and `w=200`. So `w=10` is clearly better than the flat prior when the declared
rate is right (`0.520` against `0.335` at lift 0.20 after 2000 units per arm; `0.140`
against `0.065` at lift 0.10 after 4000), and it is the largest weight on the grid that
still has power when the rate is wrong by a factor of two. Weights of 50 and above buy
more power only when the caller is certain of the rate and give none when they are not. A
caller who is confident in the declared rate may take that power by declaring the heavier
`Beta` prior on an explicit `SequentialRegistration` (`SequentialModel.control_prior` and
`treatment_prior`), which the automatic route then leaves untouched.

The grid also shows what the exact route cannot do at these sample sizes: at a declared rate
of `0.1` the best cell is `0.105` (w=200, lift 0.20 after 2000 units per arm), at `0.02` it is
`0.010`, and most cells at both rates are `0.000`. These are power measurements,
not calibration evidence for an alternative method. The asymptotic route
(`kind="asymptotic_mean"`) is available, but rare-conversion calibration at
base rates `0.10` or below is unverified by this grid and the bounded release audit.

## Sequential monitoring of a conversion metric

A conversion or retention metric needs no hand-written registration. Declare the
inference kind and, if you know it, the control arm's usual conversion rate:

<!-- skip: next "requires a finalized unit summary frame" -->

```python
from increment import Analysis, AnalysisPlan, InferenceSpec, MetricSpec, Randomized

plan = AnalysisPlan(
    primary="purchase",
    inference=InferenceSpec(kind="always_valid", baseline_rate=0.03),
)
analysis = Analysis.from_unit_summary(
    finalized_frame,
    unit="user_id",
    group="variant",
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    metrics=[MetricSpec(name="purchase", type="conversion")],
    experiment_id="checkout-experiment",
    plan=plan,
    exposure_date="enrolled_on",
)
row = analysis.run()[0]
row.stat_sig()
```

`baseline_rate` is the conversion rate you expect in the control arm: read it off the
same metric over the weeks before the experiment (the number the power analysis used).
It only sharpens monitoring; a wrong value costs power, never validity (the table above
shows the cost of declaring half the true rate). Leave it out if you do not know it.
For an uptake-only encouragement plan with no outcome metrics, the rate describes
control uptake instead. With outcome metrics present, it tunes only their priors;
the additional compliance model retains its flat uptake prior.

Before any outcome is read the source binds a `SequentialRegistration` with one
`bernoulli` model per metric and the same `Beta(w * p0, w * (1 - p0))` prior
in each arm (`w = DEFAULT_BERNOULLI_PRIOR_WEIGHT`, chosen above; `Beta(1, 1)`
without a `baseline_rate`). Exact monitoring accepts the declared treatment
arms and creates a cell for each metric/arm pair, or metric/arm/level for a
segmented frame. A primary's error allocation is split across treatment arms;
whole-window secondaries share `q` over their complete metric-by-arm family,
capped at each procedure's allocation. Segmented e-BH shares the breakout
view's `q` over its complete cell roster, including absent levels.

The registration identity binds the metrics, design, source mapping,
`baseline_rate` and declared roster. An unchanged declaration continues an
existing process; a changed roster or prefix refuses continuation. Existing
two-arm declarations retain their identities. Multi-arm asymptotic monitoring
and multi-arm sequential-secondary power planning remain unsupported.
A YAML plan writes
`inference: {kind: always_valid, baseline_rate: 0.03}`; the float binds the decimal it
was typed as. A mean or ratio metric under a registration-less `always_valid` plan
refuses with `definition.inference.always_valid_metric_type`, naming the metric and
the `asymptotic_mean` route; `baseline_rate` with any other kind or with an explicit
registration refuses with `definition.inference.baseline_rate_route`.

Exact or asymptotic: the exact route is anytime-valid for every sample size and every
proper prior, and its power depends on the prior only through the weight measured in
the table above; the asymptotic route (`kind="asymptotic_mean"`) needs no prior, tunes
from `expected_decision_sample_size`, and is an asymptotic approximation. Prefer the
exact route when its finite-sample guarantee is needed. The asymptotic route
also admits binary and mean metrics in one registration and CUPED adjustment.
Low power of the exact route at rare conversion rates does not establish the
asymptotic route's calibration there.

## Ordinary continuous monitoring

Mean, ratio, conversion and retention metrics can use the ordinary plan/source
flow without constructing a registration or supplying proof-like population
flags. When an asymptotic mean inference is selected, Increment derives the
immutable source and definition identity, fixed treatment allocation, metric
roster, and per-cell nominal alpha allocations for the e-BH secondary family
from the plan and source metadata before reading outcomes. Tuning is frozen
from a planned expected decision sample size (`5000` by default) and each
compiled procedure's allocated alpha: with `t=N*rho^2`, `rho` solves `t-log1p(t)=-2*log(alpha)`.
The operational per-arm withholding threshold starts at the mathematical
minimum of two observations; neither choice is a finite-sample guarantee or a
heavy-tail safeguard.

The method assumes iid stationary potential-outcome vectors, independent fixed
Bernoulli assignment, finalized scalar outcomes, finite moments of order
`2+delta`, positive limiting arm variances, a positive population control mean,
consistency/no interference, and outcome-independent joint reveal. These are
documented scope assumptions, not facts inferred or certified from observed
data. Registration is frozen before capture; changed mappings, assignments,
rosters, or continuation prefixes refuse before outcomes are read.

## Continuous scalar means

Use `InferenceSpec(kind="asymptotic_mean")` for mean metrics and for conversion
or retention metrics whose 0/1 outcome is monitored as a rate; the retained
centered scatter of a 0/1 outcome is its Bernoulli variance, so no separate
binary law is needed on this route.
The source constructs the registered `AsymptoticMean` runtime before capture.
This is a distinct asymptotic sequential construction. It does not reinterpret
NIG/NIW priors, `AlwaysValid`, or a Gaussian planning effect scale.

The declared model is iid stationary unit potential-outcome vectors, independent
Bernoulli assignment at a fixed probability in (0, 1), finalized scalar outcomes,
finite absolute moments of order 2+delta for some delta>0, positive limiting arm
variances, and a fixed positive population control mean. Treatment means may be
negative. Outcome-independent joint reveal, consistency and no interference are
assumptions; timestamps and checksums cannot establish them. A single randomized
treatment/control comparison is admitted; adaptive assignment, arbitrary
allocation schedules, calendar drift and outcome-dependent reveal are not.
Registered segments must be fixed by pre-assignment attributes; post-treatment
segmentation does not preserve this assignment model.

Declare the plan and assignment before reading outcomes:

<!-- skip: next "requires a finalized unit summary frame" -->

```python
from increment import Analysis, AnalysisPlan, InferenceSpec, MetricSpec, Randomized

specs = [MetricSpec(name="revenue", type="mean")]
design = Randomized(
    control_group="control",
    allocation={"control": 0.5, "treatment": 0.5},
)
plan = AnalysisPlan(
    primary="revenue",
    inference=InferenceSpec(
        kind="asymptotic_mean",
        expected_decision_sample_size=5000,  # Optional pre-outcome planning target.
    ),
)
analysis = Analysis.from_unit_summary(
    finalized_frame,
    unit="unit",
    group="arm",
    design=design,
    metrics=specs,
    experiment_id="revenue-experiment",
    plan=plan,
    exposure_date="enrolled_on",
)
row = analysis.run()[0]
result = row.require_asymptotic_sequential_result()
```

Definitions-based experiments supply the same fixed weights through
`Experiment.allocation`, and the same inference policy in their plan. Published
artifacts store the bound registration; reopening or extending one never retunes it
from outcomes. `ScalarMeanModel` and explicit registrations remain available for
inspection and replay, not as prerequisites for this workflow.

The library enforces commitment before its own observation capture. It cannot
establish that nobody examined the outcomes elsewhere beforehand.

The actual clock is N=nc+nt finalized independent units. With centered sums of
squares M2a, the empirical variance convention is sa²=M2a/na and Va=sa²/na.
For a relative null lift ell0, r=1+ell0, the shifted contrast and boundary are

```text
Dhat_r = meanT - r * meanC
Vr = VT + r² * VC
lambda = rho^-2
K_N = (1 + lambda/N) * (log1p(N/lambda) - 2*log(alpha))
boundary² = Vr * K_N
```

The ratio confidence set inverts `(meanT-r*meanC)² <= K_N*(VT+r²*VC)` over
all real r. `result.bounds.components` retains bounded intervals, rays,
disconnected sets, full sets and empty sets in ratio coordinates; subtract one
from finite endpoints for lift. There is no -1 lift floor. Rational centered
moments, outward logarithms and rational root enclosures avoid raw-sum
cancellation, overflow and arbitrary near-singularity tolerances. A displayed
numeric point can be unavailable while the confidence set remains available.
Before both arm starts, with fewer than two observations in either arm, or with
zero observed variance, the result is a full set with a precise availability
reason. This is not evidence of zero population variance.

The start is a per-arm operational withholding threshold, **not finite-sample
calibration**. Fixed rho and start are frozen in model, registration and checkpoint
identity. The validity regime is asymptotic CS approximation for each fixed
supported distribution, following the time-uniform linearization argument for
the arm-mean contrast and the AsympCS framework of
[Waudby-Smith et al.](https://arxiv.org/html/2103.06476v7).
No delayed-start coverage-limit claim or universal finite start guarantee is
made. Smaller `expected_decision_sample_size` values also increase early-look
approximation risk; this is not only a power choice. The selected calibration
checks start with at least 40 units per arm, not the default operational
threshold of two, and do not verify every smaller planning target.

For `greater`, discovery requires Dhat_r>sqrt(Vr*K_N); for `less`, it
requires Dhat_r<-sqrt(Vr*K_N). A one-sided cell outside a family builds `K_N`
at `2*alpha` and spends one tail of that symmetric two-sided boundary; by
symmetry that tail crosses with probability at most `alpha` plus half the
chance of crossing both tails at different looks (simulated at `alpha = 0.05`
over 20,000 Gaussian null paths to 20,000 units: upper tail 0.033, both tails
never). Its reported set is a ray whose finite endpoint equals the two-sided
endpoint at `2*alpha`. A family member builds `K_N` at `alpha` itself: its set,
at the registered level or reinverted after selection, holds exactly the
ratios whose sign-gated plug-in mixture value stays below `1/alpha`, the evidence e-BH
selects on, so `row.stat_sig()` and the family verdict read the same evidence.
Planning (`ArmPlanningProcedure.standard(inference=InferenceSpec(kind="asymptotic_mean"))`)
builds the same boundary: a one-sided sequential secondary, a family member, is
sized against `K_N` at its own `alpha`, and a one-sided primary or guardrail
against `K_N` at `2*alpha`.
A one-sided cell outside a family registered at `alpha >= 1/2` refuses with
`sequential.asymptotic_mean.invalid`. Empty geometry withholds decisions.

`AsymptoticSequentialEvidence` carries the checkpoint, actual arm counts, model,
null, direction, allocation, start, tuning, geometry and decision. Its
`result.log_e` is sign-gated asymptotic log evidence, capped for ratio laws by
denominator stability as described below; family selection also masks ratios
unresolved at its nominal reporting ceiling. It has no finite-sample p-value.
Frame/readout exports include
`sequential_validity_regime`, `sequential_alpha`, `sequential_components`, and
numeric-null `sequential_log_e`. JSON replay recomputes geometry and rejects
changed evidence. Identical capture is idempotent; frozen results require a
verified ancestor prefix. The same model follows the existing frame, panel,
native, artifact, moments-export, current as-of and registered breakout paths.
Native/artifact segment capture retains its immutable-property-relation refusal.

Continuous (metric/arm) secondary families are selected by **e-BH** at the
plan's `q`, labeled `asymptotic_sequential` whenever any member is asymptotic
(`finite_sample` only if every member is exact -- see Secondaries above).
Register every member with `family=True` and its nominal `alpha`; a selected
member's interval is reinverted at `min(q * R / m, nominal_alpha)`, and every
row in the family reports the same `family_guarantee`/`family_nominal_alpha`.
A member without usable evidence contributes log evidence of minus infinity
and is never selected; a recorded failure in place of evidence refuses the
whole family.

Registered breakout (segment/view) families split by construction. A
continuous breakout -- every family cell under a count-clock asymptotic law
(`scalar_mean` or a linearised joint law below), each judged against the
same boundary at its own registered allocation -- registers
`correction="bonferroni"` and uses **predeclared Bonferroni**, labeled FWER with
`validity_regime="asymptotic_sequential"`: register every member with
`family=True` and its fixed `alpha`; the sum must not exceed registration
`q`. Missing members retain their budgets and make no discovery. View
families require `MultiplicitySpec(correction="bonferroni")` and a
registered segment roster. There is no e-BH, p-value conversion or `R*q/m`
selected-interval reinversion on this axis.
The asymptotic label does not promise finite-start FWER. A discrete
(Bernoulli) breakout instead registers `correction="bh"` and is dispatched
to the SAME e-BH-at-`q` family selection as an ordinary metric/arm
secondary above -- `m` is the full retained roster, with missing cells at log
evidence minus infinity, and e-value evidence and
`min(q * R / m, nominal_alpha)` reinversion apply exactly as they do there.
One asymptotic registration may also carry an exact Bernoulli uptake cell:
under an encouragement design, whole-window automatic registration composes scalar-mean
ITT cells with a Bernoulli compliance cell, and a compliance policy declared
with `family=True` joins the same e-BH family, labeled `asymptotic_sequential`.
Other exact and continuous models require separate registrations.

### Adjusted means and ratio metrics: the joint laws

The same route generalises to three further laws that share the scalar law's
contract, boundary and geometry and differ only in the per-unit vector each
arm retains with its full centered scatter:

| law | retained per unit | contrast | selected automatically when |
| --- | --- | --- | --- |
| `scalar_mean` | $(Y)$ | ratio of arm means | a mean, conversion or retention metric |
| `adjusted_mean` | $(Y, X)$ | ratio of CUPED-adjusted arm means | a mean metric with `covariate=` and a CUPED method |
| `ratio_mean` | $(N, D)$ | ratio of arm ratios $E[N]/E[D]$ | a ratio metric |
| `adjusted_ratio_mean` | $(N, D, X)$ | ratio of adjusted arm ratios | a ratio metric with `covariate=` and a CUPED method |

Each is a delta-method linearisation at the observed means: the contrast
`f_t - r * f_c` has, with respect to the two retained mean vectors, gradients
affine in `r`, so `V(r)` is the quadratic form of those gradients over each
arm's scatter divided by `n_a²`, and `(f_t - r f_c)² <= K_N V(r)` is inverted
with the scalar law's solver and boundary. The adjusted laws fit $\theta$ from
the retained within-arm cross moments at every look (the standard CUPED
coefficient, the same inverse-n weighted slope the fixed-horizon fit uses) and
anchor both arms on the pooled covariate mean, whose coupling the gradients
carry; the ratio laws carry the $-2\,\mathrm{Cov}(N, D)$ term through the
cross scatter, and the adjusted ratio law the covariance the shared covariate
induces between the adjusted components. What is exact: the retained state,
the coefficient, the anchor, every gradient and quadratic coefficient, and
$K_N$. What is asymptotic: the boundary itself and the linearisation with its
nuisance plug-ins, whose error is an order below the boundary width under the
contract (Lindon, Ham, Tingley and Bojinov 2022,
[arXiv:2210.08589](https://arxiv.org/abs/2210.08589); Schmit and Miller
2022, [pdf](https://svenschmit.com/assets/pdf/code_2022_ci.pdf)). The
`adjusted_*` laws are admitted on `AsymptoticMean` only; the exact Bernoulli
route admits neither fitted nor predeclared CUPED. The adjusted laws need a
per-unit covariate: `from_unit_summary` supplies it through `MetricSpec(covariate=...)`,
while `from_definitions` derives it from `n_pre_periods > 0`.
`from_unit_day_artifact` additionally requires each adjusted metric's published
`cuped_preperiod` extension; default publication does not include it.
See [CUPED](cuped.md#the-standard-coefficient-on-the-asymptotic-route) for
extension selection. Unit panels refuse these laws.
The ratio laws declare
`positive_population_denominators=True`; a zero or insufficiently separated
observed denominator leaves that cell unavailable at the requested level
(`zero_denominator_mean` or `denominator_near_zero`), not its whole family.
For registered scalar and ratio observations, `MetricSpec(missing="zero")`
applies to outcome coordinates before they enter the retained sequential
state: null and NaN numerator values become exact zeroes (and the same
policy applies to a ratio denominator when declared). This outcome policy is
independent of covariate missingness: `covariate_missing` controls how a
declared pre-period covariate is imputed or refused and never changes the
metric outcome. The sequential route does not silently clip infinities.

Ratio-family log evidence depends on the stopped state, not its current
inversion alpha, and is capped by denominator stability. This capped evidence
is dominated by the underlying mixture value, not itself a martingale.
The family keeps every roster slot and masks a ratio unresolved at the
nominal reporting ceiling; selection then requires resolved denominators
at the selected reporting level too.

A ratio unavailable at a tighter registered allocation can therefore become
reportable at a later, wider selected level, even when its checkpoint is
frozen. Reinversion uses that same stopped state and refreshes its availability
note. Alpha-independent failures -- before-start or missing-arm states,
nonpositive denominators, and degenerate variances -- still abstain.
New laws retain new state shapes, so no snapshot version bump or migration of
existing `scalar_mean` checkpoints was needed.

Percentile winsorization, clustering, repeated units, switchback,
observational inference and additive sequential decisions remain unsupported.
Power solvers explicitly refuse `AsymptoticMean` with
`sequential.route.unsupported`; their log-SE Gaussian planning approximations
are not power for this count boundary. Tests of deterministic stopping and
replay verify implementation contracts, not empirical calibration or a theorem.

## Explicit research campaign

The retained stopping, family, and numerical stress cases are an explicit
research campaign rather than an implicit unbounded pytest tier. Run the
sequential-specific CLI with a new output directory:

```bash
uv run python -m calibration.sequential \
  --output /tmp/sequential-campaign \
  --profile smoke \
  --budget-seconds 3600 \
  --case-seconds 60
```

The CLI uses the shared hard-budget runner, allocates bounded per-case
deadlines, and writes an immutable manifest plus one checkpoint per declared
case. Each checkpoint records completed, failed, timeout, or unresolved status;
failures and unavailable numerical cases are retained rather than relabeled as
success. This campaign does not add support for models or transformations
outside the registered raw-likelihood contract.

Replication journals retain the simulator's selection, availability,
certified-interval, and point-reason counters. They do not measure interim
finite-look confidence coverage.

For a one-draw diagnostic, add `--case 0 --repetitions 1`. A completed draw is
not campaign certification: the CLI exits 2 while the campaign remains
incomplete, even when its controller and workers exit 0.

## Primary references

The public Bernoulli e-process is qualified by a predeclared common
joint-unit filtration: every relevant outcome for a unit is revealed together,
in committed order, and the process may stop at any observed prefix. This is
the setting of [Ville's maximal inequality](https://doi.org/10.1007/BF01503646)
and the time-uniform concentration framework of
[Howard et al.](https://arxiv.org/abs/1810.08240). Gaussian kernels in this
package remain private numerical diagnostics and are not covered by the public
anytime-valid contract.
