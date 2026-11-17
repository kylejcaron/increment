# Family inference contracts

Use this reference to interpret a family's members, error-control level,
selection-adjusted intervals or saved decisions. For choosing primary,
guardrail and secondary roles, start with [Multiplicity](../guides/multiplicity.md).

A cell identifies a metric, treatment arm and, where applicable, segment.
A correction covers its declared cells, not the rows currently displayed.
Alpha division can nest across metrics and arms; different selection
procedures are not applied successively to overlapping families. Only the
chosen decision method supplies a hypothesis's selection evidence;
sensitivity estimates are not additional hypotheses.

## Error control by plan role

### Primary and unassigned metrics

Primaries receive `alpha / n_primaries`, divided again across each metric's
non-control arms. This controls familywise error (FWER): the chance of at least
one false primary rejection. Under exact sequential laws the bound is
`<= alpha` over every look. Asymptotic laws inherit an approximation, including
the [one-sided both-tails qualification](sequential-contracts.md#one-sided-boundaries).
An unassigned sequential metric uses `alpha` for its own test, with no
multiplicity claim. Primaries have `discovery=None`; selection applies to
secondaries and breakouts.

`validate_sequential_plan` refuses a cell above its role's allowance and names
that allowance. The compiled procedure is the ceiling: a manually registered
cell can be tighter, never looser. Supported fixed-horizon prior-bound rows
retain secondary membership based on sampling evidence; a prior does not
alter the sampling correction. Unsupported prior/inference combinations
still refuse.

### Guardrails: one-sided non-inferiority, uncorrected across guardrails

A guardrail asks whether an unacceptable loss can be ruled out. Its adverse
side follows the metric's `preferred_direction` and margin, not the plan's
`alternative` (which applies to primaries/secondaries). Declare `margin` or
`margin_abs` on the metric or its `ExperimentMetric` entry; omitting both
raises `plan.guardrail.margin_required` before data is read.

Rejecting the null of harm at least as large as the margin certifies the
guardrail. Alpha is **not divided across guardrails**. For exact-law tests,
each false certification has probability at most `alpha`. The compound rule
“ship only if every guardrail is certified” is an intersection-union test
(Berger 1982): shipping while any guardrail is truly harmful also has
probability at most `alpha`. Dividing by the number `k` would tighten that
bound to `alpha/k` but lose power unnecessarily for a target of `alpha`.
Asymptotic guardrails retain their approximation and one-sided qualification;
the intersection-union argument does not remove either.

`result.decisions` passes only when every required primary rejects on its
favorable side, every guardrail is available and certified, and assignment
integrity is qualified. Adverse primary rejection blocks with
`decision.primary.adverse`; an uncertified guardrail blocks with
`decision.guardrail.not_certified`. An uncertified guardrail is not a finding
of harm. See [program verdicts](result-fields.md#program-verdict).

If you use row verdicts directly, apply the all-guardrails rule yourself.
Shipping when only most pass does not get its single-alpha bound; the joint
probability that any of `k` guardrails is falsely certified has only the
weaker union bound `k*alpha`.

### Secondaries

Fixed-horizon secondaries use Benjamini–Hochberg (BH) on p-values at `q`.
Its false discovery rate claim bounds the expected fraction of selected cells
that are false positives, not FWER or a per-cell type-I error. It requires
independence or positive dependence (PRDS) across the metric × arm × segment
family; this is not checked.

Sequential metric/arm secondaries use [e-BH](#e-bh-under-sequential-inference).
Guardrails share neither its budget nor its selection step. No separate
secondary toggle or fixed-roster alpha split is needed.

## Family metadata and metric selection

Lift, breakout and daily/as-of rows retain `role`, interval alpha,
`multiplicity_status` and, for source-scoped families, `family_id`.
The status describes declaration/selection; it does not apply a new correction.

- **`undeclared_plan`:** no role declared (`role=None`); unadjusted.
- **`unassigned_in_plan`:** plan exists, metric has `role="unassigned"`; unadjusted.
- **`declared_plan`:** primary, secondary or guardrail with its role-specific procedure.
- **`exploratory_unadjusted`:** exploratory, outside an adjusted family.
- **`exploratory_family`:** exploratory member of a BH/e-BH or Bonferroni family.

A sequential uptake cell can have `role=None` and `declared_plan` when its
compliance policy has `family=True`: it belongs to and counts in the secondary
e-BH family. Five undeclared metrics each retain alpha `.05`; five primaries
with one treatment arm share it at `.01` each. `declared_plan` does not imply
all roles have joint error control—guardrails retain unsplit alpha.

`readout_table` puts a common informative status in its header. With mixed
statuses it marks known unadjusted/exploratory rows with superscripts and
explains their counts in the header; declared-plan rows need no marker.
Objects and `to_frame()` retain the original status and family ID.

`results.metadata.scope.families` retains the source-scoped member lists.
Filtering and slicing preserve those lists and partial-view indications;
saved readout envelopes preserve them too. Choosing a subset for display
cannot recover alpha or turn a partial decision into a complete one. Registered
run options cannot omit or add decision metrics. Adding exploratory output
cannot enlarge or reallocate the declared primary family.

Assigned and triggered populations have separate family identities and
allocations. Concatenating readouts preserves each source's families rather
than creating one jointly controlled experiment-wide family. Identify a claim
by `family_id` **and source/population scope**, not merely a family name.
An affirmative exploratory selection stays exploratory: a fixed BH family
retains `q`, full family size, FCR-adjusted intervals and the dependence
qualification, not a confirmatory label.

## Missing and incomplete families

Unavailable is a result state; a non-rejection is how some such cells count
in selection. Neither means “no effect.” Do not erase a cell to make a family
look complete.

### Fixed-horizon whole-experiment families

Every secondary metric contributes a cell for each treatment arm with an
enrolled unit. Missing arm/control **metric values** keep that cell as a
non-rejection, counted in `family_size` (conservative p-value 1). Primary
allocation likewise includes enrolled treatment arms even before that metric's
values arrive. An allocated arm with **no enrolled unit** is excluded from
these families, but its unavailable rows remain with `readout.cell.missing_arm`.
Without enrollment counts, the split/family uses arms reached by each metric's
values.

Cells excluded from selection evidence for outcome reasons—zero variance,
nonpositive means, extreme ratios or no usable observations—stay counted and
never select, even at `q=1`. A missing relative point alone does not imply
exclusion: another supported sampling representation may still supply valid
selection evidence. Other missing, failed or wrong-type evidence raises
`family.evidence.incomplete` with immutable `failed`, `missing` and `invalid`
cell identities. A missing estimate is not interchangeable with missing raw
statistics or a malformed request.

This clustered example keeps a secondary with zero relative variance alongside
an informative one. The zero metric cannot supply the relative test, but still
counts in the family:

```python
import increment
import polars as pl

frame = pl.DataFrame(
    [
        {
            "unit": f"{arm}:{cluster}:{member}",
            "arm": arm,
            "cluster": f"{arm}:{cluster}",
            "zero": 1.0 + cluster % 3 if arm == "control" else 0.0,
            "strong": 10.0 + cluster / 10 + (2.0 if arm == "treatment" else 0.0),
        }
        for arm in ("control", "treatment")
        for cluster in range(20)
        for member in range(3)
    ]
)
experiment, reader = increment.from_frame(
    frame,
    shape="summary",
    unit="unit",
    group="arm",
    control="control",
    cluster="cluster",
    metrics=[increment.MetricSpec(name=name, type="mean") for name in ("zero", "strong")],
    plan=increment.AnalysisPlan(secondaries=["zero", "strong"], q=0.05),
)
results = increment.run(experiment, reader).readout()
zero = next(row for row in results if row.metric == "zero")
print(zero.relative_unavailable_reason, zero.discovery, zero.family_size)
filtered = results.filter(lambda row: row.metric == "strong")
assert filtered.metadata.scope.families == results.metadata.scope.families
assert filtered[0].family_size == 2
```

<!-- output -->
```text
zero_relative_variance False 2
```

### When a family cannot be evaluated

The finite-sample binomial method has a compute ceiling of 1,000,000,000 units
per arm. These compact summaries deliberately exceed it for one treatment
arm; they do not allocate a billion rows. Requesting that method for the
secondary family must identify the failed arm, not silently select the other
arm as though it were the whole family. Capture the expected cell warning
as well as the family error:

```python
import warnings

from increment.errors import CapabilityError, IncrementWarning
from increment.estimators import moments_from_summaries

summaries = [
    {"metric": "converted", "group_id": "control", "n": 1000, "successes": 100},
    {"metric": "converted", "group_id": "small", "n": 1000, "successes": 130},
    {"metric": "converted", "group_id": "large", "n": 1_000_000_001, "successes": 110_000_000},
]
moments = moments_from_summaries(summaries, metrics={"converted": "conversion"}, control="control")
experiment, reader = increment.from_moments(
    moments,
    metrics={"converted": "conversion"},
    control="control",
    plan=increment.AnalysisPlan(
        secondaries=[
            increment.ExperimentMetric(
                metric="converted",
                decision_method=increment.MethodSpec(
                    name="unadjusted", conversion_inference="finite_sample"
                ),
            )
        ],
        q=0.10,
    ),
)
with warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter("always", IncrementWarning)
    try:
        increment.run(experiment, reader)
    except CapabilityError as error:
        assert error.code == "family.evidence.incomplete"
        assert any(key.group_id == "large" for key in error.context["failed"])
        print(error.code)
    else:
        raise AssertionError("This family needs unavailable finite-sample evidence")
assert any(notice.message.code == "readouts.run.cell_refused" for notice in notices)
```

<!-- output -->
```text
family.evidence.incomplete
```

For a failing **breakout** family, read the view with
`MultiplicitySpec(correction="none")` to inspect each cell's reason. That
uncorrected diagnostic view is not a repaired family claim.

### Registered sequential families

Every declared metric × arm × segment remains a member, even when no units
enroll. A typed result with no usable evidence—a missing arm, too few units,
zero observed variance or unresolved denominator—contributes log evidence
minus infinity, stays counted and is never selected. Singular/zero-event
prefixes retain their observations and may later become informative.
A recorded failure or missing required evidence instead refuses the **whole
family** with `family.evidence.incomplete`; it is not permission to shrink it.

## Segmented views

`AnalysisPlan.view_multiplicity` is independent of whole-experiment
`secondaries`/`q`. `MultiplicitySpec` accepts `"bh"`, `"bonferroni"` or `"none"`.
Only `"bh"` accepts `q`, defaulting to `.10`; supplying it with another
correction refuses.

Without an override, randomized breakouts use BH, encouragement breakouts
are uncorrected, and as-of views are uncorrected unless segmented. Registered
asymptotic breakouts instead default to fixed-roster Bonferroni.

### Fixed-horizon eligible cells

Randomized breakouts use each metric's declared hypothesis, including its
one-sided alternative and relative/absolute margin. Absolute margins are
tested on the additive reference in metric units, not by checking a relative
interval against zero. A guardrail hypothesis does not turn its segments
into registered ship decisions: every segment is exploratory.

The nominal level is `AnalysisPlan.alpha`, without the whole-window primary/arm
split. No correction uses that level per cell. Bonferroni divides it once by
each metric's eligible arm/segment count `k` (five eligible cells use
`alpha/5`, even with several primary metrics). BH selects one flat
metric × arm × segment family and reports selected intervals at
`min(q*R/m, alpha)`.

- A segment missing an arm, or with fewer than two units under a method that
  estimates variance, retains an unavailable row (`few_units`, `no_control_arm`
  or `no_treatment_arm`) but is excluded from `family_size`. Exact-binomial
  conversion/retention needs no estimated variance and stays at one per arm.
- Under randomization, arm counts carry no outcome information, so conditioning
  BH or Bonferroni membership on them preserves its applicable error statement.
  Bonferroni counts every eligible treatment-arm/segment pair for that metric.
- Fewer than 50 per arm adds `low_reliability`, not exclusion or withholding;
  see [small-segment flags](result-fields.md#small-segments-are-flagged-not-withheld).
- Outcome-based exclusions, including a control lost to nonpositive means,
  stay counted as non-rejections. With `MetricSpec(missing="skip")`, one observed
  unit is a count exclusion (`few_units`), while no usable observations count
  as a non-rejection. This relies on the skipped metric's assumption that
  missingness is unrelated to outcomes within each arm.
- Any other unestimable member raises `family.evidence.incomplete`.

Scalar CUPED, supported ratios and fixed winsor caps keep their estimators.
Clustered, quantile, estimated-percentile-cap and observational restrictions
remain; shifted multi-covariate linear-CUPED and encouragement segments are not
implemented. Registered sequential breakouts use their checkpoint route.

Compared with earlier compiled segment behavior, forwarding a declared
alternative corrects formerly two-sided one-sided zero-null results.
Two-sided zero-null allocations and standalone defaults are unchanged.
Adding a margin or changing a member's tail can change BH selection and
selected intervals for its siblings too; not every zero-margin output remains
unchanged.

### Registered sequential segments

A dataframe's `InferenceSpec.segments` declares all levels before outcomes;
absent levels remain counted. Exact Bernoulli breakouts use e-BH with the
breakout view's `q`, including a `MultiplicitySpec` override—not a family
inferred from observed rows. Exact multi-arm whole-window monitoring counts
every secondary metric/arm and splits primary alpha across treatment arms.

Continuous breakouts instead use predeclared Bonferroni (`correction="bonferroni"`),
with `family=True` and fixed alpha for each cell; allocations sum to at most
registration `q`. Missing members keep their budgets and make no discovery.
The label is FWER with `validity_regime="asymptotic_sequential"`, **not**
finite-start FWER. There is no e-BH, p-value conversion, reselection or selected
interval reinversion on this axis. Explicitly uncorrected views remain
available; BH is refused for asymptotic breakouts.

Encouragement cannot combine predeclared segments with compliance `family=True`.
Use `family=False` for separately allocated uptake, or omit segments for an
unsegmented joint family. Existing encouragement breakout-readout restrictions
still apply. See [segment labels and source support](sequential-contracts.md#predeclared-segments).

## e-BH under sequential inference

[e-BH (Wang and Ramdas 2022)](https://arxiv.org/abs/2009.02824) is ordinary BH on
`1/e`. It controls FDR at `<=q` under arbitrary dependence and stopping time
**only when every input is a valid e-process under the registered joint
reveal**. `e_bh_select` assumes this property; it cannot establish it.
The exact Bernoulli route provides finite-sample evidence under its assumptions.
`AsymptoticMean` uses sign-gated plug-in mixture values from its asymptotic
sampling construction; estimated variance does not make them exact e-values
or let an oracle martingale theorem supply a finite-sample bound.

A family row reports `family_guarantee="finite_sample"` only when all members
are exact; any asymptotic member makes it `"asymptotic_sequential"`. Every row
shares `family_nominal_alpha`, the guarantee and `family_size=m`. Whole-window
continuous secondaries register `family=True` and their nominal alpha.
An encouragement family may include exact uptake alongside asymptotic ITT,
but the family's guarantee remains asymptotic.

Selection uses the complete registered member list and a verified joint
snapshot. Current members must share one reveal prefix; frozen members must
prove an ancestor. Models, cells, registration and reveal identity must match.
Ratio evidence is capped by denominator stability and masked if unresolved at
the nominal reporting ceiling; it can select only when its selected reporting
level also resolves the denominator. See [ratio availability](sequential-contracts.md#denominator-availability-and-selection).

The guarantee covers the discovery set at the current stopping date, not the
union across dates. Triggered families additionally require correct triggering,
no pre-trigger treatment effect, trigger-order stationarity, and no lookahead
from assigned results; the same status labels express **conditional** control,
not checks of those assumptions. See [triggered family decisions](sequential-contracts.md#per-cell-coverage-and-the-family-verdict).

### Freezing a metric

`capture_sequential(experiment, reader, finalized=True, freeze=["metric_name"])`
stops each observed registered cell for that metric, across treatment arms and
segments, at the capture's evidence. A cell with no observations in either arm
has nothing to freeze and remains monitored; name it again at a later capture.
Raw-record producers use `declare_sequential_freeze(snapshot, ["metric_name"])`.

Freezing is part of the hashed prefix, propagates through appends and parent
proofs, and cannot retroactively choose an earlier look. `run()` has no runtime
override. It fixes **evidence**, not the family verdict: e-BH reselects from
current/frozen evidence at every look, never a running maximum. A frozen
secondary's `discovery` and selected interval can change as other cells move.

Remembering a prior selection would violate e-BH's requirement that each
selected e-value reach `m/(q*R)` for the reported selected count `R`.
Selected intervals are reinverted from the same stopped likelihood state;
[the FCR construction (Xu, Wang and Ramdas 2022)](https://arxiv.org/abs/2203.12572)
preserves its applicable selected-set guarantee, not a union-over-looks claim.

### Older asymptotic families

Asymptotic metric/arm families previously used Bonferroni FWER rather than
e-BH FDR. Registration records the committed rule. An old snapshot, moments
export or compiled plan raises `sequential.continuation.legacy`, rather than
silently weakening its promise. Finish on the release that saved it, or
start a new registration by capturing without `previous=`. A restart is not
permission to spend alpha again on already-inspected outcomes without a valid
new protocol.

## FCR re-estimation of discovered cells

A secondary/breakout selected by BH or e-BH is re-estimated at
`fcr_alpha=min(q*R/m, nominal_alpha)`, with `R` selected cells and `m` full family
members. FCR means false coverage rate: the expected fraction of reported
selected intervals missing their targets, under the procedure's assumptions.
With few discoveries among many tests, intervals widen relative to nominal;
as more select, they narrow only up to the nominal-alpha cap.

Selection and individual significance differ. If `q*R/m` exceeds the nominal
cap, a selected row's reported interval need not exclude its null. Use
`discovery` for family selection and `stat_sig()` for the row's test, not one
as a substitute for the other. The exact/asymptotic label follows the family
members, not the fact that its interval was adjusted.

Only selected hypotheses are re-estimated on routes that filter estimation
evidence. Quantile families filter to the control and selected arms: an
unselected arm keeps its nominal estimate even if its interval at the stricter
corrected level would be too wide to report.

Sequential reinversion uses the same current or stopped state and preserves
its confidence-set geometry—bounded, unbounded, disconnected, full or empty.
Do not rebuild an interval from a point and standard error or drop a set
because its point is null. See [sequential output fields](result-fields.md#sequential-inference-two-coordinate-systems).

## Two-phase stored runs

`increment.run(experiment, reader, results_store=..., decision_log=...)`
checks the hash-only `Registration` and portable compiled plan before reading
data, then creates a validated `Run` from its cells. With `ResultsStore`, it
persists and reopens that pinned run before final verdicts; without a store,
the same evaluator uses the validated in-memory run.

`RunResult` exposes cells, effect projections, decisions and each view's full
rows/family fields through `readout(view)`. `RunResult.from_handle(handle,
registration=...)` reevaluates saved cells. Saved readout envelopes also
preserve membership, row status and family identity.

`DecisionLog` writes fixed-family events only after run/decision validation.
Replay resolves the pinned run and the versioned compiled-plan payload stored
with run and event. The source family stays fixed; these are not online FDR
procedures across runs or experiments. See [saved formats](saved-formats.md)
for portable versions and [artifact/replay contracts](artifacts-and-replay.md).
