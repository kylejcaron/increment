# Reading results

`Analysis.run()` returns one of two result families. The family fixes the row
type, but the scale of a value is a property of each row, so check every row
before comparing or combining rows. First establish population, estimand and
inference regime; then read the point with its interval or confidence set,
availability reason and selection qualification. The canonical
[Multiplicity](multiplicity.md) table explains the existing status literals
and guarantees; this guide explains how to interpret those disclosures on
result objects and tables.

| Evidence | Returns | Row type | Scale |
|---|---|---|---|
| Arm comparisons: `from_unit_summary`, `from_unit_panel`, `from_unit_day_artifact`, `from_definitions`, `from_moments` | `LiftEstimates` | `LiftEstimate` | Usually relative lift: `0.05` is 5% above control. Read each row's `value_scale`. |
| Switchback: `from_switchback_panel` | `ContrastResults` | `ContrastResult` | Additive difference in the metric's own units. |

A `LiftEstimate` can be absolute. An encouragement design's additive LATE row
and an observational metric reported with `run(value_scale={metric: "absolute"})`
carry `r.value_scale == "absolute"`, and then `r.lift.value` and its interval are
in the metric's own units. An encouragement design can also emit a relative LATE
sibling with the same estimand, so a LATE does not imply a scale. Check
`r.value_scale` (and `r.estimand`) on every row, and never compare a relative lift
with an additive value numerically.
`LiftEstimates.to_frame` and the readout rows have a `value_scale` column.
`ContrastResults.to_frame` does not, because every contrast is additive; only
the readout adapter below adds `value_scale="absolute"` to contrast rows.

The `readout_table` coeftable keeps per-row estimate evidence compact:
decision-scope completeness, partial-view state, integrity, availability,
multiplicity, and advisories appear as source-attributed qualification bullets
in the header. Source identifiers are abbreviated there; the full identifiers
remain in result objects and evidence columns. Each effect row includes a
compact interval-interpretation label; the full restrained explanation remains
in the collection `repr`. Failure rows keep the registered failure explanation
in a `Failure` column and their full structured context on the result object.

Scoped result collections also carry qualifications in their default
`repr`: integrity and scope completeness precede availability, multiplicity,
advisories, row qualifications, and effects. A failed integrity check is not
hidden by an otherwise favorable estimate, and `not_rejected` means no
detected mismatch, not proof of valid randomization. The collection display
includes retained scope even when filtering leaves no visible rows.

Interpretation separates the tested-null outcome from effect direction.
An interval containing the tested null is inconclusive, not evidence of no
effect. Excluding a nonzero margin can meet that test without establishing an
increase or decrease: direction is stated only when the interval also excludes
zero on the result's own axis. Absolute-native estimates and contrasts use
additive intervals; relative rows with an absolute margin use their retained
additive interval; sequential ratio bounds are interpreted after conversion to
lift coordinates. Daily arm means are values, not treatment/control contrasts.
Secondary selection status is shown separately from interval evidence; neither
is a ship/no-ship verdict.

`to_frame()` on scoped collections and `estimates_to_readout()` retain portable
scalar and canonical-JSON evidence columns, including comparison identity,
`analysis_population`, `source_snapshot_id`, integrity status/code/context,
required/available/hidden counts, unavailable identities and reasons,
advisory codes/messages/context, posterior-unavailability reasons, and
multiplicity family status. Nested disclosures are canonical JSON strings;
numeric nulls remain numeric nulls. These fields retain source and population
attribution across flattened and concatenated results.

`readout_table(collection)` retains collection qualifications in its header.
The zero-visible-row view is a deliberate exception to tables' usual reliance
on displayed row qualifications: it still renders retained scope and integrity
for a scoped collection, while row-only empty input reports that scope and
integrity are unknown. Export the collection's JSON (`model_dump_json()`) when
you need to preserve the empty view's complete scope evidence; a zero-row
frame/table has no row on which to store attribution.

An affirmative `r.discovery` is the raw selection result, not another
significance verdict: it can differ from `r.stat_sig()`. `r.p_value()` is the
row's own unadjusted sampling p-value; it is never divided by a multiplicity
factor. The row's decision compares it with the level its construction decides
at, shown as `decision_alpha` in the row's representation: `r.lift.alpha` for a
two-sided or open-sided interval, half of it for a closed one-sided interval
(the display alpha is central-equivalent), `r.binomial_set.decision_alpha` for an
exact conversion row, and the single-tail `r.abs_alpha` for an additive decision.
A Bonferroni-split primary or a family reissue sets it below the plan's nominal
alpha, so `p_value() <= decision_alpha` holds exactly when `stat_sig()` (an exact
binomial row decides with a strict `<`; a `low_reliability` breakout cell reports
its p-value but withholds the verdict). The family verdict is `r.discovery`. The ordinary
representation and `readout_table` place a selection qualification next to
that result, including its procedure, exploratory/confirmatory role, retained
family, and applicable guarantee or assumption. These disclosures do not
change the stored selection or interval.
`r.family_procedure` preserves the canonical correction name (`"e_bh"`,
`"bh"`, or `"bonferroni"`) alongside the family scope and selection. Rows
restored from payloads predating this field keep it as `None`; their display
reports that the procedure was not recorded rather than inferring it from
the interval or guarantee.

Ordinary `str(row)` and `repr(row)` use the same compact semantic display for
arm, breakout, daily, and additive contrast rows. A missing point is not
necessarily a failed cell: a point-unavailable result can retain an unbounded
or disconnected confidence set, whose geometry remains visible in the display.
For a coded cell failure only, `row.reason` returns the persisted
`failure_context["reason"]` when present, or the failure code otherwise. It is
derived from the failure fields and is not persisted; other unavailable
sampling, posterior, point, and interval reasons remain on their own fields.

When a triggered experiment is read without a `population` argument, `Analysis.run()` keeps
assigned and triggered results as separate rows. Pass `population="assigned"` or
`population="triggered"` to select one population; a selected result never pools the two.
Under a registered sequential plan each population reads its own retained chain of the
captured checkpoint; a triggered row never borrows an assigned checkpoint, and a process
committed without a triggered chain keeps an unavailable row with its refusal code and reason.

Daily and as-of `DailyLiftEstimate` rows have a compact representation for
ordinary printing: identity/date, estimate and interval geometry, scale,
inference/status, and any unavailable reason. Sequential bounds are labeled
as ratios; exact checkpoint and certificate state remains available through
`row.sequential_result` when explicitly inspected.

## What to read

For a `LiftEstimate` row `r`:

| Question | Read | Notes |
|---|---|---|
| How big is the effect? | `r.lift.value` | `r.lift` is `None` when no finite point exists. |
| How uncertain is it? | `r.lift.lb`, `r.lift.ub`, `r.lift.level` | An endpoint is `None` on the side named by `r.lift.open_side`. Both are `None` when relative inference is unavailable. |
| Where is the interval when `r.lift` is `None`? | `r.binomial_set`, `r.relative_confidence_set`, `r.confidence_set`, `r.sequential_result.bounds`; the additive `r.abs_lb`, `r.abs_ub` | The set that owns the row's interval, if the row has one. A `None` upper bound on a binomial set means unbounded. Some rows have no relative set at all; see below. |
| Is it significant? | `r.stat_sig()` | Honors one-sided tests, nonzero nulls, and the row's own inference. `None` when the row has no usable sampling evidence (a failed cell, or `r.sampling_available` False, as for a percentile-winsorized primary), matching the readout frame's `stat_sig`; never read `None` as a non-rejection. `BreakoutEstimate` and `DailyLiftEstimate` rows carry the same `stat_sig()` and `p_value()`. |
| Was it selected within a family? | `r.role`, `r.discovery`, `r.family_procedure`, `r.family_q` | See [Multiplicity](multiplicity.md). |
| Why is something missing? | `r.relative_unavailable_reason`, `r.relative_confidence_set.point_unavailable_reason`, `r.confidence_set.relative` endpoint `status` and `reason`, `r.sequential_result.point_reason` | The exact persisted reason. |
| Which inference regime? | `r.inference` | `"fixed"` is a single-look interval. `"always_valid"` has a finite-sample anytime-valid guarantee; `"asymptotic_mean"` supports repeated monitoring only under its asymptotic assumptions, without a finite-start guarantee. See [Sequential inference](sequential-inference.md). |

For a `ContrastResult` row `c`, read `c.estimate.value`, `c.estimate.lb`,
`c.estimate.ub`, `c.null_abs`, and `c.alternative`. All are additive.

Scoped readouts carry a `source_snapshot_id` that identifies the canonical
request, pinned source evidence and source identity, and integer sufficient
statistics such as counts. The identity excludes floating-point aggregates
and derived estimates, so the same unchanged evidence and request retains its
identity even when warehouse reduction order changes the last bits of a
floating-point result. Different requests, effective methods, populations, or
sources remain distinct; a saved result retains the identity it was created
with. If the request, pinned source identity, and counts stay the same while
floating-point outcome data change, this identity cannot distinguish those
changes. It is an identity for the recorded inputs, not a checksum of all
numeric outputs.

## Read the result as a decision, not just a number

For the question you intend to answer, first confirm the population, estimand,
scale, and analysis window. Then read the point together with its interval or
confidence set, the null and alternative, the inference regime, and the row's
decision status. An interval excluding zero is not a substitute for checking
the declared null, direction, family status, and whether the result is complete.
For relative and additive rows, keep the scale attached to every value.

Before acting on a collection, check that all required cells are represented.
A required cell may be an estimate or an unavailable row; unavailable rows
preserve the requested cell identity and carry the reason. Do not interpret a
missing estimate as a null effect or silently drop a failed cell. Use the
reason to choose the next step: correct missing or invalid source evidence,
request a supported analysis, or report that the requested result could not be
estimated. A partial result is not a complete decision.

### Interpret status and scope

| Integrity status | Meaning |
|---|---|
| `failed` | A mismatch was detected; do not let a favorable effect hide it. |
| `not_rejected` | No mismatch was detected; this does not prove valid randomization. |
| `not_checked_missing_declaration` | The assignment law was not declared, so it could not be checked. |
| `not_checked_missing_counts` | Required assignment counts were unavailable. |
| `unsupported_assignment` | This source or design cannot evaluate the declared assignment law. |
| `not_applicable` | The check does not apply to this source/design/population; it is not a pass. |

Use assignment evidence from the assigned population at its declared
randomization grain. Triggered counts describe an outcome subset, not
assignment-law evidence.

- **`exploratory_unadjusted` means no family adjustment.** Treat that result
  as exploratory rather than as a confirmatory discovery. If the question is
  decision-critical, define the appropriate family and role before using
  results to decide; see [Multiplicity](multiplicity.md). Filtering displayed
  rows does not change the retained family scope.
- **A sampling result does not imply a posterior.** Sampling estimates,
  confidence sets, p-values, and significance remain distinct from posterior
  estimates and probabilities. If `posterior_available` is false, use its
  recorded reason; do not reconstruct a posterior from sampling fields. Read
  the valid sampling interval if that is the question, or choose a supported
  posterior route if a Bayesian decision is required.
- **A triggered result answers a different population question.** It includes
  units at their first eligible trigger and anchors their outcome window there.
  Keep it separate from the assigned-population result and its multiplicity
  family. Conditioning on triggering is causally valid only under the
  experiment's assumptions; triggering affected by treatment changes the
  estimand and can introduce selection bias. Use the assigned result for an
  assignment-based estimate when that is the intended question.

After resolving the status, report the estimate with its scale, interval or
set, population, and any unavailable or multiplicity caveat. If a requirement
cannot be supported by the source or construction, use the specific coded
refusal and its stated route forward instead of substituting another
population or inference method.


## Fixed horizon: guard the missing point

The collection summary leads with the metric results. It shows integrity and
scope qualifications only when they affect how to read those results. This
example includes an exact conversion row with zero control conversions, so its
relative point is unavailable even though the confidence set remains informative.

```python
from random import Random

import polars as pl

from increment import Analysis, Randomized

rng = Random(13)
n = 200
arms = ["treatment" if rng.random() < 0.5 else "control" for _ in range(n)]
df = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(n)],
        "variant": arms,
        "revenue": [10.0 + (i * 7) % 11 + (arm == "treatment") for i, arm in enumerate(arms)],
        "converted": [int(arm == "treatment" and i % 5 < 2) for i, arm in enumerate(arms)],
    }
)
results = Analysis.from_unit_summary(
    df,
    unit="user_id",
    group="variant",
    design=Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    ),
    metrics={"revenue": "mean", "converted": "conversion"},
).run()
print(results)

for r in results:
    if r.lift is None or r.lift.lb is None or r.lift.ub is None:
        # no point, or a withheld or open bound; str shows the row's set or reason
        print(str(r))
        continue
    print(
        f"{r.metric} / {r.group_id}: "
        f"lift={r.lift.value:+.2%} [{r.lift.lb:+.2%}, {r.lift.ub:+.2%}] "
        f"significant={r.stat_sig()}"
    )

# The default frame retains qualified evidence; selected views retain requested
# projection columns and view_partial.
decision_table = results.to_frame(backend="polars", columns="summary")
custom_table = results.to_frame(
    backend="polars",
    columns=["metric", "group_id", "method", "lift", "lb", "ub"],
)
```

`to_frame(columns="summary")` is a compact per-model preset; the default
`columns=None` returns the full projection, and unknown custom column names
raise `results.frame.unknown_columns`.

??? info "What the summary preset keeps"
    `columns=None` remains the full projection. Each `"summary"` preset contains
    existing projection fields only, including row identity (`estimand` and
    `value_scale`), `decision_scope_complete`, and `multiplicity_status`; breakout
    and daily summaries also retain segment identity, and daily summaries retain
    unavailability reasons. Models without a summary preset refuse. Every view
    appends `view_partial`, including custom column sequences, so a sliced
    collection remains visibly partial. A custom sequence is checked against the
    projection's column names and returned in canonical projection order before
    `view_partial`; unknown names raise `results.frame.unknown_columns`.


<!-- output -->
```text
assigned — treatment vs control
converted interpretation: excludes zero at 95.0% confidence; effect direction supported (increase).
converted additive effect: 0.36036, set [0.270641, 0.45008].
converted binomial set central: [1.61559, +∞] at 0.95 confidence.
converted test alternative: two-sided; null (relative lift): 0.
converted result identity: method unadjusted; method role decision; estimand itt.
revenue (outcome, relative lift): 0.0946974, set [0.0327365, 0.160376] at 0.95 confidence.
revenue interpretation: excludes zero at 95.0% confidence; effect direction supported (increase).
revenue additive effect: 1.39812, set [0.504942, 2.29129].
revenue test alternative: two-sided; null (relative lift): 0.
revenue result identity: method unadjusted; method role decision; estimand itt.
Integrity: not_rejected — no detected mismatch; not proof of valid randomization.
Multiplicity (outcome): undeclared_plan; family retained scope.
revenue / treatment: lift=+9.47% [+3.27%, +16.04%] significant=True
LiftEstimate(metric='converted', group='treatment', method='unadjusted', role='decision', estimand='itt', population='assigned', inference='fixed', alternative='two-sided', scale='relative', value=unavailable (relative), interval=(+1.616, +inf), route='finite_sample', discovery=None)
```

## Read the row's scale

- An exact conversion row whose control arm has zero events. The ratio has a
  zero denominator, but the exact binomial set `r.binomial_set` still bounds the
  lift.
- A joint ratio set with no representable point. The set is
  `r.relative_confidence_set`, and `r.relative_confidence_set.point_unavailable_reason`
  names the reason.
- A fixed-horizon mean row whose arm mean is not positive:
  `r.relative_unavailable_reason == "nonpositive_arm_mean"`. The log-scale
  relative lift is undefined and there is **no** relative set. The additive
  difference survives in `r.abs_diff`; `r.abs_lb` and `r.abs_ub` can be `None`,
  including when both arms are constant. Check the bounds before use. When the
  additive interval is present and the row's null is the unshifted zero (no
  absolute or relative margin declared), it is the row's decision: `r.p_value()`
  is the additive test of the difference against zero under the row's
  alternative, using the reference the interval was cut from (Welch `t` or
  Normal, `r.abs_reference_kind`), and `r.stat_sig()` is whether that interval
  excludes zero. Such a row takes part in its family on that evidence and, when
  selected, has its additive interval reissued at the family level. A row with
  no additive interval, or one that declares a relative margin (which has no
  additive counterpart), keeps `r.p_value() == 1.0` and `r.stat_sig()` False: a
  non-rejection, never a discovery.
- A mixed-CUPED (`cuped_linear`) row whose relative projection cannot be
  represented: `r.relative_unavailable_reason == "relative_unrepresentable"`,
  with the exact refusal in `r.relative_error_code`. The relative point lies
  outside what a float can hold (the log-scale lift underflows to a total loss
  or overflows) or the projection is not finite; the additive difference and its
  `t` interval survive in `r.abs_diff`, `r.abs_lb` and `r.abs_ub`, and an
  absolute-margin guardrail still decides from them. When only the relative
  *interval* is unrepresentable (at the nominal level or after FCR widening),
  the row keeps its point and sampling statistics, `r.lift.lb`/`r.lift.ub` are
  `None`, `r.relative_unavailable_reason` stays `None` and
  `r.relative_error_code` names the refusal; `r.p_value()` still answers.
- A row whose joint relative covariance is unusable: it cannot be represented as
  a float (extremely large scores) or is not positive semidefinite (a singular
  positive semidefinite covariance is still usable).
  `r.relative_unavailable_reason` is `"joint_covariance_unrepresentable"` or
  `"joint_covariance_indefinite"`, and there is **no** relative set. Whether the
  point survives depends on the producer: an adjusted estimate with extremely large
  scores has `r.lift is None`, while other rows keep the point and withhold
  only its bounds (see "A point without an interval"). Only `r.abs_diff` is
  guaranteed to survive: `r.abs_se`, `r.abs_lb`, and `r.abs_ub` can each be
  `None`, so check them before use.
- A winsorized metric with a `confidence_set`. A missing point leaves
  `r.confidence_set.relative` with a status and reason per endpoint (for
  example `unbounded` with `denominator_nonseparation`), and the additive
  interval stays in `r.abs_lb` and `r.abs_ub`.
- A sequential row whose point is withheld. The confidence set is in
  `r.sequential_result.bounds`, and `r.sequential_result.point_reason` names the
  reason. An asymptotic set can itself be unavailable: check that
  `r.sequential_result.bounds.status` is not `"unavailable"` before reading
  endpoints, and read `bounds.reason` when it is.

!!! warning "Check scale and population before comparing rows"
    A `LiftEstimate` can be absolute: encouragement designs can return
    additive or relative LATE rows, and observational rows can be made
    absolute with `run(value_scale={metric: "absolute"})`. Read
    `r.value_scale` and `r.estimand` before comparing values.
    `LiftEstimates.to_frame` and the readout rows have a `value_scale` column.
    `ContrastResults.to_frame` does not, because every contrast is additive; only
    the readout adapter below adds `value_scale="absolute"` to contrast rows.

    When a triggered experiment is read without a `population` argument,
    `Analysis.run()` keeps assigned and triggered results as separate rows;
    select with `population="assigned"` or `population="triggered"`. A triggered row belongs
    to its own multiplicity family. Conditioning on triggering is causally
    valid only under the experiment's assumptions; triggering affected by
    treatment changes the estimand and can introduce selection bias.

??? info "Statistical details: absolute lift rows"
    A `LiftEstimate` can be absolute. An encouragement design's additive LATE row
    and an observational metric reported with `run(value_scale={metric: "absolute"})`
    carry `r.value_scale == "absolute"`, and then `r.lift.value` and its interval are
    in the metric's own units. An encouragement design can also emit a relative LATE
    sibling with the same estimand, so a LATE does not imply a scale. Check
    `r.value_scale` (and `r.estimand`) on every row, and never compare a relative lift
    with an additive value numerically.

## What to read

For a `LiftEstimate` row `r`:

| Question | Read | Notes |
|---|---|---|
| How big is the effect? | `r.lift.value` | `r.lift` is `None` when no finite point exists. |
| How uncertain is it? | `r.lift.lb`, `r.lift.ub`, `r.lift.level` | An endpoint is `None` on the side named by `r.lift.open_side`. Both are `None` when relative inference is unavailable. |
| Where is the interval when `r.lift` is `None`? | `r.binomial_set`, `r.relative_confidence_set`, `r.confidence_set`, `r.sequential_result.bounds`; the additive `r.abs_lb`, `r.abs_ub` | The set that owns the row's interval, if the row has one. A `None` upper bound on a binomial set means unbounded. Some rows have no relative set at all; see [below](#fixed-horizon-guard-the-missing-point). |
| Is it significant? | `r.stat_sig()` | Tri-state: `True` means the row meets its decision rule, `False` means it does not, and `None` when the row has no usable sampling evidence (a failed cell, or `r.sampling_available` False, as for a percentile-winsorized primary), matching the readout frame's `stat_sig`; never read `None` as a non-rejection. Honors direction and the row's inference. `BreakoutEstimate` and `DailyLiftEstimate` rows carry the same `stat_sig()` and `p_value()`. |
| Was it selected within a family? | `r.role`, `r.discovery`, `r.family_procedure`, `r.family_q` | See [Multiplicity](multiplicity.md). |
| Why is something missing? | `r.relative_unavailable_reason`, `r.relative_confidence_set.point_unavailable_reason`, `r.confidence_set.relative` endpoint `status` and `reason`, `r.sequential_result.point_reason`; `row.reason` for a coded cell failure | The exact persisted reason. For a coded cell failure only, `row.reason` returns the persisted `failure_context["reason"]` when present, or the failure code otherwise. It is derived from the failure fields and is not persisted; other unavailable sampling, posterior, point, and interval reasons remain on their own fields. |
| Which inference regime? | `r.inference` | `"fixed"` is a single-look interval. `"always_valid"` has a finite-sample anytime-valid guarantee; `"asymptotic_mean"` supports repeated monitoring only under its asymptotic assumptions, without a finite-start guarantee. See [Sequential inference](sequential-inference.md). |

For a `ContrastResult` row `c`, read `c.estimate.value`, `c.estimate.lb`,
`c.estimate.ub`, `c.null_abs`, and `c.alternative`. All are additive.

`str(row)` and `repr(row)` share one compact display for arm, breakout,
daily, and contrast rows, including a set-only row's unbounded or
disconnected set and a `DailyLiftEstimate`'s unavailable reason.

## Significance, p-values and selection

`r.p_value()` is the row's **unadjusted** sampling p-value; it is never
divided by a multiplicity factor. For fixed-horizon rows, `r.stat_sig()` compares
that p-value with `decision_alpha`, the level the row's own construction decides
at, which a Bonferroni-split primary or a family reissue sets below the plan's nominal
alpha. Registered sequential rows have no p-value (`p_value()` refuses); their
`stat_sig()` verdict reads the sequential result directly. `r.discovery` is different:
it is the raw family selection result, not another significance verdict, and can
differ from `r.stat_sig()`.

??? info "Statistical details: decision level and disclosures"
    For fixed-horizon rows, the decision compares the p-value with the level
    shown as `decision_alpha` in the row's representation: `r.lift.alpha` for a
    two-sided or open-sided interval, half of it for a closed one-sided interval
    (the display alpha is central-equivalent), `r.binomial_set.decision_alpha` for an
    exact conversion row, and the additive interval's central-equivalent `r.abs_alpha`
    for an additive decision: use `r.abs_alpha` for a two-sided alternative and
    `r.abs_alpha / 2` for a one-sided alternative.
    A Bonferroni-split primary or a family reissue sets it below the plan's nominal
    alpha, so `p_value() <= decision_alpha` holds exactly when `stat_sig()` (an exact
    binomial row decides with a strict `<`; a fixed-horizon `low_reliability`
    breakout cell reports its p-value but withholds its Wald verdict).
    Sequential rows instead decide from their retained confidence sequence:
    a small segment's unit-count flag does not invalidate that construction.

    A conversion family member carrying `delta_evidence.composite` reports
    its count-based composite p-value. Its exact reading retains its nominal
    level and tested null. Retesting a copied row with a changed `null_lift`
    reads the same counts against that new null at the retained nominal level;
    it does not reuse the original null's exact p-value.

    `r.family_procedure` preserves the canonical correction name (`"e_bh"`,
    `"bh"`, or `"bonferroni"`) alongside the family scope and selection. Rows
    restored from payloads predating this field keep it as `None`; their display
    reports that the procedure was not recorded rather than inferring it from
    the interval or guarantee.

    The ordinary representation and `readout_table` place a selection
    qualification next to the result, including its procedure,
    exploratory/confirmatory role, retained family, and applicable guarantee
    or assumption; these disclosures do not change the stored selection or
    interval. `readout_table` keeps estimate evidence compact and shows
    decision-scope completeness, partial-view state, and multiplicity
    provenance as header notes when informative;
    the fields remain on result objects and in `to_frame()`, and failure
    rows keep a concise `Failure` column with their full structured context
    on the result object. With registered sequential inference, each
    population reads its own retained chain of the captured checkpoint; a
    triggered row never borrows an assigned checkpoint, and a process
    committed without a triggered chain keeps an unavailable row with its
    refusal code and reason. Inspect `row.sequential_result` for exact
    checkpoint and certificate state.

    Daily and as-of `DailyLiftEstimate` rows have a compact representation for
    ordinary printing: identity/date, estimate and interval geometry, scale,
    inference/status, and any unavailable reason. Sequential bounds are labeled
    as ratios; exact checkpoint and certificate state remains available through
    `row.sequential_result` when explicitly inspected.
    For a `LiftEstimate` row `r`, inspect `r.sequential_result`.

    Scoped readouts carry a `source_snapshot_id` that identifies the
    canonical request, pinned source evidence and source identity, and
    integer sufficient statistics such as counts. The identity excludes
    floating-point aggregates and derived estimates, so the same unchanged
    evidence and request retains its identity even when warehouse reduction
    order changes the last bits of a floating-point result. Different
    requests, effective methods, populations, or sources remain distinct; a
    saved result retains the identity it was created with. If the request,
    pinned source identity, and counts stay the same while floating-point
    outcome data change, this identity cannot distinguish those changes. It
    is an identity for the recorded inputs, not a checksum of all numeric
    outputs.

## Before you act

Confirm the population, estimand, scale, and window; then read the point with
its interval or set, the null and alternative, the inference regime, and the
decision status. An interval excluding zero is not a substitute for that
check. Report the estimate with its scale, interval or set, population, and
any unavailable or multiplicity caveat.

!!! warning "Readings that are not what they look like"
    - **A missing estimate is not a null effect.** Unavailable rows keep the
      cell identity and carry a reason; a partial result is not a complete
      decision.
    - **`NotApplicable` assignment integrity is not a passing check.** It
      could not be evaluated for that source, design, or population.
    - **`exploratory_unadjusted` is not a confirmatory discovery.** No family
      adjustment was applied. For a decision-critical question, define the
      family and role before reading results ([Multiplicity](multiplicity.md));
      filtering displayed rows does not change the retained family scope.
    - **A sampling result is not a posterior.** If `posterior_available` is
      false, use its recorded reason; do not reconstruct a posterior from
      sampling fields.
    - **A triggered result conditions on triggering.** It includes units at
      their first eligible trigger and anchors their window there. Use the
      assigned result for an assignment-based estimate.

??? info "What to do in each case"
    Before acting on a collection, check that every required cell is present
    as an estimate or an unavailable row. Use the reason to correct the
    source, request a supported analysis, or report that the result could not
    be estimated. If the source or construction cannot support a requirement,
    use the coded refusal and its stated route forward instead of
    substituting another population or inference method.

    For assignment-law checks, use the assigned population at its stated
    assignment grain; triggered counts describe an outcome subset and do not
    validate randomization. For a Bayesian decision, read the sampling
    interval if that is the question, or choose a supported posterior route.

## When a point is missing

`r.lift` is `None` when no finite point estimate exists. Do not assume a
missing point comes with a set: when `r.relative_unavailable_reason` is set,
the row has **no** relative set, so read the additive `r.abs_diff`, plus
`r.abs_lb` and `r.abs_ub` when they are not `None`. When the reason is
`None`, read whichever set the table names.

| Cause | Where the evidence is |
|---|---|
| Exact conversion row whose control arm has zero events | The ratio has a zero denominator, but `r.binomial_set` still bounds the lift. |
| Joint ratio set with no representable point | `r.relative_confidence_set`; `r.relative_confidence_set.point_unavailable_reason` names the reason. |
| Fixed-horizon mean row whose arm mean is not positive | `r.relative_unavailable_reason == "nonpositive_arm_mean"`: the log-scale relative lift is undefined and there is **no** relative set. `r.abs_diff` survives; `r.abs_lb` and `r.abs_ub` can be `None`, including when both arms are constant. |
| Joint relative covariance unusable | `r.relative_unavailable_reason` is `"joint_covariance_unrepresentable"` (not representable as a float, extremely large scores) or `"joint_covariance_indefinite"` (not positive semidefinite; a singular one is still usable), with **no** relative set. Only `r.abs_diff` is guaranteed; `r.abs_se`, `r.abs_lb`, and `r.abs_ub` can each be `None`. |
| Winsorized metric with a `confidence_set` | `r.confidence_set.relative` keeps a status and reason per endpoint (for example `unbounded` with `denominator_nonseparation`); the additive interval stays in `r.abs_lb` and `r.abs_ub`. |
| Sequential row whose point is withheld | `r.sequential_result.bounds`, with `r.sequential_result.point_reason`. Check that `r.sequential_result.bounds.status` is not `"unavailable"` before reading endpoints, and read `bounds.reason` when it is. |

??? info "Statistical details: how a row without a relative set decides"
    For a nonpositive arm mean, when the additive interval is present and
    the row's null is the unshifted zero (no absolute or relative margin
    declared), it is the row's decision: `r.p_value()` is the additive test
    of the difference against zero under the row's alternative, using the
    reference the interval was cut from (Welch `t` or Normal,
    `r.abs_reference_kind`), and `r.stat_sig()` is whether that interval
    excludes zero. Such a row takes part in its family on that evidence and,
    when selected, has its additive interval reissued at the family level. A
    row with no additive interval, or one that declares a relative margin
    (which has no additive counterpart), keeps `r.p_value() == 1.0` and
    `r.stat_sig()` False: a non-rejection, never a discovery.

    For an unusable joint covariance, whether the point survives depends on
    the producer: an adjusted estimate with extremely large scores has
    `r.lift is None`, while other rows keep the point and withhold only its
    bounds (see [A point without an interval](#a-point-without-an-interval)).

A missing point does not erase the evidence. The `converted` row is
significant; its interval lives on `r.binomial_set`, and the upper end is
unbounded. `require_lift()` returns the point or raises a coded refusal, for
when a missing point should stop the program rather than be handled:

```python
from increment import CodedError

row = next(r for r in results if r.metric == "converted")
assert row.lift is None and row.reference_kind == "binomial"
assert row.stat_sig()
bset = row.binomial_set
assert bset.lower > 0 and bset.upper is None  # lift scale; None means unbounded

try:
    row.require_lift()
except CodedError as err:
    assert err.code == "estimation.results.lift.binomial_lift_availability"
else:
    raise AssertionError("a set-only row has no point to require")
```

Two of the other causes are reproducible with small frames. A mean metric
whose control mean is negative has no relative lift and no relative set; its
additive interval carries the decision:

```python
k = 40
negative = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * k)],
        "variant": ["control"] * k + ["treatment"] * k,
        "delta": [-3.0 + (i * 7) % 5 for i in range(k)] + [-1.0 + (i * 7) % 5 for i in range(k)],
    }
)
flat_mean = Analysis.from_unit_summary(
    negative,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"delta": "mean"},
).run()[0]

assert flat_mean.lift is None and flat_mean.relative_confidence_set is None
assert flat_mean.relative_unavailable_reason == "nonpositive_arm_mean"
assert flat_mean.abs_lb < flat_mean.abs_diff < flat_mean.abs_ub
assert flat_mean.stat_sig() is (flat_mean.abs_lb > 0.0)  # the additive interval decides
assert flat_mean.p_value() <= 1.0
```

A winsorized metric whose control arm sums to zero keeps a confidence set with
a finite lower and an unbounded upper relative endpoint, each with its own
status:

```python
from increment import MetricSpec

zero_control = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * k)],
        "variant": ["control"] * k + ["treatment"] * k,
        "revenue": [0.0] * k + [float(1 + (i * 7) % 5) for i in range(k)],
    }
)
winsorized = Analysis.from_unit_summary(
    zero_control,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[
        MetricSpec(
            name="revenue",
            type="mean",
            winsorization={
                "upper_percentile": 0.9,
                "support": {"lower": 0.0, "provenance": "Revenue is nonnegative by definition"},
                "inference": {"method": "joint-rank-projection-v1"},
            },
        )
    ],
).run()[0]

relative = winsorized.confidence_set.relative
assert winsorized.lift is None and winsorized.reference_kind == "confidence_set"
assert relative.lower.status == "finite" and relative.upper.status == "unbounded"
assert winsorized.abs_lb < winsorized.abs_diff < winsorized.abs_ub
```

### A point without an interval

A point can exist while its relative interval does not. When the estimated
variance of the relative contrast is exactly zero, the row keeps the point,
withholds the relative bounds and relative-scale decision, and names the
reason; a row with `null_abs` can still decide from available additive
bounds. Clustered data whose values are identical within each arm is one
example:

```python
m = 100
degenerate = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * m)],
        "variant": ["control"] * m + ["treatment"] * m,
        "store": [f"s{i // 2:03d}" for i in range(2 * m)],
        "revenue": [10.0] * m + [11.0] * m,
    }
)
flat = Analysis.from_unit_summary(
    degenerate,
    unit="user_id",
    group="variant",
    control="control",
    metrics={"revenue": "mean"},
    cluster="store",
).run()[0]

assert flat.lift is not None and flat.lift.lb is None and flat.lift.ub is None
assert flat.relative_unavailable_reason == "zero_relative_variance"
assert not flat.stat_sig()  # no interval, so no decision
assert flat.abs_diff == 1.0  # the additive difference is still reported
```

Check `lb` and `ub` against `None`, or read `r.relative_unavailable_reason`
first, before formatting them. A finite point with
`"joint_covariance_unrepresentable"` or `"joint_covariance_indefinite"` is
the same case: its bounds are `None`, so the loop above prints its `repr`.

## One-sided tests

A directional conversion test returns an open interval. `open_side` names the
unbounded end, and that bound is `None`:

```python
from increment import AnalysisPlan, MetricSpec

n = 100
purchases = pl.DataFrame(
    {
        "user_id": [f"u{i:03d}" for i in range(2 * n)],
        "variant": ["control"] * n + ["treatment"] * n,
        "purchased": [1] * 30 + [0] * 70 + [1] * 45 + [0] * 55,
    }
)
directional = Analysis.from_unit_summary(
    purchases,
    unit="user_id",
    group="variant",
    control="control",
    metrics=[MetricSpec(name="purchased", type="conversion")],
    plan=AnalysisPlan(primary="purchased", alternative="greater"),
).run()[0]

lift = directional.lift
assert lift.open_side == "upper" and lift.ub is None
assert lift.lb <= lift.value  # the finite bound and the point share one scale
assert lift.excludes(0.0) and directional.stat_sig()
```

`excludes()` reads only the finite side of an open interval. Read `open_side`
instead of assuming it is set: an ordinary fixed-horizon mean row tests the
same direction but returns a closed interval.

??? info "Statistical details: one-sided levels"
    A one-sided test spends all of `alpha` on its single tail, so its `level`
    is the central equivalent: `level=0.9` for the default `alpha=0.05`. A
    secondary row selected into the discovery family is the exception: its
    interval is selection-adjusted, so it keeps its open side at a different
    `level` (for example `open_side == "upper"` at `level=0.975`).

## Sequential inference: two coordinate systems

A sequential row stores its confidence set in exact **ratio** coordinates,
where `1.0` means no effect, and displays it on the **lift** scale, where
`0.0` means no effect. Know which one you are reading:

| Where | Scale |
|---|---|
| `r.sequential_result.bounds` and the `ratio_set` in the row's `repr` | Ratio, exact `Fraction` endpoints. |
| `r.lift.value` | Lift point estimate (float); `0.0` means no effect. |
| `r.lift.lb`, `r.lift.ub`; frame columns `lb`, `ub` | Displayed lift bounds: ratio endpoints minus one, rounded outward to floats; withheld when no single interval is displayed. |
| Frame columns `sequential_lower`, `sequential_upper`; sequential readout `lower`, `higher` | Retained-set endpoints on the lift scale, rounded outward to floats. For a disconnected set these are outer endpoints, **not a confidence interval**; inspect `sequential_status` and `sequential_components`. |

!!! warning "A disconnected set is not an interval"
    Asymptotic sets can be disconnected, empty, or unbounded. For a
    disconnected or empty set `r.lift` carries only the point, because
    collapsing a set to its hull would misstate it; read `sequential_status`
    and `sequential_components` for the geometry. A missing endpoint is
    `None` and is never replaced by a number on the wrong scale. Lift-scale
    endpoints are never narrower than the ratio bounds, because rounding
    moves them outward.

```python
import polars as pl

from increment import Analysis, AnalysisPlan, InferenceSpec, MetricSpec, Randomized

k = 200
monitored = pl.DataFrame(
    {
        "unit": [f"u{i:04d}" for i in range(2 * k)],
        "arm": ["control"] * k + ["treatment"] * k,
        "enrolled_on": [1] * (2 * k),
        "revenue": [10.0 + (i * 37) % 17 for i in range(k)]
        + [11.5 + (i * 37) % 17 for i in range(k)],
    }
)
sequential = Analysis.from_unit_summary(
    monitored,
    unit="unit",
    group="arm",
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    metrics=[MetricSpec(name="revenue", type="mean")],
    experiment_id="revenue-monitor",
    plan=AnalysisPlan(
        primary="revenue",
        inference=InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=400),
    ),
    exposure_date="enrolled_on",
).run()
row = sequential[0]
assert row.inference != "fixed"

lift = row.lift  # lift scale
bounds = row.sequential_result.bounds  # ratio scale
assert lift.lb <= lift.value <= lift.ub
assert float(bounds.lower) <= 1 + lift.value <= float(bounds.upper)
assert lift.lb <= float(bounds.lower) - 1 and float(bounds.upper) - 1 <= lift.ub

frame = sequential.to_frame(backend="polars").row(0, named=True)
assert frame["sequential_status"] == "bounded"
# The frame's sequential endpoints are lift-scale: they surround the lift
# point, which a ratio-scale interval around 1.0 would not.
assert frame["sequential_lower"] <= frame["lift"] <= frame["sequential_upper"]
```

The exact `always_valid` route for conversion metrics reads the same way. Its
bounds are `Fraction` ratios and its certificate adds `log_e`, the evidence
against the registered null:

```python
from fractions import Fraction

registered = pl.DataFrame(
    [
        {"unit": f"{i:08d}-{arm}", "arm": arm, "outcome": value, "exposure": i}
        for i, pair in enumerate(zip([0, 0, 0, 1] * 50, [0, 1, 1, 1] * 50, strict=True))
        for arm, value in zip(("control", "treatment"), pair, strict=True)
    ]
)
exact = Analysis.from_unit_summary(
    registered,
    unit="unit",
    group="arm",
    exposure_date="exposure",
    metrics=[MetricSpec(name="outcome", type="conversion")],
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    experiment_id="experiment",
    plan=AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", baseline_rate=Fraction(3, 10)),
    ),
).run()[0]

result = exact.require_sequential_result()
assert exact.inference == "always_valid" and exact.stat_sig()
assert result.bounds.lower <= 1 + Fraction(exact.lift.value) <= result.bounds.upper
assert exact.lift.lb <= float(result.bounds.lower) - 1
assert result.log_e > 0
```

See [Registered sequential inference](sequential-inference.md) for how these
sets are built, including the e-BH secondary family.

## Switchback contrasts are additive

`from_switchback_panel` returns `ContrastResults`. Each `ContrastResult`
carries an additive `estimate` in the metric's own units, with `null_abs` as
its null. This panel has four independent blocks over a three-unit roster:

```python
import math

from increment import (
    Randomized,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)

orders = [("control", "treatment")] * 2 + [("treatment", "control")] * 2
shifts = [-1, 1, -2, 2]
panel = pl.DataFrame(
    [
        {
            "unit": f"u{unit}",
            "cycle": block,
            "period": period,
            "step": 0,
            "group": group,
            "value": 10.0 + (effect + shifts[block]) * (group == "treatment"),
        }
        for unit, effect in enumerate([4, 6, 8])
        for block, order in enumerate(orders)
        for period, group in enumerate(order)
    ]
)
contrasts = Analysis.from_switchback_panel(
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
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    ),
).run()
c = contrasts[0]

estimate = c.estimate
assert estimate.lb <= estimate.value <= estimate.ub and estimate.open_side is None
assert c.alternative == "two-sided" and c.null_abs == 0.0
assert math.isclose(estimate.value, 6.0)  # the mean unit effect, in outcome units
```

The estimate is not a relative lift. `ContrastResult` has no `stat_sig()`;
the readout adapter below supplies the verdict. See
[Switchback contrasts](switchback.md) for the estimand and its unit-cycle
`standard_error_unavailable_reason`, [Multiplicity](multiplicity.md) for
`role`, `discovery`, and `family_q`, and
[Observational comparisons](observational.md#near-zero-baselines-reporting-the-additive-effect)
for `relative_unavailable_reason`.

## Results as frames and flat rows

Every result list converts to a frame in your preferred library with
`to_frame(backend="polars")` (keyword-only for `ContrastResults`). A
`relative_confidence_set` or winsor `confidence_set` appears only as a `repr`
text column of the same name, so read those sets from the result object and
use `abs_lb` and `abs_ub` for the additive interval.

| Row | Frame columns |
|---|---|
| Set-only exact binomial row | `set_lower`, `set_upper`, `set_level`, `set_numerical_qualification`, from `binomial_set` only. |
| Sequential row | `sequential_lower`, `sequential_upper`, `sequential_status`, `sequential_components`. |

??? info "Statistical details: `set_numerical_qualification`"
    `scipy_special_function_error_model_conditional_v1` marks an enclosure
    conditional on the deployed SciPy/Boost special-function error model;
    `legacy_unrecorded_v1` marks an older serialized set that records no
    arithmetic qualification claim.

```python
table = results.to_frame(backend="polars")
converted = table.filter(pl.col("metric") == "converted").row(0, named=True)
assert converted["lift"] is None
assert converted["set_lower"] is not None and converted["set_upper"] is None

assert "estimate" in contrasts.to_frame(backend="polars").columns
```

`increment.tables.estimates_to_readout` adapts either family to one flat
`dict` per row, so one loop or one dataframe reads a mixed list of
`LiftEstimate`, `BreakoutEstimate`, `DailyLiftEstimate`, and `ContrastResult`
rows:

| Key | Meaning |
|---|---|
| `lift`, `lower`, `higher`, `level`, `open_side` | The row's estimate and interval. On a contrast row `lift` holds the additive estimate, so read it with `value_scale`. `open_side` comes from the row's `lift`, so a set-only row's open upper end shows only as `higher is None`. |
| `stat_sig`, `null_lift`, `null_abs` | The verdict and the nulls it was tested against. |
| `value_scale`, `estimand`, `inference`, `role`, `discovery` | Row identity and status. Contrast rows set `value_scale="absolute"` and `group_id=treatment_group`. |
| `decision_scale`, `decision_available` | `LiftEstimate`, `BreakoutEstimate`, and `DailyLiftEstimate` only: the scale the verdict is made on (`"absolute"` for a declared absolute margin, an additive-zero decision, or a native absolute row; `"relative"` otherwise), and whether a `False` verdict is a genuine non-rejection rather than the absence of a test. |
| `sequential_*` | Added when the row is sequential. |

```python
from increment.tables import estimates_to_readout

rows = estimates_to_readout([*results, *contrasts])
by_metric = {row["metric"]: row for row in rows}

unbounded = by_metric["converted"]
assert unbounded["lift"] is None and unbounded["lower"] > 0
assert unbounded["higher"] is None and unbounded["stat_sig"]

additive = by_metric["value"]
assert additive["value_scale"] == "absolute"
assert additive["group_id"] == additive["treatment_group"] == "treatment"
assert additive["lower"] <= additive["lift"] <= additive["higher"]
```

??? info "Install notes"
    The adapter needs only the core install: `estimates_to_readout` imports
    without `pandas` or `coeftable`, and in a wheel-only environment with
    `pyarrow` and neither of them it converted a zero-control conversion row
    to `lift=None`, a finite `lower`, `higher=None`, and `stat_sig=True`.
    Rendering the rows with `readout_table` requires the `tables` extra.
    `to_frame()` uses the input dataframe's library by default. Results without
    a dataframe input prefer `pandas`, then `polars`, then `pyarrow`; pass
    `backend="pandas"`, `"polars"`, or `"pyarrow"` to choose explicitly.
