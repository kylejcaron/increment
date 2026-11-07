"""Exploratory family correction of finished estimate rows.

``select_exploratory_family`` applies the flat-family rule of ``increment.estimation.family`` to
finished rows: whole-window and segment decision rows, uncorrected, are one BH family, and each
selected interval is reissued at the FCR level from the construction the row persists (see
``_fcr_reinterval``). Each member's evidence is the estimators' own derivation, so this module
sits above the estimation engine.

Pure functions - no I/O, no ibis, no warehouse query construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from increment.errors import (
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    refusals,
    refuse,
)
from increment.estimation._fcr_reinterval import classify, nominal_alpha
from increment.estimation.decision_types import (
    ArmHypothesisKey,
    DecisionComputation,
    DecisionFailure,
    SegmentHypothesisKey,
)
from increment.estimation.engine import _lift_decision_bundle, escalate_family_member
from increment.estimation.family import _require_q, family_discovery, select_family
from increment.estimation.multiplicity import design_excluded
from increment.estimation.readout_types import refuse_legacy_sampling
from increment.estimation.results import reinterval_selected, with_score_margin

if TYPE_CHECKING:
    from pydantic import BaseModel

    from increment.breakout.estimates import BreakoutEstimate
    from increment.decision import HypothesisKey, TestEvidence
    from increment.estimation._fcr_reinterval import Construction
    from increment.estimation.results import LiftEstimate


def _row_labels(rows: Sequence[str]) -> str:
    return ", ".join(rows)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.family.exploratory_row": RefusalSpec(
            "estimation.family.exploratory_row",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be whole-window LiftEstimate or segment "
                f"BreakoutEstimate rows; {_row_labels(rows)} are not (a day-axis or other row "
                "type is not a hypothesis of one whole-window family)"
            ),
        ),
        "estimation.family.exploratory_pre_corrected": RefusalSpec(
            "estimation.family.exploratory_pre_corrected",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be uncorrected: {_row_labels(rows)} already carry "
                "a family correction, and a cell is corrected by exactly one procedure"
            ),
        ),
        "estimation.family.exploratory_non_decision": RefusalSpec(
            "estimation.family.exploratory_non_decision",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be decision rows: {_row_labels(rows)} are "
                "sensitivity rows, which are companions of a decision row, not hypotheses"
            ),
        ),
        "estimation.family.exploratory_sequential": RefusalSpec(
            "estimation.family.exploratory_sequential",
            UnsupportedRequestError,
            lambda *, rows, **_: (
                "exploratory family selection is not available for sequential inference: "
                f"{_row_labels(rows)} carry always-valid or registered intervals, and e-BH over "
                "arbitrary segment cells has no validity argument yet. Pass fixed-horizon rows"
            ),
        ),
        "estimation.family.exploratory_construction": RefusalSpec(
            "estimation.family.exploratory_construction",
            UnsupportedRequestError,
            lambda *, rows, constructions, **_: (
                "exploratory family selection cannot reissue these intervals at the FCR level "
                "without approximating them: "
                + "; ".join(
                    f"{row} ({kind})" for row, kind in zip(rows, constructions, strict=True)
                )
                + ". Leave these rows out of the family"
            ),
        ),
    },
)

#: Fields a family correction writes; a row carrying any of them is already corrected.
_FAMILY_FIELDS = (
    "discovery",
    "family_axes",
    "family_q",
    "family_threshold",
    "family_procedure",
    "family_guarantee",
    "family_nominal_alpha",
    "family_size",
)
#: Reissued interval fields and their disclosures; everything else is the row's own. A
#: delta-route conversion row reissued below its dense level becomes an exact binomial row, so
_INTERVAL_FIELDS = (
    "lift",
    "binomial_set",
    "delta_evidence",
    "reference_kind",
    "reference_df",
    "dof",
    "relative_confidence_set",
    "abs_lb",
    "abs_ub",
    "abs_alpha",
    "abs_reference_kind",
    "abs_reference_df",
    "allocation_alpha",
    "note",
    "relative_error_code",
    "relative_error_context",
)
#: Each selected row is capped at its own nominal level below, so selection runs uncapped.
_UNCAPPED = math.nextafter(1.0, 0.0)
_ADMISSION_ORDER = (
    "estimation.family.exploratory_row",
    "estimation.family.exploratory_pre_corrected",
    "estimation.family.exploratory_non_decision",
    "estimation.family.exploratory_sequential",
)


#: A segment cell's scope: (dimension, segment value, fact source).
_Scope = tuple[str, str, str | None]


@dataclass(frozen=True, slots=True)
class _SourcedSegmentKey(SegmentHypothesisKey):
    """A segment hypothesis that also names the fact source it was read from."""

    source: str | None


@dataclass(frozen=True, slots=True)
class _Member:
    """One exploratory-family row with its hypothesis key and the estimate evidence reads."""

    row: LiftEstimate | BreakoutEstimate
    key: ArmHypothesisKey | _SourcedSegmentKey
    scope: _Scope | None
    view: LiftEstimate | None
    construction: Construction
    tested: bool = True


def _label(index: int, row: object) -> str:
    metric = getattr(row, "metric", None)
    if metric is None:
        return f"rows[{index}] ({type(row).__name__})"
    label = f"{metric}/{getattr(row, 'group_id', '?')}"
    dimension = getattr(row, "dimension", None)
    if dimension is None:
        return label
    source = getattr(row, "source", None)
    where = f"{dimension}={getattr(row, 'dimension_value', '?')}"
    return f"{label}[{where}{'' if source is None else f' @{source}'}]"


#: The stable reason each admission hazard reports, per row, in its refusal's ``reasons``.
_ADMISSION_REASONS = {
    "estimation.family.exploratory_row": "not a whole-window or segment estimate row",
    "estimation.family.exploratory_pre_corrected": "already family-corrected",
    "estimation.family.exploratory_non_decision": "sensitivity row, not a decision row",
    "estimation.family.exploratory_sequential": "sequential inference",
}


def _admission_hazard(row: object) -> str | None:
    """The first reason *row* cannot join an exploratory family, as its refusal code."""
    if not callable(getattr(row, "_family_view", None)) or getattr(row, "ds", None) is not None:
        return "estimation.family.exploratory_row"
    if any(getattr(row, name, None) is not None for name in _FAMILY_FIELDS):
        return "estimation.family.exploratory_pre_corrected"
    if getattr(row, "method_role", None) != "decision":
        return "estimation.family.exploratory_non_decision"
    if (
        getattr(row, "inference", "fixed") != "fixed"
        or getattr(row, "reference_kind", None) == "sequential"
        or getattr(row, "sequential_result", None) is not None
    ):
        return "estimation.family.exploratory_sequential"
    if getattr(row, "sampling_available", None) is not True and (
        getattr(row, "prior_shrunk", False) or getattr(row, "prior_spec", None) is not None
    ):
        refuse_legacy_sampling(row)
    return None


def _require_admissible(rows: Sequence[object]) -> None:
    """Refuse, naming every offending row, the first hazard that applies to any row."""
    offenders: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        code = _admission_hazard(row)
        if code is not None:
            offenders.setdefault(code, []).append(_label(index, row))
    for code in _ADMISSION_ORDER:
        if code in offenders:
            spec = _REFUSALS[code]
            labels = tuple(offenders[code])
            refuse(spec, rows=labels, reasons=(_ADMISSION_REASONS[code],) * len(labels))


def exploratory_family_exclusion(row: LiftEstimate | BreakoutEstimate) -> str | None:
    """Why *row* cannot join an exploratory family, or ``None`` when it can.

    This is the predicate ``select_exploratory_family`` applies, so a caller can leave an
    excluded row out of the family and report the reason instead of letting one hazardous cell
    refuse every cell. The string is the reason the refusal carries in its ``reasons`` context
    (``constructions`` for an interval that cannot be reissued). A cell excluded by design
    (too few units, no control arm, no treatment arm) is not excluded here: the family returns
    it unchanged.
    """
    code = _admission_hazard(row)
    if code is not None:
        return _ADMISSION_REASONS[code]
    view = row._family_view()
    if view is None:
        return None
    construction, reason = classify(view)
    return reason if construction is None else None


def _scope_of(row: object) -> _Scope | None:
    """A segment cell's scope: its dimension, segment value and fact source.

    The same segment value read from two fact sources is two cells, so the source is part of the
    identity. Whole-window rows have no scope.
    """
    dimension = getattr(row, "dimension", None)
    if dimension is None:
        return None
    return (dimension, cast("Any", row).dimension_value, getattr(row, "source", None))


def _members(rows: Sequence[LiftEstimate | BreakoutEstimate]) -> list[_Member]:
    """Key every row and classify how its interval is reissued, refusing what cannot be. A
    delta-route conversion row is read as a family member: its view carries the composite
    p-value of its persisted counts (`engine.escalate_family_member`), the evidence
    ``run_breakout(correction="bh")`` reads for the same cell. A binomial margin row saved with
    Wald additive bounds is read with its score interval (`results.with_score_margin`),
    the test its reissue inverts."""
    members: list[_Member] = []
    obstructed: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        scope = _scope_of(row)
        key = _in_scope(ArmHypothesisKey(row.metric, row.group_id, row.estimand), scope)
        view = row._family_view()
        if view is not None and row.method_role == "decision":
            view = with_score_margin(escalate_family_member(view))
        construction, reason = ("unavailable", "") if view is None else classify(view)
        if construction is None:
            obstructed.append((_label(index, row), reason))
            continue
        tested = view is not None or not design_excluded(row)
        members.append(_Member(row, key, scope, view, construction, tested))
    if obstructed:
        refuse(
            _REFUSALS["estimation.family.exploratory_construction"],
            rows=tuple(label for label, _ in obstructed),
            constructions=tuple(reason for _, reason in obstructed),
        )
    return members


def _in_scope(
    hypothesis: HypothesisKey, scope: _Scope | None
) -> ArmHypothesisKey | _SourcedSegmentKey:
    arm = cast("ArmHypothesisKey", hypothesis)
    if scope is None:
        return arm
    return _SourcedSegmentKey(arm.metric, arm.group_id, arm.estimand, *scope)


def _exploratory_computation(members: Sequence[_Member]) -> DecisionComputation[Any]:
    """Typed p-value evidence for every member, from the engine's own derivation.

    Each scope's rows (one segment of one fact source, or the whole window) go through
    ``_lift_decision_bundle`` together, as ``run_breakout`` does, and are re-keyed to that scope;
    an excluded cell becomes the failure that run keeps.
    """
    views: dict[_Scope | None, list[LiftEstimate]] = {}
    for member in members:
        if member.view is not None:
            views.setdefault(member.scope, []).append(member.view)
    evidence: dict[HypothesisKey, TestEvidence] = {}
    failures: dict[HypothesisKey, DecisionFailure] = {}
    for scope, scope_views in views.items():
        bundle = _lift_decision_bundle(scope_views, inference=None)
        for value in bundle.evidence.values():
            key = _in_scope(value.hypothesis, scope)
            evidence[key] = replace(value, hypothesis=key)
        for failure in bundle.failures.values():
            key = _in_scope(failure.hypothesis, scope)
            failures[key] = DecisionFailure(key, failure.code, failure.context)
    for member in members:
        if member.view is None:
            reason = getattr(member.row, "excluded", None)
            failures[member.key] = DecisionFailure(
                member.key,
                f"breakout.{reason}",
                {"metric": member.row.metric, "group_id": member.row.group_id, "reason": reason},
            )
    return DecisionComputation(results=(), evidence=evidence, failures=failures)


def _stamped[R: BaseModel](row: R, view: LiftEstimate | None, **family: object) -> R:
    """*row* with *view*'s reissued interval and the family record, revalidated as a whole."""
    fields = type(row).model_fields
    data: dict[str, Any] = {name: getattr(row, name) for name in fields}
    if view is not None:
        data.update({name: getattr(view, name) for name in _INTERVAL_FIELDS if name in fields})
        if "scale" in fields:
            data["scale"] = view.scale
    data.update(family)
    data["multiplicity_status"] = "exploratory_family"
    return type(row)(**data)


def _nominal_call_alpha(row: LiftEstimate | BreakoutEstimate) -> float | None:
    view = row._family_view()
    if view is None:
        return None
    construction, _ = classify(view)
    return None if construction is None else nominal_alpha(view, construction)


def select_exploratory_family(
    rows: Sequence[LiftEstimate | BreakoutEstimate], *, q: float
) -> tuple[LiftEstimate | BreakoutEstimate, ...]:
    """Correct whole-window and segment decision rows as one Benjamini-Hochberg family.

    ``rows`` are uncorrected ``LiftEstimate`` rows (one per metric and arm) and
    ``BreakoutEstimate`` rows (one per metric, arm and segment). Every decision row is one
    hypothesis of a single family at FDR level ``q``; the rows come back in the same order and
    types. Each carries ``discovery``, ``family_axes`` (``("metric", "arm")``, plus
    ``family_q``, ``family_threshold`` (the realized ``R*q/m``, ``None`` when nothing is selected),
    ``family_nominal_alpha`` (the original level for selected rows and their reissue cap), and
    ``family_size`` (``m``). Only selected rows are reissued and carry a nominal alpha.

    Selection is ``select_family`` over each member's family evidence -- the p-value the
    estimators themselves emit, which for a delta-route conversion row is the composite of its
    persisted counts rather than its delta p-value -- so discovery equals ``bh_select`` on that
    evidence, ties at the threshold included. A selected
    row's interval is reissued at the Benjamini-Yekutieli level ``1 - R*q/m``, capped at the
    row's own nominal level, from the construction the row persists: the Wald interval from its
    raw statistics and reference, the exact binomial set from its counts, or the Fieller set
    from its joint reference, each with its one-sided geometry. That equals the interval
    ``run_breakout(correction="bh")`` returns for the same cell. A delta-route conversion row
    is read as a family member: its persisted counts give the composite p-value the family
    selects on (read on demand here, carried on every tested row), and a selected one is
    reissued by those counts at the corrected level, the exact binomial set where they are not
    dense there -- again what ``run_breakout(correction="bh")`` reports. A row whose relative interval is
    unavailable (a non-positive arm mean) reports only its additive interval; an absolute margin
    can select it, and that interval is reissued from its persisted ``abs_alpha`` at the same
    level, never narrower than the nominal one. An exact binomial margin row saved with Wald
    additive bounds is read with the score interval its own level cuts, selected on that score
    test and returned with it. Any other unselected row keeps its interval.

    A cell excluded for an outcome-based reason (zero variance, non-positive mean, extreme
    ratio) stays in ``m`` as a non-rejection, as in ``run_breakout``. A cell excluded by design
    (too few units, no control arm, no treatment arm) is no hypothesis: it is returned
    unchanged, outside the family.

    The call refuses before any work, naming every offending row: rows that are not whole-window
    or segment rows (``estimation.family.exploratory_row``), rows already corrected
    (``estimation.family.exploratory_pre_corrected``), sensitivity rows
    (``estimation.family.exploratory_non_decision``), sequential rows
    (``estimation.family.exploratory_sequential``), and rows whose interval cannot be reissued
    without approximation (``estimation.family.exploratory_construction``: quantile, percentile
    winsorized, additive-scale and similar constructions, and an additive-only interval whose
    ``abs_alpha`` was not persisted). A cell whose evidence is unavailable for any other reason
    refuses the family as ``run_breakout`` does (``family.evidence.incomplete``). An empty input
    returns ``()``.
    """
    _require_q(q)
    rows = tuple(rows)
    _require_admissible(rows)
    members = _members(rows)
    tested = [member for member in members if member.tested]
    if not tested:
        return tuple(member.row for member in members)
    outcome = select_family(
        [(member.key, member.row) for member in tested],
        q,
        None,
        _UNCAPPED,
        computation=_exploratory_computation(tested),
    )

    axes = ("metric", "arm", "segment") if any(m.scope for m in tested) else ("metric", "arm")
    family = {
        "family_axes": axes,
        "family_procedure": "bh",
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
        "family_size": outcome.n_family,
    }
    corrected: list[LiftEstimate | BreakoutEstimate] = []
    for member in members:
        if not member.tested:
            corrected.append(member.row)
            continue
        selected = family_discovery(outcome, member.key)
        view = member.view
        if selected and view is not None:
            construction = member.construction
            threshold = outcome.realized_threshold
            assert construction is not None and threshold is not None
            alpha = min(threshold, nominal_alpha(view, construction))
            source_view = view
            view = reinterval_selected(view, construction, threshold)
            view = source_view._with_reissued_allocation(view, alpha)
        corrected.append(
            _stamped(
                member.row,
                view,
                discovery=selected,
                family_nominal_alpha=_nominal_call_alpha(member.row) if selected else None,
                **family,
            )
        )
    return tuple(corrected)
