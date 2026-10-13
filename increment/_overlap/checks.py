"""Ingress-neutral validation counts and the one place that turns them into refusals.

Each ingress computes a :class:`Violations` record with its own engine; the
refusal order and codes live here, so the same hazard raises the same code on
every path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from increment._overlap.design import OverlapDesign
from increment._overlap.refusals import _raise


@dataclass(frozen=True, slots=True)
class ExperimentViolations:
    eligibility_missing: int = 0
    unknown_arm: int = 0
    unknown_arm_examples: tuple[str, ...] = ()
    eligible_unassigned: int = 0
    ineligible_assigned: int = 0
    assigned_at_missing: int = 0


@dataclass(frozen=True, slots=True)
class Violations:
    missing_units: int = 0
    duplicate_units: int = 0
    duplicate_examples: tuple[str, ...] = ()
    first: ExperimentViolations = field(default_factory=ExperimentViolations)
    second: ExperimentViolations = field(default_factory=ExperimentViolations)


@dataclass(frozen=True, slots=True)
class TargetViolations:
    window_missing: int = 0
    window_order: int = 0
    outcome_precedes_assignment: int = 0
    window_lengths: tuple[str, ...] = ()
    distinct_window_lengths: int = 1
    metric_missing: tuple[tuple[str, int], ...] = ()
    metric_nonfinite: tuple[tuple[str, int], ...] = ()


def refuse_unit_violations(design: OverlapDesign, found: Violations) -> None:
    if found.missing_units:
        _raise("overlap.ingress.unit_missing", count=found.missing_units)
    if found.duplicate_units:
        _raise(
            "overlap.ingress.duplicate_units",
            count=found.duplicate_units,
            examples=found.duplicate_examples,
        )
    for experiment, v in ((design.first, found.first), (design.second, found.second)):
        name = experiment.name
        if v.eligibility_missing:
            _raise(
                "overlap.ingress.eligibility_missing", experiment=name, count=v.eligibility_missing
            )
        if v.ineligible_assigned:
            _raise(
                "overlap.ingress.ineligible_assigned", experiment=name, count=v.ineligible_assigned
            )
        if v.eligible_unassigned:
            _raise(
                "overlap.ingress.eligible_unassigned", experiment=name, count=v.eligible_unassigned
            )
        if v.unknown_arm:
            _raise(
                "overlap.ingress.unknown_arm",
                experiment=name,
                count=v.unknown_arm,
                control=experiment.control,
                treatment=experiment.treatment,
                examples=v.unknown_arm_examples,
            )
        if v.assigned_at_missing:
            _raise(
                "overlap.ingress.assignment_time_missing",
                experiment=name,
                count=v.assigned_at_missing,
            )


def refuse_empty_target(design: OverlapDesign, target_units: int) -> None:
    if target_units == 0:
        _raise("overlap.target.empty", first=design.first.name, second=design.second.name)


def refuse_target_violations(found: TargetViolations) -> None:
    if found.window_missing:
        _raise("overlap.timing.window_missing", count=found.window_missing)
    if found.window_order:
        _raise("overlap.timing.window_order", count=found.window_order)
    if found.outcome_precedes_assignment:
        _raise(
            "overlap.timing.outcome_precedes_assignment", count=found.outcome_precedes_assignment
        )
    if found.distinct_window_lengths > 1:
        _raise(
            "overlap.timing.horizon_unaligned",
            count=found.distinct_window_lengths,
            examples=found.window_lengths,
        )
    for metric, count in found.metric_missing:
        if count:
            _raise("overlap.outcome.missing", metric=metric, count=count)
    for metric, count in found.metric_nonfinite:
        if count:
            _raise("overlap.outcome.nonfinite", metric=metric, count=count)


def example_labels(values: Iterable[object], limit: int = 3) -> tuple[str, ...]:
    """A deterministic, bounded sample of offending values for a refusal message."""
    return tuple(sorted({str(v) for v in values})[:limit])
