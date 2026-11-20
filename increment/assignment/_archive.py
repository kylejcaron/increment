"""Internal access to identities and probabilities in a compiled archive."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, TypeGuard

from ._config import _is_scalar, _Layer
from ._evaluate import AllocationEvaluator
from ._hash import _FULL


@dataclass(frozen=True, slots=True)
class ArchivedFactor:
    """Compiled rules for one experiment on a specific decision target."""

    experiment: str
    revision: str
    layer: str | None
    arms: tuple[tuple[str, int], ...]
    arm_draw_id: str
    can_assign: bool
    always_assign: bool
    holdout_domain: bytes | None
    held_out_bps: int


def has_archived_target(target: object, evaluator: AllocationEvaluator) -> TypeGuard[str]:
    """Check whether a persisted target identifies a known decision rule."""
    return _is_scalar(target) and target in evaluator._allocation.targets


def describe_archived_factor(
    evaluator: AllocationEvaluator, target: str, experiment: str
) -> ArchivedFactor | Literal["unknown_target", "not_in_target"]:
    """Describe a focal experiment without exposing the allocator's internal rules."""
    allocation = evaluator._allocation
    configured = allocation.targets.get(target) if _is_scalar(target) else None
    if configured is None:
        return "unknown_target"
    if not _is_scalar(experiment):
        return "not_in_target"
    width = _FULL
    layer = None
    if isinstance(configured, _Layer):
        interval = next(
            (row for row in configured.ranges if row.experiment.key == experiment), None
        )
        if interval is None:
            return "not_in_target"
        layer = target
        width = interval.end - interval.start
        configured = interval.experiment
    elif configured.key != experiment:
        return "not_in_target"
    holdout = allocation.covered.get(target)
    held_out_bps = 0 if holdout is None else holdout.held_out_bps
    return ArchivedFactor(
        experiment=configured.key,
        revision=allocation.revision,
        layer=layer,
        arms=tuple((arm.key, arm.weight_bps) for arm in configured.arms),
        arm_draw_id=sha256(configured.arm_prefix).hexdigest(),
        can_assign=width > 0 and configured.enrollment_bps > 0 and held_out_bps < _FULL,
        always_assign=width == _FULL and configured.enrollment_bps == _FULL and held_out_bps == 0,
        holdout_domain=None if holdout is None else holdout.prefix,
        held_out_bps=held_out_bps,
    )
