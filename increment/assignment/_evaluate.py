"""Evaluate decisions and holdout membership from compiled allocation rules."""

from bisect import bisect_right
from dataclasses import dataclass, field

from ._config import (
    _Allocation,
    _Arm,
    _check_target,
    _compile,
    _Holdout,
    _invalid_target,
    _Layer,
    _Range,
    _unit_bytes,
)
from ._hash import _FULL, _draw
from ._types import Decision, Membership


def _start(interval: _Range | _Arm) -> int:
    return interval.start


@dataclass(frozen=True, slots=True)
class AllocationEvaluator:
    """Make repeatable decisions from the snapshot returned by `compile_allocation`."""

    _allocation: _Allocation = field(repr=False)

    def _membership(self, holdout: _Holdout, unit: bytes) -> Membership:
        held_out = _draw(holdout.prefix, unit) < holdout.held_out_bps
        return {
            "protocol": 1,
            "revision": self._allocation.revision,
            "holdout": holdout.key,
            "group": "held_out" if held_out else "eligible",
            "group_bps": holdout.held_out_bps if held_out else _FULL - holdout.held_out_bps,
        }

    def holdout_membership(self, holdout_key: str, unit_id: str) -> Membership:
        """Return the unit's holdout group without deciding on an experiment.

        Parameters
        ----------
        holdout_key : str
            Configured holdout to evaluate.
        unit_id : str
            Stable, nonempty identifier for the assignment unit.

        Returns
        -------
        Membership
            The group and its probability in basis points.
        """
        _check_target(holdout_key, "holdout_key")
        holdout = self._allocation.holdouts.get(holdout_key)
        if holdout is None:
            _invalid_target("holdout_key", "must name a configured holdout")
        return self._membership(holdout, _unit_bytes(unit_id))

    def decide(self, target: str, unit_id: str) -> Decision:
        """Evaluate holdout, layer, enrollment and arm selection, in that order.

        Parameters
        ----------
        target : str
            Layer or standalone experiment to evaluate.
        unit_id : str
            Stable, nonempty identifier for the assignment unit.

        Returns
        -------
        Decision
            The outcome, with probabilities for the stages it reached.
        """
        _check_target(target, "target")
        selected = self._allocation.targets.get(target)
        if selected is None:
            _invalid_target("target", "must name a layer or a standalone experiment")
        unit = _unit_bytes(unit_id)
        holdout = self._allocation.covered.get(target)
        membership = self._membership(holdout, unit) if holdout is not None else None
        decision: Decision = {
            "protocol": 1,
            "revision": self._allocation.revision,
            "target": target,
            "membership": membership,
            "experiment": None,
            "arm": None,
            "status": "unallocated",
            "layer_bps": None,
            "enrollment_bps": None,
            "arm_bps": None,
        }
        if membership is not None and membership["group"] == "held_out":
            decision["status"] = "held_out"
            return decision

        layer_bps = _FULL
        if isinstance(selected, _Layer):
            bucket = _draw(selected.prefix, unit)
            index = bisect_right(selected.ranges, bucket, key=_start) - 1
            if index < 0:
                return decision
            interval = selected.ranges[index]
            if bucket >= interval.end:
                return decision
            selected = interval.experiment
            layer_bps = interval.end - interval.start

        decision["experiment"] = selected.key
        decision["layer_bps"] = layer_bps
        decision["enrollment_bps"] = selected.enrollment_bps
        if _draw(selected.enroll_prefix, unit) >= selected.enrollment_bps:
            decision["status"] = "not_enrolled"
            return decision

        bucket = _draw(selected.arm_prefix, unit)
        arm = selected.arms[bisect_right(selected.arms, bucket, key=_start) - 1]
        decision["status"] = "assigned"
        decision["arm"] = arm.key
        decision["arm_bps"] = arm.weight_bps
        return decision


def compile_allocation(config: object) -> AllocationEvaluator:
    """Validate a decoded v1 configuration and snapshot its allocation rules.

    Parameters
    ----------
    config : object
        Decoded configuration with a protocol, revision, experiments and layers,
        and optionally holdouts.

    Returns
    -------
    AllocationEvaluator
        Evaluator unaffected by subsequent changes to the configuration.
    """
    return AllocationEvaluator(_compile(config))
