"""Validate declared source operations before handing out source capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._evidence_types import SourceContext
from increment.errors import refuse
from increment.sources import require_operation

if TYPE_CHECKING:
    from increment._binding import SourceBinding
    from increment.semantics.models import Experiment
    from increment.sources import SourceOperation, _BuiltinMomentSource


def _require_arm_binding(binding: SourceBinding, method: str) -> SourceBinding:
    if binding.state.family != "arm_moments":
        from increment._evidence_refusals import CONTRAST_UNAVAILABLE

        refuse(CONTRAST_UNAVAILABLE, method=method)
    return binding


def _arm_experiment(binding: SourceBinding) -> Experiment | None:
    """The arm binding's semantic experiment, or None when its source carries none."""
    _require_arm_binding(binding, "experiment")
    try:
        return binding.experiment
    except AttributeError:
        return None


def _require_analysis_operation[T](
    source: _BuiltinMomentSource,
    operation: SourceOperation,
    protocol: type[T] | None = None,
    *,
    message: str,
) -> T:
    declaration = cast(Any, source).evidence
    if (
        operation == "allocation_history"
        and "allocation_history" not in declaration.operations
        and declaration.assignment_relation == "reduced"
    ):
        from increment._evidence_refusals import refuse_assignment_identity

        refuse_assignment_identity(source, operation=operation)
    offered = tuple(sorted(declaration.operations))
    if not isinstance(getattr(source, "context", None), SourceContext):
        from increment._evidence_refusals import refuse_operation

        refuse_operation(source, operation, offered, route=message)
    if operation not in offered:
        from increment._evidence_refusals import refuse_operation

        refuse_operation(source, operation, offered, route=message)
    return require_operation(source, operation, protocol)
