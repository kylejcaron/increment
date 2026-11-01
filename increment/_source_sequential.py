"""Source-bound sequential capture and checkpoint retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from increment._binding import SourceBinding, binding_for
from increment._evidence_types import SequentialCaptureRequest

if TYPE_CHECKING:
    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver
    from increment.sequential_state import SequentialSnapshot


def _read_retained(
    source: Any,
    execution_context: EvidenceExecutionContext | None,
    read: Callable[[Any], SequentialSnapshot],
) -> SequentialSnapshot:
    """Read the source directly, or through a run's borrowed scope and retain the result."""
    if execution_context is None:
        return read(source)
    from increment._source_types import is_custom_source
    from increment.readouts._evidence import _builtin_source

    execution_context.validate_request(
        source, tuple(source.context.metrics), tuple(source.context.configs)
    )
    with execution_context.borrow() as scoped:
        snapshot = read(scoped if is_custom_source(source) else _builtin_source(scoped))
    if is_custom_source(source):
        return snapshot
    from increment.sequential_source import adopt_source_snapshot

    return adopt_source_snapshot(source, snapshot)


def _capture_sequential_binding(
    binding: SourceBinding,
    request: SequentialCaptureRequest,
    execution_context: EvidenceExecutionContext | None = None,
) -> SequentialSnapshot:
    from increment.readouts._evidence import load_sequential_capture

    return _read_retained(
        binding.source,
        execution_context,
        lambda source: load_sequential_capture(source, request),
    )


def _sequential_snapshot_binding(
    binding: SourceBinding,
    previous: SequentialSnapshot | None = None,
    execution_context: EvidenceExecutionContext | None = None,
) -> SequentialSnapshot:
    from increment.readouts._evidence import current_source_snapshot

    return _read_retained(
        binding.source,
        execution_context,
        lambda source: current_source_snapshot(source, previous=previous),
    )


def capture_sequential(
    experiment: Experiment,
    resolver: Resolver,
    *,
    finalized: bool,
    previous=None,
    as_of=None,
    freeze: Sequence[str] = (),
) -> SequentialSnapshot:
    """Capture a registered finalized prefix; freshness is not finalization.

    Pass a previous snapshot to verify append-only continuation. Corrections
    to any retained unit outcome, assignment or definition refuse the path.
    Panel and relational sources require an explicit ``as_of`` horizon. The
    capture is retained by the resolver's source, so a later
    ``run(experiment, resolver)`` reads it.

    ``freeze`` names metrics to stop monitoring at exactly this capture:
    every registered cell of each named metric (each arm and segment) that
    has data keeps this capture's evidence at every later look; see
    ``declare_sequential_freeze``. A frozen secondary still takes part in
    its family, whose e-BH selection is redone at each look over frozen and
    current evidence, so its discovery and selected interval can change.

    A triggered family guarantee requires stopping and freezing decisions
    based only on already-revealed triggered-chain information. Assigned
    results can reveal future triggered outcomes, including the same
    metric, and must not inform those decisions.
    """
    from increment.readouts._execution_context import current_execution_context

    request = SequentialCaptureRequest(
        finalized=finalized, previous=previous, as_of=as_of, freeze=tuple(freeze)
    )
    return _capture_sequential_binding(
        binding_for(experiment, resolver), request, current_execution_context()
    )


def sequential_snapshot(
    experiment: Experiment, resolver: Resolver, *, previous=None
) -> SequentialSnapshot:
    """Retrieve the source's current exact checkpoint, optionally proving its parent."""
    from increment.readouts._execution_context import current_execution_context

    return _sequential_snapshot_binding(
        binding_for(experiment, resolver), previous, current_execution_context()
    )
