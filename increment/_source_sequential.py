"""Source-bound sequential capture and checkpoint retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from increment._binding import binding_for
from increment._evidence_types import SequentialCaptureRequest

if TYPE_CHECKING:
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver
    from increment.sequential_state import SequentialSnapshot


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
    from increment.readouts._evidence import retained_sequential_capture
    from increment.readouts._execution_context import current_execution_context

    request = SequentialCaptureRequest(
        finalized=finalized, previous=previous, as_of=as_of, freeze=tuple(freeze)
    )
    return retained_sequential_capture(
        binding_for(experiment, resolver).source, request, current_execution_context()
    )


def sequential_snapshot(
    experiment: Experiment, resolver: Resolver, *, previous=None
) -> SequentialSnapshot:
    """Retrieve the source's current exact checkpoint, optionally proving its parent."""
    from increment.readouts._evidence import retained_sequential_snapshot
    from increment.readouts._execution_context import current_execution_context

    return retained_sequential_snapshot(
        binding_for(experiment, resolver).source, previous, current_execution_context()
    )
