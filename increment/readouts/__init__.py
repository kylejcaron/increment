"""The readout set: every function takes a MomentSource and a Design and returns an estimate, without naming a data substrate.

Grain-specific readouts (daily, asof_lift) require the source to declare
the matching capability up front, so an unsupported request raises
CapabilityError naming the missing grain rather than failing deep inside
estimation.

breakout delegates randomized segments to
increment.breakout.estimates.run_breakout, excluding segments with no
usable control arm; encouragement estimation instead warns and skips
control-free segments and suppresses late rows after a weak first stage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from increment.kernel.context import EvidenceExecutionContext
from increment.readouts._asof import asof_lift
from increment.readouts._breakout import breakout
from increment.readouts._daily import daily
from increment.readouts._run import arm_moments, run
from increment.readouts._srm import srm


def run_scoped(
    analysis: Any,
    operation: Callable[[], Any] | None = None,
    *,
    coherence: Literal["captured", "live"] = "captured",
    include_breakouts: bool = False,
    on_context: Callable[[EvidenceExecutionContext], None] | None = None,
    **options: Any,
) -> Any:
    """Run related analysis readouts or operations under one shared evidence capture."""
    if "population" in options:
        from increment.readouts._line4 import run_scoped_line4

        return run_scoped_line4(
            analysis,
            operation,
            coherence=coherence,
            include_breakouts=include_breakouts,
            on_context=on_context,
            **options,
        )
    if operation is None:
        from increment.readouts._line2 import run_scoped_line2

        return run_scoped_line2(analysis, coherence=coherence, **options)
    from increment.readouts._line3 import run_scoped_line3

    return run_scoped_line3(
        analysis,
        operation,
        coherence=coherence,
        include_breakouts=include_breakouts,
        on_context=on_context,
    )


__all__ = [
    "arm_moments",
    "asof_lift",
    "breakout",
    "daily",
    "run",
    "run_scoped",
    "srm",
]
