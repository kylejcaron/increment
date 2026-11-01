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

from increment.readouts._asof import asof_lift
from increment.readouts._breakout import breakout
from increment.readouts._daily import daily
from increment.readouts._run import arm_moments, run
from increment.readouts._srm import srm

__all__ = [
    "arm_moments",
    "asof_lift",
    "breakout",
    "daily",
    "run",
    "srm",
]
