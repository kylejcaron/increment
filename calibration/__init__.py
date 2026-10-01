"""Calibration certification for increment's inference procedures.

This package lives outside ``testpaths`` on purpose: it is a certification
campaign, not a unit-test suite, and a bare ``pytest`` run must never collect
it. Invoke it as ``python -m calibration``.
"""

from __future__ import annotations

from calibration.journal import DrawJournal, JournalError, JournalTotals, verify
from calibration.profile import (
    CalibrationProfile,
    Campaign,
    CellSet,
    ProfileError,
    available,
    campaigns,
    load,
)
from calibration.stopping import (
    SEQUENTIAL_RULES,
    FixedDesign,
    FixedStopping,
    SequentialStopping,
    StoppingError,
    StoppingRule,
    build_rule,
)

__all__ = [
    "SEQUENTIAL_RULES",
    "CalibrationProfile",
    "Campaign",
    "CellSet",
    "DrawJournal",
    "FixedDesign",
    "FixedStopping",
    "JournalError",
    "JournalTotals",
    "ProfileError",
    "SequentialStopping",
    "StoppingError",
    "StoppingRule",
    "available",
    "build_rule",
    "campaigns",
    "load",
    "verify",
]
