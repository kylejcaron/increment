"""Sentinel labels for units outside the primary assignment population.

Leaf module: no imports. `increment.sources` re-exports both names.
"""

from __future__ import annotations

# Key unit_counts() uses for units excluded via on_unassigned="exclude";
# SRM diagnostics report it without counting it as a chi-square degree of freedom.
UNASSIGNED_LABEL = "(unassigned)"

# Key for units seen in more than one group, dropped by first_exposures
# before counting; like UNASSIGNED_LABEL, never an arm or a degree of freedom.
MIXED_ASSIGNMENT_LABEL = "(mixed assignment)"
