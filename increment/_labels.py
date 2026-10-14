"""Sentinel labels for units outside the primary assignment population.

Leaf module: no imports. `increment.sources` re-exports both names.
"""

from __future__ import annotations

# Key unit_counts() uses for units excluded via on_unassigned="exclude";
# SRM diagnostics report it without counting it as a chi-square degree of freedom.
UNASSIGNED_LABEL = "(unassigned)"


def canonical_group_label(value: object, *, boolean: bool) -> str | None:
    """Render a nonmissing group identity consistently across dataframes."""
    if value is None:
        return None
    if boolean and isinstance(value, bool):
        return "true" if value else "false"
    label = str(value)
    return label.lower() if boolean else label


# Key for units seen in more than one group, dropped by first_exposures
# before counting; like UNASSIGNED_LABEL, never an arm or a degree of freedom.
MIXED_ASSIGNMENT_LABEL = "(mixed assignment)"
