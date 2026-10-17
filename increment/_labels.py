"""Shared group-label utilities and sentinels."""

from __future__ import annotations

from collections.abc import Sequence

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


def boolean_group_count_aliases(
    group_ids: Sequence[object], count_labels: Sequence[object]
) -> dict[str, str]:
    """Alias count spellings only when their Boolean identity is unambiguous."""
    literal_group_labels = {label for label in group_ids if isinstance(label, str)}
    count_label_set = {label for label in count_labels if isinstance(label, str)}
    boolean_values = {group_id for group_id in group_ids if isinstance(group_id, bool)}
    aliases: dict[str, str] = {}
    for value in boolean_values:
        spelling = str(value)
        canonical = "true" if value else "false"
        if (
            spelling in count_label_set
            and spelling not in literal_group_labels
            and canonical not in count_label_set
        ):
            aliases[spelling] = canonical
    return aliases


# Key for units seen in more than one group, dropped by first_exposures
# before counting; like UNASSIGNED_LABEL, never an arm or a degree of freedom.
MIXED_ASSIGNMENT_LABEL = "(mixed assignment)"
