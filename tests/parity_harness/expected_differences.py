"""Recorded differences between a frozen capture and the current route.

Each record names one scenario and ingress, the identity field that differs and
why. Every other value of the case is still compared. Both the frozen and the
current value are pinned, so a change on either side fails the comparison, and
a record whose difference no longer occurs fails until it is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExpectedDifference:
    scenario: str
    ingress: str
    field: str
    frozen: str
    current: str
    reason: str

    def applied_to(self, expected: Any) -> Any:
        """The frozen outcome with every occurrence of the frozen value, as a value or a
        mapping key, replaced by the current one. Refuses a record the outcome does not
        contain."""
        found = False

        def replace(value: Any) -> Any:
            nonlocal found
            if isinstance(value, dict):
                return {replace(key): replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            if value == self.frozen:
                found = True
                return self.current
            return value

        migrated = replace(expected)
        assert found, f"{self.scenario}/{self.ingress} no longer records {self.field} {self.frozen}"
        return migrated


EXPECTED_DIFFERENCES = {
    (difference.scenario, difference.ingress): difference
    for difference in (
        ExpectedDifference(
            scenario="triggered_encouragement_compliance_asof",
            ingress="from_definitions",
            # Cell and row `source_snapshot_id`, `scope.snapshot_id` and the `scope.by_source` key.
            field="source_snapshot_id",
            frozen="sha256:250f3b61e63d68f59821d42f3f56bbdd3969d660903abb8e649ddbf6c3541bce",
            current="sha256:555fa9fd774dee87a087655dc870ca4bf2123da6d3298bb69e83ba7c9ad9c782",
            reason=(
                "A compliance-only view run through `run` requests no outcome metrics, so its "
                "snapshot request differs from the removed "
                "`run_asof_lift(estimands=('compliance',))`, which requested the declared "
                "outcome metrics; the estimates are unchanged."
            ),
        ),
    )
}


def with_expected_difference(scenario: str, ingress: str, expected: Any) -> Any:
    """The frozen outcome of one case, migrated by its recorded difference when it has one."""
    difference = EXPECTED_DIFFERENCES.get((scenario, ingress))
    return expected if difference is None else difference.applied_to(expected)
