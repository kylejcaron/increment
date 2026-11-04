"""Recorded differences between a frozen capture and the current route.

Each record names one scenario and ingress, the identity field that differs and
why. Every other value of the case is still compared. Both the frozen and the
current value are pinned, so a change on either side fails the comparison, and
a record whose difference no longer occurs fails until it is removed. Fields a
row gained after the capture are projected away only when their value follows
from the row's own captured evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tests.parity_harness.comparison import nested_close

BinomialCounts = tuple[int, int, int, int]


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


_RECORDS = (
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
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_panel",
        field="refusal code",
        frozen="frame.missing_policy.panel_drop",
        current="frame.missing_policy.panel_skip",
        reason=(
            "The panel refusal of an observed-only outcome policy is renamed with the policy; "
            "the retired code maps to the current one in `RETIRED_CODES`."
        ),
    ),
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_panel",
        field="refusal context missing",
        frozen="drop",
        current="skip",
        reason="The refusal names the declared policy, now spelled missing='skip'.",
    ),
)
EXPECTED_DIFFERENCES = {
    pair: tuple(record for record in _RECORDS if (record.scenario, record.ingress) == pair)
    for pair in dict.fromkeys((record.scenario, record.ingress) for record in _RECORDS)
}


def with_expected_difference(scenario: str, ingress: str, expected: Any) -> Any:
    """The frozen outcome of one case, migrated by each of its recorded differences."""
    for difference in EXPECTED_DIFFERENCES.get((scenario, ingress), ()):
        expected = difference.applied_to(expected)
    return expected


def _binomial_arm_facts(counts: BinomialCounts) -> dict[str, int | float]:
    x_c, n_c, x_t, n_t = counts
    return {"n_control": n_c, "n_treat": n_t, "mean_control": x_c / n_c, "mean_treat": x_t / n_t}


def without_binomial_arm_facts(
    expected: Any,
    actual: Any,
    counts: Mapping[tuple[str, str], BinomialCounts] | None = None,
) -> Any:
    """``actual`` without the arm counts and rates exact binomial rows gained after a capture.

    Only a ``reference_kind == "binomial"`` row is projected, and only a field its captured
    row lacks or holds as null. The counts must equal ``n_c`` and ``n_t`` exactly, as integers
    on a typed row; the rates must equal ``x_c / n_c`` and ``x_t / n_t`` within the harness
    tolerance. Both come from the row's own ``binomial_set`` or, for a flattened row without
    one, from ``counts`` keyed by ``(metric, group_id)`` as ``(x_c, n_c, x_t, n_t)``. Every other
    field is left to the caller's comparison. Lists and tuples pair by position, so a captured
    model's ``(type name, fields)`` form is projected too.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_binomial_arm_facts(expected.get(key), value, counts)
            for key, value in actual.items()
        }
        if actual.get("reference_kind") != "binomial":
            return projected
        exact = actual.get("binomial_set")
        if isinstance(exact, tuple) and len(exact) == 2:
            exact = exact[1]
        if isinstance(exact, Mapping):
            implied = _binomial_arm_facts((exact["x_c"], exact["n_c"], exact["x_t"], exact["n_t"]))
        elif counts is not None:
            implied = _binomial_arm_facts(counts[(actual["metric"], actual["group_id"])])
        else:
            return projected
        for field, value in implied.items():
            if actual.get(field) is None or expected.get(field) is not None:
                continue
            if field in ("n_control", "n_treat"):
                # A frame holds counts in a float column; a typed row holds integers.
                count_types = (int,) if isinstance(exact, Mapping) else (int, float)
                assert type(actual[field]) in count_types and actual[field] == value, (
                    field,
                    value,
                    actual[field],
                )
            else:
                assert nested_close(value, actual[field]), (field, value, actual[field])
            if field in expected:
                projected[field] = None
            else:
                del projected[field]
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_binomial_arm_facts(left, right, counts)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual
