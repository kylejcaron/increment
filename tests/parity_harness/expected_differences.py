"""Recorded differences between a frozen capture and the current route.

Each record names one scenario and ingress, the identity field that differs and
why. Every other value of the case is still compared. Both the frozen and the
current value are pinned, so a change on either side fails the comparison, and
a record whose difference no longer occurs fails until it is removed. Fields a
row gained after the capture are projected away only when their value follows
from the row's own captured evidence.
"""

from __future__ import annotations

import json
import math
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
    frozen: str | Mapping[str, Any] | None
    current: str | Mapping[str, Any] | None
    reason: str
    key: str | None = None

    def applied_to(self, expected: Any) -> Any:
        """The frozen outcome with every occurrence of the frozen value, as a value or a
        mapping key, replaced by the current one; a record naming a ``key`` replaces only
        values held under that mapping key. Refuses a record the outcome does not contain."""
        found = False

        def replace(value: Any, held_under: Any = None) -> Any:
            nonlocal found
            if (self.key is None or held_under == self.key) and value == self.frozen:
                found = True
                return self.current
            if isinstance(value, dict):
                return {replace(key): replace(item, key) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            return value

        migrated = replace(expected)
        assert found, f"{self.scenario}/{self.ingress} no longer records {self.field} {self.frozen}"
        return migrated


# The favorable side each declared direction names; any other declaration names none.
_DECLARED_SIDES = {"increase": "greater", "decrease": "less"}


def declared_directions(experiment: Any) -> dict[str, str | None]:
    """Each metric's declared ``preferred_direction``, None where ``experiment`` declares none;
    an experiment's metric payload holds the field only when it is declared."""
    return {
        name: dict(metric).get("preferred_direction") for name, metric in experiment.metrics.items()
    }


def with_declared_favorable_sides(expected: Any, declared: Mapping[str, str | None]) -> Any:
    """The frozen capture with each compiled plan it records given the favorable sides its
    metrics declare, which plans captured earlier left out.

    A plan is a mapping holding ``procedures`` and ``view_policies``, or that mapping as JSON
    under ``decision_plan``. An arm procedure gains "greater" when ``declared`` names its
    metric's ``preferred_direction`` "increase" and "less" for "decrease"; any other
    declaration adds none, so a side the current plan records for it fails the caller's
    comparison. Refuses a plan that already records a side or names a metric ``declared``
    lacks, so the rule fails once a capture records the sides itself.
    """

    def migrate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
        procedures = {}
        for metric, procedure in plan["procedures"].items():
            side = None
            if procedure.get("kind") in {"relative", "absolute"}:
                assert "favorable_side" not in procedure, f"{metric} already records its side"
                assert metric in declared, f"{metric} has no recorded declaration"
                side = _DECLARED_SIDES.get(declared[metric] or "")
            procedures[metric] = (
                procedure if side is None else {**procedure, "favorable_side": side}
            )
        return {**plan, "procedures": procedures}

    def migrate(value: Any, held_under: Any = None) -> Any:
        if held_under == "decision_plan" and isinstance(value, str):
            return json.dumps(
                migrate_plan(json.loads(value)), sort_keys=True, separators=(",", ":")
            )
        if isinstance(value, dict):
            if "procedures" in value and "view_policies" in value:
                return migrate_plan(value)
            return {key: migrate(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [migrate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(migrate(item) for item in value)
        return value

    return migrate(expected)


def without_stamped_sequential_direction(
    expected: Mapping[str, Any], actual: Mapping[str, Any], declared: Mapping[str, str | None]
) -> dict[str, Any]:
    """``actual`` without the ``preferred_direction`` a sequential row gained after its capture.

    Only a sequential row whose captured value is null is projected, and only when its value
    is its metric's declared direction in ``declared``; every other row is returned as is.
    """
    projected = dict(actual)
    if (
        actual.get("inference") in {"always_valid", "asymptotic_mean"}
        and expected.get("preferred_direction") is None
        and actual.get("preferred_direction") is not None
        and actual["preferred_direction"] == declared.get(actual["metric"])
    ):
        projected["preferred_direction"] = None
    return projected


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
    ExpectedDifference(
        scenario="sequential-mixed-quantile-catalog-checkpoint",
        ingress="from_unit_summary",
        field="retained_row.preferred_direction",
        key="preferred_direction",
        frozen=None,
        current="increase",
        reason=(
            "A sequential row carries its metric's declared preferred_direction, as a "
            "fixed-horizon row does; the captured row recorded none."
        ),
    ),
)
# Typed consumer inputs are frozen separately from the outcomes above.
_CONSUMER_INPUT_RECORDS = (
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_unit_summary",
        # Source `specs[].missing` and the consumer context's `metric_missingness` value.
        field="specs[].missing",
        frozen="drop",
        current="skip",
        reason="The source declares the observed-only policy, now spelled missing='skip'.",
    ),
    ExpectedDifference(
        scenario="audit-compliance-only-outcome-missingness",
        ingress="from_moments",
        # Source and consumer `context.metric_missingness`.
        field="context.metric_missingness",
        key="metric_missingness",
        frozen={},
        current={"revenue": "skip"},
        reason=(
            "A moments source records each metric declared missing='skip', which qualifies "
            "its rows' population; the captured source recorded none."
        ),
    ),
)
# A whole frozen disposition the current route replaces with another one.
_PANEL_ASOF_WITHOUT_EXPOSURE_DATE = {
    "kind": "refusal",
    "code": "evidence.grain_unavailable",
    "stage": "run",
    "context": {
        "constructor": "FramePanelSource",
        "offered": ["total"],
        "requested": "asof",
        "route": (
            "pass exposure_date=<column> naming each unit's exposure date; if every unit was "
            "exposed on the same day, add a column holding that date"
        ),
    },
}
# The failures a retired whole-run refusal listed, which the run now returns as unavailable rows.
_ALL_CELLS_FAILED = (
    (
        "edge-zero-variance-arm",
        {
            "code": "estimation.engine.lift_guard",
            "context": {
                "display": (
                    "Degenerate data: an arm has zero variance (e.g. all-identical outcomes or a "
                    "saturated conversion arm), cannot estimate lift uncertainty for that arm"
                ),
                "group_id": "treatment",
                "method": "unadjusted",
                "metric": "revenue",
                "reason": "zero_variance",
            },
            "estimand": "itt",
            "group_id": "treatment",
            "method": "unadjusted",
            "metric": "revenue",
        },
    ),
    (
        "edge-near-zero-ratio-denominator",
        {
            "code": "estimation.variance.ratio_moments_nonpositive_denominator_mean",
            "context": {"d_bar": 0.0, "group_id": "control", "method": "unadjusted", "metric": "r"},
            "estimand": "itt",
            "group_id": "treatment",
            "method": "unadjusted",
            "metric": "r",
        },
    ),
)
_DISPOSITION_RECORDS = tuple(
    ExpectedDifference(
        scenario=f"{view}-{base}",
        ingress="from_unit_panel",
        field="disposition",
        frozen=frozen,
        current=_PANEL_ASOF_WITHOUT_EXPOSURE_DATE,
        reason=(
            "The scenario's panel declares no exposure_date, so it offers only whole-window "
            "grains and refuses the as-of view before reading; the frozen outcome counted days "
            "from a common start the panel never declared."
        ),
    )
    for view, base, frozen in (
        ("asof", "breakout_store_dimension", {"kind": "cells"}),
        ("asof", "breakout_boolean_store_dimension", {"kind": "cells"}),
        ("asof", "conversion_route_breakout_segments", {"kind": "cells"}),
        (
            "asof_lift",
            "breakout_store_dimension",
            {
                "kind": "refusal",
                "code": "estimation.cuped.arm_no_covariate",
                "stage": "run",
                "context": {"arm_group": "control", "arm_metric": "revenue_cuped"},
            },
        ),
        (
            "asof_lift",
            "breakout_boolean_store_dimension",
            {
                "kind": "refusal",
                "code": "estimation.cuped.arm_no_covariate",
                "stage": "run",
                "context": {"arm_group": "control", "arm_metric": "revenue_cuped"},
            },
        ),
        ("asof_lift", "conversion_route_breakout_segments", {"kind": "cells"}),
    )
) + tuple(
    ExpectedDifference(
        scenario=scenario,
        ingress="from_unit_summary",
        field="disposition",
        frozen={
            "kind": "refusal",
            "code": "readout.estimate_lift_every",
            "stage": "run",
            "context": {"failures": [failure]},
        },
        current={"kind": "unavailable", "stage": "run", "cells": [failure]},
        reason=(
            "The run's only cell failed; the retired `readout.estimate_lift_every` refusal gives "
            "way to that cell's unavailable row, carrying the failure it listed."
        ),
    )
    for scenario, failure in _ALL_CELLS_FAILED
)


def _by_case(
    records: tuple[ExpectedDifference, ...],
) -> dict[tuple[str, str], tuple[ExpectedDifference, ...]]:
    return {
        pair: tuple(record for record in records if (record.scenario, record.ingress) == pair)
        for pair in dict.fromkeys((record.scenario, record.ingress) for record in records)
    }


EXPECTED_DIFFERENCES = _by_case(_RECORDS)
CONSUMER_INPUT_DIFFERENCES = _by_case(_CONSUMER_INPUT_RECORDS)
DISPOSITION_DIFFERENCES = _by_case(_DISPOSITION_RECORDS)


def with_expected_difference(
    scenario: str,
    ingress: str,
    expected: Any,
    *,
    differences: Mapping[tuple[str, str], tuple[ExpectedDifference, ...]] = EXPECTED_DIFFERENCES,
) -> Any:
    """The frozen capture of one case, migrated by each of its recorded differences."""
    for difference in differences.get((scenario, ingress), ()):
        expected = difference.applied_to(expected)
    return expected


def _binomial_arm_facts(counts: BinomialCounts) -> dict[str, int | float]:
    x_c, n_c, x_t, n_t = counts
    return {"n_control": n_c, "n_treat": n_t, "mean_control": x_c / n_c, "mean_treat": x_t / n_t}


def _without_gained_unit_arm_facts(
    expected: Mapping[str, Any], actual: Mapping[str, Any], projected: dict[str, Any]
) -> dict[str, Any]:
    """``projected`` without the arm unit counts an unclustered ratio or a quantile row gained,
    and the arm ratios a ratio row gained. Their captured evidence holds neither, so the counts
    must be positive integers and the ratios finite."""
    metric_type = actual.get("metric_type")
    if metric_type == "quantile":
        gained: tuple[str, ...] = ("n_control", "n_treat")
    elif metric_type == "ratio" and actual.get("n_clusters") is None:
        gained = ("n_control", "n_treat", "mean_control", "mean_treat")
    else:
        return projected
    for field in gained:
        value = actual.get(field)
        if value is None or expected.get(field) is not None:
            continue
        if field in ("n_control", "n_treat"):
            # A frame holds counts in a float column; a typed row holds integers.
            assert type(value) in (int, float) and value >= 1 and float(value).is_integer(), (
                field,
                value,
            )
        else:
            assert type(value) is float and math.isfinite(value), (field, value)
        if field in expected:
            projected[field] = None
        else:
            del projected[field]
    return projected


def without_gained_arm_facts(
    expected: Any,
    actual: Any,
    counts: Mapping[tuple[str, str], BinomialCounts] | None = None,
) -> Any:
    """``actual`` without the arm counts and values rows gained after a capture.

    Only a field the captured row lacks or holds as null is projected. On an exact binomial row
    (``reference_kind == "binomial"``) the counts must equal ``n_c`` and ``n_t`` exactly, as
    integers on a typed row; the rates must equal ``x_c / n_c`` and ``x_t / n_t`` within the
    harness tolerance. Both come from the row's own ``binomial_set`` or, for a flattened row
    without one, from ``counts`` keyed by ``(metric, group_id)`` as ``(x_c, n_c, x_t, n_t)``.
    Ratio and quantile rows follow ``_without_gained_unit_arm_facts``. Every other field is
    left to the caller's comparison. Lists and tuples pair by position, so a captured model's
    ``(type name, fields)`` form is projected too.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_gained_arm_facts(expected.get(key), value, counts)
            for key, value in actual.items()
        }
        if actual.get("reference_kind") != "binomial":
            return _without_gained_unit_arm_facts(expected, actual, projected)
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
            without_gained_arm_facts(left, right, counts)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


def _registered_family_cell(row: Mapping[str, Any]) -> bool:
    """Whether ``row``'s own sequential evidence registers its cell in a family."""
    result = row.get("sequential_result")
    checkpoint = result.get("checkpoint") if isinstance(result, Mapping) else None
    cell = checkpoint.get("cell") if isinstance(checkpoint, Mapping) else None
    return isinstance(cell, Mapping) and cell.get("family") is True


def without_post_capture_family_size(expected: Any, actual: Any) -> Any:
    """``actual`` without the ``family_size`` a family row gained after its capture.

    Two kinds of row gained it: a fixed-horizon BH segment row, and a row whose own
    ``sequential_result`` registers its cell in a sequential family. Only such a row with a
    positive integer ``family_size`` is projected, and only when its captured row lacks the
    field or holds it as null. Lists and tuples pair by position, as in
    ``without_gained_arm_facts``.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        projected = {
            key: without_post_capture_family_size(expected.get(key), value)
            for key, value in actual.items()
        }
        size = actual.get("family_size")
        gained = (
            actual.get("family_procedure") == "bh" and actual.get("dimension_value") is not None
        ) or _registered_family_cell(actual)
        if gained and type(size) is int and size >= 1 and expected.get("family_size") is None:
            if "family_size" in expected:
                projected["family_size"] = None
            else:
                del projected["family_size"]
        return projected
    if (
        isinstance(expected, (list, tuple))
        and isinstance(actual, (list, tuple))
        and len(expected) == len(actual)
    ):
        paired = [
            without_post_capture_family_size(left, right)
            for left, right in zip(expected, actual, strict=True)
        ]
        return tuple(paired) if isinstance(actual, tuple) else paired
    return actual


def without_metric_outcome_columns(expected: Any, actual: Any) -> Any:
    """``actual`` consumer inputs without the ``metric_outcome_columns`` each source context
    gained after ``expected`` was captured.

    Only a context whose capture lacks the field is projected, and only when its value follows
    from the captured source inputs: a declared dataframe's metric specs name each metric's value
    column (the numerator for a ratio) and any denominator; every other source names none.
    """
    derived: dict[str, set[str]] = {}
    for source_input in expected["source_inputs"]:
        if source_input["kind"] != "declared_source_frame":
            continue
        for spec in source_input["configuration"]["source_declaration"]["specs"]:
            value = spec["numerator"] if spec["type"] == "ratio" else spec["value_column"]
            derived[spec["name"]] = {value or spec["name"]} | (
                {spec["denominator"]} if spec["denominator"] else set()
            )

    def project(captured: Any, live: Any) -> Any:
        if isinstance(captured, Mapping) and isinstance(live, Mapping):
            fields = dict(live)
            if (
                "metric_missingness" in captured
                and "metric_outcome_columns" not in captured
                and "metric_outcome_columns" in fields
            ):
                columns = fields.pop("metric_outcome_columns")
                assert {name: set(value) for name, value in columns.items()} == derived, columns
            return {key: project(captured.get(key), value) for key, value in fields.items()}
        if isinstance(captured, list) and isinstance(live, list) and len(captured) == len(live):
            return [project(left, right) for left, right in zip(captured, live, strict=True)]
        return live

    return project(expected, actual)
