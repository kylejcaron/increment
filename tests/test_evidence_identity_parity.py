from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from increment.result_envelope import ResultCell
from tests.parity_harness.comparison import TOLERANCE

# Cutover name → explicitly registered (scenario_id, ingress) pairs.
REGISTERED: Mapping[str, frozenset[tuple[str, str]]] = {}
NEW_PATH_ADAPTERS: Mapping[tuple[str, str], Callable[[str, str], Mapping[str, object]]] = {}
SEEDED_RESAMPLING_FIELD_PATHS = frozenset(
    {
        "retained_row.confidence_set.raw.bootstrap_replicates",
        "retained_row.confidence_set.raw.permutation_p_value",
    }
)
_SKIP_COUNT = 0


@cache
def _capture() -> dict:
    from scripts.capture_evidence_identity import load_operations_capture

    path = (
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    return load_operations_capture(path)


def test_capture_reuses_decoded_fixture() -> None:
    assert _capture() is _capture()


def _comparable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(payload))
    copied.pop("provenance", None)
    return copied


def _json_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_context(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_context(item) for item in value]
    return value


def _compare(expected: Any, actual: Any, path: str = "") -> None:
    if expected is None or actual is None or isinstance(expected, (bool, str)):
        assert expected == actual
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        assert (expected == 0.0) == (actual == 0.0)
        assert math.copysign(1, expected) == math.copysign(1, actual)
        if expected != 0.0:
            rtol = 1e-6 if path in SEEDED_RESAMPLING_FIELD_PATHS else 1e-9
            assert math.isclose(expected, actual, rel_tol=rtol, abs_tol=TOLERANCE)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and expected.keys() == actual.keys()
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}" if path else str(key))
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(expected) == len(actual)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}.{index}")
    else:
        assert expected == actual


def test_compare_normalizes_tuple_and_list_containers_recursively() -> None:
    expected = {"payload": [{"intervals": [], "context": {"offsets": [1, 2]}}]}
    actual = {"payload": ({"intervals": (), "context": {"offsets": (1, 2)}},)}
    _compare(expected, actual)


def _assert_payload_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    canonical_expected = ResultCell.from_payload(expected).to_payload()
    _compare(_comparable_payload(canonical_expected), _comparable_payload(actual))


def _assert_operation_matches(
    expected: Mapping[str, Any], operation: Callable[[], ResultCell]
) -> None:
    from increment.errors import CodedError

    try:
        actual = operation()
    except CodedError as error:
        assert expected["status"] == "refusal"
        assert expected["code"] == error.code
        _compare(expected["context"], dict(error.context))
    else:
        assert expected["status"] == "available"
        expected_cell = expected["result_cell"]
        assert isinstance(expected_cell, Mapping)
        _assert_payload_matches(expected_cell, actual.to_payload())


def _assert_cell_matches(expected: Mapping[str, Any], actual: object) -> None:
    from increment.result_envelope import relevel

    expected_cell = expected["result_cell"]
    assert isinstance(expected_cell, Mapping)
    actual_payload = actual.to_payload() if isinstance(actual, ResultCell) else actual
    assert isinstance(actual_payload, Mapping)
    actual_payload = cast(Mapping[str, Any], actual_payload)
    _assert_payload_matches(expected_cell, actual_payload)
    actual_cell = (
        actual if isinstance(actual, ResultCell) else ResultCell.from_payload(actual_payload)
    )
    expected_operations = expected["operations"]
    assert isinstance(expected_operations, Mapping)
    for alpha, operation in expected_operations["relevel"].items():
        assert isinstance(operation, Mapping)
        _assert_operation_matches(
            operation,
            lambda alpha=float(alpha): relevel(actual_cell, alpha=alpha),
        )


def _assert_outcome_matches(expected: Mapping[str, Any], actual: Mapping[str, object]) -> None:
    assert expected["kind"] == actual["kind"]
    if expected["kind"] == "refusal":
        assert expected["code"] == actual["code"]
        assert expected["stage"] == actual["stage"]
        assert _json_context(expected["context"]) == _json_context(actual["context"])
        if "aggregate_evidence" in expected:
            assert expected["aggregate_evidence"] == actual["aggregate_evidence"]
    elif expected["kind"] == "absent":
        assert expected["reason"] == actual["reason"]
    elif expected["kind"] == "cells":
        expected_cells = expected["cells"]
        actual_cells = actual["cells"]
        assert isinstance(expected_cells, Mapping)
        assert isinstance(actual_cells, Mapping)
        assert expected_cells.keys() == actual_cells.keys()
        for key, expected_cell in expected_cells.items():
            assert isinstance(expected_cell, Mapping)
            _assert_cell_matches(expected_cell, actual_cells[key])
        _compare(expected["metadata"], actual["metadata"])
        for field in (
            "population_identity",
            "daily_values",
            "family_selection",
            "aggregate_evidence",
        ):
            if field in expected:
                if field == "aggregate_evidence":
                    assert expected[field] == actual[field]
                else:
                    _compare(expected[field], actual[field])


def _offset_payload(cell: Mapping[str, Any], fraction: float) -> dict[str, Any]:
    payload = copy.deepcopy(dict(cell))
    summary = payload["summary"]
    point = summary["point"]
    assert isinstance(point, (int, float)) and point != 0
    summary["point"] = point * (1.0 + fraction)
    return payload


def test_capture_scenario_inventory_matches_fixture() -> None:
    from scripts.capture_evidence_identity import iter_capture_keys

    expected = set(iter_capture_keys())
    record = _capture()
    actual = {
        (scenario_id, ingress)
        for scenario_id, scenario in record["scenarios"].items()
        for ingress in scenario
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("scenario_id", "ingress", "expected"),
    sorted(
        (scenario_id, ingress, outcome)
        for scenario_id, scenario in _capture()["scenarios"].items()
        for ingress, outcome in scenario.items()
    ),
)
def test_frozen_capture_outcome(
    scenario_id: str, ingress: str, expected: Mapping[str, Any]
) -> None:
    global _SKIP_COUNT
    key = (scenario_id, ingress)
    if key not in REGISTERED.get("pre-kernel", frozenset()):
        _SKIP_COUNT += 1
        if os.getenv("INCREMENT_REQUIRE_ALL_REGISTERED") == "1":
            pytest.fail(f"new path not registered for {scenario_id}/{ingress}")
        if summary := os.getenv("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a") as stream:
                stream.write(f"Skipped frozen capture outcomes: {_SKIP_COUNT}\n")
        pytest.skip(f"new path not registered for {scenario_id}/{ingress}")
    adapter = NEW_PATH_ADAPTERS.get(key)
    if adapter is None:
        pytest.fail(f"registered scenario/ingress has no adapter: {scenario_id}/{ingress}")
    try:
        actual = adapter(scenario_id, ingress)
    except pytest.skip.Exception as error:
        pytest.fail(
            f"registered adapter skipped for {scenario_id}/{ingress}: {error}",
            pytrace=False,
        )
    _assert_outcome_matches(expected, actual)


def test_frozen_capture_replays_every_outcome_against_itself() -> None:
    for scenario in _capture()["scenarios"].values():
        for expected in scenario.values():
            actual = _actual_from_capture(expected) if expected["kind"] == "cells" else expected
            _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_frozen_capture_rejects_missing_or_extra_cell(mutation: str) -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "cells" and outcome["cells"]
    )
    _assert_unchanged_capture_matches(expected)
    actual = {
        "kind": "cells",
        "cells": dict(expected["cells"]),
        "metadata": expected["metadata"],
    }
    if mutation == "missing":
        actual["cells"].pop(next(iter(actual["cells"])))
    else:
        actual["cells"]["unexpected-identity"] = next(iter(actual["cells"].values()))
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_frozen_capture_rejects_ingress_metadata_mismatch() -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "cells"
    )
    _assert_unchanged_capture_matches(expected)
    assert isinstance(expected["metadata"], Mapping)
    actual: dict[str, object] = {
        "kind": "cells",
        "cells": {key: cell["result_cell"] for key, cell in expected["cells"].items()},
        "metadata": {**expected["metadata"], "unexpected": True},
    }
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize("field", ("kind", "code", "stage"))
def test_frozen_capture_rejects_refusal_mismatch(field: str) -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "refusal"
    )
    _assert_unchanged_capture_matches(expected)
    actual = dict(expected)
    actual[field] = {
        "kind": "absent",
        "code": "different-refusal",
        "stage": "construct" if expected["stage"] == "run" else "run",
    }[field]
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize(
    "context",
    (
        {"code": "nested.failure", "value": 2},
        {"code": "nested.different", "value": 1},
    ),
)
def test_frozen_capture_rejects_refusal_context_mismatch(context: dict[str, object]) -> None:
    expected = {
        "kind": "refusal",
        "code": "outer.refusal",
        "stage": "run",
        "context": {"code": "nested.failure", "value": 1},
    }
    _assert_outcome_matches(expected, expected)
    actual = {**expected, "context": context}
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_frozen_capture_rejects_absence_reason_mismatch() -> None:
    expected = {"kind": "absent", "reason": "constructor not declared"}
    _assert_outcome_matches(expected, expected)
    actual = {"kind": "absent", "reason": "different absence reason"}
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_registered_control_group_refusal_matches_frozen_tuple_context() -> None:
    from increment.errors import CodedError
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS

    expected = _capture()["scenarios"]["estimation.engine.control_group_found"]["refusal"]
    scenario = cast(
        Callable[[], object], REFUSAL_SCENARIOS["estimation.engine.control_group_found"]
    )
    with pytest.raises(CodedError) as exc_info:
        scenario()
    error = exc_info.value
    assert any(isinstance(value, tuple) for value in error.context.values())
    actual = {
        "kind": "refusal",
        "code": error.code,
        "stage": "run",
        "context": dict(error.context),
    }
    _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize("fraction", [0.25, 1e-7])
def test_frozen_capture_rejects_constant_relative_offset(fraction: float) -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "cells"
        and any(
            isinstance(item.get("result_cell"), Mapping)
            and isinstance(item["result_cell"].get("summary"), Mapping)
            and isinstance(item["result_cell"]["summary"].get("point"), (int, float))
            and item["result_cell"]["summary"]["point"] != 0
            for item in outcome["cells"].values()
        )
    )
    _assert_unchanged_capture_matches(expected)
    actual = _actual_from_capture(expected)
    actual_cells = actual["cells"]
    assert isinstance(actual_cells, dict)
    actual_cells = cast(dict[str, object], actual_cells)
    offset_key = next(
        key
        for key, cell in expected["cells"].items()
        if isinstance(cell.get("result_cell"), Mapping)
        and isinstance(cell["result_cell"].get("summary"), Mapping)
        and isinstance(cell["result_cell"]["summary"].get("point"), (int, float))
        and cell["result_cell"]["summary"]["point"] != 0
    )
    payload = expected["cells"][offset_key]["result_cell"]
    actual_cells[offset_key] = _offset_payload(payload, fraction)
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize(
    ("expected", "actual"),
    ((0.0, -5e-10), (0.0, 5e-10), (1e-12, -1e-12)),
)
def test_frozen_comparison_enforces_zero_and_sign_before_tolerance(
    expected: float, actual: float
) -> None:
    with pytest.raises(AssertionError):
        _compare(expected, actual)


def test_registered_adapter_skip_fails(monkeypatch) -> None:
    import tests.test_evidence_identity_parity as parity

    scenario_id, scenario = next(iter(_capture()["scenarios"].items()))
    ingress, expected = next(iter(scenario.items()))
    key = (scenario_id, ingress)
    monkeypatch.setattr(parity, "REGISTERED", {"pre-kernel": frozenset({key})})
    monkeypatch.setattr(
        parity,
        "NEW_PATH_ADAPTERS",
        {key: lambda *_args: pytest.skip("adapter skipped")},
    )

    try:
        test_frozen_capture_outcome(scenario_id, ingress, expected)
    except pytest.skip.Exception as error:
        pytest.fail(f"registered adapter skip escaped: {error}")
    except pytest.fail.Exception:
        pass
    else:
        pytest.fail("registered adapter skip did not fail the comparison")


def _actual_from_capture(expected: Mapping[str, Any]) -> dict[str, object]:
    return {
        "kind": expected["kind"],
        "cells": {
            key: ResultCell.from_payload(cell["result_cell"])
            for key, cell in expected.get("cells", {}).items()
        },
        "metadata": expected.get("metadata"),
        "population_identity": expected.get("population_identity"),
        "daily_values": expected.get("daily_values"),
        "family_selection": expected.get("family_selection"),
        "aggregate_evidence": expected.get("aggregate_evidence"),
    }


def _assert_unchanged_capture_matches(expected: Mapping[str, Any]) -> None:
    actual = _actual_from_capture(expected) if expected["kind"] == "cells" else expected
    _assert_outcome_matches(expected, actual)


@pytest.mark.parametrize("field", ("population_identity", "daily_values", "family_selection"))
def test_frozen_capture_rejects_ingress_evidence_mismatch(field: str) -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "cells"
    )
    _assert_unchanged_capture_matches(expected)
    actual = _actual_from_capture(expected)
    actual[field] = {"unexpected": True}

    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_frozen_capture_rejects_operation_mismatch() -> None:
    capture = _capture()
    scenario_id, ingress, expected = next(
        (scenario_id, ingress, outcome)
        for scenario_id, scenario in capture["scenarios"].items()
        for ingress, outcome in scenario.items()
        if outcome["kind"] == "cells"
        and any(
            operation["status"] == "available"
            for cell in outcome["cells"].values()
            for operations in cell["operations"].values()
            for operation in operations.values()
        )
    )
    _assert_unchanged_capture_matches(expected)
    expected = copy.deepcopy(expected)
    key, cell = next(
        (key, cell)
        for key, cell in expected["cells"].items()
        if any(
            operation["status"] == "available"
            for operations in cell["operations"].values()
            for operation in operations.values()
        )
    )
    operation = cell["operations"]["relevel"]["0.01"]
    baseline_point = capture["scenarios"][scenario_id][ingress]["cells"][key]["operations"][
        "relevel"
    ]["0.01"]["result_cell"]["summary"]["point"]
    operation["result_cell"]["summary"]["point"] += 1.0
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, _actual_from_capture(expected))
    assert (
        capture["scenarios"][scenario_id][ingress]["cells"][key]["operations"]["relevel"]["0.01"][
            "result_cell"
        ]["summary"]["point"]
        == baseline_point
    )


def test_frozen_capture_rejects_aggregate_moment_envelope_mismatch() -> None:
    expected = {
        "kind": "cells",
        "cells": {},
        "metadata": None,
        "population_identity": {"status": "unavailable"},
        "daily_values": [],
        "family_selection": {"status": "unavailable"},
        "aggregate_evidence": {
            "status": "available",
            "moments_format": 13,
            "moments": [{"source_identity": "digest-source", "moment": 1.0}],
        },
    }
    _assert_outcome_matches(expected, expected)
    actual = dict(expected)
    actual["aggregate_evidence"] = {
        "status": "available",
        "moments_format": 13,
        "moments": [{"source_identity": "digest-source", "moment": 2.0}],
    }

    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_aggregate_moment_inputs_are_compared_exactly() -> None:
    expected = {
        "kind": "cells",
        "cells": {},
        "metadata": None,
        "aggregate_evidence": {
            "status": "available",
            "moments_format": 13,
            "moments": [{"cy1": 0.123456789}],
        },
    }
    _assert_outcome_matches(expected, expected)
    actual = {
        **expected,
        "aggregate_evidence": {
            "status": "available",
            "moments_format": 13,
            "moments": [{"cy1": 0.1234567891}],
        },
    }

    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)
