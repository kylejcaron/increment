from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from increment.result_envelope import ResultCell
from tests.analysis_factory import held_source
from tests.parity_harness.comparison import TOLERANCE

# Line-one owns whole-window parity pairs not assigned to lines 2–7.
_LINE1_SCENARIOS = frozenset(
    {
        "edge-empty-arm",
        "edge-zero-variance-arm",
        "edge-near-zero-ratio-denominator",
        "edge-near-zero-positive-ratio-denominator",
        "edge-nonfinite-ratio-denominator",
        "arm-facts-parity",
        "categorical-cuped-moments-format-13",
        "bare_metric_name_summary_panel",
        "cuped_covariate_name_summary_panel",
        "exact_counts_conversion_window_boundary_export_from_definitions",
        "exact_counts_conversion_window_boundary_export_from_unit_panel",
        "exact_counts_retention_cohort_boundary_export_from_definitions",
        "exact_counts_retention_cohort_boundary_export_from_unit_panel",
        "audit-panel-missing-zero-inferred",
        "mean_ratio_conversion_cuped_fixed_horizon",
        "artifact_context_format_two",
        "artifact_context_format_one_refusal",
        "dashboard-group-data",
        "audit-prior-reset-family",
        "portable_subset_with_winsorized_sibling",
        "audit-artifact-rounded-aggregates",
        "audit-cuped-covariate-rescaling",
        "audit-ratio-neighboring-means",
        "lift_prior_normal_mean_ratio_conversion_cuped",
        "lift_prior_student_t_mean_ratio_conversion_cuped",
        "lift_prior_mixture_mean_ratio_conversion_cuped",
        "exploratory-metrics",
        "window_spelling_zulu_vs_offset",
        "window_spelling_offset_vs_offset",
        "audit-panel-component-horizon",
        "audit-pinned-dimension-freshness",
        "audit-panel-component-horizon-ratio",
        "audit-panel-component-horizon-ratio-complete",
        "multiplicity_primary_secondaries_guardrail",
        "nonpositive_mean_additive_row_matches_across_sources",
        "clustered_negative_control_mean_produces_a_fieller_set",
        "clustered_assignment_integrity_scope",
        "clustered_zero_relative_variance_preserves_point_and_additive_interval",
        "audit-cluster-export-refusal",
        "conversion_route_straddles_the_dense_threshold",
        "audit-uptake-timestamp-boundaries",
        "audit-compliance-only-outcome-missingness",
        "audit-compliance-only-empty-catalog",
        "quantile_tied_rounded_outcomes",
        "audit-quantile-family",
        "audit-panel-pre-exposure-and-selection",
        "winsor-percentile-three-arm",
        "winsor-percentile-zero-inclusive-2-arm",
        "winsor-percentile-size-routed-analytic",
        "mixed-cuped-multicovariate-fixed-horizon",
    }
)
_LINE1_INGRESSES = (
    "from_definitions",
    "from_moments",
    "from_unit_panel",
    "from_unit_summary",
)
_LINE1_PAIRS = frozenset(
    (scenario_id, ingress) for scenario_id in _LINE1_SCENARIOS for ingress in _LINE1_INGRESSES
)
REGISTERED: Mapping[str, frozenset[tuple[str, str]]] = {"line1": _LINE1_PAIRS}
NEW_PATH_ADAPTERS: Mapping[tuple[str, str], Callable[[str, str], Mapping[str, object]]] = {}
SEEDED_RESAMPLING_FIELD_PATHS = frozenset(
    {
        "retained_row.confidence_set.raw.bootstrap_replicates",
        "retained_row.confidence_set.raw.permutation_p_value",
    }
)
_MODE_METADATA_FIELDS = frozenset({"coherence", "effective_modes"})


def _mode_metadata(metadata: object) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    metadata_by_key = cast(Mapping[str, object], metadata)
    return {key: metadata_by_key[key] for key in _MODE_METADATA_FIELDS if key in metadata_by_key}


def _metadata_without_modes(metadata: object) -> object:
    if not isinstance(metadata, Mapping):
        return metadata
    return {key: value for key, value in metadata.items() if key not in _MODE_METADATA_FIELDS}


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


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


# Deliberate changes made after the pre-kernel capture, keyed by component and the exact
# captured status reason. #270 added likelihood sequential results, so a cell with neither
# sequential kind now states that no sequential result of any kind is retained.
_POST_CAPTURE_STATUS_REASONS: Mapping[tuple[str, str], str] = {
    ("sequential", "no asymptotic sequential result is retained"): (
        "no sequential result is retained"
    ),
}


def _with_post_capture_reasons(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Express captured reasons in current wording; cells restored from stored payloads
    retain the captured wording, so both sides of a comparison are normalized."""
    migrated = copy.deepcopy(dict(payload))
    for name, component in migrated.get("components", {}).items():
        status = component["status"]
        replacement = _POST_CAPTURE_STATUS_REASONS.get((name, status.get("reason")))
        if replacement is not None and component.get("kind") is None:
            status["reason"] = replacement
    return migrated


def _assert_payload_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    canonical_expected = ResultCell.from_payload(expected).to_payload()
    _compare(
        _comparable_payload(_with_post_capture_reasons(canonical_expected)),
        _comparable_payload(_with_post_capture_reasons(actual)),
    )


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


def _assert_outcome_matches(
    expected: Mapping[str, Any], actual: Mapping[str, object]
) -> Mapping[str, object]:
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
        _compare(
            _metadata_without_modes(expected["metadata"]),
            _metadata_without_modes(actual["metadata"]),
        )
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
        return _mode_metadata(actual["metadata"])
    return {}


def _line1_read(analysis: Any, case: Any) -> Any:
    metrics = None if case.metrics is None else list(case.metrics)
    return analysis.run(
        estimands=case.estimands,
        prior=case.prior,
        metrics=metrics,
        decision_method=case.decision_method,
    )


def _enter_patch(stack: Any, target: Any, name: str, replacement: Any) -> None:
    from unittest.mock import patch

    stack.enter_context(patch.object(target, name, replacement))


def _run_with_read_counter(analysis: Any, case: Any) -> tuple[Any, Mapping[str, int]]:
    from collections import Counter
    from contextlib import ExitStack

    source = held_source(analysis)
    source_type = type(source)
    names = (
        ("unit_frame",)
        if callable(getattr(source, "unit_frame", None))
        else (
            ("readout_snapshot",)
            if callable(getattr(source, "readout_snapshot", None))
            else tuple(
                name
                for name in ("moments", "moments_for_method")
                if callable(getattr(source, name, None))
            )
        )
    )
    counts: Counter[str] = Counter()
    depth = 0
    with ExitStack() as stack:
        for name in names:
            original = getattr(source_type, name)

            def counted(this, *args, _original=original, _name=name, **kwargs):
                nonlocal depth
                if depth == 0:
                    metrics = kwargs.get("metrics") if _name == "readout_snapshot" else None
                    if metrics is not None:
                        for metric in metrics:
                            counts[str(getattr(metric, "name", metric))] += 1
                    else:
                        metric = args[0] if args else kwargs.get("metric")
                        key = getattr(metric, "name", str(metric)) if metric is not None else _name
                        counts[str(key)] += 1
                depth += 1
                try:
                    return _original(this, *args, **kwargs)
                finally:
                    depth -= 1

            _enter_patch(stack, source_type, name, counted)
        return _line1_read(analysis, case), counts


def _line1_actual_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    """The current outcome of a registered pair under the capture's recorded conditions.

    The frozen capture ran every DuckDB connection with ``threads=1``
    (``scripts/capture_evidence_identity.capture`` wraps ``_capture_operations`` in
    ``_single_thread_duckdb``); the actual side reuses that same helper so a
    comparison never mixes multi-threaded reduction order into the oracle.
    """
    from scripts.capture_evidence_identity import _single_thread_duckdb

    with _single_thread_duckdb():
        return _line1_outcome(scenario_id, ingress)


def _line1_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:  # noqa: PLR0915
    import json
    import warnings
    from contextlib import ExitStack
    from datetime import UTC, datetime

    from increment import Analysis
    from increment.errors import CodedError, InvalidRequestError
    from increment.result_envelope import AnalysisEnvelope
    from scripts.capture_evidence_identity import (
        _aggregate_evidence,
        _population_identity,
    )
    from tests.parity_harness.cases import PARITY_CASES, _close_parity_analysis

    cases = {case.id: case for case in PARITY_CASES}
    case = cases[scenario_id]
    if ingress not in case.build:
        return {
            "kind": "absent",
            "reason": case.waive.get(ingress, "constructor ingress is not declared"),
        }

    aggregate_rows: list[dict[str, object]] = []
    payloads: dict[str, object] = {}
    try:
        if ingress == "from_moments":
            original_from_moments = Analysis.from_moments

            def capture_moments(cls, rows, **kwargs):
                aggregate_rows.extend(copy.deepcopy(dict(row)) for row in rows)
                return original_from_moments(rows, **kwargs)

            with ExitStack() as stack:
                _enter_patch(stack, Analysis, "from_moments", classmethod(capture_moments))
                analysis = case.build[ingress]()
        else:
            analysis = case.build[ingress]()
    except CodedError as error:
        outcome = {
            "kind": "refusal",
            "code": error.code,
            "stage": "construct",
            "context": dict(error.context),
        }
        aggregate = _aggregate_evidence(payloads, aggregate_rows)
        if aggregate is not None:
            payload_ref = cast(Mapping[str, Any], aggregate["moments"])
            aggregate["moments"] = payloads[payload_ref["payload_hash"]]
            outcome["aggregate_evidence"] = aggregate
        return outcome

    try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, read_count = _run_with_read_counter(analysis, case)
            assert all(count == 1 for count in read_count.values()), (
                f"{scenario_id}/{ingress} performed repeated captured reads: {read_count}"
            )
            metadata_value = getattr(results, "metadata", None)
            cells = []
            for row in results:
                cells.extend(
                    row.as_cells(
                        metadata_value,
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    )
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                population_identity = _population_identity(analysis, case, cells)
        except CodedError as error:
            outcome: dict[str, object] = {
                "kind": "refusal",
                "code": error.code,
                "stage": "run",
                "context": dict(error.context),
            }
            aggregate = _aggregate_evidence(payloads, aggregate_rows)
            if aggregate is not None:
                payload_ref = cast(Mapping[str, Any], aggregate["moments"])
                aggregate["moments"] = payloads[payload_ref["payload_hash"]]
                outcome["aggregate_evidence"] = aggregate
            return outcome

        family_selection: dict[str, object] = {"status": "unavailable"}
        if cells:
            try:
                from increment.estimation.family import select_exploratory_family

                selected = select_exploratory_family(list(results), q=0.05)
                family_selection = {
                    "status": "available",
                    "q": 0.05,
                    "results": json.loads(
                        json.dumps([row.model_dump(mode="json") for row in selected])
                    ),
                }
            except (CodedError, InvalidRequestError) as error:
                family_selection = {
                    "status": "refusal",
                    "code": error.code,
                    "context": dict(error.context),
                }
        family_selection = cast(dict[str, object], _json_compatible(family_selection))
        cells_by_hash: dict[str, object] = {}
        for cell in cells:
            cell_hash = cell.identity.hash()
            assert cell_hash not in cells_by_hash
            cells_by_hash[cell_hash] = cell
        outcome = {
            "kind": "cells",
            "cells": cells_by_hash,
            "daily_values": [],
            "family_selection": family_selection,
            "metadata": AnalysisEnvelope(tuple(cells), metadata_value).to_payload()["metadata"],
            "population_identity": population_identity,
        }
        aggregate = _aggregate_evidence(payloads, aggregate_rows)
        if aggregate is not None:
            payload_ref = cast(Mapping[str, Any], aggregate["moments"])
            aggregate["moments"] = payloads[payload_ref["payload_hash"]]
            outcome["aggregate_evidence"] = aggregate
        return outcome
    finally:
        _close_parity_analysis(analysis)


NEW_PATH_ADAPTERS = dict.fromkeys(_LINE1_PAIRS, _line1_actual_outcome)


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


def test_line1_registration_inventory_matches_frozen_capture() -> None:
    captured = _capture()["scenarios"]
    fixture_pairs = {
        (scenario_id, ingress) for scenario_id, outcomes in captured.items() for ingress in outcomes
    }
    assert _LINE1_PAIRS <= fixture_pairs
    assert set(NEW_PATH_ADAPTERS) == _LINE1_PAIRS
    counts = {
        kind: sum(
            captured[scenario_id][ingress]["kind"] == kind for scenario_id, ingress in _LINE1_PAIRS
        )
        for kind in ("cells", "refusal", "absent")
    }
    assert counts == {"cells": 144, "refusal": 19, "absent": 37}


@pytest.mark.integration
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
    registered = {pair for pairs in REGISTERED.values() for pair in pairs}
    if key not in registered:
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
    modes = _assert_outcome_matches(expected, actual)
    if key in _LINE1_PAIRS and expected["kind"] == "cells":
        assert modes["coherence"] == "captured"
        effective_modes = modes["effective_modes"]
        assert isinstance(effective_modes, Mapping)
        assert effective_modes
        assert set(effective_modes.values()) == {"captured"}


@pytest.mark.integration
def test_readout_snapshot_identity_is_hash_seed_independent() -> None:
    script = """
import json
from tests.test_evidence_identity_parity import _line1_actual_outcome
scenarios = (
    ("categorical-cuped-moments-format-13", "from_moments"),
    ("mixed-cuped-multicovariate-fixed-horizon", "from_moments"),
    ("dashboard-group-data", "from_unit_summary"),
)
for scenario, ingress in scenarios:
    outcome = _line1_actual_outcome(scenario, ingress)
    print(json.dumps({
        "scenario": scenario,
        "snapshot_id": outcome["metadata"]["scope"]["snapshot_id"],
    }, sort_keys=True))
"""
    snapshots = []
    for seed in ("0", "4"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        snapshots.append(result.stdout)
    assert snapshots[0] == snapshots[1]


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


def test_frozen_capture_rejects_captured_refusal_context_mutation() -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "refusal"
    )
    _assert_unchanged_capture_matches(expected)
    actual = copy.deepcopy(expected)
    actual["context"] = {**expected["context"], "unexpected": "changed"}
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_frozen_capture_rejects_absence_reason_mismatch() -> None:
    expected = {"kind": "absent", "reason": "constructor not declared"}
    _assert_outcome_matches(expected, expected)
    actual = {"kind": "absent", "reason": "different absence reason"}
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, actual)


def test_frozen_capture_rejects_captured_absence_reason_mutation() -> None:
    expected = next(
        outcome
        for scenario in _capture()["scenarios"].values()
        for outcome in scenario.values()
        if outcome["kind"] == "absent"
    )
    _assert_unchanged_capture_matches(expected)
    actual = dict(expected)
    actual["reason"] = "different absence reason"
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
    monkeypatch.setattr(parity, "REGISTERED", {"line1": frozenset({key})})
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
