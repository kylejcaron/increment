from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from increment.result_envelope import ResultCell
from tests.analysis_factory import held_source
from tests.parity_harness import registrations_line5 as _line5
from tests.parity_harness.comparison import TOLERANCE
from tests.parity_harness.expected_differences import with_expected_difference
from tests.parity_harness.registrations_line2 import (
    LINE2_ASOF_PAIRS,
    LINE2_PAIRS,
    line2_actual_outcome,
)
from tests.parity_harness.registrations_line3 import LINE3_PAIRS, line3_actual_outcome
from tests.parity_harness.registrations_line4 import merge_line4
from tests.parity_harness.registrations_line6 import ADAPTERS as _LINE6_ADAPTERS
from tests.parity_harness.registrations_line6 import PAIRS as _LINE6_PAIRS
from tests.parity_harness.registrations_line7 import ADAPTERS as _LINE7_ADAPTERS
from tests.parity_harness.registrations_line7 import PAIRS as _LINE7_PAIRS

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

    root = Path(__file__).parent / "fixtures" / "evidence_identity"
    capture = load_operations_capture(root / "pre-kernel" / "operations.json.gz")
    asof = load_operations_capture(root / "pre-kernel-asof" / "operations.json.gz")
    capture["scenarios"].update(asof["scenarios"])
    return capture


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


def _compare_aggregate(expected: Any, actual: Any) -> None:
    """Compare aggregate moment inputs at a practical floating-point tolerance.

    Reduction order and platform math libraries move low bits; a centered sum whose true
    value is zero lands anywhere within accumulated rounding. Values within ``TOLERANCE``
    of zero are equal, and others must agree to ``rel_tol=1e-9``.
    """
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(expected, (int, float)) and not isinstance(expected, bool)
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        if abs(expected) <= TOLERANCE and abs(actual) <= TOLERANCE:
            return
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=TOLERANCE)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and expected.keys() == actual.keys()
        for key in expected:
            _compare_aggregate(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _compare_aggregate(left, right)
    else:
        assert expected == actual


def test_compare_normalizes_tuple_and_list_containers_recursively() -> None:
    expected = {"payload": [{"intervals": [], "context": {"offsets": [1, 2]}}]}
    actual = {"payload": ({"intervals": (), "context": {"offsets": (1, 2)}},)}
    _compare(expected, actual)


def test_qr_snapshot_normalization_sorts_families_and_compares_fit_numbers() -> None:
    expected = {
        "families": [
            {"name": "primary", "family_id": "expected-a", "source_snapshot_id": "expected"},
            {"name": "guardrail", "family_id": "expected-b", "source_snapshot_id": "expected"},
        ],
        "ordered_values": [1, 2],
        "aggregate_evidence": {
            "moments": [
                {
                    "metric": "revenue",
                    "cuped_evidence": json.dumps({"kind": "fit", "theta": 1.0}),
                }
            ]
        },
    }
    actual = {
        "families": [
            {"name": "guardrail", "family_id": "actual-b", "source_snapshot_id": "actual"},
            {"name": "primary", "family_id": "actual-a", "source_snapshot_id": "actual"},
        ],
        "ordered_values": [1, 2],
        "aggregate_evidence": {
            "moments": [
                {
                    "metric": "revenue",
                    "cuped_evidence": json.dumps({"kind": "fit", "theta": 1.0 + 1e-10}),
                }
            ]
        },
    }
    expected_normalized = _decode_fitted_cuped_evidence(_without_snapshot_identity(expected))
    actual_normalized = _decode_fitted_cuped_evidence(_without_snapshot_identity(actual))

    _compare(expected_normalized, actual_normalized)
    assert expected_normalized["ordered_values"] == actual_normalized["ordered_values"]


# Normalize origin/main's exact post-capture unavailable reason from #261, and #270's
# likelihood-sequential status only when the current counterpart retains that result.
_POST_CAPTURE_STATUS_REASONS: Mapping[
    tuple[str, str, str | None, str, str | None],
    tuple[str, str | None, str, str | None],
] = {
    (
        "sequential",
        "unavailable",
        "results.component.unavailable",
        "no asymptotic sequential result is retained",
        None,
    ): (
        "unavailable",
        "results.component.unavailable",
        "no sequential result is retained",
        None,
    ),
    (
        "sequential",
        "unavailable",
        "results.component.unavailable",
        "no sequential result is retained",
        None,
    ): (
        "available",
        None,
        "likelihood sequential geometry is retained; no portable operation is registered",
        "likelihood_sequential",
    ),
}


def _status_key(
    name: str, component: Mapping[str, Any]
) -> tuple[str, str, str | None, str, str | None]:
    status = component["status"]
    return (
        name,
        status["state"],
        status.get("code"),
        status.get("reason"),
        component.get("kind"),
    )


def _with_post_capture_reasons(
    payload: Mapping[str, Any], *, actual: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Normalize only the captured status records changed by #270."""
    migrated = copy.deepcopy(dict(payload))
    current = None if actual is None else actual.get("components", {}).get("sequential")
    likelihood_available = current is not None and _status_key("sequential", current) == (
        "sequential",
        "available",
        None,
        "likelihood sequential geometry is retained; no portable operation is registered",
        "likelihood_sequential",
    )
    for name, component in migrated.get("components", {}).items():
        key = _status_key(name, component)
        if (
            key in _POST_CAPTURE_STATUS_REASONS
            and key[3] == "no asymptotic sequential result is retained"
        ):
            replacement = _POST_CAPTURE_STATUS_REASONS[key]
            state, code, reason, kind = replacement
            component["status"]["state"] = state
            component["status"]["code"] = code
            component["status"]["reason"] = reason
            component["kind"] = kind
            key = _status_key(name, component)
        replacement = None
        if name == "sequential" and likelihood_available:
            replacement = _POST_CAPTURE_STATUS_REASONS.get(key)
        if replacement is not None:
            state, code, reason, kind = replacement
            component["status"]["state"] = state
            component["status"]["code"] = code
            component["status"]["reason"] = reason
            component["reason"] = reason if kind is not None else component.get("reason")
            component["kind"] = kind
    return migrated


# Match test_metric_evidence's QR precedent: LAPACK QR bits vary across BLAS platforms.
def _without_snapshot_identity(value: Any, *, clear_content_digest: bool = False) -> Any:
    """Remove only snapshot fields admitted by the owning parity contract."""
    if isinstance(value, Mapping):
        normalized = {
            key: _without_snapshot_identity(item, clear_content_digest=clear_content_digest)
            for key, item in value.items()
        }
        for key in ("source_snapshot_id", "snapshot_id", "decision_snapshot_id", "family_id"):
            if key in normalized:
                normalized[key] = None
        if clear_content_digest and "source_content_digest" in normalized:
            normalized["source_content_digest"] = None
        families = normalized.get("families")
        if isinstance(families, list):
            normalized["families"] = sorted(
                families, key=lambda item: json.dumps(item, sort_keys=True)
            )
        retained = value.get("retained_family")
        if isinstance(retained, str):
            decoded_retained = json.loads(retained)
            if isinstance(decoded_retained, list):
                normalized["retained_family"] = json.dumps(
                    sorted(
                        (
                            _without_snapshot_identity(
                                item, clear_content_digest=clear_content_digest
                            )
                            for item in decoded_retained
                        ),
                        key=lambda item: json.dumps(item, sort_keys=True),
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
        if isinstance(value.get("by_source"), Mapping):
            normalized["by_source"] = sorted(
                (
                    _without_snapshot_identity(item, clear_content_digest=clear_content_digest)
                    for item in value["by_source"].values()
                ),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _without_snapshot_identity(item, clear_content_digest=clear_content_digest)
            for item in value
        ]
    return value


def _decode_fitted_cuped_evidence(value: Any) -> Any:
    """Compare serialized fitted evidence numerically without relaxing declarations."""
    if isinstance(value, Mapping):
        return {
            key: (
                json.loads(item)
                if key == "cuped_evidence" and isinstance(item, str)
                else _decode_fitted_cuped_evidence(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_decode_fitted_cuped_evidence(item) for item in value]
    return value


def _with_post_capture_summary_intervals(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> Mapping[str, Any]:
    """#276 made sequential summaries list their retained bounds as one interval.

    The pre-kernel capture recorded the same bounds with no interval. Only that exact
    change is accepted: the captured summary has no intervals, and the actual summary
    lists exactly its own, unchanged lower and upper bounds.
    """
    summary = expected.get("summary")
    actual_summary = actual.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(actual_summary, Mapping):
        return expected
    bounds = (summary.get("lower"), summary.get("upper"))
    if tuple(summary.get("intervals") or ()) != () or bounds == (None, None):
        return expected
    actual_intervals = [tuple(item) for item in actual_summary.get("intervals") or ()]
    if actual_intervals != [(actual_summary.get("lower"), actual_summary.get("upper"))]:
        return expected
    migrated = copy.deepcopy(dict(expected))
    migrated["summary"] = {**summary, "intervals": [list(bounds)]}
    return migrated


def _project_post_capture_artifact_digests(expected: Any, actual: Any) -> Any:
    """Drop only artifact digest fields absent from the frozen capture schema."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return {
            key: _project_post_capture_artifact_digests(expected[key], value)
            if key in expected
            else value
            for key, value in actual.items()
            if key in expected
            or key not in {"artifact_manifest_sha256", "artifact_relation_digests"}
        }
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return actual
        return [
            _project_post_capture_artifact_digests(left, right)
            for left, right in zip(expected, actual, strict=True)
        ]
    return actual


def _assert_payload_matches(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, snapshotless: bool = False
) -> None:
    canonical_expected = _with_post_capture_summary_intervals(
        ResultCell.from_payload(expected).to_payload(), actual
    )
    expected_payload = _comparable_payload(
        _with_post_capture_reasons(canonical_expected, actual=actual)
    )
    actual_payload = _comparable_payload(_with_post_capture_reasons(actual))
    if snapshotless:
        expected_digest = expected_payload.get("identity", {}).get("source_content_digest")
        actual_digest = actual_payload.get("identity", {}).get("source_content_digest")
        clear_content_digest = expected_digest is None and actual_digest is not None
        expected_payload = _without_snapshot_identity(
            expected_payload, clear_content_digest=clear_content_digest
        )
        actual_payload = _without_snapshot_identity(
            actual_payload, clear_content_digest=clear_content_digest
        )
    _compare(expected_payload, actual_payload)


def _assert_operation_matches(
    expected: Mapping[str, Any],
    operation: Callable[[], ResultCell],
    *,
    snapshotless: bool = False,
    reissue_alpha: float | None = None,
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
        actual_payload = actual.to_payload()
        expected_retained = expected_cell.get("retained_row")
        actual_retained = actual_payload.get("retained_row")
        if isinstance(expected_retained, Mapping) and isinstance(actual_retained, Mapping):
            added_fields = {
                field
                for field in ("allocation_alpha", "family_nominal_alpha")
                if expected_retained.get(field) is None and actual_retained.get(field) is not None
            }
            if added_fields:
                # Step 8 adds these fields to re-issued rows; frozen operations predate them.
                if "allocation_alpha" in added_fields and reissue_alpha is not None:
                    allocation_alpha = actual_retained.get("allocation_alpha")
                    assert isinstance(allocation_alpha, (int, float))
                    max_allocation = reissue_alpha
                    if expected_retained.get("alternative") in {"less", "greater"}:
                        max_allocation *= 2.0
                    assert 0.0 < allocation_alpha <= max_allocation
                actual_payload = dict(actual_payload)
                actual_payload["retained_row"] = {
                    key: None if key in added_fields and key in expected_retained else value
                    for key, value in actual_retained.items()
                    if key not in added_fields or key in expected_retained
                }
                expected_projection = expected_cell.get("projection_fields")
                actual_projection = actual_payload.get("projection_fields")
                if isinstance(expected_projection, Mapping) and isinstance(
                    actual_projection, Mapping
                ):
                    normalized_projection = dict(actual_projection)
                    for field in added_fields:
                        if field in expected_projection:
                            normalized_projection[field] = None
                        else:
                            normalized_projection.pop(field, None)
                    actual_payload["projection_fields"] = normalized_projection
        _assert_payload_matches(expected_cell, actual_payload, snapshotless=snapshotless)


def _identity_hash_without_snapshot(value: object, *, clear_content_digest: bool = False) -> str:
    if isinstance(value, ResultCell):
        identity = value.identity
    else:
        payload = cast(Mapping[str, Any], value).get("result_cell", value)
        assert isinstance(payload, Mapping)
        identity = ResultCell.from_payload(cast(Mapping[str, Any], payload)).identity
    update = {"source_snapshot_id": None}
    if clear_content_digest:
        update["source_content_digest"] = None
    return identity.model_copy(update=update).hash()


def _assert_cell_matches(
    expected: Mapping[str, Any], actual: object, *, snapshotless: bool = False
) -> None:
    from increment.result_envelope import relevel

    expected_cell = expected["result_cell"]
    assert isinstance(expected_cell, Mapping)
    actual_payload = actual.to_payload() if isinstance(actual, ResultCell) else actual
    assert isinstance(actual_payload, Mapping)
    actual_payload = cast(Mapping[str, Any], actual_payload)
    expected_retained = expected_cell.get("retained_row")
    actual_retained = actual_payload.get("retained_row")
    if isinstance(expected_retained, Mapping) and isinstance(actual_retained, Mapping):
        missing_fields = {
            field
            for field in ("allocation_alpha", "family_nominal_alpha")
            if expected_retained.get(field) is None
        }
        allocation_alpha = actual_retained.get("allocation_alpha")
        allowed_post_capture_fields: set[str] = set()
        if allocation_alpha is not None:
            assert isinstance(allocation_alpha, (int, float))
            assert 0.0 < allocation_alpha < 1.0
            assert actual_retained.get("discovery") is True
            allowed_post_capture_fields.update(missing_fields)
        elif (
            "family_nominal_alpha" in missing_fields
            and actual_retained.get("family_nominal_alpha") is not None
        ):
            assert actual_retained.get("discovery") is True
            assert actual_retained.get("family_procedure") == "bh"
            allowed_post_capture_fields.add("family_nominal_alpha")
        for field in missing_fields - allowed_post_capture_fields:
            assert actual_retained.get(field) is None
        if allowed_post_capture_fields:
            # Step 8 re-issues selected cells; frozen captures predate these row fields.
            normalized_retained = dict(actual_retained)
            for field in allowed_post_capture_fields:
                if field in expected_retained:
                    normalized_retained[field] = None
                else:
                    normalized_retained.pop(field)
            actual_payload = dict(actual_payload)
            actual_payload["retained_row"] = normalized_retained
            expected_projection = expected_cell.get("projection_fields")
            actual_projection = actual_payload.get("projection_fields")
            if isinstance(expected_projection, Mapping) and isinstance(actual_projection, Mapping):
                normalized_projection = dict(actual_projection)
                for field in allowed_post_capture_fields:
                    if field in expected_projection:
                        normalized_projection[field] = None
                    else:
                        normalized_projection.pop(field, None)
                actual_payload["projection_fields"] = normalized_projection
    _assert_payload_matches(expected_cell, actual_payload, snapshotless=snapshotless)
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
            snapshotless=snapshotless,
            reissue_alpha=float(alpha),
        )


def _identity_fields(cell: object) -> Mapping[str, object]:
    if isinstance(cell, ResultCell):
        identity = cell.identity
    else:
        payload = cast(Mapping[str, Any], cell).get("result_cell", cell)
        identity = ResultCell.from_payload(cast(Mapping[str, Any], payload)).identity
    return identity.model_dump(mode="json")


def _assert_same_cell_keys(
    expected: Mapping[Any, Any], actual: Mapping[Any, Any], snapshot_inputs: object = None
) -> None:
    """Compare identity keys, naming the differing identity fields and the actual snapshot's
    hashed inputs when they disagree."""
    if expected.keys() == actual.keys():
        return
    expected_fields = [_identity_fields(cell) for cell in expected.values()]
    actual_fields = [_identity_fields(cell) for cell in actual.values()]
    differing = sorted(
        {
            name
            for left, right in zip(expected_fields, actual_fields, strict=False)
            for name in left.keys() | right.keys()
            if left.get(name) != right.get(name)
        }
    )
    details = {
        name: (
            [fields.get(name) for fields in expected_fields],
            [fields.get(name) for fields in actual_fields],
        )
        for name in differing
    }
    raise AssertionError(
        f"cell identity keys differ in fields {differing}: {details}; "
        f"actual snapshot inputs: {snapshot_inputs}"
    )


def _has_platform_sensitive_scope_component(actual: Mapping[str, object]) -> bool:
    """Skip unstable snapshots for QR evidence and pinned artifact generations only."""
    if actual.get("_artifact_generation_identity") is True:
        return True
    components = actual.get("_scope_evidence_components", ())
    if isinstance(components, (list, tuple)):
        for item in components:
            if isinstance(item, Mapping):
                kind = item.get("kind")
                if isinstance(kind, str) and kind.startswith("warehouse_qr:"):
                    return True
    inputs = actual.get("_scope_snapshot_inputs", ())
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            request = item.get("request") if isinstance(item, Mapping) else None
            if isinstance(request, Mapping) and request.get("adjustment_input_identity"):
                return True
    return False


def _compare_family_selection(expected: object, actual: object) -> None:
    if (
        isinstance(expected, Mapping)
        and isinstance(actual, Mapping)
        and isinstance(expected.get("results"), list)
        and isinstance(actual.get("results"), list)
    ):
        expected_rows = cast(list[object], cast(Mapping[str, object], expected)["results"])
        actual_rows = cast(list[object], cast(Mapping[str, object], actual)["results"])
        assert len(expected_rows) == len(actual_rows)
        comparable = dict(actual)
        comparable_rows = []
        for expected_row, actual_row in zip(expected_rows, actual_rows, strict=True):
            assert isinstance(expected_row, Mapping) and isinstance(actual_row, Mapping)
            row = dict(actual_row)
            added = {
                field
                for field in ("allocation_alpha", "family_nominal_alpha")
                if expected_row.get(field) is None and actual_row.get(field) is not None
            }
            if added:
                if "allocation_alpha" in added:
                    assert actual_row.get("discovery") is True
                    allocation_alpha = cast(Mapping[str, object], actual_row)["allocation_alpha"]
                    assert isinstance(allocation_alpha, (int, float))
                    assert 0.0 < allocation_alpha < 1.0
                    nominal = actual_row.get("family_nominal_alpha")
                    if isinstance(nominal, (int, float)):
                        maximum = nominal
                        if actual_row.get("alternative") in {"less", "greater"}:
                            maximum *= 2.0
                        assert allocation_alpha <= maximum
                if "family_nominal_alpha" in added:
                    assert actual_row.get("family_procedure") == "bh"
                # Step 8 adds family-row caps and reissue allocations; frozen captures predate them.
                for field in added:
                    if field in expected_row:
                        row[field] = None
                    else:
                        row.pop(field)
            comparable_rows.append(row)
        comparable["results"] = comparable_rows
        _compare(expected, comparable)
        return
    _compare(expected, actual)


def _assert_outcome_matches(
    expected: Mapping[str, Any], actual: Mapping[str, object]
) -> Mapping[str, object]:
    assert expected["kind"] == actual["kind"]
    if expected["kind"] == "refusal":
        assert expected["code"] == actual["code"]
        assert expected["stage"] == actual["stage"]
        assert _json_context(expected["context"]) == _json_context(actual["context"])
        if "aggregate_evidence" in expected:
            _compare_aggregate(expected["aggregate_evidence"], actual["aggregate_evidence"])
    elif expected["kind"] == "absent":
        assert expected["reason"] == actual["reason"]
    elif expected["kind"] == "cells":
        expected_cells = expected["cells"]
        actual_cells = actual["cells"]
        assert isinstance(expected_cells, Mapping)
        assert isinstance(actual_cells, Mapping)
        snapshotless = _has_platform_sensitive_scope_component(actual)
        cells_to_compare: Any
        actual_cells_by_identity: Any
        if snapshotless:
            expected_by_identity = {
                _identity_hash_without_snapshot(
                    cell,
                    clear_content_digest=(
                        ResultCell.from_payload(cell["result_cell"]).identity.source_content_digest
                        is None
                    ),
                ): cell
                for cell in expected_cells.values()
            }
            actual_by_identity = {
                _identity_hash_without_snapshot(
                    cell,
                    clear_content_digest=(
                        all(
                            ResultCell.from_payload(
                                expected_cell["result_cell"]
                            ).identity.source_content_digest
                            is None
                            for expected_cell in expected_cells.values()
                        )
                    ),
                ): cell
                for cell in actual_cells.values()
            }
            assert len(expected_by_identity) == len(expected_cells)
            assert len(actual_by_identity) == len(actual_cells)
            assert expected_by_identity.keys() == actual_by_identity.keys()
            cells_to_compare = expected_by_identity.items()
            actual_cells_by_identity = actual_by_identity
        else:
            _assert_same_cell_keys(
                expected_cells, actual_cells, actual.get("_scope_snapshot_inputs")
            )
            cells_to_compare = expected_cells.items()
            actual_cells_by_identity = actual_cells
        for identity, expected_cell in cells_to_compare:
            assert isinstance(expected_cell, Mapping)
            actual_cell = cast(Any, actual_cells_by_identity)[identity]
            _assert_cell_matches(expected_cell, actual_cell, snapshotless=snapshotless)
        expected_metadata = _metadata_without_modes(expected["metadata"])
        actual_metadata = _metadata_without_modes(actual["metadata"])
        if snapshotless:
            expected_metadata = _without_snapshot_identity(expected_metadata)
            actual_metadata = _without_snapshot_identity(actual_metadata)
        if actual.get("_artifact_generation_identity") is True:
            actual_metadata = _project_post_capture_artifact_digests(
                expected_metadata, actual_metadata
            )
        _compare(expected_metadata, actual_metadata)
        for field in (
            "population_identity",
            "daily_values",
            "family_selection",
            "aggregate_evidence",
        ):
            if field in expected:
                expected_value = expected[field]
                actual_value = actual[field]
                if snapshotless:
                    expected_value = _without_snapshot_identity(expected_value)
                    actual_value = _without_snapshot_identity(actual_value)
                    if field == "aggregate_evidence":
                        expected_value = _decode_fitted_cuped_evidence(expected_value)
                        actual_value = _decode_fitted_cuped_evidence(actual_value)
                if field == "aggregate_evidence":
                    _compare_aggregate(expected_value, actual_value)
                elif field == "family_selection":
                    _compare_family_selection(expected_value, actual_value)
                else:
                    _compare(expected_value, actual_value)
        modes = _mode_metadata(actual["metadata"])
        if modes:
            return modes
        return cast(Mapping[str, object], actual.get("_execution_modes", {}))
    return {}


def _line1_read(bound: Any, ingress: str, case: Any, capture: Any = None) -> Any:
    from tests.parity_harness.runner import _read

    return _read(bound, ingress, case, capture=capture)


def _enter_patch(stack: Any, target: Any, name: str, replacement: Any) -> None:
    from unittest.mock import patch

    stack.enter_context(patch.object(target, name, replacement))


def _run_with_read_counter(
    bound: Any, ingress: str, case: Any, capture: Any = None
) -> tuple[Any, Mapping[str, int]]:
    from collections import Counter
    from contextlib import ExitStack

    source = held_source(bound[1])
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
        return _line1_read(bound, ingress, case, capture), counts


def _line1_actual_outcome(
    scenario_id: str, ingress: str, *, scoped: bool = False, line4: bool = False
) -> Mapping[str, object]:
    """The current outcome of a registered pair under the capture's recorded conditions.

    The frozen capture ran every DuckDB connection with ``threads=1``
    (``scripts/capture_evidence_identity.capture`` wraps ``_capture_operations`` in
    ``_single_thread_duckdb``); the actual side reuses that same helper so a
    comparison never mixes multi-threaded reduction order into the oracle.
    """
    from increment.readouts import _design_scope, _randomized_scope
    from scripts.capture_evidence_identity import _single_thread_duckdb

    scope_components: list[dict[str, Any]] = []
    snapshot_inputs: list[dict[str, Any]] = []
    original_design_snapshot_id = _design_scope._design_snapshot_id
    original_randomized_snapshot_id = _randomized_scope._randomized_snapshot_id

    def record(source: Mapping[str, Any], request: Mapping[str, Any]) -> None:
        scope_components.extend(source.get("components", ()))
        snapshot_inputs.append(
            {
                "components": [
                    (item.get("kind"), item.get("sha256")) for item in source.get("components", ())
                ],
                "request": _json_context(dict(request)),
            }
        )

    def capture_design_scope(src: Any, source: Mapping[str, Any], request: Mapping[str, Any]):
        record(source, request)
        return original_design_snapshot_id(src, source, request)

    def capture_randomized_scope(source: Mapping[str, Any], request: Mapping[str, Any]):
        record(source, request)
        return original_randomized_snapshot_id(source, request)

    with _single_thread_duckdb(), ExitStack() as stack:
        _enter_patch(stack, _design_scope, "_design_snapshot_id", capture_design_scope)
        _enter_patch(stack, _randomized_scope, "_randomized_snapshot_id", capture_randomized_scope)
        outcome = _line1_outcome(scenario_id, ingress, scoped=scoped, line4=line4)
    if outcome.get("kind") == "cells":
        return {
            **outcome,
            "_scope_evidence_components": scope_components,
            "_scope_snapshot_inputs": snapshot_inputs,
        }
    return outcome


def _line1_outcome(  # noqa: PLR0915
    scenario_id: str, ingress: str, *, scoped: bool = False, line4: bool = False
) -> Mapping[str, object]:
    import json
    import warnings
    from contextlib import ExitStack
    from datetime import UTC, datetime

    from increment import builders
    from increment.errors import CodedError, InvalidRequestError
    from increment.evidence import SequentialCaptureRequest
    from increment.result_envelope import AnalysisEnvelope
    from scripts.capture_evidence_identity import (
        _aggregate_evidence,
        _population_identity,
    )
    from tests.parity_harness.cases import PARITY_CASES, _capture_kwargs, _close_parity_binding

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
            original_from_moments = builders.from_moments

            def capture_moments(rows, **kwargs):
                aggregate_rows.extend(copy.deepcopy(dict(row)) for row in rows)
                return original_from_moments(rows, **kwargs)

            with ExitStack() as stack:
                _enter_patch(stack, builders, "from_moments", capture_moments)
                bound = case.build[ingress]()
        else:
            bound = case.build[ingress]()
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
                # One run reads every declared view under one capture; a sequential case
                # takes its checkpoint inside that run, before any view reads.
                capture = (
                    SequentialCaptureRequest(finalized=True, **_capture_kwargs(bound))
                    if (scoped or line4) and case.sequential
                    else None
                )
                results, read_count = _run_with_read_counter(bound, ingress, case, capture)
            assert all(count == 1 for count in read_count.values()), (
                f"{scenario_id}/{ingress} performed repeated captured reads: {read_count}"
            )
            metadata_value = getattr(results, "metadata", None)
            daily_values = []
            cells = []
            for row in results:
                if hasattr(row, "as_cells"):
                    cells.extend(
                        row.as_cells(
                            metadata_value,
                            created_at=datetime(2026, 1, 1, tzinfo=UTC),
                        )
                    )
                else:
                    daily_values.append(row.model_dump(mode="json"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                population_identity = _population_identity(bound, case, cells)
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
            "daily_values": daily_values,
            "family_selection": family_selection,
            "metadata": AnalysisEnvelope(tuple(cells), metadata_value).to_payload()["metadata"],
            "population_identity": population_identity,
        }
        source_metadata = getattr(results, "source", None)
        if isinstance(source_metadata, Mapping):
            outcome["source_components"] = source_metadata.get("components", ())
        aggregate = _aggregate_evidence(payloads, aggregate_rows)
        if aggregate is not None:
            payload_ref = cast(Mapping[str, Any], aggregate["moments"])
            aggregate["moments"] = payloads[payload_ref["payload_hash"]]
            outcome["aggregate_evidence"] = aggregate
        return outcome
    finally:
        _close_parity_binding(bound)


NEW_PATH_ADAPTERS = dict.fromkeys(_LINE1_PAIRS, _line1_actual_outcome)


def _line4_actual_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    return _line1_actual_outcome(scenario_id, ingress, line4=True)


REGISTERED = {**REGISTERED, "line2": LINE2_PAIRS, "line3": LINE3_PAIRS}
NEW_PATH_ADAPTERS = {
    **NEW_PATH_ADAPTERS,
    **dict.fromkeys(LINE2_PAIRS, line2_actual_outcome),
    **dict.fromkeys(LINE3_PAIRS, line3_actual_outcome),
}

REGISTERED = {**REGISTERED, "line2": LINE2_PAIRS | LINE2_ASOF_PAIRS}
NEW_PATH_ADAPTERS = {
    **NEW_PATH_ADAPTERS,
    **dict.fromkeys(LINE2_PAIRS | LINE2_ASOF_PAIRS, line2_actual_outcome),
}

REGISTERED, NEW_PATH_ADAPTERS = merge_line4(REGISTERED, NEW_PATH_ADAPTERS, _line4_actual_outcome)


REGISTERED = {**REGISTERED, **_line5.REGISTERED}
NEW_PATH_ADAPTERS = {**NEW_PATH_ADAPTERS, **_line5.NEW_PATH_ADAPTERS}


REGISTERED = {**REGISTERED, "line6": _LINE6_PAIRS}
NEW_PATH_ADAPTERS = {**NEW_PATH_ADAPTERS, **_LINE6_ADAPTERS}

REGISTERED = {**REGISTERED, "line7": _LINE7_PAIRS}
NEW_PATH_ADAPTERS = {**NEW_PATH_ADAPTERS, **_LINE7_ADAPTERS}


def test_line7_registry_matches_frozen_artifact_inventory() -> None:
    """Pin top-level frozen outcome kinds; some `cells` records contain daily-only values."""
    from tests.parity_harness.registrations_line7 import _artifact_outcomes

    frozen = _artifact_outcomes()
    assert _LINE7_PAIRS == {(scenario_id, "from_unit_day_artifact") for scenario_id in frozen}
    assert len(frozen) == 112
    assert {
        kind: sum(outcome["kind"] == kind for outcome in frozen.values())
        for kind in ("cells", "refusal", "absent")
    } == {"cells": 93, "refusal": 11, "absent": 8}


def _offset_payload(cell: Mapping[str, Any], fraction: float) -> dict[str, Any]:
    payload = copy.deepcopy(dict(cell))
    summary = payload["summary"]
    point = summary["point"]
    assert isinstance(point, (int, float)) and point != 0
    summary["point"] = point * (1.0 + fraction)
    return payload


def test_capture_scenario_inventory_matches_fixture() -> None:
    from scripts.capture_evidence_identity import iter_capture_keys

    expected = set(iter_capture_keys()) | LINE2_ASOF_PAIRS
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
    assert set(NEW_PATH_ADAPTERS) == set().union(*REGISTERED.values())
    counts = {
        kind: sum(
            captured[scenario_id][ingress]["kind"] == kind for scenario_id, ingress in _LINE1_PAIRS
        )
        for kind in ("cells", "refusal", "absent")
    }
    assert counts == {"cells": 144, "refusal": 19, "absent": 37}


def test_line3_registration_inventory_matches_frozen_capture() -> None:
    captured = _capture()["scenarios"]
    expected_scenarios = {
        "audit-sequential-messy-mean-conversion",
        "audit-sequential-missing-zero",
        "exact_binomial_beyond_the_former_arm_ceiling",
        "scalar_sequential_fixed_threshold_winsor",
        "scalar_sequential_mean_cuped_asymptotic",
        "sequential-mixed-quantile-catalog-checkpoint",
        "sequential_asymptotic_mean_revenue",
        "sequential_automatic_multi_arm_exact",
        "sequential_boolean_segments_automatic_asymptotic_mean",
        "sequential_boolean_segments_automatic_exact_bernoulli",
        "sequential_boolean_segments_registered_asymptotic_mean",
        "sequential_boolean_segments_registered_exact_bernoulli",
        "sequential_explicit_registration_native_artifact_identity",
        "sequential_registered_registration_native_artifact_identity",
    }
    expected_pairs = {
        (scenario_id, ingress)
        for scenario_id in expected_scenarios
        for ingress in ("from_definitions", "from_moments", "from_unit_panel", "from_unit_summary")
    }
    assert LINE3_PAIRS == expected_pairs
    counts = {
        kind: sum(captured[scenario][ingress]["kind"] == kind for scenario, ingress in LINE3_PAIRS)
        for kind in ("cells", "refusal", "absent")
    }
    assert counts == {"cells": 42, "refusal": 14, "absent": 0}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("scenario_id", "ingress", "expected"),
    [
        # The artifact ingress materializes identity for 4,004,003 units (~535s vs ~185s via definitions).
        pytest.param(
            scenario_id,
            ingress,
            outcome,
            marks=[
                pytest.mark.slow,
                pytest.mark.numerical_certification,
                pytest.mark.stress,
                pytest.mark.timeout(900),
            ],
        )
        if (scenario_id, ingress)
        == ("exact_binomial_beyond_the_former_arm_ceiling", "from_unit_day_artifact")
        else pytest.param(
            scenario_id,
            ingress,
            outcome,
            marks=[pytest.mark.slow, pytest.mark.numerical_certification],
        )
        if scenario_id == "exact_binomial_beyond_the_former_arm_ceiling"
        # Scale scenarios taking 15-130s per pair run in the slow tier.
        else pytest.param(scenario_id, ingress, outcome, marks=pytest.mark.slow)
        if scenario_id == "mixed-cuped-multicovariate-fixed-horizon"
        else (scenario_id, ingress, outcome)
        for scenario_id, ingress, outcome in sorted(
            (scenario_id, ingress, outcome)
            for scenario_id, scenario in _capture()["scenarios"].items()
            for ingress, outcome in scenario.items()
        )
    ],
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
    modes = _assert_outcome_matches(
        with_expected_difference(scenario_id, ingress, expected), actual
    )
    if key in set().union(*REGISTERED.values()) and expected["kind"] == "cells":
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
    operation["result_cell"]["retained_row"]["lift"]["log_mean"] += 1.0
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


def test_aggregate_moment_inputs_match_at_practical_tolerance() -> None:
    def outcome(cy1: float, cy2: float) -> dict[str, Any]:
        return {
            "kind": "cells",
            "cells": {},
            "metadata": None,
            "aggregate_evidence": {
                "status": "available",
                "moments_format": 13,
                "moments": [{"cy1": cy1, "cy2": cy2}],
            },
        }

    expected = outcome(-2.1316282072803006e-14, 31.807265452763186)
    # Rounding noise around a zero centered sum and a last-bit difference agree.
    _assert_outcome_matches(expected, outcome(-1.7763568394002505e-14, 31.807265452763183))
    _assert_outcome_matches(expected, outcome(0.0, 31.807265452763186))
    # A material difference does not.
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, outcome(0.0, 31.80729))
    with pytest.raises(AssertionError):
        _assert_outcome_matches(expected, outcome(1e-6, 31.807265452763186))
