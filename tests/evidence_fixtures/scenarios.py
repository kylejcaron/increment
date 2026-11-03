from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from decimal import Decimal
from functools import cache
from itertools import chain, islice
from pathlib import Path
from typing import Any, Literal, cast

from increment._analysis_config import UNSET, overlay_configs
from increment._binding import binding_for
from increment._evidence_types import (
    BreakoutSlicesRequest,
    BreakoutUnitRowsRequest,
    ContrastStatsRequest,
    DayAxisRequest,
    MetricEvidenceRequest,
)
from increment._execute import readout_source
from increment._source_identity import source_identity
from increment.evidence import (
    AssignmentRequest,
    BreakoutIdentity,
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    CountsRequest,
    EvidenceScope,
    MethodFitRequest,
    MomentsRequest,
    SequentialCaptureRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from increment.readouts._evidence import artifact_provenance, day_axis_route
from increment.semantics.design import Encouragement
from tests.evidence_fixtures.temporal import (
    acquire_contrast,
    acquire_day_axis,
    acquire_sequential,
)
from tests.evidence_fixtures.transcript import (
    Transcript,
    _acquire_warehouse_input,
    _digest,
    _frame_payload,
    _plain,
    acquire_transcript,
)
from tests.parity_harness.cases import (
    PARITY_CASES,
    BoundWithMomentInputs,
    _capture_kwargs,
    _close_parity_binding,
)

TEMPORAL_IDS = frozenset(
    {
        "triggered_daily",
        "triggered_daily_lift",
        "triggered_asof",
        "triggered_asof_lift",
        "triggered_encouragement_compliance_asof",
    }
)
ASOF_TRANSFORM_BASES = (
    "breakout_store_dimension",
    "breakout_boolean_store_dimension",
    "conversion_route_breakout_segments",
    "audit-retention-breakout-cohorts",
)


@dataclass(frozen=True, slots=True)
class ScenarioIngress:
    scenario_id: str
    ingress: str
    disposition: Literal["source", "refusal", "absent"]
    family: str
    temporal: bool
    required_representations: tuple[str, ...]
    authority: tuple[str, ...]
    expected_refusal_code: str | None = None
    declared_reason: str | None = None
    expected_absence_type: str | None = None
    expected_absence_field: str | None = None
    expected_refusal_stage: str | None = None
    expected_refusal_context: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class AcquiredScenario:
    scenario_id: str
    ingress: str
    family: str
    records: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    declaration_sha256: str
    input_sha256: str
    actual_representations: tuple[str, ...]
    missing_representations: tuple[str, ...]
    source_input_hashes: tuple[str, ...]
    source_membership_identity: dict[str, Any] | None


def _family(case: Any) -> str:
    identifier = case.id.lower()
    markers = (
        ("cuped", "cuped"),
        ("quantile", "quantile"),
        ("winsor-percentile", "winsor"),
        ("cluster", "cluster"),
        ("observational", "observational"),
        ("encouragement", "encouragement"),
        ("compliance", "encouragement"),
        ("breakout", "breakout"),
        ("conversion", "conversion"),
        ("ratio", "ratio"),
    )
    for marker, family in markers:
        if marker in identifier:
            return family
    return "mean"


def _requirements(case: Any, family: str, ingress: str) -> tuple[str, ...]:
    if case.sequential:
        representations = ["SEQ", "N"]
        if case.metrics:
            representations.append("ME")
        if family in {"cluster", "encouragement"}:
            representations.append("AID")
        if family == "encouragement":
            representations.append("Q")
        if family == "cuped":
            representations.append("F")
        return tuple(representations)
    if case.id.startswith(("asof-", "asof_lift-")):
        return ("DAX", "B", "ME", "N")
    if case.view is not None or case.id in TEMPORAL_IDS:
        return ("DAX", "N", "Q", "AID") if "encouragement" in case.id else ("DAX", "N")
    if case.id == "audit-switchback-neighboring-integers":
        return ("X", "AID", "N")
    if "triggered-sitewide" in case.id:
        return ("ME", "N", "AID")
    has_breakout = case.breakout_dimension is not None or "multi-metric-breakout" in case.id
    if ingress == "from_moments":
        if family == "cuped":
            return ("M", "F", "N")
        if family == "cluster":
            return ("M", "N", "AID")
        if family == "encouragement" and has_breakout:
            return ("B", "N", "AID") if not case.metrics else ("B", "M", "Q", "N", "AID")
        if family == "encouragement" and not case.metrics:
            return ("N", "AID", "Q")
        return ("M", "N")
    if family == "encouragement" and has_breakout:
        required = ("B", "Q", "N", "AID") if not case.metrics else ("B", "ME", "Q", "N", "AID")
    else:
        required = {
            "cuped": ("ME", "U", "N", "F"),
            "quantile": ("ME", "U", "N"),
            "winsor": ("ME", "U(raw)", "N"),
            "cluster": ("ME", "N", "AID"),
            "breakout": ("B", "ME", "N"),
            "observational": ("ME", "U", "N"),
            "encouragement": ("ME", "Q", "N", "AID"),
        }.get(family, ("ME", "N"))
    if not case.metrics and family == "encouragement":
        required = tuple(value for value in required if value != "ME")
    return required


def _actual_representations(records: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    representations = set()
    request_symbols = {
        "MetricEvidenceRequest": "ME",
        "MomentsRequest": "M",
        "CountsRequest": "N",
        "AssignmentRequest": "AID",
        "UnitRowsRequest": "U",
        "MethodFitRequest": "F",
        "DayAxisRequest": "DAX",
        "SequentialCaptureRequest": "SEQ",
        "ContrastStatsRequest": "X",
        "ComplianceRequest": "Q",
        "BreakoutSlicesRequest": "B",
        "BreakoutMethodMomentsRequest": "B",
        "BreakoutUnitRowsRequest": "B",
    }
    for record in records:
        request_type = record["request_type"]
        representation = request_symbols.get(request_type)
        if representation is not None:
            representations.add(representation)
        answer = record["answer"]
        if request_type == "SequentialCaptureRequest":
            checkpoint = answer["checkpoint"]
            uptake_metrics = {
                model["metric"]
                for model in checkpoint["registration"]["models"]
                if model.get("observable") == "uptake"
            }
            if uptake_metrics and uptake_metrics <= {
                state["metric"] for state in checkpoint["states"]
            }:
                representations.add("Q")
        if request_type == "AssignmentRequest" and answer.get("arm_counts"):
            representations.add("N")
        if request_type == "UnitRowsRequest" and record["request"].get("outcome_stage") == "raw":
            representations.add("U(raw)")
        if request_type == "MetricEvidenceRequest" and answer.get("method_fit_answers"):
            representations.add("F")
        if request_type == "MomentsRequest" and record["answer_type"] == "MomentRows":
            if any(
                isinstance(row, Mapping)
                and row.get("record_kind") == "cuped_evidence"
                and row.get("cuped_evidence")
                for row in answer
            ):
                representations.add("F")
    return tuple(sorted(representations))


def _missing_representations(
    required: tuple[str, ...],
    actual: tuple[str, ...],
    source_refusals: Iterable[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    refused = set()
    for item in source_refusals:
        representation = item["representation"]
        if representation == "F":
            expected_code = "evidence.operation_unavailable"
            expected_request = "MethodFitRequest"
            expected_context = {
                "constructor": "MomentsSource",
                "requested": "method_fit",
                "offered": ["warehouse", "local", "retained"],
                "route": "use a source that retains method-specific evidence",
            }
        elif representation == "ME":
            if "SEQ" not in actual or not item.get("checkpoint_registration_id"):
                raise ValueError("windowed total refusal requires its acquired sequential input")
            expected_code = "source.frame.windowed_encouragement_total"
            expected_request = "MetricEvidenceRequest"
            expected_context = {
                "metric": item["request"]["metric"]["name"],
                "route": (
                    "use grain='asof', or compute each unit's windowed value upstream "
                    "and declare it as an unwindowed metric on from_unit_summary"
                ),
            }
        else:
            raise ValueError(f"unclassified source refusal representation: {representation}")
        if (
            item["code"] != expected_code
            or item["request_type"] != expected_request
            or _plain(item["context"]) != expected_context
        ):
            raise ValueError(f"source refusal does not match its declared contract: {item}")
        refused.add(representation)
    return tuple(
        representation
        for representation in required
        if representation not in actual and representation not in refused
    )


def _authority(family: str) -> tuple[str, ...]:
    return {
        "cuped": ("OI", "DR", "OO", "IL", "BC"),
        "quantile": ("DR", "OO", "BC"),
        "winsor": ("DR", "OO", "BC"),
        "cluster": ("DR", "OO", "BC"),
        "breakout": ("DR", "OO", "BC"),
        "observational": ("DR", "OO", "IL", "BC"),
        "encouragement": ("DR", "OO", "BC"),
        "preflight": ("OO",),
    }.get(family, ("OI", "DR", "OO", "BC"))


def _source_plain(value: Any) -> Any:
    """Encode original source state without losing Decimal or non-string keys."""
    if isinstance(value, Decimal):
        return {"$source_decimal": str(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _source_plain(item) for key, item in sorted(value.items())}
        pairs = sorted(
            ([_source_plain(key), _source_plain(item)] for key, item in value.items()),
            key=lambda pair: repr(pair[0]),
        )
        return {"$source_mapping": pairs}
    if isinstance(value, (tuple, list)):
        return [_source_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_source_plain(item) for item in value), key=repr)
    if hasattr(value, "__dataclass_fields__"):
        return {name: _source_plain(getattr(value, name)) for name in value.__dataclass_fields__}
    return _plain(value)


def _breakout_identities(source: Any, experiment: Any) -> tuple[BreakoutIdentity, ...]:
    definitions = tuple(getattr(experiment, "breakouts", ()))
    if definitions:
        return tuple(BreakoutIdentity(item.property, item.source) for item in definitions)
    capabilities = getattr(source, "evidence", None)
    return tuple(BreakoutIdentity(name, None) for name in getattr(capabilities, "breakouts", ()))


_SOURCE_INPUT_BATCH_SIZE = 1024


def _source_input_bytes(value: Any) -> bytes:
    """Encode original source receipts without reordering observations."""
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _source_row_hashes(
    metadata: Mapping[str, Any], batches: Iterable[list[Any]]
) -> tuple[str, str, int]:
    """Hash the canonical source object while retaining only one row batch."""
    import hashlib

    claim = hashlib.sha256()
    audit = hashlib.sha256()
    row_count = 0

    def emit(chunk: bytes) -> None:
        claim.update(chunk)
        audit.update(chunk)

    emit(b"{")
    for index, key in enumerate(sorted((*metadata, "rows"))):
        if index:
            emit(b",")
        emit(_source_input_bytes(key))
        emit(b":")
        if key == "rows":
            emit(b"[")
            for batch in batches:
                if not batch:
                    continue
                if row_count:
                    emit(b",")
                emit(_source_input_bytes(batch)[1:-1])
                row_count += len(batch)
            emit(b"]")
        else:
            emit(_source_input_bytes(metadata[key]))
    emit(b"}")
    return claim.hexdigest(), audit.hexdigest(), row_count


def _source_table_batches(cursor: Any) -> Iterable[list[Any]]:
    while rows := cursor.fetchmany(_SOURCE_INPUT_BATCH_SIZE):
        yield _plain(rows)


def _source_plain_batches(rows: Iterable[Any]) -> Iterable[list[Any]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, _SOURCE_INPUT_BATCH_SIZE)):
        yield _source_plain(batch)


def _source_frame_payloads(value: Any) -> Iterable[dict[str, Any]]:
    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module == "narwhals":
        to_native = getattr(value, "to_native", None)
        if not callable(to_native):
            raise TypeError(f"Unsupported tabular transcript carrier: {type(value).__qualname__}")
        native = to_native()
        if native is value:
            raise TypeError("Narwhals frame did not expose a native tabular carrier")
        yield from _source_frame_payloads(native)
    elif module == "pandas":
        for start in range(0, max(len(value), 1), _SOURCE_INPUT_BATCH_SIZE):
            yield _frame_payload(value.iloc[start : start + _SOURCE_INPUT_BATCH_SIZE])
    elif module in {"polars", "pyarrow"}:
        row_count = value.height if module == "polars" else value.num_rows
        for start in range(0, max(row_count, 1), _SOURCE_INPUT_BATCH_SIZE):
            yield _frame_payload(value.slice(start, _SOURCE_INPUT_BATCH_SIZE))
    else:
        raise TypeError(f"Unsupported tabular transcript carrier: {type(value).__qualname__}")


def _source_inputs(
    source: Any,
    scenario_id: str,
    ingress: str,
    requests: tuple[object, ...],
    *,
    hash_audits: list[str],
) -> list[dict[str, Any]]:
    """Hash original in-memory source rows plus fixed declaration/configuration."""
    import hashlib

    def digest(value):
        original_bytes = _source_input_bytes(value)
        claimed = hashlib.sha256(original_bytes).hexdigest()
        hash_audits.append(hashlib.sha256(original_bytes).hexdigest())
        return claimed

    def rows_digest(metadata, batches):
        claimed, audit, row_count = _source_row_hashes(metadata, batches)
        hash_audits.append(audit)
        return claimed, row_count

    declaration_method = getattr(source, "declaration_identity", None)

    declaration = _plain(declaration_method()) if callable(declaration_method) else {}
    configuration = {
        "scenario_id": scenario_id,
        "ingress": ingress,
        "source_declaration": declaration,
        "request_coordinates": [_plain(request) for request in requests],
        "source_identity": _plain(source_identity(source)),
    }
    source_inputs: list[dict[str, Any]] = []
    frame = getattr(source, "_source_frame", None)
    if frame is None:
        frame = getattr(source, "_frame", None)
    if frame is not None:
        payloads = iter(_source_frame_payloads(frame))
        frame_payload = next(payloads)
        claimed, row_count = rows_digest(
            {key: frame_payload[key] for key in ("columns", "schema")},
            chain((frame_payload["rows"],), (payload["rows"] for payload in payloads)),
        )
        source_inputs.append(
            {
                "kind": "declared_source_frame",
                "identity": f"{scenario_id}:{ingress}:{type(source).__qualname__}",
                "configuration": {
                    **configuration,
                    "frame_columns": frame_payload["columns"],
                    "frame_schema": frame_payload["schema"],
                    "row_count": row_count,
                },
                "sha256": claimed,
            }
        )
    backend = getattr(source, "_con", None)
    raw_connection = getattr(backend, "con", None)
    if frame is None and raw_connection is not None:
        try:
            table_names = tuple(
                sorted(row[0] for row in raw_connection.execute("show tables").fetchall())
            )
        except Exception as exc:
            raise TypeError(
                f"cannot enumerate declared source tables for {type(source).__qualname__}"
            ) from exc
        for table_name in table_names:
            quoted_name = '"' + table_name.replace('"', '""') + '"'
            schema = raw_connection.execute(f"describe {quoted_name}").fetchall()
            claimed, row_count = rows_digest(
                {"schema": _plain(schema)},
                _source_table_batches(raw_connection.execute(f"select * from {quoted_name}")),
            )
            source_inputs.append(
                {
                    "kind": "declared_source_table",
                    "identity": f"{scenario_id}:{ingress}:{table_name}",
                    "configuration": {
                        **configuration,
                        "table_name": table_name,
                        "table_schema": _plain(schema),
                        "row_count": row_count,
                    },
                    "sha256": claimed,
                }
            )
    if not source_inputs and hasattr(source, "_rows"):
        claimed, _row_count = rows_digest(
            _source_plain(
                {
                    "assignment_counts": source._assignment_counts,
                    "compliance": source._compliance,
                    "population": source._population,
                }
            ),
            _source_plain_batches(source._rows),
        )
        source_inputs.append(
            {
                "kind": "declared_source_state",
                "identity": f"{scenario_id}:{ingress}:{type(source).__qualname__}",
                "configuration": configuration,
                "sha256": claimed,
            }
        )
    manifest = getattr(source, "_manifest", None)
    if not source_inputs and manifest is not None:
        manifest_payload = manifest.model_dump(mode="json")
        source_inputs.append(
            {
                "kind": "declared_artifact_manifest",
                "identity": (
                    f"{scenario_id}:{ingress}:{manifest.artifact_id}:{manifest.generation_id}"
                ),
                "configuration": {
                    **configuration,
                    "artifact_id": str(manifest.artifact_id),
                    "generation_id": str(manifest.generation_id),
                },
                "sha256": digest(manifest_payload),
            }
        )
    if not source_inputs and hasattr(source, "_stats"):
        source_state = _source_plain(
            {
                "stats": source._stats,
                "assignment_evidence": source._assignment_evidence,
                "roster": source._roster,
                "order_cycles": source._order_cycles,
            }
        )
        source_inputs.append(
            {
                "kind": "declared_source_state",
                "identity": f"{scenario_id}:{ingress}:{type(source).__qualname__}",
                "configuration": configuration,
                "sha256": digest(source_state),
            }
        )
    if not source_inputs:
        raise TypeError(
            f"no original source rows are exposed for {scenario_id}/{ingress} "
            f"({type(source).__qualname__})"
        )
    return source_inputs


@cache
def _frozen_dispositions() -> Mapping[tuple[str, str], Mapping[str, Any]]:
    from scripts.capture_evidence_identity import load_operations_capture

    root = Path(__file__).parents[1] / "fixtures" / "evidence_identity"
    outcomes = {}
    for name in ("pre-kernel", "pre-kernel-asof"):
        capture = load_operations_capture(root / name / "operations.json.gz")
        for scenario_id, ingresses in capture["scenarios"].items():
            for ingress, outcome in ingresses.items():
                coordinate = (scenario_id, ingress)
                if coordinate in outcomes:
                    raise ValueError(f"duplicate frozen scenario coordinate: {coordinate}")
                outcomes[coordinate] = {
                    key: outcome[key]
                    for key in ("kind", "code", "stage", "context", "reason")
                    if key in outcome
                }
    return outcomes


def _scenario_case(scenario_id: str) -> Any:
    prefix = next((value for value in ("asof_lift-", "asof-") if scenario_id.startswith(value)), "")
    base_id = scenario_id.removeprefix(prefix) if prefix else scenario_id
    case = next((candidate for candidate in PARITY_CASES if candidate.id == base_id), None)
    if case is None:
        raise KeyError(f"unknown declared scenario: {scenario_id}")
    return replace(case, id=scenario_id, view=prefix[:-1]) if prefix else case


def scenario_roster(*, include_temporal: bool = True) -> tuple[ScenarioIngress, ...]:
    """Return fixed scenario and refusal declarations, never emitted answers."""
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS

    cases: dict[str, Any] = {}
    for case in PARITY_CASES:
        if case.id in cases:
            raise ValueError(f"duplicate declared scenario id: {case.id}")
        cases[case.id] = case
    roster = []
    for case in cases.values():
        temporal_case = bool(case.sequential or case.view is not None or case.id in TEMPORAL_IDS)
        ingresses = set(case.build) | set(case.waive)
        for ingress in sorted(ingresses):
            temporal = temporal_case or ingress == "from_switchback_panel"
            if temporal and not include_temporal:
                continue
            outcome = _frozen_dispositions()[(case.id, ingress)]
            disposition = cast(
                Literal["source", "refusal", "absent"],
                {"cells": "source", "refusal": "refusal", "absent": "absent"}[outcome["kind"]],
            )
            family = _family(case)
            roster.append(
                ScenarioIngress(
                    case.id,
                    ingress,
                    disposition,
                    family,
                    temporal,
                    _requirements(case, family, ingress) if disposition == "source" else (),
                    _authority(family),
                    expected_refusal_code=outcome.get("code"),
                    declared_reason=outcome.get("reason", case.waive.get(ingress)),
                    expected_absence_type=(
                        case.expected_absence[ingress].error.__name__
                        if ingress in case.expected_absence
                        else "NoSourceBuilder"
                        if disposition == "absent"
                        else None
                    ),
                    expected_absence_field=(
                        case.expected_absence[ingress].field
                        if ingress in case.expected_absence
                        else "case.build"
                        if disposition == "absent"
                        else None
                    ),
                    expected_refusal_stage=outcome.get("stage"),
                    expected_refusal_context=outcome.get("context", {}),
                ),
            )
    if include_temporal:
        for prefix in ("asof-", "asof_lift-"):
            for scenario_id in ASOF_TRANSFORM_BASES:
                case = cases[scenario_id]
                for ingress in (
                    "from_definitions",
                    "from_moments",
                    "from_unit_panel",
                    "from_unit_summary",
                ):
                    built = ingress in case.build
                    outcome = _frozen_dispositions()[(f"{prefix}{scenario_id}", ingress)]
                    disposition = cast(
                        Literal["source", "refusal", "absent"],
                        {"cells": "source", "refusal": "refusal", "absent": "absent"}[
                            outcome["kind"]
                        ],
                    )
                    roster.append(
                        ScenarioIngress(
                            f"{prefix}{scenario_id}",
                            ingress,
                            disposition,
                            "breakout",
                            True,
                            ("DAX", "B", "ME", "N") if disposition == "source" else (),
                            ("DR", "OO", "BC"),
                            expected_refusal_code=outcome.get("code"),
                            declared_reason=outcome.get("reason"),
                            expected_absence_type="NoSourceBuilder" if not built else None,
                            expected_absence_field="case.build" if not built else None,
                            expected_refusal_stage=outcome.get("stage"),
                            expected_refusal_context=outcome.get("context", {}),
                        ),
                    )
    for code in REFUSAL_SCENARIOS:
        outcome = _frozen_dispositions()[(code, "refusal")]
        roster.append(
            ScenarioIngress(
                code,
                "refusal",
                "refusal" if outcome["kind"] == "refusal" else "absent",
                "preflight",
                False,
                (),
                ("OO",),
                expected_refusal_code=outcome.get("code"),
                declared_reason=outcome.get("reason"),
                expected_absence_type="Unreachable" if outcome["kind"] == "absent" else None,
                expected_absence_field="REFUSAL_SCENARIOS" if outcome["kind"] == "absent" else None,
                expected_refusal_stage=outcome.get("stage"),
                expected_refusal_context=outcome.get("context", {}),
            )
        )
    if include_temporal:
        coordinates = {(entry.scenario_id, entry.ingress) for entry in roster}
        if len(coordinates) != len(roster) or coordinates != _frozen_dispositions().keys():
            raise ValueError("declared scenario coordinates do not match the frozen inventory")
    return tuple(roster)


def observe_scenario_disposition(scenario_id: str, ingress: str) -> Mapping[str, Any]:
    """Check a refusal or absence against its exact frozen public contract."""
    from increment import workflows
    from increment.errors import CodedError
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable
    from tests.parity_harness.runner import _read

    expected = _frozen_dispositions()[(scenario_id, ingress)]
    if expected["kind"] == "cells":
        raise ValueError("successful scenarios require typed input acquisition")
    if ingress == "refusal":
        callback = REFUSAL_SCENARIOS[scenario_id]
        if isinstance(callback, Unreachable):
            actual = {"kind": "absent", "reason": callback.reason}
        else:
            try:
                callback()
            except CodedError as error:
                actual = {
                    "kind": "refusal",
                    "code": error.code,
                    "stage": "run",
                    "context": _plain(error.context),
                }
            else:
                raise AssertionError(f"{scenario_id}: required refusal callback returned")
    else:
        case = _scenario_case(scenario_id)
        if ingress not in case.build:
            actual = {
                "kind": "absent",
                "reason": case.waive.get(ingress, "constructor ingress is not declared"),
            }
        else:
            bound = None
            stage = "construct"
            actual = None
            try:
                bound = case.build[ingress]()
                stage = "run"
                if case.sequential:
                    workflows.capture_sequential(*bound, finalized=True, **_capture_kwargs(bound))
                _read(bound, ingress, case)
            except CodedError as error:
                actual = {
                    "kind": "refusal",
                    "code": error.code,
                    "stage": stage,
                    "context": _plain(error.context),
                }
            finally:
                if bound is not None:
                    _close_parity_binding(bound)
            if actual is None:
                raise AssertionError(f"{scenario_id}/{ingress}: required refusal returned")
    if actual != expected:
        raise AssertionError(
            f"{scenario_id}/{ingress}: frozen disposition {expected!r} != observed {actual!r}"
        )
    return {"expected": expected, "observed": actual}


def scenario_coverage_payload(
    *, include_temporal: bool = False, acquired: Iterable[AcquiredScenario] = ()
) -> dict[str, Any]:
    """Return fixed declarations plus actual request and representation coverage."""
    roster = scenario_roster(include_temporal=include_temporal)
    families: dict[str, dict[str, int]] = {}
    for entry in roster:
        counts = families.setdefault(
            entry.family, {"source": 0, "refusal": 0, "absent": 0, "temporal": 0}
        )
        counts[entry.disposition] += 1
        counts["temporal"] += int(entry.temporal)
    source_coordinates = {
        (entry.scenario_id, entry.ingress) for entry in roster if entry.disposition == "source"
    }
    acquisitions = tuple(acquired)
    acquired_coordinates = [(item.scenario_id, item.ingress) for item in acquisitions]
    if len(set(acquired_coordinates)) != len(acquired_coordinates):
        raise ValueError("duplicate acquired scenario coordinate")
    unexpected = set(acquired_coordinates) - source_coordinates
    if unexpected:
        raise ValueError(f"unexpected acquired scenario coordinates: {unexpected}")
    request_symbols = {
        "MetricEvidenceRequest": "ME",
        "MomentsRequest": "M",
        "CountsRequest": "N",
        "AssignmentRequest": "AID",
        "UnitRowsRequest": "U",
        "MethodFitRequest": "F",
        "ComplianceRequest": "Q",
        "ComplianceSeriesRequest": "Q",
        "BreakoutSlicesRequest": "B",
        "BreakoutMethodMomentsRequest": "B",
        "BreakoutUnitRowsRequest": "B",
    }
    for item in acquisitions:
        actual = set(item.actual_representations)
        for record in item.records:
            representation = request_symbols.get(record["request_type"])
            if representation:
                actual.add(representation)
            if (
                record["request_type"] == "UnitRowsRequest"
                and record["request"].get("outcome_stage") == "raw"
            ):
                actual.add("U(raw)")
        entry = next(
            entry
            for entry in roster
            if (entry.scenario_id, entry.ingress) == (item.scenario_id, item.ingress)
        )
        missing = _missing_representations(
            entry.required_representations,
            tuple(actual),
            cast(
                Iterable[Mapping[str, Any]],
                item.provenance.get("source_refusals", ()),
            ),
        )
        if missing != item.missing_representations:
            raise ValueError(
                f"{item.scenario_id}/{item.ingress} representation coverage does not match records"
            )
    return {
        "source": "PARITY_CASES plus REFUSAL_SCENARIOS and declared as-of transforms",
        "temporal_included": include_temporal,
        "scenario_ingresses": [
            {
                "scenario_id": entry.scenario_id,
                "ingress": entry.ingress,
                "disposition": entry.disposition,
                "family": entry.family,
                "required_representations": entry.required_representations,
                "authority": entry.authority,
                "expected_refusal_code": entry.expected_refusal_code,
                "declared_reason": entry.declared_reason,
                "expected_absence_type": entry.expected_absence_type,
                "expected_absence_field": entry.expected_absence_field,
                "expected_refusal_stage": entry.expected_refusal_stage,
                "expected_refusal_context": entry.expected_refusal_context,
            }
            for entry in roster
        ],
        "acquisitions": [
            {
                "scenario_id": item.scenario_id,
                "ingress": item.ingress,
                "family": item.family,
                "request_types": tuple(record["request_type"] for record in item.records),
                "request_coordinates": tuple(
                    {"request_type": record["request_type"], "request": record["request"]}
                    for record in item.records
                ),
                "actual_representations": item.actual_representations,
                "missing_representations": item.missing_representations,
                "source_provenance": item.provenance,
                "declaration_sha256": item.declaration_sha256,
                "input_sha256": item.input_sha256,
            }
            for item in acquisitions
        ],
        "unacquired_source_coordinates": sorted(source_coordinates - set(acquired_coordinates)),
        "family_counts": families,
        "totals": {
            "scenario_ingresses": len(roster),
            "scenario_ids": len({entry.scenario_id for entry in roster}),
            "source": sum(entry.disposition == "source" for entry in roster),
            "refusal": sum(entry.disposition == "refusal" for entry in roster),
            "absent": sum(entry.disposition == "absent" for entry in roster),
            "temporal": sum(entry.temporal for entry in roster),
            "acquired": len(acquisitions),
            "complete_acquisitions": sum(not item.missing_representations for item in acquisitions),
        },
    }


def refusal_contract_roster() -> tuple[tuple[str, bool, str | None], ...]:
    """List runnable refusal owners separately from constructor refusals."""
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable

    return tuple(
        (
            code,
            not isinstance(scenario, Unreachable),
            scenario.reason if isinstance(scenario, Unreachable) else None,
        )
        for code, scenario in sorted(REFUSAL_SCENARIOS.items())
    )


def _declared_covariates(source: Any, config: Any) -> tuple[Any, ...]:
    configured = tuple(config.covariates)
    if configured:
        return configured
    adjustment = getattr(source.context.design, "adjustment", None)
    return tuple(getattr(adjustment, "covariates", ()))


def _metric_requests(
    source: Any,
    population: Literal["assigned", "triggered"],
    *,
    by: tuple[str, ...] = (),
    configs: tuple[Any, ...] | None = None,
) -> tuple[MetricEvidenceRequest, ...]:
    from increment._analysis_config import effective_methods

    context = source.context
    design = context.design
    control = str(design.control_group)
    requests = []
    for config in context.configs if configs is None else configs:
        covariates = _declared_covariates(source, config)
        methods = effective_methods(config, design=design)
        requests.append(
            MetricEvidenceRequest(
                metric=config.metric,
                control_group=control,
                by=by,
                methods=tuple(methods),
                covariates=covariates,
                population=population,
            )
        )
    return tuple(requests)


def _unit_requests(
    source: Any,
    population: Literal["assigned", "triggered"],
    *,
    outcome_stage: Literal["transformed", "raw"] = "transformed",
    configs: tuple[Any, ...] | None = None,
) -> tuple[UnitRowsRequest, ...]:
    configs = tuple(source.context.configs) if configs is None else configs
    requests: list[UnitRowsRequest] = []
    for config in configs:
        covariates = tuple(
            str(getattr(item, "metric", getattr(item, "name", item)))
            for item in _declared_covariates(source, config)
        )
        requests.append(
            UnitRowsRequest(
                config.metric,
                covariates=covariates,
                outcome_stage=outcome_stage,
                population=population,
            )
        )
    return tuple(requests)


def _method_fit_requests(
    source: Any,
    population: Literal["assigned", "triggered"],
    *,
    configs: tuple[Any, ...] | None = None,
) -> tuple[MethodFitRequest, ...]:
    context = source.context
    arm_order = tuple(sorted(source.unit_counts()))
    return tuple(
        MethodFitRequest(
            metric=config.metric,
            method=method,
            control=str(context.design.control_group),
            arm_order=arm_order,
            population=population,
        )
        for config in (context.configs if configs is None else configs)
        for method in (config.decision_method, *config.sensitivity_methods)
        if method.variance_reduction in {"cuped", "cuped_multicovariate"}
    )


def _method_fit_refusals(
    source: Any, provider: Any, requests: tuple[MethodFitRequest, ...]
) -> tuple[dict[str, Any], ...]:
    from increment.errors import CodedError

    refusals = []
    for request in requests:
        try:
            request_evidence(provider, request)
        except CodedError as exc:
            if exc.code != "evidence.operation_unavailable":
                raise
            refusals.append(
                {
                    "representation": "F",
                    "request_type": "MethodFitRequest",
                    "request": _plain(request),
                    "code": exc.code,
                    "context": _plain(exc.context),
                    "capabilities": _plain(source.evidence),
                }
            )
        else:
            raise TypeError(
                f"{type(source).__qualname__} supplied method-fit data despite its refusal recipe"
            )
    return tuple(refusals)


def _sequential_metric_inputs(
    source: Any,
    provider: Any,
    requests: tuple[MetricEvidenceRequest, ...],
    checkpoint: Mapping[str, Any],
) -> tuple[tuple[MetricEvidenceRequest, ...], tuple[dict[str, Any], ...]]:
    from increment.errors import CodedError

    if not (
        isinstance(source.context.design, Encouragement) and source.evidence.shape == "unit_panel"
    ):
        return requests, ()
    supported = []
    refusals = []
    for request in requests:
        try:
            request_evidence(provider, request)
        except CodedError as error:
            if error.code != "source.frame.windowed_encouragement_total":
                raise
            if request.metric.name not in {state["metric"] for state in checkpoint["states"]}:
                raise ValueError(
                    "refused total metric is missing from the acquired checkpoint"
                ) from error
            refusals.append(
                {
                    "representation": "ME",
                    "request_type": "MetricEvidenceRequest",
                    "request": _plain(request),
                    "code": error.code,
                    "context": _plain(error.context),
                    "capabilities": _plain(source.evidence),
                    "checkpoint_registration_id": checkpoint["registration_id"],
                }
            )
        else:
            supported.append(request)
    return tuple(supported), tuple(refusals)


def declared_requests(
    source: Any,
    family: str,
    ingress: str,
    population: Literal["assigned", "triggered"] = "assigned",
    *,
    configs: tuple[Any, ...] | None = None,
) -> tuple[object, ...]:
    """Build requests from fixed ingress, metric, method, breakout, and population declarations."""
    context = source.context
    configs = tuple(context.configs) if configs is None else configs
    declaration_identity = getattr(source, "declaration_identity", None)
    declaration = declaration_identity() if callable(declaration_identity) else {}
    experiment = declaration.get("experiment") if isinstance(declaration, Mapping) else None
    definitions = tuple(getattr(experiment, "breakouts", ()))
    breakouts = _breakout_identities(source, experiment)
    requests: list[object] = (
        [MomentsRequest(config.metric, population=population) for config in configs]
        if ingress == "from_moments"
        else list(_metric_requests(source, population, configs=configs))
    )
    requests.append(CountsRequest(kind="unit", population=population))
    if family == "cluster":
        requests.append(CountsRequest(kind="cluster", population=population))
    populations = getattr(getattr(source, "evidence", None), "populations", frozenset())
    if {"assigned", "triggered"}.issubset(populations):
        requests.extend(
            AssignmentRequest(experiment=context.study_id, population=member)
            for member in ("assigned", "triggered")
        )
    elif family == "cluster" or isinstance(context.design, Encouragement):
        requests.append(AssignmentRequest(experiment=context.study_id, population=population))
    if isinstance(context.design, Encouragement):
        requests.append(ComplianceRequest(design=context.design, population=population))
    if (
        family == "cuped"
        and getattr(getattr(source, "evidence", None), "method_fit", "none") != "none"
    ):
        requests.extend(_method_fit_requests(source, population, configs=configs))
    if breakouts:
        requests.append(
            BreakoutSlicesRequest(
                breakouts=breakouts,
                breakout_definitions=definitions,
                metrics=tuple(config.metric for config in configs),
                population=population,
                configs=configs,
            )
        )
        for config in configs:
            for method in (config.decision_method, *config.sensitivity_methods):
                requests.extend(
                    BreakoutMethodMomentsRequest(
                        config.metric, method, breakout, population=population
                    )
                    for breakout in breakouts
                )
            if _declared_covariates(source, config) and ingress != "from_moments":
                requests.extend(
                    BreakoutUnitRowsRequest(
                        config.metric,
                        breakout,
                        population=population,
                        covariates=tuple(
                            str(getattr(item, "metric", getattr(item, "name", item)))
                            for item in _declared_covariates(source, config)
                        ),
                    )
                    for breakout in breakouts
                )
    if ingress in {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
    } and family in {
        "cuped",
        "quantile",
        "winsor",
        "observational",
    }:
        requests.extend(
            _unit_requests(
                source,
                population,
                configs=configs,
                outcome_stage="raw" if family == "winsor" else "transformed",
            )
        )
    declarations = [(type(request).__name__, repr(request)) for request in requests]
    if len(declarations) != len(set(declarations)):
        raise ValueError("duplicate declared evidence request")
    return tuple(requests)


def _artifact_breakout_unit_records(
    slices: tuple[Any, ...],
    requests: tuple[object, ...],
    *,
    scenario_id: str,
    ingress: str,
    source_key: Any,
) -> dict[int, Mapping[str, Any]]:
    records = {}
    for index, request in enumerate(requests):
        if not isinstance(request, BreakoutUnitRowsRequest):
            continue
        selected_slice = next(item for item in slices if item.identity == request.breakout)
        scoped = acquire_transcript(
            evidence_provider(selected_slice.source),
            scenario_id=scenario_id,
            ingress=ingress,
            requests=(request,),
            baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
            provenance={"source_key": source_key},
        )
        records[index] = scoped.records[0]
    return records


def _acquire_breakout_scope_inputs(  # noqa: PLR0915
    item: Any,
    request: BreakoutSlicesRequest,
    *,
    input_kind: Literal["materialized_breakout_slice", "summary_breakout_source_slice"],
    configs: tuple[Any, ...],
    estimands: tuple[str, ...] | None,
    format_inputs: tuple[Any, ...],
    warehouse_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retain closed inputs and actual component associations from one pinned slice."""
    from increment.readouts._breakout import _load_cuped_breakout_frames
    from increment.readouts._evidence import fits_uncovered_cells_locally
    from increment.readouts._metric_rows import _load_breakout_metric_rows

    source_name = item.stamp_source if item.stamp_source is not None else item.source_name
    dimension = item.identity.property
    scope = EvidenceScope(
        grain="total",
        population=request.population,
        metrics=tuple(config.metric for config in configs),
        configs=configs,
        breakouts=(item.identity,),
        execution=True,
    )
    scoped = request_evidence(evidence_provider(item.source), scope)
    with cast(Any, scoped).source as pinned_provider:
        source = pinned_provider.source
        context = source.context
        identity = _plain(source_identity(source))
        provenance = _plain(artifact_provenance(source))
        methods = {
            config.metric.name: (
                []
                if config.methods_explicitly_empty
                else [config.decision_method, *config.sensitivity_methods]
            )
            for config in configs
        }
        unit_components = []
        method_evidence = {}
        if any(
            method.variance_reduction == "cuped_multicovariate"
            for metric_methods in methods.values()
            for method in metric_methods
        ):
            _frames, _covariates, _roles, unit_components = _load_cuped_breakout_frames(
                source,
                configs,
                methods,
                dimension,
                population=request.population,
                source_name=source_name,
            )
            method_evidence = {
                config.metric.name: _load_breakout_metric_rows(
                    source,
                    config.metric,
                    methods[config.metric.name],
                    dimension=dimension,
                    control_group=context.design.control_group,
                    source_name=source_name,
                    population=request.population,
                )
                for config in configs
            }
        if fits_uncovered_cells_locally(source):
            local_fit_metrics = frozenset(
                name
                for name, evidence in method_evidence.items()
                if any(answer.origin == "unavailable" for _, answer in evidence.method_fit_answers)
            )
            if local_fit_metrics:
                _frames, _covariates, _roles, fallback_components = _load_cuped_breakout_frames(
                    source,
                    configs,
                    methods,
                    dimension,
                    population=request.population,
                    source_name=source_name,
                    local_fit_metrics=local_fit_metrics,
                )
                unit_components.extend(fallback_components)
        entries: list[dict[str, Any]] = []
        by_metric = {config.metric.name: config for config in configs}
        for component in unit_components:
            config = by_metric[component["metric"]]
            unit_request = BreakoutUnitRowsRequest(
                metric=config.metric,
                breakout=BreakoutIdentity(dimension, source_name),
                population=request.population,
                covariates=tuple(covariate.name for covariate in config.covariates),
            )
            # A second read in the same pinned scope supplies the original wire frame.
            # Its transformed multiset must independently match the actual helper claim.
            answer = cast(Any, request_evidence(pinned_provider, unit_request))
            entries.append(
                {
                    "usage": "local_breakout_unit_frame",
                    "record": {
                        "request_type": "BreakoutUnitRowsRequest",
                        "request": _plain(unit_request),
                        "answer_type": "BreakoutUnitRows",
                        "answer": {
                            "identity": _plain(answer.identity),
                            "route": answer.route,
                            "frame": _frame_payload(answer.frame),
                            "unit_frame_has_dimension": answer.unit_frame_has_dimension,
                        },
                    },
                    "component": _plain(component),
                }
            )
        payloads = list(
            dict.fromkeys(
                [
                    receipt["payload"]
                    for receipt in format_inputs
                    if isinstance(receipt.get("payload"), str)
                ]
                + [
                    row["cuped_evidence"]
                    for evidence in method_evidence.values()
                    for row in evidence.rows
                    if isinstance(row.get("cuped_evidence"), str)
                ]
            )
        )
        for name, evidence in method_evidence.items():
            warehouse_cache: dict[Any, Any] = {}
            components = [
                component
                for component in evidence.evidence_components
                if component[0].startswith(
                    ("warehouse_qr_breakout:", "categorical_cells_breakout:")
                )
            ]
            component_index = 0
            for fit_request, answer in evidence.method_fit_answers:
                _acquire_warehouse_input(
                    source, fit_request, answer, warehouse_receipts, warehouse_cache
                )
                if (
                    fit_request.metric.name != name
                    or fit_request.population != request.population
                    or fit_request.cell is None
                    or fit_request.cell.breakout != BreakoutIdentity(dimension, source_name)
                ):
                    raise ValueError("breakout fit answer is not associated with this slice")
                component = None
                if answer.origin == "unavailable":
                    usage = "unavailable_breakout_fit"
                else:
                    categorical = answer.origin == "retained_categorical" and answer.failure is None
                    usage = (
                        "retained_categorical_breakout_fit"
                        if categorical
                        else "warehouse_breakout_fit"
                    )
                    kind, sha256, count = components[component_index]
                    component_index += 1
                    cell = fit_request.cell
                    value = "__null__" if cell.is_null else cell.value
                    prefix = (
                        "categorical_cells_breakout" if categorical else "warehouse_qr_breakout"
                    )
                    if kind != f"{prefix}:{fit_request.method.name}:{dimension}:{value}":
                        raise ValueError("breakout fit component does not match its acquired cell")
                    component = {
                        "kind": kind,
                        "metric": name,
                        "population": request.population,
                        "source": source_name,
                        "dimension": dimension,
                        "sha256": sha256,
                        "count": count,
                    }
                entries.append(
                    {
                        "usage": usage,
                        "record": {
                            "request_type": "MethodFitRequest",
                            "request": _plain(fit_request),
                            "answer_type": "MethodFitAnswer",
                            "answer": _plain(answer),
                        },
                        "component": component,
                    }
                )
            if component_index != len(components):
                raise ValueError("breakout fit components lack acquired answers")
        declarations = [_digest(entry["record"]["request"]) for entry in entries]
        if len(declarations) != len(set(declarations)):
            raise ValueError("duplicate breakout component input request")
        policy = context.plan.view_policies.for_view(
            "breakout", mechanism=context.design.mechanism, segmented=True
        )
        config_snapshots = []
        for config in sorted(configs, key=lambda config: config.metric.name):
            method_snapshots = []
            for method in methods[config.metric.name]:
                snapshot = method.model_dump(mode="python")
                if method.covariates is None:
                    snapshot.pop("covariates", None)
                method_snapshots.append(snapshot)
            snapshot = {
                "metric": config.metric.model_dump(mode="json"),
                "methods": method_snapshots,
                "prior": None if config.prior is None else config.prior.model_dump(mode="json"),
                "prior_is_global": config.prior_is_global,
            }
            if config.covariates:
                snapshot["covariates"] = [
                    covariate.model_dump(mode="json") for covariate in config.covariates
                ]
            config_snapshots.append(snapshot)
        return {
            "input_kind": input_kind,
            "identity": _plain(item.identity),
            "source_name": item.source_name,
            "stamp_source": item.stamp_source,
            "route": day_axis_route(source),
            "context": _plain(context),
            "source_identity": identity,
            "artifact_provenance": provenance,
            "request": _plain(
                {
                    "source_identity": identity,
                    "metrics": [
                        config.metric.model_dump(mode="json")
                        for config in sorted(configs, key=lambda config: config.metric.name)
                    ],
                    "configs": config_snapshots,
                    "dimension": dimension,
                    "estimands": estimands,
                    "correction": "bh" if policy.correction == "e_bh" else policy.correction,
                    "q": policy.q if policy.q is not None else context.plan.q,
                    "source_name": source_name,
                    "analysis_population": request.population,
                }
            ),
            "entries": entries,
            "categorical_payloads": payloads,
        }


def _day_capture_options(source, case, request, configuration):
    from increment._analysis_config import UNSET
    from increment._lift_options import LiftOptions

    request = replace(request, estimands=case.estimands)
    options = None
    if request.caller.endswith("_lift"):
        options = LiftOptions.resolve(
            source,
            request.metrics,
            decision_method=case.decision_method if case.decision_method is not None else UNSET,
            sensitivity_methods=UNSET,
            prior=case.prior,
            alpha=source.context.plan.alpha,
        )
        request = replace(
            request,
            covariate_metrics=frozenset(
                metric.name for metric in request.metrics if options.needs_covariate(metric)
            ),
        )
        configuration["day_axis_options"] = {
            "kwargs": _plain(options.daily_lift_kwargs()),
            "configs": _plain(options.configs),
        }
    return request, options


def _runtime_metadata():
    import importlib.metadata
    import platform
    import sys

    import increment

    versions = {}
    for distribution in ("increment", "pydantic", "numpy", "scipy", "pyarrow", "narwhals"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "production_root": str(Path(increment.__file__).resolve().parent),
        "dependency_versions": versions,
    }


def _ordinary_method_fit_refusals(source, provider, case, ingress, configs):
    if (
        _family(case) == "cuped"
        and ingress == "from_moments"
        and getattr(getattr(source, "evidence", None), "method_fit", "none") == "none"
    ):
        return _method_fit_refusals(
            source,
            provider,
            _method_fit_requests(source, case.population, configs=configs),
        )
    return ()


def _breakout_scope_receipts(source, slices, request, case, format_inputs, warehouse_receipts):
    scope_slices = tuple(slices.slices)
    summary_source = getattr(source, "summary_breakout_source", None)
    input_kind: Literal["materialized_breakout_slice", "summary_breakout_source_slice"] = (
        "materialized_breakout_slice"
    )
    if summary_source is not None:
        selected = cast(Any, request_evidence(evidence_provider(summary_source), request))
        scope_slices = tuple(selected.slices)
        input_kind = "summary_breakout_source_slice"
    return [
        _acquire_breakout_scope_inputs(
            item,
            request,
            input_kind=input_kind,
            configs=tuple(
                overlay_configs(
                    tuple(item.source.context.metrics),
                    item.source.context.configs,
                    methods=None,
                    prior=None,
                    decision_method=case.decision_method
                    if case.decision_method is not None
                    else UNSET,
                    prior_override=case.prior,
                )
            ),
            estimands=None,
            format_inputs=format_inputs,
            warehouse_receipts=warehouse_receipts,
        )
        for item in scope_slices
    ]


def _attach_warehouse_inputs(configuration, receipts):
    if receipts:
        configuration.setdefault("assembly", {})["warehouse_input_receipts"] = receipts


def acquire_scenario(  # noqa: PLR0915
    scenario_id: str, ingress: str, *, observe_membership: bool = True
) -> AcquiredScenario:
    """Acquire typed inputs from one declared ingress without invoking a readout."""
    if _frozen_dispositions()[(scenario_id, ingress)]["kind"] != "cells":
        raise ValueError("refusals and absences require observe_scenario_disposition")
    transform_prefix = (
        "asof_lift-"
        if scenario_id.startswith("asof_lift-")
        else "asof-"
        if scenario_id.startswith("asof-")
        else None
    )
    is_asof_transform = transform_prefix is not None
    base_scenario_id = (
        scenario_id[len(transform_prefix) :] if transform_prefix is not None else scenario_id
    )
    case = next((candidate for candidate in PARITY_CASES if candidate.id == base_scenario_id), None)
    if case is None:
        raise KeyError(f"unknown declared scenario: {scenario_id}")
    try:
        build = case.build[ingress]
    except KeyError as exc:
        raise ValueError(f"{scenario_id} has no source builder for ingress {ingress}") from exc
    bound = build()
    format_inputs = bound.moment_inputs if isinstance(bound, BoundWithMomentInputs) else ()
    stack = ExitStack()
    try:
        binding = binding_for(*bound)
        source = readout_source(binding, population=case.population)
        provider = evidence_provider(source)
        warehouse_receipts: list[dict[str, Any]] = []
        warehouse_inputs: tuple[list[dict[str, Any]], dict[Any, Any]] = (warehouse_receipts, {})
        source_refusals: list[dict[str, Any]] = []
        declaration_identity = getattr(source, "declaration_identity", None)
        source_key = _plain(declaration_identity()) if callable(declaration_identity) else None
        source_input_hashes = []
        original_inputs = _source_inputs(
            source,
            scenario_id,
            ingress,
            (),
            hash_audits=source_input_hashes,
        )

        def bound_inputs(requests):
            return [
                {
                    **record,
                    "configuration": {
                        **record["configuration"],
                        "request_coordinates": [_plain(request) for request in requests],
                    },
                }
                for record in original_inputs
            ]

        from increment._analysis_config import select_metrics
        from increment.kernel.contracts import contract_hash
        from tests.parity_harness.views import view_scopes

        public_view = case.view or ("breakout" if case.breakout_dimension else "run")
        public_scopes = view_scopes(
            bound[0],
            public_view,
            **(
                {"population": case.population, "dimension": case.breakout_dimension}
                if case.view is not None
                else {}
            ),
        )
        populations = tuple(dict.fromkeys(scope.population for scope in public_scopes))

        if ingress == "from_switchback_panel":
            metrics = tuple(source.context.metrics)
            consumer_context = _plain(source.context)
        else:
            metrics = tuple(select_metrics(source.context.metrics, case.metrics, caller="run"))
            configs = overlay_configs(
                metrics,
                source.context.configs,
                methods=None,
                prior=None,
                decision_method=case.decision_method if case.decision_method is not None else UNSET,
                prior_override=case.prior,
            )
            consumer_context = _plain(replace(source.context, metrics=metrics, configs=configs))
        from scripts.capture_evidence_identity import _population_identity

        compliance_only = (
            all(scope.grain == "total" for scope in public_scopes)
            and case.estimands is not None
            and set(case.estimands) == {"compliance"}
            and isinstance(source.context.design, Encouragement)
        )
        membership_metrics = () if compliance_only else tuple(metric.name for metric in metrics)
        source_membership_identity = (
            _population_identity(
                bound,
                case,
                (),
                population_selection=populations,
                metric_selection=membership_metrics,
            )
            if observe_membership
            else None
        )
        consumer_configuration: dict[str, Any] = {
            "context": consumer_context,
            "base_context": _plain(source.context),
            "format_inputs": _plain(format_inputs),
            "estimands": _plain(case.estimands),
            "view": case.view,
            "population": case.population,
            "scopes": _plain(public_scopes),
        }
        if ingress != "from_switchback_panel" and not (
            case.sequential
            or case.view is not None
            or is_asof_transform
            or scenario_id in TEMPORAL_IDS
        ):
            from increment.kernel.context import EvidenceExecutionContext

            ordinary_by_identity = {
                _digest(_plain(request)): request
                for population in populations
                for request in declared_requests(
                    source, _family(case), ingress, population, configs=configs
                )
            }
            ordinary_requests = tuple(ordinary_by_identity.values())
            design = source.context.design
            scope = EvidenceScope(
                grain="total",
                population=case.population,
                metrics=metrics,
                configs=configs,
                snapshot=True,
                breakouts=_breakout_identities(source, getattr(binding, "experiment", None)),
                populations=populations,
                assignment_requests=tuple(
                    request
                    for request in ordinary_requests
                    if isinstance(request, AssignmentRequest)
                ),
                uptake_facts=(design.uptake.fact,) if isinstance(design, Encouragement) else (),
            )
            execution_context = EvidenceExecutionContext(
                provider=provider,
                scope=scope,
                run_id=f"evidence-fixture:{scenario_id}:{ingress}",
            )
            stack.enter_context(execution_context.evidence_scope())
            provider = stack.enter_context(
                execution_context.borrow(source=source, metrics=metrics, configs=configs)
            )
            from increment.readouts._run import (
                _restore_legacy_prior_exclusion,
                _run_plan_and_roster,
            )

            plans_and_rosters = {
                population: _run_plan_and_roster(source, design, configs, population)
                for population in populations
            }
            consumer_configuration["assembly"] = {
                "source_identity": _plain(source_identity(source)),
                "artifact_provenance": _plain(artifact_provenance(source)),
                "identity_plan": _plain(
                    _restore_legacy_prior_exclusion(source, configs, source.context.plan)
                ),
                "calculation_plans": {
                    population: _plain(plan)
                    for population, (plan, _roster) in plans_and_rosters.items()
                },
                "rosters": {
                    population: _plain(roster)
                    for population, (_plan, roster) in plans_and_rosters.items()
                },
            }

        if case.sequential:
            from increment.kernel.context import EvidenceExecutionContext

            request = SequentialCaptureRequest(finalized=True, **_capture_kwargs(bound))
            scope = EvidenceScope(grain="total", population=case.population, metrics=metrics)
            execution_context = EvidenceExecutionContext(
                provider=provider,
                scope=scope,
                run_id=f"evidence-fixture:{scenario_id}:{ingress}",
            )
            provider = stack.enter_context(execution_context.evidence_scope())
            records = acquire_sequential(
                source,
                request,
                scope,
                execution_context,
                readout_scope=public_scopes[0],
                configs=configs,
                estimands=case.estimands,
            )
            requests = (request,)
        elif case.view is not None or scenario_id in TEMPORAL_IDS:
            from increment.kernel.context import EvidenceExecutionContext

            grain = (
                "daily"
                if case.view in {"daily", "daily_lift"}
                or scenario_id in {"triggered_daily", "triggered_daily_lift"}
                else "asof"
            )
            lift = case.view in {"daily_lift", "asof_lift"} or scenario_id.endswith("_lift")
            experiment = getattr(binding, "experiment", None)
            if grain == "daily":
                caller = "run_daily_lift" if lift else "run_daily"
            else:
                caller = "run_asof_lift" if lift else "run_asof"
            request = DayAxisRequest(
                caller=caller,
                grain=grain,
                metrics=metrics,
                dimension=case.breakout_dimension,
                population=case.population,
                experiment=experiment,
            )
            request, day_options = _day_capture_options(
                source,
                case,
                request,
                consumer_configuration,
            )
            scope = EvidenceScope(
                grain=grain,
                population=case.population,
                metrics=metrics,
                breakouts=_breakout_identities(source, experiment),
            )
            execution_context = EvidenceExecutionContext(
                provider=provider,
                scope=scope,
                run_id=f"evidence-fixture:{scenario_id}:{ingress}",
            )
            provider = stack.enter_context(execution_context.evidence_scope())
            records = acquire_day_axis(
                source, request, scope, execution_context, options=day_options
            )
            requests = (request,)
        elif is_asof_transform:
            from increment.kernel.context import EvidenceExecutionContext

            experiment = getattr(binding, "experiment", None)
            breakouts = _breakout_identities(source, experiment)
            day_request = DayAxisRequest(
                caller="run_asof_lift" if transform_prefix == "asof_lift-" else "run_asof",
                grain="asof",
                metrics=metrics,
                dimension=case.breakout_dimension,
                population=case.population,
                experiment=experiment,
            )
            day_request, day_options = _day_capture_options(
                source,
                case,
                day_request,
                consumer_configuration,
            )
            scope = EvidenceScope(
                grain="asof",
                population=case.population,
                metrics=metrics,
                breakouts=breakouts,
            )
            execution_context = EvidenceExecutionContext(
                provider=provider,
                scope=scope,
                run_id=f"evidence-fixture:{scenario_id}:{ingress}",
            )
            provider = stack.enter_context(execution_context.evidence_scope())
            day_records = acquire_day_axis(
                source,
                day_request,
                scope,
                execution_context,
                options=day_options,
            )
            base_requests = declared_requests(
                source, _family(case), ingress, case.population, configs=configs
            )
            breakout_request = next(
                (item for item in base_requests if isinstance(item, BreakoutSlicesRequest)), None
            )
            acquisition_requests = tuple(
                item for item in base_requests if item is not breakout_request
            )
            provider = stack.enter_context(
                execution_context.borrow(source=source, metrics=metrics, configs=configs)
            )
            ordinary = acquire_transcript(
                provider,
                scenario_id=scenario_id,
                ingress=ingress,
                requests=acquisition_requests,
                baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
                warehouse_inputs=warehouse_inputs,
                provenance={
                    "source_type": type(source).__qualname__,
                    "source_key": source_key,
                    "source_declaration_sha256": (
                        contract_hash("evidence_fixture_source_declaration", source_key)
                        if source_key is not None
                        else None
                    ),
                    "source_inputs": bound_inputs((day_request, *base_requests)),
                    "configuration_identities": [
                        _plain(item) for item in (day_request, *base_requests)
                    ],
                    "population": case.population,
                },
            )
            records = (*day_records, *ordinary.records)
            requests = (day_request, *base_requests)
            if breakout_request is not None:
                slices = cast(Any, request_evidence(provider, breakout_request))
                breakout_record = {
                    "request_type": "BreakoutSlicesRequest",
                    "request": _plain(breakout_request),
                    "answer_type": "MaterializedBreakoutSlices",
                    "answer": _plain(
                        {
                            "scope": {
                                "population": breakout_request.population,
                                "breakouts": breakout_request.breakouts,
                                "metrics": breakout_request.metrics,
                                "configs": breakout_request.configs,
                            },
                            "route": slices.route,
                            "slices": [
                                {
                                    "identity": item.identity,
                                    "source_name": item.source_name,
                                    "stamp_source": item.stamp_source,
                                }
                                for item in slices.slices
                            ],
                        }
                    ),
                }
                records_list = list(records)
                records_list.insert(1 + base_requests.index(breakout_request), breakout_record)
                records = tuple(records_list)
        elif ingress == "from_switchback_panel":
            from increment._evidence_scopes import switchback_scope_provider
            from increment.kernel.context import EvidenceExecutionContext

            metrics = tuple(source.context.metrics)
            requests = tuple(ContrastStatsRequest(metric) for metric in metrics)
            scope = EvidenceScope(grain="total", population="assigned", metrics=metrics)
            execution_context = EvidenceExecutionContext(
                provider=switchback_scope_provider(source),
                scope=scope,
                run_id=f"evidence-fixture:{scenario_id}:{ingress}",
            )
            stack.enter_context(execution_context.evidence_scope())
            records = tuple(
                acquire_contrast(source, request, execution_context) for request in requests
            )
        else:
            requests = ordinary_requests
            source_refusals.extend(
                _ordinary_method_fit_refusals(source, provider, case, ingress, configs)
            )
            breakout_request = next(
                (request for request in requests if isinstance(request, BreakoutSlicesRequest)),
                None,
            )
            acquisition_requests = requests
            breakout_record = None
            scoped_unit_records: dict[int, Mapping[str, Any]] = {}
            if breakout_request is not None:
                slices = cast(Any, request_evidence(provider, breakout_request))
                materialized_slices = tuple(slices.slices)
                scope_inputs = _breakout_scope_receipts(
                    source,
                    slices,
                    breakout_request,
                    case,
                    format_inputs,
                    warehouse_receipts,
                )
                if scope_inputs:
                    consumer_configuration["assembly"]["breakout_scope_inputs"] = scope_inputs
                if slices.route == "artifact":
                    scoped_unit_records = _artifact_breakout_unit_records(
                        materialized_slices,
                        requests,
                        scenario_id=scenario_id,
                        ingress=ingress,
                        source_key=source_key,
                    )
                breakout_record = {
                    "request_type": "BreakoutSlicesRequest",
                    "request": _plain(breakout_request),
                    "answer_type": "MaterializedBreakoutSlices",
                    "answer": _plain(
                        {
                            "scope": {
                                "population": breakout_request.population,
                                "breakouts": breakout_request.breakouts,
                                "metrics": breakout_request.metrics,
                                "configs": breakout_request.configs,
                            },
                            "route": slices.route,
                            "slices": [
                                {
                                    "identity": item.identity,
                                    "source_name": item.source_name,
                                    "stamp_source": item.stamp_source,
                                }
                                for item in materialized_slices
                            ],
                        }
                    ),
                }
                acquisition_requests = tuple(
                    request
                    for index, request in enumerate(requests)
                    if request is not breakout_request and index not in scoped_unit_records
                )
            transcript = acquire_transcript(
                provider,
                scenario_id=scenario_id,
                ingress=ingress,
                requests=acquisition_requests,
                baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
                warehouse_inputs=warehouse_inputs,
                provenance={
                    "source_type": type(source).__qualname__,
                    "source_key": source_key,
                    "consumer_configuration": consumer_configuration,
                    "coherence": {
                        "coherence": execution_context.coherence,
                        "effective_modes": dict(execution_context.effective_modes),
                    },
                    "source_declaration_sha256": (
                        contract_hash("evidence_fixture_source_declaration", source_key)
                        if source_key is not None
                        else None
                    ),
                    "source_inputs": bound_inputs(requests),
                    "configuration_identities": [_plain(request) for request in requests],
                    "population": case.population,
                    "metrics": tuple(config.metric.name for config in source.context.configs),
                    "source_refusals": source_refusals,
                },
            )
            _attach_warehouse_inputs(consumer_configuration, warehouse_receipts)
            transcript = replace(
                transcript,
                provenance={
                    **transcript.provenance,
                    "consumer_configuration": consumer_configuration,
                },
            )
            if breakout_record is not None or scoped_unit_records:
                original_records = iter(transcript.records)
                records_list = [
                    breakout_record
                    if request is breakout_request
                    else scoped_unit_records[index]
                    if index in scoped_unit_records
                    else next(original_records)
                    for index, request in enumerate(requests)
                ]
                transcript = replace(
                    transcript,
                    records=tuple(records_list),
                    provenance={
                        key: value
                        for key, value in transcript.provenance.items()
                        if key not in {"declaration_sha256", "input_sha256"}
                    },
                )
            payload = transcript.to_payload()
            Transcript.from_payload(
                payload,
                expected_requests=requests,
                expected_baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
                expected_scenario_id=scenario_id,
                expected_ingress=ingress,
            )
            records = transcript.records
            actual = _actual_representations(records)
            required = _requirements(case, _family(case), ingress)
            missing = _missing_representations(required, actual, source_refusals)
            return AcquiredScenario(
                scenario_id,
                ingress,
                _family(case),
                records,
                transcript.provenance,
                payload["declaration_sha256"],
                payload["input_sha256"],
                actual,
                missing,
                tuple(source_input_hashes),
                source_membership_identity,
            )

        if (
            case.sequential
            or case.view is not None
            or scenario_id in TEMPORAL_IDS
            or ingress == "from_switchback_panel"
        ):
            count_populations = (case.population,) if case.sequential else populations
            auxiliary: list[object] = (
                []
                if ingress == "from_switchback_panel"
                else [
                    CountsRequest(kind="unit", population=population)
                    for population in count_populations
                ]
            )
            family = _family(case)
            if case.sequential:
                metric_requests, metric_refusals = _sequential_metric_inputs(
                    source,
                    provider,
                    _metric_requests(source, case.population, configs=configs),
                    records[0]["answer"]["checkpoint"],
                )
                auxiliary.extend(metric_requests)
                source_refusals.extend(metric_refusals)
                method_fit_requests = _method_fit_requests(source, case.population, configs=configs)
                method_fit = getattr(getattr(source, "evidence", None), "method_fit", "none")
                if ingress == "from_moments" and method_fit == "none":
                    source_refusals.extend(
                        _method_fit_refusals(source, provider, method_fit_requests)
                    )
                elif method_fit != "none":
                    auxiliary.extend(method_fit_requests)
            if ingress == "from_switchback_panel" or family in {"cluster", "encouragement"}:
                auxiliary.append(
                    AssignmentRequest(
                        experiment=source.context.study_id,
                        population="assigned"
                        if ingress == "from_switchback_panel"
                        else case.population,
                        all_periods=ingress == "from_switchback_panel",
                    )
                )
            if family == "encouragement" and (
                not case.sequential
                or getattr(getattr(source, "evidence", None), "compliance_axis", "none") != "none"
            ):
                auxiliary.append(
                    ComplianceRequest(
                        design=cast(Encouragement, source.context.design),
                        as_of=_capture_kwargs(bound).get(
                            "as_of",
                            getattr(
                                getattr(binding, "experiment", None),
                                "observation_horizon_day",
                                None,
                            ),
                        ),
                        population=case.population,
                    )
                )
            supplemental = acquire_transcript(
                provider,
                scenario_id=scenario_id,
                ingress=ingress,
                requests=tuple(auxiliary),
                baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
                warehouse_inputs=warehouse_inputs,
                provenance={
                    "source_type": type(source).__qualname__,
                    "source_key": source_key,
                    "source_declaration_sha256": contract_hash(
                        "evidence_fixture_source_declaration", source_key
                    ),
                    "source_inputs": bound_inputs((*requests, *auxiliary)),
                    "configuration_identities": [_plain(item) for item in (*requests, *auxiliary)],
                    "population": case.population,
                    "source_refusals": source_refusals,
                },
            )
            records = (*records, *supplemental.records)
            requests = (*requests, *auxiliary)
        if source_key is None:
            raise TypeError(f"source declaration identity missing for {scenario_id}/{ingress}")
        source_inputs = bound_inputs(requests)
        _attach_warehouse_inputs(consumer_configuration, warehouse_receipts)
        provenance = {
            "source_type": type(source).__qualname__,
            "source_key": source_key,
            "consumer_configuration": consumer_configuration,
            "coherence": {
                "coherence": execution_context.coherence,
                "effective_modes": dict(execution_context.effective_modes),
            },
            "source_declaration_sha256": contract_hash(
                "evidence_fixture_source_declaration", source_key
            ),
            "source_inputs": source_inputs,
            "configuration_identities": [_plain(request) for request in requests],
            "population": case.population,
            "source_refusals": source_refusals,
        }
        transcript = Transcript(
            baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
            scenario_id=scenario_id,
            ingress=ingress,
            runtime=_runtime_metadata(),
            provenance=provenance,
            records=records,
        )
        payload = transcript.to_payload()
        Transcript.from_payload(
            payload,
            expected_requests=requests,
            expected_baseline_revision="7c324b66f9f758eb78298e6ecf2d1aea29c66709",
            expected_scenario_id=scenario_id,
            expected_ingress=ingress,
        )
        actual = _actual_representations(records)
        required = (
            ("DAX", "B", "ME", "N")
            if is_asof_transform
            else _requirements(case, _family(case), ingress)
        )
        missing = _missing_representations(required, actual, source_refusals)
        return AcquiredScenario(
            scenario_id,
            ingress,
            _family(case),
            records,
            payload["provenance"],
            payload["declaration_sha256"],
            payload["input_sha256"],
            actual,
            missing,
            tuple(source_input_hashes),
            source_membership_identity,
        )
    finally:
        stack.close()
        _close_parity_binding(bound)


def assert_roster_exact(
    expected: Iterable[ScenarioIngress], actual: Iterable[ScenarioIngress]
) -> None:
    """Reject missing, duplicate, or unexpected declared coordinates."""
    expected_rows, actual_rows = tuple(expected), tuple(actual)
    if len(set(expected_rows)) != len(expected_rows):
        raise ValueError("duplicate expected scenario coordinate")
    if len(set(actual_rows)) != len(actual_rows):
        raise ValueError("duplicate actual scenario coordinate")
    missing = set(expected_rows) - set(actual_rows)
    unexpected = set(actual_rows) - set(expected_rows)
    if missing or unexpected:
        raise ValueError(f"scenario roster mismatch: missing={missing}, unexpected={unexpected}")
