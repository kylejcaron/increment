from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

import increment
from increment._evidence_types import (
    BreakoutSlicesRequest,
    BreakoutUnitRows,
    BreakoutUnitRowsRequest,
    ContrastStatsRequest,
    DayAxisRequest,
    MetricEvidence,
    MetricEvidenceRequest,
)
from increment.errors import CodedError, RefusalSpec
from increment.evidence import (
    AssignmentEvidence,
    AssignmentRequest,
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    ComplianceSummary,
    CountEvidence,
    CountsRequest,
    MethodFitAnswer,
    MethodFitRequest,
    MomentsRequest,
    ScalarBreakoutRows,
    SequentialCaptureRequest,
    UnitRowsRequest,
    request_evidence,
)

_TYPED_MAPPING = "$transcript_mapping"
_TYPED_TUPLE = "$transcript_tuple"
_TYPED_DECIMAL = "$transcript_decimal"


_SCHEMA_VERSION = 2

_REQUEST_ANSWERS = {
    "MetricEvidenceRequest": "MetricEvidence",
    "MomentsRequest": "MomentRows",
    "CountsRequest": "CountEvidence",
    "AssignmentRequest": "AssignmentEvidence",
    "UnitRowsRequest": "UnitRows",
    "MethodFitRequest": "MethodFitAnswer",
    "DayAxisRequest": "MaterializedDayAxisRows",
    "SequentialCaptureRequest": "SequentialCapture",
    "ContrastStatsRequest": "ContrastStats",
    "ComplianceRequest": "ComplianceSummary",
    "ComplianceSeriesRequest": "ComplianceSeries",
    "BreakoutSlicesRequest": "MaterializedBreakoutSlices",
    "BreakoutMethodMomentsRequest": "ScalarBreakoutRows",
    "BreakoutUnitRowsRequest": "BreakoutUnitRows",
}
_REQUEST_CLASSES = {
    cls.__name__: cls
    for cls in (
        MetricEvidenceRequest,
        MomentsRequest,
        CountsRequest,
        AssignmentRequest,
        UnitRowsRequest,
        MethodFitRequest,
        DayAxisRequest,
        SequentialCaptureRequest,
        ContrastStatsRequest,
        ComplianceRequest,
        ComplianceSeriesRequest,
        BreakoutSlicesRequest,
        BreakoutMethodMomentsRequest,
        BreakoutUnitRowsRequest,
    )
}
_REQUEST_TYPES = frozenset(_REQUEST_CLASSES)
_DIRECT_CAPTURE_REQUEST_TYPES = frozenset(
    {
        "DayAxisRequest",
        "SequentialCaptureRequest",
        "ContrastStatsRequest",
        "ComplianceRequest",
        "ComplianceSeriesRequest",
        "BreakoutSlicesRequest",
        "BreakoutMethodMomentsRequest",
        "BreakoutUnitRowsRequest",
    }
)
_ANSWER_FIELDS = {
    "MetricEvidence": frozenset(
        {
            "rows",
            "unit_frame",
            "observed_arms",
            "n_treatment_arms",
            "evidence_kind",
            "evidence_sha256",
            "evidence_count",
            "decision_evidence_components",
            "cluster",
            "control_group",
            "evidence_components",
            "covariate_columns",
            "method_rows",
            "method_refusals",
            "method_cell_refusals",
            "method_fit_answers",
        }
    ),
    "CountEvidence": frozenset({"population", "grain", "counts", "unit_counts"}),
    "AssignmentEvidence": frozenset(
        {
            "experiment",
            "population",
            "versions",
            "periods",
            "arm_counts",
            "independence_grain",
            "probabilities",
            "replay_status",
            "order_draws",
            "randomization_law",
        }
    ),
    "MomentRows": None,
    "UnitRows": frozenset({"frame"}),
    "MethodFitAnswer": frozenset(
        {"origin", "envelope", "cells", "failure", "fit_identity", "evidence_sha256"}
    ),
    "MaterializedDayAxisRows": frozenset(
        {
            "scope",
            "route",
            "slices",
            "arm_roster",
            "compliance_series",
            "scope_request",
            "artifact_provenance",
        }
    ),
    "SequentialCapture": frozenset(
        {"scope", "checkpoint", "roster", "source_identity", "breakout"}
    ),
    "ContrastStats": frozenset({"scope", "value"}),
    "ComplianceSummary": frozenset(
        {
            "version",
            "study_id",
            "cohort",
            "window_days",
            "one_sided",
            "cluster",
            "as_of",
            "arms",
            "control_group",
        }
    ),
    "ComplianceSeries": None,
    "MaterializedBreakoutSlices": frozenset({"scope", "route", "slices"}),
    "ScalarBreakoutRows": frozenset({"rows", "refusals"}),
    "BreakoutUnitRows": frozenset({"identity", "route", "frame", "unit_frame_has_dimension"}),
}
_TYPED_ANSWER_CLASSES = {
    "MetricEvidence": MetricEvidence,
    "CountEvidence": CountEvidence,
    "AssignmentEvidence": AssignmentEvidence,
    "MethodFitAnswer": MethodFitAnswer,
    "ComplianceSummary": ComplianceSummary,
    "ScalarBreakoutRows": ScalarBreakoutRows,
    "BreakoutUnitRows": BreakoutUnitRows,
}
_TYPED_WIRE_EXTRA_FIELDS = {
    "ComplianceSummary": frozenset({"version"}),
}
for _answer_type, _answer_class in _TYPED_ANSWER_CLASSES.items():
    declared_fields = frozenset(field.name for field in fields(_answer_class))
    declared_fields |= _TYPED_WIRE_EXTRA_FIELDS.get(_answer_type, frozenset())
    if _ANSWER_FIELDS[_answer_type] != declared_fields:
        raise RuntimeError(f"{_answer_type} transcript field inventory is incomplete")
if frozenset(field.name for field in fields(RefusalSpec)) != frozenset(
    {"code", "error_type", "render", "template", "keys"}
):
    raise RuntimeError("RefusalSpec transcript field inventory needs review")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "baseline_revision",
        "scenario_id",
        "ingress",
        "runtime",
        "provenance",
        "declaration_sha256",
        "input_sha256",
        "records",
    }
)
_RECORD_FIELDS = frozenset({"request_type", "request", "answer_type", "answer"})


def _frame_payload(value: Any) -> dict[str, Any]:
    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module == "narwhals":
        to_native = getattr(value, "to_native", None)
        if not callable(to_native):
            raise TypeError(f"Unsupported tabular transcript carrier: {type(value).__qualname__}")
        native = to_native()
        if native is value:
            raise TypeError("Narwhals frame did not expose a native tabular carrier")
        return _frame_payload(native)
    if module == "polars" and hasattr(value, "to_dicts"):
        columns = list(value.columns)
        schema = {name: str(dtype) for name, dtype in value.schema.items()}
        rows = value.to_dicts()
    elif module == "pandas" and hasattr(value, "to_dict"):
        columns = list(value.columns)
        schema = {str(name): str(dtype) for name, dtype in value.dtypes.items()}
        rows = value.to_dict(orient="records")
    elif module == "pyarrow" and hasattr(value, "schema"):
        columns = list(value.schema.names)
        schema = {field.name: str(field.type) for field in value.schema}
        if hasattr(value, "to_pylist"):
            rows = value.to_pylist()
        else:
            columns_data = [value[column].to_pylist() for column in columns]
            rows = [dict(zip(columns, row, strict=True)) for row in zip(*columns_data, strict=True)]
    else:
        raise TypeError(f"Unsupported tabular transcript carrier: {type(value).__qualname__}")
    return {"columns": columns, "schema": schema, "rows": _plain(rows)}


def _typed_declaration(record: Mapping[str, Any]) -> list[Any]:
    return [record["request_type"], record["request"]]


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {_TYPED_DECIMAL: str(value)}
    if isinstance(value, Enum):
        from increment.estimation.cuped.encoding import categorical_missing_wire_value

        missing_category = categorical_missing_wire_value(value)
        if missing_category is not None:
            return missing_category
        raise TypeError(f"Unsupported enum transcript carrier: {type(value).__qualname__}")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value:
            return {"$transcript_float": "nan"}
        if value in (float("inf"), float("-inf")):
            return {"$transcript_float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, RefusalSpec):
        if value.template is None:
            raise TypeError("render-based refusal specs are not plain transcript values")
        return {
            "code": value.code,
            "error_type": f"{value.error_type.__module__}.{value.error_type.__qualname__}",
            "template": value.template,
            "keys": sorted(value.keys),
        }
    if isinstance(value, CodedError):
        return {"code": value.code, "message": str(value), "context": _plain(value.context)}
    from increment.decision import CompiledDecisionPlan

    if isinstance(value, CompiledDecisionPlan):
        from increment.decision_wire import compiled_plan_to_dict

        return _plain(compiled_plan_to_dict(value))
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="python"))
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        encoded = [(_plain_key(key), _plain(item)) for key, item in value.items()]
        if all(isinstance(key, str) for key in value):
            if len(value) == 1 and next(iter(value)) in {
                _TYPED_MAPPING,
                _TYPED_TUPLE,
                _TYPED_DECIMAL,
                "$transcript_float",
            }:
                return {_TYPED_MAPPING: [[key, item] for key, item in sorted(encoded)]}
            return dict(sorted(encoded))
        encoded.sort(key=lambda pair: _canonical(pair[0]))
        return {_TYPED_MAPPING: [[key, item] for key, item in encoded]}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=_canonical)
    if hasattr(value, "item"):
        return _plain(value.item())
    if hasattr(value, "to_pylist"):
        return _plain(value.to_pylist())
    raise TypeError(f"No plain transcript encoding for {type(value).__qualname__}")


def _plain_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return {_TYPED_TUPLE: [_plain_key(item) for item in value]}
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return _plain(value)
    raise TypeError(f"Unsupported transcript mapping key: {type(value).__qualname__}")


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if value.keys() == {"$transcript_float"}:
            return {
                "nan": float("nan"),
                "inf": float("inf"),
                "-inf": float("-inf"),
            }[value["$transcript_float"]]
        if value.keys() == {_TYPED_TUPLE}:
            items = value[_TYPED_TUPLE]
            if not isinstance(items, list):
                raise ValueError("typed transcript tuple items must be a list")
            return tuple(_restore(item) for item in items)
        if value.keys() == {_TYPED_DECIMAL}:
            text = value[_TYPED_DECIMAL]
            if not isinstance(text, str):
                raise ValueError("typed transcript decimal must be encoded as text")
            return Decimal(text)
        if value.keys() == {_TYPED_MAPPING}:
            pairs = value[_TYPED_MAPPING]
            if not isinstance(pairs, list):
                raise ValueError("typed transcript mapping pairs must be a list")
            restored = {}
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("typed transcript mapping entry must be a pair")
                key = _restore(pair[0])
                try:
                    if key in restored:
                        raise ValueError("duplicate typed transcript mapping key")
                    restored[key] = _restore(pair[1])
                except TypeError as exc:
                    raise ValueError("unhashable typed transcript mapping key") from exc
            return restored
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _is_plain_json(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int, float):
        return True
    if isinstance(value, list):
        return all(_is_plain_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_plain_json(item) for key, item in value.items())
    return False


def _validate_frame_payload(frame: Any) -> None:
    if (
        not isinstance(frame, dict)
        or frame.keys() != {"columns", "schema", "rows"}
        or not isinstance(frame["columns"], list)
        or not all(isinstance(column, str) for column in frame["columns"])
        or not isinstance(frame["schema"], dict)
        or set(frame["schema"]) != set(frame["columns"])
        or not all(isinstance(dtype, str) for dtype in frame["schema"].values())
        or not isinstance(frame["rows"], list)
        or any(not isinstance(row, dict) for row in frame["rows"])
    ):
        raise ValueError("frame fields are incomplete or unexpected")


def _validate_answer_payload(answer_type: str, answer: Any) -> None:
    try:
        _restore(answer)
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError("answer contains malformed typed transcript data") from exc
    answer_fields = _ANSWER_FIELDS[answer_type]
    if answer_fields is None:
        if not isinstance(answer, list):
            raise ValueError(f"{answer_type} answer must be a list")
        if answer_type == "ComplianceSeries":
            _validate_compliance_series(answer)
        return
    if not isinstance(answer, dict) or answer.keys() != answer_fields:
        raise ValueError("answer fields are incomplete or unexpected")
    if not _is_plain_json(answer):
        raise ValueError("answer contains a non-plain value")
    validator = _SPECIAL_ANSWER_VALIDATORS.get(answer_type)
    if validator is not None:
        validator(answer)


def _valid_refusal_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.keys() == {"code", "error_type", "template", "keys"}
        and isinstance(value["code"], str)
        and isinstance(value["error_type"], str)
        and isinstance(value["template"], str)
        and isinstance(value["keys"], list)
        and all(isinstance(key, str) for key in value["keys"])
    )


def _validate_metric_evidence(answer: dict[str, Any]) -> None:
    if any(
        not isinstance(item, list)
        or len(item) != 3
        or not isinstance(item[0], str)
        or not _valid_refusal_payload(item[1])
        or not isinstance(item[2], dict)
        for item in answer["method_refusals"]
    ):
        raise ValueError("metric method refusal fields are invalid")
    if any(
        not isinstance(item, list)
        or len(item) != 5
        or not all(isinstance(item[index], str) for index in (0, 1, 2))
        or not _valid_refusal_payload(item[3])
        or not isinstance(item[4], dict)
        for item in answer["method_cell_refusals"]
    ):
        raise ValueError("metric cell refusal fields are invalid")
    request_fields = frozenset(field.name for field in fields(MethodFitRequest))
    for pair in answer["method_fit_answers"]:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], dict)
            or pair[0].keys() != request_fields
            or not isinstance(pair[1], dict)
            or pair[1].keys() != _ANSWER_FIELDS["MethodFitAnswer"]
        ):
            raise ValueError("metric method-fit answer fields are invalid")


def _validate_compliance_summary(summary: Any) -> None:
    fields_expected = {
        "version",
        "study_id",
        "cohort",
        "window_days",
        "one_sided",
        "cluster",
        "as_of",
        "arms",
        "control_group",
    }
    arm_fields = {
        "group_id",
        "n_units",
        "uptake_total",
        "n_clusters",
        "ref_uptake",
        "cluster_uptake1",
        "cluster_uptake2",
        "ref_size",
        "cluster_size1",
        "cluster_size2",
        "cluster_cross",
    }
    if (
        not isinstance(summary, dict)
        or summary.keys() != fields_expected
        or type(summary["version"]) is not int
        or summary["version"] != 1
        or not isinstance(summary["arms"], list)
        or any(not isinstance(arm, dict) or arm.keys() != arm_fields for arm in summary["arms"])
    ):
        raise ValueError("compliance summary fields are incomplete or unexpected")


def _validate_compliance_series(series: list[Any]) -> None:
    for point in series:
        if (
            not isinstance(point, dict)
            or point.keys() != {"day", "summary"}
            or not isinstance(point["day"], str)
        ):
            raise ValueError("compliance series points are invalid")
        _validate_compliance_summary(point["summary"])


def _valid_observation_edges(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict)
        and all(
            isinstance(name, str)
            and isinstance(edges, list)
            and len(edges) == 2
            and edges[0] is not None
            and isinstance(edges[0], str)
            and (edges[1] is None or isinstance(edges[1], str))
            for name, edges in value.items()
        )
    )


def _validate_day_axis(answer: dict[str, Any]) -> None:
    if (
        not isinstance(answer["scope"], dict)
        or not isinstance(answer["route"], str)
        or not isinstance(answer["slices"], list)
        or (
            answer["arm_roster"] is not None
            and (
                not isinstance(answer["arm_roster"], list)
                or any(not isinstance(arm, str) for arm in answer["arm_roster"])
            )
        )
        or not isinstance(answer["scope_request"], dict)
        or not isinstance(answer["artifact_provenance"], dict)
        or answer["artifact_provenance"].keys() != {"manifest_sha256", "relation_digests"}
        or (
            answer["compliance_series"] is not None
            and (
                not isinstance(answer["compliance_series"], dict)
                or answer["compliance_series"].keys() != {"request", "answer"}
            )
        )
    ):
        raise ValueError("day-axis answer fields are invalid")
    fields_expected = {
        "metrics",
        "dimension",
        "source_name",
        "observation_edges",
        "experiment",
        "rows",
        "view",
        "consumer_rows",
    }
    for item in answer["slices"]:
        if (
            not isinstance(item, dict)
            or item.keys() != fields_expected
            or not isinstance(item["metrics"], list)
            or (item["dimension"] is not None and not isinstance(item["dimension"], str))
            or (item["source_name"] is not None and not isinstance(item["source_name"], str))
            or not _valid_observation_edges(item["observation_edges"])
            or not isinstance(item["rows"], list)
            or any(not isinstance(row, dict) for row in item["rows"])
            or item["view"] not in (None, "daily", "asof", "cohort")
            or not isinstance(item["consumer_rows"], list)
            or any(not isinstance(row, dict) for row in item["consumer_rows"])
        ):
            raise ValueError("day-axis slice fields are incomplete or unexpected")


def _validate_sequential(answer: dict[str, Any]) -> None:
    if (
        not isinstance(answer["scope"], dict)
        or not isinstance(answer["checkpoint"], dict)
        or not isinstance(answer["source_identity"], dict)
        or not isinstance(answer["roster"], dict)
        or answer["roster"].keys()
        != {
            "arms",
            "source",
            "complete",
            "counts",
            "integrity_counts",
            "arms_by_metric",
            "known_arms",
        }
        or (
            answer["breakout"] is not None
            and (
                not isinstance(answer["breakout"], dict)
                or answer["breakout"].keys()
                != {
                    "route",
                    "artifact_provenance",
                    "scope_request",
                    "source_components",
                }
            )
        )
    ):
        raise ValueError("sequential capture fields are invalid")


def _validate_contrast(answer: dict[str, Any]) -> None:
    if not isinstance(answer["scope"], dict) or not isinstance(answer["value"], dict):
        raise ValueError("contrast statistics fields are invalid")


def _validate_materialized_breakouts(answer: dict[str, Any]) -> None:
    scope = answer["scope"]
    if (
        not isinstance(scope, dict)
        or scope.keys() != {"population", "breakouts", "metrics", "configs"}
        or not isinstance(scope["population"], str)
        or not isinstance(scope["breakouts"], list)
        or not isinstance(scope["metrics"], list)
        or not isinstance(scope["configs"], list)
        or not isinstance(answer["route"], str)
        or not isinstance(answer["slices"], list)
    ):
        raise ValueError("materialized breakout fields are invalid")
    for item in answer["slices"]:
        if (
            not isinstance(item, dict)
            or item.keys() != {"identity", "source_name", "stamp_source"}
            or not isinstance(item["identity"], dict)
            or item["identity"].keys() != {"property", "source"}
            or not isinstance(item["identity"]["property"], str)
            or (
                item["identity"]["source"] is not None
                and not isinstance(item["identity"]["source"], str)
            )
            or (item["source_name"] is not None and not isinstance(item["source_name"], str))
            or (item["stamp_source"] is not None and not isinstance(item["stamp_source"], str))
        ):
            raise ValueError("materialized breakout slice fields are invalid")


def _validate_scalar_breakout_rows(answer: dict[str, Any]) -> None:
    if (
        not isinstance(answer["rows"], list)
        or any(not isinstance(row, dict) for row in answer["rows"])
        or not isinstance(answer["refusals"], list)
    ):
        raise ValueError("scalar breakout rows are invalid")
    refusal_fields = {"code", "error_type", "template", "keys"}
    for refusal in answer["refusals"]:
        if (
            not isinstance(refusal, list)
            or len(refusal) != 4
            or not isinstance(refusal[0], str)
            or not isinstance(refusal[1], str)
            or not isinstance(refusal[2], dict)
            or refusal[2].keys() != refusal_fields
            or not all(
                isinstance(refusal[2][field], str) for field in ("code", "error_type", "template")
            )
            or not isinstance(refusal[2]["keys"], list)
            or not all(isinstance(key, str) for key in refusal[2]["keys"])
            or not isinstance(refusal[3], dict)
        ):
            raise ValueError("scalar breakout refusal fields are invalid")


def _validate_breakout_unit_rows(answer: dict[str, Any]) -> None:
    identity = answer["identity"]
    if (
        not isinstance(identity, dict)
        or identity.keys() != {"property", "source"}
        or not isinstance(identity["property"], str)
        or (identity["source"] is not None and not isinstance(identity["source"], str))
        or not isinstance(answer["route"], str)
        or type(answer["unit_frame_has_dimension"]) is not bool
    ):
        raise ValueError("breakout unit-row fields are invalid")
    _validate_frame_payload(answer["frame"])


_SPECIAL_ANSWER_VALIDATORS = {
    "MetricEvidence": _validate_metric_evidence,
    "MaterializedDayAxisRows": _validate_day_axis,
    "SequentialCapture": _validate_sequential,
    "ContrastStats": _validate_contrast,
    "ComplianceSummary": _validate_compliance_summary,
    "MaterializedBreakoutSlices": _validate_materialized_breakouts,
    "ScalarBreakoutRows": _validate_scalar_breakout_rows,
    "BreakoutUnitRows": _validate_breakout_unit_rows,
}


def _request_type(request: Any) -> str:
    return type(request).__name__


def _encode_answer(answer: Any, request_type: str) -> tuple[str, Any]:
    if request_type == "BreakoutSlicesRequest":
        raise TypeError("BreakoutSlices must be materialized without retaining source objects")
    if request_type == "ComplianceSeriesRequest":
        if not isinstance(answer, tuple):
            raise TypeError("ComplianceSeries answer must be a typed series tuple")
        return "ComplianceSeries", [
            {"day": _plain(day), "summary": _plain(summary.to_wire())} for day, summary in answer
        ]
    if isinstance(answer, MetricEvidence):
        payload = {
            "rows": _plain(answer.rows),
            "unit_frame": None if answer.unit_frame is None else _frame_payload(answer.unit_frame),
            "observed_arms": sorted(answer.observed_arms),
            "n_treatment_arms": answer.n_treatment_arms,
            "evidence_kind": answer.evidence_kind,
            "evidence_sha256": answer.evidence_sha256,
            "evidence_count": answer.evidence_count,
            "decision_evidence_components": _plain(answer.decision_evidence_components),
            "cluster": answer.cluster,
            "control_group": answer.control_group,
            "evidence_components": _plain(answer.evidence_components),
            "covariate_columns": _plain(answer.covariate_columns),
            "method_rows": _plain(answer.method_rows),
            "method_refusals": _plain(answer.method_refusals),
            "method_cell_refusals": _plain(answer.method_cell_refusals),
            "method_fit_answers": _plain(answer.method_fit_answers),
        }
        return "MetricEvidence", payload
    if isinstance(answer, CountEvidence):
        return "CountEvidence", _plain(answer)
    if isinstance(answer, AssignmentEvidence):
        return "AssignmentEvidence", _plain(answer)
    if isinstance(answer, MethodFitAnswer):
        return "MethodFitAnswer", _plain(answer)
    if isinstance(answer, ComplianceSummary):
        return "ComplianceSummary", _plain(answer.to_wire())
    if isinstance(answer, ScalarBreakoutRows):
        return "ScalarBreakoutRows", _plain(answer)
    if isinstance(answer, BreakoutUnitRows):
        return (
            "BreakoutUnitRows",
            {
                "identity": _plain(answer.identity),
                "route": _plain(answer.route),
                "frame": _frame_payload(answer.frame),
                "unit_frame_has_dimension": answer.unit_frame_has_dimension,
            },
        )
    if isinstance(answer, tuple) and all(isinstance(row, Mapping) for row in answer):
        return "MomentRows", _plain(answer)
    if type(answer).__module__.split(".", maxsplit=1)[0] in {"polars", "pandas", "pyarrow"}:
        return "UnitRows", {"frame": _frame_payload(answer)}
    raise TypeError(f"Unsupported acquired evidence answer: {type(answer).__qualname__}")


def _validate_record(record: Any) -> None:
    if not isinstance(record, dict) or record.keys() != _RECORD_FIELDS:
        raise ValueError("record fields are incomplete or unexpected")
    request_type = record["request_type"]
    if request_type not in _REQUEST_TYPES:
        raise ValueError("unsupported request_type")
    answer_type = record["answer_type"]
    if answer_type not in _ANSWER_FIELDS:
        raise ValueError("unsupported answer_type")
    if answer_type != _REQUEST_ANSWERS[request_type]:
        raise ValueError("request and answer type mismatch")
    _validate_answer_payload(answer_type, record["answer"])


def _bind_provenance(
    value: Mapping[str, Any],
    authority: Mapping[str, str],
    *,
    require_source_inputs: bool = False,
) -> dict[str, Any]:
    configuration = value.get("consumer_configuration")
    assembly = configuration.get("assembly") if isinstance(configuration, Mapping) else None
    closed_inputs = {}
    if isinstance(assembly, Mapping):
        for key in ("breakout_scope_inputs", "warehouse_input_receipts"):
            if key in assembly:
                inputs = assembly[key]
                if not isinstance(inputs, list) or not _is_plain_json(inputs):
                    raise ValueError(f"{key} must contain closed wire values")
                closed_inputs[key] = inputs
    if closed_inputs:
        value = deepcopy(value)
        for key in closed_inputs:
            del value["consumer_configuration"]["assembly"][key]
    provenance = _plain(value)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a plain mapping")
    for key, inputs in closed_inputs.items():
        provenance["consumer_configuration"]["assembly"][key] = deepcopy(inputs)
    for field, expected in authority.items():
        if field in provenance and provenance[field] != expected:
            raise ValueError(f"provenance {field} mismatch")
        provenance[field] = expected
    source_inputs = provenance.get("source_inputs")
    if require_source_inputs and not source_inputs:
        raise ValueError("direct capture provenance requires source_inputs")
    if source_inputs is not None:
        if not isinstance(source_inputs, list) or any(
            not _valid_source_input(item) for item in source_inputs
        ):
            raise ValueError("source_inputs provenance is invalid")
    if require_source_inputs:
        source_declaration = provenance.get("source_declaration_sha256")
        if (
            not isinstance(provenance.get("source_type"), str)
            or not provenance["source_type"]
            or not isinstance(provenance.get("source_key"), dict)
            or not isinstance(source_declaration, str)
            or len(source_declaration) != 64
            or any(character not in "0123456789abcdef" for character in source_declaration)
            or not isinstance(provenance.get("configuration_identities"), list)
        ):
            raise ValueError("direct capture source declaration provenance is invalid")
        from increment.kernel.contracts import contract_hash

        if (
            contract_hash("evidence_fixture_source_declaration", provenance["source_key"])
            != source_declaration
        ):
            raise ValueError("direct capture source declaration digest mismatch")
    return provenance


def _valid_source_input(value: Any) -> bool:
    if not isinstance(value, dict) or not _is_plain_json(value):
        return False
    if "sha256" not in value or not isinstance(value["sha256"], str):
        return False
    if len(value["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in value["sha256"]
    ):
        return False
    if value.keys() == {"path", "sha256"}:
        return isinstance(value["path"], str) and bool(value["path"])
    return (
        value.keys() == {"kind", "identity", "configuration", "sha256"}
        and isinstance(value["kind"], str)
        and bool(value["kind"])
        and isinstance(value["identity"], str)
        and bool(value["identity"])
        and isinstance(value["configuration"], dict)
    )


def _validate_direct_provenance(
    provenance: Mapping[str, Any],
    requests: tuple[object, ...],
    scenario_id: str,
    ingress: str,
) -> None:
    expected_coordinates = [_plain(request) for request in requests]
    if provenance["configuration_identities"] != expected_coordinates:
        raise ValueError("direct capture configuration identities mismatch")
    for item in provenance["source_inputs"]:
        configuration = item.get("configuration")
        if configuration is None:
            continue
        if (
            configuration.get("scenario_id") != scenario_id
            or configuration.get("ingress") != ingress
            or configuration.get("source_declaration") != provenance["source_key"]
            or configuration.get("request_coordinates") != expected_coordinates
        ):
            raise ValueError("source input configuration does not match the transcript")


@dataclass(frozen=True, slots=True)
class Transcript:
    baseline_revision: str
    scenario_id: str
    ingress: str
    runtime: dict[str, Any]
    provenance: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    def to_payload(self) -> dict[str, Any]:
        for record in self.records:
            _validate_record(record)
        declarations = [_typed_declaration(record) for record in self.records]
        input_answers = [record["answer"] for record in self.records]
        declaration_sha256 = _digest(declarations)
        input_sha256 = _digest(input_answers)
        authority = {
            "baseline_revision": self.baseline_revision,
            "ingress": self.ingress,
            "declaration_sha256": declaration_sha256,
            "input_sha256": input_sha256,
        }
        provenance = _bind_provenance(
            self.provenance,
            authority,
            require_source_inputs=any(
                record["request_type"] in _DIRECT_CAPTURE_REQUEST_TYPES for record in self.records
            ),
        )
        if any(record["request_type"] in _DIRECT_CAPTURE_REQUEST_TYPES for record in self.records):
            _validate_direct_provenance(
                provenance,
                tuple(record["request"] for record in self.records),
                self.scenario_id,
                self.ingress,
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "baseline_revision": self.baseline_revision,
            "scenario_id": self.scenario_id,
            "ingress": self.ingress,
            "runtime": dict(self.runtime),
            "provenance": provenance,
            "declaration_sha256": declaration_sha256,
            "input_sha256": input_sha256,
            "records": [dict(record) for record in self.records],
        }

    def answer_for(self, request: object) -> Any:
        declaration = _plain(request)
        matches = [
            record
            for record in self.records
            if record["request_type"] == _request_type(request) and record["request"] == declaration
        ]
        if len(matches) != 1:
            raise ValueError("request is missing or duplicated in transcript")
        return _restore(matches[0]["answer"])

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        expected_requests: tuple[object, ...],
        expected_baseline_revision: str | None = None,
        expected_scenario_id: str | None = None,
        expected_ingress: str | None = None,
    ) -> Transcript:
        if not isinstance(payload, dict) or payload.keys() != _TOP_LEVEL_FIELDS:
            raise ValueError("transcript fields are incomplete or unexpected")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported schema_version")
        for field in ("baseline_revision", "scenario_id", "ingress"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise ValueError(f"{field} must be a nonempty string")
        if (
            expected_baseline_revision is not None
            and payload["baseline_revision"] != expected_baseline_revision
        ):
            raise ValueError("baseline revision mismatch")
        if expected_scenario_id is not None and payload["scenario_id"] != expected_scenario_id:
            raise ValueError("scenario_id mismatch")
        if expected_ingress is not None and payload["ingress"] != expected_ingress:
            raise ValueError("ingress mismatch")
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        expected = [(_request_type(req), _plain(req)) for req in expected_requests]
        if any(kind not in _REQUEST_TYPES for kind, _ in expected):
            raise ValueError("unsupported request_type")
        actual = []
        for record in records:
            _validate_record(record)
            actual.append((record["request_type"], record["request"]))
        if len(actual) != len({_canonical(item) for item in actual}):
            raise ValueError("duplicate transcript request")
        if len(expected) != len({_canonical(item) for item in expected}):
            raise ValueError("duplicate expected request")
        expected_keys = {_canonical(item) for item in expected}
        actual_keys = {_canonical(item) for item in actual}
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        if missing or unexpected:
            raise ValueError(
                f"transcript request roster mismatch: missing={missing}, unexpected={unexpected}"
            )
        # Request membership is order-insensitive; the digest still binds the recorded order.
        declarations = [_typed_declaration(record) for record in records]
        answers = [record["answer"] for record in records]
        if payload["declaration_sha256"] != _digest(declarations):
            raise ValueError("declaration digest mismatch")
        if payload["input_sha256"] != _digest(answers):
            raise ValueError("input digest mismatch")
        runtime = payload["runtime"]
        provenance = payload["provenance"]
        runtime_fields = {"python", "platform", "production_root", "dependency_versions"}
        if (
            not isinstance(runtime, dict)
            or runtime.keys() != runtime_fields
            or not all(
                isinstance(runtime.get(key), str) and runtime[key]
                for key in runtime_fields - {"dependency_versions"}
            )
            or not isinstance(runtime["dependency_versions"], dict)
            or not all(
                isinstance(name, str) and isinstance(version, str) and version
                for name, version in runtime["dependency_versions"].items()
            )
            or not isinstance(provenance, dict)
        ):
            raise ValueError("runtime and provenance fields are invalid")
        expected_provenance = {
            "baseline_revision": payload["baseline_revision"],
            "ingress": payload["ingress"],
            "declaration_sha256": payload["declaration_sha256"],
            "input_sha256": payload["input_sha256"],
        }
        if any(provenance.get(field) != value for field, value in expected_provenance.items()):
            raise ValueError("provenance authority does not match the transcript")
        _bind_provenance(
            provenance,
            expected_provenance,
            require_source_inputs=any(
                record["request_type"] in _DIRECT_CAPTURE_REQUEST_TYPES for record in records
            ),
        )
        if any(record["request_type"] in _DIRECT_CAPTURE_REQUEST_TYPES for record in records):
            _validate_direct_provenance(
                provenance,
                tuple(_plain(request) for request in expected_requests),
                payload["scenario_id"],
                payload["ingress"],
            )
        return cls(
            baseline_revision=payload["baseline_revision"],
            scenario_id=payload["scenario_id"],
            ingress=payload["ingress"],
            runtime=dict(runtime),
            provenance=deepcopy(provenance),
            records=tuple(deepcopy(record) for record in records),
        )


def _warehouse_input_aggregate(relation: Any) -> Any:
    """Build the ordered raw-relation fingerprint without narrowing its arm domain."""
    import ibis

    row_key = ibis.literal("").cast("string")
    for name in relation.columns:
        value = relation[name]
        for text in (
            value.isnull().cast("string"),
            ibis.coalesce(value.cast("string"), ibis.literal("")),
        ):
            text = cast(Any, text)
            row_key = row_key.concat(text.length().cast("string").concat(":").concat(text))
    zero = ibis.literal(0).cast("decimal(38, 9)")
    return relation.aggregate(
        hash_sum=ibis.coalesce(cast(Any, row_key.hash().cast("decimal(38, 9)")).sum(), zero),
        salted_hash_sum=ibis.coalesce(
            cast(Any, row_key.concat("\x1d").hash().cast("decimal(38, 9)")).sum(), zero
        ),
        row_count=relation.count(),
    )


def _warehouse_input_preimage(owner: Any, request: MethodFitRequest) -> dict[str, Any]:
    """Reacquire the backend aggregate within the answer's still-open source scope."""
    import ibis

    from increment._source_identity import source_identity
    from increment.query.cuped import resolved_multicovariate_request

    cell = request.cell
    breakout = None if cell is None else cell.breakout
    resolved = resolved_multicovariate_request(
        owner,
        request.metric,
        request.method,
        control=request.control,
        arm_order=request.arm_order,
        population=request.population,
        breakout_dimension=None if breakout is None else breakout.property,
        breakout_source=None if breakout is None else breakout.source,
        breakout_value=cast(Any, None if cell is None or cell.is_null else cell.value),
        breakout_is_null=False if cell is None else cell.is_null,
    )
    with owner.cuped_relation(request.metric, resolved) as (relation, execute, _scratch):
        schema = [[name, str(relation.schema()[name])] for name in relation.columns]
        row = execute(_warehouse_input_aggregate(relation)).to_pylist()[0]
        backend = ibis.get_backend(relation)
        return {
            "source_identity": _plain(source_identity(owner)),
            "backend": {
                "name": str(backend.name),
                "module": type(backend).__module__,
                "qualname": type(backend).__qualname__,
                "version": str(backend.version),
                "ibis_version": ibis.__version__,
            },
            "preimage": {
                "protocol": "warehouse-method-fit-input-v1",
                "schema": schema,
                "hash_sum": str(row["hash_sum"]),
                "salted_hash_sum": str(row["salted_hash_sum"]),
                "row_count": int(row["row_count"]),
            },
        }


def _acquire_warehouse_input(
    source: Any,
    request: MethodFitRequest,
    answer: MethodFitAnswer,
    receipts: list[dict[str, Any]],
    cache: dict[Any, Any],
) -> None:
    if answer.origin != "warehouse" or answer.evidence_sha256 is None:
        return
    owner = getattr(source, "_source", None) or source
    declaration = _plain(request)
    key = (id(owner), _canonical(declaration))
    cached = cache.get(key)
    if cached is None:
        receipt = {
            "request": declaration,
            **_warehouse_input_preimage(owner, request),
        }
        cache[key] = (owner, receipt)
        receipts.append(receipt)
    else:
        cached_owner, receipt = cached
        if cached_owner is not owner:
            raise ValueError("warehouse input receipt belongs to a different source scope")
    preimage = receipt["preimage"]
    encoded = json.dumps(
        [
            preimage["protocol"],
            preimage["schema"],
            preimage["hash_sum"],
            preimage["salted_hash_sum"],
            preimage["row_count"],
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != answer.evidence_sha256:
        raise ValueError("warehouse input aggregate does not match its acquired answer")


def acquire_transcript(
    provider: Any,
    *,
    scenario_id: str,
    ingress: str,
    requests: tuple[object, ...],
    baseline_revision: str,
    provenance: dict[str, Any],
    warehouse_inputs: tuple[list[dict[str, Any]], dict[Any, Any]] | None = None,
) -> Transcript:
    """Acquire actual typed answers through the source's existing evidence dispatcher."""
    if not requests:
        raise ValueError("an acquisition transcript needs at least one declared request")
    if any(_request_type(request) not in _REQUEST_TYPES for request in requests):
        raise ValueError("unsupported request_type")
    declarations = [(_request_type(req), _plain(req)) for req in requests]
    if len(declarations) != len({_canonical(item) for item in declarations}):
        raise ValueError("duplicate acquisition request")
    records = []
    for request in requests:
        answer = request_evidence(provider, request)
        if warehouse_inputs is not None:
            receipts, cache = warehouse_inputs
            pairs = (
                ((request, answer),)
                if isinstance(request, MethodFitRequest)
                else answer.method_fit_answers
                if isinstance(answer, MetricEvidence)
                else ()
            )
            for fit_request, fit_answer in pairs:
                _acquire_warehouse_input(
                    provider.source,
                    fit_request,
                    cast(MethodFitAnswer, fit_answer),
                    receipts,
                    cache,
                )
        answer_type, answer_payload = _encode_answer(answer, _request_type(request))
        if answer_type != _REQUEST_ANSWERS[_request_type(request)]:
            raise TypeError(f"Unsupported answer for {_request_type(request)}")
        records.append(
            {
                "request_type": _request_type(request),
                "request": _plain(request),
                "answer_type": answer_type,
                "answer": answer_payload,
            }
        )
    input_sha256 = _digest([record["answer"] for record in records])
    provenance = {
        **provenance,
        "baseline_revision": baseline_revision,
        "declaration_sha256": _digest([[kind, declaration] for kind, declaration in declarations]),
        "input_sha256": input_sha256,
        "ingress": ingress,
    }
    version_names = ("increment", "pydantic", "numpy", "scipy", "pyarrow", "narwhals")
    versions = {}
    for distribution in version_names:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return Transcript(
        baseline_revision=baseline_revision,
        scenario_id=scenario_id,
        ingress=ingress,
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "production_root": str(Path(increment.__file__).resolve().parent),
            "dependency_versions": versions,
        },
        provenance=_bind_provenance(provenance, {}),
        records=tuple(records),
    )
