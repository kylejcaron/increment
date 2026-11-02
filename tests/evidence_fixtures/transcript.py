from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from increment._evidence_types import (
    AssignmentEvidence,
    AssignmentRequest,
    CountEvidence,
    CountsRequest,
    MetricEvidence,
    MetricEvidenceRequest,
    MomentsRequest,
)
from increment.evidence import request_evidence

_SCHEMA_VERSION = 1

_REQUEST_TYPES = {
    cls.__name__
    for cls in (MetricEvidenceRequest, MomentsRequest, CountsRequest, AssignmentRequest)
}
_REQUEST_ANSWERS = {
    "MetricEvidenceRequest": "MetricEvidence",
    "MomentsRequest": "MomentRows",
    "CountsRequest": "CountEvidence",
    "AssignmentRequest": "AssignmentEvidence",
}
_ANSWER_FIELDS = {
    "MetricEvidence": frozenset(
        {
            "rows",
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
}
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


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("Non-finite transcript values are unsupported")
        return value
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Transcript mappings require string keys")
        return {key: _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=_canonical)
    if hasattr(value, "to_pylist"):
        return _plain(value.to_pylist())
    raise TypeError(f"No plain transcript encoding for {type(value).__qualname__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _request_type(request: Any) -> str:
    return type(request).__name__


def _encode_answer(answer: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(answer, MetricEvidence):
        if (
            answer.unit_frame is not None
            or answer.method_refusals
            or answer.method_cell_refusals
            or answer.method_fit_answers
        ):
            raise TypeError("MetricEvidence includes a representation outside the scalar codec")
        return "MetricEvidence", {
            "rows": _plain(answer.rows),
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
        }
    if isinstance(answer, CountEvidence):
        return "CountEvidence", _plain(answer)
    if isinstance(answer, AssignmentEvidence):
        return "AssignmentEvidence", _plain(answer)
    if isinstance(answer, tuple) and all(isinstance(row, Mapping) for row in answer):
        return "MomentRows", _plain(answer)
    raise TypeError(f"Unsupported acquired evidence answer: {type(answer).__qualname__}")


@dataclass(frozen=True, slots=True)
class Transcript:
    baseline_revision: str
    scenario_id: str
    ingress: str
    runtime: dict[str, str]
    provenance: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    def to_payload(self) -> dict[str, Any]:
        declarations = [record["request"] for record in self.records]
        input_answers = [record["answer"] for record in self.records]
        return {
            "schema_version": _SCHEMA_VERSION,
            "baseline_revision": self.baseline_revision,
            "scenario_id": self.scenario_id,
            "ingress": self.ingress,
            "runtime": dict(self.runtime),
            "provenance": _plain(self.provenance),
            "declaration_sha256": _digest(declarations),
            "input_sha256": _digest(input_answers),
            "records": [dict(record) for record in self.records],
        }

    def answer_for(self, request: object) -> Any:
        declaration = _plain(request)
        matches = [record for record in self.records if record["request"] == declaration]
        if len(matches) != 1:
            raise ValueError("request is missing or duplicated in transcript")
        return _plain(matches[0]["answer"])

    @classmethod
    def from_payload(cls, payload: Any, *, expected_requests: tuple[object, ...]) -> Transcript:
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
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        expected = [(_request_type(req), _plain(req)) for req in expected_requests]
        if any(kind not in _REQUEST_TYPES for kind, _ in expected):
            raise ValueError("unsupported request_type")
        actual = []
        for record in records:
            if not isinstance(record, dict) or record.keys() != _RECORD_FIELDS:
                raise ValueError("record fields are incomplete or unexpected")
            if record["request_type"] not in _REQUEST_TYPES:
                raise ValueError("unsupported request_type")
            answer_type = record["answer_type"]
            answer = record["answer"]
            if answer_type not in _ANSWER_FIELDS:
                raise ValueError("unsupported answer_type")
            if answer_type != _REQUEST_ANSWERS[record["request_type"]]:
                raise ValueError("request and answer type mismatch")
            fields = _ANSWER_FIELDS[answer_type]
            if fields is None:
                if not isinstance(answer, list):
                    raise ValueError("MomentRows answer must be a list")
            elif not isinstance(answer, dict) or answer.keys() != fields:
                raise ValueError("answer fields are incomplete or unexpected")
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
        declarations = [record["request"] for record in records]
        answers = [record["answer"] for record in records]
        if payload["declaration_sha256"] != _digest(declarations):
            raise ValueError("declaration digest mismatch")
        if payload["input_sha256"] != _digest(answers):
            raise ValueError("input digest mismatch")
        runtime = payload["runtime"]
        provenance = payload["provenance"]
        if (
            not isinstance(runtime, dict)
            or runtime.keys() != {"python", "platform"}
            or not all(isinstance(value, str) and value for value in runtime.values())
            or not isinstance(provenance, dict)
        ):
            raise ValueError("runtime and provenance fields are invalid")
        return cls(
            baseline_revision=payload["baseline_revision"],
            scenario_id=payload["scenario_id"],
            ingress=payload["ingress"],
            runtime=dict(runtime),
            provenance=_plain(provenance),
            records=tuple(_plain(record) for record in records),
        )


def acquire_transcript(
    provider: Any,
    *,
    scenario_id: str,
    ingress: str,
    requests: tuple[object, ...],
    baseline_revision: str,
    provenance: dict[str, Any],
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
        answer_type, answer_payload = _encode_answer(answer)
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
    return Transcript(
        baseline_revision=baseline_revision,
        scenario_id=scenario_id,
        ingress=ingress,
        runtime={"python": sys.version.split()[0], "platform": platform.platform()},
        provenance=_plain(provenance),
        records=tuple(records),
    )
