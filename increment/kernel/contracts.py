from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator, model_validator
from pydantic_core import PydanticSerializationError

from increment import _evidence_types
from increment._canonical import CanonicalJSONError, canonical_json_bytes
from increment._evidence_types import BreakoutIdentity
from increment.declarations import (
    VARIANCE_REDUCTION,
    CellKey,
    Method,
    estimator_id_for_selector,
)
from increment.errors import CodedModel, InvalidRequestError, raiser, refusals
from increment.semantics.models import Metric

CONTRACTS_SCHEMA_VERSION: int = 1

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "kernel.scope.invalid": "Invalid scope field {field!r}",
        "kernel.method.implementation_ref_missing": "Method callable field {field!r} has no implementation reference",
        "kernel.request.invalid": "Invalid analysis request: {field}",
        "kernel.request.unportable": "Request kind {kind!r} field {field!r} is not portable",
        "kernel.request.capture_not_supported": (
            "Saved capture is an execution step; call analysis.capture_sequential(...) then run",
            (),
        ),
        "kernel.request.population_scope_only": "Set population on Scope(population=...)",
        "kernel.run.provider_execution_unavailable": (
            "BoundResolver verifies providers but cannot execute them; use a bound Analysis resolver",
            (),
        ),
        "kernel.request.scope_mismatch": "Request kind {kind!r} field {field!r} disagrees with scope",
        "kernel.definitions.catalog_missing": "Offline definitions catalog is missing for {path!r}; bind the definitions with a connection first",
        "kernel.definitions.catalog_stale": "Offline definitions catalog is stale for {path!r}; rebind the definitions with a connection",
        "kernel.request.datetime_timezone": "Request datetime field {field!r} must be timezone-aware",
        "kernel.method.duplicate_instance": "Method instance id {method_instance_id!r} is duplicated",
        "kernel.context.closed": "Execution context is closed",
        "kernel.context.undeclared_stream": "Execution context does not declare stream {stream!r}",
        "kernel.snapshot.integrity": (
            "Captured stream {stream!r} can no longer be replayed",
            ("snapshot_id",),
        ),
    },
)
_refuse = raiser(_REFUSALS)


def contract_hash(kind: str, payload: object) -> str:
    encoded = canonical_json_bytes(
        {"schema": CONTRACTS_SCHEMA_VERSION, "kind": kind, "payload": payload}
    )
    return hashlib.sha256(encoded).hexdigest()


class _ContractModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Scope(_ContractModel):
    view: Literal["run", "daily", "asof", "breakout", "switchback"]
    grain: Literal["total", "daily", "asof"]
    through_day: date | None = None
    day_domain: tuple[date, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"
    time_basis: Literal["calendar", "cohort"] = "calendar"
    completion_policy: bool = False
    segment: tuple[BreakoutIdentity, ...] = ()
    metrics: tuple[Metric, ...] = Field(default=(), exclude_if=lambda value: not value)

    @field_validator("segment", mode="before")
    @classmethod
    def _normalize_segment(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, BreakoutIdentity) or isinstance(value, Mapping):
            return (value,)
        return value

    @model_validator(mode="after")
    def _validate_scope(self) -> Scope:
        invalid = (
            (
                self.view in {"run", "breakout", "switchback"}
                and (self.grain != "total" or self.through_day is not None or self.day_domain)
            )
            or (self.view == "run" and bool(self.segment))
            or (self.view == "breakout" and not self.segment)
            or (self.view == "daily" and self.grain != "daily")
            or (self.view == "asof" and self.grain != "asof")
            or (self.view == "breakout" and self.segment is None)
            or (bool(self.day_domain) and tuple(sorted(set(self.day_domain))) != self.day_domain)
            or (self.view != "switchback" and bool(self.metrics))
        )
        if invalid:
            _refuse("kernel.scope.invalid", field="view/grain/through_day/segment/metrics")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Scope:
        return cls.model_validate(dict(payload))

    def hash(self) -> str:
        return contract_hash("scope", self.canonical_payload())


class ImplementationRef(_ContractModel):
    import_path: str
    version_pin: str
    config_digest: str

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ImplementationRef:
        return cls.model_validate(dict(payload))

    def hash(self) -> str:
        return contract_hash("implementation_ref", self.canonical_payload())


class CellIdentity(_ContractModel):
    key: SkipValidation[CellKey]
    source_snapshot_id: str | None
    source_content_digest: str | None = Field(default=None, exclude_if=lambda value: value is None)
    experiment_id: str | None = None
    analysis_spec_hash: str | None = None
    metric_hash: str | None = None
    estimator_id: str | None = None
    method_config_hash: str | None = None
    method_instance_id: str | None = None
    trigger_name: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_identity_hash: str | None = Field(default=None, exclude_if=lambda value: value is None)
    scope: Scope | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_key(cls, value: object) -> object:
        if isinstance(value, Mapping):
            fields = dict(value)
            key = fields.get("key")
            if key is not None and not isinstance(key, CellKey):
                fields["key"] = CellKey.model_validate(key)
            return fields
        return value

    @classmethod
    def from_row(cls, row: object) -> CellIdentity:
        return cls(
            key=CellKey.from_row(row),
            source_snapshot_id=getattr(row, "source_snapshot_id", None),
            source_content_digest=getattr(row, "source_content_digest", None),
        )

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CellIdentity:
        return cls.model_validate(dict(payload))

    def hash(self) -> str:
        payload = self.canonical_payload()
        payload.pop("source_content_digest", None)
        return contract_hash("cell_identity", payload)


def method_estimator_id(method: Method) -> str | None:
    if method.is_observational:
        return estimator_id_for_selector(method.name)
    return VARIANCE_REDUCTION.get(method.variance_reduction)


def metric_hash(metric: BaseModel) -> str:
    return contract_hash("metric", metric.model_dump(mode="json"))


def method_config_hash(
    method: Method,
    implementations: Mapping[str, ImplementationRef] | None = None,
    *,
    method_instance_id: str,
) -> str:
    payload: dict[str, object] = {}
    implementations = implementations or {}
    for name in type(method).model_fields:
        if name == "name":
            continue
        value = getattr(method, name)
        if name in {"propensity_learner", "outcome_learner"}:
            if value is not None:
                ref = implementations.get(name)
                if ref is None:
                    _refuse("kernel.method.implementation_ref_missing", field=name)
                payload[name] = {
                    "method_instance_id": method_instance_id,
                    "learner_role": name,
                    "import_path": ref.import_path,
                    "config_digest": ref.config_digest,
                }
            else:
                payload[name] = None
        else:
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="json")
            elif isinstance(value, tuple):
                value = [
                    item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                    for item in value
                ]
            payload[name] = value
    return contract_hash("method_config", payload)


class MethodIdentity(_ContractModel):
    estimator_id: str
    method_config_hash: str
    method_role: Literal["decision", "sensitivity"]
    method_instance_id: str

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MethodIdentity:
        return cls.model_validate(dict(payload))

    def hash(self) -> str:
        return contract_hash("method_identity", self.canonical_payload())


class Registration(_ContractModel):
    registration_hash: str

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Registration:
        return cls.model_validate(dict(payload))

    @classmethod
    def from_plan(cls, plan: Any) -> Registration:
        from importlib import import_module

        encode = import_module("increment.decision_wire").registration_payload
        return cls(registration_hash=contract_hash("registration", encode(plan)))

    def hash(self) -> str:
        return self.registration_hash

    def matches(self, plan: Any) -> bool:
        return self.registration_hash == type(self).from_plan(plan).registration_hash


@dataclass(frozen=True, slots=True)
class CompiledPlanBinding:
    """Versioned portable plan payload bound to a hash-only registration."""

    registration_hash: str
    version: Literal[1]
    _payload: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        if self.version != 1 or not self.registration_hash:
            _refuse("kernel.request.invalid", field="compiled plan binding")
        payload = _freeze_descriptor_value(dict(self._payload))
        if not isinstance(payload, Mapping):
            _refuse("kernel.request.invalid", field="compiled plan payload")
        object.__setattr__(self, "_payload", payload)
        if not Registration(registration_hash=self.registration_hash).matches(self.resolve()):
            _refuse("kernel.request.invalid", field="registration/plan mismatch")

    @classmethod
    def from_plan(cls, registration: Registration, plan: Any) -> CompiledPlanBinding:
        from importlib import import_module

        encode = import_module("increment.decision_wire").compiled_plan_to_dict
        return cls(registration.registration_hash, 1, encode(plan))

    @property
    def plan_payload(self) -> dict[str, object]:
        return cast(dict[str, object], _thaw_descriptor_value(self._payload))

    def resolve(self) -> Any:
        from importlib import import_module

        decode = import_module("increment.decision_wire").compiled_plan_from_dict
        return decode(self.plan_payload)

    def validate(self, registration: Registration) -> Any:
        plan = self.resolve()
        if self.registration_hash != registration.registration_hash or not registration.matches(
            plan
        ):
            _refuse("kernel.request.invalid", field="registration/plan mismatch")
        return plan

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "registration_hash": self.registration_hash,
            "plan_payload": self.plan_payload,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CompiledPlanBinding:
        return cls(
            registration_hash=cast(str, payload["registration_hash"]),
            version=cast(Literal[1], payload["version"]),
            _payload=cast(Mapping[str, object], payload["plan_payload"]),
        )

    def to_payload(self) -> dict[str, object]:
        return self.canonical_payload()


def _thaw_descriptor_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_descriptor_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_descriptor_value(item) for item in value]
    return value


def analysis_spec_hash(
    *,
    design: BaseModel,
    metric_hashes: Mapping[str, str],
    registration_hash: str,
    source_refs: Mapping[str, str],
    methods: Sequence[MethodIdentity],
    scope: Scope,
) -> str:
    method_instance_ids = [method.method_instance_id for method in methods]
    if len(set(method_instance_ids)) != len(method_instance_ids):
        duplicate = next(
            instance_id
            for instance_id in method_instance_ids
            if method_instance_ids.count(instance_id) > 1
        )
        _refuse("kernel.method.duplicate_instance", method_instance_id=duplicate)
    return contract_hash(
        "analysis_spec",
        {
            "design": design.model_dump(mode="json"),
            "metric_hashes": dict(metric_hashes),
            "registration_hash": registration_hash,
            "source_refs": dict(source_refs),
            "methods": [
                method.canonical_payload()
                for method in sorted(methods, key=lambda item: item.method_instance_id)
            ],
            "scope": scope.canonical_payload(),
        },
    )


def execution_identity_hash(
    *,
    analysis_spec_hash: str,
    implementations: Mapping[tuple[str, str], ImplementationRef],
    package_version: str,
    portable: bool,
) -> str:
    bound_implementations = [
        {
            "method_instance_id": method_instance_id,
            "learner_role": learner_role,
            "implementation": ref.canonical_payload(),
        }
        for (method_instance_id, learner_role), ref in sorted(implementations.items())
    ]
    return contract_hash(
        "execution_identity",
        {
            "analysis_spec_hash": analysis_spec_hash,
            "implementations": bound_implementations,
            "package_version": package_version,
            "portable": portable,
        },
    )


def _request_value(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            _refuse("kernel.request.datetime_timezone", field="datetime")
        normalized = value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        return {"type": "datetime", "value": normalized}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return {
            field.name: _request_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _request_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_request_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_request_value(item) for item in value), key=canonical_json_bytes)
    return value


def _freeze_descriptor_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_descriptor_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_descriptor_value(item) for item in value)
    return value


def _thaw_descriptor_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_descriptor_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_descriptor_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RequestDescriptor:
    kind: str
    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_descriptor_value(dict(self.fields)))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "fields": _thaw_descriptor_value(self.fields),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RequestDescriptor:
        return cls(
            kind=cast(str, payload["kind"]),
            fields=cast(Mapping[str, object], payload["fields"]),
        )


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    requests: tuple[RequestDescriptor, ...]
    scope: Scope
    registration_hash: str
    source_ref: str

    def __post_init__(self) -> None:
        if not self.requests or any(
            not isinstance(request, RequestDescriptor) for request in self.requests
        ):
            _refuse("kernel.request.invalid", field="requests")
        for request in self.requests:
            fields = request.fields
            for field_name in ("population", "grain", "completed_windows_only"):
                value = fields.get(field_name)
                if field_name == "population" and request.kind == "CountsRequest":
                    if fields.get("kind") == "triggered":
                        value = "triggered"
                expected = (
                    self.scope.population
                    if field_name == "population"
                    else self.scope.grain
                    if field_name == "grain"
                    else self.scope.completion_policy
                )
                if value is not None and value != expected:
                    _refuse("kernel.request.scope_mismatch", kind=request.kind, field=field_name)

            if request.kind == "DayAxisRequest":
                caller = fields.get("caller")
                if isinstance(caller, str):
                    expected_view = (
                        "daily"
                        if caller in {"run_daily", "run_daily_lift"}
                        else "asof"
                        if caller in {"run_asof", "run_asof_lift"}
                        else None
                    )
                    if expected_view is not None and self.scope.view != expected_view:
                        _refuse("kernel.request.scope_mismatch", kind=request.kind, field="caller")
            breakout_only = request.kind in {
                "BreakoutSlicesRequest",
                "BreakoutMethodMomentsRequest",
                "BreakoutUnitRowsRequest",
            }
            if request.kind == "MethodFitRequest":
                cell = fields.get("cell")
                breakout_only = isinstance(cell, Mapping) and cell.get("breakout") is not None
            if breakout_only and self.scope.view != "breakout":
                _refuse("kernel.request.scope_mismatch", kind=request.kind, field="view")

    @classmethod
    def from_requests(
        cls,
        requests: Sequence[object],
        *,
        scope: Scope,
        registration_hash: str,
        source_ref: str,
    ) -> AnalysisRequest:
        descriptors: list[RequestDescriptor] = []
        for request in requests:
            kind = type(request).__name__
            if (
                type(request).__module__ != _evidence_types.__name__
                or not dataclasses.is_dataclass(request)
                or (not kind.endswith("Request") and kind != "EvidenceScope")
            ):
                _refuse("kernel.request.invalid", field="requests")
            fields: dict[str, object] = {}
            for request_field in dataclasses.fields(request):
                try:
                    value = _request_value(getattr(request, request_field.name))
                    canonical_json_bytes(value)
                except (CanonicalJSONError, PydanticSerializationError):
                    _refuse("kernel.request.unportable", kind=kind, field=request_field.name)
                fields[request_field.name] = value
            descriptors.append(RequestDescriptor(kind=kind, fields=fields))
        if not descriptors:
            _refuse("kernel.request.invalid", field="requests")
        return cls(
            requests=tuple(descriptors),
            scope=scope,
            registration_hash=registration_hash,
            source_ref=source_ref,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "requests": [request.canonical_payload() for request in self.requests],
            "scope": self.scope.canonical_payload(),
            "registration_hash": self.registration_hash,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AnalysisRequest:
        return cls(
            requests=tuple(
                RequestDescriptor.from_payload(request)
                for request in cast(Sequence[Mapping[str, object]], payload["requests"])
            ),
            scope=Scope.from_payload(cast(Mapping[str, object], payload["scope"])),
            registration_hash=cast(str, payload["registration_hash"]),
            source_ref=cast(str, payload["source_ref"]),
        )

    def hash(self) -> str:
        return contract_hash("analysis_request", self.canonical_payload())


@runtime_checkable
class ExecutionContext(Protocol):
    run_id: str
    snapshot_manifest: Mapping[str, str]

    def evidence_scope(self) -> AbstractContextManager[object]: ...


@runtime_checkable
class SchemaCatalog(Protocol):
    def columns(self, source_ref: str) -> Mapping[str, str]: ...


@runtime_checkable
class RunMetadata(Protocol):
    @property
    def scope(self) -> object: ...
    @property
    def cells(self) -> object: ...

    @property
    def partial(self) -> bool: ...

    @property
    def partial_reason(self) -> object: ...
