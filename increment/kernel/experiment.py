from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel

from increment._canonical import canonical_json_bytes
from increment._evidence_types import EvidenceScope
from increment._source_types import EvidenceCapabilities, ScalarEvidenceDeclaration
from increment.errors import CodedError
from increment.kernel.contracts import (
    AnalysisRequest,
    ImplementationRef,
    Registration,
    Scope,
    _refuse,
    contract_hash,
    declared_breakouts,
)


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A named, portable reference to evidence and its recorded declarations."""

    name: str
    uri: str
    expected_digest: str | None = None
    schema: Mapping[str, str] | None = None
    capabilities: EvidenceCapabilities | None = None
    declaration_version: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.uri or self.declaration_version < 1:
            _refuse("kernel.request.invalid", field="source name, URI, or declaration version")
        if self.schema is not None:
            object.__setattr__(self, "schema", MappingProxyType(dict(sorted(self.schema.items()))))
        if self.capabilities is not None and self.capabilities.scalar is not None:
            scalar = self.capabilities.scalar
            frozen = dataclasses.replace(
                scalar,
                bindings=(
                    None
                    if scalar.bindings is None
                    else MappingProxyType(
                        {name: tuple(bindings) for name, bindings in scalar.bindings.items()}
                    )
                ),
            )
            object.__setattr__(
                self, "capabilities", dataclasses.replace(self.capabilities, scalar=frozen)
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "uri": self.uri,
            "expected_digest": self.expected_digest,
            "schema": None if self.schema is None else dict(self.schema),
            "capabilities": _capability_payload(self.capabilities),
            "declaration_version": self.declaration_version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SourceReference:
        raw_capabilities = payload.get("capabilities")
        expected_digest = payload.get("expected_digest")
        schema = payload.get("schema")
        return cls(
            name=str(payload["name"]),
            uri=str(payload["uri"]),
            expected_digest=expected_digest if isinstance(expected_digest, str) else None,
            schema=cast(Mapping[str, str], schema) if isinstance(schema, Mapping) else None,
            capabilities=(
                _capabilities_from_payload(cast(Mapping[str, object], raw_capabilities))
                if isinstance(raw_capabilities, Mapping)
                else None
            ),
            declaration_version=int(cast(int, payload.get("declaration_version", 1))),
        )


def _json_value(value: Any) -> Any:
    if callable(value):
        _refuse("kernel.method.implementation_ref_missing", field="callable")
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=canonical_json_bytes)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _serialization_model(value: Any) -> Any:
    """Normalize immutable containers without changing declared model fields."""
    if callable(value):
        return value
    if isinstance(value, BaseModel):
        return type(value).model_construct(
            _fields_set=value.model_fields_set,
            **{
                name: _serialization_model(getattr(value, name))
                for name in type(value).model_fields
            },
        )
    if isinstance(value, Mapping):
        return {key: _serialization_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_serialization_model(item) for item in value)
    return value


def _freeze(value: Any) -> Any:
    if callable(value):
        return value
    if isinstance(value, BaseModel):
        fields = _serialization_model(value).model_dump(mode="python")
        return _freeze({name: getattr(value, name) for name in fields})
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _freeze(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        )
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _capability_payload(value: EvidenceCapabilities | None) -> dict[str, object] | None:
    return None if value is None else _json_value(value)


def _capabilities_from_payload(payload: Mapping[str, object]) -> EvidenceCapabilities:
    values: dict[str, Any] = dict(payload)
    for name in ("grains", "operations", "count_grains", "populations", "sql_grains"):
        values[name] = frozenset(values.get(name, ()))
    if values.get("available_covariates") is not None:
        values["available_covariates"] = frozenset(values["available_covariates"])
    values["breakouts"] = tuple(values.get("breakouts", ()))
    scalar = values.get("scalar")
    if isinstance(scalar, Mapping):
        scalar_values = dict(scalar)
        for name in ("anonymous_metrics", "covariate_grains"):
            if scalar_values.get(name) is not None:
                scalar_values[name] = frozenset(scalar_values[name])
        for name in ("recipe_configs",):
            scalar_values[name] = tuple(scalar_values.get(name, ()))
        scalar_values["bindings"] = scalar_values.get("bindings")
        from pydantic import TypeAdapter

        from increment._analysis_config import ResolvedMetricConfig
        from increment.semantics.models import CupedCovariate, CupedCovariateBinding

        scalar_values["recipe_configs"] = TypeAdapter(
            tuple[ResolvedMetricConfig, ...]
        ).validate_python(scalar_values["recipe_configs"])
        if scalar_values["bindings"] is not None:
            scalar_values["bindings"] = TypeAdapter(
                dict[str, tuple[CupedCovariateBinding | CupedCovariate, ...]]
            ).validate_python(scalar_values["bindings"])
        values["scalar"] = ScalarEvidenceDeclaration(**scalar_values)
    return EvidenceCapabilities(**values)


@dataclass(frozen=True, slots=True)
class Experiment:
    """Immutable, serializable description of an analysis and its data sources."""

    experiment_id: str
    design: Mapping[str, object]
    metrics: Mapping[str, object]
    registration: Registration
    source_refs: tuple[SourceReference, ...]
    scopes: tuple[Scope, ...]
    implementation_refs: Mapping[tuple[str, str], ImplementationRef] = dataclasses.field(
        default_factory=dict
    )
    package_version: str = "0"
    analysis_context: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    operations: Mapping[str, str] = dataclasses.field(default_factory=dict)
    request_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if "capture" in self.request_options:
            _refuse("kernel.request.capture_not_supported")
        if "population" in self.request_options:
            _refuse("kernel.request.population_scope_only")
        if not self.experiment_id or not self.source_refs or not self.scopes:
            _refuse("kernel.request.invalid", field="experiment id, source references, or scopes")
        names = tuple(source.name for source in self.source_refs)
        if len(set(names)) != len(names):
            _refuse("kernel.request.invalid", field="duplicate source reference names")
        scope_hashes = tuple(scope.hash() for scope in self.scopes)
        if len(set(scope_hashes)) != len(scope_hashes):
            _refuse("kernel.request.invalid", field="duplicate experiment scopes")
        object.__setattr__(self, "design", _freeze(self.design))
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "scopes", tuple(self.scopes))
        object.__setattr__(
            self, "implementation_refs", MappingProxyType(dict(self.implementation_refs))
        )
        object.__setattr__(self, "analysis_context", _freeze(self.analysis_context))
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))
        object.__setattr__(self, "request_options", _freeze(self.request_options))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "design": _json_value(self.design),
            "metrics": _json_value(self.metrics),
            "registration": self.registration.canonical_payload(),
            "source_refs": [item.canonical_payload() for item in self.source_refs],
            "scopes": [item.canonical_payload() for item in self.scopes],
            "implementation_refs": [
                {
                    "method_instance_id": key[0],
                    "learner_role": key[1],
                    "implementation": value.canonical_payload(),
                }
                for key, value in sorted(self.implementation_refs.items())
            ],
            "package_version": self.package_version,
            "analysis_context": _json_value(self.analysis_context),
            "operations": dict(self.operations),
            "request_options": _json_value(self.request_options),
        }

    def to_json(self, *, portable: bool = True) -> str:
        try:
            return canonical_json_bytes(self.canonical_payload()).decode("utf-8")
        except (TypeError, ValueError) as exc:
            if portable:
                raise CodedError(
                    "Experiment contains values that cannot be represented portably",
                    code="kernel.method.implementation_ref_missing",
                    context={"field": "callable"},
                ) from exc
            raise

    @classmethod
    def from_json(cls, value: str | bytes) -> Experiment:
        payload = json.loads(value)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            _refuse("kernel.request.invalid", field="unsupported experiment format")
        implementations = {
            (
                str(item["method_instance_id"]),
                str(item["learner_role"]),
            ): ImplementationRef.from_payload(item["implementation"])
            for item in payload.get("implementation_refs", ())
        }
        scopes = tuple(Scope.from_payload(item) for item in payload["scopes"])
        operations = dict(payload.get("operations", {}))
        migrated_scopes: list[Scope] = []
        for scope in scopes:
            legacy_hash = scope.hash()
            operation = operations.get(legacy_hash)
            expected = {"daily": "run_daily", "asof": "run_asof"}.get(scope.view)
            if expected is not None and operation == expected:
                scope = scope.model_copy(update={"values": True})
                operations.pop(legacy_hash, None)
            migrated_scopes.append(scope)
        return cls(
            experiment_id=str(payload["experiment_id"]),
            design=payload["design"],
            metrics=payload["metrics"],
            registration=Registration.from_payload(payload["registration"]),
            source_refs=tuple(
                SourceReference.from_payload(item) for item in payload["source_refs"]
            ),
            scopes=tuple(migrated_scopes),
            implementation_refs=implementations,
            package_version=str(payload["package_version"]),
            analysis_context=payload.get("analysis_context", {}),
            operations=operations,
            request_options=payload.get("request_options", {}),
        )

    def with_views(self, *scopes: Scope) -> Experiment:
        """Return this declaration with the selected scopes in declared order."""
        if not scopes or len({scope.hash() for scope in scopes}) != len(scopes):
            _refuse("kernel.request.invalid", field="empty or duplicate experiment scopes")
        return dataclasses.replace(self, scopes=tuple(scopes), operations={})

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def provides(self) -> frozenset[str]:
        return frozenset(
            f"{source.name}.{name}" for source in self.source_refs for name in (source.schema or {})
        )

    def compile(self) -> tuple[EvidenceScope, tuple[AnalysisRequest, ...]]:
        from pydantic import TypeAdapter

        from increment.declarations import thaw
        from increment.semantics.models import Metric

        metrics = TypeAdapter(tuple[Metric, ...]).validate_python(
            thaw(tuple(self.metrics.values()))
        )
        selections = tuple(tuple(scope.metrics) or metrics for scope in self.scopes)
        breakouts = tuple(declared_breakouts(self.source_refs, scope) for scope in self.scopes)
        source = self.source_refs[0].name
        populations = tuple(dict.fromkeys(scope.population for scope in self.scopes))
        inventory = EvidenceScope(
            metrics=tuple(
                {metric.name: metric for selection in selections for metric in selection}.values()
            ),
            population=populations[0],
            populations=populations,
            breakouts=tuple(
                dict.fromkeys(identity for selection in breakouts for identity in selection)
            ),
        )
        requests = tuple(
            AnalysisRequest.from_requests(
                (
                    EvidenceScope(
                        metrics=selection,
                        population=scope.population,
                        grain=scope.grain,
                        breakouts=resolved_breakouts,
                    ),
                ),
                scope=scope,
                registration_hash=self.registration.registration_hash,
                source_ref=source,
            )
            for scope, selection, resolved_breakouts in zip(
                self.scopes, selections, breakouts, strict=True
            )
        )
        return inventory, requests

    @property
    def analysis_spec_hash(self) -> str:
        from increment._runnability import scope_operation

        return contract_hash(
            "analysis_spec",
            {
                "experiment_id": self.experiment_id,
                "design": _json_value(self.design),
                "metrics": _json_value(self.metrics),
                "registration_hash": self.registration.registration_hash,
                "source_refs": [item.canonical_payload() for item in self.source_refs],
                "scopes": [item.canonical_payload() for item in self.scopes],
                "operations": {scope.hash(): scope_operation(self, scope) for scope in self.scopes},
                "request_options": _json_value(self.request_options),
            },
        )

    @property
    def execution_identity_hash(self) -> str:
        return contract_hash(
            "execution_identity",
            {
                "analysis_spec_hash": self.analysis_spec_hash,
                "implementations": [
                    {
                        "method_instance_id": key[0],
                        "learner_role": key[1],
                        "implementation": value.canonical_payload(),
                    }
                    for key, value in sorted(self.implementation_refs.items())
                ],
                "package_version": self.package_version,
            },
        )
