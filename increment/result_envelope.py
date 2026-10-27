"""Portable result cells retaining the complete inference row and readout scope."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, Literal, cast

import pyarrow as pa
from pydantic import BaseModel

from increment._interpretation import EffectEstimate, project_effect
from increment.errors import CodedError
from increment.estimation._fcr_reinterval import classify, reinterval_selected
from increment.estimation.readout_types import FamilyScope, ReadoutMetadata
from increment.estimation.results import LiftEstimate
from increment.kernel.contracts import CellIdentity

_SCHEMA_VERSION = 1
_METADATA_KEY = b"increment.readout_metadata"
_PROJECTION_FIELDS = (
    "null_lift",
    "null_abs",
    "role",
    "multiplicity_status",
    "discovery",
    "family_id",
    "family_axes",
    "family_q",
    "family_threshold",
    "family_procedure",
    "family_guarantee",
    "family_nominal_alpha",
    "family_size",
)


def _safe_projection_fields(row: LiftEstimate) -> dict[str, Any]:
    row_fields = row.model_dump(mode="json")
    return {name: row_fields[name] for name in _PROJECTION_FIELDS if name in row_fields}


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _restore_row(value: object) -> LiftEstimate:
    return LiftEstimate.model_validate_json(_json(_plain(value)))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


class Operation(StrEnum):
    RELEVEL = "relevel"
    COMPARE_MARGIN = "compare_margin"
    COMBINE_FAMILY = "combine_family"


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    state: Literal["available", "unavailable", "advisory"]
    code: str | None = None
    reason: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    advisories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "advisories", tuple(sorted(self.advisories)))


@dataclass(frozen=True, slots=True)
class RetainedState:
    construction: str | None
    reason: str
    kind: Literal["lift_sampling", "posterior", "asymptotic_sequential"]
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze(self.fields))


Component = tuple[ComponentStatus, RetainedState | None]


@dataclass(frozen=True, slots=True)
class Provenance:
    run_id: str | None
    execution_identity_hash: str | None
    package_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise CodedError(
                "created_at must be timezone-aware",
                code="results.cell.provenance_timestamp_invalid",
                context={"field": "created_at"},
            )


@dataclass(frozen=True, slots=True)
class ResultCell:
    identity: CellIdentity
    label: str
    family: FamilyScope | None
    source_snapshot_id: str | None
    components: Mapping[Literal["sampling", "posterior", "sequential"], Component]
    provenance: Provenance
    summary: EffectEstimate | None
    projection_fields: Mapping[str, Any] = field(default_factory=dict)
    supports: Mapping[Literal["sampling", "posterior", "sequential"], frozenset[Operation]] = field(
        init=False
    )

    def __post_init__(self) -> None:
        components = MappingProxyType(dict(self.components))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "projection_fields", _freeze(self.projection_fields))
        support: dict[str, frozenset[Operation]] = {}
        for name in ("sampling", "posterior", "sequential"):
            status, state = components[name]
            operations: set[Operation] = set()
            if (
                name == "sampling"
                and status.state in {"available", "advisory"}
                and state is not None
            ):
                row = _restore_row(state.fields)
                if _can_reconstruct_relevel(row) and _retained_relevel_alpha(row) is not None:
                    operations.add(Operation.RELEVEL)
            support[name] = frozenset(operations)
        object.__setattr__(self, "supports", MappingProxyType(support))

    def __reduce__(self):
        return _restore_result_cell, (self.to_payload(),)

    @classmethod
    def from_row(
        cls,
        row: LiftEstimate,
        metadata: ReadoutMetadata | None = None,
        *,
        created_at: datetime | None = None,
    ) -> ResultCell:
        construction, reason = classify(row)
        identity = CellIdentity.from_row(row)
        key = identity.key
        raw_outcomes_present = row.confidence_set is not None
        row_fields = row.model_dump(mode="json")
        sampling_state = (
            None
            if raw_outcomes_present
            else RetainedState(construction, reason, "lift_sampling", row_fields)
        )
        record = None
        if metadata is not None:
            record = next(
                (
                    item
                    for item in metadata.cells
                    if item.source_snapshot_id == row.source_snapshot_id and item.cell == key
                ),
                None,
            )
        failure = None if record is None else record.failure
        sampling = None if record is None else record.sampling
        advisory_codes = () if record is None else tuple(item.code for item in record.advisories)
        sampling_unavailable = (
            raw_outcomes_present
            or (sampling is not None and not sampling.available)
            or row.sampling_available is False
            or (row.sampling_available is None and row.prior_shrunk)
            or (failure is not None)
            or (
                row.failure_code is not None
                and row.lift is None
                and row.binomial_set is None
                and row.confidence_set is None
            )
        )
        if sampling_unavailable:
            sampling_status = ComponentStatus(
                "unavailable",
                code=(
                    failure.code
                    if failure is not None
                    else (
                        sampling.reason_code
                        if sampling is not None
                        else row.sampling_reason_code
                        or row.failure_code
                        or "results.component.unavailable"
                    )
                ),
                reason=reason,
                context=(
                    failure.context
                    if failure is not None
                    else (row.sampling_reason_context or row.failure_context or {"reason": reason})
                )
                or {},
                advisories=advisory_codes,
            )
        else:
            sampling_status = ComponentStatus(
                "advisory" if advisory_codes else "available",
                reason=reason if construction is None else None,
                advisories=advisory_codes,
            )
        posterior_available = (
            False
            if raw_outcomes_present
            else (
                (
                    record.posterior.available
                    if record is not None and record.posterior is not None
                    else None
                )
                if row.posterior_components is None and row.posterior_latent_mean is None
                else True
            )
        )
        posterior_available = bool(posterior_available)
        posterior_reason = (
            record.posterior.reason_code
            if record is not None and record.posterior is not None
            else row.posterior_reason_code
            or row.posterior_prob_favorable_reason_code
            or "posterior state is not retained"
        )
        posterior_reason_context = (
            record.posterior.reason_context or {}
            if record is not None and record.posterior is not None
            else row.posterior_reason_context or {}
        )
        posterior_status = (
            ComponentStatus("available")
            if posterior_available
            else ComponentStatus(
                "unavailable",
                code=posterior_reason or "results.component.unavailable",
                reason=posterior_reason,
                context=posterior_reason_context,
            )
        )
        posterior_state = (
            RetainedState(construction, reason, "posterior", row_fields)
            if posterior_available
            else None
        )
        from increment.estimation.sequential_result import AsymptoticSequentialResult

        sequential_result = row.sequential_result
        if isinstance(sequential_result, AsymptoticSequentialResult):
            sequential_state = RetainedState(
                construction,
                "alpha is registered; operations follow the registered procedure",
                "asymptotic_sequential",
                row_fields,
            )
            sequential_status = ComponentStatus(
                "available",
                reason="alpha is registered; operations follow the registered procedure",
            )
        else:
            sequential_state = None
            sequential_status = ComponentStatus(
                "unavailable",
                code="results.component.unavailable",
                reason="no asymptotic sequential result is retained",
            )
        family = None if metadata is None else metadata.family_of(row.source_snapshot_id, key)
        try:
            package_version = version("increment")
        except PackageNotFoundError:
            package_version = "0+unknown"
        stamp = created_at or datetime.now(UTC)
        summary = (
            _summary(row)
            if sampling_status.state != "unavailable" or sequential_status.state == "available"
            else None
        )
        return cls(
            identity=CellIdentity.from_row(row),
            label=row.method,
            family=family,
            source_snapshot_id=row.source_snapshot_id,
            components={
                "sampling": (sampling_status, sampling_state),
                "posterior": (posterior_status, posterior_state),
                "sequential": (sequential_status, sequential_state),
            },
            provenance=Provenance(None, None, package_version, stamp),
            summary=summary,
            projection_fields=_safe_projection_fields(row),
        )

    def retained_row(self) -> LiftEstimate:
        state = self.components["sampling"][1]
        if state is None:
            raise CodedError(
                "No retained lift row",
                code="results.operation.unsupported",
                context={"reason": self.components["sampling"][0].reason},
            )
        return _restore_row(state.fields)

    def _projection_row(self) -> LiftEstimate:
        state = (
            self.components["sampling"][1]
            or self.components["sequential"][1]
            or self.components["posterior"][1]
        )
        if state is not None:
            return _restore_row(state.fields)
        status = self.components["sampling"][0]
        key = self.identity.key
        reason = status.reason or "sampling evidence is unavailable"
        context = dict(status.context) or {"reason": reason}
        code = status.code or "results.component.unavailable"
        return LiftEstimate(
            metric=key.metric,
            group_id=key.group_id or key.treatment_group or "unknown",
            method=key.method or self.label,
            method_role=key.method_role or "decision",
            inference=key.inference or "fixed",
            alternative=key.alternative or "two-sided",
            estimand=cast(
                Literal[
                    "itt", "compliance", "late", "ate", "plr_slope", "overlap_subpopulation_ate"
                ],
                key.estimand or "itt",
            ),
            analysis_population=key.analysis_population,
            value_scale=key.value_scale or "relative",
            ds=key.ds,
            source_snapshot_id=self.source_snapshot_id,
            sampling_available=False,
            sampling_reason_code=code,
            sampling_reason_context=context,
            failure_code=code,
            failure_context=context,
            **dict(self.projection_fields),
        )

    def to_payload(self) -> dict[str, Any]:
        sampling_state = self.components["sampling"][1]
        if sampling_state is None:
            row_payload = None
        else:
            row_payload = _plain(sampling_state.fields)
        comps = {}
        for name, (status, state) in self.components.items():
            comps[name] = {
                "status": {
                    "state": status.state,
                    "code": status.code,
                    "reason": status.reason,
                    "context": _plain(status.context),
                    "advisories": list(status.advisories),
                },
                "construction": None if state is None else state.construction,
                "reason": None if state is None else state.reason,
                "kind": None if state is None else state.kind,
                "supports": sorted(op.value for op in self.supports[name]),
            }
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity.model_dump(mode="json"),
            "label": self.label,
            "family": None if self.family is None else self.family.model_dump(mode="json"),
            "source_snapshot_id": self.source_snapshot_id,
            "retained_row": row_payload,
            "projection_fields": _plain(self.projection_fields),
            "components": comps,
            "provenance": {
                "run_id": self.provenance.run_id,
                "execution_identity_hash": self.provenance.execution_identity_hash,
                "package_version": self.provenance.package_version,
                "created_at": self.provenance.created_at.isoformat(),
            },
            "summary": None
            if self.summary is None
            else {
                "point": self.summary.point,
                "lower": self.summary.lower,
                "upper": self.summary.upper,
                "alpha": self.summary.alpha,
                "value_scale": self.summary.value_scale,
                "reason": self.summary.reason,
                "geometry": self.summary.geometry,
                "intervals": self.summary.intervals,
                "qualification": self.summary.qualification,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ResultCell:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise CodedError(
                "Unsupported result-cell schema",
                code="results.cell.schema_unsupported",
                context={"schema_version": payload.get("schema_version")},
            )
        row_dump = payload.get("retained_row")
        row = None if row_dump is None else _restore_row(row_dump)
        comp_data = payload["components"]
        components: dict[Literal["sampling", "posterior", "sequential"], Component] = {}
        for name in ("sampling", "posterior", "sequential"):
            component = comp_data[name]
            raw_status = component["status"]
            if raw_status.get("state") not in {"available", "unavailable", "advisory"}:
                raise CodedError(
                    "Malformed result-cell component status",
                    code="results.cell.schema_unsupported",
                    context={"component": name},
                )
            construction = component.get("construction")
            if construction not in {None, "wald", "conversion", "binomial", "joint", "unavailable"}:
                raise CodedError(
                    "Unsupported result-cell construction",
                    code="results.cell.schema_unsupported",
                    context={"construction": construction},
                )
            kind = component.get("kind")
            if kind not in {None, "lift_sampling", "posterior", "asymptotic_sequential"}:
                raise CodedError(
                    "Unsupported result-cell retained state",
                    code="results.cell.schema_unsupported",
                    context={"kind": kind},
                )
            status = ComponentStatus(
                raw_status["state"],
                raw_status.get("code"),
                raw_status.get("reason"),
                raw_status.get("context") or {},
                tuple(raw_status.get("advisories") or ()),
            )
            state = (
                None
                if row is None or kind is None
                else RetainedState(
                    construction, component.get("reason") or "", kind, row.model_dump(mode="json")
                )
            )
            components[name] = (status, state)
        prov = payload["provenance"]
        summary_data = payload.get("summary")
        if summary_data is not None:
            summary_data = dict(summary_data)
            summary_data["intervals"] = tuple(
                tuple(interval) for interval in summary_data.get("intervals", ())
            )
        return cls(
            identity=CellIdentity.model_validate(payload["identity"]),
            label=payload["label"],
            family=None
            if payload.get("family") is None
            else FamilyScope.model_validate(payload["family"]),
            source_snapshot_id=payload.get("source_snapshot_id"),
            components=components,
            provenance=Provenance(
                prov.get("run_id"),
                prov.get("execution_identity_hash"),
                prov["package_version"],
                datetime.fromisoformat(prov["created_at"]),
            ),
            summary=None if summary_data is None else EffectEstimate(**summary_data),
            projection_fields=payload.get("projection_fields") or {},
        )

    @staticmethod
    def to_arrow(cells: Sequence[ResultCell]) -> pa.Table:
        rows = []
        for cell in cells:
            payload = cell.to_payload()
            entry: dict[str, Any] = {
                "schema_version": _SCHEMA_VERSION,
                "identity": _json(payload["identity"]),
                "label": cell.label,
                "family": None if cell.family is None else _json(cell.family),
                "source_snapshot_id": cell.source_snapshot_id,
                "provenance": _json(payload["provenance"]),
                "summary": None if cell.summary is None else _json(payload["summary"]),
                "retained_row": None
                if payload["retained_row"] is None
                else _json(payload["retained_row"]),
                "projection_fields": _json(payload["projection_fields"]),
            }
            for name in ("sampling", "posterior", "sequential"):
                component = payload["components"][name]
                status = component["status"]
                entry[name] = {
                    "status": status["state"],
                    "code": status["code"],
                    "reason": status["reason"],
                    "context": _json(status["context"]),
                    "advisories": status["advisories"],
                    "construction": component["construction"],
                    "supports": component["supports"],
                }
            rows.append(entry)
        schema = _arrow_schema()
        table = pa.Table.from_pylist(rows, schema=schema)
        return table

    @staticmethod
    def from_arrow(table: pa.Table) -> list[ResultCell]:
        cells = []
        for entry in table.to_pylist():
            components = {}
            row_dump = None if entry["retained_row"] is None else json.loads(entry["retained_row"])
            row = None if row_dump is None else _restore_row(row_dump)
            construction, reason = (
                (None, "no retained sampling row") if row is None else classify(row)
            )
            for name in ("sampling", "posterior", "sequential"):
                comp = entry[name]
                stored_construction = comp["construction"]
                if stored_construction not in {
                    None,
                    "wald",
                    "conversion",
                    "binomial",
                    "joint",
                    "unavailable",
                }:
                    raise CodedError(
                        "Unsupported result-cell construction",
                        code="results.cell.schema_unsupported",
                        context={"component": name, "construction": stored_construction},
                    )
                from increment.estimation.sequential_result import AsymptoticSequentialResult

                has_state = row is not None and (
                    name == "sampling"
                    or (name == "posterior" and comp["status"] in {"available", "advisory"})
                    or (
                        name == "sequential"
                        and isinstance(row.sequential_result, AsymptoticSequentialResult)
                    )
                )
                expected_construction = construction if has_state else None
                if stored_construction != expected_construction:
                    raise CodedError(
                        "Mismatched result-cell construction",
                        code="results.cell.schema_unsupported",
                        context={
                            "component": name,
                            "construction": stored_construction,
                            "expected": expected_construction,
                        },
                    )
                status = {
                    "state": comp["status"],
                    "code": comp["code"],
                    "reason": comp["reason"],
                    "context": json.loads(comp["context"] or "{}"),
                    "advisories": comp["advisories"] or [],
                }
                available_state = (
                    name == "sampling"
                    or (name == "posterior" and comp["status"] in {"available", "advisory"})
                    or (
                        name == "sequential"
                        and row is not None
                        and isinstance(row.sequential_result, AsymptoticSequentialResult)
                    )
                )
                kind = {
                    "sampling": "lift_sampling",
                    "posterior": "posterior",
                    "sequential": "asymptotic_sequential",
                }[name]
                state_reason = (
                    "alpha is registered; operations follow the registered procedure"
                    if name == "sequential"
                    else reason
                )
                components[name] = {
                    "status": status,
                    "construction": stored_construction,
                    "reason": state_reason,
                    "kind": kind if available_state and row_dump is not None else None,
                    "supports": comp["supports"] or [],
                }
            payload = {
                "schema_version": entry["schema_version"],
                "identity": json.loads(entry["identity"]),
                "label": entry["label"],
                "family": None if entry["family"] is None else json.loads(entry["family"]),
                "source_snapshot_id": entry["source_snapshot_id"],
                "retained_row": row_dump,
                "projection_fields": json.loads(entry["projection_fields"])
                if entry.get("projection_fields")
                else {},
                "components": components,
                "provenance": json.loads(entry["provenance"]),
                "summary": None if entry["summary"] is None else json.loads(entry["summary"]),
            }
            cells.append(ResultCell.from_payload(payload))
        return cells


def _restore_result_cell(payload: Mapping[str, Any]) -> ResultCell:
    return ResultCell.from_payload(payload)


def _arrow_schema() -> pa.Schema:
    comp = pa.struct(
        [
            ("status", pa.string()),
            ("code", pa.string()),
            ("reason", pa.string()),
            ("context", pa.string()),
            ("advisories", pa.list_(pa.string())),
            ("construction", pa.string()),
            ("supports", pa.list_(pa.string())),
        ]
    )
    return pa.schema(
        [
            ("schema_version", pa.int32()),
            ("identity", pa.string()),
            ("label", pa.string()),
            ("family", pa.string()),
            ("source_snapshot_id", pa.string()),
            ("provenance", pa.string()),
            ("summary", pa.string()),
            ("retained_row", pa.string()),
            ("projection_fields", pa.string()),
            ("sampling", comp),
            ("posterior", comp),
            ("sequential", comp),
        ]
    )


def _can_reconstruct_relevel(row: LiftEstimate) -> bool:
    relative = (
        row.estimand == "itt"
        and row.value_scale == "relative"
        and row.scale == "log"
        and row.quantile_p_value is None
        and row.independent_mean_reference is None
        and row.relative_unavailable_reason is None
        and row.lift is not None
        and row.lift.log_mean is not None
        and row.lift.log_se is not None
        and row.lift.open_side is None
    )
    additive = (
        row.estimand == "itt"
        and row.abs_diff is not None
        and row.abs_se is not None
        and row.abs_reference_df is not None
        and row.lift is None
        and row.relative_unavailable_reason is not None
    )
    return row.reference_kind == "t" and row.reference_df is not None and (relative or additive)


def _retained_relevel_alpha(row: LiftEstimate) -> float | None:
    alphas = [
        alpha
        for alpha in (
            None if row.lift is None else row.lift.alpha,
            row.abs_alpha,
        )
        if alpha is not None
    ]
    if not alphas and row.confidence_set is not None:
        alphas.append(row.confidence_set.alpha)
    if not alphas and row.relative_confidence_set is not None:
        alphas.append(row.relative_confidence_set.alpha)
    if not alphas:
        return None
    retained = min(alphas)
    return retained if row.alternative == "two-sided" else retained / 2.0


def _conservative_relevel(row: LiftEstimate, alpha: float) -> LiftEstimate:
    retained_alpha = _retained_relevel_alpha(row)
    if retained_alpha is None or alpha > retained_alpha:
        raise CodedError(
            "Portable releveling cannot narrow beyond the retained interval level.",
            code="results.operation.alpha_above_allocation",
            context={"requested_alpha": alpha, "retained_alpha": retained_alpha},
        )
    return row.reintervalize(alpha)


def _summary(row: LiftEstimate) -> EffectEstimate:
    return project_effect(row)


def _cell_from_arrow_payload(payload: Mapping[str, object]) -> ResultCell:
    return ResultCell.from_payload(payload)


def relevel(cell: ResultCell, *, alpha: float) -> ResultCell:
    _require(cell, Operation.RELEVEL)
    updated = _conservative_relevel(cell.retained_row(), alpha)
    return _replace_row(cell, updated)


def _reinterval(cell: ResultCell, *, threshold: float) -> ResultCell:
    row = cell.retained_row()
    construction, reason = classify(row)
    if construction is None:
        raise CodedError(
            "Unsupported result operation",
            code="results.operation.unsupported",
            context={"operation": "reinterval", "reason": reason},
        )
    updated = reinterval_selected(row, construction, threshold)
    return _replace_row(cell, updated)


def _require(cell: ResultCell, operation: Operation) -> None:
    if operation not in cell.supports["sampling"]:
        status, state = cell.components["sampling"]
        reason = (
            state.reason if state is not None else status.reason or "no retained sampling state"
        )
        raise CodedError(
            "Unsupported result operation",
            code="results.operation.unsupported",
            context={"operation": operation.value, "reason": reason},
        )


def _replace_row(cell: ResultCell, row: LiftEstimate) -> ResultCell:
    metadata_components = dict(cell.components)
    construction, reason = classify(row)
    row_fields = row.model_dump(mode="json")
    old_status, old_state = metadata_components["sampling"]
    if old_state is not None:
        sampling_status = ComponentStatus(
            old_status.state,
            old_status.code,
            reason if construction is None else old_status.reason,
            old_status.context,
            old_status.advisories,
        )
        metadata_components["sampling"] = (
            sampling_status,
            RetainedState(construction, reason, "lift_sampling", row_fields),
        )
    for name in ("posterior", "sequential"):
        status, state = metadata_components[name]
        if state is not None:
            metadata_components[name] = (
                status,
                RetainedState(construction, reason, state.kind, row_fields),
            )
    summary = (
        _summary(row)
        if old_state is not None or metadata_components["sequential"][0].state == "available"
        else None
    )
    return ResultCell(
        cell.identity,
        cell.label,
        cell.family,
        row.source_snapshot_id,
        metadata_components,
        cell.provenance,
        summary,
        _safe_projection_fields(row),
    )


@dataclass(frozen=True, slots=True)
class AnalysisEnvelope:
    cells: tuple[ResultCell, ...]
    metadata: ReadoutMetadata | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "cells": [cell.to_payload() for cell in self.cells],
            "metadata": None if self.metadata is None else self.metadata.model_dump(mode="json"),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnalysisEnvelope:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise CodedError(
                "Unsupported result-cell schema",
                code="results.cell.schema_unsupported",
                context={"schema_version": payload.get("schema_version")},
            )
        metadata = payload.get("metadata")
        return cls(
            tuple(ResultCell.from_payload(cell) for cell in payload["cells"]),
            None if metadata is None else ReadoutMetadata.model_validate(metadata),
        )

    def to_arrow(self) -> pa.Table:
        table = ResultCell.to_arrow(self.cells)
        metadata = dict(table.schema.metadata or {})
        if self.metadata is not None:
            metadata[_METADATA_KEY] = _json(self.metadata).encode("utf-8")
        return table.replace_schema_metadata(metadata)

    @classmethod
    def from_arrow(cls, table: pa.Table) -> AnalysisEnvelope:
        encoded = (table.schema.metadata or {}).get(_METADATA_KEY)
        metadata = None if encoded is None else ReadoutMetadata.model_validate(json.loads(encoded))
        return cls(tuple(ResultCell.from_arrow(table)), metadata)


def render(envelope: AnalysisEnvelope, audience: Literal["brief", "full"] = "brief") -> str:
    if audience not in {"brief", "full"}:
        raise CodedError(
            "Unsupported result rendering audience",
            code="results.render.audience_invalid",
            context={"audience": audience},
        )
    from increment._interpretation import (
        interpret_readout,
        render_readout_brief,
        render_readout_text,
    )

    rows = [cell._projection_row() for cell in envelope.cells]
    interpretation = interpret_readout(rows, envelope.metadata)
    brief = render_readout_brief(envelope.cells, interpretation)
    if audience == "brief":
        return "\n".join(brief)
    return "\n".join((render_readout_text(interpretation), *brief))
