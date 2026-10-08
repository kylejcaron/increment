"""Immutable source-scoped readout identities and portable result metadata.

This module intentionally has no query-layer dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment._immutable import _FrozenMapping
from increment.decision import MultiplicityFamily
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    UnsupportedRequestError,
    WireFormatError,
    refusals,
    refuse,
)
from increment.estimation._sequential_likelihood import LikelihoodCertificate
from increment.estimation.decision_types import (
    ArmHypothesisKey,
    AsymptoticSequentialEvidence,
    ContrastHypothesisKey,
    EValueEvidence,
    PValueEvidence,
    SegmentHypothesisKey,
)
from increment.estimation.inference import Normal
from increment.estimation.priors import MixturePrior, StudentTPrior
from increment.estimation.sequential_result import AsymptoticSequentialResult
from increment.sequential_state import SequentialCheckpoint

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "readout.cell.key_invalid": "Invalid {kind} readout cell fields: {fields}",
        "readout.collection.mutation_unsupported": "Collection mutation {operation} is unsupported for {model}",
        "readout.collection.concat_type_mismatch": "Cannot concatenate {left} and {right}",
        "readout.collection.concat_cell_collision": "Conflicting cell {cell_key} across snapshots {snapshot_ids}",
        "readout.collection.duplicate_cell_key": "Duplicate cell {cell_key} in source {source_snapshot_id}",
        "readout.failure.hypothesis_key_collision": "Failure hypothesis {hypothesis} maps to multiple cells {cell_keys}",
        "readout.collection.concat_metadata_mismatch": "Metadata differs for snapshots {snapshot_ids}: {fields}",
        "readout.legacy.sampling_unreconstructible": "Legacy sampling cannot be reconstructed for {cell_key}: {missing_fields}; {remedy}",
    },
)
_UNSUPPORTED_REFUSALS = refusals(
    UnsupportedRequestError,
    {
        "readout.scope.source_digest_unavailable": "No source digest for {source} via {route}",
        "readout.scope.request_not_canonical": "Request cannot be canonicalized: {fields}",
        "readout.legacy.scope_unreconstructible": "Legacy scope cannot be reconstructed for {source}: {missing_fields}; {remedy}",
    },
)
_CELL_FAILURE_REFUSALS = refusals(
    InvalidRequestError,
    {
        "readout.cell.missing_arm": "Metric {metric} lacks declared arm {group_id}",
        "readout.cell.missing_metric_observations": "Metric {metric} has no usable observations for {group_id}",
        "readout.cell.unsupported_request": "Requested cell is unsupported: {reason}",
    },
)
_WIRE_REFUSALS = refusals(
    WireFormatError,
    {
        "readout.serialization.unsupported_version": "Unsupported envelope {kind} version {received}; supported {supported}",
        "readout.serialization.model_mismatch": "Collection {collection} does not contain row model {row_model}",
        "readout.serialization.duplicate_cell_key": "Duplicate serialized cell {cell_key} in {source_snapshot_id}",
        "readout.serialization.noncanonical_order": "Serialized {section} is not canonically ordered",
        "readout.serialization.identity_mismatch": "Serialized identity mismatch for {source_snapshot_id}: {reason} ({cell_key})",
        "readout.serialization.duplicate_object_key": "Duplicate JSON object key {key}",
        "readout.serialization.sequential_snapshot_required": "Serialized checkpoints require sequential snapshot {checkpoint_ids}",
    },
)

Population = Literal["assigned", "triggered"]
HypothesisKey = ArmHypothesisKey | SegmentHypothesisKey | ContrastHypothesisKey
SamplingEvidence = PValueEvidence | EValueEvidence | AsymptoticSequentialEvidence
Prior = Normal | StudentTPrior | MixturePrior


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


class _ReadoutModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    @model_validator(mode="after")
    def _freeze_mappings(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                canonical_digest_bytes(thaw(value))
                object.__setattr__(self, name, freeze(value))
        return self

    @field_serializer("*", when_used="json")
    def _serialize_values(self, value):
        return thaw(value)


class CellKey(_ReadoutModel):
    kind: Literal["arm", "segment", "contrast"]
    metric: str
    method: str | None
    method_role: Literal["decision", "sensitivity"] | None
    estimand: str | None
    aggregation: Literal["sum", "any"] | None = None
    analysis_population: Population
    value_scale: Literal["relative", "absolute"] | None = None
    inference: Literal["fixed", "always_valid", "asymptotic_mean"] | None = None
    alternative: Literal["two-sided", "greater", "less"] | None = None
    group_id: str | None = None
    control_group: str | None = None
    treatment_group: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    source: str | None = None
    ds: date | datetime | str | int | float | None = None
    ds_basis: Literal["calendar", "cohort"] | None = None

    @classmethod
    def from_row(cls, row):
        contrast = hasattr(row, "control_group")
        segment = getattr(row, "dimension", None) is not None
        values: dict[str, Any] = {name: getattr(row, name, None) for name in cls.model_fields}
        values.update(
            kind="contrast" if contrast else "segment" if segment else "arm",
            analysis_population=getattr(row, "analysis_population", "assigned"),
        )
        return cls(**values)

    def __hash__(self):
        return hash(canonical_json_bytes(self.model_dump(mode="json")))

    @field_validator("ds", mode="before")
    @classmethod
    def _deserialize_day_axis(cls, value):
        if isinstance(value, Mapping) and set(value) == {"label"}:
            return value["label"]
        if isinstance(value, Mapping) and set(value) == {"datetime"}:
            return datetime.fromisoformat(value["datetime"])
        return value

    @model_validator(mode="after")
    def _identity_axes(self):
        valid = (
            self.kind == "arm"
            and self.group_id is not None
            and all(
                value is None
                for value in (
                    self.control_group,
                    self.treatment_group,
                    self.dimension,
                    self.dimension_value,
                    self.aggregation,
                )
            )
            or self.kind == "segment"
            and self.group_id is not None
            and self.dimension is not None
            and self.dimension_value is not None
            and all(
                value is None
                for value in (self.control_group, self.treatment_group, self.aggregation)
            )
            or self.kind == "contrast"
            and self.control_group is not None
            and self.treatment_group is not None
            and self.aggregation is not None
            and all(
                value is None
                for value in (
                    self.group_id,
                    self.dimension,
                    self.dimension_value,
                    self.source,
                    self.ds,
                    self.ds_basis,
                    self.value_scale,
                )
            )
        )
        if self.ds_basis is not None and self.ds is None:
            valid = False
        if self.method is None:
            valid = valid and self.method_role is None and self.estimand is None
        else:
            valid = valid and self.method_role is not None and self.estimand is not None
        if self.kind == "contrast" and self.analysis_population != "assigned":
            valid = False
        if (self.dimension is None) != (self.dimension_value is None):
            valid = False
        if not valid:
            refuse_readout(
                "readout.cell.key_invalid",
                kind=self.kind,
                fields=tuple(
                    name
                    for name in (
                        "group_id",
                        "control_group",
                        "treatment_group",
                        "dimension",
                        "dimension_value",
                        "aggregation",
                        "method",
                        "method_role",
                        "estimand",
                    )
                    if getattr(self, name) is None
                ),
            )
        return self

    @field_serializer("ds", when_used="json")
    def _serialize_ds(self, value):
        return serialize_day_axis(value)

    def hypothesis(self) -> HypothesisKey:
        if self.kind == "arm":
            assert self.group_id is not None and self.estimand is not None
            return ArmHypothesisKey(self.metric, self.group_id, self.estimand)
        if self.kind == "segment":
            assert (
                self.group_id is not None
                and self.estimand is not None
                and self.dimension is not None
                and self.dimension_value is not None
            )
            return SegmentHypothesisKey(
                self.metric, self.group_id, self.estimand, self.dimension, self.dimension_value
            )
        assert self.control_group is not None and self.treatment_group is not None
        return ContrastHypothesisKey(self.metric, self.control_group, self.treatment_group)


def serialize_day_axis(value):
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return {"label": value}
    return value


class CellFailure(_ReadoutModel):
    hypothesis: HypothesisKey
    code: str
    context: Mapping[str, Any]


class SamplingInference(_ReadoutModel):
    available: bool
    evidence: SamplingEvidence | None = None
    reason_code: str | None = None
    reason_context: Mapping[str, Any] | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def _decode_infinite_evidence(cls, value):
        if isinstance(value, Mapping) and isinstance(value.get("log_e"), Mapping):
            marker = value["log_e"]
            if marker == {"special": "negative_infinity"}:
                return {**value, "log_e": float("-inf")}
            if marker == {"special": "positive_infinity"}:
                return {**value, "log_e": float("inf")}
        return value

    @field_serializer("evidence", mode="wrap", when_used="json")
    def _serialize_infinite_evidence(self, value, handler):
        from math import isinf

        payload = handler(value)
        if isinstance(value, EValueEvidence) and isinf(value.log_e):
            payload = dict(payload)
            payload["log_e"] = {
                "special": "negative_infinity" if value.log_e < 0 else "positive_infinity"
            }
        return payload


SamplingInference.model_rebuild(
    _types_namespace={
        "HypothesisKey": HypothesisKey,
        "LikelihoodCertificate": LikelihoodCertificate,
        "SequentialCheckpoint": SequentialCheckpoint,
        "AsymptoticSequentialResult": AsymptoticSequentialResult,
    }
)


class PosteriorInference(_ReadoutModel):
    available: bool | None = None
    model: Literal["normal", "mixture"] | None = None
    scale: Literal["log", "linear"] | None = None
    prior: Prior | None = None
    reason_code: str | None = None
    reason_context: Mapping[str, Any] | None = None


class IntegrityStatus(StrEnum):
    FAILED = "failed"
    NOT_REJECTED = "not_rejected"
    NOT_CHECKED_MISSING_DECLARATION = "not_checked_missing_declaration"
    NOT_CHECKED_MISSING_COUNTS = "not_checked_missing_counts"
    UNSUPPORTED_ASSIGNMENT = "unsupported_assignment"
    NOT_APPLICABLE = "not_applicable"


class IntegrityResult(_ReadoutModel):
    status: IntegrityStatus
    analysis_population: Population
    construction: Literal["always_valid", "none"]
    alpha: float | None
    observed: Mapping[str, int] | None
    expected: Mapping[str, float] | None
    randomization_grain: str | None
    code: str | None
    context: Mapping[str, Any]


class CellRecord(_ReadoutModel):
    cell: CellKey
    source_snapshot_id: str
    failure: CellFailure | None = None
    sampling: SamplingInference | None = None
    posterior: PosteriorInference | None = None


class PopulationRoster(_ReadoutModel):
    analysis_population: Population
    arms: tuple[str, ...]
    source: Literal["declared_allocation", "experiment_counts", "observed_union", "unknown"]
    complete: bool


class FamilyScope(_ReadoutModel):
    family_id: str
    analysis_population: Population
    source_snapshot_id: str
    view: Literal["run", "breakout", "daily", "asof"]
    dimension: str | None
    source: str | None
    name: str
    family: MultiplicityFamily | None = None
    members: tuple[CellKey, ...]
    complete: bool

    @model_validator(mode="after")
    def _canonical_members(self):
        if tuple(sorted(set(self.members), key=cell_order)) != self.members:
            refuse_readout("readout.serialization.noncanonical_order", section="family members")
        expected = family_identity(
            self.source_snapshot_id,
            self.analysis_population,
            self.view,
            self.dimension,
            self.source,
            self.name,
        )
        if self.family_id != expected:
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=self.source_snapshot_id,
                cell_key=None,
                reason="family id mismatch",
            )
        return self


def family_identity(source_snapshot_id, analysis_population, view, dimension, source, name):
    return (
        "sha256:"
        + sha256(
            canonical_json_bytes(
                {
                    "kind": "increment.readout.family",
                    "version": 1,
                    "source_snapshot_id": source_snapshot_id,
                    "analysis_population": analysis_population,
                    "view": view,
                    "dimension": dimension,
                    "source": source,
                    "name": name,
                }
            )
        ).hexdigest()
    )


class SourceReadoutScope(_ReadoutModel):
    source_snapshot_id: str
    source: Any = None
    cells: tuple[CellKey, ...]
    decision_cells: tuple[CellKey, ...]
    rosters: tuple[PopulationRoster, ...]
    decision_complete_by_population: Mapping[Population, bool]
    integrity: tuple[IntegrityResult, ...] = ()


class ReadoutScope(_ReadoutModel):
    snapshot_id: str
    cells: tuple[CellKey, ...]
    decision_cells: tuple[CellKey, ...]
    populations: tuple[Population, ...]
    families: tuple[FamilyScope, ...] = ()
    by_source: Mapping[str, SourceReadoutScope]

    @model_validator(mode="after")
    def _scope_membership(self):
        if tuple(sorted(set(self.cells), key=cell_order)) != self.cells:
            refuse_readout("readout.serialization.noncanonical_order", section="scope cells")
        if tuple(sorted(set(self.decision_cells), key=cell_order)) != self.decision_cells:
            refuse_readout("readout.serialization.noncanonical_order", section="decision cells")
        all_cells = set()
        all_decisions = set()
        for source_id, source in self.by_source.items():
            if source_id != source.source_snapshot_id:
                refuse_readout(
                    "readout.serialization.identity_mismatch",
                    source_snapshot_id=source_id,
                    cell_key=None,
                    reason="source key mismatch",
                )
            if (
                tuple(sorted(set(source.cells), key=cell_order)) != source.cells
                or tuple(sorted(set(source.decision_cells), key=cell_order))
                != source.decision_cells
            ):
                refuse_readout("readout.serialization.noncanonical_order", section="source cells")
            if not set(source.decision_cells) <= set(source.cells):
                refuse_readout(
                    "readout.serialization.identity_mismatch",
                    source_snapshot_id=source_id,
                    cell_key=None,
                    reason="decision cell outside scope",
                )
            all_cells.update(source.cells)
            all_decisions.update(source.decision_cells)
        if all_cells != set(self.cells) or all_decisions != set(self.decision_cells):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=self.snapshot_id,
                cell_key=None,
                reason="scope union mismatch",
            )
        if tuple(sorted(self.families, key=lambda family: family.family_id)) != self.families:
            refuse_readout("readout.serialization.noncanonical_order", section="families")
        if len({family.family_id for family in self.families}) != len(self.families):
            refuse_readout(
                "readout.serialization.duplicate_cell_key",
                source_snapshot_id=self.snapshot_id,
                cell_key="duplicate family id",
            )
        return self

    def decision_complete(self, population: Population) -> bool:
        values = [
            source.decision_complete_by_population[population]
            for source in self.by_source.values()
            if population in source.decision_complete_by_population
        ]
        return bool(values) and all(values)


class ReadoutMetadata(_ReadoutModel):
    scope: ReadoutScope
    cells: tuple[CellRecord, ...]
    partial: bool = False
    partial_reason: tuple[Literal["slice", "filter", "cross_snapshot_concat"], ...] = ()

    @property
    def sampling(self):
        return _FrozenMapping(
            {
                (record.source_snapshot_id, record.cell): record.sampling
                for record in self.cells
                if record.sampling is not None
            }
        )

    @property
    def posterior(self):
        return _FrozenMapping(
            {
                (record.source_snapshot_id, record.cell): record.posterior
                for record in self.cells
                if record.posterior is not None
            }
        )

    @model_validator(mode="after")
    def _records_are_unique_and_scoped(self):
        seen = set()
        for record in self.cells:
            key = (record.source_snapshot_id, record.cell)
            if key in seen:
                refuse_readout(
                    "readout.serialization.duplicate_cell_key",
                    source_snapshot_id=record.source_snapshot_id,
                    cell_key=record.cell.model_dump(mode="json"),
                )
            seen.add(key)
            source = self.scope.by_source.get(record.source_snapshot_id)
            if source is None or record.cell not in source.cells:
                refuse_readout(
                    "readout.serialization.identity_mismatch",
                    source_snapshot_id=record.source_snapshot_id,
                    cell_key=record.cell.model_dump(mode="json"),
                    reason="record outside scope",
                )
            if record.failure is not None:
                if record.failure.hypothesis != record.cell.hypothesis():
                    refuse_readout(
                        "readout.serialization.identity_mismatch",
                        source_snapshot_id=record.source_snapshot_id,
                        cell_key=record.cell.model_dump(mode="json"),
                        reason="failure hypothesis mismatch",
                    )
                if record.failure.context.get("method") not in (None, record.cell.method):
                    refuse_readout(
                        "readout.serialization.identity_mismatch",
                        source_snapshot_id=record.source_snapshot_id,
                        cell_key=record.cell.model_dump(mode="json"),
                        reason="failure method mismatch",
                    )
        expected = {
            (source_id, cell)
            for source_id, source in self.scope.by_source.items()
            for cell in source.cells
        }
        if seen != expected:
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=self.scope.snapshot_id,
                cell_key=None,
                reason="metadata record union mismatch",
            )
        if (
            tuple(
                sorted(
                    self.cells,
                    key=lambda record: (record.source_snapshot_id, cell_order(record.cell)),
                )
            )
            != self.cells
        ):
            refuse_readout("readout.serialization.noncanonical_order", section="metadata cells")
        return self

    @property
    def failures(self):
        return _FrozenMapping(
            {
                (record.source_snapshot_id, record.cell): record.failure
                for record in self.cells
                if record.failure is not None
            }
        )

    def record(self, source_snapshot_id, cell):
        return next(
            record
            for record in self.cells
            if record.source_snapshot_id == source_snapshot_id and record.cell == cell
        )

    def family_of(self, source_snapshot_id, cell):
        return next(
            (
                family
                for family in self.scope.families
                if family.source_snapshot_id == source_snapshot_id and cell in family.members
            ),
            None,
        )


class StreamingDigest:
    """Order-independent row multiset digest with constant additional state."""

    __slots__ = ("count", "accumulator")

    def __init__(self):
        self.count = 0
        self.accumulator = 0

    def update(self, row):
        self.accumulator = (
            self.accumulator + int.from_bytes(sha256(_digest_json_bytes(row)).digest(), "big")
        ) % (1 << 256)
        self.count += 1

    def hexdigest(self):
        return sha256(
            canonical_json_bytes(
                {
                    "kind": "increment.readout.rows",
                    "version": 1,
                    "count": self.count,
                    "accumulator": f"{self.accumulator:064x}",
                }
            )
        ).hexdigest()


def refuse_readout(code: str, **context: object) -> NoReturn:
    registry = (
        _WIRE_REFUSALS
        if code in _WIRE_REFUSALS
        else _UNSUPPORTED_REFUSALS
        if code in _UNSUPPORTED_REFUSALS
        else _CELL_FAILURE_REFUSALS
        if code in _CELL_FAILURE_REFUSALS
        else _REFUSALS
    )
    refuse(registry[code], **context)


def refuse_legacy_sampling(row: Any) -> NoReturn:
    """Refuse a prior-bound row whose prior-free sampling reference was not persisted."""
    refuse_readout(
        "readout.legacy.sampling_unreconstructible",
        cell_key={
            "metric": row.metric,
            "group_id": row.group_id,
            "method": row.method,
            "estimand": row.estimand,
        },
        missing_fields=("prior_free_reference_kind", "prior_free_reference_df"),
        remedy="recompute_from_source",
    )


def _restore_collection(collection_type, rows, metadata, source, sequential_snapshot):
    return collection_type(
        rows, metadata=metadata, source=source, sequential_snapshot=sequential_snapshot
    )


def _digest_json_bytes(value):
    return canonical_digest_bytes(value)


def cell_order(cell):
    return canonical_json_bytes(cell.model_dump(mode="json"))


def validate_collection(rows, metadata):
    if metadata is None:
        return
    seen = set()
    for row in rows:
        cell = CellKey.from_row(row)
        source_id = getattr(row, "source_snapshot_id", None)
        pair = (source_id, cell)
        if pair in seen:
            refuse_readout(
                "readout.collection.duplicate_cell_key",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
            )
        seen.add(pair)
        source = metadata.scope.by_source.get(source_id)
        if source is None or cell not in source.cells:
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="row outside source scope",
            )
        if hasattr(row, "sampling_available") and row.sampling_available is None:
            refuse_readout(
                "readout.legacy.sampling_unreconstructible",
                cell_key=cell.model_dump(mode="json"),
                missing_fields=("prior_free_reference_kind", "prior_free_reference_df"),
                remedy="recompute_from_source",
            )
        record = metadata.record(source_id, cell)
        failure = record.failure
        if getattr(row, "failure_code", None) != (
            None if failure is None else failure.code
        ) or getattr(row, "failure_context", None) != (
            None if failure is None else failure.context
        ):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="failure status mismatch",
            )
        sampling = record.sampling
        if sampling is not None and (
            row.sampling_available,
            row.sampling_reason_code,
            row.sampling_reason_context,
        ) != (sampling.available, sampling.reason_code, sampling.reason_context):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="sampling status mismatch",
            )
        posterior = record.posterior
        if posterior is not None and (
            row.posterior_available,
            row.posterior_model,
            row.posterior_scale,
            row.posterior_reason_code,
            row.posterior_reason_context,
        ) != (
            posterior.available,
            posterior.model,
            posterior.scale,
            posterior.reason_code,
            posterior.reason_context,
        ):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="posterior status mismatch",
            )
        complete = source.decision_complete_by_population.get(cell.analysis_population)
        if hasattr(row, "decision_scope_complete") and row.decision_scope_complete != complete:
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="decision completeness mismatch",
            )
        family = metadata.family_of(source_id, cell)
        if getattr(row, "family_id", None) != (None if family is None else family.family_id):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=source_id,
                cell_key=cell.model_dump(mode="json"),
                reason="family mismatch",
            )
    required = {(record.source_snapshot_id, record.cell) for record in metadata.cells}
    if not metadata.partial and seen != required:
        refuse_readout(
            "readout.serialization.identity_mismatch",
            source_snapshot_id=metadata.scope.snapshot_id,
            cell_key=None,
            reason="collection rows do not cover complete scope",
        )


def partial_metadata(metadata, rows, reason):
    if metadata is None:
        return None
    visible = {(row.source_snapshot_id, CellKey.from_row(row)) for row in rows}
    required = {(record.source_snapshot_id, record.cell) for record in metadata.cells}
    partial = visible != required
    reasons = (
        tuple(
            item
            for item in ("slice", "filter", "cross_snapshot_concat")
            if item in (*metadata.partial_reason, reason)
        )
        if partial
        else ()
    )
    return metadata.model_copy(update={"partial": partial, "partial_reason": reasons})


def _merge_source_scopes(left, right, ids):
    sources: dict[str, SourceReadoutScope] = {}
    source_descriptors = {}
    descriptor_keys = {}
    for collection, metadata in ((left, left.metadata), (right, right.metadata)):
        for source_id, scope in metadata.scope.by_source.items():
            descriptor = scope.source
            if (
                descriptor is not None
                and collection.source is not None
                and canonical_digest_bytes(descriptor) != canonical_digest_bytes(collection.source)
            ):
                refuse_readout(
                    "readout.collection.concat_metadata_mismatch",
                    snapshot_ids=ids,
                    fields=("source",),
                )
            if descriptor is None:
                descriptor = collection.source
            descriptor_key = canonical_digest_bytes(descriptor)
            if source_id not in sources:
                sources[source_id] = scope
                source_descriptors[source_id] = descriptor
                descriptor_keys[source_id] = descriptor_key
                continue
            if descriptor_keys[source_id] != descriptor_key:
                refuse_readout(
                    "readout.collection.concat_metadata_mismatch",
                    snapshot_ids=ids,
                    fields=("source",),
                )
            existing = sources[source_id]
            if existing.model_copy(update={"source": None}) != scope.model_copy(
                update={"source": None}
            ):
                refuse_readout(
                    "readout.collection.concat_metadata_mismatch",
                    snapshot_ids=ids,
                    fields=("by_source",),
                )
            if existing.source is None and scope.source is not None:
                sources[source_id] = scope
    result_source = (
        left.source
        if left.source is not None
        and right.source is not None
        and canonical_digest_bytes(left.source) == canonical_digest_bytes(right.source)
        else None
    )
    if result_source is None:
        for source_id, descriptor in source_descriptors.items():
            if descriptor is not None and sources[source_id].source is None:
                sources[source_id] = sources[source_id].model_copy(
                    update={"source": freeze(descriptor)}
                )
    return sources, result_source


def _record_order(record: CellRecord) -> tuple[str, bytes]:
    return record.source_snapshot_id, cell_order(record.cell)


def _merge_cell_identities(
    left, right, ids
) -> tuple[dict[tuple[str, CellKey], CellRecord], dict[tuple[str, CellKey], Any]]:
    records: dict[tuple[str, CellKey], CellRecord] = {
        (record.source_snapshot_id, record.cell): record for record in left.metadata.cells
    }
    for record in right.metadata.cells:
        key = (record.source_snapshot_id, record.cell)
        if key in records and records[key] != record:
            refuse_readout(
                "readout.collection.concat_cell_collision",
                snapshot_ids=ids,
                cell_key=record.cell.model_dump(mode="json"),
            )
        records[key] = record
    rows = {}
    for row in (*left, *right):
        key = (row.source_snapshot_id, CellKey.from_row(row))
        if key in rows and rows[key] != row:
            refuse_readout(
                "readout.collection.concat_cell_collision",
                snapshot_ids=ids,
                cell_key=key[1].model_dump(mode="json"),
            )
        rows[key] = row
    return records, rows


def concat_collection(left, right):
    if type(left) is not type(right):
        refuse_readout(
            "readout.collection.concat_type_mismatch",
            left=type(left).__name__,
            right=type(right).__name__,
        )
    if left.metadata is None and right.metadata is None:
        if left.source != right.source or left.sequential_snapshot != right.sequential_snapshot:
            refuse_readout(
                "readout.collection.concat_metadata_mismatch",
                snapshot_ids=(),
                fields=("source", "sequential_snapshot"),
            )
        return type(left)(
            [*left, *right], source=left.source, sequential_snapshot=left.sequential_snapshot
        )
    if left.metadata is None or right.metadata is None:
        refuse_readout(
            "readout.collection.concat_metadata_mismatch",
            snapshot_ids=tuple(
                sorted(
                    metadata.scope.snapshot_id
                    for metadata in (left.metadata, right.metadata)
                    if metadata is not None
                )
            ),
            fields=("metadata",),
        )
    if left.sequential_snapshot != right.sequential_snapshot:
        refuse_readout(
            "readout.collection.concat_metadata_mismatch",
            snapshot_ids=tuple(
                sorted((left.metadata.scope.snapshot_id, right.metadata.scope.snapshot_id))
            ),
            fields=("sequential_snapshot",),
        )
    if left.metadata.scope.snapshot_id != right.metadata.scope.snapshot_id and (
        left.sequential_snapshot is not None or right.sequential_snapshot is not None
    ):
        refuse_readout(
            "readout.collection.concat_metadata_mismatch",
            snapshot_ids=tuple(
                sorted((left.metadata.scope.snapshot_id, right.metadata.scope.snapshot_id))
            ),
            fields=("sequential_snapshot",),
        )
    a: ReadoutMetadata = left.metadata
    b: ReadoutMetadata = right.metadata
    ids = sorted([a.scope.snapshot_id, b.scope.snapshot_id])
    cross = a.scope.snapshot_id != b.scope.snapshot_id
    sources, result_source = _merge_source_scopes(left, right, ids)
    records, rows = _merge_cell_identities(left, right, ids)
    families = {family.family_id: family for family in a.scope.families}
    for family in b.scope.families:
        if family.family_id in families and families[family.family_id] != family:
            refuse_readout(
                "readout.collection.concat_metadata_mismatch",
                snapshot_ids=ids,
                fields=("families",),
            )
        families[family.family_id] = family
    snapshot = (
        a.scope.snapshot_id
        if not cross
        else "sha256:"
        + sha256(
            canonical_json_bytes(
                {"kind": "increment.readout.concat", "version": 1, "sources": sorted(sources)}
            )
        ).hexdigest()
    )
    scope = ReadoutScope(
        snapshot_id=snapshot,
        cells=tuple(sorted(set(a.scope.cells) | set(b.scope.cells), key=cell_order)),
        decision_cells=tuple(
            sorted(set(a.scope.decision_cells) | set(b.scope.decision_cells), key=cell_order)
        ),
        populations=tuple(sorted(set(a.scope.populations) | set(b.scope.populations))),
        families=tuple(families[key] for key in sorted(families)),
        by_source=dict(sorted(sources.items())),
    )
    ordered_records = sorted(records.values(), key=_record_order)
    metadata = ReadoutMetadata(scope=scope, cells=tuple(ordered_records))
    metadata = partial_metadata(
        metadata,
        rows.values(),
        "cross_snapshot_concat" if cross else "slice",
    )
    return type(left)(
        rows.values(),
        metadata=metadata,
        source=result_source,
        sequential_snapshot=getattr(left, "sequential_snapshot", None),
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            refuse_readout("readout.serialization.duplicate_object_key", key=key)
        result[key] = value
    return result


class ReadoutResults:
    """Versioned saved-result boundary; decoding returns the existing collection."""

    @staticmethod
    def model_validate_json(text):
        import json

        from increment.breakout.estimates import (
            BreakoutEstimates,
            DailyLiftEstimates,
            DailyMetricValues,
            LiftEstimates,
        )
        from increment.estimation.contrast_results import ContrastResult, ContrastResults
        from increment.sequential_state import SequentialSnapshot

        payload = json.loads(text, object_pairs_hook=_unique_object)
        if payload.get("kind") != "increment.readout" or payload.get("schema_version") != 1:
            refuse_readout(
                "readout.serialization.unsupported_version",
                kind=payload.get("kind"),
                received=payload.get("schema_version"),
                supported=(1,),
            )
        collection_name = payload.get("collection")
        models: dict[str, type[BaseModel]] = {
            "LiftEstimates": LiftEstimates._model,
            "BreakoutEstimates": BreakoutEstimates._model,
            "DailyMetricValues": DailyMetricValues._model,
            "DailyLiftEstimates": DailyLiftEstimates._model,
            "ContrastResults": ContrastResult,
        }
        model = models.get(collection_name)
        if model is None or model.__name__ != payload.get("row_model"):
            refuse_readout(
                "readout.serialization.model_mismatch",
                collection=collection_name,
                row_model=payload.get("row_model"),
            )
        metadata = (
            None
            if payload.get("metadata") is None
            else ReadoutMetadata.model_validate(payload["metadata"])
        )
        if metadata is not None and metadata.scope.snapshot_id != payload.get("snapshot_id"):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=payload.get("snapshot_id"),
                cell_key=None,
                reason="snapshot mismatch",
            )
        sequential = (
            None
            if payload.get("sequential_snapshot") is None
            else SequentialSnapshot.model_validate_json(json.dumps(payload["sequential_snapshot"]))
        )
        rows: list[Any] = [model.model_validate(row) for row in payload["rows"]]
        checkpoints = []
        for row in rows:
            result = getattr(row, "sequential_result", None)
            if result is not None:
                checkpoints.append(result.checkpoint)
        if metadata is not None:
            for record in metadata.cells:
                evidence = None if record.sampling is None else record.sampling.evidence
                checkpoint = getattr(evidence, "checkpoint", None)
                if checkpoint is not None:
                    checkpoints.append(checkpoint)
        if checkpoints and sequential is None:
            refuse_readout(
                "readout.serialization.sequential_snapshot_required",
                checkpoint_ids=tuple(
                    sorted(checkpoint.checkpoint_id for checkpoint in checkpoints)
                ),
            )
        if sequential is not None:
            for checkpoint in checkpoints:
                checkpoint.verify_snapshot(sequential)
        kwargs = {
            "metadata": metadata,
            "source": payload.get("source"),
            "sequential_snapshot": sequential,
        }
        if collection_name == "LiftEstimates":
            return LiftEstimates(rows, **kwargs)
        if collection_name == "BreakoutEstimates":
            return BreakoutEstimates(rows, **kwargs)
        if collection_name == "DailyMetricValues":
            return DailyMetricValues(rows, **kwargs)
        if collection_name == "DailyLiftEstimates":
            return DailyLiftEstimates(rows, **kwargs)
        if collection_name == "ContrastResults":
            return ContrastResults(rows, **kwargs)
        refuse_readout(
            "readout.serialization.model_mismatch",
            collection=collection_name,
            row_model=payload.get("row_model"),
        )


def dump_collection(results):
    import json

    metadata = results.metadata
    return json.dumps(
        {
            "kind": "increment.readout",
            "schema_version": 1,
            "collection": type(results).__name__,
            "row_model": results._model.__name__,
            "snapshot_id": None if metadata is None else metadata.scope.snapshot_id,
            "source": results.source,
            "metadata": None if metadata is None else metadata.model_dump(mode="json"),
            "sequential_snapshot": None
            if results.sequential_snapshot is None
            else results.sequential_snapshot.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in results],
        },
        allow_nan=False,
    )
