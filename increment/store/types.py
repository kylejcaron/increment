from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from typing import Protocol, cast

import pyarrow as pa

from increment._canonical import canonical_json_bytes
from increment.estimation.readout_types import ReadoutMetadata
from increment.kernel.contracts import (
    BreakoutIdentity,
    CellIdentity,
    CompiledPlanBinding,
    contract_hash,
)
from increment.result_envelope import (
    AnalysisEnvelope,
    ResultCell,
    _standard_error,
    _standard_error_df,
)
from increment.store._refusals import refuse_store


@dataclass(frozen=True, slots=True)
class Run:
    experiment_id: str
    run_id: str
    analysis_spec_hash: str
    execution_identity_hash: str
    registration_hash: str | None
    expected_roster: tuple[CellIdentity, ...]
    cells: tuple[ResultCell, ...]
    metadata: ReadoutMetadata | None
    captured_at: datetime
    committed_at: datetime
    scope_analysis_hashes: tuple[str, ...] = ()
    cell_scope_hashes: tuple[tuple[str, str], ...] = ()
    manifest_version: int = 2
    decision_plan: CompiledPlanBinding | None = None
    _stored_metadata_sha256: str | None = dataclass_field(default=None, repr=False, compare=False)
    _normalized_metadata_sha256: str | None = dataclass_field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:  # noqa: C901
        for field in ("experiment_id", "run_id", "analysis_spec_hash", "execution_identity_hash"):
            if not getattr(self, field):
                refuse_store(
                    "store.run.identity_incomplete",
                    "A complete run identity is required",
                    field=field,
                )
        for field in ("captured_at", "committed_at"):
            value = getattr(self, field)
            if value.tzinfo is None or value.utcoffset() is None:
                refuse_store(
                    "store.run.timestamp_invalid",
                    "Run timestamps must be timezone-aware",
                    field=field,
                )
        if self.committed_at < self.captured_at:
            refuse_store(
                "store.run.timestamp_invalid",
                "Run commit cannot precede capture",
                field="committed_at",
            )
        expected = tuple(identity.hash() for identity in self.expected_roster)
        actual = tuple(cell.identity.hash() for cell in self.cells)
        if (
            not expected
            or len(set(expected)) != len(expected)
            or len(set(actual)) != len(actual)
            or set(expected) != set(actual)
        ):
            refuse_store(
                "store.run.roster_mismatch",
                "Run cells must exactly match its expected roster",
                expected=len(expected),
                actual=len(actual),
            )
        for cell in self.cells:
            identity = cell.identity
            if (
                identity.source_snapshot_id is None
                or identity.scope is None
                or cell.source_snapshot_id is None
            ):
                refuse_store(
                    "store.run.identity_incomplete",
                    "Cell snapshot and scope identity are required",
                    field="source_snapshot_id/scope",
                )
            expected_segment = (
                (
                    BreakoutIdentity(
                        property=identity.key.dimension or "",
                        source=identity.key.source,
                    ),
                )
                if identity.key.kind == "segment"
                else ()
            )
            if (
                cell.source_snapshot_id != identity.source_snapshot_id
                or identity.key.analysis_population != identity.scope.population
                or identity.scope.segment != expected_segment
            ):
                refuse_store(
                    "store.run.identity_mismatch",
                    "Cell identity disagrees with its snapshot or scope",
                    field="source_snapshot_id/population/segment",
                )
            if identity.experiment_id != self.experiment_id:
                refuse_store(
                    "store.run.identity_mismatch",
                    "Run header disagrees with cell identity",
                    field="experiment_id",
                )
            if (
                cell.provenance.run_id != self.run_id
                or cell.provenance.execution_identity_hash != self.execution_identity_hash
                or cell.provenance.source_content_digest != identity.source_content_digest
            ):
                refuse_store(
                    "store.run.provenance_mismatch",
                    "Cell provenance disagrees with its run",
                    run_id=self.run_id,
                )
        if len({cell.provenance.artifact_manifest_sha256 for cell in self.cells}) > 1:
            refuse_store(
                "store.run.provenance_mismatch",
                "Cell artifact manifests disagree within the run",
                run_id=self.run_id,
            )
        if len({cell.provenance.artifact_relation_digests for cell in self.cells}) > 1:
            refuse_store(
                "store.run.provenance_mismatch",
                "Cell artifact relation digests disagree within the run",
                run_id=self.run_id,
            )
        if self.decision_plan is not None:
            if self.registration_hash != self.decision_plan.registration_hash:
                refuse_store(
                    "store.run.identity_mismatch",
                    "Compiled decision plan differs from the run registration",
                )
            if self.manifest_version < 3:
                object.__setattr__(self, "manifest_version", 3)
        scopes = self.scope_analysis_hashes or (self.analysis_spec_hash,)
        if len(set(scopes)) != len(scopes) or not scopes or (self.analysis_spec_hash not in scopes):
            refuse_store("store.run.scope_manifest_invalid", "Run scope manifest is invalid")
        object.__setattr__(self, "scope_analysis_hashes", tuple(scopes))
        mapping = dict(self.cell_scope_hashes)
        if (
            not self.cell_scope_hashes
            or len(mapping) != len(self.cell_scope_hashes)
            or set(mapping) != set(actual)
            or not set(mapping.values()) <= set(scopes)
        ):
            refuse_store(
                "store.run.scope_mapping_invalid",
                "Cell-to-scope mapping must cover the roster exactly once",
            )
        for cell in self.cells:
            scope_hash = mapping[cell.identity.hash()]
            if cell.identity.analysis_spec_hash != scope_hash:
                refuse_store(
                    "store.run.identity_mismatch",
                    "Cell analysis identity disagrees with scope manifest",
                    field="analysis_spec_hash",
                )
        if self.metadata is None:
            refuse_store("store.run.metadata_mismatch", "Complete run metadata is required")
        if not callable(getattr(self.metadata, "model_dump", None)):
            refuse_store(
                "store.run.metadata_mismatch", "Run metadata must support its portable schema"
            )
        records = getattr(self.metadata, "cells", ())
        metadata_keys = [(record.source_snapshot_id, record.cell) for record in records]
        cell_keys = [(cell.identity.source_snapshot_id, cell.identity.key) for cell in self.cells]
        if len(set(metadata_keys)) != len(metadata_keys) or set(metadata_keys) != set(cell_keys):
            refuse_store(
                "store.run.metadata_mismatch",
                "Metadata records must correspond one-to-one with stored cells",
            )
        scope = getattr(self.metadata, "scope", None)
        if scope is not None:
            for family in scope.families:
                family_keys = {(family.source_snapshot_id, key) for key in family.members}
                if not family_keys <= set(cell_keys):
                    refuse_store(
                        "store.run.metadata_mismatch",
                        "Family members must resolve to stored cells",
                        family_id=family.family_id,
                    )

    @property
    def source_content_digest(self) -> str | None:
        digests = {cell.identity.source_content_digest for cell in self.cells}
        if None in digests:
            return None
        return "sha256:" + contract_hash(
            "store_source_content_v1",
            {"source_content_digests": sorted(cast(set[str], digests))},
        )

    @property
    def source_content_digest_unavailable_reason(self) -> str | None:
        if self.source_content_digest is None:
            return (
                getattr(self.metadata, "source_content_digest_unavailable_reason", None)
                or "source_input_digest_unavailable"
            )
        return None

    @property
    def artifact_manifest_sha256(self) -> str | None:
        return self.cells[0].provenance.artifact_manifest_sha256

    @property
    def artifact_relation_digests(self) -> tuple[tuple[str, str], ...]:
        return self.cells[0].provenance.artifact_relation_digests

    @property
    def manifest_hash(self) -> str:
        """Identity of the committed metadata, retaining stored order while unchanged."""
        current_metadata_sha256 = self._metadata_sha256()
        stored_metadata_sha256 = (
            self._stored_metadata_sha256
            if self._normalized_metadata_sha256 == current_metadata_sha256
            else None
        )
        return self._manifest_hash(stored_metadata_sha256 or current_metadata_sha256)

    def _current_manifest_hash(self) -> str:
        """Hash the current canonical representation for newly written stores."""
        return self._manifest_hash(self._metadata_sha256())

    def _metadata_sha256(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(cast(ReadoutMetadata, self.metadata).model_dump(mode="json"))
            ).hexdigest()
        )

    def _manifest_hash(self, metadata_sha256: str) -> str:
        payload: dict[str, object] = {
            "manifest_version": self.manifest_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "analysis_spec_hash": self.analysis_spec_hash,
            "execution_identity_hash": self.execution_identity_hash,
            "registration_hash": self.registration_hash,
            "source_content_digest": self.source_content_digest,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "artifact_relation_digests": [list(item) for item in self.artifact_relation_digests],
            "cell_scope_hashes": [list(item) for item in self.cell_scope_hashes],
            "scope_analysis_hashes": list(self.scope_analysis_hashes),
            "expected_roster": [identity.hash() for identity in self.expected_roster],
            "cell_payload_sha256": [
                [
                    cell.identity.hash(),
                    "sha256:" + hashlib.sha256(canonical_json_bytes(cell.to_payload())).hexdigest(),
                ]
                for cell in sorted(self.cells, key=lambda item: item.identity.hash())
            ],
            "metadata_sha256": metadata_sha256,
            "captured_at": self.captured_at.isoformat(),
            "committed_at": self.committed_at.isoformat(),
        }
        if self.manifest_version >= 3:
            payload["decision_plan"] = (
                None if self.decision_plan is None else self.decision_plan.canonical_payload()
            )
        return "sha256:" + contract_hash(f"store_run_manifest_v{self.manifest_version}", payload)


class RunHandle(Protocol):
    @property
    def run(self) -> Run: ...

    def get(self, identity: CellIdentity) -> ResultCell: ...
    def envelope(self) -> AnalysisEnvelope: ...


@dataclass(frozen=True, slots=True)
class HistorySelector:
    metric_hash: str
    estimator_id: str
    method_config_hash: str
    method_instance_id: str
    scope_hash: str
    contrast: str | None = None
    role: str | None = None


def _matches_contrast(run: Run, cell: ResultCell, contrast: str) -> bool:
    key = cell.identity.key
    if key.kind == "contrast":
        return (
            key.control_group is not None
            and key.treatment_group is not None
            and f"{key.control_group}:{key.treatment_group}" == contrast
        )
    if key.kind not in {"arm", "segment"} or run.metadata is None:
        return False
    source = run.metadata.scope.by_source.get(cell.source_snapshot_id)
    if source is None:
        return False
    return any(
        roster.analysis_population == key.analysis_population
        and roster.control_group is not None
        and key.group_id is not None
        and f"{roster.control_group}:{key.group_id}" == contrast
        for roster in source.rosters
    )


def history_table(records: list[tuple[Run, ResultCell]]) -> pa.Table:
    import pyarrow as pa

    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("execution_identity_hash", pa.string()),
            ("cell", pa.string()),
            ("point", pa.float64()),
            ("lower", pa.float64()),
            ("upper", pa.float64()),
            ("se", pa.float64()),
            ("df", pa.float64()),
            ("reason_code", pa.string()),
        ]
    )
    rows = []
    for run, cell in records:
        payload = cell.to_payload()
        retained = payload["row_payload"] or payload["retained_row"]
        summary = cell.summary
        rows.append(
            {
                "run_id": run.run_id,
                "execution_identity_hash": run.execution_identity_hash,
                "cell": canonical_json_bytes(cell.to_payload()).decode(),
                "point": None if summary is None else summary.point,
                "lower": None if summary is None else summary.lower,
                "upper": None if summary is None else summary.upper,
                "se": _standard_error(retained),
                "df": _standard_error_df(retained),
                "reason_code": cell.components["sampling"][0].code,
            }
        )
    return pa.Table.from_pylist(rows, schema=schema)


def _number(payload: object, *names: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidates = [payload, *(value for value in payload.values() if isinstance(value, dict))]
    for candidate in candidates:
        for name in names:
            value = candidate.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


@dataclass(frozen=True, slots=True)
class _PinnedRun:
    run: Run

    def get(self, identity: CellIdentity) -> ResultCell:
        matches = [cell for cell in self.run.cells if cell.identity.hash() == identity.hash()]
        if len(matches) != 1:
            refuse_store(
                "store.cell.not_found",
                "Cell identity does not resolve exactly once",
                identity_hash=identity.hash(),
            )
        return matches[0]

    def envelope(self) -> AnalysisEnvelope:
        return AnalysisEnvelope(self.run.cells, self.run.metadata)  # type: ignore[arg-type]


def pinned_run(run: Run) -> RunHandle:
    return _PinnedRun(run)


def run_digest(run: Run) -> str:
    return (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes({"manifest": run.manifest_hash})).hexdigest()
    )


def run_record_key(experiment_id: str, analysis_spec_hash: str, run_id: str) -> str:
    encoded = canonical_json_bytes(
        {
            "schema": 1,
            "kind": "increment.store.run_record_key",
            "payload": [experiment_id, analysis_spec_hash, run_id],
        }
    )
    return hashlib.sha256(encoded).hexdigest()
