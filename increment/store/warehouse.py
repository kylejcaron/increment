from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from increment._warehouse_namespace import (
    MAX_IDENTIFIER_BYTES,
    check_schema_name_length,
    ensure_schema,
    schema_exists,
    split_schema,
)
from increment.kernel.program_refusals import refuse_store, refuse_unavailable_results_schema
from increment.store.parquet import ParquetResultsStore
from increment.store.types import (
    HistorySelector,
    Run,
    RunHandle,
    _matches_contrast,
    history_table,
    pinned_run,
    run_record_key,
)

_RUN_TABLE_PREFIX = "increment_run_"
_RUN_KEY_DIGITS = MAX_IDENTIFIER_BYTES - len(_RUN_TABLE_PREFIX)
_RUN_TABLE_NAME = re.compile(rf"{_RUN_TABLE_PREFIX}[0-9a-f]{{{_RUN_KEY_DIGITS}}}")
_MISSING_NAMESPACE = re.compile(
    r"\b(?:schema|database|dataset)\b.*\b(?:does not exist|not found)\b", re.IGNORECASE | re.DOTALL
)
# Snowflake reports a schema the role may not read as "does not exist or not authorized".
_UNAUTHORIZED = re.compile(r"authori[sz]|permission|privilege|access denied", re.IGNORECASE)


def _reports_missing_schema(error: Exception, schema: str) -> bool:
    """Whether *error* is the backend saying *schema* does not exist, and nothing else."""
    message = str(error)
    return (
        schema.lower() in message.lower()
        and _MISSING_NAMESPACE.search(message) is not None
        and _UNAUTHORIZED.search(message) is None
    )


def run_table_name(run_key: str) -> str:
    """A run's table: the prefix and the key's first 49 hex digits, 63 bytes in all."""
    return _RUN_TABLE_PREFIX + run_key[:_RUN_KEY_DIGITS]


class WarehouseResultsStore:
    """Publish one immutable, generation-scoped table per completed run.

    A declared schema, ``schema`` or ``catalog.schema``, is created when the
    first run is saved; reading never creates it. Saving refuses with
    ``store.run.schema_unavailable`` when the schema can be neither found nor
    created. A schema or catalog name longer than the warehouse allows
    (PostgreSQL 63 bytes, Snowflake 255 characters, BigQuery 1,024 bytes)
    refuses at construction with ``store.schema_name_too_long``.
    """

    def __init__(self, backend, *, schema: str | None = None) -> None:
        location = None if schema is None else split_schema(schema)
        if location is not None:
            check_schema_name_length(
                backend, store="WarehouseResultsStore", catalog=location[0], schema=location[1]
            )
        self.backend = backend
        self.schema = schema
        self._schema_ready = schema is None

    def _ensure_schema(self) -> None:
        schema = self.schema
        if self._schema_ready or schema is None:
            return
        location = split_schema(schema)
        if location is None or not ensure_schema(
            self.backend, schema=location[1], catalog=location[0]
        ):
            refuse_unavailable_results_schema(schema)
        self._schema_ready = True

    def _name(self, experiment_id: str, analysis_spec_hash: str, run_id: str) -> str:
        return run_table_name(run_record_key(experiment_id, analysis_spec_hash, run_id))

    def put(self, run: Run) -> None:
        from increment.result_envelope import AnalysisEnvelope

        if run.empty_selection_reason is not None:
            refuse_store(
                "store.run.empty",
                "An empty metric selection cannot be saved",
                reason=run.empty_selection_reason,
            )

        self._ensure_schema()
        name = self._name(run.experiment_id, run.analysis_spec_hash, run.run_id)
        if name in self.backend.list_tables(database=self.schema):
            refuse_store(
                "store.run.duplicate",
                "Run IDs are immutable and already committed",
                run_id=run.run_id,
            )
        table = AnalysisEnvelope(run.cells, run.metadata).to_arrow(include_capture_metadata=True)
        metadata = dict(table.schema.metadata or {})
        metadata[b"increment.run_manifest.v2"] = json.dumps(
            {
                "experiment_id": run.experiment_id,
                "run_id": run.run_id,
                "analysis_spec_hash": run.analysis_spec_hash,
                "execution_identity_hash": run.execution_identity_hash,
                "registration_hash": run.registration_hash,
                "source_content_digest": run.source_content_digest,
                "source_content_digest_unavailable_reason": run.source_content_digest_unavailable_reason,
                "artifact_manifest_sha256": run.artifact_manifest_sha256,
                "artifact_relation_digests": [list(item) for item in run.artifact_relation_digests],
                "scope_analysis_hashes": list(run.scope_analysis_hashes),
                "cell_scope_hashes": [list(item) for item in run.cell_scope_hashes],
                "expected_roster": [item.model_dump(mode="json") for item in run.expected_roster],
                "manifest_version": run.manifest_version,
                "captured_at": run.captured_at.isoformat(),
                "committed_at": run.committed_at.isoformat(),
                "decision_plan": (
                    None if run.decision_plan is None else run.decision_plan.to_payload()
                ),
                "manifest_hash": run._current_manifest_hash(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        fd, temporary = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        try:
            pq.write_table(table.replace_schema_metadata(metadata), temporary)
            payload = Path(temporary).read_bytes()
        finally:
            os.unlink(temporary)
        storage = pa.table({"payload": pa.array([payload], type=pa.binary())})
        try:
            self.backend.create_table(name, obj=storage, database=self.schema, overwrite=False)
        except Exception:
            if name in self.backend.list_tables(database=self.schema):
                refuse_store(
                    "store.run.duplicate",
                    "Run IDs are immutable and already committed",
                    run_id=run.run_id,
                )
            raise

    def _load_name(self, name: str) -> Run:
        payloads = (
            self.backend.table(name, database=self.schema).to_pyarrow()["payload"].to_pylist()
        )
        if len(payloads) != 1:
            refuse_store("store.run.corrupt", "Warehouse run record is corrupt", table=name)
        fd, temporary = tempfile.mkstemp(suffix=".parquet")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payloads[0])
            return ParquetResultsStore(Path(temporary).parent)._load(Path(temporary))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _is_committed_name(name: str) -> bool:
        return _RUN_TABLE_NAME.fullmatch(name) is not None

    def _read_manifest_name(self, name: str) -> dict[str, object]:
        try:
            payloads = (
                self.backend.table(name, database=self.schema).to_pyarrow()["payload"].to_pylist()
            )
        except Exception as exc:
            refuse_store(
                "store.run.corrupt",
                "Committed warehouse run record is unreadable or inconsistent",
                table=name,
                error=type(exc).__name__,
            )
        if len(payloads) != 1:
            refuse_store(
                "store.run.corrupt",
                "Warehouse run record must contain one payload",
                table=name,
            )
        try:
            metadata = pq.read_schema(pa.BufferReader(payloads[0])).metadata or {}
            manifest = json.loads(metadata[b"increment.run_manifest.v2"])
        except Exception as exc:
            refuse_store(
                "store.run.corrupt",
                "Committed warehouse run record is unreadable or inconsistent",
                table=name,
                error=type(exc).__name__,
            )
        if not isinstance(manifest, dict):
            refuse_store(
                "store.run.corrupt",
                "Warehouse run manifest must be an object",
                table=name,
            )
        experiment_id = manifest.get("experiment_id")
        analysis_spec_hash = manifest.get("analysis_spec_hash")
        run_id = manifest.get("run_id")
        if not all(
            isinstance(value, str) and value
            for value in (experiment_id, analysis_spec_hash, run_id)
        ):
            refuse_store(
                "store.run.corrupt",
                "Warehouse run manifest is missing routing coordinates",
                table=name,
            )
        if self._name(experiment_id, analysis_spec_hash, run_id) != name:
            refuse_store(
                "store.run.corrupt",
                "Committed warehouse run routing coordinates do not match its table",
                table=name,
            )
        return cast(dict[str, object], manifest)

    def _load_committed_name(self, name: str) -> Run:
        try:
            return self._load_name(name)
        except Exception as exc:
            refuse_store(
                "store.run.corrupt",
                "Committed warehouse run record is unreadable or inconsistent",
                table=name,
                error=type(exc).__name__,
            )

    def _table_names(self) -> list[str]:
        """The store's tables; none in a declared schema that does not exist."""
        location = None if self.schema is None else split_schema(self.schema)
        if (
            location is not None
            and schema_exists(self.backend, schema=location[1], catalog=location[0]) is False
        ):
            return []
        try:
            return list(self.backend.list_tables(database=self.schema))
        except Exception as error:
            if location is not None and _reports_missing_schema(error, location[1]):
                return []
            raise

    def open_run(
        self, experiment_id: str, analysis_spec_hash: str, run_id: str | None = None
    ) -> RunHandle:
        names = self._table_names()
        if run_id is not None:
            name = self._name(experiment_id, analysis_spec_hash, run_id)
            if name in names:
                run = self._load_committed_name(name)
                if (
                    run.experiment_id != experiment_id
                    or run.analysis_spec_hash != analysis_spec_hash
                    or run.run_id != run_id
                ):
                    refuse_store(
                        "store.run.corrupt",
                        "Committed run identity does not match the requested key",
                        run_id=run_id,
                    )
                matches = [run]
            else:
                matches = []
        else:
            matches = []
            for name in names:
                if not self._is_committed_name(name):
                    continue
                manifest = self._read_manifest_name(name)
                if (
                    manifest.get("experiment_id") != experiment_id
                    or manifest.get("analysis_spec_hash") != analysis_spec_hash
                ):
                    continue
                run = self._load_committed_name(name)
                if (
                    run.experiment_id != experiment_id
                    or run.analysis_spec_hash != analysis_spec_hash
                ):
                    refuse_store(
                        "store.run.corrupt",
                        "Committed warehouse run identity does not match its manifest",
                        table=name,
                    )
                matches.append(run)
        if not matches:
            refuse_store(
                "store.run.not_found",
                "No completed run matches the request",
                experiment_id=experiment_id,
                analysis_spec_hash=analysis_spec_hash,
                run_id=run_id,
            )
        return pinned_run(max(matches, key=lambda run: run.committed_at))

    def query(self, experiment_id: str, analysis_spec_hash: str, run_id: str | None = None):
        return self.open_run(experiment_id, analysis_spec_hash, run_id).envelope().to_arrow()

    def history(self, experiment_id: str, selector: HistorySelector):
        records = []
        for name in self._table_names():
            if not self._is_committed_name(name):
                continue
            manifest = self._read_manifest_name(name)
            if manifest.get("experiment_id") != experiment_id:
                continue
            run = self._load_committed_name(name)
            cells = [
                cell
                for cell in run.cells
                if (
                    cell.identity.metric_hash == selector.metric_hash
                    and cell.identity.estimator_id == selector.estimator_id
                    and cell.identity.method_config_hash == selector.method_config_hash
                    and cell.identity.method_instance_id == selector.method_instance_id
                    and cell.identity.scope is not None
                    and cell.identity.scope.hash() == selector.scope_hash
                )
                and (selector.contrast is None or _matches_contrast(run, cell, selector.contrast))
                and (selector.role is None or cell.identity.key.method_role == selector.role)
            ]
            if len(cells) > 1:
                refuse_store(
                    "store.history.ambiguous",
                    "History selector resolves ambiguously",
                    run_id=run.run_id,
                    matches=len(cells),
                )
            if cells:
                records.append((run, cells[0]))
        return history_table(records)
