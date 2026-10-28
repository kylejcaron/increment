from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from increment.kernel.contracts import CellIdentity
from increment.result_envelope import AnalysisEnvelope
from increment.store._refusals import refuse_store
from increment.store.types import (
    HistorySelector,
    Run,
    RunHandle,
    _matches_contrast,
    history_table,
    pinned_run,
    run_record_key,
)

if TYPE_CHECKING:
    from increment.store.producer import PersistenceIdentity


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(pq.read_schema(path).metadata[b"increment.run_manifest.v2"])
    except Exception as exc:
        refuse_store(
            "store.run.corrupt",
            "Committed run record is unreadable or inconsistent",
            run_id=path.stem,
            error=type(exc).__name__,
        )
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(field), str) or not manifest[field]
        for field in ("experiment_id", "analysis_spec_hash", "run_id")
    ):
        refuse_store(
            "store.run.corrupt",
            "Committed run record is unreadable or inconsistent",
            run_id=path.stem,
            error="ValueError",
        )
    if path.stem != run_record_key(
        manifest["experiment_id"], manifest["analysis_spec_hash"], manifest["run_id"]
    ):
        refuse_store(
            "store.run.corrupt",
            "Committed run routing coordinates do not match its record key",
            run_id=path.stem,
        )
    return manifest


class ParquetResultsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def persist_results(
        self,
        results: object,
        experiment_id: str,
        run_id: str,
        identity_record: PersistenceIdentity,
    ) -> None:
        from increment.store.producer import persist_results

        persist_results(results, self, experiment_id, run_id, identity_record)

    def _path(self, experiment_id: str, analysis_spec_hash: str, run_id: str) -> Path:
        return self.path / f"{run_record_key(experiment_id, analysis_spec_hash, run_id)}.parquet"

    def put(self, run: Run) -> None:
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
                "captured_at": run.captured_at.isoformat(),
                "committed_at": run.committed_at.isoformat(),
                "manifest_version": run.manifest_version,
                "manifest_hash": run.manifest_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        table = table.replace_schema_metadata(metadata)
        target = self._path(run.experiment_id, run.analysis_spec_hash, run.run_id)
        fd, temporary = tempfile.mkstemp(dir=self.path, prefix=".run-")
        os.close(fd)
        try:
            pq.write_table(table, temporary)
            try:
                os.link(temporary, target)
            except FileExistsError:
                refuse_store(
                    "store.run.duplicate",
                    "Run IDs are immutable and already committed",
                    run_id=run.run_id,
                )
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self, path: Path) -> Run:
        table = pq.read_table(path)
        manifest = json.loads(table.schema.metadata[b"increment.run_manifest.v2"])
        envelope = AnalysisEnvelope.from_arrow(table)
        run = Run(
            experiment_id=manifest["experiment_id"],
            run_id=manifest["run_id"],
            analysis_spec_hash=manifest["analysis_spec_hash"],
            execution_identity_hash=manifest["execution_identity_hash"],
            registration_hash=manifest["registration_hash"],
            expected_roster=tuple(
                CellIdentity.from_payload(v) for v in manifest["expected_roster"]
            ),
            cells=envelope.cells,
            metadata=envelope.metadata,
            captured_at=datetime.fromisoformat(manifest["captured_at"]),
            committed_at=datetime.fromisoformat(manifest["committed_at"]),
            scope_analysis_hashes=tuple(manifest["scope_analysis_hashes"]),
            cell_scope_hashes=tuple(tuple(item) for item in manifest["cell_scope_hashes"]),
            manifest_version=manifest["manifest_version"],
        )
        if (
            run.manifest_hash != manifest["manifest_hash"]
            or run.source_content_digest != manifest["source_content_digest"]
            or run.artifact_manifest_sha256 != manifest["artifact_manifest_sha256"]
            or [list(item) for item in run.artifact_relation_digests]
            != manifest["artifact_relation_digests"]
        ):
            refuse_store(
                "store.run.corrupt", "Stored run manifest does not match payload", run_id=run.run_id
            )
        return run

    @staticmethod
    def _is_committed_path(path: Path) -> bool:
        return path.suffix == ".parquet" and re.fullmatch(r"[0-9a-f]{64}", path.stem) is not None

    def _load_committed(self, path: Path) -> Run:
        try:
            return self._load(path)
        except Exception as exc:
            refuse_store(
                "store.run.corrupt",
                "Committed run record is unreadable or inconsistent",
                run_id=path.stem,
                error=type(exc).__name__,
            )

    def open_run(
        self, experiment_id: str, analysis_spec_hash: str, run_id: str | None = None
    ) -> RunHandle:
        if run_id is not None:
            path = self._path(experiment_id, analysis_spec_hash, run_id)
            if path.exists():
                run = self._load_committed(path)
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
            for path in self.path.glob("*.parquet"):
                if not self._is_committed_path(path):
                    continue
                manifest = _read_manifest(path)
                if (
                    manifest.get("experiment_id") != experiment_id
                    or manifest.get("analysis_spec_hash") != analysis_spec_hash
                ):
                    continue
                matches.append(self._load_committed(path))
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
        for path in self.path.glob("*.parquet"):
            if not self._is_committed_path(path):
                continue
            if _read_manifest(path).get("experiment_id") != experiment_id:
                continue
            run = self._load_committed(path)
            found = [
                cell
                for cell in run.cells
                if cell.identity.metric_hash == selector.metric_hash
                and (
                    selector.estimator_id is None
                    or cell.identity.estimator_id == selector.estimator_id
                )
                and (
                    selector.scope_hash is None
                    or (
                        cell.identity.scope is not None
                        and cell.identity.scope.hash() == selector.scope_hash
                    )
                )
                and (selector.contrast is None or _matches_contrast(run, cell, selector.contrast))
                and (selector.role is None or cell.identity.key.method_role == selector.role)
            ]
            if len(found) > 1:
                refuse_store(
                    "store.history.ambiguous",
                    "History selector resolves ambiguously",
                    run_id=run.run_id,
                    matches=len(found),
                )
            if found:
                records.append((run, found[0]))
        return history_table(records)
