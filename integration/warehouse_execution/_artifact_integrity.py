"""Artifact integrity and bounded transfers through public analysis ingress."""

from __future__ import annotations

import datetime as dt
from contextlib import ExitStack
from uuid import uuid4

import ibis
import pyarrow as pa
import pytest

from increment import Analysis
from increment.artifacts import SourceSnapshotEvidence
from increment.query.artifact_contract import (
    ArtifactContractError,
    unit_day_artifact_extension_catalog,
)
from increment.query.session import WarehouseArtifactStore
from integration.warehouse_execution._suite import (
    _create_scratch_namespace,
    _drop_probe_generation,
    _drop_scratch_namespace,
    _qualified_table_identifier,
    _scratch_catalog,
)


def _bounded_transfers(con, monkeypatch) -> list[int]:
    transfers = []
    for method in ("to_pyarrow", "to_pyarrow_batches"):
        original = getattr(con, method)

        def guarded(table, *args, _original=original, **kwargs):
            names = set((table if hasattr(table, "schema") else table.as_table()).schema().names)
            assert "unit_id" not in names, names
            result = _original(table, *args, **kwargs)
            if isinstance(result, pa.Table):
                transfers.append(result.num_rows)
                assert result.num_rows <= 256
                return result
            if isinstance(result, pa.Scalar):
                transfers.append(1)
                return result

            def bounded_batches():
                with result:
                    for batch in result:
                        transfers.append(batch.num_rows)
                        assert batch.num_rows <= 256
                        yield batch

            return pa.RecordBatchReader.from_batches(result.schema, bounded_batches())

        monkeypatch.setattr(con, method, guarded)
    return transfers


def _integrity_events() -> list[dict]:
    rows = []
    for arm in ("control", "treatment"):
        for i in range(60):
            for event, day, value in (
                ("enrolled", 9, None),
                ("purchase", 5, float(i % 5 + 1)),
                ("purchase", 10, float(i % 5 + i % 3 + 2 + (arm == "treatment"))),
            ):
                rows.append(
                    {
                        "user_id": f"{arm}{i}",
                        "ts": dt.datetime(2025, 1, day, 8),
                        "event": event,
                        "experiment_id": "exp" if event == "enrolled" else None,
                        "group_id": arm if event == "enrolled" else None,
                        "value": value,
                        "country": "US" if i % 2 else "GB",
                    }
                )
    return rows


def _damage_extension_relation(monkeypatch, damage: str) -> None:
    from increment.query import artifact_extensions

    wire_table = artifact_extensions._wire_table

    def damaged_extension(rows, role):
        table, local_rows = wire_table(rows, role)
        if role == "cuped_preperiod":
            if damage == "duplicate":
                table = table.union(table.limit(1), distinct=False)
            elif damage == "domain":
                table = table.mutate(x=ibis.literal(float("nan")))
            elif damage == "exposure":
                table = table.mutate(unit_id=table.unit_id + "-absent")
        return table, local_rows

    monkeypatch.setattr(artifact_extensions, "_wire_table", damaged_extension)


def run_artifact_integrity_probe(con, *, dialect: str, monkeypatch, tmp_path, damage: str) -> None:
    name = f"events_integrity_{uuid4().hex[:10]}"
    schema = f"artifact_integrity_{uuid4().hex[:10]}"
    source = _qualified_table_identifier(con, dialect, name)
    definitions = tmp_path / f"{name}.yaml"
    definitions.write_text(f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source}
    timestamp_column: ts
    entities: [user_id]
    properties:
      - {{name: country, column: country, dtype: string, as_of: static}}
    facts:
      - {{name: enrolled, column: null}}
      - {{name: purchase, column: value}}
exposures:
  - {{name: enrollment, fact: enrolled}}
metrics:
  - {{name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 3}}
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-08T00:00:00
    control_group: control
    n_pre_periods: 7
    breakouts: [{{property: country}}]
    plan:
      secondaries:
        - metric: revenue
          decision_method: {{name: cuped, variance_reduction: cuped}}
""")
    with ExitStack() as cleanup:
        con.create_table(name, _integrity_events())
        cleanup.callback(con.drop_table, name, force=True)
        _create_scratch_namespace(con, dialect, schema)
        cleanup.callback(_drop_scratch_namespace, con, schema, _scratch_catalog(con, dialect))
        transfers = _bounded_transfers(con, monkeypatch)
        cutoff = dt.datetime(2025, 1, 31, tzinfo=dt.UTC)
        native = Analysis.from_definitions(
            "exp",
            definitions,
            con,
            store="none",
            source_snapshot_evidence=SourceSnapshotEvidence(cutoff, {"events": cutoff}),
        )
        cleanup.callback(native.close)
        context = native.artifact_context
        store = WarehouseArtifactStore(con, schema_name=schema)
        requests = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind in {"cuped_preperiod", "assignment_counts", "breakout_dimension"}
        ]
        assert {request.kind for request in requests} == {
            "cuped_preperiod",
            "assignment_counts",
            "breakout_dimension",
        }
        expected = native.run()[0]
        _damage_extension_relation(monkeypatch, damage)
        if damage in {"duplicate", "domain"}:
            with pytest.raises(ArtifactContractError) as caught:
                native.publish_unit_day_artifact(store, extensions=requests)
            assert caught.value.code == "artifact.extension.invalid"
            assert transfers
            return
        ref = native.publish_unit_day_artifact(store, extensions=requests)
        cleanup.callback(_drop_probe_generation, con, store, ref, dialect, schema_name=schema)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        cleanup.callback(adopted.close)
        if damage == "exposure":
            with pytest.raises(ArtifactContractError) as caught:
                adopted.run()
            assert caught.value.code == "artifact.extension.invalid"
            return
        actual = adopted.run()[0]
        assert actual.method == "cuped"
        assert actual.lift.value == pytest.approx(expected.lift.value, rel=1e-12)
        assert actual.lift.lb == pytest.approx(expected.lift.lb, rel=1e-12)
        assert actual.lift.ub == pytest.approx(expected.lift.ub, rel=1e-12)
        assert adopted.run_breakout(metrics=["revenue"], population="assigned")
        assert transfers
        if damage == "tamper":
            with store.open_snapshot(ref) as snapshot:
                manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            relation = next(
                ext for ext in manifest.extensions if ext.kind == "cuped_preperiod"
            ).relation.relation
            con.raw_sql(f'UPDATE "{relation.schema_name}"."{relation.name}" SET x = x + 1.0')
            adopted.close()
            reopened = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
            cleanup.callback(reopened.close)
            with pytest.raises(ArtifactContractError) as caught:
                reopened.run()
            assert caught.value.code == "artifact.snapshot.mixed"
