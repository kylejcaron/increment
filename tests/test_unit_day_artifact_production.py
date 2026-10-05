"""Producer-side unit-day artifact publication contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import ibis
import pytest

from examples._seed import seed_event_log
from increment import Analysis
from increment.errors import CapabilityError, CodedError
from increment.query.artifact_contract import (
    ARTIFACT_DOMAIN_ROOT,
    ArtifactContractError,
)
from increment.query.artifact_digest import (
    canonical_json,
    context_sha256,
    extension_definition_sha256,
    extension_source_provenance_sha256,
    manifest_sha256,
)
from increment.query.session import WarehouseArtifactStore
from increment.semantics.artifact import (
    ArtifactContext,
    BaseRelations,
    BreakoutDimensionExtension,
    MeasureManifest,
    RelationLocator,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
)


def _context() -> ArtifactContext:
    payload = (
        '{"context_format":2,"experiment_name":"demo",'
        '"window_days":{"end":null,"observation_horizon":null,"start":"2025-01-01"}}'
    )
    return ArtifactContext(
        canonical_json=payload,
        sha256=hashlib.sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + payload.encode()).hexdigest(),
    )


def _breakout_definition_hashes(property_name: str, source_name: str) -> tuple[str, str, str, str]:
    """Canonical definition/source JSON and their hashes for one breakout_dimension request."""
    definition_json = canonical_json({"property_name": property_name, "source_name": source_name})
    source_json = canonical_json(
        {
            "source_recipe_format": 2,
            "recipe_sha256": hashlib.sha256(source_name.encode()).hexdigest(),
        }
    )
    return (
        definition_json,
        source_json,
        extension_definition_sha256("breakout_dimension", definition_json),
        extension_source_provenance_sha256("breakout_dimension", source_json),
    )


def _breakout_catalog_context(dimensions: list[tuple[str, str]]) -> ArtifactContext:
    """Build a context whose extension_catalog matches one breakout_dimension
    request per (property_name, source_name) pair."""
    entries = []
    for property_name, source_name in dimensions:
        definition_json, source_json, definition_sha, source_sha = _breakout_definition_hashes(
            property_name, source_name
        )
        entries.append(
            {
                "request": {
                    "kind": "breakout_dimension",
                    "property_name": property_name,
                    "source_name": source_name,
                },
                "canonical_definition_json": definition_json,
                "canonical_source_recipe_json": source_json,
                "definition_sha256": definition_sha,
                "source_provenance_sha256": source_sha,
            }
        )
    ordered = sorted(entries, key=lambda item: canonical_json(item["request"]).encode())
    window_days = {"start": "2025-01-01", "end": None, "observation_horizon": None}
    raw = canonical_json(
        {"context_format": 2, "extension_catalog": ordered, "window_days": window_days}
    )
    return ArtifactContext(canonical_json=raw, sha256=context_sha256({"canonical_json": raw}))


def _publish(
    store: WarehouseArtifactStore,
    *,
    unit_id: str = "u1",
    experiment_id: str = "demo",
    after_manifest: Callable[[Any, Any, Any], None] | None = None,
    swallow_manifest_error: bool = False,
    before_manifest: Callable[[Any], None] | None = None,
) -> tuple[Any, Any, Any]:
    """Publish one real generation; returns (ref, exposure_ref, measure_ref).

    `after_manifest(ref, exposure_ref, measure_ref)` runs inside the open
    publication right after the manifest is committed."""
    context = _context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": experiment_id,
                        "unit_id": unit_id,
                        "group_id": "control",
                        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                        "first_exposure_date": date(2024, 1, 1),
                    }
                ]
            ),
        )
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": experiment_id,
                        "unit_id": unit_id,
                        "ds": date(2024, 1, 1),
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        if before_manifest is not None:
            before_manifest(publication)
        body: dict[str, Any] = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": experiment_id,
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "day_boundary": "UTC",
            "first_ds": date(2024, 1, 1),
            "last_ds": date(2024, 1, 1),
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
        }
        prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
        body["manifest_sha256"] = manifest_sha256(prototype)
        try:
            ref = publication.publish_manifest(UnitDayArtifactManifest(**body))
        except Exception:
            if not swallow_manifest_error:
                raise
            return None, exposure_ref, measure_ref
        if after_manifest is not None:
            after_manifest(ref, exposure_ref, measure_ref)
    return ref, exposure_ref, measure_ref


def test_publication_is_manifest_last_and_snapshot_is_immutable() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    context = _context()
    rows = [
        {
            "experiment_id": "demo",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
            "first_exposure_date": date(2024, 1, 1),
        }
    ]
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation("exposures", ibis.memtable(rows))
        assert store.visible_manifests == ()
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "ds": date(2024, 1, 1),
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        manifest_body: dict[str, Any] = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": "demo",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "day_boundary": "UTC",
            "first_ds": date(2024, 1, 1),
            "last_ds": date(2024, 1, 1),
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
        }
        prototype = UnitDayArtifactManifest.model_construct(**manifest_body, manifest_sha256="")
        manifest_body["manifest_sha256"] = manifest_sha256(prototype)
        ref = publication.publish_manifest(UnitDayArtifactManifest(**manifest_body))
    assert store.visible_manifests == (ref,)
    namespaced_tables = set(con.list_tables(database="artifact"))
    default_tables = set(con.list_tables())
    assert {"ud_manifest_index", exposure_ref.relation.name} <= namespaced_tables
    assert {"ud_manifest_index", exposure_ref.relation.name}.isdisjoint(default_tables)
    assert exposure_ref.relation.name not in default_tables
    with store.open_snapshot(ref) as snapshot:
        assert snapshot.artifact_id == ref.artifact_id
        result = (
            snapshot.verify_relation(exposure_ref, expected_role="exposures")
            .execute()
            .to_dict("records")
        )
        assert result[0]["experiment_id"] == "demo"
        assert result[0]["unit_id"] == "u1"
    store.drop_generation(ref.artifact_id, ref.generation_id)
    with pytest.raises(ArtifactContractError) as dropped:
        with store.open_snapshot(ref):
            pass
    assert dropped.value.code == "artifact.generation.dropped"
    with (
        pytest.raises(ArtifactContractError),
        store.open_snapshot(ref.model_copy(update={"generation_id": uuid4()})),
    ):
        pass


def test_drop_generation_erases_relations_and_adoptability() -> None:
    context = _context()
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    exposures = [
        {
            "experiment_id": "demo",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
            "first_exposure_date": date(2024, 1, 1),
        }
    ]
    stats = [
        {
            "experiment_id": "demo",
            "unit_id": "u1",
            "ds": date(2024, 1, 1),
            "measure_key": "measure_a",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        }
    ]
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation("exposures", ibis.memtable(exposures))
        measure_ref = publication.write_relation("measure_stats", ibis.memtable(stats))
        body: dict[str, Any] = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": "demo",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "day_boundary": "UTC",
            "first_ds": date(2024, 1, 1),
            "last_ds": date(2024, 1, 1),
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
        }
        prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
        body["manifest_sha256"] = manifest_sha256(prototype)
        ref = publication.publish_manifest(UnitDayArtifactManifest(**body))

    cached = WarehouseArtifactStore(con, schema_name="artifact")
    assert ref in cached.visible_manifests

    store.drop_generation(ref.artifact_id, ref.generation_id)

    reopened = WarehouseArtifactStore(con, schema_name="artifact")
    assert reopened.visible_manifests == ()
    remaining = set(con.list_tables(database="artifact"))
    assert exposure_ref.relation.name not in remaining
    assert measure_ref.relation.name not in remaining
    for candidate in (store, cached, reopened):
        assert candidate.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with candidate.open_snapshot(ref):
                pass
        assert refusal.value.code == "artifact.generation.dropped"


def test_drop_generation_second_erase_failure_is_retryable(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, exposure_ref, measure_ref = _publish(store)
    cached = WarehouseArtifactStore(con, schema_name="artifact")

    real_drop_table = con.drop_table
    calls = {"n": 0}

    def flaky_drop_table(name, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated erasure failure")
        return real_drop_table(name, **kwargs)

    monkeypatch.setattr(con, "drop_table", flaky_drop_table)
    with pytest.raises(RuntimeError, match="simulated erasure failure"):
        store.drop_generation(ref.artifact_id, ref.generation_id)
    monkeypatch.setattr(con, "drop_table", real_drop_table)

    target_names = {exposure_ref.relation.name, measure_ref.relation.name}
    remaining = set(con.list_tables(database="artifact"))
    assert len(target_names & remaining) == 1

    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    for candidate in (store, cached, fresh):
        assert candidate.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with candidate.open_snapshot(ref):
                pass
        assert refusal.value.code == "artifact.generation.dropped"

    fresh.drop_generation(ref.artifact_id, ref.generation_id)
    remaining_after_retry = set(con.list_tables(database="artifact"))
    assert target_names.isdisjoint(remaining_after_retry)


def _abort_after_manifest(
    store: WarehouseArtifactStore, error: BaseException
) -> tuple[Any, Any, Any]:
    """Publish, then raise `error` from the body; returns the refs seen at abort."""
    seen: list[tuple[Any, Any, Any]] = []

    def abort(ref: Any, exposure_ref: Any, measure_ref: Any) -> None:
        seen.append((ref, exposure_ref, measure_ref))
        raise error

    with pytest.raises(type(error)) as raised:
        _publish(store, after_manifest=abort)
    assert raised.value is error
    return seen[0]


def _assert_invalidated(con: Any, ref: Any, exposure_ref: Any, measure_ref: Any) -> None:
    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    assert fresh.visible_manifests == ()
    with pytest.raises(ArtifactContractError) as refusal:
        with fresh.open_snapshot(ref):
            pass
    assert refusal.value.code == "artifact.generation.dropped"
    remaining = set(con.list_tables(database="artifact"))
    assert {exposure_ref.relation.name, measure_ref.relation.name}.isdisjoint(remaining)
    tombstones = con.table("ud_manifest_dropped", database="artifact").execute()
    assert str(ref.generation_id) in set(tombstones["generation_id"])


def test_abort_after_manifest_publication_invalidates_the_generation() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    refs = _abort_after_manifest(store, RuntimeError("body failed"))
    assert store.visible_manifests == ()
    _assert_invalidated(con, *refs)


@pytest.mark.parametrize("error", [RuntimeError("body failed"), KeyboardInterrupt()])
@pytest.mark.parametrize("cleanup_error", [RuntimeError("tombstone"), KeyboardInterrupt()])
def test_abort_cleanup_failure_keeps_original_error_and_is_recoverable(
    monkeypatch, error: BaseException, cleanup_error: BaseException
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert

    def flaky_insert(name, *args, **kwargs):
        if name == "ud_manifest_dropped":
            raise cleanup_error
        return real_insert(name, *args, **kwargs)

    monkeypatch.setattr(con, "insert", flaky_insert)
    ref, exposure_ref, measure_ref = _abort_after_manifest(store, error)
    monkeypatch.setattr(con, "insert", real_insert)

    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    assert fresh.visible_manifests == (ref,)

    fresh.drop_generation(ref.artifact_id, ref.generation_id)
    _assert_invalidated(con, ref, exposure_ref, measure_ref)


def _insert_commits_then_raises(monkeypatch, con: Any, error: BaseException) -> Any:
    real_insert = con.insert

    def commit_then_raise(name, *args, **kwargs):
        result = real_insert(name, *args, **kwargs)
        if name == "ud_manifest_index":
            raise error
        return result

    monkeypatch.setattr(con, "insert", commit_then_raise)
    return real_insert


def _publication_relation_names(con: Any) -> set[str]:
    return {
        name
        for name in con.list_tables(database="artifact")
        if name.startswith("ud_") and not name.startswith("ud_manifest")
    }


@pytest.mark.parametrize("error", [RuntimeError("ack lost"), KeyboardInterrupt()])
def test_manifest_insert_commits_then_raises_invalidates_generation(
    monkeypatch, error: BaseException
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = _insert_commits_then_raises(monkeypatch, con, error)
    with pytest.raises(type(error)) as raised:
        _publish(store)
    assert raised.value is error
    monkeypatch.setattr(con, "insert", real_insert)

    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    assert fresh.visible_manifests == ()
    assert store.visible_manifests == ()
    assert _publication_relation_names(con) == set()
    assert len(con.table("ud_manifest_dropped", database="artifact").execute()) == 1


@pytest.mark.parametrize("error", [RuntimeError("submitted"), KeyboardInterrupt()])
def test_manifest_insert_committing_after_abort_stays_invisible(
    monkeypatch, error: BaseException
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert
    delayed: list[tuple[Any, Any, Any]] = []

    def submit_but_commit_later(name, obj, *args, **kwargs):
        if name == "ud_manifest_index":
            delayed.append((name, obj, kwargs))
            raise error
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", submit_but_commit_later)
    with pytest.raises(type(error)) as raised:
        _publish(store)
    assert raised.value is error
    assert _publication_relation_names(con) == set()

    (name, obj, kwargs) = delayed[0]
    real_insert(name, obj, **kwargs)  # the late commit lands after the abort

    row = obj.execute().iloc[0]
    ref = UnitDayArtifactRef(
        artifact_id=UUID(row["artifact_id"]),
        generation_id=UUID(row["generation_id"]),
        manifest=RelationLocator(schema="artifact", name=row["manifest_name"]),
        manifest_sha256=row["manifest_sha256"],
    )
    for candidate in (store, WarehouseArtifactStore(con, schema_name="artifact")):
        assert candidate.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with candidate.open_snapshot(ref):
                pass
        assert refusal.value.code == "artifact.generation.dropped"


def _tombstone_write_fails(monkeypatch, con: Any, *, index_commits: bool) -> BaseException:
    real_insert = con.insert
    error = RuntimeError("body failed")

    def fail(name, *args, **kwargs):
        if name == "ud_manifest_dropped":
            raise ConnectionError("tombstone unavailable")
        if name == "ud_manifest_index":
            if index_commits:
                real_insert(name, *args, **kwargs)
            raise error
        return real_insert(name, *args, **kwargs)

    monkeypatch.setattr(con, "insert", fail)
    return error


def test_tombstone_failure_after_committed_insert_preserves_generation(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    error = _tombstone_write_fails(monkeypatch, con, index_commits=True)
    with pytest.raises(RuntimeError) as raised:
        _publish(store)
    assert raised.value is error
    monkeypatch.undo()

    assert len(_publication_relation_names(con)) == 2
    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    (ref,) = fresh.visible_manifests
    fresh.drop_generation(ref.artifact_id, ref.generation_id)
    assert WarehouseArtifactStore(con, schema_name="artifact").visible_manifests == ()
    assert _publication_relation_names(con) == set()


def test_tombstone_failure_without_row_preserves_and_names_relations(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    error = _tombstone_write_fails(monkeypatch, con, index_commits=False)
    with pytest.raises(RuntimeError) as raised:
        _publish(store)
    assert raised.value is error
    monkeypatch.undo()

    preserved = _publication_relation_names(con)
    assert len(preserved) == 2
    notes = "\n".join(raised.value.__notes__)
    artifact_match = re.search(r"artifact_id=([0-9a-f-]{36})", notes)
    generation_match = re.search(r"generation_id=([0-9a-f-]{36})", notes)
    assert artifact_match is not None and generation_match is not None
    artifact_id = UUID(artifact_match.group(1))
    generation_id = UUID(generation_match.group(1))
    named = {
        token
        for listing in re.findall(r"relations=\[([^\]]*)\]", notes)
        for token in listing.split(", ")
    }
    assert named == {f"artifact.{name}" for name in preserved}

    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    assert fresh.visible_manifests == ()
    fresh.drop_generation(artifact_id, generation_id)
    assert _publication_relation_names(con) == preserved


def test_pre_manifest_abort_leaves_no_manifest_and_no_relations() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    with pytest.raises(RuntimeError, match="early"):
        with store.begin_publication(expected_context=_context()) as publication:
            exposure_ref = publication.write_relation(
                "exposures",
                ibis.memtable(
                    [
                        {
                            "experiment_id": "demo",
                            "unit_id": "u1",
                            "group_id": "control",
                            "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                            "first_exposure_date": date(2024, 1, 1),
                        }
                    ]
                ),
            )
            raise RuntimeError("early")
    assert store.visible_manifests == ()
    assert exposure_ref.relation.name not in set(con.list_tables(database="artifact"))


def test_drop_generation_invalidation_failure_preserves_generation(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, exposure_ref, measure_ref = _publish(store)

    real_insert = con.insert

    def flaky_insert(name, *args, **kwargs):
        if name == "ud_manifest_dropped":
            raise RuntimeError("simulated tombstone failure")
        return real_insert(name, *args, **kwargs)

    monkeypatch.setattr(con, "insert", flaky_insert)
    with pytest.raises(RuntimeError, match="simulated tombstone failure"):
        store.drop_generation(ref.artifact_id, ref.generation_id)

    remaining = set(con.list_tables(database="artifact"))
    assert {exposure_ref.relation.name, measure_ref.relation.name} <= remaining

    for candidate in (store, WarehouseArtifactStore(con, schema_name="artifact")):
        assert candidate.visible_manifests == (ref,)
        with candidate.open_snapshot(ref) as snapshot:
            rows = (
                snapshot.verify_relation(measure_ref, expected_role="measure_stats")
                .execute()
                .to_dict("records")
            )
        assert rows[0]["sum_value"] == 2.0


def test_drop_generation_preserves_unrelated_concurrent_publication() -> None:
    con = ibis.duckdb.connect()
    store_a = WarehouseArtifactStore(con, schema_name="artifact")
    ref_a, _exposure_a, _measure_a = _publish(store_a, unit_id="ua")
    store_b = WarehouseArtifactStore(con, schema_name="artifact")

    store_a.drop_generation(ref_a.artifact_id, ref_a.generation_id)
    ref_b, _exposure_b, measure_b = _publish(store_b, unit_id="ub")
    store_a.drop_generation(ref_a.artifact_id, ref_a.generation_id)
    store_a.drop_generation(ref_a.artifact_id, ref_a.generation_id)

    assert store_b.visible_manifests == (ref_b,)
    with store_b.open_snapshot(ref_b) as snapshot:
        rows = (
            snapshot.verify_relation(measure_b, expected_role="measure_stats")
            .execute()
            .to_dict("records")
        )
    assert rows[0]["unit_id"] == "ub"
    assert rows[0]["sum_value"] == 2.0


def test_open_snapshot_refuses_live_relation_read_after_drop(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, exposure_ref, _measure_ref = _publish(store)
    reader = WarehouseArtifactStore(con, schema_name="artifact")

    with reader.open_snapshot(ref) as snapshot:
        store.drop_generation(ref.artifact_id, ref.generation_id)
        with pytest.raises(ArtifactContractError) as refusal:
            snapshot.verify_relation(exposure_ref, expected_role="exposures")
        assert refusal.value.code == "artifact.generation.dropped"

    # Inject the drop between the pre-check and the live table lookup so the
    # exception re-check, not the pre-check, must classify the race.
    ref2, exposure_ref2, _measure_ref2 = _publish(store, unit_id="u2")
    reader2 = WarehouseArtifactStore(con, schema_name="artifact")
    real_table = con.table

    def racing_table(name, **kwargs):
        if name == exposure_ref2.relation.name:
            store.drop_generation(ref2.artifact_id, ref2.generation_id)
        return real_table(name, **kwargs)

    with reader2.open_snapshot(ref2) as snapshot2:
        monkeypatch.setattr(con, "table", racing_table)
        with pytest.raises(ArtifactContractError) as refusal2:
            snapshot2.verify_relation(exposure_ref2, expected_role="exposures")
        assert refusal2.value.code == "artifact.generation.dropped"


def test_drop_generation_refuses_foreign_manifest_locators() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref_a, exposure_a, measure_a = _publish(store, unit_id="ua")
    ref_b, exposure_b, measure_b = _publish(store, unit_id="ub")

    with store.open_snapshot(ref_a) as snapshot:
        manifest_a = snapshot.read_manifest(ref_a.manifest, expected_sha256=ref_a.manifest_sha256)

    # Corrupt A's retained exposures locator to point at B's real relation,
    # while keeping A's own identity fields self-consistent.
    corrupted_exposures = manifest_a.base.exposures.model_copy(
        update={"relation": exposure_b.relation}
    )
    corrupted_manifest = manifest_a.model_copy(
        update={"base": manifest_a.base.model_copy(update={"exposures": corrupted_exposures})}
    )
    corrupted_manifest = corrupted_manifest.model_copy(
        update={"manifest_sha256": manifest_sha256(corrupted_manifest)}
    )
    corrupted_json = json.dumps(corrupted_manifest.model_dump(mode="json"), separators=(",", ":"))
    con.con.execute(
        "update artifact.ud_manifest_index set manifest_json = ?, manifest_sha256 = ? "
        "where artifact_id = ? and generation_id = ?",
        [
            corrupted_json,
            corrupted_manifest.manifest_sha256,
            str(ref_a.artifact_id),
            str(ref_a.generation_id),
        ],
    )

    with pytest.raises(ArtifactContractError) as refusal:
        store.drop_generation(ref_a.artifact_id, ref_a.generation_id)
    assert refusal.value.code in {
        "artifact.refresh.invalid_ref",
        "artifact.identifier.unsafe",
        "artifact.manifest.invalid",
    }

    remaining = set(con.list_tables(database="artifact"))
    assert {
        exposure_a.relation.name,
        measure_a.relation.name,
        exposure_b.relation.name,
    } <= remaining
    visible = store.visible_manifests
    assert ref_b in visible
    assert len(visible) == 2
    assert any(
        candidate.artifact_id == ref_a.artifact_id
        and candidate.generation_id == ref_a.generation_id
        for candidate in visible
    )
    with store.open_snapshot(ref_b) as snapshot:
        rows = (
            snapshot.verify_relation(measure_b, expected_role="measure_stats")
            .execute()
            .to_dict("records")
        )
    assert rows[0]["unit_id"] == "ub"


def test_drop_generation_dropped_refusal_carries_generation_context() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, _exposure_ref, _measure_ref = _publish(store)
    store.drop_generation(ref.artifact_id, ref.generation_id)
    with pytest.raises(ArtifactContractError) as refusal:
        with store.open_snapshot(ref):
            pass
    assert refusal.value.code == "artifact.generation.dropped"
    assert refusal.value.context["generation_id"] == str(ref.generation_id)


def test_drop_generation_duplicate_index_refusal_carries_generation_context() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, _exposure_ref, _measure_ref = _publish(store)
    row = con.con.execute(
        "select artifact_id, generation_id, manifest_name, manifest_sha256, manifest_json "
        "from artifact.ud_manifest_index where artifact_id = ? and generation_id = ?",
        [str(ref.artifact_id), str(ref.generation_id)],
    ).fetchone()
    con.con.execute("insert into artifact.ud_manifest_index values (?, ?, ?, ?, ?)", list(row))

    with pytest.raises(ArtifactContractError) as refusal:
        store.drop_generation(ref.artifact_id, ref.generation_id)
    assert refusal.value.code == "artifact.manifest.invalid"
    assert refusal.value.context == {
        "artifact_id": str(ref.artifact_id),
        "generation_id": str(ref.generation_id),
    }


def test_publish_manifest_refuses_relation_written_but_not_referenced() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    context = _context()
    with pytest.raises(ArtifactContractError) as refusal:
        with store.begin_publication(expected_context=context) as publication:
            exposure_ref = publication.write_relation(
                "exposures",
                ibis.memtable(
                    [
                        {
                            "experiment_id": "demo",
                            "unit_id": "u1",
                            "group_id": "control",
                            "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                            "first_exposure_date": date(2024, 1, 1),
                        }
                    ]
                ),
            )
            measure_ref = publication.write_relation(
                "measure_stats",
                ibis.memtable(
                    [
                        {
                            "experiment_id": "demo",
                            "unit_id": "u1",
                            "ds": date(2024, 1, 1),
                            "measure_key": "measure_a",
                            "n_events": 1,
                            "sum_value": 2.0,
                            "min_value": 2.0,
                            "max_value": 2.0,
                        }
                    ]
                ),
            )
            # Written but never bound into the manifest below — an orphan.
            orphan_ref = publication.write_relation(
                "breakout_dimension",
                ibis.memtable(
                    [
                        {
                            "experiment_id": "demo",
                            "unit_id": "u1",
                            "value_is_missing": False,
                            "dimension_value": "unused",
                        }
                    ]
                ),
            )
            body: dict[str, Any] = {
                "artifact_format": 1,
                "artifact_id": publication.artifact_id,
                "generation_id": publication.generation_id,
                "experiment_id": "demo",
                "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                "day_boundary": "UTC",
                "first_ds": date(2024, 1, 1),
                "last_ds": date(2024, 1, 1),
                "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
                "measures": (
                    MeasureManifest(
                        measure_key="measure_a",
                        source_provenance_sha256="0" * 64,
                        freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                    ),
                ),
                "metric_measures": (),
                "context": context,
            }
            prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
            body["manifest_sha256"] = manifest_sha256(prototype)
            publication.publish_manifest(UnitDayArtifactManifest(**body))
    assert refusal.value.code == "artifact.manifest.unreferenced_relation"
    assert orphan_ref.relation.name in cast(Any, refusal.value.context["relation_names"])
    assert store.visible_manifests == ()
    assert orphan_ref.relation.name not in set(con.list_tables(database="artifact"))


def test_store_construction_refuses_when_tombstone_table_is_missing_and_uncreatable(
    monkeypatch,
) -> None:
    con = ibis.duckdb.connect()
    WarehouseArtifactStore(con, schema_name="artifact")
    con.con.execute("drop table artifact.ud_manifest_dropped")

    real_create_table = con.create_table

    def refuse_dropped_table(name, *args, **kwargs):
        if name == "ud_manifest_dropped":
            raise RuntimeError("simulated read-only backend")
        return real_create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", refuse_dropped_table)
    with pytest.raises(CapabilityError) as refusal:
        WarehouseArtifactStore(con, schema_name="artifact")
    assert refusal.value.code == "artifact.store.namespace"


def test_store_construction_refuses_when_tombstone_table_has_wrong_column_type() -> None:
    con = ibis.duckdb.connect()
    con.create_database("artifact", force=True)
    con.create_table(
        "ud_manifest_dropped",
        schema=ibis.schema(
            {"artifact_id": "string", "generation_id": "int64", "dropped_at": "string"}
        ),
        database="artifact",
        overwrite=False,
    )
    with pytest.raises(CapabilityError) as refusal:
        WarehouseArtifactStore(con, schema_name="artifact")
    assert refusal.value.code == "artifact.store.namespace"
    assert refusal.value.context["wrong_type_columns"] == ("generation_id",)


def test_store_construction_refuses_when_tombstone_table_is_missing_a_column() -> None:
    con = ibis.duckdb.connect()
    con.create_database("artifact", force=True)
    con.create_table(
        "ud_manifest_dropped",
        schema=ibis.schema({"artifact_id": "string", "generation_id": "string"}),
        database="artifact",
        overwrite=False,
    )
    with pytest.raises(CapabilityError) as refusal:
        WarehouseArtifactStore(con, schema_name="artifact")
    assert refusal.value.code == "artifact.store.namespace"
    assert refusal.value.context["missing_columns"] == ("dropped_at",)


def _index_schema_refusal(**columns: str) -> CapabilityError:
    con = ibis.duckdb.connect()
    con.create_database("artifact", force=True)
    con.create_table(
        "ud_manifest_index",
        schema=ibis.schema(columns),
        database="artifact",
        overwrite=False,
    )
    with pytest.raises(CapabilityError) as refusal:
        WarehouseArtifactStore(con, schema_name="artifact")
    assert refusal.value.code == "artifact.store.namespace"
    assert refusal.value.context["table"] == "ud_manifest_index"
    return refusal.value


_INDEX_STRINGS = {
    "artifact_id": "string",
    "generation_id": "string",
    "manifest_name": "string",
    "manifest_sha256": "string",
    "manifest_json": "string",
}


def test_store_construction_refuses_when_index_table_is_missing_a_column() -> None:
    columns = {k: v for k, v in _INDEX_STRINGS.items() if k != "manifest_json"}
    refusal = _index_schema_refusal(**columns)
    assert refusal.context["missing_columns"] == ("manifest_json",)


def test_store_construction_refuses_when_index_table_has_an_extra_column() -> None:
    refusal = _index_schema_refusal(**_INDEX_STRINGS, note="string")
    assert refusal.context["extra_columns"] == ("note",)


def test_store_construction_refuses_when_index_table_has_wrong_column_type() -> None:
    refusal = _index_schema_refusal(**{**_INDEX_STRINGS, "manifest_sha256": "int64"})
    assert refusal.context["wrong_type_columns"] == ("manifest_sha256",)


def test_store_construction_refuses_when_index_table_is_missing_and_uncreatable(
    monkeypatch,
) -> None:
    con = ibis.duckdb.connect()
    real_create_table = con.create_table

    def refuse_index_table(name, *args, **kwargs):
        if name == "ud_manifest_index":
            raise RuntimeError("simulated read-only backend")
        return real_create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", refuse_index_table)
    with pytest.raises(CapabilityError) as refusal:
        WarehouseArtifactStore(con, schema_name="artifact")
    assert refusal.value.code == "artifact.store.namespace"
    assert refusal.value.context["table"] == "ud_manifest_index"


def test_store_construction_adopts_index_table_created_by_a_concurrent_store(
    monkeypatch,
) -> None:
    con = ibis.duckdb.connect()
    real_create_table = con.create_table

    def lose_create_race(name, *args, **kwargs):
        if name == "ud_manifest_index":
            # The winner's table lands first, then this create reports a conflict.
            real_create_table(name, *args, **kwargs)
            raise RuntimeError("relation already exists")
        return real_create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", lose_create_race)
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, _exposure_ref, _measure_ref = _publish(store)
    assert store.visible_manifests == (ref,)


def test_reverse_ordered_index_table_publishes_and_adopts_like_native() -> None:
    from tests.test_unit_day_artifact_facade import _native

    con, native, context, _default_store = _native()
    con.drop_table("ud_manifest_index", database="artifacts")
    con.create_table(
        "ud_manifest_index",
        schema=ibis.schema(dict(reversed(list(_INDEX_STRINGS.items())))),
        database="artifacts",
        overwrite=False,
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(store)

    row = con.table("ud_manifest_index", database="artifacts").to_pyarrow().to_pylist()
    assert len(row) == 1
    assert row[0]["artifact_id"] == str(ref.artifact_id)
    assert row[0]["generation_id"] == str(ref.generation_id)
    assert row[0]["manifest_name"] == ref.manifest.name
    assert row[0]["manifest_sha256"] == ref.manifest_sha256
    assert json.loads(row[0]["manifest_json"])["manifest_sha256"] == ref.manifest_sha256

    reopened = WarehouseArtifactStore(con, schema_name="artifacts")
    assert reopened.visible_manifests == (ref,)
    adopted = Analysis.from_unit_day_artifact(reopened, ref, expected_context=context)
    try:
        metric = native.metrics[0].name
        native_rows = native.run(metrics=[metric])
        adopted_rows = adopted.run(metrics=[metric])
        assert len(adopted_rows) == len(native_rows) > 0
        for got, want in zip(adopted_rows, native_rows, strict=True):
            for field in ("abs_diff", "abs_se", "abs_lb", "abs_ub", "abs_alpha"):
                assert getattr(got, field) == pytest.approx(getattr(want, field), rel=1e-9)
    finally:
        adopted.close()
        native.close()


def test_same_role_relations_keep_distinct_locators_and_snapshot_reads_second() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    context = _breakout_catalog_context([("first_dim", "facts"), ("second_dim", "facts")])
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "group_id": "control",
                        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                        "first_exposure_date": date(2024, 1, 1),
                    }
                ]
            ),
        )
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "ds": date(2024, 1, 1),
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        dimension_rows = ibis.memtable(
            [
                {
                    "experiment_id": "demo",
                    "unit_id": "u1",
                    "value_is_missing": False,
                    "dimension_value": "first",
                }
            ]
        )
        first_dimension_ref = publication.write_relation("breakout_dimension", dimension_rows)
        second_dimension_ref = publication.write_relation(
            "breakout_dimension",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "value_is_missing": False,
                        "dimension_value": "second",
                    }
                ]
            ),
        )
        assert second_dimension_ref.relation.name.endswith("breakout_dimension_2")
        _, _, first_def_sha, first_src_sha = _breakout_definition_hashes("first_dim", "facts")
        _, _, second_def_sha, second_src_sha = _breakout_definition_hashes("second_dim", "facts")
        first_extension = BreakoutDimensionExtension(
            relation=first_dimension_ref,
            definition_sha256=first_def_sha,
            source_provenance_sha256=first_src_sha,
            dimension_name="first_dim",
            source_name="facts",
        )
        second_extension = BreakoutDimensionExtension(
            relation=second_dimension_ref,
            definition_sha256=second_def_sha,
            source_provenance_sha256=second_src_sha,
            dimension_name="second_dim",
            source_name="facts",
        )
        manifest_body: dict[str, Any] = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": "demo",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "day_boundary": "UTC",
            "first_ds": date(2024, 1, 1),
            "last_ds": date(2024, 1, 1),
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
            "extensions": (first_extension, second_extension),
        }
        prototype = UnitDayArtifactManifest.model_construct(**manifest_body, manifest_sha256="")
        manifest_body["manifest_sha256"] = manifest_sha256(prototype)
        ref = publication.publish_manifest(UnitDayArtifactManifest(**manifest_body))
    with store.open_snapshot(ref) as snapshot:
        assert snapshot.verify_relation(
            second_dimension_ref, expected_role="breakout_dimension"
        ).execute()["dimension_value"].tolist() == ["second"]
    assert first_dimension_ref.relation.name != second_dimension_ref.relation.name


def test_publication_handle_isolation_and_abort_has_no_manifest() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    context = _context()
    with store.begin_publication(expected_context=context) as first:
        with store.begin_publication(expected_context=context) as second:
            assert first.artifact_id != second.artifact_id
            assert first.generation_id != second.generation_id
            first.write_relation(
                "exposures",
                ibis.memtable(
                    ibis.schema(
                        {
                            "experiment_id": "string",
                            "unit_id": "string",
                            "group_id": "string",
                            "first_exposure_ts": "timestamp",
                            "first_exposure_date": "date",
                        }
                    )
                    .to_pyarrow()
                    .empty_table()
                ),
            )
            second.write_relation(
                "exposures",
                ibis.memtable(
                    ibis.schema(
                        {
                            "experiment_id": "string",
                            "unit_id": "string",
                            "group_id": "string",
                            "first_exposure_ts": "timestamp",
                            "first_exposure_date": "date",
                        }
                    )
                    .to_pyarrow()
                    .empty_table()
                ),
            )
        # No manifest was published by either handle.
    assert store.visible_manifests == ()


def test_store_namespace_contains_metadata_and_relations() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    relation_name: str
    with store.begin_publication(expected_context=_context()) as publication:
        relation = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "group_id": "control",
                        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                        "first_exposure_date": date(2024, 1, 1),
                    }
                ]
            ),
        )
        relation_name = relation.relation.name
        namespaced_tables = set(con.list_tables(database="artifact"))
        default_tables = set(con.list_tables())
        assert relation_name in namespaced_tables
        assert relation_name not in default_tables
        assert con.table(relation_name, database="artifact").execute()["unit_id"].tolist() == ["u1"]

    namespaced_tables = set(con.list_tables(database="artifact"))
    default_tables = set(con.list_tables())
    assert "ud_manifest_index" in namespaced_tables
    assert "ud_manifest_index" not in default_tables


def test_invalid_namespace_refuses_before_any_warehouse_ddl() -> None:
    con = ibis.duckdb.connect()

    with pytest.raises(CapabilityError) as raised:
        WarehouseArtifactStore(con, schema_name="analytics.prod")

    assert raised.value.code == "artifact.store.namespace"
    assert (
        con.raw_sql(
            "select schema_name from information_schema.schemata "
            "where schema_name = 'analytics.prod'"
        ).fetchall()
        == []
    )
    assert (
        con.raw_sql(
            "select table_schema, table_name from information_schema.tables "
            "where table_schema = 'analytics.prod'"
        ).fetchall()
        == []
    )


# Guards publication from a definitions analysis configured with store="none".
def test_publish_unit_day_artifact_works_from_a_storeless_definitions_analysis() -> None:
    con = ibis.duckdb.connect()
    seed_event_log(con)
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    store = WarehouseArtifactStore(con, schema_name="artifact_publisher_marker")
    assert analysis.publish_unit_day_artifact(store)


def test_handled_manifest_insert_failure_still_tombstones_before_erasing(monkeypatch) -> None:
    """The caller catches an ambiguous manifest-insert failure inside the publication
    and leaves the block normally: a late-committing row must still never be visible."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert
    delayed: list[tuple[Any, Any, Any]] = []

    def submit_but_commit_later(name, obj, *args, **kwargs):
        if name == "ud_manifest_index":
            delayed.append((name, obj, kwargs))
            raise RuntimeError("submitted")
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", submit_but_commit_later)
    ref, _, _ = _publish(store, swallow_manifest_error=True)
    assert ref is None
    assert _publication_relation_names(con) == set()

    (name, obj, kwargs) = delayed[0]
    real_insert(name, obj, **kwargs)  # the late commit lands after the block exited

    row = obj.execute().iloc[0]
    late = UnitDayArtifactRef(
        artifact_id=UUID(row["artifact_id"]),
        generation_id=UUID(row["generation_id"]),
        manifest=RelationLocator(schema="artifact", name=row["manifest_name"]),
        manifest_sha256=row["manifest_sha256"],
    )
    for candidate in (store, WarehouseArtifactStore(con, schema_name="artifact")):
        assert candidate.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with candidate.open_snapshot(late):
                pass
        assert refusal.value.code == "artifact.generation.dropped"


def test_handled_manifest_insert_failure_with_failed_tombstone_is_surfaced(monkeypatch) -> None:
    """If the tombstone cannot be written the relations are kept and the caller is told."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert

    injected: list[RuntimeError] = []

    def failing_tombstone(name, obj, *args, **kwargs):
        if name == "ud_manifest_dropped":
            injected.append(RuntimeError("tombstone unavailable"))
            raise injected[-1]
        if name == "ud_manifest_index":
            raise RuntimeError("index unavailable")
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", failing_tombstone)
    with pytest.raises(CodedError) as raised:
        _publish(store, swallow_manifest_error=True)
    assert raised.value.code == "query.session.warehouse_artifact.publication_state_unknown"
    kept = _publication_relation_names(con)
    assert len(kept) == 2
    relations = raised.value.context["relations"]
    assert isinstance(relations, tuple)
    assert set(relations) == {f"artifact.{name}" for name in kept}
    assert raised.value.context["tombstoned"] is False
    assert raised.value.__cause__ is injected[0]
    monkeypatch.setattr(con, "insert", real_insert)
    assert WarehouseArtifactStore(con, schema_name="artifact").visible_manifests == ()


def _fail_one_relation_drop(
    monkeypatch, con: Any, error: BaseException | None = None, *, persistent: bool = False
) -> tuple[list[str], list[str], Any]:
    """Make the first dropped relation fail (every retry too when `persistent`);
    returns (attempted, failed, error)."""
    real_drop = con.drop_table
    attempted: list[str] = []
    failed: list[str] = []
    error = error or ConnectionError("drop unavailable")

    def flaky_drop(name, *args, **kwargs):
        if not name.startswith("ud_"):
            return real_drop(name, *args, **kwargs)  # digest scratch tables
        attempted.append(name)
        if not failed or (persistent and name == failed[0]):
            if not failed:
                failed.append(name)
            raise error
        return real_drop(name, *args, **kwargs)

    monkeypatch.setattr(con, "drop_table", flaky_drop)
    return attempted, failed, error


def test_handled_insert_failure_with_failed_erasure_reports_retained_relations(
    monkeypatch,
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert

    def index_unavailable(name, obj, *args, **kwargs):
        if name == "ud_manifest_index":
            raise RuntimeError("index unavailable")
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", index_unavailable)
    attempted, failed, error = _fail_one_relation_drop(monkeypatch, con)
    with pytest.raises(CodedError) as raised:
        _publish(store, swallow_manifest_error=True)
    assert raised.value.code == "query.session.warehouse_artifact.publication_state_unknown"
    relations = raised.value.context["relations"]
    assert isinstance(relations, tuple)
    assert relations == (f"artifact.{failed[0]}",)
    assert len(attempted) == 2  # erasure continued past the failed drop
    assert raised.value.context["tombstoned"] is True
    assert raised.value.__cause__ is error
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}
    assert WarehouseArtifactStore(con, schema_name="artifact").visible_manifests == ()


def test_aborting_publication_with_failed_erasure_keeps_original_and_drops_the_rest(
    monkeypatch,
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_insert = con.insert
    original = RuntimeError("index unavailable")

    def index_unavailable(name, obj, *args, **kwargs):
        if name == "ud_manifest_index":
            raise original
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", index_unavailable)
    attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con)
    with pytest.raises(RuntimeError) as raised:
        _publish(store)
    assert raised.value is original
    assert len(attempted) == 2
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}
    assert WarehouseArtifactStore(con, schema_name="artifact").visible_manifests == ()


_BASE_ROWS = {
    "exposures": {
        "experiment_id": "demo",
        "unit_id": "u1",
        "group_id": "control",
        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
        "first_exposure_date": date(2024, 1, 1),
    },
    "measure_stats": {
        "experiment_id": "demo",
        "unit_id": "u1",
        "ds": date(2024, 1, 1),
        "measure_key": "measure_a",
        "n_events": 1,
        "sum_value": 2.0,
        "min_value": 2.0,
        "max_value": 2.0,
    },
}


def _write_base_relations(publication: Any) -> None:
    for role, row in _BASE_ROWS.items():
        publication.write_relation(role, ibis.memtable([row]))


@pytest.mark.parametrize("drop_error", [ConnectionError("drop unavailable"), KeyboardInterrupt()])
def test_early_abort_with_failed_drop_keeps_original_and_erases_the_rest(
    monkeypatch, drop_error: BaseException
) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con, drop_error)
    original = RuntimeError("body failed before the manifest insert")
    with pytest.raises(RuntimeError) as raised:
        with store.begin_publication(expected_context=_context()) as publication:
            _write_base_relations(publication)
            raise original
    assert raised.value is original
    assert len(attempted) == 2  # a failed drop, even an interrupt, does not stop the rest
    retained = f"artifact.{failed[0]}"
    assert any(retained in note for note in original.__notes__)
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}
    assert WarehouseArtifactStore(con, schema_name="artifact").visible_manifests == ()


def test_handled_early_error_with_failed_drop_raises_cleanup_incomplete(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    attempted, failed, error = _fail_one_relation_drop(monkeypatch, con)
    with pytest.raises(CodedError) as raised:
        with store.begin_publication(expected_context=_context()) as publication:
            _write_base_relations(publication)  # the caller handled its own failure and left
            artifact_id, generation_id = publication.artifact_id, publication.generation_id
    assert raised.value.code == "query.session.warehouse_artifact.publication_cleanup_incomplete"
    assert raised.value.context["relations"] == (f"artifact.{failed[0]}",)
    assert raised.value.context["artifact_id"] == artifact_id
    assert raised.value.context["generation_id"] == generation_id
    assert raised.value.__cause__ is error
    assert len(attempted) == 2
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}


def _fail_second_relation_digest(monkeypatch, store: Any, error: BaseException) -> None:
    """Let the first relation persist, then fail the second one's digest."""
    real = store._digest_table_for_write
    seen = {"n": 0}

    def digest(table, role):
        seen["n"] += 1
        if seen["n"] == 2:
            raise error
        return real(table, role)

    monkeypatch.setattr(store, "_digest_table_for_write", digest)


def test_abort_reports_relation_leaked_by_failed_digest_and_failed_drop(monkeypatch) -> None:
    """Proves a created-then-unregistered relation is reported; DuckDB only, not a hosted drop."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    digest_error = RuntimeError("digest failed")
    _fail_second_relation_digest(monkeypatch, store, digest_error)
    _attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con, persistent=True)
    with pytest.raises(RuntimeError) as raised:
        with store.begin_publication(expected_context=_context()) as publication:
            _write_base_relations(publication)
    assert raised.value is digest_error
    assert len(failed) == 1
    assert any(f"artifact.{failed[0]}" in note for note in digest_error.__notes__)
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}  # the registered relation was dropped


def test_handled_digest_failure_with_failed_drop_raises_cleanup_incomplete(monkeypatch) -> None:
    """Proves a handled leak refuses on normal exit; DuckDB only, not a hosted drop."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    _fail_second_relation_digest(monkeypatch, store, RuntimeError("digest failed"))
    _attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con, persistent=True)
    with pytest.raises(CodedError) as raised:
        with store.begin_publication(expected_context=_context()) as publication:
            with pytest.raises(RuntimeError, match="digest failed"):
                _write_base_relations(publication)
            artifact_id, generation_id = publication.artifact_id, publication.generation_id
    assert raised.value.code == "query.session.warehouse_artifact.publication_cleanup_incomplete"
    assert raised.value.context["relations"] == (f"artifact.{failed[0]}",)
    assert raised.value.context["artifact_id"] == artifact_id
    assert raised.value.context["generation_id"] == generation_id
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}


def test_create_table_failure_with_failed_drop_still_reports_the_locator(monkeypatch) -> None:
    """Proves a create that may have landed is tracked; DuckDB only, not a hosted create."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_create = con.create_table
    create_error = ConnectionError("create outcome unknown")

    def create_then_fail(name, *args, **kwargs):
        real_create(name, *args, **kwargs)  # the table lands, the client sees an error
        raise create_error

    monkeypatch.setattr(con, "create_table", create_then_fail)
    _attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con, persistent=True)
    with pytest.raises(ConnectionError) as raised:
        with store.begin_publication(expected_context=_context()) as publication:
            publication.write_relation("exposures", ibis.memtable([_BASE_ROWS["exposures"]]))
    assert raised.value is create_error
    assert any(f"artifact.{failed[0]}" in note for note in create_error.__notes__)
    monkeypatch.undo()
    assert _publication_relation_names(con) == {failed[0]}


def _ambiguous_insert_with_failed_tombstone(monkeypatch, con: Any, store: Any):
    """Leave a never-published pair whose manifest row can still commit later."""
    real_insert = con.insert
    delayed: list[tuple[Any, Any, Any]] = []

    def flaky_insert(name, obj, *args, **kwargs):
        if name == "ud_manifest_dropped":
            raise RuntimeError("tombstone unavailable")
        if name == "ud_manifest_index":
            delayed.append((name, obj, kwargs))
            raise RuntimeError("submitted")
        return real_insert(name, obj, *args, **kwargs)

    monkeypatch.setattr(con, "insert", flaky_insert)
    with pytest.raises(CodedError) as raised:
        _publish(store, swallow_manifest_error=True)
    monkeypatch.setattr(con, "insert", real_insert)
    return raised.value, real_insert, delayed[0]


def test_abandon_generation_hides_a_late_row_for_a_never_published_pair(monkeypatch) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    refusal, real_insert, (name, obj, kwargs) = _ambiguous_insert_with_failed_tombstone(
        monkeypatch, con, store
    )
    kept = _publication_relation_names(con)
    assert len(kept) == 2

    fresh = WarehouseArtifactStore(con, schema_name="artifact")
    fresh.drop_generation(refusal.context["artifact_id"], refusal.context["generation_id"])
    fresh.abandon_generation(refusal.context["artifact_id"], refusal.context["generation_id"])

    real_insert(name, obj, **kwargs)  # the ambiguous insert commits after the abandon
    row = obj.execute().iloc[0]
    late = UnitDayArtifactRef(
        artifact_id=UUID(row["artifact_id"]),
        generation_id=UUID(row["generation_id"]),
        manifest=RelationLocator(schema="artifact", name=row["manifest_name"]),
        manifest_sha256=row["manifest_sha256"],
    )
    for candidate in (fresh, WarehouseArtifactStore(con, schema_name="artifact")):
        assert candidate.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as hidden:
            with candidate.open_snapshot(late):
                pass
        assert hidden.value.code == "artifact.generation.dropped"
    for relation in kept:
        con.drop_table(relation, database="artifact")
    assert _publication_relation_names(con) == set()
    assert fresh.visible_manifests == ()


def _tombstone_count(con: Any, artifact_id: UUID, generation_id: UUID) -> int:
    dropped = con.table("ud_manifest_dropped", database="artifact")
    rows = con.to_pyarrow(dropped).to_pylist()
    return sum(
        row["artifact_id"] == str(artifact_id) and row["generation_id"] == str(generation_id)
        for row in rows
    )


def test_abandon_generation_is_idempotent_for_seen_and_unseen_pairs() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, _, _ = _publish(store)
    other, _, _ = _publish(store, unit_id="u2")

    for _ in range(2):
        store.abandon_generation(ref.artifact_id, ref.generation_id)
    assert _tombstone_count(con, ref.artifact_id, ref.generation_id) == 1
    assert store.visible_manifests == (other,)
    with pytest.raises(ArtifactContractError) as hidden:
        with store.open_snapshot(ref):
            pass
    assert hidden.value.code == "artifact.generation.dropped"
    store.drop_generation(ref.artifact_id, ref.generation_id)  # still finishes the erase
    assert len(_publication_relation_names(con)) == 2  # only `other` remains

    unseen = (uuid4(), uuid4())
    for _ in range(2):
        store.abandon_generation(*unseen)
    assert _tombstone_count(con, *unseen) == 1


@pytest.mark.parametrize(
    "args",
    [
        ("not-a-uuid", uuid4()),
        (uuid4(), None),
        (str(uuid4()), str(uuid4())),
    ],
)
def test_abandon_generation_refuses_non_uuid_arguments(args: tuple[Any, Any]) -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    with pytest.raises(ArtifactContractError) as refusal:
        store.abandon_generation(*args)
    assert refusal.value.code == "artifact.refresh.invalid_ref"
    assert len(con.to_pyarrow(con.table("ud_manifest_dropped", database="artifact"))) == 0
    context = refusal.value.context
    rejected = tuple(
        name
        for name, value in (("artifact_id", args[0]), ("generation_id", args[1]))
        if not isinstance(value, UUID)
    )
    assert context["rejected"] == rejected
    for name, value in (("artifact_id", args[0]), ("generation_id", args[1])):
        if name in rejected:
            assert context[f"{name}_type"] == type(value).__name__
    assert context["route"]


def test_manifest_publication_is_refused_over_a_relation_whose_write_and_cleanup_failed(
    monkeypatch,
) -> None:
    """A caller that handles an extension write failure cannot publish over the leaked table.

    DuckDB only: the leak is injected as a digest failure plus a failing drop."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    real_digest = store._digest_table_for_write

    def digest(table, role):
        if role == "breakout_dimension":
            raise RuntimeError("extension digest failed")
        return real_digest(table, role)

    monkeypatch.setattr(store, "_digest_table_for_write", digest)
    _attempted, failed, _ = _fail_one_relation_drop(monkeypatch, con, persistent=True)

    def write_failing_extension(publication: Any) -> None:
        with pytest.raises(RuntimeError, match="extension digest failed"):
            publication.write_relation("breakout_dimension", ibis.memtable([{"unit_id": "u1"}]))

    with pytest.raises(ArtifactContractError) as raised:
        _publish(store, before_manifest=write_failing_extension)
    assert raised.value.code == "artifact.manifest.unreferenced_relation"
    assert raised.value.context["relation_names"] == (failed[0],)
    monkeypatch.undo()
    assert store.visible_manifests == ()
    assert _publication_relation_names(con) == {failed[0]}  # base relations erased, leak named


def test_refresh_against_a_different_context_names_the_artifact() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifact")
    ref, _, _ = _publish(store)
    other = _breakout_catalog_context([("d", "facts")])
    assert other.sha256 != _context().sha256
    with pytest.raises(ArtifactContractError) as refusal:
        store.begin_publication(expected_context=other, refresh_of=ref)
    assert refusal.value.code == "artifact.refresh.context_mismatch"
    assert dict(refusal.value.context) == {"artifact_id": str(ref.artifact_id)}
