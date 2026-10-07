"""Definitions/adopted unit-day artifact parity and trust-boundary tests."""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import ibis
import pytest

from increment.analysis import Analysis
from increment.errors import CapabilityError
from increment.query.artifact_contract import (
    ArtifactContractError,
    ArtifactStore,
)
from increment.query.artifact_reader import ARTIFACT_OPERATIONS
from increment.query.schemas import (
    ARTIFACT_RELATION_PRIMARY_KEYS,
    ARTIFACT_RELATION_SCHEMAS,
    UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
)
from increment.query.source import open_artifact
from increment.semantics.models import Breakout
from tests.analysis_factory import _native_source, lift_rows
from tests.test_unit_day_artifact_facade import (
    _definitions,
    _extensions,
    _native,
)

_EXPECTED_ARTIFACT_SCHEMAS = {
    "breakout_dimension": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("value_is_missing", "BOOLEAN", False),
        ("dimension_value", "STRING", False),
    ),
    "factor_dimension": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("value_is_missing", "BOOLEAN", False),
        ("factor_value", "STRING", False),
    ),
    "cluster_identity": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("cluster_id", "STRING", False),
    ),
    "cuped_preperiod": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("x", "FLOAT64", False),
    ),
    "assignment_counts": (
        ("experiment_id", "STRING", False),
        ("population", "STRING", False),
        ("group_id", "STRING", False),
        ("n_units", "INT64", False),
        ("n_randomization_units", "INT64", False),
    ),
    "trigger_population": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("first_trigger_ts", "TIMESTAMP_UTC_US", False),
    ),
    "encouragement_uptake": (
        ("experiment_id", "STRING", False),
        ("unit_id", "STRING", False),
        ("uptake", "BOOLEAN", False),
        ("first_uptake_ts", "TIMESTAMP_UTC_US", True),
    ),
    "site_volume": (
        ("experiment_id", "STRING", False),
        ("ds", "DATE", False),
        ("measure_key", "STRING", False),
        ("n_events", "INT64", False),
        ("sum_value", "FLOAT64", False),
        ("min_value", "FLOAT64", False),
        ("max_value", "FLOAT64", False),
    ),
}

_EXPECTED_ARTIFACT_PRIMARY_KEYS = {
    "breakout_dimension": ("experiment_id", "unit_id"),
    "factor_dimension": ("experiment_id", "unit_id"),
    "cluster_identity": ("experiment_id", "unit_id"),
    "cuped_preperiod": ("experiment_id", "unit_id"),
    "assignment_counts": ("experiment_id", "population", "group_id"),
    "trigger_population": ("experiment_id", "unit_id"),
    "encouragement_uptake": ("experiment_id", "unit_id"),
    "site_volume": ("experiment_id", "ds", "measure_key"),
}


def _assert_rows_equal(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    extra_keys: tuple[str, ...] = (),
    ignored_fields: frozenset[str] = frozenset(),
) -> None:
    key_fields = ("ds", "experiment_id", "metric", "group_id", *extra_keys)

    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {tuple(row.get(field) for field in key_fields): row for row in rows}

    left_by_key = keyed(left)
    right_by_key = keyed(right)
    assert set(left_by_key) == set(right_by_key)
    for key, expected in left_by_key.items():
        actual = right_by_key[key]
        assert set(actual) == set(expected), key
        for field, expected_value in expected.items():
            if field in ignored_fields:
                continue
            actual_value = actual[field]
            if isinstance(expected_value, float) and isinstance(actual_value, float):
                assert math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-10), (
                    key,
                    field,
                    expected_value,
                    actual_value,
                )
            else:
                assert actual_value == expected_value, (key, field)


def _metric(source: Any, name: str = "purchase_rate") -> Any:
    return next(metric for metric in source.context.metrics if metric.name == name)


def _artifact_source(store: Any, reference: Any, context: Any) -> Any:
    """Open the public artifact source an adopted facade reads through."""
    return open_artifact(store, reference, expected_context=context)


def _artifact_relation_table(role: str, rows: list[dict[str, Any]]) -> Any:
    type_names = {
        "STRING": "string",
        "INT64": "int64",
        "FLOAT64": "float64",
        "DATE": "date",
        "TIMESTAMP_UTC_US": "timestamp('UTC')",
        "BOOLEAN": "boolean",
    }
    return ibis.memtable(
        rows,
        schema={
            field: type_names[type_tag]
            for field, type_tag, _nullable in UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role]
        },
    )


def _replace_artifact_relation(
    connection: Any, relation: Any, role: str, rows: list[dict[str, Any]]
) -> None:
    connection.drop_table(
        relation.relation.name, database=relation.relation.schema_name, force=True
    )
    connection.create_table(
        relation.relation.name,
        _artifact_relation_table(role, rows),
        database=relation.relation.schema_name,
        overwrite=False,
    )


# Tests that never write to the store share one publication; each opens its
# own adopted facade from it. One worker owns the group so it is built once.
_shared_publication = pytest.mark.xdist_group("unit_day_artifact_parity")


@pytest.fixture(scope="module")
def publication():
    connection, definitions, context, store = _native()
    reference = definitions.publish_unit_day_artifact(store)
    try:
        yield connection, definitions, context, store, reference
    finally:
        definitions.close()


@pytest.fixture
def published_pair(publication):
    _connection, definitions, context, store, reference = publication
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        yield definitions, adopted, context, store, reference
    finally:
        adopted.close()


@_shared_publication
@pytest.mark.parametrize("ingress", ["definitions", "artifact"])
@pytest.mark.parametrize(
    ("field", "value"), [("n", 0), ("n", -1), ("successes", -1), ("successes", "above_n")]
)
def test_fixed_export_refuses_invalid_backend_counts(
    publication, ingress, field, value, monkeypatch, tmp_path
):
    import pyarrow as pa

    from increment.errors import WireFormatError

    connection, native, context, store, reference = publication
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    to_pyarrow = connection.to_pyarrow

    def corrupted_counts(*args, **kwargs):
        table = to_pyarrow(*args, **kwargs)
        if (
            isinstance(table, pa.Table)
            and table.num_rows
            and {"n", "successes"} <= set(table.column_names)
        ):
            values = table.column(field).to_pylist()
            values[0] = table.column("n")[0].as_py() + 1 if value == "above_n" else value
            index = table.schema.get_field_index(field)
            table = table.set_column(
                index, field, pa.array(values, type=table.schema.field(field).type)
            )
        return table

    monkeypatch.setattr(connection, "to_pyarrow", corrupted_counts)
    path = tmp_path / "invalid.parquet"
    try:
        analysis = native if ingress == "definitions" else adopted
        with pytest.raises(WireFormatError) as exc:
            analysis.export(path)
        assert exc.value.code == "moments.count_out_of_range"
        assert exc.value.context["field"] == field
        assert not path.exists()
    finally:
        adopted.close()


def test_nonbinary_native_and_artifact_exports_keep_nullable_integer_counts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    definitions = _definitions(tmp_path)
    payload = yaml.safe_load(definitions.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["plan"] = {"secondaries": ["avg_session_duration"]}
    definitions.write_text(yaml.safe_dump(payload))
    _connection, native, context, store = _native(definitions=definitions)
    adopted = None
    try:
        reference = native.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
        for name, analysis in (("definitions", native), ("artifact", adopted)):
            path = tmp_path / f"{name}.parquet"
            analysis.export(path)
            table = pq.read_table(path)
            assert table.schema.field("successes").type == pa.int64(), name
            assert table.column("successes").to_pylist() == [None, None], name
    finally:
        if adopted is not None:
            adopted.close()
        native.close()


# One CUPED metric with breakout and factor evidence, published once for the
# read-only covariate parity checks below.
_shared_covariate_publication = pytest.mark.xdist_group("unit_day_artifact_covariate_parity")


@pytest.fixture(scope="module")
def covariate_publication(tmp_path_factory: pytest.TempPathFactory):
    definitions = _definitions(
        tmp_path_factory.mktemp("covariate"),
        site_volume_only=True,
        cuped_metrics=("purchase_rate",),
    )
    _connection, native, context, store = _native(definitions=definitions, with_pre_period=True)
    requests = _extensions(context, "breakout_dimension", "factor_dimension", "cuped_preperiod")
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    try:
        yield native, context, store, reference
    finally:
        native.close()


@pytest.fixture
def covariate_pair(covariate_publication):
    native, context, store, reference = covariate_publication
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        yield native, adopted
    finally:
        adopted.close()


def test_adopted_reduction_does_no_upstream_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from increment.query import session as session_module

    _connection, native, context, store = _native()
    reference = native.publish_unit_day_artifact(store)
    baseline = lift_rows(native.run(metrics=["purchase_rate"]))[0].require_lift().value

    def upstream_work(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adopted reduction attempted upstream work")

    # Native fact tables are built through this module seam, so any callback
    # here after publication means adoption fell back to upstream work.
    monkeypatch.setattr(session_module, "_dim_joined_fact_table", upstream_work)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_rows = lift_rows(adopted.run(metrics=["purchase_rate"]))
    assert adopted_rows[0].require_lift().value == pytest.approx(baseline)
    adopted.close()
    native.close()


@_shared_publication
@pytest.mark.parametrize("grain", ["total", "daily", "asof"])
def test_definitions_and_adopted_reducers_match_staggered_denominators(
    published_pair: tuple[Any, ...], grain: str
) -> None:
    native, _adopted, context, store, reference = published_pair
    native_source = _native_source(native)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        if grain == "total":
            native_moments = native_source.moments_source(
                metrics=(native_metric,), narrow_cuped=True
            )
            adopted_moments = adopted_source
        else:
            native_moments = native_source.day_source(metrics=(native_metric,))
            adopted_moments = adopted_source.day_source(metrics=(adopted_metric,))
        native_rows = cast(Any, native_moments.moments(native_metric, grain=cast(Any, grain)))
        adopted_rows = adopted_moments.moments(adopted_metric, grain=cast(Any, grain))
    finally:
        adopted_source.close()
    _assert_rows_equal(native_rows, adopted_rows)
    if grain in {"daily", "asof"}:
        denominators = {(row["ds"], row["group_id"]): row["n"] for row in native_rows}
        assert len({day for day, _group in denominators}) > 1
        assert all(value > 0 for value in denominators.values())


def test_calendar_day_boundary_is_preserved_across_adoption(tmp_path: Path) -> None:
    definitions_path = _definitions(tmp_path)
    payload = __import__("yaml").safe_load(definitions_path.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["day_boundary"] = "UTC-05:00"
    definitions_path.write_text(__import__("yaml").safe_dump(payload, sort_keys=False))
    _connection, native, context, store = _native(definitions=definitions_path)
    reference = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        native_rows = native_source.day_source(metrics=(native_metric,)).moments(
            native_metric, grain="daily"
        )
        adopted_rows = adopted_source.day_source(metrics=(adopted_metric,)).moments(
            adopted_metric, grain="daily"
        )
        _assert_rows_equal(native_rows, adopted_rows)
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


# Missing source capabilities must refuse rather than fabricate native evidence.
@_shared_publication
def test_adopted_native_only_operations_refuse_with_exact_codes(
    published_pair: tuple[Any, ...],
) -> None:
    _native_analysis, adopted, _context, _store, _reference = published_pair
    for call, code in (
        (adopted.materialize, "facade.analysis.operation"),
        (adopted.panel_sql, "facade.analysis.operation"),
        (adopted.summary_sql, "artifact.operation.unsupported"),
        (
            lambda: adopted.dashboard_group_data(metrics=adopted.metrics),
            "facade.analysis.operation",
        ),
    ):
        with pytest.raises(CapabilityError) as raised:
            call()
        assert raised.value.code == code
        assert raised.value.context


def test_extension_operation_declarations_follow_selected_manifest_extensions(
    tmp_path: Path,
) -> None:
    definitions = _definitions(tmp_path, site_volume_only=True)
    _connection, native, context, store = _native(definitions=definitions)
    requested = _extensions(
        context,
        "breakout_dimension",
        "factor_dimension",
        "assignment_counts",
        "site_volume",
    )
    reference = native.publish_unit_day_artifact(store, extensions=requested)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        assert adopted_source.operations == ARTIFACT_OPERATIONS | {
            "artifact_experiment",
            "breakout_source",
            "breakout_sources",
            "breakout_summaries",
            "factor_summaries",
            "sitewide_evidence",
        }
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


@pytest.mark.parametrize("kind", sorted(_EXPECTED_ARTIFACT_SCHEMAS))
def test_every_extension_schema_and_primary_key_is_exact(kind: str) -> None:
    assert ARTIFACT_RELATION_SCHEMAS[kind] == _EXPECTED_ARTIFACT_SCHEMAS[kind]
    assert ARTIFACT_RELATION_PRIMARY_KEYS[kind] == _EXPECTED_ARTIFACT_PRIMARY_KEYS[kind]


# Guards every approved operation with either native/adopted value parity or an exact refusal.
@pytest.mark.slow
def test_task5_approved_operations_have_values_or_coded_refusals(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, site_volume_only=True, cuped_metrics=("purchase_rate",))
    _connection, native, context, store = _native(definitions=definitions, with_pre_period=True)
    requests = _extensions(
        context,
        "breakout_dimension",
        "factor_dimension",
        "cuped_preperiod",
        "site_volume",
    )
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        breakout = Breakout(property="country", source="event_log")

        native_moments = native_source.moments_source(metrics=(native_metric,))
        adopted_moments = adopted_source.moments_source(metrics=(adopted_metric,))
        _assert_rows_equal(
            cast(Any, native_moments.moments(native_metric)),
            adopted_moments.moments(adopted_metric),
        )

        native_days = native_source.day_source(metrics=(native_metric,))
        adopted_days = adopted_source.day_source(metrics=(adopted_metric,))
        _assert_rows_equal(
            native_days.moments(native_metric, grain="daily"),
            adopted_days.moments(adopted_metric, grain="daily"),
            ignored_fields=frozenset({"ref_x", "cx1", "cx2", "cxy", "cxd", "x_role"}),
        )

        native_breakout = native_source.breakout_source(breakout, metrics=(native_metric,))
        adopted_breakout = adopted_source.breakout_source(breakout, metrics=(adopted_metric,))
        _assert_rows_equal(
            native_breakout.moments(native_metric, by=["country"]),
            adopted_breakout.moments(adopted_metric, by=["country"]),
            extra_keys=("country",),
        )
        native_sources = native_source.breakout_sources((breakout,), metrics=(native_metric,))
        adopted_sources = adopted_source.breakout_sources((breakout,), metrics=(adopted_metric,))
        assert len(native_sources) == len(adopted_sources) == 1
        _assert_rows_equal(
            native_sources[0].moments(native_metric, by=["country"]),
            adopted_sources[0].moments(adopted_metric, by=["country"]),
            extra_keys=("country",),
        )

        native_breakout_views = native.breakout_summaries(metrics=[native_metric])
        adopted_breakout_views = adopted.breakout_summaries(metrics=[adopted_metric])
        assert set(native_breakout_views) == set(adopted_breakout_views)
        for key in native_breakout_views:
            _assert_rows_equal(
                native_breakout_views[key]["group_summary"].to_pylist(),
                adopted_breakout_views[key]["group_summary"].to_pylist(),
                extra_keys=("country",),
            )
        native_factor_views = native.factor_summaries()
        adopted_factor_views = adopted.factor_summaries()
        assert set(native_factor_views) == set(adopted_factor_views)
        for key in native_factor_views:
            _assert_rows_equal(
                native_factor_views[key].to_pylist(),
                adopted_factor_views[key].to_pylist(),
                extra_keys=("country",),
            )

        native_site = native_source.sitewide_evidence(native_metric)
        adopted_site = adopted_source.sitewide_evidence(adopted_metric)
        assert adopted_site.site_total == pytest.approx(native_site.site_total)
        assert adopted_site.site_total_denominator == native_site.site_total_denominator
        # Native materialize returns the same Analysis for chaining; the
        # adopted facade refuses with the exact coded contract.
        assert native.materialize() is native
        for call, code in (
            (adopted.materialize, "facade.analysis.operation"),
            (adopted.panel_sql, "facade.analysis.operation"),
            (adopted.summary_sql, "artifact.operation.unsupported"),
        ):
            with pytest.raises(Exception) as raised:
                call()
            assert raised.value.code == code  # ty: ignore[unresolved-attribute]
            assert raised.value.context  # ty: ignore[unresolved-attribute]
        with pytest.raises(Exception) as raised:
            adopted.materialize()
        assert raised.value.code == "facade.analysis.operation"  # ty: ignore[unresolved-attribute]
        assert native.panel_sql()[native_metric.name]
        assert native.summary_sql()[native_metric.name]

        import pyarrow.parquet as pq

        native_path = tmp_path / "operations-native.parquet"
        adopted_path = tmp_path / "operations-adopted.parquet"
        native_source.export_moments(native_path)
        adopted_source.export_moments(adopted_path)
        native_rows = pq.read_table(native_path).to_pylist()
        adopted_rows = pq.read_table(adopted_path).to_pylist()
        exported_fields = ("experiment_id", "metric", "group_id", "n", "ref_y", "cy1", "cy2")
        _assert_rows_equal(
            [{field: row[field] for field in exported_fields} for row in native_rows],
            [{field: row[field] for field in exported_fields} for row in adopted_rows],
        )
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


# Guards clustered publication/adoption by comparing cluster-identity evidence values.
@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster")
def test_cluster_identity_extension_matches_native_sitewide_evidence(tmp_path: Path) -> None:
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered

    connection = ibis.duckdb.connect()
    connection.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(n_stores_per_arm=40, units_per_store=2),
    )
    definitions = tmp_path / "clustered.yml"
    definitions.write_text(
        _defs_yaml_clustered(trigger=None).replace(
            "start: 2024-01-01T00:00:00",
            "start: 2024-01-01T00:00:00\n    end: 2024-01-09T00:00:00",
        )
    )
    loaded = load(definitions)
    native = Analysis.from_definitions("exp", definitions, connection)
    context = artifact_context(loaded, loaded.experiments[0], "error")
    store = WarehouseArtifactStore(connection, schema_name="artifacts")
    requests = _extensions(context, "cluster_identity", "site_volume")
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source, "revenue")
        adopted_metric = _metric(adopted_source, "revenue")
        native_site = native_source.sitewide_evidence(native_metric)
        adopted_site = adopted_source.sitewide_evidence(adopted_metric)
        assert adopted_site.site_total == pytest.approx(native_site.site_total)
        assert adopted_site.cluster == native_site.cluster == "store_id"
        assert adopted_site.cluster_counts == native_site.cluster_counts
        assert adopted_site.unit_counts == native_site.unit_counts
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


def _cluster_switchback_events() -> Any:
    """A late unexposed event extends fact freshness, not the adopted unit-day rows."""
    import numpy as np
    import pyarrow as pa

    rows: dict[str, list[Any]] = {
        "user_id": [],
        "group_id": [],
        "store_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }

    def add(uid: str, arm: str, ts: Any, event: str) -> None:
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["store_id"].append(uid)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(0.0)
        rows["experiment_id"].append("exp")

    def day(n: int) -> Any:
        return np.datetime64(f"2025-01-{n:02d}T00:00:00")

    for i in range(60):
        add(f"c{i}", "control", day(1), "enrolled")
        add(f"c{i}", "control", day(2), "converted")
    for i in range(30):
        add(f"t{i}", "treatment", day(1), "enrolled")
        add(f"t{i}", "treatment", day(2), "converted")
    for i in range(30, 60):
        add(f"t{i}", "treatment", day(9), "enrolled")
    add("unexposed", "control", day(20), "converted")
    return pa.table(rows)


def _defs_yaml_cluster_switchback() -> str:
    return """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM switchback_cluster_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: converted
        column: null
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: conv
    type: conversion
    entity: user_id
    fact: converted
    window_days: 3
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    cluster: store_id
    start: 2025-01-01T00:00:00
    end: 2025-01-09T00:00:00
    observation_end: 2025-01-20T00:00:00
    control_group: control
    plan:
      primary: [conv]
"""


def test_cluster_extension_spine_parity_matches_native_for_late_enrolled_treatment(
    tmp_path: Path,
) -> None:
    """The extended observation window retains the late treatment cohort.

    An unexposed unit's later event advances freshness beyond the artifact's last row.
    """
    from increment.plan import compile_decision_plan
    from increment.query.artifact_publish import artifact_context
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseArtifactStore, WarehouseSession
    from increment.query.source import open_artifact
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load

    connection = ibis.duckdb.connect()
    connection.create_table("switchback_cluster_events", obj=_cluster_switchback_events())
    definitions_path = tmp_path / "switchback.yml"
    definitions_path.write_text(_defs_yaml_cluster_switchback())
    definitions = load(definitions_path)
    experiment = definitions.experiments[0]
    design = Randomized(control_group="control")
    native = DefinitionsMomentSource(
        WarehouseSession(connection, definitions),
        experiment,
        definitions.metrics,
        store="none",
        on_mixed_assignment="error",
        design=design,
        plan=compile_decision_plan(
            experiment.plan, definitions.metrics, path="warehouse", design=design
        ),
    )
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="switchback_artifacts")
    requests = _extensions(context, "cluster_identity")
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = open_artifact(store, reference, expected_context=context)
    try:
        native_metric = _metric(native, "conv")
        adopted_metric = _metric(adopted, "conv")
        native_rows = {row["group_id"]: row for row in native.moments(native_metric, grain="total")}
        adopted_rows = {
            row["group_id"]: row for row in adopted.moments(adopted_metric, grain="total")
        }
        assert set(native_rows) == set(adopted_rows) == {"control", "treatment"}
        assert native_rows["control"]["n"] == 60, "native control count regressed"
        assert native_rows["treatment"]["n"] == 60, "native treatment count regressed"
        assert adopted_rows["control"]["n"] == native_rows["control"]["n"] == 60
        assert adopted_rows["treatment"]["n"] == native_rows["treatment"]["n"] == 60, (
            "the late-enrolled treatment cohort must not vanish through the "
            "cluster-extension facade"
        )
        native_lift = native_rows["treatment"]["ref_y"] / native_rows["control"]["ref_y"] - 1.0
        adopted_lift = adopted_rows["treatment"]["ref_y"] / adopted_rows["control"]["ref_y"] - 1.0
        assert native_lift == pytest.approx(-0.5)
        assert adopted_lift == pytest.approx(native_lift)
    finally:
        adopted.close()
        native.close()
        connection.disconnect()


@pytest.mark.slow
def test_breakout_factor_and_sitewide_operations_match_adopted_values(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, site_volume_only=True, cuped_metrics=("purchase_rate",))
    _connection, native, context, store = _native(definitions=definitions, with_pre_period=True)
    requests = _extensions(
        context, "breakout_dimension", "factor_dimension", "cuped_preperiod", "site_volume"
    )
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        breakout = Breakout(property="country", source="event_log")
        native_breakout = native_source.breakout_source(breakout, metrics=(native_metric,))
        adopted_breakout = adopted_source.breakout_source(breakout, metrics=(adopted_metric,))
        _assert_rows_equal(
            native_breakout.moments(native_metric, by=["country"]),
            adopted_breakout.moments(adopted_metric, by=["country"]),
            extra_keys=("country",),
        )
        native_factor = native_source.factor_summaries(metrics=(native_metric,))
        adopted_factor = adopted_source.factor_summaries(metrics=(adopted_metric,))
        assert set(native_factor) == set(adopted_factor)
        for key in native_factor:
            _assert_rows_equal(
                native_factor[key].to_pylist(),
                adopted_factor[key].to_pylist(),
                extra_keys=("country",),
            )
        native_site = native_source.sitewide_evidence(native_metric)
        adopted_site = adopted_source.sitewide_evidence(adopted_metric)
        assert adopted_site.site_total == pytest.approx(native_site.site_total)
        assert adopted_site.site_total_denominator == native_site.site_total_denominator
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


_COVARIATE_FIELDS = frozenset({"ref_x", "cx1", "cx2", "cxy", "cxd", "x_role"})


# Guards artifact Boolean dimension labels against drifting from the native spelling.
@pytest.mark.slow
def test_boolean_dimension_labels_match_native_and_string_case_is_preserved(
    tmp_path: Path,
) -> None:
    import yaml

    from examples._seed import seed_event_log
    from increment.query.session import WarehouseArtifactStore
    from tests.test_unit_day_artifact_facade import _expected_context

    path = _definitions(tmp_path, breakouts=[{"property": "country"}], site_volume_only=True)
    payload = yaml.safe_load(path.read_text())
    source = next(item for item in payload["fact_sources"] if item["name"] == "event_log")
    source["properties"].append(
        {"name": "is_vip", "column": "is_vip", "dtype": "bool", "as_of": "static"}
    )
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["breakouts"] = [{"property": "is_vip"}, {"property": "country"}]
    experiment["factors"] = [{"property": "is_vip"}]
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    con = ibis.duckdb.connect()
    seed_event_log(con)
    con.raw_sql("ALTER TABLE analytics.event_log ADD COLUMN is_vip BOOLEAN")
    con.raw_sql(
        "UPDATE analytics.event_log SET is_vip = CASE hash(user_id) % 3 "
        "WHEN 0 THEN TRUE WHEN 1 THEN FALSE ELSE NULL END"
    )
    native = Analysis.from_definitions("new_onboarding_v2", path, con)
    context = _expected_context(path)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    requests = _extensions(context, "breakout_dimension", "factor_dimension")
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        native_views = native.breakout_summaries()
        adopted_views = adopted.breakout_summaries()
        assert set(native_views) == set(adopted_views)
        labels: dict[str, set[Any]] = {}
        for key, native_view in native_views.items():
            native_rows = native_view["group_summary"].to_pylist()
            adopted_rows = adopted_views[key]["group_summary"].to_pylist()
            dimension = next(name for name in ("is_vip", "country") if name in native_rows[0])
            _assert_rows_equal(
                native_rows, adopted_rows, extra_keys=(dimension,), ignored_fields=_COVARIATE_FIELDS
            )
            labels[dimension] = {row[dimension] for row in adopted_rows}
        assert labels["is_vip"] == {"true", "false", "__null__"}
        assert "US" in labels["country"]

        native_factor = native.factor_summaries()
        adopted_factor = adopted.factor_summaries()
        assert set(native_factor) == set(adopted_factor)
        for key in native_factor:
            adopted_rows = adopted_factor[key].to_pylist()
            _assert_rows_equal(
                native_factor[key].to_pylist(),
                adopted_rows,
                extra_keys=("is_vip",),
                ignored_fields=_COVARIATE_FIELDS,
            )
            assert {row["is_vip"] for row in adopted_rows} <= {"true", "false", "__null__"}
    finally:
        adopted.close()
        native.close()


# Guards adopted factor summaries against dropping the CUPED covariate moments.
@_shared_covariate_publication
def test_adopted_factor_summaries_carry_the_native_covariate_moments(
    covariate_pair: tuple[Any, ...],
) -> None:
    native, adopted = covariate_pair
    key = "purchase_rate:country:event_log"
    native_factor = native.factor_summaries()
    adopted_factor = adopted.factor_summaries()
    assert set(native_factor) == set(adopted_factor) == {key}
    native_rows = native_factor[key].to_pylist()
    assert any(abs(float(row["ref_x"])) > 0 for row in native_rows)
    _assert_rows_equal(native_rows, adopted_factor[key].to_pylist(), extra_keys=("country",))


# Guards the adopted daily breakout table against being reduced at the wrong grain.
@_shared_covariate_publication
def test_adopted_daily_breakout_summaries_match_the_native_day_axis(
    covariate_pair: tuple[Any, ...],
) -> None:
    native, adopted = covariate_pair
    key = "purchase_rate:country:event_log"
    native_daily = native.breakout_summaries()[key]["daily_group_summary"].to_pylist()
    adopted_daily = adopted.breakout_summaries()[key]["daily_group_summary"].to_pylist()
    assert len({row.get("ds") for row in adopted_daily}) > 1
    _assert_rows_equal(
        native_daily,
        adopted_daily,
        extra_keys=("country",),
        ignored_fields=frozenset({"ref_x", "cx1", "cx2", "cxy", "cxd", "x_role"}),
    )


def test_triggered_operations_match_and_preserve_zero_arm_counts(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    _connection, native, context, store = _native(definitions=definitions)
    requests = _extensions(context, "assignment_counts", "trigger_population")
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        assert "triggered_counts" in adopted_source.operations
        assert "triggered_source" in adopted_source.operations
        assert adopted_source.triggered_counts() == native_source.triggered_counts()
        assert adopted_source.trigger_rates() == pytest.approx(native_source.trigger_rates())
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        native_rows = cast(Any, native_source.triggered_source().moments(native_metric))
        adopted_rows = adopted_source.triggered_source().moments(adopted_metric)
        _assert_rows_equal(
            native_rows,
            adopted_rows,
            ignored_fields=frozenset({"ref_x", "cx1", "cx2", "cxy", "cxd", "x_role"}),
        )
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


@_shared_publication
def test_absent_extension_is_identical_coded_refusal(published_pair: tuple[Any, ...]) -> None:
    _native_analysis, _adopted, context, store, reference = published_pair
    adopted_source = _artifact_source(store, reference, context)
    try:
        metric = _metric(adopted_source)
        with pytest.raises(ArtifactContractError) as raised:
            adopted_source.moments(metric, by=["country"])
        assert raised.value.code == "artifact.extension.missing"
    finally:
        adopted_source.close()


@_shared_publication
def test_fixed_trusted_ref_tamper_is_rejected(publication: tuple[Any, ...]) -> None:
    _connection, _native_analysis, context, store, reference = publication
    tampered = reference.model_copy(update={"manifest_sha256": "0" * 64})
    with pytest.raises(ArtifactContractError) as raised:
        Analysis.from_unit_day_artifact(store, tampered, expected_context=context)
    assert raised.value.code == "artifact.refresh.invalid_ref"


# Guards both base-role manifest bindings against identity, role, and primary-key drift.
@_shared_publication
@pytest.mark.parametrize("role", ["exposures", "measure_stats"])
@pytest.mark.parametrize("mutation", ["artifact_id", "role", "primary_key"])
def test_manifest_base_relation_mutations_refuse_exactly(
    published_pair: tuple[Any, ...], role: str, mutation: str
) -> None:
    _native_analysis, _adopted, context, store, reference = published_pair
    source = _artifact_source(store, reference, context)
    relation = getattr(source.manifest.base, role)
    if mutation == "artifact_id":
        updates = {"artifact_id": uuid4()}
    elif mutation == "role":
        updates = {"role": "measure_stats" if role == "exposures" else "exposures"}
    else:
        updates = {
            "primary_key": (
                ("unit_id", "experiment_id")
                if role == "exposures"
                else ("experiment_id", "ds", "unit_id", "measure_key")
            )
        }
    bad_relation = relation.model_copy(update=updates)
    source._manifest = source.manifest.model_copy(
        update={"base": source.manifest.base.model_copy(update={role: bad_relation})}
    )
    try:
        with pytest.raises(ArtifactContractError) as raised:
            source.moments(_metric(source))
        assert raised.value.code == "artifact.snapshot.mixed"
    finally:
        source.close()


# Replacing both artifact and caller ref together remains outside this race guard.


# Guards a verify-then-rebind race by mutating the physical relation after verification.
def test_verify_then_rebind_keeps_the_original_snapshot() -> None:
    connection, native, context, store = _native()
    reference = native.publish_unit_day_artifact(store)
    state: dict[str, Any] = {"rebound": False, "rebound_rows": [], "warehouse_rows": []}

    class RacingSnapshot:
        def __init__(self, snapshot: Any, owner: Any) -> None:
            self._snapshot = snapshot
            self._owner = owner
            self.artifact_id = snapshot.artifact_id
            self.generation_id = snapshot.generation_id

        def read_manifest(self, locator: Any, *, expected_sha256: str) -> Any:
            return self._snapshot.read_manifest(locator, expected_sha256=expected_sha256)

        def verify_relation(self, relation: Any, *, expected_role: str) -> Any:
            result = self._snapshot.verify_relation(relation, expected_role=expected_role)
            if expected_role == "measure_stats":
                relation_name = relation.relation.name
                relation_schema = relation.relation.schema_name
                self._owner.connection.raw_sql(
                    f'UPDATE "{relation_schema}"."{relation_name}" '
                    "SET sum_value = 999999.0, min_value = 999999.0, max_value = 999999.0"
                )
                state["warehouse_rows"] = self._owner.connection.to_pyarrow(
                    self._owner.connection.table(relation_name, database=relation_schema)
                ).to_pylist()
            self._owner.rebound_rows = [
                {
                    "experiment_id": "rebound",
                    "unit_id": "replacement",
                    "group_id": "replacement",
                }
            ]
            state["rebound"] = True
            state["rebound_rows"] = self._owner.rebound_rows
            return result

        def execute(self, expression: Any) -> Any:
            return self._snapshot.execute(expression)

    class RacingStore:
        rebound_rows: list[dict[str, str]] = []

        def __init__(self) -> None:
            self.connection = connection

        def validate_locator(self, locator: Any, **kwargs: Any) -> None:
            store.validate_locator(locator, **kwargs)

        @contextmanager
        def open_snapshot(self, fixed_ref: Any):
            with store.open_snapshot(fixed_ref) as snapshot:
                yield RacingSnapshot(snapshot, self)

    adopted = Analysis.from_unit_day_artifact(
        cast(ArtifactStore, RacingStore()), reference, expected_context=context
    )
    try:
        rows = lift_rows(adopted.run(metrics=["purchase_rate"]))
        assert state["rebound"]
        assert state["rebound_rows"][0]["unit_id"] == "replacement"
        assert state["warehouse_rows"]
        assert state["warehouse_rows"][0]["sum_value"] == 999999.0
        assert rows[0].require_lift().value == pytest.approx(
            lift_rows(native.run(metrics=["purchase_rate"]))[0].require_lift().value
        )
        assert rows[0].require_lift().value != 999999.0
    finally:
        adopted.close()
        native.close()


@pytest.mark.slow
def test_export_and_fused_breakout_summaries_match_the_definitions_source(
    tmp_path: Path,
) -> None:
    definitions = _definitions(tmp_path, site_volume_only=True, cuped_metrics=("purchase_rate",))
    _connection, native, context, store = _native(definitions=definitions, with_pre_period=True)
    requests = _extensions(
        context, "breakout_dimension", "cuped_preperiod", "factor_dimension", "site_volume"
    )
    reference = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        native_views = native_source.breakout_summaries(metrics=(native_metric,))
        adopted_views = adopted_source.breakout_summaries(metrics=(adopted_metric,))
        assert set(native_views) == set(adopted_views)
        for key in native_views:
            _assert_rows_equal(
                native_views[key]["group_summary"].to_pylist(),
                adopted_views[key]["group_summary"].to_pylist(),
                extra_keys=("country",),
            )
            _assert_rows_equal(
                native_views[key]["daily_group_summary"].to_pylist(),
                adopted_views[key]["daily_group_summary"].to_pylist(),
                extra_keys=("country",),
                ignored_fields=frozenset({"ref_x", "cx1", "cx2", "cxy", "cxd", "x_role"}),
            )

        import pyarrow.parquet as pq

        native_path = tmp_path / "native.parquet"
        adopted_path = tmp_path / "adopted.parquet"
        native_source.export_moments(native_path)
        adopted_source.export_moments(adopted_path)
        native_rows = [
            row
            for row in pq.read_table(native_path).to_pylist()
            if row["metric"] == "purchase_rate"
        ]
        adopted_rows = [
            row
            for row in pq.read_table(adopted_path).to_pylist()
            if row["metric"] == "purchase_rate"
        ]
        exported_fields = (
            "experiment_id",
            "metric",
            "group_id",
            "n",
            "ref_y",
            "cy1",
            "cy2",
            "successes",
        )
        _assert_rows_equal(
            [{field: row[field] for field in exported_fields} for row in native_rows],
            [{field: row[field] for field in exported_fields} for row in adopted_rows],
        )
        assert all(type(row["successes"]) is int for row in adopted_rows)
        assert {row["moments_format"] for row in adopted_rows} == {10}
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


# Guards caller-pinned context and generation values before any store access.
@_shared_publication
def test_caller_trusted_context_and_generation_are_not_replaceable(
    publication: tuple[Any, ...],
) -> None:
    connection, _native_analysis, context, store, reference = publication
    from increment.query.artifact_publish import artifact_context
    from increment.semantics.loader import load

    other_definitions = load("examples/definitions")
    other_experiment = next(
        item for item in other_definitions.experiments if item.name == "pricing_tier_test"
    )
    other = Analysis.from_definitions("pricing_tier_test", "examples/definitions", connection)
    other_context = artifact_context(other_definitions, other_experiment, "error")
    try:
        with pytest.raises(ArtifactContractError) as context_error:
            Analysis.from_unit_day_artifact(store, reference, expected_context=other_context)
        assert context_error.value.code == "artifact.refresh.context_mismatch"
        bad_generation = reference.model_copy(update={"generation_id": uuid4()})
        with pytest.raises(ArtifactContractError) as generation_error:
            Analysis.from_unit_day_artifact(store, bad_generation, expected_context=context)
        assert generation_error.value.code == "artifact.refresh.invalid_ref"

        class NoStoreAccess:
            def validate_locator(self, *_args: Any, **_kwargs: Any) -> None:
                raise AssertionError("context validation reached the store")

            def open_snapshot(self, *_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("context validation reached the store")

        unsupported_context = context.model_copy(update={"context_format": 1})
        with pytest.raises(ArtifactContractError) as format_error:
            Analysis.from_unit_day_artifact(
                cast(ArtifactStore, NoStoreAccess()),
                reference,
                expected_context=unsupported_context,
            )
        assert format_error.value.code == "artifact.format.unsupported"
    finally:
        other.close()


@_shared_publication
def test_base_relation_generation_binding_is_coded(published_pair: tuple[Any, ...]) -> None:
    _native_analysis, _adopted, context, store, reference = published_pair
    source = _artifact_source(store, reference, context)
    base = source.manifest.base
    bad_exposures = base.exposures.model_copy(update={"generation_id": uuid4()})
    source._manifest = source.manifest.model_copy(
        update={"base": base.model_copy(update={"exposures": bad_exposures})}
    )
    try:
        with pytest.raises(ArtifactContractError) as raised:
            source.moments(_metric(source))
        assert raised.value.code == "artifact.snapshot.mixed"
    finally:
        source.close()


# Guards adopted base rows against key, domain, duplicate, impossible, and non-finite mutations.
# One mutation per refusal code stays in the fast tier; the rest publish a fresh artifact each.
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("exposure_key", "artifact.snapshot.mixed"),
        pytest.param("exposure_date", "artifact.snapshot.mixed", marks=pytest.mark.slow),
        pytest.param("undeclared_measure", "artifact.snapshot.mixed", marks=pytest.mark.slow),
        pytest.param("out_of_domain_day", "artifact.snapshot.mixed", marks=pytest.mark.slow),
        ("duplicate_key", "artifact.digest.primary_key"),
        pytest.param("impossible_stats", "artifact.snapshot.mixed", marks=pytest.mark.slow),
        ("nonfinite_stats", "artifact.digest.nullable"),
    ],
)
def test_adopted_row_mutations_refuse_with_exact_codes(mutation: str, expected_code: str) -> None:
    connection, native, context, store = _native()
    reference = native.publish_unit_day_artifact(store)
    source = _artifact_source(store, reference, context)
    try:
        role = "exposures" if mutation.startswith("exposure") else "measure_stats"
        relation = getattr(source.manifest.base, role)
        rows = connection.to_pyarrow(
            connection.table(relation.relation.name, database=relation.relation.schema_name)
        ).to_pylist()
        assert rows
        if mutation == "exposure_key":
            rows[0]["experiment_id"] = "tampered_experiment"
        elif mutation == "exposure_date":
            rows[0]["first_exposure_date"] = rows[0]["first_exposure_date"] + timedelta(days=1)
        elif mutation == "undeclared_measure":
            rows[0]["measure_key"] = "undeclared_measure"
        elif mutation == "out_of_domain_day":
            rows[0]["ds"] = date(1900, 1, 1)
        elif mutation == "duplicate_key":
            rows.append(dict(rows[0]))
        elif mutation == "impossible_stats":
            rows[0]["n_events"] = 0
        else:
            rows[0]["sum_value"] = float("nan")
        _replace_artifact_relation(connection, relation, role, [dict(row) for row in rows])
        with pytest.raises(Exception) as raised:
            source.moments(_metric(source))
        assert raised.value.code == expected_code  # ty: ignore[unresolved-attribute]
    finally:
        source.close()
        native.close()


# Guards publication from exposing a manifest after impossible sufficient statistics.
@pytest.mark.parametrize("mutation", ["invalid_n_events", "inconsistent_extrema"])
def test_publication_impossible_aggregate_refuses_and_leaves_no_manifest(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection, native, _context, store = _native()
    from increment.query import artifact_publish as artifact_publish_module

    original = artifact_publish_module.post_exposure_stats

    def invalid_stats(*args: Any, **kwargs: Any) -> Any:
        stats = original(*args, **kwargs)
        rows = connection.to_pyarrow(stats).to_pylist()
        if rows:
            if mutation == "invalid_n_events":
                rows[0]["n_events"] = 0
            else:
                rows[0]["sum_value"] = float(rows[0]["min_value"]) - 1.0
        return ibis.memtable(rows)

    monkeypatch.setattr(artifact_publish_module, "post_exposure_stats", invalid_stats)
    try:
        with pytest.raises(Exception) as raised:
            native.publish_unit_day_artifact(store)
        assert raised.value.code == "artifact.aggregate.impossible"  # ty: ignore[unresolved-attribute]
        assert store.visible_manifests == ()
    finally:
        native.close()


# Guards one frozen artifact.extension.missing refusal for every absent extension family.
@_shared_publication
@pytest.mark.parametrize(
    ("kind", "request_payload"),
    [
        (
            "breakout_dimension",
            {"kind": "breakout_dimension", "property_name": "country", "source_name": "event_log"},
        ),
        (
            "factor_dimension",
            {"kind": "factor_dimension", "property_name": "country", "source_name": "event_log"},
        ),
        ("cluster_identity", {"kind": "cluster_identity", "cluster_name": "account"}),
        ("cuped_preperiod", {"kind": "cuped_preperiod", "metric_name": "purchase_rate"}),
        ("assignment_counts", {"kind": "assignment_counts", "populations": ("assigned",)}),
        ("trigger_population", {"kind": "trigger_population", "trigger_name": "clicked"}),
        ("encouragement_uptake", {"kind": "encouragement_uptake", "uptake_name": "clicked"}),
        ("site_volume", {"kind": "site_volume", "metric_names": ("purchase_rate",)}),
    ],
)
def test_every_absent_extension_family_has_the_same_coded_refusal(
    published_pair: tuple[Any, ...], kind: str, request_payload: dict[str, Any]
) -> None:
    _native_analysis, _adopted, context, store, reference = published_pair
    assert request_payload["kind"] == kind
    adopted_source = _artifact_source(store, reference, context)
    try:
        with pytest.raises(ArtifactContractError) as adopted_error:
            adopted_source._extension(request_payload)
        assert adopted_error.value.code == "artifact.extension.missing"
    finally:
        adopted_source.close()
    from increment.query.artifact_extensions import read_extension

    with store.open_snapshot(reference) as snapshot:
        with pytest.raises(ArtifactContractError) as native_error:
            read_extension(
                snapshot,
                None,
                request=request_payload,
                context=context,
                experiment_id=reference.artifact_id.hex,
            )
    assert native_error.value.code == "artifact.extension.missing"


def test_site_volume_zero_relation_and_incomplete_coverage_are_distinct(
    tmp_path: Path,
) -> None:
    from increment.query.artifact_extensions import read_site_volume_extension

    connection, native, context, store = _native(
        definitions=_definitions(tmp_path, site_volume_only=True)
    )
    connection.raw_sql("DELETE FROM analytics.event_log WHERE event = 'purchase'")
    request = _extensions(context, "site_volume")[0]
    reference = native.publish_unit_day_artifact(store, extensions=(request,))
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    adopted_source = _artifact_source(store, reference, context)
    try:
        metric = _metric(adopted_source)
        extension = next(
            item for item in adopted_source.manifest.extensions if item.kind == "site_volume"
        )
        with store.open_snapshot(reference) as snapshot:
            rows = read_site_volume_extension(
                snapshot,
                extension,
                request=request,
                context=context,
                experiment_id=reference.artifact_id.hex,
            )
        assert rows
        assert all(row["n_events"] == 0 and row["sum_value"] == 0.0 for row in rows)
        assert adopted_source.sitewide_evidence(metric).site_total == 0.0
        incomplete = extension.model_copy(
            update={
                "last_ds": extension.last_ds + timedelta(days=1),
                "freshness": extension.freshness.model_copy(update={"declared_complete": False}),
            }
        )
        with store.open_snapshot(reference) as snapshot:
            with pytest.raises(ArtifactContractError) as raised:
                read_site_volume_extension(
                    snapshot,
                    incomplete,
                    request=request,
                    context=context,
                    experiment_id=reference.artifact_id.hex,
                )
        assert raised.value.code == "artifact.extension.invalid"
    finally:
        adopted_source.close()
        adopted.close()
        native.close()


def _publication_context() -> Any:
    import hashlib

    from increment.query.artifact_contract import ARTIFACT_DOMAIN_ROOT
    from increment.semantics.artifact import ArtifactContext

    payload = (
        '{"context_format":2,"experiment_name":"demo",'
        '"window_days":{"end":null,"observation_horizon":null,"start":"2025-01-01"}}'
    )
    return ArtifactContext(
        canonical_json=payload,
        sha256=hashlib.sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + payload.encode()).hexdigest(),
    )


def _publish_and_open_exposures(
    directory: Path, rows: int, *, memory_limit: str | None = None
) -> dict[str, Any]:
    """Publish a `rows`-row exposures relation on a fresh file-backed DuckDB,
    then open the artifact and verify that relation. Reports the relation ref
    and the Python-heap peak (MiB) and wall time of each step."""
    import time
    import tracemalloc
    from datetime import UTC, datetime

    from increment.query.artifact_digest import manifest_sha256
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import (
        BaseRelations,
        MeasureManifest,
        UnitDayArtifactManifest,
    )

    def traced(step: Any) -> tuple[Any, float, float]:
        tracemalloc.start()
        started = time.perf_counter()
        try:
            result = step()
            peak = tracemalloc.get_traced_memory()[1] / 1024**2
            return result, peak, time.perf_counter() - started
        finally:
            tracemalloc.stop()

    con = ibis.duckdb.connect(str(directory / "warehouse.duckdb"), threads=1)
    try:
        if memory_limit is not None:
            con.raw_sql(f"SET memory_limit='{memory_limit}'")
            con.raw_sql(f"SET temp_directory='{directory / 'spill'}'")
        con.raw_sql(
            f"""create table exposures_source as select 'exp1' as experiment_id,
            'unit_' || i as unit_id,
            CASE WHEN i % 2 = 0 THEN 'control' ELSE 'treatment' END as group_id,
            (TIMESTAMPTZ '2024-01-01 00:00:00+00' + to_microseconds(i)) as first_exposure_ts,
            DATE '2024-01-01' as first_exposure_date
            from range({rows}) t(i)"""
        )
        table = con.table("exposures_source")
        stats = ibis.memtable(
            [
                {
                    "experiment_id": "exp1",
                    "unit_id": "unit_0",
                    "ds": date(2024, 1, 1),
                    "measure_key": "measure_a",
                    "n_events": 1,
                    "sum_value": 2.0,
                    "min_value": 2.0,
                    "max_value": 2.0,
                }
            ]
        )
        store = WarehouseArtifactStore(con, schema_name="artifacts")
        context = _publication_context()
        with store.begin_publication(expected_context=context) as warm:
            warm.write_relation("exposures", table.limit(10))

        def publish() -> Any:
            with store.begin_publication(expected_context=context) as publication:
                exposures = publication.write_relation("exposures", table)
                body: dict[str, Any] = {
                    "artifact_format": 1,
                    "artifact_id": publication.artifact_id,
                    "generation_id": publication.generation_id,
                    "experiment_id": "exp1",
                    "created_at": datetime(2024, 1, 1, tzinfo=UTC),
                    "day_boundary": "UTC",
                    "first_ds": date(2024, 1, 1),
                    "last_ds": date(2024, 1, 1),
                    "base": BaseRelations(
                        exposures=exposures,
                        measure_stats=publication.write_relation("measure_stats", stats),
                    ),
                    "measures": (
                        MeasureManifest(
                            measure_key="measure_a",
                            source_provenance_sha256="0" * 64,
                            freshness={
                                "loaded_through": date(2024, 1, 1),
                                "declared_complete": True,
                            },
                        ),
                    ),
                    "metric_measures": (),
                    "context": context,
                }
                prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
                body["manifest_sha256"] = manifest_sha256(prototype)
                return publication.publish_manifest(UnitDayArtifactManifest(**body)), exposures

        (ref, exposures), publish_peak, publish_seconds = traced(publish)

        def open_and_verify() -> int:
            with store.open_snapshot(ref) as snapshot:
                verified = snapshot.verify_relation(exposures, expected_role="exposures")
                return int(con.to_pyarrow(verified.count()).as_py())

        verified_rows, open_peak, open_seconds = traced(open_and_verify)
        return {
            "exposures": exposures,
            "verified_rows": verified_rows,
            "publish_peak_mib": publish_peak,
            "open_peak_mib": open_peak,
            "publish_seconds": publish_seconds,
            "open_seconds": open_seconds,
        }
    finally:
        con.disconnect()


# Both scale tests read the same unconstrained 1M-row run: publishing it once
# per module halves their cost, and the group keeps them on one worker so the
# module fixture is not rebuilt per xdist process.
@pytest.fixture(scope="module")
def unconstrained_million_row_publication(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    return _publish_and_open_exposures(tmp_path_factory.mktemp("million-rows"), 1_000_000)


# Neither publishing nor opening may pull warehouse-scale data into the Python
# heap: the content digest runs in the warehouse and returns at most 4,096
# bucket rows, so the heap peak must not track relation size.
@pytest.mark.slow
@pytest.mark.xdist_group("artifact_scale")
def test_publish_and_open_python_heap_does_not_grow_with_relation_size(
    tmp_path: Path, unconstrained_million_row_publication: dict[str, Any]
) -> None:
    small = _publish_and_open_exposures(tmp_path, 5_000)
    large = unconstrained_million_row_publication
    assert (small["verified_rows"], large["verified_rows"]) == (5_000, 1_000_000)
    for step in ("publish_peak_mib", "open_peak_mib"):
        assert small[step] < 6.0, (step, small, large)
        assert large[step] < small[step] + 2.0, (step, small, large)


# The warehouse-side share is governed by the warehouse's own memory manager:
# under a 64 MB DuckDB limit with a spill directory, publishing and verifying a
# 1M-row relation still succeed, with the digest an unconstrained run computes.
@pytest.mark.slow
@pytest.mark.xdist_group("artifact_scale")
def test_publish_and_open_complete_under_a_tight_duckdb_memory_limit(
    tmp_path: Path, unconstrained_million_row_publication: dict[str, Any]
) -> None:
    free = unconstrained_million_row_publication
    limited = _publish_and_open_exposures(tmp_path, 1_000_000, memory_limit="64MB")
    assert limited["verified_rows"] == 1_000_000
    assert limited["exposures"].digest_format == 2
    assert limited["exposures"].content_sha256 == free["exposures"].content_sha256


@pytest.mark.slow
def test_streaming_publication_validates_batches_without_full_stats_fetch(monkeypatch):
    from increment.query import artifact_publish as module
    from increment.query.artifact_digest import digest_relation_sql_v2

    con, native, _context, store = _native(
        definitions=Path(__file__).resolve().parents[1] / "examples" / "definitions"
    )
    arrow = con.to_pyarrow
    validate = module.validate_unit_day_aggregate_rows
    batch_sizes = []

    def guarded(expression, **kwargs):
        # Scalar aggregates (counts) have no columns and are always allowed.
        columns = getattr(expression, "columns", ())
        assert "n_events" not in columns, "statistics fetched in full"
        return arrow(expression, **kwargs)

    def checked(rows):
        assert len(rows) <= 128
        batch_sizes.append(len(rows))
        validate(rows)

    monkeypatch.setattr(con, "to_pyarrow", guarded)
    monkeypatch.setattr(module, "_DIGEST_BATCH_ROWS", 128)
    monkeypatch.setattr(module, "validate_unit_day_aggregate_rows", checked)
    try:
        ref = native.publish_unit_day_artifact(store)
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        measure = manifest.base.measure_stats
        assert len(batch_sizes) > 1
        assert sum(batch_sizes) == measure.row_count
        stored = con.table(measure.relation.name, database="artifacts")
        rows = arrow(stored).to_pylist()
        recomputed = digest_relation_sql_v2(
            con,
            stored,
            "measure_stats",
            ARTIFACT_RELATION_SCHEMAS["measure_stats"],
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["measure_stats"],
        )
        assert measure.digest_format == 2
        assert recomputed.content_sha256 == measure.content_sha256
        assert manifest.last_ds == max(row["ds"] for row in rows)
    finally:
        native.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("single_metric", [True, False])
def test_publication_keeps_validated_stats_when_source_changes(
    monkeypatch, tmp_path, single_metric
):
    from examples._seed import seed_event_log
    from increment.query.artifact_publish import ArtifactPublisher

    con, native, context, store = _native(
        definitions=_definitions(tmp_path, open_ended=True, site_volume_only=single_metric)
    )
    seed_event_log(con, n_units=200)
    baseline_ref = native.publish_unit_day_artifact(store)
    with store.open_snapshot(baseline_ref) as snapshot:
        baseline_manifest = snapshot.read_manifest(
            baseline_ref.manifest, expected_sha256=baseline_ref.manifest_sha256
        )
        baseline_stats = snapshot.verify_relation(
            baseline_manifest.base.measure_stats, expected_role="measure_stats"
        )
        baseline_rows = con.to_pyarrow(baseline_stats).to_pylist()
    baseline = Analysis.from_unit_day_artifact(store, baseline_ref, expected_context=context)
    validate = ArtifactPublisher._validate_stats_batches
    validated_rows = []
    columns = ["unit_id", "ds", "n_events", "sum_value", "min_value", "max_value"]

    def validate_then_change(self, stats):
        last_ds = validate(self, stats)
        validated_rows.extend(con.to_pyarrow(stats.select(columns)).to_pylist())
        con.raw_sql(
            "UPDATE analytics.event_log SET event_at = event_at + INTERVAL 365 DAY, "
            "revenue = revenue + 1000 WHERE event = 'purchase'"
        )
        return last_ds

    monkeypatch.setattr(ArtifactPublisher, "_validate_stats_batches", validate_then_change)
    adopted = None
    try:
        ref = native.publish_unit_day_artifact(store)
        assert validated_rows
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            stats = snapshot.verify_relation(
                manifest.base.measure_stats, expected_role="measure_stats"
            )
            published_rows = con.to_pyarrow(stats).to_pylist()
            rows = [{column: row[column] for column in columns} for row in published_rows]

        def ordered(rows):
            return sorted(rows, key=lambda row: tuple(row[column] for column in columns))

        assert ordered(rows) == ordered(validated_rows)
        assert ordered(published_rows) == ordered(baseline_rows)
        assert max(row["ds"] for row in rows) <= manifest.last_ds
        assert manifest.first_ds == baseline_manifest.first_ds
        assert manifest.last_ds == baseline_manifest.last_ds
        assert manifest.measures == baseline_manifest.measures
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        for metric in adopted._src.context.metrics:
            for grain in ("total", "daily", "asof"):
                expected = baseline._src.moments(metric, grain=grain)
                assert expected
                _assert_rows_equal(expected, adopted._src.moments(metric, grain=grain))
    finally:
        if adopted is not None:
            adopted.close()
        baseline.close()
        native.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("failure", [None, "validation", "publication"])
def test_publication_cleans_materialized_inputs_and_stats(monkeypatch, tmp_path, failure):
    from increment.query.artifact_publish import ArtifactPublisher

    con, native, _context, store = _native(
        definitions=_definitions(tmp_path, site_volume_only=True)
    )
    create = con.create_table
    materialized = []

    def tracked(name, obj=None, **kwargs):
        table = create(name, obj, **kwargs)
        if kwargs.get("temp"):
            materialized.append(table)
        return table

    def broken(*args, **kwargs):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(con, "create_table", tracked)
    if failure == "validation":
        monkeypatch.setattr(ArtifactPublisher, "_validate_stats_batches", broken)
    elif failure == "publication":
        monkeypatch.setattr(store, "_write_relation", broken)
    try:
        if failure is None:
            native.publish_unit_day_artifact(store)
        else:
            with pytest.raises(RuntimeError):
                native.publish_unit_day_artifact(store)
            assert store.visible_manifests == ()
        assert all(table.op().name not in con.list_tables() for table in materialized)
    finally:
        native.close()
        con.disconnect()


@pytest.mark.slow
def test_publication_pins_ratio_streams_before_inter_stream_mutation(monkeypatch, tmp_path):
    import yaml

    from examples._seed import seed_event_log

    definitions = _definitions(tmp_path)
    payload = yaml.safe_load(definitions.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["plan"] = {"secondaries": ["aov"]}
    experiment["n_pre_periods"] = 0
    definitions.write_text(yaml.safe_dump(payload))
    con, native, _context, store = _native(definitions=definitions)
    seed_event_log(con, n_units=200)

    def published():
        ref = native.publish_unit_day_artifact(store)
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            stats = snapshot.verify_relation(
                manifest.base.measure_stats, expected_role="measure_stats"
            )
            rows = con.to_pyarrow(stats.order_by(["measure_key", "unit_id", "ds"])).to_pylist()
        return manifest, rows

    create = con.create_table
    captures = []

    def capture_then_mutate(name, obj=None, **kwargs):
        table = create(name, obj, **kwargs)
        if kwargs.get("temp") and "_stream" in table.columns:
            captures.append(table)
            if len(captures) == 1:
                # Both ratio parts read purchase; delete it after the first CTAS.
                con.raw_sql("DELETE FROM analytics.event_log WHERE event = 'purchase'")
        return table

    try:
        baseline_manifest, baseline_rows = published()
        assert baseline_rows
        assert len({row["measure_key"] for row in baseline_rows}) == 2
        monkeypatch.setattr(con, "create_table", capture_then_mutate)
        manifest, rows = published()
        assert captures, "mutation hook never ran"
        remaining = con.sql(
            "SELECT count(*) AS n FROM analytics.event_log WHERE event = 'purchase'"
        )
        assert con.to_pyarrow(remaining).to_pylist() == [{"n": 0}]
        assert rows == baseline_rows
        assert manifest.measures == baseline_manifest.measures
    finally:
        native.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("open_ended", [False, True])
def test_adoption_known_total_horizon_preserves_unit_population(tmp_path, open_ended):
    import warnings

    from examples._seed import seed_event_log
    from increment.query.builders import declared_binary_metrics, group_summary

    con, native, context, store = _native(definitions=_definitions(tmp_path, open_ended=open_ended))
    seed_event_log(con, n_units=200)
    adopted = None
    try:
        ref = native.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        source = adopted._src
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", UserWarning)
            for metric in source.context.metrics:
                # Unit-grain still discovers the endpoint from the dense relation.
                totals = source._reduction_query(metric, "unit")
                expected = con.to_pyarrow(
                    group_summary(totals, binary_metrics=declared_binary_metrics([metric]))
                ).to_pylist()
                assert expected
                _assert_rows_equal(expected, source._reduce(metric, "total"))
                assert source._reduce(metric, "total", population_units=frozenset()) == []
        assert any(issubclass(item.category, UserWarning) for item in recorded)
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
        con.disconnect()


def _windowed_definitions(
    tmp_path: Path, *, start: str, end: str, observation_end: str, day_boundary: str, name: str
) -> Path:
    import yaml

    path = _definitions(tmp_path)
    payload = yaml.safe_load(path.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment.update(
        start=start, end=end, observation_end=observation_end, day_boundary=day_boundary
    )
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _route_evidence(native: Any, adopted: Any, store: Any, reference: Any, context: Any) -> dict:
    """Daily rows and the lift estimate each route reports for one publication."""
    adopted_source = _artifact_source(store, reference, context)
    try:
        native_source = _native_source(native)
        native_metric = _metric(native_source)
        adopted_metric = _metric(adopted_source)
        return {
            "native_rows": native_source.day_source(metrics=(native_metric,)).moments(
                native_metric, grain="daily"
            ),
            "adopted_rows": adopted_source.day_source(metrics=(adopted_metric,)).moments(
                adopted_metric, grain="daily"
            ),
            "native_lift": lift_rows(native.run(metrics=["purchase_rate"]))[0].require_lift().value,
            "adopted_lift": lift_rows(adopted.run(metrics=["purchase_rate"]))[0]
            .require_lift()
            .value,
        }
    finally:
        adopted_source.close()


@pytest.mark.parametrize("day_boundary", ["UTC", "UTC-05:00"])
def test_aware_spellings_of_one_instant_agree_on_both_routes(
    tmp_path: Path, day_boundary: str
) -> None:
    """One declared instant is one window: rows, spine and estimates match across
    both warehouse routes and both spellings."""
    spellings = {
        "zulu": ("2025-01-16T00:00:00Z", "2025-02-10T00:00:00Z", "2025-03-15T00:00:00Z"),
        "offset": (
            "2025-01-15T19:00:00-05:00",
            "2025-02-09T19:00:00-05:00",
            "2025-03-14T19:00:00-05:00",
        ),
    }
    evidence = {}
    for label, (start, end, horizon) in spellings.items():
        definitions = _windowed_definitions(
            tmp_path,
            start=start,
            end=end,
            observation_end=horizon,
            day_boundary=day_boundary,
            name=label,
        )
        _connection, native, context, store = _native(definitions=definitions)
        reference = native.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
        try:
            evidence[label] = _route_evidence(native, adopted, store, reference, context)
        finally:
            adopted.close()
            native.close()
    for label, routes in evidence.items():
        _assert_rows_equal(routes["native_rows"], routes["adopted_rows"])
        assert routes["adopted_lift"] == pytest.approx(routes["native_lift"], rel=1e-9), label
    _assert_rows_equal(evidence["zulu"]["native_rows"], evidence["offset"]["native_rows"])
    assert evidence["zulu"]["native_lift"] == pytest.approx(
        evidence["offset"]["native_lift"], rel=1e-9
    )


def test_a_naive_utc_minus_five_experiment_publishes_and_adopts_like_from_definitions(
    tmp_path: Path,
) -> None:
    """Public compilation binds the declared local days, and an adopted analysis reads them back."""
    import json
    from datetime import date

    from increment.query.artifact_contract import artifact_source_mapping
    from increment.semantics.loader import load
    from increment.semantics.models import window_days
    from increment.sequential_source import native_observation_mapping

    definitions = _windowed_definitions(
        tmp_path,
        start="2025-01-15",
        end="2025-01-20",
        observation_end="2025-03-15",
        day_boundary="UTC-05:00",
        name="naive",
    )
    _connection, native, context, store = _native(definitions=definitions)
    reference = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        evidence = _route_evidence(native, adopted, store, reference, context)
        _assert_rows_equal(evidence["native_rows"], evidence["adopted_rows"])
        assert evidence["adopted_lift"] == pytest.approx(evidence["native_lift"], rel=1e-9)
        assert json.loads(context.canonical_json)["window_days"] == {
            "start": date(2025, 1, 15).isoformat(),
            "end": date(2025, 1, 20).isoformat(),
            "observation_horizon": date(2025, 3, 15).isoformat(),
        }
        stored = json.loads(context.canonical_json)["window_days"]
        adopted_days = window_days(adopted.experiment)
        assert {edge: day and day.isoformat() for edge, day in adopted_days.items()} == stored
        defs = load(definitions)
        experiment = defs.experiment("new_onboarding_v2")
        assert experiment is not None
        assert artifact_source_mapping(context) == native_observation_mapping(defs, experiment)
    finally:
        adopted.close()
        native.close()
