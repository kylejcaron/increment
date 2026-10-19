"""General warehouse QR parity without transferring unit-keyed evidence."""

from __future__ import annotations

from typing import cast

import ibis
import numpy as np
import pytest

from increment import Analysis
from increment.estimation.results import LiftEstimate
from increment.query.native_source import DefinitionsMomentSource
from tests.parity_harness import matrix_cases, matrix_data
from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows


def _scratch_session(connection, tmp_path):
    from increment.query.session import WarehouseSession
    from increment.semantics.loader import load

    definitions_path = tmp_path / "scratch.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("scratch_source"))
    return WarehouseSession(connection, load(definitions_path))


def _positive_diagonal(r, rhs):
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return signs[:, None] * r, signs[:, None] * rhs


@pytest.mark.parametrize("p", [1, 4, 20])
def test_duckdb_householder_factor_matches_numpy(p, tmp_path):
    from increment.query._cuped_qr import factor_design_relation

    rng = np.random.default_rng(913 + p)
    n = max(4 * p + 17, 80)
    design = rng.normal(size=(n, p))
    design /= np.max(np.abs(design), axis=0)
    outcomes = rng.normal(size=(n, 2))
    outcomes /= np.max(np.abs(outcomes), axis=0)
    columns = [f"x{index}" for index in range(p)]
    rhs_columns = [f"y{index}" for index in range(outcomes.shape[1])]
    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            **dict(zip(columns, x, strict=True)),
            **dict(zip(rhs_columns, y, strict=True)),
        }
        for index, (x, y) in enumerate(zip(design, outcomes, strict=True))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)
    try:
        with session.qr_scratch() as scratch:
            actual = factor_design_relation(
                connection.table("factor_input"),
                design_columns=tuple(columns),
                rhs_columns=tuple(rhs_columns),
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
        expected_q, expected_r = np.linalg.qr(design, mode="reduced")
        expected_rhs = expected_q.T @ outcomes
        expected_r, expected_rhs = _positive_diagonal(expected_r, expected_rhs)
        actual_r, actual_rhs = _positive_diagonal(
            np.asarray(actual.triangular_factor), np.asarray(actual.transformed_rhs)
        )
        np.testing.assert_allclose(actual_r, expected_r, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1e-10, atol=1e-12)
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_duckdb_householder_factor_handles_large_offset_small_spread(tmp_path):
    from increment.query._cuped_qr import factor_design_relation

    offsets = np.array([1.0e15 + (index % 2) * 0.125 for index in range(80)])
    design = ((offsets - 1.0e15) / 0.125).reshape(-1, 1)
    outcome = np.linspace(-1.0, 1.0, len(design))[:, None]
    rows = [
        {"unit_id": f"unit-{index:04d}", "x": float(x[0]), "y": float(y[0])}
        for index, (x, y) in enumerate(zip(design, outcome, strict=True))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)
    try:
        with session.qr_scratch() as scratch:
            actual = factor_design_relation(
                connection.table("factor_input"),
                design_columns=("x",),
                rhs_columns=("y",),
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
        expected_q, expected_r = np.linalg.qr(design, mode="reduced")
        expected_rhs = expected_q.T @ outcome
        expected_r, expected_rhs = _positive_diagonal(expected_r, expected_rhs)
        actual_r, actual_rhs = _positive_diagonal(
            np.asarray(actual.triangular_factor), np.asarray(actual.transformed_rhs)
        )
        np.testing.assert_allclose(actual_r, expected_r, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1e-10, atol=1e-12)
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_duckdb_householder_factor_uses_pivoted_r_rank_rule(tmp_path):
    from increment.errors import CodedError
    from increment.query._cuped_qr import factor_design_relation

    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            "x0": float(index),
            "x1": float(index),
            "y": float(index),
        }
        for index in range(80)
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)
    try:
        with session.qr_scratch() as scratch, pytest.raises(CodedError) as raised:
            factor_design_relation(
                connection.table("factor_input"),
                design_columns=("x0", "x1"),
                rhs_columns=("y",),
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
                covariate_names=("x0", "x1"),
            )
        assert raised.value.code == "estimation.cuped_linear.rank_deficiency"
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_qr_scratch_alternates_and_cleans_materialized_relations(tmp_path):
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}, {"x": 2.0}])
    session = _scratch_session(connection, tmp_path)
    relation = connection.table("scratch_source")
    try:
        with session.qr_scratch() as scratch:
            for _ in range(4):
                relation = scratch.materialize_next(relation.mutate(x=relation.x + 1.0))
                assert len(session.materialized_names) == 1
        assert session.materialized_names == []
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_qr_scratch_drop_failure_exposes_retained_locator(tmp_path, monkeypatch):
    from increment.errors import CodedError

    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}, {"x": 2.0}])
    session = _scratch_session(connection, tmp_path)
    original_drop = type(connection).drop_table

    def denied_drop(backend, name, *args, **kwargs):
        if str(name).startswith("_cuped_qr_"):
            raise PermissionError("simulated temporary-relation DROP denial")
        return original_drop(backend, name, *args, **kwargs)

    monkeypatch.setattr(type(connection), "drop_table", denied_drop)
    try:
        with pytest.raises(CodedError) as raised:
            with session.qr_scratch() as scratch:
                scratch.materialize_next(connection.table("scratch_source"))
        assert raised.value.code == "query.session.qr_scratch_cleanup_failed"
        assert raised.value.context["relations"]
        assert session.materialized_names
    finally:
        monkeypatch.setattr(type(connection), "drop_table", original_drop)
        session.drop_materialized()
        connection.disconnect()


def test_artifact_snapshot_qr_scratch_materializes_and_cleans_relations():
    from increment.query.session import WarehouseArtifactStore

    connection = ibis.duckdb.connect()
    connection.create_table("artifact_qr_input", obj=[{"x": 2.0}, {"x": 4.0}])
    store = WarehouseArtifactStore(connection)
    try:
        with store.qr_scratch() as scratch:
            relation = scratch.materialize_next(connection.table("artifact_qr_input"))
            assert relation.count().execute() == 2
            assert scratch.leftovers
        assert not scratch.leftovers
    finally:
        connection.disconnect()


@pytest.mark.parametrize("metric", ["revenue", "rpo"])
def test_native_mixed_cuped_uses_warehouse_qr_without_unit_transfer(metric, tmp_path, monkeypatch):
    units = matrix_data.linear_cuped_rows()
    connection = ibis.duckdb.connect()
    connection.create_table("qr_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "qr.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("qr_events"))
    warehouse = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("general warehouse QR transferred unit-keyed data")

    monkeypatch.setattr(DefinitionsMomentSource, "unit_frame", forbidden_unit_transfer)
    try:
        actual = cast(list[LiftEstimate], warehouse.run(metrics=[metric]))
        expected = cast(list[LiftEstimate], local.run(metrics=[metric]))
        actual_rows = {(row.metric, row.group_id, row.method): row for row in actual}
        expected_rows = {(row.metric, row.group_id, row.method): row for row in expected}
        assert actual_rows.keys() == expected_rows.keys()
        for key, row in actual_rows.items():
            reference = expected_rows[key]
            assert row.sampling_reason_code == reference.sampling_reason_code
            if row.sampling_reason_code is None:
                assert row.lift is not None and reference.lift is not None
                for field in ("value", "log_se", "dof", "lb", "ub"):
                    assert getattr(row.lift, field) == pytest.approx(
                        getattr(reference.lift, field), rel=1e-9, abs=1e-11
                    )
    finally:
        local.close()
        warehouse.close()
        connection.disconnect()
