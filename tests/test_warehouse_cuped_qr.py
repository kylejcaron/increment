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


def test_duckdb_qr_coefficients_and_unit_quantities_match_local_fit(tmp_path):
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.query._cuped_qr import (
        factor_design_relation,
        solve_qr_coefficients,
        warehouse_qr_unit_summary,
    )

    rng = np.random.default_rng(20261014)
    arms_order = ("control", "variant-a", "variant-b")
    arms = np.repeat(arms_order, 48)
    x = np.concatenate(
        [
            1.0e8 + np.linspace(-4.0, 4.0, 48),
            -3.0e7 + np.linspace(-19.0, 19.0, 48),
            8.0e8 + np.linspace(-27.0, 27.0, 48),
        ]
    )
    y = np.concatenate(
        [
            4.0e5 + 1.7 * x[:48] + rng.normal(0.0, 0.2, 48),
            8.0e5 + 1.7 * x[48:96] + rng.normal(0.0, 0.3, 48),
            2.0e6 + 1.7 * x[96:] + rng.normal(0.0, 0.4, 48),
        ]
    )
    local = fit_cuped_multicovariate(
        y,
        arms,
        control="control",
        arm_order=arms_order,
        covariates={"x": x},
        kinds={"x": "continuous"},
    )

    x_scale = max(
        abs(float(x[np.flatnonzero(arms == arm)[0]] - x[index]))
        for arm in arms_order
        for index in np.flatnonzero(arms == arm)
    )
    y_fit = np.empty_like(y)
    sqrt_weights = np.empty_like(y)
    design_unscaled = np.zeros((len(y), len(arms_order) + 1))
    for arm_index, arm in enumerate(arms_order):
        indices = np.flatnonzero(arms == arm)
        x_anchor = x[indices[0]]
        encoded = (x[indices] - x_anchor) / x_scale
        x_centered = encoded - encoded.mean()
        y_centered = y[indices] - y[indices].mean()
        weight = 1.0 / (len(indices) * (len(indices) - 1))
        sqrt_weight = np.sqrt(weight)
        sqrt_weights[indices] = sqrt_weight
        y_fit[indices] = y_centered
        design_unscaled[indices, arm_index] = sqrt_weight
        design_unscaled[indices, -1] = sqrt_weight * x_centered
    design_scales = np.max(np.abs(design_unscaled), axis=0)
    design = design_unscaled / design_scales
    rhs_scale = float(np.max(np.abs(sqrt_weights * y_fit)))
    rhs = sqrt_weights * y_fit / rhs_scale
    design_columns = tuple(f"d{index}" for index in range(design.shape[1]))
    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            "arm": str(arms[index]),
            "sqrt_weight": float(sqrt_weights[index]),
            "y_fit": float(y_fit[index]),
            "rhs": float(rhs[index]),
            **{
                name: float(value)
                for name, value in zip(design_columns, design[index], strict=True)
            },
        }
        for index in range(len(y))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)

    def bounded_execute(expression):
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= 1
        return result

    try:
        with session.qr_scratch() as scratch:
            factor = factor_design_relation(
                connection.table("factor_input"),
                design_columns=design_columns,
                rhs_columns=("rhs",),
                order_columns=("unit_id",),
                execute=bounded_execute,
                scratch=scratch,
                covariate_start=len(arms_order),
                covariate_names=("x",),
            )
            coefficients = solve_qr_coefficients(
                factor,
                design_scales=tuple(design_scales),
                outcome_scales=(rhs_scale,),
            )
            summary = warehouse_qr_unit_summary(
                connection.table("factor_input"),
                factor,
                design_columns=design_columns,
                outcome_columns=("y_fit",),
                sqrt_weight_column="sqrt_weight",
                design_scales=tuple(design_scales),
                coefficients=coefficients,
                execute=bounded_execute,
                scratch=scratch,
            )
            residual_sum = bounded_execute(
                summary.relation.aggregate(
                    _residual_square_sum=summary.relation["_qr_residual_0"].pow(2).sum()
                )
            ).to_pylist()[0]["_residual_square_sum"]
        np.testing.assert_allclose(
            coefficients[0][-1] / x_scale, local.theta[0], rtol=1e-9, atol=1e-11
        )
        assert summary.max_leverage == pytest.approx(local.max_leverage, rel=1e-9)
        assert summary.sum_leverage == pytest.approx(local.leverage_trace, rel=1e-9)
        assert summary.refusal_witness == ()
        expected_residual = np.empty_like(y)
        for arm in arms_order:
            indices = np.flatnonzero(arms == arm)
            expected_residual[indices] = (
                y[indices] - y[indices].mean() - local.theta[0] * (x[indices] - x[indices].mean())
            )
        assert residual_sum == pytest.approx(float(expected_residual @ expected_residual), rel=1e-9)
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_duckdb_qr_unit_summary_preserves_leverage_refusal_witness(tmp_path):
    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.query._cuped_qr import (
        factor_design_relation,
        solve_qr_coefficients,
        warehouse_qr_unit_summary,
    )

    arms_order = ("control", "treatment")
    arms = np.repeat(arms_order, 8)
    x = np.concatenate((np.arange(8, dtype=float), np.array([0, 0, 0, 0, 0, 0, 90, 100])))
    y = np.concatenate((np.linspace(2, 3, 8), np.linspace(4, 7, 8)))
    with pytest.raises(CodedError) as local_error:
        fit_cuped_multicovariate(
            y,
            arms,
            control=arms_order[0],
            arm_order=arms_order,
            covariates={"x": x},
            kinds={"x": "continuous"},
        )
    assert local_error.value.code == "estimation.cuped_linear.support_region"

    sqrt_weight = float(np.sqrt(1 / (8 * 7)))
    design_unscaled = np.zeros((len(y), len(arms_order) + 1))
    y_fit = np.empty_like(y)
    for arm_index, arm in enumerate(arms_order):
        indices = np.flatnonzero(arms == arm)
        x_centered = x[indices] - x[indices].mean()
        design_unscaled[indices, arm_index] = sqrt_weight
        design_unscaled[indices, -1] = sqrt_weight * x_centered
        y_fit[indices] = y[indices] - y[indices].mean()
    design_scales = np.max(np.abs(design_unscaled), axis=0)
    design = design_unscaled / design_scales
    rhs_scale = float(np.max(np.abs(sqrt_weight * y_fit)))
    rhs = sqrt_weight * y_fit / rhs_scale
    expected_q, _ = np.linalg.qr(design_unscaled, mode="reduced")
    expected_max = float(np.max(np.sum(expected_q**2, axis=1)))
    design_columns = tuple(f"d{index}" for index in range(design.shape[1]))
    rows = [
        {
            "unit_id": f"unit-{index:02d}",
            "sqrt_weight": sqrt_weight,
            "y_fit": float(y_fit[index]),
            "rhs": float(rhs[index]),
            **{
                name: float(value)
                for name, value in zip(design_columns, design[index], strict=True)
            },
        }
        for index in range(len(y))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)

    def bounded_execute(expression):
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= 1
        return result

    try:
        with session.qr_scratch() as scratch:
            factor = factor_design_relation(
                connection.table("factor_input"),
                design_columns=design_columns,
                rhs_columns=("rhs",),
                order_columns=("unit_id",),
                execute=bounded_execute,
                scratch=scratch,
                covariate_start=len(arms_order),
                covariate_names=("x",),
            )
            coefficients = solve_qr_coefficients(
                factor,
                design_scales=tuple(design_scales),
                outcome_scales=(rhs_scale,),
            )
            summary = warehouse_qr_unit_summary(
                connection.table("factor_input"),
                factor,
                design_columns=design_columns,
                outcome_columns=("y_fit",),
                sqrt_weight_column="sqrt_weight",
                design_scales=tuple(design_scales),
                coefficients=coefficients,
                execute=bounded_execute,
                scratch=scratch,
            )
        assert summary.max_leverage == pytest.approx(expected_max, rel=1e-9)
        assert summary.refusal_witness == (("max_leverage", summary.max_leverage, 0.45),)
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
