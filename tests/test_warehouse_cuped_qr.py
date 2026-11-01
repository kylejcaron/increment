"""General warehouse QR parity without transferring unit-keyed evidence."""

from __future__ import annotations

import math
from typing import Any, cast

import ibis
import numpy as np
import pytest

from increment import (
    Scope,
    from_definitions,
    from_frame,
    from_moments,
    from_unit_day_artifact,
)
from increment.artifacts import publish_unit_day_artifact
from increment.estimation.results import LiftEstimate
from increment.query.native_source import DefinitionsMomentSource
from increment.workflows import export_moments
from tests.analysis_factory import view_readout
from tests.parity_harness import matrix_cases, matrix_data
from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows


def _scratch_session(connection, tmp_path):
    from increment.query.session import WarehouseSession
    from increment.semantics.loader import load

    definitions_path = tmp_path / "scratch.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("scratch_source"))
    return WarehouseSession(connection, load(definitions_path))


def _guard_no_unit_keyed_transfers(monkeypatch):
    from increment.query.artifact_contract import ImmutableArtifactSnapshot
    from increment.query.session import WarehouseSession

    def guarded(original):
        def execute(self, expression, *args, **kwargs):
            result = original(self, expression, *args, **kwargs)
            names = getattr(getattr(result, "schema", None), "names", ())
            if "unit_id" in names and getattr(result, "num_rows", 0):
                pytest.fail("warehouse QR transferred a unit-keyed row")
            return result

        return execute

    monkeypatch.setattr(WarehouseSession, "fetch_arrow", guarded(WarehouseSession.fetch_arrow))
    monkeypatch.setattr(
        ImmutableArtifactSnapshot, "execute", guarded(ImmutableArtifactSnapshot.execute)
    )


def _assert_no_transferred_extreme(expression):
    import ibis
    import sqlglot

    sql = ibis.to_sql(expression, dialect="duckdb")
    statement = sqlglot.parse_one(sql, read="duckdb")
    assert statement.args.get("limit") is None
    assert statement.args.get("order") is None
    for projection in statement.expressions:
        selected = projection.this if isinstance(projection, sqlglot.exp.Alias) else projection
        if type(selected).__name__ in {"AnyValue", "First", "Max", "Median", "Min"}:
            assert isinstance(projection, sqlglot.exp.Alias) and (
                projection.alias in {"_qr_max_leverage", "_qr_mean_exponent"}
                or projection.alias.startswith(("scale_", "zero_"))
            ), sql


def _positive_diagonal(r, rhs):
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return signs[:, None] * r, signs[:, None] * rhs


@pytest.mark.integration
def test_qr_scratch_reuses_two_temp_relations_without_intermediate_drops(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1}])
    connection.create_table("factor_input", obj=[{"x": 1}])
    session = _scratch_session(connection, tmp_path)
    drop_calls = []
    drop_statements = []
    original_drop = connection.drop_table
    original_raw_sql = connection.raw_sql

    def tracked_drop(*args, **kwargs):
        drop_calls.append(args[0])
        return original_drop(*args, **kwargs)

    def tracked_raw_sql(sql, *args, **kwargs):
        if "DROP TABLE" in str(sql).upper():
            drop_statements.append(str(sql))
        return original_raw_sql(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "drop_table", tracked_drop)
    monkeypatch.setattr(connection, "raw_sql", tracked_raw_sql)
    try:
        with session.qr_scratch() as scratch:
            first = scratch.materialize_next(connection.table("factor_input"))
            second = scratch.materialize_next(first.mutate(x=first.x + 1))
            third = scratch.materialize_next(second.mutate(x=second.x + 1))
            assert third.execute().to_dict("records") == [{"x": 3}]
        assert drop_calls == []
        assert len(drop_statements) == 1
        assert len(drop_statements[0].split(";")) == 2
    finally:
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("p", [1, 4, 20, pytest.param(50, marks=pytest.mark.slow)])
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


@pytest.mark.integration
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


@pytest.mark.integration
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
            "group_id": str(arms[index]),
            "sqrt_weight": float(sqrt_weights[index]),
            "arm": str(arms[index]),
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
                row_identity_columns=("unit_id",),
                design_scales=tuple(design_scales),
                arms=arms_order,
                outcome_arm_exponents=((0, 0, 0),),
                centered_outcome_means=((0.0, 0.0, 0.0),),
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


@pytest.mark.integration
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
            "unit_id": f"unit-{index:04d}",
            "group_id": str(arms[index]),
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
                row_identity_columns=("unit_id",),
                design_scales=tuple(design_scales),
                arms=arms_order,
                outcome_arm_exponents=((0, 0),),
                centered_outcome_means=((0.0, 0.0),),
                coefficients=coefficients,
                execute=bounded_execute,
                scratch=scratch,
            )
        assert summary.max_leverage == pytest.approx(expected_max, rel=1e-9)
        assert summary.refusal_witness == (("max_leverage", summary.max_leverage, 0.45),)
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "snowflake", "bigquery"])
def test_qr_unit_summary_statement_size_at_supported_p(dialect, tmp_path, monkeypatch):
    from increment.query._cuped_qr import QRFactor, warehouse_qr_unit_summary

    width = 203
    design_columns = tuple(f"d{index}" for index in range(width))
    row = {
        "unit_id": "one",
        "sqrt_weight": 1.0,
        "y_fit": 1.0,
        "group_id": "arm",
        **{name: float(index + 1) / width for index, name in enumerate(design_columns)},
    }
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=[row])
    session = _scratch_session(connection, tmp_path)
    statements = []
    original_create = type(connection).create_table

    def capture_statement(backend, name, expression, *args, **kwargs):
        if str(name).startswith("_cuped_qr_"):
            statements.append(str(ibis.to_sql(expression, dialect=dialect)))
        return original_create(backend, name, expression, *args, **kwargs)

    monkeypatch.setattr(type(connection), "create_table", capture_statement)
    try:
        factor = QRFactor(
            triangular_factor=tuple(
                tuple(1.0 if row_index == column_index else 0.0 for column_index in range(width))
                for row_index in range(width)
            ),
            transformed_rhs=tuple((0.0,) for _ in range(width)),
        )
        with session.qr_scratch() as scratch:
            warehouse_qr_unit_summary(
                connection.table("factor_input"),
                factor,
                design_columns=design_columns,
                outcome_columns=("y_fit",),
                sqrt_weight_column="sqrt_weight",
                row_identity_columns=("unit_id",),
                design_scales=(1.0,) * width,
                arms=("arm",),
                outcome_arm_exponents=((0,),),
                centered_outcome_means=((0.0,),),
                coefficients=((0.0,) * width,),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
        assert len(statements) >= 3
        assert max(map(len, statements)) < 100_000
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_bm_feature_batches_respect_postgres_column_budget_at_p203():
    from increment.query._cuped_qr import _bm_job_batches

    width = 23
    base_columns = tuple(f"base_{index}" for index in range(250))
    jobs = tuple(
        (
            f"t2_{job}",
            f"score_{job}",
            tuple(f"feature_{job}_{index}" for index in range(2 * width)),
        )
        for job in range(36)
    )
    batches = _bm_job_batches(jobs, width=width, base_columns=base_columns)

    assert len(batches) > 1
    assert tuple(job for batch in batches for job in batch) == jobs
    assert all(
        len(base_columns) + sum(1 + len(features) for _, _, features in batch) <= 1200
        for batch in batches
    )


@pytest.mark.integration
@pytest.mark.parametrize("dialect", ["bigquery", "snowflake"])
def test_qr_factor_aggregates_compile_without_filter(dialect, tmp_path):
    from increment.query._cuped_qr import factor_design_relation

    rows = [
        {"unit_id": f"u{index}", "x0": float(index + 1), "x1": float(index % 2), "y": float(index)}
        for index in range(6)
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("filter_compile_input", obj=rows)
    session = _scratch_session(connection, tmp_path)
    compiled = []

    def execute(expression):
        compiled.append(ibis.to_sql(expression, dialect=dialect))
        return connection.to_pyarrow(expression)

    try:
        with session.qr_scratch() as scratch:
            factor_design_relation(
                connection.table("filter_compile_input"),
                design_columns=("x0", "x1"),
                rhs_columns=("y",),
                order_columns=("unit_id",),
                execute=execute,
                scratch=scratch,
            )
        assert compiled
        assert all(
            "FILTER(" not in sql.upper() and "FILTER (" not in sql.upper() for sql in compiled
        )
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
def test_warehouse_inference_reductions_match_dense_oracle_and_local_fit(  # noqa: PLR0915
    tmp_path,
):
    from increment.estimation.cuped import (
        FitState,
        ProjectionSpec,
        fit_cuped_multicovariate,
    )
    from increment.estimation.cuped.inference import finalize_fit
    from increment.query._cuped_qr import (
        factor_design_relation,
        solve_qr_coefficients,
        warehouse_inference_summary,
        warehouse_qr_unit_summary,
    )

    arms_order = ("control", "treatment")
    arm_values = np.repeat(arms_order, 32)
    x = np.concatenate(
        (1.0e8 + np.linspace(-5.0, 6.0, 32), 1.0e8 + 30.0 + np.linspace(-12.0, 8.0, 32))
    )
    centered_by_arm = np.concatenate(
        (
            np.linspace(-5.0, 6.0, 32) - np.mean(np.linspace(-5.0, 6.0, 32)),
            np.linspace(-12.0, 8.0, 32) - np.mean(np.linspace(-12.0, 8.0, 32)),
        )
    )
    y = np.concatenate(
        (
            10.0 + 0.35 * centered_by_arm[:32] + np.linspace(-0.4, 0.5, 32),
            12.0 + 0.35 * centered_by_arm[32:] + np.linspace(0.3, -0.6, 32),
        )
    )
    local = fit_cuped_multicovariate(
        y,
        arm_values,
        control="control",
        arm_order=arms_order,
        covariates={"x": x},
        kinds={"x": "continuous"},
    )
    indices = tuple(np.flatnonzero(arm_values == arm) for arm in arms_order)
    offsets = tuple(float(x[index[0]]) for index in indices)
    covariate_scale = max(
        abs(float(x[row] - offsets[arm]))
        for arm, arm_indices in enumerate(indices)
        for row in arm_indices
    )
    encoded = np.asarray(
        [
            (float(x[row]) - offsets[arm]) / covariate_scale
            for arm, rows in enumerate(indices)
            for row in rows
        ]
    )
    centered = encoded.copy()
    centered_means = []
    for arm_indices in indices:
        centered_means.append(float(encoded[arm_indices].mean()))
        centered[arm_indices] -= centered_means[-1]
    outcome_means = tuple(float(y[index].mean()) for index in indices)
    y_fit = np.asarray(
        [float(y[row]) - outcome_means[arm] for arm, rows in enumerate(indices) for row in rows]
    )
    outcome_arm_exponents = (
        tuple(math.frexp(float(np.linalg.norm(y_fit[index])))[1] for index in indices),
    )
    centered_outcome_means = ((0.0, 0.0),)
    sqrt_weight = 1.0 / np.sqrt(len(indices[0]) * (len(indices[0]) - 1))
    sqrt_weights = np.full(len(y), sqrt_weight)
    design_unscaled = np.zeros((len(y), len(arms_order) + 1))
    for arm, arm_indices in enumerate(indices):
        design_unscaled[arm_indices, arm] = sqrt_weight
        design_unscaled[arm_indices, -1] = sqrt_weight * centered[arm_indices]
    design_scales = np.max(np.abs(design_unscaled), axis=0)
    design = design_unscaled / design_scales
    rhs_scale = float(np.max(np.abs(sqrt_weights * y_fit)))
    rhs = sqrt_weights * y_fit / rhs_scale
    design_columns = tuple(f"d{index}" for index in range(design.shape[1]))
    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            "group_id": str(arm_values[index]),
            "arm_index": int(arm_values[index] == "treatment"),
            "sqrt_weight": float(sqrt_weights[index]),
            "x_centered": float(centered[index]),
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

    def guarded_execute(expression):
        assert type(expression.op()).__name__ in {"Aggregate", "Project", "SQLQueryResult"}
        _assert_no_transferred_extreme(expression)
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= len(arms_order)
        assert "unit_id" not in result.column_names
        return result

    try:
        with session.qr_scratch() as scratch:
            factor = factor_design_relation(
                connection.table("factor_input"),
                design_columns=design_columns,
                rhs_columns=("rhs",),
                order_columns=("unit_id",),
                execute=guarded_execute,
                scratch=scratch,
                covariate_start=len(arms_order),
                covariate_names=("x",),
            )
            coefficients = solve_qr_coefficients(
                factor,
                design_scales=tuple(design_scales),
                outcome_scales=(rhs_scale,),
            )
            unit_summary = warehouse_qr_unit_summary(
                connection.table("factor_input"),
                factor,
                design_columns=design_columns,
                outcome_columns=("y_fit",),
                sqrt_weight_column="sqrt_weight",
                row_identity_columns=("unit_id",),
                design_scales=tuple(design_scales),
                arms=arms_order,
                outcome_arm_exponents=outcome_arm_exponents,
                centered_outcome_means=centered_outcome_means,
                coefficients=coefficients,
                execute=guarded_execute,
                scratch=scratch,
            )
            fit_identity = ("warehouse-qr-inference-oracle",)
            state = FitState(
                identity=fit_identity,
                eligible_arms=arms_order,
                withheld_arms=(),
                refusal_contexts=(),
                counts=(len(indices[0]), len(indices[1])),
                layout=local.layout,
                offsets=(offsets,),
                scales=(covariate_scale,),
                centered_means=((centered_means[0],), (centered_means[1],)),
                anchor=local.anchor,
                triangular_factor=factor.triangular_factor,
                transformed_rhs=(tuple(row[0] for row in factor.transformed_rhs),),
                slopes=(coefficients[0][len(arms_order) :],),
                adjusted_means=local.adjusted_means,
                theta=local.theta,
                n_min=len(indices[0]),
                p=1,
                is_ratio=False,
                control="control",
            )
            treatment_mean, control_mean = local.adjusted_means[1], local.adjusted_means[0]
            relative_reference = min(treatment_mean, control_mean)
            projection_specs = (
                ProjectionSpec(
                    fit_identity + ("absolute", "treatment", "control", ""),
                    "absolute",
                    (-1.0, 1.0),
                    1.0,
                    1.0,
                    0,
                    "treatment",
                    "control",
                ),
                ProjectionSpec(
                    fit_identity + ("absolute", "control", "treatment", ""),
                    "absolute",
                    (1.0, -1.0),
                    1.0,
                    1.0,
                    0,
                    "control",
                    "treatment",
                ),
                ProjectionSpec(
                    fit_identity + ("relative", "treatment", "control", ""),
                    "relative",
                    (
                        -relative_reference / control_mean,
                        relative_reference / treatment_mean,
                    ),
                    1.0,
                    relative_reference,
                    0,
                    "treatment",
                    "control",
                ),
                ProjectionSpec(
                    fit_identity + ("relative", "control", "treatment", ""),
                    "relative",
                    (
                        relative_reference / control_mean,
                        -relative_reference / treatment_mean,
                    ),
                    1.0,
                    relative_reference,
                    0,
                    "control",
                    "treatment",
                ),
            )
            projections = (
                local.absolute("treatment"),
                local.absolute("control", control="treatment"),
                local.relative("treatment"),
                local.relative("control", control="treatment"),
            )
            summary = warehouse_inference_summary(
                state,
                unit_summary,
                factor,
                design_scales=tuple(design_scales),
                sqrt_weight_column="sqrt_weight",
                covariate_columns=("x_centered",),
                projection_specs=projection_specs,
                projection_estimates=tuple(item.estimate for item in projections),
                outcome_arm_exponents=outcome_arm_exponents,
                execute=guarded_execute,
                scratch=scratch,
            )
            finalized = finalize_fit(state, summary)

        q_expected, r_expected = np.linalg.qr(design, mode="reduced")
        signs = np.sign(np.diag(np.asarray(factor.triangular_factor)) / np.diag(r_expected))
        q_expected *= signs[None, :]
        h_expected = np.sum(q_expected**2, axis=1)
        weighted_q = sqrt_weights[:, None] * q_expected
        np.testing.assert_allclose(summary.u, weighted_q.T @ weighted_q, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            summary.adjusted_component_covariance,
            local.covariance,
            rtol=1e-9,
            atol=1e-11,
        )
        for index, (spec, projection) in enumerate(zip(projection_specs, projections, strict=True)):
            assert summary.scaled_projected_score_squares[index][5] * math.sqrt(
                summary.scaled_projected_score_squares[index][6]
            ) == pytest.approx(projection.se, rel=1e-9, abs=1e-11)
            jacobian = np.asarray(spec.jacobian)
            contrast = np.concatenate(
                (
                    jacobian,
                    -np.asarray(
                        [
                            sum(
                                jacobian[arm]
                                * (
                                    centered_means[arm]
                                    + offsets[arm] / covariate_scale
                                    - local.anchor[0] / covariate_scale
                                )
                                for arm in range(len(arms_order))
                            )
                        ]
                    ),
                )
            )
            common_scale = float(np.max(np.abs(contrast)))
            coefficient = np.linalg.solve(
                np.asarray(factor.triangular_factor).T,
                (contrast / common_scale) / design_scales,
            )
            t = sqrt_weights * (q_expected @ coefficient) / np.sqrt(1.0 - h_expected)
            feature = np.column_stack(
                (q_expected * (sqrt_weights * t)[:, None], q_expected * (t / sqrt_weights)[:, None])
            )
            expected_s2 = float(t @ t)
            expected_s4 = float((t * t) @ (t * t))
            expected_g = feature.T @ feature
            expected_g2 = feature.T @ (feature * (t * t)[:, None])
            assert summary.s2[index][0] == pytest.approx(expected_s2, rel=1e-9, abs=1e-11)
            assert summary.s4[index][0] == pytest.approx(expected_s4, rel=1e-9, abs=1e-11)
            np.testing.assert_allclose(summary.g[index][0], expected_g, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(summary.g2[index][0], expected_g2, rtol=1e-9, atol=1e-11)
            result = (
                finalized.absolute("treatment")
                if index == 0
                else finalized.absolute("control", control="treatment")
                if index == 1
                else finalized.relative("treatment")
                if index == 2
                else finalized.relative("control", control="treatment")
            )
            assert result.estimate == pytest.approx(projection.estimate, rel=1e-9, abs=1e-11)
            assert result.se == pytest.approx(projection.se, rel=1e-9, abs=1e-11)
            assert result.dof == pytest.approx(projection.dof, rel=1e-9, abs=1e-11)
            assert result.lower == pytest.approx(projection.lower, rel=1e-9, abs=1e-11)
            assert result.upper == pytest.approx(projection.upper, rel=1e-9, abs=1e-11)
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.integration
def test_warehouse_ratio_bm_reductions_match_local_components(tmp_path):  # noqa: PLR0915
    from increment.estimation.cuped import FitState, ProjectionSpec, fit_cuped_multicovariate
    from increment.query._cuped_qr import (
        factor_design_relation,
        solve_qr_coefficients,
        warehouse_inference_summary,
        warehouse_qr_unit_summary,
    )

    arms_order = ("control", "treatment")
    arm_values = np.repeat(arms_order, 30)
    raw_x = np.concatenate((np.linspace(-4.0, 4.0, 30), 10.0 + np.linspace(-7.0, 7.0, 30)))
    x_centered_by_arm = np.concatenate(
        (
            np.linspace(-4.0, 4.0, 30) - np.mean(np.linspace(-4.0, 4.0, 30)),
            np.linspace(-7.0, 7.0, 30) - np.mean(np.linspace(-7.0, 7.0, 30)),
        )
    )
    numerator = np.concatenate(
        (
            8.0 + 0.25 * x_centered_by_arm[:30] + np.linspace(-0.2, 0.3, 30),
            10.0 + 0.25 * x_centered_by_arm[30:] + np.linspace(0.25, -0.15, 30),
        )
    )
    denominator = np.concatenate(
        (
            2.0 + 0.04 * x_centered_by_arm[:30] + np.linspace(-0.04, 0.05, 30),
            2.5 + 0.04 * x_centered_by_arm[30:] + np.linspace(0.03, -0.05, 30),
        )
    )
    local = fit_cuped_multicovariate(
        None,
        arm_values,
        control="control",
        arm_order=arms_order,
        numerator=numerator,
        denominator=denominator,
        covariates={"x": raw_x},
        kinds={"x": "continuous"},
    )
    indices = tuple(np.flatnonzero(arm_values == arm) for arm in arms_order)
    offsets = tuple(float(raw_x[index[0]]) for index in indices)
    covariate_scale = max(
        abs(float(raw_x[row] - offsets[arm]))
        for arm, arm_indices in enumerate(indices)
        for row in arm_indices
    )
    encoded = np.asarray(
        [
            (float(raw_x[row]) - offsets[arm]) / covariate_scale
            for arm, arm_rows in enumerate(indices)
            for row in arm_rows
        ]
    )
    centered = encoded.copy()
    centered_means = []
    for arm_indices in indices:
        centered_means.append(float(encoded[arm_indices].mean()))
        centered[arm_indices] -= centered_means[-1]
    outcome_arrays = (numerator, denominator)
    outcome_centers = tuple(
        np.asarray(
            [
                float(values[row]) - float(values[indices[arm]].mean())
                for arm, arm_rows in enumerate(indices)
                for row in arm_rows
            ]
        )
        for values in outcome_arrays
    )
    outcome_arm_exponents = tuple(
        tuple(math.frexp(float(np.linalg.norm(values[index])))[1] for index in indices)
        for values in outcome_centers
    )
    centered_outcome_means = tuple(tuple(0.0 for _ in arms_order) for _ in outcome_centers)
    sqrt_weight = 1.0 / np.sqrt(len(indices[0]) * (len(indices[0]) - 1))
    sqrt_weights = np.full(len(arm_values), sqrt_weight)
    design_unscaled = np.zeros((len(arm_values), len(arms_order) + 1))
    for arm, arm_indices in enumerate(indices):
        design_unscaled[arm_indices, arm] = sqrt_weight
        design_unscaled[arm_indices, -1] = sqrt_weight * centered[arm_indices]
    design_scales = np.max(np.abs(design_unscaled), axis=0)
    design = design_unscaled / design_scales
    rhs_unscaled = np.column_stack(outcome_centers) * sqrt_weights[:, None]
    outcome_scales = np.max(np.abs(rhs_unscaled), axis=0)
    rhs = rhs_unscaled / outcome_scales
    design_columns = tuple(f"d{index}" for index in range(design.shape[1]))
    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            "group_id": str(arm_values[index]),
            "sqrt_weight": float(sqrt_weights[index]),
            "x_centered": float(centered[index]),
            "num_fit": float(outcome_centers[0][index]),
            "den_fit": float(outcome_centers[1][index]),
            **{f"rhs{family}": float(rhs[index, family]) for family in range(2)},
            **{
                name: float(value)
                for name, value in zip(design_columns, design[index], strict=True)
            },
        }
        for index in range(len(arm_values))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)

    def guarded_execute(expression):
        assert type(expression.op()).__name__ in {"Aggregate", "Project", "SQLQueryResult"}
        _assert_no_transferred_extreme(expression)
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= len(arms_order)
        assert "unit_id" not in result.column_names
        return result

    try:
        with session.qr_scratch() as scratch:
            factor = factor_design_relation(
                connection.table("factor_input"),
                design_columns=design_columns,
                rhs_columns=("rhs0", "rhs1"),
                order_columns=("unit_id",),
                execute=guarded_execute,
                scratch=scratch,
                covariate_start=len(arms_order),
                covariate_names=("x",),
            )
            coefficients = solve_qr_coefficients(
                factor,
                design_scales=tuple(design_scales),
                outcome_scales=tuple(outcome_scales),
            )
            unit_summary = warehouse_qr_unit_summary(
                connection.table("factor_input"),
                factor,
                design_columns=design_columns,
                outcome_columns=("num_fit", "den_fit"),
                sqrt_weight_column="sqrt_weight",
                row_identity_columns=("unit_id",),
                design_scales=tuple(design_scales),
                arms=arms_order,
                outcome_arm_exponents=outcome_arm_exponents,
                centered_outcome_means=centered_outcome_means,
                coefficients=coefficients,
                execute=guarded_execute,
                scratch=scratch,
            )
            fit_identity = ("warehouse-qr-ratio-inference-oracle",)
            state = FitState(
                identity=fit_identity,
                eligible_arms=arms_order,
                withheld_arms=(),
                refusal_contexts=(),
                counts=(30, 30),
                layout=local.layout,
                offsets=(offsets,),
                scales=(covariate_scale,),
                centered_means=((centered_means[0],), (centered_means[1],)),
                anchor=local.anchor,
                triangular_factor=factor.triangular_factor,
                transformed_rhs=tuple(
                    tuple(row[family] for row in factor.transformed_rhs) for family in range(2)
                ),
                slopes=tuple(coefficients[family][len(arms_order) :] for family in range(2)),
                adjusted_means=local.adjusted_means,
                theta=local.theta,
                n_min=30,
                p=1,
                is_ratio=True,
                control="control",
            )
            means = np.asarray(local.adjusted_means).reshape((2, 2))
            control_num, treatment_num = means[0]
            control_den, treatment_den = means[1]
            treatment_ratio = treatment_num / treatment_den
            control_ratio = control_num / control_den
            log_jacobian = (
                -1.0 / control_num,
                1.0 / treatment_num,
                1.0 / control_den,
                -1.0 / treatment_den,
            )
            absolute_jacobian = (
                -1.0 / control_den,
                1.0 / treatment_den,
                control_ratio / control_den,
                -treatment_ratio / treatment_den,
            )
            specs = (
                ProjectionSpec(
                    fit_identity + ("ratio", "treatment", "control", "log"),
                    "ratio",
                    log_jacobian,
                    1.0,
                    1.0,
                    0,
                    "treatment",
                    "control",
                ),
                ProjectionSpec(
                    fit_identity + ("ratio", "treatment", "control", "absolute"),
                    "ratio",
                    absolute_jacobian,
                    1.0,
                    1.0,
                    0,
                    "treatment",
                    "control",
                ),
            )
            projections = (
                local.ratio("treatment", scale="log"),
                local.ratio("treatment", scale="absolute"),
            )
            summary = warehouse_inference_summary(
                state,
                unit_summary,
                factor,
                design_scales=tuple(design_scales),
                sqrt_weight_column="sqrt_weight",
                covariate_columns=("x_centered",),
                projection_specs=specs,
                projection_estimates=tuple(item.estimate for item in projections),
                outcome_arm_exponents=outcome_arm_exponents,
                execute=guarded_execute,
                scratch=scratch,
            )

        q_expected, r_expected = np.linalg.qr(design, mode="reduced")
        signs = np.sign(np.diag(np.asarray(factor.triangular_factor)) / np.diag(r_expected))
        q_expected *= signs[None, :]
        h_expected = np.sum(q_expected**2, axis=1)
        weighted_q = sqrt_weights[:, None] * q_expected
        np.testing.assert_allclose(summary.u, weighted_q.T @ weighted_q, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            summary.adjusted_component_covariance, local.covariance, rtol=1e-9, atol=1e-11
        )
        for projection_index, (spec, projection) in enumerate(zip(specs, projections, strict=True)):
            np.testing.assert_allclose(
                summary.scaled_projected_score_squares[projection_index][5]
                * math.sqrt(summary.scaled_projected_score_squares[projection_index][6]),
                projection.se,
                rtol=1e-9,
                atol=1e-11,
            )
            arm_jacobians = np.asarray(spec.jacobian).reshape((2, len(arms_order)))
            contrasts = []
            for family in range(2):
                contrast_x = sum(
                    arm_jacobians[family, arm]
                    * (
                        centered_means[arm]
                        + offsets[arm] / covariate_scale
                        - local.anchor[0] / covariate_scale
                    )
                    for arm in range(len(arms_order))
                )
                contrasts.append(np.concatenate((arm_jacobians[family], np.asarray([-contrast_x]))))
            common_scale = max(float(np.max(np.abs(value))) for value in contrasts)
            expected_components = []
            for contrast in contrasts:
                coefficient = np.linalg.solve(
                    np.asarray(factor.triangular_factor).T,
                    (contrast / common_scale) / design_scales,
                )
                t = sqrt_weights * (q_expected @ coefficient) / np.sqrt(1.0 - h_expected)
                feature = np.column_stack(
                    (
                        q_expected * (sqrt_weights * t)[:, None],
                        q_expected * (t / sqrt_weights)[:, None],
                    )
                )
                expected_components.append(
                    (
                        float(t @ t),
                        float((t * t) @ (t * t)),
                        feature.T @ feature,
                        feature.T @ (feature * (t * t)[:, None]),
                    )
                )
            for family, (s2, s4, gram, weighted_gram) in enumerate(expected_components):
                assert summary.s2[projection_index][family] == pytest.approx(
                    s2, rel=1e-9, abs=1e-11
                )
                assert summary.s4[projection_index][family] == pytest.approx(
                    s4, rel=1e-9, abs=1e-11
                )
                np.testing.assert_allclose(
                    summary.g[projection_index][family], gram, rtol=1e-9, atol=1e-11
                )
                np.testing.assert_allclose(
                    summary.g2[projection_index][family], weighted_gram, rtol=1e-9, atol=1e-11
                )
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("perturbation", [0.0, 1.0e-12])
def test_duckdb_householder_factor_uses_pivoted_r_rank_rule(perturbation, tmp_path):
    from increment.errors import CodedError
    from increment.query._cuped_qr import factor_design_relation

    rows = [
        {
            "unit_id": f"unit-{index:04d}",
            "x0": float(index),
            "x1": float(index) + perturbation * float(np.sin(index)),
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


@pytest.mark.integration
def test_qr_scratch_pins_input_while_alternating_work_relations(tmp_path):
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}, {"x": 2.0}])
    session = _scratch_session(connection, tmp_path)
    relation = connection.table("scratch_source")
    try:
        with session.qr_scratch() as scratch:
            pinned = scratch.materialize_input(relation)
            relation = pinned
            assert len(session.materialized_names) == 1
            for index in range(4):
                relation = scratch.materialize_next(relation.mutate(x=relation.x + 1.0))
                assert len(session.materialized_names) == min(index + 2, 3)
                assert connection.to_pyarrow(pinned).num_rows == 2
            assert len(session.materialized_names) == 3
        assert session.materialized_names == []
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
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


@pytest.mark.integration
def test_qr_scratch_create_permission_failure_leaves_no_work_relation(tmp_path, monkeypatch):
    from increment.errors import CodedError

    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    original_create = type(connection).create_table

    def denied_create(backend, name, *args, **kwargs):
        if str(name).startswith("_cuped_qr_"):
            raise PermissionError("simulated temporary-relation CREATE denial")
        return original_create(backend, name, *args, **kwargs)

    monkeypatch.setattr(type(connection), "create_table", denied_create)
    try:
        with pytest.raises(CodedError) as raised:
            with session.qr_scratch() as scratch:
                scratch.materialize_next(connection.table("scratch_source"))
        assert raised.value.code == "query.session.snapshot.materialization_failed"
        assert session.materialized_names == []
    finally:
        monkeypatch.setattr(type(connection), "create_table", original_create)
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
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


@pytest.mark.integration
def test_warehouse_qr_encoder_matches_local_layout(tmp_path):
    import ibis

    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.frame import CupedCovariate
    from increment.query._cuped_qr import (
        encode_qr_covariates,
        factor_design_relation,
        prepare_qr_factor_input,
    )

    rows = [
        {
            "unit_id": f"c{index}",
            "group_id": "control",
            "y": float(index),
            "x": None if index % 17 == 0 else float((index * 7) % 13),
            "x_copy": None if index % 17 == 0 else float((index * 14) % 13),
            "kind": ("a", "b")[index % 2],
        }
        for index in range(200)
    ] + [
        {
            "unit_id": f"t{index}",
            "group_id": "treatment",
            "y": float(index + 1),
            "x": None if index % 17 == 0 else float((index * 11) % 13),
            "x_copy": None if index % 17 == 0 else float((index * 22) % 13),
            "kind": ("a", "b")[index % 2],
        }
        for index in range(200)
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("qr_encode_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    declarations = (
        CupedCovariate(name="x", kind="continuous", missing="impute-indicator"),
        CupedCovariate(name="x_copy", kind="continuous", missing="impute-indicator"),
        CupedCovariate(name="kind", kind="categorical"),
    )
    local = fit_cuped_multicovariate(
        [row["y"] for row in rows],
        [row["group_id"] for row in rows],
        control="control",
        arm_order=("control", "treatment"),
        covariates={
            "x": [row["x"] for row in rows],
            "x_copy": [row["x_copy"] for row in rows],
            "kind": [row["kind"] for row in rows],
        },
        kinds={"x": "continuous", "x_copy": "continuous", "kind": "categorical"},
        missing={"x": "impute-indicator", "x_copy": "impute-indicator"},
    )
    try:
        with session.qr_scratch() as scratch:
            encoded = encode_qr_covariates(
                connection.table("qr_encode_input"),
                declarations,
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
            )
            assert encoded.layout == local.layout
            assert encoded.layout.shared_indicators == (("x", "x_copy"),)
            np.testing.assert_allclose(
                encoded.offsets,
                (
                    (
                        math.fsum(
                            float(row["x"] or 0.0) for row in rows if row["group_id"] == "control"
                        )
                        / 200,
                        math.fsum(
                            float(row["x"] or 0.0) for row in rows if row["group_id"] == "treatment"
                        )
                        / 200,
                    ),
                    (
                        math.fsum(
                            float(row["x_copy"] or 0.0)
                            for row in rows
                            if row["group_id"] == "control"
                        )
                        / 200,
                        math.fsum(
                            float(row["x_copy"] or 0.0)
                            for row in rows
                            if row["group_id"] == "treatment"
                        )
                        / 200,
                    ),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ),
            )
            assert all(math.frexp(scale)[0] == 0.5 for scale in encoded.scales)
            prepared = prepare_qr_factor_input(
                encoded,
                outcome_columns=("y",),
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
            assert prepared.counts == (200, 200)
            assert len(prepared.design_columns) == len(local.layout.encoded_columns) + 2
            factor = factor_design_relation(
                prepared.relation,
                design_columns=prepared.design_columns,
                rhs_columns=prepared.rhs_columns,
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
                covariate_start=2,
                covariate_names=local.layout.encoded_columns,
            )
            assert len(factor.triangular_factor) == len(prepared.design_columns)
    finally:
        connection.disconnect()


def _assert_fitted_evidence_rejections(exported, metric_spec, method, design):
    import json

    from increment._canonical import canonical_json_bytes
    from increment.errors import CodedError

    with pytest.raises(CodedError) as corrupted:
        from_moments(
            [
                {**row, "cuped_evidence": "{"}
                if row.get("record_kind") == "cuped_evidence"
                else row
                for row in exported
            ],
            metrics=[metric_spec],
            design=design,
        )
    assert corrupted.value.code == "moments.cuped_evidence.invalid"

    def rewrite_payload(change):
        rewritten = []
        for row in exported:
            if row.get("record_kind") == "cuped_evidence":
                payload = json.loads(row["cuped_evidence"])
                if payload["kind"] == "fitted_multicovariate":
                    change(payload)
                    row = {
                        **row,
                        "cuped_evidence": canonical_json_bytes(payload).decode(),
                    }
            rewritten.append(row)
        return rewritten

    for change in (
        lambda payload: payload.update(source_sha256="0" * 64),
        lambda payload: payload.update(population="triggered"),
        lambda payload: payload["evidence"]["inference_summary"]["projection_specs"].pop(),
    ):
        with pytest.raises(CodedError) as malformed:
            from_moments(rewrite_payload(change), metrics=[metric_spec], design=design)
        assert malformed.value.code == "moments.cuped_evidence.invalid"

    envelope = next(row for row in exported if row.get("record_kind") == "cuped_evidence")
    with pytest.raises(CodedError) as duplicate:
        from_moments([*exported, envelope], metrics=[metric_spec], design=design)
    assert duplicate.value.code in {
        "moments.cuped_evidence.duplicate",
        "moments.cuped_evidence.invalid",
    }
    covariates = tuple(method.covariates or ())
    changed_method = method.model_copy(
        update={
            "covariates": (
                covariates[0].model_copy(update={"name": "changed"}),
                *covariates[1:],
            )
        }
    )
    with pytest.raises(CodedError):
        from_moments(
            exported,
            metrics=[metric_spec.model_copy(update={"decision_method": changed_method})],
            design=design,
        )

    scoped_replay = None
    try:
        scoped_replay = from_moments(
            exported,
            metrics=[metric_spec],
            design=design.model_copy(update={"control_group": "treatment"}),
        )
        with pytest.raises(CodedError):
            view_readout(*scoped_replay, metrics=[metric_spec.name])
    finally:
        if scoped_replay is not None:
            scoped_replay[1].close()


def _assert_fit_coordinates_equal(left_rows, right_rows):
    import json

    from increment._cuped_evidence import parse_cuped_evidence

    def state(rows):
        envelope = next(
            row["cuped_evidence"]
            for row in rows
            if row.get("record_kind") == "cuped_evidence"
            and json.loads(row["cuped_evidence"])["kind"] == "fitted_multicovariate"
        )
        parsed = parse_cuped_evidence(envelope)
        assert parsed.fit_state is not None
        return parsed.fit_state

    left = state(left_rows)
    right = state(right_rows)
    assert left.layout == right.layout
    for field in (
        "offsets",
        "scales",
        "centered_means",
        "anchor",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(left, field), dtype=float),
            np.asarray(getattr(right, field), dtype=float),
            rtol=1e-9,
            atol=1e-11,
        )


# Keep the 34-column, three-batch ratio path covered; CI needs a longer timeout for DuckDB.
@pytest.mark.timeout(600)
@pytest.mark.integration
@pytest.mark.slow
def test_native_three_arm_ratio_cuped_batches_bm_scores_against_local_fit(tmp_path):
    import datetime as dt

    import pyarrow as pa
    import yaml

    from increment import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(20261021)
    arms = ("control", "variant-a", "variant-b")
    covariate_names = tuple(f"x{index}" for index in range(34))
    units_per_arm = 170
    assert len(covariate_names) / units_per_arm <= 0.2
    units = []
    events = []
    exposure = dt.datetime(2025, 1, 10, 8)
    pre = dt.datetime(2025, 1, 7, 9)
    post = dt.datetime(2025, 1, 11, 9)
    for arm_index, arm in enumerate(arms):
        for index in range(units_per_arm):
            values = rng.normal(size=len(covariate_names))
            revenue = 12.0 + arm_index * 0.4 + 0.7 * values[0] + rng.normal(scale=0.3)
            orders = float(5 + index % 4)
            unit_id = f"{arm}-{index}"
            unit = {
                "user_id": unit_id,
                "variant": arm,
                "revenue": revenue,
                "orders": orders,
                **dict(zip(covariate_names, values.tolist(), strict=True)),
            }
            units.append(unit)
            events.extend(
                (
                    {
                        "user_id": unit_id,
                        "ts": exposure,
                        "event": "enrolled",
                        "experiment_id": "mixed_cuped",
                        "group_id": arm,
                        "value": None,
                        **dict.fromkeys(covariate_names),
                    },
                    {
                        "user_id": unit_id,
                        "ts": pre,
                        "event": "history",
                        "experiment_id": None,
                        "group_id": None,
                        "value": None,
                        **dict(zip(covariate_names, values.tolist(), strict=True)),
                    },
                    {
                        "user_id": unit_id,
                        "ts": post,
                        "event": "revenue",
                        "experiment_id": None,
                        "group_id": None,
                        "value": revenue,
                        **dict.fromkeys(covariate_names),
                    },
                )
            )
            events.extend(
                {
                    "user_id": unit_id,
                    "ts": post + dt.timedelta(minutes=minute),
                    "event": "order",
                    "experiment_id": None,
                    "group_id": None,
                    "value": None,
                    **dict.fromkeys(covariate_names),
                }
                for minute in range(int(orders))
            )

    definitions = yaml.safe_load(mixed_cuped_definitions_yaml("qr_many_covariates"))
    fact_source = definitions["fact_sources"][0]
    fact_source["properties"].extend(
        {
            "name": name,
            "column": name,
            "dtype": "float",
            "as_of": "pre_exposure",
        }
        for name in covariate_names
    )
    experiment = definitions["experiments"][0]
    experiment["allocation"] = dict.fromkeys(arms, 1 / len(arms))
    declarations = [
        {"name": name, "kind": "continuous", "property": name, "source": "events"}
        for name in covariate_names
    ]
    method = experiment["plan"]["guardrails"][0]["decision_method"]
    method["covariates"] = declarations
    experiment["plan"] = {"primary": {"metric": "rpo", "decision_method": method}}
    definitions_path = tmp_path / "qr-many-covariates.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))

    local_method = Method.cuped_multicovariate(
        covariates=[CupedCovariate(name=name, kind="continuous") for name in covariate_names]
    )
    local_experiment, local_resolver = from_frame(
        pa.Table.from_pylist(units),
        shape="summary",
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(
                name="rpo",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                decision_method=local_method,
            )
        ],
        plan=AnalysisPlan(primary="rpo"),
        design=Randomized(control_group="control", allocation=dict.fromkeys(arms, 1 / len(arms))),
    )
    connection = ibis.duckdb.connect()
    connection.create_table("qr_many_covariates", obj=events)
    warehouse_experiment, warehouse_resolver = from_definitions(
        definitions_path, connection, experiment_name="mixed_cuped"
    )
    try:
        actual = cast(
            list[LiftEstimate],
            view_readout(warehouse_experiment, warehouse_resolver, metrics=["rpo"]),
        )
        expected = cast(
            list[LiftEstimate], view_readout(local_experiment, local_resolver, metrics=["rpo"])
        )
        actual_rows = {(row.metric, row.group_id): row for row in actual}
        expected_rows = {(row.metric, row.group_id): row for row in expected}
        assert actual_rows.keys() == expected_rows.keys()
        for key, row in actual_rows.items():
            reference = expected_rows[key]
            assert row.sampling_reason_code is None
            assert reference.sampling_reason_code is None
            assert row.lift is not None and reference.lift is not None
            assert row.lift.value == pytest.approx(reference.lift.value, rel=1e-9, abs=1e-11)
            assert row.lift.log_se == pytest.approx(reference.lift.log_se, rel=1e-9, abs=1e-11)
            assert row.dof == pytest.approx(reference.dof, rel=1e-9, abs=1e-11)
    finally:
        cast(Any, warehouse_resolver).close()
        local_resolver.close()
        connection.disconnect()


@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        ("sparse_cells", "component_contributors"),
        ("insufficient_compression", "compression"),
        ("zero_within_cell_scatter", "component_scatter"),
    ],
)
@pytest.mark.parametrize("ingress", ["native", "artifact"])
@pytest.mark.integration
@pytest.mark.slow
def test_categorical_export_guard_misses_replay_from_fitted_qr(  # noqa: PLR0915
    fixture, reason, ingress, tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    import pyarrow.parquet as pq
    import yaml

    from increment import Method
    from increment.errors import CodedError
    from increment.frame import MetricSpec
    from increment.query._cuped_cells import categorical_cells_from_relation
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from increment.semantics.models import AnalysisPlan

    count = (
        20
        if fixture == "insufficient_compression"
        else 60
        if fixture == "zero_within_cell_scatter"
        else 200
    )
    units = matrix_data.linear_cuped_rows(n_per_arm=count)
    for row in units:
        index = int(row["user_id"][1:])
        if fixture == "sparse_cells":
            row["country"] = "rare" if index < 2 else f"level-{index % 3}"
        else:
            row["country"] = f"level-{index % 3}"
        if fixture == "zero_within_cell_scatter":
            category = int(row["country"].split("-")[1])
            row["revenue"] = float(10 + category + (2 if row["variant"] == "treatment" else 0))
    connection = ibis.duckdb.connect()
    connection.create_table(f"qr_{fixture}", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / f"{fixture}.yaml"
    definitions = yaml.safe_load(mixed_cuped_definitions_yaml(f"qr_{fixture}"))
    experiment_data = definitions["experiments"][0]
    method_data = experiment_data["plan"]["guardrails"][0]["decision_method"]
    method_data["covariates"] = [
        {"name": "country", "kind": "categorical", "property": "country", "source": "events"}
    ]
    experiment_data["plan"] = {"primary": {"metric": "revenue", "decision_method": method_data}}
    definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))

    native = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    definition_set = load(definitions_path)
    experiment = definition_set.experiment("mixed_cuped")
    metric = definition_set.metric("revenue")
    assert experiment is not None and metric is not None
    binding = experiment.bindings["revenue"]
    assert binding.decision_method is not None
    method = Method.model_validate(binding.decision_method.model_dump())
    cell_relation = connection.create_table(
        f"cells_{fixture}",
        obj=[
            {"group_id": row["variant"], "country": row["country"], "y": row["revenue"]}
            for row in units
        ],
    )
    request = cast(
        Any,
        SimpleNamespace(
            identity=("categorical-export", fixture),
            control="control",
            arm_order=("control", "treatment"),
            method=method,
            population="assigned",
            breakout_dimension=None,
            breakout_source=None,
            breakout_value=None,
            breakout_is_null=False,
        ),
    )
    with pytest.raises(CodedError) as cell_refusal:
        categorical_cells_from_relation(
            cell_relation,
            connection.to_pyarrow,
            metric=metric,
            request=request,
        )
    assert cell_refusal.value.code == "query.cuped.general_qr_required"
    assert cell_refusal.value.context["reason"] == reason

    artifact = None
    if ingress == "artifact":
        store = WarehouseArtifactStore(connection, schema_name=f"qr_{fixture}_artifact")
        context = artifact_context(definition_set, experiment, "error")
        extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
        reference = publish_unit_day_artifact(*native, store, extensions=extensions)
        artifact = from_unit_day_artifact(store, reference, expected_context=context)
        source = artifact
    else:
        source = native

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("categorical QR export transferred unit-keyed data")

    monkeypatch.setattr(DefinitionsMomentSource, "unit_frame", forbidden_unit_transfer)
    _guard_no_unit_keyed_transfers(monkeypatch)
    try:
        actual = cast(list[LiftEstimate], view_readout(*source, metrics=["revenue"]))
        destination = tmp_path / f"{ingress}-{fixture}.parquet"
        export_moments(*source, destination)
        exported = pq.read_table(destination).to_pylist()
        envelopes = [
            json.loads(row["cuped_evidence"])
            for row in exported
            if row.get("record_kind") == "cuped_evidence"
        ]
        assert [payload["kind"] for payload in envelopes] == ["fitted_multicovariate"]

        binding = experiment.bindings["revenue"]
        assert binding.decision_method is not None
        metric_spec = MetricSpec(
            name="revenue",
            type="mean",
            decision_method=Method.model_validate(binding.decision_method.model_dump()),
        )
        replay_experiment, replay_resolver = from_moments(
            exported,
            metrics=[metric_spec],
            design=experiment.resolved_design(),
            plan=AnalysisPlan(primary="revenue"),
        )
        try:
            replay_rows = cast(
                list[LiftEstimate],
                view_readout(replay_experiment, replay_resolver, metrics=["revenue"]),
            )
            actual_rows = {(row.metric, row.group_id): row for row in actual}
            replay_by_key = {(row.metric, row.group_id): row for row in replay_rows}
            assert replay_by_key.keys() == actual_rows.keys()
            for key, row in replay_by_key.items():
                reference = actual_rows[key]
                assert row.sampling_reason_code is None
                assert reference.sampling_reason_code is None
                assert row.require_lift().value == pytest.approx(
                    reference.require_lift().value, rel=1e-10, abs=1e-11
                )
                assert row.require_lift().log_se == pytest.approx(
                    reference.require_lift().log_se, rel=1e-10, abs=1e-11
                )
                assert row.dof == pytest.approx(reference.dof, rel=1e-10, abs=1e-11)
        finally:
            replay_resolver.close()
    finally:
        if artifact is not None:
            artifact[1].close()
        cast(Any, native[1]).close()
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("metric", ["revenue", "rpo"])
def test_native_mixed_cuped_uses_warehouse_qr_without_unit_transfer(metric, tmp_path, monkeypatch):
    units = matrix_data.linear_cuped_rows()
    connection = ibis.duckdb.connect()
    connection.create_table("qr_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "qr.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("qr_events"))
    warehouse = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("general warehouse QR transferred unit-keyed data")

    monkeypatch.setattr(DefinitionsMomentSource, "unit_frame", forbidden_unit_transfer)
    _guard_no_unit_keyed_transfers(monkeypatch)
    try:
        actual = cast(list[LiftEstimate], view_readout(*warehouse, metrics=[metric]))
        expected = cast(list[LiftEstimate], view_readout(*local, metrics=[metric]))
        actual_rows = {(row.metric, row.group_id, row.method): row for row in actual}
        expected_rows = {(row.metric, row.group_id, row.method): row for row in expected}
        assert actual_rows.keys() == expected_rows.keys()
        for key, row in actual_rows.items():
            reference = expected_rows[key]
            assert row.sampling_reason_code == reference.sampling_reason_code
            if row.sampling_reason_code is None:
                assert row.lift is not None and reference.lift is not None
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(row.lift, field) == pytest.approx(
                        getattr(reference.lift, field), rel=1e-9, abs=1e-11
                    )
        import pyarrow.parquet as pq

        from increment.estimation.engine import Method
        from increment.frame import MetricSpec
        from increment.semantics.loader import load
        from increment.semantics.models import AnalysisPlan

        experiment = load(definitions_path).experiment("mixed_cuped")
        assert experiment is not None
        binding = experiment.bindings[metric]
        assert binding.decision_method is not None
        method = Method.model_validate(binding.decision_method.model_dump())
        metric_spec = MetricSpec(
            name=metric,
            type="mean" if metric == "revenue" else "ratio",
            numerator="revenue" if metric == "rpo" else None,
            denominator="orders" if metric == "rpo" else None,
            decision_method=method,
        )
        destination = tmp_path / f"{metric}-moments.parquet"
        export_moments(*warehouse, destination)
        exported = pq.read_table(destination).to_pylist()
        frame_destination = tmp_path / f"{metric}-frame-moments.parquet"
        export_moments(*local, frame_destination)
        frame_exported = pq.read_table(frame_destination).to_pylist()
        _assert_fit_coordinates_equal(exported, frame_exported)
        assert {row["moments_format"] for row in exported} == {13}
        _assert_fitted_evidence_rejections(
            exported, metric_spec, method, experiment.resolved_design()
        )
        replay_experiment, replay_resolver = from_moments(
            exported,
            metrics=[metric_spec],
            design=experiment.resolved_design(),
            plan=AnalysisPlan(alpha=0.2, primary=metric),
        )
        try:
            replay_rows = cast(
                list[LiftEstimate],
                view_readout(replay_experiment, replay_resolver, metrics=[metric]),
            )
            replay_map = {(row.metric, row.group_id) for row in replay_rows}
            assert replay_map == {(row.metric, row.group_id) for row in actual}
            for row in replay_rows:
                original = actual_rows[(row.metric, row.group_id, row.method)]
                assert row.require_lift().value == pytest.approx(
                    original.require_lift().value, rel=1e-9, abs=1e-11
                )
                assert row.require_lift().lb != pytest.approx(original.require_lift().lb)
                assert row.require_lift().ub != pytest.approx(original.require_lift().ub)
        finally:
            replay_resolver.close()
    finally:
        local[1].close()
        cast(Any, warehouse[1]).close()
        connection.disconnect()


def _assert_fit_replay(source: tuple[Any, Any], metric, binding, design, expected, destination):
    import pyarrow.parquet as pq

    from increment.estimation.engine import Method
    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan

    metric_spec = MetricSpec(
        name=metric,
        type="mean" if metric == "revenue" else "ratio",
        numerator="revenue" if metric == "rpo" else None,
        denominator="orders" if metric == "rpo" else None,
        decision_method=Method.model_validate(binding.decision_method.model_dump()),
    )
    export_moments(*source, destination)
    exported = pq.read_table(destination).to_pylist()
    assert {row["moments_format"] for row in exported} == {13}
    experiment, resolver = from_moments(
        exported, metrics=[metric_spec], design=design, plan=AnalysisPlan(alpha=0.2, primary=metric)
    )
    try:
        replay_rows = cast(list[LiftEstimate], view_readout(experiment, resolver, metrics=[metric]))
        assert {(row.metric, row.group_id) for row in replay_rows} == {
            (row.metric, row.group_id) for row in expected
        }
        expected_rows = {(row.metric, row.group_id, row.method): row for row in expected}
        for row in replay_rows:
            original = expected_rows[(row.metric, row.group_id, row.method)]
            assert row.require_lift().value == pytest.approx(
                original.require_lift().value, rel=1e-9, abs=1e-11
            )
            assert row.require_lift().lb != pytest.approx(original.require_lift().lb)
    finally:
        resolver.close()
    return exported


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("metric", ["revenue", "rpo"])
def test_artifact_mixed_cuped_uses_warehouse_qr_without_unit_transfer(
    metric, tmp_path, monkeypatch
):
    from increment.query import source as artifact_source
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load

    units = matrix_data.linear_cuped_rows()
    connection = ibis.duckdb.connect()
    connection.create_table("qr_artifact_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "qr_artifact.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("qr_artifact_events"))
    native = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="qr_compact_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = publish_unit_day_artifact(*native, store, extensions=extensions)
    artifact = from_unit_day_artifact(store, reference, expected_context=context)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("artifact general warehouse QR transferred unit-keyed data")

    monkeypatch.setattr(
        artifact_source._ArtifactFacadeSource, "unit_frame", forbidden_unit_transfer
    )
    _guard_no_unit_keyed_transfers(monkeypatch)
    try:
        actual = cast(list[LiftEstimate], view_readout(*artifact, metrics=[metric]))
        expected = cast(list[LiftEstimate], view_readout(*local, metrics=[metric]))
        actual_rows = {(row.metric, row.group_id, row.method): row for row in actual}
        expected_rows = {(row.metric, row.group_id, row.method): row for row in expected}
        assert actual_rows.keys() == expected_rows.keys()
        for key, row in actual_rows.items():
            reference_row = expected_rows[key]
            assert row.sampling_reason_code == reference_row.sampling_reason_code
            if row.sampling_reason_code is None:
                assert row.lift is not None and reference_row.lift is not None
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(row.lift, field) == pytest.approx(
                        getattr(reference_row.lift, field), rel=1e-9, abs=1e-11
                    )
                assert row.dof == pytest.approx(reference_row.dof, rel=1e-9, abs=1e-11)
        artifact_exported = _assert_fit_replay(
            artifact,
            metric,
            experiment.bindings[metric],
            experiment.resolved_design(),
            actual,
            tmp_path / f"artifact-{metric}-moments.parquet",
        )
        import pyarrow.parquet as pq

        frame_destination = tmp_path / f"artifact-{metric}-frame-moments.parquet"
        export_moments(*local, frame_destination)
        frame_exported = pq.read_table(frame_destination).to_pylist()
        _assert_fit_coordinates_equal(artifact_exported, frame_exported)
    finally:
        local[1].close()
        artifact[1].close()
        cast(Any, native[1]).close()
        connection.disconnect()


@pytest.mark.integration
def test_qr_outcome_anchor_falls_back_before_overflow(tmp_path):
    from increment.query._cuped_qr import (
        encode_qr_covariates,
        factor_design_relation,
        prepare_qr_factor_input,
    )

    extreme = 1.7e308
    rows = [
        {"unit_id": f"{arm}{index}", "group_id": arm, "y": value}
        for arm in ("control", "treatment")
        for index, value in enumerate((extreme, -extreme))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("qr_extreme_outcome", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    try:
        relation = connection.table("qr_extreme_outcome")
        with session.qr_scratch() as scratch:
            encoded = encode_qr_covariates(
                relation,
                (),
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
            )
            prepared = prepare_qr_factor_input(
                encoded,
                outcome_columns=("y",),
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
            assert prepared.outcome_offsets == ((0.0, 0.0),)
            assert np.isfinite(prepared.outcome_scales).all()
            factor = factor_design_relation(
                prepared.relation,
                design_columns=prepared.design_columns,
                rhs_columns=prepared.rhs_columns,
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
                covariate_start=2,
                covariate_names=(),
            )
            assert np.isfinite(factor.transformed_rhs).all()
            assert np.isfinite(factor.triangular_factor).all()
    finally:
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("scales", [(1.0e-300, 1.0e300), (1.0e300, 1.0e-300)])
def test_duckdb_householder_factor_handles_tiny_and_huge_scales(scales, tmp_path):
    from increment.query._cuped_qr import factor_design_relation

    n = 16
    first = np.tile(np.array([1.0, -1.0]), n // 2)
    second = np.tile(np.array([1.0, 1.0, -1.0, -1.0]), n // 4)
    design = np.column_stack((first, second))
    outcomes = np.column_stack((first * scales[0], second * scales[1]))
    rows = [
        {
            "unit_id": f"scale-{index:02d}",
            "x0": float(x[0]),
            "x1": float(x[1]),
            "y0": float(y[0]),
            "y1": float(y[1]),
        }
        for index, (x, y) in enumerate(zip(design, outcomes, strict=True))
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    connection.create_table("scale_factor_input", obj=rows)
    session = _scratch_session(connection, tmp_path)
    try:
        with session.qr_scratch() as scratch:
            actual = factor_design_relation(
                connection.table("scale_factor_input"),
                design_columns=("x0", "x1"),
                rhs_columns=("y0", "y1"),
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
        np.testing.assert_allclose(actual_r, expected_r, rtol=1e-10, atol=1e-14)
        for component, scale in enumerate(scales):
            np.testing.assert_allclose(
                actual_rhs[:, component],
                expected_rhs[:, component],
                rtol=1e-10,
                atol=scale * 1e-14,
            )
    finally:
        session.drop_materialized()


@pytest.mark.integration
@pytest.mark.parametrize("sql_kind", ["integer", "decimal", "boolean"])
def test_qr_numeric_sql_domains_match_local_fitter(sql_kind, tmp_path):
    from decimal import Decimal

    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.frame import CupedCovariate
    from increment.query._cuped_qr import (
        encode_qr_covariates,
        factor_design_relation,
        prepare_qr_factor_input,
        solve_qr_coefficients,
    )

    rows = []
    arms = ("control", "treatment")
    for arm_index, arm in enumerate(arms):
        for index in range(36):
            if sql_kind == "integer":
                x = index + arm_index * 3
                y = index % 7 + arm_index
            elif sql_kind == "decimal":
                x = Decimal(index + arm_index * 3) / Decimal(10)
                y = Decimal(index % 7 + arm_index) / Decimal(4)
            else:
                x = bool(index % 2)
                y = bool(index % 7 < 3 + arm_index)
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": x,
                    "y": y,
                }
            )
    connection = ibis.duckdb.connect()
    connection.create_table("numeric_qr_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    numeric_x = np.asarray([float(row["x"]) for row in rows])
    numeric_y = np.asarray([float(row["y"]) for row in rows])
    arm_values = np.asarray([row["group_id"] for row in rows])
    local = fit_cuped_multicovariate(
        numeric_y,
        arm_values,
        control="control",
        arm_order=arms,
        covariates={"x": numeric_x},
        kinds={"x": "continuous"},
    )
    try:
        relation = connection.table("numeric_qr_input")
        if sql_kind == "integer":
            assert relation["x"].type().is_integer()
            assert relation["y"].type().is_integer()
        elif sql_kind == "decimal":
            assert relation["x"].type().is_decimal()
            assert relation["y"].type().is_decimal()
        else:
            assert relation["x"].type().is_boolean()
            assert relation["y"].type().is_boolean()
        with session.qr_scratch() as scratch:
            encoded = encode_qr_covariates(
                relation,
                (CupedCovariate(name="x", kind="continuous"),),
                arms=arms,
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
            )
            prepared = prepare_qr_factor_input(
                encoded,
                outcome_columns=("y",),
                arms=arms,
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
            factor = factor_design_relation(
                prepared.relation,
                design_columns=prepared.design_columns,
                rhs_columns=prepared.rhs_columns,
                order_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
                covariate_start=len(arms),
                covariate_names=encoded.layout.encoded_columns,
            )
            coefficients = solve_qr_coefficients(
                factor,
                design_scales=prepared.design_scales,
                outcome_scales=prepared.outcome_scales,
            )
        assert coefficients[0][-1] / encoded.scales[0] == pytest.approx(
            local.theta[0], rel=1e-9, abs=1e-12
        )
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("case", ["empty_arm", "null_outcome", "nan_outcome"])
def test_qr_input_refusals_match_local_codes(case, tmp_path):
    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.query._cuped_qr import encode_qr_covariates, prepare_qr_factor_input

    rows = []
    for arm in ("control", "treatment"):
        for index in range(16):
            outcome = float(index + (arm == "treatment"))
            if case == "null_outcome" and index == 3:
                outcome = None
            if case == "nan_outcome" and index == 3:
                outcome = float("nan")
            rows.append(
                {
                    "unit_id": f"{arm}-{index:02d}",
                    "group_id": arm,
                    "y": outcome,
                }
            )
    arms = ("control", "treatment", "withheld") if case == "empty_arm" else ("control", "treatment")
    local_outcome = np.asarray(
        [np.nan if row["y"] is None else row["y"] for row in rows], dtype=float
    )
    with pytest.raises(CodedError) as local_error:
        fit_cuped_multicovariate(
            local_outcome,
            np.asarray([row["group_id"] for row in rows]),
            control="control",
            arm_order=arms,
            covariates={},
            kinds={},
        )
    connection = ibis.duckdb.connect()
    connection.create_table("qr_refusal_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    try:
        with pytest.raises(CodedError) as warehouse_error:
            with session.qr_scratch() as scratch:
                encoded = encode_qr_covariates(
                    connection.table("qr_refusal_input"),
                    (),
                    arms=arms,
                    row_identity_columns=("unit_id",),
                    execute=connection.to_pyarrow,
                )
                prepare_qr_factor_input(
                    encoded,
                    outcome_columns=("y",),
                    arms=arms,
                    row_identity_columns=("unit_id",),
                    execute=connection.to_pyarrow,
                    scratch=scratch,
                )
        assert warehouse_error.value.code == local_error.value.code
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
def test_qr_outcome_centers_cancelling_extremes_like_local_fit(tmp_path):
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.query._cuped_qr import encode_qr_covariates, prepare_qr_factor_input

    rows = []
    for arm, ordinary in (("control", 1.0), ("treatment", 2.0)):
        outcomes = [1.0e300, -1.0e300, *([ordinary] * 8)]
        rows.extend(
            {
                "unit_id": f"{arm}-{index:02d}",
                "group_id": arm,
                "y": value,
            }
            for index, value in enumerate(outcomes)
        )
    arms = ("control", "treatment")
    local = fit_cuped_multicovariate(
        np.asarray([row["y"] for row in rows]),
        np.asarray([row["group_id"] for row in rows]),
        control=arms[0],
        arm_order=arms,
        covariates={},
        kinds={},
    )
    connection = ibis.duckdb.connect()
    connection.create_table("canceling_qr_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    try:
        relation = connection.table("canceling_qr_input")
        encoded = encode_qr_covariates(
            relation,
            (),
            arms=arms,
            row_identity_columns=("unit_id",),
            execute=connection.to_pyarrow,
        )
        with session.qr_scratch() as scratch:
            prepared = prepare_qr_factor_input(
                encoded,
                outcome_columns=("y",),
                arms=arms,
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
        warehouse_means = np.asarray(prepared.outcome_offsets[0]) + np.asarray(
            prepared.centered_outcome_means[0]
        )
        scale = max(abs(float(row["y"])) for row in rows)
        for arm_index, arm in enumerate(arms):
            values = [float(row["y"]) for row in rows if row["group_id"] == arm]
            exact_mean = math.fsum(values) / len(values)
            ulp_tolerance = 4 * math.ulp(scale)
            assert abs(warehouse_means[arm_index] - exact_mean) <= ulp_tolerance
            assert abs(local.adjusted_means[arm_index] - exact_mean) <= ulp_tolerance
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_unchanged_warehouse_qr_requests_keep_stable_identity():
    from types import SimpleNamespace

    from increment.estimation.engine import Method
    from increment.query.cuped import resolved_multicovariate_request
    from increment.semantics.models import CupedCovariate

    source = SimpleNamespace(context=SimpleNamespace(study_id="stable-qr-identity"))
    metric = SimpleNamespace(name="metric")
    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    first = resolved_multicovariate_request(
        source, metric, method, control="control", arm_order=("control", "treatment")
    )
    repeated = resolved_multicovariate_request(
        source, metric, method, control="control", arm_order=("control", "treatment")
    )
    assert first.identity == repeated.identity
    changed = resolved_multicovariate_request(
        source, metric, method, control="treatment", arm_order=("control", "treatment")
    )
    assert first.identity != changed.identity


@pytest.mark.integration
def test_warehouse_fit_input_digest_accepts_dataframe_snapshot_results():
    from contextlib import contextmanager
    from types import SimpleNamespace

    import ibis
    import pandas as pd
    import pyarrow as pa

    from increment.estimation.engine import Method
    from increment.query.cuped import method_fit_input_digest
    from increment.semantics.models import CupedCovariate

    connection = ibis.duckdb.connect()
    connection.create_table(
        "digest_input",
        obj=[
            {"unit_id": "u1", "group_id": "control", "y": 1.0, "x": 2.0},
            {"unit_id": "u2", "group_id": "treatment", "y": 3.0, "x": 4.0},
        ],
    )
    relation = connection.table("digest_input")
    aggregate = {"hash_sum": 11, "salted_hash_sum": 13, "row_count": 2}

    class Source:
        context = SimpleNamespace(study_id="dataframe-snapshot-digest")
        result = None

        @contextmanager
        def cuped_relation(self, _metric, _request):
            yield relation, lambda _query: self.result, None

    source = Source()
    request = SimpleNamespace(
        metric=SimpleNamespace(name="metric"),
        method=Method.cuped_multicovariate(
            covariates=[CupedCovariate(name="x", kind="continuous")]
        ),
        control="control",
        arm_order=("control", "treatment"),
        population="assigned",
        cell=None,
    )
    try:
        source.result = pd.DataFrame([aggregate])
        dataframe_digest = method_fit_input_digest(source, request)
        source.result = pa.Table.from_pylist([aggregate])
        arrow_digest = method_fit_input_digest(source, request)
        assert dataframe_digest == arrow_digest
    finally:
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("denominator_scale", (1.0e-50, 1.0e-40, 1.0e-20))
def test_qr_ratio_with_near_zero_denominators_matches_local_fit(denominator_scale, tmp_path):
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.estimation.engine import Method
    from increment.query._cuped_qr import fit_warehouse_qr
    from increment.query.cuped import _ResolvedCupedRequest
    from increment.semantics.models import CupedCovariate, Measure, RatioMetric

    rows = []
    arms = ("control", "treatment")
    for arm_index, arm in enumerate(arms):
        for index in range(60):
            coordinate = index / 59.0
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": coordinate,
                    "y": 3.0 + 0.2 * arm_index + 0.4 * coordinate + 0.05 * np.sin(index),
                    "y_den": denominator_scale
                    * (1.0 + 0.02 * np.sin(index * 0.7) + arm_index * 0.001),
                }
            )
    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    metric = RatioMetric(
        name="ratio",
        entity="unit",
        numerator=Measure(fact="num", aggregation="sum"),
        denominator=Measure(fact="den", aggregation="sum"),
    )
    request = _ResolvedCupedRequest(
        identity=("near-zero-ratio",),
        control=arms[0],
        arm_order=arms,
        method=method,
    )
    values = np.asarray([row["y"] for row in rows])
    denominators = np.asarray([row["y_den"] for row in rows])
    groups = np.asarray([row["group_id"] for row in rows])
    coordinates = np.asarray([row["x"] for row in rows])
    local = fit_cuped_multicovariate(
        None,
        groups,
        control=arms[0],
        arm_order=arms,
        covariates={"x": coordinates},
        kinds={"x": "continuous"},
        numerator=values,
        denominator=denominators,
    )
    connection = ibis.duckdb.connect()
    connection.create_table("near_zero_ratio", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)

    def guarded_execute(expression):
        _assert_no_transferred_extreme(expression)
        result = connection.to_pyarrow(expression)
        assert "unit_id" not in result.column_names
        return result

    try:
        with session.qr_scratch() as scratch:
            actual = fit_warehouse_qr(
                connection.table("near_zero_ratio"),
                metric=metric,
                request=request,
                execute=guarded_execute,
                scratch=scratch,
            )
        for name in ("theta", "adjusted_means", "covariance", "dof", "max_leverage"):
            assert np.asarray(getattr(actual, name)) == pytest.approx(
                np.asarray(getattr(local, name)), rel=1e-9, abs=1e-12
            )
        assert len(actual._projections) == len(local._projections)
        for actual_projection, local_projection in zip(
            actual._projections, local._projections, strict=True
        ):
            assert actual_projection[:4] == local_projection[:4]
            assert actual_projection[4:] == pytest.approx(local_projection[4:], rel=1e-9, abs=1e-12)
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
def test_qr_tiny_mean_large_denominator_spread_refuses_magnitude_range(tmp_path):
    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate
    from increment.query._cuped_qr import fit_warehouse_qr
    from increment.query.cuped import _ResolvedCupedRequest
    from increment.semantics.models import Measure, RatioMetric

    arms = ("control", "treatment")
    rows = []
    denominator_values = {
        "control": (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
        "treatment": (3.0e-300, -2.98e-300, 0.01e-300, 0.01e-300, 0.01e-300, 0.01e-300),
    }
    covariate_values = {"x": []}
    for arm_index, arm in enumerate(arms):
        for index, denominator in enumerate(denominator_values[arm]):
            x = float(index // 3)
            covariate_values["x"].append(x)
            rows.append(
                {
                    "unit_id": f"{arm}-{index}",
                    "group_id": arm,
                    "x": x,
                    "y": 2.0 + arm_index + index,
                    "y_den": denominator,
                }
            )
    declarations = [CupedCovariate(name="x", kind="continuous")]
    method = Method.cuped_multicovariate(covariates=declarations)
    metric = RatioMetric(
        name="ratio",
        entity="unit",
        numerator=Measure(fact="num", aggregation="sum"),
        denominator=Measure(fact="den", aggregation="sum"),
    )
    request = _ResolvedCupedRequest(
        identity=("tiny-denominator-support",),
        control=arms[0],
        arm_order=arms,
        method=method,
    )
    with pytest.raises(CodedError) as local_error:
        fit_cuped_multicovariate(
            None,
            [row["group_id"] for row in rows],
            control=arms[0],
            arm_order=arms,
            covariates=covariate_values,
            kinds={"x": "continuous"},
            numerator=[row["y"] for row in rows],
            denominator=[row["y_den"] for row in rows],
        )
    assert local_error.value.code == "estimation.cuped_linear.support_region"

    connection = ibis.duckdb.connect()
    connection.create_table("tiny_denominator_support", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)

    def guarded_execute(expression):
        _assert_no_transferred_extreme(expression)
        result = connection.to_pyarrow(expression)
        assert "unit_id" not in result.column_names
        return result

    try:
        with pytest.raises(CodedError) as actual_error:
            with session.qr_scratch() as scratch:
                fit_warehouse_qr(
                    connection.table("tiny_denominator_support"),
                    metric=metric,
                    request=request,
                    execute=guarded_execute,
                    scratch=scratch,
                )
        assert actual_error.value.code == "query.cuped.magnitude_range"
        assert actual_error.value.context["metric"] == "ratio"
        assert actual_error.value.context["component"] == "denominator"
        assert actual_error.value.context["bound"] == "lower"
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
def test_qr_encoder_refuses_infinite_covariates_with_coded_error(tmp_path):
    from increment.errors import InvalidRequestError
    from increment.frame import CupedCovariate
    from increment.query._cuped_qr import encode_qr_covariates

    connection = ibis.duckdb.connect()
    connection.create_table(
        "qr_infinite_covariate",
        obj=[
            {"unit_id": "c0", "group_id": "control", "x": 1.0},
            {"unit_id": "c1", "group_id": "control", "x": 2.0},
            {"unit_id": "t0", "group_id": "treatment", "x": float("inf")},
            {"unit_id": "t1", "group_id": "treatment", "x": 3.0},
        ],
    )
    try:
        with pytest.raises(InvalidRequestError) as raised:
            encode_qr_covariates(
                connection.table("qr_infinite_covariate"),
                (CupedCovariate(name="x", kind="continuous"),),
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
            )
        assert raised.value.code == "estimation.cuped_linear.nonfinite"
        assert raised.value.context["name"] == "x"
    finally:
        connection.disconnect()


@pytest.mark.integration
def test_qr_encoder_routes_more_than_200_levels_with_coded_refusal():
    from increment.errors import CapabilityError
    from increment.frame import CupedCovariate
    from increment.query._cuped_qr import encode_qr_covariates

    rows = [
        {"unit_id": f"{arm}-{index}", "group_id": arm, "category": f"level-{index:03d}"}
        for arm in ("control", "treatment")
        for index in range(201)
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("qr_many_levels", obj=rows)
    try:
        with pytest.raises(CapabilityError) as raised:
            encode_qr_covariates(
                connection.table("qr_many_levels"),
                (CupedCovariate(name="category", kind="categorical"),),
                arms=("control", "treatment"),
                row_identity_columns=("unit_id",),
                execute=connection.to_pyarrow,
            )
        assert raised.value.code == "query.cuped.general_qr_required"
        assert raised.value.context["reason"] == "categorical_levels"
    finally:
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("invalid_arm", ("variant-b", "control"))
def test_ratio_qr_withholds_nonpositive_arm_before_fit(invalid_arm, tmp_path):
    from contextlib import contextmanager

    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.estimation.engine import Method, estimate_cuped_linear_fit
    from increment.frame import CupedCovariate
    from increment.query.cuped import (
        _ResolvedCupedRequest,
        fit_multicovariate_source,
    )
    from increment.semantics.models import Measure, RatioMetric

    arms = ("control", "variant-a", "variant-b")
    rows = []
    for arm_index, arm in enumerate(arms):
        for index in range(48):
            x = index / 47.0
            den = -1.0 if arm == invalid_arm else 1.0 + 0.01 * np.sin(index)
            slope = 7.0 if arm == invalid_arm else 0.4
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": x,
                    "y": 4.0 + arm_index + slope * x + 0.02 * np.cos(index),
                    "y_den": den,
                }
            )
    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    metric = RatioMetric(
        name="ratio",
        entity="unit",
        numerator=Measure(fact="num", aggregation="sum"),
        denominator=Measure(fact="den", aggregation="sum"),
    )
    request = _ResolvedCupedRequest(
        identity=("ratio-withheld",),
        control="control",
        arm_order=arms,
        method=method,
    )
    connection = ibis.duckdb.connect()
    connection.create_table("ratio_withheld_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)

    class Source:
        @contextmanager
        def cuped_relation(self, _metric, _request):
            yield (
                connection.table("ratio_withheld_input"),
                connection.to_pyarrow,
                session.qr_scratch,
            )

    try:
        evidence = fit_multicovariate_source(Source(), metric, request)
        assert evidence.arm_order == arms
        withheld = dict(evidence.withheld)
        assert invalid_arm in withheld
        assert withheld[invalid_arm].code == (
            "estimation.variance.ratio_moments_nonpositive_denominator_mean"
        )
        if invalid_arm == "control":
            assert evidence.fit is None
        else:
            assert evidence.fit is not None
            assert evidence.fit.arm_order == ("control", "variant-a")

        projected = estimate_cuped_linear_fit(
            metric,
            evidence,
            arm_order=arms,
            control_group="control",
            method=method,
            method_role="sensitivity",
            alpha=0.05,
            alternative="two-sided",
        )
        output = {row.group_id: row for row in projected.results}
        if invalid_arm == "control":
            assert all(row.failure_code == withheld["control"].code for row in output.values())
        else:
            assert output["variant-b"].failure_code == withheld["variant-b"].code
            actual = output["variant-a"]
            local_mask = np.asarray([row["group_id"] != "variant-b" for row in rows], dtype=bool)
            local = fit_cuped_multicovariate(
                None,
                np.asarray([row["group_id"] for row in rows], dtype=str)[local_mask],
                control="control",
                arm_order=("control", "variant-a"),
                covariates={"x": np.asarray([row["x"] for row in rows], dtype=float)[local_mask]},
                kinds={"x": "continuous"},
                numerator=np.asarray([row["y"] for row in rows], dtype=float)[local_mask],
                denominator=np.asarray([row["y_den"] for row in rows], dtype=float)[local_mask],
            )
            expected = local.ratio("variant-a", control="control", scale="log")
            assert actual.lift is not None
            assert actual.lift.value == pytest.approx(math.expm1(expected.estimate), rel=1e-9)
            assert actual.lift.log_se == pytest.approx(expected.se, rel=1e-9)
            assert actual.dof == pytest.approx(expected.dof, rel=1e-9)
    finally:
        session.drop_materialized()
        connection.disconnect()


@pytest.mark.integration
def test_qr_null_factor_entries_refuse_rank_deficiency_instead_of_crashing(tmp_path):
    from increment.errors import CodedError
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.query._cuped_qr import factor_design_relation

    arms = ("control", "treatment")
    rows = [
        {
            "unit_id": "c0",
            "arm_control": 1.0,
            "arm_treatment": 0.0,
            "x0": 0.0,
            "x1": 0.0,
            "x_missing": None,
            "x2": 0.0,
            "y": 1.0,
        },
        {
            "unit_id": "c1",
            "arm_control": 1.0,
            "arm_treatment": 0.0,
            "x0": 1.0,
            "x1": 0.0,
            "x_missing": None,
            "x2": 1.0,
            "y": 2.0,
        },
        {
            "unit_id": "t0",
            "arm_control": 0.0,
            "arm_treatment": 1.0,
            "x0": 0.0,
            "x1": 1.0,
            "x_missing": None,
            "x2": 1.0,
            "y": 3.0,
        },
        {
            "unit_id": "t1",
            "arm_control": 0.0,
            "arm_treatment": 1.0,
            "x0": 1.0,
            "x1": 1.0,
            "x_missing": None,
            "x2": 0.0,
            "y": 5.0,
        },
    ]
    with pytest.raises(CodedError) as local_error:
        fit_cuped_multicovariate(
            np.asarray([row["y"] for row in rows]),
            np.asarray(["control", "control", "treatment", "treatment"]),
            control="control",
            arm_order=arms,
            covariates={
                name: np.asarray([row[source] for row in rows], dtype=object)
                for name, source in (("x0", "x0"), ("x1", "x1"), ("x2", "x2"))
            },
            kinds={"x0": "continuous", "x1": "continuous", "x2": "continuous"},
            missing={"x2": "impute-indicator"},
        )

    connection = ibis.duckdb.connect()
    connection.create_table(
        "qr_null_factor_input",
        obj=[{name: value for name, value in row.items() if name != "x_missing"} for row in rows],
    )
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    try:
        with pytest.raises(CodedError) as warehouse_error:
            with session.qr_scratch() as scratch:
                factor_design_relation(
                    connection.table("qr_null_factor_input").mutate(
                        x_missing=ibis.null().cast("float64")
                    ),
                    design_columns=(
                        "arm_control",
                        "arm_treatment",
                        "x0",
                        "x1",
                        "x_missing",
                        "x2",
                    ),
                    rhs_columns=("y",),
                    order_columns=("unit_id",),
                    execute=connection.to_pyarrow,
                    scratch=scratch,
                    covariate_start=2,
                    covariate_names=("x0", "x1", "x_missing", "x2"),
                )
        assert warehouse_error.value.code == local_error.value.code
        assert warehouse_error.value.code == "estimation.cuped_linear.rank_deficiency"
    finally:
        session.drop_materialized()
        connection.disconnect()


def test_qr_breakout_degenerate_segments_match_local_refusal_codes():
    import pandas as pd

    from increment.errors import CodedError, IncrementWarning
    from increment.estimation.cuped import fit_cuped_multicovariate
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, MultiplicitySpec

    rows = []
    for index in range(4):
        rows.append(
            {
                "unit": f"empty-{index}",
                "group": "control",
                "segment": "empty-arm",
                "outcome": float(index),
                "x0": float(index % 2),
                "x1": 0.0,
                "x2": float(index % 2),
                "category": None,
            }
        )
    for arm in ("control", "treatment"):
        for index in range(2):
            rows.append(
                {
                    "unit": f"null-{arm}-{index}",
                    "group": arm,
                    "segment": "all-null",
                    "outcome": float(index + (arm == "treatment")),
                    "x0": float(index),
                    "x1": float(index),
                    "x2": float(1 - index),
                    "category": None,
                }
            )
    method = Method.cuped_multicovariate(
        covariates=(
            CupedCovariate(name="x0", kind="continuous"),
            CupedCovariate(name="x1", kind="continuous"),
            CupedCovariate(name="x2", kind="continuous"),
            CupedCovariate(
                name="category",
                kind="categorical",
                missing="impute-indicator",
            ),
        )
    )
    experiment, resolver = from_frame(
        pd.DataFrame(rows),
        shape="summary",
        unit="unit",
        group="group",
        metrics=(MetricSpec(name="outcome", decision_method=method),),
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
        breakouts=("segment",),
    )
    with pytest.warns(IncrementWarning):
        results = view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
    by_segment = {row.dimension_value: row for row in results}
    assert set(by_segment) == {"empty-arm", "all-null"}
    assert by_segment["empty-arm"].excluded == "no_control_arm"
    null_rows = [row for row in rows if row["segment"] == "all-null"]
    with pytest.raises(CodedError) as local_error:
        fit_cuped_multicovariate(
            np.asarray([row["outcome"] for row in null_rows]),
            np.asarray([row["group"] for row in null_rows]),
            control="control",
            arm_order=("control", "treatment"),
            covariates={
                name: np.asarray([row[name] for row in null_rows], dtype=object)
                for name in ("x0", "x1", "x2", "category")
            },
            kinds={
                "x0": "continuous",
                "x1": "continuous",
                "x2": "continuous",
                "category": "categorical",
            },
            missing={"category": "impute-indicator"},
        )
    assert by_segment["all-null"].failure_code == local_error.value.code
    assert by_segment["all-null"].failure_code == "estimation.cuped_linear.rank_deficiency"


@pytest.mark.integration
@pytest.mark.slow
def test_withheld_ratio_arm_identity_matches_native_and_artifact(  # noqa: PLR0915
    tmp_path, monkeypatch
):
    import pyarrow as pa
    import yaml

    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.plan import AnalysisPlan
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load

    arms = ("control", "treatment", "withheld")
    units = []
    for arm_index, arm in enumerate(arms):
        for index in range(48):
            units.append(
                {
                    "user_id": f"{arm}-{index:03d}",
                    "variant": arm,
                    "revenue": float(4 + arm_index + index % 7),
                    "orders": 0.0 if arm == "withheld" else float(1 + index % 4),
                    "pre_revenue": None if index % 9 == 0 else float(8 + index % 13),
                    "region": None if index % 6 == 0 else ("east" if index % 2 == 0 else "west"),
                    "country": ("north", "south", "west")[index % 3],
                    "device": ("mobile", "desktop")[index % 2],
                    "new_user": "new" if index % 4 == 0 else "existing",
                    "signups": float(index % 9),
                }
            )
    events = mixed_cuped_event_rows(units)
    definitions = yaml.safe_load(mixed_cuped_definitions_yaml("qr_ratio_withheld"))
    experiment = definitions["experiments"][0]
    experiment["allocation"] = dict.fromkeys(arms, 1 / 3)
    experiment["plan"] = {"primary": {"metric": "rpo", **experiment["plan"]["guardrails"][0]}}
    definitions_path = tmp_path / "qr-ratio-withheld.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))

    covariates = [
        CupedCovariate(name="pre_revenue", kind="continuous", missing="impute-indicator"),
        CupedCovariate(name="country", kind="categorical"),
        CupedCovariate(name="device", kind="categorical"),
        CupedCovariate(name="new_user", kind="categorical"),
    ]
    method = Method.cuped_multicovariate(covariates=covariates)
    local = from_frame(
        pa.Table.from_pylist(units),
        shape="summary",
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(
                name="rpo",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                preferred_direction="increase",
                decision_method=method,
            )
        ],
        plan=AnalysisPlan(primary="rpo"),
        design=Randomized(
            control_group="control",
            allocation=dict.fromkeys(arms, 1 / 3),
        ),
    )
    connection = ibis.duckdb.connect()
    connection.create_table("qr_ratio_withheld", obj=events)
    native = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    artifact = None
    try:
        definition_set = load(definitions_path)
        experiment = definition_set.experiment("mixed_cuped")
        assert experiment is not None
        store = WarehouseArtifactStore(connection, schema_name="qr_ratio_withheld_artifact")
        context = artifact_context(definition_set, experiment, "error")
        extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
        reference = publish_unit_day_artifact(*native, store, extensions=extensions)
        artifact = from_unit_day_artifact(store, reference, expected_context=context)
        _guard_no_unit_keyed_transfers(monkeypatch)

        import warnings

        from increment.errors import IncrementWarning

        def run_with_withheld_notice(analysis: tuple[Any, Any]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IncrementWarning)
                rows = cast(list[LiftEstimate], view_readout(*analysis, metrics=["rpo"]))
            assert any(
                isinstance(warning.message, IncrementWarning)
                and warning.message.code == "readouts.run.cell_refused"
                and warning.message.context["group_id"] == "withheld"
                for warning in caught
            )
            return {row.group_id: row for row in rows}

        expected = run_with_withheld_notice(local)
        for analysis in (native, artifact):
            actual = run_with_withheld_notice(analysis)
            assert actual.keys() == expected.keys()
            for group_id, row in expected.items():
                result = actual[group_id]
                assert result.sampling_reason_code == row.sampling_reason_code
                if row.lift is not None:
                    assert result.lift is not None
                    assert result.lift.value == pytest.approx(row.lift.value, rel=1e-9, abs=1e-11)
                    assert result.lift.log_se == pytest.approx(row.lift.log_se, rel=1e-9, abs=1e-11)
                    assert result.dof == pytest.approx(row.dof, rel=1e-9, abs=1e-11)
    finally:
        local[1].close()
        if artifact is not None:
            artifact[1].close()
        cast(Any, native[1]).close()
        connection.disconnect()
