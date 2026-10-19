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


def _positive_diagonal(r, rhs):
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return signs[:, None] * r, signs[:, None] * rhs


@pytest.mark.parametrize("p", [1, 4, 20, 50])
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
                row_identity_columns=("unit_id",),
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
                row_identity_columns=("unit_id",),
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


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "snowflake", "bigquery"])
def test_qr_unit_summary_statement_size_at_supported_p(dialect, tmp_path, monkeypatch):
    from increment.query._cuped_qr import QRFactor, warehouse_qr_unit_summary

    width = 203
    design_columns = tuple(f"d{index}" for index in range(width))
    row = {
        "unit_id": "one",
        "sqrt_weight": 1.0,
        "y_fit": 1.0,
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
                coefficients=((0.0,) * width,),
                execute=connection.to_pyarrow,
                scratch=scratch,
            )
        assert len(statements) >= 4
        assert max(map(len, statements)) < 100_000
    finally:
        session.drop_materialized()
        connection.disconnect()


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
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= 1
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
            covariance_jacobian = (
                np.asarray(spec.jacobian)
                * (spec.scale / spec.divisor)
                * np.ldexp(1.0, spec.exponent)
            )
            expected_score_scale = np.sqrt(
                covariance_jacobian
                @ np.asarray(summary.adjusted_component_covariance)
                @ covariance_jacobian
            )
            assert projection.se == pytest.approx(expected_score_scale, rel=1e-9, abs=1e-11)
            assert summary.scaled_projected_score_squares[index][5] == pytest.approx(
                projection.se, rel=1e-9, abs=1e-11
            )
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
        result = connection.to_pyarrow(expression)
        assert result.num_rows <= 1
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
                summary.scaled_projected_score_squares[projection_index][5],
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
            assert encoded.offsets == ((7.0, 11.0), (1.0, 9.0), (0.0, 0.0), (0.0, 0.0))
            assert encoded.scales[0] == pytest.approx(11.0)
            assert encoded.scales[1] == pytest.approx(11.0)
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
    _guard_no_unit_keyed_transfers(monkeypatch)
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
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(row.lift, field) == pytest.approx(
                        getattr(reference.lift, field), rel=1e-9, abs=1e-11
                    )
                assert row.dof == pytest.approx(reference.dof, rel=1e-9, abs=1e-11)
    finally:
        local.close()
        warehouse.close()
        connection.disconnect()


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
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(connection, schema_name="qr_compact_artifacts")
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    artifact = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)

    def forbidden_unit_transfer(*args, **kwargs):
        pytest.fail("artifact general warehouse QR transferred unit-keyed data")

    monkeypatch.setattr(
        artifact_source._ArtifactFacadeSource, "unit_frame", forbidden_unit_transfer
    )
    _guard_no_unit_keyed_transfers(monkeypatch)
    try:
        actual = cast(list[LiftEstimate], artifact.run(metrics=[metric]))
        expected = cast(list[LiftEstimate], local.run(metrics=[metric]))
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
    finally:
        local.close()
        artifact.close()
        native.close()
        connection.disconnect()


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
        np.testing.assert_allclose(warehouse_means, local.adjusted_means, rtol=1e-9, atol=1e-12)
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


def test_qr_ratio_with_near_zero_denominators_matches_local_fit(tmp_path):
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
                    "y_den": 1.0e-150 * (1.0 + 0.02 * np.sin(index * 0.7) + arm_index * 0.001),
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
    try:
        with session.qr_scratch() as scratch:
            actual = fit_warehouse_qr(
                connection.table("near_zero_ratio"),
                metric=metric,
                request=request,
                execute=connection.to_pyarrow,
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
