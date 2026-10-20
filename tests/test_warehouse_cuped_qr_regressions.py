from __future__ import annotations

import math
from types import SimpleNamespace

import ibis
import pytest

from increment.errors import CapabilityError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.engine import Method
from increment.frame import CupedCovariate
from increment.query._cuped_qr import fit_warehouse_qr
from increment.query.cuped import _ResolvedCupedRequest
from tests.test_warehouse_cuped_qr import _scratch_session

ARMS = ("control", "treatment")


def _fit_pair(
    rows: list[dict[str, float | str]],
    *,
    ratio: bool,
    tmp_path,
    transfers=None,
    scratch_statements=None,
    arms: tuple[str, ...] = ARMS,
    categorical: bool = False,
):
    groups = [str(row["group_id"]) for row in rows]
    x_values = [float(row["x"]) for row in rows]
    y_values = [float(row["y"]) for row in rows]
    covariates = {"x": x_values}
    kinds = {"x": "continuous"}
    declarations = [CupedCovariate(name="x", kind="continuous")]
    if categorical:
        covariates["category"] = [str(row["category"]) for row in rows]
        kinds["category"] = "categorical"
        declarations.append(CupedCovariate(name="category", kind="categorical"))
    if ratio:
        local = fit_cuped_multicovariate(
            None,
            groups,
            control=arms[0],
            arm_order=arms,
            covariates=covariates,
            kinds=kinds,
            numerator=y_values,
            denominator=[float(row["y_den"]) for row in rows],
        )
    else:
        local = fit_cuped_multicovariate(
            y_values,
            groups,
            control=arms[0],
            arm_order=arms,
            covariates=covariates,
            kinds=kinds,
        )
    connection = ibis.duckdb.connect()
    connection.create_table("probe_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    method = Method.cuped_multicovariate(covariates=declarations)
    request = _ResolvedCupedRequest(
        identity=("warehouse_qr_regression",),
        control=arms[0],
        arm_order=arms,
        method=method,
    )
    original_create = type(connection).create_table

    def record_scratch_sql(backend, name, expression, *args, **kwargs):
        if str(name).startswith("_cuped_qr_") and scratch_statements is not None:
            scratch_statements.append(ibis.to_sql(expression, dialect="duckdb"))
        return original_create(backend, name, expression, *args, **kwargs)

    type(connection).create_table = record_scratch_sql

    def execute(expression):
        result = connection.to_pyarrow(expression)
        if transfers is not None:
            transfers.extend(result.to_pylist())
        return result

    try:
        with session.qr_scratch() as scratch:
            actual = fit_warehouse_qr(
                connection.table("probe_input"),
                metric=SimpleNamespace(name="probe", type="ratio" if ratio else "mean"),
                request=request,
                execute=execute,
                scratch=scratch,
            )
        return local, actual
    finally:
        type(connection).create_table = original_create
        session.drop_materialized()
        connection.disconnect()


def _assert_magnitude_refusal(rows, *, ratio, tmp_path, component, bound):
    transfers = []
    with pytest.raises(CapabilityError) as error:
        _fit_pair(rows, ratio=ratio, tmp_path=tmp_path, transfers=transfers)
    assert error.value.code == "query.cuped.magnitude_range"
    assert error.value.context["metric"] == "probe"
    assert error.value.context["component"] == component
    assert error.value.context["bound"] == bound
    assert transfers
    assert {row["group_id"] for row in transfers} == set(ARMS)
    assert all(
        all(name == "group_id" or name.startswith("_magnitude_") for name in row)
        for row in transfers
    )
    assert all(not {"y", "y_den", "x"} & set(row) for row in transfers)


def _mixed_scale_rows(scale_exponent: int, *, denominator: bool = False, joint: bool = False):
    arms = ("control", "variant-a", "variant-b")
    rows = []
    outcome_exponent = 0 if denominator else scale_exponent
    for arm_index, arm in enumerate(arms):
        for index in range(60):
            sign = -1 if index % 2 else 1
            row = {
                "unit_id": f"{arm}-{index:03d}",
                "group_id": arm,
                "x": index / 59 + arm_index * 0.1,
                "category": ("north", "south", "west")[index % 3],
                "y": (3.0 + arm_index * 0.8 + sign * (0.1 + index / 1000))
                * math.ldexp(1.0, outcome_exponent),
                "y_den": 2.0 + arm_index * 0.1 + sign * (0.1 + index / 1000),
            }
            if denominator or joint:
                row["y_den"] *= math.ldexp(1.0, scale_exponent)
            rows.append(row)
    return rows


def _assert_projection_scaled(actual, reference, factor: float):
    for arm in actual.arm_order[1:]:
        for getter, scale_factor in (
            (lambda result, arm=arm: result.absolute(arm), factor),
            (lambda result, arm=arm: result.relative(arm), 1.0),
        ):
            observed = getter(actual)
            expected = getter(reference)
            assert math.isclose(observed.dof, expected.dof, rel_tol=1e-12)
            for field in ("estimate", "se", "lower", "upper"):
                assert math.isclose(
                    getattr(observed, field),
                    getattr(expected, field) * scale_factor,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ), (arm, field, observed, expected, scale_factor)


def _assert_ratio_projection_scaled(
    actual, reference, factor: float, *, absolute_dof_invariant: bool
):
    for arm in actual.arm_order[1:]:
        observed_absolute = actual.ratio(arm, scale="absolute")
        expected_absolute = reference.ratio(arm, scale="absolute")
        if absolute_dof_invariant:
            assert math.isclose(observed_absolute.dof, expected_absolute.dof, rel_tol=1e-12)
        absolute_fields = (
            ("estimate", "se", "lower", "upper") if absolute_dof_invariant else ("estimate", "se")
        )
        for field in absolute_fields:
            assert math.isclose(
                getattr(observed_absolute, field),
                getattr(expected_absolute, field) * factor,
                rel_tol=1e-12,
                abs_tol=0.0,
            ), (arm, "absolute", field, observed_absolute, expected_absolute, factor)

        observed_log = actual.ratio(arm, scale="log")
        expected_log = reference.ratio(arm, scale="log")
        if absolute_dof_invariant:
            assert math.isclose(observed_log.dof, expected_log.dof, rel_tol=1e-12)
        log_fields = (
            ("estimate", "se", "lower", "upper") if absolute_dof_invariant else ("estimate", "se")
        )
        for field in log_fields:
            assert math.isclose(
                getattr(observed_log, field),
                getattr(expected_log, field),
                rel_tol=1e-12,
                abs_tol=0.0,
            ), (arm, "log", field, observed_log, expected_log)


def _same_scale_projection(actual, reference, factor: float, *, ratio: bool) -> bool:
    for arm in actual.arm_order[1:]:
        projections = (
            (
                (
                    actual.ratio(arm, scale="absolute"),
                    reference.ratio(arm, scale="absolute"),
                    factor,
                ),
                (actual.ratio(arm, scale="log"), reference.ratio(arm, scale="log"), 1.0),
            )
            if ratio
            else (
                (actual.absolute(arm), reference.absolute(arm), factor),
                (actual.relative(arm), reference.relative(arm), 1.0),
            )
        )
        for observed, expected, expected_factor in projections:
            if not math.isclose(observed.dof, expected.dof, rel_tol=1e-12):
                return False
            for field in ("estimate", "se", "lower", "upper"):
                if not math.isclose(
                    getattr(observed, field),
                    getattr(expected, field) * expected_factor,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    return False
    return True


def _check_scale_equivariance(scale_exponent, tmp_path):
    arms = ("control", "variant-a", "variant-b")
    base_rows = _mixed_scale_rows(0)
    scaled_rows = _mixed_scale_rows(scale_exponent)
    factor = math.ldexp(1.0, scale_exponent)

    base_mean_local, base_mean_warehouse = _fit_pair(
        base_rows, ratio=False, tmp_path=tmp_path, arms=arms, categorical=True
    )
    mean_local, mean_warehouse = _fit_pair(
        scaled_rows, ratio=False, tmp_path=tmp_path, arms=arms, categorical=True
    )
    assert _same_scale_projection(mean_warehouse, base_mean_warehouse, factor, ratio=False)
    assert _same_scale_projection(mean_local, base_mean_local, factor, ratio=False)
    assert _same_scale_projection(mean_warehouse, mean_local, 1.0, ratio=False)

    base_ratio_local, base_ratio_warehouse = _fit_pair(
        base_rows, ratio=True, tmp_path=tmp_path, arms=arms, categorical=True
    )
    assert _same_scale_projection(base_ratio_warehouse, base_ratio_local, 1.0, ratio=True)

    numerator_local, numerator_warehouse = _fit_pair(
        scaled_rows, ratio=True, tmp_path=tmp_path, arms=arms, categorical=True
    )
    _assert_ratio_projection_scaled(
        numerator_warehouse, base_ratio_warehouse, factor, absolute_dof_invariant=False
    )
    assert _same_scale_projection(numerator_warehouse, numerator_local, 1.0, ratio=True)

    denominator_local, denominator_warehouse = _fit_pair(
        _mixed_scale_rows(scale_exponent, denominator=True),
        ratio=True,
        tmp_path=tmp_path,
        arms=arms,
        categorical=True,
    )
    inverse_factor = math.ldexp(1.0, -scale_exponent)
    _assert_ratio_projection_scaled(
        denominator_warehouse,
        base_ratio_warehouse,
        inverse_factor,
        absolute_dof_invariant=False,
    )
    assert _same_scale_projection(denominator_warehouse, denominator_local, 1.0, ratio=True)

    joint_local, joint_warehouse = _fit_pair(
        _mixed_scale_rows(scale_exponent, joint=True),
        ratio=True,
        tmp_path=tmp_path,
        arms=arms,
        categorical=True,
    )
    _assert_ratio_projection_scaled(
        joint_warehouse, base_ratio_warehouse, 1.0, absolute_dof_invariant=True
    )
    assert _same_scale_projection(joint_warehouse, joint_local, 1.0, ratio=True)


@pytest.mark.parametrize("scale_exponent", [-167, 0, 163])
def test_warehouse_qr_scale_equivariance_smoke(scale_exponent, tmp_path):
    _check_scale_equivariance(scale_exponent, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("scale_exponent", [-167, -100, 0, 100, 163])
def test_warehouse_qr_scale_equivariance_sweep(scale_exponent, tmp_path):
    _check_scale_equivariance(scale_exponent, tmp_path)


def test_out_of_range_same_sign_means_refuse(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": index / 59,
            "y": 1e308 * (0.9 + 0.05 * arm_index + 0.002 * index) + 1e305 * math.sin(index),
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=False, tmp_path=tmp_path, component="outcome", bound="upper"
    )


def test_warehouse_qr_ratio_absolute_is_unavailable_when_se_overflows(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": (1 + 0.01 * arm_index) * 1e190 + (-1 if index % 2 else 1) * 1e200,
            "y_den": 1e-110,
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=True, tmp_path=tmp_path, component="numerator", bound="upper"
    )


def test_warehouse_qr_support_transfers_do_not_reconstruct_denominator_extreme(
    tmp_path,
):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": 3 + 0.2 * arm_index + 0.05 * math.sin(index),
            "y_den": 8.0 if index == 59 else 1.0,
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    transfers = []
    scratch_statements = []
    _local, _actual = _fit_pair(
        rows,
        ratio=True,
        tmp_path=tmp_path,
        transfers=transfers,
        scratch_statements=scratch_statements,
    )
    numeric = [
        float(value)
        for row in transfers
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0
    ]
    assert all(not math.isclose(value, 8.0, rel_tol=1e-12, abs_tol=1e-12) for value in numeric)
    assert all(
        not math.isclose(left / right, 8.0, rel_tol=1e-12, abs_tol=1e-12)
        for left in numeric
        for right in numeric
        if right != 0
    )
    assert scratch_statements
    import sqlglot
    from sqlglot import exp

    for sql in scratch_statements:
        statement = sqlglot.parse_one(sql, read="duckdb")
        assert statement.args.get("limit") is None
        for projection in statement.expressions:
            alias = projection.alias
            assert alias not in {"_qr_den_max", "_qr_den_mean", "_qr_mean_max"}, sql
            selected = projection.this if isinstance(projection, exp.Alias) else projection
            if type(selected).__name__ in {"AnyValue", "First", "Max", "Median", "Min"}:
                raise AssertionError(sql)


def test_warehouse_qr_refuses_wide_arm_magnitudes(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": (3 + 0.05 * math.sin(index)) * (1e-200 if arm_index == 0 else 1e200),
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=False, tmp_path=tmp_path, component="outcome", bound="lower"
    )


def test_warehouse_qr_refuses_out_of_range_ratio_outcome(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": (3 + 0.2 * arm_index + 0.05 * math.sin(index)) * 1e200,
            "y_den": 1e-100,
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=True, tmp_path=tmp_path, component="numerator", bound="upper"
    )


def test_subnormal_outcome_scores_refuse_outside_range(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": (3 + arm_index + (-1 if index % 2 else 1)) * 1e-322,
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=False, tmp_path=tmp_path, component="outcome", bound="lower"
    )


def test_nonfinite_refusal_precedes_magnitude_refusal():
    from increment.errors import CodedError
    from increment.query._cuped_cells import require_warehouse_magnitude_range

    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "y": (
                float("inf")
                if arm == "control" and index == 0
                else 1e100
                if arm == "control" and index == 1
                else 3.0
            ),
        }
        for arm in ARMS
        for index in range(60)
    ]
    connection = ibis.duckdb.connect()
    connection.create_table("nonfinite_input", obj=rows)
    try:
        with pytest.raises(CodedError) as error:
            require_warehouse_magnitude_range(
                connection.table("nonfinite_input"),
                metric="probe",
                components=(("outcome", "y"),),
                execute=connection.to_pyarrow,
            )
        assert error.value.code == "estimation.cuped_linear.nonfinite"
        assert error.value.context == {"name": "y", "row": 0}
    finally:
        connection.disconnect()


def test_near_maximum_ratio_magnitude_refuses_before_fit(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 0.0,
            "y": (1 + 0.01 * arm_index) * 1e290 + (-1 if index % 2 else 1) * 9.4e299,
            "y_den": 1e-9,
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(
        rows, ratio=True, tmp_path=tmp_path, component="numerator", bound="upper"
    )


def test_proportional_ratio_components_refuse_out_of_range_magnitude(tmp_path):
    rows = []
    for arm_index, arm in enumerate(ARMS):
        for index in range(60):
            x = 1 + 2 * arm_index + index / 59
            denominator = (
                1e-100 * (2.5 - x) + 5e-110 if arm_index == 0 else 3e-100 - 1e-100 * (x - 2.5)
            )
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": x,
                    "y": math.ldexp(denominator, 1000),
                    "y_den": denominator,
                }
            )
    _assert_magnitude_refusal(
        rows, ratio=True, tmp_path=tmp_path, component="numerator", bound="upper"
    )


def test_warehouse_qr_refuses_out_of_range_continuous_covariate(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 1e100 + index,
            "y": 3.0 + arm_index + 0.1 * math.sin(index),
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range(60)
    ]
    _assert_magnitude_refusal(rows, ratio=False, tmp_path=tmp_path, component="x", bound="upper")


def test_artifact_assigned_breakout_uses_single_canonical_escape(tmp_path):
    from increment import Analysis
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_data
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    units = matrix_data.linear_cuped_rows(n_per_arm=120)
    for row in units:
        row["region"] = "__null__" if row["region"] == "east" else None
    connection = ibis.duckdb.connect()
    connection.create_table("reserved_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "assigned_reserved.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("reserved_events", breakout=True))
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="assigned_roster_artifacts")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    try:
        expected = {row.dimension_value for row in native.run_breakout()}
        observed = {row.dimension_value for row in adopted.run_breakout()}
        assert observed == expected
    finally:
        adopted.close()
        native.close()
        connection.disconnect()
