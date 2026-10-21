from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import ibis
import pytest

from increment.errors import CapabilityError, CodedError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.engine import Method
from increment.frame import CupedCovariate
from increment.query._cuped_qr import fit_warehouse_qr
from increment.query.cuped import _ResolvedCupedRequest
from tests.test_warehouse_cuped_qr import _scratch_session

pytestmark = pytest.mark.integration

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
    statement_log=None,
    missing_policy=None,
    sql_nan: bool = False,
    local_failure=None,
):
    groups = [str(row["group_id"]) for row in rows]
    x_values = [float(row["x"]) for row in rows]
    y_values = [float(row["y"]) for row in rows]
    covariates = {"x": x_values}
    kinds = {"x": "continuous"}
    declarations = [CupedCovariate(name="x", kind="continuous", missing=missing_policy or "refuse")]
    if categorical:
        covariates["category"] = [str(row["category"]) for row in rows]
        kinds["category"] = "categorical"
        declarations.append(CupedCovariate(name="category", kind="categorical"))
    local_error = None
    try:
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
                missing={"x": missing_policy} if missing_policy else None,
            )
        else:
            local = fit_cuped_multicovariate(
                y_values,
                groups,
                control=arms[0],
                arm_order=arms,
                covariates=covariates,
                kinds=kinds,
                missing={"x": missing_policy} if missing_policy else None,
            )
    except Exception as error:
        if local_failure is None or getattr(error, "code", None) != local_failure:
            raise
        local, local_error = None, error
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

    def record_scratch_sql(backend, name, expression=None, *args, **kwargs):
        if str(name).startswith("_cuped_qr_") and expression is not None:
            sql = ibis.to_sql(expression, dialect="duckdb")
            if scratch_statements is not None:
                scratch_statements.append(sql)
            if statement_log is not None:
                statement_log.append(sql)
        if expression is None:
            return original_create(backend, name, *args, **kwargs)
        return original_create(backend, name, expression, *args, **kwargs)

    type(connection).create_table = record_scratch_sql
    original_drop = type(connection).drop_table

    def record_scratch_drop(backend, name, *args, **kwargs):
        if str(name).startswith("_cuped_qr_") and statement_log is not None:
            statement_log.append("")
        return original_drop(backend, name, *args, **kwargs)

    type(connection).drop_table = record_scratch_drop

    def execute(expression):
        if statement_log is not None:
            statement_log.append(ibis.to_sql(expression, dialect="duckdb"))
        result = connection.to_pyarrow(expression)
        if transfers is not None:
            transfers.extend(result.to_pylist())
        return result

    try:
        with session.qr_scratch() as scratch:
            relation = connection.table("probe_input")
            if sql_nan:
                relation = relation.mutate(x=ibis.literal(float("nan")).cast("float64"))
            actual = fit_warehouse_qr(
                relation,
                metric=SimpleNamespace(name="probe", type="ratio" if ratio else "mean"),
                request=request,
                execute=execute,
                scratch=scratch,
            )
        return (local_error if local_error is not None else local), actual
    finally:
        type(connection).create_table = original_create
        type(connection).drop_table = original_drop
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


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("scale_exponent", [-167, 0, 163])
def test_warehouse_qr_scale_equivariance_smoke(scale_exponent, tmp_path):
    _check_scale_equivariance(scale_exponent, tmp_path)


@pytest.mark.integration
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
                components=(("outcome", "outcome", "y"),),
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


@pytest.mark.integration
@pytest.mark.slow
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


def test_warehouse_qr_p4_three_arm_fit_has_bounded_statements_and_sql(tmp_path):
    rows = _mixed_scale_rows(0)
    categories = ("north", "south", "west", "east")
    for index, row in enumerate(rows):
        row["category"] = categories[index % len(categories)]
    statements = []
    _fit_pair(
        rows,
        ratio=False,
        tmp_path=tmp_path,
        arms=("control", "variant-a", "variant-b"),
        categorical=True,
        statement_log=statements,
    )
    phases = {
        "magnitude": ("_magnitude",),
        "encoder/metadata": ("_qr_covariate", "_qr_category", "_qr_scale"),
        "factor stages": ("_qr_dot_", "_qr_v2", "_qr_alpha"),
        "staging": ("_qr_inverse", "_qr_q_"),
        "leverage": ("_qr_h", "_qr_residual"),
        "HC2": ("_qr_hc2",),
        "projections": ("_qr_projection_score",),
        "BM": ("_qr_bm_",),
    }
    counts = {
        phase: sum(any(term in str(sql).lower() for term in terms) for sql in statements)
        for phase, terms in phases.items()
    }
    assert len(statements) < 60, (
        f"statements={len(statements)}, max_sql={max(map(len, statements))}, phases={counts}"
    )
    assert max(map(len, statements)) < 200_000
    bm_statements = [sql for sql in statements if "_qr_bm_" in str(sql).lower()]
    assert bm_statements
    assert max(map(len, bm_statements)) <= 80_000


@pytest.mark.parametrize("offset", [1.0e12, 1.0e15])
@pytest.mark.parametrize("ratio", [False, True])
def test_relative_projection_preserves_small_effect_at_large_offset(offset, ratio, tmp_path):
    rows = []
    for arm, shift in (("control", 0.0), ("treatment", 1.0)):
        for index in range(64):
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": float(index % 11),
                    "y": offset + shift + float(index % 7),
                    "y_den": 1.0 + 0.01 * math.sin(index),
                }
            )
    local, actual = _fit_pair(rows, ratio=ratio, tmp_path=tmp_path)
    for expected_projection, observed_projection in zip(
        local._projections, actual._projections, strict=True
    ):
        assert observed_projection[4] == pytest.approx(expected_projection[4], rel=1e-8, abs=1e-18)
        if observed_projection[0] == "relative":
            assert observed_projection[4] != 0.0


def test_arm_confounded_constant_covariate_refuses_like_local_fit(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 1.0 if arm == "control" else 2.0,
            "y": float(index + (arm == "treatment")),
        }
        for arm in ARMS
        for index in range(64)
    ]
    with pytest.raises(CodedError) as error:
        _fit_pair(
            rows,
            ratio=False,
            tmp_path=tmp_path,
            local_failure="estimation.cuped_linear.rank_deficiency",
        )
    assert error.value.code == "estimation.cuped_linear.rank_deficiency"


def test_globally_constant_covariate_is_dropped_like_local_fit(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": 1.0,
            "y": float(index + (arm == "treatment")),
        }
        for arm in ARMS
        for index in range(64)
    ]
    local, actual = _fit_pair(rows, ratio=False, tmp_path=tmp_path)
    assert actual.theta == pytest.approx(local.theta)
    assert actual.adjusted_means == pytest.approx(local.adjusted_means)


def test_categorical_cardinality_refuses_before_transferring_labels(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{category}-{repeat}",
            "group_id": arm,
            "x": float(category + repeat),
            "y": float(category % 9 + repeat),
            "category": f"level-{category:03d}",
        }
        for arm in ARMS
        for category in range(201)
        for repeat in range(2)
    ]
    transfers = []
    with pytest.raises(CapabilityError) as error:
        _fit_pair(
            rows,
            ratio=False,
            categorical=True,
            tmp_path=tmp_path,
            transfers=transfers,
            local_failure="estimation.cuped_linear.support_region",
        )
    assert error.value.code == "query.cuped.general_qr_required"
    assert all("category" not in row for row in transfers)


@pytest.mark.integration
@pytest.mark.slow
def test_globally_constant_covariate_parity_across_native_artifact_and_local(tmp_path):
    from increment import Analysis
    from increment.estimation.results import LiftEstimate
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=64)
    for unit in units:
        unit["pre_revenue"] = 1.0
    connection = ibis.duckdb.connect()
    connection.create_table("constant_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "constant_covariate.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("constant_events"))
    native = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="constant_qr_artifacts")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = native.publish_unit_day_artifact(store, extensions=extensions)
    artifact = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)
    try:
        expected = {
            row.method: row for row in cast(list[LiftEstimate], local.run(metrics=["revenue"]))
        }
        for analysis in (native, artifact):
            actual = {
                row.method: row
                for row in cast(list[LiftEstimate], analysis.run(metrics=["revenue"]))
            }
            assert actual.keys() == expected.keys()
            for method, row in actual.items():
                actual_lift, expected_lift = row.lift, expected[method].lift
                assert actual_lift is not None and expected_lift is not None
                assert actual_lift.value == pytest.approx(expected_lift.value)
    finally:
        local.close()
        artifact.close()
        native.close()
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("offset", [1.0e12, 1.0e15])
def test_native_readout_relative_lift_matches_local_at_large_offset(offset, tmp_path):
    from increment import Analysis
    from increment.estimation.results import LiftEstimate
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=64)
    for unit in units:
        unit["revenue"] = offset + (1.0 if unit["variant"] == "treatment" else 0.0)
    connection = ibis.duckdb.connect()
    connection.create_table("qr_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "large_offset.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("qr_events"))
    warehouse = Analysis.from_definitions("mixed_cuped", definitions_path, connection)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)
    try:
        actual = cast(list[LiftEstimate], warehouse.run(metrics=["revenue"]))
        expected = cast(list[LiftEstimate], local.run(metrics=["revenue"]))
        observed = {row.method: row for row in actual}["linear"]
        reference = {row.method: row for row in expected}["linear"]
        assert observed.lift is not None and reference.lift is not None
        for field in ("value", "lb", "ub"):
            assert getattr(observed.lift, field) == pytest.approx(
                getattr(reference.lift, field), rel=1e-6, abs=1e-18
            )
        assert observed.lift.value != 0.0
    finally:
        warehouse.close()
        local.close()
        connection.disconnect()


@pytest.mark.parametrize("ratio", [False, True])
def test_far_apart_positive_means_match_local_relative_projection(ratio, tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": float(index % 13),
            "y": ((1.0 if arm == "control" else 1.0e-20) * (1.0 + 0.1 * math.sin(index))),
            "y_den": 1.0 + 0.05 * math.cos(index),
        }
        for arm in ARMS
        for index in range(64)
    ]
    local, actual = _fit_pair(rows, ratio=ratio, tmp_path=tmp_path)
    for expected, observed in zip(local._projections, actual._projections, strict=True):
        assert observed[4] == pytest.approx(expected[4], rel=1e-8, abs=1e-12)


def test_all_nan_imputed_covariate_matches_local_constant_drop(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": float("nan"),
            "y": float(index % 7 + (arm == "treatment")),
        }
        for arm in ARMS
        for index in range(64)
    ]
    local, actual = _fit_pair(
        rows,
        ratio=False,
        tmp_path=tmp_path,
        missing_policy="impute-indicator",
        sql_nan=True,
    )
    assert actual.theta == pytest.approx(local.theta)
    assert actual.adjusted_means == pytest.approx(local.adjusted_means)


def test_ratio_projection_preserves_adjustment_contrast_at_large_means(tmp_path):
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": float(u) + (0.02 if arm == "treatment" else 0.0),
            "y": 1.0e15 + u + v,
            "y_den": 1.0,
        }
        for arm in ARMS
        for index in range(64)
        for u, v in ((0, -1), (0, 1), (1, -1), (1, 1))[index % 4 : index % 4 + 1]
    ]
    local, actual = _fit_pair(rows, ratio=True, tmp_path=tmp_path)
    ratio_projections = [
        (expected, observed)
        for expected, observed in zip(local._projections, actual._projections, strict=True)
        if expected[0] == "ratio"
    ]
    assert len(ratio_projections) == 4
    for expected, observed in ratio_projections:
        assert observed[4] == pytest.approx(expected[4], rel=1e-8, abs=1e-18)
