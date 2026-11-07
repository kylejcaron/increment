from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import permutations
from types import SimpleNamespace
from typing import Any, cast

import ibis
import numpy as np
import pytest
import sqlglot

from increment import View, from_definitions, from_unit_day_artifact
from increment.artifacts import publish_unit_day_artifact
from increment.errors import CapabilityError, CodedError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.cuped.multicovariate import M1_MAX_LEVERAGE_RATIO
from increment.estimation.engine import Method
from increment.frame import CupedCovariate
from increment.query._cuped_qr import fit_warehouse_qr
from increment.query.cuped import _ResolvedCupedRequest
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import view_readout
from tests.test_warehouse_cuped_qr import _scratch_session

pytestmark = pytest.mark.integration

ARMS = ("control", "treatment")
THREE_ARMS = ("control", "variant-a", "variant-b")


@contextmanager
def _scratch_capture(
    connection, *, scratch_statements, scratch_columns, statement_log
) -> Iterator[None]:
    if scratch_statements is None and scratch_columns is None and statement_log is None:
        yield
        return
    backend_type = type(connection)
    original_create = backend_type.create_table
    original_drop = backend_type.drop_table

    def record_scratch_sql(backend, name, expression=None, *args, **kwargs):
        if str(name).startswith("_cuped_qr_") and expression is not None:
            if scratch_columns is not None:
                scratch_columns.append(tuple(expression.columns))
            if scratch_statements is not None or statement_log is not None:
                sql = ibis.to_sql(expression, dialect="duckdb")
                for log in (scratch_statements, statement_log):
                    if log is not None:
                        log.append(sql)
        if expression is None:
            return original_create(backend, name, *args, **kwargs)
        return original_create(backend, name, expression, *args, **kwargs)

    def record_scratch_drop(backend, name, *args, **kwargs):
        if str(name).startswith("_cuped_qr_") and statement_log is not None:
            statement_log.append("")
        return original_drop(backend, name, *args, **kwargs)

    backend_type.create_table = record_scratch_sql
    backend_type.drop_table = record_scratch_drop
    try:
        yield
    finally:
        backend_type.create_table = original_create
        backend_type.drop_table = original_drop


class _QRWarehouse:
    """One DuckDB warehouse whose input tables stay unchanged across QR fits."""

    def __init__(self, tmp_path, tables: Mapping[str, list[dict[str, Any]]]) -> None:
        self.rows = dict(tables)
        self.connection = ibis.duckdb.connect()
        for name, rows in self.rows.items():
            self.connection.create_table(name, obj=rows)
        self.connection.create_table("scratch_source", obj=[{"x": 1.0}])
        self.session = _scratch_session(self.connection, tmp_path)

    def __enter__(self) -> _QRWarehouse:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.session.drop_materialized()
        self.connection.disconnect()

    def fit(
        self,
        table: str,
        *,
        ratio: bool,
        arms: tuple[str, ...],
        declarations: Sequence[CupedCovariate],
        transfers=None,
        scratch_statements=None,
        scratch_columns=None,
        statement_log=None,
        sql_nan: bool = False,
    ):
        connection = self.connection
        request = _ResolvedCupedRequest(
            identity=("warehouse_qr_regression",),
            control=arms[0],
            arm_order=arms,
            method=Method.cuped_multicovariate(covariates=list(declarations)),
        )

        def execute(expression):
            if statement_log is not None:
                statement_log.append(ibis.to_sql(expression, dialect="duckdb"))
            result = connection.to_pyarrow(expression)
            if transfers is not None:
                transfers.extend(result.to_pylist())
            return result

        with (
            _scratch_capture(
                connection,
                scratch_statements=scratch_statements,
                scratch_columns=scratch_columns,
                statement_log=statement_log,
            ),
            self.session.qr_scratch() as scratch,
        ):
            relation = connection.table(table)
            if sql_nan:
                relation = relation.mutate(x=ibis.literal(float("nan")).cast("float64"))
            actual = fit_warehouse_qr(
                relation,
                metric=SimpleNamespace(name="probe", type="ratio" if ratio else "mean"),
                request=request,
                execute=execute,
                scratch=scratch,
            )
        # Later fits reuse this warehouse, so no scratch relation may outlive a fit.
        assert not [name for name in connection.list_tables() if name.startswith("_cuped_qr_")]
        return actual


def _local_fit(rows, *, ratio: bool, arms: tuple[str, ...], kinds: dict[str, str], missing=None):
    covariates = {
        name: [float(row[name]) if kind == "continuous" else str(row[name]) for row in rows]
        for name, kind in kinds.items()
    }
    outcome = [float(row["y"]) for row in rows]
    return fit_cuped_multicovariate(
        None if ratio else outcome,
        [str(row["group_id"]) for row in rows],
        control=arms[0],
        arm_order=arms,
        covariates=covariates,
        kinds=kinds,
        missing=missing,
        numerator=outcome if ratio else None,
        denominator=[float(row["y_den"]) for row in rows] if ratio else None,
    )


def _probe_covariates(*, categorical: bool, missing_policy=None):
    kinds = {"x": "continuous"}
    declarations = [CupedCovariate(name="x", kind="continuous", missing=missing_policy or "refuse")]
    if categorical:
        kinds["category"] = "categorical"
        declarations.append(CupedCovariate(name="category", kind="categorical"))
    return kinds, declarations


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
    kinds, declarations = _probe_covariates(categorical=categorical, missing_policy=missing_policy)
    try:
        local = _local_fit(
            rows,
            ratio=ratio,
            arms=arms,
            kinds=kinds,
            missing={"x": missing_policy} if missing_policy else None,
        )
    except Exception as error:
        if local_failure is None or getattr(error, "code", None) != local_failure:
            raise
        local = error
    with _QRWarehouse(tmp_path, {"probe_input": rows}) as warehouse:
        actual = warehouse.fit(
            "probe_input",
            ratio=ratio,
            arms=arms,
            declarations=declarations,
            transfers=transfers,
            scratch_statements=scratch_statements,
            statement_log=statement_log,
            sql_nan=sql_nan,
        )
    return local, actual


def _scale_pair(warehouse: _QRWarehouse, table: str, *, ratio: bool):
    kinds, declarations = _probe_covariates(categorical=True)
    local = _local_fit(warehouse.rows[table], ratio=ratio, arms=THREE_ARMS, kinds=kinds)
    actual = warehouse.fit(table, ratio=ratio, arms=THREE_ARMS, declarations=declarations)
    return local, actual


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
    rows = []
    outcome_exponent = 0 if denominator else scale_exponent
    for arm_index, arm in enumerate(THREE_ARMS):
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


@backend_fixture()
def scale_warehouse(scale_exponent, tmp_path):
    with _QRWarehouse(
        tmp_path,
        {
            "base": _mixed_scale_rows(0),
            "scaled": _mixed_scale_rows(scale_exponent),
            "denominator_scaled": _mixed_scale_rows(scale_exponent, denominator=True),
            "joint_scaled": _mixed_scale_rows(scale_exponent, joint=True),
        },
    ) as warehouse:
        yield warehouse


def _check_scale_equivariance(scale_exponent, warehouse):
    factor = math.ldexp(1.0, scale_exponent)

    base_mean_local, base_mean_warehouse = _scale_pair(warehouse, "base", ratio=False)
    mean_local, mean_warehouse = _scale_pair(warehouse, "scaled", ratio=False)
    assert _same_scale_projection(mean_warehouse, base_mean_warehouse, factor, ratio=False)
    assert _same_scale_projection(mean_local, base_mean_local, factor, ratio=False)
    assert _same_scale_projection(mean_warehouse, mean_local, 1.0, ratio=False)

    base_ratio_local, base_ratio_warehouse = _scale_pair(warehouse, "base", ratio=True)
    assert _same_scale_projection(base_ratio_warehouse, base_ratio_local, 1.0, ratio=True)

    numerator_local, numerator_warehouse = _scale_pair(warehouse, "scaled", ratio=True)
    _assert_ratio_projection_scaled(
        numerator_warehouse, base_ratio_warehouse, factor, absolute_dof_invariant=False
    )
    assert _same_scale_projection(numerator_warehouse, numerator_local, 1.0, ratio=True)

    denominator_local, denominator_warehouse = _scale_pair(
        warehouse, "denominator_scaled", ratio=True
    )
    inverse_factor = math.ldexp(1.0, -scale_exponent)
    _assert_ratio_projection_scaled(
        denominator_warehouse,
        base_ratio_warehouse,
        inverse_factor,
        absolute_dof_invariant=False,
    )
    assert _same_scale_projection(denominator_warehouse, denominator_local, 1.0, ratio=True)

    joint_local, joint_warehouse = _scale_pair(warehouse, "joint_scaled", ratio=True)
    _assert_ratio_projection_scaled(
        joint_warehouse, base_ratio_warehouse, 1.0, absolute_dof_invariant=True
    )
    assert _same_scale_projection(joint_warehouse, joint_local, 1.0, ratio=True)


# Overflow and underflow show at the extremes; power-of-two scaling is exact in between.
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("scale_exponent", [-167, 163])
def test_warehouse_qr_scale_equivariance_sweep(scale_exponent, scale_warehouse):
    _check_scale_equivariance(scale_exponent, scale_warehouse)


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


@pytest.mark.slow
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
    native_experiment, native_resolver = from_definitions(
        definitions_path, connection, experiment_name="mixed_cuped"
    )
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="assigned_roster_artifacts")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=extensions
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, reference, expected_context=context
    )
    try:
        expected = {
            row.dimension_value
            for row in view_readout(
                native_experiment, native_resolver, View(view="breakout", grain="total")
            )
        }
        observed = {
            row.dimension_value
            for row in view_readout(
                adopted_experiment, adopted_resolver, View(view="breakout", grain="total")
            )
        }
        assert observed == expected
    finally:
        adopted_resolver.close()
        cast(Any, native_resolver).close()
        connection.disconnect()


@pytest.mark.slow
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


@pytest.mark.slow
def test_compact_three_arm_ratio_cuped_batches_match_local_fit(tmp_path):
    names = tuple(f"x{index}" for index in range(34))
    # Inference needs p / n_min within the M1 leverage-ratio bound; 170 per arm is the minimum.
    units_per_arm = 170
    assert len(names) / units_per_arm <= M1_MAX_LEVERAGE_RATIO < len(names) / (units_per_arm - 1)
    rng = np.random.default_rng(20261021)
    rows = []
    for arm_index, arm in enumerate(THREE_ARMS):
        for index in range(units_per_arm):
            values = rng.normal(size=len(names))
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "y": 12.0 + arm_index * 0.4 + 0.7 * values[0] + rng.normal(scale=0.3),
                    "y_den": float(5 + index % 4),
                    **dict(zip(names, values.tolist(), strict=True)),
                }
            )
    local = _local_fit(rows, ratio=True, arms=THREE_ARMS, kinds=dict.fromkeys(names, "continuous"))
    transfers, scratch_columns = [], []
    with _QRWarehouse(tmp_path, {"probe_input": rows}) as warehouse:
        actual = warehouse.fit(
            "probe_input",
            ratio=True,
            arms=THREE_ARMS,
            declarations=[CupedCovariate(name=name, kind="continuous") for name in names],
            transfers=transfers,
            scratch_columns=scratch_columns,
        )

    assert actual.p == len(names) == 34
    # Three arms give six ordered contrasts; mirrors share, so each of the three pairs
    # needs six BM jobs (absolute, relative, two log-ratio and two absolute-ratio
    # components): 18 jobs. A job carries 2 * (3 + 34) + 1 = 75 columns, so the
    # 1,200-column budget holds 15 jobs beside group_id: one full and one partial batch.
    feature_relations = [
        columns for columns in scratch_columns if "group_id" in columns and "t0" in columns
    ]
    assert len(feature_relations) == 2
    assert all(len(columns) <= 1200 for columns in feature_relations)
    jobs = [sum(name.startswith("t") for name in columns) for columns in feature_relations]
    assert jobs[-1] < jobs[0]
    assert transfers
    assert not any("unit_id" in row for row in transfers)
    for arm, comparison in permutations(THREE_ARMS, 2):
        projections = {
            "absolute": (
                actual.absolute(arm, control=comparison),
                local.absolute(arm, control=comparison),
            ),
            **{
                f"ratio-{scale}": (
                    actual.ratio(arm, control=comparison, scale=scale),
                    local.ratio(arm, control=comparison, scale=scale),
                )
                for scale in ("log", "absolute")
            },
        }
        for projection, (observed, expected) in projections.items():
            for field in ("estimate", "se", "dof"):
                assert getattr(observed, field) == pytest.approx(
                    getattr(expected, field), rel=1e-9, abs=1e-11
                ), (arm, comparison, projection, field)


@pytest.mark.slow
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


def test_warehouse_fitted_scatter_matches_local_fit(tmp_path):
    rng = np.random.default_rng(2604)
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "x": float(1.0e6 + rng.normal(5.0, 2.0) + 0.3 * arm_index),
            "category": ("a", "b", "c")[index % 3],
            "y": float(rng.normal(10.0, 1.0)),
        }
        for arm_index, arm in enumerate(ARMS)
        for index in range((70, 90)[arm_index])
    ]
    local, actual = _fit_pair(rows, ratio=False, tmp_path=tmp_path, categorical=True)
    for warehouse, retained in zip(
        actual.within_arm_scatter, local.within_arm_scatter, strict=True
    ):
        assert np.asarray(warehouse) == pytest.approx(np.asarray(retained), rel=1e-9)
    warehouse_balance = actual.covariate_balance("treatment")
    local_balance = local.covariate_balance("treatment")
    assert warehouse_balance is not None and local_balance is not None
    assert warehouse_balance.statistic == pytest.approx(local_balance.statistic, rel=1e-9)


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
    from increment import from_definitions, from_unit_day_artifact
    from increment.artifacts import publish_unit_day_artifact
    from increment.estimation.results import LiftEstimate
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.analysis_factory import view_readout
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=64)
    for unit in units:
        unit["pre_revenue"] = 1.0
    connection = ibis.duckdb.connect()
    connection.create_table("constant_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "constant_covariate.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("constant_events"))
    native = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(connection, schema_name="constant_qr_artifacts")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    reference = publish_unit_day_artifact(*native, store, extensions=extensions)
    artifact = from_unit_day_artifact(store, reference, expected_context=context)
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)
    try:
        expected = {
            row.method: row
            for row in cast(list[LiftEstimate], view_readout(*local, metrics=["revenue"]))
        }
        for bound in (native, artifact):
            actual = {
                row.method: row
                for row in cast(list[LiftEstimate], view_readout(*bound, metrics=["revenue"]))
            }
            assert actual.keys() == expected.keys()
            for method, row in actual.items():
                actual_lift, expected_lift = row.lift, expected[method].lift
                assert actual_lift is not None and expected_lift is not None
                assert actual_lift.value == pytest.approx(expected_lift.value)
    finally:
        local[1].close()
        artifact[1].close()
        cast(Any, native[1]).close()
        connection.disconnect()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("offset", [1.0e12, 1.0e15])
def test_native_readout_relative_lift_matches_local_at_large_offset(offset, tmp_path):
    from increment import from_definitions
    from increment.estimation.results import LiftEstimate
    from tests.analysis_factory import view_readout
    from tests.parity_harness import matrix_cases, matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=64)
    for unit in units:
        unit["revenue"] = offset + (1.0 if unit["variant"] == "treatment" else 0.0)
    connection = ibis.duckdb.connect()
    connection.create_table("qr_events", obj=mixed_cuped_event_rows(units))
    definitions_path = tmp_path / "large_offset.yaml"
    definitions_path.write_text(mixed_cuped_definitions_yaml("qr_events"))
    warehouse = from_definitions(definitions_path, connection, experiment_name="mixed_cuped")
    local = matrix_cases.linear_cuped_frame_case(units, panel=False)
    try:
        actual = cast(list[LiftEstimate], view_readout(*warehouse, metrics=["revenue"]))
        expected = cast(list[LiftEstimate], view_readout(*local, metrics=["revenue"]))
        observed = {row.method: row for row in actual}["linear"]
        reference = {row.method: row for row in expected}["linear"]
        assert observed.lift is not None and reference.lift is not None
        for field in ("value", "lb", "ub"):
            assert getattr(observed.lift, field) == pytest.approx(
                getattr(reference.lift, field), rel=1e-6, abs=1e-18
            )
        assert observed.lift.value != 0.0
    finally:
        cast(Any, warehouse[1]).close()
        local[1].close()
        connection.disconnect()


@pytest.mark.slow
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


@pytest.mark.slow
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


@backend_fixture(scope="module")
def three_arm_ratio_fit(tmp_path_factory):
    """A categorical three-arm ratio fit and every DuckDB statement it issued."""
    statements: list[str] = []
    _, declarations = _probe_covariates(categorical=True)
    rows = _mixed_scale_rows(0, denominator=True)
    with _QRWarehouse(tmp_path_factory.mktemp("qr_literals"), {"probe_input": rows}) as warehouse:
        fit = warehouse.fit(
            "probe_input",
            ratio=True,
            arms=THREE_ARMS,
            declarations=declarations,
            statement_log=statements,
        )
    return fit, tuple(sql for sql in statements if sql)


def test_qr_float_literals_reach_duckdb_as_the_doubles_they_name(three_arm_ratio_fit):
    _, statements = three_arm_ratio_fit
    named: dict[str, float] = {}
    for sql in statements:
        for node in sqlglot.parse_one(sql, read="duckdb").find_all(sqlglot.exp.Literal):
            parent = node.parent
            if node.is_string and isinstance(parent, sqlglot.exp.Cast):
                if parent.to.is_type("double") and math.isfinite(float(node.this)):
                    named[parent.sql("duckdb")] = float(node.this)
            elif node.is_number and not node.this.isdigit():
                literal = parent if isinstance(parent, sqlglot.exp.Neg) else node
                named[literal.sql("duckdb")] = float(literal.sql("duckdb"))
    texts = sorted(named)
    connection = ibis.duckdb.connect()
    try:
        read = {}
        for start in range(0, len(texts), 500):
            chunk = texts[start : start + 500]
            row = connection.raw_sql(
                "SELECT " + ", ".join(f"CAST(1 AS DOUBLE) * ({text})" for text in chunk)
            ).fetchone()
            read.update(zip(chunk, row, strict=True))
    finally:
        connection.disconnect()

    assert named
    assert {text: value for text, value in read.items() if value != named[text]} == {}


def test_qr_mirrored_contrasts_have_identical_uncertainty(three_arm_ratio_fit):
    fit, _ = three_arm_ratio_fit
    uncertainty = {
        (kind, scale, arm, comparison): (se, dof)
        for kind, arm, comparison, scale, _estimate, se, dof in fit._projections
    }
    mirrored = {
        (kind, scale, arm, comparison): uncertainty[(kind, scale, comparison, arm)]
        for kind, scale, arm, comparison in uncertainty
    }

    assert mirrored == uncertainty
