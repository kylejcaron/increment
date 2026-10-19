from __future__ import annotations

import math
from types import SimpleNamespace

import ibis
import numpy as np

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
):
    groups = [str(row["group_id"]) for row in rows]
    x_values = [float(row["x"]) for row in rows]
    y_values = [float(row["y"]) for row in rows]
    covariates = {"x": x_values}
    if ratio:
        local = fit_cuped_multicovariate(
            None,
            groups,
            control=ARMS[0],
            arm_order=ARMS,
            covariates=covariates,
            kinds={"x": "continuous"},
            numerator=y_values,
            denominator=[float(row["y_den"]) for row in rows],
        )
    else:
        local = fit_cuped_multicovariate(
            y_values,
            groups,
            control=ARMS[0],
            arm_order=ARMS,
            covariates=covariates,
            kinds={"x": "continuous"},
        )
    connection = ibis.duckdb.connect()
    connection.create_table("probe_input", obj=rows)
    connection.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(connection, tmp_path)
    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    request = _ResolvedCupedRequest(
        identity=("warehouse_qr_regression",),
        control=ARMS[0],
        arm_order=ARMS,
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


def test_warehouse_qr_retains_same_sign_1e308_means(tmp_path):
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
    local, actual = _fit_pair(rows, ratio=False, tmp_path=tmp_path)
    np.testing.assert_allclose(actual.adjusted_means, local.adjusted_means, rtol=1e-12, atol=0)
    for observed, expected in zip(actual._projections, local._projections, strict=True):
        assert observed[:4] == expected[:4]
        np.testing.assert_allclose(observed[4:], expected[4:], rtol=1e-8, atol=0, equal_nan=True)


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
    local, actual = _fit_pair(rows, ratio=True, tmp_path=tmp_path)
    expected = [item for item in local._projections if item[0] == "ratio" and item[3] == "absolute"]
    observed = [
        item for item in actual._projections if item[0] == "ratio" and item[3] == "absolute"
    ]
    assert all(not math.isfinite(item[5]) for item in expected), expected
    assert all(not math.isfinite(item[5]) for item in observed), (expected, observed)


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


def test_warehouse_qr_relative_scores_survive_wide_arm_scales(tmp_path):
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
    local, actual = _fit_pair(rows, ratio=False, tmp_path=tmp_path)
    assert math.isclose(
        actual.relative("treatment").se, local.relative("treatment").se, rel_tol=1e-8
    )


def test_warehouse_qr_constant_denominator_preserves_representable_ratio_se(tmp_path):
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
    local, actual = _fit_pair(rows, ratio=True, tmp_path=tmp_path)
    observed = actual.ratio("treatment", scale="absolute")
    expected = local.ratio("treatment", scale="absolute")
    assert math.isfinite(expected.se) and expected.se > 0, expected
    assert math.isclose(observed.se, expected.se, rel_tol=1e-8), (expected, observed)


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
