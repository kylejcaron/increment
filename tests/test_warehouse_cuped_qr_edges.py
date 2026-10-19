from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import ibis
import numpy as np
import pytest

from increment.errors import CodedError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.engine import Method
from increment.frame import CupedCovariate
from increment.query._cuped_qr import encode_qr_covariates, fit_warehouse_qr
from increment.query.cuped import _ResolvedCupedRequest, fit_multicovariate_source
from tests.test_warehouse_cuped_qr import _scratch_session

ARMS = ("control", "treatment")


@pytest.mark.parametrize(
    "case",
    [
        "overflow_mean",
        "cancelling_extremes",
        "overflow_constant_covariate",
        "tiny_relative",
        "score_1e-200",
        "score_1e200",
        "tiny_den_support",
        "large_den_support",
        "zero_score",
        "wide_ratio",
        "original_optional_absolute",
    ],
)
def test_qr_extreme_scales_match_local_inference(case, tmp_path):  # noqa: PLR0915
    ratio = case.endswith("den_support") or case in {"wide_ratio", "original_optional_absolute"}
    rows = []
    for arm_index, arm in enumerate(ARMS):
        count = 10 if case == "cancelling_extremes" else 60
        for index in range(count):
            x = 0.0 if case == "cancelling_extremes" else index / 59
            y = 3.0 + 0.2 * arm_index + 0.4 * x + 0.05 * math.sin(index)
            if case == "overflow_mean":
                y = 1e308 * (0.9 + 0.05 * arm_index + 0.002 * index) + 1e305 * math.sin(index)
            if case == "cancelling_extremes":
                magnitudes = (1e300, -1e300, *(1.0 for _ in range(8)))
                y = (1.0 + arm_index) * magnitudes[index]
            if case == "overflow_constant_covariate":
                x = 1e308
            if case == "tiny_relative":
                y *= 1e-310
            if case == "score_1e-200":
                y *= 1e-200
            if case == "score_1e200":
                y *= 1e200
            if case == "zero_score":
                y, x = 3e-300, 0.0
            row = {"unit_id": f"{arm}-{index:03d}", "group_id": arm, "x": x, "y": y}
            if case == "tiny_den_support":
                row["y"] = 3e-300
                row["x"] = 0.0
                row["y_den"] = 6e-299 if index == 59 else 1e-302
            if case == "original_optional_absolute":
                row["y"] *= 1e150
                row["y_den"] = 1e-160 * (1 + 0.001 * math.sin(index * 0.7))
            if case == "wide_ratio":
                row["y"] *= 1e200
                row["y_den"] = 1e-200 * (1 + 0.001 * math.sin(index * 0.7))
            if case == "large_den_support":
                row["y_den"] = 1e200 * (1 + 0.02 * math.sin(index * 0.7))
            rows.append(row)

    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    metric = SimpleNamespace(name="probe", type="ratio" if ratio else "mean")
    groups = [str(row["group_id"]) for row in rows]
    covariates = {"x": [float(row["x"]) for row in rows]}
    local_error = None
    try:
        if ratio:
            local = fit_cuped_multicovariate(
                None,
                groups,
                control=ARMS[0],
                arm_order=ARMS,
                covariates=covariates,
                kinds={"x": "continuous"},
                numerator=[float(row["y"]) for row in rows],
                denominator=[float(row["y_den"]) for row in rows],
            )
        else:
            local = fit_cuped_multicovariate(
                [float(row["y"]) for row in rows],
                groups,
                control=ARMS[0],
                arm_order=ARMS,
                covariates=covariates,
                kinds={"x": "continuous"},
            )
    except CodedError as error:
        local_error = error

    con = ibis.duckdb.connect()
    con.create_table("probe_input", obj=rows)
    con.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(con, tmp_path)
    request = _ResolvedCupedRequest(
        identity=("scale-parity", case), control=ARMS[0], arm_order=ARMS, method=method
    )

    def execute(expression):
        return con.to_pyarrow(expression)

    try:
        if local_error is not None:
            with pytest.raises(CodedError) as actual_error:
                with session.qr_scratch() as scratch:
                    fit_warehouse_qr(
                        con.table("probe_input"),
                        metric=metric,
                        request=request,
                        execute=execute,
                        scratch=scratch,
                    )
            assert actual_error.value.code == local_error.code
            assert actual_error.value.context["region"] == local_error.context["region"]
            return
        with session.qr_scratch() as scratch:
            actual = fit_warehouse_qr(
                con.table("probe_input"),
                metric=metric,
                request=request,
                execute=execute,
                scratch=scratch,
            )
        if case in {"overflow_mean", "cancelling_extremes"}:
            for arm_index, arm in enumerate(ARMS):
                values = [float(row["y"]) for row in rows if row["group_id"] == arm]
                target = math.fsum(value / len(values) for value in values)
                for fit in (actual, local):
                    observed = fit.adjusted_means[arm_index]
                    assert abs(observed - target) <= 4 * math.ulp(max(abs(v) for v in values))
                    if case == "cancelling_extremes" and fit is actual:
                        assert math.isclose(observed, (0.8, 1.6)[arm_index], rel_tol=1e-12), (
                            actual.adjusted_means
                        )
            for observed, expected in zip(actual._projections, local._projections, strict=True):
                assert observed[:4] == expected[:4]
                assert tuple(math.isfinite(value) for value in observed[5:]) == tuple(
                    math.isfinite(value) for value in expected[5:]
                )
        else:
            np.testing.assert_allclose(
                actual.adjusted_means, local.adjusted_means, rtol=1e-9, atol=0
            )
            for observed, expected in zip(actual._projections, local._projections, strict=True):
                assert observed[:4] == expected[:4]
                np.testing.assert_allclose(
                    observed[4:], expected[4:], rtol=1e-7, atol=0, equal_nan=True
                )
    finally:
        session.drop_materialized()
        con.disconnect()


def test_categorical_ratio_withholds_invalid_arm_before_fit(tmp_path):
    from increment.semantics.models import Measure, RatioMetric

    arms = ("control", "variant-a", "variant-b")
    rows = [
        {
            "unit_id": f"{arm}-{index:03d}",
            "group_id": arm,
            "category": str(index % 2),
            "y": 3.0 + arm_index + 0.02 * index + 0.01 * math.sin(index),
            "y_den": (-1.0 if arm == "variant-b" else 1.0) + 0.01 * math.sin(index * 0.7),
        }
        for arm_index, arm in enumerate(arms)
        for index in range(60)
    ]
    method = Method.cuped_multicovariate(
        covariates=[CupedCovariate(name="category", kind="categorical")]
    )
    metric = RatioMetric(
        name="ratio",
        entity="unit",
        numerator=Measure(fact="num", aggregation="sum"),
        denominator=Measure(fact="den", aggregation="sum"),
    )
    request = _ResolvedCupedRequest(
        identity=("categorical-withheld",),
        control=arms[0],
        arm_order=arms,
        method=method,
    )
    con = ibis.duckdb.connect()
    con.create_table("probe_input", obj=rows)
    con.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(con, tmp_path)

    class Source:
        def cuped_outcome_domains(self, _metric):
            return ("continuous", "continuous")

        @contextmanager
        def cuped_relation(self, _metric, _request):
            yield con.table("probe_input"), con.to_pyarrow, session.qr_scratch

    try:
        evidence = fit_multicovariate_source(Source(), metric, request)
        from increment.query.cuped import _WarehouseCupedFitEvidence

        assert isinstance(evidence, _WarehouseCupedFitEvidence)
        assert dict(evidence.withheld).keys() == {"variant-b"}
        assert evidence.fit is not None
        assert evidence.fit.arm_order == ("control", "variant-a")
    finally:
        session.drop_materialized()
        con.disconnect()


@pytest.mark.parametrize("route", ["native", "artifact"])
def test_breakout_null_and_reserved_label_stay_distinct(route, tmp_path):
    import warnings

    from increment import Analysis
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load
    from tests.parity_harness import matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=120)
    for row in units:
        row["region"] = "__null__" if row["region"] == "east" else None
    con = ibis.duckdb.connect()
    con.create_table("mixed_breakout_events", obj=mixed_cuped_event_rows(units))
    path = tmp_path / "reserved.yaml"
    path.write_text(mixed_cuped_definitions_yaml("mixed_breakout_events", breakout=True))
    native = Analysis.from_definitions("mixed_cuped", path, con)
    adopted = None
    try:
        analysis = native
        if route == "artifact":
            definitions = load(path)
            experiment = definitions.experiment("mixed_cuped")
            assert experiment is not None
            context = artifact_context(definitions, experiment, "error")
            store = WarehouseArtifactStore(con, schema_name="reserved_artifacts")
            extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
            reference = native.publish_unit_day_artifact(store, extensions=extensions)
            adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
            analysis = adopted
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rows = analysis.run_breakout(metrics=["revenue"])
        cells = {row.dimension_value: row for row in rows}
        assert set(cells) == {"__null__", "\\__null__"}
        assert cells["__null__"].lift is not None
        assert cells["\\__null__"].lift is not None
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
        con.disconnect()


@pytest.mark.parametrize("unsupported_count", [1, 60])
def test_breakout_fit_uses_admitted_arm_roster(unsupported_count, tmp_path):
    rows = []
    for arm_index, arm in enumerate(("control", "treatment", "variant-b")):
        count = unsupported_count if arm == "variant-b" else 60
        for index in range(count):
            rows.append(
                {
                    "unit_id": f"{arm}-{index:03d}",
                    "group_id": arm,
                    "x": index / 59,
                    "y": -100.0
                    if arm == "variant-b" and unsupported_count > 1
                    else 3.0 + arm_index + 0.4 * index / 59 + 0.05 * math.sin(index),
                }
            )
    method = Method.cuped_multicovariate(covariates=[CupedCovariate(name="x", kind="continuous")])
    request = _ResolvedCupedRequest(
        identity=("breakout-admitted",),
        control="control",
        arm_order=ARMS,
        method=method,
        breakout_dimension="region",
        breakout_value="east",
    )
    metric = SimpleNamespace(name="probe", type="mean")
    con = ibis.duckdb.connect()
    con.create_table("probe_input", obj=rows)
    con.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(con, tmp_path)

    class Source:
        @contextmanager
        def cuped_relation(self, _metric, _request):
            yield con.table("probe_input"), con.to_pyarrow, session.qr_scratch

    try:
        fit = fit_multicovariate_source(Source(), metric, request)
        assert fit.arm_order == ARMS
    finally:
        session.drop_materialized()
        con.disconnect()


def test_negative_infinite_covariate_uses_coded_refusal():
    con = ibis.duckdb.connect()
    rows = [
        {
            "unit_id": f"{arm}-{index}",
            "group_id": arm,
            "x": float("-inf") if arm == "treatment" and index == 0 else float(index),
        }
        for arm in ARMS
        for index in range(3)
    ]
    con.create_table("probe_input", obj=rows)
    try:
        with pytest.raises(CodedError) as refused:
            encode_qr_covariates(
                con.table("probe_input"),
                [CupedCovariate(name="x", kind="continuous")],
                arms=ARMS,
                row_identity_columns=["unit_id"],
                execute=con.to_pyarrow,
            )
        assert refused.value.code == "estimation.cuped_linear.nonfinite"
        assert refused.value.context["name"] == "x"
    finally:
        con.disconnect()


def test_public_cuped_readout_preserves_nonfinite_fit_refusal(tmp_path):
    from increment import Analysis
    from tests.parity_harness import matrix_data
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    units = matrix_data.linear_cuped_rows(n_per_arm=120)
    units[0]["pre_revenue"] = float("inf")
    con = ibis.duckdb.connect()
    con.create_table("headline_failure_events", obj=mixed_cuped_event_rows(units))
    path = tmp_path / "headline_failure.yaml"
    path.write_text(mixed_cuped_definitions_yaml("headline_failure_events"))
    analysis = Analysis.from_definitions("mixed_cuped", path, con)
    try:
        (row,) = analysis.run(metrics=["revenue"])
        row = cast(Any, row)
        assert row.failure_code == "estimation.cuped_linear.nonfinite"
        assert row.failure_context["name"] == "pre_revenue"
    finally:
        analysis.close()
        con.disconnect()
