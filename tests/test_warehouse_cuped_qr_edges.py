from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import ibis
import pytest

from increment.errors import CodedError
from increment.estimation.engine import Method
from increment.frame import CupedCovariate
from increment.query._cuped_qr import encode_qr_covariates, fit_warehouse_qr
from increment.query.cuped import _ResolvedCupedRequest, fit_multicovariate_source
from tests.test_warehouse_cuped_qr import _scratch_session

pytestmark = pytest.mark.integration

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
def test_qr_extreme_scales_refuse_out_of_range_rms(case, tmp_path):
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
    con = ibis.duckdb.connect()
    con.create_table("probe_input", obj=rows)
    con.create_table("scratch_source", obj=[{"x": 1.0}])
    session = _scratch_session(con, tmp_path)
    request = _ResolvedCupedRequest(
        identity=("scale-parity", case), control=ARMS[0], arm_order=ARMS, method=method
    )

    transfers = []

    def execute(expression):
        result = con.to_pyarrow(expression)
        transfers.extend(result.to_pylist())
        return result

    expected_component, expected_bound = {
        "overflow_mean": ("outcome", "upper"),
        "cancelling_extremes": ("outcome", "upper"),
        "overflow_constant_covariate": ("x", "upper"),
        "tiny_relative": ("outcome", "lower"),
        "score_1e-200": ("outcome", "lower"),
        "score_1e200": ("outcome", "upper"),
        "tiny_den_support": ("numerator", "lower"),
        "large_den_support": ("denominator", "upper"),
        "zero_score": ("outcome", "lower"),
        "wide_ratio": ("numerator", "upper"),
        "original_optional_absolute": ("numerator", "upper"),
    }[case]
    try:
        with pytest.raises(CodedError) as actual_error:
            with session.qr_scratch() as scratch:
                fit_warehouse_qr(
                    con.table("probe_input"),
                    metric=metric,
                    request=request,
                    execute=execute,
                    scratch=scratch,
                )
        assert actual_error.value.code == "query.cuped.magnitude_range"
        assert actual_error.value.context["component"] == expected_component
        assert actual_error.value.context["bound"] == expected_bound
        assert {row["group_id"] for row in transfers} == set(ARMS)
        assert all(
            all(name == "group_id" or name.startswith("_magnitude_") for name in row)
            for row in transfers
        )
        assert all(not {"unit_id", "x", "y", "y_den"} & set(row) for row in transfers)
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "route",
    ["native", pytest.param("artifact", marks=pytest.mark.slow)],
)
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
        assert not isinstance(fit, CodedError)
        assert cast(Any, fit).arm_order == ARMS
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
