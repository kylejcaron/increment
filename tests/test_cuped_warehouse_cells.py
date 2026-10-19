from __future__ import annotations

import json
from dataclasses import asdict

import ibis
import numpy as np
import pandas as pd
import pytest

from increment.errors import CapabilityError
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.cuped.cells import fit_categorical_cells
from increment.semantics.models import CupedCovariate, MeanMetric, Measure, RatioMetric

pytestmark = pytest.mark.integration


def _cells_frame(
    counts,
    *,
    ratio=False,
    zero_scatter=False,
    denominator_constant=False,
    label_width=1,
    arm_width=1,
):
    rows = []
    for a, arm in enumerate(("c", "t")):
        for pattern, count in enumerate(counts):
            for i in range(count):
                noise = 0.0 if zero_scatter else np.sin(0.7 * i + a)
                rows.append(
                    {
                        "unit_id": f"{arm}-{pattern}-{i}",
                        "group_id": arm * arm_width,
                        "segment": ("A" if pattern < 2 else "B") * label_width,
                        "device": "desktop" if pattern % 2 else "mobile",
                        "y": 8.0 + a * 0.5 + pattern * 0.2 + noise,
                        **(
                            {
                                "y_den": 3.0
                                + pattern * 0.1
                                + (0.0 if denominator_constant else noise * 0.1)
                            }
                            if ratio
                            else {}
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _metric(ratio=False):
    if ratio:
        return RatioMetric(
            name="outcome",
            entity="unit",
            numerator=Measure(fact="num", aggregation="sum"),
            denominator=Measure(fact="den", aggregation="sum"),
        )
    return MeanMetric(name="outcome", entity="unit", fact="num", aggregation="sum")


def _request(arm_width=1):
    from increment.estimation.engine import Method
    from increment.query.cuped import _ResolvedCupedRequest

    return _ResolvedCupedRequest(
        identity=("snapshot", "outcome", "assigned"),
        control="c" * arm_width,
        arm_order=("c" * arm_width, "t" * arm_width),
        method=Method.cuped_multicovariate(
            covariates=[
                CupedCovariate(name="segment", kind="categorical"),
                CupedCovariate(name="device", kind="categorical"),
            ]
        ),
    )


def _produce(frame, metric=None, request=None):
    from increment.query._cuped_cells import categorical_cells_from_relation

    con = ibis.duckdb.connect()
    con.create_table("units", obj=frame)
    transfers = []

    def execute(expression):
        result = con.to_pyarrow(expression)
        transfers.append((tuple(result.column_names), result.to_pylist()))
        return result

    try:
        cells = categorical_cells_from_relation(
            con.table("units"),
            execute,
            metric=metric or _metric("y_den" in frame),
            request=request or _request(),
        )
        return cells, transfers
    finally:
        con.disconnect()


@pytest.mark.parametrize("ratio", [False, True])
@pytest.mark.parametrize(
    "arm_scales", [(1.0, 1.0), (1e-200, 1e-200), (1e155, 1e155), (1e-200, 1e155)]
)
def test_categorical_warehouse_eligible_cells_preserve_original_unit_inference(ratio, arm_scales):
    frame = _cells_frame((41, 29, 37, 33), ratio=ratio)
    scale = frame.group_id.map(dict(zip(("c", "t"), arm_scales, strict=True)))
    for component in ("y", "y_den") if ratio else ("y",):
        frame[component] *= scale
    cells, transfers = _produce(frame)
    components = ["y", "y_den"] if ratio else ["y"]
    arm_means = frame.groupby("group_id", sort=False)[components].mean().to_numpy().T
    assert np.asarray(cells.outcome_references) == pytest.approx(arm_means, rel=1e-14, abs=0)
    fit = fit_categorical_cells(cells, control="c")
    expected = fit_cuped_multicovariate(
        None if ratio else frame.y.to_numpy(),
        frame.group_id.to_numpy(),
        control="c",
        arm_order=("c", "t"),
        covariates={name: frame[name].to_numpy() for name in ("segment", "device")},
        kinds={"segment": "categorical", "device": "categorical"},
        numerator=frame.y.to_numpy() if ratio else None,
        denominator=frame.y_den.to_numpy() if ratio else None,
    )
    assert np.asarray(fit.adjusted_means) == pytest.approx(
        np.asarray(expected.adjusted_means), rel=1e-10, abs=0
    )
    for name in ("theta", "covariance", "max_leverage", "mean_leverage", "leverage_trace"):
        assert np.asarray(getattr(fit, name)) == pytest.approx(
            np.asarray(getattr(expected, name)), rel=1e-10, abs=1e-12
        )
    for actual, reference in zip(fit._projections, expected._projections, strict=True):
        assert actual[:4] == reference[:4]
        assert actual[4:] == pytest.approx(reference[4:], rel=1e-10, abs=0.0, nan_ok=True)
    assert len(transfers) == 2
    assert len(transfers[0][1]) == 1
    assert len(transfers[1][1]) == 8
    bound = transfers[0][1][0]["encoded_bytes"]
    assert len(json.dumps(asdict(cells), ensure_ascii=True).encode()) <= bound <= 8 * 1024 * 1024


def test_categorical_warehouse_reference_preserves_cancelling_extreme_outcomes():
    frame = pd.DataFrame(
        {
            "unit_id": [f"{arm}-{i:02d}" for arm in ("c", "t") for i in range(20)],
            "group_id": np.repeat(["c", "t"], 20),
            "segment": "A",
            "device": "mobile",
            "y": np.concatenate(([1e155, -1e155], np.ones(18), [1e155, -1e155], np.full(18, 2.0))),
        }
    )
    cells, _ = _produce(frame)
    fit = fit_categorical_cells(cells, control="c")
    expected = fit_cuped_multicovariate(
        frame.y.to_numpy(),
        frame.group_id.to_numpy(),
        control="c",
        arm_order=("c", "t"),
        covariates={name: frame[name].to_numpy() for name in ("segment", "device")},
        kinds={"segment": "categorical", "device": "categorical"},
    )
    assert np.asarray(fit.adjusted_means) == pytest.approx(
        np.asarray(expected.adjusted_means), rel=1e-10, abs=0
    )
    for actual, reference in zip(
        (fit.absolute("t"), fit.relative("t")),
        (expected.absolute("t"), expected.relative("t")),
        strict=True,
    ):
        for field in ("estimate", "se", "dof", "lower", "upper"):
            assert getattr(actual, field) == pytest.approx(
                getattr(reference, field), rel=1e-10, abs=0
            )


def test_categorical_warehouse_first_sums_retain_survivors_beside_large_terms():
    def arm(offset):
        return np.concatenate(([1e12], offset + 0.1 * np.arange(18), [-1e12]))

    frame = pd.DataFrame(
        {
            "unit_id": [f"{arm}-{i:02d}" for arm in ("c", "t") for i in range(20)],
            "group_id": np.repeat(["c", "t"], 20),
            "segment": "A",
            "device": "mobile",
            "y": np.concatenate((arm(1 / 3), arm(2 / 3))),
        }
    )
    cells, _ = _produce(frame)
    fit = fit_categorical_cells(cells, control="c")
    expected = fit_cuped_multicovariate(
        frame.y.to_numpy(),
        frame.group_id.to_numpy(),
        control="c",
        arm_order=("c", "t"),
        covariates={name: frame[name].to_numpy() for name in ("segment", "device")},
        kinds={"segment": "categorical", "device": "categorical"},
    )
    assert np.asarray(fit.adjusted_means) == pytest.approx(
        np.asarray(expected.adjusted_means), rel=1e-10, abs=0
    )
    for field in ("estimate", "se", "dof"):
        assert getattr(fit.absolute("t"), field) == pytest.approx(
            getattr(expected.absolute("t"), field), rel=1e-10, abs=0
        )


def _survivor_frame(tiers):
    def arm(offset):
        extremes = [value for tier in tiers for value in (tier, -tier)]
        values = offset + 0.25 * np.arange(20 - len(extremes))
        return np.concatenate(([extremes[0]], values, extremes[1:]))

    return pd.DataFrame(
        {
            "unit_id": [f"{arm}-{i:02d}" for arm in ("c", "t") for i in range(20)],
            "group_id": np.repeat(["c", "t"], 20),
            "segment": "A",
            "device": "mobile",
            "y": np.concatenate((arm(1.0), arm(2.0))),
        }
    )


def test_categorical_warehouse_sums_survivors_beside_cancelling_extremes_exactly():
    frame = _survivor_frame((1.2345678901234567e155,))
    cells, _ = _produce(frame)
    fit = fit_categorical_cells(cells, control="c")
    expected = fit_cuped_multicovariate(
        frame.y.to_numpy(),
        frame.group_id.to_numpy(),
        control="c",
        arm_order=("c", "t"),
        covariates={name: frame[name].to_numpy() for name in ("segment", "device")},
        kinds={"segment": "categorical", "device": "categorical"},
    )
    assert np.asarray(fit.adjusted_means) == pytest.approx(
        np.asarray(expected.adjusted_means), rel=1e-10, abs=0
    )
    for field in ("estimate", "se", "dof"):
        assert getattr(fit.absolute("t"), field) == pytest.approx(
            getattr(expected.absolute("t"), field), rel=1e-10, abs=0
        )


def test_categorical_warehouse_unresolvable_summation_range_refuses_before_transfer():
    from increment.query._cuped_cells import categorical_cells_from_relation

    con = ibis.duckdb.connect()
    con.create_table("units", obj=_survivor_frame((1.2345678901234567e155, 1e78)))
    transfers = []

    def execute(expression):
        result = con.to_pyarrow(expression)
        transfers.append(tuple(result.column_names))
        return result

    try:
        with pytest.raises(CapabilityError) as raised:
            categorical_cells_from_relation(
                con.table("units"), execute, metric=_metric(), request=_request()
            )
    finally:
        con.disconnect()
    assert raised.value.code == "query.cuped.general_qr_required"
    assert raised.value.context["reason"] == "component_range"
    assert raised.value.context["arm"] == "c"
    assert raised.value.context["component"] == "outcome"
    route = raised.value.context["route"]
    assert isinstance(route, str) and "from_unit_summary" in route
    assert len(transfers) == 1
    assert not any(name.startswith(("ref_", "first_", "cross_")) for name in transfers[0])


def test_categorical_encoded_bound_accounts_for_long_arm_labels():
    frame = _cells_frame((41, 29, 37, 33), arm_width=10_000)
    cells, transfers = _produce(frame, request=_request(10_000))
    bound = transfers[0][1][0]["encoded_bytes"]
    assert len(json.dumps(asdict(cells), ensure_ascii=True).encode()) <= bound <= 8 * 1024 * 1024


@pytest.mark.parametrize("ratio", [False, True])
def test_categorical_warehouse_magnitudes_match_local_evidence(ratio):
    from increment.estimation.cuped.cells import categorical_cells

    frame = _cells_frame((41, 29, 37, 33), ratio=ratio)
    cells, _ = _produce(frame)
    local = categorical_cells(
        None if ratio else frame.y.to_numpy(),
        frame.group_id.to_numpy(),
        control="c",
        arm_order=("c", "t"),
        covariates={name: frame[name].to_numpy() for name in ("segment", "device")},
        numerator=frame.y.to_numpy() if ratio else None,
        denominator=frame.y_den.to_numpy() if ratio else None,
        identity=("snapshot", "outcome", "assigned"),
    )
    assert cells.outcome_exponents == local.outcome_exponents
    assert np.asarray(cells.outcome_magnitudes) == pytest.approx(
        np.asarray(local.outcome_magnitudes), rel=1e-14, abs=0
    )


@pytest.mark.parametrize(
    ("counts", "options", "reason"),
    [
        ((999, 1, 1, 999), {}, "component_contributors"),
        ((998, 2, 2, 998), {}, "component_contributors"),
        ((40, 40, 40, 40), {"zero_scatter": True}, "component_scatter"),
        ((40, 40, 40, 40), {"ratio": True, "denominator_constant": True}, "component_scatter"),
        ((3, 3, 3, 3), {}, "compression"),
        ((40, 40, 40, 40), {"label_width": 100_000}, "encoded_bytes"),
        ((40, 40, 40, 40), {"arm_width": 100_000}, "encoded_bytes"),
    ],
)
def test_categorical_warehouse_ineligible_request_transfers_no_outcome_cells(
    counts, options, reason
):
    from increment.query._cuped_cells import categorical_cells_from_relation

    con = ibis.duckdb.connect()
    con.create_table("units", obj=_cells_frame(counts, **options))
    transfers = []

    def execute(expression):
        result = con.to_pyarrow(expression)
        transfers.append((tuple(result.column_names), result.num_rows))
        return result

    try:
        with pytest.raises(CapabilityError) as raised:
            categorical_cells_from_relation(
                con.table("units"),
                execute,
                metric=_metric(options.get("ratio", False)),
                request=_request(options.get("arm_width", 1)),
            )
        assert raised.value.code == "query.cuped.general_qr_required"
        assert raised.value.context["reason"] == reason
        route = raised.value.context["route"]
        assert isinstance(route, str)
        assert "from_unit_summary" in route
        assert len(transfers) == 1
        assert transfers[0][1] == 1
        assert all(not name.startswith(("ref_", "first_", "cross_")) for name in transfers[0][0])
    finally:
        con.disconnect()


@pytest.mark.parametrize("column_type", ["bigint", "boolean", "decimal(18, 4)"])
def test_categorical_warehouse_non_floating_outcome_column_refuses_before_execution(column_type):
    from increment.query._cuped_cells import categorical_cells_from_relation

    con = ibis.duckdb.connect()
    con.create_table("units", obj=_cells_frame((40, 40, 40, 40)))

    def execute(expression):
        pytest.fail("Known discrete domains must refuse before executing warehouse queries")

    try:
        relation = con.sql(
            f"select unit_id, group_id, segment, device, cast(y > 8 as {column_type}) as y from units"
            if column_type == "boolean"
            else f"select unit_id, group_id, segment, device, cast(y as {column_type}) as y from units"
        )
        with pytest.raises(CapabilityError) as raised:
            categorical_cells_from_relation(relation, execute, metric=_metric(), request=_request())
        assert raised.value.context["reason"] == "outcome_domain"
    finally:
        con.disconnect()


def test_categorical_warehouse_discrete_domain_refuses_before_query_execution():
    from increment.query._cuped_cells import categorical_cells_from_relation

    con = ibis.duckdb.connect()
    con.create_table("units", obj=_cells_frame((40, 40, 40, 40)))

    def execute(expression):
        pytest.fail("Known discrete domains must refuse before executing warehouse queries")

    try:
        with pytest.raises(CapabilityError) as raised:
            categorical_cells_from_relation(
                con.table("units"),
                execute,
                metric=_metric().model_copy(update={"aggregation": "count"}),
                request=_request(),
            )
        assert raised.value.context["reason"] == "outcome_domain"
    finally:
        con.disconnect()


@pytest.fixture
def categorical_sources(tmp_path, request):
    from dataclasses import replace
    from typing import cast

    import pyarrow as pa
    import yaml

    from increment import Analysis
    from increment.estimation.engine import Method
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.query.source import _ArtifactFacadeSource
    from increment.semantics.loader import load
    from increment.semantics.models import CupedCovariateBinding
    from tests.analysis_factory import _moment_source, _native_source
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    discrete = getattr(request, "param", False)
    frame = _cells_frame((8, 8, 8, 8) if discrete else (41, 29, 37, 33))
    if discrete:
        frame["y"] = np.tile([0] * 7 + [1], 8)
    units = [
        {
            "user_id": row.unit_id,
            "variant": "control" if row.group_id == "c" else "treatment",
            "revenue": row.y,
            "orders": 1,
            "signups": 0,
            "pre_revenue": 1.0,
            "region": "all",
            "country": row.segment,
            "device": row.device,
            "new_user": "no",
        }
        for row in frame.itertuples()
    ]
    data = yaml.safe_load(mixed_cuped_definitions_yaml("events"))
    plan = data["experiments"][0]["plan"]
    plan.pop("guardrails")
    plan.pop("secondaries")
    declarations = [
        CupedCovariateBinding(
            name="segment", kind="categorical", property="country", source="events"
        ),
        CupedCovariateBinding(
            name="device", kind="categorical", property="device", source="events"
        ),
    ]
    method = Method.cuped_multicovariate(covariates=declarations)
    plan["primary"]["decision_method"] = method.model_dump(mode="json", exclude_none=True)
    path = tmp_path / "categorical.yaml"
    path.write_text(yaml.safe_dump(data))
    con = ibis.duckdb.connect()
    events = mixed_cuped_event_rows(units)
    if discrete:
        for event in events:
            if event["value"] is not None:
                event["value"] = int(event["value"])
    con.create_table("events", obj=pa.Table.from_pylist(events))
    if discrete:
        assert con.table("events").value.type().is_integer()
    native = Analysis.from_definitions("mixed_cuped", path, con)
    definitions = load(path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    store = WarehouseArtifactStore(con, schema_name="categorical_artifacts")
    ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(
        store, ref, expected_context=artifact_context(definitions, experiment, "error")
    )
    request = replace(
        _request(), method=method, control="control", arm_order=("control", "treatment")
    )
    native_source = _native_source(native)
    artifact_source = cast("_ArtifactFacadeSource", _moment_source(adopted))
    try:
        yield native_source, artifact_source, request
    finally:
        native.close()
        adopted.close()
        con.disconnect()


def test_categorical_source_fitter_uses_native_and_verified_artifact_relations(
    categorical_sources, monkeypatch
):
    from increment.query.cuped import fit_multicovariate_source
    from increment.query.session import WarehouseSession
    from increment.query.source import _ArtifactFacadeSource

    native_source, artifact_source, request = categorical_sources
    transferred = []
    original_fetch = WarehouseSession.fetch_arrow
    original_execute = type(artifact_source._snapshot).execute

    def fetch(session, expression, **kwargs):
        assert "unit_id" not in expression.columns
        result = original_fetch(session, expression, **kwargs)
        assert not ({"unit_id", "group_id"} <= set(result.column_names))
        transferred.append(result.num_rows)
        return result

    def execute_snapshot(snapshot, expression):
        assert "unit_id" not in expression.columns
        return original_execute(snapshot, expression)

    def no_units(*args, **kwargs):
        pytest.fail("The categorical producer must not invoke a unit-frame materialization")

    monkeypatch.setattr(WarehouseSession, "fetch_arrow", fetch)
    monkeypatch.setattr(type(artifact_source._snapshot), "execute", execute_snapshot)
    monkeypatch.setattr(type(native_source), "unit_frame", no_units)
    monkeypatch.setattr(_ArtifactFacadeSource, "unit_frame", no_units)
    left = fit_multicovariate_source(native_source, native_source.context.metrics[0], request)
    right = fit_multicovariate_source(artifact_source, artifact_source.context.metrics[0], request)
    for name in ("theta", "adjusted_means", "covariance", "dof", "max_leverage"):
        assert np.asarray(getattr(left, name)) == pytest.approx(
            np.asarray(getattr(right, name)), rel=1e-10, abs=1e-12
        )
    assert transferred and max(transferred) <= 8


@pytest.mark.parametrize("categorical_sources", [True], indirect=True)
@pytest.mark.parametrize("ingress", ["native", "artifact"])
def test_integer_fact_domain_refuses_before_categorical_transfer(
    categorical_sources, ingress, monkeypatch
):
    from increment.query.cuped import fit_multicovariate_source
    from increment.query.session import WarehouseSession

    native, artifact, request = categorical_sources
    source = native if ingress == "native" else artifact
    transferred = []
    original_fetch = WarehouseSession.fetch_arrow
    original_execute = type(artifact._snapshot).execute

    def fetch(session, expression, **kwargs):
        transferred.append(tuple(expression.columns))
        assert not any(name.startswith(("ref_", "first_", "cross_")) for name in expression.columns)
        return original_fetch(session, expression, **kwargs)

    def execute_snapshot(snapshot, expression):
        transferred.append(tuple(expression.columns))
        assert not any(name.startswith(("ref_", "first_", "cross_")) for name in expression.columns)
        return original_execute(snapshot, expression)

    monkeypatch.setattr(WarehouseSession, "fetch_arrow", fetch)
    monkeypatch.setattr(type(artifact._snapshot), "execute", execute_snapshot)
    with pytest.raises(CapabilityError) as raised:
        fit_multicovariate_source(source, source.context.metrics[0], request)
    assert raised.value.code == "query.cuped.general_qr_required"
    assert raised.value.context["reason"] == "outcome_domain"


@pytest.mark.parametrize(("segments", "eligible"), [(64, True), (65, False)])
def test_categorical_cell_budget_and_eightfold_boundary_are_enforced_in_sql(segments, eligible):
    from increment.query._cuped_cells import categorical_cells_from_relation

    frame = pd.DataFrame(
        [
            {
                "unit_id": f"{arm}-{segment}-{device}-{i}",
                "group_id": arm,
                "segment": f"s{segment}",
                "device": f"d{device}",
                "y": 5.0 + 0.01 * segment + np.sin(i),
            }
            for arm in ("c", "t")
            for segment in range(segments)
            for device in range(32)
            for i in range(8)
        ]
    )
    con = ibis.duckdb.connect()
    con.create_table("units", obj=frame)
    transfers = []

    def execute(expression):
        result = con.to_pyarrow(expression)
        transfers.append(result.num_rows)
        return result

    try:
        if eligible:
            cells = categorical_cells_from_relation(
                con.table("units"), execute, metric=_metric(), request=_request()
            )
            assert sum(sum(row) for row in cells.counts) == 32768
            assert transfers == [1, 4096]
        else:
            with pytest.raises(CapabilityError) as raised:
                categorical_cells_from_relation(
                    con.table("units"), execute, metric=_metric(), request=_request()
                )
            assert raised.value.context["reason"] == "cell_count"
            assert transfers == [1]
    finally:
        con.disconnect()


@pytest.mark.parametrize("domain", ["discrete", "bounded"])
def test_known_outcome_domain_refusal_precedes_source_acquisition(domain):
    from increment.query.cuped import fit_multicovariate_source
    from increment.semantics.models import Winsorization

    updates = (
        {"aggregation": "count"}
        if domain == "discrete"
        else {"winsorization": Winsorization(lower_value=0.0, upper_value=10.0)}
    )

    with pytest.raises(CapabilityError) as raised:
        fit_multicovariate_source(None, _metric().model_copy(update=updates), _request())
    assert raised.value.context["reason"] == "outcome_domain"
