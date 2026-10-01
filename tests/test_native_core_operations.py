"""Direct behavior tests for definitions-backed native core operations."""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest

from increment import Analysis
from increment.errors import CapabilityError
from increment.estimation.engine import Method
from increment.sources import MomentsSource
from tests.analysis_factory import _native_source


def _assert_operation_refusal(
    error: CapabilityError,
    *,
    operation: str,
    request: dict[str, object],
    offered: object,
) -> None:
    """Required operation fields and transport preserve the refusal contract."""
    assert type(error) is CapabilityError
    assert error.code == "source.native.operation"
    assert error.context["route"]
    assert error.context["operation"] == operation
    assert error.context["request"] == request
    assert error.context["offered"] == offered
    for copied in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
        assert type(copied) is CapabilityError
        assert copied.code == error.code
        assert copied.context == error.context
        assert str(copied) == str(error)


def test_triggered_counts_are_the_complete_triggered_workflow(tmp_path):
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(tmp_path, _events(n_per_arm=200, trigger_rate=0.2))
    source = _native_source(analysis)
    try:
        grain, counts, unit_counts = source.triggered_counts()
        result = analysis.srm(
            expected={"C": 0.5, "T": 0.5},
            population="triggered",
        )
    finally:
        analysis.close()

    assert grain == "unit"
    assert unit_counts == {}
    assert counts == result.observed
    assert sum(counts.values()) < 400


def test_moments_source_preserves_catalog(con):
    analysis = Analysis.from_definitions(
        "new_onboarding_v2",
        "examples/definitions/",
        con,
        store="none",
    )
    source = _native_source(analysis)
    try:
        selected = tuple(source.context.metrics[:2])
        moments = source.moments_source(metrics=selected, population="assigned")
        assert isinstance(moments, MomentsSource)
        assert tuple(metric.name for metric in moments.context.metrics) == tuple(
            metric.name for metric in source.context.metrics
        )
        for metric in selected:
            assert moments.moments(metric)

    finally:
        analysis.close()


@pytest.mark.slow
def test_plan_alpha_narrows_the_reported_absolute_interval() -> None:
    """A compiled plan resolved at construction is the level every readout
    answers under: a wider plan alpha reports a narrower interval."""
    import ibis

    from examples._seed import EXPERIMENT_ID, seed_event_log
    from increment.semantics.loader import load
    from tests.analysis_factory import lift_rows, make_analysis

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=600, seed=4, with_pre_period=True)
    defs = load("examples/definitions")
    experiment = defs.experiment(EXPERIMENT_ID)
    assert experiment is not None

    widths = {}
    for alpha in (0.05, 0.40):
        analysis = make_analysis(
            con,
            defs,
            experiment=experiment,
            plan=experiment.plan.model_copy(update={"alpha": alpha}),
        )
        row = next(r for r in lift_rows(analysis.run()) if r.metric == "purchase_rate")
        assert row.abs_ub is not None and row.abs_lb is not None
        widths[alpha] = row.abs_ub - row.abs_lb

    assert widths[0.40] < widths[0.05]


@pytest.mark.slow
def test_call_wide_cuped_override_reaches_every_metric(seeded_pre_period_con, seeded_defs):
    from tests.analysis_factory import lift_rows

    analysis = Analysis.from_definitions(
        "new_onboarding_v2",
        seeded_defs,
        seeded_pre_period_con,
        store="none",
    )
    try:
        unadjusted = lift_rows(analysis.run())
        results = lift_rows(
            analysis.run(decision_method=Method(name="cuped", variance_reduction="cuped"))
        )
    finally:
        analysis.close()

    assert results
    assert {result.method for result in results} == {"cuped"}
    # The override reads the pre-period covariate the metric bindings do not
    # declare: the adjusted intervals are no wider than the same data's
    # unadjusted ones, and at least one is genuinely tighter.
    baseline = {(row.metric, row.group_id): row for row in unadjusted}
    adjusted = {(row.metric, row.group_id): row for row in results}
    assert adjusted.keys() == baseline.keys()
    se_pairs: list[tuple[float, float]] = []
    for key, row in adjusted.items():
        base_se = baseline[key].abs_se
        if row.abs_se is None or base_se is None:
            continue
        se_pairs.append((row.abs_se, base_se))
    assert len(se_pairs) == len(adjusted)
    assert all(adjusted_se <= base_se for adjusted_se, base_se in se_pairs)
    assert any(adjusted_se < base_se for adjusted_se, base_se in se_pairs)


def test_moments_source_refuses_a_clustered_native_source(tmp_path):
    """Cluster transport refuses before reads, including on a closed backend."""
    import ibis

    from tests.test_analysis_cluster import _DEFS_TEMPLATE

    definitions = tmp_path / "definitions.yaml"
    definitions.write_text(_DEFS_TEMPLATE.format(cluster_line="    cluster: store_id"))
    con = ibis.duckdb.connect()
    analysis = Analysis.from_definitions("store_test", definitions, con)
    source = _native_source(analysis)
    con.disconnect()
    try:
        with pytest.raises(CapabilityError) as raised:
            source.moments_source(metrics=source.context.metrics)
    finally:
        analysis.close()

    for error in (
        raised.value,
        pickle.loads(pickle.dumps(raised.value)),
        copy.deepcopy(raised.value),
    ):
        assert type(error) is CapabilityError
        assert error.code == "source.moments.cluster_grain"
        assert error.context["operation"] == "moments_source"
        assert error.context["source"] == "native"
        assert error.context["cluster"] == "store_id"
        assert error.context["design"] == "randomized"
        assert error.context["route_forward"]


def test_triggered_counts_without_a_declared_trigger_keeps_the_null_trigger(con):
    """The refused population names the experiment and its actual, absent trigger."""
    analysis = Analysis.from_definitions(
        "new_onboarding_v2",
        "examples/definitions/",
        con,
        store="none",
    )
    source = _native_source(analysis)
    try:
        with pytest.raises(CapabilityError) as raised:
            source.triggered_counts()
    finally:
        analysis.close()

    _assert_operation_refusal(
        raised.value,
        operation="triggered_counts",
        request={"experiment": "new_onboarding_v2", "population": "triggered", "trigger": None},
        offered=("assigned",),
    )


_PARTIAL_QUERY_EXPOSURE_DEFS_YAML = """
fact_sources:
  - name: purchases
    sql: SELECT * FROM purchase_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: value
exposures:
  - name: enrollment_sql
    sql: SELECT user_id AS unit_id, first_exposure_ts AS ts FROM enrollment_q
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: enrollment_sql
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
"""


def test_query_exposure_refusal_names_missing_and_returned_columns(tmp_path):
    """A query exposure without an assignment column refuses with the columns
    enrollment needs, the missing subset, and the columns the query returned."""
    import datetime as dt

    import ibis

    defs_path = Path(tmp_path) / "defs.yaml"
    defs_path.write_text(_PARTIAL_QUERY_EXPOSURE_DEFS_YAML)
    con = ibis.duckdb.connect()
    con.create_table(
        "purchase_events",
        obj=[{"user_id": "u1", "ts": dt.datetime(2025, 1, 2, 9), "value": 10.0}],
    )
    con.create_table(
        "enrollment_q",
        obj=[{"user_id": "u1", "first_exposure_ts": dt.datetime(2025, 1, 1, 9)}],
    )
    analysis = Analysis.from_definitions("exp", defs_path, con)
    try:
        with pytest.raises(CapabilityError) as raised:
            analysis.allocation_history()
    finally:
        analysis.close()

    _assert_operation_refusal(
        raised.value,
        operation="exposure_events",
        request={
            "experiment": "exp",
            "exposure": "enrollment_sql",
            "required": ("group_id", "ts", "unit_id"),
            "missing": ("group_id",),
        },
        offered=("ts", "unit_id"),
    )


def test_export_moments_is_source_owned_and_portable(con, tmp_path):
    analysis = Analysis.from_definitions(
        "new_onboarding_v2",
        "examples/definitions/",
        con,
        store="none",
    )
    source = _native_source(analysis)
    path = Path(tmp_path) / "native-moments.parquet"
    try:
        source.export_moments(path)
    finally:
        analysis.close()

    table = pq.read_table(path)
    assert table.schema.metadata[b"increment.moments_format"] == b"8"
    rows = table.to_pylist()
    assert rows
    assert {row["moments_format"] for row in rows} == {8}
    payloads = {json.dumps(json.loads(row["assignment_counts"]), sort_keys=True) for row in rows}
    assert payloads == {json.dumps({"control": 2, "treatment": 2}, sort_keys=True)}


def test_panel_and_summary_sql_are_lazy(con, monkeypatch):
    analysis = Analysis.from_definitions(
        "new_onboarding_v2",
        "examples/definitions/",
        con,
        store="none",
    )

    def no_query(*args: Any, **kwargs: Any):
        raise AssertionError("SQL introspection must not execute a warehouse query")

    monkeypatch.setattr(con, "to_pyarrow", no_query)
    try:
        panel = analysis.panel_sql()
        summary = analysis.summary_sql()
        expected = [metric.name for metric in analysis.metrics]
    finally:
        analysis.close()

    assert set(panel) == set(expected)
    assert set(summary) == set(expected)
    assert all(sql.strip() for sql in [*panel.values(), *summary.values()])


def _cluster_conflict_events() -> list[dict[str, object]]:
    import datetime as dt

    return [
        # u1: admitted, in-window, two conflicting cluster labels (Task 3's
        # s9 -> s1 fixture) -- the malformed source this gate exists for.
        {
            "user_id": "u1",
            "event_at": dt.datetime(2025, 1, 2, 9),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "treatment",
            "store_id": "s9",
            "revenue": None,
        },
        {
            "user_id": "u1",
            "event_at": dt.datetime(2025, 1, 2, 10),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "treatment",
            "store_id": "s1",
            "revenue": None,
        },
        {
            "user_id": "u1",
            "event_at": dt.datetime(2025, 1, 3, 9),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
            "revenue": 5.0,
        },
        # u2: admitted, single clean label.
        {
            "user_id": "u2",
            "event_at": dt.datetime(2025, 1, 2, 9),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "control",
            "store_id": "s2",
            "revenue": None,
        },
        {
            "user_id": "u2",
            "event_at": dt.datetime(2025, 1, 3, 9),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
            "revenue": 4.0,
        },
        # u3: an out-of-window exposure with a conflicting label must be
        # excluded by the enrollment-window scope before the uniqueness
        # check ever sees it -- admission policy unchanged, unit stays
        # cleanly admitted under its single in-window label s3.
        {
            "user_id": "u3",
            "event_at": dt.datetime(2024, 12, 25, 9),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "control",
            "store_id": "s99",
            "revenue": None,
        },
        {
            "user_id": "u3",
            "event_at": dt.datetime(2025, 1, 2, 9),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "control",
            "store_id": "s3",
            "revenue": None,
        },
        {
            "user_id": "u3",
            "event_at": dt.datetime(2025, 1, 3, 9),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
            "revenue": 4.5,
        },
        # u4: mixed assignment (both arms), each with its own cluster
        # label -- excluded from admission under on_mixed_assignment=
        # "exclude" before the uniqueness check ever sees it either.
        {
            "user_id": "u4",
            "event_at": dt.datetime(2025, 1, 2, 9),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "control",
            "store_id": "s4",
            "revenue": None,
        },
        {
            "user_id": "u4",
            "event_at": dt.datetime(2025, 1, 2, 10),
            "event": "exposure",
            "experiment_id": "native_cluster",
            "group_id": "treatment",
            "store_id": "s44",
            "revenue": None,
        },
        {
            "user_id": "u4",
            "event_at": dt.datetime(2025, 1, 3, 9),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "store_id": None,
            "revenue": 6.0,
        },
    ]


_CLUSTER_CONFLICT_DEFS_YAML = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM native_cluster_conflict_events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - name: exposure
        column: null
      - name: purchase
        column: revenue
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: native_cluster
    exposure: enrolled
    unit: user_id
    cluster: store_id
    start: 2025-01-01
    plan: {secondaries: [revenue]}
    control_group: control
"""


def _cluster_conflict_analysis(tmp_path) -> tuple[Analysis, Any]:
    import ibis

    con = ibis.duckdb.connect()
    con.create_table("native_cluster_conflict_events", obj=_cluster_conflict_events())
    definitions = tmp_path / "defs.yaml"
    definitions.write_text(_CLUSTER_CONFLICT_DEFS_YAML)
    analysis = Analysis(
        "native_cluster", definitions, con, store="none", on_mixed_assignment="exclude"
    )
    return analysis, con


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_native_cluster_conflict_refuses_before_readout(tmp_path) -> None:
    """Raw exposure rows carrying two non-null cluster labels for one unit
    must refuse before a clustered readout or artifact publication can
    silently pick one. Out-of-window and excluded mixed-assignment rows
    keep their existing admission policy and never trigger the refusal."""
    from increment.errors import CodedError
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import AssignmentCountsRequest

    analysis, con = _cluster_conflict_analysis(tmp_path)
    with pytest.raises(CodedError) as readout_error:
        analysis.run()
    assert readout_error.value.code == "query.integrity.cluster_conflict"

    store = WarehouseArtifactStore(con, schema_name="native_cluster_conflict_artifact")
    with pytest.raises(CodedError) as publish_error:
        analysis.publish_unit_day_artifact(store, extensions=[AssignmentCountsRequest()])
    assert publish_error.value.code == "query.integrity.cluster_conflict"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cluster_identity_extension_alone_refuses_a_raw_label_conflict(tmp_path) -> None:
    """Publishing ONLY a `cluster_identity` extension (no `assignment_counts`)
    must still hit the raw uniqueness gate: `_cluster_extension_spec` reads
    deduplicated exposures directly, so without its own call into
    `_validate_cluster_labels`, two conflicting raw labels for one unit would
    silently collapse to the first-seen label instead of refusing."""
    from increment.errors import CodedError
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import ClusterIdentityRequest

    analysis, con = _cluster_conflict_analysis(tmp_path)
    cluster = cast("str", analysis.experiment.cluster)
    store = WarehouseArtifactStore(con, schema_name="cluster_identity_only_artifact")
    with pytest.raises(CodedError) as publish_error:
        analysis.publish_unit_day_artifact(
            store, extensions=[ClusterIdentityRequest(cluster_name=cluster)]
        )
    assert publish_error.value.code == "query.integrity.cluster_conflict"


def test_compliance_design_mismatch_refusal_survives_transport(con):
    from increment.errors import InvalidRequestError
    from increment.semantics.design import Encouragement
    from tests.test_analysis_asof import _analysis_with_asof_encouragement_events

    analysis = _analysis_with_asof_encouragement_events(con)
    source = _native_source(analysis)
    expected = source.context.design
    assert isinstance(expected, Encouragement)
    received = Encouragement.model_validate(
        {**expected.model_dump(), "allocation": {"control": 0.25, "treatment": 0.75}}
    )
    try:
        with pytest.raises(InvalidRequestError) as caught:
            source.compliance_summary(received)
    finally:
        analysis.close()

    error = caught.value
    assert error.code == "source.native.compliance_design_mismatch"
    received_context = error.context["received"]
    assert isinstance(received_context, Mapping)
    assert ("allocation", {"control": 0.25, "treatment": 0.75}) in received_context.items()
    for copied in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
        assert type(copied) is InvalidRequestError
        assert copied.code == error.code
        assert copied.context == error.context
