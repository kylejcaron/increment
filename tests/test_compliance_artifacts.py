"""Real in-memory native/artifact transport for the declared uptake cohort."""

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import ibis
import pyarrow as pa
import pytest

from increment import Analysis, readouts
from increment.errors import CapabilityError, CodedError, IncrementRuntimeWarning
from increment.estimation.encouragement import estimate_compliance
from increment.frame import from_unit_panel
from increment.query.artifact_publish import artifact_context
from increment.query.artifact_reader import ArtifactMomentSource
from increment.query.session import WarehouseArtifactStore
from increment.query.source import open_artifact
from increment.semantics.models import Definitions
from increment.sources import MomentsSource
from tests.analysis_factory import _native_source, lift_rows, make_analysis
from tests.test_compliance_summary import design
from tests.warning_codes import warning_codes

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class ClickTiming:
    """Uptake window and click/exposure timing knobs for `native_fixture`."""

    uptake_window: int = 7
    boundary_click_offsets: list[timedelta] | None = None
    exposure_hour: int = 0


def native_fixture(
    order=("full", "drop"),
    *,
    clustered=False,
    separate_uptake=False,
    triggered=False,
    staggered=False,
    outcome=(8, "sum", 7),
    timing=None,
    open_ended=False,
    outcome_state="present",
    warehouse_connection=None,
    declared=False,
):
    timing = timing or ClickTiming()
    uptake_window = timing.uptake_window
    boundary_click_offsets = timing.boundary_click_offsets
    start = datetime(2025, 1, 1, timing.exposure_hour, tzinfo=UTC)
    outcome_day, full_aggregation, outcome_window = outcome
    events = []
    panel = []
    pairs = [(2, 0), (4, 1), (7, 5), (12, 11)] * 3 if clustered else [(20, 10)]
    for group in ("control", "treatment"):
        for g, (size, n_uptake) in enumerate(pairs):
            for i in range(size):
                unit = f"{group}{g}_{i}"
                event = {
                    "user_id": unit,
                    "experiment_id": "uptake_test",
                    "group_id": group,
                    "cluster_id": f"{group}{g}",
                    "revenue": None,
                }
                events.append(
                    dict(
                        event,
                        event="exposed",
                        ts=start + timedelta(days=2 if staggered and i % 2 else 0),
                    )
                )
                if triggered and i < 8:
                    events.append(dict(event, event="triggered", ts=start))
                # Half of qualifying takers act at day zero, half at day six.
                takes = group == "treatment" and i < n_uptake
                click_day = (0 if i % 2 == 0 else 6) if takes else 7
                click_ts = (
                    start + boundary_click_offsets[i]
                    if boundary_click_offsets is not None
                    and group == "treatment"
                    and i < len(boundary_click_offsets)
                    else start + timedelta(days=click_day)
                )
                if group == "treatment":
                    events.append(dict(event, event="clicked", ts=click_ts))
                    # Repeated and pre-exposure clicks never change binary uptake.
                    events.append(dict(event, event="clicked", ts=start - timedelta(days=1)))
                y = 2.0 + (i % 2) + 2 * takes
                events.append(
                    dict(
                        event,
                        event="full_event",
                        ts=start + timedelta(days=outcome_day),
                        revenue=y,
                    )
                )
                if i < 9 or i == 10:
                    events.append(
                        dict(event, event="drop_event", ts=start + timedelta(days=8), revenue=y)
                    )
                for day in range(10):
                    panel.append(
                        {
                            "unit": unit,
                            "arm": group,
                            "ds": start.date() + timedelta(days=day),
                            "exposed": start.date(),
                            "clicked": float(group == "treatment" and day == click_day),
                            "y": y if day == 8 else 0.0,
                        }
                    )
    if warehouse_connection is None:
        con = ibis.duckdb.connect()
        con.create_table("events", obj=pa.Table.from_pylist(events))
    else:
        con = warehouse_connection(events)
    if outcome_state == "absent":
        con.raw_sql("DELETE FROM events WHERE event IN ('full_event', 'drop_event')")
    elif outcome_state == "null":
        con.raw_sql("UPDATE events SET revenue = NULL")
    definitions = Definitions.model_validate(
        {
            "dialect": con.name,
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "properties": [{"name": "revenue", "column": "revenue", "dtype": "float"}],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "triggered", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "full_event", "column": "revenue"},
                        {"name": "drop_event", "column": "revenue"},
                    ],
                }
            ],
            "exposures": [
                {"name": "enrolled", "fact": "exposed"},
                {"name": "triggered", "fact": "triggered"},
            ],
            "metrics": [
                {
                    "name": name,
                    "type": "mean",
                    "entity": "user_id",
                    "fact": f"{name}_event",
                    "aggregation": full_aggregation if name == "full" else "avg_event",
                    **({"window_days": outcome_window} if staggered else {}),
                    **(
                        {"filters": [{"property": "revenue", "op": "gt", "values": [1000]}]}
                        if outcome_state == "filtered"
                        else {}
                    ),
                }
                for name in order
            ],
            "experiments": [
                {
                    "name": "uptake_test",
                    "exposure": "enrolled",
                    "trigger": "triggered" if triggered else None,
                    "unit": "user_id",
                    "cluster": "cluster_id" if clustered else None,
                    "start": start,
                    "end": None if open_ended else start + timedelta(days=9),
                    "control_group": "control",
                    "plan": {
                        "secondaries": list(order),
                    },
                    **(
                        {
                            "design": {
                                "mechanism": "encouragement",
                                "uptake": {"fact": "clicked", "window_days": uptake_window},
                                "one_sided": True,
                                "exclusion_restriction": {
                                    "acknowledged": True,
                                    "justification": "Only uptake changes the outcome",
                                },
                            }
                        }
                        if declared
                        else {}
                    ),
                }
            ],
        }
    )
    if separate_uptake:
        body = definitions.model_dump(mode="python")
        body["fact_sources"] = list(body["fact_sources"])
        body["fact_sources"][0]["facts"] = [
            fact for fact in body["fact_sources"][0]["facts"] if fact["name"] != "clicked"
        ]
        body["fact_sources"].append(
            {
                "name": "uptake_events",
                "sql": "SELECT * FROM uptake_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "clicked", "column": None}],
            }
        )
        definitions = Definitions.model_validate(body)
        table = con.table("events")
        con.create_table("uptake_events", obj=table.filter(table.event == "clicked"), temp=True)
    supplied = None if declared else design(uptake_window)
    if supplied is None:
        analysis = make_analysis(con, definitions)
    else:
        analysis = make_analysis(con, definitions, _design=supplied)
    context = artifact_context(
        definitions, definitions.experiments[0], "error", encouragement_uptake=supplied
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    return con, analysis, context, store, panel


@pytest.mark.parametrize("order", [("full", "drop"), ("drop", "full"), ("drop",)])
def test_native_artifact_original_witness_and_matching_late(order):
    con, analysis, context, store, _panel = native_fixture(order)
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (_native_source(analysis), artifact):
                results = readouts.run(source, estimands=["compliance", "late"])
                comp = next(row for row in results if row.estimand == "compliance")
                assert comp.require_lift().value == pytest.approx(0.5)
                assert comp.require_lift().lb is not None
                for metric in source.context.metrics:
                    rows = {row["group_id"]: row for row in source.moments(metric)}
                    t, c = rows["treatment"], rows["control"]
                    stage = t["sum_d"] / t["n"] - c["sum_d"] / c["n"]
                    assert stage == pytest.approx(0.9 if metric.name == "drop" else 0.5)
                    itt = t["ref_y"] + t["cy1"] / t["n"] - c["ref_y"] - c["cy1"] / c["n"]
                    late = next(
                        row
                        for row in results
                        if row.metric == metric.name
                        and row.estimand == "late"
                        and row.value_scale == "absolute"
                    )
                    assert late.require_lift().value == pytest.approx(itt / stage)
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("declared", [False, True], ids=["supplied-design", "declared-design"])
def test_native_and_artifact_agree_for_each_supported_design_form(declared):
    """Both input forms keep the full typed design, every compliance row and interval,
    and one shared sequential source recipe and definition identity."""
    from increment.query.source import _artifact_source_context
    from increment.semantics.design import Encouragement
    from increment.sequential_source import sequential_definition_id

    con, analysis, context, store, _panel = native_fixture(declared=declared)
    try:
        native = _native_source(analysis)
        assert isinstance(native.context.design, Encouragement)
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            _, rehydrated = _artifact_source_context(context)
            assert rehydrated.design == native.context.design
            assert artifact.context.design == native.context.design
            keyed = []
            for source in (native, artifact):
                rows = {}
                for row in readouts.run(source, estimands=["compliance", "late"]):
                    lift = row.require_lift()
                    rows[(row.metric, row.group_id, row.estimand, row.value_scale)] = (
                        lift.value,
                        lift.lb,
                        lift.ub,
                    )
                keyed.append(rows)
            assert {key[2] for key in keyed[0]} >= {"compliance", "late"}
            assert keyed[0].keys() == keyed[1].keys()
            for key, native_values in keyed[0].items():
                assert keyed[1][key] == pytest.approx(native_values, rel=1e-9, abs=1e-12), key
            mappings = [native._sequential_observation_mapping()]
            mappings.append(artifact._sequential_observation_mapping())
            assert mappings[0] == mappings[1]
            identities = {
                sequential_definition_id(
                    source.context.metrics, source.context.design, source_mapping=mapping
                )
                for source, mapping in zip((native, artifact), mappings, strict=True)
            }
            assert len(identities) == 1
    finally:
        analysis.close()
        con.disconnect()


def test_declared_seven_day_window_and_asof_match_frame_native_and_artifact():
    con, analysis, context, store, panel = native_fixture(("full",))
    frame = from_unit_panel(
        pa.Table.from_pylist(panel),
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        exposure_date="exposed",
        metrics={"y": "mean"},
        design=design(7),
        experiment_id="uptake_test",
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            native_day = _native_source(analysis).day_source(
                metrics=_native_source(analysis).context.metrics
            )
            for day, expected in [(0, 0.25), (5, 0.25), (6, 0.5), (7, 0.5), (9, 0.5)]:
                as_of = date(2025, 1, 1) + timedelta(days=day)
                summaries = [
                    source.compliance_summary(design(7), as_of=as_of)
                    for source in (frame, _native_source(analysis), native_day, artifact)
                ]
                assert all(summary == summaries[0] for summary in summaries)
                for summary in summaries:
                    arm = summary.arm("treatment")
                    assert arm is not None
                    assert arm.uptake_total / arm.n_units == expected
            assert frame.compliance_summary(design(7)) == artifact.compliance_summary(design(7))
            with pytest.raises(CodedError) as mismatch:
                artifact.compliance_summary(design(6))
            assert mismatch.value.code == "source.compliance_summary.design_mismatch"
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("legacy_coverage", [False, True], ids=["declared", "legacy"])
def test_uptake_after_outcome_day_extends_native_and_artifact_spine(legacy_coverage):
    """A qualifying next-day uptake survives when outcomes end on exposure day."""
    con, analysis, context, store, _panel = native_fixture(
        ("full",),
        timing=ClickTiming(
            uptake_window=1,
            boundary_click_offsets=[timedelta(hours=21)] * 10,
            exposure_hour=12,
        ),
        outcome=(0, "sum", 7),
        open_ended=True,
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            if legacy_coverage:
                from increment.semantics.artifact import EncouragementUptakeExtension

                artifact._manifest = artifact.manifest.model_copy(
                    update={
                        "extensions": tuple(
                            EncouragementUptakeExtension.model_validate(
                                ext.model_dump(mode="json", exclude={"observation_edge"})
                            )
                            if ext.kind == "encouragement_uptake"
                            else ext
                            for ext in artifact.manifest.extensions
                        )
                    }
                )
            for source in (_native_source(analysis), artifact):
                metric = source.context.metrics[0]
                total = next(
                    row for row in source.moments(metric) if row["group_id"] == "treatment"
                )
                assert total["sum_d"] == pytest.approx(10.0)
                asof = {
                    row["ds"]: row
                    for row in source.moments(metric, grain="asof")
                    if row["group_id"] == "treatment"
                }
                assert asof[date(2025, 1, 2)]["sum_d"] == pytest.approx(10.0)
                assert asof[date(2025, 1, 2)]["n"] == 20
    finally:
        analysis.close()
        con.disconnect()


def test_late_uptake_preserves_calendar_average_total_scope():
    con, analysis, context, store, _panel = native_fixture(
        ("full",),
        open_ended=True,
        timing=ClickTiming(uptake_window=1),
        outcome=(1, "avg_calendar_day", 7),
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (_native_source(analysis), artifact):
                rows = {row["group_id"]: row for row in source.moments(source.context.metrics[0])}
                for arm, total in (("control", 25.0), ("treatment", 35.0)):
                    row = rows[arm]
                    assert row["n"] == 20
                    assert row["n"] * row["ref_y"] + row["cy1"] == pytest.approx(total)
    finally:
        analysis.close()
        con.disconnect()


def check_uptake_outcome_coverage(outcome_day, *, warehouse_connection=None):
    from integration.warehouse_execution._suite import _drop_probe_generation

    con, analysis, context, store, _panel = native_fixture(
        ("full",),
        open_ended=True,
        separate_uptake=True,
        staggered=True,
        timing=ClickTiming(uptake_window=1),
        outcome=(outcome_day, "avg_calendar_day", 4),
        warehouse_connection=warehouse_connection,
    )
    cleanup = ExitStack()
    cleanup.callback(con.disconnect)
    cleanup.callback(analysis.close)
    try:
        ref = analysis.publish_unit_day_artifact(store)
        cleanup.callback(_drop_probe_generation, con, store, ref, con.name, schema_name="artifacts")
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (_native_source(analysis), artifact):
                metric = source.context.metrics[0]
                rows = source.moments(metric, grain="asof")
                last = max(row["ds"] for row in rows)
                assert last == date(2025, 1, 8)
                final = {row["group_id"]: row for row in rows if row["ds"] == last}
                expected_n, expected_sums = (
                    (10, {"control": 10.0, "treatment": 15.0})
                    if outcome_day == 1
                    else (20, {"control": 20.0, "treatment": 27.5})
                )
                assert set(final) == set(expected_sums)
                for arm, expected_sum in expected_sums.items():
                    row = final[arm]
                    assert row["n"] == expected_n
                    assert row["n"] * row["ref_y"] + row["cy1"] == pytest.approx(expected_sum)
                completed = source.moments(metric, grain="asof", completed_windows_only=True)
                if outcome_day == 1:
                    assert completed == []
                else:
                    final = {row["group_id"]: row for row in completed if row["ds"] == last}
                    assert set(final) == {"control", "treatment"}
                    for arm, expected_sum in (("control", 5.0), ("treatment", 7.5)):
                        row = final[arm]
                        assert row["n"] == 10
                        assert row["n"] * row["ref_y"] + row["cy1"] == pytest.approx(expected_sum)
    finally:
        cleanup.close()


@pytest.mark.parametrize("outcome_day", [1, 3], ids=["unfinished", "mature"])
def test_uptake_coverage_cannot_supply_outcome_observations(outcome_day):
    check_uptake_outcome_coverage(outcome_day)


def test_total_and_asof_uptake_share_elapsed_timestamp_boundaries():
    offsets = [timedelta(0), timedelta(hours=21), timedelta(days=1)]
    con, analysis, context, store, _panel = native_fixture(
        ("full",),
        timing=ClickTiming(uptake_window=1, boundary_click_offsets=offsets, exposure_hour=12),
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (_native_source(analysis), artifact):
                summary = source.compliance_summary(design(1))
                treatment = summary.arm("treatment")
                assert treatment is not None
                assert treatment.n_units == 20
                assert treatment.uptake_total == pytest.approx(5.0)
                rows = {
                    row["ds"]: row
                    for row in source.moments(source.context.metrics[0], grain="asof")
                    if row["group_id"] == "treatment"
                }
                assert rows[date(2025, 1, 1)]["sum_d"] / rows[date(2025, 1, 1)][
                    "n"
                ] == pytest.approx(4 / 20)
                assert rows[date(2025, 1, 2)]["sum_d"] / rows[date(2025, 1, 2)][
                    "n"
                ] == pytest.approx(5 / 20)
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("metric_free", [False, True])
def test_new_native_artifact_export_reload_preserves_available_uncertainty(
    tmp_path, clustered, metric_free
):
    import pyarrow.parquet as pq

    con, analysis, context, store, _panel = native_fixture(
        () if metric_free else ("full",), clustered=clustered
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            native_summary = _native_source(analysis).compliance_summary(design(7))
            with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext() as rec:
                expected = estimate_compliance(native_summary, design(7)).results[0]
            if clustered:
                assert rec is not None
                assert "estimation.engine.small_total_clusters" in warning_codes(rec)
                import math

                from scipy.stats import t as student_t

                pairs = [(2, 0), (4, 1), (7, 5), (12, 11)] * 3
                k = len(pairs)
                n = sum(size for size, _ in pairs)
                rate = sum(uptake for _, uptake in pairs) / n
                variance = sum((uptake - rate * size) ** 2 for size, uptake in pairs) / (
                    k * (k - 1) * (n / k) ** 2
                )
                # Zero control uptake contributes no variance: Welch uses treatment K - 1.
                half_width = student_t.isf(0.025, k - 1) * math.sqrt(variance)
                assert expected.require_lift().value == pytest.approx(rate)
                assert expected.require_lift().lb == pytest.approx(rate - half_width)
                assert expected.require_lift().ub == pytest.approx(rate + half_width)
            for i, source in enumerate((_native_source(analysis), artifact)):
                path = tmp_path / f"cube{i}.parquet"
                source.export_moments(path)
                reloaded = MomentsSource(
                    pq.read_table(path).to_pylist(),
                    metrics=source.context.metrics,
                    study_id="uptake_test",
                    design=design(7),
                )
                with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext() as rec:
                    actual = estimate_compliance(
                        reloaded.compliance_summary(design(7)), design(7)
                    ).results[0]
                assert actual.require_lift().value == pytest.approx(expected.require_lift().value)
                assert actual.require_lift().lb is not None and actual.require_lift().ub is not None
                assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb)
                assert actual.require_lift().ub == pytest.approx(expected.require_lift().ub)
                portable = Analysis.from_moments(
                    pq.read_table(path).to_pylist(),
                    metrics=[],
                    design=design(7),
                )
                with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
                    public = lift_rows(portable.run(estimands=["compliance"]))[0].require_lift()
                assert (public.value, public.lb, public.ub) == pytest.approx(
                    (
                        expected.require_lift().value,
                        expected.require_lift().lb,
                        expected.require_lift().ub,
                    )
                )
                assert reloaded.context.cluster == source.context.cluster
                if clustered:
                    assert rec is not None
                    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
                    assert reloaded.cluster_counts() == {"control": 12, "treatment": 12}
            in_process = _native_source(analysis).moments_source(
                metrics=_native_source(analysis).context.metrics
            )
            assert in_process.compliance_summary(design(7)) == _native_source(
                analysis
            ).compliance_summary(design(7))
    finally:
        analysis.close()
        con.disconnect()


def test_legacy_artifact_is_identified_before_any_relation_execution(monkeypatch):
    con, analysis, context, store, _panel = native_fixture(("full",))
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            manifest = artifact.manifest
            for extensions in (
                tuple(ext for ext in manifest.extensions if ext.kind != "encouragement_uptake"),
                tuple(
                    ext.model_copy(update={"extension_version": 1})
                    if ext.kind == "encouragement_uptake"
                    else ext
                    for ext in manifest.extensions
                ),
            ):
                artifact._manifest = manifest.model_copy(update={"extensions": extensions})
                with monkeypatch.context() as patch:
                    patch.setattr(
                        type(artifact._snapshot),
                        "execute",
                        lambda *_: pytest.fail("legacy refusal must precede execution"),
                    )
                    with pytest.raises(CapabilityError) as error:
                        artifact.compliance_summary(design(7))
                    assert error.value.code == "source.compliance_summary.legacy_uptake_state"
    finally:
        analysis.close()
        con.disconnect()


def test_base_artifact_reader_implements_compliance_protocol():
    con, analysis, context, store, _panel = native_fixture(("full",))
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with ArtifactMomentSource.open(store, ref, expected_context=context) as source:
            expected = _native_source(analysis).compliance_summary(design(7))
            assert source.compliance_summary(design(7)) == expected
    finally:
        analysis.close()
        con.disconnect()


def test_public_asof_api_reports_one_design_level_row_per_day():
    con, analysis, context, store, panel = native_fixture()
    frame = Analysis.from_unit_panel(
        pa.Table.from_pylist(panel),
        unit="unit",
        group="arm",
        date="ds",
        exposure_date="exposed",
        metrics={"y": "mean"},
        design=design(7),
        experiment_id="uptake_test",
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        try:
            for source in (analysis, frame, adopted):
                results = source.run_asof_lift(estimands=("compliance",))
                compliance = [row for row in results if row.estimand == "compliance"]
                keys = [(row.ds, row.group_id) for row in compliance]
                assert len(keys) == len(set(keys)) == 10
                assert all(row.metric == "uptake" and row.role is None for row in compliance)
                for row in compliance:
                    assert isinstance(row.ds, date)
                    expected = 0.25 if row.ds < date(2025, 1, 7) else 0.5
                    assert row.require_lift().value == pytest.approx(expected)
                    assert row.require_lift().lb is not None
        finally:
            adopted.close()
    finally:
        frame.close()
        analysis.close()
        con.disconnect()


def test_current_artifact_rejects_contradictory_uptake_state(monkeypatch):
    from dataclasses import replace

    from increment.query.artifact_publish import ArtifactPublisher

    original = ArtifactPublisher._uptake_extension_spec

    def corrupt(self, *args, **kwargs):
        spec = original(self, *args, **kwargs)
        assert not isinstance(spec.rows, list)
        relation = spec.rows.mutate(
            uptake=ibis.literal(True),
            first_uptake_ts=ibis.null().cast("timestamp('UTC')"),
        )
        return replace(spec, rows=relation)

    monkeypatch.setattr(ArtifactPublisher, "_uptake_extension_spec", corrupt)
    con, analysis, context, store, _panel = native_fixture(("full",))
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as source:
            with pytest.raises(CodedError) as error:
                source.compliance_summary(design(7))
            assert error.value.code == "source.compliance_summary.invalid_state"
    finally:
        analysis.close()
        con.disconnect()


def test_uptake_publication_does_not_collect_the_relation_in_python(monkeypatch):
    con, analysis, context, store, _panel = native_fixture(("full",))
    original = con.to_pyarrow

    def bounded_only(expression, *args, **kwargs):
        # Scalar aggregates (e.g. the digest's duplicate-key/count checks)
        # have no `.schema()`; only a table fetch can pull uptake rows into Python.
        columns = getattr(expression, "columns", None)
        if columns is not None and "first_uptake_ts" in columns:
            pytest.fail("the uptake relation must go directly to the artifact writer")
        return original(expression, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", bounded_only)
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as source:
            arm = source.compliance_summary(design(7)).arm("treatment")
            assert arm is not None and arm.uptake_total == 10
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("separate_uptake", [False, True])
@pytest.mark.parametrize("open_ended", [False, True])
def test_publication_pins_uptake_before_metric_validation(monkeypatch, separate_uptake, open_ended):
    from increment.query.artifact_publish import ArtifactPublisher

    con, analysis, context, store, _panel = native_fixture(
        ("full",), separate_uptake=separate_uptake, open_ended=open_ended
    )
    original = ArtifactPublisher._validate_stats_batches
    mutated = False

    def mutate_after_validation(self, stats):
        nonlocal mutated
        last_day = original(self, stats)
        if not mutated:
            table = "uptake_events" if separate_uptake else "events"
            con.raw_sql(f"UPDATE {table} SET ts = ts + INTERVAL 365 DAY WHERE event = 'clicked'")
            mutated = True
        return last_day

    try:
        expected = _native_source(analysis).compliance_summary(design(7))
        expected_dates = _native_source(analysis).compliance_dates()
        monkeypatch.setattr(ArtifactPublisher, "_validate_stats_batches", mutate_after_validation)
        ref = analysis.publish_unit_day_artifact(store)
        assert mutated
        with open_artifact(store, ref, expected_context=context) as artifact:
            actual = artifact.compliance_summary(design(7))
            assert actual == expected
            assert artifact.compliance_dates() == expected_dates
            arm = actual.arm("treatment")
            assert arm is not None and arm.uptake_total == 10
        live = _native_source(analysis).compliance_summary(design(7)).arm("treatment")
        assert live is not None and live.uptake_total == 0
    finally:
        analysis.close()
        con.disconnect()


def test_publication_pins_outcome_and_uptake_before_inter_stream_mutation(monkeypatch):
    con, analysis, context, store, _panel = native_fixture(("full",))
    create = con.create_table
    captures = []

    def capture_then_mutate(name, obj=None, **kwargs):
        table = create(name, obj, **kwargs)
        if kwargs.get("temp") and "_stream" in table.columns:
            captures.append(table)
            if len(captures) == 1:
                con.raw_sql("DELETE FROM events WHERE event = 'clicked'")
                con.raw_sql("UPDATE events SET revenue = revenue + 100 WHERE event = 'full_event'")
        return table

    try:
        source = _native_source(analysis)
        metric = source.context.metrics[0]
        expected_moments = source.moments(metric)
        expected_compliance = source.compliance_summary(design(7))
        monkeypatch.setattr(con, "create_table", capture_then_mutate)
        ref = analysis.publish_unit_day_artifact(store)
        assert captures, "mutation hook never ran"
        with open_artifact(store, ref, expected_context=context) as artifact:
            assert artifact.compliance_summary(design(7)) == expected_compliance
            assert {row["group_id"]: row for row in artifact.moments(metric)} == {
                row["group_id"]: row for row in expected_moments
            }
        live = source.compliance_summary(design(7)).arm("treatment")
        assert live is not None and live.uptake_total == 0
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.parametrize("clustered", [False, True])
def test_triggered_compliance_matches_population_across_sources(tmp_path, clustered):
    import pyarrow.parquet as pq

    con, analysis, context, store, _panel = native_fixture(
        ("full",), clustered=clustered, triggered=True
    )
    try:
        native = _native_source(analysis)
        from increment.query.artifact_contract import unit_day_artifact_extension_catalog

        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind == "trigger_population"
        ]
        ref = analysis.publish_unit_day_artifact(store, extensions=extensions)
        with open_artifact(store, ref, expected_context=context) as artifact:
            expected_units = 63 if clustered else 8
            expected_uptake = 42 if clustered else 8
            sources = [
                native.triggered_source(),
                native.moments_source(metrics=native.context.metrics, population="triggered"),
                cast(Any, artifact).triggered_source(),
            ]
            summaries = [source.compliance_summary(design(7)) for source in sources]
            assert all(summary == summaries[0] for summary in summaries)
            assert summaries[0] != native.compliance_summary(design(7))
            for source, summary in zip(sources, summaries, strict=True):
                arm = summary.arm("treatment")
                assert arm is not None
                assert arm.n_units == expected_units
                assert arm.uptake_total == expected_uptake
                assert source.unit_counts() == {
                    "control": expected_units,
                    "treatment": expected_units,
                }
            path = tmp_path / "triggered.parquet"
            cast(Any, sources[-1]).export_moments(path)
            cube = MomentsSource(
                pq.read_table(path).to_pylist(),
                metrics=native.context.metrics,
                study_id="uptake_test",
                design=design(7),
            )
            assert cube.compliance_summary(design(7)) == summaries[0]
            with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext() as rec:
                results = analysis.run(estimands=["compliance"])
            if clustered:
                assert rec is not None
                assert "estimation.engine.small_total_clusters" in warning_codes(rec)
            compliance = next(
                row
                for row in results
                if row.estimand == "compliance" and row.analysis_population == "triggered"
            )
            assert compliance.analysis_population == "triggered"
            assert compliance.require_lift().value == pytest.approx(
                expected_uptake / expected_units
            )
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("path", ["analysis", "readouts"])
@pytest.mark.parametrize("outcome_window", [7, 9])
@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("metric_name", ["full", "drop"])
def test_native_artifact_completed_compliance_staggered_enrollment(
    path, outcome_window, observed, metric_name
):
    con, analysis, context, store, _panel = native_fixture(
        (metric_name,), staggered=True, outcome=(8, "sum", outcome_window)
    )
    if not observed:
        con.raw_sql(f"DELETE FROM events WHERE event = '{metric_name}_event'")
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
            try:
                for source, asof_lift in (
                    (_native_source(analysis), analysis.run_asof_lift),
                    (artifact, adopted.run_asof_lift),
                ):
                    route = (
                        asof_lift
                        if path == "analysis"
                        else lambda source=source, **kw: readouts.asof_lift(source, **kw)
                    )
                    results = route(estimands=["compliance"], completed_windows_only=True)
                    values = {row.ds: row.require_lift().value for row in results}
                    assert values == pytest.approx(
                        {date(2025, 1, 8): 0.5, date(2025, 1, 9): 0.5, date(2025, 1, 10): 0.75}
                    )
                    assert all(row.inference == "fixed" for row in results)
            finally:
                adopted.close()
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("outcome_state", ["present", "absent", "null", "filtered"])
@pytest.mark.parametrize("separate_uptake", [False, True])
def test_open_ended_compliance_uses_persisted_enrollment_uptake_coverage(
    outcome_state, separate_uptake
):
    con, analysis, context, store, _panel = native_fixture(
        ("full",), open_ended=True, outcome_state=outcome_state, separate_uptake=separate_uptake
    )
    try:
        uptake_table = "uptake_events" if separate_uptake else "events"
        # A repeat after the uptake window extends observation, without changing uptake.
        con.raw_sql(
            f"INSERT INTO {uptake_table} SELECT user_id, experiment_id, group_id, cluster_id, "
            "revenue, event, TIMESTAMPTZ '2025-01-11 00:00:00+00' "
            f"FROM {uptake_table} WHERE event = 'clicked' LIMIT 1"
        )
        con.raw_sql(
            f"INSERT INTO {uptake_table} SELECT 'outsider', experiment_id, group_id, cluster_id, "
            "revenue, event, TIMESTAMPTZ '2025-02-01 00:00:00+00' "
            f"FROM {uptake_table} WHERE event = 'clicked' LIMIT 1"
        )
        expected = {
            date(2025, 1, 1) + timedelta(days=day): 0.25 if day < 6 else 0.5 for day in range(11)
        }
        native = _native_source(analysis)
        assert list(native.compliance_dates()) == list(expected)
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
            try:
                for source, asof_lift in (
                    (native, analysis.run_asof_lift),
                    (artifact, adopted.run_asof_lift),
                ):
                    for run in (
                        lambda source=source, **kw: readouts.asof_lift(source, **kw),
                        asof_lift,
                    ):
                        results = run(estimands=["compliance"])
                        values = {row.ds: row.require_lift().value for row in results}
                        assert values == pytest.approx(expected)
                        assert len(results) == len(expected)
                        assert all(row.require_lift().lb is not None for row in results)
            finally:
                adopted.close()
            con.raw_sql(f"DELETE FROM {uptake_table} WHERE event = 'clicked'")
            assert list(artifact.compliance_dates()) == list(expected)
            assert list(native.compliance_dates()) == [date(2025, 1, 1)]
            results = readouts.asof_lift(artifact, estimands=["compliance"])
            assert {row.ds: row.require_lift().value for row in results} == pytest.approx(expected)
    finally:
        analysis.close()
        con.disconnect()


def test_older_timestamp_artifact_keeps_digest_and_uses_known_compliance_coverage():
    from increment.semantics.artifact import EncouragementUptakeExtension

    con, analysis, context, store, _panel = native_fixture(
        ("full",), open_ended=True, outcome_state="absent"
    )
    try:
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            extension = next(
                ext for ext in artifact.manifest.extensions if ext.kind == "encouragement_uptake"
            )
            old_payload = extension.model_dump(mode="json", exclude={"observation_edge"})
            old_extension = EncouragementUptakeExtension.model_validate(old_payload)
            assert old_extension.model_dump(mode="json") == old_payload
            artifact._manifest = artifact.manifest.model_copy(
                update={
                    "extensions": tuple(
                        old_extension if ext.kind == "encouragement_uptake" else ext
                        for ext in artifact.manifest.extensions
                    )
                }
            )
            assert list(artifact.compliance_dates()) == [
                date(2025, 1, 1) + timedelta(days=day) for day in range(7)
            ]
            assert artifact.moments(artifact.context.metrics[0], grain="asof") == []
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("mutate", [False, True])
def test_publication_preserves_relevant_event_rows_and_watermarks(monkeypatch, mutate):
    from increment.query.artifact_publish import ArtifactPublisher

    con, analysis, context, store, _panel = native_fixture(("full",))

    try:
        con.raw_sql(
            "INSERT INTO events SELECT user_id, experiment_id, group_id, cluster_id, "
            "revenue, event, ts - INTERVAL 365 DAY FROM events WHERE event = 'full_event'"
        )
        con.raw_sql(
            "INSERT INTO events SELECT 'outsider', experiment_id, group_id, cluster_id, "
            "revenue, event, ts + INTERVAL 1 DAY FROM events WHERE event = 'full_event' LIMIT 1"
        )
        metric = _native_source(analysis).context.metrics[0]
        expected = _native_source(analysis).moments(metric)
        validate = ArtifactPublisher._validate_stats_batches

        def mutate_after_validation(self, stats):
            result = validate(self, stats)
            con.raw_sql("UPDATE events SET ts = ts + INTERVAL 365 DAY WHERE event = 'full_event'")
            return result

        if mutate:
            monkeypatch.setattr(
                ArtifactPublisher, "_validate_stats_batches", mutate_after_validation
            )
        ref = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            arm = artifact.compliance_summary(design(7)).arm("treatment")
            assert arm is not None and arm.uptake_total == 10
            assert {row["group_id"]: row for row in artifact.moments(metric)} == {
                row["group_id"]: row for row in expected
            }
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            stats = snapshot.verify_relation(
                manifest.base.measure_stats, expected_role="measure_stats"
            )
            event_rows = con.to_pyarrow(stats).to_pylist()
        assert all(row["unit_id"] != "outsider" for row in event_rows)
        assert all(row["ds"].year == 2025 for row in event_rows)
        assert manifest.measures[0].freshness.loaded_through == date(2025, 1, 10)
        assert manifest.measures[0].event_horizon == date(2025, 1, 10)
    finally:
        analysis.close()
        con.disconnect()


def test_publication_keeps_canonical_clusters_when_live_exposures_are_corrected(monkeypatch):
    con, analysis, context, store, _panel = native_fixture(("full",), clustered=True)
    source = _native_source(analysis)
    try:
        expected = source.compliance_summary(design(7))
        source_type = type(source)
        validate = source_type._validate_mixed_assignments

        def correct_old_cluster_after_assignment_validation(pinned):
            validate(pinned)
            con.raw_sql(
                "INSERT INTO events SELECT user_id, experiment_id, group_id, 'corrected', "
                "revenue, event, ts + INTERVAL 1 SECOND FROM events "
                "WHERE event = 'exposed' AND user_id = 'treatment0_0'"
            )

        # A correction to live exposures landing after the assignment gate must
        # not rewrite the clusters the published generation pinned canonically.
        with monkeypatch.context() as patch:
            patch.setattr(
                source_type,
                "_validate_mixed_assignments",
                correct_old_cluster_after_assignment_validation,
            )
            reference = analysis.publish_unit_day_artifact(store)
        with open_artifact(store, reference, expected_context=context) as adopted:
            assert adopted.compliance_summary(design(7)) == expected
            assert adopted.cluster_counts() == {"control": 12, "treatment": 12}
        with pytest.raises(CodedError) as raised:
            source.compliance_summary(design(7))
        assert raised.value.code == "query.integrity.cluster_conflict"
    finally:
        analysis.close()
        con.disconnect()


def test_design_summary_requires_complete_state_even_with_plan_override(tmp_path):
    import pyarrow.parquet as pq

    from increment.semantics.models import AnalysisPlan
    from increment.sources import (
        ASSIGNMENT_COUNTS_FIELD,
        COMPLIANCE_SUMMARY_FIELD,
        DECISION_PLAN_FIELD,
    )

    con, analysis, _context, _store, _panel = native_fixture(())
    try:
        path = tmp_path / "design-only.parquet"
        _native_source(analysis).export_moments(path)
        envelope = pq.read_table(path).to_pylist()[0]
        for field, code in (
            (ASSIGNMENT_COUNTS_FIELD, "source.compliance_summary.invalid_state"),
            (COMPLIANCE_SUMMARY_FIELD, "source.compliance_summary.invalid_state"),
            (DECISION_PLAN_FIELD, "moments.plan.invalid"),
        ):
            damaged = dict(envelope)
            del damaged[field]
            with pytest.raises(CodedError) as raised:
                MomentsSource(
                    [damaged],
                    metrics=[],
                    study_id="uptake_test",
                    design=design(7),
                    plan=AnalysisPlan(),
                )
            assert raised.value.code == code
        for counts in (
            "{",
            "null",
            "[]",
            '{"control": true}',
            '{"control": -1}',
            '{"control": 1.5}',
            "{}",
        ):
            with pytest.raises(CodedError) as raised:
                Analysis.from_moments(
                    [{**envelope, ASSIGNMENT_COUNTS_FIELD: counts}],
                    metrics=[],
                    design=design(7),
                    plan=AnalysisPlan(),
                )
            assert raised.value.code == "source.compliance_summary.invalid_state"
    finally:
        analysis.close()
        con.disconnect()


def test_design_summary_rejects_incompatible_shape_identity_catalog_and_plans(tmp_path):
    import pyarrow.parquet as pq

    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import DECISION_PLAN_FIELD

    con, analysis, _context, _store, _panel = native_fixture(())
    try:
        path = tmp_path / "design-only.parquet"
        _native_source(analysis).export_moments(path)
        envelope = pq.read_table(path).to_pylist()[0]
        metric = MeanMetric(name="outcome", entity="user", fact="revenue")
        outcome_plan = compile_decision_plan(
            AnalysisPlan(), [metric], path="frame", design=design(7)
        )
        cases = (
            ([envelope, envelope], [], AnalysisPlan(), "uptake_test"),
            ([{**envelope, "metric": "outcome"}], [], AnalysisPlan(), "uptake_test"),
            ([envelope], [], AnalysisPlan(), "other-study"),
            ([envelope], [metric], AnalysisPlan(), "uptake_test"),
            ([envelope], [], outcome_plan, "uptake_test"),
            (
                [{**envelope, DECISION_PLAN_FIELD: compiled_plan_to_json(outcome_plan)}],
                [],
                AnalysisPlan(),
                "uptake_test",
            ),
        )
        for rows, metrics, plan, study_id in cases:
            with pytest.raises(CodedError) as raised:
                MomentsSource(rows, metrics=metrics, study_id=study_id, design=design(7), plan=plan)
            assert raised.value.code == "source.compliance_summary.invalid_state"
        for version in (7, 8, 9):
            with pytest.raises(CodedError) as raised:
                MomentsSource(
                    [{**envelope, "moments_format": version}],
                    metrics=[],
                    study_id="uptake_test",
                    design=design(7),
                    plan=AnalysisPlan(),
                )
            assert raised.value.code == "moments.format.invalid"
    finally:
        analysis.close()
        con.disconnect()
