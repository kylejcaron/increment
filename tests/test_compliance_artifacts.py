"""Real in-memory native/artifact transport for the declared uptake cohort."""

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import ibis
import pyarrow as pa
import pytest

from increment import Scope, from_frame, from_moments, from_unit_day_artifact, readouts, run
from increment.artifacts import SourceSnapshotEvidence, publish_unit_day_artifact
from increment.errors import CapabilityError, CodedError, IncrementRuntimeWarning
from increment.estimation.encouragement import estimate_compliance
from increment.frame import from_unit_panel
from increment.query.artifact_contract import unit_day_artifact_extension_catalog
from increment.query.artifact_publish import artifact_context
from increment.query.artifact_reader import ArtifactMomentSource
from increment.query.session import WarehouseArtifactStore
from increment.query.source import open_artifact
from increment.semantics.models import Definitions
from increment.sources import MomentsSource
from tests._evidence_requests import (
    _cluster_counts,
    _compliance,
    _compliance_dates,
    _moments,
    _unit_counts,
)
from tests.analysis_factory import (
    bound_source,
    lift_rows,
    make_binding,
    make_binding_like,
    view_readout,
)
from tests.test_compliance_summary import design
from tests.test_unit_day_artifact_facade import _publication_digest
from tests.warning_codes import warning_codes

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class ClickTiming:
    """Uptake window and click/exposure timing knobs for `native_fixture`."""

    uptake_window: int = 7
    boundary_click_offsets: list[timedelta] | None = None
    exposure_hour: int = 0
    day_boundary: str | None = None


_ASSIGNED = Scope(view="run", grain="total")
_TRIGGERED = Scope(view="run", grain="total", population="triggered")


def _both_populations(experiment, resolver, **options):
    """Whole-window rows of the assigned and triggered populations, read from one capture."""
    result = run(experiment.with_views(_ASSIGNED, _TRIGGERED), resolver, **options)
    return [*result.readout(_ASSIGNED), *result.readout(_TRIGGERED)]


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
    source_snapshot_evidence=None,
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
                    **({"day_boundary": timing.day_boundary} if timing.day_boundary else {}),
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
                                    "justification": "Only uptake changes the outcome"
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
    cutoff = start + timedelta(days=30)
    evidence = source_snapshot_evidence or (
        SourceSnapshotEvidence(cutoff, {"events": cutoff}) if triggered else None
    )
    supplied = None if declared else design(window=uptake_window)
    analysis = make_binding(con, definitions, design=supplied, source_snapshot_evidence=evidence)
    context = artifact_context(
        definitions, definitions.experiments[0], "error", encouragement_uptake=supplied
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    return con, analysis, context, store, panel


@pytest.fixture(scope="module")
def _shared_compliance_publications():
    publications = {}

    def publication(
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
        declared=False,
        source_snapshot_evidence=None,
    ):
        timing = timing or ClickTiming()
        snapshot_key = (
            None
            if source_snapshot_evidence is None
            else (
                source_snapshot_evidence.observation_cutoff_ts,
                tuple(sorted(source_snapshot_evidence.complete_through_by_feed.items())),
            )
        )
        key = (
            tuple(order),
            clustered,
            separate_uptake,
            triggered,
            staggered,
            tuple(outcome),
            timing.uptake_window,
            None if timing.boundary_click_offsets is None else tuple(timing.boundary_click_offsets),
            timing.exposure_hour,
            timing.day_boundary,
            open_ended,
            outcome_state,
            declared,
            snapshot_key,
        )
        if key not in publications:
            con, native, context, store, panel = native_fixture(
                order,
                clustered=clustered,
                separate_uptake=separate_uptake,
                triggered=triggered,
                staggered=staggered,
                outcome=outcome,
                timing=timing,
                open_ended=open_ended,
                outcome_state=outcome_state,
                declared=declared,
                source_snapshot_evidence=source_snapshot_evidence,
            )
            try:
                experiment, resolver = native
                ref = publish_unit_day_artifact(experiment, resolver, store)
                value = con, native, context, store, panel, ref
                publications[key] = value, _publication_digest(con, store, ref)
            finally:
                native[1].close()
        return publications[key][0]

    yield publication
    with ExitStack() as cleanup:
        for (con, _native_pair, _context, store, _panel, ref), before in publications.values():
            cleanup.callback(con.disconnect)
            assert _publication_digest(con, store, ref) == before


@pytest.fixture
def readonly_compliance_publications(_shared_compliance_publications):
    with ExitStack() as cleanup:

        def publication(*args, **options):
            con, recipe, context, store, panel, ref = _shared_compliance_publications(
                *args, **options
            )
            native = make_binding_like(recipe[1])
            cleanup.callback(native[1].close)
            return con, native, context, store, [dict(row) for row in panel], ref

        yield publication


@pytest.mark.parametrize("order", [("full", "drop"), ("drop", "full"), ("drop",)])
@pytest.mark.xdist_group("compliance_readonly")
def test_native_artifact_original_witness_and_matching_late(
    order, readonly_compliance_publications
):
    con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
        order
    )
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (bound_source(resolver), artifact):
                results = readouts.run(source, estimands=["compliance", "late"])
                comp = next(row for row in results if row.estimand == "compliance")
                assert comp.require_lift().value == pytest.approx(0.5)
                assert comp.require_lift().lb is not None
                for metric in source.context.metrics:
                    rows = {row["group_id"]: row for row in _moments(source, metric)}
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
        resolver.close()


@pytest.mark.parametrize("declared", [False, True], ids=["supplied-design", "declared-design"])
@pytest.mark.xdist_group("compliance_readonly")
def test_native_and_artifact_agree_for_each_supported_design_form(
    declared, readonly_compliance_publications
):
    """Both input forms keep the full typed design, every compliance row and interval,
    and one shared sequential source recipe and definition identity."""
    from increment.query.source import _artifact_source_context
    from increment.semantics.design import Encouragement
    from increment.sequential_source import sequential_definition_id

    con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
        declared=declared
    )
    try:
        native = bound_source(resolver)
        assert isinstance(native.context.design, Encouragement)
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
        resolver.close()


@pytest.mark.xdist_group("compliance_readonly")
def test_declared_seven_day_window_and_asof_match_frame_native_and_artifact(
    readonly_compliance_publications,
):
    con, (experiment, resolver), context, store, panel, ref = readonly_compliance_publications(
        ("full",)
    )
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
        with open_artifact(store, ref, expected_context=context) as artifact:
            native_day = bound_source(resolver).day_source(
                metrics=bound_source(resolver).context.metrics
            )
            for day, expected in [(0, 0.25), (5, 0.25), (6, 0.5), (7, 0.5), (9, 0.5)]:
                as_of = date(2025, 1, 1) + timedelta(days=day)
                summaries = [
                    _compliance(source, design(7), as_of=as_of)
                    for source in (frame, bound_source(resolver), native_day, artifact)
                ]
                assert all(summary == summaries[0] for summary in summaries)
                for summary in summaries:
                    arm = summary.arm("treatment")
                    assert arm is not None
                    assert arm.uptake_total / arm.n_units == expected
            assert _compliance(frame, design(7)) == _compliance(artifact, design(7))
            with pytest.raises(CodedError) as mismatch:
                _compliance(artifact, design(6))
            assert mismatch.value.code == "source.compliance_summary.design_mismatch"
    finally:
        resolver.close()


@pytest.mark.parametrize("legacy_coverage", [False, True], ids=["declared", "legacy"])
def test_uptake_after_outcome_day_extends_native_and_artifact_spine(legacy_coverage):
    """A qualifying next-day uptake survives when outcomes end on exposure day."""
    con, (experiment, resolver), context, store, _panel = native_fixture(
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
        ref = publish_unit_day_artifact(experiment, resolver, store)
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
            for source in (bound_source(resolver), artifact):
                metric = source.context.metrics[0]
                total = next(
                    row for row in _moments(source, metric) if row["group_id"] == "treatment"
                )
                assert total["sum_d"] == pytest.approx(10.0)
                asof = {
                    row["ds"]: row
                    for row in _moments(source, metric, grain="asof")
                    if row["group_id"] == "treatment"
                }
                assert asof[date(2025, 1, 2)]["sum_d"] == pytest.approx(10.0)
                assert asof[date(2025, 1, 2)]["n"] == 20
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.xdist_group("compliance_readonly")
def test_pinned_uptake_after_outcome_day_keeps_native_and_artifact_asof_aligned(
    readonly_compliance_publications,
):
    """Pinned native and artifact spines include the same cutoff day and divisors."""
    con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
        ("full",),
        timing=ClickTiming(
            uptake_window=1,
            boundary_click_offsets=[timedelta(hours=21)] * 10,
            exposure_hour=12,
        ),
        outcome=(0, "avg_calendar_day", 7),
        open_ended=True,
        source_snapshot_evidence=SourceSnapshotEvidence(datetime(2025, 1, 3, 23, 59, tzinfo=UTC)),
    )
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            sources = (bound_source(resolver), artifact)
            asof_rows = []
            for source in sources:
                metric = source.context.metrics[0]
                asof_rows.append(
                    {
                        row["ds"]: row
                        for row in _moments(source, metric, grain="asof")
                        if row["group_id"] == "treatment"
                    }
                )
            native, published = asof_rows
            assert native.keys() == published.keys()
            assert max(native) == date(2025, 1, 3)
            for day in native:
                assert native[day]["n"] == published[day]["n"]
                assert native[day]["sum_d"] == pytest.approx(published[day]["sum_d"])
                for field in ("ref_y", "cy1", "cy2"):
                    assert native[day][field] == pytest.approx(published[day][field])
    finally:
        resolver.close()


@pytest.mark.xdist_group("compliance_readonly")
def test_published_uptake_respects_certified_local_day_edge(readonly_compliance_publications):
    """Assigned uptake events beyond the certified edge are absent in both sources."""
    con, (analysis_experiment, analysis_resolver), context, store, _panel, ref = (
        readonly_compliance_publications(
            ("full",),
            timing=ClickTiming(
                uptake_window=5,
                boundary_click_offsets=[timedelta(days=2)] * 9 + [timedelta(days=4)],
                exposure_hour=12,
            ),
            outcome=(0, "sum", 7),
            open_ended=True,
            source_snapshot_evidence=SourceSnapshotEvidence(
                datetime(2025, 1, 10, 12, tzinfo=UTC),
                {"events": datetime(2025, 1, 4, 0, tzinfo=UTC)},
            ),
        )
    )
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            native = bound_source(analysis_resolver)
            sources = (native, artifact)
            for source in sources:
                metric = source.context.metrics[0]
                total = next(
                    row for row in _moments(source, metric) if row["group_id"] == "treatment"
                )
                assert total["sum_d"] == pytest.approx(9.0)
                summary = _compliance(source, design(5))
                arm = summary.arm("treatment")
                assert arm is not None and arm.uptake_total == pytest.approx(9.0)
            assert _compliance(native, design(5)) == _compliance(artifact, design(5))
            expected_compliance_dates = [
                date(2025, 1, 1),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ]
            assert list(_compliance_dates(native)) == expected_compliance_dates
            assert list(_compliance_dates(artifact)) == expected_compliance_dates
            for source in sources:
                asof = {
                    row["ds"]: row
                    for row in _moments(source, source.context.metrics[0], grain="asof")
                    if row["group_id"] == "treatment"
                }
                assert asof[date(2025, 1, 3)]["sum_d"] == pytest.approx(9.0)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        with adopted_resolver:
            native_rows = view_readout(
                analysis_experiment, analysis_resolver, estimands=["itt", "late"]
            )
            artifact_rows = view_readout(
                adopted_experiment, adopted_resolver, estimands=["itt", "late"]
            )
        native_results = {
            row.estimand: row
            for row in native_rows
            if row.metric == "full" and row.analysis_population == "assigned"
        }
        artifact_results = {
            row.estimand: row
            for row in artifact_rows
            if row.metric == "full" and row.analysis_population == "assigned"
        }
        assert set(native_results) == set(artifact_results) == {"itt", "late"}
        for estimand in ("itt", "late"):
            native_result = native_results[estimand]
            artifact_result = artifact_results[estimand]
            assert native_result.relative_unavailable_reason == (
                artifact_result.relative_unavailable_reason
            )
            assert native_result.abs_diff == pytest.approx(artifact_result.abs_diff)
            if native_result.lift is not None:
                assert artifact_result.lift is not None
                assert native_result.lift.value == pytest.approx(artifact_result.lift.value)
    finally:
        analysis_resolver.close()


@pytest.mark.integration
def test_completed_asof_requires_uptake_feed_certification_and_preserves_late_parity():
    uptake_edge = datetime(2025, 1, 4, tzinfo=UTC)
    outcome_edge = datetime(2025, 1, 31, tzinfo=UTC)
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        separate_uptake=True,
        staggered=True,
        timing=ClickTiming(uptake_window=7),
        outcome=(8, "sum", 7),
        open_ended=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2025, 1, 31, 23, 59, tzinfo=UTC),
            {"events": outcome_edge, "uptake_events": uptake_edge},
        ),
    )
    try:
        con.raw_sql(
            "UPDATE uptake_events SET ts = TIMESTAMP '2025-01-05 00:00:00' "
            "WHERE user_id = 'treatment0_0' AND ts >= TIMESTAMP '2025-01-01 00:00:00'"
        )
        ref = publish_unit_day_artifact(analysis_experiment, analysis_resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            extension = next(
                ext for ext in artifact.manifest.extensions if ext.kind == "encouragement_uptake"
            )
            assert extension.certified_edge == date(2025, 1, 3)
            native = bound_source(analysis_resolver)
            metric = native.context.metrics[0]
            native_rows = _moments(native, metric, grain="asof")
            artifact_rows = _moments(artifact, metric, grain="asof")
            native_by_day = {
                row["ds"]: row for row in native_rows if row["group_id"] == "treatment"
            }
            artifact_by_day = {
                row["ds"]: row for row in artifact_rows if row["group_id"] == "treatment"
            }
            assert native_by_day.keys() == artifact_by_day.keys()
            assert native_by_day[date(2025, 1, 8)]["sum_d"] == pytest.approx(
                artifact_by_day[date(2025, 1, 8)]["sum_d"]
            )
            assert native_by_day[date(2025, 1, 8)]["sum_d"] == pytest.approx(4.0)
            native_completed = _moments(native, metric, grain="asof", completed_windows_only=True)
            artifact_completed = _moments(
                artifact, metric, grain="asof", completed_windows_only=True
            )
            assert native_completed == artifact_completed == ()
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        with adopted_resolver:
            native_late = view_readout(
                analysis_experiment,
                analysis_resolver,
                Scope(view="asof", grain="asof", completion_policy=True),
                estimands=["late"],
            )
            artifact_late = view_readout(
                adopted_experiment,
                adopted_resolver,
                Scope(view="asof", grain="asof", completion_policy=True),
                estimands=["late"],
            )
            assert native_late == artifact_late
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.parametrize(
    ("exposure_hour", "day_boundary", "uptake_complete_through", "is_completed"),
    [
        (0, None, datetime(2025, 1, 7, tzinfo=UTC), False),
        (0, None, datetime(2025, 1, 8, tzinfo=UTC), True),
        (12, None, datetime(2025, 1, 8, tzinfo=UTC), False),
        (12, None, datetime(2025, 1, 9, tzinfo=UTC), True),
        (5, "UTC-05:00", datetime(2025, 1, 8, 5, tzinfo=UTC), True),
    ],
)
def test_completed_asof_uptake_certification_uses_elapsed_timestamp_boundary(
    exposure_hour, day_boundary, uptake_complete_through, is_completed
):
    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",),
        separate_uptake=True,
        staggered=True,
        timing=ClickTiming(uptake_window=7, exposure_hour=exposure_hour, day_boundary=day_boundary),
        outcome=(8, "sum", 7),
        open_ended=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2025, 1, 31, 23, 59, tzinfo=UTC),
            {
                "events": datetime(2025, 1, 31, 23, 59, tzinfo=UTC),
                "uptake_events": uptake_complete_through,
            },
        ),
    )
    try:
        con.raw_sql(
            f"UPDATE events SET ts = TIMESTAMP '2025-01-01 {exposure_hour:02}:00:00' "
            "WHERE event = 'exposed'"
        )
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            native = bound_source(resolver)
            metric = native.context.metrics[0]
            native_completed = _moments(native, metric, grain="asof", completed_windows_only=True)
            artifact_completed = _moments(
                artifact, metric, grain="asof", completed_windows_only=True
            )
            assert native_completed == artifact_completed
            asof_date = date(2025, 1, 8)
            native_compliance = _compliance(
                native, design(7), as_of=asof_date, completed_windows_only=True
            )
            artifact_compliance = _compliance(
                artifact, design(7), as_of=asof_date, completed_windows_only=True
            )
            assert native_compliance == artifact_compliance
            assert sum(arm.n_units for arm in native_compliance.arms) == (40 if is_completed else 0)
            completed_days = {row["ds"] for row in native_completed}
            assert (date(2025, 1, 8) in completed_days) is is_completed
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("exposure_ns", "certified_day", "completed"),
    [
        (0, 8, True),
        (999, 8, False),
        (1_000, 8, False),
        (999, 9, True),
        (86_400_000_000_000 - 1, 8, False),
        (86_400_000_000_000 - 1, 9, True),
    ],
)
def test_submicrosecond_exposure_preserves_completed_uptake_cohorts(
    exposure_ns, certified_day, completed
):
    def connect(events):
        connection = ibis.duckdb.connect()
        connection.create_table("events", obj=pa.Table.from_pylist(events))
        connection.raw_sql(
            "CREATE OR REPLACE TABLE events AS "
            "SELECT make_timestamp_ns(CASE WHEN event = 'exposed' "
            f"THEN epoch_us(TIMESTAMP '2025-01-01') * 1000 + {exposure_ns} "
            "ELSE epoch_us(ts) * 1000 END) AS ts, * EXCLUDE (ts) FROM events"
        )
        return connection

    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        separate_uptake=True,
        staggered=True,
        timing=ClickTiming(uptake_window=7),
        outcome=(6, "sum", 7),
        open_ended=True,
        warehouse_connection=connect,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2025, 1, 31, tzinfo=UTC),
            {
                "events": datetime(2025, 1, 31, tzinfo=UTC),
                "uptake_events": datetime(2025, 1, certified_day, tzinfo=UTC),
            },
        ),
    )
    try:
        ref = publish_unit_day_artifact(analysis_experiment, analysis_resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            native = bound_source(analysis_resolver)
            metric = native.context.metrics[0]
            for source in (native, artifact):
                summary = _compliance(
                    source, design(7), as_of=date(2025, 1, 9), completed_windows_only=True
                )
                assert sum(arm.n_units for arm in summary.arms) == (40 if completed else 0)
                rows = _moments(source, metric, grain="asof", completed_windows_only=True)
                assert bool(rows) is completed
            assert _moments(native, metric, grain="asof", completed_windows_only=True) == _moments(
                artifact, metric, grain="asof", completed_windows_only=True
            )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        with adopted_resolver:
            expected = view_readout(
                analysis_experiment,
                analysis_resolver,
                Scope(view="asof", grain="asof", completion_policy=True),
                estimands=["late"],
            )
            actual = view_readout(
                adopted_experiment,
                adopted_resolver,
                Scope(view="asof", grain="asof", completion_policy=True),
                estimands=["late"],
            )
            assert [
                (row.ds, row.group_id, row.sampling_available, row.failure_code, row.lift)
                for row in actual
            ] == [
                (row.ds, row.group_id, row.sampling_available, row.failure_code, row.lift)
                for row in expected
            ]
            assert any(row.sampling_available for row in actual) is completed
    finally:
        analysis_resolver.close()
        con.disconnect()


def test_midnight_window_uptake_floor_preserves_daily_and_asof_compliance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        timing=ClickTiming(uptake_window=14),
        outcome=(8, "sum", 7),
        open_ended=True,
    )
    exposure_ns = 1_735_689_600_000_000_999
    uptake_ns = exposure_ns + 14 * 24 * 60 * 60 * 1_000_000_000 - 1_000
    con.raw_sql(
        f"""
        CREATE OR REPLACE TABLE events AS
        SELECT * EXCLUDE (ts),
               CASE
                   WHEN event = 'exposed' THEN make_timestamp_ns({exposure_ns})
                   WHEN event = 'full_event'
                       THEN make_timestamp_ns({exposure_ns} + 1209600000000000)
                   WHEN event = 'clicked' AND user_id = 'treatment0_10'
                       AND ts >= TIMESTAMP '2025-01-01'
                   THEN make_timestamp_ns({uptake_ns})
                   ELSE make_timestamp_ns(epoch_us(ts) * 1000)
               END AS ts
        FROM events
        """
    )
    try:
        reference = publish_unit_day_artifact(analysis_experiment, analysis_resolver, store)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, reference, expected_context=context
        )
        with (
            open_artifact(store, reference, expected_context=context) as artifact,
            adopted_resolver,
        ):
            native = bound_source(analysis_resolver)
            for as_of, expected_uptake in (
                (date(2025, 1, 13), 14),
                (date(2025, 1, 14), 15),
                (date(2025, 1, 15), 15),
            ):
                native_summary = _compliance(native, design(14), as_of=as_of)
                artifact_summary = _compliance(artifact, design(14), as_of=as_of)
                assert native_summary == artifact_summary, as_of
                treatment = native_summary.arm("treatment")
                assert treatment is not None
                assert treatment.uptake_total == expected_uptake

            def lift_fields(rows):
                return [
                    (
                        row.ds,
                        row.group_id,
                        row.estimand,
                        row.lift,
                        row.unavailable,
                        row.n_treat,
                        row.n_control,
                    )
                    for row in rows
                ]

            assert lift_fields(
                view_readout(
                    analysis_experiment,
                    analysis_resolver,
                    Scope(view="daily", grain="daily"),
                    estimands=("compliance",),
                )
            ) == lift_fields(
                view_readout(
                    adopted_experiment,
                    adopted_resolver,
                    Scope(view="daily", grain="daily"),
                    estimands=("compliance",),
                )
            )
            assert lift_fields(
                view_readout(
                    analysis_experiment,
                    analysis_resolver,
                    Scope(view="asof", grain="asof"),
                    estimands=("compliance",),
                )
            ) == lift_fields(
                view_readout(
                    adopted_experiment,
                    adopted_resolver,
                    Scope(view="asof", grain="asof"),
                    estimands=("compliance",),
                )
            )

            with open_artifact(store, reference, expected_context=context) as edge_artifact:
                snapshot = edge_artifact._snapshot
                edge_artifact._manifest = edge_artifact.manifest.model_copy(
                    update={
                        "extensions": tuple(
                            extension.model_copy(
                                update={
                                    "observation_edge": date(2025, 1, 15),
                                    "certified_edge": date(2025, 1, 15),
                                }
                            )
                            if extension.kind == "encouragement_uptake"
                            else extension
                            for extension in edge_artifact.manifest.extensions
                        )
                    }
                )
                snapshot_type = type(snapshot)
                verify_relation = snapshot_type.verify_relation

                def verify_boundary_relation(self, *args, **kwargs):
                    relation = verify_relation(self, *args, **kwargs)
                    if kwargs.get("expected_role") == "encouragement_uptake":
                        boundary = ibis.literal(datetime(2025, 1, 15, tzinfo=UTC))
                        relation = relation.mutate(
                            first_uptake_ts=ibis.ifelse(
                                relation.unit_id == "treatment0_10",
                                boundary,
                                relation.first_uptake_ts,
                            )
                        )
                    return relation

                monkeypatch.setattr(snapshot_type, "verify_relation", verify_boundary_relation)
                for as_of, expected_uptake in (
                    (date(2025, 1, 14), 14),
                    (date(2025, 1, 15), 15),
                ):
                    summary = _compliance(edge_artifact, design(14), as_of=as_of)
                    treatment = summary.arm("treatment")
                    assert treatment is not None
                    assert treatment.uptake_total == expected_uptake
                metric = edge_artifact.context.metrics[0]
                treatment_total = next(
                    row for row in _moments(edge_artifact, metric) if row["group_id"] == "treatment"
                )
                assert treatment_total["sum_d"] == pytest.approx(15.0)
                boundary_asof = next(
                    row
                    for row in _moments(edge_artifact, metric, grain="asof")
                    if row["group_id"] == "treatment" and row["ds"] == date(2025, 1, 15)
                )
                assert boundary_asof["sum_d"] == pytest.approx(15.0)
                edge_adopted_experiment, edge_adopted_resolver = from_unit_day_artifact(
                    store, reference, expected_context=context
                )

                with edge_adopted_resolver:
                    endpoint_late = next(
                        row
                        for row in view_readout(
                            edge_adopted_experiment, edge_adopted_resolver, estimands=("late",)
                        )
                        if row.estimand == "late"
                    )
                    ordinary_late = next(
                        row
                        for row in view_readout(
                            adopted_experiment, adopted_resolver, estimands=("late",)
                        )
                        if row.estimand == "late"
                    )
                    assert endpoint_late.require_lift().value == pytest.approx(
                        ordinary_late.require_lift().value
                    )
                    endpoint_asof = next(
                        row
                        for row in view_readout(
                            edge_adopted_experiment,
                            edge_adopted_resolver,
                            Scope(view="asof", grain="asof"),
                            estimands=("late",),
                        )
                        if row.ds == date(2025, 1, 15)
                    )
                    ordinary_asof = next(
                        row
                        for row in view_readout(
                            adopted_experiment,
                            adopted_resolver,
                            Scope(view="asof", grain="asof"),
                            estimands=("late",),
                        )
                        if row.ds == date(2025, 1, 15)
                    )
                    assert endpoint_asof.require_lift().value == pytest.approx(
                        ordinary_asof.require_lift().value
                    )
    finally:
        analysis_resolver.close()
        con.disconnect()


def test_triggered_asof_uptake_accumulates_from_assignment_before_trigger():
    cutoff = datetime(2025, 1, 10, 23, 59, tzinfo=UTC)
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        triggered=True,
        outcome=(8, "sum", 7),
        open_ended=True,
        source_snapshot_evidence=SourceSnapshotEvidence(cutoff),
    )
    try:
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMP '2025-01-02 00:00:00' "
            "WHERE user_id = 'treatment0_0' AND event = 'exposed'"
        )
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMP '2025-01-03 00:00:00' "
            "WHERE user_id = 'treatment0_0' AND event = 'clicked' "
            "AND ts >= TIMESTAMP '2025-01-01 00:00:00'"
        )
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMP '2025-01-04 00:00:00' "
            "WHERE user_id = 'treatment0_0' AND event = 'triggered'"
        )
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        with open_artifact(store, ref, expected_context=context) as artifact:
            native = bound_source(analysis_resolver)
            metric = native.context.metrics[0]
            native_rows = _moments(
                native.day_source(metrics=(metric,), population="triggered"),
                metric,
                grain="asof",
            )
            artifact_rows = _moments(
                artifact.day_source(metrics=(metric,), population="triggered"),
                metric,
                grain="asof",
            )
            native_treatment = {
                row["ds"]: row for row in native_rows if row["group_id"] == "treatment"
            }
            artifact_treatment = {
                row["ds"]: row for row in artifact_rows if row["group_id"] == "treatment"
            }
            assert native_treatment.keys() == artifact_treatment.keys()
            assert native_treatment[date(2025, 1, 4)]["sum_d"] == pytest.approx(4.0)
            assert artifact_treatment[date(2025, 1, 4)]["sum_d"] == pytest.approx(
                native_treatment[date(2025, 1, 4)]["sum_d"]
            )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        with adopted_resolver:
            native_late = view_readout(
                analysis_experiment,
                analysis_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=["late"],
            )
            artifact_late = view_readout(
                adopted_experiment,
                adopted_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=["late"],
            )
            native_values = {
                row.ds: row.require_lift().value
                for row in native_late
                if row.estimand == "late" and row.ds == date(2025, 1, 9)
            }
            artifact_values = {
                row.ds: row.require_lift().value
                for row in artifact_late
                if row.estimand == "late" and row.ds == date(2025, 1, 9)
            }
            assert native_values == pytest.approx(artifact_values)
            assert native_values
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.parametrize(
    ("assignment_ts", "uptake_complete_through", "expected_units"),
    [
        ("2025-01-01 00:00:00", datetime(2025, 1, 8, tzinfo=UTC), 8),
        ("2025-01-01 12:00:00", datetime(2025, 1, 8, tzinfo=UTC), 7),
        ("2025-01-01 12:00:00", datetime(2025, 1, 9, tzinfo=UTC), 8),
    ],
)
def test_triggered_completed_asof_uses_assignment_timestamp_for_uptake_certification(
    assignment_ts, uptake_complete_through, expected_units
):
    cutoff = datetime(2025, 1, 31, 23, 59, tzinfo=UTC)
    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",),
        separate_uptake=True,
        triggered=True,
        staggered=True,
        timing=ClickTiming(uptake_window=7),
        outcome=(8, "sum", 7),
        open_ended=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            cutoff, {"events": cutoff, "uptake_events": uptake_complete_through}
        ),
    )
    try:
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMP '2025-01-01 00:00:00' WHERE event = 'exposed'"
        )
        con.raw_sql(
            f"UPDATE events SET ts = TIMESTAMP '{assignment_ts}' "
            "WHERE user_id = 'treatment0_0' AND event = 'exposed'"
        )
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMP '2025-01-04 00:00:00' "
            "WHERE user_id = 'treatment0_0' AND event = 'triggered'"
        )
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        with open_artifact(store, ref, expected_context=context) as artifact:
            native = bound_source(resolver)
            metric = native.context.metrics[0]
            native_rows = _moments(
                native.day_source(metrics=(metric,), population="triggered"),
                metric,
                grain="asof",
                completed_windows_only=True,
            )
            artifact_rows = _moments(
                artifact.day_source(metrics=(metric,), population="triggered"),
                metric,
                grain="asof",
                completed_windows_only=True,
            )
            assert native_rows == artifact_rows
            treatment_at_completion = [
                row
                for row in native_rows
                if row["ds"] == date(2025, 1, 11) and row["group_id"] == "treatment"
            ]
            assert treatment_at_completion[0]["n"] == expected_units
    finally:
        resolver.close()
        con.disconnect()


def test_late_uptake_preserves_calendar_average_total_scope():
    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",),
        open_ended=True,
        timing=ClickTiming(uptake_window=1),
        outcome=(1, "avg_calendar_day", 7),
    )
    try:
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (bound_source(resolver), artifact):
                rows = {row["group_id"]: row for row in _moments(source, source.context.metrics[0])}
                for arm, total in (("control", 25.0), ("treatment", 35.0)):
                    row = rows[arm]
                    assert row["n"] == 20
                    assert row["n"] * row["ref_y"] + row["cy1"] == pytest.approx(total)
    finally:
        resolver.close()
        con.disconnect()


def check_uptake_outcome_coverage(outcome_day, *, warehouse_connection=None):
    from integration.warehouse_execution._suite import _drop_probe_generation

    con, (experiment, resolver), context, store, _panel = native_fixture(
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
    cleanup.callback(resolver.close)
    try:
        ref = publish_unit_day_artifact(experiment, resolver, store)
        cleanup.callback(_drop_probe_generation, con, store, ref, con.name, schema_name="artifacts")
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (bound_source(resolver), artifact):
                metric = source.context.metrics[0]
                rows = _moments(source, metric, grain="asof")
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
                completed = _moments(source, metric, grain="asof", completed_windows_only=True)
                if outcome_day == 1:
                    assert completed == ()
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


@pytest.mark.xdist_group("compliance_readonly")
def test_total_and_asof_uptake_share_elapsed_timestamp_boundaries(readonly_compliance_publications):
    offsets = [timedelta(0), timedelta(hours=21), timedelta(days=1)]
    con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
        ("full",),
        timing=ClickTiming(uptake_window=1, boundary_click_offsets=offsets, exposure_hour=12),
    )
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            for source in (bound_source(resolver), artifact):
                summary = _compliance(source, design(1))
                treatment = summary.arm("treatment")
                assert treatment is not None
                assert treatment.n_units == 20
                assert treatment.uptake_total == pytest.approx(5.0)
                rows = {
                    row["ds"]: row
                    for row in _moments(source, source.context.metrics[0], grain="asof")
                    if row["group_id"] == "treatment"
                }
                assert rows[date(2025, 1, 1)]["sum_d"] / rows[date(2025, 1, 1)][
                    "n"
                ] == pytest.approx(4 / 20)
                assert rows[date(2025, 1, 2)]["sum_d"] / rows[date(2025, 1, 2)][
                    "n"
                ] == pytest.approx(5 / 20)
    finally:
        resolver.close()


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("metric_free", [False, True])
@pytest.mark.integration
@pytest.mark.xdist_group("compliance_readonly")
def test_new_native_artifact_export_reload_preserves_available_uncertainty(
    tmp_path, clustered, metric_free, readonly_compliance_publications
):
    import pyarrow.parquet as pq

    con, (analysis_experiment, analysis_resolver), context, store, _panel, ref = (
        readonly_compliance_publications(() if metric_free else ("full",), clustered=clustered)
    )
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            native_summary = _compliance(bound_source(analysis_resolver), design(7))
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
            for i, source in enumerate((bound_source(analysis_resolver), artifact)):
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
                        _compliance(reloaded, design(7)), design(7)
                    ).results[0]
                assert actual.require_lift().value == pytest.approx(expected.require_lift().value)
                assert actual.require_lift().lb is not None and actual.require_lift().ub is not None
                assert actual.require_lift().lb == pytest.approx(expected.require_lift().lb)
                assert actual.require_lift().ub == pytest.approx(expected.require_lift().ub)
                portable_experiment, portable_resolver = from_moments(
                    pq.read_table(path).to_pylist(), metrics=[], design=design(7)
                )
                with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
                    public = lift_rows(
                        view_readout(
                            portable_experiment, portable_resolver, estimands=["compliance"]
                        )
                    )[0].require_lift()
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
                    assert _cluster_counts(reloaded) == {"control": 12, "treatment": 12}
            in_process = bound_source(analysis_resolver).moments_source(
                metrics=bound_source(analysis_resolver).context.metrics
            )
            assert _compliance(in_process, design(7)) == _compliance(
                bound_source(analysis_resolver), design(7)
            )
    finally:
        analysis_resolver.close()


def test_legacy_artifact_is_identified_before_any_relation_execution(monkeypatch):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",))
    try:
        ref = publish_unit_day_artifact(experiment, resolver, store)
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
                        _compliance(artifact, design(7))
                    assert error.value.code == "source.compliance_summary.legacy_uptake_state"
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.xdist_group("compliance_readonly")
def test_base_artifact_reader_implements_compliance_protocol(readonly_compliance_publications):
    con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
        ("full",)
    )
    try:
        with ArtifactMomentSource.open(store, ref, expected_context=context) as source:
            expected = _compliance(bound_source(resolver), design(7))
            assert _compliance(source, design(7)) == expected
    finally:
        resolver.close()


@pytest.mark.xdist_group("compliance_readonly")
def test_public_asof_api_reports_one_design_level_row_per_day(readonly_compliance_publications):
    con, (experiment, resolver), context, store, panel, ref = readonly_compliance_publications()
    frame_experiment, frame_resolver = from_frame(
        pa.Table.from_pylist(panel),
        shape="panel",
        unit="unit",
        group="arm",
        date="ds",
        exposure_date="exposed",
        metrics={"y": "mean"},
        design=design(7),
        experiment_id="uptake_test",
    )
    try:
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (frame_experiment, frame_resolver),
                (adopted_experiment, adopted_resolver),
            ):
                results = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(view="asof", grain="asof"),
                    estimands=("compliance",),
                )
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
            adopted_resolver.close()
    finally:
        frame_resolver.close()
        resolver.close()


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
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",))
    try:
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as source:
            with pytest.raises(CodedError) as error:
                _compliance(source, design(7))
            assert error.value.code == "source.compliance_summary.invalid_state"
    finally:
        resolver.close()
        con.disconnect()


def test_uptake_publication_does_not_collect_the_relation_in_python(monkeypatch):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",))
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
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as source:
            arm = _compliance(source, design(7)).arm("treatment")
            assert arm is not None and arm.uptake_total == 10
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.parametrize("separate_uptake", [False, True])
@pytest.mark.parametrize("open_ended", [False, True])
def test_publication_pins_uptake_before_metric_validation(monkeypatch, separate_uptake, open_ended):
    from increment.query.artifact_publish import ArtifactPublisher

    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",), separate_uptake=separate_uptake, open_ended=open_ended
    )
    original = ArtifactPublisher._validate_stats_relation
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
        expected = _compliance(bound_source(resolver), design(7))
        expected_dates = _compliance_dates(bound_source(resolver))
        monkeypatch.setattr(ArtifactPublisher, "_validate_stats_relation", mutate_after_validation)
        ref = publish_unit_day_artifact(experiment, resolver, store)
        assert mutated
        with open_artifact(store, ref, expected_context=context) as artifact:
            actual = _compliance(artifact, design(7))
            assert actual == expected
            assert _compliance_dates(artifact) == expected_dates
            arm = actual.arm("treatment")
            assert arm is not None and arm.uptake_total == 10
        live = _compliance(bound_source(resolver), design(7)).arm("treatment")
        assert live is not None and live.uptake_total == 0
    finally:
        resolver.close()
        con.disconnect()


def test_publication_pins_outcome_and_uptake_before_inter_stream_mutation(monkeypatch):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",))
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
        source = bound_source(resolver)
        metric = source.context.metrics[0]
        expected_moments = _moments(source, metric)
        expected_compliance = _compliance(source, design(7))
        monkeypatch.setattr(con, "create_table", capture_then_mutate)
        ref = publish_unit_day_artifact(experiment, resolver, store)
        assert captures, "mutation hook never ran"
        with open_artifact(store, ref, expected_context=context) as artifact:
            assert _compliance(artifact, design(7)) == expected_compliance
            assert {row["group_id"]: row for row in _moments(artifact, metric)} == {
                row["group_id"]: row for row in expected_moments
            }
        live = _compliance(source, design(7)).arm("treatment")
        assert live is not None and live.uptake_total == 0
    finally:
        resolver.close()
        con.disconnect()


_TRIGGERED_LIFT_HISTORY = (
    Scope(view="daily", grain="daily", population="triggered"),
    Scope(view="asof", grain="asof", population="triggered"),
)


def _assert_clustered_artifact_outcome_history_refused(experiment, resolver) -> None:
    from tests.capability_consumers_auxiliary import assert_refusal

    with pytest.raises(CapabilityError) as refusal:
        view_readout(
            experiment,
            resolver,
            Scope(view="daily", grain="daily", values=True, population="triggered"),
        )
    assert refusal.value.code == "facade.analysis.clustered_day_axis"
    assert_refusal(
        "triggered_clustered_day_axis",
        "clustered_artifact_outcome_history_refused",
        refusal.value,
        stage="read",
    )


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.integration
def test_triggered_compliance_matches_population_across_sources(tmp_path, clustered):
    import pyarrow.parquet as pq

    from tests.capability_consumers_auxiliary import assert_support

    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",), clustered=clustered, triggered=True
    )
    try:
        native = bound_source(resolver)
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind in {"trigger_population", "trigger_measure_stats"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        with open_artifact(store, ref, expected_context=context) as artifact:
            expected_units = 63 if clustered else 8
            expected_uptake = 42 if clustered else 8
            sources = [
                native.triggered_source(),
                native.moments_source(metrics=native.context.metrics, population="triggered"),
                cast(Any, artifact).triggered_source(),
            ]
            summaries = [
                _compliance(source, design(7), population="triggered") for source in sources
            ]
            assert all(summary == summaries[0] for summary in summaries)
            assert summaries[0] != _compliance(native, design(7))
            for source, summary in zip(sources, summaries, strict=True):
                arm = summary.arm("treatment")
                assert arm is not None
                assert arm.n_units == expected_units
                assert arm.uptake_total == expected_uptake
                assert _unit_counts(source) == {
                    "control": expected_units,
                    "treatment": expected_units,
                }
            path = tmp_path / "triggered.parquet"
            sources[-1].export_moments(path)
            cube = MomentsSource(
                pq.read_table(path).to_pylist(),
                metrics=native.context.metrics,
                study_id="uptake_test",
                design=design(7),
            )
            assert _compliance(cube, design(7)) == summaries[0]
            with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext() as rec:
                results = _both_populations(experiment, resolver, estimands=["compliance"])
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
            adopted_experiment, adopted_resolver = from_unit_day_artifact(
                store, ref, expected_context=context
            )
            with adopted_resolver:
                if clustered:
                    _assert_clustered_artifact_outcome_history_refused(
                        adopted_experiment, adopted_resolver
                    )
                for scope in _TRIGGERED_LIFT_HISTORY:
                    with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
                        native_rows = view_readout(
                            experiment, resolver, scope, estimands=["compliance"]
                        )
                    if clustered:
                        assert_support(
                            "triggered_clustered_day_axis",
                            "clustered_compliance_history_supported",
                            native_rows,
                        )
                    with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
                        artifact_rows = view_readout(
                            adopted_experiment, adopted_resolver, scope, estimands=["compliance"]
                        )
                    if clustered:
                        assert_support(
                            "triggered_clustered_day_axis",
                            "clustered_artifact_compliance_history_supported",
                            artifact_rows,
                        )
                    assert {row.analysis_population for row in native_rows} == {"triggered"}
                    assert {row.analysis_population for row in artifact_rows} == {"triggered"}
                    assert {(row.ds, row.group_id, row.estimand) for row in native_rows} == {
                        (row.ds, row.group_id, row.estimand) for row in artifact_rows
                    }
                    for native_row, artifact_row in zip(native_rows, artifact_rows, strict=True):
                        assert native_row.require_lift().value == pytest.approx(
                            artifact_row.require_lift().value
                        )
                        assert native_row.require_lift().lb == pytest.approx(
                            artifact_row.require_lift().lb
                        )
                        assert native_row.require_lift().ub == pytest.approx(
                            artifact_row.require_lift().ub
                        )
                    native_frame = native_rows.to_frame(backend="polars")
                    artifact_frame = artifact_rows.to_frame(backend="polars")
                    assert native_frame.columns == artifact_frame.columns
                    for frame in (native_frame, artifact_frame):
                        assert "analysis_population" in frame.columns
                        assert "estimand" in frame.columns
                        assert set(frame.get_column("analysis_population")) == {"triggered"}
                        assert set(frame.get_column("estimand")) == {"compliance"}
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.integration
def test_triggered_compliance_history_extends_after_trigger_to_uptake_and_maturity(
    tmp_path,
):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",), triggered=True)
    con.raw_sql("UPDATE events SET ts = ts + INTERVAL 2 DAY WHERE event = 'triggered'")
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            actual = []
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (adopted_experiment, adopted_resolver),
            ):
                for scope in _TRIGGERED_LIFT_HISTORY:
                    rows = view_readout(
                        source_experiment, source_resolver, scope, estimands=("compliance",)
                    )
                    actual.append([(row.ds, row.require_lift().value) for row in rows])
                complete = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(
                        view="asof", grain="asof", completion_policy=True, population="triggered"
                    ),
                    estimands=("compliance",),
                )
                assert {row.ds for row in complete} == {
                    date(2025, 1, 8) + timedelta(days=offset) for offset in range(3)
                }
            expected_days = [date(2025, 1, 3) + timedelta(days=day) for day in range(8)]
            expected = [(day, 0.5 if day < date(2025, 1, 7) else 1.0) for day in expected_days]
            date_lists = [[day for day, _value in rows] for rows in actual]
            assert all(days == expected_days for days in date_lists)
            for rows in actual:
                assert [value for _day, value in rows] == pytest.approx(
                    [value for _day, value in expected]
                )
        finally:
            adopted_resolver.close()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore::increment.errors.IncrementRuntimeWarning")
@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.integration
def test_sparse_triggered_compliance_keeps_later_estimable_looks(clustered):
    con, (experiment, resolver), _context, _store, _panel = native_fixture(
        ("full",), clustered=clustered, triggered=True
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-02 00:00:00+00' WHERE event = 'triggered'"
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-01 00:00:00+00' "
        "WHERE event = 'triggered' AND user_id IN ('control0_0', 'treatment0_0')"
    )
    try:
        rows = view_readout(
            experiment,
            resolver,
            Scope(view="asof", grain="asof", population="triggered"),
            estimands=("compliance",),
        )
        by_cell = {(row.ds, row.group_id): row for row in rows}
        assert set(by_cell) >= {(day, "treatment") for day in (date(2025, 1, 1), date(2025, 1, 2))}
        assert by_cell[(date(2025, 1, 1), "treatment")].unavailable == "few_units"
        assert by_cell[(date(2025, 1, 2), "treatment")].require_lift().value is not None
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_history_emits_missing_multi_treatment_cells():
    con, (experiment, resolver), _context, _store, _panel = native_fixture(
        ("full",), triggered=True, open_ended=True
    )
    con.raw_sql(
        "INSERT INTO events VALUES "
        "('t2_0', 'uptake_test', 'T2', 't2_0', NULL, 'exposed', "
        "TIMESTAMPTZ '2025-01-01 00:00:00+00'), "
        "('t2_0', 'uptake_test', 'T2', 't2_0', NULL, 'triggered', "
        "TIMESTAMPTZ '2025-01-03 00:00:00+00'), "
        "('t2_0', 'uptake_test', 'T2', 't2_0', NULL, 'clicked', "
        "TIMESTAMPTZ '2025-01-03 00:00:00+00')"
    )
    try:
        rows = view_readout(
            experiment,
            resolver,
            Scope(view="asof", grain="asof", population="triggered"),
            estimands=("compliance",),
        )
        early = {
            (row.ds, row.group_id): row
            for row in rows
            if row.ds in (date(2025, 1, 1), date(2025, 1, 2))
        }
        assert (date(2025, 1, 1), "T2") in early
        assert (date(2025, 1, 2), "T2") in early
        assert all(
            early[(day, "T2")].failure_code == "readout.cell.missing_arm"
            for day in (date(2025, 1, 1), date(2025, 1, 2))
        )
        assert all(
            early[(day, "treatment")].require_lift().value is not None
            for day in (date(2025, 1, 1), date(2025, 1, 2))
        )
        assert all(
            early[(day, "T2")].unavailable == "no_treatment_arm"
            and early[(day, "T2")].n_control > 0
            for day in (date(2025, 1, 1), date(2025, 1, 2))
        )
        jan3 = {(row.ds, row.group_id): row for row in rows if row.ds == date(2025, 1, 3)}
        assert jan3[(date(2025, 1, 3), "T2")].unavailable == "few_units"
        assert jan3[(date(2025, 1, 3), "treatment")].require_lift().value is not None
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_completed_compliance_respects_uptake_certification_edge(tmp_path):
    cutoff = datetime(2025, 1, 31, tzinfo=UTC)
    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",),
        triggered=True,
        separate_uptake=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            cutoff,
            {
                "events": cutoff,
                "uptake_events": datetime(2025, 1, 3, 23, 59, tzinfo=UTC),
            },
        ),
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-11 00:00:00+00' WHERE event = 'triggered'"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (adopted_experiment, adopted_resolver),
            ):
                rows = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
                assert rows == []
                completed = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(
                        view="asof", grain="asof", completion_policy=True, population="triggered"
                    ),
                    estimands=("compliance",),
                )
                assert completed == []
        finally:
            adopted_resolver.close()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_history_uses_population_independent_horizon():
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",), triggered=True, open_ended=True
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-12 00:00:00+00' "
        "WHERE user_id = 'treatment0_9' AND event = 'clicked'"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            expected = [date(2025, 1, 1) + timedelta(days=day) for day in range(12)]
            native_rows = view_readout(
                analysis_experiment,
                analysis_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=("compliance",),
            )
            assert [row.ds for row in native_rows] == expected
            rows = view_readout(
                adopted_experiment,
                adopted_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=("compliance",),
            )
            assert [row.ds for row in rows] == expected
        finally:
            adopted_resolver.close()
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_history_uses_uptake_feed_completion_not_trigger_feed():
    cutoff = datetime(2025, 1, 31, 23, 59, tzinfo=UTC)
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        triggered=True,
        separate_uptake=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            cutoff,
            {
                "events": datetime(2025, 1, 3, 23, 59, tzinfo=UTC),
                "uptake_events": cutoff,
            },
        ),
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            native_dates = [
                row.ds
                for row in view_readout(
                    analysis_experiment,
                    analysis_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
            ]
            artifact_dates = [
                row.ds
                for row in view_readout(
                    adopted_experiment,
                    adopted_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
            ]
            assert native_dates == artifact_dates
            assert len(native_dates) > 2
        finally:
            adopted_resolver.close()
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_history_keeps_triggers_after_uptake_events():
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",), triggered=True, open_ended=True
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-11 00:00:00+00' WHERE event = 'triggered'"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            native_dates = [
                row.ds
                for row in view_readout(
                    analysis_experiment,
                    analysis_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
            ]
            artifact_dates = [
                row.ds
                for row in view_readout(
                    adopted_experiment,
                    adopted_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
            ]
            assert native_dates == artifact_dates
            assert native_dates[0] == date(2025, 1, 11)
            assert native_dates[-1] == date(2025, 1, 11)
        finally:
            adopted_resolver.close()
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_history_stops_at_pinned_observation_edge():
    cutoff = datetime(2025, 1, 5, 23, 59, tzinfo=UTC)
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        triggered=True,
        source_snapshot_evidence=SourceSnapshotEvidence(cutoff, {"events": cutoff}),
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            native_rows = view_readout(
                analysis_experiment,
                analysis_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=("compliance",),
            )
            rows = view_readout(
                adopted_experiment,
                adopted_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=("compliance",),
            )
            dates = [row.ds for row in native_rows]
            assert dates == [row.ds for row in rows]
            assert dates
            assert max(dates) <= cutoff.date()
        finally:
            adopted_resolver.close()
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_reopen_clamps_stale_persisted_edge():
    import json

    from increment.query.artifact_digest import manifest_sha256

    cutoff = datetime(2025, 1, 5, 23, 59, tzinfo=UTC)
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",),
        triggered=True,
        source_snapshot_evidence=SourceSnapshotEvidence(cutoff),
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        stale_extensions = tuple(
            ext.model_copy(update={"observation_edge": date(2025, 1, 30)})
            if ext.kind == "encouragement_uptake"
            else ext
            for ext in manifest.extensions
        )
        stale_manifest = manifest.model_copy(update={"extensions": stale_extensions})
        digest = manifest_sha256(stale_manifest)
        payload = stale_manifest.model_copy(update={"manifest_sha256": digest}).model_dump(
            mode="json"
        )
        manifest_json = json.dumps(payload, separators=(",", ":")).replace("'", "''")
        con.raw_sql(
            'UPDATE "artifacts"."ud_manifest_index" '
            f"SET manifest_json = '{manifest_json}', manifest_sha256 = '{digest}' "
            f"WHERE artifact_id = '{ref.artifact_id}' AND generation_id = '{ref.generation_id}'"
        )
        reopened_store = WarehouseArtifactStore(con, schema_name="artifacts")
        reopened_ref = ref.model_copy(update={"manifest_sha256": digest})
        with reopened_store.open_snapshot(reopened_ref) as snapshot:
            reopened_manifest = snapshot.read_manifest(
                reopened_ref.manifest, expected_sha256=reopened_ref.manifest_sha256
            )
        persisted_extension = next(
            ext for ext in reopened_manifest.extensions if ext.kind == "encouragement_uptake"
        )
        assert persisted_extension.observation_edge == date(2025, 1, 30)
        native_rows = view_readout(
            analysis_experiment,
            analysis_resolver,
            Scope(view="asof", grain="asof", population="triggered"),
            estimands=("compliance",),
        )
        reopened_experiment, reopened_resolver = from_unit_day_artifact(
            reopened_store, reopened_ref, expected_context=context
        )
        with reopened_resolver:
            artifact_rows = view_readout(
                reopened_experiment,
                reopened_resolver,
                Scope(view="asof", grain="asof", population="triggered"),
                estimands=("compliance",),
            )
        assert [row.ds for row in artifact_rows] == [row.ds for row in native_rows]
        assert [row.require_lift().value for row in artifact_rows] == pytest.approx(
            [row.require_lift().value for row in native_rows]
        )
        assert [row.failure_code for row in artifact_rows] == [
            row.failure_code for row in native_rows
        ]
        assert artifact_rows
        assert max(cast(date, row.ds) for row in artifact_rows) <= cutoff.date()
    finally:
        analysis_resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_keeps_missing_control_date_and_continues(tmp_path):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",), triggered=True)
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-04 00:00:00+00' "
        "WHERE event = 'triggered' AND group_id = 'treatment'"
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-05 00:00:00+00' "
        "WHERE event = 'triggered' AND group_id = 'control'"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (adopted_experiment, adopted_resolver),
            ):
                rows = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
                jan4 = [row for row in rows if row.ds == date(2025, 1, 4)]
                assert len(jan4) == 1
                assert jan4[0].lift is None
                assert jan4[0].failure_code == "readout.cell.missing_arm"
                jan5 = next(row for row in rows if row.ds == date(2025, 1, 5))
                assert (jan5.n_control, jan5.n_treat) == (8, 8)
                assert any(row.ds == date(2025, 1, 5) and row.lift is not None for row in rows)
        finally:
            adopted_resolver.close()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
def test_triggered_compliance_completes_control_only_date_with_missing_treatment():
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",), triggered=True)
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-04 00:00:00+00' "
        "WHERE event = 'triggered' AND group_id = 'control'"
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-05 00:00:00+00' "
        "WHERE event = 'triggered' AND group_id = 'treatment'"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (adopted_experiment, adopted_resolver),
            ):
                rows = view_readout(
                    source_experiment,
                    source_resolver,
                    Scope(view="asof", grain="asof", population="triggered"),
                    estimands=("compliance",),
                )
                jan4 = next(row for row in rows if row.ds == date(2025, 1, 4))
                assert jan4.group_id == "treatment"
                assert jan4.lift is None
                assert jan4.failure_code == "readout.cell.missing_arm"
                assert (jan4.n_control, jan4.n_treat) == (8, 0)
                jan5 = next(row for row in rows if row.ds == date(2025, 1, 5))
                assert jan5.lift is not None
                assert (jan5.n_control, jan5.n_treat) == (8, 8)
        finally:
            adopted_resolver.close()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.integration
def test_triggered_compliance_insufficient_early_sample_continues(clustered):
    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",), triggered=True, clustered=clustered
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-05 00:00:00+00' WHERE event = 'triggered'"
    )
    con.raw_sql(
        "UPDATE events SET ts = TIMESTAMPTZ '2025-01-04 00:00:00+00' "
        "WHERE event = 'triggered' AND user_id IN ('control0_0', 'treatment0_0')"
    )
    try:
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        try:
            for source_experiment, source_resolver in (
                (experiment, resolver),
                (adopted_experiment, adopted_resolver),
            ):
                with pytest.warns(IncrementRuntimeWarning) if clustered else nullcontext():
                    rows = view_readout(
                        source_experiment,
                        source_resolver,
                        Scope(view="asof", grain="asof", population="triggered"),
                        estimands=("compliance",),
                    )
                early = next(row for row in rows if row.ds == date(2025, 1, 4))
                assert early.lift is None
                assert early.failure_code == "readout.cell.missing_metric_observations"
                later = next(row for row in rows if row.ds == date(2025, 1, 5))
                assert later.lift is not None
        finally:
            adopted_resolver.close()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore:fetch_arrow_table.*:DeprecationWarning")
@pytest.mark.integration
def test_triggered_encouragement_itt_and_late_match_artifact_uptake(tmp_path):
    con, (analysis_experiment, analysis_resolver), context, store, _panel = native_fixture(
        ("full",), triggered=True
    )
    try:
        con.raw_sql(
            "UPDATE events SET ts = TIMESTAMPTZ '2025-01-04 00:00:00+00' WHERE event = 'triggered'"
        )
        native = bound_source(analysis_resolver)
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "trigger_measure_stats", "encouragement_uptake"}
        ]
        ref = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=extensions
        )
        with open_artifact(store, ref, expected_context=context) as artifact:
            assert any(ext.kind == "encouragement_uptake" for ext in artifact.manifest.extensions)
            native_uptake = _compliance(
                native.triggered_source(), design(7), population="triggered"
            )
            artifact_uptake = _compliance(
                cast(Any, artifact).triggered_source(), design(7), population="triggered"
            )
            assert artifact_uptake == native_uptake
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, ref, expected_context=context
        )
        with adopted_resolver:
            native_rows = _both_populations(
                analysis_experiment, analysis_resolver, estimands=["itt", "late"]
            )
            artifact_rows = _both_populations(
                adopted_experiment, adopted_resolver, estimands=["itt", "late"]
            )
        native_triggered = {
            row.estimand: row
            for row in native_rows
            if row.analysis_population == "triggered" and row.metric == "full"
        }
        artifact_triggered = {
            row.estimand: row
            for row in artifact_rows
            if row.analysis_population == "triggered" and row.metric == "full"
        }
        assert set(native_triggered) == set(artifact_triggered) == {"itt", "late"}
        for estimand in ("itt", "late"):
            assert native_triggered[estimand].require_lift().value == pytest.approx(
                artifact_triggered[estimand].require_lift().value
            )
            assert native_triggered[estimand].require_lift().value != 0
    finally:
        analysis_resolver.close()
        con.disconnect()


def _asof_lift(experiment, resolver):
    """The public as-of lift view as a callable taking the source-level keywords."""

    def read(*, completed_windows_only: bool = False, **options):
        scope = Scope(view="asof", grain="asof", completion_policy=completed_windows_only)
        return view_readout(experiment, resolver, scope, **options)

    return read


@pytest.mark.parametrize("path", ["analysis", "readouts"])
@pytest.mark.parametrize("outcome_window", [7, 9])
@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("metric_name", ["full", "drop"])
@pytest.mark.xdist_group("compliance_readonly")
def test_native_artifact_completed_compliance_staggered_enrollment(
    path, outcome_window, observed, metric_name, readonly_compliance_publications
):
    if observed:
        con, (experiment, resolver), context, store, _panel, ref = readonly_compliance_publications(
            (metric_name,), staggered=True, outcome=(8, "sum", outcome_window)
        )
    else:
        con, (experiment, resolver), context, store, _panel = native_fixture(
            (metric_name,), staggered=True, outcome=(8, "sum", outcome_window)
        )
        con.raw_sql(f"DELETE FROM events WHERE event = '{metric_name}_event'")
        ref = publish_unit_day_artifact(experiment, resolver, store)
    try:
        with open_artifact(store, ref, expected_context=context) as artifact:
            adopted_experiment, adopted_resolver = from_unit_day_artifact(
                store, ref, expected_context=context
            )
            try:
                for source, asof_lift in (
                    (bound_source(resolver), _asof_lift(experiment, resolver)),
                    (artifact, _asof_lift(adopted_experiment, adopted_resolver)),
                ):
                    route = (
                        asof_lift
                        if path == "analysis"
                        else lambda source=source, **kw: readouts.asof_lift(source, **kw)
                    )
                    results = route(estimands=["compliance"], completed_windows_only=True)
                    values = {row.ds: row.require_lift().value for row in results}
                    assert values == pytest.approx(
                        {
                            date(2025, 1, 8): 0.5,
                            date(2025, 1, 9): 0.5,
                            date(2025, 1, 10): 0.75,
                        }
                    )
                    assert all(row.inference == "fixed" for row in results)
            finally:
                adopted_resolver.close()
    finally:
        resolver.close()
        if not observed:
            con.disconnect()


@pytest.mark.parametrize("outcome_state", ["present", "absent", "null", "filtered"])
@pytest.mark.parametrize("separate_uptake", [False, True])
def test_open_ended_compliance_uses_persisted_enrollment_uptake_coverage(
    outcome_state, separate_uptake
):
    con, (experiment, resolver), context, store, _panel = native_fixture(
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
        native = bound_source(resolver)
        assert list(_compliance_dates(native)) == list(expected)
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            adopted_experiment, adopted_resolver = from_unit_day_artifact(
                store, ref, expected_context=context
            )
            try:
                for source, asof_lift in (
                    (native, _asof_lift(experiment, resolver)),
                    (artifact, _asof_lift(adopted_experiment, adopted_resolver)),
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
                adopted_resolver.close()
            con.raw_sql(f"DELETE FROM {uptake_table} WHERE event = 'clicked'")
            assert list(_compliance_dates(artifact)) == list(expected)
            assert list(_compliance_dates(native)) == [date(2025, 1, 1)]
            results = readouts.asof_lift(artifact, estimands=["compliance"])
            assert {row.ds: row.require_lift().value for row in results} == pytest.approx(expected)
    finally:
        resolver.close()
        con.disconnect()


def test_older_timestamp_artifact_keeps_digest_and_uses_known_compliance_coverage():
    from increment.semantics.artifact import EncouragementUptakeExtension

    con, (experiment, resolver), context, store, _panel = native_fixture(
        ("full",), open_ended=True, outcome_state="absent"
    )
    try:
        ref = publish_unit_day_artifact(experiment, resolver, store)
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
            assert list(_compliance_dates(artifact)) == [
                date(2025, 1, 1) + timedelta(days=day) for day in range(7)
            ]
            assert _moments(artifact, artifact.context.metrics[0], grain="asof") == ()
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.parametrize("mutate", [False, True])
def test_publication_preserves_relevant_event_rows_and_watermarks(monkeypatch, mutate):
    from increment.query.artifact_publish import ArtifactPublisher

    con, (experiment, resolver), context, store, _panel = native_fixture(("full",))

    try:
        con.raw_sql(
            "INSERT INTO events SELECT user_id, experiment_id, group_id, cluster_id, "
            "revenue, event, ts - INTERVAL 365 DAY FROM events WHERE event = 'full_event'"
        )
        con.raw_sql(
            "INSERT INTO events SELECT 'outsider', experiment_id, group_id, cluster_id, "
            "revenue, event, ts + INTERVAL 1 DAY FROM events WHERE event = 'full_event' LIMIT 1"
        )
        metric = bound_source(resolver).context.metrics[0]
        expected = _moments(bound_source(resolver), metric)
        validate = ArtifactPublisher._validate_stats_relation

        def mutate_after_validation(self, stats):
            result = validate(self, stats)
            con.raw_sql("UPDATE events SET ts = ts + INTERVAL 365 DAY WHERE event = 'full_event'")
            return result

        if mutate:
            monkeypatch.setattr(
                ArtifactPublisher, "_validate_stats_relation", mutate_after_validation
            )
        ref = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, ref, expected_context=context) as artifact:
            arm = _compliance(artifact, design(7)).arm("treatment")
            assert arm is not None and arm.uptake_total == 10
            assert {row["group_id"]: row for row in _moments(artifact, metric)} == {
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
        resolver.close()
        con.disconnect()


def test_publication_keeps_canonical_clusters_when_live_exposures_are_corrected(monkeypatch):
    con, (experiment, resolver), context, store, _panel = native_fixture(("full",), clustered=True)
    source = bound_source(resolver)
    try:
        expected = _compliance(source, design(7))
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
            reference = publish_unit_day_artifact(experiment, resolver, store)
        with open_artifact(store, reference, expected_context=context) as adopted:
            assert _compliance(adopted, design(7)) == expected
            assert _cluster_counts(adopted) == {"control": 12, "treatment": 12}
        with pytest.raises(CodedError) as raised:
            _compliance(source, design(7))
        assert raised.value.code == "query.integrity.cluster_conflict"
    finally:
        resolver.close()
        con.disconnect()


def test_design_summary_requires_complete_state_even_with_plan_override(tmp_path):
    import pyarrow.parquet as pq

    from increment.semantics.models import AnalysisPlan
    from increment.sources import (
        ASSIGNMENT_COUNTS_FIELD,
        COMPLIANCE_SUMMARY_FIELD,
        DECISION_PLAN_FIELD,
    )

    con, (experiment, resolver), _context, _store, _panel = native_fixture(())
    try:
        path = tmp_path / "design-only.parquet"
        bound_source(resolver).export_moments(path)
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
                from_moments(
                    [{**envelope, ASSIGNMENT_COUNTS_FIELD: counts}],
                    metrics=[],
                    design=design(7),
                    plan=AnalysisPlan(),
                )
            assert raised.value.code == "source.compliance_summary.invalid_state"
    finally:
        resolver.close()
        con.disconnect()


def test_design_summary_rejects_incompatible_shape_identity_catalog_and_plans(tmp_path):
    import pyarrow.parquet as pq

    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import DECISION_PLAN_FIELD

    con, (experiment, resolver), _context, _store, _panel = native_fixture(())
    try:
        path = tmp_path / "design-only.parquet"
        bound_source(resolver).export_moments(path)
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
        with pytest.raises(CodedError) as future:
            MomentsSource(
                [{**envelope, "moments_format": 14}],
                metrics=[],
                study_id="uptake_test",
                design=design(7),
                plan=AnalysisPlan(),
            )
        assert future.value.code == "moments.format.unsupported_future"
    finally:
        resolver.close()
        con.disconnect()
