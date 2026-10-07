from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import ibis
import pyarrow as pa
import pytest
import yaml

from examples._seed import seed_event_log
from increment._moment_plan import SLOTS
from increment.analysis import Analysis
from increment.errors import CapabilityError, CodedError
from increment.estimation.armstats import ArmStats
from increment.query import session as session_module
from increment.query.artifact_contract import (
    ArtifactContractError,
    compile_unit_day_artifact_context,
    unit_day_artifact_extension_catalog,
    validate_artifact_context,
)
from increment.query.artifact_digest import ArtifactDigestError, canonical_json
from increment.query.artifact_publish import artifact_context
from increment.query.session import WarehouseArtifactStore
from increment.query.source import open_artifact
from increment.semantics.artifact import ArtifactContext, UnitDayArtifactManifest
from increment.semantics.loader import load
from increment.semantics.models import Definitions
from tests.analysis_factory import _native_source, lift_rows

_PUBLICATION_GOLDEN = (
    Path(__file__).parent / "fixtures" / "unit_day_artifact_publication_golden.json"
)
# The format-2 manifest published before the context bound its window days. Saved
# verbatim; never regenerate it.
_PUBLICATION_GOLDEN_BEFORE_WINDOW_DAYS = (
    Path(__file__).parent
    / "fixtures"
    / "unit_day_artifact_publication_golden_format2_prechange.json"
)
# The manifest the previous writer produced for the same publication, when every
# relation carried digest_format 1. Saved verbatim; never regenerate it.
_PUBLICATION_GOLDEN_FORMAT_1 = (
    Path(__file__).parent / "fixtures" / "unit_day_artifact_publication_golden_format1.json"
)
# Site-volume context and catalog entry published before coverage followed the enrollment
# end. Saved verbatim; never regenerate it.
_SITE_VOLUME_CONTROL_PRECHANGE = (
    Path(__file__).parent / "fixtures" / "unit_day_artifact_site_volume_control_prechange.json"
)


class _FixedUUID:
    def __init__(self) -> None:
        self._values = iter(
            (
                uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                uuid.UUID("11111111-1111-1111-1111-111111111111"),
                uuid.UUID("22222222-2222-2222-2222-222222222222"),
                uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
            )
        )

    def uuid4(self) -> uuid.UUID:
        return next(self._values)

    def __getattr__(self, name: str) -> object:
        return getattr(uuid, name)


def _normalized_manifest_bytes(manifest: UnitDayArtifactManifest) -> str:
    body = manifest.model_dump(mode="json")
    body["created_at"] = "CREATED_AT"
    body["manifest_sha256"] = "MANIFEST_SHA256"
    return canonical_json(body)


def _normalized_saved_bytes(payload: str) -> str:
    """A saved golden's manifest bytes, blanked the same way for comparison."""
    body = json.loads(payload)
    body["created_at"] = "CREATED_AT"
    body["manifest_sha256"] = "MANIFEST_SHA256"
    return canonical_json(body)


def _expected_context(definitions: str | Path, experiment_name: str = "new_onboarding_v2"):
    """The context a consumer compiles for the artifact this analysis publishes."""
    defs = load(definitions)
    experiment = defs.experiment(experiment_name)
    assert experiment is not None
    return artifact_context(defs, experiment, "error")


def _canonical_definitions(definitions: Definitions) -> Definitions:
    """UTC-label naive timestamps so a payload round-trips through JSON as aware."""

    def aware(value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        if isinstance(value, dict):
            return {key: aware(item) for key, item in value.items()}
        if isinstance(value, list):
            return [aware(item) for item in value]
        if isinstance(value, tuple):
            return tuple(aware(item) for item in value)
        return value

    return Definitions.model_validate(aware(definitions.model_dump(mode="python")))


def _merge_centered_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool breakout rows with the same production law as native reduction."""
    arms = [ArmStats.model_validate({**row, "study_id": row["experiment_id"]}) for row in rows]
    pooled = ArmStats.combine(arms)
    return {"n": pooled.n, **{name: getattr(pooled, name) for name in SLOTS}}


_DEFINITIONS = Path(__file__).parents[1] / "examples" / "definitions"


def _definitions(
    tmp_path: Path,
    *,
    trigger: str | None = None,
    breakouts: list[dict[str, object]] | None = None,
    second_breakout_source: bool = False,
    cuped_metrics: tuple[str, ...] = (),
    site_volume_only: bool = False,
    open_ended: bool = False,
    empty_plan: bool = False,
    observation_end: str | None = None,
    day_boundary: str | None = None,
) -> Path:
    payload: dict[str, object] = {}
    for name in ("fact_sources.yaml", "exposures.yaml", "metrics.yaml", "experiments.yaml"):
        values = yaml.safe_load((_DEFINITIONS / name).read_text())
        payload.update(values)
    experiment = next(
        item
        for item in payload["experiments"]  # ty: ignore[not-iterable]
        if item["name"] == "new_onboarding_v2"
    )
    if trigger is not None:
        payload["exposures"].append({"name": trigger, "fact": "page_view"})  # ty: ignore[unresolved-attribute]
        experiment["trigger"] = trigger
    if breakouts is not None:
        experiment["breakouts"] = breakouts
        if any(item["property"] == "platform" for item in breakouts):
            for source in payload["fact_sources"]:  # ty: ignore[not-iterable]
                for prop in source.get("properties", []):
                    if prop["name"] == "platform":
                        prop["as_of"] = "static"
    if observation_end is not None:
        experiment["observation_end"] = observation_end
    if day_boundary is not None:
        experiment["day_boundary"] = day_boundary
    if open_ended:
        experiment["end"] = None
    if empty_plan:
        experiment["plan"] = {}
    if site_volume_only:
        experiment["plan"] = {"secondaries": ["purchase_rate"]}
    if cuped_metrics:
        method = {"name": "cuped", "variance_reduction": "cuped"}
        plan = experiment["plan"]
        for role in ("primary", "secondaries", "guardrails"):
            entries = plan.get(role)
            if entries is None:
                continue
            entries = [entries] if isinstance(entries, str) else entries
            plan[role] = [
                (
                    {
                        **({"metric": entry} if isinstance(entry, str) else entry),
                        "sensitivity_methods": [method],
                    }
                    if (entry if isinstance(entry, str) else entry["metric"]) in cuped_metrics
                    else entry
                )
                for entry in entries
            ]
    if second_breakout_source:
        payload["fact_sources"].append(  # ty: ignore[unresolved-attribute]
            {
                "name": "event_log_secondary",
                "sql": "SELECT * FROM analytics.event_log",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [{"name": "secondary_marker", "column": None}],
                "properties": [
                    {
                        "name": "country",
                        "column": "country_code",
                        "dtype": "string",
                        "as_of": "static",
                    }
                ],
            }
        )
        experiment["factors"] = []
        for source in payload["fact_sources"]:  # ty: ignore[not-iterable]
            for prop in source.get("properties", []):
                if prop["name"] == "platform":
                    prop["as_of"] = "static"
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _native(
    *,
    definitions: str | Path = "examples/definitions",
    with_pre_period: bool = False,
    with_late_returns: bool = False,
):
    con = ibis.duckdb.connect()
    seed_event_log(
        con,
        with_pre_period=with_pre_period,
        with_late_returns=with_late_returns,
    )
    analysis = Analysis.from_definitions("new_onboarding_v2", definitions, con)
    context = _expected_context(definitions)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    return con, analysis, context, store


def _published(
    *,
    extensions=(),
    definitions: str | Path = "examples/definitions",
    with_pre_period: bool = False,
    with_late_returns: bool = False,
):
    con, native, context, store = _native(
        definitions=definitions,
        with_pre_period=with_pre_period,
        with_late_returns=with_late_returns,
    )
    ref = native.publish_unit_day_artifact(store, extensions=extensions)
    return con, native, context, store, ref


def _extensions(context, *kinds: str):
    return [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in kinds
    ]


def _assert_moment_rows_agree(native_row: dict[str, Any], adopted_row: dict[str, Any]) -> None:
    """Every wire slot (n, x_role, and the full SLOTS family) must agree
    between the two paths, within a floor scaled to the row's own size."""
    assert native_row["group_id"] == adopted_row["group_id"]
    assert native_row["n"] == adopted_row["n"]
    assert native_row["x_role"] == adopted_row["x_role"]
    floor = abs(native_row["n"]) * 1e-9
    for slot in SLOTS:
        native_value = native_row[slot]
        adopted_value = adopted_row[slot]
        if native_value is None or adopted_value is None:
            assert native_value is None and adopted_value is None, slot
            continue
        assert adopted_value == pytest.approx(native_value, rel=1e-9, abs=floor), slot


def test_representative_publication_matches_golden_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = json.loads(_PUBLICATION_GOLDEN.read_text())
    monkeypatch.setattr(session_module, "uuid", _FixedUUID())
    definitions = _definitions(tmp_path, breakouts=[{"property": "country"}])
    _con, native, context, store = _native(definitions=definitions)
    extensions = _extensions(context, "breakout_dimension", "assignment_counts")
    ref = native.publish_unit_day_artifact(store, extensions=extensions)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert _normalized_manifest_bytes(manifest) == _normalized_saved_bytes(
        expected["manifest_bytes"]
    )
    assert [
        {
            "role": role,
            "schema_sha256": getattr(manifest.base, role).schema_sha256,
            "content_sha256": getattr(manifest.base, role).content_sha256,
            "row_count": getattr(manifest.base, role).row_count,
        }
        for role in ("exposures", "measure_stats")
    ] == expected["relation_digests"]


def test_format_two_golden_changes_only_by_the_bound_window_days() -> None:
    """The only payload deltas are the added window days and the source recipe digest they
    change; the relation data and relation digests are byte-identical."""
    before = json.loads(_PUBLICATION_GOLDEN_BEFORE_WINDOW_DAYS.read_text())
    after = json.loads(_PUBLICATION_GOLDEN.read_text())
    old_manifest = json.loads(before["manifest_bytes"])
    new_manifest = json.loads(after["manifest_bytes"])
    old_payload = json.loads(old_manifest["context"]["canonical_json"])
    new_payload = json.loads(new_manifest["context"]["canonical_json"])

    assert set(new_payload) - set(old_payload) == {"window_days"}
    assert set(old_payload) <= set(new_payload)
    assert (
        new_payload["source_mapping"]["recipe_sha256"]
        != old_payload["source_mapping"]["recipe_sha256"]
    )
    del new_payload["window_days"]
    new_payload["source_mapping"] = old_payload["source_mapping"]
    assert new_payload == old_payload

    assert new_manifest["context"]["sha256"] != old_manifest["context"]["sha256"]
    assert new_manifest["manifest_sha256"] != old_manifest["manifest_sha256"]
    for manifest in (old_manifest, new_manifest):
        del manifest["context"], manifest["manifest_sha256"], manifest["created_at"]
    assert new_manifest == old_manifest
    assert after["relation_digests"] == before["relation_digests"]


@pytest.mark.slow
def test_previous_writer_relations_match_the_saved_digests_and_its_context_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved = json.loads(_PUBLICATION_GOLDEN_FORMAT_1.read_text())
    monkeypatch.setattr(session_module, "uuid", _FixedUUID())
    # With no roles on format 2, the writer publishes the previous writer's relations.
    monkeypatch.setattr(session_module, "ARTIFACT_BASE_RELATION_ROLES", ())
    definitions = _definitions(tmp_path, breakouts=[{"property": "country"}])
    _con, native, context, store = _native(definitions=definitions)
    extensions = _extensions(context, "breakout_dimension", "assignment_counts")
    ref = native.publish_unit_day_artifact(store, extensions=extensions)
    monkeypatch.undo()
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    native.close()
    assert manifest.base.exposures.digest_format == manifest.base.measure_stats.digest_format == 1
    assert [
        {
            "role": role,
            "schema_sha256": getattr(manifest.base, role).schema_sha256,
            "content_sha256": getattr(manifest.base, role).content_sha256,
            "row_count": getattr(manifest.base, role).row_count,
        }
        for role in ("exposures", "measure_stats")
    ] == saved["relation_digests"]

    # The saved manifest embeds a format-1 context, which admission refuses by name.
    old_context = ArtifactContext.model_validate(json.loads(saved["manifest_bytes"])["context"])
    assert old_context.context_format == 1
    with pytest.raises(CodedError) as refused:
        validate_artifact_context(old_context)
    assert refused.value.code == "artifact.format.unsupported"
    assert refused.value.context["received"] == 1
    assert refused.value.context["supported"] == 2


# Guards the reader from recomputing a stored digest with any codec other than
# the one its digest_format names: the previous writer's format-1 base
# relations must verify on a backend that now publishes the fast format.
def test_previous_writer_base_relations_still_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "ARTIFACT_BASE_RELATION_ROLES", ())
    _con, native, context, store, ref = _published()
    monkeypatch.undo()
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert manifest.base.exposures.digest_format == manifest.base.measure_stats.digest_format == 1
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:
        adopted_rows = lift_rows(adopted.run(metrics=["purchase_rate"]))
        native_rows = lift_rows(native.run(metrics=["purchase_rate"]))
        assert [row.require_lift().value for row in adopted_rows] == pytest.approx(
            [row.require_lift().value for row in native_rows]
        )
    finally:
        adopted.close()
        native.close()


def test_publish_open_runs_the_same_arm_operations_and_reopens() -> None:
    _con, native, context, store, ref = _published()
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    native_rows = lift_rows(native.run(metrics=["purchase_rate"]))
    adopted_rows = lift_rows(adopted.run(metrics=["purchase_rate"]))
    assert adopted_rows[0].metric == native_rows[0].metric
    assert adopted_rows[0].require_lift().value == pytest.approx(
        native_rows[0].require_lift().value
    )
    assert len(adopted.run_daily(metrics=["purchase_rate"])) == len(
        native.run_daily(metrics=["purchase_rate"])
    )
    adopted.close()
    assert lift_rows(adopted.run(metrics=["purchase_rate"]))


def _add_treatment_units(con: Any) -> None:
    """Append new treatment-arm units so live data differs from earlier readings."""
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "SELECT * REPLACE (user_id || '_late' AS user_id, session_id || '_late' AS session_id) "
        "FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id <> "
        "(SELECT min(group_id) FROM analytics.event_log WHERE experiment_id = 'new_onboarding_v2')"
    )


def _purchase_lift(analysis: Analysis) -> float:
    return lift_rows(analysis.run(metrics=["purchase_rate"]))[0].require_lift().value


def test_closed_artifact_analysis_reopens_its_pinned_generation_not_a_newer_one() -> None:
    con, native, context, store, ref = _published()
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    before = _purchase_lift(adopted)
    adopted.close()
    _add_treatment_units(con)
    newer = native.publish_unit_day_artifact(store, refresh_of=ref)
    with Analysis.from_unit_day_artifact(store, newer, expected_context=context) as latest:
        assert _purchase_lift(latest) != pytest.approx(before)
    assert _purchase_lift(adopted) == pytest.approx(before)
    adopted.close()
    assert _purchase_lift(adopted) == pytest.approx(before)


def test_dashboard_snapshot_reads_the_pinned_source_and_leaves_the_outer_analysis_live() -> None:
    from increment._source_operations import DashboardSnapshotPayload

    con, native, _context, _store = _native()
    metrics = [m for m in load("examples/definitions").metrics if m.name == "purchase_rate"]
    seen: dict[str, float] = {}

    def operation(isolated: Analysis) -> DashboardSnapshotPayload:
        seen["pinned_before"] = _purchase_lift(isolated)
        _add_treatment_units(con)
        seen["pinned_after"] = _purchase_lift(isolated)
        return DashboardSnapshotPayload(
            allocation=None,
            allocation_refusal=None,
            allocation_history=None,
            allocation_history_refusal=None,
            estimates=(),
            group_data=(),
            explore=(),
        )

    try:
        native.dashboard_snapshot(operation, metrics=metrics)
        assert seen["pinned_after"] == pytest.approx(seen["pinned_before"])
        assert _purchase_lift(native) != pytest.approx(seen["pinned_before"])
    finally:
        native.close()


def test_dashboard_snapshot_refuses_an_artifact_backed_analysis_and_it_stays_usable() -> None:
    _con, _native_analysis, context, store, ref = _published()
    metrics = [m for m in load("examples/definitions").metrics if m.name == "purchase_rate"]
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    before = _purchase_lift(adopted)

    def operation(_isolated: Analysis) -> Any:
        raise AssertionError("the handler must not run on an artifact-backed analysis")

    with pytest.raises(CapabilityError) as raised:
        adopted.dashboard_snapshot(operation, metrics=metrics)
    assert raised.value.code == "facade.analysis.operation"
    assert raised.value.context["operation"] == "readout_snapshot"
    adopted.close()
    assert _purchase_lift(adopted) == pytest.approx(before)


def test_artifact_source_reports_closed_only_after_close() -> None:
    _con, _native_analysis, context, store, ref = _published()
    source = open_artifact(store, ref, expected_context=context)
    assert source.closed is False
    source.close()
    assert source.closed is True
    source.close()
    assert source.closed is True


@pytest.mark.parametrize("closer", ["parent", "triggered"])
def test_closing_either_view_of_a_snapshot_closes_both(tmp_path: Path, closer: str) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    _con, native, context, store = _native(definitions=definitions)
    ref = native.publish_unit_day_artifact(
        store, extensions=_extensions(context, "trigger_population", "assignment_counts")
    )
    parent = open_artifact(store, ref, expected_context=context)
    triggered = cast("Any", parent.triggered_source())
    assert (parent.closed, triggered.closed) == (False, False)
    (parent if closer == "parent" else triggered).close()
    assert (parent.closed, triggered.closed) == (True, True)
    parent.close()
    triggered.close()
    assert (parent.closed, triggered.closed) == (True, True)


def test_public_extension_selection_uses_the_immutable_catalog() -> None:
    _con, native, context, store = _native()
    request = next(
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "breakout_dimension"
    )
    ref = native.publish_unit_day_artifact(store, extensions=[request])
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    assert adopted.run_breakout(metrics=["purchase_rate"])
    with pytest.raises(ArtifactContractError):
        native.publish_unit_day_artifact(store, extensions=[request, request])
    with pytest.raises(ArtifactContractError):
        native.publish_unit_day_artifact(
            store,
            extensions=[{"kind": "breakout_dimension", "property_name": request.property_name}],
        )


def test_artifact_reader_and_source_openers_share_hydrated_contract() -> None:
    from increment.query.artifact_reader import open_artifact as reader_open
    from increment.query.source import open_artifact as facade_open

    _con, native, context, store = _native()
    request = _extensions(context, "breakout_dimension")[0]
    ref = native.publish_unit_day_artifact(store, extensions=[request])

    with (
        reader_open(store, ref, expected_context=context) as reader,
        facade_open(store, ref, expected_context=context) as facade,
    ):
        assert type(reader) is type(facade)
        assert reader.context.plan.declared is True
        assert reader.context.plan.declared == facade.context.plan.declared
        assert reader.context.design == facade.context.design
        assert reader.context.metrics == facade.context.metrics
        assert reader.operations == facade.operations
        assert "breakout_source" in reader.operations
        assert reader.breakouts == facade.breakouts == (request.property_name,)


def test_refresh_preserves_artifact_identity_and_rejects_cross_context() -> None:
    _con, native, context, store, first = _published()
    second = native.publish_unit_day_artifact(store, refresh_of=first)
    assert second.artifact_id == first.artifact_id and second.generation_id != first.generation_id
    other = Analysis.from_definitions("pricing_tier_test", "examples/definitions", _con)
    other_context = _expected_context("examples/definitions", "pricing_tier_test")
    with pytest.raises(ArtifactContractError):
        other.publish_unit_day_artifact(store, refresh_of=first)
    assert len(store.visible_manifests) == 2
    with pytest.raises(ArtifactContractError):
        Analysis.from_unit_day_artifact(store, first, expected_context=other_context)


# Guards triggered dispatch from silently returning only the assigned population.
def test_artifact_run_returns_assigned_and_triggered(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    _con, native, context, store = _native(definitions=definitions)
    selected = _extensions(context, "trigger_population", "assignment_counts")
    assert {request.kind for request in selected} == {
        "trigger_population",
        "assignment_counts",
    }
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    rows = lift_rows(adopted.run(metrics=["purchase_rate"]))
    assert rows
    assert {row.analysis_population for row in rows} == {"assigned", "triggered"}


# Guards site-volume lookup from confusing a metric's logical name with its physical measure key.
def test_site_volume_serves_logical_metric_names(tmp_path: Path) -> None:

    definitions = _definitions(tmp_path, site_volume_only=True)
    _con, native, context, store = _native(definitions=definitions)
    metric = next(item for item in native.metrics if item.name == "purchase_rate")
    assert metric.name != metric.fact
    selected = _extensions(context, "site_volume")
    assert selected
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    result = adopted.sitewide("purchase_rate")
    assert result.site_total_volume > 0  # ty: ignore[unresolved-attribute]


def test_site_volume_coverage_with_no_declared_metrics_omits_extension(
    tmp_path: Path,
) -> None:
    definitions = _definitions(tmp_path, empty_plan=True)
    _con, native, _context, store = _native(definitions=definitions)
    coverage = {
        "first_ds": "2024-01-01",
        "last_ds": "2024-01-02",
        "freshness": {"loaded_through": "2024-01-02", "declared_complete": True},
    }
    context = compile_unit_day_artifact_context(
        native.experiment.name,
        load(definitions),
        site_volume_coverage=coverage,
    )
    assert not any(
        entry.request.kind == "site_volume"
        for entry in unit_day_artifact_extension_catalog(context)
    )
    assert native.publish_unit_day_artifact(store)


# Guards reduction from accepting stored date rows with a non-canonical physical timestamp type.
def test_non_canonical_stored_rows_refuse() -> None:
    _con, _native_analysis, context, store, ref = _published()
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    stats_relation = manifest.base.measure_stats.relation
    stats_name = stats_relation.name
    stats_schema = stats_relation.schema_name
    _con.raw_sql(
        f'''CREATE OR REPLACE TABLE "{stats_schema}"."{stats_name}" AS
        SELECT experiment_id, unit_id, CAST(ds AS TIMESTAMP) AS ds, measure_key,
               n_events, sum_value, min_value, max_value
        FROM "{stats_schema}"."{stats_name}"'''
    )
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    with pytest.raises(ArtifactDigestError) as raised:
        lift_rows(adopted.run(metrics=["purchase_rate"]))
    assert raised.value.code == "artifact.digest.type"


# Guards CUPED dispatch from dropping pre-period covariate moments on breakout and retention-day paths.
def test_adopted_breakout_cuped_covariates_present(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, cuped_metrics=("purchase_rate",))
    _con, native, context, store = _native(definitions=definitions, with_pre_period=True)
    assert native.experiment.n_pre_periods > 0
    requests = _extensions(context, "breakout_dimension", "cuped_preperiod")
    assert any(request.kind == "cuped_preperiod" for request in requests)
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    metric = adopted.metrics[0]
    breakout = adopted.experiment.breakouts[0]
    with open_artifact(store, ref, expected_context=context) as source:
        breakout_rows: Any = source.breakout_moments(
            metric,
            breakout,
            grain="total",
            include_covariate=True,
        ).rows
    native_rows: Any = (
        _native_source(native)
        .breakout_source(breakout, metrics=[metric])
        .moments(metric, by=["country"])
    )
    assert breakout_rows and native_rows
    assert any(abs(float(row["ref_x"])) > 0 for row in breakout_rows)
    assert any(abs(float(row["ref_x"])) > 0 for row in native_rows)
    assert sorted(row["ref_x"] for row in breakout_rows) == pytest.approx(
        sorted(row["ref_x"] for row in native_rows)
    )
    assert adopted.run_breakout(metrics=["purchase_rate"])


# Guards bounded-retention daily lift from losing its non-null CUPED covariate moments.
def test_adopted_retention_cuped_covariates_present(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, cuped_metrics=("d7_retention",))
    _con, native, context, store = _native(
        definitions=definitions,
        with_pre_period=True,
        with_late_returns=True,
    )
    assert native.experiment.n_pre_periods > 0
    selected = _extensions(context, "cuped_preperiod")
    assert selected
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    retention = next(metric for metric in adopted.metrics if metric.name == "d7_retention")
    with open_artifact(store, ref, expected_context=context) as source:
        daily_rows = source.moments(retention, grain="daily", include_covariate=True)
    native_retention = next(metric for metric in native.metrics if metric.name == "d7_retention")
    native_rows = (
        _native_source(native)
        .day_source(metrics=[native_retention])
        .moments(native_retention, grain="daily", include_covariate=True)
    )
    assert daily_rows and native_rows
    assert any(abs(float(row["ref_x"])) > 0 for row in daily_rows)
    assert any(abs(float(row["ref_x"])) > 0 for row in native_rows)
    assert sorted(row["ref_x"] for row in daily_rows) == pytest.approx(
        sorted(row["ref_x"] for row in native_rows)
    )
    assert adopted.run_daily_lift(metrics=["d7_retention"])


# Dimension extensions preserve missing-property units and exclude fact-only units.
@pytest.mark.slow
@pytest.mark.parametrize("as_of", ["static", "pre_exposure"])
def test_dimension_extensions_are_left_joined_to_exposures(tmp_path: Path, as_of: str) -> None:
    definitions = _definitions(
        tmp_path,
        breakouts=[{"property": "country"}],
    )
    payload = yaml.safe_load(definitions.read_text())
    for fact_source in payload["fact_sources"]:
        for prop in fact_source.get("properties", []):
            if prop["name"] == "country":
                prop["as_of"] = as_of
                fact_source["sql"] = (
                    f"SELECT * FROM ({fact_source['sql']}) AS property_rows "
                    "WHERE user_id <> 'u00000'"
                )
    payload["exposures"].append(
        {
            "name": "independent_enrollment",
            "sql": (
                "SELECT user_id AS unit_id, event_at AS ts, group_id "
                "FROM analytics.event_log WHERE event = 'page_view' "
                "AND experiment_id = 'new_onboarding_v2'"
            ),
        }
    )
    for experiment in payload["experiments"]:
        if experiment["name"] == "new_onboarding_v2":
            experiment["exposure"] = "independent_enrollment"
    definitions.write_text(yaml.safe_dump(payload))
    con, native, context, store = _native(definitions=definitions, with_pre_period=True)
    try:
        event_log = con.table("event_log", database="analytics").to_pyarrow()
        rows = event_log.to_pylist()
        for row in rows:
            if row["user_id"] == "u00001":
                row["country_code"] = None
        rows.append(
            {
                "event_at": datetime(2025, 2, 10, 9, tzinfo=UTC),
                "user_id": "u99999",
                "session_id": "u99999-sghost",
                "event": "freshness-only",
                "revenue": None,
                "duration_s": None,
                "country_code": "GB",
                "device_type": "web",
                "plan": "free",
                "experiment_id": None,
                "group_id": None,
            }
        )
        con.create_table(
            "event_log",
            pa.Table.from_pylist(rows, schema=event_log.schema),
            database="analytics",
            overwrite=True,
        )

        requests = [
            request
            for request in _extensions(context, "breakout_dimension", "factor_dimension")
            if request.property_name == "country"
        ]
        assert {request.kind for request in requests} == {
            "breakout_dimension",
            "factor_dimension",
        }
        ref = native.publish_unit_day_artifact(store, extensions=requests)
        with open_artifact(store, ref, expected_context=context) as source:
            exposure_ids = {str(row["unit_id"]) for row in source._exposure_rows()}
            for request in requests:
                request_payload = request.model_dump(mode="json")
                extension = source._extension(request_payload)
                rows = source._read_extension(extension, request=request_payload)
                by_unit = {row["unit_id"]: row for row in rows}
                assert set(by_unit) == exposure_ids
                assert "u99999" not in by_unit
                assert by_unit["u00000"]["value_is_missing"] is True
                assert by_unit["u00001"]["value_is_missing"] is True
    finally:
        native.close()
        con.disconnect()


# Guards partial breakout extension coverage from refusing by name.
def test_incomplete_breakout_coverage_refuses_before_reduction(tmp_path: Path) -> None:
    definitions = _definitions(
        tmp_path,
        breakouts=[{"property": "country"}, {"property": "platform"}],
    )
    _con, native, context, store = _native(definitions=definitions)
    country_request = next(
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "breakout_dimension" and entry.request.property_name == "country"
    )
    ref = native.publish_unit_day_artifact(store, extensions=[country_request])
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    with pytest.raises(ArtifactContractError) as raised:
        adopted.run_breakout(metrics=["purchase_rate"])
    assert raised.value.code == "artifact.extension.missing"


# Guards refused artifact opens from leaking the snapshot handle after context validation fails.
def test_refused_open_releases_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _con, native, context, store, ref = _published()
    bad_context = _expected_context("examples/definitions", "pricing_tier_test")
    original = store.open_snapshot
    state = {"entered": 0, "exited": 0, "active": 0}

    @contextmanager
    def spy_open_snapshot(snapshot_ref):
        state["entered"] += 1
        state["active"] += 1
        try:
            with original(snapshot_ref) as snapshot:
                yield snapshot
        finally:
            state["active"] -= 1
            state["exited"] += 1

    monkeypatch.setattr(store, "open_snapshot", spy_open_snapshot)
    with pytest.raises(ArtifactContractError) as raised:
        Analysis.from_unit_day_artifact(store, ref, expected_context=bad_context)
    assert raised.value.code == "artifact.refresh.context_mismatch"
    assert state == {"entered": 1, "exited": 1, "active": 0}
    native.close()


# Guards same-name breakout routing from collapsing dimensions resolved from distinct fact sources.
@pytest.mark.slow
def test_multi_source_same_property_breakouts_route_exactly(tmp_path: Path) -> None:
    definitions = _definitions(
        tmp_path,
        breakouts=[
            {"property": "country", "source": "event_log"},
            {"property": "country", "source": "event_log_secondary"},
        ],
        second_breakout_source=True,
    )
    _con, native, context, store = _native(definitions=definitions)
    requests = _extensions(context, "breakout_dimension")
    assert len(requests) == 2
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    daily = adopted.run_daily(dimension="country", metrics=["purchase_rate"])
    assert daily and {row.source for row in daily} == {"event_log", "event_log_secondary"}
    breakout = adopted.run_breakout(metrics=["purchase_rate"])
    assert breakout and {row.source for row in breakout} == {
        "event_log",
        "event_log_secondary",
    }
    daily_lift = adopted.run_daily_lift(dimension="country", metrics=["purchase_rate"])
    assert daily_lift and {row.source for row in daily_lift} == {
        "event_log",
        "event_log_secondary",
    }
    asof = adopted.run_asof(dimension="country", metrics=["purchase_rate"])
    assert asof and {row.source for row in asof} == {"event_log", "event_log_secondary"}
    asof_lift = adopted.run_asof_lift(dimension="country", metrics=["purchase_rate"])
    assert asof_lift and {row.source for row in asof_lift} == {
        "event_log",
        "event_log_secondary",
    }


def test_open_ended_experiment_omits_site_volume(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, open_ended=True)
    _con, native, context, store = _native(definitions=definitions)
    assert not any(
        entry.request.kind == "site_volume"
        for entry in unit_day_artifact_extension_catalog(context)
    )
    ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    assert lift_rows(adopted.run(metrics=["purchase_rate"]))


@pytest.mark.slow
def test_triggered_artifact_serves_full_population_extensions(tmp_path: Path) -> None:
    definitions = _definitions(
        tmp_path,
        trigger="session_start",
        breakouts=[{"property": "country"}],
        cuped_metrics=("purchase_rate",),
    )
    _con, native, context, store = _native(definitions=definitions, with_pre_period=True)
    selected = _extensions(
        context, "trigger_population", "assignment_counts", "breakout_dimension", "cuped_preperiod"
    )
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    assert lift_rows(adopted.run(metrics=["purchase_rate"]))
    metric = adopted.metrics[0]
    with open_artifact(store, ref, expected_context=context) as source:
        triggered_rows = source.triggered_source().moments(
            metric,
            grain="total",
            by=["country"],
            include_covariate=True,
        )
    assert triggered_rows
    assert all(row["country"] is not None for row in triggered_rows)
    assert any(row["ref_x"] is not None for row in triggered_rows)


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster")
def test_clustered_sitewide_supplies_assignment_counts(tmp_path: Path) -> None:
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered

    con = ibis.duckdb.connect()
    con.create_table("cluster_trigger_events", obj=_clustered_trigger_events(n_stores_per_arm=40))
    definitions = tmp_path / "clustered.yml"
    definitions.write_text(
        _defs_yaml_clustered(trigger=None).replace(
            "start: 2024-01-01T00:00:00", "start: 2024-01-01T00:00:00\n    end: 2024-01-09T00:00:00"
        )
    )
    native = Analysis.from_definitions("exp", definitions, con)
    context = _expected_context(definitions, "exp")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    selected = _extensions(context, "site_volume", "cluster_identity")
    assert selected
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    result = adopted.sitewide("revenue")
    assert result.delta is not None  # ty: ignore[unresolved-attribute]
    from increment.errors import CapabilityError

    for method in (
        adopted.run_daily,
        adopted.run_daily_lift,
        adopted.run_asof,
        adopted.run_asof_lift,
    ):
        with pytest.raises(CapabilityError) as raised:
            method(metrics=["revenue"])
        assert raised.value.code == "facade.analysis.clustered_day_axis"
        assert raised.value.context["method"] == method.__name__


@pytest.mark.slow
def test_clustered_artifact_moments_transport_refuses_before_writing(tmp_path: Path) -> None:
    import warnings

    from increment.errors import IncrementRuntimeWarning
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered
    from tests.warning_codes import warning_codes

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "cluster_trigger_events", obj=_clustered_trigger_events(n_stores_per_arm=8)
        )
        payload = yaml.safe_load(_defs_yaml_clustered(trigger=None))
        payload["metrics"] = [
            {
                "name": "revenue",
                "type": "ratio",
                "entity": "user_id",
                "numerator": {"fact": "revenue", "aggregation": "sum", "window_days": 7},
                "denominator": {"fact": "revenue", "aggregation": "count", "window_days": 7},
            }
        ]
        definitions = tmp_path / "clustered-transport.yml"
        definitions.write_text(yaml.safe_dump(payload))
        with Analysis.from_definitions("exp", definitions, con) as native:
            context = _expected_context(definitions, "exp")
            store = WarehouseArtifactStore(con, schema_name="artifacts")
            ref = native.publish_unit_day_artifact(
                store, extensions=_extensions(context, "cluster_identity", "assignment_counts")
            )
            with Analysis.from_unit_day_artifact(store, ref, expected_context=context) as adopted:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", IncrementRuntimeWarning)
                    (native_row,) = lift_rows(native.run())
                    (artifact_row,) = lift_rows(adopted.run())
                assert set(warning_codes(caught)) == {"estimation.engine.small_total_clusters"}
                for row in (native_row, artifact_row):
                    assert row.reference_kind == "t"
                    assert row.reference_df == 7
                    assert row.n_clusters == 16
                assert artifact_row.require_lift().lb == pytest.approx(native_row.require_lift().lb)
                assert artifact_row.require_lift().ub == pytest.approx(native_row.require_lift().ub)
                destination = tmp_path / "artifact-moments.parquet"
                destination.write_bytes(b"preserve")
                with pytest.raises(CapabilityError) as raised:
                    adopted.export(destination)
                assert raised.value.code == "source.moments.cluster_grain"
                assert raised.value.context["operation"] == "export_moments"
                assert raised.value.context["source"] == "artifact"
                assert raised.value.context["cluster"] == "store_id"
                assert raised.value.context["design"] == "randomized"
                assert raised.value.context["mechanism"] == "randomized"
                assert destination.read_bytes() == b"preserve"
    finally:
        con.disconnect()


def _triggered_definitions(tmp_path: Path) -> Path:
    base = Path("examples/definitions")
    payload: dict[str, object] = {}
    for name in ("fact_sources.yaml", "exposures.yaml", "metrics.yaml", "experiments.yaml"):
        payload.update(yaml.safe_load((base / name).read_text()))
    payload["exposures"].append({"name": "session_start", "fact": "page_view"})  # ty: ignore[unresolved-attribute]
    experiment = next(
        e
        for e in payload["experiments"]  # ty: ignore[not-iterable]
        if e["name"] == "new_onboarding_v2"
    )
    experiment["trigger"] = "session_start"
    experiment["breakouts"] = [{"property": "country"}]
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _triggered_subset_definitions(tmp_path: Path) -> Path:
    """A trigger admitting a proper subset (not all) of enrolled units, with
    CUPED turned on so both trigger-restriction and covariate reads are exercised."""
    base = Path("examples/definitions")
    payload: dict[str, object] = {}
    for name in ("fact_sources.yaml", "exposures.yaml", "metrics.yaml", "experiments.yaml"):
        payload.update(yaml.safe_load((base / name).read_text()))
    payload["exposures"].append(  # ty: ignore[unresolved-attribute]
        {
            "name": "country_us_activation",
            "fact": "page_view",
            "filters": [{"property": "country", "op": "equals", "values": ["US"]}],
        }
    )
    experiment = next(
        e
        for e in payload["experiments"]  # ty: ignore[not-iterable]
        if e["name"] == "new_onboarding_v2"
    )
    experiment["trigger"] = "country_us_activation"
    experiment["breakouts"] = [{"property": "country"}]
    experiment["plan"]["secondaries"] = [
        (
            {
                "metric": "purchase_rate",
                "sensitivity_methods": [{"name": "cuped", "variance_reduction": "cuped"}],
            }
            if entry == "purchase_rate"
            else entry
        )
        for entry in experiment["plan"]["secondaries"]
    ]
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.mark.slow
def test_triggered_subset_dimension_and_cuped_extensions_match_native(tmp_path: Path) -> None:
    """A trigger that admits only a subset of enrolled units must yield the
    same moments, breakout sums, and public lift estimates on both paths."""
    definitions = _triggered_subset_definitions(tmp_path)
    _con, native, context, store = _native(definitions=definitions, with_pre_period=True)
    requests = _extensions(
        context, "trigger_population", "assignment_counts", "breakout_dimension", "cuped_preperiod"
    )
    assert any(request.kind == "cuped_preperiod" for request in requests)
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:
        native_source = _native_source(native)
        native_triggered = native_source.triggered_source()
        # Confirm the fixture is genuinely a proper subset, not 100% coverage
        # (the bug this test guards is invisible at 100% triggered).
        triggered_count = sum(native_triggered.unit_counts().values())
        enrolled_count = sum(native_source.unit_counts().values())
        assert 0 < triggered_count < enrolled_count

        native_metric = native.metrics[0]
        adopted_metric = adopted.metrics[0]
        # Native lacks a triggered, breakout-dimensioned "total" summary (see
        # below), so the reference is the triggered population's undimensioned
        # CUPED moments; dimensioned artifact rows must sum to its per-arm totals.
        native_rows = cast(
            "list[dict[str, Any]]",
            native_triggered.moments(native_metric, grain="total", include_covariate=True),
        )
        with open_artifact(store, ref, expected_context=context) as source:
            triggered = source.triggered_source()
            adopted_rows = cast(
                "list[dict[str, Any]]",
                triggered.moments(adopted_metric, grain="total", include_covariate=True),
            )
            adopted_breakout_rows = cast(
                "list[dict[str, Any]]",
                triggered.moments(
                    adopted_metric, grain="total", by=["country"], include_covariate=True
                ),
            )
        assert native_rows and adopted_rows and adopted_breakout_rows
        native_by_group = {row["group_id"]: row for row in native_rows}
        adopted_by_group = {row["group_id"]: row for row in adopted_rows}
        assert set(native_by_group) == set(adopted_by_group)
        for group_id, native_row in native_by_group.items():
            adopted_row = adopted_by_group[group_id]
            _assert_moment_rows_agree(native_row, adopted_row)

        # Merge centered rows with the same between-partition correction used
        # by the native reducer, then compare every wire slot (including
        # optional covariance and uptake slots) with the native total.
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in adopted_breakout_rows:
            assert row["country"] == "US"
            by_group.setdefault(row["group_id"], []).append(row)
        merged_by_group = {
            group_id: _merge_centered_rows(rows) for group_id, rows in by_group.items()
        }
        for group_id, native_row in native_by_group.items():
            assert group_id in merged_by_group
            merged_row = merged_by_group[group_id]
            for slot_name in ("n", *SLOTS):
                actual = merged_row[slot_name]
                expected = native_row[slot_name]
                if expected is None:
                    assert actual is None, (group_id, slot_name)
                else:
                    assert actual == pytest.approx(
                        expected, rel=1e-9, abs=abs(native_row["n"]) * 1e-9
                    ), (group_id, slot_name)
        # A declared trigger's breakout has no public route on either path;
        # both must refuse it identically, not silently diverge.
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as native_raised:
            native.run_breakout()
        with pytest.raises(CapabilityError) as adopted_raised:
            adopted.run_breakout()
        assert native_raised.value.code == adopted_raised.value.code

        # The public whole-window readout (decision + CUPED sensitivity,
        # both assigned- and triggered-population rows) must match exactly.
        native_lift = lift_rows(native.run(metrics=["purchase_rate"]))
        adopted_lift = lift_rows(adopted.run(metrics=["purchase_rate"]))

        def _row_key(row: Any) -> tuple[Any, ...]:
            return (row.metric, row.method, row.method_role, row.group_id, row.analysis_population)

        native_by_key = {_row_key(row): row for row in native_lift}
        adopted_by_key = {_row_key(row): row for row in adopted_lift}
        assert native_by_key and set(native_by_key) == set(adopted_by_key)
        for key, native_lift_row in native_by_key.items():
            adopted_lift_row = adopted_by_key[key]
            native_lift_value = native_lift_row.require_lift()
            adopted_lift_value = adopted_lift_row.require_lift()
            assert adopted_lift_value.value == pytest.approx(native_lift_value.value, rel=1e-9), key
            assert adopted_lift_value.lb == pytest.approx(native_lift_value.lb, rel=1e-9), key
            assert adopted_lift_value.ub == pytest.approx(native_lift_value.ub, rel=1e-9), key
    finally:
        adopted.close()
        native.close()


@pytest.mark.slow
def test_artifact_run_breakout_refuses_a_declared_trigger(tmp_path: Path) -> None:
    """An artifact-backed readout that cannot honor the declared trigger must
    refuse exactly as the definitions-backed one does."""
    from increment.errors import CapabilityError

    con = ibis.duckdb.connect()
    seed_event_log(con)
    definitions = _triggered_definitions(tmp_path)
    native = Analysis.from_definitions("new_onboarding_v2", definitions, con)
    context = _expected_context(definitions)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    extensions = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in ("trigger_population", "assignment_counts", "breakout_dimension")
    ]
    ref = native.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)

    with pytest.raises(CapabilityError) as raised:
        native.run_breakout()
    assert raised.value.code == "facade.analysis.trigger_unsupported"
    with pytest.raises(CapabilityError) as raised:
        adopted.run_breakout()
    assert raised.value.code == "facade.analysis.trigger_unsupported"


def test_site_volume_extension_counts_non_enrolled_units_in_window(tmp_path: Path) -> None:
    """Regression: the enrolled-unit semi-join must not shrink site_volume's
    whole-population reading to enrolled units only. The site branch counts every
    unit's events in [start, end], independent of enrollment (matching
    builders.py's site_volume() -- "No exposure join: a unit that never appears in
    any exposure table still counts", tests/query/test_builders.py:6008). Seeds one
    purchase event for a user who is never exposed at all, inside the experiment
    window, and asserts the whole-site reading includes it."""
    definitions = _definitions(tmp_path, site_volume_only=True)
    con, native, context, store = _native(definitions=definitions)
    con.raw_sql(
        """
        INSERT INTO analytics.event_log
        (event_at, user_id, session_id, event, revenue, duration_s,
         country_code, device_type, plan, experiment_id, group_id)
        VALUES
        (TIMESTAMP '2025-01-20 00:00:00', 'never_enrolled_user', 'never_enrolled_user-s0',
         'purchase', NULL, NULL, 'US', 'desktop', 'free', NULL, NULL)
        """
    )
    expected_events = (
        con.sql(
            "SELECT COUNT(*) AS n FROM analytics.event_log WHERE event = 'purchase' "
            "AND event_at BETWEEN TIMESTAMP '2025-01-15' AND TIMESTAMP '2025-02-15 23:59:59'"
        )
        .to_pandas()["n"]
        .iloc[0]
    )
    selected = _extensions(context, "site_volume")
    assert selected
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    result = adopted.sitewide("purchase_rate")
    assert result.site_total_volume == pytest.approx(float(expected_events))  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize(
    ("day_boundary", "window_start", "window_end"),
    [
        (None, "2025-01-15 00:00:00", "2025-02-16 00:00:00"),
        ("UTC-05:00", "2025-01-15 05:00:00", "2025-02-16 05:00:00"),
    ],
)
def test_site_volume_coverage_ends_at_the_enrollment_end_not_the_observation_end(
    tmp_path: Path, day_boundary: str | None, window_start: str, window_end: str
) -> None:
    """Site volume is measured over [start, end]; a later observation_end must not
    widen the declared coverage, the stored rows, or the adopted total."""
    from increment.query.artifact_extensions import read_site_volume_extension

    definitions = _definitions(
        tmp_path,
        site_volume_only=True,
        observation_end="2025-03-01",
        day_boundary=day_boundary,
    )
    con, native, context, store = _native(definitions=definitions)
    con.raw_sql(
        """
        INSERT INTO analytics.event_log
        (event_at, user_id, session_id, event, revenue, duration_s,
         country_code, device_type, plan, experiment_id, group_id)
        VALUES
        (TIMESTAMP '2025-02-15 20:00:00', 'edge_a', 'edge_a-s0', 'purchase',
         NULL, NULL, 'US', 'desktop', 'free', NULL, NULL),
        (TIMESTAMP '2025-02-16 02:00:00', 'edge_b', 'edge_b-s0', 'purchase',
         NULL, NULL, 'US', 'desktop', 'free', NULL, NULL),
        (TIMESTAMP '2025-02-20 12:00:00', 'after_end', 'after_end-s0', 'purchase',
         NULL, NULL, 'US', 'desktop', 'free', NULL, NULL)
        """
    )
    expected = float(
        con.sql(
            "SELECT COUNT(*) AS n FROM analytics.event_log WHERE event = 'purchase' "
            f"AND event_at >= TIMESTAMP '{window_start}' AND event_at < TIMESTAMP '{window_end}'"
        )
        .to_pandas()["n"]
        .iloc[0]
    )
    (request,) = _extensions(context, "site_volume")
    ref = native.publish_unit_day_artifact(store, extensions=(request,))
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        extension = next(item for item in manifest.extensions if item.kind == "site_volume")
        rows = read_site_volume_extension(
            snapshot,
            extension,
            request=request,
            context=context,
            experiment_id=manifest.experiment_id,
        )
    assert extension.first_ds == date(2025, 1, 15)
    assert extension.last_ds == date(2025, 2, 15)
    assert extension.freshness.loaded_through == date(2025, 2, 15)
    assert {row["ds"] for row in rows} == {
        date(2025, 1, 15) + timedelta(days=offset) for offset in range(32)
    }
    native_total = native.sitewide("purchase_rate").site_total_volume
    adopted_total = adopted.sitewide("purchase_rate").site_total_volume  # ty: ignore[unresolved-attribute]
    assert native_total == pytest.approx(expected)
    assert adopted_total == pytest.approx(expected)


@pytest.mark.parametrize("label", ["utc", "utc_minus_05"])
def test_site_volume_context_without_observation_end_is_unchanged(
    tmp_path: Path, label: str
) -> None:
    """Without an observation_end the horizon equals the end, so the context and the
    site-volume catalog entry keep the bytes published before coverage followed the end."""
    frozen = json.loads(_SITE_VOLUME_CONTROL_PRECHANGE.read_text())[label]
    definitions = _definitions(
        tmp_path,
        site_volume_only=True,
        day_boundary="UTC-05:00" if label == "utc_minus_05" else None,
    )
    context = _expected_context(definitions)
    (entry,) = [
        item
        for item in unit_day_artifact_extension_catalog(context)
        if item.request.kind == "site_volume"
    ]
    assert context.canonical_json == frozen["context_canonical_json"]
    assert entry.definition_sha256 == frozen["site_volume_definition_sha256"]
    assert entry.canonical_definition_json == frozen["site_volume_definition_json"]


def test_site_volume_extension_matches_unpinned_total_when_filtered_on_a_joined_dim(
    tmp_path: Path,
) -> None:
    """Regression: the fact-level site_volume fix alone is not enough when the metric's
    own filter reads a property that only exists via a joined DimSource (fact_resolution.py's
    _dim_joined_fact_table -> query/dims.py's join_dims, always a LEFT join -- see that
    module's docstring). Pre-fix, _pinned_source's dims loop scoped every joined dim to
    enrolled units regardless of which fact source pulled it in, so a non-enrolled unit's
    fact row would LEFT JOIN to a NULL property and silently fail the metric's filter,
    re-shrinking the site_volume total back toward the enrolled population one join away
    from the already-fixed fact source. Builds a dedicated DimSource + FactSource + Metric
    (metrics.yaml's `filters:` mechanism, the same one avg_session_duration already uses to
    filter on the same-table `platform` property) so the property is reachable ONLY through
    the dim join, then asserts the published total equals a direct, unpinned SQL sum over
    the same window -- including a synthetic purchase from a user who is never enrolled at
    all (same injection technique as the test above)."""
    payload: dict[str, object] = {}
    for name in ("fact_sources.yaml", "exposures.yaml", "metrics.yaml", "experiments.yaml"):
        values = yaml.safe_load((_DEFINITIONS / name).read_text())
        payload.update(values)
    experiment = next(
        item
        for item in payload["experiments"]  # ty: ignore[not-iterable]
        if item["name"] == "new_onboarding_v2"
    )
    payload["dim_sources"] = [
        {
            "name": "user_loyalty",
            "sql": "SELECT DISTINCT user_id, 'gold' AS loyalty_tier FROM analytics.event_log",
            "entity": "user_id",
            "properties": [
                {
                    "name": "loyalty_tier",
                    "column": "loyalty_tier",
                    "dtype": "string",
                    "as_of": "static",
                }
            ],
        }
    ]
    payload["fact_sources"].append(  # ty: ignore[unresolved-attribute]
        {
            "name": "loyalty_purchases",
            "sql": (
                "SELECT event_at, user_id, session_id, revenue, duration_s, country_code, "
                "device_type, plan, experiment_id, group_id, 'loyalty_purchase' AS event "
                "FROM analytics.event_log WHERE event = 'purchase'"
            ),
            "timestamp_column": "event_at",
            "entities": ["user_id"],
            "dims": ["user_loyalty"],
            "facts": [{"name": "loyalty_purchase", "column": "revenue"}],
        }
    )
    payload["metrics"].append(  # ty: ignore[unresolved-attribute]
        {
            "type": "mean",
            "name": "loyalty_purchase_site_volume",
            "description": "Gold-tier purchase revenue, whole-site",
            "entity": "user_id",
            "preferred_direction": "increase",
            "fact": "loyalty_purchase",
            "aggregation": "sum",
            "window_days": 14,
            "filters": [{"property": "loyalty_tier", "op": "equals", "values": ["gold"]}],
        }
    )
    experiment["breakouts"] = []
    experiment["plan"] = {"secondaries": ["loyalty_purchase_site_volume"]}
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    con, native, context, store = _native(definitions=path)
    con.raw_sql(
        """
        INSERT INTO analytics.event_log
        (event_at, user_id, session_id, event, revenue, duration_s,
         country_code, device_type, plan, experiment_id, group_id)
        VALUES
        (TIMESTAMP '2025-01-20 00:00:00', 'never_enrolled_user', 'never_enrolled_user-s0',
         'purchase', 42.0, NULL, 'US', 'desktop', 'free', NULL, NULL)
        """
    )
    expected_total = (
        con.sql(
            "SELECT SUM(revenue) AS total FROM analytics.event_log WHERE event = 'purchase' "
            "AND event_at BETWEEN TIMESTAMP '2025-01-15' AND TIMESTAMP '2025-02-15 23:59:59'"
        )
        .to_pandas()["total"]
        .iloc[0]
    )
    selected = _extensions(context, "site_volume")
    assert selected
    ref = native.publish_unit_day_artifact(store, extensions=selected)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    result = adopted.sitewide("loyalty_purchase_site_volume")
    assert result.site_total_volume == pytest.approx(float(expected_total))  # ty: ignore[unresolved-attribute]


@pytest.mark.slow
def test_mixed_daily_metrics_preserve_calendar_and_cohort_axes(con, tmp_path):
    from increment.semantics.models import RetentionMetric
    from tests.parity_harness.cases import _publish_and_adopt

    payload = _canonical_definitions(load(_DEFINITIONS)).model_dump(mode="json")
    for metric in payload["metrics"]:
        if metric["name"] == "d7_retention":
            metric["threshold_days"] = (7, 14)
    definitions_path = tmp_path / "mixed-axis.yaml"
    definitions_path.write_text(yaml.safe_dump(payload))
    native = Analysis.from_definitions("new_onboarding_v2", definitions_path, con)
    metrics = [
        metric
        for metric in native.metrics
        if metric.name in ("avg_session_duration", "d7_retention")
    ]
    assert {metric.name for metric in metrics} == {"avg_session_duration", "d7_retention"}
    methods = ("run_daily", "run_daily_lift")
    expected = {method: list(getattr(native, method)(metrics=metrics)) for method in methods}
    artifact = _publish_and_adopt(con, native)
    try:
        for method in methods:
            actual = list(getattr(artifact, method)(metrics=metrics))

            def identity(row):
                return row.metric, row.group_id, row.ds

            expected_by_id = {identity(row): row for row in expected[method]}
            assert {identity(row) for row in actual} == expected_by_id.keys()
            assert {row.metric for row in actual} == {metric.name for metric in metrics}
            for row in actual:
                reference = expected_by_id[identity(row)]
                metric = next(metric for metric in metrics if metric.name == row.metric)
                basis = "cohort" if isinstance(metric, RetentionMetric) else "calendar"
                assert row.ds_basis == reference.ds_basis == basis
                estimate = row.value if method == "run_daily" else row.lift
                expected_estimate = reference.value if method == "run_daily" else reference.lift
                if expected_estimate is None:
                    assert estimate is None
                else:
                    assert estimate is not None
                    assert (estimate.value, estimate.lb, estimate.ub) == pytest.approx(
                        (expected_estimate.value, expected_estimate.lb, expected_estimate.ub)
                    )
    finally:
        artifact.close()


def _rewritten_definitions(tmp_path: Path, name: str, edit: Any) -> Path:
    """A copy of the example definitions with the onboarding experiment (and root) edited."""
    source = _definitions(tmp_path)
    payload = yaml.safe_load(source.read_text())
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    edit(payload, experiment)
    target = tmp_path / f"{name}.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False))
    return target


def test_equal_instants_in_different_offsets_share_one_context_for_adoption_and_refresh(
    tmp_path: Path,
) -> None:
    def spelled(start: str, end: str):
        def edit(_payload: dict[str, Any], experiment: dict[str, Any]) -> None:
            experiment["start"] = start
            experiment["end"] = end

        return edit

    utc = _rewritten_definitions(
        tmp_path, "utc", spelled("2025-01-15T00:00:00Z", "2025-02-15T00:00:00Z")
    )
    offset = _rewritten_definitions(
        tmp_path, "offset", spelled("2025-01-14T19:00:00-05:00", "2025-02-14T19:00:00-05:00")
    )
    utc_context = _expected_context(utc)
    offset_context = _expected_context(offset)
    assert utc_context.sha256 == offset_context.sha256
    assert utc_context == offset_context

    _con, native, _context, store = _native(definitions=offset)
    ref = native.publish_unit_day_artifact(store)
    refreshed = native.publish_unit_day_artifact(store, refresh_of=ref)
    adopted = Analysis.from_unit_day_artifact(store, refreshed, expected_context=utc_context)
    try:
        assert lift_rows(adopted.run(metrics=["purchase_rate"]))[0].require_lift().value == (
            pytest.approx(lift_rows(native.run(metrics=["purchase_rate"]))[0].require_lift().value)
        )
    finally:
        adopted.close()
        native.close()


@pytest.mark.parametrize("declared", [False, True], ids=["inherited", "explicit"])
def test_adopted_experiment_keeps_inherited_versus_explicit_day_boundary(
    tmp_path: Path, declared: bool
) -> None:
    def edit(payload: dict[str, Any], experiment: dict[str, Any]) -> None:
        payload["day_boundary"] = "UTC-05:00"
        if declared:
            experiment["day_boundary"] = "UTC+01:00"

    path = _rewritten_definitions(tmp_path, "boundary", edit)
    _con, native, context, store = _native(definitions=path)
    ref = native.publish_unit_day_artifact(store)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:
        expected = "UTC+01:00" if declared else "UTC-05:00"
        assert adopted.experiment.day_boundary == expected
        assert ("day_boundary" in adopted.experiment.model_dump()) is declared
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        assert manifest.day_boundary == expected
    finally:
        adopted.close()
        native.close()


def _secret_sql_definitions(tmp_path: Path, secrets: dict[str, str]) -> Path:
    def edit(payload: dict[str, Any], _experiment: dict[str, Any]) -> None:
        payload["fact_sources"][0]["sql"] += f" /* {secrets['used_fact']} */"
        payload["fact_sources"].append(
            {
                "name": "unused_events",
                "sql": f"SELECT 1 AS user_id, 2 AS ts /* {secrets['unused_fact']} */",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "unused_fact", "column": None}],
            }
        )
        payload["dim_sources"] = [
            {
                "name": "unused_dim",
                "sql": f"SELECT 1 AS user_id, 'x' AS tier /* {secrets['dim']} */",
                "entity": "user_id",
                "properties": [
                    {"name": "tier", "column": "tier", "dtype": "string", "as_of": "static"}
                ],
            }
        ]
        for exposure in payload["exposures"]:
            if "sql" in exposure:
                exposure["sql"] += f" /* {secrets['exposure']} */"

    return _rewritten_definitions(tmp_path, "secrets", edit)


def test_publication_never_persists_or_returns_source_sql(tmp_path: Path) -> None:
    secrets = {
        "used_fact": "SECRET_USED_FACT_2f8a",
        "unused_fact": "SECRET_UNUSED_FACT_91cc",
        "dim": "SECRET_DIM_d04e",
        "exposure": "SECRET_EXPOSURE_6b73",
    }
    path = _secret_sql_definitions(tmp_path, secrets)
    con, native, context, store = _native(definitions=path)
    extensions = _extensions(context, "breakout_dimension", "assignment_counts")
    ref = native.publish_unit_day_artifact(store, extensions=extensions)
    try:
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        index = con.table("ud_manifest_index", database="artifacts")
        stored = [row["manifest_json"] for row in index.to_pyarrow().to_pylist()]
        assert stored
        outputs = [
            context.canonical_json,
            manifest.model_dump_json(),
            ref.model_dump_json(),
            *stored,
        ]
        for output in outputs:
            for secret in secrets.values():
                assert secret not in output
    finally:
        native.close()
