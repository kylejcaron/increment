from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import ibis
import pyarrow as pa
import pytest
import yaml

from examples._seed import seed_event_log
from increment import Scope, from_definitions, from_unit_day_artifact, workflows
from increment._moment_plan import SLOTS
from increment.analysis import Analysis
from increment.artifacts import SourceSnapshotEvidence, publish_unit_day_artifact
from increment.errors import CapabilityError, CodedError
from increment.estimation.armstats import ArmStats
from increment.estimation.diagnostics import SRMResult
from increment.estimators import sitewide
from increment.evidence import (
    CountEvidence,
    CountsRequest,
    MomentRows,
    MomentsRequest,
    evidence_provider,
    request_evidence,
)
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
from increment.workflows import experiment_definition, export_moments, srm
from tests._shared_cache import fixture_cache_key, get_or_build
from tests.analysis_factory import bound_analysis, bound_source, lift_rows, view_readout


def _seed_event_log_input(*, with_pre_period: bool, with_late_returns: bool) -> pa.Table:
    """Load the real producer's immutable event rows for a fresh consumer DB."""
    root = Path(__file__).resolve().parents[1]
    source_key = fixture_cache_key(root, Path(__file__), root / "examples" / "_seed.py")
    key = f"unit-day-event-log-pre-{with_pre_period}-late-{with_late_returns}-{source_key}"

    def build() -> pa.Table:
        producer_con = ibis.duckdb.connect()
        try:
            seed_event_log(
                producer_con,
                with_pre_period=with_pre_period,
                with_late_returns=with_late_returns,
            )
            return producer_con.table("event_log", database="analytics").to_pyarrow()
        finally:
            producer_con.disconnect()

    if os.environ.get("INCREMENT_DISABLE_FIXTURE_CACHE") == "1":
        return build()
    return get_or_build(root / ".cache" / "increment-fixtures", key, build)


def _moments(source, metric, **kwargs) -> MomentRows:
    return cast(
        MomentRows,
        request_evidence(evidence_provider(source), MomentsRequest(metric, **kwargs)),
    )


def _counts(source, kind, *, population="assigned"):
    evidence = cast(
        CountEvidence,
        request_evidence(evidence_provider(source), CountsRequest(kind, population=population)),
    )
    return evidence.counts if kind == "assignment" else evidence.unit_counts or evidence.counts


_PUBLICATION_GOLDEN = (
    Path(__file__).parent / "fixtures" / "unit_day_artifact_publication_golden.json"
)
# The format-2 manifest published after the context bound its window days.
# Saved verbatim as a structural golden; never regenerate.
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
    for relation in body["base"].values():
        relation["digest_format"] = "DIGEST_FORMAT"
        relation["content_sha256"] = "CONTENT_SHA256"
    for extension in body["extensions"]:
        extension["relation"]["digest_format"] = "DIGEST_FORMAT"
        extension["relation"]["content_sha256"] = "CONTENT_SHA256"
    return canonical_json(body)


def _normalized_saved_bytes(payload: str) -> str:
    """A saved golden's manifest bytes, blanked the same way for comparison."""
    body = json.loads(payload)
    body["created_at"] = "CREATED_AT"
    body["manifest_sha256"] = "MANIFEST_SHA256"
    for relation in body["base"].values():
        relation["digest_format"] = "DIGEST_FORMAT"
        relation["content_sha256"] = "CONTENT_SHA256"
    for extension in body["extensions"]:
        extension["relation"]["digest_format"] = "DIGEST_FORMAT"
        extension["relation"]["content_sha256"] = "CONTENT_SHA256"
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
    sitewide_count: bool = False,
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
    sitewide_metric = "purchase_rate"
    if sitewide_count:
        payload["metrics"].append(  # ty: ignore[unresolved-attribute]
            {
                "type": "mean",
                "name": "purchase_count",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "count",
            }
        )
        sitewide_metric = "purchase_count"
    if site_volume_only:
        experiment["plan"] = {"secondaries": [sitewide_metric]}
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


def _definitions_copy(definitions: str | Path) -> tuple[Path, Path]:
    """Copy *definitions* into a fresh directory; return the copy and its directory.

    The definitions builder writes a catalog beside the file it reads, so a shared
    path (``examples/definitions``) must never be bound directly.
    """
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp())
    source_path = Path(definitions)
    copied = root / "definitions"
    if source_path.is_dir():
        shutil.copytree(source_path, copied)
    else:
        copied.mkdir()
        shutil.copy2(source_path, copied / source_path.name)
        copied = copied / source_path.name
    return copied, root


def _native(
    *,
    definitions: str | Path = "examples/definitions",
    with_pre_period: bool = False,
    with_late_returns: bool = False,
    source_snapshot_evidence: SourceSnapshotEvidence | None = None,
) -> tuple[Any, tuple[Any, Any], ArtifactContext, WarehouseArtifactStore]:
    import shutil
    import weakref

    con = ibis.duckdb.connect()
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    con.create_table(
        "event_log",
        _seed_event_log_input(
            with_pre_period=with_pre_period,
            with_late_returns=with_late_returns,
        ),
        database="analytics",
        overwrite=True,
    )
    copied, root = _definitions_copy(definitions)
    native = from_definitions(
        copied,
        con,
        experiment_name="new_onboarding_v2",
        source_snapshot_evidence=source_snapshot_evidence,
    )
    weakref.finalize(native[1], shutil.rmtree, root, True)
    context = _expected_context(definitions)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    return con, native, context, store


def _published(
    *,
    extensions=(),
    definitions: str | Path = "examples/definitions",
    with_pre_period: bool = False,
    with_late_returns: bool = False,
):
    import shutil

    from increment.artifacts import artifact_context

    con, native, context, store = _native(
        definitions=definitions,
        with_pre_period=with_pre_period,
        with_late_returns=with_late_returns,
    )
    copied, root = _definitions_copy(definitions)
    try:
        experiment, resolver = from_definitions(copied, con, experiment_name="new_onboarding_v2")
        try:
            trusted_context = artifact_context(experiment, resolver)
            assert trusted_context == context
            context = trusted_context
            ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
        finally:
            cast(Any, resolver).close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
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


def _assert_breakout_rows_agree(native_rows: Any, artifact_rows: Any) -> None:
    def key(row: Any) -> tuple[Any, ...]:
        return (
            row.analysis_population,
            row.metric,
            row.method,
            row.group_id,
            row.dimension,
            row.dimension_value,
            row.source,
            row.estimand,
        )

    native_by_key = {key(row): row for row in native_rows}
    artifact_by_key = {key(row): row for row in artifact_rows}
    assert native_by_key and set(native_by_key) == set(artifact_by_key)
    for row_key, native_row in native_by_key.items():
        artifact_row = artifact_by_key[row_key]
        assert artifact_row.n_control == native_row.n_control, row_key
        assert artifact_row.n_treat == native_row.n_treat, row_key
        assert artifact_row.excluded == native_row.excluded, row_key
        assert artifact_row.failure_code == native_row.failure_code, row_key
        assert (artifact_row.lift is None) == (native_row.lift is None), row_key
        if native_row.lift is not None:
            assert artifact_row.lift is not None
            assert artifact_row.lift.value == pytest.approx(native_row.lift.value), row_key
            assert artifact_row.lift.lb == pytest.approx(native_row.lift.lb), row_key
            assert artifact_row.lift.ub == pytest.approx(native_row.lift.ub), row_key
    assert {row.analysis_population for row in native_rows} == {"assigned", "triggered"}
    assert {row.analysis_population for row in artifact_rows} == {"assigned", "triggered"}


@pytest.mark.slow
def test_representative_publication_matches_golden_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = json.loads(_PUBLICATION_GOLDEN.read_text())
    expected_manifest = json.loads(expected["manifest_bytes"])
    domains = {
        "avg_session_duration": "continuous",
        "d7_retention": "discrete",
        "purchase_rate": "continuous",
    }
    measure_domains = {
        binding["measure_key"]: domains[binding["metric_name"]]
        for binding in expected_manifest["metric_measures"]
    }
    for measure in expected_manifest["measures"]:
        measure["outcome_domain"] = measure_domains[measure["measure_key"]]
    monkeypatch.setattr(session_module, "uuid", _FixedUUID())
    definitions = _definitions(tmp_path, breakouts=[{"property": "country"}])
    _con, (experiment, resolver), context, store = _native(definitions=definitions)
    extensions = _extensions(context, "breakout_dimension", "assignment_counts")
    ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert all(extension.relation.digest_format == 4 for extension in manifest.extensions)
    assert _normalized_manifest_bytes(manifest) == _normalized_saved_bytes(
        canonical_json(expected_manifest)
    )
    assert [
        {
            "role": role,
            "schema_sha256": getattr(manifest.base, role).schema_sha256,
            "content_sha256": getattr(manifest.base, role).content_sha256,
            "row_count": getattr(manifest.base, role).row_count,
        }
        for role in ("exposures", "measure_stats")
    ] == [
        {
            "role": "exposures",
            "schema_sha256": "69f2a545d85730683e9c035ac635730f513eef0d7221e45c5768089c450b92cb",
            "content_sha256": "1ea3d95b9070cc9c047b545b6325f08240bf308327a2ed9798a6ce1274e65065",
            "row_count": 4000,
        },
        {
            "role": "measure_stats",
            "schema_sha256": "f5f205435874c9454583a7c787bead1d2167f730f5e7e1edb7721a6fcd5ba11e",
            "content_sha256": "2d4bcc824a394b172262eb5039b73e3e1e8d4125d1fb28c2ed8405611994e6ca",
            "row_count": 9840,
        },
    ]
    assert all(
        getattr(manifest.base, role).digest_format == 4 for role in ("exposures", "measure_stats")
    )


def test_legacy_format_two_golden_changes_only_by_the_bound_window_days() -> None:
    """Bound window days update context while saved format-2 relation digests stay stable."""
    before = json.loads(_PUBLICATION_GOLDEN_BEFORE_WINDOW_DAYS.read_text())
    after = json.loads(_PUBLICATION_GOLDEN.read_text())
    old_manifest = json.loads(before["manifest_bytes"])
    new_manifest = json.loads(after["manifest_bytes"])
    old_payload = json.loads(old_manifest["context"]["canonical_json"])
    new_payload = json.loads(new_manifest["context"]["canonical_json"])

    # This format-2 publication predates the explicit-q marker but includes all
    # currently required context fields, so it still validates and rehydrates.
    legacy = UnitDayArtifactManifest.model_validate(new_manifest)
    validate_artifact_context(legacy.context)
    assert "plan_q_explicit" not in new_payload
    from increment.query.source import _artifact_source_context

    _experiment, legacy_source = _artifact_source_context(legacy.context)
    assert not legacy_source.plan.q_explicit
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
    assert all(
        old_manifest["base"][role]["digest_format"] == 2
        and new_manifest["base"][role]["digest_format"] == 2
        for role in ("exposures", "measure_stats")
    )
    for manifest in (old_manifest, new_manifest):
        del manifest["context"], manifest["manifest_sha256"], manifest["created_at"]
    for role in ("exposures", "measure_stats"):
        old_relation = old_manifest["base"][role]
        new_relation = new_manifest["base"][role]
        assert new_relation["schema_sha256"] == old_relation["schema_sha256"]
        assert new_relation["row_count"] == old_relation["row_count"]
        assert new_relation["content_sha256"] == old_relation["content_sha256"]
    assert new_manifest == old_manifest
    before_digests = {item["role"]: item for item in before["relation_digests"]}
    after_digests = {item["role"]: item for item in after["relation_digests"]}
    assert before_digests == after_digests


@pytest.mark.slow
def test_previous_manifest_context_is_refused() -> None:
    saved = json.loads(_PUBLICATION_GOLDEN_FORMAT_1.read_text())
    old_context = ArtifactContext.model_validate(json.loads(saved["manifest_bytes"])["context"])
    assert old_context.context_format == 1
    with pytest.raises(CodedError) as refused:
        validate_artifact_context(old_context)
    assert refused.value.code == "artifact.format.unsupported"
    assert refused.value.context["received"] == 1
    assert refused.value.context["supported"] == 2


@pytest.mark.slow
def test_format_one_local_codec_matches_saved_previous_writer_relation_digests(
    tmp_path: Path,
) -> None:
    """Keep the explicit historical local codec's format-1 meaning pinned."""
    import shutil
    from contextlib import closing

    from increment.query.artifact_digest import digest_relation_stream, stream_canonical_rows
    from increment.query.schemas import ARTIFACT_RELATION_PRIMARY_KEYS, ARTIFACT_RELATION_SCHEMAS

    saved = json.loads(_PUBLICATION_GOLDEN_FORMAT_1.read_text())
    con, native, _context, store, ref = _published(
        definitions=_definitions(tmp_path, breakouts=[{"property": "country"}])
    )
    try:
        observed = []
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            for role in ("exposures", "measure_stats"):
                relation_ref = getattr(manifest.base, role)
                relation = snapshot.verify_relation(relation_ref, expected_role=role)
                rows, row_count, tmpdir = stream_canonical_rows(
                    relation,
                    ARTIFACT_RELATION_SCHEMAS[role],
                    con,
                    primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[role],
                )
                try:
                    with closing(rows):
                        digest = digest_relation_stream(
                            role,
                            ARTIFACT_RELATION_SCHEMAS[role],
                            rows,
                            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[role],
                            row_count=row_count,
                        )
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                observed.append(
                    {
                        "role": role,
                        "schema_sha256": digest.schema_sha256,
                        "content_sha256": digest.content_sha256,
                        "row_count": digest.row_count,
                    }
                )
        assert observed == saved["relation_digests"]
    finally:
        native[1].close()
        con.disconnect()


# A warehouse without a SQL verifier refuses before publication can assemble,
# persist, hash, or transfer any unit-level relation.
@pytest.mark.integration
@pytest.mark.slow
def test_publication_refuses_without_warehouse_digest_verifier_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "_SQL_DIGEST_BACKENDS", frozenset())
    con, (experiment, resolver), _context, store = _native()
    real_to_pyarrow = con.to_pyarrow

    def guarded_to_pyarrow(table, *args, **kwargs):
        names = set(table.schema().names) if hasattr(table, "schema") else set()
        assert not {"unit_id", "first_exposure_ts"} <= names
        assert not {"unit_id", "ds"} <= names
        return real_to_pyarrow(table, *args, **kwargs)

    def unexpected_batch_transfer(*_args, **_kwargs):
        raise AssertionError("publication started relation digest transfer before preflight")

    monkeypatch.setattr(con, "to_pyarrow", guarded_to_pyarrow)
    monkeypatch.setattr(con, "to_pyarrow_batches", unexpected_batch_transfer)
    try:
        with pytest.raises(CodedError) as refused:
            publish_unit_day_artifact(experiment, resolver, store)
        assert refused.value.code == "query.session.warehouse_artifact.verification_unavailable"
        assert refused.value.context["relation"] == "<publication>"
        assert refused.value.context["digest_format"] == 3
        assert "republish from trusted definitions" in str(refused.value.context["route"])
        assert not store.visible_manifests
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.integration
@pytest.mark.slow
def test_publish_open_runs_the_same_arm_operations_and_reopens() -> None:
    from increment.estimation.engine import Method

    _con, (native_experiment, native_resolver), context, store, ref = _published()
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert manifest.context == context
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    unadjusted = Method(name="unadjusted")
    native_rows = lift_rows(
        view_readout(
            native_experiment,
            native_resolver,
            metrics=["purchase_rate"],
            decision_method=unadjusted,
        )
    )
    adopted_rows = lift_rows(
        view_readout(
            adopted_experiment,
            adopted_resolver,
            metrics=["purchase_rate"],
            decision_method=unadjusted,
        )
    )
    assert adopted_rows[0].metric == native_rows[0].metric
    assert adopted_rows[0].require_lift().value == pytest.approx(
        native_rows[0].require_lift().value
    )
    assert len(
        view_readout(
            adopted_experiment,
            adopted_resolver,
            Scope(view="daily", grain="daily", values=True),
            metrics=["purchase_rate"],
        )
    ) == len(
        view_readout(
            native_experiment,
            native_resolver,
            Scope(view="daily", grain="daily", values=True),
            metrics=["purchase_rate"],
        )
    )
    adopted_resolver.close()
    assert lift_rows(
        view_readout(
            adopted_experiment,
            adopted_resolver,
            metrics=["purchase_rate"],
            decision_method=unadjusted,
        )
    )


def _add_treatment_units(con: Any) -> None:
    """Append new treatment-arm units so live data differs from earlier readings."""
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "SELECT * REPLACE (user_id || '_late' AS user_id, session_id || '_late' AS session_id) "
        "FROM analytics.event_log "
        "WHERE experiment_id = 'new_onboarding_v2' AND group_id <> "
        "(SELECT min(group_id) FROM analytics.event_log WHERE experiment_id = 'new_onboarding_v2')"
    )


def _purchase_lift(pair: tuple[Any, Any]) -> float:
    from increment.estimation.engine import Method

    experiment, resolver = pair
    return (
        lift_rows(
            view_readout(
                experiment,
                resolver,
                metrics=["purchase_rate"],
                decision_method=Method(name="unadjusted"),
            )
        )[0]
        .require_lift()
        .value
    )


def _population_readouts(
    experiment: Any, resolver: Any, view: Literal["run", "breakout"] = "run", **options: Any
) -> tuple[Any, list[Any]]:
    """One capture of a triggered experiment's assigned and triggered views.

    Returns the run and the rows of both populations, assigned first.
    """
    from increment import run

    scopes = [
        Scope(view=view, grain="total", population=population)
        for population in ("assigned", "triggered")
    ]
    result = run(experiment.with_views(*scopes), resolver, **options)
    return result, [row for scope in scopes for row in result.readout(scope)]


@pytest.mark.integration
def test_artifact_run_is_effectively_captured_when_live_is_requested() -> None:
    con, native, context, store, ref = _published()
    try:
        experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
        with resolver:
            results = view_readout(
                experiment, resolver, metrics=["purchase_rate"], coherence="live"
            )
            assert results
            assert results.metadata is not None
            assert results.metadata.coherence == "captured"
            assert results.metadata.effective_modes
            assert set(results.metadata.effective_modes.values()) == {"captured"}
    finally:
        native[1].close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.integration
def test_closed_artifact_analysis_reopens_its_pinned_generation_not_a_newer_one() -> None:
    con, (experiment, resolver), context, store, ref = _published()
    adopted = from_unit_day_artifact(store, ref, expected_context=context)
    before = _purchase_lift(adopted)
    adopted[1].close()
    _add_treatment_units(con)
    newer = publish_unit_day_artifact(experiment, resolver, store, refresh_of=ref)
    latest = from_unit_day_artifact(store, newer, expected_context=context)
    with latest[1]:
        assert _purchase_lift(latest) != pytest.approx(before)
    assert _purchase_lift(adopted) == pytest.approx(before)
    adopted[1].close()
    assert _purchase_lift(adopted) == pytest.approx(before)


def test_dashboard_snapshot_reads_the_pinned_source_and_leaves_the_outer_analysis_live() -> None:
    from increment._source_operations import DashboardSnapshotPayload
    from increment.estimation.engine import Method
    from tests.analysis_factory import bound_analysis

    con, native, _context, _store = _native()
    metrics = [m for m in load("examples/definitions").metrics if m.name == "purchase_rate"]
    seen: dict[str, float] = {}

    def facade_lift(isolated: Analysis) -> float:
        rows = isolated.run(metrics=["purchase_rate"], decision_method=Method(name="unadjusted"))
        return lift_rows(rows)[0].require_lift().value

    def operation(isolated: Analysis) -> DashboardSnapshotPayload:
        seen["pinned_before"] = facade_lift(isolated)
        _add_treatment_units(con)
        seen["pinned_after"] = facade_lift(isolated)
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
        bound_analysis(native[1]).dashboard_snapshot(operation, metrics=metrics)
        assert seen["pinned_after"] == pytest.approx(seen["pinned_before"])
        assert _purchase_lift(native) != pytest.approx(seen["pinned_before"])
    finally:
        native[1].close()


@pytest.mark.slow
def test_dashboard_snapshot_refuses_an_artifact_backed_analysis_and_it_stays_usable() -> None:
    from tests.analysis_factory import bound_analysis

    _con, _native_analysis, context, store, ref = _published()
    metrics = [m for m in load("examples/definitions").metrics if m.name == "purchase_rate"]
    adopted = from_unit_day_artifact(store, ref, expected_context=context)
    before = _purchase_lift(adopted)

    def operation(_isolated: Analysis) -> Any:
        raise AssertionError("the handler must not run on an artifact-backed analysis")

    with pytest.raises(CapabilityError) as raised:
        bound_analysis(adopted[1]).dashboard_snapshot(operation, metrics=metrics)
    assert raised.value.code == "evidence.operation_unavailable"
    assert raised.value.context["requested"] == "readout_snapshot"
    adopted[1].close()
    assert _purchase_lift(adopted) == pytest.approx(before)


def test_artifact_source_reports_closed_only_after_close() -> None:
    _con, _native_analysis, context, store, ref = _published()
    source = open_artifact(store, ref, expected_context=context)
    assert source.closed is False
    source.close()
    assert source.closed is True
    source.close()
    assert source.closed is True


@pytest.mark.slow
@pytest.mark.parametrize("closer", ["parent", "triggered"])
def test_closing_either_view_of_a_snapshot_closes_both(tmp_path: Path, closer: str) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    _con, (experiment, resolver), context, store = _native(
        definitions=definitions,
        source_snapshot_evidence=SourceSnapshotEvidence(datetime(2030, 1, 1, tzinfo=UTC)),
    )
    ref = publish_unit_day_artifact(
        experiment,
        resolver,
        store,
        extensions=_extensions(context, "trigger_population", "assignment_counts"),
    )
    parent = open_artifact(store, ref, expected_context=context)
    triggered = cast("Any", parent.triggered_source())
    assert (parent.closed, triggered.closed) == (False, False)
    (parent if closer == "parent" else triggered).close()
    assert (parent.closed, triggered.closed) == (True, True)
    parent.close()
    triggered.close()
    assert (parent.closed, triggered.closed) == (True, True)


@pytest.mark.slow
def test_public_extension_selection_uses_the_immutable_catalog() -> None:
    _con, (native_experiment, native_resolver), context, store = _native()
    request = next(
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "breakout_dimension"
    )
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=[request])
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    assert view_readout(
        adopted_experiment,
        adopted_resolver,
        Scope(view="breakout", grain="total"),
        metrics=["purchase_rate"],
    )
    with pytest.raises(ArtifactContractError):
        publish_unit_day_artifact(
            native_experiment, native_resolver, store, extensions=[request, request]
        )
    with pytest.raises(ArtifactContractError):
        publish_unit_day_artifact(
            native_experiment,
            native_resolver,
            store,
            extensions=cast(
                Any, [{"kind": "breakout_dimension", "property_name": request.property_name}]
            ),
        )


def test_artifact_reader_and_source_openers_share_hydrated_contract() -> None:
    from increment.query.artifact_reader import open_artifact as reader_open
    from increment.query.source import open_artifact as facade_open

    _con, (experiment, resolver), context, store = _native()
    request = _extensions(context, "breakout_dimension")[0]
    ref = publish_unit_day_artifact(experiment, resolver, store, extensions=[request])

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


@pytest.mark.slow
def test_refresh_preserves_artifact_identity_and_rejects_cross_context(
    example_definitions: Path,
) -> None:
    _con, (native_experiment, native_resolver), context, store, first = _published()
    second = publish_unit_day_artifact(native_experiment, native_resolver, store, refresh_of=first)
    assert second.artifact_id == first.artifact_id and second.generation_id != first.generation_id
    other_experiment, other_resolver = from_definitions(
        example_definitions, _con, experiment_name="pricing_tier_test"
    )
    other_context = _expected_context("examples/definitions", "pricing_tier_test")
    with pytest.raises(ArtifactContractError):
        publish_unit_day_artifact(other_experiment, other_resolver, store, refresh_of=first)
    assert len(store.visible_manifests) == 2
    with pytest.raises(ArtifactContractError):
        from_unit_day_artifact(store, first, expected_context=other_context)


@pytest.mark.integration
def test_changed_fact_sql_changes_persisted_spec_identity(tmp_path: Path):
    from increment.store import ParquetResultsStore

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_definitions = _definitions(first_dir)
    second_definitions = _definitions(second_dir)
    payload = yaml.safe_load(second_definitions.read_text())
    source = next(item for item in payload["fact_sources"] if item["name"] == "event_log")
    source["sql"] += "\n/* declaration changed */"
    second_definitions.write_text(yaml.safe_dump(payload, sort_keys=False))
    store = ParquetResultsStore(tmp_path / "runs")
    hashes = []
    for definitions in (first_definitions, second_definitions):
        con, (experiment, resolver), _context, _artifact_store = _native(definitions=definitions)
        try:
            import warnings

            from increment.errors import IncrementWarning

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # `run(..., results_store=)` saves under a run registration rather than
                # the readout spec hash these assertions read; kept on the legacy route.
                rows = bound_analysis(resolver).run(store=store)
            assert any(
                isinstance(item.message, IncrementWarning)
                and item.message.code == "analysis.cuped.covariate_unused"
                for item in caught
            )
            assert rows.metadata is not None
            hashes.append(rows.metadata.analysis_spec_hash)
        finally:
            resolver.close()
            con.disconnect()
    assert hashes[0] != hashes[1]


@pytest.mark.integration
@pytest.mark.slow
def test_fresh_artifact_generation_preserves_content_digest(tmp_path: Path):
    from increment.store import ParquetResultsStore

    result_store = ParquetResultsStore(tmp_path / "runs")
    _con, (native_experiment, native_resolver), context, artifact_store, first = _published()
    second = publish_unit_day_artifact(
        native_experiment, native_resolver, artifact_store, refresh_of=first
    )
    first_experiment, first_resolver = from_unit_day_artifact(
        artifact_store, first, expected_context=context
    )
    second_experiment, second_resolver = from_unit_day_artifact(
        artifact_store, second, expected_context=context
    )
    try:
        first_rows = bound_analysis(first_resolver).run(store=result_store)
        second_rows = bound_analysis(second_resolver).run(store=result_store)
        first_cells = first_rows.as_cells()
        second_cells = second_rows.as_cells()
        first_digests = {cell.identity.source_content_digest for cell in first_cells}
        second_digests = {cell.identity.source_content_digest for cell in second_cells}
        assert first_rows.metadata is not None
        assert second_rows.metadata is not None
        assert first_rows.metadata.analysis_spec_hash == second_rows.metadata.analysis_spec_hash
        assert first_rows.metadata.analysis_spec_hash is not None
        assert second_rows.metadata.analysis_spec_hash is not None
        assert first_rows.metadata.scope.snapshot_id != second_rows.metadata.scope.snapshot_id
        assert (
            result_store.open_run(
                "new_onboarding_v2", first_rows.metadata.analysis_spec_hash
            ).run.run_id
            == second_rows.metadata.run_id
        )
        assert None not in first_digests
        assert None not in second_digests
        assert first_cells
        assert first_cells[0].provenance.artifact_manifest_sha256 == first.manifest_sha256
        assert first_cells[0].provenance.artifact_relation_digests
        assert first_cells[0].provenance.source_content_digest != first.manifest_sha256
        assert first_digests and first_digests == second_digests
    finally:
        first_resolver.close()
        second_resolver.close()


# Guards triggered dispatch from silently returning only the assigned population.
@pytest.mark.integration
@pytest.mark.slow
def test_artifact_run_returns_assigned_and_triggered(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    _con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2030, 1, 1, tzinfo=UTC),
            {"event_log": datetime(2030, 1, 1, tzinfo=UTC)},
        ),
    )
    selected = _extensions(
        context, "trigger_population", "trigger_measure_stats", "assignment_counts"
    )
    assert {request.kind for request in selected} == {
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
    }
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    try:
        first, first_rows = _population_readouts(
            adopted_experiment, adopted_resolver, metrics=["purchase_rate"]
        )
        second, second_rows = _population_readouts(
            adopted_experiment, adopted_resolver, metrics=["purchase_rate"]
        )
        assert first_rows and second_rows
        assert {row.analysis_population for row in first_rows} == {"assigned", "triggered"}
        assert first.metadata is not None and second.metadata is not None

        def order(result):
            assert result.metadata is not None
            return tuple(
                (
                    record.cell.analysis_population,
                    record.cell.metric,
                    record.cell.group_id,
                )
                for record in result.metadata.cells
            )

        first_order = order(first)
        assert first_order == order(second)
        populations = tuple(item[0] for item in first_order)
        assert populations == tuple(sorted(populations))
    finally:
        adopted_resolver.close()
        native_resolver.close()
        _con.disconnect()


# Guards site-volume lookup from confusing a metric's logical name with its physical measure key.
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_site_volume_serves_logical_metric_names(tmp_path: Path) -> None:

    definitions = _definitions(tmp_path, site_volume_only=True, sitewide_count=True)
    _con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
    metric = next(
        item
        for item in workflows.metrics(native_experiment, native_resolver)
        if item.name == "purchase_count"
    )
    assert metric.name != cast(Any, metric).fact
    selected = _extensions(context, "site_volume")
    assert selected
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    result = sitewide(adopted_experiment, adopted_resolver, "purchase_count")
    assert result.site_total_volume > 0  # ty: ignore[unresolved-attribute]


def test_site_volume_coverage_with_no_declared_metrics_omits_extension(
    tmp_path: Path,
) -> None:
    definitions = _definitions(tmp_path, empty_plan=True)
    _con, (experiment, resolver), _context, store = _native(definitions=definitions)
    coverage = {
        "first_ds": "2024-01-01",
        "last_ds": "2024-01-02",
        "freshness": {"loaded_through": "2024-01-02", "declared_complete": True},
    }
    context = compile_unit_day_artifact_context(
        experiment_definition(experiment, resolver).name,
        load(definitions),
        site_volume_coverage=coverage,
    )
    assert not any(
        entry.request.kind == "site_volume"
        for entry in unit_day_artifact_extension_catalog(context)
    )
    assert publish_unit_day_artifact(experiment, resolver, store)


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
    experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
    with pytest.raises(ArtifactDigestError) as raised:
        lift_rows(view_readout(experiment, resolver, metrics=["purchase_rate"]))
    assert raised.value.code == "artifact.digest.type"


@pytest.mark.integration
@pytest.mark.slow
def test_duckdb_cuped_artifact_journey_uses_bounded_integrity_transfers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    definitions = _definitions(tmp_path, cuped_metrics=("purchase_rate",))
    con, (experiment, resolver), context, store = _native(
        definitions=definitions, with_pre_period=True
    )
    forbidden_columns = (
        {"unit_id", "group_id", "first_exposure_ts", "first_exposure_date"},
        {"unit_id", "ds", "n_events", "sum_value", "min_value", "max_value"},
        {"experiment_id", "unit_id", "x"},
        {"experiment_id", "unit_id", "value_is_missing", "dimension_value"},
    )
    real_to_pyarrow = con.to_pyarrow
    real_to_batches = con.to_pyarrow_batches
    digest_batch_sizes = []

    def guarded_to_pyarrow(table, *args, **kwargs):
        if hasattr(table, "schema"):
            names = set(table.schema().names)
            assert not any(columns <= names for columns in forbidden_columns), (
                "unit rows crossed the Python materialization boundary"
            )
        return real_to_pyarrow(table, *args, **kwargs)

    def bounded_digest_batches(table, *args, **kwargs):
        digest_batch_sizes.append(kwargs.get("chunk_size"))
        return real_to_batches(table, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", guarded_to_pyarrow)
    monkeypatch.setattr(con, "to_pyarrow_batches", bounded_digest_batches)
    adopted = None
    try:
        requests = _extensions(
            context, "cuped_preperiod", "assignment_counts", "breakout_dimension"
        )
        ref = publish_unit_day_artifact(experiment, resolver, store, extensions=requests)
        adopted = from_unit_day_artifact(store, ref, expected_context=context)
        rows = lift_rows(view_readout(*adopted, metrics=["purchase_rate"]))
        assert rows
        assert {row.analysis_population for row in rows} == {"assigned"}
        assert any(row.method == "cuped" for row in rows)
        assert view_readout(
            *adopted,
            Scope(view="breakout", grain="total", population="assigned"),
            metrics=["purchase_rate"],
        )
        assert digest_batch_sizes == []

        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        covariate = next(
            extension for extension in manifest.extensions if extension.kind == "cuped_preperiod"
        )
        relation = covariate.relation.relation
        con.raw_sql(f'UPDATE "{relation.schema_name}"."{relation.name}" SET x = x + 1.0')
        adopted[1].close()
        adopted = from_unit_day_artifact(store, ref, expected_context=context)
        with pytest.raises(ArtifactContractError) as raised:
            view_readout(*adopted, metrics=["purchase_rate"])
        assert raised.value.code == "artifact.snapshot.mixed"
    finally:
        if adopted is not None:
            adopted[1].close()
        resolver.close()
        con.disconnect()


@pytest.mark.integration
@pytest.mark.slow
def test_naive_exposure_and_trigger_timestamps_and_declared_covariate_are_warehouse_only(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    definitions = _definitions(tmp_path, trigger="session_start")
    payload = yaml.safe_load(Path(definitions).read_text())
    event_source = next(item for item in payload["fact_sources"] if item["name"] == "event_log")
    event_source["sql"] = event_source["sql"].replace(
        "SELECT *",
        "SELECT *, CASE WHEN user_id LIKE '%0' THEN 1.0 ELSE 2.0 END AS tenure",
        1,
    )
    event_source["properties"].append(
        {"name": "tenure", "column": "tenure", "dtype": "float", "as_of": "static"}
    )
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["n_pre_periods"] = 0
    experiment["end"] = date(2025, 3, 31)
    experiment["plan"]["secondaries"] = [
        {
            "metric": "purchase_rate",
            "decision_method": {
                "name": "scalar",
                "variance_reduction": "cuped",
                "covariates": [
                    {
                        "name": "pre_tenure",
                        "kind": "continuous",
                        "property": "tenure",
                        "source": "event_log",
                        "missing": "zero",
                    }
                ],
            },
        }
    ]
    Path(definitions).write_text(yaml.safe_dump(payload))

    def set_edge_exposure(con) -> str:
        first_page_view = (
            con.table("event_log", database="analytics")
            .filter(ibis._.event == "page_view")
            .order_by("event_at")
            .select("user_id", "event_at")
            .limit(1)
            .execute()
        )
        unit_id = str(first_page_view.iloc[0]["user_id"])
        original_ts = first_page_view.iloc[0]["event_at"].isoformat(sep=" ")
        con.raw_sql(
            "UPDATE analytics.event_log "
            "SET event_at = TIMESTAMP '2025-03-06 23:00:00' "
            f"WHERE event = 'page_view' AND user_id = '{unit_id}' "
            f"AND event_at = TIMESTAMP '{original_ts}'"
        )
        return unit_id

    evidence = SourceSnapshotEvidence(
        datetime(2025, 3, 31, tzinfo=UTC), {"event_log": datetime(2025, 3, 31, tzinfo=UTC)}
    )

    con_utc, (native_utc_experiment, native_utc_resolver), _context_utc, _store_utc = _native(
        definitions=definitions, source_snapshot_evidence=evidence
    )

    con_la, (native_la_experiment, native_la_resolver), context_la, store_la = _native(
        definitions=definitions, source_snapshot_evidence=evidence
    )

    real_to_pyarrow_batches = con_la.to_pyarrow_batches
    real_to_pyarrow = con_la.to_pyarrow
    forbidden_columns = (
        {"unit_id", "group_id", "first_exposure_ts", "first_exposure_date"},
        {"experiment_id", "unit_id", "value"},
        {"experiment_id", "unit_id", "first_trigger_ts"},
        {"experiment_id", "unit_id", "ds", "measure_key", "n_events", "sum_value"},
    )

    def guarded_to_pyarrow(table, *args, **kwargs):
        if hasattr(table, "schema"):
            names = set(table.schema().names)
            assert not any(columns <= names for columns in forbidden_columns), (
                "unit-level artifact rows crossed the Python boundary"
            )
        return real_to_pyarrow(table, *args, **kwargs)

    def guarded_to_pyarrow_batches(table, *args, **kwargs):
        if hasattr(table, "schema"):
            names = set(table.schema().names)
            assert not any(columns <= names for columns in forbidden_columns), (
                "unit-level artifact rows crossed the Python boundary"
            )
        return real_to_pyarrow_batches(table, *args, **kwargs)

    monkeypatch.setattr(con_la, "to_pyarrow_batches", guarded_to_pyarrow_batches)

    monkeypatch.setattr(con_la, "to_pyarrow", guarded_to_pyarrow)
    adopted = None
    try:
        unit_id = set_edge_exposure(con_utc)
        assert set_edge_exposure(con_la) == unit_id
        con_utc.raw_sql("SET TimeZone='UTC'")
        _, expected = _population_readouts(
            native_utc_experiment, native_utc_resolver, metrics=["purchase_rate"]
        )

        con_la.raw_sql("SET TimeZone='America/Los_Angeles'")
        ref = publish_unit_day_artifact(
            native_la_experiment,
            native_la_resolver,
            store_la,
            extensions=_extensions(
                context_la,
                "unit_covariate",
                "trigger_population",
                "trigger_measure_stats",
                "assignment_counts",
            ),
        )
        adopted = from_unit_day_artifact(store_la, ref, expected_context=context_la)
        adopted_experiment, adopted_resolver = adopted
        _, actual = _population_readouts(
            adopted_experiment, adopted_resolver, metrics=["purchase_rate"]
        )
        assert [(row.metric, row.group_id, row.method) for row in actual] == [
            (row.metric, row.group_id, row.method) for row in expected
        ]
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert actual_row.require_lift().value == pytest.approx(
                expected_row.require_lift().value, rel=1e-12
            )
            assert actual_row.require_lift().lb == pytest.approx(
                expected_row.require_lift().lb, rel=1e-12
            )
            assert actual_row.require_lift().ub == pytest.approx(
                expected_row.require_lift().ub, rel=1e-12
            )
        with store_la.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            exposure_relation = snapshot.verify_relation(
                manifest.base.exposures, expected_role="exposures"
            )
            row = (
                exposure_relation.filter(ibis._.unit_id == unit_id)
                .select("first_exposure_ts", "first_exposure_date")
                .execute()
                .to_dict("records")[0]
            )
            trigger_ref = next(
                item for item in manifest.extensions if item.kind == "trigger_population"
            )
            trigger_relation = snapshot.verify_relation(
                trigger_ref.relation, expected_role="trigger_population"
            )
            trigger_row = (
                trigger_relation.filter(ibis._.unit_id == unit_id)
                .select("first_trigger_ts")
                .execute()
                .to_dict("records")[0]
            )
        assert row["first_exposure_ts"] == datetime(2025, 3, 6, 23, tzinfo=UTC)
        assert row["first_exposure_date"].date() == date(2025, 3, 6)
        assert trigger_row["first_trigger_ts"] == datetime(2025, 3, 6, 23, tzinfo=UTC)
    finally:
        if adopted is not None:
            adopted[1].close()
        native_utc_resolver.close()
        native_la_resolver.close()
        con_utc.disconnect()
        con_la.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("property_name", "dtype", "expression"),
    [
        ("integer_property", "int", "CAST(1 AS BIGINT)"),
        ("boolean_property", "bool", "user_id LIKE '%0'"),
    ],
)
def test_publication_accepts_integer_and_boolean_property_covariates(
    tmp_path: Path, property_name: str, dtype: str, expression: str
) -> None:
    definitions = _definitions(tmp_path)
    payload = yaml.safe_load(definitions.read_text())
    source = next(item for item in payload["fact_sources"] if item["name"] == "event_log")
    source["sql"] = source["sql"].replace(
        "SELECT *", f"SELECT *, {expression} AS {property_name}", 1
    )
    source["properties"].append(
        {"name": property_name, "column": property_name, "dtype": dtype, "as_of": "static"}
    )
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["plan"]["secondaries"] = [
        {
            "metric": "purchase_rate",
            "decision_method": {
                "name": "scalar",
                "variance_reduction": "cuped",
                "covariates": [
                    {
                        "name": property_name,
                        "kind": "continuous",
                        "property": property_name,
                        "source": "event_log",
                        "missing": "zero",
                    }
                ],
            },
        }
    ]
    definitions.write_text(yaml.safe_dump(payload, sort_keys=False))
    con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
    try:
        reference = publish_unit_day_artifact(
            native_experiment,
            native_resolver,
            store,
            extensions=_extensions(context, "unit_covariate"),
        )
        assert reference.artifact_id
    finally:
        native_resolver.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.integration
def test_adopted_run_warns_for_published_cuped_covariate(tmp_path: Path) -> None:
    import warnings

    from tests.warning_codes import warning_codes, warning_context

    definitions = _definitions(tmp_path, cuped_metrics=("purchase_rate",))
    con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions, with_pre_period=True
    )
    requests = _extensions(context, "cuped_preperiod")
    assert any(request.metric_name == "purchase_rate" for request in requests)
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=requests)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            view_readout(
                adopted_experiment,
                adopted_resolver,
                metrics=["purchase_rate"],
                sensitivity_methods=[],
            )
        assert warning_codes(caught).count("analysis.cuped.covariate_unused") == 1
        context_values = warning_context(caught, "analysis.cuped.covariate_unused")
        assert context_values["metric_names"] == ("purchase_rate",)
    finally:
        adopted_resolver.close()
        native_resolver.close()
        con.disconnect()


# Guards CUPED dispatch from dropping pre-period covariate moments on breakout and retention-day paths.
@pytest.mark.integration
@pytest.mark.slow
def test_adopted_breakout_cuped_covariates_present(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, cuped_metrics=("purchase_rate",))
    _con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions, with_pre_period=True
    )
    assert experiment_definition(native_experiment, native_resolver).n_pre_periods > 0
    requests = _extensions(context, "breakout_dimension", "cuped_preperiod")
    assert any(request.kind == "cuped_preperiod" for request in requests)
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=requests)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    metric = workflows.metrics(adopted_experiment, adopted_resolver)[0]
    breakout = experiment_definition(adopted_experiment, adopted_resolver).breakouts[0]
    with open_artifact(store, ref, expected_context=context) as source:
        breakout_rows = _moments(
            source, metric, grain="total", by=("country",), include_covariate=True
        )
    native_rows = _moments(
        bound_source(native_resolver).breakout_source(breakout, metrics=[metric]),
        metric,
        by=("country",),
    )
    assert breakout_rows and native_rows
    assert any(abs(float(row["ref_x"])) > 0 for row in breakout_rows)
    assert any(abs(float(row["ref_x"])) > 0 for row in native_rows)
    assert sorted(row["ref_x"] for row in breakout_rows) == pytest.approx(
        sorted(row["ref_x"] for row in native_rows)
    )
    assert view_readout(
        adopted_experiment,
        adopted_resolver,
        Scope(view="breakout", grain="total"),
        metrics=["purchase_rate"],
    )


# Guards bounded-retention daily lift from losing its non-null CUPED covariate moments.
@pytest.mark.integration
@pytest.mark.slow
def test_adopted_retention_cuped_covariates_present(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, cuped_metrics=("d7_retention",))
    _con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions,
        with_pre_period=True,
        with_late_returns=True,
    )
    import warnings

    from tests.warning_codes import warning_codes, warning_context

    assert experiment_definition(native_experiment, native_resolver).n_pre_periods > 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view_readout(
            native_experiment, native_resolver, metrics=["d7_retention"], sensitivity_methods=[]
        )
    assert "analysis.cuped.covariate_unused" in warning_codes(caught)
    assert warning_context(caught, "analysis.cuped.covariate_unused")["metric_names"] == (
        "d7_retention",
    )
    selected = _extensions(context, "cuped_preperiod")
    assert selected
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view_readout(
            adopted_experiment, adopted_resolver, metrics=["d7_retention"], sensitivity_methods=[]
        )
    assert "analysis.cuped.covariate_unused" in warning_codes(caught)
    assert warning_context(caught, "analysis.cuped.covariate_unused")["metric_names"] == (
        "d7_retention",
    )
    retention = next(
        metric
        for metric in workflows.metrics(adopted_experiment, adopted_resolver)
        if metric.name == "d7_retention"
    )
    with open_artifact(store, ref, expected_context=context) as source:
        daily_rows = _moments(source, retention, grain="daily", include_covariate=True)
    native_retention = next(
        metric
        for metric in workflows.metrics(native_experiment, native_resolver)
        if metric.name == "d7_retention"
    )
    native_rows = _moments(
        bound_source(native_resolver).day_source(metrics=[native_retention]),
        native_retention,
        grain="daily",
        include_covariate=True,
    )
    assert daily_rows and native_rows
    assert any(abs(float(row["ref_x"])) > 0 for row in daily_rows)
    assert any(abs(float(row["ref_x"])) > 0 for row in native_rows)
    assert sorted(row["ref_x"] for row in daily_rows) == pytest.approx(
        sorted(row["ref_x"] for row in native_rows)
    )
    assert view_readout(
        adopted_experiment,
        adopted_resolver,
        Scope(view="daily", grain="daily"),
        metrics=["d7_retention"],
    )


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
    con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions, with_pre_period=True
    )
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
        ref = publish_unit_day_artifact(
            native_experiment, native_resolver, store, extensions=requests
        )
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
        native_resolver.close()
        con.disconnect()


# Guards partial breakout extension coverage from refusing by name.
@pytest.mark.slow
def test_incomplete_breakout_coverage_refuses_before_reduction(tmp_path: Path) -> None:
    definitions = _definitions(
        tmp_path,
        breakouts=[{"property": "country"}, {"property": "platform"}],
    )
    _con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
    country_request = next(
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind == "breakout_dimension" and entry.request.property_name == "country"
    )
    ref = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=[country_request]
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    with pytest.raises(ArtifactContractError) as raised:
        view_readout(
            adopted_experiment,
            adopted_resolver,
            Scope(view="breakout", grain="total"),
            metrics=["purchase_rate"],
        )
    assert raised.value.code == "artifact.extension.missing"


# Guards refused artifact opens from leaking the snapshot handle after context validation fails.
@pytest.mark.slow
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
        from_unit_day_artifact(store, ref, expected_context=bad_context)
    assert raised.value.code == "artifact.refresh.context_mismatch"
    assert state == {"entered": 1, "exited": 1, "active": 0}
    native[1].close()


# Guards same-name breakout routing from collapsing dimensions resolved from distinct fact sources.
@pytest.mark.slow
@pytest.mark.integration
def test_multi_source_same_property_breakouts_route_exactly(tmp_path: Path) -> None:
    from increment import run
    from increment.evidence import BreakoutIdentity

    definitions = _definitions(
        tmp_path,
        breakouts=[
            {"property": "country", "source": "event_log"},
            {"property": "country", "source": "event_log_secondary"},
        ],
        second_breakout_source=True,
    )
    requests = _extensions(_expected_context(definitions), "breakout_dimension")
    assert len(requests) == 2
    _con, native, context, store, ref = _published(extensions=requests, definitions=definitions)
    native[1].close()
    experiment, resolver = from_unit_day_artifact(store, ref, expected_context=context)
    country = (BreakoutIdentity("country", None),)
    scopes = (
        Scope(view="breakout", grain="total"),
        *(
            Scope(view=view, grain=view, values=values, segment=country)
            for view in ("daily", "asof")
            for values in (False, True)
        ),
    )
    both = {("event_log", "country"), ("event_log_secondary", "country")}
    with resolver:
        result = run(
            experiment.with_views(*scopes),
            resolver,
            metrics=["purchase_rate"],
        )
        by_view: dict[tuple[str, bool], set[tuple[str | None, str | None]]] = {}
        for cell in result.cells:
            scope = cell.identity.scope
            assert scope is not None
            by_view.setdefault((scope.view, scope.values), set()).add(
                (cell.identity.key.source, cell.identity.key.dimension)
            )
        assert by_view == {(scope.view, scope.values): both for scope in scopes}
        assert len(result.run.metadata.captures) == 1
        secondary = (BreakoutIdentity("country", "event_log_secondary"),)
        qualified = run(
            experiment.with_views(Scope(view="breakout", grain="total", segment=secondary)),
            resolver,
            metrics=["purchase_rate"],
        )
    assert qualified.cells
    assert {
        (cell.identity.key.source, cell.identity.key.dimension) for cell in qualified.cells
    } == {("event_log_secondary", "country")}


@pytest.mark.slow
def test_open_ended_experiment_omits_site_volume(tmp_path: Path) -> None:
    definitions = _definitions(tmp_path, open_ended=True)
    _con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
    assert not any(
        entry.request.kind == "site_volume"
        for entry in unit_day_artifact_extension_catalog(context)
    )
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    assert lift_rows(view_readout(adopted_experiment, adopted_resolver, metrics=["purchase_rate"]))


@pytest.mark.slow
@pytest.mark.integration
def test_triggered_artifact_serves_full_population_extensions(tmp_path: Path) -> None:
    definitions = _definitions(
        tmp_path,
        trigger="session_start",
        breakouts=[{"property": "country"}],
        cuped_metrics=("purchase_rate",),
    )
    _con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions,
        with_pre_period=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2030, 1, 1, tzinfo=UTC), {"event_log": datetime(2030, 1, 1, tzinfo=UTC)}
        ),
    )
    selected = _extensions(
        context,
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
        "breakout_dimension",
        "cuped_preperiod",
    )
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    assert lift_rows(view_readout(adopted_experiment, adopted_resolver, metrics=["purchase_rate"]))
    metric = workflows.metrics(adopted_experiment, adopted_resolver)[0]
    from increment.evidence import MomentsRequest, evidence_provider, request_evidence

    with open_artifact(store, ref, expected_context=context) as source:
        triggered_rows = cast(
            MomentRows,
            request_evidence(
                evidence_provider(source),
                MomentsRequest(
                    metric=metric,
                    grain="total",
                    by=("country",),
                    include_covariate=True,
                    population="triggered",
                ),
            ),
        )
    assert triggered_rows
    assert all(row["country"] is not None for row in triggered_rows)
    assert any(row["ref_x"] is not None for row in triggered_rows)


@pytest.mark.slow
def test_clustered_artifact_run_integrity_uses_randomization_unit_counts(tmp_path: Path) -> None:
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered

    con = ibis.duckdb.connect()
    con.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(
            n_stores_per_arm=25,
            units_per_store={"C": 2, "T": 10},
            trigger_rate=0.5,
        ),
    )
    definitions = tmp_path / "clustered.yml"
    definitions.write_text(
        _defs_yaml_clustered(trigger=None).replace(
            "control_group: C",
            "control_group: C\n    allocation: {C: 0.5, T: 0.5}\n"
            "    allocation_scheme: independent",
        )
    )
    native_experiment, native_resolver = from_definitions(definitions, con, experiment_name="exp")
    context = _expected_context(definitions, "exp")
    store = WarehouseArtifactStore(con, schema_name="cluster_integrity_artifact")
    extensions = _extensions(context, "assignment_counts", "cluster_identity")
    ref = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=extensions
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )

    results = view_readout(adopted_experiment, adopted_resolver)

    with open_artifact(store, ref, expected_context=context) as source:
        assert _counts(source, "assignment") == {"C": 25, "T": 25}

    metadata = results.metadata
    assert metadata is not None
    scope = metadata.scope
    assert scope is not None
    (integrity,) = next(iter(scope.by_source.values())).integrity
    assert integrity.randomization_grain == "cluster"
    assert integrity.observed == {"C": 25, "T": 25}
    assert integrity.expected == {"C": 0.5, "T": 0.5}
    source_payload = results.source
    assert source_payload is not None
    assert any(
        component["kind"] == "assignment_counts" and component["population"] == "assigned"
        for component in source_payload["components"]
    )
    assert integrity.status == "not_rejected"
    adopted_resolver.close()
    cast(Any, native_resolver).close()


@pytest.mark.slow
def test_artifact_assignment_counts_preserve_audit_counts_and_registered_refusal(
    tmp_path: Path,
) -> None:
    from increment.query.artifact_reader import ArtifactMomentSource
    from tests.test_analysis_facade import _insert_null_assignment, _mixed_assignment_connection

    con = _mixed_assignment_connection()
    _insert_null_assignment(con)
    definitions = _definitions(tmp_path)
    definition_payload = yaml.safe_load(definitions.read_text())
    experiment_payload = next(
        item for item in definition_payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment_payload["allocation"] = {"control": 0.5, "treatment": 0.5}
    experiment_payload["allocation_scheme"] = "independent"
    definitions.write_text(yaml.safe_dump(definition_payload, sort_keys=False))
    native_experiment, native_resolver = from_definitions(
        definitions, con, experiment_name="new_onboarding_v2", on_mixed_assignment="exclude"
    )
    loaded = load(definitions)
    experiment = loaded.experiment("new_onboarding_v2")
    assert experiment is not None
    context = artifact_context(loaded, experiment, "exclude")
    store = WarehouseArtifactStore(con, schema_name="assignment_count_audit_artifact")
    reference = publish_unit_day_artifact(
        native_experiment,
        native_resolver,
        store,
        extensions=_extensions(context, "assignment_counts"),
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, reference, expected_context=context
    )
    expected = {
        "control": 99,
        "treatment": 100,
        "(mixed assignment)": 1,
        "(unassigned)": 1,
    }
    try:
        with open_artifact(store, reference, expected_context=context) as source:
            assert _counts(source, "unit") == expected
            assert _counts(source, "assignment") == expected

        native_srm = srm(
            native_experiment, native_resolver, expected={"control": 0.5, "treatment": 0.5}
        )
        artifact_srm = srm(
            adopted_experiment, adopted_resolver, expected={"control": 0.5, "treatment": 0.5}
        )
        assert isinstance(native_srm, SRMResult)
        assert isinstance(artifact_srm, SRMResult)
        assert artifact_srm.observed == native_srm.observed
        assert artifact_srm.mixed_assignment_units == native_srm.mixed_assignment_units
        assert artifact_srm.unassigned_units == native_srm.unassigned_units

        with ArtifactMomentSource.open(store, reference, expected_context=context) as raw_source:
            from increment.evidence import CountsRequest, evidence_provider, request_evidence

            with pytest.raises(ArtifactContractError) as raised:
                request_evidence(
                    evidence_provider(raw_source),
                    CountsRequest(kind="assignment", population="triggered"),
                )
        assert raised.value.code == "artifact.evidence.unavailable"
    finally:
        adopted_resolver.close()
        cast(Any, native_resolver).close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster")
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_clustered_sitewide_supplies_assignment_counts(tmp_path: Path) -> None:
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered

    con = ibis.duckdb.connect()
    con.create_table("cluster_trigger_events", obj=_clustered_trigger_events(n_stores_per_arm=40))
    definitions = tmp_path / "definitions.yaml"
    definitions_text = _defs_yaml_clustered(trigger=None).replace("    window_days: 7\n", "")
    definitions.write_text(
        definitions_text.replace(
            "start: 2024-01-01T00:00:00", "start: 2024-01-01T00:00:00\n    end: 2024-01-09T00:00:00"
        )
    )
    experiment, resolver = from_definitions(definitions, con, experiment_name="exp")
    context = _expected_context(definitions, "exp")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    selected = _extensions(context, "site_volume", "cluster_identity")
    assert selected
    ref = publish_unit_day_artifact(experiment, resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    result = sitewide(adopted_experiment, adopted_resolver, "revenue")
    assert result.delta is not None  # ty: ignore[unresolved-attribute]
    from increment.errors import CapabilityError

    for method, scope in (
        ("run_daily", Scope(view="daily", grain="daily", values=True)),
        ("run_daily_lift", Scope(view="daily", grain="daily")),
        ("run_asof", Scope(view="asof", grain="asof", values=True)),
        ("run_asof_lift", Scope(view="asof", grain="asof")),
    ):
        with pytest.raises(CapabilityError) as raised:
            view_readout(adopted_experiment, adopted_resolver, scope, metrics=["revenue"])
        assert raised.value.code == "facade.analysis.clustered_day_axis"
        assert raised.value.context["method"] == method


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
        native_experiment, native_resolver = from_definitions(
            definitions, con, experiment_name="exp"
        )
        with cast(Any, native_resolver):
            context = _expected_context(definitions, "exp")
            store = WarehouseArtifactStore(con, schema_name="artifacts")
            ref = publish_unit_day_artifact(
                native_experiment,
                native_resolver,
                store,
                extensions=_extensions(context, "cluster_identity", "assignment_counts"),
            )
            adopted_experiment, adopted_resolver = from_unit_day_artifact(
                store, ref, expected_context=context
            )
            with adopted_resolver:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", IncrementRuntimeWarning)
                    (native_row,) = lift_rows(view_readout(native_experiment, native_resolver))
                    (artifact_row,) = lift_rows(view_readout(adopted_experiment, adopted_resolver))
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
                    export_moments(adopted_experiment, adopted_resolver, destination)
                assert raised.value.code == "evidence.cluster_unavailable"
                assert set(raised.value.context) == {"constructor", "requested", "offered", "route"}
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
    evidence = SourceSnapshotEvidence(
        datetime(2030, 1, 1, tzinfo=UTC), {"event_log": datetime(2030, 1, 1, tzinfo=UTC)}
    )
    _con, (native_experiment, native_resolver), context, store = _native(
        definitions=definitions, with_pre_period=True, source_snapshot_evidence=evidence
    )
    requests = _extensions(
        context,
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
        "breakout_dimension",
        "cuped_preperiod",
    )
    assert any(request.kind == "cuped_preperiod" for request in requests)
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=requests)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    try:
        native_source = bound_source(native_resolver)
        from increment.evidence import (
            CountsRequest,
            MomentsRequest,
            UnitRowsRequest,
            evidence_provider,
            request_evidence,
        )

        native_provider = evidence_provider(native_source)
        # Confirm the fixture is genuinely a proper subset, not 100% coverage.
        triggered_count = sum(
            cast(
                CountEvidence,
                request_evidence(
                    native_provider, CountsRequest(kind="unit", population="triggered")
                ),
            ).counts.values()
        )
        enrolled_count = sum(
            cast(
                CountEvidence,
                request_evidence(
                    native_provider, CountsRequest(kind="unit", population="assigned")
                ),
            ).counts.values()
        )
        assert 0 < triggered_count < enrolled_count

        native_metric = workflows.metrics(native_experiment, native_resolver)[0]
        adopted_metric = workflows.metrics(adopted_experiment, adopted_resolver)[0]
        native_rows = cast(
            "list[dict[str, Any]]",
            request_evidence(
                native_provider,
                MomentsRequest(
                    metric=native_metric,
                    grain="total",
                    include_covariate=True,
                    population="triggered",
                ),
            ),
        )

        with open_artifact(store, ref, expected_context=context) as source:
            provider = evidence_provider(source)
            adopted_rows = cast(
                "list[dict[str, Any]]",
                request_evidence(
                    provider,
                    MomentsRequest(
                        metric=adopted_metric,
                        grain="total",
                        include_covariate=True,
                        population="triggered",
                    ),
                ),
            )
            adopted_breakout_rows = cast(
                "list[dict[str, Any]]",
                request_evidence(
                    provider,
                    MomentsRequest(
                        metric=adopted_metric,
                        grain="total",
                        by=("country",),
                        include_covariate=True,
                        population="triggered",
                    ),
                ),
            )
            native_frame = request_evidence(
                native_provider,
                UnitRowsRequest(metric=native_metric, population="triggered"),
            )
            adopted_frame = request_evidence(
                provider,
                UnitRowsRequest(metric=adopted_metric, population="triggered"),
            )
            assert (
                native_rows
                and adopted_rows
                and adopted_breakout_rows
                and {
                    row["unit_id"]: (row["group_id"], row["y"])
                    for row in cast(Any, native_frame).to_pylist()
                }
                == {
                    row["unit_id"]: (row["group_id"], row["y"])
                    for row in cast(Any, adopted_frame).to_pylist()
                }
            )
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
        # With trigger and breakout-dimension evidence, the artifact route is
        # supported; see docs/reference/capabilities-by-entry-point.md,
        # "Triggered auxiliary readouts."
        _assert_breakout_rows_agree(
            _population_readouts(
                native_experiment, native_resolver, "breakout", metrics=["purchase_rate"]
            )[1],
            _population_readouts(
                adopted_experiment, adopted_resolver, "breakout", metrics=["purchase_rate"]
            )[1],
        )

        # The public whole-window readout (decision + CUPED sensitivity,
        # both assigned- and triggered-population rows) must match exactly.
        _, native_lift = _population_readouts(
            native_experiment, native_resolver, metrics=["purchase_rate"]
        )
        _, adopted_lift = _population_readouts(
            adopted_experiment, adopted_resolver, metrics=["purchase_rate"]
        )

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
        adopted_resolver.close()
        native_resolver.close()


def _triggered_scalar_covariate_case(
    tmp_path: Path, *, binding: str, scenario: str
) -> tuple[Any, Any, Any, str, str]:
    """Publish a scalar covariate case whose assigned and triggered populations differ."""

    definitions_path = _triggered_subset_definitions(tmp_path)
    payload = yaml.safe_load(definitions_path.read_text(encoding="utf-8"))
    source = next(source for source in payload["fact_sources"] if source["name"] == "event_log")
    experiment = next(
        experiment
        for experiment in payload["experiments"]
        if experiment["name"] == "new_onboarding_v2"
    )
    metric = "purchase_rate"
    covariate_name = "tenure" if binding == "property" else "preperiod_page_views"
    missing = {
        "outside": "refuse",
        "impute": "impute",
        "impute_outside": "impute",
        "all_null": "impute",
    }[scenario]

    if binding == "property":
        if scenario in {"outside", "impute_outside"}:
            value_sql = (
                "CASE WHEN country_code = 'US' THEN "
                "CAST(abs(hash(user_id)) % 100 AS DOUBLE) ELSE NULL END"
            )
        elif scenario == "impute":
            seed = _seed_event_log_input(with_pre_period=True, with_late_returns=False)
            missing_user = next(
                row["user_id"] for row in seed.to_pylist() if row["country_code"] == "US"
            )
            value_sql = (
                f"CASE WHEN user_id = '{missing_user}' THEN NULL ELSE "
                "CAST(abs(hash(user_id)) % 100 AS DOUBLE) END"
            )
        else:
            value_sql = (
                "CASE WHEN country_code = 'US' THEN NULL ELSE "
                "CAST(abs(hash(user_id)) % 100 AS DOUBLE) END"
            )
        source["sql"] = source["sql"].replace("SELECT *", f"SELECT *, {value_sql} AS tenure", 1)
        source.setdefault("properties", []).append(
            {"name": "tenure", "column": "tenure", "dtype": "float", "as_of": "static"}
        )
        covariate = {"name": covariate_name, "kind": "continuous", "missing": missing}
        experiment["plan"]["secondaries"] = [
            item
            for item in experiment["plan"]["secondaries"]
            if not (isinstance(item, dict) and item.get("metric") == metric)
        ] + [
            {
                "metric": metric,
                "decision_method": {
                    "name": "cuped",
                    "variance_reduction": "cuped",
                    "covariates": [covariate],
                },
            }
        ]
    else:
        filters = (
            [{"property": "country", "op": "equals", "values": ["US"]}]
            if scenario in {"outside", "impute_outside"}
            else (
                [{"property": "country", "op": "not_equals", "values": ["US"]}]
                if scenario == "all_null"
                else []
            )
        )
        measure_fact = "purchase" if scenario in {"outside", "impute_outside"} else "page_view"
        covariate = {
            "name": covariate_name,
            "kind": "continuous",
            "missing": missing,
            "measure": {"fact": measure_fact, "aggregation": "count", "filters": filters},
        }
        experiment["plan"]["secondaries"] = [
            item
            for item in experiment["plan"]["secondaries"]
            if not (isinstance(item, dict) and item.get("metric") == metric)
        ] + [
            {
                "metric": metric,
                "decision_method": {
                    "name": "cuped",
                    "variance_reduction": "cuped",
                    "covariates": [covariate],
                },
            }
        ]

    definitions_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    con, native, context, store = _native(
        definitions=definitions_path,
        with_pre_period=True,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=datetime(2030, 1, 1, tzinfo=UTC),
            complete_through_by_feed={"event_log": datetime(2030, 1, 1, tzinfo=UTC)},
        ),
    )
    if binding == "measure" and scenario == "outside":
        con.raw_sql(
            """
            INSERT INTO analytics.event_log
            SELECT
                exposure.event_at - INTERVAL '1 day',
                exposure.user_id,
                exposure.session_id || '-m6-' || CAST(samples.i AS VARCHAR),
                'purchase',
                CAST(1.0 AS DOUBLE),
                CAST(NULL AS DOUBLE),
                exposure.country_code,
                exposure.device_type,
                exposure.plan,
                CAST(NULL AS VARCHAR),
                CAST(NULL AS VARCHAR)
            FROM analytics.event_log AS exposure
            CROSS JOIN range(3) AS samples(i)
            WHERE exposure.experiment_id = 'new_onboarding_v2'
              AND exposure.event = 'page_view'
              AND exposure.country_code = 'US'
              AND samples.i < abs(hash(exposure.user_id) % 3) + 1
            """
        )
    extensions = _extensions(
        context,
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
        "unit_covariate",
        "cuped_measure_covariate",
    )
    native_experiment, native_resolver = native
    published = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=extensions
    )
    adopted = from_unit_day_artifact(store, published, expected_context=context)
    return con, native, adopted, metric, covariate_name


@pytest.mark.slow
@pytest.mark.parametrize("binding", ["property", "measure"])
def test_triggered_scalar_refuse_ignores_missing_outside_population(
    tmp_path: Path, binding: str
) -> None:
    con, native, (experiment, resolver), metric_name, _covariate_name = (
        _triggered_scalar_covariate_case(tmp_path, binding=binding, scenario="outside")
    )
    try:
        rows = _population_readouts(experiment, resolver, metrics=[metric_name])[1]
        assert rows
        assert {row.analysis_population for row in rows} == {"assigned", "triggered"}
        triggered_cuped = next(
            row for row in rows if row.analysis_population == "triggered" and row.method == "cuped"
        )
        assert triggered_cuped.lift is not None
    finally:
        resolver.close()
        native[1].close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("binding", ["property", "measure"])
def test_triggered_scalar_imputation_ignores_missing_outside_population(
    tmp_path: Path, binding: str
) -> None:
    con, native, (experiment, resolver), metric_name, _covariate_name = (
        _triggered_scalar_covariate_case(tmp_path, binding=binding, scenario="impute_outside")
    )
    try:
        rows = _population_readouts(experiment, resolver, metrics=[metric_name])[1]
        triggered_cuped = next(
            row for row in rows if row.analysis_population == "triggered" and row.method == "cuped"
        )
        assert triggered_cuped.lift is not None
    finally:
        resolver.close()
        native[1].close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("binding", ["property", "measure"])
def test_triggered_scalar_imputation_uses_triggered_population_mean(
    tmp_path: Path, binding: str
) -> None:
    (
        con,
        (native_experiment, native_resolver),
        (adopted_experiment, adopted_resolver),
        metric_name,
        _covariate_name,
    ) = _triggered_scalar_covariate_case(tmp_path, binding=binding, scenario="impute")
    try:
        expected = _population_readouts(native_experiment, native_resolver, metrics=[metric_name])[
            1
        ]
        actual = _population_readouts(adopted_experiment, adopted_resolver, metrics=[metric_name])[
            1
        ]
        expected_triggered = next(row for row in expected if row.analysis_population == "triggered")
        actual_triggered = next(row for row in actual if row.analysis_population == "triggered")
        expected_lift = expected_triggered.require_lift()
        actual_lift = actual_triggered.require_lift()
        assert actual_lift.value == pytest.approx(expected_lift.value, rel=1e-9)
        assert actual_lift.log_se == pytest.approx(expected_lift.log_se, rel=1e-9)
    finally:
        adopted_resolver.close()
        native_resolver.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("binding", ["property", "measure"])
def test_triggered_scalar_all_null_within_population_refuses(tmp_path: Path, binding: str) -> None:
    con, native, (experiment, resolver), metric_name, _covariate_name = (
        _triggered_scalar_covariate_case(tmp_path, binding=binding, scenario="all_null")
    )
    try:
        rows = _population_readouts(experiment, resolver, metrics=[metric_name])[1]
        triggered_cuped = next(
            row for row in rows if row.analysis_population == "triggered" and row.method == "cuped"
        )
        assert triggered_cuped.failure_code == "frame.validation.metric_covariate_impute_all_null"
    finally:
        resolver.close()
        native[1].close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.integration
def test_artifact_run_breakout_supports_trigger_with_required_evidence(tmp_path: Path) -> None:
    """Trigger breakout parity needs trigger and breakout-dimension evidence.

    The support condition is recorded in
    ``docs/reference/capabilities-by-entry-point.md`` under "Triggered
    auxiliary readouts".
    """
    con = ibis.duckdb.connect()
    seed_event_log(con)
    definitions = _triggered_definitions(tmp_path)
    native_experiment, native_resolver = from_definitions(
        definitions,
        con,
        experiment_name="new_onboarding_v2",
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime(2030, 1, 1, tzinfo=UTC), {"event_log": datetime(2030, 1, 1, tzinfo=UTC)}
        ),
    )
    context = _expected_context(definitions)
    store = WarehouseArtifactStore(con, schema_name="artifacts")

    required_kinds = {
        "trigger_population",
        "trigger_measure_stats",
        "assignment_counts",
        "breakout_dimension",
    }
    extensions = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in required_kinds
    ]
    ref = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=extensions
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    try:
        _assert_breakout_rows_agree(
            _population_readouts(
                native_experiment, native_resolver, "breakout", metrics=["purchase_rate"]
            )[1],
            _population_readouts(
                adopted_experiment, adopted_resolver, "breakout", metrics=["purchase_rate"]
            )[1],
        )
    finally:
        adopted_resolver.close()
        cast(Any, native_resolver).close()
        con.disconnect()


def _forbid_connection_queries(connection, monkeypatch):
    for method in ("execute", "to_pyarrow"):
        if getattr(connection, method, None) is None:
            continue

        def forbidden_query(*args, _method=method, **kwargs):
            pytest.fail(f"source query {_method!r} occurred before the required evidence refusal")

        monkeypatch.setattr(connection, method, forbidden_query)


def test_trigger_artifact_publication_requires_explicit_cutoff_before_pinning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from increment.errors import CapabilityError

    con, (experiment, resolver), context, store = _native(
        definitions=_triggered_definitions(tmp_path)
    )
    _forbid_connection_queries(con, monkeypatch)
    try:
        with pytest.raises(CapabilityError) as raised:
            publish_unit_day_artifact(
                experiment, resolver, store, extensions=_extensions(context, "trigger_population")
            )
        assert raised.value.code == "source.native.trigger_evidence_required"
    finally:
        resolver.close()


def test_triggered_assignment_count_publication_requires_snapshot_evidence_before_pinning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definitions = _triggered_definitions(tmp_path)
    con, (experiment, resolver), context, store = _native(definitions=definitions)
    request = _extensions(context, "assignment_counts")[0]
    _forbid_connection_queries(con, monkeypatch)
    try:
        with pytest.raises(CapabilityError) as raised:
            publish_unit_day_artifact(experiment, resolver, store, extensions=[request])
        assert raised.value.code == "source.native.trigger_evidence_required"
    finally:
        resolver.close()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_site_volume_extension_counts_non_enrolled_units_in_window(tmp_path: Path) -> None:
    """Regression: the enrolled-unit semi-join must not shrink site_volume's
    whole-population reading to enrolled units only. The site branch counts every
    unit's events in [start, end], independent of enrollment (matching
    builders.py's site_volume() -- "No exposure join: a unit that never appears in
    any exposure table still counts", tests/query/test_builders.py:6008). Seeds one
    purchase event for a user who is never exposed at all, inside the experiment
    window, and asserts the whole-site reading includes it."""
    definitions = _definitions(tmp_path, site_volume_only=True, sitewide_count=True)
    con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
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
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    result = sitewide(adopted_experiment, adopted_resolver, "purchase_count")
    assert result.site_total_volume == pytest.approx(float(expected_events))  # ty: ignore[unresolved-attribute]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
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
        sitewide_count=True,
        observation_end="2025-03-01",
        day_boundary=day_boundary,
    )
    con, (native_experiment, native_resolver), context, store = _native(definitions=definitions)
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
    ref = publish_unit_day_artifact(
        native_experiment, native_resolver, store, extensions=(request,)
    )
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
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
    native_impact = sitewide(native_experiment, native_resolver, "purchase_count")
    adopted_impact = sitewide(adopted_experiment, adopted_resolver, "purchase_count")
    native_total = native_impact.site_total_volume  # ty: ignore[unresolved-attribute]
    adopted_total = adopted_impact.site_total_volume  # ty: ignore[unresolved-attribute]
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


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
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
            "filters": [{"property": "loyalty_tier", "op": "equals", "values": ["gold"]}],
        }
    )
    experiment["breakouts"] = []
    experiment["plan"] = {"secondaries": ["loyalty_purchase_site_volume"]}
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    con, (native_experiment, native_resolver), context, store = _native(definitions=path)
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
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store, extensions=selected)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    result = sitewide(adopted_experiment, adopted_resolver, "loyalty_purchase_site_volume")
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
    native = from_definitions(definitions_path, con, experiment_name="new_onboarding_v2")
    metrics = [
        metric
        for metric in workflows.metrics(*native)
        if metric.name in ("avg_session_duration", "d7_retention")
    ]
    assert {metric.name for metric in metrics} == {"avg_session_duration", "d7_retention"}
    scopes = {
        "run_daily": Scope(view="daily", grain="daily", values=True),
        "run_daily_lift": Scope(view="daily", grain="daily"),
    }
    expected = {
        method: list(view_readout(*native, scope, metrics=metrics))
        for method, scope in scopes.items()
    }
    artifact = _publish_and_adopt(con, native)
    try:
        for method, scope in scopes.items():
            actual = list(view_readout(*artifact, scope, metrics=metrics))

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
        artifact[1].close()


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


@pytest.mark.slow
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

    from increment.estimation.engine import Method

    unadjusted = Method(name="unadjusted")
    _con, (native_experiment, native_resolver), _context, store = _native(definitions=offset)
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store)
    refreshed = publish_unit_day_artifact(native_experiment, native_resolver, store, refresh_of=ref)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, refreshed, expected_context=utc_context
    )
    try:
        assert lift_rows(
            view_readout(
                adopted_experiment,
                adopted_resolver,
                metrics=["purchase_rate"],
                decision_method=unadjusted,
            )
        )[0].require_lift().value == pytest.approx(
            lift_rows(
                view_readout(
                    native_experiment,
                    native_resolver,
                    metrics=["purchase_rate"],
                    decision_method=unadjusted,
                )
            )[0]
            .require_lift()
            .value
        )
    finally:
        adopted_resolver.close()
        native_resolver.close()


@pytest.mark.slow
@pytest.mark.parametrize("declared", [False, True], ids=["inherited", "explicit"])
def test_adopted_experiment_keeps_inherited_versus_explicit_day_boundary(
    tmp_path: Path, declared: bool
) -> None:
    def edit(payload: dict[str, Any], experiment: dict[str, Any]) -> None:
        payload["day_boundary"] = "UTC-05:00"
        if declared:
            experiment["day_boundary"] = "UTC+01:00"

    path = _rewritten_definitions(tmp_path, "boundary", edit)
    _con, (native_experiment, native_resolver), context, store = _native(definitions=path)
    ref = publish_unit_day_artifact(native_experiment, native_resolver, store)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )
    try:
        expected = "UTC+01:00" if declared else "UTC-05:00"
        assert experiment_definition(adopted_experiment, adopted_resolver).day_boundary == expected
        assert (
            "day_boundary"
            in experiment_definition(adopted_experiment, adopted_resolver).model_dump()
        ) is declared
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        assert manifest.day_boundary == expected
    finally:
        adopted_resolver.close()
        native_resolver.close()


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


@pytest.mark.slow
def test_publication_never_persists_or_returns_source_sql(tmp_path: Path) -> None:
    secrets = {
        "used_fact": "SECRET_USED_FACT_2f8a",
        "unused_fact": "SECRET_UNUSED_FACT_91cc",
        "dim": "SECRET_DIM_d04e",
        "exposure": "SECRET_EXPOSURE_6b73",
    }
    path = _secret_sql_definitions(tmp_path, secrets)
    con, (experiment, resolver), context, store = _native(definitions=path)
    extensions = _extensions(context, "breakout_dimension", "assignment_counts")
    ref = publish_unit_day_artifact(experiment, resolver, store, extensions=extensions)
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
        resolver.close()


pytestmark = pytest.mark.integration
