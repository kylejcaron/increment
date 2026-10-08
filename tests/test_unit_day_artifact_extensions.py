from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from increment.query._native_triggered import TriggeredPopulationSource
from increment.query.artifact_contract import (
    ArtifactContractError,
    ArtifactSnapshot,
    unit_day_artifact_extension_catalog,
)
from increment.query.artifact_digest import (
    canonical_json,
    content_sha256,
    extension_definition_sha256,
    extension_source_provenance_sha256,
    schema_sha256,
)
from increment.query.artifact_extensions import (
    encode_contrast_extension,
    encode_extension,
    encode_observational_extension,
    read_contrast_extension,
    read_extension,
    read_observational_extension,
    read_site_volume_extension,
    validate_contrast_extension,
    validate_observational_extension,
)
from increment.query.artifact_publish import artifact_context
from increment.query.native_contract import SitewideEvidence
from increment.query.schemas import ARTIFACT_RELATION_PRIMARY_KEYS, ARTIFACT_RELATION_SCHEMAS
from increment.semantics.artifact import (
    ArtifactContext,
    ArtifactRelationRef,
    AssignmentCountsExtension,
    SiteVolumeExtension,
    TriggerMeasureStatsExtension,
    TriggerPopulationExtension,
)
from increment.sources import BreakoutMomentsSource


def _expect_code(code, operation):
    with pytest.raises(ArtifactContractError) as caught:
        operation()
    assert caught.value.code == code


AID = UUID("00000000-0000-0000-0000-000000000001")
GID = UUID("00000000-0000-0000-0000-000000000002")


def _recipe(source):
    """The hash-only recipe object a context stores for a source recipe."""
    return {
        "source_recipe_format": 2,
        "recipe_sha256": hashlib.sha256(canonical_json(source).encode()).hexdigest(),
    }


def _context(request, definition, source):
    entry = {
        "request": request,
        "canonical_definition_json": canonical_json(definition),
        "canonical_source_recipe_json": canonical_json(source),
        "definition_sha256": extension_definition_sha256(
            request["kind"], canonical_json(definition)
        ),
        "source_provenance_sha256": extension_source_provenance_sha256(
            request["kind"], canonical_json(source)
        ),
    }
    return _context_catalog([entry])


def _context_catalog(entries):
    from increment.query.artifact_digest import context_sha256

    ordered = sorted(entries, key=lambda item: canonical_json(item["request"]).encode())
    payload = {
        "context_format": 2,
        "extension_catalog": ordered,
        "window_days": {"start": "2025-01-01", "end": None, "observation_horizon": None},
    }
    raw = canonical_json(payload)
    return ArtifactContext(canonical_json=raw, sha256=context_sha256({"canonical_json": raw}))


class _SiteSource:
    operations = frozenset({"sitewide_evidence"})

    def sitewide_evidence(self, metric, *, include_ratio=False):
        raise AssertionError("rows are supplied directly")


class _Publication:
    artifact_id = AID
    generation_id = GID

    def __init__(self):
        self.calls = []

    def write_relation(self, role, table):
        rows = table.to_pyarrow().to_pylist()
        self.calls.append((role, table))
        return ArtifactRelationRef(
            artifact_id=AID,
            generation_id=GID,
            role=role,
            relation={"name": f"{role}_r"},
            schema_sha256=schema_sha256(role, ARTIFACT_RELATION_SCHEMAS[role]),
            content_sha256=content_sha256(
                role,
                ARTIFACT_RELATION_SCHEMAS[role],
                rows,
                primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[role],
            ),
            row_count=len(rows),
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[role],
        )

    def publish_manifest(self, manifest):
        raise AssertionError("extension producer must not publish the manifest")


class _AllSource:
    operations = frozenset(
        {
            "breakout_source",
            "day_source",
            "triggered_counts",
            "triggered_source",
            "sitewide_evidence",
        }
    )

    def breakout_source(self, breakout, *, metrics=()):
        return self

    def day_source(self, *, metrics=()):
        return self

    def triggered_counts(self):
        return ("unit", {}, {})

    def assignment_counts(self, *, population="assigned"):
        return {"control": 1}

    def triggered_source(self):
        return self

    def sitewide_evidence(self, metric, *, include_ratio=False):
        return self


class _RowsOperationSource(_AllSource):
    def __init__(self, kind, rows):
        self.kind = kind
        self.rows = rows
        self.called = []

    def breakout_source(self, breakout, *, metrics=()):
        self.called.append("breakout_source")
        return self.rows

    def day_source(self, *, metrics=()):
        self.called.append("day_source")
        return self.rows

    def triggered_source(self):
        self.called.append("triggered_source")
        return self.rows

    def triggered_counts(self):
        self.called.append("triggered_counts")
        return ("unit", {"control": 1}, {"control": 1})

    def assignment_counts(self, *, population="assigned"):
        self.called.append("assignment_counts")
        return {"control": 1}

    def sitewide_evidence(self, metric, *, include_ratio=False):
        self.called.append("sitewide_evidence")
        if self.kind == "site_volume":
            return SitewideEvidence(
                metric=metric,
                site_total=2.0,
                arm_stats=(),
                control_group="control",
                cluster=None,
            )
        return self.rows


class _MomentsShapedDayResult:
    """Day-source result carrying moments rather than unit-grain rows."""

    def moments(self, **_kwargs):
        raise AssertionError("moments must not be read")


class _ConcreteResultSource(_RowsOperationSource):
    def breakout_source(self, breakout, *, metrics=()):
        return object.__new__(BreakoutMomentsSource)

    def day_source(self, *, metrics=()):
        return _MomentsShapedDayResult()

    def triggered_source(self):
        self.called.append("triggered_source")
        return object.__new__(TriggeredPopulationSource)


class _Snapshot:
    artifact_id = AID
    generation_id = GID

    def __init__(self, relation, rows):
        self.relation = relation
        self.rows = rows
        self.calls = 0

    def read_manifest(self, locator, *, expected_sha256):
        raise AssertionError("manifest lookup is not used by extension tests")

    def verify_relation(self, ref, *, expected_role):
        assert ref == self.relation
        self.calls += 1
        return self.rows

    def execute(self, expression):
        return expression

    def batches(self, expression):
        import pyarrow as pa

        return pa.Table.from_pylist(self.execute(expression)).to_batches(max_chunksize=1024)


def _valid_request():
    return {"kind": "site_volume", "metric_names": ["revenue"]}


def _empty_site_volume_extension() -> SiteVolumeExtension:
    relation = ArtifactRelationRef(
        artifact_id=AID,
        generation_id=GID,
        role="site_volume",
        relation={"name": "site_volume_r"},
        schema_sha256=schema_sha256("site_volume", ARTIFACT_RELATION_SCHEMAS["site_volume"]),
        content_sha256=content_sha256(
            "site_volume",
            ARTIFACT_RELATION_SCHEMAS["site_volume"],
            [],
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
        ),
        row_count=0,
        primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
    )
    return SiteVolumeExtension(
        relation=relation,
        definition_sha256="0" * 64,
        source_provenance_sha256="0" * 64,
        measure_keys=("revenue",),
        first_ds=date(2024, 1, 1),
        last_ds=date(2024, 1, 2),
        freshness={"loaded_through": date(2024, 1, 2), "declared_complete": True},
    )


def _trigger_measure_extension(rows):
    schema = ARTIFACT_RELATION_SCHEMAS["trigger_measure_stats"]
    primary_key = ARTIFACT_RELATION_PRIMARY_KEYS["trigger_measure_stats"]
    relation = ArtifactRelationRef(
        artifact_id=AID,
        generation_id=GID,
        role="trigger_measure_stats",
        relation={"name": "trigger_measure_stats_r"},
        schema_sha256=schema_sha256("trigger_measure_stats", schema),
        content_sha256=content_sha256(
            "trigger_measure_stats", schema, rows, primary_key=primary_key
        ),
        row_count=len(rows),
        primary_key=primary_key,
    )
    return TriggerMeasureStatsExtension(
        relation=relation,
        definition_sha256="0" * 64,
        source_provenance_sha256="1" * 64,
        trigger_name="checkout",
        metric_names=("revenue",),
        trigger_population_content_sha256="2" * 64,
        observation_cutoff_ts=datetime(2025, 1, 20, tzinfo=UTC),
        complete_through_ts=datetime(2025, 1, 15, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("row", "experiment_id"),
    [
        (
            {
                "experiment_id": "exp",
                "unit_id": "u1",
                "ds": date(2025, 1, 2),
                "measure_key": "revenue",
                "n_events": 1,
                "sum_value": 4.0,
                "min_value": 4.0,
                "max_value": 4.0,
            },
            "exp",
        ),
        (
            {
                "experiment_id": "exp",
                "unit_id": "u1",
                "ds": date(2025, 1, 2),
                "measure_key": "revenue",
                "n_events": -1,
                "sum_value": 4.0,
                "min_value": 4.0,
                "max_value": 4.0,
            },
            "exp",
        ),
        (
            {
                "experiment_id": "other",
                "unit_id": "u1",
                "ds": date(2025, 1, 2),
                "measure_key": "revenue",
                "n_events": 1,
                "sum_value": 4.0,
                "min_value": 4.0,
                "max_value": 4.0,
            },
            "exp",
        ),
    ],
    ids=["valid", "negative-event-count", "wrong-experiment"],
)
def test_trigger_measure_extension_rejects_invalid_counts_and_experiment_domain(row, experiment_id):
    rows = [row]
    extension = _trigger_measure_extension(rows)
    request = {
        "kind": "trigger_measure_stats",
        "trigger_name": "checkout",
        "metric_names": ("revenue",),
    }
    if row["n_events"] == 1 and row["experiment_id"] == experiment_id:
        assert (
            read_extension(
                _Snapshot(extension.relation, rows),
                extension,
                request=request,
                experiment_id=experiment_id,
            )
            == rows
        )
        return
    with pytest.raises(ArtifactContractError):
        read_extension(
            _Snapshot(extension.relation, rows),
            extension,
            request=request,
            experiment_id=experiment_id,
        )


def test_trigger_measure_extension_rejects_duplicate_primary_keys():
    row = {
        "experiment_id": "exp",
        "unit_id": "u1",
        "ds": date(2025, 1, 2),
        "measure_key": "revenue",
        "n_events": 1,
        "sum_value": 4.0,
        "min_value": 4.0,
        "max_value": 4.0,
    }
    rows = [row, row.copy()]
    extension = _trigger_measure_extension([row])
    with pytest.raises(ArtifactContractError):
        read_extension(
            _Snapshot(extension.relation, rows),
            extension,
            request={
                "kind": "trigger_measure_stats",
                "trigger_name": "checkout",
                "metric_names": ("revenue",),
            },
            experiment_id="exp",
        )


def test_read_rejects_mismatched_request_kind_as_coded_identity_refusal():
    extension = _empty_site_volume_extension()
    snapshot = _Snapshot(extension.relation, [])
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            snapshot,
            extension,
            request={"kind": "breakout_dimension", "property_name": "country"},
            experiment_id="exp",
        ),
    )
    assert snapshot.calls == 0


def test_read_rejects_unregistered_stored_kind_as_coded_refusal():
    extension = _empty_site_volume_extension().model_copy(update={"kind": "stored_future"})
    snapshot = _Snapshot(extension.relation, [])
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            snapshot,
            extension,
            request=_valid_request(),
            experiment_id="exp",
        ),
    )
    assert snapshot.calls == 0


def test_encode_writes_through_explicit_publication_handle_and_uses_exact_catalog_entry():
    request = _valid_request()
    definition = {
        "kind": "site_volume",
        "metric_names": ["revenue"],
        "measure_keys": ["revenue"],
        "first_ds": "2024-01-01",
        "last_ds": "2024-01-02",
        "freshness": {"loaded_through": "2024-01-02", "declared_complete": True},
    }
    source_recipe = _recipe({"source": "facts", "measure_keys": ["revenue"]})
    context = _context(request, definition, source_recipe)
    publication = _Publication()
    rows = [
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 1),
            "measure_key": "revenue",
            "n_events": 1,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 2),
            "measure_key": "revenue",
            "n_events": 1,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
    ]
    extension = encode_extension(
        _SiteSource(),
        request,
        publication,
        context=context,
        rows=rows,
        definition=definition,
        source_recipe=source_recipe,
        experiment_id="exp",
    )
    assert isinstance(extension, SiteVolumeExtension)
    assert publication.calls[0][0] == "site_volume"
    assert publication.calls[0][1].schema().names == tuple(
        ARTIFACT_RELATION_SCHEMAS["site_volume"][i][0]
        for i in range(len(ARTIFACT_RELATION_SCHEMAS["site_volume"]))
    )


def test_read_validates_relation_and_returns_site_volume_rows():
    request = _valid_request()
    relation = ArtifactRelationRef(
        artifact_id=AID,
        generation_id=GID,
        role="site_volume",
        relation={"name": "site_volume_r"},
        schema_sha256=schema_sha256("site_volume", ARTIFACT_RELATION_SCHEMAS["site_volume"]),
        content_sha256=content_sha256(
            "site_volume",
            ARTIFACT_RELATION_SCHEMAS["site_volume"],
            [],
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
        ),
        row_count=0,
        primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
    )
    extension = SiteVolumeExtension(
        relation=relation,
        definition_sha256="0" * 64,
        source_provenance_sha256="0" * 64,
        measure_keys=("revenue",),
        first_ds=date(2024, 1, 1),
        last_ds=date(2024, 1, 2),
        freshness={"loaded_through": date(2024, 1, 2), "declared_complete": True},
    )
    snapshot = _Snapshot(relation, [])
    result = read_site_volume_extension(snapshot, extension, request=request, experiment_id="exp")
    assert result == [
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 1),
            "measure_key": "revenue",
            "n_events": 0,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 2),
            "measure_key": "revenue",
            "n_events": 0,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
    ]


def test_site_volume_round_trip_accepts_roundoff_but_rejects_impossible_totals():
    request = _valid_request()
    definition = {
        "kind": "site_volume",
        "metric_names": ["revenue"],
        "measure_keys": ["revenue"],
        "first_ds": "2024-01-01",
        "last_ds": "2024-01-01",
        "freshness": {"loaded_through": "2024-01-01", "declared_complete": True},
    }
    recipe = _recipe({"source": "facts", "measure_keys": ["revenue"]})
    context = _context(request, definition, recipe)
    total = 0.0
    for _ in range(10):
        total += 0.1
    row = {
        "experiment_id": "exp",
        "ds": date(2024, 1, 1),
        "measure_key": "revenue",
        "n_events": 10,
        "sum_value": total,
        "min_value": 0.1,
        "max_value": 0.1,
    }

    def encode(candidate):
        return encode_extension(
            _SiteSource(),
            request,
            _Publication(),
            context=context,
            rows=[candidate],
            definition=definition,
            source_recipe=recipe,
            experiment_id="exp",
        )

    extension = encode(row)
    assert isinstance(extension, SiteVolumeExtension)
    recovered = read_site_volume_extension(
        _Snapshot(extension.relation, [row]), extension, request=request, experiment_id="exp"
    )
    assert recovered[0]["sum_value"] == pytest.approx(1.0, abs=2e-16, rel=0)
    for impossible in (1.0 - 2e-15, 1.0 + 2e-15):
        corrupt = {**row, "sum_value": impossible}
        relation = extension.relation.model_copy(
            update={
                "content_sha256": content_sha256(
                    "site_volume",
                    ARTIFACT_RELATION_SCHEMAS["site_volume"],
                    [corrupt],
                    primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
                )
            }
        )
        corrupt_extension = extension.model_copy(update={"relation": relation})
        with pytest.raises(ArtifactContractError) as encoded:
            encode(corrupt)
        with pytest.raises(ArtifactContractError) as decoded:
            read_site_volume_extension(
                _Snapshot(relation, [corrupt]),
                corrupt_extension,
                request=request,
                experiment_id="exp",
            )
        assert encoded.value.code == decoded.value.code == "artifact.extension.invalid"


def test_missing_extension_and_observational_or_contrast_are_coded_refusals():
    _expect_code(
        "artifact.extension.missing",
        lambda: read_extension(
            cast(ArtifactSnapshot, SimpleNamespace()), None, request=_valid_request()
        ),
    )
    source = SimpleNamespace(operations=frozenset())
    _expect_code(
        "artifact.evidence.unavailable",
        lambda: encode_extension(source, _valid_request(), _Publication(), rows=[], context=None),
    )


def test_evidence_gap_raises_artifact_contract_error_with_evidence_code():
    """A source that cannot supply an extension's requested evidence refuses
    with artifact.evidence.unavailable, never the capability-gap code."""
    source = SimpleNamespace(operations=frozenset())
    with pytest.raises(ArtifactContractError) as caught:
        encode_extension(source, _valid_request(), _Publication(), rows=[], context=None)
    assert isinstance(caught.value, ArtifactContractError)
    assert caught.value.code == "artifact.evidence.unavailable"


def test_site_volume_rejects_incomplete_absent_rows():
    rows = [
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 1),
            "measure_key": "revenue",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        }
    ]
    relation = ArtifactRelationRef(
        artifact_id=AID,
        generation_id=GID,
        role="site_volume",
        relation={"name": "site_volume_r"},
        schema_sha256=schema_sha256("site_volume", ARTIFACT_RELATION_SCHEMAS["site_volume"]),
        content_sha256=content_sha256(
            "site_volume",
            ARTIFACT_RELATION_SCHEMAS["site_volume"],
            rows,
            primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
        ),
        row_count=1,
        primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["site_volume"],
    )
    extension = SiteVolumeExtension(
        relation=relation,
        definition_sha256="0" * 64,
        source_provenance_sha256="0" * 64,
        measure_keys=("revenue",),
        first_ds=date(2024, 1, 1),
        last_ds=date(2024, 1, 2),
        freshness={"loaded_through": date(2024, 1, 2), "declared_complete": False},
    )
    with pytest.raises(ArtifactContractError) as raised:
        read_site_volume_extension(
            _Snapshot(relation, rows), extension, request=_valid_request(), experiment_id="exp"
        )
    assert raised.value.code == "artifact.extension.invalid"


def _all_extension_cases():
    return [
        (
            {"kind": "breakout_dimension", "property_name": "country", "source_name": "dims"},
            {"kind": "breakout_dimension", "property_name": "country"},
            [
                {
                    "experiment_id": "exp",
                    "unit_id": "u1",
                    "value_is_missing": False,
                    "dimension_value": "US",
                }
            ],
        ),
        (
            {"kind": "factor_dimension", "property_name": "plan", "source_name": "dims"},
            {"kind": "factor_dimension", "property_name": "plan"},
            [
                {
                    "experiment_id": "exp",
                    "unit_id": "u1",
                    "value_is_missing": False,
                    "factor_value": "pro",
                }
            ],
        ),
        (
            {"kind": "cluster_identity", "cluster_name": "account"},
            {"kind": "cluster_identity", "cluster_name": "account"},
            [{"experiment_id": "exp", "unit_id": "u1", "cluster_id": "c1"}],
        ),
        (
            {"kind": "cuped_preperiod", "metric_name": "revenue"},
            {
                "kind": "cuped_preperiod",
                "metric_name": "revenue",
                "measure_key": "revenue",
                "aggregation": "sum",
                "window_start_days": -7,
                "window_end_days": 0,
            },
            [{"experiment_id": "exp", "unit_id": "u1", "x": 3.0}],
        ),
        (
            {"kind": "assignment_counts", "populations": ["assigned"]},
            {"kind": "assignment_counts", "populations": ["assigned"]},
            [
                {
                    "experiment_id": "exp",
                    "population": "assigned",
                    "group_id": "control",
                    "n_units": 1,
                    "n_randomization_units": 1,
                }
            ],
        ),
        (
            {"kind": "trigger_population", "trigger_name": "trigger"},
            {
                "kind": "trigger_population",
                "trigger_name": "trigger",
                "observation_cutoff_ts": "2024-01-03T00:00:00Z",
                "complete_through_ts": "2024-01-02T00:00:00Z",
            },
            [
                {
                    "experiment_id": "exp",
                    "unit_id": "u1",
                    "first_trigger_ts": datetime(2024, 1, 1, tzinfo=UTC),
                }
            ],
        ),
        (
            {"kind": "encouragement_uptake", "uptake_name": "clicked"},
            {
                "kind": "encouragement_uptake",
                "uptake_name": "clicked",
                "window_days": 7,
                "one_sided": True,
            },
            [
                {
                    "experiment_id": "exp",
                    "unit_id": "u1",
                    "uptake": True,
                    "first_uptake_ts": datetime(2025, 1, 1, tzinfo=UTC),
                }
            ],
        ),
        (
            {"kind": "site_volume", "metric_names": ["revenue"]},
            {
                "kind": "site_volume",
                "metric_names": ["revenue"],
                "measure_keys": ["revenue"],
                "first_ds": "2024-01-01",
                "last_ds": "2024-01-01",
                "freshness": {"loaded_through": "2024-01-01", "declared_complete": True},
            },
            [
                {
                    "experiment_id": "exp",
                    "ds": date(2024, 1, 1),
                    "measure_key": "revenue",
                    "n_events": 1,
                    "sum_value": 2.0,
                    "min_value": 2.0,
                    "max_value": 2.0,
                }
            ],
        ),
    ]


def _trigger_population_fixture(request, definition, rows):
    recipe = _recipe({"source": "facts", "request": request["kind"]})
    context = _context(request, definition, recipe)
    primary_key = ARTIFACT_RELATION_PRIMARY_KEYS["trigger_population"]
    relation = ArtifactRelationRef(
        artifact_id=AID,
        generation_id=GID,
        role="trigger_population",
        relation={"name": "trigger_population_r"},
        schema_sha256=schema_sha256(
            "trigger_population", ARTIFACT_RELATION_SCHEMAS["trigger_population"]
        ),
        content_sha256=content_sha256(
            "trigger_population",
            ARTIFACT_RELATION_SCHEMAS["trigger_population"],
            rows,
            primary_key=primary_key,
        ),
        row_count=len(rows),
        primary_key=primary_key,
    )
    extension = TriggerPopulationExtension(
        relation=relation,
        definition_sha256=extension_definition_sha256(
            "trigger_population", canonical_json(definition)
        ),
        source_provenance_sha256=extension_source_provenance_sha256(
            "trigger_population", canonical_json(recipe)
        ),
        trigger_name=request["trigger_name"],
        observation_cutoff_ts=definition["observation_cutoff_ts"],
        complete_through_ts=definition["complete_through_ts"],
    )
    return context, extension


@pytest.mark.parametrize(("req", "definition", "rows"), _all_extension_cases())
def test_every_closed_extension_producer_and_reader_round_trips(req, definition, rows):
    recipe = _recipe({"source": "facts", "request": req["kind"]})
    extension = encode_extension(
        _AllSource(),
        req,
        _Publication(),
        context=_context(req, definition, recipe),
        rows=rows,
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    result = read_extension(
        _Snapshot(extension.relation, rows),
        extension,
        request=req,
        context=_context(req, definition, recipe),
        experiment_id="exp",
    )
    assert result == rows


def test_same_kind_catalog_entries_select_exact_request():
    selected = {"kind": "site_volume", "metric_names": ["revenue"]}
    other = {"kind": "site_volume", "metric_names": ["clicks"]}
    definition = {
        "kind": "site_volume",
        "metric_names": ["revenue"],
        "measure_keys": ["revenue"],
        "first_ds": "2024-01-01",
        "last_ds": "2024-01-01",
        "freshness": {"loaded_through": "2024-01-01", "declared_complete": True},
    }
    other_definition = {**definition, "metric_names": ["clicks"], "measure_keys": ["clicks"]}
    recipe = _recipe({"source": "facts"})
    context = _context_catalog(
        [
            {
                "request": req,
                "canonical_definition_json": canonical_json(defn),
                "canonical_source_recipe_json": canonical_json(recipe),
                "definition_sha256": extension_definition_sha256(
                    cast(str, req["kind"]), canonical_json(defn)
                ),
                "source_provenance_sha256": extension_source_provenance_sha256(
                    cast(str, req["kind"]), canonical_json(recipe)
                ),
            }
            for req, defn in ((selected, definition), (other, other_definition))
        ]
    )
    rows = [
        {
            "experiment_id": "exp",
            "ds": date(2024, 1, 1),
            "measure_key": "revenue",
            "n_events": 1,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        }
    ]
    ext = encode_extension(
        _AllSource(),
        selected,
        _Publication(),
        context=context,
        rows=rows,
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    assert isinstance(ext, SiteVolumeExtension)
    assert ext.measure_keys == ("revenue",)
    assert (
        read_extension(
            _Snapshot(ext.relation, rows),
            ext,
            context=context,
            experiment_id="exp",
        )
        == rows
    )


@pytest.mark.parametrize(("req", "definition", "rows"), _all_extension_cases())
def test_every_extension_rejects_missing_operation_and_extension(req, definition, rows):
    with pytest.raises(ArtifactContractError) as raised1:
        encode_extension(
            SimpleNamespace(operations=frozenset()), req, _Publication(), context=None, rows=rows
        )
    assert raised1.value.code == "artifact.evidence.unavailable"
    with pytest.raises(ArtifactContractError) as raised2:
        read_extension(_Snapshot(None, []), None, request=req)
    assert raised2.value.code == "artifact.extension.missing"


@pytest.mark.parametrize(("req", "definition", "rows"), _all_extension_cases())
def test_every_extension_rejects_schema_null_domain_fk_and_join_mismatch(req, definition, rows):
    recipe = _recipe({"source": "facts", "request": req["kind"]})
    context = _context(req, definition, recipe)
    extension = encode_extension(
        _AllSource(),
        req,
        _Publication(),
        context=context,
        rows=rows,
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    bad_row = dict(rows[0])
    bad_row[next(iter(bad_row))] = None
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, [bad_row]),
            extension,
            request=req,
            context=context,
            experiment_id="exp",
        )
    assert raised.value.code == "artifact.extension.invalid"
    missing_field = dict(rows[0])
    missing_field.pop(next(iter(missing_field)))
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, [missing_field]),
            extension,
            request=req,
            context=context,
            experiment_id="exp",
        )
    assert raised.value.code == "artifact.extension.invalid"
    wrong_fk = dict(rows[0])
    wrong_fk["experiment_id"] = "other"
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, [wrong_fk]),
            extension,
            request=req,
            context=context,
            experiment_id="exp",
        )
    assert raised.value.code == "artifact.extension.invalid"
    snapshot = _Snapshot(extension.relation, rows)
    if req["kind"] not in {"assignment_counts", "site_volume", "trigger_population"}:
        with pytest.raises(ArtifactContractError) as raised:
            read_extension(
                snapshot,
                extension,
                request=req,
                experiment_id="exp",
                exposure_keys=[("exp", "missing")],
            )
        assert raised.value.code == "artifact.extension.invalid"


@pytest.mark.parametrize(("req", "definition", "rows"), _all_extension_cases())
def test_every_extension_rejects_cross_role_key_generation_and_provenance(req, definition, rows):
    recipe = _recipe({"source": "facts", "request": req["kind"]})
    context = _context(req, definition, recipe)
    extension = encode_extension(
        _AllSource(),
        req,
        _Publication(),
        context=context,
        rows=rows,
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, rows),
            extension.model_copy(
                update={"relation": extension.relation.model_copy(update={"role": "exposures"})}
            ),
            request=req,
        )
    assert raised.value.code == "artifact.extension.invalid"
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, rows),
            extension.model_copy(
                update={"relation": extension.relation.model_copy(update={"primary_key": ("bad",)})}
            ),
            request=req,
        )
    assert raised.value.code == "artifact.extension.invalid"
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, rows),
            extension.model_copy(
                update={
                    "relation": extension.relation.model_copy(
                        update={"generation_id": UUID("00000000-0000-0000-0000-000000000003")}
                    )
                }
            ),
            request=req,
        )
    assert raised.value.code == "artifact.snapshot.mixed"
    with pytest.raises(ArtifactContractError) as raised:
        read_extension(
            _Snapshot(extension.relation, rows),
            extension.model_copy(update={"definition_sha256": "f" * 64}),
            request=req,
            context=context,
        )
    assert raised.value.code == "artifact.extension.invalid"


@pytest.mark.parametrize(("req", "definition", "rows"), _all_extension_cases())
def test_source_driven_producers_invoke_declared_task5_operation(req, definition, rows):
    operation = {
        "breakout_dimension": "breakout_source",
        "factor_dimension": "breakout_source",
        "cluster_identity": "day_source",
        "cuped_preperiod": "day_source",
        "assignment_counts": "triggered_counts",
        "encouragement_uptake": "day_source",
        "trigger_population": "triggered_source",
        "site_volume": "sitewide_evidence",
    }[req["kind"]]
    recipe = _recipe({"source": "facts", "request": req["kind"]})
    source = _RowsOperationSource(req["kind"], rows)
    if req["kind"] == "site_volume":
        with pytest.raises(ArtifactContractError) as caught:
            encode_extension(
                source,
                req,
                _Publication(),
                context=_context(req, definition, recipe),
                definition=definition,
                source_recipe=recipe,
                experiment_id="exp",
            )
        assert caught.value.code == "artifact.evidence.unavailable"
    else:
        encode_extension(
            source,
            req,
            _Publication(),
            context=_context(req, definition, recipe),
            definition=definition,
            source_recipe=recipe,
            experiment_id="exp",
        )
    expected_calls = ["assignment_counts"] if req["kind"] == "assignment_counts" else [operation]
    assert source.called == expected_calls


@pytest.mark.parametrize(
    "kind",
    [
        "breakout_dimension",
        "factor_dimension",
        "cluster_identity",
        "cuped_preperiod",
        "encouragement_uptake",
        "trigger_population",
    ],
)
def test_concrete_task5_result_objects_refuse_without_truthful_unit_rows(kind):
    req, definition, _rows = next(
        case for case in _all_extension_cases() if case[0]["kind"] == kind
    )
    recipe = _recipe({"source": "facts", "request": kind})
    source = _ConcreteResultSource(kind, [])
    _expect_code(
        "artifact.evidence.unavailable",
        lambda: encode_extension(
            source,
            req,
            _Publication(),
            context=_context(req, definition, recipe),
            definition=definition,
            source_recipe=recipe,
            experiment_id="exp",
        ),
    )


def test_clustered_assignment_preserves_randomization_and_unit_counts():
    request = {"kind": "assignment_counts", "populations": ["assigned", "triggered"]}
    definition = {"kind": "assignment_counts", "populations": ["assigned", "triggered"]}
    recipe = _recipe({"source": "facts", "request": "assignment_counts"})

    class ClusteredSource(_RowsOperationSource):
        _experiment = SimpleNamespace(cluster="account")

        def assignment_counts(self, *, population="assigned"):
            self.called.append("assignment_counts")
            return {"control": 2}

        def unit_counts(self):
            self.called.append("unit_counts")
            return {"control": 10}

        def triggered_counts(self):
            self.called.append("triggered_counts")
            return ("cluster", {"control": 2}, {"control": 7})

    source = ClusteredSource("assignment_counts", [])
    publication = _Publication()
    extension = encode_extension(
        source,
        request,
        publication,
        context=_context(request, definition, recipe),
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    assert sorted(source.called) == ["assignment_counts", "triggered_counts", "unit_counts"]
    rows = publication.calls[0][1].to_pyarrow().to_pylist()
    assert rows[0]["n_units"] == 10
    assert rows[0]["n_randomization_units"] == 2
    assert rows[1]["n_units"] == 7
    assert rows[1]["n_randomization_units"] == 2
    assert extension.relation.row_count == 2


def test_trigger_population_join_coverage_accepts_subset_and_rejects_superset():
    request, definition, _ = next(
        case for case in _all_extension_cases() if case[0]["kind"] == "trigger_population"
    )
    rows = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "first_trigger_ts": datetime(2024, 1, 1, tzinfo=UTC),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "first_trigger_ts": datetime(2024, 1, 2, tzinfo=UTC),
        },
    ]
    context, extension = _trigger_population_fixture(request, definition, rows)
    assert extension.observation_cutoff_ts == datetime(2024, 1, 3, tzinfo=UTC)
    assert extension.complete_through_ts == datetime(2024, 1, 2, tzinfo=UTC)
    _ = read_extension(
        _Snapshot(extension.relation, rows),
        extension,
        request=request,
        context=context,
        experiment_id="exp",
        exposure_keys=[("exp", "u1"), ("exp", "u2"), ("exp", "u3")],
    )
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            _Snapshot(extension.relation, rows),
            extension,
            request=request,
            context=context,
            experiment_id="exp",
            exposure_keys=[("exp", "u1")],
        ),
    )


@pytest.mark.parametrize(
    "operation",
    [
        encode_observational_extension,
        read_observational_extension,
        validate_observational_extension,
        encode_contrast_extension,
        read_contrast_extension,
        validate_contrast_extension,
    ],
)
def test_observational_and_contrast_operations_refuse(operation):
    _expect_code("artifact.evidence.unavailable", operation)


def test_assignment_accounting_labels_become_descriptor_audit_metadata():
    request = {"kind": "assignment_counts", "populations": ["assigned"]}
    definition = {"kind": "assignment_counts", "populations": ["assigned"]}
    recipe = _recipe({"source": "facts", "request": "assignment_counts"})

    class AuditedSource(_RowsOperationSource):
        def assignment_counts(self, *, population="assigned"):
            self.called.append("assignment_counts")
            return {"control": 5, "(mixed assignment)": 2, "(unassigned)": 1}

        def unit_counts(self):
            self.called.append("unit_counts")
            return {"control": 5, "(mixed assignment)": 2, "(unassigned)": 1}

    source = AuditedSource("assignment_counts", [])
    publication = _Publication()
    extension = encode_extension(
        source,
        request,
        publication,
        context=_context(request, definition, recipe),
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    rows = publication.calls[0][1].to_pyarrow().to_pylist()
    assert {row["group_id"] for row in rows} == {"control"}
    assert isinstance(extension, AssignmentCountsExtension)
    assert extension.mixed_unit_count == 2
    assert extension.unassigned_unit_count == 1

    tampered = extension.model_copy(update={"mixed_unit_count": 0})
    snapshot = _Snapshot(tampered.relation, rows)
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            snapshot,
            tampered,
            request=request,
            context=_context(request, definition, recipe),
            experiment_id="exp",
        ),
    )


def test_zero_trigger_arms_encode_as_zero_count_rows():
    request = {"kind": "assignment_counts", "populations": ["assigned", "triggered"]}
    definition = {"kind": "assignment_counts", "populations": ["assigned", "triggered"]}
    recipe = _recipe({"source": "facts", "request": "assignment_counts"})

    class SparseTriggerSource(_RowsOperationSource):
        def assignment_counts(self, *, population="assigned"):
            self.called.append("assignment_counts")
            return {"control": 2, "treatment": 3}

        def unit_counts(self):
            self.called.append("unit_counts")
            return {"control": 2, "treatment": 3}

        def triggered_counts(self):
            self.called.append("triggered_counts")
            return ("unit", {"control": 1}, {"control": 1})

    publication = _Publication()
    encode_extension(
        SparseTriggerSource("assignment_counts", []),
        request,
        publication,
        context=_context(request, definition, recipe),
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    rows = publication.calls[0][1].to_pyarrow().to_pylist()
    triggered = {r["group_id"]: r for r in rows if r["population"] == "triggered"}
    assert triggered["treatment"]["n_units"] == 0
    assert triggered["treatment"]["n_randomization_units"] == 0
    assert triggered["control"]["n_randomization_units"] == 1


def test_non_integer_audit_counts_refuse_before_fingerprint():
    request = {"kind": "assignment_counts", "populations": ["assigned"]}
    definition = {"kind": "assignment_counts", "populations": ["assigned"]}
    recipe = _recipe({"source": "facts", "request": "assignment_counts"})

    class AuditedSource(_RowsOperationSource):
        def assignment_counts(self, *, population="assigned"):
            self.called.append("assignment_counts")
            return {"control": 5, "(mixed assignment)": 2, "(unassigned)": 1}

        def unit_counts(self):
            self.called.append("unit_counts")
            return {"control": 5, "(mixed assignment)": 2, "(unassigned)": 1}

    publication = _Publication()
    extension = encode_extension(
        AuditedSource("assignment_counts", []),
        request,
        publication,
        context=_context(request, definition, recipe),
        definition=definition,
        source_recipe=recipe,
        experiment_id="exp",
    )
    rows = publication.calls[0][1].to_pyarrow().to_pylist()
    tampered = extension.model_copy(update={"mixed_unit_count": 2.5})
    snapshot = _Snapshot(extension.relation, rows)
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            snapshot,
            tampered,
            request=request,
            context=_context(request, definition, recipe),
            experiment_id="exp",
        ),
    )


@pytest.mark.parametrize("bad_count", [2.0, "2", True, -1])
def test_wire_audit_counts_reject_non_strict_integers(bad_count):
    from increment.semantics.artifact import AssignmentCountsExtension

    payload = {
        "kind": "assignment_counts",
        "extension_version": 1,
        "relation": {
            "digest_format": 1,
            "artifact_id": str(AID),
            "generation_id": str(GID),
            "relation": {"name": "assignment_counts", "role": "assignment_counts"},
            "row_count": 1,
            "content_sha256": "0" * 64,
        },
        "definition_sha256": "0" * 64,
        "source_provenance_sha256": "0" * 64,
        "populations": ["assigned"],
        "mixed_unit_count": bad_count,
        "unassigned_unit_count": 0,
        "assignment_fingerprint_sha256": "0" * 64,
    }
    import pydantic

    with pytest.raises(pydantic.ValidationError) as excinfo:
        AssignmentCountsExtension.model_validate(payload)
    errors = excinfo.value.errors()
    located = [error for error in errors if error["loc"] == ("mixed_unit_count",)]
    assert len(located) == 1, [error["loc"] for error in errors]


def test_read_rejects_relabeled_registered_kind_as_coded_identity_refusal():
    """A stored model relabeled to another registered kind must refuse, not crash."""
    extension = _empty_site_volume_extension().model_copy(update={"kind": "breakout_dimension"})
    snapshot = _Snapshot(extension.relation, [])
    _expect_code(
        "artifact.extension.invalid",
        lambda: read_extension(
            snapshot,
            extension,
            request={"kind": "breakout_dimension", "property_name": "country"},
            experiment_id="exp",
        ),
    )
    assert snapshot.calls == 0


@pytest.mark.slow
def test_clustered_artifact_publishes_cluster_grain_randomization_counts(tmp_path) -> None:
    """n_randomization_units is the randomization-grain count: under a declared
    cluster it is the number of CLUSTERS, which the reader hands back as
    grain='cluster'. Publishing unit counts there inflates cluster-robust n."""
    import ibis

    from increment import Analysis
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.session import WarehouseArtifactStore
    from increment.query.source import open_artifact
    from increment.semantics.loader import load
    from tests.analysis_factory import _native_source
    from tests.test_analysis_trigger import _clustered_trigger_events, _defs_yaml_clustered

    con = ibis.duckdb.connect()
    con.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(n_stores_per_arm=10, trigger_rate=0.5),
    )
    definitions = tmp_path / "defs.yml"
    definitions.write_text(_defs_yaml_clustered())
    analysis = Analysis.from_definitions("exp", definitions, con, store="none")

    expected_grain, expected_counts, _units = _native_source(analysis).triggered_counts()
    assert expected_grain == "cluster"

    store = WarehouseArtifactStore(con, schema_name="cluster_counts_artifact")
    context = artifact_context(load(definitions), analysis.experiment, "error")
    wanted = {"assignment_counts", "trigger_population", "cluster_identity"}
    requests = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in wanted
    ]
    ref = analysis.publish_unit_day_artifact(store, extensions=requests)

    adopted = open_artifact(store, ref, expected_context=context)
    try:
        grain, counts, _adopted_units = adopted.triggered_counts()
        assert grain == "cluster"
        assert counts == expected_counts
    finally:
        adopted.close()


def test_dimension_extension_missing_an_exposure_key_refuses_on_read():
    """A dimension extension whose rows omit a unit present in exposures is
    refused with the join-key coverage code when the reader threads the
    source's exposure keys into the coverage check."""
    import ibis

    from increment import Analysis
    from tests.test_unit_day_artifact_facade import _extensions, _native

    con, native, context, store = _native()
    requests = _extensions(context, "breakout_dimension")
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:
        # Inject a ghost exposure row so the extension's stored coverage is short one key.
        original_ensure = adopted._src._ensure

        def _ensure_with_ghost_exposure(role):
            table = original_ensure(role)
            if role == "exposures":
                first_row = table.limit(1).execute().to_dict("records")[0]
                ghost = ibis.memtable(
                    [{**first_row, "unit_id": "ghost-unit"}], schema=table.schema()
                )
                table = table.union(ghost)
            return table

        adopted._src._ensure = _ensure_with_ghost_exposure
        with pytest.raises(ArtifactContractError) as exc:
            adopted.run_breakout(metrics=["purchase_rate"])
        assert exc.value.code == "artifact.extension.invalid"
    finally:
        adopted.close()
        native.close()


def test_site_volume_extension_read_through_read_extension_accepts_exposure_keys(
    tmp_path,
) -> None:
    """A site-volume extension read always threads exposure keys, even though
    its noop exposure-coverage check accepts them without filtering rows: an
    adopted artifact serves the same whole-site volume as the native path."""
    from increment import Analysis
    from tests.test_unit_day_artifact_facade import _definitions, _extensions, _native

    definitions = _definitions(tmp_path, site_volume_only=True)
    con, native, context, store = _native(definitions=definitions)
    requests = _extensions(context, "site_volume")
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:
        expected = native.sitewide("purchase_rate")
        result = adopted.sitewide("purchase_rate")
        assert result.site_total_volume > 0  # ty: ignore[unresolved-attribute]
        assert result.site_total_volume == pytest.approx(  # ty: ignore[unresolved-attribute]
            expected.site_total_volume
        )
    finally:
        adopted.close()
        native.close()


def test_extension_relation_publishes_and_reopens_under_digest_format_one() -> None:
    """Only the base relations move to digest_format 2. An extension relation
    keeps format 1, which validate_extension recomputes on read, so a
    breakout extension still opens and gives the same breakout rows."""
    from increment import Analysis
    from tests.test_unit_day_artifact_facade import _extensions, _native

    _con, native, context, store = _native()
    requests = _extensions(context, "breakout_dimension")
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    assert manifest.base.exposures.digest_format == 2
    assert [e.relation.digest_format for e in manifest.extensions] == [1] * len(requests)
    adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    try:

        def rows(analysis):
            breakout = analysis.run_breakout(metrics=["purchase_rate"])
            return sorted((str(row.dimension_value), row.abs_diff) for row in breakout)

        adopted_rows, native_rows = rows(adopted), rows(native)
        assert adopted_rows
        assert [value for value, _ in adopted_rows] == [value for value, _ in native_rows]
        assert [diff for _, diff in adopted_rows] == pytest.approx(
            [diff for _, diff in native_rows]
        )
    finally:
        adopted.close()
        native.close()


def test_site_volume_publication_deduplicates_shared_physical_recipes(tmp_path) -> None:
    """Logical names sharing a recipe publish one physical row set and adopt independently."""
    import yaml

    from increment import Analysis
    from increment.estimation.sitewide import SitewideImpact
    from tests.test_unit_day_artifact_facade import _definitions, _extensions, _native

    definitions = _definitions(tmp_path, site_volume_only=True)
    payload = yaml.safe_load(definitions.read_text())
    original = next(metric for metric in payload["metrics"] if metric["name"] == "purchase_rate")
    payload["metrics"].append({**original, "name": "purchase_rate_copy"})
    payload["metrics"].append(
        {
            "type": "mean",
            "name": "revenue_per_user_site_volume",
            "description": "Revenue per enrolled user, whole-site control",
            "entity": "user_id",
            "preferred_direction": "increase",
            "fact": "purchase",
            "aggregation": "sum",
            "window_days": 7,
        }
    )
    experiment = next(
        item for item in payload["experiments"] if item["name"] == "new_onboarding_v2"
    )
    experiment["plan"] = {
        "secondaries": [
            "purchase_rate",
            "purchase_rate_copy",
            "revenue_per_user_site_volume",
        ]
    }
    definitions.write_text(yaml.safe_dump(payload, sort_keys=False))

    connection, native, context, store = _native(definitions=definitions)
    adopted = None
    try:
        expected = {
            name: native.sitewide(name).site_total_volume
            for name in (
                "purchase_rate",
                "purchase_rate_copy",
                "revenue_per_user_site_volume",
            )
        }
        requests = _extensions(context, "site_volume")
        assert len(requests) == 1
        reference = native.publish_unit_day_artifact(store, extensions=requests)
        adopted = Analysis.from_unit_day_artifact(store, reference, expected_context=context)

        for name, total in expected.items():
            result = adopted.sitewide(name)
            assert isinstance(result, SitewideImpact)
            assert result.site_total_volume == pytest.approx(total)

        with store.open_snapshot(reference) as snapshot:
            manifest = snapshot.read_manifest(
                reference.manifest, expected_sha256=reference.manifest_sha256
            )
            extension = next(item for item in manifest.extensions if item.kind == "site_volume")
            rows = (
                snapshot.verify_relation(extension.relation, expected_role="site_volume")
                .execute()
                .to_dict("records")
            )
            keys = [(row["experiment_id"], row["ds"], row["measure_key"]) for row in rows]
        assert len(keys) == len(set(keys))
        declared_keys = next(
            json.loads(entry.canonical_definition_json)["measure_keys"]
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind == "site_volume"
        )
        assert {key for _experiment_id, _ds, key in keys} == set(declared_keys)
    finally:
        if adopted is not None:
            adopted.close()
        native.close()
        connection.disconnect()
