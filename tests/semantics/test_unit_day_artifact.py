import json
import subprocess
import sys
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from increment._canonical import canonical_json_bytes
from increment.errors import DefinitionError
from increment.semantics.artifact import (
    ArtifactContext,
    ArtifactExtensionCatalogEntry,
    ArtifactRelationRef,
    AssignmentCountsRequest,
    BaseRelations,
    BreakoutDimensionRequest,
    Freshness,
    MeasureManifest,
    RelationLocator,
    SimpleMetricMeasure,
    SiteVolumeExtension,
    SiteVolumeRequest,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
    _artifact_context_digest,
    _artifact_manifest_digest,
    _extension_definition_digest,
)
from increment.semantics.models import AnalysisPlan, Experiment

_SHA = "0" * 64


def _relation(artifact_id, generation_id, role, primary_key):
    return ArtifactRelationRef(
        artifact_id=artifact_id,
        generation_id=generation_id,
        role=role,
        relation=RelationLocator(catalog="warehouse", schema="analytics", name=role),
        schema_sha256=_SHA,
        content_sha256=_SHA,
        row_count=0,
        primary_key=primary_key,
    )


def _manifest():
    artifact_id = uuid4()
    generation_id = uuid4()
    base = BaseRelations(
        exposures=_relation(artifact_id, generation_id, "exposures", ("experiment_id", "unit_id")),
        measure_stats=_relation(
            artifact_id,
            generation_id,
            "measure_stats",
            ("experiment_id", "unit_id", "ds", "measure_key"),
        ),
    )
    experiment = Experiment(
        name="exp",
        exposure="assigned",
        unit="unit_id",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        control_group="control",
        plan=AnalysisPlan(primary="conversion"),
    )
    context_json = canonical_json_bytes(
        {
            "context_format": 2,
            "extension_catalog": [],
            "definitions": {
                "day_boundary": "UTC",
                "metrics": [
                    {
                        "entity": "unit_id",
                        "fact": "orders",
                        "name": "conversion",
                        "type": "mean",
                    }
                ],
            },
            "experiment": experiment.model_dump(mode="json"),
            "experiment_name": "exp",
            "window_days": {"start": "2025-01-01", "end": None, "observation_horizon": None},
        }
    ).decode("utf-8")
    context = ArtifactContext(
        canonical_json=context_json,
        sha256=_artifact_context_digest(context_json),
    )
    values: dict[str, Any] = {
        "artifact_id": artifact_id,
        "generation_id": generation_id,
        "experiment_id": "exp",
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
        "day_boundary": "UTC",
        "first_ds": date(2025, 1, 1),
        "last_ds": date(2025, 1, 3),
        "base": base,
        "measures": (
            MeasureManifest(
                measure_key="orders",
                source_provenance_sha256=_SHA,
                freshness=Freshness(loaded_through=date(2025, 1, 3), declared_complete=True),
            ),
        ),
        "metric_measures": (SimpleMetricMeasure(metric_name="conversion", measure_key="orders"),),
        "context": context,
        "manifest_sha256": _SHA,
    }
    digest_input = UnitDayArtifactManifest.model_construct(**values)
    values["manifest_sha256"] = _artifact_manifest_digest(digest_input)
    return UnitDayArtifactManifest(**values)


def test_artifact_context_lives_in_artifact_module():
    import increment.semantics.artifact as artifact
    from increment.semantics.models import ArtifactContext

    assert ArtifactContext is artifact.ArtifactContext
    assert ArtifactContext.__module__ == "increment.semantics.artifact"


@pytest.mark.slow
def test_models_legacy_explicit_import_loads_artifact_only_on_demand():
    code = """
import sys

import increment.semantics.models

assert "increment.semantics.artifact" not in sys.modules
from increment.semantics.models import ArtifactContext
from increment.semantics.artifact import ArtifactContext as MovedArtifactContext
assert ArtifactContext is MovedArtifactContext
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_models_legacy_exports_support_explicit_and_wildcard_imports_lazily():
    import increment.semantics.artifact as artifact
    import increment.semantics.models as models

    public_models_names = {name for name in vars(models) if not name.startswith("_")}
    assert "ArtifactContext" not in vars(models)
    assert public_models_names <= set(models.__all__)

    namespace: dict[str, Any] = {}
    exec("from increment.semantics.models import *", namespace)

    assert set(models.__all__) <= namespace.keys()
    assert namespace["Definitions"] is models.Definitions
    lazy_names = [name for name in models.__all__ if name not in vars(models)]
    assert lazy_names
    for name in lazy_names:
        assert namespace[name] is getattr(artifact, name)
    assert "ArtifactContext" not in vars(models)


def test_valid_manifest_is_frozen_and_round_trips_deterministically():
    manifest = _manifest()
    assert (
        manifest.model_dump()
        == UnitDayArtifactManifest.model_validate(manifest.model_dump()).model_dump()
    )
    assert (
        manifest.model_dump_json()
        == UnitDayArtifactManifest.model_validate_json(manifest.model_dump_json()).model_dump_json()
    )
    with pytest.raises(ValidationError):
        manifest.experiment_id = "changed"


def test_relation_locator_alias_and_identifier_syntax_are_strict():
    locator = RelationLocator(catalog="warehouse", schema="analytics", name="unit_day")
    assert locator.model_dump() == {
        "catalog": "warehouse",
        "schema": "analytics",
        "name": "unit_day",
    }
    assert '"schema":"analytics"' in locator.model_dump_json()
    for bad in ("analytics.table", '"table"', "table;drop", "table name", ""):
        expected = ValidationError if bad == "" else DefinitionError
        with pytest.raises(expected):
            RelationLocator(name=bad)
    with pytest.raises(DefinitionError):
        RelationLocator(schema_name="analytics.prod", name="table")


def test_relation_locator_round_trips_a_hyphenated_catalog():
    locator = RelationLocator(catalog="release-project-123", schema="analytics", name="unit_day")
    restored = RelationLocator.model_validate_json(locator.model_dump_json())
    assert restored.catalog == "release-project-123"
    assert restored.schema_name == "analytics"
    assert restored.name == "unit_day"


@pytest.mark.parametrize(
    "catalog",
    [
        "release.project",
        '"release-project"',
        "`release-project`",
        "release project",
        "release;drop",
    ],
)
def test_relation_locator_refuses_sql_fragments_in_catalog(catalog):
    with pytest.raises(DefinitionError):
        RelationLocator(catalog=catalog, schema="analytics", name="unit_day")


@pytest.mark.parametrize("field", ["schema_name", "name"])
def test_relation_locator_keeps_non_catalog_identifiers_strict(field):
    with pytest.raises(DefinitionError):
        RelationLocator(**{"name": "unit_day", field: "not-an-identifier"})


def test_ref_and_required_relation_bindings_refuse_tampering():
    manifest = _manifest()
    bad = manifest.model_dump()
    bad["base"]["measure_stats"]["generation_id"] = str(uuid4())
    with pytest.raises(DefinitionError) as exc_info:
        UnitDayArtifactManifest.model_validate(bad)
    assert exc_info.value.code == "definition.base_relations.share_generation_id"

    with pytest.raises(DefinitionError):
        UnitDayArtifactRef(
            artifact_id=manifest.artifact_id,
            generation_id=manifest.generation_id,
            manifest=manifest.base.exposures.relation,
            manifest_sha256="A" * 64,
        )


def test_metric_bindings_require_declared_measures_and_ratio_distinctness():
    manifest = _manifest()
    bad = manifest.model_dump()
    bad["metric_measures"][0]["measure_key"] = "missing"
    with pytest.raises(DefinitionError) as exc_info:
        UnitDayArtifactManifest.model_validate(bad)
    assert exc_info.value.code == "definition.unit_day.metric_references_undeclared"

    with pytest.raises(DefinitionError) as exc_info:
        from increment.semantics.artifact import RatioMetricMeasure

        RatioMetricMeasure(
            metric_name="ratio",
            numerator_measure_key="orders",
            denominator_measure_key="orders",
        )
    assert exc_info.value.code == "definition.ratio_metric.numerator_denominator_differ"


def test_closed_extension_requests_normalize_without_accepting_unknown_fields():
    request = AssignmentCountsRequest(populations=("triggered", "assigned"))
    assert request.populations == ("assigned", "triggered")
    request = SiteVolumeRequest(metric_names=("z", "a"))
    assert request.metric_names == ("a", "z")
    with pytest.raises(ValidationError):
        BreakoutDimensionRequest(
            property_name="country",
            source_name="facts",
            extra="nope",  # ty: ignore[unknown-argument] -- contract test intentionally passes forbidden field
        )
    with pytest.raises(DefinitionError):
        AssignmentCountsRequest(populations=("assigned", "assigned"))


def test_context_and_extension_catalog_hashes_refuse_recanonicalization_and_tampering():
    with pytest.raises(DefinitionError) as exc_info:
        ArtifactContext(canonical_json='{ "experiment": "exp" }', sha256=_SHA)
    assert exc_info.value.code == "definition.already_canonical_json"

    with pytest.raises(DefinitionError):
        ArtifactExtensionCatalogEntry(
            request=BreakoutDimensionRequest(property_name="country", source_name="facts"),
            canonical_definition_json='{"name":"country"}',
            canonical_source_recipe_json='{"source":"facts"}',
            definition_sha256=_SHA,
            source_provenance_sha256=_SHA,
        )


def test_site_volume_extension_binds_declared_simple_and_ratio_measures() -> None:
    base_manifest = _manifest()
    relation = _relation(
        base_manifest.artifact_id,
        base_manifest.generation_id,
        "site_volume",
        ("experiment_id", "ds", "measure_key"),
    )
    definition = {
        "kind": "site_volume",
        "metric_names": ["conversion"],
        "measure_keys": ["orders"],
        "first_ds": "2025-01-01",
        "last_ds": "2025-01-03",
        "freshness": {"loaded_through": "2025-01-03", "declared_complete": True},
    }
    definition_json = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    source_json = '{"recipe":"site"}'
    request = SiteVolumeRequest(metric_names=("conversion",))
    entry = ArtifactExtensionCatalogEntry(
        request=request,
        canonical_definition_json=definition_json,
        canonical_source_recipe_json=source_json,
        definition_sha256=_extension_definition_digest(
            "site_volume", definition_json, "extension-definition"
        ),
        source_provenance_sha256=_extension_definition_digest(
            "site_volume", source_json, "extension-source"
        ),
    )
    context_json = json.dumps(
        {
            "context_format": 2,
            "experiment": "exp",
            "extension_catalog": [entry.model_dump(mode="json")],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    context = ArtifactContext(
        canonical_json=context_json,
        sha256=_artifact_context_digest(context_json),
    )
    extension = SiteVolumeExtension(
        relation=relation,
        measure_keys=("orders",),
        first_ds=date(2025, 1, 1),
        last_ds=date(2025, 1, 3),
        freshness=Freshness(loaded_through=date(2025, 1, 3), declared_complete=True),
        definition_sha256=entry.definition_sha256,
        source_provenance_sha256=entry.source_provenance_sha256,
    )
    digest_input = base_manifest.model_copy(update={"context": context, "extensions": (extension,)})
    values = digest_input.model_dump()
    values["manifest_sha256"] = _artifact_manifest_digest(digest_input)
    valid = UnitDayArtifactManifest(**values)
    assert isinstance(valid.extensions[0], SiteVolumeExtension)
    assert valid.extensions[0].measure_keys == ("orders",)

    bad = valid.model_dump()
    bad["extensions"][0]["measure_keys"] = ["missing"]
    with pytest.raises(DefinitionError) as exc_info:
        UnitDayArtifactManifest.model_validate(bad)
    assert exc_info.value.code == "definition.unit_day.manifest_extension_absent"


# ── refusal-code coverage: artifact/relation/manifest field validators ──


def test_artifact_field_validators_carry_codes():
    with pytest.raises(DefinitionError) as exc_info:
        ArtifactContext(canonical_json="not json", sha256=_SHA)
    assert exc_info.value.code == "definition.json"

    with pytest.raises(DefinitionError) as exc_info:
        ArtifactContext(canonical_json="{}", sha256="not-hex")
    assert exc_info.value.code == "definition.lowercase_64_hex"

    with pytest.raises(DefinitionError) as exc_info:
        ArtifactContext(canonical_json='{"a": 1}', sha256=_SHA)
    assert exc_info.value.code == "definition.already_canonical_json"


def test_artifact_relation_ref_field_validators_carry_codes():
    artifact_id = uuid4()
    generation_id = uuid4()
    with pytest.raises(DefinitionError) as exc_info:
        ArtifactRelationRef(
            artifact_id=artifact_id,
            generation_id=generation_id,
            role="exposures",
            relation=RelationLocator(catalog="w", schema="a", name="exposures"),
            schema_sha256=_SHA,
            content_sha256=_SHA,
            row_count=0,
            primary_key=(),
        )
    assert exc_info.value.code == "definition.artifact_relation.primary_key_contain"

    with pytest.raises(DefinitionError) as exc_info:
        ArtifactRelationRef(
            artifact_id=artifact_id,
            generation_id=generation_id,
            role="exposures",
            relation=RelationLocator(catalog="w", schema="a", name="exposures"),
            schema_sha256=_SHA,
            content_sha256=_SHA,
            row_count=0,
            primary_key=("unit_id", "unit_id"),
        )
    assert exc_info.value.code == "definition.artifact_relation.primary_key_members"


def test_base_relations_field_validators_carry_codes():
    artifact_id = uuid4()
    generation_id = uuid4()
    exposures = _relation(artifact_id, generation_id, "exposures", ("experiment_id", "unit_id"))
    measure_stats = _relation(
        artifact_id,
        generation_id,
        "measure_stats",
        ("experiment_id", "unit_id", "ds", "measure_key"),
    )
    with pytest.raises(DefinitionError) as exc_info:
        BaseRelations(exposures=measure_stats, measure_stats=measure_stats)
    assert exc_info.value.code == "definition.base_relations.exposures_relation_use"

    with pytest.raises(DefinitionError) as exc_info:
        BaseRelations(exposures=exposures, measure_stats=exposures)
    assert exc_info.value.code == "definition.base_relations.measure_stats_relation"

    other_generation = _relation(
        artifact_id, uuid4(), "measure_stats", ("experiment_id", "unit_id", "ds", "measure_key")
    )
    with pytest.raises(DefinitionError) as exc_info:
        BaseRelations(exposures=exposures, measure_stats=other_generation)
    assert exc_info.value.code == "definition.base_relations.share_generation_id"

    other_artifact = _relation(
        uuid4(), generation_id, "measure_stats", ("experiment_id", "unit_id", "ds", "measure_key")
    )
    with pytest.raises(DefinitionError) as exc_info:
        BaseRelations(exposures=exposures, measure_stats=other_artifact)
    assert exc_info.value.code == "definition.base_relations.share_artifact_id"


def test_extension_request_field_validators_carry_codes():
    with pytest.raises(DefinitionError) as exc_info:
        AssignmentCountsRequest(populations=("assigned", "assigned"))
    assert exc_info.value.code == "definition.assignment_populations_contain"

    with pytest.raises(DefinitionError) as exc_info:
        SiteVolumeRequest(metric_names=())
    assert exc_info.value.code == "definition.site_volume.metric_names_empty"
