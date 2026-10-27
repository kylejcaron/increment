from __future__ import annotations

from typing import Any, cast

import pytest

from increment.errors import CodedError
from increment.kernel.schema import CatalogSchemaCatalog, load_schema_catalog, write_schema_catalog


def test_declared_and_persisted_schemas_resolve_without_mutable_aliases(tmp_path):
    source = {"facts": {"region": "string", "amount": "float"}}
    catalog = CatalogSchemaCatalog(properties=source)
    source["facts"]["region"] = "int"
    columns = catalog.columns("facts")
    assert columns == {"amount": "float", "region": "string"}
    with pytest.raises(TypeError):
        cast(Any, columns)["region"] = "int"

    persisted = CatalogSchemaCatalog(
        properties={},
        persisted={"bound": {"unit_id": "string", "y": "float"}},
        definitions_digest="d1",
    )
    path = tmp_path / "catalog.json"
    write_schema_catalog(path, persisted)
    loaded = load_schema_catalog(path, definitions_digest="d1")
    assert loaded.columns("bound") == {"unit_id": "string", "y": "float"}


def test_property_dtypes_and_name_only_capabilities_resolve_offline():
    from increment import EvidenceCapabilities
    from increment.semantics.models import Property

    declarations = {
        "events": {
            "country": Property(name="country", column="country", dtype="string"),
            "revenue": Property(name="revenue", column="revenue", dtype="float"),
        }
    }
    catalog = CatalogSchemaCatalog(properties=declarations)
    capabilities = EvidenceCapabilities(
        kind="definitions",
        grains=frozenset(),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset(),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset({"country", "revenue"}),
    )

    assert capabilities.available_covariates == frozenset(catalog.columns("events"))
    assert catalog.columns("events") == {"country": "string", "revenue": "float"}


def test_bound_frame_schema_and_known_empty_source_are_distinct():
    catalog = CatalogSchemaCatalog(
        properties={"empty": {}},
        persisted={"frame": {"unit_id": "string", "value": "float"}},
    )

    assert catalog.columns("frame") == {"unit_id": "string", "value": "float"}
    assert catalog.columns("empty") == {}
    with pytest.raises(CodedError) as missing:
        catalog.columns("unbound")
    assert missing.value.code == "kernel.request.invalid"


def test_missing_source_refuses_and_changed_definition_is_stale(tmp_path):
    path = tmp_path / "catalog.json"
    write_schema_catalog(
        path,
        CatalogSchemaCatalog(
            properties={}, persisted={"source": {"x": "int"}}, definitions_digest="old"
        ),
    )
    loaded = load_schema_catalog(path, definitions_digest="new")
    assert loaded.columns("source") == {}
    assert loaded.stale_reason("source") == "kernel.schema.stale"
    mismatched = load_schema_catalog(path, source_refs={"source": "changed"})
    assert mismatched.columns("source") == {}
    assert mismatched.stale_reason("source") == "kernel.schema.stale"
    with pytest.raises(CodedError) as missing:
        loaded.columns("missing")
    assert missing.value.code == "kernel.request.invalid"
