"""Normative format-1 unit-day artifact digest vectors and refusals."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from increment.query.artifact_digest import (
    DOMAIN_ROOT,
    ArtifactDigestError,
    canonical_json_bytes,
    canonical_row_bytes,
    canonical_rows_bytes,
    canonical_schema_bytes,
    content_sha256,
    context_sha256,
    digest_relation,
    digest_relation_stream,
    extension_definition_sha256,
    manifest_sha256,
    schema_sha256,
)
from increment.query.schemas import (
    UNIT_DAY_ARTIFACT_PRIMARY_KEYS,
    UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
)

FIXTURE = Path(__file__).parent / "fixtures" / "unit_day_artifact_digest_v1.json"


def _vectors() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["vectors"]


def _rows_for_codec(vector: dict[str, Any]) -> list[dict[str, Any]]:
    type_by_name = {field["name"]: field["type"] for field in vector["fields"]}
    rows = []
    for source in vector["rows"]:
        row = dict(source)
        for name, type_name in type_by_name.items():
            if row[name] is None:
                continue
            if type_name == "DATE":
                row[name] = date.fromisoformat(row[name])
            elif type_name == "TIMESTAMP_UTC_US":
                row[name] = datetime.fromisoformat(row[name].replace("Z", "+00:00"))
        rows.append(row)
    return rows


def test_literal_external_vectors_match_codec() -> None:
    """Expected bytes and hashes are loaded literally, never generated here."""
    for vector in _vectors():
        fields = vector["fields"]
        role = vector["role"]
        from increment.query.schemas import LEGACY_ENCOURAGEMENT_UPTAKE_SCHEMA

        schema = (
            LEGACY_ENCOURAGEMENT_UPTAKE_SCHEMA
            if vector["name"] == "encouragement_uptake_boundaries"
            else UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role]
        )
        expected_fields = [
            {"name": name, "type": type_tag, "nullable": nullable}
            for name, type_tag, nullable in schema
        ]
        assert fields == expected_fields
        assert vector["primary_key"] == list(UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role])
        rows = _rows_for_codec(vector)
        primary_key = vector["primary_key"]
        expected_schema = bytes.fromhex(vector["schema_bytes_hex"])
        expected_rows = bytes.fromhex(vector["rows_bytes_hex"])
        assert canonical_schema_bytes(fields) == expected_schema
        assert canonical_rows_bytes(fields, rows, primary_key=primary_key) == expected_rows
        result = digest_relation(role, fields, rows, primary_key=primary_key)
        assert result.schema_sha256 == vector["schema_sha256"]
        assert result.content_sha256 == vector["content_sha256"]
        assert schema_sha256(role, fields) == vector["schema_sha256"]
        assert (
            content_sha256(role, fields, rows, primary_key=primary_key) == vector["content_sha256"]
        )


def test_external_refusal_vectors_cover_boundary_rejections() -> None:
    refusals = json.loads(FIXTURE.read_text())["refusals"]
    assert {entry["name"] for entry in refusals} >= {
        "nonfinite_float",
        "null_schema_type",
        "timestamp_without_utc",
    }
    for entry in refusals:
        name = entry["name"]
        expected_code = entry["code"]
        with pytest.raises(ArtifactDigestError) as raised:
            if name == "nonfinite_float":
                canonical_json_bytes({"value": float(entry["input"])})
            elif name == "null_schema_type":
                canonical_schema_bytes([{"name": "x", "type": entry["input"], "nullable": True}])
            else:
                canonical_row_bytes(
                    [{"name": "ts", "type": "TIMESTAMP_UTC_US", "nullable": False}],
                    {"ts": datetime.fromisoformat(entry["input"])},
                )
        assert raised.value.code == expected_code


def test_literal_unicode_nullable_and_int64_vectors() -> None:
    special = json.loads(FIXTURE.read_text())["special_vectors"]
    assert {entry["name"] for entry in special} >= {"unicode_delimiter_strings", "int64_boundaries"}
    for vector in special:
        rows = _rows_for_codec(vector)
        assert canonical_schema_bytes(vector["fields"]) == bytes.fromhex(vector["schema_bytes_hex"])
        assert canonical_rows_bytes(
            vector["fields"], rows, primary_key=vector["primary_key"]
        ) == bytes.fromhex(vector["rows_bytes_hex"])
        assert schema_sha256(vector["role"], vector["fields"]) == vector["schema_sha256"]
        assert (
            content_sha256(
                vector["role"], vector["fields"], rows, primary_key=vector["primary_key"]
            )
            == vector["content_sha256"]
        )


def test_literal_manifest_context_and_generation_vectors() -> None:
    vectors = json.loads(FIXTURE.read_text())["manifest_vectors"]
    first, second = vectors[:2]
    assert manifest_sha256(first["body"]) == first["manifest_sha256"]
    assert manifest_sha256(second["body"]) == second["manifest_sha256"]
    assert first["manifest_sha256"] != second["manifest_sha256"]
    context = vectors[2]
    assert context_sha256(context["body"]) == context["context_sha256"]


def test_canonical_ordering_and_negative_zero() -> None:
    fields = [
        {"name": "k", "type": "STRING", "nullable": False},
        {"name": "value", "type": "FLOAT64", "nullable": False},
    ]
    rows = [{"k": "z", "value": -0.0}, {"k": "a", "value": 1.5}]
    reordered = list(reversed(rows))
    assert canonical_rows_bytes(fields, rows, primary_key=("k",)) == canonical_rows_bytes(
        fields, reordered, primary_key=("k",)
    )
    assert canonical_row_bytes(fields, {"k": "z", "value": -0.0}) == canonical_row_bytes(
        fields, {"k": "z", "value": 0.0}
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_refuse(value: float) -> None:
    fields = [{"name": "value", "type": "FLOAT64", "nullable": False}]
    with pytest.raises(ArtifactDigestError) as row_raised:
        canonical_row_bytes(fields, {"value": value})
    assert row_raised.value.code == "artifact.digest.nonfinite"
    with pytest.raises(ArtifactDigestError) as json_raised:
        canonical_json_bytes({"value": value})
    assert json_raised.value.code == "artifact.digest.nonfinite"


def test_public_canonical_json_bytes_preserves_digest_refusal() -> None:
    from increment import canonical_json_bytes as public_canonical_json_bytes

    with pytest.raises(ArtifactDigestError) as raised:
        public_canonical_json_bytes(object())

    assert raised.value.code == "artifact.digest.json"


def test_null_and_key_tamper_refuse() -> None:
    fields = [
        {"name": "id", "type": "STRING", "nullable": False},
        {"name": "value", "type": "INT64", "nullable": True},
    ]
    with pytest.raises(ArtifactDigestError) as null_key:
        canonical_rows_bytes(fields, [{"id": None, "value": 1}], primary_key=("id",))
    assert null_key.value.code == "artifact.digest.primary_key"
    with pytest.raises(ArtifactDigestError) as duplicate_key:
        canonical_rows_bytes(
            fields,
            [{"id": "a", "value": 1}, {"id": "a", "value": None}],
            primary_key=("id",),
        )
    assert duplicate_key.value.code == "artifact.digest.primary_key"
    with pytest.raises(ArtifactDigestError) as nullable:
        canonical_row_bytes([{"name": "id", "type": "STRING", "nullable": False}], {"id": None})
    assert nullable.value.code == "artifact.digest.nullable"


def test_schema_and_row_tamper_changes_digest() -> None:
    fields = [{"name": "id", "type": "STRING", "nullable": False}]
    rows = [{"id": "stable"}]
    original = digest_relation("exposures", fields, rows, primary_key=("id",))
    changed = digest_relation("exposures", fields, [{"id": "tampered"}], primary_key=("id",))
    assert original.content_sha256 != changed.content_sha256
    changed_schema = [{"name": "id", "type": "STRING", "nullable": True}]
    assert schema_sha256("exposures", fields) != schema_sha256("exposures", changed_schema)


def test_canonical_json_recanonicalization_and_manifest_self_exclusion() -> None:
    payload = {"z": 1, "a": "é", "negative_zero": -0.0}
    canonical = canonical_json_bytes(payload)
    assert canonical == b'{"a":"\xc3\xa9","negative_zero":0,"z":1}'
    assert manifest_sha256({**payload, "manifest_sha256": "ignored"}) == manifest_sha256(payload)
    assert canonical_json_bytes(json.loads(canonical)) == canonical


@pytest.mark.parametrize("value", [1e10, 1e20, 1e21, -1e20])
def test_extension_hash_roundtrips_finite_float_boundaries(value: float) -> None:
    payload = canonical_json_bytes({"value": value}).decode()
    expected = hashlib.sha256(
        DOMAIN_ROOT + b"extension-definition\x00" + b"cuped_preperiod\x00" + payload.encode()
    ).hexdigest()
    assert extension_definition_sha256("cuped_preperiod", payload) == expected


def test_extension_hash_rejects_unsafe_integer_and_duplicate_keys() -> None:
    with pytest.raises(ArtifactDigestError) as unsafe:
        extension_definition_sha256("cuped_preperiod", '{"value":9007199254740993}')
    assert unsafe.value.code == "artifact.digest.json"
    with pytest.raises(ArtifactDigestError) as duplicate:
        extension_definition_sha256("cuped_preperiod", '{"value":1,"value":1}')
    assert duplicate.value.code == "artifact.digest.recanonicalize"


def test_refusal_contexts_carry_the_rejected_field_and_value_and_survive_pickle() -> None:
    fields = [
        {"name": "id", "type": "STRING", "nullable": False},
        {"name": "value", "type": "FLOAT64", "nullable": True},
    ]
    refusals = {}
    with pytest.raises(ArtifactDigestError) as raised:
        canonical_schema_bytes([fields[0], {"name": "value", "type": "VARCHAR2"}])
    refusals["schema"] = raised.value
    with pytest.raises(ArtifactDigestError) as raised:
        canonical_row_bytes(fields, {"id": "a", "value": math.inf})
    refusals["cell"] = raised.value
    with pytest.raises(ArtifactDigestError) as raised:
        canonical_rows_bytes(
            fields, [{"id": "a", "value": 1.0}, {"id": "a", "value": 2.0}], primary_key=("id",)
        )
    refusals["key"] = raised.value
    with pytest.raises(ArtifactDigestError) as raised:
        rows = (row for row in [{"id": "a", "value": None}])
        canonical_rows_bytes(fields, rows, primary_key=("id",))  # ty: ignore[invalid-argument-type]
    refusals["rows"] = raised.value

    schema, cell, key = (refusals[name].context for name in ("schema", "cell", "key"))
    assert (schema["position"], schema["field"], schema["value"]) == (1, "value", "VARCHAR2")
    assert (cell["field"], cell["value"]) == ("value", math.inf)
    assert (key["primary_key"], key["value"]) == (("id",), ("a",))
    assert refusals["rows"].code == "artifact.digest.rows"
    for error in refusals.values():
        restored = pickle.loads(pickle.dumps(error))
        assert (restored.code, restored.context) == (error.code, error.context)


def test_stream_row_count_outside_u64_reports_the_range_bounds() -> None:
    with pytest.raises(ArtifactDigestError) as raised:
        digest_relation_stream("exposures", (("id", "STRING", False),), iter(()), row_count=-1)
    context = raised.value.context
    assert raised.value.code == "artifact.digest.range"
    assert (context["quantity"], context["value"]) == ("row count", -1)
    assert (context["minimum"], context["maximum"]) == (0, 2**64 - 1)


def test_temporal_boundaries_and_utc_requirement() -> None:
    fields = [
        {"name": "d", "type": "DATE", "nullable": False},
        {"name": "ts", "type": "TIMESTAMP_UTC_US", "nullable": False},
    ]
    row = {"d": date(1970, 1, 1), "ts": datetime(1970, 1, 1, tzinfo=UTC)}
    assert canonical_row_bytes(fields, row)
    with pytest.raises(ArtifactDigestError) as raised:
        canonical_row_bytes(fields, {"d": date.today(), "ts": datetime(1970, 1, 1)})
    assert raised.value.code == "artifact.digest.timestamp"
