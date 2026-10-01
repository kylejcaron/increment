"""Pure format-1 canonical bytes and SHA-256 codec for unit-day artifacts."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import pickle
import shutil
import struct
import tempfile
import uuid
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz

from increment._canonical import CanonicalJSONError, _mapping, canonical_json_loads
from increment._canonical import canonical_json_bytes as _canonical_json_bytes
from increment.errors import (
    RefusalSpec,
    WireFormatError,
    _safe_error_value,
    raiser,
    refusals,
)

if TYPE_CHECKING:
    import ibis.expr.types as ir

_DIGEST_BATCH_ROWS = 4096

DOMAIN_ROOT = b"increment.unit-day-artifact\x00v1\x00"

# `digest_relation_sql_v2` (SQL text tokens) and `digest_relation_bucketed`
# (`canonical_row_bytes`) share this v2 domain but not format ids: a stored
# `digest_format`, never the connected backend, selects the verify codec.
# DOMAIN_ROOT stays fixed: manifest, context, request and extension hashes embed it.
_DOMAIN_ROOT_V2 = b"increment.unit-day-artifact\x00v2\x00"

#: `digest_relation_sql_v2`'s row encoding: SQL text tokens.
SQL_DIGEST_FORMAT = 2
#: `digest_relation_bucketed`'s row encoding: binary `canonical_row_bytes`.
BUCKETED_DIGEST_FORMAT = 3


class DigestType(IntEnum):
    NULL = 0
    STRING = 1
    INT64 = 2
    FLOAT64 = 3
    DATE = 4
    TIMESTAMP_UTC_US = 5
    BOOLEAN = 6


_PHYSICAL_TYPE_ROUTE = "declare the field type as one of " + ", ".join(
    tag.name for tag in DigestType if tag is not DigestType.NULL
)

_ALIASES = {
    "STRING": 1,
    "VARCHAR": 1,
    "TEXT": 1,
    "CHAR": 1,
    "UTF8": 1,
    "INT64": 2,
    "BIGINT": 2,
    "INTEGER": 2,
    "INT": 2,
    "FLOAT64": 3,
    "DOUBLE": 3,
    "REAL": 3,
    "DATE": 4,
    "TIMESTAMP_UTC_US": 5,
    "TIMESTAMPTZ": 5,
    "TIMESTAMP WITH TIME ZONE": 5,
    "BOOLEAN": 6,
    "BOOL": 6,
}


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    type_tag: DigestType
    nullable: bool = False

    @property
    def type(self) -> DigestType:
        return self.type_tag


@dataclass(frozen=True, slots=True)
class RelationDigests:
    role: str
    schema_sha256: str
    content_sha256: str
    row_count: int
    schema_bytes: bytes
    rows_bytes: bytes | None
    digest_format: int = 1

    @property
    def schema_digest(self) -> str:
        return self.schema_sha256

    @property
    def content_digest(self) -> str:
        return self.content_sha256

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "schema_sha256": self.schema_sha256,
            "content_sha256": self.content_sha256,
            "row_count": self.row_count,
        }


class ArtifactDigestError(WireFormatError):
    """Public coded refusal raised by canonical digest operations."""


_FIELD_REFUSAL = "artifact digest field {field!r}: {constraint}; got {value!r}; {route}"
REFUSALS: dict[str, RefusalSpec] = refusals(
    ArtifactDigestError,
    {
        "artifact.digest.schema": (
            "artifact digest schema declaration {position} (field {field!r}): "
            "{constraint}; got {value!r}; {route}"
        ),
        "artifact.digest.range": (
            "artifact digest {quantity} (field {field!r}) must lie in "
            "[{minimum}, {maximum}]; got {value!r}; {route}"
        ),
        "artifact.digest.row": _FIELD_REFUSAL,
        "artifact.digest.type": _FIELD_REFUSAL,
        "artifact.digest.timestamp": _FIELD_REFUSAL,
        "artifact.digest.nullable": _FIELD_REFUSAL,
        "artifact.digest.nonfinite": _FIELD_REFUSAL,
        "artifact.digest.primary_key": (
            "artifact digest primary key {primary_key!r} (field {field!r}): "
            "{constraint}; got {value!r}; {route}"
        ),
        "artifact.digest.rows": "artifact digest rows: {constraint}; got {value!r}; {route}",
        "artifact.digest.role": (
            "artifact digest relation role: {constraint}; got {value!r}; {route}"
        ),
        "artifact.digest.json": _FIELD_REFUSAL,
        "artifact.digest.recanonicalize": _FIELD_REFUSAL,
    },
)
_raise = raiser(REFUSALS)

# Quantities named by `artifact.digest.range` refusals.
_FIELD_COUNT = "schema field count"
_ROW_COUNT = "row count"
_CELL_BYTES = "cell payload byte length"

_CANONICAL_JSON_ROUTE = (
    "encode only canonical JSON values: str keys, finite numbers, integers within "
    "+/-(2**53 - 1), timezone-aware datetimes, and lists or tuples instead of sets"
)


def _field(value: object, position: int) -> FieldSpec:
    if isinstance(value, FieldSpec):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) not in (2, 3):
            _raise(
                "artifact.digest.schema",
                position=position,
                field=None,
                constraint="a field declaration sequence holds (name, type[, nullable])",
                value=len(value),
                route="declare the field as (name, type) or (name, type, nullable)",
            )
        name, typ, nullable = value[0], value[1], value[2] if len(value) == 3 else False
    else:
        mapping = _mapping(value)
        if mapping is not None:
            name, typ, nullable = (
                mapping.get("name"),
                mapping.get("type", mapping.get("type_tag", mapping.get("dtype"))),
                mapping.get("nullable", False),
            )
        else:
            name, typ, nullable = (
                getattr(value, "name", None),
                getattr(value, "type", getattr(value, "type_tag", None)),
                getattr(value, "nullable", False),
            )
    if not isinstance(name, str) or not name:
        _raise(
            "artifact.digest.schema",
            position=position,
            field=None,
            constraint="field names must be non-empty str",
            value=_safe_error_value(name),
            route="give the field declaration a non-empty str name",
        )
    if isinstance(typ, DigestType):
        tag = typ
    elif isinstance(typ, int) and typ in range(7):
        tag = DigestType(typ)
    elif isinstance(typ, str) and typ.strip().upper() in _ALIASES:
        tag = DigestType(_ALIASES[typ.strip().upper()])
    else:
        _raise(
            "artifact.digest.schema",
            position=position,
            field=name,
            constraint="field types must name a digest physical type",
            value=_safe_error_value(typ),
            route=_PHYSICAL_TYPE_ROUTE,
        )
    if tag is DigestType.NULL:
        _raise(
            "artifact.digest.schema",
            position=position,
            field=name,
            constraint="NULL marks a cell, not a field's physical type",
            value=_safe_error_value(typ),
            route=_PHYSICAL_TYPE_ROUTE,
        )
    if not isinstance(nullable, bool):
        _raise(
            "artifact.digest.schema",
            position=position,
            field=name,
            constraint="nullable must be a bool",
            value=_safe_error_value(nullable),
            route="declare nullable as True or False",
        )
    return FieldSpec(cast(str, name), tag, nullable)


def _fields(schema: object) -> tuple[FieldSpec, ...]:
    if isinstance(schema, Mapping):
        schema = schema.get("fields", tuple({"name": k, "type": v} for k, v in schema.items()))
    if not isinstance(schema, Sequence) or isinstance(schema, (str, bytes, bytearray)):
        _raise(
            "artifact.digest.schema",
            position=None,
            field=None,
            constraint="a schema is a sequence of field declarations or a name-to-type mapping",
            value=_safe_error_value(schema),
            route="pass the schema as a list of field declarations or a {name: type} mapping",
        )
    result = tuple(_field(item, position) for position, item in enumerate(schema))
    names = [x.name for x in result]
    if len(names) != len(set(names)):
        position, name = next(
            (position, name) for position, name in enumerate(names) if name in names[:position]
        )
        _raise(
            "artifact.digest.schema",
            position=position,
            field=name,
            constraint="field names must be unique within the schema",
            value=names.count(name),
            route="rename or remove the repeated field declaration",
        )
    return result


def _u32(value: int, field: str | None, quantity: str) -> bytes:
    if not 0 <= value < 2**32:
        _raise(
            "artifact.digest.range",
            quantity=quantity,
            field=field,
            minimum=0,
            maximum=2**32 - 1,
            value=_safe_error_value(value),
            route=f"supply a {quantity} that fits an unsigned 32-bit integer",
        )
    return struct.pack(">I", value)


def _u64(value: int, field: str | None, quantity: str) -> bytes:
    if not 0 <= value < 2**64:
        _raise(
            "artifact.digest.range",
            quantity=quantity,
            field=field,
            minimum=0,
            maximum=2**64 - 1,
            value=_safe_error_value(value),
            route=f"supply a {quantity} that fits an unsigned 64-bit integer",
        )
    return struct.pack(">Q", value)


def canonical_schema_bytes(schema: object) -> bytes:
    fields = _fields(schema)
    out = bytearray(_u32(len(fields), None, _FIELD_COUNT))
    for position, f in enumerate(fields):
        try:
            name = f.name.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            _raise(
                "artifact.digest.schema",
                position=position,
                field=f.name,
                constraint="field names must be UTF-8 encodable",
                value=f.name[exc.start : exc.end],
                route="remove unpaired surrogates from the field name",
            )
        out.extend(_u32(len(name), f.name, "field name byte length"))
        out.extend(name)
        out.extend((int(f.type_tag), 1 if f.nullable else 0))
    return bytes(out)


def _lookup(row: object, name: str) -> object:
    mapping = _mapping(row)
    if mapping is None:
        _raise(
            "artifact.digest.row",
            field=name,
            constraint="rows must be mappings or models",
            value=_safe_error_value(row),
            route="pass each row as a mapping or pydantic model keyed by field name",
        )
    if name not in mapping:
        _raise(
            "artifact.digest.row",
            field=name,
            constraint="rows must carry every declared field; value lists the row's fields",
            value=tuple(_safe_error_value(key) for key in mapping),
            route="add the declared field to every row or remove it from the schema",
        )
    return mapping[name]


def _date_days(value: object, field: str) -> int:
    if not isinstance(value, date) or isinstance(value, datetime):
        _raise(
            "artifact.digest.type",
            field=field,
            constraint="DATE cells need a datetime.date that is not a datetime",
            value=_safe_error_value(value),
            route="pass datetime.date values for DATE fields",
        )
    return (value - date(1970, 1, 1)).days


def _timestamp_us(value: object, field: str) -> int:
    if not isinstance(value, datetime):
        _raise(
            "artifact.digest.type",
            field=field,
            constraint="TIMESTAMP_UTC_US cells need a datetime.datetime",
            value=_safe_error_value(value),
            route="pass timezone-aware datetime.datetime values in UTC",
        )
    if (
        value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        offset = value.utcoffset()
        _raise(
            "artifact.digest.timestamp",
            field=field,
            constraint="TIMESTAMP_UTC_US cells need an explicit UTC offset of 0 seconds",
            value=None if offset is None else offset.total_seconds(),
            route="attach tzinfo=datetime.UTC or convert with .astimezone(datetime.UTC)",
        )
    if value.fold:
        _raise(
            "artifact.digest.timestamp",
            field=field,
            constraint="TIMESTAMP_UTC_US cells need fold=0",
            value=value.fold,
            route="construct the UTC datetime with fold=0",
        )
    d = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return d.days * 86400000000 + d.seconds * 1000000 + d.microseconds


def _cell(f: FieldSpec, value: object) -> tuple[bytes, object]:
    if value is None:
        if not f.nullable:
            _raise(
                "artifact.digest.nullable",
                field=f.name,
                constraint="non-nullable fields must hold a value",
                value=None,
                route="supply a value or declare the field nullable",
            )
        return bytes([0]) + _u64(0, f.name, _CELL_BYTES), None
    tag = f.type_tag
    if tag is DigestType.NULL:
        _raise(
            "artifact.digest.type",
            field=f.name,
            constraint="NULL-typed fields cannot carry a value",
            value=_safe_error_value(value),
            route=_PHYSICAL_TYPE_ROUTE,
        )
    if tag is DigestType.STRING:
        if not isinstance(value, str):
            _raise(
                "artifact.digest.type",
                field=f.name,
                constraint="STRING cells need a str",
                value=_safe_error_value(value),
                route="pass str values for STRING fields",
            )
        try:
            payload = value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            _raise(
                "artifact.digest.type",
                field=f.name,
                constraint="STRING cells must be UTF-8 encodable",
                value=value[exc.start : exc.end],
                route="remove unpaired surrogates from the text",
            )
        return bytes([tag]) + _u64(len(payload), f.name, _CELL_BYTES) + payload, payload
    if tag is DigestType.INT64:
        if isinstance(value, bool) or not isinstance(value, int) or not -(2**63) <= value < 2**63:
            _raise(
                "artifact.digest.type",
                field=f.name,
                constraint="INT64 cells need a non-bool int in [-2**63, 2**63 - 1]",
                value=_safe_error_value(value),
                route="pass int values within the signed 64-bit range",
            )
        payload = value.to_bytes(8, "big", signed=True)
        return bytes([tag]) + _u64(8, f.name, _CELL_BYTES) + payload, value
    if tag is DigestType.FLOAT64:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _raise(
                "artifact.digest.type",
                field=f.name,
                constraint="FLOAT64 cells need a non-bool int or float",
                value=_safe_error_value(value),
                route="pass int or float values for FLOAT64 fields",
            )
        try:
            number = float(value)
        except (OverflowError, ValueError):
            _raise(
                "artifact.digest.nonfinite",
                field=f.name,
                constraint="FLOAT64 cells must convert to a finite binary64 value",
                value=_safe_error_value(value),
                route="pass a value within binary64 range",
            )
        if not math.isfinite(number):
            _raise(
                "artifact.digest.nonfinite",
                field=f.name,
                constraint="FLOAT64 cells must be finite",
                value=number,
                route="replace NaN and infinities with finite values, or NULL in a nullable field",
            )
        payload = struct.pack(">d", 0.0 if number == 0.0 else number)
        return bytes([tag]) + _u64(8, f.name, _CELL_BYTES) + payload, number
    if tag is DigestType.DATE:
        days = _date_days(value, f.name)
        if not -(2**31) <= days < 2**31:
            _raise(
                "artifact.digest.range",
                quantity="DATE day offset from 1970-01-01",
                field=f.name,
                minimum=-(2**31),
                maximum=2**31 - 1,
                value=days,
                route="use a date whose day offset fits a signed 32-bit integer",
            )
        payload = days.to_bytes(4, "big", signed=True)
        return bytes([tag]) + _u64(4, f.name, _CELL_BYTES) + payload, days
    if tag is DigestType.TIMESTAMP_UTC_US:
        micros = _timestamp_us(value, f.name)
        if not -(2**63) <= micros < 2**63:
            _raise(
                "artifact.digest.range",
                quantity="TIMESTAMP_UTC_US microsecond offset from 1970-01-01T00:00:00Z",
                field=f.name,
                minimum=-(2**63),
                maximum=2**63 - 1,
                value=micros,
                route="use a timestamp whose microsecond offset fits a signed 64-bit integer",
            )
        payload = micros.to_bytes(8, "big", signed=True)
        return bytes([tag]) + _u64(8, f.name, _CELL_BYTES) + payload, micros
    if tag is DigestType.BOOLEAN:
        if not isinstance(value, bool):
            _raise(
                "artifact.digest.type",
                field=f.name,
                constraint="BOOLEAN cells need a bool",
                value=_safe_error_value(value),
                route="pass True or False for BOOLEAN fields",
            )
        payload = bytes([int(value)])
        return bytes([tag]) + _u64(1, f.name, _CELL_BYTES) + payload, value
    _raise(
        "artifact.digest.type",
        field=f.name,
        constraint="field type tags must be a digest physical type",
        value=_safe_error_value(tag),
        route=_PHYSICAL_TYPE_ROUTE,
    )


def canonical_row_bytes(schema: object, row: object) -> bytes:
    fields = _fields(schema)
    out = bytearray(_u32(len(fields), None, _FIELD_COUNT))
    for f in fields:
        out.extend(_cell(f, _lookup(row, f.name))[0])
    return bytes(out)


def _sort_key(
    fields: tuple[FieldSpec, ...], row: object, pk: tuple[str, ...]
) -> tuple[object, ...]:
    by_name = {f.name: f for f in fields}
    out = []
    for name in pk:
        if name not in by_name:
            _raise(
                "artifact.digest.primary_key",
                primary_key=pk,
                field=name,
                constraint="primary-key fields must be declared in the schema",
                value=name,
                route=f"choose primary-key fields from {tuple(by_name)!r}",
            )
        f = by_name[name]
        value = _lookup(row, name)
        if value is None:
            _raise(
                "artifact.digest.primary_key",
                primary_key=pk,
                field=name,
                constraint="primary-key fields must not be NULL",
                value=None,
                route="supply a value for every primary-key field",
            )
        out.append(_cell(f, value)[1])
    return tuple(out)


def _check_primary_key(fields: tuple[FieldSpec, ...], pk: tuple[str, ...]) -> None:
    """Refuse a primary key that repeats a field or names an undeclared one."""
    if len(pk) != len(set(pk)):
        name = next(name for position, name in enumerate(pk) if name in pk[:position])
        _raise(
            "artifact.digest.primary_key",
            primary_key=pk,
            field=name,
            constraint="primary-key fields must be distinct",
            value=pk.count(name),
            route="name each primary-key field once",
        )
    field_names = {f.name for f in fields}
    if any(name not in field_names for name in pk):
        name = next(name for name in pk if name not in field_names)
        _raise(
            "artifact.digest.primary_key",
            primary_key=pk,
            field=name,
            constraint="primary-key fields must be declared in the schema",
            value=name,
            route=f"choose primary-key fields from {tuple(f.name for f in fields)!r}",
        )


def canonical_rows_bytes(
    schema: object, rows: Sequence[object], *, primary_key: Sequence[str] = ()
) -> bytes:
    fields = _fields(schema)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _raise(
            "artifact.digest.rows",
            constraint="rows must be a non-string sequence",
            value=_safe_error_value(rows),
            route="pass the rows as a list or tuple",
        )
    pk = tuple(primary_key)
    _check_primary_key(fields, pk)
    if len(rows) > 1 and not pk:
        _raise(
            "artifact.digest.primary_key",
            primary_key=pk,
            field=None,
            constraint="multi-row relations must declare a primary key",
            value=len(rows),
            route="pass primary_key naming the relation's unique key fields",
        )
    decorated = [(_sort_key(fields, row, pk), row) for row in rows]
    decorated.sort(key=lambda x: x[0])
    out = bytearray()
    previous = None
    for key, row in decorated:
        if previous is not None and key == previous:
            _raise(
                "artifact.digest.primary_key",
                primary_key=pk,
                field=None,
                constraint="primary-key values must be unique",
                value=tuple(_safe_error_value(_lookup(row, name)) for name in pk),
                route="remove or merge the rows sharing this primary-key value",
            )
        previous = key
        out.extend(canonical_row_bytes(fields, row))
    return bytes(out)


def _role_bytes(role: str) -> bytes:
    if not isinstance(role, str) or not role:
        _raise(
            "artifact.digest.role",
            constraint="relation roles must be non-empty str",
            value=_safe_error_value(role),
            route="pass the relation's role name",
        )
    try:
        return role.encode("ascii", "strict")
    except UnicodeEncodeError:
        _raise(
            "artifact.digest.role",
            constraint="relation roles must be ASCII",
            value=_safe_error_value(role),
            route="use an ASCII relation role name",
        )


def _domain(role: str) -> bytes:
    return DOMAIN_ROOT + _role_bytes(role) + b"\x00"


def _domain_v2(role: str) -> bytes:
    return _DOMAIN_ROOT_V2 + _role_bytes(role) + b"\x00"


def schema_sha256(role: str, schema: object) -> str:
    return hashlib.sha256(
        _domain(role) + b"schema\x00" + canonical_schema_bytes(schema)
    ).hexdigest()


def content_sha256(
    role: str, schema: object, rows: Sequence[object], *, primary_key: Sequence[str] = ()
) -> str:
    fields = _fields(schema)
    schema_raw = canonical_schema_bytes(fields)
    rows_raw = canonical_rows_bytes(fields, rows, primary_key=primary_key)
    domain = _domain(role)
    return hashlib.sha256(
        domain
        + b"content\x00"
        + hashlib.sha256(domain + b"schema\x00" + schema_raw).digest()
        + _u64(len(rows), None, _ROW_COUNT)
        + rows_raw
    ).hexdigest()


def digest_relation(
    role: str, schema: object, rows: Sequence[object], *, primary_key: Sequence[str] = ()
) -> RelationDigests:
    fields = _fields(schema)
    schema_raw = canonical_schema_bytes(fields)
    rows_raw = canonical_rows_bytes(fields, rows, primary_key=primary_key)
    domain = _domain(role)
    sh = hashlib.sha256(domain + b"schema\x00" + schema_raw).hexdigest()
    ch = hashlib.sha256(
        domain + b"content\x00" + bytes.fromhex(sh) + _u64(len(rows), None, _ROW_COUNT) + rows_raw
    ).hexdigest()
    return RelationDigests(role, sh, ch, len(rows), schema_raw, rows_raw)


def stream_canonical_rows(
    table: ir.Table,
    schema: object,
    con: Any,
    *,
    primary_key: Sequence[str] = (),
    run_rows: int = _DIGEST_BATCH_ROWS,
    fan_in: int = 16,
) -> tuple[Generator[object, None, None], int, str]:
    """Externally sort by the format's key, independently of warehouse collation.

    Python row memory is O(run_rows + fan_in). The caller must close the
    returned iterator and remove its directory, including on digest failure.
    Preparation failures remove their own partial runs before propagating.
    """
    if run_rows < 1 or fan_in < 2:
        _raise(
            "artifact.digest.rows",
            constraint="run_rows must be at least 1 and fan_in at least 2",
            value={"run_rows": _safe_error_value(run_rows), "fan_in": _safe_error_value(fan_in)},
            route="pass run_rows >= 1 and fan_in >= 2",
        )
    fields = _fields(schema)
    pk = tuple(primary_key)
    _check_primary_key(fields, pk)

    def key(row: object) -> Any:
        return _sort_key(fields, row, pk)

    def read(path: str) -> Generator[object, None, None]:
        with open(path, "rb") as handle:
            while True:
                try:
                    row = pickle.load(handle)
                except EOFError:
                    return
                yield row

    def merge(paths: Sequence[str]) -> Generator[object, None, None]:
        with ExitStack() as stack:
            readers = [stack.enter_context(closing(read(path))) for path in paths]
            yield from heapq.merge(*readers, key=key)

    def write(path: str, rows: Iterable[object]) -> None:
        with open(path, "wb") as handle:
            pickler = pickle.Pickler(handle)
            for row in rows:
                pickler.dump(row)
                # Each row is independent; retaining the memo defeats streaming.
                pickler.clear_memo()

    tmpdir = tempfile.mkdtemp(prefix="incr_digest_")
    try:
        paths: list[str] = []
        row_count = 0
        batches = con.to_pyarrow_batches(table, chunk_size=run_rows)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"fetch_record_batch\(\) is deprecated.*",
                    category=DeprecationWarning,
                    module=r"ibis\.backends\.duckdb",
                )
                for batch in batches:
                    # Enforce the bound even if a backend ignores chunk_size.
                    for start in range(0, batch.num_rows, run_rows):
                        rows = batch.slice(start, run_rows).to_pylist()
                        rows.sort(key=key)
                        row_count += len(rows)
                        path = os.path.join(tmpdir, f"run_{len(paths)}.pkl")
                        write(path, rows)
                        paths.append(path)
                        del rows
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        round_idx = 0
        while len(paths) > fan_in:
            next_paths: list[str] = []
            for start in range(0, len(paths), fan_in):
                group = paths[start : start + fan_in]
                path = os.path.join(tmpdir, f"merge_{round_idx}_{start}.pkl")
                with closing(merge(group)) as merged:
                    write(path, merged)
                for old_path in group:
                    os.remove(old_path)
                next_paths.append(path)
            paths = next_paths
            round_idx += 1
        return merge(paths), row_count, tmpdir
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def digest_relation_stream(
    role: str,
    schema: object,
    rows: Iterable[object],
    *,
    primary_key: Sequence[str] = (),
    row_count: int,
) -> RelationDigests:
    """Hash canonically ordered rows without retaining their concatenated bytes.

    Use stream_canonical_rows to establish ordering before calling this function.
    The declared count seeds the unchanged format-1 content prefix.
    """
    fields = _fields(schema)
    pk = tuple(primary_key)
    _check_primary_key(fields, pk)
    if row_count > 1 and not pk:
        _raise(
            "artifact.digest.primary_key",
            primary_key=pk,
            field=None,
            constraint="multi-row relations must declare a primary key",
            value=_safe_error_value(row_count),
            route="pass primary_key naming the relation's unique key fields",
        )
    schema_raw = canonical_schema_bytes(fields)
    domain = _domain(role)
    sh = hashlib.sha256(domain + b"schema\x00" + schema_raw).hexdigest()
    running = hashlib.sha256(
        domain + b"content\x00" + bytes.fromhex(sh) + _u64(row_count, None, _ROW_COUNT)
    )
    previous_key = None
    seen = 0
    for row in rows:
        key = _sort_key(fields, row, pk)
        if previous_key is not None and key == previous_key:
            _raise(
                "artifact.digest.primary_key",
                primary_key=pk,
                field=None,
                constraint="primary-key values must be unique",
                value=tuple(_safe_error_value(_lookup(row, name)) for name in pk),
                route="remove or merge the rows sharing this primary-key value",
            )
        previous_key = key
        running.update(canonical_row_bytes(fields, row))
        seen += 1
    if seen != row_count:
        _raise(
            "artifact.digest.rows",
            constraint="the streamed row count must equal the declared row_count",
            value={"streamed": seen, "declared": _safe_error_value(row_count)},
            route="pass row_count equal to the number of rows the stream yields",
        )
    return RelationDigests(role, sh, running.hexdigest(), row_count, schema_raw, None)


def _iter_row_dicts(
    table: ir.Table, con: Any, run_rows: int = _DIGEST_BATCH_ROWS
) -> tuple[Iterator[dict[str, object]], int]:
    """Fetch `table`'s rows as an Arrow-batch-driven generator of dicts, with
    no external sort and no disk spill -- the same batch-fetch loop
    `stream_canonical_rows` uses, minus the run-file writes that loop needs
    for its external merge sort. Row count is queried once up front so the
    generator itself never needs to buffer the whole relation to count it.
    """
    row_count = int(con.to_pyarrow(table.count()).as_py())

    def _rows() -> Generator[dict[str, object], None, None]:
        batches = con.to_pyarrow_batches(table, chunk_size=run_rows)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"fetch_record_batch\(\) is deprecated.*",
                    category=DeprecationWarning,
                    module=r"ibis\.backends\.duckdb",
                )
                for batch in batches:
                    yield from batch.to_pylist()
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    return _rows(), row_count


_BUCKET_HEX_LEN = 3  # 4096 buckets: each bucket's group_concat stays well
# under every targeted backend's string-aggregate limit at real row counts
# (a 10M-row relation averages ~2,441 rows/bucket, ~156 KB concatenated).
# An ordered string aggregate buffers its whole input, so bucket digests are
# aggregated over bucket ranges holding about this many rows per query.
_DIGEST_ROWS_PER_PASS = 65_536


def _digest_passes(row_count: int) -> int:
    """Smallest power-of-two bucket-range count (a divisor of 4096) whose
    ranges hold at most `_DIGEST_ROWS_PER_PASS` rows on average."""
    passes = 1
    while passes < 16**_BUCKET_HEX_LEN and row_count > passes * _DIGEST_ROWS_PER_PASS:
        passes *= 2
    return passes


def _bucket_key(row_hash_hex: str) -> str:
    return row_hash_hex[:_BUCKET_HEX_LEN]


def digest_relation_bucketed(
    con: Any, table: ir.Table, role: str, schema: object, *, primary_key: Sequence[str] = ()
) -> RelationDigests:
    """Order-independent content digest: SHA-256 per row, bucketed by the
    row's own hash prefix, ordered-concat + hashed within each bucket,
    then hashed again across buckets in bucket order.

    Used only where the SQL path (`digest_relation_sql_v2`) is unavailable
    for the connected backend; stamps `BUCKETED_DIGEST_FORMAT`, not the SQL
    path's format id, since the two hash different row encodings.

    Duplicate-primary-key detection runs in SQL (`_assert_unique_primary_key_sql`,
    the same check `digest_relation_sql_v2` uses), so Python memory never holds
    one key per row. Row hashes are buffered for one fetch batch, then
    appended to one spill file per bucket prefix, one file open at a time.
    Python memory stays bounded by the batch size plus one bucket's rows,
    not the whole relation, and open file handles do not grow with it.
    """
    fields = _fields(schema)
    pk = tuple(primary_key)
    _check_primary_key(fields, pk)
    row_count = int(con.to_pyarrow(table.count()).as_py())
    if row_count > 1 and not pk:
        _raise(
            "artifact.digest.primary_key",
            primary_key=pk,
            field=None,
            constraint="multi-row relations must declare a primary key",
            value=row_count,
            route="pass primary_key naming the relation's unique key fields",
        )
    _assert_cells_valid_sql(con, table, fields, pk, role)
    _assert_unique_primary_key_sql(con, table, pk, role)
    schema_raw = canonical_schema_bytes(fields)
    sh = schema_sha256(role, schema)
    rows, streamed_row_count = _iter_row_dicts(table, con)
    tmpdir = tempfile.mkdtemp(prefix="incr_digest_bucket_")
    try:
        buckets: set[str] = set()
        pending: dict[str, list[str]] = {}
        pending_rows = 0
        seen = 0

        def _flush() -> None:
            # Append each bucket's buffered hashes and close at once, so at most
            # one spill file is open however many buckets the relation touches.
            for key, hashes in pending.items():
                path = os.path.join(tmpdir, f"bucket_{key}.txt")
                with open(path, "a", encoding="ascii") as handle:
                    handle.write("\n".join(hashes))
                    handle.write("\n")
            pending.clear()

        for row in rows:
            row_hash_hex = hashlib.sha256(canonical_row_bytes(fields, row)).hexdigest()
            bucket_key = _bucket_key(row_hash_hex)
            buckets.add(bucket_key)
            pending.setdefault(bucket_key, []).append(row_hash_hex)
            pending_rows += 1
            seen += 1
            if pending_rows >= _DIGEST_BATCH_ROWS:
                _flush()
                pending_rows = 0
        _flush()
        if seen != row_count or seen != streamed_row_count:
            _raise(
                "artifact.digest.rows",
                constraint="every counted row must be hashed exactly once",
                value={"hashed": seen, "counted": row_count, "recounted": streamed_row_count},
                route=f"digest {role!r} from a relation that does not change while it is read",
            )
        domain = _domain_v2(role)
        pairs = bytearray()
        for bucket_key in sorted(buckets):
            with open(
                os.path.join(tmpdir, f"bucket_{bucket_key}.txt"), encoding="ascii"
            ) as spilled:
                hashes = sorted(line.rstrip("\n") for line in spilled)
            bucket_digest = hashlib.sha256("".join(hashes).encode()).hexdigest()
            pairs.extend(bucket_key.encode())
            pairs.extend(b":")
            pairs.extend(bucket_digest.encode())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    ch = hashlib.sha256(
        domain
        + b"content\x00"
        + bytes.fromhex(sh)
        + _u64(row_count, None, _ROW_COUNT)
        + bytes(pairs)
    ).hexdigest()
    return RelationDigests(
        role, sh, ch, row_count, schema_raw, None, digest_format=BUCKETED_DIGEST_FORMAT
    )


# Backends whose SQL digest v2 path is live-verified. Other dialects compile in
# tests/test_unit_day_artifact_digest_v2_dialects.py but join only after a live
# float round-trip probe against a real connection.
_SQL_DIGEST_BACKENDS = frozenset({"duckdb"})

# Native SHA-256-returning-hex-text function per backend. Compile-verified
# for all four via ibis.to_sql(dialect=...) with no live connection; only
# duckdb is also live-executed (see tests/test_unit_day_artifact_digest_v2.py).
_SQL_DIGEST_SHA256: dict[str, Callable[[ir.StringValue], ir.StringValue]] = {}
# Same hash as 32 raw bytes, and its lowercase-hex rendering. Row hashes are
# held in binary (half the width of hex) while waiting to be aggregated.
_SQL_DIGEST_SHA256_BINARY: dict[str, Callable[[ir.StringValue], ir.BinaryValue]] = {}
_SQL_DIGEST_BINARY_HEX: dict[str, Callable[[ir.BinaryValue], ir.StringValue]] = {}


class _Sha2Binary(ops.Value):
    """Emit Snowflake's binary hash without SQL parser name normalization."""

    arg: ops.Value[dt.String]
    bits: ops.Value[dt.Integer]

    dtype = dt.binary
    shape = rlz.shape_like("arg")


def _compile_snowflake_sha2_binary(_compiler: object, _op: _Sha2Binary, *, arg: Any, bits: Any):
    import sqlglot.expressions as sge

    return sge.Anonymous(this="SHA2_BINARY", expressions=[arg, bits])


def _register_snowflake_sha2_binary() -> None:
    from ibis.backends.sql.compilers.snowflake import SnowflakeCompiler

    # ibis resolves a compiler method from the op's class name; there is no public registry.
    SnowflakeCompiler.visit__Sha2Binary = _compile_snowflake_sha2_binary  # ty: ignore[unresolved-attribute]


_register_snowflake_sha2_binary()


def _register_sha256_backends() -> None:
    # ibis.udf.scalar.builtin generates these dynamically; ty sees `(...) -> Value`
    # rather than the declared signature -- a stub-precision gap, not a real type error.
    @ibis.udf.scalar.builtin(name="sha256")
    def _duckdb_sha256(x: str) -> str: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="unhex")
    def _duckdb_unhex(x: str) -> bytes: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="hex")
    def _duckdb_hex(x: bytes) -> str: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="sha2")
    def _snowflake_sha2(x: str, bits: int) -> str: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="hex_encode")
    def _snowflake_hex_encode(x: bytes, case: int) -> str: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="to_hex")
    def _bigquery_to_hex(x: bytes) -> str: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="digest")
    def _postgres_digest(x: str, algo: str) -> bytes: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="encode")
    def _postgres_encode(x: bytes, fmt: str) -> str: ...  # ty: ignore[empty-body]

    _SQL_DIGEST_SHA256["duckdb"] = _duckdb_sha256  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_SHA256["snowflake"] = lambda col: _snowflake_sha2(col, 256)  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_SHA256["bigquery"] = lambda col: _bigquery_to_hex(  # ty: ignore[invalid-assignment]
        col.hashbytes("sha256")
    )
    _SQL_DIGEST_SHA256["postgres"] = lambda col: _postgres_encode(  # ty: ignore[invalid-assignment]
        _postgres_digest(col, "sha256"), "hex"
    )
    _SQL_DIGEST_SHA256_BINARY["duckdb"] = lambda col: _duckdb_unhex(_duckdb_sha256(col))  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_SHA256_BINARY["snowflake"] = lambda col: _Sha2Binary(
        cast("ops.Value[dt.String]", col.op()),
        cast("ops.Value[dt.Integer]", ibis.literal(256).op()),
    ).to_expr()
    _SQL_DIGEST_SHA256_BINARY["bigquery"] = lambda col: col.hashbytes("sha256")
    _SQL_DIGEST_SHA256_BINARY["postgres"] = lambda col: _postgres_digest(col, "sha256")  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_BINARY_HEX["duckdb"] = lambda col: cast("ir.StringValue", _duckdb_hex(col)).lower()
    _SQL_DIGEST_BINARY_HEX["snowflake"] = lambda col: _snowflake_hex_encode(col, 0)  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_BINARY_HEX["bigquery"] = lambda col: _bigquery_to_hex(col)  # ty: ignore[invalid-assignment]
    _SQL_DIGEST_BINARY_HEX["postgres"] = lambda col: _postgres_encode(col, "hex")  # ty: ignore[invalid-assignment]


_register_sha256_backends()

# High-precision FLOAT64-to-injective-string per backend. DuckDB's 17-digit printf
# showed no collisions over 20,001 sampled doubles; other backends use documented
# round-trip defaults and join _SQL_DIGEST_BACKENDS only after the same live probe.
_SQL_DIGEST_FLOAT_TOKEN: dict[str, Callable[[ir.FloatingValue], ir.StringValue]] = {}


def _register_float_token_backends() -> None:
    @ibis.udf.scalar.builtin(name="printf")
    def _duckdb_printf17g(fmt: str, x: float) -> str: ...  # ty: ignore[empty-body]

    _SQL_DIGEST_FLOAT_TOKEN["duckdb"] = lambda col: _duckdb_printf17g(  # ty: ignore[invalid-assignment]
        "%.17g", col
    )
    _SQL_DIGEST_FLOAT_TOKEN["snowflake"] = lambda col: col.cast("string")
    _SQL_DIGEST_FLOAT_TOKEN["bigquery"] = lambda col: col.cast("string")
    _SQL_DIGEST_FLOAT_TOKEN["postgres"] = lambda col: col.cast("string")


_register_float_token_backends()

# TIMESTAMP_UTC_US as integer epoch microseconds. A string cast renders in the
# session time zone, so the same stored instant would digest differently per
# connection. Only duckdb's is live-verified; the others are compile-pinned.
_SQL_DIGEST_TIMESTAMP_TOKEN: dict[str, Callable[[ir.TimestampValue], ir.StringValue]] = {}


def _register_timestamp_token_backends() -> None:
    @ibis.udf.scalar.builtin(name="epoch_us")
    def _duckdb_epoch_us(x: datetime) -> int: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="date_part")
    def _date_part(part: str, x: datetime) -> float: ...  # ty: ignore[empty-body]

    @ibis.udf.scalar.builtin(name="unix_micros")
    def _bigquery_unix_micros(x: datetime) -> int: ...  # ty: ignore[empty-body]

    _SQL_DIGEST_TIMESTAMP_TOKEN["duckdb"] = lambda col: _duckdb_epoch_us(col).cast("string")
    _SQL_DIGEST_TIMESTAMP_TOKEN["snowflake"] = lambda col: (
        _date_part("epoch_microsecond", col).cast("int64").cast("string")
    )
    _SQL_DIGEST_TIMESTAMP_TOKEN["bigquery"] = lambda col: _bigquery_unix_micros(col).cast("string")
    _SQL_DIGEST_TIMESTAMP_TOKEN["postgres"] = lambda col: (
        (cast("ir.FloatingValue", _date_part("epoch", col)) * 1_000_000)
        .round()
        .cast("int64")
        .cast("string")
    )


_register_timestamp_token_backends()


def _sql_digest_sha256_expr(col: ir.StringValue, *, dialect: str) -> ir.StringValue:
    """SHA-256-hex of `col` for `dialect`, via the backend-specific binding
    registered in `_SQL_DIGEST_SHA256`. `dialect` is explicit (not inferred
    from `col`) so the binding is compile-testable with no connection."""
    return _SQL_DIGEST_SHA256[dialect](col)


def _sql_digest_sha256_binary_expr(col: ir.StringValue, *, dialect: str) -> ir.BinaryValue:
    """SHA-256 of `col` as 32 raw bytes for `dialect`."""
    return _SQL_DIGEST_SHA256_BINARY[dialect](col)


def _sql_binary_hex_expr(col: ir.BinaryValue, *, dialect: str) -> ir.StringValue:
    """Lowercase hex of binary `col` for `dialect`: the same text the hex hash
    function returns, so a binary-held row hash renders back byte-identically."""
    return _SQL_DIGEST_BINARY_HEX[dialect](col)


def _row_token_expr(
    table: ir.Table, tagged_fields: Sequence[tuple[str, int, str]], *, dialect: str
) -> ir.StringValue:
    """One `tag(2 hex) + len(16-digit decimal, zero-padded) + payload` token
    per field, concatenated in schema order. `tagged_fields` is
    `(column_name, type_tag, kind)` where `kind` selects the payload
    encoding: `"float"` and `"timestamp"` use the backend's registered token
    (full-precision float text; epoch microseconds, independent of the
    session time zone), every other kind a plain string cast, which is exact
    and injective for STRING/INT64/DATE/BOOLEAN. `dialect` is explicit so
    the construction is compile-testable against an unbound table."""
    parts = []
    for name, tag, kind in tagged_fields:
        col = table[name]
        if kind == "float":
            payload = _SQL_DIGEST_FLOAT_TOKEN[dialect](col)
        elif kind == "timestamp":
            payload = _SQL_DIGEST_TIMESTAMP_TOKEN[dialect](col)
        else:
            payload = col.cast("string")
        length = payload.length().cast("string").lpad(16, "0")
        # ibis.literal's overload return type isn't resolved from a literal
        # `type="string"` kwarg by ty's stubs; the runtime value is a StringScalar.
        token = ibis.literal(f"{tag:02x}", type="string") + length + payload  # ty: ignore[unsupported-operator]
        if col.type().nullable:
            token = (col.isnull()).ifelse(ibis.literal("00" + "0" * 16, type="string"), token)
        parts.append(token)
    expr = parts[0]
    for part in parts[1:]:
        expr = expr + part
    return expr


def _assert_unique_primary_key_sql(con: Any, table: ir.Table, pk: Sequence[str], role: str) -> None:
    if not pk:
        return
    # A grouped count would need one hash entry per key, which DuckDB does not spill at
    # small memory limits. A sorted window holds one partition at a time; the second row
    # of a partition marks exactly one repeated key value.
    keys = [table[name] for name in pk]
    ranked = table.select(_key_rank=ibis.row_number().over(group_by=keys, order_by=keys))
    dup_count = int(con.to_pyarrow(ranked.filter(ranked["_key_rank"] == 1).count()).as_py())
    if dup_count:
        _raise(
            "artifact.digest.primary_key",
            primary_key=tuple(pk),
            field=None,
            constraint="primary-key values must be unique; value counts the repeated key values",
            value=dup_count,
            route=f"remove or merge the duplicate primary-key rows of {role!r}",
        )


def _count_where(conditions: Sequence[ir.BooleanValue]) -> ir.NumericScalar | None:
    if not conditions:
        return None
    first, *rest = conditions
    for condition in rest:
        first = first | condition
    # Column predicates are boolean columns; SUM counts the matching rows.
    return cast("ir.BooleanColumn", first).sum()


def _assert_cells_valid_sql(
    con: Any, table: ir.Table, fields: Sequence[FieldSpec], pk: Sequence[str], role: str
) -> None:
    """The SQL row token never routes through `_cell`, so apply its per-cell
    refusals here, in one query and in `_cell`'s precedence: NULL primary key,
    then NULL in a non-nullable field, then non-finite FLOAT64."""
    checks = {
        "pk_nulls": _count_where([table[name].isnull() for name in pk]),
        "nulls": _count_where([table[f.name].isnull() for f in fields if not f.nullable]),
        "nonfinite": _count_where(
            [
                table[f.name].isnan() | table[f.name].isinf()
                for f in fields
                if f.type_tag is DigestType.FLOAT64
            ]
        ),
    }
    metrics = [expr.name(name) for name, expr in checks.items() if expr is not None]
    if not metrics:
        return
    counts = con.to_pyarrow(table.aggregate(metrics)).to_pylist()[0]
    if counts.get("pk_nulls"):
        _raise(
            "artifact.digest.primary_key",
            primary_key=tuple(pk),
            field=None,
            constraint="primary-key fields must not be NULL; value counts the offending rows",
            value=int(counts["pk_nulls"]),
            route=f"supply every primary-key field in {role!r}",
        )
    if counts.get("nulls"):
        _raise(
            "artifact.digest.nullable",
            field=tuple(f.name for f in fields if not f.nullable),
            constraint="non-nullable fields must hold a value; value counts the offending rows",
            value=int(counts["nulls"]),
            route=f"fill the NULL cells of {role!r} or declare those fields nullable",
        )
    if counts.get("nonfinite"):
        _raise(
            "artifact.digest.nonfinite",
            field=tuple(f.name for f in fields if f.type_tag is DigestType.FLOAT64),
            constraint="FLOAT64 cells must be finite; value counts the offending rows",
            value=int(counts["nonfinite"]),
            route=f"replace NaN and infinities in {role!r} with finite values or NULL",
        )


def digest_relation_sql_v2(
    con: Any, table: ir.Table, role: str, schema: object, *, primary_key: Sequence[str] = ()
) -> RelationDigests:
    """SQL fast path (gated to `_SQL_DIGEST_BACKENDS`). Row hashes are
    materialized once, as 32 raw bytes, into a temporary table; bucket digests
    are then aggregated over their lowercase hex, one bucket range at a time.
    An ordered string aggregate buffers its whole input, so ranges are sized
    to ~`_DIGEST_ROWS_PER_PASS` rows without hashing any row twice. All of it
    stays inside the backend's own memory manager (which may spill), never
    the Python heap: only the populated buckets' (bucket_index, bucket_digest) pairs --
    at most 4096 rows -- cross the client boundary. Raises
    artifact.digest.primary_key before hashing if `table` has a NULL or
    duplicate primary key value."""
    if con.name not in _SQL_DIGEST_BACKENDS:
        _raise(
            "artifact.digest.rows",
            constraint="SQL digest v2 runs only on live-verified backends",
            value={"backend": con.name, "enabled": tuple(sorted(_SQL_DIGEST_BACKENDS))},
            route="digest this backend's relation with digest_relation_bucketed",
        )
    fields = _fields(schema)
    pk = tuple(primary_key)
    _check_primary_key(fields, pk)
    if not pk:
        counted = int(con.to_pyarrow(table.count()).as_py())
        if counted > 1:
            _raise(
                "artifact.digest.primary_key",
                primary_key=pk,
                field=None,
                constraint="multi-row relations must declare a primary key",
                value=counted,
                route="pass primary_key naming the relation's unique key fields",
            )
    _assert_cells_valid_sql(con, table, fields, pk, role)
    _assert_unique_primary_key_sql(con, table, pk, role)
    schema_raw = canonical_schema_bytes(fields)
    sh = schema_sha256(role, schema)
    kinds = {DigestType.FLOAT64: "float", DigestType.TIMESTAMP_UTC_US: "timestamp"}
    tagged_fields = [(f.name, int(f.type_tag), kinds.get(f.type_tag, "other")) for f in fields]
    row_token = _row_token_expr(table, tagged_fields, dialect=con.name)

    def sha256_hex(col: ir.StringValue) -> ir.StringValue:
        return _sql_digest_sha256_expr(col, dialect=con.name)

    row_hashes = table.select(h=_sql_digest_sha256_binary_expr(row_token, dialect=con.name))
    hashed_name = f"incr_digest_{uuid.uuid4().hex}"
    hashed = con.create_table(hashed_name, row_hashes, temp=True)
    try:
        hex_hashes = hashed.select(h=_sql_binary_hex_expr(hashed.h, dialect=con.name))
        buckets = hex_hashes.select(bucket=hex_hashes.h.substr(0, _BUCKET_HEX_LEN), h=hex_hashes.h)
        row_count = int(con.to_pyarrow(hashed.count()).as_py())
        rows: list[dict[str, Any]] = []
        step = 16**_BUCKET_HEX_LEN // _digest_passes(row_count)
        for start in range(0, 16**_BUCKET_HEX_LEN, step):
            part = buckets.filter(buckets.bucket >= f"{start:0{_BUCKET_HEX_LEN}x}")
            if start + step < 16**_BUCKET_HEX_LEN:
                part = part.filter(part.bucket < f"{start + step:0{_BUCKET_HEX_LEN}x}")
            bucket_digests = (
                part.group_by("bucket")
                .aggregate(
                    bucket_digest=sha256_hex(part.h.group_concat(sep="", order_by=part.h)),
                    n=part.count(),
                )
                .order_by("bucket")
            )
            rows.extend(con.to_pyarrow(bucket_digests).to_pylist())
    finally:
        con.drop_table(hashed_name, force=True)
    bucketed = sum(record["n"] for record in rows)
    if bucketed != row_count:
        _raise(
            "artifact.digest.rows",
            constraint="every hashed row must land in exactly one bucket",
            value={"bucketed": bucketed, "hashed": row_count},
            route="digest this backend's relation with digest_relation_bucketed",
        )
    domain = _domain_v2(role)
    pairs = bytearray()
    for record in rows:
        pairs.extend(record["bucket"].encode())
        pairs.extend(b":")
        pairs.extend(record["bucket_digest"].encode())
    ch = hashlib.sha256(
        domain
        + b"content\x00"
        + bytes.fromhex(sh)
        + _u64(row_count, None, _ROW_COUNT)
        + bytes(pairs)
    ).hexdigest()
    return RelationDigests(
        role, sh, ch, row_count, schema_raw, None, digest_format=SQL_DIGEST_FORMAT
    )


def canonical_json_bytes(value: object) -> bytes:
    try:
        return _canonical_json_bytes(value)
    except CanonicalJSONError as exc:
        # The encoder reports its violated rule and code, not the offending path.
        _raise(
            exc.code,
            field=None,
            constraint=str(exc),
            value=_safe_error_value(value),
            route=_CANONICAL_JSON_ROUTE,
        )


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode()


def _without(value: object, key: str) -> object:
    mapping = _mapping(value)
    if mapping is None:
        _raise(
            "artifact.digest.json",
            field=None,
            constraint="hashed payloads must be mappings or models",
            value=_safe_error_value(value),
            route="pass the payload as a mapping or pydantic model",
        )
    return {name: item for name, item in mapping.items() if name != key}


def manifest_sha256(manifest: object) -> str:
    return hashlib.sha256(
        DOMAIN_ROOT + b"manifest\x00" + canonical_json_bytes(_without(manifest, "manifest_sha256"))
    ).hexdigest()


def context_sha256(context: object) -> str:
    mapping = _mapping(context)
    if mapping is not None and isinstance(mapping.get("canonical_json"), str):
        payload = cast(str, mapping["canonical_json"]).encode()
    else:
        payload = canonical_json_bytes(_without(context, "sha256"))
    return hashlib.sha256(DOMAIN_ROOT + b"context\x00" + payload).hexdigest()


def request_sha256(request: object) -> str:
    return hashlib.sha256(DOMAIN_ROOT + b"request\x00" + canonical_json_bytes(request)).hexdigest()


def _extension_hash(kind: str, marker: bytes, payload: str) -> str:
    try:
        parsed = canonical_json_loads(payload)
    except CanonicalJSONError as exc:
        _raise(
            exc.code,
            field=None,
            constraint=str(exc),
            value=_safe_error_value(payload),
            route=f"pass canonical_json(...) of the {kind!r} extension payload",
        )
    except (TypeError, json.JSONDecodeError) as exc:
        _raise(
            "artifact.digest.recanonicalize",
            field=None,
            constraint=str(exc),
            value=_safe_error_value(payload),
            route=f"pass canonical_json(...) of the {kind!r} extension payload",
        )
    if canonical_json_bytes(parsed).decode() != payload:
        _raise(
            "artifact.digest.recanonicalize",
            field=None,
            constraint="extension JSON must equal its canonical re-encoding",
            value=_safe_error_value(payload),
            route=f"pass canonical_json(...) of the {kind!r} extension payload",
        )
    return hashlib.sha256(
        DOMAIN_ROOT + marker + kind.encode() + b"\x00" + payload.encode()
    ).hexdigest()


def extension_definition_sha256(kind: str, canonical_definition_json: str) -> str:
    return _extension_hash(kind, b"extension-definition\x00", canonical_definition_json)


def extension_source_provenance_sha256(kind: str, canonical_source_recipe_json: str) -> str:
    return _extension_hash(kind, b"extension-source\x00", canonical_source_recipe_json)


canonical_schema = canonical_schema_bytes
canonical_row = canonical_row_bytes
canonical_rows = canonical_rows_bytes
compute_schema_sha256 = schema_sha256
compute_content_sha256 = content_sha256
compute_relation_digests = digest_relation
hash_manifest = manifest_sha256
hash_context = context_sha256
__all__ = [
    "DOMAIN_ROOT",
    "BUCKETED_DIGEST_FORMAT",
    "SQL_DIGEST_FORMAT",
    "ArtifactDigestError",
    "REFUSALS",
    "DigestType",
    "FieldSpec",
    "RelationDigests",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_row",
    "canonical_row_bytes",
    "canonical_rows",
    "canonical_rows_bytes",
    "canonical_schema",
    "canonical_schema_bytes",
    "compute_content_sha256",
    "compute_relation_digests",
    "compute_schema_sha256",
    "content_sha256",
    "context_sha256",
    "digest_relation",
    "digest_relation_bucketed",
    "digest_relation_sql_v2",
    "digest_relation_stream",
    "extension_definition_sha256",
    "extension_source_provenance_sha256",
    "hash_context",
    "hash_manifest",
    "manifest_sha256",
    "request_sha256",
    "schema_sha256",
    "stream_canonical_rows",
]
