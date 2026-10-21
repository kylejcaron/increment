"""Unit-day artifact content digest formats 2, 3, and 4 use an order-independent
hash tree (SHA-256 per row, bucketed by row hash, ordered-concat within each
bucket, hashed again across buckets in bucket order). SQL format 4 is used for
all enabled backends; SQL format 2 verifies existing DuckDB relations; format 3
uses one client-side pass with a different row encoding."""

from __future__ import annotations

import time
from datetime import date

import ibis
import pytest

from increment.query.artifact_digest import (
    digest_relation_bucketed,
    digest_relation_sql_v2,
)
from tests._unit_boundary import backend_fixture

pytestmark = pytest.mark.integration

_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("ds", "DATE", False),
    ("treated", "BOOLEAN", False),
)
_PK = ("experiment_id", "unit_id", "ds")

_ROWS = [
    {"experiment_id": "exp1", "unit_id": "u1", "ds": date(2024, 1, 1), "treated": True},
    {"experiment_id": "exp1", "unit_id": "u2", "ds": date(2024, 1, 2), "treated": False},
    {"experiment_id": "exp1", "unit_id": "u3", "ds": date(2024, 1, 3), "treated": True},
]


@backend_fixture()
def con():
    return ibis.duckdb.connect()


def _table(con, rows):
    schema = ibis.schema({name: type_.lower() for name, type_, _n in _SCHEMA})
    return con.create_table(f"t_{id(rows)}", ibis.memtable(rows, schema=schema), overwrite=True)


def test_sql_digest_is_order_independent(con):
    forward = digest_relation_sql_v2(con, _table(con, _ROWS), "exposures", _SCHEMA, primary_key=_PK)
    backward = digest_relation_sql_v2(
        con, _table(con, list(reversed(_ROWS))), "exposures", _SCHEMA, primary_key=_PK
    )
    assert forward.content_sha256 == backward.content_sha256
    assert forward.row_count == backward.row_count == 3
    assert forward.digest_format == 4


_PORTABLE_SCHEMA = (
    ("id", "INT64", False),
    ("text", "STRING", True),
    ("integer", "INT64", True),
    ("float", "FLOAT64", True),
    ("date", "DATE", True),
    ("timestamp", "TIMESTAMP_UTC_US", True),
    ("boolean", "BOOLEAN", True),
)


def _portable_digest_rows():
    import sys
    from datetime import UTC, datetime

    return [
        (
            0,
            "",
            -(2**63),
            -0.0,
            date(1, 1, 1),
            datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
            False,
        ),
        (
            1,
            "多言語/🧪",
            2**63 - 1,
            0.0,
            date(9999, 12, 31),
            datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=UTC),
            True,
        ),
        (2, None, -(2**53), sys.float_info.min, None, None, None),
        (
            3,
            "e\u0301",
            2**53,
            sys.float_info.max,
            date(2000, 2, 29),
            datetime(2000, 1, 1, 0, 0, 0, 999999, tzinfo=UTC),
            False,
        ),
        (
            4,
            "🧪",
            None,
            -sys.float_info.max,
            date(1900, 1, 1),
            datetime(1900, 1, 1, 0, 0, 0, 1, tzinfo=UTC),
            True,
        ),
        (
            5,
            "empty-is-not-null",
            1,
            float.fromhex("0x0.0000000000001p-1022"),
            date(1969, 12, 31),
            datetime(1969, 12, 31, 23, 59, 59, 1, tzinfo=UTC),
            False,
        ),
        (6, "NaN", 0, float("nan"), date(1970, 1, 1), datetime(1970, 1, 1, tzinfo=UTC), True),
        (
            7,
            "positive infinity",
            -1,
            float("inf"),
            None,
            datetime(2024, 2, 29, 23, 59, 59, 999999, tzinfo=UTC),
            False,
        ),
        (8, "negative infinity", 1, float("-inf"), date(2024, 1, 1), None, True),
        (9, None, None, None, None, None, None),
    ]


def _portable_digest_arrow():
    import pyarrow as pa

    rows = _portable_digest_rows()
    types = (
        pa.int64(),
        pa.string(),
        pa.int64(),
        pa.float64(),
        pa.date32(),
        pa.timestamp("us", tz="UTC"),
        pa.bool_(),
    )
    return pa.Table.from_arrays(
        [
            pa.array([row[index] for row in rows], type=field_type)
            for index, field_type in enumerate(types)
        ],
        names=[name for name, _type, _nullable in _PORTABLE_SCHEMA],
    )


def test_sql_digest_format2_duckdb_verify_codec_stays_pinned(con):
    """Existing DuckDB relations retain the format-2 decimal-float verifier."""
    table = con.create_table("portable_digest_v2_spec", _portable_digest_arrow())
    digest = digest_relation_sql_v2(
        con, table, "portable_spec", _PORTABLE_SCHEMA, primary_key=("id",), digest_format=2
    )
    assert digest.digest_format == 2
    assert (
        digest.content_sha256 == "389d14c938d31a916b43a48610a969d16305e5219c8eb725c02e8bbcfa81f259"
    )


def test_sql_digest_format4_portable_edge_relation_is_pinned(con):
    table = con.create_table("portable_digest_v4_spec", _portable_digest_arrow())
    digest = digest_relation_sql_v2(
        con, table, "portable_spec", _PORTABLE_SCHEMA, primary_key=("id",)
    )
    assert digest.digest_format == 4
    assert (
        digest.content_sha256 == "28b8c5135e6da8dcddeca6854a0bb9cc86a205eee4f6b3e26795589124e669e5"
    )


def test_sql_digest_format4_positive_zero_content_is_pinned(con):
    import struct

    import pyarrow as pa

    arrow = _portable_digest_arrow()
    float_index = arrow.schema.get_field_index("float")
    values = arrow.column(float_index).to_pylist()
    values[0] = 0.0
    assert struct.pack(">d", values[0]) == bytes.fromhex("0000000000000000")
    stored_arrow = arrow.set_column(float_index, "float", pa.array(values, type=pa.float64()))
    table = con.create_table("portable_digest_v4_positive_zero", stored_arrow)
    digest = digest_relation_sql_v2(
        con, table, "portable_spec", _PORTABLE_SCHEMA, primary_key=("id",)
    )
    assert (
        digest.content_sha256 == "1df82b505bc1c38e07f7e492d7d9fc8e130c1331151fd6b5066827b32481632b"
    )


def test_sql_digest_detects_row_mutation(con):
    baseline = digest_relation_sql_v2(
        con, _table(con, _ROWS), "exposures", _SCHEMA, primary_key=_PK
    )
    mutated = [dict(_ROWS[0], treated=False)] + _ROWS[1:]
    changed = digest_relation_sql_v2(
        con, _table(con, mutated), "exposures", _SCHEMA, primary_key=_PK
    )
    assert baseline.content_sha256 != changed.content_sha256
    assert baseline.schema_sha256 == changed.schema_sha256


def test_sql_digest_detects_a_copied_row_via_row_count_and_digest(con):
    """A row copied under a new key changes row_count, which is hashed, and
    adds a row hash, so the digest cannot match the original relation."""
    baseline = digest_relation_sql_v2(
        con, _table(con, _ROWS), "exposures", _SCHEMA, primary_key=_PK
    )
    copied = _table(con, _ROWS + [dict(_ROWS[0], unit_id="u4")])
    grown = digest_relation_sql_v2(con, copied, "exposures", _SCHEMA, primary_key=_PK)
    assert grown.row_count == 4
    assert grown.content_sha256 != baseline.content_sha256


def test_both_format_2_paths_require_a_primary_key_for_multi_row_relations(con):
    from increment.query.artifact_digest import ArtifactDigestError

    attempts = (
        lambda: digest_relation_sql_v2(con, _table(con, _ROWS), "exposures", _SCHEMA),
        lambda: digest_relation_bucketed(con, _table(con, _ROWS), "exposures", _SCHEMA),
    )
    for attempt in attempts:
        with pytest.raises(ArtifactDigestError) as raised:
            attempt()
        assert raised.value.code == "artifact.digest.primary_key"


def test_sql_digest_refuses_duplicate_primary_key(con):
    from increment.query.artifact_digest import ArtifactDigestError

    dup = _ROWS + [dict(_ROWS[0])]
    with pytest.raises(ArtifactDigestError) as raised:
        digest_relation_sql_v2(con, _table(con, dup), "exposures", _SCHEMA, primary_key=_PK)
    assert raised.value.code == "artifact.digest.primary_key"


def test_python_fallback_is_order_independent_and_matches_row_count(con):
    forward = digest_relation_bucketed(
        con, _table(con, _ROWS), "exposures", _SCHEMA, primary_key=_PK
    )
    backward = digest_relation_bucketed(
        con, _table(con, list(reversed(_ROWS))), "exposures", _SCHEMA, primary_key=_PK
    )
    assert forward.content_sha256 == backward.content_sha256
    assert forward.row_count == backward.row_count == 3
    assert forward.digest_format == 3


def test_python_fallback_detects_row_mutation(con):
    baseline = digest_relation_bucketed(
        con, _table(con, _ROWS), "exposures", _SCHEMA, primary_key=_PK
    )
    mutated = [dict(_ROWS[0], treated=False)] + _ROWS[1:]
    changed = digest_relation_bucketed(
        con, _table(con, mutated), "exposures", _SCHEMA, primary_key=_PK
    )
    assert baseline.content_sha256 != changed.content_sha256


_STATS_SCHEMA = (
    ("unit_id", "STRING", False),
    ("value", "FLOAT64", False),
    ("note", "STRING", True),
)


@pytest.mark.parametrize(
    ("bad_row", "code"),
    [
        ({"unit_id": None, "value": 1.0, "note": None}, "artifact.digest.primary_key"),
        ({"unit_id": "u2", "value": None, "note": "x"}, "artifact.digest.nullable"),
    ],
)
def test_every_digest_path_refuses_the_same_bad_cell_with_the_same_code(con, bad_row, code):
    from increment.query.artifact_digest import ArtifactDigestError, content_sha256

    rows = [{"unit_id": "u1", "value": 0.5, "note": None}, bad_row]
    schema = ibis.schema({"unit_id": "string", "value": "float64", "note": "string"})
    table = con.create_table("stats", ibis.memtable(rows, schema=schema), overwrite=True)
    attempts = {
        "sql": lambda: digest_relation_sql_v2(
            con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
        ),
        "fallback": lambda: digest_relation_bucketed(
            con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
        ),
        "format_1": lambda: content_sha256(
            "measure_stats", _STATS_SCHEMA, rows, primary_key=("unit_id",)
        ),
    }
    for path, attempt in attempts.items():
        with pytest.raises(ArtifactDigestError) as raised:
            attempt()
        assert raised.value.code == code, path


def test_null_and_empty_string_in_a_nullable_field_digest_differently(con):
    schema = ibis.schema({"unit_id": "string", "value": "float64", "note": "string"})

    def digests(note):
        rows = [{"unit_id": "u1", "value": 0.5, "note": note}]
        table = con.create_table("notes", ibis.memtable(rows, schema=schema), overwrite=True)
        sql = digest_relation_sql_v2(
            con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
        )
        fallback = digest_relation_bucketed(
            con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
        )
        return sql.content_sha256, fallback.content_sha256

    null_sql, null_fallback = digests(None)
    empty_sql, empty_fallback = digests("")
    assert null_sql != empty_sql
    assert null_fallback != empty_fallback


_TS_SCHEMA = (("unit_id", "STRING", False), ("ts", "TIMESTAMP_UTC_US", False))


def test_sql_digest_is_independent_of_the_session_time_zone():
    from datetime import UTC, datetime

    rows = [
        {"unit_id": "u1", "ts": datetime(1969, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)},
        {"unit_id": "u2", "ts": datetime(2024, 3, 10, 7, 30, tzinfo=UTC)},
    ]
    digests = set()
    for zone in ("UTC", "America/New_York", "Australia/Adelaide"):
        con = ibis.duckdb.connect()
        con.raw_sql(f"SET TimeZone='{zone}'")
        table = con.create_table(
            "ts_rows",
            ibis.memtable(rows, schema={"unit_id": "string", "ts": "timestamp('UTC')"}),
        )
        digests.add(
            digest_relation_sql_v2(
                con, table, "exposures", _TS_SCHEMA, primary_key=("unit_id",)
            ).content_sha256
        )
    assert len(digests) == 1


@pytest.mark.slow
def test_sql_digest_does_not_depend_on_how_many_bucket_passes_run(con, monkeypatch):
    from increment.query import artifact_digest

    con.raw_sql(
        """create table many as select 'exp1' as experiment_id, 'u' || i as unit_id,
        (DATE '2024-01-01' + CAST(i % 9 AS INTEGER)) as ds, (i % 3 = 0) as treated
        from range(3000) t(i)"""
    )
    digests = set()
    # 1, 16 and 256 bucket ranges for 3,000 rows.
    for rows_per_pass in (10**9, 188, 12):
        monkeypatch.setattr(artifact_digest, "_DIGEST_ROWS_PER_PASS", rows_per_pass)
        digest = digest_relation_sql_v2(
            con, con.table("many"), "exposures", _SCHEMA, primary_key=_PK
        )
        assert digest.row_count == 3000
        digests.add(digest.content_sha256)
    assert len(digests) == 1


def test_binary_held_row_hash_renders_back_to_the_hex_hash(con):
    from increment.query.artifact_digest import (
        _sql_binary_hex_expr,
        _sql_digest_sha256_binary_expr,
        _sql_digest_sha256_expr,
    )

    table = con.create_table(
        "tokens", ibis.memtable({"x": ["", "a", "é", "0" * 200]}), overwrite=True
    )
    binary = _sql_digest_sha256_binary_expr(table.x, dialect="duckdb")
    result = con.to_pyarrow(
        table.select(
            via_binary=_sql_binary_hex_expr(binary, dialect="duckdb"),
            direct=_sql_digest_sha256_expr(table.x, dialect="duckdb"),
        )
    ).to_pylist()
    assert all(row["via_binary"] == row["direct"] for row in result)


@pytest.mark.slow
def test_sql_digest_throughput_floor(con):
    """Regression guard against falling back to streaming rows through Python,
    which measured 20,739 rows/s: the floor sits at that old rate so a loaded
    worker cannot flake it, while the SQL path measures ~1M rows/s here."""
    con.raw_sql(
        """
        create table bench as
        select
          'exp1' as experiment_id,
          'unit_' || i as unit_id,
          (DATE '2024-01-01' + CAST(i % 90 AS INTEGER)) as ds,
          (i % 2 = 0) as treated
        from range(200000) t(i)
        """
    )
    table = con.table("bench")
    start = time.perf_counter()
    digest = digest_relation_sql_v2(con, table, "exposures", _SCHEMA, primary_key=_PK)
    elapsed = time.perf_counter() - start
    assert digest.row_count == 200_000
    assert elapsed > 0
    assert 200_000 / elapsed >= 20_000


@pytest.mark.slow
def test_bucketed_digest_runs_under_a_low_open_file_limit(con):
    """Thousands of buckets must not need thousands of open files at once."""
    resource = pytest.importorskip("resource")
    n = 5000
    rows = [{"unit_id": f"u{i}", "value": float(i), "note": None} for i in range(n)]
    schema = ibis.schema({"unit_id": "string", "value": "float64", "note": "string"})
    table = con.create_table("many", ibis.memtable(rows, schema=schema), overwrite=True)
    unlimited = digest_relation_bucketed(
        con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 96), hard))
    try:
        limited = digest_relation_bucketed(
            con, table, "measure_stats", _STATS_SCHEMA, primary_key=("unit_id",)
        )
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert limited.content_sha256 == unlimited.content_sha256
    assert limited.row_count == n
