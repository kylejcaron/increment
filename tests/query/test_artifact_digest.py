"""Streaming format-1 digest regressions."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import ibis
import pytest

from increment.query import artifact_digest as codec


@pytest.mark.slow
def test_streaming_digest_multiround_parity():
    schema = (
        ("id", "STRING", False),
        ("value", "FLOAT64", True),
        ("timestamp", "TIMESTAMP_UTC_US", False),
        ("day", "DATE", False),
    )
    rows = [
        {
            "id": key,
            "value": None if i == 2 else float(i),
            "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
            "day": date(2025, 1, 1),
        }
        for i, key in enumerate(["é", "z", "A", "😀", "a", "Z", "中", "aa", "\x00"])
    ]
    expected = codec.digest_relation("measure_stats", schema, rows, primary_key=("id",))
    con = ibis.duckdb.connect()
    try:
        stream, count, directory = codec.stream_canonical_rows(
            ibis.memtable(rows), schema, con, primary_key=("id",), run_rows=2, fan_in=2
        )
        try:
            actual = codec.digest_relation_stream(
                "measure_stats", schema, stream, primary_key=("id",), row_count=count
            )
            assert actual.content_sha256 == expected.content_sha256
            assert actual.schema_sha256 == expected.schema_sha256
            assert actual.row_count == len(rows)
            assert actual.rows_bytes is None
            assert expected.rows_bytes
        finally:
            stream.close()
            shutil.rmtree(directory)
    finally:
        con.disconnect()


@pytest.mark.parametrize("pk,count", [(("id", "id"), 0), (("missing",), 0), ((), 2)])
def test_streaming_digest_validates_declaration_before_consuming(pk, count):
    def unreadable():
        raise AssertionError("invalid declaration consumed rows")
        yield

    with pytest.raises(codec.ArtifactDigestError) as error:
        codec.digest_relation_stream(
            "exposures",
            (("id", "STRING", False),),
            unreadable(),
            primary_key=pk,
            row_count=count,
        )
    assert error.value.code == "artifact.digest.primary_key"


@pytest.mark.parametrize(
    "rows,count,code",
    [
        ([{"id": "a"}, {"id": "a"}], 2, "artifact.digest.primary_key"),
        ([{"id": "a"}], 2, "artifact.digest.rows"),
        ([{"id": "a"}, {"id": "b"}], 1, "artifact.digest.rows"),
        ([{"id": None}], 1, "artifact.digest.primary_key"),
    ],
)
def test_streaming_digest_rejects_invalid_rows(rows, count, code):
    with pytest.raises(codec.ArtifactDigestError) as error:
        codec.digest_relation_stream(
            "exposures",
            (("id", "STRING", False),),
            iter(rows),
            primary_key=("id",),
            row_count=count,
        )
    assert error.value.code == code


def test_streaming_digest_empty_parity():
    schema = (("id", "STRING", False),)
    old = codec.digest_relation("exposures", schema, [], primary_key=("id",))
    new = codec.digest_relation_stream(
        "exposures", schema, iter(()), primary_key=("id",), row_count=0
    )
    assert old.as_dict() == new.as_dict()


def test_streaming_sort_cleans_partial_runs_on_fetch_failure(tmp_path, monkeypatch):
    import tempfile

    import pyarrow as pa

    original = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: original(dir=tmp_path, **kw))

    class BrokenConnection:
        def to_pyarrow_batches(self, table, *, chunk_size):
            yield pa.record_batch({"id": ["b", "a"]})
            raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError):
        codec.stream_canonical_rows(
            ibis.table({"id": "string"}),
            (("id", "STRING", False),),
            BrokenConnection(),
            primary_key=("id",),
        )
    assert list(Path(tmp_path).iterdir()) == []


@pytest.mark.slow
@pytest.mark.parametrize("duplicate", [False, True])
def test_streaming_sort_empty_and_cross_run_duplicates(duplicate):
    schema = (("id", "STRING", False),)
    rows = [{"id": value} for value in ["z", "b", "a", "c", "b"]] if duplicate else []
    con = ibis.duckdb.connect()
    ibis_schema = ibis.schema({"id": "string"})
    table = ibis.memtable(
        rows if rows else ibis_schema.to_pyarrow().empty_table(), schema=ibis_schema
    )
    try:
        stream, count, directory = codec.stream_canonical_rows(
            table, schema, con, primary_key=("id",), run_rows=1, fan_in=2
        )
        try:
            if duplicate:
                with pytest.raises(codec.ArtifactDigestError) as error:
                    codec.digest_relation_stream(
                        "exposures", schema, stream, primary_key=("id",), row_count=count
                    )
                assert error.value.code == "artifact.digest.primary_key"
            else:
                assert count == 0
                assert list(stream) == []
        finally:
            stream.close()
            shutil.rmtree(directory)
    finally:
        con.disconnect()


def test_streaming_sort_cleans_runs_on_merge_failure(tmp_path, monkeypatch):
    import heapq
    import tempfile

    import pyarrow as pa

    original = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: original(dir=tmp_path, **kw))

    class Connection:
        def to_pyarrow_batches(self, table, *, chunk_size):
            for key in "abcde":
                yield pa.record_batch({"id": [key]})

    def broken(*args, **kwargs):
        raise OSError("merge failed")

    monkeypatch.setattr(heapq, "merge", broken)
    with pytest.raises(OSError):
        codec.stream_canonical_rows(
            ibis.table({"id": "string"}),
            (("id", "STRING", False),),
            Connection(),
            primary_key=("id",),
            run_rows=1,
            fan_in=2,
        )
    assert list(tmp_path.iterdir()) == []
