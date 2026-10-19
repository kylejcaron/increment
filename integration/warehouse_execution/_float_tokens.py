"""Native format-2 token parity on pinned binary64 inputs."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from uuid import uuid4

import ibis
import numpy as np
import pyarrow as pa

from increment.query.artifact_digest import _float_payload_expr


def float_token_cases(*, random_count: int = 500) -> list[dict]:
    fixture = json.loads(
        (Path(__file__).parents[2] / "tests/fixtures/artifact_float17g.json").read_text()
    )
    cases = fixture["cases"]
    if random_count > 500:
        bits = np.random.default_rng(fixture["seed"]).bit_generator.random_raw(random_count)
        cases = [*cases, *({"bits": f"{int(b):016x}"} for b in bits[500:])]
    return cases


def run_float_token_probe(con, *, dialect: str, random_count: int = 500) -> None:
    cases = float_token_cases(random_count=random_count)
    values = [struct.unpack(">d", bytes.fromhex(case["bits"]))[0] for case in cases]
    arrow = pa.table({"id": range(len(cases)), "value": pa.array(values, type=pa.float64())})
    duckdb = ibis.duckdb.connect()
    name = f"float_token_spec_{uuid4().hex[:10]}"
    created = False
    try:
        oracle = duckdb.create_table(name, arrow)
        expected = duckdb.to_pyarrow(
            oracle.select(id=oracle.id, token=_float_payload_expr(oracle.value, dialect="duckdb"))
        ).to_pylist()
        expected.sort(key=lambda row: row["id"])
        for case, row in zip(cases, expected, strict=True):
            if "token" in case:
                assert row["token"] == case["token"], case["bits"]
        table = con.create_table(name, arrow)
        created = True
        actual = con.to_pyarrow(
            table.select(id=table.id, token=_float_payload_expr(table.value, dialect=dialect))
        ).to_pylist()
        actual.sort(key=lambda row: row["id"])
        assert len(actual) == len(expected)
        for case, got, want in zip(cases, actual, expected, strict=True):
            assert got == want, case["bits"]
    finally:
        duckdb.disconnect()
        if created:
            con.drop_table(name, force=True)
