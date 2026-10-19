"""Native format-4 token parity on pinned binary64 inputs."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa

from increment.query._artifact_float import float64_token_py as _float64_token_py
from increment.query.artifact_digest import _float_payload_expr


def float_token_cases(*, random_count: int = 500) -> list[dict]:
    fixture = json.loads(
        (Path(__file__).parents[2] / "tests/fixtures/artifact_float64_cases.json").read_text()
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
    expected = [
        {"id": index, "token": _float64_token_py(value)} for index, value in enumerate(values)
    ]
    name = f"float_token_spec_{uuid4().hex[:10]}"
    created = False
    try:
        table = con.create_table(name, arrow)
        created = True
        actual = con.to_pyarrow(
            table.select(
                id=table.id,
                token=_float_payload_expr(table.value, dialect=dialect, digest_format=4),
            )
        ).to_pylist()
        actual.sort(key=lambda row: row["id"])
        assert actual == expected
    finally:
        if created:
            con.drop_table(name, force=True)
