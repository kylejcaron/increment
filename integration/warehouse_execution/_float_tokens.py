"""Native format-4 token parity on pinned binary64 inputs."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa

from increment.query._artifact_float import (
    float64_token_py as _float64_token_py,
)
from increment.query._artifact_float import (
    float64_token_sql as _float64_token_sql,
)
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
    source_expected = [
        {"id": index, "token": _float64_token_py(value)} for index, value in enumerate(values)
    ]
    name = f"float_token_spec_{uuid4().hex[:10]}"
    destination: str | None = None
    created = False
    try:
        if dialect == "bigquery":
            from io import BytesIO

            import pyarrow.parquet as pq
            from google.cloud import bigquery

            parquet = BytesIO()
            pq.write_table(arrow, parquet)
            parquet.seek(0)
            destination = f"{con.current_catalog}.{con.current_database}.{name}"
            job = con.client.load_table_from_file(
                parquet,
                destination,
                size=parquet.getbuffer().nbytes,
                job_config=bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET),
            )
            created = True
            job.result()
            table = con.sql(
                f"SELECT CAST(`id` AS INT64) AS `id`, `value`, "
                f"IEEE_DIVIDE(1.0, `value`) AS `reciprocal` FROM `{destination}`"
            )
            actual_rows = con.to_pyarrow(
                table.select(
                    id=table.id,
                    value=table.value,
                    reciprocal=table.reciprocal,
                    token=_float_payload_expr(table.value, dialect=dialect, digest_format=4),
                )
            ).to_pylist()
            expected = []
            negative_zero = "8000000000000000"
            for row in actual_rows:
                index = row["id"]
                if cases[index]["bits"] == negative_zero:
                    if (
                        row["value"] != 0.0
                        or not math.isinf(row["reciprocal"])
                        or row["reciprocal"] < 0
                    ):
                        raise AssertionError(
                            "BigQuery Parquet load did not normalize -0 to stored +0: "
                            f"value={row['value']!r}, reciprocal={row['reciprocal']!r}"
                        )
                elif _float64_token_py(row["value"]) != source_expected[index]["token"]:
                    raise AssertionError(
                        "BigQuery Parquet load changed a non-zero FLOAT64 fixture value: "
                        f"id={index}, source={values[index]!r}, stored={row['value']!r}"
                    )
                expected.append({"id": index, "token": _float64_token_py(row["value"])})
            literal_sql = _float64_token_sql("CAST('-0' AS FLOAT64)", dialect="bigquery")
            literal_token = (
                con.raw_sql(f"SELECT {literal_sql} AS token")
                .to_arrow(create_bqstorage_client=False, progress_bar_type=None)
                .to_pylist()
            )
            assert literal_token == [{"token": "-0"}]
            actual = [{"id": row["id"], "token": row["token"]} for row in actual_rows]
        else:
            table = con.create_table(name, arrow)
            created = True
            actual = con.to_pyarrow(
                table.select(
                    id=table.id,
                    token=_float_payload_expr(table.value, dialect=dialect, digest_format=4),
                )
            ).to_pylist()
            expected = source_expected
        expected.sort(key=lambda row: row["id"])
        actual.sort(key=lambda row: row["id"])
        if len(actual) != len(expected):
            raise AssertionError(
                f"{dialect} FLOAT64 token row-count mismatch: "
                f"expected={len(expected)}, actual={len(actual)}"
            )
        for wanted, observed in zip(expected, actual, strict=True):
            if wanted != observed:
                index = wanted["id"]
                stored_value = (
                    next(row["value"] for row in actual_rows if row["id"] == index)
                    if dialect == "bigquery"
                    else values[index]
                )
                raise AssertionError(
                    f"{dialect} FLOAT64 token mismatch: id={index}, "
                    f"source_value={values[index]!r}, stored_value={stored_value!r}, "
                    f"bits={cases[index]['bits']}, expected={wanted['token']!r}, "
                    f"actual={observed['token']!r}"
                )
    finally:
        if created:
            con.drop_table(name, force=True)
