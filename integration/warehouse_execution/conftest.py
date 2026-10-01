"""Shared fixtures for real-backend warehouse execution tests.

``_require_env`` FAILS the test (never skips) when required env is
absent: a dedicated release-gate execution of these tests must never
read an unconfigured backend as anything other than a hard failure --
a skip would let the release gate silently pass without ever having
executed the check it exists to make.
"""

from __future__ import annotations

import os

import pytest


def _require_env(*names: str) -> dict[str, str]:
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.fail(
            f"missing required env var(s) for live backend test: {missing} -- "
            "this is a hard failure, not a skip: a dedicated warehouse-backend "
            "execution must never silently pass without running"
        )
    return {name: value for name, value in values.items() if value is not None}


@pytest.fixture
def postgres_connect():
    import ibis

    env = _require_env(
        "INCREMENT_TEST_POSTGRES_HOST",
        "INCREMENT_TEST_POSTGRES_PORT",
        "INCREMENT_TEST_POSTGRES_USER",
        "INCREMENT_TEST_POSTGRES_PASSWORD",
        "INCREMENT_TEST_POSTGRES_DB",
    )

    def connect():
        return ibis.postgres.connect(
            host=env["INCREMENT_TEST_POSTGRES_HOST"],
            port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
            user=env["INCREMENT_TEST_POSTGRES_USER"],
            password=env["INCREMENT_TEST_POSTGRES_PASSWORD"],
            database=env["INCREMENT_TEST_POSTGRES_DB"],
        )

    return connect


@pytest.fixture
def postgres_con(postgres_connect):
    con = postgres_connect()
    try:
        yield con
    finally:
        con.disconnect()


@pytest.fixture
def snowflake_con():
    import ibis

    env = _require_env(
        "INCREMENT_TEST_SNOWFLAKE_ACCOUNT",
        "INCREMENT_TEST_SNOWFLAKE_USER",
        "INCREMENT_TEST_SNOWFLAKE_PRIVATE_KEY",
        "INCREMENT_TEST_SNOWFLAKE_WAREHOUSE",
        "INCREMENT_TEST_SNOWFLAKE_DATABASE",
        "INCREMENT_TEST_SNOWFLAKE_SCHEMA",
    )
    # A Snowflake service user holds no password, so the connector takes a DER
    # private key whose public half is registered on that user.
    from cryptography.hazmat.primitives import serialization

    passphrase = os.environ.get("INCREMENT_TEST_SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    parsed = serialization.load_pem_private_key(
        env["INCREMENT_TEST_SNOWFLAKE_PRIVATE_KEY"].encode(),
        password=passphrase.encode() if passphrase else None,
    )
    con = ibis.snowflake.connect(
        account=env["INCREMENT_TEST_SNOWFLAKE_ACCOUNT"],
        user=env["INCREMENT_TEST_SNOWFLAKE_USER"],
        private_key=parsed.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        warehouse=env["INCREMENT_TEST_SNOWFLAKE_WAREHOUSE"],
        database=env["INCREMENT_TEST_SNOWFLAKE_DATABASE"],
        schema=env["INCREMENT_TEST_SNOWFLAKE_SCHEMA"],
    )
    # Snowflake silently ignores a connect-time `schema=` the role cannot see,
    # leaving no current schema and a later, misleading "CREATE TEMPSTAGE" error.
    # Fail at the fixture boundary instead of hiding it with `USE SCHEMA`.
    with con.con.cursor() as cur:
        cur.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        current_database, current_schema = cur.fetchone()
    if current_schema is None:
        database = env["INCREMENT_TEST_SNOWFLAKE_DATABASE"]
        schema = env["INCREMENT_TEST_SNOWFLAKE_SCHEMA"]
        try:
            con.raw_sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}")
            con.raw_sql(f"USE SCHEMA {database}.{schema}")
        except Exception as exc:
            con.disconnect()
            pytest.fail(
                f"connected to Snowflake account {env['INCREMENT_TEST_SNOWFLAKE_ACCOUNT']!r} "
                f"as {env['INCREMENT_TEST_SNOWFLAKE_USER']!r}, but schema {database}.{schema} "
                f"has no current schema after connect (current_database={current_database!r}) "
                f"and could not be created/selected either: {exc}. The connecting role most "
                "likely lacks USAGE (or CREATE SCHEMA) privileges on this schema -- grant them "
                "as ACCOUNTADMIN, e.g. "
                f"`GRANT OWNERSHIP ON SCHEMA {database}.{schema} TO ROLE SYSADMIN;` "
                "This is a hard failure, not a skip or a silent USE SCHEMA."
            )
    try:
        yield con
    finally:
        con.disconnect()


@pytest.fixture
def bigquery_con():
    import ibis

    env = _require_env("INCREMENT_TEST_BIGQUERY_PROJECT", "INCREMENT_TEST_BIGQUERY_DATASET")
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.fail("missing GOOGLE_APPLICATION_CREDENTIALS for live BigQuery test")
    con = ibis.bigquery.connect(
        project_id=env["INCREMENT_TEST_BIGQUERY_PROJECT"],
        dataset_id=env["INCREMENT_TEST_BIGQUERY_DATASET"],
    )
    try:
        yield con
    finally:
        con.disconnect()
