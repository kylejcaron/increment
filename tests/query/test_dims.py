"""join_dims codegen: equi, range, changelog windowing, NULL retention."""

import datetime as dt

import ibis
import pytest

from increment.errors import InvalidRequestError
from increment.query.dims import join_dims
from increment.semantics.models import DimSource

# Every test here gets its own fresh, isolated `con` (see the fixture
# below) - tables can never leak between tests.
pytestmark = pytest.mark.creates_tables

T0 = dt.datetime(2025, 1, 1)
T1 = dt.datetime(2025, 2, 1)
T2 = dt.datetime(2025, 3, 1)
FOREVER = dt.datetime(9999, 12, 31)


@pytest.fixture
def con():
    return ibis.duckdb.connect()  # in-memory


def _table(con, name, data):
    # ibis 12's duckdb backend takes an ibis expression here; a plain dict of
    # lists is not a documented obj type, so wrap it in memtable.
    return con.create_table(name, ibis.memtable(data))


def _col(rows, name):
    """List a column with SQL NULL as ``None``, not pandas' object-dtype ``nan``."""
    return [None if v is None or v != v else v for v in rows[name]]


def _fact(con):
    return _table(
        con,
        "fact",
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "ts": [T0, T2, T1, T1],
            "amount": [10.0, 20.0, 30.0, 40.0],
        },
    )


def _static_dim_model():
    return DimSource.model_validate(
        {
            "name": "users",
            "sql": "unused -- table supplied directly in tests",
            "entity": "user_id",
            "properties": [{"name": "country", "column": "country", "as_of": "static"}],
        }
    )


def test_static_dim_equi_join(con):
    fact = _fact(con)
    dim = _table(con, "dim_users", {"user_id": ["u1", "u2"], "country": ["DE", "US"]})
    out = join_dims(fact, ts_column="ts", dims=[(_static_dim_model(), dim)]).order_by(
        "ts", "user_id"
    )
    rows = out.execute()
    assert _col(rows, "country") == ["DE", "US", None, "DE"]  # u3 has no dim row -> NULL


def test_static_dim_left_join_never_drops_fact_rows(con):
    fact = _fact(con)
    dim = _table(con, "dim_empty", {"user_id": ["zz"], "country": ["XX"]})
    out = join_dims(fact, ts_column="ts", dims=[(_static_dim_model(), dim)])
    assert out.count().execute() == 4


def _ranged_dim_model():
    return DimSource.model_validate(
        {
            "name": "user_plan",
            "sql": "unused",
            "entity": "user_id",
            "validity": {"valid_from": "valid_from", "valid_to": "valid_to"},
            "properties": [{"name": "plan", "column": "plan", "as_of": "pre_exposure"}],
        }
    )


def test_ranged_dim_resolves_version_at_fact_time(con):
    fact = _fact(con)
    dim = _table(
        con,
        "snap_plan",
        {
            "user_id": ["u1", "u1", "u2"],
            "plan": ["free", "pro", "free"],
            "valid_from": [T0, T1, T0],
            "valid_to": [T1, FOREVER, FOREVER],
        },
    )
    out = join_dims(fact, ts_column="ts", dims=[(_ranged_dim_model(), dim)]).order_by(
        "ts", "user_id"
    )
    rows = out.execute()
    # u1@T0 -> free (first version), u1@T2 -> pro (second), u2@T1 -> free, u3 -> NULL
    assert _col(rows, "plan") == ["free", "free", None, "pro"]


def test_null_valid_to_is_coalesced_to_sentinel(con):
    fact = _fact(con)
    dim = _table(
        con,
        "snap_plan_null",
        {
            "user_id": ["u1", "u1"],
            "plan": ["free", "pro"],
            "valid_from": [T0, T1],
            "valid_to": [T1, None],  # open current row stored as NULL
        },
    )
    out = join_dims(fact, ts_column="ts", dims=[(_ranged_dim_model(), dim)]).order_by("ts")
    rows = out.filter(out.user_id == "u1").execute()
    assert _col(rows, "plan") == ["free", "pro"]  # NULL close must behave as far-future


def _changelog_dim_model():
    return DimSource.model_validate(
        {
            "name": "user_plan",
            "sql": "unused",
            "entity": "user_id",
            "validity": {"changed_at": "changed_at"},
            "properties": [{"name": "plan", "column": "plan", "as_of": "pre_exposure"}],
        }
    )


def test_changelog_windows_into_ranges(con):
    fact = _fact(con)
    log = _table(
        con,
        "plan_changes",
        {
            "user_id": ["u1", "u1", "u2"],
            "plan": ["free", "pro", "free"],
            "changed_at": [T0, T1, T0],
        },
    )
    out = join_dims(fact, ts_column="ts", dims=[(_changelog_dim_model(), log)]).order_by(
        "ts", "user_id"
    )
    rows = out.execute()
    # Same resolution as the ranged test: latest change at or before ts.
    assert _col(rows, "plan") == ["free", "free", None, "pro"]


def test_fact_events_before_first_change_get_null(con):
    fact = _fact(con)
    log = _table(con, "late_log", {"user_id": ["u1"], "plan": ["pro"], "changed_at": [T2]})
    out = join_dims(fact, ts_column="ts", dims=[(_changelog_dim_model(), log)]).order_by("ts")
    rows = out.filter(out.user_id == "u1").execute()
    assert _col(rows, "plan") == [None, "pro"]  # T0 predates history -> NULL, T2 -> pro


def test_plain_dim_duplicate_rows_rejected_before_fanout(con):
    # Two dim rows for the same entity would otherwise fan the fact row out
    # into two joined rows, silently doubling additive aggregates.
    fact = _fact(con)
    dim = _table(con, "dim_users_dup", {"user_id": ["u1", "u1"], "country": ["DE", "US"]})
    with pytest.raises(InvalidRequestError) as exc_info:
        join_dims(fact, ts_column="ts", dims=[(_static_dim_model(), dim)])
    assert exc_info.value.code == "query.dims.dim_source_multiple"
    assert exc_info.value.context["entity"] == "user_id"
    assert exc_info.value.context["name"] == "users"


def test_changelog_tied_changed_at_rejected(con):
    # Two changelog rows sharing one changed_at resolve the tie by physical
    # row order via LEAD; reject the ambiguity instead of guessing.
    fact = _fact(con)
    log = _table(
        con,
        "plan_changes_tied",
        {"user_id": ["u1", "u1"], "plan": ["free", "pro"], "changed_at": [T0, T0]},
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        join_dims(fact, ts_column="ts", dims=[(_changelog_dim_model(), log)])
    assert exc_info.value.code == "query.dims.dim_source_two"
    assert exc_info.value.context["entity"] == "user_id"
    assert exc_info.value.context["name"] == "user_plan"


def test_ranged_dim_overlapping_validity_rejected(con):
    # Two explicit validity ranges overlapping for the same entity are the
    # same ambiguity as a changelog tie, just declared directly.
    fact = _fact(con)
    dim = _table(
        con,
        "snap_plan_overlap",
        {
            "user_id": ["u1", "u1"],
            "plan": ["free", "pro"],
            "valid_from": [T0, T0.replace(day=15)],
            "valid_to": [T1, FOREVER],
        },
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        join_dims(fact, ts_column="ts", dims=[(_ranged_dim_model(), dim)])
    assert exc_info.value.code == "query.dims.dim_source_overlapping"
    assert exc_info.value.context["entity"] == "user_id"
    assert exc_info.value.context["name"] == "user_plan"


def test_property_name_colliding_with_fact_column_rejected(con):
    fact = _table(
        con,
        "fact_with_country",
        {"user_id": ["u1"], "ts": [T0], "country": ["??"]},
    )
    dim = _table(con, "dim_users2", {"user_id": ["u1"], "country": ["DE"]})
    with pytest.raises(InvalidRequestError) as exc_info:
        join_dims(fact, ts_column="ts", dims=[(_static_dim_model(), dim)])
    assert exc_info.value.code == "query.dims.dim_source_property"
    assert exc_info.value.context["column"] == "country"
    assert exc_info.value.context["name"] == "users"


def test_fact_source_reserved_column_rejected(con):
    # A versioned dim's range projection introduces valid_from/valid_to; a fact source that already projects one of those names must be refused, not silently suffixed by the join.
    fact = _table(
        con,
        "fact_with_valid_from",
        {"user_id": ["u1"], "ts": [T0], "valid_from": ["oops"]},
    )
    dim = _table(
        con,
        "dim_plan_reserved",
        {"user_id": ["u1"], "plan": ["free"], "valid_from": [T0], "valid_to": [FOREVER]},
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        join_dims(fact, ts_column="ts", dims=[(_ranged_dim_model(), dim)])
    assert exc_info.value.code == "query.dims.dim_source_fact"
    assert exc_info.value.context["name"] == "user_plan"


def test_two_dims_chain(con):
    fact = _fact(con)
    dim = _table(con, "dim_users3", {"user_id": ["u1", "u2", "u3"], "country": ["DE", "US", "GB"]})
    log = _table(con, "plan_changes2", {"user_id": ["u1"], "plan": ["free"], "changed_at": [T0]})
    out = join_dims(
        fact,
        ts_column="ts",
        dims=[(_static_dim_model(), dim), (_changelog_dim_model(), log)],
    )
    assert set(out.columns) >= {"user_id", "ts", "amount", "country", "plan"}
    assert out.count().execute() == 4


def _tier_dim_model():
    return DimSource.model_validate(
        {
            "name": "user_tier",
            "sql": "unused",
            "entity": "user_id",
            "validity": {"valid_from": "valid_from", "valid_to": "valid_to"},
            "properties": [{"name": "tier", "column": "tier", "as_of": "pre_exposure"}],
        }
    )


def test_two_versioned_dims_chain_no_column_leak(con):
    # Chains a ranged dim then a changelog dim (the only versioned+versioned combo test_two_dims_chain doesn't cover) - iteration 1's range columns must not leak into iteration 2's reserved-column guard.
    fact = _fact(con)
    tier = _table(
        con,
        "snap_tier",
        {
            "user_id": ["u1", "u2"],
            "tier": ["gold", "silver"],
            "valid_from": [T0, T0],
            "valid_to": [FOREVER, FOREVER],
        },
    )
    log = _table(con, "plan_changes4", {"user_id": ["u1"], "plan": ["free"], "changed_at": [T0]})
    out = join_dims(
        fact,
        ts_column="ts",
        dims=[(_tier_dim_model(), tier), (_changelog_dim_model(), log)],
    )
    assert set(out.columns) >= {"user_id", "ts", "amount", "tier", "plan"}
    assert out.count().execute() == 4


def test_join_dims_compiles_to_snowflake(con):
    # Repo convention (tests/query/test_builders.py): every query builder
    # gets a cross-backend render test - no live warehouse, just compile.
    fact = _fact(con)
    dim = _table(con, "dim_users4", {"user_id": ["u1"], "country": ["DE"]})
    log = _table(con, "plan_changes3", {"user_id": ["u1"], "plan": ["free"], "changed_at": [T0]})
    out = join_dims(
        fact,
        ts_column="ts",
        dims=[(_static_dim_model(), dim), (_changelog_dim_model(), log)],
    )
    sql = ibis.to_sql(out, dialect="snowflake")
    assert "LEFT" in sql.upper()  # the join strategy survives transpilation
