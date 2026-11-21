from __future__ import annotations

from importlib import import_module

import ibis
import pyarrow as pa
import pytest

from increment.errors import CodedError
from increment.query.spillover import WarehouseSpilloverSource
from increment.spillover import SpilloverComparison, SpilloverDesign
from tests._unit_boundary import backend_fixture

pytestmark = pytest.mark.integration

_ROLES = ("roster", "offer_counts", "group_policy", "offered", "outcomes")
_SIZES = (4, 2, 4, 8, 5, 3) * 2


@backend_fixture()
def trial():
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE SCHEMA trial")
    rows: dict[str, list[dict]] = {key: [] for key in _ROLES}
    policies = ("base", "more", "full")
    strata = ("north", "south")
    for g, size in enumerate(_SIZES):
        policy = policies[g % 6 // 2]
        quotas = {"base": 0, "more": size // 2, "full": size}
        rows["group_policy"].append({"group_id": g, "policy": policy})
        for p in policies:
            rows["offer_counts"].append({"group_id": g, "policy": p, "offers": quotas[p]})
        for j in range(size):
            person = 100 * g + j
            rows["roster"].append({"person_id": person, "group_id": g, "stratum": strata[g // 6]})
            offered = j < quotas[policy]
            rows["offered"].append({"person_id": person, "offered": offered})
            if not offered:
                events = (1, 0, 1, 3, 0, 0)[g % 6]
                rows["outcomes"].append(
                    {"person_id": person, "outcome": int(j - quotas[policy] < events)}
                )
    design = SpilloverDesign(
        groups_per_policy={s: dict.fromkeys(policies, 2) for s in strata},
        comparisons=(SpilloverComparison(policy="more", reference="base", meaningful_effect=0.1),),
        outcome="purchase",
        horizon="one week",
    )
    source = WarehouseSpilloverSource(con, {key: f"trial.{key}" for key in _ROLES})

    def load(roles):
        for key in roles:
            con.create_table(key, obj=rows[key], database="trial")

    try:
        load(_ROLES[:2])
        yield con, design, source, lambda: load(_ROLES[2:])
    finally:
        con.disconnect()


def _guard_reads(monkeypatch, con, *, planning):
    """Check shapes at the backend boundary, including failed reconciliations."""
    fetch = con.to_pyarrow

    def guarded(expr, *args, **kwargs):
        schema = expr.as_table().schema()
        names = set(schema.names)
        result = fetch(expr, *args, **kwargs)
        assert "person_id" not in names
        assert planning or "group_id" not in names
        if names == {"bucket", "bucket_digest", "n"}:
            assert result.num_rows <= 4096
        elif planning and "group_id" in names:
            assert "person_id" not in names and "outcome" not in names
            assert result.num_rows <= len(_SIZES) * 3
        elif names == {"stratum", "n"}:
            assert result.num_rows <= 2
        elif {"policy", "stratum"} <= names:
            assert not {"group_id", "person_id"} & names
            assert result.num_rows <= 2 * 3
        else:
            assert all(t.is_numeric() or t.is_boolean() for t in schema.types), schema
            assert isinstance(result, pa.Scalar) or len(result) <= 1
        return result

    monkeypatch.setattr(con, "to_pyarrow", guarded)
    monkeypatch.setattr(con, "execute", lambda *a, **k: pytest.fail("unexpected pandas transfer"))
    monkeypatch.setattr(con, "to_pyarrow_batches", lambda *a, **k: pytest.fail("unexpected stream"))


def test_frozen_profiles_and_bounded_two_stage_cells(trial, monkeypatch):
    from tests.query.statement_census import count_statements

    con, design, source, launch = trial
    with monkeypatch.context() as guards, count_statements(con) as census:
        _guard_reads(guards, con, planning=True)
        population = source.freeze(design)
    assert not any(f'"trial"."{role}"' in sql for sql in census.statements for role in _ROLES[2:])
    assert [g.size for g in population.groups] == list(_SIZES)
    launch()
    original_temps = set(con.list_tables(database=("temp", "main")))
    with monkeypatch.context() as guards:
        _guard_reads(guards, con, planning=False)
        cells = source.observe(population)
    assert set(con.list_tables(database=("temp", "main"))) == original_temps
    actual = {
        (c.stratum, c.policy): (c.groups, c.mean_total, c.ss, c.unoffered, c.events) for c in cells
    }
    assert actual == {
        (s, p): expected
        for s in ("north", "south")
        for p, expected in [("base", (2, 0.5, 0.5, 6, 1)), ("more", (2, 4.0, 8.0, 6, 4))]
    }


def test_unverified_digest_refuses_before_relation_lookup(trial, monkeypatch):
    con, design, _, _ = trial
    with monkeypatch.context() as guards, pytest.raises(CodedError) as exc:
        guards.setattr(con, "name", "unverified")
        guards.setattr(con, "table", lambda *a, **kw: pytest.fail("opened a relation"))
        source = WarehouseSpilloverSource(con, {role: role for role in _ROLES})
        source.freeze(design)
    assert exc.value.code == "query.session.warehouse_artifact.verification_unavailable"
    assert "spillover.from_frame" in str(exc.value.context["route"])


@pytest.mark.parametrize(
    "change,code,field",
    [
        # Both preassignment changes preserve every group size and policy count.
        (
            "UPDATE trial.roster SET group_id = 1-group_id WHERE person_id IN (0,100)",
            "spillover.roster.changed",
            "roster",
        ),
        (
            "UPDATE trial.offer_counts SET offers = 1 WHERE group_id = 4 AND policy = 'more'",
            "spillover.roster.changed",
            "roster",
        ),
        (
            "UPDATE trial.roster SET stratum = chr(8195) WHERE person_id = 0",
            "spillover.design.invalid",
            "roster",
        ),
        (
            "INSERT INTO trial.offer_counts SELECT * FROM trial.offer_counts LIMIT 1",
            "spillover.design.duplicate",
            "offer_counts",
        ),
        (
            "UPDATE trial.offer_counts SET group_id = 99999 WHERE group_id = 0 AND policy = 'more'",
            "spillover.design.quota",
            "offer_counts",
        ),
        (
            "UPDATE trial.group_policy SET policy = CASE policy WHEN 'base' THEN 'more' ELSE 'base' END WHERE group_id IN (0,8)",
            "spillover.log.quota",
            "group_policy",
        ),
        ("INSERT INTO trial.offered VALUES (99999, false)", "spillover.log.keys", "offered"),
        (
            "UPDATE trial.offered SET offered = false WHERE person_id = 400",
            "spillover.log.quota",
            "offered",
        ),
        ("INSERT INTO trial.outcomes VALUES (400, 2)", "spillover.log.binary", "outcomes"),
        ("INSERT INTO trial.outcomes VALUES (99999, 0)", "spillover.log.keys", "outcomes"),
        (
            "UPDATE trial.outcomes SET outcome = NULL WHERE person_id = 0",
            "spillover.log.keys",
            "outcomes",
        ),
        ("ALTER TABLE trial.offered ALTER person_id TYPE VARCHAR", "spillover.log.keys", "offered"),
    ],
)
def test_refuses_before_transferring_invalid_logs(trial, monkeypatch, change, code, field):
    con, design, source, launch = trial
    population = source.freeze(design)
    launch()
    con.raw_sql(change)
    original_temps = set(con.list_tables(database=("temp", "main")))
    with monkeypatch.context() as guards, pytest.raises(CodedError) as exc:
        _guard_reads(guards, con, planning=False)
        source.observe(population)
    assert (exc.value.code, exc.value.context["field"]) == (code, field)
    assert set(con.list_tables(database=("temp", "main"))) == original_temps


def test_observation_uses_one_immutable_capture(trial, monkeypatch):
    con, design, source, launch = trial
    population = source.freeze(design)
    launch()
    fetch = con.to_pyarrow

    def edit_live_outcomes(expr, *args, **kwargs):
        monkeypatch.setattr(con, "to_pyarrow", fetch)
        con.raw_sql("DELETE FROM trial.outcomes")
        return fetch(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", edit_live_outcomes)
    assert sum(c.events for c in source.observe(population)) == 10
    with pytest.raises(CodedError) as exc:
        source.observe(population)
    assert exc.value.code == "spillover.log.keys"


@pytest.mark.slow(reason="compiles every executed spillover reduction on four dialects")
def test_active_spillover_queries_translate_on_verified_dialects(trial, monkeypatch):
    """Compile portable spillover reductions at the real database boundary."""
    con, design, source, launch = trial
    fetch = con.to_pyarrow
    dialects = ("duckdb", "postgres", "snowflake", "bigquery")
    for dialect in dialects:
        import_module(f"ibis.backends.sql.compilers.{dialect}")

    def compile_and_fetch(expr, *args, **kwargs):
        # Bucket hashes use this connection's native digest, not portable analysis SQL.
        if "bucket_digest" not in expr.as_table().columns:
            for dialect in dialects:
                ibis.to_sql(expr, dialect=dialect)
        return fetch(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", compile_and_fetch)
    population = source.freeze(design)
    launch()
    assert {(c.stratum, c.policy): c.mean_total for c in source.observe(population)} == {
        (s, p): expected
        for s in ("north", "south")
        for p, expected in (("base", 0.5), ("more", 4.0))
    }


def test_registered_tables_do_not_follow_database_namespace_changes(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import three_policy_trial, write_study

    con = ibis.duckdb.connect()
    design, frames = three_policy_trial()
    path = tmp_path / "study.yaml"
    write_study(path, design, {role: role for role in frames})
    try:
        for schema in ("original", "other"):
            con.create_database(schema)
            for role, frame in frames.items():
                con.create_table(role, frame, database=schema)
        con.raw_sql("UPDATE other.outcomes SET outcome = NOT outcome")
        con.raw_sql("SET schema = 'original'")
        experiment, reader = from_definitions(path, con, family="spillover")
        with reader:
            con.raw_sql("SET schema = 'other'")
            rows = run(experiment, reader).comparisons
        assert rows[1].estimate == pytest.approx(0.480952380952380909, rel=0, abs=1e-14)
        with pytest.raises(CodedError) as exc:
            from_definitions(
                path, con, family="spillover", expected_registration=experiment.registration_id
            )
        assert exc.value.code == "spillover.roster.changed"
    finally:
        con.disconnect()


def test_case_insensitive_connection_cannot_match_a_different_person(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import three_policy_trial, write_study

    con = ibis.duckdb.connect()
    design, frames = three_policy_trial()
    path = tmp_path / "study.yaml"
    write_study(path, design, {role: role for role in frames})
    try:
        con.raw_sql("SET default_collation = 'nocase'")
        for role, frame in frames.items():
            con.create_table(role, frame)
        experiment, reader = from_definitions(path, con, family="spillover")
        with reader:
            con.raw_sql("UPDATE outcomes SET person_id = 'STORE-0/1' WHERE person_id = 'store-0/1'")
            with pytest.raises(CodedError) as exc:
                run(experiment, reader)
        assert (exc.value.code, exc.value.context["field"]) == ("spillover.log.keys", "outcomes")
    finally:
        con.disconnect()
