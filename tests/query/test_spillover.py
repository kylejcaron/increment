from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import import_module

import ibis
import pyarrow as pa
import pytest

from increment import Definitions
from increment.errors import CodedError
from increment.query.session import SourceSnapshotEvidence
from increment.query.spillover import WarehouseSpilloverSource
from increment.semantics.models import Filter, SpilloverDeclaration
from tests._unit_boundary import backend_fixture

pytestmark = pytest.mark.integration

_ROLES = ("roster", "offer_counts", "group_policy", "assignments", "purchases")
_SIZES = (4, 2, 4, 8, 5, 3) * 2
_ANCHOR = datetime(2025, 1, 1, 12, tzinfo=UTC)
# Local midnight after the seventh day in UTC-05:00.
_END = datetime(2025, 1, 8, 5, tzinfo=UTC)
_COMPLETE = SourceSnapshotEvidence(_END, {"purchases": _END})


def _definitions(allocations, *, schema="trial", dimension=False, observation_end="2025-01-09"):
    def name(role):
        return f"{schema}.{role}" if schema else role

    filters = [Filter(property="channel", op="equals", values=("web",))]
    if dimension:
        filters.append(Filter(property="eligible", op="equals", values=(True,)))
    declarations = {
        "fact_sources": [
            {
                "name": "draws",
                "sql": f"WITH Draws AS (SELECT * FROM {name('assignments')}) SELECT * FROM draws",
                "entities": ["person_id"],
                "timestamp_column": "assigned_at",
                "facts": [{"name": "assignment"}],
            },
            {
                "name": "purchases",
                "sql": f"SELECT * FROM {name('purchases')}",
                "entities": ["person_id"],
                "timestamp_column": "purchased_at",
                "facts": [{"name": "purchase", "column": "price"}],
                "properties": [{"name": "channel", "column": "sales_channel"}],
                "dims": ["preferred"] if dimension else [],
            },
        ],
        "exposures": [{"name": "coupon_draw", "fact": "assignment"}],
        "metrics": [
            {
                "name": "bought",
                "type": "conversion",
                "entity": "person_id",
                "fact": "purchase",
                "window_days": 7,
                "filters": filters,
            }
        ],
        "experiments": [
            {
                "name": "pilot",
                "exposure": "coupon_draw",
                "unit": "person_id",
                "control_group": "no_offer",
                "start": "2025-01-01",
                "end": "2025-01-03",
                "observation_end": observation_end,
                "day_boundary": "UTC-05:00",
                "plan": {"primary": "bought"},
                "design": {
                    "mechanism": "spillover",
                    "offered_group": "offered",
                    "groups_per_policy": allocations,
                    "comparisons": [
                        {"policy": "more", "reference": "base", "meaningful_effect": 0.1}
                    ],
                    **{r: name(r) for r in _ROLES[:3]},
                },
            }
        ],
    }
    if dimension:
        declarations["dim_sources"] = [
            {
                "name": "preferred",
                "entity": "person_id",
                "sql": f"SELECT * FROM {name('preferred')}",
                "properties": [
                    {"name": "eligible", "column": "eligible", "dtype": "bool", "as_of": "static"}
                ],
            }
        ]
    return Definitions.model_validate(declarations)


def _bind(con, definitions, evidence=_COMPLETE):
    experiment = definitions.experiment("pilot")
    assert experiment is not None and isinstance(experiment.design, SpilloverDeclaration)
    (name,) = experiment.metric_names
    metric = definitions.metric(name)
    assert metric is not None and metric.type == "conversion" and metric.window_days is not None
    law = experiment.design.parameters(experiment.control_group).law(
        metric.name, metric.window_days, experiment.plan.alpha
    )
    source = WarehouseSpilloverSource(
        con,
        definitions=definitions,
        experiment=experiment,
        metric=metric,
        source_snapshot_evidence=evidence,
    )
    return law, source


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
            rows["assignments"].append(
                {
                    "person_id": person,
                    "assigned_at": _ANCHOR,
                    "event": "assignment",
                    "group_id": "offered" if offered else "no_offer",
                    "experiment_id": "pilot",
                }
            )
            events = (1, 0, 1, 3, 0, 0)[g % 6]
            if not offered and j - quotas[policy] < events:
                rows["purchases"].append(
                    {
                        "person_id": person,
                        "purchased_at": _ANCHOR + timedelta(hours=1),
                        "event": "purchase",
                        "sales_channel": "web",
                        "price": 3,
                    }
                )
    # Duplicate, filtered, same-time and next-window events must not add buyers.
    rows["purchases"].append(rows["purchases"][0].copy())
    for ts, channel in ((_ANCHOR, "web"), (_END, "web"), (_ANCHOR + timedelta(hours=1), "shop")):
        rows["purchases"].append(
            {
                "person_id": 1,
                "purchased_at": ts,
                "event": "purchase",
                "sales_channel": channel,
                "price": None,
            }
        )
    rows["assignments"].append(dict(rows["assignments"][0], experiment_id="other"))
    definitions = _definitions({s: dict.fromkeys(policies, 2) for s in strata})
    design, source = _bind(con, definitions)

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
        assert names.isdisjoint(
            {"person_id", "unit_id", "first_exposure_ts", "assigned_at", "purchased_at"}
        )
        assert planning or "group_id" not in names
        if names == {"bucket", "bucket_digest", "n"}:
            assert result.num_rows <= 4096
        elif planning and names == {"catalog", "location"}:
            assert result.num_rows == 1
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


@backend_fixture()
def portable_trial(tmp_path):
    import yaml

    from tests.spillover_trial import three_policy_trial, write_study

    design, frames = three_policy_trial()
    path = tmp_path / "pilot.yaml"
    portable = tmp_path / "shop.yaml"
    write_study(path, design, {role: f"trial.{role}" for role in _ROLES}, dialect="duckdb")
    native = yaml.safe_load(path.read_text())
    del native["metrics"]
    native["fact_sources"][0]["dims"] = ["accounts"]
    native["dim_sources"] = [
        {
            "name": "accounts",
            "entity": "person_id",
            "sql": "SELECT * FROM trial.account_attributes",
            "properties": [{"name": "tier", "column": "tier", "as_of": "static"}],
        }
    ]
    native["metric_definitions"] = {
        "file": portable.name,
        "use": ["bought"],
        "sources": {"orders": "purchases"},
        "measures": {"web_orders": "coupon_purchase"},
        "metrics": {"bought": {"entity": "customer", "window_days": 7}},
    }
    path.write_text(yaml.safe_dump(native))
    portable.write_text("""\
version: 1
sources:
  orders:
    table: sales
    row_key: order_id
    entities: {customer: person_id}
    time: purchased_at
    dimensions: {channel: {column: channel, type: string}}
  members: {table: eligible, row_key: member_id, entities: {customer: member_id}}
populations:
  eligible_customers: {source: members, entity: customer}
filters:
  web_only: {source: orders, conditions: [{dimension: channel, op: eq, value: web}]}
measures:
  web_orders: {source: orders, count_rows: true, filter: web_only}
metrics:
  bought: {conversion: {event_count: web_orders, population: eligible_customers}}
""")
    con = ibis.duckdb.connect()
    try:
        con.create_database("trial")
        for role in _ROLES[:2]:
            con.create_table(role, frames[role], database="trial")
        yield con, path, portable, frames
    finally:
        con.disconnect()


@pytest.mark.parametrize(
    "change,reason",
    [
        # Same cardinality; only one offered member's identity differs by case.
        (
            "UPDATE main.eligible SET member_id = upper(member_id) WHERE member_id = 'store-0/0'",
            "eligible roster",
        ),
        # A repeated receipt still invalidates a purchase by someone outside the trial.
        (
            "INSERT INTO trial.purchases SELECT * REPLACE ('outside' AS person_id) "
            "FROM trial.purchases WHERE channel = 'shop'",
            "order_id",
        ),
    ],
)
def test_shared_conversion_checks_population_and_event_grain_in_its_capture(
    portable_trial, monkeypatch, change, reason
):
    from increment.semantics.loader import load
    from tests.query.statement_census import count_statements
    from tests.spillover_trial import complete_receipt

    con, path, _, frames = portable_trial
    con.raw_sql("SET default_collation = 'nocase'")
    design, source = _bind(con, load(path), complete_receipt())
    population = source.freeze(design)
    con.raw_sql("CREATE TABLE eligible AS SELECT person_id AS member_id FROM trial.roster")
    purchases = frames["purchases"]
    frames["purchases"] = purchases.append_column("order_id", pa.array(range(len(purchases))))
    for role in _ROLES[2:]:
        con.create_table(role, frames[role], database="trial")
    con.raw_sql("CREATE TABLE trial.eligible AS SELECT * FROM main.eligible WHERE false")
    con.raw_sql("SET schema = 'trial'")

    fetch = con.to_pyarrow

    def change_after_capture(expr, *args, **kwargs):
        monkeypatch.setattr(con, "to_pyarrow", fetch)
        con.raw_sql(change)
        return fetch(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", change_after_capture)
    with count_statements(con) as census:
        assert sum(cell.events for cell in source.observe(population)) == 12
    assert not any("account_attributes" in sql.lower() for sql in census.statements)
    with monkeypatch.context() as guards, pytest.raises(CodedError) as exc:
        _guard_reads(guards, con, planning=False)
        source.observe(population)
    assert exc.value.code == "definition.shared.metric_unsupported"
    assert exc.value.context["metric"] == "bought"
    assert reason in str(exc.value.context["reason"])


def test_shared_conversion_binds_roster_location_key_and_event_grain_before_launch(portable_trial):
    from increment.semantics.loader import load

    con, path, portable, _ = portable_trial
    text = portable.read_text()
    _, source = _bind(con, load(path))
    for old, new in (
        ("order_id", "receipt_id"),
        ("member_id", "user_id"),
        ("table: eligible", "table: enrolled"),
    ):
        portable.write_text(text.replace(old, new))
        _, rebound = _bind(con, load(path))
        assert rebound.recipe != source.recipe, new
    portable.write_text(text)
    # Only the portable roster depends on the connection's default namespace.
    con.raw_sql("SET schema = 'trial'")
    _, rebound = _bind(con, load(path))
    assert rebound.recipe != source.recipe
    # A new attachment with the same qualified name is not the registered population.
    portable.write_text(text.replace("table: eligible", "table: members.main.eligible"))
    con.raw_sql("ATTACH ':memory:' AS members")
    _, attached = _bind(con, load(path))
    con.raw_sql("DETACH members")
    con.raw_sql("ATTACH ':memory:' AS members")
    _, replacement = _bind(con, load(path))
    assert replacement.recipe != attached.recipe


def test_frozen_profiles_and_bounded_two_stage_cells(trial, monkeypatch):
    from tests.query.statement_census import count_statements

    con, law, _, launch = trial
    with monkeypatch.context() as guards, count_statements(con) as census:
        _guard_reads(guards, con, planning=True)
        design, source = _bind(con, _definitions(law.groups_per_policy))
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


def test_shared_feed_history_does_not_enlarge_the_warehouse_capture(trial, monkeypatch):
    con, design, source, launch = trial
    population = source.freeze(design)
    launch()
    create = con.create_table
    sizes = []

    def count_written(*args, **kwargs):
        table = create(*args, **kwargs)
        if kwargs.get("temp"):
            sizes.append(con.to_pyarrow(table.count()).as_py())
        return table

    monkeypatch.setattr(con, "create_table", count_written)
    expected = source.observe(population)
    initial = sum(sizes)
    assert initial > 0
    sizes.clear()
    con.raw_sql("""
        INSERT INTO trial.purchases
        SELECT 99999, TIMESTAMP '2025-01-02', 'purchase', 'web', 1 FROM range(100)
        UNION ALL
        SELECT 0, TIMESTAMP '2020-01-01', 'purchase', 'web', 1 FROM range(100)
        UNION ALL
        SELECT 0, TIMESTAMP '2030-01-01', 'purchase', 'web', 1 FROM range(100);
        INSERT INTO trial.assignments
        SELECT 0, TIMESTAMP '2025-01-02', 'assignment', 'no_offer', 'other' FROM range(100)
    """)
    assert source.observe(population) == expected
    assert sum(sizes) == initial


@pytest.mark.slow
def test_policy_cell_repeated_fraction_preserves_mean_and_zero_spread():
    # Every group has size 10, and 1/3 of its 3, 6 or 9 unoffered people
    # have events. The cell mean is 10/3, not a sum of rounded group totals.
    con = ibis.duckdb.connect(threads=1)
    try:
        con.raw_sql("""
            CREATE VIEW profiles AS
            SELECT g AS group_id, IF(g % 2 = 0, 'base', 'more') AS policy,
                   IF(g % 4 < 2, 6, 9) AS base_free
            FROM range(32768) AS t(g);
            CREATE VIEW people AS
            SELECT 10 * group_id + j AS person_id, group_id, j,
                   base_free - IF(policy = 'more', 3, 0) AS n_free
            FROM profiles CROSS JOIN range(10) AS t(j);
            CREATE VIEW roster AS SELECT person_id, group_id, 'all' AS stratum FROM people;
            CREATE VIEW group_policy AS SELECT group_id, policy FROM profiles;
            CREATE VIEW offer_counts AS
                SELECT group_id, 'base' AS policy, 10 - base_free AS offers FROM profiles
                UNION ALL SELECT group_id, 'more', 13 - base_free FROM profiles;
            CREATE VIEW assignments AS
                SELECT person_id, TIMESTAMP '2025-01-01 12:00:00' AS assigned_at,
                       'assignment' AS event, IF(j >= n_free, 'offered', 'no_offer') AS group_id
                FROM people;
            CREATE VIEW purchases AS
                SELECT person_id, TIMESTAMP '2025-01-01 13:00:00' AS purchased_at,
                       'purchase' AS event, 'web' AS sales_channel, 3 AS price
                FROM people WHERE 3 * j < n_free
        """)
        design, source = _bind(
            con, _definitions({"all": {"base": 16384, "more": 16384}}, schema="")
        )
        cells = source.observe(source.freeze(design))
        assert {cell.policy for cell in cells} == {"base", "more"}
        for cell in cells:
            assert cell.mean_total == pytest.approx(10 / 3, abs=1e-14, rel=0)
            assert cell.ss == 0
    finally:
        con.disconnect()


def test_unverified_digest_refuses_before_relation_lookup(trial, monkeypatch):
    con, _, _, _ = trial
    with monkeypatch.context() as guards, pytest.raises(CodedError) as exc:
        guards.setattr(con, "name", "unverified")
        guards.setattr(con, "table", lambda *a, **kw: pytest.fail("opened a relation"))
        _bind(con, _definitions({"all": {"base": 2, "more": 2}}))
    assert exc.value.code == "query.session.warehouse_artifact.verification_unavailable"
    assert "from_frame" in str(exc.value.context["route"])


def test_binding_admits_source_sql_in_the_connection_dialect_before_reading(trial, monkeypatch):
    con, law, _, _ = trial
    definitions = _definitions(law.groups_per_policy)
    draws, purchases = definitions.fact_sources
    purchases = purchases.model_copy(update={"sql": "SELECT 1; SELECT 2"})
    invalid = definitions.model_copy(update={"fact_sources": (draws, purchases)})
    monkeypatch.setattr(con, "table", lambda *a, **kw: pytest.fail("opened a relation"))
    with pytest.raises(CodedError) as exc:
        _bind(con, invalid)
    assert exc.value.code == "definition.sql.statement_count"
    assert exc.value.context["dialect"] == "duckdb"


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
        (
            "UPDATE trial.assignments SET person_id = 99999 WHERE person_id = 0",
            "spillover.log.keys",
            "offered",
        ),
        (
            "UPDATE trial.assignments SET group_id = 'no_offer' WHERE person_id = 400",
            "spillover.log.quota",
            "offered",
        ),
        (
            "UPDATE trial.assignments SET group_id = 'redeemed' WHERE person_id = 1",
            "spillover.log.keys",
            "offered",
        ),
        (
            "INSERT INTO trial.assignments SELECT * FROM trial.assignments WHERE person_id = 1",
            "spillover.log.keys",
            "offered",
        ),
        (
            "UPDATE trial.assignments SET assigned_at = NULL WHERE person_id = 1",
            "spillover.log.keys",
            "offered",
        ),
        (
            "ALTER TABLE trial.assignments ALTER person_id TYPE VARCHAR",
            "spillover.log.keys",
            "offered",
        ),
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
        con.raw_sql("DELETE FROM trial.purchases")
        return fetch(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", edit_live_outcomes)
    assert sum(c.events for c in source.observe(population)) == 10
    assert sum(c.events for c in source.observe(population)) == 0


def test_offered_customer_still_requires_the_declared_observation_days(trial):
    con, law, _, launch = trial
    definitions = _definitions(law.groups_per_policy, observation_end="2025-01-08")
    late = _END + timedelta(days=3)
    _, source = _bind(con, definitions, SourceSnapshotEvidence(late, {"purchases": late}))
    population = source.freeze(law)
    launch()
    con.raw_sql("""
        UPDATE trial.assignments SET assigned_at = assigned_at + INTERVAL 2 DAY
        WHERE person_id = 400
    """)
    with pytest.raises(CodedError) as exc:
        source.observe(population)
    assert (exc.value.code, exc.value.context["field"]) == (
        "spillover.window.incomplete",
        "observation_end",
    )


def test_filtered_conversion_waits_for_dimension_feed_through_local_midnight(trial):
    con, law, _, launch = trial
    definitions = _definitions(law.groups_per_policy, dimension=True)
    _, source = _bind(con, definitions)
    population = source.freeze(law)
    launch()
    con.raw_sql("CREATE TABLE trial.preferred AS SELECT 0 AS person_id, true AS eligible")
    for through in ({}, {"preferred": _END - timedelta(microseconds=1)}):
        receipt = SourceSnapshotEvidence(_END, {"purchases": _END, **through})
        _, incomplete = _bind(con, definitions, receipt)
        with pytest.raises(CodedError) as exc:
            incomplete.observe(population)
        assert (exc.value.code, exc.value.context["field"]) == (
            "spillover.window.incomplete",
            "preferred",
        )
    receipt = SourceSnapshotEvidence(_END, {"purchases": _END, "preferred": _END})
    _, complete = _bind(con, definitions, receipt)
    assert complete.recipe == source.recipe
    cells = complete.observe(population)
    assert sum(cell.events for cell in cells) == 1
    # The same eligible purchases selected at the fact source give identical cells.
    plain = _definitions(law.groups_per_policy)
    draws, sales = plain.fact_sources
    sales = sales.model_copy(update={"sql": sales.sql + " WHERE person_id = 0"})
    _, direct = _bind(con, plain.model_copy(update={"fact_sources": (draws, sales)}))
    assert cells == direct.observe(population)


def test_null_value_in_measured_conversion_refuses(trial):
    con, law, source, launch = trial
    population = source.freeze(law)
    launch()
    con.raw_sql("UPDATE trial.purchases SET price=NULL WHERE person_id=0")
    with pytest.raises(CodedError) as exc:
        source.observe(population)
    assert exc.value.code == "source.native.fact_null_values"
    assert exc.value.context["fact"] == "purchase"
    assert exc.value.context["null_rows"] == 2


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


def test_column_mapping_preserves_estimates_but_cannot_rebind_a_saved_trial(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import (
        complete_receipt,
        frame_trial,
        spillover_rows,
        three_policy_trial,
        write_study,
    )

    design, frames = three_policy_trial()
    trial, reader = frame_trial(design, frames)
    with reader:
        expected = spillover_rows(run(trial, reader))

    def compare(result):
        for actual, local in zip(spillover_rows(result), expected, strict=True):
            assert (actual.policy, actual.reference) == (local.policy, local.reference)
            assert actual.estimate == pytest.approx(local.estimate, abs=1e-14, rel=0)
            assert (actual.lower, actual.upper) == pytest.approx(
                (local.lower, local.upper), abs=1e-10, rel=0
            )

    path = tmp_path / "study.yaml"
    names = {role: role for role in _ROLES}
    roles = {"person_id": "customer_id", "group_id": "store_id", "stratum": "region"}
    con = ibis.duckdb.connect(tmp_path / "offers.duckdb")
    try:
        for role in _ROLES:
            frame = frames[role]
            renamed = frame.rename_columns([roles.get(c, c) for c in frame.column_names])
            if "store_id" in renamed.column_names:
                renamed = renamed.append_column("group_id", renamed["store_id"])
            if "region" in renamed.column_names:
                renamed = renamed.append_column("another_region", renamed["region"])
            con.create_table(role, renamed)
        mapped = design | {"unit": "customer_id", "cluster": "store_id", "stratum": "region"}
        write_study(path, mapped, names)
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            compare(run(experiment, reader))
        for column in ({"cluster": "group_id"}, {"stratum": "another_region"}, {"stratum": None}):
            write_study(path, mapped | column, names)
            rebound, reader = from_definitions(
                path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
            )
            with reader:
                assert rebound.to_json() != experiment.to_json()
                with pytest.raises(CodedError) as stale:
                    run(experiment, reader)
                assert stale.value.code == "kernel.run.execution_unavailable"
                compare(run(rebound, reader))
    finally:
        con.disconnect()


def test_registered_tables_do_not_follow_database_namespace_changes(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import (
        complete_receipt,
        spillover_rows,
        three_policy_trial,
        write_study,
    )

    con = ibis.duckdb.connect()
    design, frames = three_policy_trial()
    path = tmp_path / "study.yaml"
    names = {role: role for role in _ROLES}
    write_study(path, design, names)
    try:
        for schema in ("original", "other"):
            con.create_database(schema)
            for role in _ROLES:
                con.create_table(role, frames[role], database=schema)
        con.raw_sql("DELETE FROM other.purchases")
        con.raw_sql("SET schema = 'original'")
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        # Changing only the registered confidence level, even to the default
        # (omitted in YAML), invalidates the receipt against these same tables.
        write_study(path, design | {"alpha": 0.05}, names)
        _, changed_reader = from_definitions(path, con, experiment_name="pilot")
        with changed_reader, pytest.raises(CodedError) as stale:
            run(experiment, changed_reader)
        assert stale.value.code == "kernel.run.execution_unavailable"
        write_study(path, design, names)
        with reader:
            con.raw_sql("SET schema = 'other'")
            rows = spillover_rows(run(experiment, reader))
        assert rows[1].estimate == pytest.approx(0.480952380952380909, rel=0, abs=1e-14)
        _, changed_reader = from_definitions(path, con, experiment_name="pilot")
        with changed_reader, pytest.raises(CodedError) as exc:
            run(experiment, changed_reader)
        assert exc.value.code == "kernel.run.execution_unavailable"
    finally:
        con.disconnect()


def test_registration_binds_duckdb_file_behind_attached_log_name(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import complete_receipt, three_policy_trial, write_study

    design, frames = three_policy_trial()
    study = tmp_path / "study.yaml"
    names = {role: role for role in _ROLES} | {"purchases": "events.main.purchases"}
    write_study(study, design, names)
    main_file = tmp_path / "trial.duckdb"
    original = tmp_path / "original.duckdb"
    other = tmp_path / "other.duckdb"
    con = ibis.duckdb.connect(main_file)
    try:
        con.attach(original, name="events")
        for role in ("roster", "offer_counts"):
            con.create_table(role, frames[role])
        experiment, reader = from_definitions(study, con, experiment_name="pilot")
        with reader:
            assert str(original) not in experiment.to_json()
        for role in ("group_policy", "assignments"):
            con.create_table(role, frames[role])
    finally:
        con.disconnect()

    # The location, not the contents or lifetime of the connection, is registered.
    for file in (original, other):
        writer = ibis.duckdb.connect(file)
        try:
            writer.create_table("purchases", frames["purchases"])
        finally:
            writer.disconnect()
    con = ibis.duckdb.connect(main_file)
    try:
        con.attach(original, name="events")
        rebound, reader = from_definitions(
            study, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            assert experiment.registration == rebound.registration
            assert len(run(experiment, reader).cells) == 2
            con.detach("events")
            with pytest.raises(CodedError) as detached:
                run(experiment, reader)
            assert detached.value.code == "spillover.roster.changed"
            con.attach(other, name="events")
            with pytest.raises(CodedError) as changed:
                run(experiment, reader)
            assert changed.value.code == "spillover.roster.changed"
        _, changed_reader = from_definitions(study, con, experiment_name="pilot")
        with changed_reader, pytest.raises(CodedError) as changed:
            run(experiment, changed_reader)
        assert changed.value.code == "kernel.run.execution_unavailable"
    finally:
        con.disconnect()


def test_independent_in_memory_databases_do_not_share_registration(tmp_path):
    from increment import from_definitions
    from tests.spillover_trial import three_policy_trial, write_study

    design, frames = three_policy_trial()
    study = tmp_path / "study.yaml"
    write_study(study, design, {role: role for role in _ROLES})
    receipts = []
    for _ in range(2):
        con = ibis.duckdb.connect()
        try:
            for role in ("roster", "offer_counts"):
                con.create_table(role, frames[role])
            experiment, reader = from_definitions(study, con, experiment_name="pilot")
            with reader:
                receipts.append(experiment.registration)
        finally:
            con.disconnect()
    assert receipts[0] != receipts[1]


def test_case_insensitive_connection_cannot_match_a_different_person(tmp_path):
    from increment import from_definitions, run
    from tests.spillover_trial import complete_receipt, three_policy_trial, write_study

    con = ibis.duckdb.connect()
    design, frames = three_policy_trial()
    path = tmp_path / "study.yaml"
    write_study(path, design, {role: role for role in _ROLES})
    try:
        con.raw_sql("SET default_collation = 'nocase'")
        for role in _ROLES:
            con.create_table(role, frames[role])
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            con.raw_sql(
                "UPDATE assignments SET person_id = 'STORE-0/1' WHERE person_id = 'store-0/1'"
            )
            with pytest.raises(CodedError) as exc:
                run(experiment, reader)
        assert (exc.value.code, exc.value.context["field"]) == ("spillover.log.keys", "offered")
    finally:
        con.disconnect()
