from __future__ import annotations

import ibis
import pytest
import yaml

from increment import from_definitions, run
from increment.errors import CodedError
from tests._unit_boundary import backend_fixture
from tests.spillover_trial import (
    WAREHOUSE_ROLES,
    complete_receipt,
    spillover_rows,
    three_policy_trial,
    write_study,
)

pytestmark = pytest.mark.integration


@backend_fixture()
def trial(tmp_path):
    db = ibis.duckdb.connect(threads=1)
    design, frames = three_policy_trial()
    study = tmp_path / "study.yaml"
    write_study(study, design, {r: f"memory.main.{r}" for r in WAREHOUSE_ROLES})
    declarations = yaml.safe_load(study.read_text())

    def bind(sql):
        declarations["fact_sources"][0]["sql"] = sql
        study.write_text(yaml.safe_dump(declarations))
        return from_definitions(
            study, db, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )

    try:
        for role in WAREHOUSE_ROLES:
            if role != "purchases":
                db.create_table(role, frames[role])
        yield db, frames["purchases"], bind
    finally:
        db.disconnect()


def _assert_registered_purchases(experiment, reader):
    with reader:
        rows = spillover_rows(run(experiment, reader))
    assert rows[1].estimate == pytest.approx(0.480952380952380909, rel=0, abs=1e-14)


def test_registered_purchase_filters_do_not_follow_the_current_schema(trial):
    db, purchases, bind = trial
    db.raw_sql("CREATE SCHEMA original; CREATE SCHEMA other; SET schema = 'original'")
    db.create_table("purchases", purchases)
    db.raw_sql("CREATE TABLE included AS SELECT person_id FROM purchases")
    db.raw_sql("CREATE TABLE excluded (person_id VARCHAR)")
    db.raw_sql("CREATE TABLE other.included (person_id VARCHAR)")
    db.raw_sql("CREATE TABLE other.excluded AS SELECT person_id FROM purchases")
    experiment, reader = bind(
        "SELECT purchases.*, 'coupon_purchase' AS event FROM purchases "
        "SEMI JOIN included USING (person_id) ANTI JOIN excluded USING (person_id)"
    )
    db.raw_sql("SET schema = 'other'")
    _assert_registered_purchases(experiment, reader)


def test_schema_qualified_columns_disambiguate_identically_named_join_sources(trial):
    db, purchases, bind = trial
    db.raw_sql("CREATE SCHEMA east; CREATE SCHEMA west")
    db.create_table("events", purchases, database="east")
    db.raw_sql("CREATE TABLE west.events AS SELECT DISTINCT person_id FROM east.events")
    study = bind(
        "WITH Events AS (SELECT * FROM west.events) "
        "SELECT east.events.person_id, east.events.purchased_at, east.events.channel, "
        "'coupon_purchase' AS event FROM east.events "
        "JOIN west.events ON east.events.person_id = west.events.person_id "
        "JOIN events c ON east.events.person_id = c.person_id"
    )
    _assert_registered_purchases(*study)


def test_search_path_binds_the_first_schema_containing_the_table(trial):
    db, purchases, bind = trial
    db.raw_sql("CREATE SCHEMA first; CREATE SCHEMA second; CREATE SCHEMA third")
    db.create_table("purchases", purchases, database="second")
    db.raw_sql("CREATE TABLE third.purchases AS FROM second.purchases LIMIT 0")
    db.raw_sql("SET search_path = 'first,second,third'")
    study = bind("SELECT *, 'coupon_purchase' AS event FROM purchases")
    db.raw_sql("CREATE TABLE first.purchases AS FROM third.purchases; SET schema = 'third'")
    _assert_registered_purchases(*study)


def test_two_part_attached_catalog_accepts_a_future_table(trial):
    db, purchases, bind = trial
    db.raw_sql("ATTACH ':memory:' AS feed")
    study = bind("SELECT *, 'coupon_purchase' AS event FROM feed.purchases")
    db.create_table("purchases", purchases, database=("feed", "main"))
    _assert_registered_purchases(*study)


def test_ambiguous_catalog_and_schema_name_refuses_during_binding(trial):
    db, _, bind = trial
    db.raw_sql("ATTACH ':memory:' AS feed; CREATE SCHEMA memory.feed")
    with pytest.raises(CodedError) as exc:
        bind("SELECT *, 'coupon_purchase' AS event FROM feed.purchases")
    assert exc.value.code == "spillover.design.invalid"


def test_cross_catalog_scalar_macro_refuses_at_registration(trial):
    db, _, bind = trial
    db.raw_sql("ATTACH ':memory:' AS feed")
    db.raw_sql("CREATE TABLE feed.main.channels AS SELECT 'web' AS channel")
    db.raw_sql("CREATE MACRO measured_channel() AS (SELECT channel FROM feed.main.channels)")
    with pytest.raises(CodedError) as exc:
        bind(
            "SELECT person_id, purchased_at, measured_channel() AS channel, "
            "'coupon_purchase' AS event FROM memory.main.purchases"
        )
    assert exc.value.code == "spillover.design.invalid"


def test_session_macro_cannot_replace_a_registered_builtin(trial):
    db, purchases, bind = trial
    db.create_table("purchases", purchases)
    experiment, reader = bind(
        "SELECT *, 'coupon_purchase' AS event FROM purchases WHERE ends_with(channel, 'eb')"
    )
    db.raw_sql("CREATE SCHEMA preview; CREATE MACRO preview.ends_with(x, y) AS false")
    db.raw_sql("SET schema = 'preview'")
    with reader, pytest.raises(CodedError) as exc:
        run(experiment, reader)
    assert exc.value.code == "spillover.roster.changed"
