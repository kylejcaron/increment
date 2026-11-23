from __future__ import annotations

import ibis
import pytest

from increment import from_definitions, run
from increment.errors import CodedError
from tests.spillover_trial import (
    WAREHOUSE_ROLES,
    complete_receipt,
    spillover_rows,
    three_policy_trial,
    write_study,
)

pytestmark = pytest.mark.integration


def _trial(con, path):
    design, frames = three_policy_trial()
    write_study(path, design, {role: role for role in WAREHOUSE_ROLES})
    for role in WAREHOUSE_ROLES:
        if role != "purchases":
            con.create_table(role, frames[role])
    return frames["purchases"]


def test_nested_view_registration_follows_files_not_attachment_aliases(tmp_path):
    path = tmp_path / "study.yaml"
    database = tmp_path / "trial.duckdb"
    original = tmp_path / "original.duckdb"
    other = tmp_path / "other.duckdb"
    con = ibis.duckdb.connect(database)
    try:
        purchases = _trial(con, path)
        for file in (original, other):
            con.attach(file, name="events")
            con.create_table("raw_sales", purchases, database=("events", "main"))
            con.detach("events")
        con.attach(original, name="events")
        con.raw_sql("CREATE VIEW ledger AS SELECT * FROM events.main.raw_sales")
        con.raw_sql(
            "CREATE VIEW purchases AS WITH raw_sales AS "
            "(SELECT * FROM trial.main.ledger) SELECT * FROM raw_sales"
        )
        experiment, reader = from_definitions(path, con, experiment_name="pilot")
        with reader:
            registration = experiment.registration
    finally:
        con.disconnect()

    con = ibis.duckdb.connect(database)
    try:
        con.attach(original, name="events")
        rebound, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            assert rebound.registration == registration
            estimates = [row.estimate for row in spillover_rows(run(experiment, reader))]
            assert estimates == pytest.approx((19 / 630, 101 / 210), rel=0, abs=1e-14)
            con.detach("events")
            con.attach(other, name="events")
            with pytest.raises(CodedError) as changed:
                run(experiment, reader)
            assert changed.value.code == "spillover.roster.changed"
        _, reader = from_definitions(path, con, experiment_name="pilot")
        with reader, pytest.raises(CodedError) as stale:
            run(experiment, reader)
        assert stale.value.code == "kernel.run.execution_unavailable"
    finally:
        con.disconnect()


def test_late_observation_view_cannot_introduce_another_database(tmp_path):
    path = tmp_path / "study.yaml"
    con = ibis.duckdb.connect()
    try:
        purchases = _trial(con, path)
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            con.attach(tmp_path / "events.duckdb", name="events")
            con.create_table("raw_sales", purchases, database=("events", "main"))
            con.raw_sql("CREATE VIEW purchases AS SELECT * FROM events.main.raw_sales")
            with pytest.raises(CodedError) as changed:
                run(experiment, reader)
            assert changed.value.code == "spillover.roster.changed"
    finally:
        con.disconnect()


def test_unqualified_view_ingredient_cannot_be_shadowed_by_a_temporary_table(tmp_path):
    path = tmp_path / "study.yaml"
    con = ibis.duckdb.connect()
    try:
        purchases = _trial(con, path)
        con.create_table("raw_sales", purchases)
        con.raw_sql("CREATE VIEW purchases AS SELECT * FROM raw_sales")
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            con.raw_sql("CREATE TEMP TABLE raw_sales AS SELECT * FROM main.raw_sales LIMIT 0")
            with pytest.raises(CodedError) as changed:
                run(experiment, reader)
            assert changed.value.code == "spillover.roster.changed"
        with pytest.raises(CodedError) as ambiguous:
            from_definitions(path, con, experiment_name="pilot")
        assert ambiguous.value.code == "spillover.design.invalid"
        assert ambiguous.value.context["field"] == "relations"
    finally:
        con.disconnect()


def test_redefined_view_in_same_database_invalidates_saved_registration(tmp_path):
    path = tmp_path / "study.yaml"
    con = ibis.duckdb.connect()
    try:
        purchases = _trial(con, path)
        con.create_table("raw_sales", purchases)
        con.raw_sql("CREATE VIEW purchases AS SELECT * FROM memory.main.raw_sales")
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            con.raw_sql(
                "CREATE OR REPLACE VIEW purchases AS "
                "SELECT * FROM memory.main.raw_sales WHERE channel != 'web'"
            )
            with pytest.raises(CodedError) as changed:
                run(experiment, reader)
            assert changed.value.code == "spillover.roster.changed"
        _, reader = from_definitions(path, con, experiment_name="pilot")
        with reader, pytest.raises(CodedError) as stale:
            run(experiment, reader)
        assert stale.value.code == "kernel.run.execution_unavailable"
    finally:
        con.disconnect()
