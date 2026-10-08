"""Trigger membership keeps assignment identity and a true eligible anchor."""

from __future__ import annotations

import datetime as dt

import ibis
import pandas as pd
import pytest

from increment.query.builders import triggered_population

pytestmark = pytest.mark.creates_tables


@pytest.fixture
def con():
    return ibis.duckdb.connect()


def _exposures(con, rows):
    return con.create_table(
        "exposures",
        ibis.memtable(
            {
                "unit_id": [row[0] for row in rows],
                "group_id": [row[1] for row in rows],
                "experiment_id": ["exp"] * len(rows),
                "first_exposure_ts": ["2025-01-05T09:00:00Z"] * len(rows),
            },
            schema={
                "unit_id": "string",
                "group_id": "string",
                "experiment_id": "string",
                "first_exposure_ts": "timestamp('UTC')",
            },
        ),
    )


def _triggers(con, unit_ids):
    if not unit_ids:
        return con.create_table(
            "triggers",
            con.sql(
                "SELECT CAST(NULL AS VARCHAR) AS unit_id, "
                "CAST(NULL AS TIMESTAMPTZ) AS ts WHERE FALSE"
            ),
        )
    return con.create_table(
        "triggers",
        ibis.memtable(
            {
                "unit_id": list(unit_ids),
                "ts": ["2025-01-06T12:00:00Z"] * len(unit_ids),
            },
            schema={"unit_id": "string", "ts": "timestamp('UTC')"},
        ),
    )


def _population(exposures, triggers):
    return triggered_population(
        exposures, triggers, observation_cutoff=dt.datetime(2025, 1, 10, tzinfo=dt.UTC)
    )


def test_keeps_only_units_that_triggered(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T"), ("u3", "C"), ("u4", "T")])
    triggers = _triggers(con, ["u2", "u3"])
    got = _population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"]) == ["u2", "u3"]


def test_preserves_assignment_identity_and_adds_anchor(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, ["u2"])
    got = _population(exposures, triggers).to_pandas()
    assert set(got.columns) == {
        "unit_id",
        "group_id",
        "experiment_id",
        "first_exposure_ts",
        "first_trigger_ts",
    }
    assert got.loc[got["unit_id"] == "u2", "group_id"].item() == "T"


def test_does_not_duplicate_a_unit_that_triggered_twice(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, ["u2", "u2", "u2"])
    assert len(_population(exposures, triggers).to_pandas()) == 1


def test_a_trigger_for_an_unenrolled_unit_is_ignored(con):
    exposures = _exposures(con, [("u1", "C")])
    triggers = _triggers(con, ["u1", "u99"])
    got = _population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"]) == ["u1"]


def test_no_triggers_yields_an_empty_population(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, [])
    assert len(_population(exposures, triggers).to_pandas()) == 0


def test_a_null_unit_id_never_matches(con):
    exposures = _exposures(con, [("u1", "C"), (None, "T")])
    triggers = _triggers(con, ["u1", None])
    got = _population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"].dropna()) == ["u1"]
    assert len(got) == 1


def test_first_eligible_trigger_is_filtered_before_min_and_cutoff_is_inclusive(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = con.create_table(
        "timed_triggers",
        ibis.memtable(
            {
                "unit_id": ["u1", "u1", "u1", "u1", "u2"],
                "ts": [
                    "2025-01-03T08:00:00Z",
                    "2025-01-20T12:00:00Z",
                    "2025-01-20T12:00:00Z",
                    "2025-01-20T12:00:01Z",
                    "2025-01-20T12:00:01Z",
                ],
            },
            schema={"unit_id": "string", "ts": "timestamp('UTC')"},
        ),
    )
    got = triggered_population(
        exposures,
        triggers,
        observation_cutoff=dt.datetime(2025, 1, 20, 12, tzinfo=dt.UTC),
    ).to_pandas()
    assert got[["unit_id", "first_trigger_ts"]].to_dict("records") == [
        {"unit_id": "u1", "first_trigger_ts": pd.Timestamp("2025-01-20T12:00:00Z")}
    ]


def test_preassignment_only_trigger_does_not_enroll(con):
    exposures = _exposures(con, [("u1", "C")])
    triggers = con.create_table(
        "timed_triggers",
        ibis.memtable(
            {"unit_id": ["u1"], "ts": ["2025-01-03T08:00:00Z"]},
            schema={"unit_id": "string", "ts": "timestamp('UTC')"},
        ),
    )
    got = triggered_population(
        exposures,
        triggers,
        observation_cutoff=dt.datetime(2025, 1, 20, tzinfo=dt.UTC),
    ).to_pandas()
    assert got.empty


def test_staggered_cluster_members_keep_assignment_cluster_and_individual_anchor(con):
    exposures = con.create_table(
        "cluster_exposures",
        ibis.memtable(
            {
                "unit_id": ["u1", "u2"],
                "group_id": ["C", "C"],
                "experiment_id": ["exp", "exp"],
                "cluster_id": ["c1", "c1"],
                "first_exposure_ts": ["2025-01-05T09:00:00Z"] * 2,
            },
            schema={
                "unit_id": "string",
                "group_id": "string",
                "experiment_id": "string",
                "cluster_id": "string",
                "first_exposure_ts": "timestamp('UTC')",
            },
        ),
    )
    triggers = con.create_table(
        "cluster_triggers",
        ibis.memtable(
            {
                "unit_id": ["u1", "u2"],
                "ts": ["2025-01-06T12:00:00Z", "2025-01-09T12:00:00Z"],
            },
            schema={"unit_id": "string", "ts": "timestamp('UTC')"},
        ),
    )
    rows = triggered_population(
        exposures,
        triggers,
        observation_cutoff=dt.datetime(2025, 1, 10, tzinfo=dt.UTC),
    ).to_pandas()
    assert rows.set_index("unit_id")["cluster_id"].to_dict() == {"u1": "c1", "u2": "c1"}
    anchors = rows.set_index("unit_id")["first_trigger_ts"]
    assert anchors["u1"] != anchors["u2"]
