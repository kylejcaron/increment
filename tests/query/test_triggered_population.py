"""Narrowing the analysis population to units that could be treated.

A trigger is a second first-occurrence fact. `triggered_population` keeps
only the enrolled units that also produced a trigger, leaving every
enrollment column (arm, first exposure timestamp) untouched - everything
downstream reads those, so the narrowed table has to be substitutable for
the unnarrowed one.
"""

from __future__ import annotations

import ibis
import pytest

from increment.query.builders import triggered_population

# Every test here gets its own fresh, isolated `con` (see the fixture
# below) - tables can never leak between tests.
pytestmark = pytest.mark.creates_tables


@pytest.fixture
def con():
    return ibis.duckdb.connect()


def _exposures(con, rows):
    return con.create_table(
        "exposures",
        ibis.memtable(
            {
                "unit_id": [r[0] for r in rows],
                "group_id": [r[1] for r in rows],
                "experiment_id": ["exp"] * len(rows),
            }
        ),
    )


def _triggers(con, unit_ids):
    return con.create_table(
        "triggers",
        ibis.memtable(
            {"unit_id": list(unit_ids), "group_id": ["x"] * len(unit_ids)},
            schema={"unit_id": "string", "group_id": "string"},
        ),
    )


def test_keeps_only_units_that_triggered(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T"), ("u3", "C"), ("u4", "T")])
    triggers = _triggers(con, ["u2", "u3"])
    got = triggered_population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"]) == ["u2", "u3"]


def test_preserves_the_enrollment_columns(con):
    """Arm assignment and every other enrollment column must survive: the
    narrowed table replaces the unnarrowed one everywhere downstream."""
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, ["u2"])
    got = triggered_population(exposures, triggers).to_pandas()
    assert set(got.columns) == {"unit_id", "group_id", "experiment_id"}
    assert got.loc[got["unit_id"] == "u2", "group_id"].item() == "T"


def test_does_not_duplicate_a_unit_that_triggered_twice(con):
    """A semi-join, not an inner join: two trigger rows for one unit must
    not double that unit's weight in every downstream moment."""
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, ["u2", "u2", "u2"])
    got = triggered_population(exposures, triggers).to_pandas()
    assert len(got) == 1


def test_a_trigger_for_an_unenrolled_unit_is_ignored(con):
    """Triggering without being enrolled is not analyzable - there is no
    arm to attribute the unit to."""
    exposures = _exposures(con, [("u1", "C")])
    triggers = _triggers(con, ["u1", "u99"])
    got = triggered_population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"]) == ["u1"]


def test_no_triggers_yields_an_empty_population(con):
    exposures = _exposures(con, [("u1", "C"), ("u2", "T")])
    triggers = _triggers(con, [])
    assert len(triggered_population(exposures, triggers).to_pandas()) == 0


def test_a_null_unit_id_never_matches(con):
    """A null unit_id has no arm to attribute a trigger to - it must
    drop out of the triggered population even when both sides carry a
    null-keyed row, matching the semi-join's `=` semantics."""
    exposures = _exposures(con, [("u1", "C"), (None, "T")])
    triggers = _triggers(con, ["u1", None])
    got = triggered_population(exposures, triggers).to_pandas()
    assert sorted(got["unit_id"].dropna()) == ["u1"]
    assert len(got) == 1
