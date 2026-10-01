"""Per-unit covariate join: numeric casts to float64, categorical stays a
string; neither ever string-coalesces a missing value."""

from __future__ import annotations

import ibis
import pytest

from increment.query.builders import join_unit_covariates


def _joined(
    rows: list[dict[str, object]],
    props: list[dict[str, object]],
    name: str,
    *,
    categorical: bool = False,
):
    con = ibis.duckdb.connect()
    table = con.create_table("t", obj=rows)
    properties = con.create_table("p", obj=props)
    joined = join_unit_covariates(table, properties, name, categorical=categorical)
    return {r["unit_id"]: r for r in con.to_pyarrow(joined).to_pylist()}


def test_join_unit_covariates_casts_numeric_and_preserves_null():
    rows = _joined([{"unit_id": "a"}, {"unit_id": "b"}], [{"unit_id": "a", "tenure": 12}], "tenure")
    assert rows["a"]["tenure"] == pytest.approx(12.0)
    assert isinstance(rows["a"]["tenure"], float)
    assert rows["b"]["tenure"] is None


def test_join_unit_covariates_casts_bool_to_zero_one():
    rows = _joined(
        [{"unit_id": "a"}, {"unit_id": "b"}],
        [{"unit_id": "a", "activated": True}, {"unit_id": "b", "activated": False}],
        "activated",
    )
    assert rows["a"]["activated"] == pytest.approx(1.0)
    assert rows["b"]["activated"] == pytest.approx(0.0)


def test_join_unit_covariates_takes_the_property_value_over_a_same_named_column():
    rows = _joined([{"unit_id": "a", "tenure": -1.0}], [{"unit_id": "a", "tenure": 3.0}], "tenure")
    assert rows["a"]["tenure"] == pytest.approx(3.0)


def test_join_unit_covariates_keeps_a_categorical_level_as_string_and_null():
    """A categorical covariate keeps its level label verbatim (case, digits
    and all) and a unit without a value stays NULL -- never a sentinel
    level, never a number."""
    rows = _joined(
        [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}],
        [{"unit_id": "a", "region": "West"}, {"unit_id": "b", "region": "7"}],
        "region",
        categorical=True,
    )
    assert rows["a"]["region"] == "West"
    assert rows["b"]["region"] == "7"
    assert isinstance(rows["b"]["region"], str)
    assert rows["c"]["region"] is None
