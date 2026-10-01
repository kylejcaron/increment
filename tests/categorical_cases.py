"""Shared fixtures for categorical observational adjustment tests.

One confounded data-generating process where a numeric ``spend`` and a
categorical ``region`` both drive assignment and the outcome, served two
ways: the raw ``region`` string column, and the oracle a careful user builds
by hand -- one 0/1 column per non-modal level (``region_west``,
``region_north``), the modal ``east`` as the omitted reference. Every
estimator that accepts the raw column must reproduce the oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

LEVELS = ("east", "west", "north")
# ``east`` is modal by a wide margin, so every training subset agrees on
# the reference level and the fitted encoding is the oracle's own.
LEVEL_SHARES = (0.55, 0.30, 0.15)
DUMMY_COLUMNS = ("region_west", "region_north")


def categorical_units(
    n: int, seed: int, *, effect: float = 0.3, arms: int = 2
) -> dict[str, np.ndarray]:
    """Per-unit arrays: ids, arm, outcome, spend and region, confounded."""
    rng = np.random.default_rng(seed)
    region = rng.choice(np.array(LEVELS), size=n, p=LEVEL_SHARES)
    spend = rng.normal(size=n)
    west = (region == "west").astype(float)
    north = (region == "north").astype(float)
    logit = -0.2 + 0.6 * spend + 0.9 * west - 0.7 * north
    if arms == 2:
        treated = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
        arm = np.where(treated, "T", "C")
        dose = treated.astype(float)
    else:
        score = logit + rng.normal(scale=0.8, size=n)
        cuts = np.quantile(score, np.arange(1, arms) / arms)
        code = np.searchsorted(cuts, score)
        labels = np.array(["C", *(f"T{i}" for i in range(1, arms))])
        arm = labels[code]
        dose = (code > 0).astype(float)
    y = 1.0 + 0.5 * spend + 0.8 * west - 0.4 * north + effect * dose + rng.normal(scale=0.7, size=n)
    return {
        "user_id": np.array([f"u{i}" for i in range(n)]),
        "variant": arm,
        "revenue": y,
        "spend": spend,
        "region": region,
    }


def raw_table(units: dict[str, np.ndarray]) -> pa.Table:
    """The frame a user would actually hold: ``region`` as strings."""
    return pa.table({name: values.tolist() for name, values in units.items()})


def dummy_table(units: dict[str, np.ndarray]) -> pa.Table:
    """The oracle encoding: modal-reference 0/1 dummies in level order."""
    columns: dict[str, Any] = {
        name: values.tolist() for name, values in units.items() if name != "region"
    }
    region = units["region"]
    for level, column in zip(LEVELS[1:], DUMMY_COLUMNS, strict=True):
        columns[column] = (region == level).astype(float).tolist()
    return pa.table(columns)


def with_nulls(table: pa.Table, column: str, positions: list[int]) -> pa.Table:
    """*table* with *column* nulled at *positions*."""
    values = table[column].to_pylist()
    for i in positions:
        values[i] = None
    return table.set_column(table.schema.get_field_index(column), column, pa.array(values))


def assert_rows_match(expected: Any, actual: Any, *, rel: float = 1e-9, skip: tuple[str, ...] = ()):
    """Two public result payloads agree field by field within *rel*;
    string fields under *skip* (notes that name columns) are not compared."""
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if key in skip:
                continue
            assert_rows_match(value, actual[key], rel=rel, skip=skip)
    elif isinstance(expected, list | tuple):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            assert_rows_match(left, right, rel=rel, skip=skip)
    elif isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=rel, abs=1e-12)
    else:
        assert actual == expected
