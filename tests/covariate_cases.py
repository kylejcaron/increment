"""Shared warehouse fixtures for per-unit covariate tests.

Each unit carries a numeric ``tenure`` recorded before its exposure, so the
same per-unit truth can be fed to ``from_definitions`` and to the dataframe
oracle ``from_unit_summary``. `categorical_defs_and_con` adds a categorical
``region`` (a string property) beside it, optionally missing on some units.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import ibis

_OBSERVATIONAL_DESIGN = """    design:
      mechanism: observational
      covariates:
        - {property: tenure, source: events}
"""


def _defs_yaml(table: str, experiment: str, *, observational: bool) -> str:
    design = _OBSERVATIONAL_DESIGN if observational else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM {table}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: exposure, column: null}}
      - {{name: purchase, column: revenue}}
    properties:
      - {{name: tenure, column: tenure, dtype: float, as_of: pre_exposure}}
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: {experiment}
    exposure: enrolled
    unit: user_id
    control_group: control
    start: "2025-08-01"
    end: "2025-08-07"
    plan: {{secondaries: [revenue]}}
{design}"""


def _unit_rows(
    experiment: str, unit: str, arm: str, tenure: float | None, revenue: float
) -> list[dict[str, Any]]:
    """Exposure, a strictly pre-exposure property row, one in-window purchase,
    and one purchase after the window so every unit's window is observed."""
    return [
        {
            "user_id": unit,
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "exposure",
            "experiment_id": experiment,
            "group_id": arm,
            "revenue": None,
            "tenure": None,
        },
        {
            "user_id": unit,
            "ts": dt.datetime(2025, 7, 20, 0, 0, 0),
            "event": "exposure",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "tenure": tenure,
        },
        {
            "user_id": unit,
            "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": revenue,
            "tenure": None,
        },
        {
            "user_id": unit,
            "ts": dt.datetime(2025, 8, 20, 0, 0, 0),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "revenue": 100.0,
            "tenure": None,
        },
    ]


def covariate_defs_and_con(
    tmp_path: Path, *, observational: bool = False
) -> tuple[Path, Any, dict[str, float], dict[str, str], dict[str, float]]:
    """20 units, two arms, a deterministic tenure per unit.

    Returns ``(defs_path, con, tenure, group, revenue)`` keyed by unit id.
    """
    rows: list[dict[str, Any]] = []
    tenure: dict[str, float] = {}
    group: dict[str, str] = {}
    revenue: dict[str, float] = {}
    for i, arm in enumerate(["control"] * 10 + ["treatment"] * 10):
        unit = f"u{i}"
        tenure[unit] = 10.0 + i
        group[unit] = arm
        revenue[unit] = 5.0 + 0.1 * i + 0.03 * (i % 4) * tenure[unit]
        rows.extend(_unit_rows("cov_test", unit, arm, tenure[unit], revenue[unit]))
    con = ibis.duckdb.connect()
    con.create_table("cov_events", obj=rows)
    defs_path = tmp_path / "cov_defs.yaml"
    defs_path.write_text(_defs_yaml("cov_events", "cov_test", observational=observational))
    return defs_path, con, tenure, group, revenue


def confounded_defs_and_con(
    tmp_path: Path,
) -> tuple[Path, Any, dict[str, float], dict[str, str], dict[str, float]]:
    """Observational design where tenure drives both treatment and outcome.

    Returns ``(defs_path, con, tenure, group, revenue)`` keyed by unit id.
    """
    import numpy as np

    rng = np.random.default_rng(3)
    rows: list[dict[str, Any]] = []
    tenure: dict[str, float] = {}
    group: dict[str, str] = {}
    revenue: dict[str, float] = {}
    for i in range(40):
        unit = f"u{i}"
        t = float(rng.normal(100, 15))
        arm = "treatment" if (t > 100) == (i % 3 != 0) else "control"
        tenure[unit] = t
        group[unit] = arm
        revenue[unit] = (
            20.0 + 0.05 * t + (2.0 if arm == "treatment" else 0.0) + float(rng.normal(0, 1))
        )
        rows.extend(_unit_rows("obs_test", unit, arm, t, revenue[unit]))
    con = ibis.duckdb.connect()
    con.create_table("obs_events", obj=rows)
    defs_path = tmp_path / "obs_defs.yaml"
    defs_path.write_text(_defs_yaml("obs_events", "obs_test", observational=True))
    return defs_path, con, tenure, group, revenue


_CATEGORICAL_DESIGN = """    design:
      mechanism: observational
      covariates:
        - {property: tenure, source: events}
        - {property: region, source: events}
"""

CATEGORICAL_LEVELS = ("east", "west", "north")


def _categorical_defs_yaml(table: str, experiment: str) -> str:
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM {table}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: exposure, column: null}}
      - {{name: purchase, column: revenue}}
    properties:
      - {{name: tenure, column: tenure, dtype: float, as_of: pre_exposure}}
      - {{name: region, column: region, dtype: string, as_of: pre_exposure}}
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: {experiment}
    exposure: enrolled
    unit: user_id
    control_group: control
    start: "2025-08-01"
    end: "2025-08-07"
    plan: {{primary: revenue}}
{_CATEGORICAL_DESIGN}"""


def _categorical_unit_rows(
    experiment: str, unit: str, arm: str, tenure: float, region: str | None, revenue: float
) -> list[dict[str, Any]]:
    """`_unit_rows` with a ``region`` column: the pre-exposure record carries
    the unit's level (NULL when missing), every other record none."""
    rows = _unit_rows(experiment, unit, arm, tenure, revenue)
    for row in rows:
        row["region"] = None
    rows[1]["region"] = region
    return rows


def categorical_defs_and_con(
    tmp_path: Path, *, null_units: tuple[str, ...] = ()
) -> tuple[Path, Any, dict[str, dict[str, Any]]]:
    """80 units, two arms; a numeric ``tenure`` and a categorical ``region``
    (modal ``east``) both nudge each unit's arm and its outcome, so an
    adjustment that mishandles either moves the estimate. Units named in
    *null_units* have no pre-exposure ``region`` (NULL).

    Returns ``(defs_path, con, units)``; ``units`` maps unit id to the
    per-unit truth (``variant``, ``tenure``, ``region``, ``revenue``) a
    dataframe oracle reads directly.
    """
    import numpy as np

    rng = np.random.default_rng(29)
    rows: list[dict[str, Any]] = []
    units: dict[str, dict[str, Any]] = {}
    for i in range(80):
        unit = f"u{i}"
        tenure = float(rng.normal(100, 15))
        region = CATEGORICAL_LEVELS[int(rng.choice(3, p=(0.55, 0.30, 0.15)))]
        west, north = float(region == "west"), float(region == "north")
        score = 0.02 * (tenure - 100) + 0.5 * west - 0.4 * north + float(rng.normal(0, 1.0))
        arm = "treatment" if score > 0 else "control"
        revenue = (
            20.0
            + 0.05 * tenure
            + 1.5 * west
            - 0.8 * north
            + (2.0 if arm == "treatment" else 0.0)
            + float(rng.normal(0, 1))
        )
        units[unit] = {
            "variant": arm,
            "tenure": tenure,
            "region": None if unit in null_units else region,
            "revenue": revenue,
        }
        rows.extend(
            _categorical_unit_rows("cat_test", unit, arm, tenure, units[unit]["region"], revenue)
        )
    con = ibis.duckdb.connect()
    con.create_table("cat_events", obj=rows)
    defs_path = tmp_path / "cat_defs.yaml"
    defs_path.write_text(_categorical_defs_yaml("cat_events", "cat_test"))
    return defs_path, con, units
