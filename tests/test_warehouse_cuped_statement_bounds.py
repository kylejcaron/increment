"""Warehouse statements issued by one CUPED QR readout and by one dashboard snapshot."""

from __future__ import annotations

import collections
import datetime as dt
from typing import Any, cast

import ibis
import numpy as np
import pytest
import yaml
from ibis.backends.duckdb import Backend

from increment import from_definitions
from tests.analysis_factory import view_readout
from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml

pytestmark = pytest.mark.integration

# 6 reads outside the fit, and 28 in it: 1 range check + 5 encoder + 4 outcome
# + 1 factor + 1 denominator + 1 leverage + 1 score covariance + 1 U
# + 5 projection scales + 8 BM statements (18 jobs of 508 sums, at most 1,199
# sums beside the group key per statement).
THREE_ARM_RATIO_FETCHES = 34


def _ratio_events(names: tuple[str, ...], units_per_arm: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(20261103)
    exposure = dt.datetime(2025, 1, 10, 8)
    pre = dt.datetime(2025, 1, 7, 9)
    post = dt.datetime(2025, 1, 11, 9)
    empty = dict.fromkeys(names)
    events: list[dict[str, object]] = []
    for arm_index, arm in enumerate(("control", "variant-a", "variant-b")):
        for index in range(units_per_arm):
            values = rng.normal(size=len(names))
            unit = f"{arm}-{index}"
            revenue = 12.0 + arm_index * 0.4 + 0.7 * values[0] + rng.normal(scale=0.3)
            common = {"user_id": unit, "experiment_id": None, "group_id": None, "value": None}
            events.extend(
                (
                    {**common, **empty, "ts": exposure, "event": "enrolled"}
                    | {"experiment_id": "mixed_cuped", "group_id": arm},
                    {**common, "ts": pre, "event": "history"}
                    | dict(zip(names, values.tolist(), strict=True)),
                    {**common, **empty, "ts": post, "event": "revenue", "value": revenue},
                )
            )
            events.extend(
                {**common, **empty, "ts": post + dt.timedelta(minutes=minute), "event": "order"}
                for minute in range(5 + index % 4)
            )
    return events


def test_qr_three_arm_ratio_fetches_at_most(tmp_path, monkeypatch):
    arms = ("control", "variant-a", "variant-b")
    names = tuple(f"x{index}" for index in range(8))
    definitions = yaml.safe_load(mixed_cuped_definitions_yaml("qr_statement_bounds"))
    # The events carry only these covariates, so they are the source's declared properties.
    definitions["fact_sources"][0]["properties"] = [
        {"name": name, "column": name, "dtype": "float", "as_of": "pre_exposure"} for name in names
    ]
    experiment_data = definitions["experiments"][0]
    experiment_data["allocation"] = dict.fromkeys(arms, 1 / len(arms))
    method = experiment_data["plan"]["guardrails"][0]["decision_method"]
    method["covariates"] = [
        {"name": name, "kind": "continuous", "property": name, "source": "events"} for name in names
    ]
    experiment_data["plan"] = {"primary": {"metric": "rpo", "decision_method": method}}
    path = tmp_path / "qr-statement-bounds.yaml"
    path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    connection = ibis.duckdb.connect()
    connection.create_table("qr_statement_bounds", obj=_ratio_events(names, 40))
    experiment, resolver = from_definitions(path, connection, experiment_name="mixed_cuped")
    fetch = Backend.to_pyarrow
    fetches = 0

    def counted(self, *args, **kwargs):
        nonlocal fetches
        fetches += 1
        return fetch(self, *args, **kwargs)

    monkeypatch.setattr(Backend, "to_pyarrow", counted)
    try:
        view_readout(experiment, resolver, metrics=["rpo"])
    finally:
        cast(Any, resolver).close()
        connection.disconnect()
    assert fetches <= THREE_ARM_RATIO_FETCHES


# Small fixtures trip estimator reliability warnings; the subject is which data is read.
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")
def test_one_dashboard_snapshot_reads_each_statement_once(tmp_path, monkeypatch):
    from increment.dashboard import prepare_dashboard
    from tests.test_dashboard_capture import CONFIG, _workspace

    fetch = Backend.to_pyarrow
    statements = collections.Counter()

    def counted(self, expression, *args, **kwargs):
        statements[ibis.to_sql(expression, dialect="duckdb")] += 1
        return fetch(self, expression, *args, **kwargs)

    with _workspace(tmp_path, units=20) as (_, experiment, resolver):
        monkeypatch.setattr(Backend, "to_pyarrow", counted)
        prepare_dashboard(experiment, resolver, config=CONFIG)
    assert statements
    assert {sql: count for sql, count in statements.items() if count > 1} == {}


_ALONE = """
    plan:
      primary: checkout_conversion
"""
_AMONG_OTHERS = """
    plan:
      primary: checkout_conversion
      secondaries:
        - revenue_per_user
        - session_seconds
"""


def _dashboard_reads_of(metric: str, tmp_path, monkeypatch, *, plan: str) -> int:
    """Warehouse statements one dashboard snapshot issues that read ``metric``'s evidence."""
    from increment.dashboard import prepare_dashboard
    from tests.test_dashboard_capture import CONFIG, METRICS, _workspace

    fetch = Backend.to_pyarrow
    reads = 0

    def counted(self, expression, *args, **kwargs):
        nonlocal reads
        reads += f"'{metric}'" in ibis.to_sql(expression, dialect="duckdb")
        return fetch(self, expression, *args, **kwargs)

    tmp_path.mkdir()
    with (
        _workspace(tmp_path, metrics=METRICS, plan=plan) as (_, experiment, resolver),
        monkeypatch.context() as patch,
    ):
        patch.setattr(Backend, "to_pyarrow", counted)
        prepare_dashboard(experiment, resolver, config=CONFIG)
    return reads


# Small fixtures trip estimator reliability warnings; the subject is which data is read.
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")
def test_a_dashboard_reads_a_metric_as_often_alone_as_among_others(tmp_path, monkeypatch):
    """Each Explore view reads a metric's evidence once, however many metrics are declared."""
    metric = "checkout_conversion"
    alone = _dashboard_reads_of(metric, tmp_path / "alone", monkeypatch, plan=_ALONE)
    among = _dashboard_reads_of(metric, tmp_path / "among", monkeypatch, plan=_AMONG_OTHERS)
    assert alone > 0
    assert among == alone
