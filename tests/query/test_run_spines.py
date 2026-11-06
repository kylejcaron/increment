"""A run fits each warehouse CUPED spine once and keeps nothing after it ends."""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import ibis
import pytest

from increment import View, from_definitions, run
from increment._run_spines import run_spines
from increment.evidence import BreakoutIdentity
from increment.query import _cuped_qr
from increment.query.session import SourceSnapshotEvidence
from tests.analysis_factory import declare_views, held_source
from tests.parity_harness import matrix_data
from tests.query.statement_census import count_statements
from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow(reason="each warehouse CUPED fit issues about 140 statements"),
]

_LATE = dt.datetime(2025, 1, 13, tzinfo=dt.UTC)
# Before the second-day outcome events; the first-day ones keep the fit estimable.
_EARLY = dt.datetime(2025, 1, 11, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    units = matrix_data.linear_cuped_rows(n_per_arm=60)
    events = mixed_cuped_event_rows(units)
    blank = dict.fromkeys(events[0])
    for index, row in enumerate(units):
        events.append(
            {
                **blank,
                "user_id": row["user_id"],
                "ts": dt.datetime(2025, 1, 10, 20),
                "event": "revenue",
                "value": 1.0 + index % 7 / 7,
            }
        )
        # Every fifth unit of each arm never reaches the triggering surface.
        if index % 5:
            events.append(
                {
                    **blank,
                    "user_id": row["user_id"],
                    "ts": dt.datetime(2025, 1, 10, 9),
                    "event": "saw",
                    "experiment_id": "mixed_cuped",
                    "group_id": row["variant"],
                }
            )
    con = ibis.duckdb.connect()
    con.create_table("spine_events", obj=events)
    path = tmp_path_factory.mktemp("spines") / "definitions.yaml"
    path.write_text(
        mixed_cuped_definitions_yaml("spine_events", breakout=True)
        .replace(
            "      - {name: signup, column: null}\n",
            "      - {name: signup, column: null}\n      - {name: saw, column: null}\n",
        )
        .replace(
            "  - {name: enrollment, fact: enrolled}\n",
            "  - {name: enrollment, fact: enrolled}\n  - {name: saw_surface, fact: saw}\n",
        )
        .replace(
            "    exposure: enrollment\n", "    exposure: enrollment\n    trigger: saw_surface\n"
        )
    )
    yield con, path
    con.disconnect()


def _reader(warehouse, cutoff: dt.datetime = _LATE):
    con, path = warehouse
    return from_definitions(
        path,
        con,
        experiment_name="mixed_cuped",
        source_snapshot_evidence=SourceSnapshotEvidence(cutoff, {"spine_events": cutoff}),
    )


def _estimates(readout) -> list[tuple[object, ...]]:
    return sorted(
        (
            row.dimension_value,
            row.group_id,
            row.method,
            row.n_control,
            row.n_treat,
            None if row.lift is None else (row.lift.value, row.lift.lb, row.lift.ub),
            row.failure_code,
        )
        for row in readout
    )


def _tables(con) -> set[tuple[str, str, str]]:
    # In-memory inputs are views Ibis registers and releases with their expressions.
    return set(
        con.raw_sql(
            "SELECT table_catalog, table_schema, table_name FROM information_schema.tables "
            "WHERE table_type <> 'VIEW'"
        ).fetchall()
    )


def test_identical_spine_requests_materialize_once(warehouse):
    con, _ = warehouse
    every = View(view="breakout")
    region = View.breakout((BreakoutIdentity("region", None),))
    experiment, reader = _reader(warehouse)
    experiment = declare_views(experiment, every, region)
    with reader:
        with count_statements(con) as alone:
            run(experiment, reader, views=(every,), metrics=["revenue"])
        with count_statements(con) as shared:
            result = run(experiment, reader, views=(every, region), metrics=["revenue"])
        with count_statements(con) as again:
            run(experiment, reader, views=(every, region), metrics=["revenue"])

    # The second view's cells are the first view's spines; a later run fits its own.
    assert shared.by_kind["create_temp"] == alone.by_kind["create_temp"]
    assert shared.by_kind["drop"] == alone.by_kind["drop"]
    assert again.by_kind == shared.by_kind
    assert _estimates(result.readout(region)) == _estimates(result.readout(every))


def _revenue_fit(con, pinned, population: str):
    config = next(item for item in pinned.context.configs if item.metric.name == "revenue")
    with count_statements(con) as census:
        _, envelope = pinned.fit_cuped_multicovariate(
            config.metric,
            config.decision_method,
            control="control",
            arm_order=("control", "treatment"),
            population=population,
        )
    return envelope.fit, census.by_kind["create_temp"]


def test_spines_differ_by_population_and_horizon(warehouse):
    con, _ = warehouse
    with run_spines():
        _, reader = _reader(warehouse)
        with reader:
            source = cast(Any, held_source(reader))
            metrics = [item.metric for item in source.context.configs]
            with source.readout_snapshot(metrics=metrics, population="assigned") as pinned:
                assigned, assigned_created = _revenue_fit(con, pinned, "assigned")
                triggered, triggered_created = _revenue_fit(con, pinned, "triggered")
        _, earlier_reader = _reader(warehouse, _EARLY)
        with earlier_reader:
            source = cast(Any, held_source(earlier_reader))
            with source.readout_snapshot(metrics=metrics, population="assigned") as pinned:
                earlier, earlier_created = _revenue_fit(con, pinned, "assigned")

    assert assigned_created > 0
    assert triggered_created > 0
    assert earlier_created > 0
    assert assigned.n_by_arm == (60, 60)
    assert triggered.n_by_arm == (48, 48)
    assert earlier.n_by_arm == assigned.n_by_arm
    # The earlier snapshot excludes the second-day revenue events.
    assert all(
        before < after
        for before, after in zip(earlier.adjusted_means, assigned.adjusted_means, strict=True)
    )


def test_failed_run_drops_every_spine(warehouse, monkeypatch):
    con, _ = warehouse
    every = View(view="breakout")
    overall = View(view="run")
    experiment, reader = _reader(warehouse)
    experiment = declare_views(experiment, every, overall)
    summaries = []
    summarize = _cuped_qr.warehouse_qr_unit_summary

    def interrupted(*args, **kwargs):
        summaries.append(None)
        # The breakout view fitted its two cells; the overall view's fit stops mid-way.
        if len(summaries) == 3:
            raise RuntimeError("interrupted fit")
        return summarize(*args, **kwargs)

    monkeypatch.setattr(_cuped_qr, "warehouse_qr_unit_summary", interrupted)
    before = _tables(con)
    with reader, pytest.raises(RuntimeError, match="interrupted fit"):
        run(experiment, reader, views=(every, overall), metrics=["revenue"])

    assert len(summaries) == 3
    assert _tables(con) == before
