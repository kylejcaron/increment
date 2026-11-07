"""A run fits each warehouse CUPED spine once and keeps nothing after it ends."""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ibis
import pytest

from increment import View, from_definitions, run
from increment.errors import IncrementWarning
from increment.evidence import BreakoutIdentity
from increment.query.session import SourceSnapshotEvidence
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import declare_views
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


@backend_fixture(scope="module")
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


def _cells(readout) -> dict[object, Any]:
    return {row.dimension_value: row for row in readout if row.method == "linear"}


def test_spines_differ_by_population_and_horizon(warehouse):
    con, _ = warehouse
    assigned_view = View(view="breakout", population="assigned")
    triggered_view = View(view="breakout", population="triggered")
    experiment, reader = _reader(warehouse)
    experiment = declare_views(experiment, assigned_view, triggered_view)
    with reader:
        with count_statements(con) as assigned_only:
            run(experiment, reader, views=(assigned_view,), metrics=["revenue"])
        # The triggered east cell's 24 units per arm fall outside the fit's support region;
        # its exclusion advisory is asserted below through the cell's failure code.
        with count_statements(con) as both, warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            result = run(
                experiment, reader, views=(assigned_view, triggered_view), metrics=["revenue"]
            )
    # One run reads one capture, so an earlier as-of is read by its own run.
    earlier_experiment, earlier_reader = _reader(warehouse, _EARLY)
    earlier_experiment = declare_views(earlier_experiment, assigned_view)
    with earlier_reader, count_statements(con) as earlier_run:
        earlier_result = run(
            earlier_experiment, earlier_reader, views=(assigned_view,), metrics=["revenue"]
        )
    assigned = _cells(result.readout(assigned_view))
    triggered = _cells(result.readout(triggered_view))
    earlier = _cells(earlier_result.readout(assigned_view))

    # The triggered view fits its own spines in the run that already fitted the assigned ones.
    assert both.by_kind["create_temp"] > assigned_only.by_kind["create_temp"]
    assert earlier_run.by_kind["create_temp"] > 0
    assert {cell: (row.n_control, row.n_treat) for cell, row in assigned.items()} == {
        "east": (30, 30),
        "west": (30, 30),
    }
    # Every fifth unit of each arm, alternating between regions, is untriggered.
    assert triggered["east"].failure_code == "estimation.cuped_linear.support_region"
    assert (triggered["west"].n_control, triggered["west"].n_treat) == (24, 24)
    assert triggered["west"].lift.value != assigned["west"].lift.value
    for cell, row in earlier.items():
        assert (row.n_control, row.n_treat) == (30, 30)
        # The earlier snapshot excludes the second-day revenue events, which carry the effect.
        assert abs(row.abs_diff) < abs(assigned[cell].abs_diff)


class _LosingConnection:
    """Delegate to the warehouse connection, failing at one chosen statement."""

    def __init__(self, connection: Any, marker: str, occurrence: int, hits: list[str]):
        self._connection = connection
        self._marker = marker
        self._occurrence = occurrence
        self._hits = hits

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._connection, name)
        if name not in {"execute", "sql"} or not callable(attribute):
            return attribute

        def checked(*args: Any, **kwargs: Any) -> Any:
            if args and self._marker in str(args[0]):
                self._hits.append(str(args[0]))
                if len(self._hits) == self._occurrence:
                    raise RuntimeError("warehouse connection lost")
            result = attribute(*args, **kwargs)
            if result is self._connection:
                return _LosingConnection(result, self._marker, self._occurrence, self._hits)
            return result

        return checked


@contextmanager
def _connection_lost_at(backend: Any, marker: str, occurrence: int) -> Iterator[list[str]]:
    original = backend.con
    hits: list[str] = []
    backend.con = _LosingConnection(original, marker, occurrence, hits)
    try:
        yield hits
    finally:
        backend.con = original


def test_failed_run_drops_every_spine(warehouse):
    con, _ = warehouse
    every = View(view="breakout")
    overall = View(view="run")
    experiment, reader = _reader(warehouse)
    experiment = declare_views(experiment, every, overall)
    with reader:
        before = _tables(con)
        # Each fit reads its leverage summary once, after its work relations exist: the
        # breakout view fits its two cells and the overall view's fit stops mid-way.
        with (
            _connection_lost_at(con, "_qr_max_leverage", 3) as hits,
            pytest.raises(RuntimeError, match="warehouse connection lost"),
        ):
            run(experiment, reader, views=(every, overall), metrics=["revenue"])
        after = _tables(con)

    assert len(hits) == 3
    assert after == before
