"""Explore data is captured with the headline, not read from the warehouse afterwards.

``prepare_dashboard`` pins the warehouse once. Every Explore choice the dashboard offers must
come from that same pinned read, so a later warehouse change cannot make the headline and the
trajectories describe different data, and a re-preparation must deliberately see the new data.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import random
import re
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.breakout.estimates import BreakoutEstimates, DailyLiftEstimates, DailyMetricValues
from increment.dashboard import DashboardConfig, load_explore, prepare_dashboard
from increment.dashboard._app import build_payload
from increment.errors import CodedError
from tests.test_dashboard import STOREFRONT_WINDOW, _arm_rows

# Small deterministic fixtures legitimately trip estimator reliability warnings; the subject
# here is which data is read, not estimator reliability.
pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

CONFIG = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})

_EVENT_LOG_SOURCE = """
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
      - name: purchase
        column: revenue
      - name: session_end
        column: duration_s
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
"""

# The breakout names the `profiles` source, which no metric fact reads from: its stream is only
# available to a pinned read if the pin is told to capture it.
_PROFILE_SOURCES = """
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
      - name: purchase
        column: revenue
      - name: session_end
        column: duration_s
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
  - name: profiles
    sql: |
      SELECT * FROM analytics.profiles
    timestamp_column: updated_at
    entities:
      - user_id
    facts:
      - name: profile_seen
        column: null
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
"""

_EXPOSURES = """
exposures:
  - name: first_page_view
    fact: page_view
"""

_METRICS = """
metrics:
  - type: conversion
    name: checkout_conversion
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14

  - type: mean
    name: revenue_per_user
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14

  - type: mean
    name: session_seconds
    entity: user_id
    preferred_direction: decrease
    fact: session_end
    aggregation: sum
    window_days: 14

  - type: retention
    name: ever_purchased_again
    entity: user_id
    preferred_direction: increase
    fact: purchase
    threshold_days: 1
"""

_PLAN = """
    plan:
      view_multiplicity:
        correction: bonferroni
      primary: checkout_conversion
      secondaries:
        - revenue_per_user
        - ever_purchased_again
      guardrails:
        - session_seconds
"""


def _experiments(*, breakout_sources: tuple[str | None, ...]) -> str:
    breakout = "".join(
        "      - property: country\n" + ("" if source is None else f"        source: {source}\n")
        for source in breakout_sources
    )
    if breakout:
        breakout = f"    breakouts:\n{breakout}"
    return f"""
experiments:
  - name: storefront_refresh
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-14
    n_pre_periods: 0
    control_group: control
{_PLAN}{breakout}"""


def _event_rows(units: int) -> list[dict[str, Any]]:
    rng = random.Random(5)
    rows = _arm_rows(
        rng,
        experiment_id="storefront_refresh",
        group_id="control",
        prefix="sc",
        units=units,
        start=STOREFRONT_WINDOW[0],
        convert_rate=0.3,
    ) + _arm_rows(
        rng,
        experiment_id="storefront_refresh",
        group_id="treatment",
        prefix="st",
        units=units,
        start=STOREFRONT_WINDOW[0],
        convert_rate=0.4,
    )
    # Each metric window closes 14 days after the last exposure; every fact needs a later event.
    rows.extend(
        {
            "event_at": dt.datetime(2025, 5, 1, 12, 0),
            "user_id": "keepalive",
            "event": event,
            "experiment_id": None,
            "group_id": None,
            "revenue": 1.0 if event == "purchase" else None,
            "duration_s": 1.0 if event == "session_end" else None,
            "country_code": "US",
        }
        for event in ("page_view", "purchase", "session_end")
    )
    return rows


@contextmanager
def _workspace(
    tmp_path,
    *,
    units: int = 30,
    breakout_sources: tuple[str | None, ...] = ("event_log",),
    metrics: str = _METRICS,
):
    import ibis
    import pyarrow as pa

    from increment import Analysis

    rows = _event_rows(units)
    schema = pa.schema(
        [
            ("event_at", pa.timestamp("us")),
            ("user_id", pa.string()),
            ("event", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", pa.string()),
            ("revenue", pa.float64()),
            ("duration_s", pa.float64()),
            ("country_code", pa.string()),
        ]
    )
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE SCHEMA analytics")
    con.create_table("event_log", pa.Table.from_pylist(rows, schema=schema), database="analytics")
    profiles = "profiles" in breakout_sources
    if profiles:
        countries = {row["user_id"]: row["country_code"] for row in rows}
        con.create_table(
            "profiles",
            pa.Table.from_pylist(
                [
                    {
                        "user_id": user,
                        "updated_at": dt.datetime(2025, 1, 1),
                        "country_code": country,
                    }
                    for user, country in countries.items()
                ]
            ),
            database="analytics",
        )
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    for name, body in (
        ("fact_sources.yaml", _PROFILE_SOURCES if profiles else _EVENT_LOG_SOURCE),
        ("exposures.yaml", _EXPOSURES),
        ("metrics.yaml", metrics),
        ("experiments.yaml", _experiments(breakout_sources=breakout_sources)),
    ):
        (definitions / name).write_text(body)
    analysis = Analysis.from_definitions("storefront_refresh", str(definitions), con)
    try:
        yield con, analysis
    finally:
        with contextlib.suppress(Exception):
            analysis.close()
        with contextlib.suppress(Exception):
            con.disconnect()


def _double_the_treatment_arm(con) -> None:
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "SELECT event_at, user_id || '_late', event, experiment_id, group_id, "
        "revenue, duration_s, country_code FROM analytics.event_log "
        "WHERE experiment_id = 'storefront_refresh' AND group_id = 'treatment'"
    )


_NAMES = ("checkout_conversion", "revenue_per_user", "ever_purchased_again", "session_seconds")
_SCOPES = (None, ("event_log", "country"))
# (view, completed windows only), including every state the dashboard can ask for.
_STATES = (
    ("cumulative_lift", False),
    ("cumulative_lift", True),
    ("cumulative_values", False),
    ("cumulative_values", True),
    ("daily_values", False),
    ("segments", False),
)


_IDENTITY = (
    "ds",
    "metric",
    "group_id",
    "method",
    "method_role",
    "estimand",
    "dimension",
    "dimension_value",
    "source",
)


def _stable(value: Any) -> Any:
    """Warehouse reductions are unordered and floats vary slightly between identical reads."""
    if isinstance(value, float):
        return 0.0 if abs(value) < 1e-12 else float(f"{value:.9g}")
    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in value.items()
            if key not in {"source_snapshot_id", "family_id"}
        }
    if isinstance(value, list | tuple):
        return [_stable(item) for item in value]
    return value


def _fingerprint(rows, *, semantic: bool = False) -> list[str]:
    dumped = [
        _stable(row.model_dump(mode="json")) if semantic else row.model_dump(mode="json")
        for row in rows
    ]
    dumped.sort(key=lambda row: [json.dumps([row.get(name) for name in _IDENTITY], default=str)])
    return [json.dumps(row, sort_keys=True, default=str) for row in dumped]


def _snapshot_ids(rows) -> set[tuple[tuple[Any, ...], str | None, str | None]]:
    identities = set()
    for row in rows:
        dumped = row.model_dump(mode="json")
        identities.add(
            (
                tuple(dumped.get(name) for name in _IDENTITY),
                dumped.get("source_snapshot_id"),
                dumped.get("family_id"),
            )
        )
    return identities


def _live(analysis, *, view, metric, complete, scope):
    """The same call the snapshot is expected to have made, against the live warehouse now."""
    names = list(_NAMES) if metric is None else [metric]
    dimension = None if scope is None else scope[1]
    if view == "cumulative_lift":
        rows = analysis.run_asof_lift(
            metrics=names, completed_windows_only=complete, dimension=dimension
        )
    elif view == "cumulative_values":
        rows = analysis.run_asof(
            metrics=names, completed_windows_only=complete, dimension=dimension
        )
    elif view == "daily_values":
        rows = analysis.run_daily(metrics=names, dimension=dimension)
    else:
        rows = [
            row
            for row in analysis.run_breakout(metrics=names)
            if row.dimension == "country" and row.method_role == "decision"
        ]
    if scope is not None:
        rows = [row for row in rows if row.source == scope[0]]
    return _fingerprint(rows, semantic=True)


def _outcome(call) -> tuple[str, ...] | list[str]:
    try:
        return call()
    except CodedError as error:
        return ("refused", type(error).__name__, error.code)


def _loaded(analysis, snapshot, *, view, metric, complete, scope, semantic=False):
    def load():
        return _fingerprint(
            load_explore(
                analysis,
                snapshot=snapshot,
                metric=metric,
                view=view,
                completed_windows_only=complete,
                breakout=scope,
            ),
            semantic=semantic,
        )

    return _outcome(load)


def _requests():
    for view, complete in _STATES:
        for metric in (None, *_NAMES):
            for scope in _SCOPES:
                if view == "segments" and scope is None:
                    continue  # a segment view is only defined for a declared choice
                yield {"view": view, "metric": metric, "complete": complete, "scope": scope}


@pytest.mark.slow
def test_every_explore_state_is_the_pinned_read_even_after_the_warehouse_changes(tmp_path):
    with _workspace(tmp_path) as (con, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        requests = list(_requests())
        expected = {
            index: _outcome(lambda request=request: _live(analysis, **request))
            for index, request in enumerate(requests)
        }
        before = {
            index: _loaded(analysis, snapshot, **request) for index, request in enumerate(requests)
        }
        before_semantic = {
            index: _loaded(analysis, snapshot, **request, semantic=True)
            for index, request in enumerate(requests)
        }
        assert before_semantic == expected
        # The matrix is not vacuous: some states carry rows and some are original refusals.
        assert any(isinstance(value, list) and value for value in expected.values())
        assert any(isinstance(value, tuple) for value in expected.values())

        _double_the_treatment_arm(con)

        live_after = {
            index: _outcome(lambda request=request: _live(analysis, **request))
            for index, request in enumerate(requests)
        }
        assert live_after != expected, "the warehouse change must be visible to a live read"
        after = {
            index: _loaded(analysis, snapshot, **request) for index, request in enumerate(requests)
        }
        assert after == before


def _payload_text(analysis, snapshot) -> str:
    """The whole rendered payload, with the table renderer's random element ids numbered."""
    text = json.dumps(build_payload(analysis, snapshot=snapshot), sort_keys=True)
    ids = dict.fromkeys(re.findall(r'id=\\"([a-z]{10})\\"', text))
    for number, element_id in enumerate(ids):
        text = text.replace(element_id, f"element{number}")
    return text


@pytest.mark.slow
def test_dashboard_payload_ignores_changes_after_preparation_until_reprepared(tmp_path):
    with _workspace(tmp_path, units=20) as (con, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        before = _payload_text(analysis, snapshot)

        _double_the_treatment_arm(con)

        assert _payload_text(analysis, snapshot) == before
        fresh = prepare_dashboard(analysis, config=CONFIG)
        assert fresh.readout_rows != snapshot.readout_rows
        assert _payload_text(analysis, fresh) != before

        def treatment_count(captured):
            if captured.allocation is not None:
                return captured.allocation.observed["treatment"]
            counts = {
                row.assigned_units
                for row in captured.group_data
                if row.group_id == "treatment" and row.analysis_population == "assigned"
            }
            assert counts and None not in counts
            assert len(counts) == 1
            count = next(iter(counts))
            assert isinstance(count, int)
            return count

        assert treatment_count(fresh) == 2 * treatment_count(snapshot)


def test_explore_reads_nothing_from_the_warehouse_after_preparation(tmp_path):
    with _workspace(tmp_path) as (con, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        expected = {
            index: _loaded(analysis, snapshot, **request)
            for index, request in enumerate(_requests())
        }
        con.disconnect()
        for index, request in enumerate(_requests()):
            assert _loaded(analysis, snapshot, **request) == expected[index]


@pytest.mark.slow
@pytest.mark.parametrize("scope", [None, ("profiles", "country")])
def test_breakout_property_in_its_own_source_is_part_of_the_pin(tmp_path, scope):
    with _workspace(tmp_path, breakout_sources=("profiles",)) as (con, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        request = {
            "view": "cumulative_values",
            "metric": "checkout_conversion",
            "complete": False,
            "scope": scope,
        }
        before = _loaded(analysis, snapshot, **request)
        assert isinstance(before, list) and before
        if scope is not None:
            segments = {json.loads(row)["dimension_value"] for row in before}
            assert segments == {"US", "CA"}

        con.raw_sql("UPDATE analytics.profiles SET country_code = 'XX'")

        after = _loaded(analysis, snapshot, **request)
        assert after == before
        fresh = prepare_dashboard(analysis, config=CONFIG)
        if scope is not None:
            moved = load_explore(
                analysis,
                snapshot=fresh,
                metric="checkout_conversion",
                view="cumulative_values",
                breakout=scope,
            )
            assert {row.dimension_value for row in moved} == {"XX"}


_REFUSED_SOURCE = "test.profiles_unavailable"


def _refuse_the_profiles_breakout(monkeypatch) -> None:
    """A source that refuses every read of the `profiles` breakout, and only that breakout."""
    from increment.errors import CapabilityError
    from increment.query import native_source

    def refuse() -> None:
        raise CapabilityError("profiles is unavailable", code=_REFUSED_SOURCE, context={})

    day_moments = native_source._DaySource.breakout_moments
    scoped_sources = native_source.DefinitionsMomentSource.breakout_sources

    def breakout_moments(self, metric, breakout, **kwargs):
        if breakout.source == "profiles":
            refuse()
        return day_moments(self, metric, breakout, **kwargs)

    def breakout_sources(self, breakouts, **kwargs):
        if any(breakout.source == "profiles" for breakout in breakouts):
            refuse()
        return scoped_sources(self, breakouts, **kwargs)

    monkeypatch.setattr(native_source._DaySource, "breakout_moments", breakout_moments)
    monkeypatch.setattr(native_source.DefinitionsMomentSource, "breakout_sources", breakout_sources)


@pytest.mark.slow
def test_a_refused_breakout_never_hides_another_source_of_the_same_property(tmp_path, monkeypatch):
    event_log, profiles = ("event_log", "country"), ("profiles", "country")
    requests = [
        {"view": view, "metric": metric, "complete": complete}
        for view, complete in _STATES
        for metric in (None, "checkout_conversion")
    ]
    sources = ("event_log", "profiles")
    with _workspace(tmp_path, units=20, breakout_sources=sources) as (_, analysis):
        healthy = prepare_dashboard(analysis, config=CONFIG)
        baseline = [
            _loaded(analysis, healthy, **request, scope=event_log, semantic=True)
            for request in requests
        ]
        answered = [
            isinstance(_loaded(analysis, healthy, **request, scope=profiles, semantic=True), list)
            for request in requests
        ]
        assert any(answered)
        assert any(isinstance(value, list) and value for value in baseline)
        # One declared breakout per choice reads exactly what the whole dimension shows for it.
        for request, value in zip(requests, baseline, strict=True):
            assert value == _outcome(lambda r=request: _live(analysis, **r, scope=event_log))

        _refuse_the_profiles_breakout(monkeypatch)
        degraded = prepare_dashboard(analysis, config=CONFIG)
        for request, value, was_answered in zip(requests, baseline, answered, strict=True):
            assert _loaded(analysis, degraded, **request, scope=event_log, semantic=True) == value
            refused = _loaded(analysis, degraded, **request, scope=profiles, semantic=True)
            assert isinstance(refused, tuple)
            if was_answered:
                assert refused == ("refused", "CapabilityError", _REFUSED_SOURCE)


def test_an_omitted_breakout_source_shows_the_source_it_resolves_to(tmp_path):
    with _workspace(tmp_path, breakout_sources=(None,)) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        for view in ("segments", "cumulative_lift"):
            request = {"view": view, "metric": "checkout_conversion", "complete": False}
            shown = _loaded(analysis, snapshot, **request, scope=(None, "country"), semantic=True)
            assert isinstance(shown, list) and shown
            assert {json.loads(row)["source"] for row in shown} == {"event_log"}
            assert shown == _live(analysis, **request, scope=("event_log", "country"))


def test_each_state_keeps_its_own_original_refusal(tmp_path):
    retention = "ever_purchased_again"
    with _workspace(tmp_path) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        for view, complete in (("daily_values", False), ("cumulative_values", True)):
            with pytest.raises(CodedError) as live:
                if view == "daily_values":
                    analysis.run_daily(metrics=[retention])
                else:
                    analysis.run_asof(metrics=[retention], completed_windows_only=True)
            with pytest.raises(CodedError) as first:
                load_explore(
                    analysis,
                    snapshot=snapshot,
                    metric=retention,
                    view=view,
                    completed_windows_only=complete,
                )
            with pytest.raises(CodedError) as second:
                load_explore(
                    analysis,
                    snapshot=snapshot,
                    metric=retention,
                    view=view,
                    completed_windows_only=complete,
                )
            assert type(first.value) is type(live.value)
            assert first.value.code == live.value.code
            assert dict(first.value.context) == dict(live.value.context)
            assert str(first.value) == str(live.value)
            assert first.value is not second.value

            # One metric's refusal never hides another's series.
            other = load_explore(
                analysis,
                snapshot=snapshot,
                metric="checkout_conversion",
                view=view,
                completed_windows_only=complete,
            )
            assert other
            assert {row.metric for row in other} == {"checkout_conversion"}
            # The whole-family call is itself refused with the same code.
            with pytest.raises(CodedError) as batch:
                load_explore(
                    analysis,
                    snapshot=snapshot,
                    metric=None,
                    view=view,
                    completed_windows_only=complete,
                )
            assert batch.value.code == live.value.code


def test_each_metric_series_is_independent_of_the_metric_selection(tmp_path):
    with _workspace(tmp_path, breakout_sources=()) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        first_run = analysis.run(metrics=["session_seconds"])
        second_run = analysis.run(metrics=["session_seconds"])
        assert _snapshot_ids(first_run) == _snapshot_ids(second_run)
        everyone = load_explore(analysis, snapshot=snapshot, metric=None, view="cumulative_lift")
        assert isinstance(everyone, DailyLiftEstimates)
        one = load_explore(
            analysis, snapshot=snapshot, metric="session_seconds", view="cumulative_lift"
        )
        expected = analysis.run_asof_lift(metrics=["session_seconds"])
        assert _fingerprint(one, semantic=True) == _fingerprint(expected, semantic=True)
        assert _snapshot_ids(one) == _snapshot_ids(expected)
        repeat = analysis.run_asof_lift(metrics=["session_seconds"])
        assert _snapshot_ids(expected) == _snapshot_ids(repeat)
        full = analysis.run_asof_lift(metrics=list(_NAMES))
        assert _fingerprint(everyone, semantic=True) == _fingerprint(full, semantic=True)
        assert _snapshot_ids(everyone) == _snapshot_ids(full)
        values = load_explore(
            analysis, snapshot=snapshot, metric="session_seconds", view="daily_values"
        )
        assert isinstance(values, DailyMetricValues)
        values_expected = analysis.run_daily(metrics=["session_seconds"])
        assert _fingerprint(values, semantic=True) == _fingerprint(values_expected, semantic=True)
        assert _snapshot_ids(values) == _snapshot_ids(values_expected)
        assert (
            load_explore(analysis, snapshot=snapshot, metric="checkout_conversion", view="segments")
            == BreakoutEstimates()
        )


def test_loaded_collections_are_independent_copies_of_the_capture(tmp_path):
    with _workspace(tmp_path) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=CONFIG)
        first = load_explore(
            analysis, snapshot=snapshot, metric="checkout_conversion", view="cumulative_values"
        )
        kept = _fingerprint(first)
        kept_ids = _snapshot_ids(first)
        assert kept
        assert isinstance(first, DailyMetricValues)
        with pytest.raises(CodedError) as caught:
            first.clear()
        assert caught.value.code == "readout.collection.mutation_unsupported"
        again = load_explore(
            analysis, snapshot=snapshot, metric="checkout_conversion", view="cumulative_values"
        )
        assert again is not first
        assert _fingerprint(first) == kept
        assert _fingerprint(again) == kept
        assert _snapshot_ids(first) == kept_ids
        assert _snapshot_ids(again) == kept_ids


@pytest.mark.slow
def test_registered_sequential_explore_is_captured_at_its_checkpoint():
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(sequential=True) as (connection, analysis):
        analysis.capture_sequential(finalized=True, as_of=dt.date(2025, 1, 16))
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        snapshot = prepare_dashboard(analysis, config=config)
        live_lift = _fingerprint(analysis.run_asof_lift(metrics=["conversion"]), semantic=True)
        live_values = _fingerprint(analysis.run_asof(metrics=["mean"]), semantic=True)
        assert live_lift and live_values
        connection.raw_sql("UPDATE events SET value = value * 10 WHERE event = 'purchase'")
        assert _fingerprint(analysis.run_asof(metrics=["mean"]), semantic=True) != live_values
        lift = load_explore(
            analysis, snapshot=snapshot, metric="conversion", view="cumulative_lift"
        )
        assert _fingerprint(lift, semantic=True) == live_lift
        values = load_explore(analysis, snapshot=snapshot, metric="mean", view="cumulative_values")
        assert _fingerprint(values, semantic=True) == live_values
