"""Explore data is captured with the headline, not read from the warehouse afterwards.

``prepare_dashboard`` pins the warehouse once. Every Explore choice the dashboard offers must
come from that same pinned read, so a later warehouse change cannot make the headline and the
trajectories describe different data, and a re-preparation must deliberately see the new data.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime as dt
import io
import json
import random
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment import Scope
from increment.breakout.estimates import BreakoutEstimates, DailyLiftEstimates, DailyMetricValues
from increment.dashboard import (
    DashboardConfig,
    DashboardSnapshot,
    ExploreView,
    load_explore,
    prepare_dashboard,
    readout_csv,
    render_header,
    render_health,
    render_metric_details,
    render_results,
)
from increment.dashboard._app import build_payload
from increment.dashboard._data import enriched_rows
from increment.dashboard._html import overview_notes, overview_rows
from increment.errors import CapabilityError, CodedError, InvalidRequestError
from increment.evidence import BreakoutIdentity
from increment.workflows import capture_sequential
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import view_readout
from tests.dashboard_snapshots import ADDED, COUNTRY

# Small deterministic fixtures legitimately trip estimator reliability warnings; the subject
# here is which data is read, not estimator reliability.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning"),
]

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

    from increment import from_definitions

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
    experiment, resolver = from_definitions(
        str(definitions), con, experiment_name="storefront_refresh"
    )
    try:
        yield con, experiment, resolver
    finally:
        with contextlib.suppress(Exception):
            cast(Any, resolver).close()
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


def _live(experiment, resolver, *, view, metric, complete, scope):
    """The same view the snapshot is expected to have read, against the live warehouse now."""
    names = list(_NAMES) if metric is None else [metric]
    segment = None if scope is None else BreakoutIdentity(scope[1], None)
    if view == "cumulative_lift":
        live = Scope(view="asof", grain="asof", completion_policy=complete, segment=segment)
        rows = view_readout(experiment, resolver, live, metrics=names)
    elif view == "cumulative_values":
        live = Scope(
            view="asof", grain="asof", values=True, completion_policy=complete, segment=segment
        )
        rows = view_readout(experiment, resolver, live, metrics=names)
    elif view == "daily_values":
        live = Scope(view="daily", grain="daily", values=True, segment=segment)
        rows = view_readout(experiment, resolver, live, metrics=names)
    else:
        rows = [
            row
            for row in view_readout(
                experiment, resolver, Scope(view="breakout", grain="total"), metrics=names
            )
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


def _loaded(experiment, snapshot, *, view, metric, complete, scope, semantic=False):
    def load():
        return _fingerprint(
            load_explore(
                experiment,
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
    with _workspace(tmp_path) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        requests = list(_requests())
        expected = {
            index: _outcome(lambda request=request: _live(experiment, resolver, **request))
            for index, request in enumerate(requests)
        }
        before = {
            index: _loaded(experiment, snapshot, **request)
            for index, request in enumerate(requests)
        }
        before_semantic = {
            index: _loaded(experiment, snapshot, **request, semantic=True)
            for index, request in enumerate(requests)
        }
        assert before_semantic == expected
        # The matrix is not vacuous: some states carry rows and some are original refusals.
        assert any(isinstance(value, list) and value for value in expected.values())
        assert any(isinstance(value, tuple) for value in expected.values())

        _double_the_treatment_arm(con)

        live_after = {
            index: _outcome(lambda request=request: _live(experiment, resolver, **request))
            for index, request in enumerate(requests)
        }
        assert live_after != expected, "the warehouse change must be visible to a live read"
        after = {
            index: _loaded(experiment, snapshot, **request)
            for index, request in enumerate(requests)
        }
        assert after == before


def _payload_text(experiment, snapshot) -> str:
    """The whole rendered payload, with the table renderer's random element ids numbered."""
    text = json.dumps(build_payload(experiment, snapshot=snapshot), sort_keys=True)
    ids = dict.fromkeys(re.findall(r'id=\\"([a-z]{10})\\"', text))
    for number, element_id in enumerate(ids):
        text = text.replace(element_id, f"element{number}")
    return text


@pytest.mark.slow
def test_dashboard_payload_ignores_changes_after_preparation_until_reprepared(tmp_path):
    with _workspace(tmp_path, units=20) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        before = _payload_text(experiment, snapshot)

        _double_the_treatment_arm(con)

        assert _payload_text(experiment, snapshot) == before
        fresh = prepare_dashboard(experiment, resolver, config=CONFIG)
        assert fresh.readout_rows != snapshot.readout_rows
        assert _payload_text(experiment, fresh) != before

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


@pytest.mark.slow
def test_explore_reads_nothing_from_the_warehouse_after_preparation(tmp_path):
    with _workspace(tmp_path) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        expected = {
            index: _loaded(experiment, snapshot, **request)
            for index, request in enumerate(_requests())
        }
        con.disconnect()
        for index, request in enumerate(_requests()):
            assert _loaded(experiment, snapshot, **request) == expected[index]


@pytest.mark.slow
@pytest.mark.parametrize("scope", [None, ("profiles", "country")])
def test_breakout_property_in_its_own_source_is_part_of_the_pin(tmp_path, scope):
    with _workspace(tmp_path, breakout_sources=("profiles",)) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        request = {
            "view": "cumulative_values",
            "metric": "checkout_conversion",
            "complete": False,
            "scope": scope,
        }
        before = _loaded(experiment, snapshot, **request)
        assert isinstance(before, list) and before
        if scope is not None:
            segments = {json.loads(row)["dimension_value"] for row in before}
            assert segments == {"US", "CA"}

        con.raw_sql("UPDATE analytics.profiles SET country_code = 'XX'")

        after = _loaded(experiment, snapshot, **request)
        assert after == before
        fresh = prepare_dashboard(experiment, resolver, config=CONFIG)
        if scope is not None:
            moved = load_explore(
                experiment,
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
    with _workspace(tmp_path, units=20, breakout_sources=sources) as (_, experiment, resolver):
        healthy = prepare_dashboard(experiment, resolver, config=CONFIG)
        baseline = [
            _loaded(experiment, healthy, **request, scope=event_log, semantic=True)
            for request in requests
        ]
        answered = [
            isinstance(_loaded(experiment, healthy, **request, scope=profiles, semantic=True), list)
            for request in requests
        ]
        assert any(answered)
        assert any(isinstance(value, list) and value for value in baseline)
        # One declared breakout per choice reads exactly what the whole dimension shows for it.
        for request, value in zip(requests, baseline, strict=True):
            assert value == _outcome(
                lambda r=request: _live(experiment, resolver, **r, scope=event_log)
            )

        _refuse_the_profiles_breakout(monkeypatch)
        degraded = prepare_dashboard(experiment, resolver, config=CONFIG)
        for request, value, was_answered in zip(requests, baseline, answered, strict=True):
            assert _loaded(experiment, degraded, **request, scope=event_log, semantic=True) == value
            refused = _loaded(experiment, degraded, **request, scope=profiles, semantic=True)
            assert isinstance(refused, tuple)
            if was_answered:
                assert refused == ("refused", "CapabilityError", _REFUSED_SOURCE)


@pytest.mark.slow
def test_an_omitted_breakout_source_shows_the_source_it_resolves_to(tmp_path):
    with _workspace(tmp_path, breakout_sources=(None,)) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        for view in ("segments", "cumulative_lift"):
            request = {"view": view, "metric": "checkout_conversion", "complete": False}
            shown = _loaded(experiment, snapshot, **request, scope=(None, "country"), semantic=True)
            assert isinstance(shown, list) and shown
            assert {json.loads(row)["source"] for row in shown} == {"event_log"}
            assert shown == _live(experiment, resolver, **request, scope=("event_log", "country"))


@pytest.mark.slow
def test_each_state_keeps_its_own_original_refusal(tmp_path):
    retention = "ever_purchased_again"
    with _workspace(tmp_path) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        for view, complete in (("daily_values", False), ("cumulative_values", True)):
            with pytest.raises(CodedError) as live:
                if view == "daily_values":
                    view_readout(
                        experiment,
                        resolver,
                        Scope(view="daily", grain="daily", values=True),
                        metrics=[retention],
                    )
                else:
                    view_readout(
                        experiment,
                        resolver,
                        Scope(view="asof", grain="asof", values=True, completion_policy=True),
                        metrics=[retention],
                    )
            with pytest.raises(CodedError) as first:
                load_explore(
                    experiment,
                    snapshot=snapshot,
                    metric=retention,
                    view=view,
                    completed_windows_only=complete,
                )
            with pytest.raises(CodedError) as second:
                load_explore(
                    experiment,
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
                experiment,
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
                    experiment,
                    snapshot=snapshot,
                    metric=None,
                    view=view,
                    completed_windows_only=complete,
                )
            assert batch.value.code == live.value.code


@pytest.mark.slow
def test_each_metric_series_is_independent_of_the_metric_selection(tmp_path):
    with _workspace(tmp_path, breakout_sources=()) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        first_run = view_readout(experiment, resolver, metrics=["session_seconds"])
        second_run = view_readout(experiment, resolver, metrics=["session_seconds"])
        assert _snapshot_ids(first_run) == _snapshot_ids(second_run)
        everyone = load_explore(experiment, snapshot=snapshot, metric=None, view="cumulative_lift")
        assert isinstance(everyone, DailyLiftEstimates)
        one = load_explore(
            experiment, snapshot=snapshot, metric="session_seconds", view="cumulative_lift"
        )
        asof_lift = Scope(view="asof", grain="asof")
        expected = view_readout(experiment, resolver, asof_lift, metrics=["session_seconds"])
        assert _fingerprint(one, semantic=True) == _fingerprint(expected, semantic=True)
        assert _snapshot_ids(one) == _snapshot_ids(expected)
        repeat = view_readout(experiment, resolver, asof_lift, metrics=["session_seconds"])
        assert _snapshot_ids(expected) == _snapshot_ids(repeat)
        full = view_readout(experiment, resolver, asof_lift, metrics=list(_NAMES))
        assert _fingerprint(everyone, semantic=True) == _fingerprint(full, semantic=True)
        assert _snapshot_ids(everyone) == _snapshot_ids(full)
        values = load_explore(
            experiment, snapshot=snapshot, metric="session_seconds", view="daily_values"
        )
        assert isinstance(values, DailyMetricValues)
        values_expected = view_readout(
            experiment,
            resolver,
            Scope(view="daily", grain="daily", values=True),
            metrics=["session_seconds"],
        )
        assert _fingerprint(values, semantic=True) == _fingerprint(values_expected, semantic=True)
        assert _snapshot_ids(values) == _snapshot_ids(values_expected)
        assert (
            load_explore(
                experiment, snapshot=snapshot, metric="checkout_conversion", view="segments"
            )
            == BreakoutEstimates()
        )


@pytest.mark.slow
def test_loaded_collections_are_independent_copies_of_the_capture(tmp_path):
    with _workspace(tmp_path) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        first = load_explore(
            experiment, snapshot=snapshot, metric="checkout_conversion", view="cumulative_values"
        )
        kept = _fingerprint(first)
        kept_ids = _snapshot_ids(first)
        assert kept
        assert isinstance(first, DailyMetricValues)
        with pytest.raises(CodedError) as caught:
            first.clear()
        assert caught.value.code == "readout.collection.mutation_unsupported"
        again = load_explore(
            experiment, snapshot=snapshot, metric="checkout_conversion", view="cumulative_values"
        )
        assert again is not first
        assert _fingerprint(first) == kept
        assert _fingerprint(again) == kept
        assert _snapshot_ids(first) == kept_ids
        assert _snapshot_ids(again) == kept_ids


@pytest.mark.slow
def test_registered_sequential_explore_is_captured_at_its_checkpoint():
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(sequential=True) as (connection, experiment, resolver):
        capture_sequential(experiment, resolver, finalized=True, as_of=dt.date(2025, 1, 16))
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        snapshot = prepare_dashboard(experiment, resolver, config=config)
        asof_lift = Scope(view="asof", grain="asof")
        asof_values = Scope(view="asof", grain="asof", values=True)
        live_lift = _fingerprint(
            view_readout(experiment, resolver, asof_lift, metrics=["conversion"]), semantic=True
        )
        live_values = _fingerprint(
            view_readout(experiment, resolver, asof_values, metrics=["mean"]), semantic=True
        )
        assert live_lift and live_values
        connection.raw_sql("UPDATE events SET value = value * 10 WHERE event = 'purchase'")
        assert (
            _fingerprint(
                view_readout(experiment, resolver, asof_values, metrics=["mean"]), semantic=True
            )
            != live_values
        )
        lift = load_explore(
            experiment, snapshot=snapshot, metric="conversion", view="cumulative_lift"
        )
        assert _fingerprint(lift, semantic=True) == live_lift
        values = load_explore(
            experiment, snapshot=snapshot, metric="mean", view="cumulative_values"
        )
        assert _fingerprint(values, semantic=True) == live_values


STOREFRONT_WINDOW = (dt.date(2025, 1, 15), dt.date(2025, 2, 14))


PRICING_WINDOW = (dt.date(2025, 3, 1), dt.date(2025, 3, 31))


FACT_SOURCES = """
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
        description: A page view
      - name: purchase
        column: revenue
        description: A purchase with revenue
      - name: session_end
        column: duration_s
        description: Session ended with a duration
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
        description: Country, fixed per unit in this fixture
"""


EXPOSURES = """
exposures:
  - name: first_page_view
    fact: page_view
    description: First page view inside the analysis window
"""


METRICS = """
metrics:
  - type: conversion
    name: checkout_conversion
    description: Did the user check out within 14 days of exposure?
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14

  - type: mean
    name: revenue_per_user
    description: Purchase revenue per enrolled user
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14

  - type: mean
    name: session_seconds
    description: Seconds spent in session per enrolled user
    entity: user_id
    preferred_direction: decrease
    fact: session_end
    aggregation: sum
    window_days: 14

  - type: conversion
    name: purchase_rate
    description: Did the user purchase within 14 days of exposure?
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14
"""


EXPERIMENTS = """
experiments:
  - name: storefront_refresh
    description: A refreshed storefront against the current storefront
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-14
    n_pre_periods: 0
    control_group: control
    allocation_scheme: independent
    plan:
      view_multiplicity:
        correction: bonferroni
      primary: checkout_conversion
      secondaries:
        - revenue_per_user
      guardrails:
        - session_seconds
    breakouts:
      - property: country
        source: event_log

  - name: pricing_copy
    description: Reworded pricing copy against the current copy
    exposure: first_page_view
    unit: user_id
    start: 2025-03-01
    end: 2025-03-31
    n_pre_periods: 0
    control_group: baseline
    allocation_scheme: independent
    plan:
      primary: purchase_rate

  - name: dual_primary
    description: Two declared primary metrics, which the dashboard refuses
    exposure: first_page_view
    unit: user_id
    start: 2025-03-01
    end: 2025-03-31
    n_pre_periods: 0
    control_group: control
    plan:
      primary:
        - checkout_conversion
        - revenue_per_user
"""


def _arm_rows(
    rng: random.Random,
    *,
    experiment_id: str,
    group_id: str,
    prefix: str,
    units: int,
    start: dt.date,
    convert_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(units):
        user = f"{prefix}{index:05d}"
        exposure_at = dt.datetime.combine(start, dt.time(9, 0)) + dt.timedelta(
            days=index % 10, minutes=index % 37
        )
        country = "US" if index % 3 else "CA"
        rows.append(
            {
                "event_at": exposure_at,
                "user_id": user,
                "event": "page_view",
                "experiment_id": experiment_id,
                "group_id": group_id,
                "revenue": None,
                "duration_s": None,
                "country_code": country,
            }
        )
        rows.append(
            {
                "event_at": exposure_at + dt.timedelta(hours=1),
                "user_id": user,
                "event": "session_end",
                "experiment_id": experiment_id,
                "group_id": group_id,
                "revenue": None,
                "duration_s": 40.0 + rng.random() * 20.0,
                "country_code": country,
            }
        )
        if rng.random() < convert_rate:
            rows.append(
                {
                    "event_at": exposure_at + dt.timedelta(days=1),
                    "user_id": user,
                    "event": "purchase",
                    "experiment_id": experiment_id,
                    "group_id": group_id,
                    "revenue": 20.0 + rng.random() * 30.0,
                    "duration_s": None,
                    "country_code": country,
                }
            )
    return rows


def _live_snapshot(connection, definitions, experiment_name: str, config: DashboardConfig):
    experiment, resolver = _bind(connection, definitions, experiment_name)
    with resolver:
        return prepare_dashboard(experiment, resolver, config=config)


@pytest.fixture(scope="session")
def dashboard_definitions(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("dashboard_definitions")
    for name, body in (
        ("fact_sources.yaml", FACT_SOURCES),
        ("exposures.yaml", EXPOSURES),
        ("metrics.yaml", METRICS),
        ("experiments.yaml", EXPERIMENTS),
    ):
        (path / name).write_text(body)
    return str(path)


@backend_fixture(scope="session")
def dashboard_con():
    import ibis
    import pyarrow as pa

    rng = random.Random(11)
    rows = (
        _arm_rows(
            rng,
            experiment_id="storefront_refresh",
            group_id="control",
            prefix="sc",
            units=600,
            start=STOREFRONT_WINDOW[0],
            convert_rate=0.20,
        )
        + _arm_rows(
            rng,
            experiment_id="storefront_refresh",
            group_id="treatment",
            prefix="st",
            units=600,
            start=STOREFRONT_WINDOW[0],
            convert_rate=0.25,
        )
        + _arm_rows(
            rng,
            experiment_id="pricing_copy",
            group_id="baseline",
            prefix="pb",
            units=700,
            start=PRICING_WINDOW[0],
            convert_rate=0.15,
        )
        + _arm_rows(
            rng,
            experiment_id="pricing_copy",
            group_id="candidate",
            prefix="pc",
            units=300,
            start=PRICING_WINDOW[0],
            convert_rate=0.19,
        )
    )
    # Each metric window closes 14 days after the last exposure, and the
    # observable bound is per fact: every fact needs an event past that date
    # or censoring drops every enrolled unit.
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
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    con.create_table("event_log", pa.Table.from_pylist(rows, schema=schema), database="analytics")
    return con


def _bind(con, definitions: str | Path, experiment: str):
    from increment import from_definitions

    return from_definitions(definitions, con, experiment_name=experiment)


def _experiment(con, definitions: str | Path, experiment: str):
    """The declaration alone; snapshot consumers never read through a resolver."""
    declared, resolver = _bind(con, definitions, experiment)
    resolver.close()
    return declared


@backend_fixture(scope="session")
def live_storefront(dashboard_con, dashboard_definitions) -> DashboardSnapshot:
    return _live_snapshot(
        dashboard_con,
        dashboard_definitions,
        "storefront_refresh",
        DashboardConfig(
            expected_allocation={"control": 0.5, "treatment": 0.5}, source_label="Unit fixture"
        ),
    )


@backend_fixture(scope="session")
def live_pricing(dashboard_con, dashboard_definitions) -> DashboardSnapshot:
    return _live_snapshot(
        dashboard_con,
        dashboard_definitions,
        "pricing_copy",
        DashboardConfig(
            expected_allocation={"baseline": 0.7, "candidate": 0.3},
            title="Pricing copy rewrite",
            provenance={"Fixture": "in-memory"},
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_preparation_preserves_registered_sequential_decisions(law, tmp_path):
    from increment import run
    from increment.estimation.results import LiftEstimate
    from increment.workflows import capture_sequential
    from tests.test_sequential_public_sources import _native_pair

    connection, _, experiment, resolver = _native_pair(tmp_path, law)
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=dt.date(2025, 1, 16))
        expected = run(experiment, resolver).readout()[0]
        assert isinstance(expected, LiftEstimate)
        snapshot = prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        )
        observed = snapshot.estimates[0]
        assert observed.stat_sig() is True
        assert observed.require_sequential_result() == expected.require_sequential_result()
    finally:
        resolver.close()
        connection.disconnect()


@pytest.mark.slow
@pytest.mark.integration
def test_preparation_reads_every_section_from_one_pin_and_leaves_the_caller_live(
    dashboard_con, dashboard_definitions, live_pricing, monkeypatch
):
    import contextlib

    import ibis

    from tests.analysis_factory import bound_source

    con = ibis.duckdb.connect()
    try:
        con.raw_sql("CREATE SCHEMA analytics")
        con.create_table(
            "event_log",
            dashboard_con.table("event_log", database="analytics").to_pyarrow(),
            database="analytics",
        )
        experiment, resolver = _bind(con, dashboard_definitions, "pricing_copy")
        source = bound_source(resolver)
        pin = source.readout_snapshot

        @contextlib.contextmanager
        def double_the_candidates_once_pinned(**options):
            with pin(**options) as pinned:
                con.raw_sql(
                    "INSERT INTO analytics.event_log "
                    "SELECT event_at, user_id || '_new', event, experiment_id, group_id, "
                    "revenue, duration_s, country_code FROM analytics.event_log "
                    "WHERE experiment_id = 'pricing_copy' AND group_id = 'candidate'"
                )
                yield pinned

        with monkeypatch.context() as patch:
            patch.setattr(source, "readout_snapshot", double_the_candidates_once_pinned)
            snapshot = prepare_dashboard(experiment, resolver, config=live_pricing.config)

        pinned_counts = {"baseline": 700, "candidate": 300}
        assert snapshot.allocation is not None
        assert snapshot.allocation.observed == pinned_counts
        final_counts = {row["group_id"]: row["n_cumulative"] for row in snapshot.allocation_history}
        assert final_counts == pinned_counts
        assert snapshot.health is not None
        (allocation_check,) = (
            check for check in snapshot.health.checks if check.check == "assigned_allocation"
        )
        assert allocation_check.srm is not None
        assert allocation_check.srm.observed == pinned_counts
        assert {row.group_id: row.assigned_units for row in snapshot.group_data} == pinned_counts
        for field in ("lift", "lower", "higher"):
            assert snapshot.readout_rows[0][field] == pytest.approx(
                live_pricing.readout_rows[0][field]
            )
        fresh = prepare_dashboard(experiment, resolver, config=live_pricing.config)
        assert fresh.allocation is not None
        assert fresh.allocation.observed == {"baseline": 700, "candidate": 600}
        assert fresh.readout_rows[0]["lower"] != snapshot.readout_rows[0]["lower"]
        resolver.close()
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("change", ["control", "window", "policy", "breakouts"])
def test_explore_rejects_same_name_binding_changes_before_query(
    dashboard_con, dashboard_definitions, live_pricing, tmp_path, change
):
    import shutil

    import yaml

    definitions = tmp_path / "definitions"
    shutil.copytree(dashboard_definitions, definitions)
    experiments_path = definitions / "experiments.yaml"
    experiments = yaml.safe_load(experiments_path.read_text())
    declared = next(item for item in experiments["experiments"] if item["name"] == "pricing_copy")
    if change == "control":
        declared["control_group"] = "candidate"
    elif change == "policy":
        declared["plan"]["alpha"] = 0.01
    elif change == "breakouts":
        declared["breakouts"] = [{"property": "country", "source": "event_log"}]
    else:
        metrics_path = definitions / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        metric = next(item for item in metrics["metrics"] if item["name"] == "purchase_rate")
        metric["window_days"] = 15
        metrics_path.write_text(yaml.safe_dump(metrics))
    experiments_path.write_text(yaml.safe_dump(experiments))
    experiment = _experiment(dashboard_con, definitions, "pricing_copy")
    with pytest.raises(InvalidRequestError) as error:
        load_explore(experiment, snapshot=live_pricing, metric="purchase_rate", view="daily_values")
    assert error.value.code == "dashboard.invalid_view"


@pytest.mark.slow
def test_explore_accepts_an_edit_to_a_saved_metric_the_experiment_does_not_declare(
    dashboard_con, dashboard_definitions, live_pricing, tmp_path
):
    import shutil

    import yaml

    definitions = tmp_path / "definitions"
    shutil.copytree(dashboard_definitions, definitions)
    metrics_path = definitions / "metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text())
    other = next(item for item in metrics["metrics"] if item["name"] == "session_seconds")
    other["window_days"] = 3
    metrics_path.write_text(yaml.safe_dump(metrics))
    experiment = _experiment(dashboard_con, definitions, "pricing_copy")
    loaded = load_explore(
        experiment, snapshot=live_pricing, metric="purchase_rate", view="daily_values"
    )
    assert {row.metric for row in loaded} == {"purchase_rate"}


@pytest.mark.slow
def test_even_split_experiment_binds_declared_metadata(live_storefront: DashboardSnapshot) -> None:
    assert live_storefront.experiment_name == "storefront_refresh"
    assert live_storefront.title == "storefront_refresh"
    assert (live_storefront.control_group, live_storefront.treatment_group) == (
        "control",
        "treatment",
    )
    assert live_storefront.primary_metric == "checkout_conversion"
    assert [metric.name for metric in live_storefront.metrics] == [
        "checkout_conversion",
        "revenue_per_user",
        "session_seconds",
    ]
    assert live_storefront.breakouts == (("event_log", "country"),)
    assert {row["inference"] for row in live_storefront.readout_rows} == {"fixed"}


def test_renamed_unequal_experiment_binds_its_own_metadata(live_pricing: DashboardSnapshot) -> None:
    assert live_pricing.experiment_name == "pricing_copy"
    assert live_pricing.title == "Pricing copy rewrite"
    assert (live_pricing.control_group, live_pricing.treatment_group) == ("baseline", "candidate")
    assert live_pricing.primary_metric == "purchase_rate"
    assert live_pricing.config.expected_allocation == {"baseline": 0.7, "candidate": 0.3}
    assert live_pricing.breakouts == ()
    assert {row["inference"] for row in live_pricing.readout_rows} == {"fixed"}


def test_allocation_check_uses_configured_shares(live_pricing: DashboardSnapshot) -> None:
    allocation = live_pricing.allocation
    assert allocation is not None
    assert allocation.expected == {"baseline": 0.7, "candidate": 0.3}
    assert allocation.observed == {"baseline": 700, "candidate": 300}
    assert allocation.is_srm is False


@pytest.mark.slow
def test_dashboard_readout_normalizes_relative_allocation_weights(
    dashboard_con, dashboard_definitions
) -> None:
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "pricing_copy")
    with resolver:
        snapshot = prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1}),
        )
    assert snapshot.allocation is not None
    assert snapshot.allocation.expected == {"baseline": 0.5, "candidate": 0.5}
    text = render_header(snapshot).text
    assert "baseline 50%" in text
    assert "candidate 50%" in text
    assert "baseline 100%" not in text
    assert "candidate 100%" not in text


def test_rows_carry_the_tested_alternative(live_storefront: DashboardSnapshot) -> None:
    tails = {row["metric"]: row["alternative"] for row in live_storefront.readout_rows}
    alternatives = {estimate.metric: estimate.alternative for estimate in live_storefront.estimates}
    assert tails == alternatives
    # A margin-less guardrail is tested on one side only.
    assert tails["session_seconds"] == "less"
    assert tails["checkout_conversion"] == "two-sided"


def test_rows_correspond_one_to_one_with_estimates(live_storefront: DashboardSnapshot) -> None:
    assert len(live_storefront.readout_rows) == len(live_storefront.estimates)
    for row, estimate in zip(live_storefront.readout_rows, live_storefront.estimates, strict=True):
        assert row["metric"] == estimate.metric
        assert row["group_id"] == estimate.group_id
        assert row["lift"] == pytest.approx(estimate.require_lift().value)


@pytest.mark.slow
def test_snapshot_owns_nested_readout_data(live_storefront: DashboardSnapshot) -> None:
    rows = [dict(row) for row in live_storefront.readout_rows]
    components: list[dict[str, Any]] = [{"bounds": {"lower": 0.25}, "labels": ["registered"]}]
    rows[0]["sequential_components"] = components
    snapshot = dataclasses.replace(live_storefront, readout_rows=tuple(rows))
    before = readout_csv(snapshot)
    components[0]["bounds"]["lower"] = 99.0
    components[0]["labels"].append("changed")
    assert readout_csv(snapshot) == before
    frozen = snapshot.readout_rows[0]["sequential_components"][0]
    with pytest.raises(TypeError):
        frozen["bounds"]["lower"] = 0.0
    with pytest.raises(AttributeError):
        frozen["labels"].append("changed")


def test_snapshot_allocation_maps_are_owned_and_immutable(
    live_storefront: DashboardSnapshot,
) -> None:
    assert live_storefront.allocation is not None
    original = live_storefront.allocation
    payload = original.model_dump()
    allocation = type(original).model_validate(payload)
    snapshot = dataclasses.replace(live_storefront, allocation=allocation)
    assert snapshot.allocation is not None
    before = snapshot.allocation.model_dump_json()
    for name in ("observed", "expected", "unit_counts"):
        payload[name]["control"] = 999
        with pytest.raises(TypeError):
            getattr(allocation, name)["control"] = 999
    assert snapshot.allocation.model_dump_json() == before
    restored = type(original).model_validate_json(allocation.model_dump_json())
    assert restored.observed == original.observed
    assert restored.expected == original.expected
    assert restored.unit_counts == original.unit_counts


def test_allocation_history_rows_are_immutable(live_storefront: DashboardSnapshot) -> None:
    assert live_storefront.allocation_history_refusal is None
    raw = [dict(live_storefront.allocation_history[0])]
    copied = dataclasses.replace(live_storefront, allocation_history=tuple(raw))
    original_count = copied.allocation_history[0]["n_daily"]
    raw[0]["n_daily"] += 100
    assert copied.allocation_history[0]["n_daily"] == original_count
    row = live_storefront.allocation_history[0]
    with pytest.raises(TypeError):
        row["n_daily"] = 0  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        live_storefront.allocation_history.append(row)  # ty: ignore[unresolved-attribute]


def test_allocation_history_reconciles_with_the_renamed_unequal_experiment(
    live_pricing: DashboardSnapshot,
) -> None:
    """The 70/30 renamed arms' real cumulative totals, not a derived guess."""
    assert live_pricing.allocation_history_refusal is None
    assert live_pricing.allocation is not None
    rows = live_pricing.allocation_history
    assert {row["experiment_id"] for row in rows} == {"pricing_copy"}
    assert {row["group_id"] for row in rows} == {"baseline", "candidate"}
    assert list(rows) == sorted(rows, key=lambda row: (row["ds"], row["group_id"]))
    final_totals = {
        group: max(row["n_cumulative"] for row in rows if row["group_id"] == group)
        for group in ("baseline", "candidate")
    }
    assert final_totals == live_pricing.allocation.observed


def test_prepared_config_rejects_mutation(live_pricing: DashboardSnapshot) -> None:
    with pytest.raises(TypeError):
        live_pricing.config.expected_allocation["baseline"] = 0.1  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        live_pricing.readout_rows[0]["lift"] = 0.0  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        live_pricing.primary_metric = "other"  # ty: ignore[invalid-assignment]


@pytest.mark.integration
@pytest.mark.slow
def test_mutating_caller_allocation_cannot_change_a_prepared_result(
    dashboard_con, dashboard_definitions
) -> None:
    allocation = {"baseline": 0.7, "candidate": 0.3}
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation=allocation)
    with resolver:
        snapshot = prepare_dashboard(experiment, resolver, config=config)

    allocation["baseline"] = 0.2
    allocation["candidate"] = 0.8

    assert snapshot.allocation is not None
    assert snapshot.allocation.expected == {"baseline": 0.7, "candidate": 0.3}
    assert snapshot.config.expected_allocation == {"baseline": 0.7, "candidate": 0.3}


def test_preparation_refuses_allocation_without_the_control_arm(
    dashboard_con, dashboard_definitions
) -> None:
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with resolver, pytest.raises(InvalidRequestError) as caught:
        prepare_dashboard(experiment, resolver, config=config)
    assert caught.value.code == "dashboard.invalid_config"
    assert caught.value.context["control"] == "baseline"


def test_preparation_surfaces_an_arm_label_absent_from_the_data(
    dashboard_con, dashboard_definitions
) -> None:
    """An unusable request keeps the source's own code, not a health badge."""
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation={"baseline": 0.7, "challenger": 0.3})
    with resolver, pytest.raises(InvalidRequestError) as caught:
        prepare_dashboard(experiment, resolver, config=config)
    assert caught.value.code == "estimation.diagnostics.expected_keys_do"
    assert "candidate" in caught.value.context["counts_keys"]  # ty: ignore[unsupported-operator]


def test_preparation_refuses_more_than_one_declared_primary(
    dashboard_con, dashboard_definitions
) -> None:
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "dual_primary")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with resolver, pytest.raises(CapabilityError) as caught:
        prepare_dashboard(experiment, resolver, config=config)
    assert caught.value.code == "dashboard.unsupported_experiment"
    assert caught.value.context["primaries"] == ("checkout_conversion", "revenue_per_user")


@backend_fixture()
def storefront_experiment(dashboard_con, dashboard_definitions):
    return _experiment(dashboard_con, dashboard_definitions, "storefront_refresh")


@backend_fixture()
def pricing_experiment(dashboard_con, dashboard_definitions):
    return _experiment(dashboard_con, dashboard_definitions, "pricing_copy")


def test_load_explore_refuses_an_undeclared_metric(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="not_a_metric",
            view="daily_values",
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["metric"] == "not_a_metric"


def test_load_explore_refuses_an_unknown_view(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="not_a_view",  # ty: ignore[invalid-argument-type]
        )
    assert caught.value.code == "dashboard.invalid_view"


def test_load_explore_refuses_daily_values_with_a_cumulative_maturity_flag(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    """Daily values are disjoint slices; a cumulative maturity flag cannot be honoured."""
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="daily_values",
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "daily_values"


def test_segment_view_refuses_a_cumulative_maturity_flag(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="segments",
            breakout=("event_log", "country"),
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "segments"


def test_load_explore_refuses_a_snapshot_and_source_describing_different_experiments(
    pricing_experiment, live_storefront: DashboardSnapshot
) -> None:
    """A mismatched pair must not silently render the other binding's stale data."""
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            pricing_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="daily_values",
        )
    assert caught.value.code == "dashboard.invalid_view"
    context = caught.value.context
    assert context["snapshot_experiment"] == "storefront_refresh"
    assert context["analysis_experiment"] == "pricing_copy"


def test_load_explore_segments_need_an_explicit_choice_when_breakouts_are_declared(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="segments",
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["declared"] == (("event_log", "country"),)


def test_load_explore_segments_refuse_an_undeclared_breakout(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="segments",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


def test_load_explore_segments_with_no_declared_breakouts_return_empty(
    pricing_experiment, live_pricing: DashboardSnapshot
) -> None:
    """A truthful empty state, not a query the experiment never declared."""
    result = load_explore(
        pricing_experiment, snapshot=live_pricing, metric="purchase_rate", view="segments"
    )
    assert list(result) == []


def test_load_explore_segments_filter_by_dimension_source_and_method_without_leaking_rows(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    """A row from another dimension, source, or method role must never leak in."""
    from increment._source_operations import DashboardExploreCapture
    from increment.breakout.estimates import BreakoutEstimates

    choice = ("event_log", "country")
    wanted = load_explore(
        storefront_experiment,
        snapshot=live_storefront,
        metric="checkout_conversion",
        view="segments",
        breakout=choice,
    )
    assert len(wanted) == 2
    decoys = [
        wanted[0].model_copy(update={"dimension": "plan", "dimension_value": "free"}),
        wanted[0].model_copy(update={"method_role": "component"}),
        wanted[0].model_copy(update={"source": "other_source"}),
    ]
    key = ("assigned", "segments", "checkout_conversion", False, choice)
    captured = dataclasses.replace(
        live_storefront,
        explore={
            **live_storefront.explore,
            key: DashboardExploreCapture.answered(
                key, BreakoutEstimates([*wanted, *decoys]), collection=BreakoutEstimates
            ),
        },
    )
    filtered = load_explore(
        storefront_experiment,
        snapshot=captured,
        metric="checkout_conversion",
        view="segments",
        breakout=choice,
    )
    assert isinstance(filtered, BreakoutEstimates)
    assert len(filtered) == 2
    assert {e.dimension for e in filtered} == {"country"}
    assert {e.source for e in filtered} == {"event_log"}
    assert {e.method_role for e in filtered} == {"decision"}


def test_load_explore_segments_with_no_metric_returns_every_declared_metric(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    """``metric=None`` covers every declared metric via one real breakout call."""
    from increment.breakout.estimates import BreakoutEstimates

    result = load_explore(
        storefront_experiment,
        snapshot=live_storefront,
        metric=None,
        view="segments",
        breakout=("event_log", "country"),
    )
    assert isinstance(result, BreakoutEstimates)
    declared = tuple(metric.name for metric in live_storefront.metrics)
    assert declared == ("checkout_conversion", "revenue_per_user", "session_seconds")
    assert {e.metric for e in result} == set(declared)
    # Exactly one non-control arm x two segments (US, CA) per declared metric.
    for metric in declared:
        assert sum(1 for e in result if e.metric == metric) == 2


def _trajectory_fingerprint(rows) -> list[tuple[Any, ...]]:
    """A trajectory as (segment, day, arm, point) so any slice drift is visible."""
    out = []
    for row in rows:
        point = row.lift if hasattr(row, "lift") else row.value
        out.append(
            (
                row.dimension_value,
                row.ds,
                row.group_id,
                None if point is None else (point.value, point.lb, point.ub, point.level),
            )
        )
    return out


_COUNTRY_SEGMENT = BreakoutIdentity("country", None)


@pytest.mark.parametrize(
    ("view", "engine", "kwargs"),
    [
        ("cumulative_lift", Scope(view="asof", grain="asof", segment=_COUNTRY_SEGMENT), {}),
        (
            "cumulative_values",
            Scope(view="asof", grain="asof", values=True, segment=_COUNTRY_SEGMENT),
            {},
        ),
        (
            "cumulative_values",
            Scope(
                view="asof",
                grain="asof",
                values=True,
                completion_policy=True,
                segment=_COUNTRY_SEGMENT,
            ),
            {"completed_windows_only": True},
        ),
        (
            "daily_values",
            Scope(view="daily", grain="daily", values=True, segment=_COUNTRY_SEGMENT),
            {},
        ),
    ],
    ids=["asof-lift", "asof-values", "asof-values-completed", "daily-values"],
)
def test_declared_breakout_temporal_views_are_the_engine_segment_trajectories(
    dashboard_con,
    dashboard_definitions,
    live_storefront: DashboardSnapshot,
    view: ExploreView,
    engine: Scope,
    kwargs: dict,
) -> None:
    from tests.analysis_factory import view_readout

    metric = "checkout_conversion"
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "storefront_refresh")
    with resolver:
        segmented = load_explore(
            experiment,
            snapshot=live_storefront,
            metric=metric,
            view=view,
            breakout=COUNTRY,
            **kwargs,
        )
        direct = view_readout(experiment, resolver, engine, metrics=[metric])
    assert segmented, "a declared breakout must produce a segment trajectory"
    actual, expected = _trajectory_fingerprint(segmented), _trajectory_fingerprint(direct)
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row[:3] == expected_row[:3]
        if expected_row[3] is None:
            assert actual_row[3] is None
        else:
            assert actual_row[3] == pytest.approx(expected_row[3])
    assert {row.dimension_value for row in segmented} == {"US", "CA"}
    assert {(row.dimension, row.source) for row in segmented} == {("country", "event_log")}
    whole = load_explore(experiment, snapshot=live_storefront, metric=metric, view=view, **kwargs)
    assert {row.dimension for row in whole} == {None}
    assert _trajectory_fingerprint(whole) != _trajectory_fingerprint(segmented)


def test_completed_windows_gate_changes_the_segment_absolute_series(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    def load(*, complete: bool):
        return load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="cumulative_values",
            breakout=COUNTRY,
            completed_windows_only=complete,
        )

    provisional, mature = load(complete=False), load(complete=True)
    assert min(row.ds for row in mature) > min(row.ds for row in provisional)
    assert not any(row.value is None for row in mature)


def test_temporal_breakouts_refuse_an_undeclared_dimension(
    storefront_experiment, live_storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_experiment,
            snapshot=live_storefront,
            metric="checkout_conversion",
            view="cumulative_lift",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


@pytest.mark.slow
def test_exploring_never_changes_the_confirmatory_snapshot(
    live_storefront: DashboardSnapshot, dashboard_con, dashboard_definitions
) -> None:

    before = (live_storefront.estimates, live_storefront.readout_rows, readout_csv(live_storefront))
    build_payload(
        _experiment(dashboard_con, dashboard_definitions, "storefront_refresh"),
        snapshot=live_storefront,
    )
    after = (live_storefront.estimates, live_storefront.readout_rows, readout_csv(live_storefront))
    assert after == before


def _copied_definitions(tmp_path: Path, definitions: str, name: str, change: Any) -> Path:
    """The shared definitions with one experiment's declaration changed by ``change``."""
    import shutil

    import yaml

    copied = tmp_path / "definitions"
    shutil.copytree(definitions, copied)
    path = copied / "experiments.yaml"
    document = yaml.safe_load(path.read_text())
    change(next(item for item in document["experiments"] if item["name"] == name))
    path.write_text(yaml.safe_dump(document))
    return copied


@pytest.mark.slow
@pytest.mark.integration
def test_an_engine_refusal_is_visible_for_its_own_state_only(
    dashboard_con, dashboard_definitions, tmp_path
) -> None:

    def segmented_bh(declared: dict[str, Any]) -> None:
        declared["plan"]["view_multiplicity"] = {"correction": "bh"}

    definitions = _copied_definitions(
        tmp_path, dashboard_definitions, "storefront_refresh", segmented_bh
    )
    experiment, resolver = _bind(dashboard_con, definitions, "storefront_refresh")
    with resolver:
        captured = prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        )
    payload = build_payload(experiment, snapshot=captured)
    refused = payload["explore"]["country"]["checkout_conversion"]["cumulative_lift"]
    assert refused["pointCount"] == 0
    assert "facade.analysis.bh_segmented_asof" in refused["html"]
    assert payload["explore"]["overall"]["checkout_conversion"]["cumulative_lift"]["pointCount"] > 0
    assert (
        payload["explore"]["country"]["checkout_conversion"]["cumulative_values"]["pointCount"] > 0
    )


@pytest.mark.integration
@pytest.mark.slow
def test_health_renders_not_applicable_allocation_as_not_checked(
    dashboard_con, dashboard_definitions, tmp_path
) -> None:
    from increment.dashboard._app import health_status

    def undeclared_scheme(declared: dict[str, Any]) -> None:
        del declared["allocation_scheme"]

    definitions = _copied_definitions(
        tmp_path, dashboard_definitions, "pricing_copy", undeclared_scheme
    )
    experiment, resolver = _bind(dashboard_con, definitions, "pricing_copy")
    with resolver:
        snapshot = prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"baseline": 0.7, "candidate": 0.3}),
        )

    assert health_status(snapshot)["kind"] == "not_checked"
    assert snapshot.allocation is None
    assert snapshot.allocation_refusal is not None
    code, reason = snapshot.allocation_refusal
    assert code == "dashboard.allocation_not_applicable"
    assert snapshot.allocation_history == ()
    assert snapshot.allocation_history_refusal == snapshot.allocation_refusal
    html = render_health(snapshot).text
    assert "not checked" in html.lower()
    assert reason in html
    technical = re.search(
        r"<details[^>]*><summary>Technical details</summary>(.*?)</details>", html, re.DOTALL
    )
    assert technical is not None
    assert "integrity.allocation_scheme_missing" in technical.group(1)


@pytest.mark.slow
@pytest.mark.integration
def test_unexpected_engine_failures_are_not_swallowed(
    dashboard_con, dashboard_definitions, monkeypatch
) -> None:
    from increment.query import native_source

    failure = RuntimeError("warehouse connection lost")
    moments = native_source._DaySource.moments

    def broken(self, metric, *, grain="daily", **kwargs: Any) -> Any:
        if grain == "daily":
            raise failure
        return moments(self, metric, grain=grain, **kwargs)

    monkeypatch.setattr(native_source._DaySource, "moments", broken)
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "storefront_refresh")
    with resolver, pytest.raises(RuntimeError) as caught:
        prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        )
    assert caught.value is failure


@pytest.mark.slow
def test_dashboard_snapshot_and_csv_retain_scoped_readout_metadata(
    dashboard_con, dashboard_definitions
):
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "storefront_refresh")
    with resolver:
        snapshot = prepare_dashboard(
            experiment,
            resolver,
            config=DashboardConfig(expected_allocation={"control": 0.7, "treatment": 0.3}),
        )
    assert snapshot.allocation is not None and snapshot.allocation.is_srm

    scoped = snapshot.readout_collection
    assert scoped is not None
    assert scoped.metadata is not None
    assert scoped.metadata.scope.by_source
    primary_only = scoped.filter(lambda row: row.metric == "checkout_conversion")
    rows = enriched_rows(primary_only)
    primary = next(row for row in rows if row["metric"] == "checkout_conversion")
    assert primary["view_partial"] is True
    assert primary["analysis_population"] == "assigned"
    assert primary["integrity_status"] == "not_checked_missing_declaration"
    hidden = json.loads(primary["hidden_cells"])
    assert any(item["cell"]["metric"] == "session_seconds" for item in hidden)

    dashboard_snapshot = dataclasses.replace(
        snapshot,
        estimates=tuple(primary_only),
        readout_collection=primary_only,
        population_collections={"assigned": primary_only},
        population_estimates={"assigned": tuple(primary_only)},
        readout_rows=rows,
    )
    exported = list(csv.DictReader(io.StringIO(readout_csv(dashboard_snapshot).decode("utf-8"))))
    assert len(exported) == 1
    row = exported[0]
    assert row["view_partial"] == "True"
    assert row["analysis_population"] == "assigned"
    assert row["integrity_status"] == "not_checked_missing_declaration"
    exported_hidden = json.loads(row["hidden_cells"])
    assert any(item["cell"]["metric"] == "session_seconds" for item in exported_hidden)
    assert row["hidden_cells"] == json.dumps(
        exported_hidden, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert json.loads(row["retained_family"])
    summary_html = render_header(dashboard_snapshot).text
    assert "partial" in summary_html
    assert "session_seconds" in summary_html
    assert "SRM not checked" in summary_html and "allocation" in summary_html
    readout_html = render_results(dashboard_snapshot).text
    assert "partial" in readout_html
    for surface in (summary_html, readout_html):
        assert "Dashboard allocation check" in surface
        assert "configured expected control 70% / treatment 30%" in surface
        assert "mismatch detected" in surface
        assert "integrity.allocation_missing_or_nonconstant" in surface
        assert surface.index("mismatch detected") < surface.index("Analysis assignment check")


def _assert_added_exploratory_readout(snapshot):
    from html import escape

    from increment.dashboard import render_metric_details

    captures = [
        capture
        for key, capture in snapshot.explore.items()
        if key[1] == "overview_whole" and key[2] is None and key[4] is None
    ]
    assert captures
    collection = captures[0].load()
    assert collection.metadata is not None
    advisory = next(
        item
        for record in collection.metadata.cells
        if record.cell.metric == "purchase_rate"
        for item in record.advisories
        if item.code == "estimation.engine.small_total_clusters"
    )
    details = render_metric_details(snapshot, metric="purchase_rate").text
    assert "purchase_rate" in details
    assert advisory.code in details
    assert escape(advisory.message) in details
    assert snapshot.overview is not None
    added = next(row for row in snapshot.overview.exploratory if row.metric == "purchase_rate")
    assert added.note is not None and added.note in details
    assert "session_seconds" not in details


@pytest.mark.slow
def test_dashboard_round_trips_real_unavailable_cell_and_advisories(tmp_path: Path, dashboard_con):
    facts = (
        FACT_SOURCES.replace(
            "SELECT * FROM analytics.event_log",
            "SELECT *, concat(group_id, '_', country_code) AS store_id FROM analytics.event_log",
        )
        .replace(
            "    properties:\n",
            "    properties:\n"
            "      - name: store_id\n"
            "        column: store_id\n"
            "        dtype: string\n"
            "        as_of: static\n"
            "        description: Randomization cluster\n",
            1,
        )
        .replace(
            "    properties:\n",
            "      - name: no_data\n"
            "        column: revenue\n"
            "        description: No observations\n"
            "    properties:\n",
            1,
        )
    )
    metrics = (
        METRICS
        + """
  - type: ratio
    name: empty_guardrail
    entity: user_id
    preferred_direction: decrease
    numerator: {fact: purchase, aggregation: sum, window_days: 14}
    denominator: {fact: no_data, aggregation: count, window_days: 14}
"""
    )
    experiments = (
        EXPERIMENTS.replace(
            "    unit: user_id\n",
            "    unit: user_id\n    cluster: store_id\n",
            1,
        )
        .replace(
            "      guardrails:\n        - session_seconds\n",
            "      guardrails:\n        - session_seconds\n        - empty_guardrail\n",
            1,
        )
        .replace("    breakouts:\n      - property: country\n        source: event_log\n", "")
    )
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    for name, content in (
        ("fact_sources.yaml", facts),
        ("exposures.yaml", EXPOSURES),
        ("metrics.yaml", metrics),
        ("experiments.yaml", experiments),
    ):
        (definitions / name).write_text(content)

    from increment.dashboard._app import build_payload, document

    experiment, resolver = _bind(dashboard_con, definitions, "storefront_refresh")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snapshot = prepare_dashboard(
                experiment,
                resolver,
                config=DashboardConfig(
                    expected_allocation={"control": 0.9999, "treatment": 0.0001},
                    exploratory_metrics=("purchase_rate",),
                ),
            )
        assert snapshot.allocation is not None and snapshot.allocation.is_srm
        failed = next(row for row in snapshot.readout_rows if row["metric"] == "empty_guardrail")
        unavailable = json.loads(failed["unavailable_cells"])
        reason = next(item for item in unavailable if item["cell"]["metric"] == "empty_guardrail")
        assert reason["code"] == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
        advisory = next(
            item
            for row in snapshot.readout_rows
            for item in json.loads(row["advisories"])
            if item["code"] == "estimation.engine.small_total_clusters"
        )
        assert "below 40" in advisory["message"]

        summary_html = render_header(snapshot).text
        readout_html = render_results(snapshot).text
        payload = build_payload(experiment, snapshot=snapshot)
        report_text = payload["report"]["qualifications"]
        for surface in (summary_html, readout_html, report_text):
            assert reason["code"] in surface
            assert advisory["code"] in surface
            assert "below 40" in surface
        assert advisory["message"] in report_text
        shell = document(payload)
        assert payload["qualifications"] == payload["report"]["qualifications"]
        assert "integrity.allocation_missing_or_nonconstant" in payload["qualifications"]
        assert shell.index('id="hero-meta"') < shell.index('id="hero-qualifications"')
        assert "hero-meta').textContent = str(data.meta)" in shell
        assert payload["meta"]
        assert shell.index('id="hero-qualifications"') < shell.index('id="hero-primary-effect"')
        assert shell.index('id="report-qualifications"') < shell.index('id="report-primary-effect"')
        _assert_added_exploratory_readout(snapshot)

        exported = list(csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8"))))
        exported_failed = next(row for row in exported if row["metric"] == "empty_guardrail")
        exported_reason = next(
            item
            for item in json.loads(exported_failed["unavailable_cells"])
            if item["cell"]["metric"] == "empty_guardrail"
        )
        assert exported_reason["code"] == reason["code"]
        exported_advisory = next(
            item
            for row in exported
            for item in json.loads(row["advisories"])
            if item["code"] == "estimation.engine.small_total_clusters"
        )
        assert exported_advisory["message"] == advisory["message"]
    finally:
        resolver.close()


@pytest.mark.slow
def test_clustered_overview_fallback_preserves_each_metric_collection(
    tmp_path: Path, dashboard_con
):
    facts = FACT_SOURCES.replace(
        "SELECT * FROM analytics.event_log",
        "SELECT *, concat(group_id, '_', country_code) AS store_id FROM analytics.event_log",
    ).replace(
        "    properties:\n",
        "    properties:\n"
        "      - name: store_id\n"
        "        column: store_id\n"
        "        dtype: string\n"
        "        as_of: static\n",
        1,
    )
    metrics = (
        METRICS
        + """
  - type: mean
    name: added_revenue
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14
  - type: mean
    name: added_duration
    entity: user_id
    preferred_direction: decrease
    fact: session_end
    aggregation: sum
    window_days: 14
  - type: quantile
    name: p90_revenue
    quantile: 0.9
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 14
"""
    )
    experiments = EXPERIMENTS.replace(
        "    unit: user_id\n", "    unit: user_id\n    cluster: store_id\n", 1
    ).replace("    breakouts:\n      - property: country\n        source: event_log\n", "")
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    for name, content in (
        ("fact_sources.yaml", facts),
        ("exposures.yaml", EXPOSURES),
        ("metrics.yaml", metrics),
        ("experiments.yaml", experiments),
    ):
        (definitions / name).write_text(content)

    from typing import cast

    from increment.breakout.estimates import LiftEstimates
    from increment.dashboard import render_dashboard

    added = ("added_revenue", "added_duration", "p90_revenue")
    experiment, resolver = _bind(dashboard_con, definitions, "storefront_refresh")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snapshot = prepare_dashboard(
                experiment,
                resolver,
                config=DashboardConfig(
                    expected_allocation={"control": 0.5, "treatment": 0.5},
                    exploratory_metrics=added,
                ),
            )
        capture = snapshot.explore[("assigned", "overview_whole", None, False, None)]
        collection = cast(LiftEstimates, capture.load())
        assert {row.metric for row in collection} >= set(added[:2])
        assert collection.metadata is not None
        assert {record.cell.metric for record in collection.metadata.cells} >= set(added[:2])

        overview = snapshot.overview
        assert overview is not None
        assert {row.metric for row in overview.exploratory} >= set(added[:2])
        refusal = next(item for item in overview.refusals if item[0] == "p90_revenue")
        assert refusal[2].code == "arm.metric.quantile_cluster"

        details = render_metric_details(snapshot, metric="added_revenue").text
        assert "added_revenue" in details
        assert render_dashboard(experiment, snapshot=snapshot) is not None
        payload = build_payload(experiment, snapshot=snapshot)
        notes = payload["overview"]["overall"]["notes"]
        assert any("p90_revenue" in note and refusal[2].code in note for note in notes)
    finally:
        resolver.close()


_SAVED = (
    _METRICS
    + f"""
  - type: mean
    name: {ADDED}
    description: Added purchase count per enrolled user.
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: count
    window_days: 14
"""
)


QUANTILE = "p90_revenue"


_WITH_QUANTILE = (
    _SAVED
    + f"""
  - type: quantile
    name: {QUANTILE}
    quantile: 0.9
    entity: user_id
    fact: purchase
    aggregation: sum
"""
)


SECOND = "page_views_per_user"


_TWO_SAVED = (
    _SAVED
    + f"""
  - type: mean
    name: {SECOND}
    entity: user_id
    preferred_direction: increase
    fact: page_view
    aggregation: count
    window_days: 14
"""
)


def _config(*added: str) -> DashboardConfig:
    return DashboardConfig(
        expected_allocation={"control": 0.5, "treatment": 0.5}, exploratory_metrics=added
    )


@pytest.fixture(scope="module")
def _selected_family_snapshot(tmp_path_factory):
    # Enough units that the exploratory BH family selects at least one cell.
    with _workspace(
        tmp_path_factory.mktemp("overview-selected-snapshot"), units=1000, metrics=_SAVED
    ) as (_, experiment, resolver):
        return prepare_dashboard(experiment, resolver, config=_config(ADDED))


@pytest.mark.integration
@pytest.mark.slow
def test_one_family_covers_added_metrics_and_every_segment_cell(_selected_family_snapshot):
    snapshot = _selected_family_snapshot
    overview = snapshot.overview
    assert overview is not None and overview.family_refusal is None
    segments = overview.segments[COUNTRY]
    cells = [*overview.exploratory, *segments]
    tested = [cell for cell in cells if cell.family_size is not None]
    assert {cell.metric for cell in overview.exploratory} == {ADDED}
    assert {cell.metric for cell in segments} >= {ADDED, "checkout_conversion"}
    assert len(tested) == overview.family_size > len(overview.exploratory)
    assert {cell.family_size for cell in tested} == {overview.family_size}
    selected = next(cell for cell in tested if cell.discovery is True)
    assert selected.lift is not None
    restored = type(selected).model_validate_json(selected.model_dump_json())
    assert restored.lift is not None
    assert restored.discovery is True
    assert restored.family_procedure == "bh"
    assert restored.family_q == pytest.approx(selected.family_q)
    assert restored.family_size == selected.family_size
    assert restored.family_axes == selected.family_axes
    assert restored.lift.level == pytest.approx(selected.lift.level)
    assert restored.lift.lb == pytest.approx(selected.lift.lb)
    assert restored.lift.ub == pytest.approx(selected.lift.ub)
    from increment.breakout.estimates import BreakoutEstimate
    from increment.dashboard._html import overview_table
    from increment.estimation.results import LiftEstimate

    if isinstance(selected, BreakoutEstimate):
        assert isinstance(restored, BreakoutEstimate)
        segments = dict(overview.segments)
        segments[COUNTRY] = tuple(restored if row is selected else row for row in segments[COUNTRY])
        restored_overview = dataclasses.replace(overview, segments=segments)
        scope = COUNTRY
    else:
        assert isinstance(restored, LiftEstimate)
        restored_overview = dataclasses.replace(
            overview,
            exploratory=tuple(restored if row is selected else row for row in overview.exploratory),
        )
        scope = None
    restored_snapshot = dataclasses.replace(snapshot, overview=restored_overview)
    representation = repr(restored)
    rendered = overview_table(restored_snapshot, scope)
    assert "Benjamini-Hochberg" in representation
    assert "FCR-adjusted" in representation
    assert "independence or positive dependence" in representation
    assert "Benjamini-Hochberg" in rendered
    assert {cell.family_q for cell in tested} == {overview.family_q}
    # The marker on every exploratory overview row is its family discovery.
    for row in overview_rows(snapshot, COUNTRY):
        if row["metric"] == ADDED or row.get("dimension") is not None:
            assert row["stat_sig"] is bool(row.get("discovery"))


@pytest.mark.integration
@pytest.mark.slow
def test_added_metrics_keep_their_own_absolute_value_series(tmp_path):

    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        explore = build_payload(experiment, snapshot=snapshot)["explore"]["overall"][ADDED]
        for view in ("cumulative_values", "daily_values", "cumulative_lift"):
            assert explore[view]["pointCount"] > 0


@pytest.mark.integration
@pytest.mark.slow
def test_a_metric_the_breakout_refuses_keeps_every_other_metrics_segments(tmp_path):
    with _workspace(tmp_path, metrics=_WITH_QUANTILE) as (_, experiment, resolver):
        # The quantile metric is offered but not shown; its refusal is still reported.
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        shown = {cell.metric for cell in overview.segments[COUNTRY]}
        assert {ADDED, "checkout_conversion"} <= shown
        assert QUANTILE not in shown
        refused = {(metric, place) for metric, place, _ in overview.refusals}
        assert (QUANTILE, "country") in refused
        assert {metric for metric, _, _ in overview.refusals} == {QUANTILE}
        assert any(QUANTILE in note for note in overview_notes(snapshot, COUNTRY))


@pytest.mark.integration
@pytest.mark.slow
def test_two_sources_of_one_property_are_separate_comparisons(tmp_path):
    sources = ("event_log", "profiles")
    with _workspace(tmp_path, metrics=_SAVED, breakout_sources=sources) as (
        _,
        experiment,
        resolver,
    ):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        by_source = {source: overview.segments[(source, "country")] for source in sources}
        assert all(by_source.values())
        tested = [
            cell
            for cells in (overview.exploratory, *by_source.values())
            for cell in cells
            if cell.family_size is not None
        ]
        assert len(tested) == overview.family_size


@pytest.mark.integration
@pytest.mark.slow
def test_choosing_what_to_show_never_changes_the_family(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        hidden = prepare_dashboard(experiment, resolver, config=_config())
        shown = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert shown.readout_rows == hidden.readout_rows
        assert [model.name for model in hidden.offered_metrics] == [ADDED]
        assert hidden.exploratory_metrics == () and hidden.overview is not None
        assert shown.overview is not None
        assert hidden.overview.family_size == shown.overview.family_size

        def cells(overview) -> list:
            rows = [*overview.exploratory, *(r for s in overview.segments.values() for r in s)]
            return [row.model_dump(mode="json") for row in rows]

        assert cells(hidden.overview) == cells(shown.overview)


@pytest.mark.integration
def test_an_undeclared_exploratory_metric_is_refused_before_preparation(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        con.disconnect()
        for name in ("not_a_saved_metric", "checkout_conversion"):
            with pytest.raises(InvalidRequestError) as caught:
                prepare_dashboard(experiment, resolver, config=_config(name))
            assert caught.value.code == "dashboard.invalid_config"
            assert caught.value.context["available"] == (ADDED,)


@pytest.mark.slow
@pytest.mark.integration
def test_added_metrics_come_from_the_pinned_read(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert snapshot.overview is not None
        before = snapshot.overview.exploratory
        _double_the_treatment_arm(con)
        assert snapshot.overview.exploratory == before
        fresh = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert fresh.overview is not None
        dumped = [row.model_dump(mode="json") for row in fresh.overview.exploratory]
        assert dumped != [row.model_dump(mode="json") for row in before]


def _page_payload(page: str) -> dict:
    import json
    import re

    match = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', page)
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.slow
@pytest.mark.integration
def test_the_dashboard_widget_shows_added_metrics_without_reading_again(tmp_path):
    from increment.dashboard._app import DashboardWidget

    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config())
        widget = DashboardWidget(experiment, snapshot=snapshot)
        offered = _page_payload(widget.document)["exploratory"]
        assert [item["key"] for item in offered["available"]] == [ADDED]
        assert offered["selected"] == []

        # A later warehouse change never reaches the page: selection re-renders this snapshot.
        _double_the_treatment_arm(con)
        widget.exploratory_metrics = [ADDED]
        assert widget.status == ""
        payload = _page_payload(widget.document)
        assert payload["exploratory"]["selected"] == [ADDED]
        assert ADDED in {metric["key"] for metric in payload["metrics"]}
        assert widget._snapshot.readout_rows == snapshot.readout_rows
        assert widget._snapshot.overview == snapshot.overview
        assert widget._snapshot.computed_at == snapshot.computed_at

        page = widget.document
        widget.exploratory_metrics = ["not_a_saved_metric"]
        assert "dashboard.invalid_config" in widget.status
        assert widget.exploratory_metrics == [ADDED]
        assert widget.document == page

        widget.exploratory_metrics = []
        assert widget.status == ""
        assert ADDED not in {metric["key"] for metric in _page_payload(widget.document)["metrics"]}


_REPORT_ONLY = (
    _SAVED
    + """
  - type: total
    name: total_revenue
    fact: purchase
    aggregation: sum
  - type: active
    name: active_users
    entity: user_id
    fact: page_view
"""
)


@pytest.mark.integration
def test_only_estimable_saved_metrics_are_offered(tmp_path):
    from increment.workflows import available_metrics

    with _workspace(tmp_path, metrics=_REPORT_ONLY) as (con, experiment, resolver):
        assert [metric.name for metric in available_metrics(experiment, resolver)] == [ADDED]
        con.disconnect()
        with pytest.raises(InvalidRequestError) as caught:
            prepare_dashboard(experiment, resolver, config=_config("total_revenue"))
        assert caught.value.code == "dashboard.invalid_config"


_OBSERVATIONAL = """    design:
      mechanism: observational
      covariates:
        - {property: country, source: event_log}
"""


@pytest.mark.integration
def test_a_non_randomized_design_is_refused_before_any_read(tmp_path, monkeypatch):
    import tests.test_dashboard_capture as capture

    declared = capture._experiments
    monkeypatch.setattr(
        capture, "_experiments", lambda **options: declared(**options) + _OBSERVATIONAL
    )
    with _workspace(tmp_path) as (con, experiment, resolver):
        con.disconnect()
        with pytest.raises(CapabilityError) as caught:
            prepare_dashboard(experiment, resolver, config=_config())
        assert caught.value.code == "dashboard.unsupported_experiment"
        assert caught.value.context["mechanism"] == "observational"


def _flat(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        return {
            k: v for key, item in value.items() for k, v in _flat(item, f"{prefix}{key}.").items()
        }
    return {prefix.rstrip("."): value}


def _rows(collection) -> list[dict[str, object]]:
    rows = [_flat(row.model_dump(mode="json")) for row in collection]
    rows = [
        {key: value for key, value in row.items() if key not in {"source_snapshot_id", "family_id"}}
        for row in rows
    ]
    return sorted(
        rows, key=lambda row: tuple(str(row.get(k)) for k in ("ds", "group_id", "dimension_value"))
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["bonferroni", "bh"])
@pytest.mark.integration
def test_offered_metrics_read_together_match_each_metric_read_alone(
    tmp_path, monkeypatch, correction
):
    import tests.test_dashboard_capture as capture
    from increment import Scope
    from increment.dashboard import load_explore
    from increment.evidence import BreakoutIdentity
    from tests.analysis_factory import view_readout
    from tests.parity_harness.comparison import nested_close

    monkeypatch.setattr(
        capture,
        "_PLAN",
        capture._PLAN.replace("correction: bonferroni", f"correction: {correction}"),
    )
    source, dimension = COUNTRY
    alone: dict[tuple[ExploreView, tuple[str, str] | None], Scope] = {
        ("cumulative_lift", None): Scope(view="asof", grain="asof"),
        ("cumulative_values", None): Scope(view="asof", grain="asof", values=True),
        ("daily_values", None): Scope(view="daily", grain="daily", values=True),
        ("segments", COUNTRY): Scope(
            view="breakout", grain="total", segment=BreakoutIdentity(dimension, source)
        ),
    }
    # Enough units that a segment is a BH discovery, so a joint family would move its threshold.
    with _workspace(tmp_path, metrics=_TWO_SAVED, units=2000) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED, SECOND))
        discoveries = 0
        for name in (ADDED, SECOND):
            for (view, breakout), scope in alone.items():
                expected = view_readout(
                    experiment, resolver, scope, metrics=[], exploratory_metrics=[name]
                )
                captured = load_explore(
                    experiment, snapshot=snapshot, metric=name, view=view, breakout=breakout
                )
                want, got = _rows(expected), _rows(captured)
                assert len(got) == len(want) > 0
                assert nested_close(want, got)
                if view == "segments":
                    discoveries += sum(bool(getattr(row, "discovery", None)) for row in captured)
        assert discoveries > 0 or correction == "bonferroni"
