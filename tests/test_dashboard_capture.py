"""Explore data is captured with the headline, not read from the warehouse afterwards.

``prepare_dashboard`` pins the warehouse once. Every Explore choice the dashboard offers must
come from that same pinned read, so a later warehouse change cannot make the headline and the
trajectories describe different data, and a re-preparation must deliberately see the new data.

Consumers are checked on frozen preparations: real ``prepare_dashboard`` output saved under
``tests/fixtures/dashboard`` together with what the public readout answered for every Explore
request at the same moment. ``test_frozen_snapshot_matches_a_fresh_capture`` pins each saved
preparation to a fresh capture of its source; regenerate them with
``uv run --extra dashboard python -m tests.test_dashboard_capture --freeze``. The live tests
here are the capture contracts themselves: the pin, a change after preparation, and a source
that refuses one breakout.
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
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment import View
from increment.breakout.estimates import BreakoutEstimates, DailyMetricValues, LiftEstimates
from increment.dashboard import (
    DashboardConfig,
    DashboardSnapshot,
    load_explore,
    prepare_dashboard,
    readout_csv,
    render_dashboard,
    render_health,
    render_metric_details,
    render_results,
)
from increment.dashboard._app import DashboardWidget, build_payload, document, health_status
from increment.dashboard._data import enriched_rows, show_exploratory_metrics
from increment.dashboard._html import overview_notes, overview_rows, overview_table
from increment.errors import (
    CapabilityError,
    CodedError,
    IncrementRuntimeWarning,
    InvalidRequestError,
)
from increment.evidence import BreakoutIdentity
from increment.workflows import capture_sequential
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import view_readout
from tests.dashboard_snapshots import (
    ADDED,
    COUNTRY,
    capture_differences,
    captured_form,
    frozen,
    preparation_differences,
    write_frozen,
)
from tests.parity_harness.comparison import TOLERANCE

# Small deterministic fixtures legitimately trip estimator reliability warnings; the subject
# here is which data is read, not estimator reliability.
pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

CONFIG = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})


def _json_cell(row: Mapping[str, object], name: str) -> Any:
    cell = row[name]
    assert isinstance(cell, str)
    return json.loads(cell)


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
        - metric: session_seconds
          margin: 0.02
"""


def _experiments(*, breakout_sources: tuple[str | None, ...], plan: str = _PLAN) -> str:
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
{plan}{breakout}"""


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
    plan: str = _PLAN,
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
        ("experiments.yaml", _experiments(breakout_sources=breakout_sources, plan=plan)),
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
PROFILES = ("profiles", "country")
# (view, completed windows only), including every state the dashboard can ask for.
_STATES = (
    ("cumulative_lift", False),
    ("cumulative_lift", True),
    ("cumulative_values", False),
    ("cumulative_values", True),
    ("daily_values", False),
    ("segments", False),
)
# Identities of one preparation's read, which another preparation of changed data may not share.
_READ_IDENTITIES = frozenset({"source_snapshot_id", "family_id"})


def _may_differ_from_engine(request: tuple[Any, ...]) -> frozenset[str]:
    """Fields left out when a loaded view is compared with the engine's read of the request.

    The whole-experiment lift family and daily values of declared metrics are the caller's own
    read, identities included. Other views are narrowed to one declared breakout or taken from
    a wider read, so their values, not their read identities, are what must agree.
    """
    view, metric, _, breakout = request
    same_read = breakout is None and view in ("cumulative_lift", "daily_values")
    return frozenset() if same_read and metric in (None, *_NAMES) else _READ_IDENTITIES


def _offered_shown(snapshot: DashboardSnapshot) -> DashboardSnapshot:
    """``snapshot`` as a caller who adds every offered metric sees it."""
    return show_exploratory_metrics(
        snapshot, tuple(model.name for model in snapshot.offered_metrics)
    )


_IDENTITY = (
    "ds",
    "ds_basis",
    "metric",
    "group_id",
    "method",
    "method_role",
    "estimand",
    "dimension",
    "dimension_value",
    "source",
)


def _ordered(rows) -> tuple[Any, ...]:
    """Warehouse reductions are unordered; rows are compared in identity order."""
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                [str(getattr(row, name, None)) for name in _IDENTITY],
                repr(captured_form(row)),
            ),
        )
    )


def _explore_requests(snapshot: DashboardSnapshot) -> Iterator[tuple[Any, ...]]:
    """Every (view, metric, completed windows only, breakout) a caller can load."""
    names = [model.name for model in (*snapshot.metrics, *snapshot.offered_metrics)]
    for view, complete in _STATES:
        for metric in (None, *names):
            for breakout in (None, *snapshot.breakouts):
                if view == "segments" and breakout is None:
                    continue  # a segment view is only defined for a declared choice
                yield view, metric, complete, breakout


def _explored(experiment, snapshot: DashboardSnapshot, request: tuple[Any, ...]) -> Any:
    """One loaded view as (collection type, rows), or the refusal it raises."""
    view, metric, complete, breakout = request
    try:
        loaded = load_explore(
            experiment,
            snapshot=snapshot,
            metric=metric,
            view=view,
            completed_windows_only=complete,
            breakout=breakout,
        )
    except CodedError as refusal:
        return refusal
    return type(loaded), _ordered(loaded)


def _engine_read(experiment, resolver, snapshot: DashboardSnapshot, request) -> Any:
    """The same request answered by the public readout, independently of the dashboard."""
    view, metric, complete, breakout = request
    added = metric in {model.name for model in snapshot.offered_metrics}
    names = [model.name for model in snapshot.metrics] if metric is None else [metric]
    segment = None if breakout is None else BreakoutIdentity(breakout[1], None)
    if view == "segments":
        scope = View(view="breakout", grain="total")
    elif view == "daily_values":
        scope = View(view="daily", grain="daily", values=True, segment=segment)
    else:
        scope = View(
            view="asof",
            grain="asof",
            values=view == "cumulative_values",
            completion_policy=complete,
            segment=segment,
        )
    try:
        if added:
            read = view_readout(experiment, resolver, scope, metrics=[], exploratory_metrics=names)
        else:
            read = view_readout(experiment, resolver, scope, metrics=names)
    except CodedError as refusal:
        return refusal
    rows = list(read)
    if view == "segments":
        rows = [
            row for row in rows if row.dimension == breakout[1] and row.method_role == "decision"
        ]
    if breakout is not None:
        rows = [row for row in rows if row.source == breakout[0]]
    return (BreakoutEstimates if view == "segments" else type(read)), _ordered(rows)


def _payload_text(experiment, snapshot) -> str:
    """The whole rendered payload, with the table renderer's random element ids numbered."""
    text = json.dumps(build_payload(experiment, snapshot=snapshot), sort_keys=True)
    ids = dict.fromkeys(re.findall(r'id=\\"([a-z]{10})\\"', text))
    for number, element_id in enumerate(ids):
        text = text.replace(element_id, f"element{number}")
    return text


def _page_payload(page: str) -> dict:
    match = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', page)
    assert match is not None
    return json.loads(match.group(1))


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
        - metric: session_seconds
          margin: 0.02
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

# The storefront clustered by store: an empty ratio guardrail leaves one unavailable cell, four
# clusters draw a small-cluster advisory, and a saved quantile metric is refused under clustering.
_CLUSTERED_FACT_SOURCES = (
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
_CLUSTERED_METRICS = (
    METRICS
    + """
  - type: ratio
    name: empty_guardrail
    entity: user_id
    preferred_direction: decrease
    numerator: {fact: purchase, aggregation: sum, window_days: 14}
    denominator: {fact: no_data, aggregation: count, window_days: 14}
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
_CLUSTERED_EXPERIMENTS = (
    EXPERIMENTS.replace("    unit: user_id\n", "    unit: user_id\n    cluster: store_id\n", 1)
    .replace(
        "      guardrails:\n        - metric: session_seconds\n          margin: 0.02\n",
        "      guardrails:\n"
        "        - metric: session_seconds\n"
        "          margin: 0.02\n"
        "        - metric: empty_guardrail\n"
        "          margin: 0.02\n",
        1,
    )
    .replace("    breakouts:\n      - property: country\n        source: event_log\n", "")
)


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


def _write_definitions(
    path: Path,
    *,
    facts: str = FACT_SOURCES,
    metrics: str = METRICS,
    experiments: str = EXPERIMENTS,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("fact_sources.yaml", facts),
        ("exposures.yaml", EXPOSURES),
        ("metrics.yaml", metrics),
        ("experiments.yaml", experiments),
    ):
        (path / name).write_text(body)
    return path


def _dashboard_connection():
    """Both experiments' events: 600/600 storefront units and 700/300 pricing units."""
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


@pytest.fixture(scope="session")
def dashboard_definitions(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(_write_definitions(tmp_path_factory.mktemp("dashboard_definitions")))


@backend_fixture(scope="session")
def dashboard_con():
    return _dashboard_connection()


def _bind(con, definitions: str | Path, experiment: str):
    from increment import from_definitions

    return from_definitions(definitions, con, experiment_name=experiment)


def _experiment(con, definitions: str | Path, experiment: str):
    """The declaration alone; snapshot consumers never read through a resolver."""
    declared, resolver = _bind(con, definitions, experiment)
    resolver.close()
    return declared


# Frozen preparations. Each source is rebuilt identically by the freeze and by the tests that
# compare a live preparation with the saved one.


SECOND = "page_views_per_user"


_ADDED_METRIC = f"""
  - type: mean
    name: {ADDED}
    description: Added purchase count per enrolled user.
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: count
    window_days: 14
"""
_SAVED = _METRICS + _ADDED_METRIC


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


@contextmanager
def _explore_source(tmp_path: Path):
    """Two sources of one breakout property and two offered metrics, shown or not.

    Enough units that the exploratory family selects a cell; no allocation scheme is declared,
    so the allocation check is not applicable.
    """
    with _workspace(
        tmp_path, units=1000, metrics=_TWO_SAVED, breakout_sources=("event_log", "profiles")
    ) as (con, experiment, resolver):
        yield con, experiment, resolver, _config()


_BH_Q = 0.08
_BH_PLAN = _PLAN.replace("correction: bonferroni\n", f"correction: bh\n        q: {_BH_Q}\n")


@contextmanager
def _bh_source(tmp_path: Path):
    """Segments corrected under a BH breakout policy, with two offered metrics.

    At this size each estimable metric selects one of its two segments, so the cutoff ``q / 2``
    is below the test level and a selected interval is reissued at it.
    """
    with _workspace(tmp_path, units=300, metrics=_TWO_SAVED, plan=_BH_PLAN) as (
        con,
        experiment,
        resolver,
    ):
        yield con, experiment, resolver, _config()


@contextmanager
def _storefront_source(tmp_path: Path):
    """The clustered storefront; two clusters per arm cannot match 99.99/0.01 configured shares."""
    con = _dashboard_connection()
    definitions = _write_definitions(
        tmp_path / "definitions",
        facts=_CLUSTERED_FACT_SOURCES,
        metrics=_CLUSTERED_METRICS,
        experiments=_CLUSTERED_EXPERIMENTS,
    )
    experiment, resolver = _bind(con, definitions, "storefront_refresh")
    try:
        yield (
            con,
            experiment,
            resolver,
            DashboardConfig(expected_allocation={"control": 0.9999, "treatment": 0.0001}),
        )
    finally:
        resolver.close()
        con.disconnect()


@contextmanager
def _pricing_source(tmp_path: Path):
    """Renamed 700/300 arms configured with relative 7:3 weights."""
    con = _dashboard_connection()
    experiment, resolver = _bind(con, _write_definitions(tmp_path / "definitions"), "pricing_copy")
    try:
        yield (
            con,
            experiment,
            resolver,
            DashboardConfig(
                expected_allocation={"baseline": 7, "candidate": 3},
                title="Pricing copy rewrite",
                provenance={"Fixture": "in-memory"},
            ),
        )
    finally:
        resolver.close()
        con.disconnect()


@contextmanager
def _triggered_source(tmp_path: Path):
    """A declared trigger: assigned and triggered populations, and one offered metric."""
    from tests.test_dashboard_population import _triggered_dashboard

    experiment, resolver = _triggered_dashboard(tmp_path, extra_metric=True)
    try:
        yield None, experiment, resolver, DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        resolver.close()


@contextmanager
def _triggered_pair_source(tmp_path: Path):
    """Two declared metrics, each lift view read for the assigned and triggered populations."""
    from tests.test_dashboard_population import _triggered_dashboard

    experiment, resolver = _triggered_dashboard(tmp_path, declare_extra=True)
    try:
        yield None, experiment, resolver, DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        resolver.close()


_FROZEN_SOURCES = {
    "bh": _bh_source,
    "explore": _explore_source,
    "storefront": _storefront_source,
    "pricing": _pricing_source,
    "triggered": _triggered_source,
    "triggered_pair": _triggered_pair_source,
}
# The saved metrics pricing_copy does not declare, in catalog order.
_PRICING_OFFERED = ("checkout_conversion", "revenue_per_user", "session_seconds")


def _prepared(experiment, resolver, config: DashboardConfig) -> DashboardSnapshot:
    # Clustered small-sample fits warn beside their advisories; the advisories are the subject.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare_dashboard(experiment, resolver, config=config)


def _freeze(names: list[str]) -> None:
    """Capture the named frozen preparations through ``prepare_dashboard`` and save them.

    Every Explore request of ``explore`` and every segment request of ``bh`` is also saved as
    the public readout's answer to that request alone.
    """
    with tempfile.TemporaryDirectory() as scratch:
        for name in names:
            directory = Path(scratch) / name
            directory.mkdir()
            with _FROZEN_SOURCES[name](directory) as (_, experiment, resolver, config):
                snapshot = _prepared(experiment, resolver, config)
                requests = [
                    request
                    for request in _explore_requests(snapshot)
                    if name == "explore" or (name == "bh" and request[0] == "segments")
                ]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    engine = {
                        request: _engine_read(experiment, resolver, snapshot, request)
                        for request in requests
                    }
            path = write_frozen(name, experiment=experiment, snapshot=snapshot, engine=engine)
            print(path)


# Capture contracts against a live source.


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_FROZEN_SOURCES))
def test_frozen_snapshot_matches_a_fresh_capture(tmp_path, name):
    saved = frozen(name)
    with _FROZEN_SOURCES[name](tmp_path) as (_, experiment, resolver, config):
        fresh = _prepared(experiment, resolver, config)
    assert experiment.to_json() == saved.experiment.to_json()
    assert preparation_differences(fresh, saved.snapshot) == []


@pytest.mark.integration
@pytest.mark.slow
def test_dashboard_payload_ignores_changes_after_preparation_until_reprepared(tmp_path):
    with _workspace(tmp_path, units=20, metrics=_SAVED) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        before = _payload_text(experiment, snapshot)
        requests = list(_explore_requests(snapshot))
        explored = [captured_form(_explored(experiment, snapshot, r)) for r in requests]
        assert snapshot.overview is not None
        overview = captured_form(snapshot.overview)

        _double_the_treatment_arm(con)

        assert _payload_text(experiment, snapshot) == before
        assert [captured_form(_explored(experiment, snapshot, r)) for r in requests] == explored
        assert captured_form(snapshot.overview) == overview
        fresh = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert fresh.readout_rows != snapshot.readout_rows
        assert fresh.overview is not None
        assert [row.model_dump(mode="json") for row in fresh.overview.exploratory] != [
            row.model_dump(mode="json") for row in snapshot.overview.exploratory
        ]
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


@pytest.mark.integration
@pytest.mark.slow
def test_breakout_property_in_its_own_source_is_part_of_the_pin(tmp_path):
    saved = frozen("explore")
    with _explore_source(tmp_path) as (con, experiment, resolver, config):
        con.raw_sql("UPDATE analytics.profiles SET country_code = 'XX'")
        fresh = _offered_shown(prepare_dashboard(experiment, resolver, config=config))
    kept_view = _offered_shown(saved.snapshot)
    request = ("cumulative_values", "checkout_conversion", False, PROFILES)
    kept = _explored(saved.experiment, kept_view, request)
    assert not isinstance(kept, CodedError)
    assert {row.dimension_value for row in kept[1]} == {"US", "CA"}
    moved = _explored(experiment, fresh, request)
    assert not isinstance(moved, CodedError)
    assert {row.dimension_value for row in moved[1]} == {"XX"}
    # Whatever does not read the profiles stream is the same read as before the change.
    for other in _explore_requests(saved.snapshot):
        if other[3] != PROFILES:
            assert (
                capture_differences(
                    _explored(experiment, fresh, other),
                    _explored(saved.experiment, kept_view, other),
                    without=_READ_IDENTITIES,
                )
                == []
            ), other


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


@pytest.mark.integration
@pytest.mark.slow
def test_a_refused_breakout_never_hides_another_source_of_the_same_property(tmp_path, monkeypatch):
    saved = frozen("explore")
    _refuse_the_profiles_breakout(monkeypatch)
    with _explore_source(tmp_path) as (_, experiment, resolver, config):
        degraded = _offered_shown(prepare_dashboard(experiment, resolver, config=config))
    kept_view = _offered_shown(saved.snapshot)
    answered = {"event_log": 0, "profiles": 0}
    for request in _explore_requests(saved.snapshot):
        healthy = _explored(saved.experiment, kept_view, request)
        shown = _explored(experiment, degraded, request)
        if request[3] == PROFILES:
            assert isinstance(shown, CodedError), request
            if not isinstance(healthy, CodedError):
                answered["profiles"] += 1
                assert shown.code == _REFUSED_SOURCE
        else:
            answered["event_log"] += request[3] == COUNTRY and not isinstance(healthy, CodedError)
            assert capture_differences(shown, healthy, without=_READ_IDENTITIES) == [], request
    assert answered["profiles"] and answered["event_log"]


@pytest.mark.integration
@pytest.mark.slow
def test_an_omitted_breakout_source_shows_the_source_it_resolves_to(tmp_path):
    with _workspace(tmp_path, breakout_sources=(None,)) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=CONFIG)
        for view in ("segments", "cumulative_lift"):
            request = (view, "checkout_conversion", False, (None, "country"))
            shown = _explored(experiment, snapshot, request)
            assert not isinstance(shown, CodedError) and shown[1]
            assert {row.source for row in shown[1]} == {"event_log"}
            resolved = (view, "checkout_conversion", False, COUNTRY)
            expected = _engine_read(experiment, resolver, snapshot, resolved)
            assert capture_differences(shown, expected, without=_READ_IDENTITIES) == []


QUANTILE = "p90_revenue"
# One declared metric, one saved metric and one saved quantile: the fewest reads that put a
# declared, an added and a refused metric in one breakout.
_PRIMARY_ADDED_AND_QUANTILE = (
    """
metrics:
  - type: conversion
    name: checkout_conversion
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14
"""
    + _ADDED_METRIC
    + f"""
  - type: quantile
    name: {QUANTILE}
    quantile: 0.9
    entity: user_id
    fact: purchase
    aggregation: sum
"""
)


@pytest.mark.integration
@pytest.mark.slow
def test_a_metric_the_breakout_refuses_keeps_every_other_metrics_segments(tmp_path):
    """A quantile has no segment breakout; its refusal stays its own and hides no other cell."""
    plan = "    plan:\n      primary: checkout_conversion\n"
    with _workspace(tmp_path, metrics=_PRIMARY_ADDED_AND_QUANTILE, plan=plan) as (
        _,
        experiment,
        resolver,
    ):
        # The saved metrics are offered but not shown; their segments and refusals still count.
        snapshot = prepare_dashboard(experiment, resolver, config=_config())
    overview = snapshot.overview
    assert overview is not None and overview.family_refusal is None
    shown = {cell.metric for cell in overview.segments[COUNTRY]}
    assert shown == {"checkout_conversion", ADDED}
    assert {metric for metric, _, _ in overview.refusals} == {QUANTILE}
    (refusal,) = (error for metric, place, error in overview.refusals if place == COUNTRY[1])
    assert isinstance(refusal, CapabilityError)
    assert refusal.code == "readout.metric.quantile_breakout"
    assert refusal.context["metric"] == QUANTILE
    assert any(
        QUANTILE in note and refusal.code in note for note in overview_notes(snapshot, COUNTRY)
    )


# A ratio over the purchase count: its denominator is a rare 0/1 count, so the estimator warns
# that small cells are too right-skewed, in every view that estimates them.
_SKEWED_RATIO = """
  - type: ratio
    name: revenue_per_purchase
    entity: user_id
    preferred_direction: increase
    numerator: {fact: purchase, aggregation: sum, window_days: 14}
    denominator: {fact: purchase, aggregation: count, window_days: 14}
"""


@pytest.mark.integration
@pytest.mark.slow
def test_preparation_shows_each_distinct_warning_once(tmp_path):
    """Cells several views estimate warn once per preparation, even when every warning shows."""
    plan = (
        "    plan:\n      primary: checkout_conversion\n      secondaries: [revenue_per_purchase]\n"
    )
    with (
        _workspace(tmp_path, metrics=_METRICS + _SKEWED_RATIO, plan=plan) as (
            _,
            experiment,
            resolver,
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        prepare_dashboard(experiment, resolver, config=CONFIG)
    shown = [
        (record.category, str(record.message), getattr(record.message, "context", {}).get("metric"))
        for record in caught
    ]
    assert any(issubclass(category, IncrementRuntimeWarning) for category, _, _ in shown)
    assert len(shown) == len(set(shown))


@pytest.mark.integration
@pytest.mark.slow
def test_registered_sequential_explore_is_captured_at_its_checkpoint():
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(sequential=True) as (connection, experiment, resolver):
        capture_sequential(experiment, resolver, finalized=True, as_of=dt.date(2025, 1, 16))
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        snapshot = prepare_dashboard(experiment, resolver, config=config)
        asof_lift = View(view="asof", grain="asof")
        asof_values = View(view="asof", grain="asof", values=True)
        live_lift = view_readout(experiment, resolver, asof_lift, metrics=["conversion"])
        live_values = view_readout(experiment, resolver, asof_values, metrics=["mean"])
        assert live_lift and live_values
        connection.raw_sql("UPDATE events SET value = value * 10 WHERE event = 'purchase'")
        changed = view_readout(experiment, resolver, asof_values, metrics=["mean"])

        def differences(rows, expected) -> list[str]:
            return capture_differences(_ordered(rows), _ordered(expected), without=_READ_IDENTITIES)

        assert differences(changed, live_values) != []
        lift = load_explore(
            experiment, snapshot=snapshot, metric="conversion", view="cumulative_lift"
        )
        assert differences(lift, live_lift) == []
        values = load_explore(
            experiment, snapshot=snapshot, metric="mean", view="cumulative_values"
        )
        assert differences(values, live_values) == []


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_preparation_preserves_registered_sequential_decisions(law, tmp_path):
    from increment import run
    from increment.estimation.results import LiftEstimate
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


@pytest.mark.integration
@pytest.mark.slow
def test_preparation_reads_every_section_from_one_pin_and_leaves_the_caller_live(
    dashboard_con, dashboard_definitions, monkeypatch
):
    import ibis

    from tests.analysis_factory import bound_source

    saved = frozen("pricing")
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
            snapshot = prepare_dashboard(experiment, resolver, config=saved.snapshot.config)

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
        # Every section, the headline and every Explore view included, is the unchanged read.
        assert preparation_differences(snapshot, saved.snapshot) == []
        fresh = prepare_dashboard(experiment, resolver, config=saved.snapshot.config)
        assert fresh.allocation is not None
        assert fresh.allocation.observed == {"baseline": 700, "candidate": 600}
        assert fresh.readout_rows[0]["lower"] != snapshot.readout_rows[0]["lower"]
        resolver.close()
    finally:
        con.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("change", ["control", "window", "policy", "breakouts"])
def test_explore_rejects_same_name_binding_changes_before_query(
    dashboard_con, dashboard_definitions, tmp_path, change
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
        load_explore(
            experiment,
            snapshot=frozen("pricing").snapshot,
            metric="purchase_rate",
            view="daily_values",
        )
    assert error.value.code == "dashboard.invalid_view"


@pytest.mark.integration
def test_explore_accepts_an_edit_to_a_saved_metric_the_experiment_does_not_declare(
    dashboard_con, dashboard_definitions, tmp_path
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
        experiment, snapshot=frozen("pricing").snapshot, metric="purchase_rate", view="daily_values"
    )
    assert {row.metric for row in loaded} == {"purchase_rate"}


@pytest.mark.integration
def test_preparation_refuses_allocation_without_the_control_arm(
    dashboard_con, dashboard_definitions
) -> None:
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with resolver, pytest.raises(InvalidRequestError) as caught:
        prepare_dashboard(experiment, resolver, config=config)
    assert caught.value.code == "dashboard.invalid_config"
    assert caught.value.context["control"] == "baseline"


@pytest.mark.integration
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


@pytest.mark.integration
def test_preparation_refuses_more_than_one_declared_primary(
    dashboard_con, dashboard_definitions
) -> None:
    experiment, resolver = _bind(dashboard_con, dashboard_definitions, "dual_primary")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with resolver, pytest.raises(CapabilityError) as caught:
        prepare_dashboard(experiment, resolver, config=config)
    assert caught.value.code == "dashboard.unsupported_experiment"
    assert caught.value.context["primaries"] == ("checkout_conversion", "revenue_per_user")


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


@pytest.mark.integration
def test_an_undeclared_exploratory_metric_is_refused_before_preparation(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        con.disconnect()
        for name in ("not_a_saved_metric", "checkout_conversion"):
            with pytest.raises(InvalidRequestError) as caught:
                prepare_dashboard(experiment, resolver, config=_config(name))
            assert caught.value.code == "dashboard.invalid_config"
            assert caught.value.context["available"] == (ADDED,)


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
    # An observational design refuses a relative margin when it compiles, before the dashboard.
    monkeypatch.setattr(
        capture,
        "_experiments",
        lambda **options: (
            declared(**options).replace("margin: 0.02", "margin_abs: 1.0") + _OBSERVATIONAL
        ),
    )
    with _workspace(tmp_path) as (con, experiment, resolver):
        con.disconnect()
        with pytest.raises(CapabilityError) as caught:
            prepare_dashboard(experiment, resolver, config=_config())
        assert caught.value.code == "dashboard.unsupported_experiment"
        assert caught.value.context["mechanism"] == "observational"


# Consumers of a frozen preparation: no source, resolver or connection exists here.


def test_every_explore_state_is_the_engine_read_of_its_own_key():
    """Each view, metric, maturity gate and breakout is the public readout of that request.

    Added metrics are each the read of that metric alone, never part of another selection's
    family, and a refused state keeps the engine's own type, code and context.
    """
    saved = frozen("explore")
    snapshot = _offered_shown(saved.snapshot)
    outcomes = {
        request: _explored(saved.experiment, snapshot, request)
        for request in _explore_requests(snapshot)
    }
    assert outcomes.keys() == saved.engine.keys()
    for request, expected in saved.engine.items():
        omitted = _may_differ_from_engine(request)
        assert capture_differences(outcomes[request], expected, without=omitted) == [], request
    # The matrix is not vacuous: states carry rows for every scope and some are refusals.
    for breakout in (None, *saved.snapshot.breakouts):
        assert any(
            not isinstance(value, CodedError) and value[1]
            for request, value in outcomes.items()
            if request[3] == breakout
        )
    assert any(isinstance(value, CodedError) for value in outcomes.values())


def test_loaded_views_are_independent_copies_of_the_capture():
    saved = frozen("explore")

    def load(metric: str, view: Any, complete: bool = False) -> Any:
        return load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric=metric,
            view=view,
            completed_windows_only=complete,
        )

    first = load("checkout_conversion", "cumulative_values")
    kept = captured_form(first)
    assert first
    assert isinstance(first, DailyMetricValues)
    with pytest.raises(CodedError) as caught:
        first.clear()
    assert caught.value.code == "readout.collection.mutation_unsupported"
    again = load("checkout_conversion", "cumulative_values")
    assert again is not first
    assert captured_form(first) == kept
    assert captured_form(again) == kept
    for view, complete in (("daily_values", False), ("cumulative_values", True)):
        with pytest.raises(CodedError) as one:
            load("ever_purchased_again", view, complete)
        with pytest.raises(CodedError) as other:
            load("ever_purchased_again", view, complete)
        assert one.value is not other.value
        assert captured_form(one.value) == captured_form(other.value)


def test_declared_breakout_temporal_views_are_the_engine_segment_trajectories():
    """Each declared breakout's series carries its own source; the whole experiment carries none."""
    saved = frozen("explore")
    for view, complete in _STATES[:-1]:
        whole = _explored(
            saved.experiment, saved.snapshot, (view, "checkout_conversion", complete, None)
        )
        assert not isinstance(whole, CodedError)
        assert {row.dimension for row in whole[1]} == {None}
        for breakout in saved.snapshot.breakouts:
            request = (view, "checkout_conversion", complete, breakout)
            segmented = _explored(saved.experiment, saved.snapshot, request)
            assert not isinstance(segmented, CodedError) and segmented[1], request
            assert {row.dimension_value for row in segmented[1]} == {"US", "CA"}
            assert {(row.dimension, row.source) for row in segmented[1]} == {
                (breakout[1], breakout[0])
            }
            assert capture_differences(segmented, whole) != []


def test_completed_windows_gate_changes_the_segment_absolute_series():
    saved = frozen("explore")

    def load(*, complete: bool):
        return load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="cumulative_values",
            breakout=COUNTRY,
            completed_windows_only=complete,
        )

    provisional, mature = load(complete=False), load(complete=True)
    assert min(row.ds for row in mature) > min(row.ds for row in provisional)
    assert not any(row.value is None for row in mature)


def test_even_split_experiment_binds_declared_metadata() -> None:
    snapshot = frozen("explore").snapshot
    assert snapshot.experiment_name == "storefront_refresh"
    assert snapshot.title == "storefront_refresh"
    assert (snapshot.control_group, snapshot.treatment_group) == ("control", "treatment")
    assert snapshot.primary_metric == "checkout_conversion"
    assert [metric.name for metric in snapshot.metrics] == list(_NAMES)
    assert [metric.name for metric in snapshot.offered_metrics] == [ADDED, SECOND]
    assert snapshot.exploratory_metrics == ()
    assert snapshot.breakouts == (COUNTRY, PROFILES)
    assert {row["inference"] for row in snapshot.readout_rows} == {"fixed"}


def test_renamed_unequal_experiment_binds_its_own_metadata() -> None:
    snapshot = frozen("pricing").snapshot
    assert snapshot.experiment_name == "pricing_copy"
    assert snapshot.title == "Pricing copy rewrite"
    assert (snapshot.control_group, snapshot.treatment_group) == ("baseline", "candidate")
    assert snapshot.primary_metric == "purchase_rate"
    assert snapshot.config.expected_allocation == {"baseline": 7, "candidate": 3}
    assert snapshot.breakouts == ()
    assert {row["inference"] for row in snapshot.readout_rows} == {"fixed"}


def test_allocation_check_uses_configured_shares() -> None:
    """Relative 7:3 weights are tested and shown as 70/30 shares, never as raw weights."""
    snapshot = frozen("pricing").snapshot
    allocation = snapshot.allocation
    assert allocation is not None
    assert allocation.expected == {"baseline": 0.7, "candidate": 0.3}
    assert allocation.observed == {"baseline": 700, "candidate": 300}
    assert allocation.is_srm is False
    text = render_results(snapshot).text
    assert "baseline 70%" in text
    assert "candidate 30%" in text
    assert "baseline 700%" not in text
    assert "candidate 300%" not in text


def test_rows_carry_the_tested_alternative() -> None:
    snapshot = frozen("explore").snapshot
    tails = {row["metric"]: row["alternative"] for row in snapshot.readout_rows}
    alternatives = {estimate.metric: estimate.alternative for estimate in snapshot.estimates}
    assert tails == alternatives
    # A guardrail's non-inferiority test is one-sided.
    assert tails["session_seconds"] == "less"
    assert tails["checkout_conversion"] == "two-sided"


def test_rows_correspond_one_to_one_with_estimates() -> None:
    snapshot = frozen("explore").snapshot
    assert len(snapshot.readout_rows) == len(snapshot.estimates)
    for row, estimate in zip(snapshot.readout_rows, snapshot.estimates, strict=True):
        assert row["metric"] == estimate.metric
        assert row["group_id"] == estimate.group_id
        assert row["lift"] == pytest.approx(estimate.require_lift().value)


def test_snapshot_owns_nested_readout_data() -> None:
    rows = [dict(row) for row in frozen("explore").snapshot.readout_rows]
    components: list[dict[str, Any]] = [{"bounds": {"lower": 0.25}, "labels": ["registered"]}]
    rows[0]["sequential_components"] = components
    snapshot = dataclasses.replace(frozen("explore").snapshot, readout_rows=tuple(rows))
    before = readout_csv(snapshot)
    components[0]["bounds"]["lower"] = 99.0
    components[0]["labels"].append("changed")
    assert readout_csv(snapshot) == before
    captured = snapshot.readout_rows[0]["sequential_components"]
    assert isinstance(captured, tuple)
    copied = captured[0]
    assert isinstance(copied, MappingProxyType)
    with pytest.raises(TypeError):
        copied["bounds"]["lower"] = 0.0  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        copied["labels"].append("changed")  # ty: ignore[unresolved-attribute]


def test_snapshot_allocation_maps_are_owned_and_immutable() -> None:
    saved = frozen("pricing").snapshot
    assert saved.allocation is not None
    original = saved.allocation
    payload = original.model_dump()
    allocation = type(original).model_validate(payload)
    snapshot = dataclasses.replace(saved, allocation=allocation)
    assert snapshot.allocation is not None
    before = snapshot.allocation.model_dump_json()
    for name in ("observed", "expected", "unit_counts"):
        payload[name]["baseline"] = 999
        with pytest.raises(TypeError):
            getattr(allocation, name)["baseline"] = 999
    assert snapshot.allocation.model_dump_json() == before
    restored = type(original).model_validate_json(allocation.model_dump_json())
    assert restored.observed == original.observed
    assert restored.expected == original.expected
    assert restored.unit_counts == original.unit_counts


def test_allocation_history_rows_are_immutable() -> None:
    saved = frozen("pricing").snapshot
    assert saved.allocation_history_refusal is None
    raw = [dict(saved.allocation_history[0])]
    copied = dataclasses.replace(saved, allocation_history=tuple(raw))
    original_count = copied.allocation_history[0]["n_daily"]
    assert isinstance(original_count, int)
    raw[0]["n_daily"] = original_count + 100
    assert copied.allocation_history[0]["n_daily"] == original_count
    row = saved.allocation_history[0]
    with pytest.raises(TypeError):
        row["n_daily"] = 0  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        saved.allocation_history.append(row)  # ty: ignore[unresolved-attribute]


def test_allocation_history_reconciles_with_the_renamed_unequal_experiment() -> None:
    """The 70/30 renamed arms' real cumulative totals, not a derived guess."""
    snapshot = frozen("pricing").snapshot
    assert snapshot.allocation_history_refusal is None
    assert snapshot.allocation is not None
    rows = snapshot.allocation_history
    assert {row["experiment_id"] for row in rows} == {"pricing_copy"}
    assert {row["group_id"] for row in rows} == {"baseline", "candidate"}
    assert list(rows) == sorted(rows, key=lambda row: (row["ds"], row["group_id"]))
    cumulative: dict[str, list[int]] = {"baseline": [], "candidate": []}
    for row in rows:
        group, count = row["group_id"], row["n_cumulative"]
        assert isinstance(group, str) and isinstance(count, int)
        cumulative[group].append(count)
    final_totals = {group: max(counts) for group, counts in cumulative.items()}
    assert final_totals == snapshot.allocation.observed


def test_prepared_config_rejects_mutation() -> None:
    snapshot = frozen("pricing").snapshot
    with pytest.raises(TypeError):
        snapshot.config.expected_allocation["baseline"] = 0.1  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        snapshot.readout_rows[0]["lift"] = 0.0  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.primary_metric = "other"  # ty: ignore[invalid-assignment]


def test_load_explore_refuses_an_undeclared_metric() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment, snapshot=saved.snapshot, metric="not_a_metric", view="daily_values"
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["metric"] == "not_a_metric"


def test_load_explore_refuses_an_unknown_view() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="not_a_view",  # ty: ignore[invalid-argument-type]
        )
    assert caught.value.code == "dashboard.invalid_view"


def test_load_explore_refuses_daily_values_with_a_cumulative_maturity_flag() -> None:
    """Daily values are disjoint slices; a cumulative maturity flag cannot be honoured."""
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="daily_values",
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "daily_values"


def test_segment_view_refuses_a_cumulative_maturity_flag() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="segments",
            breakout=COUNTRY,
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "segments"


def test_load_explore_refuses_a_snapshot_and_source_describing_different_experiments() -> None:
    """A mismatched pair must not silently render the other binding's stale data."""
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            frozen("pricing").experiment,
            snapshot=frozen("explore").snapshot,
            metric="checkout_conversion",
            view="daily_values",
        )
    assert caught.value.code == "dashboard.invalid_view"
    context = caught.value.context
    assert context["snapshot_experiment"] == "storefront_refresh"
    assert context["analysis_experiment"] == "pricing_copy"


def test_load_explore_segments_need_an_explicit_choice_when_breakouts_are_declared() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="segments",
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["declared"] == (COUNTRY, PROFILES)


def test_load_explore_segments_refuse_an_undeclared_breakout() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="segments",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


def test_temporal_breakouts_refuse_an_undeclared_dimension() -> None:
    saved = frozen("explore")
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            saved.experiment,
            snapshot=saved.snapshot,
            metric="checkout_conversion",
            view="cumulative_lift",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


def test_load_explore_segments_with_no_declared_breakouts_return_empty() -> None:
    """A truthful empty state, not a query the experiment never declared."""
    saved = frozen("pricing")
    result = load_explore(
        saved.experiment, snapshot=saved.snapshot, metric="purchase_rate", view="segments"
    )
    assert result == BreakoutEstimates()


def test_load_explore_segments_filter_by_dimension_source_and_method_without_leaking_rows() -> None:
    """A row from another dimension, source, or method role must never leak in."""
    from increment._source_operations import DashboardExploreCapture

    saved = frozen("explore")
    wanted = load_explore(
        saved.experiment,
        snapshot=saved.snapshot,
        metric="checkout_conversion",
        view="segments",
        breakout=COUNTRY,
    )
    assert len(wanted) == 2
    decoys = [
        wanted[0].model_copy(update={"dimension": "plan", "dimension_value": "free"}),
        wanted[0].model_copy(update={"method_role": "component"}),
        wanted[0].model_copy(update={"source": "profiles"}),
    ]
    key = ("assigned", "segments", "checkout_conversion", False, COUNTRY)
    captured = dataclasses.replace(
        saved.snapshot,
        explore={
            **saved.snapshot.explore,
            key: DashboardExploreCapture.answered(
                key, BreakoutEstimates([*wanted, *decoys]), collection=BreakoutEstimates
            ),
        },
    )
    filtered = load_explore(
        saved.experiment,
        snapshot=captured,
        metric="checkout_conversion",
        view="segments",
        breakout=COUNTRY,
    )
    assert isinstance(filtered, BreakoutEstimates)
    assert len(filtered) == 2
    assert {e.dimension for e in filtered} == {"country"}
    assert {e.source for e in filtered} == {"event_log"}
    assert {e.method_role for e in filtered} == {"decision"}


@pytest.mark.integration
def test_load_explore_segments_with_no_metric_reads_every_declared_metric(tmp_path) -> None:
    """The disconnected all-metric view retains the guardrail's segment hypothesis."""
    with _explore_source(tmp_path) as (_, experiment, resolver, config):
        snapshot = _prepared(experiment, resolver, config)
    assert tuple(metric.name for metric in snapshot.metrics) == _NAMES
    rows = load_explore(
        experiment,
        snapshot=snapshot,
        metric=None,
        view="segments",
        breakout=COUNTRY,
    )
    assert isinstance(rows, BreakoutEstimates)
    assert {row.metric for row in rows} == set(_NAMES)
    guardrails = [row for row in rows if row.metric == "session_seconds"]
    assert guardrails
    assert all(
        (row.null_lift, row.null_abs, row.alternative, row.role)
        == (0.02, None, "less", "exploratory")
        for row in guardrails
    )
    assert all(row.p_value() is not None for row in guardrails)


@pytest.mark.integration
def test_exploring_never_changes_the_confirmatory_snapshot() -> None:
    saved = frozen("pricing")
    snapshot = _offered_shown(saved.snapshot)
    before = captured_form(
        (snapshot.estimates, snapshot.readout_rows, readout_csv(snapshot), snapshot.explore)
    )
    build_payload(saved.experiment, snapshot=snapshot)
    after = captured_form(
        (snapshot.estimates, snapshot.readout_rows, readout_csv(snapshot), snapshot.explore)
    )
    assert after == before


@pytest.mark.integration
def test_an_engine_refusal_is_visible_for_its_own_state_only() -> None:
    saved = frozen("explore")
    snapshot = show_exploratory_metrics(saved.snapshot, (ADDED,))
    payload = build_payload(saved.experiment, snapshot=snapshot)
    country, overall = payload["explore"]["event_log:country"], payload["explore"]["overall"]
    refused = country[ADDED]["daily_values"]
    assert refused["pointCount"] == 0
    assert "facade.analysis.undeclared_metric_for_dimension" in refused["html"]
    assert overall[ADDED]["daily_values"]["pointCount"] > 0
    assert country[ADDED]["cumulative_values"]["pointCount"] > 0
    retention = overall["ever_purchased_again"]
    assert retention["daily_values"]["pointCount"] == 0
    assert "breakout.retention.unbounded" in retention["daily_values"]["html"]
    assert retention["cumulative_values"]["pointCount"] > 0
    # A BH breakout policy refuses segmented cumulative lift; the overall lift and the segmented
    # absolute series stay available.
    bh = frozen("bh")
    explore = build_payload(bh.experiment, snapshot=bh.snapshot)["explore"]
    segmented = explore["country"]["checkout_conversion"]
    assert segmented["cumulative_lift"]["pointCount"] == 0
    assert "facade.analysis.bh_segmented_asof" in segmented["cumulative_lift"]["html"]
    assert explore["overall"]["checkout_conversion"]["cumulative_lift"]["pointCount"] > 0
    assert segmented["cumulative_values"]["pointCount"] > 0


def test_health_renders_not_applicable_allocation_as_not_checked() -> None:
    snapshot = frozen("explore").snapshot
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


def test_dashboard_snapshot_and_csv_retain_scoped_readout_metadata():
    snapshot = frozen("storefront").snapshot
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
    hidden = _json_cell(primary, "hidden_cells")
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
    readout_html = render_results(dashboard_snapshot).text
    assert "partial" in readout_html
    assert "session_seconds" in readout_html
    assert "SRM not checked" in readout_html and "allocation" in readout_html
    assert "Dashboard allocation check" in readout_html
    assert "configured expected control 99.99% / treatment 0.01%" in readout_html
    assert "mismatch detected" in readout_html
    assert "integrity.allocation_missing_or_nonconstant" in readout_html
    assert readout_html.index("mismatch detected") < readout_html.index("Analysis assignment check")


def _assert_added_exploratory_readout(snapshot):
    from html import escape

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


@pytest.mark.integration
def test_dashboard_round_trips_real_unavailable_cell_and_advisories():
    saved = frozen("storefront")
    snapshot = show_exploratory_metrics(saved.snapshot, ("purchase_rate",))
    assert snapshot.allocation is not None and snapshot.allocation.is_srm
    failed = next(row for row in snapshot.readout_rows if row["metric"] == "empty_guardrail")
    unavailable = _json_cell(failed, "unavailable_cells")
    reason = next(item for item in unavailable if item["cell"]["metric"] == "empty_guardrail")
    assert reason["code"] == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
    advisory = next(
        item
        for row in snapshot.readout_rows
        for item in _json_cell(row, "advisories")
        if item["code"] == "estimation.engine.small_total_clusters"
    )
    assert "below 40" in advisory["message"]

    readout_html = render_results(snapshot).text
    payload = build_payload(saved.experiment, snapshot=snapshot)
    report_text = payload["report"]["qualifications"]
    for surface in (readout_html, report_text):
        assert reason["code"] in surface
        assert advisory["code"] in surface
        assert "below 40" in surface
    assert advisory["message"] in report_text
    assert "integrity.allocation_missing_or_nonconstant" in report_text
    shell = document(payload)
    assert "hero-meta').textContent = str(data.meta)" in shell
    # The hero leaves the qualifications to Readout and Report.
    assert "hero-qualifications" not in shell
    assert payload["meta"]
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


@pytest.mark.integration
def test_a_refused_added_metric_keeps_every_other_metrics_overview_rows():
    """A metric the clustered read refuses keeps its own refusal; the others keep their rows."""
    saved = frozen("storefront")
    added = ("added_revenue", "added_duration", "p90_revenue")
    snapshot = show_exploratory_metrics(saved.snapshot, added)
    capture = snapshot.explore[("assigned", "overview_whole", None, False, None)]
    collection = cast(LiftEstimates, capture.load())
    assert {row.metric for row in collection} >= set(added[:2])
    assert collection.metadata is not None
    assert {record.cell.metric for record in collection.metadata.cells} >= set(added[:2])

    overview = snapshot.overview
    assert overview is not None and overview.family_refusal is None
    assert {row.metric for row in overview.exploratory} >= set(added[:2])
    assert "p90_revenue" not in {row.metric for row in overview.exploratory}
    assert {(metric, place) for metric, place, _ in overview.refusals} == {
        ("p90_revenue", "whole experiment")
    }
    refusal = next(item for item in overview.refusals if item[0] == "p90_revenue")
    assert refusal[2].code == "arm.metric.quantile_cluster"
    assert any("p90_revenue" in note for note in overview_notes(snapshot, None))

    details = render_metric_details(snapshot, metric="added_revenue").text
    assert "added_revenue" in details
    assert render_dashboard(saved.experiment, snapshot=snapshot) is not None
    payload = build_payload(saved.experiment, snapshot=snapshot)
    notes = payload["overview"]["overall"]["notes"]
    assert any("p90_revenue" in note and refusal[2].code in note for note in notes)


def test_one_family_covers_added_metrics_and_every_segment_cell():
    """Added whole-experiment cells and every segment of each source are one BH family."""
    snapshot = frozen("explore").snapshot
    overview = snapshot.overview
    assert overview is not None and overview.family_refusal is None
    by_source = {breakout: overview.segments[breakout] for breakout in (COUNTRY, PROFILES)}
    assert all(by_source.values())
    cells = [*overview.exploratory, *(cell for cells in by_source.values() for cell in cells)]
    tested = [cell for cell in cells if cell.family_size is not None]
    assert {cell.metric for cell in overview.exploratory} == {ADDED, SECOND}
    for segments in by_source.values():
        assert {cell.metric for cell in segments} >= {ADDED, "checkout_conversion"}
    assert len(tested) == overview.family_size > len(overview.exploratory)
    assert {cell.family_size for cell in tested} == {overview.family_size}
    assert overview.family_size == 2 + 2 * 2 * 6
    assert overview.exclusions == ()
    failed = next(cell for cell in overview.exploratory if cell.metric == SECOND)
    assert failed.failure_code == "estimation.engine.lift_guard"
    assert failed.failure_context is not None
    assert failed.failure_context["reason"] == "zero_variance"
    assert failed.sampling_available is False and failed.discovery is False
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
    from increment.estimation.results import LiftEstimate

    if isinstance(selected, BreakoutEstimate):
        assert isinstance(restored, BreakoutEstimate)
        scope = (selected.source, selected.dimension)
        segments = dict(overview.segments)
        segments[scope] = tuple(restored if row is selected else row for row in segments[scope])
        restored_overview = dataclasses.replace(overview, segments=segments)
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
    for breakout in by_source:
        for row in overview_rows(snapshot, breakout):
            if row["metric"] in (ADDED, SECOND) or row.get("dimension") is not None:
                assert row["stat_sig"] is bool(row.get("discovery"))


def test_choosing_what_to_show_never_changes_the_family():
    """Offered metrics are read and counted whether shown or not."""
    hidden = frozen("explore").snapshot
    shown = show_exploratory_metrics(hidden, (ADDED,))
    assert [model.name for model in hidden.offered_metrics] == [ADDED, SECOND]
    assert hidden.exploratory_metrics == () and hidden.overview is not None
    assert [model.name for model in shown.exploratory_metrics] == [ADDED]
    assert shown.readout_rows == hidden.readout_rows
    assert shown.overview is not None
    assert {cell.metric for cell in hidden.overview.exploratory} == {ADDED, SECOND}
    assert captured_form(shown.overview) == captured_form(hidden.overview)


# The plan's default test level.
_ALPHA = 0.05


def _bh_cutoff(p_values: list[float | None], q: float) -> float | None:
    """Benjamini-Hochberg step-up: the largest k with p_(k) <= k q / m sets the cutoff k q / m.

    A hypothesis without a p-value counts in m and is never selected.
    """
    m = len(p_values)
    ranked = sorted(p for p in p_values if p is not None)
    k = max((i for i, p in enumerate(ranked, start=1) if p <= i * q / m), default=0)
    return k * q / m if k else None


def test_each_metrics_segments_are_its_own_bh_step_up():
    """Under a BH breakout policy each metric's segments are one family of its own cells.

    Each metric's segments are the public readout of that metric alone on the same data, so
    no metric shown beside it enters its family. The cells' p-values alone set the cutoff and
    the selection, and a selected interval is reissued at the cutoff when it is below the test
    level.
    """
    saved = frozen("bh")
    snapshot = _offered_shown(saved.snapshot)
    close = partial(pytest.approx, rel=TOLERANCE, abs=TOLERANCE)
    selected = rejected = reissued = 0
    for model in (*snapshot.metrics, *snapshot.offered_metrics):
        rows = load_explore(
            saved.experiment,
            snapshot=snapshot,
            metric=model.name,
            view="segments",
            breakout=COUNTRY,
        )
        assert isinstance(rows, BreakoutEstimates) and rows, model.name
        request = ("segments", model.name, False, COUNTRY)
        alone = saved.engine[request]
        assert (
            capture_differences(
                (type(rows), _ordered(rows)), alone, without=_may_differ_from_engine(request)
            )
            == []
        ), model.name
        p_values = [row.p_value() for row in rows]
        cutoff = _bh_cutoff(p_values, _BH_Q)
        for row, p in zip(rows, p_values, strict=True):
            chosen = cutoff is not None and p is not None and p <= cutoff
            assert row.family_procedure == "bh"
            assert row.family_q == close(_BH_Q)
            assert row.family_threshold == (None if cutoff is None else close(cutoff))
            assert row.discovery is chosen, (row.metric, row.dimension_value)
            if p is None:
                continue
            assert row.lift is not None
            level = min(cutoff, _ALPHA) if chosen and cutoff is not None else _ALPHA
            display_alpha = (
                level
                if row.alternative == "two-sided" or row.lift.open_side is not None
                else 2 * level
            )
            assert row.lift.alpha == close(display_alpha), (row.metric, row.dimension_value)
            selected += chosen
            rejected += not chosen
            reissued += level < _ALPHA
    assert selected and rejected and reissued


@pytest.mark.integration
def test_added_metrics_keep_their_own_absolute_value_series():
    saved = frozen("pricing")
    snapshot = _offered_shown(saved.snapshot)
    explore = build_payload(saved.experiment, snapshot=snapshot)["explore"]["overall"]
    for name in _PRICING_OFFERED:
        for view in ("cumulative_values", "daily_values", "cumulative_lift"):
            assert explore[name][view]["pointCount"] > 0, (name, view)


@pytest.mark.integration
def test_the_dashboard_widget_shows_added_metrics_without_reading_again():
    saved = frozen("pricing")
    snapshot = saved.snapshot
    added = _PRICING_OFFERED[0]
    widget = DashboardWidget(saved.experiment, snapshot=snapshot)
    offered = _page_payload(widget.document)["exploratory"]
    assert [item["key"] for item in offered["available"]] == list(_PRICING_OFFERED)
    assert offered["selected"] == []

    # Selection re-renders this snapshot: nothing here can read a source.
    widget.exploratory_metrics = [added]
    assert widget.status == ""
    payload = _page_payload(widget.document)
    assert payload["exploratory"]["selected"] == [added]
    assert added in {metric["key"] for metric in payload["metrics"]}
    assert widget._snapshot.readout_rows == snapshot.readout_rows
    assert widget._snapshot.overview == snapshot.overview
    assert widget._snapshot.computed_at == snapshot.computed_at

    page = widget.document
    widget.exploratory_metrics = ["not_a_saved_metric"]
    assert "dashboard.invalid_config" in widget.status
    assert widget.exploratory_metrics == [added]
    assert widget.document == page

    widget.exploratory_metrics = []
    assert widget.status == ""
    assert added not in {metric["key"] for metric in _page_payload(widget.document)["metrics"]}


if __name__ == "__main__":
    if sys.argv[1:2] != ["--freeze"] or not set(sys.argv[2:]) <= _FROZEN_SOURCES.keys():
        print(
            f"usage: python -m tests.test_dashboard_capture --freeze [{' '.join(_FROZEN_SOURCES)}]",
            file=sys.stderr,
        )
        sys.exit(2)
    _freeze(sys.argv[2:] or list(_FROZEN_SOURCES))
