"""One event log and its derived dataframes for every parity-matrix cell.

The event log is the single source of truth. The warehouse routes read it as the
``events`` table; the dataframe routes read frames derived from it here, bucketing
days at the cell's ``day_boundary`` themselves (a frame carries no boundary lever;
see ``tests/test_frame_window_parity.py``). The experiment window is one pair of
instants for both boundaries, so a boundary only moves day labels, window days and
retention bands, never which events exist.

Edge units (``i % 6 == 4``) are exposed at 03:00Z, which is 22:00 on the previous
day under ``UTC-05:00``: their exposure day, window days and retention band differ by
boundary, and a route that buckets in UTC instead fails on exactly them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import pyarrow as pa

from . import dataset as ds

DayBoundary = Literal["utc", "fixed_offset"]
Missing = Literal["error", "zero", "drop", "impute"]

N_PER_ARM = 60
OFFSETS: dict[str, dt.timedelta] = {"utc": dt.timedelta(0), "fixed_offset": dt.timedelta(hours=-5)}
BOUNDARY_SPELLING = {"utc": "UTC", "fixed_offset": "UTC-05:00"}
WINDOW_START = "2025-01-10T00:00:00Z"
WINDOW_END = "2025-01-20T00:00:00Z"
WINDOW_DAYS = 2
RETENTION_BAND = (3, 5)
QUANTILE = 0.5
WINSOR_FIXED_UPPER = 12.0
WINSOR_PERCENTILE_UPPER = 0.9

_EXPOSED_AT = dt.datetime(2025, 1, 10, 9)
_EDGE_EXPOSED_AT = dt.datetime(2025, 1, 10, 3)
_FRESH_AT = dt.datetime(2025, 1, 19, 9)
_ARMS = (("control", "c"), ("treatment", "t"))


def is_edge(i: int) -> bool:
    return i % 6 == 4


def local_day(at: dt.datetime, boundary: str) -> dt.date:
    return (at + OFFSETS[boundary]).date()


def observation_end_day(boundary: str) -> dt.date:
    """The declared window end as a day at *boundary* (`Experiment.end_day`)."""
    return (dt.datetime(2025, 1, 20) + OFFSETS[boundary]).date()


def _event(unit: str, at: dt.datetime, kind: str, **fields: Any) -> dict[str, Any]:
    row = {
        **ds._row(unit, at, kind, **{k: v for k, v in fields.items() if k in _ROW_FIELDS}),
        "cluster_id": fields.get("cluster_id"),
        "tenure": fields.get("tenure"),
    }
    return row


_ROW_FIELDS = frozenset({"group_id", "experiment_id", "store_id", "revenue", "sess", "latency"})


@dataclass(frozen=True)
class Unit:
    id: str
    arm: str
    index: int
    store: str
    cluster: str
    tenure: float
    exposed_at: dt.datetime
    pre_revenue: float
    pre_converted: int
    purchases: tuple[tuple[dt.datetime, float], ...]
    sessions: tuple[dt.datetime, ...]
    latency: tuple[dt.datetime, float] | None


def units(*, positive: bool = False) -> list[Unit]:
    """The enrolled units. *positive* gives every unit an in-window purchase (a
    percentile winsorization pilot needs strictly positive outcomes)."""
    out: list[Unit] = []
    for arm, prefix in _ARMS:
        treated = arm == "treatment"
        for i in range(N_PER_ARM):
            exposed = _EDGE_EXPOSED_AT if is_edge(i) else _EXPOSED_AT
            pre = 2.0 + (i % 5) if i % 5 != 0 else 0.0
            purchases: list[tuple[dt.datetime, float]] = []
            if positive or i % 3 != 0:
                purchases.append(
                    (exposed + dt.timedelta(hours=6), 3.0 + treated + (i % 5) + 0.5 * pre)
                )
                if i % 2 == 0:
                    purchases.append((exposed + dt.timedelta(days=1, hours=6), 2.0 + (i % 3)))
                # The retention band [3, 5) is probed three ways: clear (a day-3 purchase),
                # and one purchase whose band day depends on the boundary -- 18:00 on day 4
                # lands on UTC day 5 but local day 4 for a 09:00Z exposure, and 06:00 on day 4
                # lands on local day 5 but UTC day 4 for an edge unit exposed at 03:00Z.
                if i % 4 == 0:
                    purchases.append((exposed + dt.timedelta(days=3, hours=6), 1.0 + (i % 2)))
                elif i % 4 == 1:
                    purchases.append((exposed + dt.timedelta(days=4, hours=18), 1.0))
                elif i % 4 == 2:
                    purchases.append((exposed + dt.timedelta(days=4, hours=6), 1.0))
            if i % 10 == 9:
                purchases.append((exposed + dt.timedelta(hours=7), 40.0))
            # A unit with no session event, or no latency event, has that input absent: a
            # warehouse reads it as zero, a frame under `zero` or `drop` as a NULL cell.
            sessions: list[dt.datetime] = []
            if i % 4 != 3:
                sessions.append(exposed + dt.timedelta(hours=7))
                if i % 2 == 0:
                    sessions.append(exposed + dt.timedelta(days=1, hours=7))
            latency = (
                None
                if i % 5 == 3
                else (
                    exposed + dt.timedelta(hours=8),
                    100.0 + float((i * 7) % 23) + 10.0 * treated,
                )
            )
            out.append(
                Unit(
                    id=f"{prefix}{i}",
                    arm=arm,
                    index=i,
                    store=ds.store_for(i),
                    cluster=f"{arm}-k{i - (i % 3 == 2)}",
                    tenure=float(i % 7) + 0.25 * i,
                    exposed_at=exposed,
                    pre_revenue=pre,
                    pre_converted=int(pre > 0),
                    purchases=tuple(purchases),
                    sessions=tuple(sessions),
                    latency=latency,
                )
            )
    return out


def event_rows(*, positive: bool = False) -> list[dict[str, Any]]:
    """The warehouse ``events`` table: exposures, pre-period and outcome events,
    and one unenrolled unit whose late events only establish data freshness."""
    rows: list[dict[str, Any]] = []
    for u in units(positive=positive):
        common = {"store_id": u.store, "cluster_id": u.cluster}
        rows.append(
            _event(
                u.id,
                u.exposed_at,
                "exposure",
                group_id=u.arm,
                experiment_id="exp",
                **common,
            )
        )
        rows.append(
            _event(
                u.id,
                u.exposed_at - dt.timedelta(days=2),
                "profile",
                tenure=u.tenure,
                **common,
            )
        )
        if u.pre_revenue > 0:
            rows.append(
                _event(
                    u.id,
                    u.exposed_at - dt.timedelta(days=4),
                    "purchase",
                    revenue=u.pre_revenue,
                    **common,
                )
            )
        rows.extend(_event(u.id, at, "purchase", revenue=r, **common) for at, r in u.purchases)
        rows.extend(_event(u.id, at, "session_end", sess=1, **common) for at in u.sessions)
        if u.latency is not None:
            rows.append(_event(u.id, u.latency[0], "latency", latency=u.latency[1], **common))
    for kind, fields in (
        ("purchase", {"revenue": 1.0}),
        ("session_end", {"sess": 1}),
        ("latency", {"latency": 100.0}),
    ):
        rows.append(_event("fresh", _FRESH_AT, kind, **fields))
    return rows


def duckdb_connection(rows: list[dict[str, Any]]) -> Any:
    import ibis

    con = ibis.duckdb.connect()
    table = pa.Table.from_pylist(rows)
    for i, field in enumerate(table.schema):
        if pa.types.is_null(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.float64()))
    con.create_table("events", obj=table)
    return con


def _after_exposure(u: Unit, at: dt.datetime) -> bool:
    return at >= u.exposed_at


# NULLs mark absent events, so warehouse zeros and frame zero-filling agree. Distinct input
# masks exercise numerator-only, denominator-only, both-missing and complete ratio units.
# Explicit zeros among non-purchasers keep conversion/drop nondegenerate; retention panels
# similarly mix NULL and explicit-zero non-purchase days.
def revenue_missing(u: Unit) -> bool:
    return not u.purchases and u.index % 6 == 3


def conversion_missing(u: Unit) -> bool:
    return not u.purchases and u.index % 6 == 0


def sessions_missing(u: Unit) -> bool:
    return not u.sessions


def latency_missing(u: Unit) -> bool:
    return u.latency is None


def retention_missing(u: Unit) -> bool:
    return u.index % 2 == 0


def summary_rows(boundary: str, *, nulls: bool, positive: bool = False) -> list[dict[str, Any]]:
    """One row per unit: unwindowed post-exposure totals over the whole window.

    *nulls* leaves each input a unit has no event for as NULL, for the units in that input's
    missing set (see `revenue_missing`), instead of the caller's zero fill.
    """
    rows = []
    for u in units(positive=positive):
        total = sum(r for at, r in u.purchases if _after_exposure(u, at))
        bought = any(_after_exposure(u, at) for at, _ in u.purchases)
        rows.append(
            {
                "user_id": u.id,
                "variant": u.arm,
                "store": u.store,
                "cluster_id": u.cluster,
                "tenure": u.tenure,
                "exposed_on": local_day(u.exposed_at, boundary),
                "revenue": None if nulls and revenue_missing(u) else total,
                "converted": None if nulls and conversion_missing(u) else int(bought),
                "sessions": None if nulls and sessions_missing(u) else len(u.sessions),
                "latency": None
                if nulls and latency_missing(u)
                else (0.0 if u.latency is None else u.latency[1]),
                "pre_revenue": u.pre_revenue,
                "pre_converted": u.pre_converted,
            }
        )
    return rows


def panel_rows(
    boundary: str, *, nulls: bool, positive: bool = False, daily_conversion: bool = False
) -> list[dict[str, Any]]:
    """One row per unit per local day, dense from the first exposure day to the window end day.

    Under *nulls* a day with no event of an input is NULL for revenue (every unit), sessions
    and latency (every unit); a unit in `conversion_missing` carries NULL conversion on every
    day; and `returned` is NULL on a day without a purchase for the units in
    `retention_missing` and an explicit 0 for the rest. Otherwise the caller's zero fill.
    """
    first = local_day(_EDGE_EXPOSED_AT, boundary)
    last = observation_end_day(boundary)
    days = [first + dt.timedelta(days=n) for n in range((last - first).days + 1)]
    rows = []
    for u in units(positive=positive):
        by_day: dict[dt.date, dict[str, Any]] = {}
        for at, r in u.purchases:
            day = local_day(at, boundary)
            slot = by_day.setdefault(day, {"revenue": 0.0, "returned": 0.0, "sessions": 0})
            slot["revenue"] += r
            slot["returned"] += 1.0
        for at in u.sessions:
            slot = by_day.setdefault(
                local_day(at, boundary), {"revenue": 0.0, "returned": 0.0, "sessions": 0}
            )
            slot["sessions"] += 1
        latency_day = None if u.latency is None else local_day(u.latency[0], boundary)
        exposed_on = local_day(u.exposed_at, boundary)
        post = [at for at, _ in u.purchases if _after_exposure(u, at)]
        first_purchase_day = local_day(min(post), boundary) if post else None
        for day in days:
            slot = by_day.get(day, {})
            bought = slot.get("returned", 0.0) > 0
            rows.append(
                {
                    "user_id": u.id,
                    "variant": u.arm,
                    "store": u.store,
                    "cluster_id": u.cluster,
                    "tenure": u.tenure,
                    "date": day,
                    "exposed_on": exposed_on,
                    "revenue": slot.get("revenue", 0.0) if bought or not nulls else None,
                    # A total reads a unit's conversion once (its first purchase day); a
                    # daily series reads whether the unit converted that day.
                    "converted": None
                    if nulls and conversion_missing(u)
                    else int(bought if daily_conversion else day == first_purchase_day),
                    "returned": None
                    if nulls and retention_missing(u) and not bought
                    else slot.get("returned", 0.0),
                    "sessions": slot.get("sessions", 0)
                    if slot.get("sessions") or not nulls
                    else None,
                    "latency": u.latency[1]
                    if u.latency is not None and day == latency_day
                    else (None if nulls else 0.0),
                    "pre_revenue": u.pre_revenue,
                    "pre_converted": u.pre_converted,
                }
            )
    return rows


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
