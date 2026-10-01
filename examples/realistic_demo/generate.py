"""Synthetic, realistic, NORMALIZED analytics warehouse generator.

Writes a small Hive-partitioned Parquet warehouse (dimension tables, an SCD2
plan-history snapshot, fact tables, and a behavioral event stream) simulating
a single checkout-redesign A/B test (``experiment_id = "checkout_redesign"``).
The warehouse is meant to be analyzed by ``increment``'s semantic layer via
YAML fact-source definitions bound to the exact table/column names below --
do not rename them without a very good reason.

Layout::

    warehouse/
      dim_experiment/experiments.parquet        # unpartitioned, written once
      dim_user/part=00000/users.parquet
      snap_user_plan/part=00000/plans.parquet
      fact_assignment/part=00000/assignments.parquet
      fact_exposure/part=00000/exposures.parquet
      fact_orders/part=00000/orders.parquet
      events/part=00000/events.parquet
      manifest.json

Partition ``p`` owns the disjoint ``user_id`` range ``[p*U+1, (p+1)*U]``;
every table's partition-``p`` file holds only that range, so referential
integrity is partition-local.

Run ``python generate.py --help`` for CLI options. The script always
verifies its own output (see ``_check_warehouse``) after writing and exits
non-zero if any invariant is violated.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Experiment geometry and effect-size constants.
#
# These values are shared across partitions so pooled results remain stable.
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "checkout_redesign"
EXPERIMENT_NAME = "Checkout Redesign"
EXPERIMENT_HYPOTHESIS = (
    "A streamlined one-page checkout increases purchase conversion without "
    "hurting short-term retention."
)
UNIT_TYPE = "user"
CONTROL_VARIANT = "control"
TREATMENT_VARIANT = "treatment"
EXPERIMENT_STATUS = "completed"
GENERATOR_VERSION = 2

#: Experiment window. Plan upgrades occur strictly after ``_END``.
_START = np.datetime64("2025-01-15T00:00:00", "us")
_EXPERIMENT_DURATION_DAYS = 30
_END = _START + np.timedelta64(_EXPERIMENT_DURATION_DAYS, "D")

_ASSIGNMENT_WINDOW_DAYS = 5.0
_EXPOSURE_LAG_DAYS = 2.0  # trigger lands 0..2 days after assignment
_TRIGGER_RATE = 0.60  # arm-independent (counterfactual triggering)

_CONVERSION_WINDOW_DAYS = 14.0
_CONTROL_PURCHASE_RATE = 0.25
_TREATMENT_PURCHASE_RATE = 0.30
_SECOND_ORDER_RATE = 0.25
_SECOND_ORDER_LAG_DAYS = 6.0  # 0..6 days after the first order
_ORDER_REFUND_RATE = 0.08
_ORDER_CURRENCY = "USD"

_RETENTION_RATE = 0.65  # arm-independent guardrail
_RETENTION_WINDOW_LOW_DAYS = 7.0
_RETENTION_WINDOW_HIGH_DAYS = 14.0  # exclusive

_RUM_LAG_DAYS = 2.0  # post-exposure page_load lands within 2 days
_RUM_LOAD_TIME_MEAN_MS = 750.0
_RUM_LOAD_TIME_SD_MS = 120.0

_PRE_HISTORY_LOW_DAYS = 2.0
_PRE_HISTORY_HIGH_DAYS = 60.0
_PRE_PAGE_LOAD_MEAN_MS = 700.0
_PRE_PAGE_LOAD_SD_MS = 150.0

_PLAN_UPGRADE_RATE = 0.30
_PLAN_UPGRADE_LAG_DAYS_LOW = 1.0  # after _END
_PLAN_UPGRADE_LAG_DAYS_HIGH = 45.0
_SIGNUP_LOOKBACK_DAYS_LOW = 30
_SIGNUP_LOOKBACK_DAYS_HIGH = 400

#: Tail traffic extends beyond every metric window, keeping freshness checks
#: meaningful for each event type and for ``fact_orders``.
_TAIL_USERS_PER_PARTITION = 200
_TAIL_OFFSET_DAYS = 40.0

_SENTINEL_VALID_TO = np.datetime64("9999-12-31T00:00:00", "us")

_US_PER_DAY = 86_400_000_000

_COUNTRIES = np.array(["US", "GB", "DE"])
_COUNTRY_WEIGHTS = np.array([0.6, 0.2, 0.2])
_CHANNELS = np.array(["organic", "paid", "referral"])
_CHANNEL_WEIGHTS = np.array([0.5, 0.3, 0.2])
_DEVICES = np.array(["ios", "android", "web"])
_DEVICE_WEIGHTS = np.array([0.45, 0.35, 0.20])
_PAGES = np.array(["/home", "/checkout"])
_PAGE_WEIGHTS = np.array([0.7, 0.3])
_REFERRERS = np.array(["google", "direct", "newsletter", "twitter", ""])


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------


def _days_to_td64us(days: np.ndarray) -> np.ndarray:
    """Fractional-day offsets (float array) -> timedelta64[us] array."""
    return (
        (np.asarray(days, dtype=np.float64) * _US_PER_DAY)
        .astype(np.int64)
        .astype("timedelta64[us]")
    )


def _running_index(counts: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... within each unit's run of ``np.repeat(units, counts)``."""
    counts = np.asarray(counts)
    return np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)


def _session_ids(rng: np.random.Generator, n: int, prefix: str) -> np.ndarray:
    raw = rng.integers(0, 2**40, size=n, dtype=np.int64)
    return np.array([f"{prefix}_{v:010x}" for v in raw])


def _assign_variant(rng: np.random.Generator, user_ids: np.ndarray) -> np.ndarray:
    """Draw independent arm assignments with constant 50/50 probability."""
    is_treatment = rng.random(len(user_ids)) < 0.5
    return np.where(is_treatment, TREATMENT_VARIANT, CONTROL_VARIANT)


def _weighted_choice(
    rng: np.random.Generator, values: np.ndarray, weights: np.ndarray, size: int
) -> np.ndarray:
    return rng.choice(values, size=size, p=weights)


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------


def _gen_dim_user(rng: np.random.Generator, user_ids: np.ndarray) -> dict:
    n = len(user_ids)
    lookback_days = rng.integers(_SIGNUP_LOOKBACK_DAYS_LOW, _SIGNUP_LOOKBACK_DAYS_HIGH, size=n)
    signup_date = _START.astype("datetime64[D]") - lookback_days.astype("timedelta64[D]")
    return {
        "user_id": user_ids,
        "signup_date": signup_date,
        "country": _weighted_choice(rng, _COUNTRIES, _COUNTRY_WEIGHTS, n),
        "acquisition_channel": _weighted_choice(rng, _CHANNELS, _CHANNEL_WEIGHTS, n),
    }


def _gen_snap_user_plan(
    rng: np.random.Generator, user_ids: np.ndarray, signup_date: np.ndarray
) -> dict:
    n = len(user_ids)
    valid_from = signup_date.astype("datetime64[us]")

    upgrade_mask = rng.random(n) < _PLAN_UPGRADE_RATE
    upgrade_lag = rng.uniform(_PLAN_UPGRADE_LAG_DAYS_LOW, _PLAN_UPGRADE_LAG_DAYS_HIGH, size=n)
    upgrade_at = _END + _days_to_td64us(upgrade_lag)

    free_valid_to = np.where(upgrade_mask, upgrade_at, _SENTINEL_VALID_TO).astype("datetime64[us]")

    free_rows = {
        "user_id": user_ids,
        "plan": np.full(n, "free"),
        "valid_from": valid_from,
        "valid_to": free_valid_to,
    }

    upgraded_ids = user_ids[upgrade_mask]
    n_up = len(upgraded_ids)
    pro_rows = {
        "user_id": upgraded_ids,
        "plan": np.full(n_up, "pro"),
        "valid_from": upgrade_at[upgrade_mask],
        "valid_to": np.full(n_up, _SENTINEL_VALID_TO, dtype="datetime64[us]"),
    }

    return {
        "user_id": np.concatenate([free_rows["user_id"], pro_rows["user_id"]]),
        "plan": np.concatenate([free_rows["plan"], pro_rows["plan"]]),
        "valid_from": np.concatenate([free_rows["valid_from"], pro_rows["valid_from"]]),
        "valid_to": np.concatenate([free_rows["valid_to"], pro_rows["valid_to"]]),
    }


# ---------------------------------------------------------------------------
# Fact tables
# ---------------------------------------------------------------------------


def _gen_fact_assignment(rng: np.random.Generator, user_ids: np.ndarray) -> dict:
    n = len(user_ids)
    variant = _assign_variant(rng, user_ids)
    lag = rng.uniform(0.0, _ASSIGNMENT_WINDOW_DAYS, size=n)
    assigned_at = _START + _days_to_td64us(lag)
    return {
        "user_id": user_ids,
        "experiment_id": np.full(n, EXPERIMENT_ID),
        "variant": variant,
        "assigned_at": assigned_at,
    }


def _gen_fact_exposure(
    rng: np.random.Generator,
    user_ids: np.ndarray,
    variant: np.ndarray,
    assigned_at: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (exposure_columns, triggered_user_ids, triggered_variant,
    triggered_base_exposed_at) -- the latter three feed orders/retention/RUM.
    """
    n = len(user_ids)
    triggered_mask = rng.random(n) < _TRIGGER_RATE  # arm-independent
    t_user_ids = user_ids[triggered_mask]
    t_variant = variant[triggered_mask]
    t_assigned_at = assigned_at[triggered_mask]
    n_t = len(t_user_ids)

    trigger_lag = rng.uniform(0.0, _EXPOSURE_LAG_DAYS, size=n_t)
    base_exposed_at = t_assigned_at + _days_to_td64us(trigger_lag)

    counts = rng.integers(1, 4, size=n_t)  # 1..3 exposure rows per triggered user
    idx = _running_index(counts)
    row_user_ids = np.repeat(t_user_ids, counts)
    row_variant = np.repeat(t_variant, counts)
    row_base = np.repeat(base_exposed_at, counts)
    jitter = idx * 0.02 + rng.uniform(0.0, 0.01, size=len(idx))
    exposed_at = row_base + _days_to_td64us(jitter)

    columns = {
        "user_id": row_user_ids,
        "experiment_id": np.full(len(row_user_ids), EXPERIMENT_ID),
        "variant": row_variant,
        "exposed_at": exposed_at,
        "surface": np.full(len(row_user_ids), "checkout"),
    }
    return columns, t_user_ids, t_variant, base_exposed_at


def _gen_fact_orders(
    rng: np.random.Generator,
    triggered_user_ids: np.ndarray,
    triggered_variant: np.ndarray,
    base_exposed_at: np.ndarray,
) -> dict:
    n_t = len(triggered_user_ids)
    purchase_rate = np.where(
        triggered_variant == TREATMENT_VARIANT,
        _TREATMENT_PURCHASE_RATE,
        _CONTROL_PURCHASE_RATE,
    )
    buy_mask = rng.random(n_t) < purchase_rate
    buyer_ids = triggered_user_ids[buy_mask]
    buyer_base = base_exposed_at[buy_mask]
    n_buy = len(buyer_ids)

    first_lag = rng.uniform(0.0, _CONVERSION_WINDOW_DAYS, size=n_buy)
    first_at = buyer_base + _days_to_td64us(first_lag)

    second_mask = rng.random(n_buy) < _SECOND_ORDER_RATE
    second_lag = rng.uniform(0.1, _SECOND_ORDER_LAG_DAYS, size=n_buy)
    second_at = first_at + _days_to_td64us(second_lag)

    order_user_ids = np.concatenate([buyer_ids, buyer_ids[second_mask]])
    ordered_at = np.concatenate([first_at, second_at[second_mask]])
    n_orders = len(order_user_ids)

    item_count = rng.integers(1, 5, size=n_orders).astype(np.int32)
    unit_price = rng.uniform(12.0, 45.0, size=n_orders)
    total_amount = np.round(item_count * unit_price, 2)
    status = np.where(rng.random(n_orders) < _ORDER_REFUND_RATE, "refunded", "completed")

    return {
        "user_id": order_user_ids,
        "ordered_at": ordered_at,
        "status": status,
        "currency": np.full(n_orders, _ORDER_CURRENCY),
        "item_count": item_count,
        "total_amount": total_amount,
    }


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------


def _blank_load_time(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _gen_events_pre(
    rng: np.random.Generator, user_ids: np.ndarray, signup_date: np.ndarray
) -> dict:
    """Every user has activity from BEFORE the experiment: 1-3 page_views
    plus exactly one page_load timing sample, both 2..60 days prior --
    but NEVER before the user's own signup_date (snap_user_plan.valid_from
    for their initial 'free' row equals signup_date, so an event dated
    before it would make that user's plan unresolvable -- a pre_exposure
    property lookup that finds no row at all, landing in the null segment
    instead of 'free'). signup lookback is drawn independently from the
    pre-history lag (30..400 days vs. 2..60 days) and CAN be shorter than
    60 days, so this cap is required, not defensive.
    """
    n = len(user_ids)
    lookback_days = (
        (_START.astype("datetime64[D]") - signup_date.astype("datetime64[D]"))
        .astype("timedelta64[D]")
        .astype(np.float64)
    )
    # 1-day margin below the user's own signup so the event is strictly after
    # it even at datetime64[D]-vs-[us] boundary precision.
    max_lag = np.minimum(_PRE_HISTORY_HIGH_DAYS, lookback_days - 1.0)
    assert np.all(max_lag >= _PRE_HISTORY_LOW_DAYS), (
        "signup lookback too short relative to _PRE_HISTORY_LOW_DAYS -- "
        "_SIGNUP_LOOKBACK_DAYS_LOW must stay well above _PRE_HISTORY_LOW_DAYS + 1"
    )

    pv_counts = rng.integers(1, 4, size=n)
    pv_user_ids = np.repeat(user_ids, pv_counts)
    pv_max_lag = np.repeat(max_lag, pv_counts)
    n_pv = len(pv_user_ids)
    pv_lag = _PRE_HISTORY_LOW_DAYS + rng.uniform(0.0, 1.0, size=n_pv) * (
        pv_max_lag - _PRE_HISTORY_LOW_DAYS
    )
    pv_ts = _START - _days_to_td64us(pv_lag)
    pv = {
        "user_id": pv_user_ids,
        "session_id": _session_ids(rng, n_pv, "pre"),
        "event_name": np.full(n_pv, "page_view"),
        "event_ts": pv_ts,
        "page": _weighted_choice(rng, _PAGES, _PAGE_WEIGHTS, n_pv),
        "referrer": rng.choice(_REFERRERS, size=n_pv),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n_pv),
        "load_time_ms": _blank_load_time(n_pv),
    }

    pl_lag = _PRE_HISTORY_LOW_DAYS + rng.uniform(0.0, 1.0, size=n) * (
        max_lag - _PRE_HISTORY_LOW_DAYS
    )
    pl_ts = _START - _days_to_td64us(pl_lag)
    pl = {
        "user_id": user_ids,
        "session_id": _session_ids(rng, n, "pre"),
        "event_name": np.full(n, "page_load"),
        "event_ts": pl_ts,
        "page": _weighted_choice(rng, _PAGES, _PAGE_WEIGHTS, n),
        "referrer": rng.choice(_REFERRERS, size=n),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n),
        "load_time_ms": np.clip(
            rng.normal(_PRE_PAGE_LOAD_MEAN_MS, _PRE_PAGE_LOAD_SD_MS, size=n), 50.0, None
        ),
    }

    return _concat_dicts([pv, pl])


def _gen_events_rum(
    rng: np.random.Generator, triggered_user_ids: np.ndarray, base_exposed_at: np.ndarray
) -> dict:
    """One or more post-exposure page loads (RUM) per triggered user, latency
    drawn from the SAME distribution regardless of arm -- no guardrail regression."""
    n = len(triggered_user_ids)
    load_counts = rng.integers(1, 5, size=n)
    rum_user_ids = np.repeat(triggered_user_ids, load_counts)
    rum_exposed_at = np.repeat(base_exposed_at, load_counts)
    n_loads = len(rum_user_ids)
    lag = rng.uniform(0.05, _RUM_LAG_DAYS, size=n_loads)
    ts = rum_exposed_at + _days_to_td64us(lag)
    return {
        "user_id": rum_user_ids,
        "session_id": _session_ids(rng, n_loads, "rum"),
        "event_name": np.full(n_loads, "page_load"),
        "event_ts": ts,
        "page": np.full(n_loads, "/checkout"),
        "referrer": rng.choice(_REFERRERS, size=n_loads),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n_loads),
        "load_time_ms": np.clip(
            rng.normal(_RUM_LOAD_TIME_MEAN_MS, _RUM_LOAD_TIME_SD_MS, size=n_loads),
            50.0,
            None,
        ),
    }


def _gen_events_retention(
    rng: np.random.Generator, triggered_user_ids: np.ndarray, base_exposed_at: np.ndarray
) -> tuple[dict, np.ndarray]:
    """~65% of triggered users return with a page_view in [7, 14) days
    after exposure -- arm-independent (the retention rate itself is not
    conditioned on variant here; only exposure was)."""
    n = len(triggered_user_ids)
    retained_mask = rng.random(n) < _RETENTION_RATE
    r_user_ids = triggered_user_ids[retained_mask]
    r_base = base_exposed_at[retained_mask]
    n_r = len(r_user_ids)
    lag = rng.uniform(_RETENTION_WINDOW_LOW_DAYS, _RETENTION_WINDOW_HIGH_DAYS, size=n_r)
    ts = r_base + _days_to_td64us(lag)
    columns = {
        "user_id": r_user_ids,
        "session_id": _session_ids(rng, n_r, "ret"),
        "event_name": np.full(n_r, "page_view"),
        "event_ts": ts,
        "page": _weighted_choice(rng, _PAGES, _PAGE_WEIGHTS, n_r),
        "referrer": rng.choice(_REFERRERS, size=n_r),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n_r),
        "load_time_ms": _blank_load_time(n_r),
    }
    return columns, retained_mask


def _gen_tail(rng: np.random.Generator, partition_user_ids: np.ndarray) -> tuple[dict, dict]:
    """Tail traffic: the first ~200 users in the partition get a page_view,
    a page_load, AND an order dated ~40 days after the experiment start --
    well past every metric's analysis window -- so freshness checks on every
    distinct event_name (and on fact_orders) never trip."""
    tail_ids = partition_user_ids[: min(_TAIL_USERS_PER_PARTITION, len(partition_user_ids))]
    n = len(tail_ids)
    jitter = rng.uniform(0.0, 1.0, size=n)
    tail_ts = _START + _days_to_td64us(_TAIL_OFFSET_DAYS + jitter)

    pv = {
        "user_id": tail_ids,
        "session_id": _session_ids(rng, n, "tail"),
        "event_name": np.full(n, "page_view"),
        "event_ts": tail_ts,
        "page": _weighted_choice(rng, _PAGES, _PAGE_WEIGHTS, n),
        "referrer": rng.choice(_REFERRERS, size=n),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n),
        "load_time_ms": _blank_load_time(n),
    }
    pl_jitter = rng.uniform(0.0, 1.0, size=n)
    pl_ts = _START + _days_to_td64us(_TAIL_OFFSET_DAYS + pl_jitter)
    pl = {
        "user_id": tail_ids,
        "session_id": _session_ids(rng, n, "tail"),
        "event_name": np.full(n, "page_load"),
        "event_ts": pl_ts,
        "page": _weighted_choice(rng, _PAGES, _PAGE_WEIGHTS, n),
        "referrer": rng.choice(_REFERRERS, size=n),
        "device_type": _weighted_choice(rng, _DEVICES, _DEVICE_WEIGHTS, n),
        "load_time_ms": np.clip(
            rng.normal(_PRE_PAGE_LOAD_MEAN_MS, _PRE_PAGE_LOAD_SD_MS, size=n), 50.0, None
        ),
    }
    events_tail = _concat_dicts([pv, pl])

    order_jitter = rng.uniform(0.0, 1.0, size=n)
    order_ts = _START + _days_to_td64us(_TAIL_OFFSET_DAYS + order_jitter)
    item_count = rng.integers(1, 5, size=n).astype(np.int32)
    unit_price = rng.uniform(12.0, 45.0, size=n)
    orders_tail = {
        "user_id": tail_ids,
        "ordered_at": order_ts,
        "status": np.where(rng.random(n) < _ORDER_REFUND_RATE, "refunded", "completed"),
        "currency": np.full(n, _ORDER_CURRENCY),
        "item_count": item_count,
        "total_amount": np.round(item_count * unit_price, 2),
    }
    return events_tail, orders_tail


def _concat_dicts(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


# ---------------------------------------------------------------------------
# Parquet schemas (exact column names/types from the task spec)
# ---------------------------------------------------------------------------

_TS = pa.timestamp("us")
_DATE = pa.date32()

SCHEMAS: dict[str, pa.Schema] = {
    "dim_experiment": pa.schema(
        [
            ("experiment_id", pa.string()),
            ("name", pa.string()),
            ("hypothesis", pa.string()),
            ("unit_type", pa.string()),
            ("start_date", _DATE),
            ("end_date", _DATE),
            ("status", pa.string()),
            ("control_variant", pa.string()),
        ]
    ),
    "dim_user": pa.schema(
        [
            ("user_id", pa.int64()),
            ("signup_date", _DATE),
            ("country", pa.string()),
            ("acquisition_channel", pa.string()),
        ]
    ),
    "snap_user_plan": pa.schema(
        [
            ("user_id", pa.int64()),
            ("plan", pa.string()),
            ("valid_from", _TS),
            ("valid_to", _TS),
        ]
    ),
    "fact_assignment": pa.schema(
        [
            ("user_id", pa.int64()),
            ("experiment_id", pa.string()),
            ("variant", pa.string()),
            ("assigned_at", _TS),
        ]
    ),
    "fact_exposure": pa.schema(
        [
            ("user_id", pa.int64()),
            ("experiment_id", pa.string()),
            ("variant", pa.string()),
            ("exposed_at", _TS),
            ("surface", pa.string()),
        ]
    ),
    "fact_orders": pa.schema(
        [
            ("order_id", pa.int64()),
            ("user_id", pa.int64()),
            ("ordered_at", _TS),
            ("status", pa.string()),
            ("currency", pa.string()),
            ("item_count", pa.int32()),
            ("total_amount", pa.float64()),
        ]
    ),
    "events": pa.schema(
        [
            ("event_id", pa.int64()),
            ("user_id", pa.int64()),
            ("session_id", pa.string()),
            ("event_name", pa.string()),
            ("event_ts", _TS),
            ("page", pa.string()),
            ("referrer", pa.string()),
            ("device_type", pa.string()),
            ("load_time_ms", pa.float64()),
        ]
    ),
}


def _table(name: str, columns: dict) -> pa.Table:
    schema = SCHEMAS[name]
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Partition orchestration
# ---------------------------------------------------------------------------

_PARTITION_ID_STRIDE = 10_000_000  # generous headroom vs. users-per-partition


def _generate_partition(p: int, users_per_partition: int, seed: int) -> dict[str, pa.Table]:
    rng = np.random.default_rng([seed, p])

    user_ids = np.arange(p * users_per_partition + 1, (p + 1) * users_per_partition + 1)

    dim_user_cols = _gen_dim_user(rng, user_ids)
    snap_user_plan_cols = _gen_snap_user_plan(rng, user_ids, dim_user_cols["signup_date"])
    assignment_cols = _gen_fact_assignment(rng, user_ids)
    exposure_cols, t_user_ids, t_variant, base_exposed_at = _gen_fact_exposure(
        rng, user_ids, assignment_cols["variant"], assignment_cols["assigned_at"]
    )
    orders_cols = _gen_fact_orders(rng, t_user_ids, t_variant, base_exposed_at)
    rum_events = _gen_events_rum(rng, t_user_ids, base_exposed_at)
    retention_events, _retained_mask = _gen_events_retention(rng, t_user_ids, base_exposed_at)
    pre_events = _gen_events_pre(rng, user_ids, dim_user_cols["signup_date"])
    tail_events, tail_orders = _gen_tail(rng, user_ids)

    all_orders = _concat_dicts([orders_cols, tail_orders])
    n_orders = len(all_orders["user_id"])
    assert n_orders < _PARTITION_ID_STRIDE, (
        f"partition {p}: {n_orders} orders exceeds _PARTITION_ID_STRIDE="
        f"{_PARTITION_ID_STRIDE}; order_id ranges would collide across partitions "
        f"-- raise _PARTITION_ID_STRIDE or shrink users_per_partition"
    )
    order_offset = p * _PARTITION_ID_STRIDE
    all_orders["order_id"] = order_offset + np.arange(n_orders, dtype=np.int64)

    all_events = _concat_dicts([pre_events, rum_events, retention_events, tail_events])
    n_events = len(all_events["user_id"])
    assert n_events < _PARTITION_ID_STRIDE, (
        f"partition {p}: {n_events} events exceeds _PARTITION_ID_STRIDE="
        f"{_PARTITION_ID_STRIDE}; event_id ranges would collide across partitions "
        f"-- raise _PARTITION_ID_STRIDE or shrink users_per_partition"
    )
    event_offset = p * _PARTITION_ID_STRIDE
    all_events["event_id"] = event_offset + np.arange(n_events, dtype=np.int64)

    return {
        "dim_user": _table("dim_user", dim_user_cols),
        "snap_user_plan": _table("snap_user_plan", snap_user_plan_cols),
        "fact_assignment": _table("fact_assignment", assignment_cols),
        "fact_exposure": _table("fact_exposure", exposure_cols),
        "fact_orders": _table("fact_orders", all_orders),
        "events": _table("events", all_events),
    }


def _dim_experiment_table() -> pa.Table:
    columns = {
        "experiment_id": [EXPERIMENT_ID],
        "name": [EXPERIMENT_NAME],
        "hypothesis": [EXPERIMENT_HYPOTHESIS],
        "unit_type": [UNIT_TYPE],
        "start_date": np.array([_START], dtype="datetime64[D]"),
        "end_date": np.array([_END], dtype="datetime64[D]"),
        "status": [EXPERIMENT_STATUS],
        "control_variant": [CONTROL_VARIANT],
    }
    return _table("dim_experiment", columns)


_TABLE_FILENAMES = {
    "dim_user": "users.parquet",
    "snap_user_plan": "plans.parquet",
    "fact_assignment": "assignments.parquet",
    "fact_exposure": "exposures.parquet",
    "fact_orders": "orders.parquet",
    "events": "events.parquet",
}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_partitioned(out: Path, table_name: str, p: int, table: pa.Table) -> None:
    part_dir = out / table_name / f"part={p:05d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / _TABLE_FILENAMES[table_name])


def generate(out: Path, partitions: int, users_per_partition: int, seed: int) -> dict:
    """Generate and write the full warehouse. Returns per-table row counts."""
    row_counts: dict[str, int] = dict.fromkeys(_TABLE_FILENAMES, 0)

    # Clear each partitioned table's directory first. Without this, re-running
    # with FEWER partitions than a previous run leaves orphaned part=NNNNN
    # directories behind -- silently mixing stale and fresh data for any
    # downstream consumer that globs partitions (the whole point of this layout).
    partitioned_tables = [name for name in _TABLE_FILENAMES if name != "dim_experiment"]
    for name in partitioned_tables:
        table_dir = out / name
        if table_dir.exists():
            shutil.rmtree(table_dir)

    for p in range(partitions):
        tables = _generate_partition(p, users_per_partition, seed)
        for name, table in tables.items():
            _write_partitioned(out, name, p, table)
            row_counts[name] += table.num_rows
    dim_experiment_dir = out / "dim_experiment"
    dim_experiment_dir.mkdir(parents=True, exist_ok=True)
    dim_experiment = _dim_experiment_table()
    pq.write_table(dim_experiment, dim_experiment_dir / "experiments.parquet")
    row_counts["dim_experiment"] = dim_experiment.num_rows

    return row_counts


def warehouse_manifest(
    *,
    partitions: int,
    users_per_partition: int,
    seed: int,
    row_counts: dict[str, int],
) -> dict:
    """Describe warehouse inputs and the generator semantics that produced it."""
    return {
        "generator_version": GENERATOR_VERSION,
        "partitions": partitions,
        "users_per_partition": users_per_partition,
        "seed": seed,
        "experiment_id": EXPERIMENT_ID,
        "row_counts": row_counts,
    }


# ---------------------------------------------------------------------------
# Post-write verification -- real assertions against the written Parquet,
# not just comments. Fails loudly (raises) on any violation.
# ---------------------------------------------------------------------------


def _read_partition(out: Path, table_name: str, p: int) -> pa.Table:
    part_dir = out / table_name / f"part={p:05d}"
    return pq.read_table(part_dir / _TABLE_FILENAMES[table_name])


def _read_all(out: Path, table_name: str, partitions: int) -> pa.Table:
    tables = [_read_partition(out, table_name, p) for p in range(partitions)]
    return pa.concat_tables(tables)


def _col(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def _check_warehouse(out: Path, partitions: int, users_per_partition: int) -> None:
    seen_user_ids: set[int] = set()

    for p in range(partitions):
        dim_user = _read_partition(out, "dim_user", p)
        partition_ids = set(_col(dim_user, "user_id").tolist())

        expected_range = set(range(p * users_per_partition + 1, (p + 1) * users_per_partition + 1))
        assert partition_ids == expected_range, (
            f"partition {p}: dim_user.user_id does not equal the owned range "
            f"[{p * users_per_partition + 1}, {(p + 1) * users_per_partition}]"
        )

        # Partitions never share user_id values.
        overlap = seen_user_ids & partition_ids
        assert not overlap, f"partition {p}: user_id overlap with earlier partitions: {overlap!r}"
        seen_user_ids |= partition_ids

        # Every user_id referenced in any fact/event table of this partition
        # exists in this same partition's dim_user rows (partition-local FK).
        for table_name in (
            "snap_user_plan",
            "fact_assignment",
            "fact_exposure",
            "fact_orders",
            "events",
        ):
            table = _read_partition(out, table_name, p)
            referenced = set(_col(table, "user_id").tolist())
            missing = referenced - partition_ids
            assert not missing, (
                f"partition {p}: {table_name} references user_id(s) not in this "
                f"partition's dim_user: {sorted(missing)[:10]!r}"
            )

    # --- Combined (cross-partition) checks -------------------------------

    fact_assignment = _read_all(out, "fact_assignment", partitions)
    fact_exposure = _read_all(out, "fact_exposure", partitions)
    fact_orders = _read_all(out, "fact_orders", partitions)
    events = _read_all(out, "events", partitions)
    snap_user_plan = _read_all(out, "snap_user_plan", partitions)

    assigned_ids = set(_col(fact_assignment, "user_id").tolist())
    exposed_ids = set(_col(fact_exposure, "user_id").tolist())
    assert exposed_ids <= assigned_ids, (
        "fact_exposure has user_id(s) with no fact_assignment row: "
        f"{sorted(exposed_ids - assigned_ids)[:10]!r}"
    )

    # Triggering rate must be close to equal across arms (counterfactual
    # triggering: the redesign shouldn't change WHO triggers). The tolerance
    # is a statistical bound, not a fixed constant: at small N the sampling
    # noise on the arm-vs-arm gap is large (SE ~ 1/sqrt(n_per_arm)), so a
    # fixed absolute tolerance would fail on perfectly valid small datasets.
    # A fixed floor at the high-N end guards against an assertion so loose it
    # stops catching a real arm-dependence bug.
    assignment_user_id = _col(fact_assignment, "user_id")
    assignment_variant = _col(fact_assignment, "variant")
    trigger_rates = {}
    arm_sizes = {}
    for arm in (CONTROL_VARIANT, TREATMENT_VARIANT):
        arm_ids = set(assignment_user_id[assignment_variant == arm].tolist())
        assert arm_ids, (
            f"no assigned users in arm {arm!r} -- increase users_per_partition "
            f"(50/50 assignment needs enough users for both arms to be non-empty)"
        )
        arm_sizes[arm] = len(arm_ids)
        trigger_rates[arm] = len(arm_ids & exposed_ids) / len(arm_ids)
    rate_gap = abs(trigger_rates[CONTROL_VARIANT] - trigger_rates[TREATMENT_VARIANT])
    # Pooled SE of a two-proportion difference at the true (arm-independent)
    # trigger probability; 6 sigma keeps the false-positive rate on genuinely
    # valid data negligible while still catching a real arm-dependence bug.
    pooled_se = math.sqrt(
        _TRIGGER_RATE
        * (1 - _TRIGGER_RATE)
        * (1 / arm_sizes[CONTROL_VARIANT] + 1 / arm_sizes[TREATMENT_VARIANT])
    )
    tolerance = max(0.02, 6 * pooled_se)
    assert rate_gap <= tolerance, (
        f"triggering rate differs by arm beyond tolerance: {trigger_rates!r} "
        f"(gap={rate_gap:.4f}, tolerance={tolerance:.4f}, arm_sizes={arm_sizes!r})"
    )

    # Freshness: every distinct events.event_name, and fact_orders, must
    # have a max timestamp well past every metric's analysis window close.
    freshness_floor = _START + np.timedelta64(int(_TAIL_OFFSET_DAYS) - 2, "D")
    event_name = _col(events, "event_name")
    event_ts = _col(events, "event_ts")
    for name in sorted(set(event_name.tolist())):
        max_ts = event_ts[event_name == name].max()
        assert max_ts >= freshness_floor, (
            f"events.event_name={name!r} max(event_ts)={max_ts} is not past the "
            f"freshness floor {freshness_floor}"
        )
    order_ts = _col(fact_orders, "ordered_at")
    assert order_ts.max() >= freshness_floor, (
        f"fact_orders max(ordered_at)={order_ts.max()} is not past the freshness "
        f"floor {freshness_floor}"
    )

    # In-band retention: strictly between 0 and 1 (not saturated).
    exposure_user_id = _col(fact_exposure, "user_id")
    exposure_ts = _col(fact_exposure, "exposed_at")
    order_idx = np.argsort(exposure_user_id, kind="stable")
    sorted_ids = exposure_user_id[order_idx]
    sorted_ts = exposure_ts[order_idx]
    unique_ids, first_pos = np.unique(sorted_ids, return_index=True)
    base_exposed_at = {}
    boundaries = np.append(first_pos, len(sorted_ids))
    for i, uid in enumerate(unique_ids):
        seg = sorted_ts[boundaries[i] : boundaries[i + 1]]
        base_exposed_at[int(uid)] = seg.min()

    page_view_mask = event_name == "page_view"
    pv_user_id = _col(events, "user_id")[page_view_mask]
    pv_ts = event_ts[page_view_mask]
    pv_by_user: dict[int, np.ndarray] = {}
    for uid, ts in zip(pv_user_id.tolist(), pv_ts, strict=True):
        pv_by_user.setdefault(uid, []).append(ts)

    low = np.timedelta64(int(_RETENTION_WINDOW_LOW_DAYS), "D")
    high = np.timedelta64(int(_RETENTION_WINDOW_HIGH_DAYS), "D")
    retained = 0
    for uid, base in base_exposed_at.items():
        ts_list = pv_by_user.get(uid)
        if ts_list is None:
            continue
        ts_arr = np.array(ts_list)
        if np.any((ts_arr >= base + low) & (ts_arr < base + high)):
            retained += 1
    assert base_exposed_at, (
        "no triggered (exposed) users at all -- increase users_per_partition; "
        "the retention/in-band checks below require at least one triggered user"
    )
    retention_rate = retained / len(base_exposed_at)
    assert 0.0 < retention_rate < 1.0, f"in-band retention rate is saturated: {retention_rate!r}"

    # No row in snap_user_plan has a NULL/None valid_to.
    valid_to = snap_user_plan.column("valid_to")
    assert valid_to.null_count == 0, (
        f"snap_user_plan.valid_to has {valid_to.null_count} null row(s)"
    )

    # order_id/event_id are assigned per-partition as p * _PARTITION_ID_STRIDE +
    # local index; the per-partition assertions in _generate_partition guard
    # against any one partition overrunning the stride, but this is the direct
    # cross-partition check that global uniqueness actually held.
    order_ids = _col(fact_orders, "order_id")
    assert len(set(order_ids.tolist())) == len(order_ids), (
        "fact_orders.order_id is not globally unique across partitions"
    )
    event_ids = _col(events, "event_id")
    assert len(set(event_ids.tolist())) == len(event_ids), (
        "events.event_id is not globally unique across partitions"
    )

    # No events (or orders) row predates its own user's earliest snap_user_plan
    # row (i.e. their signup). An event before signup would make a
    # pre_exposure plan lookup for that timestamp find no row -- the property
    # resolves to null instead of 'free', which silently breaks every
    # analysis that breaks out by plan. Regression guard for that class of bug.
    plan_uid = _col(snap_user_plan, "user_id")
    plan_from = _col(snap_user_plan, "valid_from")
    earliest_valid_from: dict[int, np.datetime64] = {}
    for uid, vf in zip(plan_uid.tolist(), plan_from, strict=True):
        prev = earliest_valid_from.get(uid)
        if prev is None or vf < prev:
            earliest_valid_from[uid] = vf
    for label, table, ts_col in (
        ("events", events, "event_ts"),
        ("fact_orders", fact_orders, "ordered_at"),
    ):
        row_uid = _col(table, "user_id")
        row_ts = _col(table, ts_col)
        floor = np.array([earliest_valid_from[u] for u in row_uid.tolist()])
        violations = row_ts < floor
        assert not violations.any(), (
            f"{label}: {int(violations.sum())} row(s) have {ts_col} before their "
            f"user's own snap_user_plan.valid_from (i.e. before signup) -- a "
            f"pre_exposure plan lookup at that timestamp would find no row"
        )

    print(
        "check: OK -- "
        f"trigger_rates={trigger_rates!r} retention_rate={retention_rate:.4f} "
        f"freshness_floor={freshness_floor}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_out() -> Path:
    return Path(__file__).resolve().parent / "warehouse"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, required=True, help="number of partitions")
    parser.add_argument(
        "--users-per-partition", type=int, required=True, help="users owned by each partition"
    )
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed")
    parser.add_argument(
        "--out", type=Path, default=None, help="warehouse output directory (default: ./warehouse)"
    )
    args = parser.parse_args(argv)

    if args.partitions < 1:
        parser.error("--partitions must be >= 1")
    if args.users_per_partition < 1:
        parser.error("--users-per-partition must be >= 1")

    out = args.out if args.out is not None else _default_out()
    out.mkdir(parents=True, exist_ok=True)

    row_counts = generate(out, args.partitions, args.users_per_partition, args.seed)

    manifest = warehouse_manifest(
        partitions=args.partitions,
        users_per_partition=args.users_per_partition,
        seed=args.seed,
        row_counts=row_counts,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _check_warehouse(out, args.partitions, args.users_per_partition)

    total_rows = sum(row_counts.values())
    print(
        f"wrote {total_rows} rows across {len(row_counts)} tables to {out} "
        f"({args.partitions} partitions x {args.users_per_partition} users)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
