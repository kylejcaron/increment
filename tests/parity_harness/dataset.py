"""One shared per-unit(-day) dataset builder feeding every Analysis
constructor a case needs, per docs.increment/AGENTS.md's "Ingress paths
and method scope": "identical per-unit data through each path must
produce the same moments and the same interval."

Every ParityCase in cases.py calls event_rows()/definitions_dict() below
(sometimes parameterised) to get its own copy of the underlying truth, then
builds each constructor it attempts from THAT copy, so a comparison across
constructors is a comparison of the same units, the same purchases, the
same pre-period covariate and the same store assignment -- never two
datasets that merely look similar.

Every case passes ONE `AnalysisPlan` (`increment.semantics.models.AnalysisPlan`,
the exact type `Experiment.plan` uses) into `definitions_dict(plan=...)` for
the definitions/artifact routes AND directly as `plan=` to `from_unit_summary`/
`from_unit_panel`/`from_moments` -- dataframe builders receive the same
plan the warehouse route declares. One nuance a probe surfaced, filed as a
tracked gap rather than treated as intended: a per-metric method override
(`sensitivity_methods=[cuped]`) is declared on an `ExperimentMetric` plan
entry for the definitions/artifact routes, but the frame path refuses that
same override at the PLAN level (`plan entry '...': sensitivity_methods=
overrides are not supported on the frame path -- declare them on the
frame's own MetricSpec for this metric instead`). There is no principled
reason `from_unit_summary`/`from_unit_panel`/`from_moments` could not
accept the identical `ExperimentMetric`-bearing plan the warehouse routes
already do; this is a real, narrower-than-necessary refusal on the
dataframe path, not a designed difference between paths -- follow the
tracker item for the actual fix. Until it lands, a harness case with a
method-overridden metric works around the gap with two plan objects that
agree on role/q/alpha but differ in WHERE the override lives: an
`ExperimentMetric`-bearing plan for definitions/artifact, and a bare-name
plan for the frame paths, with the override moved onto the corresponding
`MetricSpec.sensitivity_methods` instead. Both plans still produce the
identical row set once wired correctly -- confirmed with a probe before
writing this file: an `ExperimentMetric`-only plan on the frame path
silently produced ONE row (`cuped`) where the definitions route produces
TWO (`unadjusted` decision plus `cuped` sensitivity), a row-set mismatch
the harness's `assert_parity` now specifically exists to catch.

Every timestamp below is chosen deliberately; moving one without re-reading
the comment attached to it silently drops a whole arm to zero rather than
raising a comparable error:

- `_PURCHASE_AT`/`_SESSION_AT` sit well inside the 1-day metric window
  (exposure 09:00, window close next day 09:00) -- a purchase timestamped
  AT the window boundary is silently excluded from both arms, which zeroes
  the arm mean and trips the unrelated zero-mean-abort bug tracked elsewhere
  (a fix this harness must not depend on).
- `_PRE_PURCHASE_AT` sits inside the declared `n_pre_periods: 7` lookback
  from the experiment start -- a pre-period event outside that lookback is
  invisible to CUPED, which then imputes every unit to the same pooled
  constant and refuses with `estimation.cuped.covariate_zero_variance`.
- `_FRESHNESS_PAD_AT` is a same-fact, zero-valued event dated after the
  experiment's declared `end`. `_censor_to_observable_window` bases each
  metric's freshness bound on the fact's OWN latest observed timestamp, not
  on the declared `end`; without this padding row every unit's metric
  window "closes after the observable bound" and is censored to nothing,
  which -- again -- zeroes both arms' means.
- Store assignment (`f"s{(i // 5) % n_stores}"`) is the SAME sequence in
  both arms. Two hazards, both confirmed live: a store id prefixed by arm
  (`f"{prefix}s{i % n_stores}"`, the natural first attempt) leaves every
  segment single-arm, and every breakout comparison refuses with no
  comparison possible. A store index taken directly from `i % n_stores`
  (the next natural attempt, with `n_stores=3`) silently correlates with
  the exact-binomial non-purchaser cohort's own `i % 3 != 0` skip
  condition -- with `n_stores=3`, `i % 3 == 0` selects BOTH "store s0" AND
  "never purchases", so segment s0's `revenue`/`rps` mean is exactly zero
  for every unit in every arm, and `run_breakout` refuses with
  `family.evidence.incomplete` regardless of sample size (confirmed live
  at 4x this seed's `n_per_arm`: the failure persists unchanged). Dividing
  `i` by 5 before taking `% n_stores` decorrelates the two conditions
  (`gcd(5, 3) = 1`, so no store's cohort aligns with the skip condition);
  confirmed live with the corrected formula: every store segment carries a
  real mix of purchasers and non-purchasers in both arms, and
  `run_breakout` succeeds at this seed's ordinary `n_per_arm=60` with no
  need to enlarge the fixture at all.
- Every metric declares `preferred_direction` (both in the YAML `metrics:`
  catalog and on each frame-path `MetricSpec`) -- a guardrail role refuses
  at plan compilation (`plan.guardrail.direction`) without one, confirmed
  live by a probe attempting `definitions_dict` with an undeclared-direction
  metric under a `guardrails:` role.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import ibis
import pandas as pd
import pyarrow as pa

from increment.semantics.models import AnalysisPlan

_EXPOSURE_AT = dt.datetime(2025, 1, 10, 9)
_PRE_PURCHASE_AT = dt.datetime(2025, 1, 5, 9)  # inside the 7-day n_pre_periods lookback
_PURCHASE_AT = dt.datetime(2025, 1, 10, 15)  # inside the 1-day window, not at its boundary
_SESSION_AT = dt.datetime(2025, 1, 10, 16)
_LATENCY_AT = dt.datetime(2025, 1, 10, 15)
_FRESHNESS_PAD_AT = dt.datetime(2025, 1, 19, 9)  # after `end`; keeps censoring from firing
_EXPERIMENT_END = dt.date(2025, 1, 20)
_N_PER_ARM = 60
_N_STORES = 3


def _row(
    user_id: str,
    event_at: dt.datetime,
    event: str,
    *,
    group_id: str | None = None,
    experiment_id: str | None = None,
    store_id: str | None = None,
    revenue: float | None = None,
    sess: int | None = None,
    latency: float | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "event_at": event_at,
        "event": event,
        "group_id": group_id,
        "experiment_id": experiment_id,
        "store_id": store_id,
        "revenue": revenue,
        "sess": sess,
        "latency": latency,
    }


def store_for(i: int, n_stores: int = _N_STORES) -> str:
    """The store unit *i* of an arm belongs to in `event_rows`."""
    return f"s{(i // 5) % n_stores}"


def event_rows(*, n_per_arm: int = _N_PER_ARM, n_stores: int = _N_STORES) -> list[dict[str, Any]]:
    """Exposure, pre-period purchase (the CUPED covariate), an in-window
    purchase for two thirds of units (the other third is a genuine
    non-purchaser cohort -- the exact-binomial zero cell), an in-window
    session (the ratio denominator), and freshness padding -- for
    `2 * n_per_arm` units split evenly across `n_stores` breakout segments
    shared identically across both arms.
    """
    rows: list[dict[str, Any]] = []
    for arm, prefix in (("control", "c"), ("treatment", "t")):
        for i in range(n_per_arm):
            uid = f"{prefix}{i}"
            store = store_for(i, n_stores)
            rows.append(
                _row(
                    uid, _EXPOSURE_AT, "exposure", group_id=arm, experiment_id="exp", store_id=store
                )
            )
            rows.append(
                _row(uid, _PRE_PURCHASE_AT, "purchase", store_id=store, revenue=3.0 + (i % 4))
            )
            if i % 3 != 0:
                rows.append(
                    _row(
                        uid,
                        _PURCHASE_AT,
                        "purchase",
                        store_id=store,
                        revenue=5.0 + (arm == "treatment") + (i % 5),
                    )
                )
            rows.append(_row(uid, _SESSION_AT, "session_end", store_id=store, sess=1))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "purchase", store_id=store, revenue=0.0))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "session_end", store_id=store, sess=0))
    return rows


def duckdb_connection(rows: list[dict[str, Any]] | None = None) -> ibis.BaseBackend:
    """`_row`'s shared shape carries every optional column (`revenue`,
    `sess`, `latency`) even for a case that never sets one of them --
    e.g. `event_rows()` never sets `latency`, so that column is entirely
    `None`. `pyarrow.Table.from_pylist` infers an all-`None` column as
    its `null` type, which DuckDB refuses to create a table column with
    (confirmed live: `ibis.common.exceptions.IbisTypeError: DuckDB does
    not support creating tables with NULL typed columns`) -- cast any
    such column to `float64` (every optional column here is numeric)
    before handing the table to DuckDB.
    """
    con = ibis.duckdb.connect()
    table = pa.Table.from_pylist(rows if rows is not None else event_rows())
    for i, field in enumerate(table.schema):
        if pa.types.is_null(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.float64()))
    con.create_table("events", obj=table)
    return con


def definitions_dict(
    *, plan: AnalysisPlan, breakout: bool = False, allocation: dict[str, float] | None = None
) -> dict[str, Any]:
    """One fact source (`events`), four metrics (mean/conversion/ratio/
    CUPED-mean), sharing the rows `event_rows()` produces. `plan` is
    embedded verbatim (`plan.model_dump(mode="json")`) -- the caller
    decides role/q/alpha/inference; this function only wires the fact/metric
    catalog under it. `allocation`, when given, declares the experiment's
    assignment weights explicitly (required for a sequential `plan`, whose
    validator needs a declared allocation to size the asymptotic boundary).
    """
    experiment: dict[str, Any] = {
        "name": "exp",
        "exposure": "assignment",
        "unit": "user_id",
        "start": "2025-01-10",
        "end": _EXPERIMENT_END.isoformat(),
        "control_group": "control",
        "n_pre_periods": 7,
        "plan": plan.model_dump(mode="json"),
    }
    if allocation is not None:
        experiment["allocation"] = allocation
    if breakout:
        experiment["breakouts"] = [{"property": "store"}]
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "session_end", "column": "sess"},
                ],
                "properties": [
                    {"name": "store", "column": "store_id", "dtype": "string", "as_of": "static"}
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "conversion",
                "name": "purchase_rate",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "ratio",
                "name": "rps",
                "entity": "user_id",
                "preferred_direction": "increase",
                "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
                "denominator": {"fact": "session_end", "aggregation": "count", "window_days": 1},
            },
            {
                "type": "mean",
                "name": "revenue_cuped",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            },
        ],
        "experiments": [experiment],
    }


_WINDOW_SQL = "event_at >= timestamp '2025-01-10 09:00' and event_at < timestamp '2025-01-11 09:00'"


def unit_summary_frame(con: ibis.BaseBackend) -> pa.Table:
    """The dataframe oracle: one row per unit, aggregated from the SAME
    `events` table the definitions/artifact readers query, so the
    dataframe and warehouse projections agree on every unit's totals by
    construction rather than by coincidence."""
    return con.sql(  # ty: ignore[unresolved-attribute]
        f"""
        select
          user_id,
          any_value(store_id) as store,
          cast(min(event_at) filter (where event = 'exposure') as date) as exposure_date,
          any_value(group_id) filter (where event = 'exposure') as variant,
          sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) as revenue,
          case when sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) > 0
               then 1 else 0 end as converted,
          count(*) filter (where event = 'session_end' and {_WINDOW_SQL}) as sessions,
          sum(revenue) filter (where event = 'purchase' and event_at < timestamp '2025-01-10 09:00')
            as pre_revenue
        from events
        group by 1
        """
    ).to_pyarrow()


def unit_panel_frame(summary: pa.Table) -> pd.DataFrame:
    """A two-day panel (pre-exposure day zeroed; exposure day carrying the
    same fixed-horizon totals `unit_summary_frame` computed) built FROM the
    unit-summary projection, so panel and summary agree by construction.
    `pre_revenue` -- the CUPED covariate -- is a genuine pre-period value:
    replicated identically on both of a unit's rows, so it resolves as a
    stable per-unit constant the same way `from_unit_summary`'s one-row
    frame already carries it."""
    rows: list[dict[str, Any]] = []
    for r in summary.to_pylist():
        pre_revenue = r["pre_revenue"]
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "store": r["store"],
                "exposure_date": r["exposure_date"],
                "date": dt.date(2025, 1, 10),
                "revenue": 0.0,
                "sessions": 0,
                "converted": 0,
                "pre_revenue": pre_revenue,
            }
        )
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "store": r["store"],
                "exposure_date": r["exposure_date"],
                "date": dt.date(2025, 1, 11),
                "revenue": r["revenue"] or 0.0,
                "sessions": r["sessions"] or 0,
                "converted": r["converted"],
                "pre_revenue": pre_revenue,
            }
        )
    return pd.DataFrame(rows)


def sequential_unit_panel_frame(con: ibis.BaseBackend) -> pd.DataFrame:
    """A REAL per-day panel for the sequential case: `capture_sequential`
    on a panel source needs an explicit `exposure_date` column and a
    window bound (`MetricSpec.window_days`) resolved against each row's
    OWN calendar day -- unlike `unit_panel_frame` above (used by the
    fixed-horizon cases, whose `run()` has no `exposure_date` and simply
    sums every day present, so which day a value lands on is cosmetic),
    this function buckets `revenue` by the actual day `_PURCHASE_AT`
    falls on (the same calendar day as `_EXPOSURE_AT`) rather than a
    synthetic day-0-zero/day-1-real split -- confirmed live: the
    synthetic split produces `note='zero_arm_variance'` (the window
    [exposure, exposure+1) excludes the synthetic day carrying the real
    value), while bucketing by the real day matches the other four
    constructors' point/interval exactly (`0.14285714285714285`, the same
    value `sequential_asymptotic_mean_revenue`'s other constructors
    already produce). Also carries a daily `sessions` count (from
    `session_end`), unioned into the same per-day rows via the shared
    `daily` CTE so a case needing a same-day ratio/conversion secondary
    (revenue/sessions, or revenue > 0) does not need its own query."""
    df = con.sql(  # ty: ignore[unresolved-attribute]
        """
        with variant as (
          select user_id, any_value(group_id) as variant from events where event = 'exposure' group by 1
        ),
        daily as (
          select user_id, date_trunc('day', event_at) as day,
                 sum(revenue) filter (where event = 'purchase') as revenue,
                 sum(sess) filter (where event = 'session_end') as sessions
          from events
          where event in ('purchase', 'session_end')
          group by 1, 2
        )
        select v.user_id, v.variant, d.day, d.revenue, d.sessions
        from variant v join daily d on v.user_id = d.user_id
        """
    ).to_pyarrow()
    return pd.DataFrame(
        [
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "exposure_date": _EXPOSURE_AT.date(),
                "date": r["day"].date() if hasattr(r["day"], "date") else r["day"],
                "revenue": r["revenue"] or 0.0,
                "sessions": r["sessions"] or 0,
            }
            for r in df.to_pylist()
        ]
    )


# -- Multiplicity-bearing dataset: 3 arms (multi-arm primary), 2 secondaries
# (one BH-selected, one not), 1 guardrail. All five reachable constructors
# (switchback needs a schedule shape) give identical role/discovery/value rows.

_MULTIPLICITY_ARMS: dict[str, tuple[float, float, float]] = {
    # arm -> (revenue bump, purchase-rate bump, latency baseline)
    "control": (0.0, 0.0, 200.0),
    "treatment_a": (2.0, 0.35, 195.0),
    "treatment_b": (2.2, 0.05, 198.0),
}


def multiplicity_event_rows(*, n_per_arm: int = 80) -> list[dict[str, Any]]:
    """Three arms against one control; `revenue` is the multi-arm primary
    (Bonferroni-split across `treatment_a`/`treatment_b`), `purchase_rate`
    and `rps` are secondaries (only `purchase_rate`'s and `rps`'s
    `treatment_a` cell clears a loose q=0.30 BH threshold -- `treatment_b`'s
    `purchase_rate` cell does not, so the row set spans both a discovered
    and an undiscovered secondary cell), and `latency` is a guardrail
    (`preferred_direction="decrease"`, so its `alternative` reads "less").
    """
    rows: list[dict[str, Any]] = []
    for arm, (bump, purchase_rate_bump, latency_base) in _MULTIPLICITY_ARMS.items():
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            rows.append(_row(uid, _EXPOSURE_AT, "exposure", group_id=arm, experiment_id="exp"))
            purchased = (i % 100) < (30 + purchase_rate_bump * 100)
            if purchased:
                rows.append(_row(uid, _PURCHASE_AT, "purchase", revenue=5.0 + bump + (i % 5) * 0.1))
            rows.append(_row(uid, _SESSION_AT, "session_end", sess=1))
            rows.append(_row(uid, _LATENCY_AT, "latency_sample", latency=latency_base + (i % 7)))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "purchase", revenue=0.0))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "session_end", sess=0))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "latency_sample", latency=0.0))
    return rows


def multiplicity_allocation() -> dict[str, float]:
    return dict.fromkeys(_MULTIPLICITY_ARMS, 1 / 3)


def multiplicity_definitions_dict(*, plan: AnalysisPlan) -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "session_end", "column": "sess"},
                    {"name": "latency_sample", "column": "latency"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "conversion",
                "name": "purchase_rate",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "ratio",
                "name": "rps",
                "entity": "user_id",
                "preferred_direction": "increase",
                "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
                "denominator": {"fact": "session_end", "aggregation": "count", "window_days": 1},
            },
            {
                "type": "mean",
                "name": "latency",
                "entity": "user_id",
                "fact": "latency_sample",
                "aggregation": "avg_calendar_day",
                "window_days": 1,
                "preferred_direction": "decrease",
            },
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2025-01-10",
                "end": _EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "allocation": multiplicity_allocation(),
                "plan": plan.model_dump(mode="json"),
            }
        ],
    }


def multiplicity_summary_frame(con: ibis.BaseBackend) -> pa.Table:
    return con.sql(  # ty: ignore[unresolved-attribute]
        f"""
        select
          user_id,
          any_value(group_id) filter (where event = 'exposure') as variant,
          sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) as revenue,
          case when sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) > 0
               then 1 else 0 end as converted,
          count(*) filter (where event = 'session_end' and {_WINDOW_SQL}) as sessions,
          avg(latency) filter (where event = 'latency_sample' and {_WINDOW_SQL}) as latency
        from events
        group by 1
        """
    ).to_pyarrow()


def multiplicity_panel_frame(summary: pa.Table) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in summary.to_pylist():
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "date": dt.date(2025, 1, 10),
                "revenue": 0.0,
                "sessions": 0,
                "converted": 0,
                "latency": 0.0,
            }
        )
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "date": dt.date(2025, 1, 11),
                "revenue": r["revenue"] or 0.0,
                "sessions": r["sessions"] or 0,
                "converted": r["converted"],
                "latency": r["latency"] or 0.0,
            }
        )
    return pd.DataFrame(rows)


# -- Non-positive-arm-mean dataset, 2 arms: secondary mean `refunds` has no
# treatment-arm `refund` events (a `missing="zero"` literal zero mean), the
# additive-only hazard engine.py's `_nonpositive_mean_additive_row` covers.
# `revenue` (primary) and `converted` stay positive in both arms.

_NONPOSITIVE_MEAN_ARMS = ("control", "treatment")


def nonpositive_mean_event_rows(*, n_per_arm: int = 60) -> list[dict[str, Any]]:
    """`refund` reuses `_row`'s `revenue` column via an event name distinct
    from `purchase` -- control units carry ordinary positive refund
    amounts, treatment units carry none at all (not a zero-valued event:
    an absent one, so the aggregate is a real NULL resolved to 0 by
    `missing="zero"`)."""
    rows: list[dict[str, Any]] = []
    for arm in _NONPOSITIVE_MEAN_ARMS:
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            rows.append(_row(uid, _EXPOSURE_AT, "exposure", group_id=arm, experiment_id="exp"))
            rows.append(
                _row(
                    uid,
                    _PURCHASE_AT,
                    "purchase",
                    revenue=5.0 + (arm == "treatment") + (i % 5) * 0.1,
                )
            )
            if arm == "control":
                rows.append(_row(uid, _PURCHASE_AT, "refund", revenue=1.0 + (i % 3) * 0.1))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "purchase", revenue=0.0))
            rows.append(_row(uid, _FRESHNESS_PAD_AT, "refund", revenue=0.0))
    return rows


def nonpositive_mean_definitions_dict(*, plan: AnalysisPlan) -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "refund", "column": "revenue"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "increase",
            },
            {
                "type": "mean",
                "name": "refunds",
                "entity": "user_id",
                "fact": "refund",
                "aggregation": "sum",
                "window_days": 1,
                "preferred_direction": "decrease",
            },
            {
                "type": "conversion",
                "name": "converted",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 1,
                "preferred_direction": "increase",
            },
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2025-01-10",
                "end": _EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "plan": plan.model_dump(mode="json"),
            }
        ],
    }


def nonpositive_mean_summary_frame(con: ibis.BaseBackend) -> pa.Table:
    return con.sql(  # ty: ignore[unresolved-attribute]
        f"""
        select
          user_id,
          any_value(group_id) filter (where event = 'exposure') as variant,
          sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) as revenue,
          case when sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) > 0
               then 1 else 0 end as converted,
          sum(revenue) filter (where event = 'refund' and {_WINDOW_SQL}) as refunds
        from events
        group by 1
        """
    ).to_pyarrow()


def nonpositive_mean_panel_frame(summary: pa.Table) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in summary.to_pylist():
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "date": dt.date(2025, 1, 10),
                "revenue": 0.0,
                "converted": 0,
                "refunds": 0.0,
            }
        )
        rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["variant"],
                "date": dt.date(2025, 1, 11),
                "revenue": r["revenue"] or 0.0,
                "converted": r["converted"],
                "refunds": r["refunds"] or 0.0,
            }
        )
    return pd.DataFrame(rows)


# -- Clustered signed-ratio dataset: store-randomized `net_revenue` (`purchase`
# sum / `session_end` count) totals negative in every control store and positive
# in every treatment store. Arm-prefixed store ids keep clusters single-arm for
# `_validate_cluster_labels`; 20 per arm sits on the no-advisory boundary.

_CLUSTERED_NEGATIVE_MEAN_ARMS = ("control", "treatment")
_CLUSTERED_NEGATIVE_MEAN_CLUSTERS_PER_ARM = 20
_CLUSTERED_NEGATIVE_MEAN_UNITS_PER_CLUSTER = 3


def clustered_negative_mean_event_rows(*, zero_treatment: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in _CLUSTERED_NEGATIVE_MEAN_ARMS:
        sign = -1.0 if arm == "control" else 1.0
        if zero_treatment and arm == "treatment":
            sign = 0.0
        for k in range(_CLUSTERED_NEGATIVE_MEAN_CLUSTERS_PER_ARM):
            cluster_id = f"{arm}_c{k}"
            for u in range(_CLUSTERED_NEGATIVE_MEAN_UNITS_PER_CLUSTER):
                uid = f"{arm}{k}_{u}"
                rows.append(
                    _row(
                        uid,
                        _EXPOSURE_AT,
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                        store_id=cluster_id,
                    )
                )
                rows.append(
                    _row(uid, _PURCHASE_AT, "purchase", revenue=sign * (10.0 + (k % 5) + (u % 3)))
                )
                rows.append(_row(uid, _SESSION_AT, "session_end", sess=1))
                rows.append(_row(uid, _FRESHNESS_PAD_AT, "purchase", revenue=0.0))
                rows.append(_row(uid, _FRESHNESS_PAD_AT, "session_end", sess=0))
    return rows


def clustered_negative_mean_definitions_dict(*, plan: AnalysisPlan) -> dict[str, Any]:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "session_end", "column": "sess"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": "ratio",
                "name": "net_revenue",
                "entity": "user_id",
                "preferred_direction": "increase",
                "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
                "denominator": {"fact": "session_end", "aggregation": "count", "window_days": 1},
            },
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "cluster": "store_id",
                "start": "2025-01-10",
                "end": _EXPERIMENT_END.isoformat(),
                "control_group": "control",
                "plan": plan.model_dump(mode="json"),
            }
        ],
    }


def clustered_negative_mean_summary_frame(con: ibis.BaseBackend) -> pa.Table:
    return con.sql(  # ty: ignore[unresolved-attribute]
        f"""
        select
          user_id,
          any_value(group_id) filter (where event = 'exposure') as variant,
          any_value(store_id) filter (where event = 'exposure') as cluster_id,
          sum(revenue) filter (where event = 'purchase' and {_WINDOW_SQL}) as revenue,
          sum(sess) filter (where event = 'session_end' and {_WINDOW_SQL}) as sessions
        from events
        group by 1
        """
    ).to_pyarrow()
