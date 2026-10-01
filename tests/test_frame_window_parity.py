"""from_unit_panel must reproduce builders.unit_totals.
Both pipelines consume the SAME synthetic event log. The ibis side runs
real definitions-path builders over ibis.memtable (DuckDB in-memory); the
frame side collapses events to a one-row-per-unit-day panel and runs
from_unit_panel. Group-level additive moments must match to 1e-9 with
identical surviving unit sets. The censoring case moving 60 enrolled units
to a strict subset is what proves the harness discriminates. Covers
windowed mean/conversion/ratio, retention, two censoring cases (declared
observation_end, running-experiment fallback), and a "clicked" indicator
event stream for conversion.
Day-boundary note: every fixture is day-grain/default-UTC, so
``Experiment.day_boundary`` moves nothing here. A non-UTC boundary has NO
frame-side lever: the CALLER must pre-bucket ``ds``/``exposure_date`` with
the SAME fixed offset the warehouse side declares - otherwise the
substrates answer differently-bucketed questions and parity is undefined."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import numpy as np
import pytest

from increment.errors import IncrementWarning, InvalidRequestError
from increment.frame import MetricSpec, from_unit_panel
from increment.semantics.models import (
    AnalysisPlan,
    MeanMetric,
    Metric,
    RetentionMetric,
)
from tests.warning_codes import warning_codes

START = dt.datetime(2025, 1, 1)
N_UNITS = 60
HORIZON_DAYS = 14
ENROLL_SPREAD = 5  # staggered enrollment across 5 days


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (MeanMetric(name="m", entity="unit_id", fact="m", window_days=7), (7, 6)),
        (
            RetentionMetric(name="retained", entity="unit_id", fact="page_view", threshold_days=3),
            (3, 3),
        ),
        (
            RetentionMetric(
                name="retained_bounded",
                entity="unit_id",
                fact="page_view",
                threshold_days=(3, 10),
            ),
            (10, 9),
        ),
    ],
)
def test_maturity_arithmetic_is_shared(metric: Metric, expected: tuple[int, int]) -> None:
    """Pin the maturity offsets the frame and builder pipelines censor by."""
    from increment._window import final_maturity_day, maturity_days

    assert (maturity_days(metric), final_maturity_day(metric)) == expected


@dataclass(frozen=True)
class Case:
    id: str
    spec: MetricSpec
    observation_end: dt.date | None = None  # frame-side param / ibis observation_end
    declare_end: bool = False  # give the ibis Experiment an end?


def _event_log() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """(exposures, events): every unit enrolled; four independently
    generated per-unit-day fact streams keyed "m" (mean/ratio-numerator/
    retention source), "den" (ratio denominator, unused by non-ratio
    cases), "clicked" (conversion's binary daily indicator, unused by
    non-conversion cases), and "big" (a large-mean/tiny-CV stream, one
    day-0 event per unit at ~1e9 +- 1, unused by the other cases). Each
    added stream draws from its OWN rng stream (a distinct seed) so its
    addition never perturbs "m"/"den"'s draw order, and neither "clicked"
    nor "big" widens the (unit, day) key set _panel_rows builds - "big"
    fires only on day 0, a key every unit already contributes.
    Fixed seeds -> deterministic across repeated calls, so the ibis side
    and the frame side genuinely consume identical data despite each
    calling this function independently.
    """
    rng = np.random.default_rng(7)
    click_rng = np.random.default_rng(70)
    big_rng = np.random.default_rng(700)
    exposures: list[dict[str, Any]] = []
    events: dict[str, list[dict[str, Any]]] = {"m": [], "den": [], "clicked": [], "big": []}
    for i in range(N_UNITS):
        unit = f"u{i:03d}"
        group = "treatment" if i % 2 else "control"
        fe = START + dt.timedelta(days=int(rng.integers(0, ENROLL_SPREAD)), hours=1)
        exposures.append(
            {
                "unit_id": unit,
                "experiment_id": "parity",
                "group_id": group,
                "first_exposure_ts": fe,
            }
        )
        # One day-0 fact per unit at ~1e9 +- 1: mu~1e9 with CV~1e-9, the regime
        # where raw-sum second moments carry no recoverable variance signal.
        events["big"].append(
            {
                "unit_id": unit,
                "ts": fe + dt.timedelta(hours=9),
                "metric": "big",
                "value": 1e9 + float(big_rng.normal(0.0, 1.0)),
            }
        )
        for d in range(HORIZON_DAYS):
            if rng.random() < 0.6:  # ~60% active days
                events["m"].append(
                    {
                        "unit_id": unit,
                        "ts": fe + dt.timedelta(days=d, hours=3),
                        "metric": "m",
                        "value": float(rng.uniform(0.5, 5.0)),
                    }
                )
            if rng.random() < 0.6:
                events["den"].append(
                    {
                        "unit_id": unit,
                        "ts": fe + dt.timedelta(days=d, hours=5),
                        "metric": "den",
                        "value": float(rng.uniform(0.5, 5.0)),
                    }
                )
            if click_rng.random() < 0.4:
                events["clicked"].append(
                    {
                        "unit_id": unit,
                        "ts": fe + dt.timedelta(days=d, hours=7),
                        "metric": "clicked",
                        "value": 1.0,
                    }
                )
    # Denominator event dated strictly past every numerator event: ibis's default
    # right edge is the numerator max, so union_event_horizon must span both streams.
    max_m_ts = max(ev["ts"] for ev in events["m"])
    events["den"].append(
        {
            "unit_id": "u001",
            "ts": max_m_ts + dt.timedelta(days=2, hours=5),
            "metric": "den",
            "value": 3.0,
        }
    )
    return exposures, events


def _daily_sums(events: list[dict[str, Any]]) -> dict[tuple[str, dt.date], float]:
    per_day: dict[tuple[str, dt.date], float] = {}
    for ev in events:
        key = (ev["unit_id"], ev["ts"].date())
        per_day[key] = per_day.get(key, 0.0) + ev["value"]
    return per_day


def _panel_rows(
    exposures: list[dict[str, Any]], events: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Collapse events to one row per (unit, day); every unit gets at least
    its own day-0 (zero-valued, if inactive) row so both sides agree on the
    enrolled set - ``from_unit_panel``'s densification only knows about a
    unit if it appears at least once in the input frame. "clicked" is a
    0/1 daily indicator (1 if any clicked event fired that day), not a sum
    - conversion's semantic contract is "any occurrence", never a count.
    """
    fe_by_unit = {e["unit_id"]: e for e in exposures}
    m_by_day = _daily_sums(events["m"])
    den_by_day = _daily_sums(events["den"])
    clicked_by_day = dict.fromkeys(_daily_sums(events["clicked"]), 1.0)
    big_by_day = _daily_sums(events["big"])

    keys = set(m_by_day) | set(den_by_day) | set(clicked_by_day)
    for u, fe in fe_by_unit.items():
        keys.add((u, fe["first_exposure_ts"].date()))
    max_day = max(day for _, day in keys)
    for unit in fe_by_unit:
        for offset in range((max_day - START.date()).days + 1):
            keys.add((unit, START.date() + dt.timedelta(days=offset)))

    rows = []
    for u, day in sorted(keys):
        fe = fe_by_unit[u]
        rows.append(
            {
                "user_id": u,
                "variant": fe["group_id"],
                "day": day,
                "exposed_on": fe["first_exposure_ts"].date(),
                "m": m_by_day.get((u, day), 0.0),
                "den": den_by_day.get((u, day), 0.0),
                "clicked": clicked_by_day.get((u, day), 0.0),
                "big": big_by_day.get((u, day), 0.0),
            }
        )
    return rows


def _to_backend(rows: list[dict[str, Any]], backend: str) -> Any:
    if backend == "pandas":
        import pandas as pd

        return pd.DataFrame(rows)
    if backend == "polars":
        import polars as pl

        return pl.DataFrame(rows)
    if backend == "pyarrow":
        import pyarrow as pa

        return pa.Table.from_pylist(rows)
    raise ValueError(f"unknown backend {backend!r}")


@cache
def ibis_moments(case: Case) -> dict[str, dict[str, float]]:
    """The ibis/DuckDB ground truth for *case* - backend-independent, so
    ``functools.cache`` keys on *case* (a frozen, hashable dataclass over
    a frozen ``MetricSpec``) computes it once and reuses it across all 3
    ``backend`` parametrizations of ``test_frame_matches_builders``,
    instead of tripling this DuckDB/ibis work per case.
    """
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")

    import pandas as pd

    from increment.frame import synthesise_metric
    from increment.query.builders import (
        group_summary,
        post_exposure_stats,
        union_event_horizon,
        unit_day_spine_stats,
        unit_totals,
    )
    from increment.semantics.models import Experiment

    exposures, events = _event_log()
    end: dt.datetime | None = None
    if case.declare_end:
        assert case.observation_end is not None
        end = dt.datetime.combine(case.observation_end, dt.time())
    exp = Experiment(
        name="parity",
        exposure="exposed",
        unit="user",
        start=START,
        end=end,
        control_group="control",
        plan=AnalysisPlan(),
    )
    con = ibis.duckdb.connect()
    exp_t = ibis.memtable(pd.DataFrame(exposures))
    # y_column resolves to the numerator ("m") for a ratio spec too, so
    # this one lookup picks the right source stream for every case.
    source_key = case.spec.y_column
    ev_t = ibis.memtable(pd.DataFrame(events[source_key]))
    metric = synthesise_metric(case.spec)

    den_stats = None
    end_date = None
    if case.spec.type == "ratio":
        den_ev_t = ibis.memtable(pd.DataFrame(events["den"]))
        den_stats = post_exposure_stats(den_ev_t, exp_t, source_key="den")
        # A ratio is never single-stream: the spine's right edge must cover the
        # denominator's own event extent too, or a late den event silently vanishes.
        end_date = union_event_horizon([ev_t, den_ev_t], exp)
    spine, stats = unit_day_spine_stats(exp_t, ev_t, exp, source_key, end_date=end_date)

    totals = unit_totals(spine, stats, metric, exp, den_stats=den_stats, warn_on_censoring=False)
    df = con.to_pyarrow(group_summary(totals)).to_pylist()

    out: dict[str, dict[str, float]] = {}
    for r in df:
        entry = {
            "n": int(r["n"]),
            "ref_y": float(r["ref_y"]),
            "cy1": float(r["cy1"]),
            "cy2": float(r["cy2"]),
        }
        if case.spec.type == "ratio":
            entry["ref_den"] = float(r["ref_den"])
            entry["cden1"] = float(r["cden1"])
            entry["cden2"] = float(r["cden2"])
            entry["cyden"] = float(r["cyden"])
        out[r["group_id"]] = entry
    return out


def frame_moments(case: Case, backend: str) -> dict[str, dict[str, float]]:
    exposures, events = _event_log()
    rows = _panel_rows(exposures, events)
    frame = _to_backend(rows, backend)
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[case.spec],
        exposure_date="exposed_on",
        observation_end=case.observation_end,
    )
    metric = src.context.metrics[0]

    out: dict[str, dict[str, float]] = {}
    for r in src.moments(metric):
        entry = {"n": r["n"], "ref_y": r["ref_y"], "cy1": r["cy1"], "cy2": r["cy2"]}
        if case.spec.type == "ratio":
            entry["ref_den"] = r["ref_den"]
            entry["cden1"] = r["cden1"]
            entry["cden2"] = r["cden2"]
            entry["cyden"] = r["cyden"]
        out[r["group_id"]] = entry
    return out


def _asof_rows_by_day(
    rows: list[dict[str, Any]], *, ratio: bool
) -> dict[tuple[dt.date, str], dict[str, float]]:
    out: dict[tuple[dt.date, str], dict[str, float]] = {}
    for row in rows:
        ds = row["ds"]
        if isinstance(ds, dt.datetime):
            ds = ds.date()
        entry = {
            "n": int(row["n"]),
            "ref_y": float(row["ref_y"]),
            "cy1": float(row["cy1"]),
            "cy2": float(row["cy2"]),
        }
        if ratio:
            entry |= {
                "ref_den": float(row["ref_den"]),
                "cden1": float(row["cden1"]),
                "cden2": float(row["cden2"]),
                "cyden": float(row["cyden"]),
            }
        out[(ds, row["group_id"])] = entry
    return out


@cache
def ibis_asof_moments(
    case: Case, *, completed_windows_only: bool = False
) -> dict[tuple[dt.date, str], dict[str, float]]:
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")

    import pandas as pd

    from increment.frame import synthesise_metric
    from increment.query.builders import asof_group_summary, union_event_horizon, unit_day_panel
    from increment.semantics.models import Experiment

    exposures, events = _event_log()
    exp = Experiment(
        name="parity",
        exposure="exposed",
        unit="user",
        start=START,
        end=None,
        control_group="control",
        plan=AnalysisPlan(),
    )
    con = ibis.duckdb.connect()
    exp_t = ibis.memtable(pd.DataFrame(exposures))
    metric = synthesise_metric(case.spec)
    event_tables = {
        name: ibis.memtable(pd.DataFrame(events[name])) for name in ("m", "den", "clicked")
    }
    ev_t = event_tables[case.spec.y_column]
    # The frame fixture is densified over the union of its three event columns,
    # so its ibis reference must carry the same horizon even if a metric's stream stops sooner.
    end_date = union_event_horizon(list(event_tables.values()), exp)
    den_panel = None
    if case.spec.type == "ratio":
        den_panel = unit_day_panel(
            exp_t, event_tables["den"], exp, metric_name=case.spec.name, end_date=end_date
        )
    panel = unit_day_panel(exp_t, ev_t, exp, metric_name=case.spec.name, end_date=end_date)
    summary = asof_group_summary(
        panel, metric, den_panel=den_panel, completed_windows_only=completed_windows_only
    )
    return _asof_rows_by_day(con.to_pyarrow(summary).to_pylist(), ratio=case.spec.type == "ratio")


def frame_asof_moments(
    case: Case, backend: str, *, completed_windows_only: bool = False
) -> dict[tuple[dt.date, str], dict[str, float]]:
    exposures, events = _event_log()
    src = from_unit_panel(
        _to_backend(_panel_rows(exposures, events), backend),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[case.spec],
        exposure_date="exposed_on",
    )
    metric = src.context.metrics[0]
    return _asof_rows_by_day(
        src.moments(metric, grain="asof", completed_windows_only=completed_windows_only),
        ratio=case.spec.type == "ratio",
    )


PARITY_CASES = [
    Case(id="mean-unwindowed", spec=MetricSpec(name="m", type="mean")),
    Case(
        id="ratio-unwindowed",
        spec=MetricSpec(name="m_ratio", type="ratio", numerator="m", denominator="den"),
    ),
    Case(id="mean-window-7", spec=MetricSpec(name="m", window_days=7)),
    Case(
        id="conversion-window-3",
        spec=MetricSpec(name="conv", type="conversion", value_column="clicked", window_days=3),
    ),
    Case(
        id="ratio-window-7",
        spec=MetricSpec(
            name="m_ratio", type="ratio", numerator="m", denominator="den", window_days=7
        ),
    ),
    Case(
        id="retention-unbounded-7",
        spec=MetricSpec(name="d7", type="retention", value_column="m", threshold_days=7),
    ),
    Case(
        id="retention-band-7-14",
        spec=MetricSpec(name="d7b", type="retention", value_column="m", threshold_days=(7, 14)),
    ),
    # the discriminating case: declared end censors late enrollees on BOTH sides
    Case(
        id="mean-window-7-censored",
        spec=MetricSpec(name="m", window_days=7),
        observation_end=(START + dt.timedelta(days=8)).date(),
        declare_end=True,
    ),
    # running-experiment fallback: frame resolves the end date per spec's OWN fact
    # stream; this fixture's "m" column is populated everywhere so it can't distinguish that from the panel-wide date.
    Case(id="mean-window-7-running", spec=MetricSpec(name="m", window_days=7)),
    # mu ~ 1e9, CV ~ 1e-9: the regime the centered wire format exists for. Omitting
    # a between-group term or re-forming a raw sum is invisible at mu~1 elsewhere.
    Case(id="mean-large-mu", spec=MetricSpec(name="big")),
]

# c1-class columns: sum(v - ref_v), ~0 by construction, carrying only the last
# bits of n*ref_v - so scale the absolute tolerance to n*|ref_v| instead.
CANCELLING_REFS = {"cy1": "ref_y", "cden1": "ref_den"}


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda c: c.id)
def test_frame_matches_builders(case: Case, backend: str) -> None:
    expected = ibis_moments(case)
    if case.id == "mean-window-7-censored":
        with pytest.warns(IncrementWarning) as rec:
            actual = frame_moments(case, backend)
        assert "frame.censoring.dropped_units" in warning_codes(rec)
    else:
        actual = frame_moments(case, backend)
    assert actual.keys() == expected.keys()
    if case.id == "mean-window-7-censored":
        # An unwindowed bug reads all 60 enrolled units; the declared
        # observation_end must have actually dropped some of them.
        total_n = sum(m["n"] for m in actual.values())
        assert 0 < total_n < N_UNITS, (
            f"{case.id}: expected a strict subset of {N_UNITS} enrolled units, got {total_n}"
        )
    for group, exp_m in expected.items():
        act_m = actual[group]
        assert act_m["n"] == exp_m["n"], f"{case.id}/{group}: unit sets diverge"
        for col, exp_val in exp_m.items():
            if col == "n":
                continue
            if col in CANCELLING_REFS:
                scale = 1e-9 * max(1.0, exp_m["n"] * abs(exp_m[CANCELLING_REFS[col]]))
                assert act_m[col] == pytest.approx(exp_val, abs=scale), f"{case.id}/{group}: {col}"
                continue
            assert act_m[col] == pytest.approx(exp_val, rel=1e-9), f"{case.id}/{group}: {col}"


ASOF_CASES = [
    Case(id="mean-unwindowed", spec=MetricSpec(name="m", type="mean")),
    Case(id="mean-window-7", spec=MetricSpec(name="m", type="mean", window_days=7)),
    Case(
        id="conversion-window-3",
        spec=MetricSpec(name="conv", type="conversion", value_column="clicked", window_days=3),
    ),
    Case(
        id="ratio-window-7",
        spec=MetricSpec(
            name="m_ratio", type="ratio", numerator="m", denominator="den", window_days=7
        ),
    ),
    Case(
        id="retention-unbounded-7",
        spec=MetricSpec(name="d7", type="retention", value_column="m", threshold_days=7),
    ),
    Case(
        id="retention-band-7-14",
        spec=MetricSpec(name="d7b", type="retention", value_column="m", threshold_days=(7, 14)),
    ),
]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("case", ASOF_CASES, ids=lambda c: c.id)
def test_frame_asof_matches_builders(case: Case, backend: str) -> None:
    expected = ibis_asof_moments(case)
    actual = frame_asof_moments(case, backend)

    assert actual.keys() == expected.keys()
    for key, exp_m in expected.items():
        act_m = actual[key]
        assert act_m["n"] == exp_m["n"], f"{case.id}/{key}: unit sets diverge"
        for col, exp_val in exp_m.items():
            if col == "n":
                continue
            if col in CANCELLING_REFS:
                scale = 1e-9 * max(1.0, exp_m["n"] * abs(exp_m[CANCELLING_REFS[col]]))
                assert act_m[col] == pytest.approx(exp_val, abs=scale), f"{case.id}/{key}: {col}"
                continue
            assert act_m[col] == pytest.approx(exp_val, rel=1e-9), f"{case.id}/{key}: {col}"


def test_frame_asof_bounded_retention_completed_windows_match_builder() -> None:
    bounded = Case(
        id="retention-band-7-14",
        spec=MetricSpec(name="d7b", type="retention", value_column="m", threshold_days=(7, 14)),
    )
    expected = ibis_asof_moments(bounded, completed_windows_only=True)
    actual = frame_asof_moments(bounded, "pyarrow", completed_windows_only=True)
    provisional = frame_asof_moments(bounded, "pyarrow")

    assert actual.keys() == expected.keys()
    assert min(day for day, _ in actual) >= START.date() + dt.timedelta(days=14)
    for key, exp_m in expected.items():
        act_m = actual[key]
        assert act_m["n"] == exp_m["n"], f"{bounded.id}/{key}: unit sets diverge"
        for col, exp_val in exp_m.items():
            if col == "n":
                continue
            if col in CANCELLING_REFS:
                scale = 1e-9 * max(1.0, exp_m["n"] * abs(exp_m[CANCELLING_REFS[col]]))
                assert act_m[col] == pytest.approx(exp_val, abs=scale), f"{bounded.id}/{key}: {col}"
                continue
            assert act_m[col] == pytest.approx(exp_val, rel=1e-9), f"{bounded.id}/{key}: {col}"

    final_day = max(day for day, _ in provisional)
    assert max(day for day, _ in actual) == final_day
    for group_id in ("control", "treatment"):
        assert actual[(final_day, group_id)] == pytest.approx(provisional[(final_day, group_id)])

    unbounded = Case(
        id="retention-unbounded-7",
        spec=MetricSpec(name="d7", type="retention", value_column="m", threshold_days=7),
    )
    with pytest.raises(InvalidRequestError) as builder_refusal:
        ibis_asof_moments(unbounded, completed_windows_only=True)
    assert builder_refusal.value.code == "query.builders.asof_group_summary"
    with pytest.raises(InvalidRequestError) as frame_refusal:
        frame_asof_moments(unbounded, "pyarrow", completed_windows_only=True)
    assert frame_refusal.value.code == "frame.frame_panel.asof_moments_completed"


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_completed_window_gate_matches_frame_and_warehouse(backend: str) -> None:
    """D4's completion gate (Tasks 3/6) must agree between the warehouse and
    frame substrates on the exact fixture each task's own regression uses:
    four units, c1/t1 exposed Jan 1 and c2/t2 exposed Jan 3, a 3-day-window
    mean observed Jan 1-4. Provisional admits every unit on every day it has
    an observation; completed_windows_only admits only the unit(s) whose own
    window has closed by that date -- Jan 4, one unit per arm, cumulative
    outcome 3 (three days of revenue=1.0 each). This is a cross-source check,
    not merely each substrate agreeing with itself."""
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")

    import pyarrow as pa

    from increment.query.builders import asof_group_summary
    from increment.semantics.models import MeanMetric

    group = {"c1": "control", "c2": "control", "t1": "treatment", "t2": "treatment"}
    exposure = {
        "c1": dt.date(2026, 1, 1),
        "c2": dt.date(2026, 1, 3),
        "t1": dt.date(2026, 1, 1),
        "t2": dt.date(2026, 1, 3),
    }
    days = [dt.date(2026, 1, d) for d in (1, 2, 3, 4)]

    con = ibis.duckdb.connect()
    warehouse_rows = [
        {
            "unit_id": unit,
            "experiment_id": "e",
            "group_id": group[unit],
            "metric": "revenue",
            "ds": day,
            "n_events": 1,
            "sum_value": 1.0,
            "min_value": 1.0,
            "max_value": 1.0,
            "first_exposure_ts": dt.datetime.combine(exposure[unit], dt.time()),
            "first_exposure_date": exposure[unit],
        }
        for unit in group
        for day in days
        if day >= exposure[unit]
    ]
    panel = con.create_table(f"completion_gate_panel_{backend}", obj=warehouse_rows)
    warehouse_metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", window_days=3)
    warehouse_provisional = _asof_rows_by_day(
        con.to_pyarrow(asof_group_summary(panel, warehouse_metric)).to_pylist(), ratio=False
    )
    warehouse_decision = _asof_rows_by_day(
        con.to_pyarrow(
            asof_group_summary(panel, warehouse_metric, completed_windows_only=True)
        ).to_pylist(),
        ratio=False,
    )

    frame_rows = [
        {
            "user_id": unit,
            "variant": group[unit],
            "ds": day,
            "exposed_on": exposure[unit],
            "revenue": 1.0,
        }
        for unit in group
        for day in days
        if day >= exposure[unit]
    ]
    table = (
        pa.Table.from_pylist(frame_rows)
        if backend == "pyarrow"
        else _to_backend(frame_rows, backend)
    )
    source = from_unit_panel(
        table,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        exposure_date="exposed_on",
        metrics=[MetricSpec(name="revenue", window_days=3)],
    )
    frame_metric = source.context.metrics[0]
    frame_provisional = _asof_rows_by_day(source.moments(frame_metric, grain="asof"), ratio=False)
    frame_decision = _asof_rows_by_day(
        source.moments(frame_metric, grain="asof", completed_windows_only=True), ratio=False
    )

    # provisional: every exposed unit, every day, identically on both substrates.
    # c1/t1 (exposed Jan 1) accrue revenue=1.0/day and freeze when their 3-day
    # window closes (day_idx=3, Jan 4); c2/t2 join on Jan 3 and keep accruing.
    expected_provisional = {
        dt.date(2026, 1, 1): (1, 1.0),
        dt.date(2026, 1, 2): (1, 2.0),
        dt.date(2026, 1, 3): (2, 2.0),
        dt.date(2026, 1, 4): (2, 2.5),
    }
    expected_provisional_keys = {(day, arm) for day in days for arm in ("control", "treatment")}
    assert warehouse_provisional.keys() == expected_provisional_keys
    assert frame_provisional.keys() == expected_provisional_keys
    for key, exp_m in warehouse_provisional.items():
        ds, _group = key
        expected_n, expected_ref_y = expected_provisional[ds]
        assert exp_m["n"] == expected_n, f"provisional {key}: n"
        assert exp_m["ref_y"] == pytest.approx(expected_ref_y), f"provisional {key}: ref_y"
        assert frame_provisional[key] == pytest.approx(exp_m), f"provisional {key}"

    # completed: only Jan 4 survives, one unit per arm, cumulative outcome 3 -- on both substrates.
    expected_decision_keys = {
        (dt.date(2026, 1, 4), "control"),
        (dt.date(2026, 1, 4), "treatment"),
    }
    assert warehouse_decision.keys() == expected_decision_keys
    assert frame_decision.keys() == expected_decision_keys
    for key, exp_m in warehouse_decision.items():
        assert exp_m["n"] == 1
        assert exp_m["ref_y"] == pytest.approx(3.0)
        assert frame_decision[key] == pytest.approx(exp_m), f"completed {key}"


COMPLETED_WINDOW_CASES = [
    Case(id="mean-unwindowed", spec=MetricSpec(name="m", type="mean")),
    Case(id="mean-window-7", spec=MetricSpec(name="m", type="mean", window_days=7)),
]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("case", COMPLETED_WINDOW_CASES, ids=lambda c: c.id)
def test_asof_completed_windows_only_no_op_and_gate_match_across_backends(
    case: Case, backend: str
) -> None:
    """completed_windows_only=True is a no-op for an unbounded mean (D4) and
    a real completion boundary for a bounded one, identically on the
    warehouse and frame substrates. Retention's own (band-based) completion
    boundary is covered separately by
    test_frame_asof_bounded_retention_completed_windows_match_builder, which
    also proves retention refuses completed_windows_only when unbounded --
    a distinct rule from a plain windowed mean's no-op."""
    provisional_expected = ibis_asof_moments(case)
    decision_expected = ibis_asof_moments(case, completed_windows_only=True)
    decision_actual = frame_asof_moments(case, backend, completed_windows_only=True)

    if case.spec.window_days is None:
        assert decision_expected.keys() == provisional_expected.keys(), (
            f"{case.id}: unbounded mean must stay a no-op under completed_windows_only"
        )
    else:
        assert decision_expected.keys() < provisional_expected.keys(), (
            f"{case.id}: bounded mean must gate some provisional rows away"
        )

    assert decision_actual.keys() == decision_expected.keys()
    for key, exp_m in decision_expected.items():
        assert decision_actual[key] == pytest.approx(exp_m), f"{case.id}/{key}"


def test_frame_exposure_gate_matches_builders_at_every_supported_grain() -> None:
    ibis = pytest.importorskip("ibis")

    import pandas as pd

    from increment.query.builders import (
        asof_group_summary,
        daily_group_summary,
        first_exposures,
        group_summary,
        metric_events,
        unit_day_panel,
        unit_day_spine_stats,
        unit_totals,
    )
    from increment.semantics.models import Experiment, MeanMetric

    start = dt.datetime(2026, 1, 1)
    experiment = Experiment(
        name="exposure_gate",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=1),
        control_group="control",
        exposure="exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
    exposure_events = ibis.memtable(
        pd.DataFrame(
            [
                {
                    "unit_id": "c1",
                    "ts": start,
                    "event": "exposure",
                    "experiment_id": experiment.name,
                    "group_id": "control",
                },
                {
                    "unit_id": "t1",
                    "ts": start + dt.timedelta(days=1),
                    "event": "exposure",
                    "experiment_id": experiment.name,
                    "group_id": "treatment",
                },
            ]
        )
    )
    purchase_events = ibis.memtable(
        pd.DataFrame(
            [
                {
                    "unit_id": "c1",
                    "ts": start + dt.timedelta(hours=1),
                    "event": "purchase",
                    "value": 1.0,
                },
                {
                    "unit_id": "c1",
                    "ts": start + dt.timedelta(days=1, hours=1),
                    "event": "purchase",
                    "value": 1.0,
                },
                {
                    "unit_id": "t1",
                    "ts": start + dt.timedelta(hours=1),
                    "event": "purchase",
                    "value": 9.0,
                },
                {
                    "unit_id": "t1",
                    "ts": start + dt.timedelta(days=1, hours=1),
                    "event": "purchase",
                    "value": 2.0,
                },
            ]
        )
    )
    con = ibis.duckdb.connect()
    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(purchase_events, metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    expected = {
        "total": con.to_pyarrow(
            group_summary(unit_totals(spine, stats, metric, experiment, warn_on_censoring=False))
        ).to_pylist(),
        "daily": con.to_pyarrow(daily_group_summary(panel, metric=metric)).to_pylist(),
        "asof": con.to_pyarrow(asof_group_summary(panel, metric)).to_pylist(),
    }

    rows = [
        ("c1", "control", start.date(), start.date(), 1.0),
        ("c1", "control", (start + dt.timedelta(days=1)).date(), start.date(), 1.0),
        ("t1", "treatment", start.date(), (start + dt.timedelta(days=1)).date(), 9.0),
        (
            "t1",
            "treatment",
            (start + dt.timedelta(days=1)).date(),
            (start + dt.timedelta(days=1)).date(),
            2.0,
        ),
    ]
    frame = _to_backend(
        [
            dict(zip(["user_id", "variant", "day", "exposed_on", "revenue"], row, strict=True))
            for row in rows
        ],
        "pyarrow",
    )
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        exposure_date="exposed_on",
    )
    frame_rows = {
        grain: src.moments(src.context.metrics[0], grain=cast(Any, grain)) for grain in expected
    }
    for grain, query_rows in expected.items():
        query_values = {(row.get("ds"), row["group_id"]): _first_moment(row) for row in query_rows}
        actual_values = {
            (row.get("ds"), row["group_id"]): _first_moment(row) for row in frame_rows[grain]
        }
        assert actual_values == pytest.approx(query_values), grain

    final_day = max(row["ds"] for row in frame_rows["asof"])
    final_asof = {
        row["group_id"]: _first_moment(row) for row in frame_rows["asof"] if row["ds"] == final_day
    }
    total = {row["group_id"]: _first_moment(row) for row in frame_rows["total"]}
    assert final_asof == pytest.approx(total)


def _first_moment(row: Any) -> float:
    """Recover sum(y) from a format-2 moments row: n*ref_y + cy1."""
    return row["n"] * row["ref_y"] + row["cy1"]


def test_frame_windowed_total_excludes_pre_exposure_rows() -> None:
    """The windowed total-grain reduce
    bounded only the band's RIGHT edge (``day_idx < right_edge``), never
    day 0's LEFT edge, so a genuine pre-exposure panel row (``day_idx <
    0``) survived into a windowed mean/ratio sum or conversion flag.

    The shared ``_event_log`` fixture above can never exercise this:
    every event it generates lands at ``fe + d`` for ``d >= 0``, so a
    pre-exposure row is always the panel's zero-fill default and
    structurally invisible to this bug. This fixture deliberately puts a
    genuine, non-zero event BEFORE one unit's own ``exposure_date``.

    The ibis reference needs no explicit left-bound filter to exclude
    it: ``panel_spine`` builds its dense spine as ``first_exposure_date
    + day_offset`` for ``day_offset >= 0`` by construction, and
    ``post_exposure_stats``/``unit_day_spine_stats`` additionally
    require ``events.ts > first_exposure_ts`` - so a pre-exposure event
    structurally cannot reach it, windowed or not. That makes ibis's
    windowed total the ground truth this asserts the frame path
    against, in addition to a hand-computed expected sum.
    """
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")

    import pandas as pd

    from increment.frame import synthesise_metric
    from increment.query.builders import group_summary, unit_day_spine_stats, unit_totals
    from increment.semantics.models import Experiment

    base = dt.datetime(2025, 1, 1)
    spec = MetricSpec(name="m", window_days=7)

    exposures = [
        # u1 is exposed on day 2 - its day-0 event below is a genuine
        # PRE-exposure row (day_idx == -2).
        {
            "unit_id": "u1",
            "experiment_id": "leak",
            "group_id": "control",
            "first_exposure_ts": base + dt.timedelta(days=2, hours=1),
        },
        {
            "unit_id": "u2",
            "experiment_id": "leak",
            "group_id": "treatment",
            "first_exposure_ts": base,
        },
    ]
    events = [
        # u1: a genuine pre-exposure event (day_idx=-2) a right-edge-only filter
        # wrongly admits, plus two real in-window events (day_idx 0 and 2).
        {"unit_id": "u1", "ts": base + dt.timedelta(hours=3), "metric": "m", "value": 100.0},
        {
            "unit_id": "u1",
            "ts": base + dt.timedelta(days=2, hours=3),
            "metric": "m",
            "value": 5.0,
        },
        {
            "unit_id": "u1",
            "ts": base + dt.timedelta(days=4, hours=3),
            "metric": "m",
            "value": 5.0,
        },
        # u2: one in-window event, one past the right edge, already correctly
        # excluded before this fix - proves the fix leaves that behavior untouched.
        {"unit_id": "u2", "ts": base + dt.timedelta(hours=3), "metric": "m", "value": 3.0},
        {
            "unit_id": "u2",
            "ts": base + dt.timedelta(days=10, hours=3),
            "metric": "m",
            "value": 4.0,
        },
    ]

    metric = synthesise_metric(spec)
    exp = Experiment(
        name="leak",
        exposure="exposed",
        unit="user",
        start=base,
        end=None,
        control_group="control",
        plan=AnalysisPlan(),
    )
    exp_t = ibis.memtable(pd.DataFrame(exposures))
    ev_t = ibis.memtable(pd.DataFrame(events))
    spine, stats = unit_day_spine_stats(exp_t, ev_t, exp, "m")
    totals = unit_totals(spine, stats, metric, exp, warn_on_censoring=False)
    con = ibis.duckdb.connect()
    ibis_by_group = {r["group_id"]: r for r in con.to_pyarrow(group_summary(totals)).to_pylist()}

    # Ground truth: u1's windowed sum excludes the pre-exposure 100.0, leaving
    # 5.0 + 5.0 = 10.0; u2's 3.0 is unaffected. Wire carries n*ref_y + cy1.
    assert _first_moment(ibis_by_group["control"]) == pytest.approx(10.0)
    assert _first_moment(ibis_by_group["treatment"]) == pytest.approx(3.0)

    # Frame side: one row per (unit, day); every unit needs at least its
    # own day-0 row so densification agrees on the enrolled set.
    panel_rows = [
        {
            "user_id": "u1",
            "variant": "control",
            "day": base.date(),
            "exposed_on": (base + dt.timedelta(days=2)).date(),
            "m": 100.0,
        },
        {
            "user_id": "u1",
            "variant": "control",
            "day": (base + dt.timedelta(days=2)).date(),
            "exposed_on": (base + dt.timedelta(days=2)).date(),
            "m": 5.0,
        },
        {
            "user_id": "u1",
            "variant": "control",
            "day": (base + dt.timedelta(days=4)).date(),
            "exposed_on": (base + dt.timedelta(days=2)).date(),
            "m": 5.0,
        },
        {
            "user_id": "u2",
            "variant": "treatment",
            "day": base.date(),
            "exposed_on": base.date(),
            "m": 3.0,
        },
        {
            "user_id": "u2",
            "variant": "treatment",
            "day": (base + dt.timedelta(days=10)).date(),
            "exposed_on": base.date(),
            "m": 4.0,
        },
    ]
    frame = pd.DataFrame(panel_rows)
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[spec],
        exposure_date="exposed_on",
    )
    frame_metric = src.context.metrics[0]
    frame_by_group = {r["group_id"]: r for r in src.moments(frame_metric)}

    assert _first_moment(frame_by_group["control"]) == pytest.approx(10.0), (
        "windowed total must exclude u1's genuine pre-exposure event (100.0)"
    )
    assert _first_moment(frame_by_group["treatment"]) == pytest.approx(3.0)
    assert _first_moment(frame_by_group["control"]) == pytest.approx(
        _first_moment(ibis_by_group["control"])
    )
    assert _first_moment(frame_by_group["treatment"]) == pytest.approx(
        _first_moment(ibis_by_group["treatment"])
    )


def test_natural_day_axis_controls_inferred_maturity() -> None:
    """Natural day labels use their semantic integer order for maturity."""
    import polars as pl

    rows = [
        {
            "unit": f"{arm}{i}",
            "arm": arm,
            "day": day,
            "exposure": "d0",
            "y": float(i + 1),
        }
        for arm in ("c", "t")
        for i in range(3)
        for day in ("d0", "d9", "d10")
    ]
    spec = MetricSpec(name="y", window_days=11)

    def readout(frame: Any, *, observation_end: str | None) -> tuple[Any, Any]:
        source = from_unit_panel(
            frame,
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposure",
            observation_end=observation_end,
            control="c",
            metrics=[spec],
        )
        metric = source.context.metrics[0]
        total = {row["group_id"]: row for row in source.moments(metric)}
        asof = sorted(
            (row for row in source.moments(metric, grain="asof") if row["ds"] == "d10"),
            key=lambda row: row["group_id"],
        )
        return total, asof

    explicit_total, explicit_asof = readout(pl.DataFrame(rows), observation_end="d10")
    for permutation in (rows, list(reversed(rows))):
        inferred_total, inferred_asof = readout(pl.DataFrame(permutation), observation_end=None)
        assert inferred_total == explicit_total
        assert inferred_asof == explicit_asof
        assert {group: row["n"] for group, row in inferred_total.items()} == {"c": 3, "t": 3}
