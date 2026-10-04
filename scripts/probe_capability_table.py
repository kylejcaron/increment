"""Reproduce docs/reference/capabilities-by-entry-point.md's "What runs where" table.

Run with `uv run --extra demo --extra tables --extra dashboard python
scripts/probe_capability_table.py`.

This script drives real `Analysis` construction and never asserts a value
from prose: it either transcribes `tests.parity_harness`'s own
`PARITY_CASES` (report_harness_cells, below) through `run_case`/
`assert_parity` -- so a printed "parity: ok" means every non-waived
constructor agreed field for field, not merely that each one produced some
rows -- or it drives a handful of additional capabilities the harness does
not (yet) carry a row for, using the identical `ParityCase`/`run_case`/
`assert_parity` machinery locally (report_additional_cells, below). Every
row prints one of: "parity: ok" (agreement, or a coded, expected waiver on
every non-attempted constructor), "parity: FAILED <reason>" (a real
disagreement), or, for the two probes that call `Analysis.planning_baseline`
directly rather than `.run()`, one line per constructor naming the value or
the coded refusal.

Transcribe exactly what this prints into docs/reference/capabilities-by-entry-point.md's
"What runs where" table -- never a value carried over from a stale draft without
having been measured on this checkout.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import ibis
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from increment.analysis import Analysis
from increment.errors import CodedError
from increment.frame import MetricSpec
from increment.power import Baseline
from increment.semantics.models import AnalysisPlan, Definitions
from tests.analysis_factory import make_analysis
from tests.parity_harness import dataset as ds
from tests.parity_harness.cases import (
    _QUANTILE_TIES_DEFS_YAML,
    _QUANTILE_TIES_METRIC,
    PARITY_CASES,
    ParityCase,
    _publish_and_adopt,
    _quantile_ties_event_table,
)
from tests.parity_harness.runner import assert_parity, run_case


def report_harness_cells() -> None:
    """Every `PARITY_CASES` row -- calls `assert_parity`, not just
    `run_case`, so a printed "ok" means parity, not merely that every
    constructor produced SOME rows."""
    for case in PARITY_CASES:
        result = run_case(case)
        try:
            assert_parity(case, result)
            print(
                f"{case.id}: parity: ok ({sorted(result.rows)} agree; {sorted(result.refusals)} waived)"
            )
        except AssertionError as exc:
            print(f"{case.id}: parity: FAILED -- {exc}")


def _report_case(case: ParityCase) -> None:
    result = run_case(case)
    try:
        assert_parity(case, result)
        print(
            f"{case.id}: parity: ok ({sorted(result.rows)} agree; {sorted(result.refusals)} waived)"
        )
    except AssertionError as exc:
        print(f"{case.id}: parity: FAILED -- {exc}")


# ``planning_baseline`` works for mean metrics wherever ``arm_moments`` is
# available. Quantile metrics additionally need per-unit control rows from
# ``source.moments()``, which panel and moments-only constructors do not have.


def _arm_baseline(analysis: Analysis, metric: str) -> Baseline:
    baseline = analysis.planning_baseline(metric)
    assert isinstance(baseline, Baseline)
    return baseline


def _row_planning_baseline_mean() -> None:
    plan = AnalysisPlan(secondaries=["revenue"])
    rows = ds.event_rows()
    defs_dict = ds.definitions_dict(plan=plan)
    con = ds.duckdb_connection(rows)
    try:
        native = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        baseline = _arm_baseline(native, "revenue")
        print(
            f"planning_baseline_mean: from_definitions: yes (mean={baseline.mean:.6g}, var={baseline.var:.6g})"
        )

        native2 = make_analysis(con, Definitions.model_validate(defs_dict), experiment="exp")
        adopted = _publish_and_adopt(con, native2)
        baseline2 = _arm_baseline(adopted, "revenue")
        print(
            f"planning_baseline_mean: from_unit_day_artifact: yes "
            f"(mean={baseline2.mean:.6g}, var={baseline2.var:.6g})"
        )

        frame = ds.unit_summary_frame(con)
        summary = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero")],
            plan=plan,
        )
        baseline3 = _arm_baseline(summary, "revenue")
        print(
            f"planning_baseline_mean: from_unit_summary: yes "
            f"(mean={baseline3.mean:.6g}, var={baseline3.var:.6g})"
        )

        panel = ds.unit_panel_frame(frame)
        panel_analysis = Analysis.from_unit_panel(
            panel,
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero")],
            plan=plan,
        )
        baseline4 = _arm_baseline(panel_analysis, "revenue")
        print(
            f"planning_baseline_mean: from_unit_panel: yes "
            f"(mean={baseline4.mean:.6g}, var={baseline4.var:.6g})"
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.parquet"
            summary.export(path)
            import pyarrow.parquet as pq

            moments_rows = pq.read_table(path).to_pylist()
        moments_analysis = Analysis.from_moments(
            moments_rows, control="control", metrics=[MetricSpec(name="revenue", type="mean")]
        )
        baseline5 = _arm_baseline(moments_analysis, "revenue")
        print(
            f"planning_baseline_mean: from_moments: yes "
            f"(mean={baseline5.mean:.6g}, var={baseline5.var:.6g})"
        )
    except CodedError as exc:
        print(f"planning_baseline_mean: UNEXPECTED refusal {exc.code!r} -- {exc}")
    finally:
        con.disconnect()
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from tests.test_analysis_planning_baseline import TestPlanningBaselineOnSwitchback

    switchback = Analysis.from_switchback_panel(
        TestPlanningBaselineOnSwitchback._panel(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"orders": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.75),
            window=SwitchbackWindow(washout_steps=1, observation_steps=4, carryover_order=1),
        ),
    )
    pilot = switchback.planning_baseline("orders")
    print(f"planning_baseline_mean: from_switchback_panel: yes ({type(pilot).__name__})")


def _row_planning_baseline_quantile() -> None:
    table = _quantile_ties_event_table(n_per_arm=200, seed=5)

    def _per_unit_frame() -> pa.Table:
        latency_rows = table.filter(pc.field("event") == "latency")
        return latency_rows.select(["user_id", "group_id", "value"]).rename_columns(
            ["user_id", "group_id", "latency"]
        )

    con = ibis.duckdb.connect()
    con.create_table("events", table)
    td = tempfile.mkdtemp()
    defs_path = Path(td) / "defs.yml"
    defs_path.write_text(_QUANTILE_TIES_DEFS_YAML)
    try:
        native = Analysis.from_definitions("exp", defs_path, con)
        baseline = _arm_baseline(native, "p90_latency")
        print(f"planning_baseline_quantile: from_definitions: yes (mean={baseline.mean:.6g})")
    except CodedError as exc:
        print(f"planning_baseline_quantile: from_definitions: refused ({exc.code})")

    summary = Analysis.from_unit_summary(
        _per_unit_frame(),
        unit="user_id",
        group="group_id",
        control="C",
        metrics=[_QUANTILE_TIES_METRIC],
        plan=AnalysisPlan(secondaries=["p90_latency"]),
    )
    try:
        baseline2 = _arm_baseline(summary, "p90_latency")
        print(f"planning_baseline_quantile: from_unit_summary: yes (mean={baseline2.mean:.6g})")
    except CodedError as exc:
        print(f"planning_baseline_quantile: from_unit_summary: refused ({exc.code})")

    per_unit = _per_unit_frame().to_pylist()
    panel_rows = []
    for r in per_unit:
        panel_rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["group_id"],
                "date": dt.date(2024, 1, 2),
                "latency": 0.0,
            }
        )
        panel_rows.append(
            {
                "user_id": r["user_id"],
                "variant": r["group_id"],
                "date": dt.date(2024, 1, 8),
                "latency": r["latency"],
            }
        )
    panel_analysis = Analysis.from_unit_panel(
        pd.DataFrame(panel_rows),
        unit="user_id",
        group="variant",
        date="date",
        control="C",
        metrics=[_QUANTILE_TIES_METRIC],
        plan=AnalysisPlan(secondaries=["p90_latency"]),
    )
    try:
        baseline3 = _arm_baseline(panel_analysis, "p90_latency")
        print(f"planning_baseline_quantile: from_unit_panel: yes (mean={baseline3.mean:.6g})")
    except CodedError as exc:
        print(f"planning_baseline_quantile: from_unit_panel: refused ({exc.code})")

    try:
        native2 = Analysis.from_definitions("exp", defs_path, con)
        adopted = _publish_and_adopt(con, native2)
        baseline4 = _arm_baseline(adopted, "p90_latency")
        print(
            f"planning_baseline_quantile: from_unit_day_artifact: yes (mean={baseline4.mean:.6g})"
        )
    except CodedError as exc:
        print(f"planning_baseline_quantile: from_unit_day_artifact: refused ({exc.code})")

    try:
        with tempfile.TemporaryDirectory() as td2:
            path = Path(td2) / "m.parquet"
            summary.export(path)
        print("planning_baseline_quantile: from_moments: UNEXPECTED export success")
    except CodedError as exc:
        print(
            f"planning_baseline_quantile: from_moments: refused ({exc.code}) -- no cube can be built"
        )

    print(
        "planning_baseline_quantile: from_switchback_panel: not attempted "
        "(switchback metrics must be type='mean' or type='conversion', "
        "source.frame.switchback.metric)"
    )
    con.disconnect()


_CONSTRUCTORS = (
    "from_unit_summary",
    "from_unit_panel",
    "from_switchback_panel",
    "from_definitions",
    "from_unit_day_artifact",
    "from_moments",
)


def _row_logged_policy_contrast() -> None:
    """The logged-policy family consumes a `LoggedTrace`, which no `Analysis`
    surface produces: a SOURCE refusal on every constructor, checked against
    the live signatures rather than asserted from prose."""
    import inspect

    from increment.logged_policy import LoggedTrace, estimate_policy_contrast

    first = next(iter(inspect.signature(estimate_policy_contrast).parameters.values()))
    assert first.annotation in (LoggedTrace, "LoggedTrace"), first
    producers = [
        name
        for name, member in inspect.getmembers(Analysis)
        if callable(member)
        and not name.startswith("_")
        and "LoggedTrace" in str(inspect.signature(member).return_annotation)
    ]
    assert not producers, producers
    for constructor in _CONSTRUCTORS:
        assert callable(getattr(Analysis, constructor))
        print(
            f"logged_policy_contrast: {constructor}: refused (SOURCE -- no Analysis "
            "surface yields a LoggedTrace; ingress is LoggedTrace.from_records / from_frame)"
        )


def report_additional_cells() -> None:
    """Cells `PARITY_CASES` does not (yet) carry a row for. Extend this
    function, not just the printed labels, as each capability grows its
    own `ParityCase`; once a cell is covered there, delete its probe here
    and let `report_harness_cells()` report it instead -- two sources of
    truth for the same cell is itself a parity defect."""
    _row_planning_baseline_mean()
    _row_planning_baseline_quantile()
    _row_logged_policy_contrast()


if __name__ == "__main__":
    report_harness_cells()
    report_additional_cells()
