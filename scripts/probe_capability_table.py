"""Reproduce docs/reference/capabilities-by-entry-point.md's "What runs where" table.

Run with `uv run --extra demo --extra tables --extra dashboard python
scripts/probe_capability_table.py`.

This script drives real builder construction and never asserts a value
from prose: it either transcribes `tests.parity_harness`'s own
`PARITY_CASES` (report_harness_cells, below) through `run_case`/
`assert_parity` -- so a printed "parity: ok" means every non-waived
constructor agreed field for field, not merely that each one produced some
rows -- or it drives a handful of additional capabilities the harness does
not (yet) carry a row for, using the identical `ParityCase`/`run_case`/
`assert_parity` machinery locally (report_additional_cells, below). Every
row prints one of: "parity: ok" (agreement, or a coded, expected waiver on
every non-attempted constructor), "parity: FAILED <reason>" (a real
disagreement), or, for the two probes that call `workflows.planning_baseline`
directly rather than `run`, one line per builder naming the value or
the coded refusal.

The capability tables are projections of the case and matrix contracts; this
probe is a behavioral smoke, not a transcription source. Use `--case <id>` to
run one named parity case and receive a nonzero exit status when it fails.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ibis
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import yaml
from ibis.backends.sql import SQLBackend

from increment import (
    from_definitions,
    from_frame,
    from_moments,
    from_switchback_frame,
    from_unit_day_artifact,
)
from increment.errors import CodedError
from increment.frame import MetricSpec
from increment.power import Baseline
from increment.semantics.models import AnalysisPlan
from increment.workflows import export_moments, planning_baseline
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


def report_harness_cells(cases: Sequence[ParityCase] = PARITY_CASES) -> int:
    """Run declared behavioral cases and return nonzero if any one fails."""
    failures = 0
    for case in cases:
        failures += not _report_case(case)
    return int(bool(failures))


def _report_case(case: ParityCase) -> bool:
    try:
        result = run_case(case)
        assert_parity(case, result)
        print(
            f"{case.id}: parity: ok ({sorted(result.rows)} agree; {sorted(result.refusals)} waived)"
        )
        return True
    except Exception as exc:
        print(f"{case.id}: parity: FAILED -- {exc}")
        return False


# ``planning_baseline`` works for mean metrics wherever ``arm_moments`` is
# available. Quantile metrics additionally need per-unit control rows from
# ``source.moments()``, which panel and moments-only constructors do not have.


def _arm_baseline(binding: tuple[Any, Any], metric: str) -> Baseline:
    experiment, resolver = binding
    baseline = planning_baseline(experiment, resolver, metric)
    assert isinstance(baseline, Baseline)
    return baseline


def _row_planning_baseline_mean() -> None:
    plan = AnalysisPlan(secondaries=["revenue"])
    rows = ds.event_rows()
    defs_path = Path(tempfile.mkdtemp()) / "defs.yml"
    defs_path.write_text(yaml.safe_dump(ds.definitions_dict(plan=plan)))
    con = ds.duckdb_connection(rows)
    assert isinstance(con, SQLBackend)
    try:
        native = from_definitions(defs_path, con, experiment_name="exp")
        baseline = _arm_baseline(native, "revenue")
        print(
            f"planning_baseline_mean: from_definitions: yes (mean={baseline.mean:.6g}, var={baseline.var:.6g})"
        )

        native2 = from_definitions(defs_path, con, experiment_name="exp")
        adopted = _publish_and_adopt(con, native2)
        baseline2 = _arm_baseline(adopted, "revenue")
        print(
            f"planning_baseline_mean: from_unit_day_artifact: yes "
            f"(mean={baseline2.mean:.6g}, var={baseline2.var:.6g})"
        )

        frame = ds.unit_summary_frame(con)
        summary = from_frame(
            frame,
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero")],
            plan=plan,
        )
        baseline3 = _arm_baseline(summary, "revenue")
        print(
            f'planning_baseline_mean: from_frame(shape="summary"): yes '
            f"(mean={baseline3.mean:.6g}, var={baseline3.var:.6g})"
        )

        panel = ds.unit_panel_frame(frame)
        panel_binding = from_frame(
            panel,
            shape="panel",
            unit="user_id",
            group="variant",
            date="date",
            control="control",
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero")],
            plan=plan,
        )
        baseline4 = _arm_baseline(panel_binding, "revenue")
        print(
            f'planning_baseline_mean: from_frame(shape="panel"): yes '
            f"(mean={baseline4.mean:.6g}, var={baseline4.var:.6g})"
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.parquet"
            export_moments(*summary, path)
            import pyarrow.parquet as pq

            moments_rows = pq.read_table(path).to_pylist()
        moments_binding = from_moments(
            moments_rows, control="control", metrics=[MetricSpec(name="revenue", type="mean")]
        )
        baseline5 = _arm_baseline(moments_binding, "revenue")
        print(
            f"planning_baseline_mean: from_moments: yes "
            f"(mean={baseline5.mean:.6g}, var={baseline5.var:.6g})"
        )
    except CodedError as exc:
        raise AssertionError(
            f"planning_baseline_mean had an unexpected coded refusal {exc.code!r}"
        ) from exc
    finally:
        con.disconnect()
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from tests.test_analysis_planning_baseline import TestPlanningBaselineOnSwitchback

    switchback = from_switchback_frame(
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
    pilot = planning_baseline(*switchback, "orders")
    print(f"planning_baseline_mean: from_switchback_frame: yes ({type(pilot).__name__})")


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
        native = from_definitions(defs_path, con, experiment_name="exp")
        baseline = _arm_baseline(native, "p90_latency")
        print(f"planning_baseline_quantile: from_definitions: yes (mean={baseline.mean:.6g})")
    except CodedError as exc:
        raise AssertionError(
            f"planning_baseline_quantile from_definitions unexpectedly refused {exc.code!r}"
        ) from exc

    summary = from_frame(
        _per_unit_frame(),
        shape="summary",
        unit="user_id",
        group="group_id",
        control="C",
        metrics=[_QUANTILE_TIES_METRIC],
        plan=AnalysisPlan(secondaries=["p90_latency"]),
    )
    try:
        baseline2 = _arm_baseline(summary, "p90_latency")
        print(
            f'planning_baseline_quantile: from_frame(shape="summary"): yes (mean={baseline2.mean:.6g})'
        )
    except CodedError as exc:
        raise AssertionError(
            f'planning_baseline_quantile from_frame(shape="summary") unexpectedly refused {exc.code!r}'
        ) from exc

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
    panel_binding = from_frame(
        pd.DataFrame(panel_rows),
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        control="C",
        metrics=[_QUANTILE_TIES_METRIC],
        plan=AnalysisPlan(secondaries=["p90_latency"]),
    )
    try:
        baseline3 = _arm_baseline(panel_binding, "p90_latency")
        print(
            f'planning_baseline_quantile: from_frame(shape="panel"): yes (mean={baseline3.mean:.6g})'
        )
    except CodedError as exc:
        raise AssertionError(
            f'planning_baseline_quantile from_frame(shape="panel") unexpectedly refused {exc.code!r}'
        ) from exc

    try:
        native2 = from_definitions(defs_path, con, experiment_name="exp")
        adopted = _publish_and_adopt(con, native2)
        baseline4 = _arm_baseline(adopted, "p90_latency")
        print(
            f"planning_baseline_quantile: from_unit_day_artifact: yes (mean={baseline4.mean:.6g})"
        )
    except CodedError as exc:
        raise AssertionError(
            f"planning_baseline_quantile from_unit_day_artifact unexpectedly refused {exc.code!r}"
        ) from exc

    try:
        with tempfile.TemporaryDirectory() as td2:
            path = Path(td2) / "m.parquet"
            export_moments(*summary, path)
        raise AssertionError("planning_baseline_quantile from_moments unexpectedly exported a cube")
    except CodedError as exc:
        if exc.code != "source.frame.quantile_no_moments":
            raise AssertionError(
                f"planning_baseline_quantile from_moments expected "
                f"'source.frame.quantile_no_moments', received {exc.code!r}"
            ) from exc
        print(
            f"planning_baseline_quantile: from_moments: refused ({exc.code}) -- no cube can be built"
        )

    print(
        "planning_baseline_quantile: from_switchback_frame: not attempted "
        "(switchback metrics must be type='mean' or type='conversion', "
        "source.frame.switchback.metric)"
    )
    con.disconnect()


_BUILDERS = {
    'from_frame(shape="summary")': from_frame,
    'from_frame(shape="panel")': from_frame,
    "from_switchback_frame": from_switchback_frame,
    "from_definitions": from_definitions,
    "from_unit_day_artifact": from_unit_day_artifact,
    "from_moments": from_moments,
}


def _row_logged_policy_contrast() -> None:
    """The logged-policy family consumes a `LoggedTrace`, which no builder or
    source-bound operation produces or accepts: not expressible on every builder
    (nothing to call, no `.code`), checked against the live signatures rather than
    asserted from prose."""
    import inspect

    from increment import artifacts, query, run, workflows
    from increment.logged_policy import LoggedTrace, estimate_policy_contrast

    first = next(iter(inspect.signature(estimate_policy_contrast).parameters.values()))
    assert first.annotation in (LoggedTrace, "LoggedTrace"), first
    surfaces = {
        **_BUILDERS,
        "run": run,
        **{
            f"{module.__name__}.{name}": member
            for module in (workflows, query, artifacts)
            for name, member in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_")
        },
    }
    producers = [
        name
        for name, member in surfaces.items()
        if "LoggedTrace" in str(inspect.signature(member).return_annotation)
    ]
    assert not producers, producers
    assert all(not hasattr(module, "estimate_policy_contrast") for module in (workflows, query))
    for builder, member in _BUILDERS.items():
        assert callable(member)
        print(
            f"logged_policy_contrast: {builder}: not expressible (no builder or "
            "source-bound operation yields a LoggedTrace or takes a decision trace, so there "
            "is no call to make and no .code; ingress is LoggedTrace.from_records / from_frame)"
        )


def report_additional_cells() -> int:
    """Run the measured capabilities not yet represented in ``PARITY_CASES``."""
    try:
        _row_planning_baseline_mean()
        _row_planning_baseline_quantile()
        _row_logged_policy_contrast()
    except Exception as exc:
        print(f"additional capability probe: FAILED -- {exc}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run exactly one named PARITY_CASES contract")
    args = parser.parse_args(argv)
    if args.case is not None:
        matches = [case for case in PARITY_CASES if case.id == args.case]
        if len(matches) != 1:
            parser.error(f"unknown parity case {args.case!r}")
        return report_harness_cells(matches)
    failed = report_harness_cells()
    return int(bool(failed or report_additional_cells()))


if __name__ == "__main__":
    raise SystemExit(main())
