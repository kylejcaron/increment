"""Capture the readout baseline used by the shipped example.

The baseline covers all six public readouts, including daily, as-of, and
breakout views. Regenerate it only after explaining every changed value:
these numbers are the example's expected answers, not routine snapshots.

The script pins DuckDB to one thread so repeated captures are byte-stable.
Keep the fixture aligned with the current public semantics and update the
fixture only when the corresponding behavior change is intentional."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import ibis

from examples._seed import seed_event_log
from increment import Analysis, Method

OUT = Path("tests/fixtures/parity_baseline.json")
DEFINITIONS = "examples/definitions"
EXPERIMENT = "new_onboarding_v2"


def _arm_rows(results):
    """Arm lift rows; `run` also admits switchback contrasts, which have no arm."""
    from increment.results import LiftEstimate

    rows = []
    for row in results:
        if not isinstance(row, LiftEstimate):
            raise SystemExit("expected arm lift rows; this experiment returned contrasts")
        rows.append(row)
    return rows


def _day(value) -> str:
    """A day label; `ds` is a date for native sources and a plain label otherwise."""
    return value.isoformat() if isinstance(value, date) else str(value)


def _value_of(row) -> float | None:
    value = row.value
    if value is None:
        assert row.unavailable is not None
        return None
    return value.value


def _lift_value(row) -> float | None:
    lift = row.lift
    if lift is None:
        if getattr(row, "reference_kind", None) == "binomial":
            assert row.binomial_set is not None and not row.binomial_set.point_available
            return None
        if hasattr(row, "excluded"):
            assert row.excluded is not None
        else:
            assert row.unavailable is not None
        return None
    return lift.value


def main() -> None:
    # Pin DuckDB to one thread: parallel reductions can reorder floating-point
    # sums and change the fixture at ULP scale. The ibis version rejects the
    # equivalent config={"threads": 1} form.
    con = ibis.duckdb.connect(threads=1)
    seed_event_log(con)
    # Analysis loads the YAML from definitions_path and one shared instance
    # serves all six readouts. Its common spine bound keeps results independent
    # of readout order and of whether later calls materialize temporary tables.
    a = Analysis(EXPERIMENT, DEFINITIONS, con)

    baseline: dict[str, dict[str, float | None]] = {}

    # 1. Whole-window lift, per (metric, arm).
    baseline["run"] = {
        f"{e.metric}|{e.group_id}": _lift_value(e)
        for e in _arm_rows(a.run(decision_method=Method(name="unadjusted")))
    }

    # 2. Per-day absolute values, per (metric, arm, ds). The spine's right edge
    #    determines how many days exist at all, so this is the readout most
    #    sensitive to a change in the spine's bounds. DailyMetricValue.value
    #    is an Estimate, not a float -- .value.value is the point estimate.
    baseline["run_daily"] = {
        f"{v.metric}|{v.group_id}|{_day(v.ds)}": _value_of(v) for v in a.run_daily()
    }

    # 3. Cumulative-to-date values -- catches a spine change that shifts the
    #    cumulation window without changing any single day's value.
    baseline["run_asof"] = {
        f"{v.metric}|{v.group_id}|{_day(v.ds)}": _value_of(v) for v in a.run_asof()
    }

    # 4. Per-segment lift, which re-derives unit_totals with a dimension join.
    baseline["run_breakout"] = {
        f"{e.metric}|{e.group_id}|{e.dimension}={e.dimension_value}": _lift_value(e)
        for e in a.run_breakout(decision_method=Method(name="unadjusted"))
    }

    # 5. Per-day relative lift -- the estimand key matters: an encouragement
    #    or estimands change that adds/drops rows shows as a key-set diff.
    baseline["run_daily_lift"] = {
        f"{e.metric}|{e.group_id}|{_day(e.ds)}|{e.estimand}": _lift_value(e)
        for e in a.run_daily_lift(decision_method=Method(name="unadjusted"))
    }

    # 6. Cumulative-to-date relative lift.
    baseline["run_asof_lift"] = {
        f"{e.metric}|{e.group_id}|{_day(e.ds)}|{e.estimand}": _lift_value(e)
        for e in a.run_asof_lift(decision_method=Method(name="unadjusted"))
    }

    empty = [k for k, v in baseline.items() if not v]
    if empty:
        raise SystemExit(f"captured nothing for {empty} -- seed or experiment name is wrong")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    for readout, rows in baseline.items():
        print(f"{readout}: {len(rows)} values")


if __name__ == "__main__":
    main()
