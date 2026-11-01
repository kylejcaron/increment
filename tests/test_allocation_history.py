"""workflows.allocation_history(): the daily/cumulative assigned-unit
enrollment timeline used by the reusable dashboard's allocation view.

Sourced from the native warehouse path's own first-exposure assignment
(``query.builders.daily_exposure_counts`` over ``_get_exposures()``) --
never from a metric's sample counts, which mature on a different
schedule than enrollment. Refused before any query on a seam-family
instance (no raw event stream) or a declared cluster (a unit-level
timeline would misrepresent cluster-grain SRM).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

import ibis
import pytest

from increment import from_definitions, from_frame, from_switchback_frame
from increment.errors import CapabilityError
from increment.estimation.diagnostics import SRMResult
from increment.workflows import allocation_history, srm
from tests.analysis_factory import make_binding

_DEFS_TEMPLATE = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM ah_events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - name: exposure
        column: null
exposures:
  - name: enrolled
    fact: exposure
experiments:
  - name: ah_test
    exposure: enrolled
    unit: user_id
{cluster_line}
{day_boundary_line}
    start: 2025-08-01
    end: 2025-08-10
    plan: {{}}
    control_group: control
"""


def _row(
    unit_id: str,
    ts: dt.datetime,
    group_id: str | None,
    *,
    store_id: str = "",
) -> dict[str, Any]:
    return {
        "user_id": unit_id,
        "event_at": ts,
        "event": "exposure",
        "experiment_id": "ah_test",
        "group_id": group_id,
        "store_id": store_id,
    }


def _main_fixture_rows() -> list[dict[str, Any]]:
    """3 control + 2 treatment on day 1; 2 more control (quiet treatment)
    on day 2; one mixed-assignment unit; one unassigned unit."""
    day1 = dt.datetime(2025, 8, 1, 9, 0, 0)
    day2 = dt.datetime(2025, 8, 2, 9, 0, 0)
    rows = [
        _row("c0", day1, "control"),
        _row("c1", day1, "control"),
        _row("c2", day1, "control"),
        _row("t0", day1, "treatment"),
        _row("t1", day1, "treatment"),
        _row("c3", day2, "control"),
        _row("c4", day2, "control"),
        # Mixed assignment: same unit enrolled into both arms.
        _row("dup1", day1, "control"),
        _row("dup1", day1, "treatment"),
        # Unassigned: exposure fired with no group label.
        _row("unassigned1", day1, None),
    ]
    return rows


def _analysis(
    tmp_path,
    rows: list[dict[str, Any]],
    *,
    cluster: bool = False,
    day_boundary: str | None = None,
    on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
    suffix: str = "",
) -> Any:
    con = ibis.duckdb.connect()
    con.create_table("ah_events", obj=rows)
    cluster_line = "    cluster: store_id" if cluster else ""
    day_boundary_line = f"    day_boundary: {day_boundary!r}" if day_boundary else ""
    defs_path = tmp_path / f"defs{suffix}.yaml"
    defs_path.write_text(
        _DEFS_TEMPLATE.format(cluster_line=cluster_line, day_boundary_line=day_boundary_line)
    )
    return from_definitions(
        defs_path,
        con,
        experiment_name="ah_test",
        store="none",
        on_mixed_assignment=on_mixed_assignment,
    )


@pytest.mark.integration
def test_allocation_history_excludes_mixed_and_unassigned_units(tmp_path):
    """A unit enrolled into two arms, and a unit with no assignment label,
    contribute to neither arm's daily count -- allocation reads off the
    valid, single-arm assignment only, matching what SRM's own
    ``assignment_counts`` would report."""
    experiment, resolver = _analysis(tmp_path, _main_fixture_rows(), on_mixed_assignment="exclude")
    history = allocation_history(experiment, resolver)
    day1_control = [
        r
        for r in history.to_pylist()
        if r["ds"] == dt.date(2025, 8, 1) and r["group_id"] == "control"
    ][0]
    day1_treatment = [
        r
        for r in history.to_pylist()
        if r["ds"] == dt.date(2025, 8, 1) and r["group_id"] == "treatment"
    ][0]
    # 3 real control units (c0-c2), not 4 (dup1 excluded).
    assert day1_control["n_daily"] == 3
    # 2 real treatment units (t0-t1), not 3 (dup1 excluded); unassigned1
    # never appears under any group.
    assert day1_treatment["n_daily"] == 2


@pytest.mark.integration
def test_allocation_history_quiet_arm_gets_explicit_zero_day_row(tmp_path):
    """Treatment enrolls no one on day 2 (only control does); the quiet
    arm still gets an explicit ``n_daily=0`` row with ``n_cumulative``
    carried forward, rather than being silently absent."""
    experiment, resolver = _analysis(tmp_path, _main_fixture_rows(), on_mixed_assignment="exclude")
    rows = {
        (r["ds"], r["group_id"]): r for r in allocation_history(experiment, resolver).to_pylist()
    }
    day2_treatment = rows[(dt.date(2025, 8, 2), "treatment")]
    assert day2_treatment["n_daily"] == 0
    assert day2_treatment["n_cumulative"] == 2  # carried forward from day 1

    day2_control = rows[(dt.date(2025, 8, 2), "control")]
    assert day2_control["n_daily"] == 2
    assert day2_control["n_cumulative"] == 5


@pytest.mark.integration
def test_allocation_history_schema_and_sort_order(tmp_path):
    experiment, resolver = _analysis(tmp_path, _main_fixture_rows(), on_mixed_assignment="exclude")
    history = allocation_history(experiment, resolver)
    assert set(history.column_names) == {
        "experiment_id",
        "analysis_population",
        "ds",
        "group_id",
        "n_daily",
        "n_cumulative",
    }
    rows = history.to_pylist()
    keys = [(r["analysis_population"], r["ds"], r["group_id"]) for r in rows]
    assert keys == sorted(keys)
    assert {r["analysis_population"] for r in rows} == {"assigned"}
    assert {r["experiment_id"] for r in rows} == {"ah_test"}


@pytest.mark.integration
def test_allocation_history_final_counts_reconcile_with_assigned_srm(tmp_path):
    """Each arm's final-day ``n_cumulative`` matches ``srm()``'s own
    assigned-population observed counts -- both read off the same
    excluded-mixed/unassigned assignment, never a metric's sample size."""
    experiment, resolver = _analysis(tmp_path, _main_fixture_rows(), on_mixed_assignment="exclude")
    history = allocation_history(experiment, resolver).to_pylist()
    last_day = max(r["ds"] for r in history)
    final_counts = {r["group_id"]: r["n_cumulative"] for r in history if r["ds"] == last_day}

    srm_result = srm(
        experiment, resolver, expected={"control": 0.5, "treatment": 0.5}, inference="fixed"
    )
    assert isinstance(srm_result, SRMResult)
    assert final_counts == srm_result.observed


@pytest.mark.integration
def test_allocation_history_respects_declared_day_boundary_shift(tmp_path):
    """An exposure at 2025-08-02 03:10 UTC buckets to local day 2025-08-01
    under ``day_boundary='UTC-05:00'`` -- the same localization
    ``daily_exposure_counts`` applies directly, not a UTC calendar cast."""
    rows = [
        _row("u_shift", dt.datetime(2025, 8, 2, 3, 10, 0), "control"),
        _row("u_other", dt.datetime(2025, 8, 5, 9, 0, 0), "treatment"),
    ]
    experiment, resolver = _analysis(
        tmp_path, rows, day_boundary="UTC-05:00", on_mixed_assignment="error", suffix="_shift"
    )
    history = allocation_history(experiment, resolver).to_pylist()
    # Densification cross-joins every observed date x group, so a group
    # also gets an explicit zero row on the other's real exposure day --
    # only the row carrying real enrollment (n_daily > 0) proves the
    # localized bucketing.
    control_enrolled = {r["ds"] for r in history if r["group_id"] == "control" and r["n_daily"]}
    assert control_enrolled == {dt.date(2025, 8, 1)}
    treatment_enrolled = {r["ds"] for r in history if r["group_id"] == "treatment" and r["n_daily"]}
    assert treatment_enrolled == {dt.date(2025, 8, 5)}


@pytest.mark.integration
def test_allocation_history_refuses_a_declared_cluster(tmp_path):
    """A cluster-randomized experiment refuses before any query: a
    unit-level daily timeline would misrepresent SRM, which for this
    experiment runs over cluster-grain allocation, not units."""
    rows = [
        _row("c0", dt.datetime(2025, 8, 1, 9, 0, 0), "control", store_id="s0"),
        _row("t0", dt.datetime(2025, 8, 1, 9, 0, 0), "treatment", store_id="s1"),
    ]
    experiment, resolver = _analysis(tmp_path, rows, cluster=True, suffix="_cluster")
    with pytest.raises(CapabilityError) as caught:
        allocation_history(experiment, resolver)
    assert caught.value.code == "evidence.operation_unavailable"


def test_allocation_history_refuses_a_seam_family_instance():
    """A frame/warehouse/moments-backed analysis retains no raw exposure
    event stream to build a daily enrollment timeline from."""
    from tests.capability_consumers_auxiliary import assert_refusal

    experiment, resolver = make_binding()
    with pytest.raises(CapabilityError) as caught:
        allocation_history(experiment, resolver)
    assert caught.value.code == "evidence.operation_unavailable"
    assert_refusal(
        "triggered_auxiliary_reads", "allocation_moments_refused", caught.value, stage="read"
    )


@pytest.mark.parametrize(
    "ingress",
    ("from_unit_summary", "from_unit_panel", "from_switchback_panel", "from_moments"),
)
def test_allocation_history_refuses_non_native_ingresses(ingress):
    """Only the native event-backed source retains the allocation timeline."""
    import pandas as pd
    import pyarrow as pa

    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation
    from tests.capability_consumers_auxiliary import assert_refusal

    if ingress == "from_unit_summary":
        source = from_frame(
            pd.DataFrame({"unit": ["c", "t"], "arm": ["control", "treatment"], "value": [0, 1]}),
            shape="summary",
            unit="unit",
            group="arm",
            metrics={"value": "mean"},
            control="control",
        )
    elif ingress == "from_unit_panel":
        source = from_frame(
            pa.table(
                {
                    "unit": ["c", "t"],
                    "arm": ["control", "treatment"],
                    "day": ["2025-01-01", "2025-01-01"],
                    "value": [0, 1],
                }
            ),
            shape="panel",
            unit="unit",
            group="arm",
            date="day",
            metrics={"value": "mean"},
            control="control",
        )
    elif ingress == "from_switchback_panel":
        source = from_switchback_frame(
            pa.Table.from_pylist(
                [
                    {
                        "unit": unit,
                        "cycle": cycle,
                        "period": period,
                        "step": step,
                        "group": group,
                        "value": float(cycle + period + step),
                    }
                    for unit, order in (
                        ("u1", ("control", "treatment")),
                        ("u2", ("treatment", "control")),
                    )
                    for cycle in range(2)
                    for period, group in enumerate(order)
                    for step in range(2)
                ]
            ),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"value": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
            contrast_references={"value": UnitCycleTApproximation()},
        )
    else:
        source = make_binding()

    outcome_id = (
        "allocation_moments_refused"
        if ingress == "from_moments"
        else f"allocation_{ingress}_refused"
    )
    try:
        with pytest.raises(CapabilityError) as refused:
            allocation_history(*source)
        expected_code = (
            "evidence.assignment.identity_unavailable"
            if ingress in {"from_unit_summary", "from_unit_panel", "from_switchback_panel"}
            else "evidence.operation_unavailable"
        )
        assert refused.value.code == expected_code
        assert_refusal("triggered_auxiliary_reads", outcome_id, refused.value, stage="read")
    finally:
        source[1].close()
