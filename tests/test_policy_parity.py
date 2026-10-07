"""Cross-path policy parity and alpha-composition ownership guards."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import Literal

import pytest

from increment._policy_alpha import resolve_cell_alpha
from increment.analysis import Analysis
from increment.decision import CompiledDecisionPlan
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec, synthesise_metric
from increment.plan import compile_decision_plan
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, MultiplicitySpec

_P = 3
_A = 2
_K = 4
_ALPHA = 0.12


def _metrics() -> list[MetricSpec]:
    return [
        MetricSpec(
            name=name,
            type="mean",
            **({"preferred_direction": "decrease"} if name == "guardrail" else {}),
        )
        for name in ("primary_a", "primary_b", "primary_c", "secondary", "guardrail")
    ]


def _declared_plan() -> AnalysisPlan:
    return AnalysisPlan(
        alpha=_ALPHA,
        primary=("primary_a", "primary_b", "primary_c"),
        guardrails=("guardrail",),
        view_multiplicity=MultiplicitySpec(correction="bonferroni"),
    )


def _compiled_plan() -> CompiledDecisionPlan:
    """The plan the frame constructor compiles from this declaration."""
    return compile_decision_plan(
        _declared_plan(),
        [synthesise_metric(spec) for spec in _metrics()],
        path="frame",
        design=Randomized(control_group="control"),
    )


def _analysis() -> Analysis:
    import pandas as pd

    metric_names = (
        "primary_a",
        "primary_b",
        "primary_c",
        "secondary",
        "guardrail",
    )
    rows: list[dict[str, object]] = []
    group_effect = {"control": 0.0, "t1": 0.02, "t2": -0.02}
    for segment_index in range(_K):
        segment = f"s{segment_index}"
        for group in ("control", "t1", "t2"):
            for replicate in range(3):
                unit = f"{segment}-{group}-{replicate}"
                for day_index, day in enumerate(("2026-01-01", "2026-01-02")):
                    row: dict[str, object] = {
                        "unit_id": unit,
                        "arm": group,
                        "day": day,
                        "segment": segment,
                    }
                    for metric_index, name in enumerate(metric_names):
                        row[name] = (
                            10.0
                            + metric_index
                            + 0.1 * segment_index
                            + 0.01 * replicate
                            + 0.005 * day_index
                            + group_effect[group]
                        )
                    rows.append(row)

    return Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="unit_id",
        group="arm",
        date="day",
        control="control",
        metrics=_metrics(),
        plan=_declared_plan(),
        breakouts=("segment",),
    )


def _effective_interval_alpha(base: float, alternative: str) -> float:
    return base if alternative == "two-sided" else 2.0 * base


def _assert_rows_match_resolver(
    rows,
    *,
    plan,
    n_segments: int,
    view: Literal["asof"] | None,
    expected_roles: set[str],
) -> None:
    checked_roles: set[str] = set()
    for row in rows:
        if row.lift is None:
            continue
        procedure = plan.procedures[row.metric]
        expected = resolve_cell_alpha(
            plan,
            procedure,
            n_arms=_A,
            n_segments=n_segments,
            view=view,
        )
        expected = _effective_interval_alpha(expected, procedure.alternative)
        assert row.require_lift().alpha == pytest.approx(expected)
        if row.role is not None:
            checked_roles.add(row.role)
    assert checked_roles >= expected_roles


def test_every_confirmatory_path_matches_the_policy_resolver():
    assert len({_P, _A, _K}) == 3, "divisors must be pairwise-distinct"
    analysis = _analysis()
    plan = _compiled_plan()

    whole = analysis.run()
    daily = analysis.run_daily_lift()
    asof = analysis.run_asof_lift(dimension="segment")

    expected_roles = {"primary", "secondary", "guardrail"}
    _assert_rows_match_resolver(
        whole, plan=plan, n_segments=1, view=None, expected_roles=expected_roles
    )
    _assert_rows_match_resolver(
        daily, plan=plan, n_segments=1, view=None, expected_roles=expected_roles
    )
    _assert_rows_match_resolver(
        asof, plan=plan, n_segments=_K, view="asof", expected_roles=expected_roles
    )
    secondary = [
        row for row in whole if isinstance(row, LiftEstimate) and row.metric == "secondary"
    ]
    assert secondary and {row.discovery for row in secondary} == {False}
    assert all(row.family_axes == ("metric", "arm") for row in secondary)

    from increment.breakout.estimates import run_daily_lift
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    warehouse_metrics = [
        MeanMetric(
            name=name,
            entity="unit_id",
            fact=name,
            **({"preferred_direction": "decrease"} if name == "guardrail" else {}),
        )
        for name in (
            "primary_a",
            "primary_b",
            "primary_c",
            "secondary",
            "guardrail",
            "unassigned",
        )
    ]
    warehouse_plan = compile_decision_plan(
        AnalysisPlan(
            alpha=_ALPHA,
            primary=("primary_a", "primary_b", "primary_c"),
            secondaries=("secondary",),
            guardrails=("guardrail",),
        ),
        warehouse_metrics,
        path="warehouse",
    )
    day = date(2026, 1, 1)
    unassigned_rows = []
    for group, mean in (("control", 10.0), ("t1", 10.1), ("t2", 9.9)):
        n = 50
        sum_y = n * mean
        unassigned_rows.append(
            centered_row_from_raw_sums(
                {
                    "ds": day,
                    "experiment_id": "parity",
                    "metric": "unassigned",
                    "group_id": group,
                    "n": float(n),
                    "sum_y": sum_y,
                    "sum_y2": 4.0 * (n - 1) + sum_y**2 / n,
                    "sum_x": None,
                    "sum_x2": None,
                    "sum_xy": None,
                    "sum_den": None,
                    "sum_den2": None,
                    "sum_yden": None,
                }
            )
        )
    unassigned_metric = next(m for m in warehouse_metrics if m.name == "unassigned")
    unassigned_results = run_daily_lift(
        unassigned_rows,
        [unassigned_metric],
        control_group="control",
        plan=warehouse_plan,
    )
    expected_unassigned = resolve_cell_alpha(
        warehouse_plan,
        warehouse_plan.procedures["unassigned"],
        n_arms=_A,
        view=None,
    )
    assert unassigned_results
    assert {row.role for row in unassigned_results} == {"unassigned"}
    assert all(
        row.lift is not None and row.require_lift().alpha == pytest.approx(expected_unassigned)
        for row in unassigned_results
    )


def test_confirmatory_consumers_delegate_alpha_composition():
    root = Path(__file__).parents[1]
    offenders: list[str] = []
    sources = [
        *sorted((root / "increment" / "readouts").glob("*.py")),
        root / "increment/analysis.py",
    ]
    for path in sources:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"_conservative_divide", "_conservative_ratio"}:
                    offenders.append(f"{relative}:{node.lineno}:{node.func.id}")
    assert offenders == [], (
        "confirmatory consumers must delegate alpha composition to "
        f"increment._policy_alpha; found {offenders}"
    )
