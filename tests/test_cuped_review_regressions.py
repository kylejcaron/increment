from __future__ import annotations

from typing import Any, cast

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec
from increment.estimation.inference import Normal
from increment.estimation.results import LiftEstimate
from increment.frame import CupedCovariate
from increment.semantics.models import ExperimentMetric


def _rows(*, unsupported: bool = False, nonpositive: bool = False) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for group in ("control", "treatment"):
        for index in range(24):
            pre = (
                "treatment-only"
                if unsupported and group == "treatment" and index == 0
                else ("a", "b", "c")[index % 3]
            )
            outcome = (10.0 if group == "control" else (-2.0 if nonpositive else 12.0)) + index % 3
            rows.append(
                {
                    "unit": f"{group}-{index}",
                    "arm": group,
                    "outcome": outcome,
                    "pre": pre,
                }
            )
    return pa.Table.from_pylist(rows)


def _metric(method: Method) -> MetricSpec:
    return MetricSpec(
        name="outcome",
        preferred_direction="decrease",
        decision_method=method,
        covariates=(CupedCovariate(name="pre", kind="categorical"),),
    )


def _analysis(table: pa.Table, *, plan: AnalysisPlan | None = None) -> Analysis:
    return Analysis.from_unit_summary(
        table,
        unit="unit",
        group="arm",
        control="control",
        metrics=[_metric(Method(name="linear", variance_reduction="cuped_linear"))],
        plan=plan,
    )


def test_failed_linear_sensitivity_does_not_break_ordinary_secondary_fcr() -> None:
    analysis = _analysis(
        _rows(unsupported=True),
        plan=AnalysisPlan(q=0.2, secondaries=["outcome"]),
    )
    result = cast(
        list[LiftEstimate],
        list(
            analysis.run(
                decision_method=Method(name="plain"),
                sensitivity_methods=[Method(name="linear", variance_reduction="cuped_linear")],
            )
        ),
    )

    by_method = {row.method: row for row in result}
    assert by_method["plain"].lift is not None
    assert by_method["linear"].lift is None
    assert by_method["linear"].failure_code == "estimation.cuped_linear.level_support"


def test_fit_wide_failure_retains_direction_margin_and_posterior_reason() -> None:
    analysis = _analysis(
        _rows(unsupported=True),
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="outcome", margin_abs=0.2)]),
    )
    (row,) = cast(list[LiftEstimate], list(analysis.run(prior=Normal(mu=0.0, sigma=0.1))))

    assert row.failure_code == "estimation.cuped_linear.level_support"
    assert row.preferred_direction == "decrease"
    assert row.null_abs == pytest.approx(0.2)
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.results.lift.t_posterior_sufficient_statistics"


def test_linear_methods_with_same_construction_share_one_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import increment.estimation.cuped_linear as cuped_linear

    original = cuped_linear.fit_cuped_linear
    calls = 0

    def counted(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cuped_linear, "fit_cuped_linear", counted)
    analysis = _analysis(_rows())
    methods = [
        Method(name="linear-decision", variance_reduction="cuped_linear"),
        Method(name="linear-sensitivity", variance_reduction="cuped_linear"),
    ]

    result = cast(
        list[LiftEstimate],
        list(analysis.run(decision_method=methods[0], sensitivity_methods=[methods[1]])),
    )

    assert {row.method for row in result} == {method.name for method in methods}
    assert calls == 1


def test_relative_projection_failure_keeps_absolute_guardrail_inference() -> None:
    analysis = _analysis(
        _rows(nonpositive=True),
        plan=AnalysisPlan(guardrails=[ExperimentMetric(metric="outcome", margin_abs=0.01)]),
    )
    (row,) = cast(list[LiftEstimate], list(analysis.run()))

    assert row.lift is None
    assert row.abs_diff is not None and row.abs_diff < 0.0
    assert row.abs_se is not None and row.abs_se > 0.0
    assert row.abs_lb is not None and row.abs_ub is not None
    assert row.abs_reference_kind == "t" and row.abs_reference_df is not None
    assert row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert row.p_value() is not None
