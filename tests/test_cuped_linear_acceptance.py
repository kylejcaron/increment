from __future__ import annotations

from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest

from increment import (
    Analysis,
    AnalysisPlan,
    InferenceSpec,
    Method,
    MetricSpec,
    UptakeSpec,
)
from increment.errors import UnsupportedRequestError
from increment.estimation.results import LiftEstimate
from increment.frame import CupedCovariate
from increment.readouts import run as run_readout
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    Randomized,
)


def _linear_metric(*, winsorization: dict[str, Any] | None = None) -> MetricSpec:
    return MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
        winsorization=winsorization,
    )


def _panel_rows(*, unsupported_segment: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in ("north", "south", "unsupported") if unsupported_segment else ("north", "south"):
        for group in ("control", "treatment"):
            for i in range(12):
                category = (
                    "rare"
                    if segment == "unsupported" and group == "treatment" and i == 0
                    else ("common" if i < 6 else "other")
                )
                rows.append(
                    {
                        "unit": f"{segment}-{group}-{i}",
                        "group": group,
                        "day": 1,
                        "segment": segment,
                        "pre": category,
                        "revenue": float(i + (group == "treatment") * (3 + (i % 3))),
                    }
                )
    return rows


def _linear_breakout_analysis(*, unsupported_segment: bool = False) -> Analysis:
    return Analysis.from_unit_panel(
        pa.Table.from_pylist(_panel_rows(unsupported_segment=unsupported_segment)),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[_linear_metric()],
        breakouts=["segment"],
    )


def test_cuped_linear_breakout_emits_complete_per_cell_family_evidence() -> None:
    analysis = _linear_breakout_analysis()

    rows = analysis.run_breakout()

    assert {(row.dimension, row.dimension_value, row.group_id) for row in rows} == {
        ("segment", segment, "treatment") for segment in ("north", "south")
    }
    assert all(row.lift is not None and row.excluded is None for row in rows)
    assert all(row.method == "linear" for row in rows)
    assert all(row.lift is not None and np.isfinite(row.lift.value) for row in rows)
    assert all(row.n_control == 12 and row.n_treat == 12 for row in rows)
    assert all(row.family_axes == ("metric", "arm", "segment") for row in rows)
    assert all(row.family_procedure == "bh" for row in rows)
    assert {row.dimension_value for row in rows} == {"north", "south"}


def test_cuped_linear_breakout_localizes_unsupported_category_cell() -> None:
    analysis = _linear_breakout_analysis(unsupported_segment=True)

    from increment.errors import IncrementWarning

    with pytest.warns(IncrementWarning, match="estimation.cuped_linear.level_support"):
        rows = analysis.run_breakout()

    by_segment = {row.dimension_value: row for row in rows}
    assert set(by_segment) == {"north", "south", "unsupported"}
    assert by_segment["north"].lift is not None
    assert by_segment["south"].lift is not None
    assert by_segment["unsupported"].lift is None
    assert by_segment["unsupported"].excluded == "estimation_failed"
    assert all(row.family_axes == ("metric", "arm", "segment") for row in rows)


class _UnreadSource:
    """Forward public source metadata while making every row acquisition observable."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.read_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def moments(self, *args: Any, **kwargs: Any) -> Any:
        self.read_calls += 1
        raise AssertionError("a rejected readout acquired source moments")

    def unit_frame(self, *args: Any, **kwargs: Any) -> Any:
        self.read_calls += 1
        raise AssertionError("a rejected readout acquired source unit rows")


def _summary_source(
    *,
    metric: MetricSpec | None = None,
    design: Any = None,
    cluster: str | None = None,
    plan: AnalysisPlan | None = None,
    uptake: str | None = None,
) -> _UnreadSource:
    table = pa.table(
        {
            "unit": [f"u{i}" for i in range(16)],
            "group": ["control"] * 8 + ["treatment"] * 8,
            "revenue": [float(i + 1) for i in range(16)],
            "pre": ["a", "b"] * 8,
            "cluster": [f"c{i // 2}" for i in range(16)],
            "uptake": [0, 1] * 8,
            "exposure_date": ["2025-01-01"] * 16,
        }
    )
    analysis = Analysis.from_unit_summary(
        table,
        unit="unit",
        group="group",
        control=None if design is not None else "control",
        design=design,
        metrics=[metric or _linear_metric()],
        cluster=cluster,
        plan=plan,
        uptake=uptake,
        exposure_date="exposure_date" if plan is not None and plan.inference is not None else None,
    )
    from tests.analysis_factory import _moment_source

    return _UnreadSource(_moment_source(analysis))


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("cluster", "estimation.engine.cuped_linear_cluster"),
        ("sequential", "estimation.engine.cuped_linear_sequential"),
        ("winsor", "estimation.engine.cuped_linear_winsorization"),
        ("observational", "estimation.engine.cuped_linear_observational"),
        ("encouragement", "estimation.engine.cuped_linear_encouragement"),
    ],
)
def test_composition_gates_refuse_before_source_acquisition(case: str, expected_code: str) -> None:
    from functools import partial

    linear = cast(Method, _linear_metric().decision_method)
    if case == "cluster":
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            _summary_source(metric=_linear_metric(), cluster="cluster")
        assert raised.value.code == "source.frame.cluster_capability"
        return
    if case == "sequential":
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            _summary_source(
                metric=_linear_metric(),
                design=Randomized(
                    control_group="control", allocation={"control": 0.5, "treatment": 0.5}
                ),
                plan=AnalysisPlan(inference=InferenceSpec(kind="asymptotic_mean")),
            )
        assert raised.value.code == "sequential.route.unsupported"
        return
    elif case == "winsor":
        source = _summary_source(metric=_linear_metric(winsorization={"upper_percentile": 0.9}))
    elif case == "observational":
        source = _summary_source(
            metric=_linear_metric(),
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("pre",))
            ),
        )
    else:
        design = Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True,
                justification="The test-only encouragement affects revenue through uptake.",
            ),
        )
        source = _summary_source(metric=_linear_metric(), design=design, uptake="uptake")
    invoke = partial(
        run_readout,
        cast(Any, source),
        decision_method=linear,
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        invoke()

    assert raised.value.code == expected_code
    assert source.read_calls == 0


def test_cuped_linear_fixed_lower_and_upper_clipping_matches_preclipped_fit() -> None:
    lower, upper = 20.0, 60.0
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": 1_000.0 if index == 29 else 12.0 + index,
            "pre": ("a", "b")[index % 2],
        }
        for group in ("control", "treatment")
        for index in range(30)
    ]
    clipped_rows = [{**row, "revenue": min(max(row["revenue"], lower), upper)} for row in rows]

    clipped = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[_linear_metric(winsorization={"lower_value": lower, "upper_value": upper})],
    )
    preclipped = Analysis.from_unit_summary(
        pa.Table.from_pylist(clipped_rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[_linear_metric()],
    )
    try:
        clipped_rows_by_arm = {
            row.group_id: row for row in cast(list[LiftEstimate], list(clipped.run()))
        }
        preclipped_rows_by_arm = {
            row.group_id: row for row in cast(list[LiftEstimate], list(preclipped.run()))
        }
        for arm, row in clipped_rows_by_arm.items():
            baseline = preclipped_rows_by_arm[arm]
            assert row.lift is not None and baseline.lift is not None
            assert row.lift.value == pytest.approx(baseline.lift.value)
            assert row.lift.log_se == pytest.approx(baseline.lift.log_se)
            assert row.dof == pytest.approx(baseline.dof)
            assert (row.lift.lb, row.lift.ub) == pytest.approx((baseline.lift.lb, baseline.lift.ub))
    finally:
        clipped.close()
        preclipped.close()


def test_cuped_linear_interactions_preserve_ingress_specific_refusals() -> None:
    from increment.errors import CapabilityError

    summary_rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": 10.0 + index,
            "pre": ("a", "b")[index % 2],
            "exposure_date": "2025-01-01",
        }
        for group in ("control", "treatment")
        for index in range(24)
    ]
    summary = Analysis.from_unit_summary(
        pa.Table.from_pylist(summary_rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[_linear_metric()],
    )
    try:
        with pytest.raises(CapabilityError) as breakout:
            summary.run_breakout()
        assert breakout.value.code == "facade.analysis.operation"
        with pytest.raises(CapabilityError) as triggered_summary:
            summary.run(population="triggered")
        assert triggered_summary.value.code == "facade.analysis.trigger_unsupported"
    finally:
        summary.close()

    panel_rows = _panel_rows()
    panel = Analysis.from_unit_panel(
        pa.Table.from_pylist(panel_rows),
        unit="unit",
        group="group",
        date="day",
        control="control",
        metrics=[_linear_metric()],
    )
    try:
        with pytest.raises(CapabilityError) as triggered_panel:
            panel.run(population="triggered")
        assert triggered_panel.value.code == "facade.analysis.trigger_unsupported"
    finally:
        panel.close()

    clustered_rows = [{**row, "cluster": row["unit"].split("-")[0]} for row in panel_rows]
    with pytest.raises(CapabilityError) as panel_cluster:
        Analysis.from_unit_panel(
            pa.Table.from_pylist(clustered_rows),
            unit="unit",
            group="group",
            date="day",
            control="control",
            metrics=[_linear_metric()],
            cluster="cluster",
        )
    assert panel_cluster.value.code == "source.frame_panel.cluster_grain"


def test_cuped_linear_nonpositive_arm_failure_preserves_positive_sibling() -> None:
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": value + (index % 3),
            "pre": ("a", "b", "c")[index % 3],
        }
        for group, value in (("control", 4.0), ("positive", 6.0), ("nonpositive", -2.0))
        for index in range(18)
    ]
    result = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                pa.Table.from_pylist(rows),
                unit="unit",
                group="group",
                control="control",
                metrics=[_linear_metric()],
            ).run()
        ),
    )

    by_arm = {row.group_id: row for row in result}
    assert by_arm["positive"].lift is not None
    assert by_arm["nonpositive"].lift is None
    assert by_arm["nonpositive"].failure_code is None
    assert by_arm["nonpositive"].relative_unavailable_reason == "nonpositive_arm_mean"
    assert by_arm["nonpositive"].abs_diff is not None


def test_cuped_linear_absolute_ratio_projection_failure_preserves_log_ratio() -> None:
    metric = MetricSpec(
        name="revenue_per_order",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        decision_method=Method.cuped_multicovariate(
            name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
        ),
    )
    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": 1e150 * (1.0 + index / 1000.0 + (group == "treatment") / 100.0),
            "orders": 1e-200,
            "pre": ("a", "b", "c")[index % 3],
        }
        for group in ("control", "treatment")
        for index in range(18)
    ]
    result = cast(
        list[LiftEstimate],
        list(
            Analysis.from_unit_summary(
                pa.Table.from_pylist(rows),
                unit="unit",
                group="group",
                control="control",
                metrics=[metric],
            ).run()
        ),
    )

    treatment = next(row for row in result if row.group_id == "treatment")
    assert treatment.lift is not None
    assert treatment.abs_diff is None
