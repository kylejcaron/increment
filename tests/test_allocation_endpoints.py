from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

import increment.readouts as readouts
from increment.breakout.estimates import run_breakout, run_daily_lift
from increment.compatibility import _conservative_ratio
from increment.decision import ArmHypothesisKey, DecisionComputation, FixedInference, PValueEvidence
from increment.errors import CodedError
from increment.estimation.arm_contract import (
    ArmCompatibilityRequest,
    ArmPlanningProcedure,
    FamilyPolicy,
    MethodCapability,
    PlanningFamilyExpansion,
    RelativeDecisionPolicy,
)
from increment.estimation.family import bh_select, e_bh_select, select_family
from increment.frame import FrameTotalsSource, MetricSpec, from_unit_summary
from increment.semantics.assignment import ParallelAssignment
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, MeanMetric, MethodSpec
from tests.breakout.test_daily import _make_daily_row
from tests.breakout.test_estimates import _make_arm_row, _mean_metric
from tests.estimation.test_family import _lift_estimate
from tests.test_compatibility_contract import _axes, _metric
from tests.test_readouts_encouragement import (
    FakeMomentSource,
    _design,
    _frame_encouragement_design,
    _multi_arm_uptake_table,
    _rows,
)

MIN_SUBNORMAL = math.nextafter(0.0, 1.0)


def endpoint(value: float | Fraction, exact: Fraction) -> None:
    if isinstance(value, Fraction):
        assert value == exact
        return
    assert Fraction(value) <= exact < Fraction(math.nextafter(value, math.inf))


def _procedure(*, alpha: float, family_size: int) -> ArmPlanningProcedure:
    request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=_axes(),
        dependence="iid",
        inference=FixedInference(),
        estimand="itt",
        metric=_metric(),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=FamilyPolicy(
                kind="bonferroni",
                axes=("metric",),
                nominal_alpha=alpha,
            ),
        ),
        methods=(
            MethodCapability(
                role="decision",
                estimator="unadjusted",
                variance_reduction="none",
            ),
        ),
        prior_present=False,
    )
    return ArmPlanningProcedure(
        **request.model_dump(exclude={"methods"}),
        family_expansion=PlanningFamilyExpansion(family_size=family_size),
        decision_method=MethodSpec(name="unadjusted"),
        sensitivity_methods=(),
    )


def _three_arm_table() -> pa.Table:
    base = _multi_arm_uptake_table().to_pydict()
    original_groups = list(base["variant"])
    for index, group in enumerate(original_groups):
        if group != "t2":
            continue
        for column in base:
            base[column].append(base[column][index])
        base["user_id"][-1] = f"{base['user_id'][-1]}_t3"
        base["variant"][-1] = "t3"
    base["visits"] = list(base["revenue"])
    base["sessions"] = list(base["revenue"])
    return pa.table(base)


def _three_primary_randomized_source(*, alpha: float):
    return FrameTotalsSource.from_frame(
        _three_arm_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="revenue"),
            MetricSpec(name="visits"),
            MetricSpec(name="sessions"),
        ],
        design=Randomized(control_group="control"),
        plan=AnalysisPlan(
            alpha=alpha,
            primary=["revenue", "visits", "sessions"],
        ),
    )


def _three_primary_encouragement_source(*, alpha: float):
    return from_unit_summary(
        _three_arm_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean", "sessions": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(
            alpha=alpha,
            primary=["revenue", "visits", "sessions"],
        ),
    )


def _two_metric_randomized_source(*, alpha: float = 0.1):
    table = _multi_arm_uptake_table().append_column("visits", _multi_arm_uptake_table()["revenue"])
    return FrameTotalsSource.from_frame(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue"), MetricSpec(name="visits")],
        design=Randomized(control_group="control"),
        plan=AnalysisPlan(alpha=alpha, primary=["revenue", "visits"]),
    )


def _asof_rows(
    *, metric_names=("revenue", "visits"), arm_ids=("t1", "t2"), encouragement: bool = False
):
    rows = []
    for metric_name in metric_names:
        for row in _rows(metric=metric_name):
            row = dict(row, ds=date(2025, 1, 1))
            if row["group_id"] == "treat":
                for arm_id in arm_ids:
                    rows.append(dict(row, group_id=arm_id))
            else:
                rows.append(row)
    return rows


def test_planning_alpha_and_tail_are_exact_endpoints():
    alpha = math.nextafter(1.0, 0.0)
    procedure = _procedure(alpha=alpha, family_size=3)

    endpoint(procedure.compiled_alpha, Fraction(alpha) / 3)
    endpoint(procedure.compiled_tail_alpha, Fraction(alpha) / 6)


def test_planning_subnormal_allocation_refuses_stably():
    procedure = _procedure(alpha=MIN_SUBNORMAL, family_size=2)

    with pytest.raises(CodedError) as raised:
        _ = procedure.compiled_alpha

    assert raised.value.code == "allocation.alpha.underflow"


def test_planning_tail_subnormal_refuses_after_compiled_alpha_survives():
    procedure = _procedure(alpha=MIN_SUBNORMAL, family_size=1)
    assert procedure.compiled_alpha == MIN_SUBNORMAL

    with pytest.raises(CodedError) as raised:
        _ = procedure.compiled_tail_alpha

    assert raised.value.code == "allocation.alpha.underflow"


def test_bh_subnormal_threshold_refuses_stably():
    with pytest.raises(CodedError) as raised:
        bh_select([0.0, 1.0], MIN_SUBNORMAL)

    assert raised.value.code == "allocation.alpha.underflow"


def test_e_bh_subnormal_threshold_refuses_stably():
    with pytest.raises(CodedError) as raised:
        e_bh_select([1.0e300, 0.0], MIN_SUBNORMAL)

    assert raised.value.code == "allocation.alpha.underflow"


def test_fcr_subnormal_ratio_refuses_stably():
    """A family whose selected-interval alpha would underflow float64 refuses
    by code from the production selector rather than reporting 0.0."""
    cells = [
        (
            ArmHypothesisKey(f"m_{i}", "treatment", "itt"),
            _lift_estimate(f"m_{i}", "treatment", 0.2 if i == 0 else 0.0, 0.02),
        )
        for i in range(2)
    ]
    computation = DecisionComputation(
        results=(),
        evidence={
            key: PValueEvidence(key, "unadjusted", row.p_value(), "normal") for key, row in cells
        },
        failures={},
    )
    with pytest.raises(CodedError) as raised:
        select_family(
            cells, q=MIN_SUBNORMAL, inference=None, nominal_alpha=0.9, computation=computation
        )

    assert raised.value.code == "allocation.alpha.underflow"


def test_bh_threshold_and_identity_are_exact_endpoints():
    q = math.nextafter(1.0, 0.0)
    _, threshold = bh_select([MIN_SUBNORMAL, MIN_SUBNORMAL, 1.0], q)
    endpoint(threshold, Fraction(q) * 2 / 3)

    identity_q = math.nextafter(math.nextafter(1.0, 0.0), 0.0)
    _, identity_threshold = bh_select([0.0, 0.0, 0.0], identity_q)
    assert identity_threshold == identity_q


def test_e_bh_public_selector_respects_endpoint_threshold():
    q = math.nextafter(1.0, 0.0)
    threshold = _conservative_ratio(q, 2, 3)
    e_at_threshold = 1.0 / threshold
    log_e_at_threshold = math.log(e_at_threshold)
    assert e_bh_select([log_e_at_threshold, log_e_at_threshold, -math.inf], q) == [0, 1]
    assert e_bh_select([0.0, 0.0, -math.inf], q) == []
    endpoint(threshold, Fraction(q) * 2 / 3)


def test_selected_alpha_is_an_exact_endpoint():
    q = math.nextafter(1.0, 0.0)

    cells = [
        (
            (f"m_{i}", "treatment"),
            _lift_estimate(
                f"m_{i}",
                "treatment",
                0.2 if i == 0 else 0.0,
                0.02,
            ),
        )
        for i in range(3)
    ]
    typed_cells = [
        (ArmHypothesisKey(row.metric, row.group_id, row.estimand), row) for _key, row in cells
    ]
    computation = DecisionComputation(
        results=(),
        evidence={
            key: PValueEvidence(key, "unadjusted", row.p_value(), "normal")
            for key, row in typed_cells
        },
        failures={},
    )
    outcome = select_family(
        typed_cells,
        q=q,
        inference=None,
        nominal_alpha=0.9,
        computation=computation,
    )
    assert outcome.fcr_alpha is not None
    endpoint(outcome.fcr_alpha, Fraction(q) / 3)


def test_breakout_bonferroni_emits_exact_endpoint():
    rows = [
        row
        for segment in ("US", "CA", "GB")
        for row in (
            _make_arm_row(50, 10.0, 4.0, country=segment, group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country=segment, group_id="treatment"),
        )
    ]
    estimates = run_breakout(
        rows,
        [_mean_metric()],
        control_group="control",
        dimension="country",
        alpha=0.1,
        correction="bonferroni",
    )

    alphas = {estimate.require_lift().alpha for estimate in estimates}
    assert len(alphas) == 1
    segment_alpha = alphas.pop()
    assert segment_alpha is not None
    endpoint(segment_alpha, Fraction(0.1) / 3)


def test_daily_per_metric_bonferroni_emits_exact_endpoint():
    rows = []
    for segment in ("US", "CA", "GB"):
        for group in ("control", "treatment"):
            row = _make_daily_row(
                50, 10.0 if group == "control" else 11.0, 4.0, ds=date(2025, 1, 1), group_id=group
            )
            row["country"] = segment
            rows.append(row)
    estimates = run_daily_lift(
        rows,
        [_mean_metric()],
        control_group="control",
        dimension="country",
        alpha=0.1,
        correction="bonferroni",
    )

    alphas = {estimate.require_lift().alpha for estimate in estimates}
    assert len(alphas) == 1
    daily_alpha = alphas.pop()
    assert daily_alpha is not None
    endpoint(daily_alpha, Fraction(0.1) / 3)


def _itt_alphas(rows: Iterable[Any]) -> set[float]:
    return {row.require_lift().alpha for row in rows if row.estimand == "itt"}


def test_randomized_run_primary_arm_split_uses_composite_endpoint():
    alpha = math.nextafter(1.0, 0.0)
    rows = readouts.run(_three_primary_randomized_source(alpha=alpha))

    alphas = _itt_alphas(rows)
    assert len(alphas) == 1
    endpoint(alphas.pop(), Fraction(alpha) / 9)


def test_asof_randomized_primary_arm_split_uses_composite_endpoint():
    alpha = math.nextafter(1.0, 0.0)
    metric_names = ("revenue", "visits", "sessions")
    metrics = [MeanMetric(name=name, entity="user", fact=name) for name in metric_names]
    source = FakeMomentSource(
        _asof_rows(metric_names=metric_names, arm_ids=("t1", "t2", "t3")),
        metrics=metrics,
        capabilities={"asof"},
        design=Randomized(control_group="control"),
        plan=AnalysisPlan(alpha=alpha, primary=list(metric_names)),
    )
    rows = readouts.asof_lift(source)

    alphas = _itt_alphas(rows)
    assert len(alphas) == 1
    endpoint(alphas.pop(), Fraction(alpha) / 9)


def test_asof_encouragement_primary_arm_split_uses_endpoint():
    alpha = math.nextafter(1.0, 0.0)
    metric_names = ("revenue", "visits", "sessions")
    metrics = [MeanMetric(name=name, entity="user", fact=name) for name in metric_names]
    source = FakeMomentSource(
        _asof_rows(metric_names=metric_names, arm_ids=("t1", "t2", "t3")),
        metrics=metrics,
        capabilities={"asof"},
        design=_design(),
        plan=AnalysisPlan(alpha=alpha, primary=list(metric_names)),
    )
    rows = readouts.asof_lift(source)

    alphas = _itt_alphas(rows)
    assert len(alphas) == 1
    endpoint(alphas.pop(), Fraction(alpha) / 9)


def test_encouragement_run_primary_arm_split_uses_composite_endpoint():
    alpha = math.nextafter(1.0, 0.0)
    source = _three_primary_encouragement_source(alpha=alpha)
    rows = readouts.run(source)

    alphas = _itt_alphas(rows)
    assert len(alphas) == 1
    endpoint(alphas.pop(), Fraction(alpha) / 9)


def test_public_breakout_subnormal_alpha_refuses_stably():
    rows = [
        _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
        _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
        _make_arm_row(50, 11.0, 4.0, country="CA", group_id="treatment"),
    ]
    with pytest.raises(CodedError) as raised:
        run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alpha=MIN_SUBNORMAL,
            correction="bonferroni",
        )
    assert raised.value.code == "allocation.alpha.underflow"


def test_public_daily_subnormal_alpha_refuses_stably():
    rows = []
    for segment in ("US", "CA"):
        for group in ("control", "treatment"):
            row = _make_daily_row(
                50,
                10.0 if group == "control" else 11.0,
                4.0,
                ds=date(2025, 1, 1),
                group_id=group,
            )
            row["country"] = segment
            rows.append(row)
    with pytest.raises(CodedError) as raised:
        run_daily_lift(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alpha=MIN_SUBNORMAL,
            correction="bonferroni",
        )
    assert raised.value.code == "allocation.alpha.underflow"


def test_public_randomized_run_subnormal_alpha_refuses_at_arm_boundary():
    source = _two_metric_randomized_source(alpha=2 * MIN_SUBNORMAL)

    with pytest.raises(CodedError) as raised:
        readouts.run(source)

    assert raised.value.code == "allocation.alpha.underflow"


def test_public_randomized_asof_subnormal_alpha_refuses_at_arm_boundary():
    metrics = [MeanMetric(name=name, entity="user", fact=name) for name in ("revenue", "visits")]
    source = FakeMomentSource(
        _asof_rows(),
        metrics=metrics,
        capabilities={"asof"},
        design=Randomized(control_group="control"),
        plan=AnalysisPlan(alpha=2 * MIN_SUBNORMAL, primary=["revenue", "visits"]),
    )

    with pytest.raises(CodedError) as raised:
        readouts.asof_lift(source)

    assert raised.value.code == "allocation.alpha.underflow"


def test_public_encouragement_run_subnormal_alpha_refuses_at_arm_boundary():
    table = _multi_arm_uptake_table().append_column("visits", _multi_arm_uptake_table()["revenue"])
    source = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(alpha=2 * MIN_SUBNORMAL, primary=["revenue", "visits"]),
    )

    with pytest.raises(CodedError) as raised:
        readouts.run(source)

    assert raised.value.code == "allocation.alpha.underflow"


def test_public_encouragement_asof_subnormal_alpha_refuses_at_arm_boundary():
    metrics = [MeanMetric(name=name, entity="user", fact=name) for name in ("revenue", "visits")]
    source = FakeMomentSource(
        _asof_rows(),
        metrics=metrics,
        capabilities={"asof"},
        design=_design(),
        plan=AnalysisPlan(alpha=2 * MIN_SUBNORMAL, primary=["revenue", "visits"]),
    )

    with pytest.raises(CodedError) as raised:
        readouts.asof_lift(source)

    assert raised.value.code == "allocation.alpha.underflow"
