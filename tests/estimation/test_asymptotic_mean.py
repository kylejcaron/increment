"""Count normalization, full Fieller geometry and stopped public evidence."""

import math
from fractions import Fraction as F
from typing import Any, cast

import pytest

from increment import AsymptoticMean, SequentialCell, estimate_sequential
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.asymptotic_mean import (
    MeanSetComponent,
    _component_for_quadratic,
    count_boundary,
)
from increment.estimation.decision_types import AsymptoticSequentialEvidence, EValueEvidence
from increment.estimation.family import select_sequential_family
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential_runtime import selected_snapshot_results
from increment.sequential_state import declare_sequential_freeze_cells
from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration


def _evaluate(control, treatment, *, alpha=F(1, 20), alternative="two-sided", null=F(0), start=2):
    reg = mean_registration(
        models=(mean_model(start_count=start),),
        cells=(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                alpha=alpha,
                alternative=alternative,
                null_lift=null,
            ),
        ),
    )
    snapshot = mean_capture(reg, mean_records(control, treatment))
    computation = estimate_sequential(snapshot, AsymptoticMean(registration=reg))
    return computation.results[0], next(iter(computation.evidence.values()))


def test_public_unequal_arm_counts_shifted_contrast_and_mean_variance():
    row, evidence = _evaluate([1, 2, 3], [4, 6, 8, 10], null=F(1, 2))
    result = row.require_asymptotic_sequential_result()
    assert isinstance(evidence, AsymptoticSequentialEvidence)
    assert not isinstance(evidence, EValueEvidence)
    assert result.bounds.count == 7
    assert result.bounds.estimator_contrast == 4
    # M2_C=2, M2_T=20; estimator variances are M2/n², not M2/n.
    assert result.bounds.estimator_variance == F(20, 16) + F(9, 4) * F(2, 9)
    expected = (1 + 100 / 7) * (math.log1p(7 / 100) - 2 * math.log(0.05))
    assert result.bounds.k is not None
    assert float(result.bounds.k) == pytest.approx(expected)
    assert result.checkpoint.control.n == 3
    assert result.checkpoint.treatment.n == 4
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    from increment.errors import CapabilityError

    with pytest.raises(CapabilityError) as raised:
        row.p_value()
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_directional_boundary_spends_its_alpha_on_one_tail(alternative):
    row, _ = _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40, alternative=alternative)
    result = row.require_asymptotic_sequential_result()
    assert result.bounds.alpha == F(1, 20)
    boundary_alpha = F(1, 20) if alternative == "two-sided" else F(1, 10)
    assert result.bounds.k == count_boundary(240, boundary_alpha, F(1, 10))
    assert row.stat_sig() == (alternative != "less")
    if alternative == "greater":
        assert result.bounds.upper is None and result.bounds.lower is not None
    if alternative == "less":
        assert result.bounds.lower is None and result.bounds.upper is not None


@pytest.mark.parametrize("alternative,endpoint", [("greater", "lower"), ("less", "upper")])
def test_one_sided_endpoint_equals_the_two_sided_endpoint_at_twice_alpha(alternative, endpoint):
    one, _ = _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40, alpha=F(1, 20), alternative=alternative)
    two, _ = _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40, alpha=F(1, 10))
    one_bounds = one.require_asymptotic_sequential_result().bounds
    two_bounds = two.require_asymptotic_sequential_result().bounds
    assert getattr(one_bounds, endpoint) == getattr(two_bounds, endpoint)
    assert one_bounds.status == "ray" and two_bounds.status == "bounded"


def test_one_sided_alpha_at_or_above_one_half_refuses_before_geometry():
    with pytest.raises(CapabilityError) as raised:
        _evaluate([1, 2, 3] * 40, [8, 10, 12] * 40, alpha=F(3, 5), alternative="greater")
    assert raised.value.code == "sequential.asymptotic_mean.invalid"


def test_one_sided_set_unions_the_never_rejected_half_line():
    # Control [-2, 1, 2] has a small positive mean with large scatter: the
    # two-sided set at 2*alpha is disconnected, (-inf, -91.3] u [13.96, inf),
    # with point ratio 33 in the upper piece. "less" adds (-inf, 33], covering
    # the line; "greater" adds [33, inf), already inside the set.
    two_sided, _ = _evaluate([-2, 1, 2] * 40, [10, 12] * 40, alpha=F(1, 10))
    less, _ = _evaluate([-2, 1, 2] * 40, [10, 12] * 40, alternative="less")
    greater, _ = _evaluate([-2, 1, 2] * 40, [10, 12] * 40, alternative="greater")
    reference = two_sided.require_asymptotic_sequential_result().bounds
    assert reference.status == "disconnected"
    less_bounds = less.require_asymptotic_sequential_result().bounds
    assert less_bounds.status == "full"
    assert less_bounds.components == (MeanSetComponent(lower=None, upper=None),)
    assert not less.stat_sig()
    greater_bounds = greater.require_asymptotic_sequential_result().bounds
    assert greater_bounds.components == reference.components
    assert greater_bounds.status == "disconnected"


@pytest.mark.parametrize(
    "control,treatment,start,reason",
    [
        ([], [1, 2], 2, "missing_arm"),
        ([1], [2, 3], 2, "insufficient_arm_observations"),
        ([1, 2], [2, 3], 3, "before_declared_start"),
        ([1, 1], [2, 3], 2, "zero_arm_variance"),
    ],
)
def test_unestimable_prefix_is_vacuous(control, treatment, start, reason):
    row, _ = _evaluate(control, treatment, start=start)
    bounds = row.require_asymptotic_sequential_result().bounds
    assert not row.stat_sig()
    assert not bounds.available
    assert bounds.reason == reason
    assert len(bounds.components) == 1
    assert bounds.lower is None and bounds.upper is None


def test_unavailable_point_preserves_available_disconnected_set():
    row, _ = _evaluate([-1, 1] * 40, [10, 12] * 40)
    assert row.lift is None
    result = row.require_asymptotic_sequential_result()
    assert result.point_reason == "observed control mean is zero"
    assert result.bounds.available
    assert result.bounds.status == "disconnected"
    assert len(result.bounds.components) == 2
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


def test_signed_treatment_has_no_bernoulli_floor():
    row, _ = _evaluate([9, 11] * 80, [-11, -9] * 80, alternative="less", null=F(-3, 2))
    assert row.require_lift().value == -2
    assert row.stat_sig()
    upper = row.require_asymptotic_sequential_result().bounds.upper
    assert upper is not None and upper < 0
    lift_upper = row.require_lift().ub
    assert lift_upper is not None and lift_upper < -1


@pytest.mark.parametrize(
    "coefficients,geometry",
    [
        ((1, 0, -1), ((-1, 1),)),
        ((-1, 0, 1), ((None, -1), (1, None))),
        ((0, 2, -4), ((None, 2),)),
        ((0, -2, 4), ((2, None),)),
        ((0, 0, -1), ((None, None),)),
        ((1, 0, 1), ()),
        ((1, -2, 1), ((1, 1),)),
        ((-1, 0, -1), ((None, None),)),
    ],
)
def test_quadratic_geometry(coefficients, geometry):
    result = _component_for_quadratic(*(F(v) for v in coefficients))
    assert tuple((c.lower, c.upper) for c in result) == geometry


def test_nearly_singular_coefficient_is_not_silently_linearized():
    eps = F(1, 10**80)
    result = _component_for_quadratic(eps, F(-1), F(-1))
    assert len(result) == 1
    assert result[0].lower is not None and result[0].lower < 0
    assert result[0].upper is not None and result[0].upper > 10**80


def test_extreme_alpha_and_overflowing_second_moments_keep_exact_state():
    alpha = F(1, 10**400)
    scale = 10**200
    row, _ = _evaluate([scale, 2 * scale] * 8, [3 * scale, 5 * scale] * 8, alpha=alpha)
    bounds = row.require_asymptotic_sequential_result().bounds
    assert bounds.available and bounds.k is not None and bounds.k > 0
    assert bounds.alpha == alpha
    assert row.require_lift().value == pytest.approx(F(5, 3))
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


def test_large_offset_neighbor_values_preserve_centered_scatter():
    base = 1e100
    neighbor = math.nextafter(base, math.inf)
    row, _ = _evaluate([base, neighbor] * 3, [base, neighbor] * 4)
    cp = row.require_asymptotic_sequential_result().checkpoint
    gap = F(neighbor) - F(base)
    assert cp.control.scatter[0][0] == F(3, 2) * gap**2
    assert cp.treatment.scatter[0][0] == 2 * gap**2
    assert row.require_asymptotic_sequential_result().bounds.available


def test_fixed_roster_missing_segment_keeps_budget_and_frozen_stopping_state():
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            segment=(("segment", segment),),
            family=True,
            alpha=F(1, 40),
        )
        for segment in ("seen", "missing")
    )
    reg = mean_registration(cells=cells, q=F(1, 20))
    policy = AsymptoticMean(registration=reg)
    records = mean_records([1, 2] * 30, [9, 11] * 30, segment={"segment": "seen"})
    prefix = mean_capture(reg, records[:8])
    first = estimate_sequential(prefix, policy)
    stopped_row = next(
        row
        for row in first.results
        if row.require_asymptotic_sequential_result().checkpoint.cell.segment
        == (("segment", "seen"),)
    )
    declared = declare_sequential_freeze_cells(
        prefix, [stopped_row.require_asymptotic_sequential_result().checkpoint.cell]
    )
    full = mean_capture(reg, records, previous=declared)
    computation = estimate_sequential(full, policy)
    frozen = next(
        row
        for row in computation.results
        if row.require_asymptotic_sequential_result().checkpoint.status == "frozen"
    )
    assert (
        frozen.require_asymptotic_sequential_result().bounds
        == stopped_row.require_asymptotic_sequential_result().bounds
    )
    assert frozen.require_asymptotic_sequential_result().checkpoint.prefix_id == prefix.prefix_id
    assert mean_capture(reg, records, previous=full) == full
    outcome = select_sequential_family([], reg.q, policy, F(1, 20), computation=computation)
    assert outcome.n_family == 2
    assert outcome.fcr_alpha is None
    rows = selected_snapshot_results(full, policy, nominal_alpha=F(1, 20))
    observed = next(
        row for row in rows if row.require_asymptotic_sequential_result().bounds.available
    )
    missing = next(
        row for row in rows if not row.require_asymptotic_sequential_result().bounds.available
    )
    assert observed.discovery and not missing.discovery
    assert observed.require_asymptotic_sequential_result().decision_alpha == F(1, 40)
    rewritten = [{**records[0], "values": {"outcome": 99}}, *records[1:]]
    with pytest.raises(CapabilityError) as caught:
        mean_capture(reg, rewritten, previous=full)
    assert caught.value.code == "sequential.continuation.rewrite"


def test_registration_rejects_overspending_and_mixed_exact_family():
    from tests.sequential_cases import registration

    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=True,
            segment=(("segment", str(i)),),
            alpha=F(1, 10),
        )
        for i in range(2)
    )
    with pytest.raises(InvalidRequestError):
        mean_registration(cells=cells, q=F(1, 10))
    exact = registration().models[0].model_copy(update={"metric": "binary"})
    with pytest.raises(InvalidRequestError):
        mean_registration(
            models=(mean_model(), exact),
            cells=(
                SequentialCell(metric="outcome", group_id="treatment"),
                SequentialCell(metric="binary", group_id="treatment"),
            ),
        )


def test_point_overflow_is_distinct_from_confidence_set_availability():
    row, _ = _evaluate([F(1, 10**300), F(2, 10**300)] * 20, [10**100, 2 * 10**100] * 20)
    result = row.require_asymptotic_sequential_result()
    assert row.lift is None
    assert result.point_reason == "relative point exceeds binary64 range"
    assert result.bounds.available
    assert result.bounds.components


@pytest.mark.parametrize("field", ["k", "alpha", "available", "components"])
def test_portable_geometry_cannot_be_modified(field):
    row, _ = _evaluate([1, 2] * 20, [8, 10] * 20)
    payload = row.model_dump()
    payload["sequential_result"]["bounds"][field] = {
        "k": F(0),
        "alpha": F(1, 10),
        "available": False,
        "components": (),
    }[field]
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate(payload)
    assert raised.value.code in {"sequential.source.invalid", "sequential.asymptotic_mean.invalid"}


def test_scalar_result_cannot_be_relabelled_exact():
    row, _ = _evaluate([1, 2] * 20, [8, 10] * 20)
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate({**row.model_dump(), "inference": "always_valid"})
    assert raised.value.code == "sequential.source.invalid"
    with pytest.raises(CapabilityError):
        row.require_exact_sequential_result()


def test_empty_geometry_never_makes_opposite_directional_discoveries():
    from increment.estimation.asymptotic_mean import AsymptoticMeanSet

    for alternative, contrast in (("greater", 10), ("less", -10)):
        bounds = AsymptoticMeanSet(
            components=(),
            alpha=F(1, 20),
            alternative=alternative,
            count=10,
            k=F(1),
            available=True,
            estimator_contrast=F(contrast),
            estimator_variance=F(1),
        )
        assert bounds.empty and not bounds.rejects()


def test_non_component_element_is_refused_on_the_components_field():
    from increment.estimation.asymptotic_mean import AsymptoticMeanSet

    with pytest.raises(InvalidRequestError) as raised:
        AsymptoticMeanSet(
            components=cast(Any, (3,)),
            alpha=F(1, 20),
            alternative="greater",
            count=0,
            k=None,
            available=False,
            reason="unavailable",
        )
    assert raised.value.code == "model.field.type"
    assert raised.value.context["model"] == "AsymptoticMeanSet"
    assert raised.value.context["field"] == "components"


def test_engine_dispatch_reaches_registered_scalar_mean_evidence():
    from increment.estimation import estimate_lift
    from increment.frame import MetricSpec, synthesise_metric

    reg = mean_registration()
    snapshot = mean_capture(reg, mean_records([1, 2] * 20, [8, 10] * 20))
    computation = estimate_lift(
        [synthesise_metric(MetricSpec(name="outcome", type="mean"))],
        snapshot,
        "control",
        inference=AsymptoticMean(registration=reg),
    )
    assert isinstance(next(iter(computation.evidence.values())), AsymptoticSequentialEvidence)
    assert computation.results[0].stat_sig()


def test_centered_states_and_boundaries_agree_across_order_and_append_partitions():
    reg = mean_registration()
    base = F(10**100)
    control = [base, base + 1, base + 3, base + 9]
    treatment = [base + 2, base + 5, base + 8, base + 10]
    records = mean_records(control, treatment)
    direct = mean_capture(reg, records)
    first = mean_capture(reg, records[:4])
    appended = mean_capture(reg, records[4:], previous=first, append=True)
    reversed_order = mean_capture(reg, list(reversed(records)))
    assert direct.states == appended.states == reversed_order.states
    policy = AsymptoticMean(registration=reg)
    bounds = [
        estimate_sequential(snapshot, policy)
        .results[0]
        .require_asymptotic_sequential_result()
        .bounds
        for snapshot in (direct, appended, reversed_order)
    ]
    assert bounds[0] == bounds[1] == bounds[2]
    assert direct.prefix_id != reversed_order.prefix_id


def test_exact_and_asymptotic_policies_cannot_reinterpret_each_others_models():
    from increment import AlwaysValid
    from tests.sequential_cases import registration

    with pytest.raises(InvalidRequestError):
        AlwaysValid(registration=mean_registration())
    with pytest.raises(InvalidRequestError):
        AsymptoticMean(registration=registration("gaussian"))
