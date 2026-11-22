"""Design-matched power counts a refusal as an attempted, unsuccessful decision."""

import math
import sys
from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import polars as pl
import pytest
from scipy.stats import t

from increment import from_switchback_frame
from increment.errors import InvalidRequestError
from increment.estimation.contrast import estimate_contrast
from increment.estimation.decision_types import ContrastDecisionProcedure
from increment.estimation.results import contrast_stat_sig
from increment.power.switchback import SwitchbackBaseline
from increment.power.switchback_profiles import (
    SwitchbackConversionBaseline,
    SwitchbackProfilePlan,
    SwitchbackTrajectoryBaseline,
    switchback_profile_plan,
)
from increment.semantics.assignment import SwitchbackAssignment
from increment.semantics.design import Randomized
from increment.semantics.unit_cycle import UnitCycleTApproximation
from increment.workflows import planning_baseline
from tests.analysis_factory import bound_source, contrast_rows, view_readout


def assignment(k=5, m=4, shared=True, p=None):
    grain = "shared_block" if shared else "unit_cycle"
    sequence = (
        {"scheme": "fixed_count", "treated_periods": m, "independence_unit": grain}
        if p is None
        else {
            "scheme": "shared_schedule" if shared else "independent_bernoulli_order",
            "probability_ct": p,
        }
    )
    return SwitchbackAssignment.model_validate(
        {
            "periods_per_cycle": k,
            "sequence": sequence,
            "window": {"washout_steps": 1, "observation_steps": 4, "carryover_order": 1},
        }
    )


def model_args(a, roster=("a",), cycles=1):
    return {
        "assignment": a,
        "metric": "outcome",
        "control_group": "C",
        "treatment_group": "T",
        "roster": roster,
        "cycles_per_unit": cycles,
    }


def procedure(a, **changes):
    return ContrastDecisionProcedure.model_validate(
        dict(
            {
                "metric": "outcome",
                "role": "primary",
                "alpha": 0.05,
                "null_abs": 0.0,
                "alternative": "two-sided",
                "reference": (
                    UnitCycleTApproximation()
                    if a.sequence.independence_unit == "unit_cycle"
                    else None
                ),
            },
            **changes,
        )
    )


def decisions(xs, critical, null=0, side="two-sided"):
    n = len(xs)
    mu = sum(xs) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n * (n - 1)))
    if not se:
        return "zero_se"
    z = (mu - null) / se
    statistic = abs(z) if side == "two-sided" else -z if side == "less" else z
    return "rejected" if statistic > critical else "kept"


def assert_law(row, expected):
    total = sum(expected.values())
    for key in ("zero_se", "admission_failures", "decision_failures", "rejected"):
        p = expected[key] / total
        assert getattr(row, key) / row.attempts == pytest.approx(
            p, abs=6 * math.sqrt(p * (1 - p) / row.attempts)
        )
    assert row.power_lower <= expected["rejected"] / total <= row.power_upper
    assert row.power == row.rejected / row.attempts


def test_decision_oracle_requires_strict_null_exclusion():
    assert decisions([0, 2], 1, null=0) == "kept"
    assert decisions([0, 2], math.nextafter(1, 0), null=0) == "rejected"


@pytest.mark.parametrize(("shift", "rejected"), [(0.5, False), (0.75, True)])
def test_fixed_count_joint_admission_not_product_of_marginals(shift, rejected):
    critical = t.isf(0.025, 4)
    counts = Counter()
    for control_positions in product(range(5), repeat=5):
        values = [shift + (-1 if j == 0 else 0.25) for j in control_positions]
        key = (
            decisions(values, critical)
            if len(set(control_positions)) == 5
            else "admission_failures"
        )
        counts[key] += 1
    assert counts == {"admission_failures": 3005, ("rejected" if rejected else "kept"): 120}
    a = assignment()
    roster = ("site-a", "site-b", "site-c")
    block = ((1, 0, 0, 0, 0),) * len(roster)
    b = SwitchbackTrajectoryBaseline(**model_args(a, roster), profiles=(block,))
    plan = switchback_profile_plan(
        b, procedure(a), sizes=[5], effects=[shift], seed=41, mc_error=0.001
    )
    assert_law(plan.points[0], counts)
    assert plan.baseline.planning_grain == "shared_block"
    assert plan.reference == "block_t"
    assert plan.grid_size(shift).status == plan.grid_mde(5).status == "not_attained"
    assert SwitchbackProfilePlan.model_validate_json(plan.model_dump_json()) == plan

    def bind(frame):
        return from_switchback_frame(
            frame,
            unit="u",
            cycle="c",
            period="p",
            step="s",
            group="arm",
            metrics={"outcome": "mean"},
            assignment=a,
            identification=Randomized(control_group="C", allocation={"C": 0.2, "T": 0.8}),
        )

    for positions in ((0, 1, 2, 3, 4), (0, 0, 1, 2, 3)):
        rows = [
            {
                "u": u,
                "c": c,
                "p": p,
                "s": s,
                "arm": "T" if z else "C",
                "outcome": ((p == 0) + shift * z) if s == 2 else 0.0,
            }
            for u in roster
            for c, control in enumerate(positions)
            for p in range(5)
            for z in (p != control,)
            for s in range(5)
        ]
        if len(set(positions)) != 5:
            with pytest.raises(InvalidRequestError) as caught:
                bind(pl.DataFrame(rows))
            assert caught.value.context["reason"] == "missing_position_support"
            continue
        experiment, reader = bind(pl.DataFrame(rows))
        (actual,) = contrast_rows(view_readout(experiment, reader))
        fitted = planning_baseline(experiment, reader, "outcome")
        assert isinstance(fitted, SwitchbackTrajectoryBaseline)
        assert fitted.roster == roster and fitted.delta_ref == shift
        assert fitted.profiles == (block,) * 5
        assert actual.estimate is not None
        assert actual.estimate.value == shift
        assert actual.standard_error == 0.25
        assert actual.dof == 4
        assert contrast_stat_sig(actual) == rejected


@pytest.mark.parametrize("shared", [False, True])
def test_retained_window_equal_risk_discrete_law(shared):
    # Six independent window indicators and three independent orders. A common
    # shock copied into both periods would instead make *every* contrast zero.
    counts = Counter()
    common_shock = Counter()
    critical = t.isf(0.1, 2)
    for windows in product((0, 1), repeat=6):
        for orders in product((-1, 1), repeat=3):
            x = [Fraction(s * (windows[2 * i + 1] - windows[2 * i])) for i, s in enumerate(orders)]
            key = (
                "admission_failures" if shared and len(set(orders)) == 1 else decisions(x, critical)
            )
            counts[key] += 1
            if windows[::2] == windows[1::2]:
                common_shock[key] += 1
    assert counts["rejected"] == (72 if shared else 96)
    assert sum(counts.values()) == 512
    assert common_shock["zero_se"] == (48 if shared else 64)
    assert common_shock["rejected"] == 0
    a = assignment(k=2, shared=shared, p=0.5)
    b = SwitchbackConversionBaseline(**model_args(a), untreated_risk=0.5)
    plan = switchback_profile_plan(
        b, procedure(a, alpha=0.2), sizes=[3], effects=[0], seed=27, mc_error=0.001
    )
    assert_law(plan.points[0], counts)
    assert plan.reference == ("block_t" if shared else "unit_t_approximation")
    assert b.dependence == "independent_windows_and_roster_units_given_arm"


def test_fixed_count_conversion_joint_law_and_simultaneous_zero_hit_limit():
    a = assignment(k=3, m=1)
    b = SwitchbackConversionBaseline(**model_args(a, ("a", "b")), untreated_risk=0.0)
    proc = procedure(a, alpha=0.2, alternative="greater")
    plan = switchback_profile_plan(
        b, proc, sizes=[3], effects=[0.0, 0.5], repetitions=4096, mc_error=0.001
    )
    zero, positive = plan.points
    # Two-sided Clopper-Pearson for no hits, spending half the error
    # on each tail at each of the two prespecified grid points.
    assert zero.power == zero.power_lower == 0
    assert zero.power_upper == pytest.approx(-math.expm1(math.log(0.001 / 4) / zero.attempts))
    counts = Counter()
    critical = t.isf(0.2, 2)
    for treated in product(range(3), repeat=3):
        for hits in product(range(2), repeat=6):
            x = [Fraction(a + b, 2) for a, b in zip(hits[::2], hits[1::2], strict=True)]
            key = (
                decisions(x, critical, side="greater")
                if len(set(treated)) == 3
                else "admission_failures"
            )
            counts[key] += 1
    assert_law(positive, counts)
    assert type(b).model_validate_json(b.model_dump_json()) == b


@pytest.mark.parametrize("shared", [True, False])
def test_conversion_pilot_checks_draw_count_at_the_paired_admission_boundary(shared):
    roster = ("east", "west")

    def fit(p, orders):
        ct, tc = (0, 1), (1, 0)
        mid = len(orders) // 2
        treated = (orders, orders) if shared else (orders[:mid], orders[mid:])
        return SwitchbackConversionBaseline.from_pilot(
            assignment=assignment(k=2, shared=shared, p=p),
            metric="outcome",
            control_group="C",
            treatment_group="T",
            roster=roster,
            observed=[[(1, 1)] * len(unit) for unit in treated],
            treated=[[ct if o else tc for o in unit] for unit in treated],
            delta_ref=0,
        )

    # At 40 draws and p=.8, k=17 is below the 1e-6 exact-test threshold
    # while k=18 is above. Copying the shared roster must not double the count.
    cases = [(0.8, 40, 17, False), (0.8, 40, 18, True), (0.8, 40, 40, not shared)]
    if not shared:
        cases.extend([(0.001, 2, 2, True), (math.nextafter(0.001, 0.0), 2, 2, False)])
    for p, n, k, admitted in cases:
        orders = [True] * k + [False] * (n - k)
        if admitted:
            assert fit(p, orders).untreated_risk == 1
        else:
            with pytest.raises(InvalidRequestError) as exc:
                fit(p, orders)
            assert exc.value.code == "power.switchback.baseline"


def test_weighted_pilot_risk_and_whole_repeated_window_grain():
    a = assignment(k=2, p=0.75, shared=False)
    # CT control successes have weight 1/.75; TC control failures, 1/.25.
    z = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    y = np.array([[[1, 1], [1, 0]], [[1, 0], [1, 1]]])
    b = SwitchbackConversionBaseline.from_pilot(
        assignment=a,
        metric="outcome",
        control_group="C",
        treatment_group="T",
        roster=("b", "a"),
        observed=y,
        treated=z,
        delta_ref=0.0,
    )
    assert b.untreated_risk == 0.25
    assert (b.roster, b.cycles_per_unit, b.risk_source) == (("a", "b"), 2, "pilot_estimated")
    plan = switchback_profile_plan(
        b, procedure(a, alternative="less"), sizes=[8, 60], effects=[-0.25]
    )
    assert plan.points[0].attempts > plan.points[0].rejected > 0
    assert plan.grid_mde(60).value == -0.25
    assert plan.grid_size(-0.25).value == 60
    with pytest.raises(InvalidRequestError) as exc:
        switchback_profile_plan(b, procedure(a), sizes=[4], effects=[math.nextafter(0.75, 1.0)])
    assert exc.value.code == "power.switchback.delta"


def test_decimal_risk_at_one_accepts_endpoint_but_not_next_float():
    a = assignment(k=2, p=0.5)
    b = SwitchbackConversionBaseline(**model_args(a), untreated_risk=0.2)
    p = procedure(a, alpha=0.2)
    (row,) = switchback_profile_plan(b, p, sizes=[3], effects=[0.8], mc_error=0.001).points
    # Exactly two of three treated-minus-control indicators must be one;
    # independent of outcomes, both shared orders occur with probability 6/8.
    probability = Fraction(6, 8) * 3 * Fraction(4, 5) ** 2 * Fraction(1, 5)
    assert row.power_lower <= probability <= row.power_upper
    with pytest.raises(InvalidRequestError) as exc:
        switchback_profile_plan(b, p, sizes=[3], effects=[math.nextafter(0.8, 1)])
    assert exc.value.code == "power.switchback.delta"


def test_trajectory_pilot_reconstruction_preserves_offsets_and_roundtrips():
    k = 10
    a = assignment(k, 1, shared=False)
    z = np.eye(k, dtype=bool)[None, :, :].repeat(2, axis=0)
    y = np.broadcast_to((2**55 + np.arange(k))[None, None, :], z.shape).copy()
    y[0] += 7

    def pilot(outcomes):
        return SwitchbackTrajectoryBaseline.from_pilot(
            assignment=a,
            metric="outcome",
            control_group="C",
            treatment_group="T",
            roster=("b", "a"),
            observed=outcomes,
            treated=z,
            delta_ref=0.25,
        )

    b = pilot(y)
    assert b.roster == ("a", "b")
    assert b.profiles[0][1][1] == 2**55 + Fraction(3, 4)
    assert b.profiles[1][0][0] == 2**55 + Fraction(27, 4)
    restored = SwitchbackTrajectoryBaseline.model_validate_json(b.model_dump_json())

    def plan(model):
        return switchback_profile_plan(
            model, procedure(a), sizes=[14], effects=[1], repetitions=128, seed=5
        )

    high = plan(b)
    assert plan(restored) == high
    assert plan(pilot(y - 2**55)).points == high.points
    assert b.profile_source == "pilot_estimated"
    assert b.assignment.window.observation_steps == 4


@pytest.mark.parametrize("shared", [True, False], ids=["blocks", "units"])
def test_shifted_pilot_and_runtime_agree_at_a_large_null(shared):
    # Removing the pilot shift D leaves position effects (-2, 0, 2).
    # With one treated position the corresponding contrasts are (-3, 0, 3).
    # Every admitted three-replicate schedule is a permutation of those three.
    d = 2**55
    a = assignment(3, 1, shared=shared)
    roster = ("a",) if shared else ("a", "b", "c")
    pilot_z = np.eye(3, dtype=bool).reshape(len(roster), -1, 3)
    position = np.array([-2, 0, 2])
    baseline = SwitchbackTrajectoryBaseline.from_pilot(
        assignment=a,
        metric="outcome",
        control_group="C",
        treatment_group="T",
        roster=roster,
        observed=position + d * pilot_z,
        treated=pilot_z,
        delta_ref=d,
    )
    shift = d + 8
    proc = procedure(a, null_abs=d)
    (planned,) = switchback_profile_plan(
        baseline, proc, sizes=[3], effects=[shift], repetitions=128, seed=41
    ).points
    assert 0 < planned.rejected == planned.attempts - planned.admission_failures

    # Rotate the assignments and apply the new shift to the pilot's untreated totals.
    schedule = np.roll(pilot_z, 1, axis=2)
    rows = [
        {
            "u": roster[u],
            "c": c,
            "p": p,
            "s": s,
            "arm": "T" if schedule[u, c, p] else "C",
            "outcome": int(position[p] + shift * schedule[u, c, p]) if s == 2 else 0,
        }
        for u, c, p in np.ndindex(schedule.shape)
        for s in range(5)
    ]
    _, reader = from_switchback_frame(
        pl.DataFrame(rows),
        unit="u",
        cycle="c",
        period="p",
        step="s",
        group="arm",
        metrics={"outcome": "mean"},
        assignment=a,
        identification=Randomized(control_group="C", allocation={"C": 2 / 3, "T": 1 / 3}),
    )
    source = bound_source(reader)
    (row,) = estimate_contrast(source.contrast_stats(source.metrics[0]), proc).results
    assert row.standard_error == pytest.approx(math.sqrt(3))
    assert row.estimate is not None and row.estimate.lb == d
    assert contrast_stat_sig(row) is True  # 8/sqrt(3) exceeds t(.975, 2).


def test_whole_unit_bootstrap_preserves_cycles_and_shared_roster_dependence():
    a = assignment(3, 1, shared=False)
    flat = (0, 0, 0)
    spike = (1, 0, 0)
    b = SwitchbackTrajectoryBaseline(
        **model_args(a, ("a", "b"), 2), profiles=((flat, flat), (spike, spike))
    )
    counts = Counter()
    critical = t.isf(0.2, 1)
    for profiles in product(range(2), repeat=2):
        for treated in product(range(3), repeat=4):
            xs = [
                Fraction(1, 2)
                + profile
                * sum(1 if j == 0 else Fraction(-1, 2) for j in treated[2 * i : 2 * i + 2])
                / 2
                for i, profile in enumerate(profiles)
            ]
            key = (
                decisions(xs, critical, side="greater")
                if len(set(treated)) == 3
                else "admission_failures"
            )
            counts[key] += 1
    plan = switchback_profile_plan(
        b, procedure(a, alternative="greater", alpha=0.2), sizes=[2], effects=[0.5], mc_error=0.001
    )
    assert_law(plan.points[0], counts)
    # A block samples the *same* two profiles, whose position effects cancel.
    shared = assignment(3, 1)
    blocks = SwitchbackTrajectoryBaseline(
        **model_args(shared, ("a", "b")), profiles=((spike, (-1, 0, 0)),)
    )
    row = switchback_profile_plan(blocks, procedure(shared), sizes=[8], effects=[0.5]).points[0]
    assert row.admission_failures + row.zero_se == row.attempts and row.zero_se > 0


def test_grid_first_crossing_requires_ruling_out_smaller_candidates():
    a = assignment(k=2, p=0.5, shared=False)
    b = SwitchbackConversionBaseline(**model_args(a), untreated_risk=0.5)
    p = procedure(a, alternative="greater")
    plan = switchback_profile_plan(
        b, p, sizes=[4, 40], effects=[0, 0.5], target_power=0.75, repetitions=2048
    )
    assert (plan.grid_mde(40).status, plan.grid_mde(40).value) == ("attained", 0.5)
    assert (plan.grid_size(0.5).status, plan.grid_size(0.5).value) == ("attained", 40)
    # Re-use the same fixed draws, put target inside the *first* row's band.
    first = next(row for row in plan.points if row.n == 4 and row.delta == 0.5)
    target = (first.power_lower + first.power_upper) / 2
    ambiguous = switchback_profile_plan(
        b, p, sizes=[4, 40], effects=[0, 0.5], target_power=target, repetitions=2048
    )
    assert ambiguous.grid_size(0.5).status == "undecided"
    assert ambiguous.grid_size(0.5).value is None
    assert ambiguous.points[-1].power_lower > target


def test_pilot_keeps_measured_periods_when_caller_edits_sorted_pandas_panel():
    import pandas as pd

    panel = pd.DataFrame(
        {
            "unit": "a",
            "cycle": c,
            "period": p,
            "step": 0,
            "arm": "T" if c == p else "C",
            "outcome": p + 10 * (c == p),
        }
        for c in range(3)
        for p in range(3)
    )
    experiment, reader = from_switchback_frame(
        panel,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        metrics={"outcome": "mean"},
        assignment=SwitchbackAssignment.model_validate(
            {
                "periods_per_cycle": 3,
                "sequence": {
                    "scheme": "fixed_count",
                    "treated_periods": 1,
                    "independence_unit": "shared_block",
                },
                "window": {"washout_steps": 0, "observation_steps": 1},
            }
        ),
        identification=Randomized(control_group="C", allocation={"C": 2 / 3, "T": 1 / 3}),
    )
    panel.loc[0, "outcome"] = 1010
    with reader:
        fitted = planning_baseline(experiment, reader, "outcome")
    assert isinstance(fitted, SwitchbackTrajectoryBaseline)
    assert fitted.delta_ref == 10
    assert fitted.profiles == (((0, 1, 2),),) * 3


def test_unrepresentable_period_totals_cancel_before_runtime_and_planning_contrast():
    a = assignment(3, 1)
    rows = [
        {
            "unit": "a",
            "cycle": c,
            "period": j,
            "step": s,
            "group": "T" if c == j else "C",
            "outcome": (0.0, 0.0, 1e308, 1e308, float((j == 0) + 10 * (c == j)))[s],
        }
        for c in range(3)
        for j in range(3)
        for s in range(5)
    ]
    experiment, reader = from_switchback_frame(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"outcome": "mean"},
        assignment=a,
        identification=Randomized(control_group="C", allocation={"C": 2 / 3, "T": 1 / 3}),
    )
    (actual,) = contrast_rows(view_readout(experiment, reader))
    assert actual.estimate is not None
    assert actual.estimate.value == 10
    assert actual.standard_error == pytest.approx(0.5)
    assert contrast_stat_sig(actual)
    fitted = planning_baseline(experiment, reader, "outcome")
    assert isinstance(fitted, SwitchbackTrajectoryBaseline)
    offset = 2 * Fraction(1e308)
    assert fitted.profiles == (((offset + 1, offset, offset),),) * 3
    (row,) = switchback_profile_plan(
        fitted, procedure(a), sizes=[3], effects=[10], mc_error=0.001
    ).points
    # The six permutations each give contrasts 11, 9.5, 9.5 in some order.
    assert_law(row, Counter(admission_failures=21, rejected=6))


def test_out_of_range_report_is_not_a_rejection_and_remains_in_denominator():
    a = assignment(3, 1)
    bound = sys.float_info.max
    # Every admitted experiment treats the first position at least once;
    # its contrast 2 * bound cannot be represented even after subtraction.
    b = SwitchbackTrajectoryBaseline(**model_args(a), profiles=(((bound, -bound, -bound),),))
    p = switchback_profile_plan(b, procedure(a), sizes=[6], effects=[0], repetitions=256)
    row = p.points[0]
    assert row.admission_failures + row.decision_failures == row.attempts
    assert row.decision_failures > 0 and row.rejected == 0


def test_planner_refuses_mismatched_reference_and_invalid_models():
    a = assignment(3, 1)
    b = SwitchbackTrajectoryBaseline(**model_args(a), profiles=(((1, 0, 0),),))
    for proc in (
        procedure(a, reference=UnitCycleTApproximation()),
        procedure(a, metric="other"),
        procedure(a, forecast_column="pre"),
    ):
        with pytest.raises(InvalidRequestError) as exc:
            switchback_profile_plan(b, proc, sizes=[2], effects=[0])
        assert exc.value.code == "power.switchback.procedure"
    with pytest.raises(InvalidRequestError) as exc:
        SwitchbackTrajectoryBaseline(**model_args(a), profiles=(((1, 0),),))
    assert exc.value.code == "power.switchback.baseline"
    with pytest.raises(InvalidRequestError) as exc:
        switchback_profile_plan(b, procedure(a), sizes=[2**50], effects=[0])
    assert exc.value.code == "power.switchback.n"
    with pytest.raises(InvalidRequestError) as exc:
        SwitchbackBaseline(
            assignment=a,
            metric="outcome",
            control_group="C",
            treatment_group="T",
            aggregation="sum",
            estimand="retained_window_total_difference",
            shared_roster=("a",),
            delta_ref=0,
            sd_a=1,
            sd_g=0,
            rho=0,
        )
    assert exc.value.code == "power.switchback.baseline"
