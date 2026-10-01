"""The certified boundary-crossing evaluator behind sequential power.

Every enclosure is checked against a reference that shares no code with
the evaluator: exact tails for one look, an adaptive conditional-normal
integral for two, and a Simpson-rule recursion for more. The node floor,
the cheap union bounds, the drift derivative and the second-order box bound
consumed by the inverse are checked here as well.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.stats import norm

from increment.errors import InvalidRequestError
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import sequential
from increment.power.sequential import (
    NODES_MAX,
    NODES_MIN,
    PANEL_NODES,
    SLOPE_CAP,
    _certified_crossing,
    _cheap_bounds,
    _crossing_enclosure,
    _legendre,
    _nodes_floor,
    planning_bounds,
    sequential_expected_information_fraction,
    sequential_power,
    sequential_power_enclosure,
)

from ._references import exact_one_look, quad_two_look, simpson_walk

SECOND_DERIVATIVE_CAP = 2.0 * float(norm.pdf(1.0))
SUBNORMAL = 2.0**-1022


def _fractions(looks: int) -> list[float]:
    return [k / looks for k in range(1, looks + 1)]


_SPECS = {
    "av05": GaussianScoreMixture(),
}


@functools.cache
def _design_bounds(label: str, looks: int, alpha: float = 0.05) -> tuple[list[float], list[float]]:
    fractions = _fractions(looks)
    return planning_bounds(_SPECS[label], fractions, 0.02, alpha), fractions


@functools.cache
def _simpson(label: str, looks: int, drift: float, points: int = 4001):
    bounds, fractions = _design_bounds(label, looks)
    return simpson_walk(bounds, fractions, drift, points)


class TestOneLookExactTails:
    @pytest.mark.parametrize("c", [1.96, 8.3, 37.0])
    @pytest.mark.parametrize("drift", [0.0, 2.8, 37.0])
    @pytest.mark.parametrize("side", ["upper", "both"])
    def test_exact_tail_inside_a_relatively_tight_enclosure(self, c, drift, side):
        upper, both = exact_one_look(c, drift)
        reference = upper if side == "upper" else both
        enclosure = _certified_crossing([c], [1.0], drift, side)
        assert enclosure.resolved
        assert enclosure.lower <= reference <= enclosure.upper
        # Relative accuracy beyond the subnormal floor: a 5.7e-300 tail keeps it.
        assert enclosure.half_width <= 1e-11 * enclosure.estimate + enclosure.nodes * SUBNORMAL

    def test_tiny_tail_keeps_relative_accuracy(self):
        enclosure = _certified_crossing([37.0], [1.0], 0.0, "upper")
        assert enclosure.estimate == pytest.approx(5.7255712225239233e-300, rel=1e-12)
        assert enclosure.half_width < 1e-305


class TestTwoLookQuadReference:
    @pytest.mark.parametrize(
        ("bounds", "fractions", "drift"),
        [
            ([2.5, 2.0], [0.5, 1.0], 0.0),
            ([2.5, 2.0], [0.5, 1.0], 2.8),
            ([2.5, 2.0], [0.5, 1.0], -1.0),
            ([40.0, 9.99], [0.5, 1.0], 0.0),
        ],
    )
    @pytest.mark.parametrize("side", ["upper", "both"])
    def test_reference_inside_allowing_for_the_reference_error(
        self, bounds, fractions, drift, side
    ):
        upper, both, err_upper, err_both = quad_two_look(bounds, fractions, drift)
        reference, error = (upper, err_upper) if side == "upper" else (both, err_both)
        enclosure = _certified_crossing(bounds, fractions, drift, side)
        assert enclosure.resolved
        assert enclosure.lower - error <= reference <= enclosure.upper + error

    def test_later_look_upper_tail_preserves_null_symmetry(self):
        both = _certified_crossing([40.0, 9.99], [0.5, 1.0], 0.0, "both")
        upper = _certified_crossing([40.0, 9.99], [0.5, 1.0], 0.0, "upper")
        assert upper.estimate == pytest.approx(both.estimate / 2.0, rel=1e-12, abs=0.0)

    def test_expected_fraction_matches_the_closed_form_two_look_value(self):
        # Only the first look's exit mass matters: survivors run to t=1, so
        # E[T] = t1 * P(exit at look 1) + t2 * (1 - P(exit at look 1)).
        bounds, fractions, drift = [2.5, 2.0], [0.5, 1.0], 2.8
        exit_1 = float(
            norm.sf((bounds[0] * math.sqrt(0.5) - drift * 0.5) / math.sqrt(0.5))
        ) + float(norm.sf((bounds[0] * math.sqrt(0.5) + drift * 0.5) / math.sqrt(0.5)))
        expected = 0.5 * exit_1 + 1.0 * (1.0 - exit_1)
        enclosure = _certified_crossing(bounds, fractions, drift, "both")
        assert enclosure.expected_fraction == pytest.approx(expected, abs=1e-13)
        assert enclosure.expected_fraction is not None
        assert enclosure.expected_half is not None
        assert enclosure.expected_fraction - enclosure.expected_half <= expected
        assert expected <= enclosure.expected_fraction + enclosure.expected_half

    def test_expected_fraction_near_one_includes_subtraction_rounding(self):
        enclosure = _certified_crossing([10.0, 10.0], [0.5, 1.0], 0.0, "both")
        assert enclosure.expected_fraction == 1.0
        assert enclosure.expected_converged
        assert enclosure.expected_half is not None
        assert enclosure.expected_half >= 2.0**-52


@pytest.mark.slow
class TestMultiLookSimpsonReference:
    @pytest.mark.parametrize("looks", [5, 14])
    @pytest.mark.parametrize("label", ["av05"])
    @pytest.mark.parametrize("drift", [0.0, 2.8, -3.0])
    def test_reference_inside_both_exit_sides(self, looks, label, drift):
        bounds, fractions = _design_bounds(label, looks)
        reference = _simpson(label, looks, drift)
        for side, value in (("upper", reference.upper), ("both", reference.both)):
            enclosure = _certified_crossing(bounds, fractions, drift, side)
            assert enclosure.resolved, (label, looks, drift, side)
            assert enclosure.lower <= value <= enclosure.upper, (label, looks, drift, side)

    @pytest.mark.parametrize("label", ["av05"])
    def test_expected_fraction_matches_the_simpson_per_look_masses(self, label):
        bounds, fractions = _design_bounds(label, 5)
        reference = _simpson(label, 5, 2.8)
        refined = _simpson(label, 5, 2.8, 8001)
        expected = refined.expected_fraction(fractions)
        # Simpson is only an independent reference, not an enclosure. Use
        # four times its observed refinement difference as its own allowance.
        reference_error = 4.0 * abs(expected - reference.expected_fraction(fractions))
        enclosure = _certified_crossing(bounds, fractions, 2.8, "both")
        assert enclosure.expected_fraction == pytest.approx(expected, abs=1e-10)
        assert sequential_expected_information_fraction(
            _SPECS[label], 2.8 * 0.02, 0.02, 0.05, 5
        ) == pytest.approx(enclosure.expected_fraction, abs=1e-12)
        assert enclosure.expected_fraction is not None
        assert enclosure.expected_half is not None
        assert enclosure.expected_fraction - enclosure.expected_half - reference_error <= expected
        assert expected <= (enclosure.expected_fraction + enclosure.expected_half + reference_error)

    @pytest.mark.parametrize("label", ["av05"])
    def test_irregular_fractions_resolved_with_the_reference_inside(self, label):
        fractions = [0.05, 0.5, 0.9, 0.99, 1.0]
        bounds = planning_bounds(_SPECS[label], fractions, 0.02, 0.05)
        reference = simpson_walk(bounds, fractions, 2.8, 4001)
        enclosure = _certified_crossing(bounds, fractions, 2.8, "both")
        assert enclosure.resolved
        assert enclosure.nodes >= _nodes_floor(bounds, fractions)
        assert enclosure.lower <= reference.both <= enclosure.upper

    @pytest.mark.parametrize("drift", [40.0, 37.0, 0.0])
    def test_tiny_alpha_five_looks(self, drift):
        fractions = _fractions(5)
        bounds = planning_bounds(GaussianScoreMixture(), fractions, 0.02, 1e-300)
        reference = simpson_walk(bounds, fractions, drift, 8001)
        enclosure = _certified_crossing(bounds, fractions, drift, "both")
        assert enclosure.resolved
        assert enclosure.lower <= reference.both <= enclosure.upper

    def test_tiny_alpha_fourteen_looks_keeps_a_tight_enclosure(self):
        fractions = _fractions(14)
        bounds = planning_bounds(GaussianScoreMixture(), fractions, 0.02, 1e-300)
        reference = simpson_walk(bounds, fractions, 40.0, 8001)
        enclosure = _certified_crossing(bounds, fractions, 40.0, "both")
        assert enclosure.resolved
        assert enclosure.lower <= reference.both <= enclosure.upper
        assert enclosure.half_width < 1e-9


class TestNodeFloor:
    def test_two_look_near_duplicate_fractions_resolved(self):
        fractions = [0.999, 1.0]
        bounds = planning_bounds(GaussianScoreMixture(), fractions, 0.02, 0.05)
        upper, both, err_upper, err_both = quad_two_look(bounds, fractions, 2.8)
        enclosure = _certified_crossing(bounds, fractions, 2.8, "both")
        assert enclosure.resolved
        # The second increment is a thousandth of the first look's scale, so
        # the product of the two densities needs many nodes per boundary width.
        assert enclosure.nodes >= 256
        assert enclosure.lower - err_both <= both <= enclosure.upper + err_both

    def test_floor_is_a_power_of_two_at_least_the_minimum(self):
        bounds, fractions = _design_bounds("av05", 5)
        floor = _nodes_floor(bounds, fractions)
        assert floor >= NODES_MIN
        assert floor & (floor - 1) == 0

    def test_not_resolved_below_the_floor(self):
        fractions = _fractions(14)
        bounds = planning_bounds(GaussianScoreMixture(), fractions, 0.02, 1e-300)
        floor = _nodes_floor(bounds, fractions)
        assert 4 * NODES_MIN < floor <= NODES_MAX
        below = _crossing_enclosure(bounds, fractions, 40.0, "both", floor // 2)
        assert not below.resolved
        at_floor = _certified_crossing(bounds, fractions, 40.0, "both")
        assert at_floor.resolved
        assert at_floor.nodes >= floor

    def test_boundary_above_the_ceiling_gets_the_cheap_enclosure(self):
        # A score boundary of 2000 needs more than NODES_MAX nodes per look.
        enclosure = _certified_crossing([2000.0], [1.0], 2000.0, "upper")
        assert not enclosure.resolved
        assert enclosure.lower <= 0.5 <= enclosure.upper
        assert enclosure.upper - enclosure.lower <= 1e-12

    def test_unresolved_ceiling_refuses_a_point_estimate(self, monkeypatch):
        monkeypatch.setattr(sequential, "NODES_MAX", 64)
        enclosure = sequential_power_enclosure(
            GaussianScoreMixture(), 40.0 * 0.02, 0.02, 1e-300, 14
        )
        assert not enclosure.resolved
        with pytest.raises(InvalidRequestError) as raised:
            sequential_power(GaussianScoreMixture(), 40.0 * 0.02, 0.02, 1e-300, 14)
        assert raised.value.code == "power.boundary_crossing_quadrature"

    def test_valid_but_unconverged_node_ceiling_refuses_a_point(self, monkeypatch):
        monkeypatch.setattr(sequential, "NODES_MAX", 64)
        enclosure = sequential_power_enclosure(GaussianScoreMixture(), 0.05, 0.02, 0.05, 5)
        assert enclosure.resolved
        assert not enclosure.converged
        with pytest.raises(InvalidRequestError) as raised:
            sequential_power(GaussianScoreMixture(), 0.05, 0.02, 0.05, 5)
        assert raised.value.code == "power.boundary_crossing_quadrature"

    def test_large_schedule_is_refused_before_dense_quadrature(self):
        # Reaching the work ceiling before any walk leaves the cheap union
        # bound; a dense walk would produce a resolved enclosure instead.
        enclosure = sequential_power_enclosure(GaussianScoreMixture(), 0.05, 0.02, 0.05, 1000)
        assert not enclosure.resolved
        with pytest.raises(InvalidRequestError) as raised:
            sequential_power(GaussianScoreMixture(), 0.05, 0.02, 0.05, 1000)
        assert raised.value.code == "power.boundary_crossing_quadrature"


class TestLegendreNodeCount:
    def test_rejects_a_node_count_not_a_multiple_of_the_panel_size(self):
        with pytest.raises(InvalidRequestError) as raised:
            _legendre(2 * PANEL_NODES + 1)
        assert raised.value.code == "power.nodes_multiple"


class TestCheapBounds:
    @pytest.mark.parametrize("side", ["upper", "both"])
    def test_first_look_exit_and_union_bound_bracket_the_reference(self, side):
        bounds, fractions = _design_bounds("av05", 5)
        upper, both, _, _ = quad_two_look(bounds[:2], [0.2, 0.4], 2.8)
        lower, top = _cheap_bounds(bounds[:2], [0.2, 0.4], 2.8, side)
        reference = upper if side == "upper" else both
        assert lower <= reference <= top
        assert lower > 0.0
        assert top <= 1.0


class TestSlope:
    def test_two_look_slope_matches_the_reference_finite_difference(self):
        bounds, fractions, drift = [2.5, 2.0], [0.5, 1.0], 2.8
        h = 1e-4
        for side, index in (("both", 1), ("upper", 0)):
            plus = quad_two_look(bounds, fractions, drift + h)[index]
            minus = quad_two_look(bounds, fractions, drift - h)[index]
            enclosure = _certified_crossing(bounds, fractions, drift, side)
            assert abs(enclosure.slope - (plus - minus) / (2.0 * h)) < 1e-8
            assert abs(enclosure.slope) <= SLOPE_CAP
            assert enclosure.slope_half < 1e-9

    @pytest.mark.slow
    @pytest.mark.parametrize("looks", [5, 14])
    @pytest.mark.parametrize("label", ["av05"])
    @pytest.mark.parametrize("drift", [0.3, 2.8])
    def test_multi_look_slope_matches_the_simpson_finite_difference(self, looks, label, drift):
        bounds, fractions = _design_bounds(label, looks)
        h = 1e-4
        plus = _simpson(label, looks, drift + h)
        minus = _simpson(label, looks, drift - h)
        for side in ("both", "upper"):
            enclosure = _certified_crossing(bounds, fractions, drift, side)
            forward = getattr(plus, side)
            backward = getattr(minus, side)
            assert abs(enclosure.slope - (forward - backward) / (2.0 * h)) < 1e-8


def _box_bound(bounds, fractions, drift, half_drift, half_boundary, side):
    """The inverse's variation bound around the midpoint configuration:
    certified slope times the drift half-range, the universal second
    derivative cap, and the same-path band term for the boundary range."""
    enclosure = _certified_crossing(bounds, fractions, drift, side)
    band = 0.0
    for c, t, half_c in zip(bounds, fractions, half_boundary, strict=True):
        if half_c == 0.0:
            continue
        slack = half_c + half_drift * math.sqrt(t)
        mu = drift * math.sqrt(t)
        band += half_c * (
            float(norm.pdf(max(0.0, abs(c - mu) - slack)))
            + float(norm.pdf(max(0.0, abs(c + mu) - slack)))
        )
    drift_term = (abs(enclosure.slope) + enclosure.slope_half) * half_drift
    drift_term += 0.5 * SECOND_DERIVATIVE_CAP * half_drift * half_drift
    return enclosure, drift_term + band


@pytest.mark.slow
class TestBoxVariationBound:
    """Every configuration in a drift/boundary box stays within the second-
    order bound of the midpoint evaluation (corners plus random interior
    points, measured with the Simpson reference)."""

    @staticmethod
    def _check(bounds, fractions, drift, half_drift, half_boundary, side="both", points=2001):
        enclosure, bound = _box_bound(bounds, fractions, drift, half_drift, half_boundary, side)
        rng = np.random.default_rng(7)
        configs = [
            (
                drift + sign_d * half_drift,
                [c + sign_b * h for c, h in zip(bounds, half_boundary, strict=True)],
            )
            for sign_d in (-1.0, 1.0)
            for sign_b in (-1.0, 1.0)
        ]
        for _ in range(8):
            sign_d = rng.choice([-1.0, 1.0, rng.uniform(-1.0, 1.0)])
            configs.append(
                (
                    drift + sign_d * half_drift,
                    [
                        c + rng.uniform(-1.0, 1.0) * h
                        for c, h in zip(bounds, half_boundary, strict=True)
                    ],
                )
            )
        worst = 0.0
        for moved_drift, moved_bounds in configs:
            reference = simpson_walk(moved_bounds, fractions, moved_drift, points)
            value = reference.upper if side == "upper" else reference.both
            worst = max(worst, abs(value - enclosure.estimate))
        assert worst <= bound + enclosure.half_width
        return worst / bound

    @pytest.mark.parametrize(
        ("half_drift", "half_boundary"), [(0.05, 0.0), (0.01, 0.02), (0.2, 0.05), (0.5, 0.0)]
    )
    def test_always_valid_fourteen_looks_at_drift_2_8(self, half_drift, half_boundary):
        bounds, fractions = _design_bounds("av05", 14)
        ratio = self._check(bounds, fractions, 2.8, half_drift, [half_boundary] * 14)
        assert ratio <= 1.0

    @pytest.mark.parametrize(("drift", "half_drift"), [(40.0, 0.1), (36.0, 0.5)])
    def test_tiny_alpha_boxes(self, drift, half_drift):
        fractions = _fractions(5)
        bounds = planning_bounds(GaussianScoreMixture(), fractions, 0.02, 1e-300)
        ratio = self._check(bounds, fractions, drift, half_drift, [0.05] * 5)
        assert ratio <= 1.0


class TestPublicEvaluators:
    @pytest.mark.parametrize(
        "spec",
        [GaussianScoreMixture()],
    )
    def test_default_schedule_is_resolved(self, spec):
        enclosure = sequential_power_enclosure(spec, 0.05, 0.02, 0.05, 14)
        assert enclosure.resolved
        assert enclosure.converged
        assert enclosure.expected_converged
        assert enclosure.levels is not None
        assert enclosure.nodes >= 4 * NODES_MIN
        assert sequential_power(spec, 0.05, 0.02, 0.05, 14) == enclosure.estimate
        assert sequential_expected_information_fraction(spec, 0.05, 0.02, 0.05, 14) == (
            enclosure.expected_fraction
        )

    def test_upper_exit_uses_the_narrower_one_sided_boundary(self):
        """GaussianScoreMixture doubles alpha internally for a one-sided
        exit (see ``_always_valid_z_bound``), so the one-sided boundary is
        narrower than the two-sided one and its directional power exceeds
        the two-sided crossing probability at the same nominal alpha."""
        both = sequential_power(GaussianScoreMixture(), 0.05, 0.02, 0.05, 5)
        upper = sequential_power(GaussianScoreMixture(), 0.05, 0.02, 0.05, 5, exit_side="upper")
        assert 0.0 < both < upper <= 1.0

    @pytest.mark.parametrize("alpha", [0.0, 1.0, math.nan, math.inf])
    def test_log_scale_boundary_rejects_invalid_alpha(self, alpha):
        with pytest.raises(InvalidRequestError) as raised:
            planning_bounds(GaussianScoreMixture(), [0.5, 1.0], 0.1, alpha)
        assert raised.value.code == "power.alpha_finite"

    def test_single_look_expected_fraction_is_one(self):
        assert (
            sequential_expected_information_fraction(GaussianScoreMixture(), 0.05, 0.02, 0.05, 1)
            == 1.0
        )

    def test_finite_negative_drift_saturates_at_the_first_lower_exit(self):
        fractions = [0.5, 1.0]
        both = _certified_crossing([2.0, 2.0], fractions, -1e308, "both")
        upper = _certified_crossing([2.0, 2.0], fractions, -1e308, "upper")
        assert both.converged and both.lower >= 1.0 - 1e-13
        assert upper.converged and upper.estimate == 0.0
        assert upper.upper <= 1e-13
        assert both.expected_converged
        assert both.expected_fraction == fractions[0]
        assert both.expected_half is not None and both.expected_half >= 0.0
