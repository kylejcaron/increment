"""Unit tests for the shared tail-quantile helper."""

from __future__ import annotations

import math
import sys

import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import norm, t

from increment.errors import InvalidRequestError
from increment.estimation._tails import (
    student_t_isf,
    tail_isf,
    two_sided_critical_value,
    wald_bounds,
)


class TestStudentTIsf:
    # Independent 100-digit references, using the inputs' exact binary values.
    @pytest.mark.parametrize(
        ("dof", "tail", "expected"),
        [
            (98.0, 2.0**-1023, 13297.025372284562011807414100279),
            (10.0, 5e-301, 2.7485906095604865904463218957406e30),
            (1.5, math.ulp(0.0), 1.7992986474453173449746961726197e215),
            (2.0, math.ulp(0.0), 3.1812124520951961905648458434438e161),
            (10.0, math.ulp(0.0), 5.4907110967913065254269244910453e32),
            (98.0, math.ulp(0.0), 19072.737246712551996256737873061),
            (1e12, math.ulp(0.0), 38.467405631384415256558024373297),
            (1e12, 0.025, 1.959963984542426483009885579427),
            (1.0, 2e-309, 1.5915494309189542883298929529700e308),
            (1e-17, math.nextafter(0.5, 0.0), 1.048593924255924070818776e-4),
        ],
    )
    def test_independent_extreme_tail_reference(self, dof, tail, expected):
        assert student_t_isf(tail, dof) == pytest.approx(expected, rel=5e-13, abs=0.0)

    @pytest.mark.parametrize(
        ("dof_hex", "tail_hex", "expected"),
        [
            ("0x1.6a09e667f3bccp-27", "0x1.fffffffffffffp-2", 1.0815775783049376794e-12),
            ("0x1.6a09e667f3bccp-27", "0x1.fffff94a03595p-2", 8990.355807955564083),
            ("0x1.6a09e667f3bcep-27", "0x1.fffffffffffffp-2", 1.0815775783049375096e-12),
            ("0x1.6a09e667f3bcep-27", "0x1.fffff94a03595p-2", 8990.355807955511908),
            ("0x1.5798ee2308c3ap-26", "0x1.fffffffffffffp-2", 7.8504624022493908137e-13),
            ("0x1.5798ee2308c3ap-26", "0x1.fffff94a03595p-2", 1.5575079004630596993),
        ],
    )
    def test_tiny_shape_transition_matches_independent_inversion(self, dof_hex, tail_hex, expected):
        # Independent 90-digit incomplete-beta inversion at exact binary64 inputs.
        got = student_t_isf(float.fromhex(tail_hex), float.fromhex(dof_hex))
        assert got == pytest.approx(expected, rel=5e-13, abs=0.0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("tail", "expected"),
        [
            (float.fromhex("0x0.145f306dc9c88p-1022"), math.inf),
            (float.fromhex("0x0.145f306dc9c89p-1022"), 1.7976931348623117210e308),
        ],
    )
    def test_adjacent_overflow_boundary_tails(self, tail, expected):
        got = student_t_isf(tail, 1.0)
        if math.isinf(expected):
            assert got == math.inf
        else:
            assert math.isfinite(got)
            assert got == pytest.approx(expected, rel=2e-15)

    # Independent 100-digit beta inversion at exact binary64 inputs.
    @pytest.mark.parametrize(
        ("dof", "expected"),
        [
            (0.125, (12144.298374318486712, 47.437968039823828259)),
            (1.0, (1 + math.sqrt(2), 1.0)),
            (2.0, (math.sqrt(18 / 7), math.sqrt(2 / 3))),
            (10.0, (1.2212553950039221407, 0.69981206131243162734)),
            (98.0, (1.1572085021044755353, 0.67700143878068484389)),
            (1e12, (1.1503493803766763310, 0.67448975019632707813)),
            (math.inf, (1.1503493803760081783, 0.67448975019608174320)),
        ],
    )
    def test_ordinary_and_median(self, dof, expected):
        got = [student_t_isf(tail, dof) for tail in (0.125, 0.25)]
        assert got == pytest.approx(expected, rel=2e-12, abs=0.0)
        assert student_t_isf(0.5, dof) == 0.0

    @pytest.mark.parametrize(
        ("dof", "density_inverse"),
        [(1.0, math.pi), (2.0, 2.0 * math.sqrt(2.0)), (math.inf, math.sqrt(2.0 * math.pi))],
    )
    def test_neighbor_below_median_from_exact_density(self, dof, density_inverse):
        tail = math.nextafter(0.5, 0.0)
        # The omitted cubic term is below 1e-31 relative at this probability.
        expected = (0.5 - tail) * density_inverse
        assert student_t_isf(tail, dof) == pytest.approx(expected, rel=2e-14, abs=0.0)

    @pytest.mark.parametrize("tail", [1e-153, 1e-154, 1e-155])
    def test_beta_coordinate_clamp_transition_against_cauchy_identity(self, tail):
        expected = 1.0 / math.tan(math.pi * tail)
        assert student_t_isf(tail, 1.0) == pytest.approx(expected, rel=5e-13)

    @pytest.mark.parametrize("dof", [1e16, 1e17, 1e18, 1e19, 1e20])
    def test_normal_switch_transition(self, dof):
        # At p=.025 the leading correction is (z**3+z)/(4 nu);
        # subsequent terms are below 1e-30 relative for these degrees of freedom.
        z = 1.9599639845400542355245944305206
        expected = z + (z**3 + z) / (4.0 * dof)
        assert student_t_isf(0.025, dof) == pytest.approx(expected, rel=2e-14)

    @pytest.mark.parametrize("dof", [1e-17, 0.125, 1.0, 10.0, 1e12, math.inf])
    @pytest.mark.parametrize("tail", [0.625, 0.875, math.nextafter(0.5, 1.0)])
    def test_exact_reflection(self, dof, tail):
        assert student_t_isf(tail, dof) == -student_t_isf(1.0 - tail, dof)

    @pytest.mark.parametrize("dof", [0.0, -1.0, -math.inf, math.nan])
    @pytest.mark.parametrize("tail", [0.025, 0.5, 0.975])
    def test_invalid_dof_including_median(self, dof, tail):
        assert math.isnan(student_t_isf(tail, dof))

    @pytest.mark.parametrize("tail", [0.0, 1.0, -0.1, 1.1, math.nan, math.inf])
    def test_invalid_tail(self, tail):
        assert math.isnan(student_t_isf(tail, 10.0))

    @pytest.mark.parametrize(
        ("dof", "tail"),
        [
            (1.0, 1e-309),
            (math.ulp(0.0), math.nextafter(0.5, 0.0)),
            (1e-300, 0.25),
        ],
    )
    def test_genuinely_unrepresentable(self, dof, tail):
        assert student_t_isf(tail, dof) == math.inf

    def test_subnormal_dof_valid_at_median_and_reflects_overflow(self):
        assert student_t_isf(0.5, math.ulp(0.0)) == 0.0
        assert student_t_isf(0.75, math.ulp(0.0)) == -math.inf

    @pytest.mark.parametrize("tail", [0.025, math.ulp(0.0), math.nextafter(0.5, 0.0)])
    def test_bounded_normal_limit(self, tail):
        expected = float(norm.isf(tail))
        assert student_t_isf(tail, sys.float_info.max) == expected
        assert student_t_isf(tail, math.inf) == expected

    def test_backend_failure_is_not_reported_as_overflow(self, monkeypatch):
        from increment.estimation import _student_t

        monkeypatch.setattr(_student_t.special, "ndtri", lambda _: math.nan)
        with pytest.raises(ArithmeticError):
            student_t_isf(0.025, 10.0)


class TestTwoSidedCriticalValue:
    def test_matches_ordinary_ppf_complement_at_typical_alpha(self):
        # Sanity: agrees with the old ppf(1 - alpha/2) form where that form
        # is still numerically fine (ordinary alpha, no cancellation).
        got = two_sided_critical_value(norm.isf, 0.05, what="test")
        want = float(norm.ppf(1.0 - 0.05 / 2.0))
        assert got == pytest.approx(want, rel=1e-12)

    def test_student_t_critical_value_matches_independent_reference(self):
        got = two_sided_critical_value(student_t_isf, 0.05, 10, what="test")
        # 100-digit beta inversion at the exact binary64 value of 0.05 / 2.
        assert got == pytest.approx(2.2281388519862747157, rel=1e-12)

    def test_tiny_alpha_stays_finite_where_ppf_complement_saturates(self):
        # norm.ppf(1 - alpha/2) saturates to inf once alpha/2 rounds below
        # the float64 ulp of 1.0; isf resolves it directly.
        alpha = 1e-20
        assert not (norm.ppf(1.0 - alpha / 2.0) < float("inf"))
        crit = two_sided_critical_value(norm.isf, alpha, what="test")
        assert crit == pytest.approx(9.33604484923406, rel=1e-9)

    def test_t_tiny_alpha_stays_finite_where_ppf_complement_saturates(self):
        alpha = 1e-20
        assert not (t.ppf(1.0 - alpha / 2.0, 10) < float("inf"))
        crit = two_sided_critical_value(student_t_isf, alpha, 10, what="test")
        assert crit == pytest.approx(274.8423853162235, rel=1e-9)

    def test_refuses_alpha_outside_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.1, float("nan")):
            with pytest.raises(InvalidRequestError) as exc_info:
                two_sided_critical_value(norm.isf, bad, what="test")
            assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_student_t_resolves_finite_extreme_quantile(self):
        got = two_sided_critical_value(student_t_isf, 1e-300, 10.0, what="test")
        assert got == pytest.approx(2.7485906095604865904463218957406e30, rel=5e-13)

    def test_refuses_actual_alpha_halving_underflow(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            two_sided_critical_value(student_t_isf, math.ulp(0.0), 10.0, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"


class TestTailIsf:
    def test_direct_tail_form_for_a_beta_quantile(self):
        # diagnostics.py's use case: a direct upper-tail bound, not a
        # symmetric +/- critical value.
        frozen = beta_dist(3.0, 5.0)
        got = tail_isf(frozen.isf, 0.025, what="test")
        want = float(frozen.ppf(0.975))
        assert got == pytest.approx(want, rel=1e-9)

    def test_refuses_tail_outside_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.1, float("nan")):
            with pytest.raises(InvalidRequestError) as exc_info:
                tail_isf(norm.isf, bad, what="test")
            assert exc_info.value.code == "estimation.tails.unresolvable"

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_refuses_non_finite_result(self, value):
        with pytest.raises(InvalidRequestError) as exc_info:
            tail_isf(lambda _: value, 0.025, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_student_t_resolves_finite_extreme_quantile(self):
        got = tail_isf(student_t_isf, 5e-301, 10.0, what="test")
        assert got == pytest.approx(2.7485906095604865904463218957406e30, rel=5e-13)

    def test_student_t_refuses_genuine_overflow(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            tail_isf(student_t_isf, 1e-309, 1.0, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"


class TestCodedRefusal:
    def test_tail_isf_refuses_with_code(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            tail_isf(norm.isf, 0.0, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_two_sided_critical_value_refuses_with_code(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            two_sided_critical_value(norm.isf, 1.5, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_wald_bounds_refuses_on_overflow_with_code(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            wald_bounds(0.0, 1e300, 1e300, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"


class TestResolvableExpm1:
    def test_overflow_refuses_with_code(self):
        from increment.estimation._tails import resolvable_expm1

        with pytest.raises(InvalidRequestError) as exc_info:
            resolvable_expm1(6596.4, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_underflow_to_floor_refuses_with_code(self):
        from increment.estimation._tails import resolvable_expm1

        with pytest.raises(InvalidRequestError) as exc_info:
            resolvable_expm1(-370.9, what="test")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_ordinary_value_passes_through(self):
        from increment.estimation.inference import Normal, infer_lift

        est = infer_lift(
            metric="revenue",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5,
            se_t=0.1,
            se_c=0.1,
            prior=Normal(mu=0.0, sigma=1e6),  # diffuse: no shrinkage of the 0.5
        )
        assert est.require_lift().value == pytest.approx(math.expm1(0.5))
