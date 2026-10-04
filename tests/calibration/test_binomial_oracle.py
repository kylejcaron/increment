"""The decimal binomial oracle that the finite-sample ceiling is validated against.

It must be right on its own terms before its answer is used to grade SciPy at arm sizes the
decimal summation of ``test_binomial_rr`` cannot reach: it is checked against exact rational
arithmetic (``n <= 200``), against that decimal summation (``n <= 10**4``), and, at a billion
trials, through identities that tie independently anchored evaluations together.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb

import pytest

from calibration import binomial_oracle as oracle
from tests.estimation.test_binomial_rr import _dec_binom_cdf

#: Relative agreement demanded of the oracle against an exact reference: far below float64's 1e-16
#: yet above the roughly 1e-49 the largest-arm anchor can keep.
EXACT = Decimal("1e-45")

RATES = (1e-6, 0.001, 0.3, 0.5, 0.9999, 1 - 1e-12)


def _relative(got: Decimal, ref: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return abs(got - ref) / abs(ref)


def _decimal(value: Fraction) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return Decimal(value.numerator) / Decimal(value.denominator)


class TestAgainstExactReferences:
    @pytest.mark.parametrize("n", [1, 2, 5, 10, 63, 64, 65, 100, 200])
    @pytest.mark.parametrize("p", RATES)
    def test_pmf_equals_the_exact_rational_pmf(self, n, p):
        rate = Fraction(p)
        binomial = oracle.Binomial(n, p)
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            exact = Fraction(comb(n, k)) * rate**k * (1 - rate) ** (n - k)
            assert _relative(binomial.pmf(k), _decimal(exact)) < EXACT

    @pytest.mark.parametrize("n", [10, 100, 1_000, 10_000])
    @pytest.mark.parametrize("p", [1e-4, 0.05, 0.3, 0.5, 0.97])
    def test_cdf_and_sf_equal_the_decimal_summation(self, n, p):
        binomial = oracle.Binomial(n, p)
        for k in sorted({0, 1, 5, n // 20, n // 3, n // 2, n - 3}):
            reference = _dec_binom_cdf(k, n, p, prec=80)
            if reference == 0:
                continue
            assert _relative(binomial.cdf(k), reference) < EXACT
            with localcontext() as context:
                context.prec = 80
                complement = 1 - reference
            if complement > Decimal("1e-20"):
                assert _relative(binomial.sf(k), complement) < EXACT

    @pytest.mark.parametrize("p", [0, 1, Fraction(1, 3), Decimal("0.25")])
    def test_the_walks_agree_with_single_counts(self, p):
        n = 300
        binomial = oracle.Binomial(n, p)
        counts = [-1, 0, 1, 7, 99, 100, 101, 250, n - 1, n, n + 4]
        for single, many in (
            (binomial.cdf, binomial.cdf_many),
            (binomial.sf, binomial.sf_many),
            (binomial.pmf, binomial.pmf_many),
        ):
            got = many(counts)
            for k, value in zip(counts, got, strict=True):
                reference = single(k)
                assert value == reference or _relative(value, reference) < EXACT
        run = binomial.pmf_range(5, 40)
        for k, value in zip(range(5, 41), run, strict=True):
            reference = binomial.pmf(k)
            assert value == reference or _relative(value, reference) < EXACT


class TestDegenerateAndOutOfRange:
    def test_a_point_mass_at_zero_or_n(self):
        for p, atom in ((0, 0), (1, 10)):
            binomial = oracle.Binomial(10, p)
            assert binomial.pmf(atom) == 1
            assert binomial.cdf(atom - 1) == 0
            assert binomial.cdf(atom) == 1
            assert binomial.sf(atom - 1) == 1
            assert binomial.sf(atom) == 0

    def test_counts_outside_the_support(self):
        binomial = oracle.Binomial(50, 0.3)
        assert binomial.pmf(-1) == 0 and binomial.pmf(51) == 0
        assert binomial.cdf(-1) == 0 and binomial.cdf(50) == 1
        assert binomial.sf(-1) == 1 and binomial.sf(50) == 0

    def test_a_rate_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="p must lie"):
            oracle.Binomial(10, 1.5)


class TestAtABillionTrials:
    """Identities no anchor error survives: the cdf walked up across the mode from a lower tail
    sum, the sf summed down from the far tail, and neighbouring anchors tied by the exact ratio."""

    @pytest.mark.parametrize(
        ("n", "p"),
        [
            pytest.param(10**9, 0.05, id="billion-dense"),
            pytest.param(10**9, 1e-4, id="billion-rare"),
            pytest.param(10**8 + 7, 0.5, id="hundred-million-balanced"),
        ],
    )
    def test_cdf_walked_across_the_mode_complements_an_independent_sf(self, n, p):
        binomial = oracle.Binomial(n, p)
        sigma = int((n * p * (1 - p)) ** 0.5)
        mean = int(n * p)
        low, high = mean - 6 * sigma, mean + 3 * sigma
        walked = binomial.cdf_many([low, mean, high])
        for k, cdf in zip([low, mean, high], walked, strict=True):
            sf = binomial.sf(k)
            with localcontext() as context:
                context.prec = 80
                assert abs(cdf + sf - 1) < Decimal("1e-45"), (n, p, k)

    @pytest.mark.parametrize(("n", "p"), [(10**9, 0.05), (10**9, 1e-4), (999_999_937, 0.5)])
    def test_independent_anchors_obey_the_exact_ratio_recurrence(self, n, p):
        binomial = oracle.Binomial(n, p)
        mean = int(n * p)
        counts = [mean - 12_345, mean, mean + 12_345]
        masses = binomial.pmf_many([c for k in counts for c in (k, k + 1)])
        with localcontext() as context:
            context.prec = 80
            odds = binomial.p / (1 - binomial.p)
            for index, k in enumerate(counts):
                ratio = masses[2 * index + 1] / masses[2 * index]
                expected = Decimal(n - k) / Decimal(k + 1) * odds
                assert abs(ratio - expected) / expected < Decimal("1e-45")

    def test_a_range_of_probabilities_sums_to_the_cdf_difference(self):
        n, p = 10**9, 0.05
        binomial = oracle.Binomial(n, p)
        lo = int(n * p) - 2_000
        hi = lo + 4_000
        probabilities = binomial.pmf_range(lo, hi)
        with localcontext() as context:
            context.prec = 80
            mass = sum(probabilities)
            difference = binomial.cdf(hi) - binomial.cdf(lo - 1)
            assert abs(mass - difference) / difference < Decimal("1e-45")


class TestUlpDistance:
    def test_a_representable_value_is_zero_ulps_away(self):
        assert oracle.ulp_distance(0.25, Decimal("0.25")) == 0.0

    def test_one_neighbouring_float_is_one_ulp_away(self):
        import math

        exact = Decimal(0.3)
        assert oracle.ulp_distance(math.nextafter(0.3, 1.0), exact) == pytest.approx(1.0)

    def test_the_spacing_of_a_subnormal_is_the_smallest_one(self):
        assert oracle.ulp_distance(2.0**-1074, Decimal(0)) == pytest.approx(1.0)
