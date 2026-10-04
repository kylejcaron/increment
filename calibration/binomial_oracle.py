"""SciPy-independent binomial pmf, cdf and sf in decimal arithmetic, at any arm size.

``pmf(k) = C(n, k) p**k (1 - p)**(n - k)`` is anchored at one count through ``ln n!``: exact
factorials below ``_STIRLING_FROM``, Stirling's series above it, summed until a term falls below
the working precision. Its neighbours follow from the exact ratio ``pmf(j + 1) / pmf(j) = (n - j) /
(j + 1) * p / (1 - p)``. A tail sum starts at the count nearest the tail and ends once the
geometric bound on the remainder (the ratios fall as the walk leaves the mode) is below
``10**-(prec - 8)`` of the sum, so one tail costs ``O(sqrt(n p (1 - p)))`` terms, not ``O(n)``.

Every cdf or sf is a sum of positive terms or the complement of one on the heavy side of the
mode, so its relative error stays at the working precision wherever it is small. The anchor
exponentiates terms as large as ``n ln n`` (about 2e10 at a billion), which costs about 11 of
``prec`` digits at the largest arm; the default 60 keeps at least 45 on every result.

The module shares no code with SciPy, NumPy or ``math``'s special functions: it is the reference
the production primitives and the Clopper-Pearson endpoints are measured against.
"""

from __future__ import annotations

import decimal
import math
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from decimal import Decimal
from fractions import Fraction
from functools import cache, lru_cache

#: Working precision in decimal digits.
PRECISION = 60

#: ``ln m!`` is computed exactly below this count and by Stirling's series from it: there the
#: series' smallest term is far below ``10**-PRECISION`` (it needs about 30 terms at 64).
_STIRLING_FROM = 64

#: Most Stirling terms ever summed; reaching it without converging is an error.
_STIRLING_TERMS = 80

#: Terms between checks of a tail sum's stopping bound.
_STOP_CHECK_EVERY = 16


def _context(prec: int) -> decimal.Context:
    return decimal.Context(
        prec=prec,
        Emin=decimal.MIN_EMIN,
        Emax=decimal.MAX_EMAX,
        traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
    )


@contextmanager
def _working(prec: int) -> Iterator[None]:
    with decimal.localcontext(_context(prec)):
        yield


def precise(prec: int = PRECISION) -> AbstractContextManager[None]:
    """Decimal arithmetic at the oracle's precision and exponent range, for combining its
    results (products and sums of probabilities far below the default context's range)."""
    return _working(prec)


@cache
def _bernoulli(index: int) -> Fraction:
    """Bernoulli number ``B_index`` (``B_1 = -1/2``), from the recurrence ``sum_{j<=m} C(m+1, j)
    B_j = 0``."""
    if index == 0:
        return Fraction(1)
    total = sum(
        (Fraction(math.comb(index + 1, j)) * _bernoulli(j) for j in range(index)), Fraction(0)
    )
    return -total / (index + 1)


@cache
def _pi(prec: int) -> Decimal:
    """Machin's ``16 atan(1/5) - 4 atan(1/239)`` at *prec* digits."""
    with _working(prec + 10):

        def arctan_reciprocal(x: int) -> Decimal:
            step = Decimal(x) * Decimal(x)
            power = Decimal(1) / Decimal(x)
            total = power
            denominator = 1
            while abs(power) > Decimal(10) ** -(prec + 12):
                power = -power / step
                denominator += 2
                total += power / denominator
            return total

        return 16 * arctan_reciprocal(5) - 4 * arctan_reciprocal(239)


@cache
def _stirling_coefficients(prec: int) -> tuple[Decimal, ...]:
    """``B_2j / (2j (2j - 1))`` for ``j = 1..``, at *prec* digits."""
    with _working(prec):
        out = []
        for j in range(1, _STIRLING_TERMS + 1):
            b = _bernoulli(2 * j)
            out.append(Decimal(b.numerator) / Decimal(b.denominator) / (2 * j * (2 * j - 1)))
        return tuple(out)


@lru_cache(maxsize=65536)
def _ln_factorial(m: int, prec: int) -> Decimal:
    """``ln m!`` to about *prec* digits (absolute)."""
    with _working(prec):
        if m < _STIRLING_FROM:
            return Decimal(math.factorial(m)).ln()
        x = Decimal(m)
        total = x * x.ln() - x + (2 * _pi(prec) * x).ln() / 2
        inverse = 1 / x
        inverse_square = inverse * inverse
        power = inverse
        floor = Decimal(10) ** -prec
        for coefficient in _stirling_coefficients(prec):
            term = coefficient * power
            total += term
            if abs(term) < floor:
                return total
            power *= inverse_square
        raise ValueError(f"Stirling series for ln {m}! did not converge in {_STIRLING_TERMS} terms")


def ulp_distance(computed: float, exact: Decimal) -> float:
    """``|computed - exact|`` in units of the float64 spacing at *exact*: the spacing at its
    nearest float, or the smallest subnormal spacing where that float is subnormal or zero."""
    nearest = float(exact)
    spacing = math.ulp(nearest) if abs(nearest) >= 2.0**-1022 else 2.0**-1074
    with _working(PRECISION):
        return float(abs(Decimal(computed) - exact) / Decimal(spacing))


class Binomial:
    """``Bin(n, p)`` with ``p`` taken exactly (a float is its exact binary value).

    ``pmf``, ``cdf`` and ``sf`` are single counts; ``pmf_many``, ``cdf_many`` and ``sf_many``
    answer a set of counts in one walk, with a cdf accumulated upward and an sf downward so that
    a small value is a sum of positive terms.
    """

    def __init__(self, n: int, p: float | Decimal | Fraction, prec: int = PRECISION) -> None:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.n = n
        self.prec = prec
        with _working(prec):
            if isinstance(p, Fraction):
                self.p = Decimal(p.numerator) / Decimal(p.denominator)
            else:
                self.p = Decimal(p)
            if not (0 <= self.p <= 1):
                raise ValueError(f"p must lie in [0, 1], got {p}")
            self._interior = 0 < self.p < 1
            if self._interior:
                self._q = 1 - self.p
                self._odds = self.p / self._q
                self._inverse_odds = self._q / self.p
                self._ln_p = self.p.ln()
                self._ln_q = self._q.ln()
                self.mode = min(n, int((n + 1) * self.p))
            else:
                self.mode = 0 if self.p == 0 else n

    # -- anchors and steps ---------------------------------------------------------------------

    def log_pmf(self, k: int) -> Decimal:
        """``ln pmf(k)`` for ``0 <= k <= n`` and ``0 < p < 1``."""
        n, prec = self.n, self.prec
        with _working(prec):
            return (
                _ln_factorial(n, prec)
                - _ln_factorial(k, prec)
                - _ln_factorial(n - k, prec)
                + k * self._ln_p
                + (n - k) * self._ln_q
            )

    def pmf(self, k: int) -> Decimal:
        if not 0 <= k <= self.n:
            return Decimal(0)
        with _working(self.prec):
            if not self._interior:
                return Decimal(int(k == self.mode))
            return self.log_pmf(k).exp()

    def _up(self, j: int) -> Decimal:
        """``pmf(j + 1) / pmf(j)``."""
        return Decimal(self.n - j) * self._odds / Decimal(j + 1)

    def _down(self, j: int) -> Decimal:
        """``pmf(j - 1) / pmf(j)``."""
        return Decimal(j) * self._inverse_odds / Decimal(self.n - j + 1)

    def _tolerance(self) -> Decimal:
        return Decimal(10) ** -(self.prec - 8)

    # -- tails ---------------------------------------------------------------------------------

    def _lower_sum(self, k: int) -> Decimal:
        """``sum_{j <= k} pmf(j)`` for ``0 <= k <= mode``: terms fall as the walk descends."""
        with _working(self.prec):
            term = self.pmf(k)
            total = term
            tolerance = self._tolerance()
            j, since = k, 0
            while j > 0:
                ratio = self._down(j)
                term *= ratio
                total += term
                j -= 1
                since += 1
                if since == _STOP_CHECK_EVERY:
                    since = 0
                    ahead = self._down(j) if j > 0 else Decimal(0)
                    if ahead < 1 and term * ahead <= total * tolerance * (1 - ahead):
                        break
            return total

    def _upper_sum(self, k: int) -> Decimal:
        """``sum_{j >= k} pmf(j)`` for ``mode <= k <= n``: terms fall as the walk ascends."""
        with _working(self.prec):
            term = self.pmf(k)
            total = term
            tolerance = self._tolerance()
            j, since = k, 0
            while j < self.n:
                ratio = self._up(j)
                term *= ratio
                total += term
                j += 1
                since += 1
                if since == _STOP_CHECK_EVERY:
                    since = 0
                    ahead = self._up(j) if j < self.n else Decimal(0)
                    if ahead < 1 and term * ahead <= total * tolerance * (1 - ahead):
                        break
            return total

    def cdf(self, k: int) -> Decimal:
        """``P(X <= k)``."""
        if k < 0:
            return Decimal(0)
        if k >= self.n:
            return Decimal(1)
        with _working(self.prec):
            if not self._interior:
                return Decimal(int(k >= self.mode))
            if k <= self.mode:
                return self._lower_sum(k)
            return 1 - self._upper_sum(k + 1)

    def sf(self, k: int) -> Decimal:
        """``P(X > k)``."""
        if k < 0:
            return Decimal(1)
        if k >= self.n:
            return Decimal(0)
        with _working(self.prec):
            if not self._interior:
                return Decimal(int(k < self.mode))
            if k >= self.mode:
                return self._upper_sum(k + 1)
            return 1 - self._lower_sum(k)

    # -- sets of counts ------------------------------------------------------------------------

    def pmf_range(self, lo: int, hi: int) -> list[Decimal]:
        """``pmf(k)`` for ``k = lo..hi`` (``0 <= lo <= hi <= n``): one anchor at *lo*, then the
        ratio recurrence, whose error is below ``(hi - lo) * 10**-prec`` relative."""
        if not 0 <= lo <= hi <= self.n:
            raise ValueError(f"need 0 <= lo <= hi <= n, got {lo}, {hi}, {self.n}")
        with _working(self.prec):
            term = self.pmf(lo)
            out = [term]
            if not self._interior:
                return out + [Decimal(int(k == self.mode)) for k in range(lo + 1, hi + 1)]
            for j in range(lo, hi):
                term *= self._up(j)
                out.append(term)
            return out

    def pmf_many(self, counts: Sequence[int]) -> list[Decimal]:
        """Each count anchored on its own through ``ln n!``, with no recurrence between them."""
        return [self.pmf(int(k)) for k in counts]

    def cdf_many(self, counts: Sequence[int]) -> list[Decimal]:
        """``P(X <= k)`` for every count in *counts*, in one upward walk."""
        wanted = sorted({int(k) for k in counts})
        values: dict[int, Decimal] = {}
        inside = [k for k in wanted if 0 <= k < self.n]
        for k in wanted:
            if k < 0:
                values[k] = Decimal(0)
            elif k >= self.n:
                values[k] = Decimal(1)
        if inside and not self._interior:
            for k in inside:
                values[k] = self.cdf(k)
        elif inside:
            with _working(self.prec):
                start, stop = inside[0], inside[-1]
                want = set(inside)
                running = self.cdf(start)
                term = self.pmf(start)
                values[start] = running
                for j in range(start, stop):
                    term *= self._up(j)
                    running += term
                    if j + 1 in want:
                        values[j + 1] = running
        return [values[int(k)] for k in counts]

    def sf_many(self, counts: Sequence[int]) -> list[Decimal]:
        """``P(X > k)`` for every count in *counts*, in one downward walk."""
        wanted = sorted({int(k) for k in counts})
        values: dict[int, Decimal] = {}
        inside = [k for k in wanted if 0 <= k < self.n]
        for k in wanted:
            if k < 0:
                values[k] = Decimal(1)
            elif k >= self.n:
                values[k] = Decimal(0)
        if inside and not self._interior:
            for k in inside:
                values[k] = self.sf(k)
        elif inside:
            with _working(self.prec):
                start, stop = inside[-1], inside[0]
                want = set(inside)
                running = self.sf(start)
                term = self.pmf(start)
                values[start] = running
                for j in range(start, stop, -1):
                    # sf(j - 1) = sf(j) + pmf(j)
                    running += term
                    term *= self._down(j)
                    if j - 1 in want:
                        values[j - 1] = running
        return [values[int(k)] for k in counts]


def pmf(k: int, n: int, p: float | Decimal | Fraction, prec: int = PRECISION) -> Decimal:
    return Binomial(n, p, prec).pmf(k)


def cdf(k: int, n: int, p: float | Decimal | Fraction, prec: int = PRECISION) -> Decimal:
    return Binomial(n, p, prec).cdf(k)


def sf(k: int, n: int, p: float | Decimal | Fraction, prec: int = PRECISION) -> Decimal:
    return Binomial(n, p, prec).sf(k)
