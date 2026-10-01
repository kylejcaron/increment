"""Scalar Student-t inversion, including tails outside inverse-beta resolution."""

from __future__ import annotations

import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

from scipy import special

_U = 2.0**-53
_LOG_TWO = math.log(2.0)
_LOG_MAX = math.log(sys.float_info.max)
_LOG_TWO_PI = math.log(2.0 * math.pi)


def student_t_isf(tail: float, dof: float) -> float:
    """Invert a scalar upper tail; invalid inputs return NaN, true overflow infinity.

    A mathematical quantile above the largest finite float returns infinity,
    even if round-to-nearest would saturate to that maximum. A numerical
    backend or convergence failure raises ArithmeticError instead.
    """
    if not (0.0 < tail < 1.0) or not (dof > 0.0):
        return math.nan
    if tail == 0.5:
        return 0.0
    if tail > 0.5:
        # Sterbenz's lemma makes this complement exact for binary64 inputs.
        return -student_t_isf(1.0 - tail, dof)
    z = -float(special.ndtri(tail))
    if not math.isfinite(z) or z <= 0.0:
        raise ArithmeticError("Student-t inversion: Normal bracket failed")
    if math.isinf(dof):
        return z
    if dof > 0.5:
        normal_error = 0.5 * (-math.log1p(-0.5 / dof) + z * (z / (dof - 0.5)))
        if normal_error <= _U / 4.0:
            return z

    # Neither tiny beta shapes nor subnormal probabilities are reliable in Boost.
    if dof > math.sqrt(_U) and 2.0 * tail >= sys.float_info.min:
        a = dof / 2.0
        x = float(special.betaincinv(a, 0.5, 2.0 * tail))
        y = float(special.betainccinv(0.5, a, 2.0 * tail))
        if sys.float_info.min < x <= 1.0 and sys.float_info.min < y <= 1.0:
            q = math.sqrt(dof) * math.sqrt(y) / math.sqrt(x)
            if math.isfinite(q) and q > 0.0:
                return q

    log_dof = math.log(dof)
    log_twop = math.log(2.0 * tail)
    g = _log_a_beta_over_dof(dof)
    leading = 0.5 * log_dof - (log_twop / dof + g)
    log_x0 = log_dof - 2.0 * leading
    if log_x0 < 0.0:
        x0 = math.exp(log_x0)
        if x0 < 1.0 and -0.5 * math.log1p(-x0) <= _U:
            return _leading_quantile(tail, dof, leading)
    if dof <= math.sqrt(_U):
        return _small_dof_quantile(tail, dof, g)
    return _invert_log_tail(tail, dof, z, leading)


def _log_a_beta_over_dof(dof: float) -> float:
    """log[(nu/2) B(nu/2, 1/2)] / nu, without small-nu cancellation."""
    if dof <= 0.25:
        total = _LOG_TWO
        power = dof
        for k in range(2, 64):
            term = (1.0 - 2.0 ** (1 - k)) * float(special.zeta(k, 1.0)) * power / k
            total += -term if k % 2 == 0 else term
            power *= dof
            # The alternating remainder is below 2 nu**k / (k+1).
            if 2.0 * power / (k + 1) <= _U * abs(total) / 4.0:
                return total
        raise ArithmeticError("Student-t normalization series did not converge")
    return (_log_dof_beta(dof) - _LOG_TWO) / dof


def _log_dof_beta(dof: float) -> float:
    if dof >= 128.0:
        inv = 1.0 / dof
        inv2 = inv * inv
        correction = inv * (0.25 + inv2 * (-1.0 / 24.0 + inv2 * (1.0 / 20.0 - inv2 * 17.0 / 112.0)))
        # The next gamma-ratio term is bounded by 31/(36 nu**9).
        return 0.5 * (_LOG_TWO_PI + math.log(dof)) + correction
    return math.log(dof) + float(special.betaln(dof / 2.0, 0.5))


def _exp_split(value: float) -> float:
    exponent = math.floor(value / _LOG_TWO)
    # Splitting ln(2) keeps the integer multiplication from losing the residual.
    residual = (value - exponent * 0.6931471803691238) - exponent * 1.9082149292705877e-10
    return math.ldexp(math.exp(residual), exponent)


def _leading_quantile(tail: float, dof: float, leading: float) -> float:
    # This margin covers binary64 log arithmetic, not a probability/domain floor.
    if abs(leading - _LOG_MAX) <= 1e-10:
        return _boundary_quantile(tail, dof)
    if leading > _LOG_MAX:
        return math.inf
    return _exp_split(leading)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _log_tail_bounds(v: float, dof: float) -> tuple[float, float, float]:
    """Alternating integral bounds on log S(exp(v)), plus its log derivative."""
    log_dof = math.log(dof)
    log_x = -_softplus(2.0 * v - log_dof)
    log_y = -_softplus(log_dof - 2.0 * v)
    log_power = (dof / 2.0) * log_x
    base = log_power - _log_dof_beta(dof) - 0.5 * log_y
    inverse_square = math.exp(-2.0 * v)
    total = 1.0
    term = 1.0
    lower = 0.0
    upper = 1.0
    for k in range(1, 128):
        next_term = term * ((2.0 * k - 1.0) * inverse_square) * (dof / (dof + 2.0 * k))
        if next_term >= term:
            raise ArithmeticError("Student-t tail integral bounds did not contract")
        term = next_term
        total += -term if k % 2 else term
        if k % 2:
            lower = total
        else:
            upper = total
        if lower > 0.0 and upper - lower <= 4.0 * _U * lower:
            h = (lower + upper) / 2.0
            # Include arithmetic rounding separately from the integral remainder.
            rounding = 8.0 * _U * (abs(base) + abs(log_y) + k + 1.0)
            rounding += 2.0 * _U * (abs(log_dof) + abs(2.0 * v - log_dof)) * (abs(log_power) + 1.0)
            lo = base + math.log(lower) - rounding
            hi = base + math.log(upper) + rounding
            derivative = -dof * math.exp(log_y) / h
            return lo, hi, derivative
    raise ArithmeticError("Student-t tail integral bounds did not converge")


def _invert_log_tail(tail: float, dof: float, z: float, leading: float) -> float:
    target = math.log(tail)
    target_error = _U * abs(target)
    log_z = math.log(z)
    lower = log_z - 8.0 * _U * max(1.0, abs(log_z))
    upper = min(leading + 8.0 * _U * max(1.0, abs(leading)), _LOG_MAX)
    v = min(upper, max(lower, math.log(z) + 0.5))
    for _ in range(128):
        lo, hi, derivative = _log_tail_bounds(v, dof)
        if not (math.isfinite(lo) and math.isfinite(hi) and derivative < 0.0):
            raise ArithmeticError("Student-t log-tail evaluation failed")
        # |d log S / dv| increases with q; nu*y is a lower bound because H<=1.
        log_y_lower = -_softplus(math.log(dof) - 2.0 * lower)
        slope_floor = dof * math.exp(log_y_lower)
        if hi < target - target_error:
            upper = v
        elif lo > target + target_error:
            lower = v
        else:
            radius = (max(target - lo, hi - target) + target_error) / slope_floor
            lower = max(lower, v - radius)
            upper = min(upper, v + radius)
        tolerance = max(
            8.0 * _U * max(1.0, abs(v)),
            4.0 * (hi - lo + 2.0 * target_error) / slope_floor,
        )
        if upper - lower <= tolerance:
            return _exp_split((lower + upper) / 2.0)
        proposal = v - ((lo + hi) / 2.0 - target) / derivative
        if not lower < proposal < upper or abs(proposal - v) <= math.ulp(v):
            proposal = (lower + upper) / 2.0
        v = proposal
    raise ArithmeticError("Student-t log-tail inversion did not converge")


def _log_cosh(value: float) -> float:
    if value < 1.0:
        return math.log1p(2.0 * math.sinh(value / 2.0) ** 2)
    return value + math.log1p(math.exp(-2.0 * value)) - _LOG_TWO


def _small_dof_quantile(tail: float, dof: float, g: float) -> float:
    from scipy.integrate import quad

    mass = 1.0 - 2.0 * tail
    a_beta = math.exp(dof * g)
    r = (mass / dof) * a_beta
    # s>=r proves overflow without ever constructing the possibly zero nu/2.
    if r > _LOG_MAX - 0.5 * math.log(dof) + _LOG_TWO + 1.0:
        return math.inf
    lower = r
    upper = -math.log1p(-mass * a_beta) / dof
    s = (lower + upper) / 2.0
    for _ in range(64):
        correction, error = quad(
            lambda w, s=s: math.expm1(-dof * _log_cosh(s * w)),
            0.0,
            1.0,
            epsabs=_U / max(1.0, s),
            epsrel=8.0 * _U,
            limit=100,
        )
        if not math.isfinite(correction) or error * s > 8.0 * _U:
            raise ArithmeticError("Student-t small-dof quadrature failed")
        residual = (s - r) + s * correction
        uncertainty = s * error + 4.0 * _U * (abs(s - r) + abs(s * correction) + r)
        if residual > uncertainty:
            upper = s
        elif residual < -uncertainty:
            lower = s
        else:
            radius = (abs(residual) + uncertainty) / math.exp(-dof * _log_cosh(upper))
            lower = max(lower, s - radius)
            upper = min(upper, s + radius)
        if upper - lower <= 16.0 * _U * s:
            s = (lower + upper) / 2.0
            break
        proposal = s - residual / math.exp(-dof * _log_cosh(s))
        s = proposal if lower < proposal < upper else (lower + upper) / 2.0
    else:
        raise ArithmeticError("Student-t small-dof inversion did not converge")
    if s < 1.0:
        return math.sqrt(dof) * math.sinh(s)
    log_q = 0.5 * math.log(dof) + s - _LOG_TWO + math.log1p(-math.exp(-2.0 * s))
    return _leading_quantile(tail, dof, log_q)


@lru_cache(maxsize=4)
def _stirling_coefficients(count: int) -> tuple[Fraction, ...]:
    """Exact B_(2k)/(2k(2k-1)); no high-precision value of pi is needed."""
    bernoulli = [Fraction(1)]
    coefficients = []
    for n in range(1, 2 * count + 1):
        bernoulli.append(
            -sum((math.comb(n + 1, k) * bernoulli[k] for k in range(n)), Fraction(0)) / (n + 1)
        )
        if n % 2 == 0:
            coefficients.append(bernoulli[n] / (n * (n - 1)))
    return tuple(coefficients)


def _decimal_log_gamma_bounds(x: Decimal, precision: int) -> tuple[Decimal, Decimal]:
    """Bounds for log Gamma(x) - log(2 pi)/2 by shifted Stirling."""
    z = x
    product = Decimal(1)
    while z < 2 * precision:
        product *= z
        z += 1
    log_product = product.ln()
    main = (z - Decimal("0.5")) * z.ln() - z - log_product
    total = main
    magnitude = abs(main) + abs(log_product) + 1
    inverse = 1 / z
    inverse_square = inverse * inverse
    threshold = Decimal(10) ** (-precision - 4)
    for coefficient in _stirling_coefficients(precision):
        term = (Decimal(coefficient.numerator) / Decimal(coefficient.denominator)) * inverse
        if abs(term) < threshold:
            # Positive-real Stirling remainders have the next term's sign and bound.
            rounding = magnitude * Decimal(10) ** (-precision)
            return min(total, total + term) - rounding, max(total, total + term) + rounding
        total += term
        magnitude += abs(term)
        inverse *= inverse_square
    raise ArithmeticError("Student-t Decimal Stirling bounds did not converge")


def _boundary_quantile(tail: float, dof: float) -> float:
    """Refine the true max-float boundary, rather than comparing rounded logs."""
    precision = 64
    while True:
        # Thirty-two guard digits cover arithmetic in the recurrence/products;
        # returned intervals include a much larger explicit rounding allowance.
        with localcontext() as context:
            context.prec = precision + 32
            nu = Decimal.from_float(dof)
            p = Decimal.from_float(tail)
            half = Decimal("0.5")
            a = nu / 2
            g1 = _decimal_log_gamma_bounds(a + 1, precision)
            g2 = _decimal_log_gamma_bounds(Decimal(1), precision)
            g3 = _decimal_log_gamma_bounds(half, precision)
            g4 = _decimal_log_gamma_bounds(a + half, precision)
            log_a_lo = g1[0] - g2[1] + g3[0] - g4[1]
            log_a_hi = g1[1] - g2[0] + g3[1] - g4[0]
            log_nu = nu.ln()
            log_twop = (2 * p).ln()
            log_p = p.ln()
            allowance = (1 + abs(log_nu) + abs(log_twop) / nu) * Decimal(10) ** (-precision)
            lo = half * log_nu - (log_twop + log_a_hi) / nu - allowance
            hi = half * log_nu - (log_twop + log_a_lo) / nu + allowance

            # I_x(a,1/2)=x**a/A * sum[a/(a+k) * C(2k,k) * (x/4)**k].
            maximum = Decimal.from_float(sys.float_info.max)
            x = nu / (nu + maximum * maximum)
            term = Decimal(1)
            series = Decimal(1)
            k = 0
            epsilon = Decimal(10) ** (-precision)
            while True:
                k += 1
                term *= x * Decimal(2 * k - 1) / (2 * k) * (a + k - 1) / (a + k)
                series += term
                remainder = term * x / (1 - x)
                if remainder < epsilon:
                    break
            log_power = a * x.ln()
            base = log_power - Decimal(2).ln()
            rounding = epsilon * (1 + abs(base) + abs(log_power) + abs(log_p))
            tail_lo = base - log_a_hi + series.ln() - rounding
            tail_hi = base - log_a_lo + (series + remainder).ln() + rounding
            if log_p < tail_lo:
                return math.inf
            if log_p > tail_hi:
                # At max-float the q0/q correction is far below a binary64 ulp.
                value = float(((lo + hi) / 2).exp())
                if not math.isfinite(value):
                    raise ArithmeticError("Student-t finite boundary reconstruction failed")
                return value
        precision *= 2
