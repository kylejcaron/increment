"""Exact/conservative fixed-horizon two-independent-binomial risk-ratio inference.

Two independent arms ``X_c ~ Bin(n_c, q)`` (control) and ``X_t ~ Bin(n_t, p)``
(treatment), ``q > 0``, risk ratio ``R = p / q``, relative lift ``R - 1``.

This is a Berger-Boos (restricted-nuisance) test inversion: the nuisance
control rate ``q`` is confined to its own exact ``1 - beta`` Clopper-Pearson
interval (frozen, data-independent ``beta``), the joint-binomial tail is
maximized over that restricted range instead of profiled over the full unit
interval, and ``beta`` is charged once against every test's tail budget --
the Berger & Boos (1994) restricted-nuisance construction (see also
Fay & Hunsberger, https://arxiv.org/html/1904.05416v2, section 5.4, for the
ordering used here). The result is a coverage-valid (conservative) p-value
for every candidate ``R``, hence a coverage-valid confidence set by
inversion, WITHOUT plug-in nuisance estimates, pseudo-counts, or a Wald/SE
cutoff standing in for a coverage proof.

Ordering: ``K = n_c*X_t - n_t*X_c`` orders the unscaled difference between
arms directly, with no floating division: ``K >= 0`` iff ``X_t/n_t >=
X_c/n_c``, i.e. this is exactly the statistic that would test ``R = 1``.
Testing a general candidate ``r`` does NOT re-order the data by ``r``:
``k`` stays the single fixed observed value for the whole inversion, and
``r`` enters only through the nuisance nuisance parameterization ``p =
r*q`` inside the tail sums below.
``F_+(q,p) = P(K >= k)`` tests ``H0: R <= r`` (rejecting favors ``R > r``).
``F_-(q,p) = P(K <= k)`` tests ``H0: R >= r`` (rejecting favors ``R < r``).

For a candidate ``r >= 0``::

    p_+(r) = min(1, beta + sup_{q in [a,b]}           F_+(q, min(r*q, 1)))
    p_-(r) = min(1, beta + sup_{q in [a,b] cap [0,1/r]} F_-(q, r*q))

(``1/0`` is ``+inf`` by convention; an empty supremum contributes 0.) The
saturation region ``q > 1/r`` in ``p_+`` is INTENTIONALLY not excluded: the
composite null there boundary-tests at ``p = 1``.

``p_+`` is nondecreasing and ``p_-`` is nonincreasing in ``r`` (the
restricted composite nulls are nested), so for an allocation ``u``::

    {r : p_+(r) >= u}  = [r_lower, +inf)   for some r_lower >= 0
    {r : p_-(r) >= u}  = [0, r_upper]      for some r_upper in (0, +inf]

 A two-sided ``1 - alpha`` set inverts both tails at ``u = alpha/2``; a
 one-sided set inverts the relevant tail at the full ``alpha`` and leaves the
 other side at its natural parameter-space boundary (``R = 0`` for "less",
 ``R = +inf`` for "greater").

Zero-cell geometry (see the module docstring table in ``results.py`` for the
full case table): ``x_c > 0`` always yields BOTH a finite point (the
empirical ratio, or exactly ``-1`` relative lift when ``x_t == 0`` -- the
log ratio is undefined there, but the ratio itself is exactly zero) and a
finite two-sided interval (the upper endpoint search always terminates
because the Clopper-Pearson lower bound ``a > 0`` eventually empties the
restricted nuisance domain as ``r -> inf``). ``x_c == 0`` never yields a
finite point (the empirical ratio has a zero denominator) and the interval
is unbounded above (``a = 0`` keeps ``q = 0`` in the nuisance domain for
every ``r``, and ``F_-(0, 0) = 1`` identically there) -- callers must
represent that case as a typed, point-less confidence set, never as
``None/None``. ``upper is None`` is reserved EXACTLY for this ``x_c == 0``
case: the upper-endpoint search below distinguishes "the nuisance domain
provably empties beyond a finite bound" (``x_c > 0``: always a finite
answer or a coded numerical failure, never ``None``) from "the nuisance
domain never empties" (``x_c == 0``: analytically unbounded, no search
needed).

Numerical certification: the nuisance supremum is bounded, not merely
grid-searched (a grid maximum is only a LOWER bound on the true supremum;
using it as the p-value can understate the tail and invalidate coverage).
Each candidate subinterval ``[u, v]`` gets a certified UPPER bound on
``sup_{[u,v]} F`` as the tighter of:

* the coordinate-monotonicity bound (``F_+`` is nonincreasing in ``q`` and
  nondecreasing in ``p``; ``F_-`` is nondecreasing in ``q`` and
  nonincreasing in ``p``), which needs no derivative information and is
  finite even at a nuisance-interval endpoint of exactly 0 or 1;
* the quadratic Taylor-remainder bound ``max(F(u), F(v)) + I_max*(v-u)**2/4``
  from ``|F''(q)| <= 2*I(q)``, ``I(q) = n_c/(q*(1-q)) + n_t*r/(q*(1-r*q))``
  (both summands convex on their domain, so ``I_max = max(I(u), I(v))``
  exactly -- no sampling needed); this bound is used only when finite.

Branch-and-bound refines the worst (largest-bound) subinterval by
bisection, always returning the max of every remaining certified bound --
an anytime algorithm: the returned value is a valid conservative upper
bound on the supremum at every iteration, tightening (never invalidating)
with more refinement.

Every intermediate value combined into that certificate (a raw
``scipy.stats`` PMF/CDF/SF evaluation, their dot-product summation, and the
Clopper-Pearson endpoint inversion feeding the nuisance domain) is ordinary
float64 arithmetic: it is NEVER trusted bit-for-bit as an exact enclosure.
``_eps_margin``/``_round_outward`` below add a derived, regression-tested
outward safety margin so the reported certificate is a genuine upper bound
on the MATHEMATICAL supremum, not merely on whatever SciPy happened to
return -- see the "Floating-point outward-rounding certification" section.

Endpoint search: each crossing of the evaluated p-envelope is bracketed geometrically from the
point estimate, then narrowed by interpolated probes (a probit-linear fit in log r, under a
budget of bisection's probe count plus a small slack) until the bracket's log width is at most
``_endpoint_tolerance`` -- 1/128 of ``_count_scale`` (the log-risk-ratio standard error),
clamped to ``[2**-40, 2**-11]``. Interpolation only chooses where to probe: the reported
endpoint is the bracket end OUTSIDE the set, a probed point the evaluation itself put outside
it, so it lies beyond the evaluated crossing by at most the final bracket's log width
whatever the fit predicted. A search that stops short of its tolerance is flagged on the result
(``BinomialInterval.resolution_reached``) and disclosed by `precision_note`. That is the
search resolution of the evaluated envelope: the nuisance supremum's certification gap and
SciPy's primitive error are separate quantities. Probes are placed with SciPy's ``ndtri``, so an
endpoint is reproducible to the tolerance, not bit for bit, across SciPy and libm builds.

Scalability: a full evaluation of ``F_+``/``F_-`` enumerates every
control-success count ``i`` in ``0..n_c`` (the conditional sum over
``X_c``). For large ``n_c`` this is enumerated only near its concentration
window (the control's binomial mass is exponentially concentrated around
``n_c*q``; the window is cut with the Chernoff bound of
``_binomial_support``, which follows the nuisance rate); the omitted tail
mass is bounded above and ADDED to the returned value, keeping every
evaluation a valid conservative upper bound while making the
per-evaluation cost ``O(sqrt(n_c*q*(1-q)))`` instead of ``O(n_c)``. See
``_support_window``.
"""

from __future__ import annotations

import heapq
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

import numpy as np
import scipy.special._ufuncs as _scu
from scipy.special import ndtri as _ndtri
from scipy.stats import beta as _beta_dist

from increment._literals import ALTERNATIVE_VALUES, Alternative
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation._binomial_support import chernoff_support
from increment.estimation._tails import SCIPY_BINOMIAL_ULP_ALLOWANCE


class BinomialDataError(CodedError):
    """A per-arm/per-call binomial-method data or numerical guard.

    Distinct from :class:`~increment.errors.InvalidRequestError`: this is
    raised for a condition an individual (metric, arm) cell can hit on
    otherwise-valid input (bad reconstructed counts, an unrepresentable
    numerical allocation) -- callers convert it to a keyed
    ``DecisionFailure`` rather than aborting the whole request, mirroring
    ``LiftGuardError`` in ``inference.py``.
    """


_REFUSALS = refusals(
    BinomialDataError,
    {
        "estimation.binomial.invalid_counts": "binomial counts must satisfy 0 <= x <= n with integer n >= 1, got x={x!r}, n={n!r}",
        "estimation.binomial.tail_unrepresentable": RefusalSpec(
            "estimation.binomial.tail_unrepresentable",
            BinomialDataError,
            lambda **ctx: (
                "a required numerical tail allocation could not be certified within the "
                f"representable search range: {ctx}"
            ),
        ),
        "estimation.binomial.unknown_alternative": RefusalSpec(
            "estimation.binomial.unknown_alternative",
            InvalidRequestError,
            template='Unknown alternative={alternative!r}: must be one of "two-sided", "greater", "less"',
        ),
        "estimation.binomial.arm_too_large_for_exact_enumeration": "n_c={n_c!r}/n_t={n_t!r} exceed this method's validated arm-size ceiling ({max_arm_size!r}): this is a compute-resource applicability boundary, not a scientific one -- per-call cost keeps growing with arm size beyond it, and this method refuses rather than spend unbounded per-request compute on a single estimate",
    },
)
_raise = raiser(_REFUSALS)

UNKNOWN_ALTERNATIVE = _REFUSALS["estimation.binomial.unknown_alternative"]

#: Nuisance tail budget cap: derived ahead of each inversion from that call's
#: REQUESTED (pre-registered, not data-dependent) alpha alone. Directional FCR
#: reinversion passes its full target alpha and therefore derives a fresh
#: budget instead of reusing the original half-alpha budget; counts never tune it.
_NUISANCE_BETA_CAP = 1e-6


def nuisance_beta(alpha: float) -> float:
    """Data-independent nuisance budget for a test/interval at *alpha*."""
    return min(_NUISANCE_BETA_CAP, alpha / 32.0)


def validate_counts(x: int, n: int) -> None:
    if not isinstance(n, int) or not isinstance(x, int) or n < 1 or x < 0 or x > n:
        _raise("estimation.binomial.invalid_counts", x=x, n=n)


def validate_alternative(alternative: str) -> Alternative:
    """Validate and narrow a requested ``alternative`` string to this
    module's strict ``Literal`` type -- every public function below
    requires the narrowed type, never a bare ``str``; callers holding an
    unnarrowed ``alternative: str`` (e.g. a request-level parameter) call
    this once before reaching them, rather than casting the type away.
    """
    if alternative not in ALTERNATIVE_VALUES:
        _raise("estimation.binomial.unknown_alternative", alternative=alternative)
    return cast(Alternative, alternative)


# prose: allow-long outward-rounding error model derives _eps_margin and its SciPy ULP allowance
# --- Floating-point outward-rounding certification -----------------------
# Certified bounds use ordinary float64 arithmetic over `scipy.stats` special functions, so
# `_eps_margin` folds two error sources into one conservative additive margin:
# 1. Higham's bound for summing `m` nonnegative terms in [0, 1]: the absolute error is about
#    `m * eps`, since the true sum is a sub-probability. This covers the `np.dot` step in
#    `_tail_plus`/`_tail_minus`.
# 2. SciPy's per-call `binom.pmf/cdf/sf` error, which is not assumed negligible.
#    `SCIPY_BINOMIAL_ULP_ALLOWANCE` is checked against a `decimal` exact oracle (the incomplete-beta
#    identity for the binomial CDF at integer shapes) in
#    `tests/estimation/test_binomial_rr.py::TestScipyBinomErrorBudget`. The worst grid value
#    that did not underflow to 0 was under 500 ULPs; 2048 keeps about 4x headroom.
_FLOAT64_EPS = float(np.finfo(np.float64).eps)


def _eps_margin(term_count: int) -> float:
    """Additive safety margin bounding accumulated float64 rounding error
    across a probability computed as a dot product of *term_count* many
    nonnegative sub-probabilities, each a SciPy special-function
    evaluation. See the module-level comment above for the derivation.
    """
    return (2 * SCIPY_BINOMIAL_ULP_ALLOWANCE + max(1, term_count)) * _FLOAT64_EPS


def _round_outward(x: float, *, direction: Literal["down", "up"]) -> float:
    """One-ULP directed rounding: nudge *x* to the adjacent float64 in the
    conservative *direction*. Used so a value already computed as a
    conservative (outward-rounded) bound survives a further EXACT
    arithmetic operation (e.g. subtracting 1 to convert a risk-ratio bound
    to relative lift) without an ordinary rounding of that operation
    silently narrowing the bound back past the true value. The largest
    finite float stays itself: that operation moves it toward zero, so its
    rounding cannot have fallen short of the true value, and the neighbour
    is infinite where every consumer needs a finite endpoint.
    """
    neighbour = math.nextafter(x, -math.inf if direction == "down" else math.inf)
    return neighbour if math.isfinite(neighbour) else x


#: Small-tail solver applicability floor. Only n=1 has closed-form
#: endpoints on both sides; x=0 or x=n still needs one solver endpoint.
_CP_BETA_FLOOR = 1e-9

#: Empirical solver allowance, checked by decimal binomial-tail inversion
#: regressions. Directed rounding also protects the final subtraction.
_CP_RELATIVE_SLACK = 1e-6


@lru_cache(maxsize=64)
def clopper_pearson(x: int, n: int, beta: float) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson ``1 - beta`` interval for a binomial
    rate, outward-rounded so ``[a, b]`` is never narrower than the true
    interval (never a bare, untrusted SciPy float -- see
    "Floating-point outward-rounding certification" above).
    """
    validate_counts(x, n)
    if not (0.0 < beta < 1.0):
        _raise("estimation.binomial.tail_unrepresentable", beta=beta)
    half = beta / 2.0
    if half == 0.0:
        _raise("estimation.binomial.tail_unrepresentable", beta=beta)
    if x == 0:
        a = 0.0
    elif (x, n) == (1, 1):
        a = half  # Beta(1, 1) is Uniform(0, 1): SciPy is exact here.
    else:
        if beta < _CP_BETA_FLOOR:
            _raise("estimation.binomial.tail_unrepresentable", beta=beta, x=x, n=n)
        a_raw = float(_beta_dist.ppf(half, x, n - x + 1))
        a = max(0.0, _round_outward(a_raw * (1.0 - _CP_RELATIVE_SLACK), direction="down"))
    if x == n:
        b = 1.0
    elif (x, n) == (0, 1):
        b = min(1.0, _round_outward(1.0 - half, direction="up"))
    else:
        if beta < _CP_BETA_FLOOR:
            _raise("estimation.binomial.tail_unrepresentable", beta=beta, x=x, n=n)
        b_raw = float(_beta_dist.isf(half, x + 1, n - x))
        b = min(
            1.0,
            _round_outward(1.0 - (1.0 - b_raw) * (1.0 - _CP_RELATIVE_SLACK), direction="up"),
        )
    return a, b


# --- Joint-binomial tail sums (conditional summation over X_c) -----------
# Large arms scan only `_support_window`'s rate-aware Chernoff window and add the omitted mass
# back, so truncation cannot make a tail anti-conservative.
#: Benchmarked arm size below which a full scan beats building that window.
_FULL_ENUMERATION_THRESHOLD = 256

#: Total control-PMF mass the support window is allowed to omit, added
#: back verbatim to every tail evaluation. Six orders of magnitude below
#: `_NUISANCE_BETA_CAP`, so its effect on the reported p-value/interval is
#: negligible at any alpha this module would otherwise accept.
_SUPPORT_TRUNCATION_BUDGET = 1e-12


@lru_cache(maxsize=64)
def _support_window(
    n_c: int, a: float, b: float, budget: float = _SUPPORT_TRUNCATION_BUDGET
) -> tuple[int, int, float]:
    """A control-success-count window ``(i_lo, i_hi)`` and a rigorous
    upper bound on the PMF mass it omits, valid SIMULTANEOUSLY for every
    ``q`` in ``[a, b]`` -- computed once per distinct ``(n_c, a, b)`` and
    reused across every tail evaluation in that search.

    Each end omits at most ``budget / 2`` (`chernoff_support`, worst at the
    nuisance interval's end on that side), and an end the bound cannot cut omits
    nothing, so the window spans about ``sqrt(n_c * q * (1 - q))`` counts
    around the Clopper-Pearson interval instead of ``sqrt(n_c)``. A budget
    outside ``(0, 1)`` omits nothing.
    """
    if n_c <= _FULL_ENUMERATION_THRESHOLD or not 0.0 < budget < 1.0:
        return 0, n_c, 0.0
    tail = budget / 2.0
    i_lo, i_hi = chernoff_support(n_c, a, b, tail)
    return i_lo, i_hi, tail * ((i_lo > 0) + (i_hi < n_c))


# `scipy.stats.binom` adds fixed per-call `rv_discrete` overhead that dominates the thousands
# of tail calls in one `confidence_interval`. These wrappers call binom's underlying
# `_scu._binom_*` ufuncs with its support clamping, bit-for-bit
# (`tests/estimation/test_binomial_rr.py::TestFastBinomMatchesScipy`).
def _fast_binom_cdf(k: np.ndarray, n: int, p: float | np.ndarray) -> np.ndarray:
    clamped = np.minimum(np.maximum(k, 0), n)
    # ty: ignore[unresolved-attribute] -- private scipy ufunc; TestFastBinomMatchesScipy pins parity
    out = np.minimum(np.maximum(_scu._binom_cdf(clamped, n, p), 0.0), 1.0)
    out = np.where(k < 0, 0.0, out)
    return np.where(k >= n, 1.0, out)


def _fast_binom_sf(k: np.ndarray, n: int, p: float | np.ndarray) -> np.ndarray:
    clamped = np.minimum(np.maximum(k, 0), n)
    # ty: ignore[unresolved-attribute] -- private scipy ufunc; TestFastBinomMatchesScipy pins parity
    out = np.minimum(np.maximum(_scu._binom_sf(clamped, n, p), 0.0), 1.0)
    out = np.where(k < 0, 1.0, out)
    return np.where(k >= n, 0.0, out)


def _fast_binom_pmf(k: np.ndarray, n: int, p: float | np.ndarray) -> np.ndarray:
    clamped = np.minimum(np.maximum(k, 0), n)
    # ty: ignore[unresolved-attribute] -- private scipy ufunc; TestFastBinomMatchesScipy pins parity
    out = np.minimum(np.maximum(_scu._binom_pmf(clamped, n, p), 0.0), 1.0)
    return np.where((k < 0) | (k > n), 0.0, out)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@lru_cache(maxsize=32)
def _control_pmf(n_c: int, q: float, i_lo: int, i_hi: int) -> np.ndarray:
    return _read_only(_fast_binom_pmf(np.arange(i_lo, i_hi + 1), n_c, q))


@lru_cache(maxsize=32)
def _plus_threshold(n_c: int, n_t: int, k: int, i_lo: int, i_hi: int) -> np.ndarray:
    i = np.arange(i_lo, i_hi + 1)
    return _read_only(np.ceil((k + n_t * i) / n_c).astype(np.int64) - 1)


@lru_cache(maxsize=32)
def _minus_threshold(n_c: int, n_t: int, k: int, i_lo: int, i_hi: int) -> np.ndarray:
    i = np.arange(i_lo, i_hi + 1)
    return _read_only(np.floor((k + n_t * i) / n_c).astype(np.int64))


# A search asks for each treatment vector (a function of `p` alone; the control PMF carries `q`)
# repeatedly: `eval_at(v)` and the corner bound of every interval ending at `v` share `p(v)`.
# Caching returns the identical array a rebuild would, so the dot product and margins are
# untouched. 64 entries capture every reuse in simulated LRU traces of the 100k/1M-per-arm searches.
@lru_cache(maxsize=64)
def _treatment_tail(
    kind: Literal["plus", "minus"], n_c: int, n_t: int, k: int, i_lo: int, i_hi: int, p: float
) -> np.ndarray:
    if kind == "plus":
        return _read_only(_fast_binom_sf(_plus_threshold(n_c, n_t, k, i_lo, i_hi), n_t, p))
    return _read_only(_fast_binom_cdf(_minus_threshold(n_c, n_t, k, i_lo, i_hi), n_t, p))


def _tail_plus(
    q: float, p: float, n_c: int, n_t: int, k: int, window: tuple[int, int, float]
) -> float:
    """Certified (outward-rounded) upper bound on ``F_+(q,p) = P(K >= k)``,
    ``K = n_c*X_t - n_t*X_c``: the windowed partial sum plus both the
    omitted-support bound and the floating-point safety margin.
    """
    i_lo, i_hi, omitted = window
    pmf_i = _control_pmf(n_c, q, i_lo, i_hi)
    sf = _treatment_tail("plus", n_c, n_t, k, i_lo, i_hi, p)
    raw = float(np.dot(pmf_i, sf)) + omitted
    return min(1.0, raw + _eps_margin(i_hi - i_lo + 1))


def _tail_minus(
    q: float, p: float, n_c: int, n_t: int, k: int, window: tuple[int, int, float]
) -> float:
    """Certified (outward-rounded) upper bound on ``F_-(q,p) = P(K <= k)``."""
    i_lo, i_hi, omitted = window
    pmf_i = _control_pmf(n_c, q, i_lo, i_hi)
    cdf = _treatment_tail("minus", n_c, n_t, k, i_lo, i_hi, p)
    raw = float(np.dot(pmf_i, cdf)) + omitted
    return min(1.0, raw + _eps_margin(i_hi - i_lo + 1))


def _p_of_plus(q: float, r: float) -> float:
    return min(r * q, 1.0) if r > 0.0 else 0.0


def _i_term_control(q: float, n_c: int) -> float:
    return n_c / (q * (1.0 - q)) if 0.0 < q < 1.0 else math.inf


def _i_term_treatment(q: float, r: float, n_t: int) -> float:
    rq = r * q
    if q <= 0.0 or rq >= 1.0:
        return math.inf
    room = q * (1.0 - rq)
    # A denominator below the smallest float makes the exact quotient exceed the largest one.
    return (n_t * r) / room if room > 0.0 else math.inf


def _i_bound(u: float, v: float, r: float, n_c: int, n_t: int) -> float:
    """``max`` over ``{u, v}`` of ``I(q) = n_c/(q(1-q)) + n_t*r/(q(1-rq))``.

    Both summands are convex on their domain, so this equals the true
    supremum over ``[u, v]`` exactly -- no sampling needed.
    """
    iu = _i_term_control(u, n_c) + _i_term_treatment(u, r, n_t)
    iv = _i_term_control(v, n_c) + _i_term_treatment(v, r, n_t)
    return max(iu, iv)


def _certified_sup(
    kind: Literal["plus", "minus"],
    a: float,
    b: float,
    r: float,
    n_c: int,
    n_t: int,
    k: int,
    window: tuple[int, int, float],
    *,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Certified (conservative) upper bound on ``sup_{q in [a,b]}`` of the
    relevant tail evaluated along ``p(q)``. Always a valid upper bound,
    regardless of how many iterations run; more iterations only tighten
    it. ``tol``/``max_iter`` trade tightness for speed (a looser,
    anytime-valid bound at fewer iterations); they do not trade away
    validity.
    """
    if b < a:
        return 0.0

    if kind == "plus":

        def eval_at(q: float) -> float:
            return _tail_plus(q, _p_of_plus(q, r), n_c, n_t, k, window)

        def monotone_bound(u: float, v: float) -> float:
            # F_+ is nonincreasing in q, nondecreasing in p; p(q) is
            # nondecreasing in q, so the sup over [u,v] is bounded by
            # evaluating at the most favorable (q, p) combination.
            return _tail_plus(u, _p_of_plus(v, r), n_c, n_t, k, window)
    else:

        def eval_at(q: float) -> float:
            return _tail_minus(q, r * q, n_c, n_t, k, window)

        def monotone_bound(u: float, v: float) -> float:
            return _tail_minus(v, r * u, n_c, n_t, k, window)

    def bound(u: float, v: float, fu: float, fv: float) -> float:
        if v <= u:
            return max(fu, fv)
        mono = monotone_bound(u, v)
        imax = _i_bound(u, v, r, n_c, n_t)
        # At a singular nuisance endpoint (q=0 or rq=1), the derivative
        # envelope is infinite even though the monotone coupling bound is
        # finite and certified.  Recursing on that interval cannot tighten
        # the derivative bound.
        if not math.isfinite(imax):
            return mono
        quad = max(fu, fv) + imax * (v - u) ** 2 / 4.0
        return min(1.0, mono, quad)

    fa, fb = eval_at(a), eval_at(b)
    best_achieved = max(fa, fb)
    heap: list[tuple[float, float, float, float, float]] = []
    heapq.heappush(heap, (-bound(a, b, fa, fb), a, b, fa, fb))
    for _ in range(max_iter):
        neg_bnd, u, v, fu, fv = heap[0]
        bnd = -neg_bnd
        if bnd - best_achieved <= tol:
            break
        heapq.heappop(heap)
        mid = (u + v) / 2.0
        if mid <= u or mid >= v:
            # floating-point floor: cannot bisect further; this leaf's
            # bound stands as the certified value for its sliver.
            heapq.heappush(heap, (neg_bnd, u, v, fu, fv))
            break
        fm = eval_at(mid)
        best_achieved = max(best_achieved, fm)
        heapq.heappush(heap, (-bound(u, mid, fu, fm), u, mid, fu, fm))
        heapq.heappush(heap, (-bound(mid, v, fm, fv), mid, v, fm, fv))
    certified = max(-item[0] for item in heap)
    return min(1.0, max(certified, best_achieved))


def _p_plus_impl(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    a, b = clopper_pearson(x_c, n_c, beta)
    window = _support_window(n_c, a, b)
    k = n_c * x_t - n_t * x_c
    sup = _certified_sup("plus", a, b, r, n_c, n_t, k, window)
    return min(1.0, beta + sup)


@lru_cache(maxsize=512)
def _p_plus_cached(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    return _p_plus_impl(r, x_c, n_c, x_t, n_t, beta)


def p_plus(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    """``p_+(r)``: p-value for ``H0: R <= r`` (rejecting favors ``R > r``)."""
    if r < 0.0:
        _raise("estimation.binomial.tail_unrepresentable", r=r)
    # Keep refusal/type behavior of the uncached API for malformed,
    # unhashable arguments instead of letting functools raise while building
    # the cache key.
    try:
        hash((r, x_c, n_c, x_t, n_t, beta))
    except TypeError:
        return _p_plus_impl(r, x_c, n_c, x_t, n_t, beta)
    return _p_plus_cached(r, x_c, n_c, x_t, n_t, beta)


def _p_minus_impl(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    a, b = clopper_pearson(x_c, n_c, beta)
    upper = b if r <= 0.0 else min(b, 1.0 / r)
    if upper < a:
        return min(1.0, beta)
    window = _support_window(n_c, a, upper)
    k = n_c * x_t - n_t * x_c
    sup = _certified_sup("minus", a, upper, r, n_c, n_t, k, window)
    return min(1.0, beta + sup)


@lru_cache(maxsize=512)
def _p_minus_cached(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    return _p_minus_impl(r, x_c, n_c, x_t, n_t, beta)


def p_minus(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    """``p_-(r)``: p-value for ``H0: R >= r`` (rejecting favors ``R < r``)."""
    if r < 0.0:
        _raise("estimation.binomial.tail_unrepresentable", r=r)
    try:
        hash((r, x_c, n_c, x_t, n_t, beta))
    except TypeError:
        return _p_minus_impl(r, x_c, n_c, x_t, n_t, beta)
    return _p_minus_cached(r, x_c, n_c, x_t, n_t, beta)


def p_two(r: float, x_c: int, n_c: int, x_t: int, n_t: int, beta: float) -> float:
    """``p_two(r) = min(1, 2*min(p_+(r), p_-(r)))``."""
    return min(
        1.0, 2.0 * min(p_plus(r, x_c, n_c, x_t, n_t, beta), p_minus(r, x_c, n_c, x_t, n_t, beta))
    )


# --- Monotone inversion ----------------------------------------------------


#: Extra bisection budget for placing the tested null on its exact side of a
#: boundary; reached only when the refined bracket still contains the null.
_RESOLVE_BISECT = 60

#: Endpoint stop as a fraction of `_count_scale`, the log-risk-ratio standard error. Probes are
#: interpolated, so each halving of the stop costs about one probe in the expensive region.
_ENDPOINT_RESOLUTION = 2.0**-7
#: Finest stop, and how far below its seed a search descends before giving up on a crossing:
#: halving a ln 2 bracket to it takes about 40 steps.
_ENDPOINT_FLOOR = 2.0**-40
#: Coarsest stop, for sparse cells whose standard error is O(1).
_ENDPOINT_CAP = 2.0**-11
#: Nearest an interpolated probe sits to a bracket end, as a fraction of the stop: an estimate
#: within that distance of the crossing brackets it from both sides in one more probe.
_CLOSING_MARGIN = 0.5
#: Probes beyond plain bisection's count that interpolation may spend before the window around
#: the midpoint forces bisection.
_REFINE_SLACK = 2
#: Float64 error of a planned probe in log r, per unit of ``1 + max(|log lo|, |log hi|)``: the
#: logarithms of the bracket ends, the arithmetic planning the probe, and the exponential back.
_LOG_ROUNDING = 2.0**-48
#: P-values are clamped here before the probit, so a plateau at 0 or 1 scores a finite value.
_SCORE_FLOOR = 1e-300
_SCORE_CEILING = 1.0 - 2.0**-53


def _count_scale(x_c: int, n_c: int, x_t: int, n_t: int) -> float:
    """Delta-method standard error of ``log R`` at the continuity-adjusted proportions
    ``p = (x + 1/2) / (n + 1)``: per arm ``(1 - p) / ((n + 1) * p) = (n - x + 1/2) / ((n + 1) *
    (x + 1/2))``. Positive and finite for every ``0 <= x <= n``, zero counts included, and small
    when an arm nearly always converts. It is the unit the endpoint resolution is measured in --
    neither an interval nor a coverage statement.
    """
    return math.sqrt(
        (n_c - x_c + 0.5) / ((n_c + 1.0) * (x_c + 0.5))
        + (n_t - x_t + 0.5) / ((n_t + 1.0) * (x_t + 0.5))
    )


def _endpoint_tolerance(scale: float) -> float:
    """Bracket log width ``log(hi / lo)`` at which the endpoint search stops for a cell whose
    log-risk-ratio standard error is *scale*: ``_ENDPOINT_RESOLUTION`` of it, clamped to
    ``[_ENDPOINT_FLOOR, _ENDPOINT_CAP]``.
    """
    return min(max(scale * _ENDPOINT_RESOLUTION, _ENDPOINT_FLOOR), _ENDPOINT_CAP)


@dataclass(frozen=True, slots=True)
class _Boundary:
    """One inversion's outcome.

    ``endpoint`` is the outward bracket end (``None``: no finite ceiling below the search cap).
    ``log_width`` is ``log(hi / lo)`` of the final bracket -- ``0.0`` for an exact endpoint,
    ``inf`` while the bracket still touches 0 -- and ``reached`` says whether it is within the
    requested tolerance.
    """

    endpoint: float | None
    log_width: float
    reached: bool


class _Crossing:
    """The bracket ``[lo, hi]`` of one monotone inversion and the probit score of each probe.

    ``hi`` is at or above the crossing of ``f`` at ``target`` and ``lo`` below it, each by one
    evaluation, so the outward endpoint is ``lo`` for an increasing ``f`` (the set is
    ``[r, inf)``) and ``hi`` for a decreasing one (the set is ``[0, r]``).
    """

    __slots__ = ("_f", "_increasing", "_scores", "_target", "_z_target", "hi", "lo")

    def __init__(self, f: Callable[[float], float], target: float, *, increasing: bool) -> None:
        self._f = f
        self._target = target
        self._increasing = increasing
        self._z_target = float(_ndtri(target))
        self._scores: dict[float, float] = {}
        self.lo = 0.0
        self.hi = 0.0

    @property
    def endpoint(self) -> float:
        return self.lo if self._increasing else self.hi

    def log_width(self) -> float:
        return math.log1p((self.hi - self.lo) / self.lo) if self.lo > 0.0 else math.inf

    def at_or_above(self, r: float) -> bool:
        """Whether one evaluation puts *r* at or above the crossing; records its score."""
        p = self._f(r)
        clamped = min(max(p, _SCORE_FLOOR), _SCORE_CEILING)
        self._scores[r] = float(_ndtri(clamped)) - self._z_target
        return (p >= self._target) == self._increasing

    def narrow(self, r: float) -> None:
        if self.at_or_above(r):
            self.hi = r
        else:
            self.lo = r

    def bisect(self) -> bool:
        """Probe the geometric midpoint; ``False`` when floating point has no point inside."""
        mid = math.sqrt(self.lo) * math.sqrt(self.hi) if self.lo > 0.0 else self.hi / 2.0
        if not self.lo < mid < self.hi:
            return False
        self.narrow(mid)
        return True

    def bracket_from(self, seed: float, *, cap: float) -> bool:
        """Bracket the crossing geometrically from *seed*, within a factor of 2 at any scale.

        Doubles away from *seed* while it lies below the crossing and halves toward 0 while
        it lies above, down to ``seed * _ENDPOINT_FLOOR`` (then ``lo`` is left at 0). Doubling
        runs to *cap*, however far *seed* sits below the crossing, and ends by probing *cap*
        itself when a doubling would pass it. ``False`` means a decreasing ``f`` stayed at or
        above its target through *cap*: unbounded.
        """
        floor = seed * _ENDPOINT_FLOOR
        if self.at_or_above(seed):
            self.hi, self.lo = seed, seed / 2.0
            while self.at_or_above(self.lo):
                self.hi, self.lo = self.lo, self.lo / 2.0
                if self.lo < floor:
                    self.lo = 0.0
                    break
            return True
        self.lo, self.hi = seed, min(seed * 2.0, cap)
        while not self.at_or_above(self.hi):
            if self.hi >= cap:
                if self._increasing:
                    _raise(
                        "estimation.binomial.tail_unrepresentable",
                        target=self._target,
                        probed=self.hi,
                    )
                return False
            self.lo, self.hi = self.hi, min(self.hi * 2.0, cap)
        return True

    def _probe_point(self, tau: float, remaining: int) -> float:
        """Next probe, in log r: the probit-linear estimate of the crossing from the bracket
        ends, held at least ``_CLOSING_MARGIN * tau`` inside them, then confined to the window
        around the midpoint that leaves *remaining* probes enough to reach *tau* by bisection.
        The window is sized against *tau* less the rounding of the log coordinates, so the
        last probe lands the bracket at or below *tau* rather than a ulp above it.
        """
        a, b = math.log(self.lo), math.log(self.hi)
        s_lo, s_hi = self._scores[self.lo], self._scores[self.hi]
        mid = 0.5 * (a + b)
        x = mid if s_lo == s_hi else a + (b - a) * (s_lo / (s_lo - s_hi))
        margin = _CLOSING_MARGIN * tau
        x = min(max(x, a + margin), b - margin)
        goal = tau - _LOG_ROUNDING * (1.0 + max(abs(a), abs(b)))
        radius = max(0.5 * goal * 2.0**remaining - 0.5 * (b - a), 0.0)
        return math.exp(min(max(x, mid - radius), mid + radius))

    def refine(self, tau: float) -> None:
        """Narrow the bracket to a log width of at most *tau* (a bracket reaching 0 stays).

        Each probe is a point where the evaluation decides which side of the crossing it is on,
        so the bracket and the outward endpoint are the evaluation's own whatever the
        interpolation predicts. The probe window caps the search at
        ``ceil(log2(w / tau)) + _REFINE_SLACK`` probes for an initial bracket of log width *w*,
        sized against *tau* less the rounding of the log coordinates (`_LOG_ROUNDING`), so the
        last probe never leaves the bracket a ulp above *tau*; bisection finishes only a bracket
        that rounding still left wider.
        """
        if self.lo <= 0.0:
            return
        budget = max(0, math.ceil(math.log2(self.log_width() / tau))) + _REFINE_SLACK
        for spent in range(budget):
            if self.log_width() <= tau:
                return
            r = self._probe_point(tau, budget - spent)
            if self.lo < r < self.hi:
                self.narrow(r)
            elif not self.bisect():
                return
        while self.log_width() > tau and self.bisect():
            pass

    def place_null(self, null: float) -> None:
        """Never leave *null* on the wrong side of the reported endpoint: probe it exactly when
        it sits inside the bracket and bisect until the endpoint is no longer equal to it.
        """
        if not self.lo <= null <= self.hi:
            return
        if self.lo < null < self.hi:
            self.narrow(null)
        for _ in range(_RESOLVE_BISECT):
            if self.endpoint != null or not self.bisect():
                break


def _find_boundary(
    f: Callable[[float], float],
    target: float,
    *,
    increasing: bool,
    seed: float,
    tau: float,
    cap: float = 1e12,
    resolve: float | None = None,
) -> _Boundary:
    """Outward-rounded boundary of ``{r >= 0 : f(r) >= target}`` (an ``[r, inf)`` set if
    *increasing*, a ``[0, r]`` set otherwise), bracketed to a log width of at most *tau*.

    The crossing is bracketed geometrically from *seed* (`_Crossing.bracket_from`) and the
    bracket refined by interpolated probes under a bisection-equivalent budget
    (`_Crossing.refine`). The endpoint is the bracket end OUTSIDE the set: the lower-bound
    search reports the largest *r* confirmed outside (extending the set downward), the
    upper-bound search the smallest (extending it upward). It therefore lies beyond the
    evaluated crossing by at most the final bracket's log width: a coarser or finer *tau*
    can move it either way, never to the inside of a crossing the evaluation confirmed.
    ``None`` means the search exhausted *cap* without ``f`` dropping below *target* (only
    reachable for *increasing=False*). Callers are responsible for supplying a *cap* that
    DISTINGUISHES a genuinely unbounded set from a merely-large one: e.g.
    `confidence_interval` derives *cap* from the Clopper-Pearson nuisance bound rather than
    using this default, so ``None`` here reflects only the mathematically established case.

    A crossing below the halving floor, or a bracket floating point cannot halve, leaves
    ``reached`` false with the conservative endpoint and the width actually achieved (``inf``
    while the bracket touches 0). That is the resolution of the evaluated ``f`` only.

    *resolve* (the tested null) is never left on the wrong side of the reported boundary: when
    the search stops with it inside the unresolved bracket, it is probed exactly and the search
    continues until the reported boundary sits on the same side of it as the true one, so the
    set excludes *resolve* exactly when ``f(resolve) < target``.
    """
    seed = seed if math.isfinite(seed) and seed > 0.0 else 1.0
    if increasing:
        if f(0.0) >= target:
            return _Boundary(0.0, 0.0, True)
    else:
        if f(cap) >= target:
            return _Boundary(None, 0.0, True)
        if f(0.0) < target:
            return _Boundary(0.0, 0.0, True)
    crossing = _Crossing(f, target, increasing=increasing)
    if not crossing.bracket_from(seed, cap=cap):
        return _Boundary(None, 0.0, True)
    crossing.refine(tau)
    if resolve is not None:
        crossing.place_null(resolve)
    width = crossing.log_width()
    return _Boundary(crossing.endpoint, width, width <= tau)


def _bound_upper(
    f: Callable[[float], float],
    target: float,
    *,
    a: float,
    seed: float,
    tau: float,
    resolve: float | None = None,
) -> _Boundary:
    """Boundary of the upper (decreasing) search ``{r : f(r) >= target} =
    [0, r_upper]``, distinguishing genuine unboundedness from a numerical
    search failure.

    ``a`` (the Clopper-Pearson lower bound feeding *f*) determines which
    applies: ``a == 0`` (``x_c == 0``) means the nuisance domain never
    empties as ``r -> inf``, so ``[0, r_upper]`` is analytically
    unbounded -- returned with no endpoint and NO search at all, since no
    finite bracket could ever certify it either way. ``a > 0`` means the
    nuisance domain ``[a, min(b, 1/r)]`` becomes empty once ``r > 1/a``,
    where ``f`` collapses to the cheap constant ``beta`` (always strictly
    below *target*, since ``target >= alpha/2 > alpha/32 >=
    nuisance_beta(alpha)`` for every valid ``alpha``); a finite crossing
    is therefore GUARANTEED to exist at or before ``2/a``, and the search
    is capped there (at the largest finite float when ``2/a`` overflows)
    rather than at an arbitrary numeric ceiling. When ``1/a`` itself
    exceeds the largest float no representable point puts *f* below
    *target*, and the search raises a coded numerical failure instead of
    silently reporting an unbounded set for a case that is NOT
    analytically unbounded.
    """
    if a <= 0.0:
        return _Boundary(None, 0.0, True)
    cap = min(2.0 / a, sys.float_info.max)
    result = _find_boundary(
        f, target, increasing=False, seed=seed, tau=tau, cap=cap, resolve=resolve
    )
    if result.endpoint is None:
        # Reached only when 1/a exceeds the largest float (f(cap) < target
        # holds for every other a > 0); surfaced as a coded failure rather
        # than a silently wrong unbounded claim.
        _raise("estimation.binomial.tail_unrepresentable", target=target, cap=cap)
    return result


#: Compute-resource ceiling, not a statistical limit, sized for the largest supported design:
#: a two-sided `confidence_interval` call up to it fits the per-request budget, and larger
#: arms refuse before searching. Unbounded sizes need a closed-form or recurrence tail evaluator.
MAX_ARM_SIZE = 4_000_000


@dataclass(frozen=True, slots=True)
class BinomialInterval:
    """One directional/central binomial risk-ratio confidence set.

    ``lower``/``upper`` are risk-ratio (``R``) endpoints, never lift
    (``R - 1``) -- callers convert via `to_lift_bounds`, never a bare
    ``x - 1.0``. ``upper is None`` means genuinely unbounded (natural
    support has no finite ceiling); ``lower`` is never ``None`` (``R``'s
    natural floor, 0, is always a legitimate finite value).

    ``endpoint_log_width`` is the larger final-bracket ``log(hi / lo)`` over the searched
    endpoints (``0.0`` when every endpoint is structural, ``inf`` while a bracket still touches
    0) and ``resolution_reached`` whether each met the resolution declared for these counts.
    Both describe the search of the evaluated p-envelope only: not a bound on displacement from
    the ideal interval, not a coverage statement.
    """

    lower: float
    upper: float | None
    geometry: Literal["central", "lower_bound", "upper_bound"]
    p_value_null: float
    endpoint_log_width: float
    resolution_reached: bool


#: Leading text of the note an unresolved endpoint search leaves on a row. The disclosure holds
#: none of the separators rows join their notes with, so `without_precision_note` can lift it
#: out of a longer note.
PRECISION_NOTE_PREFIX = "binomial endpoint resolution not reached"
_NOTE_SEPARATOR = re.compile(r"( \| |; )")


def precision_note(ci: BinomialInterval) -> str | None:
    """The disclosure for *ci*'s endpoint search, or ``None`` when it met its resolution."""
    if ci.resolution_reached:
        return None
    achieved = (
        f"relative log width {ci.endpoint_log_width:.3g}"
        if math.isfinite(ci.endpoint_log_width)
        else "a bracket that still reaches zero"
    )
    return (
        f"{PRECISION_NOTE_PREFIX}: an endpoint was located only to {achieved}, "
        "and the reported bounds remain conservative (outward-rounded)"
    )


def without_precision_note(note: str | None) -> str | None:
    """*note* with any `precision_note` removed, the rest of it and its separators intact."""
    if not note:
        return None
    pieces = _NOTE_SEPARATOR.split(note)
    kept: list[str] = []
    for index in range(0, len(pieces), 2):
        if pieces[index].startswith(PRECISION_NOTE_PREFIX):
            continue
        if kept:
            kept.append(pieces[index - 1])
        kept.append(pieces[index])
    return "".join(kept) or None


def confidence_interval(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    *,
    alpha: float,
    alternative: Alternative,
    null_r: float = 1.0,
) -> BinomialInterval:
    """Invert the Berger-Boos tests into a risk-ratio confidence set, plus
    the p-value for ``H0: R = null_r`` under the same *alternative*.

    Endpoints are outward-rounded to a relative log resolution declared from the counts
    (``BinomialInterval.endpoint_log_width``); `precision_note` discloses a search that fell short.
    """
    validate_counts(x_c, n_c)
    validate_counts(x_t, n_t)
    if not (0.0 < alpha < 1.0):
        _raise("estimation.binomial.tail_unrepresentable", alpha=alpha)
    if null_r < 0.0 or not math.isfinite(null_r):
        _raise("estimation.binomial.tail_unrepresentable", null_r=null_r)
    if n_c > MAX_ARM_SIZE or n_t > MAX_ARM_SIZE:
        _raise(
            "estimation.binomial.arm_too_large_for_exact_enumeration",
            n_c=n_c,
            n_t=n_t,
            max_arm_size=MAX_ARM_SIZE,
        )
    validated_alternative = validate_alternative(alternative)
    arguments = (x_c, n_c, x_t, n_t, alpha, validated_alternative, null_r)
    try:
        hash(arguments)
    except TypeError:
        return _confidence_interval_cached.__wrapped__(*arguments)
    return _confidence_interval_cached(*arguments)


@lru_cache(maxsize=128)
def _confidence_interval_cached(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    alpha: float,
    alternative: Alternative,
    null_r: float,
) -> BinomialInterval:
    beta = nuisance_beta(alpha)
    seed = (n_c * x_t) / (n_t * x_c) if x_c > 0 and x_t > 0 else 1.0
    tau = _endpoint_tolerance(_count_scale(x_c, n_c, x_t, n_t))

    def f_plus(r: float) -> float:
        return p_plus(r, x_c, n_c, x_t, n_t, beta)

    def f_minus(r: float) -> float:
        return p_minus(r, x_c, n_c, x_t, n_t, beta)

    a, _b = clopper_pearson(x_c, n_c, beta)
    if alternative == "two-sided":
        alloc = alpha / 2.0
        lower = _find_boundary(f_plus, alloc, increasing=True, seed=seed, tau=tau, resolve=null_r)
        assert lower.endpoint is not None, "the lower search is always bounded (increasing=True)"
        upper = _bound_upper(f_minus, alloc, a=a, seed=seed, tau=tau, resolve=null_r)
        return BinomialInterval(
            lower=lower.endpoint,
            upper=upper.endpoint,
            geometry="central",
            p_value_null=p_two(null_r, x_c, n_c, x_t, n_t, beta),
            endpoint_log_width=max(lower.log_width, upper.log_width),
            resolution_reached=lower.reached and upper.reached,
        )
    if alternative == "greater":
        lower = _find_boundary(f_plus, alpha, increasing=True, seed=seed, tau=tau, resolve=null_r)
        assert lower.endpoint is not None
        return BinomialInterval(
            lower=lower.endpoint,
            upper=None,
            geometry="lower_bound",
            p_value_null=f_plus(null_r),
            endpoint_log_width=lower.log_width,
            resolution_reached=lower.reached,
        )
    if alternative == "less":
        upper = _bound_upper(f_minus, alpha, a=a, seed=seed, tau=tau, resolve=null_r)
        return BinomialInterval(
            lower=0.0,
            upper=upper.endpoint,
            geometry="upper_bound",
            p_value_null=f_minus(null_r),
            endpoint_log_width=upper.log_width,
            resolution_reached=upper.reached,
        )
    _raise("estimation.binomial.unknown_alternative", alternative=alternative)


def to_lift_bounds(ci: BinomialInterval) -> tuple[float, float | None]:
    """Convert a risk-ratio confidence interval to the relative-lift
    scale (``R - 1``), preserving each endpoint's outward-rounding
    direction through the subtraction.

    Only a SEARCH-DERIVED endpoint gets the one-ULP directed nudge: the
    exact structural floor (``lower == 0.0`` -- possible for central and
    one-sided sets when the treatment count is zero) converts directly
    (``0.0 - 1.0 == -1.0`` has zero float64 rounding error), matching this
    module's documented invariant that relative lift's floor of ``-1`` is
    always exactly attainable, not "within a ULP of -1". Every other
    bound came out of `_find_boundary`'s outward-rounded search and needs
    the nudge so an ordinary (undirected) rounding of the subtraction
    cannot silently narrow it back past the true value (see
    "Floating-point outward-rounding certification" above).
    """
    lower = -1.0 if ci.lower == 0.0 else _round_outward(ci.lower - 1.0, direction="down")
    upper = None if ci.upper is None else _round_outward(ci.upper - 1.0, direction="up")
    return lower, upper


def point_lift(x_c: int, n_c: int, x_t: int, n_t: int) -> float | None:
    """Empirical relative lift, or ``None`` when no finite point exists
    (``x_c == 0``: the empirical ratio has a zero denominator).
    """
    validate_counts(x_c, n_c)
    validate_counts(x_t, n_t)
    if x_c == 0:
        return None
    if x_t == 0:
        return -1.0
    return (n_c * x_t) / (n_t * x_c) - 1.0
