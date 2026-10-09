import math
import zlib

import numpy as np
import pytest
from scipy.special import erfinv
from scipy.stats import norm, poisson

from increment.errors import InvalidRequestError
from increment.estimation.quantile import _log_quantile_se_impl, log_quantile_se
from tests.mc import mcse, replicate


def _seed(*parts: object) -> int:
    """Deterministic per-cell seed -- Python's builtin ``hash()`` is
    randomized per-process for strings, which would make these grids
    flaky across runs."""
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF


def _lognormal_true_log_quantile(mu: float, sigma: float, q: float) -> float:
    return mu + sigma * float(np.sqrt(2) * erfinv(2 * q - 1))


def _coverage(n_sims: int, n: int, q: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    true_log_q = _lognormal_true_log_quantile(1.0, 0.5, q)

    def trial(_i: int) -> bool:
        y = rng.lognormal(1.0, 0.5, n)
        point, se = log_quantile_se(y, q)
        return abs(math.log(point) - true_log_q) <= 1.96 * se

    return replicate(n_sims, trial).rate


def test_coverage_smoke_fast():
    # small-N smoke: loose bound, must stay in the fast suite
    assert _coverage(n_sims=60, n=400, q=0.9, seed=3) > 0.85


def test_discrete_bracket_never_refuses_smoke_fast():
    """Smoke variant: Poisson(3) at n=100, q=0.9 must return a
    widened, positive-SE interval instead of refusing -- the removed
    'any tie refuses' rule rejected essentially every draw here."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        y = rng.poisson(3, 100).astype(float)
        point, se = log_quantile_se(y, q=0.9)
        assert math.isfinite(point) and point > 0.0 and se > 0.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_coverage_nominal():
    cov = _coverage(n_sims=2000, n=2000, q=0.9, seed=3)
    assert 0.93 < cov < 0.97


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize(
    "n,q",
    [(100, 0.5), (100, 0.9), (100, 0.95), (1000, 0.9), (1000, 0.95), (1000, 0.99)],
)
def test_coverage_grid_nominal(n, q):
    """Every (q, n) cell the estimator accepts must cover the true log
    quantile at >= nominal minus 4 Monte Carlo standard errors. Continuous
    lognormal data in the supported zone must never be refused - refusals
    are reserved for cells where the distribution-free bound does not
    exist, and those must refuse 100% (see test_clamp_zone_refuses)."""
    n_sims = 1500
    rng = np.random.default_rng(3)
    true_log_q = _lognormal_true_log_quantile(1.0, 0.5, q)

    def trial(_i: int) -> bool:
        y = rng.lognormal(1.0, 0.5, n)
        point, se = log_quantile_se(y, q)
        return abs(math.log(point) - true_log_q) <= 1.96 * se

    cov = replicate(n_sims, trial)
    assert cov.rate >= 0.95 - 4 * mcse(0.95, n_sims)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_clamp_zone_refuses():
    """(q=0.99, n=100) sat at ~50% actual coverage under a nominal 95%
    when the upper rank was clamped to the sample max; the whole band
    n in [25, 367] must now refuse deterministically, independent of the
    data drawn."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        y = rng.lognormal(1.0, 0.5, 100)
        with pytest.raises(InvalidRequestError) as exc_info:
            log_quantile_se(y, q=0.99)
        assert exc_info.value.code == "estimation.quantile.too_small_bound"


# --- Resolution coverage/width-cost grid ------------------------------------
# The tie-aware construction must keep coverage where the classical formula
# works and close gaps on coarsely recorded outcomes, including off-grid values:
# cents, seconds, ms, continuous lognormal and Poisson counts.


def _round_cents(y: np.ndarray) -> np.ndarray:
    return np.round(y, 2)


def _round_whole(y: np.ndarray) -> np.ndarray:
    return np.round(y)


# (mu, sigma, rounding fn, values per unit) per recording resolution.
_ROUNDED_KINDS = {
    "cents": (1.0, 0.5, _round_cents, 100),
    "seconds": (5.0, 1.0, _round_whole, 1),
    "ms": (5.0, 0.6, _round_whole, 1),
}


def _rounded_population_quantile(kind: str, q: float) -> float:
    """The recorded outcome's own q-quantile, exactly: the smallest grid
    value k/scale with P(X < (k + 1/2)/scale) >= q. (A large reference
    sample lands one grid value off whenever q sits within its sampling
    error of a grid boundary -- ms at q=0.5 and seconds at q=0.9 here.)"""
    mu, sigma, _, scale = _ROUNDED_KINDS[kind]
    return math.ceil(math.exp(mu + sigma * norm.ppf(q)) * scale - 0.5) / scale


def _off_grid_population_quantile(kind: str, q: float, share: float) -> float:
    """q-quantile of the outcome recorded on the grid except for a share of
    values recorded exactly: inf{y : F(y) >= q} for the mixture CDF."""
    mu, sigma, _, scale = _ROUNDED_KINDS[kind]

    def cdf(y: float) -> float:
        on_grid = norm.cdf((math.log((math.floor(y * scale + 1e-9) + 0.5) / scale) - mu) / sigma)
        return (1.0 - share) * on_grid + share * norm.cdf((math.log(y) - mu) / sigma)

    lo = hi = math.exp(mu + sigma * norm.ppf(q))
    while cdf(lo) >= q:
        lo /= 2.0
    while cdf(hi) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if cdf(mid) >= q else (mid, hi)
    on_grid = round(hi * scale) / scale
    return on_grid if math.isclose(hi, on_grid, rel_tol=1e-9) else hi


def _lognormal_cell_coverage(
    kind: str, n: int, q: float, alpha: float, seed: int, n_sims: int
) -> float:
    rng = np.random.default_rng(seed)
    if kind == "continuous":
        mu, sigma = 1.0, 0.5
        true_log_q = _lognormal_true_log_quantile(mu, sigma, q)

        def draw() -> np.ndarray:
            return rng.lognormal(mu, sigma, n)
    else:
        mu, sigma, round_fn, _ = _ROUNDED_KINDS[kind]
        true_log_q = math.log(_rounded_population_quantile(kind, q))

        def draw() -> np.ndarray:
            return round_fn(rng.lognormal(mu, sigma, n))

    z = norm.isf(alpha / 2.0)

    def trial(_i: int) -> bool:
        point, se = log_quantile_se(draw(), q, alpha=alpha)
        return abs(math.log(point) - true_log_q) <= z * se

    return replicate(n_sims, trial).rate


_LOGNORMAL_GRID = [
    (kind, n, q, alpha)
    for kind in ("continuous", "cents", "seconds", "ms")
    for n in (2000, 5000, 10000, 30000, 100000)
    for q in (0.5, 0.9)
    for alpha in (0.01, 0.05, 0.10)
]


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("kind,n,q,alpha", _LOGNORMAL_GRID)
def test_resolution_calibrated_coverage_grid(kind, n, q, alpha):
    """No cell in the grid may undercover, including the transition-band
    cells a binary tie-refusal switch could not resolve (e.g. seconds
    n=10000 q=0.5 alpha=0.10, ms n=2000 q=0.5 alpha=0.05)."""
    n_sims = 400
    seed = _seed("lognormal_grid", kind, n, q, alpha)
    cov = _lognormal_cell_coverage(kind, n, q, alpha, seed=seed, n_sims=n_sims)
    nominal = 1.0 - alpha
    floor = nominal - 4 * mcse(nominal, n_sims)
    assert cov >= floor, (
        f"{kind} n={n} q={q} alpha={alpha}: cov={cov} < floor={floor} ({n_sims} sims)"
    )


_POISSON_GRID = [
    (lam, n, q, alpha)
    for lam in (3.0, 5.0, 8.0)
    for n in (100, 1000)
    for q in (0.5, 0.9)
    for alpha in (0.01, 0.05, 0.10)
]


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("lam,n,q,alpha", _POISSON_GRID)
def test_resolution_calibrated_coverage_grid_poisson(lam, n, q, alpha):
    """Same grid for count-like (Poisson) data, whose q=0.9 bracket sits
    on an atom the classical formula alone cannot resolve -- the removed
    'any tie refuses' rule rejected essentially every draw here."""
    n_sims = 400
    seed = _seed("poisson_grid", lam, n, q, alpha)
    rng = np.random.default_rng(seed)
    true_q = float(poisson.ppf(q, lam))
    assert true_q > 0.0
    true_log_q = math.log(true_q)
    z = norm.isf(alpha / 2.0)

    def trial(_i: int) -> bool:
        y = rng.poisson(lam, n).astype(float)
        point, se = log_quantile_se(y, q, alpha=alpha)
        return abs(math.log(point) - true_log_q) <= z * se

    cov = replicate(n_sims, trial).rate
    nominal = 1.0 - alpha
    floor = nominal - 4 * mcse(nominal, n_sims)
    assert cov >= floor, (
        f"lam={lam} n={n} q={q} alpha={alpha}: cov={cov} < floor={floor} ({n_sims} sims)"
    )


_OFF_GRID_CELLS = [
    # (kind, n, q, share of values recorded off the grid)
    ("ms", 100_000, 0.5, 0.001),
    ("ms", 100_000, 0.5, 0.1),
    ("ms", 10_000, 0.5, 0.01),
    ("seconds", 100_000, 0.5, 0.1),
]


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("kind,n,q,share", _OFF_GRID_CELLS)
def test_coverage_with_values_recorded_off_the_grid(kind, n, q, share):
    """A few values recorded off an otherwise coarse grid must not cost
    coverage: a step read from the smallest gap between recorded values
    fell to 90% here at a 0.1% share and 60% at 10%, against a 95%
    target."""
    n_sims, alpha = 1000, 0.05
    mu, sigma, round_fn, _ = _ROUNDED_KINDS[kind]
    true_log_q = math.log(_off_grid_population_quantile(kind, q, share))
    rng = np.random.default_rng(_seed("off_grid", kind, n, q, share))
    z = norm.isf(alpha / 2.0)

    def trial(_i: int) -> bool:
        x = rng.lognormal(mu, sigma, n)
        y = round_fn(x)
        exact = rng.random(n) < share
        y[exact] = x[exact]
        point, se = log_quantile_se(y, q, alpha=alpha)
        return abs(math.log(point) - true_log_q) <= z * se

    cov = replicate(n_sims, trial).rate
    floor = (1.0 - alpha) - 4 * mcse(1.0 - alpha, n_sims)
    assert cov >= floor, f"{kind} n={n} q={q} share={share}: cov={cov} < floor={floor}"


def _price_endings(rng: np.random.Generator, lam: float, n: int) -> np.ndarray:
    """Poisson counts with half the positive values recorded 0.01 lower,
    the .99/.00 pattern of prices."""
    y = rng.poisson(lam, n).astype(float)
    y[(y > 0) & (rng.random(n) < 0.5)] -= 0.01
    return y


def _price_endings_quantile(lam: float, q: float) -> float:
    k = np.arange(0, int(poisson.isf(1e-12, lam)) + 2)
    pmf = poisson.pmf(k, lam)
    values = np.concatenate([[0.0], k[1:] - 0.01, k[1:].astype(float)])
    masses = np.concatenate([[pmf[0]], pmf[1:] / 2.0, pmf[1:] / 2.0])
    order = np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(masses[order]), q)])


def _price_endings_coverage(lam: float, n: int, q: float, alpha: float, n_sims: int) -> float:
    rng = np.random.default_rng(_seed("price_endings", lam, n, q, alpha, n_sims))
    true_log_q = math.log(_price_endings_quantile(lam, q))
    z = norm.isf(alpha / 2.0)

    def trial(_i: int) -> bool:
        point, se = log_quantile_se(_price_endings(rng, lam, n), q, alpha=alpha)
        return abs(math.log(point) - true_log_q) <= z * se

    return replicate(n_sims, trial).rate


def test_price_endings_coverage_smoke_fast():
    """Poisson(3), n=100, q=0.9 with .99/.00 endings: the classical-sized
    interval covered 78% here against 95%."""
    assert _price_endings_coverage(3.0, 100, 0.9, 0.05, n_sims=150) > 0.90


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("lam,n,q", [(3.0, 100, 0.9), (5.0, 100, 0.9), (8.0, 100, 0.5)])
def test_price_endings_coverage(lam, n, q):
    n_sims, alpha = 1000, 0.05
    cov = _price_endings_coverage(lam, n, q, alpha, n_sims)
    floor = (1.0 - alpha) - 4 * mcse(1.0 - alpha, n_sims)
    assert cov >= floor, f"lam={lam} n={n} q={q}: cov={cov} < floor={floor}"


# --- Two-arm null size on lattices ------------------------------------------
# `quantile_half_width`'s quarter-cell allowance keeps a lattice-quantile lift at
# size: without it, seconds-rounded medians at n=30000 reject a true null ~6% of
# the time at alpha=0.05 (1.5% with it). Per-arm cells pass with any allowance.


def _lattice_draw(kind: str, rng: np.random.Generator, n: int) -> np.ndarray:
    mu, sigma, round_fn, _ = _ROUNDED_KINDS[kind]
    return round_fn(rng.lognormal(mu, sigma, n))


def _null_rejection_rate(kind: str, n: int, q: float, alpha: float, n_sims: int) -> float:
    """Share of independent same-population arm pairs the production lift
    decision rejects: its p-value is dual to the reported interval, so
    ``p <= alpha`` exactly when that interval excludes a zero log ratio."""
    from increment.estimation.quantile import _quantile_p_value

    rng = np.random.default_rng(_seed("two_arm_null", kind, n, q, alpha, n_sims))

    def trial(_i: int) -> bool:
        control = _lattice_draw(kind, rng, n)
        treatment = _lattice_draw(kind, rng, n)
        return _quantile_p_value(control, treatment, q, 0.0) <= alpha

    return replicate(n_sims, trial).rate


def test_two_arm_null_size_on_a_lattice_smoke_fast():
    """Millisecond-rounded medians at n=5000: an unresolved cell, where the
    allowance is active."""
    n_sims, alpha = 200, 0.05
    assert _null_rejection_rate("ms", 5000, 0.5, alpha, n_sims) <= alpha + 4 * mcse(alpha, n_sims)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize(
    "kind,n,n_sims",
    [("seconds", 30000, 20000), ("ms", 10000, 6000)],
    ids=["seconds", "milliseconds"],
)
def test_two_arm_null_size_on_a_lattice(kind, n, n_sims):
    """Unresolved median cells, the first the one the allowance was measured
    on, replicated enough that the rate without it (about 6%) fails: the
    lift between two independent draws from one lattice population rejects
    a true null at most alpha of the time, within Monte Carlo error."""
    alpha = 0.05
    rate = _null_rejection_rate(kind, n, 0.5, alpha, n_sims)
    ceiling = alpha + 4 * mcse(alpha, n_sims)
    assert rate <= ceiling, f"{kind} n={n}: null rejection {rate} > {ceiling} ({n_sims} sims)"


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("kind", ["continuous", "cents", "ms"])
def test_resolution_calibrated_width_cost_at_n5000(kind):
    """At n=5000 (where the classical bracket already covers), the
    tie-aware construction must add ~0% width cost, clearing the <=2%
    median-inflation target with margin."""
    n = 5000
    q = 0.9
    draws = 60
    for alpha in (0.05, 0.10):
        rng = np.random.default_rng(_seed("width_cost", kind, alpha))
        inflations = []
        for _ in range(draws):
            if kind == "continuous":
                y = rng.lognormal(1.0, 0.5, n)
            else:
                mu, sigma, round_fn, _ = _ROUNDED_KINDS[kind]
                y = round_fn(rng.lognormal(mu, sigma, n))
            _point, se, classical_se = _log_quantile_se_impl(y, q, alpha)
            inflations.append((se / classical_se) - 1.0)
        median_inflation = float(np.median(inflations))
        assert median_inflation <= 0.02, (
            f"{kind} alpha={alpha}: median inflation {median_inflation:.4f}"
        )
