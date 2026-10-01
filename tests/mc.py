"""Shared Monte-Carlo coverage/MCSE helpers for statistical test suites.

Interval-coverage checks are scattered across the estimation and breakout
test tiers, most hand-rolling the same shape: loop *reps* times, count how
often an interval covers the truth, divide by *reps*. Smoke-tier thresholds
on those loops were routinely eyeballed from one seed's output rather than
stated as a formula, so a threshold a hair too tight would flake under a
different seed with no way to tell "genuinely broken" from "binomial noise."

``mcse`` is the textbook binomial Monte-Carlo standard error of an observed
coverage proportion; ``Coverage`` accumulates hits/reps and exposes it;
``nominal_band`` turns a target coverage into a ``[lo, hi]`` window stated
as ``nominal +/- k * mcse(nominal, reps)`` instead of a bare literal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


def mcse(p: float, reps: int) -> float:
    """Binomial Monte-Carlo standard error of a coverage proportion *p*
    estimated from *reps* independent replications."""
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    return math.sqrt(p * (1.0 - p) / reps)


def nominal_band(nominal: float, reps: int, k: float = 2.0) -> tuple[float, float]:
    """``[nominal - k*mcse, nominal + k*mcse]``, clipped to ``[0, 1]``.

    The standard "not seed-tuned" smoke bound: *nominal* is the coverage a
    replication loop is expected to hit (asymptotically, or as measured by
    a sibling ``parameter_recovery`` test at a larger rep count); *k*
    trades false-failure risk against how tight a check on a badly broken
    interval this actually is. ``k=2`` is the usual "nominal-coverage"
    check; a smoke test with few replications typically wants ``k=3`` or
    more so it doesn't flake on RNG variance alone.
    """
    se = mcse(nominal, reps)
    return max(0.0, nominal - k * se), min(1.0, nominal + k * se)


def binomial_error_upper_bound(k: int, reps: int, eta: float) -> float:
    """Exact one-sided ``(1 - eta)`` upper confidence bound on a Bernoulli
    error rate, given *k* errors among *reps* independent replications.

    ``beta.isf(eta, k+1, reps-k)`` -- the exact Clopper-Pearson upper
    endpoint, not a Normal/plug-in approximation -- per the repository's
    scientific-tolerance policy (choose ``delta`` via :func:`scientific_delta`,
    cap the required MC one-sided margin at ``delta/2``, and derive *eta*
    per gated cell-statistic via :func:`family_eta`). ``k=reps`` returns
    exactly ``1.0`` (no finite beta quantile exists at that domain edge).
    Never derive this by forming ``1 - eta`` and reading a quantile off
    the complementary tail -- that silently swaps upper and lower bounds.
    """
    if not 0 <= k <= reps:
        raise ValueError(f"k must be in [0, reps], got k={k}, reps={reps}")
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    if k == reps:
        return 1.0
    from scipy.stats import beta

    return float(beta.isf(eta, k + 1, reps - k))


def coverage_lower_bound(hits: int, reps: int, eta: float) -> float:
    """Exact one-sided ``(1 - eta)`` lower confidence bound on a Bernoulli
    coverage rate, given *hits* among *reps* independent replications.

    Inverts :func:`binomial_error_upper_bound` on the miss count
    (``reps - hits``): a coverage lower bound is a miss-rate upper bound
    reflected through 1, per the repository's coverage-calibration contract.

    That reflection is an identity, not a second bound. This reads the same
    beta quantile as ``binomial_error_upper_bound(reps - hits, reps, eta)``,
    and ``{true coverage < this}`` is the *same event* as ``{true error rate
    > that}``, so a gate that reports both -- and the MC margin, a third
    reading of that one quantile -- spends ONE Bonferroni slot, not two or
    three. Never re-split them when censusing slots for :func:`family_eta`:
    inflating *m* does not buy conservatism, it just wastes the budget and
    forces replication counts nobody can run.
    """
    return 1.0 - binomial_error_upper_bound(reps - hits, reps, eta)


def scientific_delta(q: float) -> float:
    """The repository's absolute excess-error tolerance at nominal
    error/noncoverage *q*: ``min(.005, .1*q)``.

    A flat ``.005`` allowance would swamp a tiny nominal *q* (e.g.
    ``q=.025`` needs ``delta=.0025``, not ``.005``); this scales down for
    q below .05 and saturates at .005 for q at/above .05.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return min(0.005, 0.1 * q)


def family_eta(family_alpha: float, m: int) -> float:
    """Per-directional-bound MC decision-error budget for *m* gated
    cell-statistics sharing one family-wise MC error allocation
    *family_alpha* (the repository default is ``.01``):
    ``family_alpha / (2*m)``, frozen against the complete preregistered
    manifest rather than reset per test file.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError(f"family_alpha must be in (0, 1), got {family_alpha}")
    return family_alpha / (2.0 * m)


def kl_chernoff_upper_bound(x: float, reps: int, eta: float) -> float:
    """Exact-KL (Chernoff) one-sided upper confidence bound for a
    FRACTIONAL bounded observation's population mean (e.g. per-replication
    FDP/FCP), not a Bernoulli count.

    Solves ``reps * [x*log(x/u) + (1-x)*log((1-x)/(1-u))] = log(1/eta)``
    for the smallest ``u in [x, 1]``, with the continuous boundary limits
    ``x=0`` (solves ``reps*log(1/(1-u)) = log(1/eta)``) and ``x=1``
    (``u=1``). Never reuse a binomial MCSE/Clopper-Pearson interval on a
    fractional per-replication observation: its variance is bounded, not
    ``Binomial(reps, x)``.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    if x >= 1.0:
        return 1.0

    target = -math.log(eta) / reps

    def kl(u: float) -> float:
        if u >= 1.0:
            return math.inf
        term_hi = 0.0 if x == 0.0 else x * math.log(x / u)
        term_lo = (1.0 - x) * math.log((1.0 - x) / (1.0 - u))
        return term_hi + term_lo

    lo, hi = x, 1.0 - 1e-15
    if kl(hi) - target <= 0.0:
        return 1.0

    from scipy.optimize import brentq

    return float(brentq(lambda u: kl(u) - target, lo, hi, xtol=1e-14, rtol=1e-14))


@dataclass
class Coverage:
    """Accumulates hit/miss outcomes from a replication loop and reports
    the resulting coverage proportion and its Monte-Carlo standard error.

    Replaces the ``hits = 0; ...; hits += cond; ...; return hits / reps``
    idiom duplicated across the coverage test suites -- callers ``record``
    each replication's hit and read ``.rate``/``.mcse``/``.band`` instead
    of re-deriving the division and the MCSE formula at every call site.
    """

    reps: int = 0
    hits: int = 0

    def record(self, hit: bool) -> None:
        self.reps += 1
        self.hits += bool(hit)

    @property
    def rate(self) -> float:
        if self.reps == 0:
            raise ValueError("no replications recorded yet")
        return self.hits / self.reps

    @property
    def mcse(self) -> float:
        return mcse(self.rate, self.reps)

    def band(self, k: float = 2.0) -> tuple[float, float]:
        """``[rate - k*mcse, rate + k*mcse]``, clipped to ``[0, 1]`` --
        for reporting the achieved coverage's own uncertainty, as opposed
        to ``nominal_band`` which states a bound around a *target*."""
        se = self.mcse
        return max(0.0, self.rate - k * se), min(1.0, self.rate + k * se)


@dataclass
class CoverageSet:
    """Named group of :class:`Coverage` accumulators sharing one
    replication loop -- e.g. the "clustered vs iid" or "cuped vs raw" twin
    coverages every cluster/CUPED coverage test compares in one pass.

    ``covset.record(clustered=cl_hit, iid=iid_hit)`` records one
    replication's outcome for every named accumulator in one call, keeping
    the loop body a single line instead of two independent counters.
    """

    _coverages: dict[str, Coverage] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Coverage:
        return self._coverages.setdefault(name, Coverage())

    def record(self, **hits: bool) -> None:
        for name, hit in hits.items():
            self[name].record(hit)

    def rates(self, *names: str) -> tuple[float, ...]:
        return tuple(self[name].rate for name in names)


def replicate(reps: int, trial: Callable[[int], bool]) -> Coverage:
    """Run ``trial(i)`` for ``i in range(reps)``, recording each boolean
    result into a fresh :class:`Coverage`. A thin convenience for the
    common single-interval replication loop."""
    cov = Coverage()
    for i in range(reps):
        cov.record(trial(i))
    return cov


def replicate_many(
    reps: int, names: Iterable[str], trial: Callable[[int], dict[str, bool]]
) -> CoverageSet:
    """Run ``trial(i)`` for ``i in range(reps)``; each call returns a
    ``{name: hit}`` mapping covering every name in *names*, recorded into a
    shared :class:`CoverageSet`."""
    names = tuple(names)
    covset = CoverageSet()
    for i in range(reps):
        hits = trial(i)
        missing = set(names) - hits.keys()
        if missing:
            raise ValueError(f"trial({i}) did not report hits for {sorted(missing)}")
        covset.record(**hits)
    return covset
