"""Exact finite-roster evaluation; see protocols/randomized_saturation.md."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction as F
from itertools import combinations, product
from math import comb, isfinite, prod
from statistics import NormalDist


@dataclass(frozen=True)
class _Tiny:
    name: str
    high_shifts: tuple[F, ...]
    low_shift: F
    target: F


_SIZES = (3, 3, 3, 3, 6)
_ZERO = (F(),) * 5
_SHARP = _Tiny("unequal/sharp", _ZERO, F(), F())
_UNIT = _Tiny("unequal/shift", (F(1),) * 5, F(), F(1))
_HET = _Tiny("unequal/heterogeneous", tuple(map(F, (2, -1, 0, 1, -1))), F(), F())
_LEAK = _Tiny("unequal/leakage", (F(1, 4),) * 5, F(1, 2), F())


def _equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise AssertionError(f"{name}: observed {observed}, expected {expected}")


def _check_tiny(case: _Tiny, tally: _Tally, naive: F) -> None:
    # The reference integrates within-cluster sampling algebraically, without enumeration.
    m_h, m_l, groups, people = 2, 3, len(_SIZES), sum(_SIZES)
    h, l = [], []
    within = F()
    for g, n in enumerate(_SIZES):
        baseline = g + F(n - 1, 2)
        h.append(n * (baseline + case.high_shifts[g]))
        l.append(n * (baseline + case.low_shift))
        person_var = F(n * (n + 1), 12)  # finite-population variance of 0,...,n-1
        for u, m in ((n // 3, m_h), (2 * n // 3, m_l)):
            within += F(n * (n - u), groups * u * m) * person_var
    diff = [a - b for a, b in zip(h, l, strict=True)]
    weight = F(groups, people)

    # Pairwise differences give sample variance independent of the candidate's centering.
    def pair_variance(values: list[F]) -> F:
        return sum(((a - b) * (a - b) for a, b in combinations(values, 2)), F()) / (
            groups * (groups - 1)
        )

    neyman = weight * weight * (pair_variance(h) / m_h + pair_variance(l) / m_l + within)
    _equal(case.name + " assignments", (tally.states, tally.mass), (12150, 12150))
    _equal(case.name + " expectation", tally.mean, sum(diff, F()) / people)
    _equal(
        case.name + " sampling variance",
        tally.variance,
        neyman - groups * pair_variance(diff) / (people * people),
    )
    _equal(case.name + " mean variance estimate", tally.mean_v, neyman)
    if case in (_SHARP, _UNIT):
        unit = int(case == _UNIT)
        _equal("pooled untreated contrast", naive, F(-2, 15) + unit)
        _equal("sampling variance anchor", tally.variance, F(20189 if unit else 18245, 1296))
        _equal("mean variance estimate anchor", tally.mean_v, F(20225 if unit else 18245, 1296))
    _equal(
        case.name + " controlled contrast", tally.mean, F(-1, 4) if case == _LEAK else case.target
    )


def _check_strata(tally: _Tally) -> None:
    _equal("stratified support", tally.states, comb(4, 2) * comb(5, 3))
    _equal("stratified mean", tally.mean, 0)
    expected = F(4**2 * 15 + 5**2 * 75, 42**2)
    _equal("stratified sampling variance", tally.variance, expected)
    _equal("stratified mean variance estimate", tally.mean_v, expected)


def _check_hyper(m: int, x: int, effect: F, fit: _Fit) -> None:
    p = F(x, m)
    _equal("hypergeometric estimate", fit.point, effect + 2 * p - 1)
    _equal("hypergeometric variance", fit.v, 2 * p * (1 - p) / (m - 1) if m > 1 else None)


@dataclass(frozen=True)
class _Arm:
    n: int
    mean: F
    variance_of_mean: F | None


def _arm(totals: Sequence[F]) -> _Arm:
    n = len(totals)
    mean = sum(totals, F()) / n
    ss = sum(((t - mean) * (t - mean) for t in totals), F())
    return _Arm(n, mean, ss / (n * (n - 1)) if n > 1 else None)


@dataclass(frozen=True)
class _Fit:
    point: F
    v: F | None


def _stratum(high: _Arm, low: _Arm, people: int) -> _Fit:
    weight = F(high.n + low.n, people)
    point = weight * (high.mean - low.mean)
    if high.variance_of_mean is None or low.variance_of_mean is None:
        return _Fit(point, None)
    return _Fit(point, weight * weight * (high.variance_of_mean + low.variance_of_mean))


_Z = F.from_float(-NormalDist().inv_cdf(0.025))
_Z2 = _Z * _Z
_LIMIT = F(55, 1000)


@dataclass
class _Tally:
    states: int = 0
    mass: int = 0
    rejected: int = 0
    missed: int = 0
    refused: Counter[str] = field(default_factory=Counter)
    sum_point: F = F()
    sum_square: F = F()
    sum_v: F = F()

    def add(self, fit: _Fit, mass: int, truth: F) -> None:
        self.states += 1
        self.mass += mass
        self.sum_point += mass * fit.point
        self.sum_square += mass * fit.point * fit.point
        if fit.v is None:
            self.refused["insufficient_clusters"] += mass
        elif not isfinite(fit.v):
            self.refused["nonfinite_variance"] += mass
        elif fit.v == 0:
            self.refused["zero_variance"] += mass
        else:
            if fit.v < 0:
                raise ArithmeticError("Negative between-cluster variance")
            self.sum_v += mass * fit.v
            threshold = _Z2 * fit.v
            self.rejected += mass * (fit.point * fit.point > threshold)
            error = fit.point - truth
            self.missed += mass * (error * error > threshold)

    @property
    def mean(self) -> F:
        return self.sum_point / self.mass

    @property
    def variance(self) -> F:
        return self.sum_square / self.mass - self.mean * self.mean

    @property
    def mean_v(self) -> F | None:
        if self.refused["insufficient_clusters"] or self.refused["nonfinite_variance"]:
            return None
        return self.sum_v / self.mass

    @property
    def rejection(self) -> F:
        return F(self.rejected, self.mass)

    @property
    def bad(self) -> F:
        return F(self.missed + self.refused.total(), self.mass)

    def null_passes(self) -> bool:
        return self.rejection <= _LIMIT and self.bad <= _LIMIT

    def report(self, label: str) -> None:
        print(
            f"{label}: states={self.states} mass={self.mass} rejected={self.rejected} "
            f"missed_available={self.missed} insufficient_clusters={self.refused['insufficient_clusters']} "
            f"zero_variance={self.refused['zero_variance']} nonfinite_variance={self.refused['nonfinite_variance']}"
        )
        print(
            f"  E(point)={self.mean} Var(point)={self.variance} E(V)={self.mean_v} "
            f"P(reject)={float(self.rejection):.8f} P(miss,available)={self.missed / self.mass:.8f} "
            f"P(unavailable)={self.refused.total() / self.mass:.8f} P(miss or unavailable)={float(self.bad):.8f}"
        )


def _untreated_choices(n: int, u: int, baseline: F) -> list[tuple[F, F]]:
    choices = []
    for indices in combinations(range(n), u):
        raw = baseline * u + sum(indices)
        choices.append((n * raw / u, raw))
    return choices


def _tiny_arms(groups: tuple[int, ...], case: _Tiny, *, high: bool) -> list[tuple[_Arm, F]]:
    choices = []
    people_observed = 0
    for g in groups:
        n = _SIZES[g]
        u = (1 if high else 2) * n // 3
        choices.append(
            _untreated_choices(n, u, g + (case.high_shifts[g] if high else case.low_shift))
        )
        people_observed += u
    return [
        (_arm([t for t, _ in draw]), sum((raw for _, raw in draw), F()) / people_observed)
        for draw in product(*choices)
    ]


def _tiny(case: _Tiny) -> None:
    tally, naive = _Tally(), F()
    groups = range(len(_SIZES))
    for high in combinations(groups, 2):
        low = tuple(g for g in groups if g not in high)
        for (h, pooled_h), (l, pooled_l) in product(
            _tiny_arms(high, case, high=True), _tiny_arms(low, case, high=False)
        ):
            tally.add(_stratum(h, l, sum(_SIZES)), 1, case.target)
            naive += pooled_h - pooled_l
    naive /= tally.mass
    _check_tiny(case, tally, naive)
    tally.report(case.name)
    print(f"  E(pooled untreated contrast)={naive}")


def _stratified() -> None:
    strata: list[list[_Fit]] = []
    for groups, n, m, shift in ((4, 3, 2, 5), (5, 6, 3, -2)):
        draws = []
        for high in combinations(range(groups), m):
            h = _arm([F(n * (g + shift)) for g in high])
            l = _arm([F(n * g) for g in range(groups) if g not in high])
            draws.append(_stratum(h, l, 42))
        strata.append(draws)
    tally = _Tally()
    for a, b in product(*strata):
        v = None if a.v is None or b.v is None else a.v + b.v
        tally.add(_Fit(a.point + b.point, v), 1, F())
    _check_strata(tally)
    tally.report("stratified/heterogeneous")


def _sentinel_arm(m: int, n: int, u: int) -> list[tuple[int, _Arm]]:
    p = F(u, n)
    rows = []
    for b in range(m + 1):
        mass = comb(m, b) * p.numerator**b * (p.denominator - p.numerator) ** (m - b)
        rows.append((mass, _arm([F(n, u)] * b + [F()] * (m - b))))
    return rows


def _sentinel(k_h: int, k_l: int) -> None:
    m, n = 40, 100
    u_h, u_l = n - k_h, n - k_l
    tally = _Tally()
    for (wh, h), (wl, l) in product(_sentinel_arm(m, n, u_h), _sentinel_arm(m, n, u_l)):
        tally.add(_stratum(h, l, 2 * m * n), wh * wl, F())
    denominator = prod(F(u, n).denominator ** m for u in (u_h, u_l))
    expected = (F(n, u_h) + F(n, u_l) - 2) / (m * n * n)
    _equal("sentinel product mass", tally.mass, denominator)
    _equal("sentinel mean", tally.mean, 0)
    _equal("sentinel sampling variance", tally.variance, expected)
    _equal("sentinel mean variance estimate", tally.mean_v, expected)
    tally.report(f"sentinel/{k_h}:{k_l}")


def _hyper(m: int, effect: F) -> _Tally:
    tally = _Tally()
    for x in range(m + 1):
        h = _arm([3 * (1 + effect)] * x + [3 * effect] * (m - x))
        l = _arm([F(3)] * (m - x) + [F()] * x)
        fit = _stratum(h, l, 6 * m)
        _check_hyper(m, x, effect, fit)
        tally.add(fit, comb(m, x) ** 2, effect)
    _equal("hypergeometric mass", tally.mass, comb(2 * m, m))
    _equal("hypergeometric mean", tally.mean, effect)
    return tally


def _planning(m: int) -> None:
    null = _hyper(m, F())
    null.report(f"hyper/{m}/null")
    if m == 1:
        _equal("unreplicated arm refusals", null.refused["insufficient_clusters"], null.mass)
        return
    grid = [null, *(_hyper(m, F(i, 100)) for i in range(1, 51))]
    print(f"  P(reject) at d=0.00,...,0.50: {', '.join(f'{float(t.rejection):.8f}' for t in grid)}")
    # Constant additive shifts preserve interval availability and coverage at the true effect.
    for t in grid:
        _equal("grid refusal mass", t.refused, null.refused)
        _equal("grid noncoverage mass", t.missed, null.missed)
    if not null.null_passes():
        print("  MDE: withheld; matched null fails")
        return
    crossing = next((i for i, t in enumerate(grid) if t.rejection >= F(4, 5)), None)
    if crossing is None:
        print("  MDE: not attained on grid")
        return
    print(f"  first 80% grid crossing: {crossing / 100:.2f}")
    for i in range(max(0, crossing - 1), crossing + 1):
        grid[i].report(f"hyper/{m}/d={i / 100:.2f}")


def main() -> None:
    print("Exact assignment masses; decision counts are weighted, states are not.")
    for case in (_SHARP, _UNIT, _HET, _LEAK):
        _tiny(case)
    _stratified()
    for high, low in ((75, 25), (99, 1)):
        _sentinel(high, low)
    for m in (1, 3, 40):
        _planning(m)


if __name__ == "__main__":
    main()
