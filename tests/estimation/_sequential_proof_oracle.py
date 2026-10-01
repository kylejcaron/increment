"""Exact finite Bernoulli witnesses, independent of the deployed implementation.

Each row reveals independent Bernoulli arms together. In a three-arm path the
first arm is the single control reused by both cells. These finite witnesses
do not certify the full sequential campaign, arbitrary horizons, or other
observation models.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Literal

F = Fraction
ONE = F(1)
Alternative = Literal["two-sided", "greater", "less"]


def _count(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("counts must be nonnegative integers")


def _fraction(value: Fraction | int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise ValueError("only exact integers and Fractions are supported")
    return F(value)


def _level(value: Fraction | int) -> Fraction:
    value = _fraction(value)
    if not 0 < value < 1:
        raise ValueError("level must lie strictly between zero and one")
    return value


@dataclass(frozen=True)
class Arm:
    n: int = 0
    successes: int = 0

    def __post_init__(self) -> None:
        _count(self.n)
        _count(self.successes)
        if self.successes > self.n:
            raise ValueError("successes cannot exceed observations")

    def append(self, bit: int) -> "Arm":
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError("observations must be binary integers")
        return Arm(self.n + 1, self.successes + bit)


@dataclass(frozen=True, init=False)
class Path:
    """Ordered simultaneous reveals and their exact probability under the specified DGP."""

    rows: tuple[tuple[int, ...], ...]
    probability: Fraction

    def __init__(self, rows: Sequence[Sequence[int]], probability: Fraction) -> None:
        rows = tuple(tuple(row) for row in rows)
        if not rows or len(rows[0]) not in (2, 3):
            raise ValueError("a path needs a positive horizon and two or three arms")
        if any(
            len(row) != len(rows[0])
            or any(type(bit) is not int or bit not in (0, 1) for bit in row)
            for row in rows
        ):
            raise ValueError("all reveals must have the same binary arm roster")
        probability = _fraction(probability)
        if not 0 <= probability <= 1:
            raise ValueError("path probability must lie in [0, 1]")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "probability", probability)

    def prefixes(self) -> Iterator[tuple[Arm, ...]]:
        """Include the empty prefix; never replace an ordered path by final counts."""
        arms = (Arm(),) * len(self.rows[0])
        yield arms
        for row in self.rows:
            arms = tuple(arm.append(bit) for arm, bit in zip(arms, row, strict=True))
            yield arms


@dataclass(frozen=True, init=False)
class Stopped:
    look: int
    evidence: tuple[Fraction, ...]
    selected: tuple[int, ...]

    def __init__(self, look: int, evidence: Sequence[Fraction], selected: Sequence[int]) -> None:
        _count(look)
        evidence = tuple(_fraction(value) for value in evidence)
        selected = tuple(selected)
        if not evidence or any(value < 0 for value in evidence):
            raise ValueError("a stopped roster needs nonnegative evidence")
        if any(type(i) is not int or not 0 <= i < len(evidence) for i in selected):
            raise ValueError("selection must refer to the retained roster")
        if len(set(selected)) != len(selected):
            raise ValueError("selected indices must be unique")
        object.__setattr__(self, "look", look)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "selected", selected)

    def false_discovery_proportion(self, null: Sequence[int]) -> Fraction:
        """Exact FDP of the stopped roster: selected null cells over all selections."""
        null = tuple(null)
        if any(type(i) is not int or not 0 <= i < len(self.evidence) for i in null):
            raise ValueError("null cells must refer to the retained roster")
        if not self.selected:
            return F(0)
        return F(sum(1 for i in self.selected if i in null), len(self.selected))

    @property
    def all_null_fdp(self) -> Fraction:
        """FDP when every retained cell is null, which is one on any rejection."""
        return self.false_discovery_proportion(range(len(self.evidence)))


def _rising(value: Fraction, count: int) -> Fraction:
    result = F(1)
    for offset in range(count):
        result *= value + offset
    return result


@cache
def predictive(arm: Arm, a: Fraction = ONE, b: Fraction = ONE) -> Fraction:
    """Ordered-sequence Q=(a)_s(b)_(n-s)/(a+b)_n; no binomial factor.

    Empty products are one, so Q(empty)=1. For Beta(1,1) this is
    1/((n+1)*choose(n,s)). Nonpositive or inexact prior shapes are invalid.
    """
    a, b = _fraction(a), _fraction(b)
    if a <= 0 or b <= 0:
        raise ValueError("Beta shapes must be positive")
    return _rising(a, arm.successes) * _rising(b, arm.n - arm.successes) / _rising(a + b, arm.n)


def likelihood(arm: Arm, probability: Fraction) -> Fraction:
    """p^s(1-p)^(n-s), with 0^0=1 and empty-arm likelihood identically one."""
    probability = _fraction(probability)
    if not 0 <= probability <= 1:
        raise ValueError("Bernoulli probability must lie in [0, 1]")
    return probability**arm.successes * (1 - probability) ** (arm.n - arm.successes)


def unrestricted(arm: Arm) -> Fraction:
    return likelihood(arm, F(arm.successes, arm.n) if arm.n else F(0))


@cache
def null_supremum(control: Arm, treatment: Arm, alternative: Alternative) -> Fraction:
    """Exact ratio=1 supremum, allowing limits at population pC=0.

    Equality pools successes and counts. For greater (pT<=pC) or less
    (pT>=pC), feasible separate MLEs maximize the concave log likelihood;
    otherwise the optimum is on the pooled equality boundary. An empty arm
    can match the observed arm, for every direction. Both empty give one.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("unknown alternative")
    if not control.n or not treatment.n:
        return unrestricted(control) * unrestricted(treatment)
    delta = treatment.successes * control.n - control.successes * treatment.n
    if (alternative == "greater" and delta <= 0) or (alternative == "less" and delta >= 0):
        return unrestricted(control) * unrestricted(treatment)
    pooled = F(control.successes + treatment.successes, control.n + treatment.n)
    return likelihood(control, pooled) * likelihood(treatment, pooled)


@cache
def evidence(control: Arm, treatment: Arm, alternative: Alternative) -> Fraction:
    q = predictive(control) * predictive(treatment)
    return q / null_supremum(control, treatment, alternative)


def _probabilities(values: Sequence[Fraction | int]) -> tuple[Fraction, ...]:
    if not isinstance(values, Sequence) or len(values) not in (2, 3):
        raise ValueError("use a positive horizon and two or three arm probabilities")
    probabilities = tuple(_fraction(value) for value in values)
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    return probabilities


def reveal_support(
    probabilities: Sequence[Fraction | int],
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Every reveal row of positive probability, paired with its exact mass.

    Degenerate arm probabilities shrink the support without perturbing any
    exact path probability, because the omitted rows carry mass zero. The
    masses of the retained rows therefore still sum to one.
    """
    validated = _probabilities(probabilities)
    support = []
    for row in product((0, 1), repeat=len(validated)):
        mass = ONE
        for bit, p in zip(row, validated, strict=True):
            mass *= p if bit else 1 - p
        if mass:
            support.append((row, mass))
    return tuple(support)


def paths_heterogeneous(horizon: int, probabilities: Sequence[Fraction | int]) -> Iterator[Path]:
    _count(horizon)
    if not horizon:
        raise ValueError("use a positive horizon and two or three arm probabilities")
    probabilities = _probabilities(probabilities)
    rows = tuple(product((0, 1), repeat=len(probabilities)))
    for reveals in product(rows, repeat=horizon):
        probability = F(1)
        for row in reveals:
            for bit, p in zip(row, probabilities, strict=True):
                probability *= p if bit else 1 - p
        yield Path(reveals, probability)


def paths(horizon: int, n_arms: int, p: Fraction) -> Iterator[Path]:
    """Enumerate all 2^(n_arms*horizon) ordered paths, including zero-mass ones."""
    _count(horizon)
    if not horizon or type(n_arms) is not int or n_arms not in (2, 3):
        raise ValueError("use a positive horizon and two or three arms")
    p = _fraction(p)
    if not 0 <= p <= 1:
        raise ValueError("null probability must lie in [0, 1]")
    yield from paths_heterogeneous(horizon, (p,) * n_arms)


def first_rejection(path: Path, alpha: Fraction, alternative: Alternative) -> Stopped:
    """Stop at the first E>=1/alpha, or report the last prefix if none rejects."""
    alpha = _level(alpha)
    if len(path.rows[0]) != 2:
        raise ValueError("singleton stopping requires two arms")
    for look, arms in enumerate(path.prefixes()):
        value = evidence(arms[0], arms[1], alternative)
        selected = (0,) if alpha * value >= 1 else ()
        if selected or look == len(path.rows):
            return Stopped(look, (value,), selected)
    raise AssertionError("a finite nonempty path has a terminal prefix")


def e_bh(values: Sequence[Fraction], q: Fraction, *, m: int) -> tuple[int, ...]:
    """Exact fixed-roster e-BH, E_(k)>=m/(q*k), including boundary ties.

    Zero-evidence cells remain in m. Passing a shortened roster is invalid.
    """
    q = _level(q)
    _count(m)
    values = tuple(_fraction(value) for value in values)
    if not m or len(values) != m or any(value < 0 for value in values):
        raise ValueError("one nonnegative evidence value is required per retained cell")
    ordered = sorted(values, reverse=True)
    qualifying = [k for k, value in enumerate(ordered, 1) if q * k * value >= m]
    if not qualifying:
        return ()
    threshold = F(m) / (q * max(qualifying))
    return tuple(i for i, value in enumerate(values) if value >= threshold)


def family_stop(path: Path, q: Fraction, alternative: Alternative) -> Stopped:
    """Predictably stop one reveal after E1+E2>=1/q, capped at the horizon.

    The decision to stop at t uses only the evidence at t-1, including both
    shared-control cells. Selection uses current evidence at t, not a past
    maximum or the evidence that scheduled the stop. The roster always has m=2.
    """
    q = _level(q)
    if len(path.rows[0]) != 3:
        raise ValueError("the shared-control family requires three arms")
    scheduled = False
    for look, (control, first, second) in enumerate(path.prefixes()):
        values = (evidence(control, first, alternative), evidence(control, second, alternative))
        if scheduled or look == len(path.rows):
            return Stopped(look, values, e_bh(values, q, m=2))
        scheduled = q * sum(values, F(0)) >= 1
    raise AssertionError("a finite nonempty path has a terminal prefix")


def adaptive_frozen_stop(path: Path, q: Fraction, alternative: Alternative) -> Stopped:
    """Bounded four-look shared-control witness with one frozen and one missing cell.

    The path has one shared control and two observed treatments.  The retained
    roster is always ``m=3``: cell 0 is frozen at the midpoint checkpoint and
    cell 2 is structurally missing, hence has exact evidence zero.  Only even
    retained indices (0 and 2) can schedule a stop, starting at look 1.
    """
    q = _level(q)
    if len(path.rows[0]) != 3 or len(path.rows) != 4:
        raise ValueError("the adaptive witness requires exactly four three-arm reveals")
    frozen: Fraction | None = None
    for look, (control, first, second) in enumerate(path.prefixes()):
        current = (
            evidence(control, first, alternative),
            evidence(control, second, alternative),
            F(0),
        )
        active = (current[0] if frozen is None else frozen, current[1], F(0))
        should_stop = look >= 1 and any(active[i] > 1 for i in (0, 2))
        if look == len(path.rows) // 2:
            frozen = current[0] if frozen is None else frozen
            active = (frozen, current[1], F(0))
        if should_stop or look == len(path.rows):
            return Stopped(look, active, e_bh(active, q, m=3))
    raise AssertionError("a finite nonempty path has a terminal prefix")


@cache
def log_bounds(value: Fraction, terms: int = 192) -> tuple[Fraction, Fraction]:
    """Independent rational enclosure of log(value), with an explicit tail bound.

    Reduce x=2^k*y with 1<=y<=2. For z=(y-1)/(y+1), log(y) equals
    2*sum(z^(2j+1)/(2j+1), j>=0). After N terms its nonnegative tail is
    at most 2*z^(2N+1)/((2N+1)*(1-z^2)). Here 0<=z<=1/3.
    """
    value = _fraction(value)
    _count(terms)
    if value <= 0 or not terms:
        raise ValueError("log requires a positive value and positive term count")
    if value == 1:
        return F(0), F(0)
    if value < 1:
        lo, hi = log_bounds(1 / value, terms)
        return -hi, -lo
    if value > 2:
        power = value.numerator.bit_length() - value.denominator.bit_length()
        if value < 2**power:
            power -= 1
        lo, hi = log_bounds(value / 2**power, terms)
        two_lo, two_hi = log_bounds(F(2), terms)
        return lo + power * two_lo, hi + power * two_hi
    z = (value - 1) / (value + 1)
    term, partial = z, F(0)
    for j in range(terms):
        partial += 2 * term / (2 * j + 1)
        term *= z * z
    tail = 2 * term / ((2 * terms + 1) * (1 - z * z))
    return partial, partial + tail
