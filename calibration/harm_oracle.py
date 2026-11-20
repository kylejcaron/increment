"""Audit harm bounds by enumerating potential tables, simplex vertices and samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from itertools import product
from math import comb, lcm
from pathlib import Path

import numpy as np
import scipy
from pydantic import BaseModel, ConfigDict

_BASE = "77ab667b227b2005e18bcc3d0f56c23b6aaec651"
_PROTOCOL = "calibration/protocols/harm_study_v1.md"
_ATOL = 1e-11
_ARMS = ((4, 4), (9, 11), (25, 20))
_GRID = tuple(map(Fraction, ("0", ".01", ".2", ".4", ".5", ".99", "1")))
_THRESHOLDS = tuple(
    map(
        Fraction,
        (
            ".0025",
            ".005",
            ".01",
            ".025",
            ".05",
            ".10",
            ".125",
            ".15",
            ".20",
            ".25",
            ".30",
            ".32",
            ".50",
        ),
    )
)

type _Table = tuple[int, int, int, int]
type _Margins = tuple[int, int]
type _Pair = tuple[Fraction, Fraction]
type _CP = tuple[_Pair, _Pair]


def potential_tables(n: int) -> Iterator[_Table]:
    """Enumerate tables in (00, 01, 10, 11) order with integer total n."""
    for c00 in range(n + 1):
        for c01 in range(n - c00 + 1):
            for c10 in range(n - c00 - c01 + 1):
                yield c00, c01, c10, n - c00 - c01 - c10


def _margins(table: _Table) -> _Margins:
    _, c01, c10, c11 = table
    return c10 + c11, c01 + c11


def potential_table_bounds(p0: Fraction, p1: Fraction) -> _Pair:
    """Find sharp harm extrema by enumerating tables with given success margins."""
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1):
        raise ValueError("success margins must lie in [0, 1]")
    n = lcm(p0.denominator, p1.denominator)
    y0, y1 = int(n * p0), int(n * p1)
    least, most = n, 0
    for c11 in range(min(y0, y1) + 1):
        c10, c01 = y0 - c11, y1 - c11
        c00 = n - c11 - c10 - c01
        if c00 >= 0:
            least, most = min(least, c10), max(most, c10)
    return Fraction(least, n), Fraction(most, n)


def _vertices(box: tuple[_Pair, ...]) -> Iterator[tuple[Fraction, ...]]:
    for free, (lo, hi) in enumerate(box):
        for fixed in product(*box[:free], *box[free + 1 :]):
            last = 1 - sum(fixed, Fraction(0))
            if lo <= last <= hi:
                yield (*fixed[:free], last, *fixed[free:])


def _require(failures: list[str], name: str, coordinate: object, *, condition: bool) -> None:
    if not condition:
        failures.append(f"{name}: {coordinate!r}")


def _identification(failures: list[str]) -> tuple[int, int, int]:
    from calibration.harm_bounds import Cell, summarize

    zero = Cell(0, 0, 0, 0)
    tables = tuple(potential_tables(8))
    pooled: dict[_Margins, set[int]] = defaultdict(set)
    for table in tables:
        pooled[_margins(table)].add(table[2])
    _require(
        failures,
        "pooled enumeration",
        (len(tables), len(pooled)),
        condition=len(tables) == 165 and len(pooled) == 81,
    )
    for (y0, y1), harms in pooled.items():
        counts = (Cell(8, y0, 8, y1), zero, zero, zero)
        plugin = summarize(counts, (0, 0, 0, 0), atom_weights=None).plugin
        lattice = potential_table_bounds(Fraction(y0, 8), Fraction(y1, 8))
        _require(
            failures,
            "fixed-margin enumeration",
            (y0, y1),
            condition=lattice == (Fraction(min(harms), 8), Fraction(max(harms), 8)),
        )
        actual = (
            set()
            if plugin is None
            else {i for i in range(9) if plugin[0] - _ATOL <= i / 8 <= plugin[1] + _ATOL}
        )
        _require(failures, "pooled harm lattice", (y0, y1), condition=actual == harms)

    pairs = 0
    stratified: dict[tuple[_Margins, _Margins], set[int]] = defaultdict(set)
    for a, b in product(potential_tables(4), repeat=2):
        stratified[_margins(a), _margins(b)].add(a[2] + b[2])
        pairs += 1
    half = Fraction(1, 2)
    for ((a0, a1), (b0, b1)), harms in stratified.items():
        counts = (Cell(4, a0, 4, a1), zero, Cell(4, b0, 4, b1), zero)
        plugin = summarize(counts, (0, 0, 1, 1), atom_weights=(half, 0, half, 0)).plugin
        close = plugin is not None and (
            abs(plugin[0] - min(harms) / 8) <= _ATOL and abs(plugin[1] - max(harms) / 8) <= _ATOL
        )
        _require(failures, "stratified endpoints", ((a0, a1), (b0, b1)), condition=close)
        envelope = pooled[(a0 + b0, a1 + b1)]
        _require(failures, "refinement", ((a0, a1), (b0, b1)), condition=harms <= envelope)

    _require(failures, "two-stratum enumeration", pairs, condition=pairs == 1225)
    _require(failures, "observational twins", (4, 4), condition=pooled[(4, 4)] == {0, 1, 2, 3, 4})
    _require(
        failures,
        "informative covariate",
        ((1, 3), (3, 1)),
        condition=stratified[((1, 3), (3, 1))] == {2, 3, 4},
    )
    _require(failures, "all harm", (8, 0), condition=pooled[(8, 0)] == {8})
    _require(failures, "pure benefit", (0, 8), condition=pooled[(0, 8)] == {0})
    _require(
        failures,
        "irrelevant covariate",
        (2, 2),
        condition=stratified[((2, 2), (2, 2))] == pooled[(4, 4)],
    )
    return len(tables), len(pooled), pairs


def _simplex(failures: list[str]) -> int:
    from calibration.harm_bounds import _weight_extreme

    # The third box admits a zero-frequency third leaf; the fourth fixes its first weight.
    cases = (
        ((".7",), ((".3", "1"),)),
        ((".1", ".5"), ((".15", ".40"), ("0", "1"))),
        ((".2", ".2", ".8"), ((".25", ".8"), (".25", ".8"), ("0", ".2"))),
        ((".6", ".1", ".1", ".3"), ((".2", ".2"), (".1", ".4"), ("0", ".6"), (".05", ".4"))),
        ((".6", ".6", ".6", ".6"), (("0", "1"),) * 4),
    )
    for raw_coefficients, raw_box in cases:
        coefficients = tuple(map(Fraction, raw_coefficients))
        box = tuple((Fraction(lo), Fraction(hi)) for lo, hi in raw_box)
        objectives = tuple(
            sum((c * w for c, w in zip(coefficients, vertex, strict=True)), Fraction(0))
            for vertex in _vertices(box)
        )
        for maximize, expected in ((False, min(objectives)), (True, max(objectives))):
            actual = _weight_extreme(
                tuple(map(float, coefficients)),
                tuple((float(lo), float(hi)) for lo, hi in box),
                maximize=maximize,
            )
            _require(
                failures,
                "simplex vertex",
                (raw_box, maximize),
                condition=abs(actual - float(expected)) <= _ATOL,
            )
    return len(cases)


def _binomial_numerators(n: int, p: Fraction, counts: range) -> Iterator[int]:
    a, b = p.as_integer_ratio()
    for k in counts:
        yield comb(n, k) * a**k * (b - a) ** (n - k)


def _tail(n: int, p: Fraction, counts: range) -> Fraction:
    return Fraction(sum(_binomial_numerators(n, p, counts)), p.denominator**n)


def _invert_tail(n: int, counts: range, delta: Fraction, *, increasing: bool) -> _Pair:
    lo, hi = Fraction(0), Fraction(1)
    for _ in range(80):
        mid = (lo + hi) / 2
        mass = _tail(n, mid, counts)
        if mass == delta:
            return mid, mid
        if (mass < delta) == increasing:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _exact_cp(k: int, n: int, delta: Fraction) -> _CP:
    zero, one = Fraction(0), Fraction(1)
    lo = (zero, zero) if k == 0 else _invert_tail(n, range(k, n + 1), delta, increasing=True)
    hi = (one, one) if k == n else _invert_tail(n, range(k + 1), delta, increasing=False)
    return lo, hi


def _cp_audit(failures: list[str]) -> tuple[dict[int, tuple[_CP, ...]], float]:
    from calibration.harm_bounds import _cp_interval

    exact: dict[int, tuple[_CP, ...]] = {}
    largest = 0.0
    for leaves in range(1, 5):
        delta = Fraction(1, 600 * leaves)
        for n in sorted({0, *(n for pair in _ARMS for n in pair)}):
            roots = tuple(_exact_cp(k, n, delta) for k in range(n + 1))
            if leaves == 1:
                exact[n] = roots
            for k, brackets in enumerate(roots):
                limits = _cp_interval(k, n, leaves)
                error = max(
                    abs(value - float(edge))
                    for value, bracket in zip(limits, brackets, strict=True)
                    for edge in bracket
                )
                largest = max(largest, error)
                _require(failures, "CP root", (leaves, k, n, error), condition=error <= _ATOL)
    return exact, largest


def _corner(p: Fraction, q: Fraction) -> _Pair:
    # A vertex of the joint-table line has at least one zero mass. Try each zero cell.
    candidates = (
        (q, 1 - p - q, Fraction(0), p),
        (1 - p, Fraction(0), p + q - 1, 1 - q),
        (Fraction(0), 1 - p, q, p - q),
        (q - p, 1 - q, p, Fraction(0)),
    )
    harms = tuple(table[2] for table in candidates if min(table) >= 0)
    return min(harms), max(harms)


def _actions(lo: Fraction, hi: Fraction) -> tuple[str, ...]:
    margin = Fraction(1, 10**12)
    return tuple(
        "clear" if hi + margin < tau else "rule_out" if lo > tau + margin else "inconclusive"
        for tau in _THRESHOLDS
    )


@dataclass(frozen=True, slots=True)
class _Sample:
    n0: int
    n1: int
    k0: int
    k1: int
    inward: tuple[_Pair, _Pair]
    outward: tuple[_Pair, _Pair]
    actions: tuple[str, ...]
    actual: tuple[tuple[float, float], tuple[float, float]]


def _samples(
    n0: int, n1: int, roots: dict[int, tuple[_CP, ...]], failures: list[str]
) -> tuple[_Sample, ...]:
    from calibration.harm_bounds import Cell, decide, summarize

    zero = Cell(0, 0, 0, 0)
    samples = []
    for k0, k1 in product(range(n0 + 1), range(n1 + 1)):
        plo, phi = roots[n0][k0]
        qlo, qhi = roots[n1][k1]
        inward = _corner(plo[1], qlo[1]), _corner(phi[0], qhi[0])
        outward = _corner(plo[0], qlo[0]), _corner(phi[1], qhi[1])
        band = summarize(
            (Cell(n0, k0, n1, n1 - k1), zero, zero, zero), (0, 0, 0, 0), atom_weights=None
        )
        actual = (band.L_ci[0], band.U_ci[0]), (band.L_ci[1], band.U_ci[1])
        coord = (n0, n1, k0, k1)
        for projected in (inward, outward):
            close = all(
                abs(actual[side][end] - float(projected[side][end])) <= _ATOL
                for side, end in product(range(2), repeat=2)
            )
            _require(failures, "projected interval", coord, condition=close)
        actions = tuple(decide(band.outer, float(tau)) for tau in _THRESHOLDS)
        _require(
            failures,
            "threshold actions",
            coord,
            condition=actions
            == _actions(inward[0][0], inward[1][1])
            == _actions(outward[0][0], outward[1][1]),
        )
        samples.append(_Sample(n0, n1, k0, k1, inward, outward, actions, actual))
    return tuple(samples)


def _membership(n: int, p: Fraction) -> tuple[tuple[int, ...], tuple[Fraction, ...]]:
    masses = tuple(Fraction(x, p.denominator**n) for x in _binomial_numerators(n, p, range(n + 1)))
    delta = Fraction(1, 600)
    accepted = tuple(
        k for k in range(n + 1) if sum(masses[: k + 1]) >= delta and sum(masses[k:]) >= delta
    )
    return accepted, masses


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class _Coordinate(_Frozen):
    n0: int
    n1: int
    p: float
    q: float


class _Probability(_Frozen):
    value: float
    exact: str

    @classmethod
    def of(cls, value: Fraction) -> _Probability:
        return cls(value=float(value), exact=str(value))


class _Actions(_Frozen):
    tau: float
    false_clear: _Probability
    false_rule_out: _Probability


class _SamplingCell(_Frozen):
    coordinate: _Coordinate
    box: _Probability
    L: _Probability
    U: _Probability
    joint: _Probability
    decisions: tuple[_Actions, ...]


def _coverage(
    sample: _Sample, truth: _Pair, failures: list[str], coordinate: _Coordinate
) -> tuple[bool, bool]:
    result = []
    for j, value in enumerate(truth):
        covered = sample.inward[0][j] <= value <= sample.inward[1][j]
        enclosing = sample.outward[0][j] <= value <= sample.outward[1][j]
        actual = sample.actual[0][j] <= value <= sample.actual[1][j]
        _require(
            failures,
            "endpoint classification",
            (coordinate, sample.k0, sample.k1, j),
            condition=covered == enclosing == actual,
        )
        result.append(covered)
    return result[0], result[1]


def _sampling_cell(
    n0: int, n1: int, p: Fraction, q: Fraction, samples: tuple[_Sample, ...], failures: list[str]
) -> _SamplingCell:
    coord = _Coordinate(n0=n0, n1=n1, p=float(p), q=float(q))
    truth = potential_table_bounds(p, 1 - q)
    k0_inside, mass0 = _membership(n0, p)
    k1_inside, mass1 = _membership(n1, q)
    probabilities = [Fraction(0)] * 5
    wrong_clear = [Fraction(0)] * len(_THRESHOLDS)
    wrong_rule_out = [Fraction(0)] * len(_THRESHOLDS)
    for sample in samples:
        mass = mass0[sample.k0] * mass1[sample.k1]
        inside = sample.k0 in k0_inside and sample.k1 in k1_inside
        L_cover, U_cover = _coverage(sample, truth, failures, coord)
        joint = L_cover and U_cover
        _require(
            failures,
            "box not contained in coverage event",
            (coord, sample.k0, sample.k1),
            condition=not inside or joint,
        )
        for i, covered in enumerate((True, inside, L_cover, U_cover, joint)):
            if covered:
                probabilities[i] += mass
        for i, (tau, action) in enumerate(zip(_THRESHOLDS, sample.actions, strict=True)):
            if action == "clear" and truth[1] >= tau:
                wrong_clear[i] += mass
                _require(
                    failures,
                    "false clearance within coverage",
                    (coord, tau, sample.k0, sample.k1),
                    condition=not U_cover,
                )
            if action == "rule_out" and truth[0] <= tau:
                wrong_rule_out[i] += mass
                _require(
                    failures,
                    "false rule-out within coverage",
                    (coord, tau, sample.k0, sample.k1),
                    condition=not L_cover,
                )
    total, box, L, U, joint = probabilities
    _require(
        failures,
        "sampling mass",
        coord,
        condition=total == 1 and joint >= box >= 1 - Fraction(4, 600),
    )
    if (n0, n1, p, q) == (25, 20, Fraction(1, 5), Fraction(2, 5)):
        _require(
            failures,
            "anchor acceptance ranges",
            (k0_inside, k1_inside),
            condition=k0_inside == tuple(range(12)) and k1_inside == tuple(range(2, 15)),
        )
        _require(
            failures, "anchor box mass", box, condition=abs(float(box) - 0.996327663382) <= _ATOL
        )
    return _SamplingCell(
        coordinate=coord,
        box=_Probability.of(box),
        L=_Probability.of(L),
        U=_Probability.of(U),
        joint=_Probability.of(joint),
        decisions=tuple(
            _Actions(
                tau=float(tau),
                false_clear=_Probability.of(clear),
                false_rule_out=_Probability.of(rule_out),
            )
            for tau, clear, rule_out in zip(_THRESHOLDS, wrong_clear, wrong_rule_out, strict=True)
        ),
    )


class _Extremum(_Frozen):
    metric: str
    coordinate: _Coordinate
    probability: _Probability
    tau: float | None = None


def _extrema(cells: tuple[_SamplingCell, ...]) -> tuple[_Extremum, ...]:
    extrema = []
    for metric in ("box", "L", "U", "joint"):
        cell = min(cells, key=lambda row: Fraction(getattr(row, metric).exact))
        extrema.append(
            _Extremum(metric=metric, coordinate=cell.coordinate, probability=getattr(cell, metric))
        )
    for metric in ("false_clear", "false_rule_out"):
        cell, decision = max(
            ((cell, decision) for cell in cells for decision in cell.decisions),
            key=lambda pair: Fraction(getattr(pair[1], metric).exact),
        )
        extrema.append(
            _Extremum(
                metric=metric,
                coordinate=cell.coordinate,
                probability=getattr(decision, metric),
                tau=decision.tau,
            )
        )
    return tuple(extrema)


class _Receipt(_Frozen):
    path: str
    sha256: str


class _Manifest(_Frozen):
    base_revision: str
    source_revision: str
    protocol_sha256: str
    sources: tuple[_Receipt, ...]
    inputs: tuple[_Receipt, ...]
    invocation: tuple[str, ...]
    python: str
    numpy: str
    scipy: str
    rng: None = None
    arm_sizes: tuple[tuple[int, int], ...] = _ARMS
    grid: tuple[str, ...] = tuple(map(str, _GRID))
    thresholds: tuple[str, ...] = tuple(map(str, _THRESHOLDS))


def _manifest(args: list[str]) -> _Manifest:
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    input_receipt = _Receipt(
        path=_PROTOCOL, sha256=hashlib.sha256((root / _PROTOCOL).read_bytes()).hexdigest()
    )
    code = ("calibration/harm_bounds.py", "calibration/harm_oracle.py", "uv.lock")
    return _Manifest(
        base_revision=_BASE,
        source_revision=source,
        protocol_sha256=input_receipt.sha256,
        sources=tuple(
            _Receipt(path=p, sha256=hashlib.sha256((root / p).read_bytes()).hexdigest())
            for p in code
        ),
        inputs=(input_receipt,),
        invocation=(sys.executable, "-m", "calibration.harm_oracle", *args),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


class _Report(_Frozen):
    manifest: _Manifest | None
    started_at: datetime
    ended_at: datetime
    status: str
    tables: int | None
    marginal_pairs: int | None
    stratum_table_pairs: int | None
    simplex_cases: int | None
    largest_cp_error: float | None
    sampling_cells: tuple[_SamplingCell, ...]
    extrema: tuple[_Extremum, ...]
    mismatches: int
    failures: tuple[str, ...]


def main(argv: list[str] | None = None) -> int:
    """Run the exact synthetic audit, recording failures and coordinates in JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    raw_args = sys.argv[1:] if argv is None else argv
    out = parser.parse_args(raw_args).out
    out.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    failures: list[str] = []
    cells: list[_SamplingCell] = []
    manifest, tables, marginal_pairs, stratum_table_pairs = None, None, None, None
    simplex_cases, largest_cp_error = None, None
    extrema = ()
    with out.open("x", encoding="utf-8") as handle:
        try:
            manifest = _manifest(raw_args)
            tables, marginal_pairs, stratum_table_pairs = _identification(failures)
            simplex_cases = _simplex(failures)
            roots, largest_cp_error = _cp_audit(failures)
            for n0, n1 in _ARMS:
                samples = _samples(n0, n1, roots, failures)
                for p, q in product(_GRID, repeat=2):
                    cells.append(_sampling_cell(n0, n1, p, q, samples, failures))
            extrema = _extrema(tuple(cells))
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        report = _Report(
            manifest=manifest,
            started_at=started,
            ended_at=datetime.now(UTC),
            status="failed" if failures else "passed",
            tables=tables,
            marginal_pairs=marginal_pairs,
            stratum_table_pairs=stratum_table_pairs,
            simplex_cases=simplex_cases,
            largest_cp_error=largest_cp_error,
            sampling_cells=tuple(cells),
            extrema=extrema,
            mismatches=len(failures),
            failures=tuple(failures),
        )
        json.dump(report.model_dump(mode="json"), handle, allow_nan=False, indent=2)
        handle.write("\n")
    print(
        f"tables={tables}, marginal_pairs={marginal_pairs}, stratum_table_pairs={stratum_table_pairs}, mismatches={report.mismatches}"
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
