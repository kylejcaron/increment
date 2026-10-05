"""Measure the finite-sample (exact binomial) route's compute ceiling, outside pytest.

Four subcommands write one JSON line per measurement (commit, machine, Python/SciPy/NumPy
versions on every line) and are meant to run serially, one at a time::

    uv run python -m scripts.measure_binomial_ceiling latency  --out /tmp/rsf0-latency.jsonl
    uv run python -m scripts.measure_binomial_ceiling ulp      --out /tmp/rsf0-ulp.jsonl
    uv run --extra demo python -m scripts.measure_binomial_ceiling recovery --out /tmp/rsf0-recovery.jsonl
    uv run python -m scripts.measure_binomial_ceiling planning --out /tmp/rsf0-planning.jsonl

``latency``
    Runs ``binomial_rr.confidence_interval`` cold in a fresh subprocess per cell and rung (two-sided,
    ``alpha = 0.05``) and records wall and CPU time, the child's peak resident set (from
    ``wait4``'s per-child ``rusage``, as ``RUSAGE_CHILDREN`` reports only the largest child so
    far), the control-count window, tail evaluations, nuisance searches and the search's own
    disclosures. A cell that exceeds ``--timeout`` is recorded as not computed.

``ulp``
    Grades SciPy's ``binom`` primitives against the independent decimal oracle
    (``calibration/binomial_oracle.py``) at the counts the production search evaluates and on a
    spread of counts across the distribution: worst error in ULPs, the error the certificate's
    margin must absorb, the Clopper-Pearson enclosure, the support window's omitted mass, and
    whether each certified tail dominates the exact one.

``recovery``
    Builds an arm inside DuckDB from ``range()`` and runs the production two-phase producer
    (``query.builders.group_summary``), then ``armstats.binary_counts``, recording the integer
    reconstruction error and the second-moment error in units of ``variance_slack`` -- plus the
    constant-0.5 corruption cell, which must still be refused. The raw moments are recorded so
    acceptance can be re-evaluated under another tolerance without rebuilding the arm.

``planning``
    Runs ``achieved_power``, ``minimum_detectable_effect`` or ``required_sample_size`` cold in a
    fresh subprocess per cell at a conversion baseline, recording CPU time, the child's peak
    resident set, the retained count cells the replay spans (at the null rate and, for a supplied
    effect, at its alternative) and the power basis -- or the coded refusal a design beyond the
    replay bound raises. ``--conversion-inference`` selects the decision method planned
    (``finite_sample``, the default, replays the finite-sample decision at every size; ``auto``
    plans the runtime's count-routed decision, closed form where its counts are dense).
    ``--rungs`` does not apply: each cell fixes its own design.

The commands read each child's resource usage with ``os.wait4`` and the machine's load and
memory with ``os.getloadavg`` and ``os.sysconf``, which exist on POSIX only: ``main`` refuses
elsewhere. The module itself imports everywhere, so its oracle-grading helpers are usable (and
tested) on any platform.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

RUNGS = (4_000_000, 16_000_000, 64_000_000, 100_000_000, 1_000_000_000)

#: ``(rate, risk ratio)`` of each latency cell: a 5% control rate (the typical dense cell), a 1e-4
#: rate, and a 50% rate, which has the widest control-count window.
LATENCY_CELLS = {
    "dense": (0.05, 1.02),
    "rare": (1e-4, 1.02),
    "dense-wide": (0.5, 1.02),
}

#: Control rates the ULP measurement is run at: those of the latency cells.
ULP_RATES = tuple(rate for rate, _ in LATENCY_CELLS.values())

ALPHA = 0.05
DEFAULT_TIMEOUT = 3600.0

_EPS = 2.0**-52
#: Values below this are not graded. Boost's small-count branch returns a wrong (typically zero)
#: value once an intermediate product underflows, observed only for true values of at most
#: 4.3e-267, and no probability of that size can move a sum by half a float64 unit.
_GRADED_FLOOR = Decimal("1e-250")
#: A graded count is on the mass of the sum it feeds when it is at most this many standard
#: deviations from the mean or at most ``_SMALL_COUNT``; the margin consumes its relative error.
#: Any other count is graded by absolute error, which is what a negligible weight can move a sum by.
_BEARING_SIGMAS = 4.5
_SMALL_COUNT = 45
#: Expected counts (``n * rate``) of the small-count cells, where Boost's finite sum raises
#: ``1 - x`` to a power near ``n``: the worst regime of the primitives' error.
SMALL_COUNT_MEANS = (3, 10, 30, 100, 300)


# --- common -----------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=Path(__file__).parent
    ).stdout.strip()


def _machine() -> str:
    brand = ""
    if sys.platform == "darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    memory_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return (
        f"{brand or platform.processor() or platform.machine()}, {os.cpu_count()} cores, "
        f"{memory_gib:.0f} GiB, {platform.platform()}"
    )


def _metadata() -> dict[str, Any]:
    import numpy
    import scipy

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "machine": _machine(),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "numpy": numpy.__version__,
        "load_average": os.getloadavg()[0],
    }


def _emit(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps({**record, "meta": _metadata()}, default=repr)
    with path.open("a") as handle:
        handle.write(line + "\n")
    print(json.dumps(record, default=repr), flush=True)


def _rungs(text: str | None) -> tuple[int, ...]:
    return RUNGS if not text else tuple(int(float(part)) for part in text.split(","))


def _peak_rss_mib(usage: Any) -> float:
    """Peak resident set in MiB of a child's ``os.wait4`` resource usage (a ``struct_rusage``)."""
    return usage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


def _run_child(arguments: Sequence[str], timeout: float) -> dict[str, Any]:
    """Run one cell in a fresh interpreter and return its JSON line plus the process's own
    wall time, CPU time and peak resident set, or the reason it produced none."""
    command = [sys.executable, "-m", "scripts.measure_binomial_ceiling", *arguments]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    timed_out = False
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            break
        if time.perf_counter() - started > timeout:
            process.kill()
            _, status, usage = os.wait4(process.pid, 0)
            timed_out = True
            break
        time.sleep(0.25)
    wall = time.perf_counter() - started
    assert process.stdout is not None and process.stderr is not None
    out, err = process.stdout.read(), process.stderr.read()
    process.returncode = os.waitstatus_to_exitcode(status)
    process.stdout.close()
    process.stderr.close()
    record: dict[str, Any] = {
        "process_wall_s": round(wall, 3),
        "process_cpu_s": round(usage.ru_utime + usage.ru_stime, 3),
        "peak_rss_mib": round(_peak_rss_mib(usage), 1),
    }
    if timed_out:
        return {**record, "status": f"not computed: exceeded the {timeout:.0f} s per-cell timeout"}
    if process.returncode != 0 or not out.strip():
        tail = err.strip().splitlines()[-1:] or ["no output"]
        return {**record, "status": f"not computed: child exited {process.returncode}: {tail[0]}"}
    return {**record, "status": "ok", **json.loads(out.strip().splitlines()[-1])}


# --- latency ----------------------------------------------------------------------------------


def _latency_cell(n: int, rate: float, ratio: float) -> dict[str, Any]:
    """One cold ``confidence_interval``: counts at the rate, the treatment at the risk ratio."""
    from increment.estimation import binomial_rr as brr

    x_c = round(n * rate)
    x_t = round(n * rate * ratio)
    counts = {"tail_evaluations": 0, "nuisance_searches": 0, "nuisance_splits": 0}

    def counted(function: Callable[..., float]) -> Callable[..., float]:
        def wrapper(*args: Any, **kwargs: Any) -> float:
            counts["tail_evaluations"] += 1
            return function(*args, **kwargs)

        return wrapper

    original_search = brr._certified_sup

    def search(*args: Any, **kwargs: Any) -> Any:
        result = original_search(*args, **kwargs)
        counts["nuisance_searches"] += 1
        counts["nuisance_splits"] += result.iterations
        return result

    with (
        mock.patch.object(brr, "_tail_plus", counted(brr._tail_plus)),
        mock.patch.object(brr, "_tail_minus", counted(brr._tail_minus)),
        mock.patch.object(brr, "_certified_sup", search),
    ):
        wall = time.perf_counter()
        cpu = time.process_time()
        interval = brr.confidence_interval(x_c, n, x_t, n, alpha=ALPHA, alternative="two-sided")
        cpu = time.process_time() - cpu
        wall = time.perf_counter() - wall

    a, b = brr.clopper_pearson(x_c, n, brr.nuisance_beta(ALPHA))
    i_lo, i_hi, omitted = brr._support_window(n, a, b)
    return {
        "kind": "latency",
        "n": n,
        "rate": rate,
        "ratio": ratio,
        "x_c": x_c,
        "x_t": x_t,
        "call_wall_s": round(wall, 3),
        "call_cpu_s": round(cpu, 3),
        "window_length": i_hi - i_lo + 1,
        "window_omitted_mass": omitted,
        "lower": interval.lower,
        "upper": interval.upper,
        "p_value_null": interval.p_value_null,
        "endpoint_log_width": interval.endpoint_log_width,
        "resolution_reached": interval.resolution_reached,
        "nuisance_gap_max": interval.nuisance_gap_max,
        "capped_probes": interval.capped_probes,
        **counts,
    }


def _latency(args: argparse.Namespace) -> None:
    out = Path(args.out)
    for n in _rungs(args.rungs):
        for cell, (rate, ratio) in LATENCY_CELLS.items():
            if args.cells and cell not in args.cells.split(","):
                continue
            result = _run_child(
                ["_latency_cell", "--n", str(n), "--rate", repr(rate), "--ratio", repr(ratio)],
                args.timeout,
            )
            _emit(
                out,
                {"kind": "latency", "n": n, "cell": cell, "rate": rate, "ratio": ratio} | result,
            )


# --- ulp --------------------------------------------------------------------------------------


@dataclass
class _Tally:
    """Worst error of one primitive against the oracle over the values it is graded on.

    A value on the mass of its sum (``bearing``) is graded by its relative error in units of
    ``eps`` (what ``_eps_margin`` consumes) and, as ULPs of the value, for reference; any other
    only by its absolute error in ``eps`` units. Values below ``_GRADED_FLOOR`` are counted and
    skipped."""

    values: int = 0
    skipped: int = 0
    worst_ulps: float = 0.0
    worst_at: int | None = None
    worst_exact: float = 0.0
    worst_relative_eps: float = 0.0
    worst_absolute_eps: float = 0.0

    def add(self, count: int, computed: float, exact: Decimal, *, bearing: bool) -> None:
        from calibration.binomial_oracle import precise, ulp_distance

        if exact < _GRADED_FLOOR:
            self.skipped += 1
            return
        self.values += 1
        ulps = ulp_distance(computed, exact)
        if ulps > self.worst_ulps:
            self.worst_ulps, self.worst_at, self.worst_exact = ulps, count, float(exact)
        with precise():
            error = abs(Decimal(computed) - exact)
            if bearing:
                self.worst_relative_eps = max(self.worst_relative_eps, _units(error / exact))
            else:
                self.worst_absolute_eps = max(self.worst_absolute_eps, _units(error))

    def merge(self, other: _Tally) -> None:
        self.values += other.values
        self.skipped += other.skipped
        if other.worst_ulps > self.worst_ulps:
            self.worst_ulps, self.worst_at, self.worst_exact = (
                other.worst_ulps,
                other.worst_at,
                other.worst_exact,
            )
        self.worst_relative_eps = max(self.worst_relative_eps, other.worst_relative_eps)
        self.worst_absolute_eps = max(self.worst_absolute_eps, other.worst_absolute_eps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "skipped_below_floor": self.skipped,
            "worst_ulps": self.worst_ulps,
            "worst_at": self.worst_at,
            "worst_exact": self.worst_exact,
            "worst_relative_eps": self.worst_relative_eps,
            "worst_absolute_eps": self.worst_absolute_eps,
        }


def _bearing(count: int, n: int, rate: float) -> bool:
    sigma = math.sqrt(n * rate * (1.0 - rate))
    return count <= _SMALL_COUNT or abs(count - n * rate) <= _BEARING_SIGMAS * sigma


def _graded(
    counts: Sequence[int], computed: Any, exact: Sequence[Decimal], n: int, rate: float
) -> _Tally:
    tally = _Tally()
    for count, value, reference in zip(counts, computed.tolist(), exact, strict=True):
        tally.add(int(count), value, reference, bearing=_bearing(int(count), n, rate))
    return tally


def _units(value: Decimal) -> float:
    return float(value / Decimal(_EPS))


def _window_check(n: int, rate: float, kind: str, q_end: str, r: float) -> dict[str, Any]:
    """One certified tail against the oracle, at the lower or upper end of the control count's
    Clopper-Pearson interval: the primitives' ULP errors over the window, the error they leave in
    the weighted sum (against the margin that must absorb it), and whether the certified value
    dominates the exact tail plus the exact omitted mass."""
    import numpy as np

    from calibration.binomial_oracle import Binomial, precise
    from increment.estimation import binomial_rr as brr

    x_c = round(n * rate)
    x_t = round(n * rate * 1.02)
    beta = brr.nuisance_beta(ALPHA)
    a, b = brr.clopper_pearson(x_c, n, beta)
    q = a if q_end == "lower" else b
    window = brr._support_window(n, a, b)
    i_lo, i_hi, omitted = window
    k = n * x_t - n * x_c
    plus = kind == "plus"
    p = brr._p_of_plus(q, r) if plus else r * q
    threshold = (brr._plus_threshold if plus else brr._minus_threshold)(n, n, k, i_lo, i_hi)
    control_f = brr._control_pmf(n, q, i_lo, i_hi)
    tail_f = brr._treatment_tail(kind, n, n, k, i_lo, i_hi, p)
    certified = (brr._tail_plus if plus else brr._tail_minus)(q, p, n, n, k, window)

    control, treatment = Binomial(n, q), Binomial(n, p)
    control_o = control.pmf_range(i_lo, i_hi)
    thresholds = threshold.tolist()
    tail_o = treatment.sf_many(thresholds) if plus else treatment.cdf_many(thresholds)
    below = control.cdf(i_lo - 1) if i_lo > 0 else Decimal(0)
    above = control.sf(i_hi) if i_hi < n else Decimal(0)

    indices = list(range(i_lo, i_hi + 1))
    pmf_tally = _graded(indices, control_f, control_o, n, q)
    tail_tally = _graded(thresholds, tail_f, tail_o, n, p)
    with precise():
        exact_sum = sum((c * t for c, t in zip(control_o, tail_o, strict=True)), Decimal(0))
        weighted = sum(
            (
                c * abs(Decimal(tf) - t) + t * abs(Decimal(cf) - c)
                for c, t, cf, tf in zip(
                    control_o, tail_o, control_f.tolist(), tail_f.tolist(), strict=True
                )
            ),
            Decimal(0),
        )
        dot_error = abs(Decimal(float(np.dot(control_f, tail_f))) - exact_sum)
        bound = min(Decimal(1), exact_sum + below + above)
        slack = Decimal(certified) - bound
        margin = brr._eps_margin(i_hi - i_lo + 1, n, n)
        return {
            "stage": "window",
            "n": n,
            "rate": rate,
            "tail": kind,
            "q": q_end,
            "ratio": r,
            "window_length": i_hi - i_lo + 1,
            "window_omitted_mass": omitted,
            "exact_omitted_mass": float(below + above),
            "certificate_dominates": slack >= 0,
            "certificate_slack_over_eps": _units(slack),
            "margin_over_eps": margin / _EPS,
            "dot_error_over_eps": _units(dot_error),
            "primitive_weighted_error_over_eps": _units(weighted),
            "control_pmf": pmf_tally.as_dict(),
            "treatment_" + ("sf" if plus else "cdf"): tail_tally.as_dict(),
        }


def _sampled_check(n: int, rate: float) -> dict[str, Any]:
    """The three primitives on a spread of counts across the distribution, the end counts and the
    mean's neighbours, with the mean at the cell's rate."""
    import numpy as np

    from calibration.binomial_oracle import Binomial
    from increment.estimation import binomial_rr as brr

    sigma = math.sqrt(n * rate * (1.0 - rate))
    mean = n * rate
    counts = {0, 1, 10, 100, n - 1, n}
    counts |= {round(mean + z * sigma) for z in np.arange(-14.0, 14.5, 0.5)}
    ordered = sorted(c for c in counts if 0 <= c <= n)
    array = np.array(ordered, dtype=np.int64)
    oracle = Binomial(n, rate)
    return {
        "stage": "sampled",
        "n": n,
        "rate": rate,
        "counts": len(ordered),
        "pmf": _graded(
            ordered, brr._fast_binom_pmf(array, n, rate), oracle.pmf_many(ordered), n, rate
        ).as_dict(),
        "cdf": _graded(
            ordered, brr._fast_binom_cdf(array, n, rate), oracle.cdf_many(ordered), n, rate
        ).as_dict(),
        "sf": _graded(
            ordered, brr._fast_binom_sf(array, n, rate), oracle.sf_many(ordered), n, rate
        ).as_dict(),
    }


def _small_count_check(n: int) -> dict[str, Any]:
    """The primitives at counts 0..45 for rates whose expected counts are ``SMALL_COUNT_MEANS``
    (not dyadic): Boost's finite-sum branch, where the error is coherent across the counts."""
    import numpy as np

    from calibration.binomial_oracle import Binomial
    from increment.estimation import binomial_rr as brr

    ordered = list(range(_SMALL_COUNT + 1))
    array = np.array(ordered, dtype=np.int64)
    tallies = {name: _Tally() for name in ("pmf", "cdf", "sf")}
    for expected in SMALL_COUNT_MEANS:
        rate = expected * 1.0123456789 / n
        oracle = Binomial(n, rate)
        for name, function, exact in (
            ("pmf", brr._fast_binom_pmf, oracle.pmf_many(ordered)),
            ("cdf", brr._fast_binom_cdf, oracle.cdf_many(ordered)),
            ("sf", brr._fast_binom_sf, oracle.sf_many(ordered)),
        ):
            tallies[name].merge(_graded(ordered, function(array, n, rate), exact, n, rate))
    return {
        "stage": "small_count",
        "n": n,
        "expected_counts": list(SMALL_COUNT_MEANS),
        **{name: tally.as_dict() for name, tally in tallies.items()},
    }


def _resolvable(endpoint: float) -> bool:
    """Neither clamped to an end of the unit interval nor so near one that the float spacing there
    is a visible share of the endpoint's complement."""
    return 0.0 < endpoint < 1.0 and 1.0 - endpoint > 1e-9


def _enclosure_checks(n: int) -> list[dict[str, Any]]:
    """The Clopper-Pearson endpoints against exact tails: the outward-rounded lower endpoint must
    leave ``P(X >= x)`` at most ``beta / 2``, the upper ``P(X <= x)``, and a resolvable endpoint
    must not leave it under a tenth of that (the allowance is relative to the smaller side)."""
    from calibration.binomial_oracle import Binomial
    from increment.estimation import binomial_rr as brr

    out = []
    for beta in (brr.nuisance_beta(ALPHA), 1e-9):
        half = Decimal(beta) / 2
        for x in sorted(
            {0, 1, 2, 3, 4, 5, 10, 100, round(n * 1e-4), round(n * 0.05), round(n * 0.5)}
            | {n - 100, n - 10, n - 5, n - 4, n - 3, n - 2, n - 1, n}
        ):
            lower, upper = brr.clopper_pearson(x, n, beta)
            lower_mass = Binomial(n, lower).sf(x - 1) if x > 0 else Decimal(0)
            upper_mass = Binomial(n, upper).cdf(x) if x < n else Decimal(0)
            tight = (not _resolvable(lower) or lower_mass >= half / 10) and (
                not _resolvable(upper) or upper_mass >= half / 10
            )
            out.append(
                {
                    "stage": "clopper_pearson",
                    "n": n,
                    "x": x,
                    "beta": beta,
                    "lower": lower,
                    "upper": upper,
                    "lower_tail_over_half_beta": float(lower_mass / half),
                    "upper_tail_over_half_beta": float(upper_mass / half),
                    "encloses": lower_mass <= half and upper_mass <= half,
                    "tight": tight,
                }
            )
    return out


def _omitted_mass_checks(n: int) -> list[dict[str, Any]]:
    """The window's reported omitted mass against the exact control mass outside it, at the
    nuisance interval's ends and three interior rates."""
    from calibration.binomial_oracle import Binomial, precise
    from increment.estimation import binomial_rr as brr

    out = []
    for name, rate in (("dense", 0.05), ("rare", 1e-4), ("dense-wide", 0.5)):
        x_c = round(n * rate)
        a, b = brr.clopper_pearson(x_c, n, brr.nuisance_beta(ALPHA))
        i_lo, i_hi, omitted = brr._support_window(n, a, b)
        worst_below = worst_above = Decimal(0)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            binomial = Binomial(n, a + (b - a) * fraction)
            worst_below = max(worst_below, binomial.cdf(i_lo - 1) if i_lo > 0 else Decimal(0))
            worst_above = max(worst_above, binomial.sf(i_hi) if i_hi < n else Decimal(0))
        with precise():
            total = worst_below + worst_above
            out.append(
                {
                    "stage": "omitted_mass",
                    "n": n,
                    "cell": name,
                    "window": [i_lo, i_hi],
                    "reported_omitted_mass": omitted,
                    "exact_below_worst": float(worst_below),
                    "exact_above_worst": float(worst_above),
                    "exact_over_reported": float(total / Decimal(omitted)) if omitted else 0.0,
                    "bounded": total <= Decimal(omitted),
                }
            )
    return out


def _ulp(args: argparse.Namespace) -> None:
    from increment.estimation import binomial_rr as brr

    out = Path(args.out)
    for n in _rungs(args.rungs):
        started = time.process_time()
        worst = {name: _Tally() for name in ("pmf", "cdf", "sf")}
        flags = {"dominates": True, "encloses": True, "tight": True, "bounded": True}
        weighted = dot = 0.0

        def record(entry: dict[str, Any]) -> dict[str, Any]:
            _emit(out, entry)
            return entry

        small = record(_small_count_check(n))
        for name, tally in worst.items():
            tally.merge(_tally_from(small[name]))
        for rate in ULP_RATES:
            sampled = record(_sampled_check(n, rate))
            for name, tally in worst.items():
                tally.merge(_tally_from(sampled[name]))
            for kind in ("plus", "minus"):
                for q_end in ("lower", "upper"):
                    for ratio in (1.0, 1.02):
                        entry = record(_window_check(n, rate, kind, q_end, ratio))
                        tail_name = "treatment_sf" if kind == "plus" else "treatment_cdf"
                        worst["pmf"].merge(_tally_from(entry["control_pmf"]))
                        worst["sf" if kind == "plus" else "cdf"].merge(
                            _tally_from(entry[tail_name])
                        )
                        flags["dominates"] &= entry["certificate_dominates"]
                        weighted = max(weighted, entry["primitive_weighted_error_over_eps"])
                        dot = max(dot, entry["dot_error_over_eps"])
        for entry in _enclosure_checks(n):
            checked = record(entry)
            flags["encloses"] &= checked["encloses"]
            flags["tight"] &= checked["tight"]
        for entry in _omitted_mass_checks(n):
            flags["bounded"] &= record(entry)["bounded"]
        allowance = brr._ulp_allowance(n)
        record(
            {
                "stage": "summary",
                "n": n,
                "allowance": allowance,
                "worst_ulps": {name: t.worst_ulps for name, t in worst.items()},
                "worst_relative_eps": {name: t.worst_relative_eps for name, t in worst.items()},
                "worst_absolute_eps": {name: t.worst_absolute_eps for name, t in worst.items()},
                "allowance_stands": all(
                    max(t.worst_relative_eps, t.worst_absolute_eps) < allowance / 2
                    for t in worst.values()
                ),
                "max_primitive_weighted_error_over_eps": weighted,
                "max_dot_error_over_eps": dot,
                "every_certificate_dominates": flags["dominates"],
                "every_cp_enclosure_holds": flags["encloses"],
                "every_cp_endpoint_tight": flags["tight"],
                "every_omitted_mass_bounded": flags["bounded"],
                "cpu_s": round(time.process_time() - started, 1),
            }
        )


def _tally_from(record: dict[str, Any]) -> _Tally:
    return _Tally(
        values=record["values"],
        skipped=record["skipped_below_floor"],
        worst_ulps=record["worst_ulps"],
        worst_at=record["worst_at"],
        worst_exact=record["worst_exact"],
        worst_relative_eps=record["worst_relative_eps"],
        worst_absolute_eps=record["worst_absolute_eps"],
    )


# --- recovery ---------------------------------------------------------------------------------


def _recovery_cells(n: int) -> dict[str, int | None]:
    """Successes of each recovery cell (``None``: the constant-0.5 corruption arm)."""
    return {
        "half": n // 2,
        "rate-0.002": round(0.002 * n),
        "rate-1e-4": round(1e-4 * n),
        "single": 1,
        "constant-0.5": None,
    }


def _permutation_multiplier(n: int) -> int:
    multiplier = 2654435761
    while math.gcd(multiplier, n) != 1:
        multiplier += 2
    return multiplier


def _recovery_cell(
    n: int, successes: int | None, memory: str, temp_limit: str, temp: str
) -> dict[str, Any]:
    """One arm built in DuckDB, run through the production producer and ``binary_counts``."""
    import duckdb
    import ibis

    from increment.errors import CodedError
    from increment.estimation.armstats import (
        _BERNOULLI_CONSISTENCY_SLACK,
        ArmStats,
        binary_counts,
        variance_slack,
    )
    from increment.query.builders import group_summary

    if successes is None:
        outcome = "CAST(0.5 AS DOUBLE)"
    else:
        # `(i * m + c) % n` is a bijection on 0..n-1, so exactly `successes` rows are 1, scattered.
        multiplier = _permutation_multiplier(n)
        outcome = (
            f"CAST(CASE WHEN ((range::HUGEINT * {multiplier} + 12345) % {n}) < {successes} "
            "THEN 1 ELSE 0 END AS DOUBLE)"
        )
    connection = ibis.duckdb.connect()
    temp_path = Path(temp)
    connection.raw_sql(f"PRAGMA temp_directory='{temp_path}'")
    connection.raw_sql(f"PRAGMA memory_limit='{memory}'")
    connection.raw_sql(f"PRAGMA max_temp_directory_size='{temp_limit}'")
    arm_table = connection.sql(
        f"SELECT 'u' AS unit_id, 'e' AS experiment_id, 'control' AS group_id, 'conv' AS metric, "
        f"{outcome} AS y, CAST(NULL AS DOUBLE) AS x, CAST(NULL AS DOUBLE) AS y_den FROM range({n})"
    )
    wall = time.perf_counter()
    cpu = time.process_time()
    row = group_summary(arm_table).execute().iloc[0]
    cpu = time.process_time() - cpu
    wall = time.perf_counter() - wall
    connection.disconnect()
    arm = ArmStats(
        study_id="e",
        metric="conv",
        group_id="control",
        n=int(row["n"]),
        ref_y=float(row["ref_y"]),
        cy1=float(row["cy1"]),
        cy2=float(row["cy2"]),
    )
    sum_y = arm.n * arm.ref_y + arm.cy1
    truth = n * 0.5 if successes is None else successes
    expected_cy2 = truth * (n - truth) / n
    slack = variance_slack(max(abs(arm.cy2), abs(expected_cy2), 1.0), n)
    try:
        recovered: Any = binary_counts(arm, "conversion")
    except CodedError as refusal:
        recovered = refusal.code
    return {
        "kind": "recovery",
        "n": n,
        "successes": successes,
        "cell_wall_s": round(wall, 2),
        "cell_cpu_s": round(cpu, 2),
        "ref_y": repr(arm.ref_y),
        "cy1": repr(arm.cy1),
        "cy2": repr(arm.cy2),
        "sum_y": repr(sum_y),
        "sum_y_error": sum_y - truth,
        "sum_y_error_ulps": abs(sum_y - truth) / math.ulp(max(abs(sum_y), 1.0)),
        "expected_cy2": expected_cy2,
        "cy2_error": arm.cy2 - expected_cy2,
        "cy2_error_over_variance_slack": abs(arm.cy2 - expected_cy2) / slack,
        "consistency_slack": _BERNOULLI_CONSISTENCY_SLACK,
        "headroom": _BERNOULLI_CONSISTENCY_SLACK / max(abs(arm.cy2 - expected_cy2) / slack, 1e-300),
        "binary_counts": recovered,
        "duckdb": duckdb.__version__,
    }


def _recovery(args: argparse.Namespace) -> None:
    out = Path(args.out)
    for n in _rungs(args.rungs):
        for name, successes in _recovery_cells(n).items():
            if args.cells and name not in args.cells.split(","):
                continue
            # The parent owns the spill directory, so a killed or timed-out child leaves none behind.
            spill = tempfile.mkdtemp(prefix="rsf0-duckdb-")
            arguments = [
                "_recovery_cell",
                "--n",
                str(n),
                "--duckdb-memory",
                args.duckdb_memory,
                "--duckdb-temp-limit",
                args.duckdb_temp_limit,
                "--duckdb-temp",
                spill,
            ]
            if successes is not None:
                arguments += ["--successes", str(successes)]
            try:
                result = _run_child(arguments, args.timeout)
            finally:
                shutil.rmtree(spill, ignore_errors=True)
            _emit(out, {"kind": "recovery", "n": n, "cell": name, "successes": successes} | result)


# --- planning ---------------------------------------------------------------------------------

#: ``(cell, planner, control rate, units per arm, relative lift)``. Planning costs the cells its
#: geometry stores, not the arm size: a supplied effect reaches `PLANNING_CELL_CEILING` near a
#: million units per arm at 5% and an effect search sooner; the rare and sparse cells expect about
#: a hundred events per arm; the dense cells from 5e6 are beyond the bound a replay is refused at.
PLANNING_CELLS = (
    ("dense-1e5", "power", 0.05, 100_000, 0.05),
    ("dense-2.5e5", "power", 0.05, 250_000, 0.03),
    ("dense-5e5", "power", 0.05, 500_000, 0.02),
    ("dense-5e5-mde", "mde", 0.05, 500_000, None),
    ("dense-1e6", "power", 0.05, 1_000_000, 0.015),
    ("dense-1e6-mde", "mde", 0.05, 1_000_000, None),
    ("dense-size", "size", 0.05, None, 0.0175),
    ("wide-1e5-mde", "mde", 0.5, 100_000, None),
    ("wide-1.9e5", "power", 0.5, 190_000, 0.03),
    ("wide-1.9e5-mde", "mde", 0.5, 190_000, None),
    ("rare-1e6", "power", 1e-4, 1_000_000, 0.5),
    ("rare-1e8", "power", 1e-6, 100_000_000, 0.5),
    ("rare-1e9", "power", 1e-7, 1_000_000_000, 0.5),
    ("dense-5e6", "power", 0.05, 5_000_000, 0.01),
    ("dense-5e6-mde", "mde", 0.05, 5_000_000, None),
    ("dense-5e7", "power", 0.05, 50_000_000, 0.003),
    ("dense-5e7-mde", "mde", 0.05, 50_000_000, None),
    # About a hundred events per arm at 1e5, 1e6, 5e6 and 5e7 units: sparse for any dense-count rule.
    ("sparse-1e5", "power", 1e-3, 100_000, 0.5),
    ("sparse-1e5-mde", "mde", 1e-3, 100_000, None),
    ("sparse-1e6-mde", "mde", 1e-4, 1_000_000, None),
    ("sparse-5e6", "power", 2e-5, 5_000_000, 0.5),
    ("sparse-5e6-mde", "mde", 2e-5, 5_000_000, None),
    ("sparse-5e7", "power", 2e-6, 50_000_000, 0.5),
    ("sparse-5e7-mde", "mde", 2e-6, 50_000_000, None),
)


def _planning_cell(
    planner: str, rate: float, n: int | None, lift: float | None, conversion_inference: str
) -> dict[str, Any]:
    """One cold planning call: ``achieved_power``, ``minimum_detectable_effect`` or
    ``required_sample_size`` at a conversion baseline under ``conversion_inference``, or the
    coded refusal it raised."""
    from increment.errors import CodedError
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.power import (
        Baseline,
        achieved_power,
        minimum_detectable_effect,
        required_sample_size,
    )
    from increment.power._binomial import refused, window_cells
    from increment.power.core import _binomial_key

    baseline = Baseline.from_proportion(rate)
    procedure = ArmPlanningProcedure.standard(
        "conversion", conversion_inference=conversion_inference
    )
    wall = time.perf_counter()
    cpu = time.process_time()
    outcome: dict[str, Any]
    units = n
    try:
        if planner == "power":
            assert n is not None and lift is not None
            result = achieved_power(n, lift, baseline, procedure)
        elif planner == "mde":
            assert n is not None
            result = minimum_detectable_effect(n, baseline, procedure)
        else:
            assert lift is not None
            result = required_sample_size(lift, baseline, procedure)
            units = result.n_per_arm
        outcome = {
            "n_per_arm": result.n_per_arm,
            "power": result.power,
            "mde_relative": result.mde_relative,
            "mde_unavailable_reason": result.mde_unavailable_reason,
            "power_basis": result.power_basis,
        }
    except CodedError as refusal:
        outcome = {"refused": refusal.code, "refusal_context": dict(refusal.context)}
    cpu = time.process_time() - cpu
    wall = time.perf_counter() - wall
    key = _binomial_key(procedure, units, units) if units else None
    cells = (0 if refused(key) else window_cells(key, rate)) if key else None
    # The rectangle of the supplied effect's alternative, which the replay classifies as well.
    alternative = (
        window_cells(key, rate, min(1.0, rate * (1.0 + lift))) if key and lift is not None else None
    )
    return {
        "kind": "planning",
        "conversion_inference": conversion_inference,
        "planner": planner,
        "rate": rate,
        "n": n,
        "lift": lift,
        "replay_cells": cells,
        "alternative_cells": alternative,
        "call_wall_s": round(wall, 3),
        "call_cpu_s": round(cpu, 3),
        **outcome,
    }


def _planning(args: argparse.Namespace) -> None:
    out = Path(args.out)
    for cell, planner, rate, n, lift in PLANNING_CELLS:
        if args.cells and cell not in args.cells.split(","):
            continue
        arguments = [
            "_planning_cell",
            "--planner",
            planner,
            "--rate",
            repr(rate),
            "--conversion-inference",
            args.conversion_inference,
        ]
        if n is not None:
            arguments += ["--n", str(n)]
        if lift is not None:
            arguments += ["--lift", repr(lift)]
        _emit(out, {"kind": "planning", "cell": cell} | _run_child(arguments, args.timeout))


# --- entry ------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("latency", "cold confidence_interval latency and memory"),
        ("ulp", "primitive ULP error, enclosure and omitted mass against the oracle"),
        ("recovery", "DuckDB producer error against binary_counts"),
        ("planning", "cold planning call cost and memory (CPU seconds, peak RSS)"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--out", required=True, help="JSONL file to append to")
        if name != "planning":
            command.add_argument("--rungs", help="comma-separated arm sizes (default: 4e6..1e9)")
        if name != "ulp":
            command.add_argument("--cells", help="comma-separated cell names to run")
            command.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
        if name == "planning":
            command.add_argument(
                "--conversion-inference", choices=("finite_sample", "auto"), default="finite_sample"
            )
        if name == "recovery":
            _duckdb_options(command)
    planning = sub.add_parser("_planning_cell")
    planning.add_argument("--planner", required=True, choices=("power", "mde", "size"))
    planning.add_argument("--rate", type=float, required=True)
    planning.add_argument("--n", type=int)
    planning.add_argument("--lift", type=float)
    planning.add_argument(
        "--conversion-inference", choices=("finite_sample", "auto"), required=True
    )
    latency = sub.add_parser("_latency_cell")
    latency.add_argument("--n", type=int, required=True)
    latency.add_argument("--rate", type=float, required=True)
    latency.add_argument("--ratio", type=float, required=True)
    recovery = sub.add_parser("_recovery_cell")
    recovery.add_argument("--n", type=int, required=True)
    recovery.add_argument("--successes", type=int)
    recovery.add_argument(
        "--duckdb-temp", required=True, help="DuckDB spill directory (parent-owned)"
    )
    _duckdb_options(recovery)
    return parser


def _duckdb_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--duckdb-memory", default="16GB", help="DuckDB memory_limit")
    command.add_argument(
        "--duckdb-temp-limit",
        default="40GB",
        help="DuckDB max_temp_directory_size: a window over a billion rows spills to disk",
    )


def main(argv: Sequence[str] | None = None) -> None:
    if os.name != "posix":
        raise SystemExit(
            "measure_binomial_ceiling reads per-child resource usage with os.wait4 and needs a "
            f"POSIX platform, not {os.name!r}"
        )
    args = _parser().parse_args(argv)
    if args.command == "latency":
        _latency(args)
    elif args.command == "ulp":
        _ulp(args)
    elif args.command == "recovery":
        _recovery(args)
    elif args.command == "planning":
        _planning(args)
    elif args.command == "_planning_cell":
        cell = _planning_cell(args.planner, args.rate, args.n, args.lift, args.conversion_inference)
        print(json.dumps(cell, default=repr))
    elif args.command == "_latency_cell":
        print(json.dumps(_latency_cell(args.n, args.rate, args.ratio), default=repr))
    elif args.command == "_recovery_cell":
        cell = _recovery_cell(
            args.n, args.successes, args.duckdb_memory, args.duckdb_temp_limit, args.duckdb_temp
        )
        print(json.dumps(cell, default=repr))


if __name__ == "__main__":
    main()
