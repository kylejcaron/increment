"""Integrate production binomial decisions over the full rare-event grid.

For each control count, a fixed feasible nuisance witness certifies a
monotone region of accepted treatment counts. Remaining counts are evaluated
through the actual adaptive production p-values; no monotonicity is assumed
for those numerical certificates. The witness removes the production tail's
omitted-mass allowance and twice its floating-point allowance before use.

Binomial concentration windows bound omitted sampling mass. Control counts
whose nuisance interval misses the true rate are charged as unresolved.
Reported error brackets include these unresolved probabilities; conditional
point-backed coverage uses the full probability of a positive control count.
Finite-point bias integrates the conditional expectation n_c*p_t/x_c - 1.

Test-at-truth acceptance is not identified with returned-interval membership.
For interval coverage, only witness acceptance is certified: exact tails are
monotone in the candidate ratio, and every finite returned endpoint is on
the rejected side of a pointwise upper certificate. A witness-accepted truth
therefore lies inside the returned interval even when certificates wobble.
All other counts contribute to an explicit noncoverage upper bound.

The manifest's ratios <= 2, arms <= 4 million, and alpha=0.05 satisfy the
production search caps and nuisance floor. Upper searches reach the empty
nuisance domain at 2/a; zero-control intervals are structurally unbounded above.

This is deterministic probability integration, not Monte Carlo. Numerical
guarantees retain the production module's stated SciPy accuracy assumptions.
Small-count regressions compare against exhaustive production evaluations.

The full-grid gate also requires actual production point/set reportability,
contracting quartile-probe intervals and increasing non-null integrated power.
Subset and interval-only runs retain diagnostics but cannot pass the complete gate.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import binom as _binom

from increment.estimation import binomial_rr
from tests.estimation.test_rare_event_calibration import MANIFEST, ManifestCell
from tests.mc import scientific_delta

ALPHA = 0.05


# Exact concentration window for either arm under the fixed true
# ``Binomial(n, p)`` DGP. The same quantile formula bounds the omitted
# sampling mass; it does not cover a nuisance interval.

_XC_TAIL_BUDGET = 1e-12
_XT_TAIL_BUDGET = 1e-12
_WINDOW_PAD = 8
_WINDOW_WIDEN_STEP = 16


def _inflate_tail(p: float, term_count: int = 1) -> float:
    """Round up using production's absolute SciPy and summation allowance."""
    return min(1.0, math.nextafter(p + binomial_rr._eps_margin(term_count), math.inf))


def _interval_mass(lo: int, hi: int, n: int, p: float) -> float:
    """Central estimate; callers separately charge the evaluation allowance."""
    if lo > hi:
        return 0.0
    if lo > n * p:
        return float(_binom.sf(lo - 1, n, p) - _binom.sf(hi, n, p))
    return float(_binom.cdf(hi, n, p) - _binom.cdf(lo - 1, n, p))


def exact_outer_window(n: int, p: float, budget: float) -> tuple[int, int, float]:
    """A certified `[lo, hi]` support window for `Binomial(n, p)` (a TRUE,
    fixed DGP parameter -- not a nuisance range) and a rigorous upper bound
    on its omitted PMF mass, via exact quantiles rather than a Chernoff
    bound (tighter, since there is no nuisance interval to cover
    simultaneously here, unlike `binomial_rr._support_window`)."""
    if p <= 0.0:
        return 0, 0, 0.0
    if p >= 1.0:
        return n, n, 0.0
    half = budget / 2.0
    lo = int(max(0, math.floor(float(_binom.ppf(half, n, p))) - _WINDOW_PAD))
    hi = int(min(n, math.ceil(float(_binom.isf(half, n, p))) + _WINDOW_PAD))

    def omitted_of(lo: int, hi: int) -> float:
        if lo == 0 and hi == n:
            return 0.0
        # Two special-function errors plus addition, under production's assumptions.
        raw = float(_binom.cdf(lo - 1, n, p)) + float(_binom.sf(hi, n, p))
        return _inflate_tail(raw, 2)

    omitted = omitted_of(lo, hi)
    widen = _WINDOW_WIDEN_STEP
    while omitted > budget and (lo > 0 or hi < n):
        lo = max(0, lo - widen)
        hi = min(n, hi + widen)
        omitted = omitted_of(lo, hi)
        widen *= 2
    return lo, hi, omitted


# Certified witness lower bound: reuse binomial_rr's outward-rounded
# primitives at one fixed evaluation point so their numerical guarantees
# transfer without reimplementing the production calculations.

_tail_plus = binomial_rr._tail_plus
_tail_minus = binomial_rr._tail_minus
_support_window = binomial_rr._support_window
_clopper_pearson = binomial_rr.clopper_pearson
_p_of_plus = binomial_rr._p_of_plus
_eps_margin = binomial_rr._eps_margin
nuisance_beta = binomial_rr.nuisance_beta


def _witness_lower_plus(
    true_p_c: float, n_c: int, n_t: int, r: float, x_c_obs: int, a: float, b: float, window
) -> Callable[[int], float] | None:
    """Rigorous, exactly x_t-non-increasing LOWER bound on the TRUE
    (uncertified) `F_+(true_p_c, ...)`, valid iff `true_p_c in [a, b]` (a
    "CP hit"): `sup_{q in [a,b]} F(q,...) >= F(true_p_c,...)` since
    true_p_c is one point in that domain, and production's certified_sup
    is itself always `>= true_sup`, so `beta + this lower bound <=
    production's real p_plus`.

    `_tail_plus`'s raw return already INFLATES the true value by its own
    support-window's omitted mass PLUS a floating-point safety margin
    (`_eps_margin`) -- both must be subtracted back out to recover a
    genuine lower bound on the exact value, not `_eps_margin` alone (an
    earlier version of this function under-subtracted here, understating
    how much the returned value could exceed the true one)."""
    if not (a <= true_p_c <= b):
        return None
    omitted = window[2]
    margin = _eps_margin(window[1] - window[0] + 1)
    correction = omitted + 2.0 * margin  # 2x margin: extra headroom costs nothing but safety

    def lower(xt: int) -> float:
        k = n_c * xt - n_t * x_c_obs
        val = _tail_plus(true_p_c, _p_of_plus(true_p_c, r), n_c, n_t, k, window)
        return max(0.0, val - correction)

    return lower


def _witness_lower_minus(
    true_p_c: float, n_c: int, n_t: int, r: float, x_c_obs: int, a: float, upper_q: float, window
) -> Callable[[int], float] | None:
    if not (a <= true_p_c <= upper_q):
        return None
    omitted = window[2]
    margin = _eps_margin(window[1] - window[0] + 1)
    correction = omitted + 2.0 * margin

    def lower(xt: int) -> float:
        k = n_c * xt - n_t * x_c_obs
        val = _tail_minus(true_p_c, r * true_p_c, n_c, n_t, k, window)
        return max(0.0, val - correction)

    return lower


def _bisect_nonincreasing(g: Callable[[int], float], target: float, n_t: int) -> int:
    """Largest `xt` in `[0, n_t]` with `g(xt) >= target`, or `-1` if even
    `g(0) < target`, for `g` PROVABLY exactly non-increasing (never
    production's own adaptive p_plus/p_minus -- see the module docstring)."""
    if g(0) < target:
        return -1
    if g(n_t) >= target:
        return n_t
    lo, hi = 0, n_t
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if g(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


def _bisect_nondecreasing(g: Callable[[int], float], target: float, n_t: int) -> int:
    """Mirror of `_bisect_nonincreasing`: smallest `xt` with `g(xt) >=
    target`, or `n_t + 1` if even `g(n_t) < target`."""
    if g(n_t) < target:
        return n_t + 1
    if g(0) >= target:
        return 0
    lo, hi = 0, n_t
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if g(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True, slots=True)
class TailResolution:
    """One tail's ('plus' or 'minus') witness-certified accept boundary,
    the small directly-resolved correction band (restricted to the TRUE
    x_t concentration window), and the explicit unresolved mass outside
    both. `T_accept` follows `binomial_rr`'s own sentinel convention
    (an out-of-range value, never `None`, marks a degenerate empty
    certified region)."""

    T_accept: int
    cp_hit: bool
    band_points: tuple[int, ...]
    resolve_at: Callable[[int], bool]
    unresolved_mass: float


def resolve_plus_tail(
    x_c_obs: int,
    n_c: int,
    n_t: int,
    beta: float,
    r: float,
    true_p_c: float,
    p_t: float,
    target: float,
    *,
    xt_tail_budget: float = _XT_TAIL_BUDGET,
) -> TailResolution:
    a, b = _clopper_pearson(x_c_obs, n_c, beta)
    window = _support_window(n_c, a, b)
    cp_hit = a <= true_p_c <= b
    lower = _witness_lower_plus(true_p_c, n_c, n_t, r, x_c_obs, a, b, window) if cp_hit else None
    if lower is not None:
        g_lower = lambda xt: min(1.0, beta + lower(xt))  # noqa: E731
        T_accept = _bisect_nonincreasing(g_lower, target, n_t)
    else:
        T_accept = -1

    xt_lo, xt_hi, _xt_omitted = exact_outer_window(n_t, p_t, xt_tail_budget)
    band_lo = max(0, T_accept + 1, xt_lo)
    band_hi = min(n_t, xt_hi)
    band_points = tuple(range(band_lo, band_hi + 1)) if band_lo <= band_hi else ()

    def resolve_at(xt: int) -> bool:
        return binomial_rr.p_plus(r, x_c_obs, n_c, xt, n_t, beta) >= target

    outside_unresolved = (xt_lo > 0 and T_accept < xt_lo - 1) or (xt_hi < n_t and T_accept < n_t)
    unresolved_mass = _xt_omitted if outside_unresolved else 0.0

    return TailResolution(T_accept, cp_hit, band_points, resolve_at, unresolved_mass)


def resolve_minus_tail(
    x_c_obs: int,
    n_c: int,
    n_t: int,
    beta: float,
    r: float,
    true_p_c: float,
    p_t: float,
    target: float,
    *,
    xt_tail_budget: float = _XT_TAIL_BUDGET,
) -> TailResolution:
    a, b_cp = _clopper_pearson(x_c_obs, n_c, beta)
    upper_q = b_cp if r <= 0.0 else min(b_cp, 1.0 / r)
    feasible = upper_q >= a
    xt_lo, xt_hi, _xt_omitted = exact_outer_window(n_t, p_t, xt_tail_budget)
    if not feasible:
        # `binomial_rr.p_minus` returns exactly `beta` for every x_t here
        # (empty restricted-nuisance domain, see its own early-return branch).
        if beta >= target:
            return TailResolution(0, True, (), lambda xt: True, 0.0)
        return TailResolution(n_t + 1, True, (), lambda xt: False, 0.0)

    window = _support_window(n_c, a, upper_q)
    cp_hit = a <= true_p_c <= upper_q
    lower = (
        _witness_lower_minus(true_p_c, n_c, n_t, r, x_c_obs, a, upper_q, window) if cp_hit else None
    )
    if lower is not None:
        g_lower = lambda xt: min(1.0, beta + lower(xt))  # noqa: E731
        T_accept = _bisect_nondecreasing(g_lower, target, n_t)
    else:
        T_accept = n_t + 1

    band_lo = max(0, xt_lo)
    band_hi = min(n_t, T_accept - 1, xt_hi)
    band_points = tuple(range(band_lo, band_hi + 1)) if band_lo <= band_hi else ()

    def resolve_at(xt: int) -> bool:
        return binomial_rr.p_minus(r, x_c_obs, n_c, xt, n_t, beta) >= target

    outside_unresolved = (xt_lo > 0 and T_accept > 0) or (xt_hi < n_t and T_accept > xt_hi + 1)
    unresolved_mass = _xt_omitted if outside_unresolved else 0.0

    return TailResolution(T_accept, cp_hit, band_points, resolve_at, unresolved_mass)


def single_tail_accept_mass(
    direction: Literal["plus", "minus"], res: TailResolution, n_t: int, p_t: float
) -> tuple[float, float]:
    """`(accept_mass, unresolved_mass)`: exact `P(production accepts r0 on
    this tail)` under `Binomial(n_t, p_t)` restricted to the RESOLVED part
    of the sample space (certified prefix/suffix via `binom.cdf`/`sf` plus
    a direct per-point sum over the band), and the separately-tracked
    unresolved mass outside the true x_t concentration window."""
    if direction == "plus":
        core = float(_binom.cdf(res.T_accept, n_t, p_t)) if res.T_accept >= 0 else 0.0
    else:
        core = float(_binom.sf(res.T_accept - 1, n_t, p_t)) if res.T_accept <= n_t else 0.0
    band_mass = sum(float(_binom.pmf(xt, n_t, p_t)) for xt in res.band_points if res.resolve_at(xt))
    return core + band_mass, res.unresolved_mass + _eps_margin(len(res.band_points) + 2)


def two_sided_covered_mass(
    res_plus: TailResolution, res_minus: TailResolution, n_t: int, p_t: float
) -> tuple[float, float]:
    """Probability both tests accept over the resolved sample space.

    This is not membership in the numerically inverted interval. The core
    and directly evaluated bands are combined without double-counting.
    """
    Tap, Tam = res_plus.T_accept, res_minus.T_accept
    lo_core, hi_core = Tam, Tap
    core_mass = 0.0
    if lo_core <= hi_core:
        core_mass = _interval_mass(lo_core, hi_core, n_t, p_t)
    band_pts = sorted(set(res_plus.band_points) | set(res_minus.band_points))
    for xt in band_pts:
        if lo_core <= xt <= hi_core:
            core_mass -= float(_binom.pmf(xt, n_t, p_t))
    resolved_mass = 0.0
    for xt in band_pts:
        fp = res_plus.resolve_at(xt) if xt in res_plus.band_points else xt <= Tap
        fm = res_minus.resolve_at(xt) if xt in res_minus.band_points else xt >= Tam
        if fp and fm:
            resolved_mass += float(_binom.pmf(xt, n_t, p_t))
    # Charge core subtraction, removed PMFs and accepted PMFs separately.
    unresolved_mass = (
        res_plus.unresolved_mass
        + res_minus.unresolved_mass
        + 3 * _eps_margin(2 * len(band_pts) + 2)
    )
    return max(0.0, core_mass) + resolved_mass, unresolved_mass


def _interval_miss_bound(
    res_plus: TailResolution,
    res_minus: TailResolution,
    n_t: int,
    p_t: float,
    *,
    zero_control: bool,
) -> float:
    lower = 0 if zero_control else res_minus.T_accept
    upper = res_plus.T_accept
    if lower > upper:
        return 1.0
    return min(
        1.0,
        _inflate_tail(float(_binom.cdf(lower - 1, n_t, p_t)))
        + _inflate_tail(float(_binom.sf(upper, n_t, p_t))),
    )


# --- Point-lift conditional bias: exact closed form -------------------------


def conditional_bias_exact(
    n_c: int, p_c: float, p_t: float, true_lift: float
) -> tuple[float | None, float]:
    """`E[point_lift | x_c > 0] - true_lift`, exact: `point_lift(x_c, n_c,
    x_t, n_t) = n_c*x_t/(n_t*x_c) - 1` is exactly linear in `x_t` for `x_c
    > 0`, so `E[point_lift | x_c] = n_c*p_t/x_c - 1` (`E[X_t] = n_t*p_t`
    exactly) -- no `x_t` windowing/production calls needed for this metric.
    Returns `(bias, P(x_c > 0))`; `bias is None` iff `P(x_c > 0) == 0`
    (only possible at `p_c == 0`, already refused upstream)."""
    p_xc_pos = float(_binom.sf(0, n_c, p_c))
    if p_xc_pos <= 0.0:
        return None, 0.0
    xs = np.arange(1, n_c + 1)
    pmf = _binom.pmf(xs, n_c, p_c)
    terms = (n_c * p_t) / xs - 1.0
    exp_point = float(np.dot(pmf, terms)) / p_xc_pos
    return exp_point - true_lift, p_xc_pos


# --- Per-cell calibration ----------------------------------------------------

METRIC_NAMES = (
    "two_sided_reject_null",  # type_I (true_rr==1) / power (true_rr!=1)
    "greater_reject_null",
    "less_reject_null",
)


def _test_error_metrics(
    acc: dict[str, float],
    true_rr: float,
    uncond_true_rr_noncov: float,
    cond_true_rr_noncov_numer: float,
    p_xc_pos: float,
    point_mass_lower: float,
    total_uncertain: float,
) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for name, label in (
        ("two_sided_reject_null", "power" if true_rr != 1.0 else "type_I"),
        ("greater_reject_null", "power" if true_rr > 1.0 else "type_I"),
        ("less_reject_null", "power" if true_rr < 1.0 else "type_I"),
    ):
        value = max(0.0, acc[name])
        metrics[f"{label}_{name.split('_')[0]}"] = {
            "value": value,
            "bound": [
                max(0.0, value - total_uncertain),
                min(1.0, value + total_uncertain),
            ],
        }
    metrics["true_rr_test_noncoverage"] = {
        "unconditional": {
            "value": max(0.0, uncond_true_rr_noncov),
            "bound": [
                max(0.0, uncond_true_rr_noncov - total_uncertain),
                min(1.0, uncond_true_rr_noncov + total_uncertain),
            ],
        },
        "conditional_on_point_available": (
            None
            if point_mass_lower == 0.0
            else {
                "value": max(0.0, cond_true_rr_noncov_numer / p_xc_pos),
                "bound": [
                    max(
                        0.0,
                        (cond_true_rr_noncov_numer - total_uncertain)
                        / min(1.0, _inflate_tail(p_xc_pos)),
                    ),
                    min(1.0, (cond_true_rr_noncov_numer + total_uncertain) / point_mass_lower),
                ],
                "conditioning_mass": p_xc_pos,
            }
        ),
    }
    return metrics


# Fixed before integration: central quartile pairs and zero/one-control probes.
# These are geometry diagnostics, not draws or probability-weighted estimates.
_PROBE_QUANTILES = (0.25, 0.5, 0.75)
_CONTRACTION_LEVELS = (0.5, 30, 100)
_POWER_LEVELS = (0.5, 100)


def production_evidence(cell: ManifestCell, alpha: float) -> dict:
    """Evaluate actual returned sets and points independently of test acceptance.

    Use the compact RR coordinate R/(1+R): its full-space diameter is 1,
    including unbounded intervals. Equal-weight quartile-pair diameters permit
    matched comparisons without pretending these probes are a sampling law.
    Zero/one-control probes additionally check the availability transition.
    """
    controls = [int(_binom.ppf(q, cell.n_c, cell.p_c)) for q in _PROBE_QUANTILES]
    treatments = [int(_binom.ppf(q, cell.n_t, cell.p_t)) for q in _PROBE_QUANTILES]
    central = [(xc, xt) for xc in controls for xt in treatments]
    counts = sorted(set(central) | {(0, treatments[1]), (1, treatments[1])})
    rows = {}
    for xc, xt in counts:
        interval = binomial_rr.confidence_interval(
            xc, cell.n_c, xt, cell.n_t, alpha=alpha, alternative="two-sided"
        )
        point = binomial_rr.point_lift(xc, cell.n_c, xt, cell.n_t)
        lo = interval.lower if interval is not None else None
        hi = interval.upper if interval is not None else None
        set_available = (
            lo is not None
            and math.isfinite(lo)
            and lo >= 0.0
            and (hi is None or (math.isfinite(hi) and hi >= lo))
        )
        point_available = point is not None and math.isfinite(point)
        matches_reference = (
            set_available and point_available == (xc > 0) and (hi is None) == (xc == 0)
        )
        diameter = None
        if set_available:
            diameter = (1.0 if hi is None else hi / (1.0 + hi)) - lo / (1.0 + lo)
        rows[xc, xt] = {
            "x_c": xc,
            "x_t": xt,
            "point": point,
            "lower_rr": lo,
            "upper_rr": hi,
            "set_available": set_available,
            "point_available": point_available,
            "reference_set_available": True,
            "reference_point_available": xc > 0,
            "reference_upper_unbounded": xc == 0,
            "matches_reference": matches_reference,
            "compact_diameter": diameter,
        }
    diameters = [rows[count]["compact_diameter"] for count in central]
    return {
        "method": "fixed_quartile_geometry_probes",
        "quantiles": _PROBE_QUANTILES,
        "rows": list(rows.values()),
        "reportability_passed": all(row["matches_reference"] for row in rows.values()),
        "mean_compact_diameter": (
            None
            if any(d is None for d in diameters)
            else math.fsum(d for d in diameters if d is not None) / len(diameters)
        ),
    }


def _trajectory_acceptance(rows: dict[float, dict], alpha: float) -> dict:
    """Compare matched event levels; no cellwise sparse-count power floor."""
    widths = [
        rows[level]["production_evidence"]["mean_compact_diameter"] for level in _CONTRACTION_LEVELS
    ]
    contraction = all(w is not None and math.isfinite(w) for w in widths) and (
        widths[0] > widths[1] + _eps_margin(1) and widths[1] > widths[2] + _eps_margin(1)
    )
    rr = rows[100]["cell"]["risk_ratio"]
    power = {}
    if rr != 1.0:
        for name in ("power_two", "power_greater" if rr > 1.0 else "power_less"):
            sparse, dense = (rows[level]["metrics"].get(name) for level in _POWER_LEVELS)
            available = sparse is not None and dense is not None
            power[name] = {
                "sparse_bound": sparse["bound"] if sparse is not None else None,
                "dense_bound": dense["bound"] if dense is not None else None,
                "passed": available
                and dense["bound"][0] > max(alpha + scientific_delta(alpha), sparse["bound"][1]),
            }
    return {
        "p_c": rows[100]["cell"]["p_c"],
        "ratio": rows[100]["cell"]["ratio"],
        "risk_ratio": rr,
        "mean_compact_diameters": widths,
        "contraction_passed": contraction,
        "power": power,
        "passed": contraction and all(record["passed"] for record in power.values()),
    }


def grid_acceptance(results: list[dict], *, alpha: float = ALPHA) -> dict:
    """Full-manifest coverage, availability, contraction and power gate.

    At fixed baseline rate, allocation and RR, 0.5 -> 30 -> 100 expected
    control events must contract production intervals. For each non-null
    trajectory, dense two-sided and correctly directed power must exceed both
    sparse power and the allowed null rejection rate, using disjoint bounds.
    """

    def key(cell: dict) -> tuple:
        return cell["p_c"], cell["expected_events"], tuple(cell["ratio"]), cell["risk_ratio"]

    expected = {key(asdict(cell)) for cell in MANIFEST}
    actual = [key(row["cell"]) for row in results]
    complete = len(actual) == len(expected) and set(actual) == expected
    groups: dict[tuple, dict[float, dict]] = {}
    for row in results:
        c = row["cell"]
        groups.setdefault((c["p_c"], tuple(c["ratio"]), c["risk_ratio"]), {})[
            c["expected_events"]
        ] = row
    comparisons = [
        _trajectory_acceptance(rows, alpha)
        for rows in groups.values()
        if all(level in rows for level in _CONTRACTION_LEVELS)
    ]
    cells_passed = bool(results) and all(row["acceptance"]["passed"] for row in results)
    return {
        "complete_manifest": complete,
        "cells_passed": cells_passed,
        "contraction_levels": _CONTRACTION_LEVELS,
        "power_levels": _POWER_LEVELS,
        "power_null_benchmark": alpha + scientific_delta(alpha),
        "comparisons": comparisons,
        "passed": complete and cells_passed and all(row["passed"] for row in comparisons),
    }


def calibrate_cell(
    cell: ManifestCell, *, alpha: float = ALPHA, interval_only: bool = False
) -> dict:
    beta = nuisance_beta(alpha)
    n_c, n_t, p_c, p_t = cell.n_c, cell.n_t, cell.p_c, cell.p_t
    true_rr = cell.risk_ratio
    lo_c, hi_c, omitted_c = exact_outer_window(n_c, p_c, _XC_TAIL_BUDGET)
    accumulation_error = binomial_rr._eps_margin(hi_c - lo_c + 1)

    acc = dict.fromkeys(METRIC_NAMES, 0.0)
    uncond_true_rr_noncov = cond_true_rr_noncov_numer = 0.0
    interval_miss = point_interval_miss = 0.0
    resolved_unresolved = 0.0
    cp_miss_mass = point_cp_miss_mass = 0.0
    window_mass = 0.0

    for x_c_obs in range(lo_c, hi_c + 1):
        w = float(_binom.pmf(x_c_obs, n_c, p_c))
        if w <= 0.0:
            continue
        window_mass += w
        a, b = binomial_rr.clopper_pearson(x_c_obs, n_c, beta)
        if not (a <= p_c <= b):
            cp_miss_mass += w
            if x_c_obs > 0:
                point_cp_miss_mass += w
            continue

        rp_true = resolve_plus_tail(x_c_obs, n_c, n_t, beta, true_rr, p_c, p_t, alpha / 2.0)
        rm_true = resolve_minus_tail(x_c_obs, n_c, n_t, beta, true_rr, p_c, p_t, alpha / 2.0)
        miss_bound = _interval_miss_bound(rp_true, rm_true, n_t, p_t, zero_control=x_c_obs == 0)
        interval_miss += w * miss_bound
        if x_c_obs > 0:
            point_interval_miss += w * miss_bound
        if interval_only:
            continue

        covered_true, unres_true = two_sided_covered_mass(rp_true, rm_true, n_t, p_t)
        resolved_unresolved += w * unres_true
        uncond_true_rr_noncov += w * (1.0 - covered_true)
        if x_c_obs > 0:
            cond_true_rr_noncov_numer += w * (1.0 - covered_true)

        if true_rr != 1.0:
            rp_null = resolve_plus_tail(x_c_obs, n_c, n_t, beta, 1.0, p_c, p_t, alpha / 2.0)
            rm_null = resolve_minus_tail(x_c_obs, n_c, n_t, beta, 1.0, p_c, p_t, alpha / 2.0)
            covered_null, unres_null = two_sided_covered_mass(rp_null, rm_null, n_t, p_t)
            resolved_unresolved += w * unres_null
        else:
            covered_null = covered_true
        acc["two_sided_reject_null"] += w * (1.0 - covered_null)

        rp_greater = resolve_plus_tail(x_c_obs, n_c, n_t, beta, 1.0, p_c, p_t, alpha)
        accept_g, unres_g = single_tail_accept_mass("plus", rp_greater, n_t, p_t)
        acc["greater_reject_null"] += w * (1.0 - accept_g)
        resolved_unresolved += w * unres_g

        rm_less = resolve_minus_tail(x_c_obs, n_c, n_t, beta, 1.0, p_c, p_t, alpha)
        accept_l, unres_l = single_tail_accept_mass("minus", rm_less, n_t, p_t)
        acc["less_reject_null"] += w * (1.0 - accept_l)
        resolved_unresolved += w * unres_l

    bias, p_xc_pos = conditional_bias_exact(n_c, p_c, p_t, cell.true_lift)
    point_mass_lower = max(0.0, math.nextafter(p_xc_pos - _eps_margin(1), -math.inf))
    total_uncertain = omitted_c + cp_miss_mass + resolved_unresolved + accumulation_error
    metrics = (
        {}
        if interval_only
        else _test_error_metrics(
            acc,
            true_rr,
            uncond_true_rr_noncov,
            cond_true_rr_noncov_numer,
            p_xc_pos,
            point_mass_lower,
            total_uncertain,
        )
    )

    metrics["true_rr_noncoverage"] = {
        "method": "witness_enclosure",
        "unconditional": {
            "value": None,
            "bound": [
                0.0,
                min(1.0, interval_miss + cp_miss_mass + omitted_c + accumulation_error),
            ],
        },
        "conditional_on_point_available": (
            None
            if point_mass_lower == 0.0
            else {
                "value": None,
                "bound": [
                    0.0,
                    min(
                        1.0,
                        (point_interval_miss + point_cp_miss_mass + omitted_c + accumulation_error)
                        / point_mass_lower,
                    ),
                ],
                "conditioning_mass": p_xc_pos,
            }
        ),
    }
    metrics["point_conditional_bias"] = {"value": bias, "p_point_available": p_xc_pos}
    evidence = production_evidence(cell, alpha)
    error_limit = alpha + scientific_delta(alpha)
    passed = metrics["true_rr_noncoverage"]["unconditional"]["bound"][1] <= error_limit
    passed = passed and all(
        record["bound"][1] <= error_limit
        for name, record in metrics.items()
        if name.startswith("type_I_")
    )
    return {
        "cell": asdict(cell),
        "truth": {"p_c": p_c, "p_t": p_t, "true_rr": true_rr, "true_lift": cell.true_lift},
        "alpha": alpha,
        "acceptance": {
            "scope": "cell_errors_and_probe_reportability",
            "error_limit": error_limit,
            "error_passed": passed,
            "reportability_passed": evidence["reportability_passed"],
            "passed": passed and evidence["reportability_passed"],
        },
        "production_evidence": evidence,
        "support": {
            "xc_window": [lo_c, hi_c],
            "xc_window_mass": window_mass,
            "omitted_xc_tail_mass": omitted_c,
            "cp_miss_mass": cp_miss_mass,
            "point_available_mass": p_xc_pos,
            "availability_basis": "structural_model; observed checks are in production_evidence",
            "set_available_mass": 1.0,
            "set_only_mass": 1.0 - p_xc_pos,
        },
        "metrics": metrics,
        "truncation_budget": {
            "xc_tail_budget": _XC_TAIL_BUDGET,
            "xt_tail_budget": _XT_TAIL_BUDGET,
            "omitted_xc_tail_mass": omitted_c,
            "cp_miss_mass": cp_miss_mass,
            "resolved_unresolved_xt_mass": resolved_unresolved,
            "accumulation_error": accumulation_error,
            "total_uncertain_mass": total_uncertain,
        },
    }


# --- CLI ----------------------------------------------------------------


def _worker(args: tuple[ManifestCell, float, bool]) -> dict:
    cell, alpha, interval_only = args
    return calibrate_cell(cell, alpha=alpha, interval_only=interval_only)


def run_grid(
    cells: tuple[ManifestCell, ...],
    *,
    alpha: float = ALPHA,
    workers: int = 1,
    interval_only: bool = False,
) -> list[dict]:
    if workers <= 1:
        return [calibrate_cell(c, alpha=alpha, interval_only=interval_only) for c in cells]
    with multiprocessing.Pool(workers) as pool:
        results = []
        for result in pool.imap(_worker, ((c, alpha, interval_only) for c in cells), chunksize=1):
            results.append(result)
            print(f"calibrated {len(results)}/{len(cells)}", file=sys.stderr, flush=True)
        return results


_PILOT_INDICES: tuple[int, ...] = tuple(
    i for i, c in enumerate(MANIFEST) if c.expected_events in (0.5, 1, 2) and c.p_c in (1e-2, 1e-1)
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pilot", action="store_true", help="diagnostic subset; cannot pass full-grid acceptance"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--interval-only",
        action="store_true",
        help="certify coverage/geometry only; missing power prevents full-grid acceptance",
    )
    args = parser.parse_args()

    cells = tuple(MANIFEST[i] for i in _PILOT_INDICES) if args.pilot else MANIFEST
    t0 = time.perf_counter()
    results = run_grid(
        cells, alpha=args.alpha, workers=args.workers, interval_only=args.interval_only
    )
    elapsed = time.perf_counter() - t0

    acceptance = grid_acceptance(results, alpha=args.alpha)
    payload = {
        "acceptance": acceptance,
        "alpha": args.alpha,
        "n_cells": len(results),
        "elapsed_seconds": elapsed,
        "interval_only": args.interval_only,
        "results": results,
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {len(results)} cells to {args.output} in {elapsed:.1f}s")
    else:
        print(text)
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
