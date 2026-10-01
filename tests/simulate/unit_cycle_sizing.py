"""Prospective unit-cycle calibration sizing; authoring it does not freeze repetitions.

The ``--ledger`` release ledger has family_alpha <= .01 and reservations, each
containing task, alpha and nominal_failure. The unit-cycle reservation
(``task == "I14"``) must already hold alpha=.001.
This program reads that ledger, never changes it, and writes a proposed frozen
schedule and entry. Run BEFORE any exploratory or confirmatory outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

DESIGNS = 2160
DIRECTIONS = DESIGNS * (2 + 8 + 3 * 8 + 15) + 5
TERMINAL_GATES = DESIGNS * 15 + 5
CHECKPOINTS = 4
TOTAL_MARGIN = 0.0025
COMPONENT_MARGIN = TOTAL_MARGIN / 2
MAX_EXPERIMENT_EVALUATIONS = 1_000_000_000


@dataclass(frozen=True)
class Gate:
    name: str
    nominal: float
    lower: float | None = None
    upper: float | None = None
    margin: float = COMPONENT_MARGIN


GATES = (
    Gate("coverage", 0.95, lower=0.945),
    Gate("availability", 1 - 1e-6, lower=0.99),
    Gate("null_size", 0.05, upper=0.055),
    # On the reference-band event, its full interval is within 2*m_ref of
    # nominal .80. Reserve that width inside the unchanged +/-.005 agreement.
    Gate(
        "mde_equivalence",
        0.80,
        lower=0.795 + 2 * COMPONENT_MARGIN,
        upper=0.805 - 2 * COMPONENT_MARGIN,
    ),
    Gate("oracle_disagreement", 0, upper=0.00125),
    Gate("useful_power", 0.90, lower=0.80),
    # No scientific rate target for a model-power precision measurement. This
    # gate is sized uniformly over all counts, hence over every nominal rate.
    Gate("power_precision", 0.5),
)


def cp_bounds(k, n, eta):
    """One-sided CP tail inversions, with outward binary64 endpoint rounding.

    Lower uses eta directly; upper uses isf, never a 1-eta quantile. scipy's
    special-function accuracy remains a numerical dependency for Main's audit.
    """
    counts = np.asarray(k)
    lo = np.where(counts == 0, 0.0, beta.ppf(eta, np.maximum(counts, 1), n - counts + 1))
    hi = np.where(counts == n, 1.0, beta.isf(eta, counts + 1, np.maximum(n - counts, 1)))
    return np.maximum(0.0, np.nextafter(lo, -np.inf)), np.minimum(1.0, np.nextafter(hi, np.inf))


def gate_bounds(gate, k, n, eta):
    if not n:
        return {"passed": False, "rate": None, "lower": None, "upper": None, "margin": None}
    lo, hi = (float(x) for x in cp_bounds(k, n, eta))
    rate = k / n
    margins = []
    if gate.lower is not None or gate.upper is None:
        margins.append(rate - lo)
    if gate.upper is not None or gate.lower is None:
        margins.append(hi - rate)
    margin = max(margins)
    passed = (
        (gate.lower is None or lo >= gate.lower)
        and (gate.upper is None or hi <= gate.upper)
        and margin <= gate.margin
    )
    return {
        "passed": passed,
        "rate": rate,
        "lower": lo,
        "upper": hi,
        "margin": margin,
        "eta": eta,
        "target_lower": gate.lower,
        "target_upper": gate.upper,
    }


def gate_decision(gate, k, n, eta):
    """Decide declared thresholds without requiring a fixed precision margin."""
    bounds = gate_bounds(gate, k, n, eta)
    status = "inconclusive"
    if n:
        lo, hi = bounds["lower"], bounds["upper"]
        if (gate.lower is not None and hi < gate.lower) or (
            gate.upper is not None and lo > gate.upper
        ):
            status = "failed"
        elif (
            (gate.lower is not None or gate.upper is not None)
            and (gate.lower is None or lo >= gate.lower)
            and (gate.upper is None or hi <= gate.upper)
        ):
            status = "certified"
    return {**bounds, "passed": status == "certified", "status": status}


def calibration_schedule(*, total_error, cases, checkpoints):
    """Allocate a family budget over cases and a summable checkpoint schedule."""
    if not math.isfinite(total_error) or not 0 < total_error < 1:
        raise ValueError("total_error must be finite and between zero and one")
    if isinstance(cases, bool) or not isinstance(cases, int) or cases < 1:
        raise ValueError("cases must be a positive integer")
    checkpoints = tuple(checkpoints)
    if (
        not checkpoints
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in checkpoints)
        or any(a >= b for a, b in zip(checkpoints, checkpoints[1:], strict=False))
    ):
        raise ValueError("checkpoints must be positive, strictly increasing integers")
    case_error = allocate(total_error, cases)
    return {
        "total_error": float(total_error),
        "case_error": case_error,
        "checkpoint_error": tuple(
            allocate(case_error, j * (j + 1)) for j in range(1, len(checkpoints) + 1)
        ),
        "checkpoints": checkpoints,
    }


def passing_count_set(gate, n, eta, chunk_size=65536):
    """All K_g(n), as disjoint inclusive integer runs; never assume monotonicity.

    Precision as a function of k is not assumed monotone. Evaluate every count
    in bounded memory; scientific inequalities and precision both participate.
    """
    runs = []
    for start in range(0, n + 1, chunk_size):
        counts = np.arange(start, min(n + 1, start + chunk_size))
        lo, hi = cp_bounds(counts, n, eta)
        rates = counts / n
        passing = np.ones(len(counts), dtype=bool)
        if gate.lower is not None:
            passing &= lo >= gate.lower
        if gate.upper is not None:
            passing &= hi <= gate.upper
        if gate.lower is not None or gate.upper is None:
            passing &= rates - lo <= gate.margin
        if gate.upper is not None or gate.lower is None:
            passing &= hi - rates <= gate.margin
        edges = np.diff(np.r_[False, passing, False].astype(np.int8))
        for left, right in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        ):
            a, b = int(counts[left]), int(counts[right - 1])
            if runs and a == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], b)
            else:
                runs.append((a, b))
    return runs


def covers_every_count(runs, n):
    """Whether the passing set is every count 0..n, in any disjoint-run encoding."""
    return (
        bool(runs)
        and runs[0][0] == 0
        and runs[-1][1] == n
        and sum(high - low + 1 for low, high in runs) == n + 1
    )


def nominal_failure(gate, n, runs):
    """Integrate the complement directly, avoiding 1 minus a near-one mass."""
    gaps, start = [], 0
    for a, b in runs:
        if a > start:
            gaps.append((start, a - 1))
        start = b + 1
    if start <= n:
        gaps.append((start, n))
    masses = []
    for a, b in gaps:
        if a == 0:
            mass = binom.cdf(b, n, gate.nominal)
        elif b == n:
            mass = binom.sf(a - 1, n, gate.nominal)
        elif b < n * gate.nominal:
            mass = binom.cdf(b, n, gate.nominal) - binom.cdf(a - 1, n, gate.nominal)
        else:
            mass = binom.sf(a - 1, n, gate.nominal) - binom.sf(b, n, gate.nominal)
        masses.append(float(mass))
    return min(1.0, math.nextafter(math.fsum(masses), math.inf)) if masses else 0.0


def terminal_size(gate, eta, zeta, *, max_repetitions):
    """First passing dyadic candidate; no unsupported claim of minimal integer R."""
    n = 2 ** (CHECKPOINTS - 1)
    while n <= max_repetitions:
        runs = passing_count_set(gate, n, allocate(eta, CHECKPOINTS * (CHECKPOINTS + 1)))
        failure = nominal_failure(gate, n, runs)
        uniform = gate.name == "power_precision"
        if covers_every_count(runs, n) if uniform else failure <= zeta:
            return {
                "terminal": n,
                "passing_counts": runs,
                "nominal_failure_upper": failure,
                "nominal": gate.nominal,
                "eta_terminal": allocate(eta, 20),
                "zeta": zeta,
                "gate": asdict(gate),
                "uniform_all_nominal_rates": uniform,
            }
        n *= 2
    raise RuntimeError(
        f"sizing incomplete for {gate.name}: explicit computation limit {max_repetitions}"
    )


def allocate(total, parts):
    exact = Fraction(total) / parts
    rounded = float(exact)
    return math.nextafter(rounded, -math.inf) if Fraction(rounded) > exact else rounded


def planning_size(eta):
    """Production two-sided power has four endpoint terms per bound."""
    epsilon = Fraction(COMPONENT_MARGIN) / 4
    with localcontext() as context:
        context.prec = 80
        ratio = (Decimal(2) / Decimal(eta)).next_plus()
        log_upper = Fraction(ratio.ln().next_plus())
    return math.ceil(log_upper / (2 * epsilon**2))


def prospective_cost(terminal, planning_n):
    """Count every sampling pass, including same-seed MDE certificate replay."""
    fixed = max(terminal, planning_n)
    cost = {
        "maximum_runtime_decisions": (3 * DESIGNS + 1) * terminal,
        "maximum_confirmation_draws": (2 * DESIGNS + 1) * terminal,
        "independent_exploratory_draws": DESIGNS * planning_n,
        "production_search_draws": DESIGNS * planning_n,
        "mde_certificate_rerun_draws": DESIGNS * planning_n,
        "fixed_model_draws": 3 * DESIGNS * fixed,
    }
    planning = 3 * DESIGNS * (planning_n + fixed)
    cost["maximum_sample_generations"] = planning + cost["maximum_confirmation_draws"]
    cost["maximum_experiment_evaluations"] = planning + cost["maximum_runtime_decisions"]
    cost["experiment_evaluation_ceiling"] = MAX_EXPERIMENT_EVALUATIONS
    return cost


def enforce_cost_ceiling(cost):
    if cost["maximum_experiment_evaluations"] > MAX_EXPERIMENT_EVALUATIONS:
        raise RuntimeError(
            "I14 certification incomplete: prospective cost "
            f"{cost['maximum_experiment_evaluations']:,} experiment evaluations exceeds "
            f"the {MAX_EXPERIMENT_EVALUATIONS:,} ceiling, including MDE certificate replay; "
            "no feasible certification implementation or release evidence exists"
        )


def freeze(ledger, *, max_repetitions):
    reservations = ledger["reservations"]
    if len({x["task"] for x in reservations}) != len(reservations):
        raise ValueError("duplicate task reservations")
    if any(
        not math.isfinite(x[key]) or x[key] <= 0
        for x in reservations
        for key in ("alpha", "nominal_failure")
    ):
        raise ValueError("every release reservation must have positive finite allocations")
    if not 0 < ledger["family_alpha"] <= 0.01:
        raise ValueError("release-wide family_alpha must be in (0,.01]")
    if sum(Fraction(x["alpha"]) for x in reservations) > Fraction(ledger["family_alpha"]):
        raise ValueError("existing release reservations exceed the ledger")
    if sum(Fraction(x["nominal_failure"]) for x in reservations) > Fraction(ledger["family_alpha"]):
        raise ValueError("nominal certification-failure reservations exceed the ledger")
    selected = [x for x in reservations if x["task"] == "I14"]
    if len(selected) != 1 or selected[0]["alpha"] != 0.001:
        raise ValueError("Main must record exactly one existing I14 .001 reservation first")
    allocation = selected[0]
    if not 0 < allocation["nominal_failure"] <= 0.01:
        raise ValueError("Main must reserve positive nominal certification-failure probability")
    # Split nominal failure between passing-count failures and all simultaneous
    # MC-band failures, including independent reference/planning uncertainty.
    eta = min(
        allocate(allocation["alpha"], DIRECTIONS),
        allocate(allocation["nominal_failure"], 2 * DIRECTIONS),
    )
    zeta = allocate(allocation["nominal_failure"], 2 * TERMINAL_GATES)
    planning_n = planning_size(eta)
    # Refuse even the terminal-zero lower cost before expensive CP sizing.
    enforce_cost_ceiling(prospective_cost(0, planning_n))
    sized = {g.name: terminal_size(g, eta, zeta, max_repetitions=max_repetitions) for g in GATES}
    terminal = max(x["terminal"] for x in sized.values())
    # Recompute every passing set at the common terminal; dyadic monotonicity
    # is not silently assumed for the intersection of precision and criteria.
    while True:
        final = {}
        for g in GATES:
            runs = passing_count_set(g, terminal, allocate(eta, 20))
            failure = nominal_failure(g, terminal, runs)
            final[g.name] = {
                "passing_counts": runs,
                "nominal_failure_upper": failure,
                "gate": asdict(g),
                "nominal": g.nominal,
                "zeta": zeta,
            }
        if all(
            (
                x["passing_counts"] == [(0, terminal)]
                if key == "power_precision"
                else x["nominal_failure_upper"] <= zeta
            )
            for key, x in final.items()
        ):
            break
        terminal *= 2
        if terminal > max_repetitions:
            raise RuntimeError("common terminal sizing incomplete at explicit computation limit")
    checkpoints = [terminal // 2 ** (CHECKPOINTS - j) for j in range(1, CHECKPOINTS + 1)]
    cost = prospective_cost(terminal, planning_n)
    enforce_cost_ceiling(cost)
    return {
        "status": "frozen_before_outcomes",
        "ledger_entry": allocation,
        "release_reservations": reservations,
        "release_family_alpha": ledger["family_alpha"],
        "directional_quantities": DIRECTIONS,
        "terminal_gates": TERMINAL_GATES,
        "eta_per_direction": eta,
        "zeta_per_gate": zeta,
        "nominal_failure_allocation": {
            "passing_count_share": allocate(allocation["nominal_failure"], 2),
            "simultaneous_band_share": allocate(allocation["nominal_failure"], 2),
            "union_bound": float(DIRECTIONS * Fraction(eta) + TERMINAL_GATES * Fraction(zeta)),
            "mde_scientific_interval": [0.795, 0.805],
            "mde_count_set_reserves_reference_width": 2 * COMPONENT_MARGIN,
            "power_equivalence": "Uniform precision plus simultaneous inclusion of common true power implies total interval discrepancy <=2*(m_ref+m_runtime)=.005",
        },
        "checkpoints": checkpoints,
        "checkpoint_eta": [allocate(eta, j * (j + 1)) for j in range(1, CHECKPOINTS + 1)],
        "terminal": terminal,
        "terminal_passing_sets": final,
        "preliminary_sizing": sized,
        "planning_repetitions": planning_n,
        "planning_mc_error": 8 * eta,
        "fixed_model_repetitions": max(terminal, planning_n),
        "fixed_model_mc_error": 8 * eta,
        "maximum_total_margin": TOTAL_MARGIN,
        "component_margin": COMPONENT_MARGIN,
        "prospective_cost": cost,
        "nominal_scope": "MDE power .80; coverage >=.95; availability >=1-1e-6; disagreement 0; witness >=.90. A shifted selected candidate may remain incomplete.",
        "stopping": "Accept only when every gate passes at a frozen checkpoint; unresolved terminal is incomplete",
        "numerical_dependency": "SciPy beta/binomial inversion with outward float rounding; Main must audit tail accuracy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-repetitions",
        type=int,
        required=True,
        help="explicit sizing computation limit, not an asserted repetition count",
    )
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    result = freeze(json.loads(raw), max_repetitions=args.max_repetitions)
    result["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest = Path(__file__).parents[1] / "unit_cycle_prospective.json"
    result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
