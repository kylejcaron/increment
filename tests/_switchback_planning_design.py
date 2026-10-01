"""Prospective switchback-planning variance experiment, independent of production calculations.

Let L be the retained length, Z~Bernoulli(p), w_CT=1/(2p), and
w_TC=1/(2(1-p)). The period model is
Y_jt = 10 + unit_intercept + h*j + (tau+U)*treatment + epsilon_jt,
where U is +/-u with equal probability and epsilon is independent uniform
on [-sqrt(3), sqrt(3)], with variance one and fourth cumulant -6/5.
U persists over a unit's cycles. For shared schedules it is a fresh common
block shock. A random unit intercept persists over cycles under both laws;
it cancels from period contrasts, so shared block contributions stay independent.
Discarded rows have additional finite-history contamination.

The two branch means are L*(tau+h), L*(tau-h); each branch variance is
L^2*u^2 + 2L/M, with M=1 for units and M=roster size for shared blocks.
Writing B=(1-p)*d_CT-p*d_TC gives Var(A)=(B^2+branch_variance)/(4p(1-p)),
Var(G)=(1-2p)^2/(4p(1-p)), Cov(A,G)=B*(1-2p)/(4p(1-p)). Distinct cycles
of one unit have Cov(A_c,A_d)=L^2*u^2, and all cross-cycle covariances
involving G vanish. Thus Var(unit A)=L^2*u^2+(Var(block A)-L^2*u^2)/C;
Var(unit G) and Cov(unit A,unit G) are divided by C exactly once.

For iid contributions X, E[S^2/N]=Var(X)/N and
Var(S^2/Var(X))=(mu4/Var(X)^2-(N-3)/(N-1))/N. This is the MCSE, not
a binomial error. For a finite-sample family bound use the order-two
U-statistic S^2=average_{i<j}(X_i-X_j)^2/2. Jensen's inequality over
pairings bounds its MGF by an average of floor(N/2) independent kernels.
The normalized kernel has variance (mu4/Var(X)^2+1)/2 and range
[0, (max(X)-min(X))^2/(2Var(X))]. Bernstein therefore applies with
R*floor(N/2) observations; no normal approximation or observed tuning.

These are variance gates only. The release-wide .005 coverage/error tolerance
does not have variance units. Interval coverage remains a separate calibration
obligation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product

from tests.mc import family_eta

REFERENCE_PER_STEP = 0.4
ALTERNATIVE_PER_STEP = 0.75
PERIOD_TREND = 0.6
PERSISTENT_SD = 0.8
NOISE_HALF_WIDTH = math.sqrt(3)
UNIT_COUNT = 16
SHARED_ROSTER = ("u0", "u1", "u2")
WASHOUT = 1
BATCH_SIZE = 64
ROOT_SEED = 202609170311
SMOKE_SEED = 202609170312
RELATIVE_TOLERANCE = 0.05
MAX_MC_MARGIN = 0.025
# A reserved slice of the release's .01 family budget, not a per-cell reset.
FAMILY_ALPHA = 0.001
GATED_STATISTICS = ("runtime_variance", "pilot_reference_variance", "pilot_shifted_variance")


@dataclass(frozen=True)
class Cell:
    shared: bool
    probability_ct: float
    cycles: int
    observation_steps: int
    carryover_order: int

    @property
    def retained(self) -> int:
        return self.observation_steps - self.carryover_order

    @property
    def n(self) -> int:
        return self.cycles if self.shared else UNIT_COUNT

    @property
    def id(self) -> str:
        law = "shared" if self.shared else "unit"
        return (
            f"{law}-p{self.probability_ct}-c{self.cycles}"
            f"-o{self.observation_steps}-k{self.carryover_order}"
        )


CELLS = tuple(
    Cell(shared, p, cycles, observations, order)
    for shared, p, cycles, observations, order in product(
        (False, True), (0.5, 0.75), (5, 20), (1, 4), (0, 1, 2)
    )
    if order < observations
)
GATED_COUNT = len(CELLS) * len(GATED_STATISTICS)
ETA = family_eta(FAMILY_ALPHA, GATED_COUNT)


def covariance(cell: Cell) -> tuple[float, float, float]:
    """Population Var(A), Var(G), Cov(A,G) from the two branch law."""
    p, length = cell.probability_ct, cell.retained
    roster = len(SHARED_ROSTER) if cell.shared else 1
    ct = length * (REFERENCE_PER_STEP + PERIOD_TREND)
    tc = length * (REFERENCE_PER_STEP - PERIOD_TREND)
    between = (1 - p) * ct - p * tc
    branch_variance = length**2 * PERSISTENT_SD**2 + 2 * length / roster
    va = (between**2 + branch_variance) / (4 * p * (1 - p))
    vg = (1 - 2 * p) ** 2 / (4 * p * (1 - p))
    cov = between * (1 - 2 * p) / (4 * p * (1 - p))
    if not cell.shared:
        persistent = length**2 * PERSISTENT_SD**2
        va = persistent + (va - persistent) / cell.cycles
        vg /= cell.cycles
        cov /= cell.cycles
    return va, vg, cov


@dataclass(frozen=True)
class Moments:
    variance: float
    fourth: float
    lower: float
    upper: float

    def variance_of_sample_variance_ratio(self, n: int) -> float:
        return (self.fourth / self.variance**2 - (n - 3) / (n - 1)) / n

    def margin(self, repetitions: int, n: int) -> float:
        """Two-sided Bernstein bound, with ETA allocated to each direction."""
        independent_pairs = repetitions * (n // 2)
        log_budget = -math.log(ETA)
        kernel_variance = (self.fourth / self.variance**2 + 1) / 2
        kernel_bound = (self.upper - self.lower) ** 2 / (2 * self.variance)
        return math.sqrt(2 * kernel_variance * log_budget / independent_pairs) + (
            2 * kernel_bound * log_budget / (3 * independent_pairs)
        )


def moments(cell: Cell, per_step: float) -> Moments:
    """Enumerate CT counts and persistent signs; integrate period noise exactly.

    Conditional on K CT cycles and U, the noise is a weighted sum of independent
    uniform noises. Its fourth cumulant is -6/5 times the coefficient fourth
    powers. Mixing m^4+6*m^2*v+3*v^2+kappa4 gives the unconditional fourth moment.
    """
    p, length = cell.probability_ct, cell.retained
    cycles = 1 if cell.shared else cell.cycles
    roster = len(SHARED_ROSTER) if cell.shared else 1
    w_ct, w_tc = 1 / (2 * p), 1 / (2 * (1 - p))
    second_terms, fourth_terms = [], []
    for k in range(cycles + 1):
        mass = math.comb(cycles, k) * p**k * (1 - p) ** (cycles - k) / 2
        g = (k * w_ct + (cycles - k) * w_tc) / cycles
        signed_g = (k * w_ct - (cycles - k) * w_tc) / cycles
        noise_variance = 2 * length * (k * w_ct**2 + (cycles - k) * w_tc**2) / (roster * cycles**2)
        fourth_cumulant = (
            -12 / 5 * length * (k * w_ct**4 + (cycles - k) * w_tc**4) / (roster**3 * cycles**4)
        )
        for sign in (-1, 1):
            centered = length * (
                per_step * (g - 1) + PERIOD_TREND * signed_g + sign * PERSISTENT_SD * g
            )
            second_terms.append(mass * (centered**2 + noise_variance))
            fourth_terms.append(
                mass
                * (
                    centered**4
                    + 6 * centered**2 * noise_variance
                    + 3 * noise_variance**2
                    + fourth_cumulant
                )
            )
    lower = min(
        w * length * (per_step + sign * PERIOD_TREND - PERSISTENT_SD - 2 * NOISE_HALF_WIDTH)
        for w, sign in ((w_ct, 1), (w_tc, -1))
    )
    upper = max(
        w * length * (per_step + sign * PERIOD_TREND + PERSISTENT_SD + 2 * NOISE_HALF_WIDTH)
        for w, sign in ((w_ct, 1), (w_tc, -1))
    )
    return Moments(math.fsum(second_terms), math.fsum(fourth_terms), lower, upper)


def repetitions(cell: Cell) -> int:
    """Size from the frozen law only, rounded to complete source batches."""
    laws = (moments(cell, REFERENCE_PER_STEP), moments(cell, ALTERNATIVE_PER_STEP))
    count = BATCH_SIZE
    while max(law.margin(count, cell.n) for law in laws) > MAX_MC_MARGIN:
        count *= 2
    low, high = 0, count // BATCH_SIZE
    while high - low > 1:
        mid = (low + high) // 2
        if max(law.margin(mid * BATCH_SIZE, cell.n) for law in laws) <= MAX_MC_MARGIN:
            high = mid
        else:
            low = mid
    return high * BATCH_SIZE


def manifest() -> dict[str, object]:
    """Serialize the prospective design without reading observed outcomes."""
    return {
        "task": "C11 variance agreement; not C12 coverage acceptance",
        "base_commit": "12893c1023e81b737149838b755e0f393ca9bfcc",
        "root_seed": ROOT_SEED,
        "seed_streams": [
            "assignment=0",
            "unit_intercept_and_persistent_effect=1",
            "period_noise=2",
        ],
        "rng": "numpy PCG64; SeedSequence([root_seed, cell_index, stream])",
        "batch_size": BATCH_SIZE,
        "family_alpha": FAMILY_ALPHA,
        "gated_statistics": GATED_STATISTICS,
        "gated_count": GATED_COUNT,
        "per_direction_eta": ETA,
        "relative_variance_tolerance": RELATIVE_TOLERANCE,
        "maximum_mc_margin": MAX_MC_MARGIN,
        "gate": "abs(observed_ratio-1)+Bernstein_margin <= .05",
        "bound": "order-two sample-variance U-statistic; exact bounded-law fourth moment",
        "reference_per_step": REFERENCE_PER_STEP,
        "alternative_per_step": ALTERNATIVE_PER_STEP,
        "period_trend": PERIOD_TREND,
        "persistent_sd": PERSISTENT_SD,
        "unit_intercept": "Uniform[-.5,.5], constant within unit/experiment; cancels",
        "period_noise": "independent Uniform[-sqrt(3),sqrt(3)]",
        "unit_count": UNIT_COUNT,
        "shared_roster": SHARED_ROSTER,
        "washout_steps": WASHOUT,
        "availability": "every attempted result required; no redraw, exclusion or null imputation",
        "coverage": "R03 conditional/unconditional accounting reported, not gated here",
        "cells": [
            {"id": cell.id, **asdict(cell), "n": cell.n, "repetitions": repetitions(cell)}
            for cell in CELLS
        ],
    }
