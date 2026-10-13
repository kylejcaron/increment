"""Preregistered mixed-covariate linear CUPED certification (Revision 7, amended by Revision 8).

Commands, run from the checkout with BLAS pinned to one thread per worker
(``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1``)::

    uv run python -m calibration.cuped_linear --campaign reduced --prerequisites
    uv run python -m calibration.cuped_linear --campaign reduced --project
    uv run python -m calibration.cuped_linear --campaign reduced --output DIR --workers 8
    uv run python -m calibration.cuped_linear --campaign full --output DIR --workers 8
    uv run python -m calibration.cuped_linear --campaign smoke --output DIR

``full`` keeps the Revision 7 gates: 107,100 eligible draws per statistic at the
strict tolerance 1/200 (coverage lower bound >= .945, mean false-coverage
proportion upper bound <= .055). ``reduced`` is the Revision 8 amendment at the
enumerated reduced tolerance 1/100 (lower bound >= .94, upper bound <= .06) with
28,000 eligible draws, sized by the same helpers so that a nominal statistic
passes with probability at least 1-.1/67 and true coverage .92 (mean false
coverage proportion .08) fails with probability at least .99; coverage numbers
are exact binomial, FCP numbers are a Bernoulli benchmark carried beside their
bounded-observation guarantees. Both spend the shared 67-statistic allocation,
including the 12 sitewide slots owned by the separate campaign, and
``--project`` prints the projected CPU-hours from the measured per-attempt
costs. Smoke exercises the same designs but cannot certify them. Revision 8
also declares eight edge cells outside the gated set before any coverage
outcome exists; see ``manifest_record``. A failed gate never licenses a Normal
interval.

Every profile needs two production prerequisites, probed before any draw:
``fit_cuped_linear`` must refuse a fit outside the M1 support region with the
registered ``estimation.cuped_linear.support_region`` code and the harness
observables, and the public ``cuped_linear`` route must run. Without them the
campaign refuses with ``calibration.cuped_linear.prerequisite``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from scipy.special import gammainc, gammaincc
from scipy.stats import binom

from calibration.journal import DrawJournal, verify
from increment.errors import CodedError, refusals, refuse
from increment.estimation import cuped_linear as production
from increment.estimation.cuped_linear import fit_cuped_linear
from tests.mc import coverage_lower_bound, family_eta, kl_chernoff_upper_bound, mcse

FAMILY_SIZE = 67
ETA = family_eta(0.01, FAMILY_SIZE)
ALPHA = 0.05
NOMINAL_COVERAGE = Fraction(19, 20)
NOMINAL_FCP = Fraction(1, 20)
# The enumerated tolerances of calibration/profiles.yaml; a profile never
# supplies a free value. A gate sits one tolerance inside nominal and the
# detectable alternative three tolerances outside it, as in Revision 7.
TOLERANCES = MappingProxyType({"strict": Fraction(1, 200), "reduced": Fraction(1, 100)})
PROFILE_TOLERANCES = MappingProxyType({"smoke": "strict", "full": "strict", "reduced": "reduced"})
REPLICATIONS = MappingProxyType({"smoke": 3, "full": 107_100, "reduced": 28_000})
FULL_REPLICATIONS = REPLICATIONS["full"]
REDUCED_REPLICATIONS = REPLICATIONS["reduced"]
SMOKE_REPLICATIONS = REPLICATIONS["smoke"]
# D8 is descriptive: 4000 eligible draws under a certifying profile, the smoke
# count under smoke, in projection and execution alike.
PRECISION_REPLICATIONS = MappingProxyType({"smoke": 3, "full": 4000, "reduced": 4000})
PASS_TARGET = 1 - 0.1 / FAMILY_SIZE
FAIL_TARGET = 0.99
# The Revision 7 family: 58 coverage and 9 mean-FCP slots. The all-pass union
# bound keeps this split although the edge cells no longer count toward m.
FAMILY_SLOTS = (58, 9)
# M1, the frozen support region. Production must refuse a fit outside it with
# the registered code and these observables; the harness audits every fit it
# receives against the same limits and blocks, never filters, a returned fit
# outside them.
SUPPORT_REGION = MappingProxyType(
    {
        "p": 203,
        "k_max": 200,
        "leverage_ratio": 0.20,
        "max_leverage": 0.45,
        "ratio_denominator_precision": 0.15,
    }
)
SUPPORT_REQUIREMENTS = MappingProxyType(
    {"minimum_level_arm_count": 2, "full_rank": True, "positive_residual_df": True}
)
# The harness recomputes rho_D from the raw draw while production reads it off
# scale-normalized moments; equal up to rounding, so a harness excess inside
# this relative tolerance is not a violation.
DENOMINATOR_PRECISION_TOLERANCE = 1e-12
SUPPORT_REGION_CODE = "estimation.cuped_linear.support_region"
PUBLIC_ROUTE_UNFINISHED_CODE = "estimation.engine.cuped_linear_unfinished"


class PrerequisiteError(CodedError):
    """A production prerequisite of the campaign is absent from this build."""


_REFUSALS = refusals(
    PrerequisiteError,
    {
        "calibration.cuped_linear.prerequisite": (
            "production prerequisites are absent from this build: {missing!r}; the campaign "
            "cannot attempt a draw until the production slice that provides them is merged",
            ("details",),
        ),
    },
)
BOUNDARY_CANDIDATES = MappingProxyType(
    {
        "p": (203, 53, 8),
        "k_max": (200, 50, 5),
        "leverage_ratio": (0.20, 0.10, 0.05),
        "max_leverage": (0.45, 0.30, 0.15),
        "minimum_level_arm_count": (2, 5, 10),
    }
)
PROFILES = tuple(PROFILE_TOLERANCES)


@dataclass(frozen=True, slots=True)
class Gates:
    """One profile's frozen acceptance rule.

    This campaign's rule is the C.9 one: the certificate is the exact one-sided
    bound at ``ETA``, the statistic passes when that bound clears the floor or
    ceiling, and the replication count is sized so that a nominal statistic
    passes with probability at least ``PASS_TARGET`` and the detectable
    alternative fails with probability at least ``FAIL_TARGET``. It is not the
    I15 operating-point design of ``calibration.stopping``, whose false-failure
    budget at ``k* = ceil(R (q + delta/2))`` is ``ETA`` itself; that design
    needs 118,700 strict and 32,000 reduced replications, and the preregistered
    campaign never declared it.
    """

    profile: str
    tolerance: str
    replications: int
    coverage_floor: float
    fcp_ceiling: float
    detectable_coverage: float
    detectable_fcp: float
    stopping: str = "fixed"
    certificate: str = (
        "exact one-sided bound at eta: Clopper-Pearson for coverage, KL-Chernoff for mean FCP"
    )


def gates(campaign: str) -> Gates:
    tolerance = PROFILE_TOLERANCES[campaign]
    delta = TOLERANCES[tolerance]
    return Gates(
        campaign,
        tolerance,
        REPLICATIONS[campaign],
        float(NOMINAL_COVERAGE - delta),
        float(NOMINAL_FCP + delta),
        float(NOMINAL_COVERAGE - 3 * delta),
        float(NOMINAL_FCP + 3 * delta),
    )


def _largest(replications: int, admissible) -> int:
    low, high = 0, replications
    while low < high:
        middle = (low + high + 1) // 2
        if admissible(middle):
            low = middle
        else:
            high = middle - 1
    return low


def _relative_entropy(x: float, u: float) -> float:
    """Bernoulli relative entropy ``KL(x || u)`` for ``x`` in ``[0, 1)``, ``u`` in ``(0, 1)``."""
    low = (1 - x) * math.log((1 - x) / (1 - u))
    return low if x == 0 else x * math.log(x / u) + low


def fcp_threshold(replications: int, ceiling: float) -> float:
    """The largest mean FCP whose KL certificate at ``ETA`` still clears *ceiling*.

    ``kl_chernoff_upper_bound(x, R, ETA) <= ceiling`` exactly when
    ``R * KL(x || ceiling) >= log(1/ETA)`` with ``x <= ceiling``, so the pass
    region of the FCP gate is ``mean <= x*`` for this real root, not an
    integer miss count: the per-replication FCP is fractional. Zero when even
    an all-zero observation cannot certify at *replications*.
    """
    from scipy.optimize import brentq

    budget = -math.log(ETA)
    if replications * _relative_entropy(0.0, ceiling) < budget:
        return 0.0
    return float(
        brentq(
            lambda x: replications * _relative_entropy(x, ceiling) - budget,
            0.0,
            ceiling - 1e-12,
            xtol=1e-15,
            rtol=1e-15,
        )
    )


def bounded_tail(replications: int, threshold: float, mean: float) -> float:
    """Hoeffding's relative-entropy bound for iid summands in ``[0, 1]``.

    For any iid ``X_i`` in ``[0, 1]`` with mean *mean*, ``P(mean(X) >= t)`` for
    ``t > mean`` and ``P(mean(X) <= t)`` for ``t < mean`` are both at most
    ``exp(-R * KL(t || mean))`` (Hoeffding 1963, Theorem 1). Unlike the
    binomial tail it holds for the fractional per-replication FCP, at the
    price of being a bound rather than an exact probability. On the wrong
    side of *mean* the tail is not small and the bound is the trivial one.
    """
    return math.exp(-replications * _relative_entropy(threshold, mean))


def _pass_bound(replications: int, threshold: float, mean: float) -> float:
    """Lower bound on ``P(mean(X) <= threshold)`` at *mean*; trivial unless above it."""
    return 1 - bounded_tail(replications, threshold, mean) if threshold > mean else 0.0


def _fail_bound(replications: int, threshold: float, mean: float) -> float:
    """Lower bound on ``P(mean(X) > threshold)`` at *mean*; trivial unless below it."""
    return 1 - bounded_tail(replications, threshold, mean) if threshold < mean else 0.0


def sizing(
    replications: int, floor: float, ceiling: float, detectable: tuple[float, float]
) -> dict:
    """Helper-derived cutoffs and the operating probabilities they imply.

    Coverage events are indicators, so their binomial pass and fail
    probabilities are exact. A replication's FCP is ``misses / selected``,
    fractional in ``[0, 1]``; its binomial numbers are the Bernoulli member of
    that bounded class, reported as a benchmark, and the ``_bound`` fields are
    the guarantees that hold for every member. ``satisfied`` is the C.9
    benchmark criterion that sizes the frozen profiles; ``satisfied_bounded``
    additionally needs the FCP guarantees to meet the same targets.
    """
    coverage_cutoff = _largest(
        replications, lambda k: coverage_lower_bound(replications - k, replications, ETA) >= floor
    )
    fcp_cutoff = _largest(
        replications,
        lambda k: kl_chernoff_upper_bound(k / replications, replications, ETA) <= ceiling,
    )
    threshold = fcp_threshold(replications, ceiling)
    nominal = float(NOMINAL_FCP)
    edge = (1 - floor, ceiling)
    record = {
        "replications": replications,
        "coverage_floor": floor,
        "fcp_ceiling": ceiling,
        "coverage_miss_cutoff": coverage_cutoff,
        "coverage_pass_at_nominal": float(binom.cdf(coverage_cutoff, replications, nominal)),
        "coverage_false_acceptance_at_tolerance_edge": float(
            binom.cdf(coverage_cutoff, replications, edge[0])
        ),
        "coverage_fail_at_detectable": float(
            binom.sf(coverage_cutoff, replications, 1 - detectable[0])
        ),
        "fcp_cutoff": fcp_cutoff,
        "fcp_threshold_mean": threshold,
        "fcp_claim": "binomial FCP numbers are the Bernoulli benchmark of a fractional [0, 1] observation; the _bound fields hold for every bounded law",
        "fcp_pass_at_nominal": float(binom.cdf(fcp_cutoff, replications, nominal)),
        "fcp_pass_at_nominal_bound": _pass_bound(replications, threshold, nominal),
        "fcp_false_acceptance_at_tolerance_edge": float(
            binom.cdf(fcp_cutoff, replications, edge[1])
        ),
        "fcp_false_acceptance_at_tolerance_edge_bound": ETA,
        "fcp_fail_at_detectable": float(binom.sf(fcp_cutoff, replications, detectable[1])),
        "fcp_fail_at_detectable_bound": _fail_bound(replications, threshold, detectable[1]),
    }
    coverage_slots, fcp_slots = FAMILY_SLOTS
    for suffix in ("", "_bound"):
        passes = min(record["coverage_pass_at_nominal"], record["fcp_pass_at_nominal" + suffix])
        fails = min(
            record["coverage_fail_at_detectable"], record["fcp_fail_at_detectable" + suffix]
        )
        record["satisfied" + (suffix and "_bounded")] = (
            passes >= PASS_TARGET and fails >= FAIL_TARGET
        )
        record["family_all_pass_lower_bound" + (suffix and "_bounded")] = (
            1
            - coverage_slots * (1 - record["coverage_pass_at_nominal"])
            - fcp_slots * (1 - record["fcp_pass_at_nominal" + suffix])
        )
    return record


@functools.cache
def derive_replications(
    floor: float,
    ceiling: float,
    detectable: tuple[float, float],
    criterion: str = "satisfied",
    step: int = 100,
) -> dict:
    """The smallest multiple of *step* whose cutoffs meet both targets.

    C.9 scanned the lattice linearly; the targets are met on every lattice
    point past the first satisfied one, so a galloping search locates the same
    point, and the slow lattice test checks that identity for both profiles.
    """
    satisfied = functools.cache(
        lambda replications: sizing(replications, floor, ceiling, detectable)[criterion]
    )
    low, high = 0, step
    while not satisfied(high):
        low, high = high, 2 * high
        if high > 1_000_000:
            raise ValueError("no replication count under one million satisfies the targets")
    while high - low > step:
        middle = (low + high) // 2 // step * step
        if satisfied(middle):
            high = middle
        else:
            low = middle
    return sizing(high, floor, ceiling, detectable)


def profile_sizing(campaign: str) -> dict:
    gate = gates(campaign)
    detectable = (gate.detectable_coverage, gate.detectable_fcp)
    frozen = sizing(gate.replications, gate.coverage_floor, gate.fcp_ceiling, detectable)
    derived = derive_replications(gate.coverage_floor, gate.fcp_ceiling, detectable)
    bounded = derive_replications(
        gate.coverage_floor, gate.fcp_ceiling, detectable, "satisfied_bounded"
    )
    return {
        "profile": campaign,
        "tolerance": gate.tolerance,
        "detectable_coverage": gate.detectable_coverage,
        "detectable_fcp": gate.detectable_fcp,
        "pass_target": PASS_TARGET,
        "fail_target": FAIL_TARGET,
        "frozen": frozen,
        "derived_replications": derived["replications"],
        "frozen_matches_derivation": derived["replications"] == gate.replications,
        "bounded_replications": bounded["replications"],
        "bounded_note": "replications at which the FCP guarantees for every bounded law also meet the targets; the frozen count is sized on the Bernoulli benchmark, so the FCP pass target is a benchmark claim there",
    }


def sizing_alternatives() -> list[dict]:
    """Replications at the detectable .92 for each candidate floor; the floor sets R."""
    rows = []
    for floor in (Fraction(189, 200), Fraction(47, 50), Fraction(187, 200), Fraction(93, 100)):
        delta = NOMINAL_COVERAGE - floor
        record = derive_replications(float(floor), float(NOMINAL_FCP + delta), (0.92, 0.08))
        rows.append({key: record[key] for key in ("coverage_floor", "fcp_ceiling", "replications")})
    return rows


@dataclass(frozen=True, slots=True)
class Cell:
    id: str
    statistic: str
    generator: str
    scale: str
    n: int
    allocation: float = 0.5
    k_country: int = 5
    encoded_p: int = 8
    sigma_den: float = 0.5
    effect: float = 0.05
    heterogeneous: bool = False
    failure_boundary: tuple[str, ...] = ()
    failure_consequence: str = "Refuse the named projection in its corresponding design region."
    gating: str = "gated"


def _manifest() -> tuple[Cell, ...]:
    rows = []
    designs = ((400, 0.9), (1000, 0.9), (2000, 0.9), (2000, 0.5))
    for index, (n, allocation) in enumerate(designs, 1):
        rows.append(Cell(f"D3-{index}", "coverage", "mean", "absolute", n, allocation))
    for n, allocation in designs:
        for effect in (0.0, 0.05):
            index = sum(c.id.startswith("D3b-") for c in rows) + 1
            rows.append(
                Cell(
                    f"D3b-{index}",
                    "coverage",
                    "relative_mean",
                    "relative",
                    n,
                    allocation,
                    effect=effect,
                )
            )
    d4 = (
        (2000, 0.5, 50, 53, ("p", "k_max")),
        (4000, 0.9, 50, 53, ("max_leverage",)),
        (20000, 0.5, 200, 203, ("p", "k_max")),
        (4000, 0.5, 200, 203, ("minimum_level_arm_count",)),
        *((2000, 0.5, 5, p, ("leverage_ratio",)) for p in (50, 100, 200)),
        *((10000, 0.9, 5, p, ("leverage_ratio",)) for p in (50, 100, 200)),
        (4000, 0.9, 50, 53, ("max_leverage",)),
    )
    for index, (n, allocation, k, p, boundary) in enumerate(d4, 1):
        rows.append(
            Cell(
                f"D4-{index}",
                "coverage",
                "mean",
                "absolute",
                n,
                allocation,
                k,
                p,
                heterogeneous=index == 11,
                failure_boundary=boundary,
                failure_consequence="Tighten only the named observable boundary to its next preregistered candidate; refuse matrix support if the tightest candidate fails.",
            )
        )
    ratio_designs = ((100, 0.5), (100, 1.5), (1000, 0.5), (1000, 1.5))
    for index, (n, sigma) in enumerate(ratio_designs, 1):
        rows.append(
            Cell(
                f"D5-{index}",
                "coverage",
                "ratio",
                "ratio_absolute",
                2 * n,
                encoded_p=6,
                sigma_den=sigma,
            )
        )
    index = 0
    for n, sigma in ratio_designs:
        for effect in (0.0, 0.05):
            for scale in ("log_ratio", "ratio_absolute"):
                index += 1
                rows.append(
                    Cell(
                        f"D5b-{index}",
                        "coverage",
                        "ratio",
                        scale,
                        2 * n,
                        encoded_p=6,
                        sigma_den=sigma,
                        effect=effect,
                    )
                )
    index = 0
    for n, sigma in ratio_designs:
        for effect in (0.0, 0.05):
            index += 1
            rows.append(
                Cell(
                    f"D5b-FCR-{index}",
                    "fcp",
                    "ratio_family",
                    "relative_ratio",
                    2 * n,
                    encoded_p=6,
                    sigma_den=sigma,
                    effect=effect,
                    failure_consequence="Block BH-selected/reintervaled reporting for every D5b family.",
                )
            )
    for index, scale in enumerate(("absolute", "relative", "ratio_absolute"), 1):
        rows.append(Cell(f"D9-{index}", "coverage", "wls", scale, 2000, encoded_p=1))
    rows.append(
        Cell(
            "D9-4",
            "fcp",
            "wls_family",
            "mixed",
            2000,
            encoded_p=1,
            failure_consequence="Block the three-metric D9 BH output.",
        )
    )
    return tuple(
        replace(row, gating=EDGE_CELLS[row.id]) if row.id in EDGE_CELLS else row for row in rows
    )


# Revision 8 edge cells, declared before any coverage outcome existed from the
# frozen admission probe below; ``manifest_record`` states each cell's basis.
EDGE_CELLS = MappingProxyType(
    {
        "D4-7": "descriptive_only",
        "D5-2": "unsupported_by_support_region",
        "D5b-5": "unsupported_by_support_region",
        "D5b-6": "unsupported_by_support_region",
        "D5b-7": "unsupported_by_support_region",
        "D5b-8": "unsupported_by_support_region",
        "D5b-FCR-3": "unsupported_by_support_region",
        "D5b-FCR-4": "unsupported_by_support_region",
    }
)
# Measured support admission and fit/admission CPU seconds per attempt, 1000
# contiguous frozen seeds per cell, from the descriptive probe at source
# 6d67ad86 (fit and admission only; public BH/reintervaling overhead excluded).
ADMISSION_ATTEMPTS = 1000
ADMISSION_PROVENANCE = MappingProxyType(
    {
        "source_sha": "6d67ad86903548ca221d05f52cc79f2565bb78e1",
        "source_digest": "ef417a328531ba54f9f3adac0c08ea69aad5846cd9a92ad9375c8b5efb9199f3",
        "summary_sha256": "27c4737d0c847263320562404dcd062e4220572a3412ef4abc8d457283dd4dd8",
        "summary_path": ".test-evidence/cuped-linear-admission-pre-c2/summary.json",
        "command": "uv run python -m calibration.cuped_linear --campaign smoke --eligibility-attempts 1000 --output .test-evidence/cuped-linear-admission-pre-c2 --workers 2",
        "attempts_per_cell": ADMISSION_ATTEMPTS,
        "scope": "fit and support admission only, before the public route existed; excludes public BH/reintervaling overhead and projection-specific refusals",
        "streams": "measured with metric>=1 drawn from generate(rep + 10000000*metric) and D4 noise from default_rng(910000+rep); both were re-keyed to spawned SeedSequences afterwards. The laws are unchanged, so each rate still estimates the same admission probability, but the probe's exceptional records are not reproducible at the current seeds; the D5b-FCR-3/4 exclusion rests on metric 0 alone (999/1000 refused)",
        "cost": "CPU seconds per attempt measured at 6d67ad86, where the per-fit support audit cost no sizing computation; the audit now reads SUPPORT_REGION directly, so the figures carry over to the current hot path",
    }
)
ADMISSION = MappingProxyType(
    {
        "D3-1": (373, 0.002853234),
        "D3-2": (985, 0.00407022),
        "D3-3": (998, 0.005865957),
        "D3-4": (1000, 0.006016889),
        "D3b-1": (373, 0.002748545),
        "D3b-2": (373, 0.002650005),
        "D3b-3": (985, 0.004125042),
        "D3b-4": (985, 0.004140911),
        "D3b-5": (998, 0.005668277),
        "D3b-6": (998, 0.005729878),
        "D3b-7": (1000, 0.005806306),
        "D3b-8": (1000, 0.006020945),
        "D4-1": (1000, 0.017372443),
        "D4-2": (865, 0.032937579),
        "D4-3": (1000, 0.526154534),
        "D4-4": (824, 0.152043471),
        "D4-5": (1000, 0.022921941),
        "D4-6": (1000, 0.048666195),
        "D4-7": (21, 0.12367619),
        "D4-8": (1000, 0.087545184),
        "D4-9": (1000, 0.190723561),
        "D4-10": (507, 0.255909108),
        "D4-11": (865, 0.03346853),
        "D5-1": (662, 0.002244201),
        "D5-2": (1, 0.001538648),
        "D5-3": (986, 0.005263982),
        "D5-4": (740, 0.005231157),
        "D5b-1": (662, 0.002410079),
        "D5b-2": (662, 0.001874178),
        "D5b-3": (662, 0.002014568),
        "D5b-4": (662, 0.001538553),
        "D5b-5": (1, 0.001420538),
        "D5b-6": (1, 0.00150858),
        "D5b-7": (1, 0.001527164),
        "D5b-8": (1, 0.001966639),
        "D5b-9": (986, 0.005338767),
        "D5b-10": (986, 0.004505668),
        "D5b-11": (986, 0.004349983),
        "D5b-12": (986, 0.004291758),
        "D5b-13": (740, 0.004225056),
        "D5b-14": (740, 0.004587079),
        "D5b-15": (740, 0.004559773),
        "D5b-16": (740, 0.004480623),
        "D5b-FCR-1": (428, 0.00289877),
        "D5b-FCR-2": (428, 0.002907539),
        "D5b-FCR-3": (0, 0.003003137),
        "D5b-FCR-4": (0, 0.003056858),
        "D5b-FCR-5": (971, 0.009064282),
        "D5b-FCR-6": (971, 0.009064291),
        "D5b-FCR-7": (543, 0.009370609),
        "D5b-FCR-8": (543, 0.008832987),
        "D9-1": (1000, 0.001424598),
        "D9-2": (1000, 0.001439892),
        "D9-3": (1000, 0.003098384),
        "D9-4": (1000, 0.00554111),
    }
)
MANIFEST = _manifest()
GATED = tuple(cell for cell in MANIFEST if cell.gating == "gated")
PRECISION_CELL = Cell(
    "D8",
    "descriptive",
    "mean",
    "absolute",
    2000,
    failure_consequence="Precision is descriptive; no acceptance threshold.",
)


def admission_summary(identifier: str) -> dict:
    admitted, cpu_seconds = ADMISSION[identifier]
    return {
        "attempts": ADMISSION_ATTEMPTS,
        "admitted": admitted,
        "admitted_fraction": admitted / ADMISSION_ATTEMPTS,
        "confidence_interval_95": [
            coverage_lower_bound(admitted, ADMISSION_ATTEMPTS, 0.025),
            1 - coverage_lower_bound(ADMISSION_ATTEMPTS - admitted, ADMISSION_ATTEMPTS, 0.025),
        ],
        "cpu_seconds_per_attempt": cpu_seconds,
    }


def cost_projection(campaign: str) -> dict:
    """Projected fit/admission CPU-hours for one profile from the frozen probe."""
    gate = gates(campaign)
    rows = []
    hours = attempts_total = 0.0
    for cell in (*GATED, PRECISION_CELL):
        # D8 reuses the D3-4 design, so it shares that cell's measured cost.
        measured = admission_summary("D3-4" if cell.id == "D8" else cell.id)
        target = PRECISION_REPLICATIONS[campaign] if cell.id == "D8" else gate.replications
        attempts = target / measured["admitted_fraction"]
        cell_hours = measured["cpu_seconds_per_attempt"] * attempts / 3600
        hours += cell_hours
        attempts_total += attempts
        rows.append(
            {
                "cell": cell.id,
                "eligible_replications": target,
                "admitted_fraction": measured["admitted_fraction"],
                "projected_attempts": attempts,
                "cpu_seconds_per_attempt": measured["cpu_seconds_per_attempt"],
                "projected_cpu_hours": cell_hours,
            }
        )
    return {
        "profile": campaign,
        "eligible_replications": gate.replications,
        "gated_cells": len(GATED),
        "projected_cpu_hours": hours,
        "projected_attempts": attempts_total,
        "cells": rows,
        "provenance": dict(ADMISSION_PROVENANCE),
    }


def _registry_codes(module) -> frozenset[str]:
    """Every coded refusal a module registers at its top level."""
    from increment.errors import RefusalSpec

    codes = set()
    for value in vars(module).values():
        if isinstance(value, RefusalSpec):
            codes.add(value.code)
        elif isinstance(value, Mapping):
            codes.update(spec.code for spec in value.values() if isinstance(spec, RefusalSpec))
    return frozenset(codes)


def _support_region_probe() -> dict:
    """Fit one tiny design outside M1 and one inside; production must refuse only the first."""
    rng = np.random.default_rng(67)
    arms = np.array(["c"] * 15 + ["t"] * 15)
    outside = {f"x{j}": rng.normal(size=30) for j in range(8)}
    y = 10 + rng.normal(size=30)
    status = {"registered": SUPPORT_REGION_CODE in _registry_codes(production)}
    try:
        fit = fit_draw(Draw(arms, outside, dict.fromkeys(outside, "continuous"), (y,), 0.0))
    except CodedError as error:
        status["refusal"] = error.code
        violations: tuple[tuple[str, float, float], ...] = ()
        if error.code == SUPPORT_REGION_CODE:
            context_violations = error.context.get("violations")
            if isinstance(context_violations, tuple):
                violations = cast("tuple[tuple[str, float, float], ...]", context_violations)
        observed = {name: (value, bound) for name, value, bound in violations}
        status["observables_match"] = observed.get("leverage_ratio") == (
            8 / 15,
            SUPPORT_REGION["leverage_ratio"],
        )
    else:
        status["refusal"] = None
        status["observables_match"] = False
        status["returned_fit_violations"] = support_violations(
            fit, Draw(arms, outside, dict.fromkeys(outside, "continuous"), (y,), 0.0)
        )
    inside = Draw(arms, {"x0": outside["x0"]}, {"x0": "continuous"}, (y,), 0.0)
    try:
        status["inside_fits"] = not support_violations(fit_draw(inside), inside)
    except CodedError as error:
        status["inside_fits"] = False
        status["inside_refusal"] = error.code
    status["satisfied"] = bool(
        status["registered"]
        and status["refusal"] == SUPPORT_REGION_CODE
        and status["observables_match"]
        and status["inside_fits"]
    )
    return status


def _public_route_probe() -> dict:
    """Run the smallest public cuped_linear analysis; the unfinished code means no route."""
    import pandas as pd

    from increment import Analysis, AnalysisPlan, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.frame import CupedCovariate, MetricSpec

    rng = np.random.default_rng(67)
    x = rng.normal(size=40)
    frame = pd.DataFrame(
        {
            "unit": np.arange(40),
            "arm": ["c"] * 20 + ["t"] * 20,
            "x": x,
            "outcome": 10 + 2 * x + rng.normal(size=40),
        }
    )
    metric = MetricSpec(
        name="outcome",
        covariates=(CupedCovariate(name="x", kind="continuous", missing="impute-indicator"),),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="c",
        metrics=[metric],
        plan=AnalysisPlan(primary="outcome"),
    )
    try:
        rows = analysis.run()
    except CodedError as error:
        if error.code != PUBLIC_ROUTE_UNFINISHED_CODE:
            raise
        return {"refusal": error.code, "satisfied": False}
    returned = isinstance(rows, LiftEstimates) and any(row.metric == "outcome" for row in rows)
    return {"refusal": None, "satisfied": bool(returned)}


@dataclass(frozen=True, slots=True)
class Prerequisite:
    requirement: str
    probe: Callable[[], dict]


PREREQUISITES = MappingProxyType(
    {
        "support_region_refusal": Prerequisite(
            f"fit_cuped_linear refuses a fit outside M1 with {SUPPORT_REGION_CODE}, naming the harness observables {tuple(SUPPORT_REGION)} with the same bounds; the harness audits every returned fit and blocks, never filters, one outside M1",
            _support_region_probe,
        ),
        "public_route": Prerequisite(
            f"Analysis.run() with Method(name='linear', variance_reduction='cuped_linear') returns LiftEstimates rows instead of {PUBLIC_ROUTE_UNFINISHED_CODE}; the FCP families and the public D7 sentinel select and reinterval through it",
            _public_route_probe,
        ),
    }
)


def prerequisites(*names: str) -> dict:
    """Probe the named production prerequisites (all by default)."""
    return {
        name: {"requirement": PREREQUISITES[name].requirement, **PREREQUISITES[name].probe()}
        for name in (names or tuple(PREREQUISITES))
    }


def require_prerequisites(*names: str) -> dict:
    """Refuse with the coded prerequisite error unless every named prerequisite holds."""
    status = prerequisites(*names)
    missing = tuple(name for name, entry in status.items() if not entry["satisfied"])
    if missing:
        refuse(
            _REFUSALS["calibration.cuped_linear.prerequisite"],
            missing=missing,
            details={name: status[name] for name in missing},
        )
    return status


def manifest_record() -> dict:
    gated = Counter(cell.statistic for cell in GATED)
    return {
        "revision": 8,
        "family_size": FAMILY_SIZE,
        "family_size_note": "m=67 is the Revision 7 shared allocation and is retained for eta although the edge cells below no longer count toward it; the retained allocation is conservative.",
        "owned_statistics": {"coverage": 46, "fcp": 9},
        "gated_statistics": {"coverage": gated["coverage"], "fcp": gated["fcp"]},
        "excluded_statistics": "T1/T4: 12 coverage slots in the shared family",
        "eta": ETA,
        "alpha": ALPHA,
        "profiles": {
            name: {**asdict(gates(name)), "sizing": profile_sizing(name)}
            for name in ("full", "reduced")
        },
        "sizing_alternatives": {
            "detectable_coverage": 0.92,
            "detectable_fcp": 0.08,
            "rows": sizing_alternatives(),
            "note": "At the detectable .92 the nominal pass target sets R through the floor alone; every candidate floor fails .92 with probability 1.0 to double precision.",
        },
        "edge_cells": {
            identifier: {
                "gating": gating,
                "admission": admission_summary(identifier),
                "basis": "exact 95% upper admission bound below .01; the design region n/arm=100 with sigma_den=1.5 is refused by the M1 denominator-precision bound and is unsupported; no other D5b family is blocked"
                if gating == "unsupported_by_support_region"
                else "the admitted draws are exactly the balanced allocations: L<=.20 with n=2000 and p=200 needs n_t=n_c=1000, P=C(2000,1000)/2^2000=.01784, observed 21/1000 all balanced; a degenerate fixed-design subset of the declared random-allocation cell, at .124 CPU-s/attempt (about 46 reduced / 175 full CPU-hours); its L=.20 boundary is gated by D4-10 and the admission rate is the reported descriptive",
            }
            for identifier, gating in EDGE_CELLS.items()
        },
        "edge_cell_evidence": dict(ADMISSION_PROVENANCE),
        "commands": {
            "environment": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1",
            "project": "uv run python -m calibration.cuped_linear --campaign reduced --project",
            "reduced": "uv run python -m calibration.cuped_linear --campaign reduced --output .test-evidence/cuped-linear-reduced --workers 8",
            "full": "uv run python -m calibration.cuped_linear --campaign full --output .test-evidence/cuped-linear-full --workers 8",
            "smoke": "uv run python -m calibration.cuped_linear --campaign smoke --output .test-evidence/cuped-linear-smoke",
        },
        "prerequisites": {name: entry.requirement for name, entry in PREREQUISITES.items()}
        | {
            "check": "run_campaign probes both before any draw and refuses with calibration.cuped_linear.prerequisite when either is absent; --prerequisites prints the probe"
        },
        "support_region": {**SUPPORT_REGION, **SUPPORT_REQUIREMENTS},
        "support_audit": f"the harness recomputes ratio_denominator_precision from the raw draw and allows a relative excess of {DENOMINATOR_PRECISION_TOLERANCE} over the bound for rounding against production's normalized-moment value; p, k_max, leverage_ratio and max_leverage are read off the fit",
        "boundary_candidates": dict(BOUNDARY_CANDIDATES),
        "seeds": {
            "design": "SeedSequence(1000 + attempted_replication): default_rng(1000 + attempted_replication), the C.3 seed",
            "noise": "SeedSequence(1000 + attempted_replication, spawn_key=(0, 1)); C.3's literal 910000 + attempted_replication shares the design entropy space and is not used",
            "independent_metric": "SeedSequence(1000 + attempted_replication, spawn_key=(metric_index,)) for metric_index >= 1; metric 0 is the frozen C.3 seed",
            "collision_freedom": "every stream of a cell is keyed by (1000 + attempted_replication, spawn_key) with a distinct spawn_key per role, so no two draws of any profile share a stream at any chunk size",
            "chunk": "attempted_replication = 100 * chunk_size * chunk_index + offset; unused reserved indices are not reassigned",
        },
        "attempt_limit": "100 * target eligible replications per cell; exhaustion refuses the region",
        "d9_design": "1000 units/arm, fixed allocation; x ~ N(0,1); y=10+2*x+effect*T+N(0,1); ratio adds .05*den to numerator for an exact absolute ratio shift",
        "ratio_truth": "Exact gamma/lognormal population integration with the generator's clipping; denominator sigma cancels, not a noisy large-sample target",
        "interpretation_notes": [
            {
                "id": "D3b-relative-scale",
                "spec": "A.10 lines 143-150 and 205; A.5 log-mean Jacobian",
                "resolution": "The generator fixes relative truth r=0 or .05. The estimator returns a log interval, so gate log1p(r). The event {log1p(r) in log-interval} is identical to {r in the exponentiated interval}; no information is lost.",
            },
            {
                "id": "D5b-FCR-effect-case",
                "spec": "A.10 lines 171-178 and 205; C.3 lines 715-728",
                "resolution": "Effect case 1 means rate_lift=.05 in the explicit generator, an additive rate shift. The selected relative-ratio interval is compared with exp(ratio_truth)-1, not literal .05. Effect case 0 uses rate_lift=0 for both independent metrics.",
            },
            {
                "id": "D9-frozen-fixture",
                "spec": "A.10 lines 179-182 and 229",
                "resolution": "Dimensions, covariate law and noise were unspecified. The d9_design entry is frozen by the implementer, not inferred from the spec, and requires domain-review sign-off before the full run.",
                "domain_review_status": "pending",
            },
            {
                "id": "ratio-population-integration",
                "spec": "A.10 lines 205 and 225; C.3 lines 730-743",
                "resolution": "Replace the finite large-sample ratio_truth approximation with analytic truncated-gamma integration of the same population law and clipping. This removes truth-estimation error without changing the generator or estimand.",
            },
            {
                "id": "D6-all-null-policy",
                "spec": "A.6 structural-zero-variance rule; A.10 line 226",
                "resolution": "The all-null refusal fixture declares missing='refuse'. With indicator-plus-zero declared, an all-null numeric column is structurally constant and is omitted with disclosure; a separate exact test requires all rows to remain.",
            },
            {
                "id": "D4-noise-stream",
                "spec": "A.10 line 206 (noise from default_rng(910000+rep))",
                "resolution": "The literal noise seed shares the design entropy space: default_rng(910000+rep) is the design stream of attempted replication rep+909000, which the chunk layout attempts at chunk size 10. The noise columns are iid N(0,1) from the design SeedSequence spawned at (0, 1) instead: the same law, collision-free at every chunk size.",
            },
            {
                "id": "FCP-operating-claims",
                "spec": "A.10 line 131 and C.9 (binomial sizing)",
                "resolution": "Coverage pass/fail probabilities are exact binomial. A replication's FCP is fractional, so its binomial numbers are the Bernoulli benchmark; the sizing record also carries Hoeffding relative-entropy bounds valid for every [0, 1] law and the replication count at which those bounds meet the targets. The runtime KL certificate is valid either way.",
            },
        ],
        "diagnostics": {
            "D8": "4000 eligible replications under a certifying profile; variance ratios vs unadjusted and scalar x1; reported, never gated",
            "point_bias_width_leverage_dof_rejection": "Descriptive means with Monte Carlo standard errors; never gates",
        },
        "cells": [asdict(cell) for cell in MANIFEST],
    }


@dataclass(frozen=True)
class Draw:
    arms: np.ndarray
    covariates: dict
    kinds: dict
    outcomes: tuple[np.ndarray, ...]
    truth: float


NOISE_KEY = (0, 1)


def design_stream(replication: int, metric: int = 0) -> np.random.SeedSequence:
    """The frozen C.3 seed ``1000 + replication`` for a cell's first metric.

    Further independent metrics of one family are spawned from that seed by
    metric index, so no attempted replication of any profile can share a
    stream with another metric; an additive offset would wrap past 1e7 draws.
    """
    return np.random.SeedSequence(1000 + replication, spawn_key=(metric,) if metric else ())


def noise_stream(replication: int) -> np.random.SeedSequence:
    """The D4 pure-noise columns, spawned from the design seed at ``NOISE_KEY``.

    C.3's literal ``default_rng(910000 + replication)`` is the design stream of
    attempted replication ``replication + 909000``, which the chunk layout
    reaches at chunk size 10; a spawn key no metric uses cannot collide.
    """
    return np.random.SeedSequence(1000 + replication, spawn_key=NOISE_KEY)


def mean_design(cell: Cell, replication: int, metric: int = 0) -> Draw:
    """The C.3 law, preserving unavailable history as null, never dropping units."""
    rng = np.random.default_rng(design_stream(replication, metric))
    n = cell.n
    treatment = rng.random(n) < cell.allocation
    history = rng.random(n) < 0.7
    x1 = np.where(history, rng.gamma(2, 1, n), 0.0)
    x2 = np.where(history, rng.poisson(3, n).astype(float), 0.0)
    country = rng.integers(0, cell.k_country, n)
    device = rng.integers(0, 2, n)
    cmu = rng.normal(0, 1, cell.k_country) if cell.k_country != 5 else np.array([0, 1, 2, -1, 0.5])
    base = 2 * x1 + 0.5 * x2 + cmu[country] + 0.8 * device + 1.5 * history
    y = (
        base
        + (1 + 0.6 * treatment) * x1
        + (1 + 0.5 * device + 0.3 * x1) * treatment
        + rng.normal(size=n) * 1.5
    )
    truth = 2.51
    if cell.generator == "relative_mean":
        control_mean = 6.7 + float(cmu.mean())
        y += treatment * (control_mean * cell.effect - truth)
        truth = math.log1p(cell.effect)
    if cell.heterogeneous:
        y += 0.4 * treatment * x1 * ((country == 0) - 1 / cell.k_country)
    covariates = {
        "x1": np.where(history, x1, np.nan),
        "x2": np.where(history, x2, np.nan),
        "country": country.astype(str),
        "device": device.astype(str),
    }
    kinds = {
        "x1": "continuous",
        "x2": "continuous",
        "country": "categorical",
        "device": "categorical",
    }
    noise_p = cell.encoded_p - (cell.k_country + 3)
    if noise_p > 0:
        noise = np.random.default_rng(noise_stream(replication)).normal(size=(n, noise_p))
        for column in range(noise_p):
            covariates[f"noise_{column}"] = noise[:, column]
            kinds[f"noise_{column}"] = "continuous"
    return Draw(np.where(treatment, "t", "c"), covariates, kinds, (y,), truth)


def population_ratio(rate_lift: float) -> float:
    """Exact E[num]/E[den] for C.3, integrating the clipped linear rate.

    Exponential tilting changes Gamma(2,1) history to Gamma(2,.6) with
    mass multiplier .6**-2. The independent lognormal multiplier cancels.
    Truncated zeroth/first moments use incomplete gamma functions; no
    Monte Carlo truth error is added to the coverage error model.
    """
    cmu = np.array([0.0, 0.3, -0.3, 0.1, 0.2])
    a = 0.2 + 0.02 * cmu + rate_lift
    threshold = (0.99 - a) / 0.03
    history_rate = a * gammainc(2, 0.6 * threshold)
    history_rate += 0.03 * (2 / 0.6) * gammainc(3, 0.6 * threshold)
    history_rate += 0.99 * gammaincc(2, 0.6 * threshold)
    numerator = np.exp(cmu) @ (0.3 * np.clip(a, 0.01, 0.99) + 0.7 / 0.6**2 * history_rate)
    denominator = np.exp(cmu).sum() * (0.3 + 0.7 / 0.6**2)
    return float(numerator / denominator)


def ratio_design(cell: Cell, replication: int, metric: int = 0) -> Draw:
    rng = np.random.default_rng(design_stream(replication, metric))
    n = cell.n
    treatment = np.arange(n) < n // 2
    history = rng.random(n) < 0.7
    x1 = np.where(history, rng.gamma(2, 1, n), 0.0)
    country = rng.integers(0, 5, n)
    cmu = np.array([0, 0.3, -0.3, 0.1, 0.2])
    denominator = np.exp(rng.normal(0, cell.sigma_den, n) + 0.4 * x1 + cmu[country])
    rate = np.clip(0.2 + 0.03 * x1 + 0.02 * cmu[country] + cell.effect * treatment, 0.01, 0.99)
    numerator = denominator * rate + rng.normal(0, 0.1, n) * denominator
    rt, rc = population_ratio(cell.effect), population_ratio(0.0)
    truth = (
        math.log(rt) - math.log(rc) if cell.scale in ("log_ratio", "relative_ratio") else rt - rc
    )
    return Draw(
        np.where(treatment, "t", "c"),
        {"x1": np.where(history, x1, np.nan), "country": country.astype(str)},
        {"x1": "continuous", "country": "categorical"},
        (numerator, denominator),
        truth,
    )


def wls_design(cell: Cell, replication: int, metric: int = 0) -> Draw:
    rng = np.random.default_rng(design_stream(replication, metric))
    treatment = np.arange(cell.n) < cell.n // 2
    x = rng.normal(size=cell.n)
    effect = cell.effect * 10 if cell.scale == "relative" else cell.effect
    y = 10 + 2 * x + effect * treatment + rng.normal(size=cell.n)
    if cell.scale == "ratio_absolute":
        denominator = np.exp(0.1 * x + rng.normal(scale=0.2, size=cell.n))
        numerator = denominator * (0.3 + cell.effect * treatment) + rng.normal(
            scale=0.1, size=cell.n
        )
        outcomes = (numerator, denominator)
    else:
        outcomes = (y,)
    truth = math.log1p(cell.effect) if cell.scale == "relative" else cell.effect
    return Draw(np.where(treatment, "t", "c"), {"x": x}, {"x": "continuous"}, outcomes, truth)


def generate(cell: Cell, replication: int, metric: int = 0) -> Draw:
    if cell.generator in ("mean", "relative_mean"):
        return mean_design(cell, replication, metric)
    if cell.generator in ("ratio", "ratio_family"):
        return ratio_design(cell, replication, metric)
    return wls_design(cell, replication, metric)


def fit_draw(draw: Draw):
    return fit_cuped_linear(
        draw.outcomes[0] if len(draw.outcomes) == 1 else None,
        draw.arms,
        control="c",
        arm_order=("c", "t"),
        covariates=draw.covariates,
        kinds=draw.kinds,
        missing=dict.fromkeys(draw.covariates, "impute-indicator"),
        numerator=draw.outcomes[0] if len(draw.outcomes) == 2 else None,
        denominator=draw.outcomes[1] if len(draw.outcomes) == 2 else None,
    )


def support_violations(fit, draw: Draw) -> dict:
    """Audit returned evidence; never turn a missing production guard into eligibility."""
    limits = dict(SUPPORT_REGION)
    limits["ratio_denominator_precision"] *= 1 + DENOMINATOR_PRECISION_TOLERANCE
    k_max = max((len(level) for level in fit.layout.levels if level is not None), default=0)
    observables = {
        "p": fit.p,
        "k_max": k_max,
        "leverage_ratio": fit.p / fit.n_min,
        "max_leverage": fit.max_leverage,
    }
    if len(draw.outcomes) == 2:
        den = draw.outcomes[1]
        observables["ratio_denominator_precision"] = max(
            float(
                den[draw.arms == a].std(ddof=1)
                / math.sqrt(sum(draw.arms == a))
                / den[draw.arms == a].mean()
            )
            for a in ("c", "t")
        )
    return {
        key: {"value": value, "bound": SUPPORT_REGION[key]}
        for key, value in observables.items()
        if value > limits[key]
    }


def projection(fit, scale: str, alpha: float = ALPHA):
    if scale == "absolute":
        return fit.absolute("t", alpha=alpha)
    if scale == "relative":
        return fit.relative("t", alpha=alpha)
    return fit.ratio(
        "t", scale="log" if scale in ("log_ratio", "relative_ratio") else "absolute", alpha=alpha
    )


def coverage_trial(cell: Cell, replication: int) -> dict:
    draw = generate(cell, replication)
    fit = fit_draw(draw)
    violations = support_violations(fit, draw)
    if violations:
        return {
            "status": "blocked",
            "reason": "production returned a fit outside M1",
            "violations": violations,
            "replication": replication,
            "seed": 1000 + replication,
        }
    result = projection(fit, cell.scale)
    diagnostics = {
        "bias": result.estimate - draw.truth,
        "width": result.upper - result.lower,
        "max_leverage": fit.max_leverage,
        "dof": result.dof,
        "rejection": float(not result.lower <= 0 <= result.upper),
    }
    if cell.id == "D8":
        diagnostics.update(precision_diagnostics(draw, result.se**2))
    return {
        "status": "eligible",
        "covered": result.lower <= draw.truth <= result.upper,
        "diagnostics": diagnostics,
    }


def precision_diagnostics(draw: Draw, adjusted_variance: float) -> dict:
    from increment.estimation.armstats import ArmStats
    from increment.estimation.cuped import fit_cuped

    x = np.nan_to_num(draw.covariates["x1"])
    y = draw.outcomes[0]
    stats = []
    for arm in ("c", "t"):
        mask = draw.arms == arm
        xx, yy = x[mask], y[mask]
        stats.append(
            ArmStats.from_raw_sums(
                study_id="s",
                metric="mean",
                group_id=arm,
                n=int(mask.sum()),
                sum_y=float(yy.sum()),
                sum_y2=float(yy @ yy),
                sum_x=float(xx.sum()),
                sum_x2=float(xx @ xx),
                sum_xy=float(xx @ yy),
            )
        )
    scalar = fit_cuped(stats)
    unadjusted = sum(y[draw.arms == arm].var(ddof=1) / sum(draw.arms == arm) for arm in ("c", "t"))
    scalar_variance = sum(scalar.residual_var(arm, scalar.theta) / arm.n for arm in stats)
    return {
        "variance_ratio_unadjusted": adjusted_variance / unadjusted,
        "variance_ratio_scalar_x1": adjusted_variance / scalar_variance,
    }


def family_members(cell: Cell, replication: int) -> tuple[tuple[Draw, str], ...]:
    count = 2 if cell.generator == "ratio_family" else 3
    members = []
    for metric in range(count):
        if count == 2:
            member = Cell(
                "member",
                "coverage",
                "ratio",
                "relative_ratio",
                cell.n,
                encoded_p=6,
                sigma_den=cell.sigma_den,
                effect=cell.effect if metric == 0 else 0.0,
            )
        else:
            scale = ("absolute", "relative", "ratio_absolute")[metric]
            member = Cell(
                "member",
                "coverage",
                "wls",
                scale,
                cell.n,
                encoded_p=1,
                effect=0.05 if metric == 0 else 0.0,
            )
        members.append((generate(member, replication, metric), member.scale))
    return tuple(members)


def family_trial(cell: Cell, replication: int) -> dict:
    """Exercise production declared-secondary BH selection and retained reintervaling."""
    import pandas as pd

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.models import AnalysisPlan

    members = family_members(cell, replication)
    count = len(members)
    draws, scales = [], []
    for metric, (draw, scale) in enumerate(members):
        fit = fit_draw(draw)
        violations = support_violations(fit, draw)
        if violations:
            return {
                "status": "blocked",
                "reason": "production returned a fit outside M1",
                "violations": violations,
                "replication": replication,
                "metric": metric,
            }
        projection(fit, scale)
        draws.append(draw)
        scales.append(scale)
    frame = {"unit": np.arange(cell.n), "arm": draws[0].arms}
    specs = []
    for metric, draw in enumerate(draws):
        name = f"metric_{metric}"
        columns = {f"{name}_{key}": value for key, value in draw.covariates.items()}
        frame.update(columns)
        declarations = tuple(
            CupedCovariate(name=f"{name}_{key}", kind=kind, missing="impute-indicator")
            for key, kind in draw.kinds.items()
        )
        kwargs = {
            "name": name,
            "covariates": declarations,
            "decision_method": Method(name="linear", variance_reduction="cuped_linear"),
        }
        if len(draw.outcomes) == 2:
            frame[f"{name}_num"], frame[f"{name}_den"] = draw.outcomes
            kwargs.update(type="ratio", numerator=f"{name}_num", denominator=f"{name}_den")
        else:
            frame[name] = draw.outcomes[0]
        specs.append(MetricSpec.model_validate(kwargs))
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(frame),
        unit="unit",
        group="arm",
        control="c",
        metrics=specs,
        plan=AnalysisPlan(secondaries=tuple(s.name for s in specs), q=ALPHA, alpha=ALPHA),
    )
    returned = analysis.run()
    if not isinstance(returned, LiftEstimates):
        return {"status": "blocked", "reason": "production did not return fixed-horizon lift rows"}
    rows = list(returned)
    if len(rows) != count:
        return {"status": "blocked", "reason": "incomplete production family", "emitted": len(rows)}
    selected = [row for row in rows if row.discovery]
    misses = 0
    for row in selected:
        index = next(i for i, spec in enumerate(specs) if spec.name == row.metric)
        if scales[index] in ("absolute", "ratio_absolute"):
            lower, upper, truth = row.abs_lb, row.abs_ub, draws[index].truth
        else:
            lift = row.require_lift()
            lower, upper, truth = lift.lb, lift.ub, math.expm1(draws[index].truth)
        if lower is None or upper is None:
            return {
                "status": "blocked",
                "reason": "selected gated-scale interval unavailable",
                "metric": row.metric,
            }
        expected_alpha = min(ALPHA * len(selected) / count, ALPHA)
        actual_alpha = (
            row.abs_alpha
            if scales[index] in ("absolute", "ratio_absolute")
            else row.require_lift().alpha
        )
        if actual_alpha is None or not math.isclose(actual_alpha, expected_alpha, rel_tol=1e-14):
            return {
                "status": "blocked",
                "reason": "selected gated-scale interval not reissued",
                "actual_alpha": actual_alpha,
                "expected_alpha": expected_alpha,
                "metric": row.metric,
            }
        misses += not lower <= truth <= upper
    return {
        "status": "eligible",
        "fcp": misses / max(len(selected), 1),
        "selected": len(selected),
        "misses": misses,
        "diagnostics": {"selected": float(len(selected)), "fcp": misses / max(len(selected), 1)},
    }


REFUSAL_CODES = frozenset(
    "estimation.cuped_linear." + name
    for name in (
        "support_region",
        "rank_deficiency",
        "level_support",
        "residual_df",
        "leverage",
        "arm_counts",
        "nonpositive_component",
        "projection_unavailable",
    )
)


def _context_value(value):
    if isinstance(value, Mapping):
        return {key: _context_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_context_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def trial(cell: Cell, replication: int) -> dict:
    try:
        return (
            family_trial(cell, replication)
            if cell.statistic == "fcp"
            else coverage_trial(cell, replication)
        )
    except CodedError as error:
        status = "refused" if error.code in REFUSAL_CODES else "blocked"
        return {
            "status": status,
            "code": error.code,
            "context": _context_value(error.context),
            "replication": replication,
            "seed": 1000 + replication,
        }


def _atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    import scipy

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_paths = (
        sorted((root / "increment").rglob("*.py"))
        + sorted((root / "calibration").glob("*.py"))
        + [root / "tests" / "mc.py"]
    )
    source_digest = hashlib.sha256()
    for path in source_paths:
        source_digest.update(path.relative_to(root).as_posix().encode())
        source_digest.update(path.read_bytes())
    return {
        "source_sha": source_sha,
        "source_digest": source_digest.hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "threads": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
    }


def _accumulate(state: dict, result: dict) -> tuple[str, ...]:
    status = result["status"]
    state["attempted"] += 1
    state[status] += 1
    tally = [status]
    if status == "refused":
        code = result["code"]
        state["refusal_codes"][code] = state["refusal_codes"].get(code, 0) + 1
        tally.append(code)
    elif status == "eligible":
        if "covered" in result:
            state["hits"] += result["covered"]
            if result["covered"]:
                tally.append("covered")
        else:
            state["fcp_sum"] += result["fcp"]
            tally.append(f"fcp:{result['misses']}/{max(result['selected'], 1)}")
        for key, value in result["diagnostics"].items():
            moment = state["diagnostics"].setdefault(key, [0.0, 0.0])
            moment[0] += value
            moment[1] += value * value
    elif status == "blocked":
        state["concern"] = result
    return tuple(tally)


def run_chunk(job: tuple) -> dict:
    cell, directory, chunk, target, chunk_size, identity = job
    directory = Path(directory) / cell.id / "chunks" / f"{chunk:06d}"
    checkpoint = directory / "checkpoint.json"
    binding = {
        "identity": identity,
        "cell": asdict(cell),
        "chunk": chunk,
        "target": target,
        "chunk_size": chunk_size,
    }
    binding = json.loads(json.dumps(binding))
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        totals = verify(directory, case_id=f"{cell.id}:{chunk}")
        if record["binding"] != binding or record["journal_digest"] != totals.digest:
            raise ValueError(f"incompatible checkpoint: {checkpoint}; use a new output directory")
        return record
    started = time.perf_counter()
    cpu_started = time.process_time()
    state = {
        "attempted": 0,
        "eligible": 0,
        "refused": 0,
        "blocked": 0,
        "hits": 0,
        "fcp_sum": 0.0,
        "refusal_codes": {},
        "diagnostics": {},
    }
    start = chunk * chunk_size * 100
    with DrawJournal(directory, case_id=f"{cell.id}:{chunk}", block_size=chunk_size) as journal:
        committed = journal.totals().completed
        replay_tallies = Counter()
        for offset in range(chunk_size * 100):
            replication = start + offset
            if offset >= committed:
                journal.started(replication)
            try:
                result = trial(cell, replication)
            except Exception as error:
                if offset >= committed:
                    journal.completed(
                        replication,
                        {"exception": type(error).__name__, "message": str(error)},
                        tally=("failed",),
                        exceptional=True,
                    )
                raise
            tally = _accumulate(state, result)
            if offset < committed:
                replay_tallies.update(tally)
                if offset + 1 == committed and dict(replay_tallies) != dict(
                    journal.totals().counters
                ):
                    raise ValueError("replayed draws disagree with the committed journal")
            else:
                journal.completed(
                    replication, result, tally=tally, exceptional=result["status"] != "eligible"
                )
            if state["eligible"] == target or state["blocked"]:
                break
        totals = journal.close()
    state.update(
        binding=binding,
        journal_digest=totals.digest,
        elapsed_seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu_started,
        status="blocked"
        if state["blocked"]
        else "complete"
        if state["eligible"] == target
        else "refused_region",
    )
    _atomic_json(checkpoint, state)
    return state


def summarize(cell: Cell, records: list[dict], target: int, campaign: str) -> dict:
    eligible = sum(record["eligible"] for record in records)
    attempted = sum(record["attempted"] for record in records)
    elapsed = sum(record["elapsed_seconds"] for record in records)
    cpu = sum(record["cpu_seconds"] for record in records)
    status = (
        "complete"
        if eligible == target
        else "blocked"
        if any(r["blocked"] for r in records)
        else "refused_region"
    )
    diagnostics = {}
    keys = {key for record in records for key in record["diagnostics"]}
    for key in keys:
        total = sum(record["diagnostics"].get(key, [0, 0])[0] for record in records)
        squares = sum(record["diagnostics"].get(key, [0, 0])[1] for record in records)
        mean = total / eligible
        se = (
            math.sqrt(max(0, (squares - total * mean) / (eligible - 1) / eligible))
            if eligible > 1
            else None
        )
        diagnostics[key] = {"mean": mean, "mcse": se, "gated": False}
        if key == "rejection":
            hits = int(round(total))
            diagnostics[key]["confidence_interval_95"] = (
                coverage_lower_bound(hits, eligible, 0.025),
                1 - coverage_lower_bound(eligible - hits, eligible, 0.025),
            )
        elif key == "fcp":
            diagnostics[key]["confidence_interval_95"] = (
                1 - kl_chernoff_upper_bound(1 - mean, eligible, 0.025),
                kl_chernoff_upper_bound(mean, eligible, 0.025),
            )
    bound = observed = diagnostic_mcse = None
    if eligible and cell.statistic == "coverage":
        hits = sum(record["hits"] for record in records)
        observed = hits / eligible
        bound = coverage_lower_bound(hits, eligible, ETA)
        diagnostic_mcse = mcse(observed, eligible)
    elif eligible and cell.statistic == "fcp":
        observed = sum(record["fcp_sum"] for record in records) / eligible
        bound = kl_chernoff_upper_bound(observed, eligible, ETA)
        diagnostic_mcse = diagnostics["fcp"]["mcse"]
    gate = gates(campaign)
    passed = bound is not None and (
        bound >= gate.coverage_floor if cell.statistic == "coverage" else bound <= gate.fcp_ceiling
    )
    evaluated = campaign != "smoke" and status == "complete" and cell.statistic != "descriptive"
    certified = evaluated and passed
    measured = (
        {
            name: cpu
            / eligible
            * (PRECISION_REPLICATIONS[name] if cell.id == "D8" else count)
            / 3600
            for name, count in REPLICATIONS.items()
            if name != "smoke"
        }
        if status == "complete"
        else None
    )
    return {
        "cell": cell.id,
        "statistic": cell.statistic,
        "status": status,
        "eligible": eligible,
        "refused": sum(r["refused"] for r in records),
        "attempted": attempted,
        "observed": observed,
        "mcse": diagnostic_mcse,
        "bound": bound,
        "gate_passed": passed if evaluated else None,
        "certified": certified,
        "elapsed_seconds": elapsed,
        "seconds_per_attempt": elapsed / attempted if attempted else None,
        "cpu_seconds": cpu,
        "cpu_seconds_per_attempt": cpu / attempted if attempted else None,
        "cpu_seconds_per_eligible": cpu / eligible if eligible else None,
        "seconds_per_eligible": elapsed / eligible if eligible else None,
        "projected_cpu_hours": measured,
        "diagnostics": diagnostics,
        "refusal_codes": dict(sum((Counter(r["refusal_codes"]) for r in records), Counter())),
        "concerns": [r["concern"] for r in records if "concern" in r],
    }


def run_campaign(output: Path, campaign: str, workers: int = 1, chunk_size: int = 100) -> dict:
    if campaign not in PROFILES or workers < 1 or chunk_size < 1:
        raise ValueError("unknown campaign or nonpositive workers/chunk size")
    # Before any draw or output: a missing production guard would otherwise
    # surface only as the first blocked cell.
    satisfied = require_prerequisites()
    target = REPLICATIONS[campaign]
    chunk_size = min(chunk_size, target)
    identity = _identity()
    frozen = manifest_record()
    gate = gates(campaign)
    binding = {
        "manifest": frozen,
        "identity": identity,
        "campaign": campaign,
        "chunk_size": chunk_size,
        "profile": {
            **asdict(gate),
            "prerequisites": satisfied,
            "certification": {cell.id: campaign != "smoke" for cell in GATED},
            "excluded": {cell.id: cell.gating for cell in MANIFEST if cell.gating != "gated"},
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
        json.dumps(binding)
    ):
        raise ValueError("output belongs to a different frozen source/profile; use a new directory")
    _atomic_json(manifest_path, binding)
    projected = cost_projection(campaign)
    print(json.dumps({"projection": projected}, sort_keys=True), flush=True)
    cells = (*GATED, PRECISION_CELL)
    targets = {cell.id: target for cell in GATED}
    targets["D8"] = PRECISION_REPLICATIONS[campaign]
    jobs = [
        (
            cell,
            str(output),
            chunk,
            min(chunk_size, targets[cell.id] - chunk * chunk_size),
            chunk_size,
            identity,
        )
        for cell in cells
        for chunk in range(math.ceil(targets[cell.id] / chunk_size))
    ]
    started = time.perf_counter()
    if workers == 1:
        records = [run_chunk(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(run_chunk, jobs))
    summaries = []
    for cell in cells:
        summary = summarize(
            cell,
            [r for r in records if r["binding"]["cell"]["id"] == cell.id],
            targets[cell.id],
            campaign,
        )
        summaries.append(summary)
        _atomic_json(output / cell.id / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    certifying = campaign != "smoke"
    result = {
        "campaign": campaign,
        "identity": identity,
        "elapsed_seconds": time.perf_counter() - started,
        "cells": summaries[:-1],
        "precision": summaries[-1],
        "excluded": binding["profile"]["excluded"],
        "full_campaign_executed": certifying,
        "projection": projected,
        "measured_projection_cpu_hours": {
            name: sum((s["projected_cpu_hours"] or {}).get(name, 0) for s in summaries)
            for name in REPLICATIONS
            if name != "smoke"
        },
        "measured_projection_complete": all(
            s["projected_cpu_hours"] is not None for s in summaries
        ),
        "status": "blocked"
        if any(s["status"] != "complete" for s in summaries)
        else "failed"
        if certifying and any(not s["gate_passed"] for s in summaries[:-1])
        else "smoke_complete"
        if campaign == "smoke"
        else "certified",
    }
    _atomic_json(output / "summary.json", result)
    return result


def admission_trial(cell: Cell, replication: int) -> dict:
    """Estimate support admission only; this cannot replace production refusals."""
    draws = (
        family_members(cell, replication)
        if cell.statistic == "fcp"
        else ((generate(cell, replication), cell.scale),)
    )
    violations = {}
    for metric, (draw, _) in enumerate(draws):
        try:
            fit = fit_draw(draw)
            for name, values in support_violations(fit, draw).items():
                violations[f"metric_{metric}:{name}"] = values
        except CodedError as error:
            if error.code not in REFUSAL_CODES:
                raise
            violations[f"metric_{metric}:{error.code}"] = _context_value(error.context)
    return {"admitted": not violations, "violations": violations, "seed": 1000 + replication}


def run_admission_cell(job: tuple) -> dict:
    cell, output, attempts, identity = job
    directory = Path(output) / cell.id
    checkpoint = directory / "checkpoint.json"
    binding = json.loads(
        json.dumps({"cell": asdict(cell), "attempts": attempts, "identity": identity})
    )
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        totals = verify(directory, case_id=cell.id)
        if record["binding"] != binding or record["journal_digest"] != totals.digest:
            raise ValueError(
                "incompatible support-admission checkpoint; use a new output directory"
            )
        return record
    counters = Counter()
    started, cpu_started = time.perf_counter(), time.process_time()
    with DrawJournal(directory, case_id=cell.id, block_size=100) as journal:
        committed = journal.totals().completed
        for replication in range(attempts):
            if replication >= committed:
                journal.started(replication)
            result = admission_trial(cell, replication)
            tally = (
                "admitted" if result["admitted"] else "not_admitted",
                *sorted(result["violations"]),
            )
            counters.update(tally)
            if replication >= committed:
                journal.completed(
                    replication, result, tally=tally, exceptional=not result["admitted"]
                )
            elif replication + 1 == committed and dict(counters) != dict(journal.totals().counters):
                raise ValueError("support-admission replay disagrees with its journal")
        totals = journal.close()
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    admitted = counters["admitted"]
    fraction = admitted / attempts
    lower = coverage_lower_bound(admitted, attempts, 0.025)
    upper = 1 - coverage_lower_bound(attempts - admitted, attempts, 0.025)
    cpu_per_attempt = cpu / attempts
    profiles = {name: count for name, count in REPLICATIONS.items() if name != "smoke"}
    result = {
        "cell": cell.id,
        "attempts": attempts,
        "admitted": admitted,
        "admitted_fraction": fraction,
        "confidence_interval_95": [lower, upper],
        "projected_attempts": {
            name: count / fraction if fraction else None for name, count in profiles.items()
        },
        "projected_attempts_interval_95": {
            name: [count / upper if upper else None, count / lower if lower else None]
            for name, count in profiles.items()
        },
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu,
        "cpu_seconds_per_attempt": cpu_per_attempt,
        "projected_cpu_hours": {
            name: cpu_per_attempt * count / fraction / 3600 if fraction else None
            for name, count in profiles.items()
        },
        "projected_cpu_hours_interval_95": {
            name: [
                cpu_per_attempt * count / upper / 3600 if upper else None,
                cpu_per_attempt * count / lower / 3600 if lower else None,
            ]
            for name, count in profiles.items()
        },
        "over_ten_attempts_per_eligible": fraction < 0.1,
        "violations": {
            key: value
            for key, value in sorted(counters.items())
            if key not in ("admitted", "not_admitted")
        },
        "gated": False,
        "cost_scope": "fit and support admission only; excludes public BH/reintervaling overhead",
        "binding": binding,
        "journal_digest": totals.digest,
    }
    _atomic_json(checkpoint, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run_admission_probe(output: Path, attempts: int, workers: int) -> dict:
    if attempts < 1 or workers < 1:
        raise ValueError("attempts and workers must be positive")
    identity = _identity()
    binding = {
        "identity": identity,
        "manifest": manifest_record(),
        "attempts": attempts,
        "scope": "descriptive support-admission probe, not certification",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(binding)):
        raise ValueError("support-admission output belongs to another source; use a new directory")
    _atomic_json(path, binding)
    jobs = [(cell, str(output), attempts, identity) for cell in MANIFEST]
    started = time.perf_counter()
    if workers == 1:
        records = [run_admission_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(run_admission_cell, jobs))
    result = {
        "scope": binding["scope"],
        "identity": identity,
        "cells": records,
        "elapsed_seconds": time.perf_counter() - started,
        "projected_cpu_hours": {
            name: sum(record["projected_cpu_hours"][name] or 0 for record in records)
            for name in REPLICATIONS
            if name != "smoke"
        },
        "projection_complete": all(
            value is not None
            for record in records
            for value in record["projected_cpu_hours"].values()
        ),
        "over_ten_attempts_cells": [
            record["cell"] for record in records if record["over_ten_attempts_per_eligible"]
        ],
        "full_campaign_executed": False,
    }
    _atomic_json(output / "summary.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--project",
        action="store_true",
        help="print the profile's projected CPU-hours from the frozen measured costs and exit",
    )
    parser.add_argument(
        "--prerequisites",
        action="store_true",
        help="probe the production prerequisites, print the result and exit 1 if any is absent",
    )
    parser.add_argument(
        "--eligibility-attempts",
        type=int,
        default=0,
        help="separate descriptive support-admission probe with contiguous frozen seeds; no gates",
    )
    args = parser.parse_args(argv)
    if args.project:
        print(json.dumps(cost_projection(args.campaign), sort_keys=True, indent=2))
        return 0
    if args.prerequisites:
        status = prerequisites()
        print(json.dumps(status, sort_keys=True, indent=2, default=str))
        return 0 if all(entry["satisfied"] for entry in status.values()) else 1
    if args.output is None:
        parser.error("--output is required unless --project or --prerequisites is given")
    if args.eligibility_attempts:
        if args.campaign != "smoke":
            parser.error(
                "the descriptive eligibility probe is separate from the certifying profiles"
            )
        run_admission_probe(args.output.resolve(), args.eligibility_attempts, args.workers)
        return 0
    try:
        result = run_campaign(args.output.resolve(), args.campaign, args.workers, args.chunk_size)
    except PrerequisiteError as error:
        refusal = {"code": error.code, "message": str(error), **_context_value(error.context)}
        print(json.dumps(refusal, sort_keys=True, indent=2, default=str), file=sys.stderr)
        return 1
    return 0 if result["status"] in ("smoke_complete", "certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
