"""Preregistered mixed-covariate linear CUPED certification (Revision 7).

Run ``uv run python -m calibration.cuped_linear --campaign smoke --output DIR``.
The full profile uses 107,100 eligible draws per statistic. Smoke exercises the
same designs, but cannot certify them. Coverage uses an exact binomial lower
bound; mean false-coverage proportion uses a bounded-observation KL upper bound.
Both spend the shared 67-statistic allocation, including the 12 sitewide slots
owned by the separate campaign. A failed gate never licenses a Normal interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from types import MappingProxyType

from calibration.profile import CalibrationProfile, CellSet
from tests.mc import family_eta

FAMILY_SIZE = 67
ETA = family_eta(0.01, FAMILY_SIZE)
FULL_REPLICATIONS = 107_100
SMOKE_REPLICATIONS = 3
ALPHA = 0.05
COVERAGE_FLOOR = 0.945
FCP_CEILING = 0.055
BOUNDARY_CANDIDATES = MappingProxyType(
    {
        "p": (203, 53, 8),
        "k_max": (200, 50, 5),
        "leverage_ratio": (0.20, 0.10, 0.05),
        "max_leverage": (0.45, 0.30, 0.15),
        "minimum_level_arm_count": (2, 5, 10),
    }
)
PROFILES = MappingProxyType(
    {
        name: CalibrationProfile(
            name=name,
            campaign="cuped_linear",
            tolerance="strict",
            delta=0.005,
            tolerance_fraction=Fraction(1, 200),
            cells=CellSet("all", "every-case", (), id_field="id", axis_field=None),
            stopping="fixed",
            sequential_rule=None,
        )
        for name in ("smoke", "full")
    }
)


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
    return tuple(rows)


MANIFEST = _manifest()


def manifest_record() -> dict:
    return {
        "revision": 7,
        "family_size": FAMILY_SIZE,
        "owned_statistics": {"coverage": 46, "fcp": 9},
        "excluded_statistics": "T1/T4: 12 coverage slots in the shared family",
        "eta": ETA,
        "eligible_replications": FULL_REPLICATIONS,
        "alpha": ALPHA,
        "coverage_floor": COVERAGE_FLOOR,
        "fcp_ceiling": FCP_CEILING,
        "support_region": {
            "p": 203,
            "k_max": 200,
            "leverage_ratio": 0.20,
            "max_leverage": 0.45,
            "ratio_denominator_precision": 0.15,
            "minimum_level_arm_count": 2,
            "full_rank": True,
            "positive_residual_df": True,
        },
        "boundary_candidates": dict(BOUNDARY_CANDIDATES),
        "seeds": {
            "design": "1000 + attempted_replication",
            "noise": "910000 + attempted_replication",
            "independent_metric": "1000 + attempted_replication + 10000000 * metric_index",
        },
        "attempt_limit": "100 * target eligible replications per cell; exhaustion refuses the region",
        "d9_design": "1000 units/arm, fixed allocation; x ~ N(0,1); y=10+2*x+effect*T+N(0,1); ratio adds .05*den to numerator for an exact absolute ratio shift",
        "ratio_truth": "Exact gamma/lognormal population integration with the generator's clipping; denominator sigma cancels, not a noisy large-sample target",
        "diagnostics": {
            "D8": "4000 eligible replications; variance ratios vs unadjusted and scalar x1; reported, never gated",
            "point_bias_width_leverage_dof_rejection": "Descriptive means with Monte Carlo standard errors; never gates",
        },
        "cells": [asdict(cell) for cell in MANIFEST],
    }
