"""Generate the statsmodels oracle for increment's overlapping-experiments contrasts.

Each case is a small explicit set of per-unit outcomes in the four joint
assignment cells of two binary experiments. The oracle fits statsmodels OLS on
the saturated design ``[1, A, B, A*B]`` with the HC2 sandwich and evaluates each
contrast ``L @ beta`` by ``t_test``, where ``L`` maps the cell-mean weights onto
the regression coefficients. statsmodels shares no code with increment's
cell-means estimator; for a saturated design the HC2 sandwich equals the sum of
per-cell ``w^2 s^2 / n`` that increment computes. Degrees of freedom are not
compared: statsmodels reports the residual df, not Welch-Satterthwaite.

Usage (from the repository root; statsmodels is an oracle-generation-only tool):
    uv run --with statsmodels python tests/oracles/generate/gen_overlap_oracle.py
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import statsmodels.api as sm  # ty: ignore[unresolved-import]  - oracle-generation-only, not a project dependency

CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))
TO_BETA = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)


def weights(contrast: str, p1: float, p2: float) -> list[float]:
    fixed = {
        "interaction": [1, -1, -1, 1],
        "joint_launch": [-1, 0, 0, 1],
        "first_only_launch": [-1, 1, 0, 0],
        "second_only_launch": [-1, 0, 1, 0],
        "first_given_second_treatment": [0, 0, -1, 1],
        "second_given_first_treatment": [0, -1, 0, 1],
        "marginal_first": [-(1 - p2), 1 - p2, -p2, p2],
        "marginal_second": [-(1 - p1), -p1, 1 - p1, p1],
    }
    return [float(w) for w in fixed[contrast]]


def case(case_id: str, rng: np.random.Generator, sizes, means, sds, p1, p2, offset=0.0) -> dict:
    cells = {
        f"{a}{b}": [
            float(round(offset + means[i] + sds[i] * z, 6)) for z in rng.standard_normal(sizes[i])
        ]
        for i, (a, b) in enumerate(CELLS)
    }
    a = np.concatenate([[c[0]] * sizes[i] for i, c in enumerate(CELLS)]).astype(float)
    b = np.concatenate([[c[1]] * sizes[i] for i, c in enumerate(CELLS)]).astype(float)
    y = np.concatenate([cells[f"{c[0]}{c[1]}"] for c in CELLS])
    x = np.column_stack([np.ones_like(a), a, b, a * b])
    fit = sm.OLS(y, x).fit(cov_type="HC2")
    expected = {}
    for contrast in (
        "interaction",
        "joint_launch",
        "first_only_launch",
        "second_only_launch",
        "first_given_second_treatment",
        "second_given_first_treatment",
        "marginal_first",
        "marginal_second",
    ):
        lmat = np.asarray(weights(contrast, p1, p2)) @ TO_BETA
        test = fit.t_test(lmat[None, :])
        expected[contrast] = {
            "estimate": float(np.asarray(test.effect).ravel()[0]),
            "se": float(np.asarray(test.sd).ravel()[0]),
        }
    return {
        "id": case_id,
        "first_probability": p1,
        "second_probability": p2,
        "cells": cells,
        "expected": expected,
    }


def main() -> None:
    rng = np.random.default_rng(20261012)
    cases = [
        case("balanced", rng, (25, 25, 25, 25), (1.0, 1.4, 1.2, 2.3), (1, 1, 1, 1), 0.5, 0.5),
        case(
            "unequal_heteroskedastic",
            rng,
            (60, 20, 12, 5),
            (0.0, 0.5, -0.3, 1.0),
            (1, 3, 0.5, 2),
            0.25,
            0.2,
        ),
        case("sign_reversal", rng, (30, 30, 10, 10), (5.0, 6.0, 5.5, 4.0), (2, 2, 2, 2), 0.5, 0.25),
        case(
            "large_offset",
            rng,
            (15, 15, 15, 15),
            (0.0, 0.1, 0.2, 0.5),
            (1, 1, 1, 1),
            0.5,
            0.5,
            offset=1e6,
        ),
    ]
    out = {
        "generator": "tests/oracles/generate/gen_overlap_oracle.py",
        "package_versions": {"numpy": version("numpy"), "statsmodels": version("statsmodels")},
        "tolerance": {
            "rel": 1e-9,
            "note": "Outcomes are rounded to 6 decimals before both fits; the two implementations agree to rounding of float64 sums, chosen at 1e-9 relative with abs=0. The large_offset case adds a 1e6 common offset, which bounds attainable relative agreement of near-zero contrasts, so it uses an absolute tolerance of 1e-7 instead.",
            "large_offset_abs": 1e-7,
        },
        "cases": cases,
    }
    path = Path(__file__).parents[1] / "fixtures" / "overlap_hc2.json"
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
