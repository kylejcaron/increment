"""Freeze construction-A, HC2 and Bell–McCaffrey reference fixtures.

No increment import is used. For full-rank centred covariates and positive
residual df, solve G theta = sum_a Cov(X,Y)_a/n_a. The projection score includes
the fitted slope and the random pooled anchor. HC2 divides only its residual
term by sqrt(1-h). The t df is tr(M)^2/tr(M^2), evaluated through small traces.
Ratio scores are summed before squaring; their two independent-component BM
working-model trace pairs are added before division. Positive adjusted means
are required for logarithms. These are floating-point reference calculations,
not exact rational arithmetic; fixtures bound the scaled design condition.

Regenerate with ``uv run python -m tests.oracles.generate.gen_cuped_linear_oracle``.
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t as student_t


def bm_traces(design, weights, leverage, contrast):
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    coefficient = weights * (design @ inverse @ contrast)
    delta = coefficient / np.sqrt(1 - leverage)
    scale = np.max(np.abs(delta))
    if scale == 0:
        raise ValueError("zero BM contrast")
    squared = (delta / scale) ** 2
    wx = weights[:, None] * design
    u = np.column_stack([wx, squared[:, None] * design])
    p = design.shape[1]
    block = np.zeros((2 * p, 2 * p))
    block[:p, :p] = inverse @ (design.T @ (squared[:, None] * design)) @ inverse
    block[:p, p:] = -inverse
    block[p:, :p] = -inverse
    cross = u.T @ u
    first = squared.sum() + np.trace(block @ cross)
    second = squared @ squared + 2 * np.trace(block @ (u.T @ (squared[:, None] * u)))
    second += np.trace(block @ cross @ block @ cross)
    # Return traces on their original common scale so stacking preserves weights.
    return float(first * scale**2), float(second * scale**4)


def reference(x, arms, outcomes, scale, alpha=0.05):
    groups = [np.flatnonzero(arms == label) for label in ("c", "t")]
    counts = np.array([len(index) for index in groups])
    shares = counts / counts.sum()
    means_x = np.array([x[index].mean(axis=0) for index in groups])
    anchor = shares @ means_x
    centered = np.empty_like(x)
    weights = np.empty(len(x))
    design = np.zeros((len(x), 2 + x.shape[1]))
    for a, index in enumerate(groups):
        centered[index] = x[index] - means_x[a]
        weights[index] = 1 / (counts[a] * (counts[a] - 1))
        design[index, a] = 1
    design[:, 2:] = centered
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    leverage = weights * np.einsum("ij,jk,ik->i", design, inverse, design)
    gram = centered.T @ (weights[:, None] * centered)
    slopes, residuals, adjusted = [], [], []
    for outcome in outcomes:
        means_y = np.array([outcome[index].mean() for index in groups])
        yc = np.empty(len(x))
        for a, index in enumerate(groups):
            yc[index] = outcome[index] - means_y[a]
        theta = np.linalg.solve(gram, centered.T @ (weights * yc))
        slopes.append(theta)
        residuals.append(yc - centered @ theta)
        adjusted.append(means_y - (means_x - anchor) @ theta)
    means = np.asarray(adjusted)
    if scale == "absolute":
        point = means[0, 1] - means[0, 0]
        jacobian = np.array([[-1.0, 1.0]])
    elif scale == "relative":
        point = np.log(means[0, 1]) - np.log(means[0, 0])
        jacobian = np.array([[-1 / means[0, 0], 1 / means[0, 1]]])
    else:
        rates = means[0] / means[1]
        if scale == "log_ratio":
            point = np.log(rates[1]) - np.log(rates[0])
            jacobian = np.array(
                [[-1 / means[0, 0], 1 / means[0, 1]], [1 / means[1, 0], -1 / means[1, 1]]]
            )
        elif scale == "ratio_absolute":
            point = rates[1] - rates[0]
            jacobian = np.array(
                [
                    [-1 / means[1, 0], 1 / means[1, 1]],
                    [rates[0] / means[1, 0], -rates[1] / means[1, 1]],
                ]
            )
        else:
            raise ValueError(scale)
    score = np.zeros(len(x))
    contrasts = []
    for theta, residual, j in zip(slopes, residuals, jacobian, strict=True):
        d = j @ (means_x - anchor)
        contrast = np.r_[j, -d]
        contrasts.append(contrast)
        coefficient = weights * (design @ inverse @ contrast)
        for a, index in enumerate(groups):
            anchor_score = ((j.sum() * shares[a] * theta) @ centered[index].T) / counts[a]
            score[index] += (
                coefficient[index] * residual[index] / np.sqrt(1 - leverage[index]) + anchor_score
            )
    # A common coefficient scale avoids lost relative magnitude when stacking.
    common_scale = max(float(np.max(np.abs(c))) for c in contrasts)
    traces = [bm_traces(design, weights, leverage, c / common_scale) for c in contrasts]
    first = sum(pair[0] for pair in traces)
    second = sum(pair[1] for pair in traces)
    dof = first * first / second
    se = np.linalg.norm(score)
    half_width = student_t.isf(alpha / 2, dof) * se
    normalized = design * np.sqrt(weights[:, None])
    normalized /= np.linalg.norm(normalized, axis=0)
    return {
        "estimate": float(point),
        "se": float(se),
        "dof": float(dof),
        "lower": float(point - half_width),
        "upper": float(point + half_width),
        "theta": np.asarray(slopes).ravel().tolist(),
        "anchor": anchor.tolist(),
        "adjusted_means": means.ravel().tolist(),
        "leverage_trace": float(leverage.sum()),
        "condition_number": float(np.linalg.cond(normalized)),
    }


def fixture_inputs():
    rng = np.random.default_rng(20261012)
    data = {}
    for name, nc, nt, kind in (
        ("p0", 37, 61, "none"),
        ("categorical", 60, 80, "category"),
        ("missing_history", 70, 90, "missing"),
        ("unequal_allocation", 50, 250, "numeric"),
        ("ratio", 120, 160, "numeric"),
    ):
        arms = np.array(["c"] * nc + ["t"] * nt)
        n = len(arms)
        covariates, kinds = {}, {}
        if kind == "none":
            x = np.empty((n, 0))
        elif kind == "category":
            category = np.resize(np.array(["a", "b", "c"]), n)
            covariates = {"country": category.tolist()}
            kinds = {"country": "categorical"}
            x = np.column_stack([category == "b", category == "c"]).astype(float)
        elif kind == "missing":
            values = rng.normal(size=(n, 2))
            null = np.arange(n) % 4 == 0
            values[null] = 0
            x = np.column_stack([values, null.astype(float)])
            covariates = {
                f"x{k}": [None if null[i] else float(values[i, k]) for i in range(n)]
                for k in range(2)
            }
            kinds = dict.fromkeys(covariates, "continuous")
        else:
            x = rng.normal(size=(n, 2))
            covariates = {f"x{k}": x[:, k].tolist() for k in range(2)}
            kinds = dict.fromkeys(covariates, "continuous")
        y = 10 + 0.4 * (arms == "t") + x @ np.full(x.shape[1], 0.7) + rng.normal(size=n)
        outcomes = [y.tolist()]
        if name == "ratio":
            den = np.exp(0.2 * x[:, 0] + rng.normal(scale=0.25, size=n))
            num = den * (0.4 + 0.05 * (arms == "t") + 0.02 * x[:, 1]) + rng.normal(
                scale=0.05, size=n
            )
            outcomes = [num.tolist(), den.tolist()]
        data[name] = {
            "arms": arms.tolist(),
            "encoded_x": x.tolist(),
            "outcomes": outcomes,
            "covariates": covariates,
            "kinds": kinds,
        }
    return data


def main():
    root = Path(__file__).resolve().parents[1]
    data = fixture_inputs()
    cases = []
    for name, inp in data.items():
        scales = ("log_ratio", "ratio_absolute") if name == "ratio" else ("absolute", "relative")
        for scale in scales:
            result = reference(
                np.asarray(inp["encoded_x"]),
                np.asarray(inp["arms"]),
                np.asarray(inp["outcomes"]),
                scale,
            )
            cases.append({"id": name, "scale": scale, **result})
    manifest = {
        "generator": "tests/oracles/generate/gen_cuped_linear_oracle.py",
        "generated_on": datetime.date.today().isoformat(),
        "increment_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root.parents[1], text=True
        ).strip(),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "estimand": "Construction A adjusted contrasts; stacked HC2 and plug-in low-rank BM t intervals",
        "tolerance": {
            "rel": 2e-10,
            "abs": 1e-11,
            "maximum_condition_number": 10,
            "note": "Scaled weighted design condition below 10; inverse/sandwich arithmetic amplifies roundoff by condition squared. 2e-10 covers small-trace cancellation without admitting a statistical approximation.",
        },
        "cases": cases,
    }
    (root / "data" / "cuped_linear.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (root / "fixtures" / "cuped_linear.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "fixtures": len(cases),
                "maximum_condition_number": max(c["condition_number"] for c in cases),
            }
        )
    )


if __name__ == "__main__":
    main()
