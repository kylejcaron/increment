"""Construction A, stacked HC2 and plug-in BM against frozen independent arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from increment.estimation.cuped import fit_cuped_multicovariate
from tests.oracles.generate.gen_cuped_linear_oracle import bm_traces, reference

_ROOT = Path(__file__).parent


def test_low_rank_trace_identity_matches_dense_residual_operator():
    rng = np.random.default_rng(71)
    design = np.column_stack([np.ones(60), rng.normal(size=(60, 3))])
    weights = np.r_[np.full(20, 1 / (20 * 19)), np.full(40, 1 / (40 * 39))]
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    hat = design @ inverse @ (weights[:, None] * design).T
    leverage = np.diag(hat)
    contrast = np.array([1.0, -0.3, 0.7, 0.2])
    coefficients = weights * (design @ inverse @ contrast)
    first, second = bm_traces(design, weights, leverage, contrast)
    delta = coefficients / np.sqrt(1 - leverage)
    residual = (np.eye(len(weights)) - hat).T * delta[None, :]
    matrix = residual @ residual.T
    assert first == pytest.approx(np.trace(matrix), rel=1e-12)
    assert second == pytest.approx(np.sum(matrix * matrix), rel=1e-12)


def test_frozen_fixtures_agree_with_independent_construction():
    manifest = json.loads((_ROOT / "fixtures" / "cuped_linear.json").read_text())
    data = json.loads((_ROOT / "data" / "cuped_linear.json").read_text())
    tolerance = manifest["tolerance"]
    for case in manifest["cases"]:
        inp = data[case["id"]]
        x = np.asarray(inp["encoded_x"], dtype=float)
        arms = np.asarray(inp["arms"])
        outcomes = np.asarray(inp["outcomes"], dtype=float)
        expected = reference(x, arms, outcomes, case["scale"])
        assert expected["condition_number"] < tolerance["maximum_condition_number"]
        for field in ("estimate", "se", "dof", "lower", "upper"):
            assert expected[field] == pytest.approx(
                case[field], rel=tolerance["rel"], abs=tolerance["abs"]
            ), (case["id"], field)
        fit = fit_cuped_multicovariate(
            outcomes[0] if len(outcomes) == 1 else None,
            arms,
            control="c",
            arm_order=("c", "t"),
            covariates=inp["covariates"],
            kinds=inp["kinds"],
            missing=dict.fromkeys(inp["covariates"], "impute-indicator"),
            numerator=outcomes[0] if len(outcomes) == 2 else None,
            denominator=outcomes[1] if len(outcomes) == 2 else None,
        )
        projection = (
            fit.absolute("t")
            if case["scale"] == "absolute"
            else fit.relative("t")
            if case["scale"] == "relative"
            else fit.ratio("t", scale="log" if case["scale"] == "log_ratio" else "absolute")
        )
        for field in ("estimate", "se", "dof", "lower", "upper"):
            assert getattr(projection, field) == pytest.approx(
                case[field], rel=tolerance["rel"], abs=tolerance["abs"]
            ), (case["id"], field)
        assert fit.adjusted_means == pytest.approx(expected["adjusted_means"], abs=1e-11)
        assert fit.leverage_trace == pytest.approx(2 + x.shape[1], abs=1e-11)
        if x.shape[1] == 0 and case["scale"] == "absolute":
            welch = sum(outcomes[0, arms == a].var(ddof=1) / sum(arms == a) for a in ("c", "t"))
            assert projection.se**2 == pytest.approx(welch, rel=1e-12)
