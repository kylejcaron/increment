"""Cross-checks the default fixed-horizon mean-difference interval against a
frozen R ``t.test(var.equal = FALSE)`` oracle (see
tests/oracles/generate/gen_welch_oracle.R). No R at test time.

Matched estimand: the unweighted two-sample difference of arm means with the
Welch (Satterthwaite) variance estimator, Welch degrees of freedom and a
two-sided central t interval. The rows are read from the absolute-axis fields
of ``Analysis.from_unit_summary(...).run()``; the oracle does not cover
cluster-robust, weighted, covariate-adjusted, ratio or sequential contrasts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from increment import Analysis

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "welch_t.json").read_text())
_TOL = _FIXTURE["tolerance"]


def _run(case: dict):
    inp = case["input"]
    control, treatment = inp["control"], inp["treatment"]
    frame = pd.DataFrame(
        {
            "unit": range(len(control) + len(treatment)),
            "group": ["C"] * len(control) + ["T"] * len(treatment),
            "y": [*control, *treatment],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="group", control="C", metrics={"y": "mean"}
    )
    (row,) = analysis.run()
    return row


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["id"])
def test_welch_difference_interval_matches_r_t_test(case):
    row = _run(case)
    assert row.abs_reference_kind == "t"
    assert row.abs_reference_df == pytest.approx(case["df"], rel=_TOL["rel"], abs=0.0)
    assert row.abs_diff == pytest.approx(case["diff"], rel=_TOL["rel"], abs=0.0)
    assert row.abs_se == pytest.approx(case["se"], rel=_TOL["rel"], abs=0.0)
    assert row.abs_lb == pytest.approx(case["lower"], rel=_TOL["rel"], abs=0.0)
    assert row.abs_ub == pytest.approx(case["upper"], rel=_TOL["rel"], abs=0.0)
