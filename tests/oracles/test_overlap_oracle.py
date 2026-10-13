"""Cross-checks the overlapping-experiments contrasts against a frozen statsmodels oracle
(see tests/oracles/generate/gen_overlap_oracle.py). No statsmodels at test time.

Matched estimand: a cell-mean contrast of two binary experiments on the jointly
eligible population, computed by statsmodels as ``L @ beta`` from OLS on the
saturated design ``[1, A, B, A*B]`` with the HC2 sandwich. The oracle checks the
estimate and standard error through ``OverlapCells.from_frame``; it does not
cover the Welch-Satterthwaite degrees of freedom, intervals, or multiplicity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from increment.overlap import (
    CONTRASTS,
    OverlapDesign,
    OverlapExperiment,
    estimate_overlap_effects,
)
from tests.overlap.fixtures import from_frame, unit_frame

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "overlap_hc2.json").read_text())
_TOL = _FIXTURE["tolerance"]


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["id"])
def test_overlap_contrasts_match_statsmodels_ols_hc2(case: dict) -> None:
    design = OverlapDesign(
        first=OverlapExperiment(name="checkout", treatment_probability=case["first_probability"]),
        second=OverlapExperiment(
            name="search",
            control="old",
            treatment="new",
            treatment_probability=case["second_probability"],
        ),
    )
    cells = {(int(k[0]), int(k[1])): v for k, v in case["cells"].items()}
    frame = unit_frame(cells, design=design)
    readout = estimate_overlap_effects(from_frame(frame, design), contrasts=CONTRASTS)
    for row in readout.rows:
        want = case["expected"][row.contrast]
        rel, abs_ = (
            (0.0, float(_TOL["large_offset_abs"]))
            if case["id"] == "large_offset"
            else (float(_TOL["rel"]), 0.0)
        )
        assert row.estimate == pytest.approx(want["estimate"], rel=rel, abs=abs_)
        assert row.se == pytest.approx(want["se"], rel=rel, abs=abs_)
