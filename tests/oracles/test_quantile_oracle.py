"""Cross-checks increment/estimation/quantile.py's log_quantile_se against
a frozen R qbinom oracle (see tests/oracles/generate/gen_quantile_oracle.R).
No R at test time.

log_quantile_se selects a discrete order-statistic bracket via the exact
binomial quantile of the rank (scipy.stats.binom.ppf/isf). The generator
is an independent R implementation of the identical discrete
rank-selection formula (R's own qbinom(), not a port of scipy's) reading
the same shared CSV, so the two order statistics it picks are expected to
be bit-identical to what log_quantile_se selects internally -- se is
compared to (log(upper)-log(lower))/(2*z) at floating-point tolerance,
not an asymptotic-tolerance band.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from increment.estimation.quantile import log_quantile_se

_HERE = Path(__file__).parent
_FIXTURE = json.loads((_HERE / "fixtures" / "quantile_woodruff.json").read_text())
_CASES = _FIXTURE["cases"]
_TOL = _FIXTURE["tolerance"]
_Z = 1.9599639845400545  # scipy.stats.norm.isf(0.025), the same reference log_quantile_se uses


def _values_for(case_id: str) -> np.ndarray:
    with open(_HERE / "data" / "quantile_values.csv") as f:
        rows = [row for row in csv.DictReader(f) if row["case_id"] == case_id]
    return np.array([float(r["value"]) for r in rows])


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_log_scale_half_width_matches_independent_qbinom_bracket(case):
    values = _values_for(case["id"])
    _, se = log_quantile_se(values, case["q"], alpha=0.05)
    r_se = (math.log(case["upper"]) - math.log(case["lower"])) / (2.0 * _Z)
    assert se == pytest.approx(r_se, rel=_TOL["se_rel"], abs=0.0)
