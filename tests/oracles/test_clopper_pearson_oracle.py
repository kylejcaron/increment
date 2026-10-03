"""Cross-checks ``clopper_pearson`` against frozen R ``binom.test`` bounds (see
tests/oracles/generate/gen_clopper_pearson_oracle.R). No R at test time.

Matched estimand: the exact two-sided single-arm Clopper-Pearson ``1 - beta``
interval for a binomial rate. Production deliberately rounds outward by a
documented relative slack (``_CP_RELATIVE_SLACK``), so the test asserts
enclosure of R's interval first and agreement within that slack second. Scope
is estimator level, one arm, moderate ``n``; two-arm and risk-ratio sets, and
arms beyond about 2^26, are not covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from increment.estimation.binomial_rr import clopper_pearson

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "clopper_pearson.json").read_text())
_TOL = _FIXTURE["tolerance"]


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["id"])
def test_interval_encloses_and_matches_r_binom_test(case):
    lo, hi = clopper_pearson(case["x"], case["n"], case["beta"])
    assert lo <= case["lower"] + _TOL["enclosure_abs"]
    assert hi >= case["upper"] - _TOL["enclosure_abs"]
    band = _TOL["slack_rel"] * _TOL["headroom"]
    assert case["lower"] - lo <= band * case["lower"] + _TOL["enclosure_abs"]
    assert hi - case["upper"] <= band * (1.0 - case["upper"]) + _TOL["enclosure_abs"]
