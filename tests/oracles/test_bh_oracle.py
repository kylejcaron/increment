"""Cross-checks ``bh_select`` against a frozen R ``p.adjust(method = "BH")``
oracle (see tests/oracles/generate/gen_bh_oracle.R). No R at test time.

Matched estimand: the Benjamini-Hochberg step-up rejection set at FDR level
``q`` over a fixed family of p-values, with tied p-values selected together.
R reports the same set as ``{i : p.adjust(p, "BH")[i] <= q}``. Estimator
level only: family construction, dependence conditions, e-BH and sequential
selection are not compared, and the vectors avoid p-values within floating
point of a ``k * q / m`` boundary, where exact rational and floating-point
adjusted values can legitimately differ in the last bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from increment.estimation.family import bh_select

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "bh_p_adjust.json").read_text())


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["id"])
def test_selected_set_matches_r_p_adjust_bh(case):
    selected, _ = bh_select(case["p"], case["q"])
    assert sorted(selected) == sorted(i - 1 for i in case["rejected"])
