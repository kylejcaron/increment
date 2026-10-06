"""The exact-binomial row's additive Wald sidecar takes its critical value
from the shared survival-form tail helpers, so an extreme alpha resolves to
a finite interval or refuses by code, and the nominal case is unchanged."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import norm

from increment.errors import InvalidRequestError
from increment.estimation._tails import two_sided_critical_value, wald_bounds
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import _binomial_abs_bounds, estimate_lift
from increment.semantics.models import ConversionMetric

# 20/200 vs 40/200: difference 0.1, hypot of the two arms' Wald standard errors.
_ABS_DIFF = 0.1
_ABS_SE = 0.0354440602504168


def test_extreme_alpha_resolves_through_the_survival_form():
    lower, upper = _binomial_abs_bounds(_ABS_DIFF, _ABS_SE, 1e-300)
    assert math.isfinite(lower) and math.isfinite(upper)
    crit = two_sided_critical_value(norm.isf, 1e-300, what="reference")
    assert (lower, upper) == wald_bounds(_ABS_DIFF, crit, _ABS_SE, what="reference")
    # norm.isf(5e-301) = 37.07; the complement form ppf(1 - 5e-301) is ppf(1.0) = inf.
    assert lower == pytest.approx(-1.213762018875256, rel=1e-12)
    assert upper == pytest.approx(1.4137620188752562, rel=1e-12)


def test_alpha_below_tail_resolution_refuses_by_code():
    # 5e-324 / 2 rounds to a zero tail probability: no quantile exists.
    with pytest.raises(InvalidRequestError) as raised:
        _binomial_abs_bounds(_ABS_DIFF, _ABS_SE, 5e-324)
    assert raised.value.code == "estimation.tails.unresolvable"


def _summary(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    out = []
    for group_id, n, successes in rows:
        arm = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id=group_id,
            n=n,
            successes=successes,
            sum_y=float(successes),
            sum_y2=float(successes),
        )
        out.append(
            {
                "experiment_id": arm.study_id,
                "metric": arm.metric,
                "group_id": arm.group_id,
                "n": arm.n,
                "successes": arm.successes,
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
        )
    return pd.DataFrame(out)


def test_nominal_alpha_sidecar_is_unchanged():
    computation = estimate_lift(
        metrics=[ConversionMetric(name="conv", entity="user", fact="conv")],
        summary=_summary([("control", 200, 20), ("treatment", 200, 40)]),
        control_group="control",
        alpha=0.05,
    )
    assert computation.failures == {}
    (row,) = computation.results
    assert row.reference_kind == "binomial"
    assert row.abs_diff == pytest.approx(_ABS_DIFF, abs=1e-15)
    assert row.abs_se == pytest.approx(_ABS_SE, abs=1e-15)
    assert row.abs_lb == pytest.approx(0.030530918443315333, abs=1e-12)
    assert row.abs_ub == pytest.approx(0.16946908155668466, abs=1e-12)
