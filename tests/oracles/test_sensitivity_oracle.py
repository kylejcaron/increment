"""Published benchmarks and independent confounded-population witnesses."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from increment.sensitivity import CohortRiskRatio, risk_ratio_sensitivity

_FIXTURE = json.loads((Path(__file__).parent / "fixtures/cohort_evalues.json").read_text())


def _study(rr: float, interval=None) -> CohortRiskRatio:
    return CohortRiskRatio(
        risk_ratio=rr,
        confidence_interval=interval,
        population="cohort",
        exposure="binary A",
        outcome="binary Y",
    )


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["id"])
def test_published_evalue_examples(case) -> None:
    result = risk_ratio_sensitivity(_study(case["risk_ratio"], case["confidence_interval"]))
    tolerance = case["absolute_tolerance"]
    assert result.point_evalue == pytest.approx(case["point_evalue"], rel=0, abs=tolerance)
    assert result.confidence_evalue == pytest.approx(
        case["confidence_evalue"], rel=0, abs=tolerance
    )


@pytest.mark.parametrize("threshold", [1.0, 1.25, 1e150])
def test_neighboring_values_match_high_precision_quadratic_root(threshold) -> None:
    rr = math.nextafter(threshold, math.inf)
    with localcontext() as ctx:
        ctx.prec = 80
        bias = Decimal(rr) / Decimal(threshold)
        # Solve the quadratic x^2 - 2*B*x + B = 0 independently in decimal arithmetic.
        expected = float((2 * bias + ((2 * bias) ** 2 - 4 * bias).sqrt()) / 2)
    result = risk_ratio_sensitivity(_study(rr), threshold=threshold)
    assert result.point_evalue == pytest.approx(expected, rel=1e-14, abs=0)
    inverse = risk_ratio_sensitivity(_study(1 / rr), threshold=1 / threshold)
    with localcontext() as ctx:
        ctx.prec = 80
        bias = Decimal(1 / threshold) / Decimal(1 / rr)
        expected = float((2 * bias + ((2 * bias) ** 2 - 4 * bias).sqrt()) / 2)
    assert inverse.point_evalue == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "strength", [Fraction(1), Fraction(3, 2), Fraction(2), Fraction(4), Fraction(10)]
)
def test_enumerated_confounder_attains_sharp_bound_and_tipping(strength) -> None:
    # Sharpness at the support boundary, not an identified cohort: all exposed
    # have U=1, and 1/strength of controls do. There is no causal exposure effect.
    # U increases outcome risk by strength. Enumerate observed risks directly.
    low_risk = Fraction(1, 20)
    high_risk = low_risk * strength
    exposed_risk = high_risk
    control_risk = high_risk / strength + low_risk * (1 - 1 / strength)
    observed_rr = exposed_risk / control_risk
    # Under intervention both arms have the same population risk, so causal RR=1.
    result = risk_ratio_sensitivity(
        _study(float(observed_rr)),
        exposure_confounder_rr=float(strength),
        confounder_outcome_rr=float(strength),
    )
    assert result.point_bound == pytest.approx(1, rel=1e-14, abs=0)
    assert result.point_evalue == pytest.approx(float(strength), rel=1e-14, abs=0)
    reverse = risk_ratio_sensitivity(
        _study(float(1 / observed_rr)),
        exposure_confounder_rr=float(strength),
        confounder_outcome_rr=float(strength),
    )
    assert reverse.point_bound == pytest.approx(1, rel=1e-14, abs=0)


def test_enumerated_confounding_population_retains_true_causal_effect() -> None:
    # P(U=1|A=1)=4/5; P(U=1|A=0)=1/5, equally prevalent exposure arms.
    # Outcome risks are 1/10, 4/10 without exposure, twice those with exposure.
    prevalence = (Fraction(1, 5), Fraction(4, 5))
    risk = ((Fraction(1, 10), Fraction(4, 10)), (Fraction(2, 10), Fraction(8, 10)))
    observed = [
        sum(risk[a][u] * (prevalence[a] if u else 1 - prevalence[a]) for u in (0, 1))
        for a in (0, 1)
    ]
    population_u = sum(prevalence) / 2
    causal = [
        sum(risk[a][u] * (population_u if u else 1 - population_u) for u in (0, 1)) for a in (0, 1)
    ]
    assert causal[1] / causal[0] == 2
    assert observed[1] / observed[0] == Fraction(17, 4)
    result = risk_ratio_sensitivity(
        _study(float(observed[1] / observed[0])), exposure_confounder_rr=4, confounder_outcome_rr=4
    )
    assert result.point_bound == pytest.approx(1.859375, rel=1e-14, abs=0)
    assert result.point_bound <= float(causal[1] / causal[0])
    reverse = risk_ratio_sensitivity(
        _study(float(observed[0] / observed[1])), exposure_confounder_rr=4, confounder_outcome_rr=4
    )
    assert reverse.point_bound >= float(causal[0] / causal[1])


def test_asymmetric_confounder_can_tip_with_one_association_below_evalue() -> None:
    result = risk_ratio_sensitivity(_study(2), exposure_confounder_rr=3, confounder_outcome_rr=4)
    assert result.bias_factor == 2
    assert result.point_bound == 1
    assert 3 < result.point_evalue < 4
