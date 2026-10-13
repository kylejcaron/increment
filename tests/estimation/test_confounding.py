"""Conservative marginal confounding sensitivity contracts."""

from __future__ import annotations

import copy
import math
import pickle
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import brentq
from scipy.stats import norm

from increment.errors import CodedError, InvalidRequestError, UnsupportedRequestError
from increment.estimation.confounding import (
    ConfoundingSensitivity,
    confounding_sensitivity,
    e_value,
    tipping_bias_factor,
)
from increment.estimation.results import (
    Estimate,
    JointContrastReference,
    LiftEstimate,
    RelativeConfidenceSet,
)


def _row(r, lo=None, hi=None, *, alternative="two-sided", **overrides) -> LiftEstimate:
    open_side = None
    if alternative == "greater" and lo is not None and hi is None:
        open_side = "upper"
    elif alternative == "less" and hi is not None and lo is None:
        open_side = "lower"
    has_interval = lo is not None or hi is not None
    fields: dict[str, Any] = {
        "metric": "conversion",
        "group_id": "treatment",
        "method": "observational-label",
        "method_role": "decision",
        "estimand": "ate",
        "value_scale": "relative",
        "inference": "fixed",
        "alternative": alternative,
        "lift": Estimate(
            value=r - 1,
            lb=None if lo is None else lo - 1,
            ub=None if hi is None else hi - 1,
            level=0.95 if has_interval else None,
            alpha=0.05 if has_interval else None,
            open_side=open_side,
        ),
    }
    inference = overrides.pop("inference", "fixed")
    fields.update(overrides)
    row = LiftEstimate(**fields)
    # Non-fixed observational rows need certified sequential state, so copy only to test refusal.
    return row if inference == "fixed" else row.model_copy(update={"inference": inference})


def _population(strata) -> dict:
    """Enumerate binary U, standardizing over every cohort arm."""
    true_risks = [0.0, 0.0]
    identified_risks = [0.0, 0.0]
    rr_eu = rr_ud = 1.0
    for stratum in strata:
        p = stratum["p"]
        propensities = stratum["propensities"]
        u_probabilities = stratum["u_probabilities"]
        risks = stratum["risks"]
        population_u = sum(f * q for f, q in zip(propensities, u_probabilities, strict=True))
        for arm in (0, 1):
            for u in (0, 1):
                q_true = population_u if u else 1 - population_u
                q_observed = u_probabilities[arm] if u else 1 - u_probabilities[arm]
                true_risks[arm] += p * q_true * risks[arm][u]
                identified_risks[arm] += p * q_observed * risks[arm][u]
        for arm_risks in risks:
            rr_ud = max(rr_ud, max(arm_risks) / min(arm_risks))
        for q_a in u_probabilities:
            for q_b in u_probabilities:
                rr_eu = max(rr_eu, q_a / q_b, (1 - q_a) / (1 - q_b))
    return {
        "true_rr": true_risks[1] / true_risks[0],
        "identified_rr": identified_risks[1] / identified_risks[0],
        "rr_eu": rr_eu,
        "rr_ud": rr_ud,
    }


def _random_population(rng, *, independent=False):
    n_strata = int(rng.integers(1, 5))
    n_arms = int(rng.integers(2, 4))
    weights = rng.dirichlet(np.ones(n_strata))
    strata = []
    for c, weight in enumerate(weights):
        propensities = 0.01 + (1 - 0.01 * n_arms) * rng.dirichlet(np.ones(n_arms))
        u_probabilities = rng.uniform(0.02, 0.98, size=n_arms)
        if independent:
            u_probabilities[:] = u_probabilities[0]
        base = math.exp(rng.uniform(math.log(1e-6), math.log(0.2)))
        u_effect = rng.uniform(1.0, 5.0)
        control = np.array([base, base * u_effect])
        risks = np.tile(control, (n_arms, 1))
        if c % 2 == 0:
            causal_effects = np.exp(rng.uniform(math.log(0.25), math.log(4.0), size=n_arms - 1))
            risks[1:] *= causal_effects[:, None]
        risks /= max(1.0, float(risks.max()))
        strata.append(
            {
                "p": float(weight),
                "propensities": propensities.tolist(),
                "u_probabilities": u_probabilities.tolist(),
                "risks": risks.tolist(),
            }
        )
    return strata


def test_no_confounding_limit():
    assert e_value(1.0) == 1.0
    result = confounding_sensitivity(_row(1.3), outcome="binary", threshold_risk_ratio=1.3)
    assert result.bias_factor_point == result.e_value_point == 1.0
    assert result.direction == "at"
    assert tipping_bias_factor(1.3, 1.3) == 1.0


@pytest.mark.parametrize("b", [1.0001, 1.5, 2.0, 3.9, 10.0, 1e6])
def test_e_value_solves_bounding_factor_equation(b):
    root = brentq(lambda m: m**2 / (2 * m - 1) - b, 1, 4 * b)
    assert e_value(b) == pytest.approx(root, rel=1e-10)


def test_published_bounding_factor_values():
    bounding_factor = 3.0 * 3.0 / (3.0 + 3.0 - 1.0)
    assert bounding_factor == pytest.approx(1.8, abs=1e-12)
    assert e_value(1.8) == pytest.approx(3.0, abs=1e-12)


def test_counterexample_is_bounded():
    bases = ((1e-6, 0.06227217045454546), (0.04, 1e-6))
    strata = [
        {
            "p": 0.5,
            "propensities": [1 - f, f],
            "u_probabilities": [0.2, 0.6],
            "risks": [[b, 3 * b] for b in bases[c]],
        }
        for c, f in enumerate((0.01, 0.99))
    ]
    population = _population(strata)
    assert population["true_rr"] == pytest.approx(1.0, abs=1e-9)
    assert population["identified_rr"] == pytest.approx(2.4463848225, rel=1e-8)
    assert population["rr_eu"] == pytest.approx(3.0, abs=1e-12)
    assert population["rr_ud"] == pytest.approx(3.0, abs=1e-12)
    assert e_value(2.4463848225) > 3.0
    assert confounding_sensitivity(_row(2.4463848225), outcome="binary").e_value_point <= 3.0


def test_bound_holds_on_random_populations():
    rng = np.random.default_rng(20261013)
    arm_counts = set()
    protective = crossings = 0
    for _ in range(300):
        strata = _random_population(rng)
        arm_counts.add(len(strata[0]["propensities"]))
        for stratum in strata:
            assert all(0.01 <= f <= 0.99 for f in stratum["propensities"])
            assert sum(stratum["propensities"]) == pytest.approx(1.0, abs=1e-15)
            assert all(0 < risk <= 1 for risks in stratum["risks"] for risk in risks)
        population = _population(strata)
        true_rr, identified_rr = population["true_rr"], population["identified_rr"]
        rr_eu, rr_ud = population["rr_eu"], population["rr_ud"]
        bounding_factor = rr_eu * rr_ud / (rr_eu + rr_ud - 1)
        assert true_rr >= identified_rr / bounding_factor**2 - 1e-12, strata
        assert true_rr <= identified_rr * bounding_factor**2 + 1e-12, strata
        protective += identified_rr < 1
        if (true_rr - 1) * (identified_rr - 1) < 0:
            crossings += 1
            result = confounding_sensitivity(_row(identified_rr), outcome="binary")
            assert result.e_value_point <= max(rr_eu, rr_ud) + 1e-9, strata
    assert arm_counts == {2, 3}
    assert protective > 0
    assert crossings > 0


def test_no_confounding_population_recovers_identified_rr():
    rng = np.random.default_rng(20261013)
    for _ in range(300):
        population = _population(_random_population(rng, independent=True))
        assert population["rr_eu"] == 1.0
        bounding_factor = (
            population["rr_eu"]
            * population["rr_ud"]
            / (population["rr_eu"] + population["rr_ud"] - 1)
        )
        assert bounding_factor == pytest.approx(1.0, abs=1e-15)
        assert population["true_rr"] == pytest.approx(population["identified_rr"], rel=1e-12)


def test_protective_estimate_uses_upper_limit():
    result = confounding_sensitivity(_row(0.80, 0.71, 0.91), outcome="binary")
    assert result.bias_factor_point == pytest.approx(math.sqrt(1 / 0.8))
    assert result.bias_factor_interval == pytest.approx(math.sqrt(1 / 0.91))
    assert result.e_value_point == pytest.approx(e_value(math.sqrt(1 / 0.8)))
    assert result.e_value_interval == pytest.approx(e_value(math.sqrt(1 / 0.91)))
    assert result.direction == "below"


@pytest.mark.parametrize("lo", [0.95, -0.2])
def test_crossing_interval_gives_one(lo):
    result = confounding_sensitivity(_row(1.4, lo, 1.8), outcome="binary")
    assert result.e_value_interval == result.bias_factor_interval == 1.0


def test_threshold_shifts_reference():
    row = _row(1.26, 1.10, 1.5)
    result = confounding_sensitivity(row, outcome="binary", threshold_risk_ratio=1.05)
    assert result.bias_factor_point == pytest.approx(math.sqrt(1.2))
    assert result.bias_factor_interval == pytest.approx(math.sqrt(1.10 / 1.05))
    assert (
        confounding_sensitivity(row, outcome="binary", threshold_risk_ratio=1.26).e_value_point
        == 1.0
    )


@pytest.mark.parametrize(
    "row",
    [
        _row(1.4, hi=1.8, alternative="less"),
        _row(0.8, lo=0.7, alternative="greater"),
        _row(
            1.4,
            relative_unavailable_reason="joint_covariance_indefinite",
            scale="linear",
            abs_diff=0.4,
        ),
        _row(1.0),
    ],
    ids=["open-lower", "open-upper", "relative-unavailable", "at-without-interval"],
)
def test_missing_threshold_side_limit_is_null_with_reason(row):
    result = confounding_sensitivity(row, outcome="binary")
    assert result.e_value_interval is None
    assert result.bias_factor_interval is None
    assert result.interval_unavailable_reason == "interval_limit_unavailable"
    assert math.isfinite(result.e_value_point)


def test_extreme_ratios():
    result = confounding_sensitivity(_row(1e160), outcome="binary")
    assert math.isfinite(result.e_value_point)
    assert result.e_value_point == pytest.approx(2e80, rel=1e-6)
    assert e_value(1e160) == pytest.approx(2e160, rel=1e-12)
    for r, t in ((2.0, 5e-324), (1e308, 1e-10)):
        with pytest.raises(InvalidRequestError) as excinfo:
            confounding_sensitivity(_row(r), outcome="binary", threshold_risk_ratio=t)
        assert excinfo.value.code == "sensitivity.confounding.ratio_not_representable"
        assert dict(excinfo.value.context) == {"observed_risk_ratio": r, "threshold_risk_ratio": t}
    with pytest.raises(UnsupportedRequestError) as excinfo:
        confounding_sensitivity(_row(5e-324), outcome="binary")
    assert excinfo.value.code == "sensitivity.confounding.risk_ratio_unavailable"

    relative = RelativeConfidenceSet(
        reference=JointContrastReference(a=0, c=0.1, var_a=0, var_c=1, cov_ac=0),
        alpha=0.05,
        geometry="all_real",
        intervals=((None, None),),
    )
    row = _row(
        1.0,
        lift=relative.estimate(),
        scale="linear",
        relative_confidence_set=relative,
        abs_diff=0.0,
    )
    result = confounding_sensitivity(row, outcome="binary")
    assert result.e_value_interval == result.bias_factor_interval == 1.0
    assert result.interval_unavailable_reason is None
    assert result.interval_level == pytest.approx(0.95)


@pytest.mark.parametrize("b", [math.inf, math.nan, 1e308])
def test_nonrepresentable_e_value_refuses(b):
    with pytest.raises(InvalidRequestError) as excinfo:
        e_value(b)
    assert excinfo.value.code == "sensitivity.confounding.ratio_not_representable"


_UNSUPPORTED = [
    pytest.param({}, {"outcome": "continuous"}, "outcome_unsupported", id="outcome-continuous"),
    *[
        pytest.param({"estimand": estimand}, {}, "estimand_unsupported", id=f"estimand-{estimand}")
        for estimand in ("itt", "plr_slope", "late", "compliance")
    ],
    pytest.param({"value_scale": "absolute"}, {}, "scale_unsupported", id="scale-absolute"),
    pytest.param(
        {"inference": "always_valid"}, {}, "sequential_unsupported", id="sequential-always-valid"
    ),
    pytest.param(
        {
            "lift": None,
            "relative_unavailable_reason": "nonpositive_arm_mean",
            "scale": "linear",
            "abs_diff": 0.4,
        },
        {},
        "risk_ratio_unavailable",
        id="lift-unavailable",
    ),
    pytest.param(
        {"lift": Estimate(value=-1.0)}, {}, "risk_ratio_unavailable", id="risk-ratio-zero"
    ),
    *[
        pytest.param({}, {"threshold_risk_ratio": t}, "threshold_invalid", id=f"threshold-{label}")
        for label, t in (("zero", 0), ("negative", -1), ("nan", math.nan), ("inf", math.inf))
    ],
    pytest.param(
        {},
        {"threshold_risk_ratio": 5e-324},
        "ratio_not_representable",
        id="ratio-overflow",
    ),
]


@pytest.mark.parametrize(("overrides", "kwargs", "code"), _UNSUPPORTED)
def test_refuses_unsupported_rows(overrides, kwargs, code):
    error_type = (
        InvalidRequestError
        if code in ("threshold_invalid", "ratio_not_representable")
        else UnsupportedRequestError
    )
    request: dict[str, Any] = {"outcome": "binary", **kwargs}
    with pytest.raises(error_type) as excinfo:
        confounding_sensitivity(_row(1.4, **overrides), **request)
    assert excinfo.value.code == f"sensitivity.confounding.{code}"
    if overrides.get("relative_unavailable_reason") is not None:
        assert (
            excinfo.value.context["relative_unavailable_reason"]
            == overrides["relative_unavailable_reason"]
        )


@pytest.mark.parametrize(("overrides", "kwargs", "code"), _UNSUPPORTED)
def test_refusals_round_trip(overrides, kwargs, code):
    request: dict[str, Any] = {"outcome": "binary", **kwargs}
    with pytest.raises(CodedError) as excinfo:
        confounding_sensitivity(_row(1.4, **overrides), **request)
    refusal = excinfo.value
    assert refusal.code == f"sensitivity.confounding.{code}"
    for clone in (copy.deepcopy(refusal), pickle.loads(pickle.dumps(refusal))):
        assert type(clone) is type(refusal)
        assert clone.code == refusal.code
        assert clone.context.keys() == refusal.context.keys()
        for key, value in refusal.context.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(cast(float, clone.context[key]))
            else:
                assert clone.context[key] == value
        with pytest.raises(TypeError):
            cast(Any, clone.context)["new"] = "changed"


@pytest.mark.parametrize("risk_ratio", [0.8, 1.2])
def test_all_real_without_displayed_limits_gives_one(risk_ratio):
    reference = JointContrastReference(a=(risk_ratio - 1) * 0.1, c=0.1, var_a=0, var_c=1, cov_ac=0)
    relative = RelativeConfidenceSet(reference=reference, alpha=0.05, alternative="greater")
    assert relative.geometry == "all_real"
    lift = relative.estimate()
    assert lift is not None and lift.lb is None and lift.ub is None
    row = _row(
        risk_ratio,
        alternative="greater",
        lift=lift,
        scale="linear",
        relative_confidence_set=relative,
        abs_diff=reference.a,
    )
    result = confounding_sensitivity(row, outcome="binary")
    assert result.bias_factor_interval == result.e_value_interval == 1.0
    assert result.interval_unavailable_reason is None


@pytest.mark.parametrize(
    ("alternative", "sign", "risk_ratio", "limit", "bias_factor", "equal_strength"),
    [
        ("greater", -1, 0.8, 0.9, 1.0540925533894598, 1.292878142824106),
        ("less", 1, 1.2, 1.1, 1.0488088481701516, 1.2750634654265705),
    ],
)
def test_directional_all_real_uses_displayed_threshold_side_limit(
    alternative, sign, risk_ratio, limit, bias_factor, equal_strength
):
    c = float(norm.isf(0.05)) / 2
    reference = JointContrastReference(a=sign * c / 5, c=c, var_a=0, var_c=0.25, cov_ac=0)
    relative = RelativeConfidenceSet(reference=reference, alpha=0.05, alternative=alternative)
    assert relative.geometry == "all_real"
    lift = relative.estimate()
    assert lift is not None
    row = _row(
        risk_ratio,
        alternative=alternative,
        lift=lift,
        scale="linear",
        relative_confidence_set=relative,
        abs_diff=reference.a,
    )
    result = confounding_sensitivity(row, outcome="binary")
    assert result.observed_risk_ratio == pytest.approx(risk_ratio, rel=1e-12)
    displayed_limit = (
        result.interval_upper_risk_ratio if risk_ratio < 1 else result.interval_lower_risk_ratio
    )
    assert displayed_limit == pytest.approx(limit, rel=1e-12)
    assert result.bias_factor_interval == pytest.approx(bias_factor, rel=1e-12)
    assert result.e_value_interval == pytest.approx(equal_strength, rel=1e-12)
    assert result.interval_level == lift.level
    assert result.interval_unavailable_reason is None


@pytest.mark.parametrize("analysis_population", ["assigned", "triggered"])
@pytest.mark.parametrize("estimand", ["ate", "overlap_subpopulation_ate"])
def test_result_preserves_input_and_identity(estimand, analysis_population):
    row = _row(
        1.4,
        1.2,
        1.8,
        estimand=estimand,
        population="overlap population",
        analysis_population=analysis_population,
        source_snapshot_id=f"snapshot-{analysis_population}",
    )
    before = row.model_dump()
    result = confounding_sensitivity(row, outcome="binary")
    assert row.model_dump() == before
    for field in (
        "metric",
        "group_id",
        "method",
        "estimand",
        "population",
        "analysis_population",
        "source_snapshot_id",
    ):
        assert getattr(result, field) == getattr(row, field)
    assert isinstance(result.assumptions, tuple)
    assert result.assumptions and all(isinstance(item, str) for item in result.assumptions)
    assert result.robustness_not_identification is True
    assert result.outcome_attested == "binary"
    assert result.bound == "marginal_squared_bounding_factor"
    assert result.observed_risk_ratio == pytest.approx(1.4)
    assert result.interval_lower_risk_ratio == pytest.approx(1.2)
    assert result.interval_upper_risk_ratio == pytest.approx(1.8)
    assert result.interval_level == 0.95
    assert result.threshold_risk_ratio == 1.0


@pytest.mark.parametrize("analysis_population", ["assigned", "triggered"])
def test_result_round_trips(analysis_population):
    row = _row(
        0.8,
        0.71,
        0.91,
        analysis_population=analysis_population,
        source_snapshot_id=f"snapshot-{analysis_population}",
    )
    result = confounding_sensitivity(row, outcome="binary")
    assert result.analysis_population == analysis_population
    assert result.source_snapshot_id == row.source_snapshot_id
    assert ConfoundingSensitivity.model_validate(result.model_dump()) == result
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(ValidationError) as frozen:
        cast(Any, result).e_value_point = 1.0
    assert frozen.value.errors()[0]["type"] == "frozen_instance"


@pytest.mark.parametrize(
    ("r", "lo", "hi", "alternative"),
    [(1.4, 1.2, None, "greater"), (0.8, None, 0.91, "less"), (1.0, 0.8, 1.2, "two-sided")],
)
def test_available_threshold_side_limit_is_used(r, lo, hi, alternative):
    result = confounding_sensitivity(_row(r, lo, hi, alternative=alternative), outcome="binary")
    assert result.e_value_interval is not None
    assert result.interval_unavailable_reason is None
    if r == 1.0:
        assert result.e_value_interval == result.bias_factor_interval == 1.0


def test_refusal_check_order():
    row = _row(1.4, estimand="itt", value_scale="absolute", inference="always_valid")
    for kwargs, code in (
        ({"outcome": "continuous", "threshold_risk_ratio": 0}, "outcome_unsupported"),
        ({"outcome": "binary", "threshold_risk_ratio": 0}, "estimand_unsupported"),
    ):
        request: dict[str, Any] = kwargs
        with pytest.raises(CodedError) as excinfo:
            confounding_sensitivity(row, **request)
        assert excinfo.value.code == f"sensitivity.confounding.{code}"
    with pytest.raises(InvalidRequestError) as excinfo:
        confounding_sensitivity(
            _row(
                1.4,
                lift=None,
                relative_unavailable_reason="nonpositive_arm_mean",
                scale="linear",
                abs_diff=0.4,
            ),
            outcome="binary",
            threshold_risk_ratio=0,
        )
    assert excinfo.value.code == "sensitivity.confounding.threshold_invalid"
