"""Consumer contracts for the standalone cohort sensitivity ingress."""

from __future__ import annotations

import math
import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pytest

from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.sensitivity import CohortRiskRatio, risk_ratio_sensitivity


def study(**changes: Any) -> CohortRiskRatio:
    inputs: dict[str, Any] = {
        "risk_ratio": 2.0,
        "population": "eligible cohort at enrollment",
        "exposure": "opted in",
        "outcome": "converted within 30 days",
        "confidence_interval": (1.5, 3.0),
    }
    return CohortRiskRatio(**(inputs | changes))


def test_declared_decision_preserves_estimate_and_reports_assumptions() -> None:
    original = study()
    result = risk_ratio_sensitivity(original, threshold=1.25)
    assert result.study is original
    assert result.study.risk_ratio == 2
    assert result.study.confidence_interval == (1.5, 3)
    assert result.threshold == 1.25
    assert result.direction == "above"
    assert result.point_evalue == pytest.approx(2.5797958971132715, rel=1e-14, abs=0)
    assert result.confidence_evalue == pytest.approx(1.6898979485566357, rel=1e-14, abs=0)
    assert result.confidence_unavailable_reason is None
    assert result.guarantee == "exact"
    assert result.assumptions
    with pytest.raises(FrozenInstanceError):
        cast(Any, result).point_evalue = 99


@pytest.mark.parametrize(
    "direction,rr,interval", [("above", 2.0, (1.5, 3.0)), ("below", 0.5, (1 / 3, 2 / 3))]
)
def test_no_confounding_preserves_bounds(direction, rr, interval) -> None:
    result = risk_ratio_sensitivity(
        study(risk_ratio=rr, confidence_interval=interval), direction=direction
    )
    assert result.bias_factor == 1
    assert result.point_bound == rr
    assert result.confidence_limit_bound == interval[0 if direction == "above" else 1]


@pytest.mark.parametrize("a,b", [(1, 1), (2, 3), (4, 4), (10, 2), (1e308, 1e308)])
def test_inverse_decisions_and_scenario_bounds_are_equivalent(a, b) -> None:
    above = risk_ratio_sensitivity(
        study(), threshold=1.25, exposure_confounder_rr=a, confounder_outcome_rr=b
    )
    below = risk_ratio_sensitivity(
        study(risk_ratio=0.5, confidence_interval=(1 / 3, 2 / 3)),
        threshold=0.8,
        exposure_confounder_rr=a,
        confounder_outcome_rr=b,
    )
    assert below.direction == "below"
    assert below.point_evalue == pytest.approx(above.point_evalue, rel=1e-14, abs=0)
    assert below.confidence_evalue == pytest.approx(above.confidence_evalue, rel=1e-14, abs=0)
    assert below.point_bound == pytest.approx(1 / above.point_bound, rel=1e-14, abs=0)
    assert above.confidence_limit_bound is not None
    assert below.confidence_limit_bound == pytest.approx(
        1 / above.confidence_limit_bound, rel=1e-14, abs=0
    )


@pytest.mark.parametrize(
    "rr,interval,threshold,direction",
    [
        (1.0, (0.8, 1.2), 1.0, "above"),
        (1.0, (0.8, 1.2), 1.0, "below"),
        (2.0, (1.5, 3.0), 2.0, "above"),
        (2.0, (1.5, 3.0), 2.5, "above"),
        (0.5, (0.3, 0.7), 0.5, "below"),
        (0.5, (0.3, 0.7), 0.4, "below"),
        (0.5, (0.3, 0.7), 1.0, "above"),
        (2.0, (1.5, 3.0), 1.0, "below"),
    ],
)
def test_decision_already_fails_has_unit_tipping_point(rr, interval, threshold, direction) -> None:
    result = risk_ratio_sensitivity(
        study(risk_ratio=rr, confidence_interval=interval), threshold=threshold, direction=direction
    )
    assert result.point_evalue == result.confidence_evalue == 1


@pytest.mark.parametrize("interval,threshold", [((0.9, 3), 1), ((1, 3), 1), ((1.2, 3), 1.5)])
def test_interval_that_does_not_establish_decision_has_unit_evalue(interval, threshold) -> None:
    result = risk_ratio_sensitivity(study(confidence_interval=interval), threshold=threshold)
    assert result.point_evalue > 1
    assert result.confidence_evalue == 1


def test_missing_uncertainty_remains_null_with_reason() -> None:
    result = risk_ratio_sensitivity(study(confidence_interval=None))
    assert result.point_evalue > 1
    assert result.confidence_evalue is None
    assert result.confidence_limit_bound is None
    assert result.confidence_unavailable_reason == "confidence_interval_not_supplied"


@pytest.mark.parametrize("restore", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_replay_runs_consumer_on_restored_summary_and_retains_result(restore) -> None:
    original = study()
    restored = restore(original)
    result = risk_ratio_sensitivity(
        restored, threshold=1.2, exposure_confounder_rr=2, confounder_outcome_rr=3
    )
    assert result.study == original
    assert restore(result) == result
    assert (
        risk_ratio_sensitivity(
            restore(result).study, threshold=1.2, exposure_confounder_rr=2, confounder_outcome_rr=3
        )
        == result
    )


def test_input_snapshots_interval_and_numpy_scalars() -> None:
    bounds = [1.5, 3.0]
    original = study(risk_ratio=np.float64(2), confidence_interval=bounds)
    bounds[0] = 0.1
    assert original.confidence_interval == (1.5, 3.0)
    assert type(original.risk_ratio) is float


@pytest.mark.parametrize(
    "field,value",
    [
        ("estimator", "iptw"),
        ("estimator", "aipw"),
        ("estimator", "dml"),
        ("estimator", "logistic"),
        ("estimator", "log_binomial"),
        ("estimand", "att"),
        ("estimand", "conditional_risk_ratio"),
        ("effect_scale", "risk_difference"),
        ("effect_scale", "odds_ratio"),
        ("effect_scale", "hazard_ratio"),
        ("effect_scale", "ratio_of_means"),
        ("inference", "always_valid"),
        ("inference", "asymptotic_mean"),
    ],
)
def test_unsupported_estimator_estimand_scale_and_inference_refuse(field, value) -> None:
    with pytest.raises(UnsupportedRequestError) as caught:
        study(**{field: value})
    assert caught.value.code == "sensitivity.study.unsupported"
    assert caught.value.context["field"] == field
    assert caught.value.context["value"] == value


def test_adjusted_estimate_refuses_and_context_is_immutable_and_transportable() -> None:
    covariates = ["age"]
    with pytest.raises(UnsupportedRequestError) as caught:
        study(adjustment_covariates=covariates)
    covariates.append("tenure")
    for error in (caught.value, deepcopy(caught.value), pickle.loads(pickle.dumps(caught.value))):
        assert error.code == "sensitivity.study.adjusted"
        assert error.context["value"] == ("age",)
        with pytest.raises(TypeError):
            cast(Any, error.context)["value"] = ()


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("risk_ratio", 0, "sensitivity.input.number"),
        ("risk_ratio", -1, "sensitivity.input.number"),
        ("risk_ratio", math.inf, "sensitivity.input.number"),
        ("risk_ratio", math.nan, "sensitivity.input.number"),
        ("risk_ratio", True, "sensitivity.input.number"),
        ("risk_ratio", "2", "sensitivity.input.number"),
        ("risk_ratio", 10**400, "sensitivity.input.number"),
        ("confidence_level", 1, "sensitivity.input.number"),
        ("confidence_level", 0, "sensitivity.input.number"),
        ("confidence_interval", (1,), "sensitivity.input.interval"),
        ("confidence_interval", (3, 1), "sensitivity.input.interval"),
        ("confidence_interval", (2.1, 3), "sensitivity.input.interval"),
        ("confidence_interval", (0, 3), "sensitivity.input.number"),
        ("confidence_interval", (1, math.inf), "sensitivity.input.number"),
        ("population", " ", "sensitivity.input.identity"),
        ("exposure", None, "sensitivity.input.identity"),
        ("outcome", "", "sensitivity.input.identity"),
    ],
)
def test_invalid_summaries_have_stable_refusals(field, value, code) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        study(**{field: value})
    assert caught.value.code == code
    assert deepcopy(caught.value).code == code
    assert pickle.loads(pickle.dumps(caught.value)).code == code


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"threshold": 0}, "sensitivity.input.number"),
        ({"threshold": math.nan}, "sensitivity.input.number"),
        ({"direction": "both"}, "sensitivity.decision.direction"),
        ({"direction": "above", "threshold": 0.9}, "sensitivity.decision.threshold"),
        ({"direction": "below", "threshold": 1.1}, "sensitivity.decision.threshold"),
        ({"exposure_confounder_rr": 0.9}, "sensitivity.input.number"),
        ({"confounder_outcome_rr": math.inf}, "sensitivity.input.number"),
        ({"confounder_outcome_rr": True}, "sensitivity.input.number"),
    ],
)
def test_invalid_decisions_and_strengths_refuse(kwargs, code) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(study(), **kwargs)
    assert caught.value.code == code


def test_an_analysis_row_cannot_be_treated_as_a_risk_ratio() -> None:
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(cast(Any, {"abs_est": 0.1}))
    assert caught.value.code == "sensitivity.input.study"


@pytest.mark.parametrize("rr", [1e200, 1e-200])
def test_representable_extreme_tipping_points_avoid_intermediate_overflow(rr) -> None:
    result = risk_ratio_sensitivity(study(risk_ratio=rr, confidence_interval=None))
    assert result.point_evalue == pytest.approx(2e200, rel=1e-14, abs=0)
    assert result.point_bound == rr


def test_large_confounding_strengths_do_not_overflow_bias_factor() -> None:
    result = risk_ratio_sensitivity(
        study(), exposure_confounder_rr=1e308, confounder_outcome_rr=1e308
    )
    assert result.bias_factor == pytest.approx(5e307, rel=1e-14, abs=0)
    assert result.point_bound > 0
    assert math.isfinite(result.point_bound)
    one = risk_ratio_sensitivity(study(), exposure_confounder_rr=1, confounder_outcome_rr=1e308)
    assert one.bias_factor == 1


@pytest.mark.parametrize("rr", [1e308, 1e-308])
def test_unrepresentable_evalue_refuses(rr) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(study(risk_ratio=rr, confidence_interval=None))
    assert caught.value.code == "sensitivity.numerical.range"


def test_unrepresentable_bound_refuses_without_inf_or_zero() -> None:
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(
            study(risk_ratio=1e-300, confidence_interval=None),
            direction="above",
            exposure_confounder_rr=1e308,
            confounder_outcome_rr=1e308,
        )
    assert caught.value.code == "sensitivity.numerical.range"
    # A below decision can challenge an already failed RR while its bound overflows.
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(
            study(risk_ratio=1e308, confidence_interval=None),
            direction="below",
            exposure_confounder_rr=10,
            confounder_outcome_rr=10,
        )
    assert caught.value.code == "sensitivity.numerical.range"


def test_nondefault_confidence_level_is_preserved() -> None:
    result = risk_ratio_sensitivity(replace(study(), confidence_level=0.99))
    assert result.study.confidence_level == 0.99


def test_subgroup_summary_preserves_declared_population() -> None:
    subgroup = replace(study(), population="predeclared new-customer subgroup")
    result = risk_ratio_sensitivity(subgroup)
    assert result.study.population == "predeclared new-customer subgroup"
    assert result.study.confidence_interval == subgroup.confidence_interval


class _BrokenRepr:
    def __repr__(self) -> str:
        raise RuntimeError("bad repr")


class _UnconvertibleFloat(float):
    def __float__(self) -> float:
        raise TypeError("no float conversion")


@pytest.mark.parametrize(
    "value",
    [
        10**10000,
        _BrokenRepr(),
        _UnconvertibleFloat(1),
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("1e400"),
    ],
    ids=[
        "huge_integer",
        "broken_repr",
        "bad_float_conversion",
        "extended_nan",
        "extended_inf",
        "extended_overflow",
    ],
)
@pytest.mark.parametrize(
    "route", ["risk_ratio", "threshold", "exposure_confounder_rr", "confounder_outcome_rr"]
)
def test_malformed_numbers_have_safe_transportable_refusals(value, route) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        if route == "risk_ratio":
            study(risk_ratio=value)
        else:
            risk_ratio_sensitivity(study(), **{route: value})
    assert caught.value.code == "sensitivity.input.number"
    assert caught.value.context["field"] == route
    assert isinstance(caught.value.context["value"], (str, int, float, bool, type(None)))
    for restored in (deepcopy(caught.value), pickle.loads(pickle.dumps(caught.value))):
        assert restored.code == caught.value.code
        assert restored.context == caught.value.context
        assert str(restored)


@pytest.mark.parametrize(
    "field,supported",
    [
        ("estimator", "cohort_risk_ratio"),
        ("estimand", "population_risk_ratio"),
        ("effect_scale", "risk_ratio"),
        ("inference", "fixed_horizon"),
    ],
)
@pytest.mark.parametrize("shape", ["scalar", "one", "two"])
def test_array_declarations_cannot_bypass_admission_or_alias_metadata(
    field, supported, shape
) -> None:
    values = {
        "scalar": np.array(supported),
        "one": np.array([supported]),
        "two": np.array([supported, "unsupported"]),
    }
    value = values[shape]
    with pytest.raises(UnsupportedRequestError) as caught:
        study(**{field: value})
    assert caught.value.code == "sensitivity.study.unsupported"
    assert caught.value.context["field"] == field
    context = dict(caught.value.context)
    value[...] = "changed"
    assert dict(caught.value.context) == context
    assert pickle.loads(pickle.dumps(caught.value)).context == context


@pytest.mark.parametrize(
    "value",
    [
        np.array("above"),
        np.array(["above"]),
        np.array(["above", "below"]),
        _BrokenRepr(),
        10**10000,
    ],
    ids=["scalar_array", "one_array", "two_array", "broken_repr", "huge_integer"],
)
def test_invalid_direction_refuses_before_membership_or_aliasing(value) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        risk_ratio_sensitivity(study(), direction=value)
    assert caught.value.code == "sensitivity.decision.direction"
    assert deepcopy(caught.value).context == caught.value.context


@pytest.mark.parametrize(
    "value",
    [np.array([]), np.array(["age"]), _BrokenRepr(), 10**10000],
    ids=["empty_array", "array", "broken_repr", "huge_integer"],
)
def test_adjustment_declarations_refuse_without_unsafe_comparison(value) -> None:
    with pytest.raises(UnsupportedRequestError) as caught:
        study(adjustment_covariates=value)
    assert caught.value.code == "sensitivity.study.adjusted"
    assert pickle.loads(pickle.dumps(caught.value)).context == caught.value.context


def test_numpy_string_scalars_are_normalized_before_admission_and_retention() -> None:
    original = study(
        estimator=np.str_("cohort_risk_ratio"),
        estimand=np.str_("population_risk_ratio"),
        effect_scale=np.str_("risk_ratio"),
        inference=np.str_("fixed_horizon"),
        population=np.str_("cohort"),
    )
    result = risk_ratio_sensitivity(original, direction=cast(Any, np.str_("above")))
    for name in ("estimator", "estimand", "effect_scale", "inference", "population"):
        assert type(getattr(original, name)) is str
    assert type(result.direction) is str
    assert result.direction == "above"
    assert result.point_bound == original.risk_ratio


class _ComparisonTrap(str):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("comparison before normalization")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("comparison before normalization")


def test_string_subclasses_are_normalized_before_comparison() -> None:
    original = study(estimator=_ComparisonTrap("cohort_risk_ratio"))
    result = risk_ratio_sensitivity(original, direction=cast(Any, _ComparisonTrap("above")))
    assert type(result.study.estimator) is str
    assert result.study.estimator == "cohort_risk_ratio"
    assert type(result.direction) is str
    assert result.direction == "above"
    with pytest.raises(UnsupportedRequestError) as caught:
        study(estimator=_ComparisonTrap("iptw"))
    assert caught.value.code == "sensitivity.study.unsupported"
    assert caught.value.context["value"] == "iptw"
