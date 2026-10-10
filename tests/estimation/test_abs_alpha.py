"""``abs_alpha``: the alpha an additive interval was cut at, persisted with the interval."""

from __future__ import annotations

import copy
import json
import math
import pickle
from datetime import date
from typing import Any

import narwhals as nw
import pytest

from increment.breakout.estimates import BreakoutEstimate, DailyLiftEstimate
from increment.breakout.projection import to_frame
from increment.errors import CodedError, InvalidRequestError
from increment.estimation.armstats import ScoreStats, centered_row_from_raw_sums
from increment.estimation.engine import estimate_lift
from increment.estimation.inference import infer_ate, infer_lift
from increment.estimation.results import (
    JointContrastReference,
    LiftEstimate,
    open_bound_from_two_sided_at_target,
)
from increment.semantics.models import ConversionMetric, MeanMetric
from increment.tables import estimates_to_readout

_MODELS = {"primary": LiftEstimate, "breakout": BreakoutEstimate, "daily": DailyLiftEstimate}


def _lift(alternative="two-sided", alpha=0.05, **extra):
    return infer_lift(
        "m",
        "t",
        "unadjusted",
        0.1,
        0.03,
        0.03,
        alpha=alpha,
        alternative=alternative,
        method_role="decision",
        **extra,
    )


def _ate(alternative="two-sided", alpha=0.05, **extra):
    return infer_ate(
        "m",
        "t",
        "unadjusted",
        point=extra.pop("point", 0.1),
        scores=ScoreStats(metric="m", contrast="t", n=64, sum_psi=0.0, sum_psi2=2.56),
        alpha=alpha,
        alternative=alternative,
        method_role="decision",
        **extra,
    )


def _arm(n, mean, var, *, group_id, metric="m"):
    total = float(n) * mean
    return centered_row_from_raw_sums(
        {
            "experiment_id": "e",
            "metric": metric,
            "group_id": group_id,
            "n": float(n),
            "sum_y": total,
            "sum_y2": var * (n - 1) + total**2 / float(n),
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def _mean_row(control_mean, treatment_mean, treatment_var, **kwargs):
    (row,) = estimate_lift(
        [MeanMetric(name="m", entity="u", fact="m")],
        [
            _arm(400, control_mean, 4.0, group_id="control"),
            _arm(400, treatment_mean, treatment_var, group_id="treatment"),
        ],
        control_group="control",
        **kwargs,
    ).results
    return row


def _conversion_row(**kwargs):
    def binary(group_id, successes):
        return centered_row_from_raw_sums(
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": group_id,
                "n": 200.0,
                "successes": successes,
                "sum_y": float(successes),
                "sum_y2": float(successes),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": None,
                "sum_den2": None,
                "sum_yden": None,
            }
        )

    (row,) = estimate_lift(
        [ConversionMetric(name="m", entity="u", fact="m")],
        [binary("control", 60), binary("treatment", 96)],
        control_group="control",
        **kwargs,
    ).results
    return row


def _cluster_row_estimate(**kwargs):
    def cluster(group, center):
        totals = [center + 0.1 * ((i % 5) - 2) for i in range(60)]
        return centered_row_from_raw_sums(
            {
                "experiment_id": "e",
                "metric": "m",
                "group_id": group,
                "n": len(totals),
                "sum_y": sum(totals),
                "sum_y2": sum(g * g for g in totals),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": float(len(totals)),
                "sum_den2": float(len(totals)),
                "sum_yden": sum(totals),
            }
        )

    (row,) = estimate_lift(
        [MeanMetric(name="m", entity="u", fact="f", aggregation="sum")],
        [cluster("C", 5.0), cluster("T", 5.6)],
        control_group="C",
        cluster="store",
        **kwargs,
    ).results
    return row


def _winsor_row(alpha=0.05):
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    raw = WinsorRawState(
        metric="revenue",
        study_id="e",
        population="assigned",
        missingness="error",
        quantile=0.75,
        inference=WinsorInferenceSpec(method="joint-rank-projection-v1"),
        support=WinsorSupport(lower=0, upper=None, provenance="External finite oracle support"),
        arms=(
            RawArm(group_id="C", values=tuple(float(1 + (i * 7) % 11) for i in range(30))),
            RawArm(group_id="T", values=tuple(float(2 + (i * 5) % 11) for i in range(30))),
        ),
    )
    return estimate_winsor_lift(raw, "C", "T", alpha=alpha)


_JOINT = JointContrastReference(
    a=2.0, c=10.0, var_a=0.04, var_c=0.01, cov_ac=-0.005, kind="t", df=7.5
)
_JOINT_NO_VARIANCE = JointContrastReference(a=2.0, c=10.0, var_a=0.0, var_c=0.0, cov_ac=0.0)

#: producer -> (row, the alpha its additive interval was cut at, in ``Estimate.alpha``'s convention)
_PRODUCERS = {
    "infer_lift": lambda: (_lift(abs_diff=1.0, abs_se=0.2), 0.05),
    "infer_lift directional": lambda: (
        _lift("greater", 0.03, abs_diff=1.0, abs_se=0.2, abs_dof=7.0),
        0.06,
    ),
    "infer_lift without a sidecar": lambda: (_lift(), None),
    "infer_lift whose Welch reference leaves no additive interval": lambda: (
        _lift(abs_diff=1.0, abs_se=0.2, arm_ns=(40, 45)),
        None,
    ),
    "infer_ate": lambda: (_ate("less", 0.04, abs_diff=1.0, abs_se=0.2), 0.08),
    "infer_ate joint set": lambda: (
        _ate(point=2.0 / 10.0, joint_reference=_JOINT, abs_dof=7.5),
        0.05,
    ),
    "infer_ate joint set, directional": lambda: (
        _ate("greater", 0.03, point=2.0 / 10.0, joint_reference=_JOINT, abs_dof=7.5),
        0.06,
    ),
    "infer_ate joint set without additive variance": lambda: (
        _ate(point=0.2, joint_reference=_JOINT_NO_VARIANCE),
        None,
    ),
    "infer_ate unavailable relative interval": lambda: (
        _ate(
            "less",
            0.04,
            abs_diff=1.0,
            abs_se=0.2,
            abs_dof=7.5,
            relative_unavailable_reason="joint_covariance_indefinite",
        ),
        0.08,
    ),
    "estimate_lift mean": lambda: (_mean_row(10.0, 12.0, 4.0, alpha=0.04), 0.04),
    "estimate_lift mean, directional": lambda: (
        _mean_row(10.0, 12.0, 4.0, alpha=0.04, alternative="less"),
        0.08,
    ),
    "estimate_lift exact binomial": lambda: (_conversion_row(alpha=0.04), 0.04),
    "estimate_lift exact binomial, directional": lambda: (
        _conversion_row(alpha=0.04, alternative="greater"),
        0.08,
    ),
    "estimate_lift non-positive arm mean": lambda: (_mean_row(1.0, 0.0, 0.0, alpha=0.04), 0.04),
    "estimate_lift non-positive arm mean, directional": lambda: (
        _mean_row(1.0, 0.0, 0.0, alpha=0.04, alternative="greater"),
        0.08,
    ),
    "estimate_lift clustered joint set": lambda: (_cluster_row_estimate(alpha=0.04), 0.04),
    "estimate_lift clustered joint set, directional": lambda: (
        _cluster_row_estimate(alpha=0.04, alternative="greater"),
        0.08,
    ),
    "winsor confidence set": lambda: (_winsor_row(0.04), 0.04),
}


@pytest.mark.parametrize("producer", list(_PRODUCERS))
def test_every_producer_persists_the_alpha_of_the_additive_interval_it_cuts(producer):
    row, expected = _PRODUCERS[producer]()

    assert row.abs_alpha == expected
    assert (row.abs_alpha is None) == (row.abs_lb is None) == (row.abs_ub is None)
    if row.abs_alpha is not None and row.lift is not None and row.lift.alpha is not None:
        assert row.abs_alpha == row.lift.alpha
    if row.abs_alpha is not None and row.relative_confidence_set is not None:
        assert row.abs_alpha == row.relative_confidence_set.alpha_eff
    if row.abs_alpha is not None and row.binomial_set is not None:
        assert row.abs_alpha == row.binomial_set.alpha


def test_winsor_reinversion_carries_the_alpha_of_the_reinverted_additive_interval():
    row = _winsor_row(0.05).reintervalize(0.1)

    assert row.confidence_set is not None
    assert row.abs_lb is not None and row.abs_alpha == 0.1 == row.confidence_set.alpha


def test_open_bound_conversion_keeps_the_central_additive_alpha():
    row = _lift("greater", 0.03, abs_diff=1.0, abs_se=0.2, abs_dof=7.0)

    converted = open_bound_from_two_sided_at_target(row)

    assert (converted.abs_lb, converted.abs_ub) == (row.abs_lb, row.abs_ub)
    assert converted.abs_alpha == row.abs_alpha == 0.06


def _payload(model_name, row=None):
    model = _MODELS[model_name]
    row = _lift("greater", 0.03, abs_diff=1.0, abs_se=0.2, abs_dof=7.0) if row is None else row
    payload: dict[str, Any] = {k: v for k, v in row.model_dump().items() if k in model.model_fields}
    if model_name == "breakout":
        payload.update(dimension="country", dimension_value="US")
    elif model_name == "daily":
        payload.update(ds=date(2026, 1, 1))
    return model, payload


@pytest.mark.parametrize("model_name", list(_MODELS))
def test_abs_alpha_survives_json_pickle_and_deepcopy(model_name):
    model, payload = _payload(model_name)
    row = model.model_validate(payload)
    assert row.abs_alpha == 0.06

    for restored in (
        model.model_validate_json(row.model_dump_json()),
        pickle.loads(pickle.dumps(row)),
        copy.deepcopy(row),
    ):
        assert restored == row
        assert restored.abs_alpha == row.abs_alpha


@pytest.mark.parametrize("model_name", list(_MODELS))
def test_a_row_serialized_before_abs_alpha_existed_still_loads(model_name):
    model, payload = _payload(model_name)
    wire = json.loads(model.model_validate(payload).model_dump_json())
    wire.pop("abs_alpha")

    legacy = model.model_validate(wire)

    assert legacy.abs_alpha is None and legacy.abs_lb is not None
    assert model.model_validate_json(legacy.model_dump_json()) == legacy


@pytest.mark.parametrize("model_name", list(_MODELS))
def test_abs_alpha_requires_the_additive_interval_it_describes(model_name):
    model, payload = _payload(model_name, _lift())
    payload["abs_alpha"] = 0.05
    with pytest.raises(CodedError) as raised:
        model.model_validate(payload)
    assert raised.value.code == "estimation.results.lift.absolute_alpha_without_interval"


@pytest.mark.parametrize("model_name", list(_MODELS))
def test_abs_alpha_may_not_contradict_the_alpha_of_the_relative_interval(model_name):
    model, payload = _payload(model_name)
    payload["abs_alpha"] = 0.02
    with pytest.raises(CodedError) as raised:
        model.model_validate(payload)
    assert raised.value.code == "estimation.results.lift.absolute_alpha_mismatch"


@pytest.mark.parametrize("model_name", list(_MODELS))
@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_abs_alpha_must_be_a_probability(model_name, bad):
    model, payload = _payload(model_name)
    payload["abs_alpha"] = bad
    with pytest.raises(InvalidRequestError):
        model.model_validate(payload)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("model_name", ["primary", "breakout"])
def test_abs_alpha_is_a_nullable_float_column(backend, model_name):
    model, with_alpha = _payload(model_name)
    _, without = _payload(model_name, _lift())
    rows = [model.model_validate(with_alpha), model.model_validate(without)]

    frame = nw.from_native(to_frame(rows, backend=backend))

    assert frame.schema["abs_alpha"] == nw.Float64()
    column = frame["abs_alpha"]
    assert column[0] == pytest.approx(0.06)
    assert column.is_null().to_list() == [False, True]


def test_abs_alpha_reaches_the_readout_row():
    known = _lift("greater", 0.03, abs_diff=1.0, abs_se=0.2, abs_dof=7.0)

    (with_alpha,) = estimates_to_readout([known])
    (without,) = estimates_to_readout([_lift()])

    assert with_alpha["abs_alpha"] == 0.06
    assert without["abs_alpha"] is None


@pytest.mark.parametrize("model_name", list(_MODELS))
def test_a_joint_row_rejects_an_additive_alpha_that_contradicts_its_set(model_name):
    row = _ate(point=0.2, joint_reference=_JOINT, abs_dof=7.5)
    model, payload = _payload(model_name, row)
    assert model.model_validate_json(model.model_validate(payload).model_dump_json())
    payload["abs_alpha"] = math.nextafter(payload["abs_alpha"], 1.0)
    with pytest.raises(CodedError) as raised:
        model.model_validate(payload)
    assert raised.value.code == "estimation.results.joint.invalid_set"


@pytest.mark.parametrize(
    ("abs_alpha", "row", "code"),
    [
        (0.05, _lift(), "estimation.results.lift.absolute_alpha_without_interval"),
        (
            0.02,
            _lift("greater", 0.03, abs_diff=1.0, abs_se=0.2, abs_dof=7.0),
            "estimation.results.lift.absolute_alpha_mismatch",
        ),
    ],
)
def test_abs_alpha_refusals_survive_pickle_and_deepcopy(abs_alpha, row, code):
    payload = {**row.model_dump(), "abs_alpha": abs_alpha}
    with pytest.raises(CodedError) as raised:
        LiftEstimate.model_validate(payload)
    refusal = raised.value
    assert refusal.code == code

    for clone in (pickle.loads(pickle.dumps(refusal)), copy.deepcopy(refusal)):
        assert (clone.code, clone.context) == (refusal.code, refusal.context)
        assert str(clone) == str(refusal)
