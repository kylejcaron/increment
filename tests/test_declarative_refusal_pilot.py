from __future__ import annotations

import math
from typing import Any, cast

import pytest

from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.power import Baseline


def _baseline(**changes: float | None) -> Baseline:
    values: dict[str, object] = {"mean": 2.0, "var": 3.0}
    values.update(cast(dict[str, object], changes))
    return Baseline.model_validate(values)


def _assert_context(actual, expected):
    """Check structured values without pinning human guidance wording."""
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key != "route":
            assert actual[key] == value


def _summary_source(*, metric=None, design=None, uptake=None):
    from tests.test_cuped_linear_acceptance import _linear_metric, _summary_source

    return _summary_source(metric=metric or _linear_metric(), design=design, uptake=uptake)


def _daily(source):
    from increment.readouts import daily

    return daily(cast(Any, source))


@pytest.mark.parametrize(
    ("hazard", "code", "context"),
    [
        ("winsorization", "estimation.engine.cuped_linear_winsorization", {"metric": "revenue"}),
        ("observational", "estimation.engine.cuped_linear_observational", {"metric": "revenue"}),
        ("encouragement", "estimation.engine.cuped_linear_encouragement", {"metric": "revenue"}),
    ],
)
def test_cuped_linear_metric_refusal_context_via_public_daily(hazard, code, context):
    from increment.semantics.design import (
        AdjustmentSet,
        Encouragement,
        ExclusionRestriction,
        Observational,
        UptakeSpec,
    )
    from tests.test_cuped_linear_acceptance import _linear_metric

    metric = _linear_metric(
        winsorization={"upper_percentile": 0.9} if hazard == "winsorization" else None
    )
    design = (
        Observational(control_group="control", adjustment=AdjustmentSet(covariates=("pre",)))
        if hazard == "observational"
        else Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake"),
            exclusion_restriction=ExclusionRestriction(
                justification="The test-only encouragement affects revenue through uptake."
            ),
        )
        if hazard == "encouragement"
        else None
    )
    source = _summary_source(
        metric=metric, design=design, uptake="uptake" if hazard == "encouragement" else None
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        _daily(source)
    assert raised.value.code == code
    assert raised.value.context == context
    assert source.read_calls == 0


def test_cuped_linear_same_metric_uses_rule_order_on_public_daily():
    from increment.semantics.design import AdjustmentSet, Observational
    from tests.test_cuped_linear_acceptance import _linear_metric

    source = _summary_source(
        metric=_linear_metric(winsorization={"upper_percentile": 0.9}),
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("pre",))
        ),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        _daily(source)
    assert raised.value.code == "estimation.engine.cuped_linear_winsorization"
    assert raised.value.context == {"metric": "revenue"}
    assert source.read_calls == 0


def test_cuped_linear_metric_order_precedes_rule_order_on_public_entrypoint():
    import pyarrow as pa

    from increment import Analysis
    from increment.semantics.design import AdjustmentSet, Observational
    from tests.test_cuped_linear_acceptance import _linear_metric

    rows = pa.table(
        {
            "unit": [f"u{i}" for i in range(16)],
            "group": ["control"] * 8 + ["treatment"] * 8,
            "revenue": [float(i + 1) for i in range(16)],
            "date": ["2025-01-01"] * 16,
            "other": [float(2 * i + 1) for i in range(16)],
            "pre": ["a", "b"] * 8,
        }
    )
    metrics = [
        _linear_metric(),
        _linear_metric(winsorization={"upper_percentile": 0.9}).model_copy(
            update={"name": "other"}
        ),
    ]
    analysis = Analysis.from_unit_panel(
        rows,
        unit="unit",
        group="group",
        date="date",
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("pre",))
        ),
        metrics=metrics,
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run()
    assert raised.value.code == "estimation.engine.cuped_linear_observational"
    assert raised.value.context == {"metric": "revenue"}


def test_non_linear_metric_has_no_cuped_linear_refusal():
    from increment import MetricSpec
    from increment.readouts import run as run_readout

    source = _summary_source(metric=MetricSpec(name="revenue"))
    row = next(row for row in run_readout(cast(Any, source)) if row.group_id == "treatment")
    assert row.require_lift().value == pytest.approx((12.5 - 4.5) / 4.5)


@pytest.mark.grid("cross inf and nan with every float field to pin refusal parity")
@pytest.mark.parametrize(
    ("field", "code", "context"),
    [
        ("mean", "power.finite", {"name": "mean"}),
        ("var", "power.finite", {"name": "var"}),
        ("cuped_rho", "power.baseline.cuped_rho", {}),
        ("compliance", "power.baseline.compliance", {}),
        ("trigger_rate", "power.baseline.trigger_rate", {}),
        ("icc", "power.baseline.icc", {}),
        ("cluster_icc", "power.baseline.cluster_icc", {}),
        ("cluster_participation", "power.finite", {"name": "cluster_participation"}),
        ("avg_cluster_size", "power.finite", {"name": "avg_cluster_size"}),
        ("cluster_size_cv", "power.finite", {"name": "cluster_size_cv"}),
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_baseline_nonfinite_values_keep_observed_refusal(field, code, context, value):
    with pytest.raises(InvalidRequestError) as raised:
        _baseline(**{field: value})
    assert raised.value.code == code
    expected = {**context, "value": value} if context else {}
    assert raised.value.context.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual = raised.value.context[key]
        if isinstance(expected_value, float) and math.isnan(expected_value):
            assert isinstance(actual, float)
            assert math.isnan(actual)
        else:
            assert actual == expected_value


@pytest.mark.parametrize(
    ("field", "valid", "invalid", "code", "context"),
    [
        ("mean", 1e-12, 0.0, "power.baseline.mean", {}),
        ("var", 1e-12, 0.0, "power.baseline.var_zero_variance", {}),
        ("cuped_rho", math.nextafter(1.0, 0.0), 1.0, "power.baseline.cuped_rho", {}),
        ("compliance", 1.0, 0.0, "power.baseline.compliance", {}),
        ("trigger_rate", 1.0, 0.0, "power.baseline.trigger_rate", {}),
        ("icc", 0.0, 1.0, "power.baseline.icc", {}),
        ("cluster_icc", 0.0, 1.0, "power.baseline.cluster_icc", {}),
        (
            "cluster_participation",
            1.0,
            0.0,
            "power.baseline.cluster_participation",
            {"cluster_participation": 0.0, "route": None},
        ),
        (
            "avg_cluster_size",
            1.0,
            math.nextafter(1.0, 0.0),
            "power.baseline.avg_cluster_size",
            {"avg_cluster_size": math.nextafter(1.0, 0.0)},
        ),
        ("cluster_size_cv", 0.0, -math.ulp(1.0), "power.baseline.cluster_size_cv", {}),
    ],
)
def test_baseline_range_boundaries_keep_acceptance_and_context(
    field, valid, invalid, code, context
):
    assert getattr(_baseline(**{field: valid}), field) == valid
    with pytest.raises(InvalidRequestError) as raised:
        _baseline(**{field: invalid})
    assert raised.value.code == code
    _assert_context(raised.value.context, context)


def test_baseline_participation_cross_field_precedes_size_range():
    with pytest.raises(InvalidRequestError) as raised:
        _baseline(cluster_participation=0.5, avg_cluster_size=0.5)
    assert raised.value.code == "power.baseline.cluster_participation_untriggered"
    _assert_context(
        raised.value.context,
        {"trigger_rate": 1.0, "cluster_participation": 0.5, "route": None},
    )


def test_baseline_participation_cross_field_precedes_cv_range():
    with pytest.raises(InvalidRequestError) as raised:
        _baseline(cluster_participation=0.5, cluster_size_cv=-0.1)
    assert raised.value.code == "power.baseline.cluster_participation_untriggered"


def test_baseline_analyzed_mean_keeps_numeric_context():
    with pytest.raises(InvalidRequestError) as analyzed:
        _baseline(avg_cluster_size=1.0, trigger_rate=0.5, cluster_participation=1.0)
    assert analyzed.value.code == "power.baseline.cluster_analyzed_mean"
    _assert_context(
        analyzed.value.context,
        {
            "avg_cluster_size": 1.0,
            "trigger_rate": 0.5,
            "cluster_participation": 1.0,
            "analyzed_mean": 0.5,
            "route": None,
        },
    )


def test_baseline_cluster_icc_keeps_numeric_context():
    with pytest.raises(InvalidRequestError) as clustered:
        _baseline(avg_cluster_size=1.0, cluster_icc=0.2)
    assert clustered.value.code == "power.baseline.cluster_icc_avg"
    _assert_context(
        clustered.value.context,
        {"cluster_icc": 0.2, "analyzed_mean": 1.0, "route": None},
    )


@pytest.mark.parametrize(
    ("field", "valid", "invalid", "code"),
    [
        ("cuped_rho", math.nextafter(-1.0, 0.0), -1.0, "power.baseline.cuped_rho"),
        (
            "compliance",
            math.nextafter(1.0, 0.0),
            math.nextafter(1.0, math.inf),
            "power.baseline.compliance",
        ),
        (
            "trigger_rate",
            math.nextafter(1.0, 0.0),
            math.nextafter(1.0, math.inf),
            "power.baseline.trigger_rate",
        ),
        ("icc", math.nextafter(0.0, 1.0), -math.ulp(1.0), "power.baseline.icc"),
        ("cluster_icc", math.nextafter(0.0, 1.0), -math.ulp(1.0), "power.baseline.cluster_icc"),
        (
            "cluster_participation",
            math.nextafter(1.0, 0.0),
            math.nextafter(1.0, math.inf),
            "power.baseline.cluster_participation",
        ),
    ],
)
def test_baseline_opposite_range_boundaries(field, valid, invalid, code):
    assert getattr(_baseline(**_boundary_inputs(field, valid)), field) == valid
    with pytest.raises(InvalidRequestError) as raised:
        _baseline(**_boundary_inputs(field, invalid))
    assert raised.value.code == code


def _boundary_inputs(field, value):
    changes = {field: value}
    if field == "cluster_icc":
        changes["avg_cluster_size"] = 2.0
    elif field == "cluster_participation":
        changes.update(trigger_rate=0.5, avg_cluster_size=2.0)
    return changes
