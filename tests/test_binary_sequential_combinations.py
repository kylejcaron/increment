"""Every method axis has a named outcome for automatic binary sequential monitoring."""

from __future__ import annotations

import pytest

from increment import InferenceSpec
from increment.errors import CodedError
from tests.binary_sequential_cases import axis_run

# axis, kind, how the axis is requested, expected ("supported" or a refusal code), reason
CASES = [
    ("clustered", "always_valid", {"cluster": "household"}, "sequential.route.unsupported", None),
    (
        "clustered",
        "asymptotic_mean",
        {"cluster": "household"},
        "sequential.route.unsupported",
        None,
    ),
    (
        "cuped_in_experiment",
        "always_valid",
        {"cuped": True},
        "sequential.transform.unpredictable",
        None,
    ),
    ("cuped_pre_period", "asymptotic_mean", {"cuped": True, "pre_period": True}, "supported", None),
    (
        "percentile_winsor",
        "asymptotic_mean",
        {"winsor_percentile": 0.99},
        "frame.metric.winsorization_applies_type",
        None,
    ),
    (
        "quantile_metric",
        "always_valid",
        {"metric_type": "quantile"},
        "definition.inference.always_valid_metric_type",
        None,
    ),
    ("multiplicity_family", "always_valid", {"secondaries": 2}, "supported", None),
    ("multiplicity_family", "asymptotic_mean", {"secondaries": 2}, "supported", None),
    ("breakout", "always_valid", {"breakout": "segment"}, "sequential.route.unsupported", None),
    ("breakout", "asymptotic_mean", {"breakout": "segment"}, "sequential.route.unsupported", None),
    (
        "predeclared_segments",
        "always_valid",
        {"breakout": "segment", "segments": ("a", "b", "c")},
        "supported",
        None,
    ),
    (
        "predeclared_segments",
        "asymptotic_mean",
        {"breakout": "segment", "segments": ("a", "b", "c")},
        "supported",
        None,
    ),
    (
        "predeclared_segments_multiplicity",
        "always_valid",
        {"breakout": "segment", "segments": ("a", "b"), "secondaries": 2},
        "supported",
        None,
    ),
    (
        "predeclared_segments_cuped_pre_period",
        "asymptotic_mean",
        {"breakout": "segment", "segments": ("a", "b"), "cuped": True, "pre_period": True},
        "supported",
        None,
    ),
    (
        "predeclared_segments_cuped_in_experiment",
        "always_valid",
        {"breakout": "segment", "segments": ("a", "b"), "cuped": True},
        "sequential.transform.unpredictable",
        None,
    ),
    (
        "predeclared_segments_clustered",
        "always_valid",
        {"breakout": "segment", "segments": ("a", "b"), "cluster": "household"},
        "sequential.route.unsupported",
        None,
    ),
    (
        "predeclared_segments_quantile",
        "always_valid",
        {"breakout": "segment", "segments": ("a", "b"), "metric_type": "quantile"},
        "definition.inference.always_valid_metric_type",
        None,
    ),
    (
        "predeclared_segments_percentile_winsor",
        "asymptotic_mean",
        {"breakout": "segment", "segments": ("a", "b"), "winsor_percentile": 0.99},
        "frame.metric.winsorization_applies_type",
        None,
    ),
    (
        "switchback",
        "always_valid",
        {"switchback": True},
        "source.frame.switchback.plan",
        "unsupported_inference",
    ),
    (
        "switchback",
        "asymptotic_mean",
        {"switchback": True},
        "source.frame.switchback.plan",
        "unsupported_inference",
    ),
]


@pytest.mark.parametrize(
    ("axis", "kind", "axes", "expected", "reason"),
    CASES,
    ids=[f"{axis}-{kind}" for axis, kind, *_ in CASES],
)
def test_binary_sequential_axis_has_a_named_outcome(axis, kind, axes, expected, reason):
    run = axis_run(InferenceSpec(kind=kind), seed=7, **axes)
    if expected == "supported":
        rows = run()
        assert rows and all(row.inference == kind for row in rows)
        return
    with pytest.raises(CodedError) as raised:
        run()
    assert raised.value.code == expected
    if reason is not None:
        assert raised.value.context["reason"] == reason
