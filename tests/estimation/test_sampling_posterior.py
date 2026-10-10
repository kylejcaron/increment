"""Sampling evidence and persisted posterior probabilities are distinct objects."""

import copy
import math
import pickle
from collections.abc import Sequence
from typing import Literal, TypedDict

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import estimate_lift
from increment.estimation.inference import Normal
from increment.estimation.priors import MixturePrior, StudentTPrior, mixture_posterior
from increment.estimation.results import LiftEstimate
from increment.semantics.models import MeanMetric, Metric
from tests.estimation._conversion_counts import CONVERSION_METRIC, arm_row, count_summary


class _MeanEstimateOptions(TypedDict):
    metrics: Sequence[Metric]
    control_group: str
    alternative: Literal["two-sided", "greater"]
    null_lift: float


class _ConversionEstimateOptions(TypedDict):
    metrics: Sequence[Metric]
    control_group: str


def identical_arms():
    return [
        arm_row(
            ArmStats.from_raw_sums(
                study_id="e", metric="rev", group_id=group, n=100, sum_y=1000, sum_y2=10100
            )
        )
        for group in ("control", "treatment")
    ]


@pytest.mark.parametrize(
    "prior",
    [
        Normal(mu=0.10, sigma=0.01),
        StudentTPrior(nu=4, scale=0.1),
        MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15)),
    ],
)
@pytest.mark.parametrize("alternative,null_lift", [("two-sided", 0), ("greater", 0.02)])
def test_prior_leaves_sampling_construction_unchanged(prior, alternative, null_lift):
    options: _MeanEstimateOptions = {
        "metrics": [MeanMetric(name="rev", entity="user", fact="rev")],
        "control_group": "control",
        "alternative": alternative,
        "null_lift": null_lift,
    }
    baseline = estimate_lift(summary=identical_arms(), **options)
    computation = estimate_lift(summary=identical_arms(), prior=prior, **options)
    (row,) = computation.results
    (reference,) = baseline.results
    assert row.lift is not None
    assert reference.lift is not None
    assert row.lift == reference.lift
    assert (row.reference_kind, row.reference_df) == (
        reference.reference_kind,
        reference.reference_df,
    )
    assert row.p_value() == reference.p_value()
    assert row.stat_sig() == reference.stat_sig()
    assert computation.evidence == baseline.evidence
    assert row.posterior_estimate is not None
    assert row.posterior_available is True
    if isinstance(prior, Normal):
        assert row.posterior_estimate != row.lift.value
    else:
        assert (row.posterior_lb, row.posterior_ub) != (row.lift.lb, row.lift.ub)
    assert (
        LiftEstimate.model_validate_json(row.model_dump_json()).chance_to_beat()
        == row.chance_to_beat()
    )

    if not isinstance(prior, Normal):
        assert row.lift is not None
        assert row.lift.log_mean is not None
        assert row.lift.log_se is not None
        posterior = mixture_posterior(row.lift.log_mean, row.lift.log_se, prior.components())
        assert row.posterior_components is not None
        assert row.chance_to_beat() == pytest.approx(posterior.survival(0.0))
        assert row.prob_beyond(0.02) == pytest.approx(posterior.survival(math.log1p(0.02)))
        assert row.prob_within(0.03) == pytest.approx(
            posterior.probability_between(math.log1p(-0.03), math.log1p(0.03))
        )
        assert row.risk_if_shipped() == pytest.approx(posterior.expected_negative_part(scale="log"))
        changed = row.model_copy(
            update={"lift": row.lift.model_copy(update={"value": -0.5, "lb": -0.9, "ub": -0.1})}
        )
        assert changed.chance_to_beat() == row.chance_to_beat()
        assert changed.prob_beyond(0.02) == row.prob_beyond(0.02)
        assert changed.risk_if_shipped() == row.risk_if_shipped()
        assert copy.deepcopy(row).posterior_components == row.posterior_components
        assert pickle.loads(pickle.dumps(row)).posterior_components == row.posterior_components


def test_sparse_conversion_preserves_exact_sampling_and_guard_reason():
    options: _ConversionEstimateOptions = {
        "metrics": [CONVERSION_METRIC],
        "control_group": "control",
    }
    baseline = estimate_lift(summary=count_summary(3, 12, 9, 12), **options)
    computation = estimate_lift(
        summary=count_summary(3, 12, 9, 12), prior=Normal(mu=0.10, sigma=0.01), **options
    )
    (row,) = computation.results
    (reference,) = baseline.results
    assert row.lift == reference.lift
    assert row.binomial_set == reference.binomial_set
    assert row.p_value() == pytest.approx(0.022660742879185826)
    assert not computation.failures
    assert row.posterior_available is False
    assert row.posterior_reason_code == "estimation.engine.lift_guard"
    assert row.posterior_reason_context is not None
    assert row.posterior_reason_context["reason"] == "delta_method_unreliable"
    assert row.chance_to_beat() is None


def test_no_prior_has_no_plugin_probability():
    (row,) = estimate_lift(
        [MeanMetric(name="rev", entity="user", fact="rev")],
        identical_arms(),
        control_group="control",
    ).results
    assert row.chance_to_beat() is None
    assert row.prob_beyond(0.01) is None
    assert row.prob_within(0.01) is None
    assert row.risk_if_shipped() is None


def test_normal_probability_reads_stored_posterior_not_sampling_interval():
    (row,) = estimate_lift(
        [MeanMetric(name="rev", entity="user", fact="rev")],
        identical_arms(),
        control_group="control",
        prior=Normal(mu=0.10, sigma=0.01),
    ).results
    assert row.chance_to_beat() == 0.9999999999999999
    assert row.lift is not None
    changed = row.model_copy(
        update={"lift": row.lift.model_copy(update={"value": -0.5, "lb": -0.9, "ub": -0.1})}
    )
    assert changed.chance_to_beat() == row.chance_to_beat()
    assert changed.prob_within(0.01) == row.prob_within(0.01)
    assert changed.risk_if_shipped() == row.risk_if_shipped()


@pytest.mark.parametrize(
    "prior",
    [
        Normal(mu=0.10, sigma=0.01),
        MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15)),
    ],
)
def test_analysis_frame_and_saved_results_preserve_sampling_and_posterior(prior):
    import pandas as pd

    from increment.analysis import Analysis
    from increment.estimation.readout_types import ReadoutResults
    from increment.frame import MetricSpec

    frame = pd.DataFrame(
        [
            {
                "unit": f"{group}-{i}",
                "group": group,
                "rev": 10 + ((i % 7) - 3) * 0.1,
            }
            for group in ("control", "treatment")
            for i in range(100)
        ]
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="group",
        control="control",
        metrics=[MetricSpec(name="rev", type="mean")],
    )
    results = analysis.run(prior=prior)
    (row,) = results
    assert isinstance(row, LiftEstimate)
    assert row.lift is not None
    assert row.posterior_estimate is not None
    p_value = row.p_value()
    assert p_value is not None
    expected = (row.lift.value, p_value, row.posterior_estimate)

    rows = results.to_frame(backend="pandas")
    assert isinstance(rows, pd.DataFrame)
    assert rows.loc[0, "posterior_available"]
    assert rows.loc[0, "posterior_estimate"] == row.posterior_estimate
    if isinstance(prior, Normal):
        assert pd.isna(rows.loc[0, "posterior_components"])
    else:
        assert isinstance(rows.loc[0, "posterior_components"], str)
        assert row.posterior_components is not None
        frame_row = LiftEstimate.model_validate(
            row.model_dump() | {"posterior_components": rows.loc[0, "posterior_components"]}
        )
        assert frame_row.posterior_components == row.posterior_components

    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    (saved_row,) = restored
    assert isinstance(saved_row, LiftEstimate)
    assert saved_row.lift is not None
    assert saved_row.posterior_estimate is not None
    saved_p_value = saved_row.p_value()
    assert saved_p_value is not None
    assert (saved_row.lift.value, saved_p_value, saved_row.posterior_estimate) == expected
    assert saved_row.posterior_components == row.posterior_components


@pytest.mark.parametrize(
    "components",
    [
        {"weights": (1.0,), "means": (0.0, 1.0), "sigmas": (1.0, 1.0)},
        {"weights": (float("nan"),), "means": (0.0,), "sigmas": (1.0,)},
        {"weights": (-0.1, 1.1), "means": (0.0, 1.0), "sigmas": (1.0, 1.0)},
        {"weights": (1.0,), "means": (0.0,), "sigmas": (0.0,)},
    ],
)
def test_invalid_stored_posterior_components_are_coded(components):
    prior = MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15))
    (row,) = estimate_lift(
        [MeanMetric(name="rev", entity="user", fact="rev")],
        identical_arms(),
        control_group="control",
        prior=prior,
    ).results
    with pytest.raises(InvalidRequestError) as exc_info:
        LiftEstimate.model_validate(row.model_dump() | {"posterior_components": components})
    assert exc_info.value.code == "estimation.results.lift.posterior_components_invalid"


def test_mixture_posterior_state_coupling_is_validated():
    prior = MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15))
    (row,) = estimate_lift(
        [MeanMetric(name="rev", entity="user", fact="rev")],
        identical_arms(),
        control_group="control",
        prior=prior,
    ).results
    with pytest.raises(InvalidRequestError) as missing:
        LiftEstimate.model_validate(row.model_dump() | {"posterior_components": None})
    assert missing.value.code == "estimation.results.lift.posterior_components_invalid"
    with pytest.raises(InvalidRequestError) as unavailable:
        LiftEstimate.model_validate(row.model_dump() | {"posterior_available": False})
    assert unavailable.value.code == "estimation.results.lift.posterior_components_invalid"
