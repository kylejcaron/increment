"""The historical quantile sequential cases are explicitly unsupported.

Woodruff order-statistic intervals are fixed-horizon evidence. The former
SE-mixture coverage claims cannot be inherited by the raw Beta/NIG/NIW runtime.
The original scientific case specifications below retain all seeds, looks,
replications and error budgets; every case now observes the required refusal
before outcomes are requested. No refused case is counted as coverage.
"""

import pytest

from increment import AlwaysValid
from increment.errors import CapabilityError
from increment.estimation.quantile import estimate_quantile_lift
from increment.semantics.models import QuantileMetric
from tests.sequential_cases import registration


class _UnreadStream:
    def __init__(self, n_trials, looks, seed, dgp):
        self.case = (n_trials, tuple(looks), seed, dgp)

    def unit_frame(self, metric, *, covariates=()):
        raise AssertionError(f"unsupported quantile accessed outcomes: {self.case}")


def _assert_refusal(*, n_trials, looks, alpha, seed, q=0.9, dgp="lognorm_0.5"):
    metric = QuantileMetric(
        name="p90", entity="unit_id", fact="latency", aggregation="sum", quantile=q
    )
    policy = AlwaysValid(registration=registration("gaussian", alpha=alpha))
    with pytest.raises(CapabilityError) as raised:
        estimate_quantile_lift(
            _UnreadStream(n_trials, looks, seed, dgp), metric, "C", alpha=alpha, inference=policy
        )
    assert raised.value.code == "sequential.route.unsupported"


# Historical acceptance bounds are retained as specifications, never as passed evidence.
HISTORICAL_AVAILABILITY_FLOOR = 0.90
HISTORICAL_SMOKE_FALSE_ALARM_CEILING = 0.5
HISTORICAL_ALPHA_SLACK = {400: 0.033, 150: 0.054}


def test_quantile_sequential_smoke_case_refuses_before_outcomes():
    _assert_refusal(n_trials=20, looks=[400, 800], alpha=0.05, seed=11)


@pytest.mark.compatibility("always-valid-quantile")
@pytest.mark.parameter_recovery
def test_quantile_sequential_historical_coverage_case_refuses_before_outcomes():
    _assert_refusal(
        n_trials=400,
        looks=[250, 500, 1000, 2000, 3000, 4000, 5000, 6000],
        alpha=0.05,
        seed=20260818,
    )


@pytest.mark.compatibility("always-valid-quantile")
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("q", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("dgp", ["lognorm_0.5", "lognorm_1.5", "pareto_1.5"])
def test_quantile_sequential_historical_tail_grid_refuses_before_outcomes(q, dgp):
    _assert_refusal(
        n_trials=150,
        looks=[250, 500, 1000, 2000, 3000, 4000, 5000, 6000],
        alpha=0.05,
        seed=20260820,
        q=q,
        dgp=dgp,
    )
