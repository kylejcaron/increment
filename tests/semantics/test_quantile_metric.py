import pytest
from pydantic import ValidationError

from increment.semantics.models import QuantileMetric


def test_quantile_metric_holds_q():
    m = QuantileMetric(name="p90_latency", entity="user", fact="latency", quantile=0.9)
    assert m.type == "quantile"
    assert m.quantile == 0.9


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.5, 1.5])
def test_quantile_bounds_enforced(bad_q):
    with pytest.raises(ValidationError):
        QuantileMetric(name="m", entity="user", fact="f", quantile=bad_q)
