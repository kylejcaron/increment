"""Quantile-specific sequential refusal before data access; fixed inference remains."""

from __future__ import annotations

import numpy as np
import pytest

from increment.estimation.quantile import estimate_quantile_lift
from increment.estimation.sequential import AlwaysValid


class _Rows:
    """Minimal per-unit source: the one method the estimator calls."""

    def __init__(self, frame):
        self._frame = frame

    def unit_frame(self, metric, *, covariates=()):
        return self._frame


def _rows(n_per_arm=600, seed=3, shift=1.2):
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    c = rng.lognormal(0.0, 0.5, size=n_per_arm)
    t = rng.lognormal(0.0, 0.5, size=n_per_arm) * shift
    return _Rows(
        pa.table(
            {
                "group_id": ["C"] * n_per_arm + ["T"] * n_per_arm,
                "y": np.concatenate([c, t]),
            }
        )
    )


def _metric():
    from increment.semantics.models import QuantileMetric

    return QuantileMetric(
        name="p90", entity="unit_id", fact="latency", aggregation="sum", quantile=0.9
    )


def test_quantile_sequential_refuses_before_unit_access():
    from increment.errors import CapabilityError
    from tests.sequential_cases import registration

    policy = AlwaysValid(registration=registration("gaussian"))

    class Unread:
        def unit_frame(self, *args, **kwargs):
            raise AssertionError("unsupported quantile route accessed source")

    with pytest.raises(CapabilityError) as raised:
        estimate_quantile_lift(Unread(), _metric(), "C", inference=policy)
    assert raised.value.code == "sequential.route.unsupported"


def test_fixed_quantile_remains_available():
    (fixed,) = estimate_quantile_lift(_rows(), _metric(), "C")
    assert fixed.inference == "fixed"
    assert fixed.require_lift().value > 0
    assert fixed.require_lift().lb is not None
    assert fixed.require_lift().ub is not None
