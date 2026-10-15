from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from increment import Analysis, CupedCovariate, Method, MetricSpec
from increment.errors import UnsupportedRequestError
from increment.estimation.results import LiftEstimate


def test_scalar_method_evidence_preserves_each_requested_family():
    rng = np.random.default_rng(417)
    n = 64
    a = rng.normal(size=2 * n)
    b = rng.normal(size=2 * n)
    frame = pd.DataFrame(
        {
            "unit": range(2 * n),
            "arm": ["control"] * n + ["treatment"] * n,
            "outcome": 20 + 2 * a - b + rng.normal(size=2 * n),
            "a": a,
            "b": b,
        }
    )
    methods = tuple(
        Method.cuped(name=name, covariates=CupedCovariate(name=name, kind="continuous"))
        for name in ("a", "b")
    )

    def run(selected):
        with Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    decision_method=selected[0],
                    sensitivity_methods=selected[1:],
                )
            ],
        ) as analysis:
            return {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}

    joint = run(methods)
    for method in methods:
        alone = run((method,))[method.name]
        assert joint[method.name].require_lift().value == pytest.approx(
            alone.require_lift().value, rel=1e-10
        )
        assert joint[method.name].require_lift().log_se == pytest.approx(
            alone.require_lift().log_se, rel=1e-10
        )


def test_source_without_method_evidence_refuses_before_reading():
    from increment.frame import from_unit_summary
    from increment.readouts import run

    source = from_unit_summary(
        pd.DataFrame(
            {
                "unit": range(8),
                "arm": ["control"] * 4 + ["treatment"] * 4,
                "outcome": [10.0, 12.0, 11.0, 14.0, 15.0, 12.0, 17.0, 14.0],
                "prior": [1.0, 2.0, 3.0, 4.0] * 2,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="prior", kind="continuous")
                ),
            )
        ],
    )

    class AggregateOnlySource:
        def __getattr__(self, name):
            if name == "moments_for_method":
                raise AttributeError(name)
            return getattr(source, name)

        def moments(self, *args, **kwargs):
            pytest.fail("unsupported route read aggregate rows")

        def unit_frame(self, *args, **kwargs):
            pytest.fail("unsupported route read unit rows")

    with pytest.raises(UnsupportedRequestError) as raised:
        run(cast(Any, AggregateOnlySource()))
    assert raised.value.code == "readout.scalar_method_evidence"
    assert raised.value.context["route"] == "AggregateOnlySource"
