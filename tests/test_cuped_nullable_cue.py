from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from increment import Analysis, MetricSpec
from increment.estimation.cuped import fit_cuped_multicovariate
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.semantics.models import CupedCovariate


def test_public_linear_cuped_accepts_nan_history_as_imputed_missingness():
    rows = []
    for arm, shift in (("control", 0.0), ("treatment", 1.0)):
        for index in range(24):
            prior = np.nan if index < 2 else float(index)
            rows.append(
                {
                    "unit": f"{arm}-{index}",
                    "arm": arm,
                    "outcome": float(index + shift),
                    "prior": prior,
                    "country": ("north", "south")[index % 2],
                }
            )
    frame = pd.DataFrame(rows)
    metric = MetricSpec(
        name="outcome",
        covariates=(
            CupedCovariate(name="prior", kind="continuous", missing="impute-indicator"),
            CupedCovariate(name="country", kind="categorical"),
        ),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )

    results = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="control", metrics=[metric]
    ).run()

    (row,) = results
    assert isinstance(row, LiftEstimate)
    expected = fit_cuped_multicovariate(
        frame["outcome"].to_numpy(),
        frame["arm"].to_list(),
        control="control",
        arm_order=("control", "treatment"),
        covariates={
            "prior": frame["prior"].to_numpy(),
            "country": frame["country"].to_numpy(dtype=object),
        },
        kinds={"prior": "continuous", "country": "categorical"},
        missing={"prior": "impute-indicator", "country": "refuse"},
    )
    assert row.require_lift().log_mean == pytest.approx(expected.relative("treatment").estimate)
    assert results.metadata is not None
    assert any(
        advisory.code == "estimation.cuped_linear.fit"
        for cell in results.metadata.cells
        for advisory in cell.advisories
    )
