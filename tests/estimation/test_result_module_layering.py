"""Result classes sit above the helpers their methods call and below their producers."""

from __future__ import annotations

from pathlib import Path

import pytest

from increment.breakout.estimates import LiftEstimates
from increment.estimation import results
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import estimate_lift
from increment.estimation.readout_types import ReadoutResults
from increment.semantics.models import MeanMetric
from scripts.check_deferred_imports import runtime_graph
from tests.estimation._conversion_counts import arm_row

PACKAGE = Path(__file__).resolve().parents[2] / "increment"
RESULT_MODULES = frozenset(
    f"increment.estimation.{name}" for name in ("results", "readout_types", "contrast_results")
)
# Modules whose functions a result method calls: they read rows and return leaf values.
HELPER_MODULES = (
    "_fcr_reinterval",
    "winsor",
    "_winsor_permutation",
    "_fixed_horizon",
    "_result_types",
    "_readout_refusals",
    "priors",
    "diagnostics",
)


def _runtime_closure(graph: dict[str, set[str]], start: str) -> set[str]:
    seen, pending = {start}, [start]
    while pending:
        for target in graph.get(pending.pop(), ()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def test_helper_modules_sit_below_the_result_modules() -> None:
    graph = runtime_graph(PACKAGE)
    reached = {
        helper: sorted(_runtime_closure(graph, f"increment.estimation.{helper}") & RESULT_MODULES)
        for helper in HELPER_MODULES
    }
    assert reached == {helper: [] for helper in HELPER_MODULES}


# reissue(0.01) / reintervalize(0.01) of the fixture row, captured at base commit daca1c61.
BASE_REISSUE = {
    "lift_lb": -0.014302265868216332,
    "lift_ub": 0.2726010789756891,
    "lift_alpha": 0.01,
    "abs_lb": -0.1481211550268895,
    "abs_ub": 2.5481211550268883,
    "abs_alpha": 0.01,
    "allocation_alpha": 0.01,
}


@pytest.fixture(scope="module")
def engine_row() -> results.LiftEstimate:
    arms = [
        arm_row(
            ArmStats.from_raw_sums(
                study_id="e", metric="rev", group_id=group, n=n, sum_y=sum_y, sum_y2=sum_y2
            )
        )
        for group, n, sum_y, sum_y2 in (("control", 40, 400, 4200), ("treatment", 50, 560, 6600))
    ]
    (row,) = estimate_lift(
        [MeanMetric(name="rev", entity="user", fact="rev")], arms, control_group="control"
    ).results
    return row


@pytest.mark.parametrize("origin", ["engine", "restored"])
@pytest.mark.parametrize("operation", ["reissue", "reintervalize"])
def test_estimator_and_restored_results_are_the_canonical_class(
    engine_row: results.LiftEstimate, origin: str, operation: str
) -> None:
    row = (
        engine_row
        if origin == "engine"
        else ReadoutResults.model_validate_json(LiftEstimates([engine_row]).model_dump_json())[0]
    )
    assert type(row) is results.LiftEstimate

    reissued = getattr(row, operation)(0.01)

    assert type(reissued) is results.LiftEstimate
    assert reissued.lift is not None
    observed = {
        "lift_lb": reissued.lift.lb,
        "lift_ub": reissued.lift.ub,
        "lift_alpha": reissued.lift.alpha,
        "abs_lb": reissued.abs_lb,
        "abs_ub": reissued.abs_ub,
        "abs_alpha": reissued.abs_alpha,
        "allocation_alpha": reissued.allocation_alpha,
    }
    assert observed == pytest.approx(BASE_REISSUE, rel=1e-12, abs=0.0)
