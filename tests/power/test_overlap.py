"""Independent checks for prospective joint-assignment and paired-outcome draws."""

import json
from decimal import Decimal
from fractions import Fraction
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from increment.errors import CodedError
from increment.power.overlap import prepare_overlap_scenario
from increment.semantics.overlap import CellId, OverlapPlan
from tests.oracles.overlap_design import (
    contrast_matrix,
    occupancy_probability,
    planned_se_at_counts,
    regress_raw_trials,
    two_point_event_probability,
)

_CELLS: tuple[CellId, ...] = ("00", "10", "01", "11")
_GOLDEN_COUNTS = ((1, 1, 5, 2), (1, 0, 2, 6), (1, 0, 4, 4), (2, 0, 5, 2), (2, 1, 4, 2))
_GOLDEN_SUCCESSES = ((0, 1, 4, 1), (0, 0, 1, 3), (0, 0, 3, 2), (1, 0, 5, 1), (2, 1, 2, 1))


@pytest.fixture
def plan() -> OverlapPlan:
    path = Path(__file__).parents[1] / "fixtures" / "overlap" / "contract_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["evidence"]["plan"]
    raw["a"].update(off_weight=3, on_weight=1)
    raw["b"].update(off_weight=1, on_weight=3)
    raw["metrics"] = [
        {
            "name": "value",
            "unit": "points",
            "definition_sha256": "1" * 64,
            "kind": "mean",
            "role": "primary",
            "better": "higher",
        },
        {
            "name": "event",
            "unit": "risk",
            "definition_sha256": "2" * 64,
            "kind": "binary_mean",
            "role": "secondary",
            "better": "higher",
        },
    ]
    raw["claims"] = [{"metric": "value", "contrast": "joint_launch", "alternative": "benefit"}]
    return OverlapPlan.model_validate(raw)


@pytest.fixture
def outcomes() -> dict[CellId, pd.DataFrame]:
    return {cell: pd.DataFrame({"event": [1, 0, 1], "value": [2.0, 0.0, 2.0]}) for cell in _CELLS}


def test_random_counts_not_fixed(plan, outcomes):
    scenario = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    assert scenario.expected_counts == (27 / 16, 9 / 16, 81 / 16, 27 / 16)
    draws = tuple(scenario.draws(replications=5, seed=718))
    assert tuple(draw.counts for draw in draws) == _GOLDEN_COUNTS


def test_tiny_arms_survive_near_unit_mixture(plan, outcomes):
    limit = 2**53 - 1
    weights = {"off_weight": limit, "on_weight": 1}
    design = plan.model_copy(
        update={
            "a": plan.a.model_copy(update=weights),
            "b": plan.b.model_copy(update=weights),
        }
    )
    constants = {cell: frame.iloc[:1] for cell, frame in outcomes.items()}
    scenario = prepare_overlap_scenario(design, n_target=limit, cell_outcomes=constants)
    assert scenario.expected_counts[3] == float(Fraction(limit, (limit + 1) ** 2))
    draw = next(scenario.draws(replications=1, seed=118))
    assert draw.counts == (limit - 3, 3, 0, 0)


def test_paired_metrics_stay_paired(plan, outcomes):
    scenario = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    for draw, successes in zip(
        scenario.draws(replications=5, seed=718), _GOLDEN_SUCCESSES, strict=True
    ):
        for n, k, value, event in zip(
            draw.counts, successes, draw.cells[:4], draw.cells[4:], strict=True
        ):
            assert event.n_success == k
            assert value.n_success is None
            if n == 0:
                assert event.anchor is event.mean_delta is event.m2 is None
                assert value.anchor is value.mean_delta is value.m2 is None
            else:
                assert (
                    event.anchor is not None
                    and event.mean_delta is not None
                    and event.m2 is not None
                )
                assert (event.anchor, event.mean_delta, event.m2) == (
                    (1.0, 0.0, 0.0) if k == n else (0.0, k / n, k * (n - k) / n)
                )
                assert (value.anchor, value.mean_delta, value.m2) == pytest.approx(
                    (2 * event.anchor, 2 * event.mean_delta, 4 * event.m2)
                )


def test_scenario_id_ignores_order_not_multiplicity(plan, outcomes):
    first = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    reordered = {
        cell: pa.Table.from_pandas(data.iloc[::-1, ::-1].astype({"event": bool}))
        for cell, data in reversed(tuple(outcomes.items()))
    }
    second = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=reordered)
    assert second.scenario_id == first.scenario_id
    assert tuple(second.draws(replications=5, seed=718)) == tuple(
        first.draws(replications=5, seed=718)
    )
    reordered["00"] = reordered["00"].slice(0, 2)
    assert (
        prepare_overlap_scenario(plan, n_target=9, cell_outcomes=reordered).scenario_id
        != first.scenario_id
    )


def test_zero_support_is_in_denominator(plan, outcomes):
    scenario = prepare_overlap_scenario(plan, n_target=0, cell_outcomes=outcomes)
    draws = tuple(scenario.draws(replications=5, seed=718))
    assert len(draws) == 5
    assert all(draw.counts == (0, 0, 0, 0) for draw in draws)
    assert all(
        cell.n_observed == 0 and cell.anchor is None for draw in draws for cell in draw.cells
    )


def test_planned_se_at_matched_counts(plan, outcomes):
    scenario = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    draw = next(scenario.draws(replications=1, seed=718))
    # Given 5 draws with four highs in 01 and 2 draws with one high in 11,
    # the sample SDs are sqrt(4/5) and sqrt(2) respectively.
    weights = (0.0, 0.0, -1.0, 1.0)
    sigma = (0.0, 0.0, sqrt(4 / 5), sqrt(2))
    assert planned_se_at_counts(weights, draw.counts, sigma) == pytest.approx(sqrt(29) / 5)
    b_only, both = draw.cells[2].m2, draw.cells[3].m2
    assert b_only is not None and both is not None
    fitted = sqrt(b_only / (5 * 4) + both / 2)
    assert fitted == pytest.approx(sqrt(29) / 5)
    assert planned_se_at_counts((-1, 1, 0, 0), (1, 0, 5, 2), sigma) is None


def test_small_n_oracle_distinguishes_occupancy_and_outcome_support():
    p = (3 / 16, 1 / 16, 9 / 16, 3 / 16)
    assert occupancy_probability(4, p, (1, 1, 1, 1)) == pytest.approx(float(Fraction(243, 8192)))
    # Four draws; observed variation in both required cells forces each of the
    # four corresponding joint outcomes to appear once (4! orderings).
    probability = two_point_event_probability(
        4, p, (2 / 3,) * 4, lambda n, k: all(0 < k[j] < n[j] for j in (0, 3))
    )
    assert probability == pytest.approx(float(Fraction(3, 2048)))
    benefit = two_point_event_probability(
        4, p, (2 / 3,) * 4, lambda n, k: n[0] == n[3] == 2 and k[3] > k[0]
    )
    assert benefit == pytest.approx(float(Fraction(9, 4096)))


def test_raw_regression_reference_uses_full_interaction():
    assignments = np.repeat(np.arange(4, dtype=np.int64), 32)[None, :]
    centers = np.asarray((0.0, 1.0, 0.0, 3.0))
    state = centers[assignments] + np.tile((-1.0, 1.0), 64)
    ref = regress_raw_trials(
        assignments,
        state[..., None],
        contrast=contrast_matrix(0.25, 0.75),
        binary_metrics=(),
        raw_boundary=np.zeros((1, 8)),
        direction=np.zeros((1, 8), dtype=np.int64),
        alpha=0.05,
    )
    assert ref.estimate[0, 0, 7] == 2
    assert ref.se[0, 0, 7] == pytest.approx(sqrt(4 / 31))
    assert ref.ready[0, 0, 7] and ref.favorable[0, 0, 7]


@pytest.mark.parametrize(
    "defect", ["absent_cell", "empty_cell", "extra_column", "binary_fraction", "nonfinite"]
)
def test_entire_scenario_is_checked_even_without_any_entrants(plan, outcomes, defect):
    changed: dict[CellId, Any] = dict(outcomes)
    if defect == "absent_cell":
        del changed["11"]
    elif defect == "empty_cell":
        changed["11"] = outcomes["11"].iloc[:0]
    elif defect == "extra_column":
        changed["11"] = outcomes["11"].assign(undeclared=0)
    elif defect == "binary_fraction":
        changed["11"] = outcomes["11"].assign(event=0.5)
    else:
        changed["11"] = outcomes["11"].assign(value=float("inf"))
    with pytest.raises(CodedError) as caught:
        prepare_overlap_scenario(plan, n_target=0, cell_outcomes=changed)
    assert caught.value.code == "overlap.plan.invalid"
    assert caught.value.context["field"] == "cell_outcomes"


def test_binary_validation_precedes_float_rounding(plan, outcomes):
    precise: dict[CellId, pd.DataFrame | pl.DataFrame] = dict(outcomes)
    precise["11"] = pl.DataFrame(
        {
            "value": [2.0],
            "event": pl.Series([Decimal("1.000000000000000001")], dtype=pl.Decimal(30, 18)),
        }
    )
    with pytest.raises(CodedError) as caught:
        prepare_overlap_scenario(plan, n_target=0, cell_outcomes=precise)
    assert caught.value.context == {
        "field": "cell_outcomes",
        "reason": "nonbinary_metric",
        "cell": "11",
    }


@pytest.mark.parametrize(
    "controls", [{"replications": 0, "seed": 718}, {"replications": 1, "seed": True}]
)
def test_draw_controls_cannot_silently_change_the_run(plan, outcomes, controls):
    scenario = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    with pytest.raises(CodedError) as caught:
        tuple(scenario.draws(**controls))
    assert caught.value.code == "overlap.plan.invalid"


def test_scaled_ss_representable_when_raw_square_overflows(plan, outcomes):
    outcomes["01"] = outcomes["01"].assign(value=[1.4e154, 0.0, 1.4e154])
    scenario = prepare_overlap_scenario(plan, n_target=9, cell_outcomes=outcomes)
    cell = next(scenario.draws(replications=1, seed=718)).cells[2]
    assert cell.mean_delta == pytest.approx(1.12e154)
    assert cell.m2 == pytest.approx(1.568e308)
