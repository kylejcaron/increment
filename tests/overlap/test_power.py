"""Planning matches runtime: analytic contrast SE at the expected allocation equals the
estimator's SE when each cell holds exactly ``N * pi_k`` units with sample variance
``sigma_k^2``; the MDE inverts the power function; planning inputs are refused with codes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from increment.errors import CodedError
from increment.overlap import CONTRASTS, estimate_overlap_effects
from increment.power import (
    overlap_contrast_power,
    overlap_contrast_se,
    overlap_minimum_detectable_effect,
    simulate_overlap_power,
)
from tests.overlap.fixtures import DESIGN, from_frame, unit_frame

N_TOTAL = 800  # pi = 0.375, 0.375, 0.125, 0.125 -> 300, 300, 100, 100
SDS = {(0, 0): 1.0, (1, 0): 2.0, (0, 1): 1.5, (1, 1): 3.0}


def _exact_cell(n: int, mean: float, sd: float, rng: np.random.Generator) -> list[float]:
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return list(mean + sd * z)


@pytest.mark.parametrize("contrast", CONTRASTS)
def test_planning_se_equals_runtime_se_at_matched_inputs(contrast: str) -> None:
    rng = np.random.default_rng(5)
    sizes = {(0, 0): 300, (1, 0): 300, (0, 1): 100, (1, 1): 100}
    frame = unit_frame({c: _exact_cell(n, 0.0, SDS[c], rng) for c, n in sizes.items()})
    runtime = estimate_overlap_effects(from_frame(frame, DESIGN), contrasts=(contrast,)).rows[0]
    planned = overlap_contrast_se(DESIGN, contrast, n_total=N_TOTAL, sd=SDS)
    assert runtime.se == pytest.approx(planned, rel=1e-12)


@pytest.mark.parametrize("contrast", ("interaction", "joint_launch", "marginal_second"))
@pytest.mark.parametrize("family_size", (1, 4))
def test_mde_inverts_power(contrast: str, family_size: int) -> None:
    mde = overlap_minimum_detectable_effect(
        DESIGN, contrast, n_total=N_TOTAL, sd=2.0, power=0.8, family_size=family_size
    )
    power = overlap_contrast_power(
        DESIGN, contrast, effect=mde, n_total=N_TOTAL, sd=2.0, family_size=family_size
    )
    assert power == pytest.approx(0.8, abs=1e-9)
    assert overlap_contrast_power(
        DESIGN, contrast, effect=-mde, n_total=N_TOTAL, sd=2.0, family_size=family_size
    ) == pytest.approx(power)


def test_interaction_needs_more_units_than_a_launch_contrast() -> None:
    interaction = overlap_minimum_detectable_effect(DESIGN, "interaction", n_total=N_TOTAL, sd=1.0)
    joint = overlap_minimum_detectable_effect(DESIGN, "joint_launch", n_total=N_TOTAL, sd=1.0)
    # Interaction uses all four cells with unit weights: var 1/300+1/300+1/100+1/100 vs 1/300+1/100.
    assert interaction / joint == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_power_at_zero_effect_is_the_test_level() -> None:
    assert overlap_contrast_power(
        DESIGN, "interaction", effect=0.0, n_total=N_TOTAL, sd=1.0, alpha=0.05, family_size=5
    ) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "code, call",
    [
        (
            "overlap.planning.n_total",
            lambda: overlap_contrast_se(DESIGN, "interaction", n_total=4, sd=1.0),
        ),
        (
            "overlap.planning.sd",
            lambda: overlap_contrast_se(DESIGN, "interaction", n_total=100, sd=0.0),
        ),
        (
            "overlap.planning.sd",
            lambda: overlap_contrast_se(DESIGN, "interaction", n_total=100, sd={(0, 0): 1.0}),
        ),
        (
            "overlap.request.contrast_unknown",
            lambda: overlap_contrast_se(DESIGN, "both", n_total=100, sd=1.0),
        ),
        (
            "overlap.planning.effect",
            lambda: overlap_contrast_power(
                DESIGN, "interaction", effect=math.nan, n_total=100, sd=1.0
            ),
        ),
        (
            "overlap.planning.power",
            lambda: overlap_minimum_detectable_effect(
                DESIGN, "interaction", n_total=100, sd=1.0, power=1.0
            ),
        ),
        (
            "overlap.planning.family_size",
            lambda: overlap_contrast_power(
                DESIGN, "interaction", effect=1.0, n_total=100, sd=1.0, family_size=0
            ),
        ),
        (
            "overlap.request.alpha",
            lambda: overlap_contrast_power(
                DESIGN, "interaction", effect=1.0, n_total=100, sd=1.0, alpha=0.0
            ),
        ),
    ],
)
def test_planning_refusals_are_coded(code: str, call) -> None:
    with pytest.raises(CodedError) as exc:
        call()
    assert exc.value.code == code


def test_simulation_smoke_runs_the_runtime_procedure() -> None:
    result = simulate_overlap_power(
        DESIGN,
        "joint_launch",
        cell_means={(0, 0): 0.0, (1, 0): 0.0, (0, 1): 0.0, (1, 1): 0.0},
        n_total=200,
        sd=1.0,
        replications=50,
        seed=1,
    )
    assert result.replications == 50 and result.availability_rate == 1.0
    assert 0.0 <= result.rejection_rate <= 0.2


def test_an_allocation_that_underflows_a_cell_is_refused() -> None:
    from increment.overlap import OverlapDesign, OverlapExperiment

    tiny = OverlapDesign(
        first=OverlapExperiment(name="a", treatment_probability=1e-200),
        second=OverlapExperiment(name="b", treatment_probability=1e-200),
    )
    with pytest.raises(CodedError) as exc:
        overlap_contrast_se(tiny, "interaction", n_total=100, sd=1.0)
    assert exc.value.code == "overlap.planning.cell_probability"
