"""Deterministic certification budget checks; no calibration draws."""

from fractions import Fraction

import pytest

from increment.power.unit_cycle import _band
from tests.simulate.unit_cycle_sizing import (
    COMPONENT_MARGIN,
    DIRECTIONS,
    MAX_EXPERIMENT_EVALUATIONS,
    Gate,
    allocate,
    calibration_schedule,
    freeze,
    gate_decision,
    planning_size,
    prospective_cost,
)


def test_checkpoint_allocations_bound_the_entire_declared_family():
    schedule = calibration_schedule(
        total_error=0.0005, cases=2160, checkpoints=(128, 512, 2048, 8192)
    )
    spent = sum(Fraction(error) for error in schedule["checkpoint_error"]) * 2160
    assert 0 < spent <= Fraction(schedule["total_error"])
    assert all(
        earlier > later
        for earlier, later in zip(
            schedule["checkpoint_error"], schedule["checkpoint_error"][1:], strict=False
        )
    )
    with pytest.raises(ValueError):
        calibration_schedule(total_error=0.0005, cases=2160, checkpoints=(512, 128))


def test_threshold_decisions_preserve_uncertainty_and_check_both_sides():
    availability = gate_decision(Gate("availability", 1, lower=0.99), 1024, 1024, 0.0001)
    assert availability["passed"] and availability["status"] == "certified"
    assert availability["margin"] > COMPONENT_MARGIN
    band = Gate("band", 0.5, lower=0.4, upper=0.6)
    assert gate_decision(band, 90, 100, 0.05)["status"] == "failed"
    assert gate_decision(band, 50, 100, 0.001)["status"] == "inconclusive"
    assert gate_decision(band, 0, 0, 0.001)["status"] == "inconclusive"


def test_production_endpoint_band_sizing_respects_total_component_margin():
    eta = allocate(0.001, DIRECTIONS)
    repetitions = planning_size(eta)
    each, radius = _band(repetitions, 8 * eta, 8)
    assert each <= eta
    assert 4 * radius <= COMPONENT_MARGIN
    assert (
        prospective_cost(0, repetitions)["maximum_experiment_evaluations"]
        > MAX_EXPERIMENT_EVALUATIONS
    )


def test_infeasible_freeze_refuses_before_terminal_sizing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("infeasible study reached expensive terminal sizing")

    monkeypatch.setattr("tests.simulate.unit_cycle_sizing.terminal_size", forbidden)
    ledger = {
        "family_alpha": 0.01,
        "reservations": [{"task": "I14", "alpha": 0.001, "nominal_failure": 0.001}],
    }
    with pytest.raises(RuntimeError):
        freeze(ledger, max_repetitions=8)
