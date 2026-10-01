"""Sizing an experiment whose feature fires for part of the population.

Two different inflations, and conflating them is the whole hazard: the
assigned-population readout needs 1/r^2 more units, the triggered readout
needs 1/r. That factor-of-1/r gap is why triggering is worth building.
"""

from __future__ import annotations

import pytest

from increment.errors import InvalidRequestError
from increment.power.core import (
    Baseline,
    PowerDesign,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)

from ._procedures import make_procedure
from ._results import available_mde


def _baseline(**kw):
    return Baseline(mean=10.0, var=25.0, **kw)


def _design():
    return PowerDesign(power=0.8)


@pytest.mark.parametrize("rate", [0.5, 0.2, 0.05])
def test_triggered_sizing_scales_as_one_over_rate(rate):
    """Assigned n grows as 1/r, not 1/r^2 - only r of assigned units enter a triggered analysis."""
    undiluted = required_sample_size(0.1, _baseline(), make_procedure(), _design()).n_total
    triggered = required_sample_size(
        0.1, _baseline(trigger_rate=rate), make_procedure(), _design()
    ).n_total
    assert triggered == pytest.approx(undiluted / rate, rel=0.02)


@pytest.mark.parametrize("rate", [0.5, 0.2, 0.05])
def test_assigned_population_sizing_scales_as_one_over_rate_squared(rate):
    """Compliance dilution grows n as 1/r^2; small relative_lift keeps log1p's
    effect scale near-linear (nonlinearity alone is an 8-9% deviation, unrelated
    to trigger_rate)."""
    undiluted = required_sample_size(0.01, _baseline(), make_procedure(), _design()).n_total
    diluted = required_sample_size(
        0.01, _baseline(compliance=rate), make_procedure(), _design()
    ).n_total
    assert diluted == pytest.approx(undiluted / rate**2, rel=0.02)


@pytest.mark.parametrize("rate", [0.5, 0.2, 0.05])
def test_triggering_is_cheaper_by_exactly_one_over_rate(rate):
    """The headline claim, asserted rather than asserted-in-prose."""
    triggered = required_sample_size(
        0.01, _baseline(trigger_rate=rate), make_procedure(), _design()
    ).n_total
    diluted = required_sample_size(
        0.01, _baseline(compliance=rate), make_procedure(), _design()
    ).n_total
    assert diluted / triggered == pytest.approx(1.0 / rate, rel=0.02)


def test_reports_assigned_and_triggered_sizes_separately():
    """A bare n is ambiguous once a trigger is declared."""
    res = required_sample_size(0.1, _baseline(trigger_rate=0.2), make_procedure(), _design())
    assert res.n_triggered_total == pytest.approx(res.n_total * 0.2, rel=0.02)
    assert res.n_triggered_per_arm == pytest.approx(res.n_per_arm * 0.2, rel=0.02)


def test_no_trigger_rate_reports_none():
    """Mirrors n_clusters_* signalling unit randomization with None."""
    res = required_sample_size(0.1, _baseline(), make_procedure(), _design())
    assert res.n_triggered_total is None
    assert res.n_triggered_per_arm is None


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_trigger_rate_outside_the_unit_interval_refused(bad):
    with pytest.raises(InvalidRequestError) as exc_info:
        _baseline(trigger_rate=bad)
    assert exc_info.value.code == "power.baseline.trigger_rate"


def test_achieved_power_round_trips_at_the_solved_size():
    """Solved n must deliver requested power; triggered fields must
    round-trip exactly at the solved size."""
    design = _design()
    baseline = _baseline(trigger_rate=0.2)
    solved = required_sample_size(0.1, baseline, make_procedure(), design)
    got = achieved_power(solved.n_per_arm, 0.1, baseline, make_procedure(), design)
    assert got.power >= design.power - 0.01
    assert got.n_triggered_per_arm == solved.n_triggered_per_arm
    assert got.n_triggered_total == solved.n_triggered_total


def test_minimum_detectable_effect_reports_triggered_fields_and_inflated_mde():
    """Fewer analyzed units under a trigger yields a coarser MDE; triggered
    fields populate the same way as the other two solvers."""
    design = _design()
    n_per_arm = 100
    undiluted = minimum_detectable_effect(n_per_arm, _baseline(), make_procedure(), design)
    triggered = minimum_detectable_effect(
        n_per_arm, _baseline(trigger_rate=0.2), make_procedure(), design
    )
    assert available_mde(triggered) > available_mde(undiluted)
    assert triggered.n_triggered_per_arm == pytest.approx(triggered.n_per_arm * 0.2, rel=0.02)
    assert triggered.n_triggered_total == pytest.approx(triggered.n_total * 0.2, rel=0.02)
    assert undiluted.n_triggered_per_arm is None
    assert undiluted.n_triggered_total is None


def test_trigger_rate_composes_with_cluster_design_effect():
    """Recruitment counts assigned members; the design effect uses analyzed members."""
    design = _design()
    baseline = _baseline(
        trigger_rate=0.2, cluster_icc=0.05, avg_cluster_size=20.0, cluster_participation=1.0
    )
    assert baseline.design_effect == pytest.approx(1.15)
    res = required_sample_size(0.1, baseline, make_procedure(dependence="cluster"), design)
    assert res.n_triggered_total == pytest.approx(res.n_total * 0.2, rel=0.02)
    # Cluster counts use the ASSIGNED (inflated) population - clusters are
    # randomized before triggering happens.
    assert res.n_clusters_per_arm == (res.n_per_arm + 19) // 20
