"""Switchback-planning source/estimator variance acceptance under a frozen bounded law.

The full 32-cell design and finite-sample MC bound live in
tests._switchback_planning_design. This module does not gate interval coverage
or assert that noncentral-t planning equals finite-branch rejection probability.

Batching observes the actual source's ContrastPartitions while forwarding every
call to its actual reducer. Complete experiments have disjoint unit IDs (unit
law) or disjoint blocks over the same roster (shared law). Splitting those
already computed contributions and calling the actual reducer/estimator is
exactly the separate-experiment calculation. No HT contribution, retained-window
aggregation, covariance reduction, or estimator SE is implemented in the harness.
The public source's pooled pilot is a legitimate larger independent pilot; its
planning calls always use the experiment's N, window and cycle metadata.
"""

from __future__ import annotations

import math
from itertools import product
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from scipy.optimize import brentq
from scipy.special import roots_genlaguerre
from scipy.stats import binomtest, nct, norm, t

import increment.switchback as source_module
from increment.errors import CodedError
from increment.estimation.contrast import (
    ContrastPartition,
    estimate_contrast,
    reduce_contrast_partitions,
)
from increment.estimation.decision_types import ContrastDecisionProcedure
from increment.frame import from_switchback_panel
from increment.power import (
    SwitchbackBaseline,
    switchback_achieved_power,
    switchback_minimum_detectable_effect,
)
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.semantics.unit_cycle import UnitCycleTApproximation
from tests._switchback_planning_design import (
    ALTERNATIVE_PER_STEP,
    BATCH_SIZE,
    CELLS,
    MAX_MC_MARGIN,
    NOISE_HALF_WIDTH,
    PERIOD_TREND,
    PERSISTENT_SD,
    REFERENCE_PER_STEP,
    RELATIVE_TOLERANCE,
    ROOT_SEED,
    SHARED_ROSTER,
    SMOKE_SEED,
    UNIT_COUNT,
    WASHOUT,
    Cell,
    covariance,
    moments,
    repetitions,
)


def _assignment(cell):
    sequence = SharedScheduleOrder if cell.shared else IndependentBernoulliOrder
    return SwitchbackAssignment(
        sequence=sequence(probability_ct=cell.probability_ct),
        window=SwitchbackWindow(
            washout_steps=WASHOUT,
            observation_steps=cell.observation_steps,
            carryover_order=cell.carryover_order,
        ),
    )


def _procedure(metric="reference", *, shared=False):
    return ContrastDecisionProcedure(
        metric=metric,
        role="primary",
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
        reference=None if shared else UnitCycleTApproximation(),
    )


def _source(frame, cell):
    return from_switchback_panel(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"reference": "mean", "alternative": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(cell),
    )


def _population_baseline(cell):
    va, vg, cov = covariance(cell)
    sa, sg = math.sqrt(va), math.sqrt(vg)
    return SwitchbackBaseline(
        assignment=_assignment(cell),
        metric="reference",
        control_group="control",
        treatment_group="treatment",
        aggregation="sum",
        estimand="retained_window_total_difference",
        cycles_per_unit=None if cell.shared else cell.cycles,
        shared_roster=SHARED_ROSTER if cell.shared else None,
        delta_ref=REFERENCE_PER_STEP * cell.retained,
        sd_a=sa,
        sd_g=sg,
        rho=cov / (sa * sg) if sg else 0.0,
    )


def _streams(seed, cell_index):
    return tuple(
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, cell_index, stream])))
        for stream in range(3)
    )


def _panel(cell, count, streams):
    """Draw outcomes, not contrast contributions; all reductions are production."""
    assignment_rng, persistent_rng, noise_rng = streams
    units = len(SHARED_ROSTER) if cell.shared else UNIT_COUNT
    steps = WASHOUT + cell.observation_steps
    shape = (count, units, cell.cycles, 2, steps)
    rep, unit, cycle, period, step = np.indices(shape)
    order_shape = (count, 1 if cell.shared else units, cell.cycles)
    ct = assignment_rng.random(order_shape) < cell.probability_ct
    treated = (period == 1) == ct[..., None, None]
    intercept = persistent_rng.uniform(-0.5, 0.5, (count, units, 1, 1, 1))
    persistent_shape = (count, 1, cell.cycles, 1, 1) if cell.shared else (count, units, 1, 1, 1)
    persistent = PERSISTENT_SD * (2 * persistent_rng.integers(0, 2, persistent_shape) - 1)
    noise = noise_rng.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, shape)
    reference = (
        10 + intercept + PERIOD_TREND * period + (REFERENCE_PER_STEP + persistent) * treated + noise
    )
    treatment_periods = treated[..., 0].reshape(count, units, 2 * cell.cycles)
    previous = np.zeros_like(treatment_periods)
    previous[..., 1:] = treatment_periods[..., :-1]
    previous = previous.reshape(count, units, cell.cycles, 2, 1)
    contaminated = (step >= WASHOUT) & (step < WASHOUT + cell.carryover_order)
    reference = reference + 25 * previous * contaminated + 1000 * (step < WASHOUT)
    alternative = reference + (ALTERNATIVE_PER_STEP - REFERENCE_PER_STEP) * treated
    # Gather every label from a small string series instead of materialising
    # one numpy unicode array per row; the resulting columns are identical.
    if cell.shared:
        names = pl.Series(list(SHARED_ROSTER)).gather(unit.ravel())
        source_cycle = rep * cell.cycles + cycle
    else:
        keys = pl.Series([f"r{r:06d}-u{u:02d}" for r in range(count) for u in range(units)])
        names = keys.gather((rep * units + unit).ravel())
        source_cycle = cycle
    ct_counts = ct.sum(axis=(1, 2))
    frame = pl.DataFrame(
        {
            "replication": rep.ravel(),
            "unit": names,
            "cycle": source_cycle.ravel(),
            "period": period.ravel(),
            "step": step.ravel(),
            "group": pl.Series(["control", "treatment"]).gather(treated.ravel().astype(np.int64)),
            "reference": reference.ravel(),
            "alternative": alternative.ravel(),
        }
    )
    return frame, ct_counts


def _capture_source(frame, cell):
    captured = {}

    def observe(parts):
        assert len(parts) == 1
        part = parts[0]
        assert part.metric not in captured
        captured[part.metric] = part
        return reduce_contrast_partitions(parts)

    # Observe real source output; never replace a production numerical result.
    with patch.object(source_module, "reduce_contrast_partitions", side_effect=observe):
        source = _source(frame, cell)
    assert set(captured) == {"reference", "alternative"}
    return source, captured


def _split(part, cell, count):
    deltas = [{} for _ in range(count)]
    cycles = [{} for _ in range(count)]
    for key, value in part.unit_deltas.items():
        replication = int(key) // cell.cycles if cell.shared else int(key.split("-")[0][1:])
        deltas[replication][key] = value
        cycles[replication][key] = part.cycles_by_unit[key]
    metadata = part.model_dump(
        exclude={
            "unit_deltas",
            "cycles_by_unit",
            "unit_slopes",
            "exact_unit_deltas",
            "ct_counts_by_unit",
        }
    )
    return [
        ContrastPartition(
            **metadata,
            unit_deltas=values,
            cycles_by_unit=sizes,
            unit_slopes={key: part.unit_slopes[key] for key in values if key in part.unit_slopes},
            exact_unit_deltas={
                key: part.exact_unit_deltas[key] for key in values if key in part.exact_unit_deltas
            },
            ct_counts_by_unit={
                key: part.ct_counts_by_unit[key] for key in values if key in part.ct_counts_by_unit
            },
        )
        for values, sizes in zip(deltas, cycles, strict=True)
    ]


def _single_frame(frame, cell, replication):
    single = frame.filter(pl.col("replication") == replication)
    if cell.shared:
        single = single.with_columns(pl.col("cycle") - replication * cell.cycles)
    return single


def _assert_metadata(result, cell):
    assert result.randomization_law == (
        "shared_schedule" if cell.shared else "independent_bernoulli_order"
    )
    assert result.independence_grain == ("shared_block" if cell.shared else "unit_cycle")
    assert result.n_units == (len(SHARED_ROSTER) if cell.shared else UNIT_COUNT)
    assert result.n_cycles == result.n_units * cell.cycles
    assert result.n_blocks == (cell.cycles if cell.shared else None)
    assert result.dof == cell.n - 1
    assert result.observation_steps == cell.observation_steps
    assert result.retained_steps == cell.retained
    assert result.carryover_order == cell.carryover_order
    assert result.control_group == "control"
    assert result.treatment_group == "treatment"
    assert result.estimand == "retained_window_total_difference"
    assert result.identifying_assumption == "no_residual_carryover_after_discarded_steps"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("p", [0.5, 0.75])
def test_exact_two_branch_public_source_smoke(shared, p):
    """Rational enumeration uses population weights, not an empirical ddof echo."""
    cell = Cell(shared, p, 4 if shared else 1, 1, 0)
    orders = (True, True, True, False) if p == 0.75 else (True, True, False, False)
    rows = []
    for index, ct in enumerate(orders):
        for period, step in product(range(2), range(2)):
            treatment = (period == 1) == ct
            branch = 2.0 if ct else -1.0
            value = 10.0 + (branch if treatment else 0.0) if step else 1000.0
            rows.append(
                {
                    "unit": "u0" if shared else f"u{index}",
                    "cycle": index if shared else 0,
                    "period": period,
                    "step": step,
                    "group": "treatment" if treatment else "control",
                    "reference": value,
                    "alternative": value + 0.75 * treatment,
                }
            )
    source = _source(pl.DataFrame(rows), cell)
    pilot = source.planning_baseline(source.metrics[0], delta_ref=0.5)
    weights = np.array([1 / (2 * p), 1 / (2 * (1 - p))])
    probabilities = np.array([p, 1 - p])
    a = np.array([2.0, -1.0]) * weights
    centered = np.column_stack((a - 0.5, weights - 1))
    exact = centered.T @ (probabilities[:, None] * centered)
    # Four enumerated outcomes give sample covariance 4/3 times population covariance.
    assert pilot.sd_a**2 * 3 / 4 == pytest.approx(exact[0, 0])
    assert pilot.sd_g**2 * 3 / 4 == pytest.approx(exact[1, 1])
    assert pilot.rho * pilot.sd_a * pilot.sd_g * 3 / 4 == pytest.approx(exact[0, 1])
    population = pilot.model_copy(
        update={"sd_a": pilot.sd_a * math.sqrt(3 / 4), "sd_g": pilot.sd_g * math.sqrt(3 / 4)}
    )
    shifted = a + 0.75 * weights
    enumerated_variance = float(np.dot(probabilities, (shifted - 1.25) ** 2))
    plan = switchback_achieved_power(4, 1.25, population, _procedure(shared=shared))
    assert plan.standard_error is not None
    assert plan.standard_error**2 == pytest.approx(enumerated_variance / 4)
    actual = estimate_contrast(
        source.contrast_stats(source.metrics[1]), _procedure("alternative", shared=shared)
    )
    assert not actual.failures
    assert actual.results[0].standard_error is not None
    assert actual.results[0].standard_error ** 2 * 3 / 4 == pytest.approx(enumerated_variance / 4)


@pytest.mark.parametrize("shared", [False, True])
def test_fourth_moment_matches_exact_small_noise_quadrature(shared):
    cell = Cell(shared, 0.75, 2, 1, 0)
    cycles, roster = (1, len(SHARED_ROSTER)) if shared else (2, 1)
    # Three-point uniform quadrature integrates every polynomial through degree five.
    nodes = np.array([-3 / math.sqrt(5), 0.0, 3 / math.sqrt(5)])
    node_weights = np.array([5 / 18, 4 / 9, 5 / 18])
    indices = np.array(list(product(range(3), repeat=2 * cycles * roster)))
    probabilities = np.prod(node_weights[indices], axis=1)
    periods = nodes[indices].reshape(-1, cycles, roster, 2).mean(axis=2)
    differences = periods[..., 1] - periods[..., 0]
    terms2, terms4 = [], []
    for orders in product((0, 1), repeat=cycles):
        mass = math.prod(0.75 if order else 0.25 for order in orders)
        for sign in (-1, 1):
            contributions = []
            for cycle, ct in enumerate(orders):
                branch = (
                    ALTERNATIVE_PER_STEP
                    + sign * PERSISTENT_SD
                    + (PERIOD_TREND + differences[:, cycle]) * (1 if ct else -1)
                )
                contributions.append(branch / (2 * (0.75 if ct else 0.25)))
            deviation = np.mean(contributions, axis=0) - ALTERNATIVE_PER_STEP
            terms2.append(mass / 2 * float(np.dot(probabilities, deviation**2)))
            terms4.append(mass / 2 * float(np.dot(probabilities, deviation**4)))
    law = moments(cell, ALTERNATIVE_PER_STEP)
    assert math.fsum(terms2) == pytest.approx(law.variance)
    assert math.fsum(terms4) == pytest.approx(law.fourth)
    va, vg, cov = covariance(cell)
    shift = ALTERNATIVE_PER_STEP - REFERENCE_PER_STEP
    assert law.variance == pytest.approx(va + 2 * shift * cov + shift**2 * vg)


@pytest.mark.slow
@pytest.mark.parametrize("shared", [False, True])
def test_batched_actual_kernels_equal_separate_public_sources(shared):
    cell = Cell(shared, 0.75, 5, 4, 2)
    frame, _ = _panel(cell, 2, _streams(SMOKE_SEED, int(shared)))
    _, captured = _capture_source(frame, cell)
    for metric_name, part in captured.items():
        for replication, single_part in enumerate(_split(part, cell, 2)):
            direct = _source(_single_frame(frame, cell, replication), cell)
            metric = next(metric for metric in direct.metrics if metric.name == metric_name)
            stats = reduce_contrast_partitions([single_part])
            direct_result = estimate_contrast(
                direct.contrast_stats(metric), _procedure(metric_name, shared=shared)
            )
            batched_result = estimate_contrast(stats, _procedure(metric_name, shared=shared))
            row, expected = batched_result.results[0], direct_result.results[0]
            _assert_metadata(row, cell)
            assert row.estimate.value == pytest.approx(expected.estimate.value, abs=1e-12)
            assert row.standard_error == pytest.approx(expected.standard_error, rel=1e-12)
            assert row.estimate.lb == pytest.approx(expected.estimate.lb, abs=1e-12)
            assert row.estimate.ub == pytest.approx(expected.estimate.ub, abs=1e-12)


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell_index,cell", list(enumerate(CELLS)), ids=[cell.id for cell in CELLS])
def test_source_estimator_and_planning_variance_calibration(
    cell_index, cell, record_property, runtime_diagnostics
):
    record_property("cell", cell.id)
    with runtime_diagnostics.stage("setup"):
        count = repetitions(cell)
        streams = _streams(ROOT_SEED, cell_index)
        reference_law = moments(cell, REFERENCE_PER_STEP)
        alternative_law = moments(cell, ALTERNATIVE_PER_STEP)
        delta_ref, delta = REFERENCE_PER_STEP * cell.retained, ALTERNATIVE_PER_STEP * cell.retained
        population = _population_baseline(cell)
        planning_procedure, runtime_procedure = (
            _procedure(shared=cell.shared),
            _procedure("alternative", shared=cell.shared),
        )
        exact_plan = switchback_achieved_power(cell.n, delta, population, planning_procedure)
        assert exact_plan.standard_error is not None
        assert exact_plan.standard_error**2 == pytest.approx(alternative_law.variance / cell.n)
        runtime_variances, pilot_reference, pilot_shifted = [], [], []
        refused: list[str] = []
        draws = cell.cycles if cell.shared else UNIT_COUNT * cell.cycles
        suspect_counts = {
            k for k in range(draws + 1) if binomtest(k, draws, cell.probability_ct).pvalue < 1e-6
        }
    runtime_diagnostics.progress(
        completed_batches=0,
        completed_draws=0,
        planned_draws=count,
        batch_size=BATCH_SIZE,
    )
    for batch_index in range(count // BATCH_SIZE):
        with runtime_diagnostics.stage("panel"):
            frame, ct_counts = _panel(cell, BATCH_SIZE, streams)
        with runtime_diagnostics.stage("source"):
            source, captured = _capture_source(frame, cell)
        with runtime_diagnostics.stage("planning"):
            pilot = source.planning_baseline(source.metrics[0], delta_ref=delta_ref)
            assert pilot.assignment == population.assignment
            assert pilot.cycles_per_unit == population.cycles_per_unit
            assert pilot.shared_roster == population.shared_roster
            assert pilot.metric == population.metric
            assert pilot.estimand == population.estimand
            assert pilot.control_group == population.control_group
            assert pilot.treatment_group == population.treatment_group
            for effect, values in ((delta_ref, pilot_reference), (delta, pilot_shifted)):
                prediction = switchback_achieved_power(cell.n, effect, pilot, planning_procedure)
                assert prediction.standard_error is not None, prediction
                assert prediction.n == cell.n and prediction.df == cell.n - 1
                assert prediction.planning_grain == ("shared_block" if cell.shared else "unit")
                values.append(prediction.standard_error**2)
        with runtime_diagnostics.stage("estimation"):
            for replication, part in enumerate(_split(captured["alternative"], cell, BATCH_SIZE)):
                # Re-enter the public constructor for splits its integrity guard could refuse.
                # Record real refusals; do not condition the law by silently redrawing.
                if int(ct_counts[replication]) in suspect_counts:
                    try:
                        _source(_single_frame(frame, cell, replication), cell)
                    except CodedError as error:
                        refused.append(error.code)
                        continue
                computation = estimate_contrast(
                    reduce_contrast_partitions([part]), runtime_procedure
                )
                row = computation.results[0]
                _assert_metadata(row, cell)
                assert row.standard_error is not None
                assert not computation.failures, computation.failures
                # An available interval is exactly a calibrated level on the estimate.
                assert row.estimate.level is not None, row.estimate
                runtime_variances.append(row.standard_error**2)
        runtime_diagnostics.progress(
            completed_batches=batch_index + 1,
            completed_draws=(batch_index + 1) * BATCH_SIZE,
            planned_draws=count,
            batch_size=BATCH_SIZE,
        )
    with runtime_diagnostics.stage("assessment"):
        record_property(
            "accounting",
            {"attempted": count, "estimable": len(runtime_variances), "refused": refused},
        )
        assert not refused, refused
        assert len(runtime_variances) == count
        statistics = (
            ("runtime_variance", runtime_variances, alternative_law, cell.n),
            ("pilot_reference_variance", pilot_reference, reference_law, BATCH_SIZE * cell.n),
            ("pilot_shifted_variance", pilot_shifted, alternative_law, BATCH_SIZE * cell.n),
        )
        _assert_variance_calibration(cell, count, statistics, record_property)


def _assert_variance_calibration(cell, count, statistics, record_property):
    for name, values, law, sample_n in statistics:
        expected = law.variance / cell.n
        ratio = math.fsum(values) / len(values) / expected
        margin = law.margin(len(values), sample_n)
        mcse = math.sqrt(law.variance_of_sample_variance_ratio(sample_n) / len(values))
        report = {
            "observed": ratio * expected,
            "expected": expected,
            "ratio": ratio,
            "relative_mcse": mcse,
            "family_relative_margin": margin,
            "repetitions": count,
            "pilot_batches": count // BATCH_SIZE,
        }
        record_property(name, report)
        assert margin <= MAX_MC_MARGIN, report
        assert abs(ratio - 1) + margin <= RELATIVE_TOLERANCE, report


@pytest.mark.parametrize("shared,p", list(product((False, True), (0.5, 0.75))))
def test_population_mde_roundtrip_has_independent_smallest_root(shared, p):
    """The existing test_switchback.py retains its exact two-root/unattainable cases."""
    cell = Cell(shared, p, 20, 4, 1)
    baseline = _population_baseline(cell)
    n, target = 64, 0.8
    critical = t.isf(0.025, n - 1)
    nodes, weights = roots_genlaguerre(128, (n - 1) / 2 - 1)
    weights /= math.gamma((n - 1) / 2)
    conditional_threshold = critical * np.sqrt(2 * nodes / (n - 1))

    def reference_power(nc):
        # Integrate the independent normal numerator over its chi-square scale.
        tails = norm.sf(conditional_threshold - nc) + norm.cdf(-conditional_threshold - nc)
        return float(weights @ tails)

    # Shared-baseline achieved power is available power, admission_probability *
    # power_given_admissible, matching the positivity refusal's admission risk.
    # The reference computes power_given_admissible, so it targets target / admission.
    admission = 1.0 - p**n - (1.0 - p) ** n if shared else 1.0
    target_given_admissible = target / admission
    level = brentq(lambda nc: reference_power(nc) - target_given_admissible, 0.0, 20.0, xtol=1e-13)
    va, a, cov = covariance(cell)
    b = cov - baseline.delta_ref * a
    c = va - 2 * baseline.delta_ref * cov + baseline.delta_ref**2 * a
    roots = np.roots([n - level**2 * a, -2 * level**2 * b, -(level**2) * c])
    favorable = sorted(
        float(root.real) for root in roots if abs(root.imag) < 1e-10 and root.real > 0
    )
    assert favorable
    result = switchback_minimum_detectable_effect(
        n, baseline, _procedure(shared=cell.shared), target_power=target
    )
    assert result.mde_abs is not None, result
    assert result.mde_abs == pytest.approx(favorable[0], rel=1e-9)
    roundtrip = switchback_achieved_power(
        n, result.mde_abs, baseline, _procedure(shared=cell.shared)
    )
    assert roundtrip.power is not None and roundtrip.power >= target
    assert roundtrip.standard_error is not None
    oracle = moments(cell, result.mde_abs / cell.retained)
    assert roundtrip.standard_error**2 == pytest.approx(oracle.variance / n)
    assert reference_power(result.mde_abs * math.sqrt(n / oracle.variance)) == pytest.approx(
        target_given_admissible
    )


@pytest.mark.parametrize("shared", [False, True])
def test_discarding_independent_noise_steps_reduces_fixed_per_step_effect_power(shared):
    """Only this law: no residual carryover, no trend, independent unit/period noise."""
    powers = []
    for carryover in (0, 1, 2):
        cell = Cell(shared, 0.5, 20, 4, carryover)
        length = cell.retained
        roster = len(SHARED_ROSTER) if shared else 1
        independent_cycles = 1 if shared else cell.cycles
        # Sum of L independent period differences; average cycles or roster exactly once.
        variance = 2 * length / (roster * independent_cycles)
        baseline = SwitchbackBaseline(
            assignment=_assignment(cell),
            metric="reference",
            control_group="control",
            treatment_group="treatment",
            aggregation="sum",
            estimand="retained_window_total_difference",
            cycles_per_unit=None if shared else cell.cycles,
            shared_roster=SHARED_ROSTER if shared else None,
            delta_ref=0.0,
            sd_a=math.sqrt(variance),
            sd_g=0.0,
            rho=0.0,
        )
        result = switchback_achieved_power(
            cell.n, 0.05 * length, baseline, _procedure(shared=cell.shared)
        )
        assert result.standard_error is not None and result.power is not None
        assert result.standard_error**2 == pytest.approx(variance / cell.n)
        critical = t.isf(0.025, cell.n - 1)
        nc = 0.05 * math.sqrt(length * roster * independent_cycles * cell.n / 2)
        # A shared baseline's achieved power already multiplies by the exact
        # probability its cell.n blocks realize both cycle orders.
        admission = 1.0 - 0.5**cell.n - 0.5**cell.n if shared else 1.0
        expected = admission * (
            nct.sf(critical, cell.n - 1, nc) + nct.cdf(-critical, cell.n - 1, nc)
        )
        assert result.power == pytest.approx(expected)
        powers.append(result.power)
    assert powers[0] > powers[1] > powers[2]
