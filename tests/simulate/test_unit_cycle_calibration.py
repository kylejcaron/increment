"""Independent representative release checks and reusable full-campaign oracles.

The original manifest is unchanged. Exhaustive calibration is an explicit,
bounded research campaign, not an unexecuted prerequisite disguised as a test.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from increment.errors import CodedError, InvalidRequestError
from increment.estimation.contrast import estimate_contrast
from increment.estimation.contrast_results import ContrastResult, ContrastResults
from increment.estimation.decision_types import ContrastDecisionProcedure, PValueEvidence
from increment.frame import from_switchback_panel
from increment.semantics.design import Randomized
from tests._unit_cycle_design import (
    CELLS,
    DESIGNS,
    MANIFEST,
    Cell,
    admission_oracle,
    analytic_evidence,
    assignment,
    draw_periods,
    enumerate_orders,
    envelope_for,
    joint_law,
    law_moments,
    population_moments,
    scalar_observation,
    scalar_reference,
    schedule,
    streams,
)
from tests.simulate.unit_cycle_sizing import (
    Gate,
    allocate,
    calibration_schedule,
    cp_bounds,
    gate_bounds,
    gate_decision,
)

RELEASE_CELLS = (
    Cell(4, 1, 0.9, "normal", 0.0, "unequal", 0, "retained_total"),
    Cell(10, 5, 0.75, "centered_lognormal", 0.9, "unequal", 0, "pre_normalized_retained_mean"),
    Cell(40, 1, 0.25, "centered_gamma", 0.5, "equal", 0, "retained_total"),
)
REPRESENTATIVES = tuple(
    dict.fromkeys(
        (
            *RELEASE_CELLS,
            *(
                next(cell for cell in CELLS if getattr(cell, axis) == value)
                for axis, values in MANIFEST["axes"].items()
                for value in values
            ),
        )
    )
)
RELEASE_REPETITIONS = 1536
RELEASE_SCHEDULE = calibration_schedule(
    total_error=allocate(MANIFEST["family_alpha"], 2),
    cases=len(RELEASE_CELLS) + 1,
    checkpoints=(RELEASE_REPETITIONS,),
)


def procedure(envelope, alternative="two-sided", null=0):
    return ContrastDecisionProcedure(
        metric="outcome",
        role="primary",
        alternative=alternative,
        null_abs=null,
        alpha=0.05,
        reference=envelope,
    )


def frame(periods, ct):
    """Panel as a pyarrow table: every ingress step runs on the calling thread.

    Polars hands each of its ~25 tiny operations per panel to its thread pool;
    under a fully loaded machine every hand-off waits on the scheduler.
    """
    unit, cycle, period = np.indices(periods.shape)
    return pa.table(
        {
            "unit": [f"u{u:06d}" for u in unit.ravel()],
            "cycle": cycle.ravel(),
            "period": period.ravel(),
            "step": np.zeros(periods.size, dtype=int),
            "group": np.where((period == 1) == ct[..., None], "treatment", "control").ravel(),
            "outcome": periods.ravel(),
        }
    )


def source(periods, ct, cell, envelope, *, reverse=False):
    data = frame(periods, ct)
    return from_switchback_panel(
        data.take(np.arange(data.num_rows - 1, -1, -1)) if reverse else data,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"outcome": "mean"},
        assignment=assignment(cell),
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        contrast_references={"outcome": envelope},
    )


def assert_wire_parity(proc, row, results):
    from increment.decision import (
        CompiledDecisionPlan,
        CompiledViewPolicies,
        FixedInference,
        MultiplicityFamily,
    )
    from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json

    policy = MultiplicityFamily(name="none")
    plan = CompiledDecisionPlan(
        declared=True,
        alpha=0.05,
        q=0.1,
        path="frame/contrast",
        inference=FixedInference(),
        procedures={"outcome": proc},
        view_policies=CompiledViewPolicies(
            asof=policy, randomized_breakout=policy, encouragement_breakout=policy
        ),
    )
    restored = compiled_plan_from_json(compiled_plan_to_json(plan))
    restored_procedure = restored.procedures["outcome"]
    assert isinstance(restored_procedure, ContrastDecisionProcedure)
    assert restored_procedure.reference == proc.reference
    assert type(row).model_validate_json(row.model_dump_json()) == row
    rendered = ContrastResults(results).to_frame(backend="polars")
    assert isinstance(rendered, pl.DataFrame)
    values = rendered.to_dicts()[0]
    assert values["dof"] is None
    assert values["mean_slope"] == row.mean_slope
    assert values["estimate"] == row.estimate.value
    assert values["lb"] == row.estimate.lb and values["ub"] == row.estimate.ub


def assert_public_parity(periods, ct, cell, effect, *, envelope=None, alternative="two-sided"):
    """Production is the object under test; truth is scalar potential arithmetic."""
    envelope = envelope_for(joint_law(cell, 0)) if envelope is None else envelope
    observed = periods + effect * np.stack((~ct, ct), -1)
    proc = procedure(envelope, alternative)
    refused, mass, _ = admission_oracle(cell.units * cell.cycles, cell.probability_ct)
    for one, orders in zip(observed, ct, strict=True):
        if int(orders.sum()) in refused:
            with pytest.raises(InvalidRequestError) as caught:
                source(one, orders, cell, envelope)
            assert caught.value.code == "source.frame.switchback.schedule"
            assert caught.value.context["reason"] == "implausible_realized_split"
            continue
        point, slope, units, slopes = scalar_observation(one, orders, cell.probability_ct)
        src = source(one, orders, cell, envelope)
        stats = src.contrast_stats(src.metrics[0])
        computation = estimate_contrast(stats, proc)
        row = computation.results[0]
        assert not computation.failures
        assert row.n_units == cell.units and row.n_cycles == cell.units * cell.cycles
        assert stats.mean_slope == pytest.approx(slope)
        assert stats.ct_cycles == int(orders.sum())
        assert stats.tc_cycles == orders.size - int(orders.sum())
        assert stats.minimum_cycles_per_unit == stats.maximum_cycles_per_unit == cell.cycles
        assert row.estimate.value == pytest.approx(point, abs=2e-12)
        assert row.standard_error == pytest.approx(
            float(np.std(units, ddof=1) / math.sqrt(cell.units)), abs=2e-12
        )
        assert row.method == "switchback_unit_variance_envelope" and row.dof is None
        assert row.dof_unavailable_reason == "not_applicable"
        assert row.refusal_probability_upper is not None
        assert Fraction(row.refusal_probability_upper) >= mass
        expected = scalar_reference(
            point,
            slope,
            envelope,
            cell.units,
            row.refusal_probability_upper,
            alternative=alternative,
        )
        for name, key in (("lb", "lower"), ("ub", "upper")):
            value = getattr(row.estimate, name)
            assert (
                value is None
                if expected[key] is None
                else value == pytest.approx(expected[key], abs=2e-12)
            )
        evidence = next(iter(computation.evidence.values()))
        assert isinstance(evidence, PValueEvidence)
        assert evidence.p_value == pytest.approx(expected["p_value"], abs=2e-12)
        assert row.reference == (
            "residual_chebyshev" if alternative == "two-sided" else "residual_cantelli"
        )
        assert row.reference_spec == envelope and row.provenance == envelope.provenance
        pilot = src.planning_baseline(src.metrics[0], delta_ref=effect)
        assert pilot.cycles_per_unit == cell.cycles and pilot.assignment == assignment(cell)
        assert pilot.sd_a == pytest.approx(float(np.std(units, ddof=1)), abs=2e-12)
        assert pilot.sd_g == pytest.approx(float(np.std(slopes, ddof=1)), abs=2e-12)
        assert_wire_parity(proc, row, computation.results)


@pytest.mark.slow
@pytest.mark.parametrize("cell", REPRESENTATIVES, ids=[cell.id for cell in REPRESENTATIVES])
def test_representative_independent_analytic_proof(cell, record_property):
    from increment._unit_cycle import unit_cycle_admission_rule, unit_cycle_envelope_cutoff
    from increment.power.unit_cycle import unit_cycle_law_moments, unit_cycle_variance_envelope

    law = joint_law(cell)
    proof = analytic_evidence(cell, law)
    record_property("analytic_proof", json.dumps(proof, allow_nan=False))
    assert proof["coverage_lower"] >= 0.95 and proof["unconditional_miss_upper"] <= 0.05
    assert proof["positive_slope_lower"] > 0 and math.isfinite(proof["width_upper"])
    assert proof["standardized_admitted_bias"] <= proof["standardized_admitted_bias_upper"] <= 0.05
    truth = law_moments(law)
    assert truth["residual_variance"] == (
        truth["variance_a"]
        - 2 * truth["reference_effect"] * truth["covariance"]
        + truth["reference_effect"] ** 2 * truth["variance_g"]
    )
    actual = unit_cycle_law_moments(law)
    assert actual.reference_effect == pytest.approx(float(truth["reference_effect"]), abs=1e-12)
    assert actual.sd_a is not None and actual.sd_g is not None
    assert actual.sd_a**2 == pytest.approx(float(truth["variance_a"]))
    assert actual.sd_g**2 == pytest.approx(float(truth["variance_g"]), abs=1e-12)
    if actual.rho is not None:
        assert actual.rho * actual.sd_a * actual.sd_g == pytest.approx(
            float(truth["covariance"]), abs=1e-12
        )
    envelope = unit_cycle_variance_envelope(law)
    assert Fraction(envelope.residual_variance_upper) >= truth["residual_variance"]
    assert envelope.residual_variance_upper == pytest.approx(float(truth["residual_variance"]))
    rule = unit_cycle_admission_rule(cell.units * cell.cycles, cell.probability_ct)
    refused, mass, _ = admission_oracle(cell.units * cell.cycles, cell.probability_ct)
    assert Fraction(rule.refusal_probability_upper) >= mass
    assert mass <= Fraction(1e-6)
    assert set(refused) == {
        k
        for k in range(cell.units * cell.cycles + 1)
        if not rule.minimum_ct <= k <= rule.maximum_ct
    }
    for alternative in ("two-sided", "greater", "less"):
        cut = unit_cycle_envelope_cutoff(
            envelope, n=cell.units, alpha=0.05, alternative=alternative
        )
        assert Fraction(cut.effective_alpha) + Fraction(cut.refusal_probability_upper) <= Fraction(
            0.05
        )
        variance = Fraction(envelope.residual_variance_upper) / cell.units
        needed = variance / Fraction(cut.effective_alpha)
        if alternative != "two-sided":
            needed *= 1 - Fraction(cut.effective_alpha)
        assert Fraction(cut.value) ** 2 >= needed


@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_index,cell",
    [(CELLS.index(cell), cell) for cell in REPRESENTATIVES],
    ids=[cell.id for cell in REPRESENTATIVES],
)
def test_representative_public_parity(cell_index, cell):
    periods, ct = draw_periods(cell, 1, streams(cell_index))
    assert_public_parity(periods, ct, cell, cell.effect)


@pytest.mark.parametrize("noise", ("normal", "centered_lognormal", "centered_gamma"))
def test_source_reference_reaches_compiled_analysis(noise):
    from increment.analysis import Analysis

    cell = Cell(4, 1, 0.9, noise, 0.9, "unequal", 0, "retained_total")
    periods, ct = draw_periods(cell, 1, streams(MANIFEST["smoke_seed"]))
    envelope = envelope_for(joint_law(cell))
    mapping = {"outcome": envelope}
    analysis = Analysis.from_switchback_panel(
        frame(periods[0], ct[0]),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"outcome": "mean"},
        assignment=assignment(cell),
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        contrast_references=mapping,
    )
    mapping.clear()
    row = analysis.run()[0]
    assert isinstance(row, ContrastResult)
    assert row.reference_spec == envelope
    assert row.method == "switchback_unit_variance_envelope"


@pytest.mark.parametrize(
    "p,rho,response_meaning",
    tuple(
        product(
            [0.25, 0.5, 0.75, 0.9],
            [0, 0.5, 0.9],
            ["retained_total", "pre_normalized_retained_mean"],
        )
    ),
)
def test_population_covariance_from_exact_noise_quadrature(p, rho, response_meaning):
    """Second moments need only centered unit-variance shocks, not their skew."""
    cell = Cell(2, 2, p, "normal", rho, "unequal", 0, response_meaning)
    counts, baseline, treatment, load = schedule(cell, 0)
    values, weights, gs = [], [], []
    # Separate the common-noise and independent-noise components of the mixture.
    for typ, reuse in product(range(2), (False, True)):
        for bits in product((0, 1), repeat=2):
            order_mass = math.prod(p if bit else 1 - p for bit in bits)
            w = np.array([1 / (2 * (p if bit else 1 - p)) for bit in bits])
            mean = np.array(
                [
                    treatment[typ, c, z] + baseline[typ, c, z] - baseline[typ, c, 1 - z]
                    for c, z in enumerate(bits)
                ]
            )
            # Independent aggregate shock SD is load/sqrt(count); two-point
            # quadrature integrates second moments exactly for every noise law.
            for signs in product((-1, 1), repeat=1 if reuse else 2):
                noise = np.array(
                    [
                        load[typ, c, z]
                        * signs[0 if reuse else c]
                        / (1 if reuse else math.sqrt(counts[typ, c, z]))
                        for c, z in enumerate(bits)
                    ]
                )
                values.append(float(np.mean(w * (mean + noise))))
                gs.append(float(w.mean()))
                weights.append(0.5 * order_mass * (rho if reuse else 1 - rho) / 2 ** len(signs))
    weights, values, gs = np.array(weights), np.array(values), np.array(gs)
    mean_a, mean_g = weights @ values, weights @ gs
    va, vg, cov = population_moments(cell, 0)
    assert mean_a == pytest.approx(0, abs=1e-12)
    assert mean_g == pytest.approx(1)
    assert weights @ ((values - mean_a) ** 2) == pytest.approx(va)
    assert weights @ ((gs - mean_g) ** 2) == pytest.approx(vg, abs=1e-12)
    assert weights @ ((values - mean_a) * (gs - mean_g)) == pytest.approx(cov, abs=1e-12)
    for effect in (-1, 0.7, 2):
        assert population_moments(cell, effect)[0] == pytest.approx(
            va + 2 * effect * cov + effect**2 * vg
        )


def test_manifest_preserves_every_axis_and_shares_only_effect_rows():
    assert len(CELLS) == MANIFEST["cell_count"] == 4320
    assert len(DESIGNS) == MANIFEST["model_design_count"] == 2160
    assert len({cell.id for cell in CELLS}) == 4320
    assert {replace(cell, effect=0) for cell in CELLS} == {cell for _, cell in DESIGNS}
    assert MANIFEST["alpha"] == 0.05 and MANIFEST["scientific_tolerance"] == 0.005
    assert (
        sum(
            MANIFEST["margin_allocation"][k] for k in ("reference_planning", "runtime_confirmation")
        )
        == 0.0025
    )
    assert "total_attempts" not in MANIFEST


@pytest.mark.slow
@pytest.mark.parametrize(
    "p,response_meaning",
    tuple(product((0.25, 0.5, 0.75, 0.9), ("retained_total", "pre_normalized_retained_mean"))),
)
def test_exact_weighted_orders_and_public_source(p, response_meaning):
    cell = Cell(2, 2, p, "normal", 0, "unequal", 0.7, response_meaning)
    _, baseline, effect, _ = schedule(cell, cell.effect)
    masses, values, slopes, counts = enumerate_orders(baseline, baseline + effect, p)
    target = float(effect.mean())
    assert math.fsum(masses) == pytest.approx(1)
    assert masses @ values.mean(axis=-1) == pytest.approx(target)
    dct = baseline[..., 1] + effect[..., 1] - baseline[..., 0]
    dtc = baseline[..., 0] + effect[..., 0] - baseline[..., 1]
    variance = float(np.sum(((1 - p) * dct - p * dtc) ** 2 / (4 * p * (1 - p)))) / 16
    assert masses @ ((values.mean(axis=-1) - target) ** 2) == pytest.approx(variance)
    assert masses[counts == 4].sum() == pytest.approx(p**4)
    assert masses[counts == 0].sum() == pytest.approx((1 - p) ** 4)
    periods, orders = [], []
    for bits in product((False, True), repeat=4):
        ct = np.array(bits).reshape(2, 2)
        periods.append(baseline + np.stack((~ct, ct), -1) * effect)
        orders.append(ct)
    assert_public_parity(np.array(periods), np.array(orders), cell, 0)
    assert np.all(slopes >= 1 / (2 * max(p, 1 - p)))


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ("two-sided", "greater", "less"))
def test_legitimate_same_order_and_zero_sample_variance_keep_envelope(alternative):
    cell = Cell(4, 1, 0.9, "normal", 0, "equal", 0, "retained_total")
    periods = np.broadcast_to(np.array([0.0, 1.0]), (16, 4, 1, 2))
    ct = np.array(list(product((False, True), repeat=4))).reshape(16, 4, 1)
    law = joint_law(cell, 0)
    envelope = envelope_for(law)
    assert_public_parity(periods, ct, cell, 0, envelope=envelope, alternative=alternative)
    for order in (False, True):
        src = source(periods[0], np.full((4, 1), order), cell, envelope)
        row = estimate_contrast(
            src.contrast_stats(src.metrics[0]), procedure(envelope, alternative)
        ).results[0]
        assert row.standard_error == 0
        assert row.estimate.lb is not None or row.estimate.ub is not None


@pytest.mark.slow
def test_large_offsets_row_order_and_frozen_declarations():
    cell = Cell(4, 5, 0.75, "centered_gamma", 0.9, "unequal", 1, "retained_total")
    law = joint_law(cell, 0)
    envelope = envelope_for(law)
    periods, ct = draw_periods(cell, 1, streams(MANIFEST["smoke_seed"]))
    periods += 2**30
    assert_public_parity(periods, ct, cell, 1, envelope=envelope)
    rows = []
    for reverse in (False, True):
        src = source(periods[0], ct[0], cell, envelope, reverse=reverse)
        rows.append(
            estimate_contrast(src.contrast_stats(src.metrics[0]), procedure(envelope)).results[0]
        )
    assert rows[0] == rows[1]
    with pytest.raises((CodedError, ValueError, TypeError)):
        envelope.residual_variance_upper = 0
    raw = law.model_dump(mode="python")
    copied = type(law).model_validate(raw)
    raw["types"] = []
    assert copied.types == law.types
    with pytest.raises((CodedError, ValueError, TypeError)):
        copied.types[0].cycles[0].ct_mean = 123


@pytest.mark.slow
@pytest.mark.parametrize(
    "effect,lambdas,lower_bound",
    [
        (0.0, (5.253887449625594, 4.5), 0.05653644209396722),
        (1.0, (4.330127018922193, 6.1), 0.05635913367335912),
    ],
)
def test_corrected_counterexample_and_corrected_envelope(effect, lambdas, lower_bound):
    from scipy.stats import chi2, norm

    cell = Cell(4, 1, 0.9, "normal", 0, "unequal", effect, "retained_total")
    counts, baseline, treatment, load = schedule(cell, effect)
    assert baseline[:, 0, 0] == pytest.approx([2, 6])
    mean = (baseline[:, 0, 1] - baseline[:, 0, 0] + treatment[:, 0, 1]) / (2 * 0.9)
    sd = load[:, 0, 1] / np.sqrt(counts[:, 0, 1]) / (2 * 0.9) / 2
    assert np.abs(mean - effect) / sd == pytest.approx(lambdas)
    bound = 0.45**4 * sum(
        chi2.cdf(2, 3) * norm.cdf(lam - 3.2 * math.sqrt(2 / 3))
        + (chi2.cdf(4, 3) - chi2.cdf(2, 3)) * norm.cdf(lam - 3.2 * math.sqrt(4 / 3))
        for lam in lambdas
    )
    assert bound == pytest.approx(lower_bound) and bound > 0.055
    envelope = envelope_for(joint_law(cell, effect))
    proof = analytic_evidence(cell)
    cutoff = scalar_reference(0, 1, envelope, 4, proof["refusal_mass_upper"])["cutoff"]
    residual_mean = mean - effect / (2 * 0.9)
    corrected_event_misses = 0.45**4 * sum(
        norm.sf((cutoff - mu) / sigma) + norm.cdf((-cutoff - mu) / sigma)
        for mu, sigma in zip(residual_mean, sd, strict=True)
    )
    assert corrected_event_misses <= 0.05
    periods, ct = draw_periods(cell, 2, streams(MANIFEST["smoke_seed"]))
    assert_public_parity(periods, ct, cell, effect, envelope=envelope)
    assert proof["unconditional_miss_upper"] <= 0.05


@pytest.mark.parametrize("rho", (0.0, 0.5, 0.9))
def test_persistent_noise_not_divided_by_cycles_twice(rho):
    one = Cell(10, 1, 0.5, "centered_gamma", rho, "equal", 0, "retained_total")
    five = replace(one, cycles=5)
    persistent = 0.4**2 + 4 * rho
    assert population_moments(five, 0)[0] == pytest.approx(
        persistent + (population_moments(one, 0)[0] - persistent) / 5
    )


def normal_witness():
    from increment.semantics.unit_cycle import UnitCycleJointLaw

    cell = Cell(40, 1, 0.5, "normal", 0, "equal", 0, "retained_total")
    raw = joint_law(cell, 0).model_dump(mode="python")
    raw["types"] = [
        {
            "weight": 1,
            "cycles": [
                {
                    "ct_mean": 0,
                    "tc_mean": 0,
                    "ct_noise_load": 1,
                    "tc_noise_load": 1,
                    "ct_innovation_count": 1,
                    "tc_innovation_count": 1,
                }
            ],
        }
    ]
    return cell, UnitCycleJointLaw.model_validate(raw)


def test_normal_witness_has_useful_power_and_contracting_width():
    from scipy.stats import norm

    cell, law = normal_witness()
    envelope = envelope_for(law)
    assert envelope.residual_variance_upper == 1
    widths = []
    for n in (40, 160):
        _, mass, _ = admission_oracle(n, 0.5)
        # A slightly larger analytic admission bound is allowed, but is not
        # used as an assertion of the actual test's floating rejection mass.
        reference = scalar_reference(1, 1, envelope, n, float(mass))
        widths.append(2 * reference["cutoff"])
        power = norm.sf((reference["cutoff"] - 1) * math.sqrt(n)) + norm.cdf(
            (-reference["cutoff"] - 1) * math.sqrt(n)
        )
        assert power - float(mass) >= 0.90
    assert 0 < widths[1] < widths[0] < math.inf


def declared_effect_grid(cell, law):
    scale = math.sqrt(float(law_moments(law)["residual_variance"]) / cell.units)
    return tuple(sorted({0.0, 1.0, *(multiple * scale for multiple in (1, 2, 4, 8, 16, 32))}))


def planning_seed(index, namespace):
    return int(
        np.random.SeedSequence([MANIFEST["dgp_seed"], index, namespace]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def potential_stream(cell, rngs):
    """Frozen batch size; checkpoints never split or reseed the generation stream."""
    size = min(512, max(1, 240000 // (cell.units * cell.cycles * 6)))
    while True:
        periods, orders = draw_periods(cell, size, rngs)
        yield from zip(periods, orders, strict=True)


class Accounting:
    """Streaming counts and actual returned-point moments, including every failure."""

    def __init__(self, effect):
        self.effect = effect
        self.r = self.p = self.i = self.h = self.rejected = self.disagreed = 0
        self.mean = self.m2 = 0.0
        self.failures = Counter()
        self.interval_reasons = Counter()
        self.same_order = 0
        self.minimum_lower = self.maximum_upper = self.mean_width = None

    def add(self, periods, ct, cell, envelope):
        self.r += 1
        self.same_order += int(ct.all() or (~ct).all())
        observed = periods + self.effect * np.stack((~ct, ct), -1)
        try:
            src = source(observed, ct, cell, envelope)
            computation = estimate_contrast(src.contrast_stats(src.metrics[0]), procedure(envelope))
            row = computation.results[0]
        except Exception as error:
            reason = (
                f"{error.code}:{dict(error.context)}"
                if isinstance(error, CodedError)
                else f"{type(error).__name__}:{error}"
            )
            self.failures[reason] += 1
            refused, _, _ = admission_oracle(cell.units * cell.cycles, cell.probability_ct)
            self.disagreed += int(int(ct.sum()) not in refused or "split" not in reason)
            return
        self.p += 1
        delta = row.estimate.value - self.mean
        self.mean += delta / self.p
        self.m2 += delta * (row.estimate.value - self.mean)
        if row.estimate.lb is None or row.estimate.ub is None:
            reasons = tuple(
                f"{failure.code}:{dict(failure.context)}"
                for failure in computation.failures.values()
            )
            self.interval_reasons[
                repr(reasons) if reasons else "missing_interval_without_failure"
            ] += 1
            self.disagreed += 1
            return
        try:
            point, slope, _, _ = scalar_observation(observed, ct, cell.probability_ct)
            evidence = next(iter(computation.evidence.values()))
            assert isinstance(evidence, PValueEvidence)
            rejected = evidence.p_value < 0.05
            oracle = scalar_reference(
                point, slope, envelope, cell.units, row.refusal_probability_upper
            )
        except Exception as error:
            self.interval_reasons[f"evidence_or_oracle:{type(error).__name__}:{error}"] += 1
            self.disagreed += 1
            return
        self.i += 1
        self.minimum_lower = (
            row.estimate.lb
            if self.minimum_lower is None
            else min(self.minimum_lower, row.estimate.lb)
        )
        self.maximum_upper = (
            row.estimate.ub
            if self.maximum_upper is None
            else max(self.maximum_upper, row.estimate.ub)
        )
        width = row.estimate.ub - row.estimate.lb
        self.mean_width = (
            width
            if self.mean_width is None
            else self.mean_width + (width - self.mean_width) / self.i
        )
        self.h += int(row.estimate.lb <= self.effect <= row.estimate.ub)
        self.rejected += int(rejected)
        self.disagreed += int(rejected != oracle["rejected"])

    def report(self):
        return {
            "R": self.r,
            "P": self.p,
            "I": self.i,
            "H": self.h,
            "attempted": self.r,
            "point_estimable": self.p,
            "interval_estimable": self.i,
            "hits": self.h,
            "rejected": self.rejected,
            "disagreement": self.disagreed,
            "excluded": 0,
            "failed": self.r - self.p,
            "failure_reasons": dict(self.failures),
            "interval_reasons": dict(self.interval_reasons),
            "all_same_order": self.same_order,
            "minimum_lower_endpoint": self.minimum_lower,
            "maximum_upper_endpoint": self.maximum_upper,
            "mean_interval_width": self.mean_width,
            "exclusion_reasons": {},
            "effect": self.effect,
            "coverage_unconditional": self.h / self.r if self.r else None,
            "coverage_conditional": self.h / self.i if self.i else None,
            "rejection_unconditional": self.rejected / self.r if self.r else None,
            "rejection_conditional": self.rejected / self.i if self.i else None,
            "returned_point_mean": self.mean if self.p else None,
            "returned_point_bias": self.mean - self.effect if self.p else None,
            "returned_point_bias_mcse": math.sqrt(self.m2 / (self.p - 1) / self.p)
            if self.p > 1
            else None,
        }


def checkpoint_gates(account, eta, *, power_target=None):
    """Unconditional bounds retain failures; parity concerns observed draws only."""
    gates = {
        "coverage": gate_decision(Gate("coverage", 0.95, lower=0.945), account.h, account.r, eta),
        "availability": gate_decision(
            Gate("availability", 1, lower=0.99), account.i, account.r, eta
        ),
        "observed_oracle_parity": {
            "status": "certified" if account.disagreed == 0 else "failed",
            "disagreements": account.disagreed,
            "scope": "observed_draws_only",
        },
    }
    split_failures = sum(v for k, v in account.failures.items() if "split" in k)
    unknown = account.r - account.i - split_failures
    lower, _ = cp_bounds(account.rejected, account.r, eta)
    _, upper = cp_bounds(account.rejected + unknown, account.r, eta)
    lower, upper = float(lower), float(upper)
    if account.effect == 0:
        gates["null_size"] = {
            "status": "certified"
            if upper <= 0.055
            else "failed"
            if lower > 0.055
            else "inconclusive",
            "target_upper": 0.055,
        }
    if power_target is not None:
        gates["power"] = {
            "status": "certified"
            if lower >= power_target
            else "failed"
            if upper < power_target
            else "inconclusive",
            "target_lower": power_target,
        }
    for name in ("null_size", "power"):
        if name in gates:
            gates[name].update(lower=lower, upper=upper, eta=eta)
    report = {
        **account.report(),
        "rejection_lower": lower,
        "rejection_upper": upper,
        "unknown_numerical_rejections": unknown,
    }
    return report, gates


def calibrate_cell(cell_index, *, repetitions, mc_error, progress=None):
    """One fixed checkpoint with independent planning and public-runtime streams."""
    from increment.power.unit_cycle import unit_cycle_model_mde, unit_cycle_model_power

    cell = replace(CELLS[cell_index], effect=0.0)
    law = joint_law(cell, 0)
    envelope = envelope_for(law)
    proc = procedure(envelope)
    grid = declared_effect_grid(cell, law)
    planning_error = allocate(allocate(mc_error, 2), 3)
    runtime_eta = allocate(allocate(mc_error, 2), 24)
    model_seed = planning_seed(cell_index, 24001)
    runtime_seed = planning_seed(cell_index, 24002)
    selection = unit_cycle_model_mde(
        law,
        proc,
        n=cell.units,
        effect_grid=grid,
        target_power=0.8,
        repetitions=repetitions,
        seed=model_seed,
        mc_error=planning_error,
    )
    predictions = {
        str(effect): unit_cycle_model_power(
            law,
            proc,
            n=cell.units,
            effect_delta=effect,
            repetitions=repetitions,
            seed=model_seed,
            mc_error=planning_error,
        )
        for effect in (0.0, 1.0)
    }
    accounts = {"null": Accounting(0.0), "nonzero": Accounting(1.0)}
    if selection.feasible_effect is not None:
        accounts["selected"] = Accounting(selection.feasible_effect)
    if progress is not None:
        progress({"kind": "planning_complete", "selection": selection.model_dump(mode="json")})
    potentials = potential_stream(cell, streams(runtime_seed))
    for attempted in range(1, repetitions + 1):
        periods, ct = next(potentials)
        for account in accounts.values():
            account.add(periods, ct, cell, envelope)
        if progress is not None and attempted % 32 == 0:
            progress(
                {
                    "kind": "runtime_progress",
                    "attempted": attempted,
                    "accounts": {name: account.report() for name, account in accounts.items()},
                }
            )
    reports, gates = {}, {}
    for name, account in accounts.items():
        reports[name], gates[name] = checkpoint_gates(
            account, runtime_eta, power_target=0.8 if name == "selected" else None
        )
    comparisons = {}
    for name in ("null", "nonzero"):
        prediction = predictions[str(accounts[name].effect)]
        difference = (
            math.nextafter(reports[name]["rejection_lower"] - prediction.upper_bound, -math.inf),
            math.nextafter(reports[name]["rejection_upper"] - prediction.lower_bound, math.inf),
        )
        comparisons[name] = {
            "runtime_minus_model": difference,
            "resolved_discrepancy": difference[0] > 0 or difference[1] < 0,
            "interpretation": "uncertainty diagnostic, not an equivalence certificate",
        }
    statuses = [gate["status"] for lane in gates.values() for gate in lane.values()]
    status = (
        "failed"
        if "failed" in statuses
        or any(comparison["resolved_discrepancy"] for comparison in comparisons.values())
        else "certified"
        if selection.feasible_effect is not None and all(state == "certified" for state in statuses)
        else "inconclusive"
    )
    return {
        "kind": "checkpoint",
        "cell": cell.id,
        "cell_index": cell_index,
        "status": status,
        "repetitions": repetitions,
        "mc_error": mc_error,
        "planning_error_per_query": planning_error,
        "runtime_error_per_tail": runtime_eta,
        "model_seed": model_seed,
        "runtime_seed": runtime_seed,
        "effect_grid": grid,
        "selection": selection.model_dump(mode="json"),
        "model_power": {
            name: result.model_dump(mode="json") for name, result in predictions.items()
        },
        "reports": reports,
        "gates": gates,
        "model_comparisons": comparisons,
    }


def write_evidence(path, record):
    with path.open("a") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("cell", RELEASE_CELLS, ids=[cell.id for cell in RELEASE_CELLS])
def test_representative_grid_calibration(cell, record_property, tmp_path):
    path = tmp_path / "representative.jsonl"
    result = calibrate_cell(
        CELLS.index(cell),
        repetitions=RELEASE_REPETITIONS,
        mc_error=RELEASE_SCHEDULE["checkpoint_error"][0],
        progress=lambda record: write_evidence(path, record),
    )
    write_evidence(path, result)
    record_property("evidence_path", str(path))
    assert result["status"] == "certified", result


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_normal_witness_has_useful_power_and_analytic_consistency(record_property, tmp_path):
    from scipy.stats import norm

    from increment._unit_cycle import unit_cycle_envelope_cutoff
    from increment.power.unit_cycle import unit_cycle_model_power

    cell, law = normal_witness()
    envelope = envelope_for(law)
    cutoff = unit_cycle_envelope_cutoff(envelope, n=cell.units, alpha=0.05, alternative="two-sided")
    _, mass, _ = admission_oracle(cell.units, 0.5)
    analytic = (1 - float(mass)) * (
        norm.sf((cutoff.value - 1) * math.sqrt(cell.units))
        + norm.cdf((-cutoff.value - 1) * math.sqrt(cell.units))
    )
    checkpoint_error = RELEASE_SCHEDULE["checkpoint_error"][0]
    model = unit_cycle_model_power(
        law,
        procedure(envelope),
        n=cell.units,
        effect_delta=1.0,
        repetitions=RELEASE_REPETITIONS,
        seed=planning_seed(len(CELLS), 24001),
        mc_error=allocate(checkpoint_error, 2),
    )
    rng = np.random.default_rng(planning_seed(len(CELLS), 24002))
    account = Accounting(1.0)
    path = tmp_path / "normal-witness.jsonl"
    for attempted in range(1, RELEASE_REPETITIONS + 1):
        ct = rng.random((cell.units, 1)) < 0.5
        periods = np.stack((~ct, ct), -1) * rng.normal(size=(cell.units, 1, 1))
        account.add(periods, ct, cell, envelope)
        if attempted % 32 == 0:
            write_evidence(path, {"kind": "runtime_progress", **account.report()})
    report, gates = checkpoint_gates(
        account, allocate(allocate(checkpoint_error, 2), 24), power_target=0.8
    )
    write_evidence(
        path,
        {
            "kind": "checkpoint",
            "analytic_power": float(analytic),
            "report": report,
            "gates": gates,
            "model_power": model.model_dump(mode="json"),
            "mc_error": checkpoint_error,
        },
    )
    record_property("evidence_path", str(path))
    assert analytic >= 0.9
    assert model.lower_bound <= analytic <= model.upper_bound
    assert report["rejection_lower"] <= analytic <= report["rejection_upper"]
    assert all(gate["status"] == "certified" for gate in gates.values()), gates


@pytest.mark.slow
def test_zero_envelope_singleton_keeps_ht_point_outside_interval():
    from increment.semantics.unit_cycle import UnitCycleJointLaw

    cell = Cell(4, 1, 0.9, "normal", 0, "equal", 2, "retained_total")
    raw = joint_law(cell).model_dump(mode="python")
    raw["types"] = [
        {
            "weight": 1,
            "cycles": [
                {
                    "ct_mean": 2,
                    "tc_mean": 2,
                    "ct_noise_load": 0,
                    "tc_noise_load": 0,
                    "ct_innovation_count": 1,
                    "tc_innovation_count": 1,
                }
            ],
        }
    ]
    law = UnitCycleJointLaw.model_validate(raw)
    envelope = envelope_for(law)
    assert envelope.residual_variance_upper == 0
    ct = np.ones((4, 1), dtype=bool)
    periods = np.broadcast_to([0.0, 2.0], (4, 1, 2))
    src = source(periods, ct, cell, envelope)
    result = estimate_contrast(src.contrast_stats(src.metrics[0]), procedure(envelope)).results[0]
    assert result.estimate.value == pytest.approx(2 / (2 * 0.9))
    assert result.estimate.lb is not None and result.estimate.ub is not None
    assert result.estimate.lb <= 2 <= result.estimate.ub
    assert result.estimate.lb == pytest.approx(2) and result.estimate.ub == pytest.approx(2)
    assert result.estimate.value < result.estimate.lb
    assert result.standard_error == 0


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ("two-sided", "greater", "less"))
def test_closed_threshold_ties_agree_with_evidence(alternative):
    from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope

    cell = Cell(4, 1, 0.5, "normal", 0, "equal", 0, "retained_total")
    raw = envelope_for(joint_law(cell)).model_dump(mode="python")
    raw["residual_variance_upper"] = 0.2
    envelope = UnitCycleVarianceEnvelope.model_validate(raw)
    from increment._unit_cycle import unit_cycle_envelope_cutoff

    cutoff = unit_cycle_envelope_cutoff(envelope, n=4, alpha=0.05, alternative=alternative)
    effect = -cutoff.value if alternative == "less" else cutoff.value
    ct = np.array([True, False, True, False]).reshape(4, 1)
    periods = effect * np.stack((~ct, ct), -1)
    src = source(periods, ct, cell, envelope)
    computation = estimate_contrast(
        src.contrast_stats(src.metrics[0]), procedure(envelope, alternative)
    )
    result = computation.results[0]
    evidence = next(iter(computation.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value >= 0.05
    assert result.estimate.lb is None or result.estimate.lb <= 0
    assert result.estimate.ub is None or result.estimate.ub >= 0


def test_unequal_population_types_and_nonzero_reference_centering():
    from increment.power.unit_cycle import unit_cycle_law_moments
    from increment.semantics.unit_cycle import UnitCycleJointLaw

    cell = Cell(4, 5, 0.9, "centered_lognormal", 0.9, "unequal", 11, "retained_total")
    raw = joint_law(cell).model_dump(mode="python")
    raw["types"][0]["weight"] = 1
    raw["types"][1]["weight"] = 3
    law = UnitCycleJointLaw.model_validate(raw)
    oracle = law_moments(law)
    actual = unit_cycle_law_moments(law)
    assert float(oracle["reference_effect"]) == pytest.approx(11.2)
    assert oracle["residual_mean"] == 0
    assert oracle["residual_variance"] != oracle["variance_a"]
    assert actual.residual_variance_upper == pytest.approx(float(oracle["residual_variance"]))


@pytest.mark.parametrize("noise", ("normal", "centered_lognormal", "centered_gamma"))
def test_small_skew_public_source_and_sizing_smoke(noise):
    from increment.power.unit_cycle import unit_cycle_power_lower_bound

    cell = Cell(4, 1, 0.9, noise, 0.9, "unequal", 1, "retained_total")
    envelope = envelope_for(joint_law(cell, 0))
    periods, ct = draw_periods(cell, 1, streams(MANIFEST["smoke_seed"]))
    assert_public_parity(periods, ct, cell, 1, envelope=envelope)
    bound = unit_cycle_power_lower_bound(envelope, procedure(envelope), n=4, effect_delta=1)
    assert bound.power_kind == "certified_power_lower_bound"


def test_cp_endpoints_and_incomplete_empty_stream():
    from tests.simulate.unit_cycle_sizing import cp_bounds

    lo, hi = cp_bounds(1, 1, 0.1)
    assert lo <= 0.1 and hi == 1
    lo, hi = cp_bounds(0, 1, 0.1)
    assert lo == 0 and hi >= 0.9
    assert not gate_bounds(Gate("coverage", 0.95, lower=0.945), 0, 0, 0.1)["passed"]
    assert not gate_bounds(Gate("coverage", 0.95, lower=0.945), 0, 10, 0.1)["passed"]


def test_returned_point_accounting_matches_r03_without_dropping_failure():
    from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope

    cell = Cell(4, 1, 0.5, "normal", 0, "equal", 0, "retained_total")
    raw = envelope_for(joint_law(cell)).model_dump(mode="python")
    raw["residual_variance_upper"] = 0.2
    envelope = UnitCycleVarianceEnvelope.model_validate(raw)
    ct = np.array([True, False, True, False]).reshape(4, 1)
    account = Accounting(0.0)
    for effect in (0.0, 2.0):
        account.add(effect * np.stack((~ct, ct), -1), ct, cell, envelope)
    invalid = np.zeros((4, 1, 2))
    invalid[0, 0, 0] = math.nan
    account.add(invalid, ct, cell, envelope)
    assert sum(account.failures.values()) == 1
    reason = next(iter(account.failures))
    # A coded source refusal, not a bare exception, is what gets recorded.
    assert reason.startswith("source.frame.")
    report = account.report()
    assert report["attempted"] == 3
    assert report["point_estimable"] == 2
    assert report["interval_estimable"] == 2
    assert report["excluded"] == 0
    assert report["failed"] == 1
    assert (report["R"], report["P"], report["I"], report["H"]) == (3, 2, 2, 1)
    assert report["coverage_conditional"] == pytest.approx(0.5)
    assert report["coverage_unconditional"] == pytest.approx(1 / 3)
    assert list(report["failure_reasons"].values()) == [1]
    # The two returned points are 0.0 and 2.0, so the mean bias is their average
    # and the MCSE is the sample SD of those two points over sqrt(2).
    assert report["returned_point_bias"] == pytest.approx(1.0, abs=1e-12)
    assert report["returned_point_bias_mcse"] == pytest.approx(1.0, abs=1e-12)


def test_exact_cycle_average_oracle_preserves_strict_evidence_inside_float_display():
    cell = Cell(4, 3, 0.5, "normal", 0, "equal", 0, "retained_total")
    envelope = envelope_for(joint_law(cell)).model_copy(update={"residual_variance_upper": 0.0})
    # Cycle effects (1, 0, 0) give exactly one third with zero residual variance.
    observed = np.zeros((4, 3, 2))
    observed[:, 0, 1] = 1
    ct = np.ones((4, 3), dtype=bool)
    null = float(Fraction(1, 3))
    src = source(observed, ct, cell, envelope)
    computation = estimate_contrast(
        src.contrast_stats(src.metrics[0]), procedure(envelope, null=null)
    )
    row = computation.results[0]
    point, slope, _, _ = scalar_observation(observed, ct, cell.probability_ct)
    assert point == Fraction(1, 3)
    expected = scalar_reference(
        point, slope, envelope, cell.units, row.refusal_probability_upper, null=null
    )
    assert row.estimate.lb == expected["lower"] == null
    assert row.estimate.ub == expected["upper"] == math.nextafter(null, math.inf)
    assert expected["rejected"] is True
    evidence = next(iter(computation.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert expected["p_value"] == evidence.p_value == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("noise", ("normal", "centered_lognormal", "centered_gamma"))
def test_small_functional_proof_and_public_parity(noise, record_property):
    cell = next(
        c
        for c in CELLS
        if c.units == 4
        and c.cycles == 1
        and c.probability_ct == 0.9
        and c.noise == noise
        and c.correlation == 0.5
        and c.counts == "unequal"
        and c.effect == 1
        and c.response_meaning == "retained_total"
    )
    test_representative_independent_analytic_proof(cell, record_property)
    test_representative_public_parity(CELLS.index(cell), cell)
