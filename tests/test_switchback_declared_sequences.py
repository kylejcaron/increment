"""Declared multi-period switchback sequence laws (issue #77, schedule slice).

The independent reference here is ``_reference_estimate``: a direct
Horvitz-Thompson construction from the declared law and the panel rows that
shares no code with ``increment.switchback``. Exact design expectations are
checked by enumerating every assignment of a small design with rationals.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from increment import Analysis
from increment.errors import InvalidRequestError, unwrap_coded
from increment.estimation.contrast_results import ContrastResult
from increment.power.switchback import SwitchbackBaseline
from increment.semantics.assignment import (
    DeclaredSequenceOrder,
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized

WINDOW = SwitchbackWindow(washout_steps=1, observation_steps=2)
ONE_CONTROL_OF_THREE = {"CTT": 1 / 3, "TCT": 1 / 3, "TTC": 1 / 3}
UNEQUAL = {"CCT": 0.4, "TTC": 0.6}  # per-period P(T) = (0.6, 0.6, 0.4)


# --------------------------------------------------------------------------- helpers


def _coded(call) -> InvalidRequestError:
    try:
        call()
    except ValidationError as exc:
        try:
            unwrap_coded(exc)
        except InvalidRequestError as coded:
            return coded
        raise
    except InvalidRequestError as coded:
        return coded
    pytest.fail("expected a coded refusal")


def _declared(law, grain="unit_cycle", window=WINDOW) -> SwitchbackAssignment:
    return SwitchbackAssignment.declared(law, independence_unit=grain, window=window)


def _identification(assignment: SwitchbackAssignment) -> Randomized:
    sequence = assignment.sequence
    share = sequence.expected_treated_share if isinstance(sequence, DeclaredSequenceOrder) else 0.5
    return Randomized(
        control_group="control", allocation={"control": 1.0 - share, "treatment": share}
    )


def _panel(
    draws: dict[tuple[str, int], str],
    *,
    outcome,
    washout: int = 1,
    observation: int = 2,
) -> pl.DataFrame:
    """Rows for ``{(unit, cycle): sequence}`` with ``outcome(unit, cycle, period, step, arm)``."""
    rows = []
    for (unit, cycle), sequence in sorted(draws.items()):
        for period, letter in enumerate(sequence):
            arm = "treatment" if letter == "T" else "control"
            for step in range(washout + observation):
                rows.append(
                    {
                        "unit": unit,
                        "cycle": cycle,
                        "period": period,
                        "step": step,
                        "group": arm,
                        "y": outcome(unit, cycle, period, step, arm),
                    }
                )
    return pl.DataFrame(rows)


def _analysis(frame, assignment, *, metrics=None, identification=None):
    return Analysis.from_switchback_panel(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=metrics or {"y": "mean"},
        identification=identification or _identification(assignment),
        assignment=assignment,
    )


def _random_draws(law, *, units, cycles, shared, seed):
    rng = np.random.default_rng(seed)
    keys, probabilities = list(law), np.array(list(law.values()))
    probabilities = probabilities / probabilities.sum()
    block = [keys[rng.choice(len(keys), p=probabilities)] for _ in range(cycles)]
    return {
        (f"u{unit:02d}", cycle): block[cycle]
        if shared
        else keys[rng.choice(len(keys), p=probabilities)]
        for unit in range(units)
        for cycle in range(cycles)
    }


def _noisy_outcome(seed, effect=3.0, trend=0.5):
    rng = np.random.default_rng(seed)
    levels: dict[str, float] = {}

    def outcome(unit, cycle, period, step, arm):
        base = levels.setdefault(unit, float(rng.normal(20, 4)))
        lift = effect * (1.0 + 0.2 * int(unit[1:]) % 3) if arm == "treatment" else 0.0
        return base + trend * (cycle * 3 + period) + lift + float(rng.normal(0, 1))

    return outcome


def _reference_estimate(frame: pl.DataFrame, law, grain, retain_from: int):
    """Independent HT construction: per-period weights from the declared law only."""
    keys = list(law)
    periods = len(keys[0])
    total = math.fsum(law.values())
    pi = [math.fsum(q for key, q in law.items() if key[k] == "T") / total for k in range(periods)]
    kept = frame.filter(pl.col("step") >= retain_from)
    totals = kept.group_by(["unit", "cycle", "period", "group"]).agg(pl.col("y").sum())
    contribution: dict[tuple[str, int], float] = {}
    for unit, cycle, period, group, y in totals.iter_rows():
        weight = 1 / pi[period] if group == "treatment" else -1 / (1 - pi[period])
        contribution[(unit, cycle)] = contribution.get((unit, cycle), 0.0) + weight * y / periods
    if grain == "unit_cycle":
        units = sorted({unit for unit, _ in contribution})
        replicates = [
            np.mean([value for (u, _), value in contribution.items() if u == unit])
            for unit in units
        ]
    else:
        cycles = sorted({cycle for _, cycle in contribution})
        replicates = [
            np.mean([value for (_, c), value in contribution.items() if c == cycle])
            for cycle in cycles
        ]
    replicates = np.asarray(replicates)
    return replicates.mean(), replicates.std(ddof=1) / math.sqrt(len(replicates)), len(replicates)


# ----------------------------------------------------------------- law declaration


@pytest.mark.parametrize(
    ("law", "reason"),
    [
        ({"CT": 1.0}, "deterministic_schedule"),
        ({"CT": 0.5, "CC": 0.5}, "period_positivity"),
        ({"CTT": 0.5, "TTC": 0.5}, "period_positivity"),
        ({"CT": 0.5, "TCT": 0.5}, "sequence_lengths"),
        ({"CT": 0.4, "TC": 0.4}, "probability_sum"),
        ({"CT": 0.5, "TX": 0.5}, "sequence_alphabet"),
        ({"CT": 0.0, "TC": 1.0}, "probability_range"),
        ({"CT": True, "TC": 0.5}, "probability_type"),
    ],
)
def test_invalid_declared_laws_refuse_with_stable_codes(law, reason):
    refusal = _coded(lambda: DeclaredSequenceOrder(sequences=law, independence_unit="unit_cycle"))
    assert refusal.code == "definition.assignment.sequence_law"
    assert refusal.context["reason"] == reason


def test_period_count_must_match_the_declared_law():
    law = DeclaredSequenceOrder(sequences=ONE_CONTROL_OF_THREE, independence_unit="unit_cycle")
    with pytest.raises(ValidationError, match="periods_per_cycle"):
        SwitchbackAssignment(periods_per_cycle=2, sequence=law, window=WINDOW)
    with pytest.raises(ValidationError, match="periods_per_cycle"):
        SwitchbackAssignment(
            periods_per_cycle=3, sequence=IndependentBernoulliOrder(), window=WINDOW
        )


def test_declared_law_round_trips_and_reports_marginal_exposure():
    assignment = _declared(UNEQUAL, "shared_block")
    restored = SwitchbackAssignment.model_validate_json(assignment.model_dump_json())
    assert restored == assignment
    assert hash(restored) == hash(assignment)
    assert assignment.periods_per_cycle == 3
    assert assignment.sequence.treatment_probabilities == pytest.approx((0.6, 0.6, 0.4))
    assert assignment.sequence.expected_treated_share == pytest.approx(1.6 / 3)
    # Paired declarations keep their existing serialized shape.
    paired = SwitchbackAssignment(sequence=SharedScheduleOrder(probability_ct=0.6), window=WINDOW)
    assert paired.model_dump()["periods_per_cycle"] == 2


def test_allocation_must_match_declared_exposure():
    assignment = _declared(UNEQUAL)
    draws = _random_draws(UNEQUAL, units=6, cycles=4, shared=False, seed=1)
    frame = _panel(draws, outcome=_noisy_outcome(1))
    refusal = _coded(
        lambda: _analysis(
            frame,
            assignment,
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
        )
    )
    assert refusal.code == "source.frame.switchback.identification"
    assert refusal.context["reason"] == "allocation_exposure_mismatch"


# ------------------------------------------------- independent reference and equivalence


@pytest.mark.parametrize("grain", ["unit_cycle", "shared_block"])
@pytest.mark.parametrize("law", [ONE_CONTROL_OF_THREE, UNEQUAL])
def test_runtime_matches_independent_ht_construction(law, grain):
    shared = grain == "shared_block"
    draws = _random_draws(
        law, units=12 if not shared else 4, cycles=5 if not shared else 16, shared=shared, seed=7
    )
    frame = _panel(draws, outcome=_noisy_outcome(7))
    result = _analysis(frame, _declared(law, grain)).run()[0]
    expected, standard_error, replicates = _reference_estimate(frame, law, grain, retain_from=1)
    assert isinstance(result, ContrastResult)
    assert result.estimate.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result.standard_error == pytest.approx(standard_error, rel=1e-10)
    assert result.dof == replicates - 1
    assert result.probability_ct is None
    assert dict(result.sequence_probabilities) == pytest.approx(law)
    assert result.method == ("switchback_block_t" if shared else "switchback_unit_t_approximation")


@pytest.mark.parametrize("p", [0.5, 0.65])
@pytest.mark.parametrize("grain", ["unit_cycle", "shared_block"])
def test_two_period_declared_law_reproduces_paired_results(p, grain):
    shared = grain == "shared_block"
    law = {"CT": p, "TC": 1 - p}
    draws = _random_draws(
        law, units=10 if not shared else 3, cycles=6 if not shared else 14, shared=shared, seed=3
    )
    frame = _panel(draws, outcome=_noisy_outcome(3))
    paired_law = (
        SharedScheduleOrder(probability_ct=p)
        if shared
        else IndependentBernoulliOrder(probability_ct=p)
    )
    paired = _analysis(frame, SwitchbackAssignment(sequence=paired_law, window=WINDOW)).run()[0]
    declared = _analysis(frame, _declared(law, grain)).run()[0]
    assert declared.estimate.value == pytest.approx(paired.estimate.value, rel=1e-12)
    assert declared.estimate.lb == pytest.approx(paired.estimate.lb, rel=1e-12)
    assert declared.estimate.ub == pytest.approx(paired.estimate.ub, rel=1e-12)
    assert declared.standard_error == pytest.approx(paired.standard_error, rel=1e-12)
    assert (declared.dof, declared.n_units, declared.n_blocks) == (
        paired.dof,
        paired.n_units,
        paired.n_blocks,
    )


def test_paired_payloads_do_not_gain_sequence_fields():
    law = {"CT": 0.5, "TC": 0.5}
    frame = _panel(
        _random_draws(law, units=6, cycles=3, shared=False, seed=2), outcome=_noisy_outcome(2)
    )
    analysis = _analysis(
        frame, SwitchbackAssignment(sequence=IndependentBernoulliOrder(), window=WINDOW)
    )
    result = analysis.run()[0]
    assert "sequence_probabilities" not in result.model_dump()
    diagnostic = analysis.assignment_diagnostic().model_dump()
    assert not {"sequence_probabilities", "sequence_counts", "realized_treated_share"} & set(
        diagnostic
    )
    declared = _analysis(frame, _declared(law)).run()[0]
    assert ContrastResult.model_validate_json(declared.model_dump_json()) == declared


@pytest.mark.parametrize(
    "law", [ONE_CONTROL_OF_THREE, UNEQUAL, {"CC": 0.25, "CT": 0.25, "TC": 0.25, "TT": 0.25}]
)
def test_exact_design_expectation_equals_the_estimand(law):
    """Enumerate every unit-cycle draw of a 2-unit x 2-cycle design with rationals.

    Fixed potential outcomes include unit, period and calendar heterogeneity;
    the HT estimate's expectation under the declared law must equal the
    period-averaged retained-window effect exactly.
    """
    keys = list(law)
    periods = len(keys[0])
    q = {key: Fraction(value).limit_denominator(1000) for key, value in law.items()}
    pi = [sum((q[k] for k in keys if k[p] == "T"), Fraction(0)) for p in range(periods)]
    cells = [(unit, cycle) for unit in range(2) for cycle in range(2)]

    def y(unit, cycle, period, treated):
        return Fraction(5 + 3 * unit + 2 * cycle - period) + (
            Fraction(4 + unit + period) if treated else 0
        )

    estimand = Fraction(0)
    for unit, cycle in cells:
        for period in range(periods):
            estimand += y(unit, cycle, period, True) - y(unit, cycle, period, False)
    estimand /= len(cells) * periods

    expectation = Fraction(0)
    for assignment in itertools.product(keys, repeat=len(cells)):
        probability = Fraction(1)
        units = {0: Fraction(0), 1: Fraction(0)}
        for (unit, cycle), sequence in zip(cells, assignment, strict=True):
            probability *= q[sequence]
            for period, letter in enumerate(sequence):
                treated = letter == "T"
                weight = 1 / pi[period] if treated else -1 / (1 - pi[period])
                units[unit] += weight * y(unit, cycle, period, treated) / (periods * 2)
        expectation += probability * (units[0] + units[1]) / 2
    assert expectation == estimand


def test_runtime_matches_reference_on_every_enumerated_draw():
    """Public API equals the independent construction across a full small enumeration."""
    keys = list(ONE_CONTROL_OF_THREE)
    assignment = _declared(ONE_CONTROL_OF_THREE)
    cells = [("u0", 0), ("u0", 1), ("u1", 0), ("u1", 1)]
    checked = refused = 0
    for draw in itertools.product(keys, repeat=len(cells)):
        draws = dict(zip(cells, draw, strict=True))
        frame = _panel(
            draws,
            outcome=lambda unit, cycle, period, step, arm: (
                10.0
                + int(unit[1]) * 3
                + cycle
                + period * 0.5
                + step
                + (2.0 if arm == "treatment" else 0)
            ),
        )
        try:
            result = _analysis(frame, assignment).run()[0]
        except InvalidRequestError as exc:
            assert exc.context["reason"] == "insufficient_exposure_support"
            refused += 1
            continue
        expected, _, _ = _reference_estimate(frame, ONE_CONTROL_OF_THREE, "unit_cycle", 1)
        assert result.estimate.value == pytest.approx(expected, rel=1e-12)
        checked += 1
    assert checked > 0 and refused > 0


# ------------------------------------------------------------- integrity and refusals


def test_undeclared_realized_sequence_refuses():
    draws = _random_draws(ONE_CONTROL_OF_THREE, units=6, cycles=3, shared=False, seed=4)
    draws[("u00", 0)] = "CCT"
    refusal = _coded(
        lambda: _analysis(_panel(draws, outcome=_noisy_outcome(4)), _declared(ONE_CONTROL_OF_THREE))
    )
    assert refusal.code == "source.frame.switchback.schedule"
    assert refusal.context["reason"] == "undeclared_sequence"


def test_shared_law_requires_one_sequence_per_block():
    draws = _random_draws(ONE_CONTROL_OF_THREE, units=3, cycles=8, shared=True, seed=5)
    draws[("u01", 2)] = next(key for key in ONE_CONTROL_OF_THREE if key != draws[("u00", 2)])
    refusal = _coded(
        lambda: _analysis(
            _panel(draws, outcome=_noisy_outcome(5)),
            _declared(ONE_CONTROL_OF_THREE, "shared_block"),
        )
    )
    assert refusal.context["reason"] == "shared_block_sequence_mismatch"


@pytest.mark.parametrize("grain", ["unit_cycle", "shared_block"])
def test_period_without_both_arms_refuses_as_insufficient_exposure(grain):
    draws = {(f"u{unit}", cycle): "CTT" for unit in range(3) for cycle in range(3)}
    refusal = _coded(
        lambda: _analysis(
            _panel(draws, outcome=_noisy_outcome(6)), _declared(ONE_CONTROL_OF_THREE, grain)
        )
    )
    assert refusal.code == "source.frame.switchback.schedule"
    assert refusal.context["reason"] == "insufficient_exposure_support"
    assert refusal.context["period"] == 0


def test_counts_incompatible_with_the_declared_law_refuse():
    law = {"CTT": 0.98, "TCT": 0.01, "TTC": 0.01}
    # 60 draws split evenly is overwhelming evidence against a 98/1/1 law.
    cycles = ["CTT", "TCT", "TTC"]
    draws = {
        (f"u{unit:02d}", cycle): cycles[(unit + cycle) % 3]
        for unit in range(20)
        for cycle in range(3)
    }
    refusal = _coded(lambda: _analysis(_panel(draws, outcome=_noisy_outcome(8)), _declared(law)))
    assert refusal.context["reason"] == "implausible_realized_sequences"


def test_panel_period_domain_must_match_declared_periods():
    draws = _random_draws({"CT": 0.5, "TC": 0.5}, units=4, cycles=3, shared=False, seed=9)
    refusal = _coded(
        lambda: _analysis(_panel(draws, outcome=_noisy_outcome(9)), _declared(ONE_CONTROL_OF_THREE))
    )
    assert refusal.code == "source.frame.switchback.domain"


def test_diagnostic_separates_declared_probabilities_from_realized_counts():
    draws = _random_draws(UNEQUAL, units=8, cycles=5, shared=False, seed=10)
    analysis = _analysis(_panel(draws, outcome=_noisy_outcome(10)), _declared(UNEQUAL))
    diagnostic = analysis.assignment_diagnostic()
    counts = {key: list(draws.values()).count(key) for key in UNEQUAL}
    assert dict(diagnostic.sequence_counts) == {key: n for key, n in counts.items() if n}
    assert dict(diagnostic.sequence_probabilities) == pytest.approx(UNEQUAL)
    assert diagnostic.treatment_probabilities == pytest.approx((0.6, 0.6, 0.4))
    treated = sum(sequence.count("T") for sequence in draws.values())
    assert diagnostic.realized_treated_share == pytest.approx(treated / (3 * len(draws)))
    assert diagnostic.ct_cycles is None and diagnostic.tc_cycles is None
    assert diagnostic.randomization_law == "independent_declared_sequence"


def test_planning_refuses_until_declared_sequence_planning_exists():
    draws = _random_draws(ONE_CONTROL_OF_THREE, units=8, cycles=3, shared=False, seed=11)
    analysis = _analysis(_panel(draws, outcome=_noisy_outcome(11)), _declared(ONE_CONTROL_OF_THREE))
    refusal = _coded(lambda: analysis.planning_baseline("y"))
    assert refusal.code == "source.frame.switchback.assignment"
    assert refusal.context["reason"] == "declared_sequence_planning_unfinished"
    baseline = _coded(
        lambda: SwitchbackBaseline(
            assignment=_declared(ONE_CONTROL_OF_THREE),
            metric="y",
            control_group="control",
            treatment_group="treatment",
            aggregation="sum",
            estimand="retained_window_total_difference",
            cycles_per_unit=3,
            delta_ref=0.0,
            sd_a=1.0,
            sd_g=0.0,
            rho=0.0,
        )
    )
    assert baseline.code == "power.switchback.baseline"


def test_variance_envelope_is_refused_for_declared_laws():
    from increment.semantics.unit_cycle import UnitCycleVarianceEnvelope
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.5, cycles=3, metric="y")
    payload = {**reference.model_dump(), "assignment": _declared(ONE_CONTROL_OF_THREE).model_dump()}
    refusal = _coded(lambda: UnitCycleVarianceEnvelope.model_validate(payload))
    assert refusal.code == "definition.unit_cycle.assignment"


def test_conversion_metrics_use_any_semantics_per_period():
    draws = _random_draws(ONE_CONTROL_OF_THREE, units=10, cycles=4, shared=False, seed=12)
    frame = _panel(
        draws,
        outcome=lambda unit, cycle, period, step, arm: float(step == 2 and arm == "treatment"),
    )
    result = _analysis(frame, _declared(ONE_CONTROL_OF_THREE), metrics={"y": "conversion"}).run()[0]
    assert result.estimand == "retained_window_conversion_difference"
    expected, _, _ = _reference_estimate(frame, ONE_CONTROL_OF_THREE, "unit_cycle", retain_from=1)
    assert result.estimate.value == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------- calibration sentinel


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("grain", ["unit_cycle", "shared_block"])
def test_interval_coverage_under_trend_and_heterogeneous_effects(grain):
    """Nominal 95% intervals cover the realized-design estimand near nominal.

    Calendar trends, unit-level heterogeneous effects and unequal exposure are
    all present. Undercoverage beyond a 3-sigma binomial band at 400 repetitions fails.
    """
    shared = grain == "shared_block"
    repetitions, covered = 400, 0
    assignment = _declared(UNEQUAL, grain)
    for seed in range(repetitions):
        rng = np.random.default_rng(10_000 + seed)
        draws = _random_draws(
            UNEQUAL, units=4 if shared else 20, cycles=24 if shared else 4, shared=shared, seed=seed
        )
        levels = {unit: rng.normal(20, 4) for unit, _ in draws}
        effects = {unit: rng.normal(3, 1) for unit, _ in draws}

        def outcome(unit, cycle, period, step, arm, levels=levels, effects=effects, rng=rng):
            lift = effects[unit] if arm == "treatment" else 0.0
            return float(levels[unit] + 0.4 * (3 * cycle + period) + lift + rng.normal(0, 2))

        frame = _panel(draws, outcome=outcome)
        try:
            result = _analysis(frame, assignment).run()[0]
        except InvalidRequestError as exc:
            assert exc.context["reason"] == "insufficient_exposure_support"
            repetitions -= 1
            continue
        # Two retained steps per period: the per-period total effect is 2x the unit effect.
        target = 2.0 * float(np.mean(list(effects.values())))
        covered += result.estimate.lb <= target <= result.estimate.ub
    coverage = covered / repetitions
    band = 3 * math.sqrt(0.95 * 0.05 / repetitions)
    # Unit-effect heterogeneity makes the replicate variance conservative for the
    # sample-average target, so only undercoverage is a failure; gross
    # overcoverage would signal a broken variance and is bounded too.
    assert 0.95 - band <= coverage <= 0.995, coverage
