"""Schedule admission probabilities agree with exact counts and the public loader."""

from fractions import Fraction
from itertools import combinations, product
from math import comb

import polars as pl
import pytest
from scipy.stats import binom

from increment import Randomized, SwitchbackObservationSchedule, from_switchback_frame, power
from increment._unit_cycle import unit_cycle_admission_rule
from increment.errors import InvalidRequestError
from increment.semantics.assignment import (
    FixedCountOrder,
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)


def _fixed(k, m, *, shared=True):
    return SwitchbackAssignment(
        periods_per_cycle=k,
        sequence=FixedCountOrder(
            treated_periods=m,
            independence_unit="shared_block" if shared else "unit_cycle",
        ),
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    )


@pytest.mark.parametrize(
    ("k", "m", "n", "expected"),
    [
        (5, 4, 12, Fraction(78612625, 244140625)),
        (
            5,
            4,
            48,
            5 * Fraction(4, 5) ** 48
            - 10 * Fraction(3, 5) ** 48
            + 10 * Fraction(2, 5) ** 48
            - 5 * Fraction(1, 5) ** 48,
        ),
        (3, 1, 3, Fraction(7, 9)),
        (4, 2, 2, Fraction(5, 6)),
        (20, 10, 2, Fraction(184755, 184756)),
        (5, 2, 2, Fraction(1)),
        (511, 255, 2, Fraction(1)),
        (512, 256, 2, Fraction(1) - Fraction(1, comb(512, 256))),
        (512, 1, 2, Fraction(1)),
    ],
)
def test_fixed_count_refusal_chance_matches_exact_counts(k, m, n, expected):
    probability = power.switchback_schedule_refusal_probability
    assert probability(_fixed(k, m), n) == pytest.approx(float(expected), rel=1e-12)


def test_refusal_chance_survives_heavy_cancellation():
    probability = power.switchback_schedule_refusal_probability
    assert probability(_fixed(200, 100), 2) == 1.0
    with pytest.raises(InvalidRequestError) as caught:
        probability(_fixed(513, 256), 2)
    assert caught.value.code == "power.switchback.baseline"


def test_shared_paired_refusal_chance_counts_single_order_schedules():
    probability = power.switchback_schedule_refusal_probability
    assignment = SwitchbackAssignment(
        sequence=SharedScheduleOrder(probability_ct=0.8),
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    )
    assert probability(assignment, 6) == pytest.approx(0.262208, rel=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize(("m", "shared", "n", "cycles"), [(2, True, 3, 3), (1, False, 2, 2)])
def test_refusal_chance_agrees_with_the_loader_on_every_schedule(m, shared, n, cycles):
    probability = power.switchback_schedule_refusal_probability
    assignment = _fixed(3, m, shared=shared)
    units = ("north",) if shared else ("north", "south")
    draws = cycles if shared else len(units) * cycles
    choices = tuple(combinations(range(3), m))
    empty = pl.DataFrame(
        schema={
            "unit": pl.String,
            "cycle": pl.Int64,
            "period": pl.Int64,
            "step": pl.Int64,
            "outcome": pl.Float64,
        }
    )
    refused = total = 0
    for schedule_draws in product(choices, repeat=draws):
        log = pl.DataFrame(
            [
                {
                    "unit": unit,
                    "cycle": cycle,
                    "period": position,
                    "arm": "T"
                    if position in schedule_draws[cycle if shared else i * cycles + cycle]
                    else "C",
                }
                for i, unit in enumerate(units)
                for cycle in range(cycles)
                for position in range(3)
            ]
        )
        try:
            from_switchback_frame(
                empty,
                unit="unit",
                cycle="cycle",
                period="period",
                step="step",
                group="arm",
                metrics=["outcome"],
                identification=Randomized(
                    control_group="C", allocation={"C": (3 - m) / 3, "T": m / 3}
                ),
                assignment=assignment,
                schedule=SwitchbackObservationSchedule(assignments=log, units=units, cycles=cycles),
            )
        except InvalidRequestError as exc:
            if exc.context.get("reason") not in {
                "missing_position_support",
                "implausible_realized_split",
                "shared_schedule_missing_cycle_order",
            }:
                raise
            assert exc.code == "source.frame.switchback.schedule"
            refused += 1
        total += 1
    assert total == (27 if shared else 81)
    assert refused / total == pytest.approx(
        probability(assignment, n, cycles_per_unit=1 if shared else cycles), abs=1e-12
    )


def test_unit_grain_multiplies_draws_and_shared_grain_does_not():
    probability = power.switchback_schedule_refusal_probability
    expected = float(Fraction(78612625, 244140625))
    assert probability(_fixed(5, 4, shared=False), 3, cycles_per_unit=4) == pytest.approx(
        expected, rel=1e-12
    )
    with pytest.raises(InvalidRequestError) as caught:
        probability(_fixed(5, 4), 3, cycles_per_unit=4)
    assert caught.value.code == "power.switchback.n"


def test_refusal_chance_stays_finite_for_very_many_draws():
    probability = power.switchback_schedule_refusal_probability
    value = probability(_fixed(5, 4), 2**40)
    assert 0 <= value < 1e-300


def test_independent_bernoulli_refusal_chance_is_the_admission_tail():
    probability = power.switchback_schedule_refusal_probability
    p = 0.8
    assignment = SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(probability_ct=p),
        window=SwitchbackWindow(washout_steps=0, observation_steps=1),
    )
    n, cycles = 32, 3
    rule = unit_cycle_admission_rule(n * cycles, p)
    expected = binom.cdf(rule.minimum_ct - 1, n * cycles, p) + binom.sf(
        rule.maximum_ct, n * cycles, p
    )
    assert probability(assignment, n, cycles_per_unit=cycles) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("n", "cycles"),
    [(True, 1), (2.0, 1), (1, 1), (2, True), (2, 1.0), (2, 0), (2**100, 1)],
)
def test_refusal_chance_rejects_unrepresentable_or_noninteger_counts(n, cycles):
    probability = power.switchback_schedule_refusal_probability
    with pytest.raises(InvalidRequestError) as caught:
        probability(_fixed(5, 4, shared=False), n, cycles_per_unit=cycles)
    assert caught.value.code == "power.switchback.n"
