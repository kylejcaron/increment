"""Focused tests for the eager narwhals switchback frame source."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import cast

import polars as pl
import pytest

from increment import Analysis
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.contrast import ContrastStats
from increment.estimation.contrast_results import ContrastResult, ContrastResults
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.frame import MetricSpec, from_switchback_panel
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    ParallelAssignment,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.semantics.models import Winsorization
from increment.semantics.unit_cycle import UnitCycleTApproximation


def test_prospective_reference_binding_and_sufficient_state_are_preserved():
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.75, cycles=2, metric="mean").model_copy(
        update={"assignment": _assignment()}
    )
    references = {"mean": reference}
    source = _source(metrics={"mean": "mean"}, contrast_references=references)
    references.clear()
    assert source.contrast_references["mean"] == reference
    assert source.context.procedures["mean"].reference == reference
    with pytest.raises(TypeError):
        source.contrast_references["mean"] = UnitCycleTApproximation()  # type: ignore[index]
    stats = source.contrast_stats(source.metrics[0])
    assert stats.mean_slope == pytest.approx(4 / 3)
    assert (stats.ct_cycles, stats.tc_cycles) == (2, 2)
    assert stats.minimum_cycles_per_unit == stats.maximum_cycles_per_unit == 2
    assert stats.washout_steps == 1
    assert ContrastStats.model_validate_json(stats.model_dump_json()) == stats


def test_ordinary_panel_uses_independent_units_for_uncertainty():
    from scipy.stats import t

    analysis = Analysis.from_switchback_panel(
        pl.DataFrame(_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=_assignment(),
    )

    result = analysis.run()[0]
    assert isinstance(result, ContrastResult)
    # Cycle-weighted unit contrasts are 42 and -102, not eight independent periods.
    assert result.estimate.value == pytest.approx(-30)
    assert result.standard_error == pytest.approx(72)
    assert result.dof == 1
    critical = t.isf(result.alpha / 2, 1)
    assert result.estimate.lb == pytest.approx(-30 - critical * 72)
    assert result.estimate.ub == pytest.approx(-30 + critical * 72)


def test_switchback_analysis_run_reports_unsupported_assignment_integrity():
    analysis = Analysis.from_switchback_panel(
        pl.DataFrame(_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=_assignment(),
    )

    results = analysis.run()

    assert results.metadata is not None
    (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
    assert integrity.status == "unsupported_assignment"
    assert integrity.code == "integrity.switchback_assignment_law"


def test_artifact_context_refuses_switchback_source_with_method_context():
    analysis = Analysis.from_switchback_panel(
        pl.DataFrame(_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=_assignment(),
    )
    with pytest.raises(CapabilityError) as raised:
        _ = analysis.artifact_context

    assert raised.value.code == "facade.analysis.contrast_unavailable"
    assert raised.value.context["method"] == "artifact_context"


@pytest.mark.parametrize("mismatch", ["assignment", "groups", "conversion", "unknown"])
def test_invalid_envelope_identity_refuses_before_frame_access(monkeypatch, mismatch):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.75, cycles=2, metric="mean").model_copy(
        update={"assignment": _assignment()}
    )
    metrics = {"mean": "mean"}
    if mismatch == "assignment":
        reference = reference.model_copy(update={"assignment": _assignment(p_ct=0.5)})
    elif mismatch == "groups":
        reference = reference.model_copy(update={"control_group": "other"})
    elif mismatch == "conversion":
        metrics = {"mean": "conversion"}
    else:
        reference = reference.model_copy(update={"metric": "unknown"})

    def fail(*args, **kwargs):
        pytest.fail("reference identity must be checked before frame access")

    monkeypatch.setattr("increment.switchback.nw.from_native", fail)
    with pytest.raises(InvalidRequestError) as caught:
        from_switchback_panel(
            pl.DataFrame(),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics=metrics,
            identification=_identification(),
            assignment=_assignment(),
            contrast_references={reference.metric: reference},
        )
    assert caught.value.code == "unit_cycle.reference_mismatch"


def test_envelope_cycle_mismatch_refuses_before_reduction(monkeypatch):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.75, cycles=1, metric="mean").model_copy(
        update={"assignment": _assignment()}
    )

    def fail(*args, **kwargs):
        pytest.fail("cycle identity must be checked before reducing outcomes")

    monkeypatch.setattr("increment.switchback._build_stats", fail)
    with pytest.raises(InvalidRequestError) as caught:
        _source(metrics={"mean": "mean"}, contrast_references={"mean": reference})
    assert caught.value.code == "unit_cycle.reference_mismatch"
    assert caught.value.context["reason"] == "cycles_per_unit"


@pytest.mark.parametrize("n", [1, 4])
def test_same_order_zero_sample_se_has_finite_envelope_inference(n):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.9, metric="mean")
    rows = [
        {
            "unit": f"u{u}",
            "cycle": 0,
            "period": period,
            "step": 0,
            "group": "control" if period == 0 else "treatment",
            "mean": float(period),
        }
        for u in range(n)
        for period in range(2)
    ]
    analysis = Analysis.from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=reference.assignment,
        contrast_references={"mean": reference},
    )
    result = analysis.run()[0]
    assert isinstance(result, ContrastResult)
    assert result.estimate.value == pytest.approx(1 / 1.8)
    assert result.mean_slope == pytest.approx(1 / 1.8)
    assert result.standard_error == (None if n == 1 else 0)
    assert result.estimate.lb is not None and result.estimate.ub is not None
    assert (result.ct_cycles, result.tc_cycles) == (n, 0)


def test_ht_cycle_average_avoids_overflow_of_a_representable_mean():
    from increment.switchback import _unit_cycle_average

    assert _unit_cycle_average([1e308, 1e308]) == 1e308
    assert _unit_cycle_average([1e308, 1e308, -1e308, -1e308]) == 0


@pytest.mark.parametrize(
    "dtype,values",
    [
        (
            "float64",
            [1.1, -2.2, 5e-324, -5e-324, 1e308, -1e308, 0.0, -0.0, 2.0**-1074, 3.0 * 2.0**1000],
        ),
        ("int64", [2**63 - 1, -(2**63), 1, -1, 0, 2**53 + 1, -(2**53) - 1]),
        ("uint64", [2**64 - 1, 0, 1, 2**32, 2**32 - 1, 2**63]),
    ],
)
def test_exact_group_sums_match_rational_sums_on_adversarial_scalars(dtype, values):
    import numpy as np

    from increment.switchback import _exact_group_sums

    array = np.asarray(values, dtype=dtype)
    assert array.tolist() == values
    signs = np.array([1 if index % 3 else -1 for index in range(len(values))])
    groups = np.arange(len(values)) % 2
    expected = [
        sum(
            (
                Fraction(int(sign)) * Fraction(value)
                for value, sign, group in zip(values, signs, groups, strict=True)
                if group == target
            ),
            Fraction(0),
        )
        for target in range(2)
    ]
    assert _exact_group_sums(array, signs, groups, 2) == expected


def _exact_envelope_source(rows, *, steps=1, cycles=1):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.75, variance=0, cycles=cycles, metric="mean").model_copy(
        update={"assignment": _assignment(washout=0, observation=steps)}
    )
    return from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=reference.assignment,
        contrast_references={"mean": reference},
    ), reference


@pytest.mark.parametrize(
    "effect,null,p_value",
    [
        (1.1, 1.1, 1.0),
        (math.nextafter(1.0, math.inf), 1.0, 0.0),
    ],
)
def test_exact_raw_scalar_envelope_residual(effect, null, p_value):
    from increment.estimation.contrast import estimate_contrast
    from tests.estimation.test_unit_cycle_envelope import procedure

    rows = [
        {
            "unit": "u0",
            "cycle": 0,
            "period": period,
            "step": 0,
            "group": "control" if period == 0 else "treatment",
            "mean": 0.0 if period == 0 else effect,
        }
        for period in range(2)
    ]
    source, reference = _exact_envelope_source(rows)
    stats = source.contrast_stats(source.metrics[0])
    exact = Fraction(effect) * Fraction(2, 3)
    assert stats.exact_delta_total == (exact.numerator, exact.denominator)
    result = estimate_contrast(stats, procedure(reference, metric="mean", null_abs=null)).results[0]
    assert result.estimate.value == float(exact)
    assert result.residual_p_value == p_value
    assert result.refusal_probability_upper == result.residual_cutoff == 0
    assert result.estimate.lb == result.estimate.ub == effect


def _five_row_true_null_rows(orders):
    return [
        {
            "unit": unit,
            "cycle": cycle,
            "period": period,
            "step": step,
            "group": "control" if (period == 0) == ct else "treatment",
            "mean": 0.0 if (period == 0) == ct and step == 4 else 1.1,
        }
        for unit, unit_orders in orders.items()
        for cycle, ct in enumerate(unit_orders)
        for period in range(2)
        for step in range(5)
    ]


@pytest.mark.parametrize("ct", [True, False])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_five_raw_rows_true_null_is_exact_under_row_permutations(ct, alternative):
    from increment.estimation.contrast import estimate_contrast
    from tests.estimation.test_unit_cycle_envelope import procedure

    rows = _five_row_true_null_rows({"u0": [ct]})
    assert math.fsum([1.1] * 5) == 5.5
    assert 5 * Fraction(1.1) - Fraction(5.5) == Fraction(1, 2**51)
    exact = Fraction(1.1) * (Fraction(2, 3) if ct else 2)
    results = []
    for ordered in (rows, rows[::-1], rows[3:] + rows[:3], rows[::2] + rows[1::2]):
        source, reference = _exact_envelope_source(ordered, steps=5)
        stats = source.contrast_stats(source.metrics[0])
        assert stats.exact_delta_total == (exact.numerator, exact.denominator)
        result = estimate_contrast(
            stats, procedure(reference, metric="mean", null_abs=1.1, alternative=alternative)
        ).results[0]
        assert result.estimate.value == float(exact)
        assert result.residual_p_value == 1.0
        if alternative != "less":
            assert result.estimate.lb == 1.1
        if alternative != "greater":
            assert result.estimate.ub == 1.1
        assert result.refusal_probability_upper == result.residual_cutoff == 0
        results.append(result.model_dump_json())
    assert all(result == results[0] for result in results)


def test_five_raw_rows_exact_state_survives_cycle_and_partition_permutations():
    from itertools import permutations

    from increment.estimation.contrast import (
        ContrastPartition,
        estimate_contrast,
        reduce_contrast_partitions,
    )
    from tests.estimation.test_unit_cycle_envelope import procedure

    orders = {"a": [True, False], "b": [True, False], "c": [True, False]}
    rows = _five_row_true_null_rows(orders)
    whole, reference = _exact_envelope_source(rows, steps=5, cycles=2)
    whole_stats = whole.contrast_stats(whole.metrics[0])
    exact_mean = Fraction(1.1) * Fraction(4, 3)
    exact_total = 3 * exact_mean
    assert whole_stats.exact_delta_total == (exact_total.numerator, exact_total.denominator)
    expected = estimate_contrast(
        whole_stats, procedure(reference, metric="mean", null_abs=1.1)
    ).results[0]
    assert expected.residual_p_value == 1.0
    assert expected.estimate.value == float(exact_mean)
    parts = []
    for unit in orders:
        unit_rows = [row for row in rows[::-1] if row["unit"] == unit]
        source, _ = _exact_envelope_source(unit_rows, steps=5, cycles=2)
        stats = source.contrast_stats(source.metrics[0])
        assert stats.exact_delta_total is not None
        assert stats.mean_slope is not None and stats.ct_cycles is not None
        part = ContrastPartition(
            metric=stats.metric,
            aggregation=stats.aggregation,
            probability_ct=stats.probability_ct,
            randomization_law=stats.randomization_law,
            independence_grain=stats.independence_grain,
            carryover_order=stats.carryover_order,
            observation_steps=stats.observation_steps,
            retained_steps=stats.retained_steps,
            control_group=stats.control_group,
            treatment_group=stats.treatment_group,
            washout_steps=stats.washout_steps,
            unit_deltas={unit: math.fsum((stats.reference_delta, stats.mean_residual))},
            exact_unit_deltas={unit: stats.exact_delta_total},
            cycles_by_unit={unit: stats.n_cycles},
            unit_slopes={unit: stats.mean_slope},
            ct_counts_by_unit={unit: stats.ct_cycles},
        )
        parts.append(ContrastPartition.model_validate_json(part.model_dump_json()))
    for ordered in permutations(parts):
        merged = reduce_contrast_partitions(ordered)
        assert merged.exact_delta_total == whole_stats.exact_delta_total
        actual = estimate_contrast(
            merged, procedure(reference, metric="mean", null_abs=1.1)
        ).results[0]
        assert actual.model_dump_json() == expected.model_dump_json()
    reversed_cycles = [dict(row, cycle=1 - row["cycle"]) for row in rows[::-1]]
    reordered, _ = _exact_envelope_source(reversed_cycles, steps=5, cycles=2)
    assert (
        reordered.contrast_stats(reordered.metrics[0]).exact_delta_total
        == whole_stats.exact_delta_total
    )


def test_exhausted_admission_budget_refuses_before_response_reduction(monkeypatch):
    from increment.semantics.models import AnalysisPlan
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.5, metric="mean")
    rows = [
        {
            "unit": f"u{u}",
            "cycle": 0,
            "period": period,
            "step": 0,
            "group": "control" if (period == 0) == (u < 20) else "treatment",
            "mean": float(period),
        }
        for u in range(40)
        for period in range(2)
    ]

    def fail(*args, **kwargs):
        pytest.fail("admission budget must be checked before response reduction")

    monkeypatch.setattr("increment.switchback._build_stats", fail)
    with pytest.raises(InvalidRequestError) as caught:
        from_switchback_panel(
            pl.DataFrame(rows),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=reference.assignment,
            contrast_references={"mean": reference},
            plan=AnalysisPlan(primary="mean", alpha=1e-20),
        )
    assert caught.value.code == "unit_cycle.error_budget_exhausted"


def _assignment(
    *, washout: int = 1, observation: int = 2, p_ct: float = 0.75, carryover_order: int = 0
):
    return SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(probability_ct=p_ct),
        window=SwitchbackWindow(
            washout_steps=washout,
            observation_steps=observation,
            carryover_order=carryover_order,
        ),
    )


def _identification() -> Randomized:
    return Randomized(
        control_group="control",
        allocation={"control": 0.5, "treatment": 0.5},
    )


def test_unit_cycle_cloning_preserves_independent_n_and_uncertainty():
    from increment.estimation.contrast import estimate_contrast
    from increment.estimation.decision_types import ContrastDecisionProcedure

    rows = _rows()
    cloned = rows + [dict(row, cycle=cast(int, row["cycle"]) + 2) for row in rows]
    proc = ContrastDecisionProcedure(
        reference=UnitCycleTApproximation(),
        metric="mean",
        role="primary",
        alpha=0.05,
        alternative="two-sided",
        null_abs=0.0,
    )
    results = []
    for panel in (rows, cloned):
        src = _source(panel, metrics={"mean": "mean"}, assignment=_assignment(p_ct=0.5))
        stats = src.contrast_stats(src.metrics[0])
        results.append(estimate_contrast(stats, proc).results[0])
    original, repeated = results
    assert original.n_units == repeated.n_units == 2
    assert repeated.n_cycles == 2 * original.n_cycles
    assert repeated.dof == original.dof
    assert repeated.estimate == original.estimate
    assert repeated.standard_error == original.standard_error
    with pytest.raises(InvalidRequestError) as caught:
        _source(rows + rows)
    assert caught.value.context["reason"] == "duplicate_schedule_cells"


def test_all_same_order_unit_point_has_no_fabricated_interval():
    from increment.estimation.contrast import estimate_contrast
    from increment.estimation.decision_types import ContrastDecisionProcedure

    rows = [
        {
            "unit": f"u{u}",
            "cycle": 0,
            "period": period,
            "step": 0,
            "group": "control" if period == 0 else "treatment",
            "mean": float(period),
        }
        for u in range(4)
        for period in range(2)
    ]
    src = _source(
        rows, metrics={"mean": "mean"}, assignment=_assignment(washout=0, observation=1, p_ct=0.9)
    )
    result = estimate_contrast(
        src.contrast_stats(src.metrics[0]),
        ContrastDecisionProcedure(
            reference=UnitCycleTApproximation(),
            metric="mean",
            role="primary",
            alpha=0.05,
            alternative="two-sided",
            null_abs=0.0,
        ),
    )
    row = result.results[0]
    assert row.estimate.value == pytest.approx(1 / 1.8)
    assert row.standard_error == 0
    assert row.estimate.lb is None and row.estimate.ub is None
    failure = next(iter(result.failures.values()))
    assert failure.code == "evidence.p_value.unavailable"
    assert failure.context["reason"] == "zero_standard_error"
    assert row.model_dump(mode="json")["estimate"]["lb"] is None
    varied = [
        dict(
            item,
            mean=cast(float, item["mean"]) + int(str(item["unit"])[1:]) * cast(int, item["period"]),
        )
        for item in rows
    ]
    other = _source(
        varied,
        metrics={"mean": "mean"},
        assignment=_assignment(washout=0, observation=1, p_ct=0.9),
    )
    available = estimate_contrast(
        other.contrast_stats(other.metrics[0]),
        ContrastDecisionProcedure(
            reference=UnitCycleTApproximation(),
            metric="mean",
            role="primary",
            alpha=0.05,
            alternative="two-sided",
            null_abs=0.0,
        ),
    ).results[0]
    output = ContrastResults([row, available]).to_frame(backend="polars")
    assert isinstance(output, pl.DataFrame)
    assert output["lb"][0] is None and output["ub"][0] is None
    assert math.isfinite(output["lb"][1]) and math.isfinite(output["ub"][1])


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Two independent units, two cycles, two periods, one washout + two observations.
    orders = {
        "u1": {0: ("control", "treatment"), 1: ("treatment", "control")},
        "u2": {0: ("treatment", "control"), 1: ("control", "treatment")},
    }
    outcomes = {
        "u1": {0: (1.0, 10.0), 1: (20.0, 2.0)},
        "u2": {0: (4.0, 40.0), 1: (50.0, 5.0)},
    }
    for unit, cycles in orders.items():
        for cycle, arms in cycles.items():
            for period, group in enumerate(arms):
                for step in range(3):
                    row: dict[str, object] = {
                        "unit": unit,
                        "cycle": cycle,
                        "period": period,
                        "step": step,
                        "group": group,
                    }
                    value = outcomes[unit][cycle][period]
                    row["mean"] = value if step else 999.0
                    row["converted"] = bool(value > 5.0) if step else False
                    rows.append(row)
    return rows


def _source(rows=None, *, metrics=None, assignment=None, contrast_references=None):
    data = (
        rows if isinstance(rows, pl.LazyFrame) else pl.DataFrame(_rows() if rows is None else rows)
    )
    chosen_assignment = assignment or _assignment()
    return from_switchback_panel(
        data,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=metrics or {"mean": "mean", "converted": "conversion"},
        identification=_identification(),
        assignment=chosen_assignment,
        contrast_references=contrast_references,
    )


def _shared_assignment(
    *, washout: int = 1, observation: int = 1, p_ct: float = 0.5, carryover_order: int = 0
):
    return SwitchbackAssignment(
        sequence=SharedScheduleOrder(probability_ct=p_ct),
        window=SwitchbackWindow(
            washout_steps=washout,
            observation_steps=observation,
            carryover_order=carryover_order,
        ),
    )


def _shared_rows(roster, block_order, outcome_fn, *, washout: int = 1, observation: int = 1):
    """One shared, complete two-period block per declared cycle, realized
    identically for every unit in *roster*. ``block_order`` maps cycle ->
    ``("control", "treatment")`` or ``("treatment", "control")``.
    ``outcome_fn(unit, cycle, period, group)`` supplies the retained-window
    value; washout steps always carry a sentinel excluded by aggregation.
    """
    rows: list[dict[str, object]] = []
    total_steps = washout + observation
    for unit in roster:
        for cycle, order in block_order.items():
            for period, group in enumerate(order):
                for step in range(total_steps):
                    value = 999.0 if step < washout else outcome_fn(unit, cycle, period, group)
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "mean": value,
                        }
                    )
    return rows


def _shared_source(rows, *, assignment=None, metrics=None):
    chosen_assignment = assignment or _shared_assignment()
    return from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=metrics or {"mean": "mean"},
        identification=_identification(),
        assignment=chosen_assignment,
    )


@pytest.mark.parametrize("carryover_order", [0, 1, 2])
@pytest.mark.parametrize("shared", [False, True])
def test_source_stats_preserve_declared_and_retained_window_lengths(shared, carryover_order):
    rows = _shared_rows(
        ["u1", "u2"],
        {0: ("control", "treatment"), 1: ("treatment", "control")},
        lambda unit, cycle, period, group: 1.0 + (group == "treatment"),
        observation=3,
    )
    assignment = (_shared_assignment if shared else _assignment)(
        observation=3, carryover_order=carryover_order, p_ct=0.5
    )
    source = _shared_source(rows, assignment=assignment)
    stats = source.contrast_stats(source.metrics[0])
    assert stats.observation_steps == 3
    assert stats.retained_steps == 3 - carryover_order
    assert stats.identifying_assumption == "no_residual_carryover_after_discarded_steps"
    assert stats.reference_delta + stats.mean_residual == pytest.approx(3 - carryover_order)
    assert ContrastStats.model_validate_json(stats.model_dump_json()) == stats


def _shared_result(rows, *, assignment=None) -> ContrastResult:
    chosen_assignment = assignment or _shared_assignment()
    analysis = Analysis.from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=chosen_assignment,
        contrast_references=(
            {"mean": UnitCycleTApproximation()}
            if isinstance(chosen_assignment.sequence, IndependentBernoulliOrder)
            else None
        ),
    )
    results = analysis.run()
    assert isinstance(results, ContrastResults)
    return results[0]


def test_switchback_source_reports_schedule_and_integrity_diagnostics():
    source = _source()

    diagnostics = source.diagnostics
    assert diagnostics.probability_ct == pytest.approx(0.75)
    assert diagnostics.schedule_complete is True
    assert diagnostics.ct_cycles == 2
    assert diagnostics.tc_cycles == 2
    assert diagnostics.washout_steps == 1
    assert diagnostics.observation_steps == 2
    assert diagnostics.washout_rows == 8
    assert diagnostics.observation_rows == 16
    assert diagnostics.carryover_order == 0
    assert diagnostics.integrity_failures == ()


def test_switchback_mean_and_conversion_aggregate_after_washout():
    source = _source()
    mean = source.contrast_stats(next(m for m in source.context.metrics if m.name == "mean"))
    conversion = source.contrast_stats(
        next(m for m in source.context.metrics if m.name == "converted")
    )

    assert isinstance(mean, ContrastStats)
    # Mean period totals use only the two post-washout observations.
    assert mean.reference_delta + mean.mean_residual == pytest.approx(-30.0)
    # Conversion uses any over the two observation steps.
    assert conversion.reference_delta + conversion.mean_residual == pytest.approx(0.0)


def test_switchback_inverse_probability_recovers_order_contrast():
    source = _source(assignment=_assignment(p_ct=0.75))
    metric = next(m for m in source.context.metrics if m.name == "mean")
    stats = source.contrast_stats(metric)

    # Each sequence contribution is inverse-probability weighted and halved
    # because every cycle contributes one treatment and one control period.
    assert stats.reference_delta + stats.mean_residual == pytest.approx(-30.0)
    assert stats.n_units == 2
    assert stats.n_cycles == 4
    assert (stats.ct_cycles, stats.tc_cycles) == (2, 2)


def _geo_rows() -> list[dict[str, object]]:
    """A switchback randomized at geo grain, three geos over two cycles.

    Period totals are constant so the expected contrast is exact arithmetic:
    each post-washout period sums to 12 under treatment and 6 under control.
    """
    orders = {
        "geo-a": {0: ("control", "treatment"), 1: ("treatment", "control")},
        "geo-b": {0: ("control", "treatment"), 1: ("control", "treatment")},
        "geo-c": {0: ("treatment", "control"), 1: ("control", "treatment")},
    }
    rows: list[dict[str, object]] = []
    for geo, cycles in orders.items():
        for cycle, arms in cycles.items():
            for period, group in enumerate(arms):
                per_step = 6.0 if group == "treatment" else 3.0
                for step in range(3):
                    rows.append(
                        {
                            "geo": geo,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            # Step 0 is washout and must be excluded; a wild value
                            # here would corrupt the total if it ever leaked in.
                            "revenue": 999.0 if step == 0 else per_step,
                        }
                    )
    return rows


def test_switchback_randomizes_at_the_declared_unit_grain_not_only_users():
    """`unit` is the randomization grain, so a geo column is a valid unit.

    Locks the geo/market case: the unit-level reference distribution must follow
    the geo count, so degrees of freedom come from the number of geos and not the
    number of users underneath, and the asymmetric inverse-probability weights
    must be applied per geo-cycle. More users per geo can still sharpen each
    geo-level aggregate; what they cannot do is add degrees of freedom.
    """
    source = from_switchback_panel(
        pl.DataFrame(_geo_rows()),
        unit="geo",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"revenue": "mean"},
        identification=_identification(),
        assignment=_assignment(p_ct=0.75),
    )

    metric = next(m for m in source.context.metrics if m.name == "revenue")
    stats = source.contrast_stats(metric)

    # Degrees of freedom come from the geo count, not the row count.
    assert stats.n_units == 3
    assert stats.n_cycles == 6

    # Per geo-cycle the weighted contribution is (12 - 6) / (2 * 0.75) = 4.0 for a
    # control-then-treatment order and (12 - 6) / (2 * 0.25) = 12.0 for the
    # reverse. Averaged within geo: geo-a 8.0, geo-b 4.0, geo-c 8.0.
    assert stats.reference_delta + stats.mean_residual == pytest.approx(20.0 / 3.0)

    diagnostics = source.diagnostics
    assert diagnostics.n_units == 3
    assert diagnostics.ct_cycles == 4
    assert diagnostics.tc_cycles == 2
    assert diagnostics.schedule_complete is True
    assert diagnostics.integrity_failures == ()


@pytest.mark.parametrize(
    "field,values,reason",
    [
        ("cycle", [0, 4], "non_contiguous_cycle_domain"),
        ("period", [0, 2], "unexpected_domain"),
        ("step", [0, 1, 4], "unexpected_domain"),
    ],
)
def test_switchback_rejects_invalid_integer_domains(field, values, reason):
    rows = _rows()
    for index, value in enumerate(values):
        rows[index][field] = value
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.domain"
    assert raised.value.context["reason"] == reason
    assert raised.value.context["field"] == field


def test_switchback_rejects_duplicate_schedule_cells():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "duplicate_schedule_cells"


def test_switchback_incomplete_schedule_counts_missing_cells():
    missing = {("u1", 0, 0, 0), ("u2", 0, 0, 1)}
    rows = [
        row
        for row in _rows()
        if (row["unit"], row["cycle"], row["period"], row["step"]) not in missing
    ]

    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)

    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "incomplete_schedule"
    assert raised.value.context["missing_cells"] == len(missing)


def test_switchback_rejects_conflicting_arms_and_third_arm():
    rows = _rows()
    rows[0]["group"] = "treatment"
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "conflicting_arms"

    rows = _rows()
    for row in rows[:3]:
        row["group"] = "other"
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "third_arm"


def test_switchback_rejects_missing_and_nonfinite_values():
    rows = _rows()
    rows[0]["mean"] = None
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.missingness"

    rows = _rows()
    rows[0]["mean"] = float("inf")
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.numeric"


def test_switchback_rejects_parallel_assignment():
    with pytest.raises(InvalidRequestError) as raised:
        from_switchback_panel(
            pl.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=ParallelAssignment(),  # ty: ignore[invalid-argument-type] -- intentional negative test for the switchback-only contract
        )
    assert raised.value.code == "source.frame.switchback.assignment"
    assert raised.value.context["expected"] == "SwitchbackAssignment"


def test_switchback_is_eager_only():
    pl = pytest.importorskip("polars")
    with pytest.raises((TypeError, ValueError)):
        _source(pl.LazyFrame(_rows()))


def test_switchback_rejects_unsupported_metric_types():
    with pytest.raises(CapabilityError) as raised:
        _source(
            metrics=[
                MetricSpec(name="ratio", type="ratio", numerator="mean", denominator="converted")
            ]
        )
    assert raised.value.code == "source.frame.switchback.metric"
    assert raised.value.context["reason"] == "unsupported_metric_type"
    assert raised.value.context["metrics"] == ("ratio",)


def test_switchback_inverse_probability_recovers_homogeneous_treatment_effect():
    rows = _rows()
    for row in rows:
        row["mean"] = 3.0 if row["group"] == "treatment" else 0.0
    source = _source(rows, metrics={"mean": "mean"}, assignment=_assignment(p_ct=0.5))
    stats = source.contrast_stats(source.context.metrics[0])
    # Two observation steps are summed, so the effect is 2 * 3.
    assert stats.reference_delta + stats.mean_residual == pytest.approx(6.0)


def test_switchback_accepts_realized_order_imbalance_and_reports_counts():
    rows = _rows()
    for row in rows:
        row["group"] = "control" if row["period"] == 0 else "treatment"
    source = _source(rows)
    assert source.diagnostics.ct_cycles == 4
    assert source.diagnostics.tc_cycles == 0
    assert source.diagnostics.integrity_failures == ()


def test_switchback_deduplicates_shared_metric_source_columns():
    source = _source(
        metrics=[
            MetricSpec(name="first", value_column="mean"),
            MetricSpec(name="second", value_column="mean"),
        ]
    )

    assert [metric.name for metric in source.metrics] == ["first", "second"]


@pytest.mark.parametrize(
    "metric,field",
    [
        (MetricSpec(name="mean", winsorization=Winsorization(upper_value=10.0)), "winsorization"),
        (MetricSpec(name="mean", missing="zero"), "missing"),
        (MetricSpec(name="mean", missing="drop"), "missing"),
        (MetricSpec(name="mean", decision_method=Method(name="unadjusted")), "decision_method"),
        (
            MetricSpec(
                name="mean",
                sensitivity_methods=(Method(name="sensitivity"),),
            ),
            "sensitivity_methods",
        ),
        (MetricSpec(name="mean", prior=Normal(mu=0.0, sigma=1.0)), "prior"),
    ],
)
def test_switchback_rejects_unsupported_metric_policies(metric, field):
    with pytest.raises(CapabilityError) as raised:
        _source(metrics=[metric])
    assert raised.value.code == "source.frame.switchback.metric"
    assert raised.value.context["field"] == field


def test_switchback_sensitivity_only_refusal_names_sensitivity_field():
    with pytest.raises(CapabilityError) as raised:
        _source(
            metrics=[
                MetricSpec(
                    name="mean",
                    sensitivity_methods=(Method(name="sensitivity"),),
                )
            ]
        )

    assert raised.value.code == "source.frame.switchback.metric"
    assert raised.value.context["field"] == "sensitivity_methods"


@pytest.mark.parametrize(
    "case,error_type,code,expected_context",
    [
        (
            "assignment",
            InvalidRequestError,
            "source.frame.switchback.assignment",
            {"reason": "unsupported_assignment", "expected": "SwitchbackAssignment"},
        ),
        (
            "identification",
            InvalidRequestError,
            "source.frame.switchback.identification",
            {"reason": "unsupported_identification", "expected": "Randomized"},
        ),
        (
            "metric",
            CapabilityError,
            "source.frame.switchback.metric",
            {"reason": "ignored_metric_option", "field": "missing", "policy": "zero"},
        ),
        (
            "columns",
            InvalidRequestError,
            "source.frame.switchback.columns",
            {"reason": "missing_column"},
        ),
        (
            "domain",
            InvalidRequestError,
            "source.frame.switchback.domain",
            {"reason": "unexpected_domain", "column": "period"},
        ),
        (
            "schedule",
            InvalidRequestError,
            "source.frame.switchback.schedule",
            {"reason": "duplicate_schedule_cells"},
        ),
        (
            "missingness",
            InvalidRequestError,
            "source.frame.switchback.missingness",
            {"reason": "null_metric", "metric": "mean"},
        ),
        (
            "numeric",
            InvalidRequestError,
            "source.frame.switchback.numeric",
            {"reason": "nonfinite_numeric", "metric": "mean"},
        ),
        (
            "units",
            InvalidRequestError,
            "source.frame.switchback.units",
            {"reason": "insufficient_independent_units", "unit_count": 1},
        ),
    ],
)
def test_switchback_refusals_are_coded_public_contracts(case, error_type, code, expected_context):
    rows = _rows()
    kwargs = {}
    if case == "assignment":
        kwargs["assignment"] = ParallelAssignment()
    elif case == "identification":
        kwargs["identification"] = object()
    elif case == "metric":
        kwargs["metrics"] = [MetricSpec(name="mean", missing="zero")]
    elif case == "columns":
        for row in rows:
            del row["cycle"]
    elif case == "domain":
        rows[0]["period"] = 2
    elif case == "schedule":
        rows.append(dict(rows[0]))
    elif case == "missingness":
        rows[0]["mean"] = None
    elif case == "numeric":
        rows[0]["mean"] = float("inf")
    elif case == "units":
        rows = [row for row in rows if row["unit"] == "u1"]

    constructor_kwargs = {
        "unit": "unit",
        "cycle": "cycle",
        "period": "period",
        "step": "step",
        "group": "group",
        "metrics": {"mean": "mean"},
        "identification": _identification(),
        "assignment": _assignment(),
        **kwargs,
    }
    with pytest.raises(error_type) as raised:
        from_switchback_panel(pl.DataFrame(rows), **constructor_kwargs)  # ty: ignore[invalid-argument-type]

    assert type(raised.value) is error_type
    assert raised.value.code == code
    for key, value in expected_context.items():
        assert raised.value.context[key] == value


def test_switchback_rejects_unequal_windows_and_insufficient_units():
    rows = _rows()[:-1]
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "incomplete_schedule"

    rows = [row for row in _rows() if row["unit"] == "u1"]
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)
    assert raised.value.code == "source.frame.switchback.units"
    assert raised.value.context["reason"] == "insufficient_independent_units"


@pytest.mark.parametrize(
    ("field", "value", "code", "reason"),
    [
        ("unit", "", "source.frame.switchback.domain", "empty_label"),
        ("unit", 1, "source.frame.switchback.domain", "non_string_label"),
        ("group", "", "source.frame.switchback.columns", "empty_label"),
        ("group", 1, "source.frame.switchback.columns", "non_string_label"),
    ],
)
def test_switchback_rejects_empty_and_non_string_labels(field, value, code, reason):
    rows = _rows()
    for row in rows:
        row[field] = value

    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)

    assert raised.value.code == code
    assert raised.value.context["column"] == field
    assert raised.value.context["reason"] == reason
    assert raised.value.context["field"] == field


def test_switchback_rejects_mean_observation_sum_overflow_with_numeric_context():
    rows = _rows()
    for row in rows:
        if row["step"] in (1, 2):
            row["mean"] = 1e308

    with pytest.raises(InvalidRequestError) as raised:
        _source(rows, metrics={"mean": "mean"})

    assert raised.value.code == "source.frame.switchback.numeric"
    assert raised.value.context["metric"] == "mean"
    assert raised.value.context["unit"] == "u1"
    assert raised.value.context["cycle"] == 0
    assert raised.value.context["reason"] == "aggregated_outcome_overflow"


def test_switchback_preserves_representable_mean_when_cycle_sum_overflows():
    from increment.estimation.contrast import estimate_contrast
    from increment.estimation.decision_types import ContrastDecisionProcedure

    rows = [row for row in _rows() if row["step"] != 2]
    for row in rows:
        if row["step"] == 1:
            row["mean"] = 1e308 if row["group"] == "treatment" else 0.0

    source = _source(
        rows,
        metrics={"mean": "mean"},
        assignment=_assignment(washout=1, observation=1, p_ct=0.5),
    )
    procedure = ContrastDecisionProcedure(
        reference=UnitCycleTApproximation(),
        metric="mean",
        role="primary",
        alpha=0.05,
        alternative="two-sided",
        null_abs=0.0,
    )
    result = estimate_contrast(source.contrast_stats(source.metrics[0]), procedure).results[0]
    assert result.estimate.value == 1e308


@pytest.mark.parametrize("field", ["cycle", "step"])
def test_switchback_sparse_huge_integer_domains_refuse_without_domain_materialization(field):
    rows = _rows()
    rows[0][field] = 10**12

    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)

    assert raised.value.code == "source.frame.switchback.domain"
    assert raised.value.context["column"] == field
    assert raised.value.context["reason"] in {
        "non_contiguous_cycle_domain",
        "unexpected_domain",
    }


def test_switchback_huge_window_declaration_uses_coded_domain_refusal():
    huge = 10**30

    with pytest.raises(InvalidRequestError) as raised:
        _source(
            _rows()[:-1],
            assignment=_assignment(washout=huge, observation=huge),
        )

    assert raised.value.code == "source.frame.switchback.domain"
    assert raised.value.context["column"] == "step"
    assert raised.value.context["reason"] == "unexpected_domain"
    assert raised.value.context["expected"] == repr(range(2 * huge))


def test_switchback_small_step_domain_refusal_preserves_structured_expected_context():
    rows = [row for row in _rows() if row["step"] != 2]

    with pytest.raises(InvalidRequestError) as raised:
        _source(rows)

    assert raised.value.code == "source.frame.switchback.domain"
    assert raised.value.context["column"] == "step"
    assert raised.value.context["expected"] == (0, 1, 2)


def test_switchback_rejects_finite_centered_reduction_overflow_with_numeric_context():
    rows = [row for row in _rows() if row["step"] != 2]
    for row in rows:
        if row["step"] != 1:
            continue
        if row["unit"] == "u1":
            is_high = row["cycle"] == 0 and row["period"] == 1
            row["mean"] = 1e308 if is_high else 0.0
        else:
            row["mean"] = 0.0

    with pytest.raises(InvalidRequestError) as raised:
        _source(
            rows,
            metrics={"mean": "mean"},
            assignment=_assignment(washout=1, observation=1, p_ct=0.5),
        )

    assert raised.value.code == "source.frame.switchback.numeric"
    assert raised.value.context["metric"] == "mean"
    assert "unit" not in raised.value.context
    assert raised.value.context["reason"] == "centered_contrast_reduction_overflow"


# from_switchback_panel: a realized CT/TC split wildly incompatible with the
# declared probability_ct is evidence the declared mechanism is wrong.


def _all_ct_rows(n_units=20, n_cycles=4):
    rows = []
    for unit in range(n_units):
        for cycle in range(n_cycles):
            for period, group in enumerate(("control", "treatment")):
                for step in range(3):
                    rows.append(
                        {
                            "unit": f"u{unit}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "y": 1.0 + (2.0 if group == "treatment" else 0.0),
                        }
                    )
    return rows


def test_realized_order_split_incompatible_with_declared_probability_refuses_before_construction():
    """probability_ct scales every inverse-probability contribution, so a
    declared 0.5 against 80 CT / 0 TC realized cycles is construction-time
    evidence the declared mechanism is wrong: it must refuse before any
    result is built, not surface a soft diagnostic after the fact. This is
    a smaller degenerate all-CT fixture; test_switchback_smoke.py reproduces
    the plan's 320-CT split through Analysis."""
    with pytest.raises(InvalidRequestError) as raised:
        from_switchback_panel(
            pl.DataFrame(_all_ct_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"y": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=2),
            ),
        )
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "implausible_realized_split"
    assert raised.value.context["ct_cycles"] == 80
    assert raised.value.context["tc_cycles"] == 0
    assert raised.value.context["probability_ct"] == 0.5


def test_correctly_declared_shared_schedule_refuses_the_same_degenerate_witness():
    """An extended all-CT degenerate split refuses under a declared shared
    schedule too: never fabricate positive width for it, and never infer
    the true law from data that merely looks compatible. The shared law's
    integrity check uses block-level draw counts, not unit-cycle-inflated
    ones, so this witness needs enough blocks (not units) to be extreme."""
    with pytest.raises(InvalidRequestError) as raised:
        from_switchback_panel(
            pl.DataFrame(_all_ct_rows(n_units=5, n_cycles=24)),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"y": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=SwitchbackAssignment(
                sequence=SharedScheduleOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=2),
            ),
        )
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "implausible_realized_split"
    # Every unit in _all_ct_rows() shares the same CT order per cycle, so
    # the block-level (not unit-cycle-inflated) counts are 24 CT blocks / 0
    # TC -- distinguishing block draws from unit-cycle draws is exactly
    # what the shared law's diagnostics must do.
    assert raised.value.context["ct_cycles"] == 24
    assert raised.value.context["tc_cycles"] == 0


def test_correctly_declared_shared_schedule_estimates_matched_per_unit_effect_with_block_counts():
    """Shared inference uses block counts and the mean-unit effect."""
    roster = ["u0", "u1", "u2"]
    unit_effect = {"u0": 4.0, "u1": 6.0, "u2": 8.0}  # mean 6.0
    block_order = {
        0: ("control", "treatment"),
        1: ("control", "treatment"),
        2: ("treatment", "control"),
        3: ("treatment", "control"),
    }
    eps = {0: -1.0, 1: 1.0, 2: -2.0, 3: 2.0}  # sums to zero; leaves theta exactly 6.0

    def outcome_fn(unit, cycle, period, group):
        is_treatment = group == "treatment"
        return 10.0 + (unit_effect[unit] + eps[cycle]) * is_treatment

    rows = _shared_rows(roster, block_order, outcome_fn)
    source = _shared_source(rows, assignment=_shared_assignment(p_ct=0.5))

    diagnostics = source.diagnostics
    assert diagnostics.n_units == 3
    assert diagnostics.n_blocks == 4
    assert (diagnostics.ct_cycles, diagnostics.tc_cycles) == (2, 2)
    assert diagnostics.integrity_failures == ()

    metric = next(m for m in source.context.metrics if m.name == "mean")
    stats = source.contrast_stats(metric)
    assert isinstance(stats, ContrastStats)
    assert stats.randomization_law == "shared_schedule"
    assert stats.independence_grain == "shared_block"
    assert stats.exact_delta_total is None
    assert stats.n_units == 3
    assert stats.n_blocks == 4
    assert stats.n_cycles == 12
    assert (stats.ct_cycles, stats.tc_cycles) == (2, 2)
    # X_b per block: [5.0, 7.0, 4.0, 8.0] (mean-unit effect 6.0 plus eps/1);
    # theta = mean_b(X_b) recovers the matched per-unit effect exactly.
    theta = stats.reference_delta + stats.mean_residual
    assert theta == pytest.approx(6.0)
    assert stats.m2_delta == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("periods", "p_ct", "expected"),
    [
        ([(1e15, 1e15 + 0.125), (1e15, 1e15), (1e15, 1e15)], 0.5, 0.125 / 3),
        ([(1e308, 1e308)] * 3, 0.5, 0.0),
        ([(1.0, 1e20), (0.0, -1e20)], 0.5, -0.5),
        ([(-1e308, 1e308), (1e308, -1e308)], 0.5, 0.0),
        ([(0.0, 1e308)] * 2, 0.75, 1e308 / 1.5),
    ],
    ids=[
        "neighboring-values",
        "large-common-offset",
        "cancellation",
        "opposite-extremes",
        "scaled-overflow",
    ],
)
def test_shared_roster_contrast_preserves_representable_effects(periods, p_ct, expected):
    roster = [f"u{i}" for i in range(len(periods))]
    values = dict(zip(roster, periods, strict=True))
    # Cycles 0-1 use CT; cycle 2 realizes TC for positivity. Reading it through
    # the swapped period index (scaled by weight_ct/weight_tc = 1/3 at p_ct=0.75
    # to stay representable) makes every cycle contribute identically, so the
    # expected value and m2_delta == 0 match an all-CT schedule.
    scale = 1.0 / 3.0 if p_ct == 0.75 else 1.0

    def outcome_fn(unit, cycle, period, group):
        if cycle in (0, 1):
            return values[unit][period]
        return values[unit][1 - period] * scale

    rows = _shared_rows(
        roster,
        {0: ("control", "treatment"), 1: ("control", "treatment"), 2: ("treatment", "control")},
        outcome_fn,
    )
    source = _shared_source(rows, assignment=_shared_assignment(p_ct=p_ct))
    stats = source.contrast_stats(source.context.metrics[0])
    assert (stats.ct_cycles, stats.tc_cycles) == (2, 1)
    assert stats.m2_delta == pytest.approx(0.0, abs=0.0)
    assert math.fsum((stats.reference_delta, stats.mean_residual)) == pytest.approx(
        expected, rel=2e-15, abs=0
    )


def test_shared_inference_uses_blocks_even_with_one_roster_member():
    rows = _shared_rows(
        ["u0"],
        {0: ("control", "treatment"), 1: ("control", "treatment"), 2: ("treatment", "control")},
        lambda unit, cycle, period, group: 10.0 + (4.0 + 2.0 * cycle) * (group == "treatment"),
    )
    result = _shared_result(rows)
    assert (result.n_units, result.n_blocks, result.dof) == (1, 3, 2)
    assert result.estimate.value == pytest.approx(6.0)
    assert result.standard_error == pytest.approx(math.sqrt(4.0 / 3))


def test_shared_schedule_refuses_a_plausible_but_degenerate_split_with_a_route_forward():
    """6/6 blocks all-CT at probability_ct=0.8 has binomial p=0.26 --
    nowhere near the 1e-6 mechanism threshold -- so only the positivity
    check catches it. Distinct from the existing all-CT witnesses in this
    file (which are extreme enough to already trip the mechanism test)."""
    block_order = dict.fromkeys(range(6), ("control", "treatment"))
    with pytest.raises(InvalidRequestError) as raised:
        from_switchback_panel(
            pl.DataFrame(
                _shared_rows(
                    ["u0", "u1", "u2"],
                    block_order,
                    lambda unit, cycle, period, group: 1.0 + (group == "treatment"),
                )
            ),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=SwitchbackAssignment(
                sequence=SharedScheduleOrder(probability_ct=0.8),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
        )
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "shared_schedule_missing_cycle_order"
    assert (raised.value.context["ct_cycles"], raised.value.context["tc_cycles"]) == (6, 0)
    assert raised.value.context["n_blocks"] == 6
    assert raised.value.context["probability_ct"] == 0.8
    assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]


def test_shared_schedule_whole_roster_cloning_preserves_estimate_se_and_evidence():
    """Cloning the roster cannot manufacture independent evidence."""
    roster = ["u0", "u1", "u2"]
    unit_effect = {"u0": 4.0, "u1": 6.0, "u2": 8.0}
    block_order = {
        0: ("control", "treatment"),
        1: ("control", "treatment"),
        2: ("treatment", "control"),
        3: ("treatment", "control"),
    }
    eps = {0: -1.0, 1: 1.0, 2: -2.0, 3: 2.0}

    def outcome_fn(unit, cycle, period, group):
        is_treatment = group == "treatment"
        base_unit = unit.removesuffix("-clone")
        return 10.0 + (unit_effect[base_unit] + eps[cycle]) * is_treatment

    original_rows = _shared_rows(roster, block_order, outcome_fn)
    cloned_roster = roster + [f"{unit}-clone" for unit in roster]
    cloned_rows = _shared_rows(cloned_roster, block_order, outcome_fn)

    original = _shared_result(original_rows)
    cloned = _shared_result(cloned_rows)
    assert cloned.n_units == 2 * original.n_units
    assert cloned.n_blocks == original.n_blocks == 4
    assert cloned.ct_cycles == original.ct_cycles == 2
    assert cloned.tc_cycles == original.tc_cycles == 2
    assert cloned.dof == original.dof == 3
    assert cloned.estimate.value == pytest.approx(original.estimate.value)
    assert cloned.standard_error == pytest.approx(original.standard_error)
    assert original.standard_error == pytest.approx(math.sqrt(10.0 / 12))
    assert cloned.estimate.lb == pytest.approx(original.estimate.lb)
    assert cloned.estimate.ub == pytest.approx(original.estimate.ub)
    assert cloned.standard_error is not None and original.standard_error is not None
    assert cloned.estimate.value / cloned.standard_error == pytest.approx(
        original.estimate.value / original.standard_error
    )


def test_shared_schedule_requires_identical_realized_order_across_roster():
    """Every member must share its block's realized order."""
    roster = ["u0", "u1"]
    block_order = {0: ("control", "treatment"), 1: ("control", "treatment")}

    def outcome_fn(unit, cycle, period, group):
        return 1.0 if group == "treatment" else 0.0

    rows = _shared_rows(roster, block_order, outcome_fn)
    for row in rows:
        if row["unit"] == "u1" and row["cycle"] == 1:
            row["group"] = "control" if row["group"] == "treatment" else "treatment"

    with pytest.raises(InvalidRequestError) as raised:
        _shared_source(rows)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "shared_block_order_mismatch"
    assert raised.value.context["cycle"] == 1


def test_switchback_missing_discarded_carryover_step_still_refuses_incomplete_schedule():
    """A row missing only from the discarded carryover region still fails
    completeness: retained-window discarding never excuses an absent
    underlying observation."""
    missing = ("u1", 0, 1, 1)
    rows = [
        row for row in _rows() if (row["unit"], row["cycle"], row["period"], row["step"]) != missing
    ]
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows, assignment=_assignment(carryover_order=1))
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "incomplete_schedule"


@pytest.mark.parametrize("shared", [False, True])
def test_switchback_unit_missing_a_whole_cycle_refuses_as_incomplete(shared):
    """Retained blocks reshape uniformly only because every unit must hold
    every cycle; a roster ragged by a whole cycle refuses at the schedule."""
    if shared:
        rows = _shared_rows(
            ["u0", "u1"],
            {0: ("control", "treatment"), 1: ("treatment", "control")},
            lambda unit, cycle, period, group: 1.0,
        )
        build = _shared_source
    else:
        rows, build = _rows(), _source
    ragged = [row for row in rows if not (row["unit"] == "u1" and row["cycle"] == 1)]
    with pytest.raises(InvalidRequestError) as raised:
        build(ragged)
    assert raised.value.code == "source.frame.switchback.schedule"
    assert raised.value.context["reason"] == "incomplete_schedule"


def test_switchback_carryover_order_retains_declared_window_only():
    """carryover_order=1 discards one additional post-washout step. Every
    retained value in ``_rows()`` repeats identically across steps 1 and 2,
    so retaining only step 2 exactly halves the aggregated contrast."""
    baseline = _source()
    baseline_mean = baseline.contrast_stats(
        next(m for m in baseline.context.metrics if m.name == "mean")
    )
    source = _source(assignment=_assignment(carryover_order=1))
    mean = source.contrast_stats(next(m for m in source.context.metrics if m.name == "mean"))
    assert mean.reference_delta + mean.mean_residual == pytest.approx(
        (baseline_mean.reference_delta + baseline_mean.mean_residual) / 2.0
    )
    assert source.diagnostics.carryover_order == 1
    assert source.diagnostics.observation_steps == 2
    assert source.diagnostics.retained_steps == 1


def test_switchback_carryover_order_excludes_exactly_the_declared_discarded_step():
    """A dedicated fixture with distinct per-step values pins the retention
    boundary precisely: carryover_order=1 must drop step 1, not any step."""
    rows: list[dict[str, object]] = []
    orders = {"u1": ("control", "treatment"), "u2": ("treatment", "control")}
    step_values = {1: 1.0, 2: 10.0}
    for unit, order in orders.items():
        for period, group in enumerate(order):
            for step in range(3):
                value = (
                    999.0
                    if step == 0
                    else step_values[step] * (2.0 if group == "treatment" else 1.0)
                )
                rows.append(
                    {
                        "unit": unit,
                        "cycle": 0,
                        "period": period,
                        "step": step,
                        "group": group,
                        "mean": value,
                    }
                )
    order0 = _source(
        rows, metrics={"mean": "mean"}, assignment=_assignment(p_ct=0.5, carryover_order=0)
    )
    order1 = _source(
        rows, metrics={"mean": "mean"}, assignment=_assignment(p_ct=0.5, carryover_order=1)
    )
    mean0 = order0.contrast_stats(next(m for m in order0.context.metrics if m.name == "mean"))
    mean1 = order1.contrast_stats(next(m for m in order1.context.metrics if m.name == "mean"))
    theta0 = mean0.reference_delta + mean0.mean_residual
    theta1 = mean1.reference_delta + mean1.mean_residual
    # Retaining both steps: (2+20)-(1+10)=11 per unit-cycle, averaged over
    # both units with p=0.5 halving each -> theta0=11.0. Retaining only
    # step 2 (>=washout(1)+carryover(1)=2): (20-10)=10 -> theta1=10.0.
    assert theta0 == pytest.approx(11.0)
    assert theta1 == pytest.approx(10.0)


@pytest.mark.parametrize("sequence_type", [IndependentBernoulliOrder, SharedScheduleOrder])
@pytest.mark.parametrize("true_order", [0, 1, 2])
@pytest.mark.parametrize("declared_order", [0, 1, 2])
def test_finite_history_matrix_recovers_only_bounded_targets(
    sequence_type, true_order, declared_order
):
    rows = []
    tau, amplitude, observation = 2.0, 4.0, 3
    for unit in ("u0", "u1"):
        previous_treatment = False
        for cycle, order in enumerate((("treatment", "control"), ("control", "treatment"))):
            for period, group in enumerate(order):
                for step in range(observation):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "mean": 100.0
                            + tau * (group == "treatment")
                            + amplitude * previous_treatment * (step < true_order),
                        }
                    )
                previous_treatment = group == "treatment"
    source = _shared_source(
        rows,
        assignment=SwitchbackAssignment(
            sequence=sequence_type(probability_ct=0.5),
            window=SwitchbackWindow(
                washout_steps=0, observation_steps=observation, carryover_order=declared_order
            ),
        ),
    )
    stats = source.contrast_stats(source.context.metrics[0])
    target = tau * (observation - declared_order)
    bias = -(amplitude / 2) * max(0, true_order - declared_order)
    estimate = math.fsum((stats.reference_delta, stats.mean_residual))
    assert estimate == pytest.approx(target + bias)
    if declared_order >= true_order:
        assert estimate == pytest.approx(target)
    else:
        assert estimate < target


@pytest.mark.parametrize("p_ct", [0.5, 0.75, 0.9])
def test_shared_schedule_exact_ht_expectation_and_variance(p_ct):
    """Enumerate CT/TC: E[X]=(D_CT+D_TC)/2; Var[X]=p(1-p)(X_CT-X_TC)^2.

    Each enumerated source needs a second, opposite-order block to satisfy
    positivity; that block's values are chosen so its own inverse-probability
    contribution exactly reproduces the single-order value being enumerated
    (X_1 == X_0), leaving observed_ct/observed_tc numerically identical to
    the pre-fix single-block computation.
    """
    d_ct = 12.0
    d_tc = 4.0
    x_ct = d_ct / (2.0 * p_ct)
    x_tc = d_tc / (2.0 * (1.0 - p_ct))
    # Compensating-block values so the added opposite-order block's own
    # contribution equals x_ct (resp. x_tc) exactly.
    ct_compensator = d_ct * (1.0 - p_ct) / p_ct
    tc_compensator = d_tc * p_ct / (1.0 - p_ct)

    def outcome_fn_ct(unit, cycle, period, group):
        # period0=control -> 0.0, period1=treatment -> d_ct (M2-M1=d_ct).
        if cycle == 0:
            return d_ct if group == "treatment" else 0.0
        return ct_compensator if group == "treatment" else 0.0

    def outcome_fn_tc(unit, cycle, period, group):
        # period0=treatment -> d_tc, period1=control -> 0.0 (M1-M2=d_tc).
        if cycle == 0:
            return d_tc if group == "treatment" else 0.0
        return tc_compensator if group == "treatment" else 0.0

    roster = ["u0", "u1"]
    ct_rows = _shared_rows(
        roster, {0: ("control", "treatment"), 1: ("treatment", "control")}, outcome_fn_ct
    )
    tc_rows = _shared_rows(
        roster, {0: ("treatment", "control"), 1: ("control", "treatment")}, outcome_fn_tc
    )

    ct_source = _shared_source(ct_rows, assignment=_shared_assignment(p_ct=p_ct))
    tc_source = _shared_source(tc_rows, assignment=_shared_assignment(p_ct=p_ct))
    ct_stats = ct_source.contrast_stats(
        next(m for m in ct_source.context.metrics if m.name == "mean")
    )
    tc_stats = tc_source.contrast_stats(
        next(m for m in tc_source.context.metrics if m.name == "mean")
    )

    observed_ct = math.fsum((ct_stats.reference_delta, ct_stats.mean_residual))
    observed_tc = math.fsum((tc_stats.reference_delta, tc_stats.mean_residual))
    assert (observed_ct, observed_tc) == pytest.approx((x_ct, x_tc))
    expectation = p_ct * observed_ct + (1.0 - p_ct) * observed_tc
    variance = (
        p_ct * (observed_ct - expectation) ** 2 + (1.0 - p_ct) * (observed_tc - expectation) ** 2
    )
    assert expectation == pytest.approx((d_ct + d_tc) / 2)
    assert variance == pytest.approx(p_ct * (1.0 - p_ct) * (x_ct - x_tc) ** 2)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_order_counts_reach_public_surfaces_independent_of_row_order(shared, backend):
    import narwhals as nw

    from increment.tables import estimates_to_readout

    pytest.importorskip(backend)
    rows = _shared_rows(
        ["u0", "u1", "u2"],
        {
            0: ("control", "treatment"),
            1: ("control", "treatment"),
            2: ("treatment", "control"),
            3: ("control", "treatment"),
        },
        lambda unit, cycle, period, group: 10.0 + (4.0 + cycle) * (group == "treatment"),
    )
    assignment = (_shared_assignment if shared else _assignment)(observation=1, p_ct=0.5)
    expected = (3, 1) if shared else (9, 3)
    for ordered_rows in (rows, list(reversed(rows))):
        source = _shared_source(ordered_rows, assignment=assignment)
        stats = source.contrast_stats(source.metrics[0])
        result = _shared_result(ordered_rows, assignment=assignment)
        restored_stats = ContrastStats.model_validate_json(stats.model_dump_json())
        restored_result = ContrastResult.model_validate_json(result.model_dump_json())
        for evidence in (source.diagnostics, stats, result, restored_stats, restored_result):
            assert (evidence.ct_cycles, evidence.tc_cycles) == expected
        frame = nw.from_native(ContrastResults([result]).to_frame(backend=backend), eager_only=True)
        row = estimates_to_readout([result])[0]
        assert (frame["ct_cycles"][0], frame["tc_cycles"][0]) == expected
        assert (row["ct_cycles"], row["tc_cycles"]) == expected


@pytest.mark.parametrize("shared", [False, True])
def test_assignment_diagnostic_refuses_inconsistent_order_count_sum(shared):
    rows = _shared_rows(
        ["u0", "u1"],
        {0: ("control", "treatment"), 1: ("treatment", "control")},
        lambda unit, cycle, period, group: 1.0 + (group == "treatment"),
    )
    assignment = (_shared_assignment if shared else _assignment)(observation=1, p_ct=0.5)
    diagnostic = _shared_source(rows, assignment=assignment).diagnostics
    values = diagnostic.model_dump()
    values["ct_cycles"] += 1
    with pytest.raises(InvalidRequestError) as raised:
        type(diagnostic).model_validate(values)
    assert raised.value.code == "estimation.contrast.order_counts"


@pytest.mark.parametrize("offset", [2**53, 2**54])
def test_neighboring_integer_responses_preserve_exact_totals_across_order_and_partitions(offset):
    from increment.estimation.contrast import (
        ContrastPartition,
        estimate_contrast,
        reduce_contrast_partitions,
    )
    from tests.estimation.test_unit_cycle_envelope import procedure

    rows = [
        {
            "unit": unit,
            "cycle": 0,
            "period": period,
            "step": 0,
            "group": "control" if (period == 0) == ct else "treatment",
            "mean": offset + (0 if (period == 0) == ct else 1),
        }
        for unit, ct in (("a", True), ("b", False))
        for period in range(2)
    ]
    expected_total = Fraction(8, 3)
    parts = []
    for selected in (rows, rows[::-1], rows[::2] + rows[1::2], rows[:2], rows[2:]):
        source, reference = _exact_envelope_source(selected)
        stats = source.contrast_stats(source.metrics[0])
        if stats.n_units == 2:
            assert stats.exact_delta_total == expected_total.as_integer_ratio()
            result = estimate_contrast(
                stats, procedure(reference, metric="mean", null_abs=1)
            ).results[0]
            assert result.estimate.value == float(expected_total / 2)
            assert result.residual_p_value == 1
        else:
            key = str(selected[0]["unit"])
            fields = {
                name: getattr(stats, name)
                for name in (
                    "metric",
                    "aggregation",
                    "probability_ct",
                    "randomization_law",
                    "independence_grain",
                    "washout_steps",
                    "carryover_order",
                    "observation_steps",
                    "retained_steps",
                    "control_group",
                    "treatment_group",
                )
            }
            parts.append(
                ContrastPartition(
                    **fields,
                    unit_deltas={key: stats.reference_delta + stats.mean_residual},
                    exact_unit_deltas={key: stats.exact_delta_total},
                    cycles_by_unit={key: 1},
                    unit_slopes={key: stats.mean_slope},
                    ct_counts_by_unit={key: stats.ct_cycles},
                )
            )
    for ordered in (parts, parts[::-1]):
        restored = [ContrastPartition.model_validate_json(p.model_dump_json()) for p in ordered]
        assert (
            reduce_contrast_partitions(restored).exact_delta_total
            == expected_total.as_integer_ratio()
        )


@pytest.mark.parametrize("offset", [0, 2**53])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("p_ct", [0.5, 0.75])
def test_neighboring_integer_period_levels_preserve_float_contrasts(offset, shared, p_ct):
    from scipy.stats import t

    rows = _shared_rows(
        ["a", "b"],
        {0: ("control", "treatment"), 1: ("treatment", "control")},
        lambda unit, cycle, period, group: offset + (group == "treatment"),
        washout=0,
        observation=1,
    )
    assignment = (_shared_assignment if shared else _assignment)(
        washout=0, observation=1, p_ct=p_ct
    )
    expected = 1 / (4 * p_ct) + 1 / (4 * (1 - p_ct))
    expected_se = abs(1 / (4 * p_ct) - 1 / (4 * (1 - p_ct))) if shared else 0
    for ordered in (rows, rows[::-1]):
        result = _shared_result(ordered, assignment=assignment)
        assert result.estimate.value == pytest.approx(expected)
        assert result.standard_error == pytest.approx(expected_se)
        if expected_se:
            radius = t.isf(result.alpha / 2, 1) * expected_se
            assert result.estimate.lb == pytest.approx(expected - radius)
            assert result.estimate.ub == pytest.approx(expected + radius)
        else:
            assert result.estimate.lb is None and result.estimate.ub is None


def test_switchback_missing_impute_refuses_with_the_shared_metric_code():
    with pytest.raises(InvalidRequestError) as raised:
        _source(metrics=[{"name": "mean", "missing": "impute"}])
    assert raised.value.code == "frame.metric.missing_impute"
    assert raised.value.context["metric"] == "mean"
