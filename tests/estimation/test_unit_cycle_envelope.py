"""Boundary regressions for prospective residual-envelope runtime inference."""

import json
import math
from fractions import Fraction

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.contrast import ContrastStats, estimate_contrast
from increment.estimation.contrast_results import ContrastResult, ContrastResults
from increment.estimation.decision_types import ContrastDecisionProcedure, PValueEvidence
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.unit_cycle import (
    ProspectiveAssumptionProvenance,
    UnitCycleTApproximation,
    UnitCycleVarianceEnvelope,
)
from increment.tables import contrast_results_to_readout


def envelope(*, p=0.8, variance=0.01, cycles=1, metric="value"):
    return UnitCycleVarianceEnvelope(
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=p),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        metric=metric,
        control_group="control",
        treatment_group="treatment",
        response_meaning="retained_total",
        cycles_per_unit=cycles,
        residual_variance_upper=variance,
        provenance=ProspectiveAssumptionProvenance(
            assumption_id="bounded-residual",
            assumption_version="1",
            justification="Prospective test population residual variance bound.",
            declaration_id="test-declaration",
        ),
    )


def stats(**changes):
    values = {
        "metric": "value",
        "aggregation": "sum",
        "probability_ct": 0.8,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 1,
        "retained_steps": 1,
        "control_group": "control",
        "treatment_group": "treatment",
        "n_units": 4,
        "n_cycles": 4,
        "reference_delta": 1.25,
        "mean_residual": 0.0,
        "m2_delta": 0.0,
        "mean_slope": 0.625,
        "ct_cycles": 4,
        "tc_cycles": 0,
        "minimum_cycles_per_unit": 1,
        "maximum_cycles_per_unit": 1,
    }
    values.update(changes)
    if "exact_delta_total" not in changes and values["ct_cycles"] is not None:
        p = Fraction(values["probability_ct"])
        slope = (values["ct_cycles"] / (2 * p) + values["tc_cycles"] / (2 * (1 - p))) / values[
            "n_cycles"
        ]
        # Default synthetic observations have a retained-total effect of exactly two.
        exact_point = (
            2 * slope if "reference_delta" not in changes else Fraction(values["reference_delta"])
        )
        total = exact_point * values["n_units"]
        values["exact_delta_total"] = (total.numerator, total.denominator)
    return ContrastStats.model_validate(values)


def procedure(reference, **changes):
    values = {
        "metric": "value",
        "role": "primary",
        "alternative": "two-sided",
        "null_abs": 0,
        "alpha": 0.05,
        "reference": reference,
    }
    values.update(changes)
    return ContrastDecisionProcedure.model_validate(values)


def test_observed_slope_interval_does_not_hull_the_ht_point():
    computation = estimate_contrast(stats(), procedure(envelope()))
    result = computation.results[0]
    assert result.estimate.value == 1.25
    assert result.estimate.lb is not None and result.estimate.ub is not None
    assert result.estimate.lb > result.estimate.value
    assert result.estimate.lb == pytest.approx((1.25 - math.sqrt(0.01 / 4 / 0.05)) / 0.625)
    assert result.estimate.ub == pytest.approx((1.25 + math.sqrt(0.01 / 4 / 0.05)) / 0.625)
    assert result.standard_error == 0
    assert result.dof is None and result.dof_unavailable_reason == "not_applicable"
    assert result.method == "switchback_unit_variance_envelope"
    assert result.reference == "residual_chebyshev"
    evidence = next(iter(computation.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == pytest.approx((0.01 / 4) / 1.25**2)
    assert not computation.failures


@pytest.mark.parametrize("alternative,open_side", [("greater", "upper"), ("less", "lower")])
def test_cantelli_direction_and_open_side(alternative, open_side):
    sign = -1 if alternative == "less" else 1
    computation = estimate_contrast(
        stats(reference_delta=sign * 1.25), procedure(envelope(), alternative=alternative)
    )
    result = computation.results[0]
    cutoff = math.sqrt((0.01 / 4) * (1 - 0.05) / 0.05)
    assert result.estimate.open_side == open_side
    assert getattr(result.estimate, "ub" if alternative == "greater" else "lb") is None
    endpoint = result.estimate.lb if alternative == "greater" else result.estimate.ub
    assert endpoint == pytest.approx(sign * (1.25 - cutoff) / 0.625)
    assert result.residual_p_value == pytest.approx((0.01 / 4) / (0.01 / 4 + 1.25**2))
    assert result.reference == "residual_cantelli"
    assert contrast_results_to_readout([result])[0]["stat_sig"] is True


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_null_is_subtracted_from_slope_not_ht_point(alternative):
    result = estimate_contrast(stats(), procedure(envelope(), null_abs=2, alternative=alternative))
    evidence = next(iter(result.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == 1
    assert contrast_results_to_readout(result.results)[0]["stat_sig"] is False


def test_zero_envelope_identifies_singleton_with_nonzero_sample_variance():
    result = estimate_contrast(
        stats(m2_delta=3), procedure(envelope(variance=0), null_abs=2)
    ).results[0]
    assert result.estimate.value == 1.25
    assert result.estimate.lb == result.estimate.ub == 2
    assert result.standard_error == 0.5
    assert result.residual_p_value == 1


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("variance", [0.0, math.ulp(0.0)])
def test_smallest_alpha_preserves_finite_envelope_evidence_and_wire(alternative, variance):
    alpha = math.ulp(0.0)
    sign = -1 if alternative == "less" else 1
    computation = estimate_contrast(
        stats(
            reference_delta=sign,
            probability_ct=0.5,
            ct_cycles=2,
            tc_cycles=2,
            mean_slope=1,
        ),
        procedure(envelope(p=0.5, variance=variance), alpha=alpha, alternative=alternative),
    )
    result = computation.results[0]
    evidence = next(iter(computation.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == (0.0 if variance == 0 else alpha)
    assert result.estimate.alpha == alpha
    assert result.estimate.level == 1.0
    for endpoint in (result.estimate.lb, result.estimate.ub):
        if endpoint is not None:
            assert math.isfinite(endpoint)
    assert ContrastResult.model_validate_json(result.model_dump_json()) == result
    assert contrast_results_to_readout([result])[0]["stat_sig"] is (variance == 0)


def test_nonrepresentable_singleton_gets_smallest_outward_float_enclosure():
    result = estimate_contrast(
        stats(reference_delta=1, probability_ct=0.75, ct_cycles=1, tc_cycles=3, mean_slope=5 / 3),
        procedure(envelope(p=0.75, variance=0)),
    ).results[0]
    assert result.estimate.lb is not None and result.estimate.ub is not None
    exact = Fraction(3, 5)
    assert Fraction(result.estimate.lb) <= exact <= Fraction(result.estimate.ub)
    assert math.nextafter(result.estimate.lb, math.inf) == result.estimate.ub
    assert result.estimate.value > result.estimate.ub


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_closed_cutoff_ties_are_not_rejections(alternative):
    alpha = 0.25 if alternative == "two-sided" else 0.5
    sign = -1 if alternative == "less" else 1
    declared = envelope(p=0.5, variance=4 if alternative != "two-sided" else 1)
    values = stats(probability_ct=0.5, mean_slope=1, reference_delta=sign * 1.0)
    result = estimate_contrast(values, procedure(declared, alpha=alpha, alternative=alternative))
    evidence = next(iter(result.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value >= alpha
    assert contrast_results_to_readout(result.results)[0]["stat_sig"] is False
    next_point = sign * math.nextafter(1, math.inf)
    next_total = Fraction(next_point) * values.n_units
    beyond = values.model_copy(
        update={
            "reference_delta": next_point,
            "exact_delta_total": (next_total.numerator, next_total.denominator),
        }
    )
    other = estimate_contrast(beyond, procedure(declared, alpha=alpha, alternative=alternative))
    evidence = next(iter(other.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value < alpha
    assert contrast_results_to_readout(other.results)[0]["stat_sig"] is True


def test_envelope_required_is_not_an_implicit_t_fallback():
    with pytest.raises(InvalidRequestError) as caught:
        estimate_contrast(stats(), procedure(None))
    assert caught.value.code == "unit_cycle.envelope_required"
    historical = estimate_contrast(stats(m2_delta=3), procedure(UnitCycleTApproximation()))
    assert historical.results[0].method == "switchback_unit_t_approximation"
    assert historical.results[0].reference == "unit_t_approximation"


@pytest.mark.parametrize(
    "changes",
    [
        {"aggregation": "any"},
        {"washout_steps": 1},
        {
            "minimum_cycles_per_unit": 1,
            "maximum_cycles_per_unit": 3,
            "n_cycles": 8,
            "ct_cycles": 4,
            "tc_cycles": 4,
        },
    ],
)
def test_incompatible_sufficient_state_refuses(changes):
    declared = envelope(cycles=2) if changes.get("n_cycles") == 8 else envelope()
    with pytest.raises(InvalidRequestError) as caught:
        estimate_contrast(stats(**changes), procedure(declared))
    assert caught.value.code == "unit_cycle.reference_mismatch"


def test_replayed_implausible_split_cannot_reset_the_budget():
    with pytest.raises(InvalidRequestError) as caught:
        estimate_contrast(
            stats(
                n_units=100,
                n_cycles=100,
                probability_ct=0.5,
                mean_slope=1,
                ct_cycles=100,
                tc_cycles=0,
            ),
            procedure(envelope(p=0.5)),
        )
    assert caught.value.code == "unit_cycle.implausible_realized_split"


def test_historical_stats_without_order_metadata_cannot_claim_envelope_inference():
    historical = stats(
        mean_slope=None,
        ct_cycles=None,
        tc_cycles=None,
        minimum_cycles_per_unit=None,
        maximum_cycles_per_unit=None,
    )
    with pytest.raises(InvalidRequestError) as caught:
        estimate_contrast(historical, procedure(envelope()))
    assert caught.value.code == "unit_cycle.sufficient_state"


def test_single_unit_envelope_has_inference_but_no_sample_se():
    result = estimate_contrast(
        stats(n_units=1, n_cycles=1, ct_cycles=1), procedure(envelope())
    ).results[0]
    assert result.standard_error is None
    assert result.standard_error_unavailable_reason == "insufficient_replicates"
    assert result.estimate.lb is not None


def test_float_only_stats_cannot_supply_exact_envelope_evidence():
    historical = stats(exact_delta_total=None, m2_delta=3)
    with pytest.raises(InvalidRequestError) as caught:
        estimate_contrast(historical, procedure(envelope()))
    assert caught.value.code == "unit_cycle.sufficient_state"
    assert estimate_contrast(historical, procedure(UnitCycleTApproximation())).results


def test_neighboring_float_zero_envelope_keeps_evidence_independent_of_display():
    effect = Fraction(1) + Fraction(1, 2**54)
    exact_point = effect * Fraction(2, 3)
    values = stats(
        n_units=1,
        n_cycles=1,
        probability_ct=0.75,
        ct_cycles=1,
        tc_cycles=0,
        mean_slope=float(Fraction(2, 3)),
        reference_delta=float(exact_point),
        exact_delta_total=(exact_point.numerator, exact_point.denominator),
    )
    result = estimate_contrast(
        values, procedure(envelope(p=0.75, variance=0), null_abs=1.0)
    ).results[0]
    assert result.estimate.value == float(exact_point)
    assert result.residual_p_value == 0.0
    assert result.estimate.lb == 1.0
    assert result.estimate.ub == math.nextafter(1.0, math.inf)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_provenance_and_numeric_nulls_survive_result_json_and_frames(backend):
    pytest.importorskip(backend)
    import narwhals as nw

    result = estimate_contrast(stats(), procedure(envelope(), alternative="greater")).results[0]
    restored = ContrastResult.model_validate_json(result.model_dump_json())
    assert restored == result
    output = nw.from_native(ContrastResults([result]).to_frame(backend=backend), eager_only=True)
    assert output["dof"].dtype == nw.Float64
    assert output["ub"].dtype == nw.Float64
    assert output["dof"].is_null().to_list() == [True]
    assert output["ub"].is_null().to_list() == [True]
    assert result.reference_spec is not None
    assert json.loads(output["reference_spec"][0]) == result.reference_spec.model_dump(mode="json")
    assert output["open_side"][0] == "upper"
    row = contrast_results_to_readout([result])[0]
    assert row["reference_spec"]["provenance"] == row["provenance"]
    assert row["dof"] is None and row["higher"] is None


def test_probability_arithmetic_never_overflows_a_huge_null_product():
    result = estimate_contrast(stats(), procedure(envelope(), null_abs=1e308)).results[0]
    assert result.residual_p_value is not None
    assert 0 <= result.residual_p_value < result.alpha
    assert result.residual_p_value > 0


def test_hand_calculated_envelope_wire_fixture():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "unit_cycle_envelope_result.json"
    payload = json.loads(path.read_text())
    expected = ContrastResult.model_validate(payload)
    actual = estimate_contrast(
        stats(probability_ct=0.5, mean_slope=1, reference_delta=2, ct_cycles=2, tc_cycles=2),
        procedure(envelope(p=0.5, variance=1), alpha=0.25),
    ).results[0]
    assert actual == expected
    assert actual.model_dump(mode="json") == payload


def test_one_sided_inference_does_not_compute_the_unbounded_endpoint():
    result = estimate_contrast(
        stats(reference_delta=1.7e308, mean_slope=1, probability_ct=0.5),
        procedure(envelope(p=0.5, variance=1e308), alpha=1e-307, alternative="greater"),
    ).results[0]
    assert result.estimate.lb is not None and math.isfinite(result.estimate.lb)
    assert result.estimate.ub is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_alpha", 0.04),
        ("refusal_probability_upper", 0.001),
        ("residual_cutoff", 1.0),
        ("residual_p_value", 0.0),
        ("standard_error_unavailable_reason", "insufficient_replicates"),
        ("provenance", envelope().provenance.model_dump(mode="json")),
        ("response_meaning", "retained_total"),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
def test_serialized_t_result_rejects_envelope_only_metadata(field, value, shared):
    from tests.estimation.test_contrast_results import _shared_result

    result = (
        _shared_result()
        if shared
        else estimate_contrast(stats(m2_delta=4), procedure(UnitCycleTApproximation())).results[0]
    )
    assert ContrastResult.model_validate_json(result.model_dump_json()) == result
    payload = result.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(InvalidRequestError) as caught:
        ContrastResult.model_validate_json(json.dumps(payload))
    assert caught.value.code == "estimation.contrast.reference_metadata"


def test_t_readout_does_not_use_mutated_residual_probability():
    result = estimate_contrast(
        stats(reference_delta=0, m2_delta=4), procedure(UnitCycleTApproximation())
    ).results[0]
    forged = result.model_copy(update={"residual_p_value": 0.0})
    table = contrast_results_to_readout([forged])
    assert table[0]["stat_sig"] is False
