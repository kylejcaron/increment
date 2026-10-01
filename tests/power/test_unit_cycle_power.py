"""Independent finite-law oracles and numerical contracts for unit-cycle planning."""

import itertools
import json
import math
import pickle
from fractions import Fraction

import pytest

from increment._unit_cycle import unit_cycle_envelope_cutoff
from increment.errors import InvalidRequestError
from increment.estimation.decision_types import ContrastDecisionProcedure, PValueEvidence
from increment.power.unit_cycle import (
    UnitCycleLawMoments,
    UnitCycleModelMdeResult,
    UnitCycleModelPowerResult,
    UnitCycleModelRequiredUnitsResult,
    _band,
    _experiments,
    _extreme_accepted,
    _Prefix,
    _reject,
    _RejectionRule,
    _sampler,
    unit_cycle_law_moments,
    unit_cycle_model_mde,
    unit_cycle_model_power,
    unit_cycle_model_required_units,
    unit_cycle_power_lower_bound,
    unit_cycle_variance_envelope,
)
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.unit_cycle import (
    CenteredGammaInnovation,
    CenteredLognormalInnovation,
    NormalInnovation,
    ProspectiveAssumptionProvenance,
    UnitCycleCycleLaw,
    UnitCycleJointLaw,
    UnitCycleTypeLaw,
    UnitCycleVarianceEnvelope,
)


def cycle(**changes):
    values = {
        "ct_mean": 0.0,
        "tc_mean": 0.0,
        "ct_noise_load": 1.0,
        "tc_noise_load": 1.0,
        "ct_innovation_count": 1,
        "tc_innovation_count": 1,
    }
    return UnitCycleCycleLaw.model_validate(dict(values, **changes))


def law(*, p=0.5, types=None, reuse=0.0, innovation=None):
    return UnitCycleJointLaw(
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=p),
            window=SwitchbackWindow(washout_steps=1, observation_steps=2),
        ),
        metric="orders",
        control_group="control",
        treatment_group="treatment",
        response_meaning="retained_total",
        types=types or (UnitCycleTypeLaw(weight=1.0, cycles=(cycle(),)),),
        innovation=innovation or NormalInnovation(),
        reuse_probability=reuse,
        provenance=ProspectiveAssumptionProvenance(
            assumption_id="finite-law-test",
            assumption_version="1",
            justification="prospectively specified finite test population",
            declaration_id="fixture",
        ),
    )


def procedure(model, **changes):
    values = {
        "metric": "orders",
        "role": "primary",
        "alternative": "two-sided",
        "null_abs": 0.0,
        "alpha": 0.05,
        "reference": unit_cycle_variance_envelope(model),
    }
    return ContrastDecisionProcedure.model_validate(dict(values, **changes))


def test_mde_certifies_a_declared_grid_effect_with_selection_safe_replay():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    decision = procedure(model)
    result = unit_cycle_model_mde(
        model,
        decision,
        n=4,
        effect_grid=(0.0, 0.5, 2.0),
        target_power=0.8,
        repetitions=128,
        seed=5,
        mc_error=0.05,
    )
    assert result.feasible_effect == 0.5
    assert result.power_at_feasible is not None
    assert result.power_at_feasible.lower_bound >= result.target_power
    assert Fraction(result.mc_error_per_effect) * 3 <= Fraction(result.mc_error)
    replay = unit_cycle_model_power(
        model,
        decision,
        n=4,
        effect_delta=0.5,
        repetitions=128,
        seed=5,
        mc_error=result.mc_error_per_effect,
    )
    assert replay == result.power_at_feasible
    assert UnitCycleModelMdeResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("reuse,cancel", [(0.0, False), (1.0, False), (1.0, True)])
def test_unrepresentable_innovation_cannot_change_zero_effective_noise(reuse, cancel):
    cycles = (
        (cycle(ct_noise_load=1, tc_noise_load=1), cycle(ct_noise_load=-1, tc_noise_load=-1))
        if cancel
        else (cycle(ct_noise_load=0, tc_noise_load=0),)
    )
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=cycles),),
        reuse=reuse,
        innovation=CenteredLognormalInnovation(shape=1e200),
    )
    result = unit_cycle_model_power(
        model, procedure(model), n=1, effect_delta=1, repetitions=8, seed=5, mc_error=0.1
    )
    assert result.sampling_failed == result.known_runtime_refused == 0
    assert result.power == 1
    assert result.rejected == result.possible_rejected == result.attempted


def rejection_rule(decision: ContrastDecisionProcedure, *, n=1):
    assert isinstance(decision.reference, UnitCycleVarianceEnvelope)
    cutoff = unit_cycle_envelope_cutoff(
        decision.reference, n=n, alpha=decision.alpha, alternative=decision.alternative
    )
    return _RejectionRule.from_design(
        decision.reference, decision, n=n, refusal=cutoff.refusal_probability_upper
    )


def runtime_probability(decision, exact_point):
    from increment.estimation.contrast import ContrastStats, estimate_contrast

    values = ContrastStats(
        metric="orders",
        aggregation="sum",
        probability_ct=0.5,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        washout_steps=1,
        carryover_order=0,
        observation_steps=2,
        retained_steps=2,
        control_group="control",
        treatment_group="treatment",
        n_units=1,
        n_cycles=1,
        reference_delta=float(exact_point),
        mean_residual=0,
        m2_delta=0,
        mean_slope=1,
        ct_cycles=1,
        tc_cycles=0,
        minimum_cycles_per_unit=1,
        maximum_cycles_per_unit=1,
        exact_delta_total=(exact_point.numerator, exact_point.denominator),
    )
    evidence = next(iter(estimate_contrast(values, decision).evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    return evidence.p_value


def enumerate_moments(model):
    """Enumerate types and full order masks, then mix conditional noise moments."""
    p = Fraction(model.assignment.sequence.probability_ct)
    total = sum(Fraction(t.weight) for t in model.types)
    c = len(model.types[0].cycles)
    entries = []
    for t in model.types:
        for orders in itertools.product((True, False), repeat=c):
            mass = Fraction(t.weight) / total
            means, loads, counts, slopes = [], [], [], []
            for item, ct in zip(t.cycles, orders, strict=True):
                chance = p if ct else 1 - p
                mass *= chance
                w = 1 / (2 * chance)
                means.append(w * Fraction(item.ct_mean if ct else item.tc_mean) / c)
                loads.append(w * Fraction(item.ct_noise_load if ct else item.tc_noise_load) / c)
                counts.append(item.ct_innovation_count if ct else item.tc_innovation_count)
                slopes.append(w / c)
            noise = Fraction(model.reuse_probability) * sum(loads) ** 2 + (
                1 - Fraction(model.reuse_probability)
            ) * sum(x**2 / k for x, k in zip(loads, counts, strict=True))
            entries.append((mass, sum(means), sum(slopes), noise))
    mean = sum(mass * a for mass, a, _, _ in entries)
    var_a = sum(mass * ((a - mean) ** 2 + noise) for mass, a, _, noise in entries)
    var_g = sum(mass * (g - 1) ** 2 for mass, _, g, _ in entries)
    cov = sum(mass * (a - mean) * (g - 1) for mass, a, g, _ in entries)
    residual = sum(mass * ((a - mean * g) ** 2 + noise) for mass, a, g, noise in entries)
    return mean, var_a, var_g, cov, residual


@pytest.mark.parametrize("reuse", [0.0, 0.3, 1.0])
@pytest.mark.parametrize(
    "innovation",
    [
        NormalInnovation(),
        CenteredGammaInnovation(shape=0.4),
        CenteredLognormalInnovation(shape=1.2),
    ],
)
def test_moments_match_independent_full_mask_enumeration(reuse, innovation):
    model = law(
        p=0.9,
        reuse=reuse,
        innovation=innovation,
        types=(
            UnitCycleTypeLaw(
                weight=0.1,
                cycles=(
                    cycle(
                        ct_mean=10,
                        tc_mean=-2,
                        ct_noise_load=3,
                        tc_noise_load=-2,
                        ct_innovation_count=2,
                        tc_innovation_count=5,
                    ),
                    cycle(ct_mean=-4, tc_mean=8, ct_noise_load=-1, tc_noise_load=4),
                ),
            ),
            UnitCycleTypeLaw(
                weight=0.2,
                cycles=(
                    cycle(ct_mean=-2, tc_mean=3, ct_noise_load=2, tc_noise_load=0),
                    cycle(ct_mean=6, tc_mean=-1, ct_innovation_count=3),
                ),
            ),
        ),
    )
    mean, va, vg, cov, residual = enumerate_moments(model)
    result = unit_cycle_law_moments(model)
    assert result.reference_effect == float(mean)
    assert result.sd_a == pytest.approx(math.sqrt(float(va)))
    assert result.sd_g == pytest.approx(math.sqrt(float(vg)))
    assert result.rho == pytest.approx(float(cov) / math.sqrt(float(va * vg)))
    upper = result.residual_variance_upper
    assert upper is not None and Fraction(upper) >= residual
    assert Fraction(math.nextafter(upper, -math.inf)) < residual
    assert UnitCycleLawMoments.model_validate_json(result.model_dump_json()) == result


def test_reference_centering_removes_random_slope_variance():
    model = law(
        p=0.9,
        types=(
            UnitCycleTypeLaw(
                weight=3, cycles=(cycle(ct_mean=7, tc_mean=7, ct_noise_load=0, tc_noise_load=0),)
            ),
        ),
    )
    result = unit_cycle_law_moments(model)
    assert result.reference_effect == 7
    assert result.sd_a is not None and result.sd_a > 0
    assert result.residual_variance_upper == 0
    assert unit_cycle_variance_envelope(model).residual_variance_upper == 0


def test_common_innovation_is_not_divided_by_multiplicities():
    item = cycle(ct_innovation_count=100, tc_innovation_count=100)
    types = (UnitCycleTypeLaw(weight=1, cycles=(item, item)),)
    assert unit_cycle_law_moments(law(types=types, reuse=1)).residual_variance_upper == 1
    independent = unit_cycle_law_moments(law(types=types, reuse=0))
    assert independent.residual_variance_upper == pytest.approx(1 / 200)


def test_opposite_common_loads_cancel_but_independent_loads_do_not():
    types = (
        UnitCycleTypeLaw(weight=1, cycles=(cycle(), cycle(ct_noise_load=-1, tc_noise_load=-1))),
    )
    assert unit_cycle_law_moments(law(types=types, reuse=1)).residual_variance_upper == 0
    assert unit_cycle_law_moments(law(types=types, reuse=0)).residual_variance_upper == 0.5


def test_large_offsets_neighboring_values_and_type_order():
    a, b = 1e150, math.nextafter(1e150, math.inf)
    types = tuple(
        UnitCycleTypeLaw(
            weight=w, cycles=(cycle(ct_mean=m, tc_mean=m, ct_noise_load=0, tc_noise_load=0),)
        )
        for w, m in ((1, a), (2, b))
    )
    model = law(p=0.5, types=types)
    exact = Fraction(2, 9) * (Fraction(a) - Fraction(b)) ** 2
    upper = unit_cycle_variance_envelope(model).residual_variance_upper
    assert Fraction(upper) >= exact
    assert Fraction(math.nextafter(upper, -math.inf)) < exact
    assert unit_cycle_variance_envelope(law(types=types[::-1])).residual_variance_upper == upper


def test_unrepresentable_sd_does_not_destroy_identified_residual():
    model = law(
        p=0.1,
        types=(
            UnitCycleTypeLaw(
                weight=1,
                cycles=(cycle(ct_mean=1.7e308, tc_mean=1.7e308, ct_noise_load=0, tc_noise_load=0),),
            ),
        ),
    )
    row = unit_cycle_law_moments(model)
    assert row.sd_a is None and row.sd_a_unavailable_reason == "unrepresentable"
    assert row.reference_effect == 1.7e308
    assert row.residual_variance_upper == 0
    assert '"sd_a":null' in row.model_dump_json()


def test_positive_subnormal_variance_rounds_up_instead_of_becoming_zero():
    model = law(
        types=(
            UnitCycleTypeLaw(
                weight=1, cycles=(cycle(ct_noise_load=math.ulp(0.0), tc_noise_load=math.ulp(0.0)),)
            ),
        )
    )
    assert unit_cycle_variance_envelope(model).residual_variance_upper == math.ulp(0.0)


def test_unrepresentable_residual_variance_refusal_reports_its_exact_magnitude():
    model = law(types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_mean=1e200, tc_mean=-1e200),)),))
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_variance_envelope(model)
    assert error.value.code == "power.unit_cycle.numerical"
    assert error.value.context["field"] == "residual_variance_upper"
    exact = Fraction(1e200) ** 2 + 1
    assert abs(Fraction(str(error.value.context["value"])) / exact - 1) < Fraction(1, 10**16)


def test_zero_sd_has_explicit_correlation_reason():
    row = unit_cycle_law_moments(law())
    assert row.sd_g == 0 and row.rho is None
    assert row.rho_unavailable_reason == "zero_variance"


def test_envelope_carries_complete_identity_and_copies_assignment():
    model = law()
    envelope = unit_cycle_variance_envelope(model)
    assert envelope.assignment == model.assignment
    assert envelope.assignment is not model.assignment
    assert envelope.provenance == model.provenance
    assert envelope.cycles_per_unit == 1
    assert envelope.response_meaning == "retained_total"
    assert envelope.effect_model == "additive_retained_aggregate_shift"


def test_lower_bound_matches_rational_cantelli_and_keeps_effect_scale():
    model = law(p=0.75)
    decision = procedure(model, alternative="less", null_abs=2)
    envelope = unit_cycle_variance_envelope(model)
    row = unit_cycle_power_lower_bound(envelope, decision, n=40, effect_delta=-10)
    cap = Fraction(math.nextafter(decision.alpha, 0)) - Fraction(row.refusal_probability_upper)
    squared = Fraction(envelope.residual_variance_upper) / 40 * (1 - cap) / cap
    radius = math.sqrt(float(squared))
    if Fraction(radius) ** 2 < squared:
        radius = math.nextafter(radius, math.inf)
    a = Fraction(12) / Fraction(1.5) - Fraction(radius)
    expected = max(
        Fraction(),
        a * a / (Fraction(envelope.residual_variance_upper) / 40 + a * a)
        - Fraction(row.refusal_probability_upper),
    )
    assert Fraction(row.lower_bound) <= expected
    assert row.lower_bound == pytest.approx(float(expected))
    assert row.power_kind == "certified_power_lower_bound"
    assert (
        unit_cycle_power_lower_bound(envelope, decision, n=40, effect_delta=3).reason
        == "non_favorable_effect"
    )


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"metric": "other"}, "decision.contrast_decision.reference_metric"),
        ({"reference": None}, "power.unit_cycle.procedure"),
    ],
)
def test_mismatched_procedure_refuses_before_sampling(changes, code):
    model = law()
    with pytest.raises(InvalidRequestError) as error:
        decision = procedure(model, **changes)
        unit_cycle_model_power(
            model, decision, n=2, effect_delta=1, repetitions=1, seed=1, mc_error=0.1
        )
    assert error.value.code == code


@pytest.mark.parametrize(
    "reference_residual,field,value",
    [
        (
            2.0,
            "procedure.reference.residual_variance_upper/envelope.residual_variance_upper",
            (2.0, 1.0),
        ),
        (None, "procedure.reference", None),
    ],
)
def test_mismatched_reference_refusal_names_the_disagreeing_envelope_fields(
    reference_residual, field, value
):
    model = law()
    envelope = unit_cycle_variance_envelope(model)
    reference = (
        None
        if reference_residual is None
        else envelope.model_copy(update={"residual_variance_upper": reference_residual})
    )
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_power_lower_bound(
            envelope, procedure(model, reference=reference), n=2, effect_delta=1
        )
    assert error.value.code == "power.unit_cycle.procedure"
    assert error.value.context["field"] == field
    assert error.value.context["value"] == value


def test_forged_law_validation_is_translated_to_law_refusal():
    forged = law().model_copy(update={"metric": ""})
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_law_moments(forged)
    assert error.value.code == "model.field.length"


@pytest.mark.parametrize(
    "nested_field,forged_value,expected_code",
    [
        (
            "sequence",
            law().assignment.sequence.model_copy(update={"probability_ct": 0.0}),
            "power.unit_cycle.law",
        ),
        (
            "window",
            law().assignment.window.model_copy(update={"carryover_order": 2}),
            "definition.assignment.carryover_order_range",
        ),
    ],
)
def test_nested_forged_assignment_is_refused_at_law_boundary(
    nested_field, forged_value, expected_code
):
    assignment = law().assignment
    forged = law().model_copy(
        update={
            "assignment": {
                "kind": assignment.kind,
                "periods_per_cycle": assignment.periods_per_cycle,
                "sequence": forged_value if nested_field == "sequence" else assignment.sequence,
                "window": forged_value if nested_field == "window" else assignment.window,
            }
        }
    )
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_law_moments(forged)
    assert error.value.code == expected_code


def test_forged_law_refusal_names_the_rejected_nested_field():
    assignment = law().assignment
    sequence = assignment.sequence.model_copy(update={"probability_ct": 0.0})
    forged = law().model_copy(update={"assignment": {**dict(assignment), "sequence": sequence}})
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_law_moments(forged)
    assert error.value.code == "power.unit_cycle.law"
    field = error.value.context["field"]
    assert isinstance(field, str)
    assert field.startswith("law.assignment")


def test_available_power_must_match_rejections_in_serialized_result():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    result = unit_cycle_model_power(
        model, procedure(model), n=1, effect_delta=0, repetitions=1, seed=5, mc_error=0.1
    )
    assert result.rejected == 0 and result.power == 0
    payload = result.model_dump(mode="json")
    payload["power"] = 1.0
    with pytest.raises(InvalidRequestError) as error:
        UnitCycleModelPowerResult.model_validate_json(json.dumps(payload))
    assert error.value.code == "power.unit_cycle.result"
    assert error.value.context["field"] == "power/rejected/attempted"
    assert error.value.context["value"] == (1.0, 0, 1)
    restored = pickle.loads(pickle.dumps(error.value))
    assert (restored.code, restored.context) == (error.value.code, error.value.context)


def test_missing_unavailable_reason_refusal_retains_the_null_pair():
    payload = unit_cycle_law_moments(law()).model_dump()
    payload["rho_unavailable_reason"] = None
    with pytest.raises(InvalidRequestError) as error:
        UnitCycleLawMoments.model_validate(payload)
    assert error.value.code == "power.unit_cycle.result"
    assert error.value.context["field"] == "rho/rho_unavailable_reason"
    assert error.value.context["value"] == (None, None)


def test_forged_procedure_validation_is_translated_before_calculation(monkeypatch):
    model = law()
    forged = procedure(model).model_copy(update={"alpha": 2.0})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid procedure reached the power calculation")

    monkeypatch.setattr("increment.power.unit_cycle.unit_cycle_variance_envelope", forbidden)
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_model_power(
            model, forged, n=2, effect_delta=1, repetitions=1, seed=1, mc_error=0.1
        )
    assert error.value.code == "model.field.range"


def test_forged_envelope_validation_is_translated_to_procedure_refusal():
    model = law()
    envelope = unit_cycle_variance_envelope(model)
    forged = envelope.model_copy(update={"cycles_per_unit": 0})
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_power_lower_bound(forged, procedure(model), n=2, effect_delta=1)
    assert error.value.code == "model.field.range"


def test_exhausted_admission_budget_refuses_before_sampling():
    model = law()
    decision = procedure(model, alpha=1e-20)
    with pytest.raises(InvalidRequestError) as error:
        unit_cycle_model_power(
            model, decision, n=40, effect_delta=1, repetitions=1, seed=1, mc_error=0.1
        )
    assert error.value.code == "unit_cycle.error_budget_exhausted"


@pytest.mark.parametrize(
    "innovation",
    [NormalInnovation(), CenteredGammaInnovation(shape=2), CenteredLognormalInnovation(shape=0.7)],
)
def test_batch_size_does_not_change_power_or_required_n_results(innovation):
    """Batching only schedules the unit stream: the public results at a fixed
    (n, seed, repetitions) must not depend on it."""
    model = law(p=0.75, reuse=0.4, innovation=innovation)
    decision = procedure(model)
    singles = unit_cycle_model_power(
        model, decision, n=3, effect_delta=1, repetitions=7, seed=14, mc_error=0.1, batch_size=1
    )
    batched = unit_cycle_model_power(
        model, decision, n=3, effect_delta=1, repetitions=7, seed=14, mc_error=0.1, batch_size=4
    )
    assert singles == batched
    required_singles = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=1,
        target_power=0.5,
        max_n=3,
        repetitions=7,
        seed=14,
        mc_error=0.1,
        batch_size=1,
    )
    required_batched = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=1,
        target_power=0.5,
        max_n=3,
        repetitions=7,
        seed=14,
        mc_error=0.1,
        batch_size=4,
    )
    assert required_singles == required_batched


def test_huge_repetition_request_streams_before_allocating_other_experiments(monkeypatch):
    import increment.power.unit_cycle as planning

    visited = []
    original = planning._prefixes

    def recording(*args, **kwargs):
        visited.append(kwargs["replicate"])
        yield from original(*args, **kwargs)

    monkeypatch.setattr(planning, "_prefixes", recording)
    stream = _experiments(_sampler(law()), n=1, repetitions=10**12, seed=1, batch_size=512)
    next(stream)
    assert visited == [0]
    stream.close()


def test_required_units_checks_every_n_and_round_trips():
    model = law()
    row = unit_cycle_model_required_units(
        model,
        procedure(model),
        effect_delta=1,
        target_power=0.8,
        max_n=4,
        repetitions=8,
        seed=1,
        mc_error=0.1,
        batch_size=3,
    )
    assert row.checked_n == 4
    assert row.attempted_across_n == 4 * 8
    assert UnitCycleModelRequiredUnitsResult.model_validate_json(row.model_dump_json()) == row


def test_zero_envelope_strict_tie_single_unit_and_wire():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    decision = procedure(model)
    null = unit_cycle_model_power(
        model, decision, n=1, repetitions=32, seed=5, mc_error=0.1, effect_delta=0
    )
    nonzero = unit_cycle_model_power(
        model, decision, n=1, repetitions=32, seed=5, mc_error=0.1, effect_delta=math.ulp(0.0)
    )
    assert null.rejected == 0 and null.power == 0
    assert nonzero.rejected == 32 and nonzero.power == 1
    assert nonzero.failed == 0 and nonzero.admitted == 32
    assert UnitCycleModelPowerResult.model_validate_json(nonzero.model_dump_json()) == nonzero


def test_fixed_effect_power_uses_one_bernoulli_error_pair():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    result = unit_cycle_model_power(
        model, procedure(model), n=1, effect_delta=1, repetitions=32, seed=5, mc_error=0.1
    )
    # One-sided Hoeffding with error .05 has radius sqrt(log(20)/64) < .22.
    assert result.power == 1
    assert result.lower_bound > 0.78
    assert result.upper_bound == 1


def test_fixed_effect_size_search_allocates_over_sizes_not_effect_curves():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    result = unit_cycle_model_required_units(
        model,
        procedure(model),
        effect_delta=1,
        target_power=0.7,
        max_n=2,
        repetitions=32,
        seed=5,
        mc_error=0.1,
    )
    # Two sizes and two one-sided tails: sqrt(log(40)/64) < .25.
    assert result.feasible_n == 1
    assert result.power_at_feasible is not None
    assert result.power_at_feasible.lower_bound > 0.75


@pytest.mark.parametrize(
    "error,method,allocations",
    [
        (1e-300, "simultaneous_binary64_hoeffding", 2**65),
        (1e-320, "simultaneous_dkw_massart", 8),
    ],
)
def test_simultaneous_bounds_preserve_extreme_error_budgets(error, method, allocations):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    result = unit_cycle_model_power(
        model,
        procedure(model),
        n=1,
        effect_delta=1,
        repetitions=8,
        seed=5,
        mc_error=error,
        simultaneous_effects=True,
    )
    assert result.mc_method == method
    assert 0 < Fraction(result.cdf_error_each) * allocations <= Fraction(error)
    assert result.power == result.upper_bound == 1
    assert result.failed == 0
    assert UnitCycleModelPowerResult.model_validate_json(result.model_dump_json()) == result


def _sampling_protocol_result(kind):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    decision = procedure(model, alternative="greater")
    if kind == "power":
        result = unit_cycle_model_power(
            model, decision, n=1, effect_delta=1, repetitions=32, seed=5, mc_error=0.1
        )
    elif kind == "mde":
        result = unit_cycle_model_mde(
            model,
            decision,
            n=1,
            effect_grid=(0.0, 1.0),
            target_power=0.4,
            repetitions=32,
            seed=5,
            mc_error=0.1,
        )
    else:
        result = unit_cycle_model_required_units(
            model,
            decision,
            effect_delta=1,
            target_power=0.5,
            max_n=1,
            repetitions=32,
            seed=5,
            mc_error=0.1,
        )
    return result


@pytest.mark.parametrize("kind", ["power", "mde", "required_units"])
@pytest.mark.parametrize("version,omit_protocol", [(1, False), (1, True), (2, False)])
def test_sampling_protocol_wire_compatibility(kind, version, omit_protocol):
    result = _sampling_protocol_result(kind)
    payload = result.model_dump(mode="json")
    compact = result.model_dump(exclude_defaults=True)
    assert compact["rng"] == payload["rng"]
    assert compact["sampler"] == payload["sampler"]
    records = [payload]
    if kind != "power":
        assert payload["power_at_feasible"] is not None
        records.append(payload["power_at_feasible"])
        assert compact["power_at_feasible"]["rng"] == records[-1]["rng"]
        assert compact["power_at_feasible"]["sampler"] == records[-1]["sampler"]
    for record in records:
        assert record["rng"] == "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2"
        assert record["sampler"] == "finite_type_reuse_exact_micro_v2"
        if version == 1:
            if omit_protocol:
                del record["rng"], record["sampler"]
            else:
                record["rng"] = "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1"
                record["sampler"] = "finite_type_reuse_exact_micro_v1"
    restored = type(result).model_validate_json(json.dumps(payload))
    restored_records = [restored.model_dump()]
    if kind != "power":
        restored_records.append(restored_records[0]["power_at_feasible"])
    for record in restored_records:
        assert record["rng"] == f"numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v{version}"
        assert record["sampler"] == f"finite_type_reuse_exact_micro_v{version}"
    assert type(result).model_validate_json(restored.model_dump_json()) == restored


@pytest.mark.parametrize("kind", ["power", "mde", "required_units"])
@pytest.mark.parametrize("legacy_field", ["rng", "sampler"])
@pytest.mark.parametrize("omit_legacy", [False, True])
@pytest.mark.parametrize("deserialize", [False, True])
def test_sampling_protocol_rejects_mixed_fields(kind, legacy_field, omit_legacy, deserialize):
    result = _sampling_protocol_result(kind)
    payload = result.model_dump(mode="json")
    if omit_legacy:
        del payload[legacy_field]
    else:
        payload[legacy_field] = payload[legacy_field].replace("v2", "v1")
    with pytest.raises(InvalidRequestError) as caught:
        if deserialize:
            type(result).model_validate_json(json.dumps(payload))
        else:
            type(result)(**payload)
    assert caught.value.code == "power.unit_cycle.result"


@pytest.mark.parametrize("kind", ["mde", "required_units"])
@pytest.mark.parametrize("legacy_record", ["parent", "child"])
@pytest.mark.parametrize("omit_legacy", [False, True])
@pytest.mark.parametrize("deserialize", [False, True])
def test_sampling_protocol_rejects_mixed_certificate(kind, legacy_record, omit_legacy, deserialize):
    result = _sampling_protocol_result(kind)
    payload = result.model_dump(mode="json")
    assert payload["power_at_feasible"] is not None
    record = payload if legacy_record == "parent" else payload["power_at_feasible"]
    for field in ("rng", "sampler"):
        if omit_legacy:
            del record[field]
        else:
            record[field] = record[field].replace("v2", "v1")
    with pytest.raises(InvalidRequestError) as caught:
        if deserialize:
            type(result).model_validate_json(json.dumps(payload))
        else:
            type(result)(**payload)
    assert caught.value.code == "power.unit_cycle.result"


@pytest.mark.parametrize("kind", ["mde", "required_units"])
@pytest.mark.parametrize("changed_record", ["parent", "child"])
def test_sampling_protocol_rejects_mixed_numpy_certificate(kind, changed_record):
    result = _sampling_protocol_result(kind)
    payload = result.model_dump(mode="json")
    assert payload["power_at_feasible"] is not None
    payload["numpy_version"] = payload["power_at_feasible"]["numpy_version"] = "1.26.4"
    restored = type(result).model_validate_json(json.dumps(payload))
    assert restored.model_dump(mode="json") == payload

    record = payload if changed_record == "parent" else payload["power_at_feasible"]
    record["numpy_version"] = "2.0.0"
    with pytest.raises(InvalidRequestError) as caught:
        type(result).model_validate_json(json.dumps(payload))
    assert caught.value.code == "power.unit_cycle.result"


@pytest.mark.parametrize("simultaneous", [False, True])
def test_power_normalizes_effect_before_rejection_and_replay(simultaneous):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    effect = 2**53 + 1
    decision = procedure(model, null_abs=float(effect))
    result = unit_cycle_model_power(
        model,
        decision,
        n=1,
        effect_delta=effect,
        repetitions=8,
        seed=5,
        mc_error=0.1,
        simultaneous_effects=simultaneous,
    )
    restored = UnitCycleModelPowerResult.model_validate_json(result.model_dump_json())
    replay = unit_cycle_model_power(
        model,
        decision,
        n=1,
        effect_delta=restored.effect_delta,
        repetitions=8,
        seed=5,
        mc_error=0.1,
        simultaneous_effects=simultaneous,
    )
    assert result.effect_delta == float(effect)
    assert result.rejected == result.possible_rejected == 0
    assert result == restored == replay


def test_required_units_normalizes_effect_before_selecting_sizes():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    effect = 2**53 + 1
    decision = procedure(model, null_abs=float(effect))
    result = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=effect,
        target_power=0.5,
        max_n=1,
        repetitions=32,
        seed=5,
        mc_error=0.1,
    )
    replay = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=result.effect_delta,
        target_power=0.5,
        max_n=1,
        repetitions=32,
        seed=5,
        mc_error=0.1,
    )
    assert result.effect_delta == float(effect)
    assert result.plausible_n is result.feasible_n is None
    assert result == replay


def test_lower_bound_normalizes_effect_before_computing_separation():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    effect = 2**53 + 1
    decision = procedure(model, null_abs=float(effect))
    result = unit_cycle_power_lower_bound(
        unit_cycle_variance_envelope(model), decision, n=1, effect_delta=effect
    )
    assert result.effect_delta == float(effect)
    assert result.lower_bound == 0
    assert result.reason == "non_favorable_effect"


def test_pointwise_bounds_cannot_be_relabelled_as_simultaneous():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    result = unit_cycle_model_power(
        model, procedure(model), n=1, effect_delta=1, repetitions=32, seed=5, mc_error=0.1
    )
    payload = result.model_dump()
    payload.update(
        mc_method="simultaneous_dkw_massart",
        cdf_bands=8,
        cdf_terms_per_bound=4,
        cdf_error_each=0.1 / 8,
    )
    with pytest.raises(InvalidRequestError) as error:
        UnitCycleModelPowerResult.model_validate(payload)
    assert error.value.code == "power.unit_cycle.result"


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_exact_deployed_ties_and_neighboring_residuals_match_runtime(alternative):
    model = law()
    variance = 0.5 if alternative == "two-sided" else 1.0
    envelope = unit_cycle_variance_envelope(model).model_copy(
        update={"residual_variance_upper": variance}
    )
    decision = procedure(
        model, alternative=alternative, alpha=math.nextafter(0.5, math.inf), reference=envelope
    )
    rule = rejection_rule(decision)
    assert rule.squared == 1
    sign = -1 if alternative == "less" else 1
    for residual, expected in (
        (Fraction(math.nextafter(1, 0)), False),
        (Fraction(1), True),
        (Fraction(math.nextafter(1, math.inf)), True),
    ):
        exact = sign * residual
        assert _reject(exact, Fraction(1), 0, decision, rule) == expected
        assert (runtime_probability(decision, exact) < decision.alpha) == expected
    assert _reject(-sign * Fraction(1), Fraction(1), 0, decision, rule) == (
        alternative == "two-sided"
    )


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_public_power_keeps_pvalue_rounding_atom_above_display_cutoff(monkeypatch, alternative):
    load = 0.25 if alternative == "two-sided" else 0.5
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=load, tc_noise_load=load),)),)
    )
    decision = procedure(model, alternative=alternative, alpha=0.25 if load == 0.25 else 0.5)
    sign = -1 if alternative == "less" else 1
    u = sign * Fraction(1, 2**57)

    def experiments(*args, **kwargs):
        for _ in range(kwargs["repetitions"]):
            yield _Prefix(1, u, Fraction(1), 1, None)

    monkeypatch.setattr("increment.power.unit_cycle._experiments", experiments)
    for effect, expected in ((sign * 0.5, False), (sign * math.nextafter(0.5, math.inf), True)):
        exact = u + Fraction(effect)
        p_value = runtime_probability(decision, exact)
        row = unit_cycle_model_power(
            model, decision, n=1, effect_delta=effect, repetitions=8, seed=17, mc_error=0.1
        )
        assert row.cutoff == 0.5
        assert abs(exact) > Fraction(row.cutoff)
        assert (p_value < decision.alpha) == expected
        assert row.rejected == 8 * expected
        if not expected:
            assert p_value == decision.alpha


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_effect_minus_nonzero_null_is_never_subtracted_as_float(alternative):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    sign = -1 if alternative == "less" else 1
    decision = procedure(model, alternative=alternative, null_abs=-sign * 2**-54)
    assert sign - decision.null_abs == sign
    assert _reject(Fraction(-sign), Fraction(1), sign, decision, rejection_rule(decision))
    assert runtime_probability(decision, Fraction()) < decision.alpha


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("variance_zero", [False, True])
def test_zero_cap_has_no_finite_positive_variance_rejection(alternative, variance_zero):
    load = 0 if variance_zero else math.ulp(0.0)
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=load, tc_noise_load=load),)),)
    )
    decision = procedure(model, alternative=alternative, alpha=math.ulp(0.0))
    sign = -1 if alternative == "less" else 1
    fixed = unit_cycle_model_power(
        model, decision, n=1, effect_delta=sign, repetitions=32, seed=1, mc_error=0.1
    )
    assert fixed.rejected == 32 * variance_zero
    mde = unit_cycle_model_mde(
        model,
        decision,
        n=1,
        effect_grid=(0.0, sign * math.ulp(0.0)),
        target_power=0.1,
        repetitions=32,
        seed=1,
        mc_error=0.1,
    )
    expected_effect = sign * math.ulp(0.0) if variance_zero else None
    assert mde.feasible_effect == expected_effect
    sizes = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=sign,
        target_power=0.1,
        max_n=2,
        repetitions=32,
        seed=1,
        mc_error=0.1,
    )
    expected_n = 1 if variance_zero else None
    assert sizes.feasible_n == expected_n
    assert sizes.unavailable_n == 0
    if not variance_zero:
        assert mde.feasible_unavailable_reason == "grid_exhausted"
        assert isinstance(decision.reference, UnitCycleVarianceEnvelope)
        bound = unit_cycle_power_lower_bound(decision.reference, decision, n=1, effect_delta=sign)
        assert bound.lower_bound == 0


def test_admission_bound_equal_to_alpha_predecessor_leaves_zero_cap():
    model = law()
    decision = procedure(model, alpha=0.25)
    assert isinstance(decision.reference, UnitCycleVarianceEnvelope)
    rule = _RejectionRule.from_design(
        decision.reference, decision, n=1, refusal=math.nextafter(0.25, 0)
    )
    assert not _reject(Fraction(10**400), Fraction(1), 1, decision, rule)


def test_runtime_and_model_agree_at_neighboring_float_zero_envelope_boundary():
    from increment.estimation.contrast import ContrastStats, estimate_contrast

    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    decision = procedure(model, null_abs=1)
    effect = math.nextafter(1.0, math.inf)
    slope = 1 / (2 * 0.75)
    exact_total = Fraction(effect) * Fraction(2, 3)
    stats = ContrastStats(
        metric="orders",
        aggregation="sum",
        probability_ct=0.75,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        washout_steps=1,
        carryover_order=0,
        observation_steps=2,
        retained_steps=2,
        control_group="control",
        treatment_group="treatment",
        n_units=1,
        n_cycles=1,
        reference_delta=effect * slope,
        mean_residual=0,
        m2_delta=0,
        mean_slope=slope,
        ct_cycles=1,
        tc_cycles=0,
        minimum_cycles_per_unit=1,
        maximum_cycles_per_unit=1,
        exact_delta_total=(exact_total.numerator, exact_total.denominator),
    )
    runtime = estimate_contrast(stats, decision)
    evidence = next(iter(runtime.evidence.values()))
    assert isinstance(evidence, PValueEvidence)
    expected = _reject(Fraction(), Fraction(2, 3), effect, decision, rejection_rule(decision))
    assert (evidence.p_value < decision.alpha) == expected


@pytest.mark.parametrize(
    "alternative,effect",
    [
        ("two-sided", math.nextafter(math.inf, 0)),
        ("two-sided", math.nextafter(math.nextafter(math.inf, 0), 0)),
        ("two-sided", -math.nextafter(math.inf, 0)),
        ("two-sided", -math.nextafter(math.nextafter(math.inf, 0), 0)),
        ("greater", math.nextafter(math.inf, 0)),
        ("greater", -math.nextafter(math.inf, 0)),
        ("less", math.nextafter(math.inf, 0)),
        ("less", -math.nextafter(math.inf, 0)),
    ],
)
def test_model_power_preserves_runtime_availability_at_finite_range(alternative, effect):
    model = law(
        types=tuple(
            UnitCycleTypeLaw(
                weight=1,
                cycles=(cycle(ct_mean=u, tc_mean=u, ct_noise_load=0, tc_noise_load=0),),
            )
            for u in (-1, 1)
        )
    )
    decision = procedure(model, alternative=alternative)
    outcomes = set()
    for residual in (-1, 1):
        try:
            probability = runtime_probability(decision, Fraction(effect) + residual)
        except InvalidRequestError as exc:
            assert exc.code == "unit_cycle.numerical"
            outcomes.add(None)
        else:
            outcomes.add(probability < decision.alpha)
    # Both population atoms have the same availability and decision here.
    assert len(outcomes) == 1
    expected = outcomes.pop()
    result = unit_cycle_model_power(
        model, decision, n=1, effect_delta=effect, repetitions=4, seed=0, mc_error=0.01
    )
    assert result.attempted == result.admitted == 4
    if expected is None:
        assert result.failed == 4
        assert result.rejected == 0
        assert result.possible_rejected == 0
        assert result.known_runtime_refused == 4
        assert result.availability_uncertified == result.sampling_failed == 0
        assert result.power is None
        assert result.power_unavailable_reason == "numerical_failures"
    else:
        assert result.failed == 0
        assert result.rejected == 4 * expected
        assert result.power == float(expected)


def affine_runtime(decision, exact_units, orders):
    """Run the specified rounded-unit adapter through the public scalar reducer."""
    from increment.estimation.contrast import (
        ContrastPartition,
        estimate_contrast,
        reduce_contrast_partitions,
    )

    p = Fraction(decision.reference.assignment.sequence.probability_ct)
    units = {str(i): value for i, value in enumerate(exact_units)}
    partition = ContrastPartition(
        metric="orders",
        aggregation="sum",
        probability_ct=float(p),
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        washout_steps=1,
        carryover_order=0,
        observation_steps=2,
        retained_steps=2,
        control_group="control",
        treatment_group="treatment",
        unit_deltas={key: float(value) for key, value in units.items()},
        exact_unit_deltas={
            key: (value.numerator, value.denominator) for key, value in units.items()
        },
        cycles_by_unit=dict.fromkeys(units, 1),
        unit_slopes={str(i): float(1 / (2 * (p if ct else 1 - p))) for i, ct in enumerate(orders)},
        ct_counts_by_unit={str(i): int(ct) for i, ct in enumerate(orders)},
    )
    return estimate_contrast(reduce_contrast_partitions([partition]), decision)


@pytest.mark.parametrize("direction,alternative", [(1, "greater"), (-1, "less")])
def test_public_planner_keeps_asymmetric_outward_endpoint_support(direction, alternative):
    model = law(
        types=tuple(
            UnitCycleTypeLaw(
                weight=weight,
                cycles=(
                    cycle(
                        ct_mean=direction * u,
                        tc_mean=direction * u,
                        ct_noise_load=0,
                        tc_noise_load=0,
                    ),
                ),
            )
            for weight, u in ((1, 100), (100, -1))
        )
    )
    decision = procedure(model, alternative=alternative)
    effect = direction * math.nextafter(math.inf, 0)
    assert runtime_probability(decision, Fraction(effect) + direction * 100) < decision.alpha
    row = unit_cycle_model_power(
        model, decision, n=1, effect_delta=effect, repetitions=8, seed=0, mc_error=0.1
    )
    assert row.failed == 0 and row.power == 1
    assert row.rejected == row.possible_rejected == row.attempted


@pytest.mark.parametrize("direction", [1, -1])
def test_identical_large_units_keep_finite_descriptive_state_and_guarantee(direction):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    decision = procedure(model, preferred_direction="increase" if direction == 1 else "decrease")
    effect = direction * math.nextafter(math.inf, 0)
    runtime = affine_runtime(decision, [Fraction(effect)] * 2, [True, False])
    assert runtime.results[0].standard_error == 0
    row = unit_cycle_model_power(
        model, decision, n=2, effect_delta=effect, repetitions=4, seed=0, mc_error=0.1
    )
    assert row.power == 1 and row.failed == 0
    assert row.rejected == row.possible_rejected == 4
    bound = unit_cycle_power_lower_bound(decision.reference, decision, n=2, effect_delta=effect)
    assert bound.lower_bound == 1


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "limit,farthest", [(0, 0), (1, 0), (1, 1), (7, 3), (5000, 4999), (5000, 0)]
)
def test_extreme_accepted_finds_the_boundary_from_any_guess(direction, limit, farthest):
    inner = 12
    outer = inner + direction * limit
    expected = inner + direction * farthest

    def accepted(rank):
        return direction * (rank - inner) <= farthest

    for guess in (expected, inner, outer, inner - direction * 40, outer + direction * 40):
        assert _extreme_accepted(accepted, inner=inner, outer=outer, guess=guess) == expected


@pytest.mark.slow
@pytest.mark.parametrize("direction,alternative", [(1, "greater"), (-1, "less")])
def test_finite_mean_with_overflowing_descriptive_variance_remains_unresolved(
    direction, alternative
):
    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    decision = procedure(model, alternative=alternative)
    effect = direction * 1e200
    units = [Fraction(effect) * Fraction(2, 3), Fraction(effect) * 2]
    assert math.isfinite(float(sum(units) / 2))
    with pytest.raises(InvalidRequestError) as caught:
        affine_runtime(decision, units, [True, False])
    assert caught.value.code == "estimation.contrast.contrast_partition_centered"
    row = unit_cycle_model_power(
        model, decision, n=2, effect_delta=effect, repetitions=64, seed=1414, mc_error=0.1
    )
    assert 0 < row.availability_uncertified < row.attempted
    assert row.known_runtime_refused == row.sampling_failed == 0
    assert row.rejected == row.attempted - row.availability_uncertified
    assert row.possible_rejected == row.attempted
    assert row.power is None and row.power_unavailable_reason == "numerical_failures"
    assert row.failures[0].reason == "descriptive_availability_uncertified"
    assert row.evaluation_scope == "affine_unit_sufficient_state"
    assert row.availability_method == "exact_reporting_certified_descriptive_v1"
    assert UnitCycleModelPowerResult.model_validate_json(row.model_dump_json()) == row
    bound = unit_cycle_power_lower_bound(decision.reference, decision, n=2, effect_delta=effect)
    assert bound.lower_bound == 0 and bound.reason == "runtime_availability_not_certified"


@pytest.mark.slow
@pytest.mark.parametrize("direction,alternative", [(1, "greater"), (-1, "less")])
def test_one_sided_availability_power_drop_preserves_public_mde_replay(direction, alternative):
    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    decision = procedure(model, alternative=alternative, null_abs=direction * 3)
    mde = unit_cycle_model_mde(
        model,
        decision,
        n=1,
        effect_grid=(
            decision.null_abs,
            math.nextafter(decision.null_abs, direction * math.inf),
            direction * math.nextafter(math.inf, 0),
        ),
        target_power=0.9,
        repetitions=256,
        seed=1414,
        mc_error=0.1,
    )
    assert mde.feasible_effect == math.nextafter(decision.null_abs, direction * math.inf)
    assert mde.diagnostic_effect == decision.null_abs and mde.failed == 0
    assert mde.feasible_effect is not None
    replay = unit_cycle_model_power(
        model,
        decision,
        n=1,
        effect_delta=mde.feasible_effect,
        repetitions=256,
        seed=1414,
        mc_error=mde.mc_error_per_effect,
        batch_size=7,
    )
    assert replay == mde.power_at_feasible
    assert replay.lower_bound >= mde.target_power
    large = unit_cycle_model_power(
        model,
        decision,
        n=1,
        effect_delta=direction * math.nextafter(math.inf, 0),
        repetitions=256,
        seed=1414,
        mc_error=0.1,
    )
    assert 0 < large.known_runtime_refused < large.attempted
    assert large.possible_rejected == large.rejected < replay.rejected
    assert large.power is None and large.lower_bound < replay.lower_bound
    assert large.lower_bound < mde.target_power
    assert UnitCycleModelMdeResult.model_validate_json(mde.model_dump_json()) == mde


@pytest.mark.slow
@pytest.mark.parametrize("direction,alternative", [(1, "greater"), (-1, "less")])
def test_mde_certificate_recomputes_failures_after_unavailable_null(direction, alternative):
    model = law(
        types=tuple(
            UnitCycleTypeLaw(
                weight=1, cycles=(cycle(ct_mean=u, tc_mean=u, ct_noise_load=0, tc_noise_load=0),)
            )
            for u in (-1, 1)
        )
    )
    null = -direction * math.nextafter(math.inf, 0)
    decision = procedure(model, alternative=alternative, null_abs=null)
    row = unit_cycle_model_mde(
        model,
        decision,
        n=1,
        effect_grid=(null, math.nextafter(null, direction * math.inf), 0.0),
        target_power=0.5,
        repetitions=256,
        seed=1414,
        mc_error=0.1,
    )
    assert row.diagnostic_effect == null
    assert row.failed == row.known_runtime_refused == 256
    assert row.feasible_effect == math.nextafter(null, direction * math.inf)
    assert row.feasible_effect is not None
    replay = unit_cycle_model_power(
        model,
        decision,
        n=1,
        effect_delta=row.feasible_effect,
        repetitions=256,
        seed=1414,
        mc_error=row.mc_error_per_effect,
        batch_size=13,
    )
    assert row.power_at_feasible == replay
    assert replay.failed == 0 and replay.rejected == replay.possible_rejected == 256


@pytest.mark.slow
def test_uncertified_descriptive_nonrejection_does_not_become_possible_rejection():
    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    row = unit_cycle_model_power(
        model,
        procedure(model, alternative="less"),
        n=2,
        effect_delta=1e200,
        repetitions=64,
        seed=1414,
        mc_error=0.1,
    )
    assert row.availability_uncertified > 0
    assert row.rejected == row.possible_rejected == row.known_runtime_refused == 0
    assert row.power is None and row.upper_bound < 1


@pytest.mark.slow
def test_required_n_recomputes_reporting_availability_and_replays_all_prefix_counts():
    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    decision = procedure(model, alternative="greater")
    effect = 0.6 * math.nextafter(math.inf, 0)
    row = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=effect,
        target_power=0.4,
        max_n=2,
        repetitions=256,
        seed=1414,
        mc_error=0.1,
        batch_size=13,
    )
    replays = [
        unit_cycle_model_power(
            model,
            decision,
            n=n,
            effect_delta=effect,
            repetitions=256,
            seed=1414,
            mc_error=row.mc_error_per_n,
            batch_size=7,
        )
        for n in (1, 2)
    ]
    assert replays[1].known_runtime_refused < replays[0].known_runtime_refused
    assert replays[1].availability_uncertified > 0
    assert row.known_runtime_refused_across_n == sum(x.known_runtime_refused for x in replays)
    assert row.availability_uncertified_across_n == sum(x.availability_uncertified for x in replays)
    assert row.failed_across_n == sum(x.failed for x in replays)
    assert row.feasible_n == 1 and row.power_at_feasible == replays[0]
    assert UnitCycleModelRequiredUnitsResult.model_validate_json(row.model_dump_json()) == row


def test_guaranteed_bound_does_not_certify_refused_max_reports():
    model = law(
        types=tuple(
            UnitCycleTypeLaw(
                weight=1, cycles=(cycle(ct_mean=u, tc_mean=u, ct_noise_load=0, tc_noise_load=0),)
            )
            for u in (-1, 1)
        )
    )
    decision = procedure(model)
    row = unit_cycle_power_lower_bound(
        decision.reference, decision, n=1, effect_delta=math.nextafter(math.inf, 0)
    )
    assert row.lower_bound == 0 and row.reason == "runtime_availability_not_certified"


def test_mde_rejects_noncanonical_effect_grid():
    model = law()
    decision = procedure(model)
    with pytest.raises(InvalidRequestError):
        unit_cycle_model_mde(
            model,
            decision,
            n=1,
            effect_grid=(0.5, 0.0),
            target_power=0.8,
            repetitions=8,
            seed=1,
            mc_error=0.1,
        )
    with pytest.raises(InvalidRequestError):
        unit_cycle_model_mde(
            model,
            decision,
            n=1,
            effect_grid=(0.0, 0.0),
            target_power=0.8,
            repetitions=8,
            seed=1,
            mc_error=0.1,
        )


@pytest.mark.parametrize("value", [None, math.inf])
def test_mde_invalid_grid_member_names_its_index_and_value(value):
    model = law()
    with pytest.raises(InvalidRequestError) as caught:
        unit_cycle_model_mde(
            model,
            procedure(model),
            n=1,
            effect_grid=(0.0, value),
            target_power=0.8,
            repetitions=8,
            seed=1,
            mc_error=0.1,
        )
    assert caught.value.code == "power.unit_cycle.input"
    assert caught.value.context["field"] == "effect_grid[1]"
    assert caught.value.context["value"] == value


def test_mde_grid_exhaustion_is_explicit_and_mutation_isolated():
    model = law()
    decision = procedure(model)
    grid = [0.0, 1e-12]
    result = unit_cycle_model_mde(
        model,
        decision,
        n=1,
        effect_grid=grid,
        target_power=0.999,
        repetitions=8,
        seed=1,
        mc_error=0.1,
    )
    grid.append(10.0)
    assert result.effect_grid == (0.0, 1e-12)
    assert result.search_status == "grid_exhausted"
    assert result.feasible_unavailable_reason == "grid_exhausted"


def _tamperable_mde_result():
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),)
    )
    return unit_cycle_model_mde(
        model,
        procedure(model),
        n=4,
        effect_grid=(0.0, 0.5, 2.0),
        target_power=0.8,
        repetitions=128,
        seed=5,
        mc_error=0.05,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"mc_error_per_effect": 0.05},
        {"effect_grid": (-1.0, 0.0, 0.5)},
        {"effect_grid": (0.0, 2.0, 3.0)},
        {"feasible_effect": 2.0},
        {"plausible_effect": 2.0},
        {"plausible_effect": None, "plausible_unavailable_reason": "grid_exhausted"},
        {"n": 8},
        {"seed": 6},
    ],
)
def test_mde_grid_result_rejects_tampered_selection_metadata(changes):
    payload = _tamperable_mde_result().model_dump()
    payload.update(changes)
    with pytest.raises(InvalidRequestError):
        UnitCycleModelMdeResult.model_validate(payload)


@pytest.mark.parametrize(
    "changes,field,value",
    [
        ({"seed": 6}, "power_at_feasible.seed/seed", (5, 6)),
        (
            {"n": 8, "seed": 6},
            "power_at_feasible.n/n/power_at_feasible.seed/seed",
            (4, 8, 5, 6),
        ),
        ({"effect_grid": (0.0, 0.5, 0.5)}, "effect_grid[2]", 0.5),
        ({"effect_grid": (-1.0, 0.0, 0.5)}, "effect_grid[0]/procedure.null_abs", (-1.0, 0.0)),
        (
            {"mc_error_per_effect": 0.05},
            "mc_error_per_effect/len(effect_grid)/mc_error",
            (0.05, 3, 0.05),
        ),
    ],
)
def test_mde_result_refusal_names_only_the_rejected_fields(changes, field, value):
    payload = _tamperable_mde_result().model_dump()
    payload.update(changes)
    with pytest.raises(InvalidRequestError) as error:
        UnitCycleModelMdeResult.model_validate(payload)
    assert error.value.code == "power.unit_cycle.result"
    assert error.value.context["field"] == field
    assert error.value.context["value"] == value


def test_mde_certificate_model_drift_is_reported_as_bounded_text():
    payload = _tamperable_mde_result().model_dump()
    payload["law"] = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    ).model_dump()
    with pytest.raises(InvalidRequestError) as error:
        UnitCycleModelMdeResult.model_validate(payload)
    assert error.value.context["field"] == "power_at_feasible.law/law"
    values = error.value.context["value"]
    assert isinstance(values, tuple)
    certificate_law, result_law = values
    for text in (certificate_law, result_law):
        assert isinstance(text, str) and len(text) <= 256
    restored = pickle.loads(pickle.dumps(error.value))
    assert restored.context == error.value.context


def test_required_n_checks_every_integer_and_matches_fixed_n_streams(monkeypatch):
    model = law(
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0.25, tc_noise_load=0.25),)),)
    )
    decision = procedure(model, alpha=0.25)
    monkeypatch.setattr(
        "increment.power.unit_cycle._unit",
        lambda *args: (Fraction(1, 2**57), Fraction(1), 1, None),
    )
    row = unit_cycle_model_required_units(
        model,
        decision,
        effect_delta=0.5,
        target_power=0.5,
        max_n=3,
        repetitions=128,
        seed=3,
        mc_error=0.1,
        batch_size=7,
    )
    assert row.checked_n == 3 and row.attempted_across_n == 3 * 128
    assert row.plausible_n == row.feasible_n == 2
    replays = [
        unit_cycle_model_power(
            model,
            decision,
            n=n,
            effect_delta=0.5,
            repetitions=128,
            seed=3,
            mc_error=row.mc_error_per_n,
        )
        for n in range(1, 4)
    ]
    assert [replay.rejected for replay in replays] == [0, 128, 128]
    assert replays[0].upper_bound < row.target_power
    assert all(replay.lower_bound >= row.target_power for replay in replays[1:])
    assert row.power_at_feasible == replays[1]
    assert UnitCycleModelRequiredUnitsResult.model_validate_json(row.model_dump_json()) == row


def test_monte_carlo_allocation_never_rounds_up_or_silently_underflows():
    each, radius = _band(10, math.ulp(0.0) * 3, 2)
    assert Fraction(each) * 2 <= Fraction(math.ulp(0.0) * 3)
    assert radius == 1
    with pytest.raises(InvalidRequestError) as error:
        _band(10, math.ulp(0.0), 2)
    assert error.value.code == "power.unit_cycle.numerical"
    assert error.value.context["field"] == "mc_error_allocation"
    assert error.value.context["value"] == (math.ulp(0.0), 2)


def test_innovation_numerical_failure_is_counted_not_redrawn():
    model = law(innovation=CenteredLognormalInnovation(shape=1e308))
    row = unit_cycle_model_power(
        model, procedure(model), n=1, effect_delta=1, repetitions=7, seed=12, mc_error=0.1
    )
    assert row.attempted == row.admitted == row.failed == 7
    assert row.sampling_failed == row.possible_rejected == 7
    assert row.rejected == row.known_runtime_refused == row.availability_uncertified == 0
    assert row.power is None and row.numerical_status == "incomplete"
    assert row.lower_bound == 0 and row.upper_bound == 1
    assert sum(x.count for x in row.failures) == 7
    assert '"power":null' in row.model_dump_json()


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_normal_useful_power_witness_has_independent_normal_tail_reference():
    from scipy.stats import norm

    model = law()
    decision = procedure(model)
    cutoff = unit_cycle_envelope_cutoff(
        unit_cycle_variance_envelope(model), n=40, alpha=0.05, alternative="two-sided"
    )
    cap = Fraction(math.nextafter(0.05, 0)) - Fraction(cutoff.refusal_probability_upper)
    boundary = math.sqrt(float(Fraction(1, 40) / cap))
    oracle = norm.sf((boundary - 1) * math.sqrt(40)) + norm.cdf((-boundary - 1) * math.sqrt(40))
    assert oracle >= 0.90
    row = unit_cycle_model_power(
        model, decision, n=40, effect_delta=1, repetitions=2048, seed=1414, mc_error=0.01
    )
    assert row.lower_bound >= 0.80
    assert row.lower_bound <= oracle <= row.upper_bound + cutoff.refusal_probability_upper


def test_existing_c11_rejects_envelope_instead_of_silently_using_nct():
    from increment.power.switchback import SwitchbackBaseline, switchback_achieved_power

    model = law()
    baseline = SwitchbackBaseline(
        assignment=model.assignment,
        metric=model.metric,
        control_group=model.control_group,
        treatment_group=model.treatment_group,
        aggregation="sum",
        estimand="retained_window_total_difference",
        cycles_per_unit=1,
        delta_ref=0,
        sd_a=1,
        sd_g=0,
        rho=0,
    )
    with pytest.raises(InvalidRequestError) as error:
        switchback_achieved_power(10, 1, baseline, procedure(model))
    assert error.value.code == "power.switchback.procedure"


@pytest.mark.parametrize(
    "effect,repetitions,target", [(1.0, 128, 0.1), (0.0, 1, 0.8), (1e200, 16, 0.1)]
)
def test_required_n_status_roundtrip_rejects_contradictory_serialized_status(
    effect, repetitions, target
):
    import json

    model = law(
        p=0.75,
        types=(UnitCycleTypeLaw(weight=1, cycles=(cycle(ct_noise_load=0, tc_noise_load=0),)),),
    )
    row = unit_cycle_model_required_units(
        model,
        procedure(model, alternative="greater"),
        effect_delta=effect,
        target_power=target,
        max_n=2,
        repetitions=repetitions,
        seed=1414,
        mc_error=0.1,
    )
    expected = (
        "numerical_incomplete" if effect == 1e200 else "bracketed" if effect else "search_limit"
    )
    assert row.search_status == expected
    assert UnitCycleModelRequiredUnitsResult.model_validate_json(row.model_dump_json()) == row
    for status in {"bracketed", "search_limit", "numerical_incomplete"} - {expected}:
        payload = row.model_dump(mode="json")
        payload["search_status"] = status
        with pytest.raises(InvalidRequestError):
            UnitCycleModelRequiredUnitsResult.model_validate_json(json.dumps(payload))
    payload = row.model_dump(mode="json")
    payload["feasible_unavailable_reason"] = "search_limit" if row.feasible_n else None
    with pytest.raises(InvalidRequestError):
        UnitCycleModelRequiredUnitsResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "reason,status",
    [
        ("unit_cycle.error_budget_exhausted", "search_limit"),
        ("unit_cycle.numerical", "numerical_incomplete"),
    ],
)
def test_required_n_serialized_unavailable_reason_controls_status(reason, status):
    import json

    model = law()
    row = unit_cycle_model_required_units(
        model,
        procedure(model),
        effect_delta=0,
        target_power=0.8,
        max_n=1,
        repetitions=1,
        seed=1,
        mc_error=0.1,
    )
    payload = row.model_dump(mode="json")
    payload.update(
        plausible_n=None,
        plausible_unavailable_reason="search_limit",
        feasible_n=None,
        feasible_unavailable_reason="search_limit",
        power_at_feasible=None,
        unavailable_n=1,
        unavailable_reasons=[{"reason": reason, "count": 1}],
        attempted_across_n=0,
        admitted_across_n=0,
        search_status=status,
    )
    restored = UnitCycleModelRequiredUnitsResult.model_validate_json(json.dumps(payload))
    assert (
        UnitCycleModelRequiredUnitsResult.model_validate_json(restored.model_dump_json())
        == restored
    )
    payload["search_status"] = (
        "search_limit" if status == "numerical_incomplete" else "numerical_incomplete"
    )
    with pytest.raises(InvalidRequestError):
        UnitCycleModelRequiredUnitsResult.model_validate_json(json.dumps(payload))
