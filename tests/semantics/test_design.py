import pytest
from pydantic import TypeAdapter, ValidationError

from increment.semantics.design import (
    AdjustmentSet,
    Design,
    Encouragement,
    ExclusionRestriction,
    IdentificationGate,
    Observational,
    Randomized,
    UptakeSpec,
)


def test_randomized_defaults():
    r = Randomized(control_group="control")
    assert r.mechanism == "randomized"
    assert r.control_group == "control"


def test_randomized_frozen_mutation_raises():
    r = Randomized(control_group="control")
    with pytest.raises(ValidationError):
        r.control_group = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_adjustment_set_requires_at_least_one_covariate():
    with pytest.raises(ValidationError):
        AdjustmentSet(covariates=())


def test_adjustment_set_accepts_covariates():
    a = AdjustmentSet(covariates=("age", "region"))
    assert a.covariates == ("age", "region")


def test_adjustment_set_frozen_mutation_raises():
    a = AdjustmentSet(covariates=("age",))
    with pytest.raises(ValidationError):
        a.covariates = ("region",)  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_identification_gate_defaults():
    g = IdentificationGate()
    assert g.overlap == "refuse"
    assert g.min_propensity == 0.01
    assert g.max_smd is None


def test_identification_gate_min_propensity_bounds():
    with pytest.raises(ValidationError):
        IdentificationGate(min_propensity=0.0)
    with pytest.raises(ValidationError):
        IdentificationGate(min_propensity=0.5)


def test_identification_gate_max_smd_rejects_negative():
    """max_smd is a magnitude (>= 0): a negative threshold would silently force
    refuse-always, exactly the misconfiguration this gate catches at declaration time."""
    with pytest.raises(ValidationError):
        IdentificationGate(max_smd=-0.1)


def test_identification_gate_max_smd_allows_zero():
    """0.0 is a legitimate (if strict) threshold: refuse on any imbalance."""
    g = IdentificationGate(max_smd=0.0)
    assert g.max_smd == 0.0


def test_identification_gate_frozen_mutation_raises():
    g = IdentificationGate()
    with pytest.raises(ValidationError):
        g.overlap = "trim"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_observational_defaults_gate():
    o = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("age",)),
    )
    assert o.mechanism == "observational"
    assert o.control_group == "control"
    assert o.adjustment.covariates == ("age",)
    assert o.gate == IdentificationGate()


def test_observational_frozen_mutation_raises():
    o = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("age",)),
    )
    with pytest.raises(ValidationError):
        o.control_group = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_design_discriminated_union_resolves_randomized():
    result = TypeAdapter(Design).validate_python(
        {"mechanism": "randomized", "control_group": "control"}
    )
    assert isinstance(result, Randomized)
    assert result.control_group == "control"


def test_design_discriminated_union_resolves_observational():
    result = TypeAdapter(Design).validate_python(
        {
            "mechanism": "observational",
            "control_group": "control",
            "adjustment": {"covariates": ["age", "region"]},
        }
    )
    assert isinstance(result, Observational)
    assert result.control_group == "control"
    assert result.adjustment.covariates == ("age", "region")
    assert result.gate == IdentificationGate()


def test_design_discriminated_union_rejects_unknown_mechanism():
    with pytest.raises(ValidationError):
        TypeAdapter(Design).validate_python(
            {"mechanism": "quasi_experimental", "control_group": "control"}
        )


def _enc(**over):
    base = {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "help_click"},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "An unclicked button is inert.",
        },
    }
    base.update(over)
    return base


def test_acknowledged_false_rejected():
    with pytest.raises(ValidationError):
        ExclusionRestriction(acknowledged=False, justification="x" * 20)  # ty: ignore[invalid-argument-type]  # proving Literal[True] at runtime


def test_justification_must_be_substantive():
    with pytest.raises(ValidationError):
        ExclusionRestriction(acknowledged=True, justification="ok")


def test_uptake_window_days_must_be_positive():
    with pytest.raises(ValidationError):
        UptakeSpec(fact="help_click", window_days=0)


def test_min_first_stage_z_floor_positive():
    with pytest.raises(ValidationError):
        TypeAdapter(Design).validate_python(_enc(min_first_stage_z=-1.0))


def test_observational_rejects_unknown_field():
    """A typo'd identification field (`gaet` for `gate`) must refuse: silently dropping it
    would discard the user's explicit overlap policy and apply the default instead."""
    with pytest.raises(ValidationError):
        Observational(
            control_group="c",
            adjustment=AdjustmentSet(covariates=("age",)),
            gaet=IdentificationGate(overlap="trim"),  # ty: ignore[unknown-argument]  # proving extra=forbid at runtime
        )


def test_randomized_rejects_unknown_field():
    """Typo'd fields must refuse loudly, not be silently swallowed; allocation is a real
    field consumed as srm()'s default expected shares."""
    with pytest.raises(ValidationError):
        Randomized(control_group="control", alocation={"control": 0.5, "treat": 0.5})  # ty: ignore[unknown-argument]
    design = Randomized(control_group="control", allocation={"control": 0.5, "treat": 0.5})
    assert design.allocation == {"control": 0.5, "treat": 0.5}


@pytest.mark.parametrize(
    "build",
    [
        lambda: Randomized(control_group="c", bogus=1),  # ty: ignore[unknown-argument]
        lambda: UptakeSpec(fact="help_click", bogus=1),  # ty: ignore[unknown-argument]
        lambda: ExclusionRestriction(
            acknowledged=True,
            justification="x" * 20,
            bogus=1,  # ty: ignore[unknown-argument]
        ),
        lambda: Encouragement(
            control_group="c",
            uptake=UptakeSpec(fact="help_click"),
            exclusion_restriction=ExclusionRestriction(acknowledged=True, justification="x" * 20),
            bogus=1,  # ty: ignore[unknown-argument]
        ),
        lambda: AdjustmentSet(covariates=("age",), bogus=1),  # ty: ignore[unknown-argument]
        lambda: IdentificationGate(bogus=1),  # ty: ignore[unknown-argument]
        lambda: Observational(
            control_group="c",
            adjustment=AdjustmentSet(covariates=("age",)),
            bogus=1,  # ty: ignore[unknown-argument]
        ),
    ],
    ids=[
        "Randomized",
        "UptakeSpec",
        "ExclusionRestriction",
        "Encouragement",
        "AdjustmentSet",
        "IdentificationGate",
        "Observational",
    ],
)
def test_every_design_model_rejects_unknown_fields(build):
    """The design layer holds the causal assumptions: every model in it must forbid unknown
    fields, matching semantics.models._Base."""
    with pytest.raises(ValidationError):
        build()


def test_adjustment_set_duplicate_covariates_refused():
    """A duplicated covariate enters the design matrix twice downstream
    (rank deficiency or silently doubled weight, learner-dependent)."""
    with pytest.raises(ValidationError) as exc_info:
        AdjustmentSet(covariates=("age", "age", "region"))
    inner = exc_info.value.errors()[0]["ctx"]["error"]
    assert inner.code == "definition.adjustment_set.adjustmentset_covariates_contains"


def test_adjustment_set_distinct_covariates_pass():
    a = AdjustmentSet(covariates=("age", "region"))
    assert a.covariates == ("age", "region")


def test_randomized_allocation_immutable_after_construction():
    """A validated allocation is a snapshot: mutating the dict the caller
    passed in, or the returned mapping, must never change it post hoc --
    otherwise SRM checks silently drift from what was declared."""
    r = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
    with pytest.raises(TypeError):
        r.allocation["control"] = 0.9  # ty: ignore[invalid-assignment]
    assert r.allocation == {"control": 0.5, "treatment": 0.5}


def test_randomized_allocation_snapshots_caller_dict():
    caller_dict = {"control": 0.5, "treatment": 0.5}
    r = Randomized(control_group="control", allocation=caller_dict)
    caller_dict["control"] = 0.9
    assert r.allocation == {"control": 0.5, "treatment": 0.5}


def test_encouragement_allocation_immutable_after_construction():
    d = TypeAdapter(Design).validate_python(_enc(allocation={"control": 0.5, "treatment": 0.5}))
    with pytest.raises(TypeError):
        d.allocation["control"] = 0.9


def test_encouragement_min_first_stage_z_rejects_infinity():
    with pytest.raises(ValidationError):
        TypeAdapter(Design).validate_python(_enc(min_first_stage_z=float("inf")))


def test_identification_gate_max_smd_rejects_infinity():
    with pytest.raises(ValidationError):
        IdentificationGate(max_smd=float("inf"))


def test_uptake_window_days_rejects_bool():
    """bool is an int subclass; True/False must never silently become 1/0."""
    with pytest.raises(ValidationError) as exc_info:
        UptakeSpec(fact="help_click", window_days=True)
    inner = exc_info.value.errors()[0]["ctx"]["error"]
    assert inner.code == "definition.assignment.reject_bool"


def test_reject_bool_code_is_shared_across_assignment_and_design():
    """The same bool guard is reached through both model entry points."""
    from increment.semantics.assignment import SwitchbackWindow

    with pytest.raises(ValidationError) as via_assignment:
        SwitchbackWindow(washout_steps=True, observation_steps=1)
    with pytest.raises(ValidationError) as via_design:
        UptakeSpec(fact="help_click", window_days=True)

    assignment_code = via_assignment.value.errors()[0]["ctx"]["error"].code
    design_code = via_design.value.errors()[0]["ctx"]["error"].code
    assert assignment_code == design_code == "definition.assignment.reject_bool"


@pytest.mark.parametrize(
    "allocation",
    [
        {},
        {"other": 0.5},
        {"control": -0.1, "other": 1.1},
        {"control": 0.0, "other": 0.0},
        {"control": 0.0, "other": 1.0},
        {"control": 1.0, "other": 0.0},
    ],
)
def test_randomized_allocation_refuses(allocation):
    with pytest.raises(ValidationError) as exc_info:
        Randomized(control_group="control", allocation=allocation)
    inner = exc_info.value.errors()[0]["ctx"]["error"]
    assert inner.code == "design.allocation.invalid"


def test_randomized_allocation_need_not_sum_to_one():
    r = Randomized(control_group="control", allocation={"control": 0.3, "other": 0.3})
    assert r.allocation == {"control": 0.3, "other": 0.3}


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_randomized_allocation_refuses_non_finite_weights(bad):
    with pytest.raises(ValidationError) as exc_info:
        Randomized(control_group="control", allocation={"control": bad, "t": 1.0})
    assert exc_info.value.errors()[0]["ctx"]["error"].code == "design.allocation.invalid"


def test_randomized_hash_equal_for_equal_designs_and_distinct_across_allocations():
    a = Randomized(control_group="control", allocation={"control": 0.5, "t": 0.5})
    b = Randomized(control_group="control", allocation={"control": 0.5, "t": 0.5})
    c = Randomized(control_group="control", allocation={"control": 0.3, "t": 0.7})
    assert hash(a) == hash(b)
    assert a == b
    assert hash(a) != hash(c)


def test_randomized_hashes_regardless_of_allocation():
    r = Randomized(control_group="control", allocation={"control": 0.5, "other": 0.5})
    assert isinstance(hash(r), int)
    r2 = Randomized(control_group="control")
    assert isinstance(hash(r2), int)


def test_encouragement_allocation_refuses():
    with pytest.raises(ValidationError) as exc_info:
        Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="took_up"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="pre-registered design"
            ),
            allocation={},
        )
    inner = exc_info.value.errors()[0]["ctx"]["error"]
    assert inner.code == "design.allocation.invalid"


def test_encouragement_hashes_regardless_of_allocation():
    e = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="took_up"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="pre-registered design"
        ),
        allocation={"control": 0.5, "other": 0.5},
    )
    assert isinstance(hash(e), int)
