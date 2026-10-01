"""Automatic exact Bernoulli registration from a one-kwarg inference declaration."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from increment.errors import DefinitionError
from increment.frame import MetricSpec, synthesise_metric
from increment.semantics.models import AnalysisPlan, InferenceSpec
from tests.binary_sequential_cases import (
    DESIGN,
    check_path_parity,
    event_rows,
    unit_rows,
)


@pytest.fixture(scope="module")
def duck():
    import ibis

    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


def _bound(specs, inference, *, design=DESIGN, **plan):
    from increment.plan import bind_automatic_sequential_plan
    from increment.sequential_source import frame_observation_mapping

    bound = bind_automatic_sequential_plan(
        AnalysisPlan(primary=specs[0].name, inference=inference, **plan),
        [synthesise_metric(s) for s in specs],
        design=design,
        source_id="experiment",
        source_mapping=frame_observation_mapping(unit="unit", group="arm"),
        transformations=specs,
        path="frame",
    )
    assert bound is not None and bound.inference is not None
    return bound.inference


def _manual(specs, roster, *, design=DESIGN, baseline_rate=None, q=F(0.1)):
    """The explicit registration a caller would write for these cells."""
    from increment import JointReveal, SequentialModel, SequentialRegistration
    from increment.sequential_source import (
        bernoulli_prior,
        frame_observation_mapping,
        sequential_definition_id,
    )
    from increment.sequential_state import canonical_id

    prior = bernoulli_prior(baseline_rate)
    binding = sequential_definition_id(
        [synthesise_metric(s) for s in specs],
        design,
        source_mapping=frame_observation_mapping(unit="unit", group="arm"),
        transformations=specs,
    )
    return SequentialRegistration(
        source_id="experiment",
        definitions_id=binding,
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id=canonical_id({"binding": binding, "reveal": "joint_units_v1"}),
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=0,
        ),
        models=tuple(
            SequentialModel(
                metric=s.name,
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            )
            for s in specs
        ),
        roster=roster,
        q=q,
    )


# Identities bound by the release before predeclared segment metadata existed; a
# checkpoint stored then must still continue under a plan that declares none.
_TWO_ARM_IDENTITY = "f4438d788d1ff43f7342aac10178a499c69a95875dfa912610568fbd25e6beef"
_FAMILY_IDENTITY = "8e0e553fd583b8e88a67012cb7c65d652d9cdb7955e9e97e9433d50282c9e0fa"


def test_segment_free_plans_keep_their_stored_registration_identity():
    from increment.sequential_source import registration_id

    specs = [MetricSpec(name="outcome", type="conversion")]
    two_arm = _bound(specs, InferenceSpec(kind="always_valid", baseline_rate=F(3, 10)))
    assert registration_id(two_arm.registration) == _TWO_ARM_IDENTITY
    family = _bound(
        [MetricSpec(name=n, type="conversion") for n in ("checkout", "signups", "sessions")],
        InferenceSpec(kind="always_valid"),
        secondaries=["signups", "sessions"],
        q=0.08,
    )
    assert registration_id(family.registration) == _FAMILY_IDENTITY
    spec = InferenceSpec(kind="always_valid", baseline_rate=F(3, 10))
    assert "segments" not in spec.model_dump()
    assert InferenceSpec.model_validate_json(spec.model_dump_json()) == spec


def test_always_valid_multi_arm_registers_every_arm_at_its_role_allocation():
    from increment import SequentialCell
    from increment.semantics.design import Randomized
    from increment.sequential_source import registration_id

    specs = [
        MetricSpec(name="checkout", type="conversion"),
        MetricSpec(name="signups", type="conversion"),
        MetricSpec(name="sessions", type="conversion"),
        MetricSpec(name="refunds", type="conversion", preferred_direction="decrease"),
    ]
    design = Randomized(control_group="control", allocation={"control": 0.4, "a": 0.3, "b": 0.3})
    inference = _bound(
        specs,
        InferenceSpec(kind="always_valid"),
        design=design,
        secondaries=["signups", "sessions"],
        guardrails=["refunds"],
        q=0.08,
    )
    assert inference.segments == {}
    cells = {(c.metric, c.group_id): c for c in inference.registration.roster}
    assert set(cells) == {(s.name, arm) for s in specs for arm in ("a", "b")}
    for arm in ("a", "b"):
        primary = cells["checkout", arm]
        assert primary.alpha == F(0.05) / 2 and primary.family is False
        for name in ("signups", "sessions"):
            assert cells[name, arm].family is True
            assert cells[name, arm].alpha == F(0.08) / 4
        guardrail = cells["refunds", arm]
        assert guardrail.alpha == F(0.05) and guardrail.family is False
        assert guardrail.alternative == "less"
    manual = _manual(
        specs,
        tuple(
            SequentialCell(
                metric=s.name,
                group_id=arm,
                alpha=cells[s.name, arm].alpha,
                family=cells[s.name, arm].family,
                alternative=cells[s.name, arm].alternative,
            )
            for s in specs
            for arm in ("a", "b")
        ),
        design=design,
        q=F(0.08),
    )
    assert registration_id(manual) == registration_id(inference.registration)


def test_always_valid_multi_arm_uptake_cells_share_the_compliance_allocation():
    from increment import SequentialCompliancePolicy
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="the prompt affects conversion only through clicks"
        ),
        allocation={"control": 0.5, "a": 0.25, "b": 0.25},
    )
    specs = [
        MetricSpec(name="converted", type="conversion"),
        MetricSpec(name="s", type="conversion"),
    ]
    inference = _bound(
        specs,
        InferenceSpec(kind="always_valid"),
        design=design,
        secondaries=["s"],
        compliance=SequentialCompliancePolicy(alpha=F(1, 20), family=True),
    )
    roster = inference.registration.roster
    uptake = {c.group_id: c for c in roster if c.estimand == "compliance"}
    assert set(uptake) == {"a", "b"}
    # Four family cells (one secondary and the uptake cell, on two arms) share q;
    # the uptake cells keep the tighter compliance share alpha / n_arms.
    assert all(c.alpha == F(1, 40) and c.family for c in uptake.values())
    assert all(c.alpha == F(0.1) / 4 and c.family for c in roster if c.metric == "s")
    assert sum(c.alpha for c in roster if c.family) <= inference.registration.q


@pytest.mark.parametrize("kind", ["always_valid", "asymptotic_mean"])
def test_segmented_family_compliance_refuses_before_automatic_binding(kind):
    import copy
    import pickle

    from increment import SequentialCompliancePolicy
    from increment.errors import CapabilityError

    with pytest.raises(CapabilityError) as raised:
        AnalysisPlan(
            primary="converted",
            inference=InferenceSpec(kind=kind, segments={"country": ("US", "CA")}),
            compliance=SequentialCompliancePolicy(alpha=F(1, 20), family=True),
        )

    assert raised.value.code == "sequential.compliance.segmented_family_unsupported"
    assert raised.value.context["segments"] == (("country", ("US", "CA")),)
    assert raised.value.context["family"] is True
    for error in (
        raised.value,
        copy.deepcopy(raised.value),
        pickle.loads(pickle.dumps(raised.value)),
    ):
        assert error.code == raised.value.code
        assert error.context == raised.value.context


def test_always_valid_segments_fix_one_cell_per_metric_arm_and_level_before_data():
    from increment import SequentialCell
    from increment.sequential_source import registration_id

    specs = [
        MetricSpec(name="outcome", type="conversion"),
        MetricSpec(name="other", type="conversion"),
    ]
    levels = ("US", "CA", "GB")
    inference = _bound(
        specs,
        InferenceSpec(kind="always_valid", segments={"country": levels}),
        secondaries=["other"],
    )
    assert inference.segments == {}
    roster = inference.registration.roster
    assert {c.segment for c in roster} == {(("country", level),) for level in levels}
    assert {(c.metric, c.group_id) for c in roster} == {(s.name, "treatment") for s in specs}
    # The default randomized breakout family is e-BH over every metric x arm x level cell.
    assert all(c.family and c.alpha == F(0.1) / 6 for c in roster)
    manual = _manual(
        specs,
        tuple(
            SequentialCell(
                metric=s.name,
                group_id="treatment",
                segment=(("country", level),),
                family=True,
                alpha=F(0.1) / 6,
            )
            for s in specs
            for level in levels
        ),
    )
    assert registration_id(manual) == registration_id(inference.registration)


@pytest.mark.parametrize(
    ("correction", "family", "alpha"),
    [("bonferroni", False, F(0.05) / 3), ("none", False, F(0.05))],
)
def test_always_valid_segments_follow_the_declared_breakout_correction(correction, family, alpha):
    from increment import MultiplicitySpec

    specs = [MetricSpec(name="outcome", type="conversion")]
    inference = _bound(
        specs,
        InferenceSpec(kind="always_valid", segments={"country": ("US", "CA", "GB")}),
        view_multiplicity=MultiplicitySpec(correction=correction),
    )
    assert all(c.family is family and c.alpha == alpha for c in inference.registration.roster)


def test_always_valid_segments_carry_the_declared_bh_view_level():
    """An explicit BH breakout view fixes its own e-BH level, and the readout
    requires the registration to carry that level: the automatic roster is
    allocated and registered at the view's q (never above each cell's compiled
    level), the same registration a caller writes by hand at that q."""
    from increment import MultiplicitySpec, SequentialCell
    from increment.sequential_source import registration_id

    specs = [
        MetricSpec(name="outcome", type="conversion"),
        MetricSpec(name="other", type="conversion"),
    ]
    levels = ("US", "CA", "GB")
    inference = _bound(
        specs,
        InferenceSpec(kind="always_valid", segments={"country": levels}),
        secondaries=["other"],
        view_multiplicity=MultiplicitySpec(correction="bh", q=0.24),
    )
    registration = inference.registration
    assert registration.q == F(0.24)
    # Six cells share the view's 0.24: 0.04 each, below both compiled levels.
    assert all(c.family and c.alpha == F(0.24) / 6 for c in registration.roster)
    manual = _manual(
        specs,
        tuple(
            SequentialCell(
                metric=s.name,
                group_id="treatment",
                segment=(("country", level),),
                family=True,
                alpha=F(0.24) / 6,
            )
            for s in specs
            for level in levels
        ),
        q=F(0.24),
    )
    assert registration_id(manual) == registration_id(registration)


def test_segment_metadata_belongs_to_automatic_registration_only():
    import json

    from increment.errors import InvalidRequestError
    from tests.sequential_cases import registration

    with pytest.raises(InvalidRequestError) as raised:
        InferenceSpec(
            kind="always_valid", registration=registration(), segments={"country": ("US",)}
        )
    assert raised.value.code == "sequential.registration.invalid"
    # A stored explicit registration that later grows the metadata refuses on replay too.
    payload = InferenceSpec(kind="always_valid", registration=registration()).model_dump(
        mode="json"
    )
    payload["segments"] = {"country": ["US"]}
    with pytest.raises(InvalidRequestError) as replayed:
        InferenceSpec.model_validate_json(json.dumps(payload))
    assert replayed.value.code == "sequential.registration.invalid"


@pytest.mark.parametrize(
    "segments",
    [
        {"country": ("US", "CA"), "plan": ("pro",)},
        {"country": ()},
        {"country": ("US", "US")},
        {"": ("US",)},
        {"country": ("",)},
    ],
)
def test_segment_metadata_names_one_dimension_with_distinct_levels(segments):
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        InferenceSpec(kind="always_valid", segments=segments)
    assert raised.value.code == "sequential.registration.invalid"


def test_segment_levels_are_labels_never_collapsed_from_native_values():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        InferenceSpec.model_validate({"kind": "always_valid", "segments": {"tier": (1, "1")}})
    levels = ["US", "CA"]
    spec = InferenceSpec(kind="always_valid", segments={"country": levels})
    levels.append("GB")
    assert spec.segments["country"] == ("US", "CA")
    assert InferenceSpec.model_validate_json(spec.model_dump_json()) == spec


def test_always_valid_conversion_registers_a_stable_bernoulli_registration():
    from increment.sequential_source import DEFAULT_BERNOULLI_PRIOR_WEIGHT, registration_id

    specs = [MetricSpec(name="outcome", type="conversion")]
    first = _bound(specs, InferenceSpec(kind="always_valid", baseline_rate=F(3, 10)))
    second = _bound(specs, InferenceSpec(kind="always_valid", baseline_rate=F(3, 10)))
    other = _bound(specs, InferenceSpec(kind="always_valid", baseline_rate=F(2, 5)))
    flat = _bound(specs, InferenceSpec(kind="always_valid"))
    assert first.baseline_rate is None and first.registration is not None
    assert registration_id(first.registration) == registration_id(second.registration)
    assert registration_id(first.registration) != registration_id(other.registration)
    assert registration_id(first.registration) != registration_id(flat.registration)
    model = first.registration.models[0]
    assert model.law == "bernoulli"
    weight = F(DEFAULT_BERNOULLI_PRIOR_WEIGHT)
    assert model.control_prior.a == weight * F(3, 10)
    assert model.control_prior.b == weight * F(7, 10)
    assert model.treatment_prior == model.control_prior
    assert flat.registration.models[0].control_prior.a == 1
    assert flat.registration.models[0].control_prior.b == 1
    cell = first.registration.roster[0]
    assert cell.alpha == F(0.05) and cell.alternative == "two-sided" and cell.family is False


def test_always_valid_mean_metric_without_registration_refuses_by_name():
    specs = [MetricSpec(name="revenue", type="mean")]
    with pytest.raises(DefinitionError) as raised:
        _bound(specs, InferenceSpec(kind="always_valid"))
    assert raised.value.code == "definition.inference.always_valid_metric_type"
    assert raised.value.context["metric"] == "revenue"
    assert raised.value.context["metric_type"] == "mean"


def test_always_valid_family_allocation_matches_the_scalar_route():
    specs = [MetricSpec(name=n, type="conversion") for n in ("checkout", "signups", "sessions")]
    inference = _bound(
        specs, InferenceSpec(kind="always_valid"), secondaries=["signups", "sessions"], q=0.08
    )
    assert inference.registration is not None
    cells = {c.metric: c for c in inference.registration.roster}
    assert cells["checkout"].alpha == F(0.05) and cells["checkout"].family is False
    for name in ("signups", "sessions"):
        assert cells[name].family is True
        assert cells[name].alpha == F(0.08) / 2


def test_end_to_end_binary_always_valid_returns_an_interval_and_a_decision():
    import pandas as pd

    from increment import Analysis

    frame = pd.DataFrame(
        [
            {"unit": f"{i:08d}-{arm}", "arm": arm, "outcome": value, "exposure": i}
            for i, pair in enumerate(zip([0, 0, 0, 1] * 50, [0, 1, 1, 1] * 50, strict=True))
            for arm, value in zip(("control", "treatment"), pair, strict=True)
        ]
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        exposure_date="exposure",
        metrics=[MetricSpec(name="outcome", type="conversion")],
        design=DESIGN,
        experiment_id="experiment",
        plan=AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", baseline_rate=F(3, 10)),
        ),
    )
    row = analysis.run()[0]
    assert row.inference == "always_valid"
    lift = row.require_lift()
    assert lift.lb is not None and lift.lb > 0
    assert row.stat_sig()
    assert row.require_sequential_result().checkpoint.model.law == "bernoulli"


def test_binary_always_valid_state_and_interval_agree_on_every_path(duck):
    check_path_parity(duck, "duckdb", "events", "always_valid")
