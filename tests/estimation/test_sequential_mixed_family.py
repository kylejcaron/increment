"""Continuous intent-to-treat composed with Bernoulli uptake compliance.

Registration-level mixing (``SequentialRegistration._complete``) and the
internal ``MixedFamily`` runtime policy that composes both laws
automatically under ``InferenceSpec(kind="asymptotic_mean")``. There is no
public ``kind="mixed_family"``: a user declares ``inference: {kind:
asymptotic_mean}`` plus ``AnalysisPlan.compliance`` and an encouragement
design, and the library composes the per-cell laws itself.
"""

from fractions import Fraction as F

import pytest

from increment import (
    JointReveal,
    PredictivePrior,
    SequentialCell,
    SequentialModel,
    SequentialRegistration,
    capture_sequential_snapshot,
    estimate_sequential,
)
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily
from increment.frame import MetricSpec, synthesise_metric
from increment.plan import bind_automatic_sequential_plan, compile_decision_plan
from increment.semantics.design import Encouragement, ExclusionRestriction, Randomized, UptakeSpec
from increment.semantics.models import AnalysisPlan, InferenceSpec
from increment.semantics.sequential import SequentialCompliancePolicy
from increment.sequential_source import frame_observation_mapping
from tests.asymptotic_cases import mean_model

_ENCOURAGEMENT = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="clicked"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="prompt affects revenue only through clicks"
    ),
    allocation={"control": 0.5, "treatment": 0.5},
)


def test_automatic_asymptotic_registration_accepts_an_encouragement_design():
    """The automatic route (from_definitions, unit-day artifact, frames) reaches
    auto_register_scalar_mean before validate_scalar_mean_design; both must admit
    encouragement, or the primary ingress path refuses what the explicit path allows."""
    specs = [MetricSpec(name="revenue", type="mean")]
    bound = bind_automatic_sequential_plan(
        AnalysisPlan(primary="revenue", inference=InferenceSpec(kind="asymptotic_mean")),
        [synthesise_metric(s) for s in specs],
        design=_ENCOURAGEMENT,
        source_id="frame",
        source_mapping=frame_observation_mapping(unit="unit", group="arm", uptake="clicked"),
        transformations=specs,
        path="frame",
    )
    assert bound is not None and bound.inference is not None
    registration = bound.inference.registration
    assert registration is not None
    # Admission means the declared metric carries a scalar-mean sampling model
    # and a treatment cell; the roster's exact contents stay unpinned (an
    # uptake cell is free to join it).
    assert any(m.metric == "revenue" and m.law == "scalar_mean" for m in registration.models)
    assert any(c.metric == "revenue" and c.group_id == "treatment" for c in registration.roster)


# ── Registration-level mixing ───────────────────────────────────────────


def _uptake_model() -> SequentialModel:
    prior = PredictivePrior(kind="beta", a=1, b=1)
    return SequentialModel(
        metric="uptake",
        observable="uptake",
        law="bernoulli",
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
    )


def _mixed_registration(*, itt_alpha=F(1, 40), uptake_alpha=F(1, 40)) -> SequentialRegistration:
    return SequentialRegistration(
        source_id="experiment",
        definitions_id="stationary-unit-mean-v1",
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id="finalized-iid-units-v1",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=14,
        ),
        models=(mean_model("rev", start_count=4), _uptake_model()),
        roster=(
            SequentialCell(metric="rev", group_id="treatment", family=True, alpha=itt_alpha),
            SequentialCell(
                metric="uptake",
                group_id="treatment",
                estimand="compliance",
                family=True,
                alpha=uptake_alpha,
            ),
        ),
        q=F(1, 10),
    )


def test_registration_permits_scalar_mean_plus_bernoulli_uptake():
    reg = _mixed_registration()
    assert {m.law for m in reg.models} == {"scalar_mean", "bernoulli"}
    assert {c.metric for c in reg.roster} == {"rev", "uptake"}


def test_registration_still_refuses_scalar_mean_plus_private_gaussian():
    prior = PredictivePrior(kind="nig", kappa=1, nu=2, mean=(0,), scale=((2,),))
    gaussian_model = SequentialModel(
        metric="other",
        law="gaussian",
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
    )
    with pytest.raises(InvalidRequestError) as raised:
        SequentialRegistration(
            source_id="experiment",
            definitions_id="stationary-unit-mean-v1",
            control_group="control",
            committed_before_data=True,
            reveal=JointReveal(
                filtration_id="finalized-iid-units-v1",
                independent_unit_vectors=True,
                simultaneous_metrics=True,
                outcome_independent_order=True,
                immutable_finalized_outcomes=True,
                longest_window_days=14,
            ),
            models=(mean_model("rev", start_count=4), gaussian_model),
            roster=(
                SequentialCell(metric="rev", group_id="treatment"),
                SequentialCell(metric="other", group_id="treatment"),
            ),
        )
    assert raised.value.code == "sequential.registration.invalid"


def test_registration_refuses_a_bernoulli_outcome_mixed_with_scalar_mean():
    prior = PredictivePrior(kind="beta", a=1, b=1)
    bernoulli_outcome = SequentialModel(
        metric="converted",
        observable="outcome",
        law="bernoulli",
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
    )
    with pytest.raises(InvalidRequestError) as raised:
        SequentialRegistration(
            source_id="experiment",
            definitions_id="stationary-unit-mean-v1",
            control_group="control",
            committed_before_data=True,
            reveal=JointReveal(
                filtration_id="finalized-iid-units-v1",
                independent_unit_vectors=True,
                simultaneous_metrics=True,
                outcome_independent_order=True,
                immutable_finalized_outcomes=True,
                longest_window_days=14,
            ),
            models=(mean_model("rev", start_count=4), bernoulli_outcome),
            roster=(
                SequentialCell(metric="rev", group_id="treatment"),
                SequentialCell(metric="converted", group_id="treatment"),
            ),
        )
    assert raised.value.code == "sequential.registration.invalid"


def test_registration_alpha_budget_still_bounded_by_q_across_both_cells():
    with pytest.raises(InvalidRequestError) as raised:
        _mixed_registration(itt_alpha=F(1, 15), uptake_alpha=F(1, 15))
    assert raised.value.code == "sequential.registration.invalid"


# ── Inference-level composition ─────────────────────────────────────────


def test_always_valid_refuses_a_mixed_registration_with_the_construction_code():
    reg = _mixed_registration()
    with pytest.raises(CapabilityError) as raised:
        AlwaysValid(registration=reg)
    assert raised.value.code == "sequential.registration.mixed_requires_asymptotic_mean"


def test_always_valid_refuses_a_pure_asymptotic_registration_by_routing_to_asymptotic_mean():
    """The misnamed-code fix: a registration with NO Bernoulli cell at all gets
    the plain routing refusal, not the mixed-family code."""
    from tests.asymptotic_cases import mean_registration

    reg = mean_registration()
    with pytest.raises(InvalidRequestError) as raised:
        AlwaysValid(registration=reg)
    assert raised.value.code == "sequential.registration.invalid"


def test_asymptotic_mean_refuses_a_mixed_registration():
    reg = _mixed_registration()
    with pytest.raises(InvalidRequestError) as raised:
        AsymptoticMean(registration=reg)
    assert raised.value.code == "sequential.registration.invalid"


def test_mixed_family_requires_both_an_asymptotic_and_a_bernoulli_model():
    from tests.asymptotic_cases import mean_registration

    pure_scalar = mean_registration()
    with pytest.raises(InvalidRequestError) as raised:
        MixedFamily(registration=pure_scalar)
    assert raised.value.code == "sequential.registration.invalid"


def test_build_inference_composes_mixed_family_from_a_public_asymptotic_mean_kind():
    """No public kind exists for this: an ordinary InferenceSpec(kind='asymptotic_mean')
    over a mixed registration compiles to a MixedFamily policy."""
    reg = _mixed_registration()
    plan = AnalysisPlan(
        primary="rev",
        q=float(reg.q),
        compliance=SequentialCompliancePolicy(alpha=F(1, 40), family=True),
        inference=InferenceSpec(kind="asymptotic_mean", registration=reg),
    )
    compiled = compile_decision_plan(
        plan,
        [synthesise_metric(MetricSpec(name="rev", type="mean"))],
        design=_ENCOURAGEMENT,
    )
    assert isinstance(compiled.inference, MixedFamily)


def test_build_inference_keeps_asymptotic_mean_for_a_pure_registration():
    from tests.asymptotic_cases import mean_registration

    reg = mean_registration()
    plan = AnalysisPlan(
        primary="outcome", inference=InferenceSpec(kind="asymptotic_mean", registration=reg)
    )
    compiled = compile_decision_plan(
        plan,
        [synthesise_metric(MetricSpec(name="outcome", type="mean"))],
        design=Randomized(control_group=reg.control_group),
    )
    assert isinstance(compiled.inference, AsymptoticMean)


def _mixed_snapshot(reg):
    rows = []
    for i in range(200):
        for arm in ("control", "treatment"):
            rev = (1, 2, 3, 4)[i % 4] if arm == "control" else (6, 10, 14, 18)[i % 4]
            uptake = int(i % 4 == 3) if arm == "control" else int(i % 4 != 0)
            rows.append(
                {
                    "unit_id": f"{i:05d}-{arm}",
                    "group_id": arm,
                    "values": {"rev": rev, "uptake": uptake},
                }
            )
    return capture_sequential_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
    )


def test_mixed_family_evaluates_both_cells_from_one_joint_prefix():
    reg = _mixed_registration()
    policy = MixedFamily(registration=reg)
    snapshot = _mixed_snapshot(reg)
    computation = estimate_sequential(snapshot, policy)
    by_metric = {row.metric: row for row in computation.results}
    assert set(by_metric) == {"rev", "uptake"}
    assert (
        by_metric["rev"].require_asymptotic_sequential_result().checkpoint.model.law
        == "scalar_mean"
    )
    assert by_metric["uptake"].require_sequential_result().checkpoint.model.law == "bernoulli"
    prefixes = {row.require_sequential_result().checkpoint.prefix_id for row in computation.results}
    assert len(prefixes) == 1


def test_mixed_family_e_bh_selects_across_both_laws_with_a_family_wide_guarantee():
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family

    reg = _mixed_registration(itt_alpha=F(1, 40), uptake_alpha=F(1, 40))
    policy = MixedFamily(registration=reg)
    snapshot = _mixed_snapshot(reg)
    computation = estimate_sequential(snapshot, policy)
    cells = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
    ]
    outcome = select_sequential_family(cells, reg.q, policy, reg.q, computation=computation)
    assert outcome.n_family == 2
    assert outcome.q == float(reg.q)
    # e-BH (universal, direction-respecting) replaces the interim per-cell
    # Bonferroni union bound: a discovery reinverts at a real fcr_alpha, and
    # the family's guarantee is the weakest regime present -- asymptotic,
    # since the roster mixes a scalar-mean cell with a Bernoulli cell.
    assert outcome.selected
    assert outcome.fcr_alpha is not None
    assert outcome.guarantee == "asymptotic_sequential"
