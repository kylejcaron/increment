from __future__ import annotations

from typing import cast

import pytest

from increment.errors import CapabilityError
from increment.evidence import (
    AssignmentEvidence,
    AssignmentRequest,
    EvidenceCapabilities,
    require_evidence,
)


class _Source:
    evidence = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset(),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=False,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset(),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=None,
    )


def test_switchback_assignment_request_preserves_order_draws() -> None:
    from increment.evidence import evidence_provider, request_evidence
    from tests.source_conformance import switchback_source

    source = switchback_source()
    evidence = request_evidence(
        evidence_provider(source),
        AssignmentRequest(experiment="frame", all_periods=True),
    )
    assert isinstance(evidence, AssignmentEvidence)
    assert evidence.periods == ("0:period=0", "0:period=1", "1:period=0", "1:period=1")
    selected = cast(
        AssignmentEvidence,
        request_evidence(
            evidence_provider(source),
            AssignmentRequest(experiment="frame", period="0:period=1"),
        ),
    )
    # The source fixture has u1 in treatment and u2 in control at cycle 0,
    # period 1; preserve that original coordinate rather than shifting it.
    assert selected.arm_counts == (
        (1, "0:period=1", "control", 1),
        (1, "0:period=1", "treatment", 1),
    )
    assert evidence.randomization_law == "independent_bernoulli_order"


def test_assignment_request_refuses_without_relation_capability() -> None:
    source = _Source()
    with pytest.raises(CapabilityError) as error:
        require_evidence(source, request=AssignmentRequest(experiment="exp"))
    assert error.value.code == "evidence.assignment.unavailable"
    assert error.value.context["offered"] == "none"
    assert "from_unit_summary" in str(error.value.context["route"])


@pytest.mark.parametrize(
    ("fixture", "experiment"),
    [
        ("frame_summary_source", "summary-conformance"),
        ("frame_panel_source", "panel-conformance"),
    ],
)
def test_frame_assignment_requests_preserve_unit_arm_counts(fixture: str, experiment: str) -> None:
    from increment.evidence import evidence_provider, request_evidence
    from tests import source_conformance

    source = getattr(source_conformance, fixture)()
    evidence = request_evidence(
        evidence_provider(source),
        AssignmentRequest(experiment=experiment),
    )
    assert isinstance(evidence, AssignmentEvidence)
    assert evidence.arm_counts == ((1, None, "control", 4), (1, None, "treatment", 4))


def test_switchback_assignment_preserves_declared_arm_labels() -> None:

    from increment.evidence import evidence_provider, request_evidence
    from increment.frame import from_switchback_panel
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation
    from tests.source_conformance import switchback_frame

    frame = switchback_frame()
    frame["group"] = frame["group"].replace({"control": "baseline", "treatment": "variant"})
    source = from_switchback_panel(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="baseline", allocation={"baseline": 0.5, "variant": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    evidence = request_evidence(
        evidence_provider(source),
        AssignmentRequest(experiment="frame", all_periods=True),
    )
    assert isinstance(evidence, AssignmentEvidence)
    assert {arm for _, _, arm, _ in evidence.arm_counts} == {"baseline", "variant"}


def test_shared_switchback_order_draws_are_counted_per_block() -> None:
    from increment.evidence import evidence_provider, request_evidence
    from increment.frame import from_switchback_panel
    from increment.semantics.assignment import (
        SharedScheduleOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from tests.source_conformance import switchback_frame

    frame = switchback_frame()
    frame["group"] = [
        "control" if (int(cycle) + int(period)) % 2 == 0 else "treatment"
        for cycle, period in zip(frame["cycle"], frame["period"], strict=True)
    ]
    source = from_switchback_panel(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=SharedScheduleOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    evidence = request_evidence(
        evidence_provider(source),
        AssignmentRequest(experiment="frame", all_periods=True),
    )
    assert isinstance(evidence, AssignmentEvidence)
    assert len(evidence.order_draws) == 2
    assert evidence.independence_grain == "shared_block"


def test_allocation_weights_are_reported_as_normalized_probabilities() -> None:
    import pandas as pd

    from increment.evidence import evidence_provider, request_evidence
    from increment.frame import from_unit_summary
    from increment.semantics.design import Randomized

    source = from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["u1", "u2"],
                "arm": ["control", "treatment"],
                "value": [1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        experiment_id="weighted",
        design=Randomized(control_group="control", allocation={"control": 3, "treatment": 1}),
    )
    evidence = request_evidence(evidence_provider(source), AssignmentRequest(experiment="weighted"))
    assert isinstance(evidence, AssignmentEvidence)
    assert evidence.probabilities == (("control", 0.75), ("treatment", 0.25))


def test_moments_constructor_answers_from_retained_assignment_counts() -> None:
    from increment.evidence import evidence_provider, request_evidence
    from tests.source_conformance import moments_source

    evidence = cast(
        AssignmentEvidence,
        request_evidence(
            evidence_provider(moments_source()),
            AssignmentRequest(experiment="moments-current-conformance"),
        ),
    )
    assert evidence.arm_counts == ((1, None, "control", 2), (1, None, "treatment", 2))


@pytest.mark.integration
def test_artifact_constructor_answers_from_explicit_assignment_extension(tmp_path) -> None:
    from increment.evidence import evidence_provider, request_evidence
    from tests.source_conformance import unit_day_artifact_source

    source = unit_day_artifact_source(tmp_path)
    evidence = cast(
        AssignmentEvidence,
        request_evidence(
            evidence_provider(source),
            AssignmentRequest(experiment="new_onboarding_v2"),
        ),
    )
    assert evidence.versions == (1,)
    assert sum(count for _, _, _, count in evidence.arm_counts) > 0


def test_moments_without_assignment_counts_refuses_with_constructor_route() -> None:
    from increment.evidence import evidence_provider, request_evidence
    from tests.source_conformance import legacy_fixed10_moments_source

    with pytest.raises(CapabilityError) as error:
        request_evidence(
            evidence_provider(legacy_fixed10_moments_source()),
            AssignmentRequest(experiment="moments-conformance"),
        )
    assert error.value.code == "evidence.assignment.unavailable"
    assert "from_moments" in str(error.value.context["route"])
