"""Coded-refusal regression tests for ``increment.decision``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from increment.decision import (
    AbsoluteArmDecisionProcedure,
    ArmHypothesisKey,
    CompiledDecisionPlan,
    CompiledViewPolicies,
    ContrastAnalysisState,
    ContrastDecisionProcedure,
    DecisionComputation,
    DecisionFailure,
    DefinitionsArmAnalysisState,
    FamilyMembership,
    FixedInference,
    MultiplicityFamily,
    NoFamily,
    PValueEvidence,
    RelativeArmDecisionProcedure,
    SeamArmAnalysisState,
)
from increment.errors import InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.sequential import AlwaysValid
from tests.sequential_cases import registration


def _procedure(**overrides: object) -> RelativeArmDecisionProcedure:
    fields: dict[str, object] = {
        "metric": "rev",
        "role": "primary",
        "decision_method": Method(name="unadjusted"),
        "alternative": "two-sided",
        "alpha": 0.05,
        "null_lift": 0.0,
    }
    fields.update(overrides)
    return RelativeArmDecisionProcedure(**fields)  # ty: ignore[invalid-argument-type] -- overrides intentionally untyped


def _plan(procedures: dict[str, object]) -> CompiledDecisionPlan:
    return CompiledDecisionPlan(
        declared=True,
        alpha=0.05,
        q=0.1,
        path="warehouse",
        inference=FixedInference(),
        procedures=procedures,  # ty: ignore[invalid-argument-type] -- intentionally untyped mapping
        view_policies=CompiledViewPolicies(
            asof=MultiplicityFamily(name="asof"),
            randomized_breakout=MultiplicityFamily(name="breakout"),
            encouragement_breakout=MultiplicityFamily(name="breakout"),
        ),
    )


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "decision.multiplicity.validate_policy",
            lambda: MultiplicityFamily(name="f", correction="bh", q=None),
        ),
        (
            "decision.multiplicity.bh",
            lambda: MultiplicityFamily(name="f", correction="none", q=0.1),
        ),
        (
            "decision.multiplicity.guarantee_mismatch",
            lambda: MultiplicityFamily(name="f", correction="none", guarantee="fwer"),
        ),
        (
            "decision.multiplicity.guarantee_mismatch",
            lambda: MultiplicityFamily(name="f", correction="bonferroni", guarantee="fdr"),
        ),
        (
            "decision.multiplicity.guarantee_mismatch",
            lambda: MultiplicityFamily(name="f", correction="bh", q=0.1, guarantee="none"),
        ),
        (
            "decision.multiplicity.guarantee_mismatch",
            lambda: MultiplicityFamily(name="f", correction="e_bh", q=0.1, guarantee="none"),
        ),
        (
            "decision.family.nofamily_membership_mark",
            lambda: FamilyMembership(family=NoFamily(), member=True),
        ),
        (
            "decision.arm_decision.one_sided_alpha",
            lambda: _procedure(alternative="greater", alpha=0.6),
        ),
        (
            "decision.relative_arm.null_lift_greater",
            lambda: _procedure(null_lift=-1.5),
        ),
        (
            "decision.absolute_arm.procedures_fixed_inference",
            lambda: AbsoluteArmDecisionProcedure(
                metric="rev",
                role="primary",
                decision_method=Method(name="unadjusted"),
                alternative="two-sided",
                alpha=0.05,
                null_abs=0.0,
                inference=AlwaysValid(registration=registration()),
            ),
        ),
        (
            "decision.arm_decision.one_sided_alpha",
            lambda: ContrastDecisionProcedure(
                metric="rev",
                role="primary",
                alternative="greater",
                null_abs=0.0,
                alpha=0.6,
            ),
        ),
        (
            "decision.compiled_decision.procedure_mapping_key",
            lambda: _plan({"wrong_key": _procedure()}),
        ),
    ],
)
def test_decision_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def test_seam_arm_analysis_state_mismatch_is_an_assertion_not_a_value_error():
    # decision.py's own compile step derives a normalized study from the
    # source's design, so a design-without-study mismatch here is a bug.
    source = SimpleNamespace(context=SimpleNamespace(design=None))
    with pytest.raises(AssertionError):
        SeamArmAnalysisState(source=source, study=object(), experiment_name="exp")  # ty: ignore[invalid-argument-type] -- deliberately wrong types to trip an internal assertion


def test_definitions_arm_analysis_state_mismatch_is_an_assertion_not_a_type_error():
    source = SimpleNamespace(context=SimpleNamespace(design=None))
    with pytest.raises(AssertionError):
        DefinitionsArmAnalysisState(
            source=source,  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            study=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            definitions=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            experiment=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            connection=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            session=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            experiment_name="exp",
            backend="duckdb",
            store="auto",
            on_mixed_assignment="error",
            exposure_lookup={},
        )


def test_contrast_analysis_state_mismatch_is_an_assertion_not_a_type_error():
    source = SimpleNamespace(context=object())
    with pytest.raises(AssertionError):
        ContrastAnalysisState(source=source, experiment_name="exp")  # ty: ignore[invalid-argument-type] -- deliberately wrong types to trip an internal assertion


def test_decision_computation_overlap_is_an_assertion_not_a_value_error():
    key = ArmHypothesisKey(metric="rev", group_id="treatment", estimand="ate")
    evidence_item = PValueEvidence(hypothesis=key, method="wald", p_value=0.5, reference="normal")
    failure_item = DecisionFailure(hypothesis=key, code="x", context={})
    with pytest.raises(AssertionError):
        DecisionComputation(results=(), evidence={key: evidence_item}, failures={key: failure_item})


@pytest.mark.slow
def test_constructor_invariants_survive_optimized_python():
    import subprocess
    import sys
    from pathlib import Path

    checks = [
        "tests/test_decision.py::test_decision_refusal_carries_code",
        "tests/test_decision.py::test_seam_arm_analysis_state_mismatch_is_an_assertion_not_a_value_error",
        "tests/test_decision.py::test_definitions_arm_analysis_state_mismatch_is_an_assertion_not_a_type_error",
        "tests/test_decision.py::test_contrast_analysis_state_mismatch_is_an_assertion_not_a_type_error",
        "tests/test_decision.py::test_decision_computation_overlap_is_an_assertion_not_a_value_error",
        "tests/estimation/test_decision_bundle.py::test_computation_rejects_mismatched_keys",
        "tests/test_errors.py::test_refusal_spec_post_init_rejects_positional_only_renderer_via_assertion",
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            "-n",
            "0",
            "-q",
            "-m",
            "",
            "-W",
            "default:assertions not in test modules or plugins will be ignored:pytest.PytestConfigWarning",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:tach",
            *checks,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
