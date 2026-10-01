from __future__ import annotations

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.compatibility_catalog import (
    CAPABILITIES,
    CAPABILITY_PROVENANCE,
    MATRIX,
    PAIR_PROVENANCE,
    PAIRS,
    SCENARIOS,
    SILENT,
    EvidenceRef,
    _scenario_axis_signature,
    validate_evidence,
    validate_scenario_outcome,
)

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_one_column_per_capability():
    assert set(MATRIX) == set(CAPABILITIES)


def test_every_warns_bearing_cell_has_advisory():
    """``warns`` is a short probe-match fragment, not user-facing prose --
    every cell that carries one must also carry a complete ``advisory``
    the renderer can show without exposing the regex fragment itself."""
    cells = [cell for column in MATRIX.values() for cell in column.values()]
    cells += list(PAIRS.values())
    for cell in cells:
        if cell.warns:
            assert cell.advisory, f"cell with warns={cell.warns!r} has no advisory"


def test_scenario_ids_are_unique():
    ids = [scenario.id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_scenario_axis_combinations_are_unique():
    signatures = [_scenario_axis_signature(scenario) for scenario in SCENARIOS]
    assert len(signatures) == len(set(signatures))


def test_scenario_axis_combinations_detect_reordered_duplicates():
    forward = replace(SCENARIOS[0], id="reordered-forward", axes=(("x", "1"), ("y", "2")))
    reordered = replace(SCENARIOS[0], id="reordered-backward", axes=(("y", "2"), ("x", "1")))
    signatures = [
        _scenario_axis_signature(forward),
        _scenario_axis_signature(reordered),
    ]
    assert len(signatures) != len(set(signatures))


def test_scenario_construction_rejects_duplicate_axis_names():
    with pytest.raises(ValueError, match="duplicate axis name"):
        replace(SCENARIOS[0], axes=(("capability", "sequential"), ("capability", "estimate")))


def test_contract_scenarios_match_accepted_outcomes():
    for scenario in SCENARIOS:
        validate_scenario_outcome(scenario, scenario.evaluate())


def test_every_scenario_has_user_explanation_and_evidence():
    for scenario in SCENARIOS:
        assert scenario.explanation.strip(), scenario.id
        assert scenario.evidence, scenario.id


def test_cell_runtime_status_maps_known_gap_to_limited():
    assert MATRIX["estimate"]["mean"].runtime_status == "supported"
    assert MATRIX["cuped"]["quantile"].runtime_status == "refused"
    assert MATRIX["estimate"]["total"].runtime_status == "not_applicable"
    # No accepted MATRIX declaration currently carries the "silent" probe
    # status (a known composition gap pinned as-is); construct one directly
    # to pin the public projection without inventing an accepted cell.
    silent = SILENT("synthetic: exercises the silent -> limited projection")
    assert silent.runtime_status == "limited"


def test_cell_runtime_status_projects_advisory_bearing_supported_to_limited():
    """A ``supported`` cell that also carries complete
    advisory prose describes a material runtime/statistical caveat, per the
    design's own LIMITED definition -- it must project to ``limited``, not
    ``supported``. A supported cell with no advisory is unaffected."""
    assert PAIRS[("cluster", "sitewide")].advisory
    assert PAIRS[("cluster", "sitewide")].runtime_status == "limited"
    # No advisory: stays SUPPORTED.
    assert MATRIX["estimate"]["mean"].advisory is None
    assert MATRIX["estimate"]["mean"].runtime_status == "supported"


def test_evidence_references_exist_and_recovery_evidence_is_tagged():
    """Every evidence path/test statically names a real function. A
    ``parameter_recovery`` ref additionally must carry both runtime pytest
    markers, including a compatibility id matching its catalog scenario.

    Delegates to ``tests.compatibility_catalog.validate_evidence`` -- the
    same function ``scripts/render_compatibility.py`` calls during direct
    generation, so pytest and the renderer can never drift apart on what
    counts as a valid evidence reference."""
    validate_evidence(SCENARIOS)


def _evidence_ref(scenario_id: str, path: str) -> EvidenceRef:
    scenario = next(s for s in SCENARIOS if s.id == scenario_id)
    return next(ref for ref in scenario.evidence if ref.path == path)


def test_plan_resolver_evidence_names_the_exact_test_class():
    """``test_quantile_refuses_before_plan_can_claim_sequential_evidence`` lives inside
    ``TestAlwaysValidQuantileWarning`` in tests/test_plan_resolver.py, not
    at module scope. The reference must carry that class qualification --
    a bare function name that happens to also be a nested method name must
    never validate merely because some node in the file has a matching
    name."""
    ref = _evidence_ref("always-valid-quantile-secondary-family", "tests/test_plan_resolver.py")
    assert (
        ref.test
        == "TestAlwaysValidQuantileWarning::test_quantile_refuses_before_plan_can_claim_sequential_evidence"
    )


def test_validate_evidence_resolves_class_qualified_method():
    """A ``Class::method`` reference must resolve to that exact method
    nested inside that exact top-level class -- not any function of that
    name anywhere in the file."""
    scenario = replace(
        SCENARIOS[0],
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_plan_resolver.py",
                test="TestAlwaysValidQuantileWarning::test_quantile_refuses_before_plan_can_claim_sequential_evidence",
            ),
        ),
    )
    validate_evidence((scenario,))


def test_validate_evidence_rejects_unqualified_reference_to_a_class_method():
    """A method nested inside a class must be referenced as
    ``Class::method``. An unqualified name must not match it by walking
    the whole file regardless of nesting -- that is the exact laxness
    that let a wrong-scoped reference validate silently."""
    scenario = replace(
        SCENARIOS[0],
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_plan_resolver.py",
                test="test_quantile_refuses_before_plan_can_claim_sequential_evidence",
            ),
        ),
    )
    with pytest.raises(ValueError, match="no top-level function"):
        validate_evidence((scenario,))


def test_validate_evidence_rejects_reference_to_a_nonexistent_class():
    scenario = replace(
        SCENARIOS[0],
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_plan_resolver.py",
                test="NotARealClass::test_quantile_refuses_before_plan_can_claim_sequential_evidence",
            ),
        ),
    )
    with pytest.raises(ValueError, match="no class named 'NotARealClass'"):
        validate_evidence((scenario,))


def test_validate_evidence_rejects_reference_to_a_missing_method_on_a_real_class():
    scenario = replace(
        SCENARIOS[0],
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_plan_resolver.py",
                test="TestAlwaysValidQuantileWarning::test_does_not_exist",
            ),
        ),
    )
    with pytest.raises(ValueError, match="no method named 'test_does_not_exist'"):
        validate_evidence((scenario,))


@pytest.mark.parameter_recovery
def _parameter_recovery_only_evidence():
    raise AssertionError("validate_evidence must inspect marks without running evidence")


@pytest.mark.parameter_recovery
class _ClassMarkedEvidence:
    @pytest.mark.compatibility("hygiene-class-method-mark-inheritance")
    def evidence_method(self):
        raise AssertionError("validate_evidence must inspect marks without running evidence")


@pytest.mark.compatibility("hygiene-inherited-marks")
class _InheritedMarkedEvidence(_ClassMarkedEvidence):
    def evidence_method(self):
        raise AssertionError("validate_evidence must inspect marks without running evidence")


def test_validate_evidence_rejects_a_scenario_missing_the_parameter_recovery_pytest_mark():
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-missing-parameter-recovery-marker",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_errors.py",
                test="test_coded_error_requires_explicit_code_and_context",
            ),
        ),
    )
    with pytest.raises(ValueError, match="missing the parameter_recovery marker"):
        validate_evidence((scenario,))


def test_validate_evidence_rejects_a_scenario_missing_the_compatibility_pytest_mark():
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-missing-compatibility-marker",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_compatibility_catalog.py",
                test="_parameter_recovery_only_evidence",
            ),
        ),
    )
    with pytest.raises(ValueError, match="missing the compatibility marker"):
        validate_evidence((scenario,))


def test_validate_evidence_rejects_a_mismatched_compatibility_marker_id():
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-mismatched-compatibility-marker",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_compatibility_catalog.py",
                test="_ClassMarkedEvidence::evidence_method",
            ),
        ),
    )
    with pytest.raises(ValueError, match="compatibility marker id does not match"):
        validate_evidence((scenario,))


def test_validate_evidence_rejects_a_nonexistent_parameter_recovery_test():
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-nonexistent-test",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/estimation/test_quantile_sequential_coverage.py",
                test="test_this_function_does_not_exist",
            ),
        ),
    )
    with pytest.raises(ValueError, match="no top-level function named"):
        validate_evidence((scenario,))


def test_validate_evidence_merges_class_and_method_marks(monkeypatch):
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-class-method-mark-inheritance",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_compatibility_catalog.py",
                test="_ClassMarkedEvidence::evidence_method",
            ),
        ),
    )
    validate_evidence((scenario,))
    with monkeypatch.context() as patch:
        patch.setattr(_ClassMarkedEvidence, "pytestmark", [])
        with pytest.raises(ValueError, match="missing the parameter_recovery marker"):
            validate_evidence((scenario,))
    with monkeypatch.context() as patch:
        patch.setattr(_ClassMarkedEvidence.evidence_method, "pytestmark", [])
        with pytest.raises(ValueError, match="missing the compatibility marker"):
            validate_evidence((scenario,))


def test_validate_evidence_merges_singleton_module_and_function_marks(monkeypatch):
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-module-marks",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_compatibility_catalog.py",
                test="_parameter_recovery_only_evidence",
            ),
        ),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            sys.modules[__name__],
            "pytestmark",
            pytest.mark.compatibility(scenario.id),
            raising=False,
        )
        validate_evidence((scenario,))
    with pytest.raises(ValueError, match="missing the compatibility marker"):
        validate_evidence((scenario,))


def test_validate_evidence_preserves_base_marks_under_subclass_marks(monkeypatch):
    scenario = replace(
        SCENARIOS[0],
        id="hygiene-inherited-marks",
        evidence=(
            EvidenceRef(
                kind="parameter_recovery",
                path="tests/test_compatibility_catalog.py",
                test="_InheritedMarkedEvidence::evidence_method",
            ),
        ),
    )
    validate_evidence((scenario,))
    monkeypatch.setattr(_ClassMarkedEvidence, "pytestmark", [])
    with pytest.raises(ValueError, match="missing the parameter_recovery marker"):
        validate_evidence((scenario,))


def test_capability_provenance_is_exhaustive():
    """Every MATRIX capability must declare
    exact composition-probe evidence and an owning production module --
    no capability may fall silently through with no provenance record."""
    assert set(CAPABILITY_PROVENANCE) == set(CAPABILITIES)


def test_pair_provenance_is_exhaustive():
    assert set(PAIR_PROVENANCE) == set(PAIRS)


def test_every_provenance_probe_names_a_real_composition_matrix_function():
    """Every declared probe reference must name a function that really
    exists in tests/test_composition_matrix.py, parsed statically like
    validate_evidence's scenario evidence checks -- a provenance record
    must never claim coverage a rename or deletion silently broke."""
    tree = ast.parse((ROOT / "tests/test_composition_matrix.py").read_text())
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    for provenance in (*CAPABILITY_PROVENANCE.values(), *PAIR_PROVENANCE.values()):
        path, _, func = provenance.probe.partition("::")
        assert path == "tests/test_composition_matrix.py", provenance.probe
        assert func in names, provenance.probe


def test_every_cell_exposes_refusal_metadata_or_explicit_unavailable_marker():
    """Every refused MATRIX/PAIRS cell must expose a stable production
    refusal code when one is declared, else the explicit ``"unavailable"``
    marker -- refusal metadata must never be silently missing. A
    non-refused cell has no refusal metadata to expose."""
    cells = [cell for column in MATRIX.values() for cell in column.values()]
    cells += list(PAIRS.values())
    for cell in cells:
        if cell.status == "refused":
            assert cell.raises is not None and cell.fragment is not None
            assert cell.refusal_code_display is not None
            assert cell.refusal_code_display == (cell.code or "unavailable")
        else:
            assert cell.refusal_code_display is None


@pytest.mark.parametrize(
    ("capability", "metric", "code"),
    [
        ("cuped", "quantile", "frame.metric.cuped_does_apply"),
        ("observational", "ratio", "estimation.adjust_common.supported_ratio_metric"),
        ("cate", "ratio", "cate.does_support_ratio"),
        ("cate", "quantile", "cate.cate_supported_quantile"),
    ],
)
def test_coded_runtime_refusals_are_declared_in_catalog(
    capability: str, metric: str, code: str
) -> None:
    assert MATRIX[capability][metric].code == code
