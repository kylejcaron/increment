"""Import-time weight checks.

``increment``'s top-level package and its estimation submodule re-export
names that live in heavier modules (the warehouse builders pull in ibis
and narwhals, ``Method``/``estimate_lift`` pull in narwhals). Those re-exports
must be lazy so that importing lightweight entry points (``increment.power``
and ``increment.estimation.armstats``) never drags ibis, narwhals, or
pyarrow into ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import increment

EXPECTED_RESULT_EXPORTS = [
    "Estimate",
    "BinomialConfidenceSet",
    "WinsorConfidenceSet",
    "WinsorInferenceSpec",
    "BootstrapReference",
    "RankReference",
    "WinsorPermutationTest",
    "IndependentMeanComponent",
    "IndependentMeanReference",
    "SetEndpoint",
    "SetInterval",
    "LiftEstimate",
    "LiftEstimates",
    "ContrastResult",
    "ContrastResults",
    "SwitchbackAssignmentDiagnostic",
    "BreakoutEstimate",
    "BreakoutEstimates",
    "DailyMetricValue",
    "DailyMetricValues",
    "DailyLiftEstimate",
    "DailyLiftEstimates",
    "HeterogeneitySummary",
    "HeterogeneitySummaries",
    "SegmentEstimate",
    "SegmentEstimates",
    "SegmentRolloutResult",
    "RolloutRecommendation",
    "RolloutRecommendations",
    "RolloutSegment",
    "RolloutSegments",
    "PowerResult",
    "SwitchbackPowerResult",
    "UnitCycleFailureCount",
    "UnitCycleLawMoments",
    "UnitCyclePowerLowerBoundResult",
    "UnitCycleModelPowerResult",
    "UnitCycleModelMdeResult",
    "UnitCycleModelRequiredUnitsResult",
    "PowerCurvePoint",
    "PowerCurve",
    "SRMResult",
    "AllocationBand",
    "NotApplicable",
    "AbsorptionResult",
    "ConfoundingSensitivity",
    "CateResult",
    "CateScoreState",
    "ClusterScore",
    "CateValidation",
    "TargetingRule",
    "TargetingSelection",
    "CateEvaluationPopulation",
    "MetricTrend",
    "SitewideImpact",
    "SitewideRatioImpact",
    "PolicyValueContrast",
]

MOVED_RESULT_NAMES = set(EXPECTED_RESULT_EXPORTS)
ADVANCED_POWER_NAMES = {
    "segment_pairwise_required_sample_size",
    "segment_pairwise_achieved_power",
    "segment_pairwise_minimum_detectable_effect",
    "joint_q_power_fixed",
    "joint_q_power_random",
}
REMOVED_NAMES = {
    "ScoreStats",
    "sitewide_impact",
    "sitewide_impact_ratio",
    "SitewideContrast",
    "SitewideRatioContrast",
}


def test_non_root_names_do_not_resolve_from_increment():
    moved_specialists = {"ArtifactStore", "MomentsRequest", "canonical_row_bytes", "estimate_cate"}
    for name in MOVED_RESULT_NAMES | ADVANCED_POWER_NAMES | REMOVED_NAMES | moved_specialists:
        assert name not in increment.__all__
        assert not hasattr(increment, name), name


def test_relocated_names_resolve_from_their_public_modules():
    import increment.power as power
    import increment.results as results

    for name in MOVED_RESULT_NAMES:
        assert getattr(results, name) is not None
    for name in ADVANCED_POWER_NAMES:
        assert getattr(power, name) is not None


def test_results_import_avoids_heavy_dependencies():
    code = (
        "import increment.results\n"
        "import sys\n"
        "heavy = {'ibis', 'pandas', 'pyarrow', 'duckdb'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_power_and_armstats_avoid_heavy_dependencies():
    code = (
        "import increment\n"
        "import increment.power\n"
        "import increment.estimation.armstats\n"
        "from increment.estimation.sequential import GaussianScoreMixture\n"
        "GaussianScoreMixture()\n"
        "import sys\n"
        "heavy = {'ibis', 'pandas', 'pyarrow', 'duckdb'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_core_entry_points_avoid_the_optional_dashboard_stack():
    """Shipping increment.dashboard must not put a UI library on the core path.

    The dashboard is imported by name (``from increment.dashboard import ...``),
    never from ``increment/__init__.py``, so marimo, CoefTable, Altair, and
    pandas stay out of every lightweight entry point.
    """
    code = (
        "import increment\n"
        "import increment.power\n"
        "import increment.frame\n"
        "import sys\n"
        "ui = {'marimo', 'coeftable', 'altair', 'pandas'} & set(sys.modules)\n"
        "assert not ui, sorted(ui)\n"
        "assert 'increment.dashboard' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_diagnostics_exports_avoid_heavy_dependencies():
    """Resolving diagnostic and robustness exports avoids frame-library imports.

    Receive-only models resolve from ``increment.results`` while their
    specialist functions come from ``increment.estimators``.
    """
    code = (
        "import increment\n"
        "import increment.results\n"
        "import increment.estimators\n"
        "increment.results.SRMResult\n"
        "increment.estimators.sample_ratio_mismatch\n"
        "increment.results.AllocationBand\n"
        "increment.estimators.allocation_posterior_bands\n"
        "increment.estimators.absorb_factor\n"
        "increment.results.AbsorptionResult\n"
        "increment.estimators.confounding_sensitivity\n"
        "increment.results.ConfoundingSensitivity\n"
        "import sys\n"
        "heavy = {'ibis', 'pandas', 'pyarrow', 'duckdb'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_estimation_engine_is_importable():
    """increment.estimation.engine imports without error (narwhals is a hard dep)."""
    from increment.estimation.engine import (  # noqa: F401
        Method,
        _df_to_arms,
        estimate_lift,
    )


def test_semantics_models_avoids_sqlglot():
    """increment.semantics.models is pydantic-only; sqlglot is loader-only and lazy."""
    code = "import increment.semantics.models\nimport sys\nassert 'sqlglot' not in sys.modules\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_estimation_and_power_avoid_ibis():
    """increment.estimation and increment.power consume moments/frames, never ibis."""
    code = (
        "import increment.estimation\n"
        "import increment.power\n"
        "import sys\n"
        "assert 'ibis' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_frame_avoids_ibis():
    """increment.frame is narwhals-only; ibis belongs to the query/artifact path."""
    code = "import increment.frame\nimport sys\nassert 'ibis' not in sys.modules\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_semantics_models_avoids_ibis():
    """increment.semantics.models is pydantic-only; ibis belongs to the query/loader path."""
    code = "import increment.semantics.models\nimport sys\nassert 'ibis' not in sys.modules\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sequential_specs_exported_from_estimator_namespace():
    from increment.estimators import AlwaysValid
    from tests.sequential_cases import registration

    assert AlwaysValid(registration=registration()).label == "always_valid"


def test_encouragement_design_exports():
    import increment.semantics.design as design

    for name in ("Encouragement", "ExclusionRestriction", "UptakeSpec"):
        assert name in increment.__all__
        assert getattr(increment, name) is getattr(design, name)


def test_winsorization_is_exported_from_public_layers():
    import increment
    from increment.semantics import Winsorization

    assert increment.Winsorization is Winsorization
    assert "Winsorization" in increment.__all__


def test_moments_from_summaries_is_in_estimator_namespace():
    import increment.estimators as estimators
    import increment.summaries as summaries

    assert "moments_from_summaries" not in increment.__all__
    assert estimators.moments_from_summaries is summaries.moments_from_summaries


def test_estimation_import_leaves_decision_core_unloaded():
    """increment.estimation must not eagerly pull decision/sources/_readout_request."""
    code = (
        "import increment.estimation\n"
        "import sys\n"
        "assert 'increment.decision' not in sys.modules, 'decision'\n"
        "assert 'increment.sources' not in sys.modules, 'sources'\n"
        "assert 'increment._readout_request' not in sys.modules, '_readout_request'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_resolving_analysis_leaves_warehouse_and_scipy_stats_unloaded():
    code = (
        "import sys\n"
        "from increment import from_frame, run\n"
        "heavy = {'ibis', 'sqlglot', 'scipy.stats'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
        "import pandas as pd\n"
        "frame = pd.DataFrame({'unit': ['u1', 'u2', 'u3', 'u4'], "
        "'group': ['control', 'control', 'treatment', 'treatment'], "
        "'value': [1.0, 2.0, 3.0, 4.0]})\n"
        "experiment, resolver = from_frame(frame, shape='summary', unit='unit', group='group', "
        "control='control', metrics={'value': 'mean'})\n"
        "run(experiment, resolver)\n"
        "heavy = {'ibis', 'sqlglot'} & set(sys.modules)\n"
        "assert not heavy, sorted(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_root_import_loads_one_module():
    code = (
        "import increment\n"
        "import sys\n"
        "assert len([m for m in sys.modules if m.startswith('increment')]) == 1\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_evidence_types_import_is_light():
    code = (
        "import increment._evidence_types\n"
        "import sys\n"
        "assert len([m for m in sys.modules if m.startswith('increment')]) <= 16\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


pytestmark = pytest.mark.integration


def test_estimation_package_import_is_light():
    code = (
        "import sys\n"
        "import increment.estimation\n"
        "loaded = {name for name in sys.modules if name.startswith('increment.')}\n"
        "assert len(loaded) <= 5, len(loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cuped_package_import_is_light():
    code = (
        "import sys\n"
        "import increment.estimation.cuped\n"
        "loaded = {name for name in sys.modules if name.startswith('increment.')}\n"
        "assert len(loaded) <= 5, len(loaded)\n"
        "from increment.estimation.cuped import fit_cuped\n"
        "assert fit_cuped\n"
        "for name in ('multicovariate', 'prepare', 'encoding'):\n"
        "    assert f'increment.estimation.cuped.{name}' not in sys.modules, name\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_estimation_export_resolves_to_its_recorded_binding():
    import importlib
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "estimation_exports.json"
    expected = json.loads(fixture.read_text())
    for package_name, exports in expected.items():
        package = importlib.import_module(package_name)
        assert set(package.__all__) == set(exports)
        for name, binding in exports.items():
            module_name, attribute = binding.split(":")
            module = importlib.import_module(module_name)
            assert getattr(package, name) is getattr(module, attribute), (
                package_name,
                name,
                binding,
            )

    from increment.estimation import binomial_rr

    assert binomial_rr.__name__ == "increment.estimation.binomial_rr"


def test_declarations_import_is_light():
    code = (
        "import sys\n"
        "import increment.declarations\n"
        "loaded = {name for name in sys.modules if name.startswith('increment.')}\n"
        "assert len(loaded) <= 16, len(loaded)\n"
        "assert not any(name.startswith('increment.estimation') for name in loaded), loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_method_construction_leaves_estimation_unloaded():
    code = (
        "import sys\n"
        "from increment.declarations import Method\n"
        "Method(name='cuped', variance_reduction='cuped', covariates=['x'])\n"
        "assert not any(name.startswith('increment.estimation') for name in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
