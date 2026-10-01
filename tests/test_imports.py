"""Import-time weight checks.

``increment``'s top-level package and its estimation submodule re-export
names that live in heavier modules (``Analysis`` pulls in ibis
and narwhals, ``Method``/``estimate_lift`` pull in narwhals). Those re-exports
must be lazy so that importing lightweight entry points (``increment.power``
and ``increment.estimation.armstats``) never drags ibis, narwhals, or
pyarrow into ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

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
    for name in MOVED_RESULT_NAMES | ADVANCED_POWER_NAMES | REMOVED_NAMES:
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
    """Accessing diagnostics/absorption result and function exports must not pull in frame libraries.

    ``increment.results`` exposes the result types while the retained root
    functions resolve from ``increment.estimation.diagnostics`` and
    ``increment.absorption``. Those modules use narwhals (a hard, lightweight
    dep), never a frame library directly, so no heavy dependency should appear
    after resolving all six public exports.
    """
    code = (
        "import increment\n"
        "import increment.results\n"
        "increment.results.SRMResult\n"
        "increment.sample_ratio_mismatch\n"
        "increment.results.AllocationBand\n"
        "increment.allocation_posterior_bands\n"
        "increment.absorb_factor\n"
        "increment.results.AbsorptionResult\n"
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


def test_sequential_specs_exported():
    import increment
    from tests.sequential_cases import registration

    assert increment.AlwaysValid(registration=registration()).label == "always_valid"


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
