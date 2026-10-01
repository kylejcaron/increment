"""Receive-only result models returned by increment workflows."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from increment.breakout.estimates import (
        BreakoutEstimate,
        BreakoutEstimates,
        DailyLiftEstimate,
        DailyLiftEstimates,
        DailyMetricValue,
        DailyMetricValues,
        LiftEstimates,
    )
    from increment.breakout.heterogeneity import (
        HeterogeneitySummaries,
        HeterogeneitySummary,
        SegmentEstimate,
        SegmentEstimates,
    )
    from increment.breakout.rollout import (
        RolloutRecommendation,
        RolloutRecommendations,
        RolloutSegment,
        RolloutSegments,
        SegmentRolloutResult,
    )
    from increment.estimation.absorption import AbsorptionResult
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.cate import CateResult, CateScoreState, ClusterScore
    from increment.estimation.contrast_results import ContrastResult, ContrastResults
    from increment.estimation.diagnostics import AllocationBand, NotApplicable, SRMResult
    from increment.estimation.results import BinomialConfidenceSet, Estimate, LiftEstimate
    from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
    from increment.estimation.targeting import (
        CateEvaluationPopulation,
        CateValidation,
        TargetingRule,
        TargetingSelection,
    )
    from increment.logged_policy import PolicyValueContrast
    from increment.power import (
        PowerCurve,
        PowerCurvePoint,
        PowerResult,
        SwitchbackPowerResult,
    )
    from increment.power.unit_cycle import (
        UnitCycleFailureCount,
        UnitCycleLawMoments,
        UnitCycleModelMdeResult,
        UnitCycleModelPowerResult,
        UnitCycleModelRequiredUnitsResult,
        UnitCyclePowerLowerBoundResult,
    )
    from increment.reporting import MetricTrend
    from increment.switchback import SwitchbackAssignmentDiagnostic
    from increment.winsor import (
        BootstrapReference,
        RankReference,
        SetEndpoint,
        SetInterval,
        WinsorConfidenceSet,
        WinsorInferenceSpec,
        WinsorPermutationTest,
    )

__all__ = [
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

_LAZY_IMPORTS = {
    "Estimate": ("increment.estimation.results", "Estimate"),
    "BinomialConfidenceSet": ("increment.estimation.results", "BinomialConfidenceSet"),
    "WinsorConfidenceSet": ("increment.winsor", "WinsorConfidenceSet"),
    "WinsorInferenceSpec": ("increment.winsor", "WinsorInferenceSpec"),
    "BootstrapReference": ("increment.winsor", "BootstrapReference"),
    "RankReference": ("increment.winsor", "RankReference"),
    "WinsorPermutationTest": ("increment.winsor", "WinsorPermutationTest"),
    "IndependentMeanComponent": ("increment.estimation.armstats", "IndependentMeanComponent"),
    "IndependentMeanReference": ("increment.estimation.armstats", "IndependentMeanReference"),
    "SetEndpoint": ("increment.winsor", "SetEndpoint"),
    "SetInterval": ("increment.winsor", "SetInterval"),
    "LiftEstimate": ("increment.estimation.results", "LiftEstimate"),
    "LiftEstimates": ("increment.breakout.estimates", "LiftEstimates"),
    "ContrastResult": ("increment.estimation.contrast_results", "ContrastResult"),
    "ContrastResults": ("increment.estimation.contrast_results", "ContrastResults"),
    "SwitchbackAssignmentDiagnostic": (
        "increment.switchback",
        "SwitchbackAssignmentDiagnostic",
    ),
    "BreakoutEstimate": ("increment.breakout.estimates", "BreakoutEstimate"),
    "BreakoutEstimates": ("increment.breakout.estimates", "BreakoutEstimates"),
    "DailyMetricValue": ("increment.breakout.estimates", "DailyMetricValue"),
    "DailyMetricValues": ("increment.breakout.estimates", "DailyMetricValues"),
    "DailyLiftEstimate": ("increment.breakout.estimates", "DailyLiftEstimate"),
    "DailyLiftEstimates": ("increment.breakout.estimates", "DailyLiftEstimates"),
    "HeterogeneitySummary": ("increment.breakout.heterogeneity", "HeterogeneitySummary"),
    "HeterogeneitySummaries": ("increment.breakout.heterogeneity", "HeterogeneitySummaries"),
    "SegmentEstimate": ("increment.breakout.heterogeneity", "SegmentEstimate"),
    "SegmentEstimates": ("increment.breakout.heterogeneity", "SegmentEstimates"),
    "SegmentRolloutResult": ("increment.breakout.rollout", "SegmentRolloutResult"),
    "RolloutRecommendation": ("increment.breakout.rollout", "RolloutRecommendation"),
    "RolloutRecommendations": ("increment.breakout.rollout", "RolloutRecommendations"),
    "RolloutSegment": ("increment.breakout.rollout", "RolloutSegment"),
    "RolloutSegments": ("increment.breakout.rollout", "RolloutSegments"),
    "PowerResult": ("increment.power", "PowerResult"),
    "SwitchbackPowerResult": ("increment.power", "SwitchbackPowerResult"),
    "UnitCycleFailureCount": ("increment.power.unit_cycle", "UnitCycleFailureCount"),
    "UnitCycleLawMoments": ("increment.power.unit_cycle", "UnitCycleLawMoments"),
    "UnitCyclePowerLowerBoundResult": ("increment.power", "UnitCyclePowerLowerBoundResult"),
    "UnitCycleModelPowerResult": ("increment.power.unit_cycle", "UnitCycleModelPowerResult"),
    "UnitCycleModelMdeResult": ("increment.power.unit_cycle", "UnitCycleModelMdeResult"),
    "UnitCycleModelRequiredUnitsResult": (
        "increment.power.unit_cycle",
        "UnitCycleModelRequiredUnitsResult",
    ),
    "PowerCurvePoint": ("increment.power", "PowerCurvePoint"),
    "PowerCurve": ("increment.power", "PowerCurve"),
    "SRMResult": ("increment.estimation.diagnostics", "SRMResult"),
    "AllocationBand": ("increment.estimation.diagnostics", "AllocationBand"),
    "NotApplicable": ("increment.estimation.diagnostics", "NotApplicable"),
    "AbsorptionResult": ("increment.estimation.absorption", "AbsorptionResult"),
    "CateScoreState": ("increment.estimation.cate", "CateScoreState"),
    "ClusterScore": ("increment.estimation.cate", "ClusterScore"),
    "CateResult": ("increment.estimation.cate", "CateResult"),
    "CateValidation": ("increment.estimation.targeting", "CateValidation"),
    "TargetingRule": ("increment.estimation.targeting", "TargetingRule"),
    "TargetingSelection": ("increment.estimation.targeting", "TargetingSelection"),
    "CateEvaluationPopulation": ("increment.estimation.targeting", "CateEvaluationPopulation"),
    "MetricTrend": ("increment.reporting", "MetricTrend"),
    "SitewideImpact": ("increment.estimation.sitewide", "SitewideImpact"),
    "SitewideRatioImpact": ("increment.estimation.sitewide", "SitewideRatioImpact"),
    "PolicyValueContrast": ("increment.logged_policy", "PolicyValueContrast"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
