"""Stable public re-exports for specialist estimator and utility APIs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "resolve": ("increment.kernel.resolver", "resolve"),
    "impute": ("increment.impute", None),
    "moments_from_summaries": ("increment.summaries", "moments_from_summaries"),
    "MetricTrends": ("increment.reporting", "MetricTrends"),
    "Normal": ("increment.estimation.inference", "Normal"),
    "AsymptoticMean": ("increment.estimation.sequential", "AsymptoticMean"),
    "AsymptoticSequentialResult": (
        "increment.estimation.sequential_result",
        "AsymptoticSequentialResult",
    ),
    "AsymptoticSequentialEvidence": (
        "increment.estimation.decision_types",
        "AsymptoticSequentialEvidence",
    ),
    "ScalarMeanModel": ("increment.semantics.sequential", "ScalarMeanModel"),
    "PredeclaredAdjustment": ("increment.semantics.sequential", "PredeclaredAdjustment"),
    "fit_predeclared_adjustment": ("increment.analysis", "fit_predeclared_adjustment"),
    "AlwaysValid": ("increment.estimation.sequential", "AlwaysValid"),
    "PredictivePrior": ("increment.semantics.sequential", "PredictivePrior"),
    "SequentialModel": ("increment.semantics.sequential", "SequentialModel"),
    "JointReveal": ("increment.semantics.sequential", "JointReveal"),
    "SequentialCell": ("increment.semantics.sequential", "SequentialCell"),
    "SequentialRegistration": ("increment.semantics.sequential", "SequentialRegistration"),
    "SequentialCompliancePolicy": ("increment.semantics.sequential", "SequentialCompliancePolicy"),
    "SequentialSnapshot": ("increment.sequential_state", "SequentialSnapshot"),
    "capture_sequential_snapshot": ("increment.sequential_state", "capture_sequential_snapshot"),
    "declare_sequential_freeze": ("increment.sequential_state", "declare_sequential_freeze"),
    "snapshot_from_json": ("increment.sequential_state", "snapshot_from_json"),
    "estimate_sequential": ("increment.estimation.sequential_runtime", "estimate_sequential"),
    "sequential_definition_id": ("increment.sequential_source", "sequential_definition_id"),
    "Covariate": ("increment.estimation.cate", "Covariate"),
    "ClusterBootstrap": ("increment.estimation.targeting", "ClusterBootstrap"),
    "SummaryStats": ("increment.estimation.armstats", "SummaryStats"),
    "SwitchbackBaseline": ("increment.power", "SwitchbackBaseline"),
    "ProspectiveAssumptionProvenance": (
        "increment.semantics.unit_cycle",
        "ProspectiveAssumptionProvenance",
    ),
    "UnitCycleVarianceEnvelope": ("increment.semantics.unit_cycle", "UnitCycleVarianceEnvelope"),
    "UnitCycleTApproximation": ("increment.semantics.unit_cycle", "UnitCycleTApproximation"),
    "UnitCycleReference": ("increment.semantics.unit_cycle", "UnitCycleReference"),
    "switchback_achieved_power": ("increment.power", "switchback_achieved_power"),
    "switchback_minimum_detectable_effect": (
        "increment.power",
        "switchback_minimum_detectable_effect",
    ),
    "switchback_required_blocks_or_units": (
        "increment.power",
        "switchback_required_blocks_or_units",
    ),
    "unit_cycle_power_lower_bound": ("increment.power", "unit_cycle_power_lower_bound"),
    "Baseline": ("increment.power", "Baseline"),
    "PowerDesign": ("increment.power", "PowerDesign"),
    "required_sample_size": ("increment.power", "required_sample_size"),
    "achieved_power": ("increment.power", "achieved_power"),
    "minimum_detectable_effect": ("increment.power", "minimum_detectable_effect"),
    "power_curve": ("increment.power", "power_curve"),
    "sample_ratio_mismatch": ("increment.estimation.diagnostics", "sample_ratio_mismatch"),
    "allocation_posterior_bands": (
        "increment.estimation.diagnostics",
        "allocation_posterior_bands",
    ),
    "absorb_factor": ("increment.absorption", "absorb_factor"),
    "confounding_sensitivity": ("increment.estimation.confounding", "confounding_sensitivity"),
    "estimate_cate": ("increment.cate", "estimate_cate"),
    "validate_cate": ("increment.cate", "validate_cate"),
    "select_targeting_rule": ("increment.cate", "select_targeting_rule"),
    "targeting_rule": ("increment.cate", "targeting_rule"),
    "segment_heterogeneity": ("increment.breakout.heterogeneity", "segment_heterogeneity"),
    "segment_contrast": ("increment.breakout.heterogeneity", "segment_contrast"),
    "segment_rollout_recommendation": (
        "increment.breakout.rollout",
        "segment_rollout_recommendation",
    ),
    "estimate_policy_contrast": ("increment.logged_policy", "estimate_policy_contrast"),
    "LoggedTrace": ("increment.logged_policy", "LoggedTrace"),
    "PolicyRegistry": ("increment.logged_policy", "PolicyRegistry"),
    "TabularPolicy": ("increment.logged_policy", "TabularPolicy"),
    "to_frame": ("increment.breakout.projection", "to_frame"),
    "IdentificationError": ("increment.estimation._adjust.overlap", "IdentificationError"),
    "CapabilityError": ("increment.errors", "CapabilityError"),
    "CodedError": ("increment.errors", "CodedError"),
    "DefinitionError": ("increment.errors", "DefinitionError"),
    "IncrementDeprecationWarning": ("increment.errors", "IncrementDeprecationWarning"),
    "IncrementRuntimeWarning": ("increment.errors", "IncrementRuntimeWarning"),
    "IncrementWarning": ("increment.errors", "IncrementWarning"),
    "InvalidRequestError": ("increment.errors", "InvalidRequestError"),
    "RefusalSpec": ("increment.errors", "RefusalSpec"),
    "UnsupportedRequestError": ("increment.errors", "UnsupportedRequestError"),
    "WarningSpec": ("increment.errors", "WarningSpec"),
    "WireFormatError": ("increment.errors", "WireFormatError"),
    "refuse": ("increment.errors", "refuse"),
    "warn": ("increment.errors", "warn"),
    "Assignment": ("increment.semantics.assignment", "Assignment"),
    "ParallelAssignment": ("increment.semantics.assignment", "ParallelAssignment"),
    "IndependentBernoulliOrder": ("increment.semantics.assignment", "IndependentBernoulliOrder"),
    "SharedScheduleOrder": ("increment.semantics.assignment", "SharedScheduleOrder"),
    "SwitchbackWindow": ("increment.semantics.assignment", "SwitchbackWindow"),
    "SwitchbackAssignment": ("increment.semantics.assignment", "SwitchbackAssignment"),
    "ParallelStudyEnvelope": ("increment._study", "ParallelStudyEnvelope"),
    "SwitchbackStudyEnvelope": ("increment._study", "SwitchbackStudyEnvelope"),
    "StudyEnvelope": ("increment._study", "StudyEnvelope"),
}

if TYPE_CHECKING:
    import increment.impute as impute
    from increment._study import (
        ParallelStudyEnvelope as ParallelStudyEnvelope,
    )
    from increment._study import (
        StudyEnvelope as StudyEnvelope,
    )
    from increment._study import (
        SwitchbackStudyEnvelope as SwitchbackStudyEnvelope,
    )
    from increment.absorption import absorb_factor as absorb_factor
    from increment.analysis import fit_predeclared_adjustment as fit_predeclared_adjustment
    from increment.breakout.heterogeneity import (
        segment_contrast as segment_contrast,
    )
    from increment.breakout.heterogeneity import (
        segment_heterogeneity as segment_heterogeneity,
    )
    from increment.breakout.projection import to_frame as to_frame
    from increment.breakout.rollout import (
        segment_rollout_recommendation as segment_rollout_recommendation,
    )
    from increment.cate import (
        estimate_cate as estimate_cate,
    )
    from increment.cate import (
        select_targeting_rule as select_targeting_rule,
    )
    from increment.cate import (
        targeting_rule as targeting_rule,
    )
    from increment.cate import (
        validate_cate as validate_cate,
    )
    from increment.errors import (
        CapabilityError as CapabilityError,
    )
    from increment.errors import (
        CodedError as CodedError,
    )
    from increment.errors import (
        DefinitionError as DefinitionError,
    )
    from increment.errors import (
        IncrementDeprecationWarning as IncrementDeprecationWarning,
    )
    from increment.errors import (
        IncrementRuntimeWarning as IncrementRuntimeWarning,
    )
    from increment.errors import (
        IncrementWarning as IncrementWarning,
    )
    from increment.errors import (
        InvalidRequestError as InvalidRequestError,
    )
    from increment.errors import (
        RefusalSpec as RefusalSpec,
    )
    from increment.errors import (
        UnsupportedRequestError as UnsupportedRequestError,
    )
    from increment.errors import (
        WarningSpec as WarningSpec,
    )
    from increment.errors import (
        WireFormatError as WireFormatError,
    )
    from increment.errors import (
        refuse as refuse,
    )
    from increment.errors import (
        warn as warn,
    )
    from increment.estimation._adjust.overlap import IdentificationError as IdentificationError
    from increment.estimation.armstats import SummaryStats as SummaryStats
    from increment.estimation.cate import Covariate as Covariate
    from increment.estimation.confounding import confounding_sensitivity as confounding_sensitivity
    from increment.estimation.decision_types import (
        AsymptoticSequentialEvidence as AsymptoticSequentialEvidence,
    )
    from increment.estimation.diagnostics import (
        allocation_posterior_bands as allocation_posterior_bands,
    )
    from increment.estimation.diagnostics import (
        sample_ratio_mismatch as sample_ratio_mismatch,
    )
    from increment.estimation.inference import Normal as Normal
    from increment.estimation.sequential import (
        AlwaysValid as AlwaysValid,
    )
    from increment.estimation.sequential import (
        AsymptoticMean as AsymptoticMean,
    )
    from increment.estimation.sequential_result import (
        AsymptoticSequentialResult as AsymptoticSequentialResult,
    )
    from increment.estimation.sequential_runtime import estimate_sequential as estimate_sequential
    from increment.estimation.targeting import ClusterBootstrap as ClusterBootstrap
    from increment.estimators.registry import (
        ESTIMATORS as ESTIMATORS,
    )
    from increment.estimators.registry import (
        estimator_id as estimator_id,
    )
    from increment.estimators.registry import (
        load as load,
    )
    from increment.estimators.registry import (
        observational_estimator_id as observational_estimator_id,
    )
    from increment.estimators.registry import (
        randomized_estimator_id as randomized_estimator_id,
    )
    from increment.estimators.registry import (
        routes as routes,
    )
    from increment.kernel.resolver import resolve as resolve
    from increment.logged_policy import (
        LoggedTrace as LoggedTrace,
    )
    from increment.logged_policy import (
        PolicyRegistry as PolicyRegistry,
    )
    from increment.logged_policy import (
        TabularPolicy as TabularPolicy,
    )
    from increment.logged_policy import (
        estimate_policy_contrast as estimate_policy_contrast,
    )
    from increment.power import (
        Baseline as Baseline,
    )
    from increment.power import (
        PowerDesign as PowerDesign,
    )
    from increment.power import (
        SwitchbackBaseline as SwitchbackBaseline,
    )
    from increment.power import (
        achieved_power as achieved_power,
    )
    from increment.power import (
        minimum_detectable_effect as minimum_detectable_effect,
    )
    from increment.power import (
        power_curve as power_curve,
    )
    from increment.power import (
        required_sample_size as required_sample_size,
    )
    from increment.power import (
        switchback_achieved_power as switchback_achieved_power,
    )
    from increment.power import (
        switchback_minimum_detectable_effect as switchback_minimum_detectable_effect,
    )
    from increment.power import (
        switchback_required_blocks_or_units as switchback_required_blocks_or_units,
    )
    from increment.power import (
        unit_cycle_power_lower_bound as unit_cycle_power_lower_bound,
    )
    from increment.reporting import MetricTrends as MetricTrends
    from increment.semantics.assignment import (
        Assignment as Assignment,
    )
    from increment.semantics.assignment import (
        IndependentBernoulliOrder as IndependentBernoulliOrder,
    )
    from increment.semantics.assignment import (
        ParallelAssignment as ParallelAssignment,
    )
    from increment.semantics.assignment import (
        SharedScheduleOrder as SharedScheduleOrder,
    )
    from increment.semantics.assignment import (
        SwitchbackAssignment as SwitchbackAssignment,
    )
    from increment.semantics.assignment import (
        SwitchbackWindow as SwitchbackWindow,
    )
    from increment.semantics.sequential import (
        JointReveal as JointReveal,
    )
    from increment.semantics.sequential import (
        PredeclaredAdjustment as PredeclaredAdjustment,
    )
    from increment.semantics.sequential import (
        PredictivePrior as PredictivePrior,
    )
    from increment.semantics.sequential import (
        ScalarMeanModel as ScalarMeanModel,
    )
    from increment.semantics.sequential import (
        SequentialCell as SequentialCell,
    )
    from increment.semantics.sequential import (
        SequentialCompliancePolicy as SequentialCompliancePolicy,
    )
    from increment.semantics.sequential import (
        SequentialModel as SequentialModel,
    )
    from increment.semantics.sequential import (
        SequentialRegistration as SequentialRegistration,
    )
    from increment.semantics.unit_cycle import (
        ProspectiveAssumptionProvenance as ProspectiveAssumptionProvenance,
    )
    from increment.semantics.unit_cycle import (
        UnitCycleReference as UnitCycleReference,
    )
    from increment.semantics.unit_cycle import (
        UnitCycleTApproximation as UnitCycleTApproximation,
    )
    from increment.semantics.unit_cycle import (
        UnitCycleVarianceEnvelope as UnitCycleVarianceEnvelope,
    )
    from increment.sequential_source import sequential_definition_id as sequential_definition_id
    from increment.sequential_state import (
        SequentialSnapshot as SequentialSnapshot,
    )
    from increment.sequential_state import (
        capture_sequential_snapshot as capture_sequential_snapshot,
    )
    from increment.sequential_state import (
        declare_sequential_freeze as declare_sequential_freeze,
    )
    from increment.sequential_state import (
        snapshot_from_json as snapshot_from_json,
    )
    from increment.summaries import moments_from_summaries as moments_from_summaries


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in {
        "ESTIMATORS",
        "estimator_id",
        "load",
        "observational_estimator_id",
        "randomized_estimator_id",
        "routes",
    }:
        from increment.estimators import registry

        return getattr(registry, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(
        set(__all__)
        | {
            "ESTIMATORS",
            "estimator_id",
            "load",
            "observational_estimator_id",
            "randomized_estimator_id",
            "routes",
        }
    )


__all__ = [
    "resolve",
    "impute",
    "moments_from_summaries",
    "MetricTrends",
    "Normal",
    "AsymptoticMean",
    "AsymptoticSequentialResult",
    "AsymptoticSequentialEvidence",
    "ScalarMeanModel",
    "PredeclaredAdjustment",
    "fit_predeclared_adjustment",
    "AlwaysValid",
    "PredictivePrior",
    "SequentialModel",
    "JointReveal",
    "SequentialCell",
    "SequentialRegistration",
    "SequentialCompliancePolicy",
    "SequentialSnapshot",
    "capture_sequential_snapshot",
    "declare_sequential_freeze",
    "snapshot_from_json",
    "estimate_sequential",
    "sequential_definition_id",
    "Covariate",
    "ClusterBootstrap",
    "SummaryStats",
    "SwitchbackBaseline",
    "ProspectiveAssumptionProvenance",
    "UnitCycleVarianceEnvelope",
    "UnitCycleTApproximation",
    "UnitCycleReference",
    "switchback_achieved_power",
    "switchback_minimum_detectable_effect",
    "switchback_required_blocks_or_units",
    "unit_cycle_power_lower_bound",
    "Baseline",
    "PowerDesign",
    "required_sample_size",
    "achieved_power",
    "minimum_detectable_effect",
    "power_curve",
    "sample_ratio_mismatch",
    "allocation_posterior_bands",
    "absorb_factor",
    "confounding_sensitivity",
    "estimate_cate",
    "validate_cate",
    "select_targeting_rule",
    "targeting_rule",
    "segment_heterogeneity",
    "segment_contrast",
    "segment_rollout_recommendation",
    "estimate_policy_contrast",
    "LoggedTrace",
    "PolicyRegistry",
    "TabularPolicy",
    "to_frame",
    "IdentificationError",
    "CapabilityError",
    "CodedError",
    "DefinitionError",
    "IncrementDeprecationWarning",
    "IncrementRuntimeWarning",
    "IncrementWarning",
    "InvalidRequestError",
    "RefusalSpec",
    "UnsupportedRequestError",
    "WarningSpec",
    "WireFormatError",
    "refuse",
    "warn",
    "Assignment",
    "ParallelAssignment",
    "IndependentBernoulliOrder",
    "SharedScheduleOrder",
    "SwitchbackWindow",
    "SwitchbackAssignment",
    "ParallelStudyEnvelope",
    "SwitchbackStudyEnvelope",
    "StudyEnvelope",
    "ESTIMATORS",
    "estimator_id",
    "load",
    "observational_estimator_id",
    "randomized_estimator_id",
    "routes",
]
