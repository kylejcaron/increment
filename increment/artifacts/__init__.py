"""Stable public re-exports for artifacts APIs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "RelationLocator": ("increment.semantics.artifact", "RelationLocator"),
    "RelationRole": ("increment.semantics.artifact", "RelationRole"),
    "UnitDayArtifactRef": ("increment.semantics.artifact", "UnitDayArtifactRef"),
    "ArtifactRelationRef": ("increment.semantics.artifact", "ArtifactRelationRef"),
    "BaseRelations": ("increment.semantics.artifact", "BaseRelations"),
    "Freshness": ("increment.semantics.artifact", "Freshness"),
    "MeasureManifest": ("increment.semantics.artifact", "MeasureManifest"),
    "SimpleMetricMeasure": ("increment.semantics.artifact", "SimpleMetricMeasure"),
    "RatioMetricMeasure": ("increment.semantics.artifact", "RatioMetricMeasure"),
    "MetricMeasure": ("increment.semantics.artifact", "MetricMeasure"),
    "ArtifactContext": ("increment.semantics.artifact", "ArtifactContext"),
    "UnitDayArtifactManifest": ("increment.semantics.artifact", "UnitDayArtifactManifest"),
    "ExtensionRefBase": ("increment.semantics.artifact", "ExtensionRefBase"),
    "BreakoutDimensionExtension": ("increment.semantics.artifact", "BreakoutDimensionExtension"),
    "FactorDimensionExtension": ("increment.semantics.artifact", "FactorDimensionExtension"),
    "ClusterIdentityExtension": ("increment.semantics.artifact", "ClusterIdentityExtension"),
    "CupedPreperiodExtension": ("increment.semantics.artifact", "CupedPreperiodExtension"),
    "AssignmentCountsExtension": ("increment.semantics.artifact", "AssignmentCountsExtension"),
    "TriggerPopulationExtension": ("increment.semantics.artifact", "TriggerPopulationExtension"),
    "TriggerMeasureStatsExtension": (
        "increment.semantics.artifact",
        "TriggerMeasureStatsExtension",
    ),
    "EncouragementUptakeExtension": (
        "increment.semantics.artifact",
        "EncouragementUptakeExtension",
    ),
    "SiteVolumeExtension": ("increment.semantics.artifact", "SiteVolumeExtension"),
    "UnitCovariateExtension": ("increment.semantics.artifact", "UnitCovariateExtension"),
    "UnitCovariateLevelExtension": ("increment.semantics.artifact", "UnitCovariateLevelExtension"),
    "ArtifactExtensionRef": ("increment.semantics.artifact", "ArtifactExtensionRef"),
    "ExtensionRequestBase": ("increment.semantics.artifact", "ExtensionRequestBase"),
    "BreakoutDimensionRequest": ("increment.semantics.artifact", "BreakoutDimensionRequest"),
    "FactorDimensionRequest": ("increment.semantics.artifact", "FactorDimensionRequest"),
    "ClusterIdentityRequest": ("increment.semantics.artifact", "ClusterIdentityRequest"),
    "CupedPreperiodRequest": ("increment.semantics.artifact", "CupedPreperiodRequest"),
    "AssignmentCountsRequest": ("increment.semantics.artifact", "AssignmentCountsRequest"),
    "TriggerPopulationRequest": ("increment.semantics.artifact", "TriggerPopulationRequest"),
    "TriggerMeasureStatsRequest": ("increment.semantics.artifact", "TriggerMeasureStatsRequest"),
    "EncouragementUptakeRequest": ("increment.semantics.artifact", "EncouragementUptakeRequest"),
    "SiteVolumeRequest": ("increment.semantics.artifact", "SiteVolumeRequest"),
    "UnitCovariateRequest": ("increment.semantics.artifact", "UnitCovariateRequest"),
    "UnitCovariateLevelRequest": ("increment.semantics.artifact", "UnitCovariateLevelRequest"),
    "ArtifactExtensionRequest": ("increment.semantics.artifact", "ArtifactExtensionRequest"),
    "ArtifactExtensionCatalogEntry": (
        "increment.semantics.artifact",
        "ArtifactExtensionCatalogEntry",
    ),
    "ArtifactPublication": ("increment.query.artifact_contract", "ArtifactPublication"),
    "ArtifactSnapshot": ("increment.query.artifact_contract", "ArtifactSnapshot"),
    "ArtifactStore": ("increment.query.artifact_contract", "ArtifactStore"),
    "compile_unit_day_artifact_context": (
        "increment.query.artifact_contract",
        "compile_unit_day_artifact_context",
    ),
    "unit_day_artifact_extension_catalog": (
        "increment.query.artifact_contract",
        "unit_day_artifact_extension_catalog",
    ),
    "SourceSnapshotEvidence": ("increment.query.session", "SourceSnapshotEvidence"),
}

if TYPE_CHECKING:
    from increment.query.artifact_contract import (
        ArtifactPublication as ArtifactPublication,
    )
    from increment.query.artifact_contract import (
        ArtifactSnapshot as ArtifactSnapshot,
    )
    from increment.query.artifact_contract import (
        ArtifactStore as ArtifactStore,
    )
    from increment.query.artifact_contract import (
        compile_unit_day_artifact_context as compile_unit_day_artifact_context,
    )
    from increment.query.artifact_contract import (
        unit_day_artifact_extension_catalog as unit_day_artifact_extension_catalog,
    )
    from increment.query.session import SourceSnapshotEvidence as SourceSnapshotEvidence
    from increment.semantics.artifact import (
        ArtifactContext as ArtifactContext,
    )
    from increment.semantics.artifact import (
        ArtifactExtensionCatalogEntry as ArtifactExtensionCatalogEntry,
    )
    from increment.semantics.artifact import (
        ArtifactExtensionRef as ArtifactExtensionRef,
    )
    from increment.semantics.artifact import (
        ArtifactExtensionRequest as ArtifactExtensionRequest,
    )
    from increment.semantics.artifact import (
        ArtifactRelationRef as ArtifactRelationRef,
    )
    from increment.semantics.artifact import (
        AssignmentCountsExtension as AssignmentCountsExtension,
    )
    from increment.semantics.artifact import (
        AssignmentCountsRequest as AssignmentCountsRequest,
    )
    from increment.semantics.artifact import (
        BaseRelations as BaseRelations,
    )
    from increment.semantics.artifact import (
        BreakoutDimensionExtension as BreakoutDimensionExtension,
    )
    from increment.semantics.artifact import (
        BreakoutDimensionRequest as BreakoutDimensionRequest,
    )
    from increment.semantics.artifact import (
        ClusterIdentityExtension as ClusterIdentityExtension,
    )
    from increment.semantics.artifact import (
        ClusterIdentityRequest as ClusterIdentityRequest,
    )
    from increment.semantics.artifact import (
        CupedPreperiodExtension as CupedPreperiodExtension,
    )
    from increment.semantics.artifact import (
        CupedPreperiodRequest as CupedPreperiodRequest,
    )
    from increment.semantics.artifact import (
        EncouragementUptakeExtension as EncouragementUptakeExtension,
    )
    from increment.semantics.artifact import (
        EncouragementUptakeRequest as EncouragementUptakeRequest,
    )
    from increment.semantics.artifact import (
        ExtensionRefBase as ExtensionRefBase,
    )
    from increment.semantics.artifact import (
        ExtensionRequestBase as ExtensionRequestBase,
    )
    from increment.semantics.artifact import (
        FactorDimensionExtension as FactorDimensionExtension,
    )
    from increment.semantics.artifact import (
        FactorDimensionRequest as FactorDimensionRequest,
    )
    from increment.semantics.artifact import (
        Freshness as Freshness,
    )
    from increment.semantics.artifact import (
        MeasureManifest as MeasureManifest,
    )
    from increment.semantics.artifact import (
        MetricMeasure as MetricMeasure,
    )
    from increment.semantics.artifact import (
        RatioMetricMeasure as RatioMetricMeasure,
    )
    from increment.semantics.artifact import (
        RelationLocator as RelationLocator,
    )
    from increment.semantics.artifact import (
        RelationRole as RelationRole,
    )
    from increment.semantics.artifact import (
        SimpleMetricMeasure as SimpleMetricMeasure,
    )
    from increment.semantics.artifact import (
        SiteVolumeExtension as SiteVolumeExtension,
    )
    from increment.semantics.artifact import (
        SiteVolumeRequest as SiteVolumeRequest,
    )
    from increment.semantics.artifact import (
        TriggerMeasureStatsExtension as TriggerMeasureStatsExtension,
    )
    from increment.semantics.artifact import (
        TriggerMeasureStatsRequest as TriggerMeasureStatsRequest,
    )
    from increment.semantics.artifact import (
        TriggerPopulationExtension as TriggerPopulationExtension,
    )
    from increment.semantics.artifact import (
        TriggerPopulationRequest as TriggerPopulationRequest,
    )
    from increment.semantics.artifact import (
        UnitCovariateExtension as UnitCovariateExtension,
    )
    from increment.semantics.artifact import (
        UnitCovariateLevelExtension as UnitCovariateLevelExtension,
    )
    from increment.semantics.artifact import (
        UnitCovariateLevelRequest as UnitCovariateLevelRequest,
    )
    from increment.semantics.artifact import (
        UnitCovariateRequest as UnitCovariateRequest,
    )
    from increment.semantics.artifact import (
        UnitDayArtifactManifest as UnitDayArtifactManifest,
    )
    from increment.semantics.artifact import (
        UnitDayArtifactRef as UnitDayArtifactRef,
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "RelationLocator",
    "RelationRole",
    "UnitDayArtifactRef",
    "ArtifactRelationRef",
    "BaseRelations",
    "Freshness",
    "MeasureManifest",
    "SimpleMetricMeasure",
    "RatioMetricMeasure",
    "MetricMeasure",
    "ArtifactContext",
    "UnitDayArtifactManifest",
    "ExtensionRefBase",
    "BreakoutDimensionExtension",
    "FactorDimensionExtension",
    "ClusterIdentityExtension",
    "CupedPreperiodExtension",
    "AssignmentCountsExtension",
    "TriggerPopulationExtension",
    "TriggerMeasureStatsExtension",
    "EncouragementUptakeExtension",
    "SiteVolumeExtension",
    "UnitCovariateExtension",
    "UnitCovariateLevelExtension",
    "ArtifactExtensionRef",
    "ExtensionRequestBase",
    "BreakoutDimensionRequest",
    "FactorDimensionRequest",
    "ClusterIdentityRequest",
    "CupedPreperiodRequest",
    "AssignmentCountsRequest",
    "TriggerPopulationRequest",
    "TriggerMeasureStatsRequest",
    "EncouragementUptakeRequest",
    "SiteVolumeRequest",
    "UnitCovariateRequest",
    "UnitCovariateLevelRequest",
    "ArtifactExtensionRequest",
    "ArtifactExtensionCatalogEntry",
    "ArtifactPublication",
    "ArtifactSnapshot",
    "ArtifactStore",
    "compile_unit_day_artifact_context",
    "unit_day_artifact_extension_catalog",
    "SourceSnapshotEvidence",
]
