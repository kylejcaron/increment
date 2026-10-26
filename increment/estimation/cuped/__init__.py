"""Scalar and multicovariate CUPED estimators."""

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "BreakoutCell",
    "BreakoutIdentity",
    "MethodFitAnswer",
    "MethodFitRequest",
    "PreparedMulticovariateFit",
    "prepare_multicovariate_fit",
    "CategoricalCells",
    "FitState",
    "InferenceSummary",
    "ProjectionSpec",
    "AdjustedRatioMoments",
    "CupedFit",
    "MISSING_CATEGORY",
    "MulticovariateCupedFit",
    "MulticovariateCupedLayout",
    "MulticovariateCupedProjection",
    "RatioCupedFit",
    "cuped_adjust",
    "fit_cuped",
    "fit_cuped_multicovariate",
    "fit_ratio_cuped",
    "pooled_theta",
]

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "BreakoutCell": ("increment._evidence_types", "BreakoutCell"),
    "BreakoutIdentity": ("increment._evidence_types", "BreakoutIdentity"),
    "MethodFitAnswer": ("increment._evidence_types", "MethodFitAnswer"),
    "MethodFitRequest": ("increment._evidence_types", "MethodFitRequest"),
    "PreparedMulticovariateFit": ("increment._evidence_types", "PreparedMulticovariateFit"),
    "prepare_multicovariate_fit": (
        "increment.estimation.cuped.prepare",
        "prepare_multicovariate_fit",
    ),
    "CategoricalCells": ("increment.estimation.cuped.types", "CategoricalCells"),
    "FitState": ("increment.estimation.cuped.types", "FitState"),
    "InferenceSummary": ("increment.estimation.cuped.types", "InferenceSummary"),
    "ProjectionSpec": ("increment.estimation.cuped.types", "ProjectionSpec"),
    "AdjustedRatioMoments": ("increment.estimation.cuped.scalar", "AdjustedRatioMoments"),
    "CupedFit": ("increment.estimation.cuped.scalar", "CupedFit"),
    "MISSING_CATEGORY": ("increment.estimation.cuped.encoding", "MISSING_CATEGORY"),
    "MulticovariateCupedFit": ("increment.estimation.cuped.types", "MulticovariateCupedFit"),
    "MulticovariateCupedLayout": (
        "increment.estimation.cuped.encoding",
        "MulticovariateCupedLayout",
    ),
    "MulticovariateCupedProjection": (
        "increment.estimation.cuped.types",
        "MulticovariateCupedProjection",
    ),
    "RatioCupedFit": ("increment.estimation.cuped.scalar", "RatioCupedFit"),
    "cuped_adjust": ("increment.estimation.cuped.scalar", "cuped_adjust"),
    "fit_cuped": ("increment.estimation.cuped.scalar", "fit_cuped"),
    "fit_cuped_multicovariate": (
        "increment.estimation.cuped.multicovariate",
        "fit_cuped_multicovariate",
    ),
    "fit_ratio_cuped": ("increment.estimation.cuped.scalar", "fit_ratio_cuped"),
    "pooled_theta": ("increment.estimation.cuped.scalar", "pooled_theta"),
}

if TYPE_CHECKING:
    from increment._evidence_types import (
        BreakoutCell,
        BreakoutIdentity,
        MethodFitAnswer,
        MethodFitRequest,
        PreparedMulticovariateFit,
    )
    from increment.estimation.cuped.encoding import MISSING_CATEGORY, MulticovariateCupedLayout
    from increment.estimation.cuped.multicovariate import fit_cuped_multicovariate
    from increment.estimation.cuped.prepare import prepare_multicovariate_fit
    from increment.estimation.cuped.scalar import (
        AdjustedRatioMoments,
        CupedFit,
        RatioCupedFit,
        cuped_adjust,
        fit_cuped,
        fit_ratio_cuped,
        pooled_theta,
    )
    from increment.estimation.cuped.types import (
        CategoricalCells,
        FitState,
        InferenceSummary,
        MulticovariateCupedFit,
        MulticovariateCupedProjection,
        ProjectionSpec,
    )


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
