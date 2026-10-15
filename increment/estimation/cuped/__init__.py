"""Scalar and multicovariate CUPED estimators."""

from increment.estimation.cuped.encoding import MISSING_CATEGORY, MulticovariateCupedLayout
from increment.estimation.cuped.multicovariate import fit_cuped_multicovariate
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

__all__ = [
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
