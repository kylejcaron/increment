import importlib
from typing import TYPE_CHECKING

from increment.estimation.absorption import AbsorptionResult, absorb_one_way
from increment.estimation.armstats import (
    ArmStats,
    IndependentMeanComponent,
    IndependentMeanReference,
    SummaryStats,
)
from increment.estimation.cate import (
    CateResult,
    CateScoreState,
    ClusterScore,
    Covariate,
    DesignSpec,
    InteractionEffect,
    WaldTest,
    fit_cate,
    prune_gram,
)
from increment.estimation.decision_types import AsymptoticSequentialEvidence
from increment.estimation.diagnostics import SRMResult, sample_ratio_mismatch
from increment.estimation.inference import (
    LiftPosterior,
    Normal,
    infer_independent_mean,
    infer_lift,
    normal_posterior,
)
from increment.estimation.meta import (
    HeterogeneityResult,
    MarginalizedSegmentIntervals,
    cochran_q,
    hksj_pooled_mean,
    marginalized_segment_intervals,
)
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.rollout import SegmentRollout, segment_rollout
from increment.estimation.sequential import AsymptoticMean
from increment.estimation.sequential_result import AsymptoticSequentialResult
from increment.estimation.targeting import (
    CateValidation,
    ClanRow,
    GroupEffect,
    PsiFn,
    RankTest,
    ScoreDesign,
    validate_cate_arrays,
)
from increment.estimation.variance import (
    MeanVarianceModel,
    RatioVarianceModel,
    Registry,
    VarianceModel,
    ratio_moments,
    se_log_mean,
    stable_log_ratio,
)
from increment.semantics.sequential import ScalarMeanModel

if TYPE_CHECKING:
    from increment.estimation.engine import Method, estimate_lift

__all__ = [
    "AsymptoticMean",
    "AsymptoticSequentialResult",
    "AsymptoticSequentialEvidence",
    "ScalarMeanModel",
    "AbsorptionResult",
    "absorb_one_way",
    "ArmStats",
    "IndependentMeanComponent",
    "IndependentMeanReference",
    "infer_independent_mean",
    "SummaryStats",
    "Normal",
    "LiftPosterior",
    "normal_posterior",
    "Estimate",
    "LiftEstimate",
    "Method",
    "VarianceModel",
    "MeanVarianceModel",
    "RatioVarianceModel",
    "Registry",
    "se_log_mean",
    "stable_log_ratio",
    "ratio_moments",
    "estimate_lift",
    "infer_lift",
    "SRMResult",
    "sample_ratio_mismatch",
    "HeterogeneityResult",
    "cochran_q",
    "hksj_pooled_mean",
    "MarginalizedSegmentIntervals",
    "marginalized_segment_intervals",
    "CateResult",
    "CateScoreState",
    "ClusterScore",
    "Covariate",
    "DesignSpec",
    "InteractionEffect",
    "WaldTest",
    "fit_cate",
    "prune_gram",
    "CateValidation",
    "ClanRow",
    "GroupEffect",
    "PsiFn",
    "RankTest",
    "ScoreDesign",
    "validate_cate_arrays",
    "SegmentRollout",
    "segment_rollout",
]

_LAZY_IMPORTS = {
    "Method": ("increment.estimation.engine", "Method"),
    "estimate_lift": ("increment.estimation.engine", "estimate_lift"),
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
