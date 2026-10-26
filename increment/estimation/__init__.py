import importlib
from typing import TYPE_CHECKING

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
    "CupedFit",
    "MulticovariateCupedFit",
    "MulticovariateCupedLayout",
    "MulticovariateCupedProjection",
    "RatioCupedFit",
    "Method",
    "VarianceModel",
    "MeanVarianceModel",
    "RatioVarianceModel",
    "Registry",
    "se_log_mean",
    "stable_log_ratio",
    "ratio_moments",
    "fit_cuped",
    "fit_cuped_multicovariate",
    "fit_ratio_cuped",
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

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
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
    "AbsorptionResult": ("increment.estimation.absorption", "AbsorptionResult"),
    "absorb_one_way": ("increment.estimation.absorption", "absorb_one_way"),
    "ArmStats": ("increment.estimation.armstats", "ArmStats"),
    "IndependentMeanComponent": ("increment.estimation.armstats", "IndependentMeanComponent"),
    "IndependentMeanReference": ("increment.estimation.armstats", "IndependentMeanReference"),
    "infer_independent_mean": ("increment.estimation.inference", "infer_independent_mean"),
    "SummaryStats": ("increment.estimation.armstats", "SummaryStats"),
    "Normal": ("increment.estimation.inference", "Normal"),
    "LiftPosterior": ("increment.estimation.inference", "LiftPosterior"),
    "normal_posterior": ("increment.estimation.inference", "normal_posterior"),
    "Estimate": ("increment.estimation.results", "Estimate"),
    "LiftEstimate": ("increment.estimation.results", "LiftEstimate"),
    "CupedFit": ("increment.estimation.cuped.scalar", "CupedFit"),
    "MulticovariateCupedFit": (
        "increment.estimation.cuped.types",
        "MulticovariateCupedFit",
    ),
    "MulticovariateCupedLayout": (
        "increment.estimation.cuped.encoding",
        "MulticovariateCupedLayout",
    ),
    "MulticovariateCupedProjection": (
        "increment.estimation.cuped.types",
        "MulticovariateCupedProjection",
    ),
    "RatioCupedFit": ("increment.estimation.cuped.scalar", "RatioCupedFit"),
    "Method": ("increment.estimation.engine", "Method"),
    "VarianceModel": ("increment.estimation.variance", "VarianceModel"),
    "MeanVarianceModel": ("increment.estimation.variance", "MeanVarianceModel"),
    "RatioVarianceModel": ("increment.estimation.variance", "RatioVarianceModel"),
    "Registry": ("increment.estimation.variance", "Registry"),
    "se_log_mean": ("increment.estimation.variance", "se_log_mean"),
    "stable_log_ratio": ("increment.estimation.variance", "stable_log_ratio"),
    "ratio_moments": ("increment.estimation.variance", "ratio_moments"),
    "fit_cuped": ("increment.estimation.cuped.scalar", "fit_cuped"),
    "fit_cuped_multicovariate": (
        "increment.estimation.cuped.multicovariate",
        "fit_cuped_multicovariate",
    ),
    "fit_ratio_cuped": ("increment.estimation.cuped.scalar", "fit_ratio_cuped"),
    "estimate_lift": ("increment.estimation.engine", "estimate_lift"),
    "infer_lift": ("increment.estimation.inference", "infer_lift"),
    "SRMResult": ("increment.estimation.diagnostics", "SRMResult"),
    "sample_ratio_mismatch": ("increment.estimation.diagnostics", "sample_ratio_mismatch"),
    "HeterogeneityResult": ("increment.estimation.meta", "HeterogeneityResult"),
    "cochran_q": ("increment.estimation.meta", "cochran_q"),
    "hksj_pooled_mean": ("increment.estimation.meta", "hksj_pooled_mean"),
    "MarginalizedSegmentIntervals": ("increment.estimation.meta", "MarginalizedSegmentIntervals"),
    "marginalized_segment_intervals": (
        "increment.estimation.meta",
        "marginalized_segment_intervals",
    ),
    "CateResult": ("increment.estimation.cate", "CateResult"),
    "CateScoreState": ("increment.estimation.cate", "CateScoreState"),
    "ClusterScore": ("increment.estimation.cate", "ClusterScore"),
    "Covariate": ("increment.estimation.cate", "Covariate"),
    "DesignSpec": ("increment.estimation.cate", "DesignSpec"),
    "InteractionEffect": ("increment.estimation.cate", "InteractionEffect"),
    "WaldTest": ("increment.estimation.cate", "WaldTest"),
    "fit_cate": ("increment.estimation.cate", "fit_cate"),
    "prune_gram": ("increment.estimation.cate", "prune_gram"),
    "CateValidation": ("increment.estimation.targeting", "CateValidation"),
    "ClanRow": ("increment.estimation.targeting", "ClanRow"),
    "GroupEffect": ("increment.estimation.targeting", "GroupEffect"),
    "PsiFn": ("increment.estimation._score_design", "PsiFn"),
    "RankTest": ("increment.estimation.targeting", "RankTest"),
    "ScoreDesign": ("increment.estimation._score_design", "ScoreDesign"),
    "validate_cate_arrays": ("increment.estimation.targeting", "validate_cate_arrays"),
    "SegmentRollout": ("increment.estimation.rollout", "SegmentRollout"),
    "segment_rollout": ("increment.estimation.rollout", "segment_rollout"),
}

if TYPE_CHECKING:
    from increment.estimation._score_design import PsiFn, ScoreDesign
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
    from increment.estimation.cuped.encoding import MulticovariateCupedLayout
    from increment.estimation.cuped.multicovariate import fit_cuped_multicovariate
    from increment.estimation.cuped.scalar import (
        CupedFit,
        RatioCupedFit,
        fit_cuped,
        fit_ratio_cuped,
    )
    from increment.estimation.cuped.types import (
        MulticovariateCupedFit,
        MulticovariateCupedProjection,
    )
    from increment.estimation.decision_types import AsymptoticSequentialEvidence
    from increment.estimation.diagnostics import SRMResult, sample_ratio_mismatch
    from increment.estimation.engine import Method, estimate_lift
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
        RankTest,
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
