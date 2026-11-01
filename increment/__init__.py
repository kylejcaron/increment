"""increment - A/B testing querying and stats engine."""
# The TYPE_CHECKING imports below provide signatures for the lazy root exports.

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from increment._run import RunResult, run
    from increment.analysis import Analysis
    from increment.builders import (
        from_definitions,
        from_frame,
        from_moments,
        from_switchback_frame,
        from_unit_day_artifact,
    )
    from increment.declarations import Method
    from increment.evidence import EvidenceProvider
    from increment.frame import MetricSpec
    from increment.kernel.contracts import Scope
    from increment.kernel.experiment import Experiment, SourceReference
    from increment.kernel.resolver import BoundResolver, Resolver, Runnable, runnable
    from increment.result_envelope import render
    from increment.semantics.design import (
        AdjustmentSet,
        Design,
        Encouragement,
        ExclusionRestriction,
        IdentificationGate,
        Observational,
        Randomized,
        UptakeSpec,
    )
    from increment.semantics.models import (
        AnalysisPlan,
        CupedCovariate,
        CupedCovariateBinding,
        Definitions,
        ExperimentMetric,
        InferenceSpec,
        MethodSpec,
        Metric,
        MultiplicitySpec,
        Winsorization,
    )

try:
    __version__ = version("increment")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "Analysis",
    "Experiment",
    "SourceReference",
    "Scope",
    "Resolver",
    "BoundResolver",
    "RunResult",
    "run",
    "render",
    "from_frame",
    "from_definitions",
    "from_unit_day_artifact",
    "from_switchback_frame",
    "from_moments",
    "Runnable",
    "runnable",
    "Definitions",
    "Metric",
    "MetricSpec",
    "CupedCovariate",
    "CupedCovariateBinding",
    "Winsorization",
    "Design",
    "Randomized",
    "Observational",
    "Encouragement",
    "UptakeSpec",
    "ExclusionRestriction",
    "AdjustmentSet",
    "IdentificationGate",
    "AnalysisPlan",
    "MultiplicitySpec",
    "ExperimentMetric",
    "InferenceSpec",
    "Method",
    "MethodSpec",
    "__version__",
    "EvidenceProvider",
]

_LAZY_IMPORTS = {
    "Analysis": ("increment.analysis", "Analysis"),
    "RunResult": ("increment._run", "RunResult"),
    "run": ("increment._run", "run"),
    "Experiment": ("increment.kernel.experiment", "Experiment"),
    "SourceReference": ("increment.kernel.experiment", "SourceReference"),
    "Resolver": ("increment.kernel.resolver", "Resolver"),
    "Scope": ("increment.kernel.contracts", "Scope"),
    "BoundResolver": ("increment.kernel.resolver", "BoundResolver"),
    "from_frame": ("increment.builders", "from_frame"),
    "render": ("increment.result_envelope", "render"),
    "from_definitions": ("increment.builders", "from_definitions"),
    "from_unit_day_artifact": ("increment.builders", "from_unit_day_artifact"),
    "from_switchback_frame": ("increment.builders", "from_switchback_frame"),
    "from_moments": ("increment.builders", "from_moments"),
    "Runnable": ("increment.kernel.resolver", "Runnable"),
    "runnable": ("increment.kernel.resolver", "runnable"),
    "MetricSpec": ("increment.frame", "MetricSpec"),
    "CupedCovariate": ("increment.semantics.models", "CupedCovariate"),
    "CupedCovariateBinding": ("increment.semantics.models", "CupedCovariateBinding"),
    "MethodSpec": ("increment.semantics.models", "MethodSpec"),
    "Design": ("increment.semantics.design", "Design"),
    "Randomized": ("increment.semantics.design", "Randomized"),
    "Observational": ("increment.semantics.design", "Observational"),
    "Encouragement": ("increment.semantics.design", "Encouragement"),
    "UptakeSpec": ("increment.semantics.design", "UptakeSpec"),
    "ExclusionRestriction": ("increment.semantics.design", "ExclusionRestriction"),
    "AdjustmentSet": ("increment.semantics.design", "AdjustmentSet"),
    "IdentificationGate": ("increment.semantics.design", "IdentificationGate"),
    "Method": ("increment.estimation.engine", "Method"),
    "Definitions": ("increment.semantics.models", "Definitions"),
    "ExperimentMetric": ("increment.semantics.models", "ExperimentMetric"),
    "Metric": ("increment.semantics.models", "Metric"),
    "Winsorization": ("increment.semantics.models", "Winsorization"),
    "AnalysisPlan": ("increment.semantics.models", "AnalysisPlan"),
    "MultiplicitySpec": ("increment.semantics.models", "MultiplicitySpec"),
    "InferenceSpec": ("increment.semantics.models", "InferenceSpec"),
    "EvidenceProvider": ("increment.evidence", "EvidenceProvider"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
