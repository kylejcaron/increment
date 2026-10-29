from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from increment.decide.evaluator import Decisions, FamilyVerdict, ProgramVerdict, Verdict, decide
from increment.decide.family import select_fixed_roster
from increment.declarations import Method
from increment.kernel.contracts import Registration
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import AnalysisPlan, Metric

if TYPE_CHECKING:
    from increment.estimation.inference import Prior


def register(
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    path: Literal["warehouse", "frame", "frame/contrast"] = "warehouse",
    design: Randomized | Encouragement | Observational | None = None,
    configs: Any = None,
    methods: Sequence[Method] | None = None,
    prior: Prior | None = None,
    estimands: Sequence[str] | None = None,
) -> Registration:
    """Compile and hash decision policy before evidence acquisition."""
    from increment.plan import compile_decision_plan

    compiled = compile_decision_plan(
        plan,
        metrics,
        path=path,
        design=design,
        configs=configs,
        methods=methods,
        prior=prior,
        estimands=estimands,
    )
    return Registration.from_plan(compiled)


__all__ = [
    "Decisions",
    "FamilyVerdict",
    "ProgramVerdict",
    "Registration",
    "Verdict",
    "decide",
    "register",
    "select_fixed_roster",
]
