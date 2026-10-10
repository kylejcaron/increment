"""Public raw-state dispatch; all numerical evidence and inversion use the kernels."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, overload

from increment.estimation.decision_types import (
    AsymptoticSequentialEvidence,
    DecisionComputation,
    EValueEvidence,
    TestEvidence,
    exact_fraction,
    sequential_hypothesis_key,
)
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import (
    ASYMPTOTIC_PROCEDURE_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
)
from increment.estimation.sequential_result import (
    AsymptoticSequentialResult,
    SequentialCheckpoint,
    SequentialInferenceResult,
    SequentialResult,
    _point,
    checkpoint_bounds,
    checkpoint_certificate,
    checkpoint_mean_bounds,
)
from increment.semantics.sequential import (
    BINARY_METRIC_TYPES,
    RATIO_LAWS,
    SCALAR_METRIC_TYPES,
    ScalarMeanModel,
)
from increment.sequential_state import (
    SequentialSnapshot,
    model_adjustment,
    registration_id,
    require_public_laws,
    sequential_refuse,
    validate_sequential_transform,
)

if TYPE_CHECKING:
    from increment.decision import HypothesisKey


def evaluate_checkpoint(
    checkpoint: SequentialCheckpoint, *, alpha: Fraction, ceiling: Fraction | None = None
) -> SequentialResult:
    """Evaluate and invert exactly the same retained state and predictive law.

    ``ceiling`` bounds decision_alpha from above; it defaults to the
    registered cell allocation (the ordinary, unselected case). Post-
    selection FCR reinversion (reinvert_selected) passes the overall test's
    own nominal_alpha instead, widening the ceiling for a selected row --
    this function and the result's own replay validator do not themselves
    distinguish selected from unselected; only the caller decides.
    """
    resolved_ceiling = checkpoint.cell.alpha if ceiling is None else ceiling
    if not 0 < alpha <= resolved_ceiling:
        sequential_refuse("source.invalid", "decision alpha cannot exceed its ceiling")
    if isinstance(checkpoint.model, ScalarMeanModel):
        return AsymptoticSequentialResult(
            construction=checkpoint.model.construction,
            checkpoint=checkpoint,
            bounds=checkpoint_mean_bounds(checkpoint, alpha),
            decision_alpha=alpha,
            alpha_ceiling=resolved_ceiling,
            point_reason=_point(checkpoint)[1],
        )
    return SequentialInferenceResult(
        checkpoint=checkpoint,
        certificate=checkpoint_certificate(checkpoint),
        bounds=checkpoint_bounds(checkpoint, alpha),
        decision_alpha=alpha,
        alpha_ceiling=resolved_ceiling,
        point_reason=_point(checkpoint)[1],
    )


def _outward(value: Fraction | None, *, lower: bool) -> float | None:
    if value is None:
        return None
    exact = value - 1
    try:
        result = float(exact)
    except OverflowError:
        return None
    if not math.isfinite(result):
        return None
    if (lower and Fraction(result) > exact) or (not lower and Fraction(result) < exact):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result if math.isfinite(result) else None


def display_estimate(result: SequentialResult):
    from increment.estimation.results import Estimate

    point, _ = _point(result.checkpoint)
    bounds = result.bounds
    lift = None
    if point is not None:
        # The full exact geometry remains in sequential_result even when a
        # finite display cannot represent an endpoint or the alpha allocation.
        lower, upper = _outward(bounds.lower, lower=True), _outward(bounds.upper, lower=False)
        alpha = float(bounds.alpha)
        open_side = (
            "upper"
            if upper is None and lower is not None
            else "lower"
            if lower is None and upper is not None
            else None
        )
        if (
            not (
                isinstance(result, AsymptoticSequentialResult)
                and len(result.bounds.components) != 1
            )
            and not bounds.empty
            and (lower is not None or upper is not None)
            and 0 < alpha < 1
            and 0 < 1 - alpha < 1
            and alpha / 2 > 0
        ):
            lift = Estimate(
                value=point, lb=lower, ub=upper, open_side=open_side, alpha=alpha, level=1 - alpha
            )
        else:
            lift = Estimate(value=point)
    return lift


def _note(result: SequentialResult) -> str | None:
    """A row's note: the evaluated set's or certificate's reason, else its point's."""
    return (
        result.bounds.reason
        if isinstance(result, AsymptoticSequentialResult)
        else result.certificate.reason
    ) or result.point_reason


def result_from_sequential(
    result: SequentialResult,
    *,
    label: str,
    discovery: bool | None = None,
    family_threshold: float | None = None,
    family_nominal_alpha: float | None = None,
) -> LiftEstimate:
    model = result.checkpoint.model
    require_public_laws((model,), "LiftEstimate construction")
    cell = result.checkpoint.cell
    lift = display_estimate(result)
    return LiftEstimate(
        metric=cell.metric,
        group_id=cell.group_id,
        analysis_population=result.checkpoint.population,
        estimand=cell.estimand,
        # The retained state already carries the adjustment: a predeclared
        # scalar or the joint moments the asymptotic route fits from.
        method="cuped" if model_adjustment(model) else "unadjusted",
        method_role="decision",
        inference=label,
        alternative=cell.alternative,
        null_lift=float(cell.null_lift),
        reference_kind="sequential",
        lift=lift,
        sequential_result=result,
        scale="linear",
        discovery=discovery,
        family_threshold=family_threshold,
        family_nominal_alpha=family_nominal_alpha,
        note=_note(result),
    )


def validate_engine_request(snapshot, metrics, *, alpha, alternative, null_lift=None):
    """Do not silently ignore effect-policy arguments at the raw-state dispatch."""
    models = {
        model.metric: model
        for model in snapshot.registration.models
        if model.observable == "outcome"
    }
    if {metric.name for metric in metrics} != set(models):
        sequential_refuse(
            "source.invalid", "engine metrics differ from the registered outcome roster"
        )
    for metric in metrics:
        model = models[metric.name]
        if model.law in ("scalar_mean", "adjusted_mean") and metric.type not in SCALAR_METRIC_TYPES:
            sequential_refuse(
                "route.unsupported",
                "scalar mean inference requires a mean, conversion or retention metric",
            )
        validate_sequential_transform(metric)
        if (metric.type == "ratio") != (model.law in ("gaussian_ratio", *RATIO_LAWS)):
            sequential_refuse("source.invalid", "metric target and observation model disagree")
        if metric.type in BINARY_METRIC_TYPES and model.law not in (
            "bernoulli",
            "scalar_mean",
            "adjusted_mean",
        ):
            sequential_refuse(
                "source.invalid", "binary outcomes require Bernoulli or scalar mean registration"
            )
    for cell in snapshot.registration.roster:
        if (
            (alpha is not None and alpha != float(cell.alpha))
            or (alternative is not None and alternative != cell.alternative)
            or (null_lift is not None and null_lift != float(cell.null_lift))
        ):
            sequential_refuse(
                "source.invalid", "engine effect policy differs from pre-data registration"
            )


@overload
def _evaluate_sequential_diagnostic(
    snapshot: SequentialSnapshot,
    inference: AsymptoticMean,
) -> tuple[AsymptoticSequentialResult, ...]: ...


@overload
def _evaluate_sequential_diagnostic(
    snapshot: SequentialSnapshot,
    inference: AlwaysValid,
) -> tuple[SequentialInferenceResult, ...]: ...


@overload
def _evaluate_sequential_diagnostic(
    snapshot: SequentialSnapshot,
    inference: MixedFamily,
) -> tuple[SequentialResult, ...]: ...


def _evaluate_sequential_diagnostic(
    snapshot: SequentialSnapshot,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
) -> tuple[SequentialResult, ...]:
    """Evaluate numerical sequential checkpoints without certifying evidence.

    This private route is intentionally usable for research diagnostics such as
    Gaussian approximations.  It never constructs ``EValueEvidence`` or a
    ``DecisionComputation`` and therefore cannot accidentally advertise a
    non-admitted anytime-valid law as public evidence.
    """
    registration = inference.registration
    if snapshot.registration_id != registration_id(registration):
        sequential_refuse("source.invalid", "snapshot and runtime registration disagree")
    frozen_map = {
        checkpoint.cell: checkpoint.model_copy(update={"status": "frozen"})
        for checkpoint in snapshot.frozen
    }
    models = {m.metric: m for m in registration.models}
    evaluated_results: list[SequentialResult] = []
    for cell in registration.roster:
        c = snapshot.arm(cell.metric, registration.control_group, cell.segment)
        t = snapshot.arm(cell.metric, cell.group_id, cell.segment)
        checkpoint = frozen_map.get(cell) or SequentialCheckpoint(
            registration_id=snapshot.registration_id,
            prefix_id=snapshot.prefix_id,
            filtration_id=registration.reveal.filtration_id,
            cell=cell,
            model=models[cell.metric],
            control=c,
            treatment=t,
            revealed_units=len(snapshot.records),
            status="current" if c.n and t.n else "missing",
            population=snapshot.population,
        )
        alpha = inference.allocated_alpha(cell.alpha, checkpoint.revealed_units)
        evaluated_results.append(evaluate_checkpoint(checkpoint, alpha=alpha))
    return tuple(evaluated_results)


def estimate_sequential(
    snapshot: SequentialSnapshot,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
) -> DecisionComputation[LiftEstimate]:
    """Evaluate the complete roster with exact or asymptotic typed evidence."""
    registration = inference.registration
    has_scalar_mean = any(isinstance(m, ScalarMeanModel) for m in registration.models)
    if has_scalar_mean != isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES):
        sequential_refuse("source.invalid", "scalar mean requires its distinct runtime policy")
    require_public_laws(registration.models, "sequential inference")
    evaluated_results = _evaluate_sequential_diagnostic(snapshot, inference)
    results: list[LiftEstimate] = []
    evidence: dict[HypothesisKey, TestEvidence] = {}
    for evaluated in evaluated_results:
        # A mixed roster's own displayed regime varies per cell (asymptotic
        # scalar-mean ITT vs. exact Bernoulli uptake): MixedFamily.label is a
        # fixed public dispatch tag, not the per-row validity regime.
        label = (
            (
                "asymptotic_mean"
                if isinstance(evaluated, AsymptoticSequentialResult)
                else "always_valid"
            )
            if isinstance(inference, MixedFamily)
            else inference.label
        )
        row = result_from_sequential(evaluated, label=label)
        hypothesis = sequential_hypothesis_key(evaluated.checkpoint.cell)
        results.append(row)
        if isinstance(evaluated, AsymptoticSequentialResult):
            evidence[hypothesis] = AsymptoticSequentialEvidence(
                hypothesis=hypothesis,
                method=row.method,
                result=evaluated,
            )
            continue
        evidence[hypothesis] = EValueEvidence(
            hypothesis=hypothesis,
            method=row.method,
            log_e=evaluated.log_e,
            process="raw_likelihood_v1",
            checkpoint=evaluated.checkpoint,
            certificate=evaluated.certificate,
        )
    return DecisionComputation[LiftEstimate](
        results=results, evidence=evidence, failures={}, sequential_snapshot=snapshot
    )


def require_selected_widening(
    result: SequentialResult,
    *,
    discovery: bool | None,
    family_threshold: float | None,
    family_nominal_alpha: float | None,
) -> None:
    """A row's decision alpha may pass its registered cell allocation only as a
    selected family member, and never past ``min(family_threshold, family_nominal_alpha)``."""
    if result.decision_alpha == result.checkpoint.cell.alpha:
        return
    if not discovery:
        sequential_refuse(
            "source.invalid",
            "only a selected row may carry a decision alpha past its registered cell allocation",
        )
    if family_threshold is None or family_nominal_alpha is None:
        sequential_refuse(
            "source.invalid",
            "a widened decision alpha requires a recorded family threshold and nominal alpha",
        )
    permitted = min(family_threshold, family_nominal_alpha)
    if float(result.decision_alpha) > permitted * (1 + 1e-9):
        sequential_refuse(
            "source.invalid",
            "reinverted decision alpha exceeds the family's realized threshold or nominal alpha",
        )


def reinvert_selected(
    row: LiftEstimate, alpha: Fraction | float, *, ceiling: Fraction | float
) -> LiftEstimate:
    """Selected-FCR inversion at the same stopped sufficient state, for
    either law, at up to the overall test's own nominal level -- not
    clamped back to the registered per-cell allocation, which an explicit
    registration may set above or below ``q/m``. The display interval, the
    result and the row note all come from the new inversion; the checkpoint
    is the one the selection read."""
    previous = row.sequential_result
    if previous is None:
        sequential_refuse("source.invalid", "selected interval lacks its sequential checkpoint")
    evaluated = evaluate_checkpoint(
        previous.checkpoint, alpha=exact_fraction(alpha), ceiling=exact_fraction(ceiling)
    )
    return row.model_copy(
        update={
            "lift": display_estimate(evaluated),
            "sequential_result": evaluated,
            "note": _note(evaluated),
        }
    )


def selected_snapshot_results(
    snapshot: SequentialSnapshot,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
    *,
    nominal_alpha: Fraction | float,
) -> list[LiftEstimate]:
    """Evaluate the retained family and invert selected cells at their stopped state."""
    from increment.estimation.family import select_sequential_family

    computation = estimate_sequential(snapshot, inference)
    family = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
        if row.require_sequential_result().checkpoint.cell.family
    ]
    if not family:
        return list(computation.results)
    outcome = select_sequential_family(
        family,
        inference.registration.q,
        inference,
        nominal_alpha,
        computation=computation,
    )
    results = []
    for row in computation.results:
        cell = row.require_sequential_result().checkpoint.cell
        if cell.family:
            selected = sequential_hypothesis_key(cell) in outcome.selected
            if selected and outcome.fcr_alpha is not None:
                row = reinvert_selected(row, outcome.fcr_alpha, ceiling=nominal_alpha)
            row = row.model_copy(
                update={
                    "discovery": selected,
                    "family_axes": ("metric", "arm", "segment")
                    if cell.segment
                    else ("metric", "arm"),
                    "family_q": float(inference.registration.q),
                    "family_threshold": outcome.realized_threshold,
                    "family_guarantee": outcome.guarantee,
                    "family_nominal_alpha": float(nominal_alpha),
                }
            )
        results.append(row)
    return results
