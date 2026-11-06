"""Public raw-state dispatch; all numerical evidence and inversion use the kernels."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, overload

from increment.estimation._sequential_likelihood import GaussianState
from increment.estimation.advisory import ReadoutAdvisory
from increment.estimation.decision_types import (
    AsymptoticSequentialEvidence,
    DecisionComputation,
    EValueEvidence,
    TestEvidence,
    exact_fraction,
    sequential_hypothesis_key,
)
from increment.estimation.diagnostics import (
    COVARIATE_BALANCE_ALPHA,
    covariate_balance_boundary,
    covariate_balance_crosses_boundary,
)
from increment.estimation.results import LiftEstimate, display_estimate
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
    ADJUSTED_LAWS,
    BINARY_METRIC_TYPES,
    RATIO_LAWS,
    SCALAR_METRIC_TYPES,
    ScalarMeanModel,
    retained_dimension,
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
    from collections.abc import Iterable, Mapping

    from increment.decision import HypothesisKey
    from increment.estimation.decision_types import AdvisoryKey


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
    state_map = {(state.metric, state.group_id, state.segment): state for state in snapshot.states}
    models = {model.metric: model for model in registration.models}
    evaluated_results: list[SequentialResult] = []
    for cell in registration.roster:
        key_c = (cell.metric, registration.control_group, cell.segment)
        key_t = (cell.metric, cell.group_id, cell.segment)
        c = state_map.get(key_c)
        t = state_map.get(key_t)
        if c is None:
            c = snapshot.arm(*key_c)
        if t is None:
            t = snapshot.arm(*key_t)
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


def _binary64_sqrt(value: Fraction) -> float:
    """``sqrt(value)`` for a nonnegative rational whose own magnitude may lie outside
    binary64 though its root does not; ``OverflowError`` when the root does too."""
    half = (value.numerator.bit_length() - value.denominator.bit_length()) // 2
    return math.ldexp(math.sqrt(value / Fraction(4) ** half), half)


def _covariate_imbalance_advisory(result: AsymptoticSequentialResult) -> ReadoutAdvisory | None:
    """The advisory of an adjusted cell whose covariate difference crosses its
    count-clock boundary at this look; none while the cell has no inference, or when
    its numbers are past the float range.

    The advisory feeds nothing, so the cell's sets and evidence are unchanged.
    """
    checkpoint, model = result.checkpoint, result.checkpoint.model
    if not isinstance(model, ScalarMeanModel) or model.law not in ADJUSTED_LAWS:
        return None
    control, treatment = checkpoint.control.kernel(), checkpoint.treatment.kernel()
    if not (
        result.bounds.available
        and isinstance(control, GaussianState)
        and isinstance(treatment, GaussianState)
    ):
        return None
    x = retained_dimension(model.law) - 1
    difference = treatment.mean[x] - control.mean[x]
    count = control.n + treatment.n
    if not covariate_balance_crosses_boundary(
        difference=difference,
        scatter_control=control.scatter[x][x],
        n_control=control.n,
        scatter_treatment=treatment.scatter[x][x],
        n_treatment=treatment.n,
        count=count,
        rho=model.rho,
        start_count=model.start_count,
    ):
        return None
    from increment.estimation.engine import _WARNINGS

    # A crossing has a positive variance and a nonzero difference. The variance
    # scales with the covariate's squared units and the boundary with 1/rho^2, so
    # roots are taken of the exact ratios; a number past the float range abstains.
    variance = treatment.scatter[x][x] / treatment.n**2 + control.scatter[x][x] / control.n**2
    try:
        magnitude = _binary64_sqrt(difference * difference / variance)
        numbers = {
            "mean_control": float(control.mean[x]),
            "mean_treatment": float(treatment.mean[x]),
            "difference": float(difference),
            "standard_error": _binary64_sqrt(variance),
            "statistic": magnitude if difference > 0 else -magnitude,
            # The statistic flags past this threshold, in the same standard errors.
            "boundary": _binary64_sqrt(covariate_balance_boundary(count, model.rho)),
        }
    except OverflowError:
        return None
    groups = (checkpoint.control.group_id, checkpoint.treatment.group_id)
    context: dict[str, object] = {
        "metric_name": checkpoint.cell.metric,
        "control_group": groups[0],
        "treatment_group": groups[1],
        # A registration retains the covariate's values, not its name.
        "covariate": None,
        **numbers,
        "count": count,
        "alpha": COVARIATE_BALANCE_ALPHA,
        "constant_arms": tuple(
            group
            for group, arm in zip(groups, (control, treatment), strict=True)
            if not arm.scatter[x][x]
        ),
    }
    spec = _WARNINGS["estimation.cuped.covariate_imbalance"]
    return ReadoutAdvisory(code=spec.code, context=context, message=spec.render(**context))


def _sequential_advisories(
    results: Iterable[SequentialResult],
) -> dict[AdvisoryKey, tuple[ReadoutAdvisory, ...]]:
    """Each cell's advisories at the inversion its result reports."""
    advisories: dict[AdvisoryKey, tuple[ReadoutAdvisory, ...]] = {}
    for result in results:
        if isinstance(result, AsymptoticSequentialResult):
            advisory = _covariate_imbalance_advisory(result)
            if advisory is not None:
                advisories[sequential_hypothesis_key(result.checkpoint.cell)] = (advisory,)
    return advisories


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
        results=results,
        evidence=evidence,
        failures={},
        sequential_snapshot=snapshot,
        advisories=_sequential_advisories(evaluated_results),
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
) -> tuple[list[LiftEstimate], Mapping[AdvisoryKey, tuple[ReadoutAdvisory, ...]]]:
    """Evaluate and reinvert the family at its stopped state, with each hypothesis' advisories.

    The advisories are those of the reported inversion: a selected cell's set can be
    available at its reporting level though unavailable at its registered allocation.
    Triggered family guarantees assume correct triggering and looks, stops and
    freezes adapted to the triggered joint filtration, not informed by early
    assigned results, including the same metric; see ``sequential_family_guarantee``.
    """
    from increment.semantics.sequential import sequential_family_rule

    procedure = sequential_family_rule(inference.registration.models, inference.registration.roster)
    from increment.estimation.family import select_sequential_family

    computation = estimate_sequential(snapshot, inference)
    family = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
        if row.require_sequential_result().checkpoint.cell.family
    ]
    if not family:
        return list(computation.results), computation.advisories
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
                    "family_procedure": procedure,
                    "family_axes": ("metric", "arm", "segment")
                    if cell.segment
                    else ("metric", "arm"),
                    "family_q": float(inference.registration.q),
                    "family_threshold": outcome.realized_threshold,
                    "family_guarantee": outcome.guarantee,
                    "family_nominal_alpha": float(nominal_alpha),
                    "family_size": outcome.n_family,
                }
            )
        results.append(row)
    return results, _sequential_advisories(row.require_sequential_result() for row in results)
