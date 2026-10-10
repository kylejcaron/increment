from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from increment._literals import PreferredDirection, Role
from increment.errors import CodedError
from increment.estimation.engine import Method, _estimate_lift, estimate_lift
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.inference import LiftGuardError
from increment.estimation.quantile import estimate_quantile_lift_computation
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily
from increment.readouts._common import _raise, _runtime_method_roles, _warn
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

# Only numeric refusals proven local to a method cell are softened;
# capability, request, source, and wire refusals abort the readout.
_CELL_ESTIMATION_GUARD_CODES = frozenset(
    {
        "estimation.cuped.covariate_zero_variance",
        "estimation.variance.ratio_moments_nonpositive_denominator_mean",
    }
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import DecisionComputation
    from increment.estimation.inference import Prior
    from increment.semantics.design import Randomized
    from increment.semantics.models import Metric


def _emit_captured_lift_warnings(
    metric_name: str,
    captured: list[Any],
    advisory_seen: set[tuple[str, str]],
) -> None:
    """Replay estimator warnings while deduplicating known per-metric advisories.

    A contrast-level advisory repeats once per treatment arm; the
    denominator-skew warning names its arm, so it repeats once per flagged
    arm instead of once per contrast.
    """
    for record in captured:
        message = str(record.message)
        key: tuple[str, str] | None
        if "is open-ended (" in message:
            key = (metric_name, "open-ended-sequential")
        elif "total clusters across arms" in message:
            key = (metric_name, "clustered-small-k")
        elif getattr(record.message, "code", None) == "estimation.engine.ratio_denominator_skew":
            key = (metric_name, f"ratio-denominator-skew:{record.message.context['arm']}")
        else:
            key = None
        if key is not None:
            if key in advisory_seen:
                continue
            advisory_seen.add(key)
        warnings.warn(record.message, record.category, stacklevel=3)


def _lift_cells_missing_control(  # noqa: PLR0913
    *,
    metric: Metric,
    by_group: Mapping[str, Mapping[str, Any]],
    control_group: str,
    family: bool,
    summary: list[Mapping[str, Any]],
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
    cluster: str | None,
    advisory_seen: set[tuple[str, str]],
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    if family:
        from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

        failures: dict[Any, Any] = {}
        for group_id in by_group:
            if group_id == control_group:
                continue
            hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                "estimation.engine.control_missing",
                {
                    "metric": metric.name,
                    "group_id": group_id,
                    "reason": "no_control_arm",
                },
            )
        return [], [], DecisionComputation(results=(), evidence={}, failures=failures)
    # Preserve estimate_lift's normal missing-control error and wording.
    captured: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            computation = estimate_lift(
                metrics=[metric],
                summary=summary,
                control_group=control_group,
                methods=methods,
                prior=prior,
                alpha=alpha,
                alternative=alternative,
                inference=inference,
                null_lift=null_lift,
                null_abs=null_abs,
                preferred_direction=preferred_direction,
                method_roles=method_roles,
                cluster=cluster,
            )
            fallback = list(computation.results)
    except Exception:
        _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
        raise
    else:
        _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
        return fallback, [], computation


def _append_lift_cell_failure(
    *,
    results: list[LiftEstimate],
    computations: list[DecisionComputation[LiftEstimate]],
    metric: Metric,
    group_id: str,
    method: Method,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
    alternative: str,
    code: str,
    context: Mapping[str, object],
) -> None:
    from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

    method_role = (method_roles or {}).get(method.name, "decision")
    failure_context = {**context, "method": method.name}
    if method_role == "decision":
        hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
        computations.append(
            DecisionComputation(
                results=(),
                evidence={},
                failures={
                    hypothesis: DecisionFailure(hypothesis, code, failure_context),
                },
            )
        )
    else:
        results.append(
            LiftEstimate(
                metric=metric.name,
                group_id=group_id,
                method=method.name,
                estimand="itt",
                value_scale="relative",
                alternative=alternative,
                method_role="sensitivity",
                failure_code=code,
                failure_context=failure_context,
            )
        )


def _estimate_lift_cells(  # noqa: PLR0913
    *,
    metric: Metric,
    summary: list[Mapping[str, Any]],
    control_group: str,
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
    cluster: str | None,
    advisory_seen: set[tuple[str, str]] | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    family: bool = False,
    route_alpha: float | None = None,
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    """Estimate randomized whole-window cells independently.

    ``estimate_lift`` accepts a whole metric slice and therefore aborts its
    entire result when one treatment arm hits an expected ``LiftGuardError``.
    A readout has a denser contract: a bad arm/method cell is warned and
    excluded while its siblings remain usable. Only ``LiftGuardError`` and
    the explicitly classified numeric guard codes below are cell-local;
    all other coded and non-coded errors propagate.
    """
    if advisory_seen is None:
        advisory_seen = set()
    from increment.estimation.engine import _df_to_arms
    from increment.sources import _validate_moment_experiment_identity

    own = [row for row in summary if str(row["metric"]) == metric.name]
    _df_to_arms(own)  # single shared ingress: refuses duplicate (metric, group_id)
    _validate_moment_experiment_identity(own)
    by_group: dict[str, Mapping[str, Any]] = {str(row["group_id"]): row for row in own}
    control = by_group.get(control_group)
    if control is None:
        return _lift_cells_missing_control(
            metric=metric,
            by_group=by_group,
            control_group=control_group,
            family=family,
            summary=summary,
            methods=methods,
            prior=prior,
            alpha=alpha,
            alternative=alternative,
            inference=inference,
            null_lift=null_lift,
            null_abs=null_abs,
            preferred_direction=preferred_direction,
            method_roles=method_roles,
            cluster=cluster,
            advisory_seen=advisory_seen,
        )

    effective_methods = methods if methods is not None else [Method(name="unadjusted")]
    results: list[LiftEstimate] = []
    refused: list[tuple[str, str, str, str]] = []
    computations: list[DecisionComputation[LiftEstimate]] = []
    for group_id, treatment in by_group.items():
        if group_id == control_group:
            continue
        for method in effective_methods:
            captured: list[Any] = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    computation = _estimate_lift(
                        metrics=[metric],
                        summary=[control, treatment],
                        control_group=control_group,
                        methods=[method],
                        prior=prior,
                        alpha=alpha,
                        alternative=alternative,
                        inference=inference,
                        null_lift=null_lift,
                        null_abs=null_abs,
                        preferred_direction=preferred_direction,
                        method_roles=method_roles,
                        cluster=cluster,
                        route_alpha=route_alpha,
                    )
                    cell_results = computation.results
                    computations.append(computation)
                    if computation.failures and not cell_results:
                        for failure in computation.failures.values():
                            reason = failure.display()
                            refused.append((metric.name, group_id, method.name, reason))
                            _warn(
                                "readouts.run.cell_refused",
                                metric_name=metric.name,
                                group_id=group_id,
                                method_name=method.name,
                                reason=reason,
                                stacklevel=3,
                            )
            except LiftGuardError as exc:
                _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
                reason = str(exc)
                refused.append((metric.name, group_id, method.name, reason))
                _append_lift_cell_failure(
                    results=results,
                    computations=computations,
                    metric=metric,
                    group_id=group_id,
                    method=method,
                    alternative=alternative,
                    method_roles=method_roles,
                    code="estimation.engine.lift_guard",
                    context={
                        "metric": metric.name,
                        "group_id": group_id,
                        "reason": exc.reason,
                        "display": reason,
                    },
                )
                _warn(
                    "readouts.run.cell_refused",
                    metric_name=metric.name,
                    group_id=group_id,
                    method_name=method.name,
                    reason=reason,
                    stacklevel=3,
                )
            except CodedError as exc:
                _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
                if exc.code not in _CELL_ESTIMATION_GUARD_CODES:
                    raise
                reason = str(exc)
                refused.append((metric.name, group_id, method.name, reason))
                _append_lift_cell_failure(
                    results=results,
                    computations=computations,
                    metric=metric,
                    group_id=group_id,
                    method=method,
                    alternative=alternative,
                    method_roles=method_roles,
                    code=exc.code,
                    context=exc.context,
                )
                _warn(
                    "readouts.run.cell_refused",
                    metric_name=metric.name,
                    group_id=group_id,
                    method_name=method.name,
                    reason=reason,
                    stacklevel=3,
                )
            except Exception:
                _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
                raise
            else:
                _emit_captured_lift_warnings(metric.name, captured, advisory_seen)
                results.extend(cell_results)
    return results, refused, _merge_decision_computations(computations)


def _raise_if_all_lift_cells_refused(
    refused: Sequence[tuple[str, str, str, str]],
    computations: Sequence[DecisionComputation[Any]],
) -> None:
    """Refuse an all-failed run with the producer's keyed failure evidence."""
    if not refused:
        return
    from increment.decision import ArmHypothesisKey

    failures = [
        {
            "metric": key.metric,
            "group_id": key.group_id,
            "estimand": key.estimand,
            "method": failure.context.get("method"),
            "code": failure.code,
            "context": dict(failure.context),
        }
        for computation in computations
        for key, failure in computation.failures.items()
        if isinstance(key, ArmHypothesisKey)
    ]
    failures.sort(
        key=lambda failure: (
            failure["metric"],
            failure["group_id"],
            failure["method"] or "",
            failure["code"],
        )
    )
    _raise("readout.estimate_lift_every", failures=failures)


def _estimate_pass(  # noqa: PLR0913
    src: MomentSource,
    metric: Metric,
    evidence: Any,
    test: Any,
    config: ResolvedMetricConfig,
    design: Randomized | Encouragement,
    *,
    alpha_for: float,
    role: Role | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]] | None,
    retry: Literal["cells", "family", "whole", "whole_asof"] = "cells",
    methods: list[Method],
    route_alpha: float | None = None,
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    """Run one randomized estimation pass with role, alpha, and retry policy.

    `evidence` is what the caller already read for this metric: total moment
    rows for a moments metric, the native unit frame for a quantile metric.
    """
    method_roles = _runtime_method_roles(methods)
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        from increment.winsor import winsor_refuse

        if not evidence:
            from increment.estimation.decision_types import DecisionComputation

            return [], [], DecisionComputation(results=(), evidence={}, failures={})

        if "_winsor_raw_state" not in evidence[0]:
            winsor_refuse(
                "raw_state_required", "Percentile inference requires the loaded raw cutoff pool."
            )
        computation = estimate_lift(
            metrics=[metric],
            summary=(),
            control_group=design.control_group,
            methods=methods,
            prior=config.prior,
            alpha=alpha_for,
            alternative=test.alternative,
            inference=inference,
            null_lift=getattr(test, "null_lift", 0.0),
            null_abs=getattr(test, "null_abs", None),
            preferred_direction=metric.declared_preferred_direction,
            cluster=src.context.cluster,
            method_roles=method_roles,
            raw_outcomes={metric.name: evidence[0]["_winsor_raw_state"]},
            winsor_references=evidence[0]["_winsor_references"],
        )
        return (
            [row.model_copy(update={"role": role}) for row in computation.results],
            [],
            computation,
        )
    if getattr(metric, "type", None) == "quantile":
        if methods == [] and retry != "family":
            return [], [], _merge_decision_computations([])
        computation = estimate_quantile_lift_computation(
            src,
            metric,
            design.control_group,
            methods=methods,
            prior=config.prior,
            alpha=alpha_for,
            method_roles=method_roles,
            inference=inference,
            unit_rows=evidence,
        )
        return (
            [row.model_copy(update={"role": role}) for row in computation.results],
            [],
            computation,
        )

    if retry in ("whole", "whole_asof"):
        computation = _estimate_lift(
            metrics=[metric],
            summary=evidence or [],
            control_group=design.control_group,
            methods=methods,
            prior=config.prior,
            alpha=alpha_for,
            alternative=test.alternative,
            inference=inference,
            null_lift=getattr(test, "null_lift", 0.0),
            null_abs=getattr(test, "null_abs", None),
            preferred_direction=metric.declared_preferred_direction,
            method_roles=method_roles,
            cluster=src.context.cluster,
            route_alpha=route_alpha,
        )
        return (
            [row.model_copy(update={"role": role}) for row in computation.results],
            [],
            computation,
        )

    rows_est, refused, computation = _estimate_lift_cells(
        metric=metric,
        summary=evidence or [],
        control_group=design.control_group,
        methods=methods,
        prior=config.prior,
        alpha=alpha_for,
        alternative=test.alternative,
        inference=inference,
        null_lift=getattr(test, "null_lift", 0.0),
        null_abs=getattr(test, "null_abs", None),
        preferred_direction=metric.declared_preferred_direction,
        cluster=src.context.cluster,
        method_roles=method_roles,
        advisory_seen=advisory_seen,
        family=retry == "family",
        route_alpha=route_alpha,
    )
    return [row.model_copy(update={"role": role}) for row in rows_est], refused, computation
