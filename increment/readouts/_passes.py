from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from increment._frame_validation import SCALAR_COVARIATE_REFUSALS
from increment._literals import PreferredDirection, Role
from increment.errors import CodedError
from increment.estimation.engine import (
    CELL_ESTIMATION_GUARD_CODES as _CELL_ESTIMATION_GUARD_CODES,
)
from increment.estimation.engine import Method, _estimate_lift, estimate_lift
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.inference import LiftGuardError
from increment.estimation.quantile import estimate_quantile_lift_computation
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily
from increment.readouts._common import _raise, _runtime_method_roles, _warn
from increment.readouts._metric_rows import MetricRows
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import DecisionComputation
    from increment.estimation.inference import Prior
    from increment.semantics.design import Randomized
    from increment.semantics.models import Metric


def _emit_captured_lift_warnings(
    captured: list[Any],
    advisory_seen: set[tuple[object, ...]],
) -> None:
    """Replay estimator warnings, deduplicating explicitly keyed warnings and
    denominator-skew warnings once per metric and flagged arm.
    """
    for record in captured:
        message = record.message
        dedupe_key = getattr(message, "dedupe_key", None)
        if dedupe_key is None and getattr(message, "code", None) == (
            "estimation.engine.ratio_denominator_skew"
        ):
            context = message.context
            dedupe_key = (message.code, context["metric"], context["arm"])
        if dedupe_key is not None:
            if dedupe_key in advisory_seen:
                continue
            advisory_seen.add(dedupe_key)
        warnings.warn(record.message, record.category, stacklevel=3)


def _method_unit_frame(
    evidence: Any, method: Method, fallback: Sequence[Any]
) -> tuple[Any, Mapping[str, str]]:
    """Return acquired evidence and the actual source column for each method alias."""
    covariates = method.covariates or tuple(fallback)
    source_columns = dict(getattr(evidence, "covariate_columns", ()))
    requested = tuple(item.name for item in covariates)
    missing = tuple(name for name in requested if name not in source_columns)
    if missing:
        from increment._analysis_config import _raise as raise_config_refusal

        raise_config_refusal("facade.analysis_config.cuped_linear_covariates", metric=method.name)
    return evidence.unit_frame, {name: source_columns[name] for name in requested}


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
    advisory_seen: set[tuple[object, ...]],
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
        _emit_captured_lift_warnings(captured, advisory_seen)
        raise
    else:
        _emit_captured_lift_warnings(captured, advisory_seen)
        return fallback, [], computation


def _append_lift_cell_failure(
    *,
    results: list[LiftEstimate],
    computations: list[DecisionComputation[LiftEstimate]] | None,
    metric: Metric,
    group_id: str,
    method: Method,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
    alternative: str,
    code: str,
    context: Mapping[str, object],
    estimand: Literal["itt", "late"] = "itt",
) -> None:
    from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

    method_role = (method_roles or {}).get(method.name, "decision")
    failure_context = {**context, "method": method.name}
    if method_role == "decision" and computations is not None:
        hypothesis = ArmHypothesisKey(metric.name, group_id, estimand)
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
                estimand=estimand,
                value_scale="absolute" if estimand == "late" else "relative",
                alternative=alternative,
                method_role=method_role,
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
    advisory_seen: set[tuple[object, ...]] | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    family: bool = False,
    family_member: bool = False,
    route_tail: float | None = None,
    evidence: MetricRows | None = None,
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
            if method.variance_reduction == "cuped":
                if evidence is None:
                    raise AssertionError("Scalar estimation requires method-scoped evidence.")
                family_rows = evidence.rows_for_method(method)
                family_own = [row for row in family_rows if str(row["metric"]) == metric.name]
                family_by_group = {str(row["group_id"]): row for row in family_own}
                method_control = family_by_group.get(control_group)
                method_treatment = family_by_group.get(group_id)
                if method_control is None or method_treatment is None:
                    continue
            else:
                method_control, method_treatment = control, treatment
            captured: list[Any] = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    computation = _estimate_lift(
                        metrics=[metric],
                        summary=[method_control, method_treatment],
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
                        family_member=family_member,
                        route_tail=route_tail,
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
                _emit_captured_lift_warnings(captured, advisory_seen)
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
                _emit_captured_lift_warnings(captured, advisory_seen)
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
                _emit_captured_lift_warnings(captured, advisory_seen)
                raise
            else:
                _emit_captured_lift_warnings(captured, advisory_seen)
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
    if failures and all(
        isinstance(failure["code"], str) and failure["code"].startswith("estimation.cuped_linear.")
        for failure in failures
    ):
        return
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
    advisory_seen: set[tuple[object, ...]] | None,
    retry: Literal["cells", "family", "whole", "whole_asof"] = "cells",
    methods: list[Method],
    family_member: bool = False,
    route_tail: float | None = None,
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    method_roles = _runtime_method_roles(methods)
    available = []
    failed_rows: list[LiftEstimate] = []
    failures: list[DecisionComputation[LiftEstimate]] = []
    for method in methods:
        if method.variance_reduction == "cuped":
            if not isinstance(evidence, MetricRows):
                raise AssertionError(
                    "Scalar estimation requires method-scoped MetricRows evidence."
                )
            try:
                evidence.rows_for_method(method)
            except CodedError as exc:
                if exc.code not in SCALAR_COVARIATE_REFUSALS:
                    raise
                for arm in sorted(evidence.observed_arms - {design.control_group}):
                    _append_lift_cell_failure(
                        results=failed_rows,
                        computations=failures,
                        metric=metric,
                        group_id=arm,
                        method=method,
                        method_roles=method_roles,
                        alternative=test.alternative,
                        code=exc.code,
                        context=exc.context,
                    )
                continue
        available.append(method)
    rows, refused, computation = _estimate_available_pass(
        src,
        metric,
        evidence,
        test,
        config,
        design,
        alpha_for=alpha_for,
        role=role,
        inference=inference,
        advisory_seen=advisory_seen,
        retry=retry,
        methods=available,
        method_roles=method_roles,
        family_member=family_member,
        route_tail=route_tail,
    )
    if not failures and not failed_rows:
        return rows, refused, computation
    failed_rows = [
        row.model_copy(
            update={
                "role": role,
                "preferred_direction": metric.declared_preferred_direction,
                "null_lift": getattr(test, "null_lift", 0.0),
                "null_abs": getattr(test, "null_abs", None),
                "posterior_available": False if config.prior is not None else None,
                "posterior_reason_code": row.failure_code if config.prior is not None else None,
                "posterior_reason_context": row.failure_context
                if config.prior is not None
                else None,
            }
        )
        for row in failed_rows
    ]
    return (
        [*failed_rows, *rows],
        refused,
        _merge_decision_computations([*failures, computation]),
    )


def _estimate_available_pass(  # noqa: PLR0913, PLR0915
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
    advisory_seen: set[tuple[object, ...]] | None,
    retry: Literal["cells", "family", "whole", "whole_asof"] = "cells",
    methods: list[Method],
    method_roles: Mapping[str, Literal["decision", "sensitivity"]],
    family_member: bool = False,
    route_tail: float | None = None,
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    """Run one randomized estimation pass with role, alpha, and retry policy.

    ``family_member`` marks the metric's decision rows as members of the plan's BH family,
    whose p-values are read below their own level; ``route_tail`` is the tail a selected
    member's corrected pass routes conversion counts at (`conversion_route.fcr_route_tail`).

    `evidence` is what the caller already read for this metric: total moment
    rows for a moments metric, the native unit frame for a quantile metric.
    """
    rows_summary = list(evidence.rows) if isinstance(evidence, MetricRows) else evidence
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        from increment.winsor import winsor_refuse

        if not rows_summary:
            from increment.estimation.decision_types import DecisionComputation

            return [], [], DecisionComputation(results=(), evidence={}, failures={})

        if "_winsor_raw_state" not in rows_summary[0]:
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
            raw_outcomes={metric.name: rows_summary[0]["_winsor_raw_state"]},
            winsor_references=rows_summary[0]["_winsor_references"],
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
            unit_rows=evidence.unit_frame if isinstance(evidence, MetricRows) else evidence,
        )
        return (
            [row.model_copy(update={"role": role}) for row in computation.results],
            [],
            computation,
        )

    if isinstance(evidence, MetricRows) and any(
        method.variance_reduction == "cuped_multicovariate" for method in methods
    ):
        from increment.estimation.engine import (
            _cuped_linear_unavailable_row,
            estimate_cuped_linear,
        )

        linear_methods = [
            method for method in methods if method.variance_reduction == "cuped_multicovariate"
        ]
        ordinary_methods = [
            method for method in methods if method.variance_reduction != "cuped_multicovariate"
        ]
        computations = []
        rows_est = []
        refused = []
        linear_cache: dict[str, tuple[Any | None, CodedError | None]] = {}
        if ordinary_methods:
            ordinary, ordinary_refused, ordinary_computation = _estimate_lift_cells(
                metric=metric,
                summary=list(evidence.rows),
                control_group=design.control_group,
                methods=ordinary_methods,
                prior=config.prior,
                alpha=alpha_for,
                alternative=test.alternative,
                inference=inference,
                null_lift=getattr(test, "null_lift", 0.0),
                null_abs=getattr(test, "null_abs", None),
                preferred_direction=metric.declared_preferred_direction,
                cluster=src.context.cluster,
                advisory_seen=advisory_seen,
                method_roles={
                    method.name: method_roles[method.name] for method in ordinary_methods
                },
                family=retry == "family",
                family_member=retry == "family",
                evidence=evidence,
            )
            rows_est.extend(ordinary)
            refused.extend(ordinary_refused)
            computations.append(ordinary_computation)
        from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

        for method in linear_methods:
            cache_key = method.model_dump_json(exclude={"name"})
            cached = linear_cache.get(cache_key)
            try:
                if cached is not None:
                    cached_bundle, cached_error = cached
                    if cached_error is not None:
                        raise cached_error
                    assert cached_bundle is not None
                    reused_results = tuple(
                        row.model_copy(
                            update={
                                "method": method.name,
                                "method_role": method_roles[method.name],
                            }
                        )
                        for row in cached_bundle.results
                    )
                    linear = DecisionComputation[LiftEstimate](
                        results=reused_results,
                        evidence=cached_bundle.evidence
                        if method_roles[method.name] == "decision"
                        else {},
                        failures=cached_bundle.failures
                        if method_roles[method.name] == "decision"
                        else {},
                        advisories=cached_bundle.advisories,
                    )
                else:
                    method_frame, method_columns = _method_unit_frame(
                        evidence, method, config.covariates
                    )
                    linear = estimate_cuped_linear(
                        metric,
                        method_frame,
                        control_group=design.control_group,
                        method=method,
                        method_role=method_roles[method.name],
                        covariate_columns=method_columns,
                        covariates=method.covariates or config.covariates,
                        alpha=alpha_for,
                        alternative=test.alternative,
                        null_abs=getattr(test, "null_abs", None),
                        null_lift=getattr(test, "null_lift", 0.0),
                        prior=config.prior,
                    )
                    linear_cache[cache_key] = (linear, None)
            except CodedError as exc:
                if exc.code not in _CELL_ESTIMATION_GUARD_CODES:
                    raise
                if cache_key not in linear_cache:
                    linear_cache[cache_key] = (None, exc)
                failures: dict[Any, DecisionFailure] = {}
                groups = sorted(evidence.observed_arms - {design.control_group})
                for group_id in groups:
                    hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
                    failure_context = exc.context
                    if method_roles[method.name] == "decision":
                        failures[hypothesis] = DecisionFailure(
                            hypothesis, exc.code, failure_context
                        )
                    else:
                        rows_est.append(
                            _cuped_linear_unavailable_row(
                                metric,
                                group_id,
                                method,
                                "sensitivity",
                                alternative=test.alternative,
                                null_lift=getattr(test, "null_lift", 0.0),
                                null_abs=getattr(test, "null_abs", None),
                                prior=config.prior,
                                code=exc.code,
                                context=failure_context,
                            )
                        )
                    refused.append((metric.name, group_id, method.name, str(exc)))
                computations.append(
                    DecisionComputation[LiftEstimate](results=(), evidence={}, failures=failures)
                )
            else:
                rows_est.extend(linear.results)
                computations.append(linear)
                # An arm the fit withheld under an estimator-domain refusal is a
                # refused cell, exactly as the unadjusted per-arm path reports it.
                for row in linear.results:
                    code = row.failure_code
                    if code is None or code.startswith("estimation.cuped_linear."):
                        continue
                    refused.append((metric.name, row.group_id, method.name, code))
                    _warn(
                        "readouts.run.cell_refused",
                        metric_name=metric.name,
                        group_id=row.group_id,
                        method_name=method.name,
                        reason=code,
                        stacklevel=3,
                    )
        computation = _merge_decision_computations(computations)
        return [row.model_copy(update={"role": role}) for row in rows_est], refused, computation
    if retry in ("whole", "whole_asof"):
        computations = [
            _estimate_lift(
                metrics=[metric],
                summary=(
                    evidence.rows_for_method(method)
                    if method.variance_reduction == "cuped" and isinstance(evidence, MetricRows)
                    else rows_summary or []
                ),
                control_group=design.control_group,
                methods=[method],
                prior=config.prior,
                alpha=alpha_for,
                alternative=test.alternative,
                inference=inference,
                null_lift=getattr(test, "null_lift", 0.0),
                null_abs=getattr(test, "null_abs", None),
                preferred_direction=metric.declared_preferred_direction,
                method_roles=method_roles,
                cluster=src.context.cluster,
                family_member=family_member,
                route_tail=route_tail,
            )
            for method in methods
        ]
        computation = _merge_decision_computations(computations)
        return (
            [row.model_copy(update={"role": role}) for row in computation.results],
            [],
            computation,
        )

    rows_est, refused, computation = _estimate_lift_cells(
        metric=metric,
        summary=rows_summary or [],
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
        family_member=family_member,
        route_tail=route_tail,
        evidence=evidence if isinstance(evidence, MetricRows) else None,
    )
    return [row.model_copy(update={"role": role}) for row in rows_est], refused, computation
