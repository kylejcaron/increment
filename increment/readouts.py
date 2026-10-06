"""The readout set: every function takes a MomentSource and a Design and returns an estimate, without naming a data substrate.

Grain-specific readouts (daily, asof_lift) require the source to declare
the matching capability up front, so an unsupported request raises
CapabilityError naming the missing grain rather than failing deep inside
estimation.

breakout delegates randomized segments to
increment.breakout.estimates.run_breakout, excluding segments with no
usable control arm; encouragement estimation instead warns and skips
control-free segments and suppresses late rows after a weak first stage.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    effective_methods,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._literals import Correction, PreferredDirection, Role, ValueScale
from increment._policy_alpha import resolve_cell_alpha
from increment._readout_request import ReadoutRequest, validate_request
from increment._readout_request import _raise as _raise_readout_request
from increment._window import resolve_window_days
from increment.breakout.estimates import (
    DEFAULT_RELIABILITY_FLOOR,
    BreakoutEstimate,
    BreakoutEstimates,
    _asof_monitoring_note,
    _breakout_estimate_row,
    _refuse,
    reject_quantile_metrics,
    run_breakout,
)
from increment.compatibility import Unsupported, refuse_unsupported
from increment.errors import (
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation.adjust import (
    ESTIMATION_ADJUST_ESTIMATE_ATE_EVERY,
    estimate_ate,
    judge_shared_prior_scales,
)
from increment.estimation.diagnostics import (
    NotApplicable,
    SRMResult,
    complete_srm_support,
    resolve_srm_expected,
    sample_ratio_mismatch,
)
from increment.estimation.encouragement import (
    ESTIMANDS,
    ESTIMATION_ENCOURAGEMENT_UNKNOWN_ESTIMAND_SUPPORTED,
    READOUT_ENCOURAGEMENT_VALUE_SCALE,
    estimate_compliance,
    estimate_encouragement,
)
from increment.estimation.engine import (
    Method,
    _validate_methods,
    estimate_lift,
)
from increment.estimation.engine import (
    merge_decision_computations as _merge_decision_computations,
)
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.inference import LiftGuardError
from increment.estimation.quantile import estimate_quantile_lift_computation
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    open_bound_from_two_sided_at_target,
)
from increment.estimation.sequential import (
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
)
from increment.semantics.design import Encouragement
from increment.sources import (
    MIXED_ASSIGNMENT_LABEL,
    UNASSIGNED_LABEL,
    MomentSource,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment._source_types import ComplianceSummary, RawOutcomeSource
    from increment.decision import CompiledDecisionPlan, DecisionComputation
    from increment.estimation.inference import Prior
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Observational, Randomized
    from increment.semantics.models import Metric


def _joint_relative_rows(rows: Sequence[LiftEstimate], *, metric: str | None = None) -> bool:
    """Joint decision rows use actual FCR alpha, not the legacy central
    equivalent. A nonpositive_arm_mean row is an ordinary additive Wald
    result (never a joint/Fieller construction, which is inherently
    two-sided), so it must not force two-sided re-estimation on its
    metric's other, healthy, directional arms.
    """
    return any(
        row.method_role == "decision"
        and row.estimand == "itt"
        and (metric is None or row.metric == metric)
        and (
            row.relative_confidence_set is not None
            or (
                row.relative_unavailable_reason is not None
                and row.relative_unavailable_reason != "nonpositive_arm_mean"
            )
        )
        for row in rows
    )


Renderer = Callable[..., str]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "readout.alwaysvalid_under_encouragement": "{method}: AlwaysValid under an encouragement design requires completed_windows_only=True so repeated looks use finalized observations",
        "readout.completed_encouragement_inference": "{method}: completed encouragement inference requires bounded outcome and uptake windows; unbounded metrics={unbound!r}, uptake_window_days={uptake_window_days!r}",
        "readout.estimate_lift_every": RefusalSpec(
            "readout.estimate_lift_every",
            UnsupportedRequestError,
            template="estimate_lift: every requested randomized metric/arm/method cell was refused by an inference guard -- no estimates remain. See the accompanying UserWarnings for metric/arm/method refusal reasons.",
        ),
        "readout.value_scale_names": "value_scale= names metrics this source does not report on this call: {unknown_value_scale!r} (selected: {metric_names!r}) -- refusing rather than silently dropping the request",
        "readout.sequential_inference_supported": RefusalSpec(
            "readout.sequential_inference_supported",
            UnsupportedRequestError,
            template="sequential inference is not supported on the whole-window run() entry point under an encouragement design -- for sequential monitoring of an encouragement rollout, use asof_lift() instead, which carries time-uniform intervals on every row (LATE included) -- declare inference=InferenceSpec(kind='always_valid', registration=...) on the source's AnalysisPlan (asof_lift() reads it from there, not a call kwarg)",
        ),
        "readout.value_scale_randomized_absolute": RefusalSpec(
            "readout.value_scale_randomized_absolute",
            UnsupportedRequestError,
            template="value_scale= is an observational-only reporting selector -- the randomized path already reports the absolute axis (abs_diff/abs_se/abs_lb/abs_ub) on every row; declare an absolute margin instead.",
        ),
        "readout.metric_declare_non": RefusalSpec(
            "readout.metric_declare_non",
            UnsupportedRequestError,
            template="metric(s) {combined} declare a non-inferiority margin (Metric.margin/margin_abs, or a plan-bound ExperimentMetric.margin), but a per-metric shifted null is not built for the as-of encouragement view -- run() applies the margin to the whole-window ITT row, or declare a one-sided plan alternative for a test against null_lift=0.0",
        ),
        "readout.breakout_correction_bh": "breakout: correction='bh' cannot use an informative prior; BH/e-BH family selection requires frequentist p-values/e-values",
        "readout.srm_source_declared": "srm() requires a source with a declared design -- this source was constructed without one (design=None). Pass design= at construction (or control=/control_group= to derive Randomized).",
    },
)

_REFUSALS["estimation.encouragement.unknown_estimand_supported"] = (
    ESTIMATION_ENCOURAGEMENT_UNKNOWN_ESTIMAND_SUPPORTED
)

_REFUSALS["estimation.adjust.estimate_ate_every"] = ESTIMATION_ADJUST_ESTIMATE_ATE_EVERY

_REFUSALS["readout.encouragement.value_scale"] = READOUT_ENCOURAGEMENT_VALUE_SCALE
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    # +1 absorbs this helper's own frame; errors.warn() absorbs its own via
    # its internal stacklevel + 1, so call sites keep their original literal.
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "readouts.run.cell_refused",
    IncrementWarning,
    lambda *, metric_name, group_id, method_name, reason: (
        f"run: metric={metric_name!r} group_id={group_id!r} "
        f"method={method_name!r} -- {reason} -- cell refused."
    ),
)
_register_warning(
    "readouts.breakout.segment_no_control_arm",
    IncrementWarning,
    lambda *, dimension, value, control_group: (
        f"breakout: segment {dimension}={value!r} has no "
        f"'{control_group}' control arm for any metric -- skipped."
    ),
)


def _sequential_inference(plan: CompiledDecisionPlan):
    """Return the optional sequential runtime object; fixed is a sentinel."""
    return plan.inference if isinstance(plan.inference, SEQUENTIAL_POLICIES) else None


def _refuse_segmented_registration(inference: AsymptoticMean | AlwaysValid | MixedFamily) -> None:
    """Only breakout() reports a segmented roster: its rows carry each cell's
    segment and exploratory role, which whole-window and as-of rows cannot."""
    if any(cell.segment for cell in inference.registration.roster):
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "segmented registrations require breakout() so segment identities are retained",
        )


def _require_design(src: MomentSource, view: str) -> Randomized | Encouragement | Observational:
    design = src.context.design
    if design is None:
        _raise_readout_request("readout.design.required", view=view)
    return design


def _runtime_methods(
    config: ResolvedMetricConfig,
    design: Randomized | Encouragement | Observational,
) -> list[Method]:
    """Return one concrete method sequence for this metric and design."""
    return list(effective_methods(config, design=design))


def _runtime_method_roles(
    methods: Sequence[Method],
) -> dict[str, Literal["decision", "sensitivity"]]:
    return {
        method.name: ("decision" if index == 0 else "sensitivity")
        for index, method in enumerate(methods)
    }


def _validate_encouragement_asof_inference(
    metrics: Sequence[Metric],
    design: Randomized | Encouragement | Observational,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    completed_windows_only: bool,
    method: str,
) -> None:
    """Require finalized bounded windows for AlwaysValid or MixedFamily inference
    under an Encouragement design; a no-op for every other combination.

    Both policies carry an exact Bernoulli uptake cell, which needs the same
    finalized-bounded-window guarantee AlwaysValid needs; AsymptoticMean alone
    (no compliance cell) does not. Shared by asof_lift and
    Analysis.run_asof_lift so both as-of routes enforce the same contract.
    """
    from increment.estimation.sequential import UPTAKE_COMPLETION_POLICIES
    from increment.semantics.design import Encouragement as _Encouragement

    if not isinstance(design, _Encouragement) or not isinstance(
        inference, UPTAKE_COMPLETION_POLICIES
    ):
        return
    if not completed_windows_only:
        _raise("readout.alwaysvalid_under_encouragement", method=method)
    unbounded = [metric.name for metric in metrics if resolve_window_days(metric) is None]
    if unbounded or design.uptake.window_days is None:
        _raise(
            "readout.completed_encouragement_inference",
            method=method,
            unbound=unbounded,
            uptake_window_days=design.uptake.window_days,
        )


def _declared_margin_names(metrics: Sequence[object]) -> list[str]:
    """Names among *metrics* declaring a guardrail margin, relative
    (Metric.margin) or absolute (Metric.margin_abs) - used to refuse
    rather than silently test two-sided vs 0 where a per-metric shifted
    null isn't supported.
    """
    return [
        cast("Metric", m_).name
        for m_ in metrics
        if any(getattr(m_, f, None) is not None for f in ("margin", "margin_abs"))
    ]


def _refuse_unsupported_quantile(
    metric: Metric, test: Any, *, cluster: str | None, by: Sequence[str]
) -> None:
    """Structural refusals shared by every quantile-metric estimation site
    in `run()`: no cluster-grain quantile moments, no breakout dimensions,
    and (for now) no one-sided/shifted-null quantile test."""
    if cluster is not None:
        refuse_unsupported(
            Unsupported("arm.metric.quantile_cluster"),
            metric=metric.name,
            cluster=cluster,
            metric_type="quantile",
        )
    if by:
        _raise_readout_request("readout.metric.quantile_breakout", metric=metric.name)
    if (
        test.alternative != "two-sided"
        or getattr(test, "null_lift", 0.0) != 0.0
        or getattr(test, "null_abs", None) is not None
    ):
        _raise_readout_request("readout.metric.quantile_alternative", metric=metric.name)


def _refuse_unsupported_by(metric: Metric, by: Sequence[str]) -> None:
    """`run()`'s non-quantile lift-cell estimator indexes rows by (metric,
    arm) only, so a nonempty by= would silently let one segment's row
    overwrite another's; refuse rather than guess. Quantile metrics take
    their own more specific path (`_refuse_unsupported_quantile`)."""
    if by:
        _raise_readout_request("readout.run.segment_unsupported", metric=metric.name)


def _refuse_if_no_treatment_arm(observed_arms: set[str], control_group: str) -> None:
    """A lift readout needs a contrast: refuse once every metric has been
    read and none carried a non-control arm, instead of returning no rows."""
    if not observed_arms - {str(control_group)}:
        _raise_readout_request(
            "readout.arms.no_treatment",
            control_group=control_group,
            observed_arms=tuple(sorted(observed_arms)),
        )


@dataclass(frozen=True, slots=True)
class MetricRows:
    """One metric's rows, loaded exactly once, quantile or moments-shaped."""

    rows: tuple[Mapping[str, Any], ...]
    unit_frame: Any | None
    observed_arms: frozenset[str]
    n_treatment_arms: int


def _load_metric_rows(
    src: MomentSource, metric: Metric, *, by: Sequence[str], control_group: str
) -> MetricRows:
    """Load one metric's rows once.

    Callers needing the quantile-specific alternative/cluster/breakout guard
    (`_refuse_unsupported_quantile`, which needs `test`/`cluster` -- outside
    this function's contract) call it before `_load_metric_rows`.
    """
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        import narwhals as nw

        from increment.estimation.winsor import _raw_state_from_source

        # Check the arm inventory before constructing a two-arm inference state.
        # Reuse this capture so the aggregate arm gate does not trigger another read.
        raw_source = cast("RawOutcomeSource", src)
        native = raw_source.unit_frame(metric, outcome_stage="raw")

        frame = nw.from_native(native, eager_only=True)
        observed = frozenset(str(g) for g in frame["group_id"].unique().to_list())
        if not observed - {str(control_group)}:
            return MetricRows(
                rows=(),
                unit_frame=None,
                observed_arms=observed,
                n_treatment_arms=0,
            )
        raw = _raw_state_from_source(src, metric, native=native)
        references = {}
        if raw.inference.method == "positive-log-kernel-bootstrap-t-v1":
            from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference

            references = {
                (metric.name, arm.group_id): full_procedure_bootstrap_reference(
                    raw, control_group, arm.group_id
                )
                for arm in raw.arms
                if arm.group_id != control_group
            }
        observed = frozenset(a.group_id for a in raw.arms)
        # Keep the immutable raw state with the loaded evidence so repeated
        # family passes reinvert without fetching or changing the cutoff pool.
        return MetricRows(
            rows=tuple(
                {"group_id": a.group_id, "_winsor_raw_state": raw, "_winsor_references": references}
                for a in raw.arms
            ),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
        )
    if getattr(metric, "type", None) == "quantile":
        import narwhals as nw

        native = src.unit_frame(metric)
        frame = nw.from_native(native, eager_only=True)
        observed = frozenset(str(g) for g in frame["group_id"].to_list())
        return MetricRows(
            rows=(),
            unit_frame=native,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
        )
    _refuse_unsupported_by(metric, by)
    rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=by))
    observed = frozenset(str(r["group_id"]) for r in rows)
    return MetricRows(
        rows=tuple(rows),
        unit_frame=None,
        observed_arms=observed,
        n_treatment_arms=len(observed - {control_group}),
    )


def _encouragement_family_config(
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    plan: CompiledDecisionPlan,
) -> tuple[dict[str, ResolvedMetricConfig], set[str]]:
    config_by_metric = {
        metric.name: config for metric, config in zip(metrics, configs, strict=True)
    }
    family_metric_names = {
        metric.name
        for metric in metrics
        if (
            plan.procedures[metric.name].role == "secondary"
            and getattr(plan.procedures[metric.name].family, "member", False)
            and config_by_metric[metric.name].prior is None
        )
    }
    return config_by_metric, family_metric_names


def _encouragement_cell_groups(
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Encouragement,
    alpha_by_metric: Mapping[str, float],
    alternative_by_metric: Mapping[str, str],
    null_lift_by_metric: Mapping[str, float],
    null_abs_by_metric: Mapping[str, float | None],
) -> list[tuple[list[Method], Prior | None, str, float, float, float | None, list[Metric]]]:
    groups: list[
        tuple[list[Method], Prior | None, str, float, float, float | None, list[Metric]]
    ] = []
    for metric, config in zip(metrics, configs, strict=True):
        alternative = alternative_by_metric[metric.name]
        alpha = alpha_by_metric[metric.name]
        null_lift = null_lift_by_metric[metric.name]
        null_abs = null_abs_by_metric[metric.name]
        metric_methods = _runtime_methods(config, design)
        for (
            group_methods,
            group_prior,
            group_alt,
            group_alpha,
            group_null_lift,
            group_null_abs,
            group_metrics,
        ) in groups:
            if (
                group_methods == metric_methods
                and group_prior == config.prior
                and group_alt == alternative
                and group_alpha == alpha
                and group_null_lift == null_lift
                and group_null_abs == null_abs
            ):
                group_metrics.append(metric)
                break
        else:
            groups.append(
                (
                    metric_methods,
                    config.prior,
                    alternative,
                    alpha,
                    null_lift,
                    null_abs,
                    [metric],
                )
            )
    return groups


def encouragement_rows(
    *,
    src: MomentSource,
    metrics: Sequence[Metric],
    rows_by_metric: Mapping[str, list[Mapping[str, Any]]],
    configs: Sequence[ResolvedMetricConfig],
    design: Encouragement,
    plan: CompiledDecisionPlan,
    estimands: Sequence[str] | None,
    cluster: str | None,
    caller: str,
) -> list[LiftEstimate]:
    """Every encouragement row for one whole-window readout.

    The single implementation behind both the dataframe and warehouse
    paths: a primary's alpha splits across its own treatment arms, a
    secondary estimates at the plan's nominal alpha and then faces
    BH/e-BH family selection, and the design-level ``uptake`` row is
    estimated exactly once at the plan's own alpha rather than inheriting
    whichever metric ran first.

    ``estimate_encouragement`` takes alpha/alternative/shifted null as
    call-wide scalars, so metrics sharing (methods, prior, alternative,
    alpha, null) are batched into one call and the results sorted back into
    declared order. A declared margin applies to the ITT row only; LATE and
    compliance carry no shifted null.
    """
    wanted = estimands if estimands is not None else ESTIMANDS
    if set(wanted) == {"compliance"}:
        compliance_summary = _validated_compliance_summary(src, design)
        computation = _design_compliance_computation(
            compliance_summary,
            design,
            wanted,
            alpha=plan.alpha,
            deferred=False,
        )
        return list(computation.results)
    if not metrics:
        return []
    reject_quantile_metrics(
        metrics,
        caller,
        reason="estimate_encouragement consumes mean-based group_summary moments, "
        "which a quantile metric has none of",
        remedy="Drop the encouragement design or the quantile metric.",
    )
    role_by_metric: dict[str, str | None] = {}
    cell_alpha_by_metric: dict[str, float] = {}
    cell_alternative_by_metric: dict[str, str] = {}
    cell_null_lift_by_metric: dict[str, float] = {}
    cell_null_abs_by_metric: dict[str, float | None] = {}
    control = str(design.control_group)
    for metric in metrics:
        test = plan.procedures[metric.name]
        cell_null_lift_by_metric[metric.name] = getattr(test, "null_lift", 0.0) or 0.0
        cell_null_abs_by_metric[metric.name] = getattr(test, "null_abs", None)
        if not plan.declared:
            # No declared plan: every metric uses its compiled default.
            role_by_metric[metric.name] = None
            cell_alpha_by_metric[metric.name] = test.alpha
            cell_alternative_by_metric[metric.name] = test.alternative
            continue
        role_by_metric[metric.name] = test.role
        if test.role == "primary":
            arm_ids = {str(r["group_id"]) for r in rows_by_metric[metric.name]}
            n_arms = len(arm_ids - {control})
        else:
            n_arms = 0
        # This first pass intentionally uses the secondary's nominal plan.alpha. The
        # family block below stamps the ITT verdict and re-estimates selected cells
        # at the capped FCR alpha; compliance and LATE rows keep discovery unset.
        cell_alpha_by_metric[metric.name] = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
        cell_alternative_by_metric[metric.name] = test.alternative
    config_by_metric, family_metric_names = _encouragement_family_config(metrics, configs, plan)

    groups = _encouragement_cell_groups(
        metrics,
        configs,
        design,
        cell_alpha_by_metric,
        cell_alternative_by_metric,
        cell_null_lift_by_metric,
        cell_null_abs_by_metric,
    )

    compliance_summary = (
        _validated_compliance_summary(src, design) if "compliance" in wanted else None
    )
    results: list[LiftEstimate] = []
    computations: list[DecisionComputation[LiftEstimate]] = []
    family_refusal_deferred = False
    for (
        group_methods,
        group_prior,
        group_alt,
        group_alpha,
        group_null_lift,
        group_null_abs,
        group_metrics,
    ) in groups:
        summary = [row for m in group_metrics for row in rows_by_metric[m.name]]
        method_roles = _runtime_method_roles(group_methods)
        try:
            computation = estimate_encouragement(
                metrics=group_metrics,
                summary=summary,
                design=design,
                estimands=tuple(name for name in wanted if name != "compliance"),
                methods=group_methods,
                prior=group_prior,
                alpha=group_alpha,
                alternative=group_alt,
                null_lift=group_null_lift or None,
                null_abs=group_null_abs,
                cluster=cluster,
                method_roles=method_roles,
            )
        except InvalidRequestError as exc:
            if exc.code != "estimation.encouragement.control.missing" or any(
                metric.name not in family_metric_names for metric in group_metrics
            ):
                raise
            family_refusal_deferred = True
            from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

            failures: dict[Any, Any] = {}
            for metric in group_metrics:
                for row in rows_by_metric[metric.name]:
                    group_id = str(row["group_id"])
                    if group_id == str(design.control_group):
                        continue
                    hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
                    failures[hypothesis] = DecisionFailure(
                        hypothesis,
                        "estimation.encouragement.control.missing",
                        {
                            "metric": metric.name,
                            "group_id": group_id,
                            "reason": "no_control_arm",
                        },
                    )
            computation = DecisionComputation[LiftEstimate](
                results=(), evidence={}, failures=failures
            )
        computations.append(computation)
        results.extend(
            r.model_copy(
                update={
                    "role": role_by_metric.get(r.metric),
                    "method_role": method_roles.get(r.method, "decision"),
                }
            )
            for r in computation.results
        )

    # Compliance is design-level: estimate it once at the plan alpha from the
    # declared uptake cohort, never outcome-filtered moments, so it cannot depend
    # on metric order, metric count, or outcome missingness policy.

    computation = _design_compliance_computation(
        compliance_summary,
        design,
        wanted,
        alpha=plan.alpha,
        deferred=family_refusal_deferred,
    )
    computations.append(computation)
    results.extend(r.model_copy(update={"role": None}) for r in computation.results)
    if plan.declared:
        results = _select_encouragement_family(
            results,
            computation=_merge_decision_computations(computations),
            metrics=metrics,
            configs=configs,
            plan=plan,
            design=design,
            rows_by_metric=rows_by_metric,
            estimands=estimands,
            cluster=cluster,
        )
    # Grouping dispatches metrics out of declared order, so stable-sort
    # back to it; compliance is design-level (identified by estimand, not
    # its "uptake" metric string, which a real outcome metric may share),
    # so it sorts first.
    order = {m.name: i for i, m in enumerate(metrics)}
    results.sort(key=lambda r: -1 if r.estimand == "compliance" else order[r.metric])
    return results


def _validated_compliance_summary(src: MomentSource, design: Encouragement) -> ComplianceSummary:
    from increment._source_types import validate_compliance_source_match

    summary = src.compliance_summary(design)
    validate_compliance_source_match(summary, src.context)
    _refuse_if_no_treatment_arm(
        {str(arm.group_id) for arm in summary.arms},
        design.control_group,
    )
    return summary


def _design_compliance_computation(
    summary: ComplianceSummary | None,
    design: Encouragement,
    wanted: Sequence[str],
    *,
    alpha: float,
    deferred: bool,
) -> DecisionComputation[LiftEstimate]:
    from increment.decision import DecisionComputation

    compliance_wanted = "compliance" in wanted
    if deferred or not compliance_wanted:
        return DecisionComputation(results=(), evidence={}, failures={})
    assert summary is not None
    return estimate_compliance(
        summary,
        design,
        alpha=alpha,
        compliance_requested=compliance_wanted,
    )


def _select_encouragement_family(
    rows: list[LiftEstimate],
    *,
    computation: DecisionComputation[LiftEstimate],
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    plan: CompiledDecisionPlan,
    design: Encouragement,
    rows_by_metric: Mapping[str, list[Mapping[str, Any]]],
    estimands: Sequence[str] | None,
    cluster: str | None,
) -> list[LiftEstimate]:
    """BH/e-BH selection over in-family encouragement secondaries.

    A metric here emits several estimand rows (itt/compliance/late), but
    the family is one cell per ``(metric, arm)`` and the ITT row is that
    cell's representative: itt and late are two estimands of the SAME
    null, so admitting both would double ``m`` for one hypothesis, and
    `select_family` already collapses same-null rows for exactly that
    reason on the method axis. compliance is excluded outright - it is
    the first-stage uptake diagnostic, not a hypothesis about the
    declared metric.

    ITT specifically, not late: a weak first stage suppresses the late
    row entirely, so keying on it would drop a metric out of the family
    precisely when its instrument is weak, shrinking ``m`` and loosening
    the bar for every surviving metric. ITT is also the only estimand
    with an unconditional randomization guarantee - late needs the
    exclusion restriction and a strong first stage, which a family-wide
    error rate should not silently inherit.

    ``discovery`` lands on the ITT rows only. A compliance or late row
    was never the tested hypothesis, and marking it a discovery would
    claim its own interval met the family's bar.
    """
    config_by_name = {m.name: c for m, c in zip(metrics, configs, strict=True)}
    in_family = {
        m.name
        for m in metrics
        if (
            plan.procedures[m.name].role == "secondary"
            and getattr(plan.procedures[m.name].family, "member", False)
            and config_by_name[m.name].prior is None
        )
    }
    if not in_family:
        return rows
    if estimands is not None and "itt" not in estimands:
        return rows
    from increment.decision import ArmHypothesisKey

    # The compiled family is every outcome metric/arm cell, not merely the
    # rows that survived estimation.  A keyed failure therefore aborts the
    # complete family instead of shrinking its denominator.
    cells: list[tuple[ArmHypothesisKey, object]] = []
    for metric_name in sorted(in_family):
        groups = {
            str(row["group_id"])
            for row in rows_by_metric[metric_name]
            if str(row["group_id"]) != str(design.control_group)
        }
        for group_id in sorted(groups):
            hypothesis = ArmHypothesisKey(metric_name, group_id, "itt")
            row = next(
                (
                    r
                    for r in rows
                    if r.metric == metric_name
                    and r.group_id == group_id
                    and r.estimand == "itt"
                    and r.method_role == "decision"
                ),
                None,
            )
            cells.append((hypothesis, row))
    if not cells:
        # Narrowed estimands can omit ITT entirely; no family was compiled.
        return rows
    outcome = select_family(
        decision_cells(cells),
        plan.q,
        _sequential_inference(plan),
        plan.alpha,
        computation=computation,
    )
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    reestimated: dict[tuple[str, str, str, str, str], LiftEstimate] = {}
    if outcome.fcr_alpha is not None:
        selected_names = {key.metric for key in outcome.selected}
        config_by_name = {m.name: c for m, c in zip(metrics, configs, strict=True)}
        for name in sorted(selected_names):
            metric = next(m for m in metrics if m.name == name)
            config = config_by_name[name]
            alternative = plan.procedures[name].alternative
            selected_alpha = _fcr_alpha_for(
                "two-sided" if _joint_relative_rows(rows, metric=name) else alternative,
                outcome.fcr_alpha,
            )
            pass_results = estimate_encouragement(
                metrics=[metric],
                summary=rows_by_metric[name],
                design=design,
                estimands=estimands if estimands is not None else ESTIMANDS,
                methods=[config.decision_method],
                prior=config.prior,
                alpha=selected_alpha,
                alternative=alternative,
                null_lift=getattr(plan.procedures[name], "null_lift", 0.0) or None,
                null_abs=getattr(plan.procedures[name], "null_abs", None),
                cluster=cluster,
                method_roles={config.decision_method.name: "decision"},
            ).results
            for r in pass_results:
                if r.estimand in ("itt", "late"):
                    reestimated[(r.metric, r.group_id, r.method, r.estimand, r.value_scale)] = (
                        open_bound_from_two_sided_at_target(r)
                    )
    out: list[LiftEstimate] = []
    for r in rows:
        if r.metric not in in_family or r.estimand not in ("itt", "late"):
            out.append(r)
            continue
        key = ArmHypothesisKey(r.metric, r.group_id, "itt")
        selected = key in outcome.selected
        match = (
            reestimated.get((r.metric, r.group_id, r.method, r.estimand, r.value_scale))
            if selected
            else None
        )
        if r.estimand != "itt":
            # LATE presents the same tested null (see the docstring). Selected
            # decision-role cells carry the corrected interval; other late rows keep
            # theirs. Every late row gets the family role, but discovery and family
            # metadata apply only to the tested ITT hypothesis.
            base = match if match is not None else r
            out.append(base.model_copy(update={"role": "secondary"}))
            continue
        if r.method_role != "decision":
            out.append(
                r.model_copy(
                    update={
                        "role": "secondary",
                        "discovery": None,
                        "family_axes": None,
                        "family_q": None,
                        "family_threshold": None,
                    }
                )
            )
            continue
        base = match if match is not None else r
        out.append(
            base.model_copy(
                update={
                    "role": "secondary",
                    "discovery": family_discovery(outcome, key),
                    **family_record,
                }
            )
        )
    return out


def _design_compliance_only(src: MomentSource, estimands: Sequence[str] | None) -> bool:
    """Fixed-horizon compliance needs design state, not outcome configuration."""
    return (
        estimands is not None
        and set(estimands) == {"compliance"}
        and isinstance(src.context.design, Encouragement)
        and not isinstance(src.context.plan.inference, SEQUENTIAL_POLICIES)
    )


def _prepare_run_request(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
) -> tuple[list[Metric], Sequence[ResolvedMetricConfig]]:
    """Resolve and validate a whole-window request before mechanism dispatch."""
    selected = select_metrics(cast("Sequence[Metric]", src.context.metrics), metrics, caller="run")
    # Fixed-horizon compliance consumes no outcome configuration.
    # Sequential requests still need their registered metric context.
    if _design_compliance_only(src, estimands):
        selected = []
    configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=configs,
        view="run",
        grain="total",
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        population=population,
    )
    validate_request(request)
    return selected, configs


def _validate_run_request(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
) -> list[Metric]:
    """Validate a run request without asking the source for moments."""
    selected, _configs = _prepare_run_request(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
    )
    return selected


def _validate_breakout_request(
    src: MomentSource,
    *,
    metrics: Sequence[str | Metric] | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    correction: Correction | None = None,
    q: float | None = None,
    dimension: str,
) -> list[Metric]:
    """Validate a native breakout request before its scoped source queries."""
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="breakout"
    )
    configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=configs,
        view="breakout",
        grain="total",
        by=(dimension,),
        dimension=dimension,
        correction=correction,
        q=q,
    )
    validate_request(request)
    return selected


def _emit_captured_lift_warnings(
    metric_name: str,
    captured: list[Any],
    advisory_seen: set[tuple[str, str]],
) -> None:
    """Replay estimator warnings while deduplicating known per-metric advisories."""
    for record in captured:
        message = str(record.message)
        if "is open-ended (" in message:
            advisory_kind = "open-ended-sequential"
        elif "total clusters across arms" in message:
            advisory_kind = "clustered-small-k"
        else:
            advisory_kind = None
        if advisory_kind is not None:
            key = (metric_name, advisory_kind)
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
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]], DecisionComputation[LiftEstimate]]:
    """Estimate randomized whole-window cells independently.

    ``estimate_lift`` accepts a whole metric slice and therefore aborts its
    entire result when one treatment arm hits an expected ``LiftGuardError``.
    A readout has a denser contract: a bad arm/method cell is warned and
    excluded while its siblings remain usable.  Only ``LiftGuardError`` is
    handled here; malformed moments and other programming/data errors still
    propagate.
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
                    computation = estimate_lift(
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
                if (method_roles or {}).get(method.name, "decision") == "decision":
                    from increment.decision import (
                        ArmHypothesisKey,
                        DecisionComputation,
                        DecisionFailure,
                    )

                    hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
                    computations.append(
                        DecisionComputation(
                            results=(),
                            evidence={},
                            failures={
                                hypothesis: DecisionFailure(
                                    hypothesis,
                                    "estimation.engine.lift_guard",
                                    {
                                        "metric": metric.name,
                                        "group_id": group_id,
                                        "method": method.name,
                                        "reason": exc.reason,
                                        "display": reason,
                                    },
                                )
                            },
                        )
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
) -> None:
    """Give an explicit result when every attempted randomized cell refused."""
    if refused:
        _raise("readout.estimate_lift_every")


def _estimate_pass(
    src: MomentSource,
    metric: Metric,
    evidence: Any,
    test: Any,
    config: ResolvedMetricConfig,
    design: Randomized | Encouragement | Observational,
    *,
    alpha_for: float,
    role: Role | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]] | None,
    retry: Literal["cells", "family", "whole", "whole_asof"] = "cells",
    methods: list[Method],
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
        computation = estimate_lift(
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
    )
    return [row.model_copy(update={"role": role}) for row in rows_est], refused, computation


def _estimate_observational(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    value_scale: Mapping[str, ValueScale] | None,
    call_prior: Prior | None,
) -> list[LiftEstimate]:
    """Estimate the observational phase after whole-window validation.

    Applies the same role machinery randomized readouts use: a primary's
    alpha is Bonferroni-split across its treatment arms, a guardrail keeps
    full alpha (intersection-union, never joins a family), and secondaries
    are BH-selected at plan.q then re-estimated at the Benjamini-Yekutieli
    FCR level for the selected cells.
    """
    metric_names = [metric.name for metric in selected]
    if value_scale:
        unknown_value_scale = set(value_scale) - set(metric_names)
        if unknown_value_scale:
            _raise(
                "readout.value_scale_names",
                metric_names=sorted(metric_names),
                unknown_value_scale=sorted(unknown_value_scale),
            )
    resolved_null_abs: dict[str, float] = {
        metric.name: float(cast(Any, plan.procedures[metric.name]).null_abs)
        for metric in selected
        if getattr(plan.procedures[metric.name], "null_abs", None) is not None
    }
    resolved_alternatives: dict[str, str] = {
        metric.name: plan.procedures[metric.name].alternative for metric in selected
    }
    judge_shared_prior_scales(
        {metric.name: metric for metric in src.context.metrics},
        selected_names={metric.name for metric in selected},
        prior=call_prior,
        value_scale=value_scale,
        stacklevel=3,
    )
    config_by_name = {metric.name: config for metric, config in zip(selected, configs, strict=True)}
    methods_by_metric = {
        metric.name: _runtime_methods(config_by_name[metric.name], design) for metric in selected
    }
    selected = [metric for metric in selected if methods_by_metric[metric.name]]
    non_secondary = [m for m in selected if plan.procedures[m.name].role != "secondary"]
    secondary = [m for m in selected if plan.procedures[m.name].role == "secondary"]

    results: list[LiftEstimate] = []
    for metric in non_secondary:
        config = config_by_name[metric.name]
        procedure = plan.procedures[metric.name]
        n_arms = (
            len(
                {
                    str(row["group_id"])
                    for row in cast("list[Mapping[str, Any]]", src.moments(metric))
                }
                - {str(design.control_group)}
            )
            if procedure.role == "primary"
            else 0
        )
        alpha = resolve_cell_alpha(plan, procedure, n_arms=n_arms, view=None)
        computation = estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name],
            metrics=[metric],
            method_roles=_runtime_method_roles(methods_by_metric[metric.name]),
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
        )
        results.extend(
            row.model_copy(update={"role": procedure.role if plan.declared else None})
            for row in computation.results
        )

    if secondary:
        results.extend(
            _estimate_observational_secondary_family(
                src,
                secondary,
                config_by_name,
                design,
                plan,
                methods_by_metric=methods_by_metric,
                value_scale=value_scale,
                resolved_null_abs=resolved_null_abs,
                resolved_alternatives=resolved_alternatives,
            )
        )

    if not results and selected:
        _raise("estimation.adjust.estimate_ate_every")
    order = {metric.name: i for i, metric in enumerate(selected)}
    results.sort(key=lambda result: order[result.metric])
    return results


def _estimate_observational_secondary_family(
    src: MomentSource,
    secondary: Sequence[Metric],
    config_by_name: Mapping[str, ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    methods_by_metric: Mapping[str, list[Method]],
    value_scale: Mapping[str, ValueScale] | None,
    resolved_null_abs: Mapping[str, float] | None,
    resolved_alternatives: Mapping[str, str] | None,
) -> list[LiftEstimate]:
    """Nominal pass at plan.alpha, BH-select at plan.q, FCR-reestimate the
    selected cells -- the observational sibling of
    _estimate_randomized_secondary_family, built on estimate_ate's own
    DecisionComputation[LiftEstimate] return value. Every secondary joins
    the family unless its config carries an informative prior (the same
    exemption randomized secondaries use).
    """
    from increment.decision import ArmHypothesisKey

    family = [m for m in secondary if config_by_name[m.name].prior is None]
    non_family = [m for m in secondary if config_by_name[m.name].prior is not None]
    roles_by_metric = {
        metric.name: _runtime_method_roles(methods_by_metric[metric.name]) for metric in secondary
    }

    out: list[LiftEstimate] = []

    def _nominal_pass(metric: Metric, alpha: float) -> DecisionComputation[LiftEstimate]:
        config = config_by_name[metric.name]
        return estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name],
            metrics=[metric],
            method_roles=roles_by_metric[metric.name],
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
        )

    for metric in non_family:
        out.extend(
            row.model_copy(update={"role": "secondary", "discovery": None})
            for row in _nominal_pass(metric, plan.alpha).results
        )
    if not family:
        return out

    nominal_by_metric: dict[str, list[LiftEstimate]] = {}
    computation_by_metric: dict[str, DecisionComputation[LiftEstimate]] = {}
    for metric in family:
        computation = _nominal_pass(metric, plan.alpha)
        nominal_by_metric[metric.name] = list(computation.results)
        computation_by_metric[metric.name] = computation

    family_cells: list[tuple[ArmHypothesisKey, object]] = []
    for metric in family:
        seen_keys: set[ArmHypothesisKey] = set()
        for row in nominal_by_metric[metric.name]:
            if row.method_role != "decision":
                continue
            key = ArmHypothesisKey(metric.name, row.group_id, row.estimand)
            seen_keys.add(key)
            family_cells.append((key, row))
        # A failed arm has no row but still occupies its family slot with a None
        # estimate; dropping it would shrink m and loosen every BH threshold.
        # select_family then keeps an allowlisted outcome-degeneracy failure as a
        # non-rejection or refuses the whole family.
        for key in computation_by_metric[metric.name].failures:
            if isinstance(key, ArmHypothesisKey) and key not in seen_keys:
                family_cells.append((key, None))
    family_computation = _merge_decision_computations(
        [computation_by_metric[metric.name] for metric in family]
    )
    outcome = select_family(
        decision_cells(family_cells),
        plan.q,
        _sequential_inference(plan),
        plan.alpha,
        computation=family_computation,
    )
    selected_cells = outcome.selected
    fcr_alpha = outcome.fcr_alpha
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    if fcr_alpha is None:
        for metric in family:
            for row in nominal_by_metric[metric.name]:
                is_decision = row.method_role == "decision"
                key = ArmHypothesisKey(row.metric, row.group_id, row.estimand)
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key) if is_decision else None,
                            "family_axes": family_record["family_axes"] if is_decision else None,
                            "family_q": family_record["family_q"] if is_decision else None,
                            "family_threshold": (
                                family_record["family_threshold"] if is_decision else None
                            ),
                        }
                    )
                )
        return out

    selected_names = {key.metric for key in selected_cells}
    reestimated: dict[str, list[LiftEstimate]] = {}
    for metric in family:
        if metric.name not in selected_names:
            continue
        config = config_by_name[metric.name]
        procedure = plan.procedures[metric.name]
        selected_alpha = _fcr_alpha_for(procedure.alternative, fcr_alpha)
        computation = estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name][:1],
            metrics=[metric],
            method_roles=roles_by_metric[metric.name],
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=selected_alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
        )
        reestimated[metric.name] = [
            open_bound_from_two_sided_at_target(r) for r in computation.results
        ]

    for metric in family:
        for row in nominal_by_metric[metric.name]:
            if row.method_role != "decision":
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": None,
                            "family_axes": None,
                            "family_q": None,
                            "family_threshold": None,
                        }
                    )
                )
                continue
            key = ArmHypothesisKey(row.metric, row.group_id, row.estimand)
            if key in selected_cells:
                match = next(
                    (
                        cell
                        for cell in reestimated.get(row.metric, [])
                        if cell.group_id == row.group_id and cell.method == row.method
                    ),
                    None,
                )
                stamped = match if match is not None else row
            else:
                stamped = row
            out.append(
                stamped.model_copy(
                    update={
                        "role": "secondary",
                        "discovery": family_discovery(outcome, key),
                        **family_record,
                    }
                )
            )
    return out


def _stamp_fcr_selected_rows(
    family: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    nominal_by_metric: Mapping[str, list[LiftEstimate]],
    reestimated: Mapping[str, list[LiftEstimate]],
    selected_cells: Collection[object],
    outcome: Any,
    family_record: Mapping[str, Any],
) -> list[LiftEstimate]:
    """Stamp family rows after selected cells receive FCR intervals."""
    from increment.decision import ArmHypothesisKey

    out: list[LiftEstimate] = []
    for metric, *_rest in family:
        for row in nominal_by_metric[metric.name]:
            if row.method_role != "decision":
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": None,
                            "family_axes": None,
                            "family_q": None,
                            "family_threshold": None,
                        }
                    )
                )
                continue
            key = ArmHypothesisKey(row.metric, row.group_id, "itt")
            if key in selected_cells:
                match = next(
                    (
                        cell
                        for cell in reestimated.get(row.metric, [])
                        if cell.group_id == row.group_id and cell.method == row.method
                    ),
                    None,
                )
                if match is None:
                    continue
                out.append(
                    match.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key),
                            **family_record,
                        }
                    )
                )
            else:
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key),
                            **family_record,
                        }
                    )
                )
    return out


def _partition_secondary_family(
    entries: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
) -> tuple[
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
]:
    family = [
        entry
        for entry in entries
        if getattr(entry[2].family, "member", False) and entry[1].prior is None
    ]
    non_family = [
        entry
        for entry in entries
        if not getattr(entry[2].family, "member", False) or entry[1].prior is not None
    ]
    return family, non_family


def _estimate_randomized_secondary_family(
    src: MomentSource,
    secondary_entries: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    design: Randomized,
    plan: CompiledDecisionPlan,
    *,
    by: Sequence[str],
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]],
    observed_arms: set[str],
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]]]:
    """Run nominal, selection, and FCR passes for randomized secondaries."""
    if not secondary_entries:
        return [], []
    nominal_by_metric: dict[str, list[LiftEstimate]] = {}
    nominal_computation_by_metric: dict[str, DecisionComputation[LiftEstimate]] = {}
    family_groups_by_metric: dict[str, set[str]] = {}
    # Evidence read for the nominal pass (moment rows or a quantile unit
    # frame), reused by the FCR pass instead of re-querying the source.
    evidence_by_metric: dict[str, Any] = {}
    refused_cells: list[tuple[str, str, str, str]] = []
    for metric, config, test, is_quantile, _metric_methods in secondary_entries:
        if is_quantile:
            from increment.estimation.decision_types import ArmHypothesisKey

            _refuse_unsupported_quantile(metric, test, cluster=src.context.cluster, by=by)
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            evidence_by_metric[metric.name] = metric_rows.unit_frame
            observed_arms |= metric_rows.observed_arms
            if metric_rows.n_treatment_arms:
                nominal, refused, computation = _estimate_pass(
                    src,
                    metric,
                    metric_rows.unit_frame,
                    test,
                    config,
                    design,
                    alpha_for=plan.alpha,
                    role="secondary",
                    inference=effective_inference,
                    advisory_seen=advisory_seen,
                    retry="family",
                    methods=_metric_methods,
                )
            else:
                # Nothing to contrast; the final arm gate decides whether the
                # whole readout refuses.
                nominal, refused, computation = [], [], _merge_decision_computations([])
            nominal_by_metric[metric.name] = nominal
            nominal_computation_by_metric[metric.name] = computation
            family_groups_by_metric[metric.name] = {
                row.group_id for row in nominal if row.estimand == "itt"
            }
            family_groups_by_metric[metric.name].update(
                str(key.group_id)
                for key in computation.failures
                if isinstance(key, ArmHypothesisKey)
            )
        else:
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            rows = list(metric_rows.rows)
            evidence_by_metric[metric.name] = rows
            observed_arms |= metric_rows.observed_arms
            nominal, refused, computation = _estimate_pass(
                src,
                metric,
                rows,
                test,
                config,
                design,
                alpha_for=plan.alpha,
                role="secondary",
                inference=effective_inference,
                advisory_seen=advisory_seen,
                retry=(
                    "family"
                    if getattr(test.family, "member", False) and config.prior is None
                    else "cells"
                ),
                methods=_metric_methods,
            )
            nominal_by_metric[metric.name] = nominal
            nominal_computation_by_metric[metric.name] = computation
            family_groups_by_metric[metric.name] = {
                str(row["group_id"])
                for row in rows
                if str(row["group_id"]) != str(design.control_group)
            }
            family_groups_by_metric[metric.name].update(
                str(cast(Any, key).group_id)
                for key in computation.failures
                if getattr(key, "group_id", None) is not None
            )
        refused_cells.extend(refused)

    family, non_family = _partition_secondary_family(secondary_entries)
    out: list[LiftEstimate] = []
    for metric, *_rest in non_family:
        out.extend(
            row.model_copy(update={"role": "secondary", "discovery": None})
            for row in nominal_by_metric[metric.name]
        )
    if not family:
        return out, refused_cells

    from increment.decision import ArmHypothesisKey

    family_cells: list[tuple[ArmHypothesisKey, object]] = []
    for metric, _config, _test, _is_quantile, _metric_methods in family:
        rows_for_metric = nominal_by_metric[metric.name]
        for group_id in sorted(family_groups_by_metric.get(metric.name, set())):
            key = ArmHypothesisKey(metric.name, group_id, "itt")
            row = next(
                (
                    candidate
                    for candidate in rows_for_metric
                    if candidate.group_id == group_id
                    and candidate.estimand == "itt"
                    and candidate.method_role == "decision"
                ),
                None,
            )
            family_cells.append((key, row))
    family_computation = _merge_decision_computations(
        [nominal_computation_by_metric[metric.name] for metric, *_rest in family]
    )
    outcome = select_family(
        decision_cells(family_cells),
        plan.q,
        _sequential_inference(plan),
        plan.alpha,
        computation=family_computation,
    )
    selected_cells = outcome.selected
    fcr_alpha = outcome.fcr_alpha
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    if fcr_alpha is None:
        out.extend(
            row.model_copy(
                update={
                    "role": "secondary",
                    "discovery": (
                        family_discovery(outcome, ArmHypothesisKey(row.metric, row.group_id, "itt"))
                        if row.method_role == "decision"
                        else None
                    ),
                    "family_axes": family_record["family_axes"]
                    if row.method_role == "decision"
                    else None,
                    "family_q": family_record["family_q"]
                    if row.method_role == "decision"
                    else None,
                    "family_threshold": (
                        family_record["family_threshold"] if row.method_role == "decision" else None
                    ),
                }
            )
            for metric, *_rest in family
            for row in nominal_by_metric[metric.name]
        )
        return out, refused_cells

    selected_names = {key.metric for key in selected_cells}
    reestimated: dict[str, list[LiftEstimate]] = {}
    for metric, config, test, is_quantile, _metric_methods in family:
        if metric.name not in selected_names:
            continue
        selected_alpha = _fcr_alpha_for(
            "two-sided"
            if _joint_relative_rows(nominal_by_metric[metric.name])
            else test.alternative,
            fcr_alpha,
        )
        reestimated[metric.name], refused, _computation = _estimate_pass(
            src,
            metric,
            evidence_by_metric[metric.name],
            test,
            config,
            design,
            alpha_for=selected_alpha,
            role="secondary",
            inference=None if is_quantile else effective_inference,
            advisory_seen=advisory_seen,
            methods=_metric_methods[:1],
        )
        reestimated[metric.name] = [
            open_bound_from_two_sided_at_target(r) for r in reestimated[metric.name]
        ]
        refused_cells.extend(refused)
    out.extend(
        _stamp_fcr_selected_rows(
            family,
            nominal_by_metric,
            reestimated,
            selected_cells,
            outcome,
            family_record,
        )
    )
    return out, refused_cells


def _estimate_randomized_non_secondary(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Randomized,
    plan: CompiledDecisionPlan,
    *,
    by: Sequence[str],
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]],
    observed_arms: set[str],
) -> tuple[
    list[LiftEstimate],
    list[tuple[str, str, str, str]],
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
]:
    """Estimate randomized primary, guardrail, and unassigned cells.

    Every arm read along the way is added to *observed_arms* so the caller
    can refuse a treatment-free readout without a second load.
    """
    out: list[LiftEstimate] = []
    refused_cells: list[tuple[str, str, str, str]] = []
    secondary_entries: list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]] = []
    for metric, config in zip(selected, configs, strict=True):
        test = plan.procedures[metric.name]
        is_quantile = getattr(metric, "type", None) == "quantile"
        metric_methods = _runtime_methods(config, design)
        if test.role == "secondary":
            secondary_entries.append((metric, config, test, is_quantile, metric_methods))
            continue
        if is_quantile:
            _refuse_unsupported_quantile(metric, test, cluster=src.context.cluster, by=by)
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            observed_arms |= metric_rows.observed_arms
            if metric_rows.n_treatment_arms == 0:
                continue
            if metric_methods == []:
                rows_est, refused = [], []
            else:
                cell_alpha = (
                    resolve_cell_alpha(plan, test, n_arms=metric_rows.n_treatment_arms, view=None)
                    if test.role == "primary"
                    else test.alpha
                )
                rows_est, refused, _computation = _estimate_pass(
                    src,
                    metric,
                    metric_rows.unit_frame,
                    test,
                    config,
                    design,
                    alpha_for=cell_alpha,
                    role=test.role if plan.declared else None,
                    inference=effective_inference,
                    advisory_seen=advisory_seen,
                    methods=metric_methods,
                )
        else:
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            observed_arms |= metric_rows.observed_arms
            if test.role == "primary" and metric_rows.n_treatment_arms == 0:
                continue
            cell_alpha = resolve_cell_alpha(
                plan, test, n_arms=metric_rows.n_treatment_arms, view=None
            )
            rows_est, refused, _computation = _estimate_pass(
                src,
                metric,
                list(metric_rows.rows),
                test,
                config,
                design,
                alpha_for=cell_alpha,
                role=test.role if plan.declared else None,
                inference=effective_inference,
                advisory_seen=advisory_seen,
                methods=metric_methods,
            )
        refused_cells.extend(refused)
        out.extend(rows_est)
    return out, refused_cells, secondary_entries


# Scheduled decomposition target; public parameter count is the API.
def run(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    _population: Literal["assigned", "triggered"] = "assigned",
) -> list[LiftEstimate]:
    """Whole-window relative lift, one row per (metric, method, non-control arm).

    Design and statistical plan are both read off *src* (``src.design``,
    ``src.plan``) rather than passed in - a `MomentSource` owns both as
    construction state. Every row is stamped with its resolved
    `role` (None only when `src.plan.declared` is False, i.e. no
    `AnalysisPlan` was ever declared).

    On the randomized fixed-horizon-or-sequential path: a primary's alpha
    (compiled procedure ``alpha``, already split evenly across every declared
    primary) is split again across this metric's own non-control arms; a
    guardrail/unassigned metric estimates at its compiled ``alpha`` unsplit. A
    secondary estimates every in-family (metric x arm) cell at the plan's
    nominal `alpha` first; under a fixed-horizon plan those p-values feed
    one `bh_select` across the whole secondary family, and the selected
    cells' intervals are re-estimated (a second pass over the same
    retained moments) at the Benjamini-Yekutieli FCR level; under an
    AlwaysValid plan, registered raw likelihood evidence drives `e_bh_select`.
    Selected intervals are reinverted at the capped FCR allocation from the
    same stopped checkpoint; unselected cells retain their nominal intervals.
    A non-family secondary (prior-bound, or a quantile metric under an
    AlwaysValid plan) estimates once at nominal, with no discovery verdict.

    Under Encouragement, dispatches to encouragement_rows, which delegates
    per metric group to estimate_encouragement instead of estimate_lift;
    estimands (default itt/compliance/late) picks which rows to report.
    Registered sequential plans instead evaluate their declared ITT and
    compliance cells; binary-uptake LATE is refused. Fixed-horizon per-metric
    shifted nulls (declared or plan-bound margins) are refused. A primary's alpha splits across
    its own treatment arms, an in-family secondary faces the same
    BH/e-BH selection every other design gets, and the design-level
    compliance row is estimated once at the plan's own alpha. A
    cluster on src uses cluster-level covariance: additive ITT and LATE
    intervals use Welch references, while relative ITT uses fixed-t Fieller
    sets with ``min(K_T - 1, K_C - 1)`` degrees of freedom. These are working
    approximations. Compliance reports the first-stage lift on the absolute
    scale; complier-relative LATE and any CUPED method are refused under a cluster.

    Under Observational, dispatches to estimate_ate (IPTW by default); a
    nonempty by= (per-segment IPTW) raises NotImplementedError. Only the
    absolute axis (a declared/plan-bound margin_abs) and value_scale are
    supported there; a relative-axis margin is refused. value_scale
    switches a metric to reporting the additive ATE in its own units, for
    when relative lift is unidentified because the control mean sits
    within 4 SEs of 0. This path applies the same role machinery the
    randomized path does: a primary's alpha is Bonferroni-split across its
    treatment arms, secondaries join a BH family at ``plan.q`` with
    Benjamini-Yekutieli FCR re-estimation for selected cells, and a
    guardrail keeps its full compiled ``alpha`` outside any family.

    decision_method, sensitivity_methods, and prior are call-wide overrides; without one, each metric
    falls back to its own declared value, then the design default.
    """
    design = _require_design(src, "run")
    selected, configs = _prepare_run_request(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        population=_population,
    )
    # Only percentile readouts need the raw source snapshot. Metadata/request
    # validation above is intentionally complete before capture.
    needs_snapshot = any(
        getattr(getattr(metric, "winsorization", None), "has_percentile", False)
        for metric in selected
    )
    if needs_snapshot:
        from increment._source_operations import ReadoutSnapshotOperation

        if isinstance(src, ReadoutSnapshotOperation):
            uptake = getattr(design, "uptake", None)
            uptake_facts = () if uptake is None else (uptake.fact,)
            with src.readout_snapshot(
                metrics=selected, population=_population, uptake_facts=uptake_facts
            ) as pinned:
                return _run_prepared(
                    pinned,
                    selected,
                    configs,
                    prior=prior,
                    by=by,
                    estimands=estimands,
                    value_scale=value_scale,
                )
    return _run_prepared(
        src,
        selected,
        configs,
        prior=prior,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
    )


def _run_prepared(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    *,
    prior: Prior | None | _Unset,
    by: Sequence[str],
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
) -> list[LiftEstimate]:
    """Estimate a validated request inside its source-owned snapshot lifetime."""
    design = _require_design(src, "run")
    plan = src.context.plan

    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout

        _refuse_segmented_registration(plan.inference)
        return sequential_readout(src, metrics=selected, estimands=estimands)
    call_prior = cast("Prior | None", None if prior is UNSET else prior)
    cluster = src.context.cluster
    if design.mechanism == "encouragement":
        if value_scale:
            _raise("readout.encouragement.value_scale")
        if isinstance(plan.inference, SEQUENTIAL_POLICIES):
            _raise("readout.sequential_inference_supported")
        compliance_only = _design_compliance_only(src, estimands)
        if by and compliance_only:
            _raise_readout_request("readout.run.segment_unsupported", metric="uptake")
        rows_by_metric: dict[str, list[Mapping[str, Any]]] = {}
        if not compliance_only:
            reject_quantile_metrics(
                selected,
                "run() under an encouragement design",
                reason="estimate_encouragement consumes mean-based group_summary moments, "
                "which a quantile metric has none of",
                remedy="Drop the encouragement design or the quantile metric.",
            )
            rows_by_metric = {
                metric.name: list(
                    _load_metric_rows(src, metric, by=by, control_group=design.control_group).rows
                )
                for metric in selected
            }
            if selected:
                _refuse_if_no_treatment_arm(
                    {str(row["group_id"]) for rows in rows_by_metric.values() for row in rows},
                    design.control_group,
                )
        return encouragement_rows(
            src=src,
            metrics=selected,
            rows_by_metric=rows_by_metric,
            configs=configs,
            design=design,
            plan=plan,
            estimands=estimands,
            cluster=cluster,
            caller="run() under an encouragement design",
        )
    if design.mechanism == "observational":
        # estimate_ate loads its own evidence per method; the arm gate reads
        # the total moments so a control-only source refuses with a stable code.
        if selected:
            _refuse_if_no_treatment_arm(
                {
                    str(row["group_id"])
                    for metric in selected
                    for row in cast("list[Mapping[str, Any]]", src.moments(metric))
                },
                design.control_group,
            )
        return _estimate_observational(
            src,
            selected,
            configs,
            design,
            plan,
            value_scale=value_scale,
            call_prior=call_prior,
        )
    if value_scale:
        _raise("readout.value_scale_randomized_absolute")
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None = _sequential_inference(
        plan
    )
    advisory_seen: set[tuple[str, str]] = set()
    observed_arms: set[str] = set()
    out, refused_cells, secondary_entries = _estimate_randomized_non_secondary(
        src,
        selected,
        configs,
        design,
        plan,
        by=by,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
    )
    secondary_out, secondary_refused = _estimate_randomized_secondary_family(
        src,
        secondary_entries,
        design,
        plan,
        by=by,
        effective_inference=effective_inference,
        advisory_seen=advisory_seen,
        observed_arms=observed_arms,
    )
    out.extend(secondary_out)
    refused_cells.extend(secondary_refused)
    if selected:
        _refuse_if_no_treatment_arm(observed_arms, design.control_group)
    if not out and refused_cells:
        _raise_if_all_lift_cells_refused(refused_cells)
    order = {metric.name: i for i, metric in enumerate(selected)}
    out.sort(key=lambda row: order[row.metric])
    return out


def arm_moments(
    src: MomentSource,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    population: Literal["assigned", "triggered"] = "assigned",
) -> list[LiftEstimate]:
    """Run the existing parallel arm-moment readout implementation."""
    return run(
        src,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        metrics=metrics,
        by=by,
        estimands=estimands,
        value_scale=value_scale,
        _population=population,
    )


def _validate_daily_inference(
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    *,
    method: str = "run_daily_lift",
) -> None:
    """Refuse sequential plans for disjoint daily and cohort slices.

    Routed through the same coded refusal the breakout path uses, so a caller
    reaching this facade gets the registered code and immutable context rather
    than an unstructured exception for the identical condition.
    """
    if inference is not None:
        _refuse(
            "readout.inference.disjoint_slices",
            view="daily" if method == "run_daily_lift" else "cohort",
            inference=type(inference).__name__,
        )


def daily(
    src: MomentSource,
    *,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    include_covariate: bool = False,
) -> list[dict[str, Any]]:
    """Per-day absolute metric values. Requires the source to offer daily grain.

    Absolute values only - carries no causal claim under an Observational
    design. metrics narrows the reported names; an unknown name raises
    before any moments run. ``include_covariate`` is an internal source
    requirement derived from requested methods, not a caller-facing
    statistical option.
    """
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="daily"
    )
    configs_by_name = {config.metric.name: config for config in src.context.configs}
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=[configs_by_name[m.name] for m in selected],
        view="daily",
        grain="daily",
        by=by,
    )
    validate_request(request)
    moment_kwargs = {"include_covariate": True} if include_covariate else {}
    return cast(
        "list[dict[str, Any]]",
        [
            r
            for m in selected
            for r in src.moments(
                m,
                grain="daily",
                by=by,
                **moment_kwargs,
            )
        ],
    )


def _load_asof_rows_by_metric(
    src: MomentSource,
    selected: Sequence[Metric],
    methods_by_metric: Mapping[str, list[Method]],
    *,
    by: Sequence[str],
    completed_windows_only: bool,
) -> dict[str, dict[Any, list[Mapping[str, Any]]]]:
    """Load and bucket each metric's as-of moments by date."""
    common_moment_kwargs: dict[str, Any] = (
        {"completed_windows_only": True} if completed_windows_only else {}
    )
    by_metric_date: dict[str, dict[Any, list[Mapping[str, Any]]]] = {}
    for metric in selected:
        moment_kwargs = dict(common_moment_kwargs)
        if any(method.variance_reduction == "cuped" for method in methods_by_metric[metric.name]):
            moment_kwargs["include_covariate"] = True
        rows = cast(
            "list[Mapping[str, Any]]",
            src.moments(metric, grain="asof", by=by, **moment_kwargs),
        )
        dated: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            dated.setdefault(row["ds"], []).append(row)
        by_metric_date[metric.name] = dated
    return by_metric_date


@dataclass(frozen=True)
class _AsofMetricContext:
    source: MomentSource
    config: ResolvedMetricConfig
    procedure: Any
    design: Randomized | Encouragement | Observational
    plan: CompiledDecisionPlan
    estimands: Sequence[str] | None
    methods: list[Method]


def _estimate_asof_metric_date(
    metric: Metric,
    date_rows: list[Mapping[str, Any]],
    date_value: Any,
    context: _AsofMetricContext,
) -> list[LiftEstimate]:
    """Estimate one fixed-horizon metric/date under the compiled role policy."""
    src, config, test = context.source, context.config, context.procedure
    design, plan, estimands = context.design, context.plan, context.estimands
    inference = _sequential_inference(plan)
    role_for_row = test.role if plan.declared else None
    if design.mechanism == "encouragement":
        n_arms = len({str(row["group_id"]) for row in date_rows} - {str(design.control_group)})
        cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
        results = estimate_encouragement(
            [metric],
            date_rows,
            design,
            estimands=tuple(
                name
                for name in (estimands if estimands is not None else ESTIMANDS)
                if name != "compliance"
            ),
            methods=context.methods,
            prior=config.prior,
            alpha=cell_alpha,
            alternative=test.alternative,
            inference=inference,
            method_roles=_runtime_method_roles(context.methods),
        ).results
        return [
            row.model_copy(
                update={
                    "ds": date_value,
                    "role": role_for_row,
                    "note": _asof_monitoring_note(
                        row.note,
                        estimand=row.estimand,
                        inference=inference,
                        design=design,
                    ),
                }
            )
            for row in results
        ]

    if test.role == "secondary":
        nominal, _refused, _computation = _estimate_pass(
            src,
            metric,
            date_rows,
            test,
            config,
            design,
            alpha_for=plan.alpha,
            role="secondary",
            inference=inference,
            advisory_seen=None,
            retry="whole_asof",
            methods=context.methods,
        )
        return [
            row.model_copy(
                update={
                    "ds": date_value,
                    "role": "secondary",
                    "discovery": None,
                    "note": _asof_monitoring_note(
                        row.note,
                        estimand=row.estimand,
                        inference=inference,
                        design=design,
                        secondary_fixed_horizon=True,
                    ),
                }
            )
            for row in nominal
        ]

    n_arms = len({str(row["group_id"]) for row in date_rows} - {str(design.control_group)})
    if test.role == "primary" and n_arms == 0:
        return []
    cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
    rows_est, _refused, _computation = _estimate_pass(
        src,
        metric,
        date_rows,
        test,
        config,
        design,
        alpha_for=cell_alpha,
        role=role_for_row,
        inference=inference,
        advisory_seen=None,
        retry="whole_asof",
        methods=context.methods,
    )
    return [
        row.model_copy(
            update={
                "ds": date_value,
                "role": role_for_row,
                "note": _asof_monitoring_note(
                    row.note,
                    estimand=row.estimand,
                    inference=inference,
                    design=design,
                ),
            }
        )
        for row in rows_est
    ]


def asof_lift(
    src: MomentSource,
    *,
    estimands: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    completed_windows_only: bool = False,
) -> list[LiftEstimate]:
    """As-of relative lift under the source's declared plan.

    Fixed-horizon rows are calculated per date with the existing role and
    multiplicity policy and carry the repeated-look caveat. Registered
    AlwaysValid returns the current labeled finalized joint-unit checkpoint;
    historical dates are not reconstructed from rounded moments. Its selected
    intervals invert the same stopped likelihood at the exact FCR allocation.
    Segmented histories require a per-date state contract and remain
    unsupported here.
    """
    if src.context.cluster is not None:
        _raise_readout_request("readout.asof.cluster", cluster=src.context.cluster)
    design = _require_design(src, "asof_lift")
    if by:
        _raise_readout_request("readout.asof.segment_unsupported", by=tuple(by))
    plan = src.context.plan
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="asof_lift"
    )
    resolved_configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    configs = {config.metric.name: config for config in resolved_configs}
    policy = plan.view_policies.for_view(
        "asof",
        mechanism=design.mechanism if design is not None else None,
        segmented=bool(by),
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=resolved_configs,
        view="asof",
        grain="asof",
        by=by,
        estimands=estimands,
        correction=normalize_display_correction(policy.correction),
        q=policy.q,
        completion_policy=completed_windows_only,
    )
    validate_request(request)
    _validate_encouragement_asof_inference(
        () if estimands is not None and set(estimands) == {"compliance"} else selected,
        design,
        inference=_sequential_inference(plan),
        completed_windows_only=completed_windows_only,
        method="readouts.asof_lift",
    )
    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_asof_readout

        _refuse_segmented_registration(plan.inference)
        return sequential_asof_readout(src, metrics=selected, estimands=estimands)
    if design.mechanism == "encouragement":
        declared = _declared_margin_names(selected)
        shifted = [
            metric.name
            for metric in selected
            if getattr(plan.procedures[metric.name], "null_lift", 0.0) != 0.0
            or getattr(plan.procedures[metric.name], "null_abs", None) is not None
        ]
        combined = list(dict.fromkeys([*shifted, *declared]))
        if combined:
            _raise("readout.metric_declare_non", combined=combined)
    from increment._frame_panel import _day_axis_label_order
    from increment._source_types import compliance_summary_series

    compliance_by_date = (
        compliance_summary_series(
            src,
            design,
            completed_windows_only=completed_windows_only,
        )
        if isinstance(design, Encouragement) and (estimands is None or "compliance" in estimands)
        else {}
    )
    outcome_metrics = [] if estimands is not None and set(estimands) == {"compliance"} else selected
    methods_by_metric = {
        metric.name: _runtime_methods(configs[metric.name], design) for metric in outcome_metrics
    }
    by_metric_date = _load_asof_rows_by_metric(
        src,
        outcome_metrics,
        methods_by_metric,
        by=by,
        completed_windows_only=completed_windows_only,
    )
    day_order = _day_axis_label_order(
        [*compliance_by_date, *(date for dated in by_metric_date.values() for date in dated)]
    )
    all_dates = sorted(day_order, key=day_order.__getitem__)
    out: list[LiftEstimate] = []
    contexts = {
        metric.name: _AsofMetricContext(
            src,
            configs[metric.name],
            plan.procedures[metric.name],
            design,
            plan,
            estimands,
            methods_by_metric[metric.name],
        )
        for metric in outcome_metrics
    }
    for date_value in all_dates:
        compliance_summary = compliance_by_date.get(date_value)
        for metric in outcome_metrics:
            date_rows = by_metric_date[metric.name].get(date_value)
            if date_rows is None:
                continue
            rows = _estimate_asof_metric_date(
                metric,
                date_rows,
                date_value,
                contexts[metric.name],
            )
            out.extend(rows)
        if compliance_summary is not None:
            assert isinstance(design, Encouragement)
            out.extend(
                r.model_copy(
                    update={
                        "ds": date_value,
                        "role": None,
                        "note": _asof_monitoring_note(
                            r.note,
                            estimand=r.estimand,
                            inference=_sequential_inference(plan),
                            design=design,
                        ),
                    }
                )
                for r in estimate_compliance(
                    compliance_summary,
                    design,
                    alpha=plan.alpha,
                    inference=_sequential_inference(plan),
                ).results
            )
    order = {metric.name: i for i, metric in enumerate(selected)}
    out.sort(
        key=lambda row: (
            day_order[row.ds],
            -1 if row.estimand == "compliance" else order[row.metric],
        )
    )
    return out


def breakout(
    src: MomentSource,
    dimension: str,
    *,
    source_name: str | None = None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str] | None = None,
    correction: Correction | None = None,
    q: float | None = None,
    estimands: Sequence[str] | None = None,
) -> BreakoutEstimates:
    """Relative lift estimated separately for every distinct value of dimension.

    Design and alpha/inference are all read off *src* (``src.design``,
    ``src.plan.alpha``, ``src.plan.inference``) rather than passed in -
    a `MomentSource` owns all three as construction state. `src.plan.alpha`/
    `.inference` are populated regardless of whether an `AnalysisPlan` was
    ever declared (`src.plan.declared`): an undeclared plan resolves to
    alpha=0.05 and inference=None.

    Delegates the randomized per-segment split to run_breakout, supplying
    each metric's moments broken out by dimension; a segment with no
    usable control arm comes back as an excluded row. Every row is
    stamped `role="exploratory"` - breakout has no plan-declared
    primary/secondary/guardrail role concept, unlike run().

    Under Encouragement, partitions moments the same way and runs
    estimate_encouragement once per (segment, metric) pair, at the
    plan's alpha directly (no split, no BH/FCR family machinery - only
    role stamping). A control-free segment warns and is skipped;
    siblings remain estimable, and a weak first stage only suppresses
    that segment's late row. correction (other than "none") and a
    declared inference are refused there.

    Refused under Observational: a per-segment contrast splits the
    adjustment set's confounding structure too, which is not identified -
    raises rather than emitting a confounded estimate.

    correction="bonferroni" divides alpha by the number of distinct
    dimension values present in each metric's own moments, computed per
    metric. correction=None (the default) resolves to "bh" under a
    randomized design, or "none" under Encouragement, which does not
    support correction - a caller EXPLICITLY passing a non-"none"
    correction under Encouragement is refused. correction="bh"
    runs run_breakout's flat BH/e-BH family across every (metric, arm,
    segment) cell this call produces in ONE run_breakout call (all
    selected metrics together, not one call per metric) at *q*,
    stamping `discovery` and re-estimating selected fixed-horizon intervals
    at the FCR level. Exact registered sequential breakouts select over their
    retained roster and reinvert selected intervals at the capped FCR
    allocation from the same stopped checkpoints. Corrected asymptotic-mean
    families instead use fixed-roster Bonferroni familywise inference. Any
    informative prior is refused with ``correction="bh"``.


    Under a fixed-horizon plan, a metric with a non-inferiority margin --
    declared on the Metric (margin/margin_abs) or bound on the plan
    (ExperimentMetric.margin/margin_abs) -- is refused with
    ``readout.margin.breakout``: fixed-horizon segment rows build no shifted
    null, so a margin would change what the same metric's stat_sig means
    between views. The whole-window guardrail read is run(). A registered
    sequential breakout instead tests each segment cell against its
    registered null, which must equal the compiled margin null.

    metrics narrows the reported names; an unknown one raises before any
    moments run.
    """
    design = _require_design(src, "breakout")
    plan = src.context.plan
    policy = plan.view_policies.for_view(
        "breakout",
        mechanism=design.mechanism if design is not None else None,
        segmented=True,
    )
    if correction is None:
        correction = normalize_display_correction(policy.correction)
    if q is None:
        # Omitted q always inherits the plan's, whether or not correction
        # was passed explicitly -- an explicit correction with no q is
        # NOT the same request as an explicit correction AND q.
        q = policy.q if policy.q is not None else plan.q
    correction = correction or "none"
    alpha = plan.alpha
    inference = _sequential_inference(plan)
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="breakout"
    )
    resolved_configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    configs = {config.metric.name: config for config in resolved_configs}
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=resolved_configs,
        view="breakout",
        grain="total",
        by=(dimension,),
        dimension=dimension,
        estimands=estimands,
        correction=correction,
        q=q,
    )
    validate_request(request)
    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_readout
        from increment.sequential_state import sequential_refuse

        rows = sequential_readout(src, metrics=selected, estimands=estimands)
        output = []
        for row in rows:
            cp = row.require_sequential_result().checkpoint
            if len(cp.cell.segment) != 1 or cp.cell.segment[0][0] != dimension:
                sequential_refuse(
                    "source.invalid", "breakout dimension differs from retained segment roster"
                )
            output.append(
                _breakout_estimate_row(
                    row,
                    dimension=dimension,
                    dimension_value=cp.cell.segment[0][1],
                    source=source_name,
                    n_treat=cp.treatment.n,
                    n_control=cp.control.n,
                    low_reliability=False,
                    discovery=row.discovery,
                    family_axes=row.family_axes,
                    family_q=row.family_q,
                    family_threshold=row.family_threshold,
                )
            )
        return BreakoutEstimates(output)

    if correction == "bh" and any(config.prior is not None for config in resolved_configs):
        _raise("readout.breakout_correction_bh")
    methods_by_metric = {
        config.metric.name: _runtime_methods(config, design) for config in resolved_configs
    }
    roles_by_metric = {
        name: _runtime_method_roles(methods) for name, methods in methods_by_metric.items()
    }
    if design.mechanism == "encouragement":
        requested_estimands = estimands if estimands is not None else ESTIMANDS
        unknown = set(requested_estimands) - set(ESTIMANDS)
        if unknown:
            _raise(
                "estimation.encouragement.unknown_estimand_supported",
                estimands=ESTIMANDS,
                unknown=sorted(unknown),
            )
        for metric_methods in methods_by_metric.values():
            _validate_methods(metric_methods)
        enc_results: list[BreakoutEstimate] = []
        for metric in selected:
            metric_methods = methods_by_metric[metric.name]
            rows = cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
            segments: dict[str, list[Mapping[str, Any]]] = {}
            for row in rows:
                segments.setdefault(str(row[dimension]), []).append(row)
            for value in sorted(segments):
                if not any(row.get("group_id") == design.control_group for row in segments[value]):
                    _warn(
                        "readouts.breakout.segment_no_control_arm",
                        dimension=dimension,
                        value=value,
                        control_group=design.control_group,
                        stacklevel=2,
                    )
                    continue
                n_by_group = {
                    str(row.get("group_id")): float(row["n"])
                    for row in segments[value]
                    if row.get("n") is not None
                }
                n_control_arm = n_by_group.get(design.control_group)
                for lift_estimate in estimate_encouragement(
                    [metric],
                    segments[value],
                    design,
                    estimands=requested_estimands,
                    methods=metric_methods,
                    prior=configs[metric.name].prior,
                    alpha=alpha,
                    method_roles=roles_by_metric[metric.name],
                ).results:
                    if lift_estimate.metric != metric.name:
                        lift_estimate = lift_estimate.model_copy(update={"metric": metric.name})
                    n_treat = n_by_group.get(str(lift_estimate.group_id))
                    enc_results.append(
                        _breakout_estimate_row(
                            lift_estimate,
                            dimension=dimension,
                            dimension_value=value,
                            source=source_name,
                            n_treat=n_treat,
                            n_control=n_control_arm,
                            low_reliability=(
                                (n_treat is not None and n_treat < DEFAULT_RELIABILITY_FLOOR)
                                or (
                                    n_control_arm is not None
                                    and n_control_arm < DEFAULT_RELIABILITY_FLOOR
                                )
                            ),
                        )
                    )
        return BreakoutEstimates(
            [row.model_copy(update={"policy_name": "compiled_plan"}) for row in enc_results]
        )
    results: list[BreakoutEstimate] = []
    if correction == "bh":
        # One flat family across every (metric, arm, segment) cell this
        # call produces: a single run_breakout call over every selected
        # metric together, not one call per metric (which would scope
        # the family to one metric at a time).
        combined_rows = [
            row
            for metric in selected
            for row in cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
        ]
        results.extend(
            run_breakout(
                combined_rows,
                selected,
                control_group=design.control_group,
                source=source_name,
                dimension=dimension,
                methods=None,
                prior=None,
                alpha=alpha,
                correction=correction,
                q=q,
                inference=inference,
                method_roles_by_metric=roles_by_metric,
                methods_by_metric=methods_by_metric,
                policy_name="compiled_plan",
            )
        )
    else:
        for metric in selected:
            rows = cast(
                "list[Mapping[str, Any]]", src.moments(metric, grain="total", by=[dimension])
            )
            metric_methods = methods_by_metric[metric.name]
            results.extend(
                run_breakout(
                    rows,
                    [metric],
                    control_group=design.control_group,
                    source=source_name,
                    dimension=dimension,
                    methods=metric_methods,
                    prior=configs[metric.name].prior,
                    alpha=alpha,
                    correction=correction,
                    inference=inference,
                    method_roles=roles_by_metric[metric.name],
                    policy_name="compiled_plan",
                )
            )
    return BreakoutEstimates(results)


def srm(
    src: MomentSource,
    *,
    expected: dict[str, float] | None = None,
    alpha: float = 0.001,
    inference: Literal["always_valid", "fixed"] = "always_valid",
) -> SRMResult | NotApplicable:
    """Sample-ratio-mismatch check on the source's per-group counts.

    Design is read off *src* (``src.design``) rather than passed in - a
    `MomentSource` owns it as construction state, per every other readout
    entry point.

    The default is an anytime-valid check for cumulative prefixes under a
    known allocation with the same conditional arm probabilities at every
    assignment. Independent categorical assignment suffices; blocked,
    adaptive, dependent, quota, exact-balance, and without-replacement
    protocols do not. Pass ``inference="fixed"`` for one predeclared
    Pearson look.

    Applies under `Encouragement` too - encouragement assignment IS
    randomized, only uptake is not; an SRM test on the assignment counts
    is exactly as meaningful as under `Randomized`. Not applicable under
    an `Observational` design: an SRM test presumes a target randomized
    allocation to compare observed counts against, which does not exist
    for a non-randomized comparison.

    When the source declares a randomization cluster, the chi-square runs
    over DISTINCT CLUSTER counts per arm (`SRMResult.grain == "cluster"`).
    Independently assigned clusters with known, constant conditional arm
    probabilities satisfy the anytime-valid contract. Per-arm unit counts ride
    along on `SRMResult.unit_counts` as context: unit imbalance under a clustered
    design is cluster-SIZE imbalance, which the randomizer never controlled.

    For ``inference="always_valid"``, pass ``expected`` or declare
    ``design.allocation``: its support is static and known before the
    cumulative prefix is observed. Fixed mode with neither expected nor
    design allocation uses equal observed-arm shares and returns
    ``log_e_value=None``; the fixed Pearson p-value controls ``is_srm``.

    ``alpha`` (default 0.001) is a diagnostic significance threshold for
    this randomization-integrity check, not part of the declared
    `AnalysisPlan` - unlike every other ``alpha=`` this effort removed
    from `run`/`breakout`/`asof_lift`/`daily`, it stays a call-time
    parameter deliberately: an SRM check answers "did the randomizer
    misbehave", a question with no plan-declared role/alpha to read.
    """
    design = src.context.design
    if design is None:
        _raise("readout.srm_source_declared")
    if design.mechanism not in ("randomized", "encouragement"):
        return NotApplicable(
            check="srm",
            reason=(
                "assignment was not randomized; a sample-ratio test presumes "
                "a target allocation to mismatch"
            ),
        )
    expected = resolve_srm_expected(
        expected,
        allocation=getattr(design, "allocation", None),
        inference=inference,
    )
    counts = complete_srm_support(src.unit_counts(), expected=expected)
    if src.context.cluster is None:
        return sample_ratio_mismatch(counts, expected=expected, alpha=alpha, inference=inference)
    # Accounting keys stay in the tested dict - sample_ratio_mismatch lifts
    # them onto their own fields with no degree of freedom.
    accounting = {
        key: counts.pop(key) for key in (UNASSIGNED_LABEL, MIXED_ASSIGNMENT_LABEL) if key in counts
    }
    return sample_ratio_mismatch(
        {
            **complete_srm_support(src.cluster_counts(), expected=expected),
            **accounting,
        },
        expected=expected,
        alpha=alpha,
        inference=inference,
        grain="cluster",
        unit_counts=counts,
    )


__all__ = [
    "arm_moments",
    "asof_lift",
    "breakout",
    "daily",
    "run",
    "srm",
]
