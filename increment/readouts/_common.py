from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import effective_methods
from increment._readout_request import _raise as _raise_readout_request
from increment._window import resolve_window_days
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
from increment.estimation.adjust import ESTIMATION_ADJUST_ESTIMATE_ATE_EVERY
from increment.estimation.encouragement import (
    ESTIMATION_ENCOURAGEMENT_UNKNOWN_ESTIMAND_SUPPORTED,
    READOUT_ENCOURAGEMENT_VALUE_SCALE,
)
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import (
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
)
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
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
            template="estimate_lift: every requested randomized metric/arm/method cell was refused by an inference guard -- no estimates remain. Failures: {failures!r}",
            keys=frozenset({"failures"}),
        ),
        "readout.value_scale_names": "value_scale= names metrics this source does not report on this call: {unknown_value_scale!r} (selected: {metric_names!r}) -- refusing rather than silently dropping the request",
        "readout.sequential_inference_supported": RefusalSpec(
            "readout.sequential_inference_supported",
            UnsupportedRequestError,
            template="sequential inference is not supported on the whole-window run() entry point under an encouragement design -- for sequential monitoring of an encouragement rollout, use asof_lift() instead, which carries time-uniform intervals on its ITT and compliance rows (binary-uptake LATE has no sequential interval; request estimands=('itt',) or ('compliance',)) -- declare inference=InferenceSpec(kind='always_valid', registration=...) on the source's AnalysisPlan (asof_lift() reads it from there, not a call kwarg)",
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
