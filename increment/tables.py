"""CoefTable renderers for experiment readouts and calendar metric trends.

Requires the tables extra (``coeftable`` and ``pandas``). Must not be
imported from ``increment.analysis``.

``readout_table`` accepts a narwhals-supported native frame (pandas,
polars, pyarrow, ...), a list of row dicts, or - via ``trend=`` - a raw
``Sequence[DailyLiftEstimate]``, converted internally via
``increment.breakout.estimates.to_frame`` and renamed to the main
table's ``segment`` column convention (``dimension_value`` -> ``segment``;
``method`` already matches). Other per-day result lists convert via that
same function directly instead.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from increment.breakout.estimates import BreakoutEstimate, DailyLiftEstimate, to_frame
from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation.contrast import contrast_evidence_available
from increment.estimation.contrast_results import ContrastResult
from increment.estimation.results import (
    BinomialConfidenceSet,
    LiftEstimate,
    _refuse_legacy_sampling,
)

if TYPE_CHECKING:
    from coeftable import CoefTable, Theme
    from coeftable.theme import Direction
    from narwhals.typing import IntoDataFrame


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "tables.readout_table_trend": "readout_table(trend=...) is missing the 'ds' column needed for the sparkline's x-axis -- got columns {columns}.",
        "tables.readout_table_trend_ds_all_null": "readout_table(trend=...): every row's 'ds' is null -- these estimates carry no per-date axis (a whole-window run() result?). Build the trend from readouts.asof_lift or run_daily_lift output, whose rows each carry their own day.",
        "tables.readout_table_trend_mixed_ds_basis": "readout_table(trend=...) mixes ds_basis values {bases}{affected}: a cohort-indexed series (ds is each unit's own exposure date) cannot share an x-axis with a calendar-indexed series. Plot the cohort view separately, or use the metric's as-of view (run_asof/run_asof_lift), which is calendar-indexed.",
        "tables.readout_table_trend_missing_key_columns": "readout_table(trend=...) is missing column(s) {missing} required by nest_by={nest_by!r} to align each trend row with the main table's metric/nest/split keys ({key_columns}).",
        "tables.readout_table_trend_duplicate_rows": RefusalSpec(
            "tables.readout_table_trend_duplicate_rows",
            InvalidRequestError,
            lambda *, rows, key, varying, key_columns: (
                f"readout_table(trend=...) has {rows} rows for "
                f"{key} -- expected exactly one. Column(s) {varying} vary "
                "within that group. Narrow `trend` (e.g. filter to one segment "
                "or method) so each (metric, "
                + ", ".join(key_columns[1:])
                + ", ds) combination resolves to a single row before passing "
                "it to readout_table."
            ),
        ),
        "tables.readout_table_nest": "readout_table(nest_by='segment') needs a single method in `data` -- the 'method' column has more than one distinct value. coeftable's CoefTable is 2-axis (metric rows x nest x split); segment and arm already occupy both slots, so there is no axis left for method. Filter `data` to one method before calling readout_table(nest_by='segment').",
        "tables.readout_table_ambiguous": "readout_table has ambiguous duplicate row keys {key}; provide distinct analysis_population values (for example 'assigned' and 'triggered').",
        "tables.readout_table_duplicate": "readout_table has duplicate visible row keys after disambiguation; include distinct metric, estimand, value_scale, nest, split, or analysis_population values.",
        "tables.trend_table_missing": "trend_table is missing required column(s) {missing} -- pass MetricTrend.to_frame() output",
        "tables.trend_table_supports": "trend_table supports at most one dimension column; got {dims}",
    },
)
_raise = raiser(_REFUSALS)

# LiftEstimate -> readout DataFrame adapter


def _binomial_stat_sig(bset: BinomialConfidenceSet, *, null_lift: float, alternative: str) -> bool:
    """``BreakoutEstimate``/``DailyLiftEstimate`` twin of ``LiftEstimate.stat_sig()``'s
    binomial branch -- those two models carry a persisted ``binomial_set`` but no
    ``stat_sig()`` method of their own, so this recomputes the same shifted-null Berger-Boos
    p-value from the set's own persisted counts/nuisance budget. Exact even for a set-only row
    with no finite point (``x_c == 0``): the tail test never reads the point estimate."""
    return bset.null_p_value(null_lift, alternative) < bset.decision_alpha


def _stat_sig(est: LiftEstimate | BreakoutEstimate | DailyLiftEstimate) -> bool | None:
    """Whether the interval excludes the null, honoring the row's own
    one-sided ``alternative`` (``Estimate.excludes`` checks both tails).
    Always False when ``low_reliability`` is set. A ``null_abs`` row
    decides against the additive interval (``abs_lb``/``abs_ub``) instead
    of the relative one, and takes precedence over a persisted
    ``binomial_set``, exactly as ``LiftEstimate.stat_sig()`` orders its own
    branches; missing additive endpoints report False rather than falling back
    to a relative interval. A missing ratio point does not erase additive evidence.
    ``LiftEstimate.stat_sig()`` for a ``LiftEstimate`` row -- the same
    decision, reused so a family-selection caller can check a row's own
    verdict without importing this (tables-extra-gated) module.

    A ``BreakoutEstimate``/``DailyLiftEstimate`` row carrying a
    persisted ``binomial_set`` (point-backed or, exclusively, a
    set-only zero-control-count row with ``lift=None``), with no
    ``null_abs`` guardrail active, instead tests its shifted null
    directly against that set via :func:`_binomial_stat_sig` -- the
    exact same construction ``LiftEstimate.stat_sig()`` uses, so a
    set-only row's real evidence is never erased just because it has no
    finite point.
    An explicit producer failure has unavailable significance (``None``), not
    a non-rejection.
    """
    if getattr(est, "failure_code", None) is not None:
        return None
    if est.sequential_result is not None:
        return est.sequential_result.rejects()
    sampling_available = getattr(est, "sampling_available", None)
    if sampling_available is False:
        return None
    if sampling_available is None and (
        isinstance(est, DailyLiftEstimate) or getattr(est, "prior_shrunk", False)
    ):
        _refuse_legacy_sampling(est)
    if getattr(est, "low_reliability", False):
        return False
    if isinstance(est, LiftEstimate):
        return est.stat_sig()
    lift = est.lift
    null_abs = getattr(est, "null_abs", None)
    if null_abs is not None:
        abs_lb, abs_ub = getattr(est, "abs_lb", None), getattr(est, "abs_ub", None)
        if est.alternative == "greater":
            return abs_lb is not None and abs_lb > null_abs
        if est.alternative == "less":
            return abs_ub is not None and abs_ub < null_abs
        return abs_lb is not None and abs_ub is not None and not (abs_lb <= null_abs <= abs_ub)
    relative = est.relative_confidence_set
    if relative is not None:
        return relative.contains(est.null_lift) is False
    if est.relative_unavailable_reason is not None:
        return False
    bset = getattr(est, "binomial_set", None)
    if bset is not None:
        return _binomial_stat_sig(
            bset, null_lift=getattr(est, "null_lift", 0.0), alternative=est.alternative
        )
    if lift is None:
        return False
    null_lift = getattr(est, "null_lift", 0.0)
    if est.alternative == "greater":
        return lift.lb is not None and lift.lb > null_lift
    if est.alternative == "less":
        return lift.ub is not None and lift.ub < null_lift
    return lift.excludes(null_lift)


def _base_liftestimate_to_row(
    est: LiftEstimate | BreakoutEstimate | DailyLiftEstimate,
) -> dict[str, Any]:
    """Preserve points, canonical confidence evidence, and actual decision metadata."""
    lift = est.lift
    region = getattr(est, "confidence_set", None)
    relative = getattr(est, "relative_confidence_set", None)
    binomial = getattr(est, "binomial_set", None)
    open_side = lift.open_side if lift is not None else None
    if region is not None:
        lower, higher, level = region.lower, region.upper, region.level
    elif lift is not None:
        lower, higher, level = lift.lb, lift.ub, lift.level
    elif binomial is not None:
        lower, higher, level = binomial.lower, binomial.upper, binomial.level
    else:
        lower = higher = level = None
    row = {
        "metric": est.metric,
        "method": est.method,
        "method_role": getattr(est, "method_role", "decision"),
        "lift": lift.value if lift is not None else None,
        "lower": lower,
        "higher": higher,
        "open_side": open_side,
        "level": level,
        "stat_sig": _stat_sig(est),
        "null_lift": getattr(est, "null_lift", 0.0),
        "null_abs": getattr(est, "null_abs", None),
        "abs_lb": getattr(est, "abs_lb", None),
        "abs_ub": getattr(est, "abs_ub", None),
        "relative_confidence_set": relative,
        "relative_unavailable_reason": getattr(est, "relative_unavailable_reason", None),
        "confidence_set": region,
        "binomial_set": binomial,
        "abs_reference_kind": getattr(est, "abs_reference_kind", None),
        "abs_reference_df": getattr(est, "abs_reference_df", None),
        "abs_alpha": getattr(est, "abs_alpha", None),
        "estimand": getattr(est, "estimand", "itt"),
        "value_scale": getattr(est, "value_scale", "relative"),
        "preferred_direction": getattr(est, "preferred_direction", None),
        "note": getattr(est, "note", None),
        "inference": getattr(est, "inference", "fixed"),
        "analysis_population": getattr(est, "analysis_population", "assigned"),
        "role": getattr(est, "role", None),
        "discovery": getattr(est, "discovery", None),
        "family_axes": getattr(est, "family_axes", None),
        "family_q": getattr(est, "family_q", None),
        "family_threshold": getattr(est, "family_threshold", None),
        "family_size": getattr(est, "family_size", None),
        "group_id": est.group_id,
    }
    components = getattr(est, "posterior_components", None)
    if components is None:
        components_json = None
    else:
        from increment._canonical import canonical_json_bytes

        components_json = canonical_json_bytes(components.model_dump(mode="json")).decode()
    row.update(
        sampling_available=getattr(est, "sampling_available", None),
        sampling_reason_code=getattr(est, "sampling_reason_code", None),
        sampling_reason_context=getattr(est, "sampling_reason_context", None),
        posterior_available=getattr(est, "posterior_available", None),
        posterior_reason_code=getattr(est, "posterior_reason_code", None),
        posterior_reason_context=getattr(est, "posterior_reason_context", None),
        posterior_model=getattr(est, "posterior_model", None),
        posterior_scale=getattr(est, "posterior_scale", None),
        posterior_estimate=getattr(est, "posterior_estimate", None),
        posterior_lb=getattr(est, "posterior_lb", None),
        posterior_ub=getattr(est, "posterior_ub", None),
        posterior_level=getattr(est, "posterior_level", None),
        posterior_alpha=getattr(est, "posterior_alpha", None),
        posterior_latent_mean=getattr(est, "posterior_latent_mean", None),
        posterior_latent_sd=getattr(est, "posterior_latent_sd", None),
        posterior_prob_favorable=getattr(est, "posterior_prob_favorable", None),
        posterior_components=components_json,
        failure_code=getattr(est, "failure_code", None),
        failure_context=getattr(est, "failure_context", None),
        source_snapshot_id=getattr(est, "source_snapshot_id", None),
        decision_scope_complete=getattr(est, "decision_scope_complete", None),
        decision_scope_reason_code=getattr(est, "decision_scope_reason_code", None),
        decision_scope_reason_context=getattr(est, "decision_scope_reason_context", None),
        family_id=getattr(est, "family_id", None),
        multiplicity_status=getattr(est, "multiplicity_status", None),
        weight_diagnostics_available=getattr(est, "weight_diagnostics_available", None),
        weight_diagnostics_reason_code=getattr(est, "weight_diagnostics_reason_code", None),
        weight_diagnostics_reason_context=getattr(est, "weight_diagnostics_reason_context", None),
        weight_definition=getattr(est, "weight_definition", None),
        weight_grain=getattr(est, "weight_grain", None),
        control_weight_ess=getattr(est, "control_weight_ess", None),
        treatment_weight_ess=getattr(est, "treatment_weight_ess", None),
        control_weight_max_share=getattr(est, "control_weight_max_share", None),
        treatment_weight_max_share=getattr(est, "treatment_weight_max_share", None),
        control_weight_n=getattr(est, "control_weight_n", None),
        treatment_weight_n=getattr(est, "treatment_weight_n", None),
    )
    if region is not None:
        row.update(
            reference_kind="confidence_set",
            inference_method=region.method,
            null_kind=region.null_kind,
            lower_status=region.relative.lower.status,
            higher_status=region.relative.upper.status,
            lower_reason=region.relative.lower.reason,
            higher_reason=region.relative.upper.reason,
            abs_lower_status=region.additive.lower.status,
            abs_higher_status=region.additive.upper.status,
            abs_lower_reason=region.additive.lower.reason,
            abs_higher_reason=region.additive.upper.reason,
        )
    return row


def _liftestimate_to_row(
    est: LiftEstimate | BreakoutEstimate | DailyLiftEstimate,
) -> dict[str, Any]:
    row = _base_liftestimate_to_row(est)
    result = est.sequential_result
    if result is not None:
        from increment.estimation.sequential_result import SequentialInferenceResult
        from increment.estimation.sequential_runtime import _outward

        row.update(
            lower=_outward(result.bounds.lower, lower=True),
            higher=_outward(result.bounds.upper, lower=False),
            level=1.0 - float(result.bounds.alpha),
            sequential_alpha=str(result.bounds.alpha),
            sequential_status=result.bounds.status,
            sequential_log_e=str(result.log_e)
            if isinstance(result, SequentialInferenceResult)
            else None,
            sequential_validity_regime=getattr(result, "validity_regime", "finite_sample"),
            sequential_components=(
                None
                if isinstance(result, SequentialInferenceResult)
                else [component.model_dump(mode="json") for component in result.bounds.components]
            ),
            sequential_point_reason=result.point_reason,
            sequential_result=result.model_dump_json(),
        )
    return row


def _decision_stat_columns(
    est: LiftEstimate | BreakoutEstimate | DailyLiftEstimate,
) -> dict[str, Any]:
    """Best-effort posterior-derived values, explicitly model-qualified."""
    if est.sequential_result is not None or getattr(est, "n_clusters", None) is not None:
        return {"posterior_chance_to_beat": None, "posterior_risk_if_shipped": None}
    if (
        getattr(est, "failure_code", None) is not None
        or getattr(est, "sampling_available", None) is False
        or getattr(est, "excluded", None) is not None
        or getattr(est, "unavailable", None) is not None
        or getattr(est, "confidence_set", None) is not None
        or getattr(est, "binomial_set", None) is not None
        or getattr(est, "relative_confidence_set", None) is not None
        or getattr(est, "relative_unavailable_reason", None) is not None
    ):
        return {"posterior_chance_to_beat": None, "posterior_risk_if_shipped": None}
    try:
        if est.alternative == "less" or getattr(est, "preferred_direction", None) == "decrease":
            return {
                "posterior_chance_to_beat": est.chance_to_beat_favorable(),  # ty: ignore[unresolved-attribute]
                "posterior_risk_if_shipped": est.risk_if_shipped_favorable(),  # ty: ignore[unresolved-attribute]
            }
        return {
            "posterior_chance_to_beat": est.chance_to_beat(),  # ty: ignore[unresolved-attribute]
            "posterior_risk_if_shipped": est.risk_if_shipped(),  # ty: ignore[unresolved-attribute]
        }
    except (AttributeError, ValueError):
        return {"posterior_chance_to_beat": None, "posterior_risk_if_shipped": None}



def contrast_results_to_readout(
    results: Sequence[ContrastResult],
) -> list[dict[str, Any]]:
    """Convert fixed switchback contrasts to readout-table row dictionaries."""
    rows: list[dict[str, Any]] = []
    for result in results:
        estimate = result.estimate
        # Sample-SE availability applies only to the explicit t approximation.
        if not contrast_evidence_available(result):
            stat_sig = False
        elif result.method == "switchback_unit_variance_envelope":
            assert result.residual_p_value is not None
            stat_sig = result.residual_p_value < result.alpha
        elif result.alternative == "greater":
            stat_sig = estimate.lb is not None and estimate.lb > result.null_abs
        elif result.alternative == "less":
            stat_sig = estimate.ub is not None and estimate.ub < result.null_abs
        else:
            stat_sig = (
                estimate.lb is not None
                and estimate.ub is not None
                and not (estimate.lb <= result.null_abs <= estimate.ub)
            )
        rows.append(
            {
                "metric": result.metric,
                "method": result.method,
                "method_role": result.method_role,
                "lift": estimate.value,
                "lower": estimate.lb,
                "higher": estimate.ub,
                "level": estimate.level,
                "stat_sig": stat_sig,
                "group_id": result.treatment_group,
                "control_group": result.control_group,
                "treatment_group": result.treatment_group,
                "null_lift": 0.0,
                "null_abs": result.null_abs,
                "abs_lb": estimate.lb,
                "abs_ub": estimate.ub,
                "estimand": result.estimand,
                "value_scale": "absolute",
                "preferred_direction": result.preferred_direction,
                "note": result.identifying_assumption,
                "identifying_assumption": result.identifying_assumption,
                "inference": result.inference,
                "analysis_population": "assigned",
                "role": result.role,
                "assignment": result.assignment,
                "randomization_law": result.randomization_law,
                "independence_grain": result.independence_grain,
                "carryover_order": result.carryover_order,
                "observation_steps": result.observation_steps,
                "retained_steps": result.retained_steps,
                "reference": result.reference,
                "aggregation": result.aggregation,
                "probability_ct": result.probability_ct,
                "standard_error": result.standard_error,
                "n_units": result.n_units,
                "n_cycles": result.n_cycles,
                "n_blocks": result.n_blocks,
                "ct_cycles": result.ct_cycles,
                "tc_cycles": result.tc_cycles,
                "dof": result.dof,
                "alternative": result.alternative,
                "alpha": result.alpha,
                "open_side": estimate.open_side,
                "dof_unavailable_reason": result.dof_unavailable_reason,
                "standard_error_unavailable_reason": result.standard_error_unavailable_reason,
                "mean_slope": result.mean_slope,
                "minimum_cycles_per_unit": result.minimum_cycles_per_unit,
                "maximum_cycles_per_unit": result.maximum_cycles_per_unit,
                "washout_steps": result.washout_steps,
                "effective_alpha": result.effective_alpha,
                "source_snapshot_id": result.source_snapshot_id,
                "decision_scope_complete": result.decision_scope_complete,
                "decision_scope_reason_code": result.decision_scope_reason_code,
                "decision_scope_reason_context": result.decision_scope_reason_context,
                "refusal_probability_upper": result.refusal_probability_upper,
                "residual_cutoff": result.residual_cutoff,
                "residual_p_value": result.residual_p_value,
                "response_meaning": result.response_meaning,
                "reference_spec": (
                    result.reference_spec.model_dump(mode="json")
                    if result.reference_spec is not None
                    else None
                ),
                "provenance": (
                    result.provenance.model_dump(mode="json")
                    if result.provenance is not None
                    else None
                ),
                "discovery": None,
                "family_axes": None,
                "family_q": None,
                "family_threshold": None,
                "family_size": None,
            }
        )
    return rows


def estimates_to_readout(
    estimates: Sequence[LiftEstimate | BreakoutEstimate | DailyLiftEstimate | ContrastResult],
) -> list[dict[str, Any]]:
    """Convert estimates into readout rows with explicitly posterior-qualified values.

    Rows from a scoped collection also carry ``view_partial`` from their
    source metadata; filtered views remain explicitly partial. Plain lists
    leave that scope status unknown.
    """
    metadata = getattr(estimates, "metadata", None)
    view_partial = None if metadata is None else metadata.partial
    rows = []
    for est in estimates:
        if isinstance(est, ContrastResult):
            row = contrast_results_to_readout([est])[0]
            row["posterior_chance_to_beat"] = None
            row["posterior_risk_if_shipped"] = None
            row["posterior_prob_favorable"] = None
            rows.append(row)
            continue
        row = _liftestimate_to_row(est)
        row["view_partial"] = view_partial
        if hasattr(est, "dimension_value") and hasattr(est, "dimension"):
            row["segment"] = est.dimension_value
            row["dimension"] = est.dimension
            source = getattr(est, "source", None)
            if source is not None:
                row["source"] = source
            if hasattr(est, "excluded"):
                row["excluded"] = est.excluded
            if hasattr(est, "unavailable"):
                row["unavailable"] = est.unavailable
            row["low_reliability"] = getattr(est, "low_reliability", False)
        ds = getattr(est, "ds", None)
        if ds is not None:
            row["ds"] = ds
        row.update(_decision_stat_columns(est))
        rows.append(row)
    return rows


# Tables - coeftable readout table


def _resolve_trend_frame(
    trend: Any,
    *,
    nest_by: Literal["arm", "method", "segment"],
    nest_column: str,
    split_column: str,
    split_columns: str | None,
    identity_labels: dict[tuple[object, ...], str],
) -> Any:
    """Align trend metric/nest/split keys with the resolved headline identities.
    Reject missing keys, all-null dates, mixed date bases and repeated
    identities at one date; layout requirements match ``readout_table``.
    """
    import narwhals as nw

    if isinstance(trend, Sequence):
        trend_estimates = list(cast("Sequence[Any]", trend))
        # Infer the model from the first element so an as-of LiftEstimate
        # series renders like a DailyLiftEstimate series; empty input keeps the daily schema.
        trend_native = to_frame(
            trend_estimates, model=DailyLiftEstimate if not trend_estimates else None
        )
    else:
        trend_native = trend
    frame = nw.from_native(trend_native, eager_only=True)

    # `method` already matches the main table's column name; only
    # `dimension_value` (a segment's own name in a dimensioned trend row)
    # needs renaming to line up with `readout_table`'s `segment` column.
    rename_map = {
        old: new for old, new in (("dimension_value", "segment"),) if old in frame.columns
    }
    if rename_map:
        frame = frame.rename(rename_map)

    if "ds" not in frame.columns:
        _raise("tables.readout_table_trend", columns=frame.columns)
    if len(frame) > 0 and bool(frame["ds"].is_null().all()):
        _raise("tables.readout_table_trend_ds_all_null")

    # `ds` means a different date depending on ds_basis (observation date
    # vs. each unit's own exposure date); refuse a frame that mixes bases.
    if "ds_basis" in frame.columns:
        bases = sorted(b for b in frame["ds_basis"].unique().to_list() if b is not None)
        if len(bases) > 1:
            if "metric" in frame.columns:
                per_basis = "; ".join(
                    f"{basis}: "
                    f"{sorted(frame.filter(nw.col('ds_basis') == basis)['metric'].unique().to_list())}"
                    for basis in bases
                )
                affected = f" (metrics by basis -- {per_basis})"
            else:
                affected = ""
            _raise("tables.readout_table_trend_mixed_ds_basis", bases=bases, affected=affected)

    key_columns = ["metric", nest_column]
    if split_columns is not None:
        key_columns.append(split_column)

    missing = [c for c in key_columns if c not in frame.columns]
    if missing:
        _raise(
            "tables.readout_table_trend_missing_key_columns",
            missing=missing,
            nest_by=nest_by,
            key_columns=key_columns,
        )

    metrics = frame["metric"].to_list()
    estimands = (
        frame["estimand"].to_list() if "estimand" in frame.columns else ["itt"] * len(metrics)
    )
    relabeled = [
        f"{metric} ({estimand})" if isinstance(estimand, str) and estimand != "itt" else metric
        for metric, estimand in zip(metrics, estimands, strict=True)
    ]
    scales = (
        frame["value_scale"].to_list()
        if "value_scale" in frame.columns
        else ["relative"] * len(metrics)
    )
    nests = frame[nest_column].to_list()
    splits = frame[split_column].to_list() if split_columns is not None else [None] * len(metrics)
    populations = (
        frame["analysis_population"].to_list()
        if "analysis_population" in frame.columns
        else ["assigned"] * len(metrics)
    )
    labels = [
        identity_labels.get(
            (
                metric,
                estimand,
                "relative" if _is_missing(scale) else scale,
                nest,
                split,
                "assigned" if _is_missing(population) else population,
            ),
            base_label,
        )
        for metric, estimand, scale, nest, split, population, base_label in zip(
            metrics,
            estimands,
            scales,
            nests,
            splits,
            populations,
            relabeled,
            strict=True,
        )
    ]
    frame = frame.with_columns(
        nw.new_series("metric", labels, backend=nw.get_native_namespace(frame))
    )

    group_keys = [*key_columns, "ds"]
    counts = frame.group_by(group_keys).agg(nw.len().alias("__n"))
    duplicated = counts.filter(nw.col("__n") > 1)
    if len(duplicated) > 0:
        key_tuple = {k: v for k, v in duplicated.rows(named=True)[0].items() if k != "__n"}
        condition = nw.all_horizontal(
            *(nw.col(column) == value for column, value in key_tuple.items()),
            ignore_nulls=True,
        )
        matching_rows = frame.filter(condition)
        other_columns = [c for c in frame.columns if c not in group_keys]
        varying = [c for c in other_columns if matching_rows[c].n_unique() > 1]
        _raise(
            "tables.readout_table_trend_duplicate_rows",
            rows=len(matching_rows),
            key=key_tuple,
            varying=varying,
            key_columns=key_columns,
        )

    return frame


# Rendered decision / metadata columns


def _is_missing(value: Any) -> bool:
    """``None``, or a float NaN (pandas' fill for keys absent in a mixed
    row list) - the two shapes "no value" takes in a readout frame."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _column_values(frame: Any, column: str) -> list[Any] | None:
    return frame[column].to_list() if column in frame.columns else None


def _format_level(level: float) -> str:
    """Interval level as a percentage, trailing zeros dropped: 95, 90,
    98.33. A Bonferroni-corrected level is an exact fraction, so plain
    ``%g`` would print 98.3333 - two decimals is the display grain."""
    return f"{level * 100:.2f}".rstrip("0").rstrip(".")


def _format_metadata_stat(value: Any, *, percentage: bool) -> str:
    """Format a rendered decision statistic in its row's display units."""
    if _is_missing(value):
        return ""
    if percentage:
        return f"{value * 100:.1f}%"
    return f"{value:+.2f}"


def _failure_disclosure(code: Any, context: Any) -> str:
    if _is_missing(code):
        return ""
    details = []
    if isinstance(context, Mapping):
        reason = context.get("reason")
        if not _is_missing(reason):
            details.append(str(reason))
        else:
            details.extend(f"{key}={value!r}" for key, value in sorted(context.items()))
    return f"{code}: {', '.join(details)}" if details else str(code)


def _interval_level_note(
    frame: Any, *, show_interval_level: bool, has_confidence_sets: bool
) -> str | None:
    """Header disclosure when every row shares ONE non-95% level - a
    Bonferroni-corrected breakout, or a table of one-sided rows. Stating
    it once beats a column repeating the same cell on every row; varying
    levels stay per-row (``_rendered_metadata_columns``). ``None`` when
    the caller opted out, levels are absent, vary, or sit at 95%."""
    if not show_interval_level:
        return None
    levels = _column_values(frame, "level")
    if levels is None:
        return None
    distinct = {v for v in levels if not _is_missing(v)}
    if len(distinct) != 1:
        return None
    (level,) = distinct
    if math.isclose(level, 0.95):
        return None
    scope = "display intervals" if has_confidence_sets else "all intervals"
    return f"{scope} {_format_level(level)}%"


def _rendered_metadata_columns(
    frame: Any, *, advisory: bool, show_interval_level: bool
) -> dict[str, list[str]]:
    """Pre-render the decision-layer and row-metadata display strings
    ``readout_table`` attaches as coeftable passthrough columns.

    Returns ``{column label: per-row strings}``; a column is included only
    when its source column exists and has at least one real value, so a
    minimal hand-built frame renders unchanged.
    ``Posterior chance to beat``, ``Posterior risk if shipped``, and
    ``Posterior P(favorable)`` are explicit model-qualified values and render
    only when ``advisory`` is set. ``Discovery`` renders a row's family verdict
    ("Yes"/"No"/blank) as its own column - a family-level BH/e-BH selection
    outcome, never conflated with ``stat_sig`` (a single row's own
    interval-excludes-null check). ``Interval`` renders per-row confidence
    levels only when they differ across rows; one shared level is a header
    note instead. No column renders a ship/no-ship verdict.
    """
    n = len(frame)
    columns: dict[str, list[str]] = {}
    failure_codes = _column_values(frame, "failure_code")
    if failure_codes is not None and any(not _is_missing(value) for value in failure_codes):
        failure_contexts = _column_values(frame, "failure_context") or [None] * n
        columns["Failure"] = [
            _failure_disclosure(code, context)
            for code, context in zip(failure_codes, failure_contexts, strict=True)
        ]

    decision_scope = _column_values(frame, "decision_scope_complete")
    if decision_scope is not None and any(not _is_missing(value) for value in decision_scope):
        columns["Decision scope"] = [
            "" if _is_missing(value) else "Complete" if bool(value) else "Incomplete"
            for value in decision_scope
        ]

    partial_view = _column_values(frame, "view_partial")
    if partial_view is not None and any(not _is_missing(value) for value in partial_view):
        columns["Readout view"] = [
            "" if _is_missing(value) else "Partial view" if bool(value) else "Full view"
            for value in partial_view
        ]
    null_lift = _column_values(frame, "null_lift") or [None] * n
    null_abs = _column_values(frame, "null_abs") or [None] * n
    row_scales = _column_values(frame, "value_scale") or ["relative"] * n

    if advisory:
        for base, label in (
            ("posterior_chance_to_beat", "Posterior chance to beat"),
            ("posterior_risk_if_shipped", "Posterior risk if shipped"),
        ):
            values = _column_values(frame, base)
            if values is not None and any(not _is_missing(v) for v in values):
                columns[label] = [
                    _format_metadata_stat(
                        value,
                        percentage=base == "posterior_chance_to_beat" or scale != "absolute",
                    )
                    for value, scale in zip(values, row_scales, strict=True)
                ]
        prob = _column_values(frame, "posterior_prob_favorable")
        any_shifted_null = any(not _is_missing(v) and v != 0.0 for v in null_lift) or any(
            not _is_missing(v) for v in null_abs
        )
        if prob is not None and any_shifted_null and any(not _is_missing(v) for v in prob):
            columns["Posterior P(favorable)"] = [
                "" if _is_missing(v) else f"{v * 100:.1f}%" for v in prob
            ]

    multiplicity_status = _column_values(frame, "multiplicity_status")
    if multiplicity_status is not None:
        labels = {
            "undeclared_plan": "Unadjusted (no declared plan)",
            "unassigned_in_plan": "Unadjusted (unassigned in plan)",
            "declared_plan": "Declared plan",
            "exploratory_unadjusted": "Exploratory (unadjusted)",
            "exploratory_family": "Exploratory family",
        }
        if any(not _is_missing(value) for value in multiplicity_status):
            columns["Multiplicity"] = [
                "" if _is_missing(value) else labels.get(value, str(value))
                for value in multiplicity_status
            ]

    discovery = _column_values(frame, "discovery")
    if discovery is not None and any(not _is_missing(v) for v in discovery):
        columns["Discovery"] = ["" if _is_missing(v) else ("Yes" if v else "No") for v in discovery]
    levels = _column_values(frame, "level")
    if show_interval_level and levels is not None:
        distinct = {v for v in levels if not _is_missing(v)}
        if len(distinct) > 1:
            columns["Interval"] = [
                "" if _is_missing(v) else f"{_format_level(v)}% CI" for v in levels
            ]

    evidence_columns = [
        _column_values(frame, name)
        for name in (
            "confidence_set",
            "relative_confidence_set",
            "binomial_set",
            "relative_unavailable_reason",
        )
    ]
    if any(values is not None for values in evidence_columns):
        missing = [None] * n
        set_text = [
            _format_confidence_set(
                value,
                relative=relative,
                binomial=binomial,
                unavailable=unavailable,
                scale=scale,
                lift=lift,
            )
            for value, relative, binomial, unavailable, scale, lift in zip(
                *(values if values is not None else missing for values in evidence_columns),
                _column_values(frame, "value_scale") or ["relative"] * n,
                _column_values(frame, "lift") or missing,
                strict=True,
            )
        ]
        if any(set_text):
            columns["Confidence set"] = set_text
            significance = _column_values(frame, "stat_sig") or missing
            verdicts = []
            for text, sig, row in zip(
                set_text, significance, frame.iter_rows(named=True), strict=True
            ):
                if not text or _is_missing(sig):
                    verdict = ""
                elif sig:
                    verdict = "Yes"
                else:
                    verdict = "No" if _decision_available(row) else "Unavailable"
                verdicts.append(verdict)
            columns["Significant"] = verdicts
    binomial_sets = _column_values(frame, "binomial_set")
    if binomial_sets is not None:
        qualifications = [_binomial_qualification_disclosure(value) for value in binomial_sets]
        if any(qualifications):
            columns["Numerical qualification"] = qualifications
    return columns


def _binomial_qualification_disclosure(value: Any) -> str:
    if _is_missing(value):
        return ""
    qualification = (
        value.get("numerical_qualification")
        if isinstance(value, Mapping)
        else getattr(value, "numerical_qualification", None)
    )
    if qualification == "scipy_special_function_error_model_conditional_v1":
        return "Computed enclosure conditional on deployed SciPy/Boost special-function error model"
    if qualification == "legacy_unrecorded_v1":
        return "Numerical qualification not recorded in legacy result"
    return "No numerical qualification claim"


def _decision_available(row: Mapping[str, Any]) -> bool:
    """Distinguish a non-rejecting confidence procedure from unavailable inference."""
    status = row.get("sequential_status")
    if not _is_missing(status):
        return status in {
            "empty",
            "full-domain",
            "interval",
            "full",
            "ray",
            "bounded",
            "disconnected",
        }
    low_reliability = row.get("low_reliability")
    if not _is_missing(low_reliability) and low_reliability:
        return False
    if row.get("stat_sig"):
        return True
    absolute = not _is_missing(row.get("null_abs"))
    region = row.get("confidence_set")
    if region is not None and not _is_missing(region):
        interval = (
            region.additive if absolute or row.get("value_scale") == "absolute" else region.relative
        )
        return all(endpoint.status != "undefined" for endpoint in (interval.lower, interval.upper))
    if absolute:
        lower = not _is_missing(row.get("abs_lb"))
        higher = not _is_missing(row.get("abs_ub"))
        return (
            (lower and higher)
            or (lower and row.get("alternative") == "greater")
            or (higher and row.get("alternative") == "less")
        )
    relative = row.get("relative_confidence_set")
    if relative is not None and not _is_missing(relative):
        return relative.geometry not in ("empty", "unavailable")
    if not _is_missing(row.get("binomial_set")):
        return True
    lower = not _is_missing(row.get("lower"))
    higher = not _is_missing(row.get("higher"))
    return (
        (lower and higher)
        or (lower and row.get("open_side") == "upper")
        or (higher and row.get("open_side") == "lower")
    )


def _format_confidence_set(
    value: Any,
    *,
    relative: Any,
    binomial: Any,
    unavailable: Any,
    scale: str,
    lift: Any,
) -> str:
    """Render typed sets only where the point-dependent interval cannot represent them."""

    def number(endpoint: float | None, *, lower: bool) -> str:
        if endpoint is None:
            return "−∞" if lower else "+∞"
        return f"{endpoint:+.1%}" if scale == "relative" else f"{endpoint:+,.4g}"

    if not _is_missing(relative):
        geometry, intervals, reason = relative.geometry, relative.intervals, relative.reason
        if not _is_missing(lift) and geometry == "bounded":
            return ""
        if geometry in ("unavailable", "empty"):
            return f"{geometry}: {reason}" if reason else geometry
    elif not _is_missing(binomial):
        if not _is_missing(lift) and binomial.upper is not None:
            return ""
        geometry = binomial.geometry
        intervals = ((binomial.lower, binomial.upper),)
    elif not _is_missing(value):
        interval = value.relative if scale == "relative" else value.additive
        endpoints = (interval.lower, interval.upper)
        if any(endpoint.status == "undefined" for endpoint in endpoints):
            return "; ".join(
                f"{label}: undefined ({endpoint.reason})"
                if endpoint.status == "undefined"
                else f"{label}: {number(endpoint.value, lower=label == 'lower')}"
                for label, endpoint in zip(("lower", "upper"), endpoints, strict=True)
            )
        unbounded = any(endpoint.status == "unbounded" for endpoint in endpoints)
        if not _is_missing(lift) and not unbounded:
            return ""
        geometry = "unbounded" if unbounded else "bounded"
        intervals = ((interval.lower.value, interval.upper.value),)
    elif not _is_missing(unavailable):
        return f"unavailable: {unavailable}"
    else:
        return ""
    text = " ∪ ".join(
        f"{'(' if lower is None else '['}{number(lower, lower=True)}, "
        f"{number(upper, lower=False)}{')' if upper is None else ']'}"
        for lower, upper in intervals
    )
    rendered = text if geometry == "bounded" else f"{geometry.replace('_', '-')}: {text}"
    if not _is_missing(relative):
        level = f"{math.fsum((1.0, -relative.alpha)) * 100:.2f}".rstrip("0").rstrip(".")
        return f"{level}% {rendered}"
    return rendered


# Role-based group disclosure - concise section labels keyed by declared-plan
# role. The values are the text readers see as each section header.
_ROLE_GROUP_LABELS: dict[str | None, str] = {
    None: "Design-level",
    "primary": "Primary",
    "secondary": "Secondaries",
    "guardrail": "Guardrails",
    "unassigned": "Unassigned",
    "exploratory": "Exploratory",
}


def _group_disclosure_column(frame: Any) -> list[str] | None:
    """Return concise role labels for coeftable's ``groups=`` column.

    ``None`` is returned when the frame has no ``role`` column or every role
    is missing, preserving ungrouped rendering for tables without a plan.
    Mixed planned/unplanned rows use the ``Design-level`` section.
    """
    raw_roles = _column_values(frame, "role")
    if raw_roles is None:
        return None
    roles: list[str | None] = [None if _is_missing(role) else role for role in raw_roles]
    if not any(role is not None for role in roles):
        return None
    return [_ROLE_GROUP_LABELS.get(role, str(role)) for role in roles]


# Keep trend labels aligned with ``_resolve_trend_frame``.
def readout_table(  # noqa: C901, PLR0915
    data: IntoDataFrame | list[dict[str, Any]],
    *,
    title: str = "Experiment Readout",
    subtitle: str = "",
    theme: Theme | None = None,
    nest_by: Literal["arm", "method", "segment"] = "arm",
    advisory: bool = False,
    show_interval_level: bool = True,
    trend: IntoDataFrame | Sequence[DailyLiftEstimate | LiftEstimate] | None = None,
    trend_label: str = "Trend",
    trend_max_ylim: float | None = None,
) -> CoefTable:
    """Build a ``coeftable`` HTML readout table from lift-estimate rows.

    ``data`` needs ``metric``, ``method``, ``lift``, ``lower``,
    ``higher``, ``stat_sig``, ``group_id`` (plus ``segment`` for
    ``nest_by="segment"``), as a narwhals-wrappable frame or
    ``estimates_to_readout`` row dicts. Optional columns change the
    render: ``estimand`` labels a non-"itt" row so colliding
    estimand/metric pairs stay distinct; ``value_scale="absolute"`` rows
    render under "Lift (absolute)" instead of "Lift %" with a blank
    forest cell; ``preferred_direction`` colors a row by the metric's
    declared favorability. ``estimates_to_readout`` emits explicitly
    posterior-qualified values which are rendered only when
    ``advisory=True``; a shifted ``null_lift`` also draws a dashed forest
    reference line. Row caveats (``note``, ``excluded``,
    ``low_reliability``, ``inference``) ride along in the frame and
    render nowhere.

    For ``binomial_set`` rows, the table adds a ``Numerical qualification``
    column. Its values identify ``scipy_special_function_error_model_conditional_v1``
    (an enclosure conditional on the deployed SciPy/Boost error model) or
    ``legacy_unrecorded_v1`` (no arithmetic qualification claim in an older result).

    A ``role`` column with at least one non-``None`` value sections the
    table by concise role labels (coeftable's ``groups=`` +
    ``collapsible_groups=True``); see ``_group_disclosure_column``.
    A ``role`` column that is absent, or ``None`` on every
    row (no plan ever declared, for any row), renders ungrouped.

    ``nest_by`` picks which dimension stacks under each metric row vs.
    splits into side-by-side columns: ``"arm"`` (default) stacks
    arms/splits by method, ``"method"`` stacks methods/splits by arm,
    ``"segment"`` stacks segments/splits by arm (``data`` must already
    be filtered to one method, or this raises). ``advisory`` explicitly
    renders stored-posterior values with posterior-qualified labels;
    ``show_interval_level`` governs interval-level disclosure - an ``Interval`` column when
    levels differ across rows, or a note appended to ``subtitle`` when
    every row shares one non-95% level (a Bonferroni-corrected breakout
    reads "all intervals 98.33%" once instead of per row).

    ``trend`` adds a per-day sparkline: a native frame, or a raw
    estimate sequence, converted via ``to_frame`` and renamed to the
    main table's ``segment`` column. A metric absent from ``trend``
    renders a blank cell rather than raising; ``None`` (default) omits
    it.

    Raises ``ValueError`` if ``nest_by="segment"`` and ``data`` has more
    than one method; if visible row keys collide without distinct
    ``analysis_population`` values; if ``trend`` is missing a required key column or
    repeats a resolved readout identity at the same ``ds`` after scale/population
    label disambiguation; or if ``trend`` mixes incompatible ``ds_basis`` values.

    The sparkline's y-domain fit is hardcoded to ``"robust"`` (IQR-based)
    so a noisy early window in a cumulative/as-of series can't dominate
    a plain min/max fit and flatten the later trend."""
    import coeftable as ct
    import narwhals as nw
    import pandas as pd

    if isinstance(data, list):
        # pandas (bundled with the tables extra) avoids great_tables' "PyArrow
        # Table support is experimental" warning that pyarrow would trigger.
        frame = nw.from_native(pd.DataFrame(data), eager_only=True)
    else:
        frame = nw.from_native(data, eager_only=True)

    if nest_by == "segment" and frame["method"].n_unique() > 1:
        _raise("tables.readout_table_nest")

    if nest_by == "arm":
        nest_column, split_column = "group_id", "method"
    elif nest_by == "method":
        nest_column, split_column = "method", "group_id"
    else:
        nest_column, split_column = "segment", "group_id"
    split_columns = split_column if frame[split_column].n_unique() > 1 else None

    # The estimand joins the row label so itt/compliance/late rows sharing a
    # metric get unique coeftable keys; scale and population join only on a
    # visible collision. Check before building CoefTable, which would otherwise
    # crash or silently overwrite a population's reading.
    metrics = frame["metric"].to_list()
    estimands = (
        frame["estimand"].to_list() if "estimand" in frame.columns else ["itt"] * len(metrics)
    )
    scales = (
        frame["value_scale"].to_list()
        if "value_scale" in frame.columns
        else ["relative"] * len(metrics)
    )
    labels = [
        f"{metric} ({estimand})" if isinstance(estimand, str) and estimand != "itt" else metric
        for metric, estimand in zip(metrics, estimands, strict=True)
    ]
    nests = frame[nest_column].to_list()
    splits = frame[split_column].to_list() if split_columns else [None] * len(labels)
    keys = list(zip(labels, nests, splits, strict=True))
    collided = {key for key, count in Counter(keys).items() if count > 1}

    if "value_scale" in frame.columns:
        for key in collided:
            indexes = [index for index, candidate in enumerate(keys) if candidate == key]
            distinct_scales = {
                scale for scale in (scales[index] for index in indexes) if not _is_missing(scale)
            }
            if len(distinct_scales) > 1:
                for index in indexes:
                    labels[index] = f"{labels[index]} ({scales[index]})"

    # Assigned and triggered estimates usually collide on every visible axis.
    # Preserve their identity in the row label. A missing or repeated
    # population is ambiguous, so reject it rather than rendering a table
    # whose rows cannot be told apart.
    keys = list(zip(labels, nests, splits, strict=True))
    collided = {key for key, count in Counter(keys).items() if count > 1}
    populations = (
        frame["analysis_population"].to_list()
        if "analysis_population" in frame.columns
        else [None] * len(labels)
    )
    if collided:
        for key in collided:
            indexes = [index for index, candidate in enumerate(keys) if candidate == key]
            values = [populations[index] for index in indexes]
            distinct = {value for value in values if not _is_missing(value)}
            if len(distinct) != len(values) or len(distinct) < 2:
                _raise("tables.readout_table_ambiguous", key=key)
            for index in indexes:
                labels[index] = f"{labels[index]} ({populations[index]})"

    # Keep one identity-to-label mapping for both headline and trend rows. The
    # scale suffix is part of the identity even when population labels are not
    # needed, so absolute and relative series cannot collapse into one trend
    # group.
    identity_labels: dict[tuple[object, ...], str] = {
        (
            metric,
            estimand,
            "relative" if _is_missing(scale) else scale,
            nest,
            split,
            "assigned" if _is_missing(population) else population,
        ): label
        for metric, estimand, scale, nest, split, population, label in zip(
            metrics,
            estimands,
            scales,
            nests,
            splits,
            populations,
            labels,
            strict=True,
        )
    }

    final_keys = list(zip(labels, nests, splits, strict=True))
    if len(set(final_keys)) != len(final_keys):
        _raise("tables.readout_table_duplicate")
    frame = frame.with_columns(
        nw.new_series("metric", labels, backend=nw.get_native_namespace(frame))
    )

    unplottable_intervals = frozenset(
        index
        for index, (region, relative, scale) in enumerate(
            zip(
                frame["confidence_set"].to_list()
                if "confidence_set" in frame.columns
                else [None] * len(frame),
                frame["relative_confidence_set"].to_list()
                if "relative_confidence_set" in frame.columns
                else [None] * len(frame),
                scales,
                strict=True,
            )
        )
        if (
            relative is not None
            and not _is_missing(relative)
            and relative.geometry == "disconnected"
        )
        or (
            region is not None
            and not _is_missing(region)
            and (
                (region.additive if scale == "absolute" else region.relative).lower.status
                == "undefined"
                or (region.additive if scale == "absolute" else region.relative).upper.status
                == "undefined"
            )
        )
    )
    if unplottable_intervals:
        # CoefTable cannot draw undefined or disconnected intervals.
        # The typed set column retains their endpoints, geometry, and reasons.
        frame = frame.with_columns(
            *(
                nw.new_series(
                    column,
                    [
                        None if index in unplottable_intervals else value
                        for index, value in enumerate(frame[column].to_list())
                    ],
                    backend=nw.get_native_namespace(frame),
                )
                for column in ("lower", "higher")
            )
        )

    # An "absolute" value_scale row (e.g. a LATE) is in its own units, not a
    # percent lift, so route it to a separate column and blank the relative one.
    has_absolute_rows = any(scale == "absolute" for scale in scales)
    if has_absolute_rows:
        nan = float("nan")
        is_absolute = [scale == "absolute" for scale in scales]
        backend = nw.get_native_namespace(frame)
        routed = []
        for column in ("lift", "lower", "higher"):
            values = frame[column].to_list()
            routed.append(
                nw.new_series(
                    column,
                    [
                        nan if absolute else value
                        for value, absolute in zip(values, is_absolute, strict=True)
                    ],
                    backend=backend,
                )
            )
            routed.append(
                nw.new_series(
                    f"{column}_absolute",
                    [
                        value if absolute else nan
                        for value, absolute in zip(values, is_absolute, strict=True)
                    ],
                    backend=backend,
                )
            )
        frame = frame.with_columns(*routed)

    display_columns = _rendered_metadata_columns(
        frame,
        advisory=advisory,
        show_interval_level=show_interval_level,
    )
    if display_columns:
        display_backend = nw.get_native_namespace(frame)
        frame = frame.with_columns(
            *(
                nw.new_series(f"__display_{index}", values, backend=display_backend)
                for index, values in enumerate(display_columns.values())
            )
        )

    shifted_null_lift = "__shifted_null_lift"
    null_lifts = (
        frame["null_lift"].to_list() if "null_lift" in frame.columns else [None] * len(frame)
    )
    shifted_null_values: list[float | None] = []
    for value, row_scale in zip(null_lifts, scales, strict=True):
        if value is None or _is_missing(value) or row_scale == "absolute":
            shifted_null_values.append(None)
            continue
        null = float(value)
        shifted_null_values.append(null if math.isfinite(null) and null != 0.0 else None)
    frame = frame.with_columns(
        nw.new_series(
            shifted_null_lift,
            shifted_null_values,
            backend=nw.get_native_namespace(frame),
        )
    )

    group_column = _group_disclosure_column(frame)
    if group_column is not None:
        frame = frame.with_columns(
            nw.new_series(
                "__group_disclosure", group_column, backend=nw.get_native_namespace(frame)
            )
        )

    coeftable_kwargs: dict[str, Any] = {}
    if theme is not None:
        coeftable_kwargs["theme"] = theme
    if group_column is not None:
        coeftable_kwargs["groups"] = "__group_disclosure"
        coeftable_kwargs["collapsible_groups"] = True

    table = ct.CoefTable(
        frame.to_native(),
        rows="metric",
        nest=nest_column,
        split_columns=split_columns,
        **coeftable_kwargs,
    ).estimate(
        "Lift %",
        "lift",
        ci=("lower", "higher"),
        fmt=ct.Percent(scale=100.0, decimals=1),
    )
    if has_absolute_rows:
        table = table.estimate(
            "Lift (absolute)",
            "lift_absolute",
            ci=("lower_absolute", "higher_absolute"),
            fmt=ct.Number(decimals=2, signed=True),
        )
    forest_type = ct.Forest
    if unplottable_intervals:

        class DefinedIntervalForest(ct.Forest):
            def cell(self, ctx: Any) -> str:
                return "" if ctx.index in unplottable_intervals else super().cell(ctx)

        forest_type = DefinedIntervalForest
    table.columns = (
        *table.columns,
        forest_type(
            "Lift Plot",
            of="Lift %",
            ref=0.0,
            symmetric=True,
            annotations=(
                ct.Rule(
                    at=shifted_null_lift,
                    axis="x",
                    color="#E17C05",
                    width=2.0,
                    dash="dashed",
                ),
            ),
        ),
    )
    for index, label in enumerate(display_columns):
        table = table.passthrough(label, f"__display_{index}")

    # Colors rows by the metric's declared favorability (e.g. a latency drop
    # is a win); keys are the final row labels, so relabeled estimand rows resolve too.
    if "preferred_direction" in frame.columns:
        direction_for_declaration: dict[str, Direction] = {
            "increase": "higher_is_better",
            "decrease": "lower_is_better",
            "neutral": "neutral",
        }
        directions: dict[str, Direction] = {
            str(label): direction_for_declaration[declared]
            for label, declared in zip(
                frame["metric"].to_list(), frame["preferred_direction"].to_list(), strict=True
            )
            if isinstance(declared, str) and declared in direction_for_declaration
        }
        if directions:
            table = table.with_direction(directions)

    if trend is not None:
        trend_frame = _resolve_trend_frame(
            trend,
            nest_by=nest_by,
            nest_column=nest_column,
            split_column=split_column,
            split_columns=split_columns,
            identity_labels=identity_labels,
        )
        table = table.sparkline(
            trend_label,
            value="lift",
            ci=("lb", "ub"),
            x="ds",
            data=trend_frame.to_native(),
            ref=0.0,
            autoscale="robust",
            max_ylim=trend_max_ylim,
            axis_fmt=ct.DateAxis(),
        )

    level_note = _interval_level_note(
        frame,
        show_interval_level=show_interval_level,
        has_confidence_sets="Confidence set" in display_columns,
    )
    if level_note:
        subtitle = f"{subtitle} - {level_note}" if subtitle else level_note
    return table.header(title, subtitle)


def trend_table(
    data: IntoDataFrame | list[dict[str, Any]],
    *,
    title: str = "Metric Report",
    subtitle: str = "",
    trend_label: str = "Trend",
) -> CoefTable:
    """Render calendar metric trends (``MetricTrend.to_frame()`` output).

    The headline row per metric (nested by segment when a dimension
    column is present) shows the latest complete period's value and CI;
    the full period series - incomplete tail included - renders as a
    sparkline beside it. A frame with no complete period falls back to
    the latest period available.
    """
    import coeftable as ct
    import narwhals as nw
    import pandas as pd

    frame = (
        nw.from_native(pd.DataFrame(data), eager_only=True)
        if isinstance(data, list)
        else nw.from_native(data, eager_only=True)
    )
    required = {"metric", "period", "period_complete", "value"}
    missing = required - set(frame.columns)
    if missing:
        _raise("tables.trend_table_missing", missing=sorted(missing))
    fixed = {
        "metric",
        "grain",
        "window",
        "period",
        "period_complete",
        "n",
        "value",
        "ci_lb",
        "ci_ub",
    }
    dims = [c for c in frame.columns if c not in fixed]
    if len(dims) > 1:
        _raise("tables.trend_table_supports", dims=dims)

    # Nest axis: the dim when present, else a constant "overall" scope -
    # CoefTable is 2-axis (rows x nest) and always nests on something.
    if dims:
        nest_column = dims[0]
    else:
        nest_column = "scope"
        frame = frame.with_columns(nw.lit("overall").alias("scope"))

    # Prefer the latest complete period per (metric, nest), falling back to
    # the latest period overall; ascending sort + keep-last works since True sorts after False.
    headline = frame.sort(["period_complete", "period"]).unique(
        subset=["metric", nest_column], keep="last"
    )

    has_ci = "ci_lb" in frame.columns and not headline["ci_lb"].is_null().all()
    table = ct.CoefTable(
        headline.to_native(),
        rows="metric",
        nest=nest_column,
        split_columns=None,
    ).estimate(
        "Value",
        "value",
        ci=("ci_lb", "ci_ub") if has_ci else None,
    )
    table = table.sparkline(
        trend_label,
        value="value",
        ci=("ci_lb", "ci_ub") if has_ci else None,
        x="period",
        data=frame.to_native(),
        ref=0.0,  # Explicit: unlike a lift %, a raw metric value has no natural
        # zero reference, so this doesn't rely on coeftable's default.
        autoscale="robust",
        axis_fmt=ct.DateAxis(),
    )
    return table.header(title, subtitle)


__all__ = [
    "contrast_results_to_readout",
    "estimates_to_readout",
    "readout_table",
    "trend_table",
]
