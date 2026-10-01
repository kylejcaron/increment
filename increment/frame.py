"""Build a MomentSource straight from a dataframe - no YAML, no warehouse.

    import increment as inc

    src = inc.Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control",
        metrics={"revenue": "mean", "converted": "conversion"},
    )
    results = src.run()

Pure narwhals over pandas/polars/pyarrow, so this works with no backend
installed and can never import ibis or `increment/query/`. `from_unit_panel`
additionally supports windows, retention thresholds and late-enrollee
censoring.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame

from increment._frame_moments import (
    _ASOF_QUANTILE_UNSUPPORTED,
    _apply_spec_missing,
    _apply_winsorization,
    _asof_moment_rows,
    _collapse_to_unit_totals,
    _compliance_arm_rows_panel,
    _compliance_arm_rows_totals,
    _daily_moment_rows,
    _DeferredRefusal,
    _moment_rows,
    _total_moment_rows,
)
from increment._frame_panel import (
    _admit_panel_totals,
    _day_axis_label_order,
    _densify_panel,
    _fact_max_observed_dates,
    _prepare_panel,
    _scratch_name,
    _validate_panel_declarations,
)
from increment._frame_validation import (  # noqa: F401
    FRAME_UNIT_FRAME_PANEL,
    _canonicalize_group_identity,
    _coerce_metric_columns,
    _coerce_source_metrics,
    _enforce_missing_policy,
    _group_counts,
    _normalize_breakout_names,
    _normalized_breakout_expr,
    _reject_cluster_request,
    _reject_covariates,
    _reject_duplicate_unit_days,
    _reject_duplicate_units,
    _reject_windowed_specs,
    _resolve_design_and_plan,
    _resolve_null_exposure,
    _resolve_unassigned,
    _resolve_uptake_column,
    _unit_identity,
    _validate_cluster_labels,
    _validate_columns,
    _validate_control,
    _validate_conversion_binary,
    _validate_day_grain_columns,
    _validate_exposure_date,
    _validate_frame_boundary,
    _validate_single_group_per_unit,
    _validate_uptake_binary,
)
from increment._metric_specs import (
    MetricsArg,
    MetricSpec,
    coerce_metrics,
    synthesise_metric,
)
from increment._window import (
    maturity_days as _maturity_days,  # noqa: F401 - retained as the private parity alias
)
from increment._window import (
    resolve_window_days as _resolve_window_days,
)
from increment.decision import CompiledDecisionPlan
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    _safe_error_value,
    raiser,
    refusals,
    refuse,
)
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import (
    AnalysisPlan,
    Metric,
    QuantileMetric,
    RetentionMetric,
)
from increment.sequential_source import SequentialSourceMixin
from increment.sources import (
    UNASSIGNED_LABEL,
    ComplianceArm,
    ComplianceSummary,
    Grain,
    SourceContext,
    SourceOperation,
)

__all__ = [
    "FramePanelSource",
    "FrameSwitchbackSource",
    "FrameTotalsSource",
    "MetricSpec",
    "SwitchbackAssignmentDiagnostic",
    "coerce_metrics",
    "from_switchback_panel",
    "from_unit_panel",
    "from_unit_summary",
    "synthesise_metric",
]

# _metric_specs.py is a narwhals-free leaf module; these three stay branded
# as frame's own public names rather than the module that defines them.
MetricSpec.__module__ = __name__
coerce_metrics.__module__ = __name__
synthesise_metric.__module__ = __name__


_FRAME_GRAIN = RefusalSpec(
    "source.frame.grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_FRAME_QUANTILE_NO_MOMENTS = RefusalSpec(
    "source.frame.quantile_no_moments",
    CapabilityError,
    template="quantile metric {metric!r} has no moment representation; it is served through unit_frame. {route}",
)
_FRAME_BREAKOUTS_UNSUPPORTED = RefusalSpec(
    "source.frame.breakouts_unsupported",
    CapabilityError,
    template="total-frame moments cannot serve breakout dimensions {dimensions!r}. {route}",
)
_FRAME_SQL_UNSUPPORTED = RefusalSpec(
    "source.frame.sql_unsupported",
    CapabilityError,
    template="frame-backed sources have no SQL representation for grain {grain!r}. {route}",
)
_FRAME_MULTI_DIMENSION = RefusalSpec(
    "source.frame.multi_dimension",
    CapabilityError,
    template="a panel moments call requires at most one breakout dimension, got {dimensions!r}. {route}",
)
_FRAME_UNDECLARED_BREAKOUT = RefusalSpec(
    "source.frame.undeclared_breakout",
    CapabilityError,
    template="breakout dimension {dimension!r} was not declared; declared breakouts are {declared!r}. {route}",
)
_FRAME_WINDOWED_ENCOURAGEMENT_TOTAL = RefusalSpec(
    "source.frame.windowed_encouragement_total",
    CapabilityError,
    template="total moments for windowed metric {metric!r} under encouragement are unavailable from a unit panel. {route}",
)
_FRAME_RETENTION_DAILY = RefusalSpec(
    "source.frame.retention_daily",
    CapabilityError,
    template="retention metric {metric!r} has no independent per-day reading. {route}",
)
_FRAME_QUANTILE_DAILY = RefusalSpec(
    "source.frame.quantile_daily",
    CapabilityError,
    template="quantile metric {metric!r} does not decompose into per-day moments. {route}",
)
_FRAME_COMPLIANCE_ASOF = RefusalSpec(
    "source.frame.compliance_asof",
    CapabilityError,
    template="{operation} cannot serve as_of={as_of!r} from a totals-shape frame without a day axis. {route}",
)
_FRAME_COMPLIANCE_NO_UPTAKE = RefusalSpec(
    "source.frame.compliance_no_uptake",
    CapabilityError,
    template="{source} has no declared encouragement uptake column. {route}",
)
_FRAME_COMPLIANCE_DESIGN_MISMATCH = RefusalSpec(
    "source.frame.compliance_design_mismatch",
    InvalidRequestError,
    template="requested design {requested_design!r} differs from source design {source_design!r}. {route}",
)
_FRAME_CLUSTER = RefusalSpec(
    "source.frame.cluster_grain",
    CapabilityError,
    template="cluster_counts() is unavailable on {source}: {because}. The randomization-grain count needs a declared cluster column; build the source with from_unit_summary(..., cluster=...) or analyse from definitions declaring Experiment.cluster.",
)
_FRAME_INTERVENTION_GRAIN = RefusalSpec(
    "source.frame.intervention_grain_without_cluster",
    InvalidRequestError,
    template="intervention_grain={intervention_grain!r} declares a whole-cluster policy deployment, but no cluster= column was declared -- there is no cluster grain to intervene on.",
)
_FRAME_SEQUENTIAL_EXPOSURE = RefusalSpec(
    "source.frame.sequential_exposure_date",
    InvalidRequestError,
    template="{constructor}: sequential inference reveals units in exposure order, so it needs each unit's exposure date. {route}",
)
_PANEL_GRAIN = RefusalSpec(
    "source.frame_panel.grain",
    CapabilityError,
    lambda *, grain, offered: (
        f"this source cannot produce grain {grain!r}; it offers {set(offered)!r}."
        + (
            " Rebuild with Analysis.from_unit_panel(..., date=...) for a day axis."
            if grain in ("daily", "asof")
            else ""
        )
    ),
)
_PANEL_CLUSTER = RefusalSpec(
    "source.frame_panel.cluster_grain",
    CapabilityError,
    lambda *, source, because, operation="cluster_counts()", cluster=None: (
        f"{operation} is unavailable on {source}"
        + (f" (cluster={cluster!r})" if cluster is not None else "")
        + f": {because}. The randomization-grain count needs a declared cluster "
        "column; build the source with from_unit_summary(..., cluster=...) or "
        "analyse from definitions declaring Experiment.cluster."
    ),
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "frame.frame_totals.metric_was_declared": "metric {name!r} was not declared on this source",
        "frame.frame_totals.unit_covariate_column": "unit_frame: covariate column(s) {missing!r} not found in the retained frame; available columns: {columns}",
        "frame.frame_totals.unit_covariate_reserved": "unit_frame: {column!r} is reserved for declared cluster metadata; the declared cluster is {cluster!r}",
        "frame.frame_panel.unit_covariate_column": "unit_frame: covariate {missing!r} is not a column in this panel frame; available columns: {columns}",
        "frame.frame_panel.unit_covariate_varies": "unit_frame: covariate {covariate!r} varies within at least one unit's own rows ({units}) -- not a per-unit covariate; aggregate it to one value per unit first (e.g. take its pre-exposure value) before declaring it as an adjustment covariate.",
        "frame.frame_totals.moments_metric_undeclared": RefusalSpec(
            "frame.frame_totals.moments_metric_undeclared",
            InvalidRequestError,
            lambda *, names, declared: (
                f"moments rows reference undeclared metric(s) {sorted(names)!r}; "
                f"declared metrics: {sorted(declared)!r}"
            ),
        ),
        "frame.frame_totals.control_group_absent": RefusalSpec(
            "frame.frame_totals.control_group_absent",
            InvalidRequestError,
            lambda *, control, groups: (
                f"control group {control!r} does not appear among the moments rows' "
                f"group_id values {sorted(groups)!r}"
            ),
        ),
        "frame.frame_panel.asof_moments_completed": "asof moments: completed_windows_only=True is contradictory for retention metric {name!r} with unbounded band {band!r}",
    },
)
_raise = raiser(_REFUSALS)


# Null/NaN policy, applied only after every metric/denominator/covariate/uptake
# column is coerced to numeric (_coerce_metric_columns): a null group is not an arm.


class FrameTotalsSource(SequentialSourceMixin):
    """``MomentSource`` built from a one-row-per-unit frame.

    Returned by :func:`from_unit_summary` / :meth:`from_frame`. Only
    ``"total"`` grain: this shape carries no day axis.

    Construction snapshots moment rows, metric metadata and raw percentile
    outcomes. Other dataframe reads retain their existing backend view.
    """

    shape: Literal["unit_summary", "unit_panel"] | None = "unit_summary"
    capabilities: frozenset[Grain] = frozenset({"total"})
    operations: frozenset[SourceOperation] = frozenset({"export_moments"})
    breakouts: tuple[str, ...] = ()

    # Public constructor signature is the API for frame-backed sources.
    def __init__(  # noqa: PLR0913
        self,
        *,
        frame: nw.DataFrame[Any],
        unit: str,
        group: str,
        moments: list[dict[str, Any]],
        metrics: list[MetricSpec],
        control: str,
        experiment_id: str,
        design: Randomized | Encouragement | Observational,
        plan: CompiledDecisionPlan,
        uptake: str | None = None,
        unassigned: int = 0,
        cluster: str | None = None,
        intervention_grain: Literal["unit", "cluster"] = "unit",
        exposure_date: str | None = None,
        synthesised_metrics: Sequence[Metric] | None = None,
        segment_columns: Mapping[str, str] | None = None,
    ) -> None:
        if intervention_grain == "cluster" and cluster is None:
            refuse(_FRAME_INTERVENTION_GRAIN, intervention_grain=intervention_grain)
        registration = getattr(plan.inference, "registration", None)
        if registration is not None:
            self.breakouts = tuple(sorted({k for c in registration.roster for k, _ in c.segment}))
        if segment_columns is None:
            # Direct construction: read the registered segment labels from the
            # frame's own typed columns, exactly as ``from_frame`` does.
            segment_columns = {}
            _validate_columns(
                frame,
                roles=[(f"registered segment {name!r}", name) for name in self.breakouts],
                metrics=(),
            )
            for dimension in self.breakouts:
                scratch = _scratch_name(frame, "__segment__")
                frame = frame.with_columns(
                    _normalized_breakout_expr(frame, dimension).alias(scratch)
                )
                segment_columns[dimension] = scratch
        self._frame = frame
        self._segment_columns: Mapping[str, str] = MappingProxyType(dict(segment_columns))
        self._unit = unit
        self._unit_dtype = frame[unit].dtype
        self._exposure_date = exposure_date
        self._group = group
        self._moments: tuple[Mapping[str, Any], ...] = tuple(
            MappingProxyType(dict(row)) for row in moments
        )
        self._specs: tuple[MetricSpec, ...] = tuple(metrics)
        self._specs_by_name: Mapping[str, MetricSpec] = MappingProxyType(
            {s.name: s for s in self._specs}
        )
        self._control = control
        self._experiment_id = experiment_id
        self._uptake = uptake
        self._unassigned = unassigned
        self._cluster = cluster
        declared_names = set(self._specs_by_name)
        undeclared = {r["metric"] for r in self._moments if r["metric"] not in declared_names}
        if undeclared:
            _raise(
                "frame.frame_totals.moments_metric_undeclared",
                names=undeclared,
                declared=declared_names,
            )
        if self._moments and getattr(plan.inference, "registration", None) is None:
            groups = {str(r["group_id"]) for r in self._moments}
            if str(control) not in groups:
                _raise("frame.frame_totals.control_group_absent", control=control, groups=groups)
        metric_catalog = tuple(
            synthesised_metrics
            if synthesised_metrics is not None
            else [synthesise_metric(s) for s in metrics]
        )
        from increment._analysis_config import resolve_configs

        self._context = SourceContext(
            study_id=experiment_id,
            design=design,
            plan=plan,
            metrics=metric_catalog,
            configs=resolve_configs(
                metric_catalog,
                bindings=None,
                specs=self._specs_by_name,
                methods=None,
                prior=None,
            ),
            cluster=cluster,
            intervention_grain=intervention_grain,
        )
        # Snapshot exact values so later caller mutation cannot change the
        # inferred population while leaving the constructed moments fixed.
        snapshots = {}
        for metric in metric_catalog:
            if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
                raw = nw.from_native(self.unit_frame(metric, outcome_stage="raw"), eager_only=True)
                snapshots[metric.name] = MappingProxyType(
                    {column: tuple(raw[column].to_list()) for column in raw.columns}
                )
        self._raw_winsor_frames = MappingProxyType(snapshots)

        from increment._frame_sequential import capture_frame_totals

        capture_frame_totals(self)

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def design(self):
        """The identification design captured at construction."""
        return self._context.design

    @property
    def plan(self) -> CompiledDecisionPlan:
        """The compiled decision plan captured at construction."""
        return self._context.plan

    def __repr__(self) -> str:
        names = ", ".join(repr(s.name) for s in self._specs)
        groups = sorted({str(r["group_id"]) for r in self._moments})
        return f"<FrameTotalsSource metrics=[{names}] groups={groups} control={self._control!r}>"

    @property
    def raw_moments(self) -> list[dict[str, Any]]:
        """The per-(metric, group) centered moments, as row mappings.

        Introspection only - the ``MomentSource`` protocol's ``moments()``
        name is a grain-dispatched method (see below), not this property.
        """
        return [dict(row) for row in self._moments]

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        if grain != "total":
            refuse(_FRAME_GRAIN, grain=grain, offered=self.capabilities)
        if getattr(metric, "type", None) == "quantile":
            refuse(
                _FRAME_QUANTILE_NO_MOMENTS,
                metric=metric.name,
                route="use readouts.run, which routes quantiles automatically",
            )
        if by:
            refuse(
                _FRAME_BREAKOUTS_UNSUPPORTED,
                dimensions=tuple(by),
                route="request an unbroken total or use a panel source with declared breakouts",
            )
        return [dict(r) for r in self._moments if r["metric"] == metric.name]

    def _spec_for(self, metric: Metric) -> MetricSpec:
        spec = self._specs_by_name.get(metric.name)
        if spec is None:
            _raise("frame.frame_totals.metric_was_declared", name=metric.name)
        return spec

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> IntoDataFrame:
        """Per-unit rows for *metric*: ``unit_id``, ``group_id``, ``y``, plus
        any requested *covariates* retained on the source panel. A
        ratio metric (``spec.denominator`` set) additionally carries
        ``y_den``, the per-unit denominator - ``y`` alone is the
        NUMERATOR for a ratio metric, never the ratio itself. A caller
        that needs the ratio reads both columns and combines them itself;
        this extends the return shape rather than adding a separate
        accessor because every existing non-ratio caller reads columns by
        name, so an extra ratio-only column is invisible to them.

        This is the unit-grain view used by the IPTW/DML estimators; a
        moments-only source can never build it. *covariates* names arbitrary
        property columns from the original frame (e.g. propensity-model
        features) - NOT the CUPED ``covariate`` field on ``MetricSpec``, which
        is already folded into ``x`` in the moments.

        When this source declares ``cluster=``, an aligned ``cluster_id``
        column carries that identity at its ORIGINAL native dtype (never one
        of *covariates* and never a model feature) - absent entirely when no
        cluster is declared. Requesting the declared ``cluster_id`` reuses
        that metadata column; any other covariate with that name refuses.
        Not string-cast here deliberately: canonical
        identity collision detection (two distinct native values that render
        the same string, e.g. the int ``1`` and the str ``"1"``) needs the
        raw native values, which a cast in this layer would already have
        erased.

        The metric's declared null/NaN policy applies here exactly as it
        does to the moments (mirroring ``_apply_spec_missing``'s combined
        predicate over ``y``/``y_den``): ``missing="zero"`` fills both,
        ``missing="drop"`` excludes a row null in either - a quantile or
        observational readout served from this view must see the same
        values the moments path declared, never a raw null the entry
        chokepoint already ruled on.
        """
        spec = self._spec_for(metric)
        if outcome_stage not in ("raw", "transformed"):
            from increment.winsor import winsor_refuse

            winsor_refuse("invalid_state", "Unknown unit outcome stage.")
        snapshots = getattr(self, "_raw_winsor_frames", {})
        if metric.name in snapshots and (outcome_stage == "raw" or not covariates):
            if covariates:
                from increment.winsor import winsor_refuse

                winsor_refuse(
                    "design_unsupported",
                    "Raw winsor snapshots do not support adjustment covariates.",
                )
            out = nw.from_dict(
                {column: list(values) for column, values in snapshots[metric.name].items()},
                backend=self._frame.implementation,
            )
            if outcome_stage == "transformed":
                out, _ = _apply_winsorization(out, spec, key_cols=("group_id",))
                if "y_raw" in out.columns:
                    out = out.drop("y_raw")
            return out.to_native()
        if "cluster_id" in covariates and self._context.cluster != "cluster_id":
            _raise(
                "frame.frame_totals.unit_covariate_reserved",
                column="cluster_id",
                cluster=self._context.cluster,
            )
        missing = [c for c in covariates if c not in self._frame.columns]
        if missing:
            _raise(
                "frame.frame_totals.unit_covariate_column",
                missing=missing,
                columns=sorted(self._frame.columns),
            )
        unit_dtype = self._unit_dtype
        ambiguous_ids = isinstance(unit_dtype, (nw.Object, nw.Categorical, nw.Enum))
        unit_ids = nw.col(self._unit)
        if not ambiguous_ids and unit_dtype != nw.String:
            unit_ids = unit_ids.cast(nw.String)
        exprs = [
            unit_ids.alias("unit_id"),
            nw.col(self._group).alias("group_id"),
            nw.col(spec.y_column).cast(nw.Float64).alias("y"),
        ]
        if spec.denominator is not None:
            exprs.append(nw.col(spec.denominator).cast(nw.Float64).alias("y_den"))
        if self._context.cluster is not None:
            # Keep raw cluster values: `_unit_design`'s `canonical_id_strings` must
            # see int `1` and str `"1"` as distinct native identities, and a string
            # cast here would merge them before the collision check.
            exprs.append(nw.col(self._context.cluster).alias("cluster_id"))
        exprs += [nw.col(c) for c in dict.fromkeys(covariates) if c != "cluster_id"]
        out = self._frame.select(*exprs)
        value_cols = ["y"] + (["y_den"] if spec.denominator is not None else [])
        if spec.missing == "zero":
            out = out.with_columns(
                *(nw.col(c).fill_nan(None).fill_null(0.0).alias(c) for c in value_cols)
            )
        elif spec.missing == "drop":
            predicate = nw.col(value_cols[0]).is_null() | nw.col(value_cols[0]).is_nan()
            for c in value_cols[1:]:
                predicate = predicate | nw.col(c).is_null() | nw.col(c).is_nan()
            out = out.filter(~predicate)
        if ambiguous_ids:
            from increment._identity import canonical_id_strings

            identities = canonical_id_strings(out["unit_id"].to_numpy(), what="unit_ids")
            out = out.with_columns(
                nw.new_series("unit_id", identities, dtype=nw.String, backend=out.implementation)
            )
        if outcome_stage == "transformed":
            out, _ = _apply_winsorization(out, spec, key_cols=("group_id",))
            if "y_raw" in out.columns:
                out = out.drop("y_raw")
        return out.to_native()

    def _per_group(self, expr: nw.Expr) -> dict[str, int]:
        """*expr* aggregated over the retained frame, keyed by group label."""
        rows = self._frame.group_by(self._group).agg(expr.alias("__count__"))
        return {str(r[self._group]): int(r["__count__"]) for r in rows.iter_rows(named=True)}

    def unit_counts(self) -> dict[str, int]:
        """Per-group unit counts.

        Counted off the retained one-row-per-unit frame - the enrolled
        population, before any metric's ``missing="drop"`` is applied at
        moment construction. Reading a metric's moment ``n`` instead would
        report that metric's post-drop count and vary with declaration
        order; the enrolled count is what the sample-ratio check needs.
        The clustered branch counts the same retained rows as units, while
        :meth:`cluster_counts` serves the randomization grain there.

        Units excluded via ``on_unassigned="exclude"`` are surfaced under
        the ``UNASSIGNED_LABEL`` key - an accounting entry, not an arm:
        the SRM check reports it without a chi-square degree of freedom
        and no estimate is ever produced for it.
        """
        counts = self._per_group(nw.len())
        if self._unassigned:
            counts[UNASSIGNED_LABEL] = self._unassigned
        return counts

    def cluster_counts(self) -> dict[str, int]:
        """Per-group distinct cluster counts - the randomization grain.

        Counted off the retained frame, which is the enrolled population:
        cluster labels were validated non-null and single-arm at
        construction, so every unit contributes to exactly one arm's count.
        """
        if self._context.cluster is None:
            refuse(
                _FRAME_CLUSTER,
                source="this frame source",
                because="no cluster column was declared",
            )
        return self._per_group(nw.col(self._context.cluster).n_unique())

    def compliance_dates(self) -> Sequence[object]:
        refuse(
            _FRAME_COMPLIANCE_ASOF,
            operation="compliance_dates",
            as_of=None,
            route="build with from_unit_panel",
        )

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state off the retained enrolled
        frame -- independent of any metric's moments, declaration order,
        count, or missingness policy.
        """
        if completed_windows_only:
            from increment._source_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        if as_of is not None:
            refuse(
                _FRAME_COMPLIANCE_ASOF,
                operation="compliance_summary",
                as_of=_safe_error_value(as_of),
                route="build with from_unit_panel for an as-of cohort",
            )
        if self._uptake is None:
            refuse(
                _FRAME_COMPLIANCE_NO_UPTAKE,
                source=type(self).__name__,
                route="construct from_unit_summary with an Encouragement design or uptake=",
            )
        if design != self._context.design:
            refuse(
                _FRAME_COMPLIANCE_DESIGN_MISMATCH,
                source_design=self._context.design.model_dump(mode="json")
                if self._context.design is not None
                else None,
                requested_design=design.model_dump(mode="json")
                if isinstance(design, Encouragement)
                else _safe_error_value(design),
                route="rebuild the source for the requested design",
            )
        rows = _compliance_arm_rows_totals(
            self._frame, group=self._group, uptake=self._uptake, cluster=self._cluster
        )
        return ComplianceSummary(
            study_id=self._experiment_id,
            control_group=str(design.control_group),
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            cluster=self._cluster,
            as_of=None,
            arms=tuple(ComplianceArm(**row) for row in rows),
        )

    def export_moments(self, path: str | Path) -> None:
        from increment.sources import export_source_moments

        export_source_moments(self, path)

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        refuse(
            _FRAME_SQL_UNSUPPORTED, grain=grain, route="use a SQL-backed source for SQL rendering"
        )

    def close(self) -> None:
        pass

    @classmethod
    # Public constructor signature is the API for summary-backed sources.
    def from_frame(  # noqa: PLR0913
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        control: str,
        metrics: MetricsArg,
        experiment_id: str = "frame",
        uptake: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        on_unassigned: Literal["error", "exclude"] = "error",
        cluster: str | None = None,
        intervention_grain: Literal["unit", "cluster"] = "unit",
        exposure_date: str | None = None,
    ) -> FrameTotalsSource:
        """Build a source from a one-row-per-unit dataframe. See :func:`from_unit_summary`."""
        specs = _coerce_source_metrics(metrics, design=design)
        _reject_windowed_specs(specs)
        synthesised_metrics = [synthesise_metric(s) for s in specs]
        from increment._analysis_config import resolve_configs

        frame_configs = resolve_configs(
            synthesised_metrics,
            bindings=None,
            specs={s.name: s for s in specs},
            methods=None,
            prior=None,
        )
        from increment.sequential_source import (
            frame_observation_mapping,
            validate_frame_registration,
        )

        uptake, uptake_role = _resolve_uptake_column(design, uptake)
        design, compiled_plan = _resolve_design_and_plan(
            design,
            control,
            plan,
            synthesised_metrics,
            frame_configs,
            source_id=experiment_id,
            source_mapping=frame_observation_mapping(
                unit=unit, group=group, uptake=uptake, exposure_date=exposure_date
            ),
            transformations=specs,
        )
        registration = getattr(compiled_plan.inference, "registration", None)
        if registration is not None and exposure_date is None:
            refuse(
                _FRAME_SEQUENTIAL_EXPOSURE,
                constructor="from_unit_summary",
                route="pass exposure_date=<column> naming each unit's exposure date or day index",
            )

        validate_frame_registration(
            registration,
            metrics=synthesised_metrics,
            specs=specs,
            design=design,
            configs=frame_configs,
            source_id=experiment_id,
            cluster=cluster,
            source_mapping=frame_observation_mapping(
                unit=unit, group=group, uptake=uptake, exposure_date=exposure_date
            ),
        )
        if cluster is not None:
            _reject_cluster_request(specs, cluster=cluster, design=design, uptake=uptake)
        nwf = nw.from_native(frame, eager_only=True)

        segment_dimensions = (
            sorted({k for c in registration.roster for k, _ in c.segment})
            if registration is not None
            else []
        )
        roles = [("unit", unit), ("group", group)]
        if exposure_date is not None:
            roles.append(("exposure_date", exposure_date))
        roles.extend((f"registered segment {name!r}", name) for name in segment_dimensions)
        if cluster is not None:
            roles.append(("cluster", cluster))
        if uptake is not None:
            roles.append((uptake_role, uptake))
        _validate_columns(nwf, roles=roles, metrics=specs)
        # Labels are read from the original typed columns, before group or
        # metric coercion can rewrite a column that also serves another role.
        segment_columns: dict[str, str] = {}
        for dimension in segment_dimensions:
            scratch = _scratch_name(nwf, "__segment__")
            nwf = nwf.with_columns(_normalized_breakout_expr(nwf, dimension).alias(scratch))
            segment_columns[dimension] = scratch
        nwf = _canonicalize_group_identity(nwf, group)
        metric_columns = sorted(
            {c for s in specs for c in (s.y_column, s.denominator, s.covariate) if c}
        )
        if uptake is not None:
            metric_columns = sorted({*metric_columns, uptake})
        nwf = _coerce_metric_columns(nwf, metric_columns)
        _validate_frame_boundary(
            nwf, unit=unit, date=None, exposure_date=exposure_date, specs=specs
        )
        if exposure_date is not None:
            _validate_day_grain_columns(nwf, [("exposure_date", exposure_date)])
        _validate_exposure_date(nwf, unit=unit, exposure_date=exposure_date, specs=specs)
        nwf, unassigned = _resolve_unassigned(
            nwf,
            unit=unit,
            group=group,
            on_unassigned=on_unassigned,
            constructor="from_unit_summary",
        )
        null_exposure = 0
        if exposure_date is not None:
            nwf, null_exposure = _resolve_null_exposure(
                nwf,
                unit=unit,
                exposure_date=exposure_date,
                on_unassigned=on_unassigned,
                constructor="from_unit_summary",
            )
        _reject_duplicate_units(nwf, unit)
        if registration is None:
            _validate_control(nwf, group, control)
        if uptake is not None:
            _validate_uptake_binary(nwf, uptake)
        nwf = _enforce_missing_policy(nwf, specs, shape="summary")
        _validate_conversion_binary(nwf, specs)
        if cluster is not None:
            # A declared cluster-randomization (or encouragement) requires
            # arm purity; a declared observational design carries dependence
            # clusters that may legitimately span both treatment values.
            _validate_cluster_labels(
                nwf,
                cluster=cluster,
                group=group,
                enforce_purity=getattr(design, "mechanism", None) != "observational",
            )

        moments = (
            []
            if registration is not None
            else _moment_rows(
                nwf,
                group=group,
                metrics=specs,
                experiment_id=experiment_id,
                uptake=uptake,
                cluster=cluster,
            )
        )
        return cls(
            frame=nwf,
            unit=unit,
            group=group,
            moments=moments,
            metrics=specs,
            control=control,
            experiment_id=experiment_id,
            design=design,
            plan=compiled_plan,
            uptake=uptake,
            unassigned=unassigned + null_exposure,
            cluster=cluster,
            exposure_date=exposure_date,
            intervention_grain=intervention_grain,
            synthesised_metrics=synthesised_metrics,
            segment_columns=segment_columns,
        )


# Public factory signature is the API for summary-backed sources.
def from_unit_summary(  # noqa: PLR0913
    frame: IntoDataFrame,
    *,
    unit: str,
    group: str,
    control: str,
    metrics: MetricsArg,
    experiment_id: str = "frame",
    uptake: str | None = None,
    design: Randomized | Encouragement | Observational | None = None,
    plan: AnalysisPlan | None = None,
    on_unassigned: Literal["error", "exclude"] = "error",
    cluster: str | None = None,
    intervention_grain: Literal["unit", "cluster"] = "unit",
    exposure_date: str | None = None,
) -> FrameTotalsSource:
    """Analyse an experiment from a one-row-per-unit dataframe.

    Parameters
    ----------
    frame : IntoDataFrame
        A narwhals-supported eager frame with one row per unit.
    unit, group : str
        Columns identifying the unit and its variant label.
    control : str
        Which value of *group* is the control arm; never guessed.
    metrics : Mapping[str, str] | Sequence[MetricSpec | Mapping]
        The terse ``{"revenue": "mean"}`` form or explicit ``MetricSpec``
        objects for covariates and ratio metrics.
    experiment_id : str
        Label carried through to the estimates.
    uptake : str | None
        Binary uptake column for an ``Encouragement`` design; defaults to
        the design's ``UptakeSpec.fact``.
    design : Randomized | Encouragement | Observational | None
        Real construction state, stored on the returned source as
        ``src.design``: omitted, it derives ``Randomized(control_group=
        control)``; given explicitly, its ``control_group`` must agree
        with *control* (refused otherwise) - one control, declared once.
    plan : AnalysisPlan | None
        Resolved against the declared *metrics* (see
        ``increment.plan.compile_decision_plan``) into ``src.plan`` - the
        primary/secondary/guardrail roles and alpha allocation
        downstream estimation reads. Omitted, every metric resolves
        ``role="unassigned"`` and ``src.plan.declared`` is ``False``.
    on_unassigned : {"error", "exclude"}
        Null-*group* policy: refuse (default), or exclude with the count
        surfaced in ``unit_counts()``.
    cluster : str | None
        Randomization-grain column when coarser than *unit*: switches
        every metric to a cluster collapse and cluster-robust variance
        (points unchanged; rejected with CUPED, non-Encouragement
        uptake, quantile, or Encouragement-ratio metrics).
    intervention_grain : {"unit", "cluster"}
        Declares that a policy fit from this source intervenes at the
        whole-*cluster* grain, not the member/unit grain. *cluster* alone
        (a randomization/dependence grain) never implies ``"cluster"``
        here; ``"cluster"`` without a declared *cluster* refuses.
    exposure_date : str | None
        Each unit's exposure date or day index. Sequential inference
        requires it and reveals units in ascending exposure order, ties
        broken by unit id, never in row order. A null exposure follows
        *on_unassigned*.

    Returns
    -------
    FrameTotalsSource
        A ``MomentSource``; feed it to ``increment.readouts.run``.

    Raises
    ------
    ValueError
        Missing/duplicated columns, an absent *control*, group nulls under
        ``on_unassigned="error"``, a non-binary conversion/uptake column,
        or an explicit *design* whose ``control_group`` disagrees with
        *control*.
    """
    return FrameTotalsSource.from_frame(
        frame,
        unit=unit,
        group=group,
        control=control,
        metrics=metrics,
        experiment_id=experiment_id,
        uptake=uptake,
        design=design,
        plan=plan,
        on_unassigned=on_unassigned,
        cluster=cluster,
        intervention_grain=intervention_grain,
        exposure_date=exposure_date,
    )


# from_unit_panel: one row per unit per day.


# Windowing/censoring/retention mirror increment.query.builders exactly
# (tests/test_frame_window_parity.py); both share the canonical
# increment._window.resolve_window_days rather than each carrying its own copy.


class FramePanelSource(SequentialSourceMixin):
    """``MomentSource`` built from a one-row-per-unit-per-day frame.

    Returned by :func:`from_unit_panel` / :meth:`from_frame`. Construction
    retains the validated sparse observations and enrolled identity roster.
    A dense unit x day spine is materialized only when a day-grain consumer
    requests one.

    Offers whole-panel ``"total"``, independent-day ``"daily"``, and
    cumulative ``"asof"`` moments.

    Construction snapshots metric metadata, group counts, and fact dates.
    Dataframes remain zero-copy; returned moments and counts are defensive
    copies.
    """

    shape: Literal["unit_summary", "unit_panel"] | None = "unit_panel"
    capabilities: frozenset[Grain] = frozenset({"total", "daily", "asof"})
    operations: frozenset[SourceOperation] = frozenset({"export_moments"})
    #: `from_unit_panel` takes no `cluster=` argument, so this shape never
    #: carries a randomization-cluster column (see `cluster_counts` below).

    # Public constructor signature is the API for frame-backed sources.
    def __init__(  # noqa: PLR0913
        self,
        *,
        panel: nw.DataFrame[Any],
        metrics: list[MetricSpec],
        control: str,
        experiment_id: str,
        design: Randomized | Encouragement | Observational,
        plan: CompiledDecisionPlan,
        group_counts: dict[str, int],
        n_filled: int,
        breakouts: Sequence[str] = (),
        uptake: str | None = None,
        window_days: int | None = None,
        first_exposure: nw.DataFrame[Any] | None = None,
        exposure: nw.DataFrame[Any] | None = None,
        observation_end: dt.date | dt.datetime | str | int | float | None = None,
        fact_max_ds: Mapping[str, Any] | None = None,
        sequential_mapping: Mapping[str, object] | None = None,
        unassigned: int = 0,
        synthesised_metrics: Sequence[Metric] | None = None,
        source_frame: nw.DataFrame[Any] | None = None,
        unit_column: str | None = None,
    ) -> None:
        self._sparse_panel = panel
        self._dense_panel: nw.DataFrame[Any] | None = None
        self._identity: nw.DataFrame[Any] | None = None
        self._value_columns = tuple(
            sorted(
                {c for spec in metrics for c in (spec.y_column, spec.denominator) if c}
                | ({uptake} if uptake is not None else set())
            )
        )
        self._sequential_mapping = MappingProxyType(dict(sequential_mapping or {}))
        if getattr(plan.inference, "registration", None) is not None and not sequential_mapping:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "source.invalid", "registered panels require their original observation mapping"
            )
        self._specs: tuple[MetricSpec, ...] = tuple(metrics)
        self._specs_by_name: Mapping[str, MetricSpec] = MappingProxyType(
            {s.name: s for s in self._specs}
        )
        self._control = control
        self._experiment_id = experiment_id
        self._group_counts: Mapping[str, int] = MappingProxyType(dict(group_counts))
        self._n_filled = n_filled
        self._source_frame = source_frame
        self._unit_column = unit_column
        self.breakouts = tuple(breakouts)
        self._uptake = uptake
        self._window_days = window_days
        self._first_exposure = first_exposure
        self._exposure = exposure
        self._observation_end = observation_end
        self._fact_max_ds: Mapping[str, Any] = MappingProxyType(
            dict(fact_max_ds) if fact_max_ds else {}
        )
        self._unassigned = unassigned
        self._moment_rows_cache: dict[tuple[Grain, str | None, bool], list[dict[str, Any]]] = {}
        self._moment_failures_cache: dict[
            tuple[Grain, str | None, bool], dict[str, _DeferredRefusal]
        ] = {}
        metric_catalog = tuple(
            synthesised_metrics
            if synthesised_metrics is not None
            else [synthesise_metric(s) for s in metrics]
        )
        from increment._analysis_config import resolve_configs

        self._context = SourceContext(
            study_id=experiment_id,
            design=design,
            plan=plan,
            metrics=metric_catalog,
            configs=resolve_configs(
                metric_catalog,
                bindings=None,
                specs={s.name: s for s in metrics},
                methods=None,
                prior=None,
            ),
            cluster=None,
        )

    def _day_panel(self) -> nw.DataFrame[Any]:
        """Materialize the zero-filled day spine only for day consumers."""
        if self._dense_panel is None:
            identity = self._identity
            if identity is None:
                identity = self._sparse_panel.select("unit_id", "group_id", *self.breakouts).unique(
                    subset=["unit_id"]
                )
                self._identity = identity
            self._dense_panel = _densify_panel(
                self._sparse_panel,
                identity=identity,
                value_columns=self._value_columns,
            )
        return self._dense_panel

    def capture_sequential(self, *, as_of, finalized: bool, previous=None):
        from increment._frame_sequential import capture_frame_panel

        return capture_frame_panel(self, as_of=as_of, finalized=finalized, previous=previous)

    @property
    def context(self) -> SourceContext:
        return self._context

    @property
    def design(self):
        """The identification design captured at construction."""
        return self._context.design

    @property
    def plan(self) -> CompiledDecisionPlan:
        """The compiled decision plan captured at construction."""
        return self._context.plan

    def __repr__(self) -> str:
        names = ", ".join(repr(s.name) for s in self._specs)
        groups = sorted(self._group_counts)
        return (
            f"<FramePanelSource metrics=[{names}] groups={groups} "
            f"control={self._control!r} densified_cells={self._n_filled}>"
        )

    @property
    def densified_cells(self) -> int:
        """Count of implicit or observed-null cells requiring zero fill.

        Independent of whether the day spine has been materialized.
        """
        return self._n_filled

    def _breakout_key(self, by: Sequence[str]) -> str | None:
        names = tuple(by)
        if len(names) > 1:
            refuse(
                _FRAME_MULTI_DIMENSION,
                dimensions=names,
                route="request one declared breakout at a time",
            )
        if not names:
            return None
        name = names[0]
        if name not in self.breakouts:
            refuse(
                _FRAME_UNDECLARED_BREAKOUT,
                dimension=name,
                declared=self.breakouts,
                route="rebuild with the breakout declared",
            )
        return name

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        if grain not in self.capabilities:
            refuse(_PANEL_GRAIN, grain=grain, offered=self.capabilities)
        if (
            grain == "total"
            and self._uptake is not None
            and _resolve_window_days(metric) is not None
        ):
            refuse(
                _FRAME_WINDOWED_ENCOURAGEMENT_TOTAL,
                metric=metric.name,
                route="use grain='asof' or collapse to one row per unit and use from_unit_summary",
            )
        breakout = self._breakout_key(by)
        if grain == "daily" and isinstance(metric, RetentionMetric):
            refuse(_FRAME_RETENTION_DAILY, metric=metric.name, route="read the total grain instead")
        if grain == "daily" and isinstance(metric, QuantileMetric):
            refuse(_FRAME_QUANTILE_DAILY, metric=metric.name, route="read the total grain instead")
        if grain == "asof" and isinstance(metric, QuantileMetric):
            refuse(_ASOF_QUANTILE_UNSUPPORTED, metric=metric.name)
        if (
            grain == "asof"
            and completed_windows_only
            and isinstance(metric, RetentionMetric)
            and metric.band[1] is None
        ):
            _raise("frame.frame_panel.asof_moments_completed", band=metric.band, name=metric.name)
        # memoize per grain and breakout, so an M-metric readout does one pass,
        # while independent dimensions retain independent cached rows.
        cache_key = (grain, breakout, completed_windows_only if grain == "asof" else False)
        rows = self._moment_rows_cache.get(cache_key)
        if rows is None:
            synthesised = cast("Sequence[Metric]", self._context.metrics)
            reducer_by = (breakout,) if breakout is not None else ()
            failures: dict[str, _DeferredRefusal] = {}
            if grain == "daily":
                rows, failures = _daily_moment_rows(
                    self._day_panel(),
                    metrics=self._specs,
                    synthesised=synthesised,
                    experiment_id=self._experiment_id,
                    exposure=self._exposure,
                    by=reducer_by,
                )
            elif grain == "asof":
                rows, failures = _asof_moment_rows(
                    self._day_panel(),
                    self._specs,
                    synthesised=synthesised,
                    experiment_id=self._experiment_id,
                    uptake=self._uptake,
                    uptake_window_days=self._window_days,
                    first_exposure=self._first_exposure,
                    exposure=self._exposure,
                    completed_windows_only=completed_windows_only,
                    observation_end=self._observation_end,
                    fact_max_ds=self._fact_max_ds,
                    by=reducer_by,
                )
            else:
                covariate_names = {
                    spec.covariate
                    for spec in self._specs
                    if spec.covariate is not None
                    and spec.window_days is None
                    and spec.type != "retention"
                }
                covariates: dict[str, nw.DataFrame[Any]] = {}
                covariate_failures: dict[str, _DeferredRefusal] = {}
                for name in covariate_names:
                    try:
                        covariates[name] = self._resolve_panel_covariate(name)
                    except InvalidRequestError as exc:
                        if exc.code != "frame.frame_panel.unit_covariate_varies":
                            raise
                        covariate_failures[name] = (_REFUSALS[exc.code], exc.context)
                # A covariate that varies within a unit is this metric's own
                # hazard, not every sibling metric's: defer it the same way
                # a collapsed-percentile-bound refusal defers above, so a
                # metric with no covariate (or a sound one) still reads.
                usable_specs = [
                    spec for spec in self._specs if spec.covariate not in covariate_failures
                ]
                usable_synthesised = [
                    metric
                    for spec, metric in zip(self._specs, synthesised, strict=True)
                    if spec.covariate not in covariate_failures
                ]
                rows, total_failures = _total_moment_rows(
                    self._sparse_panel,
                    usable_specs,
                    synthesised=usable_synthesised,
                    experiment_id=self._experiment_id,
                    uptake=self._uptake,
                    uptake_window_days=self._window_days,
                    first_exposure=self._first_exposure,
                    exposure=self._exposure,
                    observation_end=self._observation_end,
                    fact_max_ds=self._fact_max_ds,
                    by=reducer_by,
                    covariates=covariates,
                )
                failures = {
                    spec.name: covariate_failures[spec.covariate]
                    for spec in self._specs
                    if spec.covariate in covariate_failures
                }
                failures.update(total_failures)
            self._moment_rows_cache[cache_key] = rows
            self._moment_failures_cache[cache_key] = failures
        # Data-dependent failures, such as a collapsed percentile bound, are cached
        # with the rows, so a later direct request still refuses after a sibling
        # used this batch. Caching only (spec, context) lets refuse() replay any
        # reducer refusal as a fresh exception without keeping a stale traceback.
        failure = self._moment_failures_cache.get(cache_key, {}).get(metric.name)
        if failure is not None:
            failure_spec, context = failure
            refuse(failure_spec, **context)
        return [dict(r) for r in rows if r["metric"] == metric.name]

    def _spec_for(self, metric: Metric) -> MetricSpec:
        spec = self._specs_by_name.get(metric.name)
        if spec is None:
            _raise("frame.frame_totals.metric_was_declared", name=metric.name)
        return spec

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> IntoDataFrame:
        """Per-unit rows for an unwindowed, non-retention *metric*
        (mean/ratio/conversion/quantile): the same ``_collapse_to_unit_totals``
        sum ``moments(grain="total")`` already uses, with the same declared
        ``missing``/winsorization policy applied afterward (mirroring
        ``_moment_rows``, which ``moments(grain="total")`` itself feeds
        the identical collapsed frame through -- this method and that
        grain must see the same per-unit ``y``/``y_den`` values), plus any
        requested *covariates* taken from the pre-densification panel --
        one value per unit, refusing by name if a covariate genuinely
        varies across a unit's own rows (including a unit null on some
        rows and set on others: that is not a stable per-unit value
        either). A retention metric, or a windowed metric of any type,
        stays refused by name: a retention row's per-unit value depends
        on evaluating its band against the full day axis, and a windowed
        metric has no per-unit window/censoring concept in this sum --
        collapse to one row per unit and use from_unit_summary.
        """
        if outcome_stage == "raw":
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "This source does not retain exact pre-winsor unit outcomes."
            )
        spec = self._spec_for(metric)
        # This collapse has no per-unit window or censoring, so windowed metrics
        # refuse; retention also needs its band over the full day axis.
        if spec.window_days is not None or spec.type == "retention":
            refuse(
                FRAME_UNIT_FRAME_PANEL,
                metric=metric.name,
                shape="a unit panel",
                route="collapse to one row per unit and use from_unit_summary",
            )
        admitted = _admit_panel_totals(
            self._sparse_panel,
            self._exposure,
            value_columns=[c for c in (spec.y_column, spec.denominator) if c],
        )
        collapsed = _collapse_to_unit_totals(admitted, [spec])
        has_den = spec.denominator is not None
        exprs = [
            nw.col("unit_id"),
            nw.col("group_id"),
            nw.col(spec.y_column).cast(nw.Float64).alias("y"),
        ]
        if has_den:
            exprs.append(nw.col(spec.denominator).cast(nw.Float64).alias("y_den"))
        out = collapsed.select(*exprs)
        out = _apply_spec_missing(out, spec, has_x=False, has_den=has_den)
        out, _winsor_metadata = _apply_winsorization(out, spec, key_cols=("group_id",))
        if "y_raw" in out.columns:
            out = out.drop("y_raw")
        for name in covariates:
            out = self._attach_panel_covariate(out, name)
        return out.to_native()

    def _resolve_panel_covariate(self, name: str) -> nw.DataFrame[Any]:
        """The panel's per-unit value for covariate column *name*: constant
        across a unit's own rows, refused by name when it genuinely varies.

        Shared by `unit_frame()`'s ad hoc covariate reads and by
        `moments(grain="total")`'s CUPED wiring, which joins this same
        per-unit frame onto the collapsed unit totals before computing
        centered moments -- one resolution, not two.
        """
        if (
            self._source_frame is None
            or self._unit_column is None
            or name not in self._source_frame.columns
        ):
            _raise(
                "frame.frame_panel.unit_covariate_column",
                missing=name,
                columns=sorted(self._source_frame.columns)
                if self._source_frame is not None
                else [],
            )
        unit_column = self._unit_column
        source = self._source_frame
        # Normalize floating NaNs so missing covariates have the same value
        # representation on every backend.
        if source.schema[name] in (nw.Float32, nw.Float64):
            source = source.with_columns(nw.col(name).fill_nan(None).alias(name))
        count_name = _scratch_name(source, "__n_values__")
        per_unit = source.group_by(nw.col(unit_column).alias("unit_id")).agg(
            nw.col(name).first().alias(name),
            nw.col(name).n_unique().alias(count_name),
        )
        # Null counts as a value: all-null units are constant, mixed units vary.
        varying = per_unit.filter(nw.col(count_name) > 1)
        if varying.shape[0] > 0:
            _raise(
                "frame.frame_panel.unit_covariate_varies",
                covariate=name,
                units=sorted(varying.get_column("unit_id").to_list())[:5],
            )
        return per_unit.select("unit_id", name)

    def _attach_panel_covariate(self, out: nw.DataFrame[Any], name: str) -> nw.DataFrame[Any]:
        return out.join(self._resolve_panel_covariate(name), on="unit_id", how="left")

    def unit_counts(self) -> dict[str, int]:
        """Enrolled per-group unit counts.

        Units excluded via ``on_unassigned="exclude"`` (a null group label
        or, when *exposure_date* was given, a null exposure date) are
        surfaced under the ``UNASSIGNED_LABEL`` key - an accounting
        entry, not an arm: the SRM check reports it without a chi-square
        degree of freedom and no estimate is ever produced for it.
        """
        counts = dict(self._group_counts)
        if self._unassigned:
            counts[UNASSIGNED_LABEL] = self._unassigned
        return counts

    def cluster_counts(self) -> dict[str, int]:
        refuse(
            _PANEL_CLUSTER,
            source="the panel shape",
            because=(
                "from_unit_panel takes no cluster= argument, so no randomization "
                "grain coarser than the unit was ever declared"
            ),
        )

    def compliance_dates(self) -> Sequence[object]:
        """Panel labels before outcome missingness or completion filters."""
        return self._sparse_panel.get_column("ds").unique().to_list()

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state off the unit x day panel --
        independent of any metric's own y-column, missingness policy, or
        per-day emission gate. *as_of*, when given, matches this source's
        own ``ds`` day-axis values and freezes the cumulative cohort
        through that day; ``None`` reads the declared window in full.
        ``from_unit_panel`` never carries a declared cluster, so the
        returned summary is always unclustered.
        """
        if self._uptake is None:
            refuse(
                _FRAME_COMPLIANCE_NO_UPTAKE,
                source=type(self).__name__,
                route="construct from_unit_panel with an Encouragement design or uptake=",
            )
        if design != self._context.design:
            refuse(
                _FRAME_COMPLIANCE_DESIGN_MISMATCH,
                source_design=self._context.design.model_dump(mode="json")
                if self._context.design is not None
                else None,
                requested_design=design.model_dump(mode="json")
                if isinstance(design, Encouragement)
                else _safe_error_value(design),
                route="rebuild the source for the requested design",
            )
        if completed_windows_only:
            from increment._source_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        rows = _compliance_arm_rows_panel(
            self._sparse_panel,
            uptake=self._uptake,
            window_days=self._window_days,
            first_exposure=self._first_exposure,
            as_of=as_of,
            completed_windows_only=completed_windows_only,
        )
        return ComplianceSummary(
            study_id=self._experiment_id,
            control_group=str(design.control_group),
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            cluster=None,
            as_of=as_of,
            arms=tuple(ComplianceArm(**row) for row in rows),
        )

    def export_moments(self, path: str | Path) -> None:
        from increment.sources import export_source_moments

        export_source_moments(self, path)

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        refuse(
            _FRAME_SQL_UNSUPPORTED, grain=grain, route="use a SQL-backed source for SQL rendering"
        )

    def close(self) -> None:
        pass

    @classmethod
    # Public factory signature is the API for panel-backed sources.
    def from_frame(  # noqa: PLR0913, PLR0915
        cls,
        frame: IntoDataFrame,
        *,
        unit: str,
        group: str,
        date: str,
        control: str,
        metrics: MetricsArg,
        experiment_id: str = "frame",
        uptake: str | None = None,
        design: Randomized | Encouragement | Observational | None = None,
        plan: AnalysisPlan | None = None,
        exposure_date: str | None = None,
        observation_end: dt.date | dt.datetime | str | int | float | None = None,
        breakouts: Sequence[str] = (),
        on_unassigned: Literal["error", "exclude"] = "error",
    ) -> FramePanelSource:
        """Build a source from a one-row-per-unit-per-day dataframe. See :func:`from_unit_panel`."""
        raw_metrics = metrics.values() if isinstance(metrics, Mapping) else metrics
        if isinstance(design, Encouragement):
            retention_names = []
            for item in raw_metrics:
                item_type = (
                    item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None)
                )
                if item_type == "retention":
                    item_name = (
                        item.get("name") if isinstance(item, Mapping) else getattr(item, "name", "")
                    )
                    retention_names.append(str(item_name))
            if retention_names:
                from increment.semantics.design import RETENTION_UNDER_ENCOURAGEMENT

                refuse(
                    RETENTION_UNDER_ENCOURAGEMENT,
                    names=retention_names,
                )
        specs = _coerce_source_metrics(metrics, design=design)
        _reject_covariates(specs)
        synthesised_metrics = [synthesise_metric(s) for s in specs]
        if isinstance(design, Encouragement):
            retention = [spec.name for spec in specs if spec.type == "retention"]
            if retention:
                from increment.semantics.design import RETENTION_UNDER_ENCOURAGEMENT

                refuse(RETENTION_UNDER_ENCOURAGEMENT, names=retention)
        from increment._analysis_config import resolve_configs

        frame_configs = resolve_configs(
            synthesised_metrics,
            bindings=None,
            specs={s.name: s for s in specs},
            methods=None,
            prior=None,
        )
        from increment.sequential_source import (
            frame_observation_mapping,
            validate_frame_registration,
        )

        uptake, uptake_role = _resolve_uptake_column(design, uptake)
        sequential_mapping = frame_observation_mapping(
            unit=unit, group=group, uptake=uptake, date=date, exposure_date=exposure_date
        )
        design, compiled_plan = _resolve_design_and_plan(
            design,
            control,
            plan,
            synthesised_metrics,
            frame_configs,
            source_id=experiment_id,
            source_mapping=sequential_mapping,
            transformations=specs,
        )
        registration = getattr(compiled_plan.inference, "registration", None)

        validate_frame_registration(
            registration,
            metrics=synthesised_metrics,
            specs=specs,
            design=design,
            configs=frame_configs,
            source_id=experiment_id,
            source_mapping=sequential_mapping,
        )

        if registration is not None:
            registered_dimensions = {k for cell in registration.roster for k, _ in cell.segment}
            breakouts = tuple(sorted(set(breakouts) | registered_dimensions))
        breakouts = _normalize_breakout_names(
            breakouts,
            unit=unit,
            group=group,
            date=date,
            uptake=uptake,
            exposure_date=exposure_date,
            specs=specs,
        )
        nwf = nw.from_native(frame, eager_only=True)
        roles = [("unit", unit), ("group", group), ("date", date)]
        if uptake is not None:
            roles.append((uptake_role, uptake))
        if exposure_date is not None:
            roles.append(("exposure_date", exposure_date))
        roles.extend((f"breakout {name!r}", name) for name in breakouts)
        _validate_columns(nwf, roles=roles, metrics=specs)
        declarations = [("metric value", spec.y_column) for spec in specs]
        declarations.extend(
            ("metric denominator", spec.denominator)
            for spec in specs
            if spec.denominator is not None
        )
        if uptake is not None:
            declarations.append(("uptake", uptake))
        _validate_panel_declarations(declarations)
        nwf = _canonicalize_group_identity(nwf, group)
        metric_columns = sorted(
            {c for s in specs for c in (s.y_column, s.denominator, s.covariate) if c}
        )
        if uptake is not None:
            metric_columns = sorted({*metric_columns, uptake})
        nwf = _coerce_metric_columns(nwf, metric_columns)
        _validate_frame_boundary(
            nwf, unit=unit, date=date, exposure_date=exposure_date, specs=specs
        )
        _validate_day_grain_columns(
            nwf,
            [("date", date)]
            + ([("exposure_date", exposure_date)] if exposure_date is not None else []),
        )
        nwf, unassigned = _resolve_unassigned(
            nwf,
            unit=unit,
            group=group,
            on_unassigned=on_unassigned,
            constructor="from_unit_panel",
        )
        if registration is None:
            _validate_control(nwf, group, control)
        _reject_duplicate_unit_days(nwf, unit, date)
        _validate_single_group_per_unit(nwf, unit, group)
        _validate_exposure_date(nwf, unit=unit, exposure_date=exposure_date, specs=specs)
        null_exposure = 0
        if exposure_date is not None:
            nwf, null_exposure = _resolve_null_exposure(
                nwf,
                unit=unit,
                exposure_date=exposure_date,
                on_unassigned=on_unassigned,
                constructor="from_unit_panel",
            )
            if null_exposure and registration is None:
                # The exclusion may have removed an arm entirely - the
                # control check must hold on what is actually analysed.
                _validate_control(nwf, group, control)
        if uptake is not None:
            _validate_uptake_binary(nwf, uptake)
        nwf = _enforce_missing_policy(nwf, specs, shape="panel")
        _validate_conversion_binary(nwf, specs)
        identity = _unit_identity(nwf, unit=unit, group=group, breakouts=breakouts)

        value_columns = sorted({c for s in specs for c in (s.y_column, s.denominator) if c})
        if uptake is not None:
            value_columns = sorted({*value_columns, uptake})

        # Only a windowed/retention spec's running-experiment fallback reads
        # this; computed from raw nwf, before densification erases which column had a value.
        # Windowed ratios need a raw freshness bound for both components;
        # densification/zero fill must not make either side appear observed.
        windowed_y_columns = {
            column
            for spec in specs
            if spec.window_days is not None or spec.type == "retention"
            for column in (spec.y_column, spec.denominator)
            if column is not None
        }
        fact_max_ds = (
            _fact_max_observed_dates(nwf, date=date, columns=windowed_y_columns)
            if windowed_y_columns
            else {}
        )

        exposure: nw.DataFrame[Any] | None = None
        if exposure_date is not None:
            exposure = nwf.group_by(nw.col(unit).alias("unit_id")).agg(
                nw.col(exposure_date).alias("__exposure__").min()
            )

        # Each unit's own earliest observed appearance, computed before
        # densification (else min() would return the same global date for every unit).
        if exposure is not None:
            first_exposure = exposure.select(
                "unit_id", nw.col("__exposure__").alias("__first_exposure__")
            )
        else:
            observed = nwf
            if uptake is not None and isinstance(nwf.schema[date], nw.String):
                order = _day_axis_label_order(nwf.get_column(date).to_list())
                labels = sorted(order, key=order.__getitem__)
                observed = nwf.with_columns(
                    nw.col(date)
                    .replace_strict(labels, list(range(len(labels))))
                    .alias("__day_order__")
                ).sort("__day_order__")
                first_day = nw.col(date).first()
            else:
                first_day = nw.col(date).min()
            first_exposure = observed.group_by(nw.col(unit).alias("unit_id")).agg(
                first_day.alias("__first_exposure__")
            )

        sparse, group_counts, n_filled = _prepare_panel(
            nwf,
            identity=identity,
            unit=unit,
            date=date,
            value_columns=value_columns,
        )

        window_days = design.uptake.window_days if isinstance(design, Encouragement) else None

        source = cls(
            panel=sparse,
            source_frame=nwf,
            unit_column=unit,
            sequential_mapping=sequential_mapping,
            metrics=specs,
            control=control,
            experiment_id=experiment_id,
            design=design,
            plan=compiled_plan,
            group_counts=group_counts,
            n_filled=n_filled,
            breakouts=breakouts,
            uptake=uptake,
            window_days=window_days,
            first_exposure=first_exposure,
            exposure=exposure,
            observation_end=observation_end,
            fact_max_ds=fact_max_ds,
            unassigned=unassigned + null_exposure,
            synthesised_metrics=synthesised_metrics,
        )
        source._identity = identity
        return source


# Public factory signature is the API for panel-backed sources.
def from_unit_panel(  # noqa: PLR0913
    frame: IntoDataFrame,
    *,
    unit: str,
    group: str,
    date: str,
    control: str,
    metrics: MetricsArg,
    experiment_id: str = "frame",
    uptake: str | None = None,
    design: Randomized | Encouragement | Observational | None = None,
    plan: AnalysisPlan | None = None,
    exposure_date: str | None = None,
    observation_end: dt.date | dt.datetime | str | int | float | None = None,
    on_unassigned: Literal["error", "exclude"] = "error",
    breakouts: Sequence[str] = (),
    cluster: str | None = None,
) -> FramePanelSource:
    """Analyse an experiment from a one-row-per-unit-per-day dataframe.

    Parameters
    ----------
    frame : IntoDataFrame
        Eager frame with at most one row per (*unit*, *date*).
    unit, group, control, cluster : str
        As :func:`from_unit_summary`.
    date : str
        Per-row date column, day-grain; caller owns day bucketing.
    metrics : Mapping[str, str] | Sequence[MetricSpec | Mapping]
        As :func:`from_unit_summary`. An unwindowed mean, ratio or conversion
        metric may declare a CUPED covariate that is constant within each
        unit; a covariate that varies within a unit refuses. Windowed and
        retention metrics refuse a panel covariate - use
        :func:`from_unit_summary` for those. Sequential CUPED is not
        available from this constructor.
    breakouts : Sequence[str]
        Unit-stable reporting columns; nulls normalize to ``"__null__"``.
    experiment_id : str
        Label carried through to the estimates.
    uptake : str | None
        Binary per-(unit, day) uptake column, collapsed via MAX per unit.
    design : Randomized | Encouragement | Observational | None
        Real construction state, stored on the returned source as
        ``src.design``: omitted, it derives ``Randomized(control_group=
        control)`` (also consulted to derive the uptake column/window);
        given explicitly, its ``control_group`` must agree with *control*
        (refused otherwise) - one control, declared once.
    plan : AnalysisPlan | None
        Resolved against the declared *metrics* (see
        ``increment.plan.compile_decision_plan``) into ``src.plan`` - the
        primary/secondary/guardrail roles and alpha allocation
        downstream estimation reads. Omitted, every metric resolves
        ``role="unassigned"`` and ``src.plan.declared`` is ``False``.
    exposure_date : str | None
        Each unit's day-0 date; required for a windowed/retention metric.
    observation_end : datetime.date | datetime.datetime | str | int | float | None
        Last observed date; bounds whole-unit censoring, else per-metric.
    on_unassigned : {"error", "exclude"}
        Null-*group*/exposure-date policy: refuse (default) or exclude.

    Returns
    -------
    FramePanelSource
        A ``MomentSource`` offering ``"total"``/``"daily"``/``"asof"`` grain.

    Raises
    ------
    ValueError
        Missing/duplicate columns, a timezone-aware date, disallowed
        nulls, a non-binary uptake column, or an explicit *design* whose
        ``control_group`` disagrees with *control*.
    InvalidRequestError
        A metric value, ratio denominator or uptake column is named ``unit_id``,
        ``group_id`` or ``ds`` (``frame.panel.declaration_canonical_collision``).
        Rename that value column. These names remain valid for the unit/group/date
        roles themselves, and ``ds``/``group_id`` remain valid for CUPED-only covariates.
    CapabilityError
        A retention metric read at ``grain="daily"``.
    """
    if cluster is not None:
        # Same shape problem as the CUPED covariate: a repeated per-day
        # cluster label isn't decidable from arithmetic alone as one label vs. many.
        refuse(
            _PANEL_CLUSTER,
            source="a per-day panel",
            because=(
                "the panel is collapsed to one row per unit before clustering "
                "could apply, and the collapse is ambiguous on this shape. "
                "Aggregate to one row per unit yourself and use "
                "from_unit_summary(cluster=...) instead"
            ),
            operation="from_unit_panel(cluster=...)",
            cluster=cluster,
        )
    return FramePanelSource.from_frame(
        frame,
        unit=unit,
        group=group,
        date=date,
        control=control,
        metrics=metrics,
        experiment_id=experiment_id,
        uptake=uptake,
        design=design,
        plan=plan,
        exposure_date=exposure_date,
        observation_end=observation_end,
        breakouts=breakouts,
        on_unassigned=on_unassigned,
    )


# Public switchback constructor lives in its own narrow module to keep the
# ordinary panel/summary source implementation independent of contrast code.
from increment.switchback import (  # noqa: E402
    FrameSwitchbackSource,
    SwitchbackAssignmentDiagnostic,
    from_switchback_panel,
)

FrameSwitchbackSource.__module__ = __name__
SwitchbackAssignmentDiagnostic.__module__ = __name__
from_switchback_panel.__module__ = __name__
