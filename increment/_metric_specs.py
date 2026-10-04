"""Terse metric declarations shared by `frame` and `switchback`.

Leaf module: narwhals-free, no `increment.frame`/`increment.switchback`
import. `increment.frame` re-exports every name so its public API is
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from increment._literals import PreferredDirection
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.semantics.models import (
    ConversionMetric,
    MeanMetric,
    Measure,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
    Winsorization,
    _reject_bool_days,
    _validate_retention_declaration,
)

__all__ = [
    "MetricSpec",
    "MetricSpecType",
    "MetricsArg",
    "coerce_metrics",
    "synthesise_metric",
]

# Placeholder identifiers; the estimation engine only reads m.name/m.type
# and never surfaces these in a repr or error message.
_PLACEHOLDER_FACT = "__frame__"
_PLACEHOLDER_ENTITY = "__frame__"
# A ratio's two sides are distinct frame columns, so the placeholder measures
# must be distinct too: identical measures always resolve to a ratio of one and
# are refused at the declaration boundary.
_PLACEHOLDER_NUMERATOR_FACT = "__frame_numerator__"
_PLACEHOLDER_DENOMINATOR_FACT = "__frame_denominator__"

MetricSpecType = Literal["mean", "conversion", "ratio", "retention", "quantile"]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "frame.metric.type_quantile_needs": "metric {name!r} has type='quantile' and needs quantile in (0, 1), got {quantile!r}",
        "frame.metric.cuped_does_apply": "metric {name!r}: CUPED does not apply to quantile metrics (no mean to adjust) -- drop covariate=",
        "frame.metric.sets_quantile_but": "metric {name!r} sets quantile= but type={type!r}",
        "frame.metric.winsorization_applies_type": "metric {name!r}: winsorization applies only to type='mean', got {type!r}",
        "frame.metric.type_retention_needs": "metric {name!r}: type='retention' needs threshold_days -- an int N for the unbounded band [N, inf), or [a, b] for the half-open observation band [a, b)",
        "frame.metric.sets_threshold_days": "metric {name!r} sets threshold_days but type={type!r}; the observation band belongs to type='retention'",
        "frame.metric.window_days_supported": "metric {name!r}: window_days is not supported for quantile metrics on the frame path",
        "frame.metric.type_ratio_but": "metric {name!r} has type='ratio' but is missing {missing}; a ratio metric needs both a numerator and a denominator column",
        "frame.metric.type_but_sets": "metric {name!r} has type={type!r} but sets numerator/denominator; those belong to type='ratio'",
        "frame.metric.missing_impute": "metric {metric!r}: missing='impute' would fill the outcome with its own mean, shrinking variance and biasing the estimate. Route: {route}",
        "frame.metric.sets_covariate_missing": "metric {name!r} sets covariate_missing={covariate_missing!r} but declares no covariate; the policy governs the CUPED covariate column only",
        "frame.metric.method_name_cuped": "metric {name!r}: Method(name='cuped') without variance_reduction='cuped' would label an unadjusted estimate as CUPED-adjusted; pass Method(name='cuped', variance_reduction='cuped') or rename.",
        "frame.metric.cuped_supported_metrics": "metric {name!r}: CUPED is not supported for {type} metrics -- a quantile is not a mean of per-unit values, so there is no per-unit residual for a covariate slope to act on. CUPED is supported for mean, conversion, retention and ratio metrics; a ratio adjusts its numerator and denominator against covariate= with a slope each.",
        "frame.metric.declares_cuped_method": "metric {name!r} declares a CUPED method but no covariate= column -- CUPED needs a pre-period covariate",
        "frame.metrics_empty_supply": "metrics is empty; supply at least one metric to estimate",
        "frame.metrics_entries_metricspec": "metrics entries must be MetricSpec or mapping, got {type_name}",
        "frame.duplicate_metric_name": "duplicate metric name {name!r}: moments are keyed by name, so two specs sharing one name silently interleave their rows and cross-pair arms. Give each metric a distinct name.",
        "frame.metric_type_retention": "metric {name!r}: type='retention' needs threshold_days -- use the explicit MetricSpec form: MetricSpec(name={name!r}, type='retention', threshold_days=...)",
        "frame.metric_type_quantile": "metric {name!r}: type='quantile' needs quantile -- use the explicit MetricSpec form: MetricSpec(name={name!r}, type='quantile', quantile=...)",
        "frame.metric_unknown_type": "metric {name!r}: unknown type {kind!r}; expected one of 'mean', 'conversion', 'ratio'",
    },
)
_raise = raiser(_REFUSALS)


class MetricSpec(CodedModel, BaseModel):
    """One metric to estimate from a dataframe column.

    Parameters
    ----------
    name : str
        Metric name, and the value column unless *value_column* is given.
    value_column : str | None
        Read metric values from this column instead of *name*.
    type : {"mean", "conversion", "ratio", "retention", "quantile"}
        Dispatches the variance model: "conversion" is a 0/1 column,
        "quantile" needs *quantile*, "retention" needs *threshold_days*.
    winsorization : Winsorization | None
        Optional lower/upper outcome bounds for a mean metric.
    covariate : str | None
        Pre-experiment covariate column for CUPED.
    numerator, denominator : str | None
        Ratio-metric columns; both required when ``type="ratio"``.
    window_days : int | None
        Analysis window in days since exposure; ``None`` uses full
        history. Rejected on ``type="quantile"``.
    threshold_days : int | tuple[int, int] | None
        Retention band: int N is [N, inf); (a, b) is [a, b) (half-open).
    quantile : float | None
        The quantile in (0, 1) to estimate; required for ``type="quantile"``.
    missing : {"error", "zero", "drop"}
        Null/NaN policy for the value column: refuse (default), count as
        0, or drop those rows from this metric's moments. ``"impute"`` is
        refused (``frame.metric.missing_impute``): an outcome is never
        mean-filled; use ``"zero"`` or ``"drop"``, or
        ``covariate_missing="impute"`` for a covariate.
    covariate_missing : {"impute", "zero", "error"}
        Null/NaN policy for *covariate*: impute the pooled mean (default,
        warns with the count), treat as 0, or refuse.
    decision_method : Method | None
        Decision estimator for this metric. Omitted uses IPTW for an
        observational design, otherwise unadjusted; ``run(decision_method=...)``
        overrides it without changing the declared sensitivities.
    sensitivity_methods : tuple[Method, ...]
        Additional estimators reported after the decision estimator.
    prior : Normal | None
        Informative prior for this metric's ``run()`` rows; omitted = inherit
        the call's ``prior=``.
    preferred_direction : {"increase", "decrease", "neutral"} | None
        The metric's declared favorable side; ``None`` (default) means
        undeclared, not "increase" -- ``synthesise_metric`` forwards it
        onto the synthesised ``Metric`` only when explicitly set, so an
        undeclared frame metric never silently reports a favorable
        direction (see ``LiftEstimate.preferred_direction``).

    Notes
    -----
    ``window_days``/``threshold_days`` are validated here, at construction, with
    the models layer's own refusals (``definition.retention.*``,
    ``definition.models.reject_bool``): a retention band that is empty or
    negative, or ``window_days`` on a retention spec, never constructs.
    ``from_unit_summary`` carries no dates, so it refuses both instead (see
    ``CAPABILITY_TABLE``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: MetricSpecType = "mean"
    value_column: str | None = None
    covariate: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    window_days: int | None = None
    threshold_days: int | tuple[int, int] | None = None
    quantile: float | None = None
    winsorization: Winsorization | None = None
    missing: Literal["error", "zero", "drop", "impute"] = "error"
    covariate_missing: Literal["impute", "zero", "error"] = "impute"
    decision_method: Method | None = None
    sensitivity_methods: tuple[Method, ...] = ()
    prior: Normal | None = None
    preferred_direction: PreferredDirection | None = None

    @model_validator(mode="after")
    def _check_quantile(self) -> MetricSpec:
        if self.type == "quantile":
            if self.quantile is None or not 0.0 < self.quantile < 1.0:
                _raise("frame.metric.type_quantile_needs", name=self.name, quantile=self.quantile)
            if self.covariate is not None:
                _raise("frame.metric.cuped_does_apply", name=self.name)
        elif self.quantile is not None:
            _raise("frame.metric.sets_quantile_but", name=self.name, type=self.type)
        return self

    @model_validator(mode="after")
    def _check_winsorization(self) -> MetricSpec:
        if self.winsorization is not None and self.type != "mean":
            _raise("frame.metric.winsorization_applies_type", name=self.name, type=self.type)
        return self

    @field_validator("window_days", "threshold_days", mode="before")
    @classmethod
    def _no_bool_days(cls, v: Any) -> Any:
        return _reject_bool_days(v)

    @model_validator(mode="after")
    def _check_windowing(self) -> MetricSpec:
        """Pairing of ``threshold_days`` with ``type="retention"``, then the
        retention declaration itself (band shape, ``window_days``) through the
        models layer's single validator."""
        if self.type == "retention":
            if self.threshold_days is None:
                _raise("frame.metric.type_retention_needs", name=self.name)
            _validate_retention_declaration(self.name, self.threshold_days, self.window_days)
        elif self.threshold_days is not None:
            _raise("frame.metric.sets_threshold_days", name=self.name, type=self.type)
        if self.type == "quantile" and self.window_days is not None:
            _raise("frame.metric.window_days_supported", name=self.name)
        return self

    @model_validator(mode="after")
    def _check_ratio_parts(self) -> MetricSpec:
        if self.type == "ratio":
            missing = [
                part
                for part, value in (
                    ("numerator", self.numerator),
                    ("denominator", self.denominator),
                )
                if value is None
            ]
            if missing:
                _raise("frame.metric.type_ratio_but", missing=", ".join(missing), name=self.name)
            # Equal names are valid on the from_moments path, where they label
            # precomputed moments (e.g. one fact's sum over its count). The column
            # validator rejects them where names are read as frame columns.
        elif self.numerator is not None or self.denominator is not None:
            _raise("frame.metric.type_but_sets", name=self.name, type=self.type)
        return self

    @model_validator(mode="after")
    def _refuse_outcome_impute(self) -> MetricSpec:
        if self.missing == "impute":
            _raise(
                "frame.metric.missing_impute",
                metric=self.name,
                route=(
                    "missing='zero' if null means no events, missing='drop' if not observed, "
                    "or covariate_missing='impute' for a covariate column"
                ),
            )
        return self

    @model_validator(mode="after")
    def _check_covariate_missing(self) -> MetricSpec:
        if self.covariate is None and self.covariate_missing != "impute":
            _raise(
                "frame.metric.sets_covariate_missing",
                covariate_missing=self.covariate_missing,
                name=self.name,
            )
        return self

    @model_validator(mode="after")
    def _check_methods(self) -> MetricSpec:
        """Validate declared decision and sensitivity methods.

        ``covariate`` only declares that the pre-period column is available.
        A CUPED-requesting method needs a covariate to adjust. A ratio metric
        adjusts both of its components against that one column, each with its
        own slope; a quantile has no per-unit residual to adjust at all.
        """
        methods = (
            () if self.decision_method is None else (self.decision_method,)
        ) + self.sensitivity_methods
        for method in methods:
            if method.name == "cuped" and method.variance_reduction != "cuped":
                _raise("frame.metric.method_name_cuped", name=self.name)
        wants_cuped = any(method.variance_reduction == "cuped" for method in methods)
        if wants_cuped and self.type == "quantile":
            _raise("frame.metric.cuped_supported_metrics", name=self.name, type=self.type)
        if wants_cuped and self.covariate is None:
            _raise("frame.metric.declares_cuped_method", name=self.name)
        return self

    @property
    def y_column(self) -> str:
        """The column supplying ``y`` (the numerator for ratio metrics)."""
        if self.type == "ratio":
            assert self.numerator is not None  # guaranteed by _check_ratio_parts
            return self.numerator
        return self.value_column or self.name

    @property
    def source_columns(self) -> list[str]:
        """Every input column this metric reads."""
        cols = [self.y_column]
        if self.covariate is not None:
            cols.append(self.covariate)
        if self.denominator is not None:
            cols.append(self.denominator)
        return cols


MetricsArg = Mapping[str, str] | Sequence["MetricSpec | Mapping[str, Any]"]


def coerce_metrics(metrics: MetricsArg) -> list[MetricSpec]:
    """Accept the terse ``{"revenue": "mean"}`` form or explicit MetricSpecs."""
    if not metrics:
        _raise("frame.metrics_empty_supply")

    if isinstance(metrics, Mapping):
        terse = cast("Mapping[str, str]", metrics)
        return [MetricSpec(name=n, type=_as_type(n, k)) for n, k in terse.items()]

    out: list[MetricSpec] = []
    for spec in metrics:
        if isinstance(spec, MetricSpec):
            out.append(spec)
        elif isinstance(spec, Mapping):
            out.append(MetricSpec(**spec))
        else:
            _raise("frame.metrics_entries_metricspec", type_name=type(spec).__name__)

    seen: set[str] = set()
    for spec in out:
        # Moments are keyed by name: a duplicate would silently interleave
        # rows and cross-pair arms downstream.
        if spec.name in seen:
            _raise("frame.duplicate_metric_name", name=spec.name)
        seen.add(spec.name)
    return out


def _as_type(name: str, kind: str) -> MetricSpecType:
    """Validate a terse metric type, with a pointed message for the ones
    that need extra fields the terse ``{"name": "type"}`` form cannot carry.
    """
    if kind in ("mean", "conversion", "ratio"):
        return kind  # type: ignore[return-value]
    if kind == "retention":
        _raise("frame.metric_type_retention", name=name)
    if kind == "quantile":
        _raise("frame.metric_type_quantile", name=name)
    _raise("frame.metric_unknown_type", kind=kind, name=name)


def synthesise_metric(spec: MetricSpec) -> Metric:
    """Build the placeholder Metric the estimation engine dispatches on.

    ``aggregation="sum"`` on the synthesised ``MeanMetric`` (and, for the
    same reason, on both the ``RatioMetric``'s numerator and denominator
    ``Measure``s below) matches this module's own "mean"/"ratio"
    semantics exactly: `_moment_rows`/`_melt_to_unit_totals`/`_reduce_spec`
    always build ``y``/``y_den`` as the per-unit SUM of the declared
    column (then averaged across units by the estimator) - never a
    per-day average or a raw event count. Frame-backed sources never
    consulted ``.aggregation`` before `increment.query.source
    .SqlPanelSource.from_table` started routing this same placeholder through
    `increment.query.builders.unit_totals`, which DOES dispatch on it; leaving
    the pydantic default (``"count"``) would silently compute a row count
    instead of a sum for every adopted-table mean or ratio metric.

    ``window_days``/``threshold_days`` are forwarded straight through onto
    the real metric: ``MetricSpec`` already refused an empty or negative
    retention band, ``window_days`` on a retention spec, and bool day counts
    at construction with the models layer's own refusals
    (``_validate_retention_declaration``), and ``RetentionMetric`` runs the
    same validator again.

    ``preferred_direction`` is forwarded only when *spec* explicitly set it
    to a non-``None`` value (``"preferred_direction" in spec.model_fields_set
    and spec.preferred_direction is not None``), never as a bare value --
    passing the field unconditionally would make the synthesised
    ``Metric``'s own ``model_fields_set`` track it even when the caller
    never declared one, defeating the explicitness check behind
    ``Metric.declared_preferred_direction`` that ``increment.readouts`` reads. An
    explicit ``preferred_direction=None`` is treated identically to an
    unset field (both mean "undeclared") rather than forwarded, since the
    synthesised ``Metric``'s own field has no ``None`` in its type union
    and would otherwise raise a validation error naming a type the caller
    never mentioned.
    """
    direction_kwargs: dict[str, Any] = (
        {"preferred_direction": spec.preferred_direction}
        if "preferred_direction" in spec.model_fields_set and spec.preferred_direction is not None
        else {}
    )
    if spec.type == "mean":
        return MeanMetric(
            name=spec.name,
            entity=_PLACEHOLDER_ENTITY,
            fact=_PLACEHOLDER_FACT,
            aggregation="sum",
            window_days=spec.window_days,
            winsorization=spec.winsorization,
            **direction_kwargs,
        )
    if spec.type == "conversion":
        return ConversionMetric(
            name=spec.name,
            entity=_PLACEHOLDER_ENTITY,
            fact=_PLACEHOLDER_FACT,
            window_days=spec.window_days,
            **direction_kwargs,
        )
    if spec.type == "retention":
        assert spec.threshold_days is not None  # guaranteed by _check_windowing
        return RetentionMetric(
            name=spec.name,
            entity=_PLACEHOLDER_ENTITY,
            fact=_PLACEHOLDER_FACT,
            threshold_days=spec.threshold_days,
            window_days=spec.window_days,
            **direction_kwargs,
        )
    if spec.type == "quantile":
        assert spec.quantile is not None  # guaranteed by _check_quantile
        return QuantileMetric(
            name=spec.name,
            entity=_PLACEHOLDER_ENTITY,
            fact=_PLACEHOLDER_FACT,
            aggregation="sum",
            quantile=spec.quantile,
            **direction_kwargs,
        )
    # Ratio parts share the numerator's window_days, as in builders.unit_totals.
    # Both parts declare aggregation="sum": frame ratio columns already hold
    # per-unit-day values (`_frame_moments._reduce_spec` sums them); the
    # `Measure` default "count" would count rows instead of summing them.
    return RatioMetric(
        name=spec.name,
        entity=_PLACEHOLDER_ENTITY,
        numerator=Measure(
            fact=_PLACEHOLDER_NUMERATOR_FACT, aggregation="sum", window_days=spec.window_days
        ),
        denominator=Measure(
            fact=_PLACEHOLDER_DENOMINATOR_FACT, aggregation="sum", window_days=spec.window_days
        ),
        **direction_kwargs,
    )
