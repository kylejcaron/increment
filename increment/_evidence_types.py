from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import date, datetime
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypeAlias, cast

from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW
from increment.errors import (
    CapabilityError,
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.semantics.models import Breakout, Metric, RetentionMetric

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._source_types import EvidenceProvider, _BuiltinMomentSource
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Encouragement, Observational, Randomized


EvidenceSource: TypeAlias = "EvidenceProvider | _BuiltinMomentSource"  # noqa: UP040


BreakoutRoute = Literal["artifact", "native", "panel", "unsupported"]
DayAxisRoute = Literal["artifact", "moments", "native"]


@dataclass(frozen=True, slots=True)
class _WarehouseCupedFitEvidence:
    fit: Any
    arm_order: tuple[str, ...]
    withheld: tuple[tuple[str, CodedError], ...] = ()


def warehouse_cuped_fit_evidence(
    fit: Any, arm_order: tuple[str, ...], withheld: tuple[tuple[str, CodedError], ...] = ()
) -> _WarehouseCupedFitEvidence:
    """Build the single warehouse-fit envelope used by live and fitted sources."""
    return _WarehouseCupedFitEvidence(fit, tuple(arm_order), tuple(withheld))


@dataclass(frozen=True, slots=True)
class BreakoutIdentity:
    """The declared identity of a breakout dimension and its source."""

    property: str
    source: str | None


@dataclass(frozen=True, slots=True)
class BreakoutCell:
    """One requested cell within a declared breakout identity."""

    breakout: BreakoutIdentity
    value: object
    is_null: bool


@dataclass(frozen=True, slots=True)
class MethodFitRequest:
    """Request method-specific CUPED evidence for a metric or breakout cell."""

    metric: Metric
    method: Any
    control: str
    arm_order: tuple[str, ...]
    population: Literal["assigned", "triggered"]
    cell: BreakoutCell | None = None


@dataclass(frozen=True, slots=True)
class MethodFitAnswer:
    """Retained or warehouse method-fit evidence, or its coded acquisition failure."""

    origin: Literal["warehouse", "retained_fitted", "retained_categorical", "unavailable"]
    envelope: Any | None = None
    cells: Any | None = None
    failure: CodedError | None = None
    fit_identity: tuple[object, ...] = ()
    evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    manifest_sha256: str | None = None
    relation_digests: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedMulticovariateFit:
    """Selected retained, warehouse, or local input for a mixed-CUPED fit."""

    origin: Literal["warehouse", "retained_fitted", "retained_categorical", "local"]
    envelope: Any | None = None
    cells: Any | None = None
    failure: CodedError | None = None
    fit_identity: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakoutSlicesRequest:
    """Acquire source views for a batch of declared breakout identities."""

    breakouts: tuple[BreakoutIdentity, ...]
    breakout_definitions: tuple[Breakout, ...] = ()
    metrics: tuple[Metric, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"
    configs: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakoutSlice:
    identity: BreakoutIdentity
    source: Any
    source_name: str | None
    stamp_source: str | None


@dataclass(frozen=True, slots=True)
class BreakoutSlices:
    route: BreakoutRoute
    slices: tuple[BreakoutSlice, ...]


@dataclass(frozen=True, slots=True)
class BreakoutUnitRowsRequest:
    """Acquire unit-frame evidence for one breakout/metric pair."""

    metric: Metric
    breakout: BreakoutIdentity
    population: Literal["assigned", "triggered"] = "assigned"
    covariates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakoutUnitRows:
    identity: BreakoutIdentity
    route: BreakoutRoute
    frame: Any
    unit_frame_has_dimension: bool


@dataclass(frozen=True, slots=True)
class DayAxisRequest:
    caller: Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"]
    grain: Literal["daily", "asof"]
    metrics: tuple[Metric, ...]
    dimension: str | None
    completed_windows_only: bool = False
    estimands: tuple[str, ...] | None = None
    population: Literal["assigned", "triggered"] = "assigned"
    experiment: Any | None = None
    covariate_metrics: frozenset[str] = frozenset()
    by: tuple[str, ...] = ()


def day_axis_time_basis(metric: object, grain: str) -> Literal["calendar", "cohort"]:
    """Return the date coordinate used by a day-axis metric."""
    is_retention = isinstance(metric, RetentionMetric) or (
        isinstance(metric, Mapping) and metric.get("type") == "retention"
    )
    return "cohort" if grain == "daily" and is_retention else "calendar"


def _experiment_breakouts(experiment: object) -> tuple[object, ...]:
    if isinstance(experiment, Mapping):
        values = experiment.get("breakouts", ())
    else:
        values = getattr(experiment, "breakouts", ())
    return tuple(values) if isinstance(values, Sequence) else ()


def _breakout_identity(value: object) -> BreakoutIdentity | None:
    if isinstance(value, Mapping):
        property_name = value.get("property")
        source = value.get("source")
    else:
        property_name = getattr(value, "property", None)
        source = getattr(value, "source", None)
    if isinstance(property_name, str) and (source is None or isinstance(source, str)):
        return BreakoutIdentity(property_name, source)
    return None


def day_axis_source_partitions(
    experiment: object | None, dimensions: Sequence[str]
) -> tuple[tuple[str | None, tuple[BreakoutIdentity, ...], tuple[object, ...]], ...]:
    """Partition requested day-axis dimensions and declarations by source."""
    requested = tuple(dict.fromkeys(dimensions))
    declared = _experiment_breakouts(experiment) if experiment is not None else ()
    declared_pairs = tuple(
        (identity, definition)
        for definition in declared
        if (identity := _breakout_identity(definition)) is not None
    )
    selected = tuple(
        (identity, definition)
        for identity, definition in declared_pairs
        if identity.property in requested
    )
    declared_properties = {identity.property for identity, _ in selected}
    undeclared = tuple(
        BreakoutIdentity(property_name, None)
        for property_name in requested
        if property_name not in declared_properties
    )
    sources = tuple(dict.fromkeys(identity.source for identity, _ in selected)) or (None,)
    return tuple(
        (
            source,
            tuple(identity for identity, _ in selected if identity.source == source) + undeclared,
            tuple(
                definition
                for identity, definition in declared_pairs
                if identity.property not in requested or identity.source == source
            ),
        )
        for source in sources
    )


@dataclass(frozen=True, slots=True)
class DayAxisRows:
    """Acquired daily/as-of row slices and the source view used to read them."""

    source: Any
    route: DayAxisRoute
    slices: Iterator[Any]


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    """The source views and readout lifetime required by one evidence consumer."""

    metrics: tuple[Metric, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"
    grain: Literal["total", "daily", "asof"] | None = None
    dimension: str | None = None
    estimands: tuple[str, ...] | None = None
    breakouts: tuple[BreakoutIdentity, ...] = ()
    configs: tuple[Any, ...] | None = None
    uptake_facts: tuple[str, ...] = ()
    snapshot: bool = False
    execution: bool = False
    include_breakouts: bool = False
    checkpoint_only: bool = False


@dataclass(frozen=True, slots=True)
class ComplianceRequest:
    design: Encouragement
    as_of: object | None = None
    completed_windows_only: bool = False
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class ComplianceSeriesRequest:
    design: Encouragement
    completed_windows_only: bool = False
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class ContrastStatsRequest:
    """Request the pinned reduced switchback statistics for one metric."""

    metric: Metric


@dataclass(frozen=True, slots=True)
class MetricEvidenceRequest:
    metric: Metric
    control_group: str
    by: tuple[str, ...] = ()
    methods: tuple[Any, ...] = ()
    covariates: tuple[Any, ...] = ()
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One metric's aggregate rows and/or per-unit estimator evidence."""

    rows: tuple[Mapping[str, Any], ...]
    unit_frame: Any | None
    observed_arms: frozenset[str]
    n_treatment_arms: int
    evidence_kind: str
    evidence_sha256: str
    evidence_count: int
    decision_evidence_components: tuple[tuple[str, str, int], ...]
    cluster: str | None
    control_group: str
    evidence_components: tuple[tuple[str, str, int], ...] = ()
    covariate_columns: tuple[tuple[str, str], ...] = ()
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()
    method_refusals: tuple[tuple[str, RefusalSpec, Mapping[str, object]], ...] = ()
    method_cell_refusals: tuple[tuple[str, str, str, RefusalSpec, Mapping[str, object]], ...] = ()
    method_fit_answers: tuple[tuple[MethodFitRequest, MethodFitAnswer], ...] = ()

    def rows_for_method(
        self,
        method: Any,
        *,
        dimension: str | None = None,
        dimension_value: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return the produced input family, replaying only its own cell refusal."""
        from increment.errors import refuse

        for name, spec, context in self.method_refusals:
            if name == method.name:
                refuse(spec, **context)
        for name, cell_dimension, value, spec, context in self.method_cell_refusals:
            if name != method.name:
                continue
            if dimension is None:
                raise AssertionError("Partitioned scalar evidence requires a segment.")
            if cell_dimension == dimension and value == dimension_value:
                refuse(spec, **context)
        for name, rows in self.method_rows:
            if name == method.name:
                if dimension is None:
                    return rows
                return tuple(
                    row
                    for row in rows
                    if ("__null__" if row.get(dimension) is None else str(row[dimension]))
                    == dimension_value
                )
        raise AssertionError(f"Scalar evidence for method {method.name!r} was not acquired.")


@dataclass(frozen=True, slots=True)
class MomentsRequest:
    """Select aggregate metric evidence at one time grain."""

    metric: Metric
    grain: Literal["total", "daily", "asof"] = "total"
    by: tuple[str, ...] = ()
    method: Any | None = None
    include_covariate: bool = False
    completed_windows_only: bool = False
    experiment: Any | None = None
    population: Literal["assigned", "triggered"] = "assigned"


MomentRows = tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BreakoutMethodMomentsRequest:
    """Select method-specific moment rows and refusals for one breakout."""

    metric: Metric
    method: Any
    breakout: BreakoutIdentity
    population: Literal["assigned", "triggered"] = "assigned"
    completed_windows_only: bool = False


@dataclass(frozen=True, slots=True)
class ScalarBreakoutRows:
    """Method-specific moments and scalar refusals indexed by segment."""

    rows: tuple[Mapping[str, Any], ...]
    refusals: tuple[tuple[str, str, RefusalSpec, Mapping[str, object]], ...] = ()


@dataclass(frozen=True, slots=True)
class UnitRowsRequest:
    """Select per-unit observations needed for an estimator."""

    metric: Metric
    covariates: tuple[str, ...] = ()
    outcome_stage: Literal["transformed", "raw"] = "transformed"
    population: Literal["assigned", "triggered"] = "assigned"


UnitRows = Any


@dataclass(frozen=True, slots=True)
class CountsRequest:
    """Select enrolled-unit, cluster, assignment, or triggered counts."""

    kind: Literal["unit", "cluster", "assignment", "triggered"]
    population: Literal["assigned", "triggered"] = "assigned"


@dataclass(frozen=True, slots=True)
class CountEvidence:
    """Counts for the requested population and randomization grain."""

    population: str
    grain: str
    counts: Mapping[str, int]
    unit_counts: Mapping[str, int] = dataclass_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequentialSnapshotRequest:
    """Request a provider-owned current checkpoint linked to an optional parent."""

    previous: Any | None = None


@dataclass(frozen=True, slots=True)
class SequentialCaptureRequest:
    """Capture a registered finalized prefix of sequential data."""

    finalized: bool
    as_of: object | None = None
    previous: Any | None = None
    freeze: tuple[str, ...] = ()


SequentialCapture = Any


@dataclass(frozen=True, slots=True)
class EvidenceStream:
    """One logical evidence stream and the physical stream pinned to answer it."""

    identity: str
    physical: str


CaptureStatus = Literal["captured", "live"]


@dataclass(frozen=True, slots=True)
class EvidenceScopeResult:
    """A source view, physical stream inventory, and effective capture status."""

    source: Any
    scope: EvidenceScope
    streams: tuple[EvidenceStream, ...] = ()
    stream_resolver: Callable[[Any], tuple[EvidenceStream, ...]] | None = None
    capture_status: CaptureStatus | None = None
    breakout_streams: Mapping[BreakoutIdentity, str] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )


# Columns a unit frame serves in its own right; a covariate declared under one
# of these names is carried under a scratch column instead (see
# `unit_frame_covariate_columns`). The declared cluster column keeps its own
# reservation: `cluster_id` is metadata, never a covariate.
UNIT_FRAME_COLUMNS = frozenset({"unit_id", "group_id", "y", "y_den"})


def unit_frame_covariate_columns(covariates: Sequence[str]) -> dict[str, str]:
    """The unit-frame column each requested covariate is carried under.

    The identity for every name that does not collide with a frame-served
    column, so producers that emit covariates under their declared names are
    unchanged. A colliding name (a covariate called ``y``) is carried under a
    scratch column derived from the same request, so a producer and its
    consumer agree on it without a side channel and the frame never holds
    two columns of one name or silently resolves a covariate to the outcome.
    """
    names = tuple(covariates)
    columns: dict[str, str] = {}
    for name in names:
        column = name
        if name in UNIT_FRAME_COLUMNS:
            column = f"__covariate_{name}"
            while column in names or column in columns.values():
                column += "_"
        columns[name] = column
    return columns


def raise_legacy_compliance_state(*, study_id: str, reason: str) -> NoReturn:
    """Refuse identified legacy state that cannot recover cohort uptake."""
    _raise("source.compliance_summary.legacy_uptake_state", study_id=study_id, reason=reason)


def validate_compliance_completion(design: Encouragement, *, as_of: object | None) -> None:
    """Reject completion policies that cannot identify a finalized cohort."""
    if as_of is None or design.uptake.window_days is None:
        _raise("source.compliance_summary.completed_window_required")


def validate_compliance_design_match(summary: ComplianceSummary, design: Encouragement) -> None:
    """Refuse a reloaded/reused ``ComplianceSummary`` whose own design/cohort/
    time identity disagrees with the *design* the caller is requesting it
    for -- the "design/window mismatch" every producer/consumer validates
    before a readout uses the summary."""
    if not summary.matches_design(design):
        _raise(
            "source.compliance_summary.design_mismatch",
            study_id=summary.study_id,
            cohort=summary.cohort,
            window_days=summary.window_days,
            one_sided=summary.one_sided,
            design_fact=design.uptake.fact,
            design_window_days=design.uptake.window_days,
            design_one_sided=design.one_sided,
            control_group=summary.control_group,
            design_control_group=str(design.control_group),
        )


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "source.compliance_arm.n_units_positive": "ComplianceArm for group {group_id!r} needs n_units >= 1, got {n_units}",
        "source.compliance_arm.uptake_total_bounds": "ComplianceArm for group {group_id!r}: uptake_total={uptake_total!r} must be finite and within [0, n_units={n_units}]",
        "source.compliance_arm.partial_cluster_family": "ComplianceArm for group {group_id!r} carries a partial cluster-uptake bivariate family -- ref_uptake/cluster_uptake1/cluster_uptake2/ref_size/cluster_size1/cluster_size2/cluster_cross must all be present together or all be None",
        "source.compliance_arm.cluster_family_declaration_mismatch": "ComplianceArm for group {group_id!r}: n_clusters={n_clusters!r} but cluster-uptake bivariate family present={has_family} -- both must agree",
        "source.compliance_arm.n_clusters_positive": "ComplianceArm for group {group_id!r} needs n_clusters >= 1, got {n_clusters}",
        "source.compliance_arm.cluster_family_finite": "ComplianceArm for group {group_id!r}: {name}={value!r} is not finite",
        "source.compliance_summary.duplicate_arms": "ComplianceSummary for study {study_id!r} carries duplicate arm(s) {duplicates!r} -- exactly one canonical arm per group_id is required",
        "source.compliance_summary.cluster_family_mismatch": "ComplianceSummary for study {study_id!r}: declared cluster={declared_cluster!r} but arm {group_id!r} carries cluster-uptake family present={has_family} -- every arm must match the summary's own declared cluster grain",
        "source.compliance_summary.legacy_uptake_state": RefusalSpec(
            "source.compliance_summary.legacy_uptake_state",
            CapabilityError,
            template="ComplianceSummary for study {study_id!r} is unavailable from this reloaded source: {reason} -- re-export moments/artifacts built after compliance_summary support was added",
        ),
        "source.compliance_summary.design_mismatch": "ComplianceSummary for study {study_id!r} was built for cohort={cohort!r} window_days={window_days!r} one_sided={one_sided!r}, but the requested design declares uptake.fact={design_fact!r} window_days={design_window_days!r} one_sided={design_one_sided!r}; control={control_group!r} versus {design_control_group!r} -- these must agree before a caller can trust it",
        "source.compliance_summary.completed_window_required": "completed compliance requires an as-of day and a bounded uptake window",
        "source.compliance_summary.invalid_state": "Invalid compliance sufficient state: {reason}",
    },
)
_raise = raiser(_REFUSALS)


def invalid_compliance_state(reason: str) -> NoReturn:
    """Refuse malformed current state, separately from absent legacy state."""
    _raise("source.compliance_summary.invalid_state", reason=reason)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_compliance_source_match(
    summary: ComplianceSummary, context: SourceContext, *, as_of: object | None = None
) -> None:
    """Keep source identity and the requested time bound attached to the data."""
    if (summary.study_id, summary.cluster, summary.as_of) != (
        context.study_id,
        context.cluster,
        as_of,
    ):
        invalid_compliance_state("compliance source, cluster or as-of identity mismatch")


#: ComplianceArm's cluster-grain wire fields, in the compliance plan's order
#: (COMPLIANCE_ARM_FROM_CLUSTER_ROW also maps each to its group_summary slot).
_CLUSTER_UPTAKE_FIELDS = tuple(
    name for name in COMPLIANCE_ARM_FROM_CLUSTER_ROW if name != "n_clusters"
)


@dataclass(frozen=True, slots=True)
class ComplianceArm:
    """One arm's design-level uptake sufficient state.

    Unit count and uptake total, computed on the design's own declared
    enrollment/uptake cohort -- never any outcome metric's moments, so
    metric declaration order, count, and per-metric missingness cannot
    change it. When a randomization/dependence cluster is declared, also
    carries the **complete centered bivariate moments** of cluster uptake
    totals ``U_g`` and cluster sizes ``M_g``: each family's reference mean,
    first residual sum, second residual sum, and their cross moment --
    the same sufficient state
    :func:`~increment.estimation.variance.cluster_uptake_moments` reads off
    an ``ArmStats`` row with ``x_role="uptake_total"``. These fields are
    never derived by averaging per-cluster uptake *rates*: a member-weighted
    ratio-of-totals needs the totals themselves.

    ``n_clusters`` is the independent cluster count ``K``; ``n_units`` is
    the true unit count. The two are tracked separately because, at cluster
    grain, the moments above are computed over the ``K`` clusters, not the
    ``n_units`` members -- collapsing them onto a single field the way a
    cluster-grain ``ArmStats.n`` does would discard one of the two counts.
    """

    group_id: str
    n_units: int
    uptake_total: float
    n_clusters: int | None = None
    ref_uptake: float | None = None
    cluster_uptake1: float | None = None
    cluster_uptake2: float | None = None
    ref_size: float | None = None
    cluster_size1: float | None = None
    cluster_size2: float | None = None
    cluster_cross: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            invalid_compliance_state("group_id must be a nonempty string")
        if not _integer(self.n_units) or self.n_units < 1:
            _raise(
                "source.compliance_arm.n_units_positive",
                group_id=self.group_id,
                n_units=self.n_units,
            )
        if not _number(self.uptake_total) or not (0.0 <= self.uptake_total <= self.n_units):
            _raise(
                "source.compliance_arm.uptake_total_bounds",
                group_id=self.group_id,
                uptake_total=self.uptake_total,
                n_units=self.n_units,
            )
        cluster_values = tuple(getattr(self, name) for name in _CLUSTER_UPTAKE_FIELDS)
        present = [v for v in cluster_values if v is not None]
        if present and len(present) != len(cluster_values):
            _raise("source.compliance_arm.partial_cluster_family", group_id=self.group_id)
        has_family = bool(present)
        if (self.n_clusters is not None) != has_family:
            _raise(
                "source.compliance_arm.cluster_family_declaration_mismatch",
                group_id=self.group_id,
                n_clusters=self.n_clusters,
                has_family=has_family,
            )
        if self.n_clusters is not None and (
            not _integer(self.n_clusters) or not 1 <= self.n_clusters <= self.n_units
        ):
            _raise(
                "source.compliance_arm.n_clusters_positive",
                group_id=self.group_id,
                n_clusters=self.n_clusters,
            )
        for name, value in zip(_CLUSTER_UPTAKE_FIELDS, cluster_values, strict=True):
            if value is not None and not _number(value):
                _raise(
                    "source.compliance_arm.cluster_family_finite",
                    group_id=self.group_id,
                    name=name,
                    value=value,
                )

        if self.n_clusters is not None:
            try:
                self._validate_bivariate()
            except OverflowError:
                invalid_compliance_state("cluster state cannot represent the declared totals")

    def _validate_bivariate(self) -> None:
        k = self.n_clusters
        assert k is not None
        values = [getattr(self, name) for name in _CLUSTER_UPTAKE_FIELDS]
        ru, u1, u2, rm, m1, m2, um = map(Fraction, values)
        for total, reconstructed in ((self.uptake_total, k * ru + u1), (self.n_units, k * rm + m1)):
            # Accumulation and reference recovery each round at their own scale.
            slack = 64 * math.ulp(float(max(abs(total), abs(reconstructed), 1)))
            if abs(reconstructed - Fraction(total)) > slack:
                invalid_compliance_state("cluster references/residuals disagree with totals")
        vu, vm, cov = u2 - u1 * u1 / k, m2 - m1 * m1 / k, um - u1 * m1 / k
        slack = Fraction(64 * math.ulp(float(max(abs(u2), abs(m2), abs(um), 1))))
        if u2 < 0 or m2 < 0 or vu < -slack or vm < -slack:
            invalid_compliance_state("cluster second moments are inconsistent")
        if cov * cov > max(vu, 0) * max(vm, 0) + slack * (abs(vu) + abs(vm) + slack):
            invalid_compliance_state("cluster cross moment violates covariance bounds")

    def to_wire(self) -> dict[str, object]:
        """JSON-serializable payload for one arm."""
        return {
            "group_id": self.group_id,
            "n_units": self.n_units,
            "uptake_total": self.uptake_total,
            "n_clusters": self.n_clusters,
            "ref_uptake": self.ref_uptake,
            "cluster_uptake1": self.cluster_uptake1,
            "cluster_uptake2": self.cluster_uptake2,
            "ref_size": self.ref_size,
            "cluster_size1": self.cluster_size1,
            "cluster_size2": self.cluster_size2,
            "cluster_cross": self.cluster_cross,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> ComplianceArm:
        """Parse complete typed arm state, refusing malformed current payloads."""
        payload = cast("Mapping[str, Any]", cast_mapping(payload))
        expected = {"group_id", "n_units", "uptake_total", "n_clusters", *_CLUSTER_UPTAKE_FIELDS}
        if set(payload) != expected:
            invalid_compliance_state("arm payload has missing or unknown fields")
        return cls(
            group_id=payload["group_id"],
            n_units=payload["n_units"],
            uptake_total=payload["uptake_total"],
            n_clusters=payload["n_clusters"],
            **{name: payload[name] for name in _CLUSTER_UPTAKE_FIELDS},
        )


@dataclass(frozen=True, slots=True)
class ComplianceSummary:
    """Design-level compliance sufficient state: one canonical arm per group.

    Built by ``MomentSource.compliance_summary``, independent of any
    outcome metric's declaration, order, count, or missingness -- the
    population is the design's own declared enrollment/uptake cohort
    (``design.uptake.fact``, honoring ``design.uptake.window_days`` and
    ``as_of``). ``cohort``/``window_days``/``one_sided``/``cluster``/
    ``as_of`` are this summary's design/cohort/time identity: a consumer
    combining or reloading summaries validates them before use rather than
    trusting a numerically-agreeing-by-accident pair.

    Frozen and duplicate-checked at construction: mutating a caller's own
    nested arms sequence after passing it in cannot alter this summary
    (``arms`` is copied into an immutable tuple), and reassigning any field
    raises.
    """

    study_id: str
    cohort: str
    window_days: int | None
    one_sided: bool
    cluster: str | None
    as_of: object | None
    arms: tuple[ComplianceArm, ...]
    control_group: str = "control"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.study_id, self.cohort, self.control_group)
        ):
            invalid_compliance_state(
                "study, cohort and control identities must be nonempty strings"
            )
        if not isinstance(self.one_sided, bool):
            invalid_compliance_state("one_sided must be Boolean")
        if self.window_days is not None and (
            not _integer(self.window_days) or self.window_days < 1
        ):
            invalid_compliance_state("window_days must be a positive integer or null")
        if self.cluster is not None and (not isinstance(self.cluster, str) or not self.cluster):
            invalid_compliance_state("cluster must be a nonempty string or null")
        if self.as_of is not None and not (
            isinstance(self.as_of, (date, str)) or _number(self.as_of)
        ):
            invalid_compliance_state(
                "as_of must be a calendar, string or finite numeric day scalar"
            )
        if not isinstance(self.arms, Sequence) or not all(
            isinstance(arm, ComplianceArm) for arm in self.arms
        ):
            invalid_compliance_state("arms must contain typed ComplianceArm values")
        arms = tuple(sorted(self.arms, key=lambda arm: arm.group_id))
        object.__setattr__(self, "arms", arms)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for arm in arms:
            if arm.group_id in seen:
                duplicates.add(arm.group_id)
            seen.add(arm.group_id)
        if duplicates:
            _raise(
                "source.compliance_summary.duplicate_arms",
                study_id=self.study_id,
                duplicates=tuple(sorted(duplicates)),
            )
        clustered = self.cluster is not None
        for arm in arms:
            has_family = arm.n_clusters is not None
            if has_family != clustered:
                _raise(
                    "source.compliance_summary.cluster_family_mismatch",
                    study_id=self.study_id,
                    group_id=arm.group_id,
                    declared_cluster=self.cluster,
                    has_family=has_family,
                )

    def arm(self, group_id: str) -> ComplianceArm | None:
        """The canonical arm for *group_id*, or ``None`` if absent."""
        return next((a for a in self.arms if a.group_id == group_id), None)

    def matches_design(self, design: Encouragement) -> bool:
        """Whether this summary's own identity agrees with *design*'s
        declared uptake cohort -- the "design/window mismatch" check a
        reloaded (wire) summary must pass before a caller trusts it."""
        return (
            self.control_group == str(design.control_group)
            and self.cohort == design.uptake.fact
            and self.window_days == design.uptake.window_days
            and self.one_sided == design.one_sided
        )

    def to_wire(self) -> dict[str, object]:
        """Version-1 payload, with sorted arms and explicit nullable time identity.

        Total cubes require as_of=null. Direct summary serialization retains
        ISO dates and numbers; strings and datetimes are tagged to retain their type.
        """
        return {
            "version": 1,
            "study_id": self.study_id,
            "control_group": self.control_group,
            "as_of": (
                self.as_of.isoformat()
                if type(self.as_of) is date
                else {"datetime": self.as_of.isoformat()}
                if isinstance(self.as_of, datetime)
                else {"label": self.as_of}
                if isinstance(self.as_of, str)
                else self.as_of
            ),
            "cohort": self.cohort,
            "window_days": self.window_days,
            "one_sided": self.one_sided,
            "cluster": self.cluster,
            "arms": [arm.to_wire() for arm in self.arms],
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, object], *, study_id: str) -> ComplianceSummary:
        """Parse and freeze current state; malformed state has a coded refusal."""
        payload = cast("Mapping[str, Any]", cast_mapping(payload))
        expected = {
            "version",
            "study_id",
            "control_group",
            "as_of",
            "cohort",
            "window_days",
            "one_sided",
            "cluster",
            "arms",
        }
        if (
            set(payload) != expected
            or type(payload["version"]) is not int
            or payload["version"] != 1
        ):
            invalid_compliance_state("unsupported version or incomplete compliance payload")
        if payload["study_id"] != study_id:
            invalid_compliance_state("compliance study identity differs from cube")
        arms_payload = payload["arms"]
        if not isinstance(arms_payload, (list, tuple)):
            invalid_compliance_state("arms must be a sequence")
        try:
            as_of = payload["as_of"]
            if isinstance(as_of, str):
                as_of = date.fromisoformat(as_of)
            elif isinstance(as_of, Mapping):
                if set(as_of) == {"label"} and isinstance(as_of["label"], str):
                    as_of = as_of["label"]
                elif set(as_of) == {"datetime"} and isinstance(as_of["datetime"], str):
                    as_of = datetime.fromisoformat(as_of["datetime"])
                else:
                    invalid_compliance_state("invalid tagged as-of scalar")
            return cls(
                study_id=study_id,
                control_group=payload["control_group"],
                cohort=payload["cohort"],
                window_days=payload["window_days"],
                one_sided=payload["one_sided"],
                cluster=payload["cluster"],
                as_of=as_of,
                arms=tuple(ComplianceArm.from_wire(cast_mapping(row)) for row in arms_payload),
            )
        except CodedError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            invalid_compliance_state(f"malformed compliance payload: {exc}")


def cast_mapping(value: object) -> Mapping[str, object]:
    """Reject a non-mapping arm row before ``ComplianceArm.from_wire``
    would otherwise raise a less specific error."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        invalid_compliance_state("expected an arm row mapping with string keys")
    return cast("Mapping[str, object]", value)


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Immutable statistical construction state owned by a moment source."""

    study_id: str
    design: Randomized | Encouragement | Observational | None
    plan: CompiledDecisionPlan
    metrics: tuple[Metric, ...]
    configs: tuple[ResolvedMetricConfig, ...]
    cluster: str | None
    #: Declared whole-cluster policy-deployment grain: "cluster" only for an
    #: explicit cluster-level intervention, otherwise "unit". Neither `cluster`
    #: (a randomization/dependence grain) nor covariance clustering implies it.
    #: Unit-frame-serving sources pass the declared value; others keep "unit".
    intervention_grain: Literal["unit", "cluster"] = "unit"
    #: Trusted experiment declaration for the triggered population, when present.
    #: It identifies scope only; it does not certify or load triggered evidence.
    trigger_name: str | None = None

    #: Per-metric missing-value policy retained by the source that owns the
    #: metric catalog. Evidence consumers must not reach into loader internals.
    metric_missingness: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_missingness", MappingProxyType(dict(self.metric_missingness))
        )

    # SourceContext is construction state shared by every population view.
    # Triggered population selection belongs to the source-owned adapter.
