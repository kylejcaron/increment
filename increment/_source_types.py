"""Immutable source-context state and the `MomentSource` protocol.

Leaf module under `increment.decision`: only `TYPE_CHECKING` imports
point upward at `decision`, `sources`, `semantics`, or `_analysis_config`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from inspect import getattr_static
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, TypeGuard, runtime_checkable

from increment._evidence_types import ScalarBreakoutRows as ScalarBreakoutRows
from increment.errors import CapabilityError, RefusalSpec, refuse

if TYPE_CHECKING:
    from increment._evidence_types import (
        BreakoutMethodMomentsRequest,
        ComplianceRequest,
        ComplianceSeriesRequest,
        ComplianceSummary,
        CountEvidence,
        CountsRequest,
        EvidenceScope,
        EvidenceScopeResult,
        MethodFitAnswer,
        MethodFitRequest,
        MomentRows,
        MomentsRequest,
        SequentialCapture,
        SequentialCaptureRequest,
        SourceContext,
        UnitRows,
        UnitRowsRequest,
    )


def breakout_dimension_column(covariates: Sequence[str]) -> str:
    """Name the unit-frame segment column so it displaces no declared covariate.

    Producers and consumers derive the name from the same covariate request,
    so a covariate called ``__breakout_dimension`` keeps its own values.
    """
    name = "__breakout_dimension"
    while name in covariates:
        name += "_"
    return name


if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._analysis_config import ResolvedMetricConfig
    from increment.semantics.design import Encouragement
    from increment.semantics.models import CupedCovariate, Metric
    from increment.sources import SourceOperation

Grain = Literal["total", "daily", "asof"]
Shape = Literal["unit_summary", "unit_panel"]


class _BuiltinMomentSource(Protocol):
    """Legacy methods available only on built-in loaders at acquisition."""

    capabilities: frozenset[Grain]
    operations: frozenset[SourceOperation]

    @property
    def context(self) -> SourceContext:
        """Immutable design, plan, metric, cluster, and study state."""
        ...

    @property
    def breakouts(self) -> Sequence[str]:
        """Unit-stable reporting identities this source can expose."""
        ...

    @property
    def shape(self) -> Literal["unit_summary", "unit_panel"] | None:
        """The source shape when it is owned by a dataframe adapter."""
        ...

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        """Centered moments for `metric` at `grain`, optionally split by dimension.

        `completed_windows_only` applies only to "asof"-capable sources: it
        gates a retention series to closed (final) units, not the default
        open (provisional) one. Ignored elsewhere.
        """
        ...

    def unit_frame(self, metric: Metric, *, covariates: Sequence[str] = ()) -> IntoDataFrame:
        """One row per unit: `unit_id`, `group_id`, `y`, plus `covariates`.

        For a ratio metric, `y` is the numerator only; `y_den` (the
        per-unit denominator) is a separate column a caller combines with `y`.
        A declared `context.cluster` adds `cluster_id` metadata automatically,
        independently of `covariates`; callers must not request its source name.
        Each covariate is carried under `unit_frame_covariate_columns(covariates)`:
        its own name unless that is one of the frame's own columns. Only frame
        producers remap (their covariate names are the caller's own columns);
        native and artifact producers emit each binding under its declared alias
        and refuse a reserved alias before acquisition
        (`definition.experiment.covariate_name_reserved`).
        """
        ...

    def unit_counts(self) -> dict[str, int]: ...

    def cluster_counts(self) -> dict[str, int]:
        """Per-group distinct randomization-cluster counts.

        Only a source with a declared cluster column can answer this; others
        raise `cluster-grain capability refusal` instead of returning
        unit-grain counts under a cluster-grain name.
        """
        ...

    def compliance_dates(self) -> Sequence[object]:
        """Enrollment/uptake day labels, independent of outcome filters.

        Sources without as-of compliance support raise a coded capability refusal.
        """
        ...

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state, independent of any metric.

        Computed on `design`'s own declared enrollment/uptake cohort
        (`design.uptake.fact`, honoring `design.uptake.window_days`):
        declaring a different outcome metric, changing metric count or
        order, or an outcome metric's own missingness policy never changes
        the result. `as_of`, when given, matches this source's enrollment
        day-axis `ds` values returned by `compliance_dates()` and freezes
        cumulative uptake state through that day; `None` reads the declared window
        in full (the "ever" / whole-window cohort). ``completed_windows_only``
        requires an as-of day and a bounded uptake window, and admits only
        units whose uptake window has closed by that day.
        """
        ...

    def sql(self, *, grain: Grain = "total") -> dict[str, str]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ScalarEvidenceDeclaration:
    """Scalar evidence retained by a source for readout construction."""

    bindings: Mapping[str, tuple[CupedCovariate, ...]] | None
    moments_only: bool
    recipe_fixed: bool
    recipe_configs: tuple[ResolvedMetricConfig, ...]
    anonymous_metrics: frozenset[str]
    method_evidence: bool
    breakout_evidence: bool
    covariate_grains: frozenset[Grain] | None


@dataclass(frozen=True, slots=True)
class EvidenceCapabilities:
    """Evidence and product operations a source can provide."""

    kind: Literal[
        "definitions", "artifact", "unit_summary", "unit_panel", "switchback", "moments", "breakout"
    ]
    grains: frozenset[Grain]
    breakouts: tuple[str, ...]
    shape: Shape | None
    operations: frozenset[SourceOperation]
    unit_rows: bool
    raw_outcomes: bool
    method_fit: Literal["warehouse", "local", "retained", "none"]
    compliance_axis: Literal["total", "asof", "none"]
    count_grains: frozenset[Literal["unit", "cluster", "assignment", "triggered"]]
    populations: frozenset[Literal["assigned", "triggered"]]
    sequential_capture: Literal["construction", "asof", "replay", "none"]
    scalar: ScalarEvidenceDeclaration | None
    available_covariates: frozenset[str] | None
    sql_grains: frozenset[Grain] = frozenset()
    capture: Literal["snapshot", "immutable", "live"] = "live"


class _OwnedLoaderAdapter:
    """Marker for internal adapters whose ``source`` is the owned loader."""

    __slots__ = ()
    source: Any
    provider_owned: bool = False


def _loader_owner(source: object) -> object:
    return source.source if isinstance(source, _OwnedLoaderAdapter) else source


@runtime_checkable
class EvidenceProvider(Protocol):
    """Primitive request interface for custom evidence providers.

    Providers may additionally implement ``sequential_snapshot(request:
    SequentialSnapshotRequest)`` to return their exact current checkpoint;
    that optional method is deliberately not part of this runtime protocol.
    """

    evidence: EvidenceCapabilities
    context: SourceContext

    def moments(self, request: MomentsRequest) -> MomentRows: ...
    def breakout_method_moments(
        self, request: BreakoutMethodMomentsRequest
    ) -> ScalarBreakoutRows: ...
    def unit_rows(self, request: UnitRowsRequest) -> UnitRows: ...
    def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer: ...
    def compliance(self, request: ComplianceRequest) -> ComplianceSummary: ...
    def compliance_series(
        self, request: ComplianceSeriesRequest
    ) -> tuple[tuple[object, ComplianceSummary], ...]: ...
    def counts(self, request: CountsRequest) -> CountEvidence: ...
    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture: ...
    def scope(self, request: EvidenceScope) -> EvidenceScopeResult: ...


def is_builtin_source(source: object) -> TypeGuard[_BuiltinMomentSource]:
    """Identify loader instances by their defining module, never by capabilities."""
    loader_modules = (
        "increment.frame",
        "increment.sources",
        "increment.switchback",
        "increment.query",
    )
    return any(
        cls.__module__ == module or cls.__module__.startswith(f"{module}.")
        for cls in type(source).__mro__
        for module in loader_modules
    )


def is_custom_source(source: object) -> TypeGuard[EvidenceProvider]:
    """Identify a structural provider at the acquisition boundary."""
    return not is_builtin_source(source) and isinstance(source, EvidenceProvider)


def evidence_is_declared(source: object) -> bool:
    """Whether a source supplies its own evidence declaration."""
    declaration = getattr_static(source, "evidence", None)
    return declaration is not None and isinstance(
        getattr(source, "evidence", None), EvidenceCapabilities
    )


def declared_evidence(source: object) -> EvidenceCapabilities:
    """Return the source's explicit evidence declaration."""
    declaration = getattr(source, "evidence", None)
    if not isinstance(declaration, EvidenceCapabilities):
        refuse_provider_unavailable(source)
    return declaration


PROVIDER_UNAVAILABLE = RefusalSpec(
    "evidence.provider_unavailable",
    CapabilityError,
    template=(
        "{constructor} must declare EvidenceCapabilities or implement EvidenceProvider "
        "before it can provide evidence."
    ),
    keys=frozenset({"constructor"}),
)


def refuse_provider_unavailable(source: object) -> NoReturn:
    refuse(PROVIDER_UNAVAILABLE, constructor=type(source).__name__)


def compliance_summary_series(
    src: _BuiltinMomentSource,
    design: Encouragement,
    *,
    completed_windows_only: bool,
) -> dict[Any, ComplianceSummary]:
    """Read compliance on its own enrollment/uptake axis, never an outcome axis."""
    from increment._evidence_types import (
        raise_legacy_compliance_state,
        validate_compliance_source_match,
    )

    provider = getattr(src, "compliance_dates", None)
    if not callable(provider):
        raise_legacy_compliance_state(
            study_id=src.context.study_id,
            reason="source must implement the enrollment/uptake compliance_dates() contract",
        )
    # One series is one readout: a warehouse source checks assignment
    # integrity once here rather than on every per-date read.
    validated = getattr(src, "_validated_assignments", nullcontext)
    summaries = {}
    with validated():
        dates = provider()
        for ds in dict.fromkeys(dates):
            summary = src.compliance_summary(
                design, as_of=ds, completed_windows_only=completed_windows_only
            )
            validate_compliance_source_match(summary, src.context, as_of=ds)
            if summary.arms:
                summaries[ds] = summary
    return summaries
