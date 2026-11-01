"""Shared analysis execution for bound experiment sources."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment._source_types import _BuiltinMomentSource
    from increment.breakout.estimates import (
        BreakoutEstimates,
        DailyLiftEstimates,
        DailyMetricValues,
        LiftEstimates,
    )
    from increment.kernel.context import EvidenceExecutionContext
    from increment.query.artifact_contract import ArtifactContext, ArtifactStore
    from increment.query.session import WarehouseSession
    from increment.semantics.artifact import ArtifactExtensionRequest, UnitDayArtifactRef


from increment import readouts
from increment._analysis_config import (
    UNSET,
    ResolvedMetricConfig,
    _Unset,
    effective_methods,
    normalize_display_correction,
    overlay_configs,
    select_exploratory_metrics,
    select_metrics,
)
from increment._binding import _UNDECLARED_BREAKOUT, SourceBinding, _ArtifactOpenSpec
from increment._breakout_readouts import BreakoutReadouts, BreakoutRequest
from increment._day_axis import (
    _CLUSTERED_DAY_AXIS,
    _TRIGGER_UNSUPPORTED,
    DayAxisReadouts,
)
from increment._evidence_dispatch import ContrastEvidenceOperation, ContrastReadoutRequest
from increment._evidence_refusals import CONTRAST_UNAVAILABLE as _CONTRAST_UNAVAILABLE
from increment._evidence_refusals import INVALID_POPULATION as _INVALID_POPULATION
from increment._evidence_types import (
    DayAxisRequest,
    EvidenceScope,
    SequentialCaptureRequest,
    SourceContext,
)
from increment._literals import Backend, Correction, ValueScale
from increment._source_admission import _require_analysis_operation
from increment._source_operations import (
    DashboardBreakoutReads,
    DashboardGroupData,
    DashboardGroupDataOperation,
    DashboardSnapshotPayload,
    ExploratorySourceOperation,
)
from increment._source_types import declared_evidence, is_custom_source
from increment._source_workflows import EXPLORATORY_SOURCE_LIMITED as _EXPLORATORY_SOURCE_LIMITED
from increment.decision import (
    AnalysisState,
    ArmAnalysisState,
    ContrastAnalysisState,
    ContrastContext,
    ContrastSource,
    DefinitionsArmAnalysisState,
)
from increment.errors import (
    CodedError,
    RefusalSpec,
    UnsupportedRequestError,
)
from increment.errors import refuse as _refuse
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.engine import Method
from increment.evidence import evidence_provider
from increment.plan import with_unassigned_procedures
from increment.readouts._evidence import (
    NATIVE_ROUTE,
    day_axis_route,
    source_route,
)
from increment.readouts._whole_window import (
    WholeWindowExploratoryRequest,
    WholeWindowFamilyRequest,
    WholeWindowReadouts,
    WholeWindowRequest,
    reject_role_overrides_under_contrast,
    validate_contrast_request,
)
from increment.semantics.models import Breakout, Definitions, Experiment

__all__ = ["Analysis", "fit_predeclared_adjustment"]


_EXPLORATORY_SEQUENTIAL = RefusalSpec(
    "facade.analysis.exploratory_metrics_sequential",
    UnsupportedRequestError,
    template="{method}: exploratory_metrics {names!r} are chosen after data are visible, so no registered sequential model covers them under {inference}; read the declared metrics alone, or use a fixed-horizon plan.",
)
_UNCORRECTED_SEQUENTIAL = RefusalSpec(
    "facade.analysis.uncorrected_segments_sequential",
    UnsupportedRequestError,
    template="uncorrected segment rows have no fixed-horizon p-values under {inference}; read the registered breakout with run_breakout() instead.",
)


def fit_predeclared_adjustment(
    frame,
    *,
    unit: str,
    group: str,
    control: str,
    metric: str,
    covariate: str,
    covariate_missing: Literal["impute", "zero", "error"] = "impute",
):
    """Fit a sequential CUPED coefficient and covariate centre from pre-period data.

    *frame* is one row per assigned unit observed BEFORE the experiment reads
    any outcome: *metric* is the metric measured in a pre-period window and
    the in-experiment covariate is measured relative to exposure. The rows
    are reduced to the same centered moments the fixed-horizon frame path
    builds, and the coefficient is ``estimation.cuped.fit_cuped``'s inverse-n
    weighted within-arm slope with the centre its count-weighted pooled
    covariate mean, so sequential and fixed-horizon CUPED agree on what the
    coefficient is.

    The result is a declaration: pass it through ``InferenceSpec.adjustments``
    or ``ScalarMeanModel.adjustment`` and capture retains
    ``Y - coefficient * (X - center)``. Costs relative to the fixed-horizon
    fit: the coefficient is not contrast-optimal for the in-experiment
    outcome, and ``coefficient * (E[X] - center)`` is a first-order bias on
    the ratio scale the scalar-mean route reports whenever the covariate
    drifts between the pre-period and the experiment.
    """
    from fractions import Fraction

    from increment.estimation.cuped import fit_cuped
    from increment.estimation.engine import Method, _df_to_arms
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import CupedCovariate
    from increment.semantics.sequential import PredeclaredAdjustment

    covariate_declaration = CupedCovariate(
        name=covariate,
        kind="continuous",
        missing="refuse" if covariate_missing == "error" else covariate_missing,
    )
    source = from_unit_summary(
        frame,
        unit=unit,
        group=group,
        control=control,
        metrics=[
            MetricSpec(
                name=metric,
                decision_method=Method.cuped(covariates=covariate_declaration),
            )
        ],
        experiment_id="pre-period",
    )
    fit = fit_cuped(_df_to_arms(source.raw_moments))
    return PredeclaredAdjustment(
        coefficient=Fraction(fit.theta), center=Fraction(fit.mean_x_pooled)
    )


if TYPE_CHECKING:
    import pyarrow as pa
    from ibis.expr.types import Table

    from increment._health import SrmObservations
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.cate import CateResult, Covariate
    from increment.estimation.diagnostics import NotApplicable, SRMResult
    from increment.estimation.health import ExperimentHealth
    from increment.estimation.inference import Prior
    from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
    from increment.estimation.targeting import (
        CateValidation,
        ClusterBootstrap,
        TargetingRule,
        TargetingSelection,
    )
    from increment.power import Baseline
    from increment.power.switchback import SwitchbackBaseline
    from increment.query.source import ArtifactMomentSource
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Breakout, Metric
    from increment.sequential_state import SequentialCheckpoint
    from increment.switchback import SwitchbackAssignmentDiagnostic


# Public facade


def _cuped_recommendation_admitted(
    source: _BuiltinMomentSource, config: ResolvedMetricConfig
) -> bool:
    """Check the proposed CUPED method against the shared arm-support contract."""
    from increment.estimation.arm_contract import (
        MetricCapabilities,
        Supported,
        arm_cuped_advisory_support,
    )
    from increment.estimation.engine import is_open_ended
    from increment.sequential_source import adjustment_kind

    context = source.context
    metric = config.metric
    inference = context.plan.inference
    registration = getattr(inference, "registration", None)
    adjustment = adjustment_kind(registration, metric.name) if registration is not None else None
    capabilities = MetricCapabilities(
        metric_type=metric.type if metric.type not in ("total", "active") else "mean",
        value_scale="relative",
        winsorization=(
            "percentile"
            if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
            else "fixed"
            if getattr(metric, "winsorization", None) is not None
            else "none"
        ),
        outcome_window="unbounded" if is_open_ended(metric) else "bounded",
        uptake_window="not_applicable",
    )
    support = arm_cuped_advisory_support(
        dependence="cluster" if context.cluster is not None else "iid",
        metric=capabilities,
        inference=inference,
        adjustment=adjustment,
        prior_present=config.prior is not None,
    )
    return isinstance(support, Supported)


_T = TypeVar("_T")


_ACTIVE_EXECUTION_CONTEXT: ContextVar[Any | None] = ContextVar(
    "_ACTIVE_EXECUTION_CONTEXT", default=None
)


@contextmanager
def _execution_context_scope(context: Any) -> Iterator[None]:
    token = _ACTIVE_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_EXECUTION_CONTEXT.reset(token)


def _current_execution_context() -> Any | None:
    return _ACTIVE_EXECUTION_CONTEXT.get()


class Analysis:
    """Execute operations over one immutable source binding."""

    _readout_srm_observations: SrmObservations | None = None

    def __init__(self, binding: SourceBinding) -> None:
        self._binding = binding

    @classmethod
    def _from_binding(cls, binding: SourceBinding) -> Self:
        return cls(binding)

    @property
    def _state(self) -> AnalysisState:
        return self._binding.state

    @property
    def _artifact_open_spec(self) -> _ArtifactOpenSpec | None:
        return self._binding.artifact_open

    @property
    def _input_backend(self) -> Backend | None:
        return self._binding.input_backend

    @_input_backend.setter
    def _input_backend(self, value: Backend | None) -> None:
        self._binding = replace(self._binding, input_backend=value)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release this analysis's resources without closing caller connections."""
        close = getattr(self._state.source, "close", None)
        if callable(close):
            close()

    def _require_arm_state(
        self, method: str, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> ArmAnalysisState:
        if self._state.family != "arm_moments":
            if population == "triggered":
                _refuse(
                    _TRIGGER_UNSUPPORTED,
                    method=method,
                    experiment="<unknown>",
                    trigger=None,
                    route=(
                        "switchback"
                        if self._state.family == "contrast"
                        else source_route(self._src)
                    ),
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            _refuse(_CONTRAST_UNAVAILABLE, method=method)
        return self._state

    def _ensure_artifact_source(self) -> _BuiltinMomentSource:
        source = cast("_BuiltinMomentSource", self._binding.state.source)
        spec = self._binding.artifact_open
        if spec is not None and cast("ArtifactMomentSource", source).closed:
            from increment.query.source import open_artifact

            source = open_artifact(
                spec.store,
                spec.ref,
                expected_context=spec.expected_context,
                verification=spec.verification,
            )
            self._binding = self._binding.reopened(source)
        return source

    @property
    def _src(self) -> Any:
        return self._ensure_artifact_source()

    @property
    def _context(self) -> SourceContext | ContrastContext:
        return self._src.context

    @property
    def _defs(self) -> Definitions | None:
        return self._binding.definitions

    @property
    def _experiment_name(self) -> str:
        return self._binding.experiment_name

    @property
    def _backend(self) -> str | None:
        return self._binding.backend

    @property
    def _con(self) -> SQLBackend | None:
        return self._binding.connection

    @property
    def _store(self) -> Literal["auto", "always", "none"]:
        return self._binding.store

    @property
    def _on_mixed_assignment(self) -> Literal["error", "warn", "exclude"]:
        return self._binding.on_mixed_assignment

    @property
    def _session(self) -> WarehouseSession | None:
        return self._binding.session

    @property
    def _metrics(self) -> list[Metric]:
        return self._binding.metrics

    def _execution_config_for_cell(self, cell: Any) -> tuple[Any, tuple[Method, ...]]:
        key = cell.identity.key
        if key.metric == "uptake" and key.estimand == "compliance":
            uptake = getattr(self._design, "uptake", None)
            if uptake is not None:
                return uptake, (Method(name="unadjusted"),)

        configs = getattr(self._context, "configs", None)
        if configs is None:
            raise CodedError(
                "Execution method identity is unavailable",
                code="store.run.identity_incomplete",
                context={"field": "method_identity"},
            )
        config = next((item for item in configs if item.metric.name == key.metric), None)
        if config is None:
            raise CodedError(
                "Cell metric has no execution configuration",
                code="store.run.identity_incomplete",
                context={"field": "metric_identity"},
            )
        return config.metric, tuple(effective_methods(config, design=self._design))

    @property
    def _design(self) -> Randomized | Encouragement | Observational | None:
        return self._binding.design

    @property
    def _plan(self) -> CompiledDecisionPlan:
        return self._binding.plan

    @property
    def _experiment(self) -> Experiment:
        self._require_arm_state("experiment")
        return self._binding.experiment

    def publish_unit_day_artifact(
        self,
        store: ArtifactStore,
        *,
        extensions: Sequence[ArtifactExtensionRequest] = (),
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> UnitDayArtifactRef:
        """Publish this definitions-backed arm analysis as an immutable artifact."""
        from increment._source_workflows import _publish_unit_day_artifact_binding

        return _publish_unit_day_artifact_binding(
            self._binding,
            store,
            extensions=extensions,
            refresh_of=refresh_of,
        )

    def materialize(self) -> Self:
        """Rebuild spine and per-metric stats from current warehouse data.

        Each call performs a fresh scan, overriding ``store="auto"``'s
        second-reduction threshold. A no-op only for ``store="none"`` or
        an experiment declaring no metrics.

        Returns
        -------
        Self
            This analysis, using its refreshed warehouse data.

        Notes
        -----
        A frame or moments source raises :class:`CapabilityError` because
        it has no warehouse relation to materialize.
        """
        from increment._source_workflows import _materialize_binding

        _ = self._src
        _materialize_binding(self._binding)
        return self

    def dashboard_snapshot(
        self, operation: Callable[[Self], DashboardSnapshotPayload], *, metrics: Sequence[Metric]
    ) -> DashboardSnapshotPayload:
        """Prepare one complete dashboard payload against an isolated, pinned source.

        *metrics* names every metric the payload reads, so the pin covers exactly their
        fact sources. Pass the declared metrics together with any
        :attr:`available_metrics` the operation will add through ``exploratory_metrics``:
        an added metric outside the pin is refused rather than read from a later state.
        """
        self._require_arm_state("dashboard_snapshot")
        _require_analysis_operation(
            self._src,
            "readout_snapshot",
            message=(
                "dashboard_snapshot() needs a native Analysis.from_definitions instance "
                "with a warehouse-backed source."
            ),
        )
        scoped = evidence_provider(self._src).scope(
            EvidenceScope(
                metrics=tuple(metrics),
                snapshot=True,
                include_breakouts=True,
            )
        )
        with scoped.source as pinned:
            from increment.readouts._evidence import _builtin_source

            return self._with_pinned_source(_builtin_source(pinned), operation)

    def _with_pinned_source(
        self, pinned: _BuiltinMomentSource, operation: Callable[[Self], _T]
    ) -> _T:
        isolated = copy.copy(self)
        isolated._binding = self._binding.with_source(pinned)
        isolated._readout_srm_observations = {}
        try:
            return operation(isolated)
        finally:
            delattr(isolated, "_readout_srm_observations")

    def dashboard_breakout_reads(self, breakout: Breakout) -> DashboardBreakoutReads:
        """The dashboard's Explore reads through one declared breakout alone.

        A dimension-wide read covers every declared breakout of that property, so one
        source's refusal would refuse its siblings. These reads hand the source only
        *breakout*; families are computed per breakout, so their rows equal this breakout's
        rows from the dimension-wide reads. Only native ``Analysis.from_definitions``
        instances support it: other arm-family sources refuse with
        ``evidence.breakout_unavailable`` and switchback evidence with
        ``facade.analysis.contrast_unavailable``. An undeclared *breakout* is refused
        with ``facade.analysis.undeclared_breakout``.

        Parameters
        ----------
        breakout : Breakout
            One of this experiment's declared breakouts.

        Returns
        -------
        DashboardBreakoutReads
            The cumulative-lift, cumulative-value, daily-value and segment reads.
        """
        state = self._require_arm_state("dashboard_breakout_reads")
        if not isinstance(state, DefinitionsArmAnalysisState):
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self._src,
                (breakout.property,),
                tuple(declared_evidence(self._src).breakouts),
                route="dashboard_breakout_reads() requires a definitions-backed breakout source",
            )
        declared = state.experiment.breakouts
        if breakout not in declared:
            _refuse(
                _UNDECLARED_BREAKOUT,
                requested=(breakout.source, breakout.property),
                declared=tuple((item.source, item.property) for item in declared),
            )
        scoped = copy.copy(self)
        scoped._binding = self._binding.with_breakouts((breakout,))
        return DashboardBreakoutReads(breakout, scoped)

    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[DashboardGroupData, ...]:
        """Return group data for the requested assigned or triggered population."""
        self._require_arm_state("dashboard_group_data")
        selected = select_metrics(
            self._metrics,
            metrics,
            caller="dashboard_group_data",
            require_declared_definitions=True,
        )
        src = _require_analysis_operation(
            self._src,
            "dashboard_group_data",
            DashboardGroupDataOperation,
            message=(
                "dashboard_group_data() needs a native Analysis.from_definitions "
                "instance with a warehouse-backed source."
            ),
        )
        return tuple(
            src.dashboard_group_data(
                metrics=selected, checkpoints=checkpoints, population=population
            )
        )

    def allocation_history(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> pa.Table:
        """Daily and cumulative enrollment history for one named population.

        Columns are ``experiment_id``, ``ds`` (the declared day boundary),
        ``group_id``, ``n_daily``, ``n_cumulative`` and ``analysis_population``.
        Assigned histories use first-exposure dates; triggered histories use
        first eligible trigger dates and count only trigger-eligible units.
        Counts are independent of metric maturity.

        Sources without raw enrollment events and clustered experiments refuse
        this unit-grain timeline with a coded capability error.
        """
        from increment._source_diagnostics import _allocation_history_binding

        _ = self._src
        return _allocation_history_binding(self._binding, population=population)

    def health_check(
        self, *, expected_allocation: Mapping[str, float] | None = None
    ) -> ExperimentHealth:
        from increment._source_diagnostics import _health_check_binding

        _ = self._src
        return _health_check_binding(
            self._binding,
            expected_allocation=expected_allocation,
            observations=self._readout_srm_observations,
        )

    def unit_counts(self) -> dict[str, int]:
        """Return enrolled unit counts by canonical group label.

        Excluded unassigned units appear under ``"(unassigned)"``. Sources
        without total-grain count evidence refuse with a coded capability error.
        """
        from increment._source_diagnostics import _unit_counts_binding

        _ = self._src
        return _unit_counts_binding(self._binding)

    def srm(
        self,
        *,
        expected: dict[str, float] | None = None,
        alpha: float = 0.001,
        inference: Literal["always_valid", "fixed"] = "always_valid",
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SRMResult | NotApplicable:
        """Check assigned or triggered counts against a declared allocation.

        The default is an anytime-valid check for cumulative prefixes under
        a known allocation with constant conditional arm probabilities; pass
        ``inference="fixed"`` for one predeclared Pearson look. On a clustered
        experiment, the chi-square runs over distinct cluster counts per arm.

        ``expected`` supplies the target allocation for this call; otherwise
        the design must declare ``allocation``. SRM never assumes equal shares
        or infers its null from observed counts. An unavailable population or a
        missing target allocation refuses with a coded error; declare
        ``Randomized(control_group=..., allocation={...}, allocation_scheme=...)``
        to check the design allocation.

        Parameters
        ----------
        expected : dict[str, float] | None
            Expected allocation proportions per group.
        alpha : float
            Lifetime significance threshold (default 0.001).
        inference : Literal["always_valid", "fixed"]
            Evidence controlling ``is_srm``. Default ``"always_valid"``.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) checks enrolled counts.
            ``"triggered"`` narrows to the declared trigger population first.

        Returns
        -------
        SRMResult | NotApplicable
            Observational designs return :class:`NotApplicable`, since they
            have no target allocation to test against. Runs normally under
            :class:`Encouragement`.
        """
        from increment._source_diagnostics import _srm_binding

        _ = self._src
        return _srm_binding(
            self._binding,
            expected=expected,
            alpha=alpha,
            inference=inference,
            population=population,
            observations=self._readout_srm_observations,
        )

    def trigger_rates(self) -> dict[str, float]:
        """Observed triggered share per arm.

        Feed a pooled estimate (e.g. the mean of these per-arm rates) into
        ``Baseline(..., trigger_rate=...)``, then ``required_sample_size(...)``,
        to plan a follow-up at the right size.
        """
        from increment._source_diagnostics import _trigger_rates_binding

        _ = self._src
        return _trigger_rates_binding(self._binding)

    def _validate_trigger_fires_in_every_arm(self, source: object | None = None) -> None:
        from increment.readouts._evidence import validate_trigger_fires_in_every_arm

        validate_trigger_fires_in_every_arm(
            cast(Any, self._src if source is None else source),
            self._experiment.trigger,
        )

    def _cate_source(self, method: str) -> _BuiltinMomentSource:
        self._require_arm_state(method)
        experiment = getattr(self, "_experiment", None)
        if experiment is None or experiment.trigger is None:
            return cast("_BuiltinMomentSource", self._src)
        triggered = _require_analysis_operation(
            self._src,
            "triggered_source",
            message=(
                f"{method}() needs a source-owned triggered unit view; this source "
                "cannot reconstruct trigger membership."
            ),
        )
        return triggered.triggered_source()

    def sitewide(
        self,
        metric: str,
        *,
        arm: str | None = None,
        alpha: float = 0.05,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> SitewideImpact | SitewideRatioImpact:
        """Whole-site impact of *metric*'s lift to every unit.

        Native (:meth:`from_definitions`) and adopted artifact sources with a
        published site-volume extension combine enrolled-arm moments with
        whole-site volume evidence. Both populations require an additive metric
        aligned with the calendar-window site total: metrics must be unwindowed
        and use sum or count aggregation. Conversion metrics are refused because
        their per-unit indicator is not additive across event records; declare a
        count metric on the conversion fact to read conversion events site-wide.
        ``population="assigned"`` scales the assignment-window lift
        to all enrolled units. For a triggered experiment,
        ``population="triggered"`` scales the trigger-stratum lift to units in
        finalized trigger membership, while outcomes remain aligned to the
        experiment's calendar window; it additionally requires a declared
        experiment end and discloses the conditional membership assumptions
        on the result.

        One result per call, for one treatment arm - required when
        more than one non-control arm is enrolled. Every other
        enrolled arm is netted out of the counterfactual baseline and
        counted in the ship-to-all population: the number answers
        "``arm`` shipped to every unit, versus no treatment shipped."

        Parameters
        ----------
        metric : str
            Name of a metric declared on this experiment.
        arm : str | None
            ``group_id`` of the treatment arm to score.
        alpha : float
            Two-sided significance level for both intervals. Default 0.05.
        population : Literal["assigned", "triggered"]
            selects assignment membership by default or finalized trigger
            membership for a triggered experiment. Triggered sitewide impact
            uses assignment-anchored calendar-window outcomes (not trigger-
            anchored ``run()`` windows); it is conditional on membership
            independence from treatment and never-triggered units being
            unaffected.

        Returns
        -------
        SitewideImpact | SitewideRatioImpact
            ``SitewideRatioImpact`` for a :class:`RatioMetric`, else ``SitewideImpact``.

        Notes
        -----
        The readout raises :class:`CapabilityError` for sources without
        site volume, windowed or non-additive metrics, triggered impact
        without a declared experiment end, retention or quantile metrics,
        and clusters that do not decompose or whose mean cluster sizes differ by
        more than 20%. Use an unwindowed sum or count metric. Unexposed
        units in a treated cluster can bias relative impact downward.

        It raises :class:`ValueError` for an undeclared metric, no enrolled
        arm, an unknown requested arm, or alpha outside the open unit interval.
        """
        if population not in ("assigned", "triggered"):
            _refuse(_INVALID_POPULATION, population=population)
        experiment = getattr(self, "_experiment", None)
        from increment._sitewide import SitewideReadouts, SitewideRequest

        return SitewideReadouts(src=self._src, experiment=experiment).run(
            SitewideRequest(
                metric_name=metric,
                arm=arm,
                alpha=alpha,
                population=population,
            )
        )

    def export(self, path: str | Path) -> None:
        """Write this experiment's moments (per-metric ``group_summary``
        rows, unioned) to a parquet file at *path*.

        Native, frame, and artifact sources export total moments and
        design-level compliance state. :meth:`from_moments` reloads them. Every row carries a ``moments_format`` version
        stamp; :meth:`from_moments` validates and strips it, and
        refuses a file written in a future wire format instead of
        misreading it.

        A fixed-horizon export refuses a quantile metric on every source with
        ``source.frame.quantile_no_moments`` (``readout.observational.quantile``
        under an observational design), from the metric catalog alone before any
        evidence is read: a moments cube holds no per-unit values. A registered
        sequential checkpoint holds unit-record proofs, not moments rows, so it
        still exports when a catalog quantile is not among its registered models.
        """
        from increment._source_workflows import _export_moments_binding

        _ = self._src
        _export_moments_binding(self._binding, path)

    def _readout_source(
        self,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> _BuiltinMomentSource:
        """Validate a requested population; the evidence request selects it."""
        if population == "triggered":
            if is_custom_source(self._src):
                from increment._evidence_refusals import require_population

                require_population(self._src, population)
            else:
                _require_analysis_operation(
                    self._src,
                    "triggered_source",
                    message=(
                        "A triggered readout requires a source retaining the raw event stream "
                        "and its declared trigger."
                    ),
                )
        return self._src

    # Properties

    @property
    def experiment(self) -> Experiment:
        """The resolved experiment definition."""
        from increment._source_workflows import _experiment_definition_binding

        return _experiment_definition_binding(self._binding)

    @property
    def artifact_context(self) -> ArtifactContext:
        """The trusted artifact context for this definitions-backed analysis.

        Pass this value as ``expected_context`` when reopening an artifact
        published from the same analysis.
        """
        from increment._source_workflows import _artifact_context_binding

        return _artifact_context_binding(self._binding)

    @property
    def metrics(self) -> list[Metric]:
        """All metrics and guardrails for this experiment."""
        from increment._source_workflows import _metrics_binding

        return _metrics_binding(self._binding)

    @property
    def compiled_decision_plan(self) -> CompiledDecisionPlan:
        """The immutable decision plan this analysis will execute."""
        from increment._source_workflows import _compiled_decision_plan_binding

        return _compiled_decision_plan_binding(self._binding)

    @property
    def available_metrics(self) -> list[Metric]:
        """Saved metrics not declared by the experiment, in definitions order."""
        from increment._source_workflows import _available_metrics_binding

        return _available_metrics_binding(self._binding)

    def _exploratory_metrics(self, names: Sequence[str] | None, *, caller: str) -> list[Metric]:
        """Resolve ``exploratory_metrics=`` before any query; ``[]`` when none were named."""
        if not names:
            return []
        if not isinstance(self._state, DefinitionsArmAnalysisState):
            _refuse(_EXPLORATORY_SOURCE_LIMITED, method=caller)
        added = select_exploratory_metrics(
            self.available_metrics,
            {*self._experiment.metric_names, *(metric.name for metric in self._metrics)},
            names,
            caller=caller,
        )
        inference = self._plan.inference
        if getattr(inference, "registration", None) is not None:
            _refuse(
                _EXPLORATORY_SEQUENTIAL,
                method=caller,
                names=[metric.name for metric in added],
                inference=type(inference).__name__,
            )
        return added

    def _exploratory_analysis(self, added: Sequence[Metric], *, caller: str) -> Self:
        """This analysis reading from a sibling source that also carries *added*."""
        self._require_arm_state(caller)
        src = _require_analysis_operation(
            self._src,
            "exploratory_source",
            ExploratorySourceOperation,
            message=f"{caller}(exploratory_metrics=...) needs a native Analysis.from_definitions instance.",
        )
        derived = copy.copy(self)
        derived._binding = self._binding.with_source(src.exploratory_source(metrics=added))
        context = _current_execution_context()
        if context is not None:
            context.declare_source_alias(derived._src, parent=self._src)
        return derived

    # Caching and materialization

    # SQL introspection

    def build_panel_for_metric(
        self,
        exposures: Table,
        metric: Metric,
        *,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> Any:
        from increment.query._workflows import _build_panel_for_metric_binding

        return _build_panel_for_metric_binding(
            self._binding, exposures, metric, population=population
        )

    def panel_sql(self, breakouts: list[Breakout] | None = None) -> dict[str, str]:
        from increment.query._workflows import _panel_sql_binding

        return _panel_sql_binding(self._binding, breakouts)

    def summary_sql(self, breakouts: list[Breakout] | None = None) -> dict[str, str]:
        from increment.query._workflows import _summary_sql_binding

        return _summary_sql_binding(self._binding, breakouts)

    def planning_baseline(self, metric: str) -> Baseline | SwitchbackBaseline:
        """The planning input a power solver needs, computed from THIS
        analysis's own data and declared design -- i.e. a prior or pilot
        experiment (or an in-flight experiment's pre-treatment data), not
        the future test you are about to plan. If you have no prior
        experiment or pilot data, build a Baseline directly instead
        (Baseline(mean=, var=), Baseline.from_proportion,
        Baseline.from_absorption, Baseline.from_ratio).

        An arm analysis returns a :class:`~increment.power.Baseline` for
        ``required_sample_size``/``achieved_power``/
        ``minimum_detectable_effect``. Outcome moments describe the analyzed
        population: under a declared trigger, the triggered population, with
        ``trigger_rate`` its observed share of the assigned control arm;
        with no trigger declared, the assigned population and
        ``trigger_rate=1.0``. From that population's control arm: mean/var
        (a ratio metric's linearized moments; a quantile metric returns a
        ``QuantileBaseline`` over its per-unit control values); cuped_rho,
        when the metric declares CUPED, as the control arm's variance
        reduction under the runtime's own CUPED fit; compliance from the
        arms' uptake difference when the design is Encouragement; and
        cluster_icc/cluster_size_cv from contributing clusters, with ICC on
        the analyzed score. avg_cluster_size retains assigned units per
        randomized cluster; cluster_participation is the pilot fraction of
        assigned clusters contributing analyzed units.

        A switchback analysis (``from_switchback_panel``) returns the
        pilot-fitted :class:`~increment.power.switchback.SwitchbackBaseline`
        for the ``switchback_*`` solvers.

        Refuses by SOURCE, naming which, where the constructor cannot
        supply what the metric type or declared design needs -- including
        a declared trigger whose evidence the source does not carry."""
        from increment._planning_baseline import planning_baseline

        if self._state.family == "contrast":
            context = cast("ContrastContext", self._context)
            spec = select_metrics(context.metrics, [metric], caller="planning_baseline")[0]
            return cast("ContrastSource", self._src).planning_baseline(spec)
        experiment = getattr(self, "_experiment", None)
        trigger = getattr(experiment, "trigger", None)
        if trigger is None:
            from increment._source_types import is_custom_source

            if is_custom_source(self._src):
                trigger = getattr(self._context, "trigger_name", None)
        return planning_baseline(
            cast("_BuiltinMomentSource", self._src),
            metric,
            trigger=trigger,
            trigger_rates=self.trigger_rates,
        )

    def capture_sequential(
        self, *, finalized: bool, previous=None, as_of=None, freeze: Sequence[str] = ()
    ):
        """Capture a registered finalized prefix; freshness is not finalization.

        Pass a previous snapshot to verify append-only continuation. Corrections
        to any retained unit outcome, assignment or definition refuse the path.
        Panel and relational sources require an explicit ``as_of`` horizon.

        ``freeze`` names metrics to stop monitoring at exactly this capture:
        every registered cell of each named metric (each arm and segment) that
        has data keeps this capture's evidence at every later look; see
        ``declare_sequential_freeze``. A frozen secondary still takes part in
        its family, whose e-BH selection is redone at each look over frozen and
        current evidence, so its discovery and selected interval can change.

        A triggered family guarantee requires stopping and freezing decisions
        based only on already-revealed triggered-chain information. Assigned
        results can reveal future triggered outcomes, including the same
        metric, and must not inform those decisions.
        """
        from increment.readouts._evidence import load_sequential_capture

        source = cast(Any, self._state.source)
        request = SequentialCaptureRequest(
            finalized=finalized, previous=previous, as_of=as_of, freeze=tuple(freeze)
        )
        context = _current_execution_context()
        if context is None:
            return load_sequential_capture(source, request)

        from increment._source_types import is_custom_source
        from increment.readouts._evidence import _builtin_source

        context.validate_request(
            source, tuple(source.context.metrics), tuple(source.context.configs)
        )
        with context.borrow() as scoped:
            capture_source = scoped if is_custom_source(source) else _builtin_source(scoped)
            snapshot = load_sequential_capture(capture_source, request)
        if not is_custom_source(source):
            from increment.sequential_source import adopt_source_snapshot

            snapshot = adopt_source_snapshot(source, snapshot)
        return snapshot

    def sequential_snapshot(self, *, previous=None):
        """Retrieve a current exact checkpoint, optionally proving its parent."""
        from increment.readouts._evidence import current_source_snapshot

        source = cast(Any, self._state.source)
        context = _current_execution_context()
        if context is None:
            return current_source_snapshot(source, previous=previous)

        from increment._source_types import is_custom_source
        from increment.readouts._evidence import _builtin_source

        context.validate_request(
            source, tuple(source.context.metrics), tuple(source.context.configs)
        )
        with context.borrow() as scoped:
            snapshot_source = scoped if is_custom_source(source) else _builtin_source(scoped)
            snapshot = current_source_snapshot(snapshot_source, previous=previous)
        if not is_custom_source(source):
            from increment.sequential_source import adopt_source_snapshot

            snapshot = adopt_source_snapshot(source, snapshot)
        return snapshot

    def __repr__(self) -> str:
        study = getattr(self._state, "study", None)
        design = None if study is None else study.identification
        details = [f"  experiment={self._state.experiment_name!r}"]
        if design is not None:
            details.append(f"  design={type(design).__name__}")
        return "Analysis(\n" + ",\n".join(details) + "\n)"

    __str__ = __repr__

    def __repr_args__(self) -> list[tuple[str, object]]:
        study = getattr(self._state, "study", None)
        args: list[tuple[str, object]] = [("experiment", self._state.experiment_name)]
        if study is not None:
            args.append(("design", type(study.identification).__name__))
        return args

    def _persistence_identity(
        self,
        results: Any,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        extra_metrics: Sequence[Metric] = (),
        runtime_request: Mapping[str, object] | None = None,
    ):
        from increment.kernel.contracts import Registration, contract_hash
        from increment.kernel.ports import program_store_operations

        persistence_identity = program_store_operations().PersistenceIdentity

        design = cast(Any, self._design)
        if isinstance(self._state, ContrastAnalysisState):
            procedures = cast("ContrastContext", self._context).procedures
            registration_hash = Registration(
                registration_hash=contract_hash(
                    "registration",
                    {
                        "design": design.model_dump(mode="json"),
                        "procedures": {
                            name: procedure.model_dump(mode="json")
                            for name, procedure in sorted(procedures.items())
                        },
                    },
                )
            ).hash()
        else:
            procedures = self._plan.procedures
            registration_hash = Registration.from_plan(self._plan).hash()
        cells = results.as_cells()
        cell_metrics = {cell.identity.key.metric for cell in cells}
        metrics = {metric.name: metric for metric in (*self._metrics, *extra_metrics)}
        if "uptake" in cell_metrics and "uptake" not in metrics:
            from increment.semantics.models import MeanMetric

            uptake = getattr(design, "uptake", None)
            metrics["uptake"] = MeanMetric(
                name="uptake",
                entity="__uptake__",
                fact="__uptake__",
                aggregation="avg_event",
                window_days=getattr(uptake, "window_days", None),
            )
        methods: dict[tuple[str, str, str], Method] = {}
        source_refs: dict[tuple[str, str], str] = {}
        method_overrides: list[Method] = []
        if isinstance(decision_method, Method):
            method_overrides.append(decision_method)
        if sensitivity_methods is not UNSET:
            method_overrides.extend(cast(Sequence[Method], sensitivity_methods))
        from increment._canonical import declaration_value

        source_context = getattr(self._src, "context", None)
        declaration_identity = getattr(self._src, "declaration_identity", None)
        if callable(declaration_identity):
            declaration = declaration_identity()
            declaration_unavailable_reason = None
        else:
            declaration = {"context": declaration_value(source_context)}
            declaration_unavailable_reason = "source_declaration_identity_unavailable"
        source_declaration_hash = contract_hash(
            "source_declaration_v1", declaration_value(declaration)
        )
        if runtime_request:
            registration_hash = contract_hash(
                "analysis_runtime_request_v1",
                {
                    "registration_hash": registration_hash,
                    "request": declaration_value(runtime_request),
                },
            )
        for cell in cells:
            key = cell.identity.key
            estimator = key.method or "unadjusted"
            role = key.method_role or "decision"
            procedure = procedures.get(key.metric)
            declared_methods: tuple[Method | None, ...] = (
                ()
                if procedure is None
                else (
                    getattr(procedure, "decision_method", None),
                    *getattr(procedure, "sensitivity_methods", ()),
                )
            )
            selected = next(
                (
                    method
                    for method in (*method_overrides, *declared_methods)
                    if isinstance(method, Method) and method.name == estimator
                ),
                None,
            )
            if selected is None:
                selected = Method.cuped() if estimator == "cuped" else Method(name=estimator)
            methods[(key.metric, estimator, role)] = selected
            source_refs[(key.analysis_population, key.metric)] = source_declaration_hash
        return persistence_identity(
            design=design,
            metrics=metrics,
            registration_hash=registration_hash,
            methods=methods,
            source_refs=source_refs,
            source_declaration_identity_unavailable_reason=declaration_unavailable_reason,
        )

    def run(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        value_scale: Mapping[str, ValueScale] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        coherence: Literal["captured", "live"] = "captured",
        store: Any | None = None,
        _execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates | ContrastResults:
        """Run the analysis through its immutable experiment and bound resolver."""
        from increment._analysis_kernel_adapter import run_analysis

        options = {
            "decision_method": decision_method,
            "sensitivity_methods": sensitivity_methods,
            "prior": prior,
            "metrics": metrics,
            "estimands": estimands,
            "value_scale": value_scale,
            "exploratory_metrics": exploratory_metrics,
            "population": population,
            "coherence": coherence,
            "store": store,
            "_execution_context": _execution_context,
        }
        if _execution_context is None:
            return cast("LiftEstimates | ContrastResults", run_analysis(self, **options))
        with _execution_context_scope(_execution_context):
            return cast("LiftEstimates | ContrastResults", run_analysis(self, **options))

    def _run_legacy(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        value_scale: Mapping[str, ValueScale] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        coherence: Literal["captured", "live"] = "captured",
        store: Any | None = None,
        _execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates | ContrastResults:
        """Run the full A/B test analysis pipeline.

        Epistemic policy defaults come from the source's `AnalysisPlan` and
        per-metric `ExperimentMetric` bindings; the parameters below override
        them for this call only.

        Parameters
        ----------
        decision_method : Method | _Unset
            Decision estimator override; UNSET inherits each metric's
            declaration. An explicit override replaces a declared,
            non-default value for that run.
        sensitivity_methods : Sequence[Method] | _Unset
            Additional estimator overrides; UNSET inherits each declaration.
            An explicit override replaces a declared, non-default value for
            that run.
        prior : Prior | None | _Unset
            Informative prior override; UNSET inherits each declaration, None
            resets it. An explicit override replaces a declared, non-default
            value for that run. A mixture prior is refused on
            observational/encouragement rows - pass ``Normal``.
        metrics : Sequence[str | Metric] | None
            Narrow the estimated metrics. ``None`` (default) selects
            every declared metric.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: among ``"itt"``/
            ``"compliance"``/``"late"``, default all three.
        value_scale : Mapping[str, ValueScale] | None
            Observational-only: report a metric's rows as additive ATE
            instead of relative lift.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to estimate in addition to the declared
            ones (``from_definitions`` only). Their rows carry ``role="exploratory"``
            under the default unassigned procedure (two-sided, at the plan's full
            ``alpha``), join no plan family, and leave every declared row unchanged.
            ``metrics=[]`` with ``exploratory_metrics`` returns only the added rows.
        population : Literal["assigned", "triggered"] | None
            Source population for the arm readout. When omitted, return each captured population
            as separate result rows; an explicit value selects only that population. Triggered
            reads require a declared trigger and sufficient source evidence.
        coherence : Literal["captured", "live"]
            Arm-readout source consistency. ``"captured"`` (default) requests one
            source-owned snapshot; warehouse sources require TEMP materialization
            and refuse if it is unavailable. ``"live"`` opts out of that guarantee
            and marks results as live-qualified.
        store : Any | None
            Optional result store. When provided, persist the complete run
            and expose its ``run_id`` and ``analysis_spec_hash`` in result metadata.
        _execution_context : EvidenceExecutionContext | None
            Internal run owner used when this readout is composed with other operations.

        Returns
        -------
        LiftEstimates | ContrastResults
            Arm evidence returns one estimate per (metric x method x
            non-control arm). Switchback evidence returns one contrast per
            selected metric. Switchback calls accept only UNSET role/prior overrides;
            ``estimands`` and ``value_scale`` remain arm-only.
            Each arm estimate carries an explicit ``analysis_population`` axis
            (``assigned`` or ``triggered``); the two populations are distinct
            result identities.
        """

        if _execution_context is None:
            _execution_context = _current_execution_context()
        from increment.breakout.estimates import LiftEstimates

        if isinstance(self._state, ContrastAnalysisState) and population == "triggered":
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run",
                experiment="<unknown>",
                trigger=None,
                route="switchback",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        added = self._exploratory_metrics(exploratory_metrics, caller="run")
        runtime_request: dict[str, object] = {}
        if prior is not UNSET:
            runtime_request["prior"] = prior
        if estimands is not None:
            runtime_request["estimands"] = tuple(estimands)
        if value_scale is not None:
            runtime_request["value_scale"] = value_scale
        if isinstance(self._state, ContrastAnalysisState):
            rows = self._run_contrast(
                decision_method,
                sensitivity_methods,
                prior,
                metrics,
                estimands,
                value_scale,
                execution_context=_execution_context,
            )
            rows._default_backend = self._input_backend
            if store is not None:
                from uuid import uuid4

                store.persist_results(
                    rows,
                    self._state.experiment_name,
                    uuid4().hex,
                    self._persistence_identity(
                        rows, decision_method, sensitivity_methods, runtime_request=runtime_request
                    ),
                )
            return rows
        selected = select_metrics(cast("Sequence[Metric]", self._metrics), metrics, caller="run")

        def request(chosen: Sequence[Metric]) -> WholeWindowRequest:
            return WholeWindowRequest(
                metrics=tuple(chosen),
                estimands=tuple(estimands) if estimands is not None else None,
                value_scale=value_scale,
                population=population,
                coherence=coherence,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )

        reader = self._whole_window()
        if added:
            exploratory = self._exploratory_analysis(added, caller="run")
            rows = reader.run_family(
                WholeWindowFamilyRequest(
                    declared=request(selected),
                    exploratory=WholeWindowExploratoryRequest(
                        request=request(added),
                        reader=exploratory._whole_window(),
                    ),
                ),
                execution_context=_execution_context,
            )
        else:
            rows = reader.run(request(selected), execution_context=_execution_context)
        if not isinstance(rows, LiftEstimates):
            rows = LiftEstimates(rows)
        self._warn_run_advisories(
            selected,
            added,
            estimands=estimands,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
        if not added and _execution_context is not None and rows.metadata is not None:
            metadata = rows.metadata
            rows._metadata = type(metadata).model_validate(
                {
                    **metadata.__dict__,
                    "coherence": (
                        None
                        if "provider" in _execution_context.effective_modes.values()
                        else _execution_context.coherence
                    ),
                    "effective_modes": dict(_execution_context.effective_modes),
                }
            )
        rows._default_backend = getattr(self, "_input_backend", None)
        if store is not None:
            from uuid import uuid4

            store.persist_results(
                rows,
                self._state.experiment_name,
                uuid4().hex,
                self._persistence_identity(
                    rows,
                    decision_method,
                    sensitivity_methods,
                    added,
                    runtime_request=runtime_request,
                ),
            )
        return rows

    def _warn_run_advisories(
        self,
        selected: Sequence[Metric],
        added: Sequence[Metric],
        *,
        estimands: Sequence[str] | None,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
    ) -> None:
        from increment._frame_validation import _warn

        state = self._state
        assert not isinstance(state, ContrastAnalysisState)
        source = state.source

        # A compliance-only encouragement read uses no outcome configuration.
        if (
            estimands is not None
            and not {"itt", "late"}.intersection(estimands)
            and getattr(source.context.design, "mechanism", None) == "encouragement"
        ):
            return
        selected_names = {metric.name for metric in selected}
        declared_configs = {
            config.metric.name: config
            for config in source.context.configs
            if config.metric.name in selected_names
        }
        default_decision = (
            Method(name="iptw")
            if getattr(source.context.design, "mechanism", None) == "observational"
            else Method(name="unadjusted")
        )
        conflicts: dict[str, list[tuple[str, object]]] = {
            "decision_method": [],
            "sensitivity_methods": [],
            "prior": [],
        }
        for name, config in declared_configs.items():
            if (
                decision_method is not UNSET
                and not config.decision_defaulted
                and config.decision_method != default_decision
                and config.decision_method != decision_method
            ):
                conflicts["decision_method"].append((name, config.decision_method))
            if (
                sensitivity_methods is not UNSET
                and config.sensitivity_methods
                and tuple(config.sensitivity_methods)
                != tuple(cast(Sequence[Method], sensitivity_methods))
            ):
                conflicts["sensitivity_methods"].append((name, tuple(config.sensitivity_methods)))
            if (
                prior is not UNSET
                and config.prior is not None
                and not config.prior_is_global
                and config.prior != prior
            ):
                conflicts["prior"].append((name, config.prior))

        overrides = {
            "decision_method": decision_method,
            "sensitivity_methods": (
                tuple(cast(Sequence[Method], sensitivity_methods))
                if sensitivity_methods is not UNSET
                else sensitivity_methods
            ),
            "prior": prior,
        }
        for setting, declarations in conflicts.items():
            if declarations:
                _warn(
                    "analysis.run.override_replaces_declared",
                    setting=setting,
                    metrics=tuple(name for name, _ in declarations),
                    declared=tuple(declarations),
                    override=overrides[setting],
                )

        config_sources: list[tuple[Any, _BuiltinMomentSource]] = []
        if selected:
            config_sources.extend(
                (config, state.source)
                for config in overlay_configs(
                    selected,
                    state.source.context.configs,
                    methods=None,
                    prior=None,
                    prior_override=prior,
                    decision_method=decision_method,
                    sensitivity_methods=sensitivity_methods,
                )
            )
        if added:
            exploratory = self._exploratory_analysis(added, caller="run")
            exploratory_source = cast("_BuiltinMomentSource", exploratory._state.source)
            config_sources.extend(
                (config, exploratory_source)
                for config in overlay_configs(
                    added,
                    exploratory_source.context.configs,
                    methods=None,
                    prior=None,
                    prior_override=prior,
                    decision_method=decision_method,
                    sensitivity_methods=sensitivity_methods,
                )
            )
        unused: list[str] = []
        for config, source in config_sources:
            covariate_available = config.covariate_available
            design = source.context.design
            if (
                not covariate_available
                or not config.decision_defaulted
                or config.methods_explicitly_empty
                or getattr(design, "mechanism", None) == "observational"
                or any(
                    method.variance_reduction == "cuped" for method in config.sensitivity_methods
                )
                or not _cuped_recommendation_admitted(source, config)
            ):
                continue
            unused.append(config.metric.name)
        if unused:
            _warn(
                "analysis.cuped.covariate_unused",
                metric_names=tuple(unused),
                route=(
                    "Request Method(name='cuped', variance_reduction='cuped') "
                    "as decision_method or a sensitivity method in run()."
                ),
            )

    def _run_contrast(
        self,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        metrics: Sequence[str | Metric] | None,
        estimands: Sequence[str] | None,
        value_scale: Mapping[str, ValueScale] | None,
        *,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> ContrastResults:
        validate_contrast_request(
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            estimands=estimands,
            value_scale=value_scale,
        )
        context = cast("ContrastContext", self._context)
        selected = select_metrics(cast("Sequence[Metric]", context.metrics), metrics, caller="run")
        request = ContrastReadoutRequest(
            metrics=tuple(selected),
            procedures={
                metric.name: context.procedures[metric.name]
                for metric in selected
                if metric.name in context.procedures
            },
        )
        source = cast("ContrastSource", self._src)
        return ContrastEvidenceOperation().run(
            source,
            request,
            execution_context=execution_context,
        )

    def _run_scoped_readouts(self, *, coherence: Literal["captured", "live"], **options: Any):
        from increment.readouts._line2 import run_scoped_line2

        return run_scoped_line2(self, coherence=coherence, **options)

    def _whole_window(self) -> WholeWindowReadouts:
        self._require_arm_state("run")

        return WholeWindowReadouts(
            src=cast("_BuiltinMomentSource", self._src),
            plan=self._plan,
            design=self._design,
            experiment=getattr(self, "_experiment", None),
            readout_source=self._readout_source,
            validate_trigger=self._validate_trigger_fires_in_every_arm,
            arm_moments=readouts.arm_moments,
            execution_context_scope=self._scoped_execution_context,
            registered_run=readouts.run,
        )

    def estimate_cate(
        self,
        metric: str,
        *,
        control: str | None = None,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        alpha: float = 0.05,
        ard: bool = False,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
    ) -> CateResult:
        """Fit CATE using this analysis's unit source and declared intervention."""
        from increment.kernel.ports import cate_operations

        operations = cate_operations()

        if control is None:
            assert self._design is not None
            control = self._design.control_group
        return operations.estimate_cate(
            self._cate_source("estimate_cate"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            alpha=alpha,
            ard=ard,
            cluster_weight=cluster_weight,
        )

    def validate_cate(
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        n_groups: int = 5,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        include_evaluation_population: bool = False,
    ) -> CateValidation:
        """Validate CATE on an honest, cluster-atomic holdout.

        Member counts and independent cluster counts are reported separately;
        unavailable uncertainty retains its reason and numeric nulls.
        """
        from increment.kernel.ports import cate_operations

        operations = cate_operations()

        return operations.validate_cate(
            self._cate_source("validate_cate"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            n_groups=n_groups,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            include_evaluation_population=include_evaluation_population,
        )

    def targeting_rule(
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        fraction: float,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
        include_evaluation_population: bool = False,
    ) -> TargetingRule:
        """Evaluate a precommitted deployment fraction on an honest holdout."""
        from increment.kernel.ports import cate_operations

        operations = cate_operations()

        return operations.targeting_rule(
            self._cate_source("targeting_rule"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            fraction=fraction,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            deploy_grain=deploy_grain,
            include_evaluation_population=include_evaluation_population,
        )

    def select_targeting_rule(  # noqa: PLR0913
        self,
        metric: str,
        *,
        control: str,
        interact: Sequence[Covariate | str],
        adjust: Sequence[Covariate | str] = (),
        fractions: Sequence[float],
        cost_per_treated: float = 0.0,
        n_folds: int = 5,
        seed: int,
        alpha: float = 0.05,
        cluster_weight: Literal["member_count", "equal"] = "member_count",
        bootstrap: ClusterBootstrap | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
        include_evaluation_population: bool = False,
    ) -> TargetingSelection:
        """Select a deployment budget on inner folds, then report the untouched holdout."""
        from increment.kernel.ports import cate_operations

        operations = cate_operations()

        return operations.select_targeting_rule(
            self._cate_source("select_targeting_rule"),
            metric,
            control=control,
            interact=interact,
            adjust=adjust,
            fractions=fractions,
            cost_per_treated=cost_per_treated,
            n_folds=n_folds,
            seed=seed,
            alpha=alpha,
            cluster_weight=cluster_weight,
            bootstrap=bootstrap,
            deploy_grain=deploy_grain,
            include_evaluation_population=include_evaluation_population,
        )

    def assignment_diagnostic(self) -> SwitchbackAssignmentDiagnostic:
        """Return switchback assignment integrity evidence."""
        from increment._source_diagnostics import _assignment_diagnostic_binding

        return _assignment_diagnostic_binding(self._binding)

    def breakout_summaries(
        self, *, metrics: Sequence[str | Metric] | None = None
    ) -> dict[str, dict[str, pa.Table]]:
        from increment._source_workflows import _breakout_summaries_binding

        _ = self._src
        return _breakout_summaries_binding(self._binding, metrics)

    def factor_summaries(self) -> dict[str, pa.Table]:
        from increment._source_workflows import _factor_summaries_binding

        _ = self._src
        return _factor_summaries_binding(self._binding)

    def run_breakout(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] | None = None,
    ) -> BreakoutEstimates:
        """Per-segment lift estimates for every declared breakout.

        For each declared :class:`Breakout`, builds a moments source
        scoped to that breakout's totals-grain moments and hands it to
        :func:`increment.readouts.breakout`, which calls
        :func:`~increment.estimation.engine.estimate_lift` once per
        segment. Each breakout resolves its own :class:`FactSource`
        independently, so two breakouts sharing a ``property`` name
        but resolving to different sources are never conflated.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets.
        metrics : Sequence[str | Metric] | None
            Narrow the estimated metrics. ``None`` selects every declared metric.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to break out in addition to the declared
            ones (``from_definitions`` only). Their rows equal those of the same metric
            declared in a plan of its own: the plan's breakout multiplicity applies to
            the added metrics' cells as one family of their own, so no declared row
            changes. ``metrics=[]`` with ``exploratory_metrics`` returns only the added rows.
        population : Literal["assigned", "triggered"] | None
            Select one source population; ``None`` returns captured populations
            as separate result rows. Triggered reads require declared trigger
            evidence.
        Notes
        -----
        Alpha, inference, and breakout multiplicity are read from the
        resolved plan attached to this analysis. Sensitivity analyses
        require constructing another source with another plan. Every
        declared breakout is always returned (no per-call dimension
        filter); an observational panel refuses this method by name.

        Returns
        -------
        BreakoutEstimates
            One row per (breakout x metric x method x non-control arm
            x segment); encouragement paths warn and skip control-free
            segments instead. ``source`` names the resolved
            :class:`FactSource`. Empty when no breakouts are declared.
        """
        execution_context = _current_execution_context()
        if execution_context is None:
            self._require_arm_state(
                "run_breakout", population=population if population is not None else "assigned"
            )
        rows = self._run_breakout(
            _execution_context=execution_context,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            population=population,
        )
        rows._default_backend = self._input_backend
        return rows

    def _run_breakout(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        correction: Correction | None = None,
        population: Literal["assigned", "triggered"] | None = None,
        _execution_context=None,
    ) -> BreakoutEstimates:
        from increment.breakout.estimates import BreakoutEstimates

        state = self._state
        experiment = (
            state.experiment
            if isinstance(state, DefinitionsArmAnalysisState)
            else getattr(state.source, "artifact_experiment", None)
            if state.family == "arm_moments"
            else None
        )
        if population == "triggered" and experiment is not None and experiment.trigger is None:
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run_breakout",
                experiment=experiment.name,
                trigger=None,
                route="native",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        if experiment is not None and not experiment.breakouts:
            select_metrics(
                cast("Sequence[Metric]", self._src.context.metrics),
                metrics,
                caller="run_breakout",
            )
            if exploratory_metrics:
                self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
            return BreakoutEstimates([])
        populations = (
            (population,)
            if population is not None
            else (
                ("assigned", "triggered")
                if experiment is not None and experiment.trigger is not None
                else ("assigned",)
            )
        )
        if "triggered" in populations:
            if isinstance(self._state, ContrastAnalysisState):
                _refuse(
                    _TRIGGER_UNSUPPORTED,
                    method="run_breakout",
                    experiment="<unknown>",
                    trigger=None,
                    route="switchback",
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            reader = BreakoutReadouts(src=self._src, experiment=experiment)
            registration = getattr(self._plan.inference, "registration", None)
            if registration is None:

                def validate_candidate(candidate, selected_metrics):
                    if not selected_metrics:
                        return
                    breakout_policy = candidate._plan.view_policies.for_view(
                        "breakout",
                        mechanism=getattr(getattr(candidate, "_design", None), "mechanism", None),
                        segmented=True,
                    )
                    candidate_reader = BreakoutReadouts(
                        src=candidate._src,
                        experiment=getattr(candidate, "_experiment", None),
                    )
                    candidate_reader.validate_request(
                        BreakoutRequest(
                            metrics=tuple(selected_metrics),
                            decision_method=decision_method,
                            sensitivity_methods=sensitivity_methods,
                            prior=prior,
                            correction=correction
                            or normalize_display_correction(breakout_policy.correction),
                            q=(
                                breakout_policy.q
                                if breakout_policy.q is not None
                                else candidate._plan.q
                            ),
                            population="triggered",
                        )
                    )

                selected = select_metrics(
                    cast("Sequence[Metric]", self._src.context.metrics),
                    metrics,
                    caller="run_breakout",
                )
                validate_candidate(self, selected)
                added = self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
                if added:
                    exploratory = self._exploratory_analysis(added, caller="run_breakout")
                    validate_candidate(exploratory, added)
            reader.validate_populations(populations)
        combined: BreakoutEstimates | None = None
        for population in populations:
            rows = self._run_breakout_population(
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                metrics=metrics,
                exploratory_metrics=exploratory_metrics,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
            )
            combined = rows if combined is None else combined.concat(rows)
        return BreakoutEstimates([]) if combined is None else combined

    def _run_breakout_population(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        correction: Correction | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        _execution_context=None,
    ) -> BreakoutEstimates:
        """:meth:`run_breakout`, optionally under an explicit *correction* in place of the
        plan's breakout multiplicity (``"none"`` leaves every cell standing alone)."""
        from increment.breakout.estimates import BreakoutEstimates

        added = self._exploratory_metrics(exploratory_metrics, caller="run_breakout")
        if isinstance(self._state, ContrastAnalysisState):
            reject_role_overrides_under_contrast(
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        self._require_arm_state("run_breakout")
        inference = self._plan.inference
        if correction is not None and getattr(inference, "registration", None) is not None:
            _refuse(_UNCORRECTED_SEQUENTIAL, inference=type(inference).__name__)
        breakout_policy = self._plan.view_policies.for_view(
            "breakout",
            mechanism=getattr(getattr(self, "_design", None), "mechanism", None),
            segmented=True,
        )
        effective_correction = correction or normalize_display_correction(
            breakout_policy.correction
        )
        selected = select_metrics(
            cast("Sequence[Metric]", self._src.context.metrics), metrics, caller="run_breakout"
        )
        rows = (
            self._breakout_rows(
                selected,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
            )
            if selected or not added
            else BreakoutEstimates([])
        )
        if added:
            exploratory = self._exploratory_analysis(added, caller="run_breakout")
            exploratory_rows = exploratory._breakout_rows(
                added,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                correction=correction,
                population=population,
                _execution_context=_execution_context,
                _scope_source=self._src,
                _scope_analysis=self,
            )
            rows = exploratory_rows if not rows else rows.concat(exploratory_rows)
        from increment.estimation.multiplicity import row_multiplicity_status

        return BreakoutEstimates(
            [
                row.model_copy(
                    update={
                        "multiplicity_status": row_multiplicity_status(
                            row,
                            correction=effective_correction if row.family_id is not None else None,
                        )
                    }
                )
                for row in rows
            ],
            metadata=rows.metadata,
        )

    def _breakout_rows(
        self,
        selected: Sequence[Metric],
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        correction: Correction | None,
        population: Literal["assigned", "triggered"] = "assigned",
        _execution_context=None,
        _scope_source=None,
        _scope_analysis=None,
    ) -> BreakoutEstimates:
        policy = self._plan.view_policies.for_view(
            "breakout",
            mechanism=getattr(getattr(self, "_design", None), "mechanism", None),
            segmented=True,
        )
        request = BreakoutRequest(
            metrics=tuple(selected),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            correction=correction or normalize_display_correction(policy.correction),
            q=policy.q if policy.q is not None else self._plan.q,
            population=population,
        )
        experiment = getattr(self, "_experiment", None)
        source = getattr(self._src, "summary_breakout_source", None)
        reader_source = source if source is not None else self._src
        reader = BreakoutReadouts(
            src=reader_source,
            experiment=experiment,
            execution_context=_execution_context,
            scope_source=self._src if _scope_source is None else _scope_source,
            analysis=self if _scope_analysis is None else _scope_analysis,
        )
        return reader.run(request)

    def _refuse_clustered_day_axis(self, method: str) -> None:
        """A declared cluster is total-grain only; see `_CLUSTERED_DAY_AXIS`."""
        if self._experiment.cluster is not None:
            _refuse(
                _CLUSTERED_DAY_AXIS,
                method=method,
                experiment=self._experiment.name,
                cluster=self._experiment.cluster,
            )

    def run_daily(
        self,
        *,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyMetricValues:
        """Per-day absolute metric values, optionally broken out by segment.

        The time-series counterpart to :meth:`run`: reduces the same
        unit-day panel with ``daily_group_summary`` instead of
        ``group_summary``. This value-only path does not request pre-period
        covariates. Direct CUPED callers should use
        :meth:`run_daily_lift` with role-aware method overrides; that lift path uses a
        fixed pre-exposure unit covariate ``x`` with daily (or
        retention-cohort) outcome ``y`` when native pre-period moments are
        requested.

        Passing *dimension* switches to the per-day-per-segment view,
        routed through ``breakout_moments()`` for each matching
        ``(breakout, metric)`` pair.

        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Override the declared metrics + guardrails. A bounded-band
            :class:`RetentionMetric` is accepted; an
            unbounded one always raises (see ``Notes``).
        dimension : str | None
            Break each day out by one declared breakout's dimension.
            Must match a declared breakout ``property``.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); ``metrics=[]`` reads only these. A
            dimensioned read keeps its refusal of metrics the experiment does not declare.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or per (metric x day x
            arm x segment) when *dimension* is given. A slice too small
            to estimate (fewer than 2 units, or a non-positive mean)
            remains present with ``value=None`` and an ``unavailable``
            reason instead of being dropped.

        Notes
        -----
        The readout raises :class:`ValueError` for an unbounded retention
        band, an unmatched dimension, or a dimensioned request for an
        undeclared metric. Use :meth:`run_asof` or :meth:`run_asof_lift`
        for unbounded retention monitoring.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_daily")
        self._require_arm_state("run_daily")
        selected = self._select_day_axis_metrics(metrics, caller="run_daily", added=added)
        req = DayAxisRequest(
            caller="run_daily",
            grain="daily",
            metrics=tuple(selected),
            dimension=dimension,
            population=population,
        )
        rows = self._run_day_axis_values(req, _execution_context, coherence=coherence)
        rows._default_backend = self._input_backend
        rows._envelope_view = "daily"
        return rows

    def _daily_lift(self, **kwargs: Any) -> DailyLiftEstimates:
        """Internal day-axis boundary: always supply an effective compiled plan.

        Call-time Metric objects absent from the analysis plan receive locally
        compiled unassigned procedures at the plan's alpha, matching the method/prior
        defaults the facade already resolves for them.
        """
        from increment.breakout.estimates import run_daily_lift as _run_daily_lift_estimates

        plan = with_unassigned_procedures(
            self._plan,
            cast("Sequence[Metric]", kwargs["metrics"]),
            design=getattr(self, "_design", None),
        )
        return _run_daily_lift_estimates(plan=plan, **kwargs)

    def _select_day_axis_metrics(
        self,
        metrics: Sequence[str | Metric] | None,
        *,
        caller: str,
        added: Sequence[Metric] = (),
    ) -> list[Metric]:
        # Only the native route accepts undeclared call-time Metric objects.
        native = day_axis_route(self._src) == NATIVE_ROUTE
        selected = select_metrics(
            self._metrics, metrics, caller=caller, allow_undeclared_objects=native
        )
        if not added:
            return selected
        # Added metrics follow the declared ones; a shared name is a duplicate.
        return select_metrics((), [*selected, *added], caller=caller, allow_undeclared_objects=True)

    @staticmethod
    def _stamp_exploratory(
        rows: DailyLiftEstimates,
        added: Sequence[Metric],
        *,
        correction: Correction | None = None,
    ) -> DailyLiftEstimates:
        """Mark added metrics exploratory under the effective view correction."""
        from increment.breakout.estimates import DailyLiftEstimates

        names = {metric.name for metric in added}
        if not names:
            return rows
        from increment.estimation.multiplicity import stamp_multiplicity_status

        return DailyLiftEstimates(
            stamp_multiplicity_status(
                [
                    row.model_copy(update={"role": "exploratory"}) if row.metric in names else row
                    for row in rows
                ],
                correction=correction,
            ),
            metadata=rows.metadata,
        )

    @staticmethod
    def _scoped_execution_context(context: Any):
        return _execution_context_scope(context)

    def _validate_day_axis_before_context(self, req: DayAxisRequest, execution_context=None):
        if req.population == "triggered" or (
            execution_context is None
            and (req.dimension is not None or not is_custom_source(self._src))
        ):
            from increment._day_axis import validate_day_axis

            validate_day_axis(
                req,
                src=self._src,
                design=self._design,
                plan=self._plan,
                route=day_axis_route(self._src),
                experiment=getattr(self, "_experiment", None),
            )

    def _run_day_axis_values(
        self,
        req: DayAxisRequest,
        execution_context=None,
        *,
        coherence: Literal["captured", "live"] = "captured",
    ):
        self._validate_day_axis_before_context(req, execution_context)
        from increment.readouts._line2 import run_standalone_day_axis_values

        return run_standalone_day_axis_values(self, req, execution_context, coherence=coherence)

    def _day_axis(self, execution_context=None) -> DayAxisReadouts:
        return DayAxisReadouts(
            src=self._src,
            plan=self._plan,
            design=self._design,
            experiment=getattr(self, "_experiment", None),
            daily_lift=self._daily_lift,
            execution_context=execution_context,
        )

    def _preflight_line4_views(
        self,
        *,
        population: Literal["assigned", "triggered"],
        readouts: Sequence[Literal["run", "daily", "daily_lift", "asof", "asof_lift"]],
    ) -> None:
        """Validate line-four day-axis requests before its scope opens."""
        caller_by_name: dict[
            str, Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"]
        ] = {
            "daily": "run_daily",
            "daily_lift": "run_daily_lift",
            "asof": "run_asof",
            "asof_lift": "run_asof_lift",
        }
        day_axis = self._day_axis()
        experiment = getattr(self._state, "experiment", None)
        for name in readouts:
            if name == "run":
                continue
            grain = "daily" if name.startswith("daily") else "asof"
            request = DayAxisRequest(
                caller=caller_by_name[name],
                grain=grain,
                metrics=tuple(self._metrics),
                dimension=None,
                estimands=None,
                population=population,
                experiment=experiment,
            )
            day_axis.validate_request(request, lift=name.endswith("_lift"))

    def run_daily_lift(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        estimands: Sequence[str] | None = None,
        dimension: str | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyLiftEstimates:
        """Per-day relative lift estimates, optionally broken out by segment.

        The time-series counterpart to :meth:`run`: builds each metric's
        per-day moments like :meth:`run_daily`, concatenates them into
        one :func:`increment.breakout.estimates.run_daily_lift` call.
        *dimension* routes through ``breakout_moments()`` per breakout.

        The outcome ``y`` is daily, or retention-cohort, while CUPED's ``x``
        is a fixed pre-exposure unit covariate. Native CUPED is requested
        only when pre-period moments are materialized. Sequential inference
        is refused for disjoint daily/cohort slices; use
        :meth:`run_asof_lift` for cumulative monitoring.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets.
        metrics : Sequence[str | Metric] | None
            Override the declared metrics + guardrails. A bounded-band
            :class:`RetentionMetric` is accepted; an
            unbounded one always raises (see ``Notes``).
        estimands : Sequence[str] | None
            Requested encouragement estimands. Per-day LATE is refused because
            each daily first stage is weakly identified; ``("compliance",)``
            is supported as an assignment-anchored uptake estimate.
        dimension : str | None
            Break each day's lift out by one declared breakout's
            dimension. Must match a declared breakout ``property``.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` selects a unit on its first eligible trigger day;
            outcomes anchor at trigger, while compliance uptake stays anchored
            at assignment. Both require explicit source evidence and are
            supported on definitions and unit-day artifacts.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyLiftEstimates
            One row per (day x metric x method x non-control arm), or
            with a segment axis when *dimension* is given. An
            unestimable slice remains present with ``lift=None`` and an
            ``unavailable`` reason instead of being dropped.

        Notes
        -----
        The readout raises :class:`ValueError` for an unbounded retention
        band, an unmatched dimension, a dimensioned request for an undeclared
        metric, or per-day LATE: each daily first stage is weakly identified.
        It raises :class:`UnsupportedRequestError` for sequential inference
        on disjoint daily or cohort slices, which cannot support monitoring.
        """

        _execution_context = _current_execution_context()
        self._require_arm_state("run_daily_lift", population=population)
        selected = self._select_day_axis_metrics(metrics, caller="run_daily_lift")
        req = DayAxisRequest(
            caller="run_daily_lift",
            grain="daily",
            metrics=tuple(selected),
            dimension=dimension,
            estimands=tuple(estimands) if estimands is not None else None,
            population=population,
        )
        self._validate_day_axis_before_context(req, _execution_context)
        lift_options = None
        if _execution_context is not None:
            lift_options = self._day_axis(_execution_context).resolve_lift_options(
                req.metrics,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        if _execution_context is None:
            from increment.readouts._line2 import run_standalone_day_axis_lift

            rows = run_standalone_day_axis_lift(
                self,
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                coherence=coherence,
            )
        else:
            rows = self._day_axis(_execution_context).lift(
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                options=lift_options,
            )
        rows._default_backend = self._input_backend
        rows._envelope_view = "daily"
        return rows

    def run_asof(
        self,
        *,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyMetricValues:
        """Per-day absolute metric values "as of day N".


        The running-total counterpart to :meth:`run_daily`: reduces the
        same panel with ``asof_group_summary`` instead of
        ``daily_group_summary``, and deliberately skips
        ``window_bound_stats`` - the as-of builder does its own per-unit
        window masking instead of dropping a closed-window row.

        Passing *dimension* switches to the per-day-per-segment as-of
        view, routed through ``breakout_moments(grain="asof")``.

        Parameters
        ----------
        metrics : Sequence[str | Metric] | None
            Override the declared/constructed metric set.
        dimension : str | None
            Break each day's running total out by one declared dimension.
        completed_windows_only : bool
            ``False`` (default) is the provisional monitoring series: a
            unit is admitted as soon as it is post-exposure, zero-filled
            until it has a real observation; a retention metric is
            admitted at band open. ``True`` gates on full maturity
            instead: a bounded mean/conversion/ratio metric is admitted
            only once ``ds >= first_exposure_date + window_days``; under
            :class:`Encouragement`, both the outcome and uptake windows
            must be bounded and the gate is the LATER of the two right
            edges (raises otherwise); a retention metric is admitted at
            band close rather than band open. An unbounded (no
            ``window_days``) mean, conversion, or ratio metric with no
            uptake is unaffected, since it has no right edge to wait
            out; combined with a declared uptake it raises instead
            (uptake makes both windows required). Also raises for an
            unbounded retention band, which has no closing day to gate
            on either.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); ``metrics=[]`` reads only these. A
            dimensioned read keeps its refusal of metrics the experiment does not declare.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.

        Returns
        -------
        DailyMetricValues
            One row per (metric x day x arm), or with a segment axis
            when *dimension* is given. An unestimable slice remains
            present with ``value=None`` and an ``unavailable`` reason.
            A unit's contribution freezes at its last in-window value, so
            this view's final day and :meth:`run`'s whole-window estimate can legitimately
            disagree for an experiment with late enrollees.

        Notes
        -----
        The readout raises :class:`ValueError` for an unmatched dimension,
        or completed-window retention with an unbounded band.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_asof")
        self._require_arm_state("run_asof")
        selected = self._select_day_axis_metrics(metrics, caller="run_asof", added=added)
        req = DayAxisRequest(
            caller="run_asof",
            grain="asof",
            metrics=tuple(selected),
            dimension=dimension,
            completed_windows_only=completed_windows_only,
            population=population,
        )
        rows = self._run_day_axis_values(req, _execution_context, coherence=coherence)
        rows._default_backend = self._input_backend
        rows._envelope_completion_policy = completed_windows_only
        rows._envelope_view = "asof"
        return rows

    def run_asof_lift(
        self,
        *,
        decision_method: Method | _Unset = UNSET,
        sensitivity_methods: Sequence[Method] | _Unset = UNSET,
        prior: Prior | None | _Unset = UNSET,
        metrics: Sequence[str | Metric] | None = None,
        dimension: str | None = None,
        estimands: Sequence[str] | None = None,
        completed_windows_only: bool = False,
        exploratory_metrics: Sequence[str] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        coherence: Literal["captured", "live"] = "captured",
    ) -> DailyLiftEstimates:
        """As-of relative lift history, or a registered sequential checkpoint.

        For fixed-horizon inference, the running-total counterpart to
        :meth:`run_daily_lift`: as-of
        moments (:meth:`run_asof`) concatenated into one
        :func:`increment.breakout.estimates.run_daily_lift` call.

        Registered sequential inference returns only the current labeled,
        finalized checkpoint, not a reconstructed per-day history. Dimensions
        are unsupported here; use the registered breakout checkpoint instead.

        Parameters
        ----------
        decision_method : Method | _Unset
            Explicit decision-method override; UNSET inherits the declaration.
        sensitivity_methods : Sequence[Method] | _Unset
            Explicit sensitivity methods; UNSET inherits the declaration.
        prior : Prior | None | _Unset
            Explicit prior override; UNSET inherits and None resets. Native day-axis
            sources support CUPED for mean, conversion and ratio metrics when their
            fixed pre-period covariate moments are materialized (a ratio metric
            adjusts its numerator and denominator against the numerator's
            pre-period total), and for bounded retention metrics on
            cohort-indexed daily paths.
            As-of retention CUPED is accepted on the calendar axis,
            including dimensioned monitoring and bounded completed-window
            reads; unbounded bands remain monitoring-only. CUPED remains
            refused for quantile metrics, clustered day-axis
            analyses, and dataframe-panel sources.
        metrics : Sequence[str | Metric] | None
            Override the declared/constructed metric set.
        dimension : str | None
            Break each day's fixed-horizon lift out by one declared dimension.
            Unsupported for registered sequential inference.
        estimands : Sequence[str] | None
            Under :class:`Encouragement` only: ``"itt"``/``"compliance"``/``"late"``, default all three.
            A registered sequential checkpoint requested with exactly
            ``("compliance",)`` reads uptake alone and consumes no outcome metric,
            so a retention metric in the catalog does not affect it (see *Notes*).
        completed_windows_only : bool
            For fixed-horizon inference, ``False`` (default) is the provisional gate:
            a unit is admitted as soon as it is post-exposure, zero-filled
            until it has a real observation. ``True`` gates on full
            maturity: a bounded mean/conversion/ratio metric with no
            declared uptake is admitted once its own outcome window has
            closed; under :class:`Encouragement`, both the outcome and
            uptake windows must be bounded and the gate is the LATER of
            the two right edges (raises otherwise). A retention metric
            gates at band close rather than band open. An unbounded
            mean, conversion, or ratio metric with no uptake is
            unaffected; combined with a declared uptake it raises
            instead. Registered inference uses finalized outcomes; ``True``
            remains required under :class:`Encouragement` with
            ``AlwaysValid(registration=...)``.
        exploratory_metrics : Sequence[str] | None
            Names from :attr:`available_metrics` to read in addition to the declared
            metrics (``from_definitions`` only); their rows carry ``role="exploratory"``
            and ``metrics=[]`` reads only these. A registered sequential plan refuses them.
        population : Literal["assigned", "triggered"]
            ``"assigned"`` (default) retains the enrolled population.
            ``"triggered"`` admits each unit on its first eligible trigger day;
            outcomes are strictly after that unit's trigger and require explicit
            source evidence. Supported on definitions and unit-day artifacts only.
        coherence : Literal["captured", "live"]
            ``"captured"`` (default) requests one source-owned snapshot;
            ``"live"`` opts out when a warehouse cannot materialize TEMP tables.
        Notes
        -----
        Alpha, inference, and segmented as-of multiplicity come from
        the resolved plan attached to this analysis. Segmented as-of BH is
        refused, as are observational designs and unsupported
        retention/window combinations.

        The readout raises :class:`ValueError` for completed-window unbounded
        retention, an unmatched dimension, or non-ITT estimands outside an
        encouragement design. It raises :class:`CapabilityError` for retention
        outcomes under encouragement or dimensioned registered inference.
        A registered compliance-only checkpoint consumes no outcome metric,
        so neither retention refusal applies. Segmented as-of BH and
        observational designs raise :class:`UnsupportedRequestError`.

        Returns
        -------
        DailyLiftEstimates
            Fixed-horizon: one row per (day x metric x method x non-control arm),
            optionally per segment. Unestimable slices retain ``lift=None`` and
            an ``unavailable`` reason. Registered sequential inference returns
            only the current checkpoint rows, all with its date label.
        """

        _execution_context = _current_execution_context()
        added = self._exploratory_metrics(exploratory_metrics, caller="run_asof_lift")
        self._require_arm_state("run_asof_lift", population=population)
        selected = self._select_day_axis_metrics(metrics, caller="run_asof_lift", added=added)
        req = DayAxisRequest(
            caller="run_asof_lift",
            grain="asof",
            metrics=tuple(selected),
            dimension=dimension,
            completed_windows_only=completed_windows_only,
            estimands=tuple(estimands) if estimands is not None else None,
            population=population,
        )
        self._validate_day_axis_before_context(req, _execution_context)
        lift_options = None
        if _execution_context is not None:
            lift_options = self._day_axis(_execution_context).resolve_lift_options(
                req.metrics,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
            )
        asof_policy = self._plan.view_policies.for_view(
            "asof",
            mechanism=getattr(self._design, "mechanism", None),
            segmented=dimension is not None,
        )
        if _execution_context is None:
            from increment.readouts._line2 import run_standalone_day_axis_lift

            lift_rows = run_standalone_day_axis_lift(
                self,
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                coherence=coherence,
            )
        else:
            lift_rows = self._day_axis(_execution_context).lift(
                req,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                options=lift_options,
            )
        rows = self._stamp_exploratory(
            lift_rows,
            added,
            correction=normalize_display_correction(asof_policy.correction),
        )
        rows._default_backend = self._input_backend
        rows._envelope_completion_policy = completed_windows_only
        rows._envelope_view = "asof"
        return rows
