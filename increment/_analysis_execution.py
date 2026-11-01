"""Shared analysis execution for bound experiment sources."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
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


from increment._analysis_config import UNSET, _Unset
from increment._binding import SourceBinding, _ArtifactOpenSpec
from increment._breakout_readouts import run_breakout, run_breakout_rows
from increment._dashboard_capture import (
    DashboardBreakoutReads,
    DashboardCapture,
    capture_dashboard,
    dashboard_breakout_reads,
    dashboard_group_data,
)
from increment._day_axis import (
    DayAxisReadouts,
    day_axis_readouts,
    preflight_line4_views,
    require_arm_state,
    run_asof,
    run_asof_lift,
    run_daily,
    run_daily_lift,
)
from increment._evidence_types import SequentialCaptureRequest, SourceContext
from increment._execute import readout_source
from increment._exploratory import exploratory_sibling, select_day_axis_metrics, select_exploratory
from increment._literals import Backend, Correction, ValueScale
from increment._source_operations import DashboardGroupData, DashboardSnapshotPayload
from increment.decision import AnalysisState, ArmAnalysisState, ContrastContext
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.engine import Method
from increment.readouts._execution_context import (
    current_execution_context as _current_execution_context,
)
from increment.readouts._execution_context import (
    execution_context_scope as _execution_context_scope,
)
from increment.semantics.models import Breakout, Definitions, Experiment

__all__ = ["Analysis"]


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

_T = TypeVar("_T")


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
        return cast(
            "ArmAnalysisState",
            require_arm_state(self._open_binding(), method, population=population).state,
        )

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

    def _open_binding(self) -> SourceBinding:
        """This analysis's binding, reopening a closed artifact source first."""
        self._ensure_artifact_source()
        return self._binding

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
        return capture_dashboard(
            self._open_binding(),
            lambda capture: self._with_pinned_source(capture, operation),
            metrics=metrics,
        )

    def _with_pinned_source(self, capture: DashboardCapture, operation: Callable[[Self], _T]) -> _T:
        isolated = copy.copy(self)
        isolated._binding = capture.binding
        isolated._readout_srm_observations = capture.observations
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
        return dashboard_breakout_reads(self._open_binding(), breakout)

    def dashboard_group_data(
        self,
        *,
        metrics: Sequence[Metric],
        checkpoints: Mapping[str, SequentialCheckpoint] | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> tuple[DashboardGroupData, ...]:
        """Return group data for the requested assigned or triggered population."""
        return dashboard_group_data(
            self._open_binding(), metrics=metrics, checkpoints=checkpoints, population=population
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
        from increment._sitewide import _sitewide_binding

        _ = self._src
        return _sitewide_binding(self._binding, metric, arm=arm, alpha=alpha, population=population)

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
        return readout_source(self._open_binding(), population=population)

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
        from increment._exploratory import available_metrics

        return available_metrics(self._binding)

    def _exploratory_metrics(self, names: Sequence[str] | None, *, caller: str) -> list[Metric]:
        """Resolve ``exploratory_metrics=`` before any query; ``[]`` when none were named."""
        return select_exploratory(self._open_binding(), names, caller=caller)

    def _exploratory_analysis(self, added: Sequence[Metric], *, caller: str) -> Self:
        """This analysis reading from a sibling source that also carries *added*."""
        derived = copy.copy(self)
        derived._binding = exploratory_sibling(self._open_binding(), added, caller=caller)
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
        from increment._source_diagnostics import _planning_baseline_binding

        _ = self._src
        return _planning_baseline_binding(self._binding, metric)

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
        from increment.readouts._evidence import retained_sequential_capture

        request = SequentialCaptureRequest(
            finalized=finalized, previous=previous, as_of=as_of, freeze=tuple(freeze)
        )
        return retained_sequential_capture(
            self._binding.source, request, _current_execution_context()
        )

    def sequential_snapshot(self, *, previous=None):
        """Retrieve a current exact checkpoint, optionally proving its parent."""
        from increment.readouts._evidence import retained_sequential_snapshot

        return retained_sequential_snapshot(
            self._binding.source, previous, _current_execution_context()
        )

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
        from increment._source_resolver import resolver_for
        from increment.kernel.resolver import run as run_experiment

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
        }
        experiment, resolver = resolver_for(self._open_binding(), run_options=options)
        ambient = (
            nullcontext()
            if _execution_context is None
            else _execution_context_scope(_execution_context)
        )
        with ambient:
            (scoped,) = run_experiment(
                experiment,
                resolver,
                ctx=_execution_context,
                execution_context=_execution_context,
            )
        return cast("LiftEstimates | ContrastResults", scoped.collection)

    def _run_scoped_readouts(self, *, coherence: Literal["captured", "live"], **options: Any):
        from increment.readouts._line2 import run_scoped_line2

        return run_scoped_line2(self, coherence=coherence, **options)

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
        _ = self._src
        return operations.estimate_cate_binding(
            self._binding,
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
        _ = self._src
        return operations.validate_cate_binding(
            self._binding,
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
        _ = self._src
        return operations.targeting_rule_binding(
            self._binding,
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
        _ = self._src
        return operations.select_targeting_rule_binding(
            self._binding,
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
        return run_breakout(
            self._open_binding(),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            population=population,
        )

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
        return run_breakout_rows(
            self._open_binding(),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            exploratory_metrics=exploratory_metrics,
            correction=correction,
            population=population,
            execution_context=_execution_context,
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
        return run_daily(
            self._open_binding(),
            metrics=metrics,
            dimension=dimension,
            exploratory_metrics=exploratory_metrics,
            population=population,
            coherence=coherence,
        )

    def _select_day_axis_metrics(
        self,
        metrics: Sequence[str | Metric] | None,
        *,
        caller: str,
        added: Sequence[Metric] = (),
    ) -> list[Metric]:
        return select_day_axis_metrics(self._open_binding(), metrics, caller=caller, added=added)

    def _day_axis(self, execution_context=None) -> DayAxisReadouts:
        return day_axis_readouts(self._open_binding(), execution_context)

    def _preflight_line4_views(
        self,
        *,
        population: Literal["assigned", "triggered"],
        readouts: Sequence[Literal["run", "daily", "daily_lift", "asof", "asof_lift"]],
    ) -> None:
        preflight_line4_views(self._open_binding(), population=population, readouts=readouts)

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
        return run_daily_lift(
            self._open_binding(),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            estimands=estimands,
            dimension=dimension,
            population=population,
            coherence=coherence,
        )

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
        return run_asof(
            self._open_binding(),
            metrics=metrics,
            dimension=dimension,
            completed_windows_only=completed_windows_only,
            exploratory_metrics=exploratory_metrics,
            population=population,
            coherence=coherence,
        )

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

        return run_asof_lift(
            self._open_binding(),
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            metrics=metrics,
            dimension=dimension,
            estimands=estimands,
            completed_windows_only=completed_windows_only,
            exploratory_metrics=exploratory_metrics,
            population=population,
            coherence=coherence,
        )
