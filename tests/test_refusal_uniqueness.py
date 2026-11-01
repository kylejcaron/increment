from __future__ import annotations

import pytest

from increment import Scope, from_definitions, from_frame, from_moments
from increment.query import panel_sql
from increment.workflows import experiment_definition, export_moments, metrics
from tests.analysis_factory import bound_source, held_source, view_readout
from tests.refusal_registry import walk_refusal_registries


def test_every_refusal_code_resolves_to_exactly_one_spec() -> None:
    """A code may appear in more than one module-level attribute (a dict
    registry entry, or a standalone constant) -- an importer reusing a
    canonical owner's spec so its own `_raise(code)` helper resolves it
    -- but every appearance MUST be the *same* RefusalSpec/WarningSpec
    object. Two independently constructed spec instances under one code
    string is exactly how a message or exception/warning type silently
    drifts apart between entry points; a `RefusalSpec` and a `WarningSpec`
    sharing one code string is caught the same way, since they can never
    be the same object."""
    registrations = walk_refusal_registries()
    violations = {
        code: sorted({loc for loc, _ in entries})
        for code, entries in registrations.items()
        if len({id(spec) for _, spec in entries}) > 1
    }
    assert violations == {}, (
        f"{len(violations)} refusal/warning code(s) registered as independently "
        f"constructed (drift-prone) duplicates: {violations}"
    )


def test_arm_evidence_hazard_raises_the_same_code_from_every_entry_point() -> None:
    """The cluster+prior hazard (the same dispatch shape as other
    cluster+sequential/sequential-prior examples) must raise the SAME
    code whether reached through run(experiment, resolver) or the public
    estimate_lift()."""
    import pandas as pd

    from increment.errors import CapabilityError
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.estimation.engine import estimate_lift
    from increment.estimation.inference import Normal
    from increment.semantics.models import MeanMetric

    rows = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(40)],
            "store_id": [f"s{i % 20}" for i in range(40)],
            "variant": ["treatment" if i % 2 else "control" for i in range(40)],
            "revenue": [10.0 + i for i in range(40)],
        }
    )
    experiment, resolver = from_frame(
        rows,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        cluster="store_id",
    )
    with pytest.raises(CapabilityError) as via_analysis:
        view_readout(experiment, resolver, prior=Normal(mu=0.0, sigma=0.01))

    metric = MeanMetric(name="m", entity="u", fact="f", aggregation="sum")

    def _cluster_row(group: str, ys: list[float]) -> dict:
        raw = {
            "experiment_id": "e",
            "metric": "m",
            "group_id": group,
            "n": len(ys),
            "sum_y": sum(ys),
            "sum_y2": sum(y * y for y in ys),
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": sum(1.0 for _ in ys),
            "sum_den2": sum(1.0 for _ in ys),
            "sum_yden": sum(ys),
        }
        return centered_row_from_raw_sums(raw)

    cluster_rows = [
        _cluster_row("C", [5.0 + 0.1 * ((i % 5) - 2) for i in range(50)]),
        _cluster_row("T", [5.5 + 0.1 * ((i % 5) - 2) for i in range(50)]),
    ]
    with pytest.raises(CapabilityError) as via_direct:
        estimate_lift(
            [metric],
            cluster_rows,
            control_group="C",
            cluster="store",
            prior=Normal(mu=0.0, sigma=0.01),
        )

    assert via_analysis.value.code == via_direct.value.code == "arm.adjustment.cluster_prior"


def test_sequential_cuped_hazard_raises_the_same_code_from_every_entry_point() -> None:
    """The sequential-inference + non-predeclared-CUPED hazard used to crash
    with a KeyError instead of a coded refusal: ``sequential_support_refusal``
    still returned the deleted string 'readout.adjustment.sequential_cuped',
    which none of its three consumers' own refusal registries (or, for two of
    them, ``compatibility._REFUSALS``) had registered under that name anymore.

    Each consumer is exercised at the same isolation the codebase already
    uses for the typed arm-contract dispatch itself (`test_arm_contract.py`,
    `test_sequential_predictable_transforms.py` construct typed requests
    directly rather than driving a full source/query stack).  The public
    `run()`, `estimate_lift()`, and `estimate_encouragement()` paths
    also reach these checks for requests that bypass their earlier plan gates.
    Calling each consumer directly isolates the shared code contract and keeps
    the three entry points aligned.
    """
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest, _validate_sequential_adjustment
    from increment.errors import CapabilityError
    from increment.estimation.encouragement import _resolve_encouragement_methods
    from increment.estimation.engine import Method, _prepare_lift_estimation
    from increment.estimation.sequential import AlwaysValid
    from increment.frame import MetricSpec, synthesise_metric
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import MeanMetric
    from increment.sources import SourceContext
    from tests.sequential_cases import registration as reg_fixture

    # A Bernoulli registration whose models cover "outcome" only -- neither
    # "revenue" (readout path) nor "m" (engine/encouragement paths) is
    # predeclared or retained, so cuped_capability/adjustment_kind resolve
    # to the unpredictable "cuped" flavour for either metric.
    registration = reg_fixture(law="bernoulli")
    inference = AlwaysValid(registration=registration)
    cuped_method = Method(name="cuped", variance_reduction="cuped")

    # run() path: _readout_request._validate_sequential_adjustment.
    metric = synthesise_metric(MetricSpec(name="revenue"))
    design = Randomized(control_group="control")
    plan = compile_decision_plan(None, (metric,), path="frame", design=design)
    # Post-hoc-declared inference bypasses compile_decision_plan's own
    # cuped/coefficient gate (validate_compiled_encouragement_plan), which
    # refuses this exact shape before a ReadoutRequest can even exist --
    # isolating the runtime validator this test targets.
    plan = plan.model_copy(update={"inference": inference})
    config = ResolvedMetricConfig(metric, cuped_method, (), None, False)
    request = ReadoutRequest(
        context=SourceContext(
            study_id="exp",
            design=design,
            plan=plan,
            metrics=(metric,),
            configs=(config,),
            cluster=None,
        ),
        metrics=(metric,),
        configs=(config,),
        view="run",
        grain="total",
    )
    with pytest.raises(CapabilityError) as via_readout_request:
        _validate_sequential_adjustment(request)

    # Direct estimate_lift() path: engine._prepare_lift_estimation.
    with pytest.raises(CapabilityError) as via_estimate_lift:
        _prepare_lift_estimation(
            [MeanMetric(name="m", entity="u", fact="f")],
            [],
            "control",
            [cuped_method],
            None,
            0.05,
            "two-sided",
            inference,
            0.0,
            None,
            None,
            None,
        )

    # Encouragement path: encouragement._resolve_encouragement_methods.
    with pytest.raises(CapabilityError) as via_encouragement:
        _resolve_encouragement_methods(
            [cuped_method],
            ("itt",),
            None,
            inference,
            None,
            [MeanMetric(name="m", entity="u", fact="f")],
            None,
        )

    assert (
        via_readout_request.value.code
        == via_estimate_lift.value.code
        == via_encouragement.value.code
        == "arm.adjustment.sequential_cuped"
    )


def test_every_arm_contract_unsupported_code_is_registered_in_compatibility() -> None:
    """Every ``arm.*``/``contrast.*`` code the arm/contrast evidence contract
    can return as ``Unsupported(code)`` must resolve through
    ``compatibility.ARM_COMPATIBILITY_REFUSALS`` (``refuse_unsupported`` raises
    by dict lookup, so a missing registration is a KeyError instead of the
    intended coded refusal -- the exact failure mode
    ``sequential_support_refusal`` hit). Enumerated by driving the contract's
    public dispatchers -- ``arm_runtime_support`` over its request axes,
    ``arm_planning_support`` over the plans ``ArmPlanningProcedure.standard``
    builds and over real ``Baseline`` knobs, and ``ContrastEvidenceContract`` --
    not by grepping the module's source text for string literals, so a new
    branch added later is covered automatically."""
    from increment.compatibility import ARM_COMPATIBILITY_REFUSALS
    from increment.estimation.arm_contract import (
        AbsoluteDecisionPolicy,
        ArmCompatibilityRequest,
        ArmPlanningProcedure,
        ContrastCompatibilityRequest,
        ContrastEvidenceContract,
        FamilyPolicy,
        MethodCapability,
        MetricCapabilities,
        Support,
        Unsupported,
        arm_planning_support,
        arm_runtime_support,
    )
    from increment.estimation.decision_types import FixedInference
    from increment.estimation.sequential import AlwaysValid, AsymptoticMean
    from increment.power.core import Baseline
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        ParallelAssignment,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.models import InferenceSpec
    from tests.asymptotic_cases import mean_registration
    from tests.estimation.test_arm_contract import _metric_capabilities
    from tests.sequential_cases import registration
    from tests.test_compatibility_contract import _axes, _decision

    codes: set[str] = set()

    def _record(support: Support) -> None:
        if isinstance(support, Unsupported):
            codes.add(support.refusal_code)

    # Sweep arm_runtime_support's axes: dependence, inference flavour, CUPED
    # flavour, prior presence and metric type. Inference decides which CUPED
    # coefficients are admissible, so coefficient admissibility sweeps within it.
    for dependence in ("iid", "cluster"):
        for inference in (
            FixedInference(),
            AsymptoticMean(registration=mean_registration()),
            AlwaysValid(registration=registration()),
        ):
            for variance_reduction in ("none", "predeclared_cuped", "retained_cuped"):
                for metric_type in ("mean", "conversion", "retention", "ratio", "quantile"):
                    for prior_present in (False, True):
                        _record(
                            arm_runtime_support(
                                ArmCompatibilityRequest(
                                    assignment=ParallelAssignment(),
                                    analysis=_axes(),
                                    dependence=dependence,
                                    inference=inference,
                                    estimand="itt",
                                    metric=_metric_capabilities(metric_type=metric_type),
                                    decision=_decision(),
                                    methods=(
                                        MethodCapability(
                                            role="decision",
                                            estimator=(
                                                "cuped"
                                                if variance_reduction != "none"
                                                else "unadjusted"
                                            ),
                                            variance_reduction=variance_reduction,
                                        ),
                                    ),
                                    prior_present=prior_present,
                                )
                            )
                        )

    # arm_planning_support: the plans the public constructor builds, then the
    # planning-only baseline knobs -- a real Baseline, so its own validation and
    # not a duck-typed stand-in decides which knob combinations exist.
    for metric_type in ("mean", "quantile"):
        for clustered in (False, True):
            for inference in (None, InferenceSpec(kind="asymptotic_mean")):
                _record(
                    arm_planning_support(
                        ArmPlanningProcedure.standard(
                            metric_type, clustered=clustered, inference=inference
                        )
                    )
                )
    for procedure, baseline in (
        (
            ArmPlanningProcedure.standard(),
            Baseline(mean=1.0, var=1.0, cluster_icc=0.1, avg_cluster_size=25.0),
        ),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, cluster_size_cv=0.5)),
        (ArmPlanningProcedure.standard(clustered=True), Baseline(mean=1.0, var=1.0)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, icc=0.1)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, compliance=0.5)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, trigger_rate=0.5)),
        (ArmPlanningProcedure.standard(), Baseline(mean=1.0, var=1.0, cuped_rho=0.1)),
    ):
        _record(arm_planning_support(procedure, baseline=baseline))

    # ContrastEvidenceContract's assignment guard is unreachable at its current
    # field type: SwitchbackAssignment.sequence is a two-member discriminated
    # union covering exactly the accepted schemes. Contrast inference has a
    # separate direct-estimator guard, so only that code remains registered.
    contract = ContrastEvidenceContract()
    window = SwitchbackWindow(washout_steps=0, observation_steps=4)
    assignment = SwitchbackAssignment(sequence=IndependentBernoulliOrder(), window=window)
    metric = MetricCapabilities(
        metric_type="mean",
        value_scale="absolute",
        winsorization="none",
        outcome_window="bounded",
        uptake_window="not_applicable",
    )

    def _contrast_request(**overrides: object) -> ContrastCompatibilityRequest:
        fields: dict[str, object] = {
            "assignment": assignment,
            "inference": FixedInference(),
            "estimand": "ate",
            "metric": metric,
            "decision": AbsoluteDecisionPolicy(
                alternative="two-sided",
                null_abs=0.0,
                family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
            ),
        }
        fields.update(overrides)
        return ContrastCompatibilityRequest(**fields)

    for overrides in (
        {
            "decision": AbsoluteDecisionPolicy(
                alternative="two-sided",
                null_abs=0.0,
                family=FamilyPolicy(kind="bonferroni", axes=("metric",), nominal_alpha=0.05),
            )
        },
        {
            "metric": MetricCapabilities(
                metric_type="quantile",
                value_scale="absolute",
                winsorization="none",
                outcome_window="bounded",
                uptake_window="not_applicable",
            )
        },
    ):
        _record(contract.runtime_support(_contrast_request(**overrides)))

    assert codes, "the axis sweep above must produce at least one Unsupported code"
    missing = codes - set(ARM_COMPATIBILITY_REFUSALS)
    assert missing == set(), (
        f"{len(missing)} arm-contract code(s) are unregistered in "
        f"compatibility.ARM_COMPATIBILITY_REFUSALS and would KeyError on refuse: {missing}"
    )
    # arm.metric.unsupported is the bh/e_bh branch of arm_planning_support,
    # which no valid ArmPlanningProcedure reaches: its own validator refuses
    # that family before the dispatcher runs. It must still be registered.
    assert {"arm.metric.unsupported", "contrast.inference"} <= set(ARM_COMPATIBILITY_REFUSALS)


def _trap_source_reads(monkeypatch: pytest.MonkeyPatch, source: object) -> None:
    """Fail the test if a refusal reaches the source instead of preceding it."""

    def _reached(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the source was read before the refusal")

    for name in ("moments", "breakout_moments", "unit_frame"):
        if hasattr(type(source), name):
            monkeypatch.setattr(type(source), name, _reached)


@pytest.mark.parametrize("source_kind", ["moments", "unit_panel"])
def test_percentile_winsor_refuses_before_raw_unit_frame_read(
    source_kind: str, tmp_path, monkeypatch
) -> None:
    from typing import Any, cast

    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import MetricSpec
    from increment.estimators import CapabilityError

    metric = MetricSpec(name="revenue", winsorization={"upper_percentile": 0.5})
    if source_kind == "moments":
        experiment, resolver = from_frame(
            pa.table(
                {
                    "unit": [1, 2, 3, 4, 5, 6],
                    "arm": ["control", "control", "control", "treatment", "treatment", "treatment"],
                    "revenue": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            ),
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics=[metric],
        )
        path = tmp_path / "percentile-moments.parquet"
        export_moments(experiment, resolver, path)
        analysis = from_moments(
            pq.read_table(path).to_pylist(), metrics=[metric], control="control"
        )
    else:
        analysis = from_frame(
            pl.DataFrame(
                {
                    "user_id": ["c1", "c2", "t1", "t2"],
                    "variant": ["control", "control", "treatment", "treatment"],
                    "date": ["2025-01-01"] * 4,
                    "revenue": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            shape="panel",
            unit="user_id",
            group="variant",
            date="date",
            metrics=[metric],
            control="control",
        )
    _trap_source_reads(monkeypatch, cast("Any", held_source(analysis[1])))
    with pytest.raises(CapabilityError) as refusal:
        view_readout(*analysis)
    assert refusal.value.code == "evidence.raw_outcomes_unavailable"


_DAY_AXIS_HAZARD_METRICS = {
    "unbounded_retention": {
        "type": "retention",
        "name": "m",
        "entity": "user_id",
        "fact": "page_view",
        "threshold_days": 1,
        "preferred_direction": "increase",
    },
    "winsorized_mean": {
        "type": "mean",
        "name": "m",
        "entity": "user_id",
        "fact": "purchase",
        "aggregation": "sum",
        "window_days": 3,
        "winsorization": {"upper_value": 100.0},
    },
}


def _native_day_axis_analysis(tmp_path, hazard: str, *, con=None):
    import datetime as dt

    import ibis
    import yaml

    start = dt.datetime(2026, 1, 1, 9)
    rows = []
    for i in range(12):
        arm = "control" if i % 2 == 0 else "treatment"
        base = {"user_id": f"u{i:02d}", "group_id": None, "revenue": None}
        rows.append({**base, "ts": start, "event": "page_view", "group_id": arm})
        rows.append(
            {**base, "ts": start + dt.timedelta(hours=1), "event": "purchase", "revenue": 5.0 + i}
        )
        if i % 3:
            rows.append({**base, "ts": start + dt.timedelta(days=2), "event": "page_view"})
    rows.append(
        {
            "user_id": "u00",
            "group_id": None,
            "revenue": None,
            "ts": start + dt.timedelta(days=9),
            "event": "page_view",
        }
    )
    definitions = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "page_view", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "page_view"}],
        "metrics": [_DAY_AXIS_HAZARD_METRICS[hazard]],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-01",
                "observation_end": "2026-01-12",
                "control_group": "control",
                "plan": {"primary": "m"},
            }
        ],
    }
    path = tmp_path / "defs.yaml"
    path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    con = ibis.duckdb.connect() if con is None else con
    con.create_table("raw_events", obj=rows)
    return from_definitions(path, con, experiment_name="exp")


def _frame_panel_day_axis_analysis(hazard: str):
    import datetime as dt

    import pandas as pd

    from increment.frame import MetricSpec

    start = dt.date(2026, 1, 1)
    rows = [
        {
            "user_id": f"u{i}",
            "variant": "control" if i % 2 == 0 else "treatment",
            "day": start + dt.timedelta(days=d),
            "exposed_on": start,
            "value": float(d > 0 and i % 3 != 0)
            if hazard == "unbounded_retention"
            else 5.0 + i + d,
        }
        for i in range(8)
        for d in range(4)
    ]
    spec = (
        MetricSpec(name="m", type="retention", value_column="value", threshold_days=1)
        if hazard == "unbounded_retention"
        else MetricSpec(name="m", value_column="value", winsorization={"upper_value": 100.0})
    )
    return from_frame(
        pd.DataFrame(rows),
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        exposure_date="exposed_on",
        metrics=[spec],
    )


_UNBOUNDED_RETENTION_SUBSTRATES = ("definitions", "frame_panel", "unit_day_artifact")


@pytest.mark.parametrize("substrate", _UNBOUNDED_RETENTION_SUBSTRATES)
@pytest.mark.integration
def test_unbounded_retention_daily_hazard_raises_the_same_code_from_every_source(
    substrate: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbounded retention band never completes, so the day-axis views refuse
    it whatever the substrate. The shared readout gate owns that rule: the
    direct readout and the daily views of ``run`` raise one code before any
    source read."""
    from increment import readouts
    from increment.errors import CodedError

    entry_points = {}
    if substrate == "unit_day_artifact":
        from tests.test_unit_day_artifact_adoption import _adopted_artifact_fixture

        source = _adopted_artifact_fixture("retention_unbounded")["source"]
    else:
        experiment, resolver = (
            _native_day_axis_analysis(tmp_path, "unbounded_retention")
            if substrate == "definitions"
            else _frame_panel_day_axis_analysis("unbounded_retention")
        )
        source = bound_source(resolver)
        entry_points["run(daily values)"] = lambda: view_readout(
            experiment, resolver, Scope(view="daily", grain="daily", values=True)
        )
        entry_points["run(daily lift)"] = lambda: view_readout(
            experiment, resolver, Scope(view="daily", grain="daily")
        )
    _trap_source_reads(monkeypatch, source)
    entry_points["readouts.daily"] = lambda: readouts.daily(source)

    codes = {}
    for name, call in entry_points.items():
        with pytest.raises(CodedError) as raised:
            call()
        codes[name] = raised.value.code
    assert set(codes.values()) == {"breakout.retention.unbounded"}, codes


@pytest.mark.slow
@pytest.mark.parametrize("method", ["run_daily", "run_daily_lift"])
@pytest.mark.integration
def test_unbounded_retention_dimensioned_artifact_analysis_refuses_before_any_artifact_read(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dimensioned day-axis read on an adopted artifact reaches the source through
    ``breakout_moments`` rather than the readout gate, so the facade's day-axis
    validation must raise the same code before that read."""
    import ibis

    from increment.errors import CodedError
    from increment.evidence import BreakoutIdentity
    from tests.parity_harness.cases import _close_parity_binding, _publish_and_adopt
    from tests.test_analysis_breakout import _analysis_with_country_breakout

    con = ibis.duckdb.connect()
    adopted = _publish_and_adopt(
        con,
        _analysis_with_country_breakout(con, unbounded_retention=True),
        kinds=("breakout_dimension",),
    )
    experiment, resolver = adopted
    try:
        _trap_source_reads(monkeypatch, bound_source(resolver))
        dimension = experiment_definition(experiment, resolver).breakouts[0].property
        scope = Scope(
            view="daily",
            grain="daily",
            values=method == "run_daily",
            segment=BreakoutIdentity(dimension, None),
        )
        with pytest.raises(CodedError) as raised:
            view_readout(experiment, resolver, scope, metrics=["unbounded"])
        assert raised.value.code == "breakout.retention.unbounded"
    finally:
        _close_parity_binding(adopted)


@pytest.mark.slow
@pytest.mark.integration
def test_unbounded_retention_hazard_covers_every_adapter_that_serves_the_daily_axis(
    tmp_path,
) -> None:
    """Every registered adapter that can read the daily axis must appear in the
    same-code test above. The engine rule precedes the source's grain validation,
    so an adapter without the daily grain raises the same code; the policy sweep in
    `test_source_capabilities.py` declares that for every registered substrate."""
    from tests.source_conformance import arm_adapters

    daily_adapters = {
        adapter.name
        for adapter in arm_adapters(tmp_path)
        if "daily" in adapter.build()[0].capabilities
    }
    assert daily_adapters == set(_UNBOUNDED_RETENTION_SUBSTRATES)


@pytest.mark.parametrize("substrate", _UNBOUNDED_RETENTION_SUBSTRATES)
@pytest.mark.integration
def test_completed_windows_over_unbounded_retention_raises_one_code_from_every_source(
    substrate: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No unit's window over an unbounded retention band ever completes, so
    `completed_windows_only=True` is contradictory on every as-of entry point. The
    direct readout and both as-of views of ``run`` raise one code before any source
    read, rather than each adapter's own refusal."""
    from increment import readouts
    from increment.errors import CodedError
    from tests.parity_harness.cases import _close_parity_binding

    if substrate == "unit_day_artifact":
        import ibis

        from tests.parity_harness.cases import _publish_and_adopt

        con = ibis.duckdb.connect()
        bound = _publish_and_adopt(
            con, _native_day_axis_analysis(tmp_path, "unbounded_retention", con=con)
        )
    else:
        bound = (
            _native_day_axis_analysis(tmp_path, "unbounded_retention")
            if substrate == "definitions"
            else _frame_panel_day_axis_analysis("unbounded_retention")
        )
    experiment, resolver = bound
    try:
        source = bound_source(resolver)
        _trap_source_reads(monkeypatch, source)
        entry_points = {
            "run(asof values)": lambda: view_readout(
                experiment,
                resolver,
                Scope(view="asof", grain="asof", values=True, completion_policy=True),
            ),
            "run(asof lift)": lambda: view_readout(
                experiment, resolver, Scope(view="asof", grain="asof", completion_policy=True)
            ),
            "readouts.asof_lift": lambda: readouts.asof_lift(source, completed_windows_only=True),
        }
        codes = {}
        for name, call in entry_points.items():
            with pytest.raises(CodedError) as raised:
                call()
            codes[name] = raised.value.code
        assert set(codes.values()) == {"breakout.retention.completion"}, codes
    finally:
        _close_parity_binding(bound)


# The day-axis views of `run` and the readout gate behind them refuse one hazard with one
# code per entry point; within an entry point it must not depend on the source.
_WINSORIZED_DAILY_CODES = {
    "readouts.daily": "evidence.operation_unavailable",
    "run(daily values)": "evidence.operation_unavailable",
    "run(daily lift)": "evidence.operation_unavailable",
}


_WINSORIZED_DAILY_SUBSTRATES = ("definitions", "frame_panel", "unit_day_artifact")


@pytest.mark.parametrize("substrate", _WINSORIZED_DAILY_SUBSTRATES)
@pytest.mark.integration
def test_winsorized_daily_hazard_raises_one_code_per_entry_method_from_every_source(
    substrate: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from increment import readouts
    from increment.errors import CodedError
    from tests.parity_harness.cases import _close_parity_binding

    if substrate == "unit_day_artifact":
        import ibis

        from tests.parity_harness.cases import _publish_and_adopt

        con = ibis.duckdb.connect()
        bound = _publish_and_adopt(
            con, _native_day_axis_analysis(tmp_path, "winsorized_mean", con=con)
        )
    else:
        bound = (
            _native_day_axis_analysis(tmp_path, "winsorized_mean")
            if substrate == "definitions"
            else _frame_panel_day_axis_analysis("winsorized_mean")
        )
    experiment, resolver = bound
    try:
        source = bound_source(resolver)
        _trap_source_reads(monkeypatch, source)
        entry_points = {
            "run(daily values)": lambda: view_readout(
                experiment, resolver, Scope(view="daily", grain="daily", values=True)
            ),
            "run(daily lift)": lambda: view_readout(
                experiment, resolver, Scope(view="daily", grain="daily")
            ),
            "readouts.daily": lambda: readouts.daily(source),
        }
        codes = {}
        for name, call in entry_points.items():
            with pytest.raises(CodedError) as raised:
                call()
            codes[name] = raised.value.code
        assert codes == _WINSORIZED_DAILY_CODES
    finally:
        _close_parity_binding(bound)


@pytest.mark.slow
@pytest.mark.integration
def test_winsorized_daily_hazard_covers_every_adapter_that_can_express_it(tmp_path) -> None:
    """Winsorization applies to a mean metric on a source that serves the day axis. An
    adapter's default metric does not decide that: the unit-day artifact adapter defaults
    to a conversion metric but supports a winsorized mean (its supported-shape
    declaration), so it belongs to the sweep above."""
    from tests.source_conformance import arm_adapters
    from tests.test_unit_day_artifact_adoption import ARTIFACT_SUPPORTED_SHAPES

    def can_express_winsorized_mean(adapter) -> bool:
        if adapter.name == "unit_day_artifact":
            return "winsorized_mean" in ARTIFACT_SUPPORTED_SHAPES
        return adapter.build()[1].type == "mean"

    expressible = {
        adapter.name
        for adapter in arm_adapters(tmp_path)
        if "daily" in adapter.build()[0].capabilities and can_express_winsorized_mean(adapter)
    }
    assert expressible == set(_WINSORIZED_DAILY_SUBSTRATES)


def _evidence_retired_codes() -> set[str]:
    from increment.errors import RETIRED_CODES

    return {
        code
        for code, replacement in RETIRED_CODES.items()
        if (
            isinstance(replacement, str)
            and replacement.startswith("evidence.")
            or isinstance(replacement, tuple)
            and any(item.startswith("evidence.") for item in replacement)
        )
    }


def test_retired_codes_are_not_registered() -> None:
    from increment.errors import RETIRED_CODES

    registrations = walk_refusal_registries()
    retired = _evidence_retired_codes()
    assert retired.isdisjoint(registrations)
    live = set(registrations)
    for code in retired:
        replacement = RETIRED_CODES[code]
        assert replacement is not None
        replacements = (replacement,) if isinstance(replacement, str) else replacement
        assert set(replacements) <= live


def test_no_published_page_cites_a_retired_code() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    published = [root / "README.md"]
    for directory in (root / "docs", root / "examples"):
        published.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".md", ".mdx", ".html", ".py", ".ipynb"}
            and "docs/superpowers/" not in path.as_posix()
            and "release-notes" not in path.as_posix().lower()
        )
    stale = {
        code: [path.relative_to(root).as_posix() for path in published if code in path.read_text()]
        for code in sorted(_evidence_retired_codes())
    }
    assert {code: paths for code, paths in stale.items() if paths} == {}


def test_evidence_refusals_round_trip_pickle_and_deepcopy() -> None:
    import copy
    import pickle

    from increment.estimators import CapabilityError
    from tests.source_conformance import unit_summary_table

    experiment, resolver = from_frame(
        unit_summary_table(),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    try:
        with pytest.raises(CapabilityError) as raised:
            view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
        refusal = raised.value
        assert refusal.code == "evidence.grain_unavailable"
        assert set(refusal.context) == {"constructor", "requested", "offered", "route"}
        for restored in (pickle.loads(pickle.dumps(refusal)), copy.deepcopy(refusal)):
            assert type(restored) is type(refusal)
            assert restored.code == refusal.code
            assert restored.context == refusal.context
    finally:
        resolver.close()


@pytest.mark.integration
def test_each_evidence_hazard_has_one_code_on_every_constructor(
    seeded_con, seeded_defs, tmp_path
) -> None:
    from typing import Any, cast

    import pyarrow.parquet as pq

    from increment.estimators import CapabilityError
    from tests.analysis_factory import held_source
    from tests.source_conformance import unit_summary_table

    summary_experiment, summary_resolver = from_frame(
        unit_summary_table(),
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=("segment",),
    )
    moments: tuple[Any, Any] | None = None
    artifact: tuple[Any, Any] | None = None
    definitions_experiment, definitions_resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="none"
    )
    try:
        summary_source = cast("Any", held_source(summary_resolver))
        metric = metrics(summary_experiment, summary_resolver)[0]
        with pytest.raises(CapabilityError) as raised:
            summary_source.moments(metric, grain="daily")
        assert raised.value.code == "evidence.grain_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]

        with pytest.raises(CapabilityError) as raised:
            panel_sql(summary_experiment, summary_resolver)
        assert raised.value.code == "evidence.operation_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]

        path = tmp_path / "evidence-moments.parquet"
        export_moments(summary_experiment, summary_resolver, path)
        moments = from_moments(
            pq.read_table(path).to_pylist(), metrics={"revenue": "mean"}, control="control"
        )
        moments_source = cast("Any", held_source(moments[1]))
        moments_metric = metrics(*moments)[0]
        with pytest.raises(CapabilityError) as raised:
            moments_source.moments(moments_metric, by=("segment",))
        assert raised.value.code == "evidence.breakout_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]
        with pytest.raises(CapabilityError) as raised:
            moments_source.sql()
        assert raised.value.code == "evidence.operation_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]

        definition_source = cast("Any", held_source(definitions_resolver))
        with pytest.raises(CapabilityError) as raised:
            definition_source.sql(grain="daily")
        assert raised.value.code == "evidence.grain_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]
        from tests.source_conformance import unit_day_artifact_analysis

        artifact = unit_day_artifact_analysis(tmp_path)
        artifact_source = cast("Any", held_source(artifact[1]))
        with pytest.raises(CapabilityError) as raised:
            artifact_source.moments(metrics(*artifact)[0], grain="weekly")
        assert raised.value.code == "evidence.grain_unavailable"
        assert isinstance(raised.value.context["route"], str) and raised.value.context["route"]
        for request, code in (
            (
                lambda: moments_source.unit_frame(moments_metric),
                "evidence.unit_rows_unavailable",
            ),
            (
                lambda: moments_source.unit_frame(moments_metric, covariates=("spend",)),
                "evidence.covariate_unavailable",
            ),
            (
                lambda: moments_source.cluster_counts(),
                "evidence.cluster_unavailable",
            ),
            (
                lambda: moments_source.unit_frame(moments_metric, outcome_stage="raw"),
                "evidence.raw_outcomes_unavailable",
            ),
        ):
            with pytest.raises(CapabilityError) as raised:
                request()
            assert raised.value.code == code
            assert set(raised.value.context) == {
                "constructor",
                "requested",
                "offered",
                "route",
            }
    finally:
        summary_resolver.close()
        cast(Any, definitions_resolver).close()
        if moments is not None:
            moments[1].close()
        if artifact is not None:
            artifact[1].close()
