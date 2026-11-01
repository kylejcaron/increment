from __future__ import annotations

import sys
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from increment import Method, MetricSpec, from_definitions, run
from increment.builders import from_frame
from increment.errors import CodedError
from increment.evidence import SequentialCaptureRequest
from increment.kernel.contracts import ImplementationRef, Scope
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import runnable
from increment.semantics.design import AdjustmentSet, Observational
from increment.workflows import capture_sequential, export_moments, unit_counts
from tests.analysis_factory import view_readout
from tests.test_kernel_builders import _frame


def _experiment():
    return from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
    )


def _scope(cell) -> Scope:
    scope = cell.identity.scope
    assert scope is not None
    return scope


@pytest.mark.parametrize("view,grain", [("run", "total"), ("daily", "daily"), ("asof", "asof")])
def test_compile_retains_declared_metrics(view, grain):
    experiment, _ = _experiment()
    experiment = replace(experiment, scopes=(Scope(view=view, grain=grain),))
    inventory, requests = experiment.compile()
    assert tuple(metric.name for metric in inventory.metrics) == ("value",)
    assert tuple(metric["name"] for metric in requests[0].requests[0].fields["metrics"]) == (
        "value",
    )


def test_runnable_keeps_known_refusals_with_unknown_sources():
    experiment, _ = _experiment()
    experiment = replace(
        experiment,
        scopes=(Scope(view="daily", grain="daily"),),
        source_refs=(
            *experiment.source_refs,
            SourceReference(name="missing", uri="frame://missing"),
        ),
    )
    status = runnable(experiment, "value")["randomized.unadjusted"]
    assert status.unknown == ("missing.capabilities",)
    assert "evidence.grain_unavailable" in {error.code for error in status.static_refusals}


def test_runnable_checks_design_covariates_and_counts_alternatives():
    from increment.declarations import (
        CountsSpec,
        EstimatorRef,
        MomentsSpec,
        UnitRowsSpec,
        register,
        unregister_estimator,
    )

    experiment, _ = _experiment()
    capabilities = replace(experiment.source_refs[0].capabilities, unit_rows=False)
    experiment = replace(
        experiment,
        analysis_context={},
        design={"adjustment": {"covariates": ("age",)}},
        source_refs=(replace(experiment.source_refs[0], capabilities=capabilities),),
    )
    ref = EstimatorRef(
        estimator_id="randomized.count_route_review",
        module="unused_estimator",
        attribute="estimate",
        contract="panel",
        estimand="ate",
        selector=None,
        needs=((MomentsSpec("total"), UnitRowsSpec(covariates="design")), (CountsSpec("cluster"),)),
    )
    register(ref)
    try:
        assert runnable(experiment, "value")[ref.estimator_id].static_refusals
    finally:
        unregister_estimator(ref.estimator_id)
    with_rows = replace(
        experiment,
        source_refs=(
            replace(experiment.source_refs[0], capabilities=replace(capabilities, unit_rows=True)),
        ),
    )
    assert "evidence.covariate_unavailable" in {
        error.code for error in runnable(with_rows, "value")["observational.iptw"].static_refusals
    }


def test_source_reference_copies_nested_bindings():
    from increment.evidence import ScalarEvidenceDeclaration
    from increment.frame import CupedCovariate

    experiment, _ = _experiment()
    bindings = {"value": (CupedCovariate(name="age", kind="continuous"),)}
    scalar = ScalarEvidenceDeclaration(bindings, False, False, (), frozenset(), False, False, None)
    reference = replace(
        experiment.source_refs[0],
        capabilities=replace(experiment.source_refs[0].capabilities, scalar=scalar),
    )
    experiment = replace(experiment, source_refs=(reference,))
    before = experiment.hash()
    bindings.clear()
    assert experiment.hash() == before
    assert Experiment.from_json(experiment.to_json()).source_refs == experiment.source_refs


@pytest.mark.parametrize("field", ["uri", "expected_digest"])
def test_bound_builder_refuses_other_source_identity(field):
    experiment, resolver = _experiment()
    reference = replace(experiment.source_refs[0], **{field: "other://source"})
    with pytest.raises(CodedError) as error:
        run(replace(experiment, source_refs=(reference,)), resolver)
    assert error.value.code == "kernel.resolver.digest_mismatch"


def test_builder_refuses_changed_registration_and_design_before_evidence_reads():
    experiment, resolver = _experiment()
    from tests.analysis_factory import replace_bound_source
    from tests.test_metric_evidence import _CountingSource

    counted = _CountingSource(resolver.binding.source)
    replace_bound_source(resolver, counted)
    changed = replace(
        experiment,
        registration=experiment.registration.model_copy(update={"registration_hash": "changed"}),
        design={**experiment.design, "control_group": "different"},
    )
    assert changed.source_refs == experiment.source_refs
    with pytest.raises(CodedError) as error:
        run(changed, resolver)
    assert error.value.code == "kernel.run.execution_unavailable"
    assert counted.unit_frame_reads == 0


def test_builder_executes_declared_daily_scope_and_preflights_segment():
    frame = pd.concat([_frame().assign(day="2025-01-01"), _frame().assign(day="2025-01-02")])
    frame["country"] = ["US", "CA", "CA", "US"] * 20
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
        breakouts=["country"],
    )
    daily = replace(experiment, scopes=(Scope(view="daily", grain="daily"),))
    assert {cell.identity.key.ds for cell in run(daily, resolver).cells} == {
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-01-02").date(),
    }
    from increment.evidence import BreakoutIdentity

    segmented = replace(
        daily,
        scopes=(
            Scope(
                view="daily",
                grain="daily",
                segment=(BreakoutIdentity(property="country", source=None),),
            ),
        ),
    )
    assert not runnable(segmented, "value")["randomized.unadjusted"].static_refusals

    breakout = replace(
        experiment,
        scopes=(
            Scope(
                view="breakout",
                grain="total",
                segment=(BreakoutIdentity(property="country", source=None),),
            ),
        ),
    )
    breakout_result = run(breakout, resolver)
    assert breakout_result.cells
    assert {verdict.status for verdict in breakout_result.decisions.cells} == {"not_evaluated"}
    assert {
        code for verdict in breakout_result.decisions.cells for code in verdict.reason_codes
    } == {"decision.not_evaluated.monitoring_view"}


def test_multibreakout_frame_runs_every_declared_selection_and_refuses_unknown_before_reads():
    from increment.evidence import BreakoutIdentity

    frame = _frame()
    frame["country"] = ["US", "CA", "CA", "US"] * (len(frame) // 4)
    frame["device"] = ["mobile", "mobile", "desktop", "desktop"] * (len(frame) // 4)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        breakouts=("country", "device"),
    )

    def selected(*identities):
        return replace(
            experiment,
            scopes=(Scope(view="breakout", grain="total", segment=identities),),
        )

    def compiled(requested):
        _, (request,) = requested.compile()
        breakouts = cast(Any, request.requests[0].fields["breakouts"])
        return tuple((item["source"], item["property"]) for item in breakouts)

    def executed(cells):
        return tuple(
            dict.fromkeys((cell.identity.key.source, cell.identity.key.dimension) for cell in cells)
        )

    every = selected()
    device = selected(BreakoutIdentity("device", None))
    for requested in (every, device):
        assert runnable(requested, "value")["randomized.unadjusted"].static_refusals == ()
    all_cells = run(every, resolver).cells
    assert compiled(every) == executed(all_cells) == ((None, "country"), (None, "device"))
    assert all(
        _scope(cell).segment == (BreakoutIdentity(cast(str, cell.identity.key.dimension), None),)
        for cell in all_cells
    )
    by_key = {cell.identity.key: cell for cell in all_cells}
    device_cells = run(device, resolver).cells
    assert compiled(device) == executed(device_cells) == ((None, "device"),)
    for cell in device_cells:
        expected = by_key[cell.identity.key]
        assert (cell.summary, cell.family) == (expected.summary, expected.family)

    for unknown in (
        selected(BreakoutIdentity("browser", None)),
        selected(BreakoutIdentity("country", "missing_source")),
    ):
        (preflight,) = runnable(unknown, "value")["randomized.unadjusted"].static_refusals
        with _frame_read_calls() as reads, pytest.raises(CodedError) as raised:
            run(unknown, resolver)
        assert reads == []
        assert (raised.value.code, raised.value.context) == (preflight.code, preflight.context)
        assert raised.value.code == "evidence.breakout_unavailable"


def test_registered_sequential_breakout_refuses_declared_breakouts_it_cannot_supply():
    import datetime as dt

    from increment import AnalysisPlan, InferenceSpec, Randomized
    from increment.evidence import BreakoutIdentity
    from increment.workflows import sequential_snapshot

    exposed, closed = dt.date(2025, 1, 1), dt.date(2025, 1, 10)
    frame = pd.DataFrame(
        [
            {
                "unit": f"u{index}",
                "arm": "control" if index % 2 else "treatment",
                "day": day,
                "exposed_on": exposed,
                "value": value,
                "country": ("US", "CA")[(index // 2) % 2],
                "device": ("mobile", "desktop")[(index // 4) % 2],
            }
            for index in range(80)
            for day, value in ((exposed, 0.0), (dt.date(2025, 1, 2), float(index % 7 + index % 2)))
        ]
    )
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed_on",
        observation_end=closed,
        metrics=[MetricSpec(name="value", type="mean", window_days=2)],
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        plan=AnalysisPlan(
            primary="value",
            inference=InferenceSpec(kind="asymptotic_mean", segments={"country": ("US", "CA")}),
        ),
        breakouts=("country", "device"),
    )
    capture = SequentialCaptureRequest(finalized=True, as_of=closed)

    def selected(*identities):
        return replace(
            experiment, scopes=(Scope(view="breakout", grain="total", segment=identities),)
        )

    for requested_capture in (None, capture):
        with _frame_read_calls() as reads, pytest.raises(CodedError) as raised:
            run(selected(), resolver, capture=requested_capture)
        assert raised.value.code == "sequential.route.unsupported"
        assert reads == []
    with pytest.raises(CodedError) as uncaptured:
        sequential_snapshot(experiment, resolver)
    assert uncaptured.value.code == "sequential.continuation.legacy"
    country = run(selected(BreakoutIdentity("country", None)), resolver, capture=capture).cells
    assert {cell.identity.key.dimension for cell in country} == {"country"}


@pytest.mark.parametrize("clear_declared_prior", [False, True])
def test_clustered_preflight_applies_saved_prior_override(clear_declared_prior):
    from increment.errors import IncrementWarning
    from increment.estimators import Normal
    from tests.warning_codes import warning_codes

    prior = Normal(mu=0, sigma=1)
    frame = _frame()
    frame["store"] = frame["unit"]
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        cluster="store",
        metrics=[
            MetricSpec(
                name="value",
                type="mean",
                prior=prior if clear_declared_prior else None,
            )
        ],
    )
    requested = Experiment.from_json(
        replace(
            experiment,
            request_options={"prior": None if clear_declared_prior else prior},
        ).to_json()
    )
    if clear_declared_prior:
        with pytest.warns(IncrementWarning) as caught:
            assert run(requested, resolver)
        assert warning_codes(caught) == ["analysis.run.override_replaces_declared"]
        expected = set()
    else:
        with pytest.raises(CodedError) as error:
            run(requested, resolver)
        assert error.value.code == "arm.adjustment.cluster_prior"
        expected = {error.value.code}

    status = runnable(requested, "value")["randomized.unadjusted"]
    assert {error.code for error in status.static_refusals} == expected


def test_bound_custom_provider_is_verification_only():
    experiment, resolver = _experiment()
    from increment.kernel.resolver import BoundResolver

    experiment = replace(
        experiment,
        source_refs=(replace(experiment.source_refs[0], expected_digest=None),),
    )
    with pytest.raises(CodedError) as error:
        run(experiment, BoundResolver(resolver._source))
    assert error.value.code == "kernel.run.provider_execution_unavailable"


@pytest.mark.parametrize("options", [{"capture": {"finalized": True}}, {"population": "triggered"}])
def test_experiment_rejects_execution_only_request_options(options):
    experiment, _ = _experiment()
    with pytest.raises(CodedError) as error:
        replace(experiment, request_options=options)
    assert error.value.code == (
        "kernel.request.capture_not_supported"
        if "capture" in options
        else "kernel.request.population_scope_only"
    )


@pytest.mark.parametrize("options", [{"capture": {"finalized": True}}, {"population": "triggered"}])
def test_experiment_json_rejects_execution_only_request_options(options):
    experiment, _ = _experiment()
    payload = experiment.canonical_payload()
    payload["request_options"] = options
    import json

    with pytest.raises(CodedError) as error:
        Experiment.from_json(json.dumps(payload))
    assert error.value.code == (
        "kernel.request.capture_not_supported"
        if "capture" in options
        else "kernel.request.population_scope_only"
    )


def test_preflight_uses_experiment_metric_selection_for_configurations():
    from increment.estimators import Normal

    frame = _frame().assign(other=lambda data: data["value"] + 2)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        cluster="unit",
        metrics=[
            MetricSpec(name="value", type="mean"),
            MetricSpec(name="other", type="mean", prior=Normal(mu=0, sigma=1)),
        ],
    )
    experiment = replace(
        experiment,
        metrics={"value": experiment.metrics["value"]},
    )

    assert [cell.identity.key.metric for cell in run(experiment, resolver).cells] == ["value"]
    assert runnable(experiment, "value")["randomized.unadjusted"].static_refusals == ()


@pytest.mark.parametrize(
    "operation,view,grain",
    [
        ("run", "run", "total"),
        ("run_daily", "daily", "daily"),
        ("run_daily_lift", "daily", "daily"),
        ("run_asof", "asof", "asof"),
        ("run_asof_lift", "asof", "asof"),
    ],
)
def test_observational_preflight_matches_execution_by_operation(operation, view, grain):
    frame = pd.concat([_frame().assign(day="2025-01-01"), _frame().assign(day="2025-01-02")])
    frame["pre"] = frame["unit"].str.removeprefix("u").astype(float) % 7
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        metrics={"value": "mean"},
        design=Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("pre",)),
        ),
    )
    scope = Scope(view=view, grain=grain)
    requested = Experiment.from_json(
        replace(
            experiment,
            scopes=(scope,),
            operations={scope.hash(): operation},
        ).to_json()
    )
    if operation in {"run_daily_lift", "run_asof_lift"}:
        with pytest.raises(CodedError) as error:
            run(requested, resolver)
        assert error.value.code == "facade.analysis.observational_day_axis"
        expected = {error.value.code}
    else:
        assert run(requested, resolver)
        expected = set()

    status = runnable(requested, "value")["observational.iptw"]
    assert {error.code for error in status.static_refusals} == expected


@pytest.mark.parametrize("estimands", [("itt",), ("unknown",)])
def test_encouragement_daily_preflight_matches_saved_estimands(estimands):
    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _frame().assign(day="2025-01-01", uptake=[0, 1] * 20)
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        metrics={"value": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake", window_days=1),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment affects outcomes only through uptake"
            ),
        ),
    )
    requested = Experiment.from_json(
        replace(
            experiment,
            scopes=(Scope(view="daily", grain="daily"),),
            request_options={"estimands": estimands},
        ).to_json()
    )
    status = runnable(requested, "value")["randomized.unadjusted"]
    if estimands == ("itt",):
        rows = run(requested, resolver)
        assert rows.cells and {cell.identity.key.estimand for cell in rows.cells} == {"itt"}
        assert status.static_refusals == ()
    else:
        with pytest.raises(CodedError) as error:
            run(requested, resolver)
        assert error.value.code == "readout.estimands.unknown"
        assert error.value.code in {refusal.code for refusal in status.static_refusals}


def test_compliance_cell_identity_uses_declared_uptake_fact():
    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _frame().assign(
        day="2025-01-01",
        uptake=[0, 1] * 20,
        clicked=[1, 0] * 20,
    )
    metric_hashes = []
    for fact in ("uptake", "clicked"):
        experiment, resolver = from_frame(
            frame,
            shape="panel",
            unit="unit",
            group="arm",
            date="day",
            metrics={"value": "mean"},
            design=Encouragement(
                control_group="control",
                uptake=UptakeSpec(fact=fact, window_days=1),
                exclusion_restriction=ExclusionRestriction(
                    justification="assignment affects outcomes only through uptake"
                ),
            ),
        )
        result = run(experiment, resolver, estimands=("compliance",))
        cell = next(cell for cell in result.cells if cell.identity.key.estimand == "compliance")
        metric_hashes.append(cell.identity.metric_hash)

    assert len(set(metric_hashes)) == 2


def test_daily_lift_run_decisions_are_not_evaluated_against_registered_plan():
    frame = pd.concat(
        [
            _frame()
            .assign(unit=lambda rows, date=date, repeat=repeat: rows["unit"] + f"-{date}-{repeat}")
            .assign(day=f"2025-01-0{date}")
            for date in (1, 2)
            for repeat in (1, 2)
        ],
        ignore_index=True,
    )
    frame.loc[frame["arm"] == "treatment", "value"] += 100
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    result = run(replace(experiment, scopes=(Scope(view="daily", grain="daily"),)), resolver)
    assert result.cells
    assert {verdict.status for verdict in result.decisions.cells} == {"not_evaluated"}
    assert {code for verdict in result.decisions.cells for code in verdict.reason_codes} == {
        "decision.not_evaluated.monitoring_view"
    }


def test_monitoring_scope_is_not_evaluated_against_whole_experiment_plan():
    from increment import AnalysisPlan

    frame = pd.concat(
        [_frame().assign(day="2025-01-01"), _frame().assign(day="2025-01-02")],
        ignore_index=True,
    )
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
        plan=AnalysisPlan(primary="value"),
    )
    result = run(replace(experiment, scopes=(Scope(view="daily", grain="daily"),)), resolver)

    assert result.cells
    assert {cell.identity.key.method_role for cell in result.cells} == {"decision"}
    assert {verdict.status for verdict in result.decisions.cells} == {"not_evaluated"}
    assert {verdict.reason_codes for verdict in result.decisions.cells} == {
        ("decision.not_evaluated.monitoring_view",)
    }
    assert {verdict.role for verdict in result.decisions.cells} == {"primary"}


def test_partial_run_selection_cannot_pass_omitted_registered_guardrail():
    from increment import AnalysisPlan

    frame = _frame()
    for metric in ("primary", "guard_a", "guard_b"):
        frame[metric] = frame["value"] + np.where(frame["arm"] == "treatment", 100, 0)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="primary", type="mean"),
            MetricSpec(name="guard_a", type="mean", preferred_direction="increase"),
            MetricSpec(name="guard_b", type="mean", preferred_direction="increase"),
        ],
        plan=AnalysisPlan(primary="primary", guardrails=("guard_a", "guard_b")),
    )

    result = run(experiment, resolver, metrics=["primary", "guard_a"])

    assert {verdict.status for verdict in result.decisions.cells} == {"reject"}
    assert result.decisions.program.status == "not_evaluated"


def test_selected_run_retains_unselected_sequential_family_membership():
    import pandas as pd

    from tests.binary_sequential_cases import DESIGN, ROUTES, analysis_plan, unit_rows

    frame = pd.DataFrame(unit_rows(seed=29, n=180, secondaries=2))
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="user_id",
        group="variant",
        exposure_date="enrollment",
        metrics=[
            MetricSpec(name="purchase", type="conversion"),
            MetricSpec(name="secondary_0", type="conversion"),
            MetricSpec(name="secondary_1", type="conversion"),
        ],
        design=DESIGN,
        plan=analysis_plan(ROUTES["always_valid"].inference, secondaries=2),
    )

    result = run(
        experiment,
        resolver,
        metrics=["purchase"],
        estimands=["itt"],
    )

    assert {cell.identity.key.metric for cell in result.cells} == {"purchase"}
    family_members = {
        key.metric for family in result.run.metadata.scope.families for key in family.members
    }
    assert family_members >= {"purchase", "secondary_0", "secondary_1"}
    missing = [
        member
        for family in result.decisions.families
        for member in family.members
        if not member.identity_hash
    ]
    assert missing and all(member.status == "not_evaluated" for member in missing)


def test_portable_run_result_pickle_round_trip():
    import pickle

    experiment, resolver = _experiment()
    result = run(experiment, resolver)

    restored = pickle.loads(pickle.dumps(result))

    assert restored.cells == result.cells
    assert restored.decisions == result.decisions
    assert restored.run.decision_plan is not None


def test_empty_run_cannot_be_saved_with_registered_refusal(tmp_path):
    from increment.store import ParquetResultsStore

    experiment, resolver = _experiment()
    store = ParquetResultsStore(tmp_path / "results")
    with _frame_read_calls() as reads, pytest.raises(CodedError) as raised:
        run(experiment, resolver, metrics=[], results_store=store)

    assert raised.value.code == "store.run.empty"
    assert "empty" in str(raised.value).lower()
    assert reads == []
    assert not tuple((tmp_path / "results").rglob("*.parquet"))


def test_unadjusted_encouragement_run_keeps_both_late_value_scales():
    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    frame = _frame().assign(uptake=[0, 1] * 20)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="uptake", window_days=1),
            exclusion_restriction=ExclusionRestriction(
                justification="assignment affects outcomes only through uptake"
            ),
        ),
    )
    result = run(experiment, resolver)
    late_cells = [cell for cell in result.cells if cell.identity.key.estimand == "late"]

    assert {cell.identity.key.value_scale for cell in late_cells} == {"absolute", "relative"}


def test_empty_metric_selection_returns_empty_run_result():
    experiment, resolver = _experiment()

    with _frame_read_calls() as reads:
        result = run(experiment, resolver, metrics=[])
    assert reads == []
    assert result.cells == ()
    assert result.effects == ()
    assert result.decisions.cells == ()


def test_run_options_change_execution_identity():
    from increment.estimation.engine import Method

    experiment, resolver = _experiment()
    first = run(experiment, resolver)
    second = run(experiment, resolver, decision_method=Method(name="welch"))
    assert first.run.analysis_spec_hash != second.run.analysis_spec_hash
    assert first.run.execution_identity_hash != second.run.execution_identity_hash


@dataclass
class _HiddenProbabilityFactory:
    probability: float = field(repr=False)

    def __call__(self):
        return _ConfiguredPropensity(self.probability)


def test_runtime_callable_overrides_with_identical_repr_change_run_identity():
    import warnings

    experiment, resolver = _runtime_experiment(_ConfiguredPropensity)
    factories = (_HiddenProbabilityFactory(0.3), _HiddenProbabilityFactory(0.7))
    assert repr(factories[0]) == repr(factories[1])
    hashes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for factory in factories:
            result = run(
                experiment,
                resolver,
                decision_method=Method(name="iptw", propensity_learner=factory),
            )
            hashes.append(result.run.execution_identity_hash)

    assert hashes[0] != hashes[1]


def test_fold_count_method_is_runtime_only():
    from increment.estimation.engine import Method

    frame = _frame().assign(age=[0, 0, 1, 1] * 10)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("age",))
        ),
    )
    result = run(
        experiment,
        resolver,
        decision_method=Method(name="dml", folds=2),
    )
    assert result.run.decision_plan is None
    assert result.run.decision_plan_unavailable_reason == "wire.procedure.runtime_method"
    with pytest.raises(CodedError) as error:
        import pickle

        pickle.dumps(result)
    assert error.value.code == "wire.procedure.runtime_method"


def test_unportable_registered_plan_refuses_storage_before_source_read(tmp_path):
    from increment.estimation.engine import Method
    from increment.store import DecisionLog, ParquetResultsStore

    experiment, resolver = from_frame(
        _frame(),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(name="value", type="mean", decision_method=Method(name="dml", folds=2))
        ],
    )
    log_path = tmp_path / "decisions.jsonl"
    with _frame_read_calls() as reads:
        with pytest.raises(CodedError) as saved:
            run(experiment, resolver, results_store=ParquetResultsStore(tmp_path / "results"))
        with pytest.raises(CodedError) as logged:
            run(experiment, resolver, decision_log=DecisionLog(path=log_path))
    assert saved.value.code == logged.value.code == "wire.procedure.runtime_method"
    assert reads == []
    assert not tuple((tmp_path / "results").rglob("*.parquet"))
    assert not log_path.exists()


@pytest.mark.integration
def test_replayed_checkpoint_asof_preflight_uses_projection_evidence(tmp_path):
    import pyarrow.parquet as pq

    from increment.builders import from_moments
    from tests.test_sequential_public_sources import _frame as sequential_frame
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion", window_days=1)]
    analysis_experiment, analysis_resolver = from_frame(
        sequential_frame([0, 1] * 8, [1, 1] * 8).assign(day=0, exposed=0),
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        observation_end=2,
        plan=gaussian_plan(
            specs,
            law="bernoulli",
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
        ),
    )
    capture_sequential(analysis_experiment, analysis_resolver, finalized=True, as_of=1)
    expected = view_readout(
        analysis_experiment,
        analysis_resolver,
        Scope(view="asof", grain="asof", completion_policy=True),
    )
    path = tmp_path / "checkpoint.parquet"
    export_moments(analysis_experiment, analysis_resolver, path)
    experiment, resolver = from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    scopes = (
        Scope(view="run", grain="total"),
        Scope(view="asof", grain="asof", completion_policy=True),
    )
    requested = Experiment.from_json(replace(experiment, scopes=scopes).to_json())
    capabilities = requested.source_refs[0].capabilities
    assert capabilities is not None and capabilities.grains == frozenset({"total"})
    from tests.parity_harness.comparison import nested_close

    actual = run(requested, resolver)
    asof_cells = [cell for cell in actual.cells if _scope(cell).view == "asof"]
    assert asof_cells
    actual_payloads = [cell.to_payload()["row_payload"]["data"] for cell in asof_cells]
    expected_payloads = [cell.to_payload()["row_payload"]["data"] for cell in expected.as_cells()]
    assert nested_close(actual_payloads, expected_payloads)
    assert any(_scope(cell).view == "run" for cell in actual.cells)
    assert runnable(requested, "outcome")["randomized.unadjusted"].static_refusals == ()
    actual_captured = run(requested, resolver, capture=SequentialCaptureRequest(finalized=True))
    assert nested_close(
        [
            cell.to_payload()["row_payload"]["data"]
            for cell in actual_captured.cells
            if _scope(cell).view == "asof"
        ],
        expected_payloads,
    )
    values = replace(requested, operations={requested.scopes[1].hash(): "run_asof"})
    assert "evidence.grain_unavailable" in {
        refusal.code
        for refusal in runnable(values, "outcome")["randomized.unadjusted"].static_refusals
    }


@pytest.mark.integration
def test_workflow_capture_sequential_checkpoint_feeds_builder_run():
    from tests.test_sequential_public_sources import _frame as sequential_frame
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion", window_days=1)]
    experiment, resolver = from_frame(
        sequential_frame([0, 1] * 8, [1, 1] * 8).assign(day=0, exposed=0),
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
        control="control",
        metrics=specs,
        observation_end=2,
        plan=gaussian_plan(
            specs,
            law="bernoulli",
            unit="unit",
            group="arm",
            date="day",
            exposure_date="exposed",
        ),
    )
    capture_sequential(experiment, resolver, finalized=True, as_of=1)
    scope = Scope(view="asof", grain="asof", completion_policy=True)
    requested = replace(experiment, scopes=(scope,))

    result = run(requested, resolver)

    assert result.cells
    assert {cell.identity.key.ds for cell in result.cells} == {1}


@pytest.mark.parametrize(
    "option", ["experiment_id", "analysis_spec_hash", "execution_identity_hash", "compiled_plan"]
)
def test_experiment_run_refuses_unsupported_execution_options(option):
    experiment, resolver = _experiment()
    with pytest.raises(CodedError) as error:
        options: dict[str, Any] = {option: "requested"}
        run(experiment, resolver, **options)
    assert error.value.code == "kernel.run.invalid_input"


def test_run_rejects_store_request_option_before_persisting(tmp_path):
    from increment.store import ParquetResultsStore

    experiment, resolver = _experiment()
    store = ParquetResultsStore(tmp_path / "results")

    with pytest.raises(CodedError) as error:
        run(replace(experiment, request_options={"store": store}), resolver)

    assert error.value.code == "kernel.run.invalid_input"
    assert not tuple((tmp_path / "results").rglob("*.parquet"))


class _Propensity:
    fitted = 0

    def fit(self, x, y):
        type(self).fitted += 1
        return self

    def predict(self, x):
        return np.full(len(x), 0.5)


def test_analysis_accepts_declared_runtime_learner_and_executes_it():
    frame = _frame().assign(age=[0, 0, 1, 1] * 10)
    metric = MetricSpec(
        name="value",
        type="mean",
        decision_method=Method(name="iptw", propensity_learner=_Propensity),
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[metric],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("age",))
        ),
    )
    assert unit_counts(experiment, resolver) == {"control": 20, "treatment": 20}
    before = _Propensity.fitted
    assert view_readout(experiment, resolver)
    assert _Propensity.fitted > before


def test_analysis_accepts_declared_fold_count_before_portable_export():
    metric = MetricSpec(name="value", type="mean", decision_method=Method(name="dml", folds=2))
    experiment, resolver = from_frame(
        _frame(), shape="summary", unit="unit", group="arm", control="control", metrics=[metric]
    )
    assert unit_counts(experiment, resolver) == {"control": 20, "treatment": 20}
    from increment.errors import IncrementWarning

    with pytest.warns(IncrementWarning) as warnings:
        assert view_readout(experiment, resolver, decision_method=Method.unadjusted())
    assert {
        warning.message.code
        for warning in warnings
        if isinstance(warning.message, IncrementWarning)
    } == {"analysis.run.override_replaces_declared"}


def test_switchback_identity_and_estimator_compatibility():
    from increment import AnalysisPlan
    from increment.builders import from_switchback_frame
    from tests.test_frame_switchback import _assignment, _identification, _rows

    def build(alpha):
        return from_switchback_frame(
            pd.DataFrame(_rows()),
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"mean": "mean"},
            identification=_identification(),
            assignment=_assignment(),
            plan=AnalysisPlan(primary="mean", alpha=alpha),
        )

    experiment, resolver = build(0.05)
    restored = Experiment.from_json(experiment.to_json())
    assert restored.hash() == experiment.hash()
    (cell,) = run(restored, resolver).cells
    assert cell.summary is not None
    assert cell.summary.point == pytest.approx(-30)
    assert experiment.design
    assert experiment.registration != build(0.1)[0].registration
    statuses = runnable(experiment, "mean")
    assert not statuses["switchback.contrast"].static_refusals
    for estimator in ("randomized.cuped", "observational.aipw"):
        assert {error.code for error in statuses[estimator].static_refusals} == {
            "readout.contrast.override"
        }


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:metrics .* have available CUPED covariates:increment.errors.IncrementWarning"
)
def test_definitions_argument_forms_and_unused_source_round_trip(tmp_path, seeded_defs, seeded_con):
    import json
    import shutil

    import yaml

    path = tmp_path / "definitions"
    shutil.copytree(seeded_defs, path)
    source_path = path / "fact_sources.yaml"
    definitions = yaml.safe_load(source_path.read_text())
    unused = dict(definitions["fact_sources"][0])
    unused["name"] = "unused"
    unused["properties"] = []
    unused["dims"] = []
    unused["sql"] = "select * from nonexistent_unused_source"
    unused["facts"] = []
    definitions["fact_sources"].append(unused)
    source_path.write_text(yaml.safe_dump(definitions))
    forms = (
        lambda: from_definitions(path, con=seeded_con, experiment_name="new_onboarding_v2"),
        lambda: from_definitions(
            experiment_name="new_onboarding_v2", definitions_path=path, con=seeded_con
        ),
        lambda: from_definitions(path, seeded_con, experiment_name="new_onboarding_v2"),
    )
    for build in forms:
        experiment, resolver = build()
        assert experiment.source_refs[0].schema == {}
        offline, _ = from_definitions(path)
        assert offline.hash() == experiment.hash()
        assert resolver.verify(offline.source_refs, resolver.resolve(offline.source_refs)) == []
    assert run(experiment, resolver, metrics=["purchase_rate"]).cells
    captured, _ = from_definitions(path)
    assert captured.source_refs[0].schema
    assert not any(name.startswith("unused.") for name in captured.source_refs[0].schema)
    json.loads((path / ".increment-kernel-catalog.json").read_text())
    analysis_experiment, analysis_resolver = from_definitions(
        path, seeded_con, experiment_name="new_onboarding_v2"
    )
    assert isinstance(analysis_resolver, AbstractContextManager)
    with analysis_resolver:
        assert unit_counts(analysis_experiment, analysis_resolver)
        assert view_readout(analysis_experiment, analysis_resolver, metrics=["purchase_rate"])
    source_path.write_text(source_path.read_text() + "\n# changed after binding\n")
    with pytest.raises(CodedError) as stale:
        run(experiment, resolver, metrics=["purchase_rate"])
    assert stale.value.code == "kernel.definitions.catalog_stale"
    with pytest.raises(CodedError) as offline_stale:
        from_definitions(path)
    assert offline_stale.value.code == "kernel.definitions.catalog_stale"


@dataclass
class _ConfiguredPropensity:
    probability: float = 0.5
    fitted: bool = field(init=False)

    def fit(self, x, y):
        self.fitted = True
        return self

    def predict(self, x):
        return np.full(len(x), self.probability)


def _runtime_experiment(factory):
    return from_frame(
        _frame().assign(age=[0, 0, 1, 1] * 10),
        shape="summary",
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(
                name="value",
                type="mean",
                decision_method=Method(name="iptw", propensity_learner=factory),
            ),
        ],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("age",))
        ),
    )


def test_runtime_callable_configurations_cannot_share_a_bound_resolver():
    experiment_a, resolver_a = _runtime_experiment(partial(_ConfiguredPropensity, probability=0.4))
    experiment_b, _ = _runtime_experiment(partial(_ConfiguredPropensity, probability=0.6))
    assert experiment_a.registration != experiment_b.registration
    with pytest.raises(CodedError) as mismatch:
        run(experiment_b, resolver_a)
    assert mismatch.value.code in {
        "kernel.run.execution_unavailable",
        "kernel.resolver.digest_mismatch",
    }
    with pytest.raises(CodedError) as portable:
        experiment_a.hash()
    assert portable.value.code == "kernel.method.implementation_ref_missing"


@pytest.mark.parametrize("with_reference", [False, True])
def test_dataclass_learner_factory_stays_callable_until_export(with_reference):
    experiment, resolver = _runtime_experiment(_ConfiguredPropensity)
    assert run(experiment, resolver).cells
    if with_reference:
        experiment = replace(
            experiment,
            implementation_refs={
                ("value:decision", "propensity_learner"): ImplementationRef(
                    import_path=f"{__name__}._ConfiguredPropensity",
                    version_pin="1",
                    config_digest="configured-default",
                )
            },
        )
    with pytest.raises(CodedError) as portable:
        experiment.to_json()
    assert portable.value.code == "kernel.method.implementation_ref_missing"


def test_runtime_callable_run_result_refuses_portable_storage(tmp_path):
    import pickle

    from increment.store import ParquetResultsStore

    experiment, resolver = _runtime_experiment(_ConfiguredPropensity)
    result = run(experiment, resolver)

    assert result.cells
    assert result.run.decision_plan is None
    assert result.run.decision_plan_unavailable_reason == "wire.procedure.runtime_method"
    with pytest.raises(CodedError) as serialization:
        pickle.dumps(result)
    assert serialization.value.code == "wire.procedure.runtime_method"
    with pytest.raises(CodedError) as portability:
        ParquetResultsStore(tmp_path).put(result.run)
    assert portability.value.code == "wire.procedure.runtime_method"


def test_frozen_request_options_restore_typed_methods_metrics_and_priors():
    from pydantic import TypeAdapter

    from increment import Metric
    from increment.estimators import Normal

    experiment, resolver = _experiment()
    metric = TypeAdapter(Metric).validate_python(dict(experiment.metrics["value"]))
    options = {
        "decision_method": Method.unadjusted(),
        "sensitivity_methods": (),
        "metrics": [metric],
        "prior": Normal(mu=0, sigma=1),
    }
    expected = run(
        experiment,
        resolver,
        decision_method=Method.unadjusted(),
        sensitivity_methods=(),
        metrics=[metric],
        prior=Normal(mu=0, sigma=1),
    ).cells
    requested = replace(experiment, request_options=options)
    restored = Experiment.from_json(requested.to_json())
    actual = run(restored, resolver).cells
    assert [cell.retained_row().model_dump(mode="json") for cell in actual] == [
        cell.retained_row().model_dump(mode="json") for cell in expected
    ]


@pytest.mark.parametrize(
    "view,grain,population,segmented,code",
    [
        ("switchback", "total", "assigned", True, "kernel.run.execution_unavailable"),
        ("daily", "daily", "assigned", False, "facade.analysis.contrast_unavailable"),
        ("asof", "asof", "assigned", False, "facade.analysis.contrast_unavailable"),
        ("switchback", "total", "triggered", False, "facade.analysis.trigger_unsupported"),
        ("asof", "asof", "triggered", False, "facade.analysis.trigger_unsupported"),
    ],
)
def test_switchback_scope_refusals_match_execution(view, grain, population, segmented, code):
    from increment import AnalysisPlan
    from increment.builders import from_switchback_frame
    from increment.evidence import BreakoutIdentity
    from tests.test_frame_switchback import _assignment, _identification, _rows

    experiment, resolver = from_switchback_frame(
        pd.DataFrame(_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        assignment=_assignment(),
        identification=_identification(),
        plan=AnalysisPlan(primary="mean"),
    )
    requested = replace(
        experiment,
        scopes=(
            Scope(
                view=view,
                grain=grain,
                population=population,
                segment=(BreakoutIdentity(property="country", source=None),) if segmented else (),
            ),
        ),
    )
    status = runnable(requested, "mean")["switchback.contrast"]
    assert status.unknown == ()
    assert code in {error.code for error in status.static_refusals}
    with pytest.raises(CodedError) as unsupported:
        run(requested, resolver)
    assert unsupported.value.code == code
    assert any(
        error.code == unsupported.value.code and error.context == unsupported.value.context
        for error in status.static_refusals
    )


@pytest.mark.parametrize("view,grain", [("daily", "daily"), ("asof", "asof")])
def test_frozen_metric_keeps_undeclared_direction_on_day_axis(view, grain):
    from pydantic import TypeAdapter

    from increment import Metric

    frame = pd.concat([_frame().assign(day="2025-01-01"), _frame().assign(day="2025-01-02")])
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    metric = TypeAdapter(Metric).validate_python(
        {"name": "value", "type": "mean", "entity": "unit", "fact": "value", "window_days": 1}
    )
    requested = replace(
        experiment,
        scopes=(Scope(view=view, grain=grain),),
        request_options={"metrics": [metric]},
    )
    restored = Experiment.from_json(requested.to_json())
    direct = run(
        replace(experiment, scopes=requested.scopes),
        resolver,
        metrics=[metric],
    )
    actual = run(restored, resolver)
    from increment.kernel.contracts import metric_hash

    assert actual.cells[0].identity.metric_hash == metric_hash(metric)
    assert actual.run.metadata is not None and direct.run.metadata is not None
    assert actual.run.metadata.scope.snapshot_id == direct.run.metadata.scope.snapshot_id
    if view == "asof":
        assert {str(_scope(cell).through_day) for cell in actual.cells} == {
            "2025-01-01",
            "2025-01-02",
        }
    assert [
        (cell.identity.key.ds, cell.identity.key.metric, cell.summary) for cell in actual.cells
    ] == [(cell.identity.key.ds, cell.identity.key.metric, cell.summary) for cell in direct.cells]
    assert "preferred_direction" not in restored.request_options["metrics"][0]


class _CallablePropensityFactory(BaseModel):
    def __call__(self):
        return _ConfiguredPropensity()


def test_callable_model_factory_survives_preflight_and_frozen_request():
    import warnings

    from increment.errors import IncrementWarning
    from tests.parity_harness.comparison import nested_close

    factory = _CallablePropensityFactory()
    method = Method(name="iptw", propensity_learner=factory)
    experiment, resolver = _runtime_experiment(factory)
    assert not runnable(experiment, "value")["observational.iptw"].static_refusals
    with warnings.catch_warnings(record=True) as direct_warnings:
        warnings.simplefilter("always", IncrementWarning)
        direct = run(experiment, resolver, decision_method=method)
    with warnings.catch_warnings(record=True) as frozen_warnings:
        warnings.simplefilter("always", IncrementWarning)
        frozen = run(replace(experiment, request_options={"decision_method": method}), resolver)
    from tests.warning_codes import warning_codes

    assert warning_codes(direct_warnings) == warning_codes(frozen_warnings)
    assert nested_close(
        [cell.retained_row().model_dump(mode="json") for cell in frozen.cells],
        [cell.retained_row().model_dump(mode="json") for cell in direct.cells],
    )


@contextmanager
def _frame_read_calls():
    reads = []
    previous = sys.getprofile()
    names = {
        "moments",
        "unit_frame",
        "contrast_stats",
        "breakout_moments",
        "_ensure",
        "_bounded_day_panel",
        "_day_identity",
    }

    def observe(frame, event, value):
        if (
            event == "call"
            and frame.f_code.co_name in names
            and frame.f_globals.get("__name__", "").startswith("increment.")
        ):
            reads.append(frame.f_code.co_name)
        if previous is not None:
            previous(frame, event, value)

    try:
        sys.setprofile(observe)
        yield reads
    finally:
        sys.setprofile(previous)


def test_arm_experiment_refuses_switchback_scope_before_reads():
    experiment, resolver = _experiment()
    requested = replace(experiment, scopes=(Scope(view="switchback", grain="total"),))
    with _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(requested, resolver)
    assert reads == []
    assert error.value.code == "kernel.scope.family_mismatch"
    assert {
        refusal.code
        for refusal in runnable(requested, "value")["randomized.unadjusted"].static_refusals
    } == {error.value.code}


def test_unavailable_breakout_selection_refuses_alike_before_reads():
    from increment.evidence import BreakoutIdentity

    experiment, resolver = _experiment()
    selection = Scope(
        view="breakout",
        grain="total",
        segment=(BreakoutIdentity(property="country", source=None),),
    )
    requested = replace(experiment, scopes=(selection,))
    with _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(requested, resolver)
    assert reads == []
    assert error.value.code == "evidence.breakout_unavailable"
    assert [
        (refusal.code, refusal.context)
        for refusal in runnable(requested, "value")["randomized.unadjusted"].static_refusals
    ] == [(error.value.code, error.value.context)]


def test_unadjusted_preflight_preserves_declared_conversion_inference():
    frame = _frame().assign(value=[0, 0, 1, 1] * 10, store=lambda data: data["unit"])
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        cluster="store",
        metrics=[
            MetricSpec(
                name="value",
                type="conversion",
                decision_method=Method.unadjusted(conversion_inference="finite_sample"),
            )
        ],
    )
    with _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(experiment, resolver)
    assert reads == []
    assert error.value.code == "estimation.binomial.finite_sample_unavailable"
    assert {
        refusal.code
        for refusal in runnable(experiment, "value")["randomized.unadjusted"].static_refusals
    } == {error.value.code}


def test_multicovariate_preflight_requires_covariate_configuration():
    experiment, _ = _experiment()
    with _frame_read_calls() as reads:
        status = runnable(experiment, "value")["randomized.cuped_multicovariate"]
    assert reads == []
    assert {refusal.code for refusal in status.static_refusals} == {
        "estimation.engine.method_covariates"
    }


def _plain_builder(kind, *, conversion=False):
    if kind == "switchback":
        from increment import AnalysisPlan
        from increment.builders import from_switchback_frame
        from tests.test_frame_switchback import _assignment, _identification, _rows

        frame = pd.DataFrame(_rows()).rename(columns={"mean": "value"})
        frame["other"] = frame["value"] + 2
        return from_switchback_frame(
            frame,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"value": "mean", "other": "mean"},
            identification=_identification(),
            assignment=_assignment(),
            plan=AnalysisPlan(primary=("value", "other")),
        )
    frame = _frame()
    if conversion:
        from increment.estimators import Normal

        frame["value"] = [index % 2 for index in range(len(frame))]
        metrics = [
            MetricSpec(
                name="value",
                type="conversion",
                prior=Normal(mu=0, sigma=1),
                decision_method=Method.unadjusted(conversion_inference="finite_sample"),
            ),
            MetricSpec(name="other", type="mean"),
        ]
    else:
        metrics = {"value": "mean", "other": "mean"}
    frame["other"] = frame["value"] + 2
    frame["country"] = ["US", "CA", "CA", "US"] * 10
    frame["region"] = ["west", "east"] * 20
    if kind == "panel":
        frame = pd.concat([frame.assign(day="2025-01-01"), frame.assign(day="2025-01-02")])
        return from_frame(
            frame,
            shape="panel",
            date="day",
            unit="unit",
            group="arm",
            control="control",
            metrics=metrics,
            breakouts=["country", "region"],
        )
    return from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=metrics,
        breakouts=["country", "region"],
    )


@pytest.mark.slow
@settings(max_examples=30, deadline=None)
@example(
    kind="switchback",
    view="switchback",
    operation="run",
    prior="normal",
    estimands=None,
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="summary",
    view="daily",
    operation="run",
    prior="unset",
    estimands=None,
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="summary",
    view="breakout",
    operation="run",
    prior="unset",
    estimands=None,
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="summary",
    view="daily",
    operation="run_daily",
    prior="unset",
    estimands=(),
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="panel",
    view="daily",
    operation="run_daily_lift",
    prior="unset",
    estimands=(),
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="panel",
    view="daily",
    operation="run_daily_lift",
    prior="unset",
    estimands=(),
    metrics=(),
    breakout_dimension=None,
    descriptive_daily_conversion=False,
)
@example(
    kind="panel",
    view="daily",
    operation="run_daily",
    prior="unset",
    estimands=None,
    metrics=None,
    breakout_dimension=None,
    descriptive_daily_conversion=True,
)
@example(
    kind="summary",
    view="breakout",
    operation="run_breakout",
    prior="unset",
    estimands=None,
    metrics=None,
    breakout_dimension="region",
    descriptive_daily_conversion=False,
)
@given(
    kind=st.sampled_from(["summary", "panel", "switchback"]),
    view=st.sampled_from(["run", "daily", "asof", "breakout", "switchback"]),
    operation=st.one_of(
        st.none(),
        st.sampled_from(
            [
                "run",
                "run_daily",
                "run_daily_lift",
                "run_asof",
                "run_asof_lift",
                "run_breakout",
            ]
        ),
    ),
    prior=st.sampled_from(["unset", "normal", "clear"]),
    estimands=st.sampled_from([None, (), ("itt",), ("late",), ("compliance",), ("unknown",)]),
    metrics=st.sampled_from([None, (), ("value",), ("other",), ("value", "other")]),
    breakout_dimension=st.sampled_from([None, "country", "region"]),
    descriptive_daily_conversion=st.booleans(),
)
def test_builder_preflight_matches_execution_before_reads(
    kind,
    view,
    operation,
    prior,
    estimands,
    metrics,
    breakout_dimension,
    descriptive_daily_conversion,
):
    import warnings

    from increment.errors import IncrementWarning
    from increment.estimators import Normal
    from increment.evidence import BreakoutIdentity

    conversion = (
        descriptive_daily_conversion
        and kind == "panel"
        and view == "daily"
        and operation == "run_daily"
    )
    experiment, resolver = _plain_builder(kind, conversion=conversion)
    scope = Scope(
        view=view,
        grain=view if view in {"daily", "asof"} else "total",
        segment=(
            (BreakoutIdentity(breakout_dimension or "country", None),) if view == "breakout" else ()
        ),
    )
    options = {}
    if prior != "unset":
        options["prior"] = Normal(mu=0, sigma=1) if prior == "normal" else None
    if estimands is not None:
        options["estimands"] = estimands
    if metrics is not None:
        options["metrics"] = metrics
    requested = Experiment.from_json(
        replace(
            experiment,
            scopes=(scope,),
            operations={} if operation is None else {scope.hash(): operation},
            request_options=options,
        ).to_json()
    )
    estimator = "switchback.contrast" if kind == "switchback" else "randomized.unadjusted"
    with _frame_read_calls() as preflight_reads:
        status = runnable(requested, "value")[estimator]
    assert preflight_reads == []
    with _frame_read_calls() as execution_reads, warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        try:
            run(requested, resolver)
        except CodedError as error:
            expected = {error.code}
        else:
            expected = set()
    if status.static_refusals:
        assert execution_reads == []
    assert {error.code for error in status.static_refusals} <= expected


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (("value", "value"), "facade.analysis_config.duplicate_metric_name"),
        (("not_declared",), "facade.analysis_config.unknown_metric_declared"),
    ],
)
def test_switchback_preflight_reports_execution_metric_selection_refusal(metrics, expected):
    experiment, resolver = _plain_builder("switchback")
    requested = replace(experiment, request_options={"metrics": metrics})

    with pytest.raises(CodedError) as execution_error:
        run(requested, resolver)

    refusal_codes = {
        error.code for error in runnable(requested, "value")["switchback.contrast"].static_refusals
    }
    assert execution_error.value.code == expected
    assert refusal_codes == {expected}


def test_daily_lift_preflight_checks_finite_sample_conversion_compatibility():
    from increment.estimators import Normal

    frame = pd.DataFrame(
        [
            {
                "unit": unit,
                "arm": "control" if unit < 20 else "treatment",
                "day": f"2025-01-0{day}",
                "value": (unit + day) % 2,
            }
            for unit in range(40)
            for day in (1, 2)
        ]
    )
    metric = MetricSpec(
        name="value",
        type="conversion",
        prior=Normal(mu=0, sigma=1),
        decision_method=Method.unadjusted(conversion_inference="finite_sample"),
    )
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[metric],
    )
    scope = Scope(view="daily", grain="daily")
    requested = replace(
        experiment,
        scopes=(scope,),
        operations={scope.hash(): "run_daily_lift"},
        request_options={"prior": Normal(mu=0, sigma=1), "metrics": ["value"]},
    )

    with _frame_read_calls() as reads, pytest.raises(CodedError) as execution_error:
        run(requested, resolver)

    refusal_codes = {
        error.code
        for error in runnable(requested, "value")["randomized.unadjusted"].static_refusals
    }
    assert execution_error.value.code == "estimation.binomial.finite_sample_unavailable"
    assert reads == []
    assert refusal_codes == {execution_error.value.code}


@pytest.mark.parametrize(
    ("view", "grain", "operation"),
    [
        ("daily", "daily", "run_daily"),
        ("asof", "asof", "run_asof"),
    ],
)
def test_value_only_day_axis_builder_operations_roundtrip_and_return_value_scopes(
    view, grain, operation
):
    frame = pd.concat([_frame().assign(day="2025-01-01"), _frame().assign(day="2025-01-02")])
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    scope = Scope(view=view, grain=grain)
    requested = replace(
        experiment,
        scopes=(scope,),
        operations={scope.hash(): operation},
    )
    original_hashes = (
        requested.hash(),
        requested.analysis_spec_hash,
        requested.execution_identity_hash,
    )
    requested_scope = requested.scopes[0]
    restored = type(requested).from_json(requested.to_json())

    assert requested_scope.values is True
    assert (
        restored.hash(),
        restored.analysis_spec_hash,
        restored.execution_identity_hash,
    ) == original_hashes

    result = run(restored, resolver, views=(requested_scope,))

    assert result.cells
    assert all(_scope(cell).view == view for cell in result.cells)
    assert all(_scope(cell).values for cell in result.cells)
    assert all(cell.identity.key.ds is not None for cell in result.cells)
    assert all(cell.identity.key.value_scale == "absolute" for cell in result.cells)
    assert all(cell.identity.estimator_id is None for cell in result.cells)
    assert result.decisions.run_id == result.run.run_id
    returned_scope = _scope(result.cells[0]).model_copy(
        update={"through_day": None, "time_basis": "calendar"}
    )
    selected = run(restored.with_views(returned_scope), resolver)
    assert selected.cells
    assert all(_scope(cell).values for cell in selected.cells)
    assert all(cell.identity.key.value_scale == "absolute" for cell in selected.cells)
    direct_experiment, direct_resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
    )
    direct_values = view_readout(
        direct_experiment, direct_resolver, Scope(view=view, grain=grain, values=True)
    )
    direct_cell = direct_values.as_cells()[0]
    assert direct_cell.identity.scope.values is True
    direct_scope = direct_cell.identity.scope.model_copy(
        update={"through_day": None, "time_basis": "calendar"}
    )
    declarative = run(restored.with_views(direct_scope), resolver)
    assert declarative.cells
    assert all(_scope(cell).values for cell in declarative.cells)
    assert all(cell.identity.key.value_scale == "absolute" for cell in declarative.cells)
    direct_lifts = view_readout(direct_experiment, direct_resolver, Scope(view=view, grain=grain))
    lift_cell = direct_lifts.as_cells()[0]
    assert lift_cell.identity.scope.values is False
    lift_scope = lift_cell.identity.scope.model_copy(
        update={"through_day": None, "time_basis": "calendar"}
    )
    declarative_lift = run(restored.with_views(lift_scope), resolver)
    assert declarative_lift.cells
    assert all(_scope(cell).values is False for cell in declarative_lift.cells)
    assert all(cell.identity.key.value_scale == "relative" for cell in declarative_lift.cells)


def test_runtime_callable_override_run_result_refuses_portable_storage(tmp_path):
    import pickle

    from increment.store import ParquetResultsStore

    frame = _frame().assign(age=[0, 0, 1, 1] * 10)
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        metrics=["value"],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("age",))
        ),
    )

    result = run(
        experiment,
        resolver,
        decision_method=Method(name="iptw", propensity_learner=_ConfiguredPropensity),
    )

    assert result.cells
    assert result.run.decision_plan is None
    assert result.run.decision_plan_unavailable_reason == "wire.procedure.runtime_method"
    with pytest.raises(CodedError) as serialization:
        pickle.dumps(result)
    assert serialization.value.code == "wire.procedure.runtime_method"
    with pytest.raises(CodedError) as portability:
        ParquetResultsStore(tmp_path).put(result.run)
    assert portability.value.code == "wire.procedure.runtime_method"


@pytest.mark.parametrize(("effect", "expected"), [(10.0, "reject"), (0.0, "retain")])
def test_switchback_run_decides_from_contrast_procedure(effect, expected):
    from increment import AnalysisPlan, from_switchback_frame
    from tests.test_frame_switchback import _assignment, _identification

    rows = []
    for unit in range(12):
        for cycle in range(4):
            order = (
                ("control", "treatment") if (unit + cycle) % 2 == 0 else ("treatment", "control")
            )
            control_value = 10.0 + 0.3 * unit + 0.1 * cycle
            contrast = effect + (unit % 3 - 1) * 0.1 + (cycle % 2 * 2 - 1) * 0.02
            for period, group in enumerate(order):
                value = control_value + (contrast if group == "treatment" else 0.0)
                for step in range(3):
                    rows.append(
                        {
                            "unit": f"u{unit}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "mean": 999.0 if step == 0 else value,
                        }
                    )
    experiment, resolver = from_switchback_frame(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=_assignment(p_ct=0.5),
        plan=AnalysisPlan(primary="mean"),
    )

    result = run(experiment, resolver)

    assert result.decisions.cells[0].status == expected
    assert result.decisions.program.status == "block"
    assert "integrity.unsupported_assignment" in result.decisions.program.reason_codes


@pytest.mark.slow
def test_switchback_builder_returns_in_memory_run_result_and_refuses_portable_storage(tmp_path):
    import pickle

    from increment import from_switchback_frame
    from increment.store import DecisionLog, ParquetResultsStore
    from tests.test_frame_switchback import _assignment, _identification, _rows

    experiment, resolver = from_switchback_frame(
        pd.DataFrame(_rows()),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"mean": "mean"},
        identification=_identification(),
        assignment=_assignment(),
    )

    result = run(experiment, resolver)

    assert result.cells
    assert result.run.decision_plan is None
    assert result.run.decision_plan_unavailable_reason == "wire.procedure.contrast_context"
    assert result.decisions.run_id == result.run.run_id
    with pytest.raises(CodedError) as serialization:
        pickle.dumps(result)
    assert serialization.value.code == "wire.procedure.contrast_context"
    with pytest.raises(CodedError) as portability:
        ParquetResultsStore(tmp_path).put(result.run)
    assert portability.value.code == "wire.procedure.contrast_context"
    log_path = tmp_path / "decisions.jsonl"
    runs = tmp_path / "runs"
    with _frame_read_calls() as reads, pytest.raises(CodedError) as logged:
        run(
            experiment,
            resolver,
            results_store=ParquetResultsStore(runs),
            decision_log=DecisionLog(path=log_path),
        )
    assert logged.value.code == "wire.procedure.contrast_context"
    assert reads == []
    assert not log_path.exists()
    assert not tuple(runs.rglob("*.parquet"))


def test_switchback_partial_run_selection_cannot_pass_omitted_guardrail():
    from increment import AnalysisPlan, ExperimentMetric, MetricSpec, from_switchback_frame
    from tests.test_frame_switchback import _assignment, _identification

    rows = []
    for unit in range(12):
        for cycle in range(4):
            order = (
                ("control", "treatment") if (unit + cycle) % 2 == 0 else ("treatment", "control")
            )
            baseline = 10.0 + 0.3 * unit + 0.1 * cycle
            for period, group in enumerate(order):
                for step in range(3):
                    is_observed = step > 0
                    treatment = group == "treatment"
                    rows.append(
                        {
                            "unit": f"u{unit}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": 999.0
                            if not is_observed
                            else baseline + (10.0 if treatment else 0.0),
                            "guard_a": 999.0
                            if not is_observed
                            else baseline + (1.0 if treatment else 0.0),
                            "guard_b": 999.0
                            if not is_observed
                            else baseline - (1.0 if treatment else 0.0),
                        }
                    )
    experiment, resolver = from_switchback_frame(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=[
            MetricSpec(name="value", type="mean"),
            MetricSpec(name="guard_a", type="mean", preferred_direction="decrease"),
            MetricSpec(name="guard_b", type="mean", preferred_direction="decrease"),
        ],
        identification=_identification(),
        assignment=_assignment(),
        plan=AnalysisPlan(
            primary="value",
            guardrails=[
                ExperimentMetric(metric="guard_a", margin_abs=0.5),
                ExperimentMetric(metric="guard_b", margin_abs=0.5),
            ],
        ),
    )

    result = run(
        experiment,
        resolver,
        metrics=["value", "guard_a"],
    )

    assert result.decisions.program.status == "not_evaluated"
    assert "decision.guardrail.not_evaluated" in result.decisions.program.reason_codes
    assert "integrity.unsupported_assignment" in result.decisions.program.reason_codes
