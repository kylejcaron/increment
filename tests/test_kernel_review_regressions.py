from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from increment import Analysis, Method, MetricSpec, run
from increment.builders import from_frame
from increment.errors import CodedError
from increment.kernel.contracts import ImplementationRef, Scope
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import runnable
from increment.semantics.design import AdjustmentSet, Observational
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


def test_builder_refuses_changed_metric_configuration():
    experiment, resolver = _experiment()
    metric = dict(experiment.metrics["value"], name="other")
    with pytest.raises(CodedError) as error:
        run(replace(experiment, metrics={"other": metric}), resolver)
    assert error.value.code == "kernel.run.execution_unavailable"


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
    assert {row.ds for row in run(daily, resolver)} == {
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


@pytest.mark.parametrize("clear_declared_prior", [False, True])
def test_clustered_preflight_applies_saved_prior_override(clear_declared_prior):
    from increment import Normal
    from increment.errors import IncrementWarning
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
    from increment import Normal

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

    assert [row.metric for row in run(experiment, resolver)] == ["value"]
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
        assert rows and {row.estimand for row in rows} == {"itt"}
        assert status.static_refusals == ()
    else:
        with pytest.raises(CodedError) as error:
            run(requested, resolver)
        assert error.value.code == "readout.estimands.unknown"
        assert error.value.code in {refusal.code for refusal in status.static_refusals}


@pytest.mark.integration
def test_replayed_checkpoint_asof_preflight_uses_projection_evidence(tmp_path):
    import pyarrow.parquet as pq

    from increment.builders import from_moments
    from tests.test_sequential_public_sources import _frame as sequential_frame
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion", window_days=1)]
    analysis = Analysis.from_unit_panel(
        sequential_frame([0, 1] * 8, [1, 1] * 8).assign(day=0, exposed=0),
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
    analysis.capture_sequential(finalized=True, as_of=1)
    expected = analysis.run_asof_lift(completed_windows_only=True)
    path = tmp_path / "checkpoint.parquet"
    analysis.export(path)
    experiment, resolver = from_moments(
        pq.read_table(path).to_pylist(), metrics=specs, control="control"
    )
    requested = Experiment.from_json(
        replace(
            experiment,
            scopes=(Scope(view="asof", grain="asof", completion_policy=True),),
        ).to_json()
    )

    capabilities = requested.source_refs[0].capabilities
    assert capabilities is not None and capabilities.grains == frozenset({"total"})
    assert list(run(requested, resolver)) == list(expected)
    assert runnable(requested, "outcome")["randomized.unadjusted"].static_refusals == ()
    values = replace(requested, operations={requested.scopes[0].hash(): "run_asof"})
    assert "evidence.grain_unavailable" in {
        refusal.code
        for refusal in runnable(values, "outcome")["randomized.unadjusted"].static_refusals
    }


@pytest.mark.integration
def test_builder_resolver_can_capture_sequential_checkpoint_before_run():
    from increment.builders import from_frame
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
    resolver.capture_sequential(finalized=True, as_of=1)
    scope = Scope(view="asof", grain="asof", completion_policy=True)
    requested = replace(experiment, scopes=(scope,))

    rows = run(requested, resolver)

    assert rows
    assert {row.ds for row in rows} == {1}


@pytest.mark.parametrize(
    "option",
    [
        "results_store",
        "decision_log",
        "run_id",
        "experiment_id",
        "analysis_spec_hash",
        "execution_identity_hash",
        "compiled_plan",
    ],
)
def test_experiment_run_refuses_unsupported_execution_options(option):
    experiment, resolver = _experiment()
    with pytest.raises(CodedError) as error:
        options: dict[str, Any] = {option: "requested"}
        run(experiment, resolver, **options)
    assert error.value.code == "kernel.run.invalid_input"


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
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[metric],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("age",))
        ),
    )
    assert analysis.unit_counts() == {"control": 20, "treatment": 20}
    before = _Propensity.fitted
    assert analysis.run()
    assert _Propensity.fitted > before


def test_analysis_accepts_declared_fold_count_before_portable_export():
    metric = MetricSpec(name="value", type="mean", decision_method=Method(name="dml", folds=2))
    analysis = Analysis.from_unit_summary(
        _frame(), unit="unit", group="arm", control="control", metrics=[metric]
    )
    assert analysis.unit_counts() == {"control": 20, "treatment": 20}
    from increment.errors import IncrementWarning

    with pytest.warns(IncrementWarning) as warnings:
        assert analysis.run(decision_method=Method.unadjusted())
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
        )[0]

    experiment = build(0.05)
    assert experiment.design
    assert experiment.registration != build(0.1).registration
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

    from increment.builders import from_definitions

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
        lambda: from_definitions("new_onboarding_v2", path, con=seeded_con),
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
    assert run(experiment, resolver, analysis_options={"metrics": ["purchase_rate"]})
    captured, _ = from_definitions(path)
    assert captured.source_refs[0].schema
    assert not any(name.startswith("unused.") for name in captured.source_refs[0].schema)
    json.loads((path / ".increment-kernel-catalog.json").read_text())
    with Analysis.from_definitions(
        experiment_name="new_onboarding_v2", definitions_path=path, con=seeded_con
    ) as analysis:
        assert analysis.unit_counts()
        assert analysis.run(metrics=["purchase_rate"])
    source_path.write_text(source_path.read_text() + "\n# changed after binding\n")
    with pytest.raises(CodedError) as stale:
        run(experiment, resolver, analysis_options={"metrics": ["purchase_rate"]})
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
    assert run(experiment, resolver)
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


def test_frozen_request_options_restore_typed_methods_metrics_and_priors():
    from pydantic import TypeAdapter

    from increment import Metric, Normal

    experiment, resolver = _experiment()
    metric = TypeAdapter(Metric).validate_python(dict(experiment.metrics["value"]))
    options = {
        "decision_method": Method.unadjusted(),
        "sensitivity_methods": (),
        "metrics": [metric],
        "prior": Normal(mu=0, sigma=1),
    }
    expected = run(experiment, resolver, analysis_options=options)
    requested = replace(experiment, request_options=options)
    restored = Experiment.from_json(requested.to_json())
    assert list(run(restored, resolver)) == list(expected)


@pytest.mark.parametrize(
    "view,grain,population,segmented,code",
    [
        ("switchback", "total", "assigned", True, "kernel.run.execution_unavailable"),
        ("daily", "daily", "assigned", False, "facade.analysis.contrast_unavailable"),
        ("asof", "asof", "assigned", False, "facade.analysis.contrast_unavailable"),
        ("switchback", "total", "triggered", False, "facade.analysis.trigger_unsupported"),
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
        analysis_options={"metrics": [metric]},
    )
    actual = run(restored, resolver)
    assert actual.metadata is not None and direct.metadata is not None
    assert actual.metadata.scope.snapshot_id == direct.metadata.scope.snapshot_id
    assert [(row.ds, row.metric, row.lift) for row in actual] == [
        (row.ds, row.metric, row.lift) for row in direct
    ]
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
        direct = run(experiment, resolver, analysis_options={"decision_method": method})
    with warnings.catch_warnings(record=True) as frozen_warnings:
        warnings.simplefilter("always", IncrementWarning)
        frozen = run(replace(experiment, request_options={"decision_method": method}), resolver)
    from tests.warning_codes import warning_codes

    assert warning_codes(direct_warnings) == warning_codes(frozen_warnings)
    assert nested_close(
        [row.model_dump(mode="json") for row in frozen],
        [row.model_dump(mode="json") for row in direct],
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
        from increment import Normal

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
    options = {}
    if kind == "panel":
        frame = pd.concat([frame.assign(day="2025-01-01"), frame.assign(day="2025-01-02")])
        options["date"] = "day"
    return from_frame(
        frame,
        shape="panel" if kind == "panel" else "summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=metrics,
        breakouts=["country", "region"],
        **options,
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
    from increment import Normal
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
    with _frame_read_calls() as execution_reads:
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
    from increment import Normal

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
