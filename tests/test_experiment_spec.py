from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from increment import (
    CupedCovariate,
    Experiment,
    Method,
    MetricSpec,
    SourceReference,
    from_frame,
)
from increment.errors import CodedError
from increment.evidence import EvidenceCapabilities, EvidenceProvider, ScalarEvidenceDeclaration
from increment.kernel.contracts import ImplementationRef, Registration, View


class _ConstantPropensity:
    """A module-level learner factory, so pickle stores it by reference."""

    def fit(self, x: np.ndarray, d: np.ndarray) -> None:
        return None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), 0.5)


def _recorded_experiment() -> Experiment:
    scope = View(view="daily", grain="daily")
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total", "daily"}),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=ScalarEvidenceDeclaration(
            bindings={"orders": (CupedCovariate(name="pre_orders", kind="continuous"),)},
            moments_only=False,
            recipe_fixed=False,
            recipe_configs=(),
            anonymous_metrics=frozenset(),
            method_evidence=False,
            breakout_evidence=False,
            covariate_grains=None,
        ),
        available_covariates=frozenset({"pre_orders"}),
    )
    return Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="warehouse://events",
                schema={"events.orders": "double"},
                capabilities=capabilities,
            ),
        ),
        views=(scope,),
        implementation_refs={
            ("orders:decision", "propensity_learner"): ImplementationRef(
                import_path="models.propensity:build", version_pin="1", config_digest="p"
            ),
        },
        operations={scope.hash(): "run_daily_lift"},
    )


def test_experiment_round_trips_canonical_json_and_hash():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"),),
        implementation_refs={
            ("orders:decision", "propensity_learner"): ImplementationRef(
                import_path="models.propensity:build", version_pin="1", config_digest="p"
            ),
        },
    )

    encoded = experiment.to_json()
    restored = Experiment.from_json(encoded)

    assert restored.implementation_refs == experiment.implementation_refs
    assert json.loads(encoded) == json.loads(restored.to_json())
    assert restored.hash() == experiment.hash()


def test_recorded_schema_does_not_change_experiment_identity():
    def declared(schema):
        return Experiment(
            experiment_id="checkout",
            design={"mechanism": "randomized"},
            metrics={"orders": {"kind": "mean"}},
            registration=Registration(registration_hash="registration-hash"),
            source_refs=(SourceReference(name="events", uri="warehouse://events", schema=schema),),
            views=(View(view="run", grain="total"),),
        )

    unrecorded, empty, recorded = (declared(schema) for schema in (None, {}, {"t.c": "string"}))

    assert unrecorded.hash() == empty.hash() == recorded.hash()
    assert unrecorded.analysis_spec_hash == empty.analysis_spec_hash == recorded.analysis_spec_hash
    assert unrecorded.to_json() != recorded.to_json()


def test_experiment_survives_pickle_and_deepcopy():
    experiment = _recorded_experiment()

    for restored in (pickle.loads(pickle.dumps(experiment)), copy.deepcopy(experiment)):
        assert restored == experiment
        assert restored.hash() == experiment.hash()
        assert restored.to_json() == experiment.to_json()
        assert type(next(iter(restored.implementation_refs.values()))) is ImplementationRef


def test_equal_experiments_are_one_dict_key():
    experiment = _recorded_experiment()
    half, point_five = (
        dataclasses.replace(_recorded_experiment(), request_options={"null_lift": value})
        for value in (Fraction(1, 2), 0.5)
    )

    assert hash(experiment) == hash(copy.deepcopy(experiment))
    assert {experiment: 1}[pickle.loads(pickle.dumps(experiment))] == 1
    assert half == point_five
    assert half.hash() != point_five.hash()
    assert {half: 1}[point_five] == 1
    assert {point_five: 1}[half] == 1


def test_non_portable_experiment_round_trips():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "observational"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="frame://events"),),
        views=(View(view="run", grain="total"),),
        implementation_refs={
            ("orders:decision", "propensity_learner"): ImplementationRef(
                import_path=f"{__name__}._ConstantPropensity", version_pin="1", config_digest="c"
            ),
        },
        request_options={
            "decision_method": Method(name="iptw", propensity_learner=_ConstantPropensity)
        },
    )

    with pytest.raises(CodedError) as portable:
        experiment.to_json()
    assert portable.value.code == "kernel.method.implementation_ref_missing"
    assert copy.deepcopy(experiment) == experiment
    assert pickle.loads(pickle.dumps(experiment)) == experiment


def test_experiment_deep_copy_leaves_its_reader_alone():
    frame = pd.DataFrame(
        {
            "unit": [f"u{index}" for index in range(8)],
            "arm": ["control", "treatment"] * 4,
            "value": [float(index) for index in range(8)],
        }
    )
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="value", type="mean")],
    )
    memo: dict[int, object] = {}
    assert copy.deepcopy(experiment, memo) == experiment
    assert id(reader) not in memo


@pytest.mark.parametrize(
    ("design", "shown"),
    (
        ({"mechanism": "randomized"}, "randomized"),
        ({"kind": "switchback", "identification": {"mechanism": "randomized"}}, "switchback"),
    ),
)
def test_experiment_repr_is_one_bounded_line_naming_what_it_runs(design, shown):
    metrics = {f"metric_{index}_{'x' * 120}": {"kind": "mean"} for index in range(200)}
    experiment = Experiment(
        experiment_id="checkout_" + "y" * 500,
        design=design,
        metrics=metrics,
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"), View(view="breakout", grain="total")),
    )

    display = repr(experiment)

    assert len(display) <= 400
    assert "\n" not in display and "mappingproxy" not in display
    assert "checkout_" in display and repr(shown) in display
    assert "metric_0_" in display and "metric_199_" not in display
    assert "views=2" in display
    assert experiment.hash()[:12] in display


def test_experiment_repr_says_not_portable_instead_of_refusing():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"learner": len}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"),),
    )

    assert "not portable" in repr(experiment)


def test_experiment_scopes_and_nested_definitions_are_immutable():
    design = {"mechanism": "randomized"}
    experiment = Experiment(
        experiment_id="checkout",
        design=design,
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"),),
    )
    original_hash = experiment.hash()
    design["mechanism"] = "observational"

    assert experiment.design["mechanism"] == "randomized"
    assert experiment.hash() == original_hash
    attribute = "experiment_id"
    with pytest.raises((AttributeError, TypeError)):
        setattr(experiment, attribute, "changed")


def test_compile_emits_run_inventory_and_one_request_per_scope():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(
            View(view="run", grain="total"),
            View(view="daily", grain="daily"),
        ),
    )

    inventory, requests = experiment.compile()

    assert inventory.population == "assigned"
    assert len(requests) == len(experiment.views) == 2
    assert tuple(request.scope for request in requests) == experiment.views
    assert all(request.registration_hash == "registration-hash" for request in requests)
    assert all(request.source_ref == "events" for request in requests)


def test_runnable_marks_missing_capabilities_unknown_per_estimator():
    from increment import runnable

    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        views=(View(view="run", grain="total"),),
    )

    result = runnable(experiment, "orders")

    assert result
    assert result["randomized.unadjusted"].unknown == ("events.capabilities",)
    assert result["randomized.unadjusted"].static_refusals == ()


def test_bound_resolver_refuses_missing_digest_provenance():
    from types import SimpleNamespace

    from increment.evidence import BoundResolver, EvidenceCapabilities

    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=None,
    )
    provider = SimpleNamespace(evidence=capabilities, context=SimpleNamespace())
    resolver = BoundResolver(cast(EvidenceProvider, provider))
    source = SourceReference(
        name="events",
        uri="frame://events",
        expected_digest="sha256:expected",
        capabilities=capabilities,
    )

    refusals = resolver.verify((source,))

    assert [refusal.code for refusal in refusals] == ["kernel.provider.digest_mismatch"]


def test_bound_resolver_refuses_runtime_capability_mismatch():
    from types import SimpleNamespace

    from increment.evidence import BoundResolver, EvidenceCapabilities

    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=None,
    )
    changed = dataclasses.replace(capabilities, unit_rows=False)
    provider = SimpleNamespace(evidence=changed, context=SimpleNamespace(source_digest=None))
    resolver = BoundResolver(cast(EvidenceProvider, provider))
    source = SourceReference(
        name="events",
        uri="frame://events",
        capabilities=capabilities,
    )

    refusals = resolver.verify((source,))

    assert [refusal.code for refusal in refusals] == ["kernel.provider.capability_mismatch"]


@pytest.mark.parametrize("question", ["prior", "estimands", "operation"])
def test_specification_identity_distinguishes_request_declarations(question):
    from increment.estimators import Normal

    scope = View(view="daily", grain="daily")
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "encouragement"},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="frame://events"),),
        views=(scope,),
    )
    explicit_default = dataclasses.replace(experiment, operations={scope.hash(): "run_daily_lift"})
    assert explicit_default.analysis_spec_hash == experiment.analysis_spec_hash
    assert explicit_default.execution_identity_hash == experiment.execution_identity_hash
    changed = (
        dataclasses.replace(experiment, views=(View(view="daily", grain="daily", values=True),))
        if question == "operation"
        else dataclasses.replace(
            experiment,
            request_options={question: Normal(mu=0, sigma=1) if question == "prior" else ("itt",)},
        )
    )
    restored = Experiment.from_json(changed.to_json())

    assert restored.analysis_spec_hash == changed.analysis_spec_hash
    assert restored.execution_identity_hash == changed.execution_identity_hash
    assert changed.analysis_spec_hash != experiment.analysis_spec_hash
    assert changed.execution_identity_hash != experiment.execution_identity_hash


def _capability_round_trip(experiment, reference, capabilities, field, legacy, nonlegacy):
    explicit_legacy = dataclasses.replace(capabilities, **{field: legacy})
    legacy_experiment = dataclasses.replace(
        experiment,
        source_refs=(dataclasses.replace(reference, capabilities=explicit_legacy),),
    )
    assert legacy_experiment.to_json() == experiment.to_json()
    assert legacy_experiment.hash() == experiment.hash()

    changed = dataclasses.replace(capabilities, **{field: nonlegacy})
    changed_experiment = dataclasses.replace(
        experiment,
        source_refs=(dataclasses.replace(reference, capabilities=changed),),
    )
    assert changed_experiment.to_json() != experiment.to_json()
    assert changed_experiment.hash() != experiment.hash()
    if field == "fixed_triggered":
        changed_experiment = dataclasses.replace(
            changed_experiment,
            analysis_context={**experiment.analysis_context, "trigger_name": "checkout"},
            views=(View(view="run", grain="total", population="triggered"),),
        )
    restored = Experiment.from_json(changed_experiment.to_json())
    restored_capabilities = restored.source_refs[0].capabilities
    assert restored_capabilities is not None
    assert restored.to_json() == changed_experiment.to_json()
    assert restored.hash() == changed_experiment.hash()
    assert getattr(restored_capabilities, field) == nonlegacy
    return changed_experiment, restored, restored_capabilities


def _assert_windowed_quantile_consumer(capabilities, legacy_capabilities):
    from increment._analysis_config import UNSET
    from increment._breakout_readouts import BreakoutReadouts, BreakoutRequest
    from increment.semantics.models import QuantileMetric

    metric = QuantileMetric(
        name="p50_orders",
        entity="unit",
        fact="orders",
        quantile=0.5,
        window_days=7,
    )
    request = BreakoutRequest(
        metrics=(metric,),
        decision_method=UNSET,
        sensitivity_methods=UNSET,
        prior=UNSET,
        correction=None,
        q=None,
        population="assigned",
    )
    context = SimpleNamespace(plan=SimpleNamespace(inference=None))
    artifact = BreakoutReadouts(
        src=cast(Any, SimpleNamespace(evidence=capabilities, context=context)),
        experiment=None,
        breakouts=(),
    )
    with pytest.raises(CodedError) as raised:
        artifact.validate_request(request)
    assert raised.value.code == "frame.metric.window_days_supported"
    assert raised.value.context == {"name": "p50_orders"}
    native = BreakoutReadouts(
        src=cast(Any, SimpleNamespace(evidence=legacy_capabilities, context=context)),
        experiment=None,
        breakouts=(),
    )
    assert native.validate_request(request) is request


def _assert_temporal_capability_consumer(field, capabilities):
    from increment._day_axis import validate_day_axis
    from increment._evidence_types import DayAxisRequest
    from increment.errors import CapabilityError, InvalidRequestError
    from increment.semantics.models import MeanMetric, QuantileMetric

    source = SimpleNamespace(
        evidence=dataclasses.replace(
            capabilities,
            grains=frozenset({"daily", "asof"}),
            breakouts=("country",),
        ),
        context=SimpleNamespace(trigger_name=None, metrics=()),
    )
    metrics = {
        "temporal_quantiles": QuantileMetric(
            name="p50", entity="unit", fact="orders", quantile=0.5
        ),
        "temporal_clusters": MeanMetric(
            name="orders", entity="unit", fact="orders", aggregation="sum"
        ),
        "dimensioned_daily_metrics": MeanMetric(
            name="ad_hoc", entity="unit", fact="orders", aggregation="sum"
        ),
    }
    request = DayAxisRequest(
        caller="run_daily",
        grain="daily",
        metrics=(metrics[field],),
        dimension="country" if field == "dimensioned_daily_metrics" else None,
    )
    experiment = SimpleNamespace(
        name="recorded",
        trigger=None,
        cluster="household" if field == "temporal_clusters" else None,
        breakouts=(SimpleNamespace(property="country", source=None),),
    )
    expected = {
        "temporal_quantiles": (
            "breakout.quantile",
            CapabilityError,
            {
                "names": ("p50",),
                "method": "run_daily()",
                "reason": "quantiles do not decompose into per-day moments",
                "remedy": "Use run().",
            },
        ),
        "temporal_clusters": (
            "facade.analysis.clustered_day_axis",
            CapabilityError,
            {"method": "run_daily", "experiment": "recorded", "cluster": "household"},
        ),
        "dimensioned_daily_metrics": (
            "facade.analysis.undeclared_metric_for_dimension",
            InvalidRequestError,
            {"method": "run_daily", "dimension": "country", "undeclared": ("ad_hoc",)},
        ),
    }[field]
    code, error_type, context = expected
    with pytest.raises(CodedError) as raised:
        validate_day_axis(
            request,
            src=source,
            design=None,
            plan=cast(Any, SimpleNamespace(inference=None)),
            route="moments",
            experiment=cast(Any, experiment),
        )
    assert type(raised.value) is error_type
    assert raised.value.code == code
    assert raised.value.context == context

    supported = {
        "temporal_quantiles": True,
        "temporal_clusters": True,
        "dimensioned_daily_metrics": "any",
    }[field]
    source.evidence = dataclasses.replace(source.evidence, **{field: supported})
    validate_day_axis(
        request,
        src=source,
        design=None,
        plan=cast(Any, SimpleNamespace(inference=None)),
        route="moments",
        experiment=cast(Any, experiment),
    )


def _restored_temporal_case(field, rows):
    from datetime import UTC, datetime

    import polars as pl

    from increment import from_frame
    from increment._metric_specs import MetricSpec
    from increment.evidence import BreakoutIdentity
    from increment.semantics.models import AnalysisPlan, Breakout
    from increment.semantics.models import Experiment as Definition

    if field == "temporal_quantiles":
        experiment, reader = from_frame(
            pl.DataFrame(rows),
            shape="panel",
            unit="unit",
            group="group",
            date="ds",
            control="control",
            metrics=[MetricSpec(name="p50", type="quantile", quantile=0.5)],
            breakouts=["country"],
            exposure_date="ds",
        )
        semantic = Definition(
            name="checkout",
            exposure="assignment",
            unit="unit",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            control_group="control",
            plan=AnalysisPlan(),
            breakouts=(Breakout(property="country"),),
        )
        segment = ()
    elif field == "temporal_clusters":
        cluster_rows = [
            {
                "unit": row["unit"],
                "group": row["group"],
                "cluster": row["cluster"],
                "y": row["y"],
            }
            for row in rows
        ]
        experiment, reader = from_frame(
            pl.DataFrame(cluster_rows),
            shape="summary",
            unit="unit",
            group="group",
            control="control",
            metrics={"y": "mean"},
            cluster="cluster",
        )
        semantic = Definition(
            name="checkout",
            exposure="assignment",
            unit="unit",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            control_group="control",
            plan=AnalysisPlan(),
            cluster="cluster",
        )
        segment = ()
    else:
        experiment, reader = from_frame(
            pl.DataFrame(rows),
            shape="panel",
            unit="unit",
            group="group",
            date="ds",
            control="control",
            metrics={"y": "mean"},
            breakouts=["country"],
            exposure_date="ds",
        )
        semantic = Definition(
            name="checkout",
            exposure="assignment",
            unit="unit",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            control_group="control",
            plan=AnalysisPlan(),
            breakouts=(Breakout(property="country"),),
        )
        segment = (BreakoutIdentity("country", None),)

    return experiment, reader, semantic, segment


def _assert_restored_temporal_capability_case(
    field, metric, value, code, experiment, semantic, segment
):
    from increment import runnable
    from increment.errors import CapabilityError

    capabilities = experiment.source_refs[0].capabilities
    assert capabilities is not None
    scope = View(view="daily", grain="daily", values=True, segment=segment)
    context = {
        **dict(experiment.analysis_context),
        "experiment": semantic.model_dump(mode="json"),
    }
    declared = dataclasses.replace(
        capabilities,
        grains=frozenset({"total", "daily", "asof"}),
        **{field: value},
    )
    recorded = dataclasses.replace(
        experiment,
        source_refs=(dataclasses.replace(experiment.source_refs[0], capabilities=declared),),
        views=(scope,),
        analysis_context=context,
    )
    restored = Experiment.from_json(recorded.to_json())
    refusals = tuple(
        refusal
        for result in runnable(restored, metric).values()
        for refusal in result.static_refusals
    )
    relevant_codes = {
        "temporal_quantiles": "breakout.quantile",
        "temporal_clusters": "facade.analysis.clustered_day_axis",
        "dimensioned_daily_metrics": "facade.analysis.undeclared_metric_for_dimension",
    }
    if code is None:
        assert not any(refusal.code == relevant_codes[field] for refusal in refusals)
        return

    matching = [refusal for refusal in refusals if refusal.code == code]
    assert matching
    expected_contexts = {
        "breakout.quantile": (
            CapabilityError,
            {
                "names": ("p50",),
                "method": "run_daily()",
                "reason": "quantiles do not decompose into per-day moments",
                "remedy": "Use run().",
            },
        ),
        "facade.analysis.clustered_day_axis": (
            CapabilityError,
            {"method": "run_daily", "experiment": "checkout", "cluster": "cluster"},
        ),
    }
    refusal_type, expected_context = expected_contexts[code]
    assert all(type(refusal) is refusal_type for refusal in matching)
    assert all(refusal.context == expected_context for refusal in matching), [
        dict(refusal.context) for refusal in matching
    ]


def _assert_restored_temporal_runnable_consumer():
    rows = [
        {
            "unit": f"u{i}",
            "group": "control" if i < 4 else "treat",
            "ds": "2025-01-01",
            "p50": float(i),
            "y": float(i),
            "country": "us" if i % 2 == 0 else "gb",
            "cluster": f"c{i // 4}",
        }
        for i in range(8)
    ]
    cases = (
        ("temporal_quantiles", "p50", True, False),
        ("temporal_clusters", "y", True, False),
        ("dimensioned_daily_metrics", "y", "any", "declared"),
    )
    refused_codes = {
        "temporal_quantiles": "breakout.quantile",
        "temporal_clusters": "facade.analysis.clustered_day_axis",
        "dimensioned_daily_metrics": None,
    }
    for field, metric, supported, refused in cases:
        experiment, reader, semantic, segment = _restored_temporal_case(field, rows)
        try:
            for value, code in (
                (supported, None),
                (refused, refused_codes[field]),
            ):
                _assert_restored_temporal_capability_case(
                    field, metric, value, code, experiment, semantic, segment
                )
        finally:
            reader.close()


def _assert_restored_resolver_mismatches(restored, provider):
    from increment._binding import ExperimentReader
    from increment.evidence import BoundResolver

    reference = restored.source_refs[0]
    refusals = BoundResolver(cast(EvidenceProvider, provider)).verify((reference,))
    assert [refusal.code for refusal in refusals] == ["kernel.provider.capability_mismatch"]
    reader = object.__new__(ExperimentReader)
    reader._closed = False
    reader._bound = cast(Any, SimpleNamespace(state=SimpleNamespace(source=provider)))
    reader._experiment = restored
    refusals = reader._verify((reference,), provider)
    assert [refusal.code for refusal in refusals] == ["kernel.provider.capability_mismatch"]


def _assert_legacy_temporal_defaults(capabilities):
    from increment._source_types import effective_capabilities

    native_moments = dataclasses.replace(
        capabilities, kind="moments", operations=frozenset({"moments_source"})
    )
    native = effective_capabilities(native_moments)
    assert native.value_scale_overrides is False
    assert native.fixed_triggered is False
    assert native.windowed_breakout_quantiles is True
    assert native.temporal_quantiles is False
    assert native.temporal_clusters is True
    assert native.dimensioned_daily_metrics == "declared"

    native_definitions = dataclasses.replace(native_moments, kind="definitions")
    definitions = effective_capabilities(native_definitions)
    assert definitions.temporal_quantiles is False
    assert definitions.temporal_clusters is False
    assert definitions.dimensioned_daily_metrics == "declared"

    artifact_moments = dataclasses.replace(
        native_moments, operations=frozenset({"moments_source", "artifact_experiment"})
    )
    artifact = effective_capabilities(artifact_moments)
    assert artifact.temporal_quantiles is True
    assert artifact.temporal_clusters is False
    assert artifact.dimensioned_daily_metrics == "any"
    assert artifact.value_scale_overrides is True
    assert artifact.fixed_triggered is True
    assert artifact.windowed_breakout_quantiles is False


def test_capability_semantics_round_trip_and_preserve_resolver_mismatch():
    from increment._source_types import effective_capabilities

    experiment = _recorded_experiment()
    assert experiment.hash() == "2e2ae2d21857c4ff5b07d3f69fdcf14e6d5866faef4a8169983cbae04dad4278"
    reference = experiment.source_refs[0]
    capabilities = reference.capabilities
    assert capabilities is not None

    provider = SimpleNamespace(
        evidence=capabilities,
        context=SimpleNamespace(source_digest=None),
    )
    for field, legacy, nonlegacy in (
        ("value_scale_overrides", True, False),
        ("fixed_triggered", False, True),
        ("windowed_breakout_quantiles", True, False),
        ("temporal_quantiles", True, False),
        ("temporal_clusters", True, False),
        ("dimensioned_daily_metrics", "any", "declared"),
    ):
        _, restored, restored_capabilities = _capability_round_trip(
            experiment, reference, capabilities, field, legacy, nonlegacy
        )
        assert getattr(effective_capabilities(restored_capabilities), field) == nonlegacy
        if field == "windowed_breakout_quantiles":
            _assert_windowed_quantile_consumer(restored_capabilities, capabilities)
        if field in {"temporal_quantiles", "temporal_clusters", "dimensioned_daily_metrics"}:
            _assert_temporal_capability_consumer(field, restored_capabilities)
        _assert_restored_resolver_mismatches(restored, provider)
        if field == "fixed_triggered":
            from increment._runnability import scope_refusals

            assert "facade.analysis.trigger_unsupported" not in {
                refusal.code for refusal in scope_refusals(restored, restored.views[0])
            }

    _assert_legacy_temporal_defaults(capabilities)
    _assert_restored_temporal_runnable_consumer()
