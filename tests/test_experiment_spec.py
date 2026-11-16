from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from fractions import Fraction
from types import SimpleNamespace
from typing import cast

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


def _assert_restored_fixed_triggered_consumer(capabilities):
    from increment import runnable

    experiment, reader = from_frame(
        pd.DataFrame(
            {
                "unit": ("a", "b", "c", "d"),
                "group": ("control", "control", "treat", "treat"),
                "orders": (1.0, 2.0, 3.0, 4.0),
            }
        ),
        shape="summary",
        unit="unit",
        group="group",
        control="control",
        metrics={"orders": "mean"},
    )
    reader.close()
    reference = experiment.source_refs[0]
    assert reference.capabilities is not None
    for supported in (capabilities.fixed_triggered, False):
        restored = Experiment.from_json(
            dataclasses.replace(
                experiment,
                source_refs=(
                    dataclasses.replace(
                        reference,
                        capabilities=dataclasses.replace(
                            reference.capabilities,
                            fixed_triggered=supported,
                            populations=frozenset({"assigned", "triggered"}),
                            count_grains=frozenset({"unit", "triggered"}),
                        ),
                    ),
                ),
                analysis_context={**experiment.analysis_context, "trigger_name": "checkout"},
                views=(View(view="run", grain="total", population="triggered"),),
            ).to_json()
        )
        refusals = tuple(
            refusal
            for result in runnable(restored, "orders").values()
            for refusal in result.static_refusals
            if refusal.code == "facade.analysis.trigger_unsupported"
        )
        assert bool(refusals) is not supported
        assert all(
            refusal.context
            == {
                "method": "run",
                "experiment": experiment.experiment_id,
                "trigger": None,
                "route": "moments",
                "supported_sources": ("from_definitions", "from_unit_day_artifact"),
            }
            for refusal in refusals
        )


def _assert_windowed_quantile_consumer(capabilities, legacy_capabilities):
    from increment import runnable
    from increment.evidence import BreakoutIdentity

    rows = pd.DataFrame(
        {
            "unit": ("a", "b", "c", "d"),
            "group": ("control", "control", "treat", "treat"),
            "ds": ("2025-01-01",) * 4,
            "p50_orders": (1.0, 2.0, 3.0, 4.0),
            "country": ("us",) * 4,
        }
    )
    experiment, reader = from_frame(
        rows,
        shape="panel",
        unit="unit",
        group="group",
        date="ds",
        exposure_date="ds",
        control="control",
        metrics=[MetricSpec(name="p50_orders", type="quantile", quantile=0.5)],
        breakouts=["country"],
    )
    reader.close()
    payload = json.loads(experiment.to_json())
    payload["metrics"]["p50_orders"]["window_days"] = 7
    payload["analysis_context"]["configs"][0]["metric"]["window_days"] = 7
    windowed = Experiment.from_json(json.dumps(payload)).with_views(
        View(view="breakout", segment=(BreakoutIdentity("country", None),))
    )
    for declaration, expected in ((capabilities, True), (legacy_capabilities, False)):
        restored = Experiment.from_json(
            dataclasses.replace(
                windowed,
                source_refs=(
                    dataclasses.replace(
                        windowed.source_refs[0],
                        capabilities=dataclasses.replace(declaration, breakouts=("country",)),
                    ),
                ),
            ).to_json()
        )
        refusals = tuple(
            refusal
            for result in runnable(restored, "p50_orders").values()
            for refusal in result.static_refusals
            if refusal.code == "frame.metric.window_days_supported"
        )
        assert bool(refusals) is expected
        assert all(refusal.context == {"name": "p50_orders"} for refusal in refusals)


def _assert_restored_temporal_capability_case(
    field, metric, value, code, experiment, semantic, segment, reader
):
    from increment import run, runnable
    from increment.errors import CapabilityError, InvalidRequestError
    from increment.semantics.models import MeanMetric

    capabilities = experiment.source_refs[0].capabilities
    assert capabilities is not None
    ad_hoc = MeanMetric(name="ad_hoc", entity="unit", fact="y", aggregation="sum")
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
    options = (
        {"metrics": [ad_hoc.model_dump(mode="json")]}
        if field == "dimensioned_daily_metrics"
        else {}
    )
    recorded = dataclasses.replace(
        experiment,
        source_refs=(dataclasses.replace(experiment.source_refs[0], capabilities=declared),),
        views=(scope,),
        analysis_context=context,
        request_options=options,
    )
    restored = Experiment.from_json(recorded.to_json())

    if field == "dimensioned_daily_metrics":
        with pytest.raises(InvalidRequestError) as invalid_view:
            View(
                view="daily",
                grain="daily",
                values=True,
                segment=segment,
                metrics=(ad_hoc,),
            )
        assert invalid_view.value.code == "kernel.scope.invalid"
        assert invalid_view.value.context == {
            "field": "view/grain/through_day/segment/metrics/values"
        }

        expected = (
            InvalidRequestError,
            "facade.analysis_config.unknown_metric_declared",
            {"caller": "run_daily", "unknown_names": ("ad_hoc",), "declared_names": ("y",)},
        )
        refusals = tuple(
            refusal
            for result in runnable(restored, ad_hoc).values()
            for refusal in result.static_refusals
        )
        assert refusals
        assert all(type(refusal) is expected[0] for refusal in refusals)
        assert all(refusal.code == expected[1] for refusal in refusals)
        assert all(refusal.context == expected[2] for refusal in refusals), [
            dict(refusal.context) for refusal in refusals
        ]
        with pytest.raises(expected[0]) as runtime:
            run(experiment.with_views(scope), reader, views=(scope,), metrics=[ad_hoc])
        assert runtime.value.code == expected[1]
        assert runtime.value.context == expected[2]
        return

    refusals = tuple(
        refusal
        for result in runnable(restored, metric).values()
        for refusal in result.static_refusals
    )
    relevant_codes = {
        "temporal_quantiles": "breakout.quantile",
        "temporal_clusters": "facade.analysis.clustered_day_axis",
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


def _restored_temporal_case(field, rows):
    from datetime import UTC, datetime

    import polars as pl

    from increment import from_frame
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
                    field, metric, value, code, experiment, semantic, segment, reader
                )
        finally:
            reader.close()


def _assert_restored_resolver_mismatches(restored, provider):
    from increment.evidence import BoundResolver

    reference = restored.source_refs[0]
    refusals = BoundResolver(cast(EvidenceProvider, provider)).verify((reference,))
    assert [refusal.code for refusal in refusals] == ["kernel.provider.capability_mismatch"]


def _assert_legacy_temporal_defaults(capabilities):

    native_moments = dataclasses.replace(
        capabilities,
        kind="moments",
        operations=frozenset({"moments_source"}),
        value_scale_overrides=None,
        fixed_triggered=None,
        windowed_breakout_quantiles=None,
        temporal_quantiles=None,
        temporal_clusters=None,
        dimensioned_daily_metrics=None,
    )
    native = SourceReference(
        name="native", uri="warehouse://native", capabilities=native_moments
    ).capabilities
    assert native is not None
    assert native.value_scale_overrides is False
    assert native.fixed_triggered is True
    assert native.windowed_breakout_quantiles is True
    assert native.temporal_quantiles is False
    assert native.temporal_clusters is False
    assert native.dimensioned_daily_metrics == "declared"

    native_definitions = dataclasses.replace(native_moments, kind="definitions")
    definitions = SourceReference(
        name="definitions", uri="warehouse://definitions", capabilities=native_definitions
    ).capabilities
    assert definitions is not None
    assert definitions.temporal_quantiles is False
    assert definitions.temporal_clusters is False
    assert definitions.dimensioned_daily_metrics == "declared"

    artifact_moments = dataclasses.replace(
        native_moments, operations=frozenset({"moments_source", "artifact_experiment"})
    )
    artifact = SourceReference(
        name="artifact", uri="warehouse://artifact", capabilities=artifact_moments
    ).capabilities
    assert artifact is not None
    assert artifact.temporal_quantiles is True
    assert artifact.temporal_clusters is False
    assert artifact.dimensioned_daily_metrics == "any"
    assert artifact.value_scale_overrides is True
    assert artifact.fixed_triggered is True
    assert artifact.windowed_breakout_quantiles is False


def test_capability_semantics_round_trip_and_preserve_resolver_mismatch():

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
        assert getattr(restored_capabilities, field) == nonlegacy
        if field == "windowed_breakout_quantiles":
            _assert_windowed_quantile_consumer(restored_capabilities, capabilities)
        _assert_restored_resolver_mismatches(restored, provider)
        if field == "fixed_triggered":
            _assert_restored_fixed_triggered_consumer(restored_capabilities)

    _assert_legacy_temporal_defaults(capabilities)
    _assert_restored_temporal_runnable_consumer()
