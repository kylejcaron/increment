from __future__ import annotations

import copy
import copyreg
import dataclasses
import io
import json
import pickle
from fractions import Fraction
from types import MappingProxyType
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


def test_reader_stays_unpicklable_and_experiment_copies_leave_it_alone():
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

    # Cached fixtures register MappingProxyType with copyreg; pickle without that registration.
    pickler = pickle.Pickler(io.BytesIO())
    pickler.dispatch_table = {
        kind: reducer
        for kind, reducer in copyreg.dispatch_table.items()
        if kind is not MappingProxyType
    }
    with pytest.raises(TypeError):
        pickler.dump(reader)
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
