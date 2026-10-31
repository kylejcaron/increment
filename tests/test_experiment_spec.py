from __future__ import annotations

import dataclasses
import json
from typing import cast

import pytest

from increment import Experiment, SourceReference
from increment.evidence import EvidenceProvider
from increment.kernel.contracts import Registration, Scope


def test_top_level_run_resolves_and_executes_experiment():
    from increment import run

    class Adapter:
        def __init__(self):
            self.calls = []

        def resolve(self, source_refs, ctx=None):
            self.calls.append(("resolve", tuple(source_refs)))
            return object()

        def verify(self, source_refs, provider):
            self.calls.append(("verify", tuple(source_refs), provider))
            return []

        def run_experiment(self, exp, provider):
            self.calls.append(("run", exp, provider))
            return "completed"

    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        scopes=(Scope(view="run", grain="total"),),
    )
    adapter = Adapter()

    assert run(experiment, adapter) == "completed"
    assert [call[0] for call in adapter.calls] == ["resolve", "verify", "run"]


def test_experiment_round_trips_canonical_json_and_hash():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        scopes=(Scope(view="run", grain="total"),),
    )

    encoded = experiment.to_json()
    restored = Experiment.from_json(encoded)

    assert json.loads(encoded) == json.loads(restored.to_json())
    assert restored.hash() == experiment.hash()


def test_experiment_scopes_and_nested_definitions_are_immutable():
    design = {"mechanism": "randomized"}
    experiment = Experiment(
        experiment_id="checkout",
        design=design,
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="warehouse://events"),),
        scopes=(Scope(view="run", grain="total"),),
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
        scopes=(
            Scope(view="run", grain="total"),
            Scope(view="daily", grain="daily"),
        ),
    )

    inventory, requests = experiment.compile()

    assert inventory.population == "assigned"
    assert len(requests) == len(experiment.scopes) == 2
    assert tuple(request.scope for request in requests) == experiment.scopes
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
        scopes=(Scope(view="run", grain="total"),),
    )

    result = runnable(experiment, "orders")

    assert result
    assert result["randomized.unadjusted"].unknown == ("events.capabilities",)
    assert result["randomized.unadjusted"].static_refusals == ()


def test_bound_resolver_refuses_missing_digest_provenance():
    from types import SimpleNamespace

    from increment import BoundResolver
    from increment.evidence import EvidenceCapabilities

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

    assert [refusal.code for refusal in refusals] == ["kernel.resolver.digest_mismatch"]


def test_bound_resolver_refuses_runtime_capability_mismatch():
    from types import SimpleNamespace

    from increment import BoundResolver
    from increment.evidence import EvidenceCapabilities

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

    assert [refusal.code for refusal in refusals] == ["kernel.resolver.capability_mismatch"]


@pytest.mark.parametrize("question", ["prior", "estimands", "operation"])
def test_specification_identity_distinguishes_request_declarations(question):
    from increment.estimators import Normal

    scope = Scope(view="daily", grain="daily")
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "encouragement"},
        metrics={},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(SourceReference(name="events", uri="frame://events"),),
        scopes=(scope,),
    )
    explicit_default = dataclasses.replace(experiment, operations={scope.hash(): "run_daily_lift"})
    assert explicit_default.analysis_spec_hash == experiment.analysis_spec_hash
    assert explicit_default.execution_identity_hash == experiment.execution_identity_hash
    changed = dataclasses.replace(
        experiment,
        **(
            {"operations": {scope.hash(): "run_daily"}}
            if question == "operation"
            else {
                "request_options": {
                    question: Normal(mu=0, sigma=1) if question == "prior" else ("itt",)
                }
            }
        ),
    )
    restored = Experiment.from_json(changed.to_json())

    assert restored.analysis_spec_hash == changed.analysis_spec_hash
    assert restored.execution_identity_hash == changed.execution_identity_hash
    assert changed.analysis_spec_hash != experiment.analysis_spec_hash
    assert changed.execution_identity_hash != experiment.execution_identity_hash
