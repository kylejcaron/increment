from __future__ import annotations

from dataclasses import replace
from typing import cast

from increment.evidence import EvidenceCapabilities, EvidenceProvider
from increment.kernel.contracts import Registration, Scope
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import resolve, runnable


def test_resolve_refuses_scope_grain_missing_from_declared_capabilities():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(name="events", uri="frame://events", capabilities=capabilities),
        ),
        scopes=(Scope(view="daily", grain="daily"),),
    )

    refusals = resolve("randomized.unadjusted", experiment, "orders")

    assert tuple(refusal.code for refusal in refusals) == ("evidence.grain_unavailable",)

    assert tuple(
        refusal.code
        for refusal in runnable(experiment, "orders")["randomized.unadjusted"].static_refusals
    ) == ("evidence.grain_unavailable",)


def test_runnable_uses_recorded_capabilities_after_json_restore():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total", "daily"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    supported = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
                capabilities=capabilities,
            ),
        ),
        scopes=(Scope(view="daily", grain="daily"),),
    )
    unsupported = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
                capabilities=replace(capabilities, grains=frozenset({"total"})),
            ),
        ),
        scopes=(Scope(view="daily", grain="daily"),),
    )

    supported_loaded = Experiment.from_json(supported.to_json())
    unsupported_loaded = Experiment.from_json(unsupported.to_json())

    assert supported_loaded.source_refs[0].schema == unsupported_loaded.source_refs[0].schema
    assert runnable(supported_loaded, "orders")["randomized.unadjusted"].static_refusals == ()
    assert tuple(
        refusal.code
        for refusal in runnable(unsupported_loaded, "orders")[
            "randomized.unadjusted"
        ].static_refusals
    ) == ("evidence.grain_unavailable",)


def test_runnable_reports_unknown_without_recorded_capabilities_after_json_restore():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
            ),
        ),
        scopes=(
            Scope(
                view="daily",
                grain="daily",
                segment=({"property": "country", "source": None},),
            ),
        ),
    )

    restored = Experiment.from_json(experiment.to_json())

    assert runnable(restored, "orders")["randomized.unadjusted"].unknown == ("events.capabilities",)
    assert runnable(restored, "orders")["randomized.unadjusted"].static_refusals == ()


def test_runnable_names_authored_source_and_selection_for_unavailable_breakouts():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=("country",),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(name="events", uri="frame://events", capabilities=capabilities),
        ),
        scopes=(
            Scope(
                view="breakout",
                grain="total",
                segment=({"property": "unknown", "source": None},),
            ),
        ),
    )

    for candidate in (experiment, Experiment.from_json(experiment.to_json())):
        statuses = runnable(candidate, "orders")
        assert [
            (refusal.code, dict(refusal.context))
            for refusal in statuses["randomized.unadjusted"].static_refusals
        ] == [
            (
                "evidence.breakout_unavailable",
                {
                    "constructor": "events",
                    "requested": ("unknown",),
                    "offered": ("country",),
                    "route": "use a source with the requested breakout dimensions",
                },
            )
        ]
        assert {
            refusal.context["constructor"]
            for status in statuses.values()
            for refusal in status.static_refusals
            if "constructor" in refusal.context
        } == {"events"}


def test_multicovariate_runnability_uses_alternative_evidence_routes():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=False,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )

    def experiment_for(declared):
        return Experiment(
            experiment_id="checkout",
            design={"mechanism": "randomized"},
            metrics={"orders": {"kind": "mean"}},
            registration=Registration(registration_hash="registration-hash"),
            source_refs=(
                SourceReference(
                    name="events",
                    uri="frame://events",
                    capabilities=declared,
                ),
            ),
            scopes=(Scope(view="run", grain="total"),),
        )

    refused = runnable(experiment_for(capabilities), "orders")[
        "randomized.cuped_multicovariate"
    ].static_refusals
    alternative = runnable(experiment_for(replace(capabilities, unit_rows=True)), "orders")[
        "randomized.cuped_multicovariate"
    ].static_refusals

    assert tuple(refusal.code for refusal in refused) == (
        "evidence.operation_unavailable",
        "evidence.unit_rows_unavailable",
    )
    assert alternative == ()


def test_bound_resolver_refuses_changed_source_digest_during_verify():
    from types import SimpleNamespace

    from increment.kernel.resolver import BoundResolver

    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    provider = cast(
        "EvidenceProvider",
        SimpleNamespace(
            evidence=capabilities,
            context=SimpleNamespace(source_digest="source-current"),
        ),
    )
    source = SourceReference(
        name="events",
        uri="frame://events",
        expected_digest="source-recorded",
        capabilities=capabilities,
    )

    errors = BoundResolver(provider).verify((source,), provider)

    assert tuple(error.code for error in errors) == ("kernel.resolver.digest_mismatch",)
