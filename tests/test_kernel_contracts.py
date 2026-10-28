from __future__ import annotations

import copy
import hashlib
import json
import pickle
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, date
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from increment.declarations import Method
from increment.errors import CodedError
from increment.kernel.contracts import (
    AnalysisRequest,
    CellIdentity,
    ImplementationRef,
    MethodIdentity,
    Registration,
    Scope,
    analysis_spec_hash,
    execution_identity_hash,
    method_config_hash,
)


def test_daily_day_and_cumulative_through_day_have_different_scope_hashes():
    day = date(2026, 1, 5)
    daily = Scope(view="daily", grain="daily", through_day=day)
    cumulative = Scope(view="asof", grain="asof", through_day=day)
    assert daily.hash() != cumulative.hash()


def test_scope_and_implementation_ref_round_trip_with_stable_hash():
    values = (
        Scope(view="daily", grain="daily", through_day=date(2026, 1, 5)),
        ImplementationRef(
            import_path="package.model:Estimator", version_pin="1.2", config_digest="abc"
        ),
    )
    for value in values:
        decoded = type(value).from_payload(
            json.loads(json.dumps(value.canonical_payload(), sort_keys=True).encode())
        )
        assert decoded == value
        assert (
            decoded.hash() == value.hash() == type(value).model_validate(value.model_dump()).hash()
        )


def test_inconsistent_scope_refuses_with_code():
    bad = (
        {"view": "run", "grain": "daily"},
        {"view": "run", "grain": "total", "through_day": date(2026, 1, 5)},
        {"view": "breakout", "grain": "total"},
    )
    for fields in bad:
        with pytest.raises(CodedError) as raised:
            Scope.model_validate(cast(Any, fields))
        assert raised.value.code == "kernel.scope.invalid"
        assert pickle.loads(pickle.dumps(raised.value)).code == raised.value.code
        assert copy.deepcopy(raised.value).code == raised.value.code


@pytest.mark.integration
def test_contracts_import_stays_inside_the_permitted_set():
    code = "import sys, increment.kernel.contracts; print('\\n'.join(sorted(m for m in sys.modules if m.startswith('increment'))))"
    loaded = subprocess.check_output([sys.executable, "-c", code], text=True).splitlines()
    assert not any(
        name.startswith(("increment.estimation", "increment.readouts", "increment.query"))
        for name in loaded
    )


def test_cell_identity_round_trips_with_stable_hash():
    from increment.declarations import CellKey

    key = CellKey(
        kind="arm",
        metric="orders",
        method="naive",
        method_role="decision",
        estimand="itt",
        analysis_population="assigned",
        group_id="treatment",
        ds=date(2026, 1, 5),
    )
    value = CellIdentity(
        key=key,
        source_snapshot_id="snap",
        experiment_id="exp",
        analysis_spec_hash="a",
        metric_hash="m",
        estimator_id="e",
        method_config_hash="c",
        method_instance_id="i",
        scope=Scope(view="run", grain="total"),
    )
    decoded = CellIdentity.from_payload(
        json.loads(json.dumps(value.canonical_payload(), sort_keys=True).encode())
    )
    assert decoded == value and decoded.hash() == value.hash()


def test_cell_identity_preserves_iso_string_day_labels_across_round_trips():
    from increment.declarations import CellKey

    payload = {
        "kind": "arm",
        "metric": "orders",
        "method": "naive",
        "method_role": "decision",
        "estimand": "itt",
        "analysis_population": "assigned",
        "group_id": "treatment",
        "ds": {"label": "2026-01-03"},
    }
    key = CellKey.model_validate(payload)
    identity = CellIdentity(key=key, source_snapshot_id="snap")
    restored = CellIdentity.from_payload(
        json.loads(json.dumps(identity.canonical_payload(), sort_keys=True))
    )
    calendar_day = CellIdentity(
        key=CellKey.model_validate({**payload, "ds": date(2026, 1, 3)}),
        source_snapshot_id="snap",
    )

    assert identity.key.ds == "2026-01-03"
    assert restored.key.ds == "2026-01-03"
    assert restored == identity
    assert restored.hash() == identity.hash()
    assert identity.hash() != calendar_day.hash()


def test_snapshot_id_is_part_of_identity():
    from increment.declarations import CellKey

    key = CellKey(
        kind="arm",
        metric="orders",
        method=None,
        method_role=None,
        estimand=None,
        analysis_population="assigned",
        group_id="treatment",
    )
    assert (
        CellIdentity(key=key, source_snapshot_id="a").hash()
        != CellIdentity(key=key, source_snapshot_id="b").hash()
    )
    assert (
        CellIdentity.from_payload(
            {"key": key.model_dump(mode="json"), "source_snapshot_id": None}
        ).source_snapshot_id
        is None
    )

    from types import SimpleNamespace

    row = SimpleNamespace(
        metric="orders",
        group_id="treatment",
        method=None,
        method_role=None,
        estimand=None,
        analysis_population="assigned",
    )
    assert CellIdentity.from_row(row).source_snapshot_id is None


def test_method_config_hash_ignores_name_and_tracks_every_other_field():
    method_instance_id = "m"
    assert method_config_hash(
        Method(name="alpha"), method_instance_id=method_instance_id
    ) == method_config_hash(Method(name="beta"), method_instance_id=method_instance_id)
    base = Method(name="alpha")
    assert method_config_hash(base, method_instance_id=method_instance_id) != method_config_hash(
        Method(name="alpha", variance_reduction="cuped"), method_instance_id=method_instance_id
    )
    assert method_config_hash(
        Method(name="alpha", variance_reduction="cuped"), method_instance_id=method_instance_id
    ) != method_config_hash(
        Method(name="alpha", variance_reduction="cuped", covariates=("x",)),
        method_instance_id=method_instance_id,
    )
    assert method_config_hash(base, method_instance_id=method_instance_id) != method_config_hash(
        Method(name="alpha", folds=3), method_instance_id=method_instance_id
    )


def test_method_config_hash_tracks_implementation_reference_identity():
    method = Method(name="alpha", propensity_learner=lambda: None)
    first = ImplementationRef(import_path="models:first", version_pin="1", config_digest="same")
    swapped = ImplementationRef(import_path="models:second", version_pin="2", config_digest="same")
    assert method_config_hash(
        method, {"propensity_learner": first}, method_instance_id="m"
    ) != method_config_hash(method, {"propensity_learner": swapped}, method_instance_id="m")


def test_callable_without_implementation_ref_refuses_with_code():
    method = Method(name="dml", propensity_learner=lambda: None)
    with pytest.raises(CodedError) as raised:
        method_config_hash(method, method_instance_id="m")
    assert raised.value.code == "kernel.method.implementation_ref_missing"
    ref = ImplementationRef(import_path="x:y", version_pin="1", config_digest="a")
    assert method_config_hash(
        method, {"propensity_learner": ref}, method_instance_id="m"
    ) != method_config_hash(
        method,
        {
            "propensity_learner": ImplementationRef(
                import_path="x:y", version_pin="1", config_digest="b"
            )
        },
        method_instance_id="m",
    )


def test_registration_hash_survives_the_plan_wire_round_trip():
    from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    plan = compile_decision_plan(None, [MeanMetric(name="orders", entity="user", fact="orders")])
    restored = compiled_plan_from_json(compiled_plan_to_json(plan))
    assert Registration.from_plan(plan) == Registration.from_plan(restored)
    from increment.plan import AnalysisPlan

    changed = compile_decision_plan(
        AnalysisPlan(alpha=0.1), [MeanMetric(name="orders", entity="user", fact="orders")]
    )
    assert (
        Registration.from_plan(plan).registration_hash
        != Registration.from_plan(changed).registration_hash
    )
    assert Registration.from_plan(plan).hash() == Registration.from_plan(plan).registration_hash
    assert Registration.from_plan(plan).matches(plan)
    assert not Registration.from_plan(plan).matches(changed)


def test_method_identity_and_registration_round_trip_with_stable_hash():
    from increment.kernel.contracts import contract_hash

    values = (
        MethodIdentity(
            estimator_id="e",
            method_config_hash="cfg",
            method_role="decision",
            method_instance_id="m",
        ),
        Registration(registration_hash="registration"),
    )
    for value in values:
        restored = type(value).from_payload(
            json.loads(json.dumps(value.canonical_payload(), sort_keys=True).encode())
        )
        assert restored == value
        assert restored.hash() == value.hash()
        if isinstance(value, MethodIdentity):
            assert isinstance(restored, MethodIdentity)
            assert restored.hash() == contract_hash("method_identity", restored.canonical_payload())
        else:
            assert isinstance(restored, Registration)
            assert restored.hash() == restored.registration_hash


def test_analysis_spec_hash_excludes_implementation_pins():
    class Design(BaseModel):
        name: str = "d"

    scope = Scope(view="run", grain="total")
    method = MethodIdentity(
        estimator_id="e", method_config_hash="c", method_role="decision", method_instance_id="i"
    )

    def calculate(
        *,
        current_scope: Scope = scope,
        registration_hash: str = "r",
        source_refs: dict[str, str] | None = None,
        methods: list[MethodIdentity] | None = None,
    ) -> str:
        return analysis_spec_hash(
            design=Design(),
            metric_hashes={"m": "h"},
            registration_hash=registration_hash,
            source_refs=source_refs or {"s": "d"},
            methods=methods or [method],
            scope=current_scope,
        )

    first = calculate()
    assert calculate() == first
    assert calculate(current_scope=Scope(view="daily", grain="daily")) != first
    assert calculate(registration_hash="x") != first
    assert calculate(source_refs={"s": "x"}) != first
    assert calculate(methods=[method.model_copy(update={"method_config_hash": "x"})]) != first


def test_execution_identity_hash_tracks_pins_and_portability():
    ref = ImplementationRef(import_path="x:y", version_pin="1", config_digest="a")
    other = ImplementationRef(import_path="z:y", version_pin="2", config_digest="b")

    def calculate(
        implementations: dict[tuple[str, str], ImplementationRef],
        *,
        package_version: str = "1",
        portable: bool = True,
    ) -> str:
        return execution_identity_hash(
            analysis_spec_hash="a",
            implementations=implementations,
            package_version=package_version,
            portable=portable,
        )

    implementations = {("a", "propensity_learner"): ref, ("a", "outcome_learner"): other}
    first = calculate(implementations)
    assert calculate(dict(reversed(tuple(implementations.items())))) == first
    assert calculate(implementations, portable=False) != first
    assert calculate(implementations, package_version="2") != first
    changed = ImplementationRef(import_path="x:y", version_pin="3", config_digest="a")
    assert calculate({**implementations, ("a", "propensity_learner"): changed}) != first


def test_execution_identity_hash_tracks_method_and_learner_bindings():
    propensity = ImplementationRef(
        import_path="models:propensity", version_pin="1", config_digest="same"
    )
    outcome = ImplementationRef(import_path="models:outcome", version_pin="1", config_digest="same")
    args = {"analysis_spec_hash": "analysis", "package_version": "3", "portable": True}
    first = execution_identity_hash(
        implementations={
            ("method-a", "propensity_learner"): propensity,
            ("method-b", "outcome_learner"): outcome,
        },
        **args,
    )
    swapped = execution_identity_hash(
        implementations={
            ("method-a", "outcome_learner"): propensity,
            ("method-b", "propensity_learner"): outcome,
        },
        **args,
    )
    assert first != swapped


def test_learner_version_pins_are_execution_only_and_roles_are_method_bound():
    method = Method(
        name="dml",
        propensity_learner=lambda: None,
        outcome_learner=lambda: None,
    )
    refs = {
        "propensity_learner": ImplementationRef(
            import_path="models:propensity", version_pin="v1", config_digest="p"
        ),
        "outcome_learner": ImplementationRef(
            import_path="models:outcome", version_pin="v1", config_digest="o"
        ),
    }
    pinned_refs = {role: ref.model_copy(update={"version_pin": "v2"}) for role, ref in refs.items()}
    first_config = method_config_hash(method, refs, method_instance_id="m")
    pinned_config = method_config_hash(method, pinned_refs, method_instance_id="m")

    class Design(BaseModel):
        name: str = "d"

    def analysis_spec(config_hash: str) -> str:
        return analysis_spec_hash(
            design=Design(),
            metric_hashes={},
            registration_hash="r",
            source_refs={},
            methods=[
                MethodIdentity(
                    estimator_id="e",
                    method_config_hash=config_hash,
                    method_role="decision",
                    method_instance_id="m",
                )
            ],
            scope=Scope(view="run", grain="total"),
        )

    def execution_identity(
        config_hash: str, implementations: Mapping[tuple[str, str], ImplementationRef]
    ) -> str:
        return execution_identity_hash(
            analysis_spec_hash=analysis_spec(config_hash),
            implementations=implementations,
            package_version="1",
            portable=True,
        )

    first_spec = analysis_spec(first_config)
    assert pinned_config == first_config
    assert analysis_spec(pinned_config) == first_spec
    assert execution_identity(first_config, {("m", role): ref for role, ref in refs.items()}) != (
        execution_identity(pinned_config, {("m", role): ref for role, ref in pinned_refs.items()})
    )

    swapped_refs = {
        "propensity_learner": refs["outcome_learner"],
        "outcome_learner": refs["propensity_learner"],
    }
    swapped_config = method_config_hash(method, swapped_refs, method_instance_id="m")
    assert swapped_config != first_config
    assert analysis_spec(swapped_config) != first_spec


def test_analysis_request_hash_is_stable_and_order_sensitive():
    from increment.evidence import CountsRequest, MomentsRequest
    from increment.semantics.models import MeanMetric

    scope = Scope(view="run", grain="total")
    metric = MeanMetric(name="orders", entity="user", fact="orders")
    one, two = CountsRequest(kind="unit"), MomentsRequest(metric=metric)
    value = AnalysisRequest.from_requests(
        (one, two), scope=scope, registration_hash="r", source_ref="s"
    )
    assert (
        value.hash()
        == AnalysisRequest.from_requests(
            (one, two), scope=scope, registration_hash="r", source_ref="s"
        ).hash()
    )
    assert (
        AnalysisRequest.from_requests(
            (two, one), scope=scope, registration_hash="r", source_ref="s"
        ).hash()
        != value.hash()
    )
    assert (
        AnalysisRequest.from_requests(
            (CountsRequest(kind="cluster"), two),
            scope=scope,
            registration_hash="r",
            source_ref="s",
        ).hash()
        != value.hash()
    )


def test_readout_metadata_satisfies_run_metadata():
    from increment.estimation.readout_types import ReadoutMetadata, ReadoutScope
    from increment.kernel.contracts import RunMetadata

    value = ReadoutMetadata(
        scope=ReadoutScope(
            snapshot_id="snapshot", cells=(), decision_cells=(), populations=(), by_source={}
        ),
        cells=(),
    )
    assert isinstance(value, RunMetadata)
    assert hasattr(value, "scope") and hasattr(value, "cells")
    assert isinstance(value.partial, bool) and hasattr(value, "partial_reason")


def test_analysis_request_rejects_empty_or_non_doorway_requests():
    for requests in ((), (object(),)):
        with pytest.raises(CodedError) as raised:
            AnalysisRequest.from_requests(
                requests,
                scope=Scope(view="run", grain="total"),
                registration_hash="r",
                source_ref="s",
            )
        assert raised.value.code == "kernel.request.invalid"


def test_population_mismatch_refuses():
    from increment.evidence import CountsRequest

    with pytest.raises(CodedError) as raised:
        AnalysisRequest.from_requests(
            (CountsRequest(kind="unit", population="triggered"),),
            scope=Scope(view="run", grain="total"),
            registration_hash="r",
            source_ref="s",
        )
    assert raised.value.code == "kernel.request.scope_mismatch"


def test_analysis_request_round_trips_only_portable_request_descriptors():
    from increment.evidence import CountsRequest, MomentsRequest
    from increment.kernel.contracts import AnalysisRequest
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="orders", entity="user", fact="orders")
    value = AnalysisRequest.from_requests(
        (CountsRequest(kind="unit"), MomentsRequest(metric=metric)),
        scope=Scope(view="run", grain="total"),
        registration_hash="r",
        source_ref="s",
    )
    payload = json.loads(json.dumps(value.canonical_payload(), sort_keys=True))
    restored = AnalysisRequest.from_payload(payload)
    assert restored == value
    assert restored.hash() == value.hash()


def test_analysis_request_rejects_unserializable_method_with_picklable_error():
    from increment.evidence import MethodFitRequest
    from increment.semantics.models import MeanMetric

    request = MethodFitRequest(
        metric=MeanMetric(name="orders", entity="user", fact="orders"),
        method=Method(name="iptw", propensity_learner=lambda: None),
        control="control",
        arm_order=("control", "treatment"),
        population="assigned",
    )
    with pytest.raises(CodedError) as raised:
        AnalysisRequest.from_requests(
            (request,),
            scope=Scope(view="run", grain="total"),
            registration_hash="r",
            source_ref="s",
        )
    error = raised.value
    assert error.code == "kernel.request.unportable"
    assert error.context == {"kind": "MethodFitRequest", "field": "method"}
    assert pickle.loads(pickle.dumps(error)).code == error.code
    assert copy.deepcopy(error).context == error.context


def test_analysis_request_json_round_trips_date_request_fields():
    from increment.evidence import SequentialCaptureRequest

    value = AnalysisRequest.from_requests(
        (SequentialCaptureRequest(finalized=True, as_of=date(2026, 1, 3)),),
        scope=Scope(
            view="asof",
            grain="asof",
            through_day=date(2026, 1, 3),
            completion_policy=True,
        ),
        registration_hash="r",
        source_ref="s",
    )
    payload = json.loads(json.dumps(value.canonical_payload(), sort_keys=True))
    restored = AnalysisRequest.from_payload(payload)
    assert restored.canonical_payload() == payload
    assert restored.hash() == value.hash()


def test_analysis_request_hash_normalizes_equivalent_datetime_offsets():
    from datetime import datetime, timedelta, timezone

    from increment.evidence import SequentialCaptureRequest

    utc = datetime(2026, 1, 3, tzinfo=UTC)
    offset = datetime(2026, 1, 3, 1, tzinfo=timezone(timedelta(hours=1)))
    hashes = tuple(
        AnalysisRequest.from_requests(
            (SequentialCaptureRequest(finalized=True, as_of=value),),
            scope=Scope(view="asof", grain="asof"),
            registration_hash="r",
            source_ref="s",
        ).hash()
        for value in (utc, offset)
    )
    assert hashes[0] == hashes[1]


def test_analysis_request_rejects_naive_datetime_as_unportable():
    from datetime import datetime

    from increment.evidence import SequentialCaptureRequest

    with pytest.raises(CodedError) as raised:
        AnalysisRequest.from_requests(
            (SequentialCaptureRequest(finalized=True, as_of=datetime(2026, 1, 3)),),
            scope=Scope(view="asof", grain="asof"),
            registration_hash="r",
            source_ref="s",
        )
    assert raised.value.code == "kernel.request.unportable"


def test_analysis_request_json_encodes_uuid_breakout_values():
    import uuid

    from increment.evidence import BreakoutCell, BreakoutIdentity, MethodFitRequest
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="orders", entity="user", fact="orders")
    value = uuid.UUID("caa2f392-68f8-4bdc-a8ae-55f53b8a1e51")
    request = MethodFitRequest(
        metric=metric,
        method=Method.unadjusted(),
        control="control",
        arm_order=("control", "treatment"),
        population="assigned",
        cell=BreakoutCell(BreakoutIdentity("id", None), value, False),
    )
    descriptor = AnalysisRequest.from_requests(
        (request,),
        scope=Scope(
            view="breakout",
            grain="total",
            segment=BreakoutIdentity("id", None),
        ),
        registration_hash="r",
        source_ref="s",
    )
    payload = json.loads(json.dumps(descriptor.canonical_payload()))
    assert payload["requests"][0]["fields"]["cell"]["value"] == str(value)


def test_analysis_request_admits_observational_breakout_and_sequential_requests():
    from increment.declarations import Method
    from increment.evidence import (
        BreakoutIdentity,
        BreakoutUnitRowsRequest,
        MethodFitRequest,
        SequentialCaptureRequest,
        UnitRowsRequest,
    )
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="orders", entity="user", fact="orders")
    requests = (
        MethodFitRequest(
            metric=metric,
            method=Method(name="iptw"),
            control="control",
            arm_order=("control", "treatment"),
            population="assigned",
        ),
        BreakoutUnitRowsRequest(metric=metric, breakout=BreakoutIdentity("country", None)),
        SequentialCaptureRequest(finalized=True),
        UnitRowsRequest(metric=metric),
    )
    request = AnalysisRequest.from_requests(
        requests,
        scope=Scope(
            view="breakout",
            grain="total",
            segment=BreakoutIdentity("country", None),
            completion_policy=True,
        ),
        registration_hash="r",
        source_ref="s",
    )
    assert [descriptor.kind for descriptor in request.requests] == [
        "MethodFitRequest",
        "BreakoutUnitRowsRequest",
        "SequentialCaptureRequest",
        "UnitRowsRequest",
    ]


@pytest.mark.parametrize("case_index", range(9))
@given(agrees=st.booleans())
def test_request_scope_a3_axes(case_index: int, agrees: bool) -> None:
    from increment.evidence import (
        BreakoutCell,
        BreakoutIdentity,
        BreakoutMethodMomentsRequest,
        BreakoutSlicesRequest,
        BreakoutUnitRowsRequest,
        CountsRequest,
        DayAxisRequest,
        MethodFitRequest,
        MomentsRequest,
    )
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="orders", entity="user", fact="orders")
    country = BreakoutIdentity("country", "warehouse")
    device = BreakoutIdentity("device", "warehouse")
    method = Method.unadjusted()
    breakout_scope = Scope(view="breakout", grain="total", segment=device)
    run_scope = Scope(view="run", grain="total")
    daily_scope = Scope(view="daily", grain="daily")
    asof_scope = Scope(view="asof", grain="asof")
    cases = (
        (
            CountsRequest(kind="unit", population="assigned"),
            Scope(view="run", grain="total", population="assigned"),
            Scope(view="run", grain="total", population="triggered"),
            "population",
        ),
        (
            CountsRequest(kind="triggered"),
            Scope(view="run", grain="total", population="triggered"),
            Scope(view="run", grain="total", population="assigned"),
            "population",
        ),
        (
            MomentsRequest(metric=metric, grain="daily"),
            daily_scope,
            asof_scope,
            "grain",
        ),
        (
            MomentsRequest(metric=metric, completed_windows_only=True),
            Scope(view="run", grain="total", completion_policy=True),
            run_scope,
            "completed_windows_only",
        ),
        (
            DayAxisRequest(caller="run_asof", grain="asof", metrics=(metric,), dimension=None),
            asof_scope,
            asof_scope,
            "caller",
        ),
        (
            BreakoutSlicesRequest(breakouts=(country,)),
            breakout_scope,
            run_scope,
            "view",
        ),
        (
            BreakoutMethodMomentsRequest(metric=metric, method=method, breakout=country),
            breakout_scope,
            run_scope,
            "view",
        ),
        (
            BreakoutUnitRowsRequest(metric=metric, breakout=country),
            breakout_scope,
            run_scope,
            "view",
        ),
        (
            MethodFitRequest(
                metric=metric,
                method=method,
                control="control",
                arm_order=("control", "treatment"),
                population="assigned",
                cell=BreakoutCell(country, "US", False),
            ),
            breakout_scope,
            run_scope,
            "view",
        ),
    )
    request, matching, drifted, field = cases[case_index]
    scope = matching if agrees else drifted
    if case_index == 4:
        request = DayAxisRequest(
            caller="run_asof" if agrees else "run_daily",
            grain="asof",
            metrics=(metric,),
            dimension=None,
        )
        scope = asof_scope
    if agrees:
        AnalysisRequest.from_requests(
            (request,), scope=scope, registration_hash="r", source_ref="s"
        )
        return
    with pytest.raises(CodedError) as raised:
        AnalysisRequest.from_requests(
            (request,), scope=scope, registration_hash="r", source_ref="s"
        )
    assert raised.value.code == "kernel.request.scope_mismatch"
    assert raised.value.context == {"kind": type(request).__name__, "field": field}


def test_analysis_request_rejects_unportable_fields_with_code():
    from increment.evidence import SequentialCaptureRequest

    with pytest.raises(CodedError) as raised:
        AnalysisRequest.from_requests(
            (SequentialCaptureRequest(finalized=True, previous=object()),),
            scope=Scope(view="run", grain="total"),
            registration_hash="r",
            source_ref="s",
        )
    assert raised.value.code == "kernel.request.unportable"
    assert raised.value.context["field"] == "previous"


def test_analysis_spec_refuses_duplicate_method_instance_ids():
    class Design(BaseModel):
        name: str = "d"

    method = MethodIdentity(
        estimator_id="e",
        method_config_hash="c",
        method_role="decision",
        method_instance_id="same",
    )
    with pytest.raises(CodedError) as raised:
        analysis_spec_hash(
            design=Design(),
            metric_hashes={},
            registration_hash="r",
            source_refs={},
            methods=[method, method.model_copy()],
            scope=Scope(view="run", grain="total"),
        )
    assert raised.value.code == "kernel.method.duplicate_instance"


def _golden_contract_hash(kind: str, payload: object) -> str:
    from increment.kernel.contracts import canonical_json_bytes

    return hashlib.sha256(
        canonical_json_bytes({"schema": 1, "kind": kind, "payload": payload})
    ).hexdigest()


def test_method_config_hash_matches_hand_built_golden_payload():
    method = Method(
        name="ignored",
        propensity_learner=lambda: None,
        outcome_learner=lambda: None,
    )
    implementations = {
        "propensity_learner": ImplementationRef(
            import_path="models.propensity:build", version_pin="1", config_digest="p"
        ),
        "outcome_learner": ImplementationRef(
            import_path="models.outcome:build", version_pin="2", config_digest="o"
        ),
    }
    expected = {
        "variance_reduction": "none",
        "conversion_inference": "auto",
        "covariates": None,
        "procedure": None,
        "propensity_learner": {
            "method_instance_id": "method-1",
            "learner_role": "propensity_learner",
            "import_path": "models.propensity:build",
            "config_digest": "p",
        },
        "outcome_learner": {
            "method_instance_id": "method-1",
            "learner_role": "outcome_learner",
            "import_path": "models.outcome:build",
            "config_digest": "o",
        },
        "folds": None,
    }
    assert method_config_hash(
        method, implementations, method_instance_id="method-1"
    ) == _golden_contract_hash("method_config", expected)


def test_analysis_spec_hash_matches_hand_built_golden_payload():
    class Design(BaseModel):
        name: str

    scope = Scope(view="run", grain="total")
    methods = [
        MethodIdentity(
            estimator_id="e2",
            method_config_hash="c2",
            method_role="sensitivity",
            method_instance_id="b",
        ),
        MethodIdentity(
            estimator_id="e1",
            method_config_hash="c1",
            method_role="decision",
            method_instance_id="a",
        ),
    ]
    expected = {
        "design": {"name": "design"},
        "metric_hashes": {"m": "metric"},
        "registration_hash": "registration",
        "source_refs": {"source": "digest"},
        "methods": [
            {
                "estimator_id": "e1",
                "method_config_hash": "c1",
                "method_role": "decision",
                "method_instance_id": "a",
            },
            {
                "estimator_id": "e2",
                "method_config_hash": "c2",
                "method_role": "sensitivity",
                "method_instance_id": "b",
            },
        ],
        "scope": scope.canonical_payload(),
    }
    assert analysis_spec_hash(
        design=Design(name="design"),
        metric_hashes={"m": "metric"},
        registration_hash="registration",
        source_refs={"source": "digest"},
        methods=methods,
        scope=scope,
    ) == _golden_contract_hash("analysis_spec", expected)


def test_execution_identity_hash_matches_hand_built_golden_payload_and_sorts_pins():
    refs = {
        ("method-a", "outcome_learner"): ImplementationRef(
            import_path="models.outcome:build", version_pin="2", config_digest="o"
        ),
        ("method-a", "propensity_learner"): ImplementationRef(
            import_path="models.propensity:build", version_pin="1", config_digest="p"
        ),
    }
    expected = {
        "analysis_spec_hash": "analysis",
        "implementations": [
            {
                "method_instance_id": "method-a",
                "learner_role": "outcome_learner",
                "implementation": {
                    "import_path": "models.outcome:build",
                    "version_pin": "2",
                    "config_digest": "o",
                },
            },
            {
                "method_instance_id": "method-a",
                "learner_role": "propensity_learner",
                "implementation": {
                    "import_path": "models.propensity:build",
                    "version_pin": "1",
                    "config_digest": "p",
                },
            },
        ],
        "package_version": "3",
        "portable": False,
    }
    assert execution_identity_hash(
        analysis_spec_hash="analysis",
        implementations=refs,
        package_version="3",
        portable=False,
    ) == _golden_contract_hash("execution_identity", expected)


def test_execution_identity_hash_sorts_all_implementation_binding_fields():
    refs = {
        ("method-b", "outcome_learner"): ImplementationRef(
            import_path="models:build", version_pin="1", config_digest="a"
        ),
        ("method-a", "propensity_learner"): ImplementationRef(
            import_path="models:build", version_pin="2", config_digest="b"
        ),
    }
    args = {
        "analysis_spec_hash": "analysis",
        "package_version": "3",
        "portable": True,
    }
    assert execution_identity_hash(implementations=refs, **args) == execution_identity_hash(
        implementations=dict(reversed(tuple(refs.items()))), **args
    )


def test_registration_hash_matches_hand_built_golden_payload():
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    plan = compile_decision_plan(None, [MeanMetric(name="orders", entity="user", fact="orders")])
    expected = {
        "wire_version": 4,
        "declared": False,
        "alpha": 0.05,
        "q": 0.1,
        "path": "warehouse",
        "inference": {"kind": "fixed"},
        "compliance": None,
        "procedures": {
            "orders": {
                "kind": "relative",
                "metric": "orders",
                "role": "unassigned",
                "decision_method": {
                    "name": "unadjusted",
                    "variance_reduction": "none",
                    "conversion_inference": "auto",
                    "covariates": None,
                },
                "sensitivity_methods": [],
                "methods_explicitly_empty": False,
                "alternative": "two-sided",
                "prior": None,
                "prior_is_global": True,
                "alpha": 0.05,
                "family": {"family": {"kind": "none"}, "member": False},
                "inference": {"kind": "fixed"},
                "axis": "relative",
                "scale": "relative",
                "null_lift": 0.0,
            }
        },
        "view_policies": {
            "asof": {
                "kind": "multiplicity",
                "name": "asof",
                "correction": "none",
                "q": None,
                "axes": ["metric", "arm"],
                "guarantee": "none",
            },
            "encouragement_breakout": {
                "kind": "multiplicity",
                "name": "breakout",
                "correction": "none",
                "q": None,
                "axes": ["metric", "arm", "segment"],
                "guarantee": "none",
            },
            "randomized_breakout": {
                "kind": "multiplicity",
                "name": "breakout",
                "correction": "bh",
                "q": 0.1,
                "axes": ["metric", "arm", "segment"],
                "guarantee": "fdr",
            },
        },
    }
    assert Registration.from_plan(plan).registration_hash == _golden_contract_hash(
        "registration", expected
    )
