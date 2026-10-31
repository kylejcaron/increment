from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pytest

from increment.errors import CodedError
from increment.evidence import EvidenceScope, EvidenceScopeResult, MomentsRequest
from increment.kernel.context import EvidenceExecutionContext


class _Provider:
    def __init__(self) -> None:
        self.requests: list[EvidenceScope] = []
        self.entered = 0
        self.exited = 0
        self.captures = 0

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        self.requests.append(request)
        if request.snapshot:
            self.captures += 1
        status = "captured" if request.snapshot else "live"

        @contextmanager
        def entered() -> Iterator[_Provider]:
            self.entered += 1
            try:
                yield self
            finally:
                self.exited += 1

        return EvidenceScopeResult(entered(), request, capture_status=status)


def test_captured_context_opens_once_and_borrow_does_not_close_owner():
    provider = _Provider()
    context = EvidenceExecutionContext(
        cast(Any, provider),
        cast(
            Any,
            EvidenceScope(
                metrics=cast(Any, ("revenue",)),
                horizon_metrics=cast(Any, ("later",)),
            ),
        ),
        run_id="run-1",
        coherence="captured",
    )
    context.open()
    with context.borrow() as first:
        assert first is provider
    with pytest.raises(RuntimeError):
        with context.borrow():
            raise RuntimeError("borrow failure")
    with context.borrow() as second:
        assert second is provider
    assert len(provider.requests) == 1
    assert provider.requests[0].snapshot is True
    assert provider.requests[0].execution is True
    assert provider.requests[0].horizon_metrics == ("later",)
    with pytest.raises(CodedError) as refused:
        context.validate_request(provider, ("later",), ())
    assert refused.value.code == "kernel.context.undeclared_stream"
    assert refused.value.context["stream"] == "metric"
    manifest = context.snapshot_manifest
    assert manifest and manifest == context.snapshot_manifest
    assert provider.exited == 0
    context.close()
    assert provider.captures == 1
    context.close()
    assert provider.exited == 1


def test_frame_totals_capture_freezes_caller_owned_dataframe():
    import narwhals as nw
    import pandas as pd

    from increment.evidence import UnitRowsRequest, evidence_provider
    from increment.frame import FrameTotalsSource, from_unit_summary

    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": ["control", "control", "treatment", "treatment"],
            "revenue": [1.0, 2.0, 3.0, 4.0],
        }
    )
    prepared = from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = FrameTotalsSource(
        frame=nw.from_native(frame, eager_only=True),
        unit="unit",
        group="arm",
        moments=prepared.raw_moments,
        metrics=list(prepared._specs),
        control="control",
        experiment_id="frame",
        design=prepared.design,
        plan=prepared.plan,
        synthesised_metrics=prepared.context.metrics,
    )
    metric = source.context.metrics[0]
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(metrics=(metric,)),
        run_id="frame-snapshot",
        coherence="captured",
    )
    context.open()
    frame["revenue"] = [100.0, 200.0, 300.0, 400.0]
    with context.borrow() as scoped:
        rows = nw.from_native(scoped.unit_rows(UnitRowsRequest(metric)), eager_only=True)
    assert rows["y"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert context.effective_modes == {"revenue": "captured"}
    context.close()


def test_live_context_does_not_capture_and_closed_context_refuses():
    provider = _Provider()
    context = EvidenceExecutionContext(
        cast(Any, provider),
        cast(Any, EvidenceScope(metrics=cast(Any, ("revenue",)))),
        run_id="run-2",
        coherence="live",
    )
    context.open()
    assert provider.requests[0].snapshot is False
    assert context.effective_modes == {"revenue": "live"}
    assert context.snapshot_manifest == {}
    context.close()
    with pytest.raises(CodedError) as refused:
        context.open()
    assert refused.value.code == "kernel.context.closed"


def test_custom_provider_without_stream_inventory_has_provider_mode():
    class Provider(_Provider):
        evidence = object()
        context = object()

        def moments(self, *_args: Any, **_kwargs: Any) -> None: ...

        def breakout_method_moments(self, *_args: Any, **_kwargs: Any) -> None: ...

        def unit_rows(self, *_args: Any, **_kwargs: Any) -> None: ...

        def method_fit(self, *_args: Any, **_kwargs: Any) -> None: ...

        def compliance(self, *_args: Any, **_kwargs: Any) -> None: ...

        def compliance_series(self, *_args: Any, **_kwargs: Any) -> None: ...

        def counts(self, *_args: Any, **_kwargs: Any) -> None: ...

        def capture_sequential(self, *_args: Any, **_kwargs: Any) -> None: ...

    provider = Provider()
    context = EvidenceExecutionContext(
        cast(Any, provider), EvidenceScope(), run_id="provider-inventory"
    )
    context.open()
    assert context.effective_modes == {"provider": "provider"}
    context.close()


def test_fallback_inventory_records_uptake_stream_without_metrics():
    provider = _Provider()
    context = EvidenceExecutionContext(
        cast(Any, provider),
        EvidenceScope(uptake_facts=("purchase",)),
        run_id="run-uptake",
        coherence="captured",
    )
    context.open()
    assert context.effective_modes == {"fact:purchase": "captured"}
    context.close()


def test_owned_scope_passes_read_exception_to_provider_exit():
    seen: list[type[BaseException] | None] = []

    class Provider(_Provider):
        def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
            @contextmanager
            def entered() -> Iterator[_Provider]:
                try:
                    yield self
                except BaseException as error:
                    seen.append(type(error))
                    raise
                finally:
                    self.exited += 1

            return EvidenceScopeResult(entered(), request)

    provider = Provider()
    context = EvidenceExecutionContext(cast(Any, provider), EvidenceScope(), run_id="run-error")
    with pytest.raises(RuntimeError):
        with context.evidence_scope():
            raise RuntimeError("read failed")
    assert seen == [RuntimeError]
    assert provider.exited == 1


def test_borrower_exception_does_not_close_owner():
    provider = _Provider()
    context = EvidenceExecutionContext(cast(Any, provider), EvidenceScope(), run_id="run-3")
    context.open()
    with pytest.raises(RuntimeError):
        with context.borrow():
            raise RuntimeError("read failure")
    assert provider.exited == 0
    context.close()
    assert provider.exited == 1


@pytest.mark.integration
@pytest.mark.parametrize("fault", ["scope", "enter", "read", "exit"])
def test_context_faults_leave_no_duckdb_temp_tables(tmp_path, fault):
    from increment.evidence import EvidenceScope, EvidenceScopeResult, evidence_provider
    from increment.kernel.context import EvidenceExecutionContext
    from tests.analysis_factory import held_source, native_connection
    from tests.source_conformance import definitions_source

    analysis, _ = definitions_source(tmp_path, store="always", return_analysis=True)
    source = cast(Any, held_source(analysis))
    provider = evidence_provider(source)
    metric = source.context.metrics[0]
    connection = native_connection(source)
    before = connection.raw_sql(
        "SELECT table_name FROM duckdb_tables() WHERE temporary ORDER BY table_name"
    ).fetchall()

    class FaultProvider:
        evidence = provider.evidence
        context = provider.context

        def scope(self, request):
            if fault == "scope":
                raise RuntimeError("scope failure")
            scoped = provider.scope(request)

            @contextmanager
            def source_scope():
                with scoped.source as pinned:
                    if fault == "enter":
                        raise RuntimeError("enter failure")

                    class Reader:
                        def moments(self, request):
                            value = pinned.moments(request)
                            if fault == "read":
                                raise RuntimeError("read failure")
                            return value

                    yield Reader()
                if fault == "exit":
                    raise RuntimeError("exit failure")

            return EvidenceScopeResult(
                source_scope(),
                request,
                scoped.streams,
                scoped.stream_resolver,
                scoped.capture_status,
            )

    context = EvidenceExecutionContext(
        cast(Any, FaultProvider()),
        EvidenceScope(metrics=cast(Any, (metric,))),
        run_id=f"fault-{fault}",
    )
    try:
        if fault in {"scope", "enter"}:
            with pytest.raises(RuntimeError):
                context.open()
        else:
            context.open()
            if fault in {"read", "exit"}:
                captured = connection.raw_sql(
                    "SELECT table_name FROM duckdb_tables() WHERE temporary ORDER BY table_name"
                ).fetchall()
                assert len(captured) > len(before)
            if fault == "read":
                with context.borrow() as scoped:
                    with pytest.raises(RuntimeError):
                        scoped.moments(MomentsRequest(metric))
            else:
                with context.borrow() as scoped:
                    scoped.moments(MomentsRequest(metric))
            if fault == "exit":
                with pytest.raises(RuntimeError):
                    context.close()
    finally:
        try:
            context.close()
        except RuntimeError:
            assert fault == "exit"
        after = connection.raw_sql(
            "SELECT table_name FROM duckdb_tables() WHERE temporary ORDER BY table_name"
        ).fetchall()
        assert after == before
        analysis.close()


@pytest.mark.integration
def test_captured_context_refuses_when_its_pinned_stream_disappears(tmp_path):
    from increment.evidence import EvidenceScope, evidence_provider
    from tests.analysis_factory import held_source
    from tests.source_conformance import _clear_captured_source_tables, definitions_source

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
        ),
        run_id="lost-capture",
    )
    try:
        context.open()
        assert context.snapshot_manifest
        with context.borrow() as pinned:
            _clear_captured_source_tables(pinned)
        with pytest.raises(CodedError) as refused:
            with context.borrow():
                pass
        assert refused.value.code == "kernel.snapshot.integrity"
        assert refused.value.context["stream"] == "events"
        assert refused.value.context["snapshot_id"] == context.snapshot_manifest["events"]
    finally:
        context.close()
        analysis.close()


def test_assignment_request_must_be_declared_and_uses_the_pinned_provider() -> None:
    from increment.evidence import AssignmentEvidence, AssignmentRequest

    request = AssignmentRequest(experiment="exp", version=1)

    class AssignmentProvider(_Provider):
        def __init__(self, count):
            super().__init__()
            self.count = count
            self.captured = None

        def scope(self, request):
            self.requests.append(request)

            @contextmanager
            def entered():
                yield self.captured

            return EvidenceScopeResult(entered(), request, capture_status="captured")

        def assignment(self, selected):
            assert selected == request
            return AssignmentEvidence(
                experiment="exp",
                population="assigned",
                versions=(1,),
                periods=(None,),
                arm_counts=((1, None, "control", self.count),),
                independence_grain="unit",
            )

    captured = AssignmentProvider(7)
    provider = AssignmentProvider(2)
    provider.captured = captured
    scope = EvidenceScope(assignment_requests=(request,))
    context = EvidenceExecutionContext(cast(Any, provider), scope, run_id="assignment")
    pinned = context.open()
    assert provider.requests[0].assignment_requests == (request,)
    provider.count = 99
    evidence = context.assignment(request)
    assert evidence.arm_counts == ((1, None, "control", 7),)
    assert pinned is captured
    assert any(stream.startswith("assignment:exp:") for stream in context.effective_modes)
    with pytest.raises(CodedError) as refused:
        context.assignment(AssignmentRequest(experiment="other"))
    assert refused.value.code == "kernel.context.undeclared_stream"
    context.close()
