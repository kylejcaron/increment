"""Parity checks for typed providers that forward to built-in evidence loaders."""

from __future__ import annotations

import warnings
from dataclasses import fields, is_dataclass
from typing import Any, cast

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from increment import workflows
from increment.errors import CodedError, IncrementWarning
from increment.evidence import (
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountsRequest,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitRequest,
    MomentsRequest,
    SequentialCapture,
    SequentialCaptureRequest,
    SequentialSnapshotRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from tests.analysis_factory import bind_pair, held_source, view_readout
from tests.parity_harness import matrix, matrix_cases
from tests.parity_harness.cases import Bound, _close_parity_binding, _triggered_day_axis_cases
from tests.parity_harness.comparison import nested_close
from tests.parity_harness.views import read
from tests.warning_codes import warning_codes

IDENTITY_FIELDS = {"source_snapshot_id", "family_id"}


class ForwardingProvider:
    """A structural typed provider with no legacy loader methods."""

    def __init__(self, source: Any, *, scope_mode: str, experiment: Any) -> None:
        self.adapter: Any = evidence_provider(source)
        self.evidence = self.adapter.evidence
        self.context = self.adapter.context
        if experiment is not None:
            self.artifact_experiment = experiment
        self.scope_mode = scope_mode

    def moments(self, request: MomentsRequest):
        return request_evidence(self.adapter, request)

    def unit_rows(self, request: UnitRowsRequest):
        return request_evidence(self.adapter, request)

    def method_fit(self, request: MethodFitRequest):
        return request_evidence(self.adapter, request)

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest):
        return request_evidence(self.adapter, request)

    def compliance(self, request: ComplianceRequest):
        return request_evidence(self.adapter, request)

    def compliance_series(self, request: ComplianceSeriesRequest):
        return request_evidence(self.adapter, request)

    def counts(self, request: CountsRequest):
        return request_evidence(self.adapter, request)

    def capture_sequential(self, request: SequentialCaptureRequest):
        return request_evidence(self.adapter, request)

    def sequential_snapshot(
        self, request: SequentialSnapshotRequest | None = None, *, previous=None
    ) -> SequentialCapture:
        if request is None:
            request = SequentialSnapshotRequest(previous=previous)
        return self.adapter.sequential_snapshot(request)

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        if self.scope_mode == "adapter":
            return self.adapter.scope(request)

        from contextlib import contextmanager

        @contextmanager
        def provider_scope():
            yield self

        return EvidenceScopeResult(provider_scope(), request)

    def close(self) -> None:
        self.adapter.source.close()


def _provider_pair(provider: ForwardingProvider) -> Bound:
    return bind_pair(provider)


def _forwarding(builtin: Bound, mode: str) -> ForwardingProvider:
    return ForwardingProvider(
        held_source(builtin[1]),
        scope_mode=mode,
        experiment=workflows.experiment_definition(*builtin),
    )


def _unnamed_breakout_sources(builtin: Bound, mode: str) -> bool:
    experiment = workflows.experiment_definition(*builtin)
    return mode == "self" and all(item.source is None for item in experiment.breakouts)


def _without_source(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_without_source(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _without_source(item) for key, item in payload.items() if key != "source"}
    return payload


def _payload(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_payload(item) for item in value]
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="python")
        return {key: _payload(item) for key, item in result.items() if key not in IDENTITY_FIELDS}
    if is_dataclass(value):
        return {
            field.name: _payload(getattr(value, field.name))
            for field in fields(value)
            if field.name not in IDENTITY_FIELDS
        }
    if isinstance(value, dict):
        return {key: _payload(item) for key, item in value.items() if key not in IDENTITY_FIELDS}
    return value


def _leg_read(bound: Bound, cell: matrix.Cell, method: str) -> Any:
    view = matrix_cases._case_fields(cell, method)["view"]
    if cell.view == "breakout":
        return list(read(bound, "breakout"))
    return list(read(bound, "run" if view is None else view))


def _outcome(bound: Bound, cell: matrix.Cell, method: str) -> tuple[str, Any]:
    try:
        return "result", _payload(_leg_read(bound, cell, method))
    except TypeError as exc:
        pytest.fail(f"{cell.id}/{method}: harness TypeError: {exc}", pytrace=True)
    except CodedError as exc:
        return "refusal", exc.code
    except Exception as exc:  # compared on both sides; unexpected paths are divergences
        return "exception", (type(exc).__qualname__, str(exc))


def _check(cell: matrix.Cell, ingress: str, mode: str) -> list[str]:
    divergences: list[str] = []
    factory = matrix_cases._Ingress(cell)
    build = factory.definitions if ingress == "from_definitions" else factory.artifact
    with warnings.catch_warnings(record=True) as construction_warnings:
        warnings.simplefilter("always", IncrementWarning)
        builtin = build()
    metric = matrix_cases.metric_definition(cell)
    expected_warnings = (
        ["definition.metric.variable_window_days_none"]
        if metric["type"] in {"mean", "conversion"} and metric.get("window_days") is None
        else []
    )
    assert warning_codes(construction_warnings) == expected_warnings
    custom = _provider_pair(_forwarding(builtin, mode))
    try:
        for method in matrix.methods(cell):
            expected = _outcome(builtin, cell, method)
            actual = _outcome(custom, cell, method)
            if expected[0] != actual[0]:
                divergences.append(f"{method}: outcome built-in={expected!r}, provider={actual!r}")
            elif expected[0] == "result":
                left, right = expected[1], actual[1]
                if cell.view == "breakout" and _unnamed_breakout_sources(builtin, mode):
                    # A provider declares breakout properties, not the built-in catalog's
                    # default source, so it labels an unnamed source as unnamed.
                    left, right = _without_source(left), _without_source(right)
                if cell.view == "breakout":
                    assert left, f"{cell.id}/{mode}/{ingress}: built-in breakout produced no rows"
                if cell.option == "sequential":
                    assert left, (
                        f"{cell.id}/{mode}/{ingress}: built-in sequential read produced no rows"
                    )
                all_fields = (
                    sorted(set(cast(dict[str, Any], left)) | set(cast(dict[str, Any], right)))
                    if isinstance(left, dict)
                    else ["rows"]
                )
                for field in all_fields:
                    lval = left.get(field) if isinstance(left, dict) else left
                    rval = right.get(field) if isinstance(right, dict) else right
                    if not nested_close(lval, rval):
                        divergences.append(
                            f"{method}.{field}: built-in={lval!r}, provider={rval!r}"
                        )
            elif expected != actual:
                divergences.append(f"{method}: refusal built-in={expected!r}, provider={actual!r}")
    finally:
        custom[1].close()
        _close_parity_binding(builtin)
    assert not divergences, (
        f"{cell.id}/{mode}/{ingress}: forwarding provider divergences: {divergences}"
    )
    return divergences


@pytest.mark.slow
@pytest.mark.integration
@settings(max_examples=60, deadline=None)
@given(cell=st.sampled_from(matrix.parity_matrix_cells()))
@example(cell=matrix.Cell("mean", "run", "none", "utc", "error"))
def test_forwarding_provider_matches_matrix_cell(cell: matrix.Cell) -> None:
    disposition = matrix.classify(cell)
    if not any(
        disposition.legs[method]["from_definitions"].status == "supported"
        for method in matrix.methods(cell)
    ):
        return
    for mode in ("self", "adapter"):
        _check(cell, "from_definitions", mode)
        if any(
            disposition.legs[method]["from_unit_day_artifact"].status == "supported"
            for method in matrix.methods(cell)
        ):
            _check(cell, "from_unit_day_artifact", mode)


@pytest.mark.integration
@pytest.mark.parametrize(
    "cell",
    [
        matrix.Cell("mean", "run", "none", "utc", "error"),
        matrix.Cell("mean", "breakout", "none", "utc", "error"),
        matrix.Cell("conversion", "daily", "none", "utc", "error"),
        matrix.Cell("conversion", "asof", "none", "utc", "error"),
        matrix.Cell("mean", "run", "cuped", "utc", "error"),
        matrix.Cell("mean", "run", "winsor_percentile", "utc", "error"),
        matrix.Cell("mean", "run", "cluster", "utc", "error"),
    ],
    ids=["run", "breakout", "daily", "asof", "cuped", "winsor", "cluster"],
)
def test_forwarding_provider_boundary_cell(cell: matrix.Cell) -> None:
    for mode in ("self", "adapter"):
        _check(cell, "from_definitions", mode)
        if any(
            matrix.classify(cell).legs[method]["from_unit_day_artifact"].status == "supported"
            for method in matrix.methods(cell)
        ):
            _check(cell, "from_unit_day_artifact", mode)


@pytest.mark.integration
def test_forwarding_provider_breakout_produces_rows() -> None:
    cell = matrix.Cell("mean", "breakout", "none", "utc", "error")
    with warnings.catch_warnings(record=True) as construction_warnings:
        warnings.simplefilter("always", IncrementWarning)
        builtin = matrix_cases._Ingress(cell).definitions()
    assert warning_codes(construction_warnings) == ["definition.metric.variable_window_days_none"]
    custom = _provider_pair(_forwarding(builtin, "self"))
    try:
        method = matrix.methods(cell)[0]
        expected = _outcome(builtin, cell, method)
        actual = _outcome(custom, cell, method)
        assert expected[0] == "result" and expected[1], (
            f"built-in breakout produced no rows: {expected!r}"
        )
        assert actual[0] == "result" and actual[1], (
            f"forwarding breakout produced no rows: {actual!r}"
        )
    finally:
        custom[1].close()
        _close_parity_binding(builtin)


@pytest.mark.integration
def test_forwarding_provider_sequential_checkpoint_rows_match(con) -> None:
    from datetime import date

    from increment.sequential_source import adopt_source_snapshot
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    provider_pair = None
    try:
        source = held_source(native[1])
        checkpoint = workflows.capture_sequential(*native, finalized=True, as_of=date(2025, 7, 1))
        adopt_source_snapshot(source, checkpoint)
        provider_pair = _provider_pair(_forwarding(native, "self"))
        expected = _payload(view_readout(*native))
        actual = _payload(view_readout(*provider_pair))
        assert expected, "built-in sequential checkpoint produced no rows"
        assert nested_close(expected, actual), (
            f"sequential checkpoint parity: built-in={expected!r}, provider={actual!r}"
        )
    finally:
        if provider_pair is not None:
            provider_pair[1].close()
        native[1].close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.id) for case in _triggered_day_axis_cases()],
)
def test_forwarding_provider_triggered_population_boundary(case) -> None:
    from tests.parity_harness.runner import _read

    divergences = []
    for ingress in ("from_definitions", "from_unit_day_artifact"):
        for mode in ("self", "adapter"):
            builtin = case.build[ingress]()
            custom = _provider_pair(_forwarding(builtin, mode))
            try:

                def capture(analysis, ingress=ingress):
                    try:
                        return "result", _payload(_read(analysis, ingress, case))
                    except TypeError as exc:
                        pytest.fail(f"{case.id}: harness TypeError: {exc}", pytrace=True)
                    except CodedError as exc:
                        return "refusal", exc.code
                    except Exception as exc:
                        return "exception", (type(exc).__qualname__, str(exc))

                expected = capture(builtin)
                actual = capture(custom)
                if (
                    expected[0] != actual[0]
                    or (expected[0] == "result" and not nested_close(expected[1], actual[1]))
                    or (expected[0] != "result" and expected != actual)
                ):
                    divergences.append(
                        f"{case.id}/{mode}/{ingress}: built-in={expected!r}, provider={actual!r}"
                    )
            finally:
                custom[1].close()
                _close_parity_binding(builtin)
    assert not divergences, f"forwarding provider divergences: {divergences}"
