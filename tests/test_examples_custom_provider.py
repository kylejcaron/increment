from __future__ import annotations

import runpy
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from increment import CapabilityError, Metric
from increment.estimation.engine import estimate_lift
from increment.evidence import (
    DayAxisRequest,
    EvidenceProvider,
    EvidenceScope,
    EvidenceScopeResult,
    evidence_provider,
    request_evidence,
)


def test_custom_evidence_provider_answers_estimable_moments() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_evidence_provider.py"
    result = runpy.run_path(str(example))["run_example"]()
    moment_rows = result["moment_rows"]

    assert moment_rows == (
        {
            "experiment_id": "custom-demo",
            "metric": "revenue",
            "group_id": "control",
            "n": 3,
            "ref_y": 4.0,
            "cy1": 0.0,
            "cy2": 2.0,
        },
        {
            "experiment_id": "custom-demo",
            "metric": "revenue",
            "group_id": "treatment",
            "n": 3,
            "ref_y": 6.0,
            "cy1": 0.0,
            "cy2": 2.0,
        },
    )
    assert result["unsupported_code"] == "evidence.grain_unavailable"
    assert result["compliance_series_code"] == "evidence.operation_unavailable"

    metric = TypeAdapter(Metric).validate_python(
        {"type": "mean", "name": "revenue", "fact": "revenue", "entity": "customer_id"}
    )
    computation = estimate_lift([metric], moment_rows, "control")
    (estimate,) = computation.results
    assert estimate.require_lift().value == pytest.approx(0.5)


def test_example_provider_is_structural_and_refuses_day_axis_by_declaration() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "custom_evidence_provider.py")
    )
    provider = namespace["RevenueProvider"]()
    assert isinstance(provider, EvidenceProvider)
    assert evidence_provider(provider) is provider

    metric = TypeAdapter(Metric).validate_python(
        {"type": "mean", "name": "revenue", "fact": "revenue", "entity": "customer_id"}
    )
    with pytest.raises(CapabilityError) as raised:
        request_evidence(
            provider,
            DayAxisRequest(
                caller="run_daily",
                grain="daily",
                metrics=(metric,),
                dimension=None,
            ),
        )
    assert raised.value.code == "evidence.grain_unavailable"


def test_custom_provider_refuses_undeclared_assignment_count_before_dispatch() -> None:
    from dataclasses import replace

    from increment.evidence import CountsRequest

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "custom_evidence_provider.py")
    )
    base = namespace["RevenueProvider"]

    class ArtifactProvider(base):
        evidence = replace(
            base.evidence,
            kind="artifact",
            operations=frozenset({"artifact_experiment"}),
            count_grains=frozenset(),
        )
        calls = 0

        def counts(self, request):
            self.calls += 1
            raise AssertionError("undeclared count request reached provider")

    provider = ArtifactProvider()
    with pytest.raises(CapabilityError) as raised:
        request_evidence(provider, CountsRequest("assignment"))
    assert raised.value.code == "evidence.operation_unavailable"
    assert provider.calls == 0


def test_custom_provider_supports_safe_scope_and_refuses_snapshot_scope() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "custom_evidence_provider.py")
    )
    provider = namespace["RevenueProvider"]()
    scope_result = cast(EvidenceScopeResult, request_evidence(provider, EvidenceScope()))
    with scope_result.source as scoped:
        assert scoped is provider

    with pytest.raises(CapabilityError) as raised:
        request_evidence(provider, EvidenceScope(snapshot=True))
    assert raised.value.code == "evidence.operation_unavailable"
