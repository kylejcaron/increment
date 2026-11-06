from __future__ import annotations

import inspect
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import pytest

from increment.errors import CodedError


def test_unknown_row_kind_refuses_with_code():
    from increment.result_envelope import ResultCell

    fixture = (
        Path(__file__).parent / "goldens" / "decision_wire" / "result_envelope_v1.payload.json"
    )
    payload = json.loads(fixture.read_text())
    payload.update(
        schema_version=3,
        row_kind="harm_bounds",
        row_payload={"payload_version": 1, "data": {}},
    )

    with pytest.raises(CodedError) as raised:
        ResultCell.from_payload(payload)

    assert raised.value.code == "results.cell.schema_unsupported"


def test_unknown_design_mechanism_refuses_with_code():
    from increment.semantics.design import parse_design

    with pytest.raises(CodedError) as raised:
        parse_design({"mechanism": "stepped_wedge"})

    assert raised.value.code == "design.mechanism_unsupported"
    assert raised.value.context == {
        "mechanism": "stepped_wedge",
        "supported": ("randomized", "encouragement", "observational"),
    }


def test_experiment_reader_rejects_unknown_design_mechanism_before_execution():
    from increment.errors import InvalidRequestError
    from increment.kernel.experiment import Experiment

    fixture = Path(__file__).parent / "fixtures" / "definitions_experiment_reference.json"
    fixture_payload = json.loads(fixture.read_text())
    payload = fixture_payload["experiment"]
    payload["design"]["mechanism"] = "stepped_wedge"

    with pytest.raises(InvalidRequestError) as raised:
        Experiment.from_json(json.dumps(payload))

    assert type(raised.value) is InvalidRequestError
    assert raised.value.code == "design.mechanism_unsupported"
    assert raised.value.context == {
        "mechanism": "stepped_wedge",
        "supported": ("randomized", "encouragement", "observational"),
    }
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], raised.value.context)["mechanism"] = "randomized"


_FIXTURES = Path(__file__).parent / "fixtures"


def _read_saved_experiment(fixture: str) -> None:
    from increment.kernel.experiment import Experiment

    Experiment.from_json((_FIXTURES / fixture).read_text())


def _replay_exported_moments(fixture: str) -> None:
    from increment import MetricSpec, from_moments

    from_moments(
        json.loads((_FIXTURES / fixture).read_text()),
        metrics=[
            MetricSpec(name="revenue", type="mean", preferred_direction="increase"),
            MetricSpec(name="latency", type="mean", preferred_direction="decrease"),
        ],
        control="control",
    )


@pytest.mark.parametrize(
    ("read_back", "fixture", "route"),
    [
        pytest.param(
            _read_saved_experiment,
            "experiment_margin_less_guardrail_3a5b95c6_prechange.json",
            "declare ExperimentMetric(metric=..., margin=...) or margin_abs=...",
            id="experiment",
        ),
        pytest.param(
            _replay_exported_moments,
            "moments_margin_less_guardrail_80e98c86_prechange.json",
            "re-export the moments with a declared margin, or rebuild with plan= declaring "
            "ExperimentMetric(metric=..., margin=...) or margin_abs=...",
            id="moments",
        ),
        # Sequential encouragement builds no shifted null, so a margin is no route there.
        pytest.param(
            _read_saved_experiment,
            "experiment_sequential_encouragement_margin_less_guardrail_80e98c86_prechange.json",
            "list the metric as a secondary, or use a fixed-horizon plan (inference=None)",
            id="sequential_encouragement_experiment",
        ),
        # A checkpoint cube replays only its registered plan, and keeps no design.
        pytest.param(
            _replay_exported_moments,
            "moments_sequential_encouragement_margin_less_guardrail_80e98c86_prechange.json",
            "rebuild the experiment and re-export the moments with "
            "ExperimentMetric(metric=..., margin=...), or, under an encouragement design, with "
            "the metric listed as a secondary or a fixed-horizon plan (inference=None)",
            id="sequential_moments",
        ),
    ],
)
def test_saved_margin_less_guardrail_refuses_with_route(read_back, fixture, route):
    """An experiment saved, or a moments cube exported, before a guardrail needed a margin
    refuses when it is read back, so it never reaches a run; the refusal names the guardrail
    and a declaration that fixes it under the saved design and inference."""
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        read_back(fixture)

    assert raised.value.code == "plan.guardrail.margin_required"
    assert dict(raised.value.context) == {"metric": "latency", "route": route}


def test_evidence_provider_required_methods_are_frozen():
    from increment.evidence import EvidenceProvider

    required = {
        name
        for name, member in inspect.getmembers(EvidenceProvider)
        if not name.startswith("_") and callable(member)
    }

    assert required == {
        "moments",
        "breakout_method_moments",
        "unit_rows",
        "method_fit",
        "compliance",
        "compliance_series",
        "counts",
        "capture_sequential",
        "scope",
    }
