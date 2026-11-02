from __future__ import annotations

import inspect
import json
from pathlib import Path

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
