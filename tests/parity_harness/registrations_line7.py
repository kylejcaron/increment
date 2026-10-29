from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path


@cache
def _artifact_outcomes() -> Mapping[str, Mapping[str, object]]:
    from scripts.capture_evidence_identity import load_operations_capture

    capture = load_operations_capture(
        Path(__file__).parents[1]
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    return {
        scenario_id: ingresses["from_unit_day_artifact"]
        for scenario_id, ingresses in capture["scenarios"].items()
        if "from_unit_day_artifact" in ingresses
    }


def _artifact_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (scenario_id, "from_unit_day_artifact") for scenario_id in _artifact_outcomes()
    )


PAIRS = _artifact_pairs()


def line7_actual_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    from tests.parity_harness.cases import PARITY_CASES
    from tests.parity_harness.registrations_line4 import LINE4_PAIRS
    from tests.test_evidence_identity_parity import _line1_actual_outcome

    line4 = (scenario_id, "from_definitions") in LINE4_PAIRS
    case = next(case for case in PARITY_CASES if case.id == scenario_id)
    outcome = _line1_actual_outcome(
        scenario_id,
        ingress,
        scoped=case.sequential and not line4,
        line4=line4,
    )
    if outcome.get("kind") == "cells":
        return {**outcome, "_artifact_generation_identity": True}
    return outcome


ADAPTERS: Mapping[tuple[str, str], Callable[[str, str], Mapping[str, object]]] = dict.fromkeys(
    PAIRS, line7_actual_outcome
)
