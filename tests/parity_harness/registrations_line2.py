from __future__ import annotations

from collections.abc import Mapping

_LINE2_SCENARIOS = (
    "breakout_store_dimension",
    "breakout_boolean_store_dimension",
    "conversion_route_breakout_segments",
    "audit-retention-breakout-cohorts",
)
_LINE2_INGRESSES = (
    "from_definitions",
    "from_moments",
    "from_unit_panel",
    "from_unit_summary",
)
LINE2_PAIRS = frozenset(
    (scenario, ingress) for scenario in _LINE2_SCENARIOS for ingress in _LINE2_INGRESSES
)
_ASOF_SCENARIOS = tuple(f"asof-{scenario}" for scenario in _LINE2_SCENARIOS) + tuple(
    f"asof_lift-{scenario}" for scenario in _LINE2_SCENARIOS
)
LINE2_ASOF_PAIRS = frozenset(
    (scenario, ingress) for scenario in _ASOF_SCENARIOS for ingress in _LINE2_INGRESSES
)


def line2_actual_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:  # noqa: PLR0915
    import json
    import warnings
    from dataclasses import replace
    from datetime import UTC, datetime
    from typing import Any, cast

    from increment.errors import CodedError, InvalidRequestError
    from increment.result_envelope import AnalysisEnvelope
    from scripts.capture_evidence_identity import _aggregate_evidence, _population_identity
    from tests.parity_harness.cases import PARITY_CASES, _close_parity_binding
    from tests.parity_harness.runner import _read
    from tests.parity_harness.views import read

    cases = {case.id: case for case in PARITY_CASES}
    asof_view = None
    if scenario_id.startswith("asof_lift-"):
        asof_view, base_scenario = "asof_lift", scenario_id.removeprefix("asof_lift-")
    elif scenario_id.startswith("asof-"):
        asof_view, base_scenario = "asof", scenario_id.removeprefix("asof-")
    else:
        base_scenario = scenario_id
    case = cases[base_scenario]
    if asof_view is not None:
        case = replace(case, id=scenario_id, view=asof_view)
    if ingress not in case.build:
        return {
            "kind": "absent",
            "reason": case.waive.get(ingress, "constructor ingress is not declared"),
        }
    aggregate_rows: list[dict[str, object]] = []
    payloads: dict[str, object] = {}
    try:
        bound = case.build[ingress]()
    except CodedError as error:
        return {
            "kind": "refusal",
            "code": error.code,
            "stage": "construct",
            "context": dict(error.context),
        }
    if ingress == "from_moments":
        if case.moment_input_rows is None:
            raise AssertionError(f"{scenario_id} has no public moments input builder")
        aggregate_rows.extend(case.moment_input_rows())
    try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (
                    asof_view is not None
                    or ingress == "from_moments"
                    or (
                        base_scenario == "audit-retention-breakout-cohorts"
                        and ingress == "from_unit_summary"
                    )
                ):
                    results = _read(bound, ingress, case)
                else:
                    options: dict[str, Any] = {}
                    if case.metrics is not None:
                        options["metrics"] = case.metrics
                    if case.decision_method is not None:
                        options["decision_method"] = case.decision_method
                    results = read(bound, "breakout", population="assigned", **options)
            metadata_value = getattr(results, "metadata", None)
            cells = []
            daily_values = []

            for row in results:
                converter = getattr(row, "as_cells", None)
                if callable(converter):
                    cells.extend(
                        cast(Any, converter)(
                            metadata_value,
                            created_at=datetime(2026, 1, 1, tzinfo=UTC),
                        )
                    )
                else:
                    daily_values.append(row.model_dump(mode="json"))
            population_identity = _population_identity(bound, case, cells)
        except CodedError as error:
            outcome = {
                "kind": "refusal",
                "code": error.code,
                "stage": "run",
                "context": dict(error.context),
            }
            aggregate = _aggregate_evidence(payloads, aggregate_rows)
            if aggregate is not None:
                payload_ref = aggregate["moments"]
                aggregate["moments"] = payloads[payload_ref["payload_hash"]]
                outcome["aggregate_evidence"] = aggregate
            return outcome

        family_selection: dict[str, object] = {"status": "unavailable"}
        if cells:
            try:
                from increment.estimation.family import select_exploratory_family

                selected = select_exploratory_family(list(results), q=0.05)
                family_selection = {
                    "status": "available",
                    "q": 0.05,
                    "results": json.loads(
                        json.dumps([row.model_dump(mode="json") for row in selected])
                    ),
                }
            except (CodedError, InvalidRequestError) as error:
                family_selection = {
                    "status": "refusal",
                    "code": error.code,
                    "context": dict(error.context),
                }
        outcome = {
            "kind": "cells",
            "cells": {cell.identity.hash(): cell for cell in cells},
            "daily_values": daily_values,
            "family_selection": family_selection,
            "metadata": AnalysisEnvelope(tuple(cells), metadata_value).to_payload()["metadata"],
            "population_identity": population_identity,
        }
        if aggregate := _aggregate_evidence(payloads, aggregate_rows):
            payload_ref = aggregate["moments"]
            aggregate["moments"] = payloads[payload_ref["payload_hash"]]
            outcome["aggregate_evidence"] = aggregate
        return outcome
    finally:
        _close_parity_binding(bound)
