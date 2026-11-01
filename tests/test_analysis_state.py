"""Declared-plan roles on definitions and frame builders."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import ibis
import pyarrow as pa
import pytest

from increment import from_definitions, from_frame
from increment.semantics.models import AnalysisPlan, ConversionMetric
from increment.workflows import experiment_definition, metrics
from tests.analysis_factory import _make_event_log_table, make_binding_like, view_readout


def _run_roles(experiment: Any, resolver: Any) -> set[tuple[str, str | None]]:
    """The (metric, declared-plan role) pairs a whole-window run reports.

    A refused cell warns; these assertions are about the reported roles.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {(row.metric, row.role) for row in view_readout(experiment, resolver)}


def _aov_decomposition_rows() -> list[dict]:
    """Enrollment and purchase rows for ``aov_decomposition``.

    Four units per arm, plus one purchase past every metric window so the
    observable data bound clears the last cohort's window close.
    """
    rows = []
    for group_id in ("control", "treatment"):
        for index in range(4):
            unit = f"{group_id[0]}a{index:02d}"
            base = {
                "user_id": unit,
                "session_id": f"{unit}_s",
                "experiment_id": "aov_decomposition",
                "group_id": group_id,
                "country_code": "US",
                "device_type": "web",
                "plan": "free",
                "duration_s": None,
            }
            rows.append(
                base
                | {
                    "event_at": datetime(2025, 5, 2, 9, index, 0),
                    "event": "page_view",
                    "revenue": None,
                }
            )
            for order in range(1 + (index % 2)):
                rows.append(
                    base
                    | {
                        "event_at": datetime(2025, 5, 3, 9, order, 0),
                        "event": "purchase",
                        "revenue": 20.0 + index,
                    }
                )
            rows.append(
                base
                | {
                    "event_at": datetime(2025, 5, 20, 9, index, 0),
                    "event": "purchase",
                    "revenue": 5.0,
                }
            )
    return rows


def test_full_plan_resolves_every_declared_role(tmp_path):
    """A plan declaring a primary, a secondary, and a guardrail resolves
    each metric to its own role on the rows a run reports."""
    events = []
    for unit, group in (
        ("u1", "control"),
        ("u2", "control"),
        ("u3", "treatment"),
        ("u4", "treatment"),
    ):
        events.append(
            {
                "unit_id": unit,
                "event_at": datetime(2025, 1, 1),
                "experiment_id": "full_plan_exp",
                "group_id": group,
                "event": "signup",
                "revenue": None,
                "clicks": None,
                "errors": None,
            }
        )
        events.append(
            {
                "unit_id": unit,
                "event_at": datetime(2025, 1, 2),
                "experiment_id": "full_plan_exp",
                "group_id": group,
                "event": "outcome",
                "revenue": 10.0 if unit in ("u1", "u3") else None,
                "clicks": 3.0 if unit in ("u1", "u4") else None,
                "errors": 1.0 if unit in ("u2", "u4") else None,
            }
        )
    events_con = ibis.duckdb.connect()
    events_con.create_table("full_plan_events", obj=events)
    definitions_yaml = tmp_path / "definitions.yaml"
    definitions_yaml.write_text(
        """
fact_sources:
  - name: raw
    sql: |
      SELECT * FROM full_plan_events
    timestamp_column: event_at
    entities:
      - unit_id
    facts:
      - name: signup
        column: null
      - name: convert
        column: revenue
      - name: click
        column: clicks
      - name: error
        column: errors

exposures:
  - name: on_signup
    sql: |
      SELECT unit_id, event_at AS ts, group_id
      FROM full_plan_events
      WHERE experiment_id = 'full_plan_exp' AND event = 'signup'

metrics:
  - type: conversion
    name: conversion_rate
    entity: unit_id
    fact: convert
    window_days: 7
    preferred_direction: increase
  - type: conversion
    name: click_rate
    entity: unit_id
    fact: click
    window_days: 7
    preferred_direction: increase
  - type: conversion
    name: error_rate
    entity: unit_id
    fact: error
    window_days: 7
    preferred_direction: decrease

experiments:
  - name: full_plan_exp
    exposure: on_signup
    unit: unit_id
    start: "2025-01-01"
    end: "2025-02-01"
    control_group: control
    plan:
      primary: conversion_rate
      secondaries:
        - click_rate
      guardrails:
        - error_rate
"""
    )
    roles = _run_roles(
        *from_definitions(definitions_yaml, events_con, experiment_name="full_plan_exp")
    )
    assert ("conversion_rate", "primary") in roles
    assert ("click_rate", "secondary") in roles
    assert ("error_rate", "guardrail") in roles


@pytest.mark.slow
def test_plan_declaring_only_secondaries_resolves_every_test_as_secondary(example_definitions):
    """The minimal migration case - a plan with just ``secondaries:`` -
    resolves every declared test to role="secondary" under a real declared
    (not implicit-default) plan."""
    con = ibis.duckdb.connect()
    _make_event_log_table(con, extra_rows=_aov_decomposition_rows())
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="aov_decomposition"
    )
    assert {role for _, role in _run_roles(experiment, resolver)} == {"secondary"}


@pytest.mark.slow
def test_constructor_with_new_experiment_resolves_plan_against_it(con, example_definitions):
    """Constructing from a different declared experiment resolves its plan
    fresh rather than carrying the original experiment's roles forward."""
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    assert ("avg_session_duration", "secondary") in _run_roles(experiment, resolver)

    declared = experiment_definition(experiment, resolver)
    replaced = make_binding_like(
        resolver,
        experiment=declared.model_copy(
            update={
                "plan": AnalysisPlan(
                    primary="avg_session_duration", guardrails=declared.plan.guardrails
                )
            }
        ),
    )
    assert ("avg_session_duration", "primary") in _run_roles(*replaced)


def test_constructor_with_new_experiment_after_metric_narrowing_keeps_plan_catalog(
    con, example_definitions
):
    """A narrowed metric selection still resolves a replacement experiment
    against that experiment's complete declared catalog."""
    full_experiment, full_resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    full_metrics = metrics(full_experiment, full_resolver)
    purchase_rate = next(m for m in full_metrics if m.name == "purchase_rate")
    narrow_plan = AnalysisPlan(secondaries=("purchase_rate",))
    narrow_experiment, narrow_resolver = make_binding_like(
        full_resolver, [purchase_rate], plan=narrow_plan
    )  # narrows away avg_session_duration/d7_retention
    assert {metric.name for metric in metrics(narrow_experiment, narrow_resolver)} == {
        "purchase_rate"
    }
    declared = experiment_definition(narrow_experiment, narrow_resolver)
    complete_metrics = [metric for metric in full_metrics if metric.name in declared.metric_names]
    replaced = make_binding_like(
        narrow_resolver,
        complete_metrics,
        experiment=declared.model_copy(
            update={
                "plan": AnalysisPlan(
                    primary="avg_session_duration", guardrails=declared.plan.guardrails
                )
            }
        ),
    )
    assert ("avg_session_duration", "primary") in _run_roles(*replaced)


@pytest.mark.slow
def test_constructor_with_new_metric_adds_a_plan_test_entry(con, example_definitions):
    """Constructing with an additional metric gives it a real plan entry."""
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    new_metric = ConversionMetric(
        name="revenue_extra_for_plan_test",
        entity="user_id",
        fact="purchase",
        window_days=14,
    )
    assert new_metric.name not in {metric for metric, _ in _run_roles(experiment, resolver)}

    extended = make_binding_like(resolver, [*metrics(experiment, resolver), new_metric])

    assert (new_metric.name, "unassigned") in _run_roles(*extended)


def test_constructor_with_narrowed_then_adhoc_metrics_keeps_plan(con, example_definitions):
    """Narrowing then adding an ad-hoc metric keeps plan resolution valid."""
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    purchase_rate = next(m for m in metrics(experiment, resolver) if m.name == "purchase_rate")
    narrow_plan = AnalysisPlan(secondaries=("purchase_rate",))
    narrow_experiment, narrow_resolver = make_binding_like(
        resolver, [purchase_rate], plan=narrow_plan
    )  # narrows away avg_session_duration/d7_retention

    adhoc_metric = ConversionMetric(
        name="revenue_extra_for_narrow_then_add_test",
        entity="user_id",
        fact="purchase",
        window_days=14,
    )
    extended = make_binding_like(
        narrow_resolver,
        [*metrics(narrow_experiment, narrow_resolver), adhoc_metric],
        plan=narrow_plan,
    )  # must not raise

    roles = _run_roles(*extended)
    assert ("purchase_rate", "secondary") in roles
    assert (adhoc_metric.name, "unassigned") in roles


def test_from_source_seam_instance_has_a_working_plan_property() -> None:
    """A frame builder without a declared plan reports role-less rows."""
    frame = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [10.0, 12.0, 15.0, 18.0],
        }
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert {role for _, role in _run_roles(experiment, resolver)} == {None}


pytestmark = pytest.mark.integration
