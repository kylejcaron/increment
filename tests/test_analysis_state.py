"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import ibis
import pyarrow as pa

from increment import Analysis
from increment.semantics.models import AnalysisPlan, ConversionMetric
from tests.analysis_factory import _make_event_log_table, make_analysis_like

_ANALYSIS_SOURCE_SHIMS = {
    "_materialized",
    "_materialize",
    "_panel_cache",
    "_reduction_calls",
    "_mixed_assignment_units",
    "_site_volume_row",
    "_day_axis_source",
    "_breakout_moments_source",
    "_get_exposures",
    "_get_trigger_population",
    "_validate_mixed_assignments",
    "_metric_primary_events",
    "_union_event_horizon",
    "_note_reduction",
    "_build_exposure_events_table",
    "_data_as_of",
    "_resolve_breakout_props",
    "_build_pre_events",
    "_resolve_uptake",
    "_build_metric_summary",
    "_panel_sql_for_metrics",
    "_summary_sql_for_metrics",
    "_moments_for_metrics",
}


def _run_roles(analysis: Analysis) -> set[tuple[str, str | None]]:
    """The (metric, declared-plan role) pairs ``Analysis.run`` reports.

    A refused cell warns; these assertions are about the reported roles.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {(row.metric, row.role) for row in analysis.run()}


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


def test_analysis_has_no_private_source_forwarding_shims():
    """Private source implementation belongs on ``Analysis._src``.

    The broader ratchet against tests reaching into those private attributes
    lives in tests/test_test_hygiene.py::test_no_private_attribute_reach_ins.
    """
    assert _ANALYSIS_SOURCE_SHIMS.isdisjoint(vars(Analysis))


def test_full_plan_resolves_every_declared_role(tmp_path):
    """A plan declaring a primary, a secondary, and a guardrail resolves
    each metric to its own role on the rows ``Analysis.run`` reports."""
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
    analysis = Analysis("full_plan_exp", definitions_path=definitions_yaml, con=events_con)
    roles = _run_roles(analysis)
    assert ("conversion_rate", "primary") in roles
    assert ("click_rate", "secondary") in roles
    assert ("error_rate", "guardrail") in roles


def test_plan_declaring_only_secondaries_resolves_every_test_as_secondary():
    """The minimal migration case - a plan with just ``secondaries:`` -
    resolves every declared test to role="secondary" under a real declared
    (not implicit-default) plan."""
    con = ibis.duckdb.connect()
    _make_event_log_table(con, extra_rows=_aov_decomposition_rows())
    analysis = Analysis("aov_decomposition", definitions_path="examples/definitions/", con=con)
    assert {role for _, role in _run_roles(analysis)} == {"secondary"}


def test_constructor_with_new_experiment_resolves_plan_against_it(con):
    """Constructing from a different declared experiment resolves its plan
    fresh rather than carrying the original experiment's roles forward."""
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    assert ("avg_session_duration", "secondary") in _run_roles(analysis)

    analysis = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    primary="avg_session_duration",
                    guardrails=analysis.experiment.plan.guardrails,
                )
            }
        ),
    )
    assert ("avg_session_duration", "primary") in _run_roles(analysis)


def test_constructor_with_new_experiment_after_metric_narrowing_keeps_plan_catalog(con):
    """A narrowed metric selection still resolves a replacement experiment
    against that experiment's complete declared catalog."""
    full = Analysis.from_definitions("new_onboarding_v2", "examples/definitions", con, store="none")
    purchase_rate = next(m for m in full.metrics if m.name == "purchase_rate")
    narrow_plan = AnalysisPlan(secondaries=("purchase_rate",))
    analysis = make_analysis_like(
        full, [purchase_rate], plan=narrow_plan
    )  # narrows away avg_session_duration/d7_retention
    assert {metric.name for metric in analysis.metrics} == {"purchase_rate"}
    complete_metrics = [
        metric for metric in full.metrics if metric.name in analysis.experiment.metric_names
    ]
    analysis = make_analysis_like(
        analysis,
        complete_metrics,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    primary="avg_session_duration",
                    guardrails=analysis.experiment.plan.guardrails,
                )
            }
        ),
    )
    assert ("avg_session_duration", "primary") in _run_roles(analysis)


def test_constructor_with_new_metric_adds_a_plan_test_entry(con):
    """Constructing with an additional metric gives it a real plan entry."""
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    new_metric = ConversionMetric(
        name="revenue_extra_for_plan_test",
        entity="user_id",
        fact="purchase",
        window_days=14,
    )
    assert new_metric.name not in {metric for metric, _ in _run_roles(analysis)}

    analysis = make_analysis_like(analysis, [*analysis.metrics, new_metric])

    assert (new_metric.name, "unassigned") in _run_roles(analysis)


def test_constructor_with_narrowed_then_adhoc_metrics_keeps_plan(con):
    """Narrowing then adding an ad-hoc metric keeps plan resolution valid."""
    analysis = Analysis.from_definitions(
        "new_onboarding_v2", "examples/definitions", con, store="none"
    )
    purchase_rate = next(m for m in analysis.metrics if m.name == "purchase_rate")
    narrow_plan = AnalysisPlan(secondaries=("purchase_rate",))
    analysis = make_analysis_like(
        analysis, [purchase_rate], plan=narrow_plan
    )  # narrows away avg_session_duration/d7_retention

    adhoc_metric = ConversionMetric(
        name="revenue_extra_for_narrow_then_add_test",
        entity="user_id",
        fact="purchase",
        window_days=14,
    )
    analysis = make_analysis_like(
        analysis, [*analysis.metrics, adhoc_metric], plan=narrow_plan
    )  # must not raise

    roles = _run_roles(analysis)
    assert ("purchase_rate", "secondary") in roles
    assert (adhoc_metric.name, "unassigned") in roles


def test_from_source_seam_instance_has_a_working_plan_property() -> None:
    """``Analysis._from_source`` (bypassing ``__init__``) must retain the
    default un-declared plan, reported as role-less rows."""
    frame = pa.table(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["control", "control", "treatment", "treatment"],
            "revenue": [10.0, 12.0, 15.0, 18.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    assert {role for _, role in _run_roles(analysis)} == {None}
