"""Report-only metric types: total and active."""

import pytest
from pydantic import ValidationError

from increment.errors import DefinitionError, InvalidRequestError
from increment.semantics.models import ActiveMetric, Definitions, TotalMetric

FS = {
    "name": "events",
    "sql": "SELECT * FROM events",
    "timestamp_column": "event_at",
    "entities": ["user_id"],
    "facts": [
        {"name": "page_view", "column": None},
        {"name": "purchase", "column": "revenue"},
    ],
    "properties": [{"name": "country", "column": "country_code", "dtype": "string"}],
}


def test_total_metric_parses_without_entity():
    m = TotalMetric(name="weekly_revenue", fact="purchase", aggregation="sum")
    assert m.type == "total"
    assert "entity" not in TotalMetric.model_fields


def test_active_metric_requires_entity():
    m = ActiveMetric(name="wau", entity="user_id", fact="page_view")
    assert m.type == "active"
    with pytest.raises(ValidationError):
        ActiveMetric(name="wau", fact="page_view")  # ty: ignore[missing-argument]


def test_total_and_active_join_the_metric_union():
    defs = Definitions.model_validate(
        {
            "fact_sources": [FS],
            "metrics": [
                {
                    "type": "total",
                    "name": "weekly_revenue",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {"type": "active", "name": "wau", "entity": "user_id", "fact": "page_view"},
            ],
        }
    )
    assert defs.metric("weekly_revenue").type == "total"  # ty: ignore[unresolved-attribute]
    assert defs.metric("wau").type == "active"  # ty: ignore[unresolved-attribute]


def test_total_value_aggregation_rejected_on_occurrence_fact():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [FS],
                "metrics": [
                    {"type": "total", "name": "bad", "fact": "page_view", "aggregation": "sum"}
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_value.metric_uses_aggregation" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


@pytest.mark.parametrize(
    "metric",
    [
        {"type": "total", "name": "weekly_revenue", "fact": "purchase", "aggregation": "sum"},
        {"type": "active", "name": "wau", "entity": "user_id", "fact": "page_view"},
    ],
)
def test_report_only_metrics_rejected_in_experiments(metric):
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [FS],
                "exposures": [{"name": "first_pv", "fact": "page_view"}],
                "metrics": [metric],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "first_pv",
                        "unit": "user_id",
                        "start": "2026-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": [metric["name"]]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.metric_report_type" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_report_only_rejection_is_the_only_error():
    """A report-only metric in experiment.metrics must not also trip the
    unrelated entity-mismatch check (it has no entity to mismatch)."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [FS],
                "exposures": [{"name": "first_pv", "fact": "page_view"}],
                "metrics": [
                    {
                        "type": "total",
                        "name": "weekly_revenue",
                        "fact": "purchase",
                        "aggregation": "sum",
                    }
                ],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "first_pv",
                        "unit": "user_id",
                        "start": "2026-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["weekly_revenue"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.metric_report_type" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }
    errors = exc_info.value.context["errors"]
    assert isinstance(errors, tuple)
    assert len(errors) == 1


def test_report_only_metrics_rejected_in_guardrails():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [FS],
                "exposures": [{"name": "first_pv", "fact": "page_view"}],
                "metrics": [
                    {"type": "active", "name": "wau", "entity": "user_id", "fact": "page_view"},
                    {"type": "conversion", "name": "cvr", "entity": "user_id", "fact": "purchase"},
                ],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "first_pv",
                        "unit": "user_id",
                        "start": "2026-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["cvr"], "guardrails": ["wau"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.metric_report_type" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_active_filters_validated_against_source_properties():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [FS],
                "metrics": [
                    {
                        "type": "active",
                        "name": "wau_xx",
                        "entity": "user_id",
                        "fact": "page_view",
                        "filters": [{"property": "no_such_prop", "op": "equals", "values": ["x"]}],
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_filters.filter_references_unknown" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


@pytest.mark.parametrize(
    "metric,code",
    [
        (
            {
                "type": "total",
                "name": "t",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            },
            "definition.total.metric_window_days",
        ),
        (
            {
                "type": "active",
                "name": "a",
                "entity": "user_id",
                "fact": "page_view",
                "window_days": 7,
            },
            "definition.active.metric_window_days",
        ),
    ],
)
def test_report_only_metrics_reject_window_days(metric, code):
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate({"fact_sources": [FS], "metrics": [metric]})
    assert exc_info.value.code == code


def test_definitions_week_start_scalar():
    assert Definitions.model_validate({}).week_start == "monday"
    assert Definitions.model_validate({"week_start": "sunday"}).week_start == "sunday"
    with pytest.raises(InvalidRequestError) as raised:
        Definitions.model_validate({"week_start": "tuesday"})
    assert raised.value.code == "model.field.literal"
