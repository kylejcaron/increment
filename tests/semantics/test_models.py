"""Tests for the semantic models — discriminated union dispatch, alias
generation, filter arity, reference cross-checking, and extra-forbid."""

import datetime as dt
import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from increment.errors import DefinitionError, InvalidRequestError
from increment.semantics.models import (
    AdjustmentCovariate,
    AnalysisPlan,
    Breakout,
    ConversionMetric,
    Definitions,
    DimSource,
    DimValidity,
    EncouragementDeclaration,
    Experiment,
    ExperimentMetric,
    Exposure,
    Fact,
    FactSource,
    Filter,
    MeanMetric,
    Measure,
    MethodSpec,
    Metric,
    NormalPriorSpec,
    ObservationalDeclaration,
    Property,
    RatioMetric,
    RetentionMetric,
    TotalMetric,
    Winsorization,
    resolve_null_and_alternative,
)

# ── Discriminated union dispatch ───────────────────────────────────────


def test_metric_union_dispatches_on_type():
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "retention",
            "name": "d7",
            "entity": "user_id",
            "fact": "any_event",
            "threshold_days": 7,
        }
    )
    assert m.threshold_days == 7
    assert isinstance(m, RetentionMetric)


@pytest.mark.parametrize("aggregation", ["avg_event", "avg_calendar_day"])
def test_mean_metric_accepts_explicit_average_basis(aggregation):
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "mean",
            "name": "avg_rev",
            "entity": "user_id",
            "fact": "revenue",
            "aggregation": aggregation,
        }
    )
    assert isinstance(m, MeanMetric)
    assert m.aggregation == aggregation


def test_bare_avg_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        MeanMetric(
            name="avg_rev",
            entity="user_id",
            fact="revenue",
            aggregation="avg",  # ty: ignore[invalid-argument-type]
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("aggregation",)
    assert error["type"] == "literal_error"


def test_calendar_day_average_is_mean_only():
    with pytest.raises(DefinitionError) as exc_info:
        TotalMetric(name="total", fact="revenue", aggregation="avg_calendar_day")
    assert exc_info.value.code == "definition.total.metric_use_avg"
    with pytest.raises(DefinitionError) as exc_info:
        RatioMetric(
            name="ratio",
            entity="user_id",
            numerator=Measure(fact="revenue", aggregation="avg_calendar_day"),
            denominator=Measure(fact="orders", aggregation="sum"),
        )
    assert exc_info.value.code == "definition.ratio.metric_use_avg"


def test_fact_source_requires_entity():
    with pytest.raises(ValidationError) as exc_info:
        Definitions(
            fact_sources=[
                FactSource(
                    name="events",
                    sql="select * from events",
                    timestamp_column="ts",
                    entities=[],
                    facts=[Fact(name="revenue", column="revenue")],
                )
            ],
            metrics=[TotalMetric(name="total_revenue", fact="revenue", aggregation="sum")],
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("entities",)
    assert error["type"] == "too_short"


def test_mean_metric_accepts_winsorization():
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        winsorization={"upper_percentile": 0.99},
    )
    assert metric.winsorization == Winsorization(upper_percentile=0.99)
    assert metric.winsorization.has_percentile


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "definition.winsorization.least_one_bound"),
        ({"upper_percentile": 1.0}, None),
        ({"upper_value": float("inf")}, "definition.winsorization.finite"),
        (
            {"upper_percentile": 0.99, "upper_value": 10.0},
            "definition.winsorization.set_percentile_fixed",
        ),
        (
            {"lower_percentile": 0.9, "upper_percentile": 0.1},
            "definition.winsorization.lower_percentile_below",
        ),
        (
            {"lower_value": 10.0, "upper_value": 10.0},
            "definition.winsorization.lower_value_below",
        ),
    ],
)
def test_winsorization_rejects_invalid_declarations(kwargs: dict[str, float], code: str | None):
    expected = DefinitionError if code is not None else ValidationError
    with pytest.raises(expected) as exc_info:
        Winsorization.model_validate(kwargs)
    if code is not None:
        assert isinstance(exc_info.value, DefinitionError)
        assert exc_info.value.code == code


def test_non_mean_metric_rejects_winsorization():
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Metric).validate_python(
            {
                "type": "conversion",
                "name": "converted",
                "entity": "user_id",
                "fact": "purchase",
                "winsorization": {"upper_value": 1.0},
            }
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("conversion", "winsorization")
    assert error["type"] == "extra_forbidden"


def test_ratio_metric_parts_are_bare_measures():
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "ratio",
            "name": "rev_per_order",
            "entity": "user_id",
            "numerator": {"fact": "revenue", "aggregation": "sum"},
            "denominator": {"fact": "orders", "aggregation": "count"},
        }
    )
    assert m.numerator.aggregation == "sum"
    assert not hasattr(m.numerator, "entity")


def test_conversion_metric_factref_not_measure():
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "conversion",
            "name": "did_order",
            "entity": "user_id",
            "fact": "orders",
        }
    )
    assert m.type == "conversion"
    assert not hasattr(m, "aggregation")


# ── Alias generation ───────────────────────────────────────────────────


def test_alias_generation():
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "mean",
            "name": "Revenue per User",
            "entity": "user_id",
            "fact": "revenue",
            "aggregation": "sum",
        }
    )
    assert m.alias == "revenue_per_user"


def test_alias_leading_digit_prefixes_class_name():
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "mean",
            "name": "7day_active",
            "entity": "user_id",
            "fact": "login",
        }
    )
    # type(self).__name__.lower() for MeanMetric = "meanmetric"
    assert m.alias.startswith("meanmetric_")
    assert "7day_active" in m.alias


def test_alias_exposure():
    ex = Exposure(name="First Session", sql="SELECT 1")
    assert ex.alias == "first_session"


def test_alias_fact():
    f = Fact(name="Order Value", column="amount")
    assert f.alias == "order_value"


def test_alias_property():
    p = Property(name="Plan Type", column="plan")
    assert p.alias == "plan_type"


# ── Filter arity validators ────────────────────────────────────────────


def test_numeric_between_filter_requires_two_values():
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Metric).validate_python(
            {
                "type": "mean",
                "name": "big_spend",
                "entity": "user_id",
                "fact": "revenue",
                "aggregation": "sum",
                "filters": [{"property": "amount", "op": "between", "values": [50]}],
            }
        )
    assert exc_info.value.errors()[0]["ctx"]["error"].code == "definition.filter.op_exactly_values"


def test_equals_filter_requires_one_value():
    with pytest.raises(DefinitionError) as exc_info:
        Filter(property="p", op="equals", values=[])
    assert exc_info.value.code == "definition.filter.op_exactly_value"


def test_in_filter_requires_at_least_one():
    with pytest.raises(DefinitionError) as exc_info:
        Filter(property="p", op="in", values=[])
    assert exc_info.value.code == "definition.filter.op_least_value"


def test_not_in_filter_requires_at_least_one():
    with pytest.raises(DefinitionError) as exc_info:
        Filter(property="p", op="not_in", values=[])
    assert exc_info.value.code == "definition.filter.op_least_value"


def test_between_filter_happy():
    f = Filter(property="p", op="between", values=[10, 20])
    assert len(f.values) == 2


def test_in_filter_happy():
    f = Filter(property="p", op="in", values=["a", "b", "c"])
    assert len(f.values) == 3


# ── Exposure xor validator ─────────────────────────────────────────────


def test_exposure_requires_exactly_one_source():
    # Neither sql nor fact
    with pytest.raises(DefinitionError) as exc_info:
        Exposure(name="bad", sql=None, fact=None)
    assert exc_info.value.code == "definition.exposure.needs_exactly_one"

    # Both sql and fact
    with pytest.raises(DefinitionError) as exc_info:
        Exposure(name="bad", sql="SELECT 1", fact="some_fact")
    assert exc_info.value.code == "definition.exposure.needs_exactly_one"


def test_exposure_filters_with_sql_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Exposure(
            name="bad", sql="SELECT 1", filters=[{"property": "p", "op": "equals", "values": ["x"]}]
        )
    assert exc_info.value.code == "definition.exposure.filters_apply_fact"


def test_exposure_fact_based_happy():
    ex = Exposure(name="ok", fact="page_view", filters=[])
    assert ex.fact == "page_view"
    assert ex.sql is None


def test_exposure_sql_based_happy():
    ex = Exposure(name="ok", sql="SELECT * FROM users")
    assert ex.sql is not None
    assert ex.fact is None


# ── extra="forbid" at every level ──────────────────────────────────────


def test_cross_check_accumulates_every_error_and_raises_once():
    """Two independently-broken things in one Definitions tree accumulate into a single
    definition.invalid refusal carrying every (code, message) pair, not just the first."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [],
                "metrics": [
                    {"type": "conversion", "name": "m", "entity": "u", "fact": "nope"},
                    {"type": "conversion", "name": "m", "entity": "u", "fact": "nope"},
                ],
                "experiments": [
                    {
                        "name": "e",
                        "plan": {},
                        "exposure": "nope",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_metric.references_unknown_fact",
        "definition.validate_experiment.references_unknown_exposure",
        "definition.validate_metrics.duplicate_metric_name",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]
    errors = exc_info.value.context["errors"]
    assert isinstance(errors, tuple)
    codes = {c for c, _ in errors}  # ty: ignore[not-iterable]
    assert "definition.validate_metrics.duplicate_metric_name" in codes
    assert "definition.validate_experiment.references_unknown_exposure" in codes


def test_extra_forbid_at_top_level():
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        Definitions.model_validate(
            {
                "bogus_key": True,
            }
        )
    assert raised.value.code == "model.field.unknown"


def test_extra_forbid_nested_in_metric():
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Metric).validate_python(
            {
                "type": "mean",
                "name": "m",
                "entity": "u",
                "fact": "f",
                "typo_field": "nope",
            }
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("mean", "typo_field")
    assert error["type"] == "extra_forbidden"


def test_extra_forbid_nested_in_fact():
    with pytest.raises(ValidationError, match="extra"):
        Fact(name="f", column="c", typo_desc="oops")  # ty: ignore[unknown-argument]  # proving extra=forbid rejects it


def test_extra_forbid_nested_in_filter():
    with pytest.raises(ValidationError, match="extra"):
        Filter(property="p", op="equals", values=["x"], bogus_opt="y")  # ty: ignore[unknown-argument]  # proving extra=forbid rejects it


# ── Cross-reference validation on Definitions ──────────────────────────


def test_dangling_metric_fact_reference_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [],
                "metrics": [{"type": "mean", "name": "m", "entity": "u", "fact": "nope"}],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_metric.references_unknown_fact" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_dangling_exposure_reference_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [],
                "metrics": [],
                "experiments": [
                    {
                        "name": "exp",
                        "plan": {},
                        "exposure": "no_such",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.references_unknown_exposure" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_dangling_metric_reference_in_experiment_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [{"name": "e", "sql": "SELECT 1"}],
                "metrics": [],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["no_such"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.references_unknown_metrics" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_metric_entity_mismatch_with_experiment_unit():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [{"name": "ev", "column": None}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "ev"}],
                "metrics": [{"type": "mean", "name": "m", "entity": "company_id", "fact": "ev"}],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["m"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.unit_but_entity" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_ratio_metric_shorthand_with_n_pre_periods_allowed():
    """A ratio metric without a CUPED-requesting binding never touches the pre-period
    covariate, so it coexists with n_pre_periods>0; only a binding that actually requests
    CUPED is refused."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "rev", "column": "amount"},
                        {"name": "ord", "column": "id"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "r",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": ["r"]},
                    "n_pre_periods": 7,
                }
            ],
        }
    )
    assert defs.experiments[0].metric_names == ["r"]


def test_ratio_metric_cuped_binding_loads():
    """A CUPED binding on a ratio metric loads with n_pre_periods > 0: the warehouse
    covariate is the numerator's pre-period total (docs/guides/cuped.md)."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "rev", "column": "amount"},
                        {"name": "ord", "column": "id"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "r",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {
                        "secondaries": [
                            {
                                "metric": "r",
                                "decision_method": {
                                    "name": "cuped",
                                    "variance_reduction": "cuped",
                                },
                            }
                        ]
                    },
                    "n_pre_periods": 7,
                }
            ],
        }
    )
    experiment = defs.experiment("exp")
    assert experiment is not None
    assert experiment.bindings["r"].wants_cuped


def test_duplicate_metric_name_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [],
                "metrics": [
                    {"type": "mean", "name": "dup", "entity": "u", "fact": "f"},
                    {"type": "conversion", "name": "dup", "entity": "u", "fact": "f"},
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_metric.references_unknown_fact",
        "definition.validate_metrics.duplicate_metric_name",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


def test_duplicate_exposure_name_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [
                    {"name": "dup", "sql": "SELECT 1"},
                    {"name": "dup", "sql": "SELECT 2"},
                ],
                "metrics": [],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.validate_exposures.duplicate_exposure_name",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


def test_duplicate_experiment_name_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [],
                "exposures": [{"name": "e", "sql": "SELECT 1"}],
                "metrics": [],
                "experiments": [
                    {
                        "name": "dup",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    },
                    {
                        "name": "dup",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    },
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiments.duplicate_experiment_name" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_full_valid_definitions_happy_path():
    """A complete, valid set of definitions must validate without error."""
    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None, "description": "A page was viewed"},
                        {"name": "revenue", "column": "amount", "description": "Revenue per event"},
                    ],
                    "properties": [
                        {"name": "country", "column": "country_code", "dtype": "string"},
                    ],
                }
            ],
            "exposures": [
                {
                    "name": "enrolled",
                    "fact": "page_view",
                    "filters": [{"property": "country", "op": "equals", "values": ["US"]}],
                },
                {"name": "sql_enrolled", "sql": "SELECT * FROM enrollments"},
            ],
            "metrics": [
                {
                    "type": "mean",
                    "name": "avg_revenue",
                    "entity": "user_id",
                    "fact": "revenue",
                    "aggregation": "avg_event",
                },
                {
                    "type": "conversion",
                    "name": "visit_rate",
                    "entity": "user_id",
                    "fact": "page_view",
                },
                {
                    "type": "ratio",
                    "name": "rev_per_visit",
                    "entity": "user_id",
                    "numerator": {"fact": "revenue", "aggregation": "sum"},
                    "denominator": {"fact": "page_view", "aggregation": "count"},
                },
            ],
            "experiments": [
                {
                    "name": "test_v1",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2024-06-01",
                    "end": "2024-07-01",
                    "plan": {
                        "secondaries": ["avg_revenue", "rev_per_visit"],
                        "guardrails": ["visit_rate"],
                    },
                    "n_pre_periods": 0,
                    "control_group": "C",
                }
            ],
        }
    )
    assert defs.dialect == "duckdb"
    assert len(defs.fact_sources) == 1
    assert len(defs.exposures) == 2
    assert len(defs.metrics) == 3
    assert len(defs.experiments) == 1

    # Lookup methods work
    assert defs.metric("avg_revenue") is not None
    assert defs.metric("nope") is None
    assert defs.experiment("test_v1") is not None
    assert defs.experiment("nope") is None
    assert defs.fact_source_for("events") is not None
    assert defs.fact_source_for("page_view") is not None
    assert defs.fact_source_for("nope") is None


# ── Regression: filter dtype compatibility ──────────────────────────


def test_filter_dtype_mismatch_int_values_on_bool_property():
    """Values=[50] against a bool-dtype property is rejected."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "ev", "column": None}],
                        "properties": [{"name": "flag", "column": "f", "dtype": "bool"}],
                    }
                ],
                "exposures": [
                    {
                        "name": "e",
                        "fact": "ev",
                        "filters": [{"property": "flag", "op": "equals", "values": [50]}],
                    }
                ],
                "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "ev"}],
                "experiments": [
                    {
                        "name": "x",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_dtype.filter_property_incompatible" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_filter_dtype_compatible_values_pass():
    """String values on a string-dtype property pass."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": None}],
                    "properties": [{"name": "country", "column": "c", "dtype": "string"}],
                }
            ],
            "exposures": [
                {
                    "name": "e",
                    "fact": "ev",
                    "filters": [{"property": "country", "op": "in", "values": ["US", "CA"]}],
                }
            ],
            "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "ev"}],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        }
    )
    assert len(defs.metrics) == 1


def test_filter_dtype_rejects_bool_as_int():
    """bool values (True/False) are rejected for int-dtype properties."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "ev", "column": None}],
                        "properties": [{"name": "age", "column": "a", "dtype": "int"}],
                    }
                ],
                "exposures": [
                    {
                        "name": "e",
                        "fact": "ev",
                        "filters": [{"property": "age", "op": "equals", "values": [True]}],
                    }
                ],
                "metrics": [{"type": "conversion", "name": "m", "entity": "u", "fact": "ev"}],
                "experiments": [
                    {
                        "name": "x",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_dtype.filter_property_incompatible" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


# ── Regression: value aggregation on column=None fact ──────────────


def test_value_aggregation_on_occurrence_fact_rejected():
    """MeanMetric with sum aggregation on a column=None fact is rejected."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "ev", "column": None}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "ev"}],
                "metrics": [
                    {"type": "mean", "name": "m", "entity": "u", "fact": "ev", "aggregation": "sum"}
                ],
                "experiments": [
                    {
                        "name": "x",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_value.metric_uses_aggregation" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_count_aggregation_on_occurrence_fact_allowed():
    """MeanMetric with count aggregation on a column=None fact is allowed."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "ev", "column": None}],
                }
            ],
            "exposures": [{"name": "e", "fact": "ev"}],
            "metrics": [
                {"type": "mean", "name": "m", "entity": "u", "fact": "ev", "aggregation": "count"}
            ],
            "experiments": [
                {
                    "name": "x",
                    "plan": {},
                    "exposure": "e",
                    "unit": "u",
                    "start": "2024-01-01",
                    "control_group": "C",
                }
            ],
        }
    )
    assert len(defs.metrics) == 1


def test_ratio_numerator_value_aggregation_on_occurrence_fact_rejected():
    """Ratio numerator with sum aggregation on a column=None fact is rejected."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "occ", "column": None}, {"name": "val", "column": "a"}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "occ"}],
                "metrics": [
                    {
                        "type": "ratio",
                        "name": "r",
                        "entity": "u",
                        "numerator": {"fact": "occ", "aggregation": "sum"},
                        "denominator": {"fact": "val", "aggregation": "count"},
                    }
                ],
                "experiments": [
                    {
                        "name": "x",
                        "plan": {},
                        "exposure": "e",
                        "unit": "u",
                        "start": "2024-01-01",
                        "control_group": "C",
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_value.metric_uses_aggregation" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


# ── Regression: experiment.unit in source entities ─────────────────


def test_experiment_unit_not_in_exposure_source_entities():
    """Experiment unit must be in the entities of the exposure's fact source."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["session_id"],
                        "facts": [{"name": "ev", "column": None}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "ev"}],
                "metrics": [{"type": "conversion", "name": "m", "entity": "user_id", "fact": "ev"}],
                "experiments": [
                    {
                        "name": "x",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["m"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_unit.experiment_but_source" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_experiment_unit_not_in_metric_fact_source_entities():
    """Experiment unit must be in the entities of the metric's fact source."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "exp_src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [{"name": "exposure_ev", "column": None}],
                    },
                    {
                        "name": "metric_src",
                        "sql": "SELECT 2",
                        "timestamp_column": "ts",
                        "entities": ["session_id"],
                        "facts": [{"name": "metric_ev", "column": None}],
                    },
                ],
                "exposures": [{"name": "e", "fact": "exposure_ev"}],
                "metrics": [
                    {"type": "conversion", "name": "m", "entity": "session_id", "fact": "metric_ev"}
                ],
                "experiments": [
                    {
                        "name": "x",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": ["m"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_unit.experiment_but_source",
        "definition.validate_experiment.unit_but_entity",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


# ── Regression: guardrail validation parity ────────────────────────


def test_guardrail_entity_mismatch_with_experiment_unit():
    """Guardrail entity must match experiment unit (same check as metrics)."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [{"name": "ev", "column": None}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "ev"}],
                "metrics": [{"type": "mean", "name": "g", "entity": "company_id", "fact": "ev"}],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": [], "guardrails": ["g"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.unit_but_entity" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_guardrail_ratio_shorthand_with_n_pre_periods_allowed():
    """Same relaxation as test_ratio_metric_shorthand_with_n_pre_periods_allowed,
    for the guardrail namespace."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "rev", "column": "amount"},
                        {"name": "ord", "column": "id"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "g",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": [], "guardrails": ["g"]},
                    "n_pre_periods": 7,
                }
            ],
        }
    )
    assert defs.experiments[0].guardrail_names == ["g"]


def test_guardrail_ratio_cuped_binding_loads():
    """A CUPED binding on a ratio metric loads with n_pre_periods > 0: the warehouse
    covariate is the numerator's pre-period total (docs/guides/cuped.md)."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "src",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "rev", "column": "amount"},
                        {"name": "ord", "column": "id"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "sql": "SELECT 1"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "g",
                    "entity": "user_id",
                    "numerator": {"fact": "rev", "aggregation": "sum"},
                    "denominator": {"fact": "ord", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {
                        "secondaries": [],
                        "guardrails": [
                            {
                                "metric": "g",
                                "decision_method": {
                                    "name": "cuped",
                                    "variance_reduction": "cuped",
                                },
                            }
                        ],
                    },
                    "n_pre_periods": 7,
                }
            ],
        }
    )
    experiment = defs.experiment("exp")
    assert experiment is not None
    assert experiment.bindings["g"].wants_cuped


def test_experiment_unit_not_in_guardrail_fact_source_entities():
    """Experiment unit must be in the entities of the guardrail's fact source."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "exp_src",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [{"name": "exposure_ev", "column": None}],
                    },
                    {
                        "name": "guardrail_src",
                        "sql": "SELECT 2",
                        "timestamp_column": "ts",
                        "entities": ["session_id"],
                        "facts": [{"name": "guardrail_ev", "column": None}],
                    },
                ],
                "exposures": [{"name": "e", "fact": "exposure_ev"}],
                "metrics": [
                    {
                        "type": "conversion",
                        "name": "g",
                        "entity": "session_id",
                        "fact": "guardrail_ev",
                    }
                ],
                "experiments": [
                    {
                        "name": "x",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2024-01-01",
                        "control_group": "C",
                        "plan": {"secondaries": [], "guardrails": ["g"]},
                    }
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_unit.experiment_but_source",
        "definition.validate_experiment.unit_but_entity",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


# ── RetentionMetric: the observation band ──────────────────────────────


def test_retention_metric_rejects_window_days():
    """Both band edges live in threshold_days; a declared window_days raises."""
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7_retention",
            entity="user_id",
            fact="page_view",
            threshold_days=7,
            window_days=14,
        )
    assert exc_info.value.code == "definition.retention.metric_window_days"


@pytest.mark.parametrize("window_days", [0, -1])
def test_retention_metric_rejects_non_positive_window_days_with_the_coded_refusal(window_days):
    """A zero or negative ``window_days`` is the same retention hazard as a
    positive one, so ``RetentionMetric`` refuses it with the same code as
    ``MetricSpec`` rather than a raw ``ge=1`` field error."""
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7_retention",
            entity="user_id",
            fact="page_view",
            threshold_days=7,
            window_days=window_days,
        )
    assert exc_info.value.code == "definition.retention.metric_window_days"


def test_retention_metric_band_normalizes_both_shapes():
    """`band` is the single normalizer every band reader goes through."""
    bounded = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=(7, 14)
    )
    unbounded = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    assert bounded.band == (7, 14)
    assert unbounded.band == (7, None)


def test_retention_metric_rejects_empty_band():
    """An empty observation band is a definition error, not a silent all-zero metric."""
    for band in [(7, 7), (7, 3)]:
        with pytest.raises(DefinitionError) as exc_info:
            RetentionMetric(
                name="d7_retention",
                entity="user_id",
                fact="page_view",
                threshold_days=band,
            )
        assert exc_info.value.code == "definition.retention.threshold_days_upper_exceeds_lower"


def test_retention_metric_rejects_negative_band_start():
    """Days are counted from exposure (day 0), so the band cannot start before it."""
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7_retention",
            entity="user_id",
            fact="page_view",
            threshold_days=-1,
        )
    assert exc_info.value.code == "definition.retention.threshold_days_non_negative"
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7_retention",
            entity="user_id",
            fact="page_view",
            threshold_days=(-1, 14),
        )
    assert exc_info.value.code == "definition.retention.threshold_days_lower_bound_non_negative"


def test_retention_metric_accepts_both_band_shapes():
    """A [a, b] pair is a bounded band; a bare int stays valid (unbounded)."""
    bounded = RetentionMetric(
        name="d7_retention",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    assert bounded.threshold_days == (7, 14)

    unbounded = RetentionMetric(
        name="d7_retention",
        entity="user_id",
        fact="page_view",
        threshold_days=7,
    )
    assert unbounded.threshold_days == 7


def test_retention_metric_band_accepts_yaml_list():
    """The YAML declaration is a list, [a, b]: it must coerce to the tuple form."""
    m = TypeAdapter(Metric).validate_python(
        {
            "type": "retention",
            "name": "d7",
            "entity": "user_id",
            "fact": "page_view",
            "threshold_days": [7, 14],
        }
    )
    assert isinstance(m, RetentionMetric)
    assert m.threshold_days == (7, 14)


# ── Experiment.observation_end / observation_horizon ───────────────────


def test_observation_horizon_defaults_to_end():
    """An experiment that declares only `end` observes until `end`, today's behaviour."""
    exp = Experiment(
        name="e",
        unit="unit_id",
        exposure="x",
        control_group="control",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 31),
        plan=AnalysisPlan(),
    )
    assert exp.observation_end is None
    assert exp.observation_horizon == dt.datetime(2025, 1, 31)


def test_observation_horizon_uses_observation_end_when_set():
    exp = Experiment(
        name="e",
        unit="unit_id",
        exposure="x",
        control_group="control",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 31),
        observation_end=dt.datetime(2025, 2, 14),
        plan=AnalysisPlan(),
    )
    assert exp.observation_horizon == dt.datetime(2025, 2, 14)


def test_observation_end_equal_to_end_is_allowed():
    """`observation_end == end` sits exactly on the validator's `<` boundary: stopping
    observation the moment enrollment ends is a no-op extension, not a rejected inversion."""
    exp = Experiment(
        name="e",
        unit="unit_id",
        exposure="x",
        control_group="control",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 31),
        observation_end=dt.datetime(2025, 1, 31),
        plan=AnalysisPlan(),
    )
    assert exp.observation_horizon == dt.datetime(2025, 1, 31)


def test_observation_horizon_is_none_for_a_running_experiment():
    """No `end` and no `observation_end`: the panel's own extent is the bound."""
    exp = Experiment(
        name="e",
        unit="unit_id",
        exposure="x",
        control_group="control",
        start=dt.datetime(2025, 1, 1),
        plan=AnalysisPlan(),
    )
    assert exp.observation_horizon is None


def test_observation_end_before_end_is_rejected():
    """Observation cannot stop before enrollment does: that would drop enrolled units."""
    with pytest.raises(DefinitionError) as exc_info:
        Experiment(
            name="e",
            unit="unit_id",
            exposure="x",
            control_group="control",
            start=dt.datetime(2025, 1, 1),
            end=dt.datetime(2025, 1, 31),
            observation_end=dt.datetime(2025, 1, 20),
            plan=AnalysisPlan(),
        )
    assert exc_info.value.code == ("definition.experiment.observation_end_before_end")


def test_observation_end_without_end_is_rejected():
    """`observation_end` extends `end`; with no `end` there is nothing to extend."""
    with pytest.raises(DefinitionError) as exc_info:
        Experiment(
            name="e",
            unit="unit_id",
            exposure="x",
            control_group="control",
            start=dt.datetime(2025, 1, 1),
            observation_end=dt.datetime(2025, 2, 14),
            plan=AnalysisPlan(),
        )
    assert exc_info.value.code == ("definition.experiment.observation_end_requires_end")


# ── Factor declaration ──────────────────────────────────────────────


def _defs_with_factor(
    as_of: str = "static", prop: str = "country", duplicate: bool = False, dtype: str = "string"
) -> Definitions:
    """Minimal definitions with one factor on `prop`, backed by a property with the given
    `as_of`/`dtype`. `duplicate=True` declares the factor twice to exercise dedup checks."""
    factors = [{"property": prop, "source": "events"}]
    if duplicate:
        factors.append({"property": prop, "source": "events"})
    return Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": dtype,
                            "as_of": as_of,
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [],
            "experiments": [
                {
                    "name": "exp",
                    "plan": {},
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "factors": factors,
                }
            ],
        }
    )


class TestFactorDeclaration:
    def test_factor_on_a_static_property_loads(self):
        defs = _defs_with_factor(as_of="static")
        assert defs.experiments[0].factors[0].property == "country"

    def test_factor_on_a_pre_exposure_property_loads(self):
        defs = _defs_with_factor(as_of="pre_exposure")
        assert defs.experiments[0].factors[0].property == "country"

    def test_event_time_factor_is_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            _defs_with_factor(as_of="event_time")
        assert exc_info.value.code == "definition.invalid"
        assert "definition.check_factors.experiment_factor_property" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_unknown_factor_property_is_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            _defs_with_factor(as_of="static", prop="nonexistent")
        assert exc_info.value.code == "definition.invalid"
        assert "definition.resolve_breakout.experiment_property_found" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_float_dtype_factor_is_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            _defs_with_factor(as_of="static", dtype="float")
        assert exc_info.value.code == "definition.invalid"
        assert "definition.check_factors.experiment_factor_property_unsupported_dtype" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }

    def test_duplicate_factor_is_rejected(self):
        with pytest.raises(DefinitionError) as exc_info:
            _defs_with_factor(as_of="static", duplicate=True)
        assert exc_info.value.code == "definition.invalid"
        assert "definition.check_factors.experiment_duplicate_factor" in {
            c
            for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
        }


# ── preferred_direction / margin (non-inferiority guardrails) ──────────


def test_preferred_direction_defaults_increase_and_never_gates_test():
    """No margin -> the field is purely descriptive; resolved_null is inert."""
    m = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    assert m.preferred_direction == "increase"
    assert m.margin is None
    assert m.resolved_null() == (0.0, None)


def test_margin_with_increase_direction_derives_negative_null_and_greater_tail():
    """'don't lose more than 1%' -> H0: lift <= -1%, one-sided greater."""
    m = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="avg_event",
        preferred_direction="increase",
        margin=0.01,
    )
    null_lift, tail = m.resolved_null()
    assert null_lift == pytest.approx(-0.01)
    assert tail == "greater"


def test_margin_with_decrease_direction_derives_positive_null_and_less_tail():
    """latency guardrail: 'don't rise more than 2%' -> H0: lift >= +2%, less."""
    m = MeanMetric(
        name="latency_ms",
        entity="user_id",
        fact="latency",
        aggregation="avg_event",
        preferred_direction="decrease",
        margin=0.02,
    )
    null_lift, tail = m.resolved_null()
    assert null_lift == pytest.approx(0.02)
    assert tail == "less"


def test_margin_without_explicit_direction_rejected():
    """A defaulted preferred_direction must never silently pick the adverse side."""
    with pytest.raises(DefinitionError) as exc_info:
        MeanMetric(
            name="revenue", entity="user_id", fact="revenue", aggregation="avg_event", margin=0.01
        )
    assert exc_info.value.code == "definition.metric_margin_requires_direction"


def test_margin_with_neutral_direction_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="neutral",
            margin=0.01,
        )
    assert exc_info.value.code == "definition.metric_margin_non"


def test_margin_must_be_positive():
    with pytest.raises(ValidationError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="increase",
            margin=-0.01,
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("margin",)
    assert error["type"] == "greater_than"


def test_explicit_direction_without_margin_still_allowed():
    """Explicitly declaring direction (e.g. for future coloring) with no margin is fine."""
    m = MeanMetric(
        name="latency_ms",
        entity="user_id",
        fact="latency",
        aggregation="avg_event",
        preferred_direction="decrease",
    )
    assert m.resolved_null() == (0.0, None)


def test_margin_at_or_above_one_with_increase_rejected_at_declaration():
    """margin >= 1 with preferred_direction='increase' implies null_lift <= -1, a >=100%
    relative loss with no log-scale representation; the declaration validator must catch it
    by name, not leave a run-time null_lift error deep in inference."""
    for bad in (1.0, 1.5):
        with pytest.raises(DefinitionError) as exc_info:
            MeanMetric(
                name="revenue",
                entity="user_id",
                fact="revenue",
                aggregation="avg_event",
                preferred_direction="increase",
                margin=bad,
            )
        assert exc_info.value.code == "definition.metric_margin_implies_unrepresentable_null_lift"


def test_margin_above_one_with_decrease_still_allowed():
    """The decrease side has no singularity: null_lift = +margin is
    always representable."""
    m = MeanMetric(
        name="latency_ms",
        entity="user_id",
        fact="latency",
        aggregation="avg_event",
        preferred_direction="decrease",
        margin=1.5,
    )
    null_lift, tail = m.resolved_null()
    assert null_lift == pytest.approx(1.5)
    assert tail == "less"


# ── resolve_null_and_alternative (call-time margins=/null_lifts= precedence) ──


def _increase_metric(margin: float | None = None) -> MeanMetric:
    if margin is not None:
        return MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="increase",
            margin=margin,
        )
    return MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="avg_event",
        preferred_direction="increase",
    )


def test_resolve_no_overrides_no_declared_margin_is_inert():
    m = _increase_metric()
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, None, "two-sided")
    assert (null_lift, null_abs, alt) == (0.0, None, "two-sided")


def test_resolve_uses_declared_margin_when_no_override():
    m = _increase_metric(margin=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, None, "two-sided")
    assert null_lift == pytest.approx(-0.01)
    assert null_abs is None
    assert alt == "greater"


def test_resolve_margins_override_beats_declared_margin():
    m = _increase_metric(margin=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(m, {"revenue": 0.02}, None, "two-sided")
    assert null_lift == pytest.approx(-0.02)
    assert null_abs is None
    assert alt == "greater"


def test_resolve_call_time_margin_at_or_above_one_with_increase_rejected():
    """The same >= 1 domain floor guards the call-time margins= path, since the declaration
    validator alone cannot see an override arriving at call time."""
    m = _increase_metric()
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(m, {"revenue": 1.0}, None, "two-sided")
    assert exc_info.value.code == "definition.metric_margin_implies_unrepresentable_null_lift"


def test_resolve_null_lifts_override_beats_margins_and_declared():
    m = _increase_metric(margin=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(
        m, {"revenue": 0.02}, {"revenue": 0.03}, "two-sided"
    )
    # null_lifts= is the raw escape hatch: no sign flip, no implied tail.
    assert null_lift == 0.03
    assert null_abs is None
    assert alt == "two-sided"


def test_resolve_explicit_alternative_overrides_implied_tail():
    """A margin implies 'greater', but an explicit alternative= always wins, including a
    deliberate harm/futility test in the OTHER direction."""
    m = _increase_metric(margin=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, None, "less")
    assert null_lift == pytest.approx(-0.01)
    assert null_abs is None
    assert alt == "less"


def test_resolve_margins_override_on_non_explicit_direction_raises():
    m = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(m, {"revenue": 0.01}, None, "two-sided")
    assert exc_info.value.code == "definition.metric_margin_requires_direction"


def test_resolve_null_lifts_override_needs_no_direction():
    """null_lifts= is raw and signed: it never consults preferred_direction, so it works
    even on a metric with no explicitly declared polarity."""
    m = MeanMetric(
        name="revenue", entity="user_id", fact="revenue", aggregation="avg_event"
    )  # preferred_direction defaulted
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, {"revenue": 0.02}, "greater")
    assert null_lift == 0.02
    assert null_abs is None
    assert alt == "greater"


# ── Experiment.day_boundary ─────────────────────────────────────────────


def _experiment(**overrides: Any) -> Experiment:
    """Minimal valid Experiment for field-level tests."""
    kwargs: dict[str, Any] = {
        "name": "e",
        "plan": {},
        "unit": "unit_id",
        "exposure": "x",
        "control_group": "control",
        "start": dt.datetime(2025, 1, 1),
    }
    kwargs.update(overrides)
    return Experiment(**kwargs)


def test_day_boundary_defaults_utc_zero_offset():
    exp = _experiment()
    assert exp.day_boundary == "UTC"
    assert exp.day_boundary_offset == dt.timedelta(0)


def test_day_boundary_fixed_offset_parses():
    exp = _experiment(day_boundary="UTC-05:00")
    assert exp.day_boundary_offset == dt.timedelta(hours=-5)


def test_day_boundary_positive_offset_and_quarter_hours():
    assert _experiment(day_boundary="UTC+05:45").day_boundary_offset == dt.timedelta(
        hours=5, minutes=45
    )


def test_day_boundary_rejects_iana_names_with_pointed_message():
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(day_boundary="America/New_York")
    assert exc_info.value.code == "definition.day_boundary_accepts"


def test_day_boundary_rejects_out_of_range_offset():
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(day_boundary="UTC+15:00")
    assert exc_info.value.code == "definition.day_boundary_accepts"


def test_day_boundary_rejects_non_quarter_hour_minutes():
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(day_boundary="UTC-05:10")
    assert exc_info.value.code == "definition.day_boundary_accepts"


def test_definitions_day_boundary_defaults_utc():
    defs = Definitions.model_validate({})
    assert defs.day_boundary == "UTC"


def test_definitions_day_boundary_grammar_enforced():
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate({"day_boundary": "America/New_York"})
    assert exc_info.value.code == "definition.day_boundary_accepts"


# ── margin_abs (absolute-unit non-inferiority guardrails) ──────────────


def test_margin_abs_with_increase_direction_derives_negative_null_and_greater_tail():
    """'don't lose more than 1 point of conversion' -> H0: abs diff <= -0.01."""
    m = ConversionMetric(
        name="conv",
        entity="user_id",
        fact="orders",
        preferred_direction="increase",
        margin_abs=0.01,
    )
    null_abs, tail = m.resolved_null_abs()
    assert null_abs == pytest.approx(-0.01)
    assert tail == "greater"


def test_margin_abs_with_decrease_direction_derives_positive_null_and_less_tail():
    """latency guardrail: 'don't rise more than 5ms' -> H0: abs diff >= +5, less."""
    m = MeanMetric(
        name="latency_ms",
        entity="user_id",
        fact="latency",
        aggregation="avg_event",
        preferred_direction="decrease",
        margin_abs=5.0,
    )
    null_abs, tail = m.resolved_null_abs()
    assert null_abs == pytest.approx(5.0)
    assert tail == "less"


def test_margin_abs_without_explicit_direction_rejected():
    """A defaulted preferred_direction must never silently pick the adverse side."""
    with pytest.raises(DefinitionError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            margin_abs=0.01,
        )
    assert exc_info.value.code == "definition.metric_margin_abs_requires_direction"


def test_margin_abs_with_neutral_direction_rejected():
    with pytest.raises(DefinitionError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="neutral",
            margin_abs=0.01,
        )
    assert exc_info.value.code == "definition.metric_margin_abs_requires_non_neutral_direction"


def test_margin_abs_must_be_positive():
    with pytest.raises(ValidationError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="increase",
            margin_abs=-0.01,
        )
    [error] = exc_info.value.errors()
    assert error["loc"] == ("margin_abs",)
    assert error["type"] == "greater_than"


def test_margin_and_margin_abs_mutually_exclusive():
    with pytest.raises(DefinitionError) as exc_info:
        ConversionMetric(
            name="conv",
            entity="user_id",
            fact="orders",
            preferred_direction="increase",
            margin=0.01,
            margin_abs=0.01,
        )
    assert exc_info.value.code == "definition.metric_base.margin_margin_abs"


def test_margin_abs_below_minus_one_is_legal_for_resolution():
    """Additive scale has no -1 floor: a $5 margin on a $2 metric is expressible."""
    m = MeanMetric(
        name="rev",
        entity="user_id",
        fact="revenue",
        aggregation="avg_event",
        preferred_direction="increase",
        margin_abs=5.0,
    )
    null_abs, _ = m.resolved_null_abs()
    assert null_abs == pytest.approx(-5.0)


def test_resolved_null_abs_without_margin_abs_is_inert():
    m = _increase_metric()
    assert m.resolved_null_abs() == (None, None)


def _increase_metric_abs(margin_abs: float | None = None) -> MeanMetric:
    kwargs: dict = {
        "name": "revenue",
        "entity": "user_id",
        "fact": "revenue",
        "aggregation": "avg_event",
        "preferred_direction": "increase",
    }
    if margin_abs is not None:
        kwargs["margin_abs"] = margin_abs
    return MeanMetric(**kwargs)


def test_resolve_null_and_alternative_returns_abs_element():
    """Declared margin_abs resolves into the middle element; margins_abs=
    override beats the declared value. The relative null stays at 0.0."""
    m = _increase_metric_abs(margin_abs=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, None, "two-sided")
    assert null_lift == 0.0
    assert null_abs == pytest.approx(-0.01)
    assert alt == "greater"

    null_lift, null_abs, alt = resolve_null_and_alternative(
        m, None, None, "two-sided", margins_abs={"revenue": 0.05}
    )
    assert null_lift == 0.0
    assert null_abs == pytest.approx(-0.05)
    assert alt == "greater"


def test_resolve_margins_abs_override_on_plain_metric():
    m = _increase_metric_abs()
    null_lift, null_abs, alt = resolve_null_and_alternative(
        m, None, None, "two-sided", margins_abs={"revenue": 0.02}
    )
    assert null_lift == 0.0
    assert null_abs == pytest.approx(-0.02)
    assert alt == "greater"


def test_resolve_rejects_relative_and_absolute_overrides_for_same_metric():
    m = _increase_metric_abs()
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(
            m, {"revenue": 0.01}, None, "two-sided", margins_abs={"revenue": 0.01}
        )
    assert exc_info.value.code == "definition.metric_both_relative"
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(
            m, None, {"revenue": 0.01}, "two-sided", margins_abs={"revenue": 0.01}
        )
    assert exc_info.value.code == "definition.metric_both_relative"
    # declared relative margin + call-time absolute override is a conflict too
    declared_rel = _increase_metric(margin=0.01)
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(
            declared_rel, None, None, "two-sided", margins_abs={"revenue": 0.01}
        )
    assert exc_info.value.code == "definition.metric_both_relative"
    # ... and a call-time relative override against a declared absolute margin
    declared_abs = _increase_metric_abs(margin_abs=0.01)
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(declared_abs, {"revenue": 0.01}, None, "two-sided")
    assert exc_info.value.code == "definition.metric_both_relative"


def test_resolve_margins_abs_override_on_non_explicit_direction_raises():
    m = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="avg_event")
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(m, None, None, "two-sided", margins_abs={"revenue": 0.01})
    assert exc_info.value.code == "definition.metric_margin_abs_requires_direction"


def test_resolve_explicit_alternative_overrides_abs_implied_tail():
    m = _increase_metric_abs(margin_abs=0.01)
    null_lift, null_abs, alt = resolve_null_and_alternative(m, None, None, "less")
    assert null_lift == 0.0
    assert null_abs == pytest.approx(-0.01)
    assert alt == "less"


# ── rollout_cost (per-metric rollout decision threshold) ───────────────


class TestRolloutCost:
    def test_declared_rollout_cost_accepted(self):
        m = MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="increase",
            rollout_cost=0.02,
        )
        assert m.rollout_cost == 0.02

    def test_rollout_cost_requires_explicit_increase_direction(self):
        # Default (unset) preferred_direction must be refused even though it
        # defaults to "increase": same explicitness discipline as margin.
        with pytest.raises(DefinitionError) as exc_info:
            MeanMetric(
                name="revenue",
                entity="user_id",
                fact="revenue",
                aggregation="avg_event",
                rollout_cost=0.02,
            )
        assert exc_info.value.code == "definition.metric_base.rollout_cost_preferred"

    def test_rollout_cost_refuses_decrease_direction(self):
        with pytest.raises(DefinitionError) as exc_info:
            MeanMetric(
                name="latency",
                entity="user_id",
                fact="latency",
                aggregation="avg_event",
                preferred_direction="decrease",
                rollout_cost=0.02,
            )
        assert exc_info.value.code == "definition.metric_base.rollout_cost_preferred"

    def test_rollout_cost_domain(self):
        # Relative lift must exceed -1 (log1p domain); -0.02 is legal since the direction
        # guard, not the domain, is what enforces sense here.
        with pytest.raises(ValidationError):
            MeanMetric(
                name="revenue",
                entity="user_id",
                fact="revenue",
                aggregation="avg_event",
                preferred_direction="increase",
                rollout_cost=-1.0,
            )

    def test_rollout_cost_accepts_valid_negative_value(self):
        # Confirms the other side of the -1.0 boundary actually constructs: a small measured
        # loss can still be worth rolling out, since the direction guard enforces sense here.
        m = MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="avg_event",
            preferred_direction="increase",
            rollout_cost=-0.02,
        )
        assert m.rollout_cost == pytest.approx(-0.02)


# ── consistency hardening: value-aggregation gate, schedule, aliases, ──
# ── namespace shadowing, day-boundary inheritance ──────────────────────


def _hardening_defs(metrics: list[dict], **top: Any) -> Definitions:
    """Minimal valid Definitions with one occurrence fact + one value fact."""
    payload: dict[str, Any] = {
        "fact_sources": [
            {
                "name": "src",
                "sql": "SELECT 1",
                "timestamp_column": "ts",
                "entities": ["u"],
                "facts": [
                    {"name": "ev", "column": None},
                    {"name": "rev", "column": "amount"},
                ],
            }
        ],
        "exposures": [{"name": "e", "fact": "ev"}],
        "metrics": metrics,
        "experiments": [],
    }
    payload.update(top)
    return Definitions.model_validate(payload)


def test_quantile_value_aggregation_on_occurrence_fact_rejected():
    """QuantileMetric must pass the same value-aggregation gate as MeanMetric: sum-of-nothing
    on an occurrence-only fact must refuse for both."""
    with pytest.raises(DefinitionError) as exc_info:
        _hardening_defs(
            [
                {
                    "type": "quantile",
                    "name": "q",
                    "entity": "u",
                    "fact": "ev",
                    "aggregation": "sum",
                    "quantile": 0.9,
                }
            ]
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_value.metric_uses_aggregation" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_quantile_count_aggregation_on_occurrence_fact_allowed():
    defs = _hardening_defs(
        [{"type": "quantile", "name": "q", "entity": "u", "fact": "ev", "quantile": 0.9}]
    )
    assert len(defs.metrics) == 1


def test_experiment_end_before_start_refused():
    """An inverted enrollment window enrolls no units: refuse at declaration instead of
    silently returning empty results downstream."""
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(start=dt.datetime(2025, 6, 1), end=dt.datetime(2025, 1, 1))
    assert exc_info.value.code == "definition.experiment.end_before_start"


def test_experiment_same_day_end_before_start_time_allowed():
    """`end` compares at DAY granularity (time-of-day deliberately ignored): a one-day
    experiment declared start 10:00 / end midnight the same day is valid, not inverted."""
    exp = _experiment(start=dt.datetime(2025, 1, 15, 10, 0), end=dt.datetime(2025, 1, 15))
    assert exp.end is not None


def test_experiment_mixed_naive_aware_refused_not_typeerror():
    """Aware `end` + naive `observation_end` must be a named refusal, not a raw TypeError
    leaking from inside the validator."""
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
            observation_end=dt.datetime(2025, 3, 1),
        )
    assert exc_info.value.code == "definition.experiment.timezone_aware_but"


def test_experiment_mixed_naive_aware_start_end_refused_without_observation_end():
    """The tz-consistency check is three-way and up-front: a mixed start/end pair with no
    observation_end must also refuse, since the date-level inversion check alone would let
    it pass (.date() strips tzinfo) and downstream duration math would leak a bare TypeError."""
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1),
        )
    assert exc_info.value.code == "definition.experiment.timezone_aware_but"


def test_metric_lookup_uses_declared_names_not_sql_aliases():
    definitions = _hardening_defs(
        [
            {
                "type": "mean",
                "name": name,
                "entity": "u",
                "fact": "rev",
                "aggregation": aggregation,
            }
            for name, aggregation in (
                ("Revenue (US)", "sum"),
                ("Revenue [US]", "max"),
                ("???", "min"),
            )
        ]
    )

    for name, expected in (
        ("Revenue (US)", "sum"),
        ("Revenue [US]", "max"),
        ("???", "min"),
    ):
        metric = definitions.metric(name)
        assert isinstance(metric, MeanMetric)
        assert metric.aggregation == expected


def test_fact_shadowing_a_different_source_name_refused():
    """Cross-source collisions conflict with public source-name lookup precedence."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "purchase",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "conv", "column": None}],
                    },
                    {
                        "name": "other",
                        "sql": "SELECT 2",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "purchase", "column": "amount"}],
                    },
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.index_sources.fact_owned_by" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_fact_named_after_its_own_source_allowed():
    """A fact sharing its OWN source's name is unambiguous (source-first
    resolution returns the same, correct source) and stays allowed."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "purchase",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "purchase", "column": "amount"}],
                }
            ],
        }
    )
    fs = defs.fact_source_for("purchase")
    assert fs is not None and fs.name == "purchase"


def test_day_boundary_inheritance_applies_on_model_validate():
    """Inheritance is a MODEL behavior, not a loader patch: an experiment without its own
    day_boundary inherits the definitions-level default on every construction path, so
    model_validate and load() agree on day bucketing."""
    raw = {
        "day_boundary": "UTC-05:00",
        "fact_sources": [
            {
                "name": "src",
                "sql": "SELECT 1",
                "timestamp_column": "ts",
                "entities": ["u"],
                "facts": [{"name": "ev", "column": None}],
            }
        ],
        "exposures": [{"name": "e", "fact": "ev"}],
        "metrics": [],
        "experiments": [
            {
                "name": "inherits",
                "plan": {},
                "exposure": "e",
                "unit": "u",
                "start": "2025-01-01",
                "control_group": "C",
            },
            {
                "name": "declares_own",
                "plan": {},
                "exposure": "e",
                "unit": "u",
                "start": "2025-01-01",
                "control_group": "C",
                "day_boundary": "UTC+01:00",
            },
        ],
    }
    defs = Definitions.model_validate(raw)
    assert defs.experiments[0].day_boundary == "UTC-05:00"
    assert defs.experiments[1].day_boundary == "UTC+01:00"


# ── LOW-straggler refusals: resolver alternative, field bounds, ────────
# ── exposure XOR, date-dtype values, property-tier shadowing ───────────


def test_resolve_alternative_typo_refused():
    """A typo'd `alternative` must not be treated as an explicit non-default: that would
    silently discard the margin-implied tail and forward the bogus string downstream."""
    m = _increase_metric(margin=0.02)
    with pytest.raises(DefinitionError) as exc_info:
        resolve_null_and_alternative(m, None, None, "two sided")
    assert exc_info.value.code == "definition.metric_unknown_alternative"


def test_resolve_alternative_valid_values_still_pass():
    m = _increase_metric(margin=0.02)
    for alt in ("two-sided", "greater", "less"):
        resolve_null_and_alternative(m, None, None, alt)


def test_window_days_nonpositive_refused():
    for bad in (-7, 0):
        with pytest.raises(ValidationError):
            MeanMetric(name="m", entity="u", fact="f", aggregation="sum", window_days=bad)
    ok = MeanMetric(name="m", entity="u", fact="f", aggregation="sum", window_days=1)
    assert ok.window_days == 1


def test_negative_n_pre_periods_refused():
    """n_pre_periods=-5 must refuse, not construct and silently behave as 0 (a silent CUPED
    disable)."""
    with pytest.raises(ValidationError):
        _experiment(n_pre_periods=-5)
    assert _experiment(n_pre_periods=0).n_pre_periods == 0


def test_exposure_empty_sql_with_fact_refused():
    """sql='' is SET-but-empty (a template/codegen bug) and must refuse
    against a set fact, not silently degrade to the fact path."""
    with pytest.raises(DefinitionError) as exc_info:
        Exposure(name="e", sql="", fact="purchase")
    assert exc_info.value.code == "definition.exposure.needs_exactly_one"


def test_exposure_empty_sql_alone_refused():
    with pytest.raises(DefinitionError) as exc_info:
        Exposure(name="e", sql="   ")
    assert exc_info.value.code == "definition.exposure.sql_set_but"


def _fact_source_with_sql(sql: str) -> FactSource:
    return FactSource(
        name="events",
        sql=sql,
        timestamp_column="ts",
        entities=("u",),
        facts=(Fact(name="f", column=None),),
    )


def _dim_source_with_sql(sql: str) -> DimSource:
    return DimSource(
        name="users",
        sql=sql,
        entity="u",
        properties=(Property(name="country", column="country", dtype="string", as_of="static"),),
    )


_SOURCE_BUILDERS = [
    pytest.param(_fact_source_with_sql, "fact", "events", id="fact"),
    pytest.param(_dim_source_with_sql, "dimension", "users", id="dimension"),
]


@pytest.mark.parametrize("build, kind, name", _SOURCE_BUILDERS)
@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_source_sql_refused(build, kind, name, blank):
    """Blank fact/dimension SQL is a template/codegen bug; it must refuse at construction
    with its own code, not load and fail later at the warehouse."""
    with pytest.raises(DefinitionError) as exc_info:
        build(blank)
    assert exc_info.value.code == "definition.source.sql_empty"
    assert exc_info.value.context["source_kind"] == kind
    assert exc_info.value.context["source_name"] == name


@pytest.mark.parametrize("build, kind, name", _SOURCE_BUILDERS)
@pytest.mark.parametrize(
    "sql",
    ["  SELECT 1 AS x  ", "SELECT ''", "-- events\nSELECT 1 AS x"],
)
def test_nonblank_source_sql_constructs(build, kind, name, sql):
    assert build(sql).sql == sql


@pytest.mark.parametrize("build, kind, name", _SOURCE_BUILDERS)
def test_blank_source_sql_refusal_survives_pickle_and_deepcopy(build, kind, name):
    import copy
    import pickle

    with pytest.raises(DefinitionError) as exc_info:
        build("")
    for clone in (pickle.loads(pickle.dumps(exc_info.value)), copy.deepcopy(exc_info.value)):
        assert clone.code == "definition.source.sql_empty"
        assert clone.context["source_kind"] == kind
        assert clone.context["source_name"] == name


def test_date_dtype_filter_garbage_string_refused():
    """dtype='date' must refuse a garbage string value, not surface it later as a warehouse
    cast error or a silently empty filter."""
    with pytest.raises(DefinitionError) as exc_info:
        _date_filter_defs("not-a-date")
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_dtype.filter_property_incompatible" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_date_dtype_filter_iso_string_accepted_date_object_refused_at_boundary():
    """ISO strings pass the dtype check; a raw date object never reaches it, since
    Filter.values is typed str|int|float|bool, so pydantic refuses it at the boundary
    (quote dates in YAML)."""
    _date_filter_defs("2024-01-01")
    with pytest.raises(ValidationError) as exc_info:
        _date_filter_defs(dt.date(2024, 1, 1))
    errors = exc_info.value.errors()
    assert all("values" in error["loc"] for error in errors)
    assert "string_type" in {error["type"] for error in errors}


def _date_filter_defs(value: Any) -> Definitions:
    return Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "s",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                    "properties": [{"name": "d", "column": "d", "dtype": "date"}],
                }
            ],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "c",
                    "entity": "u",
                    "fact": "f",
                    "filters": [{"property": "d", "op": "equals", "values": [value]}],
                }
            ],
        }
    )


def test_property_shadowing_a_different_source_name_refused():
    """Same misroute class as the fact-tier refusal: fact_source_for resolves property names
    third, so a property named like a DIFFERENT source would resolve to the shadowing source."""
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(
            {
                "fact_sources": [
                    {
                        "name": "purchase",
                        "sql": "SELECT 1",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "conv", "column": None}],
                    },
                    {
                        "name": "other",
                        "sql": "SELECT 2",
                        "timestamp_column": "ts",
                        "entities": ["u"],
                        "facts": [{"name": "f2", "column": None}],
                        "properties": [{"name": "purchase", "column": "p", "dtype": "string"}],
                    },
                ],
            }
        )
    assert exc_info.value.code == "definition.invalid"
    assert "definition.index_sources.property_source_shares" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_property_named_after_its_own_source_allowed():
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT 1",
                    "timestamp_column": "ts",
                    "entities": ["u"],
                    "facts": [{"name": "f", "column": None}],
                    "properties": [{"name": "events", "column": "e", "dtype": "string"}],
                }
            ],
        }
    )
    assert defs.fact_source_for("events") is not None


# ── Experiment-metric bindings: MethodSpec, NormalPriorSpec, ────────────
# ── ExperimentMetric, Experiment.metric_names/guardrail_names/bindings ──


def test_experiment_metric_binding_roundtrip():
    e = Experiment.model_validate(
        {
            "name": "exp",
            "exposure": "e",
            "unit": "u",
            "start": "2026-01-01",
            "control_group": "C",
            "plan": {
                "secondaries": [
                    "revenue",
                    {
                        "metric": "orders",
                        "decision_method": {"name": "unadjusted"},
                        "sensitivity_methods": [{"name": "cuped", "variance_reduction": "cuped"}],
                        "prior": {"mu": 0.0, "sigma": 0.03},
                    },
                ]
            },
            "n_pre_periods": 14,
        }
    )
    assert e.metric_names == ["revenue", "orders"]
    binding = e.bindings["orders"]
    assert binding.decision_method is not None
    assert binding.sensitivity_methods[0].variance_reduction == "cuped"
    assert binding.prior is not None
    assert binding.prior.sigma == 0.03
    assert "revenue" not in e.bindings  # shorthand has no binding


def test_guardrail_binding_included_in_bindings_and_guardrail_names():
    e = Experiment.model_validate(
        {
            "name": "exp",
            "exposure": "e",
            "unit": "u",
            "start": "2026-01-01",
            "control_group": "C",
            "plan": {
                "guardrails": [
                    {"metric": "latency", "prior": {"mu": 0.0, "sigma": 0.01}},
                ]
            },
        }
    )
    assert e.guardrail_names == ["latency"]
    prior = e.bindings["latency"].prior
    assert prior is not None
    assert prior.sigma == 0.01


def test_cuped_binding_requires_pre_periods():
    with pytest.raises(DefinitionError) as exc_info:
        Experiment.model_validate(
            {
                "name": "exp",
                "exposure": "e",
                "unit": "u",
                "start": "2026-01-01",
                "control_group": "C",
                "n_pre_periods": 0,
                "plan": {
                    "secondaries": [
                        {
                            "metric": "orders",
                            "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                        }
                    ]
                },
            }
        )
    assert exc_info.value.code == "definition.experiment.metric_declares_cuped"


def test_cuped_method_name_without_reduction_rejected_in_binding():
    """Mirrors estimate_lift's mislabel refusal at declaration time: a method named 'cuped'
    that doesn't request cuped variance_reduction would label an unadjusted estimate as
    adjusted."""
    with pytest.raises(DefinitionError) as exc_info:
        MethodSpec(name="cuped", variance_reduction="none")
    assert exc_info.value.code == "definition.method.methodspec_name_cuped"


def test_method_spec_conversion_inference_defaults_to_auto_and_round_trips():
    assert MethodSpec(name="unadjusted").conversion_inference == "auto"
    assert MethodSpec.model_validate({"name": "unadjusted"}).conversion_inference == "auto"
    explicit = MethodSpec(name="unadjusted", conversion_inference="finite_sample")
    assert MethodSpec.model_validate(explicit.model_dump(mode="json")) == explicit
    binding = ExperimentMetric.model_validate(
        {
            "metric": "orders",
            "decision_method": {"name": "unadjusted", "conversion_inference": "finite_sample"},
        }
    )
    assert binding.wants_finite_sample
    assert not ExperimentMetric(metric="orders").wants_finite_sample


def test_method_spec_refuses_finite_sample_with_cuped_and_unknown_values():
    with pytest.raises(InvalidRequestError) as exc_info:
        MethodSpec(name="cuped", variance_reduction="cuped", conversion_inference="finite_sample")
    assert exc_info.value.code == "conversion_inference.finite_sample.cuped"
    with pytest.raises(ValidationError):
        MethodSpec(name="unadjusted", conversion_inference="asymptotic")  # ty: ignore[invalid-argument-type]


def _finite_sample_definitions(metric_type: str) -> dict:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [
            {
                "type": metric_type,
                "name": "orders",
                "entity": "user_id",
                "fact": "purchase",
                **(
                    {"threshold_days": [7, 14]}
                    if metric_type == "retention"
                    else {"window_days": 7}
                ),
                **({"aggregation": "sum"} if metric_type == "mean" else {}),
            }
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-14",
                "control_group": "control",
                "plan": {
                    "primary": {
                        "metric": "orders",
                        "decision_method": {
                            "name": "unadjusted",
                            "conversion_inference": "finite_sample",
                        },
                    }
                },
            }
        ],
    }


@pytest.mark.parametrize("metric_type", ["conversion", "retention"])
def test_a_finite_sample_binding_is_accepted_on_a_conversion_or_retention_metric(metric_type):
    definitions = Definitions.model_validate(_finite_sample_definitions(metric_type))
    (experiment,) = definitions.experiments
    assert experiment.bindings["orders"].wants_finite_sample


def test_a_finite_sample_binding_on_a_mean_metric_is_refused_at_definition_load():
    with pytest.raises(InvalidRequestError) as exc_info:
        Definitions.model_validate(_finite_sample_definitions("mean"))
    assert exc_info.value.code == "conversion_inference.finite_sample.metric_type"
    assert exc_info.value.context == {"metric_type": "mean", "metric": "orders"}


@pytest.mark.parametrize(
    "metric",
    [
        {"type": "total", "name": "orders", "fact": "purchase", "aggregation": "sum"},
        {"type": "active", "name": "orders", "entity": "user_id", "fact": "purchase"},
    ],
    ids=["total", "active"],
)
def test_a_finite_sample_binding_on_a_report_only_metric_reports_the_report_only_error(metric):
    """A report-only metric cannot join an experiment at all, so that is what a definition
    consumer sees - not advice to switch the metric's inference to ``auto``."""
    definitions = _finite_sample_definitions("conversion")
    definitions["metrics"] = [metric]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(definitions)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.metric_report_type" in {
        code
        for code, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_a_finite_sample_metric_type_refusal_does_not_mask_other_definition_errors():
    """The shared refusal is the code for an otherwise eligible experiment; once another
    definition error exists, every error is reported together instead of only the first."""
    definitions = _finite_sample_definitions("mean")
    definitions["experiments"][0]["exposure"] = "nope"
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(definitions)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.references_unknown_exposure" in {
        code
        for code, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_one_hazard_has_one_code_and_context_across_every_layer():
    """A ``finite_sample`` request on a metric that is not a conversion or retention rate, and
    on a CUPED-adjusted method, is refused with the same code and typed context whether a
    definition, a frame metric, an estimator method or a power plan carries it."""
    from increment._metric_specs import MetricSpec
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.engine import Method

    metric_code = "conversion_inference.finite_sample.metric_type"
    cuped_code = "conversion_inference.finite_sample.cuped"
    finite = {"conversion_inference": "finite_sample"}

    with pytest.raises(InvalidRequestError) as definition_metric:
        Definitions.model_validate(_finite_sample_definitions("mean"))
    assert definition_metric.value.context == {"metric_type": "mean", "metric": "orders"}
    with pytest.raises(InvalidRequestError) as frame_metric:
        MetricSpec(name="m", decision_method=Method(name="unadjusted", **finite))
    with pytest.raises(InvalidRequestError) as plan_metric:
        ArmPlanningProcedure.standard("mean", **finite)  # ty: ignore[invalid-argument-type]
    metric_errors = (definition_metric.value, frame_metric.value, plan_metric.value)
    assert {error.code for error in metric_errors} == {metric_code}
    assert {error.context["metric_type"] for error in metric_errors} == {"mean"}

    cuped = {"name": "cuped", "variance_reduction": "cuped", **finite}
    with pytest.raises(InvalidRequestError) as definition_cuped:
        MethodSpec(**cuped)  # ty: ignore[invalid-argument-type]
    with pytest.raises(InvalidRequestError) as method_cuped:
        Method(**cuped)  # ty: ignore[invalid-argument-type]
    assert {definition_cuped.value.code, method_cuped.value.code} == {cuped_code}
    assert definition_cuped.value.context == method_cuped.value.context


def test_prior_spec_requires_positive_sigma():
    with pytest.raises(ValidationError):
        NormalPriorSpec(mu=0.0, sigma=0.0)
    with pytest.raises(ValidationError):
        NormalPriorSpec(mu=0.0, sigma=-1.0)


def test_experiment_metric_binding_forbids_extra_fields():
    with pytest.raises(ValidationError, match="extra"):
        ExperimentMetric(metric="orders", bogus_field="x")  # ty: ignore[unknown-argument]  # proving extra=forbid rejects it


def test_experiment_metric_shorthand_and_binding_coexist_in_declaration_order():
    """metric_names/guardrail_names preserve declaration order across a mix of shorthand
    strings and bindings; order is never re-sorted."""
    e = Experiment.model_validate(
        {
            "name": "exp",
            "exposure": "e",
            "unit": "u",
            "start": "2026-01-01",
            "control_group": "C",
            "plan": {"secondaries": ["c", {"metric": "a"}, "b"]},
        }
    )
    assert e.metric_names == ["c", "a", "b"]


def test_module_docstring_examples_still_run():
    """The plan/inference/binding docstrings carry the field documentation a
    notebook `help()` shows, including a worked AnalysisPlan example. The
    suite does not collect doctests, so without this the example would rot
    silently the next time a field or repr changes."""
    import doctest

    import increment.semantics.models as models

    result = doctest.testmod(models, verbose=False, report=False)
    assert result.attempted > 0, "no docstring example ran -- the example was lost"
    assert result.failed == 0, f"{result.failed} docstring example(s) failed"


# ── Frozen models (g5fb) ────────────────────────────────────────────────


def test_experiment_is_frozen():
    """`Experiment` instances are immutable -- in-place field mutation
    (e.g. ``a._experiment.cluster = ...``) bypassed `Analysis`'s own
    sync-on-reassignment path; freezing forces every caller through
    `model_copy`/the copy-on-write accessor instead."""
    e = _experiment()
    with pytest.raises(ValidationError):
        e.cluster = "country"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_metric_subclasses_are_frozen():
    """Every concrete `Metric` union member is frozen, matching `Experiment`."""
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="sum")
    with pytest.raises(ValidationError):
        metric.name = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    ratio = ConversionMetric(name="conv", entity="user_id", fact="purchase")
    with pytest.raises(ValidationError):
        ratio.entity = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_metric_filters_are_immutable():
    """`FactRef.filters` is a tuple, and the `Filter` items inside it are
    themselves frozen -- `frozen=True` on the enclosing metric alone
    blocks `metric.filters = ...` but does nothing about
    `metric.filters.append(...)` or `metric.filters[0].property = ...`
    unless the container and its elements are ALSO immutable."""
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="sum",
        filters=[Filter(property="plan", op="equals", values=["pro"])],
    )
    assert isinstance(metric.filters, tuple)
    with pytest.raises(AttributeError):
        metric.filters.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            Filter(property="plan", op="equals", values=["free"])
        )
    with pytest.raises(ValidationError):
        metric.filters[0].property = "other"  # ty: ignore[invalid-assignment]  # proving items are frozen


def test_ratio_metric_numerator_and_denominator_are_frozen():
    """`RatioMetric.numerator`/`denominator` are bare `Measure`
    instances, not one of the explicitly-frozen `Metric` leaf types --
    `Measure` (via `FactRef`) must be frozen itself, or these stayed
    silently mutable even though the enclosing `RatioMetric` is frozen."""
    ratio = RatioMetric(
        name="rpo",
        entity="user_id",
        numerator=Measure(fact="purchase", aggregation="sum"),
        denominator=Measure(fact="session", aggregation="count"),
    )
    with pytest.raises(ValidationError):
        ratio.numerator.aggregation = "count"  # ty: ignore[invalid-assignment]  # proving frozen at runtime
    with pytest.raises(ValidationError):
        ratio.denominator.fact = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_analysis_plan_role_lists_are_immutable_tuples():
    """Plan role lists are tuples, and a binding entry
    (``ExperimentMetric``) inside them is frozen with immutable method-role
    fields."""
    binding = ExperimentMetric(
        metric="latency", margin=0.01, decision_method=MethodSpec(name="unadjusted")
    )
    plan = AnalysisPlan(primary="revenue", guardrails=[binding])
    assert isinstance(plan.guardrails, tuple)
    with pytest.raises(AttributeError):
        plan.guardrails.append("other")  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
    with pytest.raises(ValidationError):
        binding.decision_method = None  # ty: ignore[invalid-assignment]  # proving frozen at runtime
    with pytest.raises(AttributeError):
        binding.sensitivity_methods.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            MethodSpec(name="cuped", variance_reduction="cuped")
        )


def test_experiment_breakouts_and_factors_are_immutable_tuples():
    """`Experiment.breakouts`/`factors` are tuples of frozen `Breakout`/
    `Factor` models: neither the list nor its items can be mutated to
    silently change what a materialized readout groups or absorbs by."""
    e = _experiment(breakouts=[{"property": "country"}], factors=[{"property": "plan"}])
    assert isinstance(e.breakouts, tuple)
    assert isinstance(e.factors, tuple)
    with pytest.raises(AttributeError):
        e.breakouts.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            Breakout(property="plan")
        )
    with pytest.raises(ValidationError):
        e.breakouts[0].property = "plan"  # ty: ignore[invalid-assignment]  # proving items are frozen
    with pytest.raises(ValidationError):
        e.factors[0].property = "country"  # ty: ignore[invalid-assignment]  # proving items are frozen


def test_nested_mutation_still_refused_after_caches_would_have_warmed():
    """A metric/experiment that has already been resolved once (the
    real-world analogue of "caches are warmed" -- a plan resolved, a
    readout built) must keep refusing nested mutation for the REST of
    its lifetime, not just at construction. Re-attempts the mutation a
    second time to prove refusal is not a one-shot validator quirk."""
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="sum",
        filters=[Filter(property="plan", op="equals", values=["pro"])],
    )
    e = _experiment(breakouts=[{"property": "country"}])
    plan = e.plan

    for _ in range(2):
        with pytest.raises(ValidationError):
            metric.filters[0].values = ("enterprise",)  # ty: ignore[invalid-assignment]
        with pytest.raises(ValidationError):
            e.breakouts[0].property = "plan"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValidationError):
            plan.guardrails = (metric.name,)  # ty: ignore[invalid-assignment]


def test_frozen_models_still_construct_and_serialize_normally():
    """The immutability fix must not disturb ordinary construction,
    dict/JSON round-tripping, or `model_copy(update=...)` -- the
    supported way to get a modified variant of a frozen model.
    ``model_dump()`` (python mode) preserves the tuple container type;
    ``model_dump_json()`` still emits plain JSON arrays."""
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="sum",
        filters=[Filter(property="plan", op="equals", values=["pro"])],
    )
    dumped = metric.model_dump()
    assert dumped["filters"] == ({"property": "plan", "op": "equals", "values": ("pro",)},)
    assert MeanMetric.model_validate(dumped) == metric
    assert MeanMetric.model_validate_json(metric.model_dump_json()) == metric

    e = _experiment(breakouts=[{"property": "country"}])
    moved = e.model_copy(update={"cluster": "store_id"})
    assert moved.cluster == "store_id"
    assert moved.breakouts == e.breakouts
    assert e.cluster is None  # original untouched by the copy


# ── Frozen / immutable models: source-declaration layer (1ahq/r8e9) ────


def test_source_column_family_is_frozen():
    """SourceColumn/Fact/Property/DimValidity were not frozen: scalar
    reassignment silently changed a validated declaration with no
    revalidation of anything that depended on it."""
    fact = Fact(name="purchase", column="amount")
    with pytest.raises(ValidationError):
        fact.name = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    prop = Property(name="country", column="country_code")
    with pytest.raises(ValidationError):
        prop.name = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    validity = DimValidity(changed_at="updated_at")
    with pytest.raises(ValidationError):
        validity.changed_at = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_fact_source_is_frozen_with_immutable_tuple_fields():
    """entities/facts/properties/dims were plain mutable lists: a caller
    could append/clear them after validation, silently changing the row
    grain or schema `_rename_to_builder_cols` resolves against."""
    fs = FactSource(
        name="events",
        sql="SELECT * FROM events",
        timestamp_column="ts",
        entities=["user_id"],
        facts=[Fact(name="page_view", column=None)],
        properties=[Property(name="country", column="country")],
        dims=[],
    )
    assert isinstance(fs.entities, tuple)
    assert isinstance(fs.facts, tuple)
    assert isinstance(fs.properties, tuple)
    assert isinstance(fs.dims, tuple)
    with pytest.raises(ValidationError):
        fs.sql = "SELECT 1"  # ty: ignore[invalid-assignment]  # proving frozen at runtime
    with pytest.raises(AttributeError):
        fs.entities.append("other")  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
    with pytest.raises(AttributeError):
        fs.facts.append(Fact(name="x", column=None))  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
    with pytest.raises(AttributeError):
        fs.properties.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            Property(name="x", column="x")
        )
    with pytest.raises(ValidationError):
        fs.facts[0].name = "other"  # ty: ignore[invalid-assignment]  # proving items are frozen
    with pytest.raises(ValidationError):
        fs.properties[0].name = "other"  # ty: ignore[invalid-assignment]  # proving items are frozen


def test_dim_source_properties_is_immutable_tuple():
    dim = DimSource(
        name="users",
        sql="SELECT user_id, country FROM dim_user",
        entity="user_id",
        properties=[Property(name="country", column="country", as_of="static")],
    )
    assert isinstance(dim.properties, tuple)
    with pytest.raises(AttributeError):
        dim.properties.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            Property(name="x", column="x", as_of="static")
        )
    with pytest.raises(ValidationError):
        dim.properties[0].name = "other"  # ty: ignore[invalid-assignment]  # proving items are frozen
    with pytest.raises(ValidationError):
        dim.entity = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_exposure_filters_is_immutable_tuple():
    exposure = Exposure(
        name="enrolled",
        fact="page_view",
        filters=[Filter(property="plan", op="equals", values=["pro"])],
    )
    assert isinstance(exposure.filters, tuple)
    with pytest.raises(AttributeError):
        exposure.filters.append(  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
            Filter(property="plan", op="equals", values=["free"])
        )
    with pytest.raises(ValidationError):
        exposure.name = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_definitions_top_level_lists_are_immutable_tuples():
    """All five Definitions top-level lists could be appended or cleared
    after validation, with no revalidation of anything cross-checked
    against them."""
    defs = Definitions.model_validate({})
    assert isinstance(defs.fact_sources, tuple)
    assert isinstance(defs.dim_sources, tuple)
    assert isinstance(defs.exposures, tuple)
    assert isinstance(defs.metrics, tuple)
    assert isinstance(defs.experiments, tuple)
    with pytest.raises(AttributeError):
        defs.fact_sources.append("bogus")  # ty: ignore[unresolved-attribute]  # proving the container is a tuple
    with pytest.raises(ValidationError):
        defs.day_boundary = "UTC+01:00"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def test_definitions_properties_of_returns_immutable_tuple():
    """`properties_of` used to return the source's own internal list when
    it had no dims: a caller mutating the return value mutated the
    source's declared properties too."""
    defs = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                }
            ],
        }
    )
    fs = defs.fact_sources[0]
    result = defs.properties_of(fs)
    assert isinstance(result, tuple)
    assert result == fs.properties


def test_nested_caller_owned_instance_is_copied_not_aliased():
    """A caller-owned `Fact` instance passed directly into `FactSource`
    must be revalidated (a fresh copy), never stored by reference --
    `revalidate_instances="always"` closes this, on top of freezing."""
    original_fact = Fact(name="purchase", column="amount")
    fs = FactSource(
        name="events",
        sql="SELECT * FROM events",
        timestamp_column="ts",
        entities=["user_id"],
        facts=[original_fact],
    )
    assert fs.facts[0] is not original_fact
    assert fs.facts[0] == original_fact


# ── Round-trip fidelity: declared-vs-defaulted survives dump/reload ────
# ── (yjc9) ───────────────────────────────────────────────────────────


def test_preferred_direction_dump_excludes_when_not_declared():
    """`model_dump()` used to always write the resolved (possibly
    default) `preferred_direction`; reloading it then marked the field
    as explicitly declared even when it never was."""
    m = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="sum")
    assert m.declared_preferred_direction is None
    dumped = m.model_dump(mode="json")
    assert "preferred_direction" not in dumped
    reloaded = MeanMetric.model_validate(dumped)
    assert reloaded.declared_preferred_direction is None


def test_preferred_direction_dump_includes_when_declared():
    m = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="revenue",
        aggregation="sum",
        preferred_direction="decrease",
    )
    dumped = m.model_dump(mode="json")
    assert dumped["preferred_direction"] == "decrease"
    reloaded = MeanMetric.model_validate(dumped)
    assert reloaded.declared_preferred_direction == "decrease"


# ── Positive infinity / nan rejected in declaration fields (h6d7) ──────


def test_margin_rejects_infinity():
    with pytest.raises(ValidationError):
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="sum",
            preferred_direction="decrease",
            margin=float("inf"),
        )


def test_margin_abs_rejects_infinity():
    with pytest.raises(ValidationError):
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="sum",
            preferred_direction="decrease",
            margin_abs=float("inf"),
        )


def test_rollout_cost_rejects_infinity():
    with pytest.raises(ValidationError):
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="sum",
            preferred_direction="increase",
            rollout_cost=float("inf"),
        )


def test_normal_prior_spec_rejects_infinity_and_nan():
    with pytest.raises(ValidationError):
        NormalPriorSpec(mu=float("inf"), sigma=1.0)
    with pytest.raises(ValidationError):
        NormalPriorSpec(mu=float("nan"), sigma=1.0)
    with pytest.raises(ValidationError):
        NormalPriorSpec(mu=0.0, sigma=float("inf"))


def test_experiment_metric_binding_margin_rejects_infinity():
    with pytest.raises(ValidationError):
        ExperimentMetric(metric="revenue", margin=float("inf"))
    with pytest.raises(ValidationError):
        ExperimentMetric(metric="revenue", margin_abs=float("inf"))


# ── Declaration-time validation gaps (2e7z) ─────────────────────────────


def test_ratio_metric_rejects_identical_numerator_and_denominator():
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Metric).validate_python(
            {
                "type": "ratio",
                "name": "rpo",
                "entity": "user_id",
                "numerator": {"fact": "purchase", "aggregation": "sum"},
                "denominator": {"fact": "purchase", "aggregation": "sum"},
            }
        )
    assert (
        exc_info.value.errors()[0]["ctx"]["error"].code
        == "definition.ratio.metric_numerator_denominator"
    )


def test_method_spec_variance_reduction_rejects_unregistered_value():
    with pytest.raises(ValidationError):
        MethodSpec(name="custom", variance_reduction="bogus")  # ty: ignore[invalid-argument-type]


def test_measure_window_days_rejects_bool():
    with pytest.raises(DefinitionError) as exc_info:
        MeanMetric(
            name="revenue",
            entity="user_id",
            fact="revenue",
            aggregation="sum",
            window_days=True,
        )
    assert exc_info.value.code == "definition.models.reject_bool"


def test_n_pre_periods_rejects_bool():
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(n_pre_periods=True)
    assert exc_info.value.code == "definition.models.reject_bool"


def test_retention_threshold_days_rejects_bool():
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7",
            entity="user_id",
            fact="purchase",
            threshold_days=True,
        )
    assert exc_info.value.code == "definition.models.reject_bool"
    with pytest.raises(DefinitionError) as exc_info:
        RetentionMetric(
            name="d7",
            entity="user_id",
            fact="purchase",
            threshold_days=(True, 7),
        )
    assert exc_info.value.code == "definition.models.reject_bool"


def test_experiment_start_rejects_epoch_int_and_numeric_string():
    """`start=0`/`start='1700000000'` used to silently coerce to
    1970-01-01 / a 2023 timestamp instead of refusing."""
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(start=0)
    assert exc_info.value.code == "definition.expected_iso_8601_epoch"
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(start="1700000000")
    assert exc_info.value.code == "definition.expected_iso_8601_numeric_string"


def test_dim_validity_rejects_identical_range_columns():
    with pytest.raises(DefinitionError) as exc_info:
        DimValidity(valid_from="ts", valid_to="ts")
    assert exc_info.value.code == "definition.dim.validity_valid_from"


def test_filter_between_rejects_reversed_bounds():
    with pytest.raises(DefinitionError) as exc_info:
        Filter(property="p", op="between", values=[20, 10])
    assert exc_info.value.code == "definition.filter.op_between_values"


def test_filter_rejects_nonfinite_values():
    with pytest.raises(DefinitionError) as exc_info:
        Filter(property="p", op="gt", values=[float("inf")])
    assert exc_info.value.code == "definition.filter.values_finite"


def test_day_boundary_rejects_max_offset_with_nonzero_minutes():
    """The validator checked only hours > 14, so a nonexistent
    UTC+14:45 offset (magnitude > 14h) passed; UTC+14:00 remains the
    legitimate maximum (Kiribati)."""
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(day_boundary="UTC+14:45")
    assert exc_info.value.code == "definition.day_boundary_accepts"
    assert _experiment(day_boundary="UTC+14:00").day_boundary_offset == dt.timedelta(hours=14)


# ── Fact-source property name collisions (j62k) ─────────────────────────


def test_fact_source_property_colliding_with_reserved_builder_column_rejected():
    """The builders rename the declared entity and timestamp onto
    ``unit_id``/``ts``, so a property NAMED after one of those collides whether
    or not its own mapping is an identity."""
    for column in ("event_ts", "ts"):
        with pytest.raises(DefinitionError) as exc_info:
            FactSource(
                name="events",
                sql="SELECT * FROM events",
                timestamp_column="ts",
                entities=["user_id"],
                facts=[Fact(name="purchase", column="amount")],
                properties=[Property(name="ts", column=column)],
            )
        assert exc_info.value.code == "definition.fact.source_declares_propert"


def test_fact_source_property_colliding_with_own_fact_value_column_rejected():
    """A property named after this source's own fact value column would
    overwrite that value after the property-to-builder-column rename."""
    with pytest.raises(DefinitionError) as exc_info:
        FactSource(
            name="events",
            sql="SELECT * FROM events",
            timestamp_column="ts",
            entities=["user_id"],
            facts=[Fact(name="purchase", column="amount")],
            properties=[Property(name="amount", column="country")],
        )
    assert exc_info.value.code == "definition.fact.source_renames_propert"


def test_fact_source_property_matching_a_different_facts_logical_name_allowed():
    """Only the PHYSICAL fact value column is reserved; a property may
    share a fact's logical name freely."""
    fs = FactSource(
        name="events",
        sql="SELECT * FROM events",
        timestamp_column="ts",
        entities=["user_id"],
        facts=[Fact(name="purchase", column="amount")],
        properties=[Property(name="purchase", column="purchase_flag")],
    )
    assert fs.properties[0].name == "purchase"


class TestReviewHardening:
    """Boundary cases the declaration validators must get exactly right."""

    def test_day_boundary_inherits_on_direct_construction_too(self):
        # A model-level after-validator returning a replacement instance is not
        # installed on the direct-construction path, so inheritance must mutate.
        from increment.semantics.models import Definitions, Experiment

        exp = Experiment(
            name="e1",
            start="2026-01-01",
            control_group="C",
            plan={"primary": []},
            exposure="signup",
            unit="user",
        )
        assert "day_boundary" not in exp.model_fields_set
        # Construct for real (not model_construct) so the validator runs on the
        # direct-construction path -- the path that silently skipped inheritance.
        inherited = Definitions(
            day_boundary="UTC-05:00",
            experiments=(exp,),
            metrics=(),
            exposures=({"name": "signup", "fact": "signup_evt"},),
            fact_sources=(
                {
                    "name": "s",
                    "sql": "select 1",
                    "timestamp_column": "ts",
                    "entities": ("user",),
                    "facts": ({"name": "signup_evt", "column": None},),
                },
            ),
        )
        assert inherited.experiments[0].day_boundary == "UTC-05:00"
        # Still serialises as unset so a reload re-inherits a changed default.
        assert "day_boundary" not in inherited.experiments[0].model_fields_set

    def test_identity_property_mapping_is_allowed(self):
        from increment.semantics.models import Fact, FactSource, Property

        src = FactSource(
            name="orders",
            sql="select 1",
            timestamp_column="ts",
            entities=("user",),
            facts=(Fact(name="amount", column="amount"),),
            properties=(Property(name="amount", column="amount"),),
        )
        assert src.properties[0].name == "amount"

    def test_property_renaming_a_fact_value_column_away_is_refused(self):
        from increment.semantics.models import Fact, FactSource, Property

        with pytest.raises(DefinitionError) as exc_info:
            FactSource(
                name="orders",
                sql="select 1",
                timestamp_column="ts",
                entities=("user",),
                facts=(Fact(name="amount", column="amount"),),
                properties=(Property(name="country", column="amount"),),
            )
        assert exc_info.value.code == "definition.fact.source_renames_its"

    def test_ratio_rejects_identical_measures_regardless_of_filter_order(self):
        from increment.semantics.models import Filter, Measure, RatioMetric

        f1 = Filter(property="country", op="equals", values=("US",))
        f2 = Filter(property="tier", op="equals", values=("gold",))
        with pytest.raises(DefinitionError) as exc_info:
            RatioMetric(
                name="m",
                entity="user",
                numerator=Measure(fact="orders", aggregation="sum", filters=(f1, f2)),
                denominator=Measure(fact="orders", aggregation="sum", filters=(f2, f1)),
            )
        assert exc_info.value.code == "definition.ratio.metric_numerator_denominator"

    def test_between_accepts_equal_endpoints(self):
        from increment.semantics.models import Filter

        f = Filter(property="n", op="between", values=(5, 5))
        assert f.values == (5, 5)

    def test_between_still_rejects_a_reversed_range(self):
        from increment.semantics.models import Filter

        with pytest.raises(DefinitionError) as exc_info:
            Filter(property="n", op="between", values=(9, 2))
        assert exc_info.value.code == "definition.filter.op_between_values"


@pytest.mark.parametrize("explicit", [False, True])
def test_default_and_explicit_adjustments_are_immutable_snapshots(explicit):
    import pickle
    from collections.abc import MutableMapping
    from copy import deepcopy
    from operator import setitem
    from typing import cast

    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.semantics.sequential import PredeclaredAdjustment

    fields = {"adjustments": {}} if explicit else {}
    spec = AnalysisPlan(inference={"kind": "always_valid", **fields}).inference
    adjustment = PredeclaredAdjustment(coefficient=1, center=0)
    assert spec is not None
    for preserved in (
        spec,
        InferenceSpec.model_validate(spec.model_dump()),
        InferenceSpec.model_validate_json(spec.model_dump_json()),
        deepcopy(spec),
        spec.model_copy(deep=True),
        pickle.loads(pickle.dumps(spec)),
    ):
        with pytest.raises(TypeError):
            setitem(
                cast("MutableMapping[str, PredeclaredAdjustment]", preserved.adjustments),
                "outcome",
                adjustment,
            )
        assert "adjustments" not in preserved.model_dump(mode="json")


def test_adjustment_snapshot_survives_caller_mutation_and_json_replay():
    import pickle
    from collections.abc import MutableMapping
    from copy import deepcopy
    from fractions import Fraction
    from operator import setitem
    from typing import cast

    from increment.semantics.models import InferenceSpec
    from increment.semantics.sequential import PredeclaredAdjustment

    caller = {"outcome": {"coefficient": "1/3", "center": 2}}
    spec = InferenceSpec(kind="asymptotic_mean", adjustments=caller)
    caller["outcome"]["coefficient"] = "9"
    caller["outcome"]["center"] = 99
    caller["other"] = {"coefficient": "1", "center": 0}
    restored = InferenceSpec.model_validate_json(spec.model_dump_json())
    for preserved in (
        spec,
        restored,
        deepcopy(spec),
        spec.model_copy(deep=True),
        pickle.loads(pickle.dumps(spec)),
    ):
        assert set(preserved.adjustments) == {"outcome"}
        assert preserved.adjustments["outcome"].coefficient == Fraction(1, 3)
        assert preserved.adjustments["outcome"].center == 2
        with pytest.raises(TypeError):
            setitem(
                cast("MutableMapping[str, PredeclaredAdjustment]", preserved.adjustments),
                "other",
                PredeclaredAdjustment(coefficient=1, center=0),
            )


def test_inference_spec_repr_shows_mapping_contents():
    from fractions import Fraction

    from increment.semantics.models import InferenceSpec
    from increment.semantics.sequential import PredeclaredAdjustment

    empty = InferenceSpec(kind="asymptotic_mean")
    populated = InferenceSpec(
        kind="asymptotic_mean",
        adjustments={"outcome": PredeclaredAdjustment(coefficient=Fraction(1, 3), center=2)},
        segments={"country": ("US", "GB")},
    )

    assert "segments={}" in repr(empty)
    rendered = repr(populated)
    assert "outcome" in rendered and "coefficient" in rendered
    assert "Fraction(1, 3)" in rendered
    assert "country" in rendered and "US" in rendered and "GB" in rendered
    assert "_FrozenMapping" not in rendered
    assert "object at 0x" not in rendered


def test_invalid_adjustments_remain_refused_after_json_replay():
    import json

    from increment.errors import InvalidRequestError
    from increment.semantics.models import InferenceSpec
    from tests.sequential_cases import registration

    for spec in (
        InferenceSpec(kind="always_valid"),
        InferenceSpec(kind="always_valid", registration=registration()),
    ):
        payload = spec.model_dump(mode="json")
        payload["adjustments"] = {"outcome": {"coefficient": "1", "center": "0"}}
        with pytest.raises(InvalidRequestError) as constructed:
            InferenceSpec.model_validate(payload)
        assert constructed.value.code == "sequential.registration.invalid"
        with pytest.raises(InvalidRequestError) as replayed:
            InferenceSpec.model_validate_json(json.dumps(payload))
        assert replayed.value.code == constructed.value.code


@pytest.mark.parametrize("populated", [False, True])
def test_plan_adjustments_remain_immutable_after_copy_and_pickle(populated):
    import pickle
    from collections.abc import MutableMapping
    from copy import deepcopy
    from operator import setitem
    from typing import cast

    from increment.semantics.models import AnalysisPlan
    from increment.semantics.sequential import PredeclaredAdjustment

    declaration: dict[str, object] = {"kind": "asymptotic_mean" if populated else "always_valid"}
    if populated:
        declaration["adjustments"] = {"outcome": {"coefficient": "1/3", "center": 2}}
    plan = AnalysisPlan(inference=declaration)
    for transported in (
        deepcopy(plan),
        plan.model_copy(deep=True),
        pickle.loads(pickle.dumps(plan)),
    ):
        assert transported.model_dump(mode="json") == plan.model_dump(mode="json")
        assert transported.inference is not None
        with pytest.raises(TypeError):
            setitem(
                cast(
                    "MutableMapping[str, PredeclaredAdjustment]", transported.inference.adjustments
                ),
                "other",
                PredeclaredAdjustment(coefficient=1, center=0),
            )


class TestInferenceSpecBaselineRate:
    def test_baseline_rate_with_asymptotic_mean_refuses(self):
        from fractions import Fraction

        from increment.semantics.models import InferenceSpec

        with pytest.raises(DefinitionError) as raised:
            InferenceSpec(kind="asymptotic_mean", baseline_rate=Fraction(3, 10))
        assert raised.value.code == "definition.inference.baseline_rate_route"

    def test_baseline_rate_with_an_explicit_registration_refuses(self):
        from fractions import Fraction

        from increment.semantics.models import InferenceSpec
        from tests.sequential_cases import registration

        with pytest.raises(DefinitionError) as raised:
            InferenceSpec(
                kind="always_valid", registration=registration(), baseline_rate=Fraction(3, 10)
            )
        assert raised.value.code == "definition.inference.baseline_rate_route"

    @pytest.mark.parametrize("rate", [0, 1, 1.5, -0.1])
    def test_baseline_rate_outside_the_open_unit_interval_refuses(self, rate):
        from increment.semantics.models import InferenceSpec

        with pytest.raises(DefinitionError) as raised:
            InferenceSpec(kind="always_valid", baseline_rate=rate)
        assert raised.value.code == "definition.inference.baseline_rate_domain"

    def test_float_baseline_rate_binds_the_typed_decimal(self):
        from fractions import Fraction

        from increment.semantics.models import InferenceSpec

        spec = InferenceSpec.model_validate({"kind": "always_valid", "baseline_rate": 0.03})
        assert spec.baseline_rate == Fraction(3, 100)


def test_fact_source_property_named_after_a_range_join_column_is_allowed():
    """`valid_from`/`valid_to` are created only by a versioned dim's range join,
    which refuses the collision where it knows the dim is versioned. Requiring
    them unconditionally would reject a harmless inline property."""
    from increment.semantics.models import Fact, FactSource, Property

    for name in ("valid_from", "valid_to"):
        source = FactSource(
            name="orders",
            sql="SELECT * FROM orders",
            timestamp_column="event_ts",
            entities=["user_id"],
            facts=[Fact(name="purchase", column="amount")],
            properties=[Property(name=name, column=name)],
        )
        assert source.properties[0].name == name


def test_fact_source_property_stealing_the_timestamp_is_rejected():
    """The builders always rename `timestamp_column` onto `ts`, so a property
    reading that column loses it -- identity mappings included, since the
    builder's rename takes the column away whether or not the property moves it.

    The ENTITY rename is checked in the query layer instead: only the SELECTED
    unit entity becomes `unit_id`, and which one that is arrives with the query,
    so a source declaring several entities keeps the others available here."""
    from increment.semantics.models import Fact, FactSource, Property

    for name in ("country", "event_at"):
        with pytest.raises(DefinitionError) as exc_info:
            FactSource(
                name="orders",
                sql="SELECT * FROM orders",
                timestamp_column="event_at",
                entities=["user_id"],
                facts=[Fact(name="purchase", column="amount")],
                properties=[Property(name=name, column="event_at")],
            )
        assert exc_info.value.code == "definition.fact.source_points_propert"


def test_fact_source_property_on_a_non_selected_entity_is_allowed():
    """entities=["session_id", "user_id"] analyzed at user_id leaves session_id
    in place, so an identity property on it is valid."""
    from increment.semantics.models import Fact, FactSource, Property

    source = FactSource(
        name="orders",
        sql="SELECT * FROM orders",
        timestamp_column="event_at",
        entities=["session_id", "user_id"],
        facts=[Fact(name="purchase", column="amount")],
        properties=[Property(name="session_id", column="session_id")],
    )
    assert source.properties[0].column == "session_id"


def test_fact_source_property_on_an_unrelated_column_is_still_allowed():
    from increment.semantics.models import Fact, FactSource, Property

    source = FactSource(
        name="orders",
        sql="SELECT * FROM orders",
        timestamp_column="event_at",
        entities=["user_id"],
        facts=[Fact(name="purchase", column="amount")],
        properties=[Property(name="country", column="region")],
    )
    assert source.properties[0].column == "region"


def _two_sources_same_property(order):
    from increment.semantics.models import (
        Experiment,
        Exposure,
        Fact,
        FactSource,
        Property,
    )

    sources = [
        FactSource(
            name="a",
            sql="SELECT * FROM a",
            timestamp_column="ts",
            entities=["user_id"],
            facts=[Fact(name="a_fact", column="v")],
            properties=[Property(name="country", column="country", dtype="string", as_of="static")],
        ),
        FactSource(
            name="b",
            sql="SELECT * FROM b",
            timestamp_column="ts",
            entities=["user_id"],
            facts=[Fact(name="b_fact", column="v")],
            properties=[Property(name="country", column="country", dtype="string", as_of="static")],
        ),
    ]
    if order == "reversed":
        sources = list(reversed(sources))
    exposure = Exposure(name="assignment", sql="SELECT 1")
    experiment = Experiment(
        name="e1",
        unit="user_id",
        control_group="control",
        exposure="assignment",
        start=dt.datetime(2026, 1, 1),
        breakouts=[{"property": "country"}],
        plan={"primary": []},
    )
    return {
        "fact_sources": [s.model_dump() for s in sources],
        "exposures": [exposure.model_dump()],
        "experiments": [experiment.model_dump()],
    }


@pytest.mark.parametrize("order", ["forward", "reversed"])
def test_ambiguous_breakout_source_refuses_both_orders_via_model_validate(order):
    from increment.errors import DefinitionError
    from increment.semantics.models import Definitions

    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(_two_sources_same_property(order))
    assert exc_info.value.code == "definition.breakout.ambiguous_source"


def test_declared_breakout_source_disambiguates():
    from increment.semantics.models import Definitions

    raw = _two_sources_same_property("forward")
    raw["experiments"][0]["breakouts"] = [{"property": "country", "source": "a"}]
    Definitions.model_validate(raw)  # does not raise


def test_observation_end_same_calendar_day_as_end_is_accepted():
    from datetime import datetime

    from increment.semantics.models import AnalysisPlan, Experiment

    e = Experiment(
        name="e1",
        unit="user_id",
        control_group="control",
        exposure="assignment",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 10, 23, 59),
        observation_end=datetime(2026, 1, 10, 0, 0),
        plan=AnalysisPlan(primary=["m"]),
    )
    assert e.observation_end is not None


def test_observation_end_one_day_before_end_still_refuses():
    from datetime import datetime

    from increment.semantics.models import AnalysisPlan, Experiment

    with pytest.raises(DefinitionError) as exc_info:
        Experiment(
            name="e1",
            unit="user_id",
            control_group="control",
            exposure="assignment",
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 10, 23, 59),
            observation_end=datetime(2026, 1, 9, 23, 59),
            plan=AnalysisPlan(primary=["m"]),
        )
    assert exc_info.value.code == ("definition.experiment.observation_end_before_end")


def test_duplicate_fact_source_name_refuses_via_model_validate():
    from increment.errors import DefinitionError
    from increment.semantics.models import Definitions, Fact, FactSource, Property

    raw = {
        "fact_sources": [
            FactSource(
                name="events",
                sql="SELECT * FROM events",
                timestamp_column="ts",
                entities=["user_id"],
                facts=[Fact(name="ev1", column="v")],
                properties=[Property(name="p1", column="p1", dtype="string")],
            ).model_dump(),
            FactSource(
                name="events",
                sql="SELECT * FROM events2",
                timestamp_column="ts",
                entities=["user_id"],
                facts=[Fact(name="ev2", column="v")],
                properties=[Property(name="p2", column="p2", dtype="string")],
            ).model_dump(),
        ],
    }
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(raw)
    assert exc_info.value.code == "definition.duplicates"


# ── Coded refusal unwrap: every Definitions construction path ──────────


@pytest.mark.parametrize(
    "construct",
    [
        lambda raw: Definitions(**raw),
        lambda raw: Definitions.model_validate(raw),
        lambda raw: Definitions.model_validate_json(json.dumps(raw, default=str)),
    ],
    ids=["init", "model_validate", "model_validate_json"],
)
def test_coded_refusal_carries_same_code_on_every_construction_path(construct):
    """`.code` must survive `Definitions(**raw)`, `.model_validate`, and
    `.model_validate_json` alike: the model is the single unwrap point, not
    just one of its construction paths."""
    from increment.errors import DefinitionError

    raw = _two_sources_same_property("forward")
    with pytest.raises(DefinitionError) as exc_info:
        construct(raw)
    assert exc_info.value.code == "definition.breakout.ambiguous_source"


@pytest.mark.parametrize(
    "construct",
    [
        lambda raw: Definitions(**raw),
        lambda raw: Definitions.model_validate(raw),
        lambda raw: Definitions.model_validate_json(json.dumps(raw)),
    ],
    ids=["init", "model_validate", "model_validate_json"],
)
def test_missing_field_validation_error_stays_plain_on_every_direct_entry_path(construct):
    """A nested non-CodedModel field remains pydantic ValidationError."""
    raw = {
        "fact_sources": [
            {
                "name": "events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "ev", "column": "v"}],
            }
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        construct(raw)
    error = exc_info.value.errors()[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("fact_sources", 0, "sql")


# ── Declared designs on Experiment ──────────────────────────────────────


def _base_experiment_kwargs():
    return {
        "name": "exp",
        "exposure": "e",
        "unit": "user_id",
        "start": "2025-01-01",
        "control_group": "control",
        "plan": {"secondaries": ["revenue"]},
    }


def test_experiment_encouragement_declaration_parses_via_resolved_design():
    from increment.semantics.design import Encouragement

    exp = Experiment.model_validate(
        {
            **_base_experiment_kwargs(),
            "design": {
                "mechanism": "encouragement",
                "uptake": {"fact": "clicked", "window_days": 7},
                "one_sided": True,
                "exclusion_restriction": {
                    "acknowledged": True,
                    "justification": "assignment moves revenue only via uptake",
                },
            },
        }
    )
    assert isinstance(exp.design, EncouragementDeclaration)
    design = exp.resolved_design()
    assert isinstance(design, Encouragement)
    assert design.control_group == "control"
    assert design.uptake.fact == "clicked"
    assert design.uptake.window_days == 7
    assert design.one_sided is True


def test_experiment_observational_declaration_parses_via_resolved_design():
    from increment.semantics.design import Observational

    exp = Experiment.model_validate(
        {
            **_base_experiment_kwargs(),
            "design": {
                "mechanism": "observational",
                "covariates": [{"property": "tenure_days", "source": "users"}],
            },
        }
    )
    assert isinstance(exp.design, ObservationalDeclaration)
    assert exp.design.covariates == (AdjustmentCovariate(property="tenure_days", source="users"),)
    design = exp.resolved_design()
    assert isinstance(design, Observational)
    assert design.control_group == "control"
    assert design.adjustment.covariates == ("tenure_days",)


def test_experiment_design_randomized_mechanism_rejected():
    with pytest.raises(ValidationError):
        Experiment.model_validate(
            {
                **_base_experiment_kwargs(),
                "design": {"mechanism": "randomized"},
            }
        )


def test_experiment_without_design_dump_omits_design_key():
    from increment.semantics.design import Randomized

    exp = Experiment.model_validate(_base_experiment_kwargs())
    assert "design" not in exp.model_dump(exclude_none=False)
    assert isinstance(exp.resolved_design(), Randomized)


def test_experiment_encouragement_design_validates_uptake_fact():
    with pytest.raises(DefinitionError) as exc:
        Definitions.model_validate(
            {
                "dialect": "duckdb",
                "fact_sources": [
                    {
                        "name": "events",
                        "sql": "select 1 as user_id, current_timestamp as ts",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [{"name": "exposed", "column": None}],
                    }
                ],
                "exposures": [{"name": "e", "fact": "exposed"}],
                "metrics": [
                    {
                        "type": "mean",
                        "name": "revenue",
                        "entity": "user_id",
                        "fact": "exposed",
                        "aggregation": "count",
                    }
                ],
                "experiments": [
                    {
                        **_base_experiment_kwargs(),
                        "design": {
                            "mechanism": "encouragement",
                            "uptake": {"fact": "clicked_nonexistent"},
                            "exclusion_restriction": {
                                "acknowledged": True,
                                "justification": "assignment moves revenue only via uptake",
                            },
                        },
                    }
                ],
            }
        )
    assert exc.value.code == "definition.invalid"
    assert "definition.check_metric.references_unknown_fact" in {
        c
        for c, _ in exc.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_experiment_encouragement_design_rejects_statistical_tuning_key():
    """`min_first_stage_z` is not a YAML key: `design:` carries only
    mechanism facts, never statistical tuning -- see
    `Experiment.resolved_design()`'s docstring."""
    with pytest.raises(DefinitionError) as exc_info:
        Experiment.model_validate(
            {
                **_base_experiment_kwargs(),
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "clicked"},
                    "min_first_stage_z": 2.0,
                    "exclusion_restriction": {
                        "acknowledged": True,
                        "justification": "assignment moves revenue only via uptake",
                    },
                },
            }
        )
    assert exc_info.value.code == "definition.experiment.design_tuning_key_not_yaml"


# ── window days follow the declared day boundary ────────────────────────


def _windowed(boundary: str, **edges: Any) -> Experiment:
    return _experiment(day_boundary=boundary, **edges)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (dt.datetime(2025, 1, 15), dt.date(2025, 1, 15)),
        (
            dt.datetime(2025, 1, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
            dt.date(2025, 1, 15),
        ),
        (dt.datetime(2025, 1, 15, 5, tzinfo=dt.UTC), dt.date(2025, 1, 15)),
        (dt.datetime(2025, 1, 15, 0, tzinfo=dt.UTC), dt.date(2025, 1, 14)),
    ],
)
def test_window_day_at_a_negative_boundary_is_the_local_day(declared, expected):
    """A naive value is wall-clock at the boundary; an aware value converts to it."""
    exp = _windowed("UTC-05:00", start=declared)
    assert exp.start_day == expected


def test_window_days_report_absent_edges_as_none_and_present_ones_by_local_day():
    from increment.semantics.models import window_days

    exp = _windowed(
        "UTC+05:45",
        start=dt.datetime(2025, 1, 1, 20, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 5, 10, tzinfo=dt.UTC),
    )
    assert window_days(exp) == {
        "start": dt.date(2025, 1, 2),
        "end": dt.date(2025, 1, 5),
        "observation_horizon": dt.date(2025, 1, 5),
    }
    open_ended = _windowed("UTC")
    assert window_days(open_ended) == {
        "start": dt.date(2025, 1, 1),
        "end": None,
        "observation_horizon": None,
    }


def test_two_aware_spellings_of_one_instant_give_one_window():
    zulu = _windowed(
        "UTC-05:00",
        start=dt.datetime(2025, 1, 10, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 20, 0, tzinfo=dt.UTC),
    )
    offset = _windowed(
        "UTC-05:00",
        start=dt.datetime(2025, 1, 9, 19, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
        end=dt.datetime(2025, 1, 19, 19, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
    )
    assert (zulu.start_day, zulu.end_day) == (offset.start_day, offset.end_day)


def test_window_edges_in_one_local_day_are_accepted_at_a_negative_boundary():
    """Both edges are local 2025-01-14; comparing their UTC dates would call this inverted."""
    aware = dt.UTC
    exp = _windowed(
        "UTC-05:00",
        start=dt.datetime(2025, 1, 15, 4, tzinfo=aware),
        end=dt.datetime(2025, 1, 14, 23, tzinfo=aware),
    )
    assert exp.end is not None
    observed = _windowed(
        "UTC-05:00",
        start=dt.datetime(2025, 1, 10, tzinfo=aware),
        end=dt.datetime(2025, 1, 15, 4, tzinfo=aware),
        observation_end=dt.datetime(2025, 1, 14, 23, tzinfo=aware),
    )
    assert observed.observation_end is not None


def test_window_edges_inverted_by_local_day_are_refused_at_a_negative_boundary():
    """Local days 01-15 then 01-14: inverted even though the UTC dates agree."""
    aware = dt.UTC
    with pytest.raises(DefinitionError) as inverted:
        _windowed(
            "UTC-05:00",
            start=dt.datetime(2025, 1, 15, 6, tzinfo=aware),
            end=dt.datetime(2025, 1, 15, 4, tzinfo=aware),
        )
    assert inverted.value.code == "definition.experiment.end_before_start"
    with pytest.raises(DefinitionError) as shortened:
        _windowed(
            "UTC-05:00",
            start=dt.datetime(2025, 1, 10, tzinfo=aware),
            end=dt.datetime(2025, 1, 15, 6, tzinfo=aware),
            observation_end=dt.datetime(2025, 1, 15, 4, tzinfo=aware),
        )
    assert shortened.value.code == "definition.experiment.observation_end_before_end"
