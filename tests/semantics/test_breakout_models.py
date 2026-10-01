"""Tests for `Breakout` — source resolution, dtype compatibility, duplicate
detection, and cross-source coverage against experiment metrics."""

import pytest

from increment.errors import DefinitionError
from increment.semantics.models import Definitions

# ── Test fixtures ────────────────────────────────────────────────────


def _base_defs(**overrides):
    """Minimal valid definitions: one fact source ('events', string 'country' property), one
    exposure, one conversion metric ('visit_rate'), one experiment; callers override as needed."""
    defs = {
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
                        "dtype": "string",
                        "as_of": "static",
                    }
                ],
            }
        ],
        "exposures": [{"name": "e", "fact": "page_view"}],
        "metrics": [
            {"type": "conversion", "name": "visit_rate", "entity": "user_id", "fact": "page_view"}
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "e",
                "unit": "user_id",
                "start": "2024-01-01",
                "control_group": "C",
                "plan": {"secondaries": ["visit_rate"]},
                "breakouts": [],
            }
        ],
    }
    defs.update(overrides)
    return defs


# ── Resolution: explicit and inferred source ────────────────────────


def test_breakout_explicit_source_resolves():
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    defs = Definitions.model_validate(d)
    b = defs.experiments[0].breakouts[0]
    assert b.property == "country"
    assert b.source == "events"


def test_breakout_inferred_source_resolves():
    """Omitting `source` infers the first fact source that carries the
    property and has the experiment's unit as an entity."""
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [{"property": "country"}]
    defs = Definitions.model_validate(d)
    b = defs.experiments[0].breakouts[0]
    assert b.property == "country"
    assert b.source is None  # not resolved onto the model, just validated


def test_breakout_inferred_source_unresolvable_rejected():
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [{"property": "no_such_property"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.resolve_breakout.experiment_property_could" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


# ── dtype compatibility ─────────────────────────────────────────────


def test_breakout_float_dtype_rejected():
    d = _base_defs()
    d["fact_sources"][0]["properties"].append(
        {"name": "spend_bucket", "column": "spend", "dtype": "float"}
    )
    d["experiments"][0]["breakouts"] = [{"property": "spend_bucket", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_breakouts.experiment_breakout_property",
        "definition.check_breakouts.experiment_breakout_property_unsupported_dtype",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


def test_breakout_date_dtype_rejected():
    d = _base_defs()
    d["fact_sources"][0]["properties"].append(
        {"name": "signup_date", "column": "signup", "dtype": "date"}
    )
    d["experiments"][0]["breakouts"] = [{"property": "signup_date", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert {
        "definition.check_breakouts.experiment_breakout_property",
        "definition.check_breakouts.experiment_breakout_property_unsupported_dtype",
    } <= {c for c, _ in exc_info.value.context["errors"]}  # ty: ignore[not-iterable]


# ── Explicit source validation ──────────────────────────────────────


def test_breakout_nonexistent_explicit_source_rejected():
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "nope"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.resolve_breakout.experiment_property_references" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_breakout_explicit_source_missing_property_rejected():
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [{"property": "not_a_prop", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.resolve_breakout.experiment_property_found" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


# ── Cross-source coverage against experiment metrics ────────────────


def _cross_source_defs():
    """Two fact sources: 'events' (has 'country'), 'orders' (does not)."""
    d = _base_defs()
    d["fact_sources"].append(
        {
            "name": "orders",
            "sql": "SELECT * FROM orders",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "order", "column": "order_id"}],
            "properties": [],
        }
    )
    d["metrics"].append(
        {"type": "conversion", "name": "checkout", "entity": "user_id", "fact": "order"}
    )
    d["experiments"][0]["plan"] = {"secondaries": ["visit_rate", "checkout"]}
    return d


def test_breakout_metric_missing_property_rejected():
    d = _cross_source_defs()
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_breakouts.metric_source_does" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_breakout_metric_missing_property_skip_missing_allowed():
    d = _cross_source_defs()
    d["experiments"][0]["breakouts"] = [
        {"property": "country", "source": "events", "skip_missing": True}
    ]
    defs = Definitions.model_validate(d)
    assert defs.experiments[0].breakouts[0].skip_missing is True


def test_breakout_guardrail_missing_property_rejected():
    """Cross-source coverage check applies to guardrails too, not just metrics: a guardrail's
    fact source missing the breakout property is the same misconfiguration as a metric's."""
    d = _cross_source_defs()
    d["experiments"][0]["plan"] = {"secondaries": ["visit_rate"], "guardrails": ["checkout"]}
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_breakouts.metric_source_does" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


# ── Duplicate (property, source) detection ──────────────────────────


def test_breakout_duplicate_property_source_rejected():
    d = _base_defs()
    d["experiments"][0]["breakouts"] = [
        {"property": "country", "source": "events"},
        {"property": "country", "source": "events"},
    ]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_breakouts.experiment_duplicate_breakout" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_breakout_same_property_different_sources_allowed():
    """Same property name from two different sources is fine — it resolves
    to two distinct columns."""
    d = _base_defs()
    d["fact_sources"].append(
        {
            "name": "orders",
            "sql": "SELECT * FROM orders",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "order", "column": "order_id"}],
            "properties": [
                {
                    "name": "country",
                    "column": "country_code",
                    "dtype": "string",
                    "as_of": "static",
                }
            ],
        }
    )
    d["metrics"].append(
        {"type": "conversion", "name": "checkout", "entity": "user_id", "fact": "order"}
    )
    d["experiments"][0]["plan"] = {"secondaries": ["visit_rate", "checkout"]}
    d["experiments"][0]["breakouts"] = [
        {"property": "country", "source": "events"},
        {"property": "country", "source": "orders"},
    ]
    defs = Definitions.model_validate(d)
    assert len(defs.experiments[0].breakouts) == 2


# ── CUPED + ratio guard (binding-driven) is unaffected by breakouts ──


def test_breakout_ratio_metric_shorthand_with_n_pre_periods_loads():
    """A ratio metric without a CUPED-requesting binding coexists with n_pre_periods>0 even with
    breakouts present; only a binding that actually requests CUPED is refused (see below)."""
    d = _base_defs()
    d["fact_sources"][0]["facts"].append({"name": "revenue", "column": "amount"})
    d["metrics"].append(
        {
            "type": "ratio",
            "name": "rev_per_visit",
            "entity": "user_id",
            "numerator": {"fact": "revenue", "aggregation": "sum"},
            "denominator": {"fact": "page_view", "aggregation": "count"},
        }
    )
    d["experiments"][0]["plan"] = {"secondaries": ["visit_rate", "rev_per_visit"]}
    d["experiments"][0]["n_pre_periods"] = 7
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    defs = Definitions.model_validate(d)
    assert defs.experiments[0].metric_names == ["visit_rate", "rev_per_visit"]


def test_breakout_ratio_metric_cuped_binding_loads():
    """A CUPED-requesting binding on a ratio metric loads alongside breakouts:
    the warehouse covariate is the numerator's pre-period total."""
    d = _base_defs()
    d["fact_sources"][0]["facts"].append({"name": "revenue", "column": "amount"})
    d["metrics"].append(
        {
            "type": "ratio",
            "name": "rev_per_visit",
            "entity": "user_id",
            "numerator": {"fact": "revenue", "aggregation": "sum"},
            "denominator": {"fact": "page_view", "aggregation": "count"},
        }
    )
    d["experiments"][0]["plan"] = {
        "secondaries": [
            "visit_rate",
            {
                "metric": "rev_per_visit",
                "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
            },
        ]
    }
    d["experiments"][0]["n_pre_periods"] = 7
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    defs = Definitions.model_validate(d)
    experiment = defs.experiment(d["experiments"][0]["name"])
    assert experiment is not None
    assert experiment.bindings["rev_per_visit"].wants_cuped


# ── Property.as_of: event_time cannot back a breakout ────────────────


def test_breakout_event_time_property_rejected():
    """[speg] as_of='event_time' properties are rejected as breakout dimensions: a post-exposure
    value would bias the segment. pre_exposure/static are accepted."""
    d = _base_defs()
    d["fact_sources"][0]["properties"] = [
        {"name": "country", "column": "country_code", "dtype": "string", "as_of": "event_time"}
    ]
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_breakouts.experiment_breakout_property" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }

    for as_of in ("pre_exposure", "static"):
        d["fact_sources"][0]["properties"] = [
            {
                "name": "country",
                "column": "country_code",
                "dtype": "string",
                "as_of": as_of,
            }
        ]
        defs = Definitions.model_validate(d)
        assert defs.experiments[0].breakouts[0].property == "country"


def test_event_time_property_default_rejected_for_breakout():
    """[speg] as_of defaults to event_time, so an unannotated property is rejected by design."""
    d = _base_defs()
    # Drop the static as_of annotation so the property falls back to the event_time default.
    d["fact_sources"][0]["properties"] = [
        {"name": "country", "column": "country_code", "dtype": "string"}
    ]
    d["experiments"][0]["breakouts"] = [{"property": "country", "source": "events"}]
    with pytest.raises(DefinitionError) as exc_info:
        Definitions.model_validate(d)
    assert exc_info.value.code == "definition.invalid"
    assert "definition.check_breakouts.experiment_breakout_property" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }
