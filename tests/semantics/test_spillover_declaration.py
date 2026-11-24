"""The two-stage assignment law belongs to an ordinary named experiment."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from increment.errors import CodedError
from increment.semantics.loader import load
from increment.semantics.models import Definitions


def _catalog() -> dict[str, Any]:
    experiment = {
        "name": "store_offers",
        "exposure": "coupon_assignment",
        "unit": "customer_id",
        "start": "2026-01-01",
        "end": "2026-01-31",
        "observation_end": "2026-02-07",
        "control_group": "no_offer",
        "plan": {"primary": "purchases_7d", "alpha": 0.025},
    }
    return {
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["customer_id", "account_id"],
                "facts": [{"name": "purchase"}, {"name": "assigned"}],
                "properties": [{"name": "channel", "column": "channel"}],
            }
        ],
        "exposures": [{"name": "coupon_assignment", "fact": "assigned"}],
        "metrics": [
            {
                "name": "purchases_7d",
                "type": "conversion",
                "entity": "customer_id",
                "fact": "purchase",
                "window_days": 7,
                "filters": [{"property": "channel", "op": "equals", "values": ["shop"]}],
            }
        ],
        "experiments": [
            {**experiment, "name": "ordinary"},
            {
                **experiment,
                "design": {
                    "mechanism": "spillover",
                    "offered_group": "coupon",
                    "groups_per_policy": {"west": {"high": 3, "low": 2, "mid": 1}},
                    "comparisons": [
                        {"policy": "high", "reference": "mid", "meaningful_effect": -0.1},
                        {"policy": "mid", "reference": "low", "meaningful_effect": 0.2},
                    ],
                    "roster": {"catalog": "project-id", "schema": "trial", "name": "eligible"},
                    "offer_counts": "trial.quotas",
                    "group_policy": "store_draws",
                },
            },
        ],
    }


def test_named_catalog_binds_directed_policy_law_to_selected_conversion(tmp_path: Path):
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(_catalog()))
    definitions = load(path)
    ordinary = definitions.experiment("ordinary")
    assert ordinary is not None and ordinary.resolved_design().mechanism == "randomized"
    experiment = definitions.experiment("store_offers")
    assert experiment is not None and experiment.design is not None
    assert experiment.design.mechanism == "spillover"
    assert experiment.design.table_names() == {
        "roster": "project-id.trial.eligible",
        "offer_counts": "trial.quotas",
        "group_policy": "store_draws",
    }

    (name,) = experiment.metric_names
    metric = definitions.metric(name)
    assert metric is not None and metric.type == "conversion"
    assert (metric.name, metric.fact, metric.filters[0].values) == (
        "purchases_7d",
        "purchase",
        ("shop",),
    )
    assert metric.window_days is not None
    parameters = experiment.design.parameters(experiment.control_group)
    assert (parameters.control_group, parameters.offered_group) == ("no_offer", "coupon")
    law = parameters.law(metric.name, metric.window_days, experiment.plan.alpha)
    assert (law.outcome, law.horizon, law.alpha) == ("purchases_7d", "7 days", 0.025)
    assert law.groups_per_policy == {"west": {"low": 2, "mid": 1, "high": 3}}
    assert [(c.policy, c.reference) for c in law.comparisons] == [("high", "mid"), ("mid", "low")]
    assert Definitions.model_validate_json(definitions.model_dump_json()) == definitions
    with pytest.raises(CodedError) as refused:
        experiment.resolved_design()
    assert refused.value.code == "spillover.design.invalid"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("second_primary", "spillover.design.invalid"),
        ("variable_window", "spillover.design.invalid"),
        ("ratio", "spillover.design.invalid"),
        ("different_entity", "definition.invalid"),
        ("missing_fact", "definition.invalid"),
        ("missing_filter_property", "definition.invalid"),
        ("same_assignment_label", "spillover.design.invalid"),
        ("boolean_change", "spillover.planning.meaningful_effect"),
        ("decision_method", "spillover.design.invalid"),
        ("secondary_budget", "spillover.design.invalid"),
        ("cluster", "spillover.design.invalid"),
    ],
)
def test_unsupported_declaration_refuses_before_data(change, expected):
    raw = _catalog()
    experiment = raw["experiments"][1]
    raw["experiments"] = [experiment]
    metric = raw["metrics"][0]
    if change == "second_primary":
        raw["metrics"].append({**metric, "name": "other"})
        experiment["plan"] = {"primary": ["purchases_7d", "other"]}
    elif change == "variable_window":
        metric["window_days"] = None
    elif change == "ratio":
        raw["metrics"][0] = {
            "name": metric["name"],
            "type": "ratio",
            "entity": "customer_id",
            "numerator": {"fact": "purchase"},
            "denominator": {"fact": "assigned"},
        }
    elif change == "different_entity":
        metric["entity"] = "account_id"
    elif change == "missing_fact":
        metric["fact"] = "typo"
    elif change == "missing_filter_property":
        metric["filters"][0]["property"] = "typo"
    elif change == "same_assignment_label":
        experiment["design"]["offered_group"] = experiment["control_group"]
    elif change == "boolean_change":
        experiment["design"]["comparisons"][0]["meaningful_effect"] = True
    elif change == "decision_method":
        experiment["plan"] = {
            "primary": {"metric": metric["name"], "decision_method": {"name": "unadjusted"}}
        }
    elif change == "secondary_budget":
        experiment["plan"] = {**experiment["plan"], "q": 0.1}
    else:
        experiment["cluster"] = "store"
    with pytest.raises(CodedError) as refused:
        Definitions.model_validate(raw)
    assert refused.value.code == expected


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("duplicate", "definition.duplicates"),
        ("extra_table", "definition.validation"),
        ("unsafe_table", "spillover.design.invalid"),
        ("retired_section", "definition.validation"),
    ],
)
def test_catalog_rejects_duplicate_experiments_and_undeclared_inputs(tmp_path, change, expected):
    raw = _catalog()
    experiment = raw["experiments"][1]
    if change == "duplicate":
        experiment["name"] = "ordinary"
    elif change == "extra_table":
        experiment["design"]["outcomes"] = "purchases"
    elif change == "unsafe_table":
        experiment["design"]["group_policy"] = "SELECT * FROM group_draws"
    else:
        raw["spillovers"] = []
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(CodedError) as refused:
        load(path)
    assert refused.value.code == expected
