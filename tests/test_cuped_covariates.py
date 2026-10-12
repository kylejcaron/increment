from __future__ import annotations

import copy
import pickle

import pytest
from pydantic import ValidationError

from increment.errors import DefinitionError
from increment.semantics.models import (
    CupedCovariate,
    CupedCovariateBinding,
    ExperimentMetric,
    Measure,
    MethodSpec,
)


def test_semantic_cuped_covariate_types_and_method_are_constructible():
    covariate = CupedCovariate(name="country", kind="categorical")
    binding = CupedCovariateBinding(
        name="pre_revenue", kind="continuous", measure=Measure(fact="purchase", aggregation="sum")
    )

    assert covariate.missing == "refuse"
    assert binding.measure is not None
    assert MethodSpec(name="linear", variance_reduction="cuped_linear").variance_reduction == (
        "cuped_linear"
    )


@pytest.mark.parametrize(
    "aggregation",
    ["sum", "count", "count_distinct", "avg_event", "avg_calendar_day", "min", "max"],
)
def test_measure_covariates_accept_numeric_measure_aggregations(aggregation):
    binding = CupedCovariateBinding(
        name="pre_value",
        kind="continuous",
        measure=Measure.model_validate({"fact": "purchase", "aggregation": aggregation}),
    )

    assert binding.measure is not None
    assert binding.measure.aggregation == aggregation


def test_direct_lift_estimator_refuses_unwired_cuped_linear():
    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method, estimate_lift
    from increment.semantics.models import MeanMetric

    with pytest.raises(UnsupportedRequestError) as raised:
        estimate_lift(
            [MeanMetric(name="revenue", entity="user", fact="revenue")],
            [],
            "control",
            methods=[Method(name="linear", variance_reduction="cuped_linear")],
        )

    assert raised.value.code == "estimation.engine.cuped_linear_unfinished"


def test_definition_covariate_requires_exactly_one_binding():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="x", kind="continuous")

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_property_covariate_cannot_also_declare_measure_or_measure_source():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="x",
            kind="continuous",
            property="tenure",
            measure=Measure(fact="purchase"),
        )

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_measure_source_qualifier_is_rejected():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="x", kind="continuous", source="events", measure=Measure(fact="purchase")
        )

    assert raised.value.code == "definition.experiment.covariate_source_requires_property"


def test_definition_metric_rejects_duplicate_covariate_names():
    x = CupedCovariateBinding(name="x", kind="continuous", property="tenure")
    with pytest.raises(DefinitionError) as raised:
        ExperimentMetric(metric="revenue", covariates=(x, x))

    assert raised.value.code == "definition.experiment.covariate_names_unique"


def test_definition_covariate_refusal_roundtrips_through_copy_and_pickle():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="x", kind="continuous")
    error = raised.value

    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code
        assert restored.context == error.context


def test_unsupported_kind_is_rejected_by_the_declaration_schema():
    with pytest.raises(ValidationError):
        CupedCovariate.model_validate({"name": "x", "kind": "ordinal"})
