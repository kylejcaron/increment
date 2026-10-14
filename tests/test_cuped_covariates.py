from __future__ import annotations

import copy
import pickle

import pytest
from pydantic import ValidationError

from increment.analysis import Analysis
from increment.errors import DefinitionError, InvalidRequestError
from increment.estimation.cuped_linear import fit_cuped_linear
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
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


def test_public_run_routes_mixed_covariate_fit():
    import numpy as np
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(16)],
            "arm": ["c"] * 16 + ["t"] * 16,
            "revenue": [float(i + (arm == "t") * 2) for arm in ("c", "t") for i in range(16)],
            "prior": [float(i % 4) for _arm in ("c", "t") for i in range(16)],
            "region": ["north", "north", "south", "south"] * 8,
        }
    )
    metric = MetricSpec(
        name="revenue",
        covariates=(
            CupedCovariate(name="prior", kind="continuous"),
            CupedCovariate(name="region", kind="categorical"),
        ),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    )

    (row,) = analysis.run()
    assert isinstance(row, LiftEstimate)
    expected = (
        fit_cuped_linear(
            np.asarray(frame["revenue"].to_pylist()),
            frame["arm"].to_pylist(),
            control="c",
            arm_order=("c", "t"),
            covariates={
                "prior": np.asarray(frame["prior"].to_pylist()),
                "region": np.asarray(frame["region"].to_pylist(), dtype=object),
            },
            kinds={"prior": "continuous", "region": "categorical"},
            missing={"prior": "refuse", "region": "refuse"},
        )
        .relative("t")
        .estimate
    )

    assert row.require_lift().value == pytest.approx(np.expm1(expected))


def test_public_run_routes_joint_ratio_cuped_projection():
    import numpy as np
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(16)],
            "arm": ["c"] * 16 + ["t"] * 16,
            "revenue": [float(4 + i + (arm == "t") * 2) for arm in ("c", "t") for i in range(16)],
            "orders": [float(2 + i % 3 + (arm == "t")) for arm in ("c", "t") for i in range(16)],
            "prior": [float(i % 4) for _arm in ("c", "t") for i in range(16)],
            "region": ["north", "north", "south", "south"] * 8,
        }
    )
    metric = MetricSpec(
        name="aov",
        type="ratio",
        numerator="revenue",
        denominator="orders",
        covariates=(
            CupedCovariate(name="prior", kind="continuous"),
            CupedCovariate(name="region", kind="categorical"),
        ),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    )

    result = analysis.run()
    (row,) = result
    assert isinstance(row, LiftEstimate)
    expected = (
        fit_cuped_linear(
            None,
            frame["arm"].to_pylist(),
            control="c",
            arm_order=("c", "t"),
            covariates={
                "prior": np.asarray(frame["prior"].to_pylist()),
                "region": np.asarray(frame["region"].to_pylist(), dtype=object),
            },
            kinds={"prior": "continuous", "region": "categorical"},
            missing={"prior": "refuse", "region": "refuse"},
            numerator=np.asarray(frame["revenue"].to_pylist()),
            denominator=np.asarray(frame["orders"].to_pylist()),
        )
        .ratio("t", scale="log")
        .estimate
    )

    assert row.require_lift().value == pytest.approx(np.expm1(expected))

    reissued = row.reintervalize(0.01)
    assert reissued.require_lift().value == pytest.approx(row.require_lift().value)
    new_lift = reissued.require_lift()
    old_lift = row.require_lift()
    assert new_lift.ub is not None and new_lift.lb is not None
    assert old_lift.ub is not None and old_lift.lb is not None
    assert new_lift.ub - new_lift.lb > old_lift.ub - old_lift.lb
    assert result.metadata is not None
    assert any(
        advisory.code == "estimation.cuped_linear.fit"
        for record in result.metadata.cells
        for advisory in record.advisories
    )

    shifted = frame.set_column(
        frame.schema.get_field_index("prior"),
        "prior",
        pa.array([value + 1.0 for value in frame["prior"].to_pylist()]),
    ).set_column(
        frame.schema.get_field_index("region"),
        "region",
        pa.array(
            ["south" if value == "north" else "north" for value in frame["region"].to_pylist()]
        ),
    )
    shifted_result = Analysis.from_unit_summary(
        shifted, unit="unit", group="arm", control="c", metrics=[metric]
    ).run()
    assert shifted_result.source != result.source
    assert shifted_result[0].source_snapshot_id != row.source_snapshot_id

    with_sibling = frame.append_column("sibling", pa.array([float(i) for i in range(32)]))
    sibling_rows = Analysis.from_unit_summary(
        with_sibling,
        unit="unit",
        group="arm",
        control="c",
        metrics=[metric, MetricSpec(name="sibling")],
    ).run()
    revenue_row = next(item for item in sibling_rows if item.metric == "aov")
    assert isinstance(revenue_row, LiftEstimate)
    assert revenue_row.require_lift().value == pytest.approx(row.require_lift().value)


def test_cuped_linear_level_support_failure_remains_a_cell():
    import pyarrow as pa

    frame = pa.table(
        {
            "unit": [f"{arm}{i}" for arm in ("c", "t") for i in range(8)],
            "arm": ["c"] * 8 + ["t"] * 8,
            "revenue": [float(i + (arm == "t")) for arm in ("c", "t") for i in range(8)],
            "region": ["north"] * 8 + ["north"] * 4 + ["south"] * 4,
        }
    )
    metric = MetricSpec(
        name="revenue",
        covariates=(CupedCovariate(name="region", kind="categorical"),),
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
    )
    result = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control="c", metrics=[metric]
    ).run()

    assert len(result) == 1
    assert isinstance(result[0], LiftEstimate)
    assert result[0].failure_code == "estimation.cuped_linear.level_support"
    assert result[0].lift is None


def test_aggregate_lift_estimator_requires_per_unit_cuped_evidence():
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

    assert raised.value.code == "estimation.engine.cuped_linear_unit_evidence"


def test_definition_covariate_requires_exactly_one_binding():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="pre_x", kind="continuous")

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_property_covariate_cannot_also_declare_measure_or_measure_source():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="pre_x",
            kind="continuous",
            property="tenure",
            measure=Measure(fact="purchase"),
        )

    assert raised.value.code == "definition.experiment.covariate_binding_exactly_one"


def test_measure_source_qualifier_is_rejected():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(
            name="pre_x", kind="continuous", source="events", measure=Measure(fact="purchase")
        )

    assert raised.value.code == "definition.experiment.covariate_source_requires_property"


def test_definition_metric_rejects_duplicate_covariate_names():
    x = CupedCovariateBinding(name="pre_x", kind="continuous", property="tenure")
    with pytest.raises(DefinitionError) as raised:
        ExperimentMetric(metric="revenue", covariates=(x, x))

    assert raised.value.code == "definition.experiment.covariate_names_unique"


def test_definition_covariate_refusal_roundtrips_through_copy_and_pickle():
    with pytest.raises(DefinitionError) as raised:
        CupedCovariateBinding(name="pre_x", kind="continuous")
    error = raised.value

    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code
        assert restored.context == error.context


def test_unsupported_kind_is_rejected_by_the_declaration_schema():
    with pytest.raises(ValidationError):
        CupedCovariate.model_validate({"name": "x", "kind": "ordinal"})


def test_scalar_and_set_cuped_reject_a_nonmember_scalar_covariate():
    with pytest.raises(InvalidRequestError) as raised:
        MetricSpec(
            name="revenue",
            covariate="prior",
            covariates=(CupedCovariate(name="country", kind="categorical"),),
        )
    assert raised.value.code == "frame.metric.scalar_and_set_covariates"


def test_scalar_cuped_can_share_a_declared_covariate_with_cuped_linear():
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "unit_id": f"{arm[0]}{index}",
                "group_id": arm,
                "revenue": 10.0 + index + 1.5 * (arm == "treatment") + 0.4 * index,
                "pre_revenue": float(index),
            }
            for arm in ("control", "treatment")
            for index in range(60)
        ]
    )
    metric = MetricSpec(
        name="revenue",
        covariate="pre_revenue",
        covariates=(CupedCovariate(name="pre_revenue", kind="continuous"),),
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        sensitivity_methods=(Method(name="linear", variance_reduction="cuped_linear"),),
    )
    analysis = Analysis.from_unit_summary(
        rows,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics=(metric,),
    )
    try:
        result = analysis.run()
        estimates = [row for row in result if isinstance(row, LiftEstimate)]
        assert {row.method for row in estimates} == {"cuped", "linear"}
        assert all(row.lift is not None and row.sampling_available for row in estimates)
    finally:
        analysis.close()
