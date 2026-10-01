from typing import Any, cast

import pytest

from increment.decision import ContrastDecisionProcedure
from increment.estimation.contrast import ContrastStats, estimate_contrast
from increment.estimation.contrast_results import ContrastResult, ContrastResults
from increment.semantics.unit_cycle import UnitCycleTApproximation


def _result():
    stats = ContrastStats(
        metric="orders",
        aggregation="any",
        probability_ct=0.8,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        carryover_order=0,
        observation_steps=3,
        retained_steps=3,
        control_group="control",
        treatment_group="treatment",
        n_units=3,
        n_cycles=9,
        ct_cycles=7,
        tc_cycles=2,
        reference_delta=0.2,
        mean_residual=0.1,
        m2_delta=0.03,
    )
    procedure = ContrastDecisionProcedure(
        reference=UnitCycleTApproximation(),
        metric="orders",
        role="secondary",
        alternative="greater",
        null_abs=0.0,
        alpha=0.025,
    )
    return estimate_contrast(stats, procedure).results[0]


def test_result_models_are_typed_and_immutable():
    result = _result()
    results = ContrastResults([result])

    assert isinstance(result, ContrastResult)
    assert isinstance(results, list)
    assert results[0] is result
    with pytest.raises((TypeError, ValueError)):
        result.metric = "other"  # type: ignore[misc]


def test_slice_preserves_contrast_results():
    result = _result()

    sliced = ContrastResults([result, result])[1:]

    assert isinstance(sliced, ContrastResults)
    assert sliced == [result]
    assert hasattr(sliced, "to_frame")


def test_concatenation_preserves_contrast_results():
    result = _result()

    combined = ContrastResults([result]) + [result]

    assert isinstance(combined, ContrastResults)
    assert combined == [result, result]
    assert hasattr(combined, "to_frame")


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_to_frame_supports_declared_backends(backend):
    pytest.importorskip(backend)
    frame = cast(Any, ContrastResults([_result()]).to_frame(backend=backend))
    columns = list(frame.columns) if backend != "pyarrow" else frame.column_names
    assert columns == [
        "metric",
        "control_group",
        "treatment_group",
        "method",
        "method_role",
        "role",
        "estimand",
        "aggregation",
        "probability_ct",
        "randomization_law",
        "independence_grain",
        "carryover_order",
        "observation_steps",
        "retained_steps",
        "identifying_assumption",
        "estimate",
        "lb",
        "ub",
        "standard_error",
        "alternative",
        "preferred_direction",
        "null_abs",
        "alpha",
        "n_units",
        "n_cycles",
        "n_blocks",
        "ct_cycles",
        "tc_cycles",
        "dof",
        "assignment",
        "inference",
        "reference",
        "open_side",
        "dof_unavailable_reason",
        "standard_error_unavailable_reason",
        "mean_slope",
        "minimum_cycles_per_unit",
        "maximum_cycles_per_unit",
        "washout_steps",
        "effective_alpha",
        "refusal_probability_upper",
        "residual_cutoff",
        "residual_p_value",
        "reference_spec",
        "provenance",
        "response_meaning",
    ]
    if backend == "pyarrow":
        row = frame.to_pylist()[0]
        assert row["estimate"] == pytest.approx(_result().estimate.value)
    else:
        assert frame["estimate"][0] == pytest.approx(_result().estimate.value)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_empty_to_frame_retains_typed_schema(backend):
    pytest.importorskip(backend)
    frame = cast(Any, ContrastResults().to_frame(backend=backend))
    columns = list(frame.columns) if backend != "pyarrow" else frame.column_names
    assert columns[0] == "metric"
    assert columns[-1] == "response_meaning"
    assert len(frame) == 0


@pytest.mark.parametrize("carryover_order", [0, 1, 2])
@pytest.mark.parametrize(
    ("randomization_law", "independence_grain"),
    [("independent_bernoulli_order", "unit_cycle"), ("shared_schedule", "shared_block")],
)
def test_result_json_round_trips_every_declared_law_and_order(
    randomization_law, independence_grain, carryover_order
):
    """Wire provenance distinguishes roster size from independent block count."""
    from increment.estimation.results import Estimate

    shared = randomization_law == "shared_schedule"
    result = ContrastResult(
        method="switchback_block_t" if shared else "switchback_unit_t_approximation",
        reference_spec=None if shared else UnitCycleTApproximation(),
        reference="block_t" if shared else "unit_t_approximation",
        metric="orders",
        control_group="control",
        treatment_group="treatment",
        estimand="retained_window_total_difference",
        aggregation="sum",
        probability_ct=0.6,
        randomization_law=randomization_law,
        independence_grain=independence_grain,
        carryover_order=carryover_order,
        observation_steps=3,
        retained_steps=3 - carryover_order,
        estimate=Estimate(value=1.0, lb=0.5, ub=1.5, level=0.95),
        standard_error=0.25,
        alternative="two-sided",
        null_abs=0.0,
        alpha=0.05,
        n_units=100 if shared else 3,
        n_cycles=300 if shared else 6,
        n_blocks=3 if shared else None,
        ct_cycles=2 if shared else 4,
        tc_cycles=1 if shared else 2,
        dof=2.0,
    )
    restored = ContrastResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.randomization_law == randomization_law
    assert restored.independence_grain == independence_grain
    assert restored.carryover_order == carryover_order
    assert restored.observation_steps == 3
    assert restored.retained_steps == 3 - carryover_order


def _shared_result(**overrides):
    values = _result().model_dump()
    values.update(
        randomization_law="shared_schedule",
        independence_grain="shared_block",
        method="switchback_block_t",
        reference="block_t",
        reference_spec=None,
        n_units=100,
        n_cycles=300,
        n_blocks=3,
        ct_cycles=2,
        tc_cycles=1,
        dof=2.0,
    )
    values.update(overrides)
    return ContrastResult.model_validate(values)


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"n_blocks": None}, "estimation.contrast.replication_counts"),
        ({"n_cycles": 299}, "estimation.contrast.replication_counts"),
        ({"dof": 99.0}, "estimation.contrast.reference_metadata"),
        ({"reference": "unit_t_approximation"}, "estimation.contrast.reference_metadata"),
        ({"method": "switchback_unit_t_approximation"}, "estimation.contrast.reference_metadata"),
        ({"independence_grain": "unit_cycle"}, "estimation.contrast.assignment_metadata"),
    ],
)
def test_shared_result_rejects_false_replication_provenance(changes, code):
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        _shared_result(**changes)
    assert exc_info.value.code == code


def test_shared_reference_allows_one_unit_with_multiple_blocks():
    result = _shared_result(n_units=1, n_cycles=3)
    assert result.n_blocks == 3
    assert result.dof == 2.0


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("carryover_order", [0, 1, 2])
def test_frames_preserve_absent_and_present_independent_block_counts(backend, carryover_order):
    import narwhals as nw

    pytest.importorskip(backend)
    results = ContrastResults(
        [
            _result(),
            _shared_result(carryover_order=carryover_order, retained_steps=3 - carryover_order),
        ]
    )
    frame = nw.from_native(results.to_frame(backend=backend), eager_only=True)
    assert frame["n_blocks"].dtype == nw.Int64
    assert frame["n_blocks"].is_null().to_list() == [True, False]
    assert frame["n_blocks"].drop_nulls().to_list() == [3]
    assert frame["n_units"].to_list() == [3, 100]
    assert frame["n_cycles"].to_list() == [9, 300]
    assert frame["ct_cycles"].dtype == nw.Int64
    assert frame["tc_cycles"].dtype == nw.Int64
    assert frame["ct_cycles"].to_list() == [7, 2]
    assert frame["tc_cycles"].to_list() == [2, 1]
    assert frame["observation_steps"].dtype == nw.Int64
    assert frame["retained_steps"].dtype == nw.Int64
    assert frame["observation_steps"].to_list() == [3, 3]
    assert frame["retained_steps"].to_list() == [3, 3 - carryover_order]


@pytest.mark.parametrize("order", [True, 1.0])
def test_result_carryover_order_is_not_coerced(order):
    from increment.errors import InvalidRequestError

    values = _result().model_dump()
    values["carryover_order"] = order
    with pytest.raises(InvalidRequestError) as raised:
        ContrastResult.model_validate(values)
    assert raised.value.code == "model.field.type"


@pytest.mark.parametrize("entry", ["constructor", "model_validate", "model_validate_json"])
@pytest.mark.parametrize(
    "changes",
    [
        {"retained_steps": 2},
        {"observation_steps": 2},
        {"carryover_order": 3},
    ],
)
def test_result_rejects_inconsistent_retained_window_coded(entry, changes):
    import json

    from increment.errors import InvalidRequestError

    values = _result().model_dump()
    values.update(changes)
    with pytest.raises(InvalidRequestError) as exc_info:
        if entry == "constructor":
            ContrastResult(**values)
        elif entry == "model_validate":
            ContrastResult.model_validate(values)
        else:
            ContrastResult.model_validate_json(json.dumps(values))
    error = exc_info.value
    assert error.code == "estimation.contrast.retained_window"
    assert error.context == {
        key: values[key] for key in ("observation_steps", "retained_steps", "carryover_order")
    }
    with pytest.raises(TypeError):
        cast("dict[str, object]", error.context)["retained_steps"] = 1


@pytest.mark.parametrize("field", ["observation_steps", "retained_steps"])
def test_result_wire_requires_declared_window_lengths_and_order_counts(field):
    from increment.errors import InvalidRequestError

    values = _result().model_dump()
    del values[field]
    with pytest.raises(InvalidRequestError) as raised:
        ContrastResult.model_validate(values)
    assert raised.value.code == "model.field.missing"


@pytest.mark.parametrize("field", ["ct_cycles", "tc_cycles"])
def test_result_wire_rejects_partial_order_counts(field):
    from increment.errors import InvalidRequestError

    values = _result().model_dump()
    del values[field]
    with pytest.raises(InvalidRequestError) as raised:
        ContrastResult.model_validate(values)
    assert raised.value.code == "estimation.contrast.order_counts"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("model", ["stats", "result"])
@pytest.mark.parametrize("entry", ["constructor", "model_validate", "model_validate_json"])
@pytest.mark.parametrize("field", ["ct_cycles", "tc_cycles"])
def test_order_count_sum_is_required_at_every_public_entry(shared, model, entry, field):
    import json

    from increment.errors import InvalidRequestError
    from increment.estimation.contrast import ContrastPartition, reduce_contrast_partitions

    if model == "stats":
        original = reduce_contrast_partitions(
            [
                ContrastPartition(
                    metric="orders",
                    aggregation="sum",
                    probability_ct=0.8,
                    randomization_law=(
                        "shared_schedule" if shared else "independent_bernoulli_order"
                    ),
                    independence_grain="shared_block" if shared else "unit_cycle",
                    carryover_order=0,
                    observation_steps=3,
                    retained_steps=3,
                    control_group="control",
                    treatment_group="treatment",
                    unit_deltas={"a": 1.0, "b": 2.0},
                    cycles_by_unit={"a": 3, "b": 3},
                    ct_counts_by_unit={"a": 1, "b": 0},
                )
            ]
        )
    else:
        original = _shared_result() if shared else _result()
    constructor = type(original)
    values = original.model_dump()
    values[field] += 1
    with pytest.raises(InvalidRequestError) as raised:
        if entry == "constructor":
            constructor(**values)
        elif entry == "model_validate":
            constructor.model_validate(values)
        else:
            constructor.model_validate_json(json.dumps(values))
    assert raised.value.code == "estimation.contrast.order_counts"
    assert raised.value.context == {
        name: values[name]
        for name in ("randomization_law", "n_cycles", "n_blocks", "ct_cycles", "tc_cycles")
    }
    with pytest.raises(TypeError):
        cast("dict[str, object]", raised.value.context)[field] = 0
