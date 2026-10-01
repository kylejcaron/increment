import math
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.contrast import ContrastPartition, reduce_contrast_partitions


def part(**overrides):
    values: dict[str, Any] = {
        "metric": "revenue",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "carryover_order": 0,
        "observation_steps": 3,
        "retained_steps": 3,
        "control_group": "control",
        "treatment_group": "treatment",
        "unit_deltas": {"a": 2.0, "b": 4.0},
        "cycles_by_unit": {"a": 2, "b": 3},
    }
    values.update(overrides)
    values.setdefault("ct_counts_by_unit", dict.fromkeys(values["unit_deltas"], 1))
    if "retained_steps" not in overrides:
        values["retained_steps"] = values["observation_steps"] - values["carryover_order"]
    if values["randomization_law"] == "independent_bernoulli_order":
        values.setdefault(
            "exact_unit_deltas",
            {
                key: Fraction(value).as_integer_ratio()
                for key, value in values["unit_deltas"].items()
                if math.isfinite(value)
            },
        )
        values.setdefault("unit_slopes", dict.fromkeys(values["unit_deltas"], 1.0))
    return ContrastPartition(**values)


def test_reduce_contrast_partitions_preserves_centered_moments():
    stats = reduce_contrast_partitions(
        [
            part(unit_deltas={"a": 2.0}, cycles_by_unit={"a": 2}, ct_counts_by_unit={"a": 1}),
            part(unit_deltas={"b": 4.0}, cycles_by_unit={"b": 3}, ct_counts_by_unit={"b": 1}),
        ]
    )
    assert stats.n_units == 2
    assert stats.n_cycles == 5
    assert stats.reference_delta == 2.0
    assert stats.mean_residual == 1.0
    assert stats.m2_delta == pytest.approx(2.0)
    assert stats.observation_steps == stats.retained_steps == 3


@pytest.mark.parametrize("model", ["partition", "stats"])
def test_contrast_metadata_rejects_inconsistent_retained_lengths(model):
    from increment.estimation.contrast import ContrastStats

    original = part() if model == "partition" else reduce_contrast_partitions([part()])
    values = {name: getattr(original, name) for name in type(original).model_fields}
    values["retained_steps"] = 2
    constructor = ContrastPartition if model == "partition" else ContrastStats
    with pytest.raises(InvalidRequestError) as exc_info:
        constructor.model_validate(values)
    error = exc_info.value
    assert error.code == "estimation.contrast.retained_window"
    assert error.context == {"observation_steps": 3, "retained_steps": 2, "carryover_order": 0}
    with pytest.raises(TypeError):
        cast("dict[str, object]", error.context)["retained_steps"] = 3


def test_reduce_rejects_each_identity_mismatch():
    for field, value in {
        "metric": "other",
        "control_group": "other",
        "treatment_group": "other",
        "aggregation": "any",
        "probability_ct": 0.8,
        "carryover_order": 1,
        "observation_steps": 4,
    }.items():
        with pytest.raises(InvalidRequestError) as exc_info:
            reduce_contrast_partitions([part(), part(**{field: value})])
        assert exc_info.value.code == "estimation.contrast.contrast_partition_does"
        assert field in str(exc_info.value.context["mismatch"])

    shared = part(randomization_law="shared_schedule", independence_grain="shared_block")
    with pytest.raises(InvalidRequestError) as exc_info:
        reduce_contrast_partitions([part(), shared])
    assert exc_info.value.code == "estimation.contrast.contrast_partition_does"
    assert exc_info.value.context["mismatch"] == "randomization_law"


def test_reduce_rejects_duplicate_units_and_nonfinite_values():
    with pytest.raises(InvalidRequestError) as exc_info:
        reduce_contrast_partitions(
            [
                part(),
                part(unit_deltas={"b": 5.0}, cycles_by_unit={"b": 1}, ct_counts_by_unit={"b": 0}),
            ]
        )
    assert exc_info.value.code == "estimation.contrast.contrast_partitions_disjoint"
    with pytest.raises(InvalidRequestError) as exc_info:
        part(unit_deltas={"a": math.inf}, cycles_by_unit={"a": 1}, ct_counts_by_unit={"a": 1})
    assert exc_info.value.code == "estimation.contrast.contrast_partition.unit_deltas_finite"
    with pytest.raises(InvalidRequestError) as exc_info:
        part(unit_deltas={"a": 1.0}, cycles_by_unit={"a": 0}, ct_counts_by_unit={"a": 0})
    assert exc_info.value.code == "estimation.contrast.contrast_partition.cycles_positive_integers"


@pytest.mark.parametrize(
    "field", ["unit_deltas", "cycles_by_unit", "unit_slopes", "ct_counts_by_unit"]
)
def test_partition_mappings_are_immutable_after_validation(field):
    partition = part()
    mapping = getattr(partition, field)
    original = dict(mapping)
    assert isinstance(mapping, MappingProxyType)
    with pytest.raises(TypeError):
        dict.__setitem__(mapping, "a", 99)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dict.__ior__(mapping, {"a": 99})  # type: ignore[arg-type]
    assert dict(mapping) == original


def test_reduce_rejects_empty_and_non_disjoint_partition_inputs():
    with pytest.raises(InvalidRequestError) as exc_info:
        reduce_contrast_partitions([])
    assert exc_info.value.code == "estimation.contrast.contrast_reduction_least"
    with pytest.raises(InvalidRequestError) as exc_info:
        ContrastPartition(
            metric="m",
            aggregation="sum",
            probability_ct=0.5,
            randomization_law="independent_bernoulli_order",
            independence_grain="unit_cycle",
            carryover_order=0,
            observation_steps=3,
            retained_steps=3,
            control_group="c",
            treatment_group="t",
            unit_deltas={"a": 1.0},
            cycles_by_unit={"b": 1},
            ct_counts_by_unit={"a": 1},
        )
    assert exc_info.value.code == "estimation.contrast.contrast_partition.unit_deltas_cycles"


def test_reduction_is_stable_for_large_reference_and_small_residuals():
    stats = reduce_contrast_partitions(
        [
            part(unit_deltas={"a": 1e150}, cycles_by_unit={"a": 1}, ct_counts_by_unit={"a": 1}),
            part(
                unit_deltas={"b": 1e150 + 1e134},
                cycles_by_unit={"b": 1},
                ct_counts_by_unit={"b": 0},
            ),
        ]
    )
    assert math.isfinite(stats.mean_residual)
    assert math.isfinite(stats.m2_delta)


def test_reduction_merges_repeated_near_max_offsets_in_centered_coordinates():
    stats = reduce_contrast_partitions(
        [
            part(unit_deltas={unit: 1e308}, cycles_by_unit={unit: 1}, ct_counts_by_unit={unit: 1})
            for unit in ("a", "b", "c")
        ]
    )
    assert stats.reference_delta == 1e308
    assert stats.mean_residual == 0.0
    assert stats.m2_delta == 0.0


def test_reduction_merges_three_large_offset_partitions_without_cancellation():
    values = [1e150, 1e150 + 1e134, 1e150 - 1e134]
    residuals = [value - values[0] for value in values]
    expected_mean = math.fsum(residuals) / len(residuals)
    expected_m2 = math.fsum((residual - expected_mean) ** 2 for residual in residuals)
    stats = reduce_contrast_partitions(
        [
            part(unit_deltas={"a": values[0]}, cycles_by_unit={"a": 1}, ct_counts_by_unit={"a": 1}),
            part(unit_deltas={"b": values[1]}, cycles_by_unit={"b": 1}, ct_counts_by_unit={"b": 0}),
            part(unit_deltas={"c": values[2]}, cycles_by_unit={"c": 1}, ct_counts_by_unit={"c": 0}),
        ]
    )
    assert stats.mean_residual == pytest.approx(expected_mean)
    assert stats.m2_delta == pytest.approx(expected_m2)


@pytest.mark.parametrize("split_parts", [False, True])
def test_shared_blocks_cannot_invent_an_average_roster_size(split_parts):
    shared = {
        "randomization_law": "shared_schedule",
        "independence_grain": "shared_block",
    }
    if split_parts:
        parts = [
            part(
                **shared,
                unit_deltas={"b0": 1.0},
                cycles_by_unit={"b0": 2},
                ct_counts_by_unit={"b0": 1},
            ),
            part(
                **shared,
                unit_deltas={"b1": 2.0},
                cycles_by_unit={"b1": 4},
                ct_counts_by_unit={"b1": 0},
            ),
        ]
    else:
        parts = [
            part(
                **shared,
                unit_deltas={"b0": 1.0, "b1": 2.0},
                cycles_by_unit={"b0": 2, "b1": 4},
                ct_counts_by_unit={"b0": 1, "b1": 0},
            )
        ]
    with pytest.raises(InvalidRequestError) as raised:
        reduce_contrast_partitions(parts)
    assert raised.value.code == "estimation.contrast.shared_roster_size_mismatch"
    assert raised.value.context["roster_sizes"] == (2, 4)


@pytest.mark.parametrize("shared", [False, True])
def test_order_counts_survive_disjoint_partition_recombination(shared):
    metadata = (
        {"randomization_law": "shared_schedule", "independence_grain": "shared_block"}
        if shared
        else {}
    )
    original = part(
        **metadata,
        unit_deltas={"a": 2.0, "b": 4.0, "c": 6.0},
        cycles_by_unit={"a": 3, "b": 3, "c": 3},
        ct_counts_by_unit={"a": 1, "b": 0, "c": 1 if shared else 3},
    )
    pieces = [
        part(
            **metadata,
            unit_deltas={key: original.unit_deltas[key]},
            cycles_by_unit={key: original.cycles_by_unit[key]},
            ct_counts_by_unit={key: original.ct_counts_by_unit[key]},
        )
        for key in reversed(original.unit_deltas)
    ]
    whole = reduce_contrast_partitions([original])
    recombined = reduce_contrast_partitions(pieces)
    assert (recombined.ct_cycles, recombined.tc_cycles) == (whole.ct_cycles, whole.tc_cycles)
    assert (recombined.n_units, recombined.n_cycles, recombined.n_blocks) == (
        whole.n_units,
        whole.n_cycles,
        whole.n_blocks,
    )
    assert recombined.reference_delta == whole.reference_delta
    assert recombined.mean_residual == pytest.approx(whole.mean_residual)
    assert recombined.m2_delta == pytest.approx(whole.m2_delta)
    assert (whole.ct_cycles, whole.tc_cycles) == ((2, 1) if shared else (4, 5))


@pytest.mark.parametrize(
    "counts,code",
    [
        ({"a": 1}, "estimation.contrast.contrast_partition.unit_deltas_cycles"),
        ({"a": 1, "b": 1, "c": 0}, "estimation.contrast.contrast_partition.unit_deltas_cycles"),
        ({"a": -1, "b": 1}, "estimation.contrast.contrast_partition.ct_counts"),
        ({"a": True, "b": 1}, "estimation.contrast.contrast_partition.ct_counts"),
        ({"a": 1.0, "b": 1}, "estimation.contrast.contrast_partition.ct_counts"),
        ({"a": 3, "b": 1}, "estimation.contrast.contrast_partition.ct_counts"),
        ([], "estimation.contrast.contrast_partition.ct_counts"),
    ],
)
def test_partition_rejects_invalid_order_counts(counts, code):
    with pytest.raises(InvalidRequestError) as raised:
        part(ct_counts_by_unit=counts)
    assert raised.value.code == code


def test_shared_partition_ct_counts_cannot_count_roster_members():
    with pytest.raises(InvalidRequestError) as raised:
        part(
            randomization_law="shared_schedule",
            independence_grain="shared_block",
            ct_counts_by_unit={"a": 2, "b": 0},
        )
    assert raised.value.code == "estimation.contrast.contrast_partition.ct_counts"


def test_partition_copies_caller_owned_order_counts():
    counts = {"a": 1, "b": 0}
    partition = part(ct_counts_by_unit=counts)
    counts["a"] = 0
    counts.clear()
    assert dict(partition.ct_counts_by_unit) == {"a": 1, "b": 0}
    stats = reduce_contrast_partitions([partition])
    assert (stats.ct_cycles, stats.tc_cycles) == (1, 4)


def test_partition_slopes_counts_and_cycle_range_survive_merge_and_replay():
    from increment.estimation.contrast import ContrastStats

    left = part(
        unit_deltas={"a": 2},
        exact_unit_deltas={"a": (2, 1)},
        cycles_by_unit={"a": 1},
        unit_slopes={"a": 1},
        ct_counts_by_unit={"a": 1},
    )
    right = part(
        unit_deltas={"b": 4},
        exact_unit_deltas={"b": (4, 1)},
        cycles_by_unit={"b": 3},
        unit_slopes={"b": 1},
        ct_counts_by_unit={"b": 2},
    )
    stats = reduce_contrast_partitions([left, right])
    assert stats.mean_slope == 1
    assert stats.exact_delta_total == (6, 1)
    assert (stats.ct_cycles, stats.tc_cycles) == (3, 1)
    assert (stats.minimum_cycles_per_unit, stats.maximum_cycles_per_unit) == (1, 3)
    assert ContrastStats.model_validate_json(stats.model_dump_json()) == stats
    assert ContrastPartition.model_validate_json(left.model_dump_json()) == left
    reversed_stats = reduce_contrast_partitions([right, left])
    assert reversed_stats == stats


@pytest.mark.parametrize(
    "changes",
    [
        {"unit_slopes": {"a": 1}},
        {"unit_slopes": {"a": 0, "b": 1}},
    ],
)
def test_unit_partition_requires_valid_slopes_and_exact_counts(changes):
    with pytest.raises(InvalidRequestError) as caught:
        part(**changes)
    assert caught.value.code == "unit_cycle.sufficient_state"


def test_exact_ratios_copy_nested_inputs_and_have_canonical_json_state():
    import json

    from increment.estimation.contrast import ContrastStats

    ratios = {"b": [12, 3], "a": [4, 2]}
    partition = part(exact_unit_deltas=ratios)
    ratios["a"][0] = 999
    ratios.clear()
    assert dict(partition.exact_unit_deltas) == {"a": (2, 1), "b": (4, 1)}
    with pytest.raises(TypeError):
        partition.exact_unit_deltas["a"] = (999, 1)  # type: ignore[index]
    payload = json.loads(partition.model_dump_json())
    assert list(payload["exact_unit_deltas"]) == ["a", "b"]
    assert payload["exact_unit_deltas"] == {"a": [2, 1], "b": [4, 1]}
    assert ContrastPartition.model_validate_json(partition.model_dump_json()) == partition
    stats = reduce_contrast_partitions([partition])
    assert stats.exact_delta_total == (6, 1)
    restored = ContrastStats.model_validate_json(stats.model_dump_json())
    assert restored == stats
    data = stats.model_dump()
    ratio = [12, 2]
    data["exact_delta_total"] = ratio
    copied = ContrastStats.model_validate(data)
    ratio[0] = 999
    assert copied.exact_delta_total == (6, 1)


@pytest.mark.parametrize("ratio", [(1, 0), (1, -1), (True, 1), (1, False), (1.0, 2), (1,), "1/2"])
@pytest.mark.parametrize("model", ["partition", "stats"])
def test_exact_ratios_reject_invalid_integer_pairs(ratio, model):
    from increment.estimation.contrast import ContrastStats

    with pytest.raises(InvalidRequestError) as caught:
        if model == "partition":
            part(exact_unit_deltas={"a": ratio, "b": (4, 1)})
        else:
            values = reduce_contrast_partitions([part()]).model_dump()
            values["exact_delta_total"] = ratio
            ContrastStats.model_validate(values)
    assert caught.value.code == "unit_cycle.sufficient_state"


def test_exact_partition_state_requires_matching_units_and_unit_cycle_law():
    with pytest.raises(InvalidRequestError) as caught:
        part(exact_unit_deltas={"a": (2, 1)})
    assert caught.value.code == "unit_cycle.sufficient_state"
    with pytest.raises(InvalidRequestError) as caught:
        part(
            randomization_law="shared_schedule",
            independence_grain="shared_block",
            exact_unit_deltas={"a": (2, 1), "b": (4, 1)},
        )
    assert caught.value.code == "unit_cycle.sufficient_state"


def test_exact_partition_sum_does_not_reconstruct_from_rounded_unit_deltas():
    from itertools import permutations

    exact_values = [Fraction(10**16), Fraction(2, 3), Fraction(-(10**16))]
    parts = [
        part(
            unit_deltas={unit: float(value)},
            exact_unit_deltas={unit: (value.numerator, value.denominator)},
            cycles_by_unit={unit: 1},
        )
        for unit, value in zip(("a", "b", "c"), exact_values, strict=True)
    ]
    expected = exact_values[1]
    for ordered in permutations(parts):
        stats = reduce_contrast_partitions(ordered)
        assert stats.exact_delta_total == (expected.numerator, expected.denominator)


def test_missing_partition_exact_state_stays_unavailable_without_float_fallback():
    exact = part(unit_deltas={"a": 2}, exact_unit_deltas={"a": (2, 1)}, cycles_by_unit={"a": 1})
    historical = part(
        unit_deltas={"b": 4},
        cycles_by_unit={"b": 1},
        unit_slopes={},
        ct_counts_by_unit={},
        exact_unit_deltas={},
    )
    assert reduce_contrast_partitions([exact, historical]).exact_delta_total is None


@pytest.mark.parametrize(
    "fields",
    [
        ("unit_slopes",),
        ("ct_counts_by_unit",),
        ("exact_unit_deltas",),
        ("unit_slopes", "ct_counts_by_unit"),
        ("ct_counts_by_unit", "exact_unit_deltas"),
    ],
)
def test_partition_rejects_every_partial_metadata_combination(fields):
    payload = part().model_dump()
    for field in fields:
        payload.pop(field)
    with pytest.raises(InvalidRequestError) as caught:
        ContrastPartition.model_validate(payload)
    assert caught.value.code == "unit_cycle.sufficient_state"


def test_count_only_partition_preserves_orders_without_inventing_exact_state():
    payload = part().model_dump()
    payload.pop("unit_slopes")
    payload.pop("exact_unit_deltas")
    partition = ContrastPartition.model_validate(payload)
    restored = ContrastPartition.model_validate_json(partition.model_dump_json())
    stats = reduce_contrast_partitions([restored])
    assert (stats.ct_cycles, stats.tc_cycles) == (2, 3)
    assert stats.reference_delta + stats.mean_residual == 3
    assert stats.mean_slope is None
    assert stats.exact_delta_total is None


def test_historical_float_partition_roundtrip_and_mixed_reduction_leave_metadata_unavailable():
    from increment.estimation.contrast import ContrastStats

    payload = part().model_dump()
    for field in ("unit_slopes", "ct_counts_by_unit", "exact_unit_deltas"):
        payload.pop(field)
    legacy = ContrastPartition.model_validate(payload)
    assert ContrastPartition.model_validate_json(legacy.model_dump_json()) == legacy
    current = part(unit_deltas={"c": 6}, cycles_by_unit={"c": 1})
    for parts in ([legacy], [legacy, current], [current, legacy]):
        stats = reduce_contrast_partitions(parts)
        assert stats.reference_delta + stats.mean_residual == (3 if len(parts) == 1 else 4)
        for field in (
            "mean_slope",
            "ct_cycles",
            "tc_cycles",
            "minimum_cycles_per_unit",
            "maximum_cycles_per_unit",
            "exact_delta_total",
        ):
            assert getattr(stats, field) is None
        assert ContrastStats.model_validate_json(stats.model_dump_json()) == stats


@pytest.mark.parametrize("values", [(1e200, -1e200), (1e308, -1e308)])
@pytest.mark.parametrize("shared", [False, True])
def test_finite_partition_centered_overflow_is_coded(values, shared):
    partition = part(
        unit_deltas=dict(zip(("a", "b"), values, strict=True)),
        cycles_by_unit={"a": 2, "b": 2},
        **(
            {"randomization_law": "shared_schedule", "independence_grain": "shared_block"}
            if shared
            else {}
        ),
    )
    with pytest.raises(InvalidRequestError) as caught:
        reduce_contrast_partitions([partition])
    assert caught.value.code == "estimation.contrast.contrast_partition_centered"
