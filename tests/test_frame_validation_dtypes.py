"""Column-dtype coercion gate: every metric/denominator/covariate/uptake
column is numeric, or the frame is refused, identically on every backend.
"""

from __future__ import annotations

from datetime import date
from fractions import Fraction
from typing import cast

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from increment import Analysis, readouts
from increment.errors import InvalidRequestError
from increment.frame import MetricSpec, from_unit_panel, from_unit_summary
from increment.semantics.models import MeanMetric

_CONTROL_STRINGS = ["1.0", "2.0", "nan", "3.0", "nan", "4.0", "5.0", "nan", "6.0", "7.0"]
_TREATMENT_STRINGS = [str(float(i)) for i in range(10, 20)]
_UNIT_IDS = [f"c{i}" for i in range(10)] + [f"t{i}" for i in range(10)]
_VARIANTS = ["control"] * 10 + ["treatment"] * 10


def test_pandas_string_metric_column_refuses_with_offending_values() -> None:
    frame = pd.DataFrame(
        {"unit_id": _UNIT_IDS, "variant": _VARIANTS, "rev": _CONTROL_STRINGS + _TREATMENT_STRINGS}
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame, unit="unit_id", group="variant", control="control", metrics={"rev": "mean"}
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["offending_values"] == ("nan", "nan", "nan")


def test_pyarrow_string_metric_column_refuses_same_code_as_pandas() -> None:
    table = pa.table(
        {"unit_id": _UNIT_IDS, "variant": _VARIANTS, "rev": _CONTROL_STRINGS + _TREATMENT_STRINGS}
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            table, unit="unit_id", group="variant", control="control", metrics={"rev": "mean"}
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["offending_values"] == ("nan", "nan", "nan")


def _moments_by_group(frame):
    src = from_unit_summary(
        frame, unit="unit_id", group="variant", control="control", metrics={"rev": "mean"}
    )
    metric = cast(MeanMetric, src.context.metrics[0])
    return {row["group_id"]: row for row in src.moments(metric)}


def test_object_dtype_float_column_matches_float64_twin_bitwise() -> None:
    values = [1.5, 2.25, 3.75, 4.125, 5.0, 6.5, 7.25, 8.75, 9.125, 10.0]
    treatment_values = [v + 5.0 for v in values]
    base = pd.DataFrame(
        {"unit_id": _UNIT_IDS, "variant": _VARIANTS, "rev": values + treatment_values}
    )
    native = base.copy()
    native["rev"] = native["rev"].astype("float64")
    objectified = base.copy()
    objectified["rev"] = objectified["rev"].astype(object)

    native_rows = _moments_by_group(native)
    object_rows = _moments_by_group(objectified)
    for group in ("control", "treatment"):
        assert object_rows[group]["ref_y"] == native_rows[group]["ref_y"]
        assert object_rows[group]["cy1"] == native_rows[group]["cy1"]
        assert object_rows[group]["cy2"] == native_rows[group]["cy2"]


def test_polars_boolean_metric_casts_to_float_and_matches_its_exact_mean() -> None:
    frame = pl.DataFrame(
        {
            "unit_id": [f"c{i}" for i in range(5)] + [f"t{i}" for i in range(5)],
            "variant": ["control"] * 5 + ["treatment"] * 5,
            "converted": [True, False, True, False, True, True, True, True, False, True],
        }
    )
    src = from_unit_summary(
        frame, unit="unit_id", group="variant", control="control", metrics={"converted": "mean"}
    )
    metric = cast(MeanMetric, src.context.metrics[0])
    rows = {row["group_id"]: row for row in src.moments(metric)}
    assert rows["control"]["ref_y"] == 0.6
    assert rows["treatment"]["ref_y"] == 0.8


def test_object_dtype_nan_is_null_like_its_float64_twin_under_missing_zero() -> None:
    values = [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0]
    treatment_values = [10.0, 12.0, float("nan"), 14.0, 16.0, float("nan")]
    unit_ids = [f"c{i}" for i in range(6)] + [f"t{i}" for i in range(6)]
    variants = ["control"] * 6 + ["treatment"] * 6
    base = pd.DataFrame(
        {"unit_id": unit_ids, "variant": variants, "rev": values + treatment_values}
    )
    native = base.copy()
    native["rev"] = native["rev"].astype("float64")
    objectified = base.copy()
    objectified["rev"] = objectified["rev"].astype(object)

    spec = [MetricSpec(name="rev", missing="zero")]

    def _moments(frame):
        src = from_unit_summary(
            frame, unit="unit_id", group="variant", control="control", metrics=spec
        )
        metric = cast(MeanMetric, src.context.metrics[0])
        return {row["group_id"]: row for row in src.moments(metric)}

    native_rows = _moments(native)
    object_rows = _moments(objectified)
    for group in ("control", "treatment"):
        assert object_rows[group]["ref_y"] == native_rows[group]["ref_y"]
        assert object_rows[group]["cy1"] == native_rows[group]["cy1"]
        assert object_rows[group]["cy2"] == native_rows[group]["cy2"]


def test_pandas_nullable_string_na_is_null_like_its_float64_twin_under_missing_zero() -> None:
    numeric_values = [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0]
    numeric_treatment = [10.0, 12.0, float("nan"), 14.0, 16.0, float("nan")]
    string_values = ["1.0", pd.NA, "3.0", "4.0", pd.NA, "6.0"]
    string_treatment = ["10.0", "12.0", pd.NA, "14.0", "16.0", pd.NA]
    unit_ids = [f"c{i}" for i in range(6)] + [f"t{i}" for i in range(6)]
    variants = ["control"] * 6 + ["treatment"] * 6

    native = pd.DataFrame(
        {"unit_id": unit_ids, "variant": variants, "rev": numeric_values + numeric_treatment}
    )
    native["rev"] = native["rev"].astype("float64")
    nullable = pd.DataFrame(
        {"unit_id": unit_ids, "variant": variants, "rev": string_values + string_treatment}
    )
    nullable["rev"] = nullable["rev"].astype("string")

    spec = [MetricSpec(name="rev", missing="zero")]

    def _moments(frame):
        src = from_unit_summary(
            frame, unit="unit_id", group="variant", control="control", metrics=spec
        )
        metric = cast(MeanMetric, src.context.metrics[0])
        return {row["group_id"]: row for row in src.moments(metric)}

    native_rows = _moments(native)
    nullable_rows = _moments(nullable)
    for group in ("control", "treatment"):
        assert nullable_rows[group]["ref_y"] == native_rows[group]["ref_y"]
        assert nullable_rows[group]["cy1"] == native_rows[group]["cy1"]
        assert nullable_rows[group]["cy2"] == native_rows[group]["cy2"]


def test_object_dtype_numpy_float32_nan_is_null_like_its_float64_twin_under_missing_zero() -> None:
    numeric_values = [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0]
    numeric_treatment = [10.0, 12.0, float("nan"), 14.0, 16.0, float("nan")]
    object_values = [1.0, np.float32("nan"), 3.0, 4.0, np.float32("nan"), 6.0]
    object_treatment = [10.0, 12.0, np.float32("nan"), 14.0, 16.0, np.float32("nan")]
    unit_ids = [f"c{i}" for i in range(6)] + [f"t{i}" for i in range(6)]
    variants = ["control"] * 6 + ["treatment"] * 6

    native = pd.DataFrame(
        {"unit_id": unit_ids, "variant": variants, "rev": numeric_values + numeric_treatment}
    )
    native["rev"] = native["rev"].astype("float64")
    # A numpy object array (not a list->astype round trip) is what keeps each
    # cell's exact scalar type, so np.float32("nan") survives as float32.
    rev = np.empty(12, dtype=object)
    rev[:] = object_values + object_treatment
    objectified = pd.DataFrame({"unit_id": unit_ids, "variant": variants, "rev": rev})

    spec = [MetricSpec(name="rev", missing="zero")]

    def _moments(frame):
        src = from_unit_summary(
            frame, unit="unit_id", group="variant", control="control", metrics=spec
        )
        metric = cast(MeanMetric, src.context.metrics[0])
        return {row["group_id"]: row for row in src.moments(metric)}

    native_rows = _moments(native)
    object_rows = _moments(objectified)
    for group in ("control", "treatment"):
        assert object_rows[group]["ref_y"] == native_rows[group]["ref_y"]
        assert object_rows[group]["cy1"] == native_rows[group]["cy1"]
        assert object_rows[group]["cy2"] == native_rows[group]["cy2"]


@pytest.mark.parametrize("oversized", [10**400, Fraction(10**400, 3)])
def test_object_dtype_oversized_real_refuses_with_the_dtype_code(oversized) -> None:
    """A real too large for a float must reach the coded refusal, not OverflowError."""
    frame = pd.DataFrame(
        {
            "unit_id": ["u1", "u2", "u3", "u4"],
            "variant": ["a", "b", "a", "b"],
            "rev": pd.Series([1.0, 2.0, oversized, 3.0], dtype=object),
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        from_unit_summary(
            frame, unit="unit_id", group="variant", control="a", metrics={"rev": "mean"}
        )
    assert exc_info.value.code == "source.frame.column_dtype"
    assert exc_info.value.context["column"] == "rev"


def _group_label_frame(labels):
    return pd.DataFrame(
        [
            {
                "unit": f"{arm}:{i}",
                "arm": label,
                "day": date(2025, 1, 1),
                "latency": 10.0 * (arm + 1) + i / 100,
            }
            for arm, label in enumerate(labels)
            for i in range(100)
        ]
    )


@pytest.mark.parametrize("labels", [[1, "1"], [True, "True"], [1, True, "True"]])
def test_canonical_collision_refuses_before_quantile_estimation(labels):
    frame = _group_label_frame(["control", *labels])
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=[MetricSpec(name="latency", type="quantile", quantile=0.5)],
        )
    assert raised.value.code == "frame.validation.group_label_collision"


@pytest.mark.parametrize("panel", [False, True])
def test_canonical_collision_refuses_before_moment_reduction(panel):
    constructor = from_unit_panel if panel else from_unit_summary
    with pytest.raises(InvalidRequestError) as raised:
        constructor(
            _group_label_frame(["control", 1, "1"]),
            unit="unit",
            group="arm",
            control="control",
            metrics={"latency": "mean"},
            **({"date": "day"} if panel else {}),
        )
    assert raised.value.code == "frame.validation.group_label_collision"


@pytest.mark.parametrize("label", ["(unassigned)", "(mixed assignment)"])
def test_reserved_group_label_refuses_before_accounting(label):
    with pytest.raises(InvalidRequestError) as raised:
        from_unit_summary(
            _group_label_frame(["control", label]),
            unit="unit",
            group="arm",
            control="control",
            metrics={"latency": "mean"},
        )
    assert raised.value.code == "frame.validation.group_label_reserved"


def test_string_group_labels_retain_separate_quantile_contrasts():
    source = from_unit_summary(
        _group_label_frame(["control", "1", "2"]),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="latency", type="quantile", quantile=0.5)],
    )
    results = {row.group_id: row for row in readouts.run(source)}
    assert set(results) == {"1", "2"}
    for label, median in [("1", 20.495), ("2", 30.495)]:
        assert results[label].require_lift().value == pytest.approx(median / 10.495 - 1)
    assert source.unit_counts() == {"control": 100, "1": 100, "2": 100}


@pytest.mark.parametrize("panel", [False, True])
def test_missing_group_is_not_the_literal_nan_arm(panel):
    frame = _group_label_frame(["control", "nan"])
    frame.loc[len(frame)] = ["missing", float("nan"), date(2025, 1, 1), 100.0]
    constructor = from_unit_panel if panel else from_unit_summary
    source = constructor(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics={"latency": "mean"},
        on_unassigned="exclude",
        **({"date": "day"} if panel else {}),
    )
    assert source.unit_counts() == {"control": 100, "nan": 100, "(unassigned)": 1}
    result = readouts.run(source)[0]
    assert result.group_id == "nan"
    assert result.lift is not None
    assert result.lift.value == pytest.approx(20.495 / 10.495 - 1)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_float_nan_group_stays_unassigned_after_native_cast(backend):
    frame = _group_label_frame([0.0, 1.0])
    frame.loc[len(frame)] = ["missing", float("nan"), date(2025, 1, 1), 100.0]
    data = frame.to_dict(orient="list")
    native = {"pandas": pd.DataFrame, "polars": pl.DataFrame, "pyarrow": pa.table}[backend](data)
    canonical = nw.from_native(native, eager_only=True).select(nw.col("arm").cast(nw.String))
    control, treatment = canonical["arm"][0], canonical["arm"][100]
    source = from_unit_summary(
        native,
        unit="unit",
        group="arm",
        control=control,
        metrics={"latency": "mean"},
        on_unassigned="exclude",
    )
    assert source.unit_counts() == {control: 100, treatment: 100, "(unassigned)": 1}
    result = readouts.run(source)[0]
    assert result.group_id == treatment
    assert result.lift is not None
    assert result.lift.value == pytest.approx(20.495 / 10.495 - 1)


@pytest.mark.parametrize("panel", [False, True])
def test_numeric_group_column_accepts_canonical_string_control(panel):
    constructor = from_unit_panel if panel else from_unit_summary
    source = constructor(
        _group_label_frame([0, 1]),
        unit="unit",
        group="arm",
        control="0",
        metrics={"latency": "mean"},
        **({"date": "day"} if panel else {}),
    )
    assert source.unit_counts() == {"0": 100, "1": 100}
    result = readouts.run(source)[0]
    assert result.group_id == "1"
    assert result.lift is not None
    assert result.lift.value == pytest.approx(20.495 / 10.495 - 1)
