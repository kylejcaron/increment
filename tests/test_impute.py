"""Tests for ``increment.impute`` - the helper tier of the null/NaN policy.

Every frame-entry refusal names these helpers as its source-side fix, so
they must treat null and NaN as one "missing" concept, work identically on
every installed backend, return the same native frame type they were
given, and always report how many rows they touched.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pyarrow as pa
import pytest

from increment import impute
from increment.errors import IncrementWarning


def _backend_frames() -> list[tuple[str, Any]]:
    """The null/NaN fixture on every installed backend. ``a`` mixes a null
    and (where the backend distinguishes them) a NaN; ``b`` is complete."""
    frames: list[tuple[str, Any]] = [
        (
            "pyarrow",
            pa.table({"a": [1.0, None, float("nan"), 4.0], "b": [10.0, 20.0, 30.0, 40.0]}),
        )
    ]
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        frames.append(
            (
                "pandas",
                pd.DataFrame({"a": [1.0, None, float("nan"), 4.0], "b": [10.0, 20.0, 30.0, 40.0]}),
            )
        )
    if importlib.util.find_spec("polars") is not None:
        import polars as pl

        frames.append(
            (
                "polars",
                pl.DataFrame({"a": [1.0, None, float("nan"), 4.0], "b": [10.0, 20.0, 30.0, 40.0]}),
            )
        )
    return frames


def _column(frame: Any, name: str) -> list[float]:
    import narwhals as nw

    return nw.from_native(frame, eager_only=True)[name].to_list()


@pytest.mark.parametrize(("backend", "frame"), _backend_frames())
def test_zeros_fills_null_and_nan_identically(backend: str, frame: Any) -> None:
    out, affected = impute.zeros(frame, "a")
    assert affected == 2
    assert _column(out, "a") == [1.0, 0.0, 0.0, 4.0]
    assert type(out) is type(frame)  # same native frame type back


@pytest.mark.parametrize(("backend", "frame"), _backend_frames())
def test_pooled_mean_uses_observed_values_only(backend: str, frame: Any) -> None:
    out, affected = impute.pooled_mean(frame, "a", roles={"a": "covariate"})
    assert affected == 2
    assert _column(out, "a") == [1.0, 2.5, 2.5, 4.0]  # mean(1, 4)


@pytest.mark.parametrize(("backend", "frame"), _backend_frames())
def test_drop_null_returns_complete_cases_and_count(backend: str, frame: Any) -> None:
    out, dropped = impute.drop_null(frame, "a", "b")
    assert dropped == 2
    assert _column(out, "a") == [1.0, 4.0]
    assert _column(out, "b") == [10.0, 40.0]


def test_multi_column_affected_count_is_rows_not_cells() -> None:
    frame = pa.table({"a": [None, 2.0, None], "b": [None, None, 3.0]})
    _, affected = impute.zeros(frame, "a", "b")
    assert affected == 3  # every row touched at least once, counted once
    _, affected_a = impute.zeros(frame, "a")
    assert affected_a == 2


def test_clean_columns_are_untouched_and_report_zero() -> None:
    frame = pa.table({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out, affected = impute.pooled_mean(frame, "a", "b", roles={"a": "covariate", "b": "covariate"})
    assert affected == 0
    assert _column(out, "a") == [1.0, 2.0]


def test_pooled_mean_promotes_integer_columns() -> None:
    frame = pa.table({"a": [1, None, 4]})
    out, affected = impute.pooled_mean(frame, "a", roles={"a": "covariate"})
    assert affected == 1
    assert _column(out, "a") == [1.0, 2.5, 4.0]


def test_pooled_mean_refuses_a_column_with_no_observed_values() -> None:
    from increment.errors import InvalidRequestError

    frame = pa.table({"a": pa.array([None, None], type=pa.float64())})
    with pytest.raises(InvalidRequestError) as exc_info:
        impute.pooled_mean(frame, "a", roles={"a": "covariate"})
    assert exc_info.value.code == "impute.impute_pooled_mean"
    assert exc_info.value.context["col"] == "a"


def test_pooled_mean_refuses_a_pandas_all_missing_column() -> None:
    """pandas' mean of an all-NaN float column is NaN, not None - the
    all-missing check must reject a non-finite mean too, or the NaNs
    silently survive into the returned frame with affected > 0."""
    from increment.errors import InvalidRequestError

    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": [None, None]})
    with pytest.raises(InvalidRequestError) as exc_info:
        impute.pooled_mean(frame, "x", roles={"x": "covariate"})
    assert exc_info.value.code == "impute.impute_pooled_mean"
    assert exc_info.value.context["col"] == "x"


def test_unknown_column_is_named_with_available_alternatives() -> None:
    from increment.errors import InvalidRequestError

    frame = pa.table({"a": [1.0]})
    with pytest.raises(InvalidRequestError) as exc_info:
        impute.zeros(frame, "nope")
    assert exc_info.value.code == "impute.impute_column_found"
    assert exc_info.value.context["missing"] == ("nope",)
    assert exc_info.value.context["present"] == ("a",)


def test_no_columns_is_refused() -> None:
    from increment.errors import InvalidRequestError

    frame = pa.table({"a": [1.0]})
    with pytest.raises(InvalidRequestError) as exc_info:
        impute.drop_null(frame)
    assert exc_info.value.code == "impute.impute_name_least"
    assert exc_info.value.context["helper"] == "drop_null"


class TestPooledMeanDistinguishesEmptyFromNonFinite:
    """A non-finite mean is not proof the column is empty: blaming missing data
    sends the caller looking for the wrong problem."""

    @staticmethod
    def _call(values):
        from increment.impute import pooled_mean

        return pooled_mean(pa.table({"x": values}), "x", roles={"x": "covariate"})

    def test_an_all_null_column_reports_no_observed_values(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            self._call([None, None, None])
        assert exc_info.value.code == "impute.impute_pooled_mean"

    def test_observed_values_that_overflow_when_averaged_say_so(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            self._call([1e308, 1e308, None])
        assert exc_info.value.code == "impute.impute_pooled_mean_not_finite"
        assert exc_info.value.context["observed"] == 2

    def test_an_ordinary_gap_is_still_filled(self):
        filled, affected = self._call([1.0, 3.0, None])
        assert affected == 1
        assert filled.column("x").to_pylist() == [1.0, 3.0, 2.0]


def _only_coded_warning(record: Any) -> IncrementWarning:
    (item,) = record
    assert isinstance(item.message, IncrementWarning)
    return item.message


def _frame_with_gap() -> Any:
    return pa.table({"a": [1.0, None, 4.0], "b": [None, None, 6.0], "c": [1.0, 2.0, 3.0]})


def test_pooled_mean_refuses_a_declared_outcome_before_any_fill() -> None:
    from increment.errors import InvalidRequestError

    frame = _frame_with_gap()
    with pytest.raises(InvalidRequestError) as exc_info:
        impute.pooled_mean(frame, "a", "b", roles={"a": "covariate", "b": "outcome"})
    assert exc_info.value.code == "impute.pooled_mean_outcome"
    assert exc_info.value.context["columns"] == ("b",)
    assert "route" in exc_info.value.context
    assert frame.column("a").to_pylist() == [1.0, None, 4.0]


def test_pooled_mean_outcome_refusal_names_a_clean_outcome_column_too() -> None:
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        impute.pooled_mean(_frame_with_gap(), "c", roles={"c": "outcome"})
    assert exc_info.value.code == "impute.pooled_mean_outcome"


def test_pooled_mean_warns_per_column_counts_for_undeclared_roles() -> None:
    from increment.errors import IncrementRuntimeWarning

    frame = _frame_with_gap()
    with pytest.warns(IncrementRuntimeWarning) as record:
        out, affected = impute.pooled_mean(frame, "a", "b", "c")
    assert affected == 2
    assert _column(out, "a") == [1.0, 2.5, 4.0]
    assert _column(out, "b") == [6.0, 6.0, 6.0]
    warning = _only_coded_warning(record)
    assert warning.code == "impute.pooled_mean_role_undeclared"
    context = warning.context
    assert context["columns"] == ("a", "b")
    assert context["affected_counts"] == {"a": 1, "b": 2}
    assert "route" in context


def test_pooled_mean_warns_only_for_the_undeclared_column_that_was_filled() -> None:
    from increment.errors import IncrementRuntimeWarning

    with pytest.warns(IncrementRuntimeWarning) as record:
        impute.pooled_mean(_frame_with_gap(), "a", "b", roles={"a": "covariate"})
    context = _only_coded_warning(record).context
    assert context["columns"] == ("b",)
    assert context["affected_counts"] == {"b": 2}


def test_pooled_mean_is_silent_when_every_column_is_declared_covariate() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out, affected = impute.pooled_mean(
            _frame_with_gap(), "a", "b", roles={"a": "covariate", "b": "covariate"}
        )
    assert [w for w in record if issubclass(w.category, IncrementWarning)] == []
    assert affected == 2
    assert _column(out, "a") == [1.0, 2.5, 4.0]


def test_pooled_mean_is_silent_for_an_undeclared_column_with_no_gaps() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _, affected = impute.pooled_mean(_frame_with_gap(), "c")
    assert [w for w in record if issubclass(w.category, IncrementWarning)] == []
    assert affected == 0


def test_pooled_mean_roles_must_name_requested_columns_with_known_values() -> None:
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as unknown:
        impute.pooled_mean(_frame_with_gap(), "a", roles={"c": "covariate"})
    assert unknown.value.code == "impute.impute_roles_columns"
    assert unknown.value.context["columns"] == ("c",)

    with pytest.raises(InvalidRequestError) as bad_value:
        impute.pooled_mean(_frame_with_gap(), "a", roles={"a": "baseline"})  # ty: ignore[invalid-argument-type]
    assert bad_value.value.code == "impute.impute_roles_value"
    assert bad_value.value.context["columns"] == ("a",)


def test_pooled_mean_role_diagnostics_survive_pickle_and_deepcopy() -> None:
    import copy
    import pickle

    from increment.errors import IncrementRuntimeWarning, InvalidRequestError

    with pytest.raises(InvalidRequestError) as refusal:
        impute.pooled_mean(_frame_with_gap(), "a", roles={"a": "outcome"})
    with pytest.warns(IncrementRuntimeWarning) as record:
        impute.pooled_mean(_frame_with_gap(), "a")
    for original in (refusal.value, _only_coded_warning(record)):
        for clone in (pickle.loads(pickle.dumps(original)), copy.deepcopy(original)):
            assert type(clone) is type(original)
            assert clone.code == original.code
            assert clone.context == original.context
