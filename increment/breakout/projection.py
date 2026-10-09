"""Typed estimate-to-frame projection, separate from estimation policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal, get_args, get_origin

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel

from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.results import BinomialConfidenceSet, Estimate
from increment.estimation.sequential_result import SequentialInferenceResult

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "breakout.to_frame_model": "to_frame(): model={model} does not match estimates[0]'s actual type {inferred}",
        "breakout.to_frame_infer": "to_frame(): cannot infer the schema from an empty sequence -- pass model=<the result model class> explicitly, or call <Model>Estimates(estimates).to_frame() instead, which always knows its own model even when empty.",
    },
)
_refuse = raiser(_REFUSALS)
Backend = Literal["pandas", "polars", "pyarrow"]


def _scalar_dtype(annotation: Any) -> nw.dtypes.DType:
    """Map a pydantic field's declared type to a narwhals dtype.

    Unwraps ``X | None`` to ``X`` first. ``bool`` is checked before
    ``int`` since Python's ``bool`` is a subclass of ``int``.
    """
    args = [a for a in get_args(annotation) if a is not type(None)]
    base = args[0] if args else annotation
    if base is date or base is datetime:
        return nw.Datetime()
    if base is bool:
        # An OPTIONAL bool needs a null-carrying pandas column - see
        # _is_optional_bool and to_frame's use of it.
        return nw.Boolean()
    if base is int:
        # An OPTIONAL int must survive a None cell; pandas' numpy-backed
        # Int64 cannot, so it rides as Float64 instead.
        return nw.Float64() if type(None) in get_args(annotation) else nw.Int64()
    if base is float:
        return nw.Float64()
    return nw.String()


def _is_optional_float(annotation: Any) -> bool:
    """Whether a field is declared ``float | None``."""
    args = get_args(annotation)
    return float in args and type(None) in args


def _is_optional_bool(annotation: Any) -> bool:
    """Whether a field is declared ``bool | None`` - the one type
    pandas' numpy-backed boolean column cannot represent a null for."""
    args = get_args(annotation)
    return bool in args and type(None) in args


def _is_optional_string(annotation: Any) -> bool:
    """Whether a field is a nullable string or string-valued literal."""
    args = get_args(annotation)
    if type(None) not in args:
        return False
    return any(arg is str or get_origin(arg) is Literal for arg in args if arg is not type(None))


def _is_optional_model(annotation: Any) -> bool:
    args = get_args(annotation)
    return type(None) in args and any(
        isinstance(arg, type) and issubclass(arg, BaseModel)
        for arg in args
        if arg is not type(None)
    )


def _is_estimate_annotation(annotation: Any) -> bool:
    """Whether a field is an ``Estimate`` or optional ``Estimate``."""
    return annotation is Estimate or Estimate in get_args(annotation)


def _is_optional_binomial_set(annotation: Any) -> bool:
    """Whether a field is declared ``BinomialConfidenceSet | None`` - the
    sole confidence-set-typed field. Detected by name, not structurally
    like :func:`_is_optional_model`: it needs typed ``set_lower``/
    ``set_upper``/``set_level``/``set_numerical_qualification`` columns in
    :func:`to_frame`, not that generic BaseModel-valued-scalar's ``repr()``
    string fallback (which every OTHER model-valued field, e.g. ``prior_spec``,
    still gets)."""
    args = get_args(annotation)
    return type(None) in args and any(
        isinstance(arg, type) and issubclass(arg, BinomialConfidenceSet)
        for arg in args
        if arg is not type(None)
    )


def _frame_columns(
    model: type[BaseModel], estimate_field: str | None, binomial_set_field: str | None
) -> list[str]:
    columns: list[str] = []
    for name in model.model_fields:
        if name == estimate_field:
            columns.extend((name, "lb", "ub", "open_side"))
        elif name == "sequential_result":
            columns.extend(
                (
                    name,
                    "sequential_lower",
                    "sequential_upper",
                    "sequential_status",
                    "sequential_log_e",
                    "sequential_point_reason",
                    "sequential_validity_regime",
                    "sequential_alpha",
                    "sequential_components",
                )
            )
        elif name == binomial_set_field:
            columns.extend(("set_lower", "set_upper", "set_level", "set_numerical_qualification"))
        else:
            columns.append(name)
    return columns


def _append_frame_value(
    data: dict[str, list[Any]],
    name: str,
    value: Any,
    estimate_field: str | None,
    binomial_set_field: str | None,
) -> None:
    if name == "posterior_components":
        from increment._canonical import canonical_json_bytes

        data[name].append(
            None if value is None else canonical_json_bytes(value.model_dump(mode="json")).decode()
        )
    elif name == estimate_field:
        data[name].append(None if value is None else value.value)
        data["lb"].append(None if value is None else value.lb)
        data["ub"].append(None if value is None else value.ub)
        data["open_side"].append(None if value is None else value.open_side)
    elif name == "sequential_result":
        from increment.estimation.sequential_runtime import _outward

        data[name].append(None if value is None else value.model_dump_json())
        data["sequential_lower"].append(
            None if value is None else _outward(value.bounds.lower, lower=True)
        )
        data["sequential_upper"].append(
            None if value is None else _outward(value.bounds.upper, lower=False)
        )
        data["sequential_status"].append(None if value is None else value.bounds.status)
        data["sequential_log_e"].append(
            str(value.log_e) if isinstance(value, SequentialInferenceResult) else None
        )
        data["sequential_point_reason"].append(None if value is None else value.point_reason)
        data["sequential_validity_regime"].append(
            None if value is None else getattr(value, "validity_regime", "finite_sample")
        )
        data["sequential_alpha"].append(None if value is None else str(value.decision_alpha))
        data["sequential_components"].append(
            None
            if value is None or isinstance(value, SequentialInferenceResult)
            else value.bounds.model_dump_json(include={"components"})
        )
    elif name == binomial_set_field:
        data["set_lower"].append(None if value is None else value.lower)
        data["set_upper"].append(None if value is None else value.upper)
        data["set_level"].append(None if value is None else value.level)
        data["set_numerical_qualification"].append(
            None if value is None else value.numerical_qualification
        )
    elif value is None:
        data[name].append(None)
    elif isinstance(value, date) and not isinstance(value, datetime):
        data[name].append(datetime.combine(value, datetime.min.time()))
    elif isinstance(value, Mapping):
        from increment._canonical import canonical_json_bytes
        from increment.estimation.readout_types import thaw

        data[name].append(canonical_json_bytes(thaw(value)).decode())
    elif isinstance(value, BaseModel):
        data[name].append(repr(value))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        data[name].append(repr(value))
    else:
        data[name].append(value)


def to_frame[M: BaseModel](
    estimates: Sequence[M],
    model: type[M] | None = None,
    backend: Backend = "pandas",
) -> IntoDataFrame:
    """Convert a sequence of result models (:class:`LiftEstimate`,
    :class:`BreakoutEstimate`, :class:`DailyMetricValue`,
    :class:`DailyLiftEstimate`) to a native ``backend`` frame.

    Generic over pydantic's ``model_fields``: an ``Estimate``-typed field
    flattens into four columns (its name, plus ``lb``/``ub``/``open_side``);
    a ``binomial_set`` field (see :class:`~increment.estimation.results.
    BinomialConfidenceSet`) flattens into ``set_lower``/``set_upper``/
    ``set_level`` and ``set_numerical_qualification`` -- always the row's
    confidence-set bounds/level and arithmetic scope, even for a set-only row
    with no finite point (``<estimate field>`` and ``lb``/``ub`` stay ``None``
    there; these set columns are the row's ONLY confidence-set representation
    in that case; see :class:`BinomialConfidenceSet`); every other field
    passes through as a scalar column in declaration order. Most callers
    should use ``results.to_frame()`` on a pipeline's own result rather
    than calling this function directly.

    Parameters
    ----------
    estimates : Sequence[M]
        Any sequence of one supported result model. May be empty.
    model : type[M] | None
        Which model *estimates* holds. Required when *estimates* is
        empty, since an empty sequence carries no runtime type trace.
    backend : Backend
        Native dataframe backend to build: ``"pandas"``, ``"polars"``, or
        ``"pyarrow"``.

    Returns
    -------
    IntoDataFrame
        One row per estimate, columns in the model's field order, with
        the ``Estimate``-typed field expanded to
        ``<field name>``/``lb``/``ub``/``open_side`` and a
        ``binomial_set`` field expanded to ``set_lower``/``set_upper``/
        ``set_level``/``set_numerical_qualification``. ``open_side`` is
        ``"lower"``/``"upper"`` for a genuinely unbounded one-sided endpoint,
        and ``None`` both for a closed interval (``lb``/``ub`` both set) and for
        one (``lb``/``ub``/``value`` all ``None``) -- distinguish the
        two by whether ``<field name>`` (the point estimate) is
        ``None``.
    """
    if estimates:
        inferred = type(estimates[0])
        if model is None:
            model = inferred
        elif not isinstance(estimates[0], model):
            _refuse("breakout.to_frame_model", model=model.__name__, inferred=inferred.__name__)
    elif model is None:
        _refuse("breakout.to_frame_infer")

    estimate_field = next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_estimate_annotation(info.annotation)
        ),
        None,
    )
    binomial_set_field = next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_optional_binomial_set(info.annotation)
        ),
        None,
    )
    columns = _frame_columns(model, estimate_field, binomial_set_field)
    data: dict[str, list[Any]] = {c: [] for c in columns}
    for est in estimates:
        for name in model.model_fields:
            _append_frame_value(
                data,
                name,
                getattr(est, name),
                estimate_field,
                binomial_set_field,
            )

    schema = {
        name: (
            nw.Float64()
            if name
            in (
                estimate_field,
                "lb",
                "ub",
                "set_lower",
                "set_upper",
                "set_level",
                "sequential_lower",
                "sequential_upper",
            )
            else nw.String()
            if name
            in (
                "open_side",
                "sequential_status",
                "sequential_log_e",
                "sequential_point_reason",
                "sequential_validity_regime",
                "sequential_alpha",
                "sequential_components",
                "posterior_components",
                "set_numerical_qualification",
            )
            else _scalar_dtype(model.model_fields[name].annotation)
        )
        for name in columns
    }
    if data.get("ds"):
        non_null = [day for day in data["ds"] if day is not None]
        if non_null:
            day_type = (
                datetime
                if isinstance(non_null[0], datetime)
                else float
                if any(isinstance(day, float) for day in non_null)
                else type(non_null[0])
            )
            # A day type mixed with nulls needs a nullable dtype (an
            # optional int maps to Float64, not numpy-backed Int64,
            # which cannot hold a null cell).
            has_nulls = len(non_null) < len(data["ds"])
            schema["ds"] = _scalar_dtype(day_type | None if has_nulls else day_type)
        # else: every value is None -- keep the annotation-derived Datetime schema.
    frame = nw.from_dict(data, schema=schema, backend=backend).to_native()

    nullable_strings = [
        name
        for name, info in model.model_fields.items()
        if name != estimate_field
        and name != binomial_set_field
        and (_is_optional_string(info.annotation) or _is_optional_model(info.annotation))
        and schema[name] == nw.String()
    ]
    if estimate_field is not None:
        nullable_strings.append("open_side")
    if "sequential_result" in model.model_fields:
        nullable_strings.extend(
            (
                "sequential_status",
                "sequential_log_e",
                "sequential_point_reason",
                "sequential_validity_regime",
                "sequential_alpha",
                "sequential_components",
            )
        )
    if binomial_set_field is not None:
        nullable_strings.append("set_numerical_qualification")
    if backend == "pandas" and nullable_strings:
        # Rewritten after construction: narwhals may otherwise infer a
        # floating object column for all-null optional strings.  Pandas'
        # nullable StringDtype retains real strings and native <NA> values.
        import pandas as pd

        for name in nullable_strings:
            frame[name] = pd.array(data[name], dtype="string")

    nullable_bools = [
        name for name, info in model.model_fields.items() if _is_optional_bool(info.annotation)
    ]
    if backend == "pandas" and nullable_bools:
        # Rewritten after construction: narwhals resolves nw.Boolean() to
        # whichever pandas dtype the values happen to allow.
        import pandas as pd

        for name in nullable_bools:
            frame[name] = pd.array(data[name], dtype="boolean")
    nullable_floats = [
        name
        for name, info in model.model_fields.items()
        if name != estimate_field
        and name != binomial_set_field
        and _is_optional_float(info.annotation)
        and schema[name].is_float()
    ]
    if binomial_set_field is not None:
        nullable_floats.extend(("set_lower", "set_upper", "set_level"))
    if backend == "pandas" and nullable_floats:
        # Pandas' nullable Float64 dtype preserves unavailable values as
        # <NA>, rather than conflating them with a floating-point NaN.
        import pandas as pd

        for name in nullable_floats:
            frame[name] = pd.array(data[name], dtype="Float64")

    return frame
