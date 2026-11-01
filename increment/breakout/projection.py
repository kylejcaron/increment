"""Typed estimate-to-frame projection, separate from estimation policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel

from increment._literals import Backend, default_frame_backend
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation.results import BinomialConfidenceSet, Estimate
from increment.estimation.sequential_result import SequentialInferenceResult

if TYPE_CHECKING:
    from increment._interpretation import ReadoutInterpretation
    from increment.estimation.readout_types import ReadoutMetadata

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "breakout.to_frame_model": "to_frame(): model={model} does not match estimates[0]'s actual type {inferred}",
        "breakout.to_frame_infer": "to_frame(): cannot infer the schema from an empty sequence -- pass model=<the result model class> explicitly, or call <Model>Estimates(estimates).to_frame() instead, which always knows its own model even when empty.",
        "results.frame.unknown_columns": RefusalSpec(
            "results.frame.unknown_columns",
            InvalidRequestError,
            template="to_frame(): unknown column(s) {unknown!r}; available columns are {available!r}",
        ),
        "results.frame.empty_columns": RefusalSpec(
            "results.frame.empty_columns",
            InvalidRequestError,
            template="to_frame(columns=...) requires at least one projection column",
        ),
    },
)
_refuse = raiser(_REFUSALS)


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


def _nullable_bool_fields(
    model: type[BaseModel], estimate_field: str | None, columns: Sequence[str]
) -> list[str]:
    fields = [
        name for name, info in model.model_fields.items() if _is_optional_bool(info.annotation)
    ]
    if estimate_field is not None and "stat_sig" in columns:
        fields.append("stat_sig")
    return fields


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
    decision_row = model.__name__ in {"LiftEstimate", "BreakoutEstimate", "DailyLiftEstimate"}
    arm_fact_fields = {"n_control", "n_treat", "mean_control", "mean_treat"}
    for name in model.model_fields:
        if name in arm_fact_fields and not decision_row:
            continue
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
    if decision_row:
        columns.extend(("stat_sig", "p_value", "p_value_reason_code"))
    return columns


def _append_frame_value(
    data: dict[str, list[Any]],
    name: str,
    value: Any,
    estimate_field: str | None,
    binomial_set_field: str | None,
    binomial_set: BinomialConfidenceSet | None,
) -> None:
    if name == "posterior_components":
        from increment._canonical import canonical_json_bytes

        data[name].append(
            None if value is None else canonical_json_bytes(value.model_dump(mode="json")).decode()
        )
    elif name == estimate_field:
        data[name].append(None if value is None else value.value)
        data["lb"].append(
            value.lb
            if value is not None
            else (None if binomial_set is None else binomial_set.lower)
        )
        data["ub"].append(
            value.ub
            if value is not None
            else (None if binomial_set is None else binomial_set.upper)
        )
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


def _append_estimates_frame_values(
    data: dict[str, list[Any]],
    estimates: Sequence[BaseModel],
    model: type[BaseModel],
    estimate_field: str | None,
    binomial_set_field: str | None,
) -> None:
    for estimate in estimates:
        for name in model.model_fields:
            if name not in data and name not in (estimate_field, binomial_set_field):
                continue
            if name in {"n_control", "n_treat", "mean_control", "mean_treat"} and (
                model.__name__ not in {"LiftEstimate", "BreakoutEstimate", "DailyLiftEstimate"}
            ):
                continue
            _append_frame_value(
                data,
                name,
                getattr(estimate, name),
                estimate_field,
                binomial_set_field,
                getattr(estimate, binomial_set_field) if binomial_set_field is not None else None,
            )


def _append_decision_frame_values(
    data: dict[str, list[Any]], estimates: Sequence[BaseModel]
) -> None:
    from increment.estimation.results import row_p_value, row_stat_sig

    for estimate in estimates:
        if not callable(getattr(estimate, "stat_sig", None)):
            data["stat_sig"].append(None)
            data["p_value"].append(None)
            data["p_value_reason_code"].append(None)
            continue
        try:
            stat_sig = row_stat_sig(cast(Any, estimate))
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code != "readout.legacy.sampling_unreconstructible":
                raise
            data["stat_sig"].append(None)
            data["p_value"].append(None)
            data["p_value_reason_code"].append(code)
            continue
        data["stat_sig"].append(stat_sig)
        try:
            data["p_value"].append(row_p_value(cast(Any, estimate)))
            data["p_value_reason_code"].append(None)
        except CodedError as exc:
            data["p_value"].append(None)
            data["p_value_reason_code"].append(exc.code)


def _estimate_field(model: type[BaseModel]) -> str | None:
    return next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_estimate_annotation(info.annotation)
        ),
        None,
    )


def _binomial_set_field(model: type[BaseModel]) -> str | None:
    return next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_optional_binomial_set(info.annotation)
        ),
        None,
    )


def _projection_columns(
    model: type[BaseModel],
    estimates: Sequence[BaseModel],
    estimate_field: str | None,
    binomial_set_field: str | None,
) -> tuple[str, ...]:
    columns = _frame_columns(model, estimate_field, binomial_set_field)
    if "allocation_alpha" in columns and not any(
        getattr(estimate, "allocation_alpha", None) is not None for estimate in estimates
    ):
        return tuple(name for name in columns if name != "allocation_alpha")
    return tuple(columns)


def projection_columns(model: type[BaseModel], estimates: Sequence[BaseModel]) -> tuple[str, ...]:
    """Return the typed projection columns of ``estimates`` in canonical order."""
    return _projection_columns(model, estimates, _estimate_field(model), _binomial_set_field(model))


def select_columns(columns: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    """Return requested projection columns in canonical order, refusing unknown names."""
    requested = tuple(columns)
    unknown = tuple(name for name in requested if name not in available)
    if unknown:
        _refuse("results.frame.unknown_columns", unknown=unknown, available=tuple(available))
    if not requested:
        _refuse("results.frame.empty_columns")
    requested_set = frozenset(requested)
    return tuple(name for name in available if name in requested_set)


def to_frame[M: BaseModel](
    estimates: Sequence[M],
    model: type[M] | None = None,
    backend: Backend | None = None,
    columns: Sequence[str] | None = None,
) -> IntoDataFrame:
    """Convert a sequence of result models (:class:`LiftEstimate`,
    :class:`BreakoutEstimate`, :class:`DailyMetricValue`,
    :class:`DailyLiftEstimate`) to a native ``backend`` frame.

    Generic over pydantic's ``model_fields``: an ``Estimate``-typed field
    flattens into four columns (its name, plus ``lb``/``ub``/``open_side``);
    a ``binomial_set`` field (see :class:`~increment.estimation.results.
    BinomialConfidenceSet`) flattens into ``set_lower``/``set_upper``/
    ``set_level`` and ``set_numerical_qualification`` -- always the row's
    confidence-set bounds/level and arithmetic scope; set-only rows may expose
    their binomial bounds in ``lb``/``ub`` despite having no point estimate.
    These are also recorded in the ``set_*`` columns; every other field
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
    backend : Backend | None
        Native dataframe backend to build. When omitted, pandas is preferred,
        then polars, then pyarrow. Collections created from dataframe input
        default to that input's backend.

    columns : Sequence[str] | None
        Select known projection columns in canonical order. ``None`` keeps
        every projection column.

    Returns
    -------
    IntoDataFrame
        One row per estimate, columns in the model's field order, with
        the ``Estimate``-typed field expanded to
        ``<field name>``/``lb``/``ub``/``open_side`` and a
        ``binomial_set`` field expanded to ``set_lower``/``set_upper``/
        ``set_level``/``set_numerical_qualification``. ``open_side`` is
        ``"lower"``/``"upper"`` only for a point-backed genuinely unbounded
        endpoint; ``None`` for a closed interval and whenever there is no
        point estimate, including set-only rows (whose bounds are in ``lb``/``ub``).
    """
    backend = default_frame_backend(backend)
    if estimates:
        inferred = type(estimates[0])
        if model is None:
            model = inferred
        elif not isinstance(estimates[0], model):
            _refuse("breakout.to_frame_model", model=model.__name__, inferred=inferred.__name__)
    elif model is None:
        _refuse("breakout.to_frame_infer")

    estimate_field = _estimate_field(model)
    binomial_set_field = _binomial_set_field(model)
    available = _projection_columns(model, estimates, estimate_field, binomial_set_field)
    selected = available if columns is None else select_columns(columns, available)
    columns = list(available)
    data: dict[str, list[Any]] = {name: [] for name in columns}
    _append_estimates_frame_values(
        data,
        estimates,
        model,
        estimate_field,
        binomial_set_field,
    )

    if model.__name__ in {"LiftEstimate", "BreakoutEstimate", "DailyLiftEstimate"}:
        _append_decision_frame_values(data, estimates)
    derived_dtypes = {
        "stat_sig": nw.Boolean(),
        "p_value": nw.Float64(),
        "p_value_reason_code": nw.String(),
    }
    schema = {
        name: (
            derived_dtypes[name]
            if name in derived_dtypes
            else nw.Float64()
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
        and name in schema
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

    nullable_bools = _nullable_bool_fields(model, estimate_field, columns)
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
        and name in schema
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

    return nw.from_native(frame, eager_only=True).select(selected).to_native()


_EVIDENCE_STRING_COLUMNS = frozenset(
    {
        "source_snapshot_id",
        "analysis_population",
        "integrity_status",
        "integrity_code",
        "integrity_context",
        "roster_source",
        "posterior_reason_code",
        "posterior_reason_context",
        "failure_code",
        "failure_context",
        "failure_message",
        "unavailable_cells",
        "hidden_cells",
        "advisories",
        "multiplicity_status",
        "retained_family",
    }
)
_EVIDENCE_BOOLEAN_COLUMNS = frozenset(
    {
        "decision_scope_complete",
        "roster_complete",
        "sampling_available",
        "point_available",
        "posterior_available",
        "view_partial",
    }
)
_EVIDENCE_COUNT_COLUMNS = frozenset(
    {
        "required_cell_count",
        "available_cell_count",
        "source_cell_count",
        "unavailable_cell_count",
        "hidden_cell_count",
    }
)


def with_readout_evidence(
    native: IntoDataFrame,
    rows: Sequence[Any],
    metadata: ReadoutMetadata | None,
    *,
    backend: Backend,
    interpretation: ReadoutInterpretation | None = None,
) -> IntoDataFrame:
    """Append each row's portable qualification columns and the view's partial state.

    ``interpretation`` defaults to the one of ``rows`` alone. A caller projecting part
    of a larger result passes the whole result's, so the other rows are not hidden.
    """
    from increment._interpretation import interpret_readout, readout_evidence_columns

    frame = nw.from_native(native, eager_only=True)
    if interpretation is None:
        interpretation = interpret_readout(rows, metadata)
    evidence = [
        readout_evidence_columns(
            interpretation, row, source_snapshot_id=getattr(row, "source_snapshot_id", None)
        )
        for row in rows
    ]
    if metadata is None:
        # Without scope metadata the row's own stored availability is the
        # authority: an unknown value on a prior-bound row stays unknown
        # rather than being inferred from the presence of a point.
        for row, row_evidence in zip(rows, evidence, strict=True):
            stored = getattr(row, "sampling_available", None)
            if stored is not None or getattr(row, "prior_shrunk", False):
                row_evidence["sampling_available"] = stored
    if evidence:
        evidence_columns = {name: [row[name] for row in evidence] for name in evidence[0]}
    elif interpretation.groups and interpretation.groups[0].source_cells:
        group = interpretation.groups[0]
        empty_evidence = readout_evidence_columns(
            interpretation,
            group.source_cells[0].key,
            source_snapshot_id=group.source_snapshot_id,
        )
        evidence_columns = {name: [] for name in empty_evidence}
    else:
        evidence_columns = {}
    if evidence_columns:
        frame = frame.with_columns(
            *(
                nw.new_series(
                    name,
                    values,
                    dtype=(
                        nw.String()
                        if name in _EVIDENCE_STRING_COLUMNS
                        else nw.Boolean()
                        if name in _EVIDENCE_BOOLEAN_COLUMNS
                        else nw.Int64()
                        if name in _EVIDENCE_COUNT_COLUMNS
                        else None
                    ),
                    backend=nw.get_native_namespace(frame),
                )
                for name, values in evidence_columns.items()
            )
        )
    partial = None if metadata is None else metadata.partial
    frame = frame.with_columns(nw.lit(partial).alias("view_partial"))
    result = cast(Any, frame.to_native())
    if backend == "pandas":
        import pandas as pd

        for name in _EVIDENCE_BOOLEAN_COLUMNS:
            if name in evidence_columns:
                result[name] = pd.array(evidence_columns[name], dtype="boolean")
        result["view_partial"] = pd.array([partial] * len(result), dtype="boolean")
    return result


def result_frame(
    groups: Sequence[tuple[type[BaseModel], Sequence[Any]]],
    metadata: ReadoutMetadata | None,
    *,
    backend: Backend,
    columns: Sequence[str] | None = None,
) -> IntoDataFrame:
    """Project one result's consecutive same-model row groups as one native frame.

    Each group keeps its collection's typed projection; one interpretation of every
    row supplies the qualification columns. Groups of different models share their
    common columns, and a column one model lacks is null on the other's rows.
    ``columns`` selects typed projection columns, as ``to_frame`` does.
    """
    from increment._interpretation import interpret_readout
    from increment.estimation.contrast_results import (
        CONTRAST_FRAME_COLUMNS,
        ContrastResult,
        ContrastResults,
    )

    interpretation = interpret_readout([row for _, rows in groups for row in rows], metadata)
    frames = []
    available: dict[str, None] = {}
    for model, rows in groups:
        if model is ContrastResult:
            native = ContrastResults(rows, metadata=metadata).to_frame(backend=backend)
            typed = CONTRAST_FRAME_COLUMNS
        else:
            typed = projection_columns(model, rows)
            native = with_readout_evidence(
                to_frame(rows, model=model, backend=backend),
                rows,
                metadata,
                backend=backend,
                interpretation=interpretation,
            )
        available.update(dict.fromkeys(typed))
        frames.append(nw.from_native(native, eager_only=True))
    frame = frames[0] if len(frames) == 1 else nw.concat(frames, how="diagonal")
    if columns is None:
        return frame.to_native()
    selected = [*select_columns(columns, tuple(available)), "view_partial"]
    native = cast(Any, frame.to_native())
    if backend == "pandas":
        return native.loc[:, selected]
    return frame.select(selected).to_native()
