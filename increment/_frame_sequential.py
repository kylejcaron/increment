"""Capture registered sequential observations from dataframe sources."""

from __future__ import annotations

from fractions import Fraction

import narwhals as nw

from increment.sequential_source import frame_observation_mapping, sequential_definition_id
from increment.sequential_state import (
    _observation,
    adjustment_kind,
    capture_sequential_snapshot,
    model_adjustment,
    require_public_laws,
    sequential_refuse,
    validate_sequential_transform,
)


def _reveal_order(frame, *, unit: str, exposure: str):
    """Rows in ascending exposure order, ties broken by canonical unit id.

    String exposures are ranked with the panel path's day-axis rules, which
    refuse labels whose chronological order cannot be justified.
    """
    from increment._frame_panel import _day_axis_label_order

    rank = nw.col(exposure)
    if isinstance(frame.schema[exposure], nw.String):
        order = _day_axis_label_order(frame.get_column(exposure).to_list())
        labels = sorted(order, key=order.__getitem__)
        rank = rank.replace_strict(labels, list(range(len(labels))))
    rank_name, unit_name = "__reveal_rank__", "__reveal_unit__"
    while rank_name in frame.columns or unit_name in frame.columns:
        rank_name, unit_name = f"_{rank_name}", f"_{unit_name}"
    return frame.with_columns(
        rank.alias(rank_name), nw.col(unit).cast(nw.String).alias(unit_name)
    ).sort(rank_name, unit_name)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _fixed_clip(spec):
    """Exact bounds of a fixed winsorization; percentile bounds were refused at registration."""
    config = spec.winsorization
    if config is None:
        return None, None
    return (
        _observation(config.lower_value) if config.lower_value is not None else None,
        _observation(config.upper_value) if config.upper_value is not None else None,
    )


def _outcome_value(value, spec) -> Fraction:
    """Apply the metric's outcome policy without borrowing its covariate policy."""
    if spec.missing == "zero" and _missing(value):
        value = 0
    return _observation(value)


def _covariate_value(row, spec, center, unit) -> Fraction:
    """Read one unit's pre-period covariate under the spec's missing-covariate policy.

    A predeclared adjustment imputes its registered centre, the pre-period
    estimate of ``E[X]``; a retained joint law has no centre fixed before the
    data, so a missing covariate can only be zeroed or refused.
    """
    value = row[spec.covariate]
    if not _missing(value):
        return _observation(value)
    if spec.covariate_missing == "impute" and center is not None:
        return center
    if spec.covariate_missing == "zero":
        return Fraction(0)
    sequential_refuse(
        "source.invalid",
        f"metric {spec.name!r}: unit {row[unit]!r} has no covariate value and "
        + (
            "covariate_missing='error'"
            if spec.covariate_missing == "error"
            else "the retained joint law fixes no pre-period centre to impute; declare "
            "covariate_missing='zero' or 'error'"
        ),
    )


def _joint_observation(spec, model, unit):
    """Per-unit vector in reveal order: missing policy, fixed clip, then the law's coordinates.

    Every step is fixed before the first outcome is read. A predeclared
    adjustment folds the covariate into the retained scalar; the joint laws
    append the denominator and covariate coordinates the law declares.
    """
    lower, upper = _fixed_clip(spec)

    def observe(row):
        value = _outcome_value(row[spec.y_column], spec)
        if lower is not None and value < lower:
            value = lower
        if upper is not None and value > upper:
            value = upper
        adjustment = getattr(model, "adjustment", None)
        if adjustment is not None:
            covariate = _covariate_value(row, spec, adjustment.center, unit)
            value -= adjustment.coefficient * (covariate - adjustment.center)
        vector = [value]
        if spec.denominator is not None:
            vector.append(_outcome_value(row[spec.denominator], spec))
        if model_adjustment(model) == "retained":
            vector.append(_covariate_value(row, spec, None, unit))
        return vector[0] if len(vector) == 1 else tuple(vector)

    return observe


def capture_frame_totals(source):
    """Snapshot modeled observations at construction, before mutable backing escapes."""
    inference = source.context.plan.inference
    registration = getattr(inference, "registration", None)
    if registration is None:
        return
    require_public_laws(registration.models, "sequential source capture")
    if source.context.cluster is not None:
        sequential_refuse(
            "route.unsupported",
            "clustered sequential observations are unsupported; use fixed-horizon "
            "inference on the clustered source (valid for one planned analysis, not "
            "repeated looks)",
        )
    specs = source._specs_by_name
    definitions = sequential_definition_id(
        source.context.metrics,
        source.context.design,
        transformations=source._specs,
        source_mapping=frame_observation_mapping(
            unit=source._unit,
            group=source._group,
            uptake=source._uptake,
            exposure_date=source._exposure_date,
        ),
    )
    if registration.definitions_id != definitions:
        sequential_refuse(
            "source.invalid", "frame metric definitions or transformations differ from registration"
        )
    for model in registration.models:
        if model.observable == "uptake":
            if source._uptake is None:
                sequential_refuse(
                    "source.invalid", "registered uptake requires an encouragement source"
                )
            continue
        if model.metric not in specs:
            sequential_refuse(
                "source.invalid", "registered metric is absent from frame declarations"
            )
        spec = specs[model.metric]
        validate_sequential_transform(spec)
        if adjustment_kind(registration, model.metric) is not None and spec.covariate is None:
            sequential_refuse(
                "source.invalid",
                f"metric {model.metric!r}: the registered covariate adjustment needs the "
                "frame's covariate column (MetricSpec.covariate)",
            )
    observers = {
        model.metric: _joint_observation(specs[model.metric], model, source._unit)
        for model in registration.models
        if model.observable != "uptake"
    }

    ordered = _reveal_order(source._frame, unit=source._unit, exposure=source._exposure_date)

    def records():
        for row in ordered.iter_rows(named=True):
            values = {}
            for model in registration.models:
                if model.observable == "uptake":
                    if source._uptake is None:
                        sequential_refuse(
                            "source.invalid", "compliance requires the design uptake column"
                        )
                    values[model.metric] = row[source._uptake]
                else:
                    values[model.metric] = observers[model.metric](row)
            yield {
                "unit_id": str(row[source._unit]),
                "group_id": str(row[source._group]),
                "source_identity": {
                    "unit_column": source._unit,
                    "group_column": source._group,
                    "uptake_column": source._uptake,
                },
                "values": values,
                "segments": {
                    key: str(row[column]) for key, column in sorted(source._segment_columns.items())
                },
            }

    source._sequential_snapshot = capture_sequential_snapshot(
        registration,
        records(),
        source_id=source.context.study_id,
        definitions_id=definitions,
        finalized=True,
        assignment_counts=source.unit_counts(),
    )


def _validate_panel_models(source, registration) -> None:
    """Every registered model must fit the common reveal window and the panel's inputs."""
    from increment._frame_panel import _final_maturity_day, _resolve_window_days

    metrics = {m.name: m for m in source.context.metrics}
    for model in registration.models:
        if model.observable == "uptake":
            if (
                source._uptake is None
                or source._window_days is None
                or source._window_days > registration.reveal.longest_window_days
            ):
                sequential_refuse(
                    "source.invalid", "bounded design uptake must fit the common reveal window"
                )
            continue
        metric = metrics[model.metric]
        last = _final_maturity_day(metric)
        if (
            _resolve_window_days(metric) is None
            or last is None
            or last >= registration.reveal.longest_window_days
        ):
            sequential_refuse(
                "route.unsupported",
                "panel metrics need bounded windows covered by the common joint-reveal window",
            )
        spec = source._specs_by_name[model.metric]
        validate_sequential_transform(spec)
        if adjustment_kind(registration, model.metric) is not None:
            sequential_refuse(
                "route.unsupported",
                f"metric {model.metric!r}: a unit panel carries no per-unit pre-period "
                f"covariate, so it cannot supply the registered {model.law!r} adjustment; "
                "use a unit-summary frame with MetricSpec.covariate",
            )


def capture_frame_panel(source, *, as_of, finalized, previous=None):
    """Reveal every metric only for the common longest-window unit cohort."""
    import narwhals as nw

    from increment._frame_moments import _collapse_to_unit_totals, _reduce_spec
    from increment._frame_panel import (
        _day_axis_label_order,
        _observable_end_index,
        _resolve_window_days,
        _scratch_name,
        _with_day_index,
    )

    previous = previous if previous is not None else getattr(source, "_sequential_snapshot", None)
    registration = getattr(source.context.plan.inference, "registration", None)
    if registration is None or not finalized or source._exposure is None:
        sequential_refuse(
            "source.invalid",
            "panel capture needs registration, explicit finalization and exposure anchors",
        )
    require_public_laws(registration.models, "sequential panel capture")
    definitions = sequential_definition_id(
        source.context.metrics,
        source.context.design,
        transformations=source._specs,
        source_mapping=source._sequential_mapping,
    )
    if registration.definitions_id != definitions:
        sequential_refuse("source.invalid", "panel definitions differ from registration")
    metrics = {m.name: m for m in source.context.metrics}
    _validate_panel_models(source, registration)
    if previous is not None and previous.registration != registration:
        sequential_refuse("continuation.rewrite", "registration changed before panel access")
    dimensions = tuple(sorted({k for c in registration.roster for k, _ in c.segment}))
    labels = source._sparse_panel.get_column("ds").unique().to_list()
    order = _day_axis_label_order([*labels, as_of])
    retained = [label for label in labels if order[label] <= order[as_of]]
    keys = ["unit_id", "group_id", *dimensions]
    observable = _observable_end_index(source._exposure, as_of)
    maturity = max(0, registration.reveal.longest_window_days - 1)
    kept = (
        source._day_identity()
        .join(observable, on="unit_id", how="left")
        .filter(nw.col("__observable_days__") >= maturity)
        .select(*keys)
    )
    retained_panel = source._sparse_panel.filter(nw.col("ds").is_in(retained))
    indexed, day_index = _with_day_index(retained_panel, source._exposure)
    joint = kept
    columns = {}
    for i, model in enumerate(registration.models):
        if model.observable == "uptake":
            uptake_totals = _collapse_to_unit_totals(
                indexed,
                (),
                uptake=source._uptake,
                window_days=source._window_days,
                first_exposure=source._first_exposure,
                by=dimensions,
            ).rename({source._uptake: "__y__"})
            totals = kept.join(uptake_totals, on=keys, how="left").with_columns(
                nw.col("__y__").fill_null(0.0)
            )
        else:
            metric, spec = metrics[model.metric], source._specs_by_name[model.metric]
            right = _resolve_window_days(metric)
            bounded = indexed.filter((nw.col(day_index) >= 0) & (nw.col(day_index) < right))
            totals = _reduce_spec(
                bounded,
                spec,
                getattr(metric, "band", None),
                units_source=kept,
                by=dimensions,
                day_index=day_index,
            )
        rename = {"__y__": f"v{i}"}
        if model.law in ("gaussian_ratio", "ratio_mean"):
            rename["__y_den__"] = f"d{i}"
        keys = ["unit_id", "group_id", *dimensions]
        part = totals.select(*keys, *rename).rename(rename)
        joint = joint.join(part, on=keys, how="left")
        columns[model.metric] = tuple(rename.values())
    exposure_name = _scratch_name(joint, "__exposure__")
    joint = joint.join(
        source._exposure.select("unit_id", nw.col("__exposure__").alias(exposure_name)),
        on="unit_id",
        how="left",
    )
    joint = _reveal_order(joint, unit="unit_id", exposure=exposure_name)

    clips = {
        model.metric: (
            source._specs_by_name[model.metric],
            *_fixed_clip(source._specs_by_name[model.metric]),
        )
        for model in registration.models
        if model.observable != "uptake"
    }

    def clipped(metric, values):
        configured = clips.get(metric)
        if configured is None:
            return tuple(values)
        spec, lower, upper = configured
        head, *rest = (_outcome_value(value, spec) for value in values)
        if lower is not None and head < lower:
            head = lower
        if upper is not None and head > upper:
            head = upper
        return (head, *rest)

    def records():
        for row in joint.iter_rows(named=True):
            yield {
                "unit_id": str(row["unit_id"]),
                "group_id": str(row["group_id"]),
                "source_identity": {"first_exposure": str(row[exposure_name])},
                "segments": {k: str(row[k]) for k in dimensions},
                "values": {
                    m: clipped(m, tuple(row[c] for c in cols)) for m, cols in columns.items()
                },
            }

    snapshot = capture_sequential_snapshot(
        registration,
        records(),
        source_id=source.context.study_id,
        definitions_id=definitions,
        finalized=finalized,
        previous=previous,
        reveal_cursor=as_of,
        assignment_counts=source.unit_counts(),
    )
    source._sequential_snapshot = snapshot
    return snapshot
