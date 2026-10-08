"""Bounded joint-unit capture shared by native and immutable artifact sources."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from increment._analysis_config import effective_methods
from increment._window import resolve_window_days
from increment.semantics.sequential import BINARY_METRIC_TYPES, RATIO_LAWS
from increment.sequential_source import sequential_definition_id
from increment.sequential_state import (
    _observation,
    adjustment_kind,
    capture_sequential_snapshot,
    model_adjustment,
    require_public_laws,
    sequential_refuse,
    validate_sequential_methods,
    validate_sequential_transform,
)

if TYPE_CHECKING:
    from ibis import Table
    from ibis.backends.sql import SQLBackend


def record_batches(connection: SQLBackend, relation: Table):
    """Read bounded batches with the backend's current Arrow interface."""
    if connection.name == "duckdb":
        from ibis.backends.duckdb import Backend as DuckDBBackend

        if not isinstance(connection, DuckDBBackend):
            sequential_refuse("source.invalid", "DuckDB connection type does not match its backend")
        duckdb_relation = connection._to_duckdb_relation(relation)
        if hasattr(type(duckdb_relation), "to_arrow_reader"):
            reader = duckdb_relation.to_arrow_reader(batch_size=4096)
        else:
            reader = duckdb_relation.fetch_arrow_reader(batch_size=4096)
    else:
        reader = connection.to_pyarrow_batches(relation, chunk_size=4096)
    schema = relation.schema().to_pyarrow()
    try:
        for batch in reader:
            yield batch.cast(schema)
    finally:
        reader.close()


def validate_relational_capture(source, *, finalized, as_of, previous=None, covariate=False):
    """Reject incompatible registrations before resolving warehouse relations."""
    registration = getattr(source.context.plan.inference, "registration", None)
    if registration is None:
        sequential_refuse("source.invalid", "register the likelihood before capturing source data")
    require_public_laws(registration.models, "relational sequential capture")
    from increment.sequential_source import validate_scalar_mean_design

    validate_scalar_mean_design(registration, source.context.design)
    definitions = sequential_definition_id(
        source.context.metrics,
        source.context.design,
        source_mapping=source._sequential_observation_mapping(),
    )
    if (
        not finalized
        or definitions != registration.definitions_id
        or not isinstance(as_of, dt.date)
        or isinstance(as_of, dt.datetime)
    ):
        sequential_refuse(
            "source.invalid",
            "capture requires explicit finalization through an as_of date and matching definitions",
        )
    if source.context.cluster is not None or any(c.segment for c in registration.roster):
        sequential_refuse(
            "route.unsupported",
            "relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments",
        )
    if getattr(source.context.design, "mechanism", None) == "observational":
        sequential_refuse(
            "route.unsupported", "observational sequential causal identification is unsupported"
        )
    modeled = {model.metric for model in registration.models if model.observable == "outcome"}
    for config in source.context.configs:
        if config.metric.name not in modeled:
            continue
        validate_sequential_methods(
            registration,
            config.metric.name,
            effective_methods(config, design=source.context.design),
            prior=config.prior,
        )
        adjustment = adjustment_kind(registration, config.metric.name)
        if adjustment is not None and not covariate:
            sequential_refuse(
                "route.unsupported",
                f"metric {config.metric.name!r}: this source derives no per-unit pre-period "
                f"covariate, so it cannot supply the registered {adjustment} adjustment; "
                "declare n_pre_periods > 0 on the experiment (the covariate is the metric's "
                "own pre-period total, a ratio's numerator) or use a unit-summary frame with "
                "MetricSpec.covariate",
            )
    metrics = {m.name: m for m in source.context.metrics}
    for model in registration.models:
        if model.observable == "uptake":
            if getattr(source.context.design, "mechanism", None) != "encouragement":
                sequential_refuse("source.invalid", "uptake requires an encouragement design")
            window = source.context.design.uptake.window_days
        else:
            metric = metrics.get(model.metric)
            if metric is None:
                sequential_refuse(
                    "source.invalid", "registered metric is absent from the source catalog"
                )
            validate_sequential_transform(metric)
            if (metric.type == "ratio") != (model.law in ("gaussian_ratio", *RATIO_LAWS)):
                sequential_refuse(
                    "source.invalid", "ratio observations require a genuine joint sampling model"
                )
            if metric.type in BINARY_METRIC_TYPES and model.law not in (
                "bernoulli",
                "scalar_mean",
                "adjusted_mean",
            ):
                sequential_refuse(
                    "source.invalid",
                    "binary outcomes require Bernoulli or scalar mean registration",
                )
            window = resolve_window_days(metric)
        if window is None or window > registration.reveal.longest_window_days:
            sequential_refuse(
                "route.unsupported",
                "all observation windows must be bounded by the common registered reveal window",
            )
    if previous is not None and previous.registration != registration:
        sequential_refuse("continuation.rewrite", "registration changed before source access")
    return registration, definitions


def capture_relations(
    source,
    relation_for,
    uptake_relation,
    cohort,
    batches,
    *,
    recipe_id,
    finalized,
    as_of,
    previous=None,
    covariate=False,
    assignment_counts=None,
):
    """Reveal a common finalized cohort and accumulate one bounded Arrow batch.

    Cohort order is exposure time then unit identity. Every modeled outcome must
    be present; per-metric deletion cannot silently change the joint filtration.
    With ``covariate`` the unit relation carries ``x``, the same zero-filled
    pre-period total fixed-horizon CUPED reads: a retained adjustment appends
    it to the joint vector and a predeclared one folds it into the scalar
    exactly as frame capture does.
    """
    registration, definitions = validate_relational_capture(
        source, finalized=finalized, as_of=as_of, previous=previous, covariate=covariate
    )
    metrics = {m.name: m for m in source.context.metrics}
    joint = cohort.select(
        unit_id=cohort.unit_id.cast("string"),
        group_id=cohort.group_id.cast("string"),
        reveal_order=cohort.first_exposure_ts.cast("string"),
    )
    value_columns = {}
    predeclared = {}
    for index, model in enumerate(registration.models):
        compliance = model.observable == "uptake"
        relation = uptake_relation() if compliance else relation_for(metrics[model.metric])
        columns = {f"v{index}": relation["d" if compliance else "y"]}
        if model.law in ("gaussian_ratio", *RATIO_LAWS):
            columns[f"d{index}"] = relation["y_den"]
        adjustment = None if compliance else model_adjustment(model)
        if adjustment is not None:
            columns[f"x{index}"] = relation["x"]
        if adjustment == "predeclared":
            predeclared[model.metric] = model.adjustment
        selected = relation.select(
            unit_id=relation.unit_id.cast("string"),
            group_id=relation.group_id.cast("string"),
            **columns,
        )
        value_columns[model.metric] = tuple(columns)
        merged = joint.left_join(selected, ["unit_id", "group_id"])
        joint = merged.select(
            *[joint[name] for name in joint.columns], **{name: selected[name] for name in columns}
        )

    def values(row):
        observed = {}
        for metric, cols in value_columns.items():
            vector = tuple(row[col] for col in cols)
            adjustment = predeclared.get(metric)
            if adjustment is not None:
                outcome, covariate_value = (_observation(v) for v in vector)
                vector = (outcome - adjustment.coefficient * (covariate_value - adjustment.center),)
            observed[metric] = vector
        return observed

    def records():
        for batch in batches(joint.order_by("reveal_order", "unit_id")):
            for row in batch.to_pylist():
                yield {
                    "unit_id": row["unit_id"],
                    "group_id": row["group_id"],
                    "source_identity": {
                        "first_exposure": row["reveal_order"],
                        "source_recipe": recipe_id,
                    },
                    "values": values(row),
                }

    snapshot = capture_sequential_snapshot(
        registration,
        records(),
        source_id=source.context.study_id,
        definitions_id=definitions,
        finalized=finalized,
        previous=previous,
        reveal_cursor=as_of,
        assignment_counts=assignment_counts,
    )
    source._sequential_snapshot = snapshot
    return snapshot
