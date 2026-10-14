"""Ibis longitudinal reduction retaining per-independent-grain regression scores.

Only checks, encoding metadata and cluster moments leave the warehouse. Raw
panel outcomes are never fetched. Relations must share an executing backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import numpy as np

from increment._longitudinal_design import regression_columns, require_basis_size, validate_columns
from increment.errors import refuse
from increment.estimation.cate import Covariate
from increment.estimation.longitudinal import LongitudinalCluster, LongitudinalSummary
from increment.semantics.longitudinal import LONGITUDINAL_OBSERVATION, LongitudinalDesign, invalid


def _has_rows(table: ir.Table, predicate: Any) -> bool:
    return bool(table.filter(predicate).limit(1).count().execute())


def _finite(column: ir.Value) -> ir.BooleanValue:
    return (
        column.notnull()
        & ~cast(ir.FloatingValue, column.cast("float64")).isnan()
        & ~cast(ir.FloatingValue, column.cast("float64")).isinf()
    )


def _admit_relations(
    outcomes: ir.Table,
    roster: ir.Table,
    *,
    design: LongitudinalDesign,
    unit: str,
    group: str,
    cohort: str,
    exposure: str,
    time: str,
    outcome: str,
    cluster: str | None,
    covariates: Sequence[Covariate],
) -> tuple[ir.Table, ir.Table]:
    roster_names = [
        unit,
        group,
        cohort,
        exposure,
        *([cluster] if cluster else []),
        *(c.name for c in covariates),
    ]
    if set(roster_names) - set(roster.columns) or {unit, time, outcome} - set(outcomes.columns):
        invalid("Required longitudinal columns are missing.")
    for t, name in [(roster, unit), (outcomes, unit), *([(roster, cluster)] if cluster else [])]:
        if not (t[name].type().is_integer() or t[name].type().is_string()):
            invalid("Warehouse longitudinal identities must be integer or string columns.")
    if roster[unit].type() != outcomes[unit].type():
        invalid("Roster and outcome unit identities must use the same type.")
    for t, name in ((roster, group), (roster, cohort)):
        if not t[name].type().is_string():
            invalid("Arm and cohort columns must be strings.")
    integer_time = type(design.cohorts[0].exposure) is int
    for t, name in ((roster, exposure), (outcomes, time)):
        typ = t[name].type()
        if (integer_time and not typ.is_integer()) or (
            not integer_time and (not typ.is_timestamp() or typ.timezone is not None)
        ):
            invalid(
                "Observation/exposure columns must match integer or naive UTC timestamp schedules."
            )
    for t, name in [
        (outcomes, outcome),
        *[(roster, c.name) for c in covariates if c.kind == "continuous"],
    ]:
        if not t[name].type().is_numeric() or t[name].type().is_boolean():
            invalid("Outcomes and continuous covariates require numeric columns.", column=name)
        if _has_rows(t, ~_finite(t[name])):
            invalid(
                "Outcomes and continuous covariates must be finite and nonmissing.", column=name
            )
    for c in covariates:
        if c.kind == "categorical" and not roster[c.name].type().is_string():
            invalid(
                "Categorical longitudinal covariates require nonmissing strings.", column=c.name
            )
    if any(_has_rows(roster, roster[n].isnull()) for n in roster_names):
        invalid("Assigned roster columns cannot be missing.")
    r = roster.select(
        u=roster[unit],
        arm=roster[group],
        c=roster[cohort],
        e=roster[exposure],
        g=roster[cluster] if cluster else roster[unit],
        **{f"z{i}": roster[c.name] for i, c in enumerate(covariates)},
    )
    o = outcomes.select(u=outcomes[unit], t=outcomes[time], y=outcomes[outcome].cast("float64"))
    if int(cast(Any, r.count().execute())) == 0 or _has_rows(
        r.group_by("u").aggregate(n=r.count()), ibis._.n != 1
    ):
        invalid("Assigned roster requires exactly one row per unit and cannot be empty.")
    if _has_rows(r, ~r.arm.isin([design.control_group, design.treatment_group])) or _has_rows(
        r, ~r.c.isin([c.name for c in design.cohorts])
    ):
        invalid("Assigned roster has an undeclared arm or cohort.")
    if _has_rows(r.group_by("g").aggregate(n=r.arm.nunique()), ibis._.n != 1):
        invalid("Every randomization cluster must have one pure randomized arm.")
    valid_exposure = ibis.or_(*((r.c == c.name) & (r.e == c.exposure) for c in design.cohorts))
    if _has_rows(r, ~valid_exposure):
        invalid("Roster exposure must match its predeclared cohort in both arms.")
    return r, o


def _changes(r: ir.Table, o: ir.Table, design: LongitudinalDesign) -> ir.Table:
    if int(o.anti_join(r, "u").count().execute()) or _has_rows(o, o.t.isnull() | o.u.isnull()):
        refuse(
            LONGITUDINAL_OBSERVATION,
            message="Outcomes contain a missing identity/time or a unit outside the assigned roster.",
        )
    duplicates = o.group_by("u", "t").aggregate(n=o.count())
    if _has_rows(duplicates, duplicates.n != 1):
        refuse(LONGITUDINAL_OBSERVATION, message="Each unit/time cell must occur exactly once.")
    joined = r.join(o, "u")
    valid = ibis.or_(
        *(
            (joined.c == c.name) & joined.t.isin([*c.pre_periods, *c.post_periods])
            for c in design.cohorts
        )
    )
    if _has_rows(joined, ~valid):
        refuse(
            LONGITUDINAL_OBSERVATION, message="Outcomes contain an unscheduled observation time."
        )
    counts = o.group_by("u").aggregate(observed=o.count())
    checked = r.left_join(counts, "u")
    expected = ibis.cases(
        *[(checked.c == c.name, len(c.pre_periods) + len(c.post_periods)) for c in design.cohorts],
        else_=0,
    )
    if _has_rows(checked, checked.observed.fill_null(0) != expected):
        refuse(
            LONGITUDINAL_OBSERVATION,
            message="Missing scheduled observations; preserve the assigned roster and repair the feed, not complete-case selection.",
        )
    first_pre = ibis.or_(
        *((joined.c == c.name) & (joined.t == c.pre_periods[0]) for c in design.cohorts)
    )
    anchor = joined.filter(first_pre).select("u", anchor=joined.y)
    with_anchor = joined.join(anchor, "u")
    weights = ibis.cases(
        *[
            ((with_anchor.c == c.name) & (with_anchor.t == period), weight)
            for c in design.cohorts
            for period, weight in [
                *[(t, -1 / len(c.pre_periods)) for t in c.pre_periods],
                *zip(c.post_periods, c.post_weights, strict=True),
            ]
        ],
        else_=0.0,
    )
    changes = with_anchor.group_by("u").aggregate(
        change=((with_anchor.y - with_anchor.anchor) * weights).sum()
    )
    return r.join(changes, "u")


def _basis(
    table: ir.Table, covariates: Sequence[Covariate], design: LongitudinalDesign
) -> tuple[ir.Table, tuple[str, ...]]:
    names: list[str] = []
    for i, c in enumerate(covariates):
        col = table[f"z{i}"]
        if c.kind == "continuous":
            require_basis_size(design, len(names) + 1)
            origin = float(cast(Any, col.min().execute()))
            centered = col.cast("float64") - origin
            moments = (
                table.aggregate(mean=centered.mean(), sd=centered.std(how="sample"))
                .execute()
                .iloc[0]
            )
            if not np.isfinite([moments["mean"], moments["sd"]]).all() or not moments["sd"] > 0:
                invalid("Continuous covariates must have finite nonzero variation.", column=c.name)
            table = table.mutate(
                **{f"b{len(names)}": (centered - float(moments["mean"])) / float(moments["sd"])}
            )
            names.append(c.name)
        else:
            level_count = int(cast(Any, col.nunique().execute()))
            require_basis_size(design, len(names) + level_count - 1)
            levels = table.group_by(f"z{i}").aggregate(n=table.count()).execute()
            ordered = sorted(
                levels.itertuples(index=False, name=None), key=lambda row: (-row[1], row[0])
            )
            if len(ordered) < 2:
                invalid("Categorical covariates require at least two levels.", column=c.name)
            for level, _ in ordered[1:]:
                table = table.mutate(
                    **{f"b{len(names)}": (table[f"z{i}"] == level).cast("float64")}
                )
                names.append(f"{c.name}={level}")
    return table, tuple(names)


def _reduce(
    table: ir.Table, design: LongitudinalDesign, names: tuple[str, ...]
) -> LongitudinalSummary:
    columns = regression_columns(design, names)
    cohort_counts = tuple(
        int(cast(Any, table.filter(table.c == c.name).count().execute())) for c in design.cohorts
    )
    if not all(cohort_counts):
        invalid("Every declared cohort must occur in the assigned roster.")
    # Deterministic cohort-common change origin; no unit outcome rows are fetched.
    origins = table.group_by("c", "arm").aggregate(origin=table.change.min())
    origin_rows = origins.execute()
    response_origins = tuple(
        float(
            origin_rows.loc[(origin_rows.c == c.name) & (origin_rows.arm == arm), "origin"].iloc[0]
        )
        if ((origin_rows.c == c.name) & (origin_rows.arm == arm)).any()
        else 0.0
        for c in design.cohorts
        for arm in (design.control_group, design.treatment_group)
    )
    means = (
        table.group_by("c").aggregate(**{f"m{i}": table[f"b{i}"].mean() for i in range(len(names))})
        if names
        else None
    )
    table = table.join(origins, ["c", "arm"])
    if means is not None:
        table = table.join(means, "c")
    table = table.mutate(change=table.change - table.origin)
    xs: dict[str, Any] = {}
    for j, c in enumerate(design.cohorts):
        for arm_index, arm in enumerate((design.control_group, design.treatment_group)):
            mask = (table.c == c.name) & (table.arm == arm)
            block = (2 * j + arm_index) * (1 + len(names))
            xs[f"x{block}"] = mask.cast("float64")
            for i in range(len(names)):
                xs[f"x{block + i + 1}"] = mask.ifelse(table[f"b{i}"] - table[f"m{i}"], 0.0)
    table = table.mutate(**xs)
    aggregates: dict[str, Any] = {"units": table.count()}
    for i in range(len(columns)):
        aggregates[f"h{i}"] = (table[f"x{i}"] * table.change).sum()
        for j in range(i, len(columns)):
            aggregates[f"a{i}_{j}"] = (table[f"x{i}"] * table[f"x{j}"]).sum()
    rows = table.group_by("g", "arm").aggregate(**aggregates).execute()
    clusters: list[LongitudinalCluster] = []
    for row in rows.to_dict("records"):
        a = np.zeros((len(columns), len(columns)))
        for i in range(len(columns)):
            for j in range(i, len(columns)):
                a[i, j] = a[j, i] = row[f"a{i}_{j}"]
        clusters.append(
            LongitudinalCluster(
                identity=str(row["g"]),
                arm=row["arm"],
                n_units=int(row["units"]),
                gram=tuple(tuple(r) for r in a),
                cross=tuple(row[f"h{i}"] for i in range(len(columns))),
            )
        )
    return LongitudinalSummary(
        design=design,
        columns=columns,
        cohort_counts=cohort_counts,
        covariates=names,
        response_origins=response_origins,
        clusters=tuple(sorted(clusters, key=_cluster_identity)),
        n_observations=sum(
            n * (len(c.pre_periods) + len(c.post_periods))
            for n, c in zip(cohort_counts, design.cohorts, strict=True)
        ),
    )


def _cluster_identity(cluster: LongitudinalCluster) -> str:
    return cluster.identity


def prepare_longitudinal_query(
    outcomes: ir.Table,
    roster: ir.Table,
    *,
    design: LongitudinalDesign,
    unit: str = "unit",
    group: str = "group",
    cohort: str = "cohort",
    exposure: str = "exposure",
    time: str = "time",
    outcome: str = "outcome",
    cluster: str | None = None,
    covariates: Sequence[Covariate] = (),
) -> LongitudinalSummary:
    """Validate and reduce complete scheduled panels on their Ibis backend.

    Use immutable snapshot relations: checks and reductions issue multiple
    queries, so concurrent mutation would invalidate the observation contract.
    """
    validate_columns(
        unit=unit,
        group=group,
        cohort=cohort,
        exposure=exposure,
        time=time,
        outcome=outcome,
        cluster=cluster,
        covariates=covariates,
        design=design,
    )
    r, o = _admit_relations(
        outcomes,
        roster,
        design=design,
        unit=unit,
        group=group,
        cohort=cohort,
        exposure=exposure,
        time=time,
        outcome=outcome,
        cluster=cluster,
        covariates=covariates,
    )
    table = _changes(r, o, design)
    table, names = _basis(table, covariates, design)
    return _reduce(table, design, names)
