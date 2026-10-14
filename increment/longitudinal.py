"""Eager panel ingress for fixed-horizon randomized longitudinal estimation.

Supply an authoritative assigned roster separately from scheduled outcomes.
Every declared cell is required; the adapter never selects complete cases.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np

from increment._identity import canonical_id_strings
from increment._longitudinal_design import regression_columns, validate_columns
from increment.errors import InvalidRequestError, refuse
from increment.estimation.cate import Covariate, DesignSpec
from increment.estimation.longitudinal import (
    LongitudinalCluster,
    LongitudinalResult,
    LongitudinalSummary,
    fit_longitudinal,
)
from increment.semantics.longitudinal import (
    LONGITUDINAL_OBSERVATION,
    LongitudinalCohort,
    LongitudinalDesign,
    invalid,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
__all__ = [
    "LongitudinalCohort",
    "LongitudinalDesign",
    "LongitudinalSummary",
    "LongitudinalResult",
    "prepare_longitudinal_panel",
    "fit_longitudinal",
]


def _columns(frame: IntoDataFrame, names: Sequence[str]) -> dict[str, np.ndarray]:
    df = nw.from_native(frame, eager_only=True)
    if missing := set(names) - set(df.columns):
        invalid("Required longitudinal columns are missing.", columns=tuple(sorted(missing)))
    return {name: np.asarray(df[name].to_list(), dtype=object) for name in names}


def _finite(values: np.ndarray, *, column: str) -> np.ndarray:
    if any(v is None or isinstance(v, (bool, str)) for v in values):
        invalid(
            "Outcomes and continuous covariates require nonmissing numeric values.", column=column
        )
    try:
        out = values.astype(float)
    except (ValueError, TypeError):
        invalid("Outcomes and continuous covariates must be numeric.", column=column)
    if not np.isfinite(out).all():
        invalid("Outcomes and continuous covariates must be finite.", column=column)
    return out


def _time_matches(value: object, expected: object) -> bool:
    if type(expected) is int:
        return type(value) is int
    return (
        isinstance(value, dt.datetime)
        and value.tzinfo is None
        and not getattr(value, "nanosecond", 0)
    )


def _identities(values: np.ndarray, *, column: str, observation: bool = False) -> np.ndarray:
    try:
        return canonical_id_strings(values, what=column)
    except InvalidRequestError:
        if observation:
            refuse(
                LONGITUDINAL_OBSERVATION,
                message="Outcomes require nonmissing, noncolliding unit identities.",
            )
        invalid("Roster identities must be nonmissing and noncolliding.", column=column)


def _unit_changes(
    observations: dict[str, np.ndarray],
    roster: dict[str, np.ndarray],
    *,
    design: LongitudinalDesign,
    unit: str,
    cohort: str,
    time: str,
    outcome: str,
) -> np.ndarray:
    ids = _identities(observations[unit], column=unit, observation=True)
    roster_ids = _identities(roster[unit], column=unit)
    try:
        canonical_id_strings(np.concatenate([roster[unit], observations[unit]]), what=unit)
    except InvalidRequestError:
        invalid("Roster and outcome unit identities must use consistent native types.")
    index = {u: i for i, u in enumerate(roster_ids)}
    schedule = {c.name: c for c in design.cohorts}
    seen: dict[tuple[str, object], float] = {}
    y = _finite(observations[outcome], column=outcome)
    for u, period, value in zip(ids, observations[time], y, strict=True):
        if u not in index:
            refuse(
                LONGITUDINAL_OBSERVATION,
                message="Outcomes contain a unit outside the assigned roster.",
            )
        c = schedule[roster[cohort][index[u]]]
        if not _time_matches(period, c.exposure) or period not in (*c.pre_periods, *c.post_periods):
            refuse(
                LONGITUDINAL_OBSERVATION,
                message="Outcomes contain an unscheduled observation time.",
            )
        if (u, period) in seen:
            refuse(LONGITUDINAL_OBSERVATION, message="Each unit/time cell must occur exactly once.")
        seen[u, period] = value
    changes = np.empty(len(roster_ids))
    for i, u in enumerate(roster_ids):
        c = schedule[roster[cohort][i]]
        if any((u, period) not in seen for period in (*c.pre_periods, *c.post_periods)):
            refuse(
                LONGITUDINAL_OBSERVATION,
                message="Missing scheduled observations; preserve the assigned roster and repair the feed, not complete-case selection.",
            )
        anchor = seen[u, c.pre_periods[0]]
        pre = np.asarray([seen[u, period] - anchor for period in c.pre_periods])
        post = np.asarray([seen[u, period] - anchor for period in c.post_periods])
        changes[i] = float(np.dot(c.post_weights, post) - pre.mean())
    if not np.isfinite(changes).all():
        invalid("Change-score arithmetic overflowed; rescale outcomes.")
    return changes


def _roster_check(
    roster: dict[str, np.ndarray],
    *,
    design: LongitudinalDesign,
    unit: str,
    group: str,
    cohort: str,
    exposure: str,
    cluster: str | None,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    ids = _identities(roster[unit], column=unit)
    if len(ids) == 0 or len(set(ids)) != len(ids):
        invalid("Assigned roster requires exactly one row per unit and cannot be empty.")
    if any(type(v) is not str for v in (*roster[group], *roster[cohort])):
        invalid("Arm and cohort columns require nonmissing strings.")
    schedules = {c.name: c for c in design.cohorts}
    if any(a not in (design.control_group, design.treatment_group) for a in roster[group]):
        invalid("Roster contains an undeclared randomized arm.")
    if any(c not in schedules for c in roster[cohort]):
        invalid("Roster contains a missing or undeclared cohort.")
    if any(
        not _time_matches(e, schedules[c].exposure) or e != schedules[c].exposure
        for c, e in zip(roster[cohort], roster[exposure], strict=True)
    ):
        invalid("Roster exposure must equal its predeclared cohort exposure time in both arms.")
    grains = _identities(roster[cluster], column=cluster) if cluster else ids
    grain_rows: dict[str, list[int]] = {}
    for i, grain in enumerate(grains):
        grain_rows.setdefault(grain, []).append(i)
    for rows in grain_rows.values():
        if len({roster[group][i] for i in rows}) != 1:
            invalid("Every randomization cluster must have one pure randomized arm.")
    return ids, grain_rows


def prepare_longitudinal_panel(
    outcomes: IntoDataFrame,
    roster: IntoDataFrame,
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
    """Compress complete scheduled panels into owned per-independence-grain moments.

    Roster cohort, exposure and covariates must be fixed before treatment.
    Continuous and string-valued categorical covariates use the existing
    Covariate encoding conventions, with separate slopes by cohort and arm.
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
    r = _columns(
        roster,
        [
            unit,
            group,
            cohort,
            exposure,
            *([cluster] if cluster else []),
            *(c.name for c in covariates),
        ],
    )
    _, grain_rows = _roster_check(
        r, design=design, unit=unit, group=group, cohort=cohort, exposure=exposure, cluster=cluster
    )
    o = _columns(outcomes, [unit, time, outcome])
    changes = _unit_changes(
        o, r, design=design, unit=unit, cohort=cohort, time=time, outcome=outcome
    )
    inputs: dict[str, np.ndarray] = {}
    for c in covariates:
        if c.kind == "continuous":
            values = _finite(r[c.name], column=c.name)
            shifted = values - values.min()
            if (
                not np.isfinite(shifted).all()
                or not np.isfinite(shifted.std(ddof=1))
                or not shifted.std(ddof=1) > 0
            ):
                invalid("Continuous covariates must have finite nonzero variation.", column=c.name)
            inputs[c.name] = shifted
        else:
            if any(type(v) is not str for v in r[c.name]):
                invalid(
                    "Categorical longitudinal covariates require nonmissing strings.", column=c.name
                )
            if len(set(r[c.name])) < 2:
                invalid("Categorical covariates require at least two levels.", column=c.name)
            inputs[c.name] = r[c.name]
    spec = DesignSpec.fit(inputs, covariates)
    columns = regression_columns(design, spec.column_names())
    basis, basis_names = spec.transform(inputs)
    if not covariates:
        basis = np.empty((len(changes), 0))
    x = np.zeros((len(changes), len(columns)))
    cohort_counts = []
    origins = []
    stride = 1 + len(basis_names)
    for j, c in enumerate(design.cohorts):
        members = r[cohort] == c.name
        cohort_counts.append(int(members.sum()))
        if not members.any():
            invalid("Every declared cohort must occur in the assigned roster.", cohort=c.name)
        centered = basis - basis[members].mean(axis=0)
        for arm_index, arm in enumerate((design.control_group, design.treatment_group)):
            rows = members & (r[group] == arm)
            origin = float(changes[rows].min()) if rows.any() else 0.0
            origins.append(origin)
            changes[rows] -= origin
            block = (2 * j + arm_index) * stride
            x[rows, block : block + stride] = np.column_stack(
                [np.ones(int(rows.sum())), centered[rows]]
            )
    clusters = []
    for grain, rows in sorted(grain_rows.items()):
        xx = x[rows]
        clusters.append(
            LongitudinalCluster(
                identity=grain,
                arm=str(r[group][rows][0]),
                n_units=len(rows),
                gram=tuple(tuple(row) for row in xx.T @ xx),
                cross=tuple(xx.T @ changes[rows]),
            )
        )
    return LongitudinalSummary(
        design=design,
        columns=columns,
        cohort_counts=tuple(cohort_counts),
        n_observations=len(o[unit]),
        covariates=basis_names,
        response_origins=tuple(origins),
        clusters=tuple(clusters),
    )
