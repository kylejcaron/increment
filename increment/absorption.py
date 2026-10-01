"""Public adapter: absorb a declared factor from a moment table.

Pivots a `group_summary`-shaped frame (one row per factor level x arm)
into the parallel arrays `absorb_one_way` consumes. Lives outside
`increment/estimation/` because it touches narwhals frames; the
estimation layer stays pure-math.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import narwhals as nw
from narwhals.typing import IntoDataFrame

from increment.errors import (
    IncrementRuntimeWarning,
    InvalidRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation.absorption import AbsorptionResult, absorb_one_way

_REQUIRED = ("group_id", "n", "ref_y", "cy1", "cy2")

Renderer = Callable[..., str]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "absorption.summary_eager_dataframe": "summary must be an eager dataframe (e.g. .to_pyarrow() on an ibis Table)",
        "absorption.summary_missing_columns": "summary is missing required columns: {missing}",
        "absorption.control_group_present": "control group {control_group!r} not present; arms are {arms}",
        "absorption.absorb_factor_needs": "absorb_factor needs exactly two arms; got {arms}. Filter the table to one treated arm before calling.",
        "absorption.summary_duplicate_rows": "summary has duplicate rows for {named} -- absorb_factor needs exactly one row per (factor level, arm); a duplicate would silently overwrite the earlier row's moments and bias the effect. Aggregate to one row per cell upstream (GROUP BY {factor!r}, group_id).",
    },
)
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementRuntimeWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "absorption.factor_levels_below_sandwich_floor",
    IncrementRuntimeWarning,
    lambda *, factor, n_levels, floor: (
        f"factor {factor!r}: {n_levels} levels survived absorption, "
        f"below {floor} -- the level-clustered "
        "sandwich variance is itself noisy at this K and the t_(K-2) "
        "interval over-rejects; treat borderline significance as fragile."
    ),
)

# Warn below 40 levels, like engine.check_total_clusters for the same t_(K-2)
# reference and CR1 sandwich, but do not refuse below 10: absorption calibration
# (tests/test_absorption_calibration.py::TestCoverage::test_few_levels_still_covers)
# shows conservative coverage down to K=3. Kernel absorb_one_way stays unguarded.
_WARN_LEVELS_FOR_SANDWICH = 40


def _check_level_count(factor: str, n_levels: int) -> None:
    if n_levels < _WARN_LEVELS_FOR_SANDWICH:
        _warn(
            "absorption.factor_levels_below_sandwich_floor",
            factor=factor,
            n_levels=n_levels,
            floor=_WARN_LEVELS_FOR_SANDWICH,
            stacklevel=3,
        )


def _pooled_ref(rows: Sequence[Mapping[str, Any]]) -> float:
    """Count-weighted mean of ``y`` across *rows*, anchored on the first ref.

    Anchoring keeps the accumulator on the between-row mean spread rather
    than on ``sum(y)``, so no large magnitude is ever summed.
    """
    base = float(rows[0]["ref_y"])
    total = 0.0
    excess = 0.0
    for r in rows:
        n = float(r["n"])
        total += n
        excess += n * (float(r["ref_y"]) - base) + float(r["cy1"])
    return base if total == 0.0 else base + excess / total


def _shifted(row: Mapping[str, Any] | None, ref: float) -> tuple[float, float, float]:
    """One cell's ``(n, sum(y - ref), sum((y - ref)**2))``, from its centered
    moments via the parallel-combination identities. Absent cell -> zeros.
    """
    if row is None:
        return 0.0, 0.0, 0.0
    n = float(row["n"])
    delta = float(row["ref_y"]) - ref
    cy1 = float(row["cy1"])
    return n, cy1 + n * delta, float(row["cy2"]) + 2.0 * delta * cy1 + n * delta * delta


def absorb_factor(
    summary: IntoDataFrame,
    *,
    factor: str,
    control_group: str,
    pooling: Literal["partial", "hard", "none"] = "partial",
    alpha: float = 0.05,
) -> AbsorptionResult:
    """Absorb *factor* and return the sharpened average treatment effect.

    Parameters
    ----------
    summary : IntoDataFrame
        `group_summary`-shaped table, one row per factor level x arm,
        carrying *factor* plus `group_id, n, ref_y, cy1, cy2`.
    factor : str
        Name of the factor-level column.
    control_group : str
        Which `group_id` is the control arm.
    pooling : {"partial", "hard", "none"}
        Passed through; prefer the default.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    AbsorptionResult
        Effect on the absolute scale, with a level-clustered interval.

    Raises
    ------
    ValueError
        If required columns are missing, the control group is absent, or
        the table does not describe exactly two arms.

    Warns
    -----
    RuntimeWarning
        Below 40 levels survive absorption -- see ``_check_level_count``.

    Notes
    -----
    `absorb_one_way` takes raw `sum(y)`/`sum(y**2)` per cell, so centered
    moments are re-expressed against one global reference (the
    count-weighted pooled mean) instead of raw sums. This is exact: the
    model `y = mu + tau*D + b_g + e` is invariant to a location shift of
    `y` - only the intercept moves, so no field on `AbsorptionResult`
    reports an absolute level (`effect` is a contrast; `se`/`icc`/
    `mean_shrinkage` are all shift-invariant).
    """
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    if not isinstance(frame, nw.DataFrame):
        _raise("absorption.summary_eager_dataframe")
    missing = [c for c in (factor, *_REQUIRED) if c not in frame.columns]
    if missing:
        _raise("absorption.summary_missing_columns", missing=missing)

    rows: list[dict[str, Any]] = frame.rows(named=True)
    arms = sorted({r["group_id"] for r in rows}, key=lambda v: (v is None, str(v)))
    if control_group not in arms:
        _raise("absorption.control_group_present", control_group=control_group, arms=arms)
    treated = [a for a in arms if a != control_group]
    if len(treated) != 1:
        _raise("absorption.absorb_factor_needs", arms=arms)
    treat_group = treated[0]

    by_level: dict[Any, dict[str, dict[str, Any]]] = {}
    duplicates: list[tuple[Any, Any]] = []
    for r in rows:
        cells = by_level.setdefault(r[factor], {})
        if r["group_id"] in cells:
            duplicates.append((r[factor], r["group_id"]))
            continue
        cells[r["group_id"]] = r
    if duplicates:
        named = ", ".join(f"(level={lvl!r}, arm={arm!r})" for lvl, arm in duplicates)
        _raise("absorption.summary_duplicate_rows", factor=factor, named=named)

    # The model y = mu + tau*D + b_g + e is invariant to a location shift of y,
    # so one global reference keeps s/q off the O(n * mean**2) scale.
    ref = _pooled_ref(rows)

    n_c, s_c, q_c, n_t, s_t, q_t = [], [], [], [], [], []
    for level in sorted(by_level, key=lambda v: (v is None, str(v))):
        cells = by_level[level]
        n, s, q = _shifted(cells.get(control_group), ref)
        n_c.append(n)
        s_c.append(s)
        q_c.append(q)
        n, s, q = _shifted(cells.get(treat_group), ref)
        n_t.append(n)
        s_t.append(s)
        q_t.append(q)

    result = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t, pooling=pooling, alpha=alpha)
    _check_level_count(factor, result.n_levels_used)
    return result


__all__ = ["absorb_factor"]
