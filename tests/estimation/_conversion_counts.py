"""Four-count conversion contrasts through ``estimate_lift``, and the arms they or a producer
give, shared by the conversion-route tests and ``calibration.conversion_route``."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from increment.estimation.armstats import ArmStats
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import LiftEstimate
from increment.semantics.models import ConversionMetric

CONVERSION_METRIC = ConversionMetric(name="conv", entity="user", fact="conv")


def arm_row(arm: ArmStats) -> dict[str, Any]:
    """The centered ``group_summary`` row of one arm's stored moments."""
    return {
        "experiment_id": arm.study_id,
        "metric": arm.metric,
        "group_id": arm.group_id,
        "n": arm.n,
        "successes": arm.successes,
        "ref_y": arm.ref_y,
        "cy1": arm.cy1,
        "cy2": arm.cy2,
    }


def count_arms(x_c: int, n_c: int, x_t: int, n_t: int) -> tuple[ArmStats, ArmStats]:
    """``(control, treatment)`` arms of ``x_c`` of ``n_c`` and ``x_t`` of ``n_t`` conversions,
    centered exactly from their counts."""
    control, treatment = (
        ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id=group_id,
            n=n,
            sum_y=float(x),
            sum_y2=float(x),
            successes=x,
        )
        for group_id, n, x in (("control", n_c, x_c), ("treatment", n_t, x_t))
    )
    return control, treatment


def count_summary(x_c: int, n_c: int, x_t: int, n_t: int) -> list[dict[str, Any]]:
    """Centered ``group_summary`` rows of a control arm with ``x_c`` of ``n_c`` conversions
    and a treatment arm with ``x_t`` of ``n_t``."""
    return [arm_row(arm) for arm in count_arms(x_c, n_c, x_t, n_t)]


def producer_arm(n: int, successes: int | None, *, group_id: str = "control") -> ArmStats:
    """The ``ArmStats`` the production producer (``group_summary``) emits for an arm built
    inside DuckDB from ``range()``: ``successes`` ones scattered by a bijection of the row index
    (``None``: every unit's outcome is the constant 0.5). No row is materialized outside DuckDB."""
    import ibis

    from increment.query.builders import group_summary

    if successes is None:
        outcome = "CAST(0.5 AS DOUBLE)"
    else:
        multiplier = next(m for m in range(2654435761, 2654435761 + 1000, 2) if math.gcd(m, n) == 1)
        outcome = (
            f"CAST(CASE WHEN ((range::HUGEINT * {multiplier} + 12345) % {n}) < {successes} "
            "THEN 1 ELSE 0 END AS DOUBLE)"
        )
    totals = ibis.duckdb.connect().sql(
        f"SELECT 'u' AS unit_id, 'e' AS experiment_id, '{group_id}' AS group_id, 'conv' AS metric, "
        f"{outcome} AS y, CAST(NULL AS DOUBLE) AS x, CAST(NULL AS DOUBLE) AS y_den "
        f"FROM range({n})"
    )
    row = (
        group_summary(totals, binary_metrics=["conv"] if successes is not None else [])
        .execute()
        .iloc[0]
    )
    return ArmStats(
        study_id="e",
        metric="conv",
        group_id=group_id,
        n=int(row["n"]),
        successes=int(row["successes"]) if successes is not None else None,
        ref_y=float(row["ref_y"]),
        cy1=float(row["cy1"]),
        cy2=float(row["cy2"]),
    )


def lift_computation(
    counts: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    method: Method | None = None,
    null_lift: float | None = None,
):
    """``estimate_lift`` on the contrast ``counts = (x_c, n_c, x_t, n_t)``."""
    return estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(*counts),
        control_group="control",
        methods=[Method(name="unadjusted") if method is None else method],
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
    )


def lift_row(
    counts: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    mode: Literal["auto", "finite_sample"] = "auto",
    null_lift: float | None = None,
) -> LiftEstimate:
    """The one row of ``lift_computation``; a failed contrast fails the caller."""
    computation = lift_computation(
        counts,
        alpha=alpha,
        alternative=alternative,
        method=Method(name="unadjusted", conversion_inference=mode),
        null_lift=null_lift,
    )
    assert not computation.failures, computation.failures
    (row,) = computation.results
    return row


def runtime_rejection_rate(
    n_c: int,
    n_t: int,
    p_c: float,
    p_t: float,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    mode: Literal["auto", "finite_sample"] = "auto",
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """``(rate, standard_error, asymptotic_share)``: the share of ``reps`` seeded binomial
    count draws whose row ``estimate_lift`` decides against the null (``stat_sig``), and the
    share the route sent to the delta method. ``estimate_lift`` runs once per distinct count
    pair; every draw stays in the denominator."""
    rng = np.random.default_rng(seed)
    x_c = rng.binomial(n_c, p_c, size=reps)
    x_t = rng.binomial(n_t, p_t, size=reps)
    pairs, multiplicity = np.unique(np.stack([x_c, x_t]), axis=1, return_counts=True)
    rejected = asymptotic = 0
    for (c, t), times in zip(pairs.T, multiplicity, strict=True):
        row = lift_row((int(c), n_c, int(t), n_t), alpha=alpha, alternative=alternative, mode=mode)
        rejected += int(times) * row.stat_sig()
        asymptotic += int(times) * (row.reference_kind == "t")
    rate = rejected / reps
    return rate, math.sqrt(max(rate * (1.0 - rate), 1.0 / reps) / reps), asymptotic / reps
