"""Four-count conversion contrasts through ``estimate_lift``, shared by the conversion-route
tests and ``calibration.conversion_route``."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from increment.estimation.armstats import ArmStats
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.results import LiftEstimate
from increment.semantics.models import ConversionMetric

CONVERSION_METRIC = ConversionMetric(name="conv", entity="user", fact="conv")


def count_summary(x_c: int, n_c: int, x_t: int, n_t: int) -> list[dict[str, Any]]:
    """Centered ``group_summary`` rows of a control arm with ``x_c`` of ``n_c`` conversions
    and a treatment arm with ``x_t`` of ``n_t``."""
    rows: list[dict[str, Any]] = []
    for group_id, n, x in (("control", n_c, x_c), ("treatment", n_t, x_t)):
        arm = ArmStats.from_raw_sums(
            study_id="e", metric="conv", group_id=group_id, n=n, sum_y=float(x), sum_y2=float(x)
        )
        rows.append(
            {
                "experiment_id": "e",
                "metric": "conv",
                "group_id": group_id,
                "n": float(arm.n),
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
        )
    return rows


def lift_computation(
    counts: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    method: Method | None = None,
):
    """``estimate_lift`` on the contrast ``counts = (x_c, n_c, x_t, n_t)``."""
    return estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(*counts),
        control_group="control",
        methods=[Method(name="unadjusted") if method is None else method],
        alpha=alpha,
        alternative=alternative,
    )


def lift_row(
    counts: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    mode: Literal["auto", "finite_sample"] = "auto",
) -> LiftEstimate:
    """The one row of ``lift_computation``; a failed contrast fails the caller."""
    computation = lift_computation(
        counts,
        alpha=alpha,
        alternative=alternative,
        method=Method(name="unadjusted", conversion_inference=mode),
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
