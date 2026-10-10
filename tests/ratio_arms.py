"""Ratio-metric ``ArmStats`` builders for tests: centered moments carrying the
denominator's third moment, either from declared moments or from per-unit arrays."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from increment.estimation.armstats import ArmStats


def moment_arm(
    group_id: str,
    *,
    n: int,
    y_bar: float,
    var_y: float,
    d_bar: float,
    var_d: float,
    skew_d: float = 0.0,
    metric: str = "rpo",
    study_id: str = "e",
) -> ArmStats:
    """An arm with exactly these ddof=1 moments, a denominator of standardized
    sample skewness ``skew_d`` (``m3 / m2**1.5`` on ``1/n`` moments) and zero
    numerator/denominator covariance."""
    second = (n - 1) * var_d
    return ArmStats(
        study_id=study_id,
        metric=metric,
        group_id=group_id,
        n=n,
        ref_y=y_bar,
        cy1=0.0,
        cy2=(n - 1) * var_y,
        ref_den=d_bar,
        cden1=0.0,
        cden2=second,
        cden3=skew_d * second * math.sqrt(second / n),
        cyden=0.0,
    )


def array_arm(
    group_id: str,
    y: np.ndarray,
    d: np.ndarray,
    *,
    metric: str = "rpo",
    study_id: str = "e",
) -> ArmStats:
    """Centered moments of one arm from its per-unit numerator and denominator values."""
    ref_y = float(y.mean())
    ref_d = float(d.mean())
    ry = y - ref_y
    rd = d - ref_d
    return ArmStats(
        study_id=study_id,
        metric=metric,
        group_id=group_id,
        n=len(y),
        ref_y=ref_y,
        cy1=float(ry.sum()),
        cy2=float((ry * ry).sum()),
        ref_den=ref_d,
        cden1=float(rd.sum()),
        cden2=float((rd * rd).sum()),
        cden3=float((rd * rd * rd).sum()),
        cyden=float((ry * rd).sum()),
    )


def summary(*arms: ArmStats) -> pd.DataFrame:
    """``group_summary`` rows for *arms*, third denominator moment and any covariate included."""
    return pd.DataFrame(
        {
            "experiment_id": a.study_id,
            "metric": a.metric,
            "group_id": a.group_id,
            "n": float(a.n),
            "ref_y": a.ref_y,
            "cy1": a.cy1,
            "cy2": a.cy2,
            "ref_x": a.ref_x,
            "cx1": a.cx1,
            "cx2": a.cx2,
            "cxy": a.cxy,
            "x_role": a.x_role,
            "ref_den": a.ref_den,
            "cden1": a.cden1,
            "cden2": a.cden2,
            "cden3": a.cden3,
            "cyden": a.cyden,
            "cxden": a.cxden,
        }
        for a in arms
    )
