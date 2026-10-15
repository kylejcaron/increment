"""Absolute, relative, and ratio projection evidence for multicovariate CUPED."""

from __future__ import annotations

import math

import numpy as np

from increment.errors import InvalidRequestError
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.cuped._numeric import _fail, _projection_statistics
from increment.estimation.cuped.types import MulticovariateCupedProjection


def _cache_projections(calculator, outcomes, order):
    cached = []
    mean_values = np.asarray(calculator.adjusted_means[: len(order)])
    components = np.asarray(calculator.adjusted_means).reshape(len(outcomes), -1)
    for ti in range(1, len(order)):
        treatment = order[ti]
        for ci in range(ti):
            control = order[ci]
            _append_projection(
                cached,
                "absolute",
                treatment,
                control,
                "",
                lambda treatment=treatment, control=control: calculator._absolute_statistics(
                    treatment, control
                ),
                reverse=True,
            )
            if mean_values[ti] > 0 and mean_values[ci] > 0:
                _append_projection(
                    cached,
                    "relative",
                    treatment,
                    control,
                    "",
                    lambda treatment=treatment, control=control: calculator._relative_statistics(
                        treatment, control
                    ),
                    reverse=True,
                )
            if len(outcomes) == 2:
                if np.all(components[:, [ti, ci]] > 0):
                    for scale in ("log", "absolute"):
                        _append_projection(
                            cached,
                            "ratio",
                            treatment,
                            control,
                            scale,
                            lambda treatment=treatment, control=control, scale=scale: (
                                calculator._ratio_statistics(treatment, control, scale)
                            ),
                            reverse=True,
                        )
    return tuple(cached)


def _append_projection(cached, kind, treatment, control, scale, calculate, *, reverse=False):
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            result = calculate()
    except (FloatingPointError, ZeroDivisionError, OverflowError):
        result = None
    except InvalidRequestError as error:
        if error.code not in {
            "estimation.cuped_linear.nonfinite_projection",
            "estimation.tails.unresolvable",
        }:
            raise
        result = None
    if result is None:
        entry = (kind, treatment, control, scale, math.nan, math.nan, math.nan)
    else:
        entry = (kind, treatment, control, scale, result[0], result[1], result[2])
    cached.append(entry)
    if reverse:
        cached.append((kind, control, treatment, scale, -entry[4], entry[5], entry[6]))


def _interval(value: float, se: float, dof: float, alpha: float) -> MulticovariateCupedProjection:
    value, se, dof = _projection_statistics(value, se, dof)
    critical = two_sided_critical_value(student_t_isf, alpha, dof, what="linear CUPED projection")
    half_width = critical * se
    lower, upper = value - half_width, value + half_width
    if not np.isfinite((half_width, lower, upper)).all():
        _fail(
            "estimation.cuped_linear.nonfinite_projection",
            value=float(value),
            se=float(se),
            dof=float(dof),
        )
    return MulticovariateCupedProjection(value, se, dof, lower, upper)
