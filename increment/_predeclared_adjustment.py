"""Fit a sequential CUPED adjustment declaration from pre-period data."""

from __future__ import annotations

from typing import Literal

__all__ = ["fit_predeclared_adjustment"]


def fit_predeclared_adjustment(
    frame,
    *,
    unit: str,
    group: str,
    control: str,
    metric: str,
    covariate: str,
    covariate_missing: Literal["impute", "zero", "error"] = "impute",
):
    """Fit a sequential CUPED coefficient and covariate centre from pre-period data.

    *frame* is one row per assigned unit observed BEFORE the experiment reads
    any outcome: *metric* is the metric measured in a pre-period window and
    the in-experiment covariate is measured relative to exposure. The rows
    are reduced to the same centered moments the fixed-horizon frame path
    builds, and the coefficient is ``estimation.cuped.fit_cuped``'s inverse-n
    weighted within-arm slope with the centre its count-weighted pooled
    covariate mean, so sequential and fixed-horizon CUPED agree on what the
    coefficient is.

    The result is a declaration: pass it through ``InferenceSpec.adjustments``
    or ``ScalarMeanModel.adjustment`` and capture retains
    ``Y - coefficient * (X - center)``. Costs relative to the fixed-horizon
    fit: the coefficient is not contrast-optimal for the in-experiment
    outcome, and ``coefficient * (E[X] - center)`` is a first-order bias on
    the ratio scale the scalar-mean route reports whenever the covariate
    drifts between the pre-period and the experiment.
    """
    from fractions import Fraction

    from increment.estimation.cuped import fit_cuped
    from increment.estimation.engine import Method, _df_to_arms
    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import CupedCovariate
    from increment.semantics.sequential import PredeclaredAdjustment

    covariate_declaration = CupedCovariate(
        name=covariate,
        kind="continuous",
        missing="refuse" if covariate_missing == "error" else covariate_missing,
    )
    source = from_unit_summary(
        frame,
        unit=unit,
        group=group,
        control=control,
        metrics=[
            MetricSpec(
                name=metric,
                decision_method=Method.cuped(covariates=covariate_declaration),
            )
        ],
        experiment_id="pre-period",
    )
    fit = fit_cuped(_df_to_arms(source.raw_moments))
    return PredeclaredAdjustment(
        coefficient=Fraction(fit.theta), center=Fraction(fit.mean_x_pooled)
    )
