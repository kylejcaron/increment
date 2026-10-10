"""Analytic influence-function interval for the pooled upper-winsor contrast.

One pooled type-7 cutoff, clipped arm means, the hurdle log-kernel density at
the cutoff and the centred empirical influence variance over every pool arm;
normal quantiles close the interval. The influence score of a unit with
outcome ``y`` in arm ``g`` is

    a_g (min(y, c) - m_g) + A w_g / h(c) (F_g(c) - 1{y <= c}),

with ``A = sum_g a_g S_g(c)`` and ``h(c) = sum_g w_g (1 - pi_g) f_g^+(c)``. A
zero contributes ``min(0, c) = 0`` to its arm mean and ``1`` to the cutoff
indicator, so the estimated zero shares, the estimated cutoff and their cross
term are inside the score variance without any separate term.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from increment.estimation._winsor_bootstrap import (
    _bootstrap_observed_points,
    full_procedure_statistics_many,
)
from increment.winsor import (
    InfluenceReference,
    InfluenceSeries,
    WinsorConfidenceSet,
    WinsorRawState,
    _reference_context,
    bootstrap_point,
    influence_interval,
    raw_arm_zero_counts,
    refuse_cutoff_in_zero_atom,
    winsor_refuse,
)


def influence_references(
    raw: WinsorRawState,
    control: str,
    treatments: tuple[str, ...],
    *,
    validation_context: Mapping[object, object] | None = None,
) -> dict[str, InfluenceReference]:
    """Build every treatment's analytic reference from one pooled evaluation."""
    if (
        raw.executed_method != "influence-normal-v1"
        or not treatments
        or len(set(treatments)) != len(treatments)
        or control in treatments
    ):
        winsor_refuse(
            "invalid_state",
            "Influence inference requires its resolved method and distinct contrast arms.",
        )
    raw.arm(control)
    for treatment in treatments:
        raw.arm(treatment)
    zero_counts = raw_arm_zero_counts(raw)
    atoms = tuple(count > 0 for count in zero_counts)
    groups = [arm.group_id for arm in raw.arms]
    ci = groups.index(control)
    contrast_indices = tuple((ci, groups.index(treatment)) for treatment in treatments)
    point_summary, reference_context = _reference_context(raw, validation_context)
    cutoff, exact_means = point_summary
    means_by_group = dict(exact_means)
    refuse_cutoff_in_zero_atom(raw, sum(zero_counts))
    samples = tuple(
        np.fromiter(arm.values, dtype=np.float64, count=len(arm.values))[None, :]
        for arm in raw.arms
    )
    statistics = full_procedure_statistics_many(
        samples, raw.quantile, contrast_indices, cutoff=np.asarray([cutoff]), atoms=atoms
    )
    for observed in statistics:
        if not observed.valid()[0]:
            reason = (
                "density_unresolved"
                if not math.isfinite(float(observed.density_scaled[0]))
                or observed.density_scaled[0] <= 0
                else "studentization_degenerate"
            )
            winsor_refuse(reason, "Observed influence studentization is unavailable.")
    points = _bootstrap_observed_points(
        means_by_group[control],
        tuple(means_by_group[treatment] for treatment in treatments),
    )
    scaled_density = float(statistics[0].density_scaled[0])
    references = {}
    for treatment, observed, (ell, delta) in zip(treatments, statistics, points, strict=True):
        references[treatment] = InfluenceReference.model_validate(
            {
                "raw": raw,
                "spec": raw.inference,
                "control": control,
                "treatment": treatment,
                "observed_cutoff": cutoff,
                "scaled_density": scaled_density,
                "log_relative": InfluenceSeries(point=ell, se=float(observed.log_se[0])),
                "additive": InfluenceSeries(point=delta, se=float(observed.additive_se[0])),
            },
            context=reference_context,
        )
    return references


def influence_reference(raw: WinsorRawState, control: str, treatment: str) -> InfluenceReference:
    return influence_references(raw, control, (treatment,))[treatment]


def influence_confidence_set(
    reference: InfluenceReference,
    alpha: float = 0.05,
    *,
    validation_context: Mapping[object, object] | None = None,
) -> WinsorConfidenceSet:
    """Close the normal interval at ``alpha``; the context carries the exact point summary."""
    if not 0 < alpha < 1:
        winsor_refuse("invalid_state", "alpha must be strictly between zero and one.")
    _, validation_context = _reference_context(reference.raw, validation_context)
    return WinsorConfidenceSet.model_validate(
        {
            "raw": reference.raw,
            "control": reference.control,
            "treatment": reference.treatment,
            "alpha": alpha,
            "reference": reference,
            "relative": influence_interval(reference, alpha, relative=True),
            "additive": influence_interval(reference, alpha, relative=False),
            "point": bootstrap_point(reference),
            "additive_point": reference.additive.point,
        },
        context=validation_context,
    )
