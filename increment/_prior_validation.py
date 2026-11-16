"""Shared validation for portable prior declarations and finite mixtures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from increment.errors import (
    MODEL_FIELD_REFUSALS,
    InvalidRequestError,
    raiser,
    refusals,
    refuse,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.priors.mixture.prior_needs_least": "a mixture prior needs at least one component",
        "estimation.priors.mixture.weights_means_sigmas": "weights/means/sigmas must have the same length, got {k}/{means}/{sigmas}",
        "estimation.priors.mixture.every_weight_mean": "every mixture weight, mean, and sigma must be finite, got weights={weights}, means={means}, sigmas={sigmas}",
        "estimation.priors.mixture.every_weight": "every mixture weight must be > 0",
        "estimation.priors.mixture.every_sigma": "every mixture sigma must be > 0",
        "estimation.priors.mixture.weights_sum": "mixture weights must sum to 1, got {total}",
    },
)
_raise = raiser(_REFUSALS)


def validate_mixture_prior(
    weights: tuple[float, ...], means: tuple[float, ...], sigmas: tuple[float, ...]
) -> None:
    """Validate finite components without normalizing or reordering them."""
    k = len(weights)
    if k == 0:
        _raise("estimation.priors.mixture.prior_needs_least")
    if len(means) != k or len(sigmas) != k:
        _raise(
            "estimation.priors.mixture.weights_means_sigmas",
            k=k,
            means=len(means),
            sigmas=len(sigmas),
        )
    # NaN would pass the positivity and normalization comparisons.
    if not (
        all(math.isfinite(w) for w in weights)
        and all(math.isfinite(m) for m in means)
        and all(math.isfinite(s) for s in sigmas)
    ):
        _raise(
            "estimation.priors.mixture.every_weight_mean",
            weights=weights,
            means=means,
            sigmas=sigmas,
        )
    if any(w <= 0.0 for w in weights):
        _raise("estimation.priors.mixture.every_weight")
    if any(s <= 0.0 for s in sigmas):
        _raise("estimation.priors.mixture.every_sigma")
    total = sum(weights)
    if abs(total - 1.0) > 1e-9:
        _raise("estimation.priors.mixture.weights_sum", total=total)


def validate_relative_prior_tag(value: Mapping[Any, object]) -> None:
    """Require explicit relative-scale tagging in portable prior payloads."""
    tag = value.get("type")
    if isinstance(tag, str) and tag == "relative_lift":
        return
    code = "model.field.literal" if "type" in value else "model.field.missing"
    refuse(
        MODEL_FIELD_REFUSALS[code],
        model="RelativeLiftPrior",
        field="type",
        loc=("type",),
        value=tag,
        constraint="relative_lift",
    )
