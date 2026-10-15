"""Stable arithmetic helpers for multicovariate CUPED."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NoReturn

import numpy as np

from increment.errors import InvalidRequestError, RefusalSpec, refusals, refuse

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.cuped_linear.arm_counts": "each arm needs at least two units; arm counts are {counts!r}",
        "estimation.cuped_linear.invalid_arms": "arm labels/order/control disagree: control={control!r}, order={arm_order!r}, observed={observed!r}",
        "estimation.cuped_linear.input_shape": "{name} must be one-dimensional and align with {n} arm labels; got shape {shape}",
        "estimation.cuped_linear.nonfinite": "outcome component {name!r} contains a non-finite value at row {row}",
        "estimation.cuped_linear.missing_refused": "covariate {name!r} contains missing values under missing='refuse'",
        "estimation.cuped_linear.covariate_kind": "covariate {name!r} has unsupported kind {kind!r}",
        "estimation.cuped_linear.covariate_input": "covariate declaration/input names differ: values={values!r}, kinds={kinds!r}",
        "estimation.cuped_linear.level_support": "categorical level for {column!r} has fewer than two units in an arm: level={level!r}, counts={counts!r}",
        "estimation.cuped_linear.rank_deficiency": RefusalSpec(
            "estimation.cuped_linear.rank_deficiency",
            InvalidRequestError,
            template="encoded covariates are rank deficient: columns={columns!r}, partners={partners!r}",
        ),
        "estimation.cuped_linear.leverage": "unit leverage is too close to one: row={row}, leverage={leverage}",
        "estimation.cuped_linear.support_region": RefusalSpec(
            "estimation.cuped_linear.support_region",
            InvalidRequestError,
            template="fit is outside support region {region!r}: violations={violations!r}",
        ),
        "estimation.cuped_linear.residual_df": "residual degrees of freedom must be positive; got {df}",
        "estimation.cuped_linear.nonpositive_component": "adjusted component {family!r} for arm {arm!r} must be positive for a log or ratio projection; got {value}",
        "estimation.cuped_linear.nonfinite_projection": RefusalSpec(
            "estimation.cuped_linear.nonfinite_projection",
            InvalidRequestError,
            template="projection produced non-finite numerical evidence: value={value}, se={se}, dof={dof}",
        ),
        "estimation.cuped_linear.invalid_alpha": "alpha must be strictly between zero and one; got {alpha}",
        "estimation.cuped_linear.invalid_scale": RefusalSpec(
            "estimation.cuped_linear.invalid_scale",
            InvalidRequestError,
            template="projection scale must be 'log' or 'absolute', got {scale!r}",
        ),
        "estimation.cuped_linear.projection_arm": RefusalSpec(
            "estimation.cuped_linear.projection_arm",
            InvalidRequestError,
            template="projection requires distinct declared treatment/control arms: {arm!r}, {control!r}",
        ),
        "estimation.cuped_linear.projection_kind": RefusalSpec(
            "estimation.cuped_linear.projection_kind",
            InvalidRequestError,
            template="projection {requested!r} is incompatible with fit kind {available!r}",
        ),
        "estimation.cuped_linear.outcome_components": RefusalSpec(
            "estimation.cuped_linear.outcome_components",
            InvalidRequestError,
            template="supply either y or both numerator and denominator components",
        ),
        "estimation.cuped_linear.projection_unavailable": RefusalSpec(
            "estimation.cuped_linear.projection_unavailable",
            InvalidRequestError,
            template="projection is unavailable for fit: kind={kind!r}, arm={arm!r}, control={control!r}",
        ),
        "estimation.cuped_linear.invalid_evidence": RefusalSpec(
            "estimation.cuped_linear.invalid_evidence",
            InvalidRequestError,
            template="invalid CUPED compact evidence: reason={reason!r}",
        ),
        "estimation.cuped_linear.incompatible_evidence": RefusalSpec(
            "estimation.cuped_linear.incompatible_evidence",
            InvalidRequestError,
            template="incompatible CUPED compact evidence: reason={reason!r}",
        ),
    },
)


def _fail(code: str, **context: object) -> NoReturn:
    refuse(_REFUSALS[code], **context)


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    out.setflags(write=False)
    return out


def _stable_norm(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values), initial=0.0))
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(values / scale))


def _two_product(left: float, right: float) -> tuple[float, float]:
    product = left * right
    if hasattr(math, "fma"):
        return product, math.fma(left, right, -product)
    splitter = 134217729.0
    left_split = splitter * left
    left_high = left_split - (left_split - left)
    left_low = left - left_high
    right_split = splitter * right
    right_high = right_split - (right_split - right)
    right_low = right - right_high
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error


def _partial_sums(terms: Sequence[float]) -> tuple[float, int, float]:
    """Pieces ``(large, shift, small)`` with ``sum(terms) == ldexp(large, shift) + small``.

    The shift bounds the scaled total below the float range; terms the scaling
    would denormalise are summed at their own scale instead, so a small
    remainder beside cancelling extremes is kept.
    """
    shift = len(terms).bit_length() + 2
    floor = math.ldexp(1.0, shift - 1022)
    large = math.fsum(math.ldexp(term, -shift) for term in terms if abs(term) >= floor)
    small = math.fsum(term for term in terms if abs(term) < floor)
    return large, shift, small


def _exact_sum(terms: Sequence[float]) -> float:
    """Sum of finite floats rounded once, or within two roundings when fsum's partials overflow.

    Returns a signed infinity only when the sum itself lies beyond the float range.
    """
    try:
        return math.fsum(terms)
    except OverflowError:
        large, shift, small = _partial_sums(terms)
        try:
            return math.ldexp(large, shift) + small
        except OverflowError:
            return math.copysign(math.inf, large)


def _arm_mean(values: np.ndarray) -> float:
    """Mean of finite values whose exact sum may overflow; always representable."""
    try:
        return math.fsum(float(value) for value in values) / len(values)
    except OverflowError:
        items = tuple(float(value) for value in values)
        large, shift, small = _partial_sums(items)
        return math.ldexp(large / len(items), shift) + small / len(items)


_ScaledMean = tuple[float, float, int]
_UNIT_MEAN: _ScaledMean = (0.5, 0.0, 1)


def _normalized(parts: Sequence[float]) -> _ScaledMean:
    """``sum(parts)`` as ``(hi, lo, exponent)`` with ``(hi + lo) * 2**exponent`` exact to two doubles.

    ``hi`` lies in ``[0.5, 1)`` in magnitude and ``lo`` holds the bits its rounding
    dropped. The parts are scaled only when their plain sum overflows, so a
    subnormal mean keeps every bit; a scaled sum loses only parts far below the
    two-double precision of the total.
    """
    try:
        total = math.fsum(parts)
        remainder = math.fsum([*parts, -total])
        shift = 0
    except OverflowError:
        shift = len(parts).bit_length() + 2
        parts = [math.ldexp(part, -shift) for part in parts]
        total = math.fsum(parts)
        remainder = math.fsum([*parts, -total])
    hi, exponent = math.frexp(total)
    return hi, math.ldexp(remainder, -exponent), exponent + shift


def _row_exponent(mean: float, residual: np.ndarray, fitted: np.ndarray) -> int:
    """Power of two bringing an arm's residual and fitted contributions to at most unit scale."""
    magnitude = max(
        abs(mean) if math.isfinite(mean) else 0.0,
        float(np.max(np.abs(residual), initial=0.0)),
        float(np.max(np.abs(fitted), initial=0.0)),
    )
    return math.frexp(magnitude)[1] if magnitude else 0


def _rounded_ratio(numerator: _ScaledMean, denominator: _ScaledMean) -> float:
    """Quotient of two scaled means; raises OverflowError beyond the float range."""
    return math.ldexp(numerator[0] / denominator[0], numerator[2] - denominator[2])


def _product_terms(left: _ScaledMean, right: _ScaledMean) -> tuple[float, ...]:
    """Doubles whose sum is the product of two scaled mantissas to second order."""
    product, error = _two_product(left[0], right[0])
    return product, error, left[0] * right[1], left[1] * right[0]


def _log_cross_ratio(
    nt: _ScaledMean, nd: _ScaledMean, nc: _ScaledMean, cd: _ScaledMean
) -> tuple[float, float | None]:
    """``log(nt * cd / (nc * nd))`` from scaled means, with the ratio's excess over one.

    The cross products are built from the means' mantissas, so they can neither
    overflow nor underflow, and their difference is a correctly rounded sum of
    exact product terms: a contrast below one ulp of the rounded means survives
    in the excess, which is returned while it lies within ``log1p``'s half-unit
    range and is None otherwise. Beyond that range the log is the mantissa
    products' log plus the exponent gap times ``log 2``, which keeps the error
    at a few ulps of the result instead of a few ulps of each component's log.
    """
    gap = (nt[2] + cd[2]) - (nc[2] + nd[2])
    treatment = _product_terms(nt, cd)
    control = _product_terms(nc, nd)
    if abs(gap) <= 2:
        treatment_scale = math.ldexp(1.0, min(gap, 0))
        control_scale = math.ldexp(1.0, min(-gap, 0))
        scaled_control = math.fsum(term * control_scale for term in control)
        excess = (
            math.fsum(
                [term * treatment_scale for term in treatment]
                + [-term * control_scale for term in control]
            )
            / scaled_control
        )
        if abs(excess) <= 0.5:
            return math.log1p(excess), excess
    return math.log(math.fsum(treatment) / math.fsum(control)) + gap * math.log(2.0), None


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        _fail("estimation.cuped_linear.invalid_alpha", alpha=alpha)


def _canonical_float(value: float) -> float | str:
    """Retain a refusal observable as a finite float or its canonical text form."""
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _projection_statistics(value: float, se: float, dof: float) -> tuple[float, float, float]:
    if not np.isfinite((value, se, dof)).all() or se < 0 or dof <= 0:
        _fail(
            "estimation.cuped_linear.nonfinite_projection",
            value=_canonical_float(value),
            se=_canonical_float(se),
            dof=_canonical_float(dof),
        )
    return float(value), float(se), float(dof)
