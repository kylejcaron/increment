"""Fixed-horizon mixed-covariate CUPED with fitted-slope HC2 inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.linalg import qr as scipy_qr

from increment.estimation.armstats import ArmStats
from increment.estimation.cuped._numeric import (
    _UNIT_MEAN,
    _arm_mean,
    _canonical_float,
    _check_alpha,
    _exact_sum,
    _fail,
    _log_cross_ratio,
    _normalized,
    _projection_statistics,
    _readonly,
    _rounded_ratio,
    _row_exponent,
    _ScaledMean,
    _stable_norm,
)
from increment.estimation.cuped.encoding import (
    MulticovariateCupedLayout,
    _encode_covariates,
    _normalized_offset_difference,
)
from increment.estimation.cuped.inference import (
    _bm,
    _bm_reductions,
    _bm_terms,
    _bm_u_gram,
    _mean_covariance,
    finalize_fit,
)
from increment.estimation.cuped.projections import _cache_projections, _interval
from increment.estimation.cuped.types import (
    FitState,
    InferenceSummary,
    MulticovariateCupedFit,
    MulticovariateCupedProjection,
    ProjectionSpec,
)
from increment.estimation.variance import ratio_denominator_precision

M1_MAX_ENCODED_COVARIATES = 203
M1_MAX_CATEGORICAL_LEVELS = 200
M1_MAX_LEVERAGE_RATIO = 0.20
M1_MAX_LEVERAGE = 0.45
M1_MAX_RATIO_DENOMINATOR_PRECISION = 0.15


@dataclass(frozen=True, slots=True)
class _LinearCupedCalculator:
    arm_order: tuple[object, ...]
    control: object
    n_by_arm: tuple[int, ...]
    adjusted_means: tuple[float, ...]
    theta: tuple[float, ...]
    anchor: tuple[float, ...]
    layout: MulticovariateCupedLayout
    covariance: tuple[tuple[float, ...], ...]
    dof: float
    n_min: int
    p: int
    max_leverage: float
    mean_leverage: float
    leverage: np.ndarray
    _x: np.ndarray
    _arm_index: np.ndarray
    _components: tuple[tuple[np.ndarray, ...], ...]
    _slopes: tuple[np.ndarray, ...]
    _residuals: tuple[tuple[np.ndarray, ...], ...]
    _x_means: tuple[np.ndarray, ...]
    _x_offsets: tuple[np.ndarray, ...]
    _x_offset_anchor: np.ndarray
    _gram_factor: np.ndarray
    _weights: np.ndarray
    _design: np.ndarray
    _design_factor: np.ndarray
    _design_q: np.ndarray
    _bm_u_gram: np.ndarray
    _means: tuple[tuple[float, ...], ...]
    _row_exponents: tuple[tuple[int, ...], ...]

    def _contrast_x(self, j: np.ndarray) -> np.ndarray:
        jsum = math.fsum(float(value) for value in j)
        return np.fromiter(
            (
                math.fsum(
                    [float(j[a]) * float(self._x_means[a][k]) for a in range(len(j))]
                    + [-jsum * float(self.anchor[k])]
                    + [float(j[a]) * float(self._x_offsets[a][k]) for a in range(len(j))]
                    + [-jsum * float(self._x_offset_anchor[k])]
                )
                for k in range(self.p)
            ),
            dtype=float,
            count=self.p,
        )

    def _arm_x_difference(self, arm: int) -> np.ndarray:
        return np.fromiter(
            (
                math.fsum(
                    (
                        float(self._x_means[arm][k]),
                        -float(self.anchor[k]),
                        float(self._x_offsets[arm][k]),
                        -float(self._x_offset_anchor[k]),
                    )
                )
                for k in range(self.p)
            ),
            dtype=float,
            count=self.p,
        )

    def absolute(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> MulticovariateCupedProjection:
        """Adjusted additive arm contrast with an HC2/BM t interval."""
        _check_alpha(alpha)
        return _interval(*self._absolute_statistics(arm, control), alpha)

    def _absolute_statistics(
        self, arm: object, control: object | None
    ) -> tuple[float, float, float]:
        j = self._contrast(arm, control)
        return _projection_statistics(
            self._contrast_value(j),
            _stable_norm(self._family_score(0, j)),
            self._bm_dof(j),
        )

    def _mean(self, family: int, arm: int) -> tuple[float, ...]:
        """Adjusted mean as finite parts whose exact sum may lie beyond the float range."""
        return self._means[family * len(self.arm_order) + arm]

    def _usable(self, family: int, index: int, arm: object, label: str) -> _ScaledMean:
        """A log-scale projection needs a positive mean built from finite parts."""
        rounded = self.adjusted_means[family * len(self.arm_order) + index]
        if not rounded > 0:
            _fail(
                "estimation.cuped_linear.nonpositive_component",
                family=label,
                arm=arm,
                value=rounded,
            )
        parts = self._mean(family, index)
        if not all(math.isfinite(part) for part in parts):
            _projection_statistics(rounded, math.nan, math.nan)
        return _normalized(parts)

    def _contrast_value(self, j: np.ndarray, family: int = 0) -> float:
        """Contrast of the arms ``j`` weights, exact up to its final rounding.

        Only arms with nonzero weight enter, so an unrelated arm cannot move the
        alignment; the result is infinite only when the contrast itself is.
        """
        parts = [
            float(j[a]) * part
            for a in range(len(self.arm_order))
            if j[a]
            for part in self._mean(family, a)
        ]
        if not all(math.isfinite(part) for part in parts):
            return math.nan
        return _exact_sum(parts)

    def relative(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> MulticovariateCupedProjection:
        """Log adjusted-mean ratio with a plug-in Jacobian and t interval."""
        _check_alpha(alpha)
        return _interval(*self._relative_statistics(arm, control), alpha)

    def _relative_statistics(
        self, arm: object, control: object | None
    ) -> tuple[float, float, float]:
        ti, ci = self._pair(arm, control)
        mt = self._usable(0, ti, arm, "outcome")
        mc = self._usable(0, ci, self.arm_order[ci], "outcome")
        score = self._arm_score(0, ti, mt) - self._arm_score(0, ci, mc)
        jac = self._scaled_jacobian(len(self.arm_order), ((ti, mt, 1.0), (ci, mc, -1.0)))
        return _projection_statistics(
            self._log_relative_value(ti, ci, mt, mc), _stable_norm(score), self._bm_dof(jac)
        )

    def _log_relative_value(self, ti: int, ci: int, mt: _ScaledMean, mc: _ScaledMean) -> float:
        """``log(mean_t / mean_c)`` from the exact mean difference while the arms are within two."""
        rounded_t, rounded_c = self.adjusted_means[ti], self.adjusted_means[ci]
        if min(rounded_t, rounded_c) * 2 >= max(rounded_t, rounded_c):
            difference = _exact_sum([*self._mean(0, ti), *(-part for part in self._mean(0, ci))])
            return math.log1p(difference / rounded_c)
        return _log_cross_ratio(mt, _UNIT_MEAN, mc, _UNIT_MEAN)[0]

    def ratio(
        self,
        arm: object,
        *,
        control: object | None = None,
        scale: Literal["log", "absolute"] = "log",
        alpha: float = 0.05,
    ) -> MulticovariateCupedProjection:
        """Ratio-of-component projection retaining stacked numerator/denominator scores."""
        _check_alpha(alpha)
        return _interval(*self._ratio_statistics(arm, control, scale), alpha)

    def _ratio_statistics(
        self,
        arm: object,
        control: object | None,
        scale: Literal["log", "absolute"],
    ) -> tuple[float, float, float]:
        """Delta-method ratio projection.

        Each score is an arm's influence divided by the mean the scale needs: the
        arm's own component mean on the log scale, the arm's denominator mean on
        the absolute scale, so the absolute ratio never depends on a numerator
        log score that may be unrepresentable. The Jacobian handed to the
        degrees-of-freedom trace is the same vector scaled by the smallest mean,
        which leaves the trace ratio unchanged. The point uses the scaled means'
        cross products, so a contrast below one ulp of the rounded means survives.
        """
        if not self._components:
            _fail("estimation.cuped_linear.projection_kind", requested="ratio", available="mean")
        ti, ci = self._pair(arm, control)
        narm = len(self.arm_order)
        nd = self._usable(1, ti, arm, "denominator")
        cd = self._usable(1, ci, self.arm_order[ci], "denominator")
        nt = self._usable(0, ti, arm, "numerator")
        nc = self._usable(0, ci, self.arm_order[ci], "numerator")
        value, excess = _log_cross_ratio(nt, nd, nc, cd)
        if scale == "log":
            score = (
                self._arm_score(0, ti, nt)
                - self._arm_score(1, ti, nd)
                - self._arm_score(0, ci, nc)
                + self._arm_score(1, ci, cd)
            )
            terms = ((ti, nt, 1.0), (ci, nc, -1.0), (narm + ti, nd, -1.0), (narm + ci, cd, 1.0))
        elif scale == "absolute":
            rt, rc = _rounded_ratio(nt, nd), _rounded_ratio(nc, cd)
            score = (
                self._arm_score(0, ti, nd)
                - rt * self._arm_score(1, ti, nd)
                - self._arm_score(0, ci, cd)
                + rc * self._arm_score(1, ci, cd)
            )
            terms = ((ti, nd, 1.0), (ci, cd, -1.0), (narm + ti, nd, -rt), (narm + ci, cd, rc))
            value = excess * rc if excess is not None else rt - rc
        else:
            _fail("estimation.cuped_linear.invalid_scale", scale=scale)
        return _projection_statistics(
            value, _stable_norm(score), self._stacked_bm(self._scaled_jacobian(2 * narm, terms))
        )

    def _arm_score(self, family: int, arm: int, divisor: _ScaledMean | None = None) -> np.ndarray:
        j = np.zeros(len(self.arm_order))
        j[arm] = 1
        return self._family_score(family, j, divisor)

    @staticmethod
    def _scaled_jacobian(
        size: int, terms: tuple[tuple[int, _ScaledMean, float], ...]
    ) -> np.ndarray:
        """Jacobian ``weight / mean`` per entry, scaled by the smallest mean so no reciprocal forms."""
        return _LinearCupedCalculator._scaled_jacobian_details(size, terms)[0]

    @staticmethod
    def _scaled_jacobian_details(
        size: int, terms: tuple[tuple[int, _ScaledMean, float], ...]
    ) -> tuple[np.ndarray, float, float, int]:
        """Return a stable Jacobian and its reciprocal reference-scale metadata."""
        reference = min((mean[2], mean[0]) for _, mean, _ in terms)
        jacobian = np.zeros(size)
        for index, (high, _, exponent), weight in terms:
            jacobian[index] = math.ldexp(reference[1] / high, reference[0] - exponent) * weight
        return jacobian, 1.0, reference[1], -reference[0]

    def _pair(self, arm: object, control: object | None) -> tuple[int, int]:
        try:
            ti = self.arm_order.index(arm)
            ci = self.arm_order.index(self.control if control is None else control)
        except ValueError:
            _fail("estimation.cuped_linear.projection_arm", arm=arm, control=control)
        if ti == ci:
            _fail("estimation.cuped_linear.projection_arm", arm=arm, control=control)
        return ti, ci

    def _contrast(self, arm: object, control: object | None) -> np.ndarray:
        ti, ci = self._pair(arm, control)
        out = np.zeros(len(self.arm_order))
        out[ti] = 1
        out[ci] = -1
        return out

    def _family_score(
        self, family: int, j: np.ndarray, divisor: _ScaledMean | None = None
    ) -> np.ndarray:
        """Per-unit HC2 influence scores of the contrast ``j`` on family ``family``.

        With a ``divisor`` the scores are divided by that scaled mean. Each arm's
        residual and fitted contributions are first brought to unit scale by the
        arm's own exponent, so neither a subnormal arm nor an arm far larger than
        the divisor loses or overflows its product with the influence weights; the
        exponent gap to the divisor is applied last, to the weighted contribution.
        """
        scale = float(np.max(np.abs(j), initial=0.0))
        if scale:
            j = j / scale
        d = self._contrast_x(j)
        contrast = np.concatenate([j, -d])
        coefficients = np.linalg.solve(
            self._design_factor,
            np.linalg.solve(self._design_factor.T, contrast),
        )
        share = float(j.sum()) / sum(self.n_by_arm)
        influence = self._design @ coefficients * self._weights
        leverage_factor = np.sqrt(1 - self.leverage)
        score = np.empty(len(self._x))
        for a, residual in enumerate(self._residuals[family]):
            mask = self._arm_index == a
            fitted = self._x[mask] @ self._slopes[family]
            if divisor is None:
                score[mask] = influence[mask] * residual / leverage_factor[mask] + share * fitted
                continue
            exponent = self._row_exponents[family][a]
            contribution = influence[mask] * np.ldexp(residual, -exponent) / leverage_factor[
                mask
            ] + share * np.ldexp(fitted, -exponent)
            score[mask] = np.ldexp(contribution / divisor[0], exponent - divisor[2])
        return score * scale

    def _bm_dof(self, j: np.ndarray) -> float:
        d = self._contrast_x(j)
        return _bm(
            self._design,
            self._weights,
            self.leverage,
            self._design_factor,
            np.concatenate([j, -d]),
            q_design=self._design_q,
            u_gram=self._bm_u_gram,
        )

    def _stacked_bm(self, jac: np.ndarray) -> float:
        narm = len(self.arm_order)
        jac = jac / float(np.max(np.abs(jac)))
        contrasts = []
        for family in range(2):
            j = jac[family * narm : (family + 1) * narm]
            contrasts.append(np.concatenate([j, -self._contrast_x(j)]))
        scale = max(float(np.max(np.abs(item), initial=0.0)) for item in contrasts)
        terms = [
            _bm_terms(
                self._design,
                self._weights,
                self.leverage,
                self._design_factor,
                contrast,
                scale=scale,
                q_design=self._design_q,
                u_gram=self._bm_u_gram,
            )
            for contrast in contrasts
        ]
        trace = math.fsum(item[0] for item in terms)
        trace_square = math.fsum(item[1] for item in terms)
        return trace * trace / trace_square


def fit_cuped_multicovariate(
    y: npt.ArrayLike | None,
    arms: npt.ArrayLike,
    *,
    control: object,
    arm_order: Sequence[object],
    covariates: Mapping[str, npt.ArrayLike] | None = None,
    kinds: Mapping[str, str] | None = None,
    missing: Mapping[str, str] | None = None,
    numerator: npt.ArrayLike | None = None,
    denominator: npt.ArrayLike | None = None,
) -> MulticovariateCupedFit:
    groups, order, indices, counts, outcomes, covariates, kinds, missing = _validate_request(
        y, arms, control, arm_order, covariates, kinds, missing, numerator, denominator
    )
    x, layout, offsets, scales = _encode_covariates(
        groups, indices, order, covariates, kinds, missing
    )
    coordinate_offsets = (
        np.column_stack(
            [
                np.array(
                    [
                        _normalized_offset_difference(offset[a], offset[0], scale)
                        for a in range(len(order))
                    ]
                )
                for offset, scale in zip(offsets, scales, strict=True)
            ]
        )
        if scales
        else np.empty((len(order), 0))
    )
    calculator, leverage = _fit_calculator(
        groups, order, counts, outcomes, x, layout, control, coordinate_offsets
    )
    del x
    return _finalize_local_fit(
        groups,
        order,
        indices,
        counts,
        outcomes,
        covariates,
        layout,
        offsets,
        scales,
        control,
        calculator,
        leverage,
    )


def _finalize_local_fit(
    groups,
    order,
    indices,
    counts,
    outcomes,
    covariates,
    layout,
    offsets,
    scales,
    control,
    calculator,
    leverage,
):
    anchor = tuple(
        math.fsum(
            [calculator.anchor[i] * scales[i]]
            + [counts[a] / sum(counts) * offsets[i][a] for a in range(len(order))]
        )
        for i in range(len(scales))
    )
    theta = (
        tuple(value / scales[i % len(scales)] for i, value in enumerate(calculator.theta))
        if scales
        else ()
    )
    cached = _cache_projections(calculator, outcomes, order)
    identity = (
        "local-cuped-fit-v1",
        tuple(order),
        control,
        tuple(counts),
        layout.names,
        layout.kinds,
        layout.encoded_columns,
        anchor,
        theta,
        tuple(calculator.adjusted_means),
        _input_identity(groups, outcomes, covariates),
    )
    state = _make_fit_state(
        identity,
        order,
        indices,
        counts,
        outcomes,
        layout,
        offsets,
        scales,
        control,
        calculator,
        anchor,
        theta,
    )
    summary = _make_inference_summary(
        identity, cached, calculator, outcomes, order, leverage, _mean_covariance(calculator)
    )
    return finalize_fit(state, summary)


def _make_fit_state(
    identity,
    order,
    indices,
    counts,
    outcomes,
    layout,
    offsets,
    scales,
    control,
    calculator,
    anchor,
    theta,
):
    sqrt_weights = np.sqrt(calculator._weights)
    transformed_rhs = []
    for component in calculator._components:
        centered = np.empty(sum(counts), dtype=float)
        for arm_index, arm_rows in enumerate(indices):
            centered[arm_rows] = component[arm_index]
        transformed_rhs.append(
            tuple(float(value) for value in calculator._design_q.T @ (sqrt_weights * centered))
        )
    denominator_support = (
        tuple(
            (
                order[arm_index],
                _raw_denominator_precision(outcomes[0][rows], outcomes[1][rows], order[arm_index]),
            )
            for arm_index, rows in enumerate(indices)
        )
        if len(outcomes) == 2
        else ()
    )
    return FitState(
        identity=identity,
        eligible_arms=tuple(order),
        withheld_arms=(),
        refusal_contexts=(),
        counts=tuple(counts),
        layout=layout,
        offsets=tuple(tuple(float(value) for value in row) for row in offsets),
        scales=tuple(float(value) for value in scales),
        centered_means=tuple(tuple(float(value) for value in row) for row in calculator._x_means),
        anchor=anchor,
        triangular_factor=tuple(
            tuple(float(value) for value in row) for row in calculator._design_factor
        ),
        transformed_rhs=tuple(transformed_rhs),
        slopes=tuple(tuple(float(value) for value in row) for row in calculator._slopes),
        adjusted_means=calculator.adjusted_means,
        theta=theta,
        n_min=min(counts),
        p=calculator.p,
        is_ratio=len(outcomes) == 2,
        denominator_support=denominator_support,
        control=control,
    )


def _make_inference_summary(identity, cached, calculator, outcomes, order, leverage, covariance):
    u_gram = calculator._bm_u_gram
    projected_scores, s2_rows, s4_rows, g_rows, g2_rows = [], [], [], [], []
    for projection in cached:
        kind, treatment, comparison, projection_scale, estimate, se, _ = projection
        projected_scores.append((kind, treatment, comparison, projection_scale, estimate, se, 1.0))
        component_contrasts = _projection_bm_contrasts(calculator, projection)
        reductions = (
            tuple(
                (
                    0.0,
                    0.0,
                    np.zeros((2 * calculator._design.shape[1],) * 2),
                    np.zeros((2 * calculator._design.shape[1],) * 2),
                    u_gram,
                )
                for _ in outcomes
            )
            if component_contrasts is None
            else tuple(
                _bm_reductions(
                    calculator._design,
                    calculator._weights,
                    calculator._design_factor,
                    contrast,
                    scale=common_scale,
                    q_design=calculator._design_q,
                    u_gram=u_gram,
                )
                for contrast, common_scale in component_contrasts
            )
        )
        s2_rows.append(tuple(item[0] for item in reductions))
        s4_rows.append(tuple(item[1] for item in reductions))
        g_rows.append(
            tuple(
                tuple(tuple(float(value) for value in row) for row in item[2])
                for item in reductions
            )
        )
        g2_rows.append(
            tuple(
                tuple(tuple(float(value) for value in row) for row in item[3])
                for item in reductions
            )
        )
    specs = _projection_specs(identity, cached, calculator, order)
    return InferenceSummary(
        fit_identity=identity,
        projection_identity=identity + ("all-pairs-v1",),
        max_leverage=float(leverage.max()),
        sum_leverage=float(leverage.sum()),
        refusal_witness=(),
        adjusted_component_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        scaled_projected_score_squares=tuple(projected_scores),
        s2=tuple(s2_rows),
        s4=tuple(s4_rows),
        g=tuple(g_rows),
        g2=tuple(g2_rows),
        u=tuple(tuple(float(value) for value in row) for row in u_gram),
        projection_specs=specs,
        evidence_identity=identity,
    )


def _projection_specs(identity, cached, calculator, order):
    specs = []
    narm = len(order)
    for kind, arm, control, projection_scale, estimate, se, dof in cached:
        available = bool(np.isfinite((estimate, se, dof)).all())
        scale_metadata, divisor_metadata, exponent_metadata = 1.0, 1.0, 0
        if not available:
            jacobian = [0.0] * (2 * narm if kind == "ratio" else narm)
        elif kind == "absolute":
            jacobian = [0.0] * narm
            jacobian[order.index(arm)] = 1.0
            jacobian[order.index(control)] = -1.0
        elif kind == "relative":
            ti, ci = order.index(arm), order.index(control)
            treatment_mean = calculator._usable(0, ti, arm, "outcome")
            control_mean = calculator._usable(0, ci, control, "outcome")
            (
                jacobian,
                scale_metadata,
                divisor_metadata,
                exponent_metadata,
            ) = calculator._scaled_jacobian_details(
                narm,
                ((ti, treatment_mean, 1.0), (ci, control_mean, -1.0)),
            )
        else:
            ti, ci = order.index(arm), order.index(control)
            nt = calculator._usable(0, ti, arm, "numerator")
            nd = calculator._usable(1, ti, arm, "denominator")
            nc = calculator._usable(0, ci, control, "numerator")
            cd = calculator._usable(1, ci, control, "denominator")
            if projection_scale == "log":
                terms = ((ti, nt, 1.0), (ci, nc, -1.0), (narm + ti, nd, -1.0), (narm + ci, cd, 1.0))
            else:
                rt, rc = _rounded_ratio(nt, nd), _rounded_ratio(nc, cd)
                terms = ((ti, nd, 1.0), (ci, cd, -1.0), (narm + ti, nd, -rt), (narm + ci, cd, rc))
            (
                jacobian,
                scale_metadata,
                divisor_metadata,
                exponent_metadata,
            ) = calculator._scaled_jacobian_details(2 * narm, terms)
        specs.append(
            ProjectionSpec(
                identity=identity + (kind, arm, control, projection_scale),
                kind=kind,
                jacobian=tuple(float(value) for value in jacobian),
                scale=scale_metadata,
                divisor=divisor_metadata,
                exponent=exponent_metadata,
                arm=arm,
                control=control,
                available=available and bool(np.isfinite(jacobian).all()),
            )
        )
    return tuple(specs)


def _input_identity(groups, outcomes, covariates) -> str:
    digest = hashlib.sha256()

    def update(name, values):
        array = np.asarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        if array.dtype.kind in "biufc":
            digest.update(np.ascontiguousarray(array).tobytes())
        else:
            for value in array:
                encoded = (
                    type(value).__module__ + "." + type(value).__qualname__ + ":" + repr(value)
                ).encode("utf-8", errors="backslashreplace")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)

    update("arms", groups)
    for index, outcome in enumerate(outcomes):
        update(f"outcome:{index}", outcome)
    for name, values in covariates.items():
        update(f"covariate:{name}", values)
    return digest.hexdigest()


def _projection_bm_contrasts(calculator, row):
    kind, arm, control, scale, estimate, se, dof = row
    if not np.isfinite((estimate, se, dof)).all():
        return None
    ti, ci = calculator._pair(arm, control)
    narm = len(calculator.arm_order)
    if kind == "absolute":
        jacobians = (calculator._contrast(arm, control),)
    elif kind == "relative":
        mt = calculator._usable(0, ti, arm, "outcome")
        mc = calculator._usable(0, ci, control, "outcome")
        jacobians = (calculator._scaled_jacobian(narm, ((ti, mt, 1.0), (ci, mc, -1.0))),)
    else:
        nd = calculator._usable(1, ti, arm, "denominator")
        cd = calculator._usable(1, ci, control, "denominator")
        nt = calculator._usable(0, ti, arm, "numerator")
        nc = calculator._usable(0, ci, control, "numerator")
        if scale == "log":
            terms = ((ti, nt, 1.0), (ci, nc, -1.0), (narm + ti, nd, -1.0), (narm + ci, cd, 1.0))
        else:
            terms = (
                (ti, nd, 1.0),
                (ci, cd, -1.0),
                (narm + ti, nd, -_rounded_ratio(nt, nd)),
                (narm + ci, cd, _rounded_ratio(nc, cd)),
            )
        jacobian = calculator._scaled_jacobian(2 * narm, terms)
        jacobians = (jacobian[:narm], jacobian[narm:])
    contrasts = tuple(
        np.concatenate((jacobian, -calculator._contrast_x(jacobian))) for jacobian in jacobians
    )
    common_scale = max(float(np.max(np.abs(value), initial=0.0)) for value in contrasts)
    return tuple((contrast, common_scale) for contrast in contrasts)


def _validate_request(
    y, arms, control, arm_order, covariates, kinds, missing, numerator, denominator
):
    groups = np.asarray(arms, dtype=object)
    order = tuple(arm_order)
    if groups.ndim != 1:
        _fail("estimation.cuped_linear.input_shape", name="arms", n=len(groups), shape=groups.shape)
    observed = tuple(dict.fromkeys(groups))
    if (
        len(order) < 2
        or len(set(order)) != len(order)
        or control not in order
        or set(observed) != set(order)
    ):
        _fail(
            "estimation.cuped_linear.invalid_arms",
            control=control,
            arm_order=order,
            observed=observed,
        )
    indices = tuple(np.flatnonzero(groups == arm) for arm in order)
    counts = tuple(len(idx) for idx in indices)
    if min(counts, default=0) < 2:
        _fail("estimation.cuped_linear.arm_counts", counts=dict(zip(order, counts, strict=True)))
    if numerator is not None or denominator is not None:
        if numerator is None or denominator is None or y is not None:
            _fail("estimation.cuped_linear.outcome_components")
        outcomes = (np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    else:
        if y is None:
            _fail("estimation.cuped_linear.outcome_components")
        outcomes = (np.asarray(y, dtype=float),)
    for family, values in enumerate(outcomes):
        if values.shape != (len(groups),):
            _fail(
                "estimation.cuped_linear.input_shape",
                name="outcome component",
                n=len(groups),
                shape=values.shape,
            )
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            name = ("numerator" if family == 0 else "denominator") if len(outcomes) == 2 else "y"
            _fail("estimation.cuped_linear.nonfinite", name=name, row=int(bad[0]))
    covariates = {} if covariates is None else dict(covariates)
    kinds = {} if kinds is None else dict(kinds)
    missing = {} if missing is None else dict(missing)
    if set(covariates) != set(kinds) or not set(missing).issubset(covariates):
        _fail(
            "estimation.cuped_linear.covariate_input", values=tuple(covariates), kinds=tuple(kinds)
        )
    return groups, order, indices, counts, outcomes, covariates, kinds, missing


def _center_outcomes(outcomes, indices):
    offsets = []
    centered_outcomes = []
    for outcome in outcomes:
        centered = np.empty_like(outcome)
        arm_offsets = []
        for idx in indices:
            offset = float(np.partition(outcome[idx], len(idx) // 2)[len(idx) // 2])
            values = outcome[idx] - offset
            if not np.isfinite(values).all():
                offset = 0.0
                values = outcome[idx]
            centered[idx] = values
            arm_offsets.append(offset)
        centered_outcomes.append(centered)
        offsets.append(tuple(arm_offsets))
    return tuple(centered_outcomes), tuple(offsets)


def _fit_covariate_slopes(x, indices, centered_outcomes, local_means, layout):
    n = len(x)
    means = tuple(x[idx].mean(axis=0) for idx in indices)
    weights = np.empty(n, dtype=float)
    for idx, mean in zip(indices, means, strict=True):
        x[idx] -= mean
        weights[idx] = 1 / (len(idx) * (len(idx) - 1))
    sqrt_weights = np.sqrt(weights)
    p = x.shape[1]
    if p:
        np.multiply(x, sqrt_weights[:, None], out=x)
        scales = np.linalg.norm(x, axis=0)
        if not np.isfinite(scales).all() or np.any(scales == 0):
            deficient_columns = tuple(
                layout.encoded_columns[i]
                for i, scale in enumerate(scales)
                if not np.isfinite(scale) or scale == 0
            )
            _fail(
                "estimation.cuped_linear.rank_deficiency",
                columns=deficient_columns,
                partners=(),
            )
        np.divide(x, scales, out=x)
        if n <= 1024:
            q_covariate, r_scaled = np.linalg.qr(x, mode="reduced")
        else:
            q_covariate, r_scaled = scipy_qr(
                x, mode="economic", overwrite_a=True, check_finite=False
            )
        _, r_pivoted, pivots = scipy_qr(
            r_scaled, mode="economic", pivoting=True, check_finite=False
        )
        pivots = np.asarray(pivots)
        pivot_squares = np.diag(r_pivoted) ** 2
        rank = int(np.count_nonzero(pivot_squares > p * np.finfo(float).eps))
        if rank < p:
            _fail(
                "estimation.cuped_linear.rank_deficiency",
                columns=(layout.encoded_columns[int(pivots[rank])],),
                partners=tuple(layout.encoded_columns[int(i)] for i in pivots[:rank]),
            )
        r_covariate = r_scaled * scales[None, :]
        gram_factor = r_covariate
    else:
        q_covariate = np.empty((n, 0))
        r_scaled = np.empty((0, 0))
        scales = np.empty(0)
        gram_factor = np.empty((0, 0))
    anchor = (
        sum((len(idx) / n) * mean for idx, mean in zip(indices, means, strict=True))
        if p
        else np.empty(0)
    )
    slopes = []
    for outcome, arm_means in zip(centered_outcomes, local_means, strict=True):
        outcome_centered = np.empty(n)
        for idx, mean in zip(indices, arm_means, strict=True):
            outcome_centered[idx] = outcome[idx] - mean
        if p:
            slopes.append(
                np.linalg.solve(r_scaled, q_covariate.T @ (sqrt_weights * outcome_centered))
                / scales
            )
        else:
            slopes.append(np.empty(0))
    return (
        means,
        weights,
        sqrt_weights,
        q_covariate,
        r_scaled,
        scales,
        gram_factor,
        tuple(slopes),
        anchor,
    )


def _raw_denominator_precision(numerator, denominator, arm):
    """Apply the raw-arm denominator statistic on scale-normalized moments.

    The statistic shares ``ratio_moments``' domain, so an arm whose raw
    denominator mean is not positive is refused here under this fit's own
    code rather than the variance helper's. The sign is judged on the
    normalized scale, where no sum can overflow; the retained value is the
    raw mean recovered from it.
    """
    normalized = []
    scales = []
    for values in (numerator, denominator):
        scale = float(np.max(np.abs(values)))
        normalized.append(values / scale if scale else values)
        scales.append(scale)
    num, den = normalized
    ref_num, ref_den = float(num.mean()), float(den.mean())
    if ref_den <= 0:
        _fail(
            "estimation.cuped_linear.nonpositive_component",
            family="denominator",
            arm=arm,
            value=ref_den * scales[1],
        )
    centered_num, centered_den = num - ref_num, den - ref_den
    stats = ArmStats(
        study_id="cuped_linear",
        metric="ratio",
        group_id=str(arm),
        n=len(den),
        ref_y=ref_num,
        cy1=math.fsum(centered_num),
        cy2=float(centered_num @ centered_num),
        ref_den=ref_den,
        cden1=math.fsum(centered_den),
        cden2=float(centered_den @ centered_den),
        cyden=float(centered_num @ centered_den),
    )
    return ratio_denominator_precision(stats)


def _fit_calculator(groups, order, counts, outcomes, x, layout, control, coordinate_offsets):
    n = len(groups)
    indices = tuple(np.flatnonzero(groups == arm) for arm in order)
    centered_outcomes, outcome_offsets = _center_outcomes(outcomes, indices)
    local_means = tuple(
        tuple(_arm_mean(outcome[idx]) for idx in indices) for outcome in centered_outcomes
    )
    means, weights, sqrt_weights, q_covariate, r_scaled, scales, gram_factor, slopes, anchor = (
        _fit_covariate_slopes(x, indices, centered_outcomes, local_means, layout)
    )
    p = x.shape[1]
    arm_count = len(order)
    design = np.zeros((n, arm_count + p))
    arm_index = np.empty(n, dtype=int)
    for arm, idx in enumerate(indices):
        arm_index[idx] = arm
        design[idx, arm] = 1
    x = design[:, arm_count:]
    if p:
        np.matmul(q_covariate, r_scaled, out=x)
        x *= scales
        x /= sqrt_weights[:, None]
    offset_anchor = (
        sum((counts[a] / n) * coordinate_offsets[a] for a in range(arm_count)) if p else np.empty(0)
    )
    arm_differences = tuple(
        np.fromiter(
            (
                math.fsum(
                    (
                        float(means[a][k]),
                        -float(anchor[k]),
                        float(coordinate_offsets[a, k]),
                        -float(offset_anchor[k]),
                    )
                )
                for k in range(p)
            ),
            dtype=float,
            count=p,
        )
        for a in range(arm_count)
    )
    mean_parts = tuple(
        tuple(
            (
                local_means[f][a],
                -float(slopes[f] @ arm_differences[a]),
                outcome_offsets[f][a],
            )
            for a in range(arm_count)
        )
        for f in range(len(outcomes))
    )
    rounded_means = tuple(tuple(_exact_sum(parts) for parts in family) for family in mean_parts)
    leverage = np.empty(n)
    if n <= 1024:
        q_design, r_design = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
    else:
        q_design = np.empty_like(design)
        q_design[:, :arm_count] = 0.0
        for arm, idx in enumerate(indices):
            q_design[idx, arm] = 1 / math.sqrt(len(idx))
        q_design[:, arm_count:] = q_covariate
        r_design = np.zeros((arm_count + p, arm_count + p))
        for arm, idx in enumerate(indices):
            r_design[arm, arm] = math.sqrt(1 / (len(idx) - 1))
        r_design[arm_count:, arm_count:] = gram_factor
    bm_u_gram = _bm_u_gram(q_design, weights)
    for idx in indices:
        leverage[idx] = 1 / len(idx) + np.sum(q_covariate[idx] ** 2, axis=1)
    df = n - len(order) - p
    if df <= 0:
        _fail("estimation.cuped_linear.residual_df", df=df)
    if leverage.max() >= 1 - 1e-8:
        _fail(
            "estimation.cuped_linear.leverage",
            row=int(leverage.argmax()),
            leverage=float(leverage.max()),
        )
    categorical_levels = max(
        (
            len(levels)
            for kind, levels in zip(layout.kinds, layout.levels, strict=True)
            if kind == "categorical" and levels is not None
        ),
        default=0,
    )
    violations = []
    for observable, value, bound in (
        ("p", float(p), float(M1_MAX_ENCODED_COVARIATES)),
        ("k_max", float(categorical_levels), float(M1_MAX_CATEGORICAL_LEVELS)),
        ("leverage_ratio", float(p / min(counts)), float(M1_MAX_LEVERAGE_RATIO)),
        ("max_leverage", float(leverage.max()), float(M1_MAX_LEVERAGE)),
    ):
        if value > bound:
            violations.append((observable, value, bound))
    if len(outcomes) == 2:
        precision = max(
            _raw_denominator_precision(outcomes[0][idx], outcomes[1][idx], order[a])
            for a, idx in enumerate(indices)
        )
        if precision > M1_MAX_RATIO_DENOMINATOR_PRECISION:
            violations.append(
                (
                    "ratio_denominator_precision",
                    _canonical_float(precision),
                    float(M1_MAX_RATIO_DENOMINATOR_PRECISION),
                )
            )
    if violations:
        _fail(
            "estimation.cuped_linear.support_region",
            region="M1",
            violations=tuple(violations),
        )
    components = tuple(
        tuple(_readonly(outcome[idx]) for idx in indices) for outcome in centered_outcomes
    )
    residuals = tuple(
        tuple(
            _readonly(outcome[idx] - local_means[f][a] - x[idx] @ slopes[f])
            for a, idx in enumerate(indices)
        )
        for f, outcome in enumerate(centered_outcomes)
    )
    calculator = _LinearCupedCalculator(
        order,
        control,
        counts,
        tuple(mean for family in rounded_means for mean in family),
        tuple(float(v) for slope in slopes for v in slope),
        tuple(float(v) for v in anchor),
        layout,
        (),
        0.0,
        min(counts),
        p,
        float(leverage.max()),
        float(leverage.mean()),
        _readonly(leverage),
        _readonly(x),
        _readonly(arm_index),
        components,
        tuple(_readonly(slope) for slope in slopes),
        residuals,
        tuple(_readonly(mean) for mean in means),
        tuple(_readonly(row) for row in coordinate_offsets),
        _readonly(offset_anchor),
        _readonly(gram_factor),
        _readonly(weights),
        _readonly(design),
        _readonly(r_design),
        _readonly(q_design),
        _readonly(bm_u_gram),
        tuple(parts for family in mean_parts for parts in family),
        tuple(
            tuple(
                _row_exponent(rounded_means[f][a], residuals[f][a], x[idx] @ slopes[f])
                for a, idx in enumerate(indices)
            )
            for f in range(len(outcomes))
        ),
    )
    return calculator, leverage


def _wls_design(groups, order, x, means, p):
    design = np.zeros((len(groups), len(order) + p))
    arm_index = np.empty(len(groups), dtype=int)
    for a, arm in enumerate(order):
        idx = np.flatnonzero(groups == arm)
        arm_index[idx] = a
        design[idx, a] = 1
        design[idx, len(order) :] = x[idx]
    return design, arm_index
