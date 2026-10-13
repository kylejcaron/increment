"""Fixed-horizon mixed-covariate CUPED with fitted-slope HC2 inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NoReturn

import numpy as np
import numpy.typing as npt
from scipy.linalg import qr as scipy_qr

from increment.errors import InvalidRequestError, RefusalSpec, refusals, refuse
from increment.estimation._adjust.encoding import is_null
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.armstats import ArmStats
from increment.estimation.variance import ratio_denominator_precision, stable_log_ratio

M1_MAX_ENCODED_COVARIATES = 203
M1_MAX_CATEGORICAL_LEVELS = 200
M1_MAX_LEVERAGE_RATIO = 0.20
M1_MAX_LEVERAGE = 0.45
M1_MAX_RATIO_DENOMINATOR_PRECISION = 0.15

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
    },
)


def _fail(code: str, **context: object) -> NoReturn:
    refuse(_REFUSALS[code], **context)


class CategoricalMissing(Enum):
    """A pickle-stable categorical null level that cannot equal a user label."""

    VALUE = "typed missing category"


MISSING_CATEGORY = CategoricalMissing.VALUE


@dataclass(frozen=True, slots=True)
class LinearCupedLayout:
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    levels: tuple[tuple[object, ...] | None, ...]
    references: tuple[object, ...]
    encoded_columns: tuple[str, ...]
    structural_omissions: tuple[str, ...]
    shared_indicators: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class LinearCupedProjection:
    estimate: float
    se: float
    dof: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class _LinearCupedCalculator:
    arm_order: tuple[object, ...]
    control: object
    n_by_arm: tuple[int, ...]
    adjusted_means: tuple[float, ...]
    theta: tuple[float, ...]
    anchor: tuple[float, ...]
    layout: LinearCupedLayout
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
    _adjusted_local: tuple[float, ...]
    _outcome_offsets: tuple[tuple[float, ...], ...]

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
    ) -> LinearCupedProjection:
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

    def _contrast_value(self, j: np.ndarray, family: int = 0) -> float:
        narm = len(self.arm_order)
        local = self._adjusted_local[family * narm : (family + 1) * narm]
        offsets = self._outcome_offsets[family]
        return math.fsum(
            [float(j[a]) * local[a] for a in range(narm)]
            + [float(j[a]) * offsets[a] for a in range(narm)]
        )

    def _contrast_statistics(self, j: np.ndarray) -> tuple[float, float, float]:
        return _projection_statistics(
            self._contrast_value(j),
            _stable_norm(self._family_score(0, j)),
            self._bm_dof(j),
        )

    def _log_relative_value(self, ti: int, ci: int, family: int = 0) -> float:
        narm = len(self.arm_order)
        local = self._adjusted_local[family * narm : (family + 1) * narm]
        offsets = self._outcome_offsets[family]
        mt = self.adjusted_means[family * narm + ti]
        mc = self.adjusted_means[family * narm + ci]
        difference = math.fsum((local[ti], offsets[ti], -local[ci], -offsets[ci]))
        if min(mt, mc) * 2 >= max(mt, mc):
            return math.log1p(difference / mc)
        return stable_log_ratio(mc, mt)

    def relative(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> LinearCupedProjection:
        """Log adjusted-mean ratio with a plug-in Jacobian and t interval."""
        _check_alpha(alpha)
        return _interval(*self._relative_statistics(arm, control), alpha)

    def _relative_statistics(
        self, arm: object, control: object | None
    ) -> tuple[float, float, float]:
        jidx = self._pair(arm, control)
        means = np.asarray(self.adjusted_means)[: len(self.arm_order)]
        mt, mc = means[jidx[0]], means[jidx[1]]
        self._positive(mt, arm, "outcome")
        self._positive(mc, self.control if control is None else control, "outcome")
        jac = np.zeros(len(means))
        jac[jidx[0]] = 1 / mt
        jac[jidx[1]] = -1 / mc
        return _projection_statistics(
            self._log_relative_value(*jidx),
            _stable_norm(self._family_score(0, jac)),
            self._bm_dof(jac),
        )

    def ratio(
        self,
        arm: object,
        *,
        control: object | None = None,
        scale: Literal["log", "absolute"] = "log",
        alpha: float = 0.05,
    ) -> LinearCupedProjection:
        """Ratio-of-component projection retaining stacked numerator/denominator scores."""
        _check_alpha(alpha)
        return _interval(*self._ratio_statistics(arm, control, scale), alpha)

    def _ratio_statistics(
        self,
        arm: object,
        control: object | None,
        scale: Literal["log", "absolute"],
    ) -> tuple[float, float, float]:
        if not self._components:
            _fail("estimation.cuped_linear.projection_kind", requested="ratio", available="mean")
        ti, ci = self._pair(arm, control)
        means = np.asarray(self.adjusted_means).reshape(2, -1)
        nt, nd = means[0, ti], means[1, ti]
        nc, cd = means[0, ci], means[1, ci]
        self._positive(nd, arm, "denominator")
        self._positive(cd, self.arm_order[ci], "denominator")
        self._positive(nt, arm, "numerator")
        self._positive(nc, self.arm_order[ci], "numerator")
        if scale == "log":
            jac = np.zeros(2 * len(self.arm_order))
            jac[ti], jac[len(self.arm_order) + ti] = 1 / nt, -1 / nd
            jac[ci], jac[len(self.arm_order) + ci] = -1 / nc, 1 / cd
            value = _stable_log_cross_ratio(nt, nd, nc, cd)
        elif scale == "absolute":
            rt, rc = nt / nd, nc / cd
            jac = np.zeros(2 * len(self.arm_order))
            jac[ti], jac[len(self.arm_order) + ti] = 1 / nd, -rt / nd
            jac[ci], jac[len(self.arm_order) + ci] = -1 / cd, rc / cd
            value = rt - rc
        else:
            _fail("estimation.cuped_linear.invalid_scale", scale=scale)
        narm = len(self.arm_order)
        scores = [
            self._family_score(family, jac[family * narm : (family + 1) * narm])
            for family in range(2)
        ]
        return _projection_statistics(
            value, _stable_norm(scores[0] + scores[1]), self._stacked_bm(jac)
        )

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

    def _positive(self, value: float, arm: object, family: str) -> None:
        if not value > 0:
            _fail(
                "estimation.cuped_linear.nonpositive_component", family=family, arm=arm, value=value
            )

    def _family_score(self, family: int, j: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(j), initial=0.0))
        if scale:
            j = j / scale
        d = self._contrast_x(j)
        contrast = np.concatenate([j, -d])
        coefficients = np.linalg.solve(
            self._design_factor,
            np.linalg.solve(self._design_factor.T, contrast),
        )
        residual = np.empty(len(self._x))
        anchor_score = np.empty(len(self._x))
        for a, scores in enumerate(self._residuals[family]):
            mask = self._arm_index == a
            residual[mask] = scores
            anchor_score[mask] = (
                float(j.sum()) / sum(self.n_by_arm) * (self._x[mask] @ self._slopes[family])
            )
        influence = self._design @ coefficients * self._weights
        return (influence * residual / np.sqrt(1 - self.leverage) + anchor_score) * scale

    def _bm_dof(self, j: np.ndarray) -> float:
        d = self._contrast_x(j)
        return _bm(
            self._design,
            self._weights,
            self.leverage,
            self._design_factor,
            np.concatenate([j, -d]),
            q_design=self._design_q,
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
            )
            for contrast in contrasts
        ]
        trace = math.fsum(item[0] for item in terms)
        trace_square = math.fsum(item[1] for item in terms)
        return trace * trace / trace_square


@dataclass(frozen=True, slots=True)
class LinearCupedFit:
    """Immutable fitted evidence; unit-level input arrays are not retained."""

    arm_order: tuple[object, ...]
    control: object
    n_by_arm: tuple[int, ...]
    adjusted_means: tuple[float, ...]
    theta: tuple[float, ...]
    anchor: tuple[float, ...]
    layout: LinearCupedLayout
    covariance: tuple[tuple[float, ...], ...]
    dof: float
    n_min: int
    p: int
    max_leverage: float
    mean_leverage: float
    leverage_trace: float
    is_ratio: bool
    _projections: tuple[tuple[str, object, object, str, float, float, float], ...]

    def absolute(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> LinearCupedProjection:
        return self._lookup("absolute", arm, control, "", alpha)

    def relative(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> LinearCupedProjection:
        ti, ci = self._pair(arm, control)
        means = np.asarray(self.adjusted_means)
        if self.is_ratio:
            _fail(
                "estimation.cuped_linear.projection_kind", requested="relative", available="ratio"
            )
        self._positive(means[ti], arm, "outcome")
        self._positive(means[ci], self.arm_order[ci], "outcome")
        return self._lookup("relative", arm, control, "", alpha)

    def ratio(
        self,
        arm: object,
        *,
        control: object | None = None,
        scale: Literal["log", "absolute"] = "log",
        alpha: float = 0.05,
    ) -> LinearCupedProjection:
        if not self.is_ratio:
            _fail("estimation.cuped_linear.projection_kind", requested="ratio", available="mean")
        ti, ci = self._pair(arm, control)
        means = np.asarray(self.adjusted_means).reshape(2, -1)
        for family, values in (("numerator", means[0]), ("denominator", means[1])):
            self._positive(values[ti], arm, family)
            self._positive(values[ci], self.arm_order[ci], family)
        if scale not in ("log", "absolute"):
            _fail("estimation.cuped_linear.invalid_scale", scale=scale)
        return self._lookup("ratio", arm, control, scale, alpha)

    def _pair(self, arm: object, control: object | None) -> tuple[int, int]:
        try:
            ti = self.arm_order.index(arm)
            ci = self.arm_order.index(self.control if control is None else control)
        except ValueError:
            _fail("estimation.cuped_linear.projection_arm", arm=arm, control=control)
        if ti == ci:
            _fail("estimation.cuped_linear.projection_arm", arm=arm, control=control)
        return ti, ci

    def _positive(self, value: float, arm: object, family: str) -> None:
        if not value > 0:
            _fail(
                "estimation.cuped_linear.nonpositive_component", family=family, arm=arm, value=value
            )

    def _lookup(
        self, kind: str, arm: object, control: object | None, scale: str, alpha: float
    ) -> LinearCupedProjection:
        _check_alpha(alpha)
        ci = self.control if control is None else control
        for entry in self._projections:
            if entry[:4] == (kind, arm, ci, scale):
                _, _, _, _, estimate, se, dof = entry
                return _interval(estimate, se, dof, alpha)
        _fail("estimation.cuped_linear.projection_unavailable", kind=kind, arm=arm, control=ci)


def fit_cuped_linear(
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
) -> LinearCupedFit:
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
    public_anchor = tuple(
        math.fsum(
            [calculator.anchor[i] * scales[i]]
            + [counts[a] / sum(counts) * offsets[i][a] for a in range(len(order))]
        )
        for i in range(len(scales))
    )
    public_theta = (
        tuple(value / scales[i % len(scales)] for i, value in enumerate(calculator.theta))
        if scales
        else ()
    )
    covariance = _mean_covariance(calculator)
    object.__setattr__(
        calculator, "covariance", tuple(tuple(float(v) for v in row) for row in covariance)
    )
    contrast = np.zeros(len(order))
    contrast[order.index(next(a for a in order if a != control))] = 1
    contrast[order.index(control)] = -1
    object.__setattr__(calculator, "dof", calculator._bm_dof(contrast))
    cached = _cache_projections(calculator, outcomes, order)
    return LinearCupedFit(
        order,
        control,
        counts,
        calculator.adjusted_means,
        public_theta,
        public_anchor,
        layout,
        calculator.covariance,
        calculator.dof,
        min(counts),
        calculator.p,
        float(leverage.max()),
        float(leverage.mean()),
        float(leverage.sum()),
        len(outcomes) == 2,
        cached,
    )


def _validate_request(
    y, arms, control, arm_order, covariates, kinds, missing, numerator, denominator
):
    groups = np.asarray(arms, dtype=object)
    order = tuple(arm_order)
    if groups.ndim != 1:
        _fail("estimation.cuped_linear.input_shape", name="arms", n=len(groups), shape=groups.shape)
    observed = tuple(dict.fromkeys(groups.tolist()))
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


def _normalized_offset_difference(offset, reference, scale):
    exponent = max(math.frexp(abs(value))[1] for value in (offset, reference, scale) if value != 0)
    numerator = math.fsum((math.ldexp(offset, -exponent), -math.ldexp(reference, -exponent)))
    denominator = math.ldexp(scale, -exponent)
    if denominator == 0:
        if numerator == 0:
            return 0.0
        _fail(
            "estimation.cuped_linear.nonfinite",
            name="covariate offset",
            row=0,
        )
    result = numerator / denominator
    if not math.isfinite(result):
        _fail(
            "estimation.cuped_linear.nonfinite",
            name="covariate offset",
            row=0,
        )
    return result


def _categorical_values(raw, raw_values, n):
    if isinstance(raw, np.ndarray) and raw.dtype.kind in "US":
        values = raw_values.astype(str) if raw_values.dtype.kind == "S" else raw_values
        return values, np.zeros(n, bool)
    values = np.asarray(raw, dtype=object)
    return values, np.fromiter((is_null(value) for value in values), bool, n)


def _encode_covariates(groups, indices, order, covariates, kinds, missing):
    n = len(groups)
    columns, column_names, transforms = [], [], []
    structural, references, levels = [], [], []
    masks: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    names = tuple(covariates)
    for name, raw in covariates.items():
        if kinds[name] not in ("continuous", "categorical"):
            _fail("estimation.cuped_linear.covariate_kind", name=name, kind=kinds[name])
        raw_values = np.asarray(raw)
        if raw_values.shape != (n,):
            _fail("estimation.cuped_linear.input_shape", name=name, n=n, shape=raw_values.shape)
        if kinds[name] == "categorical":
            values, null = _categorical_values(raw, raw_values, n)
        elif kinds[name] == "continuous" and raw_values.dtype.kind in "biuf":
            values = raw_values
            null = np.isnan(raw_values) if raw_values.dtype.kind == "f" else np.zeros(n, bool)
        else:
            values = np.asarray(raw, dtype=object)
            null = np.fromiter((is_null(value) for value in values), bool, n)
        policy = missing.get(name, "refuse")
        if policy not in ("refuse", "impute-indicator") or (null.any() and policy == "refuse"):
            _fail("estimation.cuped_linear.missing_refused", name=name)
        if kinds[name] == "continuous":
            transform = _encode_numeric(
                name,
                values,
                null,
                indices,
                columns,
                column_names,
                structural,
                references,
                levels,
                masks,
            )
            if transform is not None:
                transforms.append(transform)
        else:
            before = len(columns)
            _encode_category(
                name,
                values,
                null,
                indices,
                order,
                columns,
                column_names,
                structural,
                references,
                levels,
            )
            transforms.extend([((0.0,) * len(order), 1.0)] * (len(columns) - before))
    shared = []
    for indicator, owners in masks.values():
        if np.ptp(indicator) == 0:
            continue
        columns.append(indicator)
        column_names.append(f"missing[{','.join(owners)}]")
        transforms.append(((0.0,) * len(order), 1.0))
        shared.append(tuple(owners))
    matrix = np.empty((n, len(columns)), dtype=float)
    j = 0
    while j < len(columns):
        column = columns[j]
        if isinstance(column, tuple):
            codes = column[0]
            stop = j + 1
            while stop < len(columns) and isinstance(columns[stop], tuple):
                if columns[stop][0] is not codes:
                    break
                stop += 1
            selected = np.fromiter(
                (columns[i][1] for i in range(j, stop)), dtype=int, count=stop - j
            )
            np.equal(codes[:, None], selected[None, :], out=matrix[:, j:stop])
            j = stop
        else:
            matrix[:, j] = column
            j += 1
    x = matrix
    encoded_names = tuple(column_names)
    active_transforms = tuple(transforms)
    layout = LinearCupedLayout(
        names,
        tuple(kinds[name] for name in names),
        tuple(levels),
        tuple(references),
        encoded_names,
        tuple(structural),
        tuple(shared),
    )
    return (
        x,
        layout,
        tuple(item[0] for item in active_transforms),
        tuple(item[1] for item in active_transforms),
    )


def _encode_numeric(
    name, values, null, indices, columns, column_names, structural, references, levels, masks
):
    if values.dtype.kind in "biuf":
        numeric = np.asarray(values, dtype=float)
        if null.any():
            numeric = np.where(null, 0.0, numeric)
    else:
        numeric = np.zeros(len(values), dtype=float)
        for i, value in enumerate(values):
            if not null[i]:
                try:
                    numeric[i] = float(value)
                except (TypeError, ValueError, OverflowError):
                    _fail(
                        "estimation.cuped_linear.covariate_kind",
                        name=name,
                        kind=type(value).__name__,
                    )
    if not np.isfinite(numeric).all():
        _fail(
            "estimation.cuped_linear.nonfinite",
            name=name,
            row=int(np.flatnonzero(~np.isfinite(numeric))[0]),
        )
    levels.append(None)
    references.append(0.0)
    if null.any():
        key = null.tobytes()
        if key in masks:
            masks[key][1].append(name)
        else:
            masks[key] = (null.astype(float), [name])
    offsets = []
    centered = np.empty_like(numeric)
    for group in indices:
        present = numeric[group[~null[group]]]
        offset = float(present[0]) if len(present) else 0.0
        offsets.append(offset)
        centered[group] = numeric[group] - offset
    if not np.isfinite(centered).all():
        offsets = [0.0] * len(indices)
        centered = numeric.copy()
    scale = float(np.max(np.abs(centered), initial=0.0))
    if scale == 0:
        if np.ptp(numeric) == 0:
            structural.append(name)
            return None
        columns.append(centered)
        column_names.append(name)
        return tuple(offsets), 1.0
    columns.append(centered / scale)
    column_names.append(name)
    return tuple(offsets), scale


def _encode_category(
    name, values, null, indices, order, columns, column_names, structural, references, levels
):
    labels_values = values[~null]
    if values.dtype.kind not in "US" and any(not isinstance(value, str) for value in labels_values):
        _fail(
            "estimation.cuped_linear.covariate_kind",
            name=name,
            kind="categorical values must be strings",
        )
    labels_array = np.asarray(labels_values, dtype=str)
    sorted_labels, codes = np.unique(labels_array, return_inverse=True)
    labels = tuple(str(label) for label in sorted_labels)
    full_levels: list[object] = list(labels)
    if null.any():
        full_levels.append(MISSING_CATEGORY)
    codes_full = np.full(len(values), -1, dtype=int)
    codes_full[~null] = codes.astype(int)
    if null.any():
        codes_full[null] = len(labels)
    ref_code = int(np.argmax(np.bincount(codes_full, minlength=len(full_levels))))
    references.append(full_levels[ref_code])
    levels.append(tuple(labels) + ((MISSING_CATEGORY,) if null.any() else ()))
    if len(full_levels) <= 1:
        structural.append(name)
        return
    arm_codes = np.empty(len(values), dtype=int)
    for arm_code, idx in enumerate(indices):
        arm_codes[idx] = arm_code
    per_arm = np.bincount(
        codes_full * len(order) + arm_codes,
        minlength=len(full_levels) * len(order),
    ).reshape(len(full_levels), len(order))
    deficient = np.flatnonzero(np.min(per_arm, axis=1) < 2)
    if len(deficient):
        code = int(deficient[0])
        level = full_levels[code]
        shown = "<typed-missing>" if level is MISSING_CATEGORY else level
        _fail(
            "estimation.cuped_linear.level_support",
            column=name,
            level=shown,
            counts=dict(zip(order, per_arm[code].tolist(), strict=True)),
        )
    for code, level in enumerate(full_levels):
        if code == ref_code:
            continue
        shown = "<typed-missing>" if level is MISSING_CATEGORY else repr(level)
        columns.append((codes_full, code))
        column_names.append(f"{name}[{shown}]")


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


def _fit_covariate_slopes(x, indices, centered_outcomes, layout):
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
    for outcome in centered_outcomes:
        outcome_centered = np.empty(n)
        for idx in indices:
            outcome_centered[idx] = outcome[idx] - outcome[idx].mean()
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
    """Apply the raw-arm denominator statistic on scale-normalized moments."""
    normalized = []
    for values in (numerator, denominator):
        scale = float(np.max(np.abs(values)))
        normalized.append(values / scale if scale else values)
    num, den = normalized
    ref_num, ref_den = float(num.mean()), float(den.mean())
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
    means, weights, sqrt_weights, q_covariate, r_scaled, scales, gram_factor, slopes, anchor = (
        _fit_covariate_slopes(x, indices, centered_outcomes, layout)
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
    adjusted_local = tuple(
        tuple(
            float(outcome[idx].mean() - slopes[f] @ arm_differences[a])
            for a, idx in enumerate(indices)
        )
        for f, outcome in enumerate(centered_outcomes)
    )
    adjusted_families = tuple(
        tuple(adjusted_local[f][a] + outcome_offsets[f][a] for a in range(len(indices)))
        for f in range(len(outcomes))
    )
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
            _readonly(outcome[idx] - outcome[idx].mean() - x[idx] @ slopes[f])
            for a, idx in enumerate(indices)
        )
        for f, outcome in enumerate(centered_outcomes)
    )
    calculator_means = (
        tuple(v for family in adjusted_families for v in family)
        if len(outcomes) == 2
        else adjusted_families[0]
    )
    calculator = _LinearCupedCalculator(
        order,
        control,
        counts,
        calculator_means,
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
        tuple(float(v) for family in adjusted_local for v in family),
        outcome_offsets,
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


def _cache_projections(calculator, outcomes, order):
    cached = []
    for ti in range(1, len(order)):
        treatment = order[ti]
        for ci in range(ti):
            control = order[ci]
            mean_values = np.asarray(calculator.adjusted_means[: len(order)])
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
                components = np.asarray(calculator.adjusted_means).reshape(2, -1)
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


def _mean_covariance(fit: _LinearCupedCalculator) -> np.ndarray:
    narm = len(fit.arm_order)
    nfam = len(fit._components)
    covariance = np.zeros((narm * nfam, narm * nfam))
    scores = []
    for family in range(nfam):
        for arm in range(narm):
            j = np.zeros(narm)
            j[arm] = 1
            scores.append(fit._family_score(family, j))
    for i, left in enumerate(scores):
        for j, right in enumerate(scores):
            covariance[i, j] = left @ right
    return covariance


def _bm(
    design: np.ndarray,
    weights: np.ndarray,
    leverage: np.ndarray,
    factor: np.ndarray,
    contrast: np.ndarray,
    *,
    q_design: np.ndarray | None = None,
) -> float:
    scale = float(np.max(np.abs(contrast), initial=0.0))
    if scale == 0:
        return 1.0
    trace, trace_square = _bm_terms(
        design, weights, leverage, factor, contrast, scale=scale, q_design=q_design
    )
    return trace * trace / trace_square if trace_square > 0 else 1.0


def _bm_terms(design, weights, leverage, factor, contrast, *, scale, q_design=None):
    if len(weights) <= 128:
        sqrt_weights = np.sqrt(weights)
        q, r = np.linalg.qr(design * sqrt_weights[:, None], mode="reduced")
        coefficients = np.linalg.solve(r.T, contrast / scale)
        ai = sqrt_weights * (q @ coefficients)
        hat = np.sum(q * q, axis=1)
        qscore = ai / np.sqrt(1 - hat)
        hat_transpose = sqrt_weights[:, None] * (q @ q.T) / sqrt_weights[None, :]
        projected = (np.eye(len(weights)) - hat_transpose) * qscore[None, :]
        singular = np.linalg.svd(projected, compute_uv=False)
        largest = float(singular.max(initial=0.0))
        if largest == 0:
            return 0.0, 0.0
        normalized = singular / largest
        trace = largest**2 * float(normalized @ normalized)
        trace_square = largest**4 * float((normalized**2) @ (normalized**2))
        return trace, trace_square

    sqrt_weights = np.sqrt(weights)
    if q_design is None:
        weighted_design = np.empty_like(design)
        np.multiply(design, sqrt_weights[:, None], out=weighted_design)
        q, r = scipy_qr(weighted_design, mode="economic", overwrite_a=True, check_finite=False)
    else:
        q, r = q_design, factor
    coefficients = np.linalg.solve(r.T, contrast / scale)
    ai = sqrt_weights * (q @ coefficients)
    leverage = np.einsum("ij,ij->i", q, q)
    qscore = ai / np.sqrt(1 - leverage)
    q2 = qscore * qscore
    q4 = q2 * q2
    parameter_count = design.shape[1]
    feature_count = 2 * parameter_count
    gram = np.zeros((feature_count, feature_count))
    weighted_gram = np.zeros_like(gram)
    basis_gram = np.zeros((parameter_count, parameter_count))
    block_size = 65_536
    for start in range(0, len(weights), block_size):
        stop = min(start + block_size, len(weights))
        sqrt_block = sqrt_weights[start:stop]
        q_block = q[start:stop]
        qscore_block = qscore[start:stop]
        v = q_block * (qscore_block / sqrt_block)[:, None]
        u = q_block * sqrt_block[:, None]
        feature = np.empty((stop - start, feature_count))
        feature[:, :parameter_count] = u * qscore_block[:, None]
        feature[:, parameter_count:] = v
        gram += feature.T @ feature
        weighted_gram += feature.T @ (feature * q2[start:stop, None])
        basis_gram += u.T @ u

    low_rank = np.zeros((feature_count, feature_count))
    low_rank[:parameter_count, parameter_count:] = -np.eye(parameter_count)
    low_rank[parameter_count:, :parameter_count] = -np.eye(parameter_count)
    low_rank[parameter_count:, parameter_count:] = basis_gram
    trace = float(q2.sum() + np.trace(low_rank @ gram))
    trace_square = float(
        q4.sum()
        + 2 * np.trace(low_rank @ weighted_gram)
        + np.trace(low_rank @ gram @ low_rank @ gram)
    )
    return trace, trace_square


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


def _stable_log_cross_ratio(nt: float, nd: float, nc: float, cd: float) -> float:
    ntm, nte = math.frexp(nt)
    ndm, nde = math.frexp(nd)
    ncm, nce = math.frexp(nc)
    cdm, cde = math.frexp(cd)
    treatment_exponent = nte + cde
    control_exponent = nce + nde
    if abs(treatment_exponent - control_exponent) <= 2:
        treatment_product = _two_product(ntm, cdm)
        control_product = _two_product(ncm, ndm)
        common_exponent = max(treatment_exponent, control_exponent)
        treatment_scale = math.ldexp(1.0, treatment_exponent - common_exponent)
        control_scale = math.ldexp(1.0, control_exponent - common_exponent)
        control = math.fsum(
            (control_product[0] * control_scale, control_product[1] * control_scale)
        )
        difference = math.fsum(
            (
                treatment_product[0] * treatment_scale,
                treatment_product[1] * treatment_scale,
                -control_product[0] * control_scale,
                -control_product[1] * control_scale,
            )
        )
        relative = difference / control
        if abs(relative) <= 0.5:
            return math.log1p(relative)
    return math.fsum((math.log(nt), -math.log(nd), -math.log(nc), math.log(cd)))


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


def _interval(value: float, se: float, dof: float, alpha: float) -> LinearCupedProjection:
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
    return LinearCupedProjection(value, se, dof, lower, upper)
