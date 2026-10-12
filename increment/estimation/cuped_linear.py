"""Fixed-horizon mixed-covariate CUPED with fitted-slope HC2 inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt

from increment.errors import InvalidRequestError, RefusalSpec, refusals, refuse
from increment.estimation._adjust.encoding import is_null, level_codes
from increment.estimation._tails import student_t_isf, two_sided_critical_value

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
        "estimation.cuped_linear.residual_df": "residual degrees of freedom must be positive; got {df}",
        "estimation.cuped_linear.nonpositive_component": "adjusted component {family!r} for arm {arm!r} must be positive for a log or ratio projection; got {value}",
        "estimation.cuped_linear.invalid_alpha": "alpha must be strictly between zero and one; got {alpha}",
    },
)


def _fail(code: str, **context: object) -> None:
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
    _gram_inverse: np.ndarray
    _weights: np.ndarray
    _design: np.ndarray
    _design_inverse: np.ndarray
    _hat: np.ndarray

    def absolute(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> LinearCupedProjection:
        """Adjusted additive arm contrast with an HC2/BM t interval."""
        j = self._contrast(arm, control)
        means = np.asarray(self.adjusted_means)[: len(self.arm_order)]
        return self._project(float(j @ means), j, np.ones(len(j)), alpha)

    def relative(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> LinearCupedProjection:
        """Log adjusted-mean ratio with a plug-in Jacobian and t interval."""
        jidx = self._pair(arm, control)
        means = np.asarray(self.adjusted_means)[: len(self.arm_order)]
        mt, mc = means[jidx[0]], means[jidx[1]]
        self._positive(mt, arm, "outcome")
        self._positive(mc, self.control if control is None else control, "outcome")
        jac = np.zeros(len(means))
        jac[jidx[0]] = 1 / mt
        jac[jidx[1]] = -1 / mc
        return self._project(math.log(mt / mc), jac, np.ones(len(jac)), alpha)

    def ratio(
        self,
        arm: object,
        *,
        control: object | None = None,
        scale: Literal["log", "absolute"] = "log",
        alpha: float = 0.05,
    ) -> LinearCupedProjection:
        """Ratio-of-component projection retaining stacked numerator/denominator scores."""
        if self._components is None:
            raise ValueError("ratio projection requires numerator and denominator components")
        ti, ci = self._pair(arm, control)
        means = np.asarray(self.adjusted_means).reshape(2, -1)
        nt, nd = means[0, ti], means[1, ti]
        nc, cd = means[0, ci], means[1, ci]
        self._positive(nd, arm, "denominator")
        self._positive(cd, self.arm_order[ci], "denominator")
        rt, rc = nt / nd, nc / cd
        self._positive(nt, arm, "numerator")
        self._positive(nc, self.arm_order[ci], "numerator")
        if scale == "log":
            jac = np.zeros(2 * len(self.arm_order))
            jac[ti], jac[len(self.arm_order) + ti] = 1 / nt, -1 / nd
            jac[ci], jac[len(self.arm_order) + ci] = -1 / nc, 1 / cd
            value = math.log(rt / rc)
        elif scale == "absolute":
            jac = np.zeros(2 * len(self.arm_order))
            jac[ti], jac[len(self.arm_order) + ti] = 1 / nd, -nt / nd**2
            jac[ci], jac[len(self.arm_order) + ci] = -1 / cd, nc / cd**2
            value = rt - rc
        else:
            raise ValueError("scale must be 'log' or 'absolute'")
        return self._project_components(value, jac, alpha)

    def _pair(self, arm: object, control: object | None) -> tuple[int, int]:
        try:
            ti = self.arm_order.index(arm)
            ci = self.arm_order.index(self.control if control is None else control)
        except ValueError as exc:
            raise ValueError("projection arm must be declared in arm_order") from exc
        if ti == ci:
            raise ValueError("treatment arm and control must differ")
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

    def _score(self, j: np.ndarray) -> tuple[np.ndarray, float]:
        return self._family_score(0, j)

    def _family_score(self, family: int, j: np.ndarray) -> tuple[np.ndarray, float]:
        theta = self._slopes[family]
        means_x = self._x_means
        anchor = np.asarray(self.anchor)
        d = sum(j[a] * (means_x[a] - anchor) for a in range(len(j)))
        active = np.flatnonzero(j)
        s = np.zeros((len(j), len(theta)))
        if len(active) == 2:
            ti, ci = int(np.argmax(j)), int(np.argmin(j))
            pi = np.asarray(self.n_by_arm, dtype=float) / sum(self.n_by_arm)
            k = j[ti] * pi[ci] - j[ci] * pi[ti]
            s[ti] = theta * k / j[ti]
            s[ci] = -theta * k / j[ci]
        scores = []
        for a, (xbar, outcome, n) in enumerate(
            zip(means_x, self._components[family], self.n_by_arm, strict=True)
        ):
            idx = np.flatnonzero(self._arm_index == a)
            xc = self._x[idx] - xbar
            residual = outcome - outcome.mean() - xc @ theta
            lam = 1.0 / (n * (n - 1))
            c = j[a] / n - lam * (xc @ self._gram_inverse @ d)
            score = c * residual / np.sqrt(1 - self.leverage[idx])
            if len(active) == 1:
                score += (float(j.sum()) / sum(self.n_by_arm)) * (xc @ theta)
            else:
                score += (j[a] / n) * (xc @ (theta - s[a]))
            scores.append(score)
        return np.concatenate(scores), float(d @ d)

    def _project(
        self, value: float, j: np.ndarray, _: np.ndarray, alpha: float
    ) -> LinearCupedProjection:
        _check_alpha(alpha)
        score, _variance = self._score(j)
        return _interval(value, float(np.linalg.norm(score)), self._bm_dof(j), alpha)

    def _bm_dof(self, j: np.ndarray, d_override: np.ndarray | None = None) -> float:
        d = sum(j[a] * (self._x_means[a] - np.asarray(self.anchor)) for a in range(len(j)))
        return _bm(
            self._design,
            self._weights,
            self.leverage,
            self._hat,
            self._design_inverse,
            np.concatenate([j, -d]),
        )

    def _project_components(
        self, value: float, jac: np.ndarray, alpha: float
    ) -> LinearCupedProjection:
        _check_alpha(alpha)
        narm = len(self.arm_order)
        numerator_score, _ = self._family_score(0, jac[:narm])
        denominator_score, _ = self._family_score(1, jac[narm:])
        se = float(np.linalg.norm(numerator_score + denominator_score))
        dof = self._stacked_bm(jac)
        return _interval(value, se, dof, alpha)

    def _stacked_bm(self, jac: np.ndarray) -> float:
        design = np.block(
            [
                [self._design, np.zeros_like(self._design)],
                [np.zeros_like(self._design), self._design],
            ]
        )
        weights = np.tile(self._weights, 2)
        hat = np.block(
            [
                [self._hat, np.zeros_like(self._hat)],
                [np.zeros_like(self._hat), self._hat],
            ]
        )
        lev = np.tile(self.leverage, 2)
        contrast_parts = []
        narm = len(self.arm_order)
        for family in range(2):
            j = jac[family * narm : (family + 1) * narm]
            d = sum(j[a] * (self._x_means[a] - np.asarray(self.anchor)) for a in range(narm))
            contrast_parts.append(np.concatenate([j, -d]))
        return _bm(
            design,
            weights,
            lev,
            hat,
            np.block(
                [
                    [self._design_inverse, np.zeros_like(self._design_inverse)],
                    [np.zeros_like(self._design_inverse), self._design_inverse],
                ]
            ),
            np.concatenate(contrast_parts),
        )


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
            raise ValueError("relative mean projection requires a mean fit")
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
            raise ValueError("ratio projection requires numerator and denominator components")
        ti, ci = self._pair(arm, control)
        means = np.asarray(self.adjusted_means).reshape(2, -1)
        for family, values in (("numerator", means[0]), ("denominator", means[1])):
            self._positive(values[ti], arm, family)
            self._positive(values[ci], self.arm_order[ci], family)
        if scale not in ("log", "absolute"):
            raise ValueError("scale must be 'log' or 'absolute'")
        return self._lookup("ratio", arm, control, scale, alpha)

    def _pair(self, arm: object, control: object | None) -> tuple[int, int]:
        try:
            ti = self.arm_order.index(arm)
            ci = self.arm_order.index(self.control if control is None else control)
        except ValueError as exc:
            raise ValueError("projection arm must be declared in arm_order") from exc
        if ti == ci:
            raise ValueError("treatment arm and control must differ")
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
        raise ValueError("projection was not available in this fit")


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
    x, layout = _encode_covariates(groups, indices, order, covariates, kinds, missing)
    calculator, leverage = _fit_calculator(groups, order, counts, outcomes, x, layout, control)
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
        calculator.theta,
        calculator.anchor,
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
            raise ValueError("supply either y or both numerator and denominator")
        outcomes = (np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    else:
        if y is None:
            raise ValueError("y is required for a mean fit")
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


def _encode_covariates(groups, indices, order, covariates, kinds, missing):
    n = len(groups)
    columns, column_names = [], []
    structural, references, levels = [], [], []
    masks: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    names = tuple(covariates)
    for name, raw in covariates.items():
        if kinds[name] not in ("continuous", "categorical"):
            _fail("estimation.cuped_linear.covariate_kind", name=name, kind=kinds[name])
        values = np.asarray(raw, dtype=object)
        if values.shape != (n,):
            _fail("estimation.cuped_linear.input_shape", name=name, n=n, shape=values.shape)
        null = np.fromiter((is_null(value) for value in values), bool, n)
        policy = missing.get(name, "refuse")
        if policy not in ("refuse", "impute-indicator") or (null.any() and policy == "refuse"):
            _fail("estimation.cuped_linear.missing_refused", name=name)
        if kinds[name] == "continuous":
            _encode_numeric(
                name, values, null, columns, column_names, structural, references, levels, masks
            )
        else:
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
    shared = []
    for indicator, owners in masks.values():
        if np.ptp(indicator) == 0:
            continue
        columns.append(indicator)
        column_names.append(f"missing[{','.join(owners)}]")
        shared.append(tuple(owners))
    matrix = np.column_stack(columns) if columns else np.empty((n, 0), dtype=float)
    active = [j for j in range(matrix.shape[1]) if np.ptp(matrix[:, j]) > 0]
    x = matrix[:, active]
    encoded_names = tuple(column_names[j] for j in active)
    return x, LinearCupedLayout(
        names,
        tuple(kinds[name] for name in names),
        tuple(levels),
        tuple(references),
        encoded_names,
        tuple(structural),
        tuple(shared),
    )


def _encode_numeric(
    name, values, null, columns, column_names, structural, references, levels, masks
):
    numeric = np.zeros(len(values), dtype=float)
    for i, value in enumerate(values):
        if not null[i]:
            try:
                numeric[i] = float(value)
            except (TypeError, ValueError, OverflowError):
                _fail(
                    "estimation.cuped_linear.covariate_kind", name=name, kind=type(value).__name__
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
    if np.ptp(numeric) == 0:
        structural.append(name)
    else:
        columns.append(numeric)
        column_names.append(name)


def _encode_category(
    name, values, null, indices, order, columns, column_names, structural, references, levels
):
    labels_values = [
        value for value, is_missing in zip(values, null, strict=True) if not is_missing
    ]
    if any(not isinstance(value, str) for value in labels_values):
        _fail(
            "estimation.cuped_linear.covariate_kind",
            name=name,
            kind="categorical values must be strings",
        )
    codes, labels = level_codes(labels_values)
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
    for code, level in enumerate(full_levels):
        per_arm = tuple(int(np.sum(codes_full[idx] == code)) for idx in indices)
        if min(per_arm) < 2:
            shown = "<typed-missing>" if level is MISSING_CATEGORY else level
            _fail(
                "estimation.cuped_linear.level_support",
                column=name,
                level=shown,
                counts=dict(zip(order, per_arm, strict=True)),
            )
    for code, level in enumerate(full_levels):
        if code == ref_code:
            continue
        shown = "<typed-missing>" if level is MISSING_CATEGORY else repr(level)
        columns.append((codes_full == code).astype(float))
        column_names.append(f"{name}[{shown}]")


def _fit_calculator(groups, order, counts, outcomes, x, layout, control):
    n = len(groups)
    indices = tuple(np.flatnonzero(groups == arm) for arm in order)
    means = tuple(x[idx].mean(axis=0) for idx in indices)
    centered = np.vstack([x[idx] - mean for idx, mean in zip(indices, means, strict=True)])
    p = centered.shape[1]
    group_weights = np.concatenate(
        [np.full(len(idx), 1 / (len(idx) * (len(idx) - 1))) for idx in indices]
    )
    weights = np.empty(n, dtype=float)
    for idx in indices:
        weights[idx] = 1 / (len(idx) * (len(idx) - 1))
    gram = centered.T @ (centered * group_weights[:, None])
    if p:
        scales = np.sqrt(np.diag(gram))
        deficient = _pivoted_cholesky_deficiency(
            gram / scales[:, None] / scales[None, :], layout.encoded_columns
        )
        if deficient:
            _fail(
                "estimation.cuped_linear.rank_deficiency",
                columns=deficient[0],
                partners=deficient[1],
            )
        gram_inverse = np.linalg.inv(gram)
    else:
        gram_inverse = np.empty((0, 0))
    anchor = (
        sum((len(idx) / n) * mean for idx, mean in zip(indices, means, strict=True))
        if p
        else np.empty(0)
    )
    slopes = tuple(
        gram_inverse
        @ (
            centered.T
            @ (
                group_weights
                * np.concatenate([outcome[idx] - outcome[idx].mean() for idx in indices])
            )
        )
        if p
        else np.empty(0)
        for outcome in outcomes
    )
    adjusted_families = tuple(
        tuple(
            float(outcome[idx].mean() - slopes[f] @ (means[a] - anchor))
            for a, idx in enumerate(indices)
        )
        for f, outcome in enumerate(outcomes)
    )
    design, _ = _wls_design(groups, order, x, means, p)
    inverse = np.linalg.inv(design.T @ (design * weights[:, None]))
    hat = design @ inverse @ (design * weights[:, None]).T
    leverage = np.diag(hat)
    df = n - len(order) - p
    if df <= 0:
        _fail("estimation.cuped_linear.residual_df", df=df)
    if leverage.max() >= 1 - 1e-8:
        _fail(
            "estimation.cuped_linear.leverage",
            row=int(leverage.argmax()),
            leverage=float(leverage.max()),
        )
    components = tuple(tuple(_readonly(outcome[idx]) for idx in indices) for outcome in outcomes)
    residuals = tuple(
        tuple(
            _readonly(outcome[idx] - outcome[idx].mean() - (x[idx] - means[a]) @ slopes[f])
            for a, idx in enumerate(indices)
        )
        for f, outcome in enumerate(outcomes)
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
        _readonly(np.array([order.index(a) for a in groups])),
        components,
        tuple(_readonly(slope) for slope in slopes),
        residuals,
        tuple(_readonly(mean) for mean in means),
        _readonly(gram_inverse),
        _readonly(weights),
        _readonly(design),
        _readonly(inverse),
        _readonly(hat),
    )
    return calculator, leverage


def _wls_design(groups, order, x, means, p):
    design = np.zeros((len(groups), len(order) + p))
    arm_index = np.empty(len(groups), dtype=int)
    for a, arm in enumerate(order):
        idx = np.flatnonzero(groups == arm)
        arm_index[idx] = a
        design[idx, a] = 1
        design[idx, len(order) :] = x[idx] - means[a]
    return design, arm_index


def _cache_projections(calculator, outcomes, order):
    cached = []
    for treatment in order:
        for control in order:
            if treatment == control:
                continue
            result = calculator.absolute(treatment, control=control)
            cached.append(
                ("absolute", treatment, control, "", result.estimate, result.se, result.dof)
            )
            ti, ci = order.index(treatment), order.index(control)
            mean_values = calculator.adjusted_means[: len(order)]
            if mean_values[ti] > 0 and mean_values[ci] > 0:
                result = calculator.relative(treatment, control=control)
                cached.append(
                    ("relative", treatment, control, "", result.estimate, result.se, result.dof)
                )
            if len(outcomes) == 2:
                components = np.asarray(calculator.adjusted_means).reshape(2, -1)
                if np.all(components[:, [ti, ci]] > 0):
                    for scale in ("log", "absolute"):
                        result = calculator.ratio(treatment, control=control, scale=scale)
                        cached.append(
                            (
                                "ratio",
                                treatment,
                                control,
                                scale,
                                result.estimate,
                                result.se,
                                result.dof,
                            )
                        )
    return tuple(cached)


def _mean_covariance(fit: _LinearCupedCalculator) -> np.ndarray:
    narm = len(fit.arm_order)
    nfam = len(fit._components)
    covariance = np.zeros((narm * nfam, narm * nfam))
    scores = []
    for family in range(nfam):
        for arm in range(narm):
            j = np.zeros(narm)
            j[arm] = 1
            scores.append(fit._family_score(family, j)[0])
    for i, left in enumerate(scores):
        for j, right in enumerate(scores):
            covariance[i, j] = left @ right
    return covariance


def _pivoted_cholesky_deficiency(
    matrix: np.ndarray, names: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    n = len(names)

    residual = matrix.copy()
    perm = np.arange(n)
    factor = np.zeros_like(matrix)
    threshold = n * np.finfo(float).eps
    for k in range(n):
        pivot = k + int(np.argmax(np.diag(residual)[k:]))
        if residual[pivot, pivot] <= threshold:
            deficient = tuple(names[i] for i in perm[k:])
            partners = tuple(
                names[i] for i in perm[:k] if np.max(np.abs(matrix[i, perm[k:]])) > 1e-8
            )
            return deficient, partners
        if pivot != k:
            residual[[k, pivot], :] = residual[[pivot, k], :]
            residual[:, [k, pivot]] = residual[:, [pivot, k]]
            factor[[k, pivot], :k] = factor[[pivot, k], :k]
            perm[[k, pivot]] = perm[[pivot, k]]
        factor[k, k] = math.sqrt(residual[k, k])
        if k + 1 < n:
            factor[k + 1 :, k] = residual[k + 1 :, k] / factor[k, k]
            residual[k + 1 :, k + 1 :] -= np.outer(factor[k + 1 :, k], factor[k + 1 :, k])
            residual[k + 1 :, k + 1 :] = (
                residual[k + 1 :, k + 1 :] + residual[k + 1 :, k + 1 :].T
            ) / 2
    return None


def _bm(
    design: np.ndarray,
    weights: np.ndarray,
    leverage: np.ndarray,
    hat: np.ndarray,
    inverse: np.ndarray,
    contrast: np.ndarray,
) -> float:
    ai = design @ (inverse @ contrast) * weights
    gm = (np.eye(len(ai)) - hat).T * (ai / np.sqrt(1 - leverage))[None, :]
    eigenvalues = np.linalg.eigvalsh(gm.T @ gm)
    return float(eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues))


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array, dtype=float).copy()
    out.setflags(write=False)
    return out


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        _fail("estimation.cuped_linear.invalid_alpha", alpha=alpha)


def _interval(value: float, se: float, dof: float, alpha: float) -> LinearCupedProjection:
    critical = two_sided_critical_value(student_t_isf, alpha, dof, what="linear CUPED projection")
    return LinearCupedProjection(value, se, dof, value - critical * se, value + critical * se)
