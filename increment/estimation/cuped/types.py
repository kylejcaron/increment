"""Immutable public CUPED result values."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Literal, NoReturn

import numpy as np

from increment.estimation.cuped._numeric import _check_alpha, _fail
from increment.estimation.cuped.encoding import MulticovariateCupedLayout


def _freeze_payload(value):
    if isinstance(value, np.ndarray):
        return _freeze_payload(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple((_freeze_payload(key), _freeze_payload(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _evidence_equal(left, right) -> bool:
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        if np.isnan(left) and np.isnan(right):
            return True
    if is_dataclass(left) and is_dataclass(right) and type(left) is type(right):
        return all(
            _evidence_equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _evidence_equal(lvalue, rvalue) for lvalue, rvalue in zip(left, right, strict=True)
        )
    result = left == right
    return (
        bool(result) if np.ndim(result) == 0 else bool(np.array_equal(left, right, equal_nan=True))
    )


def _invalid_evidence(reason: str) -> NoReturn:
    _fail("estimation.cuped_linear.invalid_evidence", reason=reason)


def _incompatible_evidence(reason: str) -> NoReturn:
    _fail("estimation.cuped_linear.incompatible_evidence", reason=reason)


@dataclass(frozen=True, slots=True)
class MulticovariateCupedProjection:
    estimate: float
    se: float
    dof: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class MulticovariateCupedFit:
    """Immutable fitted evidence; unit-level input arrays are not retained."""

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
    leverage_trace: float
    is_ratio: bool
    _projections: tuple[tuple[str, object, object, str, float, float, float], ...]

    def absolute(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> MulticovariateCupedProjection:
        return self._lookup("absolute", arm, control, "", alpha)

    def relative(
        self, arm: object, *, control: object | None = None, alpha: float = 0.05
    ) -> MulticovariateCupedProjection:
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
    ) -> MulticovariateCupedProjection:
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
    ) -> MulticovariateCupedProjection:
        _check_alpha(alpha)
        ci = self.control if control is None else control
        for entry in self._projections:
            if entry[:4] == (kind, arm, ci, scale):
                _, _, _, _, estimate, se, dof = entry
                from increment.estimation.cuped.projections import _interval

                return _interval(estimate, se, dof, alpha)
        _fail("estimation.cuped_linear.projection_unavailable", kind=kind, arm=arm, control=ci)


@dataclass(frozen=True, slots=True)
class FitState:
    """Serializable fit-stage evidence. All numeric payloads are immutable tuples."""

    identity: tuple[object, ...]
    eligible_arms: tuple[object, ...]
    withheld_arms: tuple[object, ...]
    refusal_contexts: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    counts: tuple[int, ...]
    layout: MulticovariateCupedLayout
    offsets: tuple[tuple[float, ...], ...]
    scales: tuple[float, ...]
    centered_means: tuple[tuple[float, ...], ...]
    anchor: tuple[float, ...]
    triangular_factor: tuple[tuple[float, ...], ...]
    transformed_rhs: tuple[tuple[float, ...], ...]
    slopes: tuple[tuple[float, ...], ...]
    adjusted_means: tuple[float, ...]
    theta: tuple[float, ...]
    n_min: int
    p: int
    is_ratio: bool
    denominator_support: tuple[tuple[object, ...], ...] = ()
    control: object | None = None

    def __post_init__(self) -> None:
        for name in (
            "identity",
            "eligible_arms",
            "withheld_arms",
            "refusal_contexts",
            "counts",
            "offsets",
            "scales",
            "centered_means",
            "anchor",
            "triangular_factor",
            "transformed_rhs",
            "slopes",
            "adjusted_means",
            "theta",
            "denominator_support",
        ):
            object.__setattr__(self, name, _freeze_payload(getattr(self, name)))
        for name in ("n_min", "p", "is_ratio", "control"):
            object.__setattr__(self, name, _freeze_payload(getattr(self, name)))
        layout_fields = fields(self.layout)
        object.__setattr__(
            self,
            "layout",
            type(self.layout)(
                *(_freeze_payload(getattr(self.layout, item.name)) for item in layout_fields)
            ),
        )
        if not self.identity:
            _invalid_evidence("FitState identity must not be empty")
        if len(self.eligible_arms) != len(self.counts) or any(n <= 0 for n in self.counts):
            _invalid_evidence("FitState arm/count shape mismatch")
        if len(set(self.eligible_arms)) != len(self.eligible_arms):
            _invalid_evidence("FitState eligible arms must be unique")
        if self.n_min != min(self.counts):
            _invalid_evidence("FitState n_min does not match arm counts")
        if len(self.offsets) != self.p or len(self.scales) != self.p:
            _invalid_evidence("FitState offsets/scales do not match encoded dimension")
        if any(len(row) != len(self.eligible_arms) for row in self.offsets):
            _invalid_evidence("FitState offsets do not match eligible arms")
        if len(self.centered_means) != len(self.eligible_arms):
            _invalid_evidence("FitState centered means do not match eligible arms")
        if any(len(row) != self.p for row in self.centered_means):
            _invalid_evidence("FitState centered mean width does not match encoded dimension")
        expected_factor = len(self.eligible_arms) + self.p
        if len(self.triangular_factor) != expected_factor or any(
            len(row) != expected_factor for row in self.triangular_factor
        ):
            _invalid_evidence("FitState triangular factor shape does not match design")
        if len(self.slopes) != len(self.transformed_rhs) or any(
            len(row) != self.p for row in self.slopes
        ):
            _invalid_evidence("FitState slope/RHS shape mismatch")
        if len(self.adjusted_means) != len(self.slopes) * len(self.eligible_arms):
            _invalid_evidence("FitState adjusted means do not match outcome components")


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Normalized Jacobian with original derivative ``jacobian * scale / divisor * 2**exponent``."""

    identity: tuple[object, ...]
    kind: str
    jacobian: tuple[float, ...]
    scale: float
    divisor: float
    exponent: int
    arm: object | None = None
    control: object | None = None
    available: bool = True

    def __post_init__(self) -> None:
        for name in ("kind", "arm", "control"):
            object.__setattr__(self, name, _freeze_payload(getattr(self, name)))
        object.__setattr__(self, "scale", float(_freeze_payload(self.scale)))
        object.__setattr__(self, "divisor", float(_freeze_payload(self.divisor)))
        object.__setattr__(self, "exponent", int(_freeze_payload(self.exponent)))
        object.__setattr__(self, "available", bool(_freeze_payload(self.available)))
        object.__setattr__(self, "identity", _freeze_payload(self.identity))
        object.__setattr__(self, "jacobian", _freeze_payload(self.jacobian))
        if not self.identity or not self.kind or not self.jacobian:
            _invalid_evidence("ProjectionSpec requires identity, kind and a nonempty Jacobian")
        if (
            not np.isfinite(self.scale)
            or not np.isfinite(self.divisor)
            or self.scale == 0
            or self.divisor == 0
        ):
            _invalid_evidence("ProjectionSpec scaling metadata must be finite and nonzero")


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    """Fit-bound sufficient evidence for HC2 and Bell–McCaffrey finalization."""

    fit_identity: tuple[object, ...]
    projection_identity: tuple[object, ...]
    max_leverage: float
    sum_leverage: float
    refusal_witness: tuple[tuple[str, object, float], ...]
    adjusted_component_covariance: tuple[tuple[float, ...], ...]
    scaled_projected_score_squares: tuple[tuple[str, object, object, str, float, float, float], ...]
    s2: tuple[tuple[float, ...], ...] = ()
    s4: tuple[tuple[float, ...], ...] = ()
    g: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...] = ()
    g2: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...] = ()
    u: tuple[tuple[float, ...], ...] = ()
    evidence_identity: tuple[object, ...] = ()
    projection_specs: tuple[ProjectionSpec, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "fit_identity",
            "projection_identity",
            "refusal_witness",
            "adjusted_component_covariance",
            "scaled_projected_score_squares",
            "s2",
            "s4",
            "g",
            "g2",
            "u",
            "evidence_identity",
            "projection_specs",
        ):
            object.__setattr__(self, name, _freeze_payload(getattr(self, name)))
        object.__setattr__(self, "max_leverage", float(_freeze_payload(self.max_leverage)))
        object.__setattr__(self, "sum_leverage", float(_freeze_payload(self.sum_leverage)))
        if not self.fit_identity or not self.projection_identity:
            _invalid_evidence("InferenceSummary identities must not be empty")
        if not 0 <= self.max_leverage < 1 or self.sum_leverage < 0:
            _invalid_evidence("InferenceSummary leverage is outside its valid range")
        size = len(self.adjusted_component_covariance)
        if any(len(row) != size for row in self.adjusted_component_covariance):
            _invalid_evidence("InferenceSummary covariance must be square")
        if self.evidence_identity and self.evidence_identity != self.fit_identity:
            _invalid_evidence("InferenceSummary evidence identity differs from fit identity")
        if len(self.s2) != len(self.s4):
            _invalid_evidence("InferenceSummary S2/S4 projection shapes differ")
        if len(self.g) != len(self.s2) or len(self.g2) != len(self.s2):
            _invalid_evidence("InferenceSummary G/G2 projection shapes differ")
        if any(
            len(s2_row) != len(s4_row) or len(s2_row) != len(g_row) or len(s2_row) != len(g2_row)
            for s2_row, s4_row, g_row, g2_row in zip(self.s2, self.s4, self.g, self.g2, strict=True)
        ):
            _invalid_evidence("InferenceSummary component reduction shapes differ")
        u_size = len(self.u)
        if any(len(row) != u_size for row in self.u):
            _invalid_evidence("InferenceSummary U must be square")
        for projection_g, projection_g2 in zip(self.g, self.g2, strict=True):
            for matrix, weighted in zip(projection_g, projection_g2, strict=True):
                if len(matrix) != 2 * u_size or len(weighted) != 2 * u_size:
                    _invalid_evidence("InferenceSummary G/G2 design dimensions differ")
                if any(len(row) != 2 * u_size for row in matrix + weighted):
                    _invalid_evidence("InferenceSummary G/G2 matrices must be square")

    def merge(self, other: InferenceSummary) -> InferenceSummary:
        if (self.fit_identity, self.projection_identity, self.evidence_identity) != (
            other.fit_identity,
            other.projection_identity,
            other.evidence_identity,
        ):
            _incompatible_evidence("cannot merge inference summaries with incompatible identities")
        if not _evidence_equal(self, other):
            _invalid_evidence("complete inference summaries are replay-only, not additive")
        return self


@dataclass(frozen=True, slots=True)
class CategoricalCells:
    """Joint cells: arm-referenced first sums and within-cell centered cross sums.

    Physical references and binary exponents are component × arm. First sums
    are arm × component × pattern, scaled by 2**outcome_exponents[component][arm].
    Cross sums are arm × component × component × pattern, scaled by both of
    the arm's component powers, so each arm keeps its own representable scale.
    Null levels use None.
    """

    identity: tuple[object, ...]
    arms: tuple[object, ...]
    patterns: tuple[tuple[object, ...], ...]
    counts: tuple[tuple[int, ...], ...]
    levels: tuple[tuple[object, ...], ...]
    null_metadata: tuple[tuple[object, ...], ...]
    outcome_references: tuple[tuple[float, ...], ...]
    centered_first_moments: tuple[tuple[tuple[float, ...], ...], ...]
    symmetric_cross_moments: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    outcome_exponents: tuple[tuple[int, ...], ...]
    covariate_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "identity",
            "arms",
            "patterns",
            "counts",
            "levels",
            "null_metadata",
            "outcome_references",
            "centered_first_moments",
            "symmetric_cross_moments",
            "outcome_exponents",
            "covariate_names",
        ):
            object.__setattr__(self, name, _freeze_payload(getattr(self, name)))
        cells = len(self.patterns)
        if not self.identity or len(self.counts) != len(self.arms):
            _invalid_evidence("CategoricalCells identity or arm shape is invalid")
        if any(len(row) != cells or any(count < 0 for count in row) for row in self.counts):
            _invalid_evidence("CategoricalCells counts do not match joint patterns")
        if len(set(self.patterns)) != cells:
            _invalid_evidence("CategoricalCells joint patterns must be unique and ordered")
        if len(self.centered_first_moments) != len(self.arms):
            _invalid_evidence("CategoricalCells outcome moments do not match arms")
        if len(self.levels) != len(self.null_metadata):
            _invalid_evidence("CategoricalCells level/null metadata shape mismatch")
        if any(
            not isinstance(metadata, tuple)
            or len(metadata) != 2
            or metadata[0] not in ("refuse", "impute-indicator")
            or not isinstance(metadata[1], bool)
            for metadata in self.null_metadata
        ):
            _invalid_evidence("CategoricalCells null metadata entries must be policy/null pairs")
        if any(
            isinstance(count, bool) or not isinstance(count, int)
            for row in self.counts
            for count in row
        ):
            _invalid_evidence("CategoricalCells counts must be integers")
        if any(
            len(arm_moments) != len(self.outcome_references)
            or any(len(component) != cells for component in arm_moments)
            for arm_moments in self.centered_first_moments
        ):
            _invalid_evidence("CategoricalCells first moments do not match joint cells")
        if len(self.symmetric_cross_moments) != len(self.arms):
            _invalid_evidence("CategoricalCells cross-moments do not match arms")
        components = len(self.outcome_references)
        if any(len(row) != len(self.arms) for row in self.outcome_references):
            _invalid_evidence("CategoricalCells outcome references do not match arms")
        if any(
            len(arm) != components
            or any(
                len(row) != components or any(len(values) != cells for values in row) for row in arm
            )
            for arm in self.symmetric_cross_moments
        ):
            _invalid_evidence("CategoricalCells cross-moments do not match outcome components")
        if len(set(self.arms)) != len(self.arms):
            _invalid_evidence("CategoricalCells arms must be unique")
        if any(len(pattern) != len(self.levels) for pattern in self.patterns):
            _invalid_evidence("CategoricalCells requires complete joint categorical patterns")
        if self.covariate_names and (
            len(self.covariate_names) != len(self.levels)
            or len(set(self.covariate_names)) != len(self.covariate_names)
        ):
            _invalid_evidence("CategoricalCells covariate names do not match categorical columns")
        for column, levels in enumerate(self.levels):
            if len(set(levels)) != len(levels) or any(
                value is not None and not isinstance(value, str) for value in levels
            ):
                _invalid_evidence("CategoricalCells levels must be unique strings or null")
            if any(pattern[column] not in levels for pattern in self.patterns):
                _invalid_evidence("CategoricalCells joint pattern contains an unknown level")
        self._validate_outcome_values()

    def _validate_outcome_values(self) -> None:
        components = len(self.outcome_references)
        if len(self.outcome_exponents) != components or any(
            len(row) != len(self.arms)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not -1073 <= value <= 1024
                for value in row
            )
            for row in self.outcome_exponents
        ):
            _invalid_evidence(
                "CategoricalCells outcome scales must be component-by-arm binary exponents"
            )
        for values in (
            self.outcome_references,
            self.centered_first_moments,
            self.symmetric_cross_moments,
        ):
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                _invalid_evidence("CategoricalCells outcome moments must be finite")
        cross = np.asarray(self.symmetric_cross_moments, dtype=float)
        if components and not np.array_equal(cross, cross.swapaxes(1, 2)):
            _invalid_evidence("CategoricalCells outcome cross-moments must be symmetric")
        for arm, counts in enumerate(self.counts):
            for pattern, count in enumerate(counts):
                if not components:
                    continue
                first = np.asarray(self.centered_first_moments[arm])[:, pattern]
                scatter = cross[arm, :, :, pattern]
                if count == 0 and (np.any(first) or np.any(scatter)):
                    _invalid_evidence("CategoricalCells empty cells must have zero moments")
                if components and np.any(np.diag(scatter) < 0):
                    _invalid_evidence("CategoricalCells outcome scatter must be nonnegative")

    def merge(self, other: CategoricalCells) -> CategoricalCells:
        from increment.estimation.cuped.cells import _merge_cells

        return _merge_cells(self, other)
