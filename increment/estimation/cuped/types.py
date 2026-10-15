"""Immutable public CUPED result values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from increment.estimation.cuped._numeric import _check_alpha, _fail
from increment.estimation.cuped.encoding import MulticovariateCupedLayout


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
        if not self.identity:
            raise ValueError("FitState identity must not be empty")
        if len(self.eligible_arms) != len(self.counts) or any(n <= 0 for n in self.counts):
            raise ValueError("FitState arm/count shape mismatch")
        if len(set(self.eligible_arms)) != len(self.eligible_arms):
            raise ValueError("FitState eligible arms must be unique")
        if self.n_min != min(self.counts):
            raise ValueError("FitState n_min does not match arm counts")
        if len(self.offsets) != self.p or len(self.scales) != self.p:
            raise ValueError("FitState offsets/scales do not match encoded dimension")
        if any(len(row) != len(self.eligible_arms) for row in self.offsets):
            raise ValueError("FitState offsets do not match eligible arms")
        if len(self.centered_means) != len(self.eligible_arms):
            raise ValueError("FitState centered means do not match eligible arms")
        if any(len(row) != self.p for row in self.centered_means):
            raise ValueError("FitState centered mean width does not match encoded dimension")
        expected_factor = len(self.eligible_arms) + self.p
        if len(self.triangular_factor) != expected_factor or any(
            len(row) != expected_factor for row in self.triangular_factor
        ):
            raise ValueError("FitState triangular factor shape does not match design")
        if len(self.slopes) != len(self.transformed_rhs) or any(
            len(row) != self.p for row in self.slopes
        ):
            raise ValueError("FitState slope/RHS shape mismatch")
        if len(self.adjusted_means) != len(self.slopes) * len(self.eligible_arms):
            raise ValueError("FitState adjusted means do not match outcome components")


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Immutable derivative and scaling metadata for one retained projection."""

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
        if not self.identity or not self.kind or not self.jacobian:
            raise ValueError("ProjectionSpec requires identity, kind and a nonempty Jacobian")
        if (
            not np.isfinite(self.scale)
            or not np.isfinite(self.divisor)
            or self.scale == 0
            or self.divisor == 0
        ):
            raise ValueError("ProjectionSpec scaling metadata must be finite and nonzero")


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    """Fit-bound sufficient evidence for HC2 and Bell–McCaffrey finalization."""

    fit_identity: tuple[object, ...]
    projection_identity: tuple[object, ...]
    max_leverage: float
    sum_leverage: float
    refusal_witness: tuple[tuple[str, object], ...]
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
        if not self.fit_identity or not self.projection_identity:
            raise ValueError("InferenceSummary identities must not be empty")
        if not 0 <= self.max_leverage < 1 or self.sum_leverage < 0:
            raise ValueError("InferenceSummary leverage is outside its valid range")
        size = len(self.adjusted_component_covariance)
        if any(len(row) != size for row in self.adjusted_component_covariance):
            raise ValueError("InferenceSummary covariance must be square")
        if self.evidence_identity and self.evidence_identity != self.fit_identity:
            raise ValueError("InferenceSummary evidence identity differs from fit identity")
        if len(self.s2) != len(self.s4):
            raise ValueError("InferenceSummary S2/S4 projection shapes differ")
        if len(self.g) != len(self.s2) or len(self.g2) != len(self.s2):
            raise ValueError("InferenceSummary G/G2 projection shapes differ")
        if any(
            len(s2_row) != len(s4_row) or len(s2_row) != len(g_row) or len(s2_row) != len(g2_row)
            for s2_row, s4_row, g_row, g2_row in zip(self.s2, self.s4, self.g, self.g2, strict=True)
        ):
            raise ValueError("InferenceSummary component reduction shapes differ")
        u_size = len(self.u)
        if any(len(row) != u_size for row in self.u):
            raise ValueError("InferenceSummary U must be square")
        for projection_g, projection_g2 in zip(self.g, self.g2, strict=True):
            for matrix, weighted in zip(projection_g, projection_g2, strict=True):
                if len(matrix) != 2 * u_size or len(weighted) != 2 * u_size:
                    raise ValueError("InferenceSummary G/G2 design dimensions differ")
                if any(len(row) != 2 * u_size for row in matrix + weighted):
                    raise ValueError("InferenceSummary G/G2 matrices must be square")

    def merge(self, other: InferenceSummary) -> InferenceSummary:
        if (self.fit_identity, self.projection_identity, self.evidence_identity) != (
            other.fit_identity,
            other.projection_identity,
            other.evidence_identity,
        ):
            raise ValueError("cannot merge inference summaries with incompatible identities")
        if self != other:
            raise ValueError("complete inference summaries are replay-only, not additive")
        return self


@dataclass(frozen=True, slots=True)
class CategoricalCells:
    """Ordered joint categorical cells with exact count and outcome moments."""

    identity: tuple[object, ...]
    arms: tuple[object, ...]
    patterns: tuple[tuple[object, ...], ...]
    counts: tuple[tuple[int, ...], ...]
    levels: tuple[tuple[object, ...], ...]
    null_metadata: tuple[tuple[object, ...], ...]
    outcome_references: tuple[tuple[float, ...], ...]
    centered_first_moments: tuple[tuple[tuple[float, ...], ...], ...]
    symmetric_cross_moments: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]

    def __post_init__(self) -> None:
        cells = len(self.patterns)
        if not self.identity or len(self.counts) != len(self.arms):
            raise ValueError("CategoricalCells identity or arm shape is invalid")
        if any(len(row) != cells or any(count < 0 for count in row) for row in self.counts):
            raise ValueError("CategoricalCells counts do not match joint patterns")
        if len(set(self.patterns)) != cells:
            raise ValueError("CategoricalCells joint patterns must be unique and ordered")
        if len(self.centered_first_moments) != len(self.arms):
            raise ValueError("CategoricalCells outcome moments do not match arms")
        if len(self.levels) != len(self.null_metadata):
            raise ValueError("CategoricalCells level/null metadata shape mismatch")
        if any(
            isinstance(count, bool) or not isinstance(count, int)
            for row in self.counts
            for count in row
        ):
            raise ValueError("CategoricalCells counts must be integers")
        if any(
            len(arm_moments) != len(self.outcome_references)
            or any(len(component) != cells for component in arm_moments)
            for arm_moments in self.centered_first_moments
        ):
            raise ValueError("CategoricalCells first moments do not match joint cells")
        if len(self.symmetric_cross_moments) != len(self.arms):
            raise ValueError("CategoricalCells cross-moments do not match arms")
