"""Fixed-horizon contrasts of independent binary assignments."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from fractions import Fraction
from typing import Annotated, Literal, NoReturn, Self

from pydantic import Field, StrictFloat, StrictInt
from scipy.special import stdtr

from increment._canonical import canonical_json_bytes
from increment._overlap_evidence import (
    CELL_IDS,
    OVERLAP_FORMAT,
    CellId,
    Count,
    Identity,
    Ordered,
    OverlapCell,
    OverlapDeliveryCount,
    OverlapIngress,
    OverlapModel,
    Sha256,
    overlap_digest,
    overlap_wire_payload,
)
from increment._overlap_refusals import EVIDENCE_INVALID, FORMAT_INVALID, NUMERIC_RANGE
from increment.errors import CodedError, refuse
from increment.estimation._tails import student_t_isf, two_sided_critical_value, wald_bounds
from increment.semantics.overlap import (
    CONTRAST_IDS,
    ContrastId,
    OverlapCapture,
    OverlapClaim,
    OverlapEvidence,
    OverlapMetric,
    OverlapPlan,
)

__all__ = ["OverlapResult", "analyze_overlap"]

type _Role = Literal["primary", "guardrail", "secondary"]
type _Direction = Literal["higher", "lower"]
type _Alternative = Literal["benefit", "difference", "noninferiority"]
type _FittedDecision = Literal["clear", "reject", "loss_beyond_tolerance", "inconclusive"]
type _Decision = Literal[_FittedDecision, "insufficient_for_decision"]
type _JointDecision = Literal["not_requested", "insufficient_evidence", "not_met", "criteria_met"]
type _Rollout = Literal["as_assigned", "as_assigned_with_delivery_conflicts"]
type _Availability = Literal["unavailable", "point_only", "descriptive", "fitted_small", "decided"]
type _Reason = Literal[
    "no_target",
    "pending_outcomes",
    "missing_outcomes",
    "empty_cell",
    "singleton_cell",
    "zero_cell_variance",
    "small_cell",
    "sparse_binary_events",
    "p_boundary_numeric_range",
]
type _ReasonPair = tuple[CellId | None, _Reason]
type _Finite = Annotated[StrictFloat, Field(allow_inf_nan=False)]
type _Positive = Annotated[_Finite, Field(gt=0)]
type _Probability = Annotated[_Finite, Field(gt=0, le=1)]
type _Weights = tuple[_Finite, _Finite, _Finite, _Finite]
type _Exact = int | Fraction
type _ExactWeights = tuple[_Exact, _Exact, _Exact, _Exact]

# In each pair the first cell is subtracted from the second. The interaction
# takes the difference of the two A simple effects, in the full difference scale.
_PAIRS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 1),),
    ((2, 3),),
    ((0, 2),),
    ((1, 3),),
    ((0, 3),),
    ((1, 0), (2, 3)),
)
_NO_POINT = frozenset(("no_target", "pending_outcomes", "missing_outcomes", "empty_cell"))
_NO_VARIANCE = frozenset(("singleton_cell", "zero_cell_variance"))
_SPARSE = frozenset(("small_cell", "sparse_binary_events"))
_INTERVAL = "simultaneous overlap interval"
_P_RANGE: _ReasonPair = (None, "p_boundary_numeric_range")


class OverlapRow(OverlapModel):
    """One raw-scale estimate; reasons preserve the identity of limiting cells."""

    metric: Identity
    role: _Role
    contrast: ContrastId
    weights: Ordered[_Weights]
    estimate: _Finite | None
    se: _Positive | None
    df: _Positive | None
    lower: _Finite | None
    upper: _Finite | None
    p_boundary: _Probability | None
    p_boundary_adj: _Probability | None
    alternative: _Alternative | None
    boundary: _Finite | None
    # Registered allocation rounded for display; null if it rounds to zero.
    alpha_member: _Probability | None
    availability: _Availability
    reasons: Ordered[tuple[Ordered[_ReasonPair], ...]]
    decision: _Decision | None

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(FORMAT_INVALID, field=field, reason=reason)


class OverlapCovariance(OverlapModel):
    """Within-metric covariance and the reasons attached to each row of the block."""

    metric: Identity
    matrix: Ordered[tuple[Ordered[tuple[_Finite | None, ...]], ...]]
    reasons: Ordered[tuple[Ordered[tuple[Ordered[_ReasonPair], ...]], ...]]

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(FORMAT_INVALID, field=field, reason=reason)


def _exact_weights(plan: OverlapPlan) -> tuple[_ExactWeights, ...]:
    a0, a = plan.a.arm_probabilities
    b0, b = plan.b.arm_probabilities
    return (
        (-b0, b0, -b, b),
        (-a0, -a, a0, a),
        (-1, 1, 0, 0),
        (0, 0, -1, 1),
        (-1, 0, 1, 0),
        (0, -1, 0, 1),
        (-1, 0, 0, 1),
        (1, -1, -1, 1),
    )


def _float_weights(c: _ExactWeights) -> _Weights:
    return float(c[0]), float(c[1]), float(c[2]), float(c[3])


def _required_cells(c: _ExactWeights, cells: tuple[OverlapCell, ...]) -> tuple[OverlapCell, ...]:
    return tuple(cell for weight, cell in zip(c, cells, strict=True) if weight)


def _reasons(
    cells: tuple[OverlapCell, ...], *, target_count: int, claimed: bool
) -> tuple[_ReasonPair, ...]:
    if not target_count:
        return ((None, "no_target"),)
    reasons: list[_ReasonPair] = []
    for cell in cells:
        if cell.n_pending:
            reasons.append((cell.cell, "pending_outcomes"))
        if cell.n_missing:
            reasons.append((cell.cell, "missing_outcomes"))
        if cell.n_pending or cell.n_missing:
            continue
        if not cell.n_assigned:
            reasons.append((cell.cell, "empty_cell"))
        elif cell.n_observed == 1:
            reasons.append((cell.cell, "singleton_cell"))
        elif cell.m2 == 0:
            reasons.append((cell.cell, "zero_cell_variance"))
    if reasons or not claimed:
        return tuple(reasons)
    for cell in cells:
        n, k = cell.n_observed, cell.n_success
        if n < 30:
            reasons.append((cell.cell, "small_cell"))
        if k is not None and min(k, n - k) < 10:
            reasons.append((cell.cell, "sparse_binary_events"))
    return tuple(reasons)


def _availability(reasons: tuple[_ReasonPair, ...], *, claimed: bool) -> _Availability:
    if any(reason in _NO_POINT for _, reason in reasons):
        return "unavailable"
    if any(reason in _NO_VARIANCE for _, reason in reasons):
        return "point_only"
    if not claimed:
        return "descriptive"
    if any(reason in _SPARSE for _, reason in reasons):
        return "fitted_small"
    return "decided"


def _rounded(value: Fraction, *, metric: str, quantity: str) -> float:
    if abs(value) > sys.float_info.max:
        refuse(NUMERIC_RANGE, metric=metric, quantity=quantity)
    rounded = float(value)
    if rounded == 0.0 and value:
        refuse(NUMERIC_RANGE, metric=metric, quantity=quantity)
    return rounded


def _point(
    cells: tuple[OverlapCell, ...],
    weights: _Weights,
    exact_weights: _ExactWeights,
    pairs: tuple[tuple[int, int], ...],
    *,
    metric: str,
) -> float:
    # Keep anchors and residuals separate so a rounded cell mean cannot erase
    # an effect. Exact arithmetic resolves ill-conditioned cancellations and
    # range edges where a pair difference need not itself be representable.
    terms: list[float] = []
    exact_terms: list[tuple[int, float, float]] = []
    for lo, hi in pairs:
        x, y = cells[lo], cells[hi]
        assert x.anchor is not None and y.anchor is not None
        assert x.mean_delta is not None and y.mean_delta is not None
        exact_terms.extend(((hi, x.anchor, y.anchor), (hi, x.mean_delta, y.mean_delta)))
    safe = True
    for hi, x, y in exact_terms:
        if x != y:
            term = weights[hi] * (y - x)
            if not math.isfinite(term) or abs(term) < sys.float_info.min:
                safe = False
                break
            terms.append(term)
    if safe:
        try:
            total = math.fsum(terms)
        except OverflowError:
            pass
        else:
            if not terms or (
                abs(total) >= sys.float_info.min
                and max(map(abs, terms)) <= 32 * abs(total) < sys.float_info.max
            ):
                return total
    exact = sum(
        (exact_weights[hi] * (Fraction(y) - Fraction(x)) for hi, x, y in exact_terms),
        Fraction(0),
    )
    return _rounded(exact, metric=metric, quantity="estimate")


def _welch(weights: _Weights, cells: tuple[OverlapCell, ...]) -> tuple[float, float]:
    # h_j = c_j**2 * M2_j / ((n_j - 1) * n_j). Normalize sqrt(h_j)
    # before squaring, so df never divides two underflowed squared variances.
    roots: list[tuple[float, int]] = []
    for weight, cell in zip(weights, cells, strict=True):
        if weight:
            n, m2 = cell.n_observed, cell.m2
            assert m2 is not None and m2 > 0 and n >= 2
            roots.append((abs(weight) * math.sqrt(m2) / math.sqrt(n * (n - 1)), n - 1))
    scale = max(root for root, _ in roots)
    relative = [((root / scale) ** 2, dof) for root, dof in roots]
    total = math.fsum(h for h, _ in relative)
    denominator = math.fsum((h / total) ** 2 / dof for h, dof in relative)
    return scale * math.sqrt(total), 1 / denominator


def _overlap_claim_decision(
    alternative: _Alternative,
    better: _Direction,
    lower: float,
    upper: float,
    margin_abs: float,
) -> _FittedDecision:
    if alternative == "difference":
        return "reject" if lower > 0 or upper < 0 else "inconclusive"
    benefit_lo, benefit_hi = (lower, upper) if better == "higher" else (-upper, -lower)
    boundary = -margin_abs if alternative == "noninferiority" else 0.0
    if benefit_lo > boundary:
        return "clear"
    if alternative == "noninferiority" and benefit_hi < boundary:
        return "loss_beyond_tolerance"
    return "inconclusive"


def _raw_boundary(metric: OverlapMetric, claim: OverlapClaim) -> float:
    if claim.alternative != "noninferiority":
        return 0.0
    return -metric.margin_abs if metric.better == "higher" else metric.margin_abs


def _member_alpha(plan: OverlapPlan) -> float | None:
    return plan.alpha / len(plan.claims) or None


def _member_tail(plan: OverlapPlan) -> float:
    # Allocate the one-sided tail directly from the registered alpha. Never
    # round it above the family budget, including at the subnormal boundary.
    exact = Fraction(plan.alpha) / (2 * len(plan.claims))
    tail = float(exact)
    return math.nextafter(tail, 0.0) if tail > exact else tail


def _boundary_p(estimate: float, boundary: float, se: float, df: float) -> float | None:
    gap = estimate - boundary
    z = abs(gap) / se if math.isfinite(gap) else abs(estimate / se - boundary / se)
    probability = 2.0 * float(stdtr(df, -z))
    return probability if 0.0 < probability <= 1.0 else None


def _row(
    metric: OverlapMetric,
    cells: tuple[OverlapCell, ...],
    index: int,
    exact_weights: _ExactWeights,
    claim: OverlapClaim | None,
    *,
    target_count: int,
    alpha: float | None,
    tail: float,
    members: int,
) -> OverlapRow:
    weights = _float_weights(exact_weights)
    support = _required_cells(exact_weights, cells)
    reasons = _reasons(support, target_count=target_count, claimed=claim is not None)
    availability = _availability(reasons, claimed=claim is not None)
    estimate = se = df = lower = upper = p = adjusted = None
    decision: _Decision | None = None
    boundary = _raw_boundary(metric, claim) if claim else None
    if availability != "unavailable":
        estimate = _point(cells, weights, exact_weights, _PAIRS[index], metric=metric.name)
    if availability in ("descriptive", "fitted_small", "decided"):
        se, df = _welch(weights, cells)
    if claim is not None and se is not None:
        assert df is not None and estimate is not None and boundary is not None
        try:
            # Doubling and the helper's halving recover the certified tail exactly.
            critical = two_sided_critical_value(student_t_isf, 2.0 * tail, df, what=_INTERVAL)
        except ArithmeticError:
            refuse(NUMERIC_RANGE, metric=metric.name, quantity="critical_value")
        lower, upper = wald_bounds(estimate, critical, se, what=_INTERVAL)
        p = _boundary_p(estimate, boundary, se, df)
        if p is None:
            reasons = (*reasons, _P_RANGE)
        else:
            adjusted = min(1.0, members * p)
        decision = (
            "insufficient_for_decision"
            if availability == "fitted_small"
            else _overlap_claim_decision(
                claim.alternative, metric.better, lower, upper, metric.margin_abs
            )
        )
    return OverlapRow(
        metric=metric.name,
        role=metric.role,
        contrast=CONTRAST_IDS[index],
        weights=weights,
        estimate=estimate,
        se=se,
        df=df,
        lower=lower,
        upper=upper,
        p_boundary=p,
        p_boundary_adj=adjusted,
        alternative=claim.alternative if claim else None,
        boundary=boundary,
        alpha_member=alpha if claim else None,
        availability=availability,
        reasons=reasons,
        decision=decision,
    )


def _overlap_rows(
    plan: OverlapPlan, cells: tuple[OverlapCell, ...], *, target_count: int
) -> tuple[OverlapRow, ...]:
    """Estimate and decide from trusted metric-major cells, without a source receipt."""
    alpha = _member_alpha(plan)
    tail = _member_tail(plan)
    weights = _exact_weights(plan)
    claims = {(claim.metric, claim.contrast): claim for claim in plan.claims}
    return tuple(
        _row(
            metric,
            group,
            index,
            weight,
            claims.get((metric.name, contrast)),
            target_count=target_count,
            alpha=alpha,
            tail=tail,
            members=len(claims),
        )
        for metric, group in zip(plan.metrics, _cell_groups(cells), strict=True)
        for index, (contrast, weight) in enumerate(zip(CONTRAST_IDS, weights, strict=True))
    )


def _cell_groups(cells: tuple[OverlapCell, ...]) -> Iterator[tuple[OverlapCell, ...]]:
    for i in range(0, len(cells), len(CELL_IDS)):
        yield cells[i : i + len(CELL_IDS)]


def _overlap_joint_decision(
    rows: tuple[OverlapRow, ...], *, delivery_conflicts: bool
) -> _JointDecision:
    launch = tuple(
        row
        for row in rows
        if row.contrast == "joint_launch"
        and row.role != "secondary"
        and row.alternative is not None
    )
    if not any(row.role == "primary" for row in launch):
        return "not_requested"
    if delivery_conflicts or any(row.availability != "decided" for row in launch):
        return "insufficient_evidence"
    return "criteria_met" if all(row.decision == "clear" for row in launch) else "not_met"


def _covariance(
    plan: OverlapPlan, cells: tuple[OverlapCell, ...], rows: tuple[OverlapRow, ...]
) -> tuple[OverlapCovariance, ...]:
    # Under independent units the cell mean covariance is diagonal. Use exact
    # ratios for the signed products: unlike df, no small term can be dropped
    # from an off-diagonal before cancellation, or rounded before rescaling.
    weights = _exact_weights(plan)
    size = len(CONTRAST_IDS)
    blocks = []
    for i, (metric, group) in enumerate(zip(plan.metrics, _cell_groups(cells), strict=True)):
        fitted = rows[i * size : (i + 1) * size]
        variances = tuple(
            Fraction(cell.m2) / (cell.n_observed * (cell.n_observed - 1))
            if cell.m2 is not None and cell.n_observed >= 2
            else Fraction(0)
            for cell in group
        )
        matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
        for j, left in enumerate(fitted):
            if left.se is None:
                continue
            for k in range(j, size):
                if fitted[k].se is not None:
                    exact = sum(
                        (
                            c * d * v
                            for c, d, v in zip(weights[j], weights[k], variances, strict=True)
                            if c and d
                        ),
                        Fraction(0),
                    )
                    matrix[j][k] = matrix[k][j] = _rounded(
                        exact, metric=metric.name, quantity="covariance"
                    )
        blocks.append(
            OverlapCovariance(
                metric=metric.name,
                matrix=tuple(tuple(row) for row in matrix),
                reasons=tuple(row.reasons for row in fitted),
            )
        )
    return tuple(blocks)


class OverlapResult(OverlapModel):
    """An as-assigned readout and the bounded evidence and plan it refers to."""

    format: StrictInt = OVERLAP_FORMAT
    plan: OverlapPlan
    plan_sha256: Sha256
    evidence_sha256: Sha256
    capture: OverlapCapture
    ingress: OverlapIngress
    replay_status: Literal["not_certified"] = "not_certified"
    eligibility_counts: Ordered[tuple[Count, Count, Count, Count]]
    gate_counts: Ordered[tuple[Count, ...]]
    delivery_counts: Ordered[tuple[OverlapDeliveryCount, ...]]
    cells: Ordered[tuple[OverlapCell, ...]]
    cell_order: Ordered[tuple[CellId, ...]] = CELL_IDS
    contrast_order: Ordered[tuple[ContrastId, ...]] = CONTRAST_IDS
    family_members: Ordered[tuple[OverlapClaim, ...]]
    family_sha256: Sha256
    alpha: _Probability
    rows: Ordered[tuple[OverlapRow, ...]]
    covariance: Ordered[tuple[OverlapCovariance, ...]]
    rollout: _Rollout
    joint_decision: _JointDecision

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(FORMAT_INVALID, field=field, reason=reason)

    @property
    def target_count(self) -> int:
        """Number of jointly assigned target units used for the readout."""
        return self.gate_counts[0]

    def _validate_overlap(self) -> Self:
        if self.format != OVERLAP_FORMAT:
            self._invalid_field("format", "unsupported_version")
        evidence = OverlapEvidence(
            plan=self.plan,
            capture=self.capture,
            ingress=self.ingress,
            replay_status=self.replay_status,
            eligibility_counts=self.eligibility_counts,
            gate_counts=self.gate_counts,
            delivery_counts=self.delivery_counts,
            cells=self.cells,
        )
        if self.evidence_sha256 != evidence.evidence_sha256:
            self._invalid_field("evidence_sha256", "digest_mismatch")
        if self.plan_sha256 != self.plan.plan_sha256:
            self._invalid_field("plan_sha256", "digest_mismatch")
        if (self.family_members, self.family_sha256, self.alpha) != (
            self.plan.claims,
            self.plan.family_sha256,
            self.plan.alpha,
        ):
            self._invalid_field("family_members", "family_mismatch")
        if self.cell_order != CELL_IDS or self.contrast_order != CONTRAST_IDS:
            self._invalid_field("cell_order", "axis_order")
        self._check_rows()
        self._check_covariance()
        conflicts = _delivery_conflicts(self.delivery_counts)
        if self.rollout != _rollout(conflicts=conflicts):
            self._invalid_field("rollout", "delivery_status_mismatch")
        if self.joint_decision != _overlap_joint_decision(self.rows, delivery_conflicts=conflicts):
            self._invalid_field("joint_decision", "decision_mismatch")
        return self

    def _check_rows(self) -> None:
        keys = tuple((m.name, c) for m in self.plan.metrics for c in CONTRAST_IDS)
        if tuple((row.metric, row.contrast) for row in self.rows) != keys:
            self._invalid_field("rows", "row_order")
        claims = {(c.metric, c.contrast): c for c in self.plan.claims}
        alpha = _member_alpha(self.plan)
        if any(row.lower is not None for row in self.rows) and not _member_tail(self.plan):
            self._invalid_field("rows", "unresolvable_tail")
        weights = _exact_weights(self.plan)
        size = len(CONTRAST_IDS)
        for i, (metric, group) in enumerate(
            zip(self.plan.metrics, _cell_groups(self.cells), strict=True)
        ):
            for row, c in zip(self.rows[i * size : (i + 1) * size], weights, strict=True):
                claim = claims.get((metric.name, row.contrast))
                registration = (
                    metric.role,
                    _float_weights(c),
                    claim.alternative if claim else None,
                    _raw_boundary(metric, claim) if claim else None,
                    alpha if claim else None,
                )
                if (
                    row.role,
                    row.weights,
                    row.alternative,
                    row.boundary,
                    row.alpha_member,
                ) != registration:
                    self._invalid_field("rows", "registration_mismatch")
                reasons = _reasons(
                    _required_cells(c, group),
                    target_count=self.target_count,
                    claimed=claim is not None,
                )
                self._check_row(row, metric, claim, reasons)

    def _check_row(
        self,
        row: OverlapRow,
        metric: OverlapMetric,
        claim: OverlapClaim | None,
        reasons: tuple[_ReasonPair, ...],
    ) -> None:
        availability = _availability(reasons, claimed=claim is not None)
        fitted = availability in ("descriptive", "fitted_small", "decided")
        band = fitted and claim is not None
        for name, expected in (
            ("estimate", availability != "unavailable"),
            ("se", fitted),
            ("df", fitted),
            ("lower", band),
            ("upper", band),
            ("decision", band),
        ):
            if (getattr(row, name) is not None) != expected:
                self._invalid_field("rows", "null_mismatch")
        if band and row.p_boundary is None:
            reasons = (*reasons, _P_RANGE)
        adjusted = None
        if row.p_boundary is not None:
            if not band:
                self._invalid_field("rows", "null_mismatch")
            adjusted = min(1.0, len(self.family_members) * row.p_boundary)
        if (
            row.p_boundary_adj != adjusted
            or row.availability != availability
            or row.reasons != reasons
        ):
            self._invalid_field("rows", "availability_mismatch")
        if band:
            assert claim is not None and row.lower is not None and row.upper is not None
            assert row.estimate is not None
            if not row.lower <= row.estimate <= row.upper:
                self._invalid_field("rows", "bounds_order")
            decision = (
                "insufficient_for_decision"
                if availability == "fitted_small"
                else _overlap_claim_decision(
                    claim.alternative, metric.better, row.lower, row.upper, metric.margin_abs
                )
            )
            if row.decision != decision:
                self._invalid_field("rows", "decision_mismatch")

    def _check_covariance(self) -> None:
        if tuple(block.metric for block in self.covariance) != tuple(
            m.name for m in self.plan.metrics
        ):
            self._invalid_field("covariance", "metric_order")
        size = len(CONTRAST_IDS)
        for i, block in enumerate(self.covariance):
            rows = self.rows[i * size : (i + 1) * size]
            if block.reasons != tuple(row.reasons for row in rows):
                self._invalid_field("covariance", "reason_mismatch")
            if len(block.matrix) != size or any(len(row) != size for row in block.matrix):
                self._invalid_field("covariance", "matrix_shape")
            for j, left in enumerate(rows):
                for k, right in enumerate(rows):
                    value = block.matrix[j][k]
                    if (value is not None) != (left.se is not None and right.se is not None):
                        self._invalid_field("covariance", "null_mismatch")
                    if value != block.matrix[k][j] or (j == k and value is not None and value <= 0):
                        self._invalid_field("covariance", "matrix_value")

    @property
    def result_sha256(self) -> str:
        """Integrity digest of this typed result, excluding its own digest."""
        return overlap_digest("result", self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return the canonical format-1 result, including its digest."""
        body = self.model_dump(mode="json")
        return canonical_json_bytes(
            {**body, "result_sha256": overlap_digest("result", body)}
        ).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str, *, expected_sha256: str) -> Self:
        """Replay exact canonical bytes against a caller-pinned result digest."""
        body = overlap_wire_payload(payload)
        claimed = body.pop("result_sha256", None)
        if not isinstance(claimed, str) or claimed != expected_sha256:
            refuse(FORMAT_INVALID, field="result_sha256", reason="digest_mismatch")
        try:
            result = cls.model_validate(body)
        except CodedError as exc:
            field = exc.context.get("field", exc.context.get("role", "result"))
            reason = exc.context.get("reason", "invalid_value")
            cause = exc.__cause__
        else:
            typed_body = result.model_dump(mode="json")
            actual = overlap_digest("result", typed_body)
            if actual != expected_sha256:
                refuse(FORMAT_INVALID, field="result_sha256", reason="digest_mismatch")
            if (
                canonical_json_bytes({**typed_body, "result_sha256": actual}).decode("utf-8")
                != payload
            ):
                refuse(FORMAT_INVALID, field="payload", reason="noncanonical_encoding")
            return result
        try:
            refuse(FORMAT_INVALID, field=field, reason=reason)
        except CodedError as exc:
            exc.__cause__ = cause
            raise


def _delivery_conflicts(deliveries: tuple[OverlapDeliveryCount, ...]) -> bool:
    return any(row.n_inconsistent > 0 for row in deliveries)


def _rollout(*, conflicts: bool) -> _Rollout:
    return "as_assigned_with_delivery_conflicts" if conflicts else "as_assigned"


def analyze_overlap(evidence: OverlapEvidence) -> OverlapResult:
    """Analyze the full registered family from validated as-assigned evidence.

    Welch–Satterthwaite intervals use a common Bonferroni allocation. They are
    approximate; neither the band nor empirical support floors certify the
    assignment law or justify general small-sample error guarantees.

    Parameters
    ----------
    evidence : OverlapEvidence
        Bounded evidence for the declared population and window.

    Returns
    -------
    OverlapResult
        The ordered rows, covariance blocks, and joint launch assessment.
    """
    if not isinstance(evidence, OverlapEvidence):
        refuse(EVIDENCE_INVALID, role="evidence", reason="typed_evidence_required", count=1)
    plan = evidence.plan
    rows = _overlap_rows(plan, evidence.cells, target_count=evidence.target_count)
    conflicts = _delivery_conflicts(evidence.delivery_counts)
    return OverlapResult(
        plan=plan,
        plan_sha256=plan.plan_sha256,
        evidence_sha256=evidence.evidence_sha256,
        capture=evidence.capture,
        ingress=evidence.ingress,
        replay_status=evidence.replay_status,
        eligibility_counts=evidence.eligibility_counts,
        gate_counts=evidence.gate_counts,
        delivery_counts=evidence.delivery_counts,
        cells=evidence.cells,
        family_members=plan.claims,
        family_sha256=plan.family_sha256,
        alpha=plan.alpha,
        rows=rows,
        covariance=_covariance(plan, evidence.cells, rows),
        rollout=_rollout(conflicts=conflicts),
        joint_decision=_overlap_joint_decision(rows, delivery_conflicts=conflicts),
    )
