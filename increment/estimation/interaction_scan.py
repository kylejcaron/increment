"""Fixed-look interaction contrasts in a frozen cross-pair family."""

from __future__ import annotations

from collections.abc import Iterator

from increment._overlap_evidence import OverlapCell
from increment._overlap_refusals import NUMERIC_RANGE, SCAN_INVALID
from increment.errors import CodedError, refuse
from increment.estimation.overlap import (
    _PAIRS,
    _availability,
    _cell_groups,
    _delivery_conflicts,
    _exact_weights,
    _ExactWeights,
    _float_weights,
    _member_tail,
    _point,
    _reasons,
    _required_cells,
    _simultaneous_interval,
    _welch,
)
from increment.semantics.interaction_scan import (
    SCAN_CONTRASTS,
    InteractionScan,
    InteractionScanPlan,
    PairDetail,
    PairEvidenceRef,
    PairInput,
    ScanCapture,
    ScanCellCounts,
    ScanDecision,
    ScanMemberKey,
    ScanReason,
    ScanRow,
    UnavailablePair,
)
from increment.semantics.overlap import CONTRAST_IDS

__all__ = ["evaluate_interaction_scan"]

_FOCAL_INDICES = {
    "A": tuple(CONTRAST_IDS.index(key) for key in ("a_given_b_off", "a_given_b_on", "interaction")),
    "B": tuple(CONTRAST_IDS.index(key) for key in ("b_given_a_off", "b_given_a_on", "interaction")),
}
_INTERVAL_REFUSALS = frozenset((NUMERIC_RANGE.code, "estimation.tails.unresolvable"))


def _unavailable_row(
    key: ScanMemberKey,
    reasons: tuple[ScanReason, ...],
    *,
    cells: tuple[ScanCellCounts, ...] = (),
    evidence_sha256: str | None = None,
) -> ScanRow:
    return ScanRow(
        key=key,
        estimate=None,
        se=None,
        df=None,
        lower=None,
        upper=None,
        availability="unavailable",
        reasons=reasons,
        cells=cells,
        evidence_sha256=evidence_sha256,
        decision=None,
    )


def _contrast(
    key: ScanMemberKey,
    group: tuple[OverlapCell, ...],
    index: int,
    exact_weights: _ExactWeights,
    *,
    target_count: int,
    tail: float | None,
    evidence_sha256: str,
) -> ScanRow:
    support = _required_cells(exact_weights, group)
    counts = tuple(
        ScanCellCounts(
            cell=cell.cell,
            n_assigned=cell.n_assigned,
            n_observed=cell.n_observed,
            n_missing=cell.n_missing,
            n_pending=cell.n_pending,
        )
        for cell in support
    )
    raw_reasons = _reasons(support, target_count=target_count, claimed=tail is not None)
    availability = _availability(raw_reasons, claimed=tail is not None)
    reasons = tuple(ScanReason(code=code, cell=cell) for cell, code in raw_reasons)
    if availability == "unavailable":
        return _unavailable_row(key, reasons, cells=counts, evidence_sha256=evidence_sha256)

    weights = _float_weights(exact_weights)
    try:
        estimate = _point(group, weights, exact_weights, _PAIRS[index], metric=key.metric)
    except CodedError as exc:
        if exc.code != NUMERIC_RANGE.code:
            raise
        return _unavailable_row(
            key,
            (*reasons, ScanReason(code=exc.code)),
            cells=counts,
            evidence_sha256=evidence_sha256,
        )

    se = df = lower = upper = None
    decision: ScanDecision | None = None
    if availability != "point_only":
        se, df = _welch(weights, group)
        if tail is None:
            reasons = (*reasons, ScanReason(code="incomplete_inventory"))
        else:
            try:
                lower, upper = _simultaneous_interval(
                    estimate, se, df, tail=tail, metric=key.metric
                )
            except CodedError as exc:
                if exc.code not in _INTERVAL_REFUSALS:
                    raise
                availability = "descriptive"
                reasons = (*reasons, ScanReason(code=exc.code))
            else:
                if availability == "decided":
                    decision = "detected" if lower > 0.0 or upper < 0.0 else "none_detected"

    return ScanRow(
        key=key,
        estimate=estimate,
        se=se,
        df=df,
        lower=lower,
        upper=upper,
        availability=availability,
        reasons=reasons,
        cells=counts,
        evidence_sha256=evidence_sha256,
        decision=decision,
    )


def _detail(pair: PairInput) -> PairDetail:
    value = pair.value
    if isinstance(value, UnavailablePair):
        return PairDetail(
            candidate=pair.candidate,
            status="unavailable",
            reasons=value.reasons,
            focal_side=None,
            evidence=None,
            eligibility_counts=None,
            gate_counts=None,
            delivery_counts=(),
            cells=(),
        )

    evidence = value.evidence
    plan = evidence.plan
    return PairDetail(
        candidate=pair.candidate,
        status="captured",
        reasons=(ScanReason(code="delivery_conflicts"),)
        if _delivery_conflicts(evidence.delivery_counts)
        else (),
        focal_side=value.focal_side,
        evidence=PairEvidenceRef(
            study_id=plan.study_id,
            generation_id=evidence.capture.generation_id,
            plan_sha256=plan.plan_sha256,
            family_sha256=plan.family_sha256,
            evidence_sha256=evidence.evidence_sha256,
        ),
        eligibility_counts=evidence.eligibility_counts,
        gate_counts=evidence.gate_counts,
        delivery_counts=evidence.delivery_counts,
        cells=evidence.cells,
    )


def _pair_rows(
    pair: PairInput, plan: InteractionScanPlan, detail: PairDetail, *, tail: float | None
) -> Iterator[ScanRow]:
    value = pair.value
    digest = detail.evidence.evidence_sha256 if detail.evidence is not None else None
    if isinstance(value, UnavailablePair) or pair.candidate.reasons:
        pair_reasons = (
            value.reasons if isinstance(value, UnavailablePair) else pair.candidate.reasons
        )
        for metric in plan.metrics:
            for contrast in SCAN_CONTRASTS:
                key = ScanMemberKey(pair=pair.candidate.key, metric=metric.name, contrast=contrast)
                yield _unavailable_row(key, pair_reasons, evidence_sha256=digest)
        return

    evidence = value.evidence
    assert digest is not None
    groups = {group[0].metric: group for group in _cell_groups(evidence.cells)}
    issues = {issue.metric: issue.reasons for issue in value.metric_issues}
    weights = _exact_weights(evidence.plan)
    indices = _FOCAL_INDICES[value.focal_side]
    for metric in plan.metrics:
        group = groups.get(metric.name)
        reasons = issues.get(metric.name, ())
        for contrast, index in zip(SCAN_CONTRASTS, indices, strict=True):
            key = ScanMemberKey(pair=pair.candidate.key, metric=metric.name, contrast=contrast)
            if reasons or group is None:
                yield _unavailable_row(
                    key, reasons or (ScanReason(code="metric_missing"),), evidence_sha256=digest
                )
            else:
                yield _contrast(
                    key,
                    group,
                    index,
                    weights[index],
                    target_count=evidence.target_count,
                    tail=tail,
                    evidence_sha256=digest,
                )


def evaluate_interaction_scan(capture: ScanCapture) -> InteractionScan:
    """Estimate the focal simple effects and their difference from qualified pairs.

    The contrast is a signed sum of four as-assigned group means, with zero
    weight on groups irrelevant to a simple effect. Independent units within
    each pair give a diagonal covariance of those means. The intervals use
    the Welch–Satterthwaite approximation and two tails per planned slot,
    including unavailable slots. Bonferroni does not require independent pairs.
    Neither the intervals nor the support floors certify the assignment law.
    """
    if not isinstance(capture, ScanCapture):
        refuse(SCAN_INVALID, field="capture", reason="typed_capture_required")
    plan = capture.plan
    members = plan.family_size
    tail = (
        _member_tail(plan.scope.alpha, members)
        if capture.coverage == "complete" and members
        else None
    )
    details = []
    rows = []
    for pair in capture.pairs:
        detail = _detail(pair)
        details.append(detail)
        rows.extend(_pair_rows(pair, plan, detail, tail=tail))
    return InteractionScan(
        plan=plan,
        capture_id=capture.capture_id,
        observed_inventory=capture.observed_inventory,
        coverage=capture.coverage,
        rows=tuple(rows),
        details=tuple(details),
    )
