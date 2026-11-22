"""Fixed-look interaction contrasts in a frozen cross-pair family."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
from itertools import batched

from increment._overlap_evidence import OverlapCell
from increment._overlap_refusals import NUMERIC_RANGE, SCAN_INVALID
from increment.errors import CodedError, refuse
from increment.estimation._tails import student_t_isf, two_sided_critical_value
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
    _rounded,
    _simultaneous_interval,
    _welch,
)
from increment.estimation.results import _fraction_endpoint, _quadratic_intervals
from increment.semantics.interaction_scan import (
    SCAN_CONTRASTS,
    CombinationCell,
    CombinationKind,
    CombinationReason,
    CombinationResult,
    CombinationRow,
    InteractionCombination,
    InteractionScan,
    InteractionScanPlan,
    ObservedInventory,
    PairDetail,
    PairEvidenceRef,
    PairInput,
    ScanCapture,
    ScanCellCounts,
    ScanCoverage,
    ScanDecision,
    ScanMemberKey,
    ScanReason,
    ScanRow,
    UnavailablePair,
)
from increment.semantics.overlap import CONTRAST_IDS

__all__ = ["evaluate_combinations", "evaluate_interaction_scan"]

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


def _relative_unavailable(row: ScanRow, *reasons: ScanReason) -> ScanRow:
    return _unavailable_row(
        row.key,
        tuple(dict.fromkeys((*row.reasons, *reasons))),
        cells=row.cells,
        evidence_sha256=row.evidence_sha256,
    )


def _relative_result(
    row: ScanRow, estimate: float, lower: float | None = None, upper: float | None = None
) -> ScanRow:
    decision = None
    if row.availability == "decided" and lower is not None and upper is not None:
        decision = "detected" if lower > 0.0 or upper < 0.0 else "none_detected"
    return row.model_copy(
        update={
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "se": None,
            "df": None,
            "decision": decision,
        }
    )


def _relative_effect(
    row: ScanRow, control: OverlapCell | None, treated: OverlapCell | None, *, tail: float | None
) -> tuple[ScanRow, Fraction | None]:
    if row.estimate is None:
        return row, None
    if control is None or control.anchor is None or control.mean_delta is None:
        return _relative_unavailable(row, ScanReason(code="relative_moments_unavailable")), None
    baseline = Fraction(control.anchor) + Fraction(control.mean_delta)
    if baseline <= 0:
        reason = ScanReason(code="relative_baseline_nonpositive", cell=control.cell)
        return _relative_unavailable(row, reason), None
    if treated is None or treated.anchor is None or treated.mean_delta is None:
        return _relative_unavailable(row, ScanReason(code="relative_moments_unavailable")), None
    effect = Fraction(treated.anchor) + Fraction(treated.mean_delta) - baseline
    lift = effect / baseline
    try:
        estimate = _rounded(lift, metric=row.key.metric, quantity="relative_lift")
        if not tail or row.lower is None or control.m2 is None or treated.m2 is None:
            return _relative_result(row, estimate), lift
        nc, nt = control.n_observed, treated.n_observed
        critical = two_sided_critical_value(
            student_t_isf, 2.0 * tail, min(nc, nt) - 1, what="relative interaction interval"
        )
        squared = Fraction(critical) ** 2
        vc = squared * Fraction(control.m2) / (nc * (nc - 1))
        vt = squared * Fraction(treated.m2) / (nt * (nt - 1))
        # Invert (effect - lift * baseline)^2 <= q² Var(treated - (1 + lift) * control).
        a = baseline**2 - vc
        if a <= 0:
            reason = ScanReason(code="relative_baseline_uncertain", cell=control.cell)
            return _relative_unavailable(row, reason), None
        ((lower, upper),) = _quadratic_intervals(
            a, -2 * (effect * baseline + vc), effect**2 - vc - vt
        )
    except ArithmeticError:
        return _relative_unavailable(row, ScanReason(code=NUMERIC_RANGE.code)), None
    except CodedError as exc:
        if exc.code not in _INTERVAL_REFUSALS:
            raise
        return _relative_unavailable(row, ScanReason(code=exc.code)), None
    return _relative_result(row, estimate, lower, upper), lift


def _relative_difference(
    row: ScanRow, off_fit: tuple[ScanRow, Fraction | None], on_fit: tuple[ScanRow, Fraction | None]
) -> ScanRow:
    off, off_lift = off_fit
    on, on_lift = on_fit
    if row.estimate is None or off_lift is None or on_lift is None:
        return _relative_unavailable(row, *off.reasons, *on.reasons)
    lower = upper = None
    try:
        estimate = _rounded(
            on_lift - off_lift,
            metric=row.key.metric,
            quantity="relative_interaction",
        )
        # Either simultaneous simple-effect interval missing means no bound on their difference.
        if (
            row.lower is not None
            and on.lower is not None
            and on.upper is not None
            and off.lower is not None
            and off.upper is not None
        ):
            lower = _fraction_endpoint(Fraction(on.lower) - Fraction(off.upper), upper=False)
            upper = _fraction_endpoint(Fraction(on.upper) - Fraction(off.lower), upper=True)
    except ArithmeticError:
        return _relative_unavailable(row, ScanReason(code=NUMERIC_RANGE.code))
    except CodedError as exc:
        if exc.code != NUMERIC_RANGE.code:
            raise
        return _relative_unavailable(row, ScanReason(code=exc.code))
    return _relative_result(row, estimate, lower, upper)


def _relative_interaction_rows(scan: InteractionScan, metric: str) -> tuple[ScanRow, ...]:
    """Project adjusted simple lifts to their difference, separately from the absolute view."""
    rows = (row for row in scan.rows if row.key.metric == metric)
    members = scan.plan.family_size
    tail = (
        _member_tail(scan.plan.scope.alpha, members)
        if scan.coverage == "complete" and members
        else None
    )
    result = []
    for detail, (off, on, direct) in zip(
        scan.details, batched(rows, len(SCAN_CONTRASTS)), strict=True
    ):
        cells = {cell.cell: cell for cell in detail.cells if cell.metric == metric}
        off_treated, on_control = ("01", "10") if detail.focal_side == "B" else ("10", "01")
        off_fit = _relative_effect(off, cells.get("00"), cells.get(off_treated), tail=tail)
        on_fit = _relative_effect(on, cells.get(on_control), cells.get("11"), tail=tail)
        result.extend((off_fit[0], on_fit[0], _relative_difference(direct, off_fit, on_fit)))
    return tuple(result)


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


def _combination_support(
    cells: tuple[CombinationCell, ...], kind: CombinationKind, assignment: str | None
) -> tuple[CombinationCell, ...]:
    if kind == "effect":
        return tuple(cell for cell in cells if cell.cell[1:] == assignment)
    if kind == "change":
        return cells
    fixed = "1" if kind == "with" else "0"
    return tuple(cell for cell in cells if cell.cell.endswith(fixed))


def _combination_fit(
    cells: tuple[CombinationCell, ...],
    kind: CombinationKind,
    assignment: str | None,
    *,
    target_count: int,
    tail: float | None,
    unverified_reason: str | None,
) -> CombinationRow:
    if not cells:
        return CombinationRow(kind, assignment, reasons=(CombinationReason("joint_feed_missing"),))
    support = _combination_support(cells, kind, assignment)
    raw = _reasons(support, target_count=target_count, claimed=tail is not None)
    availability = _availability(raw, claimed=tail is not None)
    reasons = tuple(CombinationReason(code, key) for key, code in raw)
    if availability == "unavailable":
        return CombinationRow(kind, assignment, reasons=reasons)
    if unverified_reason is not None:
        reasons += (CombinationReason(unverified_reason),)
    half = len(support) // 2
    # Pair the two focal settings. The other varied factors determine the sign.
    end = None if kind == "change" else -1
    signs = tuple(
        (-1) ** cell.cell[1:end].count("0") if kind != "effect" else 1 for cell in support[:half]
    )
    exact = (*(-sign for sign in signs), *signs)
    weights = tuple(map(float, exact))
    pairs = tuple((i, half + i) if sign > 0 else (half + i, i) for i, sign in enumerate(signs))
    metric = support[0].metric
    try:
        estimate = _point(support, weights, exact, pairs, metric=metric)
    except CodedError as exc:
        if exc.code != NUMERIC_RANGE.code:
            raise
        return CombinationRow(kind, assignment, reasons=(*reasons, CombinationReason(exc.code)))
    lower = upper = None
    decision: ScanDecision | None = None
    if tail and availability != "point_only":
        try:
            se, df = _welch(weights, support)
            lower, upper = _simultaneous_interval(estimate, se, df, tail=tail, metric=metric)
        except CodedError as exc:
            if exc.code not in _INTERVAL_REFUSALS:
                raise
            availability = "descriptive"
            reasons += (CombinationReason(exc.code),)
        else:
            if availability == "decided":
                decision = "detected" if lower > 0 or upper < 0 else "none_detected"
    return CombinationRow(kind, assignment, estimate, lower, upper, availability, reasons, decision)


def evaluate_combinations(
    plan: InteractionScanPlan,
    captured: Mapping[InteractionCombination, tuple[CombinationCell, ...]],
    *,
    coverage: ScanCoverage,
    observed: ObservedInventory | None,
) -> tuple[CombinationResult, ...]:
    """Estimate the full registered family, keeping unavailable comparisons in the divisor."""
    registration = plan.combination_registration
    if captured.keys() - set(plan.combinations):
        refuse(SCAN_INVALID, field="combination_feeds", reason="unknown_combination")
    if registration is None:
        return ()
    if observed is not None:
        plan.scope.require_cutoff(observed.cutoff)
    reason = None
    if coverage != "complete" or plan.roster_coverage != "complete":
        reason = "incomplete_inventory"
    elif (
        observed is None or observed.combination_registration_sha256 != registration.registration_id
    ):
        reason = "joint_registration_unverified"
    elif observed.coverage_for(plan.scope) != "complete" or not plan.matches_observed_roster(
        observed
    ):
        reason = "incomplete_inventory"
    family_size = plan.combination_family_size
    tail = _member_tail(plan.scope.alpha, family_size) if reason is None else None
    if tail == 0.0:
        tail, reason = None, "estimation.tails.unresolvable"
    results = []
    for declared in plan.combinations:
        by_metric = {}
        cells = captured.get(declared)
        assignments = declared.assignments
        if cells is not None:
            by_metric = {(cell.metric, cell.cell): cell for cell in cells}
            expected = {(metric.name, bits) for metric in plan.metrics for bits in assignments}
            if by_metric.keys() != expected or len(by_metric) != len(cells):
                refuse(
                    SCAN_INVALID, field="combination_cells", reason="incomplete_or_duplicate_cells"
                )
        for metric in plan.metrics:
            group = tuple(by_metric[metric.name, bits] for bits in assignments) if by_metric else ()
            for cell in group:
                cell.validate_metric_kind(binary=metric.kind == "binary_mean")
            target_count = sum(cell.n_assigned for cell in group)
            rows = tuple(
                _combination_fit(
                    group,
                    kind,
                    assignment,
                    target_count=target_count,
                    tail=tail,
                    unverified_reason=reason,
                )
                for kind, assignment in declared.contrasts
            )
            results.append(CombinationResult(declared, metric, rows, group, family_size))
    return tuple(results)
