"""Fixed-horizon declarations and replay for two independent experiments.

A declaration binds the population and archived assignment law. It cannot
certify a producer's independence, preregistration, or snapshot promises.
"""

from __future__ import annotations

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, NoReturn, Self

from pydantic import Field, StrictFloat, StrictInt

from increment._canonical import canonical_json_bytes
from increment._overlap_evidence import (
    CELL_IDS,
    DELIVERY_KEYS,
    FACTOR_IDS,
    FEED_ROLES,
    GATE_KEYS,
    MAX_EXACT_INTEGER,
    OVERLAP_FORMAT,
    Count,
    FeedRole,
    Identity,
    Instant,
    Ordered,
    OverlapIngress,
    OverlapModel,
    Sha256,
    overlap_binary_moments,
    overlap_digest,
    overlap_wire_payload,
)
from increment._overlap_evidence import CellId as CellId
from increment._overlap_evidence import OverlapCell as OverlapCell
from increment._overlap_evidence import OverlapDeliveryCount as OverlapDeliveryCount
from increment._overlap_refusals import (
    CAPTURE_INVALID,
    EVIDENCE_INVALID,
    FORMAT_INVALID,
    LAW_UNSUPPORTED,
    PLAN_INVALID,
)
from increment.errors import CodedError, refuse

ContrastId = Literal[
    "a_marginal",
    "b_marginal",
    "a_given_b_off",
    "a_given_b_on",
    "b_given_a_off",
    "b_given_a_on",
    "joint_launch",
    "interaction",
]
CONTRAST_IDS: tuple[ContrastId, ...] = (
    "a_marginal",
    "b_marginal",
    "a_given_b_off",
    "a_given_b_on",
    "b_given_a_off",
    "b_given_a_on",
    "joint_launch",
    "interaction",
)
OverlapOrigin = Literal["sdk_v1", "external"]
OverlapTarget = Literal["jointly_eligible", "jointly_eligible_and_enrolled"]


class OverlapFactor(OverlapModel):
    """Archived identity and integer arm weights from 1 through ``2**53 - 1``."""

    target: Identity
    experiment: Identity
    revision: Identity
    off: Identity
    on: Identity
    off_weight: Annotated[StrictInt, Field(gt=0, le=MAX_EXACT_INTEGER)]
    on_weight: Annotated[StrictInt, Field(gt=0, le=MAX_EXACT_INTEGER)]
    arm_draw_id: Identity
    layer: Identity | None

    def _validate_overlap(self) -> Self:
        if self.off == self.on:
            refuse(PLAN_INVALID, field="on", reason="duplicate_arm")
        return self

    @property
    def arm_probabilities(self) -> tuple[Fraction, Fraction]:
        """Exact off and on probabilities, in that order."""
        total = self.off_weight + self.on_weight
        return Fraction(self.off_weight, total), Fraction(self.on_weight, total)


class OverlapMetric(OverlapModel):
    """Additive per-unit outcome, defined independently of the assignments."""

    name: Identity
    unit: Identity
    definition_sha256: Sha256
    kind: Literal["mean", "binary_mean"]
    role: Literal["primary", "guardrail", "secondary"]
    better: Literal["higher", "lower"]
    margin_abs: StrictFloat = 0.0

    def _validate_overlap(self) -> Self:
        if not isfinite(self.margin_abs):
            refuse(PLAN_INVALID, field="margin_abs", reason="invalid_value")
        if self.role == "guardrail" and self.margin_abs <= 0:
            refuse(PLAN_INVALID, field="margin_abs", reason="guardrail_requires_positive_margin")
        if self.role != "guardrail" and self.margin_abs != 0:
            refuse(PLAN_INVALID, field="margin_abs", reason="margin_requires_guardrail")
        return self


class OverlapClaim(OverlapModel):
    """One prespecified claim in the simultaneous family."""

    metric: Identity
    contrast: ContrastId
    alternative: Literal["benefit", "difference", "noninferiority"]


class OverlapLaw(OverlapModel):
    """Versioned declaration of the assignment law conditional on entry gates."""

    scheme: Literal["independent_unit_bernoulli"]
    origin: OverlapOrigin
    archive_sha256: Ordered[tuple[Sha256, Sha256]]
    gate_basis: Literal["common_entry_archived_draws", "preassigned_roster"]
    admission: Literal["both_guaranteed", "joint_gate_required", "joint_gate_impossible"]
    rule_source_id: Identity
    rule_version: Identity
    serving_barrier_id: Identity

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        if field == "scheme" and reason == "invalid_value":
            refuse(LAW_UNSUPPORTED, reason="unsupported_scheme", factors=FACTOR_IDS)
        super()._invalid_field(field, reason)


class OverlapPlan(OverlapModel):
    """Prespecified population, binary assignments, fixed window, and claims."""

    study_id: Identity
    unit_namespace: Identity
    population_id: Identity
    eligibility_rule_sha256: Sha256
    enrollment_rule_sha256: Sha256
    registered_at: Instant
    entry_open: Instant
    start: Instant
    end: Instant
    target: OverlapTarget
    a: OverlapFactor
    b: OverlapFactor
    law: OverlapLaw
    metrics: Ordered[tuple[OverlapMetric, ...]]
    claims: Ordered[tuple[OverlapClaim, ...]]
    alpha: StrictFloat

    @property
    def cell_probabilities(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        """Exact joint probabilities in cell order 00, 10, 01, 11."""
        a0, a1 = self.a.arm_probabilities
        b0, b1 = self.b.arm_probabilities
        return a0 * b0, a1 * b0, a0 * b1, a1 * b1

    @property
    def law_sha256(self) -> str:
        """Identity of the declared assignment law and bound factors."""
        return overlap_digest("law", self.model_dump(mode="json", include={"law", "a", "b"}))

    @property
    def plan_sha256(self) -> str:
        """Identity of the entire registered plan."""
        return overlap_digest("plan", self.model_dump(mode="json"))

    @property
    def family_sha256(self) -> str:
        """Identity of the ordered family, bound to its registered plan."""
        return overlap_digest(
            "family",
            {
                "plan_sha256": self.plan_sha256,
                "claims": [claim.model_dump(mode="json") for claim in self.claims],
            },
        )

    def _check_law(self) -> None:
        a, b, law = self.a, self.b, self.law
        if a.layer is not None and a.layer == b.layer:
            refuse(LAW_UNSUPPORTED, reason="same_layer", factors=FACTOR_IDS)
        if a.experiment == b.experiment:
            refuse(LAW_UNSUPPORTED, reason="duplicate_experiment", factors=FACTOR_IDS)
        if a.arm_draw_id == b.arm_draw_id:
            refuse(LAW_UNSUPPORTED, reason="shared_arm_draw", factors=FACTOR_IDS)
        if a.target == b.target:
            refuse(LAW_UNSUPPORTED, reason="duplicate_target", factors=FACTOR_IDS)
        if law.admission == "joint_gate_impossible":
            refuse(LAW_UNSUPPORTED, reason="joint_gate_impossible", factors=FACTOR_IDS)
        if self.target == "jointly_eligible" and law.admission != "both_guaranteed":
            refuse(LAW_UNSUPPORTED, reason="admission_not_guaranteed", factors=FACTOR_IDS)
        if law.origin == "sdk_v1":
            if (law.gate_basis, law.rule_source_id, law.rule_version) != (
                "common_entry_archived_draws",
                "increment.assignment.protocol-v1",
                "1",
            ):
                refuse(LAW_UNSUPPORTED, reason="incompatible_sdk_law", factors=FACTOR_IDS)
            for factor, declaration in zip(FACTOR_IDS, (a, b), strict=True):
                if declaration.target != (declaration.layer or declaration.experiment):
                    refuse(LAW_UNSUPPORTED, reason="incompatible_sdk_target", factors=(factor,))
                if declaration.off_weight + declaration.on_weight != 10_000:
                    refuse(LAW_UNSUPPORTED, reason="incompatible_sdk_weights", factors=(factor,))
        elif law.gate_basis != "preassigned_roster":
            refuse(LAW_UNSUPPORTED, reason="external_gates_not_preassigned", factors=FACTOR_IDS)
        if not all(float(probability) > 0 for probability in self.cell_probabilities):
            refuse(LAW_UNSUPPORTED, reason="joint_probability_unrepresentable", factors=FACTOR_IDS)

    def _ordered_claims(self) -> tuple[OverlapClaim, ...]:
        metrics = {metric.name: metric for metric in self.metrics}
        if len(metrics) != len(self.metrics):
            refuse(PLAN_INVALID, field="metrics", reason="duplicate_metric")
        claims: dict[tuple[str, ContrastId], OverlapClaim] = {}
        has_primary = False
        joint_primary = False
        for claim in self.claims:
            metric = metrics.get(claim.metric)
            if metric is None:
                refuse(PLAN_INVALID, field="claims", reason="unknown_metric")
            key = (claim.metric, claim.contrast)
            if key in claims:
                refuse(PLAN_INVALID, field="claims", reason="duplicate_claim")
            if (claim.alternative == "noninferiority") != (metric.role == "guardrail"):
                refuse(PLAN_INVALID, field="claims", reason="alternative_incompatible_with_role")
            if metric.role == "primary":
                has_primary = True
                if claim.contrast == "joint_launch":
                    joint_primary = True
                    if claim.alternative != "benefit":
                        refuse(
                            PLAN_INVALID, field="claims", reason="joint_primary_requires_benefit"
                        )
            claims[key] = claim
        if not has_primary:
            refuse(PLAN_INVALID, field="claims", reason="missing_primary_claim")
        if joint_primary and any(
            metric.role != "secondary" and (metric.name, "joint_launch") not in claims
            for metric in self.metrics
        ):
            refuse(PLAN_INVALID, field="claims", reason="incomplete_joint_launch")
        return tuple(
            claims[key]
            for metric in self.metrics
            for contrast in CONTRAST_IDS
            if (key := (metric.name, contrast)) in claims
        )

    def _validate_overlap(self) -> Self:
        if not self.registered_at < self.entry_open < self.start < self.end:
            refuse(PLAN_INVALID, field="window", reason="registration_or_window_out_of_order")
        if not 0 < self.alpha < 1:
            refuse(PLAN_INVALID, field="alpha", reason="invalid_value")
        self._check_law()
        object.__setattr__(self, "claims", self._ordered_claims())
        return self


class OverlapFeed(OverlapModel):
    """Declared source identity, immutable version, and certified coverage."""

    role: FeedRole
    source_id: Identity
    version: Identity
    recipe_sha256: Sha256
    covered_from: Instant
    complete_through: Instant | None

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(CAPTURE_INVALID, role="feeds", reason=reason)

    def _validate_overlap(self) -> Self:
        if self.complete_through is not None and self.complete_through < self.covered_from:
            refuse(CAPTURE_INVALID, role=self.role, reason="coverage_out_of_order")
        return self


class OverlapCapture(OverlapModel):
    """A declared coherent generation of sources, checked against its plan.

    A version is a producer's content identifier, not a wall-clock cutoff.
    The acquisition path must bind the identifiers to the supplied sources.
    """

    generation_id: Identity
    origin: OverlapOrigin
    law_sha256: Sha256
    archive_sha256: Ordered[tuple[Sha256, Sha256]]
    cutoff: Instant
    feeds: Ordered[tuple[OverlapFeed, ...]]

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(CAPTURE_INVALID, role="snapshot", reason=reason)

    def _validate_overlap(self) -> Self:
        by_role: dict[FeedRole, OverlapFeed] = {}
        for feed in self.feeds:
            if feed.role in by_role:
                refuse(CAPTURE_INVALID, role=feed.role, reason="duplicate_feed")
            if feed.covered_from > self.cutoff or (
                feed.complete_through is not None and feed.complete_through > self.cutoff
            ):
                refuse(CAPTURE_INVALID, role=feed.role, reason="coverage_after_cutoff")
            by_role[feed.role] = feed
        for role in FEED_ROLES:
            if role not in by_role:
                refuse(CAPTURE_INVALID, role=role, reason="missing_feed")
        object.__setattr__(self, "feeds", tuple(by_role[role] for role in FEED_ROLES))
        return self

    def feed(self, role: FeedRole) -> OverlapFeed:
        """Return the declaration for one physical relation."""
        for feed in self.feeds:
            if feed.role == role:
                return feed
        refuse(CAPTURE_INVALID, role="feeds", reason="unknown_role")

    def validate_for(self, plan: OverlapPlan) -> None:
        """Refuse contradictory law or coverage declarations before acquisition."""
        if not isinstance(plan, OverlapPlan):
            refuse(PLAN_INVALID, field="plan", reason="invalid_value")
        if self.origin != plan.law.origin:
            refuse(CAPTURE_INVALID, role="decisions", reason="origin_mismatch")
        if self.archive_sha256 != plan.law.archive_sha256:
            refuse(CAPTURE_INVALID, role="decisions", reason="archive_mismatch")
        if self.law_sha256 != plan.law_sha256:
            refuse(CAPTURE_INVALID, role="decisions", reason="law_mismatch")
        if self.cutoff < plan.start:
            refuse(CAPTURE_INVALID, role="snapshot", reason="cutoff_before_start")
        for role in ("roster", "decisions", "deliveries", "outcomes"):
            feed = self.feed(role)
            covered_from = plan.start if role == "outcomes" else plan.entry_open
            if feed.covered_from > covered_from:
                refuse(CAPTURE_INVALID, role=role, reason="coverage_starts_late")
            if role == "outcomes":
                continue
            required = min(self.cutoff, plan.end) if role == "deliveries" else plan.start
            if feed.complete_through is None or feed.complete_through < required:
                refuse(CAPTURE_INVALID, role=role, reason="incomplete_coverage")

    def outcomes_cover(self, plan: OverlapPlan) -> bool:
        """Whether the declared outcome feed covers the fixed window."""
        watermark = self.feed("outcomes").complete_through
        return self.cutoff >= plan.end and watermark is not None and watermark >= plan.end


class OverlapEvidence(OverlapModel):
    """Bounded, checked evidence that retains no source rows or unit identities.

    Eligibility counts follow 00, 10, 01, 11 (true A is the first bit).
    Gate counts follow the A-major disposition cross-tab among both eligible.
    Delivery counts follow cell order, then A/B, with outside-target last.
    Cells follow declared metrics, then 00, 10, 01, 11.
    All counters fit the exact canonical JSON integer domain (0 to ``2**53 - 1``).
    """

    plan: OverlapPlan
    capture: OverlapCapture
    eligibility_counts: Ordered[tuple[Count, Count, Count, Count]]
    gate_counts: Ordered[tuple[Count, ...]]
    delivery_counts: Ordered[tuple[OverlapDeliveryCount, ...]]
    cells: Ordered[tuple[OverlapCell, ...]]
    ingress: OverlapIngress
    replay_status: Literal["not_certified"] = "not_certified"
    format: StrictInt = OVERLAP_FORMAT

    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(EVIDENCE_INVALID, role=field, reason=reason, count=1)

    @property
    def target_count(self) -> int:
        """Count of jointly randomized units in the declared population."""
        return self.gate_counts[0]

    def _check_deliveries(self, assigned: dict[CellId, int]) -> None:
        if tuple((row.cell, row.factor) for row in self.delivery_counts) != DELIVERY_KEYS:
            self._invalid_field("deliveries", "delivery_order")
        for row in self.delivery_counts:
            if row.cell == "outside_target":
                if row.n_without_delivery or row.n_matching:
                    self._invalid_field("deliveries", "outside_target_count")
                continue
            delivered = assigned[row.cell] - row.n_without_delivery
            events = row.n_matching + row.n_inconsistent
            if not 0 <= delivered <= events or (delivered == 0 and events != 0):
                self._invalid_field("deliveries", "delivery_count_mismatch")

    @staticmethod
    def _check_binary(cell: OverlapCell) -> None:
        success = cell.n_success
        if success is None:
            refuse(EVIDENCE_INVALID, role="cells", reason="missing_success_count", count=1)
        if cell.anchor is None:
            return
        observed = (cell.anchor, cell.mean_delta, cell.m2)
        if observed != overlap_binary_moments(cell.n_observed, success):
            refuse(EVIDENCE_INVALID, role="cells", reason="binary_moments_mismatch", count=1)

    def _validate_overlap(self) -> Self:
        if self.format != OVERLAP_FORMAT:
            refuse(FORMAT_INVALID, field="format", reason="unsupported_version")
        self.capture.validate_for(self.plan)
        both = self.eligibility_counts[3]
        if len(self.gate_counts) != len(GATE_KEYS) or sum(self.gate_counts) != both:
            self._invalid_field("decisions", "gate_count_mismatch")
        if self.plan.law.admission == "both_guaranteed" and self.target_count != both:
            self._invalid_field("decisions", "admission_count_mismatch")
        keys = tuple((metric.name, cell) for metric in self.plan.metrics for cell in CELL_IDS)
        if tuple((cell.metric, cell.cell) for cell in self.cells) != keys:
            self._invalid_field("cells", "cell_order")
        assigned = {cell.cell: cell.n_assigned for cell in self.cells[: len(CELL_IDS)]}
        if sum(assigned.values()) != self.target_count:
            self._invalid_field("cells", "target_count_mismatch")
        kinds = {metric.name: metric.kind for metric in self.plan.metrics}
        complete = self.capture.outcomes_cover(self.plan)
        for cell in self.cells:
            if cell.n_assigned != assigned[cell.cell]:
                self._invalid_field("cells", "assigned_count_mismatch")
            if not complete and cell.n_pending != cell.n_assigned:
                self._invalid_field("cells", "followup_not_complete")
            if kinds[cell.metric] == "binary_mean":
                self._check_binary(cell)
            elif cell.n_success is not None:
                self._invalid_field("cells", "unexpected_success_count")
        self._check_deliveries(assigned)
        return self

    @property
    def evidence_sha256(self) -> str:
        """Integrity digest of format, plan, capture and bounded evidence."""
        return overlap_digest("evidence", self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Return the exact format-1 evidence document, including its digest."""
        body = self.model_dump(mode="json")
        digest = overlap_digest("evidence", body)
        return canonical_json_bytes({**body, "evidence_sha256": digest}).decode("utf-8")

    @classmethod
    def from_json(cls, payload: str, *, expected_sha256: str) -> Self:
        """Check format, population, counts, order, and a caller-pinned digest."""
        body = overlap_wire_payload(payload)
        claimed = body.pop("evidence_sha256", None)
        try:
            evidence = cls.model_validate(body)
        except CodedError as exc:
            field = exc.context.get("field", exc.context.get("role", "evidence"))
            reason = exc.context.get("reason", "invalid_value")
            cause = exc.__cause__
        else:
            actual = evidence.evidence_sha256
            if claimed != actual or expected_sha256 != actual:
                refuse(FORMAT_INVALID, field="evidence_sha256", reason="digest_mismatch")
            if evidence.to_json() != payload:
                refuse(FORMAT_INVALID, field="payload", reason="noncanonical_encoding")
            return evidence
        try:
            refuse(FORMAT_INVALID, field=field, reason=reason)
        except CodedError as exc:
            exc.__cause__ = cause
            raise
