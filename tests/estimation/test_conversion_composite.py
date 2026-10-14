"""A conversion row the count rule routes to the delta method carries its four counts and the
level at which they become dense, and reports the composite p-value: its delta p-value at or
above that level, else the smaller of its exact p-value (read at the nominal tail) and the
level. Read at any level a family reads it at, the composite rejects only where the pipeline
routed by counts at that level rejects, so a family member keeps the interval its own level cuts
while its p-value stays valid below it."""

from __future__ import annotations

import json
import math
import pickle
import time

import numpy as np
import pytest
from scipy.stats import t as student_t

from increment.estimation.binomial_rr import FINITE_SAMPLE_MAX_ARM_SIZE, nuisance_beta, null_p_value
from increment.estimation.conversion_route import (
    CAPPED_NOTE_PREFIX,
    ESCALATED_NOTE_PREFIX,
    dense_level,
    dense_min_count,
)
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.engine import (
    Method,
    _estimate_lift,
    escalate_family_member,
    estimate_lift,
)
from increment.estimation.family import select_exploratory_family
from increment.estimation.results import (
    LiftEstimate,
    conversion_route_label,
)
from increment.semantics.models import ConversionMetric
from tests.estimation._conversion_counts import (
    CONVERSION_METRIC,
    count_summary,
    lift_computation,
    lift_row,
)

_ALTERNATIVES = ("two-sided", "greater", "less")


def _delta_p_value(row: LiftEstimate) -> float:
    """The delta route's own p-value, from the row's persisted log statistics and reference."""
    lift = row.require_lift()
    assert lift.log_mean is not None and lift.log_se is not None and row.reference_df is not None
    z = (lift.log_mean - math.log1p(row.null_lift)) / lift.log_se
    if row.alternative == "greater":
        return float(student_t.sf(z, row.reference_df))
    if row.alternative == "less":
        return float(student_t.cdf(z, row.reference_df))
    return float(2.0 * student_t.sf(abs(z), row.reference_df))


def member_row(
    counts: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    null_lift: float | None = None,
) -> LiftEstimate:
    """The decision row of a conversion contrast estimated as a member of a multiplicity
    family, which reads its p-value below its own level."""
    computation = _estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(*counts),
        control_group="control",
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        family_member=True,
    )
    assert not computation.failures, computation.failures
    (row,) = computation.results
    return row


def _exact_p_value(row: LiftEstimate, alpha: float) -> float:
    """The finite-sample route's p-value against the row's null, read at the nominal tail with
    the nominal nuisance budget: the production primitive the escalation reads."""
    evidence = row.delta_evidence
    assert evidence is not None
    tail = alpha / 2.0 if row.alternative == "two-sided" else alpha
    return null_p_value(
        1.0 + row.null_lift,
        *evidence.counts,
        nuisance_beta(alpha),
        alternative=row.alternative,
        tail=tail,
    )


class TestDeltaRowsCarryTheirCounts:
    def test_a_dense_row_persists_its_counts_dense_level_and_route(self):
        counts = (2_500, 50_000, 2_550, 50_000)
        row = lift_row(counts)
        assert row.reference_kind == "t"
        evidence = row.delta_evidence
        assert evidence is not None
        assert evidence.counts == counts
        assert evidence.dense_level == dense_level(*counts, alternative="two-sided")
        assert evidence.composite is False
        assert conversion_route_label(row) == "delta"
        assert "route='delta'" in repr(row)

    def test_a_finite_sample_row_carries_its_set_and_no_delta_evidence(self):
        row = lift_row((300, 10_000, 330, 10_000))
        assert row.reference_kind == "binomial"
        assert row.delta_evidence is None
        assert conversion_route_label(row) == "finite_sample"
        assert "route='finite_sample'" in repr(row)

    def test_a_mean_metric_row_carries_neither(self):
        from increment.semantics.models import MeanMetric

        rows = count_summary(2_500, 50_000, 2_550, 50_000)
        for row in rows:
            row["metric"] = "m"
        (row,) = estimate_lift(
            metrics=[MeanMetric(name="m", entity="user", fact="m")],
            summary=rows,
            control_group="control",
        ).results
        assert row.delta_evidence is None and row.binomial_set is None
        assert conversion_route_label(row) is None

    def test_an_additive_margin_row_decides_on_its_sidecar_and_carries_no_evidence(self):
        (row,) = estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=count_summary(2_500, 50_000, 2_700, 50_000),
            control_group="control",
            null_abs=0.001,
        ).results
        assert row.reference_kind == "t" and row.null_abs == 0.001
        assert row.delta_evidence is None

    def test_only_the_decision_row_of_a_family_member_reads_the_composite(self):
        """A sensitivity method's p-value never enters a family's selection: it keeps its
        counts and its delta p-value while the decision row reads the composite."""
        computation = _estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=count_summary(2_500, 50_000, 2_700, 50_000),
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="unadjusted_b")],
            method_roles={"unadjusted": "decision", "unadjusted_b": "sensitivity"},
            family_member=True,
        )
        by_role = {row.method_role: row for row in computation.results}
        decision, sensitivity = by_role["decision"], by_role["sensitivity"]
        assert decision.delta_evidence is not None and sensitivity.delta_evidence is not None
        assert decision.delta_evidence.composite is True
        assert sensitivity.delta_evidence.composite is False
        assert decision.delta_evidence.counts == sensitivity.delta_evidence.counts
        assert sensitivity.p_value() == _delta_p_value(sensitivity)
        assert decision.p_value() != sensitivity.p_value()

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_a_row_outside_a_family_reports_its_delta_p_value(self, alternative):
        """A sole primary, a guardrail or any row no family reads keeps the delta p-value it
        reported before counts were persisted, however far below its dense level it lies."""
        x_t = 2_200 if alternative == "less" else 2_800
        row = lift_row((2_500, 50_000, x_t, 50_000), alternative=alternative)
        evidence = row.delta_evidence
        assert evidence is not None and evidence.composite is False
        p_delta = _delta_p_value(row)
        assert p_delta < evidence.dense_level
        assert row.p_value() == p_delta
        assert evidence.escalated_p_value is None and row.note is None

    def test_a_member_read_on_demand_equals_one_estimated_as_a_member(self):
        counts = (2_500, 50_000, 2_800, 50_000)
        on_demand = escalate_family_member(lift_row(counts))
        estimated = member_row(counts)
        assert on_demand.delta_evidence == estimated.delta_evidence
        assert on_demand.p_value() == estimated.p_value()
        assert on_demand.note == estimated.note
        assert escalate_family_member(on_demand) == on_demand


class TestTheCompositePValue:
    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_a_delta_p_value_at_or_above_the_dense_level_is_the_rows_p_value(self, alternative):
        row = member_row((2_500, 50_000, 2_550, 50_000), alternative=alternative)
        evidence = row.delta_evidence
        assert evidence is not None
        p_delta = _delta_p_value(row)
        assert p_delta >= evidence.dense_level
        assert row.p_value() == p_delta
        assert evidence.escalated_p_value is None and evidence.exact_unavailable is None
        assert row.note is None

    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    @pytest.mark.parametrize("null_lift", [0.0, 0.02])
    def test_below_the_dense_level_the_exact_p_value_is_read_at_the_nominal_tail(
        self, alternative, null_lift
    ):
        alpha = 0.05
        row = member_row(
            (2_500, 50_000, 2_700, 50_000),
            alpha=alpha,
            alternative=alternative,
            null_lift=null_lift,
        )
        evidence = row.delta_evidence
        assert evidence is not None
        p_delta = _delta_p_value(row)
        assert p_delta < evidence.dense_level
        exact = _exact_p_value(row, alpha)
        assert evidence.escalated_p_value == exact
        assert row.p_value() == min(exact, evidence.dense_level)
        assert row.note is not None
        prefix = ESCALATED_NOTE_PREFIX if exact < evidence.dense_level else CAPPED_NOTE_PREFIX
        assert row.note.startswith(prefix)

    def test_an_exact_p_value_not_below_the_level_caps_the_row_at_the_level(self):
        row = member_row((2_500, 50_000, 2_644, 50_000))
        evidence = row.delta_evidence
        assert evidence is not None
        assert _delta_p_value(row) < evidence.dense_level
        assert evidence.escalated_p_value is not None
        assert evidence.escalated_p_value >= evidence.dense_level
        assert row.p_value() == evidence.dense_level
        assert row.note is not None and row.note.startswith(CAPPED_NOTE_PREFIX)

    def test_a_level_within_the_nuisance_budget_is_the_p_value_without_an_exact_reading(self):
        """``p_exact >= beta``, so once ``U <= beta`` the composite is ``U`` whatever the exact
        p-value: the row reports the level, says so, and does no exact work."""
        row = member_row((100_000, 1_000_000, 110_000, 1_000_000))
        evidence = row.delta_evidence
        assert evidence is not None
        assert evidence.dense_level <= nuisance_beta(0.05)
        assert _delta_p_value(row) < evidence.dense_level
        assert evidence.escalated_p_value is None and evidence.exact_unavailable is None
        assert row.p_value() == evidence.dense_level
        assert row.note is not None and row.note.startswith(CAPPED_NOTE_PREFIX)

    def test_an_arm_above_the_finite_sample_ceiling_records_the_refusal_and_reports_the_level(
        self,
    ):
        n = 2 * FINITE_SAMPLE_MAX_ARM_SIZE
        row = member_row((50_000, n, 60_000, n))
        evidence = row.delta_evidence
        assert evidence is not None
        assert evidence.dense_level > nuisance_beta(0.05)
        assert evidence.escalated_p_value is None
        assert (
            evidence.exact_unavailable == "estimation.binomial.finite_sample_arm_ceiling_exceeded"
        )
        assert row.p_value() == evidence.dense_level
        assert row.note is not None and row.note.startswith(CAPPED_NOTE_PREFIX)
        assert evidence.exact_unavailable in row.note

    @pytest.mark.parametrize("alternative", _ALTERNATIVES)
    def test_the_composite_agrees_with_the_interval_at_every_level_the_counts_are_dense_at(
        self, alternative
    ):
        """For ``alpha >= U`` the pipeline at ``alpha`` is the delta route: its interval excludes
        the null exactly when the composite is at most ``alpha``, on both sides of the row's
        own level and of its delta p-value."""
        counts = (2_500, 50_000, 2_700, 50_000)
        row = member_row(counts, alpha=0.05, alternative=alternative)
        evidence = row.delta_evidence
        assert evidence is not None
        level = evidence.dense_level
        p_delta = _delta_p_value(row)
        composite = row.p_value()
        assert composite is not None
        grid = sorted(
            {
                level,
                math.nextafter(level, 1.0),
                1.5 * level,
                0.05,
                0.1,
                0.2,
                p_delta,
                math.nextafter(p_delta, 0.0),
                math.nextafter(p_delta, 1.0),
            }
        )
        for alpha in grid:
            if alpha < level:
                continue
            if alternative != "two-sided" and alpha >= 0.5:
                continue
            at_alpha = lift_row(counts, alpha=alpha, alternative=alternative)
            assert at_alpha.reference_kind == "t"
            assert (composite <= alpha) == at_alpha.stat_sig(), alpha

    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    def test_below_the_dense_level_the_composite_rejects_only_where_the_exact_route_does(
        self, alternative
    ):
        """For ``alpha < U`` the pipeline at ``alpha`` is the finite-sample route: wherever the
        composite is at most ``alpha`` that route's interval excludes the null."""
        counts = (2_500, 50_000, 2_750, 50_000)
        row = member_row(counts, alpha=0.05, alternative=alternative)
        evidence = row.delta_evidence
        assert evidence is not None
        composite = row.p_value()
        assert composite is not None and composite < evidence.dense_level
        for alpha in (composite * 0.5, composite, composite * 2.0, evidence.dense_level * 0.9):
            at_alpha = lift_row(counts, alpha=alpha, alternative=alternative)
            assert at_alpha.reference_kind == "binomial"
            if composite <= alpha and composite != alpha:
                assert at_alpha.stat_sig() is True, alpha
            if not at_alpha.stat_sig():
                assert composite >= alpha, alpha


class TestAssembledFromTheProductionPrimitives:
    """``p*`` assembled from ``row_p_value`` of the delta row, ``null_p_value`` at the nominal
    tail with the nominal budget and ``dense_level`` on sampled pairs across the annulus between
    the nominal and the read level, compared with the count-routed pipeline's decision at
    non-lattice read levels, below and above the nominal level."""

    NOMINAL = 0.1
    READS = (0.0123, 0.021, 0.05, 0.0729, 0.13)

    @staticmethod
    def _pairs(
        rng: np.random.Generator, count: int, alternative: str
    ) -> list[tuple[int, int, int, int]]:
        halve = 2.0 if alternative == "two-sided" else 1.0
        lo, hi = dense_min_count(0.1 / halve), dense_min_count(0.0123 / halve)
        pairs = []
        while len(pairs) < count:
            n = int(rng.integers(8 * hi, 20 * hi))
            x_c = int(rng.integers(lo, hi))
            x_t = int(rng.integers(lo, hi))
            if min(x_c, x_t, n - x_c, n - x_t) >= lo and min(x_c, x_t) < hi:
                pairs.append((x_c, n, x_t, n))
        return pairs

    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    def test_the_assembled_composite_rejects_only_where_the_pipeline_at_the_read_level_does(
        self, alternative
    ):
        rng = np.random.default_rng(20261014)
        compared = rejected = escalated = sparse_reads = 0
        for counts in self._pairs(rng, 6, alternative):
            row = member_row(counts, alpha=self.NOMINAL, alternative=alternative)
            evidence = row.delta_evidence
            assert evidence is not None
            p_delta = _delta_p_value(row)
            level = dense_level(*counts, alternative=alternative)
            assert level is not None and level == evidence.dense_level
            if p_delta >= level:
                assembled = p_delta
            else:
                escalated += 1
                assembled = min(_exact_p_value(row, self.NOMINAL), level)
            assert assembled == row.p_value()
            for read in self.READS:
                pipeline = lift_row(counts, alpha=read, alternative=alternative)
                dense = read >= level
                assert (pipeline.reference_kind == "t") == dense
                compared += 1
                sparse_reads += not dense
                if assembled <= read and assembled != read:
                    rejected += 1
                    assert pipeline.stat_sig() is True, (counts, read)
                if dense:
                    assert (assembled <= read) == pipeline.stat_sig(), (counts, read)
        assert compared == 30
        assert rejected > 0 and escalated > 0 and 0 < sparse_reads < compared


class TestFamilyEvidenceIsTheRowsOwnPValue:
    def test_the_decision_bundle_reads_the_composite_of_a_member(self):
        computation = _estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=count_summary(2_500, 50_000, 2_700, 50_000),
            control_group="control",
            family_member=True,
        )
        (row,) = computation.results
        (evidence,) = computation.evidence.values()
        assert isinstance(evidence, PValueEvidence)
        assert evidence.p_value == row.p_value()
        assert row.p_value() != _delta_p_value(row)

    def test_the_decision_bundle_reads_the_delta_p_value_outside_a_family(self):
        computation = lift_computation((2_500, 50_000, 2_700, 50_000))
        (row,) = computation.results
        (evidence,) = computation.evidence.values()
        assert isinstance(evidence, PValueEvidence)
        assert evidence.p_value == row.p_value() == _delta_p_value(row)


def _metric_rows(name: str, counts: tuple[int, int, int, int]) -> list[dict]:
    rows = count_summary(*counts)
    for row in rows:
        row["metric"] = name
    return rows


def _whole_window_rows(cells: dict[str, tuple[int, int, int, int]], *, alpha: float = 0.05):
    rows = []
    for name, counts in cells.items():
        (row,) = estimate_lift(
            metrics=[ConversionMetric(name=name, entity="user", fact=name)],
            summary=_metric_rows(name, counts),
            control_group="control",
            alpha=alpha,
        ).results
        rows.append(row)
    return rows


class TestFcrReissueBelowTheDenseLevel:
    def test_a_selected_row_reissued_below_its_level_takes_the_exact_set_and_agrees_with_discovery(
        self,
    ):
        """Ten hypotheses at ``q = 0.05``: the selected rows' corrected level falls below their
        dense level, so each is reissued as the exact binomial set from its counts, which
        excludes the null exactly when the row is a discovery; unselected rows keep their delta
        interval and counts."""
        cells = {
            "strong_a": (2_500, 50_000, 2_800, 50_000),
            "strong_b": (2_500, 50_000, 2_780, 50_000),
        }
        cells.update({f"null_{i}": (2_500, 50_000, 2_500 + i, 50_000) for i in range(8)})
        rows = _whole_window_rows(cells)
        assert all(row.delta_evidence is not None for row in rows)
        corrected = select_exploratory_family(rows, q=0.05)
        selected = [row for row in corrected if row.discovery]
        assert {row.metric for row in selected} == {"strong_a", "strong_b"}
        for row in selected:
            assert row.family_threshold is not None
            nominal = next(r for r in rows if r.metric == row.metric)
            assert nominal.delta_evidence is not None
            assert row.family_threshold < nominal.delta_evidence.dense_level
            assert row.reference_kind == "binomial" and row.binomial_set is not None
            assert row.delta_evidence is None
            bset = row.binomial_set
            assert (bset.x_c, bset.n_c, bset.x_t, bset.n_t) == nominal.delta_evidence.counts
            assert row.stat_sig() is True
            p_value = row.p_value()
            assert p_value is not None and p_value < row.family_threshold
        for row in corrected:
            if row.discovery:
                continue
            nominal = escalate_family_member(next(r for r in rows if r.metric == row.metric))
            assert row.reference_kind == "t"
            assert row.delta_evidence == nominal.delta_evidence
            assert row.delta_evidence is not None and row.delta_evidence.composite
            assert row.p_value() == nominal.p_value()
            assert row.require_lift().lb == nominal.require_lift().lb

    def test_a_selected_row_reissued_at_or_above_its_level_keeps_the_delta_route_and_evidence(
        self,
    ):
        """Two hypotheses at ``q = 0.2``: the corrected level stays at or above the dense level,
        so the selected row is the Wald interval at that level, still carrying the composite
        evidence the family selected on."""
        cells = {"strong": (2_500, 50_000, 2_800, 50_000), "weak": (2_500, 50_000, 2_520, 50_000)}
        rows = _whole_window_rows(cells)
        corrected = select_exploratory_family(rows, q=0.2)
        strong = next(row for row in corrected if row.metric == "strong")
        nominal = escalate_family_member(next(row for row in rows if row.metric == "strong"))
        assert strong.discovery is True
        assert strong.family_threshold is not None
        assert nominal.delta_evidence is not None
        assert strong.family_threshold >= nominal.delta_evidence.dense_level
        assert strong.reference_kind == "t"
        assert strong.delta_evidence == nominal.delta_evidence
        assert strong.p_value() == nominal.p_value()
        assert strong.stat_sig() is True


class TestReissueRoutesAtTheReportedTail:
    """A selected directional row reports one open bound at the corrected level's whole tail
    (the two-sided parent is requested at half the level and opened at the full tail): the
    reissue routes its counts at that tail, on the persisted path and the source-backed one."""

    COUNTS = (2_600, 20_000, 2_900, 20_000)
    FCR = 0.03

    def _member(self):
        row = member_row(self.COUNTS, alpha=0.05, alternative="greater")
        evidence = row.delta_evidence
        assert evidence is not None
        # Dense at the reported tail, not at half of it.
        assert self.FCR / 2.0 < evidence.dense_level <= self.FCR
        return row

    def test_the_persisted_reissue_keeps_the_delta_route_where_the_reported_tail_is_dense(self):
        from increment.estimation._fcr_reinterval import classify, reinterval_at_fcr

        row = self._member()
        construction, _ = classify(row)
        assert construction == "conversion"
        reissued = reinterval_at_fcr(row, construction, self.FCR)
        assert reissued.reference_kind == "t"
        assert reissued.require_lift().alpha == self.FCR
        assert reissued.require_lift().open_side == "upper"
        assert reissued.delta_evidence == row.delta_evidence

    def test_the_source_backed_reissue_routes_at_the_same_tail(self):
        """``estimate_lift`` requested at half the corrected level for a directional row routes
        at the full tail when told it is a corrected pass, as the persisted path does."""
        from increment.estimation.conversion_route import fcr_route_tail

        computation = _estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=count_summary(*self.COUNTS),
            control_group="control",
            alpha=self.FCR / 2.0,
            alternative="greater",
            route_tail=fcr_route_tail("greater", self.FCR),
        )
        (row,) = computation.results
        assert row.reference_kind == "t"
        plain = lift_row(self.COUNTS, alpha=self.FCR / 2.0, alternative="greater")
        assert plain.reference_kind == "binomial"

    def test_an_arm_above_the_exact_ceiling_is_not_refused_where_the_reported_tail_is_dense(
        self,
    ):
        from increment.estimation._fcr_reinterval import classify, reinterval_at_fcr

        n = 2 * FINITE_SAMPLE_MAX_ARM_SIZE
        row = member_row((50_000, n, 60_000, n), alternative="greater")
        evidence = row.delta_evidence
        assert evidence is not None
        fcr = 1e-5
        assert fcr / 2.0 < evidence.dense_level <= fcr
        construction, _ = classify(row)
        assert construction is not None
        reissued = reinterval_at_fcr(row, construction, fcr)
        assert reissued.reference_kind == "t"
        assert reissued.require_lift().alpha == fcr

    def test_a_reissue_below_the_dense_level_is_cut_as_a_fresh_finite_sample_row(self):
        """Normal additive bounds with no t reference, and the row's other disclosures kept
        while the composite note, which no longer applies, is dropped."""
        from increment.estimation._fcr_reinterval import classify, reinterval_at_fcr

        warning = (
            "unadjusted group-mean comparison under an Observational design -- confounded; "
            "not a causal estimate"
        )
        row = escalate_family_member(
            lift_row((2_500, 50_000, 2_800, 50_000)).model_copy(update={"note": warning})
        )
        assert row.note is not None and row.note.startswith(warning)
        assert ESCALATED_NOTE_PREFIX in row.note
        construction, _ = classify(row)
        assert construction is not None
        reissued = reinterval_at_fcr(row, construction, 0.002)
        fresh = lift_row((2_500, 50_000, 2_800, 50_000), alpha=0.002, mode="finite_sample")
        assert reissued.reference_kind == "binomial"
        assert (reissued.abs_reference_kind, reissued.abs_reference_df) == (None, None)
        assert (reissued.abs_lb, reissued.abs_ub) == (fresh.abs_lb, fresh.abs_ub)
        assert reissued.note == warning
        assert fresh.note is None

    def test_an_exploratory_member_reissued_below_its_level_keeps_its_other_disclosures(self):
        warning = "unadjusted group-mean comparison under an Observational design -- confounded"
        cells = {
            "strong": (2_500, 50_000, 2_800, 50_000),
            **{f"null_{i}": (2_500, 50_000, 2_500 + i, 50_000) for i in range(9)},
        }
        rows = [row.model_copy(update={"note": warning}) for row in _whole_window_rows(cells)]
        corrected = select_exploratory_family(rows, q=0.05)
        strong = next(row for row in corrected if row.metric == "strong")
        assert strong.discovery is True and strong.reference_kind == "binomial"
        assert strong.note == warning
        assert (strong.abs_reference_kind, strong.abs_reference_df) == (None, None)


class TestPersistedRowsRoundTrip:
    @pytest.mark.parametrize("member", [True, False])
    @pytest.mark.parametrize(
        "counts", [(2_500, 50_000, 2_700, 50_000), (2_500, 50_000, 2_550, 50_000)]
    )
    def test_json_and_pickle_preserve_the_p_value_route_and_counts(self, counts, member):
        row = member_row(counts) if member else lift_row(counts)
        from_json = LiftEstimate.model_validate_json(row.model_dump_json())
        from_pickle = pickle.loads(pickle.dumps(row))
        for restored in (from_json, from_pickle):
            assert restored.p_value() == row.p_value()
            assert restored.delta_evidence == row.delta_evidence
            assert conversion_route_label(restored) == "delta"
            assert restored.note == row.note
        payload = json.loads(row.model_dump_json())
        assert payload["delta_evidence"]["x_t"] == counts[2]

    def test_a_row_persisted_before_the_evidence_existed_restores_as_a_plain_wald_row(self):
        """A delta-route conversion row saved without its counts cannot be told from a mean
        metric's row: it restores without evidence, states no route, and its p-value is the
        delta p-value at every level."""
        row = member_row((2_500, 50_000, 2_700, 50_000))
        payload = json.loads(row.model_dump_json())
        del payload["delta_evidence"]
        payload["note"] = None
        restored = LiftEstimate.model_validate(payload)
        assert restored.delta_evidence is None
        assert conversion_route_label(restored) is None
        assert restored.p_value() == _delta_p_value(row)
        assert restored.stat_sig() is row.stat_sig()

    def test_an_exact_reading_on_a_row_no_family_read_is_refused(self):
        row = lift_row((2_500, 50_000, 2_700, 50_000))
        assert row.delta_evidence is not None
        with pytest.raises(Exception) as raised:
            row.delta_evidence.model_copy(update={"escalated_p_value": 0.01}).model_validate(
                {**row.delta_evidence.model_dump(), "escalated_p_value": 0.01}
            )
        assert getattr(raised.value, "code", None) == "estimation.results.delta.dense_level"

    def test_evidence_on_a_non_delta_row_is_refused(self):
        row = lift_row((2_500, 50_000, 2_700, 50_000))
        exact = lift_row((2_500, 50_000, 2_700, 50_000), mode="finite_sample")
        payload = exact.model_dump(mode="python")
        payload["delta_evidence"] = row.delta_evidence
        with pytest.raises(Exception) as raised:
            LiftEstimate.model_validate(payload)
        assert getattr(raised.value, "code", None) == "estimation.results.delta.reference_mismatch"

    @pytest.mark.parametrize("nudge", ["one_float_below", "one_float_above", "far"])
    def test_a_dense_level_that_is_not_the_rules_own_is_refused(self, nudge):
        """The restored level must be dense and one float below it not: a level moved by a
        single float either way, or far off, is refused by code."""
        row = lift_row((2_500, 50_000, 2_700, 50_000))
        assert row.delta_evidence is not None
        level = row.delta_evidence.dense_level
        wrong = {
            "one_float_below": 2.0 * math.nextafter(level / 2.0, 0.0),
            "one_float_above": 2.0 * math.nextafter(level / 2.0, 1.0),
            "far": 0.01,
        }[nudge]
        payload = row.model_dump(mode="python")
        payload["delta_evidence"] = row.delta_evidence.model_copy(update={"dense_level": wrong})
        with pytest.raises(Exception) as raised:
            LiftEstimate.model_validate(payload)
        assert getattr(raised.value, "code", None) == "estimation.results.delta.dense_level"

    def test_restoring_a_row_with_evidence_costs_two_rule_evaluations(self):
        row = lift_row((2_500, 50_000, 2_700, 50_000))
        payload = row.model_dump(mode="python")
        started = time.perf_counter()
        for _ in range(200):
            LiftEstimate.model_validate(payload)
        per_row = (time.perf_counter() - started) / 200
        assert per_row < 2e-3, per_row
