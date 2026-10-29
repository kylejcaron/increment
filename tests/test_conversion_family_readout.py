"""Declaring a multiplicity plan over conversion secondaries no longer moves their intervals: a
family member is routed and cut at its own level, its p-value is the composite its counts
carry, and only a selected member's interval is reissued at the family's corrected level."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from increment import Analysis
from increment.estimation.engine import escalate_family_member
from increment.frame import MetricSpec
from increment.semantics.models import AnalysisPlan
from tests.analysis_factory import lift_rows

UNITS_PER_ARM = 50_000
ARMS = ("control", "T1", "T2")
#: Ten conversion secondaries at a 5% rate; the treatment arms differ from control by a few
#: conversions, so nothing is selected.
NULL_METRICS = tuple(f"m{i}" for i in range(10))


def _frame(successes: dict[str, dict[str, int]]) -> pl.DataFrame:
    """One row per unit and arm, a 0/1 column per metric with the given successes per arm."""
    rng = np.random.default_rng(20261014)
    frames = []
    for arm in ARMS:
        columns: dict[str, object] = {
            "user_id": [f"{arm}-{i}" for i in range(UNITS_PER_ARM)],
            "variant": [arm] * UNITS_PER_ARM,
        }
        for metric, by_arm in successes.items():
            y = np.r_[np.ones(by_arm[arm]), np.zeros(UNITS_PER_ARM - by_arm[arm])]
            rng.shuffle(y)
            columns[metric] = y
        frames.append(pl.DataFrame(columns))
    return pl.concat(frames)


def _null_successes(metrics: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        metric: {"control": 2_500, "T1": 2_500 + 3 * i, "T2": 2_500 - 2 * i}
        for i, metric in enumerate(metrics)
    }


@pytest.fixture(scope="module")
def null_frame() -> pl.DataFrame:
    return _frame(_null_successes((*NULL_METRICS, "m10")))


def _analysis(frame: pl.DataFrame, metrics: tuple[str, ...], plan: AnalysisPlan | None) -> Analysis:
    return Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=metric, type="conversion") for metric in metrics],
        plan=plan,
    )


def _by_cell(results):
    return {(row.metric, row.group_id): row for row in lift_rows(results)}


class TestTheAuditScenario:
    """50,000 units per arm, 5% conversion, three arms, ten conversion secondaries under one BH
    family: every member keeps the delta route and the interval the same data gives without a
    plan, and the readout runs without a single exact inversion."""

    def test_family_members_keep_their_no_plan_intervals(self, null_frame):
        with_plan = _analysis(null_frame, NULL_METRICS, AnalysisPlan(secondaries=NULL_METRICS))
        without = _analysis(null_frame, NULL_METRICS, None)
        family = list(lift_rows(with_plan.run()))
        plain = _by_cell(without.run())
        assert len(family) == 20
        for row in family:
            twin = plain[row.metric, row.group_id]
            assert row.reference_kind == twin.reference_kind == "t"
            assert row.delta_evidence is not None and twin.delta_evidence is not None
            # The family reads the member's composite; the row outside the plan keeps its delta
            # p-value. Under the null the two coincide (no member lies below its dense level).
            assert row.delta_evidence.composite and not twin.delta_evidence.composite
            assert row.delta_evidence.counts == twin.delta_evidence.counts
            assert row.delta_evidence.dense_level == twin.delta_evidence.dense_level
            assert row.require_lift().lb == twin.require_lift().lb
            assert row.require_lift().ub == twin.require_lift().ub
            assert row.p_value() == twin.p_value()
            assert row.discovery is False and row.family_size == 20

    def test_an_eleventh_metric_leaves_the_other_rows_intervals_and_p_values_unchanged(
        self, null_frame
    ):
        ten = _by_cell(
            _analysis(null_frame, NULL_METRICS, AnalysisPlan(secondaries=NULL_METRICS)).run()
        )
        eleven = _by_cell(
            _analysis(
                null_frame,
                (*NULL_METRICS, "m10"),
                AnalysisPlan(secondaries=(*NULL_METRICS, "m10")),
            ).run()
        )
        assert len(eleven) == 22
        for key, row in ten.items():
            grown = eleven[key]
            assert grown.family_size == 22 and row.family_size == 20
            assert grown.require_lift().lb == row.require_lift().lb
            assert grown.require_lift().ub == row.require_lift().ub
            assert grown.p_value() == row.p_value()
            assert grown.delta_evidence == row.delta_evidence


class TestASelectedMemberIsReissuedByItsCounts:
    def test_discovery_agrees_with_the_corrected_interval_below_the_dense_level(self):
        """Two strong members among ten are selected at a corrected level below their dense
        level: each is reissued as the exact binomial set from its counts and that set excludes
        the null; every unselected member keeps the delta interval the data gives without a
        plan, and the p-value the family selected on."""
        metrics = (*NULL_METRICS, "strong_a", "strong_b")
        successes = _null_successes(NULL_METRICS)
        successes["strong_a"] = {"control": 2_500, "T1": 2_800, "T2": 2_480}
        successes["strong_b"] = {"control": 2_500, "T1": 2_780, "T2": 2_510}
        frame = _frame(successes)
        family = list(lift_rows(_analysis(frame, metrics, AnalysisPlan(secondaries=metrics)).run()))
        plain = _by_cell(_analysis(frame, metrics, None).run())
        selected = [row for row in family if row.discovery]
        assert {(row.metric, row.group_id) for row in selected} == {
            ("strong_a", "T1"),
            ("strong_b", "T1"),
        }
        for row in selected:
            nominal = plain[row.metric, row.group_id]
            assert nominal.delta_evidence is not None
            assert row.family_threshold is not None
            assert row.family_threshold < nominal.delta_evidence.dense_level
            assert row.reference_kind == "binomial" and row.binomial_set is not None
            reissue_alpha = row.family_threshold / 2
            reissued = row.reintervalize(reissue_alpha)
            assert reissued.binomial_set is not None
            assert reissued.reference_kind == "binomial"
            direct = row.reissue(reissue_alpha)
            assert direct == reissued
            assert row.stat_sig() is True
            p_value = row.p_value()
            assert p_value is not None and p_value < row.family_threshold
        for row in family:
            if row.discovery:
                continue
            nominal = plain[row.metric, row.group_id]
            assert row.reference_kind == "t"
            assert row.require_lift().lb == nominal.require_lift().lb
            assert row.require_lift().ub == nominal.require_lift().ub
            assert row.p_value() == escalate_family_member(nominal).p_value()


def test_fcr_reissues_only_selected_arms_without_invalidating_nominal_siblings():
    from increment import moments_from_summaries

    n = 2_000_000_000
    summaries = [
        {"metric": "converted", "group_id": group, "n": n, "successes": successes}
        for group, successes in (("control", 50_000), ("T1", 60_000), ("T2", 2_500))
    ]
    plan = AnalysisPlan(secondaries=["converted"], alpha=0.05, q=0.001, alternative="greater")
    cube = moments_from_summaries(summaries, metrics={"converted": "conversion"}, control="control")
    analysis = Analysis.from_moments(
        cube, metrics={"converted": "conversion"}, control="control", plan=plan
    )
    readout = analysis.run()
    rows = {row.group_id: row for row in lift_rows(readout)}
    assert set(rows) == {"T1", "T2"}
    assert rows["T1"].discovery is True and rows["T1"].stat_sig() is True
    assert rows["T2"].discovery is False
    assert rows["T2"].reference_kind == "t"
    assert rows["T2"].failure_code is None and rows["T2"].sampling_available is True
    assert rows["T2"].p_value() == pytest.approx(1.0)
    assert readout.metadata is not None
    assert readout.metadata.partial is False
