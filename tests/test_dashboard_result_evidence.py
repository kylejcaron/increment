"""Dashboard result evidence: per-row outcomes, complete confidence sets, levels, warnings.

Rows come from real estimates and the real confidence-set constructors, so every assertion
reads emitted numbers, geometry and outcome classes rather than prose.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import (
    DashboardSnapshot,
    render_details,
    render_metric_details,
    render_results,
)
from increment.dashboard._format import (
    complete_confidence_set_text,
    confidence_set_text,
    headline_interval,
    outcome_label,
)
from increment.dashboard._html import results_notes
from increment.estimation.inference import _joint_additive_bounds
from increment.estimation.results import (
    Estimate,
    JointContrastReference,
    relative_confidence_set,
)
from increment.tables import estimates_to_readout
from tests import test_dashboard as dashboard_tests
from tests.dashboard_snapshots import binomial_region

storefront = dashboard_tests.storefront


PRIMARY, SECONDARY, GUARDRAIL = "checkout_conversion", "revenue_per_user", "session_seconds"


def _rebuilt(snapshot: DashboardSnapshot, estimates: tuple[Any, ...]) -> DashboardSnapshot:
    rows = estimates_to_readout(list(estimates))
    return dataclasses.replace(
        snapshot,
        estimates=estimates,
        readout_rows=tuple(
            {**row, "alternative": est.alternative}
            for row, est in zip(rows, estimates, strict=True)
        ),
    )


def _decision(snapshot: DashboardSnapshot, metric: str) -> Any:
    return next(e for e in snapshot.estimates if e.metric == metric and e.method_role == "decision")


def _with(snapshot: DashboardSnapshot, metric: str, **updates: Any) -> DashboardSnapshot:
    """The snapshot with one decision estimate changed."""
    original = _decision(snapshot, metric)
    estimates = tuple(
        original.model_copy(update=updates) if e is original else e for e in snapshot.estimates
    )
    return _rebuilt(snapshot, estimates)


def _row(snapshot: DashboardSnapshot, metric: str, method_role: str = "decision") -> Any:
    return next(
        r
        for r in snapshot.readout_rows
        if r["metric"] == metric and r["method_role"] == method_role
    )


# A row whose engine produced no point, no interval and no set.
_UNAVAILABLE = {"lift": None, "confidence_set": None, "binomial_set": None}


def _fieller(a: float, c: float, *, alternative: str = "two-sided", cov: float = 0.0):
    reference = JointContrastReference(a=a, c=c, var_a=1, var_c=1, cov_ac=cov)
    return relative_confidence_set(reference, alternative=alternative)


@pytest.mark.integration
def test_binomial_qualification_is_in_the_metric_inspector_interval_fold(storefront):
    from increment.dashboard._html import render_metric_details

    storefront = dataclasses.replace(
        storefront,
        readout_rows=tuple(
            {**row, "binomial_set": binomial_region()} if row["metric"] == PRIMARY else row
            for row in storefront.readout_rows
        ),
    )

    row = _row(storefront, "checkout_conversion")
    html = render_metric_details(storefront, metric=row["metric"]).text
    assert "How to read these intervals" in html
    assert "conditional on deployed SciPy/Boost" in html


# Complete set text


@pytest.mark.parametrize("metric", [PRIMARY, GUARDRAIL])
@pytest.mark.slow
def test_point_backed_one_sided_fieller_set_keeps_its_direction_and_own_coverage(
    storefront: DashboardSnapshot, metric: str
) -> None:
    region = relative_confidence_set(
        JointContrastReference(a=2, c=3.0, var_a=0.04, var_c=0.09, cov_ac=0.01),
        alternative="greater",
    )
    bounds = _joint_additive_bounds(2.0, 0.2, 0.05, "greater", None)
    snapshot = _with(
        storefront,
        metric,
        alternative="greater",
        lift=region.estimate(),
        relative_confidence_set=region,
        abs_diff=2.0,
        abs_se=0.2,
        abs_lb=bounds[0],
        abs_ub=bounds[1],
        abs_reference_kind="normal",
        binomial_set=None,
        confidence_set=None,
    )
    row = _row(snapshot, metric)
    assert region.geometry == "one_sided"
    central_lower, central_upper = row["lower"], row["higher"]
    assert row["level"] == pytest.approx(0.9)

    from increment.dashboard._app import _primary
    from increment.dashboard._format import interval_html, primary_method

    assert confidence_set_text(row) is None
    headline = headline_interval(row)
    assert f"{central_lower:+.1%}" in headline
    if metric == GUARDRAIL:
        assert headline.endswith(", ∞)")
        assert f"{central_upper:+.1%}" not in headline
        assert interval_html(row) == headline
        assert primary_method(row).startswith("95% ")
        primary = _primary(dataclasses.replace(snapshot, primary_metric=metric))
        assert primary["interval"] == headline
        assert primary["levelLabel"] == "95%"
        assert row["level"] == pytest.approx(0.9)
    else:
        assert f"{central_upper:+.1%}" in headline

    # The full set is one-sided (open above) at the set's own 95% coverage.
    complete = complete_confidence_set_text(row)
    assert complete is not None
    ((lower, upper),) = region.intervals
    assert upper is None
    assert f"[{lower:+.1%}, ∞)" in complete
    stated = [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)%", complete)]
    assert any(abs(value - (1 - region.alpha)) < 5e-5 for value in stated)
    assert f"{central_upper:+.1%}" not in complete


@pytest.mark.parametrize(
    "construction",
    [{"inference": "asymptotic_mean"}, {"reference_kind": "unsupported"}],
)
def test_guardrail_interval_does_not_relabel_an_unrecognized_construction(construction):
    from increment.dashboard._format import primary_method

    row = {
        "role": "guardrail",
        "alternative": "greater",
        "inference": "fixed",
        "value_scale": "relative",
        "lower": 0.02,
        "higher": 0.22,
        "level": 0.9,
        "coverage_alpha": 0.1,
        **construction,
    }

    assert headline_interval(row) == "[+2.0%, +22.0%]"
    assert primary_method(row).startswith("90% ")


@pytest.mark.parametrize(
    ("a", "c", "cov", "geometry"),
    [(10, 0, 0.0, "disconnected"), (1, 0, 0.0, "all_real"), (1, 1, -0.9, "disconnected")],
)
@pytest.mark.slow
def test_complete_set_text_keeps_every_endpoint_of_point_free_sets(
    storefront: DashboardSnapshot, a: float, c: float, cov: float, geometry: str
) -> None:
    region = _fieller(a, c, cov=cov)
    assert region.geometry == geometry
    row = _row(_with(storefront, PRIMARY, lift=None, relative_confidence_set=region), PRIMARY)
    complete = complete_confidence_set_text(row)
    assert complete is not None
    for interval in region.intervals:
        for endpoint in interval:
            assert endpoint is None or f"{endpoint:+.1%}" in complete
    assert complete.count("∞") == sum(
        endpoint is None for interval in region.intervals for endpoint in interval
    )
    assert headline_interval({**row, "role": "guardrail", "alternative": "greater"}) == complete


# Per-row outcomes


@pytest.mark.parametrize("stat_sig", [True, False, None])
@pytest.mark.parametrize("discovery", [True, False, None])
@pytest.mark.integration
def test_outcome_label_formats_captured_verdict_and_selection(
    storefront: DashboardSnapshot, stat_sig: bool | None, discovery: bool | None
) -> None:
    row = {
        **_row(storefront, SECONDARY),
        "stat_sig": stat_sig,
        "discovery": discovery,
    }
    label = outcome_label(row)
    assert label
    assert outcome_label({**row, "stat_sig": not stat_sig}) != label
    assert outcome_label({**row, "discovery": not discovery}) != label


@pytest.mark.integration
def test_valid_sets_without_a_bounded_interval_keep_captured_verdict(
    storefront: DashboardSnapshot,
) -> None:
    region = _fieller(1, 0)
    snapshot = _with(storefront, PRIMARY, lift=region.estimate(), relative_confidence_set=region)
    row = _row(snapshot, PRIMARY)
    assert outcome_label(row) == outcome_label({**row, "relative_confidence_set": None})


@pytest.mark.integration
def test_missing_verdict_is_not_converted_to_non_rejection(
    storefront: DashboardSnapshot,
) -> None:
    row = _row(storefront, PRIMARY)
    assert outcome_label({key: value for key, value in row.items() if key != "stat_sig"}) != (
        outcome_label({**row, "stat_sig": False})
    )


@pytest.mark.integration
def test_readout_qualifications_distinguish_multiple_methods_for_one_metric(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import LiftEstimates
    from increment.dashboard._data import readout_interpretation_text

    decision = _decision(storefront, PRIMARY)
    sensitivity = decision.model_copy(update={"method": "cuped", "method_role": "sensitivity"})
    snapshot = _rebuilt(storefront, (*storefront.estimates, sensitivity))
    collection = LiftEstimates(list(snapshot.estimates))

    qualifications = readout_interpretation_text(snapshot, collection=collection)

    assert "method unadjusted" in qualifications
    assert "method role decision" in qualifications
    assert "method cuped" in qualifications
    assert "method role sensitivity" in qualifications


@pytest.mark.integration
def test_metric_details_restore_metric_descriptions(storefront: DashboardSnapshot) -> None:
    declared = render_metric_details(storefront, metric=PRIMARY).text

    assert "Did the user check out within 14 days of exposure?" in declared


@pytest.mark.integration
def test_outcome_label_does_not_add_row_identity_to_layout_text(
    storefront: DashboardSnapshot,
) -> None:
    row = {**_row(storefront, PRIMARY), "method": "cuped", "method_role": "sensitivity"}
    label = outcome_label(row)
    assert label
    assert outcome_label({**row, "group_id": "different-arm"}) == label


# Levels


@pytest.mark.integration
def test_interval_levels_keep_two_decimals_in_every_disclosure(
    storefront: DashboardSnapshot,
) -> None:
    level = 1 - 0.05 / 3
    snapshot = _with(
        storefront,
        PRIMARY,
        lift=Estimate(value=0.2, lb=-0.1, ub=0.5, level=level),
    )
    surfaces = (
        results_notes(snapshot),
        render_details(snapshot).text,
        render_metric_details(snapshot, metric=PRIMARY).text,
    )
    for text in surfaces:
        stated = [float(p) / 100 for p in re.findall(r"(\d+(?:\.\d+)?)%", text)]
        assert any(abs(value - level) < 5e-5 for value in stated)
        assert "98.3%" not in text


# Adverse warnings


@pytest.mark.parametrize(
    ("lower", "higher", "adverse"),
    [(-0.03, -0.01, False), (-0.06, -0.03, True)],
)
def test_guardrail_warning_requires_evidence_beyond_safety_margin(lower, higher, adverse):
    from increment.dashboard._html import _is_adverse

    row = {
        "role": "guardrail",
        "alternative": "greater",
        "null_lift": -0.02,
        "lower": lower,
        "higher": higher,
    }
    assert _is_adverse(row) is adverse


@pytest.mark.integration
@pytest.mark.parametrize(("mean_shift", "verdict"), [(1.0, "reject"), (0.05, "not_reject")])
def test_additive_zero_decisions_reach_the_dashboard_outcome_and_null(
    mean_shift, verdict, storefront
):
    """A nonpositive-mean cell decided on its additive interval is a real test: its
    outcome is its verdict, never 'unavailable', and the tested null is the additive zero."""
    from increment.dashboard._format import null_text, primary_tone, result_outcome
    from tests.test_tables import _nonpositive_cuped_breakout_segment

    cell = _nonpositive_cuped_breakout_segment(mean_shift=mean_shift)
    (row,) = estimates_to_readout([cell])
    outcome = result_outcome(row)
    assert outcome.verdict == verdict
    assert outcome.unavailable_reason is None
    assert null_text(row) == "+0 absolute"
    assert primary_tone(row) in ("neutral", "inconclusive")
    if mean_shift == 1.0:
        guardrail = cell.model_copy(
            update={"metric": GUARDRAIL, "role": "guardrail", "alternative": "greater"}
        )
        snapshot = _rebuilt(
            storefront,
            tuple(
                guardrail if estimate.metric == GUARDRAIL else estimate
                for estimate in storefront.estimates
            ),
        )
        rendered = render_results(snapshot).text
        caption = re.search(r'<small class="inc-dashboard-guardrail">(.*?)</small>', rendered, re.S)
        assert caption is not None
        assert re.search(r"\b0\b.*absolute", re.sub(r"<[^>]+>", "", caption.group(1)))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("alternative", "relative_margin", "absolute_margin", "expected"),
    [
        ("greater", -0.02, None, "Guardrail: no more than a 2% relative decrease"),
        ("less", 0.05, None, "Guardrail: no more than a 5% relative increase"),
        ("less", 0.0, 5.0, "Guardrail: no more than 5 absolute units of increase"),
        ("greater", 0.03, None, "Guardrail: at least a 3% relative increase"),
    ],
)
def test_guardrail_metric_label_states_declared_direction_and_scale(
    storefront, alternative, relative_margin, absolute_margin, expected
):
    rows = tuple(
        {
            **row,
            "alternative": alternative,
            "null_lift": relative_margin,
            "null_abs": absolute_margin,
            "decision_scale": "relative" if absolute_margin is None else "absolute",
        }
        if row["metric"] == GUARDRAIL
        else row
        for row in storefront.readout_rows
    )
    html = render_results(dataclasses.replace(storefront, readout_rows=rows)).text
    assert expected in re.sub(r"<[^>]+>", "", html)


@pytest.mark.parametrize("state", ("stat_sig", "decision_available"))
@pytest.mark.integration
def test_guardrail_disclosures_distinguish_captured_verdict_and_availability(storefront, state):
    def readings(reject):
        rows = tuple(
            {
                **row,
                "stat_sig": False,
                "sampling_available": True,
                "decision_available": True,
                state: reject,
                "sequential_status": "unavailable"
                if state == "decision_available" and not reject
                else "interval",
                "inference": "asymptotic_mean"
                if state == "decision_available"
                else row["inference"],
                "alternative": "less",
                "null_lift": 0.05,
                "lift": 0.02,
                "lower": 0.01,
                "higher": 0.03,
            }
            if row["metric"] == GUARDRAIL
            else row
            for row in storefront.readout_rows
        )
        snapshot = dataclasses.replace(storefront, readout_rows=rows)
        readout_row = next(
            body
            for body in re.findall(r"<tr\b[^>]*>(.*?)</tr>", results_notes(snapshot), re.S)
            if GUARDRAIL in body
        )
        detail = render_metric_details(snapshot, metric=GUARDRAIL).text
        outcome = re.search(r"<dt>Test outcome</dt><dd>(.*?)</dd>", detail, re.S)
        assert outcome is not None
        return readout_row, outcome.group(1)

    rejected = readings(True)
    not_rejected = readings(False)
    assert all(a != b for a, b in zip(rejected, not_rejected, strict=True))


@pytest.mark.integration
def test_early_trajectory_caution_keeps_its_date_and_arm_without_flagging_clean_results(
    storefront: DashboardSnapshot,
) -> None:
    from functools import partial
    from html.parser import HTMLParser

    from increment._source_operations import DashboardExploreCapture
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.dashboard import render_header, render_health
    from increment.dashboard._app import build_payload
    from increment.declarations import CellKey
    from increment.estimation.advisory import ReadoutAdvisory
    from increment.estimation.readout_types import CellRecord, cell_order
    from tests.dashboard_snapshots import experiment_for, temporal_rows

    class Rows(HTMLParser):
        def __init__(self, html):
            super().__init__()
            self.rows: list[str] = []
            self.depth = 0
            self.text: list[str] = []
            self.feed(html)

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.depth += 1

        def handle_endtag(self, tag):
            if tag == "tr":
                self.depth -= 1
                if self.depth == 0:
                    self.rows.append(" ".join(self.text))
                    self.text = []

        def handle_data(self, data):
            if self.depth:
                self.text.append(data.strip())

    advisory = ReadoutAdvisory(
        code="estimation.engine.ratio_denominator_skew",
        context={
            "metric_name": SECONDARY,
            "arms": {"control": {"n": 137, "skewness": 8.25, "statistic": 0.75}},
            "threshold": 0.3,
            "unavailable_arms": {"treatment": "denominator third moment unavailable"},
        },
        message="A denominator-shape heuristic warrants caution.",
    )
    collection = storefront.readout_collection
    assert collection is not None and collection.metadata is not None
    base_metadata = collection.metadata

    def cautioned_points(*, segmented: bool) -> DailyLiftEstimates:
        rows = temporal_rows(storefront, "cumulative_lift", segmented=segmented)
        early = next(
            point
            for point in rows
            if point.metric == SECONDARY
            and point.dimension_value == ("US" if segmented else None)
            and str(point.ds) == "2025-01-15"
        )
        keys = tuple(sorted((CellKey.from_row(point) for point in rows), key=cell_order))
        source = base_metadata.scope.by_source["presentation"].model_copy(
            update={"cells": keys, "decision_cells": keys}
        )
        scope = base_metadata.scope.model_copy(
            update={"cells": keys, "decision_cells": keys, "by_source": {"presentation": source}}
        )
        metadata = base_metadata.model_copy(
            update={
                "scope": scope,
                "cells": tuple(
                    CellRecord(
                        cell=key,
                        source_snapshot_id="presentation",
                        advisories=(advisory,) if key == CellKey.from_row(early) else (),
                    )
                    for key in keys
                ),
            }
        )
        return DailyLiftEstimates(rows, metadata=metadata)

    points = cautioned_points(segmented=True)
    captures = dict(storefront.explore)
    for breakout, series in (
        (None, cautioned_points(segmented=False)),
        (("event_log", "country"), points),
    ):
        for metric in (None, *(model.name for model in storefront.metrics)):
            selected = series.filter(
                lambda point, metric=metric: metric is None or point.metric == metric
            )
            for complete in (False, True):
                key = ("assigned", "cumulative_lift", metric, complete, breakout)
                captures[key] = DashboardExploreCapture.answered(
                    key,
                    selected,
                    collection=partial(DailyLiftEstimates, metadata=selected.metadata),
                )
    snapshot = dataclasses.replace(storefront, explore=captures)
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    packet = payload["explore"]["country"][SECONDARY]["cumulative_lift"]
    assert packet["pointCount"] == 4
    rendered = packet["html"]
    parsed = Rows(rendered)
    affected = next(row for row in parsed.rows if "US" in row)
    clean = next(row for row in parsed.rows if "CA" in row)
    assert "Uncertainty caution" in affected and "some dates" in affected.lower()
    assert "Uncertainty caution" not in clean
    for value in (
        "Diagnostics",
        SECONDARY,
        "2025-01-15",
        "country",
        "US",
        "assigned",
        "control",
        "137",
        "8.25",
        "0.75",
        "0.3",
        "Sample size",
        "Statistic",
        "Threshold",
        "treatment",
        "denominator third moment unavailable",
    ):
        assert value in affected
    assert "Uncertainty caution" not in render_header(snapshot).text
    assert "Uncertainty caution" not in render_results(snapshot).text
    disclosures = re.findall(r"<details\b([^>]*)>\s*<summary>Diagnostics[^<]*</summary>", rendered)
    assert disclosures and all("open" not in attributes for attributes in disclosures)
    whole_packet = payload["wholeExperiment"]["cumulative_lift"]
    assert whole_packet["pointCount"] == 6
    whole_rows = Rows(whole_packet["html"])
    whole_affected = next(row for row in whole_rows.rows if SECONDARY in row)
    assert "Uncertainty caution" in whole_affected and "some dates" in whole_affected.lower()
    assert "2025-01-15" in whole_affected and "137" in whole_affected
    assert all("Uncertainty caution" not in row for row in whole_rows.rows if row != whole_affected)
    health = render_health(snapshot).text
    assert "<summary>Diagnostics" in health
    assert SECONDARY in health and "2025-01-15" in health and "137" in health
    central = Rows(health)
    assert len([row for row in central.rows if SECONDARY in row and "137" in row]) == 2
    clean_points = points.filter(lambda point: point.dimension_value == "CA")
    key = ("assigned", "cumulative_lift", None, False, ("event_log", "country"))
    clean_capture = DashboardExploreCapture.answered(
        key, clean_points, collection=partial(type(points), metadata=clean_points.metadata)
    )
    clean_health = render_health(dataclasses.replace(snapshot, explore={key: clean_capture})).text
    assert "<summary>Diagnostics" not in clean_health


@pytest.mark.integration
def test_computed_ratio_diagnostics_keep_segment_labels_literal_in_native_tables(
    storefront: DashboardSnapshot,
) -> None:
    from collections.abc import Mapping
    from datetime import date
    from html.parser import HTMLParser

    from increment.breakout.estimates import (
        BreakoutEstimate,
        BreakoutEstimates,
        DailyLiftEstimate,
        DailyLiftEstimates,
    )
    from increment.dashboard import render_explore
    from increment.declarations import CellKey
    from increment.estimation.engine import estimate_lift
    from increment.estimation.readout_types import CellRecord, cell_order
    from increment.semantics.models import Measure, RatioMetric
    from tests.ratio_arms import moment_arm, summary

    class Diagnostics(HTMLParser):
        def __init__(self, markup):
            super().__init__()
            self.targets: list[str] = []
            self.text: list[str] = []
            self.depth = 0
            self.region = 0
            self.feed(markup)

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "a":
                self.targets.append(attributes.get("href") or "")
            if tag == "div":
                self.depth += 1
                if "inc-dashboard-diagnostic-records" in attributes.get("class", "").split():
                    self.region = self.depth

        def handle_endtag(self, tag):
            if tag == "div":
                if self.depth == self.region:
                    self.region = 0
                self.depth -= 1

        def handle_data(self, data):
            if self.region:
                self.text.append(data)

    label = (
        "{{[Open details](javascript:alert%281%29)}}\n\n[More details](javascript:alert%282%29)\n\n"
    )
    metric = RatioMetric(
        name=SECONDARY,
        entity="user_id",
        numerator=Measure(fact="revenue"),
        denominator=Measure(fact="sessions"),
    )
    arms = tuple(
        moment_arm(arm, n=50, y_bar=mean, var_y=1.0, d_bar=1.0, var_d=4.5, metric=SECONDARY)
        for arm, mean in (("control", 2.0), ("treatment", 2.2))
    )
    computation = estimate_lift(metrics=[metric], summary=summary(*arms), control_group="control")
    assert not computation.failures
    (estimate,) = computation.results
    (advisory,) = (
        advisory
        for advisories in computation.advisories.values()
        for advisory in advisories
        if advisory.code == "estimation.engine.ratio_denominator_precision"
    )
    relative_errors = advisory.context["relative_standard_errors"]
    assert isinstance(relative_errors, Mapping)
    assert relative_errors == {
        "control": pytest.approx(0.3),
        "treatment": pytest.approx(0.3),
    }
    assert advisory.context["threshold"] == pytest.approx(0.15)
    snapshot = dataclasses.replace(
        storefront,
        metrics=tuple(metric if model.name == SECONDARY else model for model in storefront.metrics),
    )
    base: dict[str, Any] = {
        **estimate.model_dump(exclude={"ds"}),
        "source_snapshot_id": "presentation",
        "dimension": "country",
        "dimension_value": label,
        "source": "event_log",
    }
    collection = storefront.readout_collection
    assert collection is not None and collection.metadata is not None
    base_metadata = collection.metadata
    for view, collection_type, rows in (
        ("segments", BreakoutEstimates, (BreakoutEstimate(**base),)),
        (
            "cumulative_lift",
            DailyLiftEstimates,
            tuple(DailyLiftEstimate(**base, ds=date(2025, 1, day)) for day in (15, 16)),
        ),
    ):
        keys = tuple(sorted((CellKey.from_row(row) for row in rows), key=cell_order))
        source = base_metadata.scope.by_source["presentation"].model_copy(
            update={"cells": keys, "decision_cells": keys, "decision_complete_by_population": {}}
        )
        metadata = base_metadata.model_copy(
            update={
                "scope": base_metadata.scope.model_copy(
                    update={
                        "cells": keys,
                        "decision_cells": keys,
                        "by_source": {"presentation": source},
                    }
                ),
                "cells": tuple(
                    CellRecord(cell=key, source_snapshot_id="presentation", advisories=(advisory,))
                    for key in keys
                ),
            }
        )
        data = collection_type(rows, metadata=metadata)
        raw_rows = tuple(row.model_dump() for row in data)
        markup = render_explore(snapshot, data, metric=SECONDARY, view=view).text
        parsed = Diagnostics(markup)
        targets = [re.sub(r"[\x00-\x20\x7f]", "", target).lower() for target in parsed.targets]
        assert not any(
            target.startswith(("javascript:", "data:", "vbscript:")) for target in targets
        ), (view, targets)
        diagnostic_text = "".join(parsed.text)
        for value in (label, SECONDARY, "country", "control", "treatment", "50", "0.3", "0.15"):
            assert value in diagnostic_text, (view, value)
        assert "Sample size" in diagnostic_text and "Threshold" in diagnostic_text
        assert "<svg" in markup
        disclosures = re.findall(
            r"<details\b([^>]*)>\s*<summary>Diagnostics[^<]*</summary>", markup
        )
        assert disclosures and all("open" not in attributes for attributes in disclosures)
        if view == "cumulative_lift":
            assert "2025-01-15" in diagnostic_text and "2025-01-16" in diagnostic_text
        assert tuple(row.model_dump() for row in data) == raw_rows
        assert all(row.dimension_value == label for row in data)


@pytest.mark.integration
@pytest.mark.parametrize(
    "surface",
    (
        "results",
        "overview",
        "segments",
        "segment_overview",
        "inspector",
        "policies",
        "inspector_sensitivity",
        "policies_sensitivity",
    ),
)
def test_precision_caution_replaces_only_its_duplicate_note_in_native_result_tables(
    storefront: DashboardSnapshot, surface: str
) -> None:
    from functools import partial

    from increment._source_operations import DashboardExploreCapture
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.dashboard import render_explore, render_header
    from increment.dashboard._data import DashboardOverview, enriched_rows
    from increment.dashboard._html import overview_table
    from increment.declarations import CellKey
    from increment.estimation.advisory import ReadoutAdvisory
    from increment.estimation.readout_types import CellRecord

    advisory = ReadoutAdvisory(
        code="estimation.engine.ratio_denominator_precision",
        context={
            "metric_name": SECONDARY,
            "threshold": 0.1,
            "relative_standard_errors": {"control": 0.625, "treatment": None},
            "relative_standard_error_unavailable_reasons": {"treatment": "past_float_range"},
        },
        message="Recorded denominator precision caution.",
    )
    unrelated = "A separate measurement qualification remains."
    snapshot = _with(
        storefront,
        SECONDARY,
        note=f"{advisory.message}; {unrelated}",
        n_control=211,
        n_treat=307,
    )
    sensitivity_only = surface.endswith("_sensitivity")
    if sensitivity_only:
        decision = _decision(snapshot, SECONDARY)
        sensitivity = decision.model_copy(update={"method": "cuped", "method_role": "sensitivity"})
        snapshot = _rebuilt(
            snapshot,
            (
                *(
                    estimate.model_copy(update={"note": None}) if estimate is decision else estimate
                    for estimate in snapshot.estimates
                ),
                sensitivity,
            ),
        )
    collection = storefront.readout_collection
    assert collection is not None and collection.metadata is not None
    rows = snapshot.estimates
    segmented = surface in ("segments", "segment_overview")
    if segmented:
        rows = tuple(
            BreakoutEstimate(
                **_decision(snapshot, SECONDARY).model_dump(),
                dimension="country",
                dimension_value=segment,
                source="event_log",
            ).model_copy(
                update={"note": f"{advisory.message}; {unrelated}" if segment == "US" else None}
            )
            for segment in ("US", "CA")
        )
    keys = tuple(CellKey.from_row(row) for row in rows)
    source = collection.metadata.scope.by_source["presentation"].model_copy(
        update={"cells": keys, "decision_cells": keys}
    )
    metadata = collection.metadata.model_copy(
        update={
            "scope": collection.metadata.scope.model_copy(
                update={
                    "cells": keys,
                    "decision_cells": keys,
                    "by_source": {"presentation": source},
                }
            ),
            "cells": tuple(
                CellRecord(
                    cell=key,
                    source_snapshot_id="presentation",
                    advisories=(advisory,)
                    if key.metric == SECONDARY
                    and key.dimension_value in (None, "US")
                    and key.method_role == ("sensitivity" if sensitivity_only else "decision")
                    else (),
                )
                for key in keys
            ),
        }
    )
    if segmented:
        collection = BreakoutEstimates(rows, metadata=metadata)
        snapshot = _with(snapshot, SECONDARY, note=None)
        key = ("assigned", "overview_segments", None, False, ("event_log", "country"))
        snapshot = dataclasses.replace(
            snapshot,
            overview=DashboardOverview(
                exploratory=(),
                segments={("event_log", "country"): tuple(collection)},
                family_size=2,
                family_q=0.05,
            ),
            explore={
                key: DashboardExploreCapture.answered(
                    key, collection, collection=partial(type(collection), metadata=metadata)
                )
            },
        )
        rendered = (
            overview_table(snapshot, ("event_log", "country"))
            if surface == "segment_overview"
            else render_explore(snapshot, collection, metric=SECONDARY, view="segments").text
        )
    else:
        collection = type(collection)(rows, metadata=metadata)
        snapshot = dataclasses.replace(
            snapshot, readout_collection=collection, readout_rows=enriched_rows(collection)
        )
        if surface.startswith("inspector"):
            rendered = render_metric_details(snapshot, metric=SECONDARY).text
        elif surface.startswith("policies"):
            rendered = render_details(snapshot).text
        else:
            rendered = (
                render_results(snapshot).text
                if surface == "results"
                else overview_table(snapshot, None)
            )
    assert rendered.count("Uncertainty caution") == 1
    assert advisory.message not in rendered
    assert unrelated in rendered
    for value in (
        "Diagnostics",
        SECONDARY,
        "control",
        "211",
        "307",
        "0.625",
        "0.1",
        "past_float_range",
    ):
        assert value in rendered
    if sensitivity_only:
        assert "cuped" in rendered and "sensitivity" in rendered
    assert "not a passing check" in rendered
    assert "Uncertainty caution" not in render_header(snapshot).text
