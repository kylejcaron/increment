"""Explore overview presentation from constructed snapshots."""

from __future__ import annotations

import dataclasses
import re

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.breakout.estimates import BreakoutEstimates
from increment.dashboard import render_explore
from increment.dashboard._data import decision_rows
from increment.dashboard._html import overview_rows, overview_table
from tests.dashboard_snapshots import ADDED, COUNTRY, experiment_for, overview_snapshot

pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")


@pytest.fixture(scope="module")
def _saved_added_snapshot():
    return overview_snapshot()


@pytest.fixture(scope="module")
def _family_presentation_snapshot(_saved_added_snapshot):
    return _saved_added_snapshot


def _identity(row) -> tuple:
    return (row["metric"], row["group_id"], row["lift"], row["lower"], row["higher"])


@pytest.mark.integration
@pytest.mark.slow
def test_declared_whole_experiment_rows_are_the_readout_rows(_saved_added_snapshot):
    snapshot = _saved_added_snapshot
    readout = {_identity(row): row["stat_sig"] for row in decision_rows(snapshot)}
    for scope in (None, COUNTRY):
        shown = {
            _identity(row): row["stat_sig"]
            for row in overview_rows(snapshot, scope)
            if row["metric"] != ADDED and row.get("dimension") is None
        }
        assert shown == readout
    assert ADDED not in {row["metric"] for row in snapshot.readout_rows}


def test_explore_tables_keep_core_columns_without_a_result_column(_family_presentation_snapshot):

    snapshot = _family_presentation_snapshot
    overview_html = overview_table(snapshot, COUNTRY)
    headers = {
        re.sub(r"<[^>]+>", "", header).strip()
        for header in re.findall(
            r'<th\b[^>]*class="[^"]*\bgt_col_heading\b[^"]*"[^>]*>(.*?)</th>',
            overview_html,
            flags=re.DOTALL,
        )
    }
    assert {"Result", "Discovery", "Interval"}.isdisjoint(headers)
    assert "Lift Plot" in headers


def test_overview_qualification_footnotes_name_rows_and_family(_family_presentation_snapshot):
    from increment.dashboard._html import overview_table

    snapshot = _family_presentation_snapshot
    rows = overview_rows(snapshot, COUNTRY)
    qualified = [row for row in rows if row.get("selection_qualification")]
    assert qualified

    html = overview_table(snapshot, COUNTRY)
    footnote = re.search(
        r'<ul class="inc-dashboard-family-qualifications">(.*?)</ul>', html, re.DOTALL
    )
    assert footnote is not None
    body = footnote.group(1)
    qualifications = {row["selection_qualification"] for row in qualified}
    assert body.count("<li>") == len(qualifications)
    for qualification in qualifications:
        assert qualification in body
    for row in qualified:
        assert row["metric"] in body
        if row.get("dimension") is not None:
            assert row["segment"] in body


def test_overview_family_status_keeps_raw_stat_sig_separate(_family_presentation_snapshot):
    import dataclasses

    from increment.dashboard._html import overview_table
    from increment.dashboard._theme import coeftable_theme
    from increment.estimation.results import Estimate

    snapshot = _family_presentation_snapshot
    overview = snapshot.overview
    assert overview is not None
    clear = Estimate(value=0.5, lb=0.3, ub=0.7, level=0.95)
    original = overview.exploratory[0]
    source_stat_sig = original.stat_sig()

    def displayed(discovery):
        row = original.model_copy(
            update={
                "lift": clear,
                "discovery": discovery,
                "preferred_direction": "increase",
                "alternative": "greater",
            }
        )
        changed = dataclasses.replace(overview, exploratory=(row,))
        changed_snapshot = dataclasses.replace(snapshot, overview=changed)
        shown_row = next(
            candidate
            for candidate in overview_rows(changed_snapshot, None)
            if candidate["metric"] == row.metric
        )
        return shown_row, overview_table(changed_snapshot, None)

    not_selected, inconclusive_html = displayed(False)
    selected, improvement_html = displayed(True)
    assert not_selected["stat_sig"] is source_stat_sig
    assert selected["stat_sig"] is source_stat_sig
    theme = coeftable_theme(snapshot.config.theme)

    def colored_plots(markup, color):
        return sum(
            re.search(r'<rect\b[^>]*\bfill="' + re.escape(color) + '"', plot) is not None
            for plot in re.findall(r"<svg\b[^>]*>.*?</svg>", markup, re.DOTALL)
        )

    assert colored_plots(improvement_html, theme.favorable) == (
        colored_plots(inconclusive_html, theme.favorable) + 1
    )
    assert colored_plots(inconclusive_html, theme.inconclusive) == (
        colored_plots(improvement_html, theme.inconclusive) + 1
    )


@pytest.mark.slow
@pytest.mark.integration
def test_an_unexpected_update_failure_rolls_the_widget_back(monkeypatch):
    from increment.dashboard import _app

    snapshot = overview_snapshot(selected=False)
    widget = _app.DashboardWidget(experiment_for(snapshot), snapshot=snapshot)
    page = widget.document
    failure = RuntimeError("render failed")

    def lost(*_: object, **__: object) -> None:
        raise failure

    monkeypatch.setattr(_app, "show_exploratory_metrics", lost)
    with pytest.raises(RuntimeError) as caught:
        widget.exploratory_metrics = [ADDED]
    assert caught.value is failure
    assert widget.status.startswith("RuntimeError")
    assert widget.exploratory_metrics == []
    assert widget.document == page


@pytest.mark.slow
@pytest.mark.integration
def test_an_added_metric_with_a_display_unit_can_be_removed_and_re_added(
    _family_presentation_snapshot,
):
    from increment.dashboard._app import DashboardWidget

    snapshot = _family_presentation_snapshot
    snapshot = dataclasses.replace(
        snapshot, config=dataclasses.replace(snapshot.config, metric_units={ADDED: "orders"})
    )
    widget = DashboardWidget(experiment_for(snapshot), snapshot=snapshot)
    widget.exploratory_metrics = []
    assert widget.status == ""
    assert widget.exploratory_metrics == []
    widget.exploratory_metrics = [ADDED]
    assert widget.status == ""
    assert [model.name for model in widget._snapshot.exploratory_metrics] == [ADDED]


def test_added_metric_inspector_shows_its_overview_result():
    from increment.dashboard import render_metric_details

    snapshot = overview_snapshot()
    assert snapshot.overview is not None
    row = snapshot.overview.exploratory[0].model_copy(update={"discovery": False})
    snapshot = dataclasses.replace(
        snapshot, overview=dataclasses.replace(snapshot.overview, exploratory=(row,))
    )
    details = render_metric_details(snapshot, metric=ADDED).text
    assert row.lift is not None
    assert f"{row.lift.value:+.1%}" in details
    assert "Added purchase count per enrolled user." in details
    assert "In the exploratory family, not a discovery" in details
    assert "Overview result" in details


def test_added_metric_inspector_discloses_retained_coverage_budget():
    from increment.dashboard import render_metric_details

    snapshot = overview_snapshot()
    assert snapshot.overview is not None
    row = snapshot.overview.exploratory[0]
    assert row.lift is not None
    estimate = row.model_copy(
        update={"lift": row.lift.model_copy(update={"level": 1.0, "alpha": 1e-17})}
    )
    snapshot = dataclasses.replace(
        snapshot,
        overview=dataclasses.replace(snapshot.overview, exploratory=(estimate,)),
    )

    details = render_metric_details(snapshot, metric=ADDED).text
    assert "100% (α=1e-17)" in details


@pytest.mark.integration
def test_segment_reading_retains_its_own_confidence_budget():
    snapshot = overview_snapshot()
    overview = snapshot.overview
    assert overview is not None
    segment = overview.segments[COUNTRY][0]
    assert segment.lift is not None
    segment = segment.model_copy(
        update={"lift": segment.lift.model_copy(update={"level": 1.0, "alpha": 1e-17})}
    )
    snapshot = dataclasses.replace(
        snapshot, overview=dataclasses.replace(overview, segments={COUNTRY: (segment,)})
    )
    rendered = overview_table(snapshot, COUNTRY)
    assert "1e-17" in rendered
    readings = re.findall(
        r'<section\b[^>]*class="[^"]*\binc-readout-cell-evidence\b[^"]*"[^>]*>(.*?)</section>',
        rendered,
        re.S,
    )
    assert any(
        segment.metric in reading
        and str(segment.dimension_value) in reading
        and segment.method in reading
        and "100%" in reading
        and "1e-17" in reading
        for reading in readings
    )


@pytest.mark.integration
def test_segment_posterior_refusals_keep_each_segment_identity_and_context():
    snapshot = overview_snapshot()
    overview = snapshot.overview
    assert overview is not None
    original = overview.segments[COUNTRY][0]
    segments = tuple(
        original.model_copy(
            update={
                "dimension_value": segment,
                "posterior_available": False,
                "posterior_reason_code": "estimation.engine.lift_guard",
                "posterior_reason_context": {"reason": reason, "sample_size": None},
            }
        )
        for segment, reason in (("US", "us_guard"), ("CA", "ca_guard"))
    )
    snapshot = dataclasses.replace(
        snapshot, overview=dataclasses.replace(overview, segments={COUNTRY: segments})
    )
    rendered = overview_table(snapshot, COUNTRY)
    assert "estimation.engine.lift_guard" in rendered
    readings = re.findall(
        r'<section\b[^>]*class="[^"]*\binc-readout-cell-evidence\b[^"]*"[^>]*>(.*?)</section>',
        rendered,
        re.S,
    )
    for segment, reason in (("US", "us_guard"), ("CA", "ca_guard")):
        matching = [reading for reading in readings if reason in reading]
        assert len(matching) == 1
        assert segment in matching[0] and original.method in matching[0]
        assert "estimation.engine.lift_guard" in matching[0]
        assert "unknown" in matching[0]


@pytest.mark.integration
def test_segment_family_selection_is_explicit_and_unselected_forest_is_neutral():
    snapshot = overview_snapshot()
    assert snapshot.overview is not None
    original = snapshot.overview.segments[COUNTRY][0]

    def shown(discovery):
        segment = original.model_copy(update={"discovery": discovery})
        markup = render_explore(
            snapshot, BreakoutEstimates([segment]), metric=segment.metric, view="segments"
        ).text
        plots = "".join(re.findall(r"<svg\b.*?</svg>", markup, re.S))
        return markup, set(re.findall(r'\bstroke="([^"]+)"', plots))

    selected, selected_colors = shown(True)
    unselected, unselected_colors = shown(False)
    assert selected_colors != unselected_colors
    for markup, status in ((selected, "selected"), (unselected, "not selected")):
        readings = re.findall(
            r'<section\b[^>]*class="[^"]*\binc-readout-cell-evidence\b[^"]*"[^>]*>(.*?)</section>',
            markup,
            re.S,
        )
        assert any(
            original.metric in reading
            and str(original.dimension_value) in reading
            and f"<dd>{status}</dd>" in reading
            for reading in readings
        )
        headers = re.findall(r"<th\b[^>]*>(.*?)</th>", markup, re.S)
        assert not any(re.sub(r"<[^>]+>", "", header).strip() == "Discovery" for header in headers)


@pytest.mark.integration
def test_segment_numerical_notes_keep_metric_and_segment_attribution():
    snapshot = overview_snapshot()
    assert snapshot.overview is not None
    original = snapshot.overview.segments[COUNTRY][0]
    notes = {
        "US": "Precision target missed: residual 0.03.",
        "CA": "Precision target missed: residual 0.07.",
    }
    segments = BreakoutEstimates(
        original.model_copy(
            update={
                "dimension_value": segment,
                "note": note,
                "low_reliability": segment == "US",
            }
        )
        for segment, note in notes.items()
    )
    rendered = render_explore(snapshot, segments, metric=original.metric, view="segments").text
    readings = re.findall(
        r'<section\b[^>]*class="[^"]*\binc-readout-cell-evidence\b[^"]*"[^>]*>(.*?)</section>',
        rendered,
        re.S,
    )
    for segment, note in notes.items():
        matching = [reading for reading in readings if note in reading]
        assert len(matching) == 1
        assert original.metric in matching[0] and segment in matching[0]
    assert "Small sample; interpret this estimate cautiously." in rendered
    assert "flagged low reliability." not in rendered


@pytest.mark.integration
def test_overview_table_caveats_follow_the_visible_scope_without_generic_duplicates():
    snapshot = overview_snapshot()
    assert snapshot.overview is not None
    original = snapshot.overview.segments[COUNTRY][0]
    note = "Residual precision target missed: residual 0.03."
    changed = dataclasses.replace(
        snapshot,
        overview=dataclasses.replace(
            snapshot.overview,
            segments={
                **snapshot.overview.segments,
                COUNTRY: BreakoutEstimates(
                    estimate.model_copy(update={"note": note}) if estimate is original else estimate
                    for estimate in snapshot.overview.segments[COUNTRY]
                ),
            },
        ),
    )

    whole = overview_table(changed, None)
    scoped = overview_table(changed, COUNTRY)

    assert note not in whole
    assert note in scoped
    assert "Result caveats" not in scoped
    assert "gt_sourcenote" in scoped
    assert original.metric in scoped
    assert str(original.dimension_value) in scoped
