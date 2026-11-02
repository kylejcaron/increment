"""Explore overview presentation from constructed snapshots."""

from __future__ import annotations

import dataclasses
import re

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard._data import decision_rows
from increment.dashboard._html import overview_rows
from tests.dashboard_snapshots import ADDED, COUNTRY, experiment_for, overview_snapshot

pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")


@pytest.fixture
def _saved_added_snapshot():
    return overview_snapshot()


def _identity(row) -> tuple:
    return (row["metric"], row["group_id"], row["lift"], row["lower"], row["higher"])


def _family_presentation_snapshot():
    return overview_snapshot()


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


def test_explore_tables_show_only_core_readout_columns():
    from increment.dashboard._html import overview_table

    snapshot = _family_presentation_snapshot()
    overview_html = overview_table(snapshot, COUNTRY)
    assert "Discovery" in overview_html
    assert "Numerical qualification" not in overview_html
    assert "Selection qualification" not in overview_html


def test_overview_qualification_footnotes_name_rows_and_family():
    from increment.dashboard._html import overview_table

    snapshot = _family_presentation_snapshot()
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
    assert "Selection qualification" not in html


def test_only_a_discovery_is_coloured_as_a_finding():
    import dataclasses

    from increment.dashboard._html import overview_table
    from increment.dashboard._theme import coeftable_theme
    from increment.estimation.results import Estimate

    snapshot = _family_presentation_snapshot()
    overview = snapshot.overview
    assert overview is not None
    clear = Estimate(value=0.5, lb=0.3, ub=0.7, level=0.95)

    def shown(discovery: bool) -> str:
        row = overview.exploratory[0].model_copy(update={"lift": clear, "discovery": discovery})
        changed = dataclasses.replace(overview, exploratory=(row,))
        return overview_table(dataclasses.replace(snapshot, overview=changed), None)

    favorable = coeftable_theme(snapshot.config.theme).favorable
    # An unadjusted interval clearing zero is not a finding unless the family selects it.
    assert shown(False).count(favorable) < shown(True).count(favorable)
    qualified = shown(True)
    for token in (
        "Benjamini-Hochberg",
        "FCR-adjusted",
        "independence or positive dependence",
        "not confirmatory",
    ):
        assert token in qualified
    selected = overview.exploratory[0].model_copy(update={"lift": clear, "discovery": True})
    selected_overview = dataclasses.replace(overview, exploratory=(selected,))
    selected_rows = overview_rows(dataclasses.replace(snapshot, overview=selected_overview), None)
    selected_row = next(row for row in selected_rows if row["discovery"])
    assert selected_row["family_q"] == overview.family_q
    assert selected_row["family_size"] == overview.family_size
    representation = repr(selected)
    assert "Benjamini-Hochberg" in representation
    assert "FCR-adjusted" in representation
    assert "independence or positive dependence" in representation
    assert "not confirmatory" in representation


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
def test_an_added_metric_with_a_display_unit_can_be_removed_and_re_added():
    from increment.dashboard._app import DashboardWidget

    snapshot = overview_snapshot()
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
