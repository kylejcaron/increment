"""The Explore overview: confirmatory rows, added metrics, and one exploratory family.

Rows come from real DuckDB warehouses read through ``prepare_dashboard``; assertions read the
captured estimates and the rendered overview rows, never their wording.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import DashboardConfig, prepare_dashboard
from increment.dashboard._data import decision_rows
from increment.dashboard._html import overview_notes, overview_rows
from increment.errors import CapabilityError, InvalidRequestError
from tests.test_dashboard_capture import _METRICS, _double_the_treatment_arm, _workspace

pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

ADDED = "purchases_per_user"
_SAVED = (
    _METRICS
    + f"""
  - type: mean
    name: {ADDED}
    description: Added purchase count per enrolled user.
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: count
    window_days: 14
"""
)
QUANTILE = "p90_revenue"
_WITH_QUANTILE = (
    _SAVED
    + f"""
  - type: quantile
    name: {QUANTILE}
    quantile: 0.9
    entity: user_id
    fact: purchase
    aggregation: sum
"""
)
SECOND = "page_views_per_user"
_TWO_SAVED = (
    _SAVED
    + f"""
  - type: mean
    name: {SECOND}
    entity: user_id
    preferred_direction: increase
    fact: page_view
    aggregation: count
    window_days: 14
"""
)
COUNTRY = ("event_log", "country")


def _config(*added: str) -> DashboardConfig:
    return DashboardConfig(
        expected_allocation={"control": 0.5, "treatment": 0.5}, exploratory_metrics=added
    )


@pytest.fixture(scope="module")
def _saved_added_snapshot(tmp_path_factory):
    # DashboardSnapshot is read-only and holds no connection; build it from a real workspace,
    # then let that workspace close before presentation-only consumers share the snapshot.
    with _workspace(tmp_path_factory.mktemp("overview-snapshot"), metrics=_SAVED) as (
        _,
        experiment,
        resolver,
    ):
        return prepare_dashboard(experiment, resolver, config=_config(ADDED))


def _identity(row) -> tuple:
    return (row["metric"], row["group_id"], row["lift"], row["lower"], row["higher"])


def _family_presentation_snapshot():
    import datetime as dt

    from increment.breakout.estimates import BreakoutEstimate
    from increment.dashboard import DashboardConfig, DashboardSnapshot
    from increment.dashboard._data import DashboardOverview
    from increment.estimation.results import Estimate, LiftEstimate
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(
        name=ADDED,
        description="Added purchase count per enrolled user.",
        entity="user_id",
        fact="purchase",
        aggregation="count",
        window_days=14,
        preferred_direction="increase",
    )
    common: Any = {
        "method": "mean",
        "method_role": "decision",
        "lift": Estimate(value=0.2, lb=0.05, ub=0.35, level=0.95),
        "preferred_direction": "increase",
        "family_size": 2,
        "family_q": 0.05,
        "family_procedure": "bh",
        "family_nominal_alpha": 0.05,
        "family_guarantee": "finite_sample",
        "family_axes": ("metric", "breakout"),
        "discovery": True,
        "multiplicity_status": "exploratory_family",
    }
    whole = LiftEstimate(metric=ADDED, group_id="treatment", **common)
    segment = BreakoutEstimate(
        metric=ADDED,
        group_id="treatment",
        dimension="country",
        dimension_value="US",
        source="event_log",
        **common,
    )
    return DashboardSnapshot(
        experiment_name="overview",
        binding_fingerprint="fixture",
        title="Overview",
        description=None,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 31, tzinfo=dt.UTC),
        control_group="control",
        treatment_group="treatment",
        primary_metric=ADDED,
        metrics=(),
        breakouts=(COUNTRY,),
        estimates=(),
        readout_rows=(),
        allocation=None,
        allocation_refusal=("dashboard.allocation_not_applicable", "fixture"),
        allocation_history=(),
        allocation_history_refusal=None,
        config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        computed_at=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
        group_data=(),
        overview=DashboardOverview(
            exploratory=(whole,),
            segments={COUNTRY: (segment,)},
            family_size=2,
            family_q=0.05,
        ),
        exploratory_metrics=(metric,),
        offered_metrics=(metric,),
    )


@pytest.mark.xdist_group("dashboard_overview_saved_added_snapshot")
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


@pytest.fixture(scope="module")
def _selected_family_snapshot(tmp_path_factory):
    # Enough units that the exploratory BH family selects at least one cell.
    with _workspace(
        tmp_path_factory.mktemp("overview-selected-snapshot"), units=1000, metrics=_SAVED
    ) as (_, experiment, resolver):
        return prepare_dashboard(experiment, resolver, config=_config(ADDED))


@pytest.mark.integration
@pytest.mark.slow
def test_one_family_covers_added_metrics_and_every_segment_cell(_selected_family_snapshot):
    snapshot = _selected_family_snapshot
    overview = snapshot.overview
    assert overview is not None and overview.family_refusal is None
    segments = overview.segments[COUNTRY]
    cells = [*overview.exploratory, *segments]
    tested = [cell for cell in cells if cell.family_size is not None]
    assert {cell.metric for cell in overview.exploratory} == {ADDED}
    assert {cell.metric for cell in segments} >= {ADDED, "checkout_conversion"}
    assert len(tested) == overview.family_size > len(overview.exploratory)
    assert {cell.family_size for cell in tested} == {overview.family_size}
    selected = next(cell for cell in tested if cell.discovery is True)
    assert selected.lift is not None
    restored = type(selected).model_validate_json(selected.model_dump_json())
    assert restored.lift is not None
    assert restored.discovery is True
    assert restored.family_procedure == "bh"
    assert restored.family_q == pytest.approx(selected.family_q)
    assert restored.family_size == selected.family_size
    assert restored.family_axes == selected.family_axes
    assert restored.lift.level == pytest.approx(selected.lift.level)
    assert restored.lift.lb == pytest.approx(selected.lift.lb)
    assert restored.lift.ub == pytest.approx(selected.lift.ub)
    from increment.breakout.estimates import BreakoutEstimate
    from increment.dashboard._html import overview_table
    from increment.estimation.results import LiftEstimate

    if isinstance(selected, BreakoutEstimate):
        assert isinstance(restored, BreakoutEstimate)
        segments = dict(overview.segments)
        segments[COUNTRY] = tuple(restored if row is selected else row for row in segments[COUNTRY])
        restored_overview = dataclasses.replace(overview, segments=segments)
        scope = COUNTRY
    else:
        assert isinstance(restored, LiftEstimate)
        restored_overview = dataclasses.replace(
            overview,
            exploratory=tuple(restored if row is selected else row for row in overview.exploratory),
        )
        scope = None
    restored_snapshot = dataclasses.replace(snapshot, overview=restored_overview)
    representation = repr(restored)
    rendered = overview_table(restored_snapshot, scope)
    assert "Benjamini-Hochberg" in representation
    assert "FCR-adjusted" in representation
    assert "independence or positive dependence" in representation
    assert "Benjamini-Hochberg" in rendered
    assert {cell.family_q for cell in tested} == {overview.family_q}
    # The marker on every exploratory overview row is its family discovery.
    for row in overview_rows(snapshot, COUNTRY):
        if row["metric"] == ADDED or row.get("dimension") is not None:
            assert row["stat_sig"] is bool(row.get("discovery"))


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


@pytest.mark.integration
@pytest.mark.slow
def test_added_metrics_keep_their_own_absolute_value_series(tmp_path):
    from increment.dashboard._app import build_payload

    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        explore = build_payload(experiment, snapshot=snapshot)["explore"]["overall"][ADDED]
        for view in ("cumulative_values", "daily_values", "cumulative_lift"):
            assert explore[view]["pointCount"] > 0


@pytest.mark.integration
@pytest.mark.slow
def test_a_metric_the_breakout_refuses_keeps_every_other_metrics_segments(tmp_path):
    with _workspace(tmp_path, metrics=_WITH_QUANTILE) as (_, experiment, resolver):
        # The quantile metric is offered but not shown; its refusal is still reported.
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        shown = {cell.metric for cell in overview.segments[COUNTRY]}
        assert {ADDED, "checkout_conversion"} <= shown
        assert QUANTILE not in shown
        refused = {(metric, place) for metric, place, _ in overview.refusals}
        assert (QUANTILE, "country") in refused
        assert {metric for metric, _, _ in overview.refusals} == {QUANTILE}
        assert any(QUANTILE in note for note in overview_notes(snapshot, COUNTRY))


@pytest.mark.integration
@pytest.mark.slow
def test_two_sources_of_one_property_are_separate_comparisons(tmp_path):
    sources = ("event_log", "profiles")
    with _workspace(tmp_path, metrics=_SAVED, breakout_sources=sources) as (
        _,
        experiment,
        resolver,
    ):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        by_source = {source: overview.segments[(source, "country")] for source in sources}
        assert all(by_source.values())
        tested = [
            cell
            for cells in (overview.exploratory, *by_source.values())
            for cell in cells
            if cell.family_size is not None
        ]
        assert len(tested) == overview.family_size


@pytest.mark.integration
@pytest.mark.slow
def test_choosing_what_to_show_never_changes_the_family(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        hidden = prepare_dashboard(experiment, resolver, config=_config())
        shown = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert shown.readout_rows == hidden.readout_rows
        assert [model.name for model in hidden.offered_metrics] == [ADDED]
        assert hidden.exploratory_metrics == () and hidden.overview is not None
        assert shown.overview is not None
        assert hidden.overview.family_size == shown.overview.family_size

        def cells(overview) -> list:
            rows = [*overview.exploratory, *(r for s in overview.segments.values() for r in s)]
            return [row.model_dump(mode="json") for row in rows]

        assert cells(hidden.overview) == cells(shown.overview)


@pytest.mark.integration
def test_an_undeclared_exploratory_metric_is_refused_before_preparation(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        con.disconnect()
        for name in ("not_a_saved_metric", "checkout_conversion"):
            with pytest.raises(InvalidRequestError) as caught:
                prepare_dashboard(experiment, resolver, config=_config(name))
            assert caught.value.code == "dashboard.invalid_config"
            assert caught.value.context["available"] == (ADDED,)


@pytest.mark.slow
@pytest.mark.integration
def test_added_metrics_come_from_the_pinned_read(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert snapshot.overview is not None
        before = snapshot.overview.exploratory
        _double_the_treatment_arm(con)
        assert snapshot.overview.exploratory == before
        fresh = prepare_dashboard(experiment, resolver, config=_config(ADDED))
        assert fresh.overview is not None
        dumped = [row.model_dump(mode="json") for row in fresh.overview.exploratory]
        assert dumped != [row.model_dump(mode="json") for row in before]


def _page_payload(page: str) -> dict:
    import json
    import re

    match = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', page)
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.slow
@pytest.mark.integration
def test_the_dashboard_widget_shows_added_metrics_without_reading_again(tmp_path):
    from increment.dashboard._app import DashboardWidget

    with _workspace(tmp_path, metrics=_SAVED) as (con, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config())
        widget = DashboardWidget(experiment, snapshot=snapshot)
        offered = _page_payload(widget.document)["exploratory"]
        assert [item["key"] for item in offered["available"]] == [ADDED]
        assert offered["selected"] == []

        # A later warehouse change never reaches the page: selection re-renders this snapshot.
        _double_the_treatment_arm(con)
        widget.exploratory_metrics = [ADDED]
        assert widget.status == ""
        payload = _page_payload(widget.document)
        assert payload["exploratory"]["selected"] == [ADDED]
        assert ADDED in {metric["key"] for metric in payload["metrics"]}
        assert widget._snapshot.readout_rows == snapshot.readout_rows
        assert widget._snapshot.overview == snapshot.overview
        assert widget._snapshot.computed_at == snapshot.computed_at

        page = widget.document
        widget.exploratory_metrics = ["not_a_saved_metric"]
        assert "dashboard.invalid_config" in widget.status
        assert widget.exploratory_metrics == [ADDED]
        assert widget.document == page

        widget.exploratory_metrics = []
        assert widget.status == ""
        assert ADDED not in {metric["key"] for metric in _page_payload(widget.document)["metrics"]}


@pytest.mark.integration
@pytest.mark.slow
def test_an_unexpected_update_failure_rolls_the_widget_back(tmp_path, monkeypatch):
    from increment.dashboard import _app

    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        widget = _app.DashboardWidget(
            experiment, snapshot=prepare_dashboard(experiment, resolver, config=_config())
        )
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


_REPORT_ONLY = (
    _SAVED
    + """
  - type: total
    name: total_revenue
    fact: purchase
    aggregation: sum
  - type: active
    name: active_users
    entity: user_id
    fact: page_view
"""
)


@pytest.mark.integration
def test_only_estimable_saved_metrics_are_offered(tmp_path):
    from increment.workflows import available_metrics

    with _workspace(tmp_path, metrics=_REPORT_ONLY) as (con, experiment, resolver):
        assert [metric.name for metric in available_metrics(experiment, resolver)] == [ADDED]
        con.disconnect()
        with pytest.raises(InvalidRequestError) as caught:
            prepare_dashboard(experiment, resolver, config=_config("total_revenue"))
        assert caught.value.code == "dashboard.invalid_config"


@pytest.mark.slow
@pytest.mark.integration
def test_an_added_metric_with_a_display_unit_can_be_removed_and_re_added(tmp_path):
    from increment.dashboard._app import DashboardWidget

    with _workspace(tmp_path, metrics=_SAVED) as (_, experiment, resolver):
        config = dataclasses.replace(_config(ADDED), metric_units={ADDED: "orders"})
        widget = DashboardWidget(
            experiment, snapshot=prepare_dashboard(experiment, resolver, config=config)
        )
        widget.exploratory_metrics = []
        assert widget.status == ""
        assert widget.exploratory_metrics == []
        widget.exploratory_metrics = [ADDED]
        assert widget.status == ""
        assert [model.name for model in widget._snapshot.exploratory_metrics] == [ADDED]


_OBSERVATIONAL = """    design:
      mechanism: observational
      covariates:
        - {property: country, source: event_log}
"""


@pytest.mark.integration
def test_a_non_randomized_design_is_refused_before_any_read(tmp_path, monkeypatch):
    import tests.test_dashboard_capture as capture

    declared = capture._experiments
    monkeypatch.setattr(
        capture, "_experiments", lambda **options: declared(**options) + _OBSERVATIONAL
    )
    with _workspace(tmp_path) as (con, experiment, resolver):
        con.disconnect()
        with pytest.raises(CapabilityError) as caught:
            prepare_dashboard(experiment, resolver, config=_config())
        assert caught.value.code == "dashboard.unsupported_experiment"
        assert caught.value.context["mechanism"] == "observational"


def _flat(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        return {
            k: v for key, item in value.items() for k, v in _flat(item, f"{prefix}{key}.").items()
        }
    return {prefix.rstrip("."): value}


def _rows(collection) -> list[dict[str, object]]:
    rows = [_flat(row.model_dump(mode="json")) for row in collection]
    rows = [
        {key: value for key, value in row.items() if key not in {"source_snapshot_id", "family_id"}}
        for row in rows
    ]
    return sorted(
        rows, key=lambda row: tuple(str(row.get(k)) for k in ("ds", "group_id", "dimension_value"))
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["bonferroni", "bh"])
@pytest.mark.integration
def test_offered_metrics_read_together_match_each_metric_read_alone(
    tmp_path, monkeypatch, correction
):
    import tests.test_dashboard_capture as capture
    from increment import Scope
    from increment.dashboard import load_explore
    from increment.dashboard._data import ExploreView
    from increment.evidence import BreakoutIdentity
    from tests.analysis_factory import view_readout
    from tests.parity_harness.comparison import nested_close

    monkeypatch.setattr(
        capture,
        "_PLAN",
        capture._PLAN.replace("correction: bonferroni", f"correction: {correction}"),
    )
    source, dimension = COUNTRY
    alone: dict[tuple[ExploreView, tuple[str, str] | None], Scope] = {
        ("cumulative_lift", None): Scope(view="asof", grain="asof"),
        ("cumulative_values", None): Scope(view="asof", grain="asof", values=True),
        ("daily_values", None): Scope(view="daily", grain="daily", values=True),
        ("segments", COUNTRY): Scope(
            view="breakout", grain="total", segment=BreakoutIdentity(dimension, source)
        ),
    }
    # Enough units that a segment is a BH discovery, so a joint family would move its threshold.
    with _workspace(tmp_path, metrics=_TWO_SAVED, units=2000) as (_, experiment, resolver):
        snapshot = prepare_dashboard(experiment, resolver, config=_config(ADDED, SECOND))
        discoveries = 0
        for name in (ADDED, SECOND):
            for (view, breakout), scope in alone.items():
                expected = view_readout(
                    experiment, resolver, scope, metrics=[], exploratory_metrics=[name]
                )
                captured = load_explore(
                    experiment, snapshot=snapshot, metric=name, view=view, breakout=breakout
                )
                want, got = _rows(expected), _rows(captured)
                assert len(got) == len(want) > 0
                assert nested_close(want, got)
                if view == "segments":
                    discoveries += sum(bool(getattr(row, "discovery", None)) for row in captured)
        assert discoveries > 0 or correction == "bonferroni"


def test_added_metric_inspector_shows_its_overview_result():
    import datetime as dt

    from increment.dashboard import (
        DashboardConfig,
        DashboardSnapshot,
        render_metric_details,
    )
    from increment.dashboard._data import DashboardOverview
    from increment.estimation.results import Estimate, LiftEstimate
    from increment.semantics.models import MeanMetric

    added = MeanMetric(
        name=ADDED,
        description="Added purchase count per enrolled user.",
        entity="user_id",
        fact="purchase",
        aggregation="count",
        window_days=14,
        preferred_direction="increase",
    )
    row = LiftEstimate(
        metric=ADDED,
        group_id="treatment",
        method="mean",
        method_role="decision",
        lift=Estimate(value=0.2, lb=0.05, ub=0.35, level=0.95),
        preferred_direction="increase",
        family_size=1,
        family_q=0.05,
        family_procedure="bh",
        family_nominal_alpha=0.05,
        family_guarantee="finite_sample",
        family_axes=("metric",),
        discovery=False,
    )
    snapshot = DashboardSnapshot(
        experiment_name="overview",
        binding_fingerprint="fixture",
        title="Overview",
        description=None,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 31, tzinfo=dt.UTC),
        control_group="control",
        treatment_group="treatment",
        primary_metric="checkout_conversion",
        metrics=(),
        allocation=None,
        breakouts=(),
        estimates=(),
        readout_rows=(),
        allocation_refusal=("dashboard.allocation_not_applicable", "fixture"),
        allocation_history=(),
        allocation_history_refusal=None,
        config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        computed_at=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
        group_data=(),
        overview=DashboardOverview(
            exploratory=(row,),
            segments={},
            family_size=1,
            family_q=0.05,
        ),
        exploratory_metrics=(added,),
        offered_metrics=(added,),
    )
    details = render_metric_details(snapshot, metric=ADDED).text
    assert row.lift is not None
    assert f"{row.lift.value:+.1%}" in details
    assert "Added purchase count per enrolled user." in details
    assert "In the exploratory family, not a discovery" in details
    assert "Overview result" in details
