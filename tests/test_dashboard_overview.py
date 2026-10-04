"""The Explore overview: confirmatory rows, added metrics, and one exploratory family.

Rows come from real DuckDB warehouses read through ``prepare_dashboard``; assertions read the
captured estimates and the rendered overview rows, never their wording.
"""

from __future__ import annotations

import pytest

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import DashboardConfig, prepare_dashboard
from increment.dashboard._data import decision_rows
from increment.dashboard._html import overview_rows
from increment.errors import InvalidRequestError
from tests.test_dashboard_capture import _METRICS, _double_the_treatment_arm, _workspace

pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

ADDED = "purchases_per_user"
_SAVED = (
    _METRICS
    + f"""
  - type: mean
    name: {ADDED}
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
COUNTRY = ("event_log", "country")


def _config(*added: str) -> DashboardConfig:
    return DashboardConfig(
        expected_allocation={"control": 0.5, "treatment": 0.5}, exploratory_metrics=added
    )


def _identity(row) -> tuple:
    return (row["metric"], row["group_id"], row["lift"], row["lower"], row["higher"])


def test_declared_whole_experiment_rows_are_the_readout_rows(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
        readout = {_identity(row): row["stat_sig"] for row in decision_rows(snapshot)}
        for scope in (None, COUNTRY):
            shown = {
                _identity(row): row["stat_sig"]
                for row in overview_rows(snapshot, scope)
                if row["metric"] != ADDED and row.get("dimension") is None
            }
            assert shown == readout
        assert ADDED not in {row["metric"] for row in snapshot.readout_rows}


def test_one_family_covers_added_metrics_and_every_segment_cell(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        segments = overview.segments[COUNTRY]
        cells = [*overview.exploratory, *segments]
        tested = [cell for cell in cells if cell.family_size is not None]
        assert {cell.metric for cell in overview.exploratory} == {ADDED}
        assert {cell.metric for cell in segments} >= {ADDED, "checkout_conversion"}
        assert len(tested) == overview.family_size > len(overview.exploratory)
        assert {cell.family_size for cell in tested} == {overview.family_size}
        assert {cell.family_q for cell in tested} == {overview.family_q}
        # The marker on every exploratory overview row is its family discovery.
        for row in overview_rows(snapshot, COUNTRY):
            if row["metric"] == ADDED or row.get("dimension") is not None:
                assert row["stat_sig"] is bool(row.get("discovery"))


def test_only_a_discovery_is_coloured_as_a_finding(tmp_path):
    import dataclasses

    from increment.dashboard._html import overview_table
    from increment.dashboard._theme import coeftable_theme
    from increment.estimation.results import Estimate

    with _workspace(tmp_path, metrics=_SAVED) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
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


def test_added_metrics_keep_their_own_absolute_value_series(tmp_path):
    from increment.dashboard._app import build_payload

    with _workspace(tmp_path, metrics=_SAVED) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
        explore = build_payload(analysis, snapshot=snapshot)["explore"]["overall"][ADDED]
        for view in ("cumulative_values", "daily_values", "cumulative_lift"):
            assert explore[view]["pointCount"] > 0


def test_a_metric_the_breakout_refuses_keeps_every_other_metrics_segments(tmp_path):
    with _workspace(tmp_path, metrics=_WITH_QUANTILE) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED, QUANTILE))
        overview = snapshot.overview
        assert overview is not None and overview.family_refusal is None
        shown = {cell.metric for cell in overview.segments[COUNTRY]}
        assert {ADDED, "checkout_conversion"} <= shown
        assert QUANTILE not in shown
        refused = {(metric, place) for metric, place, _ in overview.refusals}
        assert (QUANTILE, "country") in refused
        assert {metric for metric, _, _ in overview.refusals} == {QUANTILE}


def test_two_sources_of_one_property_are_separate_comparisons(tmp_path):
    sources = ("event_log", "profiles")
    with _workspace(tmp_path, metrics=_SAVED, breakout_sources=sources) as (_, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
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


def test_adding_a_metric_grows_the_family_without_moving_declared_results(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (_, analysis):
        without = prepare_dashboard(analysis, config=_config())
        with_added = prepare_dashboard(analysis, config=_config(ADDED))
        assert with_added.readout_rows == without.readout_rows
        assert without.overview is not None and with_added.overview is not None
        assert with_added.overview.family_size > without.overview.family_size


def test_an_undeclared_exploratory_metric_is_refused_before_preparation(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, analysis):
        con.disconnect()
        for name in ("not_a_saved_metric", "checkout_conversion"):
            with pytest.raises(InvalidRequestError) as caught:
                prepare_dashboard(analysis, config=_config(name))
            assert caught.value.code == "dashboard.invalid_config"
            assert caught.value.context["available"] == (ADDED,)


@pytest.mark.slow
def test_added_metrics_come_from_the_pinned_read(tmp_path):
    with _workspace(tmp_path, metrics=_SAVED) as (con, analysis):
        snapshot = prepare_dashboard(analysis, config=_config(ADDED))
        assert snapshot.overview is not None
        before = snapshot.overview.exploratory
        _double_the_treatment_arm(con)
        assert snapshot.overview.exploratory == before
        fresh = prepare_dashboard(analysis, config=_config(ADDED))
        assert fresh.overview is not None
        dumped = [row.model_dump(mode="json") for row in fresh.overview.exploratory]
        assert dumped != [row.model_dump(mode="json") for row in before]
