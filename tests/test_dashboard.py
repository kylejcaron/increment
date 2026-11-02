"""Dashboard sections and exports consume constructed snapshots."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from increment.errors import InvalidRequestError

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import (
    DashboardConfig,
    DashboardSnapshot,
    ExploreView,
    readout_csv,
    render_explore,
    render_header,
    render_health,
    render_metric_details,
    render_results,
)
from increment.dashboard._data import enriched_rows
from tests.dashboard_snapshots import (
    binomial_region,
    experiment_for,
    pricing_snapshot,
    temporal_rows,
)
from tests.dashboard_snapshots import snapshot as make_snapshot


@pytest.fixture
def storefront() -> DashboardSnapshot:
    return make_snapshot()


@pytest.fixture
def pricing() -> DashboardSnapshot:
    return pricing_snapshot()


# Header and allocation qualifications.


@pytest.mark.slow
def test_dashboard_explains_missing_allocation_scheme_separately(
    pricing: DashboardSnapshot,
) -> None:
    from increment.estimation.readout_types import IntegrityStatus

    collection = pricing.readout_collection
    assert collection is not None and collection.metadata is not None
    metadata = collection.metadata
    source_id, source = next(iter(metadata.scope.by_source.items()))
    integrity = tuple(
        item.model_copy(
            update={
                "status": IntegrityStatus.NOT_CHECKED_MISSING_DECLARATION,
                "code": "integrity.allocation_scheme_missing",
                "context": {"allocation_scheme": None},
            }
        )
        if item.analysis_population == "assigned"
        else item
        for item in source.integrity
    )
    changed_source = source.model_copy(update={"integrity": integrity})
    changed_scope = metadata.scope.model_copy(
        update={"by_source": {**metadata.scope.by_source, source_id: changed_source}}
    )
    changed_collection = type(collection)(
        collection, metadata=metadata.model_copy(update={"scope": changed_scope})
    )
    snapshot = dataclasses.replace(
        pricing,
        readout_collection=changed_collection,
        population_collections={**pricing.population_collections, "assigned": changed_collection},
    )

    text = render_header(snapshot).text
    assert "the allocation scheme is not declared" in text
    assert "integrity.allocation_scheme_missing" in text
    assert "no allocation declared" not in text


def test_dashboard_names_declared_weights_not_observed_counts(pricing: DashboardSnapshot) -> None:
    from increment.estimation.readout_types import IntegrityStatus

    collection = pricing.readout_collection
    assert collection is not None and collection.metadata is not None
    metadata = collection.metadata
    source_id, source = next(iter(metadata.scope.by_source.items()))
    integrity = tuple(
        item.model_copy(
            update={
                "status": IntegrityStatus.NOT_CHECKED_MISSING_DECLARATION,
                "code": "integrity.allocation_missing_or_nonconstant",
                "context": {"allocation": None},
            }
        )
        if item.analysis_population == "assigned"
        else item
        for item in source.integrity
    )
    changed_source = source.model_copy(update={"integrity": integrity})
    changed_scope = metadata.scope.model_copy(
        update={"by_source": {**metadata.scope.by_source, source_id: changed_source}}
    )
    changed_collection = type(collection)(
        collection, metadata=metadata.model_copy(update={"scope": changed_scope})
    )
    snapshot = dataclasses.replace(
        pricing,
        readout_collection=changed_collection,
        population_collections={**pricing.population_collections, "assigned": changed_collection},
    )
    text = render_header(snapshot).text
    assert "integrity.allocation_missing_or_nonconstant" in text
    assert "declared target allocation weights" in text
    assert "allocation counts are missing or nonconstant" not in text


def test_renders_expected_shares_from_configuration(pricing: DashboardSnapshot) -> None:
    html = render_health(pricing).text
    assert "70.0%" in html
    assert "30.0%" in html
    assert "baseline" in html
    assert "candidate" in html


def test_assigned_health_collapses_inapplicable_checks_and_omits_duplicate_allocation(
    pricing: DashboardSnapshot,
) -> None:
    html = render_health(pricing).text
    assert html.count("5 trigger checks don't apply: no trigger declared.") == 1
    visible = html[: html.index("<summary>Technical details</summary>")]
    assert "Assigned Allocation" not in visible
    assert "Not applicable; this is not an evaluated check." not in visible


@pytest.mark.slow
def test_health_lists_every_triggered_check(storefront: DashboardSnapshot) -> None:
    from increment.results import ExperimentHealth, HealthCheck

    checks = tuple(
        HealthCheck(
            check=kind,
            analysis_population="triggered",
            availability="available",
            conclusion="descriptive",
            context={"description": "Captured trigger diagnostic."},
        )
        for kind in (
            "triggered_balance",
            "trigger_rates",
            "trigger_maturity",
            "feed_completeness",
            "trigger_assumptions",
        )
    )
    snapshot = dataclasses.replace(storefront, health=ExperimentHealth(checks=checks))
    html = render_health(snapshot).text
    health = snapshot.health
    assert health is not None
    triggered_checks = [
        check
        for check in health.checks
        if check.analysis_population == "triggered" and check.availability != "not_applicable"
    ]
    assert triggered_checks
    for check in triggered_checks:
        assert check.check.replace("_", " ").title() in html


def test_health_renderer_consumes_restored_health_report(
    pricing: DashboardSnapshot, tmp_path: Path
) -> None:
    from increment.results import ExperimentHealth, HealthCheck

    report = ExperimentHealth(
        checks=(
            HealthCheck(
                check="assigned_allocation",
                analysis_population="assigned",
                availability="unavailable",
                code="integrity.allocation_scheme_missing",
                context={
                    "reason": "Assignment-law admission is not_checked_missing_declaration.",
                    "route": (
                        "Declare allocation_scheme='independent' on a design that uses an "
                        "independent constant-probability draw, or use "
                        "workflows.srm(experiment, resolver, inference='fixed') for one "
                        "predeclared look."
                    ),
                },
            ),
            HealthCheck(
                check="trigger_rates",
                analysis_population="triggered",
                availability="available",
                conclusion="descriptive",
                rates={"baseline": 0.25, "candidate": 0.25},
                context={"description": "Observed trigger share by assigned arm."},
            ),
            HealthCheck(
                check="feed_completeness",
                analysis_population="triggered",
                feed="event_log",
                availability="unavailable",
                context={"reason": "No certified completeness watermark is retained."},
            ),
            HealthCheck(
                check="trigger_assumptions",
                analysis_population="triggered",
                availability="available",
                conclusion="descriptive",
                context={
                    "counterfactual_trigger_invariance": "assumed, not established",
                    "trigger_order_stationarity": "assumed, not established",
                    "no_effect_before_trigger": "pre-trigger assumption sentinel",
                    "triggered_look_predictability": "triggered-filtration assumption sentinel",
                    "family_guarantee_status": "assumed, not verified",
                    "family_guarantee": "asymptotic_sequential",
                    "risk": ("Balanced rates do not establish trigger invariance.",),
                },
            ),
        )
    )
    restored = ExperimentHealth.model_validate_json(report.model_dump_json())
    snapshot = dataclasses.replace(
        pricing,
        health=restored,
        allocation=None,
        allocation_refusal=(
            "integrity.allocation_scheme_missing",
            "independent assignment was not declared",
        ),
    )
    html_path = tmp_path / "health.html"
    html_path.write_text(render_health(snapshot).text)
    html = html_path.read_text()

    assert "Observed trigger rate" in html
    assert "No certified completeness watermark is retained." in html
    assert "asymptotic_sequential" in html
    assert "Triggered Look Policy" not in html
    assert "Balanced rates do not establish trigger invariance." in html
    assert "independent constant-probability draw" in html
    assert "one predeclared look" in html
    assert "assumed, not verified" in html
    for condition in ("no_effect_before_trigger", "triggered_look_predictability"):
        value = restored.checks[-1].context[condition]
        assert isinstance(value, str)
        assert value in html
    assert "Declare allocation_scheme" in html[: html.index("<summary>Technical details</summary>")]
    technical = re.search(
        r"<details[^>]*><summary>Technical details</summary>(.*?)</details>", html, re.DOTALL
    )
    assert technical is not None
    assert "integrity.allocation_scheme_missing" in technical.group(1)
    assert "Counterfactual Trigger Invariance" in technical.group(1)
    assert "pre-trigger assumption sentinel" in technical.group(1)
    assert "integrity.allocation_scheme_missing" not in html[: html.index("<details")]
    assert "Assignment-law admission" not in html[: html.index("<details")]


@pytest.mark.slow
def test_both_experiments_share_the_same_sections(
    storefront: DashboardSnapshot, pricing: DashboardSnapshot
) -> None:
    """Identical calls, different experiments: only the bound data differs."""
    for snapshot, enrolled, arms in (
        (storefront, 1200, ("control", "treatment")),
        (pricing, 1000, ("baseline", "candidate")),
    ):
        header = render_header(snapshot).text
        health = render_health(snapshot).text
        assert f"{enrolled:,}" in header
        assert snapshot.title in header
        assert "No allocation issues detected" in health
        for arm in arms:
            assert arm in header
            assert arm in health


def _with_primary_row(snapshot: DashboardSnapshot, **changes: Any) -> DashboardSnapshot:
    rows = tuple(
        {**dict(row), **changes} if row["metric"] == snapshot.primary_metric else row
        for row in snapshot.readout_rows
    )
    return dataclasses.replace(snapshot, readout_rows=rows)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("direction", "lower", "higher", "tone"),
    [
        ("increase", 0.02, 0.18, "favorable"),
        ("increase", -0.18, -0.02, "unfavorable"),
        ("decrease", -0.18, -0.02, "favorable"),
        ("decrease", 0.02, 0.18, "unfavorable"),
    ],
)
def test_primary_headline_colors_significant_results_by_declared_direction(
    storefront: DashboardSnapshot,
    direction: str,
    lower: float,
    higher: float,
    tone: str,
) -> None:
    snapshot = _with_primary_row(
        storefront,
        lift=(lower + higher) / 2,
        lower=lower,
        higher=higher,
        stat_sig=True,
        preferred_direction=direction,
    )

    html = render_header(snapshot).text

    assert f"inc-dashboard-primary--{tone}" in html
    assert f">Significant · {tone}</span>" in html


@pytest.mark.slow
def test_primary_headline_keeps_directionless_significance_neutral(
    storefront: DashboardSnapshot,
) -> None:
    snapshot = _with_primary_row(
        storefront,
        lift=0.10,
        lower=0.02,
        higher=0.18,
        stat_sig=True,
        preferred_direction=None,
    )

    html = render_header(snapshot).text

    assert "inc-dashboard-primary--neutral" in html
    assert ">Significant</span>" in html


# Allocation history presentation and snapshot validation.


@pytest.mark.slow
def test_allocation_history_cannot_also_be_refused(storefront: DashboardSnapshot) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        dataclasses.replace(
            storefront, allocation_history_refusal=("evidence.operation_unavailable", "unsupported")
        )
    assert caught.value.code == "dashboard.invalid_snapshot"


def test_allocation_uncertainty_remains_nonzero_at_zero_and_full_share(
    storefront: DashboardSnapshot,
) -> None:
    history = tuple(
        {
            "experiment_id": storefront.experiment_name,
            "ds": storefront.start + dt.timedelta(days=day),
            "group_id": arm,
            "n_daily": count,
            "n_cumulative": count,
        }
        for day in (0, 1)
        for arm, count in (
            (storefront.control_group, 0),
            (storefront.treatment_group, 10 if day else 0),
        )
    )
    html = render_health(dataclasses.replace(storefront, allocation_history=history)).text
    # Wilson bounds for 0/10 and 10/10; Wald intervals would collapse.
    assert "27.8%" in html
    assert "72.2%" in html


# Ownership: caller data cannot mutate prepared state.


def test_configuration_copies_caller_mappings() -> None:
    allocation = {"control": 0.5, "treatment": 0.5}
    provenance = {"Seed": "42"}
    config = DashboardConfig(expected_allocation=allocation, provenance=provenance)

    allocation["treatment"] = 9.0
    allocation["extra"] = 1.0
    provenance["Seed"] = "tampered"

    assert config.expected_allocation == {"control": 0.5, "treatment": 0.5}
    assert config.provenance == {"Seed": "42"}


# Configuration and structure refusals, before any result is computed.


@pytest.mark.parametrize(
    "allocation",
    [
        {"control": 0.5},
        {"control": 0.4, "treatment": 0.4, "holdback": 0.2},
        {"control": 0.5, "treatment": 0.0},
        {"control": 0.5, "treatment": -0.5},
        {"control": 0.5, "treatment": float("nan")},
        {"control": 0.5, "treatment": float("inf")},
        {"": 0.5, "treatment": 0.5},
        {"control": 0.5, "treatment": "half"},
    ],
)
def test_configuration_refuses_unusable_allocation(allocation: dict[str, Any]) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        DashboardConfig(expected_allocation=allocation)
    assert caught.value.code == "dashboard.invalid_config"


def test_configuration_refuses_a_blank_title() -> None:
    with pytest.raises(InvalidRequestError) as caught:
        DashboardConfig(expected_allocation={"a": 1.0, "b": 1.0}, title="  ")
    assert caught.value.code == "dashboard.invalid_config"


def test_snapshot_requires_exactly_one_allocation_outcome(
    pricing: DashboardSnapshot,
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        dataclasses.replace(pricing, allocation=None, allocation_refusal=None)
    assert caught.value.code == "dashboard.invalid_snapshot"
    with pytest.raises(InvalidRequestError):
        dataclasses.replace(pricing, allocation_refusal=("some.code", "some reason"))


# Health rendering against real SRM result shapes.


def _with_allocation(snapshot: DashboardSnapshot, **changes: Any) -> DashboardSnapshot:
    allocation = snapshot.allocation
    assert allocation is not None
    return dataclasses.replace(snapshot, allocation=allocation.model_copy(update=changes))


def test_cluster_srm_render_uses_cluster_grain_counts_and_labels(
    pricing: DashboardSnapshot,
) -> None:
    snapshot = _with_allocation(
        pricing,
        grain="cluster",
        observed={"baseline": 7, "candidate": 5},
        unit_counts={"baseline": 70, "candidate": 60},
    )
    assert snapshot.allocation is not None
    header = render_header(snapshot).text
    health = render_health(snapshot).text
    assert "Enrolled clusters" in header
    assert "Assigned clusters" in header
    assert "130 member units" in header
    assert "Enrolled clusters" in health
    assert "58.3%" in health


def test_health_keeps_assignment_warnings_when_no_mismatch_is_detected(
    pricing: DashboardSnapshot,
) -> None:
    snapshot = _with_allocation(
        pricing,
        is_srm=False,
        unassigned_units=37,
        mixed_assignment_units=12,
        low_expected_count=True,
        min_expected_count=3.5,
    )
    html = render_health(snapshot).text
    assert "No allocation issues detected" in html
    assert "37" in html
    assert "not assigned to any arm" in html
    assert "more than one arm" in html
    assert "smallest expected arm count" in html


def test_health_reads_the_always_valid_verdict_not_its_fixed_p_value(
    pricing: DashboardSnapshot,
) -> None:
    snapshot = _with_allocation(
        pricing,
        inference="always_valid",
        is_srm=True,
        fixed_p_value=0.97,
        log_e_value=6.5,
    )
    html = render_health(snapshot).text
    assert "Sample ratio mismatch detected" in html
    assert "Always-valid evidence" in html
    assert "log e-value 6.5" in html
    assert "0.97" not in html


def test_health_reports_a_fixed_horizon_check_as_a_p_value(
    pricing: DashboardSnapshot,
) -> None:
    snapshot = _with_allocation(
        pricing, inference="fixed", is_srm=False, fixed_p_value=0.42, log_e_value=None
    )
    html = render_health(snapshot).text
    assert "chi-square evidence (p = 0.42)" in html
    assert "e-value" not in html


def test_health_shows_a_refused_check_as_a_refusal_not_a_pass(
    pricing: DashboardSnapshot,
) -> None:
    snapshot = dataclasses.replace(
        pricing,
        allocation=None,
        allocation_refusal=("facade.analysis.invalid_population", "population is unsupported"),
    )
    html = render_health(snapshot).text
    assert "population is unsupported" in html
    assert "not a passing check" in html
    assert "No allocation issues detected" not in html


def test_health_separates_the_allocation_alpha_from_interval_levels(
    pricing: DashboardSnapshot,
) -> None:
    html = render_health(pricing).text
    assert "α = 0.001" in html


# Results and metric details.


def test_results_table_shows_only_core_readout_columns(pricing: DashboardSnapshot) -> None:
    html = render_results(pricing).text
    headers = {
        re.sub(r"<[^>]+>", "", header).strip()
        for header in re.findall(r"<th[^>]*>(.*?)</th>", html, flags=re.DOTALL)
    }
    allowed_columns = {
        "Lift (absolute)",
        "Lift %",
        "Lift Plot",
        "Failure",
        "Discovery",
        "Posterior chance to beat",
        "Posterior risk if shipped",
        "Posterior P(favorable)",
    }
    assert {"Lift %", "Lift Plot"} <= headers
    structural = {"group_id", "▾Primary"}
    assert headers - structural <= allowed_columns | {
        header for header in headers if "Declared plan" in header
    }
    assert "Numerical qualification" not in html
    assert "Selection qualification" not in html


def test_results_table_keeps_absolute_scale_values(pricing: DashboardSnapshot) -> None:
    rows = tuple(
        {
            **dict(row),
            "value_scale": "absolute",
            "lift": 23.456,
            "lower": 12.345,
            "higher": 34.567,
        }
        if row["metric"] == pricing.primary_metric
        else row
        for row in pricing.readout_rows
    )
    html = render_results(dataclasses.replace(pricing, readout_rows=rows)).text
    assert "Lift (absolute)" in html
    assert "23.46" in html


def test_results_keep_selection_qualification_as_a_footnote(pricing: DashboardSnapshot) -> None:
    qualification = (
        "Benjamini-Hochberg family selection at q=0.05; retained family size 12; "
        "FDR qualification requires independence or positive dependence"
    )
    rows = tuple(
        {
            **dict(row),
            "discovery": True,
            "selection_qualification": qualification,
            "family_q": 0.05,
            "family_size": 12,
        }
        if row["metric"] == pricing.primary_metric
        else row
        for row in pricing.readout_rows
    )
    html = render_results(dataclasses.replace(pricing, readout_rows=rows)).text
    assert "Selection qualification" not in html
    assert "Selection evidence for" in html
    assert pricing.primary_metric in html
    assert qualification in html


def test_results_show_available_prior_advisory_columns(pricing: DashboardSnapshot) -> None:
    rows = tuple(
        {
            **dict(row),
            "posterior_available": True,
            "posterior_chance_to_beat": 0.73,
            "posterior_risk_if_shipped": 0.27,
        }
        for row in pricing.readout_rows
    )
    snapshot = dataclasses.replace(pricing, readout_rows=rows)
    html = render_results(snapshot).text
    assert "Posterior chance to beat" in html
    assert "Posterior risk if shipped" in html


def test_results_render_every_declared_role_group(storefront: DashboardSnapshot) -> None:
    html = render_results(storefront).text
    groups = (
        ("Primary", "checkout_conversion"),
        ("Secondaries", "revenue_per_user"),
        ("Guardrails", "session_seconds"),
    )
    headings = {
        re.sub(r"<[^>]+>", "", body).strip().removeprefix("▾").strip(): group
        for group, body in re.findall(
            r'<tr\b[^>]*data-ct-group="([^"]+)"[^>]*>(.*?)</tr>', html, re.DOTALL
        )
    }
    members = re.findall(
        r'<tr\b[^>]*data-ct-group-member="([^"]+)"[^>]*>(.*?)</tr>', html, re.DOTALL
    )
    assert set(headings) == {role for role, _ in groups}
    for role, metric in groups:
        section = "".join(body for group, body in members if group == headings[role])
        assert metric in section
        assert all(other not in section for _, other in groups if other != metric)


def test_metric_details_change_nothing_about_the_snapshot(
    storefront: DashboardSnapshot,
) -> None:
    before = [e.require_lift().value for e in storefront.estimates]
    for metric in (m.name for m in storefront.metrics):
        render_metric_details(storefront, metric=metric)
    assert [e.require_lift().value for e in storefront.estimates] == before


def test_metric_details_refuse_an_undeclared_metric(storefront: DashboardSnapshot) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        render_metric_details(storefront, metric="not_a_metric")
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["metric"] == "not_a_metric"
    assert "checkout_conversion" in caught.value.context["declared"]  # ty: ignore[unsupported-operator]


def test_the_second_experiment_uses_the_same_result_sections(
    pricing: DashboardSnapshot,
) -> None:
    """No breakouts, no secondaries, no guardrails: still an ordinary case."""
    results = render_results(pricing).text

    assert "purchase_rate" in results
    # Provenance belongs to the caller's configuration, never a default.
    assert "Synthetic" not in render_header(pricing).text


# Explore tables and empty states.


@pytest.mark.parametrize("view", ["cumulative_lift", "daily_values", "cumulative_values"])
def test_all_temporal_views_render_every_declared_metric_in_one_table(
    storefront: DashboardSnapshot, view: ExploreView
) -> None:
    data = temporal_rows(storefront, view)
    expected = {metric.name for metric in storefront.metrics}
    assert {row.metric for row in data} == expected
    html = render_explore(storefront, data, metric=None, view=view).text
    for metric in expected:
        assert metric in html


def test_all_metric_daily_table_keeps_cohort_and_calendar_rows_distinct(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyMetricValues

    data = temporal_rows(storefront, "daily_values")
    assert isinstance(data, DailyMetricValues)
    mixed = DailyMetricValues(
        row.model_copy(update={"ds_basis": "cohort"})
        if row.metric == storefront.primary_metric
        else row
        for row in data
    )
    html = render_explore(storefront, mixed, metric=None, view="daily_values").text
    assert "Exposure cohort" in html
    assert "Observation date" in html


def test_render_explore_segments_show_a_truthful_empty_state_without_breakouts(
    pricing: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import BreakoutEstimates

    html = render_explore(
        pricing, BreakoutEstimates(), metric="purchase_rate", view="segments"
    ).text
    assert "No declared breakouts" in html
    assert "<table" not in html


def test_dashboard_tables_escape_warehouse_segment_values_and_arm_names(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    hostile = "<img src=x onerror=alert(1)>"
    results = render_results(
        _with_primary_row(storefront, group_id=storefront.treatment_group + hostile)
    ).text
    segment = BreakoutEstimate(
        metric=storefront.primary_metric,
        group_id=storefront.treatment_group,
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value="US" + hostile,
        lift=Estimate(value=0.1, lb=0.0, ub=0.2, level=0.95),
    )
    segments = render_explore(
        dataclasses.replace(storefront, breakouts=storefront.breakouts or ("country",)),
        BreakoutEstimates([segment]),
        metric=storefront.primary_metric,
        view="segments",
    ).text
    for html in (results, segments):
        assert hostile not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html


@pytest.mark.parametrize(
    "label",
    [
        "[marker](javascript:void%280%29)",
        "[marker](data:text/html,unsafe)",
        "[marker](vbscript:unsafe)",
        "[marker](JaVaScRiPt:void%280%29)",
        "[marker](jav&#x61;script:void%280%29)",
        "{{[marker](javascript:void%280%29)}}",
    ],
)
def test_dashboard_tables_neutralize_dangerous_label_links(
    storefront: DashboardSnapshot, label: str
) -> None:
    import re
    from html.parser import HTMLParser

    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    class Labels(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.targets.append(dict(attrs).get("href") or "")

        def handle_data(self, data):
            self.text.append(data)

    snapshot = _with_primary_row(storefront, group_id=label)
    segment = BreakoutEstimate(
        metric=storefront.primary_metric,
        group_id=storefront.treatment_group,
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=label,
        lift=Estimate(value=0.1, lb=0.0, ub=0.2, level=0.95),
    )
    rendered = (
        render_results(snapshot).text,
        render_explore(
            dataclasses.replace(storefront, breakouts=storefront.breakouts or ("country",)),
            BreakoutEstimates([segment]),
            metric=storefront.primary_metric,
            view="segments",
        ).text,
    )
    for markup in rendered:
        parsed = Labels()
        parsed.feed(markup)
        targets = [re.sub(r"[\x00-\x20\x7f]", "", target).lower() for target in parsed.targets]
        assert not any(
            target.startswith(("javascript:", "data:", "vbscript:")) for target in targets
        )
        assert "marker" in "".join(parsed.text)
    primary_rows = [
        row for row in snapshot.readout_rows if row["metric"] == storefront.primary_metric
    ]
    assert all(row["group_id"] == label for row in primary_rows)
    assert segment.lift is not None
    assert segment.dimension_value == label and segment.lift.value == 0.1


# Explore: actual temporal date basis, gap reasons, and monitoring disclosure.


def test_daily_values_report_their_actual_calendar_basis_and_gap_reasons(
    storefront: DashboardSnapshot,
) -> None:
    data = temporal_rows(storefront, "daily_values").filter(
        lambda row: row.metric == storefront.primary_metric
    )
    frame: Any = data.to_frame(backend="pandas")
    assert set(frame["ds_basis"].dropna()) == {"calendar"}
    reasons = set(frame["unavailable"].dropna())
    assert reasons == {"nonpositive_mean"}

    html = render_explore(storefront, data, metric="checkout_conversion", view="daily_values").text
    assert "nonpositive_mean" in html


@pytest.mark.slow
def test_dashboard_explore_reports_unavailable_points_in_its_plot_notes(
    storefront_payload,
) -> None:
    entry = storefront_payload["explore"]["overall"]["checkout_conversion"]["daily_values"]
    notes = " ".join(entry["notes"])
    assert "nonpositive_mean" in notes
    assert "nonpositive_mean" not in entry["html"]


# Export: CSV consumer round-trip.


def test_readout_csv_round_trips_every_row_identity_and_value(
    storefront: DashboardSnapshot,
) -> None:
    reader = list(csv.DictReader(io.StringIO(readout_csv(storefront).decode("utf-8"))))
    assert len(reader) == len(storefront.readout_rows) == len(storefront.estimates)
    for csv_row, estimate in zip(reader, storefront.estimates, strict=True):
        assert csv_row["alternative"] == estimate.alternative
        assert csv_row["metric"] == estimate.metric
        if estimate.method_role == "decision":
            # Full precision, not the rounded display figure the primary card shows.
            assert float(csv_row["lift"]) == pytest.approx(
                estimate.require_lift().value, rel=0, abs=1e-12
            )
            assert f"{estimate.require_lift().value:+,.4g}" != csv_row["lift"]


@pytest.mark.slow
def test_readout_csv_preserves_undefined_geometry_without_retained_samples(
    storefront: DashboardSnapshot,
) -> None:
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_reference
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import BootstrapReference
    from tests.estimation.test_winsor_bootstrap import _state

    raw = _state()
    payload = full_procedure_bootstrap_reference(raw, "C", "T").model_dump()
    # Retained roots force a representationally undefined upper endpoint.
    payload["log_relative"]["roots"] = (-10000.0,) * 50 + (1.0,) * 1949
    reference = BootstrapReference.model_validate(payload)
    estimate = estimate_winsor_lift(raw, "C", "T", reference=reference)
    snapshot = dataclasses.replace(
        storefront, estimates=(estimate,), readout_rows=enriched_rows((estimate,))
    )
    row = next(csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8"))))
    region = json.loads(row["confidence_set"])
    assert "raw" not in region
    assert "reference" not in region
    assert region["relative"]["lower"]["value"] == pytest.approx(-0.07150606498193245, abs=1e-12)
    assert region["relative"]["upper"]["status"] == "undefined"
    assert region["relative"]["upper"]["value"] is None
    assert region["relative"]["upper"]["reason"] == "endpoint_unrepresentable"
    assert row["higher"] == ""
    assert float(row["lift"]) == pytest.approx(0.479245283018868, abs=1e-12)


def _guardrail_snapshot(
    snapshot: DashboardSnapshot, *, alternative: str, direction: str, lb: float, ub: float
) -> DashboardSnapshot:
    """The snapshot's guardrail replaced by one whose interval is adverse."""
    from increment.estimation.results import Estimate
    from increment.tables import estimates_to_readout

    guardrail = next(e for e in snapshot.estimates if e.role == "guardrail")
    replaced = guardrail.model_copy(
        update={
            "alternative": alternative,
            "preferred_direction": direction,
            "lift": Estimate(value=(lb + ub) / 2, lb=lb, ub=ub, level=0.9),
        }
    )
    estimates = tuple(replaced if e is guardrail else e for e in snapshot.estimates)
    rows = estimates_to_readout(list(estimates))
    return dataclasses.replace(
        snapshot,
        estimates=estimates,
        readout_rows=tuple(
            {**row, "alternative": est.alternative}
            for row, est in zip(rows, estimates, strict=True)
        ),
    )


@pytest.mark.slow
def test_readout_csv_round_trips_an_adverse_guardrail_with_its_tested_alternative(
    storefront: DashboardSnapshot,
) -> None:
    """The exported CSV keeps an adverse guardrail unfavorable, not zeroed or dropped."""
    snapshot = _guardrail_snapshot(
        storefront, alternative="greater", direction="increase", lb=-0.10, ub=-0.02
    )
    reader = list(csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8"))))
    row = next(r for r in reader if r["metric"] == "session_seconds")
    assert row["alternative"] == "greater"
    assert row["preferred_direction"] == "increase"
    assert float(row["lift"]) < 0
    assert float(row["lower"]) < 0
    assert float(row["higher"]) < 0  # the whole interval stays on the adverse side
    assert row["stat_sig"] == "False"


def test_readout_csv_preserves_missing_numbers_beside_available_results(
    storefront: DashboardSnapshot,
) -> None:
    rows = tuple(
        {**row, "lift": None, "lower": None, "higher": None, "excluded": "no_observed_window"}
        if row["metric"] == "session_seconds"
        else row
        for row in storefront.readout_rows
    )
    snapshot = dataclasses.replace(storefront, readout_rows=rows)
    exported = {
        row["metric"]: row
        for row in csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8")))
    }
    missing = exported["session_seconds"]
    assert [missing[key] for key in ("lift", "lower", "higher")] == ["", "", ""]
    assert missing["excluded"] == "no_observed_window"
    available = next(row for row in rows if row["metric"] == storefront.primary_metric)
    assert float(exported[storefront.primary_metric]["lift"]) == available["lift"]


@pytest.mark.parametrize(
    "dangerous",
    (
        "=launch_formula",
        "\tlaunch_formula",
        "\rlaunch_formula",
        "\nlaunch_formula",
        "  =launch_formula",
        "\x00=launch_formula",
        "\u200b=launch_formula",
    ),
)
def test_readout_csv_neutralizes_formula_strings_without_stringifying_numbers(
    storefront: DashboardSnapshot,
    dangerous: str,
) -> None:
    rows = tuple(
        {
            **row,
            "metric": dangerous,
            "lift": -2.5,
            "excluded": "@unsafe",
        }
        if row["metric"] == storefront.primary_metric
        else row
        for row in storefront.readout_rows
    )
    exported = next(
        row
        for row in csv.DictReader(
            io.StringIO(
                readout_csv(dataclasses.replace(storefront, readout_rows=rows)).decode("utf-8")
            )
        )
        if row["metric"] == "'" + dangerous
    )
    assert exported["metric"] == "'" + dangerous
    assert exported["excluded"] == "'@unsafe"
    assert exported["lift"] == "-2.5"


COUNTRY = ("event_log", "country")


@pytest.fixture
def storefront_payload(storefront: DashboardSnapshot):
    from increment.dashboard._app import build_payload

    return build_payload(experiment_for(storefront), snapshot=storefront)


def test_health_status_is_qualified_and_never_a_ship_recommendation(
    storefront: DashboardSnapshot,
) -> None:
    from increment.dashboard._app import health_status

    refused = dataclasses.replace(
        storefront,
        allocation=None,
        allocation_refusal=("dashboard.allocation_not_applicable", "no assignment table"),
        allocation_history=(),
    )
    not_checked = health_status(refused)
    assert not_checked["kind"] == "not_checked"
    assert "dashboard.allocation_not_applicable" in not_checked["detail"]
    assert "no assignment table" in not_checked["detail"]

    assert storefront.allocation is not None
    flagged = health_status(_with_allocation(storefront, mixed_assignment_units=3))
    assert flagged["kind"] == "warning"

    assert health_status(_with_allocation(storefront, is_srm=True))["kind"] == "warning"

    healthy = health_status(storefront)
    assert healthy["kind"] == "healthy"


def _open_sided_lift(metric: str, group: str, segment: str, day: int, value: float) -> Any:
    from increment.breakout.estimates import DailyLiftEstimate
    from increment.estimation.results import Estimate

    return DailyLiftEstimate(
        metric=metric,
        group_id=group,
        method="unadjusted",
        method_role="decision",
        alternative="greater",
        ds=dt.date(2025, 1, 15) + dt.timedelta(days=day),
        lift=Estimate(value=value, lb=value - 0.05, open_side="upper", level=0.95, alpha=0.05),
        dimension="country",
        dimension_value=segment,
        source="event_log",
        sampling_available=True,
        reference_kind="normal",
        reference_df=None,
    )


@pytest.mark.slow
def test_document_embeds_untrusted_payload_text_as_inert_json(storefront_payload) -> None:
    from increment.dashboard._app import document

    hostile = "</script><script>alert(1)</script>\u2028&"
    html = document({**storefront_payload, "title": hostile, "description": hostile})
    assert "<!-- DASHBOARD_DATA -->" not in html
    blocks = re.findall(
        r"""<script[^>]*id=["']dashboard-data["'][^>]*>(.*?)</script>""", html, re.S
    )
    assert len(blocks) == 1
    block = re.match(r"(?s)(.*)", blocks[0])
    assert block is not None
    assert "</script" not in block.group(1) and "\u2028" not in block.group(1)
    assert json.loads(block.group(1))["title"] == hostile


@pytest.mark.parametrize("method_role", ["decision", "sensitivity"])
def test_removed_set_column_preserves_disconnected_numeric_bounds(
    storefront: DashboardSnapshot,
    method_role: str,
) -> None:
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    region = relative_confidence_set(JointContrastReference(a=10, c=0, var_a=1, var_c=1, cov_ac=0))
    assert region.geometry == "disconnected"
    snapshot = _with_primary_row(
        storefront,
        lift=None,
        lower=None,
        higher=None,
        relative_confidence_set=region,
        method_role=method_role,
    )
    rendered = render_results(snapshot).text
    assert "∪" in rendered
    for interval in region.intervals:
        for endpoint in interval:
            if endpoint is not None:
                assert f"{endpoint:+.1%}" in rendered


@pytest.mark.parametrize("method_role", ["decision", "sensitivity"])
def test_summary_notes_preserve_disconnected_bounds_without_a_point(
    storefront: DashboardSnapshot,
    method_role: str,
) -> None:
    from increment.dashboard import _app
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    region = relative_confidence_set(JointContrastReference(a=10, c=0, var_a=1, var_c=1, cov_ac=0))
    snapshot = _with_primary_row(
        storefront,
        lift=None,
        lower=None,
        higher=None,
        relative_confidence_set=region,
        method_role=method_role,
    )
    notes = " ".join(_app._report_notes(snapshot))
    assert "∪" in notes
    displayed = [float(value) / 100 for value in re.findall(r"([+-]?\d+(?:\.\d+)?)%", notes)]
    for interval in region.intervals:
        for endpoint in interval:
            if endpoint is not None:
                assert any(abs(value - endpoint) <= 0.0005 for value in displayed)


def test_report_notes_keep_binomial_qualification_out_of_readout(
    storefront: DashboardSnapshot,
) -> None:
    from increment.dashboard._app import _report_notes
    from increment.dashboard._format import headline_interval
    from increment.dashboard._html import results_table
    from increment.tables import _binomial_qualification_disclosure

    storefront = _with_primary_row(storefront, binomial_set=binomial_region())
    row = next(row for row in storefront.readout_rows if row["metric"] == storefront.primary_metric)
    qualification = _binomial_qualification_disclosure(row["binomial_set"])
    assert qualification is not None

    notes = "\n".join(_report_notes(storefront))
    headline = headline_interval(row)
    table = results_table(storefront)
    assert notes.count(qualification) == 1
    assert str(row["metric"]) in notes
    assert qualification not in headline
    assert qualification not in table
    assert "Numerical qualification" not in table


def test_summary_notes_attribute_each_captured_level_to_its_own_metric(
    storefront: DashboardSnapshot,
) -> None:
    from increment.dashboard import _app
    from increment.dashboard._data import decision_rows

    rows = decision_rows(storefront)
    assert len({row["level"] for row in rows}) > 1
    summary = _app._report_notes(storefront)[0]
    stated = [(m.start(), m.group()) for m in re.finditer(r"\d+(?:\.\d+)?%", summary)]
    for row in rows:
        metric_at = summary.index(str(row["metric"]))
        preceding = [text for start, text in stated if start < metric_at]
        assert float(preceding[-1].rstrip("%")) / 100 == pytest.approx(
            float(row["level"]), abs=0.00005
        )


def test_summary_levels_distinguish_methods_for_the_same_metric(
    storefront: DashboardSnapshot,
) -> None:
    from increment.dashboard import _app

    decision = storefront.readout_rows[0]
    sensitivity = {
        **decision,
        "method": "cuped" if decision["method"] != "cuped" else "unadjusted",
        "method_role": "sensitivity",
        "level": 0.8,
    }
    snapshot = dataclasses.replace(
        storefront,
        readout_rows=(*storefront.readout_rows, sensitivity),
        estimates=(
            *storefront.estimates,
            storefront.estimates[0].model_copy(
                update={"method": sensitivity["method"], "method_role": "sensitivity"}
            ),
        ),
    )
    summary = _app._report_notes(snapshot)[0]
    levels = [
        (m.start(), float(m.group()[:-1]) / 100) for m in re.finditer(r"\d+(?:\.\d+)?%", summary)
    ]
    for row in (decision, sensitivity):
        method_at = summary.index(str(row["method"]))
        preceding = [value for start, value in levels if start < method_at]
        assert preceding[-1] == pytest.approx(float(row["level"]), abs=0.00005)
    notes = "\n".join(_app._report_notes(snapshot))
    disclosure = render_results(snapshot).text
    for row in (decision, sensitivity):
        attribution = f"{row['metric']} ({row['method']}, {row['method_role']})"
        assert attribution in notes
        assert attribution in disclosure


def test_a_rising_trajectory_of_a_decrease_preferred_metric_is_coloured_adverse(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.estimation.results import Estimate

    model = next(m for m in storefront.metrics if m.preferred_direction == "decrease")
    series = temporal_rows(storefront, "cumulative_lift")
    assert isinstance(series, DailyLiftEstimates)
    rising = DailyLiftEstimates(
        row.model_copy(
            update={
                "lift": Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                "discovery": True,
                "family_q": 0.05,
                "family_procedure": "bh",
                "multiplicity_status": "exploratory_family",
            }
        )
        for row in series
    )
    from increment.tables import estimates_to_readout

    assert estimates_to_readout(rising)[0]["selection_qualification"] is not None
    html = render_explore(storefront, rising, metric=model.name, view="cumulative_lift").text
    assert "Selection evidence for" in html


def test_segmented_temporal_rendering_accepts_distinct_decision_methods(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates

    data = temporal_rows(storefront, "cumulative_lift", segmented=True)
    assert isinstance(data, DailyLiftEstimates)
    mixed = DailyLiftEstimates(
        row.model_copy(update={"method": "cuped"}) if row.metric == "revenue_per_user" else row
        for row in data
    )
    rendered = render_explore(storefront, mixed, metric=None, view="cumulative_lift").text
    assert "revenue_per_user" in rendered and storefront.primary_metric in rendered
    assert "US" in rendered and "CA" in rendered


def test_one_sided_bound_lines_keep_unavailable_dates_as_gaps(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates

    rows = DailyLiftEstimates(
        _open_sided_lift(
            storefront.primary_metric, storefront.treatment_group, "US", day, 0.02 * day
        )
        if day != 3
        else _open_sided_lift(
            storefront.primary_metric, storefront.treatment_group, "US", day, 0.0
        ).model_copy(update={"lift": None})
        for day in range(1, 6)
    )
    rendered = render_explore(
        storefront, rows, metric=storefront.primary_metric, view="cumulative_lift"
    ).text
    polylines = re.findall(r'<polyline[^>]*points="([^"]+)"', rendered)
    assert polylines
    # Each estimate and bound is two separate two-date segments, never a bridge.
    assert all(len(points.split()) == 2 for points in polylines)


@pytest.mark.slow
@pytest.mark.parametrize("surface", ["readout", "report"])
def test_complete_dashboard_retains_adverse_nonrejecting_guardrail(
    storefront: DashboardSnapshot, surface: str
) -> None:
    from increment.dashboard._app import build_payload

    snapshot = _guardrail_snapshot(
        storefront, alternative="greater", direction="increase", lb=-0.10, ub=-0.02
    )
    guardrail = next(row for row in snapshot.readout_rows if row["role"] == "guardrail")
    assert guardrail["stat_sig"] is False
    payload = build_payload(experiment_for(storefront), snapshot=snapshot)
    markup = payload["results"] if surface == "readout" else payload["report"]["results"]
    warnings = re.findall(
        r'<p[^>]*class="[^"]*\binc-dashboard-status--bad\b[^"]*"[^>]*>(.*?)</p>',
        markup,
        re.DOTALL,
    )
    assert any(str(guardrail["metric"]) in warning for warning in warnings)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_report_retains_directional_fieller_set_beside_central_interval(
    storefront: DashboardSnapshot, alternative: str
) -> None:
    from increment.dashboard._app import _report_notes
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    reference = JointContrastReference(a=0.2, c=1, var_a=0.04, var_c=0.04, cov_ac=0)
    central = relative_confidence_set(reference, alpha=0.05)
    directional = relative_confidence_set(reference, alpha=0.05, alternative=alternative)
    lower, higher = central.intervals[0]
    snapshot = _with_primary_row(
        storefront,
        lift=0.2,
        lower=lower,
        higher=higher,
        level=0.95,
        alternative=alternative,
        relative_confidence_set=directional,
        confidence_set=None,
        binomial_set=None,
    )
    notes = " ".join(_report_notes(snapshot))
    displayed = [
        float(value) / 100
        for value in re.findall(r"([+−-]?\d+(?:\.\d+)?)%", notes.replace("−", "-"))
    ]
    for bounds in directional.intervals:
        for endpoint in bounds:
            if endpoint is not None:
                assert any(abs(value - endpoint) <= 0.0005 for value in displayed)
    assert "∞" in notes


pytestmark = pytest.mark.integration
