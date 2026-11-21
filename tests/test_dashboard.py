"""Dashboard sections and exports consume constructed snapshots."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from typing import Any

import pytest

from increment.errors import CodedError, InvalidRequestError

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import (
    DashboardConfig,
    DashboardSnapshot,
    ExploreView,
    readout_csv,
    render_dashboard,
    render_explore,
    render_header,
    render_health,
    render_metric_details,
    render_results,
)
from increment.dashboard._app import _render_entry, build_payload, document
from increment.dashboard._data import enriched_rows
from increment.dashboard._html import temporal_figure
from tests.dashboard_snapshots import (
    binomial_region,
    experiment_for,
    pricing_snapshot,
    temporal_rows,
)
from tests.dashboard_snapshots import snapshot as make_snapshot


@pytest.fixture(scope="module")
def storefront() -> DashboardSnapshot:
    return make_snapshot()


@pytest.fixture(scope="module")
def pricing() -> DashboardSnapshot:
    return pricing_snapshot()


def _rendered_payload(snapshot: DashboardSnapshot) -> dict[str, Any]:
    page = render_dashboard(experiment_for(snapshot), snapshot=snapshot).widget.get_state()[
        "document"
    ]
    (embedded,) = re.findall(
        r"""<script[^>]*id=["']dashboard-data["'][^>]*>(.*?)</script>""", page, re.S
    )
    return json.loads(embedded)


# Readout qualifications and allocation.


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

    text = render_results(snapshot).text
    assert "the allocation scheme is not declared" in text
    assert "integrity.allocation_scheme_missing" in text
    assert "no allocation declared" not in text


@pytest.mark.integration
def test_missing_required_guardrail_is_named_before_optional_details(storefront):
    collection = storefront.readout_collection
    assert collection is not None
    selected = collection.filter(lambda row: row.metric != "session_seconds")
    snapshot = dataclasses.replace(
        storefront,
        estimates=tuple(selected),
        readout_rows=enriched_rows(selected),
        readout_collection=selected,
        explore={},
    )
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    visible = payload["results"].split("<details", 1)[0]
    assert "session_seconds" in visible
    assert "session_seconds" in payload["report"]["healthHtml"]


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
    text = render_results(snapshot).text
    assert "integrity.allocation_missing_or_nonconstant" in text
    assert "declared target allocation weights" in text
    assert "allocation counts are missing or nonconstant" not in text
    # The header leaves the qualifications to the Readout.
    assert "integrity.allocation_missing_or_nonconstant" not in render_header(snapshot).text


def test_renders_expected_shares_from_configuration(pricing: DashboardSnapshot) -> None:
    html = render_health(pricing).text
    assert "70.0%" in html
    assert "30.0%" in html
    assert "baseline" in html
    assert "candidate" in html


def test_dashboard_allocation_mismatch_stays_visible_before_the_readout(
    storefront: DashboardSnapshot,
) -> None:
    assert storefront.allocation is not None
    expected = {storefront.control_group: 0.25, storefront.treatment_group: 0.75}
    snapshot = dataclasses.replace(
        storefront,
        config=dataclasses.replace(storefront.config, expected_allocation=expected),
        allocation=storefront.allocation.model_copy(
            update={"expected": expected, "is_srm": True, "log_e_value": 100.0}
        ),
    )
    payload = _rendered_payload(snapshot)
    for rendered in (
        render_results(snapshot).text,
        payload["results"],
        payload["report"]["healthHtml"],
    ):
        visible = rendered.split("<details", 1)[0]
        warnings = re.findall(
            r'<p class="[^"]*inc-dashboard-status--warn[^"]*">(.*?)</p>', visible, re.S
        )
        assert any(
            "allocation" in warning.lower() and "assigned" in warning for warning in warnings
        )


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
    ("direction", "lower", "higher", "tone", "label"),
    [
        ("increase", 0.02, 0.18, "favorable", "Improvement detected"),
        ("increase", -0.18, -0.02, "unfavorable", "Decline detected"),
        ("decrease", -0.18, -0.02, "favorable", "Improvement detected"),
        ("decrease", 0.02, 0.18, "unfavorable", "Decline detected"),
    ],
)
def test_primary_headline_colors_significant_results_by_declared_direction(
    storefront: DashboardSnapshot,
    direction: str,
    lower: float,
    higher: float,
    tone: str,
    label: str,
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
    assert f">{label}</span>" in html


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

    assert "inc-dashboard-primary--detected" in html
    assert ">Difference detected</span>" in html


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


def test_results_table_keeps_absolute_scale_values(pricing: DashboardSnapshot) -> None:
    rows = tuple(
        {
            **dict(row),
            "value_scale": "absolute",
            "lift": 23.456,
            "lower": 12.345,
            "higher": 34.567,
            "note": "Absolute estimate needs interpretation.",
        }
        if row["metric"] == pricing.primary_metric
        else row
        for row in pricing.readout_rows
    )
    html = render_results(dataclasses.replace(pricing, readout_rows=rows)).text
    assert "Lift (absolute)" in html
    assert "23.46" in html
    value_cells = re.findall(
        r'<td\b[^>]*class="[^"]*\bgt_row\b[^"]*"[^>]*>(.*?)</td>',
        html.split("<tfoot>", 1)[0],
        re.DOTALL,
    )
    assert "<sup>" in next(cell for cell in value_cells if "23.46" in cell)


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
            **({"binomial_set": binomial_region()} if row["role"] == "secondary" else {}),
        }
        if row["metric"] == pricing.primary_metric or row["role"] == "secondary"
        else row
        for row in pricing.readout_rows
    )
    html = render_results(dataclasses.replace(pricing, readout_rows=rows)).text
    headers = {
        re.sub(r"<[^>]+>", "", header).strip()
        for header in re.findall(
            r'<th\b[^>]*class="[^"]*\bgt_col_heading\b[^"]*"[^>]*>(.*?)</th>',
            html,
            flags=re.DOTALL,
        )
    }
    assert {"Lift %", "Lift Plot"} <= headers
    assert "Selection evidence for" in html
    assert pricing.primary_metric in html
    assert qualification in html


@pytest.mark.integration
def test_result_caveats_keep_row_identity_in_readout_report_and_inspector(
    storefront: DashboardSnapshot,
) -> None:
    from html import unescape

    primary_index = next(
        index
        for index, row in enumerate(storefront.readout_rows)
        if row["metric"] == storefront.primary_metric
    )
    original = storefront.readout_rows[primary_index]
    primary_estimate = storefront.estimates[primary_index]
    identities = (
        ("arm&one", "welch", "decision"),
        ("arm-two", "cuped", "sensitivity"),
    )
    notes = {
        "arm&one": "Captured note for arm one.",
        "arm-two": "Captured note for arm two.",
    }
    caveated_rows = tuple(
        {
            **dict(original),
            "group_id": arm,
            "method": method,
            "method_role": role,
            "note": note,
            **({"binomial_set": binomial_region()} if arm == "arm&one" else {}),
        }
        for (arm, method, role), note in zip(identities, notes.values(), strict=True)
    )
    caveated_estimates = tuple(
        primary_estimate.model_copy(
            update={"group_id": arm, "method": method, "method_role": role, "note": note}
        )
        for (arm, method, role), note in zip(identities, notes.values(), strict=True)
    )
    rows = list(storefront.readout_rows)
    estimates = list(storefront.estimates)
    rows[primary_index] = caveated_rows[0]
    rows.append(caveated_rows[1])
    estimates[primary_index] = caveated_estimates[0]
    estimates.append(caveated_estimates[1])
    snapshot = dataclasses.replace(
        storefront,
        readout_rows=tuple(rows),
        estimates=tuple(estimates),
    )
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    readout = payload["results"]
    report = payload["report"]["results"]
    health_snapshot = dataclasses.replace(snapshot, readout_collection=None)
    health = render_health(health_snapshot).text
    inspector = render_metric_details(snapshot, metric=snapshot.primary_metric).text

    for rendered in (readout, report):
        for note in notes.values():
            assert (
                re.search(
                    rf"<sup>[^<]+</sup>\s*<span[^>]*>{re.escape(note)}</span>",
                    rendered,
                )
                is not None
            )
        assert "(Affected:" not in rendered
        assert "gt_sourcenote" in rendered
        visible_cells = re.findall(
            r'<td\b[^>]*class="[^"]*\bgt_row\b[^"]*"[^>]*>(.*?)</td>',
            rendered.split("<tfoot>", 1)[0],
            re.DOTALL,
        )
        expected_markers = set(re.findall(r"<sup>([^<]+)</sup>", rendered.split("<tfoot>", 1)[1]))
        visible_markers = {
            marker
            for cell in visible_cells
            for group in re.findall(r"<sup>([^<]+)</sup>", cell)
            for marker in group.split(", ")
        }
        assert visible_markers == expected_markers
    for arm, note in notes.items():
        assert note in inspector
        assert f"Arm {arm}" in unescape(inspector)
    assert "welch" in inspector and "cuped" in inspector
    advanced = re.search(
        r"<details[^>]*><summary>Advanced calculation record</summary>(.*?)</details>",
        inspector,
        re.DOTALL,
    )
    assert advanced is not None
    assert "open" not in advanced.group(0).split(">", 1)[0]
    assert "120/600" in advanced.group(1) and "132/600" in advanced.group(1)
    assert "arm&one" in unescape(advanced.group(1))
    for note in notes.values():
        assert note in health


@pytest.mark.integration
def test_inspector_keeps_zero_control_inputs_for_a_sensitivity_result(
    storefront: DashboardSnapshot,
) -> None:
    primary_index = next(
        index
        for index, row in enumerate(storefront.readout_rows)
        if row["metric"] == storefront.primary_metric
    )
    original = storefront.readout_rows[primary_index]
    sensitivity = {
        **original,
        "method": "cuped" if original["method"] != "cuped" else "unadjusted",
        "method_role": "sensitivity",
        "binomial_set": binomial_region().model_copy(
            update={"x_c": 0, "n_c": 610, "x_t": 13, "n_t": 620}
        ),
    }
    snapshot = dataclasses.replace(
        storefront,
        readout_rows=(*storefront.readout_rows, sensitivity),
        estimates=(
            *storefront.estimates,
            storefront.estimates[primary_index].model_copy(
                update={
                    "method": sensitivity["method"],
                    "method_role": "sensitivity",
                    "binomial_set": sensitivity["binomial_set"],
                }
            ),
        ),
    )

    inspector = render_metric_details(snapshot, metric=snapshot.primary_metric).text

    assert "0/610" in inspector and "13/620" in inspector


@pytest.mark.integration
def test_selected_result_keeps_zero_control_calculation_inputs(
    storefront: DashboardSnapshot,
) -> None:

    region = binomial_region().model_copy(update={"x_c": 0, "n_c": 610, "x_t": 13, "n_t": 620})
    estimates = tuple(
        row.model_copy(update={"binomial_set": region})
        if row.metric == storefront.primary_metric
        else row
        for row in storefront.estimates
    )
    assert storefront.readout_collection is not None
    collection = type(storefront.readout_collection)(
        estimates, metadata=storefront.readout_collection.metadata
    )
    snapshot = dataclasses.replace(
        storefront,
        estimates=estimates,
        readout_rows=enriched_rows(collection),
        readout_collection=collection,
        explore={},
    )
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    html = next(
        reference["readingHtml"]
        for reference in payload["references"]["results"].values()
        if reference["cell"]["metric"] == storefront.primary_metric
        and reference["cell"]["ds"] is None
    )

    assert "0/610" in html and "13/620" in html


@pytest.mark.integration
def test_report_readout_keeps_non_table_caveats(storefront: DashboardSnapshot) -> None:
    snapshot = _with_primary_row(
        storefront,
        **{
            "lift": None,
            "lower": None,
            "higher": None,
            "note": "Source caveat accompanies an unavailable estimate.",
            "unavailable": "engine declined this row",
            "excluded": "outside the declared population",
        },
    )
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    rendered = payload["results"] + payload["report"]["results"]
    assert "engine declined this row" in rendered
    assert "outside the declared population" in rendered


@pytest.mark.integration
@pytest.mark.parametrize("empty_readout", [False, True], ids=["no-limitations", "empty-readout"])
def test_report_omits_empty_limitations_fragment(
    pricing: DashboardSnapshot, empty_readout: bool
) -> None:
    snapshot = (
        dataclasses.replace(
            pricing,
            estimates=(),
            readout_rows=(),
            readout_collection=None,
            explore={},
        )
        if empty_readout
        else pricing
    )

    assert _rendered_payload(snapshot)["report"]["limitationsHtml"] == ""


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
    headers = {
        re.sub(r"<[^>]+>", "", header).strip()
        for header in re.findall(r"<th[^>]*>(.*?)</th>", html, flags=re.DOTALL)
    }
    assert "Interval" not in headers
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
    source_id = storefront.readout_rows[0]["source_snapshot_id"]
    assert isinstance(source_id, str)
    assert source_id in html
    unfolded = re.sub(r"<details\b.*?</details>", "", html, flags=re.DOTALL)
    assert source_id not in unfolded


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


@pytest.mark.integration
@pytest.mark.parametrize("view", ["cumulative_lift", "daily_values", "cumulative_values"])
def test_all_temporal_views_render_every_declared_metric(
    storefront: DashboardSnapshot, view: ExploreView
) -> None:
    data = temporal_rows(storefront, view)
    expected = {metric.name for metric in storefront.metrics}
    assert {row.metric for row in data} == expected
    if view == "cumulative_lift":
        data = type(data)(
            row.model_copy(update={"ds": row.ds + dt.timedelta(days=3)})
            if row.metric == storefront.primary_metric
            else row
            for row in data
        )
    html = render_explore(storefront, data, metric=None, view=view).text
    tables = re.findall(r'<table\b[^>]*\bclass="[^"]*\bgt_table\b.*?</table>', html, re.S)
    assert len(tables) == 1
    table = tables[0]
    body = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", table, re.S)
    assert body is not None
    assert len(re.findall(r"<tr\b", body.group(1))) == len(expected)
    for metric in expected:
        assert metric.replace("_", " ").title() in body.group(1)
    assert storefront.treatment_group in html
    if view != "cumulative_lift":
        assert storefront.control_group in table
        strokes = set(re.findall(r'<polyline\b[^>]*stroke="([^"]+)"', table))
        assert len(strokes) == 2
    if view == "cumulative_lift":
        endpoints = [
            float(points.split()[-1].split(",")[0])
            for points in re.findall(r'<polyline[^>]*points="([^"]+)"', html)
        ]
        assert max(endpoints) - min(endpoints) > 100


@pytest.mark.integration
def test_whole_experiment_payload_contains_core_metric_packets(
    storefront_payload: dict[str, Any], storefront: DashboardSnapshot
) -> None:
    view_keys = {
        "cumulative_lift",
        "cumulative_lift_complete",
        "cumulative_values",
        "cumulative_values_complete",
        "daily_values",
    }
    core_metrics = {metric.name for metric in storefront.metrics}
    for payload in storefront_payload["populationViews"].values():
        whole = payload["wholeExperiment"]
        assert set(whole) == view_keys
        for packet in whole.values():
            assert {"html", "caption", "notes", "pointCount"} <= set(packet)
            assert all(
                metric.replace("_", " ").title() in packet["html"] for metric in core_metrics
            )


@pytest.mark.integration
def test_whole_experiment_keeps_available_metrics_and_attributes_refusals(
    storefront: DashboardSnapshot,
) -> None:
    capture_type = type(storefront.explore[("assigned", "daily_values", None, False, None)])

    missing = storefront.metrics[-1].name
    empty = storefront.metrics[0].name
    assert empty != missing
    view = "daily_values"
    batch_key = ("assigned", view, None, False, None)
    missing_key = ("assigned", view, missing, False, None)
    empty_key = ("assigned", view, empty, False, None)
    refusal = CodedError(
        "captured metric refusal",
        code="explore.metric_refused",
        context={"metric": missing, "snapshot": "pinned", "details": {"capture": "pinned"}},
    )
    empty_rows = type(temporal_rows(storefront, view))()
    snapshot = dataclasses.replace(
        storefront,
        explore={
            **storefront.explore,
            batch_key: capture_type.refused(
                batch_key,
                CodedError(
                    "whole-experiment batch refusal",
                    code="explore.batch_refused",
                    context={"scope": "whole experiment"},
                ),
            ),
            missing_key: capture_type.refused(missing_key, refusal),
            empty_key: capture_type.answered(empty_key, empty_rows, collection=type(empty_rows)),
        },
    )
    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    packet = payload["wholeExperiment"]["daily_values"]
    html = packet["html"]
    assert missing.replace("_", " ").title() in html
    assert empty.replace("_", " ").title() in html
    for metric in storefront.metrics[1:-1]:
        assert metric.name.replace("_", " ").title() in html
    assert "explore.metric_refused" in html
    assert "snapshot" in html and "pinned" in html
    focused = payload["explore"]["overall"][missing]["daily_values"]
    assert "explore.metric_refused" in focused["caption"]
    page = document(payload)
    match = re.search(
        r'<script type="application/json" id="dashboard-data">(.*?)</script>', page, re.S
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    retained = next(
        item
        for item in embedded["references"]["refusals"].values()
        if item["code"] == "explore.metric_refused"
    )
    assert retained["context"]["details"]["capture"] == "pinned"


@pytest.mark.integration
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
    tables = re.findall(r'<table\b[^>]*\bclass="[^"]*\bgt_table\b.*?</table>', html, re.S)
    assert len(tables) == 1
    body = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", tables[0], re.S)
    assert body is not None
    assert f"{storefront.primary_metric.replace('_', ' ').title()} · Exposure cohort" in body.group(
        1
    )
    for metric in storefront.metrics:
        if metric.name != storefront.primary_metric:
            assert f"{metric.name.replace('_', ' ').title()} · Observation date" in body.group(1)
    incompatible = DailyMetricValues(
        row.model_copy(update={"ds_basis": "cohort"}) if index == 0 else row
        for index, row in enumerate(data)
    )
    refused = render_explore(storefront, incompatible, metric=None, view="daily_values").text
    assert "cannot share one axis" in refused
    assert "<svg" not in refused


@pytest.mark.integration
def test_whole_metric_table_preserves_names_with_colliding_display_labels(
    storefront: DashboardSnapshot,
) -> None:
    originals = storefront.metrics[:2]
    names = ("revenue_per_user", "Revenue_Per_User")
    renamed = dict(zip((model.name for model in originals), names, strict=True))
    snapshot = dataclasses.replace(
        storefront,
        metrics=tuple(
            model.model_copy(update={"name": renamed.get(model.name, model.name)})
            for model in storefront.metrics
        ),
        primary_metric=names[0],
        estimates=tuple(
            estimate.model_copy(update={"metric": renamed.get(estimate.metric, estimate.metric)})
            for estimate in storefront.estimates
        ),
        readout_rows=tuple(
            {**row, "metric": renamed.get(row["metric"], row["metric"])}
            for row in storefront.readout_rows
        ),
        group_data=tuple(
            dataclasses.replace(item, metric=renamed.get(item.metric, item.metric))
            for item in storefront.group_data
        ),
    )
    data = temporal_rows(storefront, "daily_values")
    data = type(data)(
        row.model_copy(update={"metric": renamed[row.metric]})
        for row in data
        if row.metric in renamed
    )
    html = render_explore(snapshot, data, metric=None, view="daily_values").text
    body = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", html, re.S)
    assert body is not None
    rows = re.findall(r"<tr\b.*?</tr>", body.group(1), re.S)
    assert len(rows) == 2
    for name in names:
        row = next(row for row in rows if name in row)
        assert len(re.findall(r"<polyline\b", row)) == 2


@pytest.mark.integration
def test_all_metric_segmented_trajectories_keep_segment_identity(
    storefront: DashboardSnapshot,
) -> None:
    data = temporal_rows(storefront, "daily_values")
    segments = ("north", "south")
    segmented = type(data)(
        row.model_copy(update={"dimension": "region", "dimension_value": segment})
        for row in data
        for segment in segments
    )
    html = render_explore(storefront, segmented, metric=None, view="daily_values").text
    tables = re.findall(r'<table\b[^>]*\bclass="[^"]*\bgt_table\b.*?</table>', html, re.S)
    assert len(tables) == len(storefront.metrics)
    for table in tables:
        body = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", table, re.S)
        assert body is not None
        rows = re.findall(r"<tr\b.*?</tr>", body.group(1), re.S)
        assert len(rows) == len(segments)
        for segment in segments:
            row = next(row for row in rows if segment in row)
            assert len(re.findall(r"<polyline\b", row)) == 2


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
    from html.parser import HTMLParser

    from increment.breakout.estimates import BreakoutEstimate, BreakoutEstimates
    from increment.estimation.results import Estimate

    class Cells(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells: list[str] = []
            self.links: list[str] = []
            self.current: list[str] | None = None

        def handle_starttag(self, tag, attrs):
            if tag in ("td", "th"):
                self.current = []
            elif tag == "a" and self.current is not None:
                self.links.append(dict(attrs).get("href") or "")

        def handle_data(self, data):
            if self.current is not None:
                self.current.append(data)

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.current is not None:
                self.cells.append("".join(self.current).strip())
                self.current = None

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
    from increment.breakout.estimates import DailyLiftEstimates

    renamed = dataclasses.replace(
        storefront,
        metrics=tuple(
            model.model_copy(update={"name": hostile})
            if model.name == storefront.primary_metric
            else model
            for model in storefront.metrics
        ),
        primary_metric=hostile,
        group_data=tuple(
            dataclasses.replace(item, metric=hostile)
            if item.metric == storefront.primary_metric
            else item
            for item in storefront.group_data
        ),
    )
    labels = (
        ("US", storefront.treatment_group),
        ("**US**", f"**{storefront.treatment_group}**"),
        ("[US](https://example.com)", f"[{storefront.treatment_group}](https://example.com)"),
    )
    temporal = render_explore(
        renamed,
        DailyLiftEstimates(
            row.model_copy(
                update={
                    "metric": hostile,
                    "dimension": "country",
                    "dimension_value": series,
                    "group_id": arm,
                }
            )
            for row in temporal_rows(storefront, "cumulative_lift")
            if row.metric == storefront.primary_metric
            for series, arm in labels
        ),
        metric=hostile,
        view="cumulative_lift",
    ).text
    for html in (results, segments, temporal):
        assert hostile not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
    parsed = Cells()
    parsed.feed("".join(re.findall(r"<table\b.*?</table>", temporal, re.S)))
    assert not parsed.links
    for series, arm in labels:
        assert series in parsed.cells and arm in parsed.cells


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
    from increment.estimation.results import Estimate

    data = temporal_rows(storefront, "daily_values").filter(
        lambda row: row.metric == storefront.primary_metric
    )
    data = type(data)(
        row.model_copy(update={"value": Estimate(value=0.3)})
        if row.group_id == storefront.control_group and row.value is not None
        else row
        for row in data
    )
    frame: Any = data.to_frame(backend="pandas")
    assert set(frame["ds_basis"].dropna()) == {"calendar"}
    reasons = set(frame["unavailable"].dropna())
    assert reasons == {"nonpositive_mean"}

    html = render_explore(storefront, data, metric="checkout_conversion", view="daily_values").text
    assert "nonpositive_mean" in html
    assert "30.0%" in html
    assert "[20.0%, 40.0%]" in html
    assert "no interval available" in html
    assert "(−∞, ∞)" not in html


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
    from increment.estimation.results import estimate_winsor_lift
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


@pytest.mark.slow
def test_interaction_report_uses_the_frozen_family_after_pair_evidence_replay(tmp_path):
    from dataclasses import replace

    from examples.interaction_discovery import open_example
    from increment.dashboard import dashboard_html
    from increment.dashboard._app import build_payload
    from increment.errors import CodedError
    from increment.overlap import bind_interaction_scan, capture_interaction_scan

    with open_example(tmp_path, triggered=True) as example:
        experiment = example.source.experiment
        hidden = example.prepare()
        replayed = example.prepare(show_second_metric=True, replay_pairs=True)
        partial = replace(
            example,
            observed_inventory=example.observed_inventory.model_copy(
                update={"coverage": "partial"}
            ),
        ).prepare()
        manifest = example.source.manifest
        first, *other_feeds = manifest.pair_feeds
        wrong_fact = first.fact.model_copy(update={"content_sha256": "a" * 64})
        wrong_feed = first.model_copy(update={"fact": wrong_fact})
        swapped = replace(
            example.source,
            manifest=manifest.model_copy(update={"pair_feeds": (wrong_feed, *other_feeds)}),
        )
        original = capture_interaction_scan(
            example.plan,
            observed_inventory=example.observed_inventory,
            evidence_by_study=example.evidence_by_study,
        )
        with pytest.raises(CodedError) as wrong_published_feed:
            bind_interaction_scan(original, source=swapped)
        assert wrong_published_feed.value.context["reason"] == "pair_evidence_mismatch"

    expected = (-0.077580046, -0.022419954)
    for snapshot in (hidden, replayed):
        scan = snapshot.interactions
        assert scan is not None and scan.plan.family_size == 18
        direct = next(
            row
            for row in scan.rows
            if row.key.metric == "save_rate"
            and row.key.contrast == "interaction"
            and row.key.pair.candidate.experiment_id == "product_cards"
        )
        assert (direct.lower, direct.upper) == pytest.approx(expected, abs=1e-7)

    hidden_payload = build_payload(experiment, snapshot=hidden)
    before = hidden_payload["populationViews"]["assigned"]["interactions"]
    after = build_payload(experiment, snapshot=replayed)["populationViews"]["assigned"][
        "interactions"
    ]
    assert [item["key"] for item in before["metricOptions"]] == ["save_rate"]
    assert {item["key"] for item in after["metricOptions"]} == {"save_rate", "share_rate"}
    assert (
        before["metrics"]["save_rate"]["explanation"]
        == after["metrics"]["save_rate"]["explanation"]
    )
    assert len(before["reversals"]) == 1
    assert before["health"] == []
    tips = next(
        pair["id"]
        for pair in before["metrics"]["save_rate"]["pairs"]
        if pair["label"] == "Search tips"
    )
    assert 'data-inc-cell="11"' in before["metrics"]["save_rate"]["details"][tips]
    assert (
        before["metrics"]["save_rate"]["details"][tips].count('data-inc-reason="empty_cell"') >= 1
    )
    assert "metrics" not in hidden_payload["populationViews"]["triggered"]["interactions"]
    incomplete = build_payload(experiment, snapshot=partial)["populationViews"]["assigned"][
        "interactions"
    ]
    assert incomplete["metrics"]["save_rate"]["explanation"] == ""
    assert not incomplete["reversals"] and incomplete["health"]
    assert tips in dashboard_html(experiment, snapshot=hidden)


@pytest.mark.slow
def test_interaction_binding_rejects_rewritten_summary_with_authentic_feed_receipts(tmp_path):
    from examples.interaction_discovery import open_example
    from increment.errors import CodedError
    from increment.overlap import bind_interaction_scan, capture_interaction_scan

    with open_example(tmp_path) as example:
        authentic = example.evidence_by_study["product_cards"]
        altered = authentic.model_dump()
        altered["cells"][0].update(n_success=0, anchor=0.0, mean_delta=0.0, m2=0.0)
        forged = type(authentic).model_validate(altered)
        candidate = capture_interaction_scan(
            example.plan,
            observed_inventory=example.observed_inventory,
            evidence_by_study={**example.evidence_by_study, "product_cards": forged},
        )
        with pytest.raises(CodedError) as error:
            bind_interaction_scan(candidate, source=example.source)
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "pair_evidence_mismatch"


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
    limitations = _rendered_payload(snapshot)["report"]["limitationsHtml"]
    notes = unescape(re.sub(r"<[^>]+>", " ", limitations))
    assert "∪" in notes
    displayed = [float(value) / 100 for value in re.findall(r"([+-]?\d+(?:\.\d+)?)%", notes)]
    for interval in region.intervals:
        for endpoint in interval:
            if endpoint is not None:
                assert any(abs(value - endpoint) <= 0.0005 for value in displayed)


def test_summary_notes_attribute_each_captured_level_to_its_own_metric(
    storefront: DashboardSnapshot,
) -> None:

    from increment.dashboard._data import decision_rows

    rows = decision_rows(storefront)
    assert len({row["level"] for row in rows}) > 1
    summary = _rendered_payload(storefront)["report"]["methodsHtml"]
    interval_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", summary, re.DOTALL)
    for row in rows:
        matching = [body for body in interval_rows if str(row["metric"]) in body and "%" in body]
        assert any(
            float(value) / 100 == pytest.approx(float(row["level"]), abs=0.00005)
            for body in matching
            for value in re.findall(r"(\d+(?:\.\d+)?)%", body)[:1]
        )


def test_summary_levels_distinguish_methods_for_the_same_metric(
    storefront: DashboardSnapshot,
) -> None:

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
    summary = _rendered_payload(snapshot)["report"]["methodsHtml"]
    interval_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", summary, re.DOTALL)
    for row in (decision, sensitivity):
        assert isinstance(row["level"], float)
        matching = [
            body
            for body in interval_rows
            if str(row["method"]) in body and str(row["method_role"]) in body and "%" in body
        ]
        assert any(
            float(value) / 100 == pytest.approx(float(row["level"]), abs=0.00005)
            for body in matching
            for value in re.findall(r"(\d+(?:\.\d+)?)%", body)[:1]
        )
    disclosure = render_results(snapshot).text
    table_rows = [
        re.sub(r"<[^>]+>", " ", body)
        for body in re.findall(r"<tr\b[^>]*>(.*?)</tr>", disclosure, re.DOTALL)
    ]
    for row in (decision, sensitivity):
        expected_level = row["level"]
        assert isinstance(expected_level, float)
        assert any(
            str(row["metric"]) in body
            and str(row["method"]) in body
            and str(row["method_role"]) in body
            and any(
                float(level) / 100 == pytest.approx(expected_level, abs=0.00005)
                for level in re.findall(r"(\d+(?:\.\d+)?)%", body)
            )
            for body in table_rows
        )


@pytest.mark.parametrize(
    "status,discovery",
    [("exploratory_family", True), ("exploratory_family", False), ("exploratory_unadjusted", None)],
)
def test_a_rising_trajectory_honors_selection_and_declared_direction(
    storefront: DashboardSnapshot, status: str, discovery: bool | None
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
                "alternative": "two-sided",
                "discovery": discovery,
                "family_q": 0.05 if discovery is not None else None,
                "family_procedure": "bh" if discovery is not None else None,
                "multiplicity_status": status,
                "decision_scope_complete": discovery is True,
                "null_lift": 0.01,
                "posterior_prob_favorable": 0.73 if discovery is not None else None,
                "posterior_available": False if discovery is None else None,
                "posterior_reason_code": "estimation.engine.lift_guard"
                if discovery is None
                else None,
                "posterior_reason_context": {"reason": "delta_method_unreliable"}
                if discovery is None
                else None,
            }
        )
        for row in series
    )
    from increment.tables import estimates_to_readout

    assert (estimates_to_readout(rising)[0]["selection_qualification"] is not None) == (
        discovery is True
    )
    html = render_explore(storefront, rising, metric=model.name, view="cumulative_lift").text
    assert ("Selection evidence for" in html) == (discovery is True)
    if discovery is not None:
        assert "Discovery" in html and ("Yes" if discovery else "No") in html
    if discovery is None:
        assert "estimation.engine.lift_guard" in html
        assert "delta_method_unreliable" in html
    else:
        assert "Posterior P(favorable)" in html and "73.0%" in html
    from increment.dashboard._theme import coeftable_theme

    theme = coeftable_theme(storefront.config.theme)
    expected = theme.inconclusive if discovery is False else theme.unfavorable
    assert f'stroke="{expected}"' in html


def test_segmented_temporal_rendering_accepts_distinct_decision_methods(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates

    data = temporal_rows(storefront, "cumulative_lift", segmented=True)
    assert isinstance(data, DailyLiftEstimates)
    earlier_date = min(row.ds for row in data)
    mixed = DailyLiftEstimates(
        row.model_copy(
            update={
                "method": "cuped" if row.metric == "revenue_per_user" else row.method,
                "analysis_population": "assigned"
                if row.metric == "revenue_per_user"
                else "triggered",
                "null_lift": 0.01,
                "posterior_prob_favorable": 0.73 if row.dimension_value == "CA" else None,
                "posterior_available": False if row.dimension_value == "US" else None,
                "posterior_reason_code": "estimation.engine.lift_guard"
                if row.dimension_value == "US"
                else None,
                "posterior_reason_context": {"reason": "us_prior_guard"}
                if row.dimension_value == "US"
                else None,
                "lift": row.lift.model_copy(update={"level": 0.9})
                if row.metric == "revenue_per_user"
                and row.dimension_value == "US"
                and row.lift is not None
                else row.lift,
            }
        )
        for row in data
        if not (
            row.metric == "revenue_per_user"
            and row.dimension_value == "US"
            and row.ds != earlier_date
        )
    )
    rendered = render_explore(storefront, mixed, metric=None, view="cumulative_lift").text
    assert "revenue_per_user" in rendered and storefront.primary_metric in rendered
    assert "US" in rendered and "CA" in rendered
    assert "triggered" in rendered
    assert len(re.findall(r'<table\b[^>]*\bclass="[^"]*\bgt_table\b', rendered)) == 3
    assert f"CA · {storefront.treatment_group}" in rendered
    assert f"US · {storefront.treatment_group}" in rendered
    assert "73.0%" in rendered and "us_prior_guard" in rendered
    assert "cuped" in rendered
    captions = re.findall(
        r'<div class="inc-dashboard-temporal-table">\s*<p[^>]*>(.*?)</p>', rendered, re.S
    )
    assert len(captions) == 3
    assert all(
        any(metric in caption for caption in captions) for metric in {r.metric for r in mixed}
    )
    assert "Method:" in " ".join(captions) and "cuped" in " ".join(captions)
    assert "Population:" in " ".join(captions)
    assert "assigned" in " ".join(captions) and "triggered" in " ".join(captions)
    details = re.findall(r"<details\b.*?</details>", rendered, re.S)
    varied = next(detail for detail in details if "revenue_per_user" in detail)
    assert str(earlier_date) in varied and str(max(row.ds for row in data)) in varied
    assert "90%" in varied


@pytest.mark.parametrize(
    "point_state", ["available", "early_undefined", "all_undefined", "finite_tail"]
)
def test_one_sided_intervals_keep_unavailable_dates_as_gaps(
    storefront: DashboardSnapshot, point_state: str
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.estimation.results import Estimate

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
    if point_state == "finite_tail":
        rows = DailyLiftEstimates(
            row.model_copy(
                update={
                    "lift": Estimate(
                        value=0.02 * day, lb=0.02 * day - 0.05, ub=0.02 * day + 0.05, level=0.95
                    )
                }
            )
            if day >= 4
            else row
            for day, row in enumerate(rows, 1)
        )
    if point_state in ("early_undefined", "all_undefined"):
        region = binomial_region()
        region = type(region).model_validate(
            {
                **region.model_dump(),
                "lower": -1.0,
                "upper": None,
                "geometry": "lower_bound",
                "x_c": 0,
                "x_t": 0,
            }
        )
        rows = DailyLiftEstimates(
            type(row).model_validate(
                {
                    **row.model_dump(),
                    "lift": None,
                    "binomial_set": region,
                    "reference_kind": "binomial",
                }
            )
            if day != 3 and (point_state == "all_undefined" or day < 3)
            else row
            for day, row in enumerate(rows, 1)
        )
    rendered = render_explore(
        storefront, rows, metric=storefront.primary_metric, view="cumulative_lift"
    ).text
    if point_state == "finite_tail":
        chart = ET.fromstring(re.findall(r"<svg\b.*?</svg>", rendered, re.S)[0])
        bands = [p for p in chart.findall(".//{*}polygon") if p.get("fill-opacity") == "0.15"]
        assert len(bands) == 1, "finite dates retain their ribbon even after an open interval"
        final_run = chart.findall(".//{*}polyline")[1]
        start = float(final_run.attrib["points"].split()[0].split(",")[0])
        assert all(
            float(p.split(",")[0]) >= start - 0.02 for p in bands[0].attrib["points"].split()
        )
    polylines = re.findall(r'<polyline[^>]*points="([^"]+)"', rendered)
    assert (
        len(polylines)
        == {
            "available": 2,
            "early_undefined": 1,
            "all_undefined": 0,
            "finite_tail": 2,
        }[point_state]
    )
    # Missing points break the estimate; open bounds stay in details, not extra CI lines.
    assert all(len(points.split()) == 2 for points in polylines)
    assert "Lower CI" not in rendered and "Upper CI" not in rendered
    interval = (
        "[+5.0%, +15.0%]"
        if point_state == "finite_tail"
        else "[+5.0%, ∞)"
        if point_state != "all_undefined"
        else "-100.0%, ∞)"
    )
    assert interval in rendered
    legends = " ".join(re.findall(r"<text\b[^>]*>(.*?)</text>", rendered))
    assert ("Estimate" in legends) == (point_state != "all_undefined")
    assert "one-sided test for an increase" in rendered
    assert "two-sided test" not in rendered


@pytest.mark.parametrize(
    ("scale", "level", "coverage"),
    [
        ("relative", 0.95, "95"),
        ("absolute", 0.95, "95"),
        ("absolute_aware", 0.95, "95"),
        ("absolute_fold", 0.95, "95"),
        pytest.param("relative", 0.9833333333333333, "98.33", id="coverage-ordinary"),
        pytest.param("relative", 0.99999, "99.999", id="coverage-near-one"),
        pytest.param("relative", 0.00001, "0.001", id="coverage-near-zero"),
        pytest.param("relative", 0.9999999999999999, None, id="coverage-adjacent-one"),
        pytest.param("relative", 5e-324, None, id="coverage-smallest-positive"),
    ],
)
def test_temporal_lift_axes_and_latest_reading_keep_the_captured_scale(
    storefront: DashboardSnapshot, scale: str, level: float, coverage: str | None
) -> None:
    from zoneinfo import ZoneInfo

    from increment.breakout.estimates import DailyLiftEstimates
    from increment.estimation.results import Estimate

    model = next(model for model in storefront.metrics if model.name == "revenue_per_user")
    seed = next(
        row for row in temporal_rows(storefront, "cumulative_lift") if row.metric == model.name
    )
    magnitude = 1 if scale == "relative" else 100
    dates = (
        [
            dt.datetime(2025, 11, 2, 1, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4))),
            dt.datetime(2025, 11, 2, 1, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
            dt.datetime(2025, 11, 2, 2, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
        ]
        if scale == "absolute_aware"
        else [dt.date(2025, 1, day) for day in range(1, 4)]
    )
    folded = scale == "absolute_fold"
    if folded:
        zone = ZoneInfo("America/New_York")
        dates = [
            dt.datetime(2025, 11, 2, 0, 45, tzinfo=zone),
            dt.datetime(2025, 11, 2, 1, 45, tzinfo=zone, fold=0),
            dt.datetime(2025, 11, 2, 1, 15, tzinfo=zone, fold=1),
        ]
    scale = scale.split("_", 1)[0]
    rows = DailyLiftEstimates(
        seed.model_copy(
            update={
                "ds": dates[day - 1],
                "value_scale": scale,
                "alternative": "less",
                "null_lift": 0.15 if scale == "relative" else 0.0,
                "null_abs": 15.0 if scale == "absolute" else None,
                "lift": Estimate(
                    value=0.1 * day * magnitude,
                    lb=0.05 * day * magnitude,
                    ub=0.15 * day * magnitude,
                    level=level,
                ),
            }
        )
        for day in range(1, 4)
    )

    rendered = temporal_figure(storefront, rows, metric=model.name, view="cumulative_lift")
    label = re.search(r"([-+\d.e]+)% intervals", rendered)
    assert label is not None
    shown = label.group(1)
    assert 0 < float(shown) < 100
    if coverage is not None:
        assert shown == coverage
    else:
        assert float(shown) == pytest.approx(level * 100, rel=1e-14, abs=0)
    svgs = re.findall(r"<svg\b.*?</svg>", rendered, re.S)
    assert svgs
    tick_text = " ".join(re.findall(r"<text\b[^>]*>(.*?)</text>", "".join(svgs)))
    assert ("%" in tick_text) == (scale == "relative")
    assert ("+30.0%" if scale == "relative" else "+30") in rendered
    chart = ET.fromstring(svgs[0])
    line = chart.find(".//{*}polyline")
    assert line is not None
    points = [tuple(map(float, point.split(","))) for point in line.attrib["points"].split()]
    assert len(points) == 3
    assert points[0][0] < points[1][0] < points[2][0]
    if folded:
        assert points[1][0] - points[0][0] == pytest.approx(
            2 * (points[2][0] - points[1][0]), abs=0.03
        )
    else:
        assert points[1][0] == pytest.approx((points[0][0] + points[2][0]) / 2, abs=0.02)
    assert points[1][1] == pytest.approx((points[0][1] + points[2][1]) / 2, abs=0.02)
    reference = next(
        line for line in chart.findall(".//{*}line") if "stroke-dasharray" in line.attrib
    )
    assert float(reference.attrib["y1"]) == pytest.approx(
        (points[0][1] + points[1][1]) / 2, abs=0.02
    )
    assert chart.find(".//{*}polygon") is not None
    assert "one-sided test for a decrease" in rendered
    assert "two-sided test" not in rendered


def test_temporal_early_outlier_keeps_its_true_ghost_and_clipped_geometry(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.estimation.results import Estimate

    seed = next(iter(temporal_rows(storefront, "cumulative_lift")))
    rows = DailyLiftEstimates(
        seed.model_copy(
            update={
                "ds": dt.date(2025, 1, day),
                "lift": Estimate(value=value, lb=value - 0.01, ub=value + 0.01, level=0.95),
            }
        )
        for day, value in enumerate((10, 0.02, 0.04, 0.06, 0.08, 0.1), 1)
    )
    rendered = render_explore(storefront, rows, metric=seed.metric, view="cumulative_lift").text
    match = re.search(r"<svg\b.*?</svg>", rendered, re.S)
    assert match is not None
    chart = ET.fromstring(match.group())
    height = float(chart.attrib["height"])
    lines = chart.findall(".//{*}polyline")
    ghost = next(line for line in lines if line.get("stroke-opacity") == "0.35")
    assert any(float(point.split(",")[1]) < 0 for point in ghost.attrib["points"].split())
    for line in lines:
        if line.get("stroke-opacity") is None:
            assert all(
                0 <= float(point.split(",")[1]) <= height for point in line.attrib["points"].split()
            )
    assert chart.find(".//{*}clipPath") is not None


@pytest.mark.slow
@pytest.mark.parametrize("surface", ["readout", "report"])
def test_complete_dashboard_retains_adverse_nonrejecting_guardrail(
    storefront: DashboardSnapshot, surface: str
) -> None:

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
    limitations = _rendered_payload(snapshot)["report"]["limitationsHtml"]
    notes = unescape(re.sub(r"<[^>]+>", " ", limitations))
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


def test_primary_coverage_discloses_retained_alpha_when_level_rounds_to_one(
    storefront: DashboardSnapshot,
) -> None:
    index = next(
        i
        for i, row in enumerate(storefront.readout_rows)
        if row["metric"] == storefront.primary_metric
    )
    estimate = storefront.estimates[index]
    assert estimate.lift is not None
    updated = list(storefront.estimates)
    updated[index] = estimate.model_copy(
        update={"lift": estimate.lift.model_copy(update={"level": 1.0, "alpha": 1e-17})}
    )
    rows = tuple(
        {**dict(row), "level": 1.0} if row["metric"] == storefront.primary_metric else row
        for row in storefront.readout_rows
    )
    snapshot = dataclasses.replace(storefront, estimates=tuple(updated), readout_rows=rows)
    html = render_header(snapshot).text
    assert "100% (α=1e-17)" in html

    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    assert payload["primary"]["levelLabel"] == "100% (α=1e-17)"
    assert "100% (α=1e-17)" in payload["primary"]["method"]
    rows = list(csv.DictReader(io.StringIO(payload["readoutCsv"])))
    primary_row = next(row for row in rows if row["metric"] == snapshot.primary_metric)
    assert primary_row["coverage_alpha"] == "1e-17"
    assert any(row["coverage_alpha"] == "" for row in rows)

    direct_rows = list(csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8"))))
    direct_primary = next(row for row in direct_rows if row["metric"] == snapshot.primary_metric)
    assert direct_primary["coverage_alpha"] == "1e-17"


def test_secondary_coverage_budget_reaches_metric_details(
    storefront: DashboardSnapshot,
) -> None:
    index = next(
        i
        for i, row in enumerate(storefront.readout_rows)
        if row["metric"] != storefront.primary_metric
    )
    estimate = storefront.estimates[index]
    assert estimate.lift is not None
    updated = list(storefront.estimates)
    updated[index] = estimate.model_copy(
        update={"lift": estimate.lift.model_copy(update={"level": 1.0, "alpha": 1e-17})}
    )
    rows = list(storefront.readout_rows)
    rows[index] = {**dict(rows[index]), "level": 1.0}
    assert storefront.readout_collection is not None
    collection = type(storefront.readout_collection)(
        updated, metadata=storefront.readout_collection.metadata
    )
    snapshot = dataclasses.replace(
        storefront,
        estimates=tuple(updated),
        readout_rows=tuple(rows),
        readout_collection=collection,
    )

    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    metric = str(rows[index]["metric"])
    detail = next(
        item["readingHtml"]
        for item in payload["references"]["results"].values()
        if item["cell"]["metric"] == metric and item["cell"]["ds"] is None
    )
    assert "100% (α=1e-17)" in detail


def test_report_keeps_equal_rounded_levels_with_distinct_retained_budgets(
    storefront: DashboardSnapshot,
) -> None:
    indices = [
        i
        for i, row in enumerate(storefront.readout_rows)
        if row["metric"] != storefront.primary_metric
        and storefront.estimates[i].confidence_set is None
        and storefront.estimates[i].lift is not None
    ][:2]
    assert len(indices) == 2
    updated = list(storefront.estimates)
    rows = list(storefront.readout_rows)
    from increment.estimation.results import Estimate

    for index, alpha in zip(indices, (1e-17, 1e-18), strict=True):
        estimate = updated[index]
        assert estimate.lift is not None
        updated[index] = estimate.model_copy(
            update={
                "lift": Estimate(
                    value=estimate.lift.value,
                    lb=estimate.lift.lb,
                    ub=estimate.lift.ub,
                    level=1.0,
                    alpha=alpha,
                )
            }
        )
        rows[index] = {**dict(rows[index]), "level": 1.0}
    snapshot = dataclasses.replace(storefront, estimates=tuple(updated), readout_rows=tuple(rows))

    payload = build_payload(experiment_for(snapshot), snapshot=snapshot)
    methods = payload["report"]["methodsHtml"]
    interval_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", methods, re.DOTALL)
    for index, alpha in zip(indices, (1e-17, 1e-18), strict=True):
        assert any(
            str(rows[index]["metric"]) in body and repr(alpha) in body for body in interval_rows
        )
    printable = re.sub(r"<details\b.*?</details>", "", methods, flags=re.DOTALL)
    assert any(
        str(rows[indices[0]]["metric"]) in body and "100% (α=1e-17)" in body
        for body in re.findall(r"<tr\b[^>]*>(.*?)</tr>", printable, re.DOTALL)
    )


def test_report_preserves_distinct_raw_levels_that_round_alike(
    storefront: DashboardSnapshot,
) -> None:
    indices = [
        i
        for i, row in enumerate(storefront.readout_rows)
        if row["metric"] != storefront.primary_metric
        and storefront.estimates[i].confidence_set is None
        and storefront.estimates[i].lift is not None
    ][:2]
    assert len(indices) == 2
    updated = list(storefront.estimates)
    rows = list(storefront.readout_rows)
    for index, level, alpha in zip(indices, (0.95, 0.95001), (0.05, 0.04999), strict=True):
        estimate = updated[index]
        assert estimate.lift is not None
        updated[index] = estimate.model_copy(
            update={"lift": estimate.lift.model_copy(update={"level": level, "alpha": alpha})}
        )
        rows[index] = {**dict(rows[index]), "level": level}
    snapshot = dataclasses.replace(storefront, estimates=tuple(updated), readout_rows=tuple(rows))

    methods = build_payload(experiment_for(snapshot), snapshot=snapshot)["report"]["methodsHtml"]
    interval_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", methods, re.DOTALL)
    for index, alpha in zip(indices, (0.05, 0.04999), strict=True):
        assert any(
            str(rows[index]["metric"]) in body and repr(alpha) in body for body in interval_rows
        )
    printable = re.sub(r"<details\b.*?</details>", "", methods, flags=re.DOTALL)
    assert any(
        str(rows[indices[0]]["metric"]) in body and "95%" in body
        for body in re.findall(r"<tr\b[^>]*>(.*?)</tr>", printable, re.DOTALL)
    )


def test_relative_set_coverage_discloses_its_own_retained_alpha() -> None:
    from increment.dashboard._format import interval_html
    from increment.estimation.inference import relative_confidence_set
    from increment.estimation.results import JointContrastReference

    region = relative_confidence_set(
        JointContrastReference(a=10.0, c=0.0, var_a=1.0, var_c=1.0, cov_ac=0.0),
        alpha=1e-17,
        alternative="greater",
    )
    assert region.geometry in {"one_sided", "disconnected", "all_real"}
    shown = interval_html({"relative_confidence_set": region, "value_scale": "relative"})
    assert shown is not None
    assert "100% (α=1e-17)" in shown


def test_winsor_set_coverage_discloses_its_own_retained_alpha(
    storefront: DashboardSnapshot,
) -> None:
    from increment.estimation.results import estimate_winsor_lift
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    raw = WinsorRawState(
        metric=next(metric.name for metric in storefront.metrics if metric.type == "mean"),
        study_id="coverage-display",
        missingness="error",
        quantile=0.5,
        inference=WinsorInferenceSpec(method="influence-normal-v1"),
        arms=(
            RawArm(group_id="control", values=tuple(float(i) for i in range(1, 41))),
            RawArm(group_id="treatment", values=tuple(float(i) * 1.1 for i in range(1, 41))),
        ),
    )
    estimate = estimate_winsor_lift(raw, "control", "treatment", alpha=1e-17)
    assert estimate.confidence_set is not None
    assert estimate.lift is not None
    without_lift_alpha = estimate.model_copy(
        update={"lift": estimate.lift.model_copy(update={"alpha": None})}
    )
    snapshot = dataclasses.replace(
        storefront,
        estimates=(without_lift_alpha,),
        readout_rows=enriched_rows((without_lift_alpha,)),
    )
    assert "100% (α=1e-17)" in render_results(snapshot).text
    row = next(csv.DictReader(io.StringIO(readout_csv(snapshot).decode("utf-8"))))
    assert row["coverage_alpha"] == "1e-17"


def test_temporal_caption_does_not_merge_rounded_equal_levels_with_different_budgets(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.estimation.results import Estimate

    seed = next(iter(temporal_rows(storefront, "cumulative_lift")))
    rows = DailyLiftEstimates(
        seed.model_copy(
            update={
                "ds": dt.date(2025, 1, day),
                "dimension": "country",
                "dimension_value": country,
                "lift": Estimate(
                    value=0.1,
                    lb=0.05,
                    ub=0.15,
                    level=1.0,
                    alpha=alpha,
                ),
            }
        )
        for country, alpha in (("DE", 1e-17), ("GB", 1e-18))
        for day in range(1, 4)
    )
    rendered = temporal_figure(storefront, rows, metric=seed.metric, view="cumulative_lift")
    assert "100% (α=1e-17)" in rendered
    assert "100% (α=1e-18)" in rendered


def test_temporal_notes_attribute_point_free_set_coverage_to_its_series(
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimate, DailyLiftEstimates
    from increment.estimation.results import Estimate
    from tests.test_tables import _binomial_lift_estimate

    seed = next(iter(temporal_rows(storefront, "cumulative_lift")))
    point_backed = seed.model_copy(
        update={
            "dimension": "country",
            "dimension_value": "DE",
            "lift": Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95, alpha=0.05),
        }
    )
    point_free_estimate = _binomial_lift_estimate(x_c=0, n_c=200, x_t=100, n_t=200, alpha=0.04)
    assert point_free_estimate.binomial_set is not None
    point_free = DailyLiftEstimate.model_validate(
        {
            **seed.model_dump(mode="python"),
            "dimension": "country",
            "dimension_value": "GB",
            "reference_kind": "binomial",
            "lift": None,
            "binomial_set": point_free_estimate.binomial_set,
            "method": point_free_estimate.method,
            "method_role": point_free_estimate.method_role,
            "value_scale": point_free_estimate.value_scale,
        }
    )
    rows = DailyLiftEstimates([point_backed, point_free])

    packet = _render_entry(
        storefront,
        rows,
        metric=seed.metric,
        view="cumulative_lift",
        complete=False,
        breakout=("event_log", "country"),
        correction=None,
    )
    notes = " ".join(packet["notes"])
    coverage_groups = re.findall(r"(\d+(?:\.\d+)?%) \(([^)]+)\)", notes)
    assert any(level == "95%" and "country=DE" in group for level, group in coverage_groups)
    assert not any(level == "95%" and "country=GB" in group for level, group in coverage_groups)
    assert any(level == "96%" and "country=GB" in group for level, group in coverage_groups)


def test_daily_value_without_alpha_keeps_its_retained_level(storefront: DashboardSnapshot) -> None:
    from datetime import date

    from increment.breakout.estimates import DailyMetricValue, DailyMetricValues
    from increment.estimation.results import Estimate
    from increment.tables import estimates_to_readout

    estimate = DailyMetricValue(
        ds=date(2025, 1, 1),
        metric="revenue",
        group_id="treatment",
        value=Estimate(value=0.1, lb=0.0, ub=0.2, level=0.95, alpha=None),
        n=10,
    )

    (row,) = estimates_to_readout(DailyMetricValues([estimate]))

    assert row["level"] == 0.95
    assert row["coverage_alpha"] is None


@pytest.mark.integration
def test_report_reading_order_qualifies_primary_and_results_with_warnings(storefront):
    from html.parser import HTMLParser

    order = []

    class ReportOrder(HTMLParser):
        def handle_starttag(self, tag, attrs):
            identity = dict(attrs).get("id")
            if identity in ("report-health-section", "report-primary-box", "report-table"):
                order.append(identity)

    page = render_dashboard(experiment_for(storefront), snapshot=storefront).widget.get_state()[
        "document"
    ]
    ReportOrder().feed(page)
    assert order == ["report-health-section", "report-primary-box", "report-table"]
