"""Behavioural checks for the optional dashboard adapter layer.

Two ordinary experiments are bound from one in-memory DuckDB fixture: an
even-split fixed-horizon experiment with a declared breakout, and a renamed
70/30 fixed-horizon experiment with none. Both go through exactly the same
preparation and rendering calls -- only construction and configuration differ.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import json
import random
import re
from typing import Any

import pytest

from increment.errors import CapabilityError, InvalidRequestError

pytest.importorskip("coeftable")
pytest.importorskip("marimo")

from increment.dashboard import (
    DashboardConfig,
    DashboardSnapshot,
    ExploreView,
    load_explore,
    prepare_dashboard,
    readout_csv,
    render_explore,
    render_header,
    render_health,
    render_metric_details,
    render_results,
)

STOREFRONT_WINDOW = (dt.date(2025, 1, 15), dt.date(2025, 2, 14))
PRICING_WINDOW = (dt.date(2025, 3, 1), dt.date(2025, 3, 31))

FACT_SOURCES = """
dialect: duckdb

fact_sources:
  - name: event_log
    sql: |
      SELECT * FROM analytics.event_log
    timestamp_column: event_at
    entities:
      - user_id
    facts:
      - name: page_view
        column: null
        description: A page view
      - name: purchase
        column: revenue
        description: A purchase with revenue
      - name: session_end
        column: duration_s
        description: Session ended with a duration
    properties:
      - name: country
        column: country_code
        dtype: string
        as_of: static
        description: Country, fixed per unit in this fixture
"""

EXPOSURES = """
exposures:
  - name: first_page_view
    fact: page_view
    description: First page view inside the analysis window
"""

METRICS = """
metrics:
  - type: conversion
    name: checkout_conversion
    description: Did the user check out within 14 days of exposure?
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14

  - type: mean
    name: revenue_per_user
    description: Purchase revenue per enrolled user
    entity: user_id
    preferred_direction: increase
    fact: purchase
    aggregation: sum
    window_days: 14

  - type: mean
    name: session_seconds
    description: Seconds spent in session per enrolled user
    entity: user_id
    preferred_direction: decrease
    fact: session_end
    aggregation: sum
    window_days: 14

  - type: conversion
    name: purchase_rate
    description: Did the user purchase within 14 days of exposure?
    entity: user_id
    preferred_direction: increase
    fact: purchase
    window_days: 14
"""

EXPERIMENTS = """
experiments:
  - name: storefront_refresh
    description: A refreshed storefront against the current storefront
    exposure: first_page_view
    unit: user_id
    start: 2025-01-15
    end: 2025-02-14
    n_pre_periods: 0
    control_group: control
    plan:
      view_multiplicity:
        correction: bonferroni
      primary: checkout_conversion
      secondaries:
        - revenue_per_user
      guardrails:
        - session_seconds
    breakouts:
      - property: country
        source: event_log

  - name: pricing_copy
    description: Reworded pricing copy against the current copy
    exposure: first_page_view
    unit: user_id
    start: 2025-03-01
    end: 2025-03-31
    n_pre_periods: 0
    control_group: baseline
    plan:
      primary: purchase_rate

  - name: dual_primary
    description: Two declared primary metrics, which the dashboard refuses
    exposure: first_page_view
    unit: user_id
    start: 2025-03-01
    end: 2025-03-31
    n_pre_periods: 0
    control_group: control
    plan:
      primary:
        - checkout_conversion
        - revenue_per_user
"""


def _arm_rows(
    rng: random.Random,
    *,
    experiment_id: str,
    group_id: str,
    prefix: str,
    units: int,
    start: dt.date,
    convert_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(units):
        user = f"{prefix}{index:05d}"
        exposure_at = dt.datetime.combine(start, dt.time(9, 0)) + dt.timedelta(
            days=index % 10, minutes=index % 37
        )
        country = "US" if index % 3 else "CA"
        rows.append(
            {
                "event_at": exposure_at,
                "user_id": user,
                "event": "page_view",
                "experiment_id": experiment_id,
                "group_id": group_id,
                "revenue": None,
                "duration_s": None,
                "country_code": country,
            }
        )
        rows.append(
            {
                "event_at": exposure_at + dt.timedelta(hours=1),
                "user_id": user,
                "event": "session_end",
                "experiment_id": experiment_id,
                "group_id": group_id,
                "revenue": None,
                "duration_s": 40.0 + rng.random() * 20.0,
                "country_code": country,
            }
        )
        if rng.random() < convert_rate:
            rows.append(
                {
                    "event_at": exposure_at + dt.timedelta(days=1),
                    "user_id": user,
                    "event": "purchase",
                    "experiment_id": experiment_id,
                    "group_id": group_id,
                    "revenue": 20.0 + rng.random() * 30.0,
                    "duration_s": None,
                    "country_code": country,
                }
            )
    return rows


@pytest.fixture(scope="session")
def dashboard_definitions(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("dashboard_definitions")
    for name, body in (
        ("fact_sources.yaml", FACT_SOURCES),
        ("exposures.yaml", EXPOSURES),
        ("metrics.yaml", METRICS),
        ("experiments.yaml", EXPERIMENTS),
    ):
        (path / name).write_text(body)
    return str(path)


@pytest.fixture(scope="session")
def dashboard_con():
    import ibis
    import pyarrow as pa

    rng = random.Random(11)
    rows = (
        _arm_rows(
            rng,
            experiment_id="storefront_refresh",
            group_id="control",
            prefix="sc",
            units=600,
            start=STOREFRONT_WINDOW[0],
            convert_rate=0.20,
        )
        + _arm_rows(
            rng,
            experiment_id="storefront_refresh",
            group_id="treatment",
            prefix="st",
            units=600,
            start=STOREFRONT_WINDOW[0],
            convert_rate=0.25,
        )
        + _arm_rows(
            rng,
            experiment_id="pricing_copy",
            group_id="baseline",
            prefix="pb",
            units=700,
            start=PRICING_WINDOW[0],
            convert_rate=0.15,
        )
        + _arm_rows(
            rng,
            experiment_id="pricing_copy",
            group_id="candidate",
            prefix="pc",
            units=300,
            start=PRICING_WINDOW[0],
            convert_rate=0.19,
        )
    )
    # Each metric window closes 14 days after the last exposure, and the
    # observable bound is per fact: every fact needs an event past that date
    # or censoring drops every enrolled unit.
    rows.extend(
        {
            "event_at": dt.datetime(2025, 5, 1, 12, 0),
            "user_id": "keepalive",
            "event": event,
            "experiment_id": None,
            "group_id": None,
            "revenue": 1.0 if event == "purchase" else None,
            "duration_s": 1.0 if event == "session_end" else None,
            "country_code": "US",
        }
        for event in ("page_view", "purchase", "session_end")
    )
    schema = pa.schema(
        [
            ("event_at", pa.timestamp("us")),
            ("user_id", pa.string()),
            ("event", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", pa.string()),
            ("revenue", pa.float64()),
            ("duration_s", pa.float64()),
            ("country_code", pa.string()),
        ]
    )
    con = ibis.duckdb.connect()
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    con.create_table("event_log", pa.Table.from_pylist(rows, schema=schema), database="analytics")
    return con


def _analysis(con, definitions: str, experiment: str):
    from increment import Analysis

    return Analysis.from_definitions(experiment, definitions, con)


@pytest.fixture(scope="session")
def storefront(dashboard_con, dashboard_definitions) -> DashboardSnapshot:
    analysis = _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")
    config = DashboardConfig(
        expected_allocation={"control": 0.5, "treatment": 0.5},
        source_label="Unit fixture",
    )
    return prepare_dashboard(analysis, config=config)


@pytest.fixture(scope="session")
def pricing(dashboard_con, dashboard_definitions) -> DashboardSnapshot:
    analysis = _analysis(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(
        expected_allocation={"baseline": 0.7, "candidate": 0.3},
        title="Pricing copy rewrite",
        provenance={"Fixture": "in-memory"},
    )
    return prepare_dashboard(analysis, config=config)


@pytest.mark.slow
@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_preparation_preserves_registered_sequential_decisions(law):
    from tests.test_sequential_public_sources import _native_fixture

    connection, _, analysis = _native_fixture(law)
    try:
        analysis.capture_sequential(finalized=True, as_of=dt.date(2025, 1, 16))
        expected = analysis.run()[0]
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
        )
        observed = snapshot.estimates[0]
        assert observed.stat_sig() is True
        assert observed.require_sequential_result() == expected.require_sequential_result()
    finally:
        analysis.close()
        connection.disconnect()


@pytest.mark.slow
def test_preparation_pins_allocation_history_and_estimates_without_rebinding_caller(
    dashboard_con, dashboard_definitions, pricing, monkeypatch
):
    import ibis

    from increment import Analysis
    from increment.estimation.diagnostics import SRMResult

    con = ibis.duckdb.connect()
    try:
        con.raw_sql("CREATE SCHEMA analytics")
        con.create_table(
            "event_log",
            dashboard_con.table("event_log", database="analytics").to_pyarrow(),
            database="analytics",
        )
        analysis = _analysis(con, dashboard_definitions, "pricing_copy")
        original_srm = Analysis.srm

        def mutate_after_allocation(pinned, **kwargs):
            allocation = original_srm(pinned, **kwargs)
            con.raw_sql(
                "INSERT INTO analytics.event_log "
                "SELECT event_at, user_id || '_new', event, experiment_id, group_id, "
                "revenue, duration_s, country_code FROM analytics.event_log "
                "WHERE experiment_id = 'pricing_copy' AND group_id = 'candidate'"
            )
            live = original_srm(analysis, **kwargs)
            assert isinstance(live, SRMResult)
            assert live.observed == {"baseline": 700, "candidate": 600}
            return allocation

        with monkeypatch.context() as patch:
            patch.setattr(Analysis, "srm", mutate_after_allocation)
            snapshot = prepare_dashboard(analysis, config=pricing.config)

        assert snapshot.allocation is not None
        assert snapshot.allocation.observed == {"baseline": 700, "candidate": 300}
        final_counts = {row["group_id"]: row["n_cumulative"] for row in snapshot.allocation_history}
        assert final_counts == snapshot.allocation.observed
        for field in ("lift", "lower", "higher"):
            assert snapshot.readout_rows[0][field] == pytest.approx(pricing.readout_rows[0][field])
        fresh = prepare_dashboard(analysis, config=pricing.config)
        assert fresh.allocation is not None
        assert fresh.allocation.observed == {"baseline": 700, "candidate": 600}
        assert fresh.readout_rows[0]["lower"] != snapshot.readout_rows[0]["lower"]
    finally:
        con.disconnect()


@pytest.mark.parametrize("change", ["control", "window", "policy", "breakouts"])
def test_explore_rejects_same_name_binding_changes_before_query(
    dashboard_con, dashboard_definitions, pricing, tmp_path, monkeypatch, change
):
    import shutil

    import yaml

    definitions = tmp_path / "definitions"
    shutil.copytree(dashboard_definitions, definitions)
    experiments_path = definitions / "experiments.yaml"
    experiments = yaml.safe_load(experiments_path.read_text())
    experiment = next(item for item in experiments["experiments"] if item["name"] == "pricing_copy")
    if change == "control":
        experiment["control_group"] = "candidate"
    elif change == "policy":
        experiment["plan"]["alpha"] = 0.01
    elif change == "breakouts":
        experiment["breakouts"] = [{"property": "country", "source": "event_log"}]
    else:
        metrics_path = definitions / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        metric = next(item for item in metrics["metrics"] if item["name"] == "purchase_rate")
        metric["window_days"] = 15
        metrics_path.write_text(yaml.safe_dump(metrics))
    experiments_path.write_text(yaml.safe_dump(experiments))
    analysis = _analysis(dashboard_con, str(definitions), "pricing_copy")

    def unexpected_query(**kwargs):
        pytest.fail("queried a source with a different dashboard binding")

    monkeypatch.setattr(analysis, "run_daily", unexpected_query)
    with pytest.raises(InvalidRequestError) as error:
        load_explore(analysis, snapshot=pricing, metric="purchase_rate", view="daily_values")
    assert error.value.code == "dashboard.invalid_view"


# Binding: the same calls serve two differently shaped experiments.


def test_even_split_experiment_binds_declared_metadata(storefront: DashboardSnapshot) -> None:
    assert storefront.experiment_name == "storefront_refresh"
    assert storefront.title == "storefront_refresh"
    assert (storefront.control_group, storefront.treatment_group) == ("control", "treatment")
    assert storefront.primary_metric == "checkout_conversion"
    assert [metric.name for metric in storefront.metrics] == [
        "checkout_conversion",
        "revenue_per_user",
        "session_seconds",
    ]
    assert storefront.breakouts == (("event_log", "country"),)
    assert {row["inference"] for row in storefront.readout_rows} == {"fixed"}


def test_renamed_unequal_experiment_binds_its_own_metadata(pricing: DashboardSnapshot) -> None:
    assert pricing.experiment_name == "pricing_copy"
    assert pricing.title == "Pricing copy rewrite"
    assert (pricing.control_group, pricing.treatment_group) == ("baseline", "candidate")
    assert pricing.primary_metric == "purchase_rate"
    assert pricing.config.expected_allocation == {"baseline": 0.7, "candidate": 0.3}
    assert pricing.breakouts == ()
    assert {row["inference"] for row in pricing.readout_rows} == {"fixed"}


def test_allocation_check_uses_configured_shares(pricing: DashboardSnapshot) -> None:
    allocation = pricing.allocation
    assert allocation is not None
    assert allocation.expected == {"baseline": 0.7, "candidate": 0.3}
    assert allocation.observed == {"baseline": 700, "candidate": 300}
    assert allocation.is_srm is False


def test_renders_expected_shares_from_configuration(pricing: DashboardSnapshot) -> None:
    html = render_health(pricing).text
    assert "70.0%" in html
    assert "30.0%" in html
    assert "baseline" in html
    assert "candidate" in html


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


def test_rows_carry_the_tested_alternative(storefront: DashboardSnapshot) -> None:
    tails = {row["metric"]: row["alternative"] for row in storefront.readout_rows}
    alternatives = {estimate.metric: estimate.alternative for estimate in storefront.estimates}
    assert tails == alternatives
    # A margin-less guardrail is tested on one side only.
    assert tails["session_seconds"] == "less"
    assert tails["checkout_conversion"] == "two-sided"


def test_rows_correspond_one_to_one_with_estimates(storefront: DashboardSnapshot) -> None:
    assert len(storefront.readout_rows) == len(storefront.estimates)
    for row, estimate in zip(storefront.readout_rows, storefront.estimates, strict=True):
        assert row["metric"] == estimate.metric
        assert row["group_id"] == estimate.group_id
        assert row["lift"] == pytest.approx(estimate.require_lift().value)


# Allocation history: real assigned-unit enrollment, not derived from samples.


def test_snapshot_owns_nested_readout_data(storefront: DashboardSnapshot) -> None:
    rows = [dict(row) for row in storefront.readout_rows]
    components: list[dict[str, Any]] = [{"bounds": {"lower": 0.25}, "labels": ["registered"]}]
    rows[0]["sequential_components"] = components
    snapshot = dataclasses.replace(storefront, readout_rows=tuple(rows))
    before = readout_csv(snapshot)
    components[0]["bounds"]["lower"] = 99.0
    components[0]["labels"].append("changed")
    assert readout_csv(snapshot) == before
    frozen = snapshot.readout_rows[0]["sequential_components"][0]
    with pytest.raises(TypeError):
        frozen["bounds"]["lower"] = 0.0
    with pytest.raises(AttributeError):
        frozen["labels"].append("changed")


def test_snapshot_allocation_maps_are_owned_and_immutable(storefront: DashboardSnapshot) -> None:
    assert storefront.allocation is not None
    original = storefront.allocation
    payload = original.model_dump()
    allocation = type(original).model_validate(payload)
    snapshot = dataclasses.replace(storefront, allocation=allocation)
    assert snapshot.allocation is not None
    before = snapshot.allocation.model_dump_json()
    for name in ("observed", "expected", "unit_counts"):
        payload[name]["control"] = 999
        with pytest.raises(TypeError):
            getattr(allocation, name)["control"] = 999
    assert snapshot.allocation.model_dump_json() == before
    restored = type(original).model_validate_json(allocation.model_dump_json())
    assert restored.observed == original.observed
    assert restored.expected == original.expected
    assert restored.unit_counts == original.unit_counts


def test_allocation_history_rows_are_immutable(storefront: DashboardSnapshot) -> None:
    assert storefront.allocation_history_refusal is None
    raw = [dict(storefront.allocation_history[0])]
    copied = dataclasses.replace(storefront, allocation_history=tuple(raw))
    original_count = copied.allocation_history[0]["n_daily"]
    raw[0]["n_daily"] += 100
    assert copied.allocation_history[0]["n_daily"] == original_count
    row = storefront.allocation_history[0]
    with pytest.raises(TypeError):
        row["n_daily"] = 0  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        storefront.allocation_history.append(row)  # ty: ignore[unresolved-attribute]


def test_allocation_history_reconciles_with_the_renamed_unequal_experiment(
    pricing: DashboardSnapshot,
) -> None:
    """The 70/30 renamed arms' real cumulative totals, not a derived guess."""
    assert pricing.allocation_history_refusal is None
    assert pricing.allocation is not None
    rows = pricing.allocation_history
    assert {row["experiment_id"] for row in rows} == {"pricing_copy"}
    assert {row["group_id"] for row in rows} == {"baseline", "candidate"}
    assert list(rows) == sorted(rows, key=lambda row: (row["ds"], row["group_id"]))
    final_totals = {
        group: max(row["n_cumulative"] for row in rows if row["group_id"] == group)
        for group in ("baseline", "candidate")
    }
    assert final_totals == pricing.allocation.observed


def test_allocation_history_cannot_also_be_refused(storefront: DashboardSnapshot) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        dataclasses.replace(
            storefront, allocation_history_refusal=("source.native.operation", "unsupported")
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


def test_prepared_config_rejects_mutation(pricing: DashboardSnapshot) -> None:
    with pytest.raises(TypeError):
        pricing.config.expected_allocation["baseline"] = 0.1  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        pricing.readout_rows[0]["lift"] = 0.0  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        pricing.primary_metric = "other"  # ty: ignore[invalid-assignment]


def test_mutating_caller_allocation_cannot_change_a_prepared_result(
    dashboard_con, dashboard_definitions
) -> None:
    allocation = {"baseline": 0.7, "candidate": 0.3}
    analysis = _analysis(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation=allocation)
    snapshot = prepare_dashboard(analysis, config=config)

    allocation["baseline"] = 0.2
    allocation["candidate"] = 0.8

    assert snapshot.allocation is not None
    assert snapshot.allocation.expected == {"baseline": 0.7, "candidate": 0.3}
    assert snapshot.config.expected_allocation == {"baseline": 0.7, "candidate": 0.3}


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


def test_preparation_refuses_allocation_without_the_control_arm(
    dashboard_con, dashboard_definitions
) -> None:
    analysis = _analysis(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with pytest.raises(InvalidRequestError) as caught:
        prepare_dashboard(analysis, config=config)
    assert caught.value.code == "dashboard.invalid_config"
    assert caught.value.context["control"] == "baseline"


def test_preparation_surfaces_an_arm_label_absent_from_the_data(
    dashboard_con, dashboard_definitions
) -> None:
    """An unusable request keeps the source's own code, not a health badge."""
    analysis = _analysis(dashboard_con, dashboard_definitions, "pricing_copy")
    config = DashboardConfig(expected_allocation={"baseline": 0.7, "challenger": 0.3})
    with pytest.raises(InvalidRequestError) as caught:
        prepare_dashboard(analysis, config=config)
    assert caught.value.code == "estimation.diagnostics.expected_keys_do"
    assert "candidate" in caught.value.context["counts_keys"]  # ty: ignore[unsupported-operator]


def test_preparation_refuses_more_than_one_declared_primary(
    dashboard_con, dashboard_definitions
) -> None:
    analysis = _analysis(dashboard_con, dashboard_definitions, "dual_primary")
    config = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    with pytest.raises(CapabilityError) as caught:
        prepare_dashboard(analysis, config=config)
    assert caught.value.code == "dashboard.unsupported_experiment"
    assert caught.value.context["primaries"] == ("checkout_conversion", "revenue_per_user")


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
    assert "facade.analysis.invalid_population" in html
    assert "population is unsupported" in html
    assert "not a passing check" in html
    assert "No allocation issues detected" not in html


def test_health_separates_the_allocation_alpha_from_interval_levels(
    pricing: DashboardSnapshot,
) -> None:
    html = render_health(pricing).text
    assert "α = 0.001" in html


# Results and metric details.


def test_results_render_every_declared_role_group(storefront: DashboardSnapshot) -> None:
    html = render_results(storefront).text
    for metric in ("checkout_conversion", "revenue_per_user", "session_seconds"):
        assert metric in html


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


# Explore: dispatch boundaries, before any advanced query commits to data.


@pytest.fixture
def storefront_analysis(dashboard_con, dashboard_definitions):
    return _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")


@pytest.fixture
def pricing_analysis(dashboard_con, dashboard_definitions):
    return _analysis(dashboard_con, dashboard_definitions, "pricing_copy")


def test_load_explore_refuses_an_undeclared_metric(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis, snapshot=storefront, metric="not_a_metric", view="daily_values"
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["metric"] == "not_a_metric"


def test_load_explore_refuses_an_unknown_view(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="not_a_view",  # ty: ignore[invalid-argument-type]
        )
    assert caught.value.code == "dashboard.invalid_view"


def test_load_explore_refuses_daily_values_with_a_cumulative_maturity_flag(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    """Daily values are disjoint slices; a cumulative maturity flag cannot be honoured."""
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="daily_values",
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "daily_values"


def test_segment_view_refuses_a_cumulative_maturity_flag(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="segments",
            breakout=("event_log", "country"),
            completed_windows_only=True,
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["view"] == "segments"


@pytest.mark.parametrize("view", ["cumulative_lift", "daily_values", "cumulative_values"])
def test_all_temporal_views_render_every_declared_metric_in_one_table(
    storefront_analysis, storefront: DashboardSnapshot, view: ExploreView
) -> None:
    data = load_explore(storefront_analysis, snapshot=storefront, metric=None, view=view)
    expected = {metric.name for metric in storefront.metrics}
    assert {row.metric for row in data} == expected
    html = render_explore(storefront, data, metric=None, view=view).text
    for metric in expected:
        assert metric in html


def test_all_metric_daily_table_keeps_cohort_and_calendar_rows_distinct(
    storefront_analysis,
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyMetricValues

    data = load_explore(storefront_analysis, snapshot=storefront, metric=None, view="daily_values")
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


def test_load_explore_refuses_a_snapshot_and_source_describing_different_experiments(
    pricing_analysis, storefront: DashboardSnapshot
) -> None:
    """A mismatched pair must not silently render the other binding's stale data."""
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            pricing_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="daily_values",
        )
    assert caught.value.code == "dashboard.invalid_view"
    context = caught.value.context
    assert context["snapshot_experiment"] == "storefront_refresh"
    assert context["analysis_experiment"] == "pricing_copy"


def test_load_explore_segments_need_an_explicit_choice_when_breakouts_are_declared(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis, snapshot=storefront, metric="checkout_conversion", view="segments"
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["declared"] == (("event_log", "country"),)


def test_load_explore_segments_refuse_an_undeclared_breakout(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="segments",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


def test_load_explore_segments_with_no_declared_breakouts_return_empty_and_issue_no_query(
    pricing_analysis, pricing: DashboardSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truthful empty state, not a query the experiment never declared."""

    def _fail(**_: Any) -> Any:
        raise AssertionError("no-breakout experiments must not issue a breakout query")

    monkeypatch.setattr(pricing_analysis, "run_breakout", _fail)
    result = load_explore(
        pricing_analysis, snapshot=pricing, metric="purchase_rate", view="segments"
    )
    assert list(result) == []


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


def test_load_explore_segments_filter_by_dimension_source_and_method_without_leaking_rows(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    """A row from another dimension, source, or method role must never leak in."""
    from increment._source_operations import DashboardExploreCapture
    from increment.breakout.estimates import BreakoutEstimates

    wanted = storefront_analysis.run_breakout(metrics=["checkout_conversion"])
    assert len(wanted) == 2
    decoys = [
        wanted[0].model_copy(update={"dimension": "plan", "dimension_value": "free"}),
        wanted[0].model_copy(update={"method_role": "component"}),
        wanted[0].model_copy(update={"source": "other_source"}),
    ]
    key = ("segments", "checkout_conversion", False, None)
    captured = dataclasses.replace(
        storefront,
        explore={
            **storefront.explore,
            key: DashboardExploreCapture.answered(
                key, wanted + decoys, collection=BreakoutEstimates
            ),
        },
    )
    filtered = load_explore(
        storefront_analysis,
        snapshot=captured,
        metric="checkout_conversion",
        view="segments",
        breakout=("event_log", "country"),
    )
    assert isinstance(filtered, BreakoutEstimates)
    assert len(filtered) == 2
    assert {e.dimension for e in filtered} == {"country"}
    assert {e.source for e in filtered} == {"event_log"}
    assert {e.method_role for e in filtered} == {"decision"}


def test_load_explore_segments_with_no_metric_returns_every_declared_metric(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    """``metric=None`` covers every declared metric via one real breakout call."""
    from increment.breakout.estimates import BreakoutEstimates

    result = load_explore(
        storefront_analysis,
        snapshot=storefront,
        metric=None,
        view="segments",
        breakout=("event_log", "country"),
    )
    assert isinstance(result, BreakoutEstimates)
    declared = tuple(metric.name for metric in storefront.metrics)
    assert declared == ("checkout_conversion", "revenue_per_user", "session_seconds")
    assert {e.metric for e in result} == set(declared)
    # Exactly one non-control arm x two segments (US, CA) per declared metric.
    for metric in declared:
        assert sum(1 for e in result if e.metric == metric) == 2


def test_load_explore_segments_resolve_an_omitted_source_when_unambiguous(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    omitted = dataclasses.replace(storefront, breakouts=((None, "country"),))
    resolved = list(
        load_explore(
            storefront_analysis,
            snapshot=omitted,
            metric="checkout_conversion",
            view="segments",
            breakout=(None, "country"),
        )
    )
    assert len(resolved) == 2
    assert {e.source for e in resolved} == {"event_log"}


def test_load_explore_segments_refuse_an_ambiguously_resolved_source(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    """Two sources under one omitted-source dimension must not be silently merged."""
    from increment._source_operations import DashboardExploreCapture
    from increment.breakout.estimates import BreakoutEstimates

    wanted = storefront_analysis.run_breakout(metrics=["checkout_conversion"])
    ambiguous = wanted + [wanted[0].model_copy(update={"source": "other_source"})]
    key = ("segments", "checkout_conversion", False, None)
    omitted = dataclasses.replace(
        storefront,
        breakouts=((None, "country"),),
        explore={
            **storefront.explore,
            key: DashboardExploreCapture.answered(key, ambiguous, collection=BreakoutEstimates),
        },
    )
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=omitted,
            metric="checkout_conversion",
            view="segments",
            breakout=(None, "country"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["sources"] == ("event_log", "other_source")


# Explore: actual temporal date basis, gap reasons, and monitoring disclosure.


def test_daily_values_report_their_actual_calendar_basis_and_gap_reasons(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    data = load_explore(
        storefront_analysis, snapshot=storefront, metric="checkout_conversion", view="daily_values"
    )
    frame: Any = data.to_frame()
    assert set(frame["ds_basis"].dropna()) == {"calendar"}
    reasons = set(frame["unavailable"].dropna())
    assert reasons == {"nonpositive_mean"}

    html = render_explore(storefront, data, metric="checkout_conversion", view="daily_values").text
    assert "nonpositive_mean" in html


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
    from increment.dashboard._data import _enriched_rows
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
        storefront, estimates=(estimate,), readout_rows=_enriched_rows((estimate,))
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


def _fingerprint(rows) -> list[tuple[Any, ...]]:
    """A trajectory as (segment, day, arm, point) so any slice drift is visible."""
    out = []
    for row in rows:
        point = row.lift if hasattr(row, "lift") else row.value
        out.append(
            (
                row.dimension_value,
                row.ds,
                row.group_id,
                None if point is None else (point.value, point.lb, point.ub, point.level),
            )
        )
    return out


@pytest.mark.parametrize(
    ("view", "engine", "kwargs"),
    [
        ("cumulative_lift", "run_asof_lift", {}),
        ("cumulative_values", "run_asof", {}),
        ("cumulative_values", "run_asof", {"completed_windows_only": True}),
        ("daily_values", "run_daily", {}),
    ],
)
def test_declared_breakout_temporal_views_are_the_engine_segment_trajectories(
    storefront_analysis, storefront: DashboardSnapshot, view: ExploreView, engine: str, kwargs: dict
) -> None:
    metric = "checkout_conversion"
    segmented = load_explore(
        storefront_analysis,
        snapshot=storefront,
        metric=metric,
        view=view,
        breakout=COUNTRY,
        **kwargs,
    )
    direct = getattr(storefront_analysis, engine)(metrics=[metric], dimension="country", **kwargs)
    assert segmented, "a declared breakout must produce a segment trajectory"
    actual, expected = _fingerprint(segmented), _fingerprint(direct)
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row[:3] == expected_row[:3]
        if expected_row[3] is None:
            assert actual_row[3] is None
        else:
            assert actual_row[3] == pytest.approx(expected_row[3])
    assert {row.dimension_value for row in segmented} == {"US", "CA"}
    assert {(row.dimension, row.source) for row in segmented} == {("country", "event_log")}
    whole = load_explore(
        storefront_analysis, snapshot=storefront, metric=metric, view=view, **kwargs
    )
    assert {row.dimension for row in whole} == {None}
    assert _fingerprint(whole) != _fingerprint(segmented)


def test_completed_windows_gate_changes_the_segment_absolute_series(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    def load(*, complete: bool):
        return load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="cumulative_values",
            breakout=COUNTRY,
            completed_windows_only=complete,
        )

    provisional, mature = load(complete=False), load(complete=True)
    assert min(row.ds for row in mature) > min(row.ds for row in provisional)
    assert not any(row.value is None for row in mature)


def test_temporal_breakouts_refuse_an_undeclared_dimension_before_any_query(
    storefront_analysis, storefront: DashboardSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(**_: Any) -> Any:
        raise AssertionError("an undeclared breakout must not reach the engine")

    monkeypatch.setattr(storefront_analysis, "run_asof_lift", _fail)
    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=storefront,
            metric="checkout_conversion",
            view="cumulative_lift",
            breakout=("event_log", "plan"),
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["requested"] == ("event_log", "plan")


def _shadow_source(rows, source: str):
    return type(rows)(row.model_copy(update={"source": source}) for row in rows)


def test_temporal_breakout_honours_the_declared_source_and_refuses_an_ambiguous_one(
    storefront_analysis, storefront: DashboardSnapshot
) -> None:
    """A dimension declared on two sources never silently shows the other source's rows."""
    from increment._source_operations import DashboardExploreCapture

    key = ("cumulative_lift", "checkout_conversion", False, "country")
    rows = load_explore(
        storefront_analysis,
        snapshot=storefront,
        metric="checkout_conversion",
        view="cumulative_lift",
        breakout=COUNTRY,
    )
    both = dataclasses.replace(
        storefront,
        breakouts=(COUNTRY, ("other_log", "country"), (None, "country")),
        explore={
            **storefront.explore,
            key: DashboardExploreCapture.answered(
                key, [*rows, *_shadow_source(rows, "other_log")], collection=type(rows)
            ),
        },
    )

    for source in ("event_log", "other_log"):
        rows = load_explore(
            storefront_analysis,
            snapshot=both,
            breakout=(source, "country"),
            metric="checkout_conversion",
            view="cumulative_lift",
        )
        assert rows and {row.source for row in rows} == {source}

    with pytest.raises(InvalidRequestError) as caught:
        load_explore(
            storefront_analysis,
            snapshot=both,
            breakout=(None, "country"),
            metric="checkout_conversion",
            view="cumulative_lift",
        )
    assert caught.value.code == "dashboard.invalid_view"
    assert caught.value.context["sources"] == ("event_log", "other_log")


@pytest.fixture(scope="module")
def storefront_payload(storefront: DashboardSnapshot, dashboard_con, dashboard_definitions):
    from increment.dashboard._app import build_payload

    analysis = _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")
    return build_payload(analysis, snapshot=storefront)


def test_exploring_never_changes_the_confirmatory_snapshot(
    storefront: DashboardSnapshot, dashboard_con, dashboard_definitions
) -> None:
    from increment.dashboard._app import build_payload

    before = (storefront.estimates, storefront.readout_rows, readout_csv(storefront))
    analysis = _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")
    build_payload(analysis, snapshot=storefront)
    after = (storefront.estimates, storefront.readout_rows, readout_csv(storefront))
    assert after == before


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
    unavailable = health_status(refused)
    assert unavailable["kind"] == "unavailable"
    assert "dashboard.allocation_not_applicable" in unavailable["detail"]
    assert "no assignment table" in unavailable["detail"]

    assert storefront.allocation is not None
    flagged = health_status(_with_allocation(storefront, mixed_assignment_units=3))
    assert flagged["kind"] == "warning"

    assert health_status(_with_allocation(storefront, is_srm=True))["kind"] == "warning"

    healthy = health_status(storefront)
    assert healthy["kind"] == "healthy"


def test_an_engine_refusal_is_visible_for_its_own_state_only(
    dashboard_con, dashboard_definitions, monkeypatch
) -> None:
    from increment.dashboard._app import build_payload

    analysis = _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")
    real = type(analysis).run_asof_lift

    def refuse_segments(self, **kwargs: Any) -> Any:
        if kwargs.get("dimension") is not None:
            raise CapabilityError(
                "segmented history needs retained states",
                code="test.segmented_history",
                context={},
            )
        return real(self, **kwargs)

    monkeypatch.setattr(type(analysis), "run_asof_lift", refuse_segments)
    captured = prepare_dashboard(
        analysis, config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
    )
    payload = build_payload(analysis, snapshot=captured)
    refused = payload["explore"]["country"]["checkout_conversion"]["cumulative_lift"]
    assert refused["pointCount"] == 0
    assert "test.segmented_history" in refused["html"]
    assert payload["explore"]["overall"]["checkout_conversion"]["cumulative_lift"]["pointCount"] > 0
    assert (
        payload["explore"]["country"]["checkout_conversion"]["cumulative_values"]["pointCount"] > 0
    )


def test_unexpected_engine_failures_are_not_swallowed(
    dashboard_con, dashboard_definitions, monkeypatch
) -> None:
    analysis = _analysis(dashboard_con, dashboard_definitions, "storefront_refresh")
    failure = RuntimeError("warehouse connection lost")

    def broken(self, **_: Any) -> Any:
        raise failure

    monkeypatch.setattr(type(analysis), "run_daily", broken)
    with pytest.raises(RuntimeError) as caught:
        prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
        )
    assert caught.value is failure


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
    )


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


def test_segmented_temporal_rendering_accepts_distinct_decision_methods(
    storefront_analysis,
    storefront: DashboardSnapshot,
) -> None:
    from increment.breakout.estimates import DailyLiftEstimates

    data = load_explore(
        storefront_analysis,
        snapshot=storefront,
        metric=None,
        view="cumulative_lift",
        breakout=COUNTRY,
    )
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


@pytest.mark.parametrize("surface", ["readout", "report"])
def test_complete_dashboard_retains_adverse_nonrejecting_guardrail(
    storefront: DashboardSnapshot, storefront_analysis, surface: str
) -> None:
    from increment.dashboard._app import build_payload

    snapshot = _guardrail_snapshot(
        storefront, alternative="greater", direction="increase", lb=-0.10, ub=-0.02
    )
    guardrail = next(row for row in snapshot.readout_rows if row["role"] == "guardrail")
    assert guardrail["stat_sig"] is False
    payload = build_payload(storefront_analysis, snapshot=snapshot)
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
