"""Triggered analysis: declaring a trigger, narrowing the population, and
reporting both effects.

A feature that fires for part of the assigned population has a true effect
and a diluted one. These tests pin that both are reported, and that a
degenerate trigger is refused rather than analyzed.
"""

from __future__ import annotations

from typing import cast

import ibis
import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis
from increment.errors import CapabilityError, DefinitionError, IncrementWarning, InvalidRequestError
from increment.estimation.results import LiftEstimate
from increment.semantics import load
from tests.warning_codes import warning_codes


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


def _defs_yaml(trigger: str | None = "saw_surface", *, allocation_scheme: str | None = None) -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    scheme_line = (
        f"    allocation: {{C: 0.5, T: 0.5}}\n    allocation_scheme: {allocation_scheme}\n"
        if allocation_scheme
        else ""
    )
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
{trigger_line}    unit: user_id
{scheme_line}
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [revenue]
"""


def _write_defs(tmp_path, trigger="saw_surface", *, allocation_scheme=None):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml(trigger, allocation_scheme=allocation_scheme))
    return p


def test_unknown_trigger_refused_at_load(tmp_path):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml("nope"))
    with pytest.raises(DefinitionError) as raised:
        load(p)
    assert raised.value.code == "definition.invalid"


def test_trigger_equal_to_exposure_refused_at_load(tmp_path):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml("assignment"))
    with pytest.raises(DefinitionError) as raised:
        load(p)
    assert raised.value.code == "definition.invalid"


def test_trigger_declaration_loads(tmp_path):
    defs = load(_write_defs(tmp_path))
    exp = defs.experiment("exp")
    assert exp is not None
    assert exp.trigger == "saw_surface"


def _events(
    n_per_arm=2000,
    trigger_rate=0.2,
    effect=0.5,
    seed=5,
    treatment_trigger_rate=None,
):
    """Only triggered treatment units are affected. The assigned-population
    lift is therefore the true lift times the trigger rate."""

    rng = np.random.default_rng(seed)
    rows = {"user_id": [], "group_id": [], "ts": [], "event": [], "value": [], "experiment_id": []}
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    revenue_ts = np.datetime64("2024-01-08T00:00:00")
    freshness_ts = np.datetime64("2024-01-09T00:00:00")

    def add(uid, arm, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        arm_trigger_rate = (
            treatment_trigger_rate
            if arm == "T" and treatment_trigger_rate is not None
            else trigger_rate
        )
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            triggered = rng.random() < arm_trigger_rate
            add(uid, arm, "enrolled", 0.0, enrolled_ts)
            if triggered:
                add(uid, arm, "saw_surface", 0.0, enrolled_ts)
            base = rng.lognormal(0.0, 0.4)
            lift = effect if (arm == "T" and triggered) else 0.0
            add(uid, arm, "revenue", base * (1.0 + lift), revenue_ts)
            add(uid, arm, "revenue", base * (1.0 + lift), freshness_ts)
    return pa.table(rows)


def _analysis(
    tmp_path, table, trigger="saw_surface", *, on_mixed_assignment="error", allocation_scheme=None
):
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    return Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, trigger, allocation_scheme=allocation_scheme),
        con,
        on_mixed_assignment=on_mixed_assignment,
    )


def test_reports_both_populations_and_the_triggered_effect_is_undiluted(tmp_path):
    rate, effect = 0.2, 0.5
    an = _analysis(tmp_path, _events(trigger_rate=rate, effect=effect))
    rows = an.run()

    by_pop = {r.analysis_population: r for r in rows if r.metric == "revenue"}
    assert set(by_pop) == {"assigned", "triggered"}

    assigned, triggered = by_pop["assigned"], by_pop["triggered"]
    # The assigned reading is the true effect diluted by the trigger rate.
    assert assigned.require_lift().value == pytest.approx(effect * rate, rel=0.25)
    # The triggered reading recovers the real thing.
    assert triggered.require_lift().value == pytest.approx(effect, rel=0.12)
    assert triggered.require_lift().value > assigned.require_lift().value

    # The table adapter must carry the population axis through the same
    # assigned/triggered result pair; otherwise the two rows collide in the
    # rendered metric key.
    from increment.tables import estimates_to_readout, readout_table

    readout_rows = estimates_to_readout([assigned, triggered])
    assert [row["analysis_population"] for row in readout_rows] == ["assigned", "triggered"]
    pytest.importorskip("coeftable")
    html = readout_table(readout_rows).gt().as_raw_html()
    assert "(assigned)" in html
    assert "(triggered)" in html


def test_observed_trigger_rate_is_reported(tmp_path):
    """The planner needs the measured rate to size a follow-up."""
    an = _analysis(tmp_path, _events(trigger_rate=0.2))
    rates = an.trigger_rates()
    assert set(rates) == {"C", "T"}
    for arm in ("C", "T"):
        assert rates[arm] == pytest.approx(0.2, abs=0.03)


@pytest.mark.parametrize(
    ("policy", "warning"),
    [("error", False), ("warn", True), ("exclude", False)],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_trigger_rates_enforces_assignment_policy(tmp_path, policy, warning):
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(trigger_rate=0.2))
    an = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path),
        con,
        on_mixed_assignment=policy,
    )
    con.raw_sql("UPDATE events SET group_id = NULL WHERE user_id = 'C0' AND event = 'enrolled'")

    if policy == "error":
        with pytest.raises(InvalidRequestError) as caught:
            an.trigger_rates()
        assert caught.value.code == "query.integrity.unassigned_assignment_units"
        return

    if warning:
        with pytest.warns(IncrementWarning) as rec:
            rates = an.trigger_rates()
        assert "query.integrity.mixed_assignments_excluded" in warning_codes(rec)
    else:
        rates = an.trigger_rates()
    assert set(rates) == {"C", "T"}


def test_one_arm_trigger_refused(tmp_path):
    """A trigger only the treatment arm can produce is the treatment."""
    table = _events(trigger_rate=0.2)
    events = table.column("event").to_pylist()
    groups = table.column("group_id").to_pylist()
    keep = [
        i for i in range(table.num_rows) if not (events[i] == "saw_surface" and groups[i] == "C")
    ]
    an = _analysis(tmp_path, table.take(keep))
    with pytest.raises(InvalidRequestError) as caught:
        an.run()
    assert caught.value.code == "query.integrity.trigger_arm_missing"


def test_no_trigger_declared_leaves_every_row_assigned(tmp_path):
    """Backwards compatibility: an experiment with no trigger behaves
    exactly as before and reports one row per metric."""
    an = _analysis(tmp_path, _events(), trigger=None)
    rows = [r for r in an.run() if r.metric == "revenue"]
    assert len(rows) == 1
    assert rows[0].analysis_population == "assigned"


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("sitewide", ("revenue",)),  # sitewide takes a required metric_name
        ("run_daily", ()),
        ("run_daily_lift", ()),
        ("run_asof", ()),
        ("run_asof_lift", ()),
        ("run_breakout", ()),
        ("breakout_summaries", ()),
        ("factor_summaries", ()),
    ],
)
def test_unsupported_readouts_refuse_a_declared_trigger(tmp_path, method_name, args):
    an = _analysis(tmp_path, _events(n_per_arm=100))
    with pytest.raises(CapabilityError) as raised:
        getattr(an, method_name)(*args)
    assert raised.value.code == "facade.analysis.trigger_unsupported"


@pytest.mark.parametrize("grain", ["daily", "asof"])
def test_triggered_day_axis_moments_refuse_instead_of_reading_assigned(tmp_path, grain):
    analysis = _analysis(tmp_path, _events(n_per_arm=100))
    method = analysis.run_daily if grain == "daily" else analysis.run_asof
    with pytest.raises(CapabilityError) as raised:
        method()
    assert raised.value.code == "facade.analysis.trigger_unsupported"


def test_srm_population_triggered_counts_the_narrowed_population(tmp_path):
    """srm(population='triggered') is a real guardrail: a trigger
    correlated with assignment shows up as an allocation imbalance there
    even when enrollment itself is balanced."""
    an = _analysis(tmp_path, _events(trigger_rate=0.2, n_per_arm=500))
    expected = {"C": 0.5, "T": 0.5}
    assigned = an.srm(expected=expected)
    triggered = an.srm(expected=expected, population="triggered")
    assert sum(assigned.observed.values()) == 1000
    assert sum(triggered.observed.values()) < sum(assigned.observed.values())


def test_triggered_run_does_not_retest_assigned_law_on_selected_counts(tmp_path):
    analysis = _analysis(
        tmp_path,
        _events(n_per_arm=500, trigger_rate=0.1, treatment_trigger_rate=0.9),
        allocation_scheme="independent",
    )

    results = analysis.run()

    scopes = {
        scope.rosters[0].analysis_population: scope
        for scope in results.metadata.scope.by_source.values()
    }
    assert scopes["assigned"].integrity[0].status == "not_rejected"
    assert scopes["assigned"].integrity[0].analysis_population == "assigned"
    (triggered_integrity,) = scopes["triggered"].integrity
    assert triggered_integrity.status == "not_checked_missing_counts"
    assert triggered_integrity.analysis_population == "triggered"
    assert triggered_integrity.observed is None
    triggered_only = results.filter(lambda row: row.analysis_population == "triggered")
    retained = next(
        scope
        for scope in triggered_only.metadata.scope.by_source.values()
        if scope.rosters[0].analysis_population == "assigned"
    )
    assert retained.integrity[0] == scopes["assigned"].integrity[0]


def test_registered_sequential_trigger_run_reuses_assigned_integrity(monkeypatch, tmp_path):
    import datetime as dt

    from increment.readouts import _sequential_scope
    from tests.binary_sequential_cases import definitions_yaml, event_rows, unit_rows
    from tests.sequential_cases import registered_native

    units = unit_rows(seed=17, n=500)
    rng = np.random.default_rng(51)
    for unit in units:
        threshold = 0.1 if unit["variant"] == "control" else 0.9
        unit["triggered"] = rng.random() < threshold
    events = event_rows(units)
    trigger_time = dt.datetime(2025, 1, 10, 10, tzinfo=dt.UTC)
    events.extend(
        {
            "user_id": unit["user_id"],
            "ts": trigger_time,
            "event": "saw_surface",
            "experiment_id": None,
            "group_id": None,
        }
        for unit in units
        if unit["triggered"]
    )
    definitions = (
        definitions_yaml("duckdb", "events", plan="      primary: purchase\n")
        .replace(
            "      - {name: buy, column: null}",
            "      - {name: buy, column: null}\n      - {name: saw_surface, column: null}",
        )
        .replace(
            "  - {name: enrollment, fact: enrolled}",
            "  - {name: enrollment, fact: enrolled}\n  - {name: saw_surface, fact: saw_surface}",
        )
        .replace(
            "    allocation: {control: 0.5, treatment: 0.5}",
            "    allocation: {control: 0.5, treatment: 0.5}\n"
            "    allocation_scheme: independent\n    trigger: saw_surface",
        )
    )
    definitions_path = tmp_path / "sequential-trigger.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(events))
    analysis = registered_native(Analysis.from_definitions("exp", definitions_path, con))
    analysis.capture_sequential(finalized=True, as_of=dt.date(2025, 1, 14))

    source_type = type(analysis._src)
    original_counts = source_type.assignment_counts
    count_calls = []

    def counted_counts(self, *, population="assigned"):
        count_calls.append(population)
        return original_counts(self, population=population)

    original_integrity = _sequential_scope.assignment_integrity
    integrity_calls = []

    def counted_integrity(*args, **kwargs):
        integrity_calls.append(kwargs.get("counts", args[1] if len(args) > 1 else None))
        return original_integrity(*args, **kwargs)

    monkeypatch.setattr(source_type, "assignment_counts", counted_counts)
    monkeypatch.setattr(_sequential_scope, "assignment_integrity", counted_integrity)
    results = analysis.run()

    (scope,) = results.metadata.scope.by_source.values()
    (integrity,) = scope.integrity
    assert count_calls == ["assigned"]
    assert len(integrity_calls) == 1
    assert integrity.analysis_population == "assigned"
    assert integrity.observed == {"control": 500, "treatment": 500}
    assert any(
        row.analysis_population == "triggered"
        and row.failure_code == "readout.cell.unsupported_request"
        for row in results
    )
    assert results.source["components"][-1]["kind"] == "assignment_counts"
    import datetime as dt

    from tests.binary_sequential_cases import definitions_yaml, event_rows, unit_rows
    from tests.sequential_cases import registered_native

    units = unit_rows(seed=17, n=500)
    rng = np.random.default_rng(51)
    for unit in units:
        threshold = 0.1 if unit["variant"] == "control" else 0.9
        unit["triggered"] = rng.random() < threshold
    events = event_rows(units)
    trigger_time = dt.datetime(2025, 1, 10, 10, tzinfo=dt.UTC)
    events.extend(
        {
            "user_id": unit["user_id"],
            "ts": trigger_time,
            "event": "saw_surface",
            "experiment_id": None,
            "group_id": None,
        }
        for unit in units
        if unit["triggered"]
    )
    definitions = (
        definitions_yaml("duckdb", "events", plan="      primary: purchase\n")
        .replace("      - {name: buy, column: null}", "      - {name: buy, column: null}\n      - {name: saw_surface, column: null}")
        .replace("  - {name: enrollment, fact: enrolled}", "  - {name: enrollment, fact: enrolled}\n  - {name: saw_surface, fact: saw_surface}")
        .replace(
            "    allocation: {control: 0.5, treatment: 0.5}",
            "    allocation: {control: 0.5, treatment: 0.5}\n"
            "    allocation_scheme: independent\n    trigger: saw_surface",
        )
    )
    definitions_path = tmp_path / "sequential-trigger.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(events))
    analysis = registered_native(Analysis.from_definitions("exp", definitions_path, con))
    analysis.capture_sequential(finalized=True, as_of=dt.date(2025, 1, 14))

    source_type = type(analysis._src)
    original = source_type.assignment_counts
    calls = []

    def counted(self, *, population="assigned"):
        calls.append(population)
        return original(self, population=population)

    monkeypatch.setattr(source_type, "assignment_counts", counted)
    results = analysis.run()

    (scope,) = results.metadata.scope.by_source.values()
    (integrity,) = scope.integrity
    assert calls == ["assigned"]
    assert integrity.analysis_population == "assigned"
    assert integrity.observed == {"control": 500, "treatment": 500}
    assert any(
        row.analysis_population == "triggered"
        and row.failure_code == "readout.cell.unsupported_request"
        for row in results
    )


def test_srm_population_triggered_refuses_on_a_seam_instance():
    import pandas as pd

    from increment import Analysis
    from increment.errors import CapabilityError

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    a = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    with pytest.raises(CapabilityError) as raised:
        a.srm(population="triggered")
    assert raised.value.code == "facade.analysis.operation"


def test_srm_rejects_unknown_population_before_assigned_readout():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    with pytest.raises(InvalidRequestError) as caught:
        analysis.srm(population="trigger")  # ty: ignore[invalid-argument-type]
    assert caught.value.code == "facade.analysis.invalid_population"
    assert caught.value.context["population"] == "trigger"


def _defs_yaml_clustered(trigger: str | None = "saw_surface") -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM cluster_trigger_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
{trigger_line}    unit: user_id
    cluster: store_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [revenue]
"""


def _clustered_trigger_events(
    n_stores_per_arm=20,
    units_per_store=5,
    trigger_rate=0.2,
    effect=0.5,
    seed=5,
):
    """Same dilution shape as `_events`, but store-clustered: every row
    also carries a `store_id`, and a store's units share triggered status
    so the population narrowing and the cluster grouping are both live."""

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "user_id": [],
        "group_id": [],
        "store_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    revenue_ts = np.datetime64("2024-01-08T00:00:00")
    freshness_ts = np.datetime64("2024-01-09T00:00:00")

    def add(uid, arm, store, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["store_id"].append(store)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        for s in range(n_stores_per_arm):
            store = f"{arm}{s}"
            store_triggered = rng.random() < trigger_rate
            arm_units_per_store = (
                units_per_store[arm] if isinstance(units_per_store, dict) else units_per_store
            )
            for u in range(arm_units_per_store):
                uid = f"{store}_{u}"
                add(uid, arm, store, "enrolled", 0.0, enrolled_ts)
                if store_triggered:
                    add(uid, arm, store, "saw_surface", 0.0, enrolled_ts)
                base = rng.lognormal(0.0, 0.4)
                lift = effect if (arm == "T" and store_triggered) else 0.0
                add(uid, arm, store, "revenue", base * (1.0 + lift), revenue_ts)
                add(uid, arm, store, "revenue", base * (1.0 + lift), freshness_ts)
    return pa.table(rows)


def test_clustered_experiment_with_trigger_narrows_to_the_triggered_population(tmp_path):
    """A clustered experiment routes its triggered readout through the
    source-owned triggered-population operation."""
    rate, effect = 0.5, 0.5
    con = ibis.duckdb.connect()
    con.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(n_stores_per_arm=40, trigger_rate=rate, effect=effect),
    )
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml_clustered())
    an = Analysis.from_definitions("exp", defs_path, con)
    rows = _lift_rows(an.run())
    by_pop = {r.analysis_population: r for r in rows if r.metric == "revenue"}
    assert set(by_pop) == {"assigned", "triggered"}
    assigned, triggered = by_pop["assigned"], by_pop["triggered"]

    # Assigned lift is the effect diluted by the trigger rate; triggered lift
    # recovers it. Equal or reversed values would mean both read the assigned source.
    assert triggered.require_lift().value > assigned.require_lift().value
    assert triggered.require_lift().value == pytest.approx(effect, rel=0.3)


def _defs_yaml_quantile(trigger: str | None = "saw_surface") -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM quantile_trigger_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
experiments:
  - name: exp
    exposure: assignment
{trigger_line}    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [p90_latency]
"""


def _quantile_trigger_events(n_per_arm=300, trigger_rate=0.3, shift=1.6, seed=7):
    """Only triggered treatment units see the latency shift, mirroring
    `_events`'s dilution shape for the quantile per-unit path."""

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "user_id": [],
        "group_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-02T01:00:00")

    def add(uid, arm, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            triggered = rng.random() < trigger_rate
            add(uid, arm, "enrolled", 0.0, enrolled_ts)
            if triggered:
                add(uid, arm, "saw_surface", 0.0, enrolled_ts)
            mult = shift if (arm == "T" and triggered) else 1.0
            add(uid, arm, "latency", rng.lognormal(0.0, 0.5) * mult, metric_ts)
    return pa.table(rows)


def test_quantile_experiment_with_trigger_narrows_to_the_triggered_population(tmp_path):
    """A quantile experiment routes its triggered unit-frame readout through
    the source-owned triggered-population operation."""
    con = ibis.duckdb.connect()
    con.create_table("quantile_trigger_events", obj=_quantile_trigger_events())
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml_quantile())
    an = Analysis.from_definitions("exp", defs_path, con)
    rows = _lift_rows(an.run())
    by_pop = {r.analysis_population: r for r in rows if r.metric == "p90_latency"}
    assert set(by_pop) == {"assigned", "triggered"}
    assigned, triggered = by_pop["assigned"], by_pop["triggered"]

    # Public assigned/triggered rows prove the native routing.
    # A triggered-only shift diluted by the trigger rate in the assigned
    # reading, undiluted in the triggered one -- equal values would mean
    # both were read off the same (assigned) source.
    assert triggered.require_lift().value > assigned.require_lift().value
