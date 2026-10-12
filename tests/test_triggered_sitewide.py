from __future__ import annotations

import copy
import datetime as dt
import math
import pickle

import ibis
import pyarrow as pa
import pytest
from scipy.stats import norm

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import CapabilityError
from increment.estimation.sitewide import SitewideRatioImpact
from increment.query.artifact_contract import ArtifactContractError
from tests.analysis_factory import _native_source, native_connection
from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
from tests.test_analysis_trigger import _defs_yaml, _events


def _analysis(tmp_path, events=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definitions = tmp_path / "triggered-sitewide.yml"
    definitions.write_text(_defs_yaml(end="2024-01-10").replace("    window_days: 7\n", ""))
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=_events(n_per_arm=100, trigger_rate=0.2, effect=0.5) if events is None else events,
    )
    return Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )


def _add_late_trigger_and_outcome(events, *, include_outcome):
    rows = events.to_pylist()
    triggered = {row["user_id"] for row in rows if row["event"] == "saw_surface"}
    unit_id = next(f"C{index}" for index in range(100) if f"C{index}" not in triggered)
    extra = [
        {
            "user_id": unit_id,
            "group_id": "C",
            "ts": dt.datetime(2024, 1, 12),
            "event": "saw_surface",
            "value": 0.0,
            "experiment_id": "exp",
        }
    ]
    if include_outcome:
        extra.append(
            {
                "user_id": unit_id,
                "group_id": "C",
                "ts": dt.datetime(2024, 1, 12),
                "event": "revenue",
                "value": 1_000_000.0,
                "experiment_id": "exp",
            }
        )
    return pa.concat_tables([events, pa.Table.from_pylist(extra, schema=events.schema)])


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_evidence_uses_calendar_window_member_outcomes(tmp_path):
    base_events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    member_only_events = _add_late_trigger_and_outcome(base_events, include_outcome=False)
    with_late_outcome_events = _add_late_trigger_and_outcome(base_events, include_outcome=True)
    member_only = _analysis(tmp_path / "member-only", member_only_events)
    with_late_outcome = _analysis(tmp_path / "late-outcome", with_late_outcome_events)
    try:
        source = with_late_outcome._src
        evidence = source.sitewide_evidence(
            with_late_outcome.metrics[0], include_ratio=False, population="triggered"
        )
        reference = member_only._src.sitewide_evidence(
            member_only.metrics[0], include_ratio=False, population="triggered"
        )
        assert evidence.analysis_population == "triggered"
        assert evidence.window == (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert evidence.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert evidence.population_counts == source.assignment_counts(population="triggered")
        assert evidence.assigned_counts == {"C": 100, "T": 100}
        assert evidence.population_counts["C"] == 17
        assert evidence.site_total == pytest.approx(
            with_late_outcome._src.sitewide_evidence(with_late_outcome.metrics[0]).site_total
        )
        assert {arm.group_id: arm.n for arm in evidence.arm_stats} == {
            arm.group_id: arm.n for arm in reference.arm_stats
        }
        assert {arm.group_id: arm.mean_y() for arm in evidence.arm_stats} == pytest.approx(
            {arm.group_id: arm.mean_y() for arm in reference.arm_stats}
        )
    finally:
        member_only.close()
        with_late_outcome.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_matches_native_evidence(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        native = _native_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = adopted._src.sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="triggered"
        )
        assert artifact.population_counts == native.population_counts
        assert artifact.assigned_counts == native.assigned_counts
        assert artifact.site_total == pytest.approx(native.site_total, rel=1e-12)
        assert {arm.group_id: arm.n for arm in artifact.arm_stats} == {
            arm.group_id: arm.n for arm in native.arm_stats
        }
        assert {arm.group_id: arm.mean_y() for arm in artifact.arm_stats} == pytest.approx(
            {arm.group_id: arm.mean_y() for arm in native.arm_stats}, rel=1e-12
        )
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_without_membership_refuses_with_republish_route(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)), analysis, kinds=("site_volume",)
        )
        with pytest.raises(ArtifactContractError) as raised:
            adopted._src.sitewide_evidence(
                adopted.metrics[0], include_ratio=False, population="triggered"
            )
        for restored in (
            copy.deepcopy(raised.value),
            pickle.loads(pickle.dumps(raised.value)),
        ):
            assert restored.code == raised.value.code
            assert restored.context == raised.value.context
        assert raised.value.code == "artifact.extension.missing"
        assert "republish" in str(raised.value.context["route"])
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


def test_triggered_sitewide_windowed_metric_refuses_before_data_load(tmp_path, monkeypatch):
    definitions = tmp_path / "windowed.yml"
    definitions.write_text(_defs_yaml(end="2024-01-10"))
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20))
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
        ),
    )
    try:
        monkeypatch.setattr(
            analysis._src,
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("loaded incompatible sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            analysis.sitewide("revenue", population="triggered")
        assert refused.value.code == "facade.analysis.sitewide_triggered_metric_unaligned"
        assert "calendar-aligned" in str(refused.value)
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_result_discloses_population_and_provenance(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        assigned = analysis.sitewide("revenue")
        result = analysis.sitewide("revenue", population="triggered")
        assert result.analysis_population == "triggered"
        assert result.window == (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert result.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert result.snapshot_id
        assert result.window == assigned.window
        assert result.snapshot_id == assigned.snapshot_id
        assert result.scaled_units == 33
        assert result.multiplicity_status == "exploratory_unadjusted"
        assert result.adjustment == "unadjusted"
        assert result.assumptions == (
            "membership_independent_of_treatment",
            "never_triggered_unaffected",
            "enrolled_units_exhaust_exposure",
        )
        assert result.allocation_balance is not None
        assert result.allocation_balance.status in {"passed", "failed"}
        assert "population='triggered'" in repr(result)
        assert "scaled_units=33" in repr(result)
        cls = type(result)
        assert cls.model_validate_json(result.model_dump_json()) == result
        assert pickle.loads(pickle.dumps(result)) == result
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_ratio_component_baseline(tmp_path):
    definitions = tmp_path / "ratio.yml"
    yaml = (
        _defs_yaml(end="2024-01-10")
        .replace(
            "      - name: saw_surface\n        column: null\n",
            "      - name: saw_surface\n        column: null\n"
            "      - name: session\n        column: value\n",
        )
        .replace(
            "  - name: revenue\n    type: mean\n    entity: user_id\n"
            "    fact: revenue\n    aggregation: sum\n    window_days: 7\n",
            "  - name: revenue_per_session\n    type: ratio\n    entity: user_id\n"
            "    numerator: {fact: revenue, aggregation: sum}\n"
            "    denominator: {fact: session, aggregation: sum}\n",
        )
    )
    yaml = yaml.replace("secondaries: [revenue]", "secondaries: [revenue_per_session]")
    definitions.write_text(yaml)
    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    triggered = {row["user_id"] for row in rows if row["event"] == "saw_surface"}
    for unit_id in {row["user_id"] for row in rows if row["event"] == "enrolled"}:
        value = 2.0 if unit_id.startswith("T") and unit_id in triggered else 1.0
        rows.append(
            {
                "user_id": unit_id,
                "group_id": unit_id[0],
                "ts": dt.datetime(2024, 1, 8),
                "event": "session",
                "value": value,
                "experiment_id": "exp",
            }
        )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        evidence = analysis._src.sitewide_evidence(
            analysis.metrics[0], include_ratio=True, population="triggered"
        )
        result = analysis.sitewide("revenue_per_session", population="triggered")
        assert isinstance(result, SitewideRatioImpact)
        arms = {arm.group_id: arm for arm in evidence.arm_stats}
        control, treatment = arms["C"], arms["T"]
        delta_num = treatment.mean_y() - control.mean_y()
        delta_den = treatment.mean_den() - control.mean_den()
        n_t = treatment.n
        n_total = sum(evidence.population_counts.values())
        assert evidence.site_total_denominator is not None
        n0 = evidence.site_total - n_t * delta_num
        d0 = evidence.site_total_denominator - n_t * delta_den
        n1 = n0 + n_total * delta_num
        d1 = d0 + n_total * delta_den
        assert delta_den != pytest.approx(0)
        assert result.baseline_numerator == pytest.approx(n0, rel=1e-12)
        assert result.baseline_denominator == pytest.approx(d0, rel=1e-12)
        assert result.shipped_numerator == pytest.approx(n1, rel=1e-12)
        assert result.shipped_denominator == pytest.approx(d1, rel=1e-12)
        assert result.baseline_ratio == pytest.approx(n0 / d0, rel=1e-12)
        assert result.shipped_ratio == pytest.approx(n1 / d1, rel=1e-12)
        assert type(result).model_validate_json(result.model_dump_json()) == result
        assert pickle.loads(pickle.dumps(result)) == result
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_sum_matches_hand_computation(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        evidence = analysis._src.sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        result = analysis.sitewide("revenue", population="triggered")
        arms = {arm.group_id: arm for arm in evidence.arm_stats}
        control = arms["C"]
        treatment = arms["T"]
        delta = treatment.mean_y() - control.mean_y()
        delta_se = math.sqrt(treatment.var_y() / treatment.n + control.var_y() / control.n)
        scaled_units = sum(evidence.population_counts.values())
        baseline = evidence.site_total - delta * treatment.n
        absolute = delta * scaled_units
        relative = absolute / baseline
        z = norm.ppf(0.975)
        absolute_se = delta_se * scaled_units
        relative_se = abs(scaled_units * evidence.site_total / baseline**2) * delta_se
        assert result.absolute_impact == pytest.approx(absolute, rel=1e-12)
        assert result.baseline_volume == pytest.approx(baseline, rel=1e-12)
        assert result.relative_impact == pytest.approx(relative, rel=1e-12)
        assert result.absolute_impact_se == pytest.approx(absolute_se, rel=1e-12)
        assert result.relative_impact_se == pytest.approx(relative_se, rel=1e-12)
        assert result.absolute_impact_lb == pytest.approx(absolute - z * absolute_se, rel=1e-12)
        assert result.absolute_impact_ub == pytest.approx(absolute + z * absolute_se, rel=1e-12)
        assert result.relative_impact_lb == pytest.approx(relative - z * relative_se, rel=1e-12)
        assert result.relative_impact_ub == pytest.approx(relative + z * relative_se, rel=1e-12)
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_assigned_and_triggered_share_provenance(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        assigned = analysis.sitewide("revenue")
        triggered = analysis.sitewide("revenue", population="triggered")
        assert assigned.window == triggered.window
        assert assigned.snapshot_id == triggered.snapshot_id
        assert assigned.membership_cutoff is None
        assert triggered.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert assigned.assumptions == ("enrolled_units_exhaust_exposure",)
        assert assigned.allocation_balance is None
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_failed_balance_is_qualified_not_refused(tmp_path):
    rows = _events(n_per_arm=100, trigger_rate=1.0, effect=0.5).to_pylist()
    filtered = [
        row
        for row in rows
        if row["event"] != "saw_surface"
        or row["group_id"] == "C"
        or int(row["user_id"][1:]) % 20 == 0
    ]
    analysis = _analysis(tmp_path, pa.Table.from_pylist(filtered))
    try:
        result = analysis.sitewide("revenue", population="triggered")
        assert result.allocation_balance is not None
        assert result.allocation_balance.status == "failed"
    finally:
        analysis.close()
