"""Native source count and quantile behavior."""

from __future__ import annotations

import ibis
import numpy as np
import pytest

from increment.analysis import Analysis
from increment.estimation.quantile import estimate_quantile_lift
from increment.semantics.models import AnalysisPlan, MultiplicitySpec
from tests.analysis_factory import (
    _make_event_log_table,
    _native_source,
    lift_rows,
    make_analysis_like,
)


@pytest.fixture(scope="module")
def pricing_con():
    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    return con


@pytest.fixture
def pricing_analysis(pricing_con):
    return Analysis.from_definitions("pricing_tier_test", "examples/definitions/", pricing_con)


def test_unit_counts_and_cluster_counts_match_srm(pricing_analysis):
    """unit_counts() (the MomentSource protocol's per-arm count) matches
    the enrolled counts srm() reports natively, and cluster_counts()
    refuses on this undeclared-cluster experiment the same way srm()'s
    own clustered branch would."""
    src = _native_source(pricing_analysis)
    srm_result = pricing_analysis.srm(expected={"control": 0.5, "treatment": 0.5})
    assert srm_result.observed
    assert src.unit_counts() == srm_result.observed

    from increment.errors import CapabilityError

    with pytest.raises(CapabilityError):
        src.cluster_counts()


def _quantile_defs_and_con(tmp_path, *, preferred_direction: str | None = None):
    direction_line = (
        f"    preferred_direction: {preferred_direction}\n" if preferred_direction else ""
    )
    yaml = f"""
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
{direction_line}experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan: {{secondaries: [p90_latency]}}
"""
    rng = np.random.default_rng(11)
    n = 200
    control = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    treatment = rng.lognormal(mean=0.0, sigma=0.6, size=n) * 1.4
    users = [f"c{i}" for i in range(n)] + [f"t{i}" for i in range(n)]
    groups = ["C"] * n + ["T"] * n
    values = np.concatenate([control, treatment])
    enroll_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-02T01:00:00")
    rows = []
    for user, group, value in zip(users, groups, values, strict=True):
        common = {"user_id": user, "experiment_id": "exp", "group_id": group}
        rows.append({**common, "ts": enroll_ts, "event": "enrolled", "value": None})
        rows.append({**common, "ts": metric_ts, "event": "latency", "value": float(value)})

    con = ibis.duckdb.connect()
    con.create_table("events", obj=rows)
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(yaml)
    return defs_path, con


def test_quantile_parity(tmp_path):
    """unit_frame()-served per-unit rows, fed to estimate_quantile_lift
    directly, must reproduce Analysis.run()'s own quantile row - the
    D4-fallback path a quantile metric always needs (quantiles have no
    moments representation, so readouts.run's src.moments() route never
    serves them). The declared plan's sole secondary is a strong enough
    shift to always be BH-selected, so `baseline` is FCR re-estimated at
    `1 - baseline.lift.level` rather than the plan's nominal alpha -- the
    direct `estimate_quantile_lift` call below matches that level
    explicitly, since this test is about `unit_frame()` data-serving
    parity, not about re-deriving the FCR alpha itself."""
    defs_path, con = _quantile_defs_and_con(tmp_path)
    analysis = Analysis.from_definitions("exp", defs_path, con)
    baseline = lift_rows(analysis.run())[0]
    assert baseline.discovery is True
    assert baseline.require_lift().level is not None
    # A single in-family secondary always selects (R=1, m=1, q=0.10 default),
    # so BH's own cutoff is exactly q=0.10 -- double the nominal 0.05, which
    # would cut a NARROWER interval than an uncorrected read. Capped at the
    # nominal alpha, with BH's cutoff still recorded for disclosure.
    baseline_lift = baseline.require_lift()
    assert baseline_lift.level == pytest.approx(0.95)
    assert baseline_lift.level is not None
    assert baseline.family_threshold == pytest.approx(0.10)
    assert baseline.family_q == pytest.approx(0.10)

    src = _native_source(analysis)
    metric = analysis.metrics[0]
    unit_rows = src.unit_frame(metric)
    (candidate,) = estimate_quantile_lift(
        type("_Src", (), {"unit_frame": staticmethod(lambda *a, **k: unit_rows)})(),
        metric,
        analysis.experiment.control_group,
        alpha=1.0 - baseline_lift.level,
    )
    candidate_lift = candidate.require_lift()
    assert candidate_lift.value == pytest.approx(baseline_lift.value, rel=1e-9)
    assert candidate_lift.lb == pytest.approx(baseline_lift.lb, rel=1e-9)
    assert candidate_lift.ub == pytest.approx(baseline_lift.ub, rel=1e-9)


def test_quantile_explicit_preferred_direction_flows_through_infer_lift(tmp_path):
    """Finding 3: estimate_quantile_lift never forwarded preferred_direction
    to infer_lift at all -- even an EXPLICITLY declared direction (e.g. p90
    latency wanting "decrease") silently produced preferred_direction=None
    on the readout, making the field inert for the metric type most likely
    to need it (the canonical latency-guardrail case)."""
    defs_path, con = _quantile_defs_and_con(tmp_path, preferred_direction="decrease")
    analysis = Analysis.from_definitions("exp", defs_path, con)
    est = lift_rows(analysis.run())[0]
    assert est.preferred_direction == "decrease"
    assert est.prob_favorable() == pytest.approx(1.0 - est.prob_beyond(est.null_lift))
    assert est.prob_favorable() != pytest.approx(est.prob_beyond(est.null_lift))


def _cluster_defs_and_con(tmp_path, *, cluster: bool):
    import datetime as dt

    template = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM cl_events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - name: exposure
        column: null
      - name: purchase
        column: revenue
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: revenue_per_user
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: store_test
    exposure: enrolled
    unit: user_id
{cluster_line}
    start: 2025-08-01
    end: 2025-08-07
    plan: {{secondaries: [revenue_per_user]}}
    control_group: control
"""
    n_stores, units_per_store = 12, 2
    rows = []
    for arm, base in (("control", 5.0), ("treatment", 5.5)):
        for s in range(n_stores):
            store = f"{arm[:1]}s{s}"
            for u in range(units_per_store):
                unit = f"{arm[:1]}{s}_{u}"
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": dt.datetime(2025, 8, 1, 9, 0, 0),
                        "event": "exposure",
                        "experiment_id": "store_test",
                        "group_id": arm,
                        "store_id": store,
                        "revenue": None,
                    }
                )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": dt.datetime(2025, 8, 7, 10, 0, 0),
                        "event": "purchase",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "revenue": base + 0.2 * s + 0.05 * u,
                    }
                )
    con = ibis.duckdb.connect()
    con.create_table("cl_events", obj=rows)
    cluster_line = "    cluster: store_id" if cluster else ""
    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(template.format(cluster_line=cluster_line))
    return defs_path, con


@pytest.mark.filterwarnings(
    "ignore:metric .* total clusters across arms .* below 40:RuntimeWarning"
)
def test_clustered_native_path_undeclared_metric_preferred_direction_is_none(tmp_path):
    """Finding 2: `_build_for_metrics`'s shared `estimate_lift(...)` call
    defaulted `preferred_direction="increase"` for every undeclared metric
    stamp. `Analysis.run()` dispatches a clustered Definitions experiment
    through the shared readout pipeline (the warehouse source directly,
    since the moments cube carries no cluster marker), so an
    undeclared metric silently reported "increase" there too."""
    defs_path, con = _cluster_defs_and_con(tmp_path, cluster=True)
    analysis = Analysis.from_definitions("store_test", defs_path, con)
    (baseline,) = lift_rows(analysis.run())
    assert baseline.preferred_direction is None


@pytest.mark.filterwarnings(
    "ignore:metric .* total clusters across arms .* below 40:RuntimeWarning"
)
def test_clustered_run_refuses_call_time_policy_on_a_declared_plan(tmp_path):
    """A declared clustered plan refuses call-time policy overrides."""
    defs_path, con = _cluster_defs_and_con(tmp_path, cluster=True)
    analysis = Analysis.from_definitions("store_test", defs_path, con)
    assert lift_rows(analysis.run())
    with pytest.raises(TypeError):
        lift_rows(analysis.run(alpha=0.20))  # ty: ignore[unknown-argument]


@pytest.mark.filterwarnings(
    "ignore:metric .* total clusters across arms .* below 40:RuntimeWarning"
)
def test_metrics_reassignment_reresolves_a_same_name_metric_swap(tmp_path):
    """Reassigning `_metrics` with a SAME-NAMED metric object carrying
    different declared fields must re-resolve the plan. A name-only
    staleness check would see the name already present in the compiled
    procedure mapping and skip re-compilation, leaving its null/alternative
    stale instead of matching its own metric -- now load-bearing, since
    every dispatch path reads these fields."""
    defs_path, con = _cluster_defs_and_con(tmp_path, cluster=False)
    analysis = Analysis.from_definitions("store_test", defs_path, con)
    name = "revenue_per_user"

    analysis = make_analysis_like(
        analysis,
        [
            m.model_copy(update={"margin_abs": 1.0, "preferred_direction": "decrease"})
            if m.name == name
            else m
            for m in analysis.metrics
        ],
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )

    (estimate,) = lift_rows(analysis.run())
    assert estimate.null_abs == pytest.approx(1.0)
    assert estimate.alternative == "less"


def test_definitions_unit_frame_serves_declared_covariate(tmp_path):
    from tests.analysis_factory import _native_source
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, _group, _revenue = covariate_defs_and_con(tmp_path)
    source = _native_source(Analysis.from_definitions("cov_test", defs_path, con))
    table = source.unit_frame(source.context.metrics[0], covariates=["tenure"])
    rows = {r["unit_id"]: r["tenure"] for r in table.to_pylist()}
    assert rows == pytest.approx(tenure)


def test_definitions_unit_frame_keeps_a_missing_covariate_null(tmp_path):
    from tests.analysis_factory import _native_source
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, _group, _revenue = covariate_defs_and_con(tmp_path)
    con.raw_sql("UPDATE cov_events SET tenure = NULL WHERE user_id = 'u3'")
    source = _native_source(Analysis.from_definitions("cov_test", defs_path, con))
    table = source.unit_frame(source.context.metrics[0], covariates=["tenure"])
    rows = {r["unit_id"]: r["tenure"] for r in table.to_pylist()}
    assert rows["u3"] is None
    assert rows["u4"] == pytest.approx(tenure["u4"])


def test_definitions_unit_frame_refuses_unresolvable_covariate(tmp_path):
    from increment.errors import CodedError
    from tests.analysis_factory import _native_source
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, _tenure, _group, _revenue = covariate_defs_and_con(tmp_path)
    source = _native_source(Analysis.from_definitions("cov_test", defs_path, con))
    with pytest.raises(CodedError) as raised:
        source.unit_frame(source.context.metrics[0], covariates=["nonexistent_property"])
    assert raised.value.code == "source.native.covariate_unresolved"
    assert raised.value.context["covariate"] == "nonexistent_property"


_AMBIGUOUS_COVARIATE_DEFS = """
dialect: duckdb
fact_sources:
  - name: a
    sql: SELECT * FROM amb_a
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: exposure, column: null}]
    properties:
      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}
      - {name: country, column: country, dtype: string, as_of: static}
      - {name: signup_date, column: signup_date, dtype: date, as_of: static}
      - {name: last_seen, column: last_seen, dtype: float, as_of: event_time}
  - name: b
    sql: SELECT * FROM amb_b
    timestamp_column: ts
    entities: [user_id]
    facts: [{name: purchase, column: revenue}]
    properties:
      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 7}
experiments:
  - name: amb_test
    exposure: enrolled
    unit: user_id
    control_group: control
    start: 2025-08-01
    end: 2025-08-07
    plan: {secondaries: [revenue]}
"""


def _ambiguous_covariate_source(tmp_path):
    import datetime as dt

    con = ibis.duckdb.connect()
    con.create_table(
        "amb_a",
        obj=[
            {
                "user_id": "u0",
                "ts": dt.datetime(2025, 8, 1),
                "event": "exposure",
                "experiment_id": "amb_test",
                "group_id": "control",
                "tenure": 1.0,
                "country": "US",
                "signup_date": dt.date(2024, 12, 1),
                "last_seen": 1.0,
            }
        ],
    )
    con.create_table(
        "amb_b",
        obj=[
            {
                "user_id": "u0",
                "ts": dt.datetime(2025, 8, 3),
                "event": "purchase",
                "revenue": 5.0,
                "tenure": 1.0,
            },
            {
                "user_id": "u0",
                "ts": dt.datetime(2025, 8, 9),
                "event": "purchase",
                "revenue": 0.0,
                "tenure": 1.0,
            },
        ],
    )
    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(_AMBIGUOUS_COVARIATE_DEFS)
    return _native_source(Analysis.from_definitions("amb_test", defs_path, con))


@pytest.mark.parametrize(
    ("covariate", "code"),
    [
        ("tenure", "source.native.covariate_ambiguous"),
        ("signup_date", "source.native.covariate_dtype"),
        ("last_seen", "source.native.covariate_as_of"),
        ("cluster_id", "source.native.covariate_reserved"),
    ],
)
def test_definitions_unit_frame_refuses_unusable_ad_hoc_covariate(tmp_path, covariate, code):
    """An ad hoc (undeclared) covariate gets the same source rule a declared
    one does: one eligible numeric or string, pre-exposure property or a
    named refusal (a date carries no adjustment meaning)."""
    from increment.errors import CodedError

    source = _ambiguous_covariate_source(tmp_path)
    with pytest.raises(CodedError) as raised:
        source.unit_frame(source.context.metrics[0], covariates=[covariate])
    assert raised.value.code == code


def test_definitions_unit_frame_serves_a_string_property_as_a_categorical_column(tmp_path):
    """A string property retains its native labels in a unit frame."""
    import pyarrow as pa

    source = _ambiguous_covariate_source(tmp_path)
    table = source.unit_frame(source.context.metrics[0], covariates=["country"])
    assert table.schema.field("country").type == pa.string()
    assert table.to_pylist()[0]["country"] == "US"
