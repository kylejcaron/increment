"""Native (definitions/warehouse) path under a declared Experiment.cluster.

End-to-end: exposure events carry the store label, first_exposures ->
panel_spine -> unit_totals thread it through, group_summary collapses by
store, and estimate_lift uses relative t and additive Welch references.
Point estimates NEVER move; only the SE and reference widen.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, cast

import ibis
import numpy as np
import pytest

from increment import Scope, from_definitions, from_frame, run
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.diagnostics import SRMResult, sample_ratio_mismatch
from increment.estimation.sitewide import SitewideImpact
from increment.estimators import sitewide
from increment.evidence import UnitRows, UnitRowsRequest, evidence_provider, request_evidence
from increment.query import summary_sql
from increment.workflows import export_moments, metrics, srm
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import (
    bound_source,
    lift_rows,
    make_binding_like,
    view_readout,
)
from tests.sequential_cases import registration


def _unit_rows(source, metric) -> UnitRows:
    return cast(
        UnitRows,
        request_evidence(evidence_provider(source), UnitRowsRequest(metric)),
    )


N_STORES = 20  # per arm -> 40 total clusters, above the small-K warning
UNITS_PER_STORE = 2

_DEFS_TEMPLATE = """
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
    allocation_scheme: independent
"""


def _event_rows(
    *, null_store_unit: str | None = None, spanning_store: bool = False
) -> list[dict[str, Any]]:
    """Exposure + purchase events; store effects make units within a store
    correlated, so the clustered SE must widen against the iid one."""
    rows: list[dict[str, Any]] = []
    for arm, base in (("control", 5.0), ("treatment", 5.5)):
        for s in range(N_STORES):
            store = f"{arm[:1]}s{s}"
            if spanning_store and arm == "treatment" and s == 0:
                store = "cs0"  # collides with control's first store
            for u in range(UNITS_PER_STORE):
                unit = f"{arm[:1]}{s}_{u}"
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": dt.datetime(2025, 8, 1, 9, 0, 0),
                        "event": "exposure",
                        "experiment_id": "store_test",
                        "group_id": arm,
                        "store_id": None if unit == null_store_unit else store,
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
    return rows


def _warehouse(tmp_path, *, cluster: bool, rows, windowed: bool) -> tuple[Any, Any]:
    con = ibis.duckdb.connect()
    con.create_table("cl_events", obj=rows if rows is not None else _event_rows())
    cluster_line = "    cluster: store_id" if cluster else ""
    defs = tmp_path / "defs.yaml"
    definitions = _DEFS_TEMPLATE.format(cluster_line=cluster_line)
    if not windowed:
        definitions = definitions.replace("    window_days: 7\n", "")
    defs.write_text(definitions)
    return con, defs


def _analysis(tmp_path, *, cluster: bool = True, rows=None, windowed: bool = True) -> Any:
    con, defs = _warehouse(tmp_path, cluster=cluster, rows=rows, windowed=windowed)
    return from_definitions(defs, con, experiment_name="store_test")


def _experiment(
    tmp_path, *, cluster: bool = True, rows=None, windowed: bool = True
) -> tuple[Any, Any]:
    con, defs = _warehouse(tmp_path, cluster=cluster, rows=rows, windowed=windowed)
    return from_definitions(defs, con, experiment_name="store_test")


@pytest.mark.parametrize(
    ("invalid", "code"),
    [
        ("conflict", "query.integrity.cluster_conflict"),
        ("null", "query.integrity.cluster_labels"),
        ("mixed", "query.integrity.cluster_labels"),
    ],
)
def test_direct_unit_frame_validates_raw_cluster_labels(tmp_path, invalid, code):
    rows = _event_rows(
        null_store_unit="c0_0" if invalid == "null" else None,
        spanning_store=invalid == "mixed",
    )
    if invalid == "conflict":
        rows.append({**rows[0], "store_id": "another_store", "event_at": dt.datetime(2025, 8, 2)})
    analysis = _analysis(tmp_path, rows=rows)
    with analysis[1]:
        source = bound_source(analysis[1])
        with pytest.raises(InvalidRequestError) as error:
            _unit_rows(source, source.context.metrics[0])
        assert error.value.code == code


def test_direct_unit_frame_accepts_observational_cluster_spanning_arms(tmp_path):
    from increment.semantics.design import AdjustmentSet, Observational

    analysis = _analysis(tmp_path, rows=_event_rows(spanning_store=True))

    with analysis[1]:
        experiment, resolver = make_binding_like(
            analysis[1],
            design=Observational(
                control_group="control", adjustment=AdjustmentSet(covariates=("baseline",))
            ),
        )
        source = bound_source(resolver)
        rows = _unit_rows(source, source.context.metrics[0]).to_pylist()

    assert len(rows) == 2 * N_STORES * UNITS_PER_STORE
    shared = {row["unit_id"]: row for row in rows if row["cluster_id"] == "cs0"}
    assert set(shared) == {"c0_0", "c0_1", "t0_0", "t0_1"}
    for unit, arm, outcome in (
        ("c0_0", "control", 5.0),
        ("c0_1", "control", 5.05),
        ("t0_0", "treatment", 5.5),
        ("t0_1", "treatment", 5.55),
    ):
        assert shared[unit]["group_id"] == arm
        assert shared[unit]["y"] == pytest.approx(outcome)


def test_clustering_widens_the_se_but_never_moves_the_point_estimate(tmp_path):
    (flat,) = lift_rows(lift_rows(view_readout(*_analysis(tmp_path, cluster=False))))
    (clustered,) = lift_rows(lift_rows(view_readout(*_analysis(tmp_path, cluster=True))))

    clustered_lift = clustered.require_lift()
    flat_lift = flat.require_lift()
    assert clustered_lift.value == pytest.approx(flat_lift.value, rel=1e-9)
    assert clustered.abs_diff == pytest.approx(flat.abs_diff, rel=1e-9)
    # Real store effects (0.2/index) dwarf within-store wiggle (0.05); at
    # 2 units/store the design effect tops out at 2, so the SE ratio approaches sqrt(2) - assert it gets most of the way there.
    assert clustered.relative_confidence_set is not None
    clustered_set = clustered.relative_confidence_set
    assert clustered_set.geometry == "bounded"
    assert clustered_set.reference.kind == "t"
    # Both runs estimate their variance from the data, so both cut a t
    # reference; clustering shows up in the SE and the degrees of freedom,
    # not in the kind of reference.
    assert flat.reference_kind == "t"
    assert flat.dof is None and clustered.dof is not None
    assert clustered.abs_se is not None and flat.abs_se is not None
    assert clustered.abs_se > 1.3 * flat.abs_se
    clustered_lo, clustered_hi = clustered_set.intervals[0]
    flat_lo, flat_hi = flat_lift.lb, flat_lift.ub
    assert clustered_lo is not None and clustered_hi is not None
    assert flat_lo is not None and flat_hi is not None
    assert clustered_hi - clustered_lo > flat_hi - flat_lo
    assert clustered.n_clusters == 2 * N_STORES
    assert clustered.dof == N_STORES - 1


def test_native_and_frame_substrates_agree_on_the_clustered_estimate(tmp_path):
    """Same population, two substrates: duckdb builders vs from_unit_summary."""
    import pandas as pd

    (native,) = lift_rows(lift_rows(view_readout(*_analysis(tmp_path, cluster=True))))

    units = [
        {
            "user_id": f"{arm[:1]}{s}_{u}",
            "variant": arm,
            "store_id": f"{arm[:1]}s{s}",
            "revenue": base + 0.2 * s + 0.05 * u,
        }
        for arm, base in (("control", 5.0), ("treatment", 5.5))
        for s in range(N_STORES)
        for u in range(UNITS_PER_STORE)
    ]
    (frame,) = lift_rows(
        lift_rows(
            view_readout(
                *from_frame(
                    pd.DataFrame(units),
                    shape="summary",
                    unit="user_id",
                    group="variant",
                    control="control",
                    metrics=[
                        {"name": "revenue_per_user", "type": "mean", "value_column": "revenue"}
                    ],
                    cluster="store_id",
                )
            )
        )
    )

    assert native.require_lift().value == pytest.approx(frame.require_lift().value, rel=1e-9)
    assert native.require_lift().log_se == pytest.approx(frame.require_lift().log_se, rel=1e-9)
    assert native.require_lift().lb == pytest.approx(frame.require_lift().lb, rel=1e-9)
    assert (native.n_clusters, native.dof) == (frame.n_clusters, frame.dof)


def test_native_and_frame_substrates_agree_on_the_clustered_srm(tmp_path):
    """The sample-ratio check reads the same two grains on both substrates:
    clusters tested, units reported."""
    import pandas as pd

    from increment.semantics.design import Randomized

    native = srm(*_analysis(tmp_path, cluster=True), expected={"control": 0.5, "treatment": 0.5})

    units = [
        {
            "user_id": f"{arm[:1]}{s}_{u}",
            "variant": arm,
            "store_id": f"{arm[:1]}s{s}",
            "revenue": base + 0.2 * s + 0.05 * u,
        }
        for arm, base in (("control", 5.0), ("treatment", 5.5))
        for s in range(N_STORES)
        for u in range(UNITS_PER_STORE)
    ]
    frame = srm(
        *from_frame(
            pd.DataFrame(units),
            shape="summary",
            unit="user_id",
            group="variant",
            metrics=[{"name": "revenue_per_user", "type": "mean", "value_column": "revenue"}],
            cluster="store_id",
            design=Randomized(
                control_group="control",
                allocation_scheme="independent",
            ),
        ),
        expected={"control": 0.5, "treatment": 0.5},
    )

    assert isinstance(native, SRMResult) and isinstance(frame, SRMResult)
    assert frame.grain == native.grain == "cluster"
    assert frame.observed == native.observed == {"control": N_STORES, "treatment": N_STORES}
    assert frame.inference == native.inference == "always_valid"
    assert frame.unit_counts == native.unit_counts
    assert (frame.df, frame.chi2_stat, frame.is_srm) == (
        native.df,
        native.chi2_stat,
        native.is_srm,
    )


_RATIO_DEFS_TEMPLATE = """
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
      - name: session_end
        column: null
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: rev_per_session
    type: ratio
    entity: user_id
    numerator: {{fact: purchase, aggregation: sum}}
    denominator: {{fact: session_end, aggregation: count}}
experiments:
  - name: store_test
    exposure: enrolled
    unit: user_id
{cluster_line}
    start: 2025-08-01
    end: 2025-08-06
    plan: {{secondaries: [rev_per_session]}}
    control_group: control
"""


def _ratio_event_rows() -> list[dict[str, Any]]:
    """Sessions vary by store, so the per-cluster denominator total is NOT
    the cluster size - the two readings of the den family diverge."""
    rows: list[dict[str, Any]] = []
    for arm, base in (("control", 5.0), ("treatment", 6.0)):
        for s in range(N_STORES):
            store = f"{arm[:1]}s{s}"
            n_sessions = 2 + (s % 3)
            for u in range(UNITS_PER_STORE):
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
                for k in range(n_sessions):
                    rows.append(
                        {
                            "user_id": unit,
                            "event_at": dt.datetime(2025, 8, 2, 10 + k, 0, 0),
                            "event": "session_end",
                            "experiment_id": None,
                            "group_id": None,
                            "store_id": None,
                            "revenue": None,
                        }
                    )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": dt.datetime(2025, 8, 2, 10, 0, 0),
                        "event": "purchase",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "revenue": n_sessions * (base + 0.3 * s + 0.02 * u),
                    }
                )
    return rows


def _ratio_analysis(tmp_path, *, cluster: bool) -> Any:
    con = ibis.duckdb.connect()
    con.create_table("cl_events", obj=_ratio_event_rows())
    defs = tmp_path / f"ratio_defs_{int(cluster)}.yaml"
    defs.write_text(
        _RATIO_DEFS_TEMPLATE.format(cluster_line="    cluster: store_id" if cluster else "")
    )
    return from_definitions(defs, con, experiment_name="store_test")


def test_clustered_ratio_metric_widens_the_se_but_never_moves_the_estimate(tmp_path):
    """A declared cluster on a RATIO metric: the den family carries the
    metric's own per-cluster session totals, so the estimand is unchanged
    (sum(revenue)/sum(sessions) per arm) while the SE goes cluster-robust
    with separate relative t and additive Welch references."""
    (flat,) = lift_rows(lift_rows(view_readout(*_ratio_analysis(tmp_path, cluster=False))))
    (clustered,) = lift_rows(lift_rows(view_readout(*_ratio_analysis(tmp_path, cluster=True))))

    clustered_lift = clustered.require_lift()
    flat_lift = flat.require_lift()
    assert clustered_lift.value == pytest.approx(flat_lift.value, rel=1e-12)
    assert clustered.abs_diff == pytest.approx(flat.abs_diff, rel=1e-12)
    assert clustered.n_clusters == 2 * N_STORES
    assert clustered.relative_confidence_set is not None
    clustered_set = clustered.relative_confidence_set
    assert clustered_set.geometry == "bounded"
    assert clustered_set.reference.kind == "t"
    # Both runs estimate their variance from the data, so both cut a t
    # reference; clustering shows up in the SE and the degrees of freedom.
    assert flat.reference_kind == "t"
    assert flat.dof is None and clustered.dof is not None
    clustered_lo, clustered_hi = clustered_set.intervals[0]
    flat_lo, flat_hi = flat_lift.lb, flat_lift.ub
    assert clustered_lo is not None and clustered_hi is not None
    assert flat_lo is not None and flat_hi is not None
    assert clustered_hi - clustered_lo > flat_hi - flat_lo
    assert clustered.dof == N_STORES - 1
    assert clustered.abs_se is not None and flat.abs_se is not None
    assert clustered.abs_se > flat.abs_se


def test_clustered_ratio_native_and_frame_substrates_agree(tmp_path):
    """The ibis collapse and the frame collapse bind the den family the same
    way for a ratio metric -- and, since both sides declare the same
    single-secondary plan, both independently run the same BH/FCR family
    selection over their own (matching) p-value, landing on the same
    selected level and re-estimated interval. The selected capped-FCR
    interval crosses its null, so ``discovery=True, stat_sig=False`` on
    both substrates: discovery records family selection, not interval
    exclusion."""
    import pandas as pd

    from increment.semantics.models import AnalysisPlan

    (native,) = lift_rows(lift_rows(view_readout(*_ratio_analysis(tmp_path, cluster=True))))
    units = [
        {
            "user_id": f"{arm[:1]}{s}_{u}",
            "variant": arm,
            "store_id": f"{arm[:1]}s{s}",
            "revenue": (2 + s % 3) * (base + 0.3 * s + 0.02 * u),
            "sessions": float(2 + s % 3),
        }
        for arm, base in (("control", 5.0), ("treatment", 6.0))
        for s in range(N_STORES)
        for u in range(UNITS_PER_STORE)
    ]
    (frame,) = lift_rows(
        lift_rows(
            view_readout(
                *from_frame(
                    pd.DataFrame(units),
                    shape="summary",
                    unit="user_id",
                    group="variant",
                    control="control",
                    metrics=[
                        {
                            "name": "rev_per_session",
                            "type": "ratio",
                            "numerator": "revenue",
                            "denominator": "sessions",
                        }
                    ],
                    cluster="store_id",
                    plan=AnalysisPlan(secondaries=["rev_per_session"]),
                )
            )
        )
    )
    assert native.relative_confidence_set is not None
    assert frame.relative_confidence_set is not None
    native_set = native.relative_confidence_set
    frame_set = frame.relative_confidence_set
    assert native_set.geometry == frame_set.geometry == "bounded"
    assert native_set.intervals[0][0] == pytest.approx(frame_set.intervals[0][0], rel=1e-9)
    assert native_set.intervals[0][1] == pytest.approx(frame_set.intervals[0][1], rel=1e-9)
    assert native_set.reference.kind == frame_set.reference.kind == "t"
    assert native_set.reference.a == pytest.approx(frame_set.reference.a, rel=1e-9)
    assert native_set.reference.c == pytest.approx(frame_set.reference.c, rel=1e-9)
    assert native_set.reference.var_a == pytest.approx(frame_set.reference.var_a, rel=1e-9)
    assert native_set.reference.var_c == pytest.approx(frame_set.reference.var_c, rel=1e-9)
    assert native_set.reference.cov_ac == pytest.approx(frame_set.reference.cov_ac, rel=1e-9)
    assert native.discovery is True and native.discovery == frame.discovery
    assert native.family_threshold == pytest.approx(0.10)
    assert native.family_threshold == frame.family_threshold
    assert native.family_q == pytest.approx(0.10) == frame.family_q
    assert native.stat_sig() is False and frame.stat_sig() is False
    assert native.require_lift().value == pytest.approx(frame.require_lift().value, rel=1e-9)
    # dof is a Welch-Satterthwaite reduction over floating-point
    # arm variances (not integer `K - 2`), so the two substrates' slightly
    # different aggregation order can differ by a ULP or two.
    assert native.n_clusters == frame.n_clusters
    assert native.dof == pytest.approx(frame.dof, rel=1e-9)


def test_summary_sql_compiles_the_clustered_collapse_without_executing(tmp_path):
    sql = summary_sql(*_analysis(tmp_path, cluster=True))["revenue_per_user"]
    assert "store_id" in sql


# ── data-quality refusals at the execution boundary ─────────────────────


def test_null_cluster_label_refuses_at_run(tmp_path):
    experiment, resolver = _analysis(tmp_path, rows=_event_rows(null_store_unit="c0_0"))
    with pytest.raises(InvalidRequestError) as raised:
        lift_rows(view_readout(experiment, resolver))
    assert raised.value.code == "query.integrity.cluster_labels"


def test_group_spanning_cluster_refuses_at_run(tmp_path):
    experiment, resolver = _analysis(tmp_path, rows=_event_rows(spanning_store=True))
    with pytest.raises(InvalidRequestError) as raised:
        lift_rows(view_readout(experiment, resolver))
    assert raised.value.code == "query.integrity.cluster_labels"


def test_null_cluster_label_refuses_at_srm(tmp_path):
    """srm() is public and callable without run(), so it must apply the same
    label gate: a null label vanishes from the distinct-cluster count while
    still landing in the unit count, skewing the chi-square either way."""
    experiment, resolver = _analysis(tmp_path, rows=_event_rows(null_store_unit="c0_0"))
    with pytest.raises(InvalidRequestError) as raised:
        srm(experiment, resolver, expected={"control": 0.5, "treatment": 0.5})
    assert raised.value.code == "query.integrity.cluster_labels"


def test_group_spanning_cluster_refuses_at_srm(tmp_path):
    """A store appearing in both arms is counted once per arm, inflating both
    cluster counts undetected unless srm() gates on it too."""
    experiment, resolver = _analysis(tmp_path, rows=_event_rows(spanning_store=True))
    with pytest.raises(InvalidRequestError) as raised:
        srm(experiment, resolver, expected={"control": 0.5, "treatment": 0.5})
    assert raised.value.code == "query.integrity.cluster_labels"


# ── capability refusals ──────────────────────────────────────────────────


def test_export_refuses_on_a_clustered_experiment(tmp_path):
    experiment, resolver = _analysis(tmp_path, cluster=True)
    destination = tmp_path / "moments.parquet"
    destination.write_bytes(b"keep me")
    with pytest.raises(CapabilityError) as raised:
        export_moments(experiment, resolver, destination)
    assert raised.value.code == "evidence.cluster_unavailable"
    assert set(raised.value.context) == {"constructor", "requested", "offered", "route"}
    assert destination.read_bytes() == b"keep me"


def test_sequential_inference_refuses_call_time_on_a_clustered_experiment(tmp_path):
    from increment.estimation.sequential import AlwaysValid
    from increment.plan import compile_decision_plan

    experiment, resolver = _analysis(tmp_path, cluster=True)
    # Clustered readouts now route through the shared readout pipeline
    # (Task 3), which reads its epistemic policy exclusively from the
    # declared AnalysisPlan -- call-time inference= is refused
    # unconditionally, declared plan or not.
    experiment, resolver = make_binding_like(
        resolver,
        plan=compile_decision_plan(None, metrics(experiment, resolver), path="warehouse"),
    )
    # Build the inference outside the refusal block so a TypeError from the
    # constructor cannot stand in for run() rejecting the keyword.
    inference = AlwaysValid(registration=registration())
    with pytest.raises(TypeError):
        run(experiment, resolver, inference=inference)  # ty: ignore[no-matching-overload]


def test_declared_plan_always_valid_inference_refuses_cluster_through_the_real_route(tmp_path):
    from tests.sequential_cases import registered_native

    analysis = _analysis(tmp_path, cluster=True)
    with pytest.raises(CapabilityError) as raised:
        registered_native(analysis)
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.parametrize(
    "scope",
    [
        Scope(view="daily", grain="daily", values=True),
        Scope(view="daily", grain="daily"),
        Scope(view="asof", grain="asof", values=True),
        Scope(view="asof", grain="asof"),
    ],
    ids=["daily_values", "daily_lift", "asof_values", "asof_lift"],
)
def test_day_axis_methods_refuse_on_a_clustered_experiment(tmp_path, scope):
    experiment, resolver = _analysis(tmp_path, cluster=True)
    with pytest.raises(CapabilityError) as raised:
        view_readout(experiment, resolver, scope)
    assert raised.value.code == "facade.analysis.clustered_day_axis"


# A balanced-cluster fixture can't police a mis-scaled clustered collapse
# alone (symmetric arms would cancel the wrong scaling), but it CAN pin the two invariants every cluster-robust path shares: point estimate never moves, only SE/reference widen.
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
def test_sitewide_serves_a_clustered_experiment(tmp_path):
    flat = sitewide(*_experiment(tmp_path, cluster=False, windowed=False), "revenue_per_user")
    clustered = sitewide(*_experiment(tmp_path, cluster=True, windowed=False), "revenue_per_user")

    assert isinstance(flat, SitewideImpact)
    assert isinstance(clustered, SitewideImpact)
    assert clustered.absolute_impact == pytest.approx(flat.absolute_impact, rel=1e-9)
    assert clustered.absolute_impact_se > flat.absolute_impact_se
    assert clustered.n_clusters == 2 * N_STORES
    assert clustered.absolute_dof == pytest.approx(2 * N_STORES - 2)

    from scipy.stats import t

    half = clustered.absolute_impact_ub - clustered.absolute_impact
    assert half == pytest.approx(
        t.ppf(0.975, 2 * N_STORES - 2) * clustered.absolute_impact_se, rel=1e-9
    )


def _sized_event_rows(*, control_units: int, treatment_units: int) -> list[dict[str, Any]]:
    """Exposure + purchase events with a chosen number of units per store in
    EACH arm, so the two arms' mean cluster sizes can be made to diverge -
    the cross-arm cluster-size imbalance the whole-site guard polices."""
    rows: list[dict[str, Any]] = []
    for arm, base, upc in (
        ("control", 5.0, control_units),
        ("treatment", 5.5, treatment_units),
    ):
        for s in range(N_STORES):
            store = f"{arm[:1]}s{s}"
            for u in range(upc):
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
    return rows


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_refuses_material_cross_arm_cluster_size_imbalance(tmp_path):
    # control 2 units/store, treatment 6 -> a 200% mean-cluster-size gap:
    # the whole-site number would confound the treatment effect with the size imbalance, so the clustered path refuses.
    rows = _sized_event_rows(control_units=2, treatment_units=6)
    with pytest.raises(CapabilityError) as raised:
        sitewide(
            *_experiment(tmp_path, cluster=True, rows=rows, windowed=False), "revenue_per_user"
        )
    assert raised.value.code == "query.builders.cluster_size_imbalance"


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_allows_a_balanced_cluster_size_profile(tmp_path):
    # Matched cluster sizes (3 units/store in both arms) clear the guard.
    rows = _sized_event_rows(control_units=3, treatment_units=3)
    impact = sitewide(
        *_experiment(tmp_path, cluster=True, rows=rows, windowed=False), "revenue_per_user"
    )
    assert isinstance(impact, SitewideImpact)
    assert math.isfinite(impact.absolute_impact)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_refuses_clustered_evidence_without_cluster_counts(tmp_path, monkeypatch):
    import dataclasses

    experiment, resolver = _experiment(
        tmp_path,
        cluster=True,
        rows=_sized_event_rows(control_units=3, treatment_units=3),
        windowed=False,
    )
    source = resolver.binding.source
    complete = source.sitewide_evidence
    monkeypatch.setattr(
        source,
        "sitewide_evidence",
        lambda metric, **kwargs: dataclasses.replace(
            complete(metric, **kwargs), cluster_counts=None
        ),
    )
    with pytest.raises(CapabilityError) as raised:
        sitewide(experiment, resolver, "revenue_per_user")
    assert raised.value.code == "facade.analysis.sitewide_cluster_counts_missing"
    assert raised.value.context["cluster"] == "store_id"


def _third_arm_rows() -> list[dict[str, Any]]:
    """A second treatment arm on its own stores, so the clustered fixture
    has three enrolled arms rather than two."""
    rows: list[dict[str, Any]] = []
    for s in range(N_STORES):
        for u in range(UNITS_PER_STORE):
            unit = f"b{s}_{u}"
            rows.append(
                {
                    "user_id": unit,
                    "event_at": dt.datetime(2025, 8, 1, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "store_test",
                    "group_id": "treatment_b",
                    "store_id": f"bs{s}",
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
                    "revenue": 5.3 + 0.2 * s + 0.05 * u,
                }
            )
    return rows


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_serves_a_clustered_multi_arm_experiment(tmp_path):
    """A second treatment arm changes nothing about cluster support:
    sitewide() still needs ``arm=`` to disambiguate (the same coded refusal
    it always raises for several enrolled non-control arms), and once
    named, nets the OTHER arm's clusters out of the baseline exactly like
    it would at iid grain."""
    rows = [*_event_rows(), *_third_arm_rows()]
    clustered_experiment, clustered_resolver = _experiment(
        tmp_path, cluster=True, rows=rows, windowed=False
    )
    flat_experiment, flat_resolver = _experiment(tmp_path, cluster=False, rows=rows, windowed=False)

    with pytest.raises(InvalidRequestError) as refused:
        sitewide(clustered_experiment, clustered_resolver, "revenue_per_user")
    assert refused.value.code == "facade.analysis.sitewide_arm_required"

    impact = sitewide(
        clustered_experiment, clustered_resolver, "revenue_per_user", arm="treatment_b"
    )
    flat_impact = sitewide(flat_experiment, flat_resolver, "revenue_per_user", arm="treatment_b")

    assert isinstance(impact, SitewideImpact)
    assert isinstance(flat_impact, SitewideImpact)
    assert impact.other_arm_ids == ("treatment",)
    assert impact.absolute_impact == pytest.approx(flat_impact.absolute_impact, rel=1e-9)
    assert impact.absolute_impact_se > flat_impact.absolute_impact_se
    # Every enrolled arm is scaled; the cluster-grain population is a
    # normalized float but still reports the same whole-unit count.
    assert impact.scaled_units == flat_impact.scaled_units == round(flat_impact.n_enrolled)
    assert flat_impact.scaled_units > flat_impact.n_control + flat_impact.n_treatment
    assert impact.unobserved_members == flat_impact.unobserved_members == 0
    assert impact.n_clusters == 3 * N_STORES
    # 2-arm pairwise (control vs treatment_b), not the 3-arm pooled dof
    # (3 * N_STORES - 2): sum-metric absolute impact never reads the
    # other co-enrolled arm.
    assert impact.absolute_dof == pytest.approx(2 * N_STORES - 2)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_still_serves_the_same_fixture_without_a_cluster(tmp_path):
    impact = sitewide(*_experiment(tmp_path, cluster=False, windowed=False), "revenue_per_user")

    assert isinstance(impact, SitewideImpact)
    assert math.isfinite(impact.absolute_impact)
    assert impact.absolute_impact_lb <= impact.absolute_impact <= impact.absolute_impact_ub


# ── sample-ratio mismatch at the randomization grain ─────────────────────


def _sized_cluster_rows(sizes: dict[str, list[int]]) -> list[dict[str, Any]]:
    """Exposure + purchase events with explicit per-arm cluster sizes:
    ``{"control": [2, 3]}`` = two stores holding 2 and 3 units."""
    rows: list[dict[str, Any]] = []
    for arm, arm_sizes in sizes.items():
        for s, size in enumerate(arm_sizes):
            store = f"{arm[:1]}s{s}"
            for u in range(size):
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
                        "revenue": 5.0 + 0.2 * s,
                    }
                )
    return rows


def test_srm_counts_clusters_not_units_on_a_clustered_experiment(tmp_path):
    """Balanced clusters of equal size: the check reads at the cluster grain
    and keeps the unit counts as labeled context."""
    result = srm(*_analysis(tmp_path, cluster=True), expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert result.grain == "cluster"
    assert result.observed == {"control": N_STORES, "treatment": N_STORES}
    assert result.unit_counts == {
        "control": N_STORES * UNITS_PER_STORE,
        "treatment": N_STORES * UNITS_PER_STORE,
    }
    assert result.is_srm is False


def test_unequal_cluster_sizes_are_not_a_sample_ratio_mismatch(tmp_path):
    """A clean 20/20 cluster split whose treatment stores are 3x larger. The
    randomizer allocated clusters, so this is not an SRM - but the unit
    counts it produces (40 vs 120) would have flagged loudly."""
    rows = _sized_cluster_rows({"control": [2] * 20, "treatment": [6] * 20})
    result = srm(*_analysis(tmp_path, rows=rows), expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 20, "treatment": 20}
    assert result.is_srm is False
    assert result.unit_counts == {"control": 40, "treatment": 120}
    assert (
        sample_ratio_mismatch(
            result.unit_counts,
            expected={"control": 0.5, "treatment": 0.5},
            inference="fixed",
        ).is_srm
        is True
    )


def test_cluster_count_skew_flags_even_when_unit_counts_balance(tmp_path):
    """The converse miss: 20 control stores against 8 treatment stores, sized
    so both arms land on 40 units. Unit counts see nothing; the cluster
    counts (what was actually randomized) flag."""
    rows = _sized_cluster_rows({"control": [2] * 20, "treatment": [5] * 8})
    result = srm(
        *_analysis(tmp_path, rows=rows),
        expected={"control": 0.5, "treatment": 0.5},
        inference="fixed",
        alpha=0.05,
    )
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 20, "treatment": 8}
    assert result.is_srm is True
    assert result.unit_counts == {"control": 40, "treatment": 40}
    assert (
        sample_ratio_mismatch(
            result.unit_counts,
            expected={"control": 0.5, "treatment": 0.5},
            inference="fixed",
        ).is_srm
        is False
    )


def test_clustered_srm_flags_a_missing_treatment_arm_in_a_cumulative_prefix(tmp_path):
    """A 14-cluster all-control prefix must retain the declared empty arm."""
    rows = _sized_cluster_rows({"control": [1] * 14})
    result = srm(*_analysis(tmp_path, rows=rows), expected={"control": 0.5, "treatment": 0.5})

    assert isinstance(result, SRMResult)
    assert result.grain == "cluster"
    assert result.observed == {"control": 14, "treatment": 0}
    assert result.unit_counts == {"control": 14, "treatment": 0}
    assert result.is_srm is True


def test_srm_stays_at_unit_grain_without_a_declared_cluster(tmp_path):
    """Same population, no declared cluster: unchanged unit-grain behaviour,
    and no descriptive unit counts to duplicate ``observed``."""
    result = srm(*_analysis(tmp_path, cluster=False), expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(result, SRMResult)
    assert result.grain == "unit"
    assert result.unit_counts == {}
    assert result.observed == {
        "control": N_STORES * UNITS_PER_STORE,
        "treatment": N_STORES * UNITS_PER_STORE,
    }
    assert result.is_srm is False


_QUANTILE_DEFS_TEMPLATE = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM ql_events
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - name: exposure
        column: null
      - name: latency
        column: value
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
experiments:
  - name: quantile_test
    exposure: enrolled
    unit: user_id
    start: 2025-08-01
    end: 2025-08-07
    plan: {secondaries: [p90_latency]}
    control_group: control
"""


def _quantile_event_rows(
    n_per_arm: int = 40, seed: int = 7, shift: float = 1.25
) -> list[dict[str, Any]]:
    """Exposure + latency events for a randomized quantile-metric source."""
    rng = np.random.default_rng(seed)
    control = rng.lognormal(mean=0.0, sigma=0.5, size=n_per_arm)
    treatment = rng.lognormal(mean=0.0, sigma=0.5, size=n_per_arm) * shift
    rows: list[dict[str, Any]] = []
    for arm, values in (("control", control), ("treatment", treatment)):
        for i, value in enumerate(values):
            unit = f"{arm[:1]}{i}"
            rows.append(
                {
                    "user_id": unit,
                    "event_at": dt.datetime(2025, 8, 1, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "quantile_test",
                    "group_id": arm,
                    "value": None,
                }
            )
            rows.append(
                {
                    "user_id": unit,
                    "event_at": dt.datetime(2025, 8, 7, 10, 0, 0),
                    "event": "latency",
                    "experiment_id": None,
                    "group_id": None,
                    "value": float(value),
                }
            )
    return rows


@backend_fixture()
def quantile_source_factory(tmp_path):
    """Build a fresh randomized quantile-metric experiment and resolver on demand."""

    def _make():
        con = ibis.duckdb.connect()
        con.create_table("ql_events", obj=_quantile_event_rows())
        defs = tmp_path / "quantile_defs.yaml"
        defs.write_text(_QUANTILE_DEFS_TEMPLATE)
        return from_definitions(defs, con, experiment_name="quantile_test")

    return _make


def test_quantile_run_loads_unit_frame_once_per_metric(quantile_source_factory, monkeypatch):

    experiment, resolver = quantile_source_factory()
    source: Any = bound_source(resolver)
    calls = {"unit_frame": 0}
    source_type = type(source)
    real_unit_frame = source_type.unit_frame

    def counting_unit_frame(self, metric, **kwargs):
        calls["unit_frame"] += 1
        return real_unit_frame(self, metric, **kwargs)

    monkeypatch.setattr(source_type, "unit_frame", counting_unit_frame)
    try:
        estimates = view_readout(experiment, resolver)
        assert calls["unit_frame"] == 1
        assert [row.metric for row in estimates] == ["p90_latency"]
        assert estimates[0].require_lift().value > 0
    finally:
        cast(Any, resolver).close()


pytestmark = pytest.mark.integration
