"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import ibis
import numpy as np
import pyarrow as pa
import pytest

from increment import readouts as readout_functions
from increment.frame import MetricSpec
from increment.plan import compile_decision_plan
from increment.semantics.models import AnalysisPlan, Definitions
from tests.analysis_factory import lift_rows, make_analysis, make_analysis_like

# The secondary-family fixtures declare these two policy values on their
# plan, so the expected interval levels need no read-back of the plan.
SECONDARY_Q = 0.10
PLAN_ALPHA = 0.05


def _execute_plan_cluster_rows() -> list[dict[str, Any]]:
    """Exposure + conversion/signup/latency facts for a clustered
    experiment with two treatment arms; arm-prefixed store ids so no
    store ever spans two arms."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    arms = [
        ("control", 0.30, 0.20, 12.0),
        ("treatment_a", 0.45, 0.35, 10.0),
        ("treatment_b", 0.50, 0.40, 9.0),
    ]
    n_stores, units_per_store = 20, 2
    for arm, conv_p, signup_p, latency_mean in arms:
        for s in range(n_stores):
            store = f"{arm}_s{s}"
            for u in range(units_per_store):
                unit = f"{arm}_{s}_{u}"
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 1, 9, 0, 0),
                        "event": "exposure",
                        "experiment_id": "execute_plan_cluster",
                        "group_id": arm,
                        "store_id": store,
                        "converted": None,
                        "signed_up": None,
                        "latency": None,
                    }
                )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 0, 0),
                        "event": "conversion",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "converted": float(rng.random() < conv_p),
                        "signed_up": None,
                        "latency": None,
                    }
                )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 5, 0),
                        "event": "signup",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "converted": None,
                        "signed_up": float(rng.random() < signup_p),
                        "latency": None,
                    }
                )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 10, 0),
                        "event": "latency_fact",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "converted": None,
                        "signed_up": None,
                        "latency": float(rng.normal(latency_mean, 1.5)),
                    }
                )
    return rows


def _execute_plan_cluster_defs() -> Definitions:
    return Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM execute_plan_cluster_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        {"name": "conversion", "column": "converted"},
                        {"name": "signup", "column": "signed_up"},
                        {"name": "latency_fact", "column": "latency"},
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "conversion",
                    "entity": "user_id",
                    "fact": "conversion",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "signups",
                    "entity": "user_id",
                    "fact": "signup",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "latency",
                    "entity": "user_id",
                    "fact": "latency_fact",
                    "aggregation": "sum",
                    "preferred_direction": "decrease",
                },
            ],
            "experiments": [
                {
                    "name": "execute_plan_cluster",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "cluster": "store_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": "control",
                    "plan": {
                        "primary": ["conversion", "signups"],
                        "guardrails": [{"metric": "latency", "margin": 0.02}],
                    },
                }
            ],
        }
    )


@pytest.fixture(scope="module")
def _execute_plan_cluster_con():
    con = ibis.duckdb.connect()
    con.create_table("execute_plan_cluster_events", obj=_execute_plan_cluster_rows())
    return con


@pytest.fixture(scope="module")
def clustered_plan_analysis(_execute_plan_cluster_con):
    return make_analysis(
        _execute_plan_cluster_con,
        _execute_plan_cluster_defs(),
        experiment="execute_plan_cluster",
    )


@pytest.fixture(scope="module")
def clustered_noplan_analysis(_execute_plan_cluster_con):
    analysis = make_analysis(
        _execute_plan_cluster_con,
        _execute_plan_cluster_defs(),
        experiment="execute_plan_cluster",
    )
    # Build an explicitly unassigned plan through the test constructor;
    # the source adapter receives it at construction time.
    analysis = make_analysis_like(
        analysis, plan=compile_decision_plan(None, analysis.metrics, path="warehouse")
    )
    return analysis


class TestClusteredRunReadsDeclaredPlan:
    """A clustered run() must honor the declared plan exactly like every
    other non-Encouragement design: roles stamped, guardrail one-sided at
    its margin, call-time policy kwargs refused."""

    def test_clustered_run_stamps_roles_and_guardrail_tail(self, clustered_plan_analysis):
        results = lift_rows(clustered_plan_analysis.run())
        by_metric = {r.metric: r for r in results}
        assert by_metric["conversion"].role == "primary"
        # 2 primaries (alpha_share=0.05/2) x 2 treatment arms
        # (cell_alpha=alpha_share/2) -- same arithmetic as
        # `tests/test_readouts_plan.py`'s primary-split test.
        assert by_metric["conversion"].require_lift().level == pytest.approx(1 - 0.05 / 2 / 2)
        g = by_metric["latency"]  # declared guardrail, margin=0.02, decrease
        assert g.role == "guardrail"
        assert g.alternative == "less"
        assert g.null_lift == pytest.approx(0.02)

    def test_clustered_run_refuses_call_time_policy(self, clustered_plan_analysis):
        with pytest.raises(TypeError):
            lift_rows(clustered_plan_analysis.run(alternative="greater"))
        with pytest.raises(TypeError):
            lift_rows(clustered_plan_analysis.run(margins={"latency": 0.05}))

    def test_undeclared_plan_refuses_call_time_kwargs(self, clustered_noplan_analysis):
        # Clustered readouts now route through the shared readout pipeline
        # (Task 3), which reads its epistemic policy exclusively from the
        # declared AnalysisPlan -- call-time alternative= is refused
        # unconditionally, declared plan or not.
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            lift_rows(clustered_noplan_analysis.run(alternative="greater"))


def _secondary_family_cluster_rows() -> list[dict[str, Any]]:
    """Exposure + 4 continuous secondary facts for a clustered experiment
    with 2 treatment arms; arm-prefixed store ids so no store spans two
    arms. Engineered effects (`sec_a`: unambiguous +15% lift in both
    treatment arms; `sec_b`/`sec_c`/`sec_d`: near-null) mirror
    `tests/test_readouts_plan.py::test_secondary_discovery_matches_hand_bh`'s
    borderline-fixture technique, so BH selects a strict, deterministic
    subset of the 8 (metric, arm) cells."""
    rng = np.random.default_rng(19)
    metric_names = ["sec_a", "sec_b", "sec_c", "sec_d"]
    control_mean, sd = 10.0, 2.0
    multiplier = {"sec_a": 1.15, "sec_b": 1.002, "sec_c": 1.01, "sec_d": 0.999}
    arms = ["control", "treatment_a", "treatment_b"]
    n_stores, units_per_store = 30, 6
    total_units_per_arm = n_stores * units_per_store

    values: dict[str, dict[str, np.ndarray]] = {}
    for name in metric_names:
        values[name] = {}
        for arm in arms:
            base = control_mean if arm == "control" else control_mean * multiplier[name]
            values[name][arm] = rng.normal(base, sd, total_units_per_arm)

    rows: list[dict[str, Any]] = []
    unit_index = dict.fromkeys(arms, 0)
    for arm in arms:
        for s in range(n_stores):
            store = f"{arm}_s{s}"
            for u in range(units_per_store):
                unit = f"{arm}_{s}_{u}"
                i = unit_index[arm]
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 1, 9, 0, 0),
                        "event": "exposure",
                        "experiment_id": "sec_fam_cluster_exp",
                        "group_id": arm,
                        "store_id": store,
                        **{f"{name}_val": None for name in metric_names},
                    }
                )
                for name in metric_names:
                    rows.append(
                        {
                            "user_id": unit,
                            "event_at": datetime(2025, 8, 2, 9, 0, 0),
                            "event": f"{name}_fact",
                            "experiment_id": None,
                            "group_id": None,
                            "store_id": None,
                            **{
                                f"{other}_val": (values[name][arm][i] if other == name else None)
                                for other in metric_names
                            },
                        }
                    )
                unit_index[arm] += 1
    return rows


def _secondary_family_cluster_defs() -> Definitions:
    metric_names = ["sec_a", "sec_b", "sec_c", "sec_d"]
    return Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM sec_fam_cluster_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        *[
                            {"name": f"{name}_fact", "column": f"{name}_val"}
                            for name in metric_names
                        ],
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": name,
                    "entity": "user_id",
                    "fact": f"{name}_fact",
                    "aggregation": "sum",
                }
                for name in metric_names
            ],
            "experiments": [
                {
                    "name": "sec_fam_cluster_exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "cluster": "store_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": "control",
                    "plan": {
                        "alpha": PLAN_ALPHA,
                        "q": SECONDARY_Q,
                        "secondaries": metric_names,
                    },
                }
            ],
        }
    )


def test_clustered_secondary_family_selects_strict_subset_with_fcr_intervals():
    """A clustered declared-plan run with 4 in-family secondaries across 2
    treatment arms (8 (metric, arm) cells): `sec_a`'s unambiguous lift is
    selected in BOTH arms, `sec_b`/`sec_c`/`sec_d` are not, in every arm --
    an exact discovery verdict per (metric, arm), not just per metric.
    Selected cells' `lift.alpha` is the selected-interval alpha `R*q/m`
    (conservatively rounded) computed by hand; non-selected cells stay at
    the plan's nominal alpha."""
    con = ibis.duckdb.connect()
    con.create_table("sec_fam_cluster_events", obj=_secondary_family_cluster_rows())
    analysis = make_analysis(
        con, _secondary_family_cluster_defs(), experiment="sec_fam_cluster_exp"
    )

    results = lift_rows(analysis.run())
    secondary = [r for r in results if r.role == "secondary"]
    by_cell = {(r.metric, r.group_id): r for r in secondary}
    assert len(by_cell) == 8  # 4 metrics x 2 treatment arms

    for arm in ("treatment_a", "treatment_b"):
        assert by_cell[("sec_a", arm)].discovery is True
        for name in ("sec_b", "sec_c", "sec_d"):
            assert by_cell[(name, arm)].discovery is False

    # R = 2 selected cells of m = 8, allocated q*R/m.
    expected_selected_alpha = SECONDARY_Q * 2 / 8
    assert expected_selected_alpha == pytest.approx(0.025)
    for arm in ("treatment_a", "treatment_b"):
        assert by_cell[("sec_a", arm)].require_lift().alpha == pytest.approx(
            expected_selected_alpha
        )
        for name in ("sec_b", "sec_c", "sec_d"):
            assert by_cell[(name, arm)].require_lift().alpha == pytest.approx(PLAN_ALPHA)


def test_clustered_secondary_family_preserves_declaration_order(con):
    """Secondaries append AFTER every other role's rows internally (they
    run through the deferred family-selection pass appended once the
    main per-metric loop finishes), so a secondary declared BEFORE a
    later-computed guardrail must still be sorted back to its declared
    position - `Experiment.metric_names` always orders primaries,
    then secondaries, then guardrails, so a guardrail's row would
    otherwise land before an earlier-declared secondary's deferred row."""
    rng = np.random.default_rng(41)

    def _arm_values(base_mean: float) -> dict[str, np.ndarray]:
        return {
            "control": rng.normal(base_mean, 2.0, 200),
            "treatment": rng.normal(base_mean, 2.0, 200),
        }

    metric_defs = {
        "primary_y": ("mean", None, _arm_values(10.0)),
        "sec_x": ("mean", None, _arm_values(10.0)),
        "guard_z": ("mean", "decrease", _arm_values(5.0)),
    }
    n = 200
    rows = []
    for group in ("control", "treatment"):
        for u in range(n):
            user = f"{group[0]}{u}"
            common = {
                "user_id": user,
                "event_at": datetime(2025, 8, 1, 9, 0, 0),
                "experiment_id": "order_exp",
                "group_id": group,
                "store_id": user,
            }
            rows.append({**common, "event": "exposure", **{f"{m}_val": None for m in metric_defs}})
            for name, (_type, _dir, values) in metric_defs.items():
                rows.append(
                    {
                        **common,
                        "event_at": datetime(2025, 8, 2, 9, 0, 0),
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "event": f"{name}_fact",
                        **{
                            f"{other}_val": (float(values[group][u]) if other == name else None)
                            for other in metric_defs
                        },
                    }
                )
    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM order_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        *[{"name": f"{n}_fact", "column": f"{n}_val"} for n in metric_defs],
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": name,
                    "entity": "user_id",
                    "fact": f"{name}_fact",
                    "aggregation": "sum",
                    **({"preferred_direction": pref_dir} if pref_dir else {}),
                }
                for name, (_type, pref_dir, _values) in metric_defs.items()
            ],
            "experiments": [
                {
                    "name": "order_exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "cluster": "store_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": "control",
                    # Declared: primary, then secondary, then guardrail -
                    # `Experiment.metric_names` always orders this way.
                    "plan": {
                        "primary": "primary_y",
                        "secondaries": ["sec_x"],
                        "guardrails": ["guard_z"],
                    },
                }
            ],
        }
    )
    con.create_table("order_events", obj=rows)
    analysis = make_analysis(con, defs, experiment="order_exp")
    results = lift_rows(analysis.run())
    assert [r.metric for r in results] == ["primary_y", "sec_x", "guard_z"]
    assert {r.metric: r.role for r in results} == {
        "primary_y": "primary",
        "sec_x": "secondary",
        "guard_z": "guardrail",
    }


def test_clustered_secondary_discovery_parity_with_frame_path():
    """Same per-unit values, same 4 declared secondaries, engineered the
    same way as the cluster test above: one path runs `readouts.run` on
    an unclustered frame seam source, the other runs native `run()` on a
    clustered Definitions-backed experiment with one unit per cluster
    (`store_id == user_id`). The two paths' variance estimators differ
    (Normal-Normal conjugate vs cluster-robust t-reference), so their
    exact interval bounds are not compared -- only that both land on the
    same BH-selected discovery set, the invariant the clustered path
    must preserve relative to the frame path."""
    from increment.estimation.family import bh_select

    metric_names = ["sec_a", "sec_b", "sec_c", "sec_d"]
    control_mean, sd = 10.0, 2.0
    multiplier = {"sec_a": 1.15, "sec_b": 1.002, "sec_c": 1.01, "sec_d": 0.999}
    n_per_arm = 600

    rng = np.random.default_rng(23)
    values: dict[str, dict[str, np.ndarray]] = {}
    for name in metric_names:
        values[name] = {
            "control": rng.normal(control_mean, sd, n_per_arm),
            "treatment": rng.normal(control_mean * multiplier[name], sd, n_per_arm),
        }

    from increment.frame import FrameTotalsSource

    user_ids = [f"{arm}_{u}" for arm in ("control", "treatment") for u in range(n_per_arm)]
    variant = ["control"] * n_per_arm + ["treatment"] * n_per_arm
    table_cols: dict[str, Any] = {"user_id": user_ids, "variant": variant}
    for name in metric_names:
        table_cols[name] = np.concatenate([values[name]["control"], values[name]["treatment"]])
    frame_src = FrameTotalsSource.from_frame(
        pa.table(table_cols),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name=name) for name in metric_names],
        plan=AnalysisPlan(secondaries=metric_names),
    )
    frame_results = readout_functions.run(frame_src)
    frame_by_metric = {r.metric: r for r in frame_results}

    rows: list[dict[str, Any]] = []
    for arm in ("control", "treatment"):
        for u in range(n_per_arm):
            unit = f"{arm}_{u}"
            rows.append(
                {
                    "user_id": unit,
                    "event_at": datetime(2025, 8, 1, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "sec_fam_singleton_exp",
                    "group_id": arm,
                    "store_id": unit,
                    **{f"{name}_val": None for name in metric_names},
                }
            )
            for name in metric_names:
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 0, 0),
                        "event": f"{name}_fact",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        **{
                            f"{other}_val": (values[name][arm][u] if other == name else None)
                            for other in metric_names
                        },
                    }
                )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM sec_fam_singleton_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        *[
                            {"name": f"{name}_fact", "column": f"{name}_val"}
                            for name in metric_names
                        ],
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": name,
                    "entity": "user_id",
                    "fact": f"{name}_fact",
                    "aggregation": "sum",
                }
                for name in metric_names
            ],
            "experiments": [
                {
                    "name": "sec_fam_singleton_exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "cluster": "store_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": "control",
                    "plan": {"secondaries": metric_names},
                }
            ],
        }
    )
    con = ibis.duckdb.connect()
    con.create_table("sec_fam_singleton_events", obj=rows)
    analysis = make_analysis(con, defs, experiment="sec_fam_singleton_exp")
    native_results = lift_rows(analysis.run())
    native_by_metric = {r.metric: r for r in native_results}

    frame_p_values = [frame_by_metric[name].p_value() for name in metric_names]
    assert all(value is not None for value in frame_p_values)
    frame_selected_idx, _ = bh_select(
        [value for value in frame_p_values if value is not None], frame_src.context.plan.q
    )
    frame_selected = {metric_names[i] for i in frame_selected_idx}
    assert frame_selected  # a strict, nonempty subset was selected
    assert frame_selected != set(metric_names)
    assert frame_selected == {name for name in metric_names if frame_by_metric[name].discovery}

    native_discovery = {name: native_by_metric[name].discovery for name in metric_names}
    native_selected = {name for name, discovery in native_discovery.items() if discovery}
    assert native_selected == frame_selected


def _quantile_secondary_family_rows() -> list[dict[str, Any]]:
    """Exposure + five continuous quantile metrics: q_a is a clear shift,
    q_b/q_c/q_d are near-null, and q_prior is a larger sampling-family
    member whose informative prior does not remove its quantile evidence."""
    rng = np.random.default_rng(31)
    metric_names = ["q_a", "q_b", "q_c", "q_d", "q_prior"]
    control_mean, sd = 10.0, 2.0
    multiplier = {"q_a": 1.20, "q_b": 1.002, "q_c": 1.01, "q_d": 0.999, "q_prior": 1.50}
    n_per_arm = 700

    values: dict[str, dict[str, np.ndarray]] = {}
    for name in metric_names:
        values[name] = {
            "control": rng.normal(control_mean, sd, n_per_arm),
            "treatment": rng.normal(control_mean * multiplier[name], sd, n_per_arm),
        }

    rows: list[dict[str, Any]] = []
    for arm in ("control", "treatment"):
        for u in range(n_per_arm):
            unit = f"{arm}_{u}"
            rows.append(
                {
                    "user_id": unit,
                    "event_at": datetime(2025, 8, 1, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "quant_sec_fam_exp",
                    "group_id": arm,
                    **{f"{name}_val": None for name in metric_names},
                }
            )
            for name in metric_names:
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 0, 0),
                        "event": f"{name}_fact",
                        "experiment_id": None,
                        "group_id": None,
                        **{
                            f"{other}_val": (values[name][arm][u] if other == name else None)
                            for other in metric_names
                        },
                    }
                )
    return rows


def _quantile_secondary_family_defs() -> Definitions:
    metric_names = ["q_a", "q_b", "q_c", "q_d", "q_prior"]
    return Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM quant_sec_fam_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        *[
                            {"name": f"{name}_fact", "column": f"{name}_val"}
                            for name in metric_names
                        ],
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "quantile",
                    "name": name,
                    "entity": "user_id",
                    "fact": f"{name}_fact",
                    "aggregation": "sum",
                    "quantile": 0.5,
                }
                for name in metric_names
            ],
            "experiments": [
                {
                    "name": "quant_sec_fam_exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": "control",
                    "plan": {
                        "alpha": PLAN_ALPHA,
                        "q": SECONDARY_Q,
                        "secondaries": [
                            "q_a",
                            "q_b",
                            "q_c",
                            "q_d",
                            {"metric": "q_prior", "prior": {"mu": 0.0, "sigma": 0.05}},
                        ],
                    },
                }
            ],
        }
    )


def test_quantile_secondary_family_includes_prior_bound_sampling_member():
    """A declared prior does not exclude the quantile cell's sampling evidence."""
    con = ibis.duckdb.connect()
    con.create_table("quant_sec_fam_events", obj=_quantile_secondary_family_rows())
    analysis = make_analysis(con, _quantile_secondary_family_defs(), experiment="quant_sec_fam_exp")

    results = lift_rows(analysis.run())
    by_metric = {r.metric: r for r in results}

    assert by_metric["q_a"].discovery is True
    assert by_metric["q_prior"].discovery is True
    for name in ("q_b", "q_c", "q_d"):
        assert by_metric[name].discovery is False
    expected_selected_alpha = SECONDARY_Q * 2 / 5
    assert by_metric["q_a"].family_size == by_metric["q_prior"].family_size == 5
    assert by_metric["q_a"].require_lift().alpha == pytest.approx(expected_selected_alpha)
    assert by_metric["q_prior"].require_lift().alpha == pytest.approx(expected_selected_alpha)
    assert by_metric["q_prior"].quantile_p_value is not None
    assert by_metric["q_prior"].sampling_available is True
    assert by_metric["q_prior"].posterior_available is True
    posterior_estimate = by_metric["q_prior"].posterior_estimate
    assert posterior_estimate is not None
    assert posterior_estimate != pytest.approx(by_metric["q_prior"].require_lift().value)
    for name in ("q_b", "q_c", "q_d"):
        assert by_metric[name].require_lift().alpha == pytest.approx(PLAN_ALPHA)
