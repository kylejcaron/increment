"""Cluster (randomization grain) plumbing through the ibis builders.

Wire contract under a declared ``Experiment.cluster``: ``group_summary``
collapses in two stages - per cluster j, ``g_j`` = sum of y over its units
and ``m_j`` = unit count - and emits the CLUSTER rows in the existing
centered moment fields: ``n`` = K, the y family (``ref_y``/``cy1``/``cy2``)
over the ``g_j`` and the denominator family
(``ref_den``/``cden1``/``cden2``/``cyden``) over the ``m_j``. The point
estimate ``(ref_y + cy1/n) / (ref_den + cden1/n)`` is exactly the unit
mean, so clustering can never move a point estimate. The ``x`` family
carries ``m_j`` as well, so cluster size survives a declared ratio metric
spending its den family on its own denominator total - unless an
encouragement uptake total claims x instead.
"""

from __future__ import annotations

import datetime as dt

import ibis
import numpy as np
import pytest

from increment.errors import CapabilityError, InvalidRequestError
from increment.query.builders import (
    check_cluster_size_balance,
    cluster_exposure_counts,
    cohort_group_summary,
    first_exposures,
    group_summary,
    metric_events,
    post_exposure_stats,
    unit_day_spine_stats,
    unit_totals,
)
from increment.query.schemas import (
    CLUSTER_EXPOSURE_COUNTS,
    GROUP_SUMMARY,
)
from increment.semantics.models import (
    AnalysisPlan,
    Experiment,
    MeanMetric,
    Measure,
    RatioMetric,
    RetentionMetric,
)


def _cancel_tol(n, ref):
    """Absolute tolerance for a cancelling residual first moment.

    ``cy1``/``cden1`` are ~0 but carry the last bits of an ``n * ref``
    sized cancellation, so they need an absolute band scaled to that
    magnitude, never a relative one against zero.
    """
    return 1e-9 * max(1.0, abs(n * ref))


def _cluster_mean(row):
    """Unit mean recovered from a clustered row's centered moments."""
    return (row["ref_y"] + row["cy1"] / row["n"]) / (row["ref_den"] + row["cden1"] / row["n"])


def _cluster_size_mean(row, ref, resid):
    """A family's cluster total over cluster SIZE - the unit-weighted per-unit
    mean, read off the x family (``ref_x``/``cx1``)."""
    return (row[ref] + row[resid] / row["n"]) / (row["ref_x"] + row["cx1"] / row["n"])


def _experiment(cluster: str | None) -> Experiment:
    return Experiment(
        name="exp_cl",
        unit="unit_id",
        cluster=cluster,
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 6),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )


@pytest.fixture(scope="module")
def cl_con():
    # DuckDB's own thread count, unlike the harness default (tests/conftest.py):
    # this module keeps the spine, totals and summary queries on parallel plans.
    return ibis.duckdb.connect(threads=None)


@pytest.fixture(scope="module")
def cl_exposure_events(cl_con):
    """Six units in four stores; the store is the randomization grain.

    control:   u1(s1), u2(s1), u3(s2)
    treatment: u4(s3), u5(s3), u6(s4)

    Every unit also carries a singleton ``solo_id`` label for the
    every-cluster-a-singleton identity tests.
    """
    rows = []
    for unit, group, store in [
        ("u1", "control", "s1"),
        ("u2", "control", "s1"),
        ("u3", "control", "s2"),
        ("u4", "treatment", "s3"),
        ("u5", "treatment", "s3"),
        ("u6", "treatment", "s4"),
    ]:
        rows.append(
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_cl",
                "group_id": group,
                "store_id": store,
                "solo_id": f"solo_{unit}",
            }
        )
    return cl_con.create_table("cl_exposure_events", obj=rows)


@pytest.fixture(scope="module")
def cl_purchase_events(cl_con):
    """Post-exposure per-unit totals: u1=10, u2=20, u3=30, u4=40, u5=50, u6=0."""
    rows = [
        {
            "unit_id": unit,
            "ts": dt.datetime(2025, 8, 2, 10, 0, 0),
            "event": "purchase",
            "amount": amt,
        }
        for unit, amt in [("u1", 10.0), ("u2", 20.0), ("u3", 30.0), ("u4", 40.0), ("u5", 50.0)]
    ]
    return cl_con.create_table("cl_purchase_events", obj=rows)


@pytest.fixture(scope="module")
def cl_click_events(cl_con):
    """Uptake fact for the cluster+uptake collapse: u1, u4, u5 clicked."""
    rows = [
        {"unit_id": unit, "ts": dt.datetime(2025, 8, 2, 9, 0, 0), "event": "click"}
        for unit in ("u1", "u4", "u5")
    ]
    return cl_con.create_table("cl_click_events", obj=rows)


@pytest.fixture(scope="module")
def cl_session_events(cl_con):
    """Ratio denominator fact: u1=2, u2=3, u3=4, u4=3, u5=5, u6=2 sessions."""
    rows = [
        {"unit_id": unit, "ts": dt.datetime(2025, 8, 2, 10 + k, 0, 0), "event": "session_end"}
        for unit, count in [("u1", 2), ("u2", 3), ("u3", 4), ("u4", 3), ("u5", 5), ("u6", 2)]
        for k in range(count)
    ]
    return cl_con.create_table("cl_session_events", obj=rows)


REVENUE = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
RPS = RatioMetric(
    name="rev_per_session",
    entity="unit_id",
    numerator=Measure(fact="purchase", aggregation="sum"),
    denominator=Measure(fact="session_end", aggregation="count"),
)


def _totals(cl_exposure_events, cl_purchase_events, experiment):
    exposures = first_exposures(cl_exposure_events, experiment)
    events = metric_events(cl_purchase_events, REVENUE, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, REVENUE.name)
    return unit_totals(spine, stats, REVENUE, experiment)


# ── plumbing: the label rides exposures -> spine -> totals ──────────────


def test_first_exposures_carries_cluster_column(cl_exposure_events):
    exposures = first_exposures(cl_exposure_events, _experiment("store_id")).execute()
    assert "store_id" in exposures.columns
    got = exposures.set_index("unit_id")["store_id"].to_dict()
    assert got == {"u1": "s1", "u2": "s1", "u3": "s2", "u4": "s3", "u5": "s3", "u6": "s4"}


def test_first_exposures_refuses_missing_cluster_column(cl_exposure_events):
    with pytest.raises(CapabilityError) as raised:
        first_exposures(cl_exposure_events, _experiment("warehouse_id"))
    assert raised.value.code == "query.builders.cluster_column_missing"


def test_unit_totals_retains_cluster_column(cl_exposure_events, cl_purchase_events):
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    df = totals.execute().set_index("unit_id")
    assert df.loc["u1", "store_id"] == "s1"
    assert df.loc["u6", "store_id"] == "s4"
    assert df.loc["u6", "y"] == 0.0  # zero-fill unaffected by the carry


def test_unit_totals_refuses_breakout_with_cluster(cl_con, cl_exposure_events, cl_purchase_events):
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    events = metric_events(cl_purchase_events, REVENUE, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, REVENUE.name)
    props = cl_con.create_table("cl_props", obj=[{"unit_id": "u1", "country": "US"}])
    with pytest.raises(CapabilityError) as raised:
        unit_totals(spine, stats, REVENUE, experiment, by=["country"], properties_table=props)
    assert raised.value.code == "query.builders.cluster_total_grain"


# ── the two-stage collapse ───────────────────────────────────────────────


def test_group_summary_two_stage_collapse_matches_hand_computed(
    cl_exposure_events, cl_purchase_events
):
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    summary = group_summary(totals, cluster="store_id").execute().set_index("group_id")

    # control: s1 -> g=30, m=2; s2 -> g=30, m=1
    ctl = summary.loc["control"]
    assert int(ctl["n"]) == 2
    assert ctl["ref_y"] == pytest.approx(30.0)  # mean(g_j)
    assert abs(ctl["cy1"]) < _cancel_tol(2, 30.0)
    assert ctl["cy2"] == pytest.approx(0.0, abs=1e-9)  # both clusters sit on ref_y
    assert ctl["ref_den"] == pytest.approx(1.5)  # mean(m_j)
    assert abs(ctl["cden1"]) < _cancel_tol(2, 1.5)
    assert ctl["cden2"] == pytest.approx(0.5**2 + 0.5**2)
    assert ctl["cyden"] == pytest.approx(0.0, abs=1e-9)

    # treatment: s3 -> g=90, m=2; s4 -> g=0, m=1
    trt = summary.loc["treatment"]
    assert int(trt["n"]) == 2
    assert trt["ref_y"] == pytest.approx(45.0)
    assert abs(trt["cy1"]) < _cancel_tol(2, 45.0)
    assert trt["cy2"] == pytest.approx(45.0**2 + 45.0**2)
    assert trt["ref_den"] == pytest.approx(1.5)
    assert abs(trt["cden1"]) < _cancel_tol(2, 1.5)
    assert trt["cden2"] == pytest.approx(0.5)
    assert trt["cyden"] == pytest.approx(45.0 * 0.5 + (-45.0) * (-0.5))

    # No uptake declared, so the x family carries per-cluster SIZE: m = (2, 1)
    # for both arms (see the uptake test below for the other claimant).
    for row in (ctl, trt):
        assert row["ref_x"] == pytest.approx(1.5)
        assert abs(row["cx1"]) < _cancel_tol(2, 1.5)
        assert row["cx2"] == pytest.approx(0.5)
        assert row["cxden"] == pytest.approx(0.5)
    assert ctl["cxy"] == pytest.approx(0.0, abs=1e-9)
    assert trt["cxy"] == pytest.approx(45.0)

    # The unit-grain binary-uptake moments have no cluster analogue.
    for field in ("sum_d", "cyd", "cy2d", "cxd"):
        value = ctl[field]
        assert value is None or (isinstance(value, float) and np.isnan(value)), field

    # Point estimate: (ref_y + cy1/n) / (ref_den + cden1/n) IS the unit mean.
    assert _cluster_mean(ctl) == pytest.approx((10 + 20 + 30) / 3)
    assert _cluster_mean(trt) == pytest.approx((40 + 50 + 0) / 3)


def test_group_summary_two_stage_collapse_carries_uptake_in_x_family(
    cl_exposure_events, cl_purchase_events, cl_click_events
):
    """With uptake=True declared, each cluster's uptake TOTAL rides the x
    slot (ArmStats.mean_x()/var_x()/cov_yx() are grain-agnostic, unlike
    the unit-grain binary-d shortcuts var_d()/cov_yd()), plus the new
    cxden = cov(uptake total, cluster size) cross moment - the
    cluster-robust LATE reduction's moments
    (increment.estimation.variance.cluster_uptake_moments). u1, u4, u5
    clicked: control D_j = (1, 0) for (s1, s2); treatment D_j = (2, 0)
    for (s3, s4). sum_d/cyd/cy2d/cxd (unit-grain uptake) stay NULL -
    they would be silently WRONG at cluster grain (binary-d formulas)."""
    exposures = first_exposures(cl_exposure_events, _experiment("store_id"))
    events = metric_events(cl_purchase_events, REVENUE, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, _experiment("store_id"), REVENUE.name)
    totals = unit_totals(
        spine, stats, REVENUE, _experiment("store_id"), uptake_events=cl_click_events
    )
    summary = group_summary(totals, cluster="store_id", uptake=True).execute().set_index("group_id")

    ctl = summary.loc["control"]
    assert ctl["ref_x"] == pytest.approx(0.5)
    assert abs(ctl["cx1"]) < _cancel_tol(2, 0.5)
    assert ctl["cx2"] == pytest.approx(0.5)
    assert ctl["cxy"] == pytest.approx(0.0, abs=1e-9)
    assert ctl["cxden"] == pytest.approx(0.5)

    trt = summary.loc["treatment"]
    assert trt["ref_x"] == pytest.approx(1.0)
    assert abs(trt["cx1"]) < _cancel_tol(2, 1.0)
    assert trt["cx2"] == pytest.approx(2.0)
    assert trt["cxy"] == pytest.approx(90.0)
    assert trt["cxden"] == pytest.approx(1.0)

    for field in ("sum_d", "cyd", "cy2d", "cxd"):
        for row in (ctl, trt):
            value = row[field]
            assert value is None or (isinstance(value, float) and np.isnan(value)), field


def test_singleton_clusters_reproduce_unclustered_moments(cl_exposure_events, cl_purchase_events):
    """Every cluster a singleton: g_j = y_j, m_j = 1, so the clustered row's
    n and y family must equal the unclustered row's exactly, and the size
    moments must be degenerate (ref_den = 1, every centered den moment 0).
    """
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("solo_id"))
    flat = group_summary(totals).execute().set_index("group_id")
    clustered = group_summary(totals, cluster="solo_id").execute().set_index("group_id")

    for group in ("control", "treatment"):
        f, c = flat.loc[group], clustered.loc[group]
        assert int(c["n"]) == int(f["n"])
        assert c["ref_y"] == pytest.approx(f["ref_y"])
        assert c["cy1"] == pytest.approx(f["cy1"], abs=_cancel_tol(int(f["n"]), f["ref_y"]))
        assert c["cy2"] == pytest.approx(f["cy2"])
        assert c["ref_den"] == pytest.approx(1.0)
        assert c["cden1"] == pytest.approx(0.0, abs=1e-9)
        assert c["cden2"] == pytest.approx(0.0, abs=1e-9)
        # m_j is constant, so every (m_j - ref_den) factor vanishes.
        assert c["cyden"] == pytest.approx(0.0, abs=1e-9)
        # The recovered unit count still equals n.
        assert c["n"] * c["ref_den"] + c["cden1"] == pytest.approx(float(f["n"]))


def test_group_summary_clustered_ratio_carries_the_metrics_own_denominator(
    cl_exposure_events, cl_purchase_events, cl_session_events
):
    """A metric named in ``ratio_metrics`` rebinds the den family from the
    cluster SIZE m_j to its own per-cluster denominator total den_j.

    Sessions: u1=2, u2=3, u3=4, u4=3, u5=5, u6=2 - so
    control  s1 -> (num 30, den 5), s2 -> (num 30, den 4);
    treatment s3 -> (num 90, den 8), s4 -> (num  0, den 2).
    """
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    num_events = metric_events(cl_purchase_events, RPS, value_column="amount")
    den_events = metric_events(cl_session_events, RPS, part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, RPS.name)
    den_stats = post_exposure_stats(den_events, exposures, source_key="den")
    totals = unit_totals(spine, stats, RPS, experiment, den_stats=den_stats)
    summary = (
        group_summary(totals, cluster="store_id", ratio_metrics=[RPS.name])
        .execute()
        .set_index("group_id")
    )

    ctl = summary.loc["control"]
    assert int(ctl["n"]) == 2
    assert ctl["ref_y"] == pytest.approx(30.0)
    assert ctl["ref_den"] == pytest.approx(4.5)  # mean(5, 4), NOT mean(2, 1)
    assert ctl["cden2"] == pytest.approx(0.5**2 + 0.5**2)
    assert ctl["cyden"] == pytest.approx(0.0, abs=1e-9)

    trt = summary.loc["treatment"]
    assert trt["ref_y"] == pytest.approx(45.0)
    assert trt["ref_den"] == pytest.approx(5.0)  # mean(8, 2)
    assert trt["cden2"] == pytest.approx(3.0**2 + 3.0**2)
    assert trt["cyden"] == pytest.approx(45.0 * 3.0 + (-45.0) * (-3.0))

    # sum(num_j)/sum(den_j) is the ratio estimand, not the unit mean.
    assert _cluster_mean(ctl) == pytest.approx(60.0 / 9.0)
    assert _cluster_mean(trt) == pytest.approx(90.0 / 10.0)


def test_group_summary_clustered_ratio_keeps_cluster_size_in_the_x_family(
    cl_exposure_events, cl_purchase_events, cl_session_events
):
    """The case the den rebinding used to lose: a declared ratio metric's
    clustered row still carries per-cluster SIZE m_j, in the x family.

    control  s1 -> (num 30, den 5, m 2), s2 -> (num 30, den 4, m 1);
    treatment s3 -> (num 90, den 8, m 2), s4 -> (num  0, den 2, m 1).
    """
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    num_events = metric_events(cl_purchase_events, RPS, value_column="amount")
    den_events = metric_events(cl_session_events, RPS, part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, RPS.name)
    den_stats = post_exposure_stats(den_events, exposures, source_key="den")
    totals = unit_totals(spine, stats, RPS, experiment, den_stats=den_stats)
    summary = (
        group_summary(totals, cluster="store_id", ratio_metrics=[RPS.name])
        .execute()
        .set_index("group_id")
    )

    for group_id, row in summary.iterrows():
        assert row["ref_x"] == pytest.approx(1.5), group_id  # mean(2, 1)
        assert abs(row["cx1"]) < _cancel_tol(2, 1.5), group_id
        assert row["cx2"] == pytest.approx(0.5), group_id

    ctl, trt = summary.loc["control"], summary.loc["treatment"]
    assert ctl["cxy"] == pytest.approx(0.0, abs=1e-9)  # g_j is flat across clusters
    assert trt["cxy"] == pytest.approx(45.0)
    assert ctl["cxden"] == pytest.approx(0.5)  # centered cross sum of (m_j, den_j)
    assert trt["cxden"] == pytest.approx(3.0)

    # The point of the carry: sum(num_j)/sum(m_j) and sum(den_j)/sum(m_j) -
    # unit-weighted means the denominator-weighted ratio cannot give.
    assert _cluster_size_mean(ctl, "ref_y", "cy1") == pytest.approx(60.0 / 3.0)
    assert _cluster_size_mean(ctl, "ref_den", "cden1") == pytest.approx(9.0 / 3.0)
    assert _cluster_size_mean(trt, "ref_y", "cy1") == pytest.approx(90.0 / 3.0)
    assert _cluster_size_mean(trt, "ref_den", "cden1") == pytest.approx(10.0 / 3.0)


def test_group_summary_clustered_mean_family_carries_cluster_size_in_the_x_family(
    cl_exposure_events, cl_purchase_events
):
    """A mean-family clustered row carries size in x too, so a whole-site
    consumer reads ONE field whatever the metric family. Here x mirrors the
    den family exactly, since that family already holds m_j."""
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    summary = group_summary(totals, cluster="store_id").execute().set_index("group_id")
    for group_id, row in summary.iterrows():
        assert row["ref_x"] == pytest.approx(row["ref_den"]), group_id
        assert row["cx2"] == pytest.approx(row["cden2"]), group_id
        assert row["cxy"] == pytest.approx(row["cyden"]), group_id
        assert row["cxden"] == pytest.approx(row["cden2"]), group_id
        assert _cluster_size_mean(row, "ref_y", "cy1") == pytest.approx(_cluster_mean(row))


def test_group_summary_refuses_uptake_without_a_d_column(cl_con, cl_exposure_events):
    """uptake=True is a DECLARATION: with no ``d`` column to read the fact
    from, refuse loudly rather than quietly leaving cluster size in x."""
    totals = cl_con.create_table(
        "cl_no_d_totals",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "exp_cl",
                "group_id": "control",
                "metric": "revenue",
                "y": 10.0,
                "store_id": "s1",
            }
        ],
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        group_summary(totals, cluster="store_id", uptake=True)
    assert exc_info.value.code == "query.builders.group_summary_uptake"
    assert exc_info.value.context["columns"] == tuple(sorted(totals.columns))


def test_group_summary_refuses_ratio_metrics_without_a_y_den_column(cl_con, cl_exposure_events):
    """ratio_metrics=[...] is a DECLARATION: with no `y_den` column to read
    the denominator from, refuse loudly rather than silently rebinding
    cluster size onto a missing family."""
    totals = cl_con.create_table(
        "cl_no_y_den_totals",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "exp_cl",
                "group_id": "control",
                "metric": "revenue",
                "y": 10.0,
                "store_id": "s1",
            }
        ],
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        group_summary(totals, cluster="store_id", ratio_metrics=["revenue"])
    assert exc_info.value.code == "query.builders.group_summary_ratio"
    assert exc_info.value.context["ratio_names"] == ("revenue",)


def test_group_summary_without_ratio_metrics_keeps_cluster_sizes(
    cl_exposure_events, cl_purchase_events, cl_session_events
):
    """The rebinding is DECLARED, never inferred: the same totals table with
    a populated y_den still collapses to cluster sizes when the metric is
    not named in ``ratio_metrics``."""
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    num_events = metric_events(cl_purchase_events, RPS, value_column="amount")
    den_events = metric_events(cl_session_events, RPS, part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, RPS.name)
    den_stats = post_exposure_stats(den_events, exposures, source_key="den")
    totals = unit_totals(spine, stats, RPS, experiment, den_stats=den_stats)
    summary = group_summary(totals, cluster="store_id").execute().set_index("group_id")
    assert summary.loc["control", "ref_den"] == pytest.approx(1.5)
    assert summary.loc["treatment", "ref_den"] == pytest.approx(1.5)


def test_group_summary_columns_match_the_declared_contract(cl_exposure_events, cl_purchase_events):
    """increment.query.schemas is the declared output contract; both
    group_summary branches must satisfy it exactly, in both directions.
    Both carry ``cxden``: cov(x, den) at unit grain, cov(x family, size) clustered.
    """
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    assert set(group_summary(totals).columns) == GROUP_SUMMARY
    assert set(group_summary(totals, cluster="store_id").columns) == GROUP_SUMMARY


def test_group_summary_cluster_rows_never_carry_a_success_count(
    cl_exposure_events, cl_purchase_events
):
    """A cluster total is not a binary outcome, even for a declared binary metric name."""
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    names = sorted(set(totals.metric.execute()))
    rows = group_summary(totals, cluster="store_id", binary_metrics=names).execute()
    assert rows["successes"].isna().all()


# ── refusals: total grain only, declared column must exist ───────────────


def test_group_summary_refuses_cluster_with_by(cl_exposure_events, cl_purchase_events):
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment("store_id"))
    with pytest.raises(CapabilityError) as raised:
        group_summary(totals, by=["country"], cluster="store_id")
    assert raised.value.code == "query.builders.cluster_total_grain"


def test_group_summary_refuses_missing_cluster_column(cl_exposure_events, cl_purchase_events):
    totals = _totals(cl_exposure_events, cl_purchase_events, _experiment(None))
    with pytest.raises(InvalidRequestError) as exc_info:
        group_summary(totals, cluster="store_id")
    assert exc_info.value.code == "query.builders.group_summary_declared"
    assert exc_info.value.context["cluster"] == "store_id"


def test_cohort_group_summary_refuses_cluster(cl_con, cl_exposure_events):
    experiment = _experiment("store_id")
    retained = RetentionMetric(
        name="retained", entity="unit_id", fact="page_view", threshold_days=(1, 3)
    )
    exposures = first_exposures(cl_exposure_events, experiment)
    views = cl_con.create_table(
        "cl_page_views",
        obj=[{"unit_id": "u1", "ts": dt.datetime(2025, 8, 2, 9, 0, 0), "event": "page_view"}],
    )
    events = metric_events(views, retained)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, retained.name)
    with pytest.raises(CapabilityError) as raised:
        cohort_group_summary(spine, stats, retained, experiment)
    assert raised.value.code == "query.builders.cluster_total_grain"


# ── per-arm cluster counts (sample-ratio check at the randomization grain) ─


def test_cluster_exposure_counts_schema(cl_exposure_events):
    exposures = first_exposures(cl_exposure_events, _experiment("store_id"))
    result = cluster_exposure_counts(exposures, _experiment("store_id"))
    assert set(result.columns) == CLUSTER_EXPOSURE_COUNTS


def test_cluster_exposure_counts_counts_both_grains(cl_exposure_events):
    """Two stores per arm, unequal sizes: 3 units each, 2 clusters each."""
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    df = cluster_exposure_counts(exposures, experiment).execute().set_index("group_id")
    assert df.loc["control", "n_clusters"] == 2
    assert df.loc["treatment", "n_clusters"] == 2
    assert df.loc["control", "n_units"] == 3
    assert df.loc["treatment", "n_units"] == 3


def test_cluster_exposure_counts_refuses_without_a_declared_cluster(cl_exposure_events):
    exposures = first_exposures(cl_exposure_events, _experiment(None))
    with pytest.raises(CapabilityError) as raised:
        cluster_exposure_counts(exposures, _experiment(None))
    assert raised.value.code == "query.builders.cluster_undeclared"


def test_cluster_exposure_counts_refuses_missing_cluster_column(cl_exposure_events):
    exposures = first_exposures(cl_exposure_events, _experiment(None))
    with pytest.raises(CapabilityError) as raised:
        cluster_exposure_counts(exposures, _experiment("store_id"))
    assert raised.value.code == "query.builders.cluster_column_missing"


def test_cluster_exposure_counts_renders_snowflake_sql(cl_exposure_events):
    experiment = _experiment("store_id")
    exposures = first_exposures(cl_exposure_events, experiment)
    ibis.to_sql(cluster_exposure_counts(exposures, experiment), dialect="snowflake")


def _balance(n_clusters, n_units, *, control="control", target="treatment"):
    check_cluster_size_balance(
        n_clusters,
        n_units,
        experiment_name="exp",
        cluster="store",
        control_group=control,
        target_group=target,
    )


def test_check_cluster_size_balance_passes_when_arms_match():
    # 20 clusters, 40 units each -> mean size 2 in both arms, zero gap.
    _balance({"control": 20, "treatment": 20}, {"control": 40, "treatment": 40})


def test_check_cluster_size_balance_refuses_asymmetric_profile():
    # control 2 units/cluster, treatment 6 -> 200% relative gap.
    with pytest.raises(CapabilityError) as raised:
        _balance({"control": 20, "treatment": 20}, {"control": 40, "treatment": 120})
    assert raised.value.code == "query.builders.cluster_size_imbalance"


def test_check_cluster_size_balance_names_the_per_arm_sizes():
    with pytest.raises(CapabilityError) as raised:
        _balance({"control": 20, "treatment": 20}, {"control": 40, "treatment": 120})
    assert raised.value.code == "query.builders.cluster_size_imbalance"
    context = raised.value.context
    assert (context["control_mean_size"], context["target_mean_size"]) == (2.0, 6.0)
    assert (context["control_group"], context["target_group"]) == ("control", "treatment")


def test_check_cluster_size_balance_boundary_is_inclusive():
    # Exactly at the threshold (mean sizes 10 vs 12 -> 20% gap) is allowed;
    # a hair beyond refuses. Only the strict excess is unidentified.
    _balance({"control": 10, "treatment": 10}, {"control": 100, "treatment": 120})
    with pytest.raises(CapabilityError) as raised:
        _balance({"control": 10, "treatment": 10}, {"control": 100, "treatment": 121})
    assert raised.value.code == "query.builders.cluster_size_imbalance"


def test_check_cluster_size_balance_ignores_non_contrasted_arms():
    # A wildly sized third arm is not this contrast's confounder: control vs
    # target match, so the guard passes despite 'other' being 10x larger.
    _balance(
        {"control": 20, "treatment": 20, "other": 20},
        {"control": 40, "treatment": 40, "other": 400},
    )


def test_check_cluster_size_balance_refuses_an_arm_with_no_clusters():
    with pytest.raises(InvalidRequestError) as exc_info:
        _balance({"control": 0, "treatment": 20}, {"control": 0, "treatment": 40})
    assert exc_info.value.code == "query.builders.check_cluster.experiment_arm_no"
    assert exc_info.value.context["group"] == "control"
