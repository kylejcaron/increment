"""Cluster-robust covariance with separate relative t and additive Welch references.

Cluster rows ARE ratio rows (numerator g_j = cluster total, denominator
m_j = cluster unit count, n = K clusters); selection is driven by the
DECLARED cluster (``estimate_lift(cluster=...)``), never row shape.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import t

from increment.errors import CapabilityError, IncrementRuntimeWarning, InvalidRequestError
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.engine import estimate_lift
from increment.estimation.inference import Normal
from increment.estimation.sequential import AlwaysValid
from increment.estimation.variance import (
    ClusterVarianceModel,
    MeanVarianceModel,
)
from increment.semantics.models import MeanMetric, RatioMetric
from tests.sequential_cases import registration
from tests.warning_codes import warning_codes

METRIC = MeanMetric(name="m", entity="u", fact="f", aggregation="sum")


def _cluster_row(group: str, gs: list[float], ms: list[float]) -> dict:
    """One centered cluster row: y family is the per-cluster total g_j, den
    family the per-cluster unit count m_j, n the cluster count."""
    raw = {
        "experiment_id": "e",
        "metric": "m",
        "group_id": group,
        "n": len(gs),
        "sum_y": sum(gs),
        "sum_y2": sum(g * g for g in gs),
        "sum_x": None,
        "sum_x2": None,
        "sum_xy": None,
        "sum_den": sum(ms),
        "sum_den2": sum(m * m for m in ms),
        "sum_yden": sum(g * m for g, m in zip(gs, ms, strict=True)),
    }
    return centered_row_from_raw_sums(raw)


def _unit_row(group: str, ys: list[float]) -> dict:
    raw = {
        "experiment_id": "e",
        "metric": "m",
        "group_id": group,
        "n": len(ys),
        "sum_y": sum(ys),
        "sum_y2": sum(y * y for y in ys),
        "sum_x": None,
        "sum_x2": None,
        "sum_xy": None,
        "sum_den": None,
        "sum_den2": None,
        "sum_yden": None,
    }
    return centered_row_from_raw_sums(raw)


def _wiggle(center: float, n: int) -> list[float]:
    """Deterministic values around *center* with modest spread (small CV,
    so infer_lift's se_log_rr guard never trips)."""
    return [center + 0.1 * ((i % 5) - 2) for i in range(n)]


# ── the model itself ─────────────────────────────────────────────────────


def test_cluster_model_se_matches_hand_computed_value():
    """K=2 clusters: g=[10, 30], m=[1, 2].

    n_bar=20, d_bar=1.5; ddof=1: var_g=200, var_m=0.5, cov_gm=10.
    var(log R) = (1/2)[200/400 + 0.5/2.25 - 2*10/30] = (1/2)(0.5/9) = 1/36,
    so se = 1/6 exactly; R = 40/3 (the unit mean).
    """
    arm = ArmStats(
        study_id="e",
        metric="m",
        group_id="T",
        n=2,
        ref_y=20.0,
        cy1=0.0,
        cy2=200.0,
        ref_den=1.5,
        cden1=0.0,
        cden2=0.5,
        cyden=10.0,
    )
    log_mean, se = ClusterVarianceModel().log_mean_se(arm)
    assert log_mean == pytest.approx(math.log(40.0 / 3.0), rel=1e-12)
    assert se == pytest.approx(1.0 / 6.0, rel=1e-12)


def test_singleton_clusters_reproduce_the_iid_se_exactly():
    """Every cluster a singleton: var_m and cov_gm vanish identically, so
    the cluster model must equal MeanVarianceModel to the last bit of the
    delta method (same centered sums, same n)."""
    unit_arm = ArmStats(study_id="e", metric="m", group_id="T", n=3, ref_y=5.0, cy1=0.0, cy2=8.0)
    # y = [3, 5, 7] with every cluster a singleton: den_j == 1 for each, so
    # the den family is exactly its reference and every centred den moment
    # vanishes.
    cluster_arm = unit_arm.model_copy(
        update={"ref_den": 1.0, "cden1": 0.0, "cden2": 0.0, "cyden": 0.0}
    )
    log_flat, se_flat = MeanVarianceModel().log_mean_se(unit_arm)
    log_cl, se_cl = ClusterVarianceModel().log_mean_se(cluster_arm)
    assert log_cl == pytest.approx(log_flat, rel=1e-12)
    assert se_cl == pytest.approx(se_flat, rel=1e-12)


# ── the engine: identity, reference, metadata ────────────────────────────


def _identity_pair(n: int = 50) -> tuple[list[dict], list[dict]]:
    yc, yt = _wiggle(5.0, n), _wiggle(5.5, n)
    unit_rows = [_unit_row("C", yc), _unit_row("T", yt)]
    cluster_rows = [
        _cluster_row("C", yc, [1.0] * n),
        _cluster_row("T", yt, [1.0] * n),
    ]
    return unit_rows, cluster_rows


def test_identity_point_estimate_and_se_match_the_iid_path():
    unit_rows, cluster_rows = _identity_pair()
    (flat,) = estimate_lift([METRIC], unit_rows, control_group="C").results
    (clustered,) = estimate_lift([METRIC], cluster_rows, control_group="C", cluster="store").results

    # IID keeps its historical scalar log-normal path; clustered inference
    # carries the joint (a, c) covariance used by its Fieller set instead.
    assert clustered.require_lift().value == pytest.approx(flat.require_lift().value, rel=1e-9)
    assert clustered.abs_diff == pytest.approx(flat.abs_diff, rel=1e-12)
    assert clustered.abs_se == pytest.approx(flat.abs_se, rel=1e-12)
    assert flat.relative_confidence_set is None
    assert clustered.relative_confidence_set is not None
    reference = clustered.relative_confidence_set.reference
    k = 50

    def mean_variance(values):
        mean = sum(values) / k
        return (k / (k - 1)) * sum((value - mean) ** 2 for value in values) / k**2

    vc = mean_variance(_wiggle(5.0, k))
    vt = mean_variance(_wiggle(5.5, k))
    assert reference.var_c == pytest.approx(vc, rel=1e-12)
    assert reference.var_a == pytest.approx(vt + vc, rel=1e-12)
    assert reference.cov_ac == pytest.approx(-vc, rel=1e-12)


def test_clustered_lift_preserves_a_small_effect_at_a_large_offset():
    """Clustered mean metrics retain signed raw-scale contrasts at huge offsets."""
    n = 20
    mean_c, mean_t = 1e15, 1e15 + 1.0

    def row(group_id: str, mean: float) -> dict:
        return {
            "experiment_id": "e",
            "metric": "m",
            "group_id": group_id,
            "n": n,
            "ref_y": mean,
            "cy1": 0.0,
            "cy2": 4.0 * (n - 1),
            "ref_x": None,
            "cx1": None,
            "cx2": None,
            "cxy": None,
            "ref_den": 1.0,
            "cden1": 0.0,
            "cden2": 0.0,
            "cyden": 0.0,
            "x_role": None,
        }

    (result,) = estimate_lift(
        [METRIC],
        [row("C", mean_c), row("T", mean_t)],
        control_group="C",
        cluster="store",
    ).results

    lift = result.require_lift()
    assert lift.value == pytest.approx(1.0 / mean_c, rel=1e-15)
    assert result.abs_diff == pytest.approx(1.0, rel=1e-12)
    assert result.relative_confidence_set is not None
    reference = result.relative_confidence_set.reference
    assert reference.var_a == pytest.approx(2 * 4.0 / n)
    assert reference.var_c == pytest.approx(4.0 / n)
    assert reference.cov_ac == pytest.approx(-4.0 / n)
    assert result.relative_confidence_set.contains(lift.value)


def test_clustered_interval_uses_separate_relative_and_additive_references():
    unit_rows, cluster_rows = _identity_pair()
    (flat,) = estimate_lift([METRIC], unit_rows, control_group="C").results
    (clustered,) = estimate_lift([METRIC], cluster_rows, control_group="C", cluster="store").results

    assert clustered.n_clusters == 100
    assert clustered.dof == pytest.approx(49.0)
    assert clustered.reference_df == pytest.approx(49.0)
    assert clustered.abs_reference_df == pytest.approx(98.0)
    assert flat.n_clusters is None and flat.dof is None

    # Independently invert the joint quadratic; this checks Fieller geometry
    # rather than treating the interval as a scalar log-SE Wald interval.
    assert clustered.relative_confidence_set is not None
    reference = clustered.relative_confidence_set.reference
    critical_sq = t.ppf(0.975, 49.0) ** 2
    a, c = reference.a, reference.c
    va, vc, cov = reference.var_a, reference.var_c, reference.cov_ac
    qa = c * c - critical_sq * vc
    qb = -2.0 * a * c + 2.0 * critical_sq * cov
    qc = a * a - critical_sq * va
    discriminant = qb * qb - 4.0 * qa * qc
    roots = sorted(
        ((-qb - math.sqrt(discriminant)) / (2.0 * qa), (-qb + math.sqrt(discriminant)) / (2.0 * qa))
    )
    assert len(clustered.relative_confidence_set.intervals) == 1
    assert clustered.relative_confidence_set.intervals[0] == pytest.approx(roots, rel=1e-12)
    assert clustered.relative_confidence_set.contains(clustered.require_lift().value)
    assert clustered.abs_ub is not None and clustered.abs_diff is not None
    assert clustered.abs_se is not None
    assert clustered.abs_ub - clustered.abs_diff == pytest.approx(
        t.ppf(0.975, clustered.abs_reference_df) * clustered.abs_se, rel=1e-9
    )


def test_decision_stats_refuse_on_a_cluster_robust_estimate():
    _, cluster_rows = _identity_pair()
    (clustered,) = estimate_lift([METRIC], cluster_rows, control_group="C", cluster="store").results
    with pytest.raises(InvalidRequestError) as exc_info:
        clustered.chance_to_beat()
    assert exc_info.value.code == "estimation.results.lift.posterior_decision_stats_cluster_robust"
    assert exc_info.value.context["reference_df"] == clustered.reference_df == pytest.approx(49.0)


def test_prob_favorable_absolute_margin_refuses_on_a_cluster_robust_estimate():
    """The additive refusal reports its independent Welch reference."""
    _, cluster_rows = _identity_pair()
    (clustered,) = estimate_lift([METRIC], cluster_rows, control_group="C", cluster="store").results
    favorable = clustered.model_copy(update={"preferred_direction": "increase", "null_abs": -0.5})
    assert favorable.abs_diff is not None and favorable.abs_se is not None
    with pytest.raises(InvalidRequestError) as exc_info:
        favorable.prob_favorable()
    assert exc_info.value.code == "estimation.results.lift.p_value_cluster_robust_null_abs"
    assert (
        exc_info.value.context["reference_df"] == favorable.abs_reference_df == pytest.approx(98.0)
    )


# ── small-K policy ───────────────────────────────────────────────────────


def _k_clusters_rows(k_per_arm: int) -> list[dict]:
    gs = _wiggle(10.0, k_per_arm)
    ms = [2.0] * k_per_arm
    return [_cluster_row("C", gs, ms), _cluster_row("T", [g * 1.1 for g in gs], ms)]


def test_below_ten_total_clusters_is_admitted_with_warning():
    with pytest.warns(IncrementRuntimeWarning) as caught:
        results = estimate_lift([METRIC], _k_clusters_rows(4), control_group="C", cluster="store")
    assert results
    assert "estimation.engine.small_total_clusters" in warning_codes(caught)


def test_below_forty_total_clusters_warns_of_over_rejection():
    with pytest.warns(IncrementRuntimeWarning) as caught:
        estimate_lift([METRIC], _k_clusters_rows(10), control_group="C", cluster="store")
    assert "estimation.engine.small_total_clusters" in warning_codes(caught)


def test_the_small_cluster_advisory_names_the_estimate_lift_caller():
    with pytest.warns(IncrementRuntimeWarning) as caught:
        estimate_lift([METRIC], _k_clusters_rows(4), control_group="C", cluster="store")
    advisories = [
        w
        for w in caught
        if getattr(w.message, "code", None) == "estimation.engine.small_total_clusters"
    ]
    assert advisories
    assert all(w.filename == __file__ for w in advisories)


def test_forty_total_clusters_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimate_lift([METRIC], _k_clusters_rows(20), control_group="C", cluster="store")


# ── structural refusals ──────────────────────────────────────────────────


def test_sequential_inference_with_cluster_refuses():
    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift(
            [METRIC],
            cluster_rows,
            control_group="C",
            cluster="store",
            inference=AlwaysValid(registration=registration("gaussian")),
        )

    assert raised.value.code == "sequential.route.unsupported"


def test_cuped_method_with_cluster_refuses():
    from increment.estimation.engine import Method

    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift(
            [METRIC],
            cluster_rows,
            control_group="C",
            cluster="store",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )

    error = raised.value
    assert error.code == "arm.adjustment.cluster_cuped"
    assert error.context["cluster"] == "store"


def test_cuped_method_with_cluster_and_ratio_metric_refuses():
    """A clustered ratio metric is servable unadjusted, but CUPED on top
    still refuses: the clustered collapse carries no per-unit covariate
    moments, and the cluster-level refusal fires before the ratio one."""
    from increment.estimation.engine import Method

    ratio = RatioMetric(
        name="m",
        entity="u",
        numerator={"fact": "f", "aggregation": "sum"},
        denominator={"fact": "g", "aggregation": "sum"},
    )
    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift(
            [ratio],
            cluster_rows,
            control_group="C",
            cluster="store",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )

    assert raised.value.code == "arm.adjustment.cluster_cuped"


def test_informative_prior_with_cluster_refuses():
    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift(
            [METRIC],
            cluster_rows,
            control_group="C",
            cluster="store",
            prior=Normal(mu=0.0, sigma=0.1),
        )

    error = raised.value
    assert error.code == "arm.adjustment.cluster_prior"
    assert error.context["cluster"] == "store"


def test_ratio_metric_with_cluster_reads_the_den_family_as_its_denominator():
    """A clustered RATIO row is the same shape with den_j (the metric's own
    per-cluster denominator total) in the den family, so the estimand is
    sum(num_j)/sum(den_j) and the SE is the same cluster-robust delta
    method. Identical rows read as a MEAN metric would give a different
    estimand only because the den family means something else - here the
    numbers ARE the metric's denominator, so both readings coincide and
    what this pins is that the ratio type is no longer refused."""
    ratio = RatioMetric(
        name="m",
        entity="u",
        numerator={"fact": "f", "aggregation": "sum"},
        denominator={"fact": "g", "aggregation": "sum"},
    )
    nums, dens = _wiggle(10.0, 20), [2.0 + (i % 3) for i in range(20)]
    rows = [_cluster_row("C", nums, dens), _cluster_row("T", [x * 1.1 for x in nums], dens)]
    # 40 total clusters sits exactly on the advisory boundary - no warning.
    (est,) = estimate_lift([ratio], rows, control_group="C", cluster="store").results
    assert est.n_clusters == 40
    assert est.dof == pytest.approx(19.0)
    expected = (sum(x * 1.1 for x in nums) / sum(dens)) / (sum(nums) / sum(dens)) - 1.0
    assert est.require_lift().value == pytest.approx(expected, rel=1e-9)


def test_quantile_metric_with_cluster_still_refuses():
    from increment.semantics.models import QuantileMetric

    quantile = QuantileMetric(name="m", entity="u", fact="f", aggregation="sum", quantile=0.5)
    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift([quantile], cluster_rows, control_group="C", cluster="store")

    error = raised.value
    assert type(error) is CapabilityError
    assert error.code == "arm.metric.quantile_cluster"
    assert error.context["metric"] == "m"
    assert error.context["metric_type"] == "quantile"
    assert error.context["cluster"] == "store"


def test_mean_metric_cluster_rows_without_declared_cluster_refuses():
    """A group_summary(cluster=...) collapse read with cluster=None
    is never legitimate at unit grain for a mean-family metric - 'n'
    counts clusters, not units, and the ref_den family only exists
    because of the clustered collapse. Without this guard,
    MeanVarianceModel silently reads it as unit-grain.

    The mean case is representative for conversion/retention too: the
    engine.py guard keys only on the (mean, conversion, retention) type
    tuple plus ref_den, and the frame validator forbids a populated
    denominator family outside type='ratio', so all three share this path."""
    _, cluster_rows = _identity_pair()
    with pytest.raises(CapabilityError) as raised:
        estimate_lift([METRIC], cluster_rows, control_group="C")

    error = raised.value
    assert error.code == "estimation.engine.cluster.denominator"
    assert error.context["metric"] == "m"
    assert error.context["metric_type"] == "mean"
    assert error.context["treatment_group"] == "T"
    assert error.context["control_group"] == "C"
    assert error.context["cluster"] is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_clustered_moments_canonicalize_joint_projection_and_round_trip(alternative):
    from increment.estimation.results import LiftEstimate

    control = [5.0, 5.1, 5.4]
    treatment = [6.0, 6.1, 6.4, 6.7, 7.3]
    (row,) = estimate_lift(
        [METRIC],
        [_cluster_row("C", control, [1.0] * 3), _cluster_row("T", treatment, [1.0] * 5)],
        control_group="C",
        cluster="store",
        alternative=alternative,
    ).results
    assert row.relative_confidence_set is not None
    reference = row.relative_confidence_set.reference
    assert row.abs_diff == reference.a
    assert row.abs_se == (math.sqrt(reference.var_a) or None)
    assert row.reference_df == 2
    assert row.abs_reference_df is not None
    assert row.abs_reference_df > row.reference_df
    assert row.abs_lb is not None and row.abs_ub is not None
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("control", "treatment", "has_additive_uncertainty", "null_abs"),
    [
        ([1, 1, 1, 1, 0], [0] * 5, True, None),
        ([1] * 5, [2] * 5, False, None),
        ([1, 1, 1, 1, 0], [0] * 5, True, 0.0),
    ],
)
def test_clustered_zero_relative_variance_retains_only_available_uncertainty(
    control, treatment, has_additive_uncertainty, null_abs
):
    from increment.decision import PValueEvidence
    from increment.estimation.results import LiftEstimate

    computation = estimate_lift(
        [METRIC],
        [_cluster_row("C", control, [20] * 5), _cluster_row("T", treatment, [20] * 5)],
        control_group="C",
        cluster="store",
        null_abs=null_abs,
    )
    (row,) = computation.results
    assert row.lift is not None
    assert row.lift.value == pytest.approx(sum(treatment) / sum(control) - 1)
    assert (row.lift.lb, row.lift.ub, row.lift.alpha) == (None, None, None)
    assert row.relative_confidence_set is None
    assert row.relative_unavailable_reason == "zero_relative_variance"
    assert row.stat_sig() is (null_abs is not None)
    if null_abs is None:
        with pytest.raises(InvalidRequestError) as exc:
            row.p_value()
        assert exc.value.code == "estimation.results.joint.unavailable"
        assert exc.value.context["reason"] == "zero_relative_variance"
    else:
        (evidence,) = computation.evidence.values()
        assert isinstance(evidence, PValueEvidence)
        assert evidence.p_value == pytest.approx(2 * t.sf(4, df=4), rel=1e-12)
    if has_additive_uncertainty:
        assert row.abs_se == pytest.approx(0.01)
        assert row.abs_lb is not None and row.abs_ub is not None and row.abs_diff is not None
        assert row.abs_lb < row.abs_diff < row.abs_ub
    else:
        assert (row.abs_se, row.abs_lb, row.abs_ub) == (None, None, None)
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
