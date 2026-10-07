"""Cluster score covariance wiring and retained support contracts.

Independent numerical targets live in test_i13_independent_witnesses.py;
scientific acceptance lives in the full frozen calibration manifest.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm

from increment import Analysis, readouts
from increment.errors import (
    CapabilityError,
    IncrementRuntimeWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation._adjust.aipw import aipw_estimate
from increment.estimation._adjust.common import _bessel_cross
from increment.estimation._adjust.dml import dml_estimate
from increment.estimation._adjust.iptw import iptw_estimate
from increment.estimation._adjust.overlap import _cluster_sq
from increment.estimation.adjust import estimate_ate
from increment.estimation.armstats import ScoreStats
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.estimation.readout_types import ReadoutResults
from increment.frame import MetricSpec, from_unit_summary
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    IdentificationGate,
    Observational,
    UptakeSpec,
)
from tests.warning_codes import warning_codes, warning_context

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate
    from increment.sources import Grain, SourceOperation

DESIGN = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("z",)))

ESTIMATORS = {"iptw": iptw_estimate, "dml": dml_estimate, "aipw": aipw_estimate}


class ConstantPropensity:
    """Learner emitting a constant propensity, sized to the prediction set
    - usable as a propensity factory for IPTW, DML, and AIPW alike."""

    def __init__(self, e: float = 0.5):
        self._e = e

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self._e)


class ConstantOutcome:
    """Learner emitting a constant outcome prediction, sized to the
    prediction set - usable as an outcome-model factory for DML/AIPW so
    their cross-fitted nuisances are fold-invariant (a no-op ``fit`` and
    a data-independent ``predict``), making the resulting psi a closed
    form computable directly from raw arrays instead of by replicating
    the fitted model."""

    def __init__(self, m: float = 0.0):
        self._m = m

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self._m)


def _table(k_per_arm: int, m: int, seed: int, *, sigma_b: float = 1.0) -> pa.Table:
    """One row per unit: arm-balanced clusters, cluster effect b_j (ICC>0),
    unit-level covariate z correlated with y."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {"u": [], "g": [], "store": [], "solo": [], "y": [], "z": []}
    uid = 0
    for arm in ("C", "T"):
        for s in range(k_per_arm):
            b = rng.normal(0.0, sigma_b)
            for _ in range(m):
                z = rng.normal(0.0, 1.0)
                rows["u"].append(f"u{uid}")
                rows["solo"].append(f"solo{uid}")
                uid += 1
                rows["g"].append(arm)
                rows["store"].append(f"{arm}_s{s}")
                rows["y"].append(5.0 + 0.5 * z + b + rng.normal(0.0, 0.5))
                rows["z"].append(z)
    return pa.table(rows)


def _source(tbl: pa.Table, cluster: str | None, *, design=None):
    return from_unit_summary(
        tbl,
        unit="u",
        group="g",
        control="C",
        metrics={"y": "mean"},
        cluster=cluster,
        design=design,
    )


def _metric(src):
    return next(m for m in src.context.metrics if m.name == "y")


def _rel_se(est: LiftEstimate) -> float:
    """Delta diagnostic from joint covariance, not an inverted Fieller endpoint."""
    assert est.relative_confidence_set is not None
    reference = est.relative_confidence_set.reference
    ratio = reference.a / reference.c
    variance = math.fsum(
        (reference.var_a, -2 * ratio * reference.cov_ac, ratio**2 * reference.var_c)
    )
    return math.sqrt(variance) / abs(reference.c)


# Cluster identity reaches adjusted estimators as `cluster_id` metadata, never as a
# requested covariate: a conforming source may expose only the canonical column.


@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_adjusted_estimators_never_request_the_declared_cluster_source_column(name):
    tbl = _table(k_per_arm=20, m=8, seed=11)
    backing = _source(tbl, cluster="store")
    requested: list[tuple[str, ...]] = []

    class StrictSource:
        """Refuses any covariate naming the declared cluster SOURCE column."""

        def __getattr__(self, attribute):
            return getattr(backing, attribute)

        def unit_frame(self, metric, *, covariates=(), **kwargs):
            requested.append(tuple(covariates))
            assert "store" not in tuple(covariates)
            return backing.unit_frame(metric, covariates=covariates, **kwargs)

    src = cast(Any, StrictSource())
    metric = _metric(backing)
    method = Method(
        name=name,
        propensity_learner=lambda: ConstantPropensity(0.5),
        outcome_learner=None if name == "iptw" else lambda: ConstantOutcome(0.0),
    )
    (est,) = estimate_ate(src, DESIGN, methods=[method]).results
    assert any("z" in covariates for covariates in requested)
    assert all("store" not in covariates for covariates in requested)
    assert "cluster_id" in backing.unit_frame(metric).column_names
    assert math.isfinite(est.require_lift().value)


# Singleton-cluster identity: global Bessel scaling of the iid score moment.


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_singleton_clusters_reproduce_the_iid_if_variance_up_to_cr1(name):
    """Singleton totals reproduce iid influence variance with Bessel scaling."""
    tbl = _table(k_per_arm=30, m=1, seed=3)
    flat = _source(tbl, cluster=None)
    solo = _source(tbl, cluster="solo")
    e0, m_const = 0.5, 0.0
    if name == "iptw":
        (f,) = iptw_estimate(flat, _metric(flat), DESIGN, learner=lambda: ConstantPropensity(e0))
        (c,) = iptw_estimate(solo, _metric(solo), DESIGN, learner=lambda: ConstantPropensity(e0))
    else:
        estimator = dml_estimate if name == "dml" else aipw_estimate
        (f,) = estimator(
            flat,
            _metric(flat),
            DESIGN,
            propensity_learner=lambda: ConstantPropensity(e0),
            outcome_learner=lambda: ConstantOutcome(m_const),
        )
        (c,) = estimator(
            solo,
            _metric(solo),
            DESIGN,
            propensity_learner=lambda: ConstantPropensity(e0),
            outcome_learner=lambda: ConstantOutcome(m_const),
        )

    assert c.n_clusters == 60
    assert f.n_clusters is None and f.dof is None
    # Singleton totals differ from IID only by the finite-sample Bessel factor.
    assert c.require_lift().value == pytest.approx(f.require_lift().value, rel=1e-12)
    assert c.abs_diff == pytest.approx(f.abs_diff, rel=1e-12)

    assert c.dof is None and c.reference_kind == "normal"
    assert f.abs_se is not None and c.abs_se is not None
    assert c.abs_se == pytest.approx(f.abs_se * math.sqrt(60 / 59), rel=1e-9)
    assert _rel_se(c) == pytest.approx(_rel_se(f) * math.sqrt(60 / 59), rel=1e-9)


# (b) Hand-computed clustered reduction, tiny K.


def test_cluster_sq_matches_hand_computed_k3():
    # psi = [1, 3, 2, 6, 4], clusters (a a b b c): mean 3.2, centered
    # totals a=-2.4, b=1.6, c=0.8 -> sum of squares 8.96.
    psi = np.array([1.0, 3.0, 2.0, 6.0, 4.0])
    inv = np.array([0, 0, 1, 1, 2])
    sum_squared, scale = _cluster_sq(psi, inv, 3)
    assert math.sqrt(sum_squared) * scale == pytest.approx(math.sqrt(8.96), rel=1e-12)


def _exact_centered_totals(values, index, length):
    """Per-value rationals: scores centered over their rows, totalled by cluster."""
    from fractions import Fraction

    exact = [Fraction(value) for value in values.tolist()]
    mean = sum(exact, Fraction(0)) / len(exact)
    centered = [value - mean for value in exact]
    if index is None:
        return centered
    totals = [Fraction(0)] * length
    for value, cluster in zip(centered, index.tolist(), strict=True):
        totals[cluster] += value
    return totals


@pytest.mark.parametrize(
    ("psi", "inv", "rescaled"),
    [
        # Cancelling 1e300 terms leave odd-significand totals that float sums lose.
        ([1e300, 1 / 3, -1e300, 2 / 3, 0.1, -0.0, 7.0, -1 / 7], [0, 0, 0, 1, 1, 2, 2, 3], False),
        # Subnormal totals: the reduction rescales by their peak.
        ([5e-324, -1e-320, 3e-321, 2.5e-315, -7e-310, 1e-312], [0, 0, 1, 1, 2, 3], True),
    ],
)
def test_cluster_sq_totals_are_exact_across_the_double_range(psi, inv, rescaled):
    from fractions import Fraction

    psi, inv = np.array(psi), np.array(inv)
    # The fifth cluster holds no rows.
    centered = _exact_centered_totals(psi, inv, 5)
    normalized, scale = _cluster_sq(psi, inv, 5)
    peak = max(abs(total) for total in centered)
    assert scale == (float(peak) if rescaled else 1.0)
    assert normalized == math.fsum(float(total / Fraction(scale)) ** 2 for total in centered)


def test_joint_totals_are_exact_under_cancellation_on_mixed_supports():
    """A comparison-supported numerator against a whole-cohort denominator,
    clustered and not: 1e18 terms cancel exactly within cluster 0, within
    cluster 1 and across the unclustered cross products."""
    from fractions import Fraction

    from increment.estimation._adjust.common import (
        ClusterSupport,
        _joint_reference_from_influences,
    )
    from increment.estimation.results import _joint_reference_from_exact

    numerator = np.array([1e18, 1 / 3, -1e18, 0.0, 0.0, 0.1, -1 / 7, 5e-324])
    denominator = np.array([0.0, 0.25, 0.0, 4e18, -4e18, -1e-310, 2 / 3, -0.0])
    rows = np.array([0, 1, 2, 5, 6, 7])
    cohort = np.array([0, 0, 0, 1, 1, 2, 2, 3])
    pair = ClusterSupport(rows, np.array([0, 0, 0, 1, 1, 2]), 3)
    layouts = [
        (pair, ClusterSupport(slice(None), cohort, 4)),
        (ClusterSupport(rows, None, None), ClusterSupport(slice(None), None, None)),
    ]
    for numerator_support, denominator_support in layouts:
        index = denominator_support.inv
        totals_c = _exact_centered_totals(denominator, index, denominator_support.k)
        if index is None:
            totals_a = _exact_centered_totals(numerator[rows], None, None)
            aligned_c = [totals_c[i] for i in rows.tolist()]
        else:
            totals_a = _exact_centered_totals(numerator[rows], index[rows], denominator_support.k)
            aligned_c = totals_c
        bessel_a, bessel_c = (
            Fraction(1) if k is None else Fraction(k, k - 1)
            for k in (numerator_support.k, denominator_support.k)
        )
        squared_n = Fraction(len(numerator) ** 2)
        var_a = bessel_a * sum(x * x for x in totals_a) / squared_n
        var_c = bessel_c * sum(x * x for x in totals_c) / squared_n
        cross = sum(x * y for x, y in zip(totals_a, aligned_c, strict=True))
        cov_ac = _bessel_cross(numerator_support.k, denominator_support.k) * cross / squared_n
        expected = _joint_reference_from_exact(
            a=1.0, c=2.0, var_a=var_a, var_c=var_c, cov_ac=cov_ac
        )
        assert expected[0] is not None
        observed = _joint_reference_from_influences(
            numerator,
            denominator,
            numerator_point=1.0,
            denominator_point=2.0,
            supports=(numerator_support, denominator_support),
        )
        assert observed == expected


def test_scorestats_clustered_se_hand_computed():
    # `cluster_variance` is the caller's final CR1-corrected value (8.96 * 3/2);
    # `se()` takes its square root with no further K/(K-1) multiplier.
    clustered = ScoreStats(
        metric="m",
        contrast="T",
        n=5,
        sum_psi=16.0,
        sum_psi2=66.0,
        cluster_variance=8.96 * 1.5,
        n_clusters=3,
    )
    assert clustered.se() == pytest.approx(math.sqrt(8.96 * 1.5) / 5, rel=1e-12)
    # iid twin uses the unit-level centered moment: 66 - 16^2/5 = 14.8.
    iid = ScoreStats(metric="m", contrast="T", n=5, sum_psi=16.0, sum_psi2=66.0)
    assert iid.se() == pytest.approx(math.sqrt(14.8) / 5, rel=1e-12)
    # DML normalizer is sum_d_tilde2, unchanged by clustering.
    dml = ScoreStats(
        metric="m",
        contrast="T",
        n=5,
        sum_psi=16.0,
        sum_psi2=66.0,
        sum_d_tilde2=2.5,
        cluster_variance=8.96 * 1.5,
        n_clusters=3,
    )
    assert dml.se() == pytest.approx(math.sqrt(8.96 * 1.5) / 2.5, rel=1e-12)


def test_scorestats_cluster_fields_must_travel_together():
    with pytest.raises(InvalidRequestError) as exc_info:
        ScoreStats(metric="m", contrast="T", n=5, sum_psi=0.0, sum_psi2=1.0, n_clusters=3)
    assert exc_info.value.code == "estimation.armstats.score_stats.cluster_variance"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_iptw_clustered_se_matches_independent_dense_computation():
    """Known propensity: compare with independent cluster-mean covariance."""
    k, m, e0 = 20, 3, 0.4
    tbl = _table(k_per_arm=k, m=m, seed=11)
    src = _source(tbl, cluster="store")
    (est,) = iptw_estimate(src, _metric(src), DESIGN, learner=lambda: ConstantPropensity(e0))

    g = np.asarray(tbl["g"])
    d = (g == "T").astype(float)
    y = np.asarray(tbl["y"], dtype=float)
    stores = np.asarray(tbl["store"])
    n = d.size
    w1, w0 = d / e0, (1 - d) / (1 - e0)
    mu1 = (w1 * y).sum() / w1.sum()
    mu0 = (w0 * y).sum() / w0.sum()
    lift = (mu1 - mu0) / mu0
    psi1 = (n / w1.sum()) * d * (y - mu1) / e0
    psi0 = (n / w0.sum()) * (1 - d) * (y - mu0) / (1 - e0)
    psi = ((psi1 - psi0) - lift * psi0) / mu0
    centered = psi - psi.mean()
    totals = np.array([centered[stores == s].sum() for s in np.unique(stores)])
    k_total = 2 * k
    cr1 = k_total / (k_total - 1)
    se = math.sqrt((totals**2).sum() * cr1) / n

    assert est.n_clusters == k_total
    assert est.dof is None and est.reference_kind == "normal"
    assert est.require_lift().value == pytest.approx(lift, rel=1e-12)
    assert _rel_se(est) == pytest.approx(se, rel=1e-9)
    # And the abs channel: same reduction over the tau-scale IF.
    tot_tau = np.array(
        [(psi1 - psi0 - (psi1 - psi0).mean())[stores == s].sum() for s in np.unique(stores)]
    )
    assert est.abs_se == pytest.approx(math.sqrt((tot_tau**2).sum() * cr1) / n, rel=1e-9)


# Small-K policy (shared with the randomized engine) and trim recounting.


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_below_ten_total_clusters_is_admitted_for_iptw():
    tbl = _table(k_per_arm=3, m=10, seed=5)
    src = _source(tbl, cluster="store")
    with pytest.warns(IncrementRuntimeWarning) as rec:
        (result,) = iptw_estimate(src, _metric(src), DESIGN, learner=ConstantPropensity)
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    assert result.n_clusters == 6


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_below_forty_total_clusters_warns_of_over_rejection():
    tbl = _table(k_per_arm=10, m=2, seed=5)
    src = _source(tbl, cluster="store")
    with pytest.warns(RuntimeWarning):
        (result,) = estimate_ate(
            src, DESIGN, methods=[Method(name="iptw", propensity_learner=ConstantPropensity)]
        ).results
    assert result.n_clusters == 20


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_small_cluster_advisory_names_the_estimate_ate_caller():
    tbl = _table(k_per_arm=3, m=10, seed=5)
    src = _source(tbl, cluster="store")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate_ate(
            src, DESIGN, methods=[Method(name="iptw", propensity_learner=ConstantPropensity)]
        )
    advisories = [
        w
        for w in caught
        if getattr(w.message, "code", None) == "estimation.engine.small_total_clusters"
    ]
    assert advisories
    assert all(w.filename == __file__ for w in advisories)


@pytest.mark.parametrize("name", ["dml", "aipw"])
def test_cross_fit_cluster_advisory_emitted_once(name):
    tbl = _table(k_per_arm=10, m=2, seed=5)
    src = _source(tbl, cluster="store")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ESTIMATORS[name](src, _metric(src), DESIGN)
    codes = warning_codes(caught)
    advisories = [c for c in codes if c == "estimation.engine.small_total_clusters"]
    assert len(advisories) == 1


def test_unit_weight_diagnostics_use_the_actual_arm_weights():
    tbl = pa.table(
        {
            "u": [f"u{i}" for i in range(24)],
            "g": ["C"] * 12 + ["T"] * 12,
            "y": [float(i % 7 + 1) for i in range(24)],
            "z": [0.0] * 24,
        }
    )
    analysis = Analysis.from_unit_summary(
        tbl,
        unit="u",
        group="g",
        metrics={"y": "mean"},
        design=DESIGN,
    )
    results = analysis.run(
        decision_method=Method(name="iptw"),
        sensitivity_methods=[Method(name="unadjusted")],
    )
    (row,) = [result for result in results if result.method == "iptw"]
    unweighted = [result for result in results if result.method == "unadjusted"]

    restored = ReadoutResults.model_validate_json(results.model_dump_json())
    frame = results.to_frame()
    restored_weighted = next(result for result in restored if result.method == "iptw")
    frame_weighted = frame.loc[frame["method"] == "iptw"].iloc[0]
    for field in (
        "control_weight_n",
        "treatment_weight_n",
        "control_weight_ess",
        "treatment_weight_ess",
        "control_weight_max_share",
        "treatment_weight_max_share",
    ):
        assert getattr(restored_weighted, field) == getattr(row, field)
        assert frame_weighted[field] == pytest.approx(getattr(row, field))
    assert unweighted
    assert all(result.weight_diagnostics_available is False for result in unweighted)
    assert all(
        result.weight_diagnostics_reason_code == "readout.weight_diagnostics.not_applicable"
        for result in unweighted
    )
    assert all(
        result.weight_diagnostics_reason_context == {"method": "unadjusted"}
        for result in unweighted
    )

    with pytest.raises(TypeError):
        unweighted[0].weight_diagnostics_reason_context["method"] = "iptw"
    assert all(
        result.weight_diagnostics_reason_context == {"method": "unadjusted"}
        for result in unweighted
    )
    assert unweighted[0].model_dump()["weight_diagnostics_reason_context"] == {
        "method": "unadjusted"
    }
    assert all(result.control_weight_ess is None for result in unweighted)
    replay_unweighted = [result for result in restored if result.method == "unadjusted"]
    assert all(result.weight_diagnostics_available is False for result in replay_unweighted)
    assert all(
        result.weight_diagnostics_reason_code == "readout.weight_diagnostics.not_applicable"
        for result in replay_unweighted
    )
    assert all(
        result.weight_diagnostics_reason_context == {"method": "unadjusted"}
        for result in replay_unweighted
    )
    frame_unweighted = frame.loc[frame["method"] == "unadjusted"]
    assert frame_unweighted["weight_diagnostics_available"].eq(False).all()
    assert (
        frame_unweighted["weight_diagnostics_reason_code"]
        .eq("readout.weight_diagnostics.not_applicable")
        .all()
    )
    assert frame_unweighted["control_weight_ess"].isna().all()
    assert all(
        json.loads(context) == {"method": "unadjusted"}
        for context in frame_unweighted["weight_diagnostics_reason_context"]
    )

    assert row.weight_diagnostics_available is True
    assert row.weight_grain == "unit"
    assert row.control_weight_n == row.treatment_weight_n == 12
    assert row.control_weight_ess == pytest.approx(12)
    assert row.treatment_weight_ess == pytest.approx(12)
    assert row.control_weight_max_share == pytest.approx(1 / 12)
    assert row.treatment_weight_max_share == pytest.approx(1 / 12)


def test_overlap_trim_recounts_clusters():
    """K is the clusters actually contributing IF terms: trimming a
    singleton cluster to overlap drops it from n_clusters and the dof."""

    class PresetPropensity:
        def __init__(self, e):
            self._e = np.asarray(e, dtype=float)

        def fit(self, X, d):
            pass

        def predict(self, X):
            return self._e

    tbl = _table(k_per_arm=21, m=1, seed=7)
    e = np.full(42, 0.5)
    e[-1] = 0.999  # one treated singleton outside the overlap gate
    design = DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
    src = _source(tbl, cluster="store")
    (est,) = iptw_estimate(src, _metric(src), design, learner=lambda: PresetPropensity(e))

    assert est.weight_diagnostics_available is True
    assert est.weight_grain == "cluster"
    assert est.control_weight_n == 21
    assert est.treatment_weight_n == 20
    assert est.control_weight_ess == pytest.approx(21)
    assert est.treatment_weight_ess == pytest.approx(20)
    assert est.control_weight_max_share == pytest.approx(1 / 21)
    assert est.treatment_weight_max_share == pytest.approx(1 / 20)
    assert est.n_clusters == 41

    kept = np.arange(42) != 41
    y = np.asarray(tbl["y"], dtype=float)[kept]
    treated = np.asarray(tbl["g"])[kept] == "T"
    assert treated.sum() == 20 and (~treated).sum() == 21
    assert est.require_lift().value == pytest.approx(y[treated].mean() / y[~treated].mean() - 1)
    assert est.dof is None and est.reference_kind == "normal"


def _three_arm_cluster_table(control_stores: int) -> pa.Table:
    """Arm-pure stores of three members each: `control_stores` control stores
    and two stores in each of the treatments T and T2."""
    stores = {
        "C": [f"C_s{s}" for s in range(control_stores)],
        "T": ["T_s0", "T_s1"],
        "T2": ["T2_s0", "T2_s1"],
    }
    rows: dict[str, list] = {"u": [], "g": [], "store": [], "y": [], "z": []}
    for arm, labels in stores.items():
        for label in labels:
            for member in range(3):
                uid = len(rows["u"])
                rows["u"].append(f"u{uid}")
                rows["g"].append(arm)
                rows["store"].append(label)
                rows["y"].append(5.0 + uid % 4 + (arm != "C"))
                rows["z"].append(float(member))
    return pa.table(rows)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_other_treatment_clusters_never_count_as_control_support():
    """Each comparison counts pure-cluster support on its own treatment and
    control rows: T2's two stores cannot stand in for a second control store,
    and a second control store admits both comparisons."""
    src = _source(_three_arm_cluster_table(control_stores=1), cluster="store")
    with pytest.raises(InvalidRequestError) as exc_info:
        iptw_estimate(src, _metric(src), DESIGN, learner=ConstantPropensity)
    assert exc_info.value.code == "estimation.adjust_overlap.cluster_arm_needs_two"
    assert (exc_info.value.context["k_t"], exc_info.value.context["k_c"]) == (2, 1)

    src = _source(_three_arm_cluster_table(control_stores=2), cluster="store")
    rows = iptw_estimate(src, _metric(src), DESIGN, learner=ConstantPropensity)
    assert [row.group_id for row in rows] == ["T", "T2"]
    assert all(row.weight_diagnostics_available is True for row in rows)
    assert all(row.weight_grain == "cluster" for row in rows)
    assert rows[0].control_weight_n == rows[1].control_weight_n == 2
    assert rows[0].control_weight_ess == pytest.approx(rows[1].control_weight_ess)
    assert rows[0].control_weight_max_share == pytest.approx(rows[1].control_weight_max_share)
    assert all(row.treatment_weight_n == 2 for row in rows)
    assert all(row.treatment_weight_ess == pytest.approx(2) for row in rows)
    assert all(row.treatment_weight_max_share == pytest.approx(0.5) for row in rows)


class _SeparatingPropensity:
    """Conditional propensity 0.999 where z > 0.5, else 0.5: both comparisons'
    odds reach 999 there, so the marginal control propensity is 1/1999."""

    def fit(self, X, d):
        pass

    def predict(self, X):
        return np.where(np.asarray(X)[:, 0] > 0.5, 0.999, 0.5)


class _LineOutcome:
    """Least-squares line in the single covariate, refitted on each training set."""

    def fit(self, X, d):
        self._coef = np.polyfit(np.asarray(X)[:, 0], np.asarray(d, dtype=float), 1)

    def predict(self, X):
        return np.polyval(self._coef, np.asarray(X)[:, 0])


def _comparison_store_table(comparison_stores: tuple[str, ...]) -> pa.Table:
    """Ten control and ten T units in each mixed comparison store (z near 0
    in the first, near 1 in any later one); three T2 units share the first
    store, and further T2 units fill twenty stores of three (z near 0)."""
    rng = np.random.default_rng(29)
    rows: dict[str, list] = {"u": [], "g": [], "store": [], "y": [], "z": []}

    def add(arm: str, store: str, shift: float) -> None:
        rows["u"].append(f"u{len(rows['u'])}")
        rows["g"].append(arm)
        rows["store"].append(store)
        rows["y"].append(5.0 + (arm != "C") + rng.normal(0.0, 1.0))
        rows["z"].append(shift + rng.uniform(-0.2, 0.2))

    for i, store in enumerate(comparison_stores):
        for arm in ("C", "T"):
            for _ in range(10):
                add(arm, store, float(i > 0))
    for _ in range(3):
        add("T2", comparison_stores[0], 0.0)
    for s in range(20):
        for _ in range(3):
            add("T2", f"T2_s{s}", 0.0)
    return pa.table(rows)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("case", ["logistic_iptw", "trimmed_aipw"])
def test_comparison_in_one_cluster_refuses_despite_other_treatment_clusters(case):
    """T and its control units share one store, so the T comparison refuses
    on its own count. The T2 comparison is admissible under the count and
    purity rules alike: its units share that store and fill twenty more, so
    it spans 21 stores that are not all arm-pure, and the refusal does not
    depend on which comparison is checked first. Within the one store the T
    comparison's self-normalized residual totals cancel in exact arithmetic,
    so a standard error could come only from the fitted-logistic estimating
    equations, which reach every cohort store, or from rounding (AIPW with
    constant outcome predictions). Default logistic IPTW sees one comparison
    store from the start; AIPW only once trimming drops the store whose units
    lose marginal control support."""
    if case == "logistic_iptw":
        src = _source(_comparison_store_table(("M0",)), cluster="store", design=DESIGN)
        with pytest.raises(InvalidRequestError) as exc_info:
            iptw_estimate(src, _metric(src), DESIGN)
    else:
        design = DESIGN.model_copy(update={"gate": IdentificationGate(overlap="trim")})
        src = _source(_comparison_store_table(("M0", "M1")), cluster="store", design=design)
        with pytest.raises(InvalidRequestError) as exc_info:
            aipw_estimate(
                src,
                _metric(src),
                design,
                propensity_learner=_SeparatingPropensity,
                outcome_learner=ConstantOutcome,
                folds=2,
            )
    assert exc_info.value.code == "estimation.adjust_overlap.cluster_needs_two"
    assert exc_info.value.context["k"] == 1


def _extraneous_store_table(*, linear: bool = False) -> pa.Table:
    """Arm-pure stores of three: three control and three T stores, plus forty
    stores holding only T2. Outcomes are arm lines in z (intercepts 2, 3, 9;
    slopes 1, 3, -1), plus store shocks and unit noise unless ``linear``."""
    rng = np.random.default_rng(23)
    lines = {"C": (2.0, 1.0), "T": (3.0, 3.0), "T2": (9.0, -1.0)}
    rows: dict[str, list] = {"u": [], "g": [], "store": [], "y": [], "z": []}
    for arm, count in (("C", 3), ("T", 3), ("T2", 40)):
        intercept, slope = lines[arm]
        for s in range(count):
            shock = 0.0 if linear else rng.normal(0.0, 1.0)
            for _ in range(3):
                z = rng.normal(0.0, 1.0)
                noise = 0.0 if linear else rng.normal(0.0, 0.5)
                rows["u"].append(f"u{len(rows['u'])}")
                rows["g"].append(arm)
                rows["store"].append(f"{arm}_s{s}")
                rows["y"].append(intercept + slope * z + shock + noise)
                rows["z"].append(z)
    return pa.table(rows)


def _store_totals(tbl: pa.Table, scores: np.ndarray) -> np.ndarray:
    """Totals of per-unit scores within each store."""
    stores = np.asarray(tbl["store"])
    return np.array([scores[stores == s].sum() for s in np.unique(stores)])


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("name", "value_scale"), [("iptw", "relative"), ("dml", "absolute"), ("dml", "relative")]
)
def test_comparison_scores_keep_their_own_cluster_count(name, value_scale):
    """The fixed-propensity IPTW contrast and the DML slope vanish outside the
    T comparison, so forty stores holding only T2 add nothing to their totals:
    the T row keeps the six-store analysis, K = 6 in K/(K-1), in n_clusters
    and in the single small-cluster advisory, although the cohort has 46
    stores. Only a relative DML row's control mean draws on every store."""
    tbl = _extraneous_store_table()
    src = _source(tbl, cluster="store", design=DESIGN)
    g = np.asarray(tbl["g"])
    y = np.asarray(tbl["y"], dtype=float)
    treated, control = (g == "T").astype(float), (g == "C").astype(float)
    cr1 = 6 / 5
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if name == "iptw":
            rows = iptw_estimate(src, _metric(src), DESIGN, learner=ConstantPropensity)
        else:
            rows = dml_estimate(
                src,
                _metric(src),
                DESIGN,
                propensity_learner=ConstantPropensity,
                outcome_learner=ConstantOutcome,
                folds=3,
                value_scale=value_scale,
            )
    est = rows[0]
    assert est.group_id == "T"
    advisory = warning_context(caught, "estimation.engine.small_total_clusters")
    assert advisory["n_clusters"] == 6
    if name == "iptw":
        # Constant marginal propensities (1/3 each): Hajek means are arm means,
        # and each influence over N is the deviation from its arm mean over
        # that arm's size.
        phi_t = treated * (y - y[g == "T"].mean()) / treated.sum()
        phi_c = control * (y - y[g == "C"].mean()) / control.sum()
        control_meat = cr1 * (_store_totals(tbl, phi_c) ** 2).sum()
        assert est.n_clusters == 6
        assert est.abs_se == pytest.approx(
            math.sqrt(cr1 * (_store_totals(tbl, phi_t - phi_c) ** 2).sum()), rel=1e-9
        )
        assert est.relative_confidence_set is not None
        reference = est.relative_confidence_set.reference
        assert reference.var_c == pytest.approx(control_meat, rel=1e-9)
        assert reference.cov_ac == pytest.approx(-control_meat, rel=1e-9)
        return
    pair = g != "T2"
    d_tilde = np.where(pair, treated - 0.5, 0.0)
    normalizer = (d_tilde**2).sum()
    theta = (d_tilde * y).sum() / normalizer
    psi = d_tilde * (y - theta * d_tilde)
    se = math.sqrt(cr1 * (_store_totals(tbl, psi) ** 2).sum()) / normalizer
    if value_scale == "absolute":
        lift = est.require_lift()
        assert lift.lb is not None and lift.ub is not None
        assert lift.value == pytest.approx(theta, rel=1e-12)
        assert (lift.ub - lift.lb) / (2 * norm.isf(0.025)) == pytest.approx(se, rel=1e-9)
        assert est.n_clusters == 6
    else:
        assert est.abs_diff == pytest.approx(theta, rel=1e-12)
        assert est.abs_se == pytest.approx(se, rel=1e-9)
        assert est.n_clusters == 46


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_aipw_population_terms_reach_stores_holding_only_other_treatments():
    """Exact arm lines leave AIPW influences with population terms only:
    (3 - 1)(z - mean z) for the T contrast and (z - mean z) for the control
    mean, on every cohort unit. Stores holding only T2 carry them, so the
    covariance sums all 46 stores with K = 46, not the comparison's six."""
    tbl = _extraneous_store_table(linear=True)
    src = _source(tbl, cluster="store", design=DESIGN)
    est = aipw_estimate(
        src,
        _metric(src),
        DESIGN,
        propensity_learner=ConstantPropensity,
        outcome_learner=_LineOutcome,
        folds=3,
    )[0]
    z = np.asarray(tbl["z"], dtype=float)
    n = z.size
    contrast = _store_totals(tbl, 2.0 * (z - z.mean()))
    control = _store_totals(tbl, z - z.mean())
    cr1 = 46 / 45
    assert est.group_id == "T"
    assert est.n_clusters == 46
    assert est.abs_se == pytest.approx(math.sqrt(cr1 * (contrast**2).sum()) / n, rel=1e-9)
    assert est.relative_confidence_set is not None
    reference = est.relative_confidence_set.reference
    assert reference.var_c == pytest.approx(cr1 * (control**2).sum() / n**2, rel=1e-9)
    assert reference.cov_ac == pytest.approx(cr1 * (contrast * control).sum() / n**2, rel=1e-9)


def _three_arm_singleton_table() -> pa.Table:
    """Thirty units per arm, each its own cluster, with covariate-dependent effects."""
    rng = np.random.default_rng(31)
    rows: dict[str, list] = {"u": [], "g": [], "solo": [], "y": [], "z": []}
    for arm, effect in (("C", 0.0), ("T", 1.0), ("T2", 2.5)):
        for _ in range(30):
            z = rng.normal(0.0, 1.0)
            uid = len(rows["u"])
            rows["u"].append(f"u{uid}")
            rows["g"].append(arm)
            rows["solo"].append(f"solo{uid}")
            rows["y"].append(5.0 + 0.5 * z + effect * (1.0 + 0.3 * z) + rng.normal(0.0, 0.5))
            rows["z"].append(z)
    return pa.table(rows)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["iptw", "dml"])
def test_multi_arm_singleton_clusters_scale_each_influence_by_its_own_support(name):
    """Singleton clusters reproduce the iid joint covariance up to each
    influence's Bessel factor. Default logistic IPTW's fitted-propensity terms
    and the DML control mean reach all 90 units (90/89); the DML slope only its
    comparison's 60 (60/59), with the geometric mean on their covariance."""
    tbl = _three_arm_singleton_table()
    estimator = iptw_estimate if name == "iptw" else dml_estimate
    flat, solo = _source(tbl, cluster=None), _source(tbl, cluster="solo")
    f = estimator(flat, _metric(flat), DESIGN)[0]
    c = estimator(solo, _metric(solo), DESIGN)[0]
    cohort = 90 / 89
    slope = cohort if name == "iptw" else 60 / 59
    assert c.group_id == f.group_id == "T"
    assert c.n_clusters == 90
    assert f.relative_confidence_set is not None and c.relative_confidence_set is not None
    iid, clustered = f.relative_confidence_set.reference, c.relative_confidence_set.reference
    assert clustered.a == pytest.approx(iid.a, rel=1e-12)
    assert clustered.var_a == pytest.approx(iid.var_a * slope, rel=1e-9)
    assert clustered.var_c == pytest.approx(iid.var_c * cohort, rel=1e-9)
    assert clustered.cov_ac == pytest.approx(iid.cov_ac * math.sqrt(slope * cohort), rel=1e-9)


@pytest.mark.parametrize(("k_a", "k"), [(2, 3), (2, 50), (2, 2**40)])
def test_mixed_support_cross_factor_preserves_psd_without_losing_precision(k_a, k):
    """Upward rounding makes a rank-one cluster Gram matrix indefinite."""
    from fractions import Fraction

    product = Fraction(k_a, k_a - 1) * Fraction(k, k - 1)
    cross = _bessel_cross(k_a, k)
    assert cross * cross <= product
    assert float(cross) == pytest.approx(math.sqrt(float(product)), rel=1e-15, abs=0.0)


# Refusals kept, removed, and rewired.


def test_encouragement_design_with_cluster_constructs_at_entry():
    """Non-CUPED, non-ratio LATE now composes with a declared cluster.
    See tests/estimation/test_late_cluster.py for the estimation-side
    coverage; this only pins the entry gate no longer refusing."""
    tbl = _table(k_per_arm=20, m=2, seed=9)
    tbl = tbl.append_column("took", pa.array([1] * tbl.num_rows))
    design = Encouragement(
        control_group="C",
        uptake=UptakeSpec(fact="took"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="test fixture, not a real design"
        ),
    )
    src = from_unit_summary(
        tbl,
        unit="u",
        group="g",
        control="C",
        metrics={"y": "mean"},
        cluster="store",
        design=design,
        uptake="took",
    )
    assert src.context.cluster == "store"


def test_ratio_metric_refuses_at_validation_with_cluster_under_observational():
    """An all-ratio clustered request refuses before adjustment estimation."""
    tbl = _table(k_per_arm=20, m=2, seed=9)
    tbl = tbl.append_column("sessions", pa.array([2.0] * tbl.num_rows))
    src = from_unit_summary(
        tbl,
        unit="u",
        group="g",
        control="C",
        metrics=[MetricSpec(name="rps", type="ratio", numerator="y", denominator="sessions")],
        cluster="store",
        design=DESIGN,
    )
    assert src.context.cluster == "store"
    for method in ("iptw", "dml", "aipw"):
        with pytest.raises(UnsupportedRequestError) as exc_info:
            readouts.run(src, decision_method=Method(name=method))
        assert exc_info.value.code == "estimation.adjust_common.supported_ratio_metric"


def test_informative_prior_refuses_with_cluster():
    tbl = _table(k_per_arm=20, m=2, seed=9)
    src = _source(tbl, cluster="store")
    with pytest.raises(CapabilityError) as raised:
        estimate_ate(src, DESIGN, prior=Normal(mu=0.0, sigma=0.1))

    error = raised.value
    assert error.code == "arm.adjustment.cluster_prior"
    assert error.context.get("cluster") == "store"


# Decision-stat refusals hold on the clustered adjusted rows (both the
# relative posterior recovery and the direct absolute-margin branch).


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_decision_stats_refuse_on_clustered_adjusted_rows():
    tbl = _table(k_per_arm=20, m=2, seed=13)
    src = from_unit_summary(
        tbl,
        unit="u",
        group="g",
        control="C",
        metrics=[MetricSpec(name="y", preferred_direction="increase")],
        cluster="store",
        design=DESIGN,
    )
    (est,) = estimate_ate(
        src,
        DESIGN,
        methods=[Method(name="iptw", propensity_learner=ConstantPropensity)],
        null_abs={"y": 0.01},
    ).results

    for method, code in (
        (est.chance_to_beat, "estimation.results.lift.posterior_decision_stats_cluster_robust"),
        (est.prob_favorable, "estimation.results.lift.p_value_cluster_robust_null_abs"),
    ):
        with pytest.raises(InvalidRequestError) as exc_info:
            method()
        assert exc_info.value.code == code
        assert est.reference_df is None


# Orchestration: estimate_ate / readouts.run thread the declaration.


def test_unadjusted_joint_route_preserves_pure_arm_welch_comparison():
    """Pure arms retain their independent Bessel/Welch approximation."""
    tbl = _table(k_per_arm=20, m=2, seed=15)
    src = _source(tbl, cluster="store")
    (est,) = estimate_ate(src, DESIGN, methods=[Method(name="unadjusted")]).results
    assert est.method == "unadjusted"
    assert est.n_clusters == 40

    g = np.asarray(tbl["g"])
    stores = np.asarray(tbl["store"])
    uniq = np.unique(stores)
    is_t = np.array([bool((g[stores == s] == "T")[0]) for s in uniq])
    k_t, k_c = int(is_t.sum()), int((~is_t).sum())

    expected_dof = float(min(k_t - 1, k_c - 1))
    assert est.dof == pytest.approx(expected_dof)
    from increment.estimation.engine import estimate_lift

    (separate,) = estimate_lift(
        [_metric(src)],
        src.moments(_metric(src)),
        "C",
        cluster="store",
    ).results
    assert est.require_lift().value == pytest.approx(separate.require_lift().value)
    assert est.require_lift().lb == pytest.approx(separate.require_lift().lb)
    assert est.require_lift().ub == pytest.approx(separate.require_lift().ub)
    assert est.abs_diff == pytest.approx(separate.abs_diff)
    assert est.abs_se == pytest.approx(separate.abs_se)
    assert est.abs_reference_df == pytest.approx(separate.abs_reference_df)
    assert est.abs_lb == pytest.approx(separate.abs_lb)
    assert est.abs_ub == pytest.approx(separate.abs_ub)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_readouts_run_clusters_the_observational_path_end_to_end():
    """The actual readout preserves the cluster and covariance provenance."""
    tbl = _table(k_per_arm=20, m=5, seed=17)
    src = from_unit_summary(
        tbl,
        unit="u",
        group="g",
        control="C",
        metrics={"y": "mean"},
        cluster="store",
        design=DESIGN,
    )
    (clustered,) = readouts.run(src)
    (flat,) = readouts.run(_source(tbl, cluster=None, design=DESIGN))
    assert clustered.method == "iptw"
    assert clustered.n_clusters == 40

    assert clustered.dof is None and clustered.reference_kind == "normal"

    clustered_lift, flat_lift = clustered.require_lift(), flat.require_lift()
    assert clustered_lift.value == pytest.approx(flat_lift.value, rel=1e-9)
    assert clustered_lift.ub is not None and clustered_lift.lb is not None
    assert flat_lift.ub is not None and flat_lift.lb is not None
    # Real cluster effects (sigma_b = 2x the unit noise) at m=5 units per
    # cluster: the clustered interval must be genuinely wider.
    assert clustered_lift.ub - clustered_lift.lb > 1.5 * (flat_lift.ub - flat_lift.lb)


class _StrictAdjustedUnitSource:
    """Custom source that serves declared covariates and canonical clusters only."""

    def __init__(self, table):
        from dataclasses import replace

        self.table = table
        self._cluster_by_unit = dict(
            zip(table["u"].to_pylist(), table["store"].to_pylist(), strict=True)
        )

        self.source = _source(table, None, design=DESIGN)
        self.context = replace(self.source.context, cluster="store")
        self.capabilities: frozenset[Grain] = self.source.capabilities
        self.operations: frozenset[SourceOperation] = self.source.operations

    @property
    def breakouts(self):
        return self.source.breakouts

    @property
    def shape(self):
        return self.source.shape

    def unit_frame(self, metric, *, covariates=()):
        requested = tuple(covariates)
        if requested != ("z",):
            raise ValueError(f"undeclared covariates requested: {requested!r}")
        frame = self.source.unit_frame(metric, covariates=requested)
        return frame.append_column(
            "cluster_id",
            pa.array([self._cluster_by_unit[u] for u in frame["unit_id"].to_pylist()]),
        )

    def cluster_counts(self):
        clusters: dict[str, set[str]] = {}
        for group, cluster in zip(
            self.table["g"].to_pylist(), self.table["store"].to_pylist(), strict=True
        ):
            clusters.setdefault(group, set()).add(cluster)
        return {group: len(values) for group, values in clusters.items()}

    def close(self) -> None:
        self.source.close()

    def moments(self, metric, **kwargs):
        return self.source.moments(metric, **kwargs)

    def unit_counts(self):
        return self.source.unit_counts()

    def compliance_dates(self):
        return self.source.compliance_dates()

    def compliance_summary(self, design, *, as_of=None, completed_windows_only=False):
        return self.source.compliance_summary(
            design, as_of=as_of, completed_windows_only=completed_windows_only
        )

    def sql(self, *, grain="total"):
        return self.source.sql(grain=grain)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_adjusted_cluster_source_uses_declared_covariates_and_canonical_ids(name):
    """Adjusted estimators request z, consume cluster_id, and retain cluster K."""
    table = _table(k_per_arm=10, m=2, seed=19)
    source = _StrictAdjustedUnitSource(table)

    reference_source = _source(table, "store", design=DESIGN)
    method = Method(
        name=name,
        propensity_learner=ConstantPropensity,
        outcome_learner=None if name == "iptw" else ConstantOutcome,
        folds=None if name == "iptw" else 2,
    )
    with pytest.warns(RuntimeWarning):
        (estimate,) = estimate_ate(source, DESIGN, methods=[method]).results
        (reference,) = estimate_ate(reference_source, DESIGN, methods=[method]).results
    assert estimate.require_lift() == reference.require_lift()
    assert estimate.relative_confidence_set == reference.relative_confidence_set
    assert estimate.n_clusters == 20


class _JointUnitSource:
    """Unit-frame source for dependence clusters spanning observed arms.

    The public dataframe constructor still enforces randomization-cluster
    purity. This adapter exercises estimate_ate without changing that gate.
    """

    def __init__(self, table, *, missing="error", ratio=False):
        from dataclasses import replace

        spec = (
            MetricSpec(name="y", type="ratio", numerator="y", denominator="den", missing=missing)
            if ratio
            else MetricSpec(name="y", missing=missing)
        )
        self.cluster_by_unit = dict(
            zip(table["u"].to_pylist(), table["store"].to_pylist(), strict=True)
        )
        self.source = from_unit_summary(
            table,
            unit="u",
            group="g",
            control="C",
            metrics=[spec],
            design=DESIGN,
        )
        self.context = replace(self.source.context, cluster="store")
        self.capabilities: frozenset[Grain] = self.source.capabilities
        self.operations: frozenset[SourceOperation] = self.source.operations
        self.requests = []

    @property
    def breakouts(self):
        return self.source.breakouts

    @property
    def shape(self):
        return self.source.shape

    def unit_frame(self, metric, *, covariates=()):
        self.requests.append(tuple(covariates))
        assert not covariates, "cluster metadata must not be requested as a covariate"
        frame = self.source.unit_frame(metric)
        assert isinstance(frame, pa.Table)
        return frame.append_column(
            "cluster_id", pa.array([self.cluster_by_unit[u] for u in frame["unit_id"].to_pylist()])
        )

    def moments(self, metric, **kwargs):
        raise AssertionError("joint unadjusted inference must consume unit_frame")

    def close(self) -> None:
        self.source.close()

    def unit_counts(self):
        return self.source.unit_counts()

    def cluster_counts(self):
        return self.source.cluster_counts()

    def compliance_dates(self):
        return self.source.compliance_dates()

    def compliance_summary(self, design, *, as_of=None, completed_windows_only=False):
        return self.source.compliance_summary(
            design, as_of=as_of, completed_windows_only=completed_windows_only
        )

    def sql(self, *, grain="total"):
        return self.source.sql(grain=grain)

    def __getattr__(self, name):
        return getattr(self.source, name)


def _joint_table(*, unequal=False, sign=1, ratio=False):
    rows = []
    for cluster in range(10):
        shock = cluster % 5 - 2
        for group in ("C", "T"):
            count = (1 + (cluster + (group == "T")) % 3) if unequal else 1
            for member in range(count):
                y = 10 + shock if group == "C" else 12 + sign * 2 * shock + member
                rows.append(
                    {
                        "u": f"{group}-{cluster}-{member}",
                        "g": group,
                        "store": f"s{cluster}",
                        "y": float(y),
                        "z": None,
                        "den": float(1 + (cluster + member) % 3) if ratio else 1.0,
                    }
                )
    return pa.Table.from_pylist(rows)


def _joint_oracle(table, *, ratio=False):
    """Exact rational raw-record covariance, independent of production helpers."""
    from fractions import Fraction

    rows = table.to_pylist()
    clusters = {r["store"] for r in rows}
    means, contributions, masses, membership = {}, {}, {}, {}
    for group in ("C", "T"):
        arm = [r for r in rows if r["g"] == group]
        denominator = sum(Fraction(r["den"]) if ratio else Fraction(1) for r in arm)
        membership[group] = {r["store"] for r in arm}
        means[group] = sum(Fraction(r["y"]) for r in arm) / denominator
        masses[group] = {
            cluster: sum(Fraction(r["den"]) if ratio else 1 for r in arm if r["store"] == cluster)
            / denominator
            for cluster in clusters
        }
        contributions[group] = {
            cluster: sum(
                Fraction(r["y"]) - means[group] * (Fraction(r["den"]) if ratio else 1)
                for r in arm
                if r["store"] == cluster
            )
            / denominator
            for cluster in clusters
        }

    def covariance(a, b):
        shared = sorted(membership[a] & membership[b])
        # Build each shock's response in every observed residual product.
        # Solve its moment equations by exact elimination, without a closed-form multiplier.
        equations = [
            [(int(i == j) - masses[a][i]) * (int(i == j) - masses[b][i]) for i in shared]
            + [Fraction(1)]
            for j in shared
        ]
        for column in range(len(shared)):
            pivot = next(i for i in range(column, len(shared)) if equations[i][column])
            equations[column], equations[pivot] = equations[pivot], equations[column]
            divisor = equations[column][column]
            equations[column] = [value / divisor for value in equations[column]]
            for i in range(len(shared)):
                if i != column:
                    factor = equations[i][column]
                    equations[i] = [
                        x - factor * y for x, y in zip(equations[i], equations[column], strict=True)
                    ]
        return sum(
            equations[i][-1] * contributions[a][g] * contributions[b][g]
            for i, g in enumerate(shared)
        )

    tt, cc, tc = covariance("T", "T"), covariance("C", "C"), covariance("T", "C")
    absolute = tt + cc - 2 * tc
    log = tt / means["T"] ** 2 + cc / means["C"] ** 2 - 2 * tc / (means["T"] * means["C"])
    return {
        "difference": float(means["T"] - means["C"]),
        "lift": float(means["T"] / means["C"] - 1),
        "absolute_variance": float(absolute),
        "log_variance": float(log),
        "control_variance": float(cc),
        "contrast_control_covariance": float(tc - cc),
        "joint_psd": absolute >= 0 and absolute * cc >= (tc - cc) ** 2,
        "cross": float(tc),
        "independent_absolute_variance": float(tt + cc),
    }


class _RawJointUnitSource(_JointUnitSource):
    """Serve extreme finite unit values without constructing unused moments."""

    def __init__(self, table):
        super().__init__(_joint_table())
        self.table = table

    def unit_frame(self, metric, *, covariates=()):
        self.requests.append(tuple(covariates))
        assert not covariates, "cluster metadata must not be requested as a covariate"
        return self.table.select(["u", "g", "y", "store"]).rename_columns(
            ["unit_id", "group_id", "y", "cluster_id"],
        )


def _joint_estimate(table, *, missing="error", ratio=False, raw=False, **kwargs) -> LiftEstimate:
    source = (
        _RawJointUnitSource(table) if raw else _JointUnitSource(table, missing=missing, ratio=ratio)
    )
    (row,) = estimate_ate(source, DESIGN, methods=[Method(name="unadjusted")], **kwargs).results
    return row


def _assert_joint_covariance(row, oracle):
    if not oracle["joint_psd"]:
        assert row.relative_confidence_set is None
        assert row.relative_unavailable_reason == "joint_covariance_indefinite"
        assert (
            row.lift is not None and row.require_lift().lb is None and row.require_lift().ub is None
        )
        return
    assert row.relative_confidence_set is not None
    reference = row.relative_confidence_set.reference
    assert reference.var_a == pytest.approx(oracle["absolute_variance"])
    assert reference.var_c == pytest.approx(oracle["control_variance"])
    assert reference.cov_ac == pytest.approx(oracle["contrast_control_covariance"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_joint_signed_covariance_matches_exact_oracle(sign, ratio):
    table = _joint_table(sign=sign, ratio=ratio)
    expected = _joint_oracle(table, ratio=ratio)
    row = _joint_estimate(table, ratio=ratio)
    assert row.n_clusters == 10
    assert row.lift is not None
    assert row.abs_diff is not None and row.abs_se is not None
    assert row.require_lift().value == pytest.approx(expected["lift"])
    assert row.abs_diff == pytest.approx(expected["difference"])
    assert row.abs_se**2 == pytest.approx(expected["absolute_variance"])
    _assert_joint_covariance(row, expected)
    assert row.reference_kind == row.abs_reference_kind == "normal"
    assert row.reference_df is None and row.abs_reference_df is None
    assert row.abs_lb == pytest.approx(row.abs_diff - norm.isf(0.025) * row.abs_se)
    assert row.relative_confidence_set is not None
    assert row.relative_confidence_set.contains(row.require_lift().value)
    if not ratio:
        assert expected["cross"] * sign > 0
        assert (row.abs_se**2 - expected["independent_absolute_variance"]) * sign < 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_joint_unequal_members_keep_observed_ratio_of_totals(ratio):
    table = _joint_table(unequal=True, ratio=ratio)
    expected = _joint_oracle(table, ratio=ratio)
    row = _joint_estimate(table, ratio=ratio)
    assert row.abs_diff is not None and row.abs_se is not None
    assert row.lift is not None
    assert row.abs_diff == pytest.approx(expected["difference"])
    assert row.require_lift().value == pytest.approx(expected["lift"])
    assert row.abs_se**2 == pytest.approx(expected["absolute_variance"])
    _assert_joint_covariance(row, expected)
    if not ratio:
        records = table.to_pylist()
        cluster_mean = {
            (g, s): np.mean([r["y"] for r in records if r["g"] == g and r["store"] == s])
            for g in ("C", "T")
            for s in {r["store"] for r in records}
        }
        equal_cluster = np.mean(
            [cluster_mean["T", f"s{i}"] - cluster_mean["C", f"s{i}"] for i in range(10)]
        )
        assert abs(row.abs_diff - equal_cluster) > 0.01


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("missing", ["drop", "zero"])
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_joint_metric_missingness_retains_only_contributing_clusters(missing, ratio):
    records = _joint_table(unequal=True, ratio=ratio).to_pylist()
    records[0]["y"] = None
    if ratio:
        records[3]["den"] = None
    records.extend(
        [
            {"u": "missing-only", "g": "T", "store": "missing", "y": None, "z": None, "den": 1.0},
            {"u": "third-arm", "g": "U", "store": "third", "y": 100.0, "z": None, "den": 1.0},
        ]
    )
    # Third-arm clusters must not enter the T/C contrast's K.
    # A third arm needs >=2 clusters under the existing arm admission rule.
    extra = [
        {
            "u": f"other-{i}",
            "g": "U",
            "store": f"other-{i}",
            "y": float(100 + i),
            "z": None,
            "den": 1.0,
        }
        for i in range(10)
    ]
    source = _JointUnitSource(pa.Table.from_pylist(records + extra), missing=missing, ratio=ratio)
    results = estimate_ate(source, DESIGN, methods=[Method(name="unadjusted")]).results
    row = next(r for r in results if r.group_id == "T")
    kept = []
    for record in records:
        if record["g"] == "U":
            continue
        if missing == "drop" and (record["y"] is None or (ratio and record["den"] is None)):
            continue
        kept.append({**record, "y": record["y"] or 0.0, "den": record["den"] or 0.0})
    oracle = _joint_oracle(pa.Table.from_pylist(kept), ratio=ratio)
    assert row.n_clusters == (10 if missing == "drop" else 11)
    assert row.abs_se is not None
    assert row.abs_diff == pytest.approx(oracle["difference"])
    assert row.abs_se**2 == pytest.approx(oracle["absolute_variance"])
    _assert_joint_covariance(row, oracle)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_joint_order_partition_and_member_replication_invariance(ratio):
    records = _joint_table(unequal=True, ratio=ratio).to_pylist()
    baseline = _joint_estimate(pa.Table.from_pylist(records), ratio=ratio)
    variants = [
        records[::-1],
        records[::2] + records[1::2],
        [{**r, "u": f"copy{j}-{r['u']}"} for j in range(3) for r in records],
        [{**r, "store": f"label-{9 - int(r['store'][1:])}"} for r in records],
    ]
    for variant in variants:
        row = _joint_estimate(pa.Table.from_pylist(variant), ratio=ratio)
        assert row.n_clusters == baseline.n_clusters == 10
        assert row.abs_diff == pytest.approx(baseline.abs_diff)
        assert row.abs_se == pytest.approx(baseline.abs_se)
        assert row.require_lift().value == pytest.approx(baseline.require_lift().value)
        assert row.relative_confidence_set is not None
        assert baseline.relative_confidence_set is not None
        assert np.asarray(row.relative_confidence_set.intervals) == pytest.approx(
            np.asarray(baseline.relative_confidence_set.intervals)
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("absolute_null", [False, True])
def test_unadjusted_joint_preserves_shifted_null_and_alternative(alternative, absolute_null):
    options = {"null_abs": {"y": 0.5}} if absolute_null else {"null_lifts": {"y": 0.1}}
    row = _joint_estimate(
        _joint_table(),
        alpha=0.01,
        alternatives={"y": alternative},
        **options,
    )
    assert row.alternative == alternative
    assert row.null_abs == (0.5 if absolute_null else None)
    assert row.null_lift == (0.0 if absolute_null else 0.1)
    alpha_eff = 0.01 if alternative == "two-sided" else 0.02
    assert row.relative_confidence_set is not None
    assert row.relative_confidence_set.alpha == 0.01
    assert row.abs_diff is not None and row.abs_se is not None
    assert row.abs_ub == pytest.approx(row.abs_diff + norm.isf(alpha_eff / 2) * row.abs_se)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_joint_keeps_absolute_native_refusal_before_loading():
    source = _JointUnitSource(_joint_table())
    with pytest.raises(InvalidRequestError) as error:
        estimate_ate(
            source, DESIGN, methods=[Method(name="unadjusted")], value_scale={"y": "absolute"}
        )
    assert error.value.code == "estimation.adjust.method_name_unadjusted"
    assert source.requests == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("scale", [1.0, 1e200, 1e300, 5e306])
def test_unadjusted_joint_avoids_squaring_large_outcomes(scale):
    table = _joint_table(unequal=True)
    baseline = _joint_estimate(table)
    scaled = table.set_column(
        table.schema.get_field_index("y"), "y", pa.array(np.asarray(table["y"]) * scale)
    )
    row = _joint_estimate(scaled, raw=True)
    assert row.abs_se is not None and row.abs_diff is not None
    assert math.isfinite(row.abs_se)
    assert row.abs_se / scale == pytest.approx(baseline.abs_se)
    assert row.abs_diff / scale == pytest.approx(baseline.abs_diff)
    assert row.lift is not None and baseline.lift is not None
    assert row.require_lift().value == pytest.approx(baseline.require_lift().value)
    if scale == 1.0:
        assert row.relative_confidence_set is not None
    else:
        assert row.relative_confidence_set is None
        assert row.relative_unavailable_reason == "joint_covariance_unrepresentable"
        assert row.require_lift().lb is None and row.require_lift().ub is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_joint_centering_preserves_neighboring_large_values():
    origin = float(2**48)
    step = math.ulp(origin)
    records = _joint_table(unequal=True).to_pylist()
    records = [{**r, "y": origin + r["y"] * step} for r in records]
    table = pa.Table.from_pylist(records)
    oracle = _joint_oracle(table)
    row = _joint_estimate(table, raw=True)
    assert row.abs_se is not None
    assert row.abs_se**2 == pytest.approx(oracle["absolute_variance"], rel=1e-12, abs=0)
    _assert_joint_covariance(row, oracle)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_joint_exact_absolute_cancellation_preserves_numeric_nulls():
    records = _joint_table().to_pylist()
    for record in records:
        shock = int(record["store"][1:]) % 5 - 2
        record["y"] = float(10 + shock + 2 * (record["g"] == "T"))
    row = _joint_estimate(pa.Table.from_pylist(records))
    assert row.abs_diff == 2.0
    assert row.abs_se is None and row.abs_lb is None and row.abs_ub is None
    assert row.abs_reference_kind is None and row.abs_reference_df is None
    assert row.relative_confidence_set is not None
    assert row.relative_confidence_set.reference.var_a == 0.0
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < lift.value < lift.ub


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("dominant_mass", [8, 9, 10])
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_response_handles_half_mass_without_a_leverage_cutoff(dominant_mass, ratio):
    records = []
    for row in _joint_table().to_pylist():
        mass = dominant_mass if row["store"] == "s0" else 1
        for copy in range(1 if ratio else mass):
            records.append(
                {
                    **row,
                    "u": f"{row['u']}-{copy}",
                    "den": float(mass) if ratio else 1.0,
                    "y": row["y"] * (mass if ratio else 1),
                }
            )
    table = pa.Table.from_pylist(records)
    expected = _joint_oracle(table, ratio=ratio)
    result = _joint_estimate(table, ratio=ratio)
    assert result.abs_se is not None
    assert result.n_clusters == 10
    assert result.abs_se**2 == pytest.approx(expected["absolute_variance"])
    _assert_joint_covariance(result, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_singular_response_retains_an_explicit_approximation():
    records = [r for r in _joint_table().to_pylist() if r["g"] == "C" or r["store"] in ("s0", "s1")]
    extra = next(r for r in records if r["g"] == "T" and r["store"] == "s1")
    records.append({**extra, "u": "extra-member"})
    table = pa.Table.from_pylist(records)
    row = _joint_estimate(table)
    assert row.n_clusters == 10 and row.abs_se is not None
    means = {
        arm: sum(r["y"] for r in records if r["g"] == arm) / sum(r["g"] == arm for r in records)
        for arm in ("T", "C")
    }
    residuals = {
        arm: {
            g: sum(r["y"] - means[arm] for r in records if r["g"] == arm and r["store"] == g)
            / sum(r["g"] == arm for r in records)
            for g in {r["store"] for r in records}
        }
        for arm in ("T", "C")
    }
    expected = sum((residuals["T"][g] - residuals["C"][g]) ** 2 for g in residuals["T"]) * 10 / 9
    assert row.abs_se**2 == pytest.approx(expected)
    assert row.relative_confidence_set is not None
    expected_covariance = (
        sum((residuals["T"][g] - residuals["C"][g]) * residuals["C"][g] for g in residuals["T"])
        * 10
        / 9
    )
    assert row.relative_confidence_set.reference.cov_ac == pytest.approx(expected_covariance)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ratio", [False, True])
def test_unadjusted_joint_pure_unequal_members_match_separate_arm_reference(ratio):
    from increment.estimation.engine import estimate_lift

    records = _joint_table(unequal=True, ratio=ratio).to_pylist()
    table = pa.Table.from_pylist([{**r, "store": f"{r['g']}-{r['store']}"} for r in records])
    spec = (
        MetricSpec(name="y", type="ratio", numerator="y", denominator="den")
        if ratio
        else MetricSpec(name="y")
    )
    source = from_unit_summary(
        table,
        unit="u",
        group="g",
        control="C",
        metrics=[spec],
        cluster="store",
        design=DESIGN,
    )
    (joint,) = estimate_ate(source, DESIGN, methods=[Method(name="unadjusted")]).results
    (separate,) = estimate_lift(
        source.context.metrics,
        source.moments(_metric(source)),
        "C",
        cluster="store",
    ).results
    assert joint.n_clusters == separate.n_clusters == 20
    assert joint.require_lift().value == pytest.approx(separate.require_lift().value)
    assert joint.relative_confidence_set is not None
    assert separate.relative_confidence_set is not None
    assert joint.abs_diff == pytest.approx(separate.abs_diff)
    assert joint.abs_se == pytest.approx(separate.abs_se)
    assert joint.reference_df == pytest.approx(separate.reference_df)
    assert joint.abs_reference_df == pytest.approx(separate.abs_reference_df)
    assert joint.require_lift().lb == pytest.approx(separate.require_lift().lb)
    assert joint.abs_lb == pytest.approx(separate.abs_lb)


def test_unadjusted_joint_admits_small_but_structurally_valid_cluster_count():
    records = [r for r in _joint_table().to_pylist() if r["store"] != "s9"]
    source = _JointUnitSource(pa.Table.from_pylist(records))
    with pytest.warns(IncrementRuntimeWarning) as rec:
        (row,) = estimate_ate(source, DESIGN, methods=[Method(name="unadjusted")]).results
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    assert row.n_clusters == 9
    assert row.reference_kind == "normal"
    assert row.reference_df is None
    assert row.relative_confidence_set is not None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_source_consumes_only_canonical_cluster_metadata():
    source = _JointUnitSource(_joint_table())
    frame = source.unit_frame(_metric(source))
    assert "cluster_id" in frame.column_names and "store" not in frame.column_names
    source.requests.clear()
    (row,) = estimate_ate(source, DESIGN, methods=[Method(name="unadjusted")]).results
    assert all(extra == () for extra in source.requests)
    assert row.n_clusters == 10 and row.abs_se is not None
    assert row.relative_confidence_set is not None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("spanning", [False, True])
def test_unadjusted_definitions_source_uses_cluster_metadata_without_covariates(tmp_path, spanning):
    from tests.analysis_factory import _native_source, make_analysis_like
    from tests.test_analysis_cluster import _analysis, _event_rows

    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("baseline",))
    )
    with _analysis(tmp_path, rows=_event_rows(spanning_store=spanning)) as analysis:
        source = _native_source(make_analysis_like(analysis, design=design))
        frame = source.unit_frame(source.context.metrics[0])
        assert "cluster_id" in frame.column_names and "store_id" not in frame.column_names
        (row,) = estimate_ate(source, design, methods=[Method(name="unadjusted")]).results
    assert row.abs_diff == pytest.approx(0.5)
    assert row.n_clusters == (39 if spanning else 40)
    assert row.relative_confidence_set is not None
    assert row.abs_se == math.sqrt(row.relative_confidence_set.reference.var_a)
    if not spanning:
        assert row.reference_df == 19
        assert row.abs_reference_df == pytest.approx(38)
        assert row.abs_se == pytest.approx(math.sqrt(2 * np.var(0.2 * np.arange(20), ddof=1) / 20))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_unadjusted_large_offset_point_and_se_are_exact_joint_projections():
    from fractions import Fraction

    from increment.estimation.results import LiftEstimate

    origin = float(2**48)
    step = math.ulp(origin)
    control = [origin, origin + step, origin]
    treatment = [origin + step, origin + 2 * step, origin + 2 * step]
    records = [
        {"u": f"{arm}{i}", "g": arm, "store": f"{arm}{i}", "y": y}
        for arm, values in (("C", control), ("T", treatment))
        for i, y in enumerate(values)
    ]
    row = _joint_estimate(pa.Table.from_pylist(records), raw=True)
    exact = (sum(map(Fraction, treatment)) - sum(map(Fraction, control))) / 3
    assert row.abs_diff == float(exact)
    assert row.abs_diff != float(sum(map(Fraction, treatment)) / 3) - float(
        sum(map(Fraction, control)) / 3
    )
    assert row.relative_confidence_set is not None
    ref = row.relative_confidence_set.reference
    assert row.abs_diff == ref.a and row.abs_se == math.sqrt(ref.var_a)
    assert row.reference_df == 2
    assert row.abs_reference_df is not None
    assert row.abs_reference_df > row.reference_df
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
