"""Independent numerical witnesses for cluster-randomized inference.

These identify particular covariance/metadata defects; they do not replace
small-cluster scientific gates or fitted cross-fit response geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from increment.estimation._adjust.overlap import _cluster_reduction


def test_oracle_aipw_retains_between_arm_superpopulation_variation():
    # Forty clusters: P(X,D)=(.4,.1,.1,.4), with two identical members each.
    x = np.repeat([0.0, 0.0, 1.0, 1.0], [16, 4, 4, 16])
    psi = np.repeat(x - 0.5, 2)
    variance, scale = _cluster_reduction(psi, np.repeat(np.arange(40), 2), 40)
    assert math.sqrt(variance) * scale / 80 == pytest.approx(math.sqrt(0.25 / 39))


def test_oracle_dml_retains_between_arm_superpopulation_variation():
    # Independent enumeration: E[psi^2]=.0208 and J=.16, hence K*V=.8125.
    scores = np.repeat([-0.02, -0.32, 0.32, 0.02], [16, 4, 4, 16])
    variance, scale = _cluster_reduction(scores, np.arange(40), 40)
    assert math.sqrt(variance) * scale / (40 * 0.16) == pytest.approx(math.sqrt(0.8125 / 39))


def test_mixed_cluster_cross_arm_covariance_is_retained():
    d = np.tile([0.0, 1.0], 10)
    inv = np.repeat(np.arange(10), 2)
    shared = np.repeat(np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 2), 2)
    # Identical paired shocks cancel in the contrast, despite nonzero arm meat.
    variance, _ = _cluster_reduction(shared * (2 * d - 1), inv, 10)
    assert variance == 0
    deltas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 2)
    scores = np.zeros(20)
    scores[1::2] = 2 * deltas
    variance, scale = _cluster_reduction(scores, inv, 10)
    assert math.sqrt(variance) * scale / 20 == pytest.approx(math.sqrt(2 / 9))


def test_fitted_logistic_hajek_cancels_saturated_null_outcome():
    from increment.estimation._adjust.learners import (
        LogisticPropensity,
        _coupled_hajek_mean_influences,
    )

    x = np.repeat([0.0, 0.0, 1.0, 1.0], [8, 2, 2, 8])[:, None]
    d = np.repeat([0.0, 1.0, 0.0, 1.0], [8, 2, 2, 8])
    y = 5 + x[:, 0]
    learner = LogisticPropensity(l2=0)
    learner.fit(x, d)
    e = learner.predict(x)
    w1, w0 = d / e, (1 - d) / (1 - e)
    mu1, mu0 = np.dot(w1, y) / w1.sum(), np.dot(w0, y) / w0.sum()
    psi0, psi1 = _coupled_hajek_mean_influences(
        [learner],
        [x],
        d.astype(int),
        e[:, None],
        np.column_stack([1 - e, e]),
        np.column_stack([20 * w0 * (y - mu0) / w0.sum(), 20 * w1 * (y - mu1) / w1.sum()]),
    ).T
    # Calibration makes both Hajek means the SAME empirical covariate mean.
    assert mu1 - mu0 == pytest.approx(0, abs=1e-6)
    assert psi1 - psi0 == pytest.approx(np.zeros(20), abs=1e-5)
    assert psi1 == pytest.approx(x[:, 0] - 0.5, abs=1e-5)


def test_absolute_welch_reference_and_json_fcr_are_independent():
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import LiftEstimate, open_bound_from_two_sided_at_target

    absolute_df = 1.0314581662190911
    row = infer_lift(
        "y",
        "T",
        "unadjusted",
        math.log(8),
        math.sqrt(8) / 80,
        math.sqrt(0.125) / 10,
        dof=3.5,
        abs_dof=absolute_df,
        abs_diff=70,
        abs_se=math.sqrt(8.125),
        n_clusters=10,
        method_role="decision",
    )
    assert row.abs_lb == pytest.approx(36.290552, abs=1e-5)
    assert row.abs_ub == pytest.approx(103.709448, abs=1e-5)
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    assert restored.reference_df == 3.5
    assert restored.abs_reference_kind == "t"
    assert restored.abs_reference_df == absolute_df
    directional = restored.model_copy(update={"alternative": "greater"})
    converted = open_bound_from_two_sided_at_target(directional)
    assert converted.abs_lb == row.abs_lb and converted.abs_ub == row.abs_ub
    assert converted.abs_reference_df == absolute_df


def test_declared_truth_is_not_the_observed_contrast():
    from tests.estimation._i13_manifest import ClusterDGP, target_weighting_dgp

    sample = ClusterDGP(20, 20, icc=0.2, effect=0.5, seed=123).draw()
    y = np.asarray(sample.table["y"])
    d = np.asarray(sample.table["g"]) == "T"
    assert sample.tau_u == 0.5
    assert sample.tau_u_relative == 0.1
    assert abs(y[d].mean() - y[~d].mean() - sample.tau_u) > 1e-6
    heterogeneous, plr = target_weighting_dgp(0)
    assert (heterogeneous.tau_c, heterogeneous.tau_u, plr) == (0.5, 0.8, 6 / 7)
    assert len({heterogeneous.tau_c, heterogeneous.tau_u, plr}) == 3


@pytest.mark.slow
def test_partially_mixed_gaussian_witness_disproves_cr1_t9():
    from scipy.integrate import quad
    from scipy.stats import beta, t

    # T^2=36 Z1^2/(Z1^2+Z2^2+5 W), W~chi2(6). Write Z1^2+Z2^2=S,
    # B=Z1^2/S~Beta(.5,.5), and S/(S+W)~Beta(1,3), independent of B.
    critical_sq = float(t.isf(0.025, 9)) ** 2
    boundary = critical_sq / 36
    probability = quad(
        lambda b: beta.sf(5 * boundary / (b + 4 * boundary), 1, 3) * beta.pdf(b, 0.5, 0.5),
        boundary,
        1,
        epsabs=1e-12,
        epsrel=1e-12,
    )[0]
    assert probability == pytest.approx(0.0609717829, abs=1e-9)
    assert probability > 0.055
    # This is an unresolved finite-sample prerequisite, not a calibration pass.


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_randomized_sharp_null_enumeration_mean():
    from itertools import combinations

    import pyarrow as pa

    from tests.estimation._i13_adapters import estimate_of
    from tests.estimation._i13_manifest import ACCEPTANCE_MANIFEST, DGPSample

    cell = next(
        c
        for c in ACCEPTANCE_MANIFEST
        if c.estimator == "mean"
        and c.design.dgp.k_t == c.design.dgp.k_c == 5
        and c.scale == "absolute_sidecar"
    )
    outcomes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 2) + 5
    assignments = list(combinations(range(10), 5))
    contrasts = np.array(
        [outcomes[list(t)].mean() - np.delete(outcomes, t).mean() for t in assignments]
    )
    errors = 0
    exact_errors = 0
    for i, treated in enumerate(assignments):
        table = pa.table(
            {
                "u": [f"u{j}" for j in range(10)],
                "g": ["T" if j in treated else "C" for j in range(10)],
                "cluster": [f"c{j}" for j in range(10)],
                "y": outcomes,
            }
        )
        sample = DGPSample(table, 0.0, 0.0, 10, target="finite_population")
        interval = estimate_of(cell, sample, i)
        errors += int(not interval.contains(0))
        p = np.mean(np.abs(contrasts) >= abs(contrasts[i]) - 1e-12)
        exact_errors += int(p <= 0.05)
    assert exact_errors / len(assignments) <= 0.05
    assert errors / len(assignments) <= 0.055


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_engine_computes_absolute_df_from_absolute_components():
    from increment.estimation.engine import estimate_lift
    from increment.semantics.models import MeanMetric

    # Singleton clusters: T variance=16, C variance=1. Absolute components
    # 16/2=8 and 1/8=.125; log components both equal .00125.
    def arm(group, n, mean, centered_ss):
        return {
            "experiment_id": "e",
            "metric": "y",
            "group_id": group,
            "n": n,
            "ref_y": mean,
            "cy1": 0.0,
            "cy2": centered_ss,
            "ref_den": 1.0,
            "cden1": 0.0,
            "cden2": 0.0,
            "cyden": 0.0,
            "ref_x": 1.0,
            "cx1": 0.0,
            "cx2": 0.0,
            "cxy": 0.0,
            "x_role": "cluster_size",
        }

    metric = MeanMetric(name="y", entity="u", fact="y", aggregation="sum")
    (row,) = estimate_lift(
        [metric],
        [arm("T", 2, 80.0, 16.0), arm("C", 8, 10.0, 7.0)],
        control_group="C",
        cluster="cluster",
    ).results
    assert row.reference_df == 1.0
    assert row.abs_reference_df == pytest.approx(1.0314581662190911)
    assert row.abs_lb == pytest.approx(36.290552, abs=1e-5)
    assert row.abs_ub == pytest.approx(103.709448, abs=1e-5)


def test_fitted_dml_fold_geometry_has_multiple_eigenvalues():
    # Twelve clusters, balanced two-fold cross-fit intercepts. Build the
    # response operator analytically, not from captured production scores.
    d = np.array([0.0] * 6 + [1.0] * 6)
    v = d - 0.5
    fold = np.tile([0, 1], 6)
    h = (fold[:, None] != fold[None, :]).astype(float) / 6
    residual = np.eye(12) - h
    slope = v @ residual / (v @ v)
    error = residual - np.outer(v, slope)
    raw = np.diag(v) @ error
    # The rejected draft additionally removed each arm's score mean.
    arm_center = np.eye(12) - ((d[:, None] == d[None, :]).astype(float) / 6)
    meat = arm_center @ raw
    q = (6 / 5) * meat.T @ meat / (v @ v) ** 2
    eig = np.linalg.eigvalsh(q)
    assert eig[-10:-1] == pytest.approx(np.full(9, 1 / 30))
    assert eig[-1] == pytest.approx(4 / 30)
    assert np.trace(q) == pytest.approx(13 / 30)
    assert slope @ slope == pytest.approx(1 / 3)
    assert np.trace(q) ** 2 / np.trace(q @ q) == pytest.approx(6.76)
    # This exact working-model calculation remains a required production repair.


def test_adjusted_sidecar_keeps_its_distinct_reference():
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.inference import infer_ate
    from increment.estimation.results import LiftEstimate

    row = infer_ate(
        "y",
        "T",
        "aipw",
        0.2,
        ScoreStats(metric="y", contrast="T", n=1, sum_psi=0.0, sum_psi2=0.01),
        dof=8,
        abs_dof=2,
        abs_diff=0.5,
        abs_se=0.2,
        method_role="decision",
    )
    assert row.abs_lb == pytest.approx(-0.360530545939228)
    assert row.abs_ub == pytest.approx(1.360530545939228)
    restored = LiftEstimate.model_validate_json(row.model_dump_json())
    assert (restored.reference_df, restored.abs_reference_df) == (8.0, 2.0)
    assert restored.abs_reference_kind == "t"


def test_absolute_t_tail_does_not_inherit_normal_primary_reference():
    from increment.errors import InvalidRequestError
    from increment.estimation.inference import infer_lift

    row = infer_lift(
        "y",
        "T",
        "unadjusted",
        0.1,
        0.02,
        0.02,
        abs_diff=1.0,
        abs_se=0.2,
        abs_dof=2.0,
        null_abs=0.5,
        method_role="decision",
    )
    assert row.reference_kind == "normal" and row.abs_reference_kind == "t"
    with pytest.raises(InvalidRequestError) as error:
        row.p_value()
    assert error.value.code == "estimation.results.lift.p_value_cluster_robust_null_abs"
    assert error.value.context["reference_df"] == 2.0


@pytest.mark.parametrize("backend", ("pandas", "polars", "pyarrow"))
def test_absolute_reference_survives_frame_conversion_with_numeric_null(backend):
    import narwhals as nw

    from increment.breakout.estimates import to_frame
    from increment.estimation.inference import infer_lift

    known = infer_lift(
        "y",
        "T",
        "unadjusted",
        0.1,
        0.02,
        0.02,
        abs_diff=1.0,
        abs_se=0.2,
        abs_dof=2.0,
        method_role="decision",
    )
    missing = infer_lift("y", "T", "unadjusted", 0.1, 0.02, 0.02, method_role="decision")
    frame = nw.from_native(to_frame([known, missing], backend=backend))
    assert frame.schema["abs_reference_df"] == nw.Float64()
    values = frame["abs_reference_df"]
    assert values[0] == pytest.approx(2.0)
    assert values.is_null().to_list() == [False, True]
    references = frame["abs_reference_kind"]
    assert references[0] == "t"
    assert references.is_null().to_list() == [False, True]


@pytest.mark.slow
def test_manifest_seed_is_independent_of_python_hash_salt():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "from tests.estimation._i13_manifest import _base; print(_base(5, 20, shock='skew').seed)"
    )
    seeds = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=root,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": salt, "PYTHONPATH": str(root)},
        ).strip()
        for salt in ("1", "98765")
    ]
    assert seeds[0] == seeds[1]


def test_chi_square_shock_uses_variance_four():
    from tests.estimation._i13_manifest import ClusterDGP

    class KnownDraws:
        def normal(self, loc: float, scale: float, *, size: int) -> np.ndarray:
            pytest.fail("the skew-shock witness must use chi-square draws")

        def chisquare(self, df: float, *, size: int) -> np.ndarray:
            assert df == 2.0 and size == 2
            return np.array([0.0, 4.0])

    # This two-point draw has mean 2 and variance 4, matching the law's moments.
    shocks = ClusterDGP(5, 5, shock="skew")._shock(KnownDraws(), 2, 3.0)
    assert shocks == pytest.approx([-3.0, 3.0])
    assert np.var(shocks) == 9.0


def _partially_mixed_unadjusted_table(shocks):
    import pyarrow as pa

    # Four pure treatment, four pure control, two mixed; ten members per arm.
    assignments = [("T", "T")] * 4 + [("C", "C")] * 4 + [("T", "C")] * 2
    return pa.Table.from_pylist(
        [
            {
                "u": f"{g}-{member}",
                "g": arm,
                "store": f"s{g}",
                "y": 5.0 + float(shocks[g]),
                "z": None,
                "den": 1.0,
            }
            for g, arms in enumerate(assignments)
            for member, arm in enumerate(arms)
        ]
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_public_unadjusted_partially_mixed_witness_counts_ten_not_twelve():
    from tests.estimation.test_adjust_cluster import (
        _assert_joint_covariance,
        _joint_estimate,
        _joint_oracle,
    )

    table = _partially_mixed_unadjusted_table([-2, -1, 0, 1, -1, 0, 1, 2, -1, 1])
    expected = _joint_oracle(table)
    row = _joint_estimate(table)
    assert row.abs_se is not None
    assert table.num_rows == 20
    assert row.n_clusters == 10  # Each arm touches six clusters, but their union has ten.
    assert row.abs_diff == pytest.approx(-0.8)
    assert row.abs_diff == pytest.approx(expected["difference"])
    assert row.require_lift().value == pytest.approx(expected["lift"])
    assert row.abs_se**2 == pytest.approx(expected["absolute_variance"])
    _assert_joint_covariance(row, expected)
    assert row.reference_df is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.slow
def test_public_unadjusted_mixed_response_correction_remains_required():
    from tests.estimation.test_adjust_cluster import _joint_estimate

    # For Y=5+B epsilon, epsilon~N(0,I_10), the absolute estimator and residual
    # map are linear. Polarization around a varying background avoids the
    # existing constant-arm guard while extracting the exact quadratic trace.
    background = [-2, -1, 0, 1, -1, 0, 1, 2, -1, 1]
    baseline = _joint_estimate(_partially_mixed_unadjusted_table(background))
    assert baseline.abs_se is not None
    true_variance = expected_variance = 0.0
    for j in range(10):
        plus = _joint_estimate(
            _partially_mixed_unadjusted_table(
                [value + int(g == j) for g, value in enumerate(background)]
            )
        )
        minus = _joint_estimate(
            _partially_mixed_unadjusted_table(
                [value - int(g == j) for g, value in enumerate(background)]
            )
        )
        assert plus.n_clusters == minus.n_clusters == baseline.n_clusters == 10
        assert plus.abs_diff is not None and minus.abs_diff is not None
        assert plus.abs_se is not None and minus.abs_se is not None
        true_variance += ((plus.abs_diff - minus.abs_diff) / 2) ** 2
        expected_variance += (plus.abs_se**2 + minus.abs_se**2) / 2 - baseline.abs_se**2
    assert true_variance == pytest.approx(0.32)
    # Joint CR1 yields .2844444444, not .32; response correction must recover .32.
    # The old separate-arm route yielded .34656 and counted twelve clusters.
    # Do not xfail, loosen, or relabel this as a small-sample calibration pass.
    assert expected_variance == pytest.approx(true_variance)


def _heterogeneous_response_table(*, unequal, ratio):
    import pyarrow as pa

    records = []
    for g in range(10):
        groups = ("T",) if g < 3 else ("C",) if g < 6 else ("T", "C")
        for arm in groups:
            count = 1 + (g + int(arm == "T")) % 3 if unequal else 2
            for member in range(count):
                denominator = float(1 + (g + member) % 3) if ratio else 1.0
                records.append(
                    {
                        "u": f"{g}-{arm}-{member}",
                        "g": arm,
                        "store": f"s{g}",
                        "y": denominator
                        * (30 + 2 * (arm == "T") + (g % 5 - 2) * (1 if arm == "T" else -1)),
                        "z": None,
                        "den": denominator,
                    }
                )
    return pa.Table.from_pylist(records)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("unequal", [False, True])
@pytest.mark.parametrize("ratio", [False, True])
@pytest.mark.slow
def test_unadjusted_response_trace_with_heterogeneous_signed_cluster_noise(sign, unequal, ratio):
    import pyarrow as pa

    from tests.estimation.test_adjust_cluster import _joint_estimate, _joint_oracle

    table = _heterogeneous_response_table(unequal=unequal, ratio=ratio)
    records = table.to_pylist()
    baseline = _joint_estimate(table, ratio=ratio)
    oracle = _joint_oracle(table, ratio=ratio)
    assert baseline.abs_se is not None
    assert baseline.abs_se**2 == pytest.approx(oracle["absolute_variance"])
    expected_variance = true_variance = 0.0
    for g in range(10):
        # Two independent inputs per cluster allow arbitrary signed common noise
        # and an additional random treatment slope, with heterogeneous variances.
        for slope in (False, True):
            changes = [
                r["den"]
                * (
                    (0.25 + 0.25 * (g % 2)) * (r["g"] == "T")
                    if slope
                    else (1 + g / 10 if r["g"] == "T" else sign * (0.5 + g / 20))
                )
                * (r["store"] == f"s{g}")
                for r in records
            ]
            response = {
                arm: sum(delta for r, delta in zip(records, changes, strict=True) if r["g"] == arm)
                / sum(r["den"] for r in records if r["g"] == arm)
                for arm in ("T", "C")
            }
            true_variance += (response["T"] - response["C"]) ** 2
            variances = []
            for direction in (-1, 1):
                perturbed = pa.Table.from_pylist(
                    [
                        {**r, "y": r["y"] + direction * delta}
                        for r, delta in zip(records, changes, strict=True)
                    ]
                )
                row = _joint_estimate(perturbed, ratio=ratio)
                assert row.abs_se is not None and row.abs_diff is not None
                assert baseline.abs_diff is not None
                assert row.n_clusters == 10
                assert row.abs_diff - baseline.abs_diff == pytest.approx(
                    direction * (response["T"] - response["C"]), abs=1e-12
                )
                variances.append(row.abs_se**2)
            expected_variance += sum(variances) / 2 - baseline.abs_se**2
    assert true_variance > 0.0
    assert expected_variance == pytest.approx(true_variance, rel=1e-10, abs=1e-12)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_negative_corrected_variance_preserves_points_without_relative_evidence():
    import pyarrow as pa

    from tests.estimation.test_adjust_cluster import _joint_estimate, _joint_oracle

    records = _partially_mixed_unadjusted_table([0] * 8 + [1, -1]).to_pylist()
    records = [{**r, "y": r["y"] + (95 if r["g"] == "T" else 5)} for r in records]
    table = pa.Table.from_pylist(records)
    oracle = _joint_oracle(table)
    assert oracle["absolute_variance"] < 0.0 < oracle["log_variance"]
    row = _joint_estimate(table)
    assert row.abs_diff == pytest.approx(90)
    assert row.abs_se is None and row.abs_lb is None and row.abs_ub is None
    assert row.abs_reference_kind is None and row.abs_reference_df is None
    assert row.lift is not None and row.require_lift().value == pytest.approx(oracle["lift"])
    assert row.require_lift().lb is None and row.require_lift().ub is None
    assert row.relative_confidence_set is None
    assert row.relative_unavailable_reason == "joint_covariance_indefinite"


def test_fieller_geometry_is_invariant_under_common_measurement_scaling():
    from increment.estimation.results import JointContrastReference, relative_confidence_set

    base = JointContrastReference(a=2.0, c=3.0, var_a=0.04, var_c=0.09, cov_ac=0.01)
    baseline = relative_confidence_set(base)
    for scale in (1e-10, 1.0, 1e10):
        scaled = JointContrastReference(
            a=base.a * scale,
            c=base.c * scale,
            var_a=base.var_a * scale**2,
            var_c=base.var_c * scale**2,
            cov_ac=base.cov_ac * scale**2,
        )
        observed = relative_confidence_set(scaled)
        assert observed.geometry == baseline.geometry
        assert np.asarray(observed.intervals) == pytest.approx(np.asarray(baseline.intervals))


def test_adjusted_joint_reference_uses_n_normalized_cluster_meat():
    from increment.estimation._adjust.common import (
        ClusterSupport,
        _joint_reference_from_influences,
    )

    numerator = np.array([1.0, 1.0, -1.0, -1.0])
    denominator = np.array([0.5, 0.5, -0.5, -0.5])
    whole = ClusterSupport(slice(None), np.array([0, 0, 1, 1]), 2)
    reference, reason = _joint_reference_from_influences(
        numerator,
        denominator,
        numerator_point=1.0,
        denominator_point=2.0,
        supports=(whole, whole),
    )
    assert reason is None and reference is not None
    assert reference.var_a == pytest.approx(1.0)
    assert reference.var_c == pytest.approx(0.25)
    assert reference.cov_ac == pytest.approx(0.5)
