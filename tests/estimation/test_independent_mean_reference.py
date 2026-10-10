"""Independent arithmetic and normal/chi-square integration for Welch means."""

import math

import pytest


def test_component_arithmetic_and_portable_welch_reference():
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.results import LiftEstimate

    c = IndependentMeanComponent.from_values("C", [1.0, 2.0, 3.0], coefficient=-1)
    t = IndependentMeanComponent.from_values("T", [2.0, 4.0, 6.0, 8.0], coefficient=1)
    ref = IndependentMeanReference(components=[c, t])
    vc, vt = 1 / 3, 5 / 3
    assert c.centered_sum_squares == 2
    assert t.centered_sum_squares == 20
    assert ref.point == 3
    assert ref.variance == pytest.approx(vc + vt)
    assert ref.df == pytest.approx((vc + vt) ** 2 / (vc**2 / 2 + vt**2 / 3))
    row = infer_ate(
        "m", "T", "independent_mean", ref.point, ref, method_role="decision", value_scale="absolute"
    )
    assert row.reference_kind == "t" and row.reference_df == ref.df
    assert row.n_clusters is None and row.dof is None
    assert row.independent_mean_reference == ref
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    p_value = row.p_value()
    assert p_value is not None
    assert p_value < 1


def test_single_normal_mean_and_directional_identity():
    from scipy.stats import t

    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate

    ref = IndependentMeanReference(
        components=(
            IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0, 5.0], coefficient=1),
        )
    )
    assert ref.df == 4
    row = infer_ate(
        "m",
        "T",
        "independent_mean",
        ref.point,
        ref,
        method_role="decision",
        value_scale="absolute",
        alpha=0.05,
    )
    greater = infer_ate(
        "m",
        "T",
        "independent_mean",
        ref.point,
        ref,
        method_role="decision",
        value_scale="absolute",
        alpha=0.025,
        alternative="greater",
    )
    assert row.lift is not None and greater.lift is not None
    assert (row.lift.lb, row.lift.ub) == (greater.lift.lb, greater.lift.ub)
    assert row.lift.lb == pytest.approx(3 - t.isf(0.025, 4) * math.sqrt(0.5))


def test_centered_components_preserve_neighbors_and_input_order():
    from increment.estimation.armstats import IndependentMeanComponent

    values = [1e16, 1e16 + 2, 1e16 + 4]
    forward = IndependentMeanComponent.from_values("C", values, coefficient=1)
    backward = IndependentMeanComponent.from_values("C", values[::-1], coefficient=1)
    assert forward == backward
    assert forward.centered_sum_squares == 8
    assert forward.variance == pytest.approx(4 / 3)


@pytest.mark.parametrize("nc,nt", [(50, 200), (200, 50)])
def test_welch_normal_chisquare_quadrature(nc, nt):
    import numpy as np
    from scipy.special import gamma, roots_genlaguerre
    from scipy.stats import norm, t

    from tests._i15_design import load_manifest

    case = next(c for c in load_manifest()["cases"] if c["case_id"] == f"ate-unequal-{nc}-{nt}")
    d = case["dgp"]
    vc, vt = d["sigma_c"] ** 2 / nc, d["sigma_t"] ** 2 / nt
    values = []
    for order in (32, 64):
        xc, wc = roots_genlaguerre(order, (nc - 1) / 2 - 1)
        xt, wt = roots_genlaguerre(order, (nt - 1) / 2 - 1)
        wc, wt = wc / gamma((nc - 1) / 2), wt / gamma((nt - 1) / 2)
        a, b = vc * 2 * xc[:, None] / (nc - 1), vt * 2 * xt[None, :] / (nt - 1)
        variance = a + b
        df = variance**2 / (a * a / (nc - 1) + b * b / (nt - 1))
        conditional_error = 2 * norm.sf(t.isf(0.025, df) * np.sqrt(variance / (vc + vt)))
        values.append(float(np.sum(wc[:, None] * wt[None, :] * conditional_error)))
    assert values[0] == pytest.approx(values[1], abs=1e-8)
    assert 0 < values[1] <= 0.055


def test_real_frame_moments_produce_components_and_adjusted_scores_stay_separate():
    import pyarrow as pa

    from increment.errors import CodedError
    from increment.estimation.armstats import (
        ArmStats,
        IndependentMeanComponent,
        IndependentMeanReference,
        ScoreStats,
    )
    from increment.estimation.inference import infer_ate
    from increment.frame import from_unit_summary

    source = from_unit_summary(
        pa.table(
            {"u": range(7), "g": ["C"] * 3 + ["T"] * 4, "m": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 8.0]}
        ),
        unit="u",
        group="g",
        control="C",
        metrics=[{"name": "m"}],
    )
    arms = tuple(
        ArmStats.model_validate({**row, "study_id": row["experiment_id"]})
        for row in source.raw_moments
    )
    ref = IndependentMeanReference(
        components=tuple(
            IndependentMeanComponent.from_arm_stats(a, coefficient=-1 if a.group_id == "C" else 1)
            for a in arms
        )
    )
    assert ref.variance == pytest.approx(2)
    from increment.estimation import infer_independent_mean

    by_group = {a.group_id: a for a in arms}
    direct = infer_independent_mean(by_group["C"], by_group["T"])
    assert direct.independent_mean_reference == ref
    with pytest.raises(CodedError) as error:
        infer_ate("m", "T", "iptw", ref.point, ref, method_role="decision", value_scale="absolute")
    assert error.value.code == "estimation.armstats.independent_mean_contract"
    scores = ScoreStats(metric="m", contrast="T", n=7, sum_psi=0, sum_psi2=10)
    adjusted = infer_ate(
        "m", "T", "iptw", 1, scores, method_role="decision", value_scale="absolute"
    )
    assert adjusted.reference_kind == "normal"
    assert adjusted.independent_mean_reference is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("lift", "value"), 9),
        (("lift", "log_mean"), 9),
        (("lift", "log_se"), 9),
        (("lift", "lb"), -99),
        (("lift", "ub"), 99),
        (("reference_df",), 99),
        (("alternative",), "greater"),
        (("independent_mean_reference", "alpha"), 0.1),
        (("independent_mean_reference", "alternative"), "less"),
        (("independent_mean_reference", "components", 0, "mean"), 9),
        (("prior_shrunk",), True),
        (("abs_diff",), 3),
        (("null_lift",), 0.25),
        (("null_abs",), 1),
    ],
)
def test_independent_mean_wire_rejects_observable_mutations(path, value):
    from increment.errors import CodedError
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.results import LiftEstimate

    ref = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0], coefficient=1),)
    )
    row = infer_ate(
        "m", "T", "independent_mean", ref.point, ref, method_role="decision", value_scale="absolute"
    )
    payload = row.model_dump(mode="json")
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(CodedError):
        LiftEstimate.model_validate(payload)


def test_independent_mean_wire_requires_reference_and_matching_method():
    from increment.errors import CodedError
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.results import LiftEstimate

    ref = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0], coefficient=1),)
    )
    row = infer_ate(
        "m", "T", "independent_mean", ref.point, ref, method_role="decision", value_scale="absolute"
    )
    missing = row.model_dump(mode="json")
    del missing["independent_mean_reference"]
    missing["lift"]["log_mean"] = 9
    with pytest.raises(CodedError):
        LiftEstimate.model_validate(missing)

    incompatible = row.model_dump(mode="json")
    incompatible["method"] = "unadjusted"
    with pytest.raises(CodedError):
        LiftEstimate.model_validate(incompatible)


def test_independent_mean_v1_wire_is_rejected():
    from increment.errors import CodedError
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.results import LiftEstimate

    ref = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0], coefficient=1),)
    )
    row = infer_ate(
        "m", "T", "independent_mean", ref.point, ref, method_role="decision", value_scale="absolute"
    )
    payload = row.model_dump(mode="json")
    reference = payload["independent_mean_reference"]
    reference["kind"] = "independent-means-welch-v1"
    for field in ("alpha", "alternative", "interval"):
        del reference[field]
    with pytest.raises(CodedError):
        LiftEstimate.model_validate(payload)


def test_independent_mean_wire_checks_interval_alpha_and_level_together():
    from increment.errors import CodedError
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.results import LiftEstimate

    ref = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0], coefficient=1),)
    )
    row = infer_ate(
        "m",
        "T",
        "independent_mean",
        ref.point,
        ref,
        method_role="decision",
        value_scale="absolute",
        alpha=0.025,
        alternative="less",
    )
    assert LiftEstimate.model_validate_json(row.model_dump_json()) == row
    payload = row.model_dump(mode="json")
    payload["lift"].update(alpha=0.1, level=0.9)
    with pytest.raises(CodedError) as error:
        LiftEstimate.model_validate(payload)
    assert error.value.code == "estimation.winsor.invalid_state"


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_independent_mean_directional_fcr_reference_roundtrip(alternative):
    from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
    from increment.estimation.inference import infer_ate
    from increment.estimation.results import LiftEstimate, open_bound_from_two_sided_at_target

    ref = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", [1.0, 2.0, 3.0, 4.0], coefficient=1),)
    )
    row = infer_ate(
        "m",
        "T",
        "independent_mean",
        ref.point,
        ref,
        method_role="decision",
        value_scale="absolute",
        alpha=0.025,
        alternative=alternative,
    )
    opened = open_bound_from_two_sided_at_target(row)
    assert opened.independent_mean_reference is not None
    assert opened.independent_mean_reference.interval == "directional-fcr"
    assert LiftEstimate.model_validate_json(opened.model_dump_json()) == opened
