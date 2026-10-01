"""Contract pins for the typed decision computation bundle."""

from typing import cast

import pytest

from increment.decision import (
    ArmHypothesisKey,
    DecisionComputation,
    EValueEvidence,
    PValueEvidence,
)
from increment.estimation.results import LiftEstimate


def _pvalue(bundle, key) -> PValueEvidence:
    return cast(PValueEvidence, bundle.evidence[key])


def _evalue(bundle, key) -> EValueEvidence:
    return cast(EValueEvidence, bundle.evidence[key])


def _arm_key(key) -> ArmHypothesisKey:
    assert isinstance(key, ArmHypothesisKey)
    return key


def _key(metric="rev"):
    return ArmHypothesisKey(metric=metric, group_id="treatment", estimand="itt")


def _evidence(key):
    return PValueEvidence(
        hypothesis=key,
        method="unadjusted",
        p_value=0.04,
        reference="normal",
    )


def test_computation_is_not_a_result_sequence():
    bundle = DecisionComputation(results=(), evidence={}, failures={})
    with pytest.raises(TypeError):
        len(bundle)  # ty: ignore[invalid-argument-type] -- contract tests deliberate non-sequence access
    with pytest.raises(TypeError):
        iter(bundle)  # ty: ignore[no-matching-overload] -- contract tests deliberate non-iterable access
    with pytest.raises(TypeError):
        bundle[0]  # ty: ignore[not-subscriptable] -- contract tests deliberate non-subscriptable access


def test_computation_rejects_mismatched_keys():
    key, other = _key(), _key("gmv")
    with pytest.raises(AssertionError):
        DecisionComputation(results=(), evidence={other: _evidence(key)}, failures={})


def test_computation_freezes_results_and_maps():
    key = _key()
    rows: list = []
    bundle = DecisionComputation(results=rows, evidence={key: _evidence(key)}, failures={})
    rows.append(object())
    assert bundle.results == ()
    with pytest.raises(TypeError):
        bundle.evidence[key] = _evidence(key)  # type: ignore[index]


@pytest.mark.parametrize(
    "context, expected",
    [
        ({"display": "detail", "reason": "reason"}, "detail"),
        ({"reason": "reason"}, "reason"),
        ({}, "unavailable"),
    ],
)
def test_decision_failure_display_preserves_detail_precedence(context, expected):
    from increment.decision import DecisionFailure

    failure = DecisionFailure(_key(), "unavailable", context)
    assert failure.display() == expected


def test_randomized_fixed_pvalue_uses_raw_sampling_pair_with_informative_prior():
    import math

    from scipy.stats import norm

    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import Normal, infer_lift

    row = infer_lift(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        log_rr=math.log(1.2) - 0.0,
        se_t=0.04,
        se_c=0.03,
        prior=Normal(mu=0.0, sigma=0.01),
        method_role="decision",
    )
    bundle = _lift_decision_bundle([row], inference=None, allow_linear=True)
    raw_z = math.log(1.2) / math.sqrt(0.04**2 + 0.03**2)
    assert _pvalue(bundle, _key()).p_value == pytest.approx(2.0 * norm.sf(abs(raw_z)))
    assert _pvalue(bundle, _key()).reference == "normal"


@pytest.mark.slow
def test_estimator_families_emit_typed_evidence_and_one_of():
    from increment.decision import ContrastHypothesisKey, EValueEvidence
    from increment.estimation.contrast import compute_contrast
    from tests.estimation.test_contrast import _procedure, _stats
    from tests.estimation.test_encouragement import _design
    from tests.estimation.test_sequential_engine import _mean_metric

    contrast = compute_contrast(_stats(), _procedure())
    contrast_key = ContrastHypothesisKey("orders", "control", "treatment")
    assert contrast_key in contrast.evidence
    assert not contrast.evidence.keys() & contrast.failures.keys()

    from increment.estimation.encouragement import estimate_encouragement
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli()
    av = estimate_encouragement(
        [_mean_metric(7)],
        snapshot,
        _design(one_sided=True),
        estimands=("itt",),
        inference=policy,
    )
    assert isinstance(next(iter(av.evidence.values())), EValueEvidence)
    assert av.results[0].stat_sig()


def test_observational_and_encouragement_linear_rows_emit_raw_pvalues():
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.encouragement import estimate_encouragement
    from tests.estimation.test_adjust import _DESIGN, _src
    from tests.estimation.test_encouragement import METRIC, _design, _rows

    observational = estimate_ate(_src(), _DESIGN)
    assert observational.evidence or observational.failures
    encouragement = estimate_encouragement(
        [METRIC], _rows(), _design(one_sided=True), estimands=("compliance", "late")
    )
    assert encouragement.evidence
    assert not encouragement.evidence.keys() & encouragement.failures.keys()


@pytest.mark.slow
def test_sequential_evalue_is_independently_computed():
    from fractions import Fraction

    from increment.estimation._certified import log_interval
    from tests.estimation.test_family import _raw_family

    _, bundle = _raw_family(c=[0, 0], t=[1, 1])
    evidence = next(iter(bundle.evidence.values()))
    assert isinstance(evidence, EValueEvidence)
    # Each Beta(1,1) ordered sequence has mass 1/3; null maximum is 1/16.
    oracle = log_interval(Fraction(16, 9))
    assert evidence.certificate.log_e is not None
    assert evidence.certificate.log_e.lo <= oracle.hi
    assert evidence.certificate.log_e.hi >= oracle.lo
    assert evidence.e_value == pytest.approx(16 / 9)


@pytest.mark.slow
@pytest.mark.parametrize(
    "alternative,c,t",
    [
        ("greater", [0, 1, 1, 1] * 24, [0, 0, 0, 1] * 24),
        ("less", [0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24),
    ],
)
def test_sequential_evalue_one_sided_uses_composite_null(alternative, c, t):
    from tests.estimation.test_family import _raw_family

    _, bundle = _raw_family(alternative=alternative, c=c, t=t)
    evidence = next(iter(bundle.evidence.values()))
    assert isinstance(evidence, EValueEvidence)
    assert evidence.log_e < 0
    assert not bundle.results[0].stat_sig()


@pytest.mark.slow
def test_one_sided_always_valid_secondary_does_not_discover_wrong_direction_effect():
    from increment import Analysis, AnalysisPlan, SequentialCell
    from increment.frame import MetricSpec
    from tests.test_sequential_public_sources import _frame, _plan

    specs = [MetricSpec(name="outcome", type="conversion")]
    base = _plan(
        specs,
        "bernoulli",
        cells=(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                alternative="greater",
                family=True,
            ),
        ),
    )
    plan = AnalysisPlan(
        secondaries=["outcome"],
        alternative="greater",
        inference=base.inference,
    )
    analysis = Analysis.from_unit_summary(
        _frame([1, 1, 0] * 100, [0, 0, 1] * 100),
        unit="unit",
        group="arm",
        control="control",
        metrics=specs,
        experiment_id="experiment",
        exposure_date="exposure",
        plan=plan,
    )
    row = analysis.run()[0]
    assert isinstance(row, LiftEstimate)
    assert row.alternative == "greater" and row.require_lift().value < 0
    assert row.discovery is False


def test_clustered_pvalue_uses_a_t_reference():
    import math

    from scipy.stats import t

    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import infer_lift

    row = infer_lift(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        log_rr=0.12 - 0.0,
        se_t=0.03,
        se_c=0.04,
        dof=8.0,
        n_clusters=10,
        method_role="decision",
    )
    bundle = _lift_decision_bundle([row], inference=None)
    evidence = _pvalue(bundle, _key())
    expected = 2.0 * t.sf(abs(0.12 / math.sqrt(0.03**2 + 0.04**2)), 8.0)
    assert evidence.p_value == pytest.approx(expected)
    assert evidence.reference == "t_8"


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_welch_only_pvalue_uses_a_t_reference_not_normal(alternative):
    import math

    from scipy.stats import t

    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import LiftEstimate

    row = infer_lift(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        log_rr=0.12 - 0.0,
        se_t=0.03,
        se_c=0.04,
        arm_ns=(4, 5),
        alternative=alternative,
        null_lift=0.02,
        method_role="decision",
    )
    df = (0.03**2 + 0.04**2) ** 2 / (0.03**4 / 3 + 0.04**4 / 4)
    z = (0.12 - math.log1p(0.02)) / math.hypot(0.03, 0.04)
    expected = {
        "greater": t.sf(z, df),
        "less": t.cdf(z, df),
        "two-sided": 2.0 * t.sf(abs(z), df),
    }[alternative]
    for estimate in (row, LiftEstimate.model_validate_json(row.model_dump_json())):
        evidence = _pvalue(_lift_decision_bundle([estimate], inference=None), _key())
        assert estimate.dof is None
        assert estimate.p_value() == pytest.approx(expected)
        assert evidence.p_value == pytest.approx(expected)
        assert evidence.reference == f"t_{df:g}"


def test_welch_only_absolute_margin_failure_displays_a_t_reference():
    from increment.errors import InvalidRequestError
    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import LiftEstimate

    row = infer_lift(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        log_rr=0.05 - 0.0,
        se_t=0.015,
        se_c=0.013,
        arm_ns=(40, 45),
        abs_diff=1.0,
        abs_se=0.4,
        null_abs=0.0,
        preferred_direction="increase",
        method_role="decision",
    )
    for estimate in (row, LiftEstimate.model_validate_json(row.model_dump_json())):
        assert estimate.abs_diff == 1.0 and estimate.abs_se == 0.4
        assert estimate.abs_lb is None and estimate.abs_ub is None
        assert estimate.require_lift().lb is not None and estimate.require_lift().ub is not None
        assert estimate.stat_sig() is False
        for method, code in (
            (estimate.p_value, "estimation.results.lift.p_value_cluster_robust_null_abs"),
            (
                estimate.prob_favorable,
                "estimation.results.lift.p_value_cluster_robust_null_abs",
            ),
        ):
            with pytest.raises(InvalidRequestError) as exc_info:
                method()
            assert exc_info.value.code == code
            assert exc_info.value.context["reference_df"] == estimate.reference_df
        bundle = _lift_decision_bundle([estimate], inference=None)
        assert _key() not in bundle.evidence
        failure = bundle.failures[_key()]
        assert failure.code == "evidence.p_value.unavailable"
        assert failure.context["reason"] == "clustered_absolute_margin"
        assert failure.context["reference_kind"] == "t"
        assert failure.context["reference_df"] == estimate.reference_df
        assert failure.display() == failure.context["display"]


def test_guarded_cell_failure_preserves_successful_sibling():
    from increment.estimation.engine import estimate_lift
    from tests.estimation.test_engine import _mean_metric

    rows = [
        {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 10.0,
            "cy1": 0.0,
            "cy2": 100.0,
        },
        {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "good",
            "n": 100,
            "ref_y": 11.0,
            "cy1": 0.0,
            "cy2": 100.0,
        },
        {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "bad",
            "n": 100,
            "ref_y": 11.0,
            "cy1": 0.0,
            "cy2": 0.0,
        },
    ]
    bundle = estimate_lift([_mean_metric()], rows, "control")
    assert [row.group_id for row in bundle.results] == ["good"]

    assert _arm_key(next(iter(bundle.failures))).group_id == "bad"


def test_prior_shrunk_row_excluded_from_evidence_when_allow_linear_false():
    from increment.decision import ArmHypothesisKey
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import Normal, infer_ate

    row = infer_ate(
        "revenue",
        "treatment",
        "iptw",
        point=0.022,
        scores=ScoreStats(
            metric="revenue", contrast="treatment", n=10000, sum_psi=0.0, sum_psi2=10000.0
        ),
        prior=Normal(mu=0.0, sigma=0.005),
        method_role="decision",
    )
    assert row.prior_shrunk is True
    assert row.note is None
    bundle = _lift_decision_bundle([row], inference=None, allow_linear=False)
    key = ArmHypothesisKey("revenue", "treatment", "ate")
    assert key not in bundle.evidence
    assert bundle.failures == {}
    assert len(bundle.results) == 1
    surviving = bundle.results[0]
    assert surviving.lift == row.lift
    assert surviving.note is not None and surviving.note.startswith("prior_shrunk:")


def test_prior_shrunk_row_emits_evidence_when_allow_linear_true():
    from increment.decision import ArmHypothesisKey
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import Normal, infer_ate

    row = infer_ate(
        "revenue",
        "treatment",
        "iptw",
        point=0.022,
        scores=ScoreStats(
            metric="revenue", contrast="treatment", n=10000, sum_psi=0.0, sum_psi2=10000.0
        ),
        prior=Normal(mu=0.0, sigma=0.005),
        method_role="decision",
    )
    bundle = _lift_decision_bundle([row], inference=None, allow_linear=True)
    key = ArmHypothesisKey("revenue", "treatment", "ate")
    assert isinstance(bundle.evidence[key], PValueEvidence)
    assert _pvalue(bundle, key).p_value == pytest.approx(0.02780689502699724, rel=1e-9)


def test_prior_shrunk_marker_appends_to_an_existing_note():
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.engine import _lift_decision_bundle
    from increment.estimation.inference import Normal, infer_ate

    row = infer_ate(
        "revenue",
        "treatment",
        "dml",
        point=0.022,
        scores=ScoreStats(
            metric="revenue", contrast="treatment", n=10000, sum_psi=0.0, sum_psi2=10000.0
        ),
        prior=Normal(mu=0.0, sigma=0.005),
        method_role="decision",
    ).model_copy(update={"note": "Observational design -- confounded; not a causal estimate"})
    bundle = _lift_decision_bundle([row], inference=None, allow_linear=False)
    surviving = bundle.results[0]
    assert surviving.note is not None
    assert surviving.note.startswith("prior_shrunk:")
    assert surviving.note.endswith("Observational design -- confounded; not a causal estimate")


def test_guarded_sensitivity_does_not_fail_decision_hypothesis(monkeypatch):
    from increment.decision import ArmHypothesisKey
    from increment.estimation.engine import LiftGuardError, Method, estimate_lift
    from increment.estimation.engine import infer_lift as real_infer_lift
    from tests.estimation.test_engine import _mean_metric

    def guarded(**kwargs):
        if kwargs["method"] == "cuped":
            raise LiftGuardError("sensitivity guard", reason="delta_method_unreliable")
        return real_infer_lift(**kwargs)

    monkeypatch.setattr("increment.estimation.engine.infer_lift", guarded)
    rows = [
        {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 10.0,
            "cy1": 0.0,
            "cy2": 100.0,
            "ref_x": 1.0,
            "cx1": 0.0,
            "cx2": 100.0,
            "cxy": 0.0,
        },
        {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "treatment",
            "n": 100,
            "ref_y": 11.0,
            "cy1": 0.0,
            "cy2": 100.0,
            "ref_x": 1.0,
            "cx1": 0.0,
            "cx2": 100.0,
            "cxy": 0.0,
        },
    ]
    bundle = estimate_lift(
        [_mean_metric()],
        rows,
        "control",
        methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
    )
    key = ArmHypothesisKey("rev", "treatment", "itt")
    assert key in bundle.evidence
    assert key not in bundle.failures


def test_guarded_decision_failure_keeps_the_guard_message_for_display(monkeypatch):
    from increment.estimation.engine import LiftGuardError, Method, estimate_lift
    from tests.estimation.test_engine import _mean_metric

    def guarded(**kwargs):
        raise LiftGuardError("guard diagnostic", reason="delta_method_unreliable")

    monkeypatch.setattr("increment.estimation.engine.infer_lift", guarded)
    arm = {"experiment_id": "e", "metric": "rev", "n": 100, "cy1": 0.0, "cy2": 100.0}
    rows = [
        {**arm, "group_id": "control", "ref_y": 10.0},
        {**arm, "group_id": "treatment", "ref_y": 11.0},
    ]
    bundle = estimate_lift([_mean_metric()], rows, "control", methods=[Method(name="unadjusted")])
    (failure,) = bundle.failures.values()
    assert failure.context["reason"] == "delta_method_unreliable"
    assert failure.display() == "guard diagnostic"
