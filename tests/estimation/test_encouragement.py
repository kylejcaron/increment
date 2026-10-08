import math
from collections.abc import MutableMapping
from typing import cast

import numpy as np
import pytest

from increment import IdentificationError
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.encouragement import _late_relative, _nn_estimate, estimate_encouragement
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.semantics.design import Encouragement
from increment.semantics.models import MeanMetric, RatioMetric, RetentionMetric
from tests.warning_codes import warning_codes


def _sequential_policy(*, uptake=False):
    from increment import PredictivePrior, SequentialCell, SequentialModel, SequentialRegistration
    from increment.estimation.sequential import AsymptoticMean
    from tests.asymptotic_cases import mean_model
    from tests.sequential_cases import registration

    base = registration("bernoulli")  # only used for reveal/control_group/definitions_id shape
    models = [mean_model("rev", start_count=4)]
    cells = [SequentialCell(metric="rev", group_id="treat")]
    if uptake:
        prior = PredictivePrior(kind="beta", a=1, b=1)
        models.append(
            SequentialModel(
                metric="uptake",
                observable="uptake",
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            )
        )
        cells.append(SequentialCell(metric="uptake", group_id="treat", estimand="compliance"))
    reg = SequentialRegistration.model_validate(
        {**base.model_dump(), "models": models, "roster": cells}
    )
    if uptake:
        from increment.estimation.sequential import MixedFamily

        return MixedFamily(registration=reg)
    return AsymptoticMean(registration=reg)


def _sequential_snapshot(policy):
    from tests.sequential_cases import capture

    rows = []
    for i in range(96):
        for arm in ("control", "treat"):
            values = {"rev": (1, 2, 3, 4)[i % 4] if arm == "control" else (6, 10, 14, 18)[i % 4]}
            if any(model.observable == "uptake" for model in policy.registration.models):
                values["uptake"] = int(i % 4 == 3) if arm == "control" else int(i % 4 != 0)
            rows.append({"unit_id": f"{i:04d}-{arm}", "group_id": arm, "values": values})
    return capture(policy.registration, rows)


def _design(**over):
    base = {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "help_click"},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "unclicked button assumed inert",
        },
    }
    base.update(over)
    return Encouragement.model_validate(base)


METRIC = MeanMetric(name="rev", entity="user_id", fact="orders", aggregation="sum")


def _rows(n=4000, tau=2.0, compliance=0.5, one_sided=True, seed=7):
    """Simulate an encouragement DGP and collapse to group_summary rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid, encouraged in (("control", 0), ("treat", 1)):
        if encouraged:
            d = rng.binomial(1, compliance, size=n)
        else:
            d = np.zeros(n) if one_sided else rng.binomial(1, 0.1, size=n)
        y = 10.0 + tau * d + rng.normal(0, 2.0, size=n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    return rows


def _by_estimand(results):
    out = {}
    for r in results:
        out.setdefault((r.estimand, r.value_scale), []).append(r)
    return out


def test_reports_all_three_estimands_by_default():
    res = estimate_encouragement([METRIC], _rows(), _design(one_sided=True)).results
    got = _by_estimand(res)
    assert ("itt", "relative") in got
    assert ("compliance", "absolute") in got
    assert ("late", "absolute") in got


@pytest.mark.parametrize("estimands", [("itt",), ("compliance",), ("itt", "compliance")])
def test_itt_and_compliance_allow_omitted_exclusion(estimands):
    declared = _design(one_sided=True)
    payload = declared.model_dump()
    payload.pop("exclusion_restriction")
    omitted = Encouragement.model_validate(payload)
    rows = _rows(n=80, one_sided=True)
    expected = estimate_encouragement([METRIC], rows, declared, estimands=estimands)
    for design in (omitted, Encouragement.model_validate_json(omitted.model_dump_json())):
        actual = estimate_encouragement([METRIC], rows, design, estimands=estimands)
        assert {row.estimand for row in actual.results} == set(estimands)
        assert actual == expected


@pytest.mark.parametrize("estimands", [None, ("late",), ("itt", "late")])
def test_late_without_exclusion_refuses_before_summary_access(estimands):
    import copy
    import pickle

    payload = _design(one_sided=True).model_dump()
    payload.pop("exclusion_restriction")
    design = Encouragement.model_validate(payload)

    def unread():
        raise AssertionError("LATE exclusion gate must precede summary access")
        yield

    with pytest.raises(IdentificationError) as raised:
        if estimands is None:
            estimate_encouragement([METRIC], unread(), design)
        else:
            estimate_encouragement([METRIC], unread(), design, estimands=estimands)
    for error in (
        raised.value,
        copy.deepcopy(raised.value),
        pickle.loads(pickle.dumps(raised.value)),
    ):
        assert error.code == "identification.encouragement.exclusion_required"
        assert error.context["estimands"] == (estimands or ("itt", "compliance", "late"))
        assert error.context["available_estimands"] == ("itt", "compliance")
        assert error.context["required_assumption"] == "exclusion_restriction"
        with pytest.raises(TypeError):
            cast(MutableMapping[str, object], error.context)["estimands"] = ()


def test_weak_first_stage_does_not_waive_missing_exclusion():
    design = Encouragement(control_group="control", uptake={"fact": "help_click"})
    with pytest.raises(IdentificationError) as raised:
        estimate_encouragement([METRIC], _rows(n=32, compliance=0), design)
    assert raised.value.code == "identification.encouragement.exclusion_required"


def test_clustered_run_stamps_t_reference_on_every_dof_row():
    from scipy.stats import t

    from increment.estimation.results import LiftEstimate
    from tests.estimation.test_late_cluster import _arm_rows

    metric = MeanMetric(name="m", entity="user_id", fact="purchase", aggregation="sum")
    rng = np.random.default_rng(13)
    _, control = _arm_rows(rng, "control", k=20, m=5, treated=False)
    _, treatment = _arm_rows(rng, "treatment", k=20, m=5, treated=True)
    computation = estimate_encouragement(
        [metric],
        [control, treatment],
        _design(one_sided=True),
        estimands=("compliance", "late"),
        cluster="store",
    )
    assert {row.estimand for row in computation.results} == {"compliance", "late"}
    compliance = treatment["ref_x"] / 5
    tau = (treatment["ref_y"] - control["ref_y"]) / (5 * compliance)
    late_components = [
        (arm["cy2"] - 2 * tau * arm["cxy"] + tau**2 * arm["cx2"]) / (20 * 19 * 25)
        for arm in (control, treatment)
    ]
    expected_df = {
        "compliance": 19.0,
        "late": sum(late_components) ** 2 / sum(v**2 / 19 for v in late_components),
    }
    for row in computation.results:
        restored = LiftEstimate.model_validate_json(row.model_dump_json())
        assert restored.reference_kind == "t"
        assert restored.reference_df == restored.dof
        assert restored.reference_df == pytest.approx(expected_df[row.estimand], rel=1e-12)
        lift = restored.require_lift()
        assert lift.log_mean is not None and lift.log_se is not None
        expected = 2.0 * t.sf(abs(lift.log_mean / lift.log_se), expected_df[row.estimand])
        assert restored.p_value() == pytest.approx(expected)


def _winsorized_rows(**metadata):
    rows = _rows()
    for row, n, n_upper in zip(rows, (4000, 4000), (3, 5), strict=True):
        row.update(winsor_n=n, winsor_n_lower=0, winsor_n_upper=n_upper, **metadata)
    return rows


def test_late_rows_carry_winsorization_diagnostics():
    """A fixed cutoff's pooled bound and per-arm cap counts reach the LATE rows."""
    rows = _winsorized_rows(winsor_upper_bound=100.0)

    results = estimate_encouragement([METRIC], rows, _design(one_sided=True)).results

    late = [r for r in results if r.estimand == "late"]
    assert late
    assert all(r.winsor_upper_bound == 100.0 for r in late)
    assert all(r.winsor_control_n == 4000 and r.winsor_control_n_upper == 3 for r in late)
    assert all(r.winsor_treatment_n == 4000 and r.winsor_treatment_n_upper == 5 for r in late)
    assert all(r.winsor_control_fraction_upper == 3 / 4000 for r in late)


def test_late_refuses_percentile_metadata_as_fixed_clipped_moments():
    """A sample percentile makes the cutoff random, so pre-clipped moments are
    not the winsorized estimator's sampling distribution: calibrating it needs
    raw unit state, and the encouragement route refuses rather than understate
    variance."""
    from increment.errors import CodedError

    rows = _winsorized_rows(winsor_upper_percentile=0.99, winsor_upper_bound=100.0)

    with pytest.raises(CodedError) as error:
        estimate_encouragement([METRIC], rows, _design(one_sided=True))
    assert error.value.code == "estimation.winsor.raw_state_required"


def test_additive_late_recovers_tau():
    res = estimate_encouragement([METRIC], _rows(tau=2.0), _design(one_sided=True)).results
    late = [r for r in res if r.estimand == "late" and r.value_scale == "absolute"][0]
    lift = late.require_lift()
    assert lift.value == pytest.approx(2.0, abs=0.35)
    assert lift.lb is not None and lift.ub is not None
    assert lift.lb < 2.0 < lift.ub


def test_complier_relative_late_recovers_ratio():
    # complier control mean = 10, complier treated mean = 12 -> +20%
    res = estimate_encouragement([METRIC], _rows(tau=2.0), _design(one_sided=True)).results
    rel = [r for r in res if r.estimand == "late" and r.value_scale == "relative"]
    assert rel, "relative LATE should be emitted when the guard passes"
    assert rel[0].require_lift().value == pytest.approx(0.20, abs=0.06)


def test_weak_instrument_suppresses_late_but_keeps_itt():
    res = estimate_encouragement(
        [METRIC], _rows(n=200, compliance=0.02), _design(one_sided=True)
    ).results
    got = _by_estimand(res)
    assert not any(k[0] == "late" for k in got)
    assert ("itt", "relative") in got
    comp = got[("compliance", "absolute")][0]
    assert comp.note is not None and "suppressed" in comp.note


def test_negative_first_stage_is_named_not_called_weak():
    """A strongly NEGATIVE first stage (encouragement REDUCED uptake) is
    an instrumentation/product bug, not a weak instrument - suppression
    is still correct, but the reason must name the sign instead of
    calling a |z| of ~18 'weak'."""
    rows = _rows(n=4000, compliance=0.01, one_sided=False)  # control uptake 0.1 >> treat 0.01
    res = estimate_encouragement([METRIC], rows, _design(one_sided=False)).results
    comp = [r for r in res if r.estimand == "compliance"]
    assert comp and comp[0].note is not None
    assert "suppressed" in comp[0].note
    assert "reduced uptake" in comp[0].note
    assert "weak instrument" not in comp[0].note


def test_pop_label_derives_from_the_uptake_fact():
    """One-sided LATE rows describe the affected subpopulation via the
    DECLARED uptake fact ('help_click takers'), not a hardcoded product
    flavor ('clickers/adopters') that is wrong for vaccinations,
    enrollments, migrations..."""
    res = estimate_encouragement([METRIC], _rows(), _design(one_sided=True)).results
    late_notes = [r.note for r in res if r.estimand == "late"]
    assert late_notes and all(n is not None for n in late_notes)
    assert all(n is not None and "help_click takers" in n for n in late_notes)
    assert all(n is not None and "clickers/adopters" not in n for n in late_notes)


def test_late_just_above_the_gate_carries_a_selection_caveat():
    """Estimates emitted just above min_first_stage_z are conditionally
    selected on a lucky first stage and lean toward the as-treated value
    under confounding - the emitted note must say so. Exact one-sided
    mixture moments give z_fs ~ 4.5, inside (gate, gate+1)."""
    rows = _analytic_encouragement_rows(
        n=400, p=0.048, m_complier=12.0, sd_complier=0.5, m_never=10.0, sd_never=0.5
    )
    res = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), estimands=("late",)
    ).results
    late = [r for r in res if r.estimand == "late"]
    assert late
    for r in late:
        assert r.note is not None and "emission gate" in r.note


def test_late_comfortably_above_the_gate_has_no_selection_caveat():
    res = estimate_encouragement([METRIC], _rows(), _design(one_sided=True)).results
    late = [r for r in res if r.estimand == "late"]
    assert late
    for r in late:
        assert r.note is None or "emission gate" not in r.note


def test_one_sided_violation_hard_errors():
    rows = _rows(one_sided=False)  # control has uptake
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], rows, _design(one_sided=True))
    assert exc_info.value.code == "estimation.encouragement.one_sided_encouragement"


def test_two_sided_compliance_works():
    res = estimate_encouragement([METRIC], _rows(one_sided=False), _design(one_sided=False)).results
    late = [r for r in res if r.estimand == "late" and r.value_scale == "absolute"]
    assert late and late[0].require_lift().value == pytest.approx(2.0, abs=0.5)
    comp = [r for r in res if r.estimand == "compliance"]
    assert comp and all(r.value_scale == "relative" for r in comp)


def test_declared_preferred_direction_reaches_itt_row_not_uptake():
    """A metric declaring preferred_direction="decrease" must reach its
    `itt` row - `prob_favorable()` raises without it. The two-sided
    `compliance` row is the synthetic uptake metric's first-stage lift,
    not the outcome's: it must stay direction-agnostic (None), not
    inherit the outcome metric's direction."""
    decrease_metric = MeanMetric(
        name="rev",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
    )
    res = estimate_encouragement(
        [decrease_metric], _rows(one_sided=False), _design(one_sided=False)
    ).results
    itt = [r for r in res if r.estimand == "itt"]
    assert itt and all(r.preferred_direction == "decrease" for r in itt)
    comp = [r for r in res if r.estimand == "compliance" and r.value_scale == "relative"]
    assert comp and all(r.preferred_direction is None for r in comp)


def test_two_sided_compliance_degrades_to_absolute_when_relative_refused():
    """Tiny arms (n=4, uptake 1/4 vs 3/4) make the RELATIVE uptake lift's
    combined log-scale SE ~ 1.05, past the delta-method guard, but
    the uptake lift itself is perfectly estimable on the absolute scale
    (B = 0.5, se = 0.354). The compliance row must degrade to the
    absolute form with the withheld reason on its note, not abort the
    whole encouragement call."""
    rows = []
    for gid, sum_d in (("control", 1.0), ("treat", 3.0)):
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": 4,
                    "sum_y": 40.0,
                    "sum_y2": 420.0,
                    "sum_d": sum_d,
                    "sum_yd": 10.0 * sum_d,
                    "sum_y2d": 105.0 * sum_d,
                }
            )
        )
    res = estimate_encouragement(
        [METRIC], rows, _design(one_sided=False), estimands=("compliance",)
    ).results
    comp = [r for r in res if r.estimand == "compliance"]
    assert len(comp) == 1
    assert comp[0].value_scale == "absolute"
    assert comp[0].scale == "linear"
    assert comp[0].require_lift().value == pytest.approx(0.5)


def test_mixture_prior_refuses_for_a_valid_encouragement_design():
    from increment.estimation.priors import StudentTPrior

    with pytest.raises(InvalidRequestError) as raised:
        estimate_encouragement(
            [METRIC], _rows(), _design(), prior=StudentTPrior(nu=4.0, scale=0.05)
        )
    assert raised.value.code == "estimation.adjust.prior.type"


def test_prior_keeps_compliance_and_late_sampling_rows_and_stores_posteriors():
    plain = estimate_encouragement([METRIC], _rows(one_sided=True), _design(one_sided=True)).results
    informative = estimate_encouragement(
        [METRIC],
        _rows(one_sided=True),
        _design(one_sided=True),
        prior=Normal(mu=0.0, sigma=0.05),
    ).results

    assert len(informative) == len(plain)
    for prior_row, plain_row in zip(informative, plain, strict=True):
        assert prior_row.require_lift() == plain_row.require_lift()
        assert prior_row.prior_shrunk is False
    supported = [row for row in informative if row.estimand in {"compliance", "late"}]
    assert supported
    assert all(row.posterior_available is True for row in supported)
    assert all(row.posterior_estimate is not None for row in supported)


def test_relative_compliance_fallback_keeps_prior_free_sampling_fields():
    rows = []
    for gid, sum_d in (("control", 1.0), ("treat", 3.0)):
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": 4,
                    "sum_y": 40.0,
                    "sum_y2": 420.0,
                    "sum_d": sum_d,
                    "sum_yd": 10.0 * sum_d,
                    "sum_y2d": 105.0 * sum_d,
                }
            )
        )
    plain = estimate_encouragement(
        [METRIC], rows, _design(one_sided=False), estimands=("compliance",)
    ).results
    informative = estimate_encouragement(
        [METRIC],
        rows,
        _design(one_sided=False),
        estimands=("compliance",),
        prior=Normal(mu=0.0, sigma=0.05),
    ).results

    assert len(informative) == len(plain) == 1
    assert informative[0].note is not None
    assert "relative uptake lift withheld" in informative[0].note
    assert informative[0].require_lift() == plain[0].require_lift()
    assert informative[0].posterior_available is True
    assert informative[0].posterior_estimate is not None


def test_unknown_estimand_rejected():
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], _rows(), _design(), estimands=("itt", "as_treated"))
    assert exc_info.value.code == "estimation.encouragement.unknown_estimand_supported"


def test_precise_extreme_complier_ratio_emits_relative_form():
    """theta = log(N1/N0) ~ 3.9 measured PRECISELY (z(N1) ~ 21, se_theta
    ~ 0.13) is exactly where the log delta method works - the relative
    row must be emitted. The old point-value guard (theta >= 2.5) refused
    on the estimate rather than on the approximation quality, which is a
    precision property of log(N1) and log(N0), not of the ratio itself.
    """
    rng = np.random.default_rng(3)
    n = 4000
    rows = []
    for gid, encouraged in (("control", 0), ("treat", 1)):
        if encouraged:
            d = rng.binomial(1, 0.1, size=n)
        else:
            d = np.zeros(n)
        # Compliers' outcome jumps to ~51 vs base ~1, so complier control
        # mean N0 stays small while N1 grows large past e^2.5 (~12.2).
        y = 1.0 + 50.0 * d + rng.normal(0, 0.5, size=n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )

    res = estimate_encouragement([METRIC], rows, _design(one_sided=True)).results
    prior_res = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), prior=Normal(mu=0.0, sigma=0.1)
    ).results
    got = _by_estimand(res)
    prior_got = _by_estimand(prior_res)

    assert ("itt", "relative") in got
    additive = got[("late", "absolute")]
    assert additive and "withheld" not in (additive[0].note or "")
    rel = got.get(("late", "relative"))
    assert rel, "a precisely-estimated extreme ratio must not be refused"
    # value = exp(theta) - 1 with theta ~ log(51/1) ~ 3.9
    assert rel[0].require_lift().value == pytest.approx(np.expm1(3.915), rel=0.05)
    assert rel[0].require_lift().lb < rel[0].require_lift().value < rel[0].require_lift().ub
    prior_rel = prior_got[("late", "relative")][0]
    assert prior_rel.require_lift() == rel[0].require_lift()
    assert prior_rel.posterior_available is True
    assert prior_rel.posterior_scale == "log"
    assert prior_rel.posterior_estimate != pytest.approx(rel[0].require_lift().value)


def _analytic_encouragement_rows(n, p, m_complier, sd_complier, m_never, sd_never):
    """Exact one-sided mixture moments, no sampling: treated compliers'
    outcomes are (m_complier, sd_complier), never-takers and the whole
    control arm are (m_never, sd_never)."""
    treat = centered_row_from_raw_sums(
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "treat",
            "n": n,
            "sum_y": n * (p * m_complier + (1 - p) * m_never),
            "sum_y2": n
            * (p * (sd_complier**2 + m_complier**2) + (1 - p) * (sd_never**2 + m_never**2)),
            "sum_d": n * p,
            "sum_yd": n * p * m_complier,
            "sum_y2d": n * p * (sd_complier**2 + m_complier**2),
        }
    )
    control = centered_row_from_raw_sums(
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "control",
            "n": n,
            "sum_y": n * m_never,
            "sum_y2": n * (sd_never**2 + m_never**2),
            "sum_d": 0.0,
            "sum_yd": 0.0,
            "sum_y2d": 0.0,
        }
    )
    return [treat, control]


def test_imprecise_complier_treated_mean_withholds_relative_form():
    """Negative-flank symmetry: a tiny complier treated mean (0.09, noise
    sd 3.5) against a precise complier control mean at n=2000/arm gives
    z(N1) = 0.81, so log(N1) is meaningless at that precision. The old
    guard had no lower flank and shipped it (theta=-4.71, se_theta=1.23:
    a log-scale 95% CI spanning a ~125x ratio range, undercovering), while
    refusing the exactly mirrored imprecision on N0. The additive row
    survives with the reason on its note, matching the N0 flank."""
    rows = _analytic_encouragement_rows(
        n=2000, p=0.5, m_complier=0.09, sd_complier=3.5, m_never=10.0, sd_never=0.5
    )
    res = estimate_encouragement([METRIC], rows, _design(one_sided=True)).results
    got = _by_estimand(res)

    assert ("late", "relative") not in got, "imprecise log(N1) must be withheld"
    additive = got[("late", "absolute")]
    assert additive and additive[0].note is not None
    assert "relative form withheld" in additive[0].note
    assert "complier treated mean too imprecise" in additive[0].note


def test_complier_precision_guards_are_symmetric_across_flanks():
    """The mirrored configuration - imprecise complier CONTROL mean,
    precise treated mean - is refused by the pre-existing N0 floor; both
    flanks now share the same precision-based refusal, so neither sign of
    theta ships a garbage-wide interval."""
    # N0 = 0.045 is the imprecise leg; estimands=("late",) skips the ITT
    # delegation, which would otherwise refuse this fixture on its own.
    rows = _analytic_encouragement_rows(
        n=2000, p=0.5, m_complier=10.0, sd_complier=0.5, m_never=0.09, sd_never=3.5
    )
    res = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), estimands=("late",)
    ).results
    got = _by_estimand(res)

    assert ("late", "relative") not in got, "imprecise log(N0) must be withheld"
    additive = got[("late", "absolute")]
    assert additive and additive[0].note is not None
    assert "relative form withheld" in additive[0].note
    assert "complier control mean too imprecise" in additive[0].note


def test_cuped_method_with_late_refuses_on_a_covariate_free_summary():
    """The blanket "late is always unadjusted" refusal is gone - CUPED now
    adjusts the LATE numerator. On a summary with no covariate moments the
    refusal is the specific covariate-not-materialised one instead, which
    names the two switches that fix it."""
    from increment.estimation.engine import Method

    with pytest.raises(InvalidRequestError) as exc:
        estimate_encouragement(
            [METRIC],
            _rows(),
            _design(),
            estimands=("late",),
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )
    assert exc.value.code == "estimation.cuped.arm_no_covariate"


def test_alternative_forwarded_to_itt_and_late_doubles_alpha_and_labels():
    """alternative= threads to the itt row (via estimate_lift/infer_lift)
    and both late rows (additive + relative, via _nn_estimate), the
    same alpha-doubling identity infer_lift/infer_ate use elsewhere.
    compliance stays two-sided: it's an instrument-strength diagnostic,
    not a hypothesis test the caller picks a tail for."""
    rows = _rows(tau=2.0)
    design = _design(one_sided=True)
    two = _by_estimand(estimate_encouragement([METRIC], rows, design, alpha=0.10).results)
    one = _by_estimand(
        estimate_encouragement([METRIC], rows, design, alpha=0.05, alternative="greater").results
    )
    for key in (("itt", "relative"), ("late", "absolute"), ("late", "relative")):
        one_r, two_r = one[key][0], two[key][0]
        assert one_r.require_lift().lb == pytest.approx(two_r.require_lift().lb, rel=1e-9), key
        assert one_r.require_lift().ub == pytest.approx(two_r.require_lift().ub, rel=1e-9), key
        assert one_r.require_lift().level == pytest.approx(0.90), key
        assert one_r.alternative == "greater", key
        assert two_r.alternative == "two-sided", key
    assert one[("compliance", "absolute")][0].alternative == "two-sided"


def test_default_alternative_reproduces_todays_behavior():
    """Default-unchanged regression: omitting alternative= on every row
    (itt/compliance/late) is byte-identical to today's pre-change output."""
    rows = _rows(tau=2.0)
    design = _design(one_sided=True)
    default = estimate_encouragement([METRIC], rows, design).results
    assert all(r.alternative == "two-sided" for r in default)
    explicit = estimate_encouragement([METRIC], rows, design, alternative="two-sided").results
    for d, e in zip(default, explicit, strict=True):
        assert d.require_lift().value == pytest.approx(e.require_lift().value, rel=1e-12)
        assert d.require_lift().lb == pytest.approx(e.require_lift().lb, rel=1e-12)
        assert d.require_lift().ub == pytest.approx(e.require_lift().ub, rel=1e-12)


def test_alternative_rejects_unknown_value():
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], _rows(), _design(), alternative="bigger")
    assert exc_info.value.code == "estimation.binomial.unknown_alternative"


def test_additive_late_row_carries_linear_scale():
    rows = estimate_encouragement([METRIC], _rows(), _design(one_sided=True)).results
    additive = [r for r in rows if r.estimand == "late" and r.value_scale == "absolute"]
    assert additive and all(r.scale == "linear" for r in additive)


def test_minimum_alpha_survives_one_sided_effective_tail():
    """The one-sided effective-tail identity holds at the smallest positive
    alpha: the displayed two-sided alpha is 2 * alpha and the interval exists.
    Asked two-sided, that alpha has no representable tail and refuses."""
    tiny = np.nextafter(0.0, 1.0)
    rows, design = _rows(), _design(one_sided=True)
    one_sided = estimate_encouragement(
        [METRIC], rows, design, estimands=("late",), alpha=tiny, alternative="greater"
    )
    additive = [
        r for r in one_sided.results if r.estimand == "late" and r.value_scale == "absolute"
    ]
    assert additive
    lift = additive[0].require_lift()
    assert lift.alpha == 2.0 * tiny
    assert lift.lb is not None and lift.ub is not None
    assert np.isfinite(lift.lb) and np.isfinite(lift.ub)
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], rows, design, estimands=("late",), alpha=tiny)
    assert exc_info.value.code == "estimation.encouragement.alpha_eff_too"


def test_cluster_reference_rejects_prior_with_canonical_code():
    with pytest.raises(InvalidRequestError) as exc_info:
        _nn_estimate(
            point=0.1,
            se=0.2,
            prior=Normal(mu=0.0, sigma=0.1),
            alpha=0.05,
            dof=3.0,
        )
    assert exc_info.value.code == "estimation.encouragement.prior_cluster_robust"


@pytest.mark.parametrize(
    "estimands,kwargs",
    [
        (("late",), {}),
        (("itt", "late"), {}),
        (("itt",), {"prior": Normal(mu=0.0, sigma=0.1)}),
        (("itt",), {"cluster": "store"}),
        (("itt",), {"methods": [Method(name="sensitivity", variance_reduction="cuped")]}),
    ],
)
def test_unsupported_sequential_encouragement_refuses_before_summary_access(estimands, kwargs):
    def unread():
        raise AssertionError("unsupported route read summary")
        yield

    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            unread(),
            _design(one_sided=True),
            estimands=estimands,
            inference=_sequential_policy(),
            **kwargs,
        )
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.slow
def test_raw_itt_and_bernoulli_uptake_share_actual_joint_likelihood_prefix():
    from increment.estimation.results import LiftEstimate

    policy = _sequential_policy(uptake=True)
    snapshot = _sequential_snapshot(policy)
    bundle = estimate_encouragement(
        [METRIC],
        snapshot,
        _design(one_sided=False),
        estimands=("itt", "compliance"),
        inference=policy,
    )
    assert {r.estimand for r in bundle.results} == {"itt", "compliance"}
    assert all(r.stat_sig() for r in bundle.results)
    assert {r.require_sequential_result().checkpoint.prefix_id for r in bundle.results} == {
        snapshot.prefix_id
    }
    for row in bundle.results:
        restored = LiftEstimate.model_validate_json(row.model_dump_json())
        assert restored.require_sequential_result() == row.require_sequential_result()
        assert restored.stat_sig()
    uptake = next(r for r in bundle.results if r.estimand == "compliance")
    assert uptake.require_lift().value == pytest.approx(2.0)
    assert uptake.require_sequential_result().checkpoint.model.law == "bernoulli"
    assert uptake.require_sequential_result().checkpoint.control.n == 96


def test_registered_encouragement_preserves_empty_methods_and_exact_roster():
    policy = _sequential_policy(uptake=True)
    snapshot = _sequential_snapshot(policy)
    empty = estimate_encouragement(
        [METRIC],
        snapshot,
        _design(one_sided=False),
        estimands=("itt", "compliance"),
        methods=[],
        inference=policy,
    )
    assert empty.results == ()
    assert empty.evidence == {}

    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            snapshot,
            _design(one_sided=False),
            estimands=("itt",),
            inference=policy,
        )
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.slow
def test_one_sided_encouragement_keeps_raw_itt_supported():
    policy = _sequential_policy()
    bundle = estimate_encouragement(
        [METRIC],
        _sequential_snapshot(policy),
        _design(one_sided=True),
        estimands=("itt",),
        inference=policy,
    )
    assert bundle.results[0].stat_sig()
    assert bundle.results[0].estimand == "itt"


def test_structural_zero_uptake_cannot_be_relabelled_as_positive_control_relative_effect():
    policy = _sequential_policy(uptake=True)
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            _sequential_snapshot(policy),
            _design(one_sided=True),
            estimands=("itt", "compliance"),
            inference=policy,
        )
    assert raised.value.code == "sequential.route.unsupported"


def test_compliance_rate_row_carries_linear_scale():
    """The one-sided compliance-rate row is an absolute-scale quantity (an
    uptake rate with a symmetric linear interval), so it must carry
    scale='linear', otherwise decision stats back-transform its interval
    through log1p/expm1 and reject it as asymmetric."""
    res = estimate_encouragement([METRIC], _rows(one_sided=True), _design(one_sided=True)).results
    comp = [r for r in res if r.estimand == "compliance" and r.value_scale == "absolute"]
    assert comp and all(r.scale == "linear" for r in comp)


def test_rounded_encouragement_moments_cannot_resume_likelihood():
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(
            [METRIC],
            _rows(),
            _design(one_sided=True),
            estimands=("itt",),
            inference=_sequential_policy(),
        )
    assert raised.value.code == "sequential.source.invalid"


RETENTION = RetentionMetric(
    name="returning",
    entity="user_id",
    fact="returned",
    threshold_days=(1, 8),
)

RATIO = RatioMetric(
    name="rev",
    entity="user_id",
    numerator={"fact": "orders", "aggregation": "sum"},
    denominator={"fact": "sessions", "aggregation": "count"},
)


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    [
        (
            {"cluster": "store", "inference": _sequential_policy()},
            "sequential.route.unsupported",
        ),
        (
            {"cluster": "store", "prior": Normal(mu=0.0, sigma=0.1)},
            "arm.adjustment.cluster_prior",
        ),
        (
            {
                "cluster": "store",
                "methods": [Method(name="cuped", variance_reduction="cuped")],
            },
            "arm.adjustment.cluster_cuped",
        ),
        (
            {"cluster": "store"},
            "estimation.encouragement.cluster.ratio",
        ),
    ],
)
def test_retention_refusal_runs_after_cluster_precedence(kwargs, expected_code):
    metrics = [RETENTION, RATIO] if expected_code.endswith("ratio") else [RETENTION]

    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement(metrics, _rows(), _design(), **kwargs)

    assert raised.value.code == expected_code


def test_retention_refusal_runs_after_method_validation():
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement(
            [RETENTION],
            _rows(),
            _design(),
            methods=[Method(name="iptw")],
        )
    assert exc_info.value.code == "estimation.engine.method_name_observational"


def test_retention_refusal_has_stable_code_and_context():
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement([RETENTION], _rows(), _design())

    error = raised.value
    assert error.code == "readout.encouragement.retention"
    assert error.context["names"] == ("returning",)


def test_cluster_ratio_refusal_has_stable_code_and_context():
    with pytest.raises(CapabilityError) as raised:
        estimate_encouragement([RATIO], _rows(), _design(), cluster="store")

    error = raised.value
    assert error.code == "estimation.encouragement.cluster.ratio"
    assert error.context["names"] == ("rev",)
    assert error.context["cluster"] == "store"


def test_cluster_warning_is_attributed_to_public_caller():
    """Small-cluster guidance names the caller, not the estimator internals."""
    import warnings as _warnings

    from increment.semantics.design import ExclusionRestriction, UptakeSpec
    from tests.estimation.test_late_cluster import _arm_rows

    metric = MeanMetric(name="m", entity="user_id", fact="purchase", aggregation="sum")
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="took"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="test fixture",
        ),
        one_sided=True,
    )
    rng = np.random.default_rng(13)
    _, control = _arm_rows(rng, "control", k=5, m=2, treated=False)
    _, treatment = _arm_rows(rng, "treatment", k=5, m=2, treated=True)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        estimate_encouragement(
            [metric],
            [control, treatment],
            design,
            estimands=("late",),
            cluster="store",
        )

    codes = warning_codes(caught)
    cluster_warnings = [
        warning
        for warning, code in zip(caught, codes, strict=False)
        if code == "estimation.engine.small_total_clusters"
    ]
    assert len(cluster_warnings) == 1
    assert cluster_warnings[0].filename == __file__


def test_ratio_late_refusal_has_stable_code_and_context():
    with pytest.raises(UnsupportedRequestError) as raised:
        estimate_encouragement([RATIO], _rows(), _design(), estimands=("late",))

    error = raised.value
    assert error.code == "estimation.encouragement.late.ratio"
    assert error.context["metric"] == "rev"


def test_zero_variance_late_returns_typed_point_estimate_not_a_crash():
    """An exactly-zero additive LATE with a well-estimated first
    stage has se=0 from ``_late_additive``'s delta-method formula.
    ``normal_posterior``'s ``Normal`` requires sigma > 0, so this must
    degenerate to a point-only estimate instead of raising."""
    rows = [
        centered_row_from_raw_sums(
            {
                "experiment_id": "s",
                "metric": "rev",
                "group_id": "control",
                "n": 100,
                "sum_y": 1000.0,
                "sum_y2": 10000.0,
                "sum_d": 0.0,
                "sum_yd": 0.0,
                "sum_y2d": 0.0,
            }
        ),
        centered_row_from_raw_sums(
            {
                "experiment_id": "s",
                "metric": "rev",
                "group_id": "treat",
                "n": 100,
                "sum_y": 1000.0,
                "sum_y2": 10000.0,
                "sum_d": 50.0,
                "sum_yd": 500.0,
                "sum_y2d": 5000.0,
            }
        ),
    ]
    results = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), estimands=("late",)
    ).results
    additive = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
    assert additive, "a zero-variance LATE must still be reported, not abort the call"
    assert additive[0].require_lift().value == pytest.approx(0.0)
    assert additive[0].require_lift().lb is None and additive[0].require_lift().ub is None


def _perfect_compliance_rows(n=100, tau=2.0, seed=11):
    """Deterministic (per-arm-constant) uptake with noisy outcomes: the
    first stage is exactly known (Var(B)=0) while the LATE numerator
    still carries genuine variance."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid, d_val in (("control", 0.0), ("treat", 1.0)):
        d = np.full(n, d_val)
        y = 10.0 + tau * d + rng.normal(0, 2.0, size=n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    return rows


def test_perfect_compliance_first_stage_z_is_infinite_not_weak():
    """Var(B)=0 with b>0 (perfect, exactly-known compliance) must
    not be misclassified as a weak instrument via the z_fs=0.0 fallback
    -- LATE must still be emitted, with no weak_first_stage failure."""
    rows = _perfect_compliance_rows()
    computed = estimate_encouragement([METRIC], rows, _design(one_sided=True), estimands=("late",))
    late = [r for r in computed.results if r.estimand == "late" and r.value_scale == "absolute"]
    assert late, "perfect compliance must not suppress LATE as a weak instrument"
    assert late[0].require_lift().value == pytest.approx(2.0, abs=0.5)
    assert not any(
        failure.code == "estimation.encouragement.late.weak_first_stage"
        for failure in computed.failures.values()
    )


def test_exactly_zero_first_stage_stays_weak():
    """Counterpart to the perfect-compliance case: b == 0 exactly (no first
    stage at all) must keep the zero/weak fallback -- only a *positive*
    exactly-known b escapes the weak-instrument gate."""
    n = 100
    rows = [
        centered_row_from_raw_sums(
            {
                "experiment_id": "s",
                "metric": "rev",
                "group_id": gid,
                "n": n,
                "sum_y": 1000.0,
                "sum_y2": 10500.0,
                "sum_d": 0.0,
                "sum_yd": 0.0,
                "sum_y2d": 0.0,
            }
        )
        for gid in ("control", "treat")
    ]
    computed = estimate_encouragement([METRIC], rows, _design(one_sided=True), estimands=("late",))
    assert not [r for r in computed.results if r.estimand == "late"]
    weak = [
        failure
        for failure in computed.failures.values()
        if failure.code == "estimation.encouragement.late.weak_first_stage"
    ]
    assert len(weak) == 1
    assert weak[0].context["first_stage_z"] == 0.0


def test_clamp_variance_refuses_beyond_roundoff_clamps_within_it():
    """The shared LATE-variance clamp in encouragement.py must
    only absorb a negative value within floating-point noise of zero;
    a materially negative value (an infeasible/inconsistent moment set)
    must raise a coded refusal instead of silently reporting se=0."""
    from increment.errors import InvalidRequestError
    from increment.estimation.encouragement import _clamp_variance

    # O(1) terms, tiny roundoff-scale deficit: still clamps to 0.0.
    assert _clamp_variance(-1e-15, magnitude=1.0, n=200, what="LATE variance") == 0.0
    # O(1) terms, a materially negative deficit: refuses.
    with pytest.raises(InvalidRequestError) as raised:
        _clamp_variance(-1.0, magnitude=1.0, n=200, what="LATE variance")
    assert raised.value.code == "estimation.encouragement.negative_variance"


def _uptake_rows(n=100, sum_d_treat=17, tau=2.0, seed=3):
    """Deterministic uptake count (pins first-stage z exactly) with a
    noisy outcome (keeps the LATE variance well-defined)."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid, sum_d in (("control", 0), ("treat", sum_d_treat)):
        d = np.zeros(n)
        d[:sum_d] = 1.0
        y = 10.0 + tau * d + rng.normal(0, 2.0, size=n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": gid,
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    return rows


def test_late_row_carries_near_gate_caveat_note():
    """A LATE row whose observed first-stage z clears the emission
    gate only narrowly (within one z-unit) must carry a persisted
    non-confirmatory caveat on ``note``, not just a transient warning."""
    rows = _uptake_rows(sum_d_treat=17)  # z_fs ~= 4.50, gate=4.0, margin=1.0
    results = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), estimands=("late",)
    ).results
    additive = next(r for r in results if r.estimand == "late" and r.value_scale == "absolute")
    assert additive.note is not None
    assert "narrowly" in additive.note


def test_late_row_has_no_near_gate_caveat_well_above_gate():
    """Counterpart to the near-gate case: well clear of the gate, no
    non-confirmatory caveat is stamped."""
    rows = _uptake_rows(sum_d_treat=50)  # z_fs ~= 9.95, comfortably above gate+margin
    results = estimate_encouragement(
        [METRIC], rows, _design(one_sided=True), estimands=("late",)
    ).results
    additive = next(r for r in results if r.estimand == "late" and r.value_scale == "absolute")
    assert additive.note is not None
    assert "narrowly" not in additive.note


def _deterministic_arm_rows(*, treatment: tuple[float, float], control: tuple[float, float]):
    """Rows for constant-outcome/constant-uptake arms with perturbed seconds.

    *treatment* and *control* are each ``(cy2, cy2d)``: the offsets added to
    the arm's raw second-moment sums, which move the centered uptake-masked
    moments the relative-LATE variance clamp judges.
    """
    n = 100
    rows = []
    for group_id, y, d, (cy2, cy2d) in (
        ("control", 10.0, 0.0, control),
        ("treatment", 12.0, 1.0, treatment),
    ):
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": "rev",
                    "group_id": group_id,
                    "n": n,
                    "sum_y": y * n,
                    "sum_y2": y * y * n + cy2,
                    "sum_d": d * n,
                    "sum_yd": y * d * n,
                    "sum_y2d": y * y * d * n + cy2d,
                }
            )
        )
    return rows


class TestRelativeLateWithExactlyEstimableComplierMeans:
    """A positive complier mean with zero variance is known EXACTLY, so the
    precision gates must read it as infinite precision. Dividing by sqrt(0)
    instead raised ZeroDivisionError from a valid deterministic outcome, before
    the point-only branch could report it."""

    @staticmethod
    def _arm(group_id: str, *, y: float, d: float):
        from increment.estimation.armstats import ArmStats

        # Deterministic outcome and uptake: every unit identical, so every
        # variance and covariance is exactly zero.
        n = 100
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=y * n,
            sum_y2=y * y * n,
            sum_d=d * n,
            sum_yd=y * d * n,
            sum_y2d=y * y * d * n,
        )

    def test_a_deterministic_outcome_returns_a_zero_se_estimate(self):
        from increment.estimation.encouragement import _late_relative

        # Treated compliers earn 12, control never-takers 10; both exact.
        treatment = self._arm("treatment", y=12.0, d=1.0)
        control = self._arm("control", y=10.0, d=0.0)
        result = _late_relative(treatment, control)
        assert not isinstance(result, str), f"refused instead of estimating: {result}"
        theta, se_theta = result
        assert math.isfinite(theta)
        assert se_theta == 0.0


class TestRelativeLateVarianceUsesTheClampPolicy:
    """A variance deficit inside floating-point noise is cancellation and
    clamps to zero -- an exactly-known mean, which the precision gate reads as
    infinite precision. A material deficit is corrupt moments and raises this
    module's coded refusal. A bare sign test got both wrong in opposite
    directions: suppressing a valid estimate on roundoff, and downgrading
    corruption to a soft withheld note."""

    @staticmethod
    def _arm(group_id: str, *, y: float, d: float, cy2: float = 0.0, cy2d: float = 0.0):
        from increment.estimation.armstats import ArmStats

        n = 100
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=y * n,
            sum_y2=y * y * n + cy2,
            sum_d=d * n,
            sum_yd=y * d * n,
            sum_y2d=y * y * d * n + cy2d,
        )

    def test_a_deterministic_outcome_is_treated_as_exactly_known(self):
        from increment.estimation.encouragement import _late_relative

        result = _late_relative(
            self._arm("treatment", y=12.0, d=1.0), self._arm("control", y=10.0, d=0.0)
        )
        assert not isinstance(result, str), f"refused instead of estimating: {result}"
        theta, se_theta = result
        assert math.isfinite(theta)
        assert se_theta == 0.0

    def test_a_material_variance_deficit_raises_the_coded_refusal(self):
        from increment.errors import CodedError

        # cy2d far above cy2 drives var_y + var_yd - 2*cov_y_yd materially
        # negative: inconsistent moments, not cancellation.
        rows = _deterministic_arm_rows(treatment=(1.0, 1e6), control=(1.0, 0.0))
        with pytest.raises(CodedError) as raised:
            estimate_encouragement([METRIC], rows, _design(one_sided=True), estimands=("late",))
        assert raised.value.code == "estimation.encouragement.negative_variance"


class TestVarianceClampScaleAndArmIsolation:
    """The tolerance must come from the constituent term magnitudes, not the
    already-cancelled residual, and each arm must be judged on its own."""

    @staticmethod
    def _arm(group_id: str, *, y: float, d: float, cy2: float, cy2d: float):
        from increment.estimation.armstats import ArmStats

        n = 100
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=y * n,
            sum_y2=y * y * n + cy2,
            sum_d=d * n,
            sum_yd=y * d * n,
            sum_y2d=y * y * d * n + cy2d,
        )

    def test_an_offsetting_positive_arm_cannot_hide_a_material_deficit(self):
        """Summing first let a corrupt arm pass behind a healthy one."""
        from increment.errors import CodedError

        # The control arm's own term is positive by the same amount, so a
        # pooled sum would cancel to zero and never refuse.
        rows = _deterministic_arm_rows(treatment=(1.0, 1e6), control=(1e6, 0.0))
        with pytest.raises(CodedError) as raised:
            estimate_encouragement([METRIC], rows, _design(one_sided=True), estimands=("late",))
        assert raised.value.code == "estimation.encouragement.negative_variance"

    def test_a_deterministic_arm_is_still_accepted(self):
        from increment.estimation.encouragement import _late_relative

        result = _late_relative(
            self._arm("treatment", y=12.0, d=1.0, cy2=0.0, cy2d=0.0),
            self._arm("control", y=10.0, d=0.0, cy2=0.0, cy2d=0.0),
        )
        assert not isinstance(result, str), f"refused instead of estimating: {result}"
        assert result[1] == 0.0


class TestClampToleranceComesFromConstituentTerms:
    """The tolerance must scale with the magnitudes the cancellation happened
    at, not with the cancelled residual. Judging a residual against itself
    shrinks the tolerance by exactly the cancellation it exists to absorb, so a
    legitimate roundoff-scale deficit gets refused."""

    def test_a_roundoff_deficit_clamps_when_judged_against_the_terms(self):
        from increment.estimation.encouragement import _clamp_variance

        # var_y + var_yd - 2*cov_y_yd over terms of size ~1e6 cancelling to
        # -1e-9: that deficit is ~1e-15 RELATIVE, unambiguously rounding.
        assert (
            _clamp_variance(-1e-9, magnitude=4e6, n=100, what="constituent-scale magnitude") == 0.0
        )

    def test_the_same_deficit_is_refused_when_judged_against_itself(self):
        """The old behaviour, pinned so the difference is explicit: with the
        residual as its own magnitude there is no tolerance left to absorb it."""
        from increment.errors import CodedError
        from increment.estimation.encouragement import _clamp_variance

        with pytest.raises(CodedError) as raised:
            _clamp_variance(-1e-9, magnitude=1e-9, n=100, what="residual-scale magnitude")
        assert raised.value.code == "estimation.encouragement.negative_variance"

    def test_a_material_deficit_is_still_refused_at_the_larger_scale(self):
        from increment.errors import CodedError
        from increment.estimation.encouragement import _clamp_variance

        # 25% of the terms' own magnitude: not rounding at any scale.
        with pytest.raises(CodedError):
            _clamp_variance(-1e6, magnitude=4e6, n=100, what="material deficit")


class TestLateRelativePassesTheConstituentMagnitude:
    """The sibling tests pin what _clamp_variance DOES with a magnitude; this
    pins what _late_relative PASSES, which is where the bug was.

    Recording the arguments rather than inferring them from a result: the
    residual is the variance of ``y * (1 - d)``, so it is positive in general
    (the control fixture below shows that) and cancels to exactly zero under
    full uptake, where var_yd and cov_y_yd both equal var_y. Neither case makes
    the two candidate magnitudes disagree in the returned estimate, so the
    arguments are the only place the difference is observable -- and the
    full-uptake arm is what discriminates: its residual-based magnitude is 0
    against a constituent magnitude of 0.18."""

    def test_the_magnitude_is_the_sum_of_term_magnitudes_not_the_residual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from increment.estimation import encouragement as enc
        from increment.estimation.armstats import ArmStats

        recorded: list[tuple[float, float, str]] = []
        original = enc._clamp_variance

        def _recording(value: float, *, magnitude: float, n: int, what: str) -> float:
            recorded.append((value, magnitude, what))
            return original(value, magnitude=magnitude, n=n, what=what)

        monkeypatch.setattr(enc, "_clamp_variance", _recording)

        def arm(group_id: str, *, y: float, d: float, spread: float) -> ArmStats:
            n = 200
            return ArmStats.from_raw_sums(
                study_id="s",
                metric="m",
                group_id=group_id,
                n=n,
                sum_y=y * n,
                sum_y2=y * y * n + spread * (n - 1),
                sum_d=d * n,
                sum_yd=y * d * n,
                sum_y2d=y * y * d * n + spread * d * (n - 1),
            )

        t_arm = arm("treatment", y=12.0, d=1.0, spread=9.0)
        c_arm = arm("control", y=10.0, d=0.0, spread=4.0)
        enc._late_relative(t_arm, c_arm)

        assert recorded, "_clamp_variance was never called"
        control_calls = [r for r in recorded if "control mean variance" in r[2]]
        assert len(control_calls) == 2, "the control variance must be clamped per arm"
        for (value, magnitude, what), a in zip(control_calls, (t_arm, c_arm), strict=True):
            expected = (abs(a.var_y()) + abs(a.var_yd()) + 2 * abs(a.cov_y_yd())) / a.n
            assert magnitude == pytest.approx(expected, rel=1e-12), what
            # The point of the fix: the scale is the terms', not the residual's.
            assert magnitude >= abs(value)
            assert magnitude > 0.0


def _armstats_no_uptake(group_id: str) -> ArmStats:
    """An arm with no materialised uptake family at all -- ``cyd`` stays
    ``None``, the precondition ``_late_relative`` refuses on by name."""
    return ArmStats(
        study_id="s", metric="rev", group_id=group_id, n=100, ref_y=1.0, cy1=0.0, cy2=10.0
    )


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.encouragement.needs_uptake_cyd",
            lambda: _late_relative(_armstats_no_uptake("treat"), _armstats_no_uptake("control")),
        ),  # estimation/encouragement.py::_late_relative
        (
            "estimation.encouragement.unknown_estimand_supported",
            lambda: estimate_encouragement(
                [METRIC], _rows(), _design(), estimands=("itt", "as_treated")
            ),
        ),  # estimation/encouragement.py::_prepare_encouragement_estimation
        (
            "estimation.encouragement.alpha_eff_too",
            lambda: estimate_encouragement(
                [METRIC],
                _rows(),
                _design(one_sided=True),
                estimands=("late",),
                alpha=np.nextafter(0.0, 1.0),
            ),
        ),  # estimation/encouragement.py::_nn_estimate (two-sided tail underflow)
        (
            "estimation.encouragement.one_sided_encouragement",
            lambda: estimate_encouragement(
                [METRIC], _rows(one_sided=False), _design(one_sided=True)
            ),
        ),  # estimation/encouragement.py::_first_stage_context
    ],
)
def test_encouragement_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def test_encouragement_itt_conversion_keeps_binomial_reference():
    """Repro: identical conversion data through Randomized and
    through Encouragement (uptake attached) must produce the SAME
    reference_kind and the SAME interval for the ITT -- the uptake
    moment does not change the ITT's sufficient statistics."""
    import pandas as pd

    from increment._metric_specs import MetricSpec
    from increment.analysis import Analysis
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import ExclusionRestriction, UptakeSpec

    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {
            "unit_id": range(2 * n),
            "group_id": ["control"] * n + ["treatment"] * n,
            "uptake": np.concatenate([np.zeros(n), rng.binomial(1, 0.4, n)]),
            "converted": np.concatenate([rng.binomial(1, 0.10, n), rng.binomial(1, 0.14, n)]),
        }
    )
    metrics = [MetricSpec(name="converted", type="conversion")]
    [randomized] = Analysis.from_unit_summary(
        df, unit="unit_id", group="group_id", control="control", metrics=metrics
    ).run()
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="uptake does not gate the conversion outcome"
        ),
    )
    [encouragement] = Analysis.from_unit_summary(
        df, unit="unit_id", group="group_id", metrics=metrics, design=design, uptake="uptake"
    ).run(estimands=["itt"])
    assert isinstance(randomized, LiftEstimate)
    assert isinstance(encouragement, LiftEstimate)
    assert randomized.reference_kind == "binomial"
    assert encouragement.reference_kind == "binomial"
    assert encouragement.binomial_set == randomized.binomial_set
