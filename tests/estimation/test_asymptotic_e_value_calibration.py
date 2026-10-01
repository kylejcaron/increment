"""Asymptotic e-value exposure and universal e-BH selection for in-family
sequential secondaries. Validity is asymptotic (Waudby-Smith,
Arbour, Sinha, Kennedy & Ramdas 2024's normal-mixture martingale) composed
with e-BH's arbitrary-dependence FDR guarantee (Ramdas et al.) for an
asymptotic, not exact, stopped-FDR bound on scalar-mean cells."""

from fractions import Fraction

import numpy as np
import pytest

from increment.errors import CapabilityError
from increment.estimation._certified import log_interval
from increment.estimation._sequential_likelihood import GaussianState
from increment.estimation.asymptotic_mean import asymptotic_mean_set, count_boundary_log_e
from tests.asymptotic_cases import mean_model


def test_asymptotic_log_e_agrees_with_the_boundary_rejection_test_two_sided():
    """Duality holds for two-sided, where no sign-gating happens."""
    rho = Fraction(1, 10)
    declaration = mean_model("outcome", rho=rho, start_count=2)
    cases = [
        (50, 50, Fraction(3), Fraction(9), Fraction(20), Fraction(20)),
        (200, 200, Fraction(5), Fraction(52, 10), Fraction(40), Fraction(38)),
        (30, 40, Fraction(1), Fraction(1), Fraction(10), Fraction(12)),
        (10, 10, Fraction(0), Fraction(0), Fraction(5), Fraction(5)),
    ]
    alphas = (Fraction(1, 20), Fraction(1, 100), Fraction(1, 2))
    for n_c, n_t, mc, mt, sc, st in cases:
        control = GaussianState(n_c, (mc,), ((sc,),))
        treatment = GaussianState(n_t, (mt,), ((st,),))
        for alpha in alphas:
            bounds = asymptotic_mean_set(
                control,
                treatment,
                declaration=declaration,
                alpha=alpha,
                null_lift=Fraction(0),
                alternative="two-sided",
            )
            assert bounds.estimator_contrast is not None
            assert bounds.estimator_variance is not None
            log_e = count_boundary_log_e(
                count=bounds.count,
                rho=rho,
                estimator_contrast=bounds.estimator_contrast,
                estimator_variance=bounds.estimator_variance,
                alternative="two-sided",
            )
            assert (log_e > (-log_interval(alpha)).hi) == bounds.rejects(), (n_c, n_t, alpha)


@pytest.mark.parametrize("alternative,sign", [("greater", 1), ("less", -1)])
def test_asymptotic_log_e_gates_on_sign_without_the_flawed_doubling(alternative, sign):
    """A large effect in the WRONG direction is -inf, never a positive
    e-value; the CORRECT-sign e-value is the raw martingale value, with no
    doubling -- doubling a two-sided martingale value for one direction is
    not a valid e-value at a stopping time."""
    rho = Fraction(1, 10)
    two_sided_from_right = count_boundary_log_e(
        count=100,
        rho=rho,
        estimator_contrast=Fraction(9 * sign),
        estimator_variance=Fraction(20),
        alternative="two-sided",
    )
    wrong_log_e = count_boundary_log_e(
        count=100,
        rho=rho,
        estimator_contrast=Fraction(-9 * sign),
        estimator_variance=Fraction(20),
        alternative=alternative,
    )
    right_log_e = count_boundary_log_e(
        count=100,
        rho=rho,
        estimator_contrast=Fraction(9 * sign),
        estimator_variance=Fraction(20),
        alternative=alternative,
    )
    assert wrong_log_e == float("-inf")
    assert right_log_e == two_sided_from_right  # sign-gated raw value, no log(2) added


@pytest.mark.parametrize("alternative,sign", [("greater", 1), ("less", -1)])
def test_a_one_sided_family_member_rejects_exactly_when_its_e_value_does(alternative, sign):
    """A family member is selected on its sign-gated e-value, so its set -- at
    the registered level and reinverted at a wider one -- rejects exactly when
    that e-value exceeds 1/alpha. A set built at 2*alpha would also reject rows
    whose e-value lies in [1/(2*alpha), 1/alpha), which e-BH can never select."""
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential, evaluate_checkpoint
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_records, mean_registration

    q = Fraction(1, 10)
    cell = SequentialCell(
        metric="outcome", group_id="treatment", family=True, alpha=q / 2, alternative=alternative
    )
    reg = mean_registration(models=(mean_model("outcome", start_count=4),), cells=(cell,), q=q)
    policy = AsymptoticMean(registration=reg)
    control = [1, 2, 3, 4] * 25
    seen = set()
    for step in range(30, 62, 2):
        treatment = [c + sign * step / 100 for c in control]
        snapshot = mean_capture(reg, mean_records(control, treatment))
        row = estimate_sequential(snapshot, policy).results[0]
        checkpoint = row.require_asymptotic_sequential_result().checkpoint
        for alpha in (cell.alpha, q):
            result = evaluate_checkpoint(checkpoint, alpha=alpha, ceiling=q)
            e_rejects = result.log_e > (-log_interval(alpha)).hi
            assert result.rejects() == e_rejects, (step, alpha)
            seen.add((alpha, e_rejects))
    assert seen == {(a, r) for a in (cell.alpha, q) for r in (True, False)}


def test_asymptotic_e_value_mean_is_at_most_one_under_the_null_one_sided():
    """E[M_n * 1{sign}] <= 1 under the null even AT A DATA-DEPENDENT
    STOPPING TIME (first look whose sign happens to be correct), while
    E[2 * M_n * 1{sign}] is not -- reproducing the Decision record's own
    probe (E[2*M_n*1{S>0}] found well above 1 at a stopping time, unlike a
    single fixed-n look where sign is roughly independent of magnitude
    under the null and doubling stays under 1). This is exactly the hazard
    the earlier "doubled two-sided value" construction had: valid only at a
    FIXED n, not at a stopping time an anytime-valid process must support."""
    rho = Fraction(1, 10)
    rng = np.random.default_rng(0)
    looks = (20, 40, 60, 80, 100, 150, 200, 300, 400)
    e_at_stop = []
    doubled_at_stop = []
    for _ in range(3000):
        control_draws = rng.normal(0.0, 1.0, size=max(looks))
        treatment_draws = rng.normal(0.0, 1.0, size=max(looks))
        value = 0.0
        for n in looks:
            contrast = float(treatment_draws[:n].mean() - control_draws[:n].mean())
            if contrast <= 0:
                continue
            variance = float(
                treatment_draws[:n].var(ddof=0) / n + control_draws[:n].var(ddof=0) / n
            )
            log_e = count_boundary_log_e(
                count=n,
                rho=rho,
                estimator_contrast=contrast,
                estimator_variance=variance,
                alternative="greater",
            )
            value = 0.0 if log_e == float("-inf") else float(np.exp(min(float(log_e), 50.0)))
            break
        e_at_stop.append(value)
        doubled_at_stop.append(2 * value)
    mean_e = float(np.mean(e_at_stop))
    mean_doubled = float(np.mean(doubled_at_stop))
    assert mean_e <= 1.10  # generous Monte Carlo slack; the point is <= ~1, not <= ~2
    assert mean_doubled > 1.10  # doubling is not a valid e-value at a stopping time


def test_asymptotic_sequential_result_exposes_a_finite_log_e():
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential
    from tests.asymptotic_cases import mean_capture, mean_records, mean_registration

    reg = mean_registration()
    policy = AsymptoticMean(registration=reg)
    snapshot = mean_capture(reg, mean_records([1, 2] * 40, [6, 10] * 40))
    computation = estimate_sequential(snapshot, policy)
    result = computation.results[0].require_asymptotic_sequential_result()
    assert isinstance(result.log_e, (Fraction, float))
    assert result.log_e == result.log_e  # not NaN


def test_family_selection_is_e_bh_for_a_pure_asymptotic_family_now():
    """e-BH replaces the Bonferroni q/n split for EVERY family, not just
    mixed ones -- no toggle exists -- and every row shares one family-wide
    guarantee."""
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    cells = (
        SequentialCell(metric="outcome", group_id="treatment", family=True, alpha=Fraction(1, 20)),
        SequentialCell(metric="other", group_id="treatment", family=True, alpha=Fraction(1, 20)),
    )
    reg = mean_registration(
        models=(mean_model("outcome"), mean_model("other")), cells=cells, q=Fraction(1, 10)
    )
    policy = AsymptoticMean(registration=reg)
    snapshot = mean_capture(
        reg, mean_records([1, 2] * 40, [6, 10] * 40, metrics=("outcome", "other"))
    )
    computation = estimate_sequential(snapshot, policy)
    family = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
    ]
    outcome = select_sequential_family(family, reg.q, policy, 0.05, computation=computation)
    assert outcome.n_family == 2
    assert outcome.guarantee == "asymptotic_sequential"


@pytest.mark.parametrize("law", ["scalar_mean", "adjusted_mean", "ratio_mean"])
def test_registered_breakout_family_keeps_fixed_bonferroni_not_e_bh(law):
    """A breakout/segment family -- every cell carries a registered
    segment -- keeps its own fixed per-cell allocation with no
    reinversion, even though the SAME shape of evidence drives e-BH
    selection and reinversion for an ordinary metric/arm secondary
    family (test_family_selection_is_e_bh_for_a_pure_asymptotic_family_
    now, above). A breakout is registered with correction="bonferroni"
    (validate_breakout_registration) -- a real FWER promise over its
    segments; e-BH's FDR guarantee is a different, weaker promise and
    must not silently substitute for it. Every count-clock law shares the
    boundary and geometry, so the breakout validator and the family outcome
    agree for the joint laws as for the scalar one."""
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential
    from increment.semantics.sequential import SequentialCell
    from increment.sequential_source import validate_breakout_registration
    from tests.asymptotic_cases import mean_capture, mean_records, mean_registration

    q = Fraction(1, 10)
    cells = tuple(
        SequentialCell(
            metric="outcome",
            group_id="treatment",
            family=True,
            segment=(("segment", s),),
            alpha=Fraction(1, 40),
        )
        for s in ("present", "absent")
    )
    model = mean_model("outcome", start_count=4, law=law)
    reg = mean_registration(models=(model,), cells=cells, q=q)
    breakout = {
        "control_group": "control",
        "dimension": "segment",
        "q": 0.1,
        "alpha_by_metric": {"outcome": Fraction(1, 20)},
    }
    with pytest.raises(CapabilityError) as raised:
        validate_breakout_registration(reg, correction="bh", **breakout)
    assert raised.value.code == "sequential.route.unsupported"
    validate_breakout_registration(reg, correction="bonferroni", **breakout)

    policy = AsymptoticMean(registration=reg)
    control, treatment = [1, 2, 3, 4] * 30, [8, 10, 12, 14] * 30
    if law != "scalar_mean":
        second = [1, -1, -1, 1] * 30 if law == "adjusted_mean" else [2, 3, 2, 3] * 30
        control = list(zip(control, second, strict=True))
        treatment = list(zip(treatment, second, strict=True))
    records = mean_records(control, treatment, segment={"segment": "present"})
    snapshot = mean_capture(reg, records)
    computation = estimate_sequential(snapshot, policy)
    family = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
    ]
    outcome = select_sequential_family(family, reg.q, policy, 0.05, computation=computation)
    assert outcome.n_family == 2
    assert outcome.fcr_alpha is None
    assert outcome.realized_threshold is None
    assert not outcome.capped
    assert outcome.guarantee == "asymptotic_sequential"
    assert len(outcome.selected) == 1


def test_a_degenerate_cell_abstains_instead_of_aborting_the_family():
    """A member without usable evidence -- here zero observed variance --
    carries log evidence of minus infinity through ``estimate_sequential``:
    it stays in the family size and is never selected, while the rest of the
    family is still selected at the thresholds of the full roster."""
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import selected_snapshot_results
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    q = Fraction(1, 10)
    cells = (
        SequentialCell(metric="outcome", group_id="treatment", family=True, alpha=q / 2),
        SequentialCell(metric="other", group_id="treatment", family=True, alpha=q / 2),
    )
    reg = mean_registration(models=(mean_model("outcome"), mean_model("other")), cells=cells, q=q)
    records = mean_records([1, 2] * 40, [6, 10] * 40, metrics=("outcome", "other"))
    for record in records:
        record["values"]["other"] = 3.0
    rows = {
        row.metric: row
        for row in selected_snapshot_results(
            mean_capture(reg, records), AsymptoticMean(registration=reg), nominal_alpha=q / 2
        )
    }
    degenerate = rows["other"].require_asymptotic_sequential_result()
    assert degenerate.bounds.reason == "zero_arm_variance"
    assert degenerate.log_e == float("-inf")
    assert rows["other"].discovery is False
    assert rows["outcome"].discovery is True
    # The abstaining member still counts in m: one selection of two.
    assert rows["outcome"].family_threshold == pytest.approx(float(q / 2))


def test_a_recorded_failure_refuses_the_whole_family():
    """A family member with a recorded failure instead of evidence aborts
    selection -- even an outcome-degeneracy code the fixed-horizon route keeps
    as a non-rejection. Sequential degeneracy is carried by the evidence itself
    (log evidence of minus infinity), never by a recorded failure."""
    from increment.estimation.decision_types import (
        DecisionComputation,
        DecisionFailure,
        sequential_hypothesis_key,
    )
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    cells = (
        SequentialCell(metric="outcome", group_id="treatment", family=True, alpha=Fraction(1, 20)),
        SequentialCell(metric="other", group_id="treatment", family=True, alpha=Fraction(1, 20)),
    )
    reg = mean_registration(
        models=(mean_model("outcome"), mean_model("other")), cells=cells, q=Fraction(1, 10)
    )
    policy = AsymptoticMean(registration=reg)
    snapshot = mean_capture(
        reg, mean_records([1, 2] * 40, [6, 10] * 40, metrics=("outcome", "other"))
    )
    computation = estimate_sequential(snapshot, policy)
    failed_key = sequential_hypothesis_key(
        next(r for r in computation.results if r.metric == "other")
        .require_sequential_result()
        .checkpoint.cell
    )
    kept_evidence = {k: v for k, v in computation.evidence.items() if k != failed_key}
    degraded = DecisionComputation(
        results=computation.results,
        evidence=kept_evidence,
        failures={
            failed_key: DecisionFailure(
                hypothesis=failed_key,
                code="estimation.engine.lift_guard",
                context={"reason": "zero variance"},
            )
        },
        sequential_snapshot=computation.sequential_snapshot,
    )
    family = [
        (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
        for row in computation.results
    ]
    with pytest.raises(CapabilityError) as raised:
        select_sequential_family(family, reg.q, policy, 0.05, computation=degraded)
    assert raised.value.code == "family.evidence.incomplete"


def test_a_selected_row_between_the_two_thresholds_excludes_the_null_only_after_widening():
    """A row whose e-value lies strictly between m/(qR) and m/q excludes
    the null once reinverted at the widened fcr_alpha = min(q*R/m,
    nominal_alpha), even though the SAME checkpoint evaluated at the tight
    per-cell cell.alpha does not."""
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_result import AsymptoticSequentialResult
    from increment.estimation.sequential_runtime import (
        estimate_sequential,
        evaluate_checkpoint,
        reinvert_selected,
        result_from_sequential,
    )
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    q = Fraction(1, 10)
    m = 4  # a 4-cell family sized so cell.alpha = q/m; a 2-of-4 selection widens to q*2/m = 2*cell.alpha
    cell = SequentialCell(metric="outcome", group_id="treatment", family=True, alpha=q / m)
    reg = mean_registration(models=(mean_model("outcome", start_count=4),), cells=(cell,), q=q)
    policy = AsymptoticMean(registration=reg)
    control = [1, 2, 3, 4] * 30
    treatment = [c + 0.5 for c in control]
    snapshot = mean_capture(reg, mean_records(control, treatment))
    checkpoint = (
        estimate_sequential(snapshot, policy)
        .results[0]
        .require_asymptotic_sequential_result()
        .checkpoint
    )
    fcr_alpha = 2 * cell.alpha  # q * R / m for R=2, m=4
    tight = evaluate_checkpoint(checkpoint, alpha=cell.alpha)
    widened = evaluate_checkpoint(checkpoint, alpha=fcr_alpha, ceiling=fcr_alpha)
    assert isinstance(tight, AsymptoticSequentialResult)
    assert isinstance(widened, AsymptoticSequentialResult)
    assert not tight.bounds.rejects(), (
        "fixture must NOT exclude the null at the tight per-cell alpha"
    )
    assert widened.bounds.rejects(), "fixture must exclude the null once widened to fcr_alpha"
    row = result_from_sequential(tight, label=policy.label)
    reinverted = reinvert_selected(row, fcr_alpha, ceiling=fcr_alpha)
    assert reinverted.require_asymptotic_sequential_result().decision_alpha == fcr_alpha
    assert reinverted.require_asymptotic_sequential_result().bounds.rejects()


def test_an_unselected_row_with_a_widened_decision_alpha_is_refused():
    """A row that is NOT a family discovery cannot carry a decision alpha
    past its registered cell allocation, even if its own alpha_ceiling
    witness claims otherwise -- alpha_ceiling alone is not sufficient
    authority; discovery/family_threshold is."""
    from increment.estimation.results import LiftEstimate
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import (
        estimate_sequential,
        evaluate_checkpoint,
        result_from_sequential,
    )
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    cell = SequentialCell(
        metric="outcome", group_id="treatment", family=True, alpha=Fraction(1, 40)
    )
    reg = mean_registration(
        models=(mean_model("outcome", start_count=4),), cells=(cell,), q=Fraction(1, 10)
    )
    policy = AsymptoticMean(registration=reg)
    snapshot = mean_capture(reg, mean_records([1, 2, 3, 4] * 30, [4.5, 6.0, 7.5, 9.0] * 30))
    checkpoint = (
        estimate_sequential(snapshot, policy)
        .results[0]
        .require_asymptotic_sequential_result()
        .checkpoint
    )
    widened = evaluate_checkpoint(checkpoint, alpha=Fraction(1, 20), ceiling=Fraction(1, 20))
    forged = result_from_sequential(
        widened, label=policy.label, discovery=True, family_threshold=1.0, family_nominal_alpha=1.0
    ).model_copy(update={"discovery": False, "family_threshold": None})
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


def test_a_selected_row_claiming_an_alpha_above_nominal_is_refused():
    """Main's final ruling: family_threshold alone is not enough to bind
    decision_alpha -- it must also never exceed the family's own
    nominal_alpha, the OTHER arm of fcr_alpha = min(realized, nominal_alpha)."""
    from increment.estimation.results import LiftEstimate
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import (
        estimate_sequential,
        evaluate_checkpoint,
        result_from_sequential,
    )
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration

    cell = SequentialCell(
        metric="outcome", group_id="treatment", family=True, alpha=Fraction(1, 40)
    )
    reg = mean_registration(
        models=(mean_model("outcome", start_count=4),), cells=(cell,), q=Fraction(1, 10)
    )
    policy = AsymptoticMean(registration=reg)
    snapshot = mean_capture(reg, mean_records([1, 2, 3, 4] * 30, [4.5, 6.0, 7.5, 9.0] * 30))
    checkpoint = (
        estimate_sequential(snapshot, policy)
        .results[0]
        .require_asymptotic_sequential_result()
        .checkpoint
    )
    # decision_alpha=0.10 is <= a generous family_threshold=0.5 (the OLD,
    # insufficient check would have accepted this) but > nominal_alpha=0.05.
    widened = evaluate_checkpoint(checkpoint, alpha=Fraction(1, 10), ceiling=Fraction(1, 10))
    forged = result_from_sequential(
        widened, label=policy.label, discovery=True, family_threshold=1.0, family_nominal_alpha=1.0
    ).model_copy(update={"discovery": True, "family_threshold": 0.5, "family_nominal_alpha": 0.05})
    with pytest.raises(CapabilityError) as raised:
        LiftEstimate.model_validate(forged.model_dump())
    assert raised.value.code == "sequential.source.invalid"


# Units per arm revealed at each look: six looks, 40 to 200 units per arm.
ASYMPTOTIC_STOPPED_SCHEDULE = (40, 32, 32, 32, 32, 32)


def _asymptotic_stopped_outcome(seed: int, *, lift: float = 0.0):
    """One replication through the deployed asymptotic family route.

    Appends one joint batch per look with ``mean_capture``, evaluates each
    prefix with ``estimate_sequential``, selects with
    ``select_sequential_family``, and stops at the first look that selects,
    so the returned outcome is a genuine stopped state. Every arm draws
    Normal(10, 2); ``lift`` shifts the treatment mean of ``outcome`` only.
    """
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from increment.estimation.sequential import AsymptoticMean
    from increment.estimation.sequential_runtime import estimate_sequential
    from increment.semantics.sequential import SequentialCell
    from tests.asymptotic_cases import mean_capture, mean_registration

    cells = tuple(
        SequentialCell(metric=metric, group_id="treatment", family=True, alpha=Fraction(1, 20))
        for metric in ("outcome", "other")
    )
    reg = mean_registration(
        models=(mean_model("outcome"), mean_model("other")), cells=cells, q=Fraction(1, 10)
    )
    policy = AsymptoticMean(registration=reg)
    rng = np.random.default_rng(seed)
    snapshot = outcome = None
    offset = 0
    for batch in ASYMPTOTIC_STOPPED_SCHEDULE:
        draws = rng.normal(10.0, 2.0, size=(batch, 2, 2))
        rows = [
            {
                "unit_id": f"{offset + unit:06d}-{arm}",
                "group_id": arm,
                "values": {
                    "outcome": float(draws[unit, index, 0]) + (lift if index else 0.0),
                    "other": float(draws[unit, index, 1]),
                },
            }
            for unit in range(batch)
            for index, arm in enumerate(("control", "treatment"))
        ]
        offset += batch
        snapshot = mean_capture(reg, rows, previous=snapshot, append=snapshot is not None)
        computation = estimate_sequential(snapshot, policy)
        family = [
            (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
            for row in computation.results
        ]
        outcome = select_sequential_family(
            family, reg.q, policy, Fraction(1, 20), computation=computation
        )
        if outcome.selected:
            break
    return outcome


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_asymptotic_family_e_bh_stopped_fdr_is_bounded_under_the_global_null():
    """Calibrated stopped FDR for asymptotic e-BH under the global null:
    six appended looks per replication, stopping at the first look that
    selects. Replications, error budget and acceptance are those of
    test_deployed_sequential_family_stopped_fdr_is_bounded_under_the_global_null
    (tests/estimation/test_family_coverage.py). Every cell is a true null, so
    a replication's stopped false discovery proportion is 1 exactly when the
    stopped family selects anything, and the prospective bound is
    ``CP_U(discoveries, reps, STOPPED_FDR_ETA) <= Q + STOPPED_FDR_DELTA``.
    """
    from tests.estimation.test_family_coverage import (
        STOPPED_FDR_DELTA,
        STOPPED_FDR_ETA,
        STOPPED_FDR_REPS,
        Q,
    )
    from tests.mc import binomial_error_upper_bound

    outcomes = [_asymptotic_stopped_outcome(seed) for seed in range(9500, 9500 + STOPPED_FDR_REPS)]
    assert all((outcome.fcr_alpha is None) == (not outcome.selected) for outcome in outcomes)
    discoveries = sum(bool(outcome.selected) for outcome in outcomes)
    upper = binomial_error_upper_bound(discoveries, len(outcomes), STOPPED_FDR_ETA)
    assert upper <= Q + STOPPED_FDR_DELTA, (discoveries, len(outcomes), upper)


def test_asymptotic_family_e_bh_stops_on_discovery_smoke():
    """Fast-tier witness for the stopped route above: a null replication
    selects nothing, and a lifted one stops on a discovery of exactly the
    lifted metric, reinverted at q * R / m over the full two-cell roster."""
    null = _asymptotic_stopped_outcome(404)
    assert null.n_family == 2
    assert not null.selected
    assert null.fcr_alpha is None
    lifted = _asymptotic_stopped_outcome(505, lift=2.0)
    assert {key.metric for key in lifted.selected} == {"outcome"}
    assert lifted.fcr_alpha == Fraction(1, 10) * len(lifted.selected) / 2
