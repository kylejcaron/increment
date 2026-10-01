"""Joint asymptotic laws: adjusted_mean, ratio_mean and adjusted_ratio_mean."""

from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

from increment import (
    Analysis,
    AsymptoticMean,
    Method,
    SequentialCell,
    declare_sequential_freeze,
    estimate_sequential,
)
from increment.errors import CodedError
from increment.estimation._certified import log_interval
from increment.estimation._sequential_likelihood import GaussianState
from increment.estimation.asymptotic_joint import (
    JointPreparation,
    _coefficient,
    _form,
    _quadratic,
    _variance_forms,
    asymptotic_joint_set,
    invert_joint,
    linearise,
    prepare_joint,
)
from increment.estimation.asymptotic_mean import (
    MeanSetComponent,
    _component_for_quadratic,
    asymptotic_mean_set,
    count_boundary,
)
from increment.estimation.sequential_result import AsymptoticSequentialResult
from increment.estimation.sequential_runtime import evaluate_checkpoint, selected_snapshot_results
from increment.frame import MetricSpec
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, InferenceSpec
from tests.asymptotic_cases import mean_capture, mean_model, mean_records, mean_registration
from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

_DESIGN = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
_CUPED = Method(name="cuped", variance_reduction="cuped")
_PLAN = AnalysisPlan(primary="m", inference=InferenceSpec(kind="asymptotic_mean"))
ALPHA = 0.05


def _frame(rng, n=600, lift=0.5):
    arms = np.array(["control", "treatment"] * (n // 2))
    x = rng.normal(10.0, 3.0, size=n)
    return pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(n)],
            "arm": arms,
            "exposure": list(range(n)),
            "m": 5.0 + 0.9 * x + rng.normal(0.0, 1.0, size=n) + lift * (arms == "treatment"),
            "x": x,
            "noise": rng.normal(0.0, 1.0, size=n),
            "d": 4.0 + 0.3 * x + rng.normal(0.0, 0.5, size=n),
        }
    )


def _run(frame, spec):
    frame = frame.copy()
    if "exposure" not in frame:
        frame["exposure"] = range(len(frame))
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[spec],
        experiment_id="e",
        plan=_PLAN,
        design=_DESIGN,
        exposure_date="exposure",
    )
    return analysis, analysis.run()[0]


def _width(row):
    return row.lift.ub - row.lift.lb


def test_adjusted_mean_narrows_with_a_predictive_covariate_and_not_with_noise():
    frame = _frame(np.random.default_rng(11))
    _, plain = _run(frame, MetricSpec(name="m", type="mean"))
    analysis, adjusted = _run(
        frame, MetricSpec(name="m", type="mean", covariate="x", decision_method=_CUPED)
    )
    _, noise = _run(
        frame, MetricSpec(name="m", type="mean", covariate="noise", decision_method=_CUPED)
    )
    model = analysis.sequential_snapshot().registration.models[0]
    assert model.law == "adjusted_mean" and adjusted.method == "cuped"
    assert _width(adjusted) < 0.5 * _width(plain)
    assert abs(_width(noise) - _width(plain)) < 0.01 * _width(plain)


def test_adjusted_mean_with_an_orthogonal_covariate_is_the_scalar_set():
    """A covariate whose within-arm cross moment is zero fits theta = 0 exactly,
    and the joint inversion collapses to the scalar route's quadratic."""
    control = [[F(y), F(x)] for y, x in zip([1, 2, 3, 4] * 20, [1, -1, -1, 1] * 20, strict=True)]
    treatment = [[F(y), F(x)] for y, x in zip([4, 6, 8, 10] * 20, [1, -1, -1, 1] * 20, strict=True)]
    joint = asymptotic_joint_set(
        GaussianState.from_rows(control, dimension=2),
        GaussianState.from_rows(treatment, dimension=2),
        declaration=mean_model("m", law="adjusted_mean"),
        alpha=F(1, 20),
        null_lift=F(0),
        alternative="two-sided",
    )
    scalar = asymptotic_mean_set(
        GaussianState.from_rows([[r[0]] for r in control], dimension=1),
        GaussianState.from_rows([[r[0]] for r in treatment], dimension=1),
        declaration=mean_model(metric="m"),
        alpha=F(1, 20),
        null_lift=F(0),
        alternative="two-sided",
    )
    assert joint == scalar


def test_ratio_metric_is_monitored_sequentially_and_an_unresolved_denominator_withholds_it():
    rng = np.random.default_rng(3)
    frame = _frame(rng)
    analysis, row = _run(frame, MetricSpec(name="m", type="ratio", numerator="m", denominator="d"))
    assert analysis.sequential_snapshot().registration.models[0].law == "ratio_mean"
    assert row.inference == "asymptotic_mean" and row.lift.lb < row.lift.value < row.lift.ub
    assert row.require_asymptotic_sequential_result().bounds.status == "bounded"
    near_zero = frame.assign(d=rng.normal(0.0, 1.0, size=len(frame)))
    _, unresolved = _run(
        near_zero, MetricSpec(name="m", type="ratio", numerator="m", denominator="d")
    )
    result = unresolved.require_asymptotic_sequential_result()
    assert result.bounds.status == "unavailable"
    assert result.bounds.reason == "denominator_near_zero"
    # No decision at this alpha, and evidence capped by the same denominators
    # cannot clear it either.
    assert not result.rejects()
    assert result.log_e < (-log_interval(result.decision_alpha)).hi


def test_ratio_law_registration_requires_the_denominator_declaration():
    from increment import ScalarMeanModel

    payload = mean_model("m", law="ratio_mean").model_dump()
    with pytest.raises(CodedError) as raised:
        ScalarMeanModel.model_validate({**payload, "positive_population_denominators": False})
    assert raised.value.code == "sequential.registration.invalid"


def _evaluate_ratio(control, treatment, *, alpha=F(1, 20), alternative="two-sided", null_lift=F(0)):
    """One ratio_mean cell over per-unit (N, D) rows, read back through the runtime."""
    registration = mean_registration(
        models=(mean_model("m", law="ratio_mean"),),
        cells=(
            SequentialCell(
                metric="m",
                group_id="treatment",
                alpha=alpha,
                alternative=alternative,
                null_lift=null_lift,
            ),
        ),
    )
    snapshot = mean_capture(registration, mean_records(control, treatment, metrics=("m",)))
    return estimate_sequential(snapshot, AsymptoticMean(registration=registration)).results[0]


def _covers(bounds, r):
    return any(
        (c.lower is None or c.lower <= r) and (c.upper is None or r <= c.upper)
        for c in bounds.components
    )


# Control (N, D) means (2, 7/3) and treatment (10, 7/3): the observed ratio
# contrast is 5, with varying denominators so the linearisation's cross term
# is live rather than a scalar mean in disguise.
_RATIO_CONTROL = [(1, 2), (2, 3), (3, 2)] * 40
_RATIO_TREATMENT = [(8, 2), (10, 3), (12, 2)] * 40


def test_joint_result_replay_rejects_a_contradictory_construction():
    from increment.errors import CodedError
    from increment.estimation.sequential_result import AsymptoticSequentialResult

    result = _evaluate_ratio(
        _RATIO_CONTROL, _RATIO_TREATMENT
    ).require_asymptotic_sequential_result()
    payload = result.model_dump(mode="json")
    assert payload["construction"] == "linearised_shifted_contrast_v1"
    restored = AsymptoticSequentialResult.model_validate(payload)
    assert restored.bounds == result.bounds and restored.log_e == result.log_e
    payload["construction"] = "direct_shifted_contrast_v1"
    with pytest.raises(CodedError) as error:
        AsymptoticSequentialResult.model_validate(payload)
    assert error.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_ratio_directional_boundary_spends_its_alpha_on_one_tail(alternative):
    row = _evaluate_ratio(_RATIO_CONTROL, _RATIO_TREATMENT, alternative=alternative)
    bounds = row.require_asymptotic_sequential_result().bounds
    assert bounds.alpha == F(1, 20)
    boundary_alpha = F(1, 20) if alternative == "two-sided" else F(1, 10)
    assert bounds.k == count_boundary(240, boundary_alpha, F(1, 10))
    assert row.stat_sig() == (alternative != "less")
    assert row.stat_sig() == (not _covers(bounds, F(1)))
    if alternative == "two-sided":
        assert bounds.status == "bounded"
    # A one-sided set always contains the half-line past the observed contrast.
    if alternative == "greater":
        assert bounds.status == "ray" and bounds.upper is None
        assert bounds.lower is not None and bounds.lower <= 5
    if alternative == "less":
        assert bounds.status == "ray" and bounds.lower is None
        assert bounds.upper is not None and bounds.upper >= 5


@pytest.mark.parametrize("alternative,endpoint", [("greater", "lower"), ("less", "upper")])
def test_ratio_one_sided_endpoint_equals_the_two_sided_endpoint_at_twice_alpha(
    alternative, endpoint
):
    one = _evaluate_ratio(_RATIO_CONTROL, _RATIO_TREATMENT, alpha=F(1, 20), alternative=alternative)
    two = _evaluate_ratio(_RATIO_CONTROL, _RATIO_TREATMENT, alpha=F(1, 10))
    one_bounds = one.require_asymptotic_sequential_result().bounds
    two_bounds = two.require_asymptotic_sequential_result().bounds
    assert getattr(one_bounds, endpoint) == getattr(two_bounds, endpoint)
    assert one_bounds.status == "ray" and two_bounds.status == "bounded"


@pytest.mark.parametrize("alpha", [F(1, 2), F(3, 5)])
def test_ratio_one_sided_alpha_at_or_above_one_half_refuses_before_geometry(alpha):
    # The control denominator mean is exactly zero, which leaves the cell
    # unavailable on its own; the alpha domain is checked before any geometry.
    control = [(1, -1), (2, 1), (3, 0)] * 40
    unresolved = _evaluate_ratio(control, _RATIO_TREATMENT)
    bounds = unresolved.require_asymptotic_sequential_result().bounds
    assert bounds.status == "unavailable" and bounds.reason == "zero_denominator_mean"
    with pytest.raises(CodedError) as raised:
        _evaluate_ratio(control, _RATIO_TREATMENT, alpha=alpha, alternative="greater")
    assert raised.value.code == "sequential.asymptotic_mean.invalid"


def test_an_unresolved_ratio_cell_withholds_only_itself_across_a_freeze():
    """A ratio look whose denominator is not yet separated from zero leaves
    that cell alone without a decision: its scalar sibling is still evaluated
    and selected over the full two-cell family, freezing the ratio cell keeps
    it decision-free, and later looks evaluate the sibling at its new state."""
    q = F(1, 10)
    registration = mean_registration(
        models=(mean_model("m", law="ratio_mean"), mean_model("s")),
        cells=tuple(
            SequentialCell(metric=m, group_id="treatment", family=True, alpha=q / 2)
            for m in ("m", "s")
        ),
        q=q,
    )
    policy = AsymptoticMean(registration=registration)

    def records(n, denominators, *, offset=0):
        rows = []
        for i in range(n):
            y = 1 + i % 4
            for arm, shift in (("control", 0), ("treatment", 7)):
                values = {"m": (y, denominators[i % 4]), "s": y + shift}
                rows.append(
                    {"unit_id": f"{offset + i:06d}-{arm}", "group_id": arm, "values": values}
                )
        return rows

    def family(snapshot):
        rows = selected_snapshot_results(snapshot, policy, nominal_alpha=q / 2)
        return {row.metric: row for row in rows}

    early = mean_capture(registration, records(20, (1, -1, 2, -1)))
    before = family(early)
    ratio = before["m"].require_asymptotic_sequential_result()
    assert ratio.bounds.reason == "denominator_near_zero" and not before["m"].discovery
    assert before["s"].discovery
    assert before["s"].family_threshold == pytest.approx(float(q / 2))  # R=1 of m=2

    frozen = declare_sequential_freeze(early, ["m"])
    later = mean_capture(
        registration, records(200, (4, 5, 6, 5), offset=20), previous=frozen, append=True
    )
    after = family(later)
    frozen_ratio = after["m"].require_asymptotic_sequential_result()
    assert frozen_ratio.checkpoint.status == "frozen"
    assert frozen_ratio.bounds == ratio.bounds and not after["m"].discovery
    scalar = after["s"].require_asymptotic_sequential_result()
    assert scalar.checkpoint.status == "current" and scalar.checkpoint.revealed_units == 440
    assert after["s"].discovery


# The ``_f7_*`` fixture: a 20-unit-per-arm ratio whose denominator is 1 for 11
# units and 0 for 9. The denominator statistic over 40 units is 220/9 ~ 24.4, resolved at
# 1/20 (boundary ~22.1) and 9/200 (~22.9) but not 1/40 (~27.0) or 1/200 (~38.3);
# contrast evidence is ~345. The scalar sibling is null unless its treatment arm shifts.
_F7_Q = F(1, 20)
_F7_NOISE = (F(-1, 10), F(1, 10)) * 5 + (F(0),)


def _f7_registration(ratio_alpha=F(9, 200), scalar_alpha=F(1, 200)):
    return mean_registration(
        models=(mean_model("ratio", law="ratio_mean"), mean_model("scalar")),
        cells=(
            SequentialCell(metric="ratio", group_id="treatment", family=True, alpha=ratio_alpha),
            SequentialCell(metric="scalar", group_id="treatment", family=True, alpha=scalar_alpha),
        ),
        q=_F7_Q,
    )


def _f7_records(n=20, *, shift=0, offset=0):
    rows = []
    for i in range(n):
        d = F(i < 11)
        for arm, slope, lift in (("control", 2, 0), ("treatment", 4, shift)):
            numerator = slope * d + (_F7_NOISE[i] if i < 11 else 0)
            values = {"ratio": (numerator, d), "scalar": 1 + i % 4 + lift}
            rows.append({"unit_id": f"{offset + i:06d}-{arm}", "group_id": arm, "values": values})
    return rows


def _family(registration, snapshot, *, nominal=_F7_Q):
    policy = AsymptoticMean(registration=registration)
    rows = selected_snapshot_results(snapshot, policy, nominal_alpha=nominal)
    return {row.metric: row for row in rows}


def test_a_selected_ratio_is_available_at_its_reporting_alpha():
    """Selection reads denominator-capped evidence, so a selected ratio's set is
    available at the level it is reported at, whatever its registered
    allocation: alone, the ``_f7_registration`` ratio would be reported at 1/40 and is not
    selected; beside a strong sibling both are reported at 1/20, including
    when the ratio's own 1/200 allocation leaves its registered set unavailable."""
    registration = _f7_registration()
    alone = _family(registration, mean_capture(registration, _f7_records()))
    assert not alone["ratio"].discovery and not alone["scalar"].discovery
    registered = alone["ratio"].require_asymptotic_sequential_result()
    assert registered.bounds.available and registered.rejects()

    strong = _family(registration, mean_capture(registration, _f7_records(shift=7)))
    tight = _f7_registration(ratio_alpha=F(1, 200), scalar_alpha=F(9, 200))
    snapshot = mean_capture(tight, _f7_records(shift=7))
    before = estimate_sequential(snapshot, AsymptoticMean(registration=tight)).results[0]
    assert before.metric == "ratio" and before.note == "denominator_near_zero"
    recovered = _family(tight, snapshot)
    for rows in (strong, recovered):
        assert rows["ratio"].discovery and rows["scalar"].discovery
        assert rows["ratio"].family_threshold == pytest.approx(float(_F7_Q))  # R=2 of m=2
        result = rows["ratio"].require_asymptotic_sequential_result()
        assert result.decision_alpha == _F7_Q and result.bounds.available
        assert rows["ratio"].note is None
    stopped = before.require_asymptotic_sequential_result()
    result = recovered["ratio"].require_asymptotic_sequential_result()
    assert result.checkpoint == stopped.checkpoint and result.log_e == stopped.log_e


@pytest.mark.parametrize("nominal", [F(1, 40), F(1, 200)])
def test_a_ratio_unresolved_at_the_nominal_alpha_stays_in_the_family_unselected(nominal):
    """A reporting level of min(q*R/m, nominal) can fall below the e-BH level:
    a ratio unresolved at the nominal level abstains before selection, still
    counted in m, and its sibling is selected at the recomputed threshold."""
    registration = _f7_registration()
    rows = _family(registration, mean_capture(registration, _f7_records(shift=7)), nominal=nominal)
    assert not rows["ratio"].discovery and rows["scalar"].discovery
    assert rows["scalar"].family_threshold == pytest.approx(1 / 40)  # R=1 of m=2
    scalar = rows["scalar"].require_asymptotic_sequential_result()
    assert scalar.decision_alpha == min(F(1, 40), nominal) and scalar.rejects()
    assert rows["ratio"].require_asymptotic_sequential_result().decision_alpha == F(9, 200)


def test_ratio_evidence_reads_the_stopped_state_not_the_inversion_alpha():
    """One checkpoint inverted at three levels keeps one capped evidence value:
    it clears -log(1/20), where the denominators are resolved, and not
    -log(1/40), where they are not, though the raw contrast clears both."""
    registration = _f7_registration()
    snapshot = mean_capture(registration, _f7_records())
    row = estimate_sequential(snapshot, AsymptoticMean(registration=registration)).results[0]
    checkpoint = row.require_asymptotic_sequential_result().checkpoint
    results = [
        evaluate_checkpoint(checkpoint, alpha=alpha, ceiling=_F7_Q)
        for alpha in (F(1, 200), F(1, 40), _F7_Q)
    ]
    for result, available in zip(results, (False, False, True), strict=True):
        assert isinstance(result, AsymptoticSequentialResult)
        assert result.bounds.available == available
    assert len({result.log_e for result in results}) == 1
    log_e = results[0].log_e
    assert (-log_interval(_F7_Q)).hi <= log_e < (-log_interval(_F7_Q / 2)).hi


def test_a_frozen_ratio_keeps_its_evidence_and_a_later_family_can_select_it():
    registration = _f7_registration()
    early = mean_capture(registration, _f7_records())
    stopped = _family(registration, early)["ratio"].require_asymptotic_sequential_result()
    frozen = declare_sequential_freeze(early, ["ratio"])
    later = mean_capture(
        registration, _f7_records(40, shift=7, offset=20), previous=frozen, append=True
    )
    rows = _family(registration, later)
    ratio = rows["ratio"].require_asymptotic_sequential_result()
    assert ratio.checkpoint.status == "frozen"
    assert ratio.checkpoint.control.n + ratio.checkpoint.treatment.n == 40
    assert ratio.log_e == stopped.log_e
    assert rows["ratio"].discovery and rows["scalar"].discovery
    assert ratio.decision_alpha == _F7_Q and ratio.bounds.available


def test_adjusted_ratio_family_evidence_uses_the_adjusted_denominator():
    """Family evidence clears only after the pooled-anchor denominator guard."""
    q = F(1, 50)
    registration = mean_registration(
        models=(mean_model("ratio", law="adjusted_ratio_mean"), mean_model("scalar")),
        cells=tuple(
            SequentialCell(metric=m, group_id="treatment", family=True, alpha=F(1, 100))
            for m in ("ratio", "scalar")
        ),
        q=q,
    )
    covariate = (-1, 1) * 5 + (0,) + (-1, 1) * 4 + (0,)
    noise = (F(-1, 10), F(-1, 10), F(1, 10), F(1, 10)) * 2
    records = []
    for i, x in enumerate(covariate):
        d = (i < 11) + F(1, 10) + F(x, 10)
        for arm, slope in (("control", 2), ("treatment", 4)):
            numerator = slope * d + (noise[i] if i < 8 else 0) + F(x, 5)
            values = {"ratio": (numerator, d, x), "scalar": 1 + i % 4}
            records.append({"unit_id": f"{i:06d}-{arm}", "group_id": arm, "values": values})
    rows = _family(registration, mean_capture(registration, records), nominal=q)
    assert rows["ratio"].discovery and not rows["scalar"].discovery
    ratio = rows["ratio"].require_asymptotic_sequential_result()
    assert ratio.decision_alpha == F(1, 100) and ratio.bounds.available


def test_adjusted_ratio_stability_includes_shared_anchor_variance():
    """The adjusted denominator includes the pooled X anchor in both arms."""
    c = GaussianState.from_rows(
        [(1 + F(i % 4, 10), int(i < 2) + F(i % 2, 100), int(i < 2)) for i in range(20)],
        dimension=3,
    )
    t = GaussianState.from_rows(
        [(3 + F(i % 4, 10), int(i < 2) + F(i % 2, 100), int(i < 2)) for i in range(20)],
        dimension=3,
    )
    declaration = mean_model(law="adjusted_ratio_mean")
    linearisation, reason = linearise(c, t, declaration)
    assert linearisation is not None and reason is None
    prepared = prepare_joint(c, t, declaration=declaration, null_lift=F(0))
    control_denominator = linearisation.control.denominator
    treatment_denominator = linearisation.treatment.denominator
    assert control_denominator is not None and treatment_denominator is not None
    denominator_variance = _form(c, control_denominator[1], control_denominator[1])
    assert control_denominator[1] == (F(0), F(1), F(-1, 2))
    assert control_denominator[2] == (F(0), F(0), F(1, 2))
    assert treatment_denominator[1] == (F(0), F(1), F(-1, 2))
    assert treatment_denominator[2] == (F(0), F(0), F(1, 2))
    denominator_variance += _form(t, control_denominator[2], control_denominator[2])
    treatment_variance = _form(t, treatment_denominator[1], treatment_denominator[1])
    treatment_variance += _form(c, treatment_denominator[2], treatment_denominator[2])
    assert treatment_variance == denominator_variance
    assert denominator_variance == F(1801, 800000)
    assert prepared.stability is not None
    assert prepared.stability == F(8820, 1801)
    assert prepared.stability != F(8820)

    k = count_boundary(40, F(1, 20), F(1, 10))
    result = invert_joint(prepared, alpha=F(1, 20), alternative="two-sided", e_value_dual=False)
    assert k > prepared.stability
    assert not result.available
    assert result.reason == "denominator_near_zero"
    assert prepared.log_e("two-sided") < (-log_interval(F(1, 20))).hi

    # Unequal arm sizes retain the same shared-anchor covariance formula and
    # remain available when both adjusted denominators are well separated.
    def rows(n, lift):
        return [(lift + F(i % 3, 10), 10 + F(i % 4, 10), i % 2) for i in range(n)]

    unequal = prepare_joint(
        GaussianState.from_rows(rows(17, 1), dimension=3),
        GaussianState.from_rows(rows(23, 4), dimension=3),
        declaration=declaration,
        null_lift=F(0),
    )
    assert unequal.positive and unequal.stability is not None
    # Direct per-observation residual sums give theta_D=9287/90519 and the
    # control denominator's score below; equal or swapped anchor weights differ.
    assert unequal.stability == F(331953184775580525, 2136433659892)
    available = invert_joint(unequal, alpha=F(1, 20), alternative="two-sided", e_value_dual=False)
    assert available.available


@pytest.mark.parametrize("alpha", [F(1, 20), F(1, 10**300), F(1, 10**400)])
def test_denominator_cap_admits_only_a_strictly_resolved_statistic(alpha):
    """Evidence at or above -log(alpha) implies the denominator statistic is
    strictly above the set's boundary at alpha, a tie included, in exact
    arithmetic at levels far below binary64; a clearly resolved statistic is
    admitted. The contrast is overwhelming, so the evidence is the cap."""
    count, rho = 40, F(1, 10)
    k = count_boundary(count, alpha, rho)
    threshold = (-log_interval(alpha)).hi

    def prepared(statistic):
        return JointPreparation(
            count, rho, contrast=F(1), variance=F(1, 10**2000), stability=statistic
        )

    tiny = F(1, 10**40)
    for statistic in (k - tiny, k, k + tiny, k + F(1, 10**6), 2 * k):
        if prepared(statistic).log_e("two-sided") >= threshold:
            assert statistic > k and prepared(statistic).resolves(k)
    assert not prepared(k).resolves(k)
    assert prepared(2 * k).log_e("two-sided") >= threshold


def test_a_constant_positive_denominator_costs_the_ratio_no_evidence():
    """Zero denominator variance restricts nothing: the ratio's evidence is the
    scalar evidence of N / 2."""
    ratio = _evaluate_ratio([(n, 2) for n in (1, 2, 3)] * 40, [(n, 2) for n in (8, 10, 12)] * 40)
    registration = mean_registration(
        cells=(SequentialCell(metric="outcome", group_id="treatment", alpha=F(1, 20)),)
    )
    records = mean_records([F(n, 2) for n in (1, 2, 3)] * 40, [F(n, 2) for n in (8, 10, 12)] * 40)
    scalar = estimate_sequential(
        mean_capture(registration, records), AsymptoticMean(registration=registration)
    ).results[0]
    result = ratio.require_asymptotic_sequential_result()
    assert result.bounds.available
    assert result.log_e == scalar.require_asymptotic_sequential_result().log_e


def test_a_negative_denominator_mean_never_becomes_evidence():
    """A tightly measured negative denominator squares to a large statistic;
    it is still unresolved at every level and abstains."""
    row = _evaluate_ratio([(1, -2), (2, -3), (3, -2)] * 40, _RATIO_TREATMENT)
    result = row.require_asymptotic_sequential_result()
    assert result.bounds.reason == "denominator_near_zero"
    assert result.log_e == float("-inf")


@pytest.mark.parametrize(
    "alternative,null_lift,directed",
    [
        ("greater", F(0), True),
        ("less", F(0), False),
        ("greater", F(5), False),
        ("less", F(5), True),
    ],
)
def test_ratio_evidence_gates_on_the_declared_contrast_sign(alternative, null_lift, directed):
    """The observed ratio is 5 with well-resolved denominators; the gated
    contrast is f_t - (1 + null_lift) f_c, so a null ratio of 6 reverses it."""
    row = _evaluate_ratio(
        _RATIO_CONTROL, _RATIO_TREATMENT, alternative=alternative, null_lift=null_lift
    )
    assert (row.require_asymptotic_sequential_result().log_e > float("-inf")) == directed


def test_joint_reasons_keep_their_precedence_and_abstain_at_every_alpha():
    """Numerators proportional to the denominators leave each arm's functional
    without variance, and denominators alternating 1 and 10 over 20 units per
    arm have statistic 20 * (11/9)^2 ~ 29.9: unresolved at 1/200, resolved at
    1/20. The reason follows readiness > denominators > variance at each level,
    and the evidence abstains at every level."""
    denominators = [1, 10] * 10
    control = GaussianState.from_rows([[2 * d, d] for d in denominators], dimension=2)
    treatment = GaussianState.from_rows([[3 * d, d] for d in denominators], dimension=2)

    def reason(declaration, alpha):
        return asymptotic_joint_set(
            control,
            treatment,
            declaration=declaration,
            alpha=alpha,
            null_lift=F(0),
            alternative="two-sided",
        ).reason

    ratio = mean_model("m", law="ratio_mean")
    assert reason(ratio, F(1, 200)) == "denominator_near_zero"
    assert reason(ratio, F(1, 20)) == "zero_arm_variance"
    assert prepare_joint(control, treatment, declaration=ratio, null_lift=F(0)).log_e(
        "two-sided"
    ) == float("-inf")
    early = mean_model("m", law="ratio_mean", start_count=50)
    assert reason(early, F(1, 200)) == "before_declared_start"


def test_ratio_one_sided_set_unions_the_never_rejected_half_line():
    # Control (N, D) rows [(-2, 2), (1, 1), (2, 3)] give a small functional 1/6
    # with large scatter: the two-sided set at 2*alpha is disconnected,
    # (-inf, -77.7] u [11.27, inf), with observed ratio 26.4 in the upper piece.
    # "less" adds (-inf, 26.4], covering the line; "greater" adds nothing new.
    control = [(-2, 2), (1, 1), (2, 3)] * 40
    treatment = [(10, 2), (12, 3)] * 40
    two_sided = _evaluate_ratio(control, treatment, alpha=F(1, 10))
    less = _evaluate_ratio(control, treatment, alternative="less")
    greater = _evaluate_ratio(control, treatment, alternative="greater")
    reference = two_sided.require_asymptotic_sequential_result().bounds
    assert reference.status == "disconnected"
    less_bounds = less.require_asymptotic_sequential_result().bounds
    assert less_bounds.status == "full"
    assert less_bounds.components == (MeanSetComponent(lower=None, upper=None),)
    assert not less.stat_sig()
    greater_bounds = greater.require_asymptotic_sequential_result().bounds
    assert greater_bounds.components == reference.components
    assert greater_bounds.status == "disconnected"
    assert greater.stat_sig()


def _induced_frame(rng, n=600):
    """Raw components whose noise cancels the covariate's correlation, so the
    ADJUSTED components carry a negative covariance the covariate induces."""
    arms = np.array(["control", "treatment"] * (n // 2))
    x = rng.normal(10.0, 3.0, size=n)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = -0.5 * e1 + rng.normal(0.0, 0.3, size=n)
    return pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(n)],
            "arm": arms,
            "m": 5.0 + 0.9 * x + e1 + 0.5 * (arms == "treatment"),
            "x": x,
            "d": 4.0 + 0.3 * x + e2,
        }
    )


def _raw_form(scatter, n, left, right):
    return sum(a * scatter[i][j] * b for i, a in enumerate(left) for j, b in enumerate(right)) / F(
        n * n
    )


def test_adjusted_ratio_narrows_ratio_and_keeps_the_covariate_induced_covariance():
    frame = _induced_frame(np.random.default_rng(5))
    _, ratio = _run(frame, MetricSpec(name="m", type="ratio", numerator="m", denominator="d"))
    analysis, adjusted = _run(
        frame,
        MetricSpec(
            name="m",
            type="ratio",
            numerator="m",
            denominator="d",
            covariate="x",
            decision_method=_CUPED,
        ),
    )
    assert analysis.sequential_snapshot().registration.models[0].law == "adjusted_ratio_mean"
    assert _width(adjusted) < _width(ratio)

    # Zeroing the adjusted components' covariance Cov(N', D') = S_ND - theta_D
    # S_NX - theta_N S_DX + theta_N theta_D S_XX (only the raw S_ND entry moves)
    # visibly narrows the set here, so the reported set carries the covariance
    # the shared covariate induces rather than a diagonal shortcut.
    snapshot = analysis.sequential_snapshot()
    states = {s.group_id: s.kernel() for s in snapshot.states}
    declaration = snapshot.registration.models[0]
    linearisation, _ = linearise(states["control"], states["treatment"], declaration)
    assert linearisation is not None
    theta_n = _coefficient(states["control"], states["treatment"], 0, 2)
    theta_d = _coefficient(states["control"], states["treatment"], 1, 2)
    k = adjusted.require_asymptotic_sequential_result().bounds.k
    arms = (
        (states["control"], linearisation.control),
        (states["treatment"], linearisation.treatment),
    )
    d0, d1 = linearisation.treatment.value, -linearisation.control.value

    def quadratic(drop):
        totals = []
        for left, right in ((0, 0), (0, 1), (1, 1)):
            total = F(0)
            for state, arm in arms:
                s = [list(row) for row in state.scatter]
                if drop:
                    s[0][1] = s[1][0] = (
                        theta_d * s[0][2] + theta_n * s[1][2] - theta_n * theta_d * s[2][2]
                    )
                vectors = (arm.slope, arm.constant)
                total += _raw_form(s, state.n, vectors[left], vectors[right])
            totals.append(total)
        slope, cross, constant = totals
        return d1 * d1 - k * slope, 2 * d0 * d1 - 2 * k * cross, d0 * d0 - k * constant

    (full,) = _component_for_quadratic(*quadratic(False))
    (dropped,) = _component_for_quadratic(*quadratic(True))
    assert full.lower is not None and full.upper is not None
    assert dropped.lower is not None and dropped.upper is not None
    assert dropped.upper - dropped.lower < F(97, 100) * (full.upper - full.lower)
    assert adjusted.require_asymptotic_sequential_result().bounds.components == (full,)


def test_contrast_quadratic_identity_holds_exactly():
    """``D(r)^2 - K V(r)`` equals the emitted quadratic at every rational r, and the
    prepared null contrast and variance are ``D(r0)`` and ``V(r0)``."""
    rng = np.random.default_rng(9)

    def rows(n):
        return [[F(int(v)) for v in row] for row in rng.integers(1, 12, size=(n, 3))]

    control = GaussianState.from_rows(rows(7), dimension=3)
    treatment = GaussianState.from_rows(rows(9), dimension=3)
    declaration = mean_model("m", law="adjusted_ratio_mean")
    linearisation, _ = linearise(control, treatment, declaration)
    assert linearisation is not None
    arms = (("c", control, linearisation.control), ("t", treatment, linearisation.treatment))
    k = count_boundary(16, F(1, 20), F(1, 10))
    d0, d1 = linearisation.treatment.value, -linearisation.control.value
    a, b, c = _quadratic(_variance_forms(arms), d0, d1, k)

    def variance(r):
        total = F(0)
        for _, state, arm in arms:
            gradient = tuple(g + r * s for g, s in zip(arm.constant, arm.slope, strict=True))
            total += _form(state, gradient, gradient)
        return total

    for r in (F(-3, 2), F(1), F(7, 3), F(19, 4)):
        assert (d0 + d1 * r) ** 2 - k * variance(r) == a * r * r + b * r + c
    prepared = prepare_joint(control, treatment, declaration=declaration, null_lift=F(4, 3))
    assert prepared.contrast == d0 + d1 * F(7, 3)
    assert prepared.variance == variance(F(7, 3))


def _looks(rng, law, n_per_look, looks, *, effect):
    """Time-uniform coverage of one replication: every look's set contains the truth."""
    registration = mean_registration(
        models=(mean_model("m", law=law),),
        cells=(SequentialCell(metric="m", group_id="treatment"),),
    )
    policy = AsymptoticMean(registration=registration)
    total = n_per_look * looks
    x = rng.normal(10.0, 3.0, size=(2, total))
    y = 5.0 + 0.9 * x + rng.normal(0.0, 1.0, size=(2, total))
    d = 4.0 + 0.3 * x + rng.normal(0.0, 0.5, size=(2, total))
    y[1] += effect
    if law == "adjusted_mean":
        values = [[(y[a][i], x[a][i]) for i in range(total)] for a in (0, 1)]
        truth = F(1) + F(effect) / F(5.0 + 9.0)
    elif law == "ratio_mean":
        values = [[(y[a][i], d[a][i]) for i in range(total)] for a in (0, 1)]
        truth = (F(14.0) + F(effect)) / F(7.0) / (F(14.0) / F(7.0))
    else:
        values = [[(y[a][i], d[a][i], x[a][i]) for i in range(total)] for a in (0, 1)]
        truth = (F(14.0) + F(effect)) / F(7.0) / (F(14.0) / F(7.0))
    snapshot = None
    covered = True
    for look in range(1, looks + 1):
        snapshot = mean_capture(
            registration,
            mean_records(
                values[0][: look * n_per_look], values[1][: look * n_per_look], metrics=("m",)
            ),
            previous=snapshot,
        )
        bounds = (
            estimate_sequential(snapshot, policy)
            .results[0]
            .require_asymptotic_sequential_result()
            .bounds
        )
        if bounds.available and not _covers(bounds, truth):
            covered = False
    return covered


@pytest.mark.parametrize("law", ["adjusted_mean", "ratio_mean", "adjusted_ratio_mean"])
def test_time_uniform_coverage_smoke(law):
    """Fast-tier twin of the calibrated cell below: 12 replications, 3 looks."""
    rng = np.random.default_rng(17)
    misses = sum(not _looks(rng, law, 40, 3, effect=0.5) for _ in range(12))
    assert misses <= 1


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("law", ["adjusted_mean", "ratio_mean", "adjusted_ratio_mean"])
def test_time_uniform_miscoverage_across_repeated_looks(law):
    """Exact one-sided Clopper-Pearson bound on the ever-miss rate across
    six appended looks, the shape ``TestDeployedRuntimeNulls`` uses."""
    rng = np.random.default_rng(23)
    reps = 300
    misses = sum(not _looks(rng, law, 50, 6, effect=0.5) for _ in range(reps))
    upper = binomial_error_upper_bound(misses, reps, family_eta(0.01, 3))
    assert upper <= ALPHA + scientific_delta(ALPHA), (misses, upper)
