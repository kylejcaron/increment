"""An exact-binomial row's additive interval is the Miettinen-Nurminen score interval for the
difference of proportions, and an absolute margin on that row is decided by the score test the
interval inverts. Expected values come from the restricted maximum-likelihood rates solved from
the likelihood equation here, independently of the production cubic."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy.optimize import brentq
from scipy.stats import binom, norm

from increment.errors import InvalidRequestError
from increment.estimation._fcr_reinterval import classify, reinterval_at_fcr
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.engine import binomial_abs_interval, estimate_lift
from increment.estimation.family import select_exploratory_family
from increment.estimation.risk_difference import score_bounds, score_p_value
from increment.result_envelope import AnalysisEnvelope, ResultCell
from increment.semantics.models import ConversionMetric
from tests.estimation._conversion_counts import CONVERSION_METRIC, count_summary, lift_row

# A binomial margin row saved at the commit before binomial rows carried a score interval:
# ``ResultCell.from_row(row).to_payload()`` of the margin row below, beside its p-value and
# verdict at that commit.
_WALD_MARGIN_CELL = Path(__file__).parents[1] / "fixtures" / "binomial_wald_margin_cell.json"


def _independent_p_value(
    x_c: int, n_c: int, x_t: int, n_t: int, delta: float, alternative: str
) -> float:
    """One-sided score p-value, the restricted rates solved from the likelihood equation
    (every count strictly between zero and its arm size)."""

    def slope(p_c: float) -> float:
        p_t = p_c + delta
        return x_t / p_t - (n_t - x_t) / (1 - p_t) + x_c / p_c - (n_c - x_c) / (1 - p_c)

    low, high = max(0.0, -delta), min(1.0, 1.0 - delta)
    margin = 1e-12 * (high - low)
    p_c = brentq(slope, low + margin, high - margin, xtol=1e-17, rtol=1e-15)
    p_t = p_c + delta
    total = n_c + n_t
    variance = (p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c) * total / (total - 1)
    z = (x_t / n_t - x_c / n_c - delta) / math.sqrt(variance)
    return float(norm.sf(z) if alternative == "greater" else norm.sf(-z))


def _crit(alpha_eff: float) -> float:
    return float(norm.isf(alpha_eff / 2.0))


def test_binomial_margin_abs_uses_score_test():
    """A conversion guardrail with a 0.01 absolute margin and an increase preferred, 10/1000
    control against 4/1000 treatment. The Wald rule gave p = 0.1416."""
    computation = estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(10, 1000, 4, 1000),
        control_group="control",
        alpha=0.05,
        alternative="greater",
        null_abs=-0.01,
        preferred_direction="increase",
    )
    (row,) = computation.results
    assert row.reference_kind == "binomial"
    assert row.abs_reference_kind == "score"
    p_value = row.p_value()
    assert p_value == pytest.approx(0.16050, rel=1e-4)
    assert p_value is not None
    (evidence,) = computation.evidence.values()
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == p_value
    assert row.abs_lb is not None
    assert row.stat_sig() is (row.abs_lb > -0.01)
    assert row.stat_sig() is (p_value < 0.05)
    # The displayed difference and standard error stay the arms' own Wald values.
    assert row.abs_diff == pytest.approx(-0.006, abs=1e-15)
    wald_se = math.hypot(math.sqrt(4 * 996 / 999) / 1000, math.sqrt(10 * 990 / 999) / 1000)
    assert row.abs_se == pytest.approx(wald_se, rel=1e-12)


@pytest.mark.parametrize(
    ("counts", "alpha_eff"),
    [
        ((48, 80, 56, 70), 0.05),
        ((10, 1000, 4, 1000), 0.1),
        # A tail this small needs the survival form: 1 - alpha/2 rounds to one.
        ((20, 200, 40, 200), 1e-300),
    ],
)
def test_score_bounds_invert_score_test(counts: tuple[int, int, int, int], alpha_eff: float):
    lower, upper, kind = binomial_abs_interval(*counts, alpha_eff)
    assert kind == "score"
    assert -1.0 < lower < upper < 1.0
    assert _independent_p_value(*counts, lower, "greater") == pytest.approx(
        alpha_eff / 2.0, rel=1e-7
    )
    assert _independent_p_value(*counts, upper, "less") == pytest.approx(alpha_eff / 2.0, rel=1e-7)


def test_alpha_below_tail_resolution_refuses_by_code():
    # 5e-324 / 2 rounds to a zero tail probability: no quantile exists.
    with pytest.raises(InvalidRequestError) as raised:
        binomial_abs_interval(20, 200, 40, 200, 5e-324)
    assert raised.value.code == "estimation.tails.unresolvable"


def test_score_interval_zero_events():
    (row,) = estimate_lift(
        metrics=[CONVERSION_METRIC],
        summary=count_summary(0, 200, 0, 200),
        control_group="control",
    ).results
    assert row.abs_se is None
    assert row.abs_reference_kind == "score"
    assert row.abs_lb is not None and row.abs_ub is not None
    assert math.isfinite(row.abs_lb) and math.isfinite(row.abs_ub)
    assert row.abs_lb < 0.0 < row.abs_ub


def test_margin_null_outside_the_unit_interval_is_impossible():
    """A difference of proportions lies in ``[-1, 1]``: a null below it is rejected against
    ``"greater"`` and two-sided alternatives and never against ``"less"``, symmetrically above,
    also where the observed difference sits on the boundary the null lies beyond."""
    cases = {
        ((50, 50, 0, 50), -2.0): {"greater": 0.0, "less": 1.0, "two-sided": 0.0},
        ((0, 50, 50, 50), 2.0): {"greater": 1.0, "less": 0.0, "two-sided": 0.0},
    }
    for (counts, null_abs), tails in cases.items():
        for alternative, expected in tails.items():
            (row,) = estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(*counts),
                control_group="control",
                alternative=alternative,
                null_abs=null_abs,
            ).results
            assert row.abs_reference_kind == "score"
            assert row.p_value() == pytest.approx(expected, abs=1e-12)
            assert row.stat_sig() is (expected < 0.05)


def test_reissued_binomial_row_keeps_score_reference():
    counts = (10, 1000, 4, 1000)
    row = lift_row(counts)
    construction, _ = classify(row)
    assert construction == "binomial"
    expected = score_bounds(*counts, _crit(0.02))

    reissued = reinterval_at_fcr(row, construction, 0.02)
    envelope = AnalysisEnvelope((ResultCell.from_row(reissued),), None)
    payload = json.loads(json.dumps(envelope.to_payload()))
    restored = AnalysisEnvelope.from_payload(payload).cells[0].retained_row()
    again = reinterval_at_fcr(restored, construction, 0.02)
    for candidate in (reissued, restored, again):
        assert candidate.abs_reference_kind == "score"
        assert candidate.abs_alpha == 0.02
        assert (candidate.abs_lb, candidate.abs_ub) == pytest.approx(expected, abs=1e-10)


def test_selected_delta_row_rerouted_to_binomial_gets_score_bounds():
    """Ten secondaries at ``q = 0.05`` select two rows, so each is reissued at the corrected
    level 0.01. The sparse row's counts are not dense there and reroute to the exact route;
    the dense row's counts are and keep the delta route."""
    cells = {
        "sparse": (2_500, 50_000, 2_800, 50_000),
        "dense": (100_000, 1_000_000, 103_000, 1_000_000),
        **{f"null_{i}": (2_500, 50_000, 2_500 + i, 50_000) for i in range(8)},
    }
    rows = []
    for name, counts in cells.items():
        summary = count_summary(*counts)
        for arm in summary:
            arm["metric"] = name
        (row,) = estimate_lift(
            metrics=[ConversionMetric(name=name, entity="user", fact=name)],
            summary=summary,
            control_group="control",
        ).results
        assert row.reference_kind == "t"
        rows.append(row)
    selected = {row.metric: row for row in select_exploratory_family(rows, q=0.05) if row.discovery}
    assert set(selected) == {"sparse", "dense"}

    sparse = selected["sparse"]
    assert sparse.reference_kind == "binomial"
    assert sparse.abs_reference_kind == "score"
    assert sparse.abs_alpha == pytest.approx(0.01, rel=1e-12)
    expected = score_bounds(*cells["sparse"], _crit(0.01))
    assert (sparse.abs_lb, sparse.abs_ub) == pytest.approx(expected, abs=1e-10)

    dense = selected["dense"]
    assert dense.reference_kind == "t"
    assert dense.abs_reference_kind in ("t", "normal")


def test_legacy_binomial_row_reads_as_stored_and_relevels_to_score():
    saved = json.loads(_WALD_MARGIN_CELL.read_text())
    stored = saved["cell"]["retained_row"]
    row = ResultCell.from_payload(saved["cell"]).retained_row()
    assert row.abs_reference_kind is None
    assert (row.abs_lb, row.abs_ub) == (stored["abs_lb"], stored["abs_ub"])
    assert row.p_value() == pytest.approx(saved["p_value"], abs=1e-12)
    assert row.stat_sig() is saved["stat_sig"]

    bset = row.binomial_set
    assert bset is not None
    reissued = reinterval_at_fcr(row, "binomial", 0.05)
    assert reissued.abs_reference_kind == "score"
    assert reissued.abs_alpha is not None
    expected = score_bounds(bset.x_c, bset.n_c, bset.x_t, bset.n_t, _crit(reissued.abs_alpha))
    assert (reissued.abs_lb, reissued.abs_ub) == pytest.approx(expected, abs=1e-10)

    # The Wald p-value 0.1416 is below q = 0.15, the score test the reissue inverts (0.1605)
    # is not: a family decides the row on the score test and reports it.
    (member,) = select_exploratory_family([row], q=0.15)
    assert member.discovery is False
    assert member.p_value() == pytest.approx(0.16050, rel=1e-4)


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("events", [2, 5, 10, 50, 100])
def test_score_margin_test_holds_its_size_at_the_margin_boundary(events):
    """Exact size of the one-sided 0.05 score test of ``p_t - p_c <= -0.01`` at the boundary
    ``p_t = p_c - 0.01``, 1,000 units per arm, summed over the count lattice (the Wald rule's
    size was 0.0785 at two expected treatment events)."""
    n, margin, alpha = 1000, 0.01, 0.05
    p_t = events / n
    p_c = p_t + margin

    def support(p: float) -> range:
        return range(min(n, math.ceil(n * p + 12 * math.sqrt(n * p * (1 - p)) + 10)) + 1)

    treated, control = support(p_t), support(p_c)
    mass_t, mass_c = binom.pmf(list(treated), n, p_t), binom.pmf(list(control), n, p_c)
    size = math.fsum(
        float(w_t * w_c)
        for x_t, w_t in zip(treated, mass_t, strict=True)
        for x_c, w_c in zip(control, mass_c, strict=True)
        if score_p_value(x_c, n, x_t, n, -margin, "greater") < alpha
    )
    assert size <= 0.0505
