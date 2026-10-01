"""Focused one-sided selected-interval FCR regressions."""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_one_sided_fcr_lower_bound_realized_fcr_is_controlled():
    import math

    import numpy as np
    from scipy.stats import norm

    from increment.estimation.family import bh_select
    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    m, q, nominal, reps = 4, 0.1, 0.05, 100_000
    rng = np.random.default_rng(73019)
    critical = {}
    for selected in range(1, m + 1):
        fcr_alpha = min(selected * q / m, nominal)
        row = infer_lift(
            metric="m",
            group_id="t",
            method="unadjusted",
            method_role="decision",
            log_rr=0.0 - 0.0,
            se_t=0.1 / math.sqrt(2),
            se_c=0.1 / math.sqrt(2),
            alpha=fcr_alpha / 2.0,
            alternative="greater",
        )
        row = open_bound_from_two_sided_at_target(row)
        lift = row.require_lift()
        assert lift.lb is not None
        critical[selected] = -math.log1p(lift.lb) / 0.1

    for signal in (2.0, 4.0, 10.0):
        truth = np.array([signal, 0.0, 0.0, 0.0])
        estimates = rng.normal(size=(reps, m)) + truth
        p = norm.sf(estimates)
        fractions = np.zeros(reps)
        for i in range(reps):
            selected, _ = bh_select(p[i].tolist(), q=q)
            if selected:
                fractions[i] = np.mean(
                    (estimates[i, selected] - critical[len(selected)]) > truth[selected]
                )
        mcse = fractions.std(ddof=1) / math.sqrt(reps)
        assert fractions.mean() <= q + 4 * mcse, (signal, fractions.mean(), mcse)


def test_one_sided_fcr_handles_large_fcr_alpha_without_refusing():
    from increment.estimation.inference import infer_lift

    fcr_alpha = 0.6
    row = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.0 - 0.0,
        se_t=0.1,
        se_c=0.1,
        alpha=fcr_alpha / 2.0,
        alternative="greater",
    )
    assert row.require_lift().alpha == pytest.approx(fcr_alpha)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_fixed_fcr_exact_tail_when_all_signals_are_selected():
    """Certain selection and a nonbinding cap reduce FCR to marginal error."""
    import math

    import numpy as np
    from scipy.stats import norm

    from increment.estimation.inference import infer_lift
    from increment.estimation.results import open_bound_from_two_sided_at_target

    q, reps, m = 0.1, 100_000, 4
    parent = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0 - 0,
        se_t=0.1 / math.sqrt(2),
        se_c=0.1 / math.sqrt(2),
        alpha=q / 2,
        alternative="greater",
    )
    row = open_bound_from_two_sided_at_target(parent)
    row_lift = row.require_lift()
    parent_lift = parent.require_lift()
    assert row_lift.lb is not None
    assert parent_lift.lb is not None
    critical = -math.log1p(row_lift.lb) / 0.1
    parent_critical = -math.log1p(parent_lift.lb) / 0.1
    errors = np.random.default_rng(73020).normal(size=(reps, m))
    # Each p <= q implies BH selects all m; nominal alpha=q leaves R*q/m uncapped.
    assert np.all(norm.sf(10 + errors) <= q)
    for threshold, target in ((critical, q), (parent_critical, q / 2)):
        fractions = (errors > threshold).mean(axis=1)
        mcse = fractions.std(ddof=1) / math.sqrt(reps)
        assert abs(fractions.mean() - target) <= 4 * mcse


def test_joint_row_displays_centrally_until_fcr_opens_its_far_bound():
    """A directional joint (Fieller) row carries the same two representations
    as a scalar one: central by default, far endpoint opened for FCR.

    The set stays closed toward the alternative either way -- that is what
    `contains`/`stat_sig` read -- so only the displayed interval moves. Both
    forms keep the full `alpha_eff` the retained endpoint was inverted at,
    which is what lets an FCR consumer treat `lb` as a bound at the total
    target alpha rather than at half of it.
    """
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.inference import infer_ate
    from increment.estimation.results import (
        JointContrastReference,
        open_bound_from_two_sided_at_target,
        relative_confidence_set,
    )

    reference = JointContrastReference(a=2.0, c=3.0, var_a=0.04, var_c=0.09, cov_ac=0.01)
    relative = relative_confidence_set(reference, alpha=0.05, alternative="greater")
    central = relative_confidence_set(reference, alpha=0.10, alternative="two-sided")

    # The closure shapes the set, never the display.
    assert relative.geometry == "one_sided" and relative.intervals[0][1] is None
    displayed = relative.estimate()
    assert displayed is not None
    assert displayed == central.estimate()
    assert displayed.lb is not None and displayed.ub is not None
    assert displayed.open_side is None
    assert displayed.alpha == pytest.approx(0.10)
    assert displayed.level == pytest.approx(0.90)

    row = infer_ate(
        metric="m",
        group_id="t",
        method="unadjusted",
        point=reference.a,
        scores=ScoreStats(metric="m", contrast="t", n=64, sum_psi=0.0, sum_psi2=0.04 * 64),
        alpha=0.05,
        alternative="greater",
        joint_reference=reference,
        method_role="decision",
    )
    assert row.require_lift() == displayed
    opened = open_bound_from_two_sided_at_target(row).require_lift()
    assert opened.ub is None
    assert opened.open_side == "upper"
    assert opened.lb == displayed.lb
    assert opened.alpha == pytest.approx(0.10)
    assert opened.level == pytest.approx(0.90)
