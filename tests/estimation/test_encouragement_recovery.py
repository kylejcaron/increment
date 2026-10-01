"""Parameter-recovery evidence for the encouragement-design LATE estimators.

Fast smoke variant runs in the default suite; the full Monte-Carlo coverage
checks are marked ``@pytest.mark.parameter_recovery`` (excluded from the
fast suite, see ``pyproject.toml``'s ``addopts``) and are the sign-error
arbiter for the additive and complier-relative delta-method variances in
``increment/estimation/encouragement.py``.
"""

from __future__ import annotations

import pytest

from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.encouragement import estimate_encouragement
from increment.semantics.design import Encouragement
from increment.semantics.models import MeanMetric


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
    import numpy as np

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


def _replicate(n_reps, n, tau, compliance, one_sided, seed, value_scale="absolute", target=None):
    """Run ``n_reps`` fresh-seed replications of the encouragement DGP,
    extract the LATE row on ``value_scale`` (``"absolute"`` or ``"relative"``),
    and return (mean bias against ``target``, empirical CI coverage).

    ``target`` defaults to ``tau`` (the additive LATE); the relative-scale
    test passes the true complier-relative ratio instead.
    """
    import numpy as np

    if target is None:
        target = tau
    hits, biases = 0, []
    for i in range(n_reps):
        rows = _rows(n=n, tau=tau, compliance=compliance, one_sided=one_sided, seed=seed + i)
        res = estimate_encouragement([METRIC], rows, _design(one_sided=one_sided)).results
        late = [r for r in res if r.estimand == "late" and r.value_scale == value_scale][0]
        biases.append(late.require_lift().value - target)
        hits += late.require_lift().lb <= target <= late.require_lift().ub
    return np.mean(biases), hits / n_reps


def test_late_recovery_smoke():
    bias, cover = _replicate(20, 2000, tau=2.0, compliance=0.5, one_sided=True, seed=11)
    assert abs(bias) < 0.3
    assert 0.80 <= cover <= 1.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_late_recovery_full_one_sided():
    bias, cover = _replicate(300, 4000, tau=2.0, compliance=0.4, one_sided=True, seed=101)
    assert abs(bias) < 0.05
    assert 0.92 <= cover <= 0.98


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_late_recovery_full_two_sided():
    bias, cover = _replicate(300, 4000, tau=2.0, compliance=0.5, one_sided=False, seed=202)
    assert abs(bias) < 0.05
    assert 0.92 <= cover <= 0.98


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_relative_late_coverage():
    """Complier-relative LATE: relative lift is the ratio of the complier
    treated mean to the complier control mean, minus one. With the ``_rows``
    DGP (complier control mean = 10, complier treated mean = 10 + tau), a
    ``tau=2.0`` treatment effect gives a true ratio of 0.20 (see
    ``test_encouragement.test_complier_relative_late_recovers_ratio``).
    """
    bias, cover = _replicate(
        300,
        4000,
        tau=2.0,
        compliance=0.5,
        one_sided=True,
        seed=303,
        value_scale="relative",
        target=0.20,
    )
    assert abs(bias) < 0.05
    assert 0.92 <= cover <= 0.98
