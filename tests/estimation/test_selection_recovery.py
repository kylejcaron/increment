"""Simulation checks: selection concentrates on the true optimum and the
outer estimate is honest for the locked rule's true per-treated effect."""

from __future__ import annotations

import numpy as np
import pytest

from increment.estimation.cate import Covariate
from increment.estimation.targeting import select_targeting_rule_arrays

SPEND = Covariate(name="spend")


def _draw(rng: np.random.Generator, n: int, noise: float) -> dict:
    """Step CATE with known geometry: top 20% by spend has effect 2.5, rest 0.5.

    With cost 1.0 the true net benefit is maximized at fraction 0.2, and the
    true per-treated effect of the locked 0.2-rule is 2.5.
    """
    spend = rng.uniform(-2.0, 2.0, n)
    d = rng.integers(0, 2, n).astype(float)
    y = 1.0 + 0.3 * spend + d * (0.5 + 2.0 * (spend > 1.2)) + noise * rng.standard_normal(n)
    ids = np.array([f"u{i}" for i in range(n)])
    return {"y": y, "d": d, "cols": {"spend": spend}, "unit_ids": ids}


def _run(rep: int, n: int, noise: float):
    rng = np.random.default_rng(1000 + rep)
    return select_targeting_rule_arrays(
        **_draw(rng, n, noise),
        interact=[SPEND],
        fractions=(0.1, 0.2, 0.4),
        cost_per_treated=1.0,
        n_folds=4,
        seed=rep,
    )


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_selection_concentrates_on_the_true_optimum_and_stays_honest():
    results = [_run(rep, n=600, noise=1.0) for rep in range(60)]
    picked = np.array([r.selected_fraction for r in results])
    # The true argmax is 0.2; a clear majority of replications must find it.
    assert (picked == 0.2).mean() > 0.6
    # Honesty: among gate-passing reps that locked 0.2, the outer per-treated
    # estimate centers on 2.5 - selection did not leak bias into the held-out test.
    values = np.array(
        [
            r.rule.policy_value.value
            for r in results
            if r.selected_fraction == 0.2 and r.rule.recommendation == "target"
        ]
    )
    assert values.size > 20
    assert abs(values.mean() - 2.5) < 0.15


def test_selection_recovery_smoke():
    """Fast-suite variant: one small replication runs end to end."""
    result = _run(0, n=200, noise=0.5)
    assert result.rule.recommendation in ("target", "simple")
    assert result.rule.validation.n_holdout == 100
