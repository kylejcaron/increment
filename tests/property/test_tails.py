"""Property tests for increment/estimation/_tails.py's two-sided critical
value: monotone in alpha, and resolvable (finite, not silently 0/inf)
down to alpha values far smaller than double precision can represent as
`1 - alpha`."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist

from increment.estimation._tails import two_sided_critical_value

_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)

_alphas = st.floats(min_value=1e-12, max_value=0.5, allow_nan=False, allow_infinity=False)


@given(a=_alphas, b=_alphas)
@_SETTINGS
def test_normal_critical_value_is_monotone_decreasing_in_alpha(a, b):
    lo, hi = sorted((a, b))
    crit_lo = two_sided_critical_value(norm_dist.isf, lo, what="normal test")
    crit_hi = two_sided_critical_value(norm_dist.isf, hi, what="normal test")
    assert crit_lo >= crit_hi


@given(a=_alphas, b=_alphas, dof=st.integers(min_value=2, max_value=500))
@_SETTINGS
def test_t_critical_value_is_monotone_decreasing_in_alpha(a, b, dof):
    lo, hi = sorted((a, b))
    crit_lo = two_sided_critical_value(t_dist.isf, lo, dof, what="t test")
    crit_hi = two_sided_critical_value(t_dist.isf, hi, dof, what="t test")
    assert crit_lo >= crit_hi


@given(alpha=st.floats(min_value=1e-150, max_value=1e-6, allow_nan=False, allow_infinity=False))
@_SETTINGS
def test_normal_critical_value_stays_finite_far_below_double_precision(alpha):
    # `1 - alpha/2` rounds to exactly 1.0 well above this range (~alpha <
    # 1e-16), which would make ppf(1 - alpha/2) silently return inf; the
    # isf-based implementation must stay finite and positive here.
    crit = two_sided_critical_value(norm_dist.isf, alpha, what="normal test")
    assert crit > 0.0
    import math

    assert math.isfinite(crit)
