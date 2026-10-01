"""Property tests for increment/estimation/sitewide.py's _combination_var:
each returned term is the delta-method variance contribution of one arm
(or the shared control-anchored term), so it must be quadratic-form
nonnegative given a valid covariance structure, and homogeneous of
degree 2 in the contraction coefficients."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from increment.estimation.sitewide import _Arm, _combination_var

_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)

_pos_var = st.floats(min_value=1e-6, max_value=1e4, allow_nan=False, allow_infinity=False)
_coef = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_rho = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def _arm_with_valid_covariance(draw):
    var_mean = draw(_pos_var)
    var_mean_den = draw(_pos_var)
    rho = draw(_rho)
    cov_mean_den = rho * (var_mean * var_mean_den) ** 0.5
    # group_id/n_units are required fields _combination_var never reads;
    # constants keep the strategy focused on the covariance structure.
    return _Arm(
        group_id="x",
        n_units=1.0,
        mean=0.0,
        var_mean=var_mean,
        mean_den=0.0,
        var_mean_den=var_mean_den,
        cov_mean_den=cov_mean_den,
    )


@given(
    control=_arm_with_valid_covariance(),
    arms=st.lists(_arm_with_valid_covariance(), min_size=1, max_size=4),
    coef_y=st.lists(_coef, min_size=1, max_size=4),
    coef_den=st.lists(_coef, min_size=1, max_size=4),
)
@_SETTINGS
def test_ratio_terms_are_nonnegative_given_valid_covariance(control, arms, coef_y, coef_den):
    n = min(len(arms), len(coef_y), len(coef_den))
    if n == 0:
        return
    terms = _combination_var(control, arms[:n], coef_y[:n], coef_den[:n])
    # tolerance: the PSD guarantee is exact in real arithmetic; float64
    # rounding on a term near zero can go slightly negative.
    floor = -1e-6 * (control.var_mean + control.var_mean_den + 1.0)
    assert all(t >= floor for t in terms)


@given(
    control=_arm_with_valid_covariance(),
    arms=st.lists(_arm_with_valid_covariance(), min_size=1, max_size=4),
    coef_y=st.lists(_coef, min_size=1, max_size=4),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@_SETTINGS
def test_sum_type_terms_are_homogeneous_of_degree_two(control, arms, coef_y, scale):
    n = min(len(arms), len(coef_y))
    if n == 0:
        return
    base = _combination_var(control, arms[:n], coef_y[:n])
    scaled = _combination_var(control, arms[:n], [scale * c for c in coef_y[:n]])
    for b, s in zip(base, scaled, strict=True):
        assert s == pytest.approx(b * scale * scale, rel=1e-9, abs=1e-9)


@given(
    control=_arm_with_valid_covariance(),
    arms=st.lists(_arm_with_valid_covariance(), min_size=1, max_size=4),
)
@_SETTINGS
def test_sum_type_terms_are_nonnegative(control, arms):
    coef_y = [1.0] * len(arms)
    terms = _combination_var(control, arms, coef_y)
    assert all(t >= -1e-9 for t in terms)
