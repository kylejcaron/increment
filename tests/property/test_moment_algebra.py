"""Property tests for increment/estimation/armstats.py's centered-moment
merge algebra: ArmStats.combine must agree with a single-pass reduction
of the concatenated data, regardless of partitioning or ordering; it
must also stay order-invariant on directly-constructed centered inputs
spanning a large reference offset, and from_raw_sums's own guard rails
must fire (warn-and-clamp or refuse) when a large constant offset pushes
the raw sum-of-squares below float64's resolution floor."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from increment.errors import InvalidRequestError
from increment.estimation.armstats import ArmStats, variance_slack
from tests.warning_codes import warning_codes

_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


def _arm(values: list[float], *, group_id: str = "T") -> ArmStats:
    n = len(values)
    return ArmStats.from_raw_sums(
        study_id="s",
        metric="m",
        group_id=group_id,
        n=n,
        sum_y=math.fsum(values),
        sum_y2=math.fsum(v * v for v in values),
    )


def _split(values: list[float], cut_points: list[int]) -> list[list[float]]:
    bounds = sorted({0, len(values), *cut_points})
    return [values[a:b] for a, b in zip(bounds, bounds[1:], strict=False) if b > a]


_small_floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


@st.composite
def _values_and_cuts(draw, min_size=4, max_size=60):
    values = draw(st.lists(_small_floats, min_size=min_size, max_size=max_size))
    n = len(values)
    cuts = draw(st.lists(st.integers(min_value=1, max_value=n - 1), max_size=4))
    return values, cuts


@given(_values_and_cuts())
@_SETTINGS
def test_combine_matches_single_pass_on_the_concatenation(data):
    values, cuts = data
    try:
        direct = _arm(values)
        partitions = [_arm(part) for part in _split(values, cuts) if part]
        combined = ArmStats.combine(partitions) if len(partitions) > 1 else partitions[0]
    except InvalidRequestError as exc:
        # A near-constant partition can legitimately exceed the
        # rounding tolerance and refuse (police_sq_sum); that is
        # correct behavior for this input, not a property violation.
        assert exc.code in {
            "estimation.armstats.centered_sum_squares",
            "estimation.armstats.arm_stats.combine_needs_least",
        }
        return
    except RuntimeWarning:
        # Same legitimate-cancellation input, but police_sq_sum warned
        # and clamped instead of refusing; filterwarnings=error raises it.
        return
    tol = variance_slack(abs(direct.cy2) + 1.0, direct.n)
    assert combined.n == direct.n
    assert combined.mean_y() == pytest.approx(direct.mean_y(), abs=1e-9)
    assert combined.cy2 == pytest.approx(direct.cy2, abs=tol)


@given(st.permutations(list(range(4, 40))).map(lambda idx: [float(i) * 0.37 - 3.1 for i in idx]))
@_SETTINGS
def test_moments_are_order_invariant(shuffled_values):
    baseline = sorted(shuffled_values)
    a = _arm(baseline)
    b = _arm(shuffled_values)
    tol = variance_slack(abs(a.cy2) + 1.0, a.n)
    assert a.mean_y() == pytest.approx(b.mean_y(), abs=1e-9)
    assert a.cy2 == pytest.approx(b.cy2, abs=tol)


_common_offsets = st.sampled_from([float(2**40), float(-(2**40)), float(2**50)])
_partition_n = st.integers(min_value=1, max_value=100)
_delta_steps = st.integers(min_value=-128, max_value=128)
_partition_cy2 = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def _partitions_and_a_permutation(draw, min_size=2, max_size=6):
    offset = draw(_common_offsets)
    specs = draw(
        st.lists(
            st.tuples(_partition_n, _delta_steps, _partition_cy2),
            min_size=min_size,
            max_size=max_size,
        )
    )
    step = math.ulp(offset)
    partitions = [
        ArmStats(
            study_id="s",
            metric="m",
            group_id="T",
            n=n,
            ref_y=offset + delta_steps * step,
            cy1=0.0,
            cy2=cy2,
        )
        for n, delta_steps, cy2 in specs
    ]
    order = draw(st.permutations(range(len(partitions))))
    return partitions, order


def _exact_pooled_moments(partitions: list[ArmStats]) -> tuple[float, float]:
    total_n = sum(part.n for part in partitions)
    exact_refs = [Fraction.from_float(part.ref_y) for part in partitions]
    exact_mean = (
        sum(
            (part.n * ref for part, ref in zip(partitions, exact_refs, strict=True)),
            start=Fraction(),
        )
        / total_n
    )
    exact_cy2 = sum(
        (
            Fraction.from_float(part.cy2) + part.n * (ref - exact_mean) ** 2
            for part, ref in zip(partitions, exact_refs, strict=True)
        ),
        start=Fraction(),
    )
    return float(exact_mean), float(exact_cy2 / (total_n - 1))


@given(_partitions_and_a_permutation())
@_SETTINGS
def test_combine_is_order_invariant_against_exact_large_offset_oracle(data):
    partitions, order = data
    expected_mean, expected_var = _exact_pooled_moments(partitions)
    baseline = ArmStats.combine(partitions)
    reordered = ArmStats.combine([partitions[i] for i in order])
    mean_tol = 4 * len(partitions) * max(math.ulp(part.ref_y) for part in partitions)
    var_tol = 128 * len(partitions) * math.ulp(max(abs(expected_var), 1.0))

    for combined in (baseline, reordered):
        assert combined.n == sum(part.n for part in partitions)
        assert combined.cy2 >= 0.0
        assert combined.mean_y() == pytest.approx(expected_mean, rel=0.0, abs=mean_tol)
        assert combined.var_y() == pytest.approx(expected_var, rel=0.0, abs=var_tol)


@given(
    residuals=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=40,
    ),
)
@_SETTINGS
def test_raw_sum_recovery_warns_or_refuses_under_offset_cancellation(residuals):
    # True signal (~n*spread**2) is at or below the raw-sum noise floor
    # (~4*eps*n*offset**2) here: police_sq_sum warns+clamps or refuses.
    offset = 1e9
    values = [offset + r for r in residuals]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            arm = _arm(values)
        except InvalidRequestError as exc:
            assert exc.code == "estimation.armstats.centered_sum_squares"
            return
    if caught:
        assert set(warning_codes(caught)) & {
            "estimation.armstats.centered_sum_squares_clamped",
            "estimation.armstats.centered_sum_squares_noise_floor",
        }
    # An exactly-equal input needs no warning (already exactly 0); the
    # clamped case can still leave a tiny unpoliced cy1**2/n residual.
    assert arm.cy2 == pytest.approx(0.0, abs=1e-6)
