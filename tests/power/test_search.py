"""Shared power-solver search helpers: float64 ordinal codec, bisection, null/reason invariant."""

from increment.power._search import (
    bisect_first_true,
    float_from_ordinal,
    float_ordinal,
    require_reason_when_null,
)


def test_float_ordinal_round_trips_and_totally_orders_the_float_line():
    values = [-1e300, -1.5, -0.0, 0.0, 1e-300, 1.5, 1e300]
    ordinals = [float_ordinal(v) for v in values]
    assert ordinals == sorted(ordinals)
    for v, o in zip(values, ordinals, strict=True):
        assert float_from_ordinal(o) == v


def test_bisect_first_true_finds_the_exact_crossing():
    # accept(n) is False for n < 7, True for n >= 7 -- the classic contract.
    assert bisect_first_true(0, 20, lambda n: n >= 7) == 7
    assert bisect_first_true(7, 7, lambda n: True) == 7


def test_require_reason_when_null_accepts_a_value_with_no_reason():
    calls = []
    require_reason_when_null(1.0, None, name="mde", invalid=calls.append)
    assert calls == []


def test_require_reason_when_null_accepts_a_null_with_a_reason():
    calls = []
    require_reason_when_null(None, "unattained", name="mde", invalid=calls.append)
    assert calls == []


def test_require_reason_when_null_rejects_both_set():
    calls = []
    require_reason_when_null(1.0, "unattained", name="mde", invalid=calls.append)
    assert len(calls) == 1


def test_require_reason_when_null_rejects_neither_set():
    calls = []
    require_reason_when_null(None, None, name="mde", invalid=calls.append)
    assert len(calls) == 1
