"""Nox pytest worker selection contracts."""

from __future__ import annotations

from noxfile import _xdist_args


def test_ci_keeps_auto_workers() -> None:
    """CI delegates worker sizing to xdist."""
    assert _xdist_args({"CI": "true", "WORKERS": "2"}, 12) == (
        "-n",
        "auto",
        "--dist",
        "loadgroup",
    )


def test_local_workers_follow_lane_allowance() -> None:
    """Local Nox uses one worker per three CPUs, at least one."""
    assert _xdist_args({}, 12) == ("-n", "4", "--dist", "loadgroup")
    assert _xdist_args({}, 2) == ("-n", "1", "--dist", "loadgroup")


def test_workers_variable_overrides_allowance() -> None:
    """WORKERS overrides the local CPU-derived allowance."""
    assert _xdist_args({"WORKERS": "2"}, 12) == (
        "-n",
        "2",
        "--dist",
        "loadgroup",
    )
