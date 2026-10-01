"""Pins every readout's numbers on the shipped example seed."""

import json
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "parity_baseline.json"

READOUTS = ("run", "run_daily", "run_asof", "run_breakout", "run_daily_lift", "run_asof_lift")


def load_parity_baseline(readout: str) -> dict[str, float | None]:
    """Key -> value for one readout, as the library currently answers it.

    Keys are pipe-joined, e.g. ``"purchase_rate|treatment"`` for ``run`` and
    ``"purchase_rate|treatment|2025-01-15"`` for the day-axis readouts. The two lift
    readouts (``run_daily_lift``, ``run_asof_lift``) carry a fourth
    component, the estimand, e.g. ``"purchase_rate|treatment|2025-01-15|itt"``.

    Regenerate with ``scripts/capture_parity_baseline.py`` - and read its
    module docstring first: a moved number is a semantics change shipping
    to every user, so it needs a reason on record before it is pinned.
    """
    if readout not in READOUTS:
        raise KeyError(f"{readout!r} not in {READOUTS}")
    return json.loads(_FIXTURE.read_text())[readout]


@pytest.mark.parametrize("readout", READOUTS)
def test_baseline_covers_every_readout(readout: str) -> None:
    """The spine feeds ALL of these - not just run().

    A silently-empty section would make that readout's gate vacuously
    pass, which is worse than having no gate at all.
    """
    rows = load_parity_baseline(readout)
    assert rows, f"{readout} section is empty -- the baseline is not usable"
    assert all(v is None or isinstance(v, (int, float)) for v in rows.values())


def test_baseline_spans_several_metrics() -> None:
    """Guards against a fixture captured with a truncated metric list."""
    metrics = {k.split("|", 1)[0] for k in load_parity_baseline("run")}
    assert len(metrics) >= 3, f"expected several metrics, got {metrics}"


def test_day_axis_baseline_is_substantial() -> None:
    """The day-axis readouts are where the spine's right edge shows up most."""
    assert len(load_parity_baseline("run_daily")) > len(load_parity_baseline("run"))


def test_parity_baseline_day_one_purchase_rate_is_unavailable() -> None:
    """Day-1 as-of purchase_rate is 0/0 in the seeded DGP - pin it so a
    regeneration that silently zero-fills gets caught.
    """
    for readout_name in ("run_asof", "run_daily"):
        base = load_parity_baseline(readout_name)
        for g in ("control", "treatment"):
            assert base[f"purchase_rate|{g}|2025-01-15"] is None


def test_parity_baseline_day_one_lift_purchase_rate_is_unavailable() -> None:
    """The lift readouts inherit the same day-1 0/0 purchase_rate - pin it
    so a regeneration that silently zero-fills gets caught here, not only by
    the slow baseline comparison in test_analysis.py.
    """
    for readout_name in ("run_daily_lift", "run_asof_lift"):
        base = load_parity_baseline(readout_name)
        assert base["purchase_rate|treatment|2025-01-15|itt"] is None
