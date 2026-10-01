"""Day-grain partitions combine to the single-pass record.

``daily_group_summary`` emits one centered record per (day, arm): each is a
PARTITION in the combination sense, referenced on its own day's mean. The
reduction across days is the delta expansion on ``ArmStats.combine``, and it
must land on exactly what a single pass over the pooled rows would produce.

The offset case is the one that matters: at mu ~ 1 the ``n_p * delta_p**2``
terms are rounding-scale and a wrong (or missing) expansion is invisible.
"""

from __future__ import annotations

import datetime as dt

import ibis
import numpy as np
import pytest

from increment.estimation.armstats import ArmStats
from increment.query.builders import daily_group_summary
from increment.semantics.models import Measure, RatioMetric

N_DAYS = 6
N_UNITS = 40


def _panel(offset: float) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """A dense unit x day panel whose per-day means genuinely differ.

    The day-to-day drift is what makes ``delta_p = ref_p - R`` nonzero, so
    the combination is exercised rather than reduced to a plain sum.
    """
    rng = np.random.default_rng(5)
    rows: list[dict] = []
    value: list[float] = []
    den: list[float] = []
    day_index: list[int] = []
    for day in range(N_DAYS):
        drift = 12.0 * day  # per-day mean shift
        for unit in range(N_UNITS):
            v = offset + drift + float(rng.normal(0.0, 1.0))
            w = offset + 3.0 * day + float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "unit_id": f"u{unit:03d}",
                    "ds": dt.date(2025, 1, 1) + dt.timedelta(days=day),
                    "experiment_id": "e",
                    "metric": "m",
                    "group_id": "treatment" if unit % 2 else "control",
                    "n_events": 1,
                    "sum_value": v,
                    "min_value": v,
                    "max_value": v,
                    "sum_value_den": w,
                    "min_value_den": w,
                    "max_value_den": w,
                }
            )
            value.append(v)
            den.append(w)
            day_index.append(day)
    return rows, np.array(value), np.array(den), np.array(day_index)


def _single_pass(y: np.ndarray, den: np.ndarray, group_id: str) -> ArmStats:
    ry, rd = float(np.mean(y)), float(np.mean(den))
    return ArmStats(
        study_id="e",
        metric="m",
        group_id=group_id,
        n=len(y),
        ref_y=ry,
        cy1=float(np.sum(y - ry)),
        cy2=float(np.sum((y - ry) ** 2)),
        ref_den=rd,
        cden1=float(np.sum(den - rd)),
        cden2=float(np.sum((den - rd) ** 2)),
        cyden=float(np.sum((y - ry) * (den - rd))),
    )


_RATIO_METRIC = RatioMetric(
    name="m",
    entity="unit_id",
    numerator=Measure(fact="num", aggregation="sum"),
    denominator=Measure(fact="den", aggregation="sum"),
)


def _daily_arms(rows: list[dict]) -> dict[str, list[ArmStats]]:
    panel = ibis.memtable(rows).drop("sum_value_den", "min_value_den", "max_value_den")
    den_panel = ibis.memtable(rows).select(
        "unit_id",
        "ds",
        "experiment_id",
        "metric",
        "group_id",
        "n_events",
        sum_value="sum_value_den",
        min_value="min_value_den",
        max_value="max_value_den",
    )
    out = (
        daily_group_summary(panel, metric=_RATIO_METRIC, den_panel=den_panel)
        .to_pyarrow()
        .to_pylist()
    )
    by_group: dict[str, list[ArmStats]] = {}
    for r in sorted(out, key=lambda r: (str(r["group_id"]), r["ds"])):
        by_group.setdefault(str(r["group_id"]), []).append(
            ArmStats(
                study_id=str(r["experiment_id"]),
                metric=str(r["metric"]),
                group_id=str(r["group_id"]),
                n=int(r["n"]),
                ref_y=float(r["ref_y"]),
                cy1=float(r["cy1"]),
                cy2=float(r["cy2"]),
                ref_den=float(r["ref_den"]),
                cden1=float(r["cden1"]),
                cden2=float(r["cden2"]),
                cyden=float(r["cyden"]),
            )
        )
    return by_group


REDUCTIONS = ("mean_y", "mean_den", "var_y", "var_den", "cov_yden")


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_daily_partitions_combine_to_the_single_pass_record(offset: float) -> None:
    rows, y, den, _ = _panel(offset)
    arms = _daily_arms(rows)
    assert set(arms) == {"control", "treatment"}
    assert all(len(v) == N_DAYS for v in arms.values())
    # Without a real spread between the day references the expansion would
    # collapse to a plain sum of centered fields and prove nothing.
    refs = [a.ref_y for a in arms["control"]]
    assert max(refs) - min(refs) > 50.0

    is_treatment = np.array([int(r["unit_id"][1:]) % 2 == 1 for r in rows])
    for group, mask in (("control", ~is_treatment), ("treatment", is_treatment)):
        combined = ArmStats.combine(arms[group])
        whole = _single_pass(y[mask], den[mask], group)
        assert combined.n == int(mask.sum())
        for name in REDUCTIONS:
            got, want = getattr(combined, name)(), getattr(whole, name)()
            assert got == pytest.approx(want, rel=1e-12), f"{group}/{name}"
