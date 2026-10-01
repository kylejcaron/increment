"""Ratio denominator-precision advisory.

A fixed-horizon unit-grain ratio row whose denominator mean is poorly
resolved in either arm carries a ``ratio_denominator_precision`` note that
names the arm, the statistic and the threshold; a well-resolved denominator
carries none. The note reaches every ingress path that estimates a ratio
metric, and the interval itself is unchanged by it.
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from increment import Analysis
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import estimate_lift
from increment.estimation.variance import (
    RATIO_DENOMINATOR_PRECISION_THRESHOLD,
    ratio_denominator_precision,
)
from increment.frame import MetricSpec
from increment.semantics.models import Measure, RatioMetric
from tests.analysis_factory import lift_rows

_METRIC = RatioMetric(
    name="rpo", entity="user", numerator=Measure(fact="num"), denominator=Measure(fact="den")
)
_SPEC = MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")


def _arm(
    group_id: str, *, n: int, y_bar: float, var_y: float, d_bar: float, var_d: float
) -> ArmStats:
    """An arm with exactly these ddof=1 moments and zero numerator/denominator covariance."""
    return ArmStats.from_raw_sums(
        study_id="e",
        metric="rpo",
        group_id=group_id,
        n=n,
        sum_y=n * y_bar,
        sum_y2=n * y_bar**2 + (n - 1) * var_y,
        sum_den=n * d_bar,
        sum_den2=n * d_bar**2 + (n - 1) * var_d,
        sum_yden=n * y_bar * d_bar,
    )


def _raw_arm(group_id: str, y: np.ndarray, d: np.ndarray) -> ArmStats:
    return ArmStats.from_raw_sums(
        study_id="e",
        metric="rpo",
        group_id=group_id,
        n=len(y),
        sum_y=float(y.sum()),
        sum_y2=float((y * y).sum()),
        sum_den=float(d.sum()),
        sum_den2=float((d * d).sum()),
        sum_yden=float((y * d).sum()),
    )


def _summary(*arms: ArmStats) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": a.study_id,
            "metric": a.metric,
            "group_id": a.group_id,
            "n": float(a.n),
            "ref_y": a.ref_y,
            "cy1": a.cy1,
            "cy2": a.cy2,
            "ref_den": a.ref_den,
            "cden1": a.cden1,
            "cden2": a.cden2,
            "cyden": a.cyden,
        }
        for a in arms
    )


def _row(control: ArmStats, treatment: ArmStats):
    computation = estimate_lift(
        metrics=[_METRIC], summary=_summary(control, treatment), control_group="control"
    )
    assert computation.failures == {}, computation.failures
    (row,) = computation.results
    return row


def test_statistic_is_the_denominator_mean_relative_standard_error():
    arm = _arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.5)
    assert ratio_denominator_precision(arm) == pytest.approx(math.sqrt(4.5 / 50) / 4.0, rel=1e-12)


def test_poorly_resolved_denominators_carry_the_advisory_naming_each_arm():
    # sqrt(4.5 / 50) / 1 = 0.3 in both arms, twice the threshold.
    control = _arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=1.0, var_d=4.5)
    treatment = _arm("treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=1.0, var_d=4.5)
    row = _row(control, treatment)
    assert row.note is not None
    assert row.note.startswith("ratio_denominator_precision:")
    assert "treatment=0.3" in row.note
    assert "control=0.3" in row.note
    assert f"exceeds {RATIO_DENOMINATOR_PRECISION_THRESHOLD:.3g}" in row.note


def test_advisory_names_only_the_arm_that_exceeds_the_threshold():
    # Control resolves its denominator to 0.1 relative SE; treatment to 0.3.
    control = _arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=1.0, var_d=0.5)
    treatment = _arm("treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=1.0, var_d=4.5)
    row = _row(control, treatment)
    assert row.note is not None
    assert "treatment=0.3" in row.note
    assert "control=" not in row.note


def test_well_resolved_denominators_carry_no_advisory():
    # sqrt(1 / 2000) / 1 = 0.022, well under the threshold.
    control = _arm("control", n=2000, y_bar=2.0, var_y=1.0, d_bar=1.0, var_d=1.0)
    treatment = _arm("treatment", n=2000, y_bar=2.2, var_y=1.0, d_bar=1.0, var_d=1.0)
    row = _row(control, treatment)
    assert row.note is None


def test_advisory_leaves_the_interval_unchanged():
    control = _arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=1.0, var_d=4.5)
    treatment = _arm("treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=1.0, var_d=4.5)
    flagged = _row(control, treatment)
    plain = flagged.model_copy(update={"note": None})
    assert flagged.note is not None
    assert flagged.require_lift() == plain.require_lift()
    assert (flagged.abs_lb, flagged.abs_ub) == (plain.abs_lb, plain.abs_ub)
    assert flagged.stat_sig() == plain.stat_sig()


# --- One heavy-denominator dataset through every ingress path ---------------


def _units(n_per_arm: int = 40, seed: int = 4) -> list[dict[str, Any]]:
    """Per-unit revenue and a right-skewed integer session count; seed 4 puts
    both arms' denominator relative SE (0.229, 0.263) above the threshold
    while the combined log-scale SE (0.34) stays inside the admission rule."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for group_id in ("control", "treatment"):
        sessions = 1 + np.round(rng.lognormal(0.0, 1.5, size=n_per_arm)).astype(int)
        revenue = rng.normal(5.0, 1.0, size=n_per_arm)
        rows.extend(
            {
                "unit_id": f"{group_id[0]}{i:03d}",
                "group_id": group_id,
                "sessions": int(sessions[i]),
                "revenue": float(revenue[i]),
            }
            for i in range(n_per_arm)
        )
    return rows


def _relative_se(values: list[int]) -> float:
    array = np.asarray(values, dtype=float)
    return math.sqrt(array.var(ddof=1) / len(array)) / array.mean()


def test_advisory_reaches_the_unit_summary_path():
    unit_rows = _units()
    (row,) = lift_rows(
        Analysis.from_unit_summary(
            pa.Table.from_pylist(unit_rows),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[_SPEC],
        ).run()
    )
    assert row.note is not None
    for group_id in ("control", "treatment"):
        stat = _relative_se([r["sessions"] for r in unit_rows if r["group_id"] == group_id])
        assert stat > RATIO_DENOMINATOR_PRECISION_THRESHOLD
        assert f"{group_id}={stat:.3g}" in row.note


_DEFS = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM rdp_events
    timestamp_column: ts
    entities: [unit_id]
    facts:
      - name: exposure
        column: null
      - name: purchase
        column: value
      - name: session
        column: null
exposures:
  - name: enrolled
    fact: exposure
metrics:
  - name: rps
    type: ratio
    entity: unit_id
    numerator:
      fact: purchase
      aggregation: sum
      window_days: 30
    denominator:
      fact: session
      aggregation: count
      window_days: 30
experiments:
  - name: rdp_test
    exposure: enrolled
    unit: unit_id
    start: 2025-01-01
    end: 2025-02-15
    plan: {secondaries: [rps]}
    control_group: control
"""


def _event_rows(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = datetime(2025, 1, 1, 9, 0, 0)
    out: list[dict[str, Any]] = []
    # Trailing non-enrolled events move each fact's loaded-data bound past every window.
    for event, value in (("session", None), ("purchase", 0.0)):
        out.append(
            {
                "unit_id": "sentinel",
                "ts": datetime(2025, 2, 1),
                "event": event,
                "value": value,
                "experiment_id": "",
                "group_id": None,
            }
        )
    for row in unit_rows:
        base = {"unit_id": row["unit_id"], "experiment_id": "", "group_id": None}
        out.append(
            {
                **base,
                "ts": start,
                "event": "exposure",
                "value": None,
                "experiment_id": "rdp_test",
                "group_id": row["group_id"],
            }
        )
        out.append(
            {**base, "ts": start + timedelta(days=1), "event": "purchase", "value": row["revenue"]}
        )
        out.extend(
            {
                **base,
                "ts": start + timedelta(days=1, minutes=k),
                "event": "session",
                "value": None,
            }
            for k in range(int(row["sessions"]))
        )
    return out


@pytest.mark.slow
def test_advisory_reaches_every_path_that_estimates_a_ratio_metric():
    """Unit summary, unit panel, definitions, unit-day artifact and portable
    moments all carry the same advisory for the same per-unit data."""
    import ibis
    import pyarrow.parquet as pq

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load

    unit_rows = _units()
    (summary_row,) = lift_rows(
        Analysis.from_unit_summary(
            pa.Table.from_pylist(unit_rows),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[_SPEC],
        ).run()
    )
    (panel_row,) = lift_rows(
        Analysis.from_unit_panel(
            pa.Table.from_pylist([{**r, "ds": datetime(2025, 1, 2).date()} for r in unit_rows]),
            unit="unit_id",
            group="group_id",
            date="ds",
            control="control",
            metrics=[_SPEC],
        ).run()
    )

    con = ibis.duckdb.connect()
    con.create_table("rdp_events", pd.DataFrame(_event_rows(unit_rows)))
    tmp = Path(tempfile.mkdtemp())
    defs_path = tmp / "defs.yaml"
    defs_path.write_text(_DEFS)
    native = Analysis("rdp_test", defs_path, con)
    (native_row,) = lift_rows(native.run())

    definitions = load(defs_path)
    experiment = next(item for item in definitions.experiments if item.name == "rdp_test")
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = native.publish_unit_day_artifact(store)
    (artifact_row,) = lift_rows(
        Analysis.from_unit_day_artifact(store, reference, expected_context=context).run()
    )

    export_path = tmp / "moments.parquet"
    native.export(export_path)
    (portable_row,) = lift_rows(
        Analysis.from_moments(
            pq.read_table(export_path).to_pylist(),
            metrics=[MetricSpec(name="rps", type="ratio", numerator="n", denominator="d")],
            control="control",
        ).run()
    )

    assert summary_row.note is not None
    assert summary_row.note.startswith("ratio_denominator_precision:")
    for row in (panel_row, native_row, artifact_row, portable_row):
        assert row.note == summary_row.note
        assert row.require_lift().value == pytest.approx(
            summary_row.require_lift().value, rel=1e-12
        )


# --- Monte Carlo: the advisory fires on the regime it was calibrated for ----


def _advisory_rate(reps: int, seed: int) -> tuple[float, int]:
    """Share of admitted rows carrying the advisory at n=50 with a
    lognormal(sigma=1.5) denominator and an independent numerator, plus the
    admitted-row count (the log-scale admission rule refuses some draws)."""
    rng = np.random.default_rng(seed)
    admitted = fired = 0
    for _ in range(reps):
        arms = []
        for group_id in ("control", "treatment"):
            d = rng.lognormal(0.0, 1.5, size=50)
            y = rng.normal(2.0, 1.0, size=50)
            arms.append(_raw_arm(group_id, y, d))
        computation = estimate_lift(
            metrics=[_METRIC], summary=_summary(*arms), control_group="control"
        )
        if not computation.results:
            continue
        admitted += 1
        note = computation.results[0].note
        fired += note is not None and note.startswith("ratio_denominator_precision:")
    return fired / admitted, admitted


def test_advisory_fires_on_heavy_tailed_small_samples_smoke():
    rate, admitted = _advisory_rate(reps=20, seed=1)
    assert admitted >= 10
    assert rate >= 0.95


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_advisory_fires_on_heavy_tailed_small_samples():
    """At n=50 under lognormal(sigma=1.5) the calibration grid measured 89.6%
    coverage and a 5th-percentile statistic of 0.21, above the 0.15 threshold,
    so the advisory should be nearly universal on admitted rows."""
    rate, admitted = _advisory_rate(reps=500, seed=2)
    assert admitted >= 350
    assert rate >= 0.95, f"advisory fired on {rate:.3f} of {admitted} admitted rows"
