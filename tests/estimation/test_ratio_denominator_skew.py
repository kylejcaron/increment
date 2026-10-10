"""Ratio denominator-skew coverage warning.

A fixed-horizon unit-grain ratio row whose denominator mean is skewed beyond
the measured under-covering region in either arm raises the coded
``estimation.engine.ratio_denominator_skew`` warning per flagged arm and
carries a ``ratio_denominator_skew`` note naming each; a row whose moments
lack the denominator's third moment says the check was unavailable instead
of passing silently. The decision is the same through every ingress path
that carries unit-level data, and the interval is unchanged by it.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy.stats import skew

from increment import Analysis
from increment.errors import IncrementWarning, WireFormatError
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.variance import (
    RATIO_DENOMINATOR_SKEW_THRESHOLD,
    ratio_denominator_mean_skewness,
)
from increment.frame import MetricSpec
from increment.semantics.models import Measure, RatioMetric
from tests.analysis_factory import lift_rows
from tests.estimation.test_ratio_denominator_precision import _DEFS, _event_rows
from tests.ratio_arms import array_arm, moment_arm, summary
from tests.warning_codes import warning_codes, warning_context

_CODE = "estimation.engine.ratio_denominator_skew"
_METRIC = RatioMetric(
    name="rpo", entity="user", numerator=Measure(fact="num"), denominator=Measure(fact="den")
)
_SPEC = MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")


def _flagged(record: list[warnings.WarningMessage]) -> dict[str, Mapping[str, object]]:
    """Context of every denominator-skew warning in *record*, keyed by arm."""
    return {
        str(w.message.context["arm"]): w.message.context
        for w in record
        if isinstance(w.message, IncrementWarning) and w.message.code == _CODE
    }


def _estimate(control: ArmStats, treatment: ArmStats, **kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        computation = estimate_lift(
            metrics=[_METRIC],
            summary=summary(control, treatment),
            control_group="control",
            **kwargs,
        )
    assert computation.failures == {}, computation.failures
    return computation.results, record


def test_statistic_is_the_denominator_mean_skewness():
    arm = moment_arm("control", n=64, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.5, skew_d=4.0)
    assert arm.skew_den() == pytest.approx(4.0, rel=1e-12)
    assert ratio_denominator_mean_skewness(arm) == pytest.approx(0.5, rel=1e-12)


def test_flagged_arm_raises_the_coded_warning_and_the_row_names_it():
    control = moment_arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.0)
    treatment = moment_arm(
        "treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=4.0, var_d=4.0, skew_d=4.0
    )
    (row,), record = _estimate(control, treatment)
    assert warning_codes(record) == [_CODE]
    context = warning_context(record, _CODE)
    assert context["metric"] == "rpo"
    assert context["arm"] == "treatment"
    assert context["n"] == 50
    assert context["skewness"] == pytest.approx(4.0, rel=1e-12)
    assert context["statistic"] == pytest.approx(4.0 / math.sqrt(50), rel=1e-12)
    assert context["threshold"] == RATIO_DENOMINATOR_SKEW_THRESHOLD
    assert row.note is not None
    assert row.note.startswith("ratio_denominator_skew:")
    assert "treatment=" in row.note
    assert "control=" not in row.note


def test_both_flagged_arms_raise_one_warning_each():
    control = moment_arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.0, skew_d=3.0)
    treatment = moment_arm(
        "treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=4.0, var_d=4.0, skew_d=4.0
    )
    (row,), record = _estimate(control, treatment)
    assert warning_codes(record) == [_CODE, _CODE]
    assert row.note is not None and "treatment=" in row.note and "control=" in row.note


def test_missing_arm_does_not_suppress_warning_for_the_other_arm():
    control = moment_arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.0).model_copy(
        update={"cden3": None}
    )
    treatment = moment_arm(
        "treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=4.0, var_d=4.0, skew_d=4.0
    )

    (row,), record = _estimate(control, treatment)

    assert set(_flagged(record)) == {"treatment"}
    assert row.note is not None
    assert "treatment=" in row.note
    assert "ratio_denominator_skew: unavailable" in row.note
    assert "control" in row.note


def test_symmetric_denominators_raise_no_warning_and_carry_no_note():
    control = moment_arm("control", n=2000, y_bar=2.0, var_y=1.0, d_bar=1.0, var_d=1.0)
    treatment = moment_arm("treatment", n=2000, y_bar=2.2, var_y=1.0, d_bar=1.0, var_d=1.0)
    (row,), record = _estimate(control, treatment)
    assert warning_codes(record) == []
    assert row.note is None


def test_warning_leaves_the_interval_unchanged():
    control = moment_arm("control", n=50, y_bar=2.0, var_y=1.0, d_bar=4.0, var_d=4.0)
    plain = moment_arm("treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=4.0, var_d=4.0)
    skewed = plain.model_copy(
        update={
            "cden3": moment_arm(
                "treatment", n=50, y_bar=2.2, var_y=1.0, d_bar=4.0, var_d=4.0, skew_d=4.0
            ).cden3
        }
    )
    (unflagged,), _ = _estimate(control, plain)
    (flagged,), record = _estimate(control, skewed)
    assert warning_codes(record) == [_CODE]
    assert flagged.require_lift().value == unflagged.require_lift().value
    assert flagged.require_lift().lb == unflagged.require_lift().lb
    assert flagged.require_lift().ub == unflagged.require_lift().ub
    assert flagged.stat_sig() == unflagged.stat_sig()


def test_rows_without_the_third_moment_say_the_check_is_unavailable():
    def raw(group_id: str, y_bar: float) -> ArmStats:
        return ArmStats.from_raw_sums(
            study_id="e",
            metric="rpo",
            group_id=group_id,
            n=50,
            sum_y=50 * y_bar,
            sum_y2=50 * y_bar**2 + 49.0,
            sum_den=200.0,
            sum_den2=50 * 16.0 + 49 * 4.0,
            sum_yden=50 * y_bar * 4.0,
        )

    control, treatment = raw("control", 2.0), raw("treatment", 2.2)
    assert control.cden3 is None
    (row,), record = _estimate(control, treatment)
    assert warning_codes(record) == []
    assert row.note is not None
    assert "ratio_denominator_skew: unavailable" in row.note
    assert "treatment" in row.note and "control" in row.note
    with pytest.raises(Exception) as refused:
        control.skew_den()
    assert refused.value.code == "estimation.armstats.arm_stats.skew_den_needs_cden3"  # ty: ignore[unresolved-attribute]


def _ratio_arm_with_covariate(group_id: str, rng: np.random.Generator, n: int) -> ArmStats:
    cov = rng.normal(5.0, 1.5, n)
    den = rng.lognormal(0.0, 1.5, n) + 0.2 * (cov - 5.0) ** 2
    num = rng.normal(2.0, 1.0, n) + 0.3 * (cov - 5.0)
    base = array_arm(group_id, num, den)
    dx, dden = cov - cov.mean(), den - den.mean()
    return base.model_copy(
        update={
            "ref_x": float(cov.mean()),
            "cx1": float(dx.sum()),
            "cx2": float((dx * dx).sum()),
            "cxy": float((dx * (num - num.mean())).sum()),
            "x_role": "covariate",
            "cxden": float((dx * dden).sum()),
        }
    )


def test_cuped_ratio_rows_carry_the_same_advisory_once_per_arm():
    rng = np.random.default_rng(3)
    control = _ratio_arm_with_covariate("control", rng, 60)
    treatment = _ratio_arm_with_covariate("treatment", rng, 60)
    assert (
        max(ratio_denominator_mean_skewness(control), ratio_denominator_mean_skewness(treatment))
        > RATIO_DENOMINATOR_SKEW_THRESHOLD
    )
    rows, record = _estimate(
        control,
        treatment,
        methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
    )
    assert {row.method for row in rows} == {"unadjusted", "cuped"}
    notes = {row.note for row in rows}
    assert len(notes) == 1 and "ratio_denominator_skew:" in next(iter(notes))
    assert len(_flagged(record)) == sum(code == _CODE for code in warning_codes(record)) >= 1


def _units(n_per_arm: int = 40, seed: int = 4) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
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


def test_unit_summary_path_decides_from_the_per_unit_skewness():
    unit_rows = _units()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        (row,) = lift_rows(
            Analysis.from_unit_summary(
                pa.Table.from_pylist(unit_rows),
                unit="unit_id",
                group="group_id",
                control="control",
                metrics=[_SPEC],
            ).run()
        )
    flagged = _flagged(record)
    assert row.note is not None and "ratio_denominator_skew:" in row.note
    for group_id in ("control", "treatment"):
        values = np.array([r["sessions"] for r in unit_rows if r["group_id"] == group_id], float)
        g1 = float(skew(values, bias=True))
        if g1 / math.sqrt(len(values)) > RATIO_DENOMINATOR_SKEW_THRESHOLD:
            assert flagged[group_id]["n"] == len(values)
            assert flagged[group_id]["skewness"] == pytest.approx(g1, rel=1e-9)
            assert f"{group_id}=" in row.note
        else:
            assert group_id not in flagged
    assert flagged


def test_clustered_ratio_rows_claim_no_coverage_check():
    rng = np.random.default_rng(5)
    rows = []
    for group_id in ("control", "treatment"):
        for store in range(25):
            for unit in range(4):
                rows.append(
                    {
                        "unit_id": f"{group_id[0]}{store:02d}{unit}",
                        "group_id": group_id,
                        "store": f"{group_id[0]}s{store:02d}",
                        "sessions": int(1 + round(rng.lognormal(0.0, 1.5))),
                        "revenue": float(rng.normal(5.0, 1.0)),
                    }
                )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        (row,) = lift_rows(
            Analysis.from_unit_summary(
                pa.Table.from_pylist(rows),
                unit="unit_id",
                group="group_id",
                control="control",
                metrics=[_SPEC],
                cluster="store",
            ).run()
        )
    assert _CODE not in warning_codes(record)
    assert row.note is None or "ratio_denominator_skew" not in row.note
    assert row.n_clusters == 50


@pytest.mark.slow
def test_decision_and_context_agree_across_every_ingress_path():
    """Unit summary, unit panel, definitions, unit-day artifact and portable
    moments raise the same warnings (arm, n, skewness) and carry the same note
    for the same per-unit data; a preceding-format cube without the third
    moment says the check is unavailable, and a current-format cube without
    it is refused."""
    import ibis
    import pyarrow.parquet as pq

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load

    unit_rows = _units()

    def captured(build):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            (row,) = lift_rows(build().run())
        flagged = _flagged(record)
        return row, flagged

    summary_row, summary_flags = captured(
        lambda: Analysis.from_unit_summary(
            pa.Table.from_pylist(unit_rows),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[_SPEC],
        )
    )
    assert summary_flags
    assert summary_row.note is not None and "ratio_denominator_skew:" in summary_row.note
    panel_row, panel_flags = captured(
        lambda: Analysis.from_unit_panel(
            pa.Table.from_pylist([{**r, "ds": datetime(2025, 1, 2).date()} for r in unit_rows]),
            unit="unit_id",
            group="group_id",
            date="ds",
            control="control",
            metrics=[_SPEC],
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("rdp_events", pd.DataFrame(_event_rows(unit_rows)))
    tmp = Path(tempfile.mkdtemp())
    defs_path = tmp / "defs.yaml"
    defs_path.write_text(_DEFS)
    native = Analysis("rdp_test", defs_path, con)
    native_row, native_flags = captured(lambda: native)
    definitions = load(defs_path)
    experiment = next(item for item in definitions.experiments if item.name == "rdp_test")
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = native.publish_unit_day_artifact(store)
    artifact_row, artifact_flags = captured(
        lambda: Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    )
    export_path = tmp / "moments.parquet"
    native.export(export_path)
    exported = pq.read_table(export_path).to_pylist()
    replay_spec = [MetricSpec(name="rps", type="ratio", numerator="n", denominator="d")]
    portable_row, portable_flags = captured(
        lambda: Analysis.from_moments(exported, metrics=replay_spec, control="control")
    )

    for row, flags in (
        (panel_row, panel_flags),
        (native_row, native_flags),
        (artifact_row, artifact_flags),
        (portable_row, portable_flags),
    ):
        assert row.note == summary_row.note
        assert flags.keys() == summary_flags.keys()
        for arm, context_ in flags.items():
            assert context_["n"] == summary_flags[arm]["n"]
            assert context_["skewness"] == pytest.approx(summary_flags[arm]["skewness"], rel=1e-9)
        assert row.require_lift().value == pytest.approx(
            summary_row.require_lift().value, rel=1e-12
        )

    previous = [{**r, "cden3": None, "moments_format": 11} for r in exported]
    previous_row, previous_flags = captured(
        lambda: Analysis.from_moments(previous, metrics=replay_spec, control="control")
    )
    assert previous_flags == {}
    assert previous_row.note is not None
    assert "ratio_denominator_skew: unavailable" in previous_row.note
    assert previous_row.require_lift().value == pytest.approx(
        summary_row.require_lift().value, rel=1e-12
    )

    overflowed = [{**r, "cden3": None} for r in exported]
    overflowed_row, overflowed_flags = captured(
        lambda: Analysis.from_moments(overflowed, metrics=replay_spec, control="control")
    )
    assert overflowed_flags == {}
    assert overflowed_row.note is not None
    assert "ratio_denominator_skew: unavailable" in overflowed_row.note
    assert overflowed_row.require_lift().value == pytest.approx(
        summary_row.require_lift().value, rel=1e-12
    )

    with pytest.raises(WireFormatError) as refused:
        Analysis.from_moments(
            [{key: value for key, value in row.items() if key != "cden3"} for row in exported],
            metrics=replay_spec,
            control="control",
        )
    assert refused.value.code == "moments.denominator_third_moment_missing"


def _flag_rate(reps: int, seed: int, *, n: int, sigma: float) -> tuple[float, int]:
    """Share of admitted rows warned, plus the admitted count, under an
    independent Normal(2, 1) numerator and a lognormal(sigma) denominator."""
    rng = np.random.default_rng(seed)
    admitted = fired = 0
    for _ in range(reps):
        arms = [
            array_arm(group_id, rng.normal(2.0, 1.0, size=n), rng.lognormal(0.0, sigma, size=n))
            for group_id in ("control", "treatment")
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            computation = estimate_lift(
                metrics=[_METRIC], summary=summary(*arms), control_group="control"
            )
        if not computation.results:
            continue
        admitted += 1
        fired += _CODE in warning_codes(record)
    return fired / admitted, admitted


def test_warning_fires_on_the_measured_region_and_not_on_a_benign_one_smoke():
    rate, admitted = _flag_rate(reps=20, seed=1, n=50, sigma=1.5)
    assert admitted >= 10
    assert rate >= 0.9
    rate, admitted = _flag_rate(reps=20, seed=2, n=400, sigma=0.5)
    assert admitted == 20
    assert rate <= 0.05


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_warning_fires_on_the_measured_region_and_not_on_a_benign_one():
    """The calibration grid (6,000 replications per cell) flagged 98.7% of
    admitted rows at n=50 under lognormal(1.5) and 0.0% at n=400 under
    lognormal(0.5)."""
    rate, admitted = _flag_rate(reps=500, seed=3, n=50, sigma=1.5)
    assert admitted >= 350
    assert rate >= 0.95, f"warned on {rate:.3f} of {admitted} admitted rows"
    rate, admitted = _flag_rate(reps=500, seed=4, n=400, sigma=0.5)
    assert rate <= 0.02, f"warned on {rate:.3f} of {admitted} benign rows"


def test_rounded_mean_two_unit_arm_keeps_a_feasible_third_moment():
    """Two values are symmetric about their mean, so the exact bound is 0; the
    producer's rounding of the mean must not make the arm refuse or the
    diagnostic claim inconsistency."""
    arm = array_arm("control", np.array([2.0, 2.5]), np.array([1.0, 1.0 + 2.0**-25 + 2.0**-52]))
    assert arm.skew_den_unavailable() is None
    assert arm.skew_den() == pytest.approx(0.0, abs=1e-6)


def test_overflowing_third_moment_is_a_diagnostic_gap_not_a_refusal():
    """A denominator whose cubed residuals overflow float64 still has sound
    second moments; the row estimates, carries the unavailable note and
    raises no warning."""
    rng = np.random.default_rng(9)
    huge = np.full(50, 1e103)
    huge[0] = 3e103
    with np.errstate(over="ignore"):
        control = array_arm("control", rng.normal(2.0, 1.0, 50), huge)
        treatment = array_arm(
            "treatment", rng.normal(2.2, 1.0, 50), np.full(50, 1e103) + rng.normal(0, 1e102, 50)
        )
    assert control.cden3 is not None and not math.isfinite(control.cden3)
    assert control.skew_den_unavailable() == "inconsistent"
    (row,), record = _estimate(control, treatment)
    assert warning_codes(record) == []
    assert row.note is not None and "ratio_denominator_skew: unavailable" in row.note
    assert "control" in row.note
    with pytest.raises(Exception) as refused:
        control.skew_den()
    assert refused.value.code == "estimation.armstats.arm_stats.skew_den_inconsistent"  # ty: ignore[unresolved-attribute]
