"""A logged assignment does not establish that its response was measured."""

from __future__ import annotations

import json
from dataclasses import replace

import polars as pl
import pyarrow as pa
import pytest

from increment import (
    MetricSpec,
    SwitchbackObservationSchedule,
    from_switchback_frame,
    run,
    workflows,
)
from increment.errors import CapabilityError, InvalidRequestError
from increment.kernel.experiment import Experiment
from increment.power.switchback import SwitchbackBaseline
from increment.store import ParquetResultsStore
from tests.analysis_factory import view_readout
from tests.test_frame_switchback import _assignment, _identification


def _log() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"unit": unit, "cycle": c, "period": p, "arm": arm}
            for u, unit in enumerate(("a", "b"))
            for c in range(2)
            for p in range(2)
            for arm in [("control", "treatment")[(u + c + p) % 2]]
        ]
    )


def _schedule(*events: str) -> SwitchbackObservationSchedule:
    return SwitchbackObservationSchedule(
        assignments=_log(), units=("a", "b"), cycles=2, complete_event_metrics=events
    )


def _observed() -> pl.DataFrame:
    # One washout and one residual carryover step, then one retained step.
    return (
        _log()
        .join(pl.DataFrame({"step": [0, 1, 2]}), how="cross")
        .with_columns(
            pl.when(pl.col("arm") == "treatment")
            .then(pl.when(pl.col("unit") == "a").then(2).otherwise(4))
            .otherwise(0)
            .alias("orders")
        )
    )


def _build(rows, schedule, *metrics, assignment=None):
    return from_switchback_frame(
        rows,
        schedule=schedule,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        metrics=metrics or {"orders": "mean"},
        identification=_identification(),
        assignment=assignment or _assignment(p_ct=0.5, carryover_order=1),
    )


@pytest.mark.parametrize("discard", ["washout", "carryover"])
def test_sparse_events_with_large_discarded_window_keep_estimates_and_schedule_counts(discard):
    discarded = 2**52
    washout = discarded if discard == "washout" else 0
    carryover = discarded - washout
    assignment = _assignment(
        washout=washout, observation=carryover + 1, carryover_order=carryover, p_ct=0.5
    )
    cycles = pl.DataFrame({"cycle": range(64)})
    log = _log().filter(pl.col("cycle") == 0).drop("cycle").join(cycles, how="cross")
    schedule = replace(_schedule("orders"), assignments=log, cycles=64)
    rows = (
        _observed()
        .filter((pl.col("cycle") == 0) & (pl.col("step") == 2) & (pl.col("orders") != 0))
        .with_columns(pl.lit(discarded).alias("step"))
        .drop("cycle")
        .join(cycles, how="cross")
    )
    events = MetricSpec(name="orders", missing="zero")
    experiment, reader = _build(rows, schedule, events, assignment=assignment)
    with reader:
        (result,) = view_readout(experiment, reader)
        diagnostic = workflows.assignment_diagnostic(experiment, reader)
    assert (result.estimate.value, result.standard_error) == pytest.approx((3, 1))
    assert (result.n_units, result.n_cycles, result.retained_steps) == (2, 128, 1)
    assert diagnostic.washout_rows == 256 * washout
    assert diagnostic.observation_rows == 256 * (carryover + 1)
    assert diagnostic.retained_rows == 256


@pytest.mark.parametrize(
    "axis,field",
    [("cycle", "value_column"), ("step", "numerator"), ("period", "denominator")],
)
def test_schedule_axes_are_not_response_columns(axis, field):
    rows = _observed().with_columns(pl.col("period").alias("visits"))
    spec = (
        MetricSpec(name="value", value_column="orders")
        if field == "value_column"
        else MetricSpec(name="rate", type="ratio", numerator="orders", denominator="visits")
    )
    for schedule in (None, _schedule()):
        with pytest.raises(InvalidRequestError) as caught:
            _build(rows, schedule, spec.model_copy(update={field: axis}))
        assert caught.value.code == "source.frame.switchback.columns"
        assert caught.value.context["reason"] == "metric_overlaps_column"
        assert caught.value.context["name"] == axis
        if field == "denominator":
            experiment, reader = _build(rows, schedule, spec)
            with reader:
                (result,) = view_readout(experiment, reader)
            # Mean orders are 3 and 0; the visit count averages 1/2 in each arm.
            assert result.estimate.value == pytest.approx(6)


@pytest.mark.parametrize("loss", ["last_cycle", "whole_unit", "unlogged_arm"])
def test_observations_cannot_supply_a_missing_assignment(loss):
    log = _log()
    if loss == "last_cycle":
        log = log.filter(pl.col("cycle") == 0)
        expected = ("a", 1, 0)
    elif loss == "whole_unit":
        log = log.filter(pl.col("unit") == "a")
        expected = ("b", 0, 0)
    else:
        log = log.with_columns(
            pl.when(pl.col("unit") == "a").then(None).otherwise(pl.col("arm")).alias("arm")
        )
        expected = ("a", 0, 0)
    with pytest.raises(InvalidRequestError) as caught:
        _build(_observed(), replace(_schedule(), assignments=log))
    assert caught.value.code == "source.frame.switchback.assignment_unrecorded"
    assert tuple(caught.value.context[k] for k in ("unit", "cycle", "period")) == expected


@pytest.mark.parametrize("loss", ["last_cycle", "whole_unit", "retained_step", "all_null"])
def test_known_assignment_requires_every_retained_response(loss):
    rows = _observed()
    if loss == "last_cycle":
        rows = rows.filter(pl.col("cycle") == 0)
        expected = ("a", 1, 0, 2)
    elif loss == "whole_unit":
        rows = rows.filter(pl.col("unit") == "a")
        expected = ("b", 0, 0, 2)
    elif loss == "retained_step":
        rows = rows.filter(pl.col("step") != 2)
        expected = ("a", 0, 0, 2)
    else:
        rows = rows.with_columns(pl.lit(None).alias("orders"))
        expected = ("a", 0, 0, 2)
    experiment, reader = _build(rows.reverse(), _schedule())
    with reader:
        (cell,) = run(experiment, reader).cells
    status, _ = cell.components["sampling"]
    assert status.code == "source.frame.switchback.response_unobserved"
    assert status.context == {
        "metric": "orders",
        "reason": "unobserved_response",
        "missing_count": 8 if loss in ("all_null", "retained_step") else 4,
        "first_missing_key": expected,
        "arm": "control" if loss in ("all_null", "retained_step") else "treatment",
    }
    assert cell.summary is not None
    assert cell.summary.point is cell.summary.lower is cell.summary.upper is None


@pytest.mark.parametrize(
    "defect", ["duplicate", "outside", "contradiction", "log_duplicate", "invalid_order"]
)
def test_observation_keys_and_labels_corroborate_the_log_even_in_washout(defect):
    rows = _observed()
    schedule = _schedule()
    first = rows.head(1)
    if defect == "duplicate":
        rows = pl.concat([rows, first])
        reason = "duplicate_schedule_cells"
    elif defect == "outside":
        rows = pl.concat([rows, first.with_columns(pl.lit("absent").alias("unit"))])
        reason = "off_schedule_observation"
    elif defect == "contradiction":
        rows = pl.concat([first.with_columns(pl.lit("treatment").alias("arm")), rows.slice(1)])
        reason = "observation_arm_mismatch"
    elif defect == "log_duplicate":
        schedule = replace(schedule, assignments=pl.concat([_log(), _log().head(1)]))
        reason = "duplicate_assignment_cells"
    else:
        # The randomization checks still run when every measurement is absent.
        rows = rows.clear()
        schedule = replace(
            schedule, assignments=_log().with_columns(pl.lit("control").alias("arm"))
        )
        reason = "invalid_period_assignment"
    with pytest.raises(InvalidRequestError) as caught:
        _build(rows, schedule)
    assert caught.value.code == "source.frame.switchback.schedule"
    assert caught.value.context["reason"] == reason


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
def test_pandas_sparse_event_alignment_preserves_neighboring_integers(dtype):
    rows = _observed().filter(pl.col("step") == 2).drop("arm").reverse().to_pandas()
    rows["orders"] = (2**55 + rows["orders"].astype("int64") // 2).astype(dtype)
    metric = MetricSpec(name="events", value_column="orders", missing="zero")
    absent_unit = _log().filter(pl.col("unit") == "a").with_columns(pl.lit("c").alias("unit"))
    schedule = replace(
        _schedule("events"),
        units=("c", "b", "a"),
        assignments=pl.concat([_log(), absent_unit]).reverse(),
    )
    experiment, reader = _build(rows, schedule, metric)
    with reader:
        (result,) = view_readout(experiment, reader)
    # The two measured units have exact contrasts 1 and 2; the absent event feed adds 0.
    assert (result.estimate.value, result.standard_error) == pytest.approx((1, (1 / 3) ** 0.5))
    assert (result.n_units, result.n_cycles, result.retained_steps) == (3, 6, 1)


def test_arrow_alignment_preserves_healthy_sibling_beside_null_payload():
    rows = _observed().filter(pl.col("step") == 2).slice(1).to_arrow()
    rows = rows.append_column("unknown value", pa.nulls(len(rows)))
    measured = MetricSpec(name="sales", value_column="orders", missing="zero")
    missing = MetricSpec(name="returns", value_column="unknown value")
    experiment, reader = _build(rows, _schedule("sales"), measured, missing)
    with reader:
        cells = {cell.identity.key.metric: cell for cell in run(experiment, reader).cells}
    summary = cells["sales"].summary
    assert summary is not None and summary.point == pytest.approx(3)
    status, _ = cells["returns"].components["sampling"]
    assert (status.code, status.context["missing_count"]) == (
        "source.frame.switchback.response_unobserved",
        8,
    )


def test_absent_events_equal_observed_zeros_but_change_the_portable_policy():
    rows = _observed().filter(pl.col("step") == 2)
    strict = MetricSpec(name="sales", value_column="orders", preferred_direction="decrease")
    events = strict.model_copy(update={"missing": "zero"})
    experiment, reader = _build(rows, _schedule(), strict)
    with reader:
        (measured,) = view_readout(experiment, reader)
    sparse, reader = _build(
        rows.filter(pl.col("orders") != 0).reverse(), _schedule("sales"), events
    )
    with reader:
        (event_result,) = view_readout(sparse, reader)
    assert (event_result.estimate.value, event_result.standard_error) == pytest.approx((3, 1))
    assert event_result.estimate == measured.estimate
    assert sparse.metrics == experiment.metrics
    assert sparse.source_refs[0].expected_digest != experiment.source_refs[0].expected_digest
    assert sparse.design["observation_design"] == {
        "units": ("a", "b"),
        "cycles": 2,
        "complete_event_metrics": ("sales",),
    }
    assert Experiment.from_json(sparse.to_json()).hash() == sparse.hash() != experiment.hash()
    reordered, reader = _build(
        rows, replace(_schedule("sales"), units=("b", "a"), assignments=_log().reverse()), events
    )
    with reader:
        assert reordered.hash() == sparse.hash()
    swapped = replace(
        _schedule("sales"),
        assignments=_log().with_columns(
            pl.when(pl.col("arm") == "control")
            .then(pl.lit("treatment"))
            .otherwise(pl.lit("control"))
            .alias("arm")
        ),
    )
    other_draw, reader = _build(rows.drop("arm"), swapped, events)
    with reader:
        assert other_draw.design == sparse.design
        assert other_draw.source_refs[0].expected_digest != sparse.source_refs[0].expected_digest


def test_sparse_zero_and_skip_require_explicit_supported_policies():
    rows = _observed()
    events = MetricSpec(name="events", value_column="orders", missing="zero")
    with pytest.raises(CapabilityError) as no_coverage:
        _build(rows, _schedule(), events)
    assert no_coverage.value.code == "source.frame.switchback.event_coverage"
    with pytest.raises(CapabilityError) as skip:
        _build(rows, _schedule(), events.model_copy(update={"missing": "skip"}))
    assert skip.value.code == "source.frame.switchback.metric"
    assert skip.value.context["policy"] == "skip"


@pytest.mark.integration
@pytest.mark.parametrize("outage", [False, True], ids=["mixed", "all-unavailable"])
def test_covered_events_and_strict_alias_keep_original_cells_and_pilots(tmp_path, outage):
    rows = _observed().filter((pl.col("step") == 2) & (pl.col("orders") != 0))
    if outage:
        rows = rows.with_columns(pl.lit(None).alias("orders"))
    events = MetricSpec(name="events", value_column="orders", missing="zero")
    required = MetricSpec(name="required", value_column="orders")
    experiment, reader = _build(rows, _schedule("events"), events, required)
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
        with pytest.raises(CapabilityError) as caught:
            workflows.planning_baseline(experiment, reader, "required")
        if not outage:
            baseline = workflows.planning_baseline(experiment, reader, "events")
            assert isinstance(baseline, SwitchbackBaseline) and baseline.metric == "events"
    context = {
        "metric": "required",
        "reason": "unobserved_response",
        "missing_count": 8 if outage else 4,
        "first_missing_key": ("a", 0, 0, 2),
        "arm": "control",
    }
    assert caught.value.code == "source.frame.switchback.response_unobserved"
    assert caught.value.context == context
    envelope = store.open_run("frame", saved.run.analysis_spec_hash, saved.run.run_id).envelope()
    cells = {cell.identity.key.metric: cell for cell in envelope.cells}
    assert cells.keys() == {"events", "required"}
    assert cells["required"].components["sampling"][0].context == context
    event_summary = cells["events"].summary
    assert event_summary is not None and event_summary.point == (None if outage else 3)
    result_frame = saved.readout().to_frame(backend="polars")
    assert isinstance(result_frame, pl.DataFrame)
    (failure,) = [row for row in result_frame.to_dicts() if row["metric"] == "required"]
    assert failure["estimate"] is failure["standard_error"] is None
    assert json.loads(failure["failure_context"]) == {
        **context,
        "first_missing_key": ["a", 0, 0, 2],
    }
    if outage:
        status, _ = cells["events"].components["sampling"]
        assert status.context == {
            **context,
            "metric": "events",
            "missing_count": 4,
            "first_missing_key": ("a", 0, 1, 2),
            "arm": "treatment",
        }


def test_ratio_missing_keys_count_once_and_never_zero_the_denominator():
    # Four absent control rows, and both parts null on one observed treatment row.
    rows = _observed().filter((pl.col("step") == 2) & (pl.col("orders") != 0))
    broken = (pl.col("unit") == "a") & (pl.col("cycle") == 0)
    rows = rows.with_columns(
        pl.when(broken).then(None).otherwise(pl.col("orders")).alias("orders"),
        pl.when(broken).then(None).otherwise(3).alias("visits"),
    )
    rate = MetricSpec(
        name="rate", type="ratio", numerator="orders", denominator="visits", missing="zero"
    )
    experiment, reader = _build(rows, _schedule("rate"), rate)
    with reader:
        (cell,) = run(experiment, reader).cells
    status, _ = cell.components["sampling"]
    assert status.context == {
        "metric": "rate",
        "reason": "unobserved_response",
        "first_missing_key": ("a", 0, 0, 2),
        "arm": "control",
        "missing_count": 5,
    }
