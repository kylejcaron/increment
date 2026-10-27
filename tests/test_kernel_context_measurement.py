from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.kernel_context_measurement import measure
from tests.parity_harness.comparison import nested_close


def test_temp_peak_preserves_workload_max_before_cleanup(monkeypatch):
    import calibration.kernel_context_measurement as measurement

    samples = iter((12, 11, 3))
    monkeypatch.setattr(measurement, "_temp_bytes", lambda _source: next(samples))
    temp_peak = measurement._TempPeak(object(), initial=5)
    temp_peak.sample()
    workload_peak = temp_peak.finish()
    cleanup_bytes = measurement._temp_bytes(object())
    assert workload_peak == 12
    assert cleanup_bytes == 3


def test_temp_peak_samples_while_workload_is_running(monkeypatch):
    import threading

    import calibration.kernel_context_measurement as measurement

    observed = threading.Event()

    class Cursor:
        def execute(self, _query):
            return self

        def fetchone(self):
            observed.set()
            return (42,)

    class Connection:
        def cursor(self):
            return Cursor()

    class Session:
        _con = Connection()

    class Source:
        _session = Session()

    monkeypatch.setattr(measurement, "_temp_bytes", lambda _source: 0)
    temp_peak = measurement._TempPeak(Source(), initial=0)
    temp_peak.start()
    assert observed.wait(timeout=1)
    assert temp_peak.finish() == 42


def test_temp_peak_stops_sampling_when_workload_raises(monkeypatch):

    import calibration.kernel_context_measurement as measurement

    class Cursor:
        def execute(self, _query):
            return self

        def fetchone(self):
            return (0,)

    class Connection:
        def cursor(self):
            return Cursor()

    class Session:
        _con = Connection()

    class Source:
        _session = Session()

    monkeypatch.setattr(measurement, "_temp_bytes", lambda _source: 0)
    sampler = measurement._TempPeak(Source(), initial=0)
    with pytest.raises(ValueError, match="readout failed"):
        with sampler:
            raise ValueError("readout failed")
    assert sampler._thread is not None
    assert not sampler._thread.is_alive()


def test_measurement_gate_checks_every_repetition_against_baseline():
    from calibration.kernel_context_measurement import Measurement, Sample, _set_gate

    def sample(output):
        return Sample(
            wall_seconds=1.0,
            peak_rss_bytes=100,
            scope_opens=0,
            pins=0,
            materializations=0,
            temp_bytes=0,
            temp_bytes_after_scope=0,
            session_lifetime_seconds=0.0,
            cleanup=True,
            outputs=(json.dumps(output),),
        )

    def result(samples):
        return Measurement(
            route="captured",
            backend="duckdb",
            rows=8,
            seed=1,
            repetitions=tuple(samples),
            wall_seconds=1.0,
            wall_min_max=(1.0, 1.0),
            peak_rss_bytes=100,
            rss_min_max=(100, 100),
            scope_opens=0,
            pins=0,
            materializations=0,
            temp_bytes=0,
            temp_bytes_after_scope=0,
            session_lifetime_seconds=0.0,
            cleanup=True,
            outputs=samples[0].outputs,
        )

    baseline = result([sample({"value": 1}), sample({"value": 1})])
    candidate = result([sample({"value": 1}), sample({"value": 2})])
    _, gated = _set_gate(baseline, candidate)
    assert gated.outputs_equal is False
    assert gated.wall_threshold_pass is False
    assert gated.rss_threshold_pass is False


def test_measurement_gate_requires_baseline_repetitions_to_agree():
    from calibration.kernel_context_measurement import Measurement, Sample, _set_gate

    def result(outputs):
        samples = tuple(
            Sample(1.0, 100, 0, 0, 0, 0, 0, 0.0, True, (json.dumps(output),)) for output in outputs
        )
        return Measurement(
            route="captured",
            backend="duckdb",
            rows=8,
            seed=1,
            repetitions=samples,
            wall_seconds=1.0,
            wall_min_max=(1.0, 1.0),
            peak_rss_bytes=100,
            rss_min_max=(100, 100),
            scope_opens=0,
            pins=0,
            materializations=0,
            temp_bytes=0,
            temp_bytes_after_scope=0,
            session_lifetime_seconds=0.0,
            cleanup=True,
            outputs=samples[0].outputs,
        )

    baseline = result([{"value": 1}, {"value": 2}])
    candidate = result([{"value": 1}, {"value": 1}])
    _, gated = _set_gate(baseline, candidate)
    assert gated.outputs_equal is False


@pytest.mark.integration
def test_stress_workload_inserts_the_requested_number_of_rows(tmp_path):
    from calibration.kernel_context_measurement import _append_units
    from increment import UnitRowsRequest, evidence_provider
    from tests.analysis_factory import held_source
    from tests.source_conformance import definitions_source

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    try:
        source = held_source(analysis)
        _append_units(source, 2)
        rows = evidence_provider(source).unit_rows(UnitRowsRequest(metric=analysis.metrics[0]))
        assert len(rows) == 6
    finally:
        analysis.close()


@pytest.mark.slow
@pytest.mark.integration
def test_measurement_records_wall_memory_materializations_and_cleanup():
    baseline = measure(
        route="per_operation",
        backend="duckdb",
        rows=8,
        seed=1,
        baseline_root=Path(__file__).parents[1],
    )
    captured = measure(route="captured", backend="duckdb", rows=8, seed=1)

    for result in (baseline, captured):
        assert result.wall_seconds > 0
        assert result.peak_rss_bytes > 0
        assert result.scope_opens >= 0
        assert result.materializations >= 0
        assert result.temp_bytes_after_scope >= 0
        assert result.session_lifetime_seconds >= 0
        assert result.cleanup
        assert len(result.repetitions) == 7
        assert result.outputs
        assert result.temp_bytes == max(sample.temp_bytes for sample in result.repetitions)
        assert result.temp_bytes_after_scope == max(
            sample.temp_bytes_after_scope for sample in result.repetitions
        )
    assert len(baseline.outputs) == len(captured.outputs)
    assert all(
        nested_close(json.loads(expected), json.loads(actual))
        for expected, actual in zip(baseline.outputs, captured.outputs, strict=True)
    )


@pytest.mark.integration
@pytest.mark.slow
def test_artifact_daily_rows_match_captured_with_parity_tolerance():
    baseline = measure(
        route="per_operation",
        backend="artifact",
        rows=8,
        seed=1,
        baseline_root=Path(__file__).parents[1],
    )
    captured = measure(route="captured", backend="artifact", rows=8, seed=1)

    assert nested_close(json.loads(baseline.outputs[2]), json.loads(captured.outputs[2]))


def test_measurement_pair_interleaves_baseline_and_candidate(monkeypatch, tmp_path):
    import calibration.kernel_context_measurement as measurement

    events = []

    def sample_child(route, backend, rows, seed, *, baseline_root):
        events.append((route, seed, baseline_root))
        return measurement.Sample(
            wall_seconds=float(seed),
            peak_rss_bytes=1,
            scope_opens=0,
            pins=0,
            materializations=0,
            temp_bytes=0,
            temp_bytes_after_scope=0,
            session_lifetime_seconds=0.0,
            cleanup=True,
            outputs=("[]",),
        )

    coherence_seeds = []

    def coherence_child(rows, seed):
        coherence_seeds.append(seed)
        return 0.1, 0.15

    monkeypatch.setattr(measurement, "_sample_child", sample_child)
    monkeypatch.setattr(measurement, "_coherence_sample_child", coherence_child)
    baseline, candidate = measurement._measure_pair(
        backend="duckdb",
        rows=8,
        seed=1,
        baseline_root=tmp_path,
    )

    assert [route for route, _seed, _root in events] == [
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
    ]
    assert len(baseline.repetitions) == len(candidate.repetitions) == 7
    assert candidate.coherence_cost_seconds == pytest.approx(0.05)
    assert coherence_seeds == list(range(1, 8))


def test_coherence_probes_run_after_workload_samples_in_separate_children(monkeypatch, tmp_path):
    import calibration.kernel_context_measurement as measurement

    events = []

    def sample_child(route, backend, rows, seed, *, baseline_root):
        events.append(("workload", route, seed))
        return measurement.Sample(
            wall_seconds=1.0,
            peak_rss_bytes=100,
            scope_opens=0,
            pins=0,
            materializations=0,
            temp_bytes=0,
            temp_bytes_after_scope=0,
            session_lifetime_seconds=0.0,
            cleanup=True,
            outputs=("[]",),
        )

    def coherence_child(rows, seed):
        events.append(("coherence", "captured", seed))
        return 0.1, 0.15

    monkeypatch.setattr(measurement, "_sample_child", sample_child)
    monkeypatch.setattr(measurement, "_coherence_sample_child", coherence_child)

    baseline, candidate = measurement._measure_pair(
        backend="duckdb",
        rows=8,
        seed=1,
        baseline_root=tmp_path,
    )

    assert [event[0] for event in events[:14]] == ["workload"] * 14
    assert [event[1] for event in events[:14]] == [
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
        "captured",
        "per_operation",
        "per_operation",
        "captured",
    ]
    assert [event[0] for event in events[14:]] == ["coherence"] * 7
    assert len(baseline.repetitions) == len(candidate.repetitions) == 7
    assert all(
        sample.peak_rss_bytes == 100
        and sample.coherence_live_seconds == 0.1
        and sample.coherence_captured_seconds == 0.15
        for sample in candidate.repetitions
    )


def test_sample_cli_keeps_workload_stdout_out_of_json(capsys, monkeypatch):
    import json
    import sys

    import calibration.kernel_context_measurement as measurement

    sample = measurement.Sample(
        wall_seconds=1.0,
        peak_rss_bytes=100,
        scope_opens=0,
        pins=0,
        materializations=0,
        temp_bytes=0,
        temp_bytes_after_scope=0,
        session_lifetime_seconds=0.0,
        cleanup=True,
        outputs=("[]",),
    )

    def noisy_sample(*args):
        print("workload diagnostic")
        return sample

    monkeypatch.setattr(sys, "argv", ["measurement", "--sample", "captured", "duckdb", "8", "1"])
    monkeypatch.setattr(measurement, "_sample", noisy_sample)

    measurement.main()
    output = capsys.readouterr()

    assert output.err == "workload diagnostic\n"
    assert json.loads(output.out)["cleanup"] is True


@pytest.mark.integration
def test_measurement_duckdb_source_disables_progress_output(tmp_path, monkeypatch):
    import calibration.kernel_context_measurement as measurement
    from tests import source_conformance

    analysis, _ = source_conformance.definitions_source(
        tmp_path, store="always", return_analysis=True
    )
    source = analysis._src
    connection = measurement._duckdb_connection(source)
    connection.raw_sql("SET enable_progress_bar = true")
    monkeypatch.setattr(
        source_conformance,
        "definitions_source",
        lambda *args, **kwargs: (analysis, None),
    )
    try:
        measurement._source("duckdb", tmp_path)
        enabled = connection.raw_sql("SELECT current_setting('enable_progress_bar')").fetchone()[0]

        assert enabled is False
    finally:
        analysis.close()
