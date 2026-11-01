from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import pytest

from increment.errors import IncrementWarning
from tests.source_conformance import definitions_source

_MUTATIONS = {
    "correction": "UPDATE raw_events SET revenue = revenue + 20 WHERE user_id = 'u3' AND event = 'purchase'",
    "deletion": "DELETE FROM raw_events WHERE user_id = 'u3' AND event = 'purchase'",
    "backdated_insertion": (
        "INSERT INTO raw_events (user_id, ts, event, group_id, country_code, revenue) "
        "VALUES ('u1', TIMESTAMP '2026-01-01 11:00:00', 'purchase', NULL, 'US', 500.0)"
    ),
}


def _breakout_values(rows: Any) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.metric,
            row.dimension_value,
            row.group_id,
            row.n_control,
            row.n_treat,
            row.mean_control,
            row.mean_treat,
            row.excluded,
        )
        for row in rows
    )


def _add_replication_units(con: Any) -> None:
    for group in ("control", "treatment"):
        for country in ("US", "CA"):
            for index in range(10):
                user = f"extra_{group}_{country}_{index}"
                value = 3 + index + (20 if group == "treatment" else 0)
                con.raw_sql(
                    "INSERT INTO raw_events (user_id, ts, event, group_id, country_code, revenue) "
                    f"VALUES ('{user}', TIMESTAMP '2026-01-01 09:00:00', 'assignment', "
                    f"'{group}', '{country}', NULL), "
                    f"('{user}', TIMESTAMP '2026-01-01 12:00:00', 'purchase', NULL, "
                    f"'{country}', {value})"
                )


def _mutate_before_breakout_reads(
    source: Any, con: Any, sql: str, coherence: str, mutated: list[str]
) -> None:
    """Change the warehouse once the run's capture is open and before its breakout view reads.

    A captured run pins every stream when ``readout_snapshot`` opens, before any view
    reads; a live run reads the warehouse as it stands, and its breakout view first
    asks the source for ``breakout_sources`` after the primary view has read.
    """
    if coherence == "captured":
        snapshot = source.readout_snapshot

        @contextmanager
        def snapshot_then_mutate(*args: Any, **kwargs: Any) -> Any:
            del source.readout_snapshot
            with snapshot(*args, **kwargs) as pinned:
                con.raw_sql(sql)
                mutated.append(sql)
                yield pinned

        source.readout_snapshot = snapshot_then_mutate
        return
    breakout_sources = source.breakout_sources

    def mutate_then_read(*args: Any, **kwargs: Any) -> Any:
        del source.breakout_sources
        con.raw_sql(sql)
        mutated.append(sql)
        return breakout_sources(*args, **kwargs)

    source.breakout_sources = mutate_then_read


@pytest.mark.integration
@pytest.mark.parametrize("mutation", tuple(_MUTATIONS))
@pytest.mark.parametrize("coherence", ("captured", "live"))
def test_run_owner_coherence_mutations_between_primary_and_breakout(tmp_path, mutation, coherence):
    from increment import Scope, run
    from increment.errors import CodedError
    from tests.analysis_factory import bound_source

    primary = Scope(view="run", grain="total")
    breakout = Scope(view="breakout", grain="total")
    baseline_dir = tmp_path / "baseline"
    mutated_dir = tmp_path / "mutated"
    baseline_dir.mkdir()
    mutated_dir.mkdir()
    (baseline_experiment, baseline), (_, baseline_con) = definitions_source(
        baseline_dir, return_binding=True, return_connection=True
    )
    (experiment, resolver), (_, con) = definitions_source(
        mutated_dir, return_binding=True, return_connection=True
    )
    _add_replication_units(baseline_con)
    _add_replication_units(con)
    try:
        with pytest.warns():
            original = run(
                baseline_experiment.with_views(primary, breakout), baseline, coherence=coherence
            ).readout(breakout)
        original_breakout = _breakout_values(original)
        mutated: list[str] = []
        _mutate_before_breakout_reads(
            bound_source(resolver), con, _MUTATIONS[mutation], coherence, mutated
        )
        try:
            with pytest.warns():
                changed = run(
                    experiment.with_views(primary, breakout), resolver, coherence=coherence
                ).readout(breakout)
        except CodedError as refusal:
            assert coherence == "captured"
            assert refusal.code == "kernel.snapshot.integrity"
            return

        assert mutated == [_MUTATIONS[mutation]]
        changed_breakout = _breakout_values(changed)
        if coherence == "captured":
            assert changed_breakout == original_breakout
        else:
            assert changed_breakout != original_breakout
            metadata = changed.metadata
            assert metadata is not None
            assert metadata.coherence == "live"
            assert set(metadata.effective_modes.values()) == {"live"}
    finally:
        baseline.close()
        resolver.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("saved", "call", "selection", "expected"),
    (
        pytest.param("live", None, "primary_and_breakout", "live", id="saved-live"),
        pytest.param("live", None, "breakout_alone", "live", id="saved-live-breakout-alone"),
        pytest.param("live", "captured", "primary_and_breakout", "captured", id="call-captured"),
        pytest.param("captured", "live", "primary_and_breakout", "live", id="call-live"),
    ),
)
def test_run_coherence_is_the_call_time_value_else_the_saved_one(
    tmp_path, saved, call, selection, expected
):
    from dataclasses import replace

    from increment import Scope, run

    primary = Scope(view="run", grain="total")
    breakout = Scope(view="breakout", grain="total")
    views = (primary, breakout) if selection == "primary_and_breakout" else (breakout,)
    (experiment, resolver), (_, con) = definitions_source(
        tmp_path, return_binding=True, return_connection=True
    )
    _add_replication_units(con)
    saved_experiment = replace(experiment.with_views(*views), request_options={"coherence": saved})
    keywords = {} if call is None else {"coherence": call}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            metadata = run(saved_experiment, resolver, **keywords).readout(breakout).metadata
    finally:
        resolver.close()

    assert metadata is not None
    assert metadata.coherence == expected
    assert set(metadata.effective_modes.values()) == {expected}
