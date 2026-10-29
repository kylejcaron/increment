from __future__ import annotations

from typing import Any

import pytest

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


@pytest.mark.integration
@pytest.mark.parametrize("mutation", tuple(_MUTATIONS))
@pytest.mark.parametrize("coherence", ("captured", "live"))
def test_run_owner_coherence_mutations_between_primary_and_breakout(
    tmp_path, monkeypatch, mutation, coherence
):
    from increment.errors import CodedError
    from increment.readouts import run_scoped

    baseline_dir = tmp_path / "baseline"
    mutated_dir = tmp_path / "mutated"
    baseline_dir.mkdir()
    mutated_dir.mkdir()
    baseline, (_, baseline_con) = definitions_source(
        baseline_dir, return_analysis=True, return_connection=True
    )
    analysis, (_, con) = definitions_source(
        mutated_dir, return_analysis=True, return_connection=True
    )
    _add_replication_units(baseline_con)
    _add_replication_units(con)
    try:
        with pytest.warns():
            original = run_scoped(baseline, coherence=coherence, breakout_correction="none")
        original_breakout = _breakout_values(original["breakout"])
        run_primary = analysis.run

        def run_then_mutate(*args, **kwargs):
            result = run_primary(*args, **kwargs)
            con.raw_sql(_MUTATIONS[mutation])
            return result

        monkeypatch.setattr(analysis, "run", run_then_mutate)
        try:
            with pytest.warns():
                changed = run_scoped(analysis, coherence=coherence, breakout_correction="none")
        except CodedError as refusal:
            assert coherence == "captured"
            assert refusal.code == "kernel.snapshot.integrity"
            return

        changed_breakout = _breakout_values(changed["breakout"])
        if coherence == "captured":
            assert changed_breakout == original_breakout
        else:
            assert changed_breakout != original_breakout
            metadata = changed["breakout"].metadata
            assert metadata is not None
            assert metadata.coherence == "live"
            assert set(metadata.effective_modes.values()) == {"live"}
    finally:
        baseline.close()
        analysis.close()
