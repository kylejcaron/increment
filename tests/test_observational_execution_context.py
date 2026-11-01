from __future__ import annotations

import datetime as dt
from typing import Any, cast

import pyarrow as pa
import pytest

from increment import from_definitions
from increment.errors import CodedError
from increment.results import LiftEstimate
from tests.analysis_factory import held_source, view_readout
from tests.covariate_cases import categorical_defs_and_con


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
@pytest.mark.integration
@pytest.mark.parametrize("mutation", ["correction", "deletion", "backdated_insert"])
@pytest.mark.parametrize("coherence", ["captured", "live"])
def test_observational_adjustment_uses_profile_stream_coherence(
    tmp_path, monkeypatch, mutation, coherence
):
    from contextlib import contextmanager

    defs_path, connection, units = categorical_defs_and_con(tmp_path)
    connection.create_table(
        "profile_rows",
        obj=pa.Table.from_pylist(
            [
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 7, 20, 10, 0),
                    "region": values["region"],
                }
                for unit, values in units.items()
            ]
        ),
    )
    definitions = (
        defs_path.read_text()
        .replace(
            "exposures:\n",
            "  - name: profile\n"
            "    sql: SELECT * FROM profile_rows\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts:\n"
            "      - {name: profile_marker, column: null}\n"
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "exposures:\n",
        )
        .replace("property: region, source: events", "property: region, source: profile")
    )
    defs_path.write_text(definitions)
    experiment, resolver = from_definitions(
        defs_path, connection, experiment_name="cat_test", store="none"
    )
    source = cast(Any, held_source(resolver))

    def apply_mutation():
        if mutation == "correction":
            connection.raw_sql("UPDATE profile_rows SET region = 'north'")
        elif mutation == "deletion":
            connection.raw_sql("DELETE FROM profile_rows")
        else:
            connection.raw_sql(
                "INSERT INTO profile_rows "
                "SELECT user_id, TIMESTAMP '2025-07-21 10:00:00', 'north' "
                "FROM profile_rows"
            )

    try:
        (before,) = view_readout(experiment, resolver, coherence="captured")
        assert isinstance(before, LiftEstimate)
        captures = 0
        if coherence == "captured":
            original_snapshot = source.readout_snapshot

            @contextmanager
            def mutate_after_capture(*args, **kwargs):
                nonlocal captures
                captures += 1
                with original_snapshot(*args, **kwargs) as pinned:
                    apply_mutation()
                    yield pinned

            monkeypatch.setattr(source, "readout_snapshot", mutate_after_capture)
        else:
            apply_mutation()

        try:
            observed_rows = view_readout(experiment, resolver, coherence=coherence)
            (observed,) = observed_rows
            assert isinstance(observed, LiftEstimate)
        except CodedError as error:
            assert coherence == "live"
            assert mutation == "deletion"
            assert error.code == "adjust.identification.missing_covariates"
            return
        assert observed_rows.metadata is not None
        assert observed_rows.metadata.coherence == coherence
        assert observed.require_lift().value is not None
        assert before.require_lift().value is not None
        if coherence == "captured":
            assert captures == 1
            assert observed.require_lift().value == pytest.approx(before.require_lift().value)
    finally:
        cast(Any, resolver).close()


@pytest.mark.filterwarnings("ignore:(IPTW|DML|AIPW) covariate balance advisory:UserWarning")
@pytest.mark.integration
def test_captured_observational_shared_sql_keeps_all_logical_stream_names(tmp_path):
    defs_path, connection, _ = categorical_defs_and_con(tmp_path)
    definitions = (
        defs_path.read_text()
        .replace(
            "exposures:\n",
            "  - name: profile\n"
            "    sql: SELECT * FROM cat_events\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts:\n"
            "      - {name: profile_marker, column: null}\n"
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "exposures:\n",
        )
        .replace("property: region, source: events", "property: region, source: profile")
    )
    defs_path.write_text(definitions)
    experiment, resolver = from_definitions(
        defs_path, connection, experiment_name="cat_test", store="none"
    )
    try:
        rows = view_readout(experiment, resolver, coherence="captured")
        assert rows
        assert rows.metadata is not None
        assert {"events", "profile"} <= set(rows.metadata.effective_modes)
    finally:
        cast(Any, resolver).close()
