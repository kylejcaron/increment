"""Experiment integrity gates: pure functions over (con, Table, Experiment)."""

from __future__ import annotations

import datetime as dt
import warnings

import pyarrow as pa
import pytest

from increment.errors import InvalidRequestError
from increment.query.integrity import (
    arm_counts,
    enforce_mixed_assignments,
    mixed_assignment_count,
    mixed_assignment_snapshot,
    validate_cluster_labels,
    validate_trigger_fires_in_every_arm,
)
from tests.warning_codes import warning_codes


def test_arm_counts_returns_per_arm_unit_counts(con, exposures):
    """u1/u3 land in treatment, u2 in control - the mixed unit u4 is
    already dropped by `first_exposures` upstream of this table."""
    counts = arm_counts(con, exposures)
    assert counts == {"treatment": 2, "control": 1}


def test_mixed_assignment_count_counts_units_seen_in_more_than_one_arm(
    con, experiment, exposure_events
):
    """u4 appears in both treatment and control - the one mixed unit."""
    assert mixed_assignment_count(con, exposure_events, experiment) == 1


def test_mixed_assignment_snapshot_clean_experiment_has_non_null_fingerprint(
    con, experiment, exposure_events
):
    """A clean population still returns an integer fingerprint."""
    clean = exposure_events.filter(exposure_events.unit_id != "u4")
    mixed, unassigned, fingerprint = mixed_assignment_snapshot(con, clean, experiment)
    assert mixed == 0
    assert unassigned == 0
    assert isinstance(fingerprint, int)


@pytest.mark.creates_tables
def test_mixed_assignment_snapshot_separates_null_and_mixed(con, experiment):
    rows = pa.table(
        {
            "unit_id": ["null_only", "control_plus_null", "control_plus_null", "mixed", "mixed"],
            "ts": [dt.datetime(2025, 8, 1, 9)] * 5,
            "event": ["exposure"] * 5,
            "experiment_id": [experiment.name] * 5,
            "group_id": pa.array([None, "control", None, "control", "treatment"], type=pa.string()),
        }
    )
    events = con.create_table("assignment_integrity_snapshot", obj=rows)
    mixed, unassigned, fingerprint = mixed_assignment_snapshot(con, events, experiment)
    assert mixed == 1
    assert unassigned == 2
    assert isinstance(fingerprint, int)


@pytest.mark.creates_tables
def test_mixed_assignment_snapshot_normalizes_null_experiment_id(con, experiment):
    """Fact sources without experiment_id retain mixed-assignment rows."""
    rows = pa.table(
        {
            "unit_id": ["u_null", "u_null"],
            "ts": [dt.datetime(2025, 8, 1, 9), dt.datetime(2025, 8, 1, 10)],
            "event": ["exposure", "exposure"],
            "experiment_id": pa.array([None, None], type=pa.string()),
            "group_id": ["control", "treatment"],
        }
    )
    events = con.create_table("mixed_null_experiment_id", obj=rows)
    mixed, unassigned, fingerprint = mixed_assignment_snapshot(con, events, experiment)
    assert mixed == 1
    assert unassigned == 0
    assert isinstance(fingerprint, int)


@pytest.mark.creates_tables
def test_mixed_assignment_fingerprint_preserves_assignment_boundaries(con, experiment):
    """Distinct assignment rows cannot collide through separator characters."""
    unit_id = "u_boundary"
    field_sep = "\x1f"
    row_sep = "\x1e"
    embedded_key = f"{experiment.name}{field_sep}{unit_id}{field_sep}"

    def snapshot(name: str, groups: list[str]) -> tuple[int, int, int]:
        rows = pa.table(
            {
                "unit_id": [unit_id, unit_id],
                "ts": [dt.datetime(2025, 8, 1, 9), dt.datetime(2025, 8, 1, 10)],
                "event": ["exposure", "exposure"],
                "experiment_id": [experiment.name, experiment.name],
                "group_id": groups,
            }
        )
        return mixed_assignment_snapshot(con, con.create_table(name, obj=rows), experiment)

    count_a, _unassigned_a, fingerprint_a = snapshot(
        "mixed_boundary_a",
        ["a", f"b{row_sep}{embedded_key}c"],
    )
    count_b, _unassigned_b, fingerprint_b = snapshot(
        "mixed_boundary_b",
        [f"a{row_sep}{embedded_key}b", "c"],
    )

    assert count_a == count_b == 1
    assert fingerprint_a != fingerprint_b


@pytest.mark.creates_tables
def test_mixed_assignment_fingerprint_preserves_field_boundaries(con, experiment):
    """Moving a separator between unit and group fields changes the fingerprint."""

    def snapshot(name: str, unit_id: str, groups: list[str]) -> tuple[int, int, int]:
        rows = pa.table(
            {
                "unit_id": [unit_id, unit_id],
                "ts": [dt.datetime(2025, 8, 1, 9), dt.datetime(2025, 8, 1, 10)],
                "event": ["exposure", "exposure"],
                "experiment_id": [experiment.name, experiment.name],
                "group_id": groups,
            }
        )
        return mixed_assignment_snapshot(con, con.create_table(name, obj=rows), experiment)

    count_a, _unassigned_a, fingerprint_a = snapshot(
        "mixed_field_boundary_a",
        "u",
        ["p\x1fa", "p\x1fb"],
    )
    count_b, _unassigned_b, fingerprint_b = snapshot(
        "mixed_field_boundary_b",
        "u\x1fp",
        ["a", "b"],
    )

    assert count_a == count_b == 1
    assert fingerprint_a != fingerprint_b


def test_enforce_mixed_assignments_error_policy_raises():
    with pytest.raises(InvalidRequestError) as error:
        enforce_mixed_assignments(3, "error", already_warned=False)
    assert error.value.code == "query.integrity.mixed_assignment_units"


def test_enforce_mixed_assignments_warn_policy_warns_once():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        already_warned = enforce_mixed_assignments(3, "warn", already_warned=False)
    assert already_warned is True
    assert len(record) == 1
    assert "query.integrity.mixed_assignments_excluded" in warning_codes(record)

    # Already warned - must not warn again, but stays warned.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        still_warned = enforce_mixed_assignments(3, "warn", already_warned=True)
    assert still_warned is True
    assert len(record) == 0


def test_enforce_mixed_assignments_distinguishes_null_policy_messages():
    # count=0 with unassigned_count=2: the NULL finding alone must still refuse,
    # which the mixed-only path does not reach.
    with pytest.raises(InvalidRequestError) as error:
        enforce_mixed_assignments(
            0,
            "error",
            unassigned_count=2,
            already_warned=False,
        )
    assert error.value.code == "query.integrity.unassigned_assignment_units"

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warned = enforce_mixed_assignments(
            0,
            "warn",
            unassigned_count=1,
            already_warned=False,
        )
    assert warned is True
    assert len(record) == 1
    assert "query.integrity.mixed_assignments_excluded" in warning_codes(record)


def test_enforce_mixed_assignments_reports_both_counts_when_both_hazards_occur():
    with pytest.raises(InvalidRequestError) as error:
        enforce_mixed_assignments(3, "error", unassigned_count=2, already_warned=False)
    assert error.value.code == "query.integrity.mixed_assignment_units"
    assert (error.value.context["mixed_count"], error.value.context["unassigned_count"]) == (3, 2)


def test_enforce_mixed_assignments_exclude_policy_neither_raises_nor_warns():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        already_warned = enforce_mixed_assignments(3, "exclude", already_warned=False)
    assert already_warned is False
    assert len(record) == 0


def test_enforce_mixed_assignments_zero_count_is_a_no_op():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for policy in ("error", "warn", "exclude"):
            already_warned = enforce_mixed_assignments(0, policy, already_warned=False)
            assert already_warned is False
    assert len(record) == 0


@pytest.mark.creates_tables
@pytest.mark.parametrize("enforce_purity", [True, False])
def test_validate_cluster_labels_raises_on_null_label(con, enforce_purity):
    tbl = con.create_table(
        f"cluster_null_{enforce_purity}",
        obj=[
            {"unit_id": "u1", "group_id": "treatment", "site": "s1"},
            {"unit_id": "u2", "group_id": "control", "site": None},
        ],
    )
    with pytest.raises(InvalidRequestError) as error:
        validate_cluster_labels(tbl, "site", "exp_test", enforce_purity=enforce_purity)
    assert error.value.context == {
        "experiment": "exp_test",
        "cluster": "site",
        "kind": "null labels",
        "value": 1,
        "constraint": "every enrolled unit must have a cluster label",
    }


@pytest.mark.creates_tables
def test_validate_cluster_labels_raises_on_label_spanning_arms(con):
    tbl = con.create_table(
        "cluster_span",
        obj=[
            {"unit_id": "u1", "group_id": "treatment", "site": "s1"},
            {"unit_id": "u2", "group_id": "control", "site": "s1"},
        ],
    )
    with pytest.raises(InvalidRequestError) as error:
        validate_cluster_labels(tbl, "site", "exp_test")
    assert error.value.context["cluster"] == "site"
    assert error.value.context["kind"] == "cross-arm labels"
    assert (
        error.value.context["constraint"]
        == "each randomization-grain cluster belongs to exactly one arm"
    )


@pytest.mark.creates_tables
def test_validate_cluster_labels_accepts_label_spanning_arms_without_purity(con):
    tbl = con.create_table(
        "cluster_span_without_purity",
        obj=[
            {"unit_id": "u1", "group_id": "treatment", "site": "s1"},
            {"unit_id": "u2", "group_id": "control", "site": "s1"},
        ],
    )
    validate_cluster_labels(tbl, "site", "exp_test", enforce_purity=False)


@pytest.mark.creates_tables
def test_validate_cluster_labels_passes_clean_labels(con):
    tbl = con.create_table(
        "cluster_clean",
        obj=[
            {"unit_id": "u1", "group_id": "treatment", "site": "s1"},
            {"unit_id": "u2", "group_id": "control", "site": "s2"},
        ],
    )
    validate_cluster_labels(tbl, "site", "exp_test")  # must not raise


def test_validate_trigger_fires_in_every_arm_raises_naming_silent_arm():
    with pytest.raises(InvalidRequestError) as error:
        validate_trigger_fires_in_every_arm({"a": 0.5, "b": 0.0}, "checkout")
    assert error.value.code == "query.integrity.trigger_arm_missing"


def test_validate_trigger_fires_in_every_arm_passes_when_every_arm_fires():
    validate_trigger_fires_in_every_arm({"a": 0.5, "b": 0.2}, "checkout")  # must not raise


def test_cluster_labels_refuse_a_unit_seen_in_two_clusters():
    """Raw exposure rows with two non-null cluster labels for one unit are a
    malformed cluster source; the gate refuses before dedup can hide it."""
    import datetime as dt

    import ibis
    import pytest

    from increment.errors import CodedError
    from increment.query.integrity import validate_cluster_uniqueness

    con = ibis.duckdb.connect()
    raw = con.create_table(
        "cluster_conflict_raw",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": "s9",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": "s1",
                "ts": dt.datetime(2025, 8, 1, 10),
            },
        ],
    )
    with pytest.raises(CodedError) as refusal:
        validate_cluster_uniqueness(raw, cluster="store_id", experiment_name="e")
    assert refusal.value.code == "query.integrity.cluster_conflict"


def test_validate_cluster_uniqueness_passes_a_null_label():
    """A unit with no cluster label at all has nothing to conflict --
    `validate_cluster_labels` owns refusing the null label itself."""
    import datetime as dt

    import ibis

    from increment.query.integrity import validate_cluster_uniqueness

    con = ibis.duckdb.connect()
    raw = con.create_table(
        "cluster_null_label_raw",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": None,
                "ts": dt.datetime(2025, 8, 1, 9),
            },
        ],
        schema={
            "unit_id": "string",
            "experiment_id": "string",
            "group_id": "string",
            "store_id": "string",
            "ts": "timestamp",
        },
    )
    validate_cluster_uniqueness(raw, cluster="store_id", experiment_name="e")  # must not raise


def test_validate_cluster_uniqueness_passes_a_repeated_identical_label():
    """A unit exposed twice under the SAME cluster label is not a conflict
    -- only two distinct non-null labels for one unit are."""
    import datetime as dt

    import ibis

    from increment.query.integrity import validate_cluster_uniqueness

    con = ibis.duckdb.connect()
    raw = con.create_table(
        "cluster_repeated_label_raw",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": "s9",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": "s9",
                "ts": dt.datetime(2025, 8, 1, 10),
            },
        ],
    )
    validate_cluster_uniqueness(raw, cluster="store_id", experiment_name="e")  # must not raise


def test_validate_cluster_uniqueness_only_sees_the_rows_it_is_given():
    """The gate checks exactly the rows it receives, nothing more -- a
    caller that pre-scopes to an admitted window/population (e.g. via
    `_scope_exposure_events` and a semi-join to admitted identities) keeps
    an out-of-window or excluded unit's conflicting label from ever
    reaching this gate, and a conflict for a different unit never taints
    an unrelated one."""
    import datetime as dt

    import ibis

    from increment.query.integrity import validate_cluster_uniqueness

    con = ibis.duckdb.connect()
    admitted = con.create_table(
        "cluster_admitted_raw",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "store_id": "s9",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "c",
                "store_id": "s2",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
        ],
    )
    validate_cluster_uniqueness(admitted, cluster="store_id", experiment_name="e")  # must not raise


def test_enforce_mixed_assignments_error_policy_is_coded():
    with pytest.raises(InvalidRequestError) as raised:
        enforce_mixed_assignments(3, "error", unassigned_count=0, already_warned=False)
    assert raised.value.code == "query.integrity.mixed_assignment_units"


def test_validate_trigger_fires_in_every_arm_is_coded():
    with pytest.raises(InvalidRequestError) as raised:
        validate_trigger_fires_in_every_arm({"control": 0.5, "treat": 0.0}, "clicked")
    assert raised.value.code == "query.integrity.trigger_arm_missing"
