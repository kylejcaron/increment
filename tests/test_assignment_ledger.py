from datetime import UTC, datetime

import pandas as pd
import pytest

from increment.errors import CapabilityError, CodedError
from increment.evidence import FrameLedger


def _rows() -> list[dict[str, object]]:
    return [
        {
            "unit": "u1",
            "experiment": "exp",
            "arm": "control",
            "period": None,
            "version": 1,
            "allocated_at": None,
            "exposed_at": datetime(2026, 1, 1, tzinfo=UTC),
            "triggered_at": None,
        }
    ]


def test_ledger_rejects_duplicate_assignment_key() -> None:
    rows = _rows() * 2
    with pytest.raises(CapabilityError) as refused:
        FrameLedger(pd.DataFrame(rows), roles={name: name for name in rows[0]}).reduce(
            experiment="exp", population="assigned"
        )
    assert refused.value.code == "evidence.assignment.invalid"
    assert refused.value.context["reason"] == "duplicate_assignment_key"


def test_ledger_accepts_pandas_column_named_column_names() -> None:
    frame = pd.DataFrame(_rows())
    frame["column_names"] = ["metadata"]

    ledger = FrameLedger(frame, roles={name: name for name in _rows()[0]})

    evidence = ledger.reduce(experiment="exp", population="assigned")
    assert evidence.arm_counts == ((1, None, "control", 1),)


def test_ledger_preserves_period_and_event_time_dimensions() -> None:
    frame = pd.DataFrame(
        [
            {**_rows()[0], "period": "cycle-1:ct", "allocated_at": None},
            {**_rows()[0], "period": "cycle-1:tc", "exposed_at": None},
        ]
    )
    ledger = FrameLedger(frame, roles={name: name for name in frame.columns})
    evidence = ledger.reduce(experiment="exp", population="assigned", all_periods=True)
    assert evidence.periods == ("cycle-1:ct", "cycle-1:tc")
    assert evidence.rows[0].allocated_at is None
    assert evidence.rows[0].exposed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert evidence.rows[1].exposed_at is None


def test_frame_ledger_validates_cluster_arm_consistency() -> None:
    frame = pd.DataFrame(
        [
            {**_rows()[0], "unit": "u1", "cluster": "c1", "arm": "control"},
            {**_rows()[0], "unit": "u2", "cluster": "c1", "arm": "treatment"},
        ]
    )
    roles = {name: name for name in frame.columns}
    with pytest.raises(CapabilityError) as refused:
        FrameLedger(frame, roles=roles).reduce(
            experiment="exp", population="assigned", key="cluster"
        )
    assert refused.value.code == "evidence.assignment.invalid"
    assert refused.value.context["reason"] == "conflicting_cluster_arms"


def test_cluster_only_ledger_counts_distinct_clusters_and_sorts_null_periods() -> None:
    frame = pd.DataFrame(
        [
            {"cluster": "c1", "experiment": "exp", "arm": "control", "version": 1, "period": None},
            {"cluster": "c1", "experiment": "exp", "arm": "control", "version": 1, "period": "p1"},
        ]
    )
    roles = {name: name for name in frame.columns}
    evidence = FrameLedger(frame, roles=roles).reduce(
        experiment="exp", population="assigned", all_periods=True, key="cluster"
    )
    assert evidence.independence_grain == "cluster"
    assert evidence.arm_counts == ((1, None, "control", 1), (1, "p1", "control", 1))


def test_ledger_exposure_join_returns_next_period_arm_as_data_only_rows() -> None:
    frame = pd.DataFrame(
        [
            {**_rows()[0], "period": "cycle-1:ct", "arm": "control"},
            {**_rows()[0], "period": "cycle-1:tc", "arm": "treatment"},
        ]
    )
    ledger = FrameLedger(frame, roles={name: name for name in frame.columns})
    result = ledger.exposure(
        experiment="exp",
        version=1,
        period_order=("cycle-1:ct", "cycle-1:tc"),
    )
    assert result == (
        ("u1", "cycle-1:ct", "control", "treatment"),
        ("u1", "cycle-1:tc", "treatment", None),
    )


def test_ledger_neighbour_lookup_is_not_a_backend_relation() -> None:
    frame = pd.DataFrame(
        [
            {**_rows()[0], "period": "cycle-1:ct", "arm": "control"},
            {**_rows()[0], "period": "cycle-1:tc", "arm": "treatment"},
        ]
    )
    result = FrameLedger(frame, roles={name: name for name in frame.columns}).exposure(
        experiment="exp",
        version=1,
        period_order=("cycle-1:ct", "cycle-1:tc"),
    )
    assert isinstance(result, tuple)


@pytest.mark.integration
def test_query_ledger_exposure_join_returns_only_data_rows() -> None:
    import ibis

    from increment.query.ledger import Ledger

    frame = pd.DataFrame(
        [
            {**_rows()[0], "period": "cycle-1:ct", "arm": "control"},
            {**_rows()[0], "period": "cycle-1:tc", "arm": "treatment"},
        ]
    )
    ledger = Ledger(ibis.memtable(frame), roles={name: name for name in frame.columns})
    result = ledger.exposure(
        experiment="exp",
        version=1,
        period_order=("cycle-1:ct", "cycle-1:tc"),
    )
    assert result == (
        ("u1", "cycle-1:ct", "control", "treatment"),
        ("u1", "cycle-1:tc", "treatment", None),
    )
    assert result[0][3] == "treatment"


def test_ledger_supports_polars_dataframes() -> None:
    pl = pytest.importorskip("polars")
    evidence = FrameLedger(pl.DataFrame(_rows()), roles={name: name for name in _rows()[0]}).reduce(
        experiment="exp", population="assigned"
    )
    assert evidence.arm_counts == ((1, None, "control", 1),)


def test_ledger_refuses_colliding_native_unit_identities() -> None:
    rows = [
        {**_rows()[0], "unit": 1},
        {**_rows()[0], "unit": "1"},
    ]
    with pytest.raises(CodedError) as refused:
        FrameLedger(pd.DataFrame(rows), roles={name: name for name in rows[0]}).reduce(
            experiment="exp", population="assigned"
        )
    assert refused.value.code == "estimation.crossfit.identity_collision"


@pytest.mark.integration
def test_query_ledger_ignores_finite_default_limit(monkeypatch) -> None:
    import ibis

    from increment.query.ledger import Ledger

    frame = pd.DataFrame(
        [
            {**_rows()[0], "unit": "u1"},
            {**_rows()[0], "unit": "u2"},
        ]
    )
    monkeypatch.setattr(ibis.options.sql, "default_limit", 1)
    evidence = Ledger(ibis.memtable(frame), roles={name: name for name in frame.columns}).reduce(
        experiment="exp", population="assigned"
    )
    assert sum(count for _, _, _, count in evidence.arm_counts) == 2
    exposure_frame = pd.DataFrame(
        [
            {**_rows()[0], "unit": "u1", "period": "p0"},
            {**_rows()[0], "unit": "u1", "period": "p1"},
        ]
    )
    exposure = Ledger(
        ibis.memtable(exposure_frame),
        roles={name: name for name in exposure_frame.columns},
    ).exposure(experiment="exp", version=1, period_order=("p0", "p1"))
    assert len(exposure) == 2


def test_frame_ledger_canonicalizes_only_the_selected_version() -> None:
    rows = [
        {**_rows()[0], "unit": 1, "version": 1},
        {**_rows()[0], "unit": "1", "version": 2},
    ]
    evidence = FrameLedger(pd.DataFrame(rows), roles={name: name for name in rows[0]}).reduce(
        experiment="exp", population="assigned", version=1
    )
    assert evidence.arm_counts == ((1, None, "control", 1),)


@pytest.mark.integration
def test_query_ledger_normalizes_nullable_periods() -> None:
    import ibis

    from increment.query.ledger import Ledger

    frame = pd.DataFrame(
        [
            {**_rows()[0], "unit": "u1", "period": None},
            {**_rows()[0], "unit": "u2", "period": "p1"},
        ]
    )
    evidence = Ledger(ibis.memtable(frame), roles={name: name for name in frame.columns}).reduce(
        experiment="exp", population="assigned", all_periods=True
    )
    assert evidence.periods == (None, "p1")
    assert evidence.arm_counts == (
        (1, None, "control", 1),
        (1, "p1", "control", 1),
    )


@pytest.mark.integration
def test_query_ledger_keeps_assignment_scope_when_triggered_count_is_empty() -> None:
    import ibis

    from increment.query.ledger import Ledger

    frame = pd.DataFrame([{**_rows()[0], "unit": "u1"}])
    evidence = Ledger(ibis.memtable(frame), roles={name: name for name in frame.columns}).reduce(
        experiment="exp", population="triggered"
    )
    assert evidence.versions == (1,)
    assert evidence.periods == (None,)
    assert evidence.arm_counts == ()


@pytest.mark.integration
def test_query_ledger_rejects_cluster_membership_changes_across_periods() -> None:
    import ibis

    from increment.query.ledger import Ledger

    rows = [
        {**_rows()[0], "unit": "u1", "cluster": "c1", "period": "p0"},
        {**_rows()[0], "unit": "u1", "cluster": "c2", "period": "p1"},
    ]
    with pytest.raises(CapabilityError) as refused:
        Ledger(ibis.memtable(pd.DataFrame(rows)), roles={name: name for name in rows[0]}).reduce(
            experiment="exp", population="assigned", all_periods=True, key="cluster"
        )
    assert refused.value.context["reason"] == "inconsistent_unit_cluster_membership"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"arm": ""}, "invalid_arm"),
        ({"arm": 1}, "invalid_arm"),
        ({"triggered_at": ""}, "invalid_event_timestamp"),
    ],
)
def test_query_ledger_preserves_row_validation(updates, reason) -> None:
    import ibis

    from increment.query.ledger import Ledger

    frame = pd.DataFrame([{**_rows()[0], "unit": "u1", **updates}])
    with pytest.raises(CapabilityError) as refused:
        Ledger(ibis.memtable(frame), roles={name: name for name in frame.columns}).reduce(
            experiment="exp", population="assigned"
        )
    assert refused.value.context["reason"] == reason


@pytest.mark.integration
def test_query_ledger_validates_untriggered_allocation_versions() -> None:
    import ibis

    from increment.query.ledger import Ledger

    rows = [
        {**_rows()[0], "unit": "u1", "version": 1, "triggered_at": None},
        {
            **_rows()[0],
            "unit": "u2",
            "version": 2,
            "triggered_at": datetime(2026, 1, 2, tzinfo=UTC),
        },
    ]
    with pytest.raises(CapabilityError) as refused:
        Ledger(ibis.memtable(pd.DataFrame(rows)), roles={name: name for name in rows[0]}).reduce(
            experiment="exp", population="triggered"
        )
    assert refused.value.context["reason"] == "mixed_allocation_versions"
