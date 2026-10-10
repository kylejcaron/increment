"""End-to-end pins for the ``aov_decomposition`` and ``session_checkout``
examples in ``examples/definitions/``.

1. **AOV decomposes exactly.** ``aov`` is a ratio metric (total revenue /
   total orders at user grain); on the log scale its lift must equal
   ``revenue_per_user`` lift minus ``orders_per_user`` lift. The seeded
   scenario is a pure mix shift - treatment only ADDS small orders, never
   changes an existing order's value - so revenue up + orders up + AOV
   *down* is the correct readout; breaking that identity or flipping a
   sign breaks the delta-method ratio path.

2. **Session-grain experiments run end to end.** ``unit: session_id`` with
   per-session assignment on the exposure event enrolls SESSIONS, not
   users: the same user legitimately appears in both arms.
"""

import math
from datetime import datetime, timedelta

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.examples]

DEFINITIONS = "examples/definitions/"


def _schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("event_at", pa.timestamp("us")),
            ("user_id", pa.string()),
            ("session_id", pa.string()),
            ("event", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", pa.string()),
            ("revenue", pa.float64()),
            ("duration_s", pa.float64()),
            ("country_code", pa.string()),
            ("device_type", pa.string()),
            ("plan", pa.string()),
        ]
    )


def _row(event_at, user_id, session_id, event, experiment_id=None, group_id=None, revenue=None):
    return {
        "event_at": event_at,
        "user_id": user_id,
        "session_id": session_id,
        "event": event,
        "experiment_id": experiment_id,
        "group_id": group_id,
        "revenue": revenue,
        "duration_s": None,
        "country_code": "US",
        "device_type": "web",
        "plan": "free",
    }


def _seed_rows():
    """Deterministic event log for both experiments.

    Returns ``(rows, expected)`` where ``expected`` carries the arm-level
    sums the assertions are computed from, so test and truth cannot drift apart.
    """
    rows = []
    t0 = datetime(2025, 5, 2, 10, 0, 0)

    # aov_decomposition: 40 users, alternating arms, pure mix shift
    rev = {"control": 0.0, "treatment": 0.0}
    orders = {"control": 0, "treatment": 0}
    n_users = {"control": 0, "treatment": 0}
    for i in range(40):
        uid, grp = f"aov_u{i}", ("treatment" if i % 2 else "control")
        n_users[grp] += 1
        rows.append(
            _row(
                t0 + timedelta(minutes=i),
                uid,
                f"aov_s{i}_0",
                "page_view",
                experiment_id="aov_decomposition",
                group_id=grp,
            )
        )
        # 2 orders/user, +1 for every 4th user; index-keyed identically
        # across arms so neither arm is zero-variance on any metric.
        base_values = [30.0 + i % 5, 50.0 + i % 7]
        if i % 4 < 2:
            base_values.append(40.0 + i % 3)
        for j, value in enumerate(base_values):
            rows.append(
                _row(
                    datetime(2025, 5, 3 + j % 2, 12, 0, 0),
                    uid,
                    f"aov_s{i}_{j + 1}",
                    "purchase",
                    revenue=value,
                )
            )
            rev[grp] += value
            orders[grp] += 1
        # Treatment: one EXTRA small incremental order - revenue and
        # orders up, AOV down, no existing order's value changed.
        if grp == "treatment":
            rows.append(
                _row(datetime(2025, 5, 5, 12, 0, 0), uid, f"aov_s{i}_3", "purchase", revenue=8.0)
            )
            rev[grp] += 8.0
            orders[grp] += 1

    # session_checkout: 60 sessions over 15 users, per-session arms
    conv = {"control": 0, "treatment": 0}
    n_sessions = {"control": 0, "treatment": 0}
    for i in range(60):
        sid, uid, grp = (
            f"sess_s{i}",
            f"sess_u{i % 15}",
            ("treatment" if i % 2 else "control"),
        )
        n_sessions[grp] += 1
        ts = t0 + timedelta(hours=i)
        rows.append(_row(ts, uid, sid, "page_view", experiment_id="session_checkout", group_id=grp))
        # (i // 2) % 4 != 0 gives treatment 73.3% (22/30) vs control's 20% (6/30);
        # unlike `i % 4 != 0`, which is never 0 for odd i (zero-variance treatment).
        purchased = ((i // 2) % 4 != 0) if grp == "treatment" else (i % 5 == 0)
        if purchased:
            rows.append(_row(ts + timedelta(minutes=5), uid, sid, "purchase", revenue=25.0))
            conv[grp] += 1

    # keepalive: unrelated unit past every window's maturity date, so
    # enrolled units are observable rather than censored out.
    for event in ("purchase", "page_view"):
        rows.append(
            _row(
                datetime(2025, 6, 20, 0, 0, 0),
                "keepalive_u",
                "keepalive_s",
                event,
                revenue=0.0 if event == "purchase" else None,
            )
        )

    expected = {
        "rev_lift": (rev["treatment"] / n_users["treatment"])
        / (rev["control"] / n_users["control"])
        - 1.0,
        "orders_lift": (orders["treatment"] / n_users["treatment"])
        / (orders["control"] / n_users["control"])
        - 1.0,
        "aov_lift": (rev["treatment"] / orders["treatment"]) / (rev["control"] / orders["control"])
        - 1.0,
        "conv_lift": (conv["treatment"] / n_sessions["treatment"])
        / (conv["control"] / n_sessions["control"])
        - 1.0,
        "n_sessions": n_sessions,
    }
    return rows, expected


@pytest.fixture(scope="module")
def seeded():
    import ibis
    import pyarrow as pa

    con = ibis.duckdb.connect()
    rows, expected = _seed_rows()
    tbl = pa.Table.from_pylist(rows, schema=_schema())
    con.raw_sql("CREATE SCHEMA IF NOT EXISTS analytics")
    con.create_table("event_log", tbl, database="analytics")
    return con, expected


def test_aov_mix_shift_decomposition(seeded):
    from increment import Analysis
    from tests.analysis_factory import lift_rows

    con, expected = seeded
    analysis = Analysis("aov_decomposition", definitions_path=DEFINITIONS, con=con)
    by_name = {est.metric: est for est in lift_rows(analysis.run())}
    assert set(by_name) == {"aov", "revenue_per_user", "orders_per_user"}

    aov = by_name["aov"].require_lift().value
    rev = by_name["revenue_per_user"].require_lift().value
    orders = by_name["orders_per_user"].require_lift().value

    # Point estimates match the arm-level sums the seed produced.
    assert rev == pytest.approx(expected["rev_lift"], rel=1e-6)
    assert orders == pytest.approx(expected["orders_lift"], rel=1e-6)
    assert aov == pytest.approx(expected["aov_lift"], rel=1e-6)

    # Log-scale identity: AOV lift = revenue lift - orders lift.
    assert math.log1p(aov) == pytest.approx(math.log1p(rev) - math.log1p(orders), abs=1e-9)

    # Mix-shift signature: pure incremental small orders read as
    # revenue up + orders up + AOV down. AOV alone would report a loss.
    assert rev > 0 and orders > 0 and aov < 0


def test_session_unit_experiment_runs_end_to_end(seeded):
    from increment import Analysis
    from increment.results import SRMResult
    from tests.analysis_factory import lift_rows

    con, expected = seeded
    analysis = Analysis("session_checkout", definitions_path=DEFINITIONS, con=con)
    (est,) = lift_rows(analysis.run())

    assert est.metric == "session_purchase_rate"
    assert est.require_lift().value == pytest.approx(expected["conv_lift"], rel=1e-6)
    # Large seeded effect (73.3% vs 20% conversion) must be detected.
    assert est.require_lift().excludes(0.0) and est.require_lift().value > 0

    # The enrolled population is SESSIONS: 30 per arm, even though only
    # 15 distinct users exist and each appears in both arms.
    srm = analysis.srm(expected={"control": 0.5, "treatment": 0.5}, inference="fixed")
    assert isinstance(srm, SRMResult)
    assert srm.observed == expected["n_sessions"]
    assert not srm.is_srm
