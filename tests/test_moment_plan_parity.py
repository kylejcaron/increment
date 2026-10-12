"""A moment built by the frame path and by a query builder agrees at matched inputs.

One case per reduction plan (unit grain with every family, cluster grain
carrying size or uptake, and the compliance bivariate reduction); the
day-axis plans are covered by tests/test_frame_window_parity.py and the
as-of uptake case in tests/test_frame_panel.py. Tolerance: relative 1e-12
(both paths compute the same two-phase sums over O(10) values; only the
backend's summation order differs, bounded by n*eps ~ 5e-14 at n=240);
the "~0 but exact" first residual sums are compared against the family's
magnitude n*|ref| instead of their own near-zero value.
"""

from __future__ import annotations

import ibis
import numpy as np
import pyarrow as pa
import pytest

from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW, SLOTS
from increment.frame import MetricSpec, from_unit_summary
from increment.query.builders import group_summary
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

_N = 240
_DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="took"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="test fixture, not a real design"
    ),
)


def _units() -> dict[str, list]:
    rng = np.random.default_rng(2026)
    x = rng.normal(5.0, 2.0, _N)
    return {
        "unit_id": [f"u{i:04d}" for i in range(_N)],
        "group_id": ["control"] * (_N // 2) + ["treatment"] * (_N - _N // 2),
        # Six clusters per arm of unequal size (40/20/10/20/10/20), so the den
        # and size families are non-degenerate; a label spanning both arms is refused.
        "store": [f"{'c' if i < _N // 2 else 't'}{(i % 6) if i % 4 else 0}" for i in range(_N)],
        "y": (3.0 + 0.8 * x + rng.normal(0.0, 1.0, _N)).tolist(),
        "x": x.tolist(),
        "y_den": (rng.poisson(4.0, _N) + 1).astype(float).tolist(),
        "d": (rng.random(_N) < 0.35).astype(float).tolist(),
    }


def _frame(units: dict[str, list]) -> pa.Table:
    return pa.table(
        {
            "user_id": units["unit_id"],
            "variant": units["group_id"],
            "store": units["store"],
            "rev": units["y"],
            "pre_rev": units["x"],
            "sessions": units["y_den"],
            "took": units["d"],
        }
    )


def _totals(con, units: dict[str, list], *, metric: str = "rps") -> ibis.Table:
    table = pa.table(
        {
            "unit_id": units["unit_id"],
            "experiment_id": ["exp"] * _N,
            "group_id": units["group_id"],
            "metric": [metric] * _N,
            "y": units["y"],
            "x": units["x"],
            "y_den": units["y_den"],
            "d": units["d"],
            "store": units["store"],
        }
    )
    return con.create_table(f"totals_{metric}", table)


def _assert_rows_agree(
    frame_rows, builder_rows, *, key: str = "group_id", cross_family: tuple[str, ...] = ()
) -> None:
    frame_by = {r[key]: r for r in frame_rows}
    builder_by = {r[key]: r for r in builder_rows}
    assert frame_by.keys() == builder_by.keys()
    for label, f_row in frame_by.items():
        b_row = builder_by[label]
        assert f_row["n"] == b_row["n"], label
        assert f_row["x_role"] == b_row["x_role"], label
        # A degenerate fixture (every cluster the same size) would zero a whole
        # family on both sides and let a mis-paired cross term pass as 0 == 0.
        for slot in cross_family:
            assert f_row[slot] != 0.0, (label, slot)
        magnitude = f_row["n"] * max(
            abs(f_row[ref] or 0.0) for ref in ("ref_y", "ref_x", "ref_den")
        )
        for slot in SLOTS:
            f, b = f_row[slot], b_row[slot]
            if f is None or b is None:
                assert f is None and b is None, (label, slot)
                continue
            assert abs(f - b) <= 1e-12 * max(1.0, abs(f), abs(b), magnitude), (label, slot, f, b)


def test_unit_grain_every_family_agrees():
    units = _units()
    frame_rows = from_unit_summary(
        _frame(units),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="rps",
                type="ratio",
                numerator="rev",
                denominator="sessions",
                covariate="pre_rev",
            )
        ],
        uptake="took",
    ).raw_moments
    con = ibis.duckdb.connect()
    builder_rows = con.to_pyarrow(group_summary(_totals(con, units))).to_pylist()
    _assert_rows_agree(frame_rows, builder_rows, cross_family=("cden3",))


@pytest.mark.parametrize("uptake", [False, True], ids=["cluster_size", "cluster_uptake"])
def test_cluster_grain_agrees(uptake: bool):
    units = _units()
    kwargs = {"uptake": "took", "design": _DESIGN} if uptake else {}
    frame_rows = from_unit_summary(
        _frame(units),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="rev")],
        cluster="store",
        **kwargs,
    ).raw_moments
    con = ibis.duckdb.connect()
    totals = _totals(con, units, metric="rev")
    builder_rows = con.to_pyarrow(group_summary(totals, cluster="store", uptake=uptake)).to_pylist()
    _assert_rows_agree(frame_rows, builder_rows, cross_family=("cden2", "cden3", "cyden", "cxden"))


def test_compliance_cluster_moments_agree():
    units = _units()
    source = from_unit_summary(
        _frame(units),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="rev")],
        cluster="store",
        uptake="took",
        design=_DESIGN,
    )
    frame_arms = {arm.group_id: arm for arm in source.compliance_summary(_DESIGN).arms}
    con = ibis.duckdb.connect()
    totals = _totals(con, units, metric="uptake")
    uptake_totals = totals.mutate(y=totals.d)
    builder_rows = {
        r["group_id"]: r
        for r in con.to_pyarrow(
            group_summary(uptake_totals, cluster="store", uptake=True)
        ).to_pylist()
    }
    assert frame_arms.keys() == builder_rows.keys()
    for group, arm in frame_arms.items():
        row = builder_rows[group]
        # Unequal cluster sizes: a mis-paired size family would otherwise compare 0 == 0.
        assert arm.cluster_size2 is not None and arm.cluster_cross is not None, group
        assert arm.cluster_size2 > 0.0 and arm.cluster_cross != 0.0, group
        for field, slot in COMPLIANCE_ARM_FROM_CLUSTER_ROW.items():
            f, b = getattr(arm, field), row[slot]
            assert abs(f - b) <= 1e-12 * max(1.0, abs(f), abs(b), arm.n_units), (group, field)
