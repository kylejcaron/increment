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

import math

import ibis
import numpy as np
import pyarrow as pa
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from increment import Method
from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW, SLOTS
from increment.frame import CupedCovariate, MetricSpec, from_unit_summary
from increment.query.builders import group_summary
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

pytestmark = pytest.mark.integration

_N = 240
_DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="took"),
    exclusion_restriction=ExclusionRestriction(justification="test fixture, not a real design"),
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


def _property_units(
    seed: int, n_units: int, partitions: int, offset: float, near_constant_den: bool, uptake: bool
) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    row = np.arange(n_units)
    x_resid = rng.normal(size=n_units) * 3.0
    y_resid = 0.4 * x_resid + rng.normal(size=n_units)
    den_resid = (row % 5 - 2) * 1e-3 if near_constant_den else rng.uniform(-2.0, 2.0, n_units)
    order = np.lexsort((rng.permutation(n_units), row % partitions))
    treatment = row >= n_units // 2
    return {
        "unit_id": [f"u{i:04d}" for i in row[order]],
        "group_id": ["treatment" if treatment[i] else "control" for i in row[order]],
        "store": ["unused"] * n_units,
        "y": (offset + y_resid)[order].tolist(),
        "x": (offset + x_resid)[order].tolist(),
        "y_den": (offset + 8.0 + den_resid)[order].tolist(),
        "d": ((row % 3 == 0) & uptake)[order].astype(float).tolist(),
    }


def _property_totals(con, units: dict[str, list]) -> ibis.Table:
    return con.create_table(
        "property_totals",
        pa.table(
            {
                "unit_id": units["unit_id"],
                "experiment_id": ["exp"] * len(units["unit_id"]),
                "group_id": units["group_id"],
                "metric": ["rps"] * len(units["unit_id"]),
                "y": units["y"],
                "x": units["x"],
                "y_den": units["y_den"],
                "d": units["d"],
            }
        ),
    )


def _assert_emitted_slots_agree(frame_rows, sql_rows, units: dict[str, list]) -> None:
    frame_by = {row["group_id"]: row for row in frame_rows}
    sql_by = {row["group_id"]: row for row in sql_rows}
    assert frame_by.keys() == sql_by.keys()
    slot_values = {"y": "y", "x": "x", "den": "y_den"}
    for group, frame_row in frame_by.items():
        sql_row = sql_by[group]
        assert frame_row["n"] == sql_row["n"], (group, "n")
        for count in ("sum_d", "successes"):
            if count in frame_row or count in sql_row:
                assert frame_row.get(count) == sql_row.get(count), (group, count)
        for slot, spec in SLOTS.items():
            frame_value, sql_value = frame_row.get(slot), sql_row.get(slot)
            assert (frame_value is None) == (sql_value is None), (group, slot)
            if frame_value is None:
                continue
            if spec.kind == "ref":
                assert math.isclose(frame_value, sql_value, rel_tol=1e-9, abs_tol=0.0), (
                    group,
                    slot,
                    frame_value,
                    sql_value,
                )
            elif spec.kind == "count":
                assert frame_value == sql_value, (group, slot, frame_value, sql_value)
            else:
                scales = []
                for reference_row in (frame_row, sql_row):
                    scale = 0.0
                    for index, row_group in enumerate(units["group_id"]):
                        if row_group != group or (spec.mask == "d" and not units["d"][index]):
                            continue
                        term = 1.0
                        for variable in spec.variables:
                            value = units[slot_values[variable]][index]
                            reference = reference_row[f"ref_{variable}"]
                            term *= value - reference
                        scale += abs(term)
                    scales.append(scale)
                scale = max(scales)
                assert abs(frame_value - sql_value) <= 1e-9 * scale, (
                    group,
                    slot,
                    frame_value,
                    sql_value,
                    scale,
                )


@example(seed=1, n_units=12, partitions=3, offset=1e9, near_constant_den=True, uptake=True)
@pytest.mark.slow
@settings(deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_units=st.integers(min_value=2, max_value=200),
    partitions=st.integers(min_value=1, max_value=7),
    offset=st.sampled_from((0.0, 1e9)),
    near_constant_den=st.booleans(),
    uptake=st.booleans(),
)
def test_frame_and_duckdb_emitters_agree(
    seed, n_units, partitions, offset, near_constant_den, uptake
):
    units = _property_units(seed, n_units, partitions, offset, near_constant_den, uptake)
    source = from_unit_summary(
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
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="pre_rev", kind="continuous")
                ),
            )
        ],
        uptake="took",
    )
    con = ibis.duckdb.connect()
    sql_rows = con.to_pyarrow(group_summary(_property_totals(con, units))).to_pylist()
    _assert_emitted_slots_agree(source.raw_moments, sql_rows, units)


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
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="pre_rev", kind="continuous")
                ),
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
