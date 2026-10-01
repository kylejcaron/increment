"""Cluster (randomization grain) support on the frame path.

Mirrors tests/query/test_cluster_builders.py on the SAME six-unit dataset,
so the two substrates are pinned against one hand-computed contract:

    control:   u1(s1, y=10), u2(s1, y=20), u3(s2, y=30)
    treatment: u4(s3, y=40), u5(s3, y=50), u6(s4, y=0)

Clustered moments (g_j = cluster sum, m_j = unit count; n is the cluster
count K, the y family is g_j, the den family is m_j and the x family is m_j
too - the slot that keeps size when a ratio spec's den holds its own
denominator total - all centered on their own group's reference):
    control:   K=2, g=(30, 30), m=(2, 1)
               ref_y=30, cy1=0, cy2=0
               ref_den=1.5, cden1=0, cden2=0.5, cyden=0
               ref_x=1.5, cx1=0, cx2=0.5, cxy=0, cxden=0.5
    treatment: K=2, g=(90, 0), m=(2, 1)
               ref_y=45, cy1=0, cy2=4050
               ref_den=1.5, cden1=0, cden2=0.5, cyden=45
               ref_x=1.5, cx1=0, cx2=0.5, cxy=45, cxden=0.5
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from increment.errors import CapabilityError, IncrementRuntimeWarning, InvalidRequestError
from increment.estimation.diagnostics import SRMResult, sample_ratio_mismatch
from increment.frame import MetricSpec, from_unit_panel, from_unit_summary
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    UptakeSpec,
)
from tests.warning_codes import warning_codes

UNITS = [
    ("u1", "control", "s1", 10.0),
    ("u2", "control", "s1", 20.0),
    ("u3", "control", "s2", 30.0),
    ("u4", "treatment", "s3", 40.0),
    ("u5", "treatment", "s3", 50.0),
    ("u6", "treatment", "s4", 0.0),
]

EXPECTED_CLUSTERED = {
    "control": {
        "n": 2,
        "ref_y": 30.0,
        "cy1": 0.0,
        "cy2": 0.0,
        "ref_den": 1.5,
        "cden1": 0.0,
        "cden2": 0.5,
        "cyden": 0.0,
        "ref_x": 1.5,
        "cx1": 0.0,
        "cx2": 0.5,
        "cxy": 0.0,
        "cxden": 0.5,
    },
    "treatment": {
        "n": 2,
        "ref_y": 45.0,
        "cy1": 0.0,
        "cy2": 4050.0,
        "ref_den": 1.5,
        "cden1": 0.0,
        "cden2": 0.5,
        "cyden": 45.0,
        "ref_x": 1.5,
        "cx1": 0.0,
        "cx2": 0.5,
        "cxy": 45.0,
        "cxden": 0.5,
    },
}


def _rows(**overrides: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"user_id": u, "variant": g, "store_id": s, "revenue": y, "solo_id": f"solo_{u}"}
        for u, g, s, y in UNITS
    ]
    for unit, patch in overrides.items():
        for row in rows:
            if row["user_id"] == unit:
                row.update(patch)
    return rows


def _frame(rows: list[dict[str, Any]], backend: str):
    if backend == "pandas":
        import pandas as pd

        return pd.DataFrame(rows)
    if backend == "polars":
        import polars as pl

        return pl.DataFrame(rows)
    import pyarrow as pa

    return pa.Table.from_pylist(rows)


def _source(rows=None, backend="pandas", **kwargs):
    defaults: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "control": "control",
        "metrics": {"revenue": "mean"},
        "cluster": "store_id",
    }
    defaults.update(kwargs)
    return from_unit_summary(_frame(rows or _rows(), backend), **defaults)


# ── the collapse, on every backend ───────────────────────────────────────


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_clustered_moments_match_hand_computed(backend):
    src = _source(backend=backend)
    by_group = {r["group_id"]: r for r in src.raw_moments}
    for group, expected in EXPECTED_CLUSTERED.items():
        row = by_group[group]
        for field, value in expected.items():
            assert row[field] == pytest.approx(value), (group, field)
        # The unit-grain uptake slots stay literal None - never a sum over
        # nothing; the x family holds cluster size (see EXPECTED_CLUSTERED).
        for slot in ("sum_d", "cyd", "cy2d", "cxd"):
            assert row[slot] is None, (group, slot)


def test_clustered_row_shape_matches_the_declared_builders_contract():
    """Both producers emit the SAME clustered wire shape, field for field:
    increment.query.schemas.GROUP_SUMMARY is the declaration, and
    tests/query/test_cluster_builders.py pins the ibis side against it."""
    from increment.query.schemas import GROUP_SUMMARY

    rows = _source().raw_moments
    assert rows
    for row in rows:
        assert set(row) == GROUP_SUMMARY


def test_clustered_moments_carry_uptake_in_x_family():
    """With uptake= given under an Encouragement design, each cluster's
    uptake TOTAL rides the x slot (grain-agnostic ArmStats.mean_x()/
    var_x()/cov_yx(), unlike the unit-grain binary-d cov_yd()/var_d()
    shortcuts), plus the new cxden cross moment - pinned against the
    same hand-computed values as
    tests/query/test_cluster_builders.py::
    test_group_summary_two_stage_collapse_carries_uptake_in_x_family.
    u1, u4, u5 took up: control D_j=(1, 0) for (s1, s2); treatment
    D_j=(2, 0) for (s3, s4)."""
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="took"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="test fixture, not a real design"
        ),
    )
    took = {"u1", "u4", "u5"}
    rows = [{**r, "took": 1 if r["user_id"] in took else 0} for r in _rows()]
    src = _source(rows=rows, design=design, uptake="took")
    by_group = {r["group_id"]: r for r in src.raw_moments}

    ctl = by_group["control"]
    assert ctl["ref_x"] == pytest.approx(0.5)
    assert ctl["cx2"] == pytest.approx(0.5)
    assert ctl["cxy"] == pytest.approx(0.0, abs=1e-9)
    assert ctl["cxden"] == pytest.approx(0.5)

    trt = by_group["treatment"]
    assert trt["ref_x"] == pytest.approx(1.0)
    assert trt["cx2"] == pytest.approx(2.0)
    assert trt["cxy"] == pytest.approx(90.0)
    assert trt["cxden"] == pytest.approx(1.0)

    for field in ("sum_d", "cyd", "cy2d", "cxd"):
        assert ctl[field] is None, field
        assert trt[field] is None, field


def test_singleton_clusters_reproduce_unclustered_moments():
    flat = {r["group_id"]: r for r in _source(cluster=None).raw_moments}
    solo = {r["group_id"]: r for r in _source(cluster="solo_id").raw_moments}
    for group, f in flat.items():
        c = solo[group]
        assert c["n"] == f["n"]
        assert c["ref_y"] == pytest.approx(f["ref_y"])
        assert c["cy1"] == pytest.approx(f["cy1"], abs=1e-9 * max(1.0, f["n"] * abs(f["ref_y"])))
        assert c["cy2"] == pytest.approx(f["cy2"])
        # m_j == 1 for every singleton cluster: the den family degenerates to
        # a constant 1, so its reference is 1 and its dispersion vanishes.
        assert c["ref_den"] == pytest.approx(1.0)
        assert c["cden1"] == pytest.approx(0.0, abs=1e-9 * f["n"])
        assert c["cden2"] == pytest.approx(0.0, abs=1e-9 * f["n"])
        assert c["cyden"] == pytest.approx(0.0, abs=1e-9 * max(1.0, f["n"] * abs(f["ref_y"])))


# ── refusals: loud, named, no silent absorption ──────────────────────────


def test_null_cluster_label_refuses():
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows=_rows(u3={"store_id": None}))
    assert raised.value.code == "source.frame.cluster_labels"
    assert raised.value.context["reason"] == "null_label"


def test_cluster_spanning_groups_refuses():
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows=_rows(u4={"store_id": "s1"}))
    assert raised.value.code == "source.frame.cluster_labels"
    assert raised.value.context["reason"] == "spanning_label"


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_null_cluster_label_refuses_before_a_spanning_label(backend):
    """One null label and one label spanning both arms: the null check
    fires first, so the null-label refusal wins over the span."""
    with pytest.raises(InvalidRequestError) as raised:
        _source(rows=_rows(u3={"store_id": None}, u4={"store_id": "s1"}), backend=backend)
    assert raised.value.code == "source.frame.cluster_labels"
    assert raised.value.context["reason"] == "null_label"


def test_missing_cluster_column_refuses():
    with pytest.raises(InvalidRequestError) as raised:
        _source(cluster="warehouse_id")
    assert raised.value.code == "source.frame.metric_missing"


def test_cluster_with_covariate_refuses():
    with pytest.raises(CapabilityError) as raised:
        _source(
            rows=_rows(),
            metrics=[MetricSpec(name="revenue", type="mean", covariate="solo_id")],
        )
    assert raised.value.code == "source.frame.cluster_capability"


def test_cluster_with_ratio_metric_and_covariate_refuses():
    """A clustered RATIO metric is servable, but not with CUPED on top: the
    covariate gate is unchanged and still fires first."""
    sessions = {"u1": 2.0, "u2": 3.0, "u3": 4.0, "u4": 3.0, "u5": 5.0, "u6": 2.0}
    rows = [{**r, "sessions": sessions[r["user_id"]]} for r in _rows()]
    with pytest.raises(CapabilityError) as raised:
        _source(
            rows=rows,
            metrics=[
                MetricSpec(
                    name="rps",
                    type="ratio",
                    numerator="revenue",
                    denominator="sessions",
                    covariate="solo_id",
                )
            ],
        )
    assert raised.value.code == "source.frame.cluster_capability"


def test_clustered_ratio_metric_carries_its_own_denominator():
    """A declared ratio metric's clustered den family holds its own
    per-cluster denominator total den_j, not the cluster size m_j.

    sessions: u1=2, u2=3, u3=4 | u4=3, u5=5, u6=2, so
    control  s1 -> (30, 5), s2 -> (30, 4);
    treatment s3 -> (90, 8), s4 -> (0, 2).
    """
    sessions = {"u1": 2.0, "u2": 3.0, "u3": 4.0, "u4": 3.0, "u5": 5.0, "u6": 2.0}
    rows = [{**r, "sessions": sessions[r["user_id"]]} for r in _rows()]
    src = _source(
        rows=rows,
        metrics=[MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")],
    )
    by_group = {r["group_id"]: r for r in src.raw_moments}
    ctl, trt = by_group["control"], by_group["treatment"]
    assert ctl["n"] == 2 and trt["n"] == 2
    assert ctl["ref_y"] == pytest.approx(30.0)
    assert ctl["ref_den"] == pytest.approx(4.5)  # mean(5, 4), NOT mean(2, 1)
    assert ctl["cden2"] == pytest.approx(0.5)
    assert ctl["cyden"] == pytest.approx(0.0, abs=1e-9)
    assert trt["ref_y"] == pytest.approx(45.0)
    assert trt["ref_den"] == pytest.approx(5.0)  # mean(8, 2)
    assert trt["cden2"] == pytest.approx(18.0)
    assert trt["cyden"] == pytest.approx(270.0)

    # Size survives the den rebinding in the x family: m = (2, 1) per arm, so
    # sum(num_j)/sum(m_j) and sum(den_j)/sum(m_j) stay recoverable.
    for row in (ctl, trt):
        assert row["ref_x"] == pytest.approx(1.5)
        assert row["cx1"] == pytest.approx(0.0, abs=1e-9)
        assert row["cx2"] == pytest.approx(0.5)
    assert ctl["cxy"] == pytest.approx(0.0, abs=1e-9)
    assert trt["cxy"] == pytest.approx(45.0)
    assert ctl["cxden"] == pytest.approx(0.5)
    assert trt["cxden"] == pytest.approx(3.0)
    assert ctl["ref_y"] / ctl["ref_x"] == pytest.approx(60.0 / 3.0)
    assert ctl["ref_den"] / ctl["ref_x"] == pytest.approx(9.0 / 3.0)
    assert trt["ref_y"] / trt["ref_x"] == pytest.approx(90.0 / 3.0)
    assert trt["ref_den"] / trt["ref_x"] == pytest.approx(10.0 / 3.0)


def test_clustered_ratio_readout_keeps_the_point_estimate_and_widens_the_se():
    """End-to-end on 20 clusters: the clustered ratio lift is bit-identical
    to the iid one (both are sum(num)/sum(den) per arm) while the SE and
    the cluster-based reference widen the interval."""
    from increment import readouts
    from increment.semantics.design import Randomized

    rows = []
    for j in range(20):
        arm = "control" if j % 2 == 0 else "treatment"
        for i in range(3):
            sess = 2.0 + (j % 3)
            rows.append(
                {
                    "user_id": f"u{j}_{i}",
                    "variant": arm,
                    "store_id": f"s{j}",
                    # A whole-store level shift is what makes the iid SE wrong.
                    "revenue": sess * (5.0 + 0.4 * j + (1.0 if arm == "treatment" else 0.0)),
                    "sessions": sess,
                }
            )
    spec = MetricSpec(name="rps", type="ratio", numerator="revenue", denominator="sessions")
    design = Randomized(control_group="control")
    # The shared small-K policy applies unchanged to the ratio path.
    with pytest.warns(IncrementRuntimeWarning) as rec:
        (clustered,) = readouts.run(_source(rows=rows, metrics=[spec], design=design))
    assert "estimation.engine.small_total_clusters" in warning_codes(rec)
    (flat,) = readouts.run(_source(rows=rows, metrics=[spec], cluster=None, design=design))

    clustered_lift = clustered.require_lift()
    flat_lift = flat.require_lift()
    assert clustered_lift.value == pytest.approx(flat_lift.value, rel=1e-12)
    assert clustered.abs_diff == pytest.approx(flat.abs_diff, rel=1e-12)
    assert clustered.n_clusters == 20
    assert clustered.dof == 9
    assert clustered.relative_confidence_set is not None
    clustered_set = clustered.relative_confidence_set
    assert clustered_set.geometry == "bounded"
    assert clustered_set.reference.kind == "t"
    # Both runs estimate their variance from the data, so both cut a t
    # reference; the clustered one just has far fewer degrees of freedom.
    assert flat.reference_kind == "t"
    assert clustered.reference_df is not None
    assert flat.reference_df is not None and flat.reference_df > clustered.reference_df
    clustered_lo, clustered_hi = clustered_set.intervals[0]
    flat_lo, flat_hi = flat_lift.lb, flat_lift.ub
    assert clustered_lo is not None and clustered_hi is not None
    assert flat_lo is not None and flat_hi is not None
    assert clustered_hi - clustered_lo > flat_hi - flat_lo


def test_cluster_with_quantile_metric_refuses():
    with pytest.raises(CapabilityError) as raised:
        _source(metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.5)])
    assert raised.value.code == "source.frame.cluster_capability"


def test_cluster_with_observational_design_constructs():
    """The adjusted estimators carry cluster-robust IF variance now, so the
    entry gate admits the pairing (tests/estimation/test_adjust_cluster.py
    covers the estimation side)."""
    design = Observational(
        control_group="control",
        adjustment=AdjustmentSet(covariates=("revenue",)),
    )
    assert _source(design=design).context.cluster == "store_id"


def test_cluster_with_encouragement_design_constructs():
    """Non-CUPED, non-ratio LATE now composes with a declared cluster.
    See tests/estimation/test_late_cluster.py for the estimation-side
    coverage; this only pins the frame-entry gate no longer refusing."""
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="test fixture, not a real design"
        ),
    )
    rows = [{**r, "took_up": 0} for r in _rows()]
    src = _source(rows=rows, design=design, uptake="took_up")
    assert src.context.cluster == "store_id"


def test_cluster_with_covariate_and_encouragement_refuses():
    """CUPED still refuses under a declared cluster, encouragement design
    included - the clustered collapse carries no per-unit covariate
    moments for its numerator adjustment."""
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="test fixture, not a real design"
        ),
    )
    rows = [{**r, "took_up": 0} for r in _rows()]
    with pytest.raises(CapabilityError) as raised:
        _source(
            rows=rows,
            design=design,
            uptake="took_up",
            metrics=[MetricSpec(name="revenue", type="mean", covariate="solo_id")],
        )
    assert raised.value.code == "source.frame.cluster_capability"


def test_from_unit_panel_refuses_cluster():
    rows = [
        {"user_id": u, "variant": g, "store_id": s, "day": f"2025-08-0{d}", "revenue": y}
        for u, g, s, y in UNITS
        for d in (1, 2)
    ]
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame["day"] = pd.to_datetime(frame["day"])
    with pytest.raises(CapabilityError) as excinfo:
        from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            cluster="store_id",
        )
    assert excinfo.value.code == "source.frame_panel.cluster_grain"
    assert excinfo.value.context["operation"] == "from_unit_panel(cluster=...)"
    assert excinfo.value.context["cluster"] == "store_id"


# ── end-to-end through the readout layer ─────────────────────────────────


def test_run_identity_singleton_clusters_match_unclustered_run():
    """Every store a single unit: the clustered run must reproduce the
    unclustered point estimate and SE exactly, with the cluster metadata
    stamped. Both runs estimate their variance from the data, so both cut a t
    reference; what still widens the clustered interval is its degrees of
    freedom -- one per cluster rather than the unclustered Welch df."""

    from increment.analysis import Analysis

    rows = [
        {
            "user_id": f"{arm[:1]}{i}",
            "variant": arm,
            "store_id": f"{arm[:1]}s{i}",
            "revenue": base + 0.1 * ((i % 5) - 2),
        }
        for arm, base in (("control", 5.0), ("treatment", 5.5))
        for i in range(25)
    ]

    def _run(cluster):
        return Analysis.from_unit_summary(
            _frame(rows, "pandas"),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            cluster=cluster,
        ).run()[0]

    flat, clustered = _run(None), _run("store_id")
    assert clustered.require_lift().value == pytest.approx(flat.require_lift().value, rel=1e-9)
    assert clustered.relative_confidence_set is not None
    clustered_set = clustered.relative_confidence_set
    assert clustered_set.geometry == "bounded"
    assert clustered.abs_se == pytest.approx(flat.abs_se, rel=1e-12)
    # Singleton clusters leave the additive inference untouched, reference
    # and all; the two runs differ only in the relative reference's df.
    assert clustered.abs_reference_kind == flat.abs_reference_kind == "t"
    assert clustered.abs_reference_df == pytest.approx(flat.abs_reference_df, rel=1e-12)
    assert clustered_set.reference.kind == "t"
    assert flat.reference_kind == "t"
    assert clustered.dof == 24.0 and flat.dof is None
    assert flat.reference_df is not None and flat.reference_df > clustered.reference_df
    clustered_lo, clustered_hi = clustered_set.intervals[0]
    flat_lift = flat.require_lift()
    flat_lo, flat_hi = flat_lift.lb, flat_lift.ub
    assert clustered_lo is not None and clustered_hi is not None
    assert flat_lo is not None and flat_hi is not None
    assert clustered_hi - clustered_lo > flat_hi - flat_lo


# ── the sample-ratio check at the randomization grain ────────────────────


def _sized_rows(sizes: dict[str, list[int]]) -> list[dict[str, Any]]:
    """One row per unit, each arm built from explicit per-cluster sizes."""
    return [
        {
            "user_id": f"{arm}_{j}_{i}",
            "variant": arm,
            "store_id": f"{arm}_s{j}",
            "revenue": 1.0 + i,
            "solo_id": f"solo_{arm}_{j}_{i}",
        }
        for arm, cluster_sizes in sizes.items()
        for j, m in enumerate(cluster_sizes)
        for i in range(m)
    ]


def _srm(
    rows=None,
    *,
    alpha: float = 0.001,
    inference: Literal["always_valid", "fixed"] = "always_valid",
    **kwargs,
):
    from increment import readouts
    from increment.semantics.design import Randomized

    kwargs.setdefault(
        "design",
        Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    )
    return readouts.srm(
        _source(rows=rows, **kwargs),
        alpha=alpha,
        inference=inference,
    )


def test_srm_counts_clusters_not_units_on_a_clustered_source():
    """control s1(u1,u2)+s2(u3), treatment s3(u4,u5)+s4(u6): K=2 per arm
    over 3 units per arm. Randomization happened over stores, so the
    chi-square reads there and the unit counts ride along as context."""
    result = _srm()
    assert isinstance(result, SRMResult)
    assert result.grain == "cluster"
    assert result.observed == {"control": 2, "treatment": 2}
    assert result.unit_counts == {"control": 3, "treatment": 3}
    assert result.df == 1
    assert result.is_srm is False


def test_cluster_srm_zero_fills_declared_missing_arm_for_always_valid_prefix():
    """A cluster-randomized cumulative prefix retains the unobserved arm."""
    result = _srm(rows=_sized_rows({"control": [1] * 14}))

    assert isinstance(result, SRMResult)
    assert result.inference == "always_valid"
    assert result.observed == {"control": 14, "treatment": 0}
    assert result.unit_counts == {"control": 14, "treatment": 0}
    assert result.is_srm is True


def test_cluster_srm_zero_fill_preserves_unexpected_observed_arm_for_strict_mismatch():
    """Cluster support completion leaves a stray observed arm for the strict diagnostic."""
    with pytest.raises(InvalidRequestError) as raised:
        _srm(rows=_sized_rows({"control": [1] * 14, "holdout": [1] * 3}))
    assert raised.value.code == "estimation.diagnostics.expected_keys_do"


def test_unit_count_skew_is_reported_but_not_tested_under_a_cluster():
    """20 stores per arm sized 2 vs 6: a unit-grain check would flag the
    40-vs-120 split, but cluster SIZE is not what the randomizer
    controlled, so only the balanced cluster counts are tested."""
    result = _srm(rows=_sized_rows({"control": [2] * 20, "treatment": [6] * 20}))
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 20, "treatment": 20}
    assert result.is_srm is False
    assert result.unit_counts == {"control": 40, "treatment": 120}
    assert sample_ratio_mismatch(result.unit_counts, inference="fixed").is_srm is True


def test_cluster_count_skew_flags_even_when_unit_counts_balance():
    """The converse miss the old unit-grain seam could not see: 20 control
    stores against 8 treatment stores, sized so both arms land on 40
    units."""
    result = _srm(
        rows=_sized_rows({"control": [2] * 20, "treatment": [5] * 8}),
        inference="fixed",
        alpha=0.05,
    )
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 20, "treatment": 8}
    assert result.is_srm is True
    assert result.unit_counts == {"control": 40, "treatment": 40}
    assert sample_ratio_mismatch(result.unit_counts, inference="fixed").is_srm is False


def test_srm_stays_at_unit_grain_without_a_declared_cluster():
    """Same six units, no cluster declared: unchanged unit-grain behaviour,
    and no descriptive unit counts duplicating ``observed``."""
    result = _srm(cluster=None)
    assert isinstance(result, SRMResult)
    assert result.grain == "unit"
    assert result.observed == {"control": 3, "treatment": 3}
    assert result.unit_counts == {}


def test_clustered_srm_keeps_the_unassigned_units_as_accounting():
    """An excluded null-label unit stays an accounting entry at cluster
    grain too: no arm, no degree of freedom, and out of the descriptive
    unit counts (it already has its own field)."""
    rows = [
        *_rows(),
        {
            "user_id": "u7",
            "variant": None,
            "store_id": "s5",
            "revenue": 1.0,
            "solo_id": "solo_u7",
        },
    ]
    result = _srm(rows=rows, on_unassigned="exclude")
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 2, "treatment": 2}
    assert result.df == 1
    assert result.unassigned_units == 1
    assert result.unit_counts == {"control": 3, "treatment": 3}


def test_unit_counts_report_units_not_clusters_under_a_declared_cluster():
    """The moments carry n = K on this shape, so unit_counts() must read the
    retained frame instead of the moments to keep its own name honest."""
    src = _source()
    assert {r["group_id"]: r["n"] for r in src.raw_moments} == {"control": 2, "treatment": 2}
    assert src.unit_counts() == {"control": 3, "treatment": 3}
    assert src.cluster_counts() == {"control": 2, "treatment": 2}


def test_cluster_counts_refuses_without_a_declared_cluster():
    with pytest.raises(CapabilityError) as raised:
        _source(cluster=None).cluster_counts()
    assert raised.value.code == "source.frame.cluster_grain"


def test_cluster_counts_refuses_on_the_panel_shape():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"user_id": u, "variant": g, "day": f"2025-08-0{d}", "revenue": y}
            for u, g, _, y in UNITS
            for d in (1, 2)
        ]
    )
    frame["day"] = pd.to_datetime(frame["day"])
    src = from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        src.cluster_counts()
    assert raised.value.code == "source.frame_panel.cluster_grain"


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("cluster", ["store_id", "cluster_id"])
def test_unit_frame_deduplicates_requested_cluster_metadata(backend, cluster):
    import narwhals as nw

    rows = [{**row, "cluster_id": row["store_id"]} for row in _rows()]
    source = _source(rows=rows, backend=backend, cluster=cluster)
    result = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=[cluster, cluster]),
        eager_only=True,
    )
    assert len(result.columns) == len(set(result.columns))
    assert result["cluster_id"].to_list() == [row["store_id"] for row in rows]
    assert result[cluster].to_list() == [row["store_id"] for row in rows]
    assert result["y"].to_list() == [row["revenue"] for row in rows]


@pytest.mark.parametrize("cluster", [None, "store_id"])
def test_unit_frame_refuses_reserved_cluster_covariate(cluster):
    rows = [{**row, "cluster_id": 123} for row in _rows()]
    source = _source(rows=rows, cluster=cluster)
    with pytest.raises(InvalidRequestError) as caught:
        source.unit_frame(source.context.metrics[0], covariates=["cluster_id"])
    assert caught.value.code == "frame.frame_totals.unit_covariate_reserved"
    assert caught.value.context["column"] == "cluster_id"
    assert caught.value.context["cluster"] == cluster
