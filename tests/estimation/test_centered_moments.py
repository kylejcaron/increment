"""CenteredMoments: the algebra view behind ArmStats and its wire round trip."""

from typing import Any

import numpy as np
import pytest

from increment.estimation.armstats import ArmStats, CenteredMoments


def _arm(x_role: str | None, *, uptake: bool) -> ArmStats:
    rng = np.random.default_rng(7)
    y = rng.normal(1e6, 1.0, 50)
    x = rng.normal(3.0, 2.0, 50) + 0.5 * (y - y.mean())
    den = rng.poisson(3.0, 50).astype(float) + 1.0
    d = (rng.random(50) < 0.4).astype(float)
    ry, rx, rd = float(y.mean()), float(x.mean()), float(den.mean())
    fields: dict[str, Any] = {
        "n": 50,
        "ref_y": ry,
        "cy1": float(np.sum(y - ry)),
        "cy2": float(np.sum((y - ry) ** 2)),
        "ref_x": rx,
        "cx1": float(np.sum(x - rx)),
        "cx2": float(np.sum((x - rx) ** 2)),
        "cxy": float(np.sum((x - rx) * (y - ry))),
        "x_role": x_role,
        "ref_den": rd,
        "cden1": float(np.sum(den - rd)),
        "cden2": float(np.sum((den - rd) ** 2)),
        "cyden": float(np.sum((y - ry) * (den - rd))),
        "cxden": float(np.sum((x - rx) * (den - rd))),
    }
    if uptake:
        fields.update(
            sum_d=float(d.sum()),
            cyd=float(np.sum(d * (y - ry))),
            cy2d=float(np.sum(d * (y - ry) ** 2)),
            cxd=float(np.sum(d * (x - rx))),
        )
    return ArmStats(study_id="s", metric="m", group_id="A", **fields)


@pytest.mark.parametrize("x_role", ["covariate", "cluster_size", "uptake_total", None])
@pytest.mark.parametrize("uptake", [False, True])
def test_wire_round_trip_through_the_view(x_role, uptake):
    """from_moments(arm.moments) is the same wire row: every slot and x_role."""
    arm = _arm(x_role, uptake=uptake)
    back = ArmStats.from_moments(arm.moments, study_id="s", metric="m", group_id="A")
    assert back.model_dump() == arm.model_dump()


def test_alias_reads_like_its_source():
    """An aliased variable reports its source's mean, variance and covariances."""
    arm = _arm("uptake_total", uptake=False)
    aliased = arm.moments.with_alias("y2", of="y")
    assert aliased.mean("y2") == arm.mean_y()
    assert aliased.var("y2", what="test") == arm.var_y()
    assert aliased.cov("y2", "den") == arm.cov_yden()
    assert aliased.cov("uptake", "y2") == arm.cov_yx()


def _moments(y: np.ndarray, x: np.ndarray, d: np.ndarray) -> CenteredMoments:
    """One partition's centered moments of ``(x, y)`` under mask ``d``, from raw values."""
    ry, rx = float(y.mean()), float(x.mean())
    dy, dx = y - ry, x - rx
    return CenteredMoments(
        n=len(y),
        variables=("x", "y"),
        ref={"x": rx, "y": ry},
        c1={
            ("x", None): float(np.sum(dx)),
            ("y", None): float(np.sum(dy)),
            ("y", "d"): float(np.sum(d * dy)),
        },
        c2={
            ("x", "x", None): float(np.sum(dx * dx)),
            ("y", "y", None): float(np.sum(dy * dy)),
            ("x", "y", None): float(np.sum(dx * dy)),
            ("y", "y", "d"): float(np.sum(d * dy * dy)),
        },
        count={"d": float(d.sum())},
    )


def test_combine_recovers_single_pass_moments_across_offset_partitions():
    """Pooling parts whose means sit near 1e6 reproduces one pass over the raw values."""
    rng = np.random.default_rng(11)
    sizes, offsets = (40, 7, 25), (1e6, 1e6 + 3.0, 1e6 - 2.0)
    ys, xs, ds = [], [], []
    for size, offset in zip(sizes, offsets, strict=True):
        y = rng.normal(offset, 1.0, size)
        ys.append(y)
        xs.append(rng.normal(offset / 2.0, 3.0, size) + 0.5 * (y - y.mean()))
        # Uptake leans on the outcome so every mask covariance is far from zero.
        ds.append(((y - offset) + rng.normal(0.0, 1.0, size) > 0.3).astype(float))
    pooled = CenteredMoments.combine(
        [_moments(y, x, d) for y, x, d in zip(ys, xs, ds, strict=True)]
    )
    y, x, d = np.concatenate(ys), np.concatenate(xs), np.concatenate(ds)

    assert pooled.n == len(y)
    assert pooled.mean("y") == pytest.approx(y.mean(), rel=1e-12)
    assert pooled.mean("x") == pytest.approx(x.mean(), rel=1e-12)
    assert pooled.var("y", what="test") == pytest.approx(y.var(ddof=1), rel=1e-12)
    assert pooled.var("x", what="test") == pytest.approx(x.var(ddof=1), rel=1e-12)
    assert pooled.cov("x", "y") == pytest.approx(np.cov(x, y, ddof=1)[0, 1], rel=1e-12)
    assert pooled.mask_mean("d") == pytest.approx(d.mean(), rel=1e-12)
    assert pooled.masked_mean("y", "d") == pytest.approx((d * y).mean(), rel=1e-12)
    assert pooled.cov_mask("y", "d") == pytest.approx(np.cov(y, d, ddof=1)[0, 1], rel=1e-12)
    assert pooled.masked_product_var("y", "d", what="test") == pytest.approx(
        (d * y).var(ddof=1), rel=1e-12
    )
    assert pooled.cov_masked_product("y", "d") == pytest.approx(
        np.cov(y, d * y, ddof=1)[0, 1], rel=1e-12
    )
