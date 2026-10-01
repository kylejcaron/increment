"""CUPED-adjusted LATE: dispatch, guards, the report identity, and the
variance-backfire annotation.

Deterministic checks only - the Monte-Carlo calibration/coverage evidence
lives in ``test_cuped_late_recovery.py``. The estimator adjusts the LATE
NUMERATOR only: the first stage stays raw so ``min_first_stage_z`` reads an
unchanged z, and the adjusted numerator is bit-for-bit the cuped ITT row's
``abs_diff`` so a report's rows stay mutually consistent.
"""

from __future__ import annotations

import math

import pytest

from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.cuped import cuped_adjust, pooled_theta
from increment.estimation.encouragement import (
    _first_stage,
    _late_additive_cuped,
    estimate_encouragement,
)
from increment.estimation.engine import Method, estimate_lift
from increment.semantics.design import Encouragement
from increment.semantics.models import MeanMetric, Measure, RatioMetric

METRIC = MeanMetric(name="rev", entity="user_id", fact="orders", aggregation="sum")
CUPED = [Method(name="cuped", variance_reduction="cuped")]


def _design(**over):
    base = {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "help_click"},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "unclicked button assumed inert",
        },
        "one_sided": False,
        "min_first_stage_z": 3.0,
    }
    base.update(over)
    return Encouragement.model_validate(base)


def _raw_row(gid, x, y, d, metric="rev"):
    """One format-1 (raw additive sums) group_summary row from per-unit arrays."""
    n = len(x)
    return {
        "experiment_id": "s",
        "metric": metric,
        "group_id": gid,
        "n": n,
        "sum_y": sum(y),
        "sum_y2": sum(v * v for v in y),
        "sum_x": sum(x),
        "sum_x2": sum(v * v for v in x),
        "sum_xy": sum(a * b for a, b in zip(x, y, strict=True)),
        "sum_d": sum(d),
        "sum_yd": sum(a * b for a, b in zip(y, d, strict=True)),
        "sum_y2d": sum(a * a * b for a, b in zip(y, d, strict=True)),
        "sum_xd": sum(a * b for a, b in zip(x, d, strict=True)),
    }


def _row(gid, x, y, d, metric="rev"):
    """One centered (format-2) group_summary row from per-unit arrays."""
    return centered_row_from_raw_sums(_raw_row(gid, x, y, d, metric))


def _units(n, *, seed, tau=2.0, theta_d=0.0, sd_noise=1.0, z=1):
    """Per-unit (x, y, d) with uptake probability tilted by ``theta_d`` * x.

    Pure-python LCG rather than numpy: this file is in the fast suite and
    every fixture here is a few hundred units.
    """
    state = seed
    xs, ys, ds = [], [], []

    def rand():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 11) / float(1 << 53)

    for _ in range(n):
        # Sum of 4 uniforms: enough spread for a well-conditioned covariate.
        x = sum(rand() for _ in range(4)) - 2.0
        p_uptake = min(max(0.35 + theta_d * x, 0.02), 0.98) if z else 0.05
        d = 1.0 if rand() < p_uptake else 0.0
        y = 5.0 + 0.9 * x + tau * d + sd_noise * (sum(rand() for _ in range(4)) - 2.0)
        xs.append(x)
        ys.append(y)
        ds.append(d)
    return xs, ys, ds


def _raw_rows(n=600, *, seed=20260811, tau=2.0, theta_d=0.0, sd_noise=1.0, metric="rev"):
    xc, yc, dc = _units(n, seed=seed, tau=tau, theta_d=theta_d, sd_noise=sd_noise, z=0)
    xt, yt, dt = _units(n, seed=seed + 7919, tau=tau, theta_d=theta_d, sd_noise=sd_noise, z=1)
    return [_raw_row("control", xc, yc, dc, metric), _raw_row("treat", xt, yt, dt, metric)]


def _rows(n=600, *, seed=20260811, tau=2.0, theta_d=0.0, sd_noise=1.0, metric="rev"):
    raw = _raw_rows(n=n, seed=seed, tau=tau, theta_d=theta_d, sd_noise=sd_noise, metric=metric)
    return [centered_row_from_raw_sums(r) for r in raw]


def _arms(rows):
    """(control, treatment) ArmStats straight from moment rows."""
    by_gid = {
        r["group_id"]: ArmStats(
            study_id=r["experiment_id"], **{k: v for k, v in r.items() if k != "experiment_id"}
        )
        for r in rows
    }
    return by_gid["control"], by_gid["treat"]


def _late(res, scale="absolute"):
    return [r for r in res if r.estimand == "late" and r.value_scale == scale]


def _TOTALS_SCHEMA():
    """``unit_totals``'s arrow schema, pinned: DuckDB refuses a NULL-typed
    column, and ``x``/``y_den`` are all-null in these fixtures."""
    import pyarrow as pa

    return pa.schema(
        [
            ("unit_id", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", pa.string()),
            ("metric", pa.string()),
            ("y", pa.float64()),
            ("x", pa.float64()),
            ("y_den", pa.float64()),
            ("d", pa.float64()),
        ]
    )


# The report identity (design validation test 2)


def test_cuped_late_numerator_is_exactly_the_cuped_itt_abs_diff():
    """cuped-LATE == cuped-ITT ``abs_diff`` / compliance, EXACTLY.

    Both sides build the numerator from the same per-pair pooled theta and
    the same per-arm-adjusted-THEN-differenced op order, so the two rows in
    one report never display mutually inconsistent ratios. Float equality
    (not approx) is the point: it is what pins the shared theta.
    """
    rows = _rows(theta_d=0.05)
    c, t = _arms(rows)

    (itt,) = estimate_lift([METRIC], rows, control_group="control", methods=CUPED).results
    c_adj, t_adj = cuped_adjust([c, t])
    numerator = t_adj.mean - c_adj.mean

    assert itt.abs_diff is not None
    assert itt.abs_diff == numerator, "ITT abs_diff must BE the adjusted-mean difference"

    b, _ = _first_stage(t, c)
    tau, _, _ = _late_additive_cuped(t, c)
    assert tau == numerator / b
    assert tau == itt.abs_diff / b


def test_cuped_late_end_to_end_matches_the_itt_over_compliance_ratio():
    """The same identity through the public surface (the conjugate posterior
    with the default near-flat prior shifts the point by ~1e-15 relative, so
    this leg is approx where the internal one is exact)."""
    rows = _rows(theta_d=0.05)
    res = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("itt", "late", "compliance"), methods=CUPED
    ).results
    (itt,) = [r for r in res if r.estimand == "itt"]
    (late,) = _late(res)
    (compliance,) = [r for r in res if r.estimand == "compliance"]

    assert itt.abs_diff is not None
    # The compliance row's absolute difference IS the raw first stage.
    assert compliance.abs_diff is not None
    assert late.require_lift().value == pytest.approx(itt.abs_diff / compliance.abs_diff, rel=1e-12)


# The cross term (cxd) is load-bearing


def test_se_carries_the_theta_cov_xd_cross_term():
    """``Cov(Y - theta*X, D) = Cov(Y,D) - theta*Cov(X,D)``.

    Pinned against the naive covariance (``cov_yd`` alone) on a DGP where
    the covariate predicts uptake: dropping the correction is what measures
    0.938 coverage instead of 0.951 in the recovery suite.
    """
    rows = _rows(theta_d=0.12)
    c, t = _arms(rows)
    _, se, _ = _late_additive_cuped(t, c)

    theta, _ = pooled_theta([c, t])
    c_adj, t_adj = cuped_adjust([c, t])
    b, var_b = _first_stage(t, c)
    tau = (t_adj.mean - c_adj.mean) / b
    var_a = t_adj.var / t.n + c_adj.var / c.n

    correct = sum((a.cov_yd() - theta * a.cov_xd()) / a.n for a in (t, c))
    naive = sum(a.cov_yd() / a.n for a in (t, c))
    assert correct != naive, "fixture must have cov(X, D) != 0 or it proves nothing"

    def _se(cov_ab):
        return math.sqrt(max((var_a - 2 * tau * cov_ab + tau**2 * var_b) / b**2, 0.0))

    assert se == _se(correct)
    assert se != _se(naive)


def _late_influence_oracle(t: ArmStats, c: ArmStats) -> tuple[float, float]:
    """``(tau, se)`` for the numerator-only CUPED Wald ratio by an explicit
    delta method over each arm's ``(mean_y, mean_x, mean_d)`` with that arm's
    full 3x3 moment covariance. Theta is the contrast-optimal inverse-n form
    written out here, not read from the estimator. The adjusted means are
    written in the pooled-anchor form and differentiated as such, so the
    anchor's cancellation in the numerator is a consequence, not an
    assumption, and the covariate-uptake cross moment enters through the
    covariance block."""
    import numpy as np

    theta = (t.cov_yx() / t.n + c.cov_yx() / c.n) / (t.var_x() / t.n + c.var_x() / c.n)
    w_t = t.n / (t.n + c.n)
    w_c = 1.0 - w_t
    anchor = w_t * t.mean_x() + w_c * c.mean_x()
    mu_t = t.mean_y() - theta * (t.mean_x() - anchor)
    mu_c = c.mean_y() - theta * (c.mean_x() - anchor)
    a = mu_t - mu_c
    b = t.mean_d() - c.mean_d()
    tau = a / b
    # d(tau)/d(a) = 1/b, d(tau)/d(b) = -a/b**2; columns are
    # (mean_y_t, mean_x_t, mean_d_t, mean_y_c, mean_x_c, mean_d_c).
    d_a = np.array(
        [1.0, -theta * (1.0 - w_t) - theta * w_t, 0.0, -1.0, theta * w_c + theta * (1.0 - w_c), 0.0]
    )
    d_b = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    grad = d_a / b - d_b * a / b**2
    sigma = np.zeros((6, 6))
    for offset, arm in ((0, t), (3, c)):
        block = np.array(
            [
                [arm.var_y(), arm.cov_yx(), arm.cov_yd()],
                [arm.cov_yx(), arm.var_x(), arm.cov_xd()],
                [arm.cov_yd(), arm.cov_xd(), arm.var_d()],
            ]
        )
        sigma[offset : offset + 3, offset : offset + 3] = block / arm.n
    return tau, math.sqrt(float(grad @ sigma @ grad))


def test_unequal_allocation_late_se_is_the_full_influence_function():
    """3:1 allocation with cov(X, D) != 0 in both arms: the public LATE row's
    SE equals an explicit outcome/uptake/covariate delta method, and drops to
    a different number if the covariate-uptake cross covariance is discarded."""
    xc, yc, dc = _units(300, seed=101, tau=2.0, theta_d=0.12, z=0)
    xt, yt, dt = _units(900, seed=202, tau=2.0, theta_d=0.12, z=1)
    # Give control some real uptake so cov(X, D) is nonzero there too.
    dc = [
        1.0 if (i % 7 == 0 and x > 0.2) else d for i, (x, d) in enumerate(zip(xc, dc, strict=True))
    ]
    rows = [_row("control", xc, yc, dc), _row("treat", xt, yt, dt)]
    c, t = _arms(rows)
    assert t.n == 3 * c.n
    assert all(a.cov_xd() != 0.0 for a in (t, c))

    tau_oracle, se_oracle = _late_influence_oracle(t, c)

    (late,) = _late(
        estimate_encouragement(
            [METRIC], rows, _design(), estimands=("late",), methods=CUPED
        ).results
    )
    lift = late.require_lift()
    assert lift.value is not None
    assert lift.lb is not None and lift.ub is not None
    assert lift.value == pytest.approx(tau_oracle, rel=1e-12)
    half_width = (lift.ub - lift.lb) / 2.0
    assert half_width == pytest.approx(1.959963984540054 * se_oracle, rel=1e-9)

    # Discarding the covariate-uptake cross covariance (a standalone adjusted
    # Y variance plus the raw cov(Y, D)) is a materially different number.
    theta, _ = pooled_theta([c, t])
    c_adj, t_adj = cuped_adjust([c, t])
    b = t.mean_d() - c.mean_d()
    var_b = t.var_d() / t.n + c.var_d() / c.n
    var_a = t_adj.var / t.n + c_adj.var / c.n
    naive_cov = sum(a.cov_yd() / a.n for a in (t, c))
    naive_se = math.sqrt((var_a - 2 * lift.value * naive_cov + lift.value**2 * var_b) / b**2)
    assert abs(naive_se / se_oracle - 1.0) > 1e-3


# Refusal matrix (design validation test 7)


def test_missing_cxd_with_covariate_present_refuses_by_name():
    """A summary aggregated before this estimator existed carries the
    covariate family but no cxd - distinguished from an unmaterialised
    covariate so the fix ('re-aggregate') is not confused with 'declare a
    covariate'."""
    rows = [{**r, "cxd": None} for r in _rows()]
    with pytest.raises(InvalidRequestError) as exc:
        estimate_encouragement([METRIC], rows, _design(), estimands=("late",), methods=CUPED)
    assert exc.value.code == "estimation.encouragement.cuped_adjusted_late"


def test_missing_covariate_reuses_the_cuped_refusal():
    rows = []
    for r in _rows():
        row = {**r, "ref_x": None, "cx1": None, "cx2": None, "cxy": None, "cxd": None}
        row.pop("x_role", None)
        rows.append(row)
    with pytest.raises(InvalidRequestError) as exc:
        estimate_encouragement([METRIC], rows, _design(), estimands=("late",), methods=CUPED)
    assert exc.value.code == "estimation.cuped.arm_no_covariate"


def test_constant_covariate_refuses_via_the_shared_theta_helper():
    """theta is undefined without covariate spread; the LATE path inherits
    cuped_adjust's refusal rather than dividing by zero."""
    rows = [
        centered_row_from_raw_sums(
            {
                **r,
                "sum_x": float(r["n"]),
                "sum_x2": float(r["n"]),
                "sum_xy": r["sum_y"],
                "sum_xd": r["sum_d"],
            }
        )
        for r in _raw_rows()
    ]
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], rows, _design(), estimands=("late",), methods=CUPED)
    assert exc_info.value.code == "estimation.cuped.covariate_zero_variance"


def test_ratio_metric_with_cuped_refused_on_the_late_only_path():
    """estimands=("late",) bypasses estimate_lift, so the ratio refusal has
    to exist here too - otherwise the numerator's LATE ships mislabelled as
    the ratio's LATE."""
    ratio = RatioMetric(
        name="rev",
        entity="user_id",
        numerator=Measure(fact="orders", aggregation="sum"),
        denominator=Measure(fact="sessions", aggregation="count"),
    )
    rows = [
        centered_row_from_raw_sums({**r, "sum_den": 100.0, "sum_den2": 200.0, "sum_yden": 150.0})
        for r in _raw_rows()
    ]
    with pytest.raises(UnsupportedRequestError) as exc_info:
        estimate_encouragement([ratio], rows, _design(), estimands=("late",), methods=CUPED)
    assert exc_info.value.code == "estimation.encouragement.late.ratio"


def test_cuped_label_without_cuped_reduction_refused_on_late_only_call():
    """The reserved-label guard lives in estimate_lift, which a late-only
    call never reaches."""
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement(
            [METRIC],
            _rows(),
            _design(),
            estimands=("late",),
            methods=[Method(name="cuped")],
        )
    assert exc_info.value.code == "estimation.engine.method.name_without_variance"


@pytest.mark.parametrize("name", ["iptw", "dml", "aipw"])
def test_observational_labels_refused_on_late_only_call(name):
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement(
            [METRIC], _rows(), _design(), estimands=("late",), methods=[Method(name=name)]
        )
    assert exc_info.value.code == "estimation.engine.method_name_observational"


def test_unregistered_variance_reduction_refused_on_late_only_call():
    with pytest.raises(UnsupportedRequestError) as exc_info:
        estimate_encouragement(
            [METRIC],
            _rows(),
            _design(),
            estimands=("late",),
            methods=[Method(name="bogus", variance_reduction="bogus_vr")],
        )
    assert exc_info.value.code == "estimation.variance.registry.no_registered_available"


def test_one_sided_control_uptake_still_hard_errors_under_cuped():
    rows = _rows()
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement(
            [METRIC], rows, _design(one_sided=True), estimands=("late",), methods=CUPED
        )
    assert exc_info.value.code == "estimation.encouragement.one_sided_encouragement"


def test_asof_shaped_moments_refuse_cuped_via_the_covariate_error():
    """Deliberately incomplete legacy-shaped rows still refuse CUPED.

    These synthetic rows omit all covariate moments, including ``cxd``; the
    refusal protects callers that try to adjust a summary that was aggregated
    without the required pre-period data.
    """
    rows = []
    for r in _rows():
        row = {**r, "ref_x": None, "cx1": None, "cx2": None, "cxy": None, "cxd": None}
        row.pop("x_role", None)
        rows.append(row)
    with pytest.raises(InvalidRequestError) as exc_info:
        estimate_encouragement([METRIC], rows, _design(), estimands=("late",), methods=CUPED)
    assert exc_info.value.code == "estimation.cuped.arm_no_covariate"


# Row shape, labels, cardinality


def test_relative_late_row_is_withheld_under_a_cuped_method():
    """Emitting the unadjusted complier-relative form under a cuped label
    would mislabel it; the additive row names the withholding instead."""
    rows = _rows()
    res = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("late",), methods=CUPED
    ).results
    assert _late(res, "relative") == []
    (additive,) = _late(res)
    assert additive.method == "cuped"
    assert additive.note is not None
    assert "relative form withheld" in additive.note
    assert "sum(x*y*d)" in additive.note

    plain = estimate_encouragement([METRIC], rows, _design(), estimands=("late",)).results
    assert _late(plain, "relative"), "the unadjusted path still emits the relative form"


def test_compliance_row_is_never_adjusted_under_a_cuped_method():
    rows = _rows()
    res = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("compliance", "late"), methods=CUPED
    ).results
    compliance = [r for r in res if r.estimand == "compliance"]
    assert compliance
    assert {r.method for r in compliance} == {"unadjusted"}


def test_one_late_row_family_per_method():
    """The per-method loop replaces the blanket refusal: an unadjusted and a
    cuped method in the same call each get their own labelled rows."""
    rows = _rows()
    res = estimate_encouragement(
        [METRIC],
        rows,
        _design(),
        estimands=("late",),
        methods=[Method(name="unadjusted"), *CUPED],
    ).results
    assert [(r.method, r.value_scale) for r in _late(res) + _late(res, "relative")] == [
        ("unadjusted", "absolute"),
        ("cuped", "absolute"),
        ("unadjusted", "relative"),
    ]


def test_default_methods_still_label_late_unadjusted():
    """methods=None must stay byte-identical to v1: one unadjusted family."""
    rows = _rows()
    default = estimate_encouragement([METRIC], rows, _design(), estimands=("late",)).results
    explicit = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("late",), methods=[Method(name="unadjusted")]
    ).results
    assert default == explicit
    assert {r.method for r in default} == {"unadjusted"}


# The gate is untouched (design validation test 6)


def _weak_rows():
    """Near-zero first stage: uptake barely differs across arms."""
    xc, yc, dc = _units(400, seed=31337, theta_d=0.0, z=0)
    xt, yt, dt = _units(400, seed=31337, theta_d=0.0, z=0)
    return [_row("control", xc, yc, dc), _row("treat", xt, yt, dt)]


def test_weak_first_stage_gate_is_identical_with_and_without_cuped():
    """Numerator-only adjustment cannot move the gate: it reads the RAW
    first stage. Same suppression, same z, same text."""
    rows = _weak_rows()
    plain = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("compliance", "late")
    ).results
    adjusted = estimate_encouragement(
        [METRIC], rows, _design(), estimands=("compliance", "late"), methods=CUPED
    ).results
    assert _late(plain) == [] and _late(adjusted) == []
    plain_notes = [r.note for r in plain if r.estimand == "compliance"]
    assert plain_notes == [r.note for r in adjusted if r.estimand == "compliance"]
    assert any("late suppressed" in (n or "") for n in plain_notes)


def test_near_gate_caveat_text_is_identical_with_and_without_cuped():
    rows = _rows(n=600)
    c, t = _arms(rows)
    b, var_b = _first_stage(t, c)
    z_fs = b / math.sqrt(var_b)
    # Park the gate just under the observed z so the near-gate margin fires.
    design = _design(min_first_stage_z=z_fs - 0.5)
    (plain,) = _late(estimate_encouragement([METRIC], rows, design, estimands=("late",)).results)
    (adjusted,) = _late(
        estimate_encouragement([METRIC], rows, design, estimands=("late",), methods=CUPED).results
    )

    # The caveat names the RAW first stage's z, so both methods carry the
    # same trailing segment - located by that observed value, not wording.
    def near_gate_segment(note: str | None) -> str:
        segments = [s for s in (note or "").split(";") if f"z={z_fs:.2f}" in s]
        assert len(segments) == 1, f"expected exactly one near-gate segment in {note!r}"
        return segments[0]

    assert near_gate_segment(plain.note) == near_gate_segment(adjusted.note)


# Backfire annotation (design validation test 9c, deterministic leg)


def _backfire_rows(n=800, seed=20260811):
    """X drives uptake hard, predicts Y weakly net of uptake, tau is large -
    the regime where a numerator-only adjustment costs power."""
    state = seed
    out = []

    def rand():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (state >> 11) / float(1 << 53)

    for gid, z in (("control", 0), ("treat", 1)):
        xs, ys, ds = [], [], []
        for _ in range(n):
            x = sum(rand() for _ in range(4)) - 2.0
            p = min(max((0.5 + 0.6 * x) if z else 0.05, 0.0), 0.95)
            d = 1.0 if rand() < p else 0.0
            y = 1.0 + 0.05 * x + 5.0 * d + 0.5 * (sum(rand() for _ in range(4)) - 2.0)
            xs.append(x)
            ys.append(y)
            ds.append(d)
        out.append(_row(gid, xs, ys, ds))
    return out


def test_backfire_annotation_fires_when_cuped_inflates_the_variance():
    """An annotation, never an automatic fallback: switching estimators on
    the observed variance is data-dependent selection and would void the
    reported interval."""
    rows = _backfire_rows()
    c, t = _arms(rows)
    _, se, se_raw = _late_additive_cuped(t, c)
    assert se > se_raw, "backfire fixture must actually inflate the SE"

    (row,) = _late(
        estimate_encouragement(
            [METRIC], rows, _design(), estimands=("late",), methods=CUPED
        ).results
    )
    assert row.note is not None
    assert "INFLATED" in row.note
    # Still emitted, still labelled cuped, still the adjusted number.
    assert row.method == "cuped"
    assert row.require_lift().lb is not None and row.require_lift().ub is not None

    # The annotation marks the wider interval: the same fixture without the
    # adjustment is the tighter interval the note points at.
    (unadjusted,) = _late(
        estimate_encouragement([METRIC], rows, _design(), estimands=("late",)).results
    )
    adjusted_lift = row.require_lift()
    plain_lift = unadjusted.require_lift()
    assert adjusted_lift.lb is not None and adjusted_lift.ub is not None
    assert plain_lift.lb is not None and plain_lift.ub is not None
    assert adjusted_lift.ub - adjusted_lift.lb > plain_lift.ub - plain_lift.lb


def test_no_backfire_annotation_when_cuped_reduces_the_variance():
    rows = _rows(theta_d=0.05)
    c, t = _arms(rows)
    _, se, se_raw = _late_additive_cuped(t, c)
    assert se < se_raw
    (row,) = _late(
        estimate_encouragement(
            [METRIC], rows, _design(), estimands=("late",), methods=CUPED
        ).results
    )
    assert "INFLATED" not in (row.note or "")


# Substrate parity (design validation test 8)


def test_cxd_parity_between_frame_and_warehouse_substrates():
    """The moment schema is a cross-substrate contract: ``from_unit_summary``
    and ``group_summary`` must agree on cxd, and therefore on the whole
    cuped-LATE row, for identical per-unit data."""
    import ibis
    import pyarrow as pa

    from increment.frame import MetricSpec, from_unit_summary
    from increment.query.builders import group_summary

    xc, yc, dc = _units(120, seed=4242, z=0)
    xt, yt, dt = _units(120, seed=8484, theta_d=0.08, z=1)
    units = {
        "unit_id": [f"u{i}" for i in range(240)],
        "experiment_id": ["s"] * 240,
        "group_id": ["control"] * 120 + ["treat"] * 120,
        "metric": ["rev"] * 240,
        "y": yc + yt,
        "x": xc + xt,
        "y_den": [None] * 240,
        "d": dc + dt,
    }

    frame_rows = from_unit_summary(
        pa.table(
            {
                "unit_id": units["unit_id"],
                "variant": units["group_id"],
                "rev": units["y"],
                "pre_rev": units["x"],
                "clicked": units["d"],
            }
        ),
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="rev", covariate="pre_rev")],
        uptake="clicked",
    ).raw_moments

    con = ibis.duckdb.connect()
    totals = con.create_table("totals", pa.table(units, schema=_TOTALS_SCHEMA()))
    wh_rows = con.to_pyarrow(group_summary(totals)).to_pylist()

    frame_by_gid = {r["group_id"]: r for r in frame_rows}
    wh_by_gid = {r["group_id"]: r for r in wh_rows}
    for gid in ("control", "treat"):
        assert frame_by_gid[gid]["cxd"] == pytest.approx(wh_by_gid[gid]["cxd"], rel=1e-12)

    def _late_row(rows):
        (row,) = _late(
            estimate_encouragement(
                [METRIC], rows, _design(), estimands=("late",), methods=CUPED
            ).results
        )
        return row.require_lift().value, row.require_lift().lb, row.require_lift().ub

    assert _late_row(frame_rows) == pytest.approx(_late_row(wh_rows), rel=1e-10)


def test_group_summary_leaves_cxd_null_without_a_covariate():
    """Unpopulated slots stay literally None - a 0.0 would slip past every
    ``is None`` guard and die much later as a zero-variance covariate."""
    import ibis
    import pyarrow as pa

    con = ibis.duckdb.connect()
    from increment.query.builders import group_summary

    totals = con.create_table(
        "totals_no_x",
        pa.table(
            {
                "unit_id": ["u1", "u2"],
                "experiment_id": ["s", "s"],
                "group_id": ["control", "treat"],
                "metric": ["rev", "rev"],
                "y": [1.0, 2.0],
                "x": [None, None],
                "y_den": [None, None],
                "d": [0.0, 1.0],
            },
            schema=_TOTALS_SCHEMA(),
        ),
    )
    rows = con.to_pyarrow(group_summary(totals)).to_pylist()
    assert all(r["cxd"] is None for r in rows)


def test_frame_leaves_cxd_null_without_an_uptake_column():
    import pyarrow as pa

    from increment.frame import MetricSpec, from_unit_summary

    src = from_unit_summary(
        pa.table(
            {
                "unit_id": ["u1", "u2", "u3", "u4"],
                "variant": ["control", "control", "treat", "treat"],
                "rev": [1.0, 2.0, 3.0, 4.0],
                "pre_rev": [0.5, 1.5, 2.5, 3.5],
            }
        ),
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="rev", covariate="pre_rev")],
    )
    assert all(r["cxd"] is None for r in src.raw_moments)
