"""Clustered unadjusted estimation from canonical unit-frame metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from increment.estimation.engine import Method
    from increment.estimation.results import LiftEstimate
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


@dataclass(frozen=True, slots=True)
class _ExactClusterArm:
    point: Fraction
    clusters: tuple[tuple[Any, Fraction, Fraction], ...]


def _exact_cluster_arm(
    values: tuple[float, ...], denominators: tuple[float, ...], cluster_ids: tuple[Any, ...]
) -> _ExactClusterArm:
    """Retain exact cluster masses and scores without a second member-level array."""
    origin = Fraction(values[0])
    zero = Fraction()
    empty = (zero, zero)
    totals: dict[Any, tuple[Fraction, Fraction]] = {}
    for key, value, denominator in zip(cluster_ids, values, denominators, strict=True):
        mass = Fraction(denominator)
        d, h = totals.get(key, empty)
        totals[key] = d + mass, h + Fraction(value) - origin * mass
    total_d = sum((d for d, _ in totals.values()), zero)
    total_h = sum((h for _, h in totals.values()), zero)
    squared_d = total_d * total_d
    return _ExactClusterArm(
        point=origin + total_h / total_d,
        clusters=tuple(
            (key, d / total_d, (total_d * h - total_h * d) / squared_d)
            for key, (d, h) in totals.items()
        ),
    )


def _exact_response_weights(
    a: tuple[Fraction, ...], b: tuple[Fraction, ...], present: tuple[bool, ...]
) -> tuple[Fraction, ...] | None:
    """Invert the same response equations before any numerical rounding."""
    active = [i for i, keep in enumerate(present) if keep]
    if not active:
        return (Fraction(),) * len(a)
    if a == b and len(active) == 2 and a[active[0]] != a[active[1]]:
        return None
    diagonal = tuple(1 - x - y for x, y in zip(a, b, strict=True))
    pivot = min((diagonal[i] for i in active), key=abs)
    scaled = tuple(
        (Fraction(1) if diagonal[i] == pivot else pivot / diagonal[i]) if present[i] else Fraction()
        for i in range(len(a))
    )
    denominator = pivot + sum((q * x * y for q, x, y in zip(scaled, a, b, strict=True)), Fraction())
    return tuple(q / denominator for q in scaled) if denominator else None


@dataclass(frozen=True, slots=True)
class _UnadjustedClusterArm:
    """Retained member identities and centered arm-mean contributions."""

    unit_ids: tuple[Any, ...]
    cluster_ids: tuple[Any, ...]
    point: float
    scores: tuple[float, ...]
    masses: tuple[float, ...]
    exact: _ExactClusterArm


def _unadjusted_response_weights(
    a: tuple[float, ...], b: tuple[float, ...], present: tuple[bool, ...]
) -> tuple[float, ...] | None:
    """Invert the diagonal-plus-rank-one response of cluster residual products.

    With R_a = I - w_a 1', unbiased covariance requires
    q_g (1 - w_ag - w_bg) + sum_h q_h w_ah w_bh = 1.
    Only clusters present in both arms have an unknown covariance component.
    """
    active = [i for i, keep in enumerate(present) if keep]
    if not active:
        return (0.0,) * len(a)
    # With two unequal masses the marginal variance is not identifiable.
    if a == b and len(active) == 2 and a[active[0]] != a[active[1]]:
        return None
    diagonal = tuple(math.fsum((1.0, -x, -y)) for x, y in zip(a, b, strict=True))
    pivot = min((diagonal[i] for i in active), key=abs)
    # Scaling by the smallest diagonal also handles an exactly zero leverage gap.
    scaled = tuple(
        (1.0 if diagonal[i] == pivot else pivot / diagonal[i]) if present[i] else 0.0
        for i in range(len(a))
    )
    denominator = math.fsum((pivot, *(q * x * y for q, x, y in zip(scaled, a, b, strict=True))))
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    weights = tuple(q / denominator for q in scaled)
    return weights if all(math.isfinite(q) for q in weights) else None


def _unadjusted_response_se(
    t: tuple[float, ...],
    c: tuple[float, ...],
    weights: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]],
) -> float | None:
    """Contract the corrected joint covariance without squaring raw outcomes."""
    score_scale = max((*map(abs, t), *map(abs, c)))
    if score_scale == 0.0:
        return 0.0
    weight_scale = max(abs(q) for component in weights for q in component)
    tt, cc, tc = weights
    terms = []
    for x, y, qt, qc, cross in zip(t, c, tt, cc, tc, strict=True):
        x, y = x / score_scale, y / score_scale
        qt, qc, cross = qt / weight_scale, qc / weight_scale, cross / weight_scale
        # Keep exact paired cancellation before squaring; retain signed corrections.
        terms.extend((cross * (x - y) ** 2, (qt - cross) * x * x, (qc - cross) * y * y))
    variance = math.fsum(terms)
    if variance < 0.0:
        return None
    return score_scale * (math.sqrt(variance) * math.sqrt(weight_scale))


@dataclass(frozen=True, slots=True)
class _UnadjustedClusterState:
    """Both arms indexed by the union of their original cluster identities."""

    treatment: _UnadjustedClusterArm
    control: _UnadjustedClusterArm
    cluster_ids: tuple[Any, ...]
    totals_t: tuple[float, ...]
    totals_c: tuple[float, ...]
    response_weights: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None

    def uncertainty(self, *, absolute: bool) -> tuple[float | None, float | None]:
        from increment.estimation.armstats import welch_satterthwaite_df

        scale_t = 1.0 if absolute else self.treatment.point
        scale_c = 1.0 if absolute else self.control.point
        t = tuple(v / scale_t for v in self.totals_t)
        c = tuple(v / scale_c for v in self.totals_c)
        ids_t, ids_c = set(self.treatment.cluster_ids), set(self.control.cluster_ids)
        if ids_t.isdisjoint(ids_c):
            se_t = math.hypot(*t) * math.sqrt(len(ids_t) / (len(ids_t) - 1))
            se_c = math.hypot(*c) * math.sqrt(len(ids_c) / (len(ids_c) - 1))
            se = math.hypot(se_t, se_c)
            df = welch_satterthwaite_df(
                (se_t / se) ** 2 if se else 0.0,
                float(len(ids_t) - 1),
                (se_c / se) ** 2 if se else 0.0,
                float(len(ids_c) - 1),
            )
            return se, df
        if self.response_weights is not None:
            return _unadjusted_response_se(t, c, self.response_weights), None
        # Contract before squaring: the signed cross-arm term must survive.
        residuals = tuple(math.fsum((a, -b)) for a, b in zip(t, c, strict=True))
        k = len(self.cluster_ids)
        return math.hypot(*residuals) * math.sqrt(k / (k - 1)), None


def _unadjusted_centered_average(values: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """Center before summing; neither raw totals nor squared outcomes are needed."""
    origin = min(values) / 2.0 + max(values) / 2.0
    offsets = tuple(value - origin for value in values)
    average = math.fsum(value / len(values) for value in offsets)
    return math.fsum((origin, average)), tuple(value - average for value in offsets)


def _unadjusted_cluster_arm(
    rows: Sequence[Mapping[str, Any]],
    metric: Metric,
    group: str,
) -> _UnadjustedClusterArm:
    from increment.estimation.variance import _raise as variance_refuse

    y = tuple(float(row["y"]) for row in rows)
    den = tuple(float(row["y_den"]) if metric.type == "ratio" else 1.0 for row in rows)
    mean_y, residual_y = _unadjusted_centered_average(y)
    mean_den, residual_den = _unadjusted_centered_average(den)
    if mean_den <= 0.0:
        variance_refuse(
            "estimation.variance.ratio_moments_nonpositive_denominator_mean",
            metric=metric.name,
            group_id=group,
            d_bar=mean_den,
        )
    n = len(rows)
    point = mean_y / mean_den
    cluster_ids = tuple(row["cluster_id"] for row in rows)
    return _UnadjustedClusterArm(
        unit_ids=tuple(row["unit_id"] for row in rows),
        cluster_ids=cluster_ids,
        point=point,
        scores=tuple(
            math.fsum((dy / n / mean_den, -(dd / n / mean_den) * point))
            for dy, dd in zip(residual_y, residual_den, strict=True)
        ),
        masses=tuple(value / n / mean_den for value in den),
        exact=_exact_cluster_arm(y, den, cluster_ids),
    )


def _unadjusted_cluster_state(
    treatment: _UnadjustedClusterArm,
    control: _UnadjustedClusterArm,
) -> _UnadjustedClusterState:
    cluster_ids = tuple(dict.fromkeys((*treatment.cluster_ids, *control.cluster_ids)))

    def totals(arm: _UnadjustedClusterArm, values: tuple[float, ...]) -> tuple[float, ...]:
        members: dict[Any, list[float]] = {key: [] for key in cluster_ids}
        for key, score in zip(arm.cluster_ids, values, strict=True):
            members[key].append(score)
        return tuple(math.fsum(members[key]) for key in cluster_ids)

    mass_t, mass_c = totals(treatment, treatment.masses), totals(control, control.masses)
    ids_t, ids_c = set(treatment.cluster_ids), set(control.cluster_ids)
    tt = _unadjusted_response_weights(mass_t, mass_t, tuple(g in ids_t for g in cluster_ids))
    cc = _unadjusted_response_weights(mass_c, mass_c, tuple(g in ids_c for g in cluster_ids))
    tc = _unadjusted_response_weights(
        mass_t, mass_c, tuple(g in ids_t and g in ids_c for g in cluster_ids)
    )
    return _UnadjustedClusterState(
        treatment,
        control,
        cluster_ids,
        totals(treatment, treatment.scores),
        totals(control, control.scores),
        (tt, cc, tc) if tt is not None and cc is not None and tc is not None else None,
    )


def _unadjusted_joint_result(
    state: _UnadjustedClusterState,
    metric: Metric,
    treatment: str,
    *,
    alpha: float,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
) -> LiftEstimate:
    from increment.estimation.inference import _joint_additive_bounds
    from increment.estimation.results import (
        Estimate,
        LiftEstimate,
        _alpha_eff_for,
        _joint_reference_from_exact,
        relative_confidence_set,
    )

    abs_se, abs_df = state.uncertainty(absolute=True)
    ids_t, ids_c = set(state.treatment.cluster_ids), set(state.control.cluster_ids)
    exact_t, exact_c = state.treatment.exact, state.control.exact
    t_by_cluster = {key: (mass, score) for key, mass, score in exact_t.clusters}
    c_by_cluster = {key: (mass, score) for key, mass, score in exact_c.clusters}
    empty = (Fraction(), Fraction())
    t = tuple(t_by_cluster.get(key, empty)[1] for key in state.cluster_ids)
    c = tuple(c_by_cluster.get(key, empty)[1] for key in state.cluster_ids)
    exact_weights = None
    if not ids_t.isdisjoint(ids_c):
        mt = tuple(t_by_cluster.get(key, empty)[0] for key in state.cluster_ids)
        mc = tuple(c_by_cluster.get(key, empty)[0] for key in state.cluster_ids)
        tt = _exact_response_weights(mt, mt, tuple(key in ids_t for key in state.cluster_ids))
        cc = _exact_response_weights(mc, mc, tuple(key in ids_c for key in state.cluster_ids))
        tc = _exact_response_weights(
            mt, mc, tuple(key in ids_t and key in ids_c for key in state.cluster_ids)
        )
        if tt is not None and cc is not None and tc is not None:
            exact_weights = tt, cc, tc
    if ids_t.isdisjoint(ids_c):
        kt, kc = len(ids_t), len(ids_c)
        vt = Fraction(kt, kt - 1) * sum((value * value for value in t), Fraction())
        vc = Fraction(kc, kc - 1) * sum((value * value for value in c), Fraction())
        var_a, var_c, cov_ac = vt + vc, vc, -vc
        kind, df = "t", float(min(kt - 1, kc - 1))
    elif exact_weights is not None:
        qtt, qcc, qtc = exact_weights
        var_a = var_c = cov_ac = Fraction()
        for tg, cg, qt, qc, cross in zip(t, c, qtt, qcc, qtc, strict=True):
            dg = tg - cg
            var_a += cross * dg * dg + (qt - cross) * tg * tg + (qc - cross) * cg * cg
            var_c += qc * cg * cg
            cov_ac += cross * dg * cg + (cross - qc) * cg * cg
        kind, df = "normal", None
    else:
        residuals = tuple(tg - cg for tg, cg in zip(t, c, strict=True))
        factor = Fraction(len(state.cluster_ids), len(state.cluster_ids) - 1)
        var_a = factor * sum((value * value for value in residuals), Fraction())
        var_c = factor * sum((value * value for value in c), Fraction())
        cov_ac = factor * sum((x * y for x, y in zip(residuals, c, strict=True)), Fraction())
        kind, df = "normal", None

    reference, unavailable_reason = _joint_reference_from_exact(
        a=exact_t.point - exact_c.point,
        c=exact_c.point,
        var_a=var_a,
        var_c=var_c,
        cov_ac=cov_ac,
        kind=kind,
        df=df,
    )
    relative_set = (
        relative_confidence_set(reference, alpha=alpha, alternative=alternative)
        if reference is not None
        else None
    )
    relative_lift = relative_set.estimate() if relative_set is not None else None
    if relative_lift is None and unavailable_reason is not None:
        try:
            point = float((exact_t.point - exact_c.point) / exact_c.point)
        except (OverflowError, ZeroDivisionError):
            point = None
        if (
            point is not None
            and math.isfinite(state.control.point)
            and state.control.point != 0.0
            and math.isfinite(point)
        ):
            relative_lift = Estimate(value=point)
    # Available joint evidence owns both serialized projections, including rounding.
    abs_diff = reference.a if reference is not None else float(exact_t.point - exact_c.point)
    if reference is not None:
        abs_se = math.sqrt(reference.var_a)
    if abs_se == 0.0:
        abs_se = None
    abs_lower, abs_upper = _joint_additive_bounds(abs_diff, abs_se, alpha, alternative, abs_df)
    return LiftEstimate(
        metric=metric.name,
        group_id=treatment,
        method="unadjusted",
        method_role="decision",
        inference="fixed",
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        preferred_direction=metric.declared_preferred_direction,
        lift=relative_lift,
        relative_confidence_set=relative_set,
        relative_unavailable_reason=unavailable_reason,
        scale="linear",
        abs_diff=abs_diff,
        abs_se=abs_se,
        abs_lb=abs_lower,
        abs_ub=abs_upper,
        abs_reference_kind=("t" if abs_df is not None else "normal")
        if abs_se is not None
        else None,
        abs_reference_df=abs_df if abs_se is not None else None,
        abs_alpha=_alpha_eff_for(alternative, alpha) if abs_lower is not None else None,
        n_clusters=len(state.cluster_ids),
        dof=df,
        reference_kind=kind,
        reference_df=df,
        note=(
            "Independent-arm covariance; additive Welch and relative fixed-t working approximations."
            if ids_t.isdisjoint(ids_c)
            else (
                "Joint distinct-cluster covariance with fitted-mean response correction; "
                "the Normal reference is asymptotic."
                if exact_weights is not None
                else "Joint distinct-cluster covariance with CR1 approximation: response "
                "correction is unavailable; the Normal reference is asymptotic."
            )
        ),
    )


def estimate_clustered_unadjusted(
    src: MomentSource,
    metric: Metric,
    control_group: str,
    method: Method,
    *,
    cluster: str,
    alpha: float,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
) -> list[LiftEstimate]:
    import narwhals as nw

    from increment.errors import refuse
    from increment.estimation.encouragement import ARM_NEEDS_TWO
    from increment.estimation.engine import _refuse as engine_refuse
    from increment.estimation.engine import (
        _validate_typed_direct_compatibility,
        check_total_clusters,
    )
    from increment.estimation.inference import (
        INFER_ATE_NULL_ABS_FINITE,
        INFER_ATE_NULL_LIFT_FINITE,
        _resolve_fixed_horizon,
    )

    _validate_typed_direct_compatibility(
        [metric],
        [method],
        cluster=cluster,
        inference=None,
        prior=None,
        alpha=alpha,
        alternative=cast("Literal['two-sided', 'greater', 'less']", alternative),
        null_lift=null_lift,
        null_abs=null_abs,
    )
    from increment.compatibility import Unsupported, refuse_unsupported

    if method.variance_reduction == "cuped":
        refuse_unsupported(Unsupported("arm.adjustment.cluster_cuped"), cluster=cluster)
    if metric.type not in ("mean", "conversion", "retention", "ratio"):
        refuse_unsupported(
            Unsupported("arm.metric.quantile_cluster"),
            metric=metric.name,
            metric_type=metric.type,
            cluster=cluster,
        )
    _resolve_fixed_horizon(
        alpha,
        alternative,
        dof=None,
        arm_ns=None,
        prior=None,
    )
    if not math.isfinite(null_lift):
        refuse(INFER_ATE_NULL_LIFT_FINITE, null_lift=null_lift)
    if null_abs is not None and not math.isfinite(null_abs):
        refuse(INFER_ATE_NULL_ABS_FINITE, null_abs=null_abs)
    frame = nw.from_native(src.unit_frame(metric), eager_only=True)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in frame.iter_rows(named=True):
        groups.setdefault(str(row["group_id"]), []).append(row)
    if control_group not in groups:
        engine_refuse(
            "estimation.engine.control_group_found",
            control_group=control_group,
            known_groups=sorted(groups),
        )
    results = []
    control_rows = groups[control_group]
    for treatment in sorted(groups.keys() - {control_group}):
        treatment_rows = groups[treatment]
        ids_t = {row["cluster_id"] for row in treatment_rows}
        ids_c = {row["cluster_id"] for row in control_rows}
        check_total_clusters(metric.name, cluster, len(ids_t | ids_c))
        if len(ids_t) < 2 or len(ids_c) < 2:
            refuse(
                ARM_NEEDS_TWO,
                metric=metric.name,
                cluster=cluster,
                k_t=len(ids_t),
                k_c=len(ids_c),
            )
        state = _unadjusted_cluster_state(
            _unadjusted_cluster_arm(treatment_rows, metric, treatment),
            _unadjusted_cluster_arm(control_rows, metric, control_group),
        )
        result = _unadjusted_joint_result(
            state,
            metric,
            treatment,
            alpha=alpha,
            alternative=alternative,
            null_lift=null_lift,
            null_abs=null_abs,
        )
        results.append(result)
    return results
