"""Actual public source/estimator adapters for each frozen acceptance entry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from increment.errors import CodedError
from increment.estimation.adjust import estimate_ate
from increment.estimation.encouragement import estimate_encouragement
from increment.estimation.engine import Method, _df_to_arms, estimate_lift
from increment.estimation.results import LiftEstimate
from increment.estimation.sitewide import SitewideContrast, sitewide_impact
from increment.frame import MetricSpec, from_unit_summary
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    UptakeSpec,
)
from tests.estimation._i13_calibration import IntervalResult
from tests.estimation._i13_manifest import AcceptanceCell, DGPSample


class OraclePropensity:
    def __init__(self, cell: AcceptanceCell):
        self.cell = cell

    def fit(self, X, d):
        pass

    def predict(self, X):
        if self.cell.design.dgp.heterogeneous:
            return np.asarray(X)[:, 0]
        dgp = self.cell.design.dgp
        return np.full(len(X), dgp.k_t / (dgp.k_t + dgp.k_c))


class BaselineOutcome:
    """Declared fixed control regression, not an oracle treated regression.

    With an oracle propensity AIPW still targets ATE. DML instead needs the
    full E[Y|X], which is supplied separately below.
    """

    def __init__(self, cell: AcceptanceCell):
        self.cell = cell

    def fit(self, X, d):
        pass

    def predict(self, X):
        dgp = self.cell.design.dgp
        x = np.asarray(X)[:, 0]
        if dgp.heterogeneous:
            return 5.0 + x * (x > 0.4) if self.cell.estimator == "dml" else np.full(len(x), 5.0)
        baseline = 5.0 + 0.3 * x
        if self.cell.estimator == "dml":
            baseline = baseline + float(dgp.effect or 0) * dgp.k_t / (dgp.k_t + dgp.k_c)
        return baseline


def truth_of(cell: AcceptanceCell, sample: DGPSample) -> float:
    theta = sample.theta_plr if cell.estimator == "dml" else sample.tau_u
    # DML's relative target uses its own observed-control mean denominator.
    # In the heterogeneous fixture Y(0)=5 plus independent zero-mean noise.
    return theta / sample.mean_c if cell.scale == "relative" else theta


def estimate_of(cell: AcceptanceCell, sample: DGPSample, _i: int) -> IntervalResult:
    rows: Sequence[LiftEstimate]
    if cell.estimator in ("iptw", "aipw", "dml"):
        design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("z",)))
        src = from_unit_summary(
            sample.table,
            unit="u",
            group="g",
            control="C",
            metrics={"y": "mean"},
            cluster="cluster",
            design=design,
        )
        metric = next(m for m in src.context.metrics if m.name == "y")
        value_scale: Literal["absolute", "relative"] = (
            "absolute" if cell.scale == "absolute" else "relative"
        )

        def propensity_factory() -> OraclePropensity:
            return OraclePropensity(cell)

        def outcome_factory() -> BaselineOutcome:
            return BaselineOutcome(cell)

        oracle = cell.nuisance == "oracle_propensity"
        crossfit = cell.estimator in ("aipw", "dml")
        method = Method(
            name=cell.estimator,
            propensity_learner=propensity_factory if oracle else None,
            outcome_learner=outcome_factory if (oracle and crossfit) else None,
            folds=cell.folds if crossfit else None,
        )
        rows = estimate_ate(
            src,
            design,
            methods=[method],
            metrics=[metric],
            value_scale={"y": value_scale},
        ).results
    elif cell.estimator == "late":
        design = Encouragement(
            control_group="C",
            uptake=UptakeSpec(fact="took"),
            one_sided=True,
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True,
                justification="DGP outcome depends only on uptake and potential outcomes",
            ),
        )
        src = from_unit_summary(
            sample.table,
            unit="u",
            group="g",
            control=design.control_group,
            metrics={"y": "mean"},
            cluster="cluster",
            design=design,
            uptake="took",
        )
        metric = next(m for m in src.context.metrics if m.name == "y")
        computation = estimate_encouragement(
            [metric],
            list(src.moments(metric)),
            design,
            estimands=("late",),
            cluster="cluster",
        )
        rows = [row for row in computation.results if row.value_scale == "absolute"]
        if not rows:
            reasons = "; ".join(f"{f.code}: {f.display()}" for f in computation.failures.values())
            return IntervalResult(
                None, None, None, unavailable_reason=reasons or "late_absolute_unavailable"
            )
    else:
        metrics = [
            MetricSpec(name="y", type="ratio", numerator="num", denominator="den")
            if cell.estimator == "ratio"
            else MetricSpec(name="y")
        ]
        src = from_unit_summary(
            sample.table,
            unit="u",
            group="g",
            control="C",
            metrics=metrics,
            cluster="cluster",
        )
        metric = next(m for m in src.context.metrics if m.name == "y")
        summary = list(src.moments(metric))
        if cell.estimator == "sitewide":
            arms = _df_to_arms(summary)
            # Sitewide's sum contrast consumes cluster size in both x and den.
            arms = [
                a.model_copy(
                    update={
                        "x_role": "cluster_size",
                        "ref_x": a.ref_den,
                        "cx1": a.cden1,
                        "cx2": a.cden2,
                        "cxy": a.cyden,
                    }
                )
                for a in arms
            ]
            control = next(a for a in arms if a.group_id == "C")
            target = next(a for a in arms if a.group_id == "T")
            result = sitewide_impact(
                SitewideContrast.from_clusters(control, target, cluster="cluster"),
                site_total_volume=float(np.asarray(sample.table["y"]).sum()),
            )
            n = len(sample.table)
            # Scale all endpoints by realized N; coverage of N*tau is unchanged.
            return IntervalResult(
                result.absolute_impact / n,
                result.absolute_impact_lb / n,
                result.absolute_impact_ub / n,
            )
        computation = estimate_lift([metric], summary, control_group="C", cluster="cluster")
        rows = computation.results
        if not rows:
            reasons = "; ".join(f"{f.code}: {f.display()}" for f in computation.failures.values())
            return IntervalResult(
                None, None, None, unavailable_reason=reasons or "lift_unavailable"
            )
    if not rows:
        return IntervalResult(None, None, None)
    (row,) = rows
    if cell.scale == "absolute_sidecar":
        return IntervalResult(row.abs_diff, row.abs_lb, row.abs_ub)
    relative = row.relative_confidence_set
    decision_reason = row.relative_unavailable_reason
    try:
        p_value = row.p_value()
    except CodedError as exc:
        p_value, decision_reason = None, exc.code
    return IntervalResult(
        None if row.lift is None else row.lift.value,
        None if row.lift is None else row.lift.lb,
        None if row.lift is None else row.lift.ub,
        None if row.lift is None else row.lift.open_side,
        p_value=p_value,
        confidence_set=relative,
        unavailable_reason=(
            row.relative_unavailable_reason
            or (
                relative.reason or relative.point_unavailable_reason
                if relative is not None
                else None
            )
        ),
        decision_unavailable_reason=decision_reason,
    )
