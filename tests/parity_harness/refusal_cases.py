from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class Unreachable:
    reason: str


def _compatibility_scenario(code: str) -> Callable[[], object]:
    """Issue the real public planning/readout request that owns ``code``."""
    if code == "compatibility.support_unsupported":
        from increment.compatibility import Unsupported, refuse_unsupported

        return lambda: refuse_unsupported(cast(Unsupported, "scenario"))
    if code == "allocation.alpha.underflow":
        from increment import readouts
        from tests.test_allocation_endpoints import _two_metric_randomized_source

        return lambda: readouts.run(_two_metric_randomized_source(alpha=5e-324))
    if code.startswith("arm."):
        from increment.estimation.arm_contract import ArmPlanningProcedure
        from increment.power.core import Baseline, PowerDesign, required_sample_size
        from increment.semantics.models import InferenceSpec, MethodSpec

        baseline = Baseline(mean=20.0, var=400.0)
        inference = None
        metric_type = "mean"
        clustered = False
        update: dict[str, object] = {}
        if code == "arm.baseline.compliance_undeclared":
            baseline = Baseline(mean=20.0, var=400.0, compliance=0.6)
            update["analysis"] = {
                **ArmPlanningProcedure.standard().analysis.model_dump(),
                "identification": "observational",
            }
        elif code == "arm.baseline.absorption_undeclared":
            baseline = Baseline(mean=20.0, var=400.0, icc=0.1)
        elif code == "arm.baseline.triggering_undeclared":
            baseline = Baseline(mean=20.0, var=400.0, trigger_rate=0.8)
        elif code == "arm.baseline.cuped_undeclared":
            baseline = Baseline(mean=20.0, var=400.0, cuped_rho=0.4)
        elif code == "arm.baseline.iid_cluster_knobs":
            baseline = Baseline(mean=20.0, var=400.0, avg_cluster_size=5.0)
        elif code == "arm.baseline.cluster_size_required":
            clustered = True
        elif code in {
            "arm.inference.cluster",
            "arm.adjustment.sequential_cuped",
            "arm.adjustment.sequential_prior",
            "arm.metric.quantile_sequential",
        }:
            inference = InferenceSpec(kind="asymptotic_mean")
        if code in {
            "arm.adjustment.cluster_cuped",
            "arm.adjustment.sequential_cuped",
            "arm.metric.quantile_cuped",
        }:
            update["decision_method"] = MethodSpec(name="cuped", variance_reduction="cuped")
        if code in {"arm.adjustment.cluster_prior", "arm.adjustment.sequential_prior"}:
            update["prior_present"] = True
        if code in {
            "arm.metric.quantile_cluster",
            "arm.metric.quantile_cuped",
            "arm.metric.quantile_sequential",
        }:
            metric_type = "quantile"
        if code in {
            "arm.adjustment.cluster_cuped",
            "arm.adjustment.cluster_prior",
            "arm.inference.cluster",
            "arm.metric.quantile_cluster",
        }:
            clustered = True
        if clustered and code != "arm.baseline.cluster_size_required":
            baseline = Baseline(mean=20.0, var=400.0, avg_cluster_size=5.0)
        if code == "arm.metric.unsupported":
            update["decision"] = object()
        procedure = ArmPlanningProcedure.standard(
            metric_type=metric_type,
            inference=inference,
            clustered=clustered,
        ).model_copy(update=update)
        return lambda: required_sample_size(
            0.05, baseline, procedure, PowerDesign(power=0.8), planned_looks=2
        )

    if code.startswith("contrast."):
        from increment.decision import FixedInference, NoFamily
        from increment.estimation.contrast import estimate_contrast
        from tests.estimation.test_contrast import _procedure, _stats

        procedure = _procedure()
        stats = _stats()
        if code == "contrast.inference":
            procedure = procedure.model_construct(
                **{
                    **procedure.model_dump(),
                    "family": NoFamily(),
                    "inference": FixedInference.model_construct(kind="always_valid"),
                }
            )
        elif code == "contrast.decision":
            procedure = procedure.model_construct(
                **{
                    **procedure.model_dump(),
                    "family": NoFamily.model_construct(kind="bonferroni"),
                }
            )
        else:
            stats = stats.model_construct(**{**stats.model_dump(), "aggregation": "ratio"})
        return lambda: estimate_contrast(stats, procedure).results[0]

    from increment.estimation.contrast import estimate_contrast
    from tests.estimation.test_contrast import _procedure, _stats

    return lambda: estimate_contrast(_stats(), _procedure())


def _engine_scenario(code: str) -> Callable[[], object]:  # noqa: C901, PLR0915
    from increment.estimation import Method, estimate_lift
    from increment.estimation.armstats import ArmStats
    from increment.semantics.models import MeanMetric

    def make_arm(*, n: int, mean: float, var: float, group_id: str, metric: str) -> ArmStats:
        sum_y = n * mean
        sum_y2 = (n - 1) * var + sum_y * sum_y / n
        return ArmStats.from_raw_sums(
            study_id="e",
            metric=metric,
            group_id=group_id,
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
        )

    def as_row(arm: ArmStats) -> dict[str, object]:
        row = arm.model_dump(mode="json")
        row["experiment_id"] = row.pop("study_id")
        return row

    control = make_arm(n=10, mean=1.0, var=1.0, group_id="control", metric="rev")
    treatment = make_arm(n=10, mean=2.0, var=1.0, group_id="treatment", metric="rev")
    summary = [as_row(control), as_row(treatment)]
    metric = MeanMetric(name="rev", entity="user", fact="rev")

    def run() -> object:  # noqa: C901, PLR0915
        if code == "estimation.engine.ratio.cuped":
            import ibis

            from tests.test_analysis_asof import _analysis_with_daily_ratio_metric

            con = ibis.duckdb.connect()
            analysis = None
            try:
                analysis = _analysis_with_daily_ratio_metric(con)
                return analysis.run_asof_lift(
                    decision_method=Method(name="cuped", variance_reduction="cuped")
                )
            finally:
                if analysis is not None:
                    analysis.close()
                con.disconnect()
        if code in {
            "estimation.engine.cuped_linear_cluster",
            "estimation.engine.cuped_linear_sequential",
        }:
            import datetime as dt

            import pyarrow as pa

            from increment import builders
            from increment.semantics.design import Randomized
            from increment.semantics.models import (
                AnalysisPlan,
                CupedCovariate,
                InferenceSpec,
            )
            from tests.parity_harness.views import read

            rows = [
                {
                    "unit": f"u{i}",
                    "group": "control" if i < 8 else "treatment",
                    "revenue": float(i + 1),
                    "pre": "a" if i % 2 == 0 else "b",
                    "cluster": f"c{i // 2}",
                    "exposure_date": dt.date(2024, 1, 1),
                }
                for i in range(16)
            ]
            frame = pa.Table.from_pylist(rows)
            if code == "estimation.engine.cuped_linear_cluster":
                analysis = builders.from_frame(
                    frame,
                    shape="summary",
                    unit="unit",
                    group="group",
                    control="control",
                    metrics={"revenue": "mean"},
                    cluster="cluster",
                )
            else:
                analysis = builders.from_frame(
                    frame,
                    shape="summary",
                    unit="unit",
                    group="group",
                    metrics={"revenue": "mean"},
                    design=Randomized(
                        control_group="control",
                        allocation={"control": 0.5, "treatment": 0.5},
                    ),
                    exposure_date="exposure_date",
                    plan=AnalysisPlan(inference=InferenceSpec(kind="asymptotic_mean")),
                )
            try:
                return read(
                    analysis,
                    decision_method=Method.cuped_multicovariate(
                        name="linear",
                        covariates=CupedCovariate(name="pre", kind="categorical"),
                    ),
                )
            finally:
                analysis[1].close()
        if code == "estimation.engine.ratio.negative_variance":
            from increment.estimation.engine import ratio_abs_diff_se

            return ratio_abs_diff_se(1.0, 1.0, 1.0, 1.0, 1000.0, 100)
        if code == "estimation.engine.control_group_found":
            return estimate_lift([metric], summary, control_group="missing")
        if code == "estimation.engine.method_names_unique":
            return estimate_lift(
                [metric],
                summary,
                control_group="control",
                methods=[Method(name="same"), Method(name="same")],
            )
        if code == "estimation.engine.method_name_observational":
            return estimate_lift(
                [metric], summary, control_group="control", methods=[Method(name="iptw")]
            )
        if code == "estimation.engine.arm.duplicate_rows":
            return estimate_lift(
                [metric],
                [as_row(control), as_row(control), as_row(treatment)],
                control_group="control",
            )
        if code == "estimation.engine.group_summary_row":
            return estimate_lift(
                [metric],
                [{"experiment_id": "e", "metric": "rev", "group_id": "control"}],
                control_group="control",
            )
        if code == "estimation.engine.group_summary_missing":
            import pandas as pd

            return estimate_lift(
                [metric], pd.DataFrame(columns=["metric"]), control_group="control"
            )
        if code == "estimation.engine.arm.invalid_count":
            row = {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "control",
                "n": 2.9,
                "ref_y": 1.0,
                "cy1": 0.0,
                "cy2": 1.0,
            }
            return estimate_lift([metric], [row], control_group="control")
        if code == "estimation.engine.carries_format_moments":
            row = {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "control",
                "n": 5,
                "sum_y": 2.0,
                "sum_y2": 1.5,
            }
            return estimate_lift([metric], [row], control_group="control")
        if code == "estimation.engine.carries_unsupported_weighted":
            row = {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "control",
                "n": 5,
                "ref_y": 1.0,
                "cy1": 0.0,
                "cy2": 1.0,
                "sum_w": 4.0,
            }
            return estimate_lift([metric], [row], control_group="control")
        if code == "estimation.engine.metric_found_group":
            other = as_row(make_arm(n=10, mean=1, var=1, group_id="control", metric="other"))
            return estimate_lift([metric], [*summary, other], control_group="control")
        if code == "estimation.engine.x_role_declared":
            row = {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "control",
                "n": 100,
                "ref_y": 1.0,
                "cy1": 0.0,
                "cy2": 19.0,
                "ref_x": 5.0,
                "cx1": 0.0,
                "cx2": 7.0,
                "cxy": 2.0,
                "x_role": None,
            }
            return estimate_lift([metric], [row], control_group="control")
        if code == "estimation.engine.cuped_linear_unit_evidence":
            from increment.frame import CupedCovariate

            return estimate_lift(
                [MeanMetric(name="revenue", entity="user", fact="revenue")],
                [],
                "control",
                methods=[
                    Method.cuped_multicovariate(
                        name="linear", covariates=CupedCovariate(name="prior", kind="continuous")
                    )
                ],
            )
        if code == "estimation.engine.cluster.denominator":
            cluster_metric = MeanMetric(
                name="clustered", entity="user", fact="revenue", aggregation="sum"
            )
            cluster_rows = []
            for group_id, mean in (("C", 5.0), ("T", 5.5)):
                sum_y = 50 * mean
                arm = ArmStats.from_raw_sums(
                    study_id="e",
                    metric="clustered",
                    group_id=group_id,
                    n=50,
                    sum_y=sum_y,
                    sum_y2=49 + sum_y * sum_y / 50,
                    sum_den=50.0,
                    sum_den2=50.0,
                    sum_yden=sum_y,
                )
                cluster_rows.append(as_row(arm))
            return estimate_lift([cluster_metric], cluster_rows, control_group="C")
        if code in {
            "estimation.engine.cuped_linear_winsorization",
            "estimation.engine.cuped_linear_observational",
            "estimation.engine.cuped_linear_encouragement",
        }:
            import pyarrow as pa

            from increment import MetricSpec, UptakeSpec, builders
            from increment.frame import CupedCovariate
            from increment.semantics.design import (
                AdjustmentSet,
                Encouragement,
                ExclusionRestriction,
                Observational,
            )
            from tests.parity_harness.views import read

            method = Method.cuped_multicovariate(
                name="linear", covariates=CupedCovariate(name="pre", kind="categorical")
            )
            metric_spec = MetricSpec(
                name="revenue",
                decision_method=method,
                winsorization=(
                    {"upper_percentile": 0.9}
                    if code == "estimation.engine.cuped_linear_winsorization"
                    else None
                ),
            )
            design = None
            uptake = None
            if code == "estimation.engine.cuped_linear_observational":
                design = Observational(
                    control_group="control", adjustment=AdjustmentSet(covariates=("pre",))
                )
            elif code == "estimation.engine.cuped_linear_encouragement":
                design = Encouragement(
                    control_group="control",
                    uptake=UptakeSpec(fact="uptake"),
                    exclusion_restriction=ExclusionRestriction(
                        justification="Test-only design with the outcome affected through uptake."
                    ),
                )
                uptake = "uptake"
            rows = [
                {
                    "unit": f"u{i}",
                    "group": "control" if i < 8 else "treatment",
                    "revenue": float(i + 1),
                    "pre": "a" if i % 2 == 0 else "b",
                    "uptake": i % 2,
                }
                for i in range(16)
            ]
            analysis = builders.from_frame(
                pa.Table.from_pylist(rows),
                shape="summary",
                unit="unit",
                group="group",
                control=None if design is not None else "control",
                design=design,
                metrics=[metric_spec],
                uptake=uptake,
            )
            try:
                return read(analysis)
            finally:
                analysis[1].close()
        if code in {
            "estimation.engine.winsorization_metadata_present",
            "estimation.engine.winsorization_metadata_differs",
        }:

            def decorated(arm, *, upper_bound: float = 3.0):
                payload = arm.model_dump()
                payload.update(
                    winsor_upper_bound=upper_bound,
                    winsor_n=10,
                    winsor_n_lower=1,
                    winsor_n_upper=1,
                )
                return ArmStats.model_validate(payload)

            left = decorated(control)
            right = (
                treatment if code.endswith("_present") else decorated(treatment, upper_bound=4.0)
            )

            def row(arm):
                payload = arm.model_dump()
                payload["experiment_id"] = payload.pop("study_id")
                return payload

            return estimate_lift([metric], [row(left), row(right)], control_group="control")
        return estimate_lift([metric], summary, control_group="control")

    return run


_ENGINE_CODES = (
    "estimation.engine.arm.duplicate_rows",
    "estimation.engine.arm.invalid_count",
    "estimation.engine.carries_format_moments",
    "estimation.engine.carries_unsupported_weighted",
    "estimation.engine.cluster.denominator",
    "estimation.engine.control_group_found",
    "estimation.engine.cuped_linear_cluster",
    "estimation.engine.cuped_linear_encouragement",
    "estimation.engine.cuped_linear_observational",
    "estimation.engine.cuped_linear_sequential",
    "estimation.engine.cuped_linear_unit_evidence",
    "estimation.engine.cuped_linear_winsorization",
    "estimation.engine.group_summary_missing",
    "estimation.engine.group_summary_row",
    "estimation.engine.method_name_observational",
    "estimation.engine.method_names_unique",
    "estimation.engine.metric_found_group",
    "estimation.engine.ratio.cuped",
    "estimation.engine.ratio.negative_variance",
    "estimation.engine.winsorization_metadata_differs",
    "estimation.engine.winsorization_metadata_present",
    "estimation.engine.x_role_declared",
)

_COMPATIBILITY_CODES = (
    "allocation.alpha.underflow",
    "arm.adjustment.cluster_cuped",
    "arm.adjustment.cluster_prior",
    "arm.adjustment.sequential_cuped",
    "arm.adjustment.sequential_prior",
    "arm.baseline.absorption_undeclared",
    "arm.baseline.cluster_size_required",
    "arm.baseline.compliance_undeclared",
    "arm.baseline.cuped_undeclared",
    "arm.baseline.iid_cluster_knobs",
    "arm.baseline.triggering_undeclared",
    "arm.inference.cluster",
    "arm.metric.quantile_cluster",
    "arm.metric.quantile_cuped",
    "arm.metric.quantile_sequential",
    "arm.metric.unsupported",
    "compatibility.denominator",
    "compatibility.multiplier",
    "compatibility.support_unsupported",
    "contrast.decision",
    "contrast.inference",
    "contrast.metric",
)

_REACHABLE_ENGINE_CODES = {
    "estimation.engine.arm.duplicate_rows",
    "estimation.engine.arm.invalid_count",
    "estimation.engine.carries_format_moments",
    "estimation.engine.carries_unsupported_weighted",
    "estimation.engine.cluster.denominator",
    "estimation.engine.control_group_found",
    "estimation.engine.cuped_linear_cluster",
    "estimation.engine.cuped_linear_sequential",
    "estimation.engine.cuped_linear_encouragement",
    "estimation.engine.cuped_linear_observational",
    "estimation.engine.cuped_linear_unit_evidence",
    "estimation.engine.cuped_linear_winsorization",
    "estimation.engine.group_summary_missing",
    "estimation.engine.group_summary_row",
    "estimation.engine.method_name_observational",
    "estimation.engine.method_names_unique",
    "estimation.engine.metric_found_group",
    "estimation.engine.ratio.cuped",
    "estimation.engine.ratio.negative_variance",
    "estimation.engine.winsorization_metadata_differs",
    "estimation.engine.winsorization_metadata_present",
    "estimation.engine.x_role_declared",
}
_UNREACHABLE_REASONS = {}
_COMPATIBILITY_UNREACHABLE_REASONS = {
    "arm.baseline.absorption_undeclared": "required_sample_size derives factor_absorption from nonzero Baseline.icc before admission.",
    "arm.baseline.cuped_undeclared": "required_sample_size derives a CUPED method from nonzero Baseline.cuped_rho before admission.",
    "arm.baseline.triggering_undeclared": "required_sample_size derives the triggered population from Baseline.trigger_rate before admission.",
    "arm.adjustment.sequential_cuped": "The public power planner rejects always_valid before admission and admits CUPED under asymptotic_mean.",
    "arm.metric.unsupported": "Validated ArmPlanningProcedure metric capabilities exclude the unsupported metric shape before admission.",
    "compatibility.denominator": "Only the private conservative-ratio helper accepts an arbitrary denominator; public plans derive it from validated inputs.",
    "compatibility.multiplier": "Only the private conservative-ratio helper accepts an arbitrary multiplier; public plans derive it from validated inputs.",
}
_NOT_PUBLICLY_REACHABLE_COMPATIBILITY = {
    code: Unreachable(_COMPATIBILITY_UNREACHABLE_REASONS[code])
    for code in _COMPATIBILITY_CODES
    if code in _COMPATIBILITY_UNREACHABLE_REASONS
}
_NOT_PUBLICLY_REACHABLE = {
    code: Unreachable(_UNREACHABLE_REASONS[code])
    for code in _ENGINE_CODES
    if code in _UNREACHABLE_REASONS
}

REFUSAL_SCENARIOS: dict[str, Callable[[], object] | Unreachable] = {
    **_NOT_PUBLICLY_REACHABLE,
    **{
        code: _engine_scenario(code)
        for code in _ENGINE_CODES
        if code not in _NOT_PUBLICLY_REACHABLE
    },
    **_NOT_PUBLICLY_REACHABLE_COMPATIBILITY,
    **{
        code: _compatibility_scenario(code)
        for code in _COMPATIBILITY_CODES
        if code not in _NOT_PUBLICLY_REACHABLE_COMPATIBILITY
    },
}
