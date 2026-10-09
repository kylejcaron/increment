"""Analysis.planning_baseline computes every derivable Baseline field
from the analysis's own data and declared design, on every constructor
that can supply it."""

from collections.abc import Mapping
from datetime import UTC, datetime

import ibis
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import CapabilityError, CodedError, IncrementWarning, UnsupportedRequestError
from increment.estimation.results import LiftEstimate
from increment.power import Baseline


def _plan(analysis, metric):
    """An arm analysis's planning Baseline (a switchback one returns a
    SwitchbackBaseline instead)."""
    baseline = analysis.planning_baseline(metric)
    assert isinstance(baseline, Baseline)
    return baseline


def _abs_se(analysis):
    row = analysis.run()[0]
    assert isinstance(row, LiftEstimate) and row.abs_se is not None
    return row.abs_se


def _unit_summary_frame(n=400, with_covariate=False, with_cluster=False):
    rows = {
        "unit_id": range(1, n + 1),
        "group_id": (["control"] * (n // 2)) + (["treatment"] * (n // 2)),
        "revenue": [10.0 + (i % 7) + i * 1e-6 for i in range(n)],
    }
    if with_covariate:
        rows["revenue_pre"] = [9.0 + (i % 7) + (i % 2) for i in range(n)]
    if with_cluster:
        rows["household_id"] = [f"h{i // 4}" for i in range(n)]
    return pd.DataFrame(rows)


class TestPlanningBaselineOnDataframeRoutes:
    def test_mean_metric_from_control_arm_data(self):
        analysis = Analysis.from_unit_summary(
            _unit_summary_frame(),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "mean"}],
        )
        baseline = _plan(analysis, "revenue")
        assert isinstance(baseline, Baseline)
        assert baseline.mean == pytest.approx(13.0, rel=0.2)
        assert baseline.var > 0.0

    def test_moments_grain_refusal_propagates_unswallowed(self, monkeypatch):
        """An unrelated moments() failure -- injected here by routing the
        source's own moments() at an unsupported grain -- must never be
        coerced into covariate_source_unavailable or silently swallowed."""
        from tests.analysis_factory import _native_source

        analysis = Analysis.from_unit_summary(
            _unit_summary_frame(with_covariate=True),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "revenue",
                    "type": "mean",
                    "covariate": "revenue_pre",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        source = _native_source(analysis)
        moments = source.moments
        monkeypatch.setattr(
            source, "moments", lambda metric, **kwargs: moments(metric, grain="daily", **kwargs)
        )
        with pytest.raises(CapabilityError) as raised:
            analysis.planning_baseline("revenue")
        assert raised.value.code == "source.frame.grain"

    def test_quantile_metric_from_control_arm_values(self):
        analysis = Analysis.from_unit_summary(
            _unit_summary_frame(),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "quantile", "quantile": 0.5}],
        )
        baseline = _plan(analysis, "revenue")
        assert baseline.mean > 0.0
        assert baseline.var > 0.0

    def test_cuped_rho_derived_from_a_declared_covariate_on_unit_summary(self):
        """cuped_rho must be derived on the dataframe route the same way
        as every other route: from the resolved metric config's decision
        method, not a definitions-only field."""
        analysis = Analysis.from_unit_summary(
            _unit_summary_frame(with_covariate=True),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "revenue",
                    "type": "mean",
                    "covariate": "revenue_pre",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        baseline = _plan(analysis, "revenue")
        assert -1.0 < baseline.cuped_rho < 1.0
        assert baseline.cuped_rho != 0.0

    def test_ratio_metric_uses_the_denominator_moments(self):
        frame = _unit_summary_frame()
        frame["revenue_den"] = [1.0 + (i % 3) for i in range(len(frame))]
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "revenue",
                    "type": "ratio",
                    "numerator": "revenue",
                    "denominator": "revenue_den",
                }
            ],
        )
        baseline = _plan(analysis, "revenue")
        # A ratio Baseline's mean is numerator/denominator, not the bare
        # numerator mean -- pins that the ratio branch, not the mean
        # branch, actually ran.
        assert baseline.mean != pytest.approx(13.0, rel=0.05)

    def test_ratio_metric_cuped_rho_matches_the_runtime_ratio_cuped_reduction(self):
        """A ratio metric that declares CUPED must have its variance
        actually reduced -- ``cuped_rho`` must reflect the runtime's own
        ``fit_ratio_cuped`` adjusted variance, not silently stay at the
        unadjusted default of 0.0."""
        frame = _unit_summary_frame()
        frame["revenue_pre"] = 2.0 * frame["revenue"] + 0.5  # genuinely correlated covariate
        frame["revenue_den"] = [1.0 + (i % 3) + i * 1e-6 for i in range(len(frame))]
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "revenue",
                    "type": "ratio",
                    "numerator": "revenue",
                    "denominator": "revenue_den",
                    "covariate": "revenue_pre",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        baseline = _plan(analysis, "revenue")
        assert -1.0 < baseline.cuped_rho < 1.0
        assert baseline.cuped_rho != 0.0
        assert baseline.effective_var < baseline.var

    def test_ratio_cuped_missing_denominator_cross_moment_surfaces_the_specific_cuped_code(
        self, tmp_path
    ):
        """A covariate that IS present but carries no cross moment with the
        ratio denominator (cxden is None) is a different failure than "no
        covariate at all" -- fit_ratio_cuped's own
        estimation.cuped.ratio_arm_no_denominator_cross must surface
        unswallowed."""
        from increment.errors import InvalidRequestError

        frame = _unit_summary_frame()
        frame["revenue_pre"] = 2.0 * frame["revenue"] + 0.5
        frame["revenue_den"] = [1.0 + (i % 3) + i * 1e-6 for i in range(len(frame))]
        metric = {
            "name": "revenue",
            "type": "ratio",
            "numerator": "revenue",
            "denominator": "revenue_den",
            "covariate": "revenue_pre",
            "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
        }
        direct = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[metric],
        )
        direct.export(tmp_path / "moments.parquet")
        # A raw-sum (format-1) moment file has no column for
        # Cov(covariate, denominator): the covariate is carried, its cross
        # moment with the denominator never materialises.
        rows = [
            {key: value for key, value in row.items() if key != "cxden"}
            for row in pq.read_table(tmp_path / "moments.parquet").to_pylist()
        ]
        adopted = Analysis.from_moments(rows, metrics=[metric], control="control")
        with pytest.raises(InvalidRequestError) as raised:
            adopted.planning_baseline("revenue")
        assert raised.value.code == "estimation.cuped.ratio_arm_no_denominator_cross"

    @pytest.mark.parametrize(
        "metric",
        [
            {"name": "revenue", "type": "mean"},
            {
                "name": "revenue",
                "type": "ratio",
                "numerator": "revenue",
                "denominator": "revenue_den",
            },
        ],
        ids=["mean", "ratio"],
    )
    def test_degenerate_covariate_preserves_cuped_code_in_all_failed_refusal(self, metric):
        """Planning and runtime both retain the specific CUPED guard code."""
        frame = _unit_summary_frame()
        frame["revenue_den"] = [1.0 + (i % 3) for i in range(len(frame))]
        frame["revenue_pre"] = 5.0
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    **metric,
                    "covariate": "revenue_pre",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        with pytest.raises(CodedError) as planned:
            analysis.planning_baseline("revenue")
        with pytest.warns(IncrementWarning) as runtime_warnings:
            with pytest.raises(UnsupportedRequestError) as runtime:
                analysis.run()
        warning = runtime_warnings[0].message
        assert isinstance(warning, IncrementWarning)
        assert warning.code == "readouts.run.cell_refused"
        failures = runtime.value.context["failures"]
        assert isinstance(failures, tuple)
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, Mapping)
        failure_context = failure.get("context")
        assert isinstance(failure_context, Mapping)
        assert failure.get("code") == "estimation.cuped.covariate_zero_variance"
        assert failure_context.get("weighted_var_x") == 0.0
        assert planned.value.code == failure.get("code")

    @pytest.mark.parametrize("metric_type", ["mean", "ratio"])
    def test_cuped_reduction_matches_the_runtime_cuped_standard_error(self, metric_type):
        """Quantities computed twice: at matched inputs (both arms share the
        control arm's covariate structure, so the runtime's fitted slopes are
        the control arm's own), the planned variance reduction 1 - rho^2 is
        the runtime's squared ratio of CUPED to unadjusted standard errors."""
        rng = np.random.default_rng(3)
        n = 300
        x = rng.normal(10.0, 2.0, n)
        y = 5.0 + 0.8 * x + rng.normal(0.0, 1.5, n)
        den = 1.0 + rng.poisson(2.0, n)
        num = den * (2.0 + 0.1 * x) + rng.normal(0.0, 1.0, n)
        # A mean keeps its slope under an additive shift; a ratio's adjusted
        # ratios must agree, so its arms are identical.
        shift = 1.0 if metric_type == "mean" else 0.0
        frame = pd.DataFrame(
            {
                "unit_id": range(2 * n),
                "group_id": ["control"] * n + ["treatment"] * n,
                "y": np.concatenate([y, y + shift]),
                "num": np.concatenate([num, num]),
                "den": np.concatenate([den, den]),
                "x": np.concatenate([x, x]),
            }
        )
        spec = (
            {"name": "m", "type": "mean", "value_column": "y"}
            if metric_type == "mean"
            else {"name": "m", "type": "ratio", "numerator": "num", "denominator": "den"}
        )
        cuped = {"name": "cuped", "variance_reduction": "cuped"}
        adjusted = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{**spec, "covariate": "x", "decision_method": cuped}],
        )
        unadjusted = Analysis.from_unit_summary(
            frame, unit="unit_id", group="group_id", control="control", metrics=[spec]
        )
        baseline = _plan(adjusted, "m")
        runtime_reduction = (_abs_se(adjusted) / _abs_se(unadjusted)) ** 2
        assert baseline.cuped_rho > 0.4
        assert baseline.effective_var / baseline.var == pytest.approx(runtime_reduction, rel=1e-12)

    def test_cluster_fields_derived_from_a_declared_cluster_on_unit_summary(self):
        """avg_cluster_size/cluster_icc/cluster_size_cv must be derived
        from the source's own declared cluster, on the same route as
        cuped_rho above."""
        analysis = Analysis.from_unit_summary(
            _unit_summary_frame(with_cluster=True),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "mean"}],
            cluster="household_id",
        )
        baseline = _plan(analysis, "revenue")
        assert baseline.avg_cluster_size == pytest.approx(4.0, rel=0.2)
        assert 0.0 <= baseline.cluster_icc < 1.0
        assert baseline.cluster_size_cv >= 0.0

    def test_cluster_icc_refuses_when_not_estimable_from_the_control_arm(self):
        """One unit per cluster in the control arm leaves no within/
        between-cluster split to estimate an ICC from -- refuse by name
        rather than dividing by zero."""
        frame = _unit_summary_frame(with_cluster=True)
        frame["household_id"] = [f"h{i}" for i in range(len(frame))]
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "mean"}],
            cluster="household_id",
        )
        with pytest.raises(CapabilityError) as raised:
            analysis.planning_baseline("revenue")
        assert raised.value.code == "analysis.planning_baseline.cluster_icc_not_estimable"

    def test_a_covariate_that_repeats_the_outcome_refuses_by_name(self):
        """No residual variance leaves nothing to plan; the readout refuses the
        same cell as degenerate, so planning must not return a Baseline for it."""
        rng = np.random.default_rng(0)
        y = rng.normal(10.0, 2.0, 400)
        frame = pd.DataFrame(
            {
                "unit_id": range(400),
                "group_id": ["control", "treatment"] * 200,
                "y": y,
                "y_copy": y,
            }
        )
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "m",
                    "type": "mean",
                    "value_column": "y",
                    "covariate": "y_copy",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        with pytest.raises(CapabilityError) as raised:
            analysis.planning_baseline("m")
        assert raised.value.code == "analysis.planning_baseline.cuped_no_residual_variance"

    @pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (3.0, 0.0), (0.1, 7.0)])
    def test_a_covariate_that_rescales_the_outcome_refuses_by_name(self, scale, offset):
        """A covariate that is an affine rescaling of the outcome leaves a
        residual variance that only rounds to a tiny positive float, not
        exactly zero or negative -- the refusal must catch that case too,
        not just the one where rounding happens to land at or below zero."""
        rng = np.random.default_rng(0)
        y = rng.normal(10.0, 2.0, 400)
        frame = pd.DataFrame(
            {
                "unit_id": range(400),
                "group_id": ["control", "treatment"] * 200,
                "y": y,
                "x": scale * y + offset,
            }
        )
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "m",
                    "type": "mean",
                    "value_column": "y",
                    "covariate": "x",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        with pytest.raises(CapabilityError) as raised:
            analysis.planning_baseline("m")
        assert raised.value.code == "analysis.planning_baseline.cuped_no_residual_variance"

    @pytest.mark.parametrize(
        ("seed", "num", "den"),
        [
            (0, (3.0, 0.0), (1.0, 20.0)),
            # Shapes whose linearized variance is far smaller than the terms
            # that cancel in it: the tolerance must scale with those terms.
            (0, (3.0, 5.0), (1.0, 1.0)),
            (3, (3.0, 5.0), (1.0, 1.0)),
            (7, (0.7, 4.0), (2.5, 0.5)),
        ],
    )
    def test_a_ratio_covariate_that_rescales_both_components_refuses_by_name(self, seed, num, den):
        """A ratio whose numerator and denominator are both affine in the
        same covariate has zero residual variance too; the refusal must
        reach ratio metrics, not just mean ones."""
        x = np.random.default_rng(seed).uniform(1.0, 10.0, 400)
        frame = pd.DataFrame(
            {
                "unit_id": range(400),
                "group_id": ["control", "treatment"] * 200,
                "num": num[0] * x + num[1],
                "den": den[0] * x + den[1],
                "x": x,
            }
        )
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {
                    "name": "m",
                    "type": "ratio",
                    "numerator": "num",
                    "denominator": "den",
                    "covariate": "x",
                    "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                }
            ],
        )
        with pytest.raises(CapabilityError) as raised:
            analysis.planning_baseline("m")
        assert raised.value.code == "analysis.planning_baseline.cuped_no_residual_variance"

    def test_clustered_ratio_icc_is_the_icc_of_its_linearized_score(self):
        """The design effect inflates the ratio's linearized variance, so the
        ICC must be that of y - R*y_den, not of the numerator. Here cluster
        activity drives both components (numerator ICC near 1) while the
        linearized score carries only a modest cluster effect."""
        rows = []
        for arm in ("control", "treatment"):
            for c in range(30):
                for u in range(2 + c % 3):
                    den = (1.0 + c % 5) * (1.0 + 0.1 * u)
                    noise = 0.1 * ((c % 4) - 1.5) + 0.3 * (u % 2)
                    rows.append(
                        {
                            "unit_id": f"{arm}{c}_{u}",
                            "group_id": arm,
                            "store": f"{arm}{c}",
                            "num": 2.0 * den + noise,
                            "den": den,
                        }
                    )
        frame = pd.DataFrame(rows)
        control = frame[frame["group_id"] == "control"]
        ratio = control["num"].sum() / control["den"].sum()
        # The linearized score, shifted positive (an ICC is shift-invariant).
        frame["score"] = frame["num"] - ratio * frame["den"] + 10.0
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[
                {"name": "rps", "type": "ratio", "numerator": "num", "denominator": "den"},
                {"name": "score", "type": "mean", "value_column": "score"},
                {"name": "num", "type": "mean", "value_column": "num"},
            ],
            cluster="store",
        )
        ratio_icc = _plan(analysis, "rps").cluster_icc
        assert ratio_icc == pytest.approx(_plan(analysis, "score").cluster_icc, rel=1e-9)
        assert 0.0 < ratio_icc < 0.5 < _plan(analysis, "num").cluster_icc

    def test_clustered_mean_and_var_are_per_unit_not_cluster_totals(self):
        """A clustered source's moments are cluster totals; the Baseline's
        mean/var are per unit, the same as the unclustered source's on the
        same rows, with the clustering carried by the design effect alone."""
        frame = _unit_summary_frame(with_cluster=True)

        def build(cluster):
            return Analysis.from_unit_summary(
                frame,
                unit="unit_id",
                group="group_id",
                control="control",
                metrics=[{"name": "revenue", "type": "mean"}],
                cluster=cluster,
            )

        unit_level = _plan(build(None), "revenue")
        baseline = _plan(build("household_id"), "revenue")
        assert (baseline.mean, baseline.var) == pytest.approx(
            (unit_level.mean, unit_level.var), rel=1e-12
        )
        assert baseline.avg_cluster_size == 4.0


_TRIGGERED_DEFS = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [unit_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: exposure
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: exposure
    fact: exposure
metrics:
  - name: revenue
    type: mean
    entity: unit_id
    fact: revenue
    aggregation: sum
    window_days: 7
  - name: revenue_p50
    type: quantile
    quantile: 0.5
    entity: unit_id
    fact: revenue
    aggregation: sum
experiments:
  - name: exp
    exposure: assignment
    trigger: exposure
    unit: unit_id
    start: "2024-01-01T00:00:00"
    control_group: control
    plan:
      primary: revenue
      secondaries: [revenue_p50]
"""


def _triggered_value(i):
    # Triggered units spend far more, so the two populations' moments differ.
    return 40.0 + (i % 7) if i % 5 == 0 else 10.0 + (i % 3)


def _triggered_analysis(
    tmp_path,
    *,
    clustered=False,
    assigned_sizes=None,
    analyzed_sizes=None,
    metric_aggregation="sum",
    source_snapshot_evidence=None,
    staggered=False,
):
    """Selected pilot memberships, or every fifth unit in an IID experiment."""
    con = ibis.duckdb.connect()
    rows = {
        "unit_id": [],
        "group_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
        **({"store_id": []} if clustered else {}),
    }
    if assigned_sizes is None:
        assigned_sizes = [10] * 20
    if analyzed_sizes is None:
        analyzed_sizes = [2] * 20
    members = (
        [(cluster, member) for cluster, size in enumerate(assigned_sizes) for member in range(size)]
        if clustered
        else [(0, member) for member in range(200)]
    )

    def add(uid, arm, event, value, ts, store_id=None):
        base = (uid, arm, ts, event, value, "exp")
        for key, item in zip(
            ("unit_id", "group_id", "ts", "event", "value", "experiment_id"), base, strict=True
        ):
            rows[key].append(item)
        if clustered:
            rows["store_id"].append(store_id)

    for arm in ("control", "treatment"):
        for i, (cluster, member) in enumerate(members):
            uid = f"{arm}{i}"
            store_id = f"{arm}-store-{cluster}"
            add(uid, arm, "enrolled", None, pd.Timestamp("2024-01-01"), store_id)
            triggered = member < analyzed_sizes[cluster] if clustered else i % 5 == 0
            trigger_ts = (
                pd.Timestamp("2024-01-02") + pd.Timedelta(days=cluster % 3)
                if staggered and clustered
                else pd.Timestamp("2024-01-01")
            )
            if triggered:
                add(uid, arm, "exposure", None, trigger_ts, store_id)
            value = (
                20 + 10 * cluster + (-4 if member % 2 == 0 else 4)
                if clustered
                else _triggered_value(i)
            )
            if staggered and clustered and triggered:
                add(
                    uid,
                    arm,
                    "revenue",
                    1000 + value,
                    pd.Timestamp("2024-01-01"),
                    store_id,
                )
            outcome_offset = 6 if staggered and clustered else 1
            outcome_ts = (
                trigger_ts + pd.Timedelta(days=outcome_offset)
                if triggered
                else pd.Timestamp("2024-01-02")
            )
            add(uid, arm, "revenue", value, outcome_ts, store_id)
            # Anchors the observable data extent past the metric window; this
            # event falls outside the window and is never summed.
            add(uid, arm, "revenue", 0.0, pd.Timestamp("2024-01-20"), store_id)
    con.create_table("events", obj=pd.DataFrame(rows))
    path = tmp_path / ("defs_triggered_clustered.yml" if clustered else "defs_triggered.yml")
    defs = _TRIGGERED_DEFS.replace(
        "    trigger: exposure",
        "    trigger: exposure\n    cluster: store_id" if clustered else "    trigger: exposure",
    )
    defs = defs.replace("    aggregation: sum", f"    aggregation: {metric_aggregation}", 1)
    path.write_text(defs)
    evidence = datetime(2030, 1, 1, tzinfo=UTC)
    evidence = source_snapshot_evidence or SourceSnapshotEvidence(evidence, {"events": evidence})
    return (
        con,
        path,
        Analysis.from_definitions(
            "exp",
            str(path),
            con,
            source_snapshot_evidence=evidence,
        ),
    )


def test_triggered_planning_baseline_refuses_when_control_has_no_trigger(tmp_path):
    con, _path, analysis = _triggered_analysis(tmp_path)
    con.raw_sql("DELETE FROM events WHERE event = 'exposure' AND group_id = 'control'")
    try:
        with pytest.raises(CodedError) as raised:
            analysis.planning_baseline("revenue")
        assert raised.value.code == "query.integrity.trigger_arm_missing"
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.filterwarnings("always::increment.errors.IncrementRuntimeWarning")
@pytest.mark.parametrize(
    ("assigned_sizes", "analyzed_sizes", "expected_df"),
    [
        ([10] * 10, [2] * 10, 9),
        ([10] * 10, [4] * 5 + [0] * 5, 4),
        ([5, 15], [4, 1], 1),
    ],
    ids=["all-clusters-contribute", "half-clusters-contribute", "unequal-analyzed-sizes"],
)
def test_triggered_cluster_source_and_artifact_preserve_assigned_and_analyzed_grains(
    tmp_path, assigned_sizes, analyzed_sizes, expected_df
):

    import warnings
    from fractions import Fraction

    from scipy.stats import chi2

    from increment import IncrementRuntimeWarning
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.power import (
        PowerDesign,
        achieved_power,
        minimum_detectable_effect,
        required_sample_size,
    )
    from increment.semantics.artifact import (
        AssignmentCountsRequest,
        ClusterIdentityRequest,
        TriggerMeasureStatsRequest,
        TriggerPopulationRequest,
    )

    def run_with_small_cluster_advisories(source, n_clusters):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IncrementRuntimeWarning)
            results = source.run(metrics=["revenue"])
        assert [
            (warning.message.code, dict(warning.message.context))
            for warning in caught
            if isinstance(warning.message, IncrementRuntimeWarning)
        ] == [
            (
                "estimation.engine.small_total_clusters",
                {
                    "metric_name": "revenue",
                    "n_clusters": n_clusters,
                    "cluster": "store_id",
                    "floor": 40,
                },
            ),
            (
                "estimation.engine.small_total_clusters",
                {
                    "metric_name": "revenue",
                    "n_clusters": 2 * sum(size > 0 for size in analyzed_sizes),
                    "cluster": "store_id",
                    "floor": 40,
                },
            ),
        ]
        return results

    groups = [
        [Fraction(20 + 10 * cluster + (-4 if member % 2 == 0 else 4)) for member in range(size)]
        for cluster, size in enumerate(analyzed_sizes)
        if size
    ]
    values = [value for group in groups for value in group]
    n, k = len(values), len(groups)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    group_means = [sum(group) / len(group) for group in groups]
    between = sum(
        len(group) * (group_mean - mean) ** 2
        for group, group_mean in zip(groups, group_means, strict=True)
    )
    within = sum(
        sum((value - group_mean) ** 2 for value in group)
        for group, group_mean in zip(groups, group_means, strict=True)
    )
    msb, msw = between / (k - 1), within / (n - k)
    n0 = (n - Fraction(sum(len(group) ** 2 for group in groups), n)) / (k - 1)
    icc = max(Fraction(0), (msb - msw) / (msb + (n0 - 1) * msw))
    analyzed_mean = Fraction(n, k)
    size_variance = sum((len(group) - analyzed_mean) ** 2 for group in groups) / k
    manual = Baseline(
        mean=float(mean),
        var=float(variance) * (n - 1) / chi2.isf(0.8, n - 1),
        trigger_rate=n / sum(assigned_sizes),
        avg_cluster_size=sum(assigned_sizes) / len(assigned_sizes),
        cluster_participation=k / len(assigned_sizes),
        cluster_icc=float(icc),
        cluster_size_cv=float(size_variance / analyzed_mean**2) ** 0.5,
    )
    con, path, analysis = _triggered_analysis(
        tmp_path, clustered=True, assigned_sizes=assigned_sizes, analyzed_sizes=analyzed_sizes
    )
    reopened = None
    try:
        native = _plan(analysis, "revenue")
        reopened = _republish(
            con,
            path,
            analysis,
            [
                TriggerPopulationRequest(trigger_name="exposure"),
                AssignmentCountsRequest(populations=("assigned", "triggered")),
                TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("revenue",)),
                ClusterIdentityRequest(cluster_name="store_id"),
            ],
        )
        artifact = _plan(reopened, "revenue")
        for baseline in (native, artifact):
            assert baseline.model_dump() == pytest.approx(manual.model_dump(), rel=1e-12)
            assert baseline.design_effect == pytest.approx(
                1 + ((1 + manual.cluster_size_cv**2) * float(analyzed_mean) - 1) * float(icc)
            )
        for source in (analysis, reopened):
            results = {
                row.analysis_population: row
                for row in run_with_small_cluster_advisories(source, 2 * len(assigned_sizes))
            }
            assert set(results) == {"assigned", "triggered"}
            assert results["assigned"].reference_df == len(assigned_sizes) - 1
            result = results["triggered"]
            assert result.reference_df == expected_df
            assert result.lift is not None
            assert result.lift.value == pytest.approx(0)

        standard = ArmPlanningProcedure.standard("mean", clustered=True)
        procedure = standard.model_copy(
            update={"analysis": standard.analysis.model_copy(update={"population": "triggered"})}
        )
        design = PowerDesign(allocation=0.5, power=0.8)
        for baseline in (native, artifact):
            actual = (
                required_sample_size(0.1, baseline, procedure, design),
                achieved_power(1000, 0.1, baseline, procedure, design),
                minimum_detectable_effect(1000, baseline, procedure, design),
            )
            expected = (
                required_sample_size(0.1, manual, procedure, design),
                achieved_power(1000, 0.1, manual, procedure, design),
                minimum_detectable_effect(1000, manual, procedure, design),
            )
            for observed, independent in zip(actual, expected, strict=True):
                assert observed.model_dump() == pytest.approx(independent.model_dump(), rel=1e-12)
    finally:
        analysis.close()
        if reopened is not None:
            reopened.close()
        con.disconnect()


@pytest.mark.filterwarnings("always::increment.errors.IncrementRuntimeWarning")
def test_staggered_cluster_triggers_anchor_planning_and_observed_counts(tmp_path):
    import warnings

    from increment import IncrementRuntimeWarning
    from increment.semantics.artifact import (
        AssignmentCountsRequest,
        ClusterIdentityRequest,
        TriggerMeasureStatsRequest,
        TriggerPopulationRequest,
    )

    assigned_sizes = [6, 6, 6, 6]
    analyzed_sizes = [2, 3, 1, 0]
    con, path, analysis = _triggered_analysis(
        tmp_path,
        clustered=True,
        assigned_sizes=assigned_sizes,
        analyzed_sizes=analyzed_sizes,
        staggered=True,
    )
    reopened = None
    try:
        native = _plan(analysis, "revenue")
        reopened = _republish(
            con,
            path,
            analysis,
            [
                TriggerPopulationRequest(trigger_name="exposure"),
                AssignmentCountsRequest(populations=("assigned", "triggered")),
                TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("revenue",)),
                ClusterIdentityRequest(cluster_name="store_id"),
            ],
        )
        artifact = _plan(reopened, "revenue")
        assert native.model_dump() == pytest.approx(artifact.model_dump(), rel=1e-12)
        assert native.mean == pytest.approx(27.0)
        assert native.trigger_rate == pytest.approx(6 / 24)
        assert native.cluster_participation == pytest.approx(3 / 4)
        for source in (analysis, reopened):
            integrity = source.srm(
                expected={"control": 0.5, "treatment": 0.5},
                inference="fixed",
                population="triggered",
            )
            assert integrity.grain == "cluster"
            assert integrity.observed == {"control": 3, "treatment": 3}
            assert integrity.unit_counts == {"control": 6, "treatment": 6}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IncrementRuntimeWarning)
                results = source.run(metrics=["revenue"])
            assert [
                (warning.message.code, dict(warning.message.context))
                for warning in caught
                if isinstance(warning.message, IncrementRuntimeWarning)
            ] == [
                (
                    "estimation.engine.small_total_clusters",
                    {
                        "metric_name": "revenue",
                        "n_clusters": 8,
                        "cluster": "store_id",
                        "floor": 40,
                    },
                ),
                (
                    "estimation.engine.small_total_clusters",
                    {
                        "metric_name": "revenue",
                        "n_clusters": 6,
                        "cluster": "store_id",
                        "floor": 40,
                    },
                ),
            ]
            result = next(row for row in results if row.analysis_population == "triggered")
            assert result.reference_df == 2
    finally:
        analysis.close()
        if reopened is not None:
            reopened.close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("late_condition", "metric_aggregation", "observed_units"),
    [
        ("store_id LIKE '%store-9'", "sum", 18),
        ("unit_id IN ('control90', 'treatment90')", "sum", 19),
        (None, "avg_event", 19),
    ],
    ids=["whole-cluster", "one-member", "mature-undefined-outcome"],
)
def test_triggered_cluster_planning_names_incomplete_metric_population(
    tmp_path, late_condition, metric_aggregation, observed_units
):
    import copy
    import pickle

    from increment.semantics.artifact import (
        AssignmentCountsRequest,
        ClusterIdentityRequest,
        TriggerMeasureStatsRequest,
        TriggerPopulationRequest,
    )

    certified_edge = datetime(2024, 1, 20, tzinfo=UTC)
    con, path, analysis = _triggered_analysis(
        tmp_path,
        clustered=True,
        assigned_sizes=[10] * 10,
        analyzed_sizes=[2] * 10,
        metric_aggregation=metric_aggregation,
        source_snapshot_evidence=SourceSnapshotEvidence(certified_edge, {"events": certified_edge}),
    )
    reopened = None
    fully_certified = None
    try:
        if late_condition is None:
            con.raw_sql(
                "DELETE FROM events WHERE event = 'revenue' "
                "AND unit_id IN ('control90', 'treatment90')"
            )
        else:
            con.raw_sql(
                "UPDATE events SET ts = ts + INTERVAL 17 DAY "
                f"WHERE {late_condition} AND ts < TIMESTAMP '2024-01-20'"
            )
        if late_condition is not None:
            later_edge = datetime(2030, 1, 1, tzinfo=UTC)
            fully_certified = Analysis.from_definitions(
                "exp",
                str(path),
                con,
                source_snapshot_evidence=SourceSnapshotEvidence(later_edge, {"events": later_edge}),
            )
            _plan(fully_certified, "revenue")
        reopened = _republish(
            con,
            path,
            analysis,
            [
                TriggerPopulationRequest(trigger_name="exposure"),
                AssignmentCountsRequest(populations=("assigned", "triggered")),
                TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("revenue",)),
                ClusterIdentityRequest(cluster_name="store_id"),
            ],
        )
        for source in (analysis, reopened):
            with pytest.raises(CapabilityError) as raised:
                _plan(source, "revenue")
            for error in (
                raised.value,
                copy.deepcopy(raised.value),
                pickle.loads(pickle.dumps(raised.value)),
            ):
                assert (
                    error.code == "analysis.planning_baseline.trigger_metric_population_unavailable"
                )
                assert error.context["metric"] == "revenue"
                assert error.context["eligible_units"] == 20
                assert error.context["observed_units"] == observed_units
    finally:
        analysis.close()
        if reopened is not None:
            reopened.close()
        if fully_certified is not None:
            fully_certified.close()
        con.disconnect()


def _republish(con, path, analysis, extensions):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load

    defs = load(str(path))
    experiment = defs.experiment("exp")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = analysis.publish_unit_day_artifact(store, extensions=extensions)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context)


class TestPlanningBaselineOnWarehouseRoutes:
    def test_declared_trigger_reads_the_triggered_population(self, tmp_path):
        """The solvers read mean/var as the analyzed population and inflate
        assigned n by 1/trigger_rate, so under a declared trigger every field
        comes from the triggered control units -- quantiles included."""
        from increment.estimation.armstats import SummaryStats

        con, _path, analysis = _triggered_analysis(tmp_path)
        try:
            triggered = np.array([_triggered_value(i) for i in range(200) if i % 5 == 0])
            expected = Baseline.from_summary(
                SummaryStats(n=triggered.size, mean=triggered.mean(), var=triggered.var(ddof=1))
            )
            baseline = _plan(analysis, "revenue")
            assert (baseline.mean, baseline.var) == pytest.approx(
                (expected.mean, expected.var), rel=1e-12
            )
            assert baseline.trigger_rate == pytest.approx(0.2, rel=1e-12)
            quantile = _plan(analysis, "revenue_p50")
            assert quantile.mean == pytest.approx(float(np.quantile(triggered, 0.5)), rel=1e-12)
            assert quantile.trigger_rate == pytest.approx(0.2, rel=1e-12)
        finally:
            con.disconnect()

    def test_artifact_with_trigger_evidence_matches_definitions(self, tmp_path):
        from increment.semantics.artifact import (
            AssignmentCountsRequest,
            TriggerMeasureStatsRequest,
            TriggerPopulationRequest,
        )

        con, path, analysis = _triggered_analysis(tmp_path)
        try:
            native = _plan(analysis, "revenue")
            reopened = _republish(
                con,
                path,
                analysis,
                [
                    TriggerPopulationRequest(trigger_name="exposure"),
                    AssignmentCountsRequest(populations=("assigned", "triggered")),
                    TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("revenue",)),
                ],
            )
            assert _plan(reopened, "revenue").model_dump() == pytest.approx(
                native.model_dump(), rel=1e-12
            )
        finally:
            con.disconnect()

    @pytest.mark.slow
    def test_triggered_cuped_planning_uses_assignment_anchored_preperiod(  # noqa: PLR0915
        self, tmp_path
    ):
        import warnings

        from scipy.optimize import brentq
        from scipy.stats import norm

        from increment.estimation.arm_contract import ArmPlanningProcedure
        from increment.estimation.armstats import SummaryStats
        from increment.estimation.engine import Method
        from increment.estimation.inference import Normal
        from increment.estimation.priors import MixturePrior, StudentTPrior
        from increment.power import PowerDesign, required_sample_size
        from increment.semantics.artifact import (
            AssignmentCountsRequest,
            CupedPreperiodRequest,
            TriggerMeasureStatsRequest,
            TriggerPopulationRequest,
        )

        con = ibis.duckdb.connect()
        rows = {
            "unit_id": [],
            "group_id": [],
            "ts": [],
            "event": [],
            "value": [],
            "sessions": [],
            "experiment_id": [],
        }

        def add(uid, arm, event, value, timestamp, *, sessions=None):
            rows["unit_id"].append(uid)
            rows["group_id"].append(arm)
            rows["ts"].append(pd.Timestamp(timestamp))
            rows["event"].append(event)
            rows["value"].append(value)
            rows["sessions"].append(sessions)
            rows["experiment_id"].append("exp")

        for arm in ("control", "treatment"):
            for i in range(40):
                uid = f"{arm}{i}"
                assignment_day = "2024-01-11" if i % 4 < 2 else "2024-01-12"
                add(uid, arm, "enrolled", None, assignment_day)
                x = float(i % 8)
                pre_day = (pd.Timestamp(assignment_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                add(uid, arm, "revenue", x, pre_day)
                # This post-assignment signal differs from the pre-period x;
                # it would contaminate a trigger-anchored CUPED covariate.
                add(uid, arm, "revenue", 100.0 if i % 4 == 0 else 50.0, "2024-01-14")
                if i % 2 == 0:
                    add(uid, arm, "exposure", None, "2024-01-17")
                    add(
                        uid,
                        arm,
                        "revenue",
                        20.0 + x + (i % 3) + (2.0 if arm == "treatment" else 0.0),
                        "2024-01-18",
                    )
                    add(
                        uid,
                        arm,
                        "sessions",
                        None,
                        "2024-01-18",
                        sessions=2.0 + i % 3,
                    )
                add(uid, arm, "revenue", 0.0, "2024-02-01")
        con.create_table("events", obj=pd.DataFrame(rows))
        defs = (
            _TRIGGERED_DEFS.replace(
                "      - name: enrolled",
                "      - name: sessions\n        column: sessions\n      - name: enrolled",
            )
            .replace(
                "experiments:\n",
                "  - name: rps\n"
                "    type: ratio\n"
                "    entity: unit_id\n"
                "    numerator:\n"
                "      fact: revenue\n"
                "      aggregation: sum\n"
                "      window_days: 7\n"
                "    denominator:\n"
                "      fact: sessions\n"
                "      aggregation: sum\n"
                "      window_days: 7\n"
                "  - name: fixed_revenue\n"
                "    type: mean\n"
                "    entity: unit_id\n"
                "    fact: revenue\n"
                "    aggregation: sum\n"
                "    window_days: 7\n"
                "    preferred_direction: increase\n"
                "    winsorization: {upper_value: 25}\n"
                "  - name: percentile_revenue\n"
                "    type: mean\n"
                "    entity: unit_id\n"
                "    fact: revenue\n"
                "    aggregation: sum\n"
                "    window_days: 7\n"
                "    preferred_direction: increase\n"
                "    winsorization:\n"
                "      upper_percentile: 0.75\n"
                "      support: {lower: 0, provenance: fixture non-negative revenue}\n"
                "experiments:\n",
            )
            .replace(
                '    start: "2024-01-01T00:00:00"',
                '    start: "2024-01-11T00:00:00"\n'
                '    end: "2024-01-15T00:00:00"\n'
                '    observation_end: "2024-02-01T00:00:00"',
            )
            .replace(
                "    plan:\n      primary: revenue\n      secondaries: [revenue_p50]",
                "    n_pre_periods: 7\n"
                "    plan:\n"
                "      primary:\n"
                "        metric: revenue\n"
                "        decision_method: {name: cuped, variance_reduction: cuped}\n"
                "      secondaries:\n"
                "        - revenue_p50\n"
                "        - metric: rps\n"
                "          decision_method: {name: cuped, variance_reduction: cuped}\n"
                "      guardrails: [fixed_revenue, percentile_revenue]",
            )
        )
        path = tmp_path / "defs_triggered_cuped.yml"
        path.write_text(defs)
        evidence = datetime(2030, 1, 1, tzinfo=UTC)
        analysis = Analysis.from_definitions(
            "exp",
            str(path),
            con,
            source_snapshot_evidence=SourceSnapshotEvidence(evidence, {"events": evidence}),
        )
        reopened = None
        try:
            indices = np.arange(0, 40, 2)
            triggered = 20.0 + indices % 8 + indices % 3
            denominators = 2.0 + indices % 3
            expected = Baseline.from_summary(
                SummaryStats(n=triggered.size, mean=triggered.mean(), var=triggered.var(ddof=1))
            )
            expected_ratio = triggered.sum() / denominators.sum()
            expected_ratio_var = (
                np.var(triggered - expected_ratio * denominators, ddof=1) / denominators.mean() ** 2
            )
            covariate = indices % 8
            theta_num = np.cov(covariate, triggered, ddof=1)[0, 1] / np.var(covariate, ddof=1)
            theta_den = np.cov(covariate, denominators, ddof=1)[0, 1] / np.var(covariate, ddof=1)
            adjusted_num = triggered - theta_num * (covariate - covariate.mean())
            adjusted_den = denominators - theta_den * (covariate - covariate.mean())
            expected_ratio_adjusted_var = (
                np.var(adjusted_num - expected_ratio * adjusted_den, ddof=1)
                / adjusted_den.mean() ** 2
            )
            expected_ratio_rho = np.sqrt(1 - expected_ratio_adjusted_var / expected_ratio_var)
            expected_rho = 98.0 / np.sqrt(100.0 * 110.0)
            expected_quantile = float(np.quantile(triggered, 0.5))
            native = _plan(analysis, "revenue")
            reopened = _republish(
                con,
                path,
                analysis,
                [
                    CupedPreperiodRequest(metric_name="revenue"),
                    CupedPreperiodRequest(metric_name="rps"),
                    TriggerPopulationRequest(trigger_name="exposure"),
                    AssignmentCountsRequest(populations=("assigned", "triggered")),
                    TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("revenue",)),
                    TriggerMeasureStatsRequest(
                        trigger_name="exposure", metric_names=("revenue_p50",)
                    ),
                    TriggerMeasureStatsRequest(trigger_name="exposure", metric_names=("rps",)),
                    TriggerMeasureStatsRequest(
                        trigger_name="exposure", metric_names=("fixed_revenue",)
                    ),
                    TriggerMeasureStatsRequest(
                        trigger_name="exposure", metric_names=("percentile_revenue",)
                    ),
                ],
            )
            artifact = _plan(reopened, "revenue")
            assert artifact.model_dump() == pytest.approx(native.model_dump(), rel=1e-12)
            native_ratio = _plan(analysis, "rps")
            artifact_ratio = _plan(reopened, "rps")
            assert artifact_ratio.model_dump() == pytest.approx(
                native_ratio.model_dump(), rel=1e-12
            )
            assert (native_ratio.mean, native_ratio.var) == pytest.approx(
                (expected_ratio, expected_ratio_var), rel=1e-12
            )
            assert native_ratio.trigger_rate == pytest.approx(0.5, rel=1e-12)
            native_quantile = _plan(analysis, "revenue_p50")
            artifact_quantile = _plan(reopened, "revenue_p50")
            for baseline in (native_quantile, artifact_quantile):
                assert baseline.mean == pytest.approx(expected_quantile, rel=1e-12)
                assert baseline.trigger_rate == pytest.approx(0.5, rel=1e-12)
            for baseline in (native, artifact):
                assert (baseline.mean, baseline.var) == pytest.approx(
                    (expected.mean, expected.var), rel=1e-12
                )
                assert baseline.trigger_rate == pytest.approx(0.5, rel=1e-12)
                assert baseline.cuped_rho == pytest.approx(expected_rho, rel=1e-12)
                assert baseline.effective_var == pytest.approx(
                    baseline.var * (1 - expected_rho**2), rel=1e-12
                )
                planned = required_sample_size(
                    0.1, baseline, ArmPlanningProcedure.standard("mean"), PowerDesign()
                )
                assert planned.n_triggered_per_arm == round(
                    planned.n_per_arm * baseline.trigger_rate
                )
                assert planned.n_triggered_total == round(planned.n_total * baseline.trigger_rate)
            assert native_ratio.cuped_rho == pytest.approx(expected_ratio_rho, rel=1e-12)
            assert native_ratio.effective_var == pytest.approx(
                expected_ratio_var * (1 - expected_ratio_rho**2), rel=1e-12
            )
            assert artifact_ratio.cuped_rho == pytest.approx(expected_ratio_rho, rel=1e-12)
            assert artifact_ratio.effective_var == pytest.approx(
                expected_ratio_var * (1 - expected_ratio_rho**2), rel=1e-12
            )

            mean_control = float(np.mean(triggered))
            mean_treatment = float(np.mean(triggered + 2.0))
            log_rr = float(np.log(mean_treatment / mean_control))
            se_log_rr = float(
                np.sqrt(
                    np.var(triggered, ddof=1) / (triggered.size * mean_control**2)
                    + np.var(triggered + 2.0, ddof=1) / (triggered.size * mean_treatment**2)
                )
            )

            def expected_posterior(prior):
                if isinstance(prior, Normal):
                    prior_variance = prior.sigma**2
                    sampling_variance = se_log_rr**2
                    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / sampling_variance)
                    posterior_mean = posterior_variance * (
                        prior.mu / prior_variance + log_rr / sampling_variance
                    )
                    posterior_sd = float(np.sqrt(posterior_variance))
                    z_975 = float(norm.ppf(0.975))
                    latent = (
                        posterior_mean,
                        posterior_mean - z_975 * posterior_sd,
                        posterior_mean + z_975 * posterior_sd,
                    )
                else:
                    mixture = prior.components() if isinstance(prior, StudentTPrior) else prior
                    prior_means = np.asarray(mixture.means, dtype=float)
                    prior_sigmas = np.asarray(mixture.sigmas, dtype=float)
                    weights = np.asarray(mixture.weights, dtype=float)
                    log_weights = np.log(weights) + norm.logpdf(
                        log_rr,
                        loc=prior_means,
                        scale=np.sqrt(prior_sigmas**2 + se_log_rr**2),
                    )
                    weights = np.exp(log_weights - log_weights.max())
                    weights /= weights.sum()
                    posterior_variances = 1.0 / (1.0 / prior_sigmas**2 + 1.0 / se_log_rr**2)
                    posterior_means = posterior_variances * (
                        prior_means / prior_sigmas**2 + log_rr / se_log_rr**2
                    )
                    posterior_sds = np.sqrt(posterior_variances)

                    def mixture_cdf(value):
                        return float(
                            np.sum(weights * norm.cdf((value - posterior_means) / posterior_sds))
                        )

                    left = float(np.min(posterior_means - 12.0 * posterior_sds))
                    right = float(np.max(posterior_means + 12.0 * posterior_sds))
                    latent = tuple(
                        brentq(
                            lambda value, probability=probability: mixture_cdf(value) - probability,
                            left,
                            right,
                        )
                        for probability in (0.5, 0.025, 0.975)
                    )
                return tuple(float(np.expm1(value)) for value in latent)

            priors = (
                Normal(mu=0.0, sigma=0.2),
                StudentTPrior(nu=5, scale=0.2, k=8),
                MixturePrior(weights=(0.5, 0.5), means=(-0.1, 0.1), sigmas=(0.2, 0.2)),
            )
            for prior in priors:
                posterior_by_ingress = {}
                expected = expected_posterior(prior)
                for ingress, source in (("definitions", analysis), ("artifact", reopened)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        plain_rows = source.run(
                            metrics=["revenue"],
                            decision_method=Method(name="unadjusted"),
                            prior=None,
                        )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        informed_rows = source.run(
                            metrics=["revenue"],
                            decision_method=Method(name="unadjusted"),
                            prior=prior,
                        )
                    plain = next(
                        row for row in plain_rows if row.analysis_population == "triggered"
                    )
                    informed = next(
                        row for row in informed_rows if row.analysis_population == "triggered"
                    )
                    assert informed.lift == plain.lift
                    assert informed.reference_kind == plain.reference_kind
                    assert informed.reference_df == plain.reference_df
                    assert informed.p_value() == pytest.approx(plain.p_value(), rel=1e-12)
                    assert informed.posterior_available is True
                    assert informed.posterior_model == (
                        "normal" if isinstance(prior, Normal) else "mixture"
                    )
                    assert informed.posterior_estimate == pytest.approx(expected[0], rel=1e-8)
                    assert informed.posterior_lb == pytest.approx(expected[1], rel=1e-8)
                    assert informed.posterior_ub == pytest.approx(expected[2], rel=1e-8)
                    posterior_by_ingress[ingress] = (
                        informed.posterior_estimate,
                        informed.posterior_lb,
                        informed.posterior_ub,
                    )
                assert posterior_by_ingress["definitions"] == pytest.approx(
                    posterior_by_ingress["artifact"], rel=1e-12
                )
            from typing import Any, cast

            from tests.analysis_factory import _native_source

            expected_percentile_cutoff = float(
                np.quantile(np.concatenate((triggered, triggered + 2.0)), 0.75)
            )
            for source in (analysis, reopened):
                triggered_source = cast(Any, _native_source(source).triggered_source())
                metrics_by_name = {metric.name: metric for metric in source.metrics}
                fixed_metric = metrics_by_name["fixed_revenue"]
                raw_fixed = triggered_source.unit_frame(
                    fixed_metric, outcome_stage="raw"
                ).to_pylist()
                clipped_fixed = triggered_source.unit_frame(
                    fixed_metric, outcome_stage="transformed"
                ).to_pylist()
                raw_fixed_by_unit = {row["unit_id"]: row["y"] for row in raw_fixed}
                clipped_fixed_by_unit = {row["unit_id"]: row["y"] for row in clipped_fixed}
                assert len(raw_fixed_by_unit) == len(clipped_fixed_by_unit) == 40
                assert clipped_fixed_by_unit == {
                    unit_id: min(value, 25.0) for unit_id, value in raw_fixed_by_unit.items()
                }

                from increment.estimation.winsor import raw_state_from_source

                percentile_metric = metrics_by_name["percentile_revenue"]
                percentile_raw = raw_state_from_source(triggered_source, percentile_metric)
                assert percentile_raw.population == "triggered"
                assert sorted(percentile_raw.arm("control").values) == sorted(triggered)
                assert sorted(percentile_raw.arm("treatment").values) == sorted(triggered + 2.0)
                raw_percentile = triggered_source.unit_frame(
                    percentile_metric, outcome_stage="raw"
                ).to_pylist()
                clipped_percentile = triggered_source.unit_frame(
                    percentile_metric, outcome_stage="transformed"
                ).to_pylist()
                raw_percentile_by_unit = {row["unit_id"]: row["y"] for row in raw_percentile}
                clipped_percentile_by_unit = {
                    row["unit_id"]: row["y"] for row in clipped_percentile
                }
                assert clipped_percentile_by_unit == {
                    unit_id: min(value, expected_percentile_cutoff)
                    for unit_id, value in raw_percentile_by_unit.items()
                }
                assert max(clipped_percentile_by_unit.values()) == pytest.approx(
                    expected_percentile_cutoff
                )
                with pytest.raises(CapabilityError) as raised:
                    source.run(metrics=["percentile_revenue"])
                assert raised.value.code == "readout.metric.percentile_winsorization"
        finally:
            analysis.close()
            if reopened is not None:
                reopened.close()
            con.disconnect()

    def test_artifact_without_trigger_evidence_refuses_by_name(self, tmp_path):
        """A declared trigger the artifact carries no evidence for refuses by
        SOURCE, naming the missing extensions, instead of planning the
        assigned population at trigger_rate=1.0."""
        con, path, analysis = _triggered_analysis(tmp_path)
        try:
            reopened = _republish(con, path, analysis, [])
            with pytest.raises(CapabilityError) as raised:
                reopened.planning_baseline("revenue")
            assert raised.value.code == "analysis.planning_baseline.trigger_evidence_unavailable"
            assert raised.value.context["trigger"] == "exposure"
            assert raised.value.context["missing"] == ("trigger_population", "assignment_counts")
        finally:
            con.disconnect()

    def test_untriggered_metric_keeps_the_default_trigger_rate(self, tmp_path):
        """No trigger declared -> trigger_rate stays at Baseline's default
        of 1.0, even though this metric's own analyzed count falls short of
        enrollment because of missing rows unrelated to any trigger."""
        con = ibis.duckdb.connect()
        n_per_arm = 175
        rows = {
            "unit_id": [],
            "group_id": [],
            "ts": [],
            "event": [],
            "value": [],
            "experiment_id": [],
        }

        def add(uid, arm, event, value, ts):
            rows["unit_id"].append(uid)
            rows["group_id"].append(arm)
            rows["ts"].append(ts)
            rows["event"].append(event)
            rows["value"].append(value)
            rows["experiment_id"].append("exp")

        for arm in ("control", "treatment"):
            for i in range(n_per_arm):
                uid = f"{arm}{i}"
                add(uid, arm, "enrolled", None, pd.Timestamp("2024-01-01"))
                if i >= 25:  # the first 25 enrolled units per arm never fire the fact
                    add(uid, arm, "revenue", 10.0 + (i % 7), pd.Timestamp("2024-01-02"))
                # Anchors the observable data extent well past the metric's
                # own window so the window has fully matured for every
                # enrolled unit; this event itself falls outside the
                # window and is never summed.
                add(uid, arm, "revenue", 0.0, pd.Timestamp("2024-01-20"))
        con.create_table("events", obj=pd.DataFrame(rows))
        defs = """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [unit_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: revenue
    type: mean
    entity: unit_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    unit: unit_id
    start: "2024-01-01T00:00:00"
    control_group: control
    plan:
      primary: revenue
"""
        p = tmp_path / "defs_untriggered.yml"
        p.write_text(defs)
        analysis = Analysis.from_definitions("exp", str(p), con)
        baseline = _plan(analysis, "revenue")
        assert baseline.trigger_rate == 1.0


_CUPED = ({"name": "cuped", "variance_reduction": "cuped"},)


def _parity_rows(*, clustered):
    """The parity harness's event shape with a covariate correlated with the
    outcome, varying ratio denominators and, when clustered, single-arm
    stores of unequal size carrying a store effect."""
    import datetime as dt

    from tests.parity_harness import dataset as ds

    rows = []
    for arm, prefix in (("control", "c"), ("treatment", "t")):
        for i in range(60):
            uid = f"{prefix}{i}"
            store = f"{prefix}{i % 13}" if clustered else f"s{i % 3}"
            store_effect = 3.0 * (i % 13) if clustered else 0.0
            rows.append(
                ds._row(
                    uid,
                    ds._EXPOSURE_AT,
                    "exposure",
                    group_id=arm,
                    experiment_id="exp",
                    store_id=store,
                )
            )
            rows.append(
                ds._row(
                    uid,
                    ds._PRE_PURCHASE_AT,
                    "purchase",
                    store_id=store,
                    revenue=2.0 + i % 5 + i % 2,
                )
            )
            if i % 3 != 0:
                revenue = 5.0 + (arm == "treatment") + i % 5 + store_effect
                rows.append(
                    ds._row(uid, ds._PURCHASE_AT, "purchase", store_id=store, revenue=revenue)
                )
            rows.append(ds._row(uid, ds._SESSION_AT, "session_end", store_id=store, sess=1))
            if i % 2:
                later = ds._SESSION_AT + dt.timedelta(minutes=5)
                rows.append(ds._row(uid, later, "session_end", store_id=store, sess=1))
            rows.append(ds._row(uid, ds._FRESHNESS_PAD_AT, "purchase", store_id=store, revenue=0.0))
            rows.append(ds._row(uid, ds._FRESHNESS_PAD_AT, "session_end", store_id=store, sess=0))
    return rows


def _parity_metrics(*, covariate):
    from increment.frame import MetricSpec

    metrics = [
        MetricSpec(name="revenue", type="mean", missing="zero"),
        MetricSpec(name="purchase_rate", type="conversion", value_column="converted"),
        MetricSpec(
            name="rps", type="ratio", numerator="revenue", denominator="sessions", missing="zero"
        ),
    ]
    if covariate:
        metrics += [
            MetricSpec(
                name="revenue_cuped",
                type="mean",
                value_column="revenue",
                covariate="pre_revenue",
                missing="zero",
                sensitivity_methods=_CUPED,
            ),
            MetricSpec(
                name="rps_cuped",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
                covariate="pre_revenue",
                missing="zero",
                sensitivity_methods=_CUPED,
            ),
        ]
    return metrics


def _parity_analyses(*, clustered):
    """One dataset through every arm constructor that can read it:
    ``(analyses, connections)``. A clustered pilot has no unit-panel route
    (``from_unit_panel`` takes no cluster) and no moments export (clustered
    moments transport refuses by SOURCE)."""
    from increment.frame import MetricSpec
    from increment.semantics.models import AnalysisPlan, Definitions, ExperimentMetric
    from tests.analysis_factory import make_analysis
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import _export_and_replay, _publish_and_adopt

    rows = _parity_rows(clustered=clustered)
    secondaries: list[str | ExperimentMetric] = ["revenue", "purchase_rate", "rps"]
    if not clustered:
        secondaries += [
            ExperimentMetric(metric=name, sensitivity_methods=_CUPED)
            for name in ("revenue_cuped", "rps_cuped")
        ]
    defs = ds.definitions_dict(plan=AnalysisPlan(secondaries=tuple(secondaries)))
    rps = next(m for m in defs["metrics"] if m["name"] == "rps")
    defs["metrics"].append({**rps, "name": "rps_cuped"})
    if clustered:
        defs["experiments"][0]["cluster"] = "store"
    connections = [ds.duckdb_connection(rows), ds.duckdb_connection(rows)]
    native, published = (
        make_analysis(con, Definitions.model_validate(defs), experiment="exp")
        for con in connections
    )
    frame = ds.unit_summary_frame(connections[0])
    metrics = _parity_metrics(covariate=not clustered)
    plan = AnalysisPlan(secondaries=tuple(m.name for m in metrics))

    def summary(cluster=None):
        return Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=metrics,
            plan=plan,
            cluster=cluster,
        )

    analyses = {
        "from_definitions": native,
        "from_unit_day_artifact": _publish_and_adopt(connections[1], published),
        "from_unit_summary": summary("store" if clustered else None),
    }
    if clustered:
        # The same rows without the cluster: the per-unit oracle for mean/var.
        analyses["unclustered"] = summary()
        return analyses, connections
    analyses["from_unit_panel"] = Analysis.from_unit_panel(
        ds.unit_panel_frame(frame),
        unit="user_id",
        group="variant",
        date="date",
        control="control",
        metrics=metrics,
        plan=plan,
    )
    analyses["from_moments"] = _export_and_replay(
        summary(),
        [
            MetricSpec(
                name=m.name,
                type=m.type,
                numerator=m.numerator,
                denominator=m.denominator,
                covariate=m.covariate,
                sensitivity_methods=m.sensitivity_methods,
            )
            for m in metrics
        ],
    )
    return analyses, connections


@pytest.mark.slow
class TestPlanningBaselineParity:
    """Identical per-unit data through every arm constructor yields the same
    Baseline, field for field, as the from_unit_summary oracle."""

    def test_every_arm_constructor_matches_the_unit_summary_oracle(self):
        analyses, connections = _parity_analyses(clustered=False)
        try:
            oracle = analyses["from_unit_summary"]
            for metric in ("revenue", "purchase_rate", "rps", "revenue_cuped", "rps_cuped"):
                expected = _plan(oracle, metric).model_dump()
                for name, analysis in analyses.items():
                    actual = _plan(analysis, metric).model_dump()
                    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (name, metric)
            assert _plan(oracle, "revenue_cuped").cuped_rho > 0.2
            assert _plan(oracle, "rps_cuped").cuped_rho > 0.05
        finally:
            for con in connections:
                con.disconnect()

    def test_clustered_constructors_match_the_unit_summary_oracle(self):
        analyses, connections = _parity_analyses(clustered=True)
        try:
            oracle = analyses["from_unit_summary"]
            for metric in ("revenue", "purchase_rate", "rps"):
                expected = _plan(oracle, metric)
                for name in ("from_definitions", "from_unit_day_artifact"):
                    actual = _plan(analyses[name], metric).model_dump()
                    assert actual == pytest.approx(expected.model_dump(), rel=1e-9), (name, metric)
                unit_level = _plan(analyses["unclustered"], metric)
                assert (expected.mean, expected.var) == pytest.approx(
                    (unit_level.mean, unit_level.var), rel=1e-12
                )
                assert expected.avg_cluster_size == pytest.approx(60 / 13, rel=1e-12)
            assert _plan(oracle, "revenue").cluster_icc > 0.1
        finally:
            for con in connections:
                con.disconnect()

    def test_encouragement_compliance_matches_across_constructors(self):
        from tests.parity_harness.cases import PARITY_CASES

        case = next(c for c in PARITY_CASES if c.id == "encouragement_declared_definitions")
        dumped = {}
        for name, build in case.build.items():
            analysis = build()
            try:
                dumped[name] = _plan(analysis, "revenue").model_dump()
            finally:
                for con in getattr(analysis, "_parity_connections", ()):
                    con.disconnect()
        oracle = dumped["from_unit_summary"]
        assert 0.0 < oracle["compliance"] < 1.0
        for name, actual in dumped.items():
            assert actual == pytest.approx(oracle, rel=1e-9), name


class TestPlanningBaselineOnSwitchback:
    @staticmethod
    def _panel():
        import polars as pl

        rng = np.random.default_rng(77411)
        rows = []
        for unit in range(12):
            orders = rng.random(5) < 0.75
            unit_trend = rng.normal()
            for cycle in range(5):
                for period in range(2):
                    treated = (period == 1) == bool(orders[cycle])
                    for step in range(5):
                        value = 40 + period * (0.7 + unit_trend) + rng.normal()
                        if step >= 2 and treated:
                            value += 0.5
                        rows.append(
                            {
                                "unit": str(unit),
                                "cycle": cycle,
                                "period": period,
                                "step": step,
                                "group": "treatment" if treated else "control",
                                "orders": value,
                            }
                        )
        return pl.DataFrame(rows)

    def test_delegates_to_the_switchback_source_and_feeds_its_planner(self):
        """Analysis.from_switchback_panel's planning_baseline is the source's
        own pilot-fitted SwitchbackBaseline, which the switchback planner
        consumes directly."""
        from increment.frame import from_switchback_panel
        from increment.power import switchback_required_blocks_or_units
        from increment.power.switchback import SwitchbackBaseline
        from increment.semantics.assignment import (
            IndependentBernoulliOrder,
            SwitchbackAssignment,
            SwitchbackWindow,
        )
        from increment.semantics.design import Randomized

        identification = Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        )
        assignment = SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.75),
            window=SwitchbackWindow(washout_steps=1, observation_steps=4, carryover_order=1),
        )
        panel = self._panel()
        source = from_switchback_panel(
            panel,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"orders": "mean"},
            identification=identification,
            assignment=assignment,
        )
        baseline = Analysis.from_switchback_panel(
            panel,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"orders": "mean"},
            identification=identification,
            assignment=assignment,
        ).planning_baseline("orders")
        assert isinstance(baseline, SwitchbackBaseline)
        assert baseline == source.planning_baseline(source.metrics[0])
        result = switchback_required_blocks_or_units(
            2.0, baseline, source.context.procedures["orders"], target_power=0.8
        )
        assert result.n is not None and result.n >= 2
        assert result.power is not None and result.power >= 0.8


class TestPlanningBaselineOnMoments:
    """from_moments declares no trigger and carries no per-unit rows."""

    def test_from_moments_keeps_the_default_trigger_rate(self, tmp_path):
        """A from_moments source declares no trigger, so trigger_rate stays at
        its Baseline default rather than raising or misreading one."""
        direct = Analysis.from_unit_summary(
            _unit_summary_frame(),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "mean"}],
        )
        direct.export(tmp_path / "moments.parquet")
        rows = pq.read_table(tmp_path / "moments.parquet").to_pylist()
        adopted = Analysis.from_moments(
            rows, metrics=[{"name": "revenue", "type": "mean"}], control="control"
        )
        baseline = _plan(adopted, "revenue")  # must not raise
        assert baseline.trigger_rate == 1.0

    def test_from_moments_refuses_a_quantile_metric_by_source(self, tmp_path):
        direct = Analysis.from_unit_summary(
            _unit_summary_frame(),
            unit="unit_id",
            group="group_id",
            control="control",
            metrics=[{"name": "revenue", "type": "mean"}],
        )
        direct.export(tmp_path / "moments.parquet")
        rows = pq.read_table(tmp_path / "moments.parquet").to_pylist()
        adopted = Analysis.from_moments(
            rows,
            metrics=[{"name": "revenue", "type": "quantile", "quantile": 0.5}],
            control="control",
        )
        with pytest.raises(CapabilityError) as raised:
            adopted.planning_baseline("revenue")
        assert raised.value.code == "analysis.planning_baseline.quantile_source_unavailable"
        assert raised.value.context["metric"] == "revenue"
