"""Analysis.planning_baseline computes every derivable Baseline field
from the analysis's own data and declared design, on every constructor
that can supply it."""

import ibis
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import Analysis
from increment.errors import CapabilityError, CodedError
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
    def test_degenerate_covariate_refuses_with_the_runtime_cuped_code(self, metric):
        """A covariate that is present but constant is a data hazard, not a
        source that lacks the covariate: planning refuses with the same code
        the runtime's CUPED fit raises on the same data."""
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
        with pytest.raises(CodedError) as runtime:
            analysis.run()
        assert (
            planned.value.code == runtime.value.code == "estimation.cuped.covariate_zero_variance"
        )

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
            if member < analyzed_sizes[cluster] if clustered else i % 5 == 0:
                add(uid, arm, "exposure", None, pd.Timestamp("2024-01-01"), store_id)
            value = (
                20 + 10 * cluster + (-4 if member % 2 == 0 else 4)
                if clustered
                else _triggered_value(i)
            )
            add(uid, arm, "revenue", value, pd.Timestamp("2024-01-02"), store_id)
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
    return con, path, Analysis.from_definitions("exp", str(path), con)


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
    from fractions import Fraction

    from scipy.stats import chi2

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
        TriggerPopulationRequest,
    )

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
            results = {row.analysis_population: row for row in source.run(metrics=["revenue"])}
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
        TriggerPopulationRequest,
    )

    con, path, analysis = _triggered_analysis(
        tmp_path,
        clustered=True,
        assigned_sizes=[10] * 10,
        analyzed_sizes=[2] * 10,
        metric_aggregation=metric_aggregation,
    )
    reopened = None
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
        reopened = _republish(
            con,
            path,
            analysis,
            [
                TriggerPopulationRequest(trigger_name="exposure"),
                AssignmentCountsRequest(populations=("assigned", "triggered")),
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
        from increment.semantics.artifact import AssignmentCountsRequest, TriggerPopulationRequest

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
                ],
            )
            assert _plan(reopened, "revenue").model_dump() == pytest.approx(
                native.model_dump(), rel=1e-12
            )
        finally:
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
