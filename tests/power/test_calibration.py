"""Power calibration: empirical power from simulation matches the analytic
model (required_sample_size using se_log_mean), verified by chaining
through the full increment pipeline. Marked
``@pytest.mark.parameter_recovery`` - excluded from the fast suite.
"""

from __future__ import annotations

import ibis
import pyarrow.compute as pc
import pytest

from increment.estimation.armstats import SummaryStats
from increment.power.core import Baseline, PowerDesign, required_sample_size
from increment.semantics.models import MeanMetric, Metric
from increment.simulate.dgp import Scenario, simulate_raw_logs
from increment.simulate.runner import _run_one_replication
from tests.power._procedures import make_procedure

# Helpers


def _control_summary_from_sim(scenario: Scenario) -> SummaryStats:
    """Extract control-arm SummaryStats for the ``count`` metric.

    Includes ALL control-arm units, even zero-event ones - the DGP only
    emits event rows for units with >0 events, so a naive group-by misses
    zeros.
    """
    import numpy as np

    raw = simulate_raw_logs(scenario)

    # Identify control-arm units from exposure events
    exposures = raw.filter(pc.equal(raw.column("event"), "exposure"))  # ty: ignore[unresolved-attribute]  - pyarrow.compute funcs are dynamically generated, no static stub coverage
    is_control = pc.equal(exposures.column("group_id"), "control")  # ty: ignore[unresolved-attribute]  - pyarrow.compute funcs are dynamically generated, no static stub coverage
    control_ids: set[str] = set(exposures.filter(is_control).column("unit_id").to_pylist())

    # Get visit-event values per unit (units with zero events join as 0)
    visits = raw.filter(pc.equal(raw.column("event"), "visit"))  # ty: ignore[unresolved-attribute]  - pyarrow.compute funcs are dynamically generated, no static stub coverage
    visit_df = visits.to_pandas()
    has_visits = visit_df[visit_df["unit_id"].isin(control_ids)].groupby("unit_id")["value"].sum()
    per_unit = np.array([has_visits.get(uid, 0.0) for uid in control_ids])

    n = len(per_unit)
    mean = float(per_unit.mean())
    var = float(per_unit.var(ddof=1))
    return SummaryStats(n=n, mean=mean, var=var)


# Fast smoke variant (AGENTS.md): keeps regressions in the extraction
# helper caught by the default suite, not only the slow Monte Carlo run.


def test_control_summary_from_sim_includes_zero_event_units():
    """``_control_summary_from_sim`` must count control units with zero events.

    Regression test: a naive group-by over the visit-events table silently
    drops units that never emitted a visit, biasing mean and variance
    downward.
    """
    scenario = Scenario(n_units=200, n_days=7, true_lift={"count": 0.0}, seed=7)
    summary = _control_summary_from_sim(scenario)

    raw = simulate_raw_logs(scenario)
    exposures = raw.filter(pc.equal(raw.column("event"), "exposure"))  # ty: ignore[unresolved-attribute]  - pyarrow.compute funcs are dynamically generated, no static stub coverage
    is_control = pc.equal(exposures.column("group_id"), "control")  # ty: ignore[unresolved-attribute]  - pyarrow.compute funcs are dynamically generated, no static stub coverage
    n_control_units = len(set(exposures.filter(is_control).column("unit_id").to_pylist()))

    assert summary.n == n_control_units, (
        f"Expected all {n_control_units} control units counted (including zero-visit "
        f"units), got n={summary.n}"
    )
    assert summary.mean >= 0
    assert summary.var >= 0


# Calibration test


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCalibration:
    """Empirical detection rate matches the analytic power calculation.

    The count DGP is Negative Binomial (``var = mu + mu^2 / phi``) with a
    MULTIPLICATIVE lift, so its treatment-arm variance grows faster than
    the planning model's equal-absolute-variance assumption. The pipeline
    is calibrated against the DGP-aware power at the planned N, and the
    planning assumption's optimism for this shape is bounded explicitly
    rather than hidden inside the Monte Carlo tolerance.
    """

    # ------------------------------------------------------------------
    _TARGET_LIFT = 0.10  # 10% relative lift
    _TARGET_POWER = 0.80  # 80% analytic power
    _PILOT_UNITS = 50_000  # units in the pilot that estimates the baseline
    _N_DAYS = 14
    # 200 replications put the 0.07 tolerance at 2.3 Monte Carlo standard
    # errors around the ~0.77 DGP-aware power; 100 left it at 1.7.
    _N_REPS = 200
    _TOLERANCE = 0.07  # |empirical - analytic| <= 0.07 per the plan's spec
    _ASSUMPTION_GAP = 0.05  # planning power minus DGP-aware power, at most
    _PILOT_SEED = 42

    @staticmethod
    def _dgp_aware_power(n_t: int, n_c: int, baseline: Baseline, lift: float) -> float:
        """Two-sided power with the treatment variance scaled the way a
        Negative Binomial's does under a multiplicative lift: ``mu + mu^2 /
        phi`` with ``1 / phi`` read off the pilot's excess variance."""
        import math

        from scipy.stats import norm

        mu_c, v_c = baseline.mean, baseline.effective_var
        mu_t = mu_c * (1.0 + lift)
        v_t = mu_t + (v_c - mu_c) * (mu_t / mu_c) ** 2
        se2 = v_t / (n_t * mu_t**2) + v_c / (n_c * mu_c**2)
        nc = math.log1p(lift) / math.sqrt(se2)
        z = float(norm.isf(0.05 / 2.0))
        return float(norm.sf(z - nc) + norm.sf(z + nc))

    def test_power_calibration(self):
        """Empirical detection rate approx the DGP-aware analytic power (+/-0.07)."""
        # --------------------------------------------------------------
        pilot = Scenario(
            n_units=self._PILOT_UNITS,
            n_days=self._N_DAYS,
            true_lift={"count": 0.0},
            unit_heterogeneity=0.0,
            seed=self._PILOT_SEED,
        )

        pilot_summary = _control_summary_from_sim(pilot)
        baseline = Baseline.from_summary(pilot_summary)

        # 2. Compute required sample size for target lift + power
        procedure = make_procedure()
        design = PowerDesign(power=self._TARGET_POWER)
        power_result = required_sample_size(
            procedure=procedure, relative_lift=self._TARGET_LIFT, baseline=baseline, design=design
        )
        req_n_total = power_result.n_total
        n_t = power_result.n_per_arm
        analytic_power = self._dgp_aware_power(n_t, req_n_total - n_t, baseline, self._TARGET_LIFT)
        assert 0.0 <= power_result.power - analytic_power <= self._ASSUMPTION_GAP, (
            f"planning power {power_result.power:.3f} vs DGP-aware {analytic_power:.3f}: the "
            "equal-absolute-variance assumption's optimism for a Negative Binomial count "
            f"exceeds {self._ASSUMPTION_GAP}"
        )

        # 3. Run replications and detect the lift
        # The scenario declares only the count metric, whose per-unit events
        # the DGP emits as "visit" facts.
        metrics: list[Metric] = [
            MeanMetric(name="count", entity="user", fact="visit", aggregation="count")
        ]
        detections = 0

        for rep in range(self._N_REPS):
            rep_scenario = pilot.model_copy(
                update={
                    "n_units": req_n_total,
                    "true_lift": {"count": self._TARGET_LIFT},
                    "seed": self._PILOT_SEED + rep * 9973 + 100,
                }
            )

            con = ibis.duckdb.connect()
            try:
                res = _run_one_replication(rep_scenario, con, metrics)
            finally:
                con.disconnect()

            # Detection: CI excludes 0 (lower bound > 0 for positive lift)
            for est in res["metric_estimates"]:
                if est.metric == "count":
                    if est.require_lift().lb is not None and est.require_lift().lb > 0:
                        detections += 1
                    break

        # 4. Compare
        empirical_power = detections / self._N_REPS
        lower = analytic_power - self._TOLERANCE
        upper = analytic_power + self._TOLERANCE

        assert lower <= empirical_power <= upper, (
            f"Empirical power {empirical_power:.3f} ({detections}/{self._N_REPS}) "
            f"outside [{lower:.3f}, {upper:.3f}] for {baseline.mean=:.3f}, "
            f"{baseline.var=:.3f}, {req_n_total=}"
        )


# Non-inferiority calibration: at true lift 0, draws the log-ratio estimator
# from its assumed Normal distribution and runs the real infer_lift decision path.


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestNonInferiorityCalibration:
    """Empirical guardrail-pass rate at true lift 0 matches target power."""

    _TARGET_POWER = 0.80
    _NULL_LIFT = -0.01
    _N_REPS = 2000
    _TOLERANCE = 0.07
    _SEED = 123

    def test_ni_power_calibration_at_true_lift_zero(self):
        import math as _math

        import numpy as np

        from increment.estimation.inference import infer_lift

        baseline = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="greater", null_lift=self._NULL_LIFT)
        design = PowerDesign(power=self._TARGET_POWER)
        n_res = required_sample_size(
            procedure=procedure, relative_lift=0.0, baseline=baseline, design=design
        )
        n_t = n_res.n_per_arm
        n_c = n_res.n_total - n_t
        se2 = baseline.effective_var / (baseline.mean**2) * (1.0 / n_t + 1.0 / n_c)
        se = _math.sqrt(se2)
        half_se = se / _math.sqrt(2.0)  # split combined SE evenly across the two arms

        rng = np.random.default_rng(self._SEED)
        log_rr_samples = rng.normal(loc=0.0, scale=se, size=self._N_REPS)  # true effect: theta=0

        passes = 0
        for log_rr in log_rr_samples:
            result = infer_lift(
                metric="m",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                log_rr=float(log_rr) - 0.0,
                se_t=half_se,
                se_c=half_se,
                alpha=procedure.compiled_alpha,
                alternative="greater",
                null_lift=self._NULL_LIFT,
            )
            lift = result.require_lift()
            if lift.lb is not None and lift.lb > self._NULL_LIFT:
                passes += 1

        empirical_power = passes / self._N_REPS
        lower = self._TARGET_POWER - self._TOLERANCE
        upper = self._TARGET_POWER + self._TOLERANCE
        assert lower <= empirical_power <= upper, (
            f"Empirical guardrail-pass rate {empirical_power:.3f} ({passes}/{self._N_REPS}) "
            f"outside [{lower:.3f}, {upper:.3f}] for n_per_arm={n_t}"
        )


# Type-I error at the null boundary: true effect fixed exactly at the margin,
# rejection rate must be ~alpha, not inflated (complements the power check above).


def _guardrail_rejection_rate(*, n_reps: int, seed: int) -> float:
    """Simulate the log-ratio estimator with its TRUE effect fixed exactly at
    the declared null (theta0=log1p(null_lift)) and return the fraction of
    replications rejecting H0 - the empirical Type-I rate at the boundary."""
    import math

    import numpy as np

    from increment.estimation.inference import infer_lift

    null_lift = -0.01
    baseline = Baseline(mean=1.0, var=2.0)
    procedure = make_procedure(alpha=0.05, alternative="greater", null_lift=null_lift)
    design = PowerDesign(power=0.8)
    # Same N a real guardrail would be sized with - large enough that the
    # asymptotic normal approximation infer_lift's decision relies on holds.
    n_res = required_sample_size(
        procedure=procedure, relative_lift=0.0, baseline=baseline, design=design
    )
    n_t = n_res.n_per_arm
    n_c = n_res.n_total - n_t
    se2 = baseline.effective_var / (baseline.mean**2) * (1.0 / n_t + 1.0 / n_c)
    se = math.sqrt(se2)
    half_se = se / math.sqrt(2.0)

    theta0 = math.log1p(null_lift)  # the TRUE effect, fixed at the null boundary
    rng = np.random.default_rng(seed)
    log_rr_samples = rng.normal(loc=theta0, scale=se, size=n_reps)

    rejections = 0
    for log_rr in log_rr_samples:
        result = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=float(log_rr) - 0.0,
            se_t=half_se,
            se_c=half_se,
            alpha=procedure.compiled_alpha,
            alternative="greater",
            null_lift=null_lift,
        )
        lift = result.require_lift()
        if lift.lb is not None and lift.lb > null_lift:
            rejections += 1
    return rejections / n_reps


def test_guardrail_type_i_rate_smoke():
    """Fast, small-N smoke variant: rejection rate at the null boundary is in
    the right ballpark (loose tolerance) - catches a broken decision rule
    (e.g. comparing against 0 instead of null_lift) without the full Monte Carlo."""
    rate = _guardrail_rejection_rate(n_reps=200, seed=7)
    assert 0.0 < rate < 0.20, f"Type-I rate {rate:.3f} wildly off alpha=0.05 (200-rep smoke)"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_guardrail_type_i_rate_at_the_null_boundary():
    """Empirical Type-I rate at the true effect fixed exactly at the
    declared margin (-1%) matches alpha=0.05 within the file's tolerance."""
    alpha = 0.05
    tolerance = 0.03
    rate = _guardrail_rejection_rate(n_reps=5000, seed=123)
    lower, upper = alpha - tolerance, alpha + tolerance
    assert lower <= rate <= upper, (
        f"Empirical Type-I rate {rate:.3f} outside [{lower:.3f}, {upper:.3f}] for alpha={alpha}"
    )


# Alternative-arm variance: the treatment arm's log-scale variance is
# evaluated at ITS OWN mean, not the control's.


def _true_log_ratio_power(n_per_arm: int, mean_c: float, var: float, lift: float) -> float:
    """Delta-method power of the log-ratio test the readout actually runs:
    each arm's variance is evaluated at ITS OWN mean."""
    import math

    from scipy.stats import norm

    mean_t = mean_c * (1.0 + lift)
    se2 = var / (n_per_arm * mean_t**2) + var / (n_per_arm * mean_c**2)
    nc = math.log1p(lift) / math.sqrt(se2)
    z = float(norm.isf(0.05 / 2.0))
    return float(norm.sf(z - nc) + norm.sf(z + nc))


def test_power_uses_the_treatment_arm_mean_under_the_alternative() -> None:
    """A relative DECREASE shrinks the treatment mean, which INFLATES that
    arm's log-scale variance. Evaluating both arms at the control mean
    over-states power and under-sizes the experiment."""
    from increment.decision import FixedInference
    from increment.estimation.arm_contract import (
        AnalysisAxes,
        ArmPlanningProcedure,
        FamilyPolicy,
        MetricCapabilities,
        PlanningFamilyExpansion,
        RelativeDecisionPolicy,
    )
    from increment.power import Baseline, achieved_power, required_sample_size
    from increment.semantics.assignment import ParallelAssignment
    from increment.semantics.models import MethodSpec

    procedure = ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=True,
            population="assigned",
            variance_adjustment="none",
        ),
        dependence="iid",
        inference=FixedInference(),
        estimand="mean",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ),
        family_expansion=PlanningFamilyExpansion(family_size=1),
        decision_method=MethodSpec(name="unadjusted", variance_reduction="none"),
        sensitivity_methods=(),
        prior_present=False,
    )
    baseline = Baseline(mean=1.0, var=1.0)
    lift = -0.20

    expected = _true_log_ratio_power(300, 1.0, 1.0, lift)
    reported = achieved_power(300, lift, baseline, procedure).power
    assert reported == pytest.approx(expected, rel=1e-3)

    sized = required_sample_size(lift, baseline, procedure)
    assert _true_log_ratio_power(sized.n_per_arm, 1.0, 1.0, lift) >= 0.80 - 1e-3
