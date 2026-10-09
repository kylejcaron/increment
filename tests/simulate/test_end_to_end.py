"""End-to-end ground-truth evaluation tests for the full increment pipeline.

Step 2 (fast smoke): validates the query + estimation flow runs on DuckDB
and produces sensible output.

Step 3 (``@pytest.mark.parameter_recovery`` calibration): 200 replications
of a zero-lift scenario check unconditional CI coverage clears an exact
one-sided Clopper-Pearson lower bound (``tests.mc.coverage_lower_bound``),
SRM detection is judged against the *configured* ``assignment_ratio``, and
nonzero-lift bias is near zero (the DGP injects lift on the same
relative-mean scale the estimator reports).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from increment.errors import InvalidRequestError
from increment.simulate.dgp import Scenario
from increment.simulate.runner import EvalResult, _KeyOutcome, _reduce_key, run_end_to_end

# Step 2: fast end-to-end smoke (always runs)


class TestSmoke:
    """One replication - validates the pipeline runs end-to-end. 2000 units
    keeps every metric's log-scale SE inside the delta-method guard's
    validity region (500 units under-samples revenue purchasers)."""

    def test_smoke_zero_lift(self):
        """Pipeline produces coherent attempted/estimable counts and
        finite nullable aggregates per metric, no NaNs."""
        scenario = Scenario(
            n_units=2000,
            n_days=14,
            true_lift={"conversion": 0.0, "count": 0.0, "revenue": 0.0},
            seed=42,
        )
        result = run_end_to_end(scenario, replications=1)

        for metric in ("conversion", "count", "revenue"):
            assert result.attempted[metric] == 1
            assert result.attempted[metric] == (
                result.point_estimable[metric] + result.excluded[metric] + result.failed[metric]
            )
            assert result.interval_estimable[metric] <= result.point_estimable[metric]

            bias = result.bias[metric]
            assert bias is not None, f"bias is None for {metric}"
            assert bias == bias, f"bias is NaN for {metric}"  # NaN != NaN
            assert abs(bias) < float("inf"), f"bias is inf for {metric}"

            # Single replication: unconditional coverage is 0.0 or 1.0 when defined.
            cov = result.coverage_unconditional[metric]
            assert cov in (0.0, 1.0), f"unexpected coverage {cov} for {metric}"

    def test_metric_estimates_carry_a_declared_preferred_direction(self):
        """A metric declaring preferred_direction="decrease" must reach its
        LiftEstimate row so prob_favorable()/chance_to_beat_favorable() work;
        a metric that never declares one still reports None, unchanged."""
        import ibis

        from increment.semantics.models import ConversionMetric, MeanMetric, Metric
        from increment.simulate.runner import _run_one_replication

        metrics: list[Metric] = [
            ConversionMetric(
                name="conversion",
                entity="user",
                fact="conversion",
                preferred_direction="decrease",
            ),
            MeanMetric(name="count", entity="user", fact="visit", aggregation="count"),
        ]
        scenario = Scenario(
            n_units=500,
            n_days=14,
            true_lift={"conversion": 0.0, "count": 0.0},
            seed=42,
        )
        con = ibis.duckdb.connect()
        try:
            result = _run_one_replication(scenario, con, metrics)
        finally:
            con.disconnect()

        by_metric = {e.metric: e for e in result["metric_estimates"]}
        assert by_metric["conversion"].preferred_direction == "decrease"
        assert by_metric["count"].preferred_direction is None

    def test_smoke_srm_not_detected_fair(self):
        """With fair assignment, the 0.1% lifetime SRM check should not flag.

        Requires a MAJORITY (>=2 of 3) independent seeds to flag, not a
        single replication; the anytime-valid default's 0.1% lifetime
        false-positive bound applies to cumulative-prefix monitoring, and
        these seeds are only a smoke guard, not a calibration claim.
        """
        flags = 0
        for seed in (99, 100, 101):
            scenario = Scenario(
                n_units=1000,
                n_days=7,
                true_lift={"conversion": 0.0},
                assignment_ratio=0.5,
                seed=seed,
            )
            result = run_end_to_end(scenario, replications=1)
            assert result.srm_rate is not None
            flags += int(result.srm_rate)
        assert flags < 2, f"SRM falsely detected in {flags}/3 seeds for fair assignment"

    def test_smoke_srm_respects_configured_ratio(self):
        """A deliberately unequal design is not a sample-ratio mismatch.

        The runner judges observed counts against the *configured*
        ``assignment_ratio``, so a 60/40 design delivering ~60/40 must not
        be flagged (genuine mismatch detection is covered by the
        ``sample_ratio_mismatch`` unit tests). Same majority-of-3-seeds
        smoke guard as the fair-assignment test.
        """
        flags = 0
        for seed in (99, 100, 101):
            scenario = Scenario(
                n_units=1000,
                n_days=7,
                true_lift={"conversion": 0.0},
                assignment_ratio=0.6,
                seed=seed,
            )
            result = run_end_to_end(scenario, replications=1)
            assert result.srm_rate is not None
            flags += int(result.srm_rate)
        assert flags < 2, f"SRM falsely detected in {flags}/3 seeds for a configured 60/40 design"

    def test_smoke_srm_detects_realized_allocation_mismatch(self):
        """A strong planned-vs-realized mismatch is detected end-to-end."""
        scenario = Scenario(
            n_units=2000,
            n_days=7,
            true_lift={"conversion": 0.0},
            assignment_ratio=0.5,
            realized_assignment_ratio=0.8,
            seed=123,
        )
        result = run_end_to_end(scenario, replications=3)

        assert result.srm_rate is not None
        assert result.srm_rate >= 2 / 3, (
            f"SRM mismatch detected in only {result.srm_rate:.0%} of replications"
        )


# Step 2b: runner-level replications refusal + CLI


class TestReplicationsValidation:
    """Both runners reject Boolean, noninteger, and nonpositive replication
    counts before ``SeedSequence``/data generation -- the accepted baseline
    instead returned a bogus empty ``EvalResult`` for 0 replications and
    leaked a raw numpy ``ValueError`` for a negative count."""

    @pytest.mark.parametrize("bad", [0, -5, 1.5, True, False])
    def test_refuses_before_generating_any_data(self, bad, monkeypatch):
        import increment.simulate.runner as runner_module

        scenario = Scenario(n_units=10, n_days=3, true_lift={"conversion": 0.0}, seed=0)
        called = {"n": 0}

        def fail_if_called(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("simulate_raw_logs must not run before the replications guard")

        monkeypatch.setattr(runner_module, "simulate_raw_logs", fail_if_called)
        with pytest.raises(InvalidRequestError) as exc_info:
            run_end_to_end(scenario, replications=bad)
        assert exc_info.value.code == "simulate.runner.replications_invalid"
        assert called["n"] == 0

    def test_cli_exits_nonzero_on_zero_replications(self, monkeypatch, capsys):
        """Exercises the real CLI entry point (``main()``, argparse and
        all): the accepted baseline printed a successful empty run for
        ``--replications 0``; the fix must fail coded before running."""
        from increment.simulate.runner import main

        monkeypatch.setattr("sys.argv", ["prog", "--replications", "0"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
        assert "simulate.runner.replications_invalid" in capsys.readouterr().err

    def test_cli_exits_nonzero_on_negative_replications(self, monkeypatch, capsys):
        from increment.simulate.runner import main

        monkeypatch.setattr("sys.argv", ["prog", "--replications", "-5"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
        assert "simulate.runner.replications_invalid" in capsys.readouterr().err

    def test_cli_valid_run_prints_available_and_null_count_fields(self, monkeypatch, capsys):
        """One valid small CLI run: available counts are real integers,
        an undefined aggregate serializes as JSON ``null``."""
        from increment.simulate.runner import main

        monkeypatch.setattr("sys.argv", ["prog", "--replications", "1"])
        main()
        payload = json.loads(capsys.readouterr().out)
        for metric in ("conversion", "count", "revenue"):
            assert payload["attempted"][metric] == 1
            assert isinstance(payload["bias"][metric], float)
        # A single replication has < 2 observations: bias_mcse is null.
        assert payload["bias_mcse"]["conversion"] is None


def test_whole_run_failure_counts_as_failed_for_every_attempted_key_including_breakout(
    monkeypatch,
):
    """A coded refusal inside one replication's own pipeline is caught
    and counted as `failed` for every attempted core AND breakout key --
    not only core metrics, and not silently dropped from the breakout
    denominator -- rather than aborting the entire run."""
    import increment.simulate.runner as runner_module

    calls = {"n": 0}
    real = runner_module._run_one_replication

    def flaky(scenario, ibis_conn, metrics):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise InvalidRequestError("boom", code="test.runner.boom", context={})
        return real(scenario, ibis_conn, metrics)

    monkeypatch.setattr(runner_module, "_run_one_replication", flaky)
    scenario = Scenario(
        n_units=500,
        n_days=7,
        true_lift={"conversion": 0.0},
        n_segments=2,
        seed=5,
    )
    result = run_end_to_end(scenario, replications=4)

    assert result.attempted["conversion"] == 4
    assert result.failed["conversion"] == 2
    assert result.failure_reasons["conversion"] == {"test.runner.boom": 2}

    for key in ("conversion:0", "conversion:1"):
        assert result.breakout_attempted[key] == 4
        assert result.breakout_failed[key] == 2
        assert result.breakout_failure_reasons[key] == {"test.runner.boom": 2}


# Step 3: calibration test (parameter_recovery mark)


@pytest.mark.xdist_group("simulate-end-to-end-calibration")
@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCalibration:
    """Monte Carlo calibration: CI coverage and SRM detection.

    200 replications of a zero-lift scenario; unconditional coverage must
    clear an exact one-sided Clopper-Pearson lower bound
    (``tests.mc.coverage_lower_bound`` at ``eta=.01``, independently per
    metric) rather than a hand-tuned symmetric band. This is a smoke-level
    gross-miscalibration gate (catches e.g. the day-zero chronology
    defect); certifying the full release scientific budget (delta<=.005, MC
    margin <=.0025) needs the much larger prospectively sized replication
    counts of the dedicated release calibration suites.
    """

    N_REPS = 200
    # 4000 units keeps combined log-scale SE inside the delta-method
    # guard's validity region (1000 units w/ heterogeneity 0.3 draws SE ~0.59).
    N_UNITS = 4000

    @pytest.fixture(scope="class")
    @classmethod
    def fair_result(cls):
        """Load the source-versioned 200-replication calibration fixture."""
        from tests._shared_cache import fixture_cache_key, get_or_build

        def build():
            scenario = Scenario(
                n_units=cls.N_UNITS,
                n_days=14,
                true_lift={"conversion": 0.0, "count": 0.0, "revenue": 0.0},
                assignment_ratio=0.5,
                unit_heterogeneity=0.3,
                seed=0,
            )
            return run_end_to_end(scenario, replications=cls.N_REPS)

        if os.environ.get("INCREMENT_DISABLE_FIXTURE_CACHE") == "1":
            return build()
        root = Path(__file__).resolve().parents[2]
        source_key = fixture_cache_key(root, Path(__file__))
        cache_root = root / ".cache" / "increment-fixtures"
        return get_or_build(cache_root, f"end-to-end-fair-{source_key}", build)

    def test_full_population_no_exclusions_or_failures(self, fair_result):
        """A healthy well-specified scenario attempts, points, and covers
        every replication -- no excluded/failed cells to hide behind."""
        for metric in ("conversion", "count", "revenue"):
            assert fair_result.attempted[metric] == self.N_REPS
            assert fair_result.point_estimable[metric] == self.N_REPS
            assert fair_result.interval_estimable[metric] == self.N_REPS
            assert fair_result.excluded[metric] == 0
            assert fair_result.failed[metric] == 0

    def test_coverage_within_bounds(self, fair_result):
        """Small-run smoke gate retains the prior under/overcoverage sensitivity.

        Full release calibration uses the prospectively sized shared manifest.
        At 200 replications, an exact lower bound of 0.84 admits 19 misses,
        one tighter than the accepted 0.90 observed-rate gate, without excessive
        false failures. The upper check rejects an always-covering interval.
        """
        from tests.mc import coverage_lower_bound

        for metric in ("conversion", "count", "revenue"):
            attempted = fair_result.attempted[metric]
            rate = fair_result.coverage_unconditional[metric]
            assert rate is not None
            hits = round(rate * attempted)
            lower_bound = coverage_lower_bound(hits, attempted, eta=0.01)
            assert lower_bound >= 0.84, (
                f"{metric} unconditional coverage {rate:.3f} ({hits}/{attempted}) has "
                f"exact lower bound {lower_bound:.4f} < 0.84"
            )
            assert rate <= 0.99, f"{metric} coverage {rate:.3f} is vacuously high"

    def test_bias_near_zero(self, fair_result):
        """Tolerance is 4x the run's own reported ``bias_mcse`` (the
        Monte Carlo SE of the mean bias), not a hand-tuned literal."""
        for metric in ("conversion", "count", "revenue"):
            bias = fair_result.bias[metric]
            se = fair_result.bias_mcse[metric]
            assert bias is not None
            assert se is not None
            tol = 4.0 * se
            assert abs(bias) < tol, f"{metric} bias = {bias:.4f} (expected near 0, tol={tol:.4f})"

    def test_srm_respects_configured_ratio(self):
        """A configured 60/40 design delivering ~60/40 must not be flagged
        as SRM; same majority-of-3-seeds smoke guard as the fair-assignment test.
        """
        flags = 0
        for seed in (42, 43, 44):
            scenario = Scenario(
                n_units=self.N_UNITS,
                n_days=14,
                true_lift={"conversion": 0.0},
                assignment_ratio=0.6,
                seed=seed,
            )
            result = run_end_to_end(scenario, replications=1)
            assert result.srm_rate is not None
            flags += int(result.srm_rate)
        assert flags < 2, f"SRM falsely detected in {flags}/3 seeds for a configured 60/40 design"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_nonzero_lift_bias_near_zero():
    """At nonzero conversion lift, mean(estimate - true_lift) ~ 0.

    Both ``Scenario.true_lift`` and the estimator's lift are relative
    lifts of the metric mean, so bias must vanish at nonzero lift too,
    not only at 0 where every scale coincides - an injection on another
    scale (e.g. odds) would surface here as a systematic offset.

    Tolerance ~4 SE of the 12-replication mean bias (~0.007).
    """
    scenario = Scenario(
        n_units=50_000,
        n_days=14,
        true_lift={"conversion": 0.20},
        unit_heterogeneity=0.5,
        seed=7,
    )
    result = run_end_to_end(scenario, replications=12)
    bias = result.bias["conversion"]
    assert bias is not None
    assert abs(bias) < 0.03, (
        f"conversion bias {bias:.4f} at true_lift=0.20 (expected ~0; a "
        f"scale mismatch between injected and estimated lift would land here)"
    )


# Step 4: breakout/CATE routing, MCSE, and SRM detection rate


class TestBreakoutRouting:
    """A segmented scenario routes one breakout call per replication
    through ``readouts.breakout``, end-to-end via ``run_end_to_end``."""

    def test_no_breakout_by_default(self):
        scenario = Scenario(n_units=500, n_days=7, true_lift={"conversion": 0.0}, seed=0)
        result = run_end_to_end(scenario, replications=1)
        assert result.breakout_attempted == {}
        assert result.breakout_bias == {}
        assert result.breakout_coverage_conditional == {}
        assert result.breakout_coverage_unconditional == {}

    @pytest.mark.slow
    def test_breakout_bias_coverage_mcse_keyed_by_metric_and_segment(self):
        scenario = Scenario(
            n_units=3000,
            n_days=7,
            true_lift={"conversion": 0.2},
            unit_heterogeneity=0.3,
            n_segments=3,
            seed=11,
        )
        result = run_end_to_end(scenario, replications=3)
        expected_keys = {f"conversion:{s}" for s in ("0", "1", "2")}
        assert expected_keys == set(result.breakout_attempted)
        assert expected_keys <= set(result.breakout_bias_mcse)
        for key in expected_keys:
            assert result.breakout_attempted[key] == 3
            assert result.breakout_attempted[key] == (
                result.breakout_point_estimable[key]
                + result.breakout_excluded[key]
                + result.breakout_failed[key]
            )
            assert result.breakout_bias[key] is not None
            mcse = result.breakout_bias_mcse[key]
            assert mcse is not None
            assert mcse >= 0.0


class TestCateRouting:
    """A covariate scenario routes one CATE fit per replication through
    ``increment.cate.estimate_cate``, end-to-end via ``run_end_to_end``."""

    def test_no_cate_by_default(self):
        scenario = Scenario(n_units=500, n_days=7, true_lift={"conversion": 0.0}, seed=0)
        result = run_end_to_end(scenario, replications=1)
        assert result.cate_ate == {}
        assert result.cate_interaction == {}

    def test_cate_ate_and_interaction_present_per_metric(self):
        scenario = Scenario(
            n_units=3000,
            n_days=7,
            true_lift={"revenue": 0.1},
            unit_heterogeneity=0.3,
            covariate_sd=1.0,
            covariate_interaction=0.5,
            seed=13,
        )
        result = run_end_to_end(scenario, replications=3)
        assert set(result.cate_ate) == {"revenue"}
        assert set(result.cate_interaction) == {"revenue"}

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_cate_interaction_recovers_sign(self):
        """The fitted mean interaction coefficient across replications must
        share the sign of ``covariate_interaction`` -- proof the CATE fit
        genuinely recovers the DGP's injected heterogeneity, not noise."""
        scenario = Scenario(
            n_units=20_000,
            n_days=7,
            true_lift={"revenue": 0.1},
            unit_heterogeneity=0.3,
            covariate_sd=1.0,
            covariate_interaction=1.0,
            seed=21,
        )
        result = run_end_to_end(scenario, replications=8)
        assert result.cate_interaction["revenue"] > 0.0, (
            f"mean fitted interaction {result.cate_interaction['revenue']:.4f} "
            f"should be positive under covariate_interaction=1.0"
        )


class TestMcseAndSrmRateAggregation:
    """``bias_mcse`` reports the Monte Carlo uncertainty on `bias`;
    `srm_rate` is the fraction of successful replications flagged, not a
    single first-replication check."""

    def test_mcse_present_alongside_bias_and_coverage(self):
        scenario = Scenario(
            n_units=2000,
            n_days=7,
            true_lift={"conversion": 0.0, "count": 0.0},
            seed=1,
        )
        result = run_end_to_end(scenario, replications=5)
        assert set(result.bias_mcse) == set(result.bias)
        for metric, se in result.bias_mcse.items():
            assert se is not None
            assert se >= 0.0, f"bias_mcse for {metric} must be nonnegative, got {se}"

    def test_bias_mcse_null_for_a_single_replication(self):
        """A one-replication run has no spread to estimate MCSE from --
        null, not NaN, per the numeric-absence contract."""
        scenario = Scenario(n_units=500, n_days=7, true_lift={"conversion": 0.0}, seed=1)
        result = run_end_to_end(scenario, replications=1)
        assert result.bias_mcse["conversion"] is None

    def test_srm_rate_averages_every_replication_not_just_the_first(self, monkeypatch):
        """Stub `sample_ratio_mismatch` to flag every other replication;
        `srm_rate` must reflect that average, not the first replication's
        (necessarily deterministic, either always-0 or always-1) result."""
        import increment.simulate.runner as runner_module

        calls = {"n": 0}

        class _FakeSRM:
            def __init__(self, is_srm: bool) -> None:
                self.is_srm = is_srm

        def fake_sample_ratio_mismatch(counts, expected=None, **kwargs):
            calls["n"] += 1
            return _FakeSRM(is_srm=calls["n"] % 2 == 0)

        monkeypatch.setattr(runner_module, "sample_ratio_mismatch", fake_sample_ratio_mismatch)
        scenario = Scenario(n_units=200, n_days=7, true_lift={"conversion": 0.0}, seed=2)
        result = run_end_to_end(scenario, replications=4)
        assert result.srm_rate == 0.5, (
            f"expected 2/4 replications flagged (0.5), got {result.srm_rate}"
        )


def test_json_serialization_emits_nulls_not_nan():
    """Native JSON never emits ``NaN`` for an undefined aggregate --
    numeric nulls only."""
    scenario = Scenario(n_units=300, n_days=5, true_lift={"conversion": 0.0}, seed=1)
    result = run_end_to_end(scenario, replications=1)
    payload = result.model_dump_json()
    assert "NaN" not in payload
    parsed = json.loads(payload)
    # Single replication: no spread to estimate bias_mcse from -> null.
    assert parsed["bias_mcse"]["conversion"] is None


def test_cate_frame_excludes_pre_and_same_time_outcomes():
    import pandas as pd

    from increment.simulate.runner import _cate_source_frame

    pdf = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "exposure",
                "ts": pd.Timestamp("2025-01-01 01:00"),
                "value": float("nan"),
            },
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "covariate",
                "ts": pd.Timestamp("2024-12-31"),
                "value": 0.0,
            },
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "visit",
                "ts": pd.Timestamp("2025-01-01 00:00"),
                "value": 1.0,
            },
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "visit",
                "ts": pd.Timestamp("2025-01-01 01:00"),
                "value": 1.0,
            },
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "visit",
                "ts": pd.Timestamp("2025-01-01 02:00"),
                "value": 1.0,
            },
        ]
    )
    frame = _cate_source_frame(pdf, "count")
    assert frame.loc[frame.unit_id == "u1", "count"].iat[0] == 1.0


def test_cate_source_frame_refuses_an_unrouted_metric():
    import pandas as pd

    from increment.errors import InvalidRequestError as _InvalidRequestError
    from increment.simulate.runner import _cate_source_frame

    pdf = pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "group_id": "control",
                "event": "exposure",
                "ts": pd.Timestamp("2025-01-01 01:00"),
                "value": float("nan"),
            },
        ]
    )
    with pytest.raises(_InvalidRequestError) as exc_info:
        _cate_source_frame(pdf, "d7_retention")
    assert exc_info.value.code == "simulate.runner.cate_routing_does"
    assert exc_info.value.context["metric_name"] == "d7_retention"


# ---------------------------------------------------------------------------
# Shared reducer contract: deterministic fixtures and boundary cases on
# increment.simulate.runner._reduce_key/_KeyOutcome directly, without mocks.


class TestReducerR4P3I2H1:
    """The primary deterministic fixture: R=4, P=3, I=2, H=1, conditional
    coverage=.5, unconditional coverage=.25, counts/reasons exactly
    reconciled. One two-sided hit, one open-lower miss, one finite point
    with no interval (an interval-unavailable reason, never double-counted
    as an exclusion), and one whole-replication failure."""

    TRUTH = 10.0

    def _outcomes(self) -> list[_KeyOutcome]:
        return [
            # two-sided [9, 12], truth=10 inside -> hit
            _KeyOutcome(status="ok", point=10.5, lb=9.0, ub=12.0, open_side=None),
            # open_side="lower": lb unavailable, membership truth <= ub; 10 <= 8 is False -> miss
            _KeyOutcome(status="ok", point=9.0, lb=None, ub=8.0, open_side="lower"),
            # finite point, no interval at all
            _KeyOutcome(
                status="ok",
                point=11.0,
                lb=None,
                ub=None,
                open_side=None,
                interval_reason="degenerate_se",
            ),
            # whole-replication failure
            _KeyOutcome(status="failed", reason="simulate.runner.whole_run_boom"),
        ]

    def test_counts_reconcile(self):
        stats = _reduce_key(self._outcomes(), self.TRUTH)
        assert stats.attempted == 4
        assert stats.point_estimable == 3
        assert stats.interval_estimable == 2
        assert stats.excluded == 0
        assert stats.failed == 1
        assert stats.attempted == stats.point_estimable + stats.excluded + stats.failed
        assert stats.interval_estimable <= stats.point_estimable <= stats.attempted
        assert stats.failure_reasons == {"simulate.runner.whole_run_boom": 1}
        assert stats.interval_unavailable_reasons == {"degenerate_se": 1}
        assert stats.exclusion_reasons == {}

    def test_coverage_matches_exact_fixture_numbers(self):
        stats = _reduce_key(self._outcomes(), self.TRUTH)
        assert stats.coverage_conditional == pytest.approx(0.5)  # 1 hit / 2 intervals
        assert stats.coverage_unconditional == pytest.approx(0.25)  # 1 hit / 4 attempted

    def test_bias_uses_only_the_three_finite_points(self):
        stats = _reduce_key(self._outcomes(), self.TRUTH)
        expected = ((10.5 - 10.0) + (9.0 - 10.0) + (11.0 - 10.0)) / 3.0
        assert stats.bias == pytest.approx(expected)
        assert stats.bias_mcse is not None


class TestReducerMembershipBoundaries:
    """Membership/exclusion boundaries the R=4 fixture doesn't itself
    need to cover: the other open side, an explicit exclusion distinct
    from a failure, and both bounds absent (unavailable, not one-sided)."""

    def test_open_side_upper_hit(self):
        # open_side="upper": ub unavailable, membership lb <= truth; 9 <= 10 -> hit
        outcomes = [_KeyOutcome(status="ok", point=9.5, lb=9.0, ub=None, open_side="upper")]
        stats = _reduce_key(outcomes, truth=10.0)
        assert stats.interval_estimable == 1
        assert stats.coverage_conditional == pytest.approx(1.0)

    def test_open_side_upper_miss(self):
        outcomes = [_KeyOutcome(status="ok", point=15.0, lb=11.0, ub=None, open_side="upper")]
        stats = _reduce_key(outcomes, truth=10.0)
        assert stats.coverage_conditional == pytest.approx(0.0)

    def test_both_bounds_absent_is_unavailable_not_one_sided(self):
        outcomes = [_KeyOutcome(status="ok", point=10.0, lb=None, ub=None, open_side=None)]
        stats = _reduce_key(outcomes, truth=10.0)
        assert stats.point_estimable == 1
        assert stats.interval_estimable == 0
        assert stats.interval_unavailable_reasons == {"no_interval": 1}

    def test_excluded_point_is_distinct_from_failed(self):
        outcomes = [
            _KeyOutcome(status="excluded", reason="estimation.engine.missing_control"),
            _KeyOutcome(status="failed", reason="simulate.runner.replications_invalid"),
        ]
        stats = _reduce_key(outcomes, truth=10.0)
        assert stats.attempted == 2
        assert stats.excluded == 1
        assert stats.failed == 1
        assert stats.point_estimable == 0
        assert stats.exclusion_reasons == {"estimation.engine.missing_control": 1}
        assert stats.failure_reasons == {"simulate.runner.replications_invalid": 1}


class TestReducerEmptyOneObservationAllUnavailable:
    """Empty/one-observation/all-unavailable boundary cases, plus a normal
    available case so an all-null implementation cannot pass."""

    def test_empty_is_fully_null(self):
        stats = _reduce_key([], truth=1.0)
        assert stats.attempted == 0
        assert stats.bias is None
        assert stats.bias_mcse is None
        assert stats.coverage_conditional is None
        assert stats.coverage_unconditional is None

    def test_single_observation_has_a_point_but_no_mcse(self):
        outcomes = [_KeyOutcome(status="ok", point=1.5, lb=1.0, ub=2.0, open_side=None)]
        stats = _reduce_key(outcomes, truth=1.0)
        assert stats.bias == pytest.approx(0.5)
        assert stats.bias_mcse is None  # needs >= 2 observations
        assert stats.coverage_conditional == pytest.approx(1.0)
        assert stats.coverage_conditional_mcse is None
        assert stats.coverage_unconditional == pytest.approx(1.0)
        assert stats.coverage_unconditional_mcse is None

    def test_all_failed_gives_zero_unconditional_coverage_with_explicit_failed_count(self):
        outcomes = [_KeyOutcome(status="failed", reason="boom") for _ in range(5)]
        stats = _reduce_key(outcomes, truth=1.0)
        assert stats.attempted == 5
        assert stats.failed == 5
        assert stats.point_estimable == 0
        assert stats.bias is None
        assert stats.coverage_conditional is None  # I=0
        assert stats.coverage_unconditional == pytest.approx(0.0)  # H=0 / R=5
        assert stats.coverage_unconditional_mcse == pytest.approx(0.0)  # plug-in at p=0
        assert stats.failure_reasons == {"boom": 5}

    def test_normal_available_case_is_not_all_null(self):
        """A trivially-all-None implementation must not pass: every field
        below is a genuine non-None number for a healthy 10-rep sample."""
        outcomes = [
            _KeyOutcome(status="ok", point=1.0 + 0.1 * i, lb=0.5, ub=1.6, open_side=None)
            for i in range(10)
        ]
        stats = _reduce_key(outcomes, truth=1.0)
        assert stats.bias is not None
        assert stats.bias_mcse is not None
        assert stats.coverage_conditional is not None
        assert stats.coverage_conditional_mcse is not None
        assert stats.coverage_unconditional is not None
        assert stats.coverage_unconditional_mcse is not None


class TestReducerOverflowSafety:
    def test_neighboring_floats_preserve_exact_mcse(self):
        lower = math.nextafter(1.0, 0.0)
        gap = 1.0 - lower
        outcomes = [
            _KeyOutcome(status="ok", point=1.0),
            _KeyOutcome(status="ok", point=lower),
        ]

        stats = _reduce_key(outcomes, truth=0.0)

        assert stats.bias_mcse == gap / 2.0

    def test_large_finite_mean_does_not_overflow(self):
        largest = sys.float_info.max
        outcomes = [_KeyOutcome(status="ok", point=largest) for _ in range(2)]

        stats = _reduce_key(outcomes, truth=0.0)

        assert stats.bias == largest
        assert stats.bias_mcse == 0.0

    def test_canceling_large_inputs_are_order_independent(self):
        largest = sys.float_info.max
        points = [largest, largest, -largest, -largest]

        forward = _reduce_key(
            [_KeyOutcome(status="ok", point=point) for point in points], truth=0.0
        )
        reordered = _reduce_key(
            [_KeyOutcome(status="ok", point=point) for point in reversed(points)], truth=0.0
        )

        assert forward.bias == 0.0
        assert reordered.bias == forward.bias
        assert reordered.bias_mcse == forward.bias_mcse

    def test_near_overflow_spread_has_finite_mcse(self):
        largest = sys.float_info.max
        outcomes = [
            _KeyOutcome(status="ok", point=-largest),
            _KeyOutcome(status="ok", point=largest),
        ]

        stats = _reduce_key(outcomes, truth=0.0)

        assert stats.bias == 0.0
        assert stats.bias_mcse == largest
        assert stats.bias_mcse is not None
        assert math.isfinite(stats.bias_mcse)


def _valid_eval_kwargs() -> dict[str, object]:
    key = "metric"
    return {
        "attempted": {key: 2},
        "point_estimable": {key: 1},
        "interval_estimable": {key: 0},
        "confidence_set_estimable": {key: 0},
        "set_only": {key: 0},
        "excluded": {key: 1},
        "failed": {key: 0},
        "failure_reasons": {key: {}},
        "exclusion_reasons": {key: {"excluded": 1}},
        "interval_unavailable_reasons": {key: {"no_interval": 1}},
        "bias": {key: 0.25},
        "bias_mcse": {key: None},
        "coverage_conditional": {key: None},
        "coverage_conditional_mcse": {key: None},
        "coverage_unconditional": {key: 0.0},
        "coverage_unconditional_mcse": {key: 0.0},
        "set_coverage_conditional": {key: None},
        "set_coverage_conditional_mcse": {key: None},
        "set_coverage_unconditional": {key: 0.0},
        "set_coverage_unconditional_mcse": {key: 0.0},
    }


class TestEvalResultValidationAndImmutability:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("bias", {"other": 0.25}),
            ("attempted", {"metric": 1}),
            ("point_estimable", {"metric": -1}),
            ("failure_reasons", {"metric": {"boom": 1}}),
            ("bias", {"metric": math.inf}),
        ],
    )
    def test_rejects_invalid_result_contracts(self, field, value):
        kwargs = _valid_eval_kwargs()
        kwargs[field] = value

        with pytest.raises(ValidationError):
            EvalResult.model_validate(kwargs)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("coverage_conditional", {"metric": 0.0}),
            ("coverage_unconditional", {"metric": None}),
            ("coverage_unconditional", {"metric": 1.01}),
            ("coverage_unconditional_mcse", {"metric": None}),
            ("coverage_unconditional_mcse", {"metric": -0.01}),
            ("bias_mcse", {"metric": 0.0}),
        ],
    )
    def test_rejects_invalid_aggregate_contracts(self, field, value):
        kwargs = _valid_eval_kwargs()
        kwargs[field] = value

        with pytest.raises(ValidationError):
            EvalResult.model_validate(kwargs)

    def test_rejects_finite_core_bias_with_no_estimable_points(self):
        kwargs = _valid_eval_kwargs()
        kwargs.update(
            point_estimable={"metric": 0},
            excluded={"metric": 2},
            exclusion_reasons={"metric": {"excluded": 2}},
            interval_unavailable_reasons={"metric": {}},
        )

        with pytest.raises(ValidationError) as raised:
            EvalResult.model_validate(kwargs)
        assert [error["type"] for error in raised.value.errors()] == [
            "simulate.runner.result_contract"
        ]

    def test_allows_unrepresentable_bias_with_estimable_points(self):
        kwargs = _valid_eval_kwargs()
        kwargs["bias"] = {"metric": None}

        result = EvalResult.model_validate(kwargs)

        assert result.bias["metric"] is None

    def test_rejects_finite_breakout_bias_with_no_estimable_points(self):
        kwargs = _valid_eval_kwargs()
        kwargs.update(
            breakout_attempted={"metric:0": 1},
            breakout_point_estimable={"metric:0": 0},
            breakout_interval_estimable={"metric:0": 0},
            breakout_confidence_set_estimable={"metric:0": 0},
            breakout_set_only={"metric:0": 0},
            breakout_excluded={"metric:0": 1},
            breakout_failed={"metric:0": 0},
            breakout_failure_reasons={"metric:0": {}},
            breakout_exclusion_reasons={"metric:0": {"excluded": 1}},
            breakout_interval_unavailable_reasons={"metric:0": {}},
            breakout_bias={"metric:0": 0.0},
            breakout_bias_mcse={"metric:0": None},
            breakout_coverage_conditional={"metric:0": None},
            breakout_coverage_conditional_mcse={"metric:0": None},
            breakout_coverage_unconditional={"metric:0": 0.0},
            breakout_coverage_unconditional_mcse={"metric:0": None},
            breakout_set_coverage_conditional={"metric:0": None},
            breakout_set_coverage_conditional_mcse={"metric:0": None},
            breakout_set_coverage_unconditional={"metric:0": 0.0},
            breakout_set_coverage_unconditional_mcse={"metric:0": None},
        )

        with pytest.raises(ValidationError) as raised:
            EvalResult.model_validate(kwargs)
        assert [error["type"] for error in raised.value.errors()] == [
            "simulate.runner.result_contract"
        ]

    def test_rejects_invalid_breakout_contract(self):
        kwargs = _valid_eval_kwargs()
        kwargs.update(
            breakout_attempted={"metric:0": 1},
            breakout_point_estimable={"metric:0": 2},
            breakout_interval_estimable={"metric:0": 0},
            breakout_confidence_set_estimable={"metric:0": 0},
            breakout_set_only={"metric:0": 0},
            breakout_excluded={"metric:0": 0},
            breakout_failed={"metric:0": 0},
            breakout_failure_reasons={"metric:0": {}},
            breakout_exclusion_reasons={"metric:0": {}},
            breakout_interval_unavailable_reasons={"metric:0": {"no_interval": 2}},
            breakout_bias={"metric:0": 0.0},
            breakout_bias_mcse={"metric:0": 0.0},
            breakout_coverage_conditional={"metric:0": None},
            breakout_coverage_conditional_mcse={"metric:0": None},
            breakout_coverage_unconditional={"metric:0": 0.0},
            breakout_coverage_unconditional_mcse={"metric:0": None},
            breakout_set_coverage_conditional={"metric:0": None},
            breakout_set_coverage_conditional_mcse={"metric:0": None},
            breakout_set_coverage_unconditional={"metric:0": 0.0},
            breakout_set_coverage_unconditional_mcse={"metric:0": None},
        )

        with pytest.raises(ValidationError):
            EvalResult.model_validate(kwargs)

    def test_copies_and_recursively_freezes_mappings_without_breaking_json(self):
        kwargs = _valid_eval_kwargs()
        attempted = {"metric": 2}
        reasons = {"metric": {"excluded": 1}}
        kwargs["attempted"] = attempted
        kwargs["exclusion_reasons"] = reasons
        result = EvalResult.model_validate(kwargs)

        attempted["metric"] = 99
        reasons["metric"]["excluded"] = 99
        assert result.attempted["metric"] == 2
        assert result.exclusion_reasons["metric"]["excluded"] == 1
        with pytest.raises(TypeError):
            result.attempted["metric"] = 99  # type: ignore[index]  # ty: ignore[invalid-assignment]
        with pytest.raises(TypeError):
            result.exclusion_reasons["metric"]["excluded"] = 99  # type: ignore[index]  # ty: ignore[invalid-assignment]

        payload = json.loads(result.model_dump_json())
        assert payload["attempted"] == {"metric": 2}
        assert payload["exclusion_reasons"] == {"metric": {"excluded": 1}}

    @pytest.mark.parametrize("field", ["breakout_attempted", "cate_ate", "cate_interaction"])
    def test_omitted_mapping_defaults_are_frozen(self, field):
        result = EvalResult.model_validate(_valid_eval_kwargs())

        with pytest.raises(TypeError):
            getattr(result, field)["new"] = 1  # type: ignore[index]
