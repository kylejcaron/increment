"""Readout dispatch estimand axis: default/rejected `estimands`, as-of LATE
trend, and per-segment breakout gating.

Exercises the `increment.readouts` functions directly (`run`,
`asof_lift`, `breakout`) against the frame seam
(`increment.frame.from_unit_summary`) where a real
`MomentSource` implementation exists, and against a bare `MomentSource`
test double where none does yet (`"asof"` grain and breakout `by=`
dimensions - neither is implemented by any concrete source in this
codebase; `readouts.asof_lift`/`readouts.breakout`'s `Encouragement`
branch are otherwise unreachable). The guardrail-restriction test goes
through `Analysis.run()` on a hand-built native instance (DuckDB,
`Analysis.__new__` + manual attribute assignment, the same pattern
`tests/test_analysis.py` already uses) - see that test's docstring for
why the layering decision landed there.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis, readouts
from increment.errors import CapabilityError, InvalidRequestError, UnsupportedRequestError
from increment.estimation import Normal
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec, from_unit_summary
from increment.readouts._common import _validate_encouragement_asof_inference
from increment.readouts._encouragement import encouragement_rows
from increment.semantics.design import Encouragement, ExclusionRestriction, Randomized, UptakeSpec
from increment.semantics.models import AnalysisPlan, Definitions, InferenceSpec, MeanMetric
from increment.sources import MomentSource, SourceContext, SourceOperation
from tests.analysis_factory import _native_source, lift_rows, make_analysis, make_analysis_like
from tests.sequential_cases import registration

METRIC = MeanMetric(name="rev", entity="user_id", fact="orders", aggregation="sum")


def _design(**over):
    base = {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "help_click"},
        "exclusion_restriction": {
            "acknowledged": True,
            "justification": "unclicked button assumed inert",
        },
    }
    base.update(over)
    return Encouragement.model_validate(base)


def _rows(n=4000, tau=2.0, compliance=0.5, one_sided=True, seed=7, metric="rev"):
    """Simulate an encouragement DGP and collapse to group_summary rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for gid, encouraged in (("control", 0), ("treat", 1)):
        if encouraged:
            d = rng.binomial(1, compliance, size=n)
        else:
            d = np.zeros(n) if one_sided else rng.binomial(1, 0.1, size=n)
        y = 10.0 + tau * d + rng.normal(0, 2.0, size=n)
        yd = y * d
        rows.append(
            centered_row_from_raw_sums(
                {
                    "experiment_id": "s",
                    "metric": metric,
                    "group_id": gid,
                    "n": n,
                    "sum_y": float(y.sum()),
                    "sum_y2": float((y**2).sum()),
                    "sum_d": float(d.sum()),
                    "sum_yd": float(yd.sum()),
                    "sum_y2d": float((y**2 * d).sum()),
                }
            )
        )
    return rows


class FakeMomentSource:
    """Literal outcome rows and uptake counts for readout orchestration tests.

    Rows are precomputed for the requested grain; only metric filtering happens
    here. Compliance dates come from rows carrying uptake sufficient state.
    """

    breakouts: tuple[str, ...] = ()
    operations: frozenset[SourceOperation] = frozenset()
    cluster: str | None = None
    shape: Literal["unit_summary", "unit_panel"] | None = None

    def __init__(
        self,
        rows,
        *,
        metrics,
        capabilities,
        study_id="fake",
        design=None,
        plan=None,
        breakouts=(),
    ):
        self._rows = [dict(r) for r in rows]
        self.breakouts = tuple(breakouts)
        self.capabilities = frozenset(capabilities)
        from increment.plan import compile_decision_plan
        from increment.sources import MomentsSource

        metric_catalog = tuple(metrics)
        # A plan-free reference source resolves the same default per-metric
        # configs the readouts consume, through the public constructor.
        reference = MomentsSource(
            [],
            metrics=metric_catalog,
            study_id=study_id,
            design=Randomized(control_group="control"),
        )
        self._context = SourceContext(
            study_id=study_id,
            design=design,
            plan=compile_decision_plan(plan, metric_catalog, path="warehouse", design=design),
            metrics=metric_catalog,
            configs=reference.context.configs,
            cluster=None,
        )

    @property
    def context(self):
        return self._context

    def moments(
        self,
        metric,
        *,
        grain="total",
        by=(),
        completed_windows_only=False,
        include_covariate: bool = False,
    ):
        return [dict(r) for r in self._rows if r["metric"] == metric.name]

    def unit_frame(self, metric, *, covariates=()):
        raise NotImplementedError

    def unit_counts(self):
        return {}

    def cluster_counts(self):
        raise CapabilityError(
            "cluster_counts() is unavailable on this test double: it declares "
            "no cluster column. The randomization-grain count needs a declared "
            "cluster column; build the source with from_unit_summary(..., "
            "cluster=...) or analyse from definitions declaring Experiment.cluster.",
            code="test.source.cluster_grain",
            context={"source": "this test double", "because": "it declares no cluster column"},
        )

    def compliance_dates(self):
        return tuple(
            dict.fromkeys(
                row["ds"]
                for row in self._rows
                if row.get("ds") is not None and row.get("sum_d") is not None
            )
        )

    def compliance_summary(self, design, *, as_of=None, completed_windows_only=False):

        from increment.sources import ComplianceArm, ComplianceSummary

        rows = [row for row in self._rows if as_of is None or row.get("ds") == as_of]
        by_arm = {}
        for row in rows:
            if row.get("sum_d") is not None and row["group_id"] not in by_arm:
                by_arm[row["group_id"]] = ComplianceArm(
                    group_id=row["group_id"], n_units=row["n"], uptake_total=row["sum_d"]
                )
        return ComplianceSummary(
            study_id=self.context.study_id,
            control_group=str(design.control_group),
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            cluster=None,
            as_of=as_of,
            arms=tuple(by_arm.values()),
        )

    def sql(self, *, grain="total"):
        raise NotImplementedError

    def close(self):
        pass


# - run(): default estimands on Encouragement, rejected on Randomized -----


def _uptake_table():
    rows = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "treatment", 25.0, 1),
        ("u06", "treatment", 30.0, 1),
        ("u07", "treatment", 28.0, 1),
        ("u08", "treatment", 15.0, 0),
    ]
    cols = list(zip(*rows, strict=True))
    return pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))


def test_run_refuses_encouragement_quantile_before_loading_unit_rows(monkeypatch):
    src = from_unit_summary(
        _uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", type="quantile", quantile=0.9)],
        uptake="clicked",
        design=_frame_encouragement_design(),
    )

    def unexpected_load(*args, **kwargs):
        raise AssertionError("unit_frame must not be loaded for an unsupported metric")

    monkeypatch.setattr(src, "unit_frame", unexpected_load)
    with pytest.raises(CapabilityError) as raised:
        readouts.run(src)
    assert raised.value.code == "breakout.quantile"
    assert raised.value.context["names"] == ("revenue",)


def test_run_default_estimands_on_encouragement_design():
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        # Low z-floor: keep LATE reported on this 4-units-per-arm fixture
        # instead of exercising weak-instrument suppression here.
        min_first_stage_z=0.5,
    )
    src = from_unit_summary(
        _uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    results = readouts.run(src)
    assert {r.estimand for r in results} == {"itt", "compliance", "late"}


def test_run_narrowed_estimands_on_encouragement_design_are_honored():
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        min_first_stage_z=0.5,
    )
    src = from_unit_summary(
        _uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    results = readouts.run(src, estimands=("itt",))
    assert {r.estimand for r in results} == {"itt"}


def test_run_rejects_estimands_on_randomized_design():
    design = Randomized(control_group="control")
    src = from_unit_summary(
        _uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=design,
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.run(src, estimands=("compliance",))
    assert exc_info.value.code == "readout.assignment.estimands"


# - asof_lift(): as-of LATE trend, monitoring note, per-date suppression


@pytest.mark.parametrize("path", ["readouts", "analysis", "native_analysis"])
@pytest.mark.parametrize("axis_state", ["missing", "noncallable"])
def test_missing_compliance_axis_refuses_before_reading_outcomes(monkeypatch, path, axis_state):
    def unavailable_outcomes(*args, **kwargs):
        pytest.fail("missing compliance axis must refuse before outcome reads")

    reached_day_source: list[str] = []

    class NativeSource(FakeMomentSource):
        operations: frozenset[SourceOperation] = frozenset({"moments_source", "day_source"})
        sitewide_evidence = unavailable_outcomes
        breakout_summaries = unavailable_outcomes
        factor_summaries = unavailable_outcomes
        breakout_source = unavailable_outcomes
        breakout_sources = unavailable_outcomes

        def day_source(self, *, metrics):
            assert metrics == [METRIC]
            reached_day_source.append("day_source")
            raise AssertionError("native day_source reached")

    design = _design(one_sided=True)
    rows = [dict(row, ds=date(2025, 1, 1)) for row in _rows(n=200)]
    source_cls = NativeSource if path == "native_analysis" else FakeMomentSource
    src = source_cls(rows, metrics=[METRIC], capabilities={"asof"}, design=design)
    src.shape = "unit_panel"
    monkeypatch.setattr(src, "moments", unavailable_outcomes)
    with monkeypatch.context() as axis_patch:
        if axis_state == "missing":
            axis_patch.delattr(FakeMomentSource, "compliance_dates")
        else:
            axis_patch.setattr(src, "compliance_dates", None)

        with pytest.raises(CapabilityError) as error:
            if path == "readouts":
                readouts.asof_lift(src)
            else:
                Analysis._from_source(src, design).run_asof_lift()
    assert error.value.code == "source.compliance_summary.legacy_uptake_state"

    if path == "native_analysis":
        # A valid compliance axis must reach the guarded native outcome read.
        with pytest.raises(AssertionError):
            Analysis._from_source(src, design).run_asof_lift()
        assert reached_day_source == ["day_source"]


@pytest.mark.parametrize("path", ["readouts", "analysis"])
@pytest.mark.parametrize(
    "selected", [("rev",), ("other",), ("absent",), ("rev", "other"), ("other", "rev")]
)
def test_custom_compliance_axis_is_independent_of_selected_outcome_dates(path, selected):
    from increment.sources import ComplianceArm, ComplianceSummary

    design = _design(one_sided=True)
    days = (date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3))
    uptake_totals = dict(zip(days, (50.0, 100.0, 150.0), strict=True))

    class EnrollmentSource(FakeMomentSource):
        def compliance_dates(self):
            return days

        def compliance_summary(self, design, *, as_of=None, completed_windows_only=False):
            assert isinstance(as_of, date)
            return ComplianceSummary(
                study_id=self.context.study_id,
                control_group="control",
                cohort=design.uptake.fact,
                window_days=design.uptake.window_days,
                one_sided=True,
                cluster=None,
                as_of=as_of,
                arms=(
                    ComplianceArm(group_id="control", n_units=200, uptake_total=0.0),
                    ComplianceArm(group_id="treat", n_units=200, uptake_total=uptake_totals[as_of]),
                ),
            )

    rows = [dict(row, ds=days[0]) for row in _rows(n=200)]
    rows += [dict(row, ds=days[2]) for row in _rows(n=200, metric="other")]
    src = EnrollmentSource(
        rows,
        metrics=[METRIC, *(METRIC.model_copy(update={"name": n}) for n in ("other", "absent"))],
        capabilities={"asof"},
        design=design,
    )
    src.shape = "unit_panel"
    results = (
        readouts.asof_lift(src, metrics=selected, estimands=("itt", "compliance"))
        if path == "readouts"
        else Analysis._from_source(src, design).run_asof_lift(
            metrics=selected, estimands=("itt", "compliance")
        )
    )

    compliance = [row for row in results if row.estimand == "compliance"]
    assert [row.ds for row in compliance] == list(days)
    assert [row.require_lift().value for row in compliance] == pytest.approx([0.25, 0.5, 0.75])
    assert all(row.require_lift().lb is not None for row in compliance)
    assert {row.ds for row in results if row.estimand == "itt"} == {
        day for name, day in (("rev", days[0]), ("other", days[2])) if name in selected
    }


@pytest.mark.parametrize("path", ["readouts", "analysis"])
def test_asof_lift_suppresses_late_on_weak_days_and_reports_it_once_strong(path):
    design = _design(one_sided=True)  # default min_first_stage_z=4.0, uptake unwindowed
    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    weak_day = _rows(n=200, compliance=0.02, seed=11)
    strong_day = _rows(n=4000, compliance=0.5, seed=12)
    for r in weak_day:
        r["ds"] = day1
    for r in strong_day:
        r["ds"] = day2
    src = FakeMomentSource(
        weak_day + strong_day, metrics=[METRIC], capabilities={"asof"}, design=design
    )
    if path == "analysis":
        src.shape = "unit_panel"

    results = (
        readouts.asof_lift(src)
        if path == "readouts"
        else Analysis._from_source(src, design).run_asof_lift()
    )

    by_date: defaultdict[date | datetime | str | int | float, set[str]] = defaultdict(set)
    for r in results:
        assert r.ds is not None
        by_date[r.ds].add(r.estimand)
    assert by_date[day1] == {"itt", "compliance"}, "weak first stage must suppress late"
    assert "late" in by_date[day2], "strong first stage must report late"

    late_rows = [r for r in results if r.estimand == "late"]
    assert late_rows
    for r in late_rows:
        assert r.ds == day2


def test_asof_lift_always_valid_requires_finalized_bounded_windows():
    from increment.semantics.models import ConversionMetric
    from tests.sequential_cases import declared_plan

    metric = ConversionMetric(name="rev", entity=METRIC.entity, fact=METRIC.fact, window_days=7)
    design = _design(uptake={"fact": "help_click", "window_days": 7})
    plan = declared_plan([metric], source_id="fake", design=design)
    rows = _rows(n=4000, compliance=0.5, seed=12)
    for row in rows:
        row["ds"] = date(2025, 1, 1)
    src = FakeMomentSource(rows, metrics=[metric], capabilities={"asof"}, design=design, plan=plan)
    with pytest.raises(InvalidRequestError) as unfinished:
        readouts.asof_lift(src, estimands=("itt",))
    assert unfinished.value.code == "readout.encouragement.asof_completion"
    with pytest.raises(CapabilityError) as raised:
        readouts.asof_lift(src, completed_windows_only=True, estimands=("itt",))
    assert raised.value.code == "sequential.continuation.legacy"


def _mixed_family_plan(*, primary="rev"):
    """AnalysisPlan composing an asymptotic scalar-mean ITT cell with an
    exact Bernoulli uptake cell (MixedFamily), via the public
    ``inference=asymptotic_mean`` plus ``compliance=`` route -- never a
    public ``mixed_family`` kind."""
    from increment import SequentialCompliancePolicy
    from tests.estimation.test_encouragement import _sequential_policy

    reg = _sequential_policy(uptake=True).registration
    return AnalysisPlan(
        primary=primary,
        inference=InferenceSpec(kind="asymptotic_mean", registration=reg),
        compliance=SequentialCompliancePolicy(alpha=reg.roster[1].alpha),
    )


def test_asof_lift_mixed_family_requires_finalized_bounded_windows():
    """A MixedFamily readout (asymptotic ITT plus exact Bernoulli uptake)
    faces the same finalized-bounded-window gate as AlwaysValid: its uptake
    cell is exact and needs the identical completed-windows guarantee."""
    from increment.estimation.sequential import MixedFamily

    design = _design(uptake={"fact": "help_click", "window_days": 7})
    plan = _mixed_family_plan()
    rows = _rows(n=4000, compliance=0.5, seed=12)
    for row in rows:
        row["ds"] = date(2025, 1, 1)
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"asof"},
        design=design,
        plan=plan,
        study_id="experiment",
    )
    assert isinstance(src.context.plan.inference, MixedFamily)
    with pytest.raises(InvalidRequestError) as unfinished:
        readouts.asof_lift(src, estimands=("itt",))
    assert unfinished.value.code == "readout.encouragement.asof_completion"
    with pytest.raises(InvalidRequestError) as unbounded:
        readouts.asof_lift(src, completed_windows_only=True, estimands=("itt",))
    assert unbounded.value.code == "readout.encouragement.asof_unbounded"


def test_validate_encouragement_asof_inference_gates_mixed_family_and_spares_asymptotic_only():
    """``_validate_encouragement_asof_inference`` (``increment.readouts._common``) must gate
    MixedFamily the same way it gates AlwaysValid -- both carry an exact
    Bernoulli cell -- while leaving plain AsymptoticMean (no compliance
    cell) untouched."""
    from increment.estimation.sequential import AsymptoticMean, MixedFamily
    from tests.estimation.test_encouragement import _sequential_policy

    design = _design(uptake={"fact": "help_click"})  # uptake left unwindowed
    mixed = _sequential_policy(uptake=True)
    assert isinstance(mixed, MixedFamily)
    with pytest.raises(InvalidRequestError) as unfinished:
        _validate_encouragement_asof_inference(
            [METRIC], design, inference=mixed, completed_windows_only=False, method="asof_lift"
        )
    assert unfinished.value.code == "readout.alwaysvalid_under_encouragement"
    with pytest.raises(InvalidRequestError) as unbounded:
        _validate_encouragement_asof_inference(
            [METRIC], design, inference=mixed, completed_windows_only=True, method="asof_lift"
        )
    assert unbounded.value.code == "readout.completed_encouragement_inference"

    asymptotic_only = _sequential_policy(uptake=False)
    assert isinstance(asymptotic_only, AsymptoticMean)
    _validate_encouragement_asof_inference(
        [METRIC],
        design,
        inference=asymptotic_only,
        completed_windows_only=False,
        method="asof_lift",
    )


@pytest.mark.slow
def test_asof_lift_raw_joint_itt_and_uptake_replay():
    """Continuous ITT under the public asymptotic mean law, composed
    automatically with Bernoulli uptake compliance via MixedFamily --
    InferenceSpec(kind="asymptotic_mean") plus AnalysisPlan.compliance,
    never a public "mixed_family" kind."""
    from increment import (
        SequentialCell,
        SequentialCompliancePolicy,
        SequentialModel,
        SequentialRegistration,
    )
    from increment.estimation.sequential import MixedFamily
    from increment.frame import from_unit_panel, synthesise_metric
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id
    from tests.asymptotic_cases import mean_model
    from tests.sequential_cases import registration

    specs = [MetricSpec(name="outcome", window_days=7)]
    design = _design(one_sided=False, uptake={"fact": "clicked", "window_days": 7})
    outcome = mean_model("outcome", start_count=4)
    uptake = SequentialModel.model_validate(
        {
            **registration("bernoulli").models[0].model_dump(),
            "metric": "uptake",
            "observable": "uptake",
        }
    )
    reg = SequentialRegistration.model_validate(
        {
            **registration("bernoulli").model_dump(),
            "source_id": "frame",
            "models": (outcome, uptake),
            "definitions_id": sequential_definition_id(
                [synthesise_metric(s) for s in specs],
                design,
                transformations=specs,
                source_mapping=frame_observation_mapping(
                    unit="unit", group="arm", date="ds", exposure_date="exposed", uptake="clicked"
                ),
            ),
            "roster": (
                SequentialCell(metric="outcome", group_id="treatment"),
                SequentialCell(metric="uptake", group_id="treatment", estimand="compliance"),
            ),
        }
    )
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", registration=reg),
        compliance=SequentialCompliancePolicy(alpha=reg.roster[0].alpha),
    )
    rows = [
        {
            "unit": f"{i:08d}-{arm}",
            "arm": arm,
            "ds": date(2025, 1, 1),
            "exposed": date(2025, 1, 1),
            "outcome": [1, 2, 3, 4][i % 4] if arm == "control" else [6, 10, 14, 18][i % 4],
            "clicked": int(i % 4 == 0) if arm == "control" else int(i % 4 != 0),
        }
        for i in range(96)
        for arm in ("control", "treatment")
    ]
    source = from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        date="ds",
        control="control",
        exposure_date="exposed",
        uptake="clicked",
        observation_end=date(2025, 1, 20),
        design=design,
        metrics=specs,
        plan=plan,
    )
    source.capture_sequential(finalized=True, as_of=date(2025, 1, 15))
    results = readouts.asof_lift(
        source, estimands=("itt", "compliance"), completed_windows_only=True
    )
    assert {row.estimand for row in results} == {"itt", "compliance"}
    assert all(row.stat_sig() for row in results)
    from increment.estimation.encouragement import estimate_encouragement

    inference = source.context.plan.inference
    assert isinstance(inference, MixedFamily)
    direct = estimate_encouragement(
        source.context.metrics,
        source.sequential_snapshot(),
        design,
        estimands=("itt", "compliance"),
        inference=inference,
    )
    assert {row.estimand: row.require_sequential_result() for row in direct.results} == {
        row.estimand: row.require_sequential_result() for row in results
    }
    assert all(LiftEstimate.model_validate_json(row.model_dump_json()) == row for row in results)


def test_validate_encouragement_asof_inference_uses_the_retention_band_not_window_days():
    """A `RetentionMetric` has no `window_days` field (its band lives in
    `threshold_days`) - the gate must read boundedness off the band's
    right edge, not the vacated attribute, or every retention metric
    (bounded or not) would be misclassified as unbounded and rejected."""
    from increment.estimation.sequential import AlwaysValid
    from increment.semantics.models import RetentionMetric

    bounded = RetentionMetric(name="ret", entity="user_id", fact="orders", threshold_days=(0, 7))
    unbounded = RetentionMetric(name="ret", entity="user_id", fact="orders", threshold_days=7)
    design = _design(uptake={"fact": "help_click", "window_days": 7})

    _validate_encouragement_asof_inference(
        [bounded],
        design,
        inference=AlwaysValid(registration=registration("gaussian")),
        completed_windows_only=True,
        method="asof_lift",
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _validate_encouragement_asof_inference(
            [unbounded],
            design,
            inference=AlwaysValid(registration=registration("gaussian")),
            completed_windows_only=True,
            method="asof_lift",
        )
    assert exc_info.value.code == "readout.completed_encouragement_inference"


def test_asof_lift_randomized_design_stamps_ds_and_reports_itt():
    """The Randomized branch (estimate_lift dispatch, no encouragement
    guard/note logic) was previously exercised only through
    Analysis.run_asof_lift's DIFFERENT day-axis pipeline, never through
    this free function directly. Every row must carry itt only, a stamped
    ds, and the series-wide fixed-alpha monitoring caveat (an as-of itt
    trend invites the same repeated looks a late trend does).
    """
    design = Randomized(control_group="control")
    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    rows_day1 = [
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "control",
            "n": 50,
            "sum_y": 500.0,
            "sum_y2": 5100.0,
            "ds": day1,
        },
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "treat",
            "n": 50,
            "sum_y": 600.0,
            "sum_y2": 7300.0,
            "ds": day1,
        },
    ]
    rows_day2 = [
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "sum_y": 1000.0,
            "sum_y2": 10200.0,
            "ds": day2,
        },
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "treat",
            "n": 100,
            "sum_y": 1200.0,
            "sum_y2": 14600.0,
            "ds": day2,
        },
    ]
    src = FakeMomentSource(
        [centered_row_from_raw_sums(r) for r in rows_day1 + rows_day2],
        metrics=[METRIC],
        capabilities={"asof"},
        design=design,
    )

    results = readouts.asof_lift(src)

    assert results
    assert {r.ds for r in results} == {day1, day2}
    assert all(r.estimand == "itt" for r in results)
    assert all(
        r.note == "monitoring readout: fixed-alpha, not valid under repeated looks" for r in results
    )


def test_asof_lift_randomized_design_forwards_preferred_direction():
    """Regression: the Randomized branch must forward preferred_direction
    to estimate_lift, same as run()'s equivalent branch - otherwise every
    as-of estimate carries preferred_direction=None and prob_favorable()
    raises for a metric that DOES declare a direction, contradicting the
    'pipeline estimates always carry one' guarantee documented elsewhere."""
    directed_metric = MeanMetric(
        name="rev",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="increase",
    )
    design = Randomized(control_group="control")
    day1 = date(2025, 1, 1)
    rows = [
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "control",
            "n": 50,
            "sum_y": 500.0,
            "sum_y2": 5100.0,
            "ds": day1,
        },
        {
            "experiment_id": "s",
            "metric": "rev",
            "group_id": "treat",
            "n": 50,
            "sum_y": 600.0,
            "sum_y2": 7300.0,
            "ds": day1,
        },
    ]
    src = FakeMomentSource(
        [centered_row_from_raw_sums(r) for r in rows],
        metrics=[directed_metric],
        capabilities={"asof"},
        design=design,
    )

    (result,) = readouts.asof_lift(src)

    assert result.preferred_direction == "increase"
    assert result.prob_favorable() is not None


def test_asof_lift_rejects_estimands_on_randomized_design():
    design = Randomized(control_group="control")
    src = FakeMomentSource([], metrics=[METRIC], capabilities={"asof"}, design=design)
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.asof_lift(src, estimands=("compliance",))
    assert exc_info.value.code == "readout.assignment.estimands"


def test_asof_lift_refused_for_observational_design():
    from increment.semantics.design import AdjustmentSet, Observational

    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x1",)))
    src = FakeMomentSource([], metrics=[METRIC], capabilities={"asof"}, design=design)
    with pytest.raises(UnsupportedRequestError) as exc_info:
        readouts.asof_lift(src)
    assert exc_info.value.code == "readout.view.observational"


class _LegacySignatureMomentSource:
    """Proxy predating operation declarations and CUPED moment keywords."""

    def __init__(self, source: FakeMomentSource):
        self._source = source

    def moments(self, metric, *, grain="total", by=(), completed_windows_only=False):
        return self._source.moments(
            metric,
            grain=grain,
            by=by,
            completed_windows_only=completed_windows_only,
        )

    def __getattr__(self, name):
        if name == "operations":
            raise AttributeError(name)
        return getattr(self._source, name)


def test_unadjusted_daily_and_asof_lift_support_legacy_moment_source():
    design = Randomized(control_group="control")
    rows = [dict(r, ds=date(2025, 1, 1)) for r in _rows(n=100, seed=32)]
    source = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"daily", "asof"},
        design=design,
    )
    source.shape = "unit_panel"
    src = cast("MomentSource", _LegacySignatureMomentSource(source))
    with pytest.raises(AttributeError):
        _ = src.operations

    assert readouts.daily(src)
    assert readouts.asof_lift(src)

    analysis = Analysis._from_source(src, design)
    assert analysis.run_daily_lift()
    assert analysis.run_asof_lift()


# - breakout(): per-segment LATE gating -------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("include_sensitivity", [False, True])
def test_encouragement_breakout_methods_are_unique_per_metric(include_sensitivity):
    frame = pa.table(
        {
            "unit": range(80),
            "arm": ["control"] * 40 + ["treat"] * 40,
            "day": [0] * 80,
            "segment": ["A" if i % 40 < 20 else "B" for i in range(80)],
            "y": [float(2 + i % 3) for i in range(80)],
            "z": [float(3 + i % 5) for i in range(80)],
            "help_click": [0.0] * 40 + [float(i % 4 != 0) for i in range(40)],
        }
    )
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit",
        group="arm",
        date="day",
        design=_design(one_sided=True),
        metrics={"y": "mean", "z": "mean"},
        breakouts=["segment"],
    )
    sensitivity = [Method(name="sensitivity")] if include_sensitivity else []
    combined = analysis.run_breakout(sensitivity_methods=sensitivity)
    separate = [
        row
        for metric in ("y", "z")
        for row in analysis.run_breakout(metrics=[metric], sensitivity_methods=sensitivity)
    ]

    def identity(row):
        return (
            row.metric,
            row.group_id,
            row.dimension_value,
            row.method,
            row.method_role,
            row.estimand,
            row.value_scale,
        )

    assert sorted(map(identity, combined)) == sorted(map(identity, separate))
    assert {row.metric for row in combined} == {"y", "z"}
    assert {row.estimand for row in combined} >= {"itt", "compliance", "late"}
    assert {row.method_role for row in combined} == (
        {"decision", "sensitivity"} if include_sensitivity else {"decision"}
    )
    for actual, expected in zip(
        sorted(combined, key=identity), sorted(separate, key=identity), strict=True
    ):
        assert (
            actual.family_axes,
            actual.discovery,
            actual.reference_kind,
            actual.n_control,
            actual.n_treat,
        ) == (
            expected.family_axes,
            expected.discovery,
            expected.reference_kind,
            expected.n_control,
            expected.n_treat,
        )
        assert (
            actual.family_q,
            actual.family_threshold,
            actual.reference_df,
            actual.abs_diff,
            actual.abs_se,
        ) == pytest.approx(
            (
                expected.family_q,
                expected.family_threshold,
                expected.reference_df,
                expected.abs_diff,
                expected.abs_se,
            )
        )
        assert actual.lift is not None and expected.lift is not None
        assert (
            actual.lift.value,
            actual.lift.lb,
            actual.lift.ub,
            actual.lift.level,
        ) == pytest.approx(
            (expected.lift.value, expected.lift.lb, expected.lift.ub, expected.lift.level)
        )
    with pytest.raises(InvalidRequestError) as duplicate:
        analysis.run_breakout(
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=[Method(name="unadjusted")],
        )
    assert duplicate.value.code == "estimation.engine.method_names_unique"


def test_breakout_emits_late_for_strong_segment_suppresses_weak():
    design = _design(one_sided=True)
    strong = _rows(n=4000, compliance=0.5, seed=21)
    weak = _rows(n=200, compliance=0.02, seed=22)
    for r in strong:
        r["cohort"] = "strong"
    for r in weak:
        r["cohort"] = "weak"
    src = FakeMomentSource(
        strong + weak,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    results = readouts.breakout(src, "cohort", correction="none")

    by_segment: defaultdict[str, set[str]] = defaultdict(set)
    for r in results:
        assert r.dimension == "cohort"
        assert r.metric == METRIC.name
        assert r.source is None
        by_segment[r.dimension_value].add(r.estimand)
    assert by_segment["strong"] >= {"itt", "compliance", "late"}
    assert "late" not in by_segment["weak"]


def test_breakout_cuped_late_fits_theta_per_segment():
    """Each segment's cuped LATE uses that segment's OWN pooled theta, the
    same per-pair rule the CUPED-ITT breakout already follows - so two
    segments with different covariate-outcome relationships get different
    adjustments rather than one borrowed number."""
    from increment.estimation.engine import Method

    rng = np.random.default_rng(918)
    rows = []
    for cohort, rho in (("habitual", 0.9), ("fickle", 0.05)):
        for gid, encouraged in (("control", 0), ("treat", 1)):
            n = 3000
            x = rng.normal(10.0, 2.0, size=n)
            d = rng.binomial(1, 0.5 if encouraged else 0.08, size=n).astype(float)
            y = 10.0 + rho * (x - 10.0) + 2.0 * d + rng.normal(0, 1.0, size=n)
            rows.append(
                centered_row_from_raw_sums(
                    {
                        "experiment_id": "s",
                        "metric": "rev",
                        "group_id": gid,
                        "cohort": cohort,
                        "n": n,
                        "sum_y": float(y.sum()),
                        "sum_y2": float((y**2).sum()),
                        "sum_x": float(x.sum()),
                        "sum_x2": float((x**2).sum()),
                        "sum_xy": float((x * y).sum()),
                        "sum_d": float(d.sum()),
                        "sum_yd": float((y * d).sum()),
                        "sum_y2d": float((y**2 * d).sum()),
                        "sum_xd": float((x * d).sum()),
                    }
                )
            )
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"total"},
        design=_design(one_sided=False),
        breakouts=("cohort",),
    )

    def _widths(methods):
        kwargs = (
            {}
            if methods is None
            else {
                "decision_method": methods[0],
                "sensitivity_methods": tuple(methods[1:]),
            }
        )
        rows = readouts.breakout(src, "cohort", estimands=("late",), correction="none", **kwargs)  # ty: ignore[invalid-argument-type]
        out = {}
        for r in rows:
            if r.estimand == "late" and r.value_scale == "absolute":
                lift = r.lift
                assert lift is not None
                assert lift.lb is not None and lift.ub is not None
                out[r.dimension_value] = lift.ub - lift.lb
        return out

    plain = _widths(None)
    adjusted = _widths([Method(name="cuped", variance_reduction="cuped")])
    assert set(adjusted) == {"habitual", "fickle"}

    # theta ~ 0.9 in the habitual segment, ~ 0 in the fickle one: a single
    # borrowed theta could not produce both outcomes.
    assert adjusted["habitual"] < 0.75 * plain["habitual"]
    assert adjusted["fickle"] == pytest.approx(plain["fickle"], rel=0.05)


# - srm(): applies under Encouragement (assignment IS randomized) --------


def test_srm_runs_under_encouragement_design_not_suppressed():
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves revenue via uptake"
        ),
        allocation={"control": 0.5, "treatment": 0.5},
    )
    src = from_unit_summary(
        _uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    result = readouts.srm(src)
    # Balanced allocation must not itself be flagged - a real SRMResult
    # should come back (encouragement assignment is randomized), not NotApplicable.
    from increment.estimation.diagnostics import NotApplicable, SRMResult

    assert isinstance(result, SRMResult)
    assert not isinstance(result, NotApplicable)


def test_breakout_rejects_estimands_on_randomized_design():
    design = Randomized(control_group="control")
    src = FakeMomentSource([], metrics=[METRIC], capabilities={"total"}, design=design)
    with pytest.raises(InvalidRequestError) as exc_info:
        readouts.breakout(src, "cohort", estimands=("late",))
    assert exc_info.value.code == "readout.assignment.estimands"


def test_srm_still_not_applicable_under_observational_design():
    from increment.semantics.design import AdjustmentSet, Observational

    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x1",)))
    src = FakeMomentSource([], metrics=[METRIC], capabilities={"total"}, design=design)
    result = readouts.srm(src)
    from increment.estimation.diagnostics import NotApplicable

    assert isinstance(result, NotApplicable)


# - breakout(): estimands= narrowing --------------------------------------


def test_breakout_honors_estimands():
    from increment.breakout.estimates import BreakoutEstimate

    design = _design(one_sided=True)
    strong = _rows(n=4000, compliance=0.5, seed=31)
    for r in strong:
        r["cohort"] = "strong"
    src = FakeMomentSource(
        strong,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    narrowed = cast(
        "list[BreakoutEstimate]",
        readouts.breakout(src, "cohort", estimands=("itt",), correction="none"),
    )
    assert {r.estimand for r in narrowed} == {"itt"}

    full = cast("list[BreakoutEstimate]", readouts.breakout(src, "cohort", correction="none"))
    assert {r.estimand for r in full} >= {"itt", "compliance", "late"}


def test_breakout_rows_carry_absolute_axis_and_arm_counts():
    """The encouragement branch mirrors the randomized breakout's field
    set: abs_diff/abs_se (the absolute-scale inputs heterogeneity reads),
    per-arm counts, and the low-sample reliability flag."""
    design = _design(one_sided=True)
    big = _rows(n=4000, compliance=0.5, seed=21)
    small = _rows(n=30, compliance=0.5, seed=22)
    for r in big:
        r["cohort"] = "big"
    for r in small:
        r["cohort"] = "small"
    src = FakeMomentSource(
        big + small,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    results = readouts.breakout(src, "cohort", estimands=("itt",), correction="none")

    by_segment = {r.dimension_value: r for r in results if r.estimand == "itt"}
    assert set(by_segment) == {"big", "small"}
    for r in by_segment.values():
        assert r.abs_diff is not None
        assert r.abs_se is not None
    b, s = by_segment["big"], by_segment["small"]
    assert (b.n_treat, b.n_control, b.low_reliability) == (4000.0, 4000.0, False)
    assert (s.n_treat, s.n_control, s.low_reliability) == (30.0, 30.0, True)

    # compliance/late rows keep abs_diff=None (their effect is `lift`) but
    # resolve n_treat/n_control/low_reliability through the group_id-keyed arm
    # map, so their rows must carry the real treatment arm's group_id.
    full_results = readouts.breakout(src, "cohort", correction="none")
    for estimand in ("compliance", "late"):
        other_by_segment = {r.dimension_value: r for r in full_results if r.estimand == estimand}
        assert set(other_by_segment) == {"big", "small"}
        ob, os_ = other_by_segment["big"], other_by_segment["small"]
        assert ob.abs_diff is None
        assert (ob.n_treat, ob.n_control, ob.low_reliability) == (4000.0, 4000.0, False)
        assert os_.abs_diff is None
        assert (os_.n_treat, os_.n_control, os_.low_reliability) == (30.0, 30.0, True)


def test_segment_heterogeneity_reads_encouragement_absolute_axis():
    """abs_diff/abs_se on the rows unlock the absolute-scale pass: an
    encouragement breakout yields BOTH heterogeneity scales, like a
    randomized one."""
    from increment.breakout.heterogeneity import segment_heterogeneity

    design = _design(one_sided=True)
    rows = []
    for cohort, seed in (("a", 21), ("b", 22), ("c", 23)):
        for r in _rows(n=4000, compliance=0.5, seed=seed):
            r["cohort"] = cohort
            rows.append(r)
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    het = segment_heterogeneity(
        readouts.breakout(src, "cohort", estimands=("itt",), correction="none")
    )

    assert {s.scale for s in het.summary} == {"relative", "absolute"}


def test_segment_heterogeneity_default_estimands_keep_itt_family_clean():
    """Under the default estimand set the breakout emits compliance/late rows
    alongside ITT; they must form their own heterogeneity families instead of
    blanking the ITT family's tau2/i2 as outcome exclusions. 5 cohorts (not
    the original 3) so ``i2`` clears cochran_q's k>=5 reporting floor."""
    from increment.breakout.heterogeneity import segment_heterogeneity

    design = _design(one_sided=True)
    rows = []
    for cohort, seed in (("a", 21), ("b", 22), ("c", 23), ("d", 24), ("e", 25)):
        for r in _rows(n=4000, compliance=0.5, seed=seed):
            r["cohort"] = cohort
            rows.append(r)
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    breakout_result = readouts.breakout(src, "cohort", correction="none")
    # Confirms this test's own precondition: the default estimand set really
    # does emit non-ITT rows, not just ITT ones.
    assert any(r.estimand == "compliance" for r in breakout_result)
    assert any(r.estimand == "late" for r in breakout_result)

    het = segment_heterogeneity(breakout_result)

    itt_abs = [s for s in het.summary if s.estimand == "itt" and s.scale == "absolute"]
    assert len(itt_abs) == 1
    assert itt_abs[0].n_excluded_outcome == 0
    assert itt_abs[0].tau2 is not None
    assert itt_abs[0].i2 is not None


def test_segment_rollout_default_estimands_keep_itt_family_clean():
    """Under the default estimand set the breakout emits compliance/late rows
    alongside ITT; the rollout recommendation must not count them as the ITT
    family's outcome exclusions, which would understate its coverage."""
    from increment.breakout.rollout import segment_rollout_recommendation

    design = _design(one_sided=True)
    rows = []
    for cohort, seed in (("a", 21), ("b", 22), ("c", 23)):
        for r in _rows(n=4000, compliance=0.5, seed=seed):
            r["cohort"] = cohort
            rows.append(r)
    src = FakeMomentSource(
        rows,
        metrics=[METRIC],
        capabilities={"total"},
        design=design,
        breakouts=("cohort",),
    )

    breakout_result = readouts.breakout(src, "cohort", correction="none")
    # Confirms this test's own precondition: the default estimand set really
    # does emit non-ITT rows, not just ITT ones.
    assert any(r.estimand == "late" for r in breakout_result)

    recommendations, segments = segment_rollout_recommendation(breakout_result)

    itt = [r for r in recommendations if r.metric == "rev" and r.estimand == "itt"]
    assert len(itt) == 1
    assert (itt[0].estimand, itt[0].value_scale) == ("itt", "relative")
    assert itt[0].n_excluded_outcome == 0
    assert itt[0].k == 3
    assert all(s.estimand == "itt" for s in segments if s.metric == "rev" and s.estimand == "itt")


@pytest.fixture(scope="module")
def con():
    import ibis

    return ibis.duckdb.connect()


def _guardrail_analysis(con):
    if "guardrail_late_events" not in con.list_tables():
        control_units = [f"c{i}" for i in range(1, 7)]
        treat_units = [f"t{i}" for i in range(1, 7)]
        clickers = {"t1", "t2", "t3", "t4"}

        exposure_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "control",
                "revenue": None,
                "errors": None,
                "experiment_id": "guardrail_late_exp",
            }
            for u in control_units
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "treatment",
                "revenue": None,
                "errors": None,
                "experiment_id": "guardrail_late_exp",
            }
            for u in treat_units
        ]
        click_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "errors": None,
                "experiment_id": None,
            }
            for u in clickers
        ]
        revenue_by_unit = {
            "c1": 8.0,
            "c2": 9.0,
            "c3": 10.0,
            "c4": 11.0,
            "c5": 9.0,
            "c6": 11.0,
            "t1": 13.0,
            "t2": 14.0,
            "t3": 15.0,
            "t4": 16.0,
            "t5": 9.0,
            "t6": 11.0,
        }
        purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "errors": None,
                "experiment_id": None,
            }
            for u, amount in revenue_by_unit.items()
        ]
        errors_by_unit = {
            "c1": 2,
            "c2": 3,
            "c3": 2,
            "c4": 3,
            "c5": 2,
            "c6": 3,
            "t1": 2,
            "t2": 3,
            "t3": 3,
            "t4": 2,
            "t5": 3,
            "t6": 2,
        }
        error_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "error_event",
                "group_id": None,
                "revenue": None,
                "errors": float(cnt),
                "experiment_id": None,
            }
            for u, cnt in errors_by_unit.items()
        ]
        con.create_table(
            "guardrail_late_events",
            obj=exposure_rows + click_rows + purchase_rows + error_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM guardrail_late_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "error_event", "column": "errors"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "errors",
                    "entity": "user_id",
                    "fact": "error_event",
                    "aggregation": "sum",
                    "preferred_direction": "decrease",
                },
            ],
            "experiments": [
                {
                    "name": "guardrail_late_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"], "guardrails": ["errors"]},
                }
            ],
        }
    )

    analysis = make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves revenue via uptake"
            ),
            one_sided=True,
            min_first_stage_z=0.001,
        ),
    )
    return analysis


def _multi_arm_guardrail_analysis(con):
    """Same shape as `_guardrail_analysis`, but with a SECOND treatment
    arm - `_guardrail_analysis` is 2-arm (control/treatment), so
    `n_arms == 1` makes the primary's `alpha_share / n_arms` division a
    no-op and the multi-arm split path is never numerically exercised."""
    if "multi_arm_guardrail_events" not in con.list_tables():
        control_units = [f"c{i}" for i in range(1, 7)]
        treat_a_units = [f"a{i}" for i in range(1, 7)]
        treat_b_units = [f"b{i}" for i in range(1, 7)]
        clickers = {"a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"}

        def _exposure_rows(units, group):
            return [
                {
                    "user_id": u,
                    "ts": datetime(2025, 1, 1, 9, 0, 0),
                    "event": "exposed",
                    "group_id": group,
                    "revenue": None,
                    "errors": None,
                    "experiment_id": "multi_arm_guardrail_exp",
                }
                for u in units
            ]

        exposure_rows = (
            _exposure_rows(control_units, "control")
            + _exposure_rows(treat_a_units, "a")
            + _exposure_rows(treat_b_units, "b")
        )
        click_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "errors": None,
                "experiment_id": None,
            }
            for u in clickers
        ]
        revenue_by_unit = {
            "c1": 8.0,
            "c2": 9.0,
            "c3": 10.0,
            "c4": 11.0,
            "c5": 9.0,
            "c6": 11.0,
            "a1": 13.0,
            "a2": 14.0,
            "a3": 15.0,
            "a4": 16.0,
            "a5": 9.0,
            "a6": 11.0,
            "b1": 12.0,
            "b2": 13.0,
            "b3": 14.0,
            "b4": 15.0,
            "b5": 10.0,
            "b6": 11.0,
        }
        purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "errors": None,
                "experiment_id": None,
            }
            for u, amount in revenue_by_unit.items()
        ]
        errors_by_unit = {
            u: 2.0 + (i % 2) for i, u in enumerate(control_units + treat_a_units + treat_b_units)
        }
        error_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "error_event",
                "group_id": None,
                "revenue": None,
                "errors": cnt,
                "experiment_id": None,
            }
            for u, cnt in errors_by_unit.items()
        ]
        con.create_table(
            "multi_arm_guardrail_events",
            obj=exposure_rows + click_rows + purchase_rows + error_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM multi_arm_guardrail_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "error_event", "column": "errors"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "errors",
                    "entity": "user_id",
                    "fact": "error_event",
                    "aggregation": "sum",
                    "preferred_direction": "decrease",
                },
            ],
            "experiments": [
                {
                    "name": "multi_arm_guardrail_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"], "guardrails": ["errors"]},
                }
            ],
        }
    )

    return make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves revenue via uptake"
            ),
            one_sided=True,
            min_first_stage_z=0.001,
        ),
    )


def test_analysis_run_native_encouragement_declared_plan_primary_splits_alpha_by_arms(con):
    """A declared primary's alpha_share must further split across
    TREATMENT arms (`alpha_share / n_arms`), not just across primaries -
    the sibling clustered/quantile tasks caught the identical
    single-arm-fixture gap by adding a second treatment arm; this pins
    the same arithmetic for the encouragement branch."""
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="revenue", guardrails=["errors"])}
        ),
    )

    results = lift_rows(bound.run())
    assert {r.group_id for r in results if r.metric == "revenue"} == {"a", "b"}

    revenue_rows = [r for r in results if r.metric == "revenue"]
    assert revenue_rows
    assert all(r.role == "primary" for r in revenue_rows)
    # 1 primary, 2 treatment arms: alpha_share/n_arms = 0.05/2 = 0.025 -> level 0.975.
    assert all(r.require_lift().level == pytest.approx(0.975) for r in revenue_rows)


def test_analysis_run_native_encouragement_declared_plan_secondary_at_nominal_alpha(con):
    """A declared secondary runs at the plan's own nominal alpha (never split
    across arms or primaries), and now gets a real BH/e-BH family verdict on
    its ITT row - the family is one cell per (metric, arm) with ITT as the
    representative, so compliance and late keep `discovery=None` because
    neither was the tested hypothesis."""
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="errors", secondaries=["revenue"])}
        ),
    )

    results = lift_rows(bound.run())
    revenue_rows = [r for r in results if r.metric == "revenue"]
    assert revenue_rows
    assert all(r.role == "secondary" for r in revenue_rows)

    itt_rows = [r for r in revenue_rows if r.estimand == "itt"]
    assert itt_rows, "the ITT row is the family's representative cell"
    for r in itt_rows:
        assert r.discovery is not None, "the tested cell must carry a verdict"
        assert r.family_axes == ("metric", "arm")
        assert r.family_q == pytest.approx(0.10)

    for r in revenue_rows:
        if r.estimand != "itt":
            assert r.discovery is None, (
                f"{r.estimand} was never tested by the family, so it must not "
                "claim a discovery verdict"
            )

    compliance_rows = [r for r in results if r.estimand == "compliance"]
    assert compliance_rows
    assert all(r.discovery is None for r in compliance_rows)

    # Nominal plan.alpha (0.05), not split across arms or primaries -> level
    # 0.95. A lone in-family secondary self-selects, so BH's own cutoff is
    # q=0.10, which the nominal-alpha cap holds back to 0.05 -- the level is
    # unchanged either way.
    assert all(r.require_lift().level == pytest.approx(0.95) for r in revenue_rows)


def test_encouragement_missing_control_defers_to_family_gate(con):
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="errors", secondaries=["revenue"])}
        ),
    )
    src = _native_source(bound)
    metrics = tuple(src.context.metrics)
    rows_by_metric: dict[str, list[Mapping[str, Any]]] = {
        metric.name: list(src.moments(metric, grain="total")) for metric in metrics
    }
    rows_by_metric["revenue"] = [
        row for row in rows_by_metric["revenue"] if str(row["group_id"]) != "control"
    ]
    with pytest.raises(CapabilityError) as raised:
        encouragement_rows(
            src=src,
            metrics=metrics,
            rows_by_metric=rows_by_metric,
            configs=src.context.configs,
            design=cast("Encouragement", src.context.design),
            plan=src.context.plan,
            estimands=None,
            cluster=None,
            caller="test",
        )
    assert raised.value.code == "family.evidence.incomplete"
    assert any("revenue" in repr(key) for key in raised.value.context["failed"])  # ty: ignore[not-iterable]


def test_late_only_family_omits_itt_keys(con):
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="errors", secondaries=["revenue"])}
        ),
    )
    revenue = [row for row in lift_rows(bound.run(estimands=("late",))) if row.metric == "revenue"]
    assert revenue
    assert {row.estimand for row in revenue} == {"late"}
    assert all(row.discovery is None for row in revenue)


def test_analysis_run_native_encouragement_family_spans_metric_and_arm(con):
    """The finding's repro: several secondaries under an encouragement design
    used to come back with `discovery=None` on every row regardless of effect
    size, while the identical plan under a randomized design produced real
    BH verdicts. The family is one cell per (metric, arm) - two metrics
    across two treatment arms is four cells, not four times the estimand
    count - and every one of them now carries a verdict."""
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            # ~11 units per arm leave the Welch reference under 10 dof, so
            # p-values run several times the Normal ones (revenue: 0.0094 and
            # 0.0145, not 0.0013 and 0.0022); q separates the metrics at these values.
            update={"plan": AnalysisPlan(q=0.03, secondaries=["revenue", "errors"])}
        ),
    )

    results = lift_rows(bound.run())
    itt_rows = [r for r in results if r.estimand == "itt" and r.metric in {"revenue", "errors"}]
    assert {(r.metric, r.group_id) for r in itt_rows} == {
        ("revenue", "a"),
        ("revenue", "b"),
        ("errors", "a"),
        ("errors", "b"),
    }
    for r in itt_rows:
        assert r.role == "secondary"
        assert r.discovery is not None, "every family cell must carry a verdict"
        assert r.family_axes == ("metric", "arm")
        assert r.family_q == pytest.approx(0.03)
    # The verdicts must actually discriminate, or a stamped-everything bug
    # would pass: revenue moves in both arms, errors moves in neither
    # (p = 1), so the family splits along the metric.
    by_cell = {(r.metric, r.group_id): r.discovery for r in itt_rows}
    assert by_cell == {
        ("revenue", "a"): True,
        ("revenue", "b"): True,
        ("errors", "a"): False,
        ("errors", "b"): False,
    }

    # Selected revenue cells are re-estimated at FCR level 1 - R*q/m =
    # 1 - 2*0.03/4 = 0.985; unselected errors cells must retain nominal level
    # 0.95, not consume a sibling's re-estimate merely because their method
    # row shares the family.
    assert all(
        r.require_lift().level == pytest.approx(0.985) for r in itt_rows if r.metric == "revenue"
    )
    assert all(
        r.require_lift().level == pytest.approx(0.95) for r in itt_rows if r.metric == "errors"
    )


def test_selected_secondary_late_row_uses_corrected_level(con):
    """A selected secondary's LATE row must carry the same FCR-corrected
    level as its ITT row -- both test the same null, and a reader
    comparing the two intervals should not find different guarantees."""
    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(q=0.03, secondaries=["revenue", "errors"])}
        ),
    )

    results = lift_rows(bound.run())

    itt_rows = [r for r in results if r.metric == "revenue" and r.estimand == "itt"]
    late_rows = [r for r in results if r.metric == "revenue" and r.estimand == "late"]
    assert itt_rows and late_rows
    by_arm_itt = {r.group_id: r.require_lift().level for r in itt_rows}
    by_arm_late = {r.group_id: r.require_lift().level for r in late_rows}
    assert by_arm_itt.keys() == by_arm_late.keys()
    for arm in by_arm_itt:
        assert by_arm_late[arm] == pytest.approx(by_arm_itt[arm]), (
            f"revenue/{arm}: itt level {by_arm_itt[arm]} != late level "
            f"{by_arm_late[arm]} -- LATE is a derived presentation of the "
            "same tested null and must carry the same corrected level"
        )
        assert by_arm_itt[arm] == pytest.approx(0.985), (
            "sanity: revenue was actually selected and re-estimated at "
            "the FCR level, not left at nominal 0.95"
        )


def test_selected_secondary_late_row_keeps_both_value_scales_at_corrected_level():
    """A selected secondary's LATE estimand emits two rows -- relative
    (complier-ratio) and absolute (additive) -- sharing the same
    (metric, group_id, method, estimand="late") tuple and differing only
    in ``value_scale``. ``_select_encouragement_family``'s re-estimation
    cache (``increment.readouts._encouragement``) must key on ``value_scale`` too: without it, the
    second late row's ``pass_results`` entry would silently overwrite the
    first in ``reestimated``, so BOTH original late rows would look up
    and receive the SAME single re-estimated object -- one value_scale
    would vanish entirely and the surviving one would be duplicated onto
    both rows. Assert both scales survive distinctly with their own lift,
    both land on the FCR-corrected level (read off the sibling ITT row,
    not re-derived from the FCR formula), both keep the family's
    ``role="secondary"`` stamp, and neither claims a discovery verdict --
    LATE was never the tested hypothesis."""
    design = _design(one_sided=True)
    other = METRIC.model_copy(update={"name": "other"})
    # rev moves strongly (tau=2.0 against sd=2.0, n=4000/arm, compliance=0.5)
    # so it is the one cell BH selects; other is a flat null so it is not,
    # keeping the family's R/m split (and the realized FCR level) legible.
    selected_rows = _rows(n=4000, tau=2.0, compliance=0.5, seed=7, metric="rev")
    unselected_rows = _rows(n=4000, tau=0.0, compliance=0.5, seed=8, metric="other")
    plan = AnalysisPlan(q=0.03, secondaries=["rev", "other"])
    src = FakeMomentSource(
        selected_rows + unselected_rows,
        metrics=[METRIC, other],
        capabilities={"total"},
        design=design,
        plan=plan,
    )

    results = readouts.run(src)
    late_rows = [r for r in results if r.metric == "rev" and r.estimand == "late"]
    itt_rows = [r for r in results if r.metric == "rev" and r.estimand == "itt"]
    assert itt_rows, "sanity: rev's ITT row must exist to compare against"
    assert all(r.discovery for r in itt_rows), "sanity: rev must actually be the selected cell"
    itt_level = {r.group_id: r.require_lift().level for r in itt_rows}

    by_arm_scale = {(r.group_id, r.value_scale): r for r in late_rows}
    assert {r.group_id for r in late_rows} == itt_level.keys()
    for arm in itt_level:
        # A value_scale-blind cache key would collapse these to one row.
        assert (arm, "relative") in by_arm_scale, (
            f"rev/{arm}: missing the relative (complier-ratio) LATE row"
        )
        assert (arm, "absolute") in by_arm_scale, (
            f"rev/{arm}: missing the absolute (additive) LATE row"
        )
        rel_row = by_arm_scale[(arm, "relative")]
        abs_row = by_arm_scale[(arm, "absolute")]
        # A value_scale-blind cache key would also make both rows read back
        # the same last-inserted re-estimated object -- they must carry
        # their own distinct lift.
        assert rel_row.require_lift().value != pytest.approx(abs_row.require_lift().value), (
            f"rev/{arm}: relative and absolute LATE rows carry the same lift "
            "value -- the re-estimation cache collapsed the two scales"
        )
        for row in (rel_row, abs_row):
            assert row.role == "secondary"
            assert row.discovery is None, (
                "LATE was never the tested hypothesis -- only the ITT row "
                "carries a discovery verdict"
            )
            assert row.require_lift().level == pytest.approx(itt_level[arm]), (
                f"rev/{arm}/{row.value_scale}: LATE level {row.require_lift().level} "
                f"!= ITT's corrected level {itt_level[arm]}"
            )


def test_guardrail_metric_reports_itt_and_late_via_analysis_run(con):
    """A guardrail's estimand set matches every other metric's: full
    itt/compliance/late by default, identical to what the dataframe,
    artifact and moments routes report for a guardrail under an
    Encouragement design."""
    analysis = _guardrail_analysis(con)

    results = lift_rows(analysis.run())

    by_base_metric: defaultdict[str, set[str]] = defaultdict(set)
    for r in results:
        by_base_metric[r.metric.removesuffix("_uptake")].add(r.estimand)

    assert by_base_metric["revenue"] >= {"itt", "late"}
    assert by_base_metric["errors"] >= {"itt", "late"}, (
        "a guardrail metric reports late alongside itt, exactly like a "
        "secondary -- its SAFE/HARM verdict is computed from the itt row "
        "regardless"
    )
    compliance_rows = [r for r in results if r.estimand == "compliance"]
    assert [r.metric for r in compliance_rows] == ["uptake"], (
        "uptake stays a single canonical row per treatment arm"
    )


def test_analysis_run_native_groups_encouragement_dispatch_by_declared_methods(con):
    """A declared method-role binding on one metric splits the combined
    summary table into separate ``estimate_encouragement`` calls per group
    - both groups must still report full, correct results, proving the
    per-group ``metric`` column filter does not drop or corrupt rows.
    Declares the SAME method ("unadjusted") the design default already
    uses, so a correct split reproduces the unbound baseline exactly."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    baseline = lift_rows(_guardrail_analysis(con).run())

    bound = _guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="unadjusted"),
                        )
                    ],
                    guardrails=["errors"],
                ),
            }
        ),
    )
    grouped = lift_rows(bound.run())

    def keys(results):
        return {(r.metric, r.group_id, r.estimand): r.require_lift().value for r in results}

    assert len(grouped) == len(baseline), (
        f"splitting into per-config groups changed row count: "
        f"{len(grouped)} vs {len(baseline)} -- duplicate or dropped row"
    )
    assert keys(grouped) == keys(baseline), (
        "splitting into per-config groups must not change any estimate"
    )


def test_analysis_run_native_grouped_encouragement_reports_one_uptake_row_per_arm(con):
    """Uptake/compliance is metric-independent - splitting dispatch into
    per-config groups must not multiply the canonical `uptake` row.
    Regression: grouped dispatch used to emit one duplicate `uptake`
    row per group."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    bound = _guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="unadjusted"),
                        )
                    ],
                    guardrails=["errors"],
                ),
            }
        ),
    )
    results = lift_rows(bound.run())
    uptake_keys = [(r.group_id, r.estimand) for r in results if r.metric == "uptake"]
    assert len(uptake_keys) == len(set(uptake_keys)), (
        f"duplicate uptake row(s) across groups: {uptake_keys}"
    )


def test_guardrail_estimands_narrowing_never_returns_unrequested_estimand(con):
    """Narrowing run(estimands=...) never returns an estimand the caller
    did not ask for, for a guardrail or any other metric."""
    analysis = _guardrail_analysis(con)

    results = lift_rows(analysis.run(metrics=["errors"], estimands=("late",)))

    assert results, "a late-only request for a guardrail still returns its late row"
    assert {r.estimand for r in results} == {"late"}


def _uptake_named_secondary_analysis(con):
    """Two in-family secondaries with symmetric roles, one of them
    literally named ``uptake`` -- the outcome-metric collision surface
    for defect 1 (``increment.readouts._encouragement`` identifying the design-level compliance
    row by ``metric == "uptake"`` instead of ``estimand == "compliance"``).

    The plan declares ``q=0.02`` (below the default 0.10) so that, with
    both secondaries selected (R=2, m=2), the realized BH/FCR cutoff
    ``R*q/m == q == 0.02`` lands strictly under the nominal ``alpha=0.05``
    and is used uncapped. That makes the re-estimated interval level
    (0.98) observably different from nominal (0.95) -- the default
    ``q=0.10`` cutoff (0.10) exceeds nominal alpha and gets capped back
    to it, so a metric-string collision that silently fell back to the
    canonical uptake pass's own (also-nominal) level would go unnoticed.
    """
    if "outcome_named_uptake_events" not in con.list_tables():
        control_units = [f"c{i}" for i in range(1, 7)]
        treat_units = [f"t{i}" for i in range(1, 7)]
        clickers = {"t1", "t2", "t3", "t4"}

        exposure_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "control",
                "revenue": None,
                "conv": None,
                "experiment_id": "outcome_named_uptake_exp",
            }
            for u in control_units
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "treatment",
                "revenue": None,
                "conv": None,
                "experiment_id": "outcome_named_uptake_exp",
            }
            for u in treat_units
        ]
        click_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "conv": None,
                "experiment_id": None,
            }
            for u in clickers
        ]
        revenue_by_unit = {
            "c1": 8.0,
            "c2": 9.0,
            "c3": 10.0,
            "c4": 11.0,
            "c5": 9.0,
            "c6": 11.0,
            "t1": 13.0,
            "t2": 14.0,
            "t3": 15.0,
            "t4": 16.0,
            "t5": 9.0,
            "t6": 11.0,
        }
        purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "conv": None,
                "experiment_id": None,
            }
            for u, amount in revenue_by_unit.items()
        ]
        conv_by_unit = {
            "c1": 9.0,
            "c2": 10.0,
            "c3": 9.0,
            "c4": 10.0,
            "c5": 9.0,
            "c6": 10.0,
            "t1": 14.0,
            "t2": 15.0,
            "t3": 14.0,
            "t4": 15.0,
            "t5": 10.0,
            "t6": 9.0,
        }
        conv_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "conv_event",
                "group_id": None,
                "revenue": None,
                "conv": amount,
                "experiment_id": None,
            }
            for u, amount in conv_by_unit.items()
        ]
        con.create_table(
            "outcome_named_uptake_events",
            obj=exposure_rows + click_rows + purchase_rows + conv_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM outcome_named_uptake_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "conv_event", "column": "conv"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                # A real outcome metric literally named "uptake" -- the same
                # string `_compliance_metric` uses for the design-level
                # compliance diagnostic's synthetic `LiftEstimate.metric`.
                {
                    "type": "mean",
                    "name": "uptake",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": "conversion",
                    "entity": "user_id",
                    "fact": "conv_event",
                    "aggregation": "sum",
                },
            ],
            "experiments": [
                {
                    "name": "outcome_named_uptake_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["uptake", "conversion"], "q": 0.02},
                }
            ],
        }
    )

    return make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves the outcome via uptake"
            ),
            one_sided=True,
            min_first_stage_z=0.001,
        ),
    )


def test_outcome_metric_named_uptake_keeps_its_own_role_and_family_verdict(con):
    """Defect 1 regression (``increment.readouts._encouragement``): a real outcome metric
    legitimately named ``uptake`` shares its ``LiftEstimate.metric`` string with the
    design-level compliance diagnostic, which `encouragement_rows` and
    `_select_encouragement_family` used to identify via
    ``r.metric == "uptake"``. That collision used to (a) drop the real
    metric's itt/late rows out of their own per-metric estimation group,
    (b) re-derive them from the *canonical uptake* pass instead -- wrong
    alpha/alternative, and a hardcoded ``role=None`` that only the
    downstream family stamp (itt-only) papered back over -- and (c) leave
    the mis-stamped late row's role permanently wrong. Fixed by keying off
    ``estimand == "compliance"`` instead, which never collides with a real
    metric's name.

    The fixture's plan declares ``q=0.02`` so the family's realized FCR
    cutoff (``R*q/m == 0.02``, both secondaries selected) lands strictly
    under nominal ``alpha=0.05`` and is used uncapped, giving the
    re-estimated rows a level (0.98) that provably differs from nominal
    (0.95). Asserting the uptake-named row's level against its
    "conversion" peer's level is what actually catches a metric-string
    collision silently keeping the uptake-named row at the (also-nominal,
    unre-estimated) canonical compliance pass's level instead.
    """
    analysis = _uptake_named_secondary_analysis(con)

    results = lift_rows(analysis.run())

    by_key = {(r.metric, r.estimand): r for r in results}

    # The real "uptake" outcome's own itt/late rows must exist, carry the
    # secondary role, and get a real family verdict -- not silently merged
    # into (or replaced by) the canonical compliance diagnostic.
    uptake_itt = by_key[("uptake", "itt")]
    assert uptake_itt.role == "secondary"
    assert uptake_itt.discovery in (True, False)
    uptake_late = by_key[("uptake", "late")]
    assert uptake_late.role == "secondary", (
        "the real uptake-named metric's late row must be stamped from its "
        "own per-metric group, not silently defaulted by the canonical "
        "compliance pass"
    )

    # Its twin secondary (no colliding name) must be treated identically --
    # same role, same verdict shape -- proving the "uptake" name itself
    # changed nothing about how its own rows were estimated.
    conversion_itt = by_key[("conversion", "itt")]
    assert conversion_itt.role == uptake_itt.role == "secondary"
    assert conversion_itt.discovery == uptake_itt.discovery is True

    # Both cells are selected (R=2, m=2) at q=0.02, so the FCR cutoff 0.02 sits
    # under nominal alpha 0.05 and applies uncapped: level 0.98, not the 0.95 a
    # metric-string collision would fall back to.
    assert uptake_itt.require_lift().level == pytest.approx(0.98)
    assert uptake_itt.require_lift().level == conversion_itt.require_lift().level, (
        "the uptake-named row's FCR-re-estimated interval level must match "
        "its family peer's exactly, proving it went through the same "
        "re-estimation pass and was not left at some other level by the "
        "metric-name collision"
    )

    # The design-level compliance diagnostic is still exactly one row per
    # treatment arm, unmodified by the outcome sharing its metric string,
    # and carries no role (it names no plan-tracked metric).
    compliance_rows = [r for r in results if r.estimand == "compliance"]
    assert len(compliance_rows) == 1
    assert compliance_rows[0].metric == "uptake"
    assert compliance_rows[0].role is None


def _uptake_guardrail_analysis(con, *, guardrail_metric_name="uptake"):
    """A declared guardrail whose metric name is ``guardrail_metric_name``
    (``"uptake"`` for the exact compliance-string collision, or an
    ``"_uptake"``-suffixed name like ``"revenue_uptake"`` for the
    ``analysis.py`` suffix-stripping collision) alongside an unrelated
    in-family secondary ("revenue").
    """
    table = f"guardrail_named_{guardrail_metric_name}_events"
    if table not in con.list_tables():
        control_units = [f"c{i}" for i in range(1, 7)]
        treat_units = [f"t{i}" for i in range(1, 7)]
        clickers = {"t1", "t2", "t3", "t4"}

        exposure_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "control",
                "revenue": None,
                "guard": None,
                "experiment_id": f"guardrail_{guardrail_metric_name}_exp",
            }
            for u in control_units
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "treatment",
                "revenue": None,
                "guard": None,
                "experiment_id": f"guardrail_{guardrail_metric_name}_exp",
            }
            for u in treat_units
        ]
        click_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "guard": None,
                "experiment_id": None,
            }
            for u in clickers
        ]
        revenue_by_unit = {
            "c1": 8.0,
            "c2": 9.0,
            "c3": 10.0,
            "c4": 11.0,
            "c5": 9.0,
            "c6": 11.0,
            "t1": 13.0,
            "t2": 14.0,
            "t3": 15.0,
            "t4": 16.0,
            "t5": 9.0,
            "t6": 11.0,
        }
        purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "guard": None,
                "experiment_id": None,
            }
            for u, amount in revenue_by_unit.items()
        ]
        guard_by_unit = {
            "c1": 2,
            "c2": 3,
            "c3": 2,
            "c4": 3,
            "c5": 2,
            "c6": 3,
            "t1": 2,
            "t2": 3,
            "t3": 3,
            "t4": 2,
            "t5": 3,
            "t6": 2,
        }
        guard_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "guard_event",
                "group_id": None,
                "revenue": None,
                "guard": float(cnt),
                "experiment_id": None,
            }
            for u, cnt in guard_by_unit.items()
        ]
        con.create_table(table, obj=exposure_rows + click_rows + purchase_rows + guard_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "guard_event", "column": "guard"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "mean",
                    "name": guardrail_metric_name,
                    "entity": "user_id",
                    "fact": "guard_event",
                    "aggregation": "sum",
                    "preferred_direction": "decrease",
                },
            ],
            "experiments": [
                {
                    "name": f"guardrail_{guardrail_metric_name}_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"], "guardrails": [guardrail_metric_name]},
                }
            ],
        }
    )

    return make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves revenue via uptake"
            ),
            one_sided=True,
            min_first_stage_z=0.001,
        ),
    )


@pytest.mark.parametrize("guardrail_metric_name", ["uptake", "revenue_uptake"])
def test_guardrail_metric_colliding_with_compliance_name_reports_requested_estimands(
    con, guardrail_metric_name
):
    """A guardrail literally named ``"uptake"`` shares the design-level
    compliance row's metric label, and ``"revenue_uptake"`` collides under
    naive ``_uptake`` suffix stripping. Both must report exactly the
    requested estimands, with the canonical compliance row told apart by
    estimand rather than metric string."""
    analysis = _uptake_guardrail_analysis(con, guardrail_metric_name=guardrail_metric_name)

    results = lift_rows(analysis.run(estimands=("late", "compliance")))

    by_metric_estimand: defaultdict[str, set[str]] = defaultdict(set)
    for r in results:
        by_metric_estimand[r.metric].add(r.estimand)

    expected_guardrail_estimands = {"late"} | (
        {"compliance"} if guardrail_metric_name == "uptake" else set()
    )
    assert by_metric_estimand[guardrail_metric_name] == expected_guardrail_estimands
    assert by_metric_estimand["revenue"] == {"late"}
    guardrail_late = [
        r for r in results if r.metric == guardrail_metric_name and r.estimand == "late"
    ]
    assert guardrail_late
    assert all(r.role == "guardrail" for r in guardrail_late)

    compliance_rows = [r for r in results if r.estimand == "compliance"]
    assert len(compliance_rows) == 1
    assert compliance_rows[0].metric == "uptake"
    assert compliance_rows[0].role is None


def test_analysis_run_native_encouragement_undeclared_plan_refuses_call_time_alternative(con):
    """Regression: an undeclared-plan native Encouragement experiment
    used to reach a private execute() escape hatch that accepted
    call-time alternative= (a state no YAML-loaded experiment can
    actually reach, since Experiment.plan is required). Now that
    Encouragement shares the same readout pipeline as every other
    design, that escape hatch is gone: call-time policy is refused
    unconditionally, declared plan or not."""
    from increment.plan import compile_decision_plan

    analysis = _guardrail_analysis(con)
    analysis = make_analysis_like(
        analysis, plan=compile_decision_plan(None, analysis.metrics, path="warehouse")
    )

    with pytest.raises(TypeError):
        lift_rows(analysis.run(alternative="greater"))  # ty: ignore[unknown-argument]


def test_analysis_run_native_encouragement_undeclared_plan_refuses_margins(con):
    from increment.plan import compile_decision_plan

    analysis = _guardrail_analysis(con)
    analysis = make_analysis_like(
        analysis, plan=compile_decision_plan(None, analysis.metrics, path="warehouse")
    )
    with pytest.raises(TypeError):
        lift_rows(analysis.run(margins={"revenue": 0.01}))  # ty: ignore[unknown-argument]


def test_analysis_run_native_encouragement_empty_selection_refuses_call_time_policy(con):
    """An empty metric selection must not silently swallow a dropped
    guardrail: `_guardrail_analysis`'s own declared plan refuses a
    non-default call-time policy kwarg the same way a non-empty selection
    does -- the early empty-selection return shares the declared-plan
    execute-family gate, not just the routed-path family's own check."""
    analysis = _guardrail_analysis(con)
    with pytest.raises(TypeError):
        lift_rows(analysis.run(metrics=[], alternative="greater"))


def test_analysis_run_native_encouragement_declared_plan_stamps_roles_and_guardrail_tail(con):
    """A declared plan (primary + margin-less guardrail, no call-time
    kwargs) must stamp every row's `role`, run the guardrail's itt row
    one-sided on its adverse tail (preferred_direction="decrease" on
    `errors` implies "less"), and leave the design-level `uptake`
    row's `role` unset (it names no plan-tracked metric)."""
    bound = _guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="revenue", guardrails=["errors"])}
        ),
    )

    results = lift_rows(bound.run())

    revenue_rows = [r for r in results if r.metric == "revenue"]
    assert revenue_rows
    assert all(r.role == "primary" for r in revenue_rows)

    errors_itt_rows = [r for r in results if r.metric == "errors" and r.estimand == "itt"]
    assert errors_itt_rows
    assert all(r.role == "guardrail" for r in errors_itt_rows)
    assert all(r.alternative == "less" for r in errors_itt_rows)

    uptake_rows = [r for r in results if r.metric == "uptake"]
    assert uptake_rows
    assert all(r.role is None for r in uptake_rows)


def _margined_guardrail_analysis(con):
    from increment.semantics.models import ExperimentMetric

    analysis = _guardrail_analysis(con)
    return make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=["revenue"],
                    guardrails=[ExperimentMetric(metric="errors", margin_abs=50.0)],
                )
            }
        ),
    )


def test_guardrail_margin_applies_to_itt_under_encouragement(con):
    """A guardrail's margin (the randomized non-inferiority construction)
    applies to its ITT row unchanged under a declared Encouragement design;
    LATE carries no margin verdict."""
    results = lift_rows(_margined_guardrail_analysis(con).run())
    itt = next(r for r in results if r.metric == "errors" and r.estimand == "itt")
    late = next(r for r in results if r.metric == "errors" and r.estimand == "late")

    # errors prefers "decrease": the absolute margin resolves to
    # (+margin_abs, "less"), exactly as under a randomized design.
    assert itt.null_abs == pytest.approx(50.0)
    assert itt.alternative == "less"
    assert late.null_abs is None
    assert late.null_lift in (None, 0.0)


def test_guardrail_margin_refuses_when_itt_not_requested(con):
    """A declared margin has no row to apply to when the request excludes
    itt entirely -- the one run request that still refuses, by name."""
    with pytest.raises(UnsupportedRequestError) as exc:
        _margined_guardrail_analysis(con).run(metrics=["errors"], estimands=("late",))
    assert exc.value.code == "readout.encouragement.margin"


def test_selected_margined_secondary_keeps_its_margin_when_reestimated(con):
    """A BH-selected secondary is re-estimated at the FCR level; the
    re-estimated ITT row must keep the secondary's declared margin rather
    than silently testing against zero."""
    from increment.semantics.models import ExperimentMetric

    bound = _multi_arm_guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        [
            m.model_copy(update={"preferred_direction": "increase"}) if m.name == "revenue" else m
            for m in bound.metrics
        ],
        experiment=bound.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    q=0.03,
                    secondaries=[ExperimentMetric(metric="revenue", margin_abs=0.5), "errors"],
                )
            }
        ),
    )

    revenue_itt = [
        r for r in lift_rows(bound.run()) if r.metric == "revenue" and r.estimand == "itt"
    ]
    assert {r.group_id for r in revenue_itt} == {"a", "b"}
    for r in revenue_itt:
        assert r.discovery is True
        assert r.require_lift().level == pytest.approx(0.985), "re-estimated at the FCR level"
        assert r.null_abs == pytest.approx(-0.5)
        assert r.alternative == "greater"


def test_analysis_run_native_encouragement_declared_plan_refuses_call_time_alternative(con):
    """Regression coverage for the encouragement branch specifically:
    a declared plan refuses a call-time `alternative=` kwarg via the
    shared dispatch gate (Task 2), even on a non-empty selection."""
    bound = _guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary="revenue", guardrails=["errors"])}
        ),
    )

    with pytest.raises(TypeError):
        lift_rows(bound.run(alternative="greater"))  # ty: ignore[unknown-argument]


def test_analysis_run_native_encouragement_refuses_unproved_default_late(con):
    """Continuous outcomes under an encouragement design have no admitted
    public sequential likelihood (increment/sequential_source.py
    validate_sequential_plan restricts registration to Bernoulli or
    registered scalar-mean observations, and scalar-mean registration is
    itself restricted to a fixed randomized design) -- so the refusal now
    surfaces at plan registration instead of at call time, but keeps the
    same coded contract."""
    from tests.sequential_cases import registered_native

    bound = _guardrail_analysis(con)
    metrics = [
        m.model_copy(update={"preferred_direction": None, "margin": None}) for m in bound.metrics
    ]
    with pytest.raises(CapabilityError) as raised:
        registered_native(bound, metrics=metrics)
    assert raised.value.code == "sequential.route.unsupported"


def test_analysis_run_native_encouragement_uptake_uses_plan_alpha_not_a_groups(con):
    """The design-level uptake row must be estimated at the plan's own
    alpha, not whichever metric group happened to run first - two
    primaries split alpha_share to 0.025 each (level 0.975), which must
    not leak into uptake's own interval (level 0.95 at plan.alpha=0.05)."""
    bound = _guardrail_analysis(con)
    bound = make_analysis_like(
        bound,
        experiment=bound.experiment.model_copy(
            update={"plan": AnalysisPlan(primary=["revenue", "errors"])}
        ),
    )

    results = lift_rows(bound.run())

    primary_rows = [r for r in results if r.metric in ("revenue", "errors") and r.role == "primary"]
    assert primary_rows
    assert all(r.require_lift().level == pytest.approx(0.975) for r in primary_rows)

    uptake_rows = [r for r in results if r.metric == "uptake"]
    assert uptake_rows
    assert all(r.require_lift().level == pytest.approx(0.95) for r in uptake_rows)


def test_run_encouragement_margin_without_itt_refuses_before_reading_moments():
    """The itt-excluded margin refusal fires at request validation, before
    the source is asked for any moments."""

    class _CountingSource(FakeMomentSource):
        moment_calls = 0

        def moments(self, metric, **kwargs):
            type(self).moment_calls += 1
            return super().moments(metric, **kwargs)

    guardrail = MeanMetric(
        name="rev",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
        margin_abs=0.5,
    )
    src = _CountingSource(_rows(), metrics=[guardrail], capabilities={"total"}, design=_design())
    with pytest.raises(UnsupportedRequestError) as exc:
        readouts.run(src, estimands=("late",))
    assert exc.value.code == "readout.encouragement.margin"
    assert cast("tuple[str, ...]", exc.value.context["names"]) == ("rev",)
    assert _CountingSource.moment_calls == 0


@pytest.mark.parametrize("declaration", [{"margin": 0.01}, {"margin_abs": 0.5}])
def test_asof_lift_encouragement_declared_margin_refused(declaration):
    """The as-of encouragement branch builds no shifted null, so a declared
    margin refuses at call time instead of degrading to two-sided-vs-0."""
    guardrail = MeanMetric(
        name="rev",
        entity="user_id",
        fact="orders",
        aggregation="sum",
        preferred_direction="decrease",
        **declaration,
    )
    rows = _rows()
    for r in rows:
        r["ds"] = date(2025, 1, 1)
    with pytest.raises(UnsupportedRequestError) as exc:
        src = FakeMomentSource(rows, metrics=[guardrail], capabilities={"asof"}, design=_design())
        readouts.asof_lift(src)
    assert exc.value.code == "readout.metric_declare_non"
    assert cast("tuple[str, ...]", exc.value.context["combined"]) == ("rev",)


def test_asof_lift_compliance_rows_carry_drift_caveat_when_late_suppressed():
    """With a weak instrument the late row is suppressed, so the compliance
    row is the only surviving open-endedness signal for an unwindowed
    uptake - the drift caveat must reach it."""
    design = _design(one_sided=True)  # uptake unwindowed
    rows = _rows(n=200, compliance=0.02, seed=11)  # weak: late suppressed
    for r in rows:
        r["ds"] = date(2025, 1, 1)
    src = FakeMomentSource(rows, metrics=[METRIC], capabilities={"asof"}, design=design)

    results = readouts.asof_lift(src)
    comp = [r for r in results if r.estimand == "compliance"]
    assert comp
    for r in comp:
        assert r.note is not None
        assert "uptake unwindowed: complier definition drifts" in r.note
    assert not [r for r in results if r.estimand == "late"]


def _two_metric_uptake_table() -> pa.Table:
    rows = [
        ("u01", "control", 10.0, 3.0, 0),
        ("u02", "control", 12.0, 4.0, 0),
        ("u03", "control", 9.0, 2.0, 0),
        ("u04", "control", 20.0, 5.0, 1),
        ("u05", "treatment", 25.0, 6.0, 1),
        ("u06", "treatment", 30.0, 7.0, 1),
        ("u07", "treatment", 28.0, 6.5, 1),
        ("u08", "treatment", 15.0, 3.5, 0),
    ]
    cols = list(zip(*rows, strict=True))
    return pa.table(
        dict(zip(["user_id", "variant", "revenue", "visits", "clicked"], cols, strict=True))
    )


def _multi_arm_uptake_table() -> pa.Table:
    rows = [
        ("u01", "control", 10.0, 0),
        ("u02", "control", 12.0, 0),
        ("u03", "control", 9.0, 0),
        ("u04", "control", 20.0, 1),
        ("u05", "t1", 25.0, 1),
        ("u06", "t1", 30.0, 1),
        ("u07", "t1", 28.0, 1),
        ("u08", "t1", 15.0, 0),
        ("u09", "t2", 22.0, 1),
        ("u10", "t2", 26.0, 1),
        ("u11", "t2", 24.0, 1),
        ("u12", "t2", 13.0, 0),
    ]
    cols = list(zip(*rows, strict=True))
    return pa.table(dict(zip(["user_id", "variant", "revenue", "clicked"], cols, strict=True)))


def _frame_encouragement_design() -> Encouragement:
    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True,
            justification="assignment only moves the outcome via uptake",
        ),
        # Low z-floor keeps LATE reported on these tiny fixtures instead of
        # exercising weak-instrument suppression.
        min_first_stage_z=0.5,
    )


def test_frame_encouragement_primary_alpha_splits_across_arms():
    """A declared primary's alpha splits across its own treatment arms, as
    it does for every other design: two arms at plan alpha 0.05 means
    0.025 per arm (level 0.975). Reporting 0.95 per arm would understate
    the multiplicity the arm axis introduces."""
    src = from_unit_summary(
        _multi_arm_uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(primary=["revenue"]),
    )

    itt = [r for r in readouts.run(src) if r.metric == "revenue" and r.estimand == "itt"]

    assert {r.group_id for r in itt} == {"t1", "t2"}
    assert all(r.require_lift().level == pytest.approx(0.975) for r in itt), [
        (r.group_id, r.require_lift().level) for r in itt
    ]


def test_frame_encouragement_emits_one_uptake_row_per_arm():
    """`uptake` is design-level, not per-metric: two metrics must not
    multiply the first-stage row. Duplicates would double-count the
    compliance readout in any table built off these rows."""
    src = from_unit_summary(
        _two_metric_uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(primary=["revenue", "visits"]),
    )

    uptake = [r for r in readouts.run(src) if r.metric == "uptake"]

    assert [(r.group_id, r.estimand) for r in uptake] == [("treatment", "compliance")]


def test_frame_encouragement_uptake_uses_plan_alpha_not_a_metrics_share():
    """The design-level `uptake` row is estimated at the plan's own
    alpha. Two primaries split alpha_share to 0.025 each (level 0.975),
    which must not leak into `uptake`'s own interval (level 0.95)."""
    src = from_unit_summary(
        _two_metric_uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(primary=["revenue", "visits"]),
    )

    results = readouts.run(src)

    primaries = [r for r in results if r.metric in ("revenue", "visits") and r.estimand == "itt"]
    assert all(r.require_lift().level == pytest.approx(0.975) for r in primaries)
    uptake = [r for r in results if r.metric == "uptake"]
    assert uptake
    assert all(r.require_lift().level == pytest.approx(0.95) for r in uptake)


def test_frame_encouragement_in_family_secondary_gets_a_discovery_verdict():
    """An in-family secondary faces BH/e-BH selection here exactly as it
    does on every other dispatch path: `discovery` is a real verdict, not
    None. Leaving it unset reports an uncorrected secondary interval as if
    it had passed a family bar it never faced."""
    src = from_unit_summary(
        _two_metric_uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(primary=["revenue"], secondaries=["visits"]),
    )

    itt = [
        r
        for r in readouts.run(src)
        if r.metric == "visits" and r.estimand == "itt" and r.role == "secondary"
    ]

    assert itt
    assert all(r.discovery is not None for r in itt)
    assert all(r.family_axes == ("metric", "arm") for r in itt)


def test_frame_encouragement_callwide_prior_excludes_secondary_family():
    """A call-wide informative prior keeps encouragement secondaries out of
    BH/FCR while preserving the primary outcome and canonical compliance
    passes."""
    src = from_unit_summary(
        _two_metric_uptake_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "visits": "mean"},
        uptake="clicked",
        design=_frame_encouragement_design(),
        plan=AnalysisPlan(q=0.02, primary=["revenue"], secondaries=["visits"]),
    )

    results = readouts.run(src, prior=Normal(mu=0.0, sigma=0.1))
    visits = [r for r in results if r.metric == "visits" and r.estimand == "itt"]
    assert visits
    assert all(r.role == "secondary" for r in visits)
    assert all(r.discovery is None for r in visits)
    assert all(r.family_axes is None for r in visits)
    assert all(r.require_lift().level == pytest.approx(0.95) for r in visits)

    revenue = [r for r in results if r.metric == "revenue" and r.estimand == "itt"]
    assert revenue
    assert all(r.role == "primary" for r in revenue)
    compliance = [r for r in results if r.estimand == "compliance"]
    assert compliance
    assert all(r.metric == "uptake" for r in compliance)
    assert all(r.discovery is None for r in compliance)


def test_frame_encouragement_compliance_ignores_outcome_config_and_order():
    """The canonical first-stage row is independent of outcome priors/methods.

    Outcome configs still apply to their own ITT rows, but declaration order
    must not decide which outcome config is borrowed for ``uptake``.
    """

    def run_with_specs(specs):
        src = from_unit_summary(
            _two_metric_uptake_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics=specs,
            uptake="clicked",
            design=_frame_encouragement_design(),
            plan=AnalysisPlan(primary=["revenue", "visits"]),
        )
        return readouts.run(src)

    methods = (
        Method(name="revenue_method"),
        Method(name="visits_method"),
    )
    declared_specs = [
        MetricSpec(
            name="revenue",
            decision_method=methods[0],
            prior=Normal(mu=0.0, sigma=0.01),
        ),
        MetricSpec(
            name="visits",
            decision_method=methods[1],
            prior=Normal(mu=0.0, sigma=0.02),
        ),
    ]
    flat_specs = [
        MetricSpec(name="revenue", decision_method=methods[0]),
        MetricSpec(name="visits", decision_method=methods[1]),
    ]

    declared = run_with_specs(declared_specs)
    reversed_declared = run_with_specs(list(reversed(declared_specs)))
    flat = run_with_specs(flat_specs)

    def by_key(rows):
        return {(r.metric, r.estimand, r.group_id): r for r in rows}

    declared_by_key = by_key(declared)
    reversed_by_key = by_key(reversed_declared)
    flat_by_key = by_key(flat)
    compliance_key = ("uptake", "compliance", "treatment")
    compliance = declared_by_key[compliance_key]

    assert compliance.method == "unadjusted"
    assert compliance.require_lift().value == pytest.approx(
        reversed_by_key[compliance_key].require_lift().value
    )
    assert compliance.require_lift().lb == pytest.approx(
        reversed_by_key[compliance_key].require_lift().lb
    )
    assert compliance.require_lift().ub == pytest.approx(
        reversed_by_key[compliance_key].require_lift().ub
    )
    for name in ("revenue", "visits"):
        key = (name, "itt", "treatment")
        assert declared_by_key[key].require_lift().value == pytest.approx(
            reversed_by_key[key].require_lift().value
        )
        assert declared_by_key[key].require_lift().value != pytest.approx(
            flat_by_key[key].require_lift().value
        )


def test_asof_encouragement_reports_one_compliance_row_at_plan_alpha():
    """Uptake is design-level: one row per (date, arm) at the plan's own alpha,
    role=None -- never one copy per outcome metric at that metric's split alpha."""
    from types import SimpleNamespace
    from typing import cast

    from increment import readouts
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import ComplianceArm, ComplianceSummary, MomentSource, MomentsSource

    def _asof_row(metric, group_id, mean, uptake_rate):
        n, var = 2000, 0.25
        sum_d = n * uptake_rate
        return {
            "ds": "2026-01-01",
            "experiment_id": "t",
            "metric": metric,
            "group_id": group_id,
            "n": n,
            "ref_y": 0.0,
            "cy1": n * mean,
            "cy2": n * (mean * mean + var),
            "sum_d": sum_d,
            "cyd": sum_d * mean,
            "cy2d": sum_d * (mean * mean + var),
        }

    metrics = [
        MeanMetric(name="rev", entity="u", fact="rev"),
        MeanMetric(name="orders", entity="u", fact="orders"),
    ]
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves the outcome via uptake"
        ),
        min_first_stage_z=0.001,
    )
    plan = compile_decision_plan(AnalysisPlan(primary=("rev", "orders")), metrics, design=design)
    rows = {
        "rev": [_asof_row("rev", "control", 1.0, 0.05), _asof_row("rev", "treatment", 1.1, 0.5)],
        "orders": [
            _asof_row("orders", "control", 2.0, 0.05),
            _asof_row("orders", "treatment", 2.1, 0.5),
        ],
    }

    def _compliance_summary(requested_design, *, as_of=None, completed_windows_only=False):
        return ComplianceSummary(
            study_id="t",
            cohort=requested_design.uptake.fact,
            window_days=requested_design.uptake.window_days,
            one_sided=requested_design.one_sided,
            cluster=None,
            as_of=as_of,
            arms=(
                ComplianceArm(group_id="control", n_units=2000, uptake_total=100.0),
                ComplianceArm(group_id="treatment", n_units=2000, uptake_total=1000.0),
            ),
        )

    source = cast(
        MomentSource,
        SimpleNamespace(
            capabilities=frozenset({"total", "asof"}),
            operations=frozenset(),
            breakouts=(),
            context=MomentsSource(
                [], metrics=tuple(metrics), study_id="t", design=design, plan=plan
            ).context,
            moments=lambda metric, **_kwargs: rows[metric.name],
            compliance_dates=lambda: ("2026-01-01",),
            compliance_summary=_compliance_summary,
        ),
    )
    out = readouts.asof_lift(source)
    compliance = [row for row in out if row.estimand == "compliance"]

    assert [(row.ds, row.group_id) for row in compliance] == [("2026-01-01", "treatment")]
    assert compliance[0].require_lift().level == 1.0 - plan.alpha
    assert compliance[0].role is None


def test_asof_compliance_ignores_per_metric_cohort_disagreement():
    """Design-level compliance comes from ``compliance_summary()`` alone:
    per-metric moments carrying different (and irrelevant) ``sum_d`` values
    -- the exact scenario the deleted cross-metric-disagreement guard used
    to refuse -- must not affect the reported compliance row at all, and
    the result must be identical regardless of metric declaration order."""
    from types import SimpleNamespace
    from typing import cast

    from increment import readouts
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import ComplianceArm, ComplianceSummary, MomentSource, MomentsSource

    def _asof_row(ds, metric, group_id, mean, n, sum_d):
        var = 0.25
        return {
            "ds": ds,
            "experiment_id": "t",
            "metric": metric,
            "group_id": group_id,
            "n": n,
            "ref_y": 0.0,
            "cy1": n * mean,
            "cy2": n * (mean * mean + var),
            "sum_d": sum_d,
            "cyd": sum_d * mean,
            "cy2d": sum_d * (mean * mean + var),
        }

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment only moves the outcome via uptake"
        ),
        min_first_stage_z=0.001,
    )

    def _make_source(metric_order):
        metrics = [MeanMetric(name=n, entity="u", fact=n) for n in metric_order]
        plan = compile_decision_plan(
            AnalysisPlan(primary=tuple(metric_order)), metrics, design=design
        )
        # "rev" and "orders" disagree on sum_d (100 vs 50) for the same date;
        # compliance comes only from compliance_summary(), never these rows.
        rows_by_metric = {
            "rev": [
                _asof_row("2026-01-01", "rev", "control", 1.0, 2000, 100.0),
                _asof_row("2026-01-01", "rev", "treatment", 1.1, 2000, 1000.0),
            ],
            "orders": [
                _asof_row("2026-01-01", "orders", "control", 2.0, 2000, 50.0),
                _asof_row("2026-01-01", "orders", "treatment", 2.1, 2000, 1000.0),
            ],
        }

        def _compliance_summary(requested_design, *, as_of=None, completed_windows_only=False):
            return ComplianceSummary(
                study_id="t",
                cohort=requested_design.uptake.fact,
                window_days=requested_design.uptake.window_days,
                one_sided=requested_design.one_sided,
                cluster=None,
                as_of=as_of,
                arms=(
                    ComplianceArm(group_id="control", n_units=2000, uptake_total=200.0),
                    ComplianceArm(group_id="treatment", n_units=2000, uptake_total=1000.0),
                ),
            )

        return cast(
            MomentSource,
            SimpleNamespace(
                capabilities=frozenset({"total", "asof"}),
                operations=frozenset(),
                breakouts=(),
                context=MomentsSource(
                    [], metrics=tuple(metrics), study_id="t", design=design, plan=plan
                ).context,
                moments=lambda metric, **_kwargs: rows_by_metric[metric.name],
                compliance_dates=lambda: ("2026-01-01",),
                compliance_summary=_compliance_summary,
            ),
        )

    results_by_order = {}
    for metric_order in (("rev", "orders"), ("orders", "rev")):
        source = _make_source(metric_order)
        out = readouts.asof_lift(source)
        compliance = [row for row in out if row.estimand == "compliance"]
        assert len(compliance) == 1
        results_by_order[metric_order] = compliance[0].require_lift().value

    values = list(results_by_order.values())
    assert values[0] == pytest.approx(values[1])


@pytest.mark.parametrize(
    "constructor",
    (
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
        "from_moments",
    ),
)
def test_encouragement_margin_guardrail_refuses_late_only_on_every_constructor(constructor):
    """A margined guardrail's verdict rides the ITT row (see
    tests/parity_harness/cases.py's `_encouragement_margin_guardrail_case`,
    which proves the full itt/compliance/late request matches across every
    path); a `late`-only request has no ITT row to carry it, and must
    refuse with the same code on every reachable constructor -- there is
    nothing left to compare on any path once every one refuses, so this
    is a focused per-constructor test rather than a second ParityCase."""
    from increment.errors import CodedError
    from tests.parity_harness.cases import _encouragement_guardrail_case

    case = _encouragement_guardrail_case(
        id="margin_late_only_probe", estimands=("late",), margin_abs=5.0
    )
    analysis = case.build[constructor]()
    try:
        with pytest.raises(CodedError) as exc_info:
            analysis.run(estimands=("late",))
        assert exc_info.value.code == "readout.encouragement.margin"
    finally:
        analysis.close()
        for connection in getattr(analysis, "_parity_connections", ()):
            connection.disconnect()


@pytest.mark.parametrize(
    ("constructor", "expected_code"),
    (
        ("from_definitions", "readout.encouragement.retention"),
        ("from_unit_day_artifact", "readout.encouragement.retention"),
        ("from_moments", "readout.encouragement.retention"),
        ("from_unit_panel", "readout.encouragement.retention"),
        ("from_unit_summary", "source.frame.constructor"),
    ),
)
def test_encouragement_retention_refuses_on_every_reachable_constructor(constructor, expected_code):
    """A retention metric under an Encouragement design is never estimable
    on any reachable path (see
    tests/parity_harness/cases.py's `_encouragement_retention_case`).
    `from_definitions`/`from_unit_day_artifact`/`from_moments`/
    `from_unit_panel` all share `readout.encouragement.retention` -- the
    panel refuses it earlier, at construction, but with the same code the
    readout layer uses for the other three, matching AGENTS.md's
    same-hazard-same-code contract. `from_unit_summary` refuses for an
    unrelated, encouragement-independent reason (`source.frame.constructor`
    -- no per-unit dates to resolve `threshold_days` against), a genuinely
    different hazard the panel does not share."""
    from increment.errors import CodedError
    from tests.parity_harness.cases import _encouragement_retention_case

    case = _encouragement_retention_case()
    if constructor == "from_unit_panel":
        with pytest.raises(CodedError) as exc_info:
            case.build[constructor]()
        assert exc_info.value.code == expected_code
        return
    if constructor == "from_unit_summary":
        with pytest.raises(CodedError) as exc_info:
            case.build[constructor]()
        assert exc_info.value.code == expected_code
        return
    analysis = case.build[constructor]()
    try:
        with pytest.raises(CodedError) as exc_info:
            analysis.run()
        assert exc_info.value.code == expected_code
    finally:
        analysis.close()
        for connection in getattr(analysis, "_parity_connections", ()):
            connection.disconnect()
