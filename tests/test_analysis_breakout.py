"""Breakout views over native definitions, frames and custom providers."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pytest

from increment import Scope, from_definitions, from_frame, query, workflows
from increment.artifacts import SourceSnapshotEvidence
from increment.breakout.estimates import BreakoutEstimate
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.evidence import BreakoutIdentity
from increment.semantics.models import AnalysisPlan, Definitions, MultiplicitySpec
from increment.workflows import (
    experiment_definition,
    metrics,
    sequential_snapshot,
    unit_counts,
)
from tests.analysis_factory import (
    bind_pair,
    make_binding,
    make_binding_like,
    view_readout,
)
from tests.sequential_cases import registration
from tests.warning_codes import warning_codes

_BREAKOUT = Scope(view="breakout", grain="total")


def _analysis_with_country_breakout(
    con,
    *,
    unbounded_retention: bool = False,
    source_snapshot_evidence=None,
    return_definitions=False,
):
    """Build the country-breakout fixture for legacy and source-operation tests."""
    if "breakout_run_events" not in con.list_tables():
        # Exposure: 2 control + 2 treatment units per country.
        exposure_groups = {
            "US": {"control": ["bu1", "bu2"], "treatment": ["bu3", "bu4"]},
            "CA": {"control": ["bu5", "bu6"], "treatment": ["bu7", "bu8"]},
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                # first_exposures' dedup semi-joins on (unit_id, experiment_id): a NULL experiment_id never matches itself (SQL NULL = NULL is not true), so every row needs the real experiment name.
                "experiment_id": "breakout_run_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        # Revenue (mean-metric fact), one purchase per unit: US control mean=10, US treatment mean=15 (+50% lift); CA control mean=20, CA treatment mean=15 (-25% lift) - opposite-signed so cross-segment contamination flips a sign.
        revenue_by_unit = {
            "bu1": 9.0,
            "bu2": 11.0,
            "bu3": 14.0,
            "bu4": 16.0,
            "bu5": 19.0,
            "bu6": 21.0,
            "bu7": 14.0,
            "bu8": 16.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in revenue_by_unit.items()
        ]
        con.create_table("breakout_run_events", obj=exposure_rows + purchase_rows)

    metric_definitions: list[dict[str, object]] = [
        {
            "type": "mean",
            "name": "revenue",
            "entity": "user_id",
            "fact": "purchase",
            "aggregation": "sum",
        }
    ]
    if unbounded_retention:
        metric_definitions.append(
            {
                "type": "retention",
                "name": "unbounded",
                "entity": "user_id",
                "fact": "purchase",
                "threshold_days": 7,
            }
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM breakout_run_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": metric_definitions,
            "experiments": [
                {
                    "name": "breakout_run_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": [metric["name"] for metric in metric_definitions]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    if return_definitions:
        return defs
    analysis = make_binding(con, defs, source_snapshot_evidence=source_snapshot_evidence)
    return analysis


@pytest.fixture(scope="module")
def country_breakout_estimates(con):
    """Plain breakout view on the country fixture, computed once;
    the tests below assert different facets of one deterministic readout."""
    experiment, resolver = _analysis_with_country_breakout(con)
    return view_readout(experiment, resolver, _BREAKOUT)


@pytest.fixture(scope="module")
def country_daily_lift_by_segment(con):
    """Plain country daily-lift view on the country fixture, computed once
    and shared the same way."""
    experiment, resolver = _analysis_with_country_breakout(con)
    return view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", segment=BreakoutIdentity("country", None)),
    )


def test_run_breakout_snapshot_identity_includes_pinned_source_evidence(con):
    first_experiment, first_resolver = _analysis_with_country_breakout(
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime.fromisoformat("2025-06-30T00:00:00+00:00"),
            {"events": datetime.fromisoformat("2025-06-30T00:00:00+00:00")},
        ),
    )
    changed_experiment, changed_resolver = _analysis_with_country_breakout(
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            datetime.fromisoformat("2025-07-01T00:00:00+00:00"),
            {"events": datetime.fromisoformat("2025-06-30T00:00:00+00:00")},
        ),
    )
    try:
        first_rows = view_readout(first_experiment, first_resolver, _BREAKOUT)
        changed_rows = view_readout(changed_experiment, changed_resolver, _BREAKOUT)
        assert first_rows and changed_rows
        assert [row.require_lift().value for row in first_rows] == pytest.approx(
            [row.require_lift().value for row in changed_rows]
        )
        assert {row.source_snapshot_id for row in first_rows}.isdisjoint(
            {row.source_snapshot_id for row in changed_rows}
        )
    finally:
        first_resolver.close()
        changed_resolver.close()


def test_custom_unit_counts_refuse_before_an_undeclared_provider_read(con):
    from dataclasses import replace
    from typing import Any, cast

    from increment.evidence import CountsRequest
    from tests.analysis_factory import held_source

    _, fixture_resolver = _analysis_with_country_breakout(con)
    source = cast(Any, held_source(fixture_resolver))
    called = False

    class Provider:
        context = source.context
        evidence = replace(source.evidence, count_grains=frozenset())

        def moments(self, request):
            raise AssertionError("unsupported moments read")

        def breakout_method_moments(self, request):
            raise AssertionError("unsupported breakout read")

        def unit_rows(self, request):
            raise AssertionError("unsupported unit-row read")

        def method_fit(self, request):
            raise AssertionError("unsupported fit read")

        def compliance(self, request):
            raise AssertionError("unsupported compliance read")

        def compliance_series(self, request):
            raise AssertionError("unsupported compliance-series read")

        def counts(self, request: CountsRequest):
            nonlocal called
            called = True
            raise AssertionError("undeclared count read")

        def capture_sequential(self, request):
            raise AssertionError("unsupported sequential read")

        def scope(self, request):
            raise AssertionError("unsupported scope")

    experiment, resolver = bind_pair(cast(Any, Provider()))
    try:
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            unit_counts(experiment, resolver)
        assert raised.value.code == "evidence.operation_unavailable"
        assert not called
    finally:
        resolver.close()
        fixture_resolver.close()


def test_custom_breakout_uses_one_outer_snapshot_scope(con):
    from contextlib import contextmanager
    from typing import Any, cast

    from increment.evidence import (
        BreakoutMethodMomentsRequest,
        ComplianceRequest,
        ComplianceSeriesRequest,
        CountsRequest,
        EvidenceScope,
        EvidenceScopeResult,
        MethodFitRequest,
        MomentsRequest,
        SequentialCaptureRequest,
        UnitRowsRequest,
        evidence_provider,
    )
    from tests.analysis_factory import held_source

    analysis_experiment, analysis_resolver = _analysis_with_country_breakout(con)
    source = cast(Any, held_source(analysis_resolver))
    builtin = evidence_provider(source)
    requests: list[tuple[EvidenceScope, bool]] = []
    active = 0

    class Provider:
        context = source.context
        evidence = source.evidence
        artifact_experiment = experiment_definition(analysis_experiment, analysis_resolver)

        def moments(self, request: MomentsRequest):
            return builtin.moments(request)

        def unit_rows(self, request: UnitRowsRequest):
            return builtin.unit_rows(request)

        def method_fit(self, request: MethodFitRequest):
            return builtin.method_fit(request)

        def breakout_method_moments(self, request: BreakoutMethodMomentsRequest):
            return builtin.breakout_method_moments(request)

        def compliance(self, request: ComplianceRequest):
            return builtin.compliance(request)

        def compliance_series(self, request: ComplianceSeriesRequest):
            return builtin.compliance_series(request)

        def counts(self, request: CountsRequest):
            return builtin.counts(request)

        def capture_sequential(self, request: SequentialCaptureRequest):
            return builtin.capture_sequential(request)

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                nonlocal active
                requests.append((request, active > 0))
                active += 1
                try:
                    yield self if not request.breakouts else source
                finally:
                    active -= 1

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    custom_experiment, custom_resolver = bind_pair(cast(Any, Provider()))
    try:
        view_readout(custom_experiment, custom_resolver, _BREAKOUT)
    finally:
        custom_resolver.close()
        analysis_resolver.close()

    snapshots = [(request, nested) for request, nested in requests if request.snapshot]
    assert len(snapshots) == 1
    assert snapshots[0][1] is False
    assert all(nested for _, nested in requests if not _.snapshot)


def test_run_breakout_returns_per_segment_estimates(country_breakout_estimates):
    """The breakout view wires breakout_summaries into
    the breakout readout and returns BreakoutEstimate rows for every
    declared breakout x metric, correctly attributed per segment (not
    cross-contaminated - US has positive lift, CA has negative)."""
    results = country_breakout_estimates

    assert len(results) == 2  # 1 metric x 1 method x 1 treatment arm x 2 segments
    for r in results:
        assert isinstance(r, BreakoutEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"
        assert r.dimension == "country"
        assert r.source == "events"  # the resolved FactSource name

    by_segment = {r.dimension_value: r for r in results}
    assert set(by_segment) == {"US", "CA"}
    assert by_segment["US"].require_lift().value > 0, (
        "US treatment (15) > control (10) -- expected positive lift"
    )
    assert by_segment["CA"].require_lift().value < 0, (
        "CA treatment (15) < control (20) -- expected negative lift"
    )


def _configured_country_breakout_analysis(con, correction: Literal["none", "bonferroni", "bh"]):
    from increment.semantics.models import ExperimentMetric, MethodSpec, NormalPriorSpec

    analysis_experiment, analysis_resolver = _analysis_with_country_breakout(con)
    return make_binding_like(
        analysis_resolver,
        experiment=experiment_definition(analysis_experiment, analysis_resolver).model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="declared"),
                            prior=NormalPriorSpec(mu=0.0, sigma=0.01),
                        )
                    ],
                    view_multiplicity=MultiplicitySpec(correction=correction),
                )
            }
        ),
    )


def test_native_breakout_ignores_incompatible_declared_method(con):
    """Native breakout prevalidation must ignore observational-only metric
    declarations on a randomized design when no call-wide method is set."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    analysis_experiment, analysis_resolver = _analysis_with_country_breakout(con)
    configured_experiment, configured_resolver = make_binding_like(
        analysis_resolver,
        experiment=experiment_definition(analysis_experiment, analysis_resolver).model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="revenue",
                            decision_method=MethodSpec(name="iptw"),
                        )
                    ]
                )
            }
        ),
    )

    with pytest.raises(InvalidRequestError) as raised:
        view_readout(configured_experiment, configured_resolver, _BREAKOUT)
    assert raised.value.code == "estimation.engine.method_name_observational"


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_native_breakout_uses_callwide_methods_not_declared_methods(con, correction):
    from increment.estimation.engine import Method

    experiment, resolver = _configured_country_breakout_analysis(con, correction)

    results = view_readout(experiment, resolver, _BREAKOUT)
    assert results
    assert {row.method for row in results} == {"declared"}

    explicit = view_readout(
        experiment,
        resolver,
        _BREAKOUT,
        decision_method=Method(name="call-wide"),
    )
    assert explicit
    assert {row.method for row in explicit} == {"call-wide"}


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_native_breakout_keeps_declared_prior_separate_from_sampling(con, correction):
    from increment.estimation import Normal
    from increment.estimation.priors import MixturePrior

    configured_experiment, configured_resolver = _configured_country_breakout_analysis(
        con, correction
    )
    baseline_experiment, baseline_resolver = make_binding_like(
        configured_resolver,
        experiment=experiment_definition(configured_experiment, configured_resolver).model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=["revenue"],
                    view_multiplicity=MultiplicitySpec(correction=correction),
                )
            }
        ),
    )

    declared_results = view_readout(configured_experiment, configured_resolver, _BREAKOUT)
    baseline_results = view_readout(baseline_experiment, baseline_resolver, _BREAKOUT)

    def keyed_lifts(results):
        return {
            (row.metric, row.group_id, row.dimension_value): row
            for row in results
            if row.lift is not None
        }

    def assert_same_sampling(actual, expected):
        assert actual.keys() == expected.keys()
        for key, row in actual.items():
            baseline_row = expected[key]
            assert row.require_lift().value == pytest.approx(baseline_row.require_lift().value)
            assert row.require_lift().lb == pytest.approx(baseline_row.require_lift().lb)
            assert row.require_lift().ub == pytest.approx(baseline_row.require_lift().ub)
            assert row.sampling_available is True
            assert baseline_row.sampling_available is True
            assert row.discovery == baseline_row.discovery
            assert row.family_size == baseline_row.family_size
            assert row.family_threshold == baseline_row.family_threshold

    declared_lifts = keyed_lifts(declared_results)
    baseline_lifts = keyed_lifts(baseline_results)
    assert_same_sampling(declared_lifts, baseline_lifts)

    prior_results = view_readout(
        baseline_experiment,
        baseline_resolver,
        _BREAKOUT,
        prior=Normal(mu=0.0, sigma=0.01),
    )
    prior_lifts = keyed_lifts(prior_results)
    assert_same_sampling(prior_lifts, baseline_lifts)
    assert any(row.posterior_estimate is not None for row in prior_lifts.values())

    mixture_prior = MixturePrior(weights=(0.4, 0.6), means=(0.0, 0.1), sigmas=(0.02, 0.15))
    mixture_results = view_readout(
        baseline_experiment,
        baseline_resolver,
        _BREAKOUT,
        prior=mixture_prior,
    )
    mixture_lifts = keyed_lifts(mixture_results)
    assert_same_sampling(mixture_lifts, baseline_lifts)
    mixture_row = next(
        row for row in mixture_lifts.values() if row.posterior_components is not None
    )
    assert mixture_row.prior_spec == mixture_prior
    family_view = mixture_row._family_view()
    assert family_view is not None
    assert family_view.posterior_components == mixture_row.posterior_components
    chance_to_beat = family_view.chance_to_beat()
    assert chance_to_beat is not None
    assert 0.0 <= chance_to_beat <= 1.0
    saved_row = type(mixture_row).model_validate_json(mixture_row.model_dump_json())
    assert saved_row.posterior_components == mixture_row.posterior_components
    saved_family_view = saved_row._family_view()
    assert saved_family_view is not None
    assert saved_family_view.chance_to_beat() == pytest.approx(chance_to_beat)
    mixture_frame = mixture_results.to_frame(backend="pandas")
    assert isinstance(mixture_frame, pd.DataFrame)
    frame_record = next(
        record
        for record in mixture_frame.to_dict(orient="records")
        if record["metric"] == mixture_row.metric
        and record["group_id"] == mixture_row.group_id
        and record["dimension_value"] == mixture_row.dimension_value
    )
    frame_record.pop("view_partial", None)
    frame_data = mixture_row.model_dump(mode="python")
    frame_data["posterior_components"] = frame_record["posterior_components"]
    frame_row = type(mixture_row).model_validate(frame_data)
    frame_family_view = frame_row._family_view()
    assert frame_family_view is not None
    assert frame_family_view.posterior_components == mixture_row.posterior_components
    assert frame_family_view.chance_to_beat() == pytest.approx(chance_to_beat)


def test_run_breakout_plan_declared_multiplicity_controls_level(con):
    """Breakout correction and alpha are read from the resolved plan."""
    _, fixture_resolver = _analysis_with_country_breakout(con)

    experiment, resolver = make_binding_like(
        fixture_resolver,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="bonferroni"),
        ),
    )
    default_alpha = view_readout(experiment, resolver, _BREAKOUT)
    assert len(default_alpha) == 2
    for r in default_alpha:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.05 / 2)

    custom_experiment, custom_resolver = make_binding_like(
        fixture_resolver,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            alpha=0.1,
            view_multiplicity=MultiplicitySpec(correction="bonferroni"),
        ),
    )
    custom_alpha = view_readout(custom_experiment, custom_resolver, _BREAKOUT)
    assert len(custom_alpha) == 2
    for r in custom_alpha:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(1 - 0.1 / 2)

    uncorrected_experiment, uncorrected_resolver = make_binding_like(
        fixture_resolver,
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="none"),
        ),
    )
    uncorrected = view_readout(uncorrected_experiment, uncorrected_resolver, _BREAKOUT)
    assert len(uncorrected) == 2
    for r in uncorrected:
        lift = r.lift
        assert lift is not None
        assert lift.level == pytest.approx(0.95)


def test_run_breakout_default_correction_runs_bh_family(country_breakout_estimates):
    """The breakout view's `correction` default is `None`, which
    delegates to `readouts.breakout()`'s own default (`"bh"` under a
    randomized design) instead of being pinned to the old `"none"` -- a
    plain breakout view now stamps a real `discovery` bool on
    every row (the BH family actually ran), not the all-`None` a
    forced `correction="none"` would leave."""
    results = country_breakout_estimates
    assert len(results) == 2
    assert all(r.discovery is not None for r in results)
    assert all(isinstance(r.discovery, bool) for r in results)


def _panel_with_borderline_country_segment():
    """3-country panel (US/CA/MX): US is a strong lift, CA is null, and
    MX (seed/shift chosen empirically) sits right on the BH boundary --
    not selected at q=0.05, selected at q=0.10 -- so a test can prove
    `q=` actually reaches the underlying selection instead of being
    silently dropped between the facade and `readouts.breakout`."""
    import pandas as pd

    rng = np.random.default_rng(2)
    rows = []
    n = 15
    for country, shift in (("US", 4.0), ("CA", 0.0), ("MX", 0.5)):
        for i in range(n):
            rows.append(
                {
                    "user_id": f"c_{country}_{i}",
                    "variant": "control",
                    "day": "2026-01-01",
                    "country": country,
                    "revenue": 10.0 + rng.normal(0, 0.5),
                }
            )
            rows.append(
                {
                    "user_id": f"t_{country}_{i}",
                    "variant": "treatment",
                    "day": "2026-01-01",
                    "country": country,
                    "revenue": 10.0 + shift + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_run_breakout_declared_bh_q_controls_selection():
    """A declared breakout BH policy controls the family q level."""
    table = _panel_with_borderline_country_segment()
    common: dict[str, Any] = {
        "unit": "user_id",
        "group": "variant",
        "date": "day",
        "control": "control",
        "metrics": {"revenue": "mean"},
        "breakouts": ["country"],
    }
    tight_experiment, tight_resolver = from_frame(
        table,
        shape="panel",
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.05),
        ),
        **common,
    )
    wide_experiment, wide_resolver = from_frame(
        table,
        shape="panel",
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
        **common,
    )

    tight_discovery = {
        r.dimension_value: r.discovery
        for r in view_readout(tight_experiment, tight_resolver, _BREAKOUT)
    }
    assert tight_discovery == {"US": True, "CA": False, "MX": False}
    wide_discovery = {
        r.dimension_value: r.discovery
        for r in view_readout(wide_experiment, wide_resolver, _BREAKOUT)
    }
    assert wide_discovery == {"US": True, "CA": False, "MX": True}


def test_run_breakout_bh_family_survives_a_degenerate_segment_cell():
    """run_breakout's own BH correction calls the SAME select_family
    path as run(); a segment cell breakout pre-drops as
    excluded='nonpositive_mean' (raw non-positive mean, caught before
    estimate_lift runs) must not abort the whole breakout for every
    other segment/metric -- the same non-rejection rule that covers
    estimation.engine.lift_guard now also covers breakout's own
    breakout.nonpositive_mean/breakout.zero_variance prescreen codes."""
    rng = np.random.default_rng(11)
    n = 100
    rows = []
    for country in ("US", "CA", "GB"):
        for group, shift in (("control", 0.0), ("treatment", 3.0)):
            for i in range(n):
                refunds = (
                    0.0
                    if (country == "GB" and group == "treatment")
                    else float(rng.normal(1.0, 0.2))
                )
                rows.append(
                    {
                        "user_id": f"{country}_{group}_{i}",
                        "variant": group,
                        "day": "2026-01-01",
                        "country": country,
                        "revenue": 10.0 + shift + rng.normal(0, 0.5),
                        "refunds": max(refunds, 0.0),
                    }
                )
    table = pd.DataFrame(rows)
    experiment, resolver = from_frame(
        table,
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean", "refunds": "mean"},
        breakouts=["country"],
        plan=AnalysisPlan(
            secondaries=["revenue", "refunds"],
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = view_readout(experiment, resolver, _BREAKOUT)
    degenerate_rows = [r for r in results if r.metric == "refunds" and r.dimension_value == "GB"]
    assert degenerate_rows and all(
        r.excluded == "nonpositive_mean" and r.lift is None for r in degenerate_rows
    )
    healthy_rows = [
        r
        for r in results
        if r.metric == "revenue" and r.dimension_value == "US" and r.excluded is None
    ]
    assert healthy_rows  # the family completed instead of aborting for GB/refunds

    from increment.breakout.estimates import BreakoutEstimate
    from increment.tables import estimates_to_readout

    for row in degenerate_rows:
        assert BreakoutEstimate.model_validate_json(row.model_dump_json()) == row
    frame = results.to_frame(backend="pandas")
    assert isinstance(frame, pd.DataFrame)
    failed_frame = frame[(frame["metric"] == "refunds") & (frame["dimension_value"] == "GB")]
    assert len(failed_frame) == len(degenerate_rows)
    assert pd.isna(failed_frame["lift"]).all()
    rendered = estimates_to_readout(results)
    failed_readout = [
        row for row in rendered if row["metric"] == "refunds" and row["segment"] == "GB"
    ]
    assert len(failed_readout) == len(degenerate_rows)
    assert all(
        row["excluded"] == "nonpositive_mean" and row["lift"] is None for row in failed_readout
    )


def test_unit_panel_breakout_refuses_triggered_population_before_relabeling():
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "date": [date(2025, 1, 1)] * 4,
                "country": ["US", "CA", "US", "CA"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        shape="panel",
        unit="unit",
        group="arm",
        date="date",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
    )
    with pytest.raises(CapabilityError) as refused:
        view_readout(
            experiment, resolver, Scope(view="breakout", grain="total", population="triggered")
        )
    assert refused.value.code == "facade.analysis.trigger_unsupported"


def test_run_breakout_bh_family_survives_a_tiny_segment_extreme_ratio_cell():
    """A tiny segment trips estimate_lift's delta-method guard (combined
    log-scale SE >= 0.5), which breakout keys as
    excluded='extreme_ratio'. run() treats the same LiftGuardError as a
    non-rejection; run_breakout must too, not abort the BH family."""
    rng = np.random.default_rng(12)
    rows = []
    for country in ("US", "CA"):
        for group, shift in (("control", 0.0), ("treatment", 1.0)):
            for i in range(500):
                rows.append(
                    {
                        "user_id": f"{country}_{group}_{i}",
                        "variant": group,
                        "day": "2026-01-01",
                        "country": country,
                        "revenue": 10.0 + shift + rng.normal(0, 5.0),
                    }
                )
    # Three units per arm, positive means, non-zero variance: combined
    # log-scale SE is 0.601 >= 0.5, so only the delta-method guard fires.
    for group, values in (("control", (1.0, 5.0, 9.0)), ("treatment", (2.0, 6.0, 10.0))):
        for i, value in enumerate(values):
            rows.append(
                {
                    "user_id": f"GB_{group}_{i}",
                    "variant": group,
                    "day": "2026-01-01",
                    "country": "GB",
                    "revenue": value,
                }
            )
    experiment, resolver = from_frame(
        pd.DataFrame(rows),
        shape="panel",
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["country"],
        plan=AnalysisPlan(
            secondaries=["revenue"],
            view_multiplicity=MultiplicitySpec(correction="bh", q=0.10),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = view_readout(experiment, resolver, _BREAKOUT)
    gb = [r for r in results if r.dimension_value == "GB"]
    assert gb and all(r.excluded == "extreme_ratio" for r in gb)
    assert any(
        r.dimension_value == "US" and r.excluded is None and r.family_q is not None for r in results
    )


def test_breakout_propagates_a_cuped_nonpositive_mean_row_unmodified(con):
    """The previously-caught gap: a CUPED-adjusted mean that turns
    non-positive in exactly one segment must reach breakout as a real
    row (relative_unavailable_reason='nonpositive_arm_mean', a real
    abs_diff) -- NOT a warning-classified exclusion, and unlike the raw
    case above, this one is NOT pre-screened (breakout's own partition
    step only inspects the RAW arm mean, not a CUPED adjustment applied
    later inside estimate_lift), so it reaches estimate_lift and gets
    the row-based fix directly.

    Only run_breakout's own from_definitions/warehouse construction
    reaches this: from_unit_summary/from_moments refuse run_breakout
    by name, and from_unit_panel refuses a declared CUPED covariate --
    so this builds a definitions-backed DuckDB fixture directly, the
    same bypass-of-load() pattern _analysis_with_country_breakout uses
    above, with an added pre-period purchase fact for the covariate.

    GB's treatment arm carries a raw post-period mean of 1.025 (positive,
    passes breakout's raw pre-screen) but a pre-period covariate mean
    (1001.5) two orders of magnitude past its own control arm's (50),
    so the fitted CUPED slope drags its adjusted mean negative while
    every other cell -- GB control and both US arms -- stays positive.
    Verified directly against the unmodified `increment.estimation.cuped
    .fit_cuped` on these exact sums (independent of the guard change
    this test targets, which only touches what happens after the
    adjusted mean is known):
    `fit_cuped([gb_control, gb_treatment]).theta == 0.03`,
    pooled covariate mean `525.75`, giving adjusted means
    `control: 9.75 - 0.03*(50 - 525.75) == 24.0225` (positive) and
    `treatment: 1.025 - 0.03*(1001.5 - 525.75) == -13.2475` (<= 0),
    each with a strictly positive residual variance (0.4167 / 0.0343 --
    not the separate zero-variance guard). The US segment's arms (raw
    and adjusted means 10.0 and 13.0, residual variance 0.0267 each)
    stay ordinary and positive throughout, confirming only
    GB/treatment/adjusted_metric is degenerate.
    """
    from increment import Method

    exposure_ts = datetime(2025, 6, 10, 9, 0, 0)
    pre_ts = datetime(2025, 5, 28, 9, 0, 0)  # inside [2025-05-27, exposure_ts)
    post_ts = datetime(2025, 6, 11, 9, 0, 0)  # strictly after exposure_ts

    # (country, group_id) -> (post-period y values, pre-period x values).
    # GB/treatment's x values run two orders of magnitude past its own
    # control arm's while its y values stay an ordinary positive mean;
    # every other cell keeps x and y on the same scale.
    arms = {
        ("GB", "control"): ([9.0, 9.5, 10.0, 10.5], [50.0, 50.0, 50.0, 50.0]),
        ("GB", "treatment"): ([0.9, 1.0, 1.3, 0.9], [1000.0, 1001.0, 1002.0, 1003.0]),
        ("US", "control"): ([9.8, 10.5, 9.5, 10.2], [50.0, 51.0, 49.0, 50.0]),
        ("US", "treatment"): ([12.8, 13.5, 12.5, 13.2], [50.0, 51.0, 49.0, 50.0]),
    }

    table_name = "cuped_nonpositive_breakout_events"
    if table_name not in con.list_tables():
        exposure_rows: list[dict] = []
        purchase_rows: list[dict] = []
        for (country, group_id), (ys, xs) in arms.items():
            for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
                unit_id = f"{country}_{group_id}_{i}"
                exposure_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": exposure_ts,
                        "event": "page_view",
                        "group_id": group_id,
                        "country_code": country,
                        "revenue": None,
                        "experiment_id": "cuped_nonpositive_breakout_exp",
                    }
                )
                purchase_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": pre_ts,
                        "event": "purchase",
                        "group_id": None,
                        "country_code": country,
                        "revenue": x,
                        "experiment_id": None,
                    }
                )
                purchase_rows.append(
                    {
                        "user_id": unit_id,
                        "ts": post_ts,
                        "event": "purchase",
                        "group_id": None,
                        "country_code": country,
                        "revenue": y,
                        "experiment_id": None,
                    }
                )
        con.create_table(table_name, obj=pd.DataFrame(exposure_rows + purchase_rows))

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "adjusted_metric",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "cuped_nonpositive_breakout_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "n_pre_periods": 14,
                    "plan": {"secondaries": ["adjusted_metric"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    experiment, resolver = make_binding(con, defs)
    results = view_readout(
        experiment,
        resolver,
        _BREAKOUT,
        decision_method=Method(name="cuped", variance_reduction="cuped"),
    )
    cuped_row = next(
        r
        for r in results
        if r.metric == "adjusted_metric" and r.relative_unavailable_reason is not None
    )
    assert cuped_row.dimension_value == "GB"
    assert cuped_row.group_id == "treatment"
    assert cuped_row.relative_unavailable_reason == "nonpositive_arm_mean"
    assert cuped_row.abs_diff is not None
    assert cuped_row.excluded is None  # a real row, not an exclusion
    other_rows = [r for r in results if r.metric == "adjusted_metric" and r is not cuped_row]
    assert other_rows and all(
        r.relative_unavailable_reason is None and r.excluded is None for r in other_rows
    ), "every other GB/US cell must stay a healthy, non-degenerate row"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("policy_correction", "correction"),
    [("bh", "bh"), ("bonferroni", "bonferroni"), ("none", "bh")],
)
def test_run_breakout_registered_raw_frame_results_roundtrip(policy_correction, correction):
    from fractions import Fraction

    import pyarrow as pa

    from increment.estimators import SequentialCell
    from increment.frame import MetricSpec, synthesise_metric
    from increment.readouts import breakout as readout_breakout
    from increment.semantics.design import Randomized
    from increment.semantics.models import MultiplicitySpec
    from increment.sources import BreakoutMomentsSource
    from tests.test_sequential_public_sources import gaussian_plan

    specs = [
        MetricSpec(name=name, type="conversion", window_days=7) for name in ("revenue", "other")
    ]
    family_member = correction == "bh"
    alpha = Fraction(1, 20) if family_member else Fraction(1, 40)
    cells = tuple(
        SequentialCell(
            metric=spec.name,
            group_id="treatment",
            segment=(("country", segment),),
            family=family_member,
            alpha=alpha,
        )
        for spec in specs
        for segment in ("US", "CA")
    )
    plan = gaussian_plan(
        specs,
        law="bernoulli",
        cells=cells,
        q=0.2,
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposed",
    ).model_copy(
        update={
            "view_multiplicity": MultiplicitySpec(
                correction=policy_correction,
                q=0.1 if policy_correction == "bh" else None,
            )
        }
    )
    rows = [
        {
            "unit": f"{segment}-{arm}-{i}",
            "arm": arm,
            "country": segment,
            "day": date(2025, 1, 1),
            "exposed": date(2025, 1, 1),
            "revenue": int(i % 32 < center),
            "other": int(i % 32 < center),
        }
        for segment, control, treatment in (("US", 10, 15), ("CA", 20, 15))
        for arm, center in (("control", control), ("treatment", treatment))
        for i in range(512)
    ]
    experiment, resolver = from_frame(
        pa.Table.from_pylist(rows),
        shape="panel",
        unit="unit",
        group="arm",
        control="control",
        date="day",
        exposure_date="exposed",
        observation_end=date(2025, 1, 20),
        metrics=specs,
        breakouts=["country"],
        plan=plan,
    )

    workflows.capture_sequential(experiment, resolver, finalized=True, as_of=date(2025, 1, 15))
    source = BreakoutMomentsSource(
        [],
        dimension="country",
        metrics=[synthesise_metric(spec) for spec in specs],
        study_id="frame",
        design=Randomized(control_group="control"),
        plan=plan,
    )
    source.adopt_sequential_snapshot(sequential_snapshot(experiment, resolver))
    result = readout_breakout(source, "country", correction=correction, q=0.2)
    inference = source.context.plan.inference
    _assert_registered_breakout_result(result, correction, inference, alpha)


def _assert_registered_breakout_result(result, correction, inference, alpha):
    from increment.breakout.estimates import BreakoutEstimate
    from increment.estimation.sequential import AlwaysValid

    assert isinstance(inference, AlwaysValid)
    assert {(row.metric, row.dimension_value): row.require_lift().value for row in result} == {
        ("revenue", "US"): 0.5,
        ("revenue", "CA"): -0.25,
        ("other", "US"): 0.5,
        ("other", "CA"): -0.25,
    }
    assert all(row.sequential_result is not None for row in result)
    for row in result:
        assert BreakoutEstimate.model_validate_json(row.model_dump_json()) == row
    assert result.metadata is not None
    assert result.metadata.scope is not None
    scopes = result.metadata.scope.families
    assert scopes
    if correction == "bh":
        (scope,) = scopes
        family = scope.family
        assert family is not None
        assert family.correction == "e_bh"
        assert family.axes == ("metric", "arm", "segment")
        assert family.q == float(inference.registration.q)
        assert family.guarantee == "fdr"
        assert len(scope.members) == 4
        assert {row.family_id for row in result} == {scope.family_id}
        assert {row.family_axes for row in result} == {family.axes}
        assert {row.family_q for row in result} == {family.q}
        assert {row.family_guarantee for row in result} == {"finite_sample"}
    else:
        assert len(scopes) == 2
        policies = [scope.family for scope in scopes]
        assert all(policy is not None for policy in policies)
        assert {policy.correction for policy in policies if policy is not None} == {"bonferroni"}
        assert {policy.axes for policy in policies if policy is not None} == {("segment",)}
        assert {policy.q for policy in policies if policy is not None} == {None}
        assert {policy.guarantee for policy in policies if policy is not None} == {"fwer"}
        assert {len(scope.members) for scope in scopes} == {2}
        for scope in scopes:
            assert scope.family is not None
            rows_for_metric = [
                row for row in result if row.metric in {cell.metric for cell in scope.members}
            ]
            assert {row.family_id for row in rows_for_metric} == {scope.family_id}
        assert {row.family_axes for row in result} == {("segment",)}
        assert {row.family_q for row in result} == {None}
        assert all(row.family_id is not None for row in result)
        assert {row.family_guarantee for row in result} == {"finite_sample"}
        assert {row.multiplicity_status for row in result} == {"exploratory_family"}
        assert {
            row.sequential_result.decision_alpha
            for row in result
            if row.sequential_result is not None
        } == {alpha}


def test_run_breakout_removed_kwargs_raise_typeerror(con):
    """alpha=/inference= are no longer breakout-view call-time
    overrides - readouts.breakout() reads both off the source's own
    declared plan now, matching run()'s cutover."""
    from increment.estimators import AlwaysValid

    experiment, resolver = _analysis_with_country_breakout(con)
    with pytest.raises(TypeError):
        view_readout(experiment, resolver, _BREAKOUT, alpha=0.1)
    with pytest.raises(TypeError):
        view_readout(
            experiment,
            resolver,
            _BREAKOUT,
            inference=AlwaysValid(registration=registration()),
        )
    with pytest.raises(TypeError):
        view_readout(experiment, resolver, _BREAKOUT, correction="bonferroni")
    with pytest.raises(TypeError):
        view_readout(experiment, resolver, _BREAKOUT, q=0.05)


def test_run_breakout_refuses_a_declared_margin_metric(con):
    """readouts.breakout() refuses any metric declaring
    Metric.margin/margin_abs with NotImplementedError - breakout rows are
    tested two-sided vs 0 and a per-segment shifted null is not built.
    This pins that refusal on the breakout view's
    definitions-backed path, routed through readouts.breakout().
    model_copy pattern mirrors
    test_analysis_run_native_declared_margin_abs_reaches_the_estimate,
    applied to the breakout path instead of run()."""
    fixture_experiment, fixture_resolver = _analysis_with_country_breakout(con)
    experiment, resolver = make_binding_like(
        fixture_resolver,
        [
            # margin_abs requires an explicitly-declared direction (the semantics
            # layer refuses a defaulted one) -- declare it so this test exercises
            # breakout()'s own margin refusal, not that upstream validation.
            m.model_copy(update={"margin_abs": 1.0, "preferred_direction": "decrease"})
            if m.name == "revenue"
            else m
            for m in metrics(fixture_experiment, fixture_resolver)
        ],
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        view_readout(experiment, resolver, _BREAKOUT)
    from contextlib import contextmanager
    from typing import Any, cast

    from increment.evidence import EvidenceScope, EvidenceScopeResult, evidence_provider
    from tests.analysis_factory import held_source

    source = cast(Any, held_source(resolver))
    builtin = evidence_provider(source)
    scope_entries = 0

    class Provider:
        context = source.context
        evidence = source.evidence
        artifact_experiment = experiment_definition(experiment, resolver)

        def moments(self, request):
            return builtin.moments(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def close(self):
            source.close()

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                nonlocal scope_entries
                scope_entries += 1
                yield source

            return EvidenceScopeResult(scoped(), request)

    assert experiment_definition(experiment, resolver) is not None
    custom_experiment, custom_resolver = bind_pair(
        cast(Any, Provider()), experiment=experiment_definition(experiment, resolver)
    )
    assert experiment_definition(custom_experiment, custom_resolver) is not None
    try:
        with pytest.raises(UnsupportedRequestError) as custom_raised:
            view_readout(custom_experiment, custom_resolver, _BREAKOUT)
        assert custom_raised.value.code == "readout.margin.breakout"
        assert scope_entries == 0
    finally:
        custom_resolver.close()
        resolver.close()
    assert raised.value.code == "readout.margin.breakout"


def test_run_breakout_empty_when_no_breakouts_declared(con, example_definitions):
    """A breakout view returns [] for an experiment with no
    declared breakouts - mirrors breakout_summaries' own empty-dict
    contract, and confirms the breakout view doesn't require breakouts to be
    present to avoid crashing. `pricing_tier_test` is the experiment with
    no breakouts declared here - `new_onboarding_v2` gained a `country`
    breakout for the example notebook (see examples/breakout.py)."""
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="pricing_tier_test"
    )
    assert experiment_definition(experiment, resolver).breakouts == ()

    assert view_readout(experiment, resolver, _BREAKOUT) == []

    with pytest.raises(InvalidRequestError) as raised:
        view_readout(experiment, resolver, _BREAKOUT, metrics=["not_declared"])
    assert raised.value.code == "facade.analysis_config.unknown_metric_declared"


def test_panel_sql_and_summary_sql_with_breakouts_include_segment_sql(con):
    """panel_sql(breakouts=...)/summary_sql(breakouts=...), called
    directly through ``increment.query`` (not the lower-level query
    builders, and not the breakout view/``breakout_summaries``, which are
    tested separately), include one valid, executable SQL entry per
    (metric, breakout) pair under the expected
    ``f"{metric.name}:{breakout.property}:{source_name}"`` key."""
    experiment, resolver = _analysis_with_country_breakout(con)
    breakouts = list(experiment_definition(experiment, resolver).breakouts)
    key = "revenue:country:events"

    panel_sql = query.panel_sql(experiment, resolver, breakouts=breakouts)
    assert key in panel_sql
    panel_df = con.sql(panel_sql[key]).execute()
    assert "country" in panel_df.columns

    summary_sql = query.summary_sql(experiment, resolver, breakouts=breakouts)
    assert key in summary_sql
    summary_df = con.sql(summary_sql[key]).execute()
    assert "country" in summary_df.columns
    assert set(summary_df["country"]) == {"US", "CA"}


def _analysis_with_missing_control_segment(con):
    """Bind a native experiment with a 3-segment country breakout
    where one segment (MX) has ONLY treatment units - no control arm at
    all - to exercise the breakout view's per-segment "no control arm,
    skip the segment" handling without losing the other 2 segments'
    already-computed estimates."""
    if "breakout_missing_control_events" not in con.list_tables():
        exposure_groups = {
            "US": {"control": ["mu1", "mu2"], "treatment": ["mu3", "mu4"]},
            "CA": {"control": ["mu5", "mu6"], "treatment": ["mu7", "mu8"]},
            "MX": {"treatment": ["mu9", "mu10"]},  # deliberately no control group
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                "experiment_id": "breakout_missing_control_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        revenue_by_unit = {
            "mu1": 9.0,
            "mu2": 11.0,
            "mu3": 14.0,
            "mu4": 16.0,
            "mu5": 19.0,
            "mu6": 21.0,
            "mu7": 14.0,
            "mu8": 16.0,
            "mu9": 25.0,
            "mu10": 27.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in revenue_by_unit.items()
        ]
        con.create_table("breakout_missing_control_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM breakout_missing_control_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "breakout_missing_control_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_binding(con, defs)
    return analysis


def test_run_breakout_sparse_segment_failure_aborts_complete_family(con):
    """A treatment-only segment fails its compiled family hypothesis."""
    experiment, resolver = _analysis_with_missing_control_segment(con)

    with pytest.warns(IncrementWarning) as rec:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            with pytest.raises(CapabilityError) as raised:
                view_readout(experiment, resolver, _BREAKOUT)

    assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)

    error = raised.value
    assert error.code == "family.evidence.incomplete"
    assert any("MX" in repr(key) for key in error.context["failed"])  # ty: ignore[not-iterable]


def _analysis_with_encouragement_country_breakout(con):
    """Native encouragement breakout with a declared uptake fact."""
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
    from increment.semantics.models import Definitions

    table_name = "encouragement_breakout_events"
    if table_name not in con.list_tables():
        exposure_groups = {
            "US": {
                "control": [f"ec{i}" for i in range(1, 16)],
                "treatment": [f"et{i}" for i in range(1, 16)],
            },
            "CA": {
                "control": [f"cc{i}" for i in range(1, 16)],
                "treatment": [f"ct{i}" for i in range(1, 16)],
            },
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                "experiment_id": "encouragement_breakout_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        clicked_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 12, 0, 0),
                "event": "clicked",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": None,
                "experiment_id": None,
            }
            for country, groups in exposure_groups.items()
            for uid in groups["treatment"][:10]
        ]
        revenue_by_unit = {
            uid: 10.0
            + 0.2 * (int(uid[2:]) % 5)
            + (
                2.0
                if uid
                in {
                    click_uid
                    for country, groups in exposure_groups.items()
                    for click_uid in groups["treatment"][:10]
                }
                else 0.0
            )
            for uid in country_by_unit
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": revenue_by_unit[uid],
                "experiment_id": None,
            }
            for uid in country_by_unit
        ]
        con.create_table(table_name, obj=exposure_rows + clicked_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "encouragement_breakout_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "end": "2025-06-04",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )
    return make_binding(
        con,
        defs,
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                justification="clicking is the only path from assignment to revenue",
            ),
            one_sided=True,
        ),
    )


def test_native_encouragement_breakout_carries_uptake_moments(con):
    """Native breakout reductions must materialize declared uptake moments."""
    experiment, resolver = _analysis_with_encouragement_country_breakout(con)

    results = view_readout(experiment, resolver, _BREAKOUT)
    assert {row.estimand for row in results} == {"itt", "compliance", "late"}


@pytest.mark.slow
def test_retention_breakout_summary_uses_mature_exposure_cohorts():
    """The small parity fixture also exercises the one-segment control."""
    from tests.parity_harness.cases import _retention_breakout_cohorts_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _retention_breakout_cohorts_case(single_segment=True)
    assert_parity(case, run_case(case))


@pytest.mark.parametrize(
    "ingress",
    ["from_definitions", pytest.param("from_unit_day_artifact", marks=pytest.mark.slow)],
)
def test_unbounded_retention_breakout_refuses_before_warehouse_access(ingress, tmp_path):
    import ibis
    import yaml

    from increment.workflows import breakout_summaries

    con = ibis.duckdb.connect()
    if ingress == "from_definitions":
        from increment import from_definitions

        definitions = _analysis_with_country_breakout(
            con, unbounded_retention=True, return_definitions=True
        )
        definitions_path = tmp_path / "breakout-definitions.yaml"
        definitions_path.write_text(
            yaml.safe_dump(definitions.model_dump(mode="json", exclude_unset=True)),
            encoding="utf-8",
        )
        with pytest.warns(IncrementWarning, match="revenue"):
            experiment, resolver = from_definitions(
                definitions_path, con, experiment_name="breakout_run_exp"
            )
        con.disconnect()
        try:
            with pytest.raises(InvalidRequestError) as caught:
                breakout_summaries(experiment, resolver, metrics=["revenue", "unbounded"])
            assert caught.value.code == "breakout.retention.unbounded"
            assert caught.value.context["names"] == ("unbounded",)
        finally:
            close = getattr(resolver, "close", None)
            if callable(close):
                close()
        return

    from tests.parity_harness.cases import _publish_and_adopt

    adopted = _publish_and_adopt(
        con,
        _analysis_with_country_breakout(con, unbounded_retention=True),
        kinds=("breakout_dimension",),
    )
    con.disconnect()
    try:
        with pytest.raises(InvalidRequestError) as caught:
            breakout_summaries(*adopted, metrics=["revenue", "unbounded"])
        assert caught.value.code == "breakout.retention.unbounded"
        assert caught.value.context["names"] == ("unbounded",)
    finally:
        adopted[1].close()


@pytest.mark.slow
def test_retention_breakout_reports_censoring_once():
    from increment.workflows import breakout_summaries
    from tests.parity_harness.cases import _close_parity_binding, _retention_breakout_cohorts_case

    case = _retention_breakout_cohorts_case(single_segment=True, late_units=20)
    bound = case.build["from_definitions"]()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", IncrementWarning)
            tables = breakout_summaries(*bound, metrics=["d7_retention"])
        assert [w.message.code for w in recorded if isinstance(w.message, IncrementWarning)] == [
            "frame.censoring.dropped_units"
        ]
        totals = tables["d7_retention:store:events"]["group_summary"].to_pylist()
        assert {row["group_id"]: row["n"] for row in totals} == {"control": 20, "treatment": 20}
    finally:
        _close_parity_binding(bound)


def _selected_breakout_sources(tmp_path):
    import ibis
    import pyarrow as pa

    from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region

    events = _with_preassignment_region(_events(n_per_arm=8, trigger_rate=1.0))
    devices = [
        "mobile" if int(user_id[1:]) % 2 else "desktop" for user_id in events["user_id"].to_pylist()
    ]
    events = events.append_column("device", pa.array(devices))
    definitions = (
        _defs_yaml(trigger=None)
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "      - {name: device, column: device, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
            1,
        )
        .replace(
            "exposures:\n",
            "  - name: profiles\n"
            "    sql: SELECT * FROM events\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts: []\n"
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "exposures:\n",
            1,
        )
        .replace(
            "    plan:\n",
            "    breakouts:\n"
            "      - {property: region, source: events}\n"
            "      - {property: device, source: events}\n"
            "      - {property: region, source: profiles}\n"
            "    plan:\n",
            1,
        )
    )
    path = tmp_path / "selected-breakouts.yml"
    path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=events)
    experiment, resolver = from_definitions(path, con, experiment_name="exp", store="none")
    return experiment, cast(Any, resolver), con


def _breakout_scope(*identities):

    return Scope(view="breakout", grain="total", segment=identities)


def _identity_order(cells):
    return tuple(
        dict.fromkeys((cell.identity.key.source, cell.identity.key.dimension) for cell in cells)
    )


@pytest.mark.integration
def test_selected_breakout_preserves_identity_selection_and_definition_order(tmp_path):
    from increment import run
    from increment._evidence_types import BreakoutSlicesRequest
    from increment.evidence import evidence_provider, request_evidence

    experiment, resolver, con = _selected_breakout_sources(tmp_path)
    source = resolver.binding.source
    identities = (
        BreakoutIdentity("device", "events"),
        BreakoutIdentity("region", "profiles"),
    )
    try:
        slices = cast(
            Any,
            request_evidence(
                evidence_provider(source),
                BreakoutSlicesRequest(
                    breakouts=identities,
                    metrics=(source.context.metrics[0],),
                    configs=source.context.configs,
                    breakout_definitions=resolver.binding.experiment.breakouts,
                ),
            ),
        )
        assert tuple(item.identity for item in slices.slices) == identities
        assert len(slices.slices) == len(identities)

        declared = (("events", "region"), ("events", "device"), ("profiles", "region"))
        for scope, expected in (
            (_breakout_scope(), declared),
            (_breakout_scope(BreakoutIdentity("region", None)), (declared[0], declared[2])),
        ):
            requested = experiment.with_views(scope)
            _, (compiled,) = requested.compile()
            assert (
                tuple(
                    (item["source"], item["property"])
                    for item in cast(Any, compiled.requests[0].fields["breakouts"])
                )
                == expected
            )
            assert _identity_order(run(requested, resolver).cells) == expected

        for view in ("daily", "asof"):
            qualified = Scope(
                view=view, grain=view, segment=(BreakoutIdentity("region", "profiles"),)
            )
            assert _identity_order(run(experiment.with_views(qualified), resolver).cells) == (
                ("profiles", "region"),
            )
    finally:
        resolver.close()
        con.disconnect()


@pytest.mark.integration
def test_distinct_property_breakout_selection_keeps_declared_family_results(tmp_path):
    from increment import run

    experiment, resolver, con = _selected_breakout_sources(tmp_path)
    try:
        every = run(experiment.with_views(_breakout_scope()), resolver).cells
        selected = run(
            experiment.with_views(
                _breakout_scope(
                    BreakoutIdentity("region", "profiles"), BreakoutIdentity("device", None)
                )
            ),
            resolver,
        ).cells
    finally:
        resolver.close()
        con.disconnect()

    assert _identity_order(selected) == (("events", "device"), ("profiles", "region"))
    by_key = {cell.identity.key: cell for cell in every}
    assert {(cell.identity.key.source, cell.identity.key.dimension) for cell in every} - set(
        _identity_order(selected)
    ) == {("events", "region")}
    assert [cell.identity.key for cell in selected] == [
        cell.identity.key
        for cell in every
        if (cell.identity.key.source, cell.identity.key.dimension) != ("events", "region")
    ]
    for cell in selected:
        expected = by_key[cell.identity.key]
        assert cell.family == expected.family
        assert cell.summary is not None and expected.summary is not None
        assert cell.summary.point == pytest.approx(expected.summary.point, rel=1e-12)
        assert cell.summary.lower == pytest.approx(expected.summary.lower, rel=1e-12)
        assert cell.summary.upper == pytest.approx(expected.summary.upper, rel=1e-12)


@pytest.mark.integration
def test_explicit_breakout_scope_is_preserved_without_expansion_flag(tmp_path):
    from increment.evidence import EvidenceScope, evidence_provider

    experiment, resolver, con = _selected_breakout_sources(tmp_path)
    source = resolver.binding.source
    provider = evidence_provider(source)
    selected = BreakoutIdentity("region", "profiles")
    try:
        explicit = provider.scope(
            EvidenceScope(
                metrics=tuple(source.context.metrics),
                breakouts=(selected,),
                include_breakouts=False,
            )
        )
        assert explicit.scope.breakouts == (selected,)
        expanded = provider.scope(
            EvidenceScope(metrics=tuple(source.context.metrics), include_breakouts=True)
        )
        assert expanded.scope.breakouts == tuple(
            BreakoutIdentity(item.property, item.source)
            for item in resolver.binding.experiment.breakouts
        )
    finally:
        resolver.close()
        con.disconnect()


def _triggered_region_breakout(tmp_path):
    import datetime as dt

    import ibis

    from increment.artifacts import SourceSnapshotEvidence
    from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region

    events = _with_preassignment_region(_events(n_per_arm=10, trigger_rate=1.0))
    definitions = (
        _defs_yaml()
        .replace(
            "    facts:\n",
            "    properties:\n      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
            1,
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
            1,
        )
        .replace(
            "    plan:\n      secondaries: [revenue]\n",
            "    plan:\n      primary: revenue\n",
            1,
        )
    )
    definitions_path = tmp_path / "triggered-breakout.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=events)
    native = from_definitions(
        definitions_path,
        con,
        experiment_name="exp",
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    return native, con


def _comparable(rows):
    return [
        row.model_dump(exclude={"source_snapshot_id", "family_id", "analysis_population"})
        for row in rows
    ]


@pytest.mark.integration
def test_custom_provider_runs_triggered_breakout_through_declared_population(tmp_path):
    import warnings

    from increment.estimators import IncrementWarning
    from increment.evidence import EvidenceScope, EvidenceScopeResult, evidence_provider
    from tests.analysis_factory import held_source

    (experiment, resolver), con = _triggered_region_breakout(tmp_path)
    source = cast(Any, held_source(resolver))
    assert experiment_definition(experiment, resolver).breakouts
    assert "triggered" in source.evidence.populations
    builtin = evidence_provider(source)
    scope_requests: list[EvidenceScope] = []

    class Provider:
        context = source.context
        evidence = source.evidence
        artifact_experiment = experiment_definition(experiment, resolver)

        def moments(self, request):
            return builtin.moments(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            scope_requests.append(request)
            from contextlib import contextmanager

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    provider_resolver = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            expected = view_readout(
                experiment,
                resolver,
                Scope(view="breakout", grain="total", population="assigned"),
                metrics=["revenue"],
            )
            provider_experiment, provider_resolver = bind_pair(Provider())
            actual = view_readout(
                provider_experiment,
                provider_resolver,
                Scope(view="breakout", grain="total", population="triggered"),
                metrics=["revenue"],
            )
    finally:
        if provider_resolver is not None:
            provider_resolver.close()
        resolver.close()
        con.disconnect()

    assert _comparable(actual) == _comparable(expected)
    assert actual
    assert {row.analysis_population for row in actual} == {"triggered"}
    # Custom providers retain the requested population on the snapshot scope.
    assert [request.population for request in scope_requests if request.snapshot] == ["triggered"]
    slice_populations = {request.population for request in scope_requests if not request.snapshot}
    assert slice_populations == {"triggered"}
    nested_identity_scopes = [
        request for request in scope_requests if request.breakouts and not request.snapshot
    ]
    assert nested_identity_scopes
    assert all(
        tuple(metric.name for metric in request.metrics) == ("revenue",)
        for request in nested_identity_scopes
    )
    assert all(request.population == "triggered" for request in nested_identity_scopes)
    assert all(request.grain is None for request in nested_identity_scopes)


@pytest.mark.integration
def test_provider_delegating_scope_to_native_keeps_triggered_breakouts(tmp_path):
    import warnings

    from increment.estimators import IncrementWarning
    from increment.evidence import evidence_provider
    from tests.analysis_factory import held_source

    (experiment, resolver), con = _triggered_region_breakout(tmp_path)
    source = cast(Any, held_source(resolver))
    builtin = evidence_provider(source)

    class Provider:
        context = source.context
        evidence = source.evidence

        def moments(self, request):
            return builtin.moments(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            return builtin.scope(request)

        def close(self):
            return None

    provider_resolver = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            expected = view_readout(
                experiment,
                resolver,
                Scope(view="breakout", grain="total", population="triggered"),
                metrics=["revenue"],
            )
            provider_experiment, provider_resolver = bind_pair(Provider())
            actual = view_readout(
                provider_experiment,
                provider_resolver,
                Scope(view="breakout", grain="total", population="triggered"),
                metrics=["revenue"],
            )
    finally:
        if provider_resolver is not None:
            provider_resolver.close()
        resolver.close()
        con.disconnect()

    assert expected
    assert {row.analysis_population for row in actual} == {"triggered"}
    assert _comparable(actual) == _comparable(expected)


def test_custom_breakout_slice_refuses_conflicting_explicit_identity(tmp_path):
    from contextlib import contextmanager

    from increment._evidence_types import BreakoutSlicesRequest
    from increment.errors import CodedError
    from increment.evidence import (
        EvidenceScope,
        EvidenceScopeResult,
        evidence_provider,
        request_evidence,
    )
    from tests.analysis_factory import held_source

    (_, native_resolver), con = _triggered_region_breakout(tmp_path)
    source = cast(Any, held_source(native_resolver))
    builtin = evidence_provider(source)
    scope_entries = 0
    reads = 0

    class Provider:
        context = source.context
        evidence = source.evidence

        def moments(self, request):
            nonlocal reads
            reads += 1
            return builtin.moments(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def unit_rows(self, request):
            nonlocal reads
            reads += 1
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            nonlocal scope_entries
            scope_entries += 1

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    try:
        slices = cast(
            Any,
            request_evidence(
                Provider(),
                BreakoutSlicesRequest(breakouts=(BreakoutIdentity("region", "source_a"),)),
            ),
        )
        breakout_slice = slices.slices[0]
        before = scope_entries
        with pytest.raises(CodedError) as raised:
            breakout_slice.source.scope(
                EvidenceScope(breakouts=(BreakoutIdentity("region", "source_b"),))
            )
        assert raised.value.code == "evidence.breakout_unavailable"
        assert scope_entries == before
        assert reads == 0
    finally:
        native_resolver.close()
        con.disconnect()


def test_custom_breakout_scope_rechecks_scoped_unit_row_capability(tmp_path):
    from contextlib import contextmanager
    from dataclasses import replace

    from increment._evidence_types import BreakoutSlicesRequest
    from increment.errors import CodedError
    from increment.evidence import (
        EvidenceScopeResult,
        UnitRowsRequest,
        evidence_provider,
        request_evidence,
    )
    from tests.analysis_factory import held_source

    (_, native_resolver), con = _triggered_region_breakout(tmp_path)
    source = cast(Any, held_source(native_resolver))
    builtin = evidence_provider(source)
    reads = 0

    class ScopedProvider:
        context = source.context
        evidence = replace(source.evidence, unit_rows=False)

        def unit_rows(self, request):
            nonlocal reads
            reads += 1
            return builtin.unit_rows(request)

    class Provider:
        context = source.context
        evidence = source.evidence

        def moments(self, request):
            return builtin.moments(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield ScopedProvider()

            return EvidenceScopeResult(scoped(), request)

    try:
        slices = cast(
            Any,
            request_evidence(
                Provider(),
                BreakoutSlicesRequest(breakouts=(BreakoutIdentity("region", "source_a"),)),
            ),
        )
        slice_source = slices.slices[0].source
        with pytest.raises(CodedError) as raised:
            request_evidence(
                evidence_provider(slice_source),
                UnitRowsRequest(source.context.metrics[0]),
            )
        assert raised.value.code == "evidence.unit_rows_unavailable"
        assert reads == 0
    finally:
        native_resolver.close()
        con.disconnect()


pytestmark = pytest.mark.integration
