"""Definitions-source mechanics and public view behavior.

Public daily/as-of/breakout parity is exercised through ``run`` views; direct
source calls below are limited to cache and covariate-request mechanics.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import shutil
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import ibis
import pytest

from examples._seed import seed_event_log
from increment import Method, Scope, from_definitions, from_unit_day_artifact, readouts, workflows
from increment.breakout.estimates import run_daily as _day_axis_values
from increment.evidence import (
    BreakoutIdentity,
    MomentRows,
    MomentsRequest,
    evidence_provider,
    request_evidence,
)
from increment.query.native_source import DefinitionsMomentSource
from increment.semantics.models import (
    AnalysisPlan,
    MeanMetric,
    MultiplicitySpec,
    Winsorization,
)
from increment.workflows import experiment_definition
from tests._override_warnings import allow_declared_overrides
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import (
    bound_source,
    make_binding_like,
    native_connection,
    view_readout,
)
from tests.query.test_builders import _analysis_with_cross_source_breakouts
from tests.test_unit_day_artifact_facade import (
    _expected_context,
    _extensions,
)
from tests.test_unit_day_artifact_facade import (
    _shared_publications as _shared_publications,
)
from tests.test_unit_day_artifact_facade import (
    readonly_publications as readonly_publications,
)

# Metric/guardrail set for new_onboarding_v2: d7_retention carries a
# declared prior binding and a bounded threshold_days band, mixed with calendar-axis metrics.
_ALL_METRICS = ["purchase_rate", "avg_session_duration", "d7_retention"]


def _moments(source, metric, **kwargs) -> MomentRows:
    return cast(
        MomentRows,
        request_evidence(evidence_provider(source), MomentsRequest(metric, **kwargs)),
    )


def _stable_aliases(sql: Any) -> str:
    """Replace ibis's per-op generated unnest aliases with positional names."""
    text = str(sql)
    for index, name in enumerate(dict.fromkeys(re.findall(r"ibis_table_unnest\w*", text))):
        text = text.replace(name, f"unnest_{index}")
    return text


def _estimate_from_row(row: Any) -> Any:
    """Return a row's estimate while validating unavailable-row metadata."""
    # Estimate itself exposes ``value``, ``lb``, and ``ub`` but no row-level
    # reason metadata. Check it before the result-row ``value`` branch so bare
    # Estimate callers (e.g. ``_estimate_tuple(e.lift)``) stay unchanged.
    if (
        hasattr(row, "value")
        and hasattr(row, "lb")
        and hasattr(row, "ub")
        and not hasattr(row, "unavailable")
        and not hasattr(row, "excluded")
    ):
        return row
    if hasattr(row, "lift"):
        estimate = row.lift
        reason = getattr(row, "excluded", None) or getattr(row, "unavailable", None)
    elif hasattr(row, "value"):
        estimate = row.value
        reason = getattr(row, "unavailable", None)
    else:
        estimate = row
        reason = None
    if estimate is None:
        if getattr(row, "reference_kind", None) == "binomial":
            assert row.binomial_set is not None and not row.binomial_set.point_available
        else:
            assert reason is not None
    return estimate


def _estimate_value(row: Any) -> float | None:
    estimate = _estimate_from_row(row)
    return None if estimate is None else estimate.value


def _estimate_tuple(row: Any) -> tuple[float | None, float | None, float | None]:
    estimate = _estimate_from_row(row)
    if estimate is None:
        return None, None, None
    return estimate.value, estimate.lb, estimate.ub


def _optional_estimate_tuple(row: Any) -> tuple[float | None, float | None, float | None]:
    return _estimate_tuple(row)


def _estimate_lb(row: Any) -> float | None:
    estimate = _estimate_from_row(row)
    return None if estimate is None else estimate.lb


def _estimate_ub(row: Any) -> float | None:
    estimate = _estimate_from_row(row)
    return None if estimate is None else estimate.ub


@pytest.fixture(scope="module")
def module_definitions(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A module copy so binding definitions cannot write into examples/."""
    target = tmp_path_factory.mktemp("definitions") / "example-definitions"
    shutil.copytree(Path("examples/definitions"), target)
    return target


@backend_fixture(scope="module")
def seeded_analysis(module_definitions: Path):
    con = ibis.duckdb.connect()
    seed_event_log(con)
    return from_definitions(module_definitions, con, experiment_name="new_onboarding_v2")


@pytest.fixture
def definitions_and_adopted_analyses(readonly_publications):
    """Open fresh consumers of the module's immutable day publication."""
    context = _expected_context("examples/definitions")
    extensions = _extensions(context, "breakout_dimension", "factor_dimension", "assignment_counts")
    _con, native, context, store, ref = readonly_publications(extensions=extensions)
    adopted = from_unit_day_artifact(store, ref, expected_context=context)
    try:
        yield native, adopted
    finally:
        adopted[1].close()


_PUBLIC_KEY_FIELDS = (
    "ds",
    "metric",
    "group_id",
    "method",
    "method_role",
    "estimand",
    "analysis_population",
    "value_scale",
    "ds_basis",
    "dimension",
    "dimension_value",
    "source",
)


def _public_row_key(row: Any) -> tuple[tuple[str, str], ...]:
    payload = row.model_dump()
    return tuple((name, repr(payload[name])) for name in _PUBLIC_KEY_FIELDS if name in payload)


def _public_row_value(row: Any) -> tuple[float | None, float | None, float | None] | None:
    estimate = getattr(row, "lift", None)
    if estimate is None:
        estimate = getattr(row, "value", None)
    if estimate is None:
        return None
    return estimate.value, estimate.lb, estimate.ub


def _run_public_view(
    analysis: tuple[Any, Any],
    view: str,
    metrics: tuple[str, ...],
    dimension: str | None,
) -> list[Any]:
    experiment, resolver = analysis
    segment = () if dimension is None else (BreakoutIdentity(dimension, None),)
    scopes = {
        "daily": Scope(view="daily", grain="daily", values=True, segment=segment),
        "daily_lift": Scope(view="daily", grain="daily", segment=segment),
        "asof": Scope(view="asof", grain="asof", values=True, segment=segment),
        "asof_lift": Scope(view="asof", grain="asof", segment=segment),
        "breakout": Scope(view="breakout", grain="total"),
    }
    if view not in scopes:
        raise AssertionError(f"unknown public view {view!r}")
    return list(view_readout(experiment, resolver, scopes[view], metrics=list(metrics)))


@pytest.mark.slow
@pytest.mark.xdist_group("definitions_readonly")
@pytest.mark.parametrize(
    ("view", "metrics", "dimension"),
    [
        ("daily", ("purchase_rate",), None),
        ("daily_lift", ("purchase_rate",), None),
        ("asof", ("purchase_rate",), None),
        ("asof_lift", ("purchase_rate",), None),
        ("breakout", ("purchase_rate",), "country"),
        ("daily", ("purchase_rate", "avg_session_duration"), None),
        ("daily_lift", ("purchase_rate", "avg_session_duration"), None),
        ("daily", ("purchase_rate", "avg_session_duration"), "country"),
        ("asof", ("purchase_rate",), "country"),
        ("asof_lift", ("purchase_rate",), "country"),
        ("asof", ("purchase_rate", "d7_retention"), None),
        ("asof_lift", ("purchase_rate", "d7_retention"), None),
        ("asof", ("d7_retention",), None),
        ("asof_lift", ("d7_retention",), None),
    ],
)
def test_definitions_and_adopted_public_views_match(
    definitions_and_adopted_analyses,
    view,
    metrics,
    dimension,
):
    """Definitions-built and adopted analyses expose the same keyed values."""
    native, adopted = definitions_and_adopted_analyses
    native_result = _run_public_view(native, view, metrics, dimension)
    adopted_result = _run_public_view(adopted, view, metrics, dimension)
    native_rows = {_public_row_key(row): _public_row_value(row) for row in native_result}
    adopted_rows = {_public_row_key(row): _public_row_value(row) for row in adopted_result}
    assert len(native_result) == len(native_rows)
    assert len(adopted_result) == len(adopted_rows)
    assert native_rows.keys() == adopted_rows.keys()
    for key, native_value in native_rows.items():
        adopted_value = adopted_rows[key]
        if native_value is None:
            assert adopted_value is None
        else:
            assert adopted_value is not None
            for adopted_part, native_part in zip(adopted_value, native_value, strict=True):
                assert adopted_part == pytest.approx(native_part, rel=1e-9, nan_ok=True), key


@pytest.mark.slow
@pytest.mark.xdist_group("definitions_readonly")
@pytest.mark.parametrize("dimension_first", [True, False], ids=["cold", "warm"])
def test_dimensioned_daily_duration_matches_raw_events(
    dimension_first, definitions_and_adopted_analyses
):
    """A fresh adoption must retain seconds, independently of earlier requests."""
    native, adopted = definitions_and_adopted_analyses
    _, native_resolver = native
    con = native_connection(bound_source(native_resolver))
    # The seed has one assignment row per user and a static country. Keep every
    # eligible user, including mobile users and users with no sessions that day.
    oracle = (
        con.sql("""
        WITH eligible AS (
            SELECT user_id, event_at AS exposed_at
            FROM analytics.event_log
            WHERE experiment_id = 'new_onboarding_v2'
              AND event = 'page_view' AND group_id = 'treatment'
              AND country_code = 'US'
              AND CAST(event_at AS DATE) BETWEEN DATE '2025-01-25' AND DATE '2025-01-31'
        ), per_user AS (
            SELECT e.user_id, COALESCE(SUM(s.duration_s), 0.0) AS seconds,
                   COUNT(s.duration_s) AS sessions
            FROM eligible e
            LEFT JOIN analytics.event_log s
              ON s.user_id = e.user_id AND s.event = 'session_end'
             AND s.device_type = 'web' AND s.duration_s IS NOT NULL
             AND s.event_at >= e.exposed_at
             AND CAST(s.event_at AS DATE) = DATE '2025-01-31'
            GROUP BY e.user_id
        )
        SELECT COUNT(*) AS n, AVG(seconds) AS mean_seconds,
               VAR_SAMP(seconds) AS variance_seconds, AVG(sessions) AS mean_sessions
        FROM per_user
    """)
        .to_pyarrow()
        .to_pylist()[0]
    )
    mean = oracle["mean_seconds"]
    # avg_calendar_day divides by one day here; its unit value is daily seconds.
    se = math.sqrt(oracle["variance_seconds"] / oracle["n"]) / mean
    half_width = NormalDist().inv_cdf(0.975) * se
    expected = (mean, mean * math.exp(-half_width), mean * math.exp(half_width))
    assert mean > oracle["mean_sessions"] > 0

    metrics = ("purchase_rate", "avg_session_duration")
    try:
        _, adopted_resolver = adopted
        with adopted_resolver:
            if not dimension_first:
                for analysis in (native, adopted):
                    _run_public_view(analysis, "daily", metrics, None)
            results = [
                _run_public_view(analysis, "daily", metrics, "country")
                for analysis in (native, adopted)
            ]
            estimates = []
            for rows in results:
                matching = [
                    row
                    for row in rows
                    if row.metric == "avg_session_duration"
                    and row.group_id == "treatment"
                    and row.ds == dt.date(2025, 1, 31)
                    and row.dimension_value == "US"
                ]
                assert len(matching) == 1
                row = matching[0]
                assert row.ds_basis == "calendar"
                assert row.dimension == "country"
                assert row.source == "event_log"
                assert row.n == oracle["n"]
                assert row.unavailable is None
                estimate = _estimate_tuple(row)
                assert estimate == pytest.approx(expected, rel=1e-9)
                estimates.append(estimate)
            assert estimates[0] == pytest.approx(estimates[1], rel=1e-9)
    finally:
        native_resolver.close()


@pytest.mark.slow
@pytest.mark.xdist_group("definitions_readonly")
@allow_declared_overrides
def test_adopted_public_run_preserves_inherited_prior(definitions_and_adopted_analyses):
    from increment.estimation.engine import Method
    from increment.estimation.inference import Normal

    native, adopted = definitions_and_adopted_analyses
    replacement = Normal(mu=0.3, sigma=0.001)
    unadjusted = Method(name="unadjusted")

    def first_row(experiment: Any, resolver: Any, **options: Any) -> Any:
        return view_readout(
            experiment, resolver, metrics=["d7_retention"], decision_method=unadjusted, **options
        )[0]

    intervals = []
    for experiment, resolver in (native, adopted):
        inherited = first_row(experiment, resolver)
        cleared = first_row(experiment, resolver, prior=None)
        replaced = first_row(experiment, resolver, prior=replacement)
        unchanged = first_row(experiment, resolver)
        values = [row.posterior_estimate for row in (inherited, cleared, replaced, unchanged)]
        assert values[0] == pytest.approx(values[3])
        assert values[0] != values[1]
        assert values[2] != values[0]
        intervals.append(values)
    for expected, actual in zip(*intervals, strict=True):
        assert actual == pytest.approx(expected)


@pytest.mark.slow
@pytest.mark.xdist_group("definitions_readonly")
def test_adopted_breakout_refusal_has_stable_extension_code(readonly_publications):
    from increment.query.artifact_contract import ArtifactContractError

    context = _expected_context("examples/definitions")
    requests = _extensions(context, "assignment_counts")
    _con, _native_pair, context, store, ref = readonly_publications(extensions=requests)
    adopted_experiment, adopted_resolver = from_unit_day_artifact(
        store, ref, expected_context=context
    )

    with adopted_resolver, pytest.raises(ArtifactContractError) as raised:
        view_readout(
            adopted_experiment,
            adopted_resolver,
            Scope(view="breakout", grain="total"),
            metrics=["purchase_rate"],
        )
    assert raised.value.code == "artifact.extension.missing"


def test_day_axis_source_extends_horizon_for_late_adhoc_metric(example_definitions):
    """A selected metric's late fact rows must extend the day-axis spine."""
    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=80)
    con.raw_sql(
        "INSERT INTO analytics.event_log "
        "(event_at, user_id, session_id, event, revenue, duration_s, "
        "country_code, device_type, plan, experiment_id, group_id) "
        "VALUES ('2025-03-10 12:00:00', 'u00000', 'u00000-late', "
        "'session_end', NULL, 99.0, 'US', 'web', 'free', NULL, NULL)"
    )
    declared_experiment, declared_resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    experiment, resolver = make_binding_like(
        declared_resolver,
        experiment=experiment_definition(declared_experiment, declared_resolver).model_copy(
            update={"end": None}
        ),
    )
    metric = MeanMetric(
        name="late_duration",
        entity="user_id",
        fact="session_end",
        aggregation="sum",
    )
    daily_rows = view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", values=True),
        metrics=["avg_session_duration", metric],
    )
    asof_rows = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof", values=True),
        metrics=["avg_session_duration", metric],
    )

    assert max(row.ds for row in daily_rows) == dt.date(2025, 3, 10)
    assert max(row.ds for row in asof_rows) == dt.date(2025, 3, 10)
    assert max(row.ds for row in daily_rows if row.metric == "avg_session_duration") == dt.date(
        2025, 2, 6
    )
    assert max(row.ds for row in asof_rows if row.metric == "avg_session_duration") == dt.date(
        2025, 3, 10
    )


@pytest.mark.integration
def test_run_daily_shares_native_day_source_across_selected_metrics(
    seeded_analysis: tuple[Any, Any], monkeypatch
):
    experiment, resolver = make_binding_like(seeded_analysis[1])
    source = bound_source(resolver)
    source_type = type(source)
    original = source_type.day_source
    acquired: list[tuple[str, ...]] = []

    def counted(self, *, metrics, population="assigned"):
        acquired.append(tuple(metric.name for metric in metrics))
        return original(self, metrics=metrics, population=population)

    monkeypatch.setattr(source_type, "day_source", counted)
    selected = workflows.metrics(experiment, resolver)[:2]
    with resolver:
        rows = view_readout(
            experiment,
            resolver,
            Scope(view="daily", grain="daily", values=True),
            metrics=list(selected),
        )

    assert rows
    assert acquired == [tuple(metric.name for metric in selected)]


@pytest.mark.slow
def test_panel_cache_distinguishes_same_name_metric_variant(seeded_analysis):
    """A caller's same-name metric variant must not reuse the declared panel."""
    _, seeded_resolver = seeded_analysis
    native = bound_source(seeded_resolver)
    declared = native.context.metrics[0]
    variant = declared.model_copy(update={"window_days": 1})

    def totals(rows):
        return {row["group_id"]: row["n"] * row["ref_y"] + row["cy1"] for row in rows}

    before = totals(_moments(native, declared))
    actual = totals(_moments(native, variant))
    _, resolver = make_binding_like(seeded_resolver, metrics=[variant, *native.context.metrics[1:]])
    with resolver:
        expected = totals(_moments(bound_source(resolver), variant))
    assert actual == pytest.approx(expected)
    assert any(actual[group] != before[group] for group in actual)


# Task 6b: the dimensioned (`dimension=<name>`) day-axis path.


# A single `dimension=` can match multiple declared Breakouts sharing one
# `.property` but resolving to different FactSources.
@backend_fixture()
def cross_source_con():
    return ibis.duckdb.connect()


def test_public_breakout_disambiguates_multiple_fact_sources_run_daily(cross_source_con):
    """`run_daily(dimension="country")` must call `breakout_moments()`
    once per `(breakout, metric)` pair, never merging the two
    `country`-property `Breakout`s (source=src_a, source=src_b) into one
    call - each result row's `source` must match the `Breakout` that
    produced it, and `breakout_moments()` itself must resolve each
    breakout's OWN `FactSource`, never the other's.
    """
    experiment, resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    matching_breakouts = [
        b for b in experiment_definition(experiment, resolver).breakouts if b.property == "country"
    ]
    assert {b.source for b in matching_breakouts} == {"src_a", "src_b"}

    results = view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
    )
    sources_seen = {r.source for r in results}
    assert sources_seen == {"src_a", "src_b"}

    # Every metric x breakout combination is represented under both sources.
    metrics_seen = {r.metric for r in results}
    assert metrics_seen == {"visit_rate", "checkout"}
    for source in ("src_a", "src_b"):
        assert {r.metric for r in results if r.source == source} == {"visit_rate", "checkout"}
    # dimension_value differs by source (src_a: US/CA, src_b: MX); wrong
    # cross-source `properties_table` resolution would still pass above.
    assert {r.dimension_value for r in results if r.source == "src_a"} == {"US", "CA"}
    assert {r.dimension_value for r in results if r.source == "src_b"} == {"MX"}


def test_public_breakout_disambiguates_multiple_fact_sources_run_daily_lift(cross_source_con):
    """`run_daily_lift(dimension="country")` concatenates every matching
    metric's `breakout_moments()` rows into ONE call PER BREAKOUT - the
    two `country` breakouts (src_a, src_b) must each produce their own
    combined call, never a merged one, so both sources' lift estimates
    appear with correctly disambiguated `source=`.
    """
    experiment, resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    matching_breakouts = [
        b for b in experiment_definition(experiment, resolver).breakouts if b.property == "country"
    ]
    assert len(matching_breakouts) == 2

    results = view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", segment=BreakoutIdentity("country", None)),
        decision_method=Method(name="unadjusted"),
    )
    assert {e.source for e in results} == {"src_a", "src_b"}
    for source in ("src_a", "src_b"):
        assert {e.metric for e in results if e.source == source} == {"visit_rate", "checkout"}
    # dimension_value differs by source (src_a: US=control-only so no lift
    # row, CA=treatment; src_b: MX); cross-source mixups would still pass above.
    assert {e.dimension_value for e in results if e.source == "src_a"} == {"CA"}
    assert {e.dimension_value for e in results if e.source == "src_b"} == {"MX"}


def test_public_breakout_disambiguates_multiple_fact_sources_run_asof(cross_source_con):
    """`run_asof(dimension="country")`'s per-`(breakout, metric)`
    `breakout_moments(grain="asof")` calls must keep both `country`
    breakouts' (src_a, src_b) rows disjoint and correctly stamped.
    """
    experiment, resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    matching_breakouts = [
        b for b in experiment_definition(experiment, resolver).breakouts if b.property == "country"
    ]
    assert len(matching_breakouts) == 2

    results = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof", values=True, segment=BreakoutIdentity("country", None)),
    )
    assert {e.source for e in results} == {"src_a", "src_b"}
    for source in ("src_a", "src_b"):
        assert {e.metric for e in results if e.source == source} == {"visit_rate", "checkout"}
    # dimension_value differs by source (src_a: US/CA, src_b: MX); a cross-source mixup would still pass above.
    assert {e.dimension_value for e in results if e.source == "src_a"} == {"US", "CA"}
    assert {e.dimension_value for e in results if e.source == "src_b"} == {"MX"}


def test_public_breakout_disambiguates_multiple_fact_sources_run_asof_lift(cross_source_con):
    """`run_asof_lift(dimension="country")` concatenates EVERY metric's
    `breakout_moments(grain="asof")` rows into one call PER BREAKOUT;
    both `country` breakouts' (src_a, src_b) lift estimates must appear,
    correctly disambiguated by `source=`.
    """
    experiment, resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    matching_breakouts = [
        b for b in experiment_definition(experiment, resolver).breakouts if b.property == "country"
    ]
    assert len(matching_breakouts) == 2

    results = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof", segment=BreakoutIdentity("country", None)),
        decision_method=Method(name="unadjusted"),
    )
    assert {e.source for e in results} == {"src_a", "src_b"}
    for source in ("src_a", "src_b"):
        assert {e.metric for e in results if e.source == source} == {"visit_rate", "checkout"}
    # dimension_value differs by source (src_a: US=control-only so no lift
    # row, CA=treatment; src_b: MX); cross-source mixups would still pass above.
    assert {e.dimension_value for e in results if e.source == "src_a"} == {"CA"}
    assert {e.dimension_value for e in results if e.source == "src_b"} == {"MX"}


def test_multi_source_day_axis_moments_refuse_before_acquisition(cross_source_con, monkeypatch):
    from increment.errors import CapabilityError

    analysis_experiment, analysis_resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    source = bound_source(analysis_resolver)
    source_type = type(source)
    original = source_type.day_source
    acquired: list[object] = []

    def counted(self, *, metrics, population="assigned"):
        acquired.append(tuple(metric.name for metric in metrics))
        return original(self, metrics=metrics, population=population)

    monkeypatch.setattr(source_type, "day_source", counted)
    with pytest.raises(CapabilityError) as refused:
        evidence_provider(source).moments(
            MomentsRequest(
                workflows.metrics(analysis_experiment, analysis_resolver)[0],
                grain="daily",
                by=("country",),
                experiment=experiment_definition(analysis_experiment, analysis_resolver),
            )
        )

    assert refused.value.code == "evidence.breakout_unavailable"
    assert acquired == []


# run_breakout genuinely reads the pre-period covariate (unlike the day-axis
# readouts, which build none) - a regression here silently widens the CI, not crashes.


@backend_fixture(scope="module")
def pre_period_analysis(module_definitions: Path):
    """`new_onboarding_v2` with 14 days of pre-exposure activity
    seeded, so `_breakout_moments_source()`'s CUPED covariate has
    genuine nonzero variance to adjust against.
    """
    con = ibis.duckdb.connect()
    seed_event_log(con, with_pre_period=True)
    return from_definitions(module_definitions, con, experiment_name="new_onboarding_v2")


# Scoped to the two metrics with a genuine pre-period covariate; d7_retention
# shares its fact with the exposure, so it has no independent pre-period signal.
_CUPED_BREAKOUT_METRICS = ["purchase_rate", "avg_session_duration"]


# `@pytest.mark.slow` keeps the CUPED warehouse reduction out of the fast lane.
@pytest.mark.slow
@pytest.mark.parametrize("correction", ["none", "bh"])
def test_public_run_breakout_preserves_cuped_variance_reduction(pre_period_analysis, correction):
    """The public breakout operation carries the pre-period covariate.

    The CUPED-adjusted interval must be materially narrower than the
    unadjusted interval on every estimable segment.
    """
    _, pre_period_resolver = pre_period_analysis
    experiment, resolver = make_binding_like(
        pre_period_resolver,
        plan=AnalysisPlan(
            view_multiplicity=MultiplicitySpec(
                correction=correction,
                q=0.10 if correction == "bh" else None,
            )
        ),
    )
    selected = [
        m for m in workflows.metrics(experiment, resolver) if m.name in _CUPED_BREAKOUT_METRICS
    ]
    breakouts = experiment_definition(experiment, resolver).breakouts
    assert breakouts, "new_onboarding_v2 must declare a country breakout"

    methods = [Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")]

    native_list = list(
        view_readout(
            experiment,
            resolver,
            Scope(view="breakout", grain="total"),
            decision_method=methods[0],
            sensitivity_methods=tuple(methods[1:]),
            metrics=selected,
        )
    )

    if correction == "bh":
        assert len(native_list) == 16
        assert {(e.metric, e.method) for e in native_list} == {
            (metric.name, method.name) for metric in selected for method in methods
        }
        decision_rows = [e for e in native_list if e.method == methods[0].name]
        assert all(e.family_q == pytest.approx(0.10) for e in decision_rows)
        assert all(e.family_axes == ("metric", "arm", "segment") for e in decision_rows)
        assert any(e.discovery is not None for e in native_list)

    if correction != "none":
        # Under BH, selected cells are re-estimated at the FCR level, so a
        # width comparison would mix variance reduction with a per-cell
        # alpha change instead of isolating the covariate.
        return

    widths: dict[tuple[str, str, str], float] = {}
    for e in native_list:
        if e.metric == "avg_session_duration" and e.excluded is None:
            assert _estimate_lb(e) is not None and _estimate_ub(e) is not None
            upper = _estimate_ub(e)
            lower = _estimate_lb(e)
            assert upper is not None and lower is not None
            widths[(e.group_id, e.dimension_value, e.method)] = upper - lower
    segments = {
        (group_id, dim_value)
        for (group_id, dim_value, method) in widths
        if method == "unadjusted" and (group_id, dim_value, "cuped") in widths
    }
    assert segments, "expected at least one estimable avg_session_duration segment"
    for group_id, dim_value in segments:
        unadjusted_w = widths[(group_id, dim_value, "unadjusted")]
        cuped_w = widths[(group_id, dim_value, "cuped")]
        assert cuped_w < unadjusted_w * 0.9, (
            f"cuped width ({cuped_w}) is not materially narrower than "
            f"unadjusted ({unadjusted_w}) for avg_session_duration "
            f"segment {(group_id, dim_value)} -- the pre-period covariate "
            "may be dropped or mis-threaded"
        )


@pytest.mark.filterwarnings(
    "ignore:fetch_arrow_table\\(\\) is deprecated, use to_arrow_table\\(\\) instead\\.:DeprecationWarning"
)
def test_public_run_breakout_disambiguates_multiple_fact_sources(cross_source_con):
    """Public breakout rows preserve each breakout's source identity."""
    _, declared_resolver = _analysis_with_cross_source_breakouts(cross_source_con)
    experiment, resolver = make_binding_like(
        declared_resolver, plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none"))
    )
    matching_breakouts = [
        b for b in experiment_definition(experiment, resolver).breakouts if b.property == "country"
    ]
    assert {b.source for b in matching_breakouts} == {"src_a", "src_b"}

    with pytest.warns(
        UserWarning,
        match=r"run_breakout: segment .* (?:excluded|skipped|no usable|fewer than)",
    ):
        results = view_readout(experiment, resolver, Scope(view="breakout", grain="total"))
    sources_seen = {r.source for r in results}
    assert sources_seen == {"src_a", "src_b"}
    assert {r.metric for r in results} == {"visit_rate", "checkout"}
    for source in ("src_a", "src_b"):
        assert {r.metric for r in results if r.source == source} == {"visit_rate", "checkout"}
    assert {r.dimension_value for r in results if r.source == "src_a"} == {"US", "CA"}
    assert {r.dimension_value for r in results if r.source == "src_b"} == {"MX"}


@pytest.mark.slow
def test_breakout_moments_source_survives_mixed_winsorization_metrics(seeded_analysis):
    """Source and public breakout paths survive mixed winsorization metrics."""
    experiment, resolver = seeded_analysis
    avg_session_duration = next(
        m for m in workflows.metrics(experiment, resolver) if m.name == "avg_session_duration"
    )
    purchase_rate = next(
        m for m in workflows.metrics(experiment, resolver) if m.name == "purchase_rate"
    )
    winsorized = avg_session_duration.model_copy(
        update={"winsorization": Winsorization(upper_value=100_000.0)}
    )

    breakout = experiment_definition(experiment, resolver).breakouts[0]
    src = bound_source(resolver)._breakout_moments_source(
        breakout, metrics=[winsorized, purchase_rate]
    )

    winsorized_rows = cast(list[dict[str, Any]], _moments(src, winsorized, by=(breakout.property,)))
    plain_rows = cast(list[dict[str, Any]], _moments(src, purchase_rate, by=(breakout.property,)))
    assert winsorized_rows, "expected at least one avg_session_duration segment row"
    assert plain_rows, "expected at least one purchase_rate segment row"
    for row in winsorized_rows:
        assert row["winsor_upper_bound"] == pytest.approx(100_000.0)
    for row in plain_rows:
        assert row["winsor_upper_bound"] is None

    rows = view_readout(
        experiment,
        resolver,
        Scope(view="breakout", grain="total"),
        metrics=[winsorized, purchase_rate],
    )
    public_winsorized_rows = [r for r in rows if r.metric == "avg_session_duration"]
    public_plain_rows = [r for r in rows if r.metric == "purchase_rate"]
    assert public_winsorized_rows
    assert public_plain_rows
    # Public breakout rows currently expose estimates but not source-level
    # winsorization bounds; the source assertions above cover that metadata.


@pytest.mark.slow
def test_direct_unadjusted_daily_asof_do_not_query_pre_period_stats(
    seeded_pre_period_con, monkeypatch, example_definitions
):
    """Unadjusted direct reductions avoid pre-period work."""
    experiment, resolver = from_definitions(
        example_definitions,
        seeded_pre_period_con,
        experiment_name="new_onboarding_v2",
        store="none",
    )
    metric = next(
        metric
        for metric in workflows.metrics(experiment, resolver)
        if metric.name == "avg_session_duration"
    )
    source = bound_source(resolver)

    # A clone with no pre-period is the pinned covariate-free execution
    # baseline. Its post-period data and definitions are otherwise identical,
    # so unadjusted-only SQL must be byte-for-byte the same as the
    # pre-period-capable source's unadjusted SQL.
    _, covariate_free_resolver = make_binding_like(
        resolver,
        experiment=experiment_definition(experiment, resolver).model_copy(
            update={"n_pre_periods": 0}
        ),
    )
    covariate_free_source = bound_source(covariate_free_resolver)

    def run_direct(source_: DefinitionsMomentSource) -> tuple[list[Any], list[Any]]:
        daily = _day_axis_values(
            readouts.daily(source_, metrics=[metric.name]), metrics=[metric], view="daily"
        )
        asof = readouts.asof_lift(
            source_, metrics=[metric.name], decision_method=Method(name="unadjusted")
        )
        return daily, asof

    def capture_queries(
        source_: DefinitionsMomentSource,
    ) -> tuple[list[str], tuple[list[Any], list[Any]]]:
        backend = native_connection(source_)
        real_to_pyarrow = backend.to_pyarrow
        queries: list[str] = []

        def capture(expr, *args, **kwargs):
            queries.append(ibis.to_sql(expr, dialect=backend.name))
            return real_to_pyarrow(expr, *args, **kwargs)

        with monkeypatch.context() as capture_patch:
            capture_patch.setattr(backend, "to_pyarrow", capture)
            rows = run_direct(source_)
        return queries, rows

    baseline_queries, _baseline_rows = capture_queries(covariate_free_source)
    direct_queries, _ = capture_queries(source)

    # Unadjusted reductions issue exactly the covariate-free baseline's work:
    # pre-period SQL shows up as an extra query or a different shape.
    assert len(direct_queries) == len(baseline_queries)
    # ibis mints a fresh alias per relational unnest op, so two equivalent
    # builds differ only in those generated names. Compare query shape.
    assert [_stable_aliases(q) for q in direct_queries] == [
        _stable_aliases(q) for q in baseline_queries
    ]


pytestmark = pytest.mark.integration
