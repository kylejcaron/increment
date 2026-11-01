"""Whole-site impact through ``increment.estimators.sitewide`` on the
definitions route: site-volume semantics, arm selection and refusals."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import ibis
import pytest
import yaml

from increment import from_definitions, from_frame
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
from increment.estimators import sitewide
from increment.semantics.loader import load
from increment.semantics.models import MeanMetric, Measure, Metric, QuantileMetric, RatioMetric
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import _make_event_log_table

_REVENUE_PER_USER = MeanMetric(
    name="revenue_per_user",
    entity="user_id",
    fact="purchase",
    aggregation="sum",
)
_REVENUE_PER_SESSION = RatioMetric(
    name="revenue_per_session",
    entity="user_id",
    numerator=Measure(fact="purchase", aggregation="sum"),
    denominator=Measure(fact="session_end", aggregation="count"),
)


def _pricing_experiment(tmp_path, con, metrics: Sequence[Metric] | None = None) -> tuple[Any, Any]:
    """``pricing_tier_test`` from the example definitions on *con*. Given
    *metrics* replace same-named catalog entries and become the plan."""
    payload = load("examples/definitions/").model_dump(mode="json", exclude_unset=True)
    if metrics is not None:
        names = {metric.name for metric in metrics}
        payload["metrics"] = [
            *(metric for metric in payload["metrics"] if metric["name"] not in names),
            *(
                {"type": metric.type, **metric.model_dump(mode="json", exclude_unset=True)}
                for metric in metrics
            ),
        ]
        for experiment in payload["experiments"]:
            if experiment["name"] == "pricing_tier_test":
                experiment["plan"] = {"primary": [metric.name for metric in metrics]}
    path = tmp_path / "pricing-definitions.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with warnings.catch_warnings():
        # Site-volume metrics are unwindowed by construction.
        warnings.filterwarnings("ignore", message="metric .* has window_days=None")
        return from_definitions(path, con, experiment_name="pricing_tier_test")


def _site_noise_rows() -> list[dict]:
    """One 'site noise' unit's purchase + session_end. That unit has no
    exposure row anywhere - it proves ``sitewide()``'s whole-site volume
    has no exposure join. Dated the same day as ``_make_event_log_table``'s
    own keepalive rows, well past every enrolled unit's 7/14-day window
    maturity, so it doubles as a second keepalive anchor rather than
    triggering censoring warnings.
    """
    base = {
        "user_id": "site_noise_1",
        "session_id": "s_site_noise_1",
        "event_at": datetime(2025, 3, 25, 0, 0, 0),
        "experiment_id": None,
        "group_id": None,
        "revenue": None,
        "duration_s": None,
        "country_code": "US",
        "device_type": "web",
        "plan": "free",
    }
    return [
        base | {"event": "purchase", "revenue": 123.45},
        base | {"event": "session_end", "duration_s": 42.0},
    ]


def _treatment_b_rows() -> list[dict]:
    """A third ``pricing_tier_test`` arm: 4 units exposed alongside the
    declared control/treatment pair, the first 2 of them purchasing $30.

    So ``treatment_b`` has mean revenue 60/4 = 15.0 against control's 10.0
    - a delta of 5.0 that is inside the whole-site purchase total and so
    must leave the counterfactual baseline. Same exposure day and same
    day-1 purchase timing as ``_pricing_tier_test_rows``, so every window
    matures against the same keepalive anchor.
    """
    rows: list[dict] = []
    for i in range(4):
        uid = f"tb{i:02d}"
        base = {
            "user_id": uid,
            "session_id": f"s_{uid}",
            "experiment_id": None,
            "group_id": None,
            "revenue": None,
            "duration_s": None,
            "country_code": "US",
            "device_type": "web",
            "plan": "free",
        }
        rows.append(
            base
            | {
                "event_at": datetime(2025, 3, 5, 9, 30 + i, 0),
                "event": "page_view",
                "experiment_id": "pricing_tier_test",
                "group_id": "treatment_b",
            }
        )
        rows.append(
            base
            | {
                "event_at": datetime(2025, 3, 5, 11, 30 + i, 0),
                "event": "session_end",
                "duration_s": 110.0 + i,
            }
        )
        if i < 2:
            rows.append(
                base
                | {
                    "event_at": datetime(2025, 3, 6, 14, 30 + i, 0),
                    "event": "purchase",
                    "revenue": 30.0,
                }
            )
    return rows


def _three_arm_experiment(tmp_path, con) -> tuple[Any, Any]:
    """``pricing_tier_test`` on a three-arm connection, scoped to the one
    sum-type metric these cases read."""
    return _pricing_experiment(tmp_path, con, [_REVENUE_PER_USER])


@backend_fixture()
def pricing_site_noise_con():
    """Fresh DuckDB (NOT the shared session ``con`` fixture - adding rows
    to that one would change every other test in this file) carrying
    ``pricing_tier_test``'s exposed population plus the site-noise unit
    (see :func:`_site_noise_rows`).
    """
    noise_con = ibis.duckdb.connect()
    _make_event_log_table(noise_con, extra_rows=_site_noise_rows())
    return noise_con


@backend_fixture()
def pricing_three_arm_con():
    """Same as ``pricing_site_noise_con`` plus a third enrolled arm, so the
    site total carries TWO treatment arms' lifts on top of non-enrolled
    noise."""
    three_arm_con = ibis.duckdb.connect()
    _make_event_log_table(three_arm_con, extra_rows=[*_site_noise_rows(), *_treatment_b_rows()])
    return three_arm_con


class TestSitewide:
    """``sitewide()`` - native-only whole-window ship-to-all impact, one
    metric at a time."""

    @pytest.mark.integration
    def test_sitewide_takes_metric_keyword(self, tmp_path, pricing_site_noise_con):
        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_USER]
        )
        positional = sitewide(experiment, resolver, "revenue_per_user")
        keyword = sitewide(experiment, resolver, metric="revenue_per_user")
        assert keyword == positional

    @pytest.mark.integration
    def test_sum_metric_includes_non_exposed_site_noise(self, tmp_path, pricing_site_noise_con):
        """A MeanMetric(aggregation='sum') sitewide() reads the WHOLE
        site's purchase revenue in the window, not just enrolled units'
        own totals - the noise purchase (a unit exposed to nothing)
        must still be counted.

        Hand-computed from ``_pricing_tier_test_rows``: control has 10
        converters x $20 = $200, treatment has 14 converters x $40 =
        $560 - the exposed-only total is $760. ``site_total_volume``
        must be $760 + the $123.45 noise purchase = $883.45, proving
        the no-exposure-join behavior differs from a naive exposed-only
        sum.
        """
        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_USER]
        )
        result = sitewide(experiment, resolver, "revenue_per_user")
        assert isinstance(result, SitewideImpact)
        assert result.analysis_population == "assigned"
        assert result.n_control == 20
        assert result.n_treatment == 20
        exposed_only_total = 200.0 + 560.0
        assert result.site_total_volume == pytest.approx(exposed_only_total + 123.45)
        assert result.site_total_volume != pytest.approx(exposed_only_total)
        # Per-unit means: control 200/20=10.0, treatment 560/20=28.0.
        assert result.delta == pytest.approx(18.0)

    @pytest.mark.integration
    def test_invalid_alpha_rejected_before_site_volume_query(
        self, tmp_path, pricing_site_noise_con, monkeypatch
    ):
        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_USER]
        )
        monkeypatch.setattr(
            pricing_site_noise_con,
            "to_pyarrow",
            lambda *_args, **_kwargs: pytest.fail("invalid alpha reached site-volume query"),
        )
        with pytest.raises(InvalidRequestError) as raised:
            sitewide(experiment, resolver, "revenue_per_user", alpha=0.0)
        assert raised.value.code == "estimation.sitewide.alpha"

    @pytest.mark.integration
    def test_ratio_metric_includes_non_exposed_site_noise(self, tmp_path, pricing_site_noise_con):
        """An unwindowed ``revenue_per_session`` (purchase-revenue sum /
        session_end count) sums BOTH parts over the whole site: the noise
        unit's purchase and session_end both count. Denominator: 40 exposed
        units' one session each, the noise unit's session_end, and the
        ``u_keepalive`` session_end inside the open-ended window = 42."""
        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_SESSION]
        )
        result = sitewide(experiment, resolver, "revenue_per_session")
        assert isinstance(result, SitewideRatioImpact)
        assert result.analysis_population == "assigned"
        assert result.n_control == 20
        assert result.n_treatment == 20
        assert result.site_total_numerator == pytest.approx(200.0 + 560.0 + 123.45)
        assert result.site_total_denominator == pytest.approx(42.0)

    @pytest.mark.integration
    def test_windowed_ratio_refuses_before_evidence(self, tmp_path, pricing_site_noise_con):
        experiment, resolver = _pricing_experiment(tmp_path, pricing_site_noise_con)
        with pytest.raises(CapabilityError) as refused:
            sitewide(experiment, resolver, "revenue_per_session")
        assert refused.value.code == "facade.analysis.sitewide_metric_unaligned"
        assert refused.value.context["window_days"] == (7, 7)

    def test_raises_capability_error_on_seam_family(self):
        """``sitewide()`` needs a definitions-backed experiment - a unit
        summary retains no raw event stream to sum site-wide over."""
        import pandas as pd

        df = pd.DataFrame(
            {"user_id": ["u1", "u2"], "variant": ["treatment", "control"], "revenue": [1.0, 2.0]}
        )
        experiment, resolver = from_frame(
            df,
            shape="summary",
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        with pytest.raises(CapabilityError) as raised:
            sitewide(experiment, resolver, "revenue")
        assert raised.value.code == "evidence.operation_unavailable"
        assert raised.value.context["requested"] == "sitewide_evidence"

    @pytest.mark.integration
    def test_winsorized_metric_refuses_with_its_own_code(self, tmp_path, pricing_site_noise_con):
        winsorized = MeanMetric(
            name="revenue_per_user",
            entity="user_id",
            fact="purchase",
            aggregation="sum",
            window_days=14,
            winsorization={"upper_value": 100.0},
        )
        experiment, resolver = _pricing_experiment(tmp_path, pricing_site_noise_con, [winsorized])
        with pytest.raises(CapabilityError) as raised:
            sitewide(experiment, resolver, "revenue_per_user")
        assert raised.value.code == "facade.analysis.sitewide_winsorized_metric"
        assert raised.value.context["metric"] == "revenue_per_user"

    @pytest.mark.integration
    def test_ratio_evidence_without_its_denominator_refuses(
        self, tmp_path, pricing_site_noise_con, monkeypatch
    ):
        import dataclasses

        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_SESSION]
        )
        source = resolver.binding.source
        complete = source.sitewide_evidence
        monkeypatch.setattr(
            source,
            "sitewide_evidence",
            lambda metric, **kwargs: dataclasses.replace(
                complete(metric, **kwargs), site_total_denominator=None
            ),
        )
        with pytest.raises(CapabilityError) as raised:
            sitewide(experiment, resolver, "revenue_per_session")
        assert raised.value.code == "facade.analysis.sitewide_ratio_denominator_missing"
        assert raised.value.context["metric"] == "revenue_per_session"

    @pytest.mark.integration
    def test_unknown_metric_name_raises_value_error(self, tmp_path, con):
        """A metric name never declared on this experiment gets a named
        ValueError, not a raw KeyError - mirrors the
        margins=/null_lifts= unknown-metric-key convention elsewhere in
        this file."""
        experiment, resolver = _pricing_experiment(tmp_path, con)
        with pytest.raises(InvalidRequestError) as raised:
            sitewide(experiment, resolver, "not_a_real_metric")
        assert raised.value.code == "facade.analysis.unknown_metric"

    @pytest.mark.integration
    def test_multi_arm_requires_an_explicit_arm(self, tmp_path, pricing_three_arm_con):
        """With a third enrolled arm, ``sitewide()`` will not pick one for
        you - the refusal names both candidates."""
        experiment, resolver = _three_arm_experiment(tmp_path, pricing_three_arm_con)
        with pytest.raises(InvalidRequestError) as raised:
            sitewide(experiment, resolver, "revenue_per_user")
        assert raised.value.code == "facade.analysis.sitewide_arm_required"
        assert set(raised.value.context["arms"]) == {"treatment", "treatment_b"}  # ty: ignore[invalid-argument-type]

    @pytest.mark.integration
    def test_multi_arm_baseline_nets_out_every_arm(self, tmp_path, pricing_three_arm_con):
        """The counterfactual baseline must subtract BOTH treatment arms'
        contributions, and the ship-to-all population must count every
        enrolled unit.

        Hand-computed: control 10 converters x $20 = $200 over 20 units
        (mean 10.0); treatment 14 x $40 = $560 over 20 (mean 28.0, delta
        18.0); treatment_b 2 x $30 = $60 over 4 (mean 15.0, delta 5.0);
        plus the $123.45 non-enrolled noise purchase, so the site total is
        $943.45. V0 = 943.45 - 18*20 - 5*4 = 563.45 - netting only the
        target arm would leave 583.45, with treatment_b's $20 of lift still
        counted as baseline. N_exp = 20 + 20 + 4 = 44.
        """
        experiment, resolver = _three_arm_experiment(tmp_path, pricing_three_arm_con)
        result = sitewide(experiment, resolver, "revenue_per_user", arm="treatment")

        assert isinstance(result, SitewideImpact)
        assert result.treatment_group == "treatment"
        assert result.n_control == 20
        assert result.n_treatment == 20
        assert result.n_enrolled == 44
        assert result.other_arm_ids == ("treatment_b",)
        assert result.site_total_volume == pytest.approx(943.45)
        assert result.delta == pytest.approx(18.0)
        assert result.baseline_volume == pytest.approx(563.45)
        assert result.baseline_volume != pytest.approx(583.45)
        assert result.absolute_impact == pytest.approx(18.0 * 44)
        assert result.relative_impact == pytest.approx(792.0 / 563.45)

    @pytest.mark.integration
    def test_multi_arm_second_arm_reuses_the_same_baseline(self, tmp_path, pricing_three_arm_con):
        """Scoring ``treatment_b`` answers the same question about a
        different arm: identical all-control baseline, its own lift scaled
        to the same enrolled population."""
        experiment, resolver = _three_arm_experiment(tmp_path, pricing_three_arm_con)
        result = sitewide(experiment, resolver, "revenue_per_user", arm="treatment_b")

        assert isinstance(result, SitewideImpact)
        assert result.treatment_group == "treatment_b"
        assert result.n_treatment == 4
        assert result.other_arm_ids == ("treatment",)
        assert result.delta == pytest.approx(5.0)
        assert result.baseline_volume == pytest.approx(563.45)
        assert result.absolute_impact == pytest.approx(5.0 * 44)
        assert result.relative_impact == pytest.approx(220.0 / 563.45)

    @pytest.mark.integration
    def test_unknown_arm_raises_value_error(self, tmp_path, pricing_three_arm_con):
        """An arm that is not enrolled (here the control group itself) gets
        a named ValueError listing what is available."""
        experiment, resolver = _three_arm_experiment(tmp_path, pricing_three_arm_con)
        with pytest.raises(InvalidRequestError) as raised:
            sitewide(experiment, resolver, "revenue_per_user", arm="control")
        assert raised.value.code == "facade.analysis.sitewide_arm_not_enrolled"

    @pytest.mark.integration
    def test_naming_the_only_arm_matches_omitting_it(self, tmp_path, pricing_site_noise_con):
        """Two-arm backward compatibility: ``arm=`` is optional with one
        treatment arm, and naming it changes nothing."""
        experiment, resolver = _pricing_experiment(
            tmp_path, pricing_site_noise_con, [_REVENUE_PER_USER]
        )

        implicit = sitewide(experiment, resolver, "revenue_per_user")
        explicit = sitewide(experiment, resolver, "revenue_per_user", arm="treatment")

        assert implicit.other_arm_ids == ()
        assert implicit.n_enrolled == 40
        assert explicit == implicit

    @pytest.mark.integration
    def test_retention_metric_capability_error_propagates(self, tmp_path, con):
        """``d7_retention`` (a declared guardrail on ``pricing_tier_test``)
        raises ``site_volume``'s own ``CapabilityError`` unmodified - a
        retention outcome is anchored to each unit's own exposure time, not
        a raw event stream."""
        experiment, resolver = _pricing_experiment(tmp_path, con)
        with pytest.raises(CapabilityError) as raised:
            sitewide(experiment, resolver, "d7_retention")
        assert raised.value.code == "query.builders.site_volume_metric_type"

    @pytest.mark.integration
    def test_quantile_metric_capability_error_propagates(self, tmp_path, con):
        """A ``QuantileMetric`` also has no site-wide reading - a
        distributional statistic is not additive across events."""
        p50_revenue = QuantileMetric(
            name="p50_revenue", entity="user_id", fact="purchase", quantile=0.5
        )
        experiment, resolver = _pricing_experiment(tmp_path, con, [p50_revenue])
        with pytest.raises(CapabilityError) as raised:
            sitewide(experiment, resolver, "p50_revenue")
        assert raised.value.code == "query.builders.site_volume_metric_type"


@pytest.mark.integration
def test_sitewide_refuses_clustered_retention_before_type_check(tmp_path, con):
    """Clustered retention refuses at the cluster-decomposition boundary,
    before the retention type's own site-volume refusal can fire."""
    definitions = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM clustered_retention_events",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "page_view", "column": None},
                ],
            }
        ],
        "exposures": [{"name": "enrolled", "fact": "exposure"}],
        "metrics": [
            {
                "type": "retention",
                "name": "d7_retention",
                "entity": "unit_id",
                "fact": "page_view",
                "threshold_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "clustered_retention",
                "exposure": "enrolled",
                "unit": "unit_id",
                "cluster": "site_id",
                "start": "2025-03-01",
                "end": "2025-03-14",
                "control_group": "control",
                "plan": {"secondaries": ["d7_retention"]},
            }
        ],
    }
    path = tmp_path / "clustered-retention.yaml"
    path.write_text(yaml.safe_dump(definitions), encoding="utf-8")
    experiment, resolver = from_definitions(path, con, experiment_name="clustered_retention")

    with pytest.raises(CapabilityError) as raised:
        sitewide(experiment, resolver, "d7_retention", alpha=0.05)
    assert raised.value.code == "facade.analysis.sitewide_cluster_metric_type"
