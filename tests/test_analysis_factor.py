"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from increment import Analysis
from increment.errors import CapabilityError
from increment.semantics.models import Definitions, Winsorization
from tests.analysis_factory import _recovered_sum, make_analysis, make_analysis_like


def _analysis_with_country_breakout(con):
    """Build an ``Analysis`` with one MeanMetric and a single
    ``country`` breakout (US/CA, deliberately different control-arm means
    so a cross-segment mixup would flip a lift's sign) - bypasses
    ``load()``'s YAML-file requirement the same way
    ``tests/query/test_builders.py``'s ``_analysis_with_cross_source_breakouts``
    does (``__new__`` plus manual attribute assignment), since
    ``Analysis``'s only file-reading step is
    ``load(definitions_path)``.
    """
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
                    "name": "breakout_run_exp",
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

    analysis = make_analysis(con, defs)
    return analysis


def _analysis_with_country_factor(con, exposure_groups=None, suffix=""):
    """Build an ``Analysis`` with one MeanMetric and a single ``country``
    factor (US/CA, 2 control + 2 treatment units per country) - the
    ``factor_summaries`` counterpart to ``_analysis_with_country_breakout``:
    identical synthetic events/definitions shape, but ``factors`` instead
    of ``breakouts`` on the experiment (same ``Analysis.__new__`` bypass;
    see that function's docstring)."""
    table = f"factor_run_events{suffix}"
    experiment = f"factor_run_exp{suffix}"
    if table not in con.list_tables():
        if exposure_groups is None:
            exposure_groups = {
                "US": {"control": ["fu1", "fu2"], "treatment": ["fu3", "fu4"]},
                "CA": {"control": ["fu5", "fu6"], "treatment": ["fu7", "fu8"]},
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
                "experiment_id": experiment,
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        revenue_by_unit = {
            "fu1": 9.0,
            "fu2": 11.0,
            "fu3": 14.0,
            "fu4": 16.0,
            "fu5": 19.0,
            "fu6": 21.0,
            "fu7": 14.0,
            "fu8": 16.0,
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
        con.create_table(table, obj=exposure_rows + purchase_rows)

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
                    "name": experiment,
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "factors": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


class TestFactorSummaries:
    """``Analysis.factor_summaries()`` - the categorical-CUPED input path,
    wired the same way as ``breakout_summaries`` via the shared
    ``_resolve_breakout_props``/``_build_breakout_properties_table``.
    """

    def test_emits_one_row_per_level_and_arm(self, con):
        """factor_summaries(con) materialises one ``group_summary`` row per
        (factor level x arm) for every declared factor - the
        ``factor_summaries`` counterpart to
        ``test_run_breakout_returns_per_segment_estimates``, minus the
        daily table (absorption only ever consumes whole-window moments).

        Pins the actual level set and each cell's ``n``/``sum_y`` against
        the fixture's real values - a plain row-count check would also
        pass if a join misfire dumped everything into one ``"__null__"``
        bin, so this asserts the real per-cell content, not just the
        shape."""
        analysis = _analysis_with_country_factor(con)

        out = analysis.factor_summaries()

        key = "revenue:country:events"
        assert key in out
        tbl = out[key].to_pylist()
        levels = {r["country"] for r in tbl}
        arms = {r["group_id"] for r in tbl}
        assert levels == {"US", "CA"}
        assert arms == {"control", "treatment"}
        assert len(tbl) == len(levels) * len(arms)

        cells = {(r["country"], r["group_id"]): r for r in tbl}
        expected = {
            ("US", "control"): (2, 20.0),
            ("US", "treatment"): (2, 30.0),
            ("CA", "control"): (2, 40.0),
            ("CA", "treatment"): (2, 30.0),
        }
        for cell_key, (n, sum_y) in expected.items():
            cell = cells[cell_key]
            assert cell["n"] == n
            assert _recovered_sum(cell["n"], cell["ref_y"], cell["cy1"]) == pytest.approx(sum_y)

    def test_rejects_percentile_winsorization(self, con):
        analysis = _analysis_with_country_factor(con)
        metric = analysis.metrics[0].model_copy(
            update={"winsorization": Winsorization(upper_percentile=0.9)}
        )
        replacement = make_analysis_like(analysis, metrics=[metric])
        analysis.close()

        with pytest.raises(CapabilityError) as raised:
            replacement.factor_summaries()
        assert raised.value.code == "source.native.operation"

    def test_returns_empty_dict_without_declared_factors(self, con):
        """factor_summaries's early-return path: an experiment that
        declares no ``factors`` returns ``{}`` rather than raising."""
        analysis = _analysis_with_country_breakout(con)
        assert analysis.factor_summaries() == {}

    def test_raises_capability_error_on_seam_family(self):
        """factor_summaries() needs a native Analysis.from_definitions
        instance, guarded the same way breakout_summaries()/run_breakout()
        already are - a frame-backed analysis has no factors to
        summarise."""
        import pandas as pd

        a = Analysis.from_unit_summary(
            pd.DataFrame(
                {"unit": ["a", "b"], "group": ["control", "treatment"], "revenue": [1.0, 2.0]}
            ),
            unit="unit",
            group="group",
            control="control",
            metrics={"revenue": "mean"},
        )
        with pytest.raises(CapabilityError) as raised:
            a.factor_summaries()
        assert raised.value.code == "facade.analysis.operation"

    def test_pre_exposure_factor_ignores_post_exposure_value(self, con):
        """The pre-exposure dispatch wired into
        ``_build_breakout_properties_table`` is shared between breakouts
        and factors via ``_resolve_breakout_props`` - prove it holds for
        ``factor_summaries`` too, not just ``run_breakout``. Unit ``pf1``
        has a pre-exposure 'loyal' value and a POST-exposure 'churned'
        value; factor_summaries must key its moments off the pre-exposure
        value only."""
        if "pre_exposure_factor_events" not in con.list_tables():
            exposure_rows = [
                {
                    "user_id": uid,
                    "ts": datetime(2025, 6, 1, 9, 0, 0),
                    "event": "page_view",
                    "group_id": group_id,
                    "revenue": None,
                    "experiment_id": "pre_exp_factor_exp",
                }
                for uid, group_id in [
                    ("pf1", "control"),
                    ("pf2", "control"),
                    ("pf3", "treatment"),
                    ("pf4", "treatment"),
                ]
            ]
            purchase_rows = [
                {
                    "user_id": uid,
                    "ts": datetime(2025, 6, 2, 9, 0, 0),
                    "event": "purchase",
                    "group_id": None,
                    "revenue": amount,
                    "experiment_id": None,
                }
                for uid, amount in [("pf1", 10.0), ("pf2", 12.0), ("pf3", 20.0), ("pf4", 24.0)]
            ]
            prop_rows = [
                {"user_id": "pf1", "ts": datetime(2025, 5, 30, 9, 0, 0), "segment": "loyal"},
                # Post-exposure row - must never surface in factor_summaries.
                {"user_id": "pf1", "ts": datetime(2025, 6, 3, 9, 0, 0), "segment": "churned"},
                {"user_id": "pf2", "ts": datetime(2025, 5, 30, 9, 0, 0), "segment": "loyal"},
                {"user_id": "pf3", "ts": datetime(2025, 5, 30, 9, 0, 0), "segment": "loyal"},
                {"user_id": "pf4", "ts": datetime(2025, 5, 30, 9, 0, 0), "segment": "loyal"},
            ]
            con.create_table("pre_exposure_factor_events", obj=exposure_rows + purchase_rows)
            con.create_table("pre_exposure_factor_props", obj=prop_rows)

        defs = Definitions.model_validate(
            {
                "dialect": "duckdb",
                "fact_sources": [
                    {
                        "name": "events",
                        "sql": "SELECT * FROM pre_exposure_factor_events",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [
                            {"name": "page_view", "column": None},
                            {"name": "purchase", "column": "revenue"},
                        ],
                        "properties": [],
                    },
                    {
                        "name": "props",
                        "sql": "SELECT * FROM pre_exposure_factor_props",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "facts": [],
                        "properties": [
                            {
                                "name": "segment",
                                "column": "segment",
                                "dtype": "string",
                                "as_of": "pre_exposure",
                            }
                        ],
                    },
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
                        "name": "pre_exp_factor_exp",
                        "exposure": "e",
                        "unit": "user_id",
                        "start": "2025-06-01",
                        "control_group": "control",
                        "plan": {"secondaries": ["revenue"]},
                        "factors": [{"property": "segment", "source": "props"}],
                    }
                ],
            }
        )
        analysis = make_analysis(con, defs)

        out = analysis.factor_summaries()
        tbl = out["revenue:segment:props"].to_pylist()
        levels = {r["segment"] for r in tbl}
        # Only "loyal" (the pre-exposure value) surfaces; "churned" (pf1's post-exposure value) never does, proving the factor path dispatches to the pre-exposure-scoped builder.
        assert levels == {"loyal"}
        cells = {r["group_id"]: r for r in tbl}
        assert cells["control"]["n"] == 2
        assert _recovered_sum(
            cells["control"]["n"], cells["control"]["ref_y"], cells["control"]["cy1"]
        ) == pytest.approx(22.0)
        assert cells["treatment"]["n"] == 2
        assert _recovered_sum(
            cells["treatment"]["n"], cells["treatment"]["ref_y"], cells["treatment"]["cy1"]
        ) == pytest.approx(44.0)

    @pytest.mark.filterwarnings(
        "ignore:factor .* levels survived absorption, below 40:RuntimeWarning"
    )
    def test_absorb_factor_end_to_end(self, con):
        """absorb_factor pivots a real factor_summaries table into
        absorb_one_way's per-level arrays and returns a sharpened ATE
        with a valid confidence interval - the end-to-end path from the
        estimation core's absorption estimator into this facade's query
        output shape."""
        from increment import absorb_factor

        # Three levels: absorption refuses two, where the residual degrees of
        # freedom (K-2) would have to be fabricated.
        analysis = _analysis_with_country_factor(
            con,
            exposure_groups={
                "US": {"control": ["fu1", "fu2"], "treatment": ["fu3", "fu4"]},
                "CA": {"control": ["fu5", "fu6"], "treatment": ["fu7", "fu8"]},
                "GB": {"control": ["fu9", "fu10"], "treatment": ["fu11", "fu12"]},
            },
            suffix="_abs",
        )
        tbl = analysis.factor_summaries()["revenue:country:events"]

        res = absorb_factor(tbl, factor="country", control_group="control")

        assert res.n_levels_used >= 3
        assert res.se > 0
        assert res.lb < res.effect < res.ub
