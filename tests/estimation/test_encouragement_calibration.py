"""End-to-end calibration of the encouragement-design LATE pipeline.

Unlike ``test_encouragement_recovery.py``, which feeds hand-aggregated
moments straight into ``estimate_encouragement``, these tests exercise the
full user-facing path: raw events in DuckDB, through the query builders'
uptake-aware ``unit_totals``/``group_summary`` reduction, through
``Analysis.run()``'s encouragement dispatch, to the additive Wald-ratio
LATE row - the same chain a real warehouse-backed analysis rides.

DGP (one-sided encouragement): control units can never take up; each
treated unit clicks with probability ``compliance``; outcome
``y = mu0 + tau * d + Normal(0, sd)``. Under the exclusion restriction the
Wald ratio targets ``tau`` exactly, so the true additive LATE is known.
"""

from __future__ import annotations

import math
from datetime import datetime

import ibis
import numpy as np
import pytest
from scipy.stats import norm

from increment.analysis import Analysis
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.semantics.models import Definitions
from tests.analysis_factory import lift_rows, make_analysis

_TS_EXPOSED = datetime(2025, 3, 1, 9, 0, 0)
_TS_CLICKED = datetime(2025, 3, 1, 10, 0, 0)
_TS_PURCHASE = datetime(2025, 3, 1, 11, 0, 0)


def _event_rows(
    rng: np.random.Generator,
    *,
    n_per_arm: int,
    tau: float,
    compliance: float,
    mu0: float = 10.0,
    sd: float = 1.0,
) -> list[dict]:
    """Simulate one replication of the one-sided encouragement DGP as raw
    warehouse events (exposure, uptake click, purchase)."""
    rows: list[dict] = []
    d_treat = rng.binomial(1, compliance, size=n_per_arm)
    for arm, prefix in (("control", "c"), ("treatment", "t")):
        for i in range(n_per_arm):
            unit = f"{prefix}{i}"
            rows.append(
                {
                    "user_id": unit,
                    "ts": _TS_EXPOSED,
                    "event": "exposed",
                    "group_id": arm,
                    "revenue": None,
                    "experiment_id": "enc_cal_exp",
                }
            )
            d = int(d_treat[i]) if arm == "treatment" else 0
            if d:
                rows.append(
                    {
                        "user_id": unit,
                        "ts": _TS_CLICKED,
                        "event": "clicked",
                        "group_id": None,
                        "revenue": None,
                        "experiment_id": None,
                    }
                )
            rows.append(
                {
                    "user_id": unit,
                    "ts": _TS_PURCHASE,
                    "event": "purchase",
                    "group_id": None,
                    "revenue": float(mu0 + tau * d + rng.normal(0.0, sd)),
                    "experiment_id": None,
                }
            )
    return rows


def _native_encouragement_analysis(con, table_name: str) -> Analysis:
    """Build a native (DuckDB, definitions-shaped) ``Analysis`` over
    *table_name* with a declared one-sided :class:`Encouragement` design."""
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
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
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
                }
            ],
            "experiments": [
                {
                    "name": "enc_cal_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-03-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
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
                acknowledged=True, justification="the encouraged button gates revenue"
            ),
            one_sided=True,
        ),
    )
    return analysis


def _run_late(con, table_name: str, rows: list[dict]) -> tuple[float, float, float, float]:
    """Load *rows* into DuckDB, run the real pipeline, and return the
    additive LATE row's (point, se, lb, ub).

    The delta-method SE is recovered from the closed-form Normal interval
    (``lb/ub = value -/+ z * se``) - exact, since ``_nn_estimate`` under
    the default near-flat prior emits precisely that interval.
    """
    con.create_table(table_name, obj=rows)
    analysis = _native_encouragement_analysis(con, table_name)
    results = lift_rows(analysis.run())
    (late,) = [r for r in results if r.estimand == "late" and r.value_scale == "absolute"]
    lift = late.require_lift()
    assert lift.lb is not None and lift.ub is not None and lift.level is not None
    z = norm.ppf((1.0 + lift.level) / 2.0)
    se = (lift.ub - lift.lb) / (2.0 * z)
    return lift.value, se, lift.lb, lift.ub


def test_encouragement_pipeline_smoke_duckdb():
    """Single seeded replication straight through real DuckDB: the full
    query -> estimation -> readout chain returns a finite additive LATE
    with a positive delta-method SE and the right sign (tau = 2 with
    sd = 1 noise at n = 40/arm puts the estimate several SEs above 0)."""
    con = ibis.duckdb.connect()
    rng = np.random.default_rng(7)
    rows = _event_rows(rng, n_per_arm=40, tau=2.0, compliance=0.6)

    tau_hat, se, lb, ub = _run_late(con, "enc_cal_smoke", rows)

    assert math.isfinite(tau_hat)
    assert math.isfinite(se) and se > 0.0
    assert lb < tau_hat < ub
    assert tau_hat > 0.0, "true LATE is +2; the point estimate must carry its sign"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_encouragement_pipeline_late_bias_and_se_calibration_duckdb():
    """Repeated seeded replications of the full DuckDB pipeline: the
    additive LATE is unbiased for the true complier effect, and the
    delta-method SE is calibrated - the empirical SD of the estimates
    matches the mean reported SE, and the 95% CI covers the truth at a
    nominal-consistent rate."""
    n_reps, n_per_arm, tau, compliance = 60, 150, 2.0, 0.6
    con = ibis.duckdb.connect()

    estimates, ses, hits = [], [], 0
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        rows = _event_rows(rng, n_per_arm=n_per_arm, tau=tau, compliance=compliance)
        tau_hat, se, lb, ub = _run_late(con, f"enc_cal_rep_{rep}", rows)
        estimates.append(tau_hat)
        ses.append(se)
        hits += lb <= tau <= ub

    bias = float(np.mean(estimates)) - tau
    # Per-rep SE ~ 0.23 here, so the Monte-Carlo SE of the mean over 60
    # reps is ~ 0.03 - |bias| < 0.10 is a ~3-sigma gate at 5% of truth.
    assert abs(bias) < 0.10, f"LATE bias {bias:+.3f} vs truth {tau}"

    sd_ratio = float(np.std(estimates, ddof=1)) / float(np.mean(ses))
    assert 0.75 <= sd_ratio <= 1.30, (
        f"delta-method SE miscalibrated: empirical SD / mean reported SE = {sd_ratio:.3f}"
    )

    coverage = hits / n_reps
    assert 0.90 <= coverage <= 0.99, f"95% CI coverage {coverage:.3f} outside [0.90, 0.99]"
