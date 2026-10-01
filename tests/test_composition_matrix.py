"""The composition matrix: what every capability does on every metric type.

Three shipped capabilities each failed to compose with a secondary metric
type, and twice the failure was a silently wrong number rather than a loud
refusal. This file makes "what does capability C do on metric type T?" a
declared, tested fact:

- ``MATRIX`` declares one cell per (capability x metric type). It is
  declared in ``tests/compatibility_catalog.py``, not here. Every cell is
  SUPPORTED (probed end to end), REFUSED (the raise and a message fragment
  are pinned), NA (structurally unreachable, with the refusal that proves
  it), or SILENT (a known composition gap, pinned as-is and filed). There is
  deliberately no UNTESTED status.
- The metric-type axis is read off the ``Metric`` union in
  ``increment/semantics/models.py`` at collection time, so a new metric type
  fails ``test_matrix_is_exhaustive`` until its cells are declared.
- The capability axis is curated as ``CAPABILITIES`` in
  ``tests/compatibility_catalog.py``; ``test_public_api_is_classified`` keeps
  the curation honest by forcing every name exported from the root, results,
  and power public tiers into a bucket.
- ``PAIRS``, also declared in ``tests/compatibility_catalog.py``, lists the
  curated capability x capability crosses (cluster x cuped, sequential x
  observational, ...) the 2-D matrix cannot carry.

Adding a capability: declare its entry point in ``PUBLIC_API`` (below, this
file), mapping to a new ``CAPABILITIES`` row in
``tests/compatibility_catalog.py``; add its column to ``MATRIX`` there; and
write its probe here.
Adding a metric type: declare its cell in every ``MATRIX`` column in
``tests/compatibility_catalog.py``.

Probes assert composition (runs / refuses / unreachable), never statistical
quality - calibration lives in each capability's own suite. Probes fail on
any warning not declared by their cell; the repository-wide warning policy
owns ordering and the sole narrowly scoped Ibis/DuckDB deprecation exemption.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Callable
from typing import Any, cast, get_args

import pyarrow as pa
import pytest

from increment import readouts
from increment.errors import CapabilityError, DefinitionError, InvalidRequestError
from increment.estimation.engine import Method
from increment.frame import MetricSpec, from_unit_panel, from_unit_summary
from increment.semantics import models as semantic_models
from increment.semantics.design import (
    AdjustmentSet,
    Encouragement,
    ExclusionRestriction,
    Observational,
    Randomized,
    UptakeSpec,
)
from increment.semantics.models import AnalysisPlan, Definitions, InferenceSpec
from tests.compatibility_catalog import (
    CAPABILITIES,
    MATRIX,
    PAIRS,
    Cell,
)
from tests.warning_codes import warning_codes

# The module-scoped ``native`` fixture runs eight analyses; one worker builds
# it once instead of every worker that draws a probe rebuilding it.
pytestmark = pytest.mark.xdist_group("composition_matrix")

# Axes


def metric_types_from_code() -> list[str]:
    """The ``type`` discriminators of the semantics ``Metric`` union.

    Read from the code, never hand-listed: a new union member instantly
    grows the matrix's metric-type axis.
    """
    union, _field = get_args(semantic_models.Metric)
    types = []
    for member in get_args(union):
        (literal_value,) = get_args(member.model_fields["type"].annotation)
        types.append(literal_value)
    return types


METRIC_TYPES = metric_types_from_code()


# Public-API classification tripwire: names exported from the root, results,
# and power tiers are bucketed below. Capability entry points map to a
# CAPABILITIES row (or carry an explicit outside-matrix reason); everything
# else is a type, error, or helper.

_POWER = "outside-matrix: power planning consumes SummaryStats only, no metric-type seam"
_DIAG = "outside-matrix: assignment diagnostic, metric-free"

PUBLIC_API: dict[str, tuple[str, str | None]] = {
    "Analysis": ("capability", "estimate"),
    "Report": ("capability", "report_calendar"),
    "AlwaysValid": ("capability", "sequential"),
    "Observational": ("capability", "observational"),
    "Encouragement": ("capability", "encouragement"),
    "estimate_cate": ("capability", "cate"),
    "validate_cate": ("capability", "cate"),
    "targeting_rule": ("capability", "cate"),
    "select_targeting_rule": ("capability", "cate"),
    "segment_heterogeneity": ("capability", "breakout"),
    "segment_contrast": ("capability", "breakout"),
    "required_sample_size": ("capability", _POWER),
    "achieved_power": ("capability", _POWER),
    "minimum_detectable_effect": ("capability", _POWER),
    "segment_pairwise_required_sample_size": ("capability", _POWER),
    "segment_pairwise_achieved_power": ("capability", _POWER),
    "segment_pairwise_minimum_detectable_effect": ("capability", _POWER),
    "joint_q_power_fixed": ("capability", _POWER),
    "joint_q_power_random": ("capability", _POWER),
    "power_curve": ("capability", _POWER),
    "sample_ratio_mismatch": ("capability", _DIAG),
    "allocation_posterior_bands": ("capability", _DIAG),
    "absorb_factor": (
        "capability",
        "outside-matrix: factor absorption; curated out of the rev-2 capability axis",
    ),
    "segment_rollout_recommendation": ("capability", "rollout"),
    "estimate_policy_contrast": (
        "capability",
        "outside-matrix: logged-policy evidence family; consumes a decision trace, not a metric",
    ),
    "LoggedTrace": ("type", None),
    "PolicyRegistry": ("type", None),
    "TabularPolicy": ("type", None),
    "PolicyValueContrast": ("type", None),
    "Definitions": ("type", None),
    "Winsorization": ("type", None),
    "ExperimentMetric": ("type", None),
    "Normal": ("type", None),
    "Metric": ("type", None),
    "Method": ("type", None),
    "Estimate": ("type", None),
    "BinomialConfidenceSet": ("type", None),
    "LiftEstimate": ("type", None),
    "LiftEstimates": ("type", None),
    "ContrastResult": ("type", None),
    "ContrastResults": ("type", None),
    "SwitchbackAssignmentDiagnostic": ("type", None),
    "BreakoutEstimate": ("type", None),
    "BreakoutEstimates": ("type", None),
    "DailyMetricValue": ("type", None),
    "DailyMetricValues": ("type", None),
    "DailyLiftEstimate": ("type", None),
    "DailyLiftEstimates": ("type", None),
    "HeterogeneitySummary": ("type", None),
    "HeterogeneitySummaries": ("type", None),
    "SegmentEstimate": ("type", None),
    "SegmentEstimates": ("type", None),
    "SummaryStats": ("type", None),
    "Baseline": ("type", None),
    "PowerDesign": ("type", None),
    # The identification union; its members are classified alongside it.
    "Design": ("type", None),
    "PowerResult": ("type", None),
    "PowerCurvePoint": ("type", None),
    "PowerCurve": ("type", None),
    "SRMResult": ("type", None),
    "AllocationBand": ("type", None),
    "AbsorptionResult": ("type", None),
    "SitewideImpact": ("type", None),
    "SitewideRatioImpact": ("type", None),
    "SegmentRolloutResult": ("type", None),
    "RolloutRecommendation": ("type", None),
    "RolloutRecommendations": ("type", None),
    "RolloutSegment": ("type", None),
    "RolloutSegments": ("type", None),
    "Covariate": ("type", None),
    "CateResult": ("type", None),
    "CateScoreState": ("type", None),
    "ClusterScore": ("type", None),
    "ClusterBootstrap": ("type", None),
    "CateValidation": ("type", None),
    "TargetingRule": ("type", None),
    "TargetingSelection": ("type", None),
    "MetricSpec": ("type", None),
    "RelationLocator": ("type", None),
    "RelationRole": ("type", None),
    "UnitDayArtifactRef": ("type", None),
    "ArtifactRelationRef": ("type", None),
    "BaseRelations": ("type", None),
    "Freshness": ("type", None),
    "MeasureManifest": ("type", None),
    "SimpleMetricMeasure": ("type", None),
    "RatioMetricMeasure": ("type", None),
    "MetricMeasure": ("type", None),
    "ArtifactContext": ("type", None),
    "UnitDayArtifactManifest": ("type", None),
    "ExtensionRefBase": ("type", None),
    "BreakoutDimensionExtension": ("type", None),
    "FactorDimensionExtension": ("type", None),
    "ClusterIdentityExtension": ("type", None),
    "CupedPreperiodExtension": ("type", None),
    "AssignmentCountsExtension": ("type", None),
    "TriggerPopulationExtension": ("type", None),
    "EncouragementUptakeExtension": ("type", None),
    "SiteVolumeExtension": ("type", None),
    "ArtifactDigestError": ("type", None),
    "DigestType": ("type", None),
    "FieldSpec": ("type", None),
    "RelationDigests": ("type", None),
    "canonical_schema_bytes": ("helper", None),
    "canonical_row_bytes": ("helper", None),
    "canonical_rows_bytes": ("helper", None),
    "canonical_json_bytes": ("helper", None),
    "schema_sha256": ("helper", None),
    "content_sha256": ("helper", None),
    "digest_relation": ("helper", None),
    "manifest_sha256": ("helper", None),
    "context_sha256": ("helper", None),
    "request_sha256": ("helper", None),
    "extension_definition_sha256": ("helper", None),
    "extension_source_provenance_sha256": ("helper", None),
    "ArtifactExtensionRef": ("type", None),
    "ExtensionRequestBase": ("type", None),
    "BreakoutDimensionRequest": ("type", None),
    "FactorDimensionRequest": ("type", None),
    "ClusterIdentityRequest": ("type", None),
    "CupedPreperiodRequest": ("type", None),
    "AssignmentCountsRequest": ("type", None),
    "TriggerPopulationRequest": ("type", None),
    "EncouragementUptakeRequest": ("type", None),
    "SiteVolumeRequest": ("type", None),
    "ArtifactExtensionRequest": ("type", None),
    "ArtifactExtensionCatalogEntry": ("type", None),
    "ArtifactPublication": ("type", None),
    "ArtifactSnapshot": ("type", None),
    "ArtifactStore": ("type", None),
    "compile_unit_day_artifact_context": ("helper", None),
    "sequential_definition_id": ("helper", None),
    "unit_day_artifact_extension_catalog": ("helper", None),
    "AnalysisPlan": ("type", None),
    "InferenceSpec": ("type", None),
    "MultiplicitySpec": ("type", None),
    "Randomized": ("type", None),
    "UptakeSpec": ("type", None),
    "ExclusionRestriction": ("type", None),
    "AdjustmentSet": ("type", None),
    "IdentificationGate": ("type", None),
    "NotApplicable": ("type", None),
    "MetricTrend": ("type", None),
    "Assignment": ("type", None),
    "ParallelAssignment": ("type", None),
    "IndependentBernoulliOrder": ("type", None),
    "SharedScheduleOrder": ("type", None),
    "SwitchbackWindow": ("type", None),
    "SwitchbackAssignment": ("type", None),
    "ParallelStudyEnvelope": ("type", None),
    "SwitchbackStudyEnvelope": ("type", None),
    "StudyEnvelope": ("type", None),
    "PredictivePrior": ("type", None),
    "SequentialModel": ("type", None),
    "JointReveal": ("type", None),
    "SequentialCell": ("type", None),
    "SequentialRegistration": ("type", None),
    "SequentialCompliancePolicy": ("type", None),
    "SequentialSnapshot": ("type", None),
    "GaussianScoreMixture": ("type", None),
    "capture_sequential_snapshot": ("helper", None),
    "declare_sequential_freeze": ("helper", None),
    "snapshot_from_json": ("helper", None),
    "estimate_sequential": ("helper", None),
    "IdentificationError": ("error", None),
    "CapabilityError": ("error", None),
    "CodedError": ("error", None),
    "DefinitionError": ("error", None),
    "InvalidRequestError": ("error", None),
    "UnsupportedRequestError": ("error", None),
    "WireFormatError": ("error", None),
    "RefusalSpec": ("type", None),
    "impute": ("helper", None),
    "refuse": ("helper", None),
    "to_frame": ("helper", None),
    "__version__": ("helper", None),
}


# Frame-substrate fixtures (tiny in-memory pyarrow tables)

_RANDOMIZED = Randomized(control_group="control")

_ENCOURAGEMENT = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="uptake"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="probe fixture: assignment only moves y via uptake"
    ),
    # Low z-floor keeps LATE emitted on a deliberately tiny fixture.
    min_first_stage_z=0.5,
)

_OBSERVATIONAL = Observational(
    control_group="control", adjustment=AdjustmentSet(covariates=["pre"])
)


def _unit_table(n: int = 16) -> pa.Table:
    """One row per unit: outcome columns for every frame-servable type."""
    groups = ["control" if i % 2 == 0 else "treatment" for i in range(n)]
    return pa.table(
        {
            "user_id": [f"u{i:02d}" for i in range(n)],
            "variant": groups,
            "exposure_date": list(range(n)),
            # The per-unit term keeps revenue continuous: with only a handful of
            # distinct values a quantile's order-statistic bracket would tie-
            # dominate and widen the Woodruff interval rather than matching it.
            "revenue": [
                10.0 + (i % 5) + 0.037 * i + (2.0 if g == "treatment" else 0.0)
                for i, g in enumerate(groups)
            ],
            "converted": [1.0 if i % 3 != 0 else 0.0 for i in range(n)],
            "orders": [1.0 + (i % 3) for i in range(n)],
            "sessions": [2.0 + ((i // 2) % 4) for i in range(n)],
            "pre": [8.0 + (i % 5) + 0.5 * (i % 3) for i in range(n)],
            "uptake": [
                1.0 if (g == "treatment" and i % 4 != 3) or (g == "control" and i % 8 == 1) else 0.0
                for i, g in enumerate(groups)
            ],
            "seg": ["a" if i % 2 == 0 else "b" for i in range(n)],
        }
    )


def _clustered_table(n: int = 80) -> pa.Table:
    """80 units in 40 clusters, arms constant within a cluster."""
    base = _unit_table(n).to_pydict()
    base["store"] = [f"s{i % 40:02d}" for i in range(n)]
    base["variant"] = ["control" if (i % 40) % 2 == 0 else "treatment" for i in range(n)]
    return pa.table(base)


def _panel_table(n: int = 8, days: int = 3) -> pa.Table:
    """One row per unit per day, for the day-axis probes."""
    rows: dict[str, list[Any]] = {
        "user_id": [],
        "variant": [],
        "ds": [],
        "revenue": [],
        "converted": [],
        "orders": [],
        "sessions": [],
    }
    d0 = dt.date(2026, 1, 1)
    for i in range(n):
        g = "control" if i % 2 == 0 else "treatment"
        for d in range(days):
            rows["user_id"].append(f"u{i:02d}")
            rows["variant"].append(g)
            rows["ds"].append(d0 + dt.timedelta(days=d))
            rows["revenue"].append(5.0 + (i % 3) + (1.0 if g == "treatment" else 0.0) + 0.1 * d)
            rows["converted"].append(1.0 if i % 2 else 0.0)
            rows["orders"].append(1.0)
            rows["sessions"].append(2.0)
    return pa.table(rows)


def _frame_spec(metric_type: str, **kw: Any) -> MetricSpec:
    """The smallest MetricSpec of *metric_type* over ``_unit_table`` columns."""
    if metric_type == "mean":
        return MetricSpec(name="revenue", **kw)
    if metric_type == "conversion":
        return MetricSpec(name="converted", type="conversion", **kw)
    if metric_type == "ratio":
        return MetricSpec(
            name="rev_ratio", type="ratio", numerator="orders", denominator="sessions", **kw
        )
    if metric_type == "quantile":
        return MetricSpec(name="revenue", type="quantile", quantile=0.5, **kw)
    if metric_type == "retention":
        return MetricSpec(name="d1", type="retention", threshold_days=(1, 3), **kw)
    raise AssertionError(f"no frame spec builder for metric type {metric_type!r}")


def _summary_source(spec: MetricSpec, *, table: pa.Table | None = None, **kw: Any):
    return from_unit_summary(
        table if table is not None else _unit_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[spec],
        exposure_date="exposure_date",
        **kw,
    )


def _assert_sane_lift(estimates: Any, metric: str) -> None:
    """A SUPPORTED cell's contract: rows exist, right fields, finite point."""
    rows = [e for e in estimates if e.metric == metric]
    assert rows, f"no estimate rows for metric {metric!r}"
    for e in rows:
        assert e.require_lift().value is None or math.isfinite(e.require_lift().value)
        if e.require_lift().lb is not None and e.require_lift().ub is not None:
            assert e.require_lift().lb <= e.require_lift().ub


def _assert_report_layer_unreachable(metric_type: str) -> None:
    """The NA proof for total/active: both construction seams refuse."""
    with pytest.raises(InvalidRequestError) as spec_error:
        MetricSpec(name="x", type=cast("Any", metric_type))
    assert spec_error.value.code == "model.field.literal"
    with pytest.raises(DefinitionError) as def_error:
        Definitions.model_validate(_bad_experiment_defs(metric_type))
    assert def_error.value.code == "definition.invalid"
    reasons = {
        key for key, _ in cast("tuple[tuple[str, str], ...]", def_error.value.context["errors"])
    }
    assert "definition.validate_experiment.metric_report_type" in reasons


def _bad_experiment_defs(metric_type: str) -> dict[str, Any]:
    """A minimal definitions dict putting a report-layer metric on an experiment."""
    metric: dict[str, Any] = {"name": "bad_metric", "type": metric_type, "fact": "purchase"}
    if metric_type != "total":
        metric["entity"] = "user_id"
    if metric_type == "total":
        metric["aggregation"] = "sum"
    if metric_type == "quantile":
        metric["aggregation"] = "sum"
        metric["quantile"] = 0.5
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "purchase", "column": "revenue"}],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "purchase"}],
        "metrics": [metric],
        "experiments": [
            {
                "name": "bad",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "control_group": "control",
                "plan": {"secondaries": ["bad_metric"]},
            }
        ],
    }


def _quantile_experiment_analysis(tmp_path):
    """A loadable native Analysis whose sole metric is a quantile.

    ``export``/``sitewide`` are native-only capabilities with no frame-path
    counterpart, so unlike every other capability's quantile refusal probe
    (routed through ``_frame_spec``/``readouts.run``), these two need a real
    ``Analysis`` - the shared ``native`` fixture's experiments never declare
    a quantile metric, since none of ITS other probes need one on an actual
    experiment. ``sitewide`` resolves the fact table's schema before it
    reaches the quantile-type refusal, so an empty (but schema-matching)
    table is required; ``export`` never queries the backend at all before
    refusing.
    """
    import ibis
    import pyarrow as pa
    import yaml

    from increment import Analysis

    defs_path = tmp_path / "quantile_only.yaml"
    defs_path.write_text(yaml.safe_dump(_bad_experiment_defs("quantile"), sort_keys=False))
    con = ibis.duckdb.connect()
    schema = pa.schema(
        [("user_id", pa.string()), ("ts", pa.timestamp("us")), ("revenue", pa.float64())]
    )
    con.create_table("raw_events", pa.Table.from_pylist([], schema=schema))
    return Analysis("bad", defs_path, con)


def _cell(capability: str, metric_type: str) -> Cell:
    cell = MATRIX[capability].get(metric_type)
    if cell is None:
        pytest.fail(f"no declared cell for ({metric_type} x {capability}) -- declare it in MATRIX")
    return cell


def _expect_refusal(cell: Cell, probe) -> None:
    assert cell.raises is not None
    with pytest.raises(cell.raises) as excinfo:
        probe()
    if cell.code is not None:
        assert getattr(excinfo.value, "code", None) == cell.code, (
            f"catalog declares code={cell.code!r} but the exception actually "
            f"raised carries code={getattr(excinfo.value, 'code', None)!r}"
        )


def _main_with_always_valid_inference(defs_path: Any, con: Any) -> Any:
    """Predeclare the bounded retention target and explicitly finalize its units."""
    from increment import Analysis
    from tests.sequential_cases import registered_native

    analysis = Analysis("exp_main", defs_path, con)
    metric = next(m for m in analysis.metrics if m.name == "d1")
    analysis = registered_native(analysis, metrics=[metric])
    analysis.capture_sequential(finalized=True, as_of=dt.date(2026, 1, 15))
    return analysis


# Native-path fixture: one in-memory DuckDB + minimal definitions, shared by
# every native-only probe (retention support, breakout, export, reports).


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    ibis = pytest.importorskip("ibis")
    pytest.importorskip("duckdb")
    import yaml

    from increment import Analysis, Report

    n = 48
    d0 = dt.datetime(2026, 1, 1, 9, 0, 0)
    rows: list[dict[str, Any]] = []
    for exp in ("exp_main", "exp_cuped", "exp_cluster"):
        for i in range(n):
            arm = "control" if i % 2 == 0 else "treatment"
            rows.append(
                {
                    "user_id": f"u{i:02d}",
                    "ts": d0,
                    "event": "page_view",
                    "group_id": arm,
                    "experiment_id": exp,
                    "revenue": None,
                    "country_code": "US" if i % 4 < 2 else "CA",
                    "store_id": f"s{i:02d}",
                }
            )
    for i in range(n):
        arm = "control" if i % 2 == 0 else "treatment"
        base = {
            "user_id": f"u{i:02d}",
            "group_id": None,
            "experiment_id": None,
            "revenue": None,
            "country_code": "US" if i % 4 < 2 else "CA",
            "store_id": f"s{i:02d}",
        }
        if i % 3 != 0:
            rows.append(
                {
                    **base,
                    "ts": d0 + dt.timedelta(hours=1),
                    "event": "purchase",
                    # The per-unit term keeps revenue continuous: with only a handful
                    # of distinct values a quantile's order-statistic bracket would
                    # tie-dominate and widen the Woodruff interval rather than matching it.
                    "revenue": 10.0 + (i % 5) + 0.037 * i + (2.0 if arm == "treatment" else 0.0),
                }
            )
        for k in range(1 + i % 2):
            rows.append({**base, "ts": d0 + dt.timedelta(hours=2, minutes=k), "event": "order"})
        if (i % 5 < 3) if arm == "treatment" else (i % 5 < 2):
            rows.append({**base, "ts": d0 + dt.timedelta(days=1, hours=2), "event": "page_view"})
        if i % 3 != 1:
            rows.append(
                {
                    **base,
                    "ts": d0 - dt.timedelta(days=3),
                    "event": "purchase",
                    "revenue": 6.0 + (i % 4),
                }
            )
        # Not `i % 2`: arms follow that parity, so the pre-period event would be
        # constant within arm. A real covariate predates assignment, and CUPED's
        # within-arm theta refuses a covariate without within-arm variation.
        if i % 3 == 0:
            rows.append({**base, "ts": d0 - dt.timedelta(days=2), "event": "page_view"})
    # Heartbeats keep every fact stream's observed extent past the last
    # window/band close, so nothing is censored out of the fixture.
    for ev in ("page_view", "purchase", "order"):
        rows.append(
            {
                "user_id": "u00",
                "ts": d0 + dt.timedelta(days=4),
                "event": ev,
                "group_id": None,
                "experiment_id": None,
                "revenue": 0.0 if ev == "purchase" else None,
                "country_code": "US",
                "store_id": "s00",
            }
        )

    defs: dict[str, Any] = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "page_view", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "order", "column": None},
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
        "exposures": [{"name": "assignment", "fact": "page_view"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 3,
            },
            {
                "type": "conversion",
                "name": "purchase_rate",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 3,
            },
            {
                "type": "ratio",
                "name": "rev_per_order",
                "entity": "user_id",
                "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 3},
                "denominator": {"fact": "order", "aggregation": "count", "window_days": 3},
            },
            {
                "type": "retention",
                "name": "d1",
                "entity": "user_id",
                "fact": "page_view",
                "threshold_days": [1, 3],
                "preferred_direction": "increase",
            },
        ],
        "experiments": [
            {
                "name": "exp_main",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-01",
                "observation_end": "2026-01-05",
                "control_group": "control",
                "plan": {
                    "secondaries": ["revenue", "purchase_rate", "rev_per_order"],
                    "guardrails": ["d1"],
                },
                "breakouts": [{"property": "country"}],
            },
            {
                "name": "exp_cuped",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-01",
                "observation_end": "2026-01-05",
                "control_group": "control",
                "n_pre_periods": 7,
                "plan": {"secondaries": ["revenue", "purchase_rate"], "guardrails": ["d1"]},
            },
            {
                "name": "exp_cluster",
                "exposure": "assignment",
                "unit": "user_id",
                "cluster": "store_id",
                "start": "2026-01-01",
                "end": "2026-01-01",
                "observation_end": "2026-01-05",
                "control_group": "control",
                "plan": {"secondaries": ["revenue", "purchase_rate"], "guardrails": ["d1"]},
            },
        ],
    }

    tmp = tmp_path_factory.mktemp("composition")
    defs_path = tmp / "defs.yaml"
    defs_path.write_text(yaml.safe_dump(defs, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)

    main = Analysis("exp_main", defs_path, con)
    cuped = Analysis("exp_cuped", defs_path, con)
    clustered = Analysis("exp_cluster", defs_path, con)
    d1 = next(m for m in main.metrics if m.name == "d1")

    export_path = tmp / "moments.parquet"
    main.export(export_path)
    import pyarrow.parquet as pq

    export_rows = pq.read_table(export_path).to_pylist()
    rehydrated = Analysis.from_moments(
        export_rows,
        metrics=[
            MetricSpec(name="revenue"),
            MetricSpec(name="purchase_rate", type="conversion"),
            MetricSpec(name="rev_per_order", type="ratio", numerator="n", denominator="d"),
            MetricSpec(name="d1", type="retention", threshold_days=(1, 3)),
        ],
        control="control",
    )

    # Report reads the same tables through experiment-less definitions
    # (unwindowed variants plus the report-only and refused types).
    report_defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": defs["fact_sources"],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {
                    "type": "conversion",
                    "name": "purchase_rate",
                    "entity": "user_id",
                    "fact": "purchase",
                },
                {
                    "type": "ratio",
                    "name": "rev_per_order",
                    "entity": "user_id",
                    "numerator": {"fact": "purchase", "aggregation": "sum"},
                    "denominator": {"fact": "order", "aggregation": "count"},
                },
                {
                    "type": "retention",
                    "name": "d1",
                    "entity": "user_id",
                    "fact": "page_view",
                    "threshold_days": [1, 3],
                    "preferred_direction": "increase",
                },
                {
                    "type": "total",
                    "name": "total_revenue",
                    "fact": "purchase",
                    "aggregation": "sum",
                },
                {"type": "active", "name": "wau", "entity": "user_id", "fact": "page_view"},
                {
                    "type": "quantile",
                    "name": "p50_revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                    "quantile": 0.5,
                },
            ],
        }
    )

    class Native:
        run = main.run()
        seq_run = _main_with_always_valid_inference(defs_path, con).run()
        cuped_run = cuped.run(decision_method=Method(name="cuped", variance_reduction="cuped"))
        breakout = main.run_breakout()
        asof = main.run_asof_lift(metrics=[d1])
        cluster_run = clustered.run()
        clustered_analysis = clustered
        analysis = main
        export_metrics = sorted({r["metric"] for r in export_rows})
        rehydrated_run = rehydrated.run()
        report = Report.from_definitions(report_defs, con)

    return Native


_REPORT_METRIC = {
    "mean": "revenue",
    "conversion": "purchase_rate",
    "ratio": "rev_per_order",
    "retention": "d1",
    "quantile": "p50_revenue",
    "total": "total_revenue",
    "active": "wau",
}
_REPORT_KW = {"grain": "day", "start": dt.date(2026, 1, 1), "end": dt.date(2026, 1, 3)}


# Enforcement tests


def test_matrix_is_exhaustive():
    """Every code-derived metric type declares a cell in every capability,
    and the table names no type or capability the code does not have."""
    assert set(MATRIX) == set(CAPABILITIES), (
        "MATRIX columns and CAPABILITIES disagree -- a new capability must "
        "declare a full column of cells"
    )
    missing = [(t, cap) for cap in CAPABILITIES for t in METRIC_TYPES if t not in MATRIX[cap]]
    assert not missing, (
        f"undeclared matrix cells {missing}: a new metric type must declare "
        "SUPPORTED/REFUSED/NA for every capability"
    )
    stale = [(t, cap) for cap in CAPABILITIES for t in MATRIX[cap] if t not in METRIC_TYPES]
    assert not stale, f"matrix declares cells for unknown metric types: {stale}"


def test_matrix_statuses_are_well_formed():
    """Refused cells pin a fragment and an exception; NA cells state a reason."""
    for cap, column in MATRIX.items():
        for t, cell in column.items():
            if cell.status == "refused":
                assert cell.fragment and cell.raises, (cap, t)
            if cell.status == "na":
                assert cell.reason, (cap, t)
            if cell.status == "silent":
                assert cell.note, (cap, t)
    for pair, cell in PAIRS.items():
        if cell.status == "refused":
            assert cell.fragment and cell.raises, pair


# Capability probes, one per capability, parameterized over metric types


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_estimate(metric_type, native):
    cell = _cell("estimate", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if metric_type == "retention":
        _assert_sane_lift(list(native.run), "d1")
        return
    spec = _frame_spec(metric_type)
    _assert_sane_lift(readouts.run(_summary_source(spec, design=_RANDOMIZED)), spec.name)


def test_retention_frame_seam_refusal_still_blankets_the_summary_path():
    """The pinned seam refusal: a one-row-per-unit summary cannot carry a
    retention band, so every summary-substrate capability refuses at entry."""
    with pytest.raises(CapabilityError) as raised:
        _summary_source(_frame_spec("retention"))
    assert raised.value.code == "source.frame.constructor"


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_cuped(metric_type, native):
    cell = _cell("cuped", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    cuped_method = [Method(name="cuped", variance_reduction="cuped")]
    if metric_type == "quantile":
        _expect_refusal(cell, lambda: _frame_spec("quantile", covariate="pre"))
        return
    if metric_type == "retention":
        rows = [e for e in native.cuped_run if e.metric == "d1"]
        assert rows and rows[0].method == "cuped"
        _assert_sane_lift(list(native.cuped_run), "d1")
        return
    if cell.status == "refused":
        spec = _frame_spec(metric_type)
        _expect_refusal(
            cell,
            lambda: readouts.run(
                _summary_source(spec, design=_RANDOMIZED), decision_method=cuped_method[0]
            ),
        )
        return
    spec = _frame_spec(metric_type, covariate="pre")
    _assert_sane_lift(
        readouts.run(_summary_source(spec, design=_RANDOMIZED), decision_method=cuped_method[0]),
        spec.name,
    )


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_cluster(metric_type, native):
    cell = _cell("cluster", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if metric_type == "retention":
        rows = [e for e in native.cluster_run if e.metric == "d1"]
        assert rows and rows[0].n_clusters == 48
        _assert_sane_lift(list(native.cluster_run), "d1")
        return
    spec = _frame_spec(metric_type)
    if cell.status == "refused":
        _expect_refusal(
            cell, lambda: _summary_source(spec, table=_clustered_table(), cluster="store")
        )
        return
    src = _summary_source(spec, table=_clustered_table(), cluster="store", design=_RANDOMIZED)
    estimates = readouts.run(src)
    _assert_sane_lift(estimates, spec.name)
    assert all(e.n_clusters == 40 for e in estimates)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_sequential(metric_type, native):
    cell = _cell("sequential", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if metric_type == "retention":
        _assert_sane_lift(list(native.seq_run), "d1")
        return
    from increment.estimation.results import LiftEstimate
    from increment.frame import synthesise_metric
    from tests.sequential_cases import declared_plan

    spec = _frame_spec(metric_type)
    av_plan = declared_plan(
        [synthesise_metric(spec)], source_id="frame", design=_RANDOMIZED, transformations=[spec]
    )

    def probe():
        return readouts.run(_summary_source(spec, design=_RANDOMIZED, plan=av_plan))

    if cell.status == "refused":
        _expect_refusal(cell, probe)
        return
    result = probe()
    _assert_sane_lift(result, spec.name)
    for row in result:
        certified = row.require_sequential_result()
        assert certified.checkpoint.control.n > 0
        assert LiftEstimate.model_validate_json(row.model_dump_json()) == row


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_observational(metric_type):
    cell = _cell("observational", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if metric_type == "retention":
        _expect_refusal(
            cell,
            lambda: _summary_source(
                _frame_spec("retention"), table=_unit_table(80), design=_OBSERVATIONAL
            ),
        )
        return
    spec = _frame_spec(metric_type)
    src = _summary_source(spec, table=_unit_table(80), design=_OBSERVATIONAL)
    if metric_type == "ratio":
        assert cell.raises is not None
        for method in ("iptw", "dml", "aipw"):
            with pytest.raises(cell.raises) as excinfo:
                readouts.run(src, decision_method=Method(name=method))
            assert getattr(excinfo.value, "code", None) == cell.code
            assert cast(Any, excinfo.value).context["method"] == method
            assert cast(Any, excinfo.value).context["metric"] == spec.name
        return
    if cell.status == "refused":
        _expect_refusal(cell, lambda: readouts.run(src))
        return
    for method in ("iptw", "dml", "aipw"):
        estimates = readouts.run(src, decision_method=Method(name=method))
        _assert_sane_lift(estimates, spec.name)
        assert all(e.method == method for e in estimates)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_encouragement(metric_type):
    cell = _cell("encouragement", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return

    def run_enc(spec: MetricSpec):
        src = _summary_source(spec, uptake="uptake", design=_ENCOURAGEMENT)
        return readouts.run(src)

    if metric_type == "retention":
        _expect_refusal(
            cell,
            lambda: _summary_source(
                _frame_spec("retention"), uptake="uptake", design=_ENCOURAGEMENT
            ),
        )
        return
    if cell.status == "refused":
        _expect_refusal(cell, lambda: run_enc(_frame_spec(metric_type)))
        return
    rows = run_enc(_frame_spec(metric_type))
    estimands = {e.estimand for e in rows}
    assert {"itt", "compliance", "late"} <= estimands
    _assert_sane_lift(rows, _frame_spec(metric_type).name)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_cate(metric_type):
    from increment import estimate_cate

    cell = _cell("cate", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return

    def probe():
        spec = _frame_spec(metric_type)
        src = _summary_source(spec, table=_unit_table(40))
        return estimate_cate(src, spec.name, control="control", interact=["pre"])

    if metric_type == "retention":
        _expect_refusal(
            cell, lambda: _summary_source(_frame_spec("retention"), table=_unit_table(40))
        )
        return
    if cell.status == "refused":
        _expect_refusal(cell, probe)
        return
    result = probe()
    assert math.isfinite(float(result.ate))


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_breakout(metric_type, native):
    cell = _cell("breakout", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if cell.status == "refused":
        spec = _frame_spec(metric_type)
        _expect_refusal(
            cell, lambda: readouts.run(_summary_source(spec, design=_RANDOMIZED), by=["seg"])
        )
        return
    name = _REPORT_METRIC[metric_type]
    rows = [e for e in native.breakout if e.metric == name]
    segments = {e.dimension_value for e in rows}
    assert segments == {"US", "CA"}
    for e in rows:
        assert e.require_lift().value is None or math.isfinite(e.require_lift().value)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_daily_asof(metric_type, native):
    cell = _cell("daily_asof", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if metric_type == "retention":
        rows = [e for e in native.asof if e.metric == "d1"]
        assert rows and all(e.ds is not None for e in rows)
        _assert_sane_lift(rows, "d1")
        return
    spec = _frame_spec(metric_type)
    src = from_unit_panel(
        _panel_table(),
        unit="user_id",
        group="variant",
        control="control",
        date="ds",
        metrics=[spec],
    )
    if cell.status == "refused":
        # Named refusal at the moments seam (quantile daily); the fragment
        # pin means a rewording must update this cell.
        _expect_refusal(cell, lambda: readouts.daily(src))
        return
    rows = readouts.daily(src)
    assert len(rows) == 6  # 3 days x 2 arms
    assert all(r["metric"] == spec.name and r["ds"] is not None for r in rows)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_report_calendar(metric_type, native):
    cell = _cell("report_calendar", metric_type)
    name = _REPORT_METRIC[metric_type]
    if cell.status == "refused":
        _expect_refusal(cell, lambda: native.report.metric(name, **_REPORT_KW))
        return
    frame = native.report.metric(name, **_REPORT_KW).to_frame(backend="pyarrow")
    assert frame.num_rows == 3  # one row per day in the window
    day0 = [r for r in frame.to_pylist() if r["period"] == dt.date(2026, 1, 1)]
    assert day0 and day0[0]["value"] is not None and math.isfinite(day0[0]["value"])
    if metric_type in ("ratio", "total", "active"):
        assert day0[0]["ci_lb"] is None or math.isnan(day0[0]["ci_lb"])
    else:
        assert day0[0]["ci_lb"] is not None and math.isfinite(day0[0]["ci_lb"])


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_report_window(metric_type, native):
    cell = _cell("report_window", metric_type)
    name = _REPORT_METRIC[metric_type]
    if cell.status == "refused":
        _expect_refusal(cell, lambda: native.report.metric(name, window=2, **_REPORT_KW))
        return
    trend = native.report.metric(name, window=2, **_REPORT_KW)
    assert trend.window == 2
    frame = trend.to_frame(backend="pyarrow")
    assert frame.num_rows == 3
    assert all(r["value"] is not None for r in frame.to_pylist())


_MOMENT_NULLS = {
    "sum_x": None,
    "sum_x2": None,
    "sum_xy": None,
    "sum_den": None,
    "sum_den2": None,
    "sum_yden": None,
}


def _moment_rows(metric_type: str) -> list[dict[str, Any]]:
    """Raw additive sums for one two-arm metric, format-1 shape."""
    if metric_type in ("mean", "quantile"):
        return [
            {
                "experiment_id": "e1",
                "metric": "m",
                "group_id": "control",
                "n": 12,
                "sum_y": 120.0,
                "sum_y2": 1208.0,
                **_MOMENT_NULLS,
            },
            {
                "experiment_id": "e1",
                "metric": "m",
                "group_id": "treatment",
                "n": 12,
                "sum_y": 144.0,
                "sum_y2": 1736.0,
                **_MOMENT_NULLS,
            },
        ]
    if metric_type in ("conversion", "retention"):
        return [
            {
                "experiment_id": "e1",
                "metric": "m",
                "group_id": "control",
                "n": 12,
                "sum_y": 5.0,
                "sum_y2": 5.0,
                **_MOMENT_NULLS,
            },
            {
                "experiment_id": "e1",
                "metric": "m",
                "group_id": "treatment",
                "n": 12,
                "sum_y": 8.0,
                "sum_y2": 8.0,
                **_MOMENT_NULLS,
            },
        ]
    assert metric_type == "ratio"
    return [
        {
            "experiment_id": "e1",
            "metric": "m",
            "group_id": "control",
            "n": 12,
            "sum_y": 120.0,
            "sum_y2": 1208.0,
            **_MOMENT_NULLS,
            "sum_den": 24.0,
            "sum_den2": 50.0,
            "sum_yden": 242.0,
        },
        {
            "experiment_id": "e1",
            "metric": "m",
            "group_id": "treatment",
            "n": 12,
            "sum_y": 144.0,
            "sum_y2": 1736.0,
            **_MOMENT_NULLS,
            "sum_den": 26.0,
            "sum_den2": 60.0,
            "sum_yden": 314.0,
        },
    ]


def _current_moment_rows(metric_type: str) -> list[dict[str, Any]]:
    from increment.decision_wire import compiled_plan_to_json
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.frame import synthesise_metric
    from increment.plan import compile_decision_plan

    metric = _moment_spec(metric_type)
    wire = compiled_plan_to_json(compile_decision_plan(None, [synthesise_metric(metric)]))
    winsor = dict.fromkeys(
        (
            "winsor_lower_percentile",
            "winsor_upper_percentile",
            "winsor_lower_bound",
            "winsor_upper_bound",
            "winsor_n",
            "winsor_n_lower",
            "winsor_n_upper",
        )
    )
    return [
        {
            **centered_row_from_raw_sums(row),
            **winsor,
            "moments_format": 8,
            "decision_plan": wire,
        }
        for row in _moment_rows(metric_type)
    ]


def _moment_spec(metric_type: str) -> MetricSpec:
    if metric_type == "ratio":
        return MetricSpec(name="m", type="ratio", numerator="num", denominator="den")
    if metric_type == "retention":
        return MetricSpec(name="m", type="retention", threshold_days=(1, 3))
    if metric_type == "quantile":
        return MetricSpec(name="m", type="quantile", quantile=0.5)
    return MetricSpec(name="m", type=cast("Any", metric_type))


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_from_moments(metric_type, native):
    from increment import Analysis

    cell = _cell("from_moments", metric_type)
    if cell.status == "na":
        # No MetricSpec type exists for report-layer metrics; the terse
        # form names the accepted set.
        with pytest.raises(InvalidRequestError) as unknown_type:
            Analysis.from_moments(
                _current_moment_rows("mean"), metrics={"m": metric_type}, control="control"
            )
        assert unknown_type.value.code == "frame.metric_unknown_type"
        assert unknown_type.value.context["kind"] == metric_type
        with pytest.raises(InvalidRequestError) as spec_error:
            MetricSpec(name="m", type=metric_type)  # type: ignore[arg-type]
        assert spec_error.value.code == "model.field.literal"
        return

    def probe():
        analysis = Analysis.from_moments(
            _current_moment_rows(metric_type),
            metrics=[_moment_spec(metric_type)],
            control="control",
        )
        return analysis.run()

    if cell.status == "refused":
        _expect_refusal(cell, probe)
        return
    _assert_sane_lift(probe(), "m")
    if metric_type == "retention":
        # The terse form still refuses, pointing at the explicit MetricSpec.
        with pytest.raises(InvalidRequestError) as terse:
            Analysis.from_moments(
                _current_moment_rows("retention"), metrics={"m": "retention"}, control="control"
            )
        assert terse.value.code == "frame.metric_type_retention"
        # And a REAL exported cube round-trips to the native answer.
        native_d1 = next(e for e in native.run if e.metric == "d1")
        rehydrated_d1 = next(e for e in native.rehydrated_run if e.metric == "d1")
        assert rehydrated_d1.require_lift().value == pytest.approx(
            native_d1.require_lift().value, rel=1e-9
        )


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_export(metric_type, native, tmp_path):
    cell = _cell("export", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    if cell.status == "refused":
        analysis = _quantile_experiment_analysis(tmp_path)
        _expect_refusal(cell, lambda: analysis.export(tmp_path / "unreachable.parquet"))
        return
    name = _REPORT_METRIC[metric_type]
    assert name in native.export_metrics
    exported = next(e for e in native.run if e.metric == name)
    rehydrated = next(e for e in native.rehydrated_run if e.metric == name)
    assert rehydrated.require_lift().value == pytest.approx(exported.require_lift().value, rel=1e-9)


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_sitewide(metric_type, native, tmp_path):
    cell = _cell("sitewide", metric_type)
    if cell.status == "na":
        _assert_report_layer_unreachable(metric_type)
        return
    name = _REPORT_METRIC[metric_type]

    def probe():
        if metric_type == "quantile":
            return _quantile_experiment_analysis(tmp_path).sitewide("bad_metric")
        return native.analysis.sitewide(name)

    if cell.status == "refused":
        _expect_refusal(cell, probe)
        return
    impact = probe()
    assert math.isfinite(impact.absolute_impact)
    assert impact.absolute_impact_lb <= impact.absolute_impact <= impact.absolute_impact_ub


@pytest.mark.parametrize("metric_type", METRIC_TYPES)
def test_rollout(metric_type, native):
    from increment import segment_rollout_recommendation

    cell = _cell("rollout", metric_type)
    if cell.status == "na":
        if metric_type == "quantile":
            # The proof of unreachability: the only producer of the input
            # refuses this type outright.
            breakout_cell = _cell("breakout", "quantile")
            spec = _frame_spec("quantile")
            _expect_refusal(
                breakout_cell,
                lambda: readouts.run(_summary_source(spec), by=["seg"]),
            )
        else:
            _assert_report_layer_unreachable(metric_type)
        return
    name = _REPORT_METRIC[metric_type]
    recommendations, segments = segment_rollout_recommendation(native.breakout)
    rows = [r for r in recommendations if r.metric == name]
    assert rows, f"no rollout recommendation for {name}"
    for r in rows:
        assert r.k == 2 and r.n_excluded_design == 0 and r.n_excluded_outcome == 0
        assert r.recommendation in {"rollout", "no_net_benefit", "refuse"}
        if r.recommendation == "refuse":
            assert r.policy_value is None
        else:
            assert r.policy_value is not None and math.isfinite(r.policy_value)
    assert {s.dimension_value for s in segments if s.metric == name} == {"US", "CA"}


# Capability x capability crosses


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda p: f"{p[0]}-{p[1]}")
def test_capability_pairs(pair, native):
    from increment.estimation.inference import Normal

    cell = PAIRS[pair]
    mean = _frame_spec("mean")
    bernoulli = _frame_spec("conversion")
    from increment.frame import synthesise_metric
    from tests.sequential_cases import declared_plan

    def av_plan(design, spec=mean):
        from increment.sequential_source import frame_observation_mapping

        return declared_plan(
            [synthesise_metric(spec)],
            source_id="frame",
            design=design,
            transformations=[spec],
            source_mapping=frame_observation_mapping(
                unit="user_id",
                group="variant",
                exposure_date="exposure_date",
                uptake="uptake" if design.mechanism == "encouragement" else None,
            ),
        )

    def clustered_src(**kw: Any):
        return _summary_source(mean, table=_clustered_table(), cluster="store", **kw)

    probes: dict[tuple[str, str], Callable[[], Any]] = {
        ("cluster", "cuped"): lambda: _summary_source(
            _frame_spec("mean", covariate="pre"), table=_clustered_table(), cluster="store"
        ),
        ("cluster", "sequential"): lambda: readouts.run(
            _summary_source(
                bernoulli,
                table=_clustered_table(),
                cluster="store",
                design=_RANDOMIZED,
                plan=av_plan(_RANDOMIZED, bernoulli),
            )
        ),
        ("cluster", "prior"): lambda: readouts.run(
            clustered_src(design=_RANDOMIZED), prior=Normal(mu=0.0, sigma=0.1)
        ),
        ("cluster", "breakout"): lambda: readouts.run(
            clustered_src(design=_RANDOMIZED), by=["seg"]
        ),
        ("cluster", "daily_asof"): lambda: from_unit_panel(
            _panel_table(),
            unit="user_id",
            group="variant",
            control="control",
            date="ds",
            metrics=[mean],
            cluster="seg",
        ),
        ("cluster", "export"): lambda: native.clustered_analysis.export("unreachable.parquet"),
        ("cluster", "sitewide"): lambda: native.clustered_analysis.sitewide("revenue"),
        ("cluster", "encouragement"): lambda: readouts.run(
            clustered_src(uptake="uptake", design=_ENCOURAGEMENT)
        ),
        ("cluster", "observational"): lambda: readouts.run(clustered_src(design=_OBSERVATIONAL)),
        ("sequential", "observational"): lambda: readouts.run(
            _summary_source(
                bernoulli,
                table=_unit_table(80),
                design=_OBSERVATIONAL,
                plan=av_plan(_OBSERVATIONAL, bernoulli),
            )
        ),
        ("sequential", "encouragement"): lambda: readouts.run(
            _summary_source(
                bernoulli,
                uptake="uptake",
                design=_ENCOURAGEMENT,
                plan=av_plan(_ENCOURAGEMENT, bernoulli),
            ),
            estimands=("itt",),
        ),
        ("breakout", "observational"): lambda: readouts.breakout(
            _summary_source(mean, table=_unit_table(80), design=_OBSERVATIONAL),
            "seg",
        ),
        ("cuped", "encouragement"): lambda: readouts.run(
            _summary_source(
                _frame_spec("mean", covariate="pre"), uptake="uptake", design=_ENCOURAGEMENT
            ),
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        ),
        ("sequential", "cuped"): lambda: readouts.run(
            _summary_source(
                _frame_spec(
                    "mean",
                    covariate="pre",
                    decision_method=Method(name="cuped", variance_reduction="cuped"),
                ),
                design=Randomized(
                    control_group="control", allocation={"control": 0.5, "treatment": 0.5}
                ),
                plan=AnalysisPlan(
                    primary="revenue", inference=InferenceSpec(kind="asymptotic_mean")
                ),
            )
        ),
    }
    probe = probes[pair]
    if cell.status == "refused":
        _expect_refusal(cell, probe)
        return
    if pair == ("cluster", "sitewide"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = probe()
        assert warning_codes(caught) == ["estimation.sitewide.cluster_baseline_assumption"]
        assert math.isfinite(result.absolute_impact)
        assert result.absolute_impact_lb <= result.absolute_impact <= result.absolute_impact_ub
        assert result.n_clusters == 48
        return
    result = probe()
    if pair == ("cluster", "observational"):
        _assert_sane_lift(result, "revenue")
        assert all(e.n_clusters == 40 for e in result)
    elif pair == ("cluster", "encouragement"):
        _assert_sane_lift(result, "revenue")
        assert {"itt", "compliance", "late"} <= {e.estimand for e in result}
        late_rows = [e for e in result if e.estimand == "late"]
        assert late_rows and all(e.n_clusters == 40 for e in late_rows)
    elif pair == ("sequential", "encouragement"):
        assert {(row.metric, row.estimand) for row in result} == {("converted", "itt")}
        for row in result:
            checkpoint = row.require_sequential_result().checkpoint
            assert checkpoint.control.n == checkpoint.treatment.n == 8
            # Five of eight units convert in each arm.
            assert row.require_lift().value == 0.0
    elif pair == ("sequential", "cuped"):
        _assert_sane_lift(result, "revenue")
        for row in result:
            checkpoint = row.require_asymptotic_sequential_result().checkpoint
            assert checkpoint.model.law == "adjusted_mean" and row.method == "cuped"
            assert len(checkpoint.control.mean) == 2
    else:
        assert pair == ("cuped", "encouragement")
        _assert_sane_lift(result, "revenue")
        assert {"itt", "compliance", "late"} <= {e.estimand for e in result}
