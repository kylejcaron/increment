from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from increment.readouts._common import _refuse_unsupported_by
from increment.readouts._source_digest import input_evidence
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._source_types import RawOutcomeSource
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True)
class MetricRows:
    """One metric's aggregate rows and/or per-unit estimator evidence."""

    rows: tuple[Mapping[str, Any], ...]
    unit_frame: Any | None
    observed_arms: frozenset[str]
    n_treatment_arms: int
    evidence_kind: str
    evidence_sha256: str
    evidence_count: int
    evidence_components: tuple[tuple[str, str, int], ...] = ()


def _frame_digest_and_arms(frame):
    from increment.estimation.readout_types import StreamingDigest

    digest = StreamingDigest()

    for row in frame.iter_rows(named=True):
        for name, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                row[name] = None
        digest.update(row)
    arms: set[str] = set()
    if "group_id" in frame.columns:
        arms = {
            str(group_id)
            for group_id in frame.get_column("group_id").to_list()
            if group_id is not None
        }
    return frozenset(arms), digest.hexdigest(), digest.count


def _rows_digest_and_arms(rows):
    from increment.estimation.readout_types import StreamingDigest

    digest = StreamingDigest()
    arms = set()
    for row in rows:
        digest.update(input_evidence(row))
        group_id = row.get("group_id")
        if group_id is not None:
            arms.add(str(group_id))
    return frozenset(arms), digest.hexdigest(), digest.count


def _rows_raw_digest_and_arms(rows):
    from increment.estimation.readout_types import StreamingDigest

    digest = StreamingDigest()
    digest.update_rows(rows)
    arms = frozenset(str(row["group_id"]) for row in rows if row.get("group_id") is not None)
    return arms, digest.hexdigest(), digest.count


def _rows_digest(rows):
    from increment.estimation.readout_types import StreamingDigest

    digest = StreamingDigest()
    for row in rows:
        digest.update(input_evidence(row))
    return digest.hexdigest(), digest.count


def _load_metric_rows(
    src: MomentSource,
    metric: Metric,
    *,
    by: Sequence[str],
    control_group: str,
    methods: Sequence[Any] = (),
    covariates: Sequence[Any] = (),
) -> MetricRows:
    """Load one metric's rows once.

    Callers needing the quantile-specific alternative/cluster/breakout guard
    (`_refuse_unsupported_quantile`, which needs `test`/`cluster` -- outside
    this function's contract) call it before `_load_metric_rows`.
    """
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        import narwhals as nw

        from increment.estimation.winsor import _raw_state_from_source

        # Check the arm inventory before constructing a two-arm inference state.
        # Reuse this capture so the aggregate arm gate does not trigger another read.
        raw_source = cast("RawOutcomeSource", src)
        native = raw_source.unit_frame(metric, outcome_stage="raw")
        frame = nw.from_native(native, eager_only=True)

        observed, evidence_sha256, evidence_count = _frame_digest_and_arms(frame)
        if not observed - {str(control_group)}:
            return MetricRows(
                rows=(),
                unit_frame=None,
                observed_arms=observed,
                n_treatment_arms=0,
                evidence_kind="unit_evidence_rows",
                evidence_sha256=evidence_sha256,
                evidence_count=evidence_count,
            )
        raw = _raw_state_from_source(src, metric, native=native)
        from increment.estimation.winsor import build_winsor_references

        treatments = tuple(arm.group_id for arm in raw.arms if arm.group_id != control_group)
        references = {
            (metric.name, treatment): reference
            for treatment, reference in build_winsor_references(
                raw, control_group, treatments
            ).items()
        }
        observed = frozenset(a.group_id for a in raw.arms)
        # Keep the immutable raw state with the loaded evidence so repeated
        # family passes reinvert without fetching or changing the cutoff pool.
        return MetricRows(
            rows=tuple(
                {"group_id": a.group_id, "_winsor_raw_state": raw, "_winsor_references": references}
                for a in raw.arms
            ),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
            evidence_kind="unit_evidence_rows",
            evidence_sha256=evidence_sha256,
            evidence_count=evidence_count,
        )
    if getattr(metric, "type", None) == "quantile":
        import narwhals as nw

        native = src.unit_frame(metric)
        frame = nw.from_native(native, eager_only=True)
        observed, evidence_sha256, evidence_count = _frame_digest_and_arms(frame)
        return MetricRows(
            rows=(),
            unit_frame=native,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
            evidence_kind="unit_evidence_rows",
            evidence_sha256=evidence_sha256,
            evidence_count=evidence_count,
        )
    _refuse_unsupported_by(metric, by)
    rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=by))
    observed, moments_sha256, moments_count = _rows_digest_and_arms(rows)
    has_linear = any(
        getattr(method, "variance_reduction", None) == "cuped_linear" for method in methods
    )
    if has_linear:
        observed, moments_sha256, moments_count = _rows_raw_digest_and_arms(rows)
    if not has_linear:
        return MetricRows(
            rows=tuple(rows),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
            evidence_kind="moments_rows",
            evidence_sha256=moments_sha256,
            evidence_count=moments_count,
            evidence_components=(("moments_rows", moments_sha256, moments_count),),
        )

    import narwhals as nw

    covariate_names = tuple(item.name for item in covariates)
    native = src.unit_frame(metric, covariates=covariate_names)
    frame = nw.from_native(native, eager_only=True)
    unit_arms, unit_sha256, unit_count = _frame_digest_and_arms(frame)
    return MetricRows(
        rows=tuple(rows),
        unit_frame=native,
        observed_arms=observed | unit_arms,
        n_treatment_arms=len((observed | unit_arms) - {control_group}),
        evidence_kind="mixed_rows",
        evidence_sha256="",
        evidence_count=moments_count + unit_count,
        evidence_components=(
            ("moments_rows", moments_sha256, moments_count),
            ("unit_evidence_rows", unit_sha256, unit_count),
        ),
    )
