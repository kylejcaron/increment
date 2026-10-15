from __future__ import annotations

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
    covariate_columns: tuple[tuple[str, str], ...] = ()
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()


def _frame_digest_and_arms(frame):
    from increment.estimation.readout_types import StreamingDigest

    columns = tuple(frame.columns)
    digest = StreamingDigest()
    digest.update_positional_rows(
        frame.iter_rows(named=False), columns=columns, normalize_frame_floats=True
    )
    arms: set[str] = set()
    if "group_id" in columns:
        observed = frame["group_id"].to_list()
        arms = {str(group_id) for group_id in observed if group_id is not None}
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
    _snapshot_active: bool = False,
) -> MetricRows:
    """Load one metric's rows once.

    Callers needing the quantile-specific alternative/cluster/breakout guard
    (`_refuse_unsupported_quantile`, which needs `test`/`cluster` -- outside
    this function's contract) call it before `_load_metric_rows`.
    """
    has_linear = any(
        getattr(method, "variance_reduction", None) == "cuped_multicovariate" for method in methods
    )
    if has_linear and not _snapshot_active:
        snapshot = getattr(src, "readout_snapshot", None)
        if snapshot is not None:
            population = getattr(src, "population", "assigned")
            with snapshot(metrics=(metric,), population=population) as pinned:
                return _load_metric_rows(
                    pinned,
                    metric,
                    by=by,
                    control_group=control_group,
                    methods=methods,
                    covariates=covariates,
                    _snapshot_active=True,
                )
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
    scalar_methods = [
        method for method in methods if getattr(method, "variance_reduction", None) == "cuped"
    ]
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()
    if scalar_methods and any(method.covariates for method in scalar_methods):
        method_moments = getattr(src, "moments_for_method", None)
        if method_moments is not None:
            rows_by_method = []
            for method in scalar_methods:
                family_rows = method_moments(metric, method, by=by)
                if family_rows is not None:
                    rows_by_method.append((method.name, tuple(family_rows)))
            method_rows = tuple(rows_by_method)
    method_components = tuple(
        (
            f"scalar_method_rows:{method_name}",
            digest,
            count,
        )
        for method_name, rows_for_method in method_rows
        for _arms, digest, count in (_rows_raw_digest_and_arms(rows_for_method),)
    )
    rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=by))
    observed, moments_sha256, moments_count = _rows_digest_and_arms(rows)
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
            evidence_components=(
                ("moments_rows", moments_sha256, moments_count),
                *method_components,
            ),
            method_rows=method_rows,
        )

    import narwhals as nw

    from increment._source_types import unit_frame_covariate_columns

    covariate_names = tuple(item.name for item in covariates)
    acquired_columns = unit_frame_covariate_columns(covariate_names)
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
            *method_components,
        ),
        method_rows=method_rows,
        covariate_columns=tuple(acquired_columns.items()),
    )
