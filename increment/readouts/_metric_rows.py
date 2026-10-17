from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from increment._analysis_config import overlay_configs
from increment._frame_validation import SCALAR_COVARIATE_REFUSALS
from increment._source_operations import ReadoutSnapshotOperation
from increment._source_types import ScalarMethodMomentSource
from increment.errors import CodedError, refuse
from increment.readouts._common import _raise, _refuse_unsupported_by
from increment.readouts._source_digest import input_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._source_types import RawOutcomeSource
    from increment.errors import RefusalSpec
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


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
    decision_evidence_components: tuple[tuple[str, str, int], ...]
    evidence_components: tuple[tuple[str, str, int], ...] = ()
    covariate_columns: tuple[tuple[str, str], ...] = ()
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()
    method_refusals: tuple[tuple[str, RefusalSpec, Mapping[str, object]], ...] = ()

    def rows_for_method(self, method: Any) -> tuple[Mapping[str, Any], ...]:
        """Return rows produced for an effective scalar method or fail loudly."""
        for name, rows in self.method_rows:
            if name == method.name:
                return rows
        for name, spec, context in self.method_refusals:
            if name == method.name:
                refuse(spec, **context)
        raise AssertionError(f"Scalar evidence for method {method.name!r} was not acquired.")


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


def _scalar_method_rows(src, metric, methods, decision, *, by):
    if not isinstance(src, ScalarMethodMomentSource):
        _raise(
            "readout.scalar_method_evidence",
            route=type(src).__name__,
            method=methods[0].name,
        )
    families = []
    refusals = []
    for method in methods:
        try:
            family_rows = tuple(src.moments_for_method(metric, method, by=by))
        except CodedError as exc:
            if exc.code not in SCALAR_COVARIATE_REFUSALS:
                raise
            refusals.append((method.name, SCALAR_COVARIATE_REFUSALS[exc.code], exc.context))
        else:
            families.append((method.name, family_rows))
    selected = next((rows for name, rows in families if name == decision.name), None)
    if selected is not None:
        rows = list(selected)
    elif decision.variance_reduction != "cuped":
        rows = list(src.moments_for_method(metric, decision, by=by))
    else:
        from increment.estimation.engine import Method

        assert any(name == decision.name for name, _spec, _context in refusals)
        rows = list(src.moments_for_method(metric, Method.unadjusted(), by=by))
    return rows, tuple(families), tuple(refusals)


def _decision_unit_component(frame, method, acquired_columns):
    import narwhals as nw

    from increment._source_types import unit_frame_covariate_columns

    names = tuple(covariate.name for covariate in method.covariates)
    canonical = unit_frame_covariate_columns(names)
    acquired = set(acquired_columns.values())
    projected = frame.select(
        *(nw.col(column) for column in frame.columns if column not in acquired),
        *(nw.col(acquired_columns[name]).alias(canonical[name]) for name in names),
    )
    return ("unit_evidence_rows", *_frame_digest_and_arms(projected)[1:])


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
    has_scalar = any(method.variance_reduction == "cuped" for method in methods)
    if (has_linear or has_scalar) and not _snapshot_active:
        if isinstance(src, ReadoutSnapshotOperation):
            population = getattr(src, "population", "assigned")
            configs = overlay_configs(
                (metric,), src.context.configs, methods=list(methods), prior=None
            )
            with src.readout_snapshot(
                metrics=(metric,), population=population, configs=configs
            ) as pinned:
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
                decision_evidence_components=(
                    ("unit_evidence_rows", evidence_sha256, evidence_count),
                ),
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
            decision_evidence_components=(("unit_evidence_rows", evidence_sha256, evidence_count),),
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
            decision_evidence_components=(("unit_evidence_rows", evidence_sha256, evidence_count),),
        )
    _refuse_unsupported_by(metric, by)
    scalar_methods = [
        method for method in methods if getattr(method, "variance_reduction", None) == "cuped"
    ]
    method_rows: tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...] = ()
    method_refusals = ()
    if scalar_methods:
        rows, method_rows, method_refusals = _scalar_method_rows(
            src, metric, scalar_methods, methods[0], by=by
        )
    elif methods and isinstance(src, ScalarMethodMomentSource):
        rows = list(src.moments_for_method(metric, methods[0], by=by))
    else:
        rows = cast("list[Mapping[str, Any]]", src.moments(metric, grain="total", by=by))
    observed, moments_sha256, moments_count = _rows_digest_and_arms(rows)
    ordinary_component = ("moments_rows", moments_sha256, moments_count)
    if has_linear:
        observed, moments_sha256, moments_count = _rows_raw_digest_and_arms(rows)
    method_components = tuple(
        (f"scalar_method:{name}", *_rows_raw_digest_and_arms(family_rows)[1:])
        for name, family_rows in method_rows
        if any(method.name == name and method.covariates for method in scalar_methods)
    )
    method_components += tuple(
        (
            f"scalar_method:{name}",
            *_rows_raw_digest_and_arms(
                ({"failure_code": spec.code, "failure_context": dict(context)},)
            )[1:],
        )
        for name, spec, context in method_refusals
    )
    decision_is_linear = bool(methods and methods[0].variance_reduction == "cuped_multicovariate")
    decision_components = (
        ("moments_rows", moments_sha256, moments_count)
        if decision_is_linear
        else ordinary_component,
        *(
            item
            for item in method_components
            if methods and item[0] == f"scalar_method:{methods[0].name}"
        ),
    )
    if not has_linear:
        return MetricRows(
            rows=tuple(rows),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=len(observed - {control_group}),
            evidence_kind="moments_rows",
            evidence_sha256=moments_sha256,
            evidence_count=moments_count,
            decision_evidence_components=decision_components,
            evidence_components=(
                ("moments_rows", moments_sha256, moments_count),
                *method_components,
            ),
            method_rows=method_rows,
            method_refusals=tuple(method_refusals),
        )

    import narwhals as nw

    from increment._source_types import unit_frame_covariate_columns

    covariate_names = tuple(item.name for item in covariates)
    acquired_columns = unit_frame_covariate_columns(covariate_names)
    native = src.unit_frame(metric, covariates=covariate_names)
    frame = nw.from_native(native, eager_only=True)
    unit_arms, unit_sha256, unit_count = _frame_digest_and_arms(frame)
    if decision_is_linear:
        decision_components += (_decision_unit_component(frame, methods[0], acquired_columns),)
    return MetricRows(
        rows=tuple(rows),
        unit_frame=native,
        observed_arms=observed | unit_arms,
        n_treatment_arms=len((observed | unit_arms) - {control_group}),
        evidence_kind="mixed_rows",
        evidence_sha256="",
        evidence_count=moments_count + unit_count,
        decision_evidence_components=decision_components,
        evidence_components=(
            ("moments_rows", moments_sha256, moments_count),
            ("unit_evidence_rows", unit_sha256, unit_count),
            *method_components,
        ),
        method_rows=method_rows,
        method_refusals=tuple(method_refusals),
        covariate_columns=tuple(acquired_columns.items()),
    )
