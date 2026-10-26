from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import overlay_configs, validate_scalar_route
from increment._evidence_loaders import view_population
from increment._evidence_types import (
    BreakoutCell,
    BreakoutIdentity,
    BreakoutMethodMomentsRequest,
    EvidenceScope,
    MethodFitRequest,
    MetricEvidence,
    MetricEvidenceRequest,
    MomentsRequest,
    UnitRowsRequest,
)
from increment._frame_validation import SCALAR_COVARIATE_REFUSALS
from increment._source_types import declared_evidence, is_custom_source
from increment.errors import CodedError
from increment.readouts._common import _refuse_unsupported_by
from increment.readouts._evidence import (
    evidence_scope,
    fits_cuped_in_warehouse,
    load_breakout_method_moments,
    load_method_fit,
    load_moments,
    load_unit_rows,
    metric_authenticator,
    retains_cuped_relation,
)
from increment.readouts._source_digest import input_evidence
from increment.semantics.design import Encouragement

if TYPE_CHECKING:
    from collections.abc import Mapping

    from increment.semantics.models import Metric
    from increment.sources import EvidenceSource


def winsor_identity(
    src: EvidenceSource,
    metric: Metric,
    *,
    population: Literal["assigned", "triggered"] | None = None,
) -> tuple[Literal["assigned", "triggered"], str]:
    """Return declared population and metric missingness for raw outcomes."""
    selected_population: Literal["assigned", "triggered"] = population or view_population(src)
    missingness = src.context.metric_missingness.get(
        metric.name,
        f"measure-unit-inclusion-v1:{getattr(metric, 'aggregation', 'undefined')}:observable-windows",
    )
    return selected_population, missingness


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

    # Moments may legitimately overflow (a third moment is then a diagnostic gap,
    # and estimators that read an overflowed second moment refuse it), so
    # non-finite values are digested with the frame-float encoding.
    digest = StreamingDigest()
    digest.update_rows(rows, normalize_frame_floats=True)
    arms = frozenset(str(row["group_id"]) for row in rows if row.get("group_id") is not None)
    return arms, digest.hexdigest(), digest.count


def _rows_digest(rows):
    from increment.estimation.readout_types import StreamingDigest

    digest = StreamingDigest()
    for row in rows:
        digest.update(input_evidence(row))
    return digest.hexdigest(), digest.count


def _scalar_method_rows(src, metric, methods, decision, *, by, population):
    scalar = declared_evidence(src).scalar
    if scalar is None or not scalar.method_evidence:
        validate_scalar_route(
            overlay_configs((metric,), src.context.configs, methods=list(methods), prior=None),
            route=type(src).__name__,
            representation="implicit",
            design=src.context.design,
        )
        rows = load_moments(
            src, MomentsRequest(metric=metric, grain="total", by=by, population=population)
        )
        return list(rows), tuple((method.name, rows) for method in methods), ()
    families = []
    refusals = []
    for method in methods:
        try:
            family_rows = load_moments(
                src, MomentsRequest(metric=metric, method=method, by=by, population=population)
            )
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
        rows = list(
            load_moments(
                src, MomentsRequest(metric=metric, method=decision, by=by, population=population)
            )
        )
    else:
        from increment.declarations import Method

        assert any(name == decision.name for name, _spec, _context in refusals)
        rows = list(
            load_moments(
                src,
                MomentsRequest(
                    metric=metric, method=Method.unadjusted(), by=by, population=population
                ),
            )
        )
    return rows, tuple(families), tuple(refusals)


def _decision_unit_component(frame, method, acquired_columns):
    import narwhals as nw

    from increment._evidence_types import unit_frame_covariate_columns

    names = tuple(covariate.name for covariate in method.covariates)
    canonical = unit_frame_covariate_columns(names)
    acquired = set(acquired_columns.values())
    projected = frame.select(
        *(nw.col(column) for column in frame.columns if column not in acquired),
        *(nw.col(acquired_columns[name]).alias(canonical[name]) for name in names),
    )
    return ("unit_evidence_rows", *_frame_digest_and_arms(projected)[1:])


def metric_evidence_scope(src: EvidenceSource, request: MetricEvidenceRequest) -> EvidenceScope:
    """Build the provider scope needed to acquire one metric's evidence."""
    snapshot = any(
        getattr(method, "variance_reduction", None) in {"cuped", "cuped_multicovariate"}
        for method in request.methods
    )
    configs = (
        overlay_configs(
            (request.metric,), src.context.configs, methods=list(request.methods), prior=None
        )
        if snapshot
        else None
    )
    design = src.context.design
    uptake_facts = (design.uptake.fact,) if isinstance(design, Encouragement) else ()
    return EvidenceScope(
        metrics=(request.metric,),
        population=request.population,
        configs=configs,
        uptake_facts=uptake_facts,
        breakouts=tuple(BreakoutIdentity(item, None) for item in request.by),
        snapshot=snapshot,
    )


def load_metric_evidence(
    src: EvidenceSource, request: MetricEvidenceRequest, *, scope_active: bool = False
) -> MetricEvidence:
    """Load one metric's rows once for a typed evidence request."""
    return _load_metric_evidence(src, request, _snapshot_active=scope_active)


def _load_percentile_winsor_evidence(
    src: EvidenceSource,
    metric: Metric,
    control_group: str,
    population: Literal["assigned", "triggered"],
) -> MetricEvidence:
    import narwhals as nw

    from increment.estimation.winsor import raw_state_from_frame, winsor_preflight

    context = src.context
    winsor_preflight(metric, context)
    population, missingness = winsor_identity(src, metric, population=population)
    native = load_unit_rows(
        src,
        UnitRowsRequest(metric=metric, outcome_stage="raw", population=population),
    )
    frame = nw.from_native(native, eager_only=True)
    observed, evidence_sha256, evidence_count = _frame_digest_and_arms(frame)
    if not observed - {str(control_group)}:
        return MetricEvidence(
            cluster=context.cluster,
            control_group=control_group,
            rows=(),
            unit_frame=None,
            observed_arms=observed,
            n_treatment_arms=0,
            evidence_kind="unit_evidence_rows",
            evidence_sha256=evidence_sha256,
            evidence_count=evidence_count,
            decision_evidence_components=(("unit_evidence_rows", evidence_sha256, evidence_count),),
        )
    raw = raw_state_from_frame(
        native,
        metric,
        context=context,
        population=population,
        missingness=missingness,
    )
    from increment.estimation.winsor import build_winsor_references

    treatments = tuple(arm.group_id for arm in raw.arms if arm.group_id != control_group)
    references = {
        (metric.name, treatment): reference
        for treatment, reference in build_winsor_references(raw, control_group, treatments).items()
    }
    observed = frozenset(arm.group_id for arm in raw.arms)
    # Keep the immutable raw state with the loaded evidence so repeated
    # family passes reinvert without fetching or changing the cutoff pool.
    return MetricEvidence(
        cluster=src.context.cluster,
        control_group=control_group,
        rows=tuple(
            {"group_id": arm.group_id, "_winsor_raw_state": raw, "_winsor_references": references}
            for arm in raw.arms
        ),
        unit_frame=None,
        observed_arms=observed,
        n_treatment_arms=len(observed - {control_group}),
        evidence_kind="unit_evidence_rows",
        evidence_sha256=evidence_sha256,
        evidence_count=evidence_count,
        decision_evidence_components=(("unit_evidence_rows", evidence_sha256, evidence_count),),
    )


def _load_metric_evidence(  # noqa: PLR0915
    src: EvidenceSource,
    request: MetricEvidenceRequest,
    *,
    _snapshot_active: bool = False,
) -> MetricEvidence:
    metric = request.metric
    by = request.by
    from increment._evidence_refusals import require_evidence

    require_evidence(src, request=request)
    control_group = request.control_group
    methods = request.methods
    covariates = request.covariates
    has_linear = any(
        getattr(method, "variance_reduction", None) == "cuped_multicovariate" for method in methods
    )
    has_scalar = any(method.variance_reduction == "cuped" for method in methods)
    if (has_linear or has_scalar) and not _snapshot_active:
        population = request.population
        configs = overlay_configs((metric,), src.context.configs, methods=list(methods), prior=None)
        uptake_facts = (
            (src.context.design.uptake.fact,)
            if isinstance(src.context.design, Encouragement)
            else ()
        )
        with evidence_scope(
            src,
            EvidenceScope(
                metrics=(metric,),
                population=population,
                configs=configs,
                uptake_facts=uptake_facts,
                snapshot=True,
            ),
        ) as pinned:
            return _load_metric_evidence(pinned, request, _snapshot_active=True)
    if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
        return _load_percentile_winsor_evidence(src, metric, control_group, request.population)
    if getattr(metric, "type", None) == "quantile":
        import narwhals as nw

        require_evidence(src, request=request)
        design = src.context.design
        if getattr(design, "mechanism", None) == "observational":
            from increment.estimation._readout_refusals import refuse_observational_quantile

            refuse_observational_quantile(metric, authenticate=metric_authenticator(src))
        native = load_unit_rows(src, UnitRowsRequest(metric=metric, population=request.population))
        frame = nw.from_native(native, eager_only=True)
        observed, evidence_sha256, evidence_count = _frame_digest_and_arms(frame)
        return MetricEvidence(
            cluster=src.context.cluster,
            control_group=control_group,
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
    scalar = declared_evidence(src).scalar
    if scalar_methods:
        rows, method_rows, method_refusals = _scalar_method_rows(
            src, metric, scalar_methods, methods[0], by=by, population=request.population
        )
    elif methods and scalar is not None and scalar.method_evidence:
        rows = list(
            load_moments(
                src,
                MomentsRequest(
                    metric=metric, method=methods[0], by=by, population=request.population
                ),
            )
        )
    else:
        rows = list(
            load_moments(
                src,
                MomentsRequest(metric=metric, grain="total", by=by, population=request.population),
            )
        )
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
        return MetricEvidence(
            cluster=src.context.cluster,
            control_group=control_group,
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

    arm_order = (str(control_group), *sorted(observed - {str(control_group)}))
    population = request.population
    method_fit_requests = tuple(
        MethodFitRequest(
            metric=metric,
            method=method,
            control=str(control_group),
            arm_order=arm_order,
            population=population,
        )
        for method in methods
        if getattr(method, "variance_reduction", None) == "cuped_multicovariate"
    )
    method_fit_answers = tuple(
        (method_fit_request, load_method_fit(src, method_fit_request))
        for method_fit_request in method_fit_requests
    )
    retained_answers = tuple(
        (method_fit_request.method.name, answer)
        for method_fit_request, answer in method_fit_answers
        if answer.failure is None and answer.origin == "retained_categorical"
    )
    retained = tuple((name, cast(Any, answer.cells)) for name, answer in retained_answers)
    cell_arms = frozenset(
        str(arm)
        for _, cells in retained
        for arm, counts in zip(cells.arms, cells.counts, strict=True)
        if sum(counts)
    )
    categorical_components = tuple(
        (f"categorical_cells:{name}", cast(str, answer.evidence_sha256), 1)
        for name, answer in retained_answers
    )
    retained_names = {name for name, _evidence in retained}
    mixed_names = {
        method.name for method in methods if method.variance_reduction == "cuped_multicovariate"
    }
    if mixed_names <= retained_names:
        if decision_is_linear:
            decision_components += tuple(
                item
                for item in categorical_components
                if item[0] == f"categorical_cells:{methods[0].name}"
            )
        return MetricEvidence(
            cluster=src.context.cluster,
            control_group=control_group,
            rows=tuple(rows),
            unit_frame=None,
            observed_arms=observed | cell_arms,
            n_treatment_arms=len((observed | cell_arms) - {control_group}),
            evidence_kind="mixed_rows",
            evidence_sha256="",
            evidence_count=moments_count + len(categorical_components),
            decision_evidence_components=decision_components,
            evidence_components=(
                ("moments_rows", moments_sha256, moments_count),
                *categorical_components,
                *method_components,
            ),
            method_rows=method_rows,
            method_refusals=tuple(method_refusals),
            method_fit_answers=method_fit_answers,
        )
    warehouse_fits = [
        (
            method_fit_request.method.name,
            answer.failure if answer.failure is not None else answer.envelope,
        )
        for method_fit_request, answer in method_fit_answers
        if answer.origin in {"warehouse", "retained_fitted"} or answer.failure is not None
    ]
    warehouse_fit_identities = [
        answer.fit_identity
        for _method_fit_request, answer in method_fit_answers
        if answer.origin in {"warehouse", "retained_fitted"} or answer.failure is not None
    ]
    needs_local_units = (
        is_custom_source(src)
        and declared_evidence(src).unit_rows
        and any(answer.origin == "unavailable" for _fit_request, answer in method_fit_answers)
    )
    fit_components = ()
    if warehouse_fits:

        def fit_evidence_record(identity, evidence):
            if isinstance(evidence, CodedError):
                return {
                    "fit_identity": identity,
                    "failure": evidence.code,
                    "fit": None,
                }
            fitted = evidence.fit
            return {
                "fit_identity": identity,
                "arm_order": evidence.arm_order,
                "withheld": tuple((arm, refusal.code) for arm, refusal in evidence.withheld),
                "fit": None
                if fitted is None
                else {
                    "arm_order": fitted.arm_order,
                    "control": fitted.control,
                    "n_by_arm": fitted.n_by_arm,
                    "adjusted_means": fitted.adjusted_means,
                    "theta": fitted.theta,
                    "anchor": fitted.anchor,
                    "layout": tuple(
                        getattr(fitted.layout, field)
                        for field in fitted.layout.__dataclass_fields__
                    ),
                    "covariance": fitted.covariance,
                    "dof": fitted.dof,
                    "n_min": fitted.n_min,
                    "p": fitted.p,
                    "max_leverage": fitted.max_leverage,
                    "mean_leverage": fitted.mean_leverage,
                    "leverage_trace": fitted.leverage_trace,
                    "is_ratio": fitted.is_ratio,
                },
            }

        fit_components = tuple(
            (
                f"warehouse_qr:{name}",
                _rows_raw_digest_and_arms((fit_evidence_record(identity, fit),))[1],
                1,
            )
            for (name, fit), identity in zip(warehouse_fits, warehouse_fit_identities, strict=True)
        )
        if not needs_local_units:
            evidence_components = (*categorical_components, *fit_components)
            decision_method_name = methods[0].name if methods else None
            return MetricEvidence(
                cluster=src.context.cluster,
                control_group=control_group,
                rows=tuple(rows),
                unit_frame=None,
                observed_arms=observed | cell_arms,
                n_treatment_arms=len((observed | cell_arms) - {control_group}),
                evidence_kind="mixed_rows",
                evidence_sha256="",
                evidence_count=moments_count + len(evidence_components),
                decision_evidence_components=(
                    decision_components
                    + tuple(
                        item
                        for item in evidence_components
                        if item[0]
                        in {
                            f"warehouse_qr:{decision_method_name}",
                            f"categorical_cells:{decision_method_name}",
                        }
                    )
                ),
                evidence_components=(
                    ("moments_rows", moments_sha256, moments_count),
                    *evidence_components,
                    *method_components,
                ),
                method_rows=method_rows,
                method_refusals=tuple(method_refusals),
                method_fit_answers=method_fit_answers,
            )

    import narwhals as nw

    from increment._evidence_types import unit_frame_covariate_columns

    covariate_names = tuple(item.name for item in covariates)
    acquired_columns = unit_frame_covariate_columns(covariate_names)
    require_evidence(src, unit_rows=True, covariates=covariate_names)
    native = load_unit_rows(
        src,
        UnitRowsRequest(
            metric=metric,
            covariates=covariate_names,
            population=request.population,
        ),
    )
    frame = nw.from_native(native, eager_only=True)
    unit_arms, unit_sha256, unit_count = _frame_digest_and_arms(frame)
    local_fit_components = (
        (*categorical_components, *fit_components) if is_custom_source(src) else ()
    )
    decision_fitted_locally = bool(
        methods
        and any(
            fit_request.method.name == methods[0].name and answer.origin == "unavailable"
            for fit_request, answer in method_fit_answers
        )
    )
    if decision_is_linear:
        if decision_fitted_locally:
            decision_components += (_decision_unit_component(frame, methods[0], acquired_columns),)
        decision_components += tuple(
            item
            for item in local_fit_components
            if item[0]
            in {
                f"categorical_cells:{methods[0].name}",
                f"warehouse_qr:{methods[0].name}",
            }
        )
    return MetricEvidence(
        cluster=src.context.cluster,
        control_group=control_group,
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
            *local_fit_components,
            *method_components,
        ),
        method_rows=method_rows,
        method_refusals=tuple(method_refusals),
        covariate_columns=tuple(acquired_columns.items()),
        method_fit_answers=method_fit_answers,
    )


def _breakout_fit_record(identity: tuple[object, ...], evidence: Any) -> dict[str, Any]:
    if isinstance(evidence, CodedError):
        return {"fit_identity": identity, "failure": evidence.code, "fit": None}
    fitted = getattr(evidence, "fit", evidence)
    withheld = tuple(
        (arm, refusal.code, dict(refusal.context))
        for arm, refusal in getattr(evidence, "withheld", ())
    )
    if fitted is None:
        return {"fit_identity": identity, "withheld": withheld, "fit": None}
    fit_values = {
        "arm_order": fitted.arm_order,
        "control": fitted.control,
        "n_by_arm": fitted.n_by_arm,
        "adjusted_means": fitted.adjusted_means,
        "theta": fitted.theta,
        "anchor": fitted.anchor,
        "layout": tuple(
            getattr(fitted.layout, field) for field in fitted.layout.__dataclass_fields__
        ),
        "covariance": fitted.covariance,
        "dof": fitted.dof,
        "n_min": fitted.n_min,
        "p": fitted.p,
        "max_leverage": fitted.max_leverage,
        "mean_leverage": fitted.mean_leverage,
        "leverage_trace": fitted.leverage_trace,
        "is_ratio": fitted.is_ratio,
    }
    projections = []
    for arm in fitted.arm_order:
        if arm == fitted.control:
            continue
        try:
            projection = (
                fitted.ratio(arm, control=fitted.control)
                if fitted.is_ratio
                else fitted.relative(arm, control=fitted.control)
            )
        except CodedError as exc:
            projections.append((arm, exc.code, dict(exc.context)))
        else:
            projections.append(
                (
                    arm,
                    projection.estimate,
                    projection.se,
                    projection.dof,
                    projection.lower,
                    projection.upper,
                )
            )
    return {
        "fit_identity": identity,
        "withheld": withheld,
        "fit": fit_values,
        "projections": tuple(projections),
    }


def acquire_breakout_method_moments(
    src, metric, method, *, by, source_name=None, population="assigned"
):
    """Acquire one method-specific breakout input through the shared evidence loader."""
    from increment._evidence_types import BreakoutIdentity, BreakoutMethodMomentsRequest
    from increment.readouts._evidence import load_breakout_method_moments

    return load_breakout_method_moments(
        src,
        BreakoutMethodMomentsRequest(
            metric=metric,
            method=method,
            breakout=BreakoutIdentity(by[0], source_name),
            population=population,
        ),
    )


def _load_breakout_metric_rows(
    src, metric, methods, *, dimension, control_group, source_name=None, population="assigned"
) -> MetricEvidence:
    """Acquire effective scalar inputs without promoting a segment failure."""
    scalar = declared_evidence(src).scalar
    if any(method.variance_reduction == "cuped" for method in methods) and not (
        scalar is not None and scalar.breakout_evidence
    ):
        validate_scalar_route(
            overlay_configs((metric,), src.context.configs, methods=list(methods), prior=None),
            route=f"{type(src).__name__}.breakout",
            representation="implicit",
            design=src.context.design,
        )
    from increment.declarations import Method

    base = (
        load_breakout_method_moments(
            src,
            BreakoutMethodMomentsRequest(
                metric=metric,
                method=Method.unadjusted(),
                breakout=BreakoutIdentity(dimension, source_name),
                population=population,
            ),
        ).rows
        if scalar is not None and scalar.breakout_evidence
        else load_moments(
            src,
            MomentsRequest(
                metric=metric,
                grain="total",
                by=(dimension,),
                population=population,
            ),
        )
    )
    breakout_fit_answers = []
    warehouse_components = []
    segment_values = (
        sorted({"__null__" if row.get(dimension) is None else str(row[dimension]) for row in base})
        if fits_cuped_in_warehouse(src) or retains_cuped_relation(src)
        else ()
    )
    for method in methods:
        if method.variance_reduction != "cuped_multicovariate":
            continue
        for segment_value in segment_values:
            segment_rows = tuple(
                row
                for row in base
                if ("__null__" if row.get(dimension) is None else str(row[dimension]))
                == segment_value
            )
            from increment.breakout.estimates import _partition_nonpositive_mean_rows

            admitted_rows = _partition_nonpositive_mean_rows(
                list(segment_rows),
                control_group=control_group,
                metric_types={metric.name: metric.type},
                binomial_gate_exempt_metrics=frozenset(),
            ).live
            arms = sorted(
                {str(row["group_id"]) for row in admitted_rows if row.get("group_id") is not None}
            )
            if control_group not in arms:
                continue
            arm_order = (control_group, *(arm for arm in arms if arm != control_group))
            is_null = segment_value == "__null__"
            request_value = None if is_null else segment_value
            method_fit_request = MethodFitRequest(
                metric=metric,
                method=method,
                control=control_group,
                arm_order=arm_order,
                population=cast(Literal["assigned", "triggered"], population),
                cell=BreakoutCell(
                    breakout=BreakoutIdentity(dimension, source_name),
                    value=request_value,
                    is_null=is_null,
                ),
            )
            answer = load_method_fit(src, method_fit_request)
            breakout_fit_answers.append((method_fit_request, answer))
            if answer.origin == "unavailable":
                # Kept so the readout can acquire unit rows for this cell's local fit.
                continue
            if answer.origin == "retained_categorical" and answer.failure is None:
                # Retained categorical cells carry no envelope; their evidence digest is the identity.
                warehouse_components.append(
                    (
                        f"categorical_cells_breakout:{method.name}:{dimension}:{segment_value}",
                        cast(str, answer.evidence_sha256),
                        1,
                    )
                )
                continue
            fit = answer.failure if answer.failure is not None else answer.envelope
            warehouse_components.append(
                (
                    f"warehouse_qr_breakout:{method.name}:{dimension}:{segment_value}",
                    _rows_raw_digest_and_arms((_breakout_fit_record(answer.fit_identity, fit),))[1],
                    1,
                )
            )
    families = []
    failures = []
    cell_failures = []
    for method in methods:
        if method.variance_reduction != "cuped":
            continue
        if scalar is None or not scalar.breakout_evidence:
            families.append((method.name, base))
            continue
        try:
            acquired = acquire_breakout_method_moments(
                src, metric, method, by=(dimension,), source_name=source_name, population=population
            )
        except CodedError as exc:
            if exc.code not in SCALAR_COVARIATE_REFUSALS:
                raise
            failures.append((method.name, SCALAR_COVARIATE_REFUSALS[exc.code], exc.context))
        else:
            families.append((method.name, acquired.rows))
            cell_failures.extend((method.name, *failure) for failure in acquired.refusals)
    observed, digest, count = _rows_digest_and_arms(base)
    ordinary_component = ("moments_rows", digest, count)
    method_components = []
    for method in methods:
        if method.variance_reduction != "cuped":
            continue
        records = (
            tuple(row for name, rows in families if name == method.name for row in rows)
            + tuple(
                {"failure_code": spec.code, "failure_context": dict(context)}
                for name, spec, context in failures
                if name == method.name
            )
            + tuple(
                {
                    "dimension": cell_dimension,
                    "dimension_value": value,
                    "failure_code": spec.code,
                    "failure_context": dict(context),
                }
                for name, cell_dimension, value, spec, context in cell_failures
                if name == method.name
            )
        )
        method_components.append(
            (f"scalar_method:{method.name}", *_rows_raw_digest_and_arms(records)[1:])
        )

    warehouse_components = tuple(warehouse_components)
    components = (ordinary_component, *method_components, *warehouse_components)
    return MetricEvidence(
        cluster=src.context.cluster,
        control_group=control_group,
        rows=base,
        unit_frame=None,
        observed_arms=observed,
        n_treatment_arms=len(observed - {control_group}),
        evidence_kind="moments_rows",
        evidence_sha256=digest,
        evidence_count=count + len(warehouse_components),
        decision_evidence_components=(
            ordinary_component,
            *(
                item
                for item in method_components
                if methods and item[0] == f"scalar_method:{methods[0].name}"
            ),
        ),
        evidence_components=components,
        method_rows=tuple(families),
        method_refusals=tuple(failures),
        method_cell_refusals=tuple(cell_failures),
        method_fit_answers=tuple(breakout_fit_answers),
    )
