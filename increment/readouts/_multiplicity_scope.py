"""Project declared multiplicity provenance onto scoped readout rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, cast

from increment._evidence_types import ArtifactProvenance
from increment._multiplicity import (
    RegisteredCell,
    guarantee_for_correction,
    registered_family_members,
)
from increment.decision import MultiplicityFamily, evidence_request_procedure
from increment.estimation.family import sequential_family_guarantee
from increment.estimation.multiplicity import (
    design_excluded,
    multiplicity_status,
    row_multiplicity_status,
)
from increment.estimation.readout_types import (
    CellAdvisories,
    CellFailure,
    CellKey,
    FamilyScope,
    Population,
    cell_order,
    family_identity,
)
from increment.readouts._metric_type import declared_metric_type, declared_metric_types
from increment.readouts._source_digest import input_evidence

_NO_ADVISORIES: CellAdvisories = MappingProxyType({})


def attach_multiplicity_scope(
    rows: Sequence[Any],
    cells: Sequence[CellKey],
    plan: Any,
    configs: Sequence[Any],
    snapshot_id: str,
    *,
    view: Literal["run", "breakout", "daily", "asof"] = "run",
    dimension: str | None = None,
    source: str | None = None,
    family_populations: set[str] | None = None,
    view_policy: MultiplicityFamily | None = None,
    decision_snapshot_id: str | None = None,
) -> tuple[list[Any], tuple[FamilyScope, ...]]:
    """Attach row provenance and the unchanged source family membership.

    Membership is resolved from the predeclared source cells and effective
    per-metric configuration, never from surviving or displayed rows.
    """
    configs_by_metric = {config.metric.name: config for config in configs}
    family_cells = {}
    cell_family = {}
    family_by_key = {}
    row_by_cell = {CellKey.from_row(row): row for row in rows}
    for cell in cells:
        procedure = plan.procedures.get(cell.metric)
        if family_populations is not None and cell.analysis_population not in family_populations:
            cell_family[cell] = None
            continue
        row = row_by_cell.get(cell)
        role = (
            None
            if cell.estimand == "compliance"
            else procedure.role
            if plan.declared and procedure is not None
            else None
            if row is None
            else row.role
        )
        membership = None if procedure is None else procedure.family
        config = configs_by_metric.get(cell.metric)
        # Encouragement selection ranks ITT only; its LATE/compliance
        # diagnostics keep their role but are not family members.
        view_member = (
            view_policy is not None
            and view_policy.correction != "none"
            and role in ("primary", "secondary", "exploratory")
            and cell.method_role == "decision"
            and cell.estimand not in ("late", "compliance")
        )
        effective_member = (
            view_member
            if view_policy is not None
            else view != "daily"
            and membership is not None
            and membership.member
            and config is not None
            and cell.estimand not in ("late", "compliance")
            and (family_populations is None or cell.analysis_population in family_populations)
        )
        if effective_member and cell.method_role != "decision":
            cell_family[cell] = None
            continue
        family = None
        if view_member:
            family = view_policy
        elif effective_member and membership is not None:
            candidate = membership.family
            if isinstance(candidate, MultiplicityFamily):
                family = candidate
        registration = _population_registration(plan, cell.analysis_population)
        if family is not None:
            family = _registered_level(family, registration, view)
        if role is None or (role == "secondary" and family is None):
            cell_family[cell] = None
            continue
        registered_asof = (
            view == "asof"
            and getattr(getattr(plan, "inference", None), "registration", None) is not None
        )
        scope_metric = cell.metric if view in ("daily", "asof") and not registered_asof else None
        scope_look = cell.ds if view in ("daily", "asof") else None
        key = (
            cell.analysis_population,
            "view" if view_member else role,
            family.name if family is not None else role,
            family,
            scope_metric,
            scope_look,
        )
        family_cells.setdefault(key, []).append(cell)
        family_by_key[key] = family
        cell_family[cell] = key
    joined = (
        _join_registered_family(
            cells,
            plan,
            cell_family,
            family_cells,
            family_by_key,
            view=view,
            family_populations=family_populations,
        )
        if view_policy is None and view in ("run", "asof")
        else frozenset()
    )

    family_scopes = []
    family_ids = {}
    for key, members in family_cells.items():
        cell_population, _, name, family, metric, look = key
        family_id = family_identity(
            snapshot_id,
            cell_population,
            view,
            dimension,
            source,
            name,
            identity_metric=metric,
            identity_look=look,
            decision_snapshot_id=decision_snapshot_id,
        )
        family_ids[key] = family_id
        family_scopes.append(
            FamilyScope(
                family_id=family_id,
                analysis_population=cell_population,
                source_snapshot_id=snapshot_id,
                view=view,
                dimension=dimension,
                source=source,
                name=name,
                identity_metric=metric,
                identity_look=look,
                decision_snapshot_id=decision_snapshot_id,
                family=family,
                members=tuple(sorted(set(members), key=cell_order)),
                complete=True,
            )
        )

    output = []
    for row in rows:
        cell = CellKey.from_row(row)
        key = cell_family[cell]
        family = None if key is None else family_by_key[key]
        procedure = plan.procedures.get(row.metric)
        role = (
            row.role
            if row.estimand == "compliance"
            else procedure.role
            if plan.declared and procedure is not None
            else row.role
        )
        selection_excluded = (
            family_populations is not None and cell.analysis_population not in family_populations
        )
        output.append(
            row.model_copy(
                update={
                    "role": role,
                    "family_id": None if key is None else family_ids[key],
                    "family_axes": None if family is None else family.axes,
                    "family_procedure": (
                        None if family is None or family.correction == "none" else family.correction
                    ),
                    "family_q": None if family is None else family.q,
                    "family_guarantee": (None if family is None else family.validity_regime),
                    "discovery": None if selection_excluded else row.discovery,
                    "family_threshold": None if selection_excluded else row.family_threshold,
                    "family_nominal_alpha": None
                    if selection_excluded
                    else row.family_nominal_alpha,
                    "multiplicity_status": "declared_plan"
                    if cell in joined
                    else multiplicity_status(role, family),
                }
            )
        )
    return output, tuple(sorted(family_scopes, key=lambda family: family.family_id))


def _population_registration(plan: Any, population: str) -> Any | None:
    """The sequential registration that fixes *population*'s families, if any."""
    inference = getattr(plan, "inference", None)
    if population == "triggered":
        return getattr(inference, "triggered_registration", None)
    return getattr(inference, "registration", None)


def _registered_level(
    family: MultiplicityFamily, registration: Any | None, view: str
) -> MultiplicityFamily:
    """A whole-window family is selected at its registration's ``q``."""
    if view == "run" and registration is not None and family.q is not None:
        return family.model_copy(update={"q": float(registration.q)})
    return family


def _registered_coordinates(cell: CellKey) -> RegisteredCell:
    """An arm or segment cell's registration roster coordinates."""
    segment = (
        ((cell.dimension, cell.dimension_value),)
        if cell.dimension is not None and cell.dimension_value is not None
        else ()
    )
    return cell.metric, cast(str, cell.group_id), cast(str, cell.estimand), segment


def _join_registered_family(
    cells: Sequence[CellKey],
    plan: Any,
    cell_family: dict[CellKey, Any],
    family_cells: dict[Any, list[CellKey]],
    family_by_key: dict[Any, Any],
    *,
    view: Literal["run", "asof"],
    family_populations: set[str] | None,
) -> frozenset[CellKey]:
    """List each registered uptake cell with its population's secondary family.

    A compliance policy declared ``family=True`` registers the uptake cell in the
    e-BH family the secondaries form, so the family's members are the
    registration's, not only the plan's secondary roles. The cell keeps its own
    role. Without secondaries it is the family's one member.
    """
    joined: set[CellKey] = set()
    for population in ("assigned", "triggered"):
        registration = _population_registration(plan, population)
        if registration is None or (
            family_populations is not None and population not in family_populations
        ):
            continue
        # A registration without a family cell registers no uptake cell in one.
        regime = sequential_family_guarantee(registration)
        if regime is None:
            continue
        secondary = {
            key[5]: key for key in family_cells if key[0] == population and key[1] == "secondary"
        }
        members = set(
            registered_family_members(
                registration,
                secondary_keys=(
                    _registered_coordinates(cell)
                    for key in secondary.values()
                    for cell in family_cells[key]
                ),
            )
        )
        for cell in cells:
            if (
                cell.analysis_population != population
                or cell.estimand != "compliance"
                or cell.method_role != "decision"
                or cell_family[cell] is not None
                or _registered_coordinates(cell) not in members
            ):
                continue
            look = cell.ds if view == "asof" else None
            key = secondary.get(look)
            if key is None:
                family = _registered_level(_secondary_policy(plan, regime), registration, view)
                key = secondary[look] = (population, "secondary", family.name, family, None, look)
                family_by_key[key] = family
            family_cells.setdefault(key, []).append(cell)
            cell_family[cell] = key
            joined.add(cell)
    return frozenset(joined)


def _secondary_policy(
    plan: Any, regime: Literal["asymptotic_sequential", "finite_sample"]
) -> MultiplicityFamily:
    """The family a sequential secondary compiles under *plan*, for an uptake cell that is
    a registered family's only member, at that family's validity *regime*."""
    return MultiplicityFamily(
        name="secondary",
        correction="e_bh",
        q=plan.q,
        axes=("metric", "arm"),
        guarantee=guarantee_for_correction("e_bh"),
        validity_regime=regime,
    )


def _attach_breakout_scope(
    rows: Sequence[Any],
    cells: Sequence[CellKey],
    snapshot_id: str,
    request: Any,
    *,
    policy: MultiplicityFamily,
    registration: Any | None,
    dimension: str | None,
    source: str | None,
) -> tuple[list[Any], tuple[FamilyScope, ...]]:
    registered = registration is not None
    correction = request["correction"]
    if registered and correction == "bh":
        correction = "e_bh"
    asymptotic = policy.validity_regime == "asymptotic_sequential" if registered else False
    validity_regime = policy.validity_regime if registered else "finite_sample"
    q = float(registration.q) if registered else request.get("q")
    guarantee = (
        "fdr" if correction in ("bh", "e_bh") else "fwer" if correction == "bonferroni" else "none"
    )
    registered_members = (
        {
            (member.metric, member.group_id, member.estimand, member.segment)
            for member in registration.roster
            if member.family or correction == "bonferroni"
        }
        if registered
        else None
    )
    # Finite-sample Bonferroni is metric-local; asymptotic registered
    # Bonferroni retains its compiled joint family axes.
    metric_local = correction == "bonferroni" and (not registered or not asymptotic)
    # A fixed-horizon BH or Bonferroni family (a registered BH one is e-BH by now) leaves
    # count-excluded cells out, as its selection and its alpha division do.
    count_excluded = (
        {CellKey.from_row(row) for row in rows if design_excluded(row)}
        if correction in ("bh", "bonferroni") and not registered
        else set()
    )
    cells_by_family: dict[tuple[Population, str | None], list[CellKey]] = {}
    for cell in cells:
        if cell.method_role != "decision" or cell in count_excluded:
            continue
        if registered_members is not None:
            segment = (
                ((cell.dimension, cell.dimension_value),)
                if cell.dimension is not None and cell.dimension_value is not None
                else ()
            )
            if (cell.metric, cell.group_id, cell.estimand, segment) not in registered_members:
                continue
        metric = cell.metric if metric_local else None
        cells_by_family.setdefault((cell.analysis_population, metric), []).append(cell)

    families_by_key: dict[tuple[Population, str | None], MultiplicityFamily] = {}
    family_ids: dict[tuple[Population, str | None], str] = {}
    family_scopes: list[FamilyScope] = []
    if correction != "none":
        for (population, metric), members in cells_by_family.items():
            name = f"{policy.name}:{metric}" if metric_local else policy.name
            axes: tuple[str, ...] = policy.axes
            if metric_local:
                # A metric-local family spans its members' segments, and their arms when several.
                arms = {cell.group_id for cell in members}
                axes = ("arm", "segment") if len(arms) > 1 else ("segment",)
            needs_q = correction in ("bh", "e_bh") or (asymptotic and correction == "bonferroni")
            family_q = q if needs_q else None
            family = MultiplicityFamily(
                name=name,
                correction=correction,
                q=family_q,
                axes=axes,
                guarantee=guarantee,
                validity_regime=validity_regime,
            )
            key = population, metric
            family_id = family_identity(
                snapshot_id, population, "breakout", dimension, source, name
            )
            families_by_key[key] = family
            family_ids[key] = family_id
            family_scopes.append(
                FamilyScope(
                    family_id=family_id,
                    analysis_population=population,
                    source_snapshot_id=snapshot_id,
                    view="breakout",
                    dimension=dimension,
                    source=source,
                    name=name,
                    family=family,
                    members=tuple(sorted(set(members), key=cell_order)),
                    complete=True,
                )
            )

    output = []
    for row in rows:
        key = (
            row.analysis_population,
            row.metric if metric_local else None,
        )
        family = families_by_key.get(key)
        belongs = (
            family is not None
            and row.method_role == "decision"
            and CellKey.from_row(row) not in count_excluded
        )
        output.append(
            row.model_copy(
                update={
                    "role": "exploratory",
                    "family_id": family_ids[key] if belongs else None,
                    "family_axes": None if not belongs else family.axes,
                    "family_procedure": None if not belongs else family.correction,
                    "family_q": None if not belongs else family.q,
                    "family_guarantee": None if not belongs else family.validity_regime,
                    "multiplicity_status": row_multiplicity_status(
                        row, correction=correction if belongs else None
                    ),
                }
            )
        )
    return output, tuple(sorted(family_scopes, key=lambda family: family.family_id))


def _identity_request(rows: Sequence[Any], request: Any) -> dict[str, Any]:
    identity_request = dict(request)
    if rows and getattr(rows[0], "sequential_result", None) is not None:
        identity_request.pop("metrics", None)
        identity_request.pop("configs", None)
    return identity_request


def _attach_family_scope(
    stamped: list[Any],
    cells: tuple[Any, ...],
    plan: Any,
    configs: Sequence[Any],
    request: Any,
    snapshot_id: str,
    *,
    view: Literal["run", "breakout", "daily", "asof"],
    design: Any,
    dimension: str | None,
    source: str | None,
    family_populations: set[str] | None,
) -> tuple[list[Any], tuple[Any, ...]]:
    """Stamp family provenance using the view's executed procedure."""
    family_source = source
    if family_source is None:
        sources = {getattr(row, "source", None) for row in stamped}
        if len(sources) == 1:
            family_source = next(iter(sources))
    if view == "breakout":
        policy = plan.view_policies.for_view(
            "breakout", mechanism=getattr(design, "mechanism", None), segmented=True
        )
        registration = getattr(getattr(plan, "inference", None), "registration", None)
        return _attach_breakout_scope(
            stamped,
            cells,
            snapshot_id,
            request,
            policy=policy,
            registration=registration,
            dimension=dimension,
            source=family_source,
        )
    view_policy = None
    if view == "asof" and getattr(getattr(plan, "inference", None), "registration", None) is None:
        view_policy = plan.view_policies.for_view(
            "asof",
            mechanism=getattr(design, "mechanism", None),
            segmented=dimension is not None,
        )
    return attach_multiplicity_scope(
        stamped,
        cells,
        plan,
        configs,
        snapshot_id,
        view=view,
        dimension=dimension,
        source=family_source,
        family_populations=family_populations,
        view_policy=view_policy,
    )


def scoped_collection(  # noqa: PLR0913
    rows: Sequence[Any],
    collection_type: Any,
    plan: Any,
    configs: Sequence[Any],
    request: Any,
    *,
    metrics: Sequence[Any],
    route: str,
    view: Literal["run", "breakout", "daily", "asof"],
    design: Any,
    dimension: str | None = None,
    source: str | None = None,
    family_populations: set[str] | None = None,
    sequential_snapshot: Any = None,
    source_components: Sequence[Mapping[str, Any]] = (),
    identity_rows: Sequence[Any] | None = None,
    artifact_provenance: ArtifactProvenance | None = None,
    advisories: CellAdvisories = _NO_ADVISORIES,
    family_cells: Sequence[CellKey] = (),
):
    """Return rows with a source snapshot and source-local readout metadata.

    ``source_components`` are provenance components of evidence the rows were
    computed from but do not carry themselves (per-unit evidence digests);
    they enter the source half of the snapshot identity, as the headline
    route's evidence components do. ``metrics`` are the declared metrics the
    rows were read for; every non-compliance row must name one of them.
    ``artifact_provenance`` names the artifact the rows were read from.
    ``advisories`` are the estimators' typed advisories keyed by each
    retained cell's full identity. ``family_cells`` are cells the request
    leaves out: the families list them, as the whole-window scope lists a
    registered family's unrequested cells, but the returned scope does not.
    """
    from hashlib import sha256

    from increment._canonical import canonical_json_bytes
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.readout_types import (
        CellKey,
        CellRecord,
        PopulationRoster,
        PosteriorInference,
        ReadoutMetadata,
        ReadoutScope,
        SamplingInference,
        SourceReadoutScope,
    )
    from increment.readouts._design_scope import _request_snapshot

    identity_rows = rows if identity_rows is None else identity_rows
    identity_request = _identity_request(identity_rows, request)
    payload: dict[str, Any] = {
        "kind": "increment.readout.snapshot",
        "version": 2,
        "collection": collection_type.__name__,
        "view": view,
        "request": _request_snapshot(
            {
                **identity_request,
                "plan": {
                    "declared": plan.declared,
                    "alpha": plan.alpha,
                    "q": plan.q,
                    "procedures": {
                        name: evidence_request_procedure(procedure, mode="python")
                        for name, procedure in plan.procedures.items()
                    },
                },
            }
        ),
        "source": {
            "kind": "readout_input_evidence",
            "route": route,
            "source": source,
            "sha256": sha256(
                canonical_json_bytes(
                    sorted(
                        (
                            input_evidence(
                                row.model_dump(mode="json", exclude={"source_snapshot_id"})
                            )
                            for row in identity_rows
                        ),
                        key=canonical_json_bytes,
                    )
                )
            ).hexdigest(),
        },
    }
    if source_components:
        payload["source"] = {
            **payload["source"],
            "components": [dict(item) for item in source_components],
        }
    snapshot_id = "sha256:" + sha256(canonical_json_bytes(payload)).hexdigest()
    metric_types = declared_metric_types(metrics)
    stamped = []
    for row in rows:
        update = {
            "source_snapshot_id": snapshot_id,
            "metric_type": declared_metric_type(row, metric_types),
        }
        if hasattr(row, "sampling_available") and row.sampling_available is None:
            update["sampling_available"] = True
        stamped.append(row.model_copy(update=update))
    cells = tuple(sorted({CellKey.from_row(row) for row in stamped}, key=cell_order))
    families = ()
    if stamped and hasattr(stamped[0], "family_id"):
        stamped, families = _attach_family_scope(
            stamped,
            tuple(sorted({*cells, *family_cells}, key=cell_order)),
            plan,
            configs,
            request,
            snapshot_id,
            view=view,
            design=design,
            dimension=dimension,
            source=source,
            family_populations=family_populations,
        )
    populations = tuple(sorted({cell.analysis_population for cell in cells}))
    groups = {cell.group_id for cell in cells if cell.group_id is not None}
    control = getattr(design, "control_group", None)
    if control is not None:
        groups.add(control)
    rosters = tuple(
        PopulationRoster(
            analysis_population=population,
            arms=tuple(sorted(groups)),
            source="observed_union",
            complete=False,
            control_group=control,
        )
        for population in populations
    )
    completeness = {}
    for population in populations:
        scoped_rows = [row for row in stamped if row.analysis_population == population]
        explicit = [
            row.decision_scope_complete
            for row in scoped_rows
            if getattr(row, "decision_scope_complete", None) is not None
        ]
        if explicit:
            completeness[population] = all(explicit)
    decision_cells = tuple(cell for cell in cells if cell.method_role == "decision")
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=cells,
        decision_cells=decision_cells,
        rosters=rosters,
        decision_complete_by_population=completeness,
        integrity=(),
    )
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=cells,
        decision_cells=decision_cells,
        populations=populations,
        families=families,
        by_source={snapshot_id: source_scope},
    )
    config_by_metric = {config.metric.name: config for config in configs}
    provenance = artifact_provenance or ArtifactProvenance()
    records: list[CellRecord] = []
    for row in stamped:
        cell = CellKey.from_row(row)
        sampling = None
        if hasattr(row, "sampling_available"):
            evidence = None
            if row.sampling_available and row.lift is not None:
                try:
                    p_value = row.p_value()
                    if isinstance(p_value, (int, float)):
                        evidence = PValueEvidence(
                            cell.hypothesis(), cell.method or "unknown", p_value, row.reference_kind
                        )
                except (AttributeError, ValueError):
                    evidence = None
            sampling = SamplingInference(
                available=row.sampling_available,
                evidence=evidence,
                reason_code=row.sampling_reason_code,
                reason_context=row.sampling_reason_context,
            )
        config = config_by_metric.get(cell.metric)
        posterior = None
        if config is not None and config.prior is not None and hasattr(row, "posterior_available"):
            posterior = PosteriorInference(
                available=row.posterior_available,
                model=row.posterior_model,
                scale=row.posterior_scale,
                prior=config.prior,
                reason_code=row.posterior_reason_code,
                reason_context=row.posterior_reason_context,
            )
        records.append(
            CellRecord(
                cell=cell,
                source_snapshot_id=snapshot_id,
                sampling=sampling,
                posterior=posterior,
                failure=(
                    CellFailure(
                        hypothesis=cell.hypothesis(),
                        code=row.failure_code,
                        context=row.failure_context or {},
                    )
                    if getattr(row, "failure_code", None) is not None and cell.method is not None
                    else None
                ),
                advisories=advisories.get(cell, ()),
            )
        )
    metadata = ReadoutMetadata(
        scope=scope,
        cells=tuple(
            sorted(records, key=lambda record: cell_order(cast("CellRecord", record).cell))
        ),
        artifact_manifest_sha256=provenance.manifest_sha256,
        artifact_relation_digests=provenance.relation_digests,
    )
    return collection_type(stamped, metadata=metadata, sequential_snapshot=sequential_snapshot)
