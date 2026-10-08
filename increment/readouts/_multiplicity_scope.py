"""Project declared multiplicity provenance onto scoped readout rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from increment.decision import MultiplicityFamily
from increment.estimation.multiplicity import multiplicity_status, row_multiplicity_status
from increment.estimation.readout_types import (
    CellKey,
    FamilyScope,
    Population,
    cell_order,
    family_identity,
)


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
        effective_member = (
            membership is not None
            and membership.member
            and config is not None
            and cell.estimand not in ("late", "compliance")
            and (family_populations is None or cell.analysis_population in family_populations)
        )
        if effective_member and cell.method_role != "decision":
            cell_family[cell] = None
            continue
        family = (
            membership.family
            if effective_member and isinstance(membership.family, MultiplicityFamily)
            else None
        )
        if role is None or (role == "secondary" and family is None):
            cell_family[cell] = None
            continue
        key = (
            cell.analysis_population,
            role,
            family.name if family is not None else role,
            family,
        )
        family_cells.setdefault(key, []).append(cell)
        family_by_key[key] = family
        cell_family[cell] = key

    family_scopes = []
    family_ids = {}
    for key, members in family_cells.items():
        cell_population, _, name, family = key
        family_id = family_identity(snapshot_id, cell_population, view, dimension, source, name)
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
                family=family,
                members=tuple(sorted(set(members), key=cell_order)),
                complete=True,
            )
        )

    output = []
    for row in rows:
        cell = CellKey.from_row(row)
        key = cell_family[cell]
        procedure = plan.procedures.get(row.metric)
        role = (
            row.role
            if row.estimand == "compliance"
            else procedure.role
            if plan.declared and procedure is not None
            else row.role
        )
        family = None if key is None else family_by_key[key]
        output.append(
            row.model_copy(
                update={
                    "role": role,
                    "family_id": None if key is None else family_ids[key],
                    "multiplicity_status": multiplicity_status(role, family),
                }
            )
        )
    return output, tuple(sorted(family_scopes, key=lambda family: family.family_id))


def _attach_breakout_scope(
    rows: Sequence[Any],
    cells: Sequence[CellKey],
    snapshot_id: str,
    request: Any,
    *,
    dimension: str | None,
    source: str | None,
) -> tuple[list[Any], tuple[FamilyScope, ...]]:
    correction = request["correction"]
    asymptotic = any(row.inference == "asymptotic_mean" for row in rows)
    validity_regime = "asymptotic_sequential" if asymptotic else "finite_sample"
    q = request.get("q")
    guarantee = (
        "fdr" if correction in ("bh", "e_bh") else "fwer" if correction == "bonferroni" else "none"
    )
    # BH/e-BH span metrics, arms and segments; Bonferroni is metric-local across segments.
    cells_by_family: dict[tuple[Population, str | None], list[CellKey]] = {}
    for cell in cells:
        if cell.method_role != "decision":
            continue
        metric = cell.metric if correction == "bonferroni" else None
        cells_by_family.setdefault((cell.analysis_population, metric), []).append(cell)

    families_by_key: dict[tuple[Population, str | None], MultiplicityFamily] = {}
    family_ids: dict[tuple[Population, str | None], str] = {}
    family_scopes: list[FamilyScope] = []
    if correction != "none":
        for (population, metric), members in cells_by_family.items():
            name = "breakout" if metric is None else f"breakout:{metric}"
            axes = ("metric", "arm", "segment") if metric is None else ("segment",)
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
            row.metric if correction == "bonferroni" else None,
        )
        family = families_by_key.get(key)
        belongs = family is not None and row.method_role == "decision"
        output.append(
            row.model_copy(
                update={
                    "role": "exploratory",
                    "family_id": family_ids[key] if belongs else None,
                    "family_axes": None if not belongs else family.axes,
                    "family_q": None if not belongs else family.q,
                    "family_guarantee": None if not belongs else family.validity_regime,
                    "multiplicity_status": row_multiplicity_status(
                        row, correction=correction if belongs else None
                    ),
                }
            )
        )
    return output, tuple(sorted(family_scopes, key=lambda family: family.family_id))


def scoped_collection(
    rows: Sequence[Any],
    collection_type: Any,
    plan: Any,
    configs: Sequence[Any],
    request: Any,
    *,
    route: str,
    view: Literal["run", "breakout", "daily", "asof"],
    design: Any,
    dimension: str | None = None,
    source: str | None = None,
):
    """Return rows with a source snapshot and source-local readout metadata."""
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

    payload = {
        "kind": "increment.readout.snapshot",
        "version": 1,
        "collection": collection_type.__name__,
        "route": route,
        "view": view,
        "request": _request_snapshot(
            {
                **request,
                "plan": {
                    "declared": plan.declared,
                    "alpha": plan.alpha,
                    "q": plan.q,
                    "procedures": {
                        name: procedure.model_dump(mode="python")
                        for name, procedure in plan.procedures.items()
                    },
                },
            }
        ),
        "source": {
            "kind": "readout_rows",
            "sha256": sha256(
                canonical_json_bytes(
                    sorted(
                        (
                            row.model_dump(mode="json", exclude={"source_snapshot_id"})
                            for row in rows
                        ),
                        key=canonical_json_bytes,
                    )
                )
            ).hexdigest(),
        },
    }
    snapshot_id = "sha256:" + sha256(canonical_json_bytes(payload)).hexdigest()
    stamped = []
    for row in rows:
        update = {"source_snapshot_id": snapshot_id}
        if hasattr(row, "sampling_available") and row.sampling_available is None:
            update["sampling_available"] = True
        stamped.append(row.model_copy(update=update))
    cells = tuple(sorted({CellKey.from_row(row) for row in stamped}, key=cell_order))
    families = ()
    if stamped and hasattr(stamped[0], "family_id"):
        family_source = source
        if family_source is None:
            sources = {getattr(row, "source", None) for row in stamped}
            if len(sources) == 1:
                family_source = next(iter(sources))
        if view == "breakout":
            stamped, families = _attach_breakout_scope(
                stamped,
                cells,
                snapshot_id,
                request,
                dimension=dimension,
                source=family_source,
            )
        else:
            stamped, families = attach_multiplicity_scope(
                stamped,
                cells,
                plan,
                configs,
                snapshot_id,
                view=view,
                dimension=dimension,
                source=family_source,
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
        )
        for population in populations
    )
    decision_cells = tuple(cell for cell in cells if cell.method_role == "decision")
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=cells,
        decision_cells=decision_cells,
        rosters=rosters,
        decision_complete_by_population={},
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
                cell=cell, source_snapshot_id=snapshot_id, sampling=sampling, posterior=posterior
            )
        )
    metadata = ReadoutMetadata(
        scope=scope,
        cells=tuple(
            sorted(records, key=lambda record: cell_order(cast("CellRecord", record).cell))
        ),
    )
    return collection_type(stamped, metadata=metadata)
