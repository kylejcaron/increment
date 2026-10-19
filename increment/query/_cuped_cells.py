"""Bounded joint categorical evidence with warehouse-only reconstruction guards."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, NoReturn, cast

import ibis
import ibis.expr.types as ir
import narwhals as nw
import numpy as np

from increment.errors import CapabilityError, RefusalSpec, refuse
from increment.estimation.cuped.cells import _pattern_key
from increment.estimation.cuped.types import CategoricalCells, _freeze_payload

if TYPE_CHECKING:
    from increment.query.cuped import _ResolvedCupedRequest

MAX_CELLS = 4096
MAX_ENCODED_BYTES = 8 * 1024 * 1024
MIN_COMPRESSION = 8
MIN_CONTRIBUTORS = 3

_GENERAL_QR = RefusalSpec(
    "query.cuped.general_qr_required",
    CapabilityError,
    template="Metric {metric!r} requires general warehouse QR ({reason}); {route}",
    keys=frozenset({"metric", "reason", "route"}),
)
REFUSALS = {_GENERAL_QR.code: _GENERAL_QR}


def general_qr_required(metric, reason: str) -> NoReturn:
    refuse(
        _GENERAL_QR,
        metric=metric.name,
        reason=reason,
        route="Use explicitly supplied local per-unit data with Analysis.from_unit_summary or Analysis.from_unit_panel; warehouse units are never downloaded automatically.",
    )


def known_outcome_domain(metric):
    if metric.type not in ("mean", "ratio") or getattr(metric, "winsorization", None) is not None:
        return True
    measures = (metric.numerator, metric.denominator) if metric.type == "ratio" else (metric,)
    return any(
        getattr(measure, "aggregation", None) in ("count", "count_distinct", "any")
        for measure in measures
    )


def _known_outcome_domain(metric, relation, outcomes):
    if known_outcome_domain(metric):
        return True
    return any(
        relation[name].type().is_integer() or relation[name].type().is_boolean()
        for name in outcomes
    )


def component_domain(measure, datatype):
    """Retain the original outcome domain before numerical preparation."""
    if getattr(measure, "aggregation", None) in ("count", "count_distinct", "any"):
        return "discrete"
    if datatype is None or datatype.is_integer() or datatype.is_boolean() or datatype.is_decimal():
        return "discrete"
    return "continuous" if datatype.is_floating() else "unknown"


def _binary_normalize(value, exponent):
    half = exponent // 2
    return value.cast("float64") / (2.0**half) / (2.0 ** (exponent - half))


def _component_exponent(value):
    maximum = value.cast("float64").abs().max()
    return ibis.ifelse(
        (maximum > 0) & ~maximum.isnan() & ~maximum.isinf(),
        (maximum.log2().floor() + 1).clip(upper=1024),
        0,
    ).cast("int64")


def _join_on(left, right, keys):
    joined = left.join(right, [left[name].identical_to(right[name]) for name in keys])
    return joined.select(
        *[left[name] for name in left.columns],
        **{name: right[name] for name in right.columns if name not in keys},
    )


def _cell_relation(relation, names, outcomes):
    keys = ("group_id", *names)
    exponents = relation.aggregate(
        **{f"exponent_{f}": _component_exponent(relation[name]) for f, name in enumerate(outcomes)}
    )
    relation = relation.cross_join(exponents)
    bases = relation.group_by("group_id").aggregate(
        **{
            f"_base_{f}": relation[name].first(order_by=relation.unit_id)
            for f, name in enumerate(outcomes)
        }
    )
    shifted = _join_on(relation, bases, ("group_id",))
    references = shifted.group_by("group_id").aggregate(
        **{
            f"ref_{f}": shifted[f"_base_{f}"].first()
            + (shifted[name].cast("float64") - shifted[f"_base_{f}"]).mean()
            for f, name in enumerate(outcomes)
        }
    )
    referenced = _join_on(relation, references, ("group_id",))
    centered = referenced.mutate(
        **{
            f"_centered_{f}": _binary_normalize(referenced[name], referenced[f"exponent_{f}"])
            - _binary_normalize(referenced[f"ref_{f}"], referenced[f"exponent_{f}"])
            for f, name in enumerate(outcomes)
        }
    )
    aggregates = {"n": centered.count()}
    for f, name in enumerate(outcomes):
        value = centered[name].cast("float64")
        finite = value.notnull() & ~value.isnan() & ~value.isinf()
        aggregates.update(
            {
                f"ref_{f}": centered[f"ref_{f}"].first(),
                f"exponent_{f}": centered[f"exponent_{f}"].first(),
                f"first_{f}": centered[f"_centered_{f}"].sum(),
                f"contributors_{f}": finite.cast("int64").sum(),
                f"distinct_{f}": value.min() < value.max(),
                f"max_centered_{f}": centered[f"_centered_{f}"].abs().max(),
            }
        )
    first = centered.group_by(list(keys)).aggregate(**aggregates)
    means = first.select(
        *keys, **{f"_mean_{f}": first[f"first_{f}"] / first.n for f in range(len(outcomes))}
    )
    residuals = _join_on(centered, means, keys)
    residuals = residuals.mutate(
        **{
            f"_residual_{f}": residuals[f"_centered_{f}"] - residuals[f"_mean_{f}"]
            for f in range(len(outcomes))
        }
    )
    cross = residuals.group_by(list(keys)).aggregate(
        **{
            f"cross_{f}_{g}": (residuals[f"_residual_{f}"] * residuals[f"_residual_{g}"]).sum()
            for f in range(len(outcomes))
            for g in range(f, len(outcomes))
        }
    )
    return _join_on(first, cross, keys)


def _preflight(cells, names, components, request, metric):
    positive, contributors = [], []
    for f in range(components):
        scatter = cells[f"cross_{f}_{f}"]
        error = 64 * np.finfo(float).eps * cells.n * cells[f"max_centered_{f}"] ** 2
        positive.append(
            cells[f"distinct_{f}"] & (scatter > error) & ~scatter.isnan() & ~scatter.isinf()
        )
        contributors.append(cells[f"contributors_{f}"])
    label_bytes = ibis.literal(0, type="int64")
    for name in ("group_id", *names):
        label_bytes += 12 * cells[name].cast("string").length().fill_null(0) + 16
    # Labels occur in both patterns and level metadata; dense arrays include zero-count cells.
    cell_count = cells.count()
    arms = cells.group_id.nunique()
    schema_bytes = sum(12 * len(name) + 64 for name in names)
    dense_bytes = 32 + 32 * (components + components * components)
    identity_bytes = len(
        json.dumps(
            {
                "identity": request.identity,
                "method": request.method.model_dump(mode="json"),
                "metric": metric.model_dump(mode="json"),
            },
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )
    encoded_bytes = (
        4096
        + identity_bytes
        + cell_count * (512 + schema_bytes)
        + 2 * label_bytes.sum()
        + arms * cell_count * dense_bytes
    )
    return cells.aggregate(
        cell_count=cell_count,
        unit_count=cells.n.sum(),
        min_contributors=cast(ir.IntegerColumn, ibis.least(*contributors)).min(),
        components_complete=cast(
            ir.BooleanColumn, ibis.and_(*[count == cells.n for count in contributors])
        ).all(),
        scatter_positive=cast(ir.BooleanColumn, ibis.and_(*positive)).all(),
        encoded_bytes=encoded_bytes,
    )


def _rows(value):
    return list(nw.from_native(value, eager_only=True).iter_rows(named=True))


def categorical_cells_from_relation(
    relation, execute, *, metric, request: _ResolvedCupedRequest
) -> CategoricalCells:
    """Reduce a pinned unit relation; only eligible joint aggregates may cross the boundary."""
    declarations = request.method.covariates or ()
    if any(item.kind != "categorical" for item in declarations):
        general_qr_required(metric, "continuous_covariates")
    names = tuple(item.name for item in declarations)
    outcomes = ("y", "y_den") if metric.type == "ratio" else ("y",)
    if _known_outcome_domain(metric, relation, outcomes):
        general_qr_required(metric, "outcome_domain")
    relation = relation.filter(relation.group_id.isin(request.arm_order))
    cells = _cell_relation(relation, names, outcomes)
    preflight = _rows(execute(_preflight(cells, names, len(outcomes), request, metric)))[0]
    count = int(preflight["cell_count"])
    if not 0 < count <= MAX_CELLS:
        general_qr_required(metric, "cell_count")
    if preflight["unit_count"] < MIN_COMPRESSION * count:
        general_qr_required(metric, "compression")
    if not preflight["components_complete"] or preflight["min_contributors"] < MIN_CONTRIBUTORS:
        general_qr_required(metric, "component_contributors")
    if not preflight["scatter_positive"]:
        general_qr_required(metric, "component_scatter")
    if preflight["encoded_bytes"] is None or preflight["encoded_bytes"] > MAX_ENCODED_BYTES:
        general_qr_required(metric, "encoded_bytes")
    fields = [*names, "group_id", "n"]
    fields += [f"{kind}_{f}" for f in range(len(outcomes)) for kind in ("ref", "first", "exponent")]
    fields += [f"cross_{f}_{g}" for f in range(len(outcomes)) for g in range(f, len(outcomes))]
    rows = _rows(execute(cells.select(*fields)))
    return _as_cells(rows, names, len(outcomes), request)


def _as_cells(rows, names, components, request):
    patterns = tuple(sorted({tuple(row[name] for name in names) for row in rows}, key=_pattern_key))
    lookup = {pattern: k for k, pattern in enumerate(patterns)}
    arms = request.arm_order
    counts = np.zeros((len(arms), len(patterns)), dtype=np.int64)
    references = np.zeros((components, len(arms)))
    first = np.zeros((len(arms), components, len(patterns)))
    cross = np.zeros((len(arms), components, components, len(patterns)))
    for row in rows:
        a = arms.index(row["group_id"])
        k = lookup[tuple(row[name] for name in names)]
        counts[a, k] = row["n"]
        for f in range(components):
            references[f, a] = row[f"ref_{f}"]
            first[a, f, k] = row[f"first_{f}"]
            for g in range(f, components):
                cross[a, f, g, k] = cross[a, g, f, k] = row[f"cross_{f}_{g}"]
    levels = tuple(
        tuple(
            sorted(
                {pattern[column] for pattern in patterns},
                key=lambda value: (value is None, value or ""),
            )
        )
        for column in range(len(names))
    )
    return CategoricalCells(
        identity=request.identity,
        arms=arms,
        patterns=patterns,
        counts=_freeze_payload(counts),
        levels=levels,
        null_metadata=tuple(
            (item.missing, None in level)
            for item, level in zip(request.method.covariates or (), levels, strict=True)
        ),
        outcome_references=_freeze_payload(references),
        centered_first_moments=_freeze_payload(first),
        symmetric_cross_moments=_freeze_payload(cross),
        covariate_names=names,
        outcome_exponents=tuple(int(rows[0][f"exponent_{f}"]) for f in range(components)),
    )
