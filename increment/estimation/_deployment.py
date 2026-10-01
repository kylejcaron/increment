"""Shared scoring identities and whole-cluster deployment budgets."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from increment._identity import canonical_id_strings
from increment.errors import InvalidRequestError, RefusalSpec, refuse

_UNSUPPORTED_UNIT = RefusalSpec(
    "estimation.targeting.unsupported_unit_deployment",
    InvalidRequestError,
    template="A declared cluster intervention requires whole-cluster deployment",
)
_INVALID_GRAIN = RefusalSpec(
    "estimation.targeting.invalid_deployment_grain",
    InvalidRequestError,
    template="Invalid intervention/deployment grains: {intervention_grain!r}, {deploy_grain!r}",
)
_IDS_REQUIRED = RefusalSpec(
    "estimation.targeting.deployment_cluster_ids_required",
    InvalidRequestError,
    template="Cluster deployment requires aligned cluster IDs",
)
_IDS_SHAPE = RefusalSpec(
    "estimation.targeting.deployment_cluster_ids_shape",
    InvalidRequestError,
    template="Deployment cluster IDs have shape {shape}; expected ({n},)",
)


class ClusterScore(BaseModel):
    """A pooled score keyed by canonical cluster identity, in canonical ID order."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    score: float = Field(allow_inf_nan=False)
    member_count: int = Field(gt=0, strict=True)


def resolve_deploy_grain(
    intervention_grain: Literal["unit", "cluster"],
    deploy_grain: Literal["unit", "cluster"] | None,
    *,
    clustered: bool,
) -> Literal["unit", "cluster"]:
    """Resolve the declared intervention, never observed arm purity or dependence."""
    if intervention_grain not in ("unit", "cluster") or deploy_grain not in (
        None,
        "unit",
        "cluster",
    ):
        refuse(_INVALID_GRAIN, intervention_grain=intervention_grain, deploy_grain=deploy_grain)
    if intervention_grain == "cluster" and deploy_grain == "unit":
        refuse(_UNSUPPORTED_UNIT)
    resolved = intervention_grain if deploy_grain is None else deploy_grain
    if (resolved == "cluster" or intervention_grain == "cluster") and not clustered:
        refuse(_IDS_REQUIRED)
    return resolved


def deployment_ids(cluster_ids: np.ndarray | None, n: int) -> np.ndarray | None:
    """Validate identities before scoring, including IDs supplied for unit scores."""
    if cluster_ids is None:
        return None
    ids = np.asarray(cluster_ids)
    if ids.shape != (n,):
        refuse(_IDS_SHAPE, shape=ids.shape, n=n)
    return canonical_id_strings(ids, what="cluster_ids")


def _pooled_mean(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    # Sum bounded terms before division: neither the sum nor the restored
    # mean can overflow, and fsum makes pooling independent of member order.
    return scale * (math.fsum((values / scale).tolist()) / values.size)


def _cluster_codes(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical labels in canonical order, each label's first row, row codes, member counts."""
    if ids.dtype.kind in "iu":
        # Canonicalizing the distinct integers renders the same labels as every row would.
        raw, first, inverse, counts = np.unique(
            ids, return_index=True, return_inverse=True, return_counts=True
        )
        labels = canonical_id_strings(raw, what="cluster_ids")
        rank = np.argsort(labels)
        position = np.empty(rank.size, dtype=np.intp)
        position[rank] = np.arange(rank.size)
        return labels[rank], first[rank], position[inverse], counts[rank]
    canonical = canonical_id_strings(ids, what="cluster_ids")
    return np.unique(canonical, return_index=True, return_inverse=True, return_counts=True)


def _pooled_scores(score: np.ndarray, inverse: np.ndarray, counts: np.ndarray) -> list[float]:
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(counts)[:-1]) if counts.size else ()
    return [_pooled_mean(score[rows]) for rows in members]


def pool_cluster_scores(score: np.ndarray, ids: np.ndarray) -> tuple[ClusterScore, ...]:
    """Arithmetic member means with row-order-independent summation."""
    labels, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return tuple(
        ClusterScore(cluster_id=str(label), score=value, member_count=int(count))
        for label, value, count in zip(
            labels, _pooled_scores(score, inverse, counts), counts, strict=True
        )
    )


class Deployment(NamedTuple):
    targeted: np.ndarray
    threshold: float | None
    achieved_fraction: float


def cluster_prefix(
    score: np.ndarray,
    ids: np.ndarray,
    fraction: float,
    weighting: Literal["member_count", "equal"],
    *,
    source_ids: np.ndarray | None = None,
) -> Deployment:
    """Longest feasible score-ordered prefix; never skip an oversized next cluster."""
    labels, first, inverse, counts = _cluster_codes(ids)
    pooled = _pooled_scores(score, inverse, counts)
    for label, value, count in zip(labels, pooled, counts, strict=True):
        if not math.isfinite(value):
            # The pooled score model refuses a non-finite mean; raise its error.
            ClusterScore(cluster_id=label, score=value, member_count=int(count))
    ordering = (
        labels
        if source_ids is None
        else canonical_id_strings(source_ids, what="cluster_ids")[first]
    )
    order = sorted(range(labels.size), key=lambda g: (-pooled[g], ordering[g], labels[g]))
    total = score.size if weighting == "member_count" else labels.size
    selected: list[int] = []
    used = 0
    threshold = None
    for g in order:
        mass = int(counts[g]) if weighting == "member_count" else 1
        # Compare shares directly so an exactly representable requested share
        # and the same computed share agree, without a tolerance permitting overspend.
        if (used + mass) / total > fraction:
            break
        used += mass
        selected.append(g)
        threshold = pooled[g]
    return Deployment(np.isin(inverse, selected), threshold, used / total if total else 0.0)
