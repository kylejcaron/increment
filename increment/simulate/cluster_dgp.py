"""Independent finite-population oracles and prospective clustered CATE cells.

No estimator supplies potential outcomes. Conditional on the covariates,
Y(a) = b(X) + sigma_a * (sqrt(ICC) U_g + sqrt(1-ICC) E_gi) + a*t(X).
U and E are independent, centered, variance-one (possibly skewed) innovations.
ICC describes residual dependence within each potential-outcome arm. Assignment
has a separate stream, including when sizes, dimensions or noise change.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
from itertools import product
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from increment.errors import CodedError, CodedModel, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from increment.estimation.targeting import CateValidation, TargetingRule
    from increment.power import PowerResult
    from increment.simulate.runner import _KeyOutcome, _KeyStats
    from increment.sources import MomentSource

Role = Literal["training", "holdout", "oracle"]
Weight = Literal["member_count", "equal"]
Grain = Literal["unit", "cluster"]

_INVALID = RefusalSpec(
    "simulate.cluster_dgp.invalid_scenario",
    InvalidRequestError,
    template="invalid clustered CATE request: {reason}",
)
_OVERLAP_ROSTER = RefusalSpec(
    "simulate.cluster_dgp.overlap_roster_unavailable",
    InvalidRequestError,
    template="The public overlap-subpopulation result does not expose its retained evaluation roster",
)


class _Frozen(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ClusteredCATEScenario(_Frozen):
    """Executable controls; no aliases or ignored descriptive labels.

    Informative sizes depend on the observed cluster signal z, which also
    modifies the effect. Observational assignment depends only on observed
    (z,x0); these are included in the adjustment set. Omitting z is an explicit
    misspecification experiment. Unit randomization with dependence clusters
    is generated faithfully even though the current randomized adapter refuses
    mixed clusters; observational unit policies have no purity restriction.
    """

    n_clusters: int = Field(default=40, ge=2, strict=True)
    members_per_cluster: int = Field(default=20, ge=1, strict=True)
    member_counts: tuple[Annotated[int, Field(strict=True, ge=1)], ...] | None = None
    size_mode: Literal["fixed", "variable", "informative"] = "fixed"
    size_spread: float = Field(default=0.6, gt=0, le=2)
    icc: float = Field(default=0.2, ge=0, lt=1)
    treatment_ratio: float = Field(default=0.5, gt=0, lt=1)
    assignment: Literal["cluster", "unit", "observational"] = "cluster"
    declare_clusters: bool = True
    intervention_grain: Grain = "unit"
    dimension: int = Field(default=2, ge=0, le=20, strict=True)
    leverage: float = Field(default=0, ge=0, le=100)
    effect: float = 0.5
    heterogeneity: float = 0.35
    size_effect: float = 0.4
    skew: float = Field(default=0, ge=0, le=3)
    outcome_scale: float = Field(default=1, gt=0)
    treated_noise_ratio: float = Field(default=1, gt=0)
    omit_confounder: bool = False
    id_style: Literal["plain", "unicode", "integer"] = "plain"
    reverse_rows: bool = False
    clone_factor: int = Field(default=1, ge=1, strict=True)
    witness: Literal["random", "original_varying_noise"] = "random"
    seed: int = Field(default=20260920, ge=0, strict=True)

    @model_validator(mode="after")
    def _combinations(self) -> ClusteredCATEScenario:
        if self.witness == "original_varying_noise" and (
            self.n_clusters not in (16, 80)
            or self.members_per_cluster not in (5, 100)
            or self.dimension != 1
            or self.assignment != "cluster"
            or self.size_mode != "fixed"
            or not self.declare_clusters
        ):
            refuse(
                _INVALID,
                reason="original witness requires K=16/80, m=5/100, dimension=1, fixed cluster assignment",
            )
        if self.witness == "original_varying_noise":
            controls = {
                "witness",
                "n_clusters",
                "members_per_cluster",
                "dimension",
                "intervention_grain",
                "clone_factor",
                "reverse_rows",
                "seed",
            }
            for name, field in type(self).model_fields.items():
                if name not in controls and getattr(self, name) != field.default:
                    refuse(
                        _INVALID, reason=f"the archived witness does not support changing {name}"
                    )
        if self.member_counts is not None:
            if self.size_mode != "variable":
                refuse(_INVALID, reason="explicit member_counts require size_mode='variable'")
            if len(self.member_counts) != self.n_clusters or any(
                isinstance(n, bool) or n < 1 for n in self.member_counts
            ):
                refuse(_INVALID, reason="member_counts require one positive integer per cluster")
        if self.intervention_grain == "cluster" and not self.declare_clusters:
            refuse(_INVALID, reason="cluster intervention requires declared cluster identity")
        if not self.declare_clusters and (self.assignment != "unit" or self.icc != 0):
            refuse(
                _INVALID, reason="unclustered cells require independent unit assignment and ICC=0"
            )
        if self.leverage and self.dimension == 0:
            refuse(_INVALID, reason="leverage requires at least one x column")
        if self.omit_confounder and self.assignment != "observational":
            refuse(_INVALID, reason="omit_confounder applies only to observational assignment")
        return self

    @property
    def covariates(self) -> tuple[str, ...]:
        return ("z", *(f"x{j}" for j in range(self.dimension)))

    @property
    def interactions(self) -> tuple[str, ...]:
        if self.witness == "original_varying_noise":
            return ("x0",)
        return self.covariates if self.dimension else ()

    @property
    def adjustment(self) -> tuple[str, ...]:
        return tuple(name for name in self.covariates if not (self.omit_confounder and name == "z"))


class ClusteredCATETruth(_Frozen):
    member_ate: float
    equal_cluster_ate: float
    conditional_member_ate: float
    conditional_equal_cluster_ate: float
    n_units: int = Field(gt=0)
    n_clusters: int = Field(gt=0)

    @model_validator(mode="after")
    def _counts(self) -> ClusteredCATETruth:
        if self.n_clusters > self.n_units:
            refuse(_INVALID, reason="truth cannot have more clusters than units")
        return self


class PolicyTruth(_Frozen):
    """Causal selected-population effect, separately from outcome and net benefit.

    Equal weighting gives each retained evaluation cluster total mass one before
    policy selection, including when a unit policy selects only part of it.
    Empty selection has an unavailable effect, and zero population net benefit.
    """

    member_effect: float | None
    equal_cluster_effect: float | None
    member_share: float = Field(ge=0, le=1)
    equal_cluster_share: float = Field(ge=0, le=1)
    member_net_benefit: float
    equal_cluster_net_benefit: float
    member_outcome: float
    equal_cluster_outcome: float
    unavailable_reason: str | None
    actions: tuple[bool, ...] = Field(min_length=1)
    rule_json: str

    @model_validator(mode="after")
    def _availability(self) -> PolicyTruth:
        selected = any(self.actions)
        if selected != (self.member_effect is not None and self.equal_cluster_effect is not None):
            refuse(_INVALID, reason="policy effect availability must match selected actions")
        if selected == (self.unavailable_reason is not None):
            refuse(_INVALID, reason="empty policy alone requires an unavailable reason")
        if not math.isclose(self.member_share, sum(self.actions) / len(self.actions)):
            refuse(_INVALID, reason="member share disagrees with policy actions")
        return self


@dataclass(frozen=True, slots=True)
class ClusteredCATEResult:
    """Owned immutable snapshots; exported arrays/tables never alias stored data."""

    scenario: ClusteredCATEScenario
    stream: Role
    replication: int
    rows: tuple[tuple[object, ...], ...]
    columns: tuple[str, ...]
    y0: tuple[float, ...]
    y1: tuple[float, ...]
    conditional_tau: tuple[float, ...]
    cluster_index: tuple[int, ...]
    sizes: tuple[int, ...]
    innovation_cluster: tuple[str, ...] | None = None
    innovation_unit: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        # Tuple conversion also protects direct constructors from caller-owned lists.
        object.__setattr__(self, "rows", tuple(tuple(row) for row in self.rows))
        for name in ("columns", "y0", "y1", "conditional_tau", "cluster_index", "sizes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.stream not in ("training", "holdout", "oracle") or self.replication < 0:
            refuse(_INVALID, reason="invalid result stream or replication")
        for name in ("innovation_cluster", "innovation_unit"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(str(value) for value in values))
        n = len(self.rows)
        if n == 0 or any(len(row) != len(self.columns) for row in self.rows):
            refuse(_INVALID, reason="result rows and columns must align")
        if any(
            not isinstance(value, (str, int, float, bool)) for row in self.rows for value in row
        ):
            refuse(_INVALID, reason="result rows require immutable scalar values")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for row in self.rows
            for value in row
        ):
            refuse(_INVALID, reason="result rows require finite numeric values")
        if any(
            values is not None and len(values) != n
            for values in (self.innovation_cluster, self.innovation_unit)
        ):
            refuse(_INVALID, reason="innovation ancestry must align with rows")
        if any(len(v) != n for v in (self.y0, self.y1, self.conditional_tau, self.cluster_index)):
            refuse(
                _INVALID, reason="potential outcomes and cluster identities must align with rows"
            )
        if not all(math.isfinite(v) for v in (*self.y0, *self.y1, *self.conditional_tau)):
            refuse(_INVALID, reason="potential outcomes must be finite")
        if not all(math.isfinite(b - a) for a, b in zip(self.y0, self.y1, strict=True)):
            refuse(_INVALID, reason="potential outcome differences must be finite")
        if len(self.sizes) != self.scenario.n_clusters or any(n < 1 for n in self.sizes):
            refuse(_INVALID, reason="result sizes must describe every cluster")
        if any(g < 0 or g >= len(self.sizes) for g in self.cluster_index):
            refuse(_INVALID, reason="result cluster index is outside the roster")
        if tuple(np.bincount(self.cluster_index)) != self.sizes:
            refuse(_INVALID, reason="result sizes disagree with the roster")
        required = {"unit_id", "cluster_id", "group_id", "y", *self.scenario.covariates}
        if len(set(self.columns)) != len(self.columns) or not required <= set(self.columns):
            refuse(_INVALID, reason="result columns must contain unique required names")
        ui = self.columns.index("unit_id")
        if len({row[ui] for row in self.rows}) != n:
            refuse(_INVALID, reason="result unit identities must be unique")
        ci = self.columns.index("cluster_id")
        roster: dict[int, object] = {}
        for g, row in zip(self.cluster_index, self.rows, strict=True):
            if g in roster and roster[g] != row[ci]:
                refuse(_INVALID, reason="result cluster labels disagree with cluster indices")
            roster[g] = row[ci]
        if len(set(roster.values())) != len(self.sizes):
            refuse(_INVALID, reason="result cluster labels must distinguish every cluster")
        yi, di = self.columns.index("y"), self.columns.index("group_id")
        for row, a, b in zip(self.rows, self.y0, self.y1, strict=True):
            if row[di] not in ("control", "treatment") or row[yi] != (
                b if row[di] == "treatment" else a
            ):
                refuse(_INVALID, reason="observed outcome disagrees with potential outcomes")

    @property
    def table(self) -> pa.Table:
        return pa.table(dict(zip(self.columns, zip(*self.rows, strict=True), strict=True)))

    @property
    def potential_outcomes(self) -> Mapping[str, np.ndarray]:
        # Bytes-backed arrays cannot regain write permission via setflags.
        def array(values: Sequence[float]) -> np.ndarray:
            return np.frombuffer(np.asarray(values, dtype=float).tobytes(), dtype=float)

        return MappingProxyType(
            {"y0": array(self.y0), "y1": array(self.y1), "tau": array(self.tau)}
        )

    @property
    def tau(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.y0, self.y1, strict=True))

    def weights(self, weighting: Weight) -> np.ndarray:
        if weighting not in ("member_count", "equal"):
            refuse(_INVALID, reason="unknown target weighting")
        return (
            np.ones(len(self.rows))
            if weighting == "member_count"
            else 1 / np.asarray(self.sizes)[np.asarray(self.cluster_index)]
        )

    @property
    def truth(self) -> ClusteredCATETruth:
        return ClusteredCATETruth(
            member_ate=float(np.mean(self.tau)),
            equal_cluster_ate=float(np.average(self.tau, weights=self.weights("equal"))),
            conditional_member_ate=float(np.mean(self.conditional_tau)),
            conditional_equal_cluster_ate=float(
                np.average(self.conditional_tau, weights=self.weights("equal"))
            ),
            n_units=len(self.rows),
            n_clusters=len(self.sizes),
        )

    @property
    def assignment_support(self) -> Literal["both_arms", "control_only", "treatment_only"]:
        groups = set(self.table["group_id"].to_pylist())
        if len(groups) == 2:
            return "both_arms"
        return "control_only" if "control" in groups else "treatment_only"

    @property
    def covariates(self) -> dict[str, np.ndarray]:
        table = self.table
        return {name: np.asarray(table[name]) for name in self.scenario.covariates}


def child_rng(seed: int, stream: Role, replication: int, component: str) -> np.random.Generator:
    """Address a stream without consuming any sibling stream's state."""
    if stream not in ("training", "holdout", "oracle"):
        refuse(_INVALID, reason="stream must be training, holdout or oracle")
    if isinstance(replication, bool) or not isinstance(replication, int) or replication < 0:
        refuse(_INVALID, reason="replication must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        refuse(_INVALID, reason="seed must be a nonnegative integer")
    digest = hashlib.blake2b(f"{stream}/{component}".encode(), digest_size=16).digest()
    address = [int.from_bytes(digest[j : j + 4], "little") for j in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, replication, *address]))


def _innovation(rng: np.random.Generator, n: int, skew: float) -> np.ndarray:
    z = rng.normal(size=n)
    # Hermite orthogonality: Var[Z+s(Z²-1)] = 1+2s².
    return (z + skew * (z * z - 1)) / math.sqrt(1 + 2 * skew * skew)


def simulate_clustered_cate(
    scenario: ClusteredCATEScenario, *, stream: Role = "training", replication: int = 0
) -> ClusteredCATEResult:
    """Draw exactly once; never condition on arm support or fitted availability."""
    s = ClusteredCATEScenario.model_validate(scenario)
    if s.witness == "original_varying_noise":
        return _original_varying_noise(s, stream=stream, replication=replication)

    def rng(component: str) -> np.random.Generator:
        return child_rng(s.seed, stream, replication, component)

    k = s.n_clusters
    z = rng("population/signal").uniform(-1, 1, k)
    if s.member_counts is not None:
        sizes = np.array(s.member_counts)
    elif s.size_mode == "fixed":
        sizes = np.full(k, s.members_per_cluster)
    else:
        driver = z if s.size_mode == "informative" else rng("population/size").uniform(-1, 1, k)
        sizes = np.maximum(
            1, np.rint(s.members_per_cluster * np.exp(s.size_spread * driver))
        ).astype(int)
    shared_noise = _innovation(rng("outcome/cluster"), k, s.skew)
    cluster_assignment = rng("assignment/cluster").random(k) < s.treatment_ratio
    records: list[dict[str, object]] = []
    y0, y1, tau, indices, innovation_clusters, innovation_units = [], [], [], [], [], []
    for g, m in enumerate(sizes):
        # Per-cluster addresses keep common members stable when a sibling grows.
        xs = [rng(f"covariate/{g}/{j}").uniform(-1, 1, int(m)) for j in range(s.dimension)]
        if s.leverage and g == 0:
            xs[0][0] *= 1 + s.leverage
        noise = math.sqrt(s.icc) * shared_noise[g] + math.sqrt(1 - s.icc) * _innovation(
            rng(f"outcome/unit/{g}"), int(m), s.skew
        )
        x0 = xs[0] if xs else np.zeros(m)
        systematic = 1 + 0.8 * z[g] + 0.6 * x0
        effect = s.effect + s.heterogeneity * x0 + s.size_effect * z[g]
        a = systematic + s.outcome_scale * noise
        b = systematic + effect + s.outcome_scale * s.treated_noise_ratio * noise
        propensity = np.full(m, s.treatment_ratio)
        if s.assignment == "observational":
            logits = math.log(s.treatment_ratio / (1 - s.treatment_ratio)) + 0.6 * x0 + 0.6 * z[g]
            propensity = np.exp(-np.logaddexp(0, -logits))
        d = (
            np.full(m, cluster_assignment[g])
            if s.assignment == "cluster"
            else rng(f"assignment/unit/{g}").random(m) < propensity
        )
        label: object = (
            1009 * g - 71
            if s.id_style == "integer"
            else f"店舗/é:{g:04d} | α"
            if s.id_style == "unicode"
            else f"g{g}"
        )
        for j in range(m):
            for clone in range(s.clone_factor):
                row: dict[str, object] = {
                    "unit_id": f"{stream}/{replication}/{g}/{j}/{clone}",
                    "cluster_id": label,
                    "group_id": "treatment" if d[j] else "control",
                    "y": float(b[j] if d[j] else a[j]),
                    "z": float(z[g]),
                    **{f"x{v}": float(xs[v][j]) for v in range(s.dimension)},
                }
                records.append(row)
                y0.append(float(a[j]))
                y1.append(float(b[j]))
                tau.append(float(effect[j]))
                innovation_clusters.append(f"{stream}/{replication}/{g}")
                innovation_units.append(f"{stream}/{replication}/{g}/{j}")
                indices.append(g)
    if s.reverse_rows:
        records.reverse()
        for values in (y0, y1, tau, indices, innovation_clusters, innovation_units):
            values.reverse()
    columns = tuple(records[0])
    return ClusteredCATEResult(
        s,
        stream,
        replication,
        tuple(tuple(row[name] for name in columns) for row in records),
        columns,
        tuple(y0),
        tuple(y1),
        tuple(tau),
        tuple(indices),
        tuple(int(m) * s.clone_factor for m in sizes),
        innovation_cluster=tuple(innovation_clusters),
        innovation_unit=tuple(innovation_units),
    )


def clustered_source(population: ClusteredCATEResult):
    """Public dataframe adapter with the DGP's actual identification contract."""
    from increment.frame import from_unit_summary
    from increment.semantics.design import AdjustmentSet, Observational, Randomized

    s = population.scenario
    design = (
        Observational(control_group="control", adjustment=AdjustmentSet(covariates=s.adjustment))
        if s.assignment == "observational"
        else Randomized(
            control_group="control",
            allocation={"control": 1 - s.treatment_ratio, "treatment": s.treatment_ratio},
        )
    )
    return from_unit_summary(
        population.table,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics={"y": "mean"},
        design=design,
        cluster="cluster_id" if s.declare_clusters else None,
        intervention_grain=s.intervention_grain,
    )


def honest_clustered_population(
    scenario: ClusteredCATEScenario, *, replication: int = 0
) -> ClusteredCATEResult:
    """Supply the public hash split with independently drawn role populations.

    Whole clusters retain their own size, covariates, outcomes and assignment.
    No draw is rejected based on support. IID unit cells use the public unit
    split directly; its disjoint independent rows already form an honest split.
    """
    from increment.estimation.targeting import _holdout_mask

    training = simulate_clustered_cate(scenario, replication=replication)
    if not scenario.declare_clusters or scenario.witness != "random":
        return training
    holdout = simulate_clustered_cate(scenario, stream="holdout", replication=replication)
    training_ids = np.asarray(training.table["cluster_id"]).astype(str)
    held = _holdout_mask(np.asarray(training.table["unit_id"]), cluster_ids=training_ids)
    role_by_cluster = {g: bool(flag) for g, flag in zip(training.cluster_index, held, strict=True)}
    rows, y0, y1, tau, groups, sizes = [], [], [], [], [], []
    innovation_clusters, innovation_units = [], []
    for g in range(scenario.n_clusters):
        population = holdout if role_by_cluster[g] else training
        positions = np.flatnonzero(np.asarray(population.cluster_index) == g)
        if scenario.reverse_rows:
            positions = positions[::-1]
        sizes.append(len(positions))
        for i in positions:
            rows.append(population.rows[i])
            y0.append(population.y0[i])
            y1.append(population.y1[i])
            tau.append(population.conditional_tau[i])
            groups.append(g)
            if population.innovation_cluster is None or population.innovation_unit is None:
                refuse(_INVALID, reason="random DGP requires original innovation ancestry")
            innovation_clusters.append(population.innovation_cluster[i])
            innovation_units.append(population.innovation_unit[i])
    if scenario.reverse_rows:
        for values in (
            rows,
            y0,
            y1,
            tau,
            groups,
            innovation_clusters,
            innovation_units,
        ):
            values.reverse()
    return ClusteredCATEResult(
        scenario,
        "training",
        replication,
        tuple(rows),
        training.columns,
        tuple(y0),
        tuple(y1),
        tuple(tau),
        tuple(groups),
        tuple(sizes),
        innovation_cluster=tuple(innovation_clusters),
        innovation_unit=tuple(innovation_units),
    )


def policy_truth(
    rule: TargetingRule, oracle: ClusteredCATEResult, *, cost: float = 0
) -> PolicyTruth:
    """Predict on an independent finite batch, not exact population truth.

    This diagnostic must not be subtracted from a point for a different batch.
    The acceptance ledger uses evaluation_policy_truth on the actual roster.
    """
    from increment.estimation.targeting import TargetingRule

    if oracle.stream != "oracle":
        refuse(_INVALID, reason="policy truth requires the independent oracle stream")
    if not math.isfinite(cost):
        refuse(_INVALID, reason="policy cost must be finite")
    frozen = TargetingRule.model_validate_json(rule.model_dump_json())
    ids = np.asarray(oracle.table["cluster_id"]) if oracle.scenario.declare_clusters else None
    actions = frozen.predict(oracle.covariates, cluster_ids=ids)
    return _action_truth(actions, oracle, cost=cost, rule_json=frozen.model_dump_json())


def _action_truth(
    actions: np.ndarray,
    oracle: ClusteredCATEResult,
    *,
    cost: float,
    rule_json: str,
    roster: np.ndarray | None = None,
    evaluation_weights: np.ndarray | None = None,
    evaluation_weighting: Weight | None = None,
) -> PolicyTruth:
    if roster is None:
        roster = np.arange(len(oracle.rows))
    elif roster.dtype == bool:
        roster = np.flatnonzero(roster)
    local_actions = actions if actions.size == roster.size else actions[roster]
    tau = np.asarray(oracle.tau)[roster]
    outcomes = np.where(local_actions, np.asarray(oracle.y1)[roster], np.asarray(oracle.y0)[roster])
    _, inverse, counts = np.unique(
        np.asarray(oracle.cluster_index)[roster], return_inverse=True, return_counts=True
    )
    result: dict[str, Any] = {}
    for prefix, weight in (("member", "member_count"), ("equal_cluster", "equal")):
        w = np.ones(roster.size) if weight == "member_count" else 1.0 / counts[inverse]
        if evaluation_weights is not None and weight == evaluation_weighting:
            w = evaluation_weights
        selected = local_actions
        result[f"{prefix}_effect"] = (
            float(np.average(tau[selected], weights=w[selected])) if selected.any() else None
        )
        result[f"{prefix}_share"] = float(np.average(selected, weights=w))
        result[f"{prefix}_net_benefit"] = float(np.average(selected * (tau - cost), weights=w))
        result[f"{prefix}_outcome"] = float(np.average(outcomes, weights=w))
    return PolicyTruth(
        **result,
        actions=tuple(bool(a) for a in local_actions),
        rule_json=rule_json,
        unavailable_reason=None
        if local_actions.any()
        else "simulate.cluster_dgp.empty_oracle_policy",
    )


@dataclass(frozen=True, slots=True)
class _EvaluationRoster:
    """A view of retained rows, without pretending it is a full DGP draw."""

    population: ClusteredCATEResult
    indices: tuple[int, ...]
    actions: tuple[bool, ...]
    base_weights: tuple[float, ...]
    nuisances: tuple[tuple[float, ...], ...] | None
    cluster_ids: tuple[str, ...] | None
    scores: tuple[float, ...]

    @property
    def scenario(self) -> ClusteredCATEScenario:
        return self.population.scenario

    @property
    def rows(self) -> tuple[tuple[object, ...], ...]:
        return tuple(self.population.rows[i] for i in self.indices)

    @property
    def table(self) -> pa.Table:
        return self.population.table.take(pa.array(self.indices))

    @property
    def tau(self) -> tuple[float, ...]:
        p = self.population
        return tuple(p.y1[i] - p.y0[i] for i in self.indices)

    @property
    def conditional_tau(self) -> tuple[float, ...]:
        return tuple(self.population.conditional_tau[i] for i in self.indices)

    @property
    def innovation_cluster(self) -> tuple[str, ...] | None:
        keys = self.population.innovation_cluster
        return None if keys is None else tuple(keys[i] for i in self.indices)

    @property
    def innovation_unit(self) -> tuple[str, ...] | None:
        keys = self.population.innovation_unit
        return None if keys is None else tuple(keys[i] for i in self.indices)


def _evaluation_roster(rule: TargetingRule, population: ClusteredCATEResult) -> _EvaluationRoster:
    snapshot = rule.validation.evaluation_population
    if snapshot is None:
        refuse(_OVERLAP_ROSTER)
    positions = {str(value): i for i, value in enumerate(population.table["unit_id"])}
    try:
        indices = tuple(positions[value] for value in snapshot.unit_ids)
    except KeyError:
        refuse(_OVERLAP_ROSTER)
    ids = snapshot.cluster_ids
    if ids is not None and ids != tuple(
        str(population.rows[i][population.columns.index("cluster_id")]) for i in indices
    ):
        refuse(_INVALID, reason="evaluation cluster identities disagree with the population")
    if (ids is not None) != population.scenario.declare_clusters:
        refuse(_INVALID, reason="evaluation cluster declaration disagrees with the population")
    if ids is None or rule.cluster_weight == "member_count":
        expected_weights = tuple(1.0 for _ in indices)
    else:
        _, inverse, counts = np.unique(np.asarray(ids), return_inverse=True, return_counts=True)
        expected_weights = tuple(float(1.0 / counts[i]) for i in inverse)
    if tuple(float(value) for value in snapshot.base_weights) != expected_weights:
        refuse(_INVALID, reason="evaluation snapshot weights disagree with retained roster")
    cols = {key: values[list(indices)] for key, values in population.covariates.items()}
    actions = rule.predict(cols, cluster_ids=None if ids is None else np.asarray(ids))
    scores = rule.score_state.score(cols)
    return _EvaluationRoster(
        population,
        indices,
        tuple(bool(a) for a in actions),
        snapshot.base_weights,
        snapshot.nuisance_predictions,
        ids,
        tuple(float(value) for value in scores),
    )


def evaluation_policy_truth(
    rule: TargetingRule,
    population: ClusteredCATEResult,
) -> tuple[PolicyTruth, float]:
    """Exact causal target and actions in the public snapshot's retained row order."""
    roster = _evaluation_roster(rule, population)
    roster_indices = np.asarray(roster.indices, dtype=int)
    truth = _action_truth(
        np.asarray(roster.actions),
        population,
        cost=0,
        rule_json=rule.model_dump_json(),
        roster=roster_indices,
        evaluation_weights=np.asarray(roster.base_weights, dtype=float),
        evaluation_weighting=rule.cluster_weight,
    )
    average = float(
        np.average(
            np.asarray(population.tau)[roster_indices],
            weights=np.asarray(roster.base_weights, dtype=float),
        )
    )
    return truth, average


@dataclass(frozen=True, slots=True)
class AffinePointReconstruction:
    """Independent affine reconstruction of one retained policy point."""

    statistic: str
    d0: float
    coefficients: tuple[float, ...]
    target_coefficients: tuple[float, ...]
    target: float
    reconstructed: float
    error: float
    bias: float | None
    innovation_coefficients: tuple[tuple[str, float], ...]
    q_h: float | None
    m4_h: float | None
    selected_fraction: float
    roundoff_bound: float


def reconstruct_affine_point(  # noqa: PLR0915
    population: ClusteredCATEResult | _EvaluationRoster,
    actions: Sequence[bool] | np.ndarray,
    *,
    statistic: Literal["effect", "uplift"],
    observed: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    nuisances: tuple[Sequence[float], ...] | None = None,
) -> AffinePointReconstruction:
    """Reconstruct effect/uplift and its DGP innovation moments on a roster.

    Randomized points use the production difference-in-means and centered-IPW
    formulas.  Observational points require the producer-frozen
    ``(propensity, m1, m0)`` predictions.  Empty selected denominators refuse
    rather than manufacturing a zero point; all-selected uplift is exactly
    zero by construction.
    """
    from increment.simulate._cluster_reconstruction import point_roundoff_bound

    if statistic not in ("effect", "uplift"):
        refuse(_INVALID, reason="unknown policy point statistic")
    if population.scenario.witness == "random" and (
        population.innovation_cluster is None or population.innovation_unit is None
    ):
        refuse(_INVALID, reason="random point reconstruction requires original innovation ancestry")
    if population.scenario.assignment != "observational" and nuisances is not None:
        refuse(_INVALID, reason="randomized reconstruction cannot use observational nuisances")
    n = len(population.rows)
    action = np.asarray(actions)
    if action.shape != (n,) or action.dtype.kind != "b":
        refuse(_INVALID, reason="point actions must be boolean and roster-aligned")
    w = np.ones(n) if weights is None else _point_vector(weights, n, "weights")
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        refuse(_INVALID, reason="point weights must be finite, positive and roster-aligned")
    selected = action
    total = math.fsum(w)
    if not math.isfinite(total):
        refuse(_INVALID, reason="point weights must have finite total mass")
    q = w / total
    selected_mass = float(np.dot(q, selected))
    if selected_mass <= 0:
        refuse(_INVALID, reason="empty selected denominator")
    v = q * selected / selected_mass
    h = v if statistic == "effect" else v - q
    ids = np.asarray(population.table["group_id"]).astype(str)
    d = (ids == "treatment").astype(float)
    y = _point_vector(population.table["y"] if observed is None else observed, n, "outcomes")
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        refuse(_INVALID, reason="observed outcomes must align and be finite")
    if population.scenario.assignment == "observational":
        if nuisances is None or len(nuisances) != 3:
            refuse(_INVALID, reason="observational reconstruction requires frozen nuisances")
        e, m1, m0 = (_point_vector(v, n, "nuisances") for v in nuisances)
        if not all(np.all(np.isfinite(v)) for v in (e, m1, m0)) or np.any((e <= 0) | (e >= 1)):
            refuse(_INVALID, reason="frozen nuisances must be finite and overlap-supported")
        k = d / e - (1 - d) / (1 - e)
        r = m1 - m0 - d * m1 / e + (1 - d) * m0 / (1 - e)
        ell = h * k
        d0 = float(np.dot(h, r))
    elif statistic == "effect":
        at = w * selected * d
        ac = w * selected * (1 - d)
        if at.sum() <= 0 or ac.sum() <= 0:
            refuse(_INVALID, reason="randomized point arm denominator is empty")
        ell = at / at.sum() - ac / ac.sum()
        d0 = 0.0
    else:
        p = float(np.dot(q, d))
        if p <= 0 or p >= 1:
            refuse(_INVALID, reason="centered-IPW arm denominator is empty")
        k = (d - p) / (p * (1 - p))
        ell = h * k - q * float(np.dot(h, k))
        d0 = 0.0
    if statistic == "uplift" and np.all(selected):
        h, ell, d0 = np.zeros(n), np.zeros(n), 0.0
    reconstructed = float(d0 + math.fsum(float(a * b) for a, b in zip(ell, y, strict=True)))
    target = float(np.dot(h, population.tau))
    bound = point_roundoff_bound(
        y,
        d,
        w,
        selected,
        statistic,
        nuisances,
        reconstructed,
        clustered=population.scenario.declare_clusters,
        scores=np.asarray(population.scores) if isinstance(population, _EvaluationRoster) else None,
    )
    if population.scenario.witness == "original_varying_noise":
        return AffinePointReconstruction(
            statistic,
            d0,
            tuple(float(v) for v in ell),
            tuple(float(v) for v in h),
            target,
            reconstructed,
            reconstructed - target,
            None,
            (),
            None,
            None,
            selected_mass,
            bound,
        )
    s = population.scenario
    cols = population.table
    z = np.asarray(cols["z"], dtype=float)
    x = np.asarray(cols["x0"], dtype=float) if s.dimension else np.zeros(n)
    tau = np.asarray(population.conditional_tau, dtype=float)
    mu = 1 + 0.8 * z + 0.6 * x + d * tau
    sigma = s.outcome_scale * np.where(d == 1, s.treated_noise_ratio, 1.0)
    delta_sigma = s.outcome_scale * (s.treated_noise_ratio - 1.0)
    b = float(d0 + np.dot(ell, mu) - np.dot(h, tau))
    di = ell * sigma - h * delta_sigma
    cluster_keys, unit_keys = population.innovation_cluster, population.innovation_unit
    if cluster_keys is None or unit_keys is None:
        refuse(_INVALID, reason="random point reconstruction requires original innovation ancestry")
    terms: dict[str, list[float]] = {}
    for key, value in zip(cluster_keys, di, strict=True):
        terms.setdefault(f"cluster:{key}", []).append(float(value))
    for key, value in zip(unit_keys, di, strict=True):
        terms.setdefault(f"unit:{key}", []).append(float(value))
    values = tuple(
        (key, math.fsum(terms[key]) * math.sqrt(s.icc if key.startswith("cluster:") else 1 - s.icc))
        for key in sorted(terms)
    )
    variance = float(math.fsum(value * value for _, value in values))
    skew3 = (6 * s.skew + 8 * s.skew**3) / (1 + 2 * s.skew**2) ** 1.5
    skew4 = (3 + 60 * s.skew**2 + 60 * s.skew**4) / (1 + 2 * s.skew**2) ** 2
    cubes = math.fsum(value**3 for _, value in values)
    fourths = math.fsum(value**4 for _, value in values)
    m4 = (
        b**4 + 6 * b**2 * variance + 4 * b * skew3 * cubes + 3 * variance**2 + (skew4 - 3) * fourths
    )
    return AffinePointReconstruction(
        statistic,
        float(d0),
        tuple(float(value) for value in ell),
        tuple(float(value) for value in h),
        target,
        reconstructed,
        reconstructed - target,
        b,
        values,
        b * b + variance,
        m4,
        float(np.dot(q, selected)),
        bound,
    )


def _point_vector(values: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        refuse(_INVALID, reason=f"point {name} must be finite numeric values aligned to the roster")
    return array.astype(float)


def check_policy_reconstruction(
    rule: TargetingRule,
    population: ClusteredCATEResult,
) -> tuple[AffinePointReconstruction, ...]:
    """Fail the campaign on an independent point or reporting disagreement.

    This is an arithmetic check only. PointAccuracy and the unresolved joint
    reporting/conditional-risk obligations remain separate release gates.
    """
    from increment.simulate._cluster_reconstruction import reconstruct_reporting

    roster = _evaluation_roster(rule, population)
    snapshot = rule.validation.evaluation_population
    assert snapshot is not None
    expected_method = (
        "frozen_dr" if population.scenario.assignment == "observational" else "centered_ipw"
    )
    if snapshot.score_method != expected_method:
        refuse(_INVALID, reason="point reconstruction requires the declared IPW or frozen DR score")
    if roster.nuisances is not None:
        if len(roster.nuisances) != 3:
            refuse(_INVALID, reason="point reconstruction requires three frozen nuisance vectors")
        e, _, _ = (
            _point_vector(values, len(roster.indices), "nuisances") for values in roster.nuisances
        )
        if np.any((e <= 0) | (e >= 1)):
            refuse(_INVALID, reason="frozen nuisance propensities must have overlap")
    gate = reconstruct_reporting(rule, roster)
    if (
        gate.rank_passed != rule.validation.passed
        or gate.recommendation != rule.recommendation
        or gate.reported != (rule.policy_value is not None)
        or gate.reported != (rule.uplift_vs_average is not None)
        or gate.rank_reason != rule.validation.autoc.unavailable_reason
    ):
        refuse(
            _INVALID, reason="independent policy reporting predicate disagrees with public result"
        )
    if not any(roster.actions):
        if rule.unavailable_reason != "estimation.targeting.empty_group":
            refuse(_INVALID, reason="empty policy lost its unavailable reason")
        return ()
    points = []
    statistics: tuple[Literal["effect", "uplift"], ...] = ("effect", "uplift")
    groups = np.asarray(roster.table["group_id"])[np.asarray(roster.actions)]
    for statistic, public in zip(
        statistics, (rule.policy_value, rule.uplift_vs_average), strict=True
    ):
        if (
            statistic == "effect"
            and population.scenario.assignment != "observational"
            and len(set(groups)) < 2
        ):
            continue
        point = reconstruct_affine_point(
            roster,
            roster.actions,
            statistic=statistic,
            weights=roster.base_weights,
            nuisances=roster.nuisances,
        )
        if public is not None and (
            not math.isfinite(public.value)
            or abs(public.value - point.reconstructed) > point.roundoff_bound
        ):
            refuse(
                _INVALID,
                reason=f"public policy {statistic} disagrees with independent affine point",
            )
        points.append(point)
    return tuple(points)


@dataclass(frozen=True, slots=True)
class PointAccuracy:
    """Prospective scientific magnitudes; no defaults inferred from outcomes.

    Accuracy is P(|reported point - finite-batch target| > tolerance | point
    reported). Availability is P(point reported) over ALL attempted calls.
    Neither quantity asserts nominal post-gate confidence intervals.
    """

    cell: str
    statistic: str
    absolute_tolerance: float
    minimum_availability: float

    def __post_init__(self) -> None:
        if not self.cell or self.statistic not in (
            "policy.effect",
            "policy.uplift",
            "selection.effect",
            "selection.uplift",
        ):
            refuse(_INVALID, reason="point accuracy must identify a policy statistic")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance <= 0:
            refuse(_INVALID, reason="point tolerance must be finite and positive")
        if not 0 < self.minimum_availability < 1:
            refuse(_INVALID, reason="point availability must be prospectively in (0,1)")


@dataclass(frozen=True, slots=True)
class StatisticSpec:
    name: str
    estimand: str
    nominal_error: float | None
    gate: Literal[
        "coverage", "size", "availability", "descriptive", "point_accuracy", "point_availability"
    ]
    accounting: tuple[str, ...] = (
        "attempted",
        "point_estimable",
        "interval_estimable",
        "excluded",
        "failed",
        "failure_reasons",
        "exclusion_reasons",
        "interval_unavailable_reasons",
        "bias",
        "bias_mcse",
        "coverage_conditional",
        "coverage_conditional_mcse",
        "coverage_unconditional",
        "coverage_unconditional_mcse",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "accounting", tuple(self.accounting))
        if (
            not self.name
            or not self.estimand
            or self.gate
            not in (
                "coverage",
                "size",
                "availability",
                "descriptive",
                "point_accuracy",
                "point_availability",
            )
        ):
            refuse(_INVALID, reason="statistic requires a name, estimand and recognized gate")
        if self.nominal_error is not None and not 0 < self.nominal_error < 1:
            refuse(_INVALID, reason="nominal error must be in (0,1)")


@dataclass(frozen=True, slots=True)
class ClusteredCell:
    name: str
    scenario: ClusteredCATEScenario
    weighting: Weight
    statistics: tuple[StatisticSpec, ...]
    purpose: Literal["calibration", "availability", "misspecification", "invariance"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", tuple(self.statistics))
        if not self.name or self.weighting not in ("member_count", "equal"):
            refuse(_INVALID, reason="cell requires a name and valid weighting")
        if self.weighting == "equal" and not self.scenario.declare_clusters:
            refuse(_INVALID, reason="equal weighting requires declared clusters")
        if self.purpose not in ("calibration", "availability", "misspecification", "invariance"):
            refuse(_INVALID, reason="unknown cell purpose")
        if self.statistics != _statistics(self.scenario, self.weighting):
            refuse(_INVALID, reason="cell statistic inventory must match its executable scenario")

    @property
    def validation_support_ceiling(self) -> float | None:
        """Necessary assignment support, not a claim of interval availability."""
        return validation_support_ceiling(self.scenario)

    @property
    def applicability_reason(self) -> str | None:
        ceiling = self.validation_support_ceiling
        if ceiling is not None and ceiling < 0.945:
            return "simulate.cluster_dgp.unconditional_coverage_exceeds_assignment_support"
        if self.scenario.assignment == "unit" and self.scenario.declare_clusters:
            return "simulate.cluster_dgp.randomized_mixed_clusters_unsupported"
        if self.purpose == "availability":
            return "simulate.cluster_dgp.small_sample_availability"
        if self.scenario.omit_confounder:
            return "simulate.cluster_dgp.omitted_observed_confounder"
        if self.scenario.witness != "random":
            return "simulate.cluster_dgp.deterministic_archived_assignment"
        return None


def validation_support_ceiling(scenario: ClusteredCATEScenario) -> float | None:
    """Upper bound P(both arms in each hash half) under Bernoulli clusters.

    Conditional on the ID partition, the two probabilities multiply. Requiring
    finite sandwich directions or usable bootstrap draws can only lower this
    ceiling. This is a necessary condition, never an availability estimator.
    """
    from increment.estimation.targeting import _holdout_mask

    if scenario.assignment != "cluster" or scenario.witness != "random":
        return None
    labels = np.array(
        [
            str(1009 * g - 71)
            if scenario.id_style == "integer"
            else f"店舗/é:{g:04d} | α"
            if scenario.id_style == "unicode"
            else f"g{g}"
            for g in range(scenario.n_clusters)
        ]
    )
    held = int(_holdout_mask(labels, cluster_ids=labels).sum())
    p = scenario.treatment_ratio

    def both(k: int) -> float:
        if k < 2:
            return 0.0
        return max(0.0, -math.expm1(k * math.log(p)) - math.exp(k * math.log1p(-p)))

    return both(held) * both(scenario.n_clusters - held)


def _statistics(s: ClusteredCATEScenario, weight: Weight) -> tuple[StatisticSpec, ...]:
    target = "member" if weight == "member_count" else "equal_cluster"
    specs = []
    if s.assignment != "observational":
        specs.append(StatisticSpec("fit.ate", f"{target} finite-population ATE", 0.05, "coverage"))
    specs.append(StatisticSpec("validation.ate", f"{target} holdout ATE", 0.05, "coverage"))
    for name in ("autoc", "qini"):
        interval_gate = "coverage" if s.declare_clusters and s.interactions else "availability"
        specs.append(
            StatisticSpec(f"validation.{name}", f"{target} holdout {name}", 0.05, interval_gate)
        )
        size_gate = (
            "size" if s.interactions and s.heterogeneity == s.size_effect == 0 else "descriptive"
        )
        specs.append(
            StatisticSpec(f"validation.{name}.reject", "one-sided rejection", 0.05, size_gate)
        )
    for group in (1, 2):
        gate = "availability" if group == 2 and not s.interactions else "coverage"
        specs.append(
            StatisticSpec(f"validation.group{group}", f"{target} GATES {group}", 0.025, gate)
        )
    for name in ("policy", "selection"):
        for quantity in ("effect", "uplift"):
            key = f"{name}.{quantity}"
            specs.extend(
                (
                    StatisticSpec(
                        key,
                        f"{target} exact evaluation-batch {quantity}; gate-conditioned",
                        None,
                        "descriptive",
                    ),
                    StatisticSpec(
                        f"{key}.accuracy",
                        "excess absolute point error conditional on reporting",
                        0.05,
                        "point_accuracy" if s.interactions else "availability",
                    ),
                    StatisticSpec(
                        f"{key}.unavailable",
                        "unavailable point / attempted calls",
                        None,
                        "point_availability" if s.interactions else "availability",
                    ),
                )
            )
    ceiling = validation_support_ceiling(s)
    if ceiling is not None and ceiling < 0.945:
        # The hash-half ceiling says nothing about full fits or stratified selection.
        specs = [
            replace(spec, gate="availability")
            if (spec.name.startswith("validation.") and spec.gate in ("coverage", "size"))
            else spec
            for spec in specs
        ]
    return tuple(specs)


def cell_inventory() -> tuple[ClusteredCell, ...]:
    """Freeze concrete cells before execution; a covering design, not a full grid.

    Small K x skew x asymmetric assignment, ICC x roster size, informative size
    x weighting x deployment, and the original varying-noise replay are retained.
    Shared-schedule and existing-estimator tables belong to their own manifests.
    """
    cells: list[ClusteredCell] = []

    def add(name: str, *, purpose="calibration", weights=("member_count", "equal"), **controls):
        scenario = ClusteredCATEScenario(**controls)
        for weight in weights:
            cells.append(
                ClusteredCell(
                    f"{name}/{weight}",
                    scenario,
                    weight,
                    _statistics(scenario, weight),
                    purpose,
                )
            )

    for icc, members in product((0.0, 0.2, 0.5), (5, 20, 100)):
        add(f"icc={icc}/m={members}", icc=icc, members_per_cluster=members, n_clusters=200)
    for k, p, skew in product((4, 10, 20, 40, 200), (0.5, 0.75, 0.9), (0.0, 1.0)):
        add(
            f"K={k}/p={p}/skew={skew}",
            n_clusters=k,
            treatment_ratio=p,
            skew=skew,
            purpose="availability" if k < 40 else "calibration",
        )
    for mode, grain in product(("variable", "informative"), ("unit", "cluster")):
        add(f"{mode}/{grain}", size_mode=mode, intervention_grain=grain, n_clusters=200)
    for dimension, leverage, p in product((0, 2, 10), (0.0, 8.0), (0.5, 0.9)):
        if dimension == 0 and leverage:
            continue
        add(
            f"dimension={dimension}/leverage={leverage}/p={p}",
            dimension=dimension,
            leverage=leverage,
            treatment_ratio=p,
            n_clusters=200,
        )
    for grain in ("unit", "cluster"):
        add(
            f"observational/mixed/{grain}",
            assignment="observational",
            intervention_grain=grain,
            size_mode="informative",
            n_clusters=200,
        )
    add(
        "observational/omitted-z",
        assignment="observational",
        omit_confounder=True,
        size_mode="informative",
        n_clusters=200,
        purpose="misspecification",
    )
    add(
        "unclustered/unit",
        assignment="unit",
        declare_clusters=False,
        icc=0,
        weights=("member_count",),
        size_effect=0,
    )
    add("randomized/mixed-refusal", assignment="unit", purpose="availability")
    for effect in (0.0, 0.5):
        add(
            f"homogeneous/effect={effect}",
            effect=effect,
            heterogeneity=0,
            size_effect=0,
            n_clusters=200,
        )
    for p, ratio in product((0.5, 0.75, 0.9), (0.25, 4.0)):
        add(
            f"unequal-arm-noise/p={p}/ratio={ratio}",
            treatment_ratio=p,
            treated_noise_ratio=ratio,
            n_clusters=200,
        )
    for style, reverse, clones in product(("plain", "unicode", "integer"), (False, True), (1, 5)):
        add(
            f"identity={style}/reverse={reverse}/clones={clones}",
            id_style=style,
            reverse_rows=reverse,
            clone_factor=clones,
            purpose="invariance",
        )
    for k, members, grain in product((16, 80), (5, 100), ("unit", "cluster")):
        add(
            f"original-varying-noise/K={k}/m={members}/{grain}",
            n_clusters=k,
            members_per_cluster=members,
            dimension=1,
            witness="original_varying_noise",
            intervention_grain=grain,
            purpose="misspecification",
        )
    return tuple(cells)


# The useful-power gate accepts at 0.80. Sizing targets a strictly higher power so
# the acceptance buffer is a declared design constant, not an integer-rounding residue.
_SIZING_ACCEPTED_POWER = 0.8
_SIZING_TARGET_POWER = 0.85


@dataclass(frozen=True, slots=True)
class ClusteredSizingCell:
    name: str
    allocation: float
    skew: float
    heterogeneity: float
    relative_effect: float
    seed: int
    statistics: tuple[str, ...] = ("rejection_agreement", "point_availability", "useful_power")

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", tuple(self.statistics))
        if not self.name or not 0 < self.allocation < 1:
            refuse(_INVALID, reason="sizing cell requires a name and allocation in (0,1)")
        if (
            not 0 <= self.skew <= 3
            or not math.isfinite(self.heterogeneity)
            or self.heterogeneity < 0
            or not math.isfinite(self.relative_effect)
            or self.relative_effect <= 0
        ):
            refuse(
                _INVALID,
                reason="sizing controls require finite nonnegative heterogeneity and positive effect",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            refuse(_INVALID, reason="sizing seed must be a nonnegative integer")
        if self.statistics != ("rejection_agreement", "point_availability", "useful_power"):
            refuse(_INVALID, reason="sizing statistics must include all acceptance gates")


def clustered_sizing_inventory() -> tuple[ClusteredSizingCell, ...]:
    """Actual rejection at public sizing, crossing imbalance, skew and effects."""
    return tuple(
        ClusteredSizingCell(
            f"sizing/p={p}/skew={skew}/heterogeneity={h}/effect={effect}",
            p,
            skew,
            h,
            effect,
            20260920 + index,
        )
        for index, (p, skew, h, effect) in enumerate(
            product(
                (0.5, 0.75, 0.9),
                (0.0, 1.0),
                (0.0, 0.5),
                (0.1, 0.2),
            )
        )
    )


def _clustered_sizing_plan(cell: ClusteredSizingCell) -> PowerResult:
    from increment.decision import FixedInference
    from increment.estimation.arm_contract import (
        AnalysisAxes,
        ArmPlanningProcedure,
        FamilyPolicy,
        MetricCapabilities,
        PlanningFamilyExpansion,
        RelativeDecisionPolicy,
    )
    from increment.power import Baseline, PowerDesign, required_sample_size
    from increment.semantics.assignment import ParallelAssignment
    from increment.semantics.models import MethodSpec

    # Y(a)=10(1+a*delta)+U+V+(1-2a)hZ; U,V have variance one.
    # Z is cluster-shared Uniform(-1,1), so both arm variances match.
    variance = 2 + cell.heterogeneity**2 / 3
    baseline = Baseline(
        mean=10,
        var=variance,
        avg_cluster_size=5,
        cluster_icc=(1 + cell.heterogeneity**2 / 3) / variance,
    )
    procedure = ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=True,
            population="assigned",
            variance_adjustment="factor_absorption",
        ),
        dependence="cluster",
        inference=FixedInference(),
        estimand="mean",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ),
        family_expansion=PlanningFamilyExpansion(family_size=1),
        decision_method=MethodSpec(name="unadjusted", variance_reduction="none"),
        sensitivity_methods=(),
        prior_present=False,
    )
    return required_sample_size(
        cell.relative_effect,
        baseline,
        procedure,
        PowerDesign(allocation=cell.allocation, power=_SIZING_TARGET_POWER),
    )


@dataclass(frozen=True, slots=True)
class ClusteredRegistration:
    """Main supplies globally allocated repetitions and bootstrap work BEFORE runs."""

    cells: tuple[ClusteredCell, ...]
    replications: int
    bootstrap_repetitions: int
    bootstrap_seed: int
    selection_seed: int
    global_gated_statistics: int | None = None
    point_accuracy: tuple[PointAccuracy, ...] = ()
    family_mc_error: float = 0.01
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "point_accuracy", tuple(self.point_accuracy))
        if not math.isfinite(self.family_mc_error) or not 0 < self.family_mc_error <= 0.01:
            refuse(_INVALID, reason="MC family reservation must be in (0,.01]")
        for name in ("replications", "bootstrap_repetitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                refuse(_INVALID, reason=f"{name} must be an integer >=2")
        if not self.cells or len({c.name for c in self.cells}) != len(self.cells):
            refuse(_INVALID, reason="registration requires unique nonempty cell names")
        for seed in (self.bootstrap_seed, self.selection_seed):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                refuse(_INVALID, reason="registered seeds must be nonnegative integers")
        declared = {(p.cell, p.statistic) for p in self.point_accuracy}
        required = {
            (c.name, stat.name.removesuffix(".accuracy"))
            for c in self.cells
            if c.purpose == "calibration"
            for stat in c.statistics
            if stat.gate == "point_accuracy"
        }
        if len(declared) != len(self.point_accuracy) or not declared <= required:
            refuse(
                _INVALID,
                reason="point accuracy declarations must uniquely match calibration statistics",
            )
        if self.global_gated_statistics is not None:
            count = self.global_gated_statistics
            local = sum(len(c.statistics) for c in clustered_sizing_inventory()) + sum(
                stat.gate in ("coverage", "size", "point_accuracy", "point_availability")
                for cell in self.cells
                if cell.purpose == "calibration"
                for stat in cell.statistics
            )
            if isinstance(count, bool) or not isinstance(count, int) or count < max(1, local):
                refuse(
                    _INVALID, reason="global gated-statistic count must include every local gate"
                )
            for _, margin, limit, _ in self.prospective_precision:
                if margin > limit:
                    refuse(
                        _INVALID,
                        reason="registered repetitions do not meet prospective MC precision",
                    )
            if declared != required:
                refuse(
                    _INVALID,
                    reason="release requires prospective point accuracy and nonvacuity magnitudes",
                )
        object.__setattr__(self, "_fingerprint", self._compute_fingerprint())

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        import json

        payload = {
            "cells": [
                {
                    "name": c.name,
                    "scenario": c.scenario.model_dump(mode="json"),
                    "weighting": c.weighting,
                    "purpose": c.purpose,
                    "support_ceiling": c.validation_support_ceiling,
                    "applicability_reason": c.applicability_reason,
                    "statistics": [asdict(s) for s in c.statistics],
                }
                for c in self.cells
            ],
            "replications": self.replications,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "selection_seed": self.selection_seed,
            "policy_estimand": "exact realized evaluation batch; original roster; oracle MC error=0",
            "point_accuracy": [asdict(p) for p in self.point_accuracy],
            "evidence_dependencies": self.evidence_dependencies,
            "sizing": [asdict(c) for c in clustered_sizing_inventory()],
            "witnesses": self.witness_inventory,
            "rng_scheme": "SeedSequence(seed,replication,blake2b128(role/component))/v1",
            "global_gated_statistics": self.global_gated_statistics,
            "scientific_delta": "min(.005,.1*q)",
            "mc_margin": "min(.0025,delta/2)",
            "sizing_mc_margin": "min(.0025,nextafter(plan.power-.8,-inf))",
            "family_mc_error": self.family_mc_error,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def seed_inventory(self) -> tuple[tuple[str, int, range], ...]:
        """Exact immutable replication schedules without expanding millions of rows."""
        return tuple((c.name, c.scenario.seed, range(self.replications)) for c in self.cells)

    @property
    def sizing_seed_inventory(self) -> tuple[tuple[str, int, range], ...]:
        return tuple(
            (c.name, c.seed, range(self.replications)) for c in clustered_sizing_inventory()
        )

    @property
    def evidence_dependencies(self) -> tuple[tuple[str, str, str], ...]:
        """Existing obligations, consumed in full by the release assertion."""
        return (
            (
                "I13",
                "tests.estimation._i13_manifest.ACCEPTANCE_MANIFEST",
                "tests.estimation._i13_calibration.R03Result.r03_gate",
            ),
            (
                "I14",
                "tests._unit_cycle_design.CELLS",
                "tests.simulate.test_unit_cycle_calibration.calibrate_cell",
            ),
            ("C12-sizing", "clustered_sizing_inventory", "evaluate_clustered_sizing"),
        )

    @property
    def prospective_precision(self) -> tuple[tuple[float, float, float, float], ...]:
        """(q, planned MC margin, allowed margin, two-point-defect detection).

        Bonferroni allocates eta=reserved_error/(2M) from Main's family. Exact
        beta upper endpoints give the rate bound; binomial survival gives the
        probability of failing acceptance at error q+.02. No observed rates
        enter these calculations. An unallocated registration is only a pilot.
        """
        from scipy.stats import beta, binom

        if self.global_gated_statistics is None:
            return ()
        eta = self.family_mc_error / (2 * self.global_gated_statistics)
        n = self.replications
        result = []
        rates = sorted(
            {
                stat.nominal_error
                for cell in self.cells
                if cell.purpose == "calibration"
                for stat in cell.statistics
                if stat.gate in ("coverage", "size") and stat.nominal_error is not None
            }
            | {0.05 for _ in self.point_accuracy}
            | {1 - p.minimum_availability for p in self.point_accuracy}
        )
        for q in rates:
            delta = min(0.005, 0.1 * q)

            def upper(errors: int) -> float:
                return 1.0 if errors == n else float(beta.isf(eta, errors + 1, n - errors))

            errors = min(n, math.ceil((q + delta) * n))
            margin = upper(errors) - errors / n
            lo, hi = -1, n
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if upper(mid) <= q + delta:
                    lo = mid
                else:
                    hi = mid
            detection = float(binom.sf(lo, n, min(1.0, q + 0.02)))
            result.append((q, margin, min(0.0025, delta / 2), detection))
        # A random reporting denominator cannot be replaced with attempted R.
        # Hoeffding bounds the chance of fewer than n_min reports by eta.
        for p in self.point_accuracy:
            n_min = math.floor(n * p.minimum_availability - math.sqrt(n * math.log(1 / eta) / 2))
            if n_min < 2:
                result.append((0.05, math.inf, 0.0025, 0.0))
                continue
            errors = min(n_min, math.ceil(0.055 * n_min))
            bound = 1.0 if errors == n_min else float(beta.isf(eta, errors + 1, n_min - errors))
            result.append(
                (0.05, bound - errors / n_min, 0.0025, float(binom.sf(errors, n_min, 0.07)))
            )
        # The useful-power lower bound must clear .8 at the actual sized design.
        margin = math.sqrt(math.log(1 / eta) / (2 * n))
        for cell in clustered_sizing_inventory():
            power = _clustered_sizing_plan(cell).power
            gap = max(0.0, math.nextafter(power - _SIZING_ACCEPTED_POWER, -math.inf))
            result.append((power, margin, min(0.0025, gap), 1 - eta))
        return tuple(result)

    @property
    def witness_inventory(self) -> tuple[str, ...]:
        """Executable deterministic tests accompanying the statistical cells."""
        return tuple(
            f"tests/simulate/test_clustered_cate.py::{name}"
            for name in (
                "test_frozen_scenario_result_arrays_and_caller_ownership",
                "test_public_fit_matches_hand_derived_nonzero_effect_and_cluster_sandwich",
                "test_exact_small_balanced_randomization_enumeration_has_unbiased_public_ate",
                "test_assignment_support_is_not_inference_availability_and_never_redraws",
                "test_public_frozen_rule_predicts_independent_oracle_and_honors_budgets",
                "test_partial_cluster_unit_policy_equal_weights_use_original_roster_sizes",
                "test_public_validation_selection_and_predictions_have_available_evidence",
                "test_observational_mixed_clusters_adjust_all_confounders_and_keep_unit_policies",
                "test_arbitrary_ids_row_order_and_exact_uniform_cloning_preserve_public_fit",
                "test_uniform_cloning_preserves_validation_and_deployment_not_just_fit",
                "test_incomplete_or_malformed_roster_has_exact_construction_refusal",
                "test_unequal_size_design_uptake_metric_order_and_complete_wire_parity",
                "test_original_varying_noise_witness_retained_with_independent_truth",
                "test_public_scalar_power_and_estimator_share_independent_cluster_variance_equation",
                "test_public_policy_and_selection_values_match_nonzero_causal_targets",
                "test_weighted_gates_tied_boundary_permutations_and_clones",
                "test_uniform_order_statistic_finite_batch_target",
                "test_exact_policy_target_preserves_evaluation_roster",
                "test_point_accuracy_and_nonvacuity_inventory_detects_changed_or_missing_public_values",
                "test_sizing_inventory_and_actual_rejection_accounting",
            )
        )


@dataclass(frozen=True, slots=True)
class ClusteredObservation:
    statistic: str
    truth: float | None
    outcome: _KeyOutcome

    def __post_init__(self) -> None:
        if self.outcome.status not in ("ok", "excluded", "failed"):
            refuse(_INVALID, reason="unknown observation status")
        if self.truth is not None and not math.isfinite(self.truth):
            refuse(_INVALID, reason="observation truth must be finite or unavailable")
        if self.outcome.status != "ok" and not self.outcome.reason:
            refuse(_INVALID, reason="failed or excluded observations require an exact reason")
        if self.outcome.status == "ok":
            if (
                self.truth is None
                or self.outcome.point is None
                or not math.isfinite(self.outcome.point)
            ):
                refuse(
                    _INVALID,
                    reason="available clustered observations require finite point and truth",
                )
            if (
                self.outcome.lb is None
                and self.outcome.ub is None
                and not self.outcome.interval_reason
            ):
                refuse(_INVALID, reason="point-only observations require an interval reason")
            if any(
                bound is not None and not math.isfinite(bound)
                for bound in (self.outcome.lb, self.outcome.ub)
            ):
                refuse(_INVALID, reason="available interval endpoints must be finite")
            if self.outcome.lb is not None and self.outcome.ub is not None:
                if self.outcome.lb > self.outcome.ub:
                    refuse(_INVALID, reason="interval endpoints must be ordered")


def _record(
    name: str, result: object, truth: float | None, reason: str | None = None
) -> ClusteredObservation:
    from increment.simulate.runner import _KeyOutcome

    if isinstance(result, CodedError):
        return ClusteredObservation(name, truth, _KeyOutcome(status="failed", reason=result.code))
    point = next(
        (
            getattr(result, attr)
            for attr in ("ate", "value", "estimate", "effect")
            if getattr(result, attr, None) is not None
        ),
        None,
    )
    if point is None:
        exact = reason or getattr(result, "unavailable_reason", None)
        if exact is None:
            refuse(_INVALID, reason=f"{name}: missing point must have an exact reason")
        return ClusteredObservation(name, truth, _KeyOutcome(status="excluded", reason=exact))
    if truth is None:
        return ClusteredObservation(
            name,
            None,
            _KeyOutcome(status="excluded", reason="simulate.cluster_dgp.empty_oracle_target"),
        )
    lb, ub = getattr(result, "lb", None), getattr(result, "ub", None)
    interval_reason = None
    if lb is None or ub is None:
        interval_reason = reason or getattr(result, "unavailable_reason", None)
        if interval_reason is None:
            refuse(_INVALID, reason=f"{name}: unavailable interval must have an exact reason")
    return ClusteredObservation(
        name,
        truth,
        _KeyOutcome(
            status="ok",
            point=float(point),
            lb=lb,
            ub=ub,
            open_side=getattr(result, "open_side", None),
            interval_reason=interval_reason,
        ),
    )


def _immutable_accounting(stats: _KeyStats) -> _KeyStats:
    return replace(
        stats,
        **{
            name: MappingProxyType(dict(getattr(stats, name)))
            for name in ("failure_reasons", "exclusion_reasons", "interval_unavailable_reasons")
        },
    )


def _coverage_hits(
    rate: float | None,
    denominator: int,
    *,
    location: str,
) -> tuple[int, int] | None:
    """Find integer hit counts whose correctly rounded division equals the rate."""
    if rate is None:
        return None
    if denominator == 0:
        refuse(_INVALID, reason=f"{location} must be null when its denominator is zero")
    exact = Fraction(rate)
    lower = max(
        0,
        math.ceil((exact + Fraction(math.nextafter(rate, -math.inf))) * denominator / 2),
    )
    upper = min(
        denominator,
        math.floor((exact + Fraction(math.nextafter(rate, math.inf))) * denominator / 2),
    )
    # Check midpoint ties without assuming the rate uniquely identifies a count.
    if lower <= upper and lower / denominator != rate:
        lower += 1
    if lower <= upper and upper / denominator != rate:
        upper -= 1
    if lower > upper:
        refuse(_INVALID, reason=f"{location} does not represent an integer hit count")
    return lower, upper


def _validate_coverage_contract(
    values: Mapping[str, object],
    *,
    attempted: int,
    interval_estimable: int,
    confidence_set_estimable: int,
    set_only: int,
) -> None:
    """Check rates, denominators, and plug-in Bernoulli MCSEs agree."""
    from increment.simulate.runner import _bernoulli_mcse

    rate_specs = (
        ("coverage_conditional", interval_estimable),
        ("coverage_unconditional", attempted),
        ("set_coverage_conditional", confidence_set_estimable),
        ("set_coverage_unconditional", attempted),
    )
    hits: dict[str, tuple[int, int] | None] = {}
    for name, denominator in rate_specs:
        hits[name] = _coverage_hits(
            cast(float | None, values[name]),
            denominator,
            location=f"evidence accounting {name}",
        )
        mcse_name = f"{name}_mcse"
        observed_mcse = cast(float | None, values[mcse_name])
        if observed_mcse is not None:
            try:
                expected_mcse = _bernoulli_mcse(cast(float, values[name]), denominator)
            except (OverflowError, ValueError):
                refuse(_INVALID, reason=f"evidence accounting {mcse_name} is unrepresentable")
            if observed_mcse != expected_mcse:
                refuse(_INVALID, reason=f"evidence accounting {mcse_name} is inconsistent")

    common: dict[str, tuple[int, int]] = {}
    for prefix in ("coverage", "set_coverage"):
        conditional = hits[f"{prefix}_conditional"] or (0, 0)
        unconditional = hits[f"{prefix}_unconditional"] or (0, 0)
        lower = max(conditional[0], unconditional[0])
        upper = min(conditional[1], unconditional[1])
        if lower > upper:
            refuse(_INVALID, reason=f"evidence accounting {prefix} rates are inconsistent")
        common[prefix] = lower, upper
    interval, confidence_set = common["coverage"], common["set_coverage"]
    if confidence_set[1] < interval[0] or confidence_set[0] > interval[1] + set_only:
        refuse(_INVALID, reason="evidence accounting interval and set hits are inconsistent")


def reduce_clustered_observations(
    observations: Sequence[ClusteredObservation],
) -> Mapping[str, _KeyStats]:
    """Use R03's reducer with per-replication truths by centering each interval."""
    from increment.simulate.runner import _reduce_key

    grouped: dict[str, list[_KeyOutcome]] = {}
    for observation in observations:
        outcome, truth = observation.outcome, observation.truth
        if outcome.status == "ok":
            if truth is None or not math.isfinite(truth):
                refuse(_INVALID, reason="estimable observations require finite independent truth")
            outcome = replace(
                outcome,
                point=outcome.point - truth if outcome.point is not None else None,
                lb=outcome.lb - truth if outcome.lb is not None else None,
                ub=outcome.ub - truth if outcome.ub is not None else None,
            )
        grouped.setdefault(observation.statistic, []).append(outcome)
    result = {}
    for key, values in grouped.items():
        result[key] = _immutable_accounting(_reduce_key(values, 0.0))
    return MappingProxyType(result)


def _key_stats_from_payload(payload: object) -> _KeyStats:
    """Parse one persisted accounting row and re-check its invariants."""
    from increment.simulate.runner import _KeyStats, _validate_aggregate_value

    if not isinstance(payload, Mapping):
        refuse(_INVALID, reason="evidence accounting rows must be mappings")
    schema = fields(_KeyStats)
    expected = {item.name for item in schema}
    if set(payload) != expected:
        refuse(
            _INVALID,
            reason=f"evidence accounting fields must be exactly {sorted(expected)!r}",
        )
    row = cast(Mapping[str, object], payload)

    counts = (
        "attempted",
        "point_estimable",
        "interval_estimable",
        "confidence_set_estimable",
        "set_only",
        "excluded",
        "failed",
    )
    values: dict[str, object] = {}
    for name in counts:
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            refuse(_INVALID, reason=f"evidence accounting {name} must be a nonnegative integer")
        values[name] = value

    reason_fields = ("failure_reasons", "exclusion_reasons", "interval_unavailable_reasons")
    for name in reason_fields:
        raw = row[name]
        if not isinstance(raw, Mapping):
            refuse(_INVALID, reason=f"evidence accounting {name} must be a mapping")
        parsed: dict[str, int] = {}
        for reason, count in raw.items():
            if (
                not isinstance(reason, str)
                or not reason
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                refuse(_INVALID, reason=f"evidence accounting {name} has malformed reason counts")
            parsed[reason] = count
        values[name] = parsed

    aggregate_fields = tuple(
        item.name for item in schema if item.name not in counts and item.name not in reason_fields
    )
    for name in aggregate_fields:
        value = row[name]
        if value is None:
            values[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            refuse(_INVALID, reason=f"evidence accounting {name} must be a finite number or null")
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            refuse(_INVALID, reason=f"evidence accounting {name} must be a finite number or null")
        if not math.isfinite(converted):
            refuse(_INVALID, reason=f"evidence accounting {name} must be a finite number or null")
        values[name] = converted

    attempted = cast(int, values["attempted"])
    point_estimable = cast(int, values["point_estimable"])
    interval_estimable = cast(int, values["interval_estimable"])
    confidence_set_estimable = cast(int, values["confidence_set_estimable"])
    set_only = cast(int, values["set_only"])
    excluded = cast(int, values["excluded"])
    failed = cast(int, values["failed"])
    if attempted != point_estimable + set_only + excluded + failed:
        refuse(_INVALID, reason="evidence accounting attempted total is inconsistent")
    if interval_estimable > point_estimable:
        refuse(_INVALID, reason="evidence accounting interval total exceeds point total")
    if confidence_set_estimable != interval_estimable + set_only:
        refuse(_INVALID, reason="evidence accounting confidence-set total is inconsistent")
    reason_totals = (
        ("failure_reasons", failed),
        ("exclusion_reasons", excluded),
        ("interval_unavailable_reasons", point_estimable - interval_estimable),
    )
    for name, expected_total in reason_totals:
        reasons = cast(dict[str, int], values[name])
        if sum(reasons.values()) != expected_total:
            refuse(_INVALID, reason=f"evidence accounting {name} total is inconsistent")

    for name in aggregate_fields:
        try:
            _validate_aggregate_value(
                name,
                cast(float | None, values[name]),
                point_estimable=point_estimable,
                interval_estimable=interval_estimable,
                confidence_set_estimable=confidence_set_estimable,
                attempted=attempted,
                location=f"evidence accounting {name}",
            )
        except PydanticCustomError as error:
            refuse(_INVALID, reason=str(error))
    _validate_coverage_contract(
        values,
        attempted=attempted,
        interval_estimable=interval_estimable,
        confidence_set_estimable=confidence_set_estimable,
        set_only=set_only,
    )
    return _KeyStats(**cast(dict[str, Any], values))


def _clustered_tables_from_payload(payload: object) -> Mapping[str, Mapping[str, _KeyStats]]:
    if not isinstance(payload, Mapping):
        refuse(_INVALID, reason="evidence tables must be a mapping")
    tables: dict[str, dict[str, _KeyStats]] = {}
    for cell, raw_table in payload.items():
        if not isinstance(cell, str) or not cell:
            refuse(_INVALID, reason="evidence cell names must be nonempty strings")
        if not isinstance(raw_table, Mapping):
            refuse(_INVALID, reason=f"evidence table {cell!r} must be a mapping")
        table: dict[str, _KeyStats] = {}
        for name, raw_row in raw_table.items():
            if not isinstance(name, str) or not name:
                refuse(_INVALID, reason="evidence statistic names must be nonempty strings")
            table[name] = _key_stats_from_payload(raw_row)
        tables[cell] = table
    return tables


@dataclass(frozen=True, slots=True)
class ClusteredEvidenceArtifact:
    """Immutable, portable evidence tied to one exact registration."""

    registration: str
    tables: Mapping[str, Mapping[str, _KeyStats]]

    def __post_init__(self) -> None:
        if not isinstance(self.registration, str) or not self.registration:
            refuse(_INVALID, reason="evidence requires a nonempty registration fingerprint")
        if any(not isinstance(cell, str) or not cell for cell in self.tables):
            refuse(_INVALID, reason="evidence cell names must be nonempty strings")
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(
                {
                    cell: MappingProxyType(
                        {name: _immutable_accounting(row) for name, row in table.items()}
                    )
                    for cell, table in self.tables.items()
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ClusteredEvidenceArtifact:
        """Load and validate JSON-compatible evidence produced by :meth:`payload`."""
        if not isinstance(payload, Mapping) or set(payload) != {"registration", "tables"}:
            refuse(_INVALID, reason="evidence payload must contain only registration and tables")
        registration = payload["registration"]
        if not isinstance(registration, str) or not registration:
            refuse(_INVALID, reason="evidence requires a nonempty registration fingerprint")
        return cls(registration, _clustered_tables_from_payload(payload["tables"]))

    def validated_tables(
        self, registration: ClusteredRegistration
    ) -> Mapping[str, Mapping[str, _KeyStats]]:
        """Refuse evidence produced by any registration other than ``registration``."""
        if self.registration != registration.fingerprint:
            refuse(_INVALID, reason="evidence registration fingerprint does not match")
        return self.tables

    def payload(self) -> dict[str, object]:
        """Return JSON-compatible evidence without dropping its registration identity."""
        return {
            "registration": self.registration,
            "tables": {
                cell: clustered_table_payload(table) for cell, table in sorted(self.tables.items())
            },
        }


def clustered_table_payload(table: Mapping[str, _KeyStats]) -> dict[str, dict[str, object]]:
    """Portable accounting rows, preserving numeric nulls and exact reason codes."""
    return {
        name: {
            field.name: dict(value) if isinstance(value, Mapping) else value
            for field in fields(row)
            for value in (getattr(row, field.name),)
        }
        for name, row in sorted(table.items())
    }


def _original_varying_noise(
    scenario: ClusteredCATEScenario, *, stream: Role, replication: int
) -> ClusteredCATEResult:
    """Archived e6v0 rows, with Y(1)-Y(0)=.12*x declared independently.

    Alternating deterministic assignment is a regression witness, not a draw
    from the randomized calibration law. Increasing m changes the eleven-row
    noise pattern; it is NOT exact uniform cloning of five rows.
    """
    child_rng(scenario.seed, stream, replication, "address-validation")
    rows, a, b, tau, ids = [], [], [], [], []
    for g in range(scenario.n_clusters):
        x = (g - 39.5) / 20
        for j in range(scenario.members_per_cluster):
            y0 = 0.4 * math.sin(g) + 0.01 * ((j % 11) - 5)
            y1 = y0 + 0.12 * x
            for clone in range(scenario.clone_factor):
                rows.append(
                    (
                        f"g{g}-u{j}-{clone}",
                        f"g{g}",
                        "treatment" if g % 2 else "control",
                        y1 if g % 2 else y0,
                        0.0,
                        x,
                    )
                )
                a.append(y0)
                b.append(y1)
                tau.append(0.12 * x)
                ids.append(g)
    if scenario.reverse_rows:
        for values in (rows, a, b, tau, ids):
            values.reverse()
    return ClusteredCATEResult(
        scenario,
        stream,
        replication,
        tuple(rows),
        ("unit_id", "cluster_id", "group_id", "y", "z", "x0"),
        tuple(a),
        tuple(b),
        tuple(tau),
        tuple(ids),
        (scenario.members_per_cluster * scenario.clone_factor,) * scenario.n_clusters,
    )


def _rank_truth(
    score: np.ndarray, tau: np.ndarray, weights: np.ndarray, *, clustered: bool
) -> tuple[float, float]:
    """Integrate top-share causal contrasts, without calling inference helpers.

    For a tied block with mass h and mean effect t, cumulative mass a and
    cumulative effect R, TOC(q)=t-ATE+(R-a*t)/q on (a,a+h]. Integrate TOC
    The unclustered statistic uses a discrete right-endpoint Riemann sum,
    averaging rank coefficients within ties.
    """
    order = np.argsort(-score, kind="stable")
    score, tau, weights = score[order], tau[order], weights[order]
    if not clustered:
        n = len(tau)
        rank = np.arange(1, n + 1, dtype=float)
        autoc_weights = (np.cumsum((1 / rank)[::-1])[::-1] - 1) / n
        qini_weights = ((n + 1) / 2 - rank) / n**2
        starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
        counts = np.diff(np.r_[starts, n])
        autoc_weights = np.repeat(np.add.reduceat(autoc_weights, starts) / counts, counts)
        qini_weights = np.repeat(np.add.reduceat(qini_weights, starts) / counts, counts)
        return float(autoc_weights @ tau), float(qini_weights @ tau)
    overall = float(np.average(tau, weights=weights))
    a, response, autoc, qini = 0.0, 0.0, 0.0, 0.0
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    block_mass = np.add.reduceat(weights, starts)
    block_effect = np.add.reduceat(weights * tau, starts) / block_mass
    block_mass /= weights.sum()
    for mass, effect in zip(block_mass, block_effect, strict=True):
        correction = response - a * effect
        autoc += (effect - overall) * mass
        if a:
            autoc += correction * math.log1p(mass / a)
        qini += (effect - overall) * (a * mass + mass * mass / 2) + correction * mass
        response += mass * effect
        a += mass
    return autoc, qini


def _causal_weighted_cut(score: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Inverse weighted CDF, retaining complete tied score blocks.

    fsum avoids row-order accumulation drift. Match the public boundary
    convention gamma_(4N+8): that many positive floating operations bound
    the relative mass/CDF rounding error, including rounded 1/m weights.
    This tolerance is arithmetic, not a fitted or statistical error band.
    """
    blocks: dict[float, list[float]] = {}
    for value, weight in zip(score, weights, strict=True):
        blocks.setdefault(float(value), []).append(float(weight))
    values = sorted(blocks)
    masses = [math.fsum(blocks[value]) for value in values]
    total = math.fsum(masses)
    rounding = (4 * len(score) + 8) * np.finfo(float).eps
    if not values or total <= 0 or rounding >= 1:
        refuse(_INVALID, reason="weighted quantile has no finite positive mass/error bound")
    gamma = rounding / (1 - rounding)
    cumulative = 0.0
    for value, mass in zip(values, masses, strict=True):
        cumulative = math.fsum((cumulative, mass))
        if cumulative / total * (1 + gamma) >= quantile:
            return value
    return values[-1]


def _validation_targets(
    population: ClusteredCATEResult, weight: Weight, rule: TargetingRule | None = None
) -> dict[str, float | None]:
    from increment.estimation.cate import Covariate, fit_cate
    from increment.estimation.targeting import _holdout_mask

    table = population.table
    ids = (
        np.asarray(table["cluster_id"]).astype(str)
        if population.scenario.declare_clusters
        else None
    )
    held = _holdout_mask(np.asarray(table["unit_id"]), cluster_ids=ids)
    tau = np.asarray(population.tau)[held]
    weights = population.weights(weight)[held]
    cols = {name: values[held] for name, values in population.covariates.items()}
    if rule is None:
        # A policy refusal must not relabel an available validation as a failure.
        # Only its training score is reconstructed; every causal target uses tau.
        d = (np.asarray(table["group_id"]) == "treatment").astype(float)
        fit = fit_cate(
            np.asarray(table["y"])[~held],
            d[~held],
            {name: values[~held] for name, values in population.covariates.items()},
            interact=[Covariate(name=name) for name in population.scenario.interactions],
            cluster_ids=None if ids is None else ids[~held],
            cluster_weight=weight,
            intervention_grain=population.scenario.intervention_grain,
        )
        score = fit.score_state.score(cols)
    else:
        score = rule.score_state.score(cols)
    targets: dict[str, float | None] = {"validation.ate": float(np.average(tau, weights=weights))}
    autoc, qini = _rank_truth(score, tau, weights, clustered=ids is not None)
    targets.update({"validation.autoc": autoc, "validation.qini": qini})
    if ids is None:
        cut = float(np.quantile(score, 0.5))
    else:
        cut = _causal_weighted_cut(score, weights, 0.5)
    for group, chosen in ((1, score <= cut), (2, score > cut)):
        targets[f"validation.group{group}"] = (
            float(np.average(tau[chosen], weights=weights[chosen])) if chosen.any() else None
        )
    return targets


@dataclass(frozen=True, slots=True)
class ClusteredReplication:
    registration: str
    cell: str
    replication: int
    observations: tuple[ClusteredObservation, ...]
    reports: tuple[tuple[str, str], ...]
    policy_oracles: tuple[tuple[str, PolicyTruth], ...]
    assignment_support: str
    point_reconstructions: tuple[tuple[str, AffinePointReconstruction], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "reports", tuple(tuple(row) for row in self.reports))
        object.__setattr__(self, "policy_oracles", tuple(tuple(row) for row in self.policy_oracles))
        object.__setattr__(
            self, "point_reconstructions", tuple(tuple(row) for row in self.point_reconstructions)
        )
        names = [row.statistic for row in self.observations]
        if len(names) != len(set(names)):
            refuse(_INVALID, reason="one replication must have exactly one outcome per statistic")


def _validation_observations(
    population: ClusteredCATEResult,
    cell: ClusteredCell,
    validation: CateValidation | CodedError,
    rule: TargetingRule | CodedError,
) -> tuple[ClusteredObservation, ...]:
    from increment.simulate.runner import _KeyOutcome

    if isinstance(validation, CodedError):
        return tuple(
            _record(spec.name, validation, None)
            for spec in cell.statistics
            if spec.name.startswith("validation.")
        )
    if validation.population is not None:
        return tuple(
            ClusteredObservation(
                spec.name,
                None,
                _KeyOutcome(
                    status="excluded",
                    reason=_OVERLAP_ROSTER.code,
                ),
            )
            for spec in cell.statistics
            if spec.name.startswith("validation.")
        )
    targets = _validation_targets(
        population,
        cell.weighting,
        None if isinstance(rule, CodedError) else rule,
    )
    rows = {
        "validation.ate": validation.holdout_ate,
        "validation.autoc": validation.autoc,
        "validation.qini": validation.qini,
        **{f"validation.group{row.group}": row for row in validation.groups},
    }
    observations = []
    for name, row in rows.items():
        reason = validation.unavailable_reason if name == "validation.ate" else None
        if name in ("validation.autoc", "validation.qini") and not cell.scenario.declare_clusters:
            reason = "simulate.cluster_dgp.rank_test_point_only"
        observations.append(_record(name, row, targets[name], reason))
    for name in ("autoc", "qini"):
        test = getattr(validation, name)
        outcome = (
            _KeyOutcome(status="excluded", reason=test.unavailable_reason)
            if test.p_value is None
            else _KeyOutcome(
                status="ok",
                point=float(test.p_value < 0.05),
                interval_reason="simulate.cluster_dgp.rejection_indicator",
            )
        )
        observations.append(ClusteredObservation(f"validation.{name}.reject", 0.0, outcome))
    return tuple(observations)


def _fit_observation(
    source: MomentSource, target: float, options: dict[str, Any]
) -> tuple[ClusteredObservation, str | None]:
    from increment.cate import estimate_cate

    try:
        fit = estimate_cate(source, "y", **options)
    except CodedError as exc:
        return _record("fit.ate", exc, target), None
    return _record("fit.ate", fit, target), fit.model_dump_json()


def evaluate_clustered_replication(
    registration: ClusteredRegistration, cell_name: str, replication: int
) -> ClusteredReplication:
    """Execute registered public calls and retain every result/refusal separately.

    Policy points are compared with exact potential outcomes on their actual
    evaluation batch. Their accuracy is gate-conditioned; availability retains
    every attempt. No oracle draws are called exact population truth.
    """
    from increment.cate import select_targeting_rule, targeting_rule, validate_cate
    from increment.estimation.targeting import ClusterBootstrap

    selected = [cell for cell in registration.cells if cell.name == cell_name]
    if len(selected) != 1 or not 0 <= replication < registration.replications:
        refuse(_INVALID, reason="cell/replication must belong to the frozen registration")
    cell = selected[0]
    s, weight = cell.scenario, cell.weighting
    population = honest_clustered_population(s, replication=replication)
    observations: list[ClusteredObservation] = []
    reports: list[tuple[str, str]] = []
    oracles: list[tuple[str, PolicyTruth]] = []
    reconstructions: list[tuple[str, AffinePointReconstruction]] = []
    try:
        source = clustered_source(population)
    except CodedError as exc:
        return ClusteredReplication(
            registration.fingerprint,
            cell.name,
            replication,
            tuple(_record(spec.name, exc, None) for spec in cell.statistics),
            (),
            (),
            population.assignment_support,
        )
    options: dict[str, Any] = {
        "control": "control",
        "interact": s.interactions,
        "cluster_weight": weight,
    }
    bootstrap_options = (
        {
            "bootstrap": ClusterBootstrap(
                seed=registration.bootstrap_seed, repetitions=registration.bootstrap_repetitions
            )
        }
        if s.declare_clusters
        else {}
    )
    ate = (
        population.truth.member_ate
        if weight == "member_count"
        else population.truth.equal_cluster_ate
    )
    if s.assignment != "observational":
        observation, report = _fit_observation(source, ate, options)
        observations.append(observation)
        if report is not None:
            reports.append(("fit", report))

    # Run validation even if the fit refused: each public operation owns its support.
    try:
        validation = validate_cate(
            source,
            "y",
            n_groups=2,
            include_evaluation_population=True,
            **options,
            **bootstrap_options,
        )
    except CodedError as exc:
        validation = exc
    try:
        rule = targeting_rule(
            source,
            "y",
            fraction=0.5,
            include_evaluation_population=True,
            **options,
            **bootstrap_options,
        )
    except CodedError as exc:
        rule = exc
    observations.extend(_validation_observations(population, cell, validation, rule))
    if not isinstance(validation, CodedError):
        reports.append(("validation", validation.model_dump_json()))

    try:
        selection = select_targeting_rule(
            source,
            "y",
            fractions=(0.0, 0.5, 1.0),
            seed=registration.selection_seed,
            n_folds=4 if s.witness == "original_varying_noise" and s.n_clusters == 16 else 2,
            include_evaluation_population=True,
            **options,
            **bootstrap_options,
        )
    except CodedError as exc:
        selection = exc
    else:
        reports.append(("selection", selection.model_dump_json()))
    for name, candidate in (
        ("policy", rule),
        ("selection", selection if isinstance(selection, CodedError) else selection.rule),
    ):
        if isinstance(candidate, CodedError):
            observations.extend(
                _record(f"{name}.{stat}", candidate, None) for stat in ("effect", "uplift")
            )
            continue
        reports.append((f"{name}.rule", candidate.model_dump_json()))
        # Algebra failures stop the campaign; they are never statistical exclusions.
        reconstructions.extend(
            (f"{name}.{point.statistic}", point)
            for point in check_policy_reconstruction(candidate, population)
        )
        try:
            truth, average = evaluation_policy_truth(candidate, population)
        except CodedError as exc:
            observations.extend(
                _record(f"{name}.{stat}", exc, None) for stat in ("effect", "uplift")
            )
            continue
        oracles.append((name, truth))
        effect = truth.member_effect if weight == "member_count" else truth.equal_cluster_effect
        reason = (
            candidate.unavailable_reason or "simulate.cluster_dgp.policy_gate_closed"
            if candidate.policy_value is None
            else "simulate.cluster_dgp.gated_policy_point_only"
        )
        observations.append(_record(f"{name}.effect", candidate.policy_value, effect, reason))
        observations.append(
            _record(
                f"{name}.uplift",
                candidate.uplift_vs_average,
                None if effect is None else effect - average,
                reason,
            )
        )
    observations.extend(_point_accuracy_observations(registration, cell, observations))
    if {obs.statistic for obs in observations} != {spec.name for spec in cell.statistics}:
        refuse(_INVALID, reason="execution did not account for every registered statistic")
    return ClusteredReplication(
        registration.fingerprint,
        cell.name,
        replication,
        tuple(observations),
        tuple(reports),
        tuple(oracles),
        population.assignment_support,
        tuple(reconstructions),
    )


def _point_accuracy_observations(
    registration: ClusteredRegistration,
    cell: ClusteredCell,
    observations: Sequence[ClusteredObservation],
) -> tuple[ClusteredObservation, ...]:
    from increment.simulate.runner import _KeyOutcome

    plans = {p.statistic: p for p in registration.point_accuracy if p.cell == cell.name}
    rows = []
    for observation in observations:
        if observation.statistic not in (
            "policy.effect",
            "policy.uplift",
            "selection.effect",
            "selection.uplift",
        ):
            continue
        name, outcome = observation.statistic, observation.outcome
        available = outcome.status == "ok"
        rows.append(
            ClusteredObservation(
                f"{name}.unavailable",
                0.0,
                _KeyOutcome(
                    status="ok",
                    point=float(not available),
                    interval_reason="simulate.cluster_dgp.reporting_indicator",
                ),
            )
        )
        if not available:
            error = outcome
        elif name not in plans:
            error = _KeyOutcome(
                status="excluded", reason="simulate.cluster_dgp.point_accuracy_not_preregistered"
            )
        else:
            assert observation.truth is not None and outcome.point is not None
            error = _KeyOutcome(
                status="ok",
                point=float(
                    abs(outcome.point - observation.truth) > plans[name].absolute_tolerance
                ),
                interval_reason="simulate.cluster_dgp.point_error_indicator",
            )
        rows.append(ClusteredObservation(f"{name}.accuracy", 0.0, error))
    return tuple(rows)


def evaluate_clustered_registration(
    registration: ClusteredRegistration,
) -> ClusteredEvidenceArtifact:
    """Run a caller-budgeted manifest and retain its exact registration identity.

    This function deliberately does not choose repetitions, MC error allocations
    or acceptance thresholds. Those must be frozen globally by the release owner.
    """
    tables = MappingProxyType(
        {
            cell.name: reduce_clustered_observations(
                tuple(
                    observation
                    for replication in range(registration.replications)
                    for observation in evaluate_clustered_replication(
                        registration, cell.name, replication
                    ).observations
                )
            )
            for cell in registration.cells
        }
    )
    return ClusteredEvidenceArtifact(registration.fingerprint, tables)
