"""Frozen cluster-randomized design and actual estimator/scale/nuisance invocation inventory.

Original design names and axes are retained, with required heteroskedastic
intersections, first stages, heterogeneous targets and useful alternatives.
Every truth is declared under a superpopulation potential-outcome law;
sharp-null finite-population enumeration is a separate acceptance experiment.
This manifest is executable infrastructure, not evidence that any cell passed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# DGP: two-level (cluster, member) additive model with independently
# derivable truths.


class _ShockGenerator(Protocol):
    """The sized draws consumed by the cluster-shock generator."""

    def normal(self, loc: float, scale: float, *, size: int) -> np.ndarray: ...

    def chisquare(self, df: float, *, size: int) -> np.ndarray: ...


@dataclass(frozen=True)
class ClusterDGP:
    """Cluster-randomized two-level DGP: ``y_ij = mu + effect_i*d_i + b_j +
    e_ij``, member i in cluster j, with cluster shock ``b_j`` and member
    noise ``e_ij``.

    ``k_t``/``k_c`` : cluster counts per arm (the K-pairs axis).
    ``size_mean``/``size_dist`` : member counts per cluster, ``"equal"``
        (every cluster exactly ``size_mean``, rounded) or ``"lognormal"``
        (``lognormal(mu, 0.75)`` scaled to the target mean).
    ``icc`` : intra-cluster correlation at ``var_ratio=1`` --
        ``sigma_b_c**2 / (sigma_b_c**2 + sigma_e**2)`` for the CONTROL arm;
        ``sigma_e**2`` is shared by both arms (only the between-cluster
        component varies by arm, see ``var_ratio``).
    ``var_ratio`` : treatment's between-cluster variance as a multiple of
        control's (``sigma_b_t**2 = var_ratio * sigma_b_c**2``); the
        unequal-arm-variance axis and its reversal (``var_ratio`` vs
        ``1/var_ratio``).
    ``shock`` : ``"normal"`` (``N(0, sigma_b**2)``) or ``"skew"``
        (centered skew-normal with the same variance, via ``scipy`` is
        avoided -- built from a squared-and-recentered normal so this
        module has no extra dependency).
    ``effect`` : the per-member additive effect for ordinary cells; ``None``
        selects the SIZE-DEPENDENT effect fixture (only meaningful at
        ``size_dist="equal", icc=0.0, var_ratio=1.0``, mirroring the
        research's exact 5/20-member, 0/1-effect target-weighting witness).
    ``leverage`` : when set, cluster 0 of the CONTROL arm is inflated to
        carry half of that arm's total members (a single dominant cluster).
    ``mixed_frac`` : fraction of clusters, if any, that mix both arms'
        members 50/50 instead of being treatment-pure (the supported
        observational mixed-cluster axis; 0.0 for every randomized/
        encouragement cell).
    ``sigma_e`` : member-level noise std, shared by both arms.
    ``seed`` : draw seed.
    """

    k_t: int
    k_c: int
    size_mean: float = 20.0
    size_dist: Literal["equal", "lognormal"] = "equal"
    icc: float = 0.0
    var_ratio: float = 1.0
    shock: Literal["normal", "skew"] = "normal"
    effect: float | None = 0.0
    leverage: bool = False
    mixed_frac: float = 0.0
    sigma_e: float = 1.0
    seed: int = 0
    first_stage: float | None = None
    heterogeneous: bool = False

    def __post_init__(self) -> None:
        if min(self.k_t, self.k_c) < 1 or self.size_mean <= 0:
            raise ValueError("positive arm counts and cluster size are required")
        if not 0 <= self.icc < 1 or self.var_ratio <= 0 or self.sigma_e < 0:
            raise ValueError("invalid variance or ICC parameters")
        if not 0 <= self.mixed_frac <= 1:
            raise ValueError("mixed_frac must be in [0, 1]")
        if self.first_stage is not None and not 0 < self.first_stage <= 1:
            raise ValueError("first_stage must be in (0, 1]")
        if self.first_stage is not None and (self.mixed_frac or self.heterogeneous):
            raise ValueError("encouragement cells require their declared pure-cluster DGP")

    def _sizes(self, rng: np.random.Generator, k: int) -> np.ndarray:
        if self.size_dist == "equal":
            return np.full(k, max(1, round(self.size_mean)), dtype=int)
        # lognormal(mu, 0.75) scaled so E[size] == size_mean; sigma=0.75 is
        # a moderately heavy right tail without a nontrivial chance of a
        # zero-member cluster after rounding.
        sigma = 0.75
        mu = math.log(self.size_mean) - 0.5 * sigma * sigma
        draws = rng.lognormal(mu, sigma, size=k)
        return np.maximum(1, np.round(draws)).astype(int)

    def _shock(self, rng: _ShockGenerator, k: int, sigma_b: float) -> np.ndarray:
        if sigma_b <= 0.0:
            return np.zeros(k)
        if self.shock == "normal":
            return rng.normal(0.0, sigma_b, size=k)
        # Centered skew shock: chi-square(df=2) recentered/rescaled to mean
        # 0, variance sigma_b**2 -- a bounded-below, right-skewed shock
        # without adding a scipy dependency for skew-normal.
        raw = rng.chisquare(df=2.0, size=k)
        raw = (raw - 2.0) / 2.0  # Var(chisq(2))=4, mean 2
        return raw * sigma_b

    def draw(self) -> DGPSample:
        if self.heterogeneous:
            return target_weighting_dgp(self.seed, self.k_t + self.k_c)[0]
        if self.effect is None:
            raise ValueError("Declare heterogeneous=True for the size/effect sampling law")
        rng = np.random.default_rng(self.seed)
        sigma_b_c = self.sigma_e * math.sqrt(self.icc / max(1.0 - self.icc, 1e-12))
        sigma_b_t = sigma_b_c * math.sqrt(self.var_ratio)

        sizes_c = self._sizes(rng, self.k_c)
        sizes_t = self._sizes(rng, self.k_t)
        if self.leverage and self.k_c > 1:
            sizes_c = sizes_c.copy()
            rest = int(sizes_c[1:].sum())
            sizes_c[0] = max(rest, 1)  # cluster 0 carries >= half the arm

        b_c = self._shock(rng, self.k_c, sigma_b_c)
        b_t = self._shock(rng, self.k_t, sigma_b_t)

        n_mixed = int(round(self.mixed_frac * min(self.k_t, self.k_c)))
        if n_mixed:
            shared = self._shock(rng, n_mixed, 1.0)
            b_c[:n_mixed] = shared * sigma_b_c
            b_t[:n_mixed] = shared * sigma_b_t
        rows: dict[str, list] = {
            key: [] for key in ("u", "g", "cluster", "y", "z", "den", "num", "took", "y0", "y1")
        }
        for arm, sizes, shocks, sigma in (
            ("C", sizes_c, b_c, sigma_b_c),
            ("T", sizes_t, b_t, sigma_b_t),
        ):
            for j, (size, shock) in enumerate(zip(sizes, shocks, strict=True)):
                z = rng.normal(size=size)
                noise = rng.normal(0.0, self.sigma_e, size=size)
                latent = shock / sigma if sigma else 0.0
                y0 = 5.0 + sigma_b_c * latent + 0.3 * z + noise
                outcome_shock_change = sigma_b_t - sigma_b_c
                if self.first_stage is not None:
                    outcome_shock_change /= self.first_stage
                y1 = y0 + self.effect + outcome_shock_change * latent
                if self.first_stage is None:
                    uptake = np.full(size, int(arm == "T"))
                else:
                    # One-sided compliance independent of both potential outcomes.
                    uptake = (rng.random(size) < self.first_stage) * int(arm == "T")
                y = np.where(uptake, y1, y0)
                den = 1.0 + rng.poisson(2.0, size=size)
                label = f"mix{j}" if j < n_mixed else f"{arm}_c{j}"
                for i in range(size):
                    rows["u"].append(f"u{len(rows['u'])}")
                    rows["g"].append(arm)
                    rows["cluster"].append(label)
                    for key, values in (
                        ("y", y),
                        ("z", z),
                        ("den", den),
                        ("num", den * y),
                        ("took", uptake),
                        ("y0", y0),
                        ("y1", y1),
                    ):
                        rows[key].append(float(values[i]))
        return DGPSample(
            table=pa.table(rows),
            tau_u=self.effect,
            tau_c=self.effect,
            theta_plr=self.effect,
            k_total=self.k_t + self.k_c - n_mixed,
            mean_c=5.0,
            tau_u_relative=self.effect / 5.0,
        )


@dataclass(frozen=True)
class DGPSample:
    table: pa.Table
    tau_u: float  # member-weighted ATE truth (population-total ratio)
    tau_c: float  # equal-cluster-weighted ATE truth
    k_total: int
    mean_c: float = 5.0
    tau_u_relative: float = 0.0
    theta_plr: float = 0.0
    target: Literal["superpopulation", "finite_population"] = "superpopulation"


def target_weighting_dgp(seed: int, n_clusters: int = 400) -> tuple[DGPSample, float]:
    """The research's exact target-weighting witness: cluster sizes 5 or 20
    with equal probability, effects 0 (size 5) or 1 (size 20), propensity
    .2 (size 5) or .6 (size 20) -- independently: ``tau_C = 1/2``,
    ``tau_U = 4/5``, ``theta_PLR = 6/7`` (see the research note, "Target
    weighting must remain explicit"). Used by ``ClusterDGP.heterogeneous`` with cluster-level assignment.

    All three targets use the superpopulation law, not realized composition.
    The second tuple entry is the same PLR target stored on the sample.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {key: [] for key in ("u", "g", "cluster", "y", "z", "y0", "y1")}
    uid = 0
    for j in range(n_clusters):
        big = rng.random() < 0.5
        size = 20 if big else 5
        effect = 1.0 if big else 0.0
        e = 0.6 if big else 0.2
        d = np.full(size, rng.random() < e)
        base = 5.0
        y0 = base + rng.normal(0.0, 0.5, size=size)
        y1 = y0 + effect
        y = np.where(d, y1, y0)
        for i in range(size):
            rows["u"].append(f"u{uid}")
            rows["g"].append("T" if d[i] else "C")
            rows["cluster"].append(f"c{j}")
            rows["y"].append(float(y[i]))
            rows["y0"].append(float(y0[i]))
            rows["y1"].append(float(y1[i]))
            rows["z"].append(float(e))
            uid += 1
    theta_plr = 6.0 / 7.0
    return (
        DGPSample(
            table=pa.table(rows),
            tau_u=4.0 / 5.0,
            tau_c=1.0 / 2.0,
            k_total=n_clusters,
            mean_c=5.0,
            tau_u_relative=(4.0 / 5.0) / 5.0,
            theta_plr=theta_plr,
        ),
        theta_plr,
    )


# ---------------------------------------------------------------------------
# The manifest: named cells, frozen before any calibration result exists.


@dataclass(frozen=True)
class ManifestCell:
    name: str
    stratum: Literal["principal", "adversarial"]
    family: Literal["mean_ratio", "late", "adjusted", "sitewide"]
    dgp: ClusterDGP
    description: str


# Per-arm cluster-count pairs and their reversals, fixed by the frozen design.
_K_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 2),
    (3, 3),
    (5, 5),
    (10, 10),
    (20, 20),
    (40, 40),
    (2, 8),
    (8, 2),
    (3, 12),
    (12, 3),
    (5, 20),
    (20, 5),
    (10, 40),
    (40, 10),
    (20, 80),
    (80, 20),
)


def _base(k_t: int, k_c: int, **overrides) -> ClusterDGP:
    payload = json.dumps([k_t, k_c, overrides], sort_keys=True).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ClusterDGP(k_t=k_t, k_c=k_c, seed=seed, **overrides)


def build_manifest() -> tuple[ManifestCell, ...]:
    """Frozen design inventory; ACCEPTANCE_MANIFEST expands actual invocations."""
    cells: list[ManifestCell] = []

    # 1) K-pairs x reversals, every combination touching mean/ratio, LATE,
    # and adjusted families at ICC=0 (principal baseline).
    for k_t, k_c in _K_PAIRS:
        stratum = "adversarial" if min(k_t, k_c) <= 3 else "principal"
        for family in ("mean_ratio", "late", "adjusted"):
            cells.append(
                ManifestCell(
                    name=f"kpair_{k_t}v{k_c}_{family}",
                    stratum=stratum,
                    family=family,
                    dgp=_base(k_t, k_c, icc=0.2, first_stage=0.8 if family == "late" else None),
                    description=f"K=({k_t},{k_c}), ICC=0.2, equal sizes, null effect.",
                )
            )

    # 2) Size axis: equal vs lognormal at means 5/20/100, one representative
    # balanced and one unbalanced K-pair.
    for size_mean in (5.0, 20.0, 100.0):
        for size_dist in ("equal", "lognormal"):
            for k_t, k_c in ((10, 10), (5, 20)):
                cells.append(
                    ManifestCell(
                        name=f"size_{size_dist}_{int(size_mean)}_{k_t}v{k_c}",
                        stratum="principal" if size_dist == "equal" else "adversarial",
                        family="mean_ratio",
                        dgp=_base(k_t, k_c, size_mean=size_mean, size_dist=size_dist, icc=0.2),
                        description=(f"{size_dist} sizes, mean={size_mean}, K=({k_t},{k_c})."),
                    )
                )

    # 3) Variance-ratio axis and reversals, crossed with a balanced and an
    # unbalanced K-pair (this is the exact axis the pure-cluster Welch fix
    # targets: unequal arm variance under a pooled reference undercovers).
    for ratio in (1.0, 4.0, 16.0, 0.25, 0.0625):
        for k_t, k_c in ((10, 10), (5, 20), (20, 5)):
            cells.append(
                ManifestCell(
                    name=f"varratio_{ratio}_{k_t}v{k_c}",
                    stratum="principal" if ratio == 1.0 else "adversarial",
                    family="adjusted",
                    dgp=_base(k_t, k_c, var_ratio=ratio, icc=0.3),
                    description=f"treatment/control between-cluster variance ratio={ratio}.",
                )
            )

    # 4) ICC axis (0, .2, .8) at a fixed moderate K-pair, both shock types.
    for icc in (0.0, 0.2, 0.8):
        for shock in ("normal", "skew"):
            cells.append(
                ManifestCell(
                    name=f"icc_{icc}_{shock}",
                    stratum="principal" if shock == "normal" else "adversarial",
                    family="mean_ratio",
                    dgp=_base(10, 10, icc=icc, shock=shock),
                    description=f"ICC={icc}, {shock} cluster shock.",
                )
            )

    # 5) Null vs nonzero effect, both families that carry a relative-scale
    # sidecar (adjusted: exercises the DML/AIPW relative-lift nonorthogonality
    # the research flags as needing nonzero-effect calibration specifically).
    for effect in (0.0, 0.5, -0.5):
        cells.append(
            ManifestCell(
                name=f"effect_{effect}_adjusted",
                stratum="principal" if effect == 0.0 else "adversarial",
                family="adjusted",
                dgp=_base(20, 20, effect=effect, icc=0.2),
                description=f"additive effect={effect}, relative-lift sidecar exercised.",
            )
        )

    # 6) High leverage: one dominant control cluster carrying half that
    # arm's members.
    for k_t, k_c in ((10, 10), (5, 20)):
        cells.append(
            ManifestCell(
                name=f"leverage_{k_t}v{k_c}",
                stratum="adversarial",
                family="adjusted",
                dgp=_base(k_t, k_c, leverage=True, icc=0.2),
                description="one control cluster holds half that arm's members.",
            )
        )

    # Mixed clusters share a shock; ingress/reference acceptance remains open.
    for frac in (0.25, 0.5):
        for k_t, k_c in ((10, 10), (5, 20)):
            cells.append(
                ManifestCell(
                    name=f"mixed_{frac}_{k_t}v{k_c}",
                    stratum="adversarial",
                    family="adjusted",
                    dgp=_base(k_t, k_c, mixed_frac=frac, icc=0.2),
                    description=f"{frac:.0%} of arm-cluster pairs share a shock and cluster identity.",
                )
            )

    # 8) Admission boundaries: total K = 9, 10, 39, 40; per-arm K = 1 (must
    # refuse), 2 (minimum admissible).
    for k_t, k_c, tag in ((5, 4, "k9"), (5, 5, "k10"), (20, 19, "k39"), (20, 20, "k40")):
        cells.append(
            ManifestCell(
                name=f"admission_{tag}",
                stratum="adversarial",
                family="mean_ratio",
                dgp=_base(k_t, k_c, icc=0.2),
                description=f"total K={k_t + k_c}: {tag} admission boundary.",
            )
        )
    for k_t, k_c, tag in ((1, 9, "arm_k1"), (2, 8, "arm_k2")):
        cells.append(
            ManifestCell(
                name=f"admission_{tag}",
                stratum="adversarial",
                family="adjusted",
                dgp=_base(k_t, k_c, icc=0.2),
                description=f"per-arm K=({k_t},{k_c}): {tag} arm-count boundary (total K=10).",
            )
        )

    # 9) Sitewide: balanced and unbalanced K-pairs at two variance ratios,
    # for the sum-metric absolute-impact Welch fix.
    for k_t, k_c in ((10, 10), (5, 20)):
        for ratio in (1.0, 4.0):
            cells.append(
                ManifestCell(
                    name=f"sitewide_{k_t}v{k_c}_ratio{ratio}",
                    stratum="principal" if ratio == 1.0 else "adversarial",
                    family="sitewide",
                    dgp=_base(k_t, k_c, var_ratio=ratio, icc=0.2),
                    description=f"sitewide sum-metric absolute impact, K=({k_t},{k_c}), ratio={ratio}.",
                )
            )

    # Required intersections apply to each estimator family, not only adjustment.
    for family in ("mean_ratio", "late", "adjusted", "sitewide"):
        for kt, kc in ((2, 8), (3, 12), (5, 20), (8, 2), (12, 3), (20, 5)):
            for ratio in (4.0, 16.0, 0.25, 0.0625):
                cells.append(
                    ManifestCell(
                        f"intersection_{family}_{kt}v{kc}_{ratio}",
                        "adversarial",
                        family,
                        _base(
                            kt,
                            kc,
                            icc=0.8,
                            var_ratio=ratio,
                            first_stage=0.8 if family == "late" else None,
                        ),
                        "Small/unbalanced K with unequal variance; no post-hoc support change.",
                    )
                )
        for kt in (20, 80):
            cells.append(
                ManifestCell(
                    f"useful_{family}_{kt}",
                    "principal",
                    family,
                    _base(
                        kt, kt, icc=0.2, effect=2.0, first_stage=0.8 if family == "late" else None
                    ),
                    "Prespecified useful alternative and contracting-width pair.",
                )
            )
    for stage in (0.8, 0.05):
        cells.append(
            ManifestCell(
                f"late_first_stage_{stage}",
                "adversarial",
                "late",
                _base(5, 20, icc=0.2, var_ratio=4.0, first_stage=stage, effect=0.5),
                "Valid instrument, one-sided uptake; strong versus near-weak first stage.",
            )
        )
    for k in (40, 200):
        cells.append(
            ManifestCell(
                f"heterogeneous_{k}",
                "adversarial",
                "adjusted",
                _base(k, k, heterogeneous=True),
                "Superpopulation member ATE=.8, cluster ATE=.5, PLR=6/7.",
            )
        )
    return tuple(cells)


MANIFEST: tuple[ManifestCell, ...] = build_manifest()


def manifest_summary() -> str:
    """One line per cell, grouped by family -- the preregistration record the
    scientific-tolerance policy requires printed/logged before results exist."""
    lines = [f"I13 manifest: {len(MANIFEST)} cells"]
    for family in ("mean_ratio", "late", "adjusted", "sitewide"):
        fam_cells = [c for c in MANIFEST if c.family == family]
        principal = sum(1 for c in fam_cells if c.stratum == "principal")
        adversarial = len(fam_cells) - principal
        lines.append(
            f"  {family}: {len(fam_cells)} cells ({principal} principal, {adversarial} adversarial)"
        )
        for c in fam_cells:
            lines.append(f"    [{c.stratum[:4]}] {c.name}: {c.description}")
    return "\n".join(lines)


type EstimatorName = Literal["mean", "ratio", "late", "sitewide", "iptw", "aipw", "dml"]
type AcceptanceScale = Literal["relative", "absolute", "absolute_sidecar"]
type NuisanceMode = Literal["none", "oracle_propensity", "fitted"]


@dataclass(frozen=True)
class AcceptanceCell:
    """One actual estimator invocation and its prespecified scientific gates."""

    design: ManifestCell
    estimator: EstimatorName
    scale: AcceptanceScale
    nuisance: NuisanceMode = "none"
    folds: int = 2

    @property
    def name(self) -> str:
        return f"{self.design.name}/{self.estimator}/{self.scale}/{self.nuisance}/folds{self.folds}"

    @property
    def support(self) -> str:
        dgp = self.design.dgp
        if dgp.k_t + dgp.k_c < 10:
            return "existing_total_cluster_refusal"
        if self.estimator in ("aipw", "dml") and min(dgp.k_t, dgp.k_c) < self.folds:
            return "existing_crossfit_fold_refusal"
        if min(dgp.k_t, dgp.k_c) < 2:
            return "existing_arm_cluster_refusal"
        if dgp.mixed_frac:
            return "mixed_ingress_and_reference_unresolved"
        if dgp.first_stage is not None and dgp.first_stage < 0.1:
            return "near_weak_reference_unresolved"
        return "ordinary_available"

    @property
    def refusal_code(self) -> str | None:
        return {
            "existing_total_cluster_refusal": "estimation.engine.lift_guard",
            "existing_crossfit_fold_refusal": "estimation.adjust_overlap.arm_cluster_but",
            "existing_arm_cluster_refusal": "estimation.adjust_overlap.cluster_arm_needs_two",
        }.get(self.support)

    @property
    def rejection_rule(self) -> str:
        if self.scale == "absolute_sidecar" or self.estimator == "sitewide":
            return "zero exclusion by central absolute interval; no unsupported tail API"
        return "public primary p_value against zero, independently of target coverage"

    @property
    def bias_tolerance(self) -> float:
        # One percent of the baseline mean, on the corresponding reporting scale.
        return 0.01 if self.scale == "relative" else 0.05

    @property
    def useful(self) -> bool:
        return self.design.name.startswith("useful_")

    @property
    def width_limit(self) -> float:
        # Fourfold K should approximately halve width. Both limits are fixed here.
        limit = 1.5 if self.design.dgp.k_t == 20 else 0.9
        return limit / 5.0 if self.scale == "relative" else limit

    @property
    def quantities(self) -> tuple[str, ...]:
        common = ("coverage", "point_availability", "interval_availability", "bias", "finite_width")
        if self.useful:
            return (
                (*common, "power", "width_limit", "contracting_width")
                if self.design.dgp.k_t == 20
                else (*common, "power", "width_limit")
            )
        if self.design.dgp.effect == 0 and not self.design.dgp.heterogeneous:
            return (*common, "null_rejection")
        return common


def build_acceptance_manifest() -> tuple[AcceptanceCell, ...]:
    cells: list[AcceptanceCell] = []
    estimators_by_family: dict[str, tuple[EstimatorName, ...]] = {
        "mean_ratio": ("mean", "ratio"),
        "late": ("late",),
        "adjusted": ("iptw", "aipw", "dml"),
        "sitewide": ("sitewide",),
    }
    for design in MANIFEST:
        estimators = estimators_by_family[design.family]
        for estimator in estimators:
            scales: tuple[AcceptanceScale, ...] = (
                ("absolute",)
                if estimator in ("late", "sitewide")
                else ("relative", "absolute_sidecar")
            )
            if design.family == "adjusted":
                scales = ("relative", "absolute", "absolute_sidecar")
            nuisances: tuple[NuisanceMode, ...] = (
                ("oracle_propensity", "fitted") if design.family == "adjusted" else ("none",)
            )
            folds = (2, 5) if estimator in ("aipw", "dml") else (2,)
            for scale in scales:
                for nuisance in nuisances:
                    for fold in folds:
                        cells.append(AcceptanceCell(design, estimator, scale, nuisance, fold))
    return tuple(cells)


ACCEPTANCE_MANIFEST = build_acceptance_manifest()
GATED_QUANTITIES = tuple(
    (cell.name, quantity) for cell in ACCEPTANCE_MANIFEST for quantity in cell.quantities
)
# Frozen once; the release ledger must combine it with the other campaigns' allocations.
I13_FAMILY_ALPHA = 0.001
I13_ETA = I13_FAMILY_ALPHA / (2 * len(GATED_QUANTITIES))
