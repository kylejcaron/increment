"""Preregistered mixed-covariate linear CUPED certification (Revision 7).

Run ``uv run python -m calibration.cuped_linear --campaign smoke --output DIR``.
The full profile uses 107,100 eligible draws per statistic. Smoke exercises the
same designs, but cannot certify them. Coverage uses an exact binomial lower
bound; mean false-coverage proportion uses a bounded-observation KL upper bound.
Both spend the shared 67-statistic allocation, including the 12 sitewide slots
owned by the separate campaign. A failed gate never licenses a Normal interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.special import gammainc, gammaincc

from calibration.journal import DrawJournal, verify
from calibration.profile import CalibrationProfile, CellSet
from increment.errors import CodedError
from increment.estimation.cuped_linear import fit_cuped_linear
from tests.mc import coverage_lower_bound, family_eta, kl_chernoff_upper_bound, mcse

FAMILY_SIZE = 67
ETA = family_eta(0.01, FAMILY_SIZE)
FULL_REPLICATIONS = 107_100
SMOKE_REPLICATIONS = 3
ALPHA = 0.05
COVERAGE_FLOOR = 0.945
FCP_CEILING = 0.055
BOUNDARY_CANDIDATES = MappingProxyType(
    {
        "p": (203, 53, 8),
        "k_max": (200, 50, 5),
        "leverage_ratio": (0.20, 0.10, 0.05),
        "max_leverage": (0.45, 0.30, 0.15),
        "minimum_level_arm_count": (2, 5, 10),
    }
)
PROFILES = MappingProxyType(
    {
        name: CalibrationProfile(
            name=name,
            campaign="cuped_linear",
            tolerance="strict",
            delta=0.005,
            tolerance_fraction=Fraction(1, 200),
            cells=CellSet("all", "every-case", (), id_field="id", axis_field=None),
            stopping="fixed",
            sequential_rule=None,
        )
        for name in ("smoke", "full")
    }
)


@dataclass(frozen=True, slots=True)
class Cell:
    id: str
    statistic: str
    generator: str
    scale: str
    n: int
    allocation: float = 0.5
    k_country: int = 5
    encoded_p: int = 8
    sigma_den: float = 0.5
    effect: float = 0.05
    heterogeneous: bool = False
    failure_boundary: tuple[str, ...] = ()
    failure_consequence: str = "Refuse the named projection in its corresponding design region."


def _manifest() -> tuple[Cell, ...]:
    rows = []
    designs = ((400, 0.9), (1000, 0.9), (2000, 0.9), (2000, 0.5))
    for index, (n, allocation) in enumerate(designs, 1):
        rows.append(Cell(f"D3-{index}", "coverage", "mean", "absolute", n, allocation))
    for n, allocation in designs:
        for effect in (0.0, 0.05):
            index = sum(c.id.startswith("D3b-") for c in rows) + 1
            rows.append(
                Cell(
                    f"D3b-{index}",
                    "coverage",
                    "relative_mean",
                    "relative",
                    n,
                    allocation,
                    effect=effect,
                )
            )
    d4 = (
        (2000, 0.5, 50, 53, ("p", "k_max")),
        (4000, 0.9, 50, 53, ("max_leverage",)),
        (20000, 0.5, 200, 203, ("p", "k_max")),
        (4000, 0.5, 200, 203, ("minimum_level_arm_count",)),
        *((2000, 0.5, 5, p, ("leverage_ratio",)) for p in (50, 100, 200)),
        *((10000, 0.9, 5, p, ("leverage_ratio",)) for p in (50, 100, 200)),
        (4000, 0.9, 50, 53, ("max_leverage",)),
    )
    for index, (n, allocation, k, p, boundary) in enumerate(d4, 1):
        rows.append(
            Cell(
                f"D4-{index}",
                "coverage",
                "mean",
                "absolute",
                n,
                allocation,
                k,
                p,
                heterogeneous=index == 11,
                failure_boundary=boundary,
                failure_consequence="Tighten only the named observable boundary to its next preregistered candidate; refuse matrix support if the tightest candidate fails.",
            )
        )
    ratio_designs = ((100, 0.5), (100, 1.5), (1000, 0.5), (1000, 1.5))
    for index, (n, sigma) in enumerate(ratio_designs, 1):
        rows.append(
            Cell(
                f"D5-{index}",
                "coverage",
                "ratio",
                "ratio_absolute",
                2 * n,
                encoded_p=6,
                sigma_den=sigma,
            )
        )
    index = 0
    for n, sigma in ratio_designs:
        for effect in (0.0, 0.05):
            for scale in ("log_ratio", "ratio_absolute"):
                index += 1
                rows.append(
                    Cell(
                        f"D5b-{index}",
                        "coverage",
                        "ratio",
                        scale,
                        2 * n,
                        encoded_p=6,
                        sigma_den=sigma,
                        effect=effect,
                    )
                )
    index = 0
    for n, sigma in ratio_designs:
        for effect in (0.0, 0.05):
            index += 1
            rows.append(
                Cell(
                    f"D5b-FCR-{index}",
                    "fcp",
                    "ratio_family",
                    "relative_ratio",
                    2 * n,
                    encoded_p=6,
                    sigma_den=sigma,
                    effect=effect,
                    failure_consequence="Block BH-selected/reintervaled reporting for every D5b family.",
                )
            )
    for index, scale in enumerate(("absolute", "relative", "ratio_absolute"), 1):
        rows.append(Cell(f"D9-{index}", "coverage", "wls", scale, 2000, encoded_p=1))
    rows.append(
        Cell(
            "D9-4",
            "fcp",
            "wls_family",
            "mixed",
            2000,
            encoded_p=1,
            failure_consequence="Block the three-metric D9 BH output.",
        )
    )
    return tuple(rows)


MANIFEST = _manifest()


def manifest_record() -> dict:
    return {
        "revision": 7,
        "family_size": FAMILY_SIZE,
        "owned_statistics": {"coverage": 46, "fcp": 9},
        "excluded_statistics": "T1/T4: 12 coverage slots in the shared family",
        "eta": ETA,
        "eligible_replications": FULL_REPLICATIONS,
        "alpha": ALPHA,
        "coverage_floor": COVERAGE_FLOOR,
        "fcp_ceiling": FCP_CEILING,
        "support_region": {
            "p": 203,
            "k_max": 200,
            "leverage_ratio": 0.20,
            "max_leverage": 0.45,
            "ratio_denominator_precision": 0.15,
            "minimum_level_arm_count": 2,
            "full_rank": True,
            "positive_residual_df": True,
        },
        "boundary_candidates": dict(BOUNDARY_CANDIDATES),
        "seeds": {
            "design": "1000 + attempted_replication",
            "noise": "910000 + attempted_replication",
            "independent_metric": "1000 + attempted_replication + 10000000 * metric_index",
            "chunk": "attempted_replication = 100 * chunk_size * chunk_index + offset; unused reserved indices are not reassigned",
        },
        "attempt_limit": "100 * target eligible replications per cell; exhaustion refuses the region",
        "d9_design": "1000 units/arm, fixed allocation; x ~ N(0,1); y=10+2*x+effect*T+N(0,1); ratio adds .05*den to numerator for an exact absolute ratio shift",
        "ratio_truth": "Exact gamma/lognormal population integration with the generator's clipping; denominator sigma cancels, not a noisy large-sample target",
        "diagnostics": {
            "D8": "4000 eligible replications; variance ratios vs unadjusted and scalar x1; reported, never gated",
            "point_bias_width_leverage_dof_rejection": "Descriptive means with Monte Carlo standard errors; never gates",
        },
        "cells": [asdict(cell) for cell in MANIFEST],
    }


@dataclass(frozen=True)
class Draw:
    arms: np.ndarray
    covariates: dict
    kinds: dict
    outcomes: tuple[np.ndarray, ...]
    truth: float


def mean_design(cell: Cell, replication: int) -> Draw:
    """The C.3 law, preserving unavailable history as null, never dropping units."""
    rng = np.random.default_rng(1000 + replication)
    n = cell.n
    treatment = rng.random(n) < cell.allocation
    history = rng.random(n) < 0.7
    x1 = np.where(history, rng.gamma(2, 1, n), 0.0)
    x2 = np.where(history, rng.poisson(3, n).astype(float), 0.0)
    country = rng.integers(0, cell.k_country, n)
    device = rng.integers(0, 2, n)
    cmu = rng.normal(0, 1, cell.k_country) if cell.k_country != 5 else np.array([0, 1, 2, -1, 0.5])
    base = 2 * x1 + 0.5 * x2 + cmu[country] + 0.8 * device + 1.5 * history
    y = (
        base
        + (1 + 0.6 * treatment) * x1
        + (1 + 0.5 * device + 0.3 * x1) * treatment
        + rng.normal(size=n) * 1.5
    )
    truth = 2.51
    if cell.generator == "relative_mean":
        control_mean = 6.7 + float(cmu.mean())
        y += treatment * (control_mean * cell.effect - truth)
        truth = math.log1p(cell.effect)
    if cell.heterogeneous:
        y += 0.4 * treatment * x1 * ((country == 0) - 1 / cell.k_country)
    covariates = {
        "x1": np.where(history, x1, np.nan),
        "x2": np.where(history, x2, np.nan),
        "country": country.astype(str),
        "device": device.astype(str),
    }
    kinds = {
        "x1": "continuous",
        "x2": "continuous",
        "country": "categorical",
        "device": "categorical",
    }
    noise_p = cell.encoded_p - (cell.k_country + 3)
    if noise_p > 0:
        noise = np.random.default_rng(910000 + replication).normal(size=(n, noise_p))
        for column in range(noise_p):
            covariates[f"noise_{column}"] = noise[:, column]
            kinds[f"noise_{column}"] = "continuous"
    return Draw(np.where(treatment, "t", "c"), covariates, kinds, (y,), truth)


def population_ratio(rate_lift: float) -> float:
    """Exact E[num]/E[den] for C.3, integrating the clipped linear rate.

    Exponential tilting changes Gamma(2,1) history to Gamma(2,.6) with
    mass multiplier .6**-2. The independent lognormal multiplier cancels.
    Truncated zeroth/first moments use incomplete gamma functions; no
    Monte Carlo truth error is added to the coverage error model.
    """
    cmu = np.array([0.0, 0.3, -0.3, 0.1, 0.2])
    a = 0.2 + 0.02 * cmu + rate_lift
    threshold = (0.99 - a) / 0.03
    history_rate = a * gammainc(2, 0.6 * threshold)
    history_rate += 0.03 * (2 / 0.6) * gammainc(3, 0.6 * threshold)
    history_rate += 0.99 * gammaincc(2, 0.6 * threshold)
    numerator = np.exp(cmu) @ (0.3 * np.clip(a, 0.01, 0.99) + 0.7 / 0.6**2 * history_rate)
    denominator = np.exp(cmu).sum() * (0.3 + 0.7 / 0.6**2)
    return float(numerator / denominator)


def ratio_design(cell: Cell, replication: int) -> Draw:
    rng = np.random.default_rng(1000 + replication)
    n = cell.n
    treatment = np.arange(n) < n // 2
    history = rng.random(n) < 0.7
    x1 = np.where(history, rng.gamma(2, 1, n), 0.0)
    country = rng.integers(0, 5, n)
    cmu = np.array([0, 0.3, -0.3, 0.1, 0.2])
    denominator = np.exp(rng.normal(0, cell.sigma_den, n) + 0.4 * x1 + cmu[country])
    rate = np.clip(0.2 + 0.03 * x1 + 0.02 * cmu[country] + cell.effect * treatment, 0.01, 0.99)
    numerator = denominator * rate + rng.normal(0, 0.1, n) * denominator
    rt, rc = population_ratio(cell.effect), population_ratio(0.0)
    truth = (
        math.log(rt) - math.log(rc) if cell.scale in ("log_ratio", "relative_ratio") else rt - rc
    )
    return Draw(
        np.where(treatment, "t", "c"),
        {"x1": np.where(history, x1, np.nan), "country": country.astype(str)},
        {"x1": "continuous", "country": "categorical"},
        (numerator, denominator),
        truth,
    )


def wls_design(cell: Cell, replication: int) -> Draw:
    rng = np.random.default_rng(1000 + replication)
    treatment = np.arange(cell.n) < cell.n // 2
    x = rng.normal(size=cell.n)
    effect = cell.effect * 10 if cell.scale == "relative" else cell.effect
    y = 10 + 2 * x + effect * treatment + rng.normal(size=cell.n)
    if cell.scale == "ratio_absolute":
        denominator = np.exp(0.1 * x + rng.normal(scale=0.2, size=cell.n))
        numerator = denominator * (0.3 + cell.effect * treatment) + rng.normal(
            scale=0.1, size=cell.n
        )
        outcomes = (numerator, denominator)
    else:
        outcomes = (y,)
    truth = math.log1p(cell.effect) if cell.scale == "relative" else cell.effect
    return Draw(np.where(treatment, "t", "c"), {"x": x}, {"x": "continuous"}, outcomes, truth)


def generate(cell: Cell, replication: int) -> Draw:
    if cell.generator in ("mean", "relative_mean"):
        return mean_design(cell, replication)
    if cell.generator in ("ratio", "ratio_family"):
        return ratio_design(cell, replication)
    return wls_design(cell, replication)


def fit_draw(draw: Draw):
    return fit_cuped_linear(
        draw.outcomes[0] if len(draw.outcomes) == 1 else None,
        draw.arms,
        control="c",
        arm_order=("c", "t"),
        covariates=draw.covariates,
        kinds=draw.kinds,
        missing=dict.fromkeys(draw.covariates, "impute-indicator"),
        numerator=draw.outcomes[0] if len(draw.outcomes) == 2 else None,
        denominator=draw.outcomes[1] if len(draw.outcomes) == 2 else None,
    )


def support_violations(fit, draw: Draw) -> dict:
    """Audit returned evidence; never turn a missing production guard into eligibility."""
    limits = manifest_record()["support_region"]
    k_max = max((len(level) for level in fit.layout.levels if level is not None), default=0)
    observables = {
        "p": fit.p,
        "k_max": k_max,
        "leverage_ratio": fit.p / fit.n_min,
        "max_leverage": fit.max_leverage,
    }
    if len(draw.outcomes) == 2:
        den = draw.outcomes[1]
        observables["ratio_denominator_precision"] = max(
            float(
                den[draw.arms == a].std(ddof=1)
                / math.sqrt(sum(draw.arms == a))
                / den[draw.arms == a].mean()
            )
            for a in ("c", "t")
        )
    return {
        key: {"value": value, "bound": limits[key]}
        for key, value in observables.items()
        if value > limits[key]
    }


def projection(fit, scale: str, alpha: float = ALPHA):
    if scale == "absolute":
        return fit.absolute("t", alpha=alpha)
    if scale == "relative":
        return fit.relative("t", alpha=alpha)
    return fit.ratio(
        "t", scale="log" if scale in ("log_ratio", "relative_ratio") else "absolute", alpha=alpha
    )


def coverage_trial(cell: Cell, replication: int) -> dict:
    draw = generate(cell, replication)
    fit = fit_draw(draw)
    violations = support_violations(fit, draw)
    if violations:
        return {
            "status": "blocked",
            "reason": "production returned a fit outside M1",
            "violations": violations,
            "replication": replication,
            "seed": 1000 + replication,
        }
    result = projection(fit, cell.scale)
    diagnostics = {
        "bias": result.estimate - draw.truth,
        "width": result.upper - result.lower,
        "max_leverage": fit.max_leverage,
        "dof": result.dof,
        "rejection": float(not result.lower <= 0 <= result.upper),
    }
    if cell.id == "D8":
        diagnostics.update(precision_diagnostics(draw, result.se**2))
    return {
        "status": "eligible",
        "covered": result.lower <= draw.truth <= result.upper,
        "diagnostics": diagnostics,
    }


def precision_diagnostics(draw: Draw, adjusted_variance: float) -> dict:
    from increment.estimation.armstats import ArmStats
    from increment.estimation.cuped import fit_cuped

    x = np.nan_to_num(draw.covariates["x1"])
    y = draw.outcomes[0]
    stats = []
    for arm in ("c", "t"):
        mask = draw.arms == arm
        xx, yy = x[mask], y[mask]
        stats.append(
            ArmStats.from_raw_sums(
                study_id="s",
                metric="mean",
                group_id=arm,
                n=int(mask.sum()),
                sum_y=float(yy.sum()),
                sum_y2=float(yy @ yy),
                sum_x=float(xx.sum()),
                sum_x2=float(xx @ xx),
                sum_xy=float(xx @ yy),
            )
        )
    scalar = fit_cuped(stats)
    unadjusted = sum(y[draw.arms == arm].var(ddof=1) / sum(draw.arms == arm) for arm in ("c", "t"))
    scalar_variance = sum(scalar.residual_var(arm, scalar.theta) / arm.n for arm in stats)
    return {
        "variance_ratio_unadjusted": adjusted_variance / unadjusted,
        "variance_ratio_scalar_x1": adjusted_variance / scalar_variance,
    }


def family_members(cell: Cell, replication: int) -> tuple[tuple[Draw, str], ...]:
    count = 2 if cell.generator == "ratio_family" else 3
    members = []
    for metric in range(count):
        if count == 2:
            member = Cell(
                "member",
                "coverage",
                "ratio",
                "relative_ratio",
                cell.n,
                encoded_p=6,
                sigma_den=cell.sigma_den,
                effect=cell.effect if metric == 0 else 0.0,
            )
        else:
            scale = ("absolute", "relative", "ratio_absolute")[metric]
            member = Cell(
                "member",
                "coverage",
                "wls",
                scale,
                cell.n,
                encoded_p=1,
                effect=0.05 if metric == 0 else 0.0,
            )
        members.append((generate(member, replication + 10_000_000 * metric), member.scale))
    return tuple(members)


def family_trial(cell: Cell, replication: int) -> dict:
    """Exercise production declared-secondary BH selection and retained reintervaling."""
    import pandas as pd

    from increment import Analysis, Method
    from increment.breakout.estimates import LiftEstimates
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.models import AnalysisPlan

    members = family_members(cell, replication)
    count = len(members)
    draws, scales = [], []
    for metric, (draw, scale) in enumerate(members):
        fit = fit_draw(draw)
        violations = support_violations(fit, draw)
        if violations:
            return {
                "status": "blocked",
                "reason": "production returned a fit outside M1",
                "violations": violations,
                "replication": replication,
                "metric": metric,
            }
        draws.append(draw)
        scales.append(scale)
    frame = {"unit": np.arange(cell.n), "arm": draws[0].arms}
    specs = []
    for metric, draw in enumerate(draws):
        name = f"metric_{metric}"
        columns = {f"{name}_{key}": value for key, value in draw.covariates.items()}
        frame.update(columns)
        declarations = tuple(
            CupedCovariate(name=f"{name}_{key}", kind=kind, missing="impute-indicator")
            for key, kind in draw.kinds.items()
        )
        kwargs = {
            "name": name,
            "covariates": declarations,
            "decision_method": Method(name="linear", variance_reduction="cuped_linear"),
        }
        if len(draw.outcomes) == 2:
            frame[f"{name}_num"], frame[f"{name}_den"] = draw.outcomes
            kwargs.update(type="ratio", numerator=f"{name}_num", denominator=f"{name}_den")
        else:
            frame[name] = draw.outcomes[0]
        specs.append(MetricSpec.model_validate(kwargs))
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(frame),
        unit="unit",
        group="arm",
        control="c",
        metrics=specs,
        plan=AnalysisPlan(secondaries=tuple(s.name for s in specs), q=ALPHA, alpha=ALPHA),
    )
    returned = analysis.run()
    if not isinstance(returned, LiftEstimates):
        return {"status": "blocked", "reason": "production did not return fixed-horizon lift rows"}
    rows = list(returned)
    if len(rows) != count:
        return {"status": "blocked", "reason": "incomplete production family", "emitted": len(rows)}
    selected = [row for row in rows if row.discovery]
    misses = 0
    for row in selected:
        index = next(i for i, spec in enumerate(specs) if spec.name == row.metric)
        if scales[index] in ("absolute", "ratio_absolute"):
            lower, upper, truth = row.abs_lb, row.abs_ub, draws[index].truth
        else:
            lift = row.require_lift()
            lower, upper, truth = lift.lb, lift.ub, math.expm1(draws[index].truth)
        if lower is None or upper is None:
            return {
                "status": "blocked",
                "reason": "selected gated-scale interval unavailable",
                "metric": row.metric,
            }
        expected_alpha = min(ALPHA * len(selected) / count, ALPHA)
        actual_alpha = (
            row.abs_alpha
            if scales[index] in ("absolute", "ratio_absolute")
            else row.require_lift().alpha
        )
        if actual_alpha is None or not math.isclose(actual_alpha, expected_alpha, rel_tol=1e-14):
            return {
                "status": "blocked",
                "reason": "selected gated-scale interval not reissued",
                "actual_alpha": actual_alpha,
                "expected_alpha": expected_alpha,
                "metric": row.metric,
            }
        misses += not lower <= truth <= upper
    return {
        "status": "eligible",
        "fcp": misses / max(len(selected), 1),
        "selected": len(selected),
        "misses": misses,
        "diagnostics": {"selected": float(len(selected)), "fcp": misses / max(len(selected), 1)},
    }


REFUSAL_CODES = frozenset(
    "estimation.cuped_linear." + name
    for name in (
        "support_region",
        "rank_deficiency",
        "level_support",
        "residual_df",
        "leverage",
        "arm_counts",
        "nonpositive_component",
        "projection_unavailable",
    )
)


def _context_value(value):
    if isinstance(value, Mapping):
        return {key: _context_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_context_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def trial(cell: Cell, replication: int) -> dict:
    try:
        return (
            family_trial(cell, replication)
            if cell.statistic == "fcp"
            else coverage_trial(cell, replication)
        )
    except CodedError as error:
        status = "refused" if error.code in REFUSAL_CODES else "blocked"
        return {
            "status": status,
            "code": error.code,
            "context": _context_value(error.context),
            "replication": replication,
            "seed": 1000 + replication,
        }


def _atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    import scipy

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_paths = (
        sorted((root / "increment").rglob("*.py"))
        + sorted((root / "calibration").glob("*.py"))
        + [root / "tests" / "mc.py"]
    )
    source_digest = hashlib.sha256()
    for path in source_paths:
        source_digest.update(path.relative_to(root).as_posix().encode())
        source_digest.update(path.read_bytes())
    return {
        "source_sha": source_sha,
        "source_digest": source_digest.hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "threads": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
    }


def _accumulate(state: dict, result: dict) -> tuple[str, ...]:
    status = result["status"]
    state["attempted"] += 1
    state[status] += 1
    tally = [status]
    if status == "refused":
        code = result["code"]
        state["refusal_codes"][code] = state["refusal_codes"].get(code, 0) + 1
        tally.append(code)
    elif status == "eligible":
        if "covered" in result:
            state["hits"] += result["covered"]
            if result["covered"]:
                tally.append("covered")
        else:
            state["fcp_sum"] += result["fcp"]
            tally.append(f"fcp:{result['misses']}/{max(result['selected'], 1)}")
        for key, value in result["diagnostics"].items():
            moment = state["diagnostics"].setdefault(key, [0.0, 0.0])
            moment[0] += value
            moment[1] += value * value
    elif status == "blocked":
        state["concern"] = result
    return tuple(tally)


def run_chunk(job: tuple) -> dict:
    cell, directory, chunk, target, chunk_size, identity = job
    directory = Path(directory) / cell.id / "chunks" / f"{chunk:06d}"
    checkpoint = directory / "checkpoint.json"
    binding = {
        "identity": identity,
        "cell": asdict(cell),
        "chunk": chunk,
        "target": target,
        "chunk_size": chunk_size,
    }
    binding = json.loads(json.dumps(binding))
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        totals = verify(directory, case_id=f"{cell.id}:{chunk}")
        if record["binding"] != binding or record["journal_digest"] != totals.digest:
            raise ValueError(f"incompatible checkpoint: {checkpoint}; use a new output directory")
        return record
    started = time.perf_counter()
    cpu_started = time.process_time()
    state = {
        "attempted": 0,
        "eligible": 0,
        "refused": 0,
        "blocked": 0,
        "hits": 0,
        "fcp_sum": 0.0,
        "refusal_codes": {},
        "diagnostics": {},
    }
    start = chunk * chunk_size * 100
    with DrawJournal(directory, case_id=f"{cell.id}:{chunk}", block_size=chunk_size) as journal:
        committed = journal.totals().completed
        replay_tallies = Counter()
        for offset in range(chunk_size * 100):
            replication = start + offset
            if offset >= committed:
                journal.started(replication)
            try:
                result = trial(cell, replication)
            except Exception as error:
                if offset >= committed:
                    journal.completed(
                        replication,
                        {"exception": type(error).__name__, "message": str(error)},
                        tally=("failed",),
                        exceptional=True,
                    )
                raise
            tally = _accumulate(state, result)
            if offset < committed:
                replay_tallies.update(tally)
                if offset + 1 == committed and dict(replay_tallies) != dict(
                    journal.totals().counters
                ):
                    raise ValueError("replayed draws disagree with the committed journal")
            else:
                journal.completed(
                    replication, result, tally=tally, exceptional=result["status"] != "eligible"
                )
            if state["eligible"] == target or state["blocked"]:
                break
        totals = journal.close()
    state.update(
        binding=binding,
        journal_digest=totals.digest,
        elapsed_seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu_started,
        status="blocked"
        if state["blocked"]
        else "complete"
        if state["eligible"] == target
        else "refused_region",
    )
    _atomic_json(checkpoint, state)
    return state


def summarize(cell: Cell, records: list[dict], target: int, campaign: str) -> dict:
    eligible = sum(record["eligible"] for record in records)
    attempted = sum(record["attempted"] for record in records)
    elapsed = sum(record["elapsed_seconds"] for record in records)
    cpu = sum(record["cpu_seconds"] for record in records)
    status = (
        "complete"
        if eligible == target
        else "blocked"
        if any(r["blocked"] for r in records)
        else "refused_region"
    )
    diagnostics = {}
    keys = {key for record in records for key in record["diagnostics"]}
    for key in keys:
        total = sum(record["diagnostics"].get(key, [0, 0])[0] for record in records)
        squares = sum(record["diagnostics"].get(key, [0, 0])[1] for record in records)
        mean = total / eligible
        se = (
            math.sqrt(max(0, (squares - total * mean) / (eligible - 1) / eligible))
            if eligible > 1
            else None
        )
        diagnostics[key] = {"mean": mean, "mcse": se, "gated": False}
        if key == "rejection":
            hits = int(round(total))
            diagnostics[key]["confidence_interval_95"] = (
                coverage_lower_bound(hits, eligible, 0.025),
                1 - coverage_lower_bound(eligible - hits, eligible, 0.025),
            )
        elif key == "fcp":
            diagnostics[key]["confidence_interval_95"] = (
                1 - kl_chernoff_upper_bound(1 - mean, eligible, 0.025),
                kl_chernoff_upper_bound(mean, eligible, 0.025),
            )
    bound = observed = diagnostic_mcse = None
    if eligible and cell.statistic == "coverage":
        hits = sum(record["hits"] for record in records)
        observed = hits / eligible
        bound = coverage_lower_bound(hits, eligible, ETA)
        diagnostic_mcse = mcse(observed, eligible)
    elif eligible and cell.statistic == "fcp":
        observed = sum(record["fcp_sum"] for record in records) / eligible
        bound = kl_chernoff_upper_bound(observed, eligible, ETA)
        diagnostic_mcse = diagnostics["fcp"]["mcse"]
    passed = bound is not None and (
        bound >= COVERAGE_FLOOR if cell.statistic == "coverage" else bound <= FCP_CEILING
    )
    evaluated = campaign == "full" and status == "complete" and cell.statistic != "descriptive"
    certified = evaluated and passed
    return {
        "cell": cell.id,
        "statistic": cell.statistic,
        "status": status,
        "eligible": eligible,
        "refused": sum(r["refused"] for r in records),
        "attempted": attempted,
        "observed": observed,
        "mcse": diagnostic_mcse,
        "bound": bound,
        "gate_passed": passed if evaluated else None,
        "certified": certified,
        "elapsed_seconds": elapsed,
        "seconds_per_attempt": elapsed / attempted if attempted else None,
        "cpu_seconds": cpu,
        "cpu_seconds_per_attempt": cpu / attempted if attempted else None,
        "cpu_seconds_per_eligible": cpu / eligible if eligible else None,
        "seconds_per_eligible": elapsed / eligible if eligible else None,
        "projected_full_cpu_hours": cpu
        / eligible
        * (4000 if cell.id == "D8" else FULL_REPLICATIONS)
        / 3600
        if status == "complete"
        else None,
        "diagnostics": diagnostics,
        "refusal_codes": dict(sum((Counter(r["refusal_codes"]) for r in records), Counter())),
        "concerns": [r["concern"] for r in records if "concern" in r],
    }


def run_campaign(output: Path, campaign: str, workers: int = 1, chunk_size: int = 100) -> dict:
    if campaign not in PROFILES or workers < 1 or chunk_size < 1:
        raise ValueError("unknown campaign or nonpositive workers/chunk size")
    target = FULL_REPLICATIONS if campaign == "full" else SMOKE_REPLICATIONS
    chunk_size = min(chunk_size, target)
    identity = _identity()
    frozen = manifest_record()
    binding = {
        "manifest": frozen,
        "identity": identity,
        "campaign": campaign,
        "chunk_size": chunk_size,
        "profile": {
            "name": campaign,
            "tolerance": PROFILES[campaign].tolerance,
            "stopping": "fixed",
            "certification": PROFILES[campaign].certification(frozen["cells"]),
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
        json.dumps(binding)
    ):
        raise ValueError("output belongs to a different frozen source/profile; use a new directory")
    _atomic_json(manifest_path, binding)
    precision = Cell(
        "D8",
        "descriptive",
        "mean",
        "absolute",
        2000,
        failure_consequence="Precision is descriptive; no acceptance threshold.",
    )
    cells = (*MANIFEST, precision)
    targets = {cell.id: target for cell in MANIFEST}
    targets["D8"] = 4000 if campaign == "full" else SMOKE_REPLICATIONS
    jobs = [
        (
            cell,
            str(output),
            chunk,
            min(chunk_size, targets[cell.id] - chunk * chunk_size),
            chunk_size,
            identity,
        )
        for cell in cells
        for chunk in range(math.ceil(targets[cell.id] / chunk_size))
    ]
    started = time.perf_counter()
    if workers == 1:
        records = [run_chunk(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(run_chunk, jobs))
    summaries = []
    for cell in cells:
        summary = summarize(
            cell,
            [r for r in records if r["binding"]["cell"]["id"] == cell.id],
            targets[cell.id],
            campaign,
        )
        summaries.append(summary)
        _atomic_json(output / cell.id / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    result = {
        "campaign": campaign,
        "identity": identity,
        "elapsed_seconds": time.perf_counter() - started,
        "cells": summaries[:-1],
        "precision": summaries[-1],
        "full_campaign_executed": campaign == "full",
        "projected_full_cpu_hours": sum(s["projected_full_cpu_hours"] or 0 for s in summaries),
        "projection_complete": all(s["projected_full_cpu_hours"] is not None for s in summaries),
        "status": "blocked"
        if any(s["status"] != "complete" for s in summaries)
        else "failed"
        if campaign == "full" and any(not s["gate_passed"] for s in summaries[:-1])
        else "smoke_complete"
        if campaign == "smoke"
        else "certified",
    }
    _atomic_json(output / "summary.json", result)
    return result


def admission_trial(cell: Cell, replication: int) -> dict:
    """Estimate support admission only; this cannot replace production refusals."""
    draws = (
        family_members(cell, replication)
        if cell.statistic == "fcp"
        else ((generate(cell, replication), cell.scale),)
    )
    violations = {}
    for metric, (draw, _) in enumerate(draws):
        try:
            fit = fit_draw(draw)
            for name, values in support_violations(fit, draw).items():
                violations[f"metric_{metric}:{name}"] = values
        except CodedError as error:
            if error.code not in REFUSAL_CODES:
                raise
            violations[f"metric_{metric}:{error.code}"] = _context_value(error.context)
    return {"admitted": not violations, "violations": violations, "seed": 1000 + replication}


def run_admission_cell(job: tuple) -> dict:
    cell, output, attempts, identity = job
    directory = Path(output) / cell.id
    checkpoint = directory / "checkpoint.json"
    binding = json.loads(
        json.dumps({"cell": asdict(cell), "attempts": attempts, "identity": identity})
    )
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text())
        totals = verify(directory, case_id=cell.id)
        if record["binding"] != binding or record["journal_digest"] != totals.digest:
            raise ValueError(
                "incompatible support-admission checkpoint; use a new output directory"
            )
        return record
    counters = Counter()
    started, cpu_started = time.perf_counter(), time.process_time()
    with DrawJournal(directory, case_id=cell.id, block_size=100) as journal:
        committed = journal.totals().completed
        for replication in range(attempts):
            if replication >= committed:
                journal.started(replication)
            result = admission_trial(cell, replication)
            tally = (
                "admitted" if result["admitted"] else "not_admitted",
                *sorted(result["violations"]),
            )
            counters.update(tally)
            if replication >= committed:
                journal.completed(
                    replication, result, tally=tally, exceptional=not result["admitted"]
                )
            elif replication + 1 == committed and dict(counters) != dict(journal.totals().counters):
                raise ValueError("support-admission replay disagrees with its journal")
        totals = journal.close()
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    admitted = counters["admitted"]
    fraction = admitted / attempts
    lower = coverage_lower_bound(admitted, attempts, 0.025)
    upper = 1 - coverage_lower_bound(attempts - admitted, attempts, 0.025)
    cpu_per_attempt = cpu / attempts
    result = {
        "cell": cell.id,
        "attempts": attempts,
        "admitted": admitted,
        "admitted_fraction": fraction,
        "confidence_interval_95": [lower, upper],
        "projected_attempts": FULL_REPLICATIONS / fraction if fraction else None,
        "projected_attempts_interval_95": [
            FULL_REPLICATIONS / upper if upper else None,
            FULL_REPLICATIONS / lower if lower else None,
        ],
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu,
        "cpu_seconds_per_attempt": cpu_per_attempt,
        "projected_cpu_hours": cpu_per_attempt * FULL_REPLICATIONS / fraction / 3600
        if fraction
        else None,
        "projected_cpu_hours_interval_95": [
            cpu_per_attempt * FULL_REPLICATIONS / upper / 3600 if upper else None,
            cpu_per_attempt * FULL_REPLICATIONS / lower / 3600 if lower else None,
        ],
        "over_ten_attempts_per_eligible": fraction < 0.1,
        "violations": {
            key: value
            for key, value in sorted(counters.items())
            if key not in ("admitted", "not_admitted")
        },
        "gated": False,
        "cost_scope": "fit and support admission only; excludes public BH/reintervaling overhead",
        "binding": binding,
        "journal_digest": totals.digest,
    }
    _atomic_json(checkpoint, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run_admission_probe(output: Path, attempts: int, workers: int) -> dict:
    if attempts < 1 or workers < 1:
        raise ValueError("attempts and workers must be positive")
    identity = _identity()
    binding = {
        "identity": identity,
        "manifest": manifest_record(),
        "attempts": attempts,
        "scope": "descriptive support-admission probe, not certification",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(binding)):
        raise ValueError("support-admission output belongs to another source; use a new directory")
    _atomic_json(path, binding)
    jobs = [(cell, str(output), attempts, identity) for cell in MANIFEST]
    started = time.perf_counter()
    if workers == 1:
        records = [run_admission_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(run_admission_cell, jobs))
    result = {
        "scope": binding["scope"],
        "identity": identity,
        "cells": records,
        "elapsed_seconds": time.perf_counter() - started,
        "projected_cpu_hours": sum(record["projected_cpu_hours"] or 0 for record in records),
        "projection_complete": all(record["projected_cpu_hours"] is not None for record in records),
        "over_ten_attempts_cells": [
            record["cell"] for record in records if record["over_ten_attempts_per_eligible"]
        ],
        "full_campaign_executed": False,
    }
    _atomic_json(output / "summary.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--eligibility-attempts",
        type=int,
        default=0,
        help="separate descriptive support-admission probe with contiguous frozen seeds; no gates",
    )
    args = parser.parse_args(argv)
    if args.eligibility_attempts:
        if args.campaign != "smoke":
            parser.error("the descriptive eligibility probe is separate from the full campaign")
        run_admission_probe(args.output.resolve(), args.eligibility_attempts, args.workers)
        return 0
    result = run_campaign(args.output.resolve(), args.campaign, args.workers, args.chunk_size)
    return 0 if result["status"] in ("smoke_complete", "certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
