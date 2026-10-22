"""Fixed-horizon BH and absolute known-variance Gaussian e-BH references,
plus the calibrated deployed-route family gate.

The original zero-centered one-sample Gaussian DGP, seeds, repetitions and
error budgets are retained. Its exact likelihood-ratio mixture is valid for
that absolute model. It is not evidence for the deployed relative Gaussian
runtime: those studies drive ``e_bh_select`` on hand-built log-e values and
so never reach registered evidence, the immutable roster, checkpoint
verification or a stopping time.

Study 5 closes that gap. It runs the deployed route end to end --
``capture_sequential_snapshot`` -> ``estimate_sequential`` ->
``select_sequential_family`` -- over repeated appended looks under a global
null, retains a never-enrolled roster cell, stops at the first look that
selects, and accepts the resulting stopped false discovery rate on an exact
one-sided Clopper-Pearson upper bound against ``Q`` plus the
scientific-tolerance policy's absolute excess allowance.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from increment.decision import (
    ArmHypothesisKey,
    DecisionComputation,
    HypothesisKey,
    PValueEvidence,
)
from increment.estimation.family import e_bh_select, select_family
from increment.estimation.results import Estimate, LiftEstimate
from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

Q = 0.10
M = 20
N_NULL = 10  # first N_NULL indices are true nulls; the rest carry EFFECT_*
N_WRONG = 5  # Study 3: the next N_WRONG indices carry -EFFECT_EBH under "greater"

# Study 1 (fixed-horizon BH) DGP.
EFFECT_BH = 0.15  # true log-lift for the M - N_NULL non-nulls (~16% relative lift)
SIGMA_BH = 0.05  # per-hypothesis posterior SE (typical of a single-look estimate)

# Study 2 (AlwaysValid e-BH) DGP: per-unit noise accumulating over LOOKS
# sequential batches of BATCH_N draws each.
EFFECT_EBH = 2.0  # true per-unit mean for the M - N_NULL non-nulls
SIGMA_EBH = 1.0  # per-unit noise sd
LOOKS = 8
BATCH_N = 2
EFFECT_SCALE = 1.0  # Absolute Gaussian predictive mean standard deviation

# Study 4 (deployed sequential family) DGP: a three-arm Bernoulli roster whose
# every cell is a true null, revealed in DEPLOYED_BATCH-unit joint batches.
DEPLOYED_ARMS = ("treatment", "variant_b", "variant_c")
DEPLOYED_RATE = 0.4  # shared control/treatment success probability: global null
DEPLOYED_LOOKS = 2
DEPLOYED_BATCH = 12  # units per arm per look
DEPLOYED_LIFTED_RATE = 0.99  # the one non-null arm in the falsifying replication
# ``variant_c`` never enrolls, so its cell stays in the family as a retained
# missing cell and e-BH keeps dividing by the full roster size.
DEPLOYED_ABSENT = DEPLOYED_ARMS[2:]

# Study 5 (calibrated stopped FDR) sizing. Units per revealed arm per look,
# so the family is evaluated at 16, 24 and 32 units per arm.
STOPPED_FDR_SCHEDULE = (16, 8, 8)
STOPPED_FDR_REPS = 120
# The repository's .01 family-wise Monte-Carlo decision error, split across both
# directional halves of the one rate in this file that is accepted on an
# exact confidence statement. The four studies above use fixed tolerances,
# not confidence bounds, and so spend nothing from this budget.
STOPPED_FDR_ETA = family_eta(0.01, 1)
# Absolute excess-discovery allowance at nominal Q: min(.005, .1*Q).
STOPPED_FDR_DELTA = scientific_delta(Q)


def _fixed_horizon_cell(idx: int, mu: float, sigma: float) -> tuple[ArmHypothesisKey, LiftEstimate]:
    """One typed p-value evidence cell with a Normal posterior."""
    z95 = 1.959963984540054
    row = LiftEstimate(
        metric=f"m{idx}",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            log_mean=mu,
            log_se=sigma,
            value=math.expm1(mu),
            lb=math.expm1(mu - z95 * sigma),
            ub=math.expm1(mu + z95 * sigma),
            level=0.95,
        ),
    )
    return ArmHypothesisKey(row.metric, row.group_id, "itt"), row


def _bh_family_fdp(
    rng: np.random.Generator, m: int, n_null: int, effect: float, sigma: float, q: float
) -> float:
    """One replication of study 1: draw a p-value per hypothesis, run
    select_family's fixed-horizon branch, return the realized false
    discovery proportion (0.0 when nothing was selected)."""
    is_null = np.arange(m) < n_null
    mu_true = np.where(is_null, 0.0, effect)
    mu_hat = mu_true + sigma * rng.standard_normal(m)
    cells = [_fixed_horizon_cell(i, mu_hat[i], sigma) for i in range(m)]
    typed = [(key, row) for key, row in cells]
    selected = select_family(
        typed,
        q=q,
        inference=None,
        nominal_alpha=0.05,
        computation=_typed_computation(typed),
    ).selected
    n_sel = len(selected)
    if n_sel == 0:
        return 0.0
    n_false = sum(1 for key in selected if is_null[int(key.metric[1:])])
    return n_false / n_sel


def _typed_computation(
    cells: list[tuple[ArmHypothesisKey, LiftEstimate]],
) -> DecisionComputation[LiftEstimate]:
    evidence: dict[HypothesisKey, PValueEvidence] = {}
    for key, row in cells:
        p_value = row.p_value()
        assert p_value is not None
        evidence[key] = PValueEvidence(key, row.method, p_value, "normal")
    return DecisionComputation[LiftEstimate](
        results=tuple(row for _, row in cells),
        evidence=evidence,
        failures={},
    )


def _ebh_selected_by_look(
    rng: np.random.Generator,
    mu_true: np.ndarray,
    sigma: float,
    q: float,
    looks: int,
    batch_n: int,
    mean_prior_sd: float,
    *,
    alternative: str = "two-sided",
) -> list[np.ndarray]:
    """Exact absolute Gaussian likelihood mixture on the original unit draws."""
    from scipy.special import log_ndtr

    m = len(mu_true)
    sums = np.zeros(m)
    masks = []
    for look in range(looks):
        batch = mu_true[:, None] + sigma * rng.standard_normal((m, batch_n))
        sums += batch.sum(axis=1)
        n = (look + 1) * batch_n
        variance = sigma * sigma / n
        mean = sums / n
        shrink = mean_prior_sd**2 / (variance + mean_prior_sd**2)
        log_e = -0.5 * np.log1p(mean_prior_sd**2 / variance) + 0.5 * mean**2 / variance * shrink
        if alternative == "greater":
            log_e += np.log(2.0) + log_ndtr(mean / np.sqrt(variance) * np.sqrt(shrink))
        selected = e_bh_select([float(value) for value in log_e], q)
        mask = np.zeros(m, dtype=bool)
        for index in selected:
            mask[index] = True
        masks.append(mask)
    return masks


def _fdp(selected: np.ndarray, is_false: np.ndarray) -> float:
    """Realized false discovery proportion (0.0 when nothing was selected)."""
    n_sel = int(selected.sum())
    return float((selected & is_false).sum() / n_sel) if n_sel else 0.0


def _ebh_family_fdps_by_look(
    rng: np.random.Generator,
    m: int,
    n_null: int,
    effect: float,
    sigma: float,
    q: float,
    looks: int,
    batch_n: int,
    mean_prior_sd: float,
) -> list[float]:
    """One replication of study 2. Returns one realized FDP per look."""
    is_null = np.arange(m) < n_null
    mu_true = np.where(is_null, 0.0, effect)
    return [
        _fdp(selected, is_null)
        for selected in _ebh_selected_by_look(rng, mu_true, sigma, q, looks, batch_n, mean_prior_sd)
    ]


def _directional_ebh_rates_by_look(
    rng: np.random.Generator,
    m: int,
    n_null: int,
    n_wrong: int,
    effect: float,
    sigma: float,
    q: float,
    looks: int,
    batch_n: int,
    mean_prior_sd: float,
) -> list[tuple[float, float]]:
    """One replication of study 3 under `alternative="greater"`: indices
    `[0, n_null)` are true nulls, `[n_null, n_null + n_wrong)` carry
    `-effect` (inside the composite null), the rest carry `+effect`.
    Returns one (composite-null FDP, wrong-direction selected share) per
    look."""
    idx = np.arange(m)
    is_null = idx < n_null
    is_wrong = (idx >= n_null) & (idx < n_null + n_wrong)
    mu_true = np.where(is_null, 0.0, np.where(is_wrong, -effect, effect))
    return [
        (_fdp(selected, is_null | is_wrong), float((selected & is_wrong).sum() / n_wrong))
        for selected in _ebh_selected_by_look(
            rng, mu_true, sigma, q, looks, batch_n, mean_prior_sd, alternative="greater"
        )
    ]


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_bh_realized_fdr_is_controlled():
    """1500 independent replications, m=20 (10 true null, 10 a moderate true
    lift), q=0.10. Under independence BH(1995) bounds FDR at (m0/m)*q =
    0.05; measured mean realized FDR ~0.053, SE ~0.0018 at this rep count
    -- allow to 0.08 for Monte-Carlo headroom, still well under the
    declared q=0.10 itself."""
    rng = np.random.default_rng(20260819)
    fdps = np.array([_bh_family_fdp(rng, M, N_NULL, EFFECT_BH, SIGMA_BH, Q) for _ in range(1500)])
    mean_fdr = float(fdps.mean())
    assert mean_fdr <= 0.08, mean_fdr


def test_bh_realized_fdr_is_controlled_smoke():
    """Small-N smoke twin of the parameter_recovery check above."""
    rng = np.random.default_rng(11)
    fdps = np.array([_bh_family_fdp(rng, M, N_NULL, EFFECT_BH, SIGMA_BH, Q) for _ in range(30)])
    mean_fdr = float(fdps.mean())
    assert mean_fdr <= 0.18, mean_fdr


@pytest.mark.parameter_recovery
@pytest.mark.slow
def test_absolute_gaussian_ebh_realized_fdr_is_controlled_across_repeated_looks():
    """Original 800-replication absolute Gaussian e-BH acceptance case."""
    rng = np.random.default_rng(20260819)
    all_fdps: list[float] = []
    for _ in range(800):
        all_fdps.extend(
            _ebh_family_fdps_by_look(
                rng, M, N_NULL, EFFECT_EBH, SIGMA_EBH, Q, LOOKS, BATCH_N, EFFECT_SCALE
            )
        )
    mean_fdr = float(np.mean(all_fdps))
    assert mean_fdr <= 0.03, mean_fdr


@pytest.mark.slow
def test_absolute_gaussian_ebh_realized_fdr_is_controlled_across_repeated_looks_smoke():
    """Small-N smoke twin of the parameter_recovery check above: fewer
    replications and fewer looks per replication, loose bound."""
    rng = np.random.default_rng(11)
    all_fdps: list[float] = []
    for _ in range(30):
        all_fdps.extend(
            _ebh_family_fdps_by_look(
                rng, M, N_NULL, EFFECT_EBH, SIGMA_EBH, Q, 4, BATCH_N, EFFECT_SCALE
            )
        )
    mean_fdr = float(np.mean(all_fdps))
    assert mean_fdr <= 0.06, mean_fdr


@pytest.mark.parameter_recovery
@pytest.mark.slow
def test_absolute_gaussian_ebh_one_sided_wrong_direction_is_not_selected_across_repeated_looks():
    """Original directional composite-null and wrong-direction budgets."""
    rng = np.random.default_rng(20260915)
    rates: list[tuple[float, float]] = []
    for _ in range(800):
        rates.extend(
            _directional_ebh_rates_by_look(
                rng, M, N_NULL, N_WRONG, EFFECT_EBH, SIGMA_EBH, Q, LOOKS, BATCH_N, EFFECT_SCALE
            )
        )
    mean_fdr, mean_wrong_share = (float(v) for v in np.mean(rates, axis=0))
    assert mean_fdr <= 0.03, mean_fdr
    assert mean_wrong_share <= 0.005, mean_wrong_share


@pytest.mark.slow
def test_absolute_gaussian_ebh_one_sided_wrong_direction_is_not_selected_across_repeated_looks_smoke():
    """Small-N smoke twin of the parameter_recovery check above."""
    rng = np.random.default_rng(11)
    rates: list[tuple[float, float]] = []
    for _ in range(30):
        rates.extend(
            _directional_ebh_rates_by_look(
                rng, M, N_NULL, N_WRONG, EFFECT_EBH, SIGMA_EBH, Q, 4, BATCH_N, EFFECT_SCALE
            )
        )
    mean_fdr, mean_wrong_share = (float(v) for v in np.mean(rates, axis=0))
    assert mean_fdr <= 0.06, mean_fdr
    assert mean_wrong_share <= 0.02, mean_wrong_share


def _deployed_family_outcome(
    seed: int,
    *,
    lifted: str | None = None,
    schedule: tuple[int, ...] = (DEPLOYED_BATCH,) * DEPLOYED_LOOKS,
    absent: tuple[str, ...] = (),
):
    """One replication through the deployed sequential family route.

    Reveals one appended joint batch per entry of ``schedule`` with
    ``capture_sequential_snapshot``, evaluates each prefix with the public
    ``estimate_sequential``, and selects with ``select_sequential_family`` --
    the registered-roster e-BH path the runtime itself takes, carrying
    likelihood evidence, checkpoint verification and the immutable roster,
    rather than a bare ``e_bh_select`` call on hand-built log-e values.
    Peeks after every batch and stops at the first look that selects, so the
    returned outcome is a genuine stopped state and not a fixed-horizon one.

    Every revealed arm draws Bernoulli(``DEPLOYED_RATE``); ``lifted`` names
    the single arm drawn at ``DEPLOYED_LIFTED_RATE`` instead, which is what
    keeps a zero-discovery claim falsifiable. Arms in ``absent`` reveal no
    units at all and stay in the roster as retained missing cells, so e-BH
    keeps dividing by the full family size.
    """
    from fractions import Fraction

    from increment import (
        AlwaysValid,
        SequentialCell,
        capture_sequential_snapshot,
        estimate_sequential,
    )
    from increment.estimation.decision_types import sequential_hypothesis_key
    from increment.estimation.family import select_sequential_family
    from tests.sequential_cases import registration

    reg = registration(
        "bernoulli",
        cells=tuple(
            SequentialCell(metric="outcome", group_id=arm, family=True) for arm in DEPLOYED_ARMS
        ),
    )
    arms = tuple(arm for arm in (reg.control_group, *DEPLOYED_ARMS) if arm not in absent)
    policy = AlwaysValid(registration=reg)
    rng = np.random.default_rng(seed)
    snapshot = outcome = None
    offset = 0
    for batch in schedule:
        rates = [DEPLOYED_LIFTED_RATE if arm == lifted else DEPLOYED_RATE for arm in arms]
        draws = rng.binomial(1, rates, size=(batch, len(arms)))
        snapshot = capture_sequential_snapshot(
            reg,
            [
                {
                    "unit_id": f"{offset + unit:06d}-{arm}",
                    "group_id": arm,
                    "values": {"outcome": int(draws[unit, index])},
                    "segments": {},
                }
                for unit in range(batch)
                for index, arm in enumerate(arms)
            ],
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=snapshot,
            append=snapshot is not None,
        )
        offset += batch
        bundle = estimate_sequential(snapshot, policy)
        cells = [
            (sequential_hypothesis_key(row.require_sequential_result().checkpoint.cell), row)
            for row in bundle.results
        ]
        outcome = select_sequential_family(
            cells, reg.q, policy, Fraction(1, 20), computation=bundle
        )
        if outcome.selected:
            break
    return outcome


def test_deployed_sequential_family_retains_missing_cells_and_stops_on_discovery():
    """Fast-tier deployed-route witness, unmarked so the default tier is
    never left with ``e_bh_select`` studies as its only family evidence.

    Two global-null replications and one lifted replication of a three-arm
    Bernoulli family run through ``capture_sequential_snapshot`` ->
    ``estimate_sequential`` -> ``select_sequential_family``, peeking after
    each appended batch. ``variant_c`` never enrolls, so what this pins is
    that the retained missing cell still counts in ``m``: the lifted
    replication's selected-interval level must be ``Q * R / 3``, not
    ``Q * R / 2``, which is the arithmetic a roster that quietly dropped
    absent cells would produce.

    The lifted replication is also the falsifiability anchor -- without it
    "selected nothing" would pass on a route that can never select. Three
    replications bound no rate;
    ``test_deployed_sequential_family_stopped_fdr_is_bounded_under_the_global_null``
    owns the calibrated bound.
    """
    roster = len(DEPLOYED_ARMS)
    outcomes = [_deployed_family_outcome(seed, absent=DEPLOYED_ABSENT) for seed in (404, 505)]
    assert [outcome.n_family for outcome in outcomes] == [roster] * len(outcomes)
    assert [len(outcome.canonical_method) for outcome in outcomes] == [roster] * len(outcomes)
    stopped_fdp = [1.0 if outcome.selected else 0.0 for outcome in outcomes]
    assert float(np.mean(stopped_fdp)) == 0.0, stopped_fdp
    assert all(outcome.fcr_alpha is None for outcome in outcomes)
    assert all(outcome.realized_threshold is None for outcome in outcomes)
    assert all(outcome.q == Q for outcome in outcomes)

    lifted = _deployed_family_outcome(808, lifted=DEPLOYED_ARMS[0], absent=DEPLOYED_ABSENT)
    assert {key.group_id for key in lifted.selected} == {DEPLOYED_ARMS[0]}
    # Benjamini-Yekutieli selected-interval level for R of m discoveries, with
    # m the COMPLETE roster including the cell that never enrolled.
    assert lifted.realized_threshold == pytest.approx(Q * len(lifted.selected) / roster)
    # Registration q is the exact Fraction(1, 10) that Q names as a float.
    assert lifted.fcr_alpha == Fraction(1, 10) * len(lifted.selected) / roster


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_deployed_sequential_family_stopped_fdr_is_bounded_under_the_global_null():
    """Calibrated stopped FDR through the deployed sequential family route.

    Call chain per look, per replication: ``capture_sequential_snapshot``
    (appended joint prefix, parent-verified) -> public
    ``estimate_sequential`` (registered roster, ``EValueEvidence``, allocated
    alpha) -> ``select_sequential_family`` (roster order, per-cell
    ``verify_snapshot``, single-prefix check, then e-BH). Nothing here calls
    ``e_bh_select`` on hand-built log-e values, which is what the four
    studies above do and why none of them constrains the deployed route.

    Every cell is a true null, so a replication's realized stopped false
    discovery proportion is 1.0 the moment the route selects anything and
    0.0 otherwise: the stopped FDR equals the probability that the stopped
    family makes any discovery. e-BH holds that at ``Q`` under an arbitrary
    stopping time because each cell's evidence is an e-process, so the
    prospective bound is ``CP_U(discoveries, reps, STOPPED_FDR_ETA) <=
    Q + STOPPED_FDR_DELTA``, exact and one-sided, never a plug-in SE band.

    ``variant_c`` never enrolls throughout, so every replication also
    certifies that a retained missing cell keeps its place in ``m`` while
    the family is being calibrated -- a roster that dropped it would test
    two hypotheses at the thresholds of three and inflate exactly the rate
    this gate bounds.

    Sizing, recorded rather than hidden: at 120 replications the acceptance
    tolerates up to 4 stopped discoveries (``CP_U(4, 120, .005) = .1013``,
    ``CP_U(5, 120, .005) = .1135``). A PROSPECTIVE margin of .0025 at
    ``p = Q`` would need roughly 100,000 replications, which is days at this
    route's cost; the bound asserted here is exactly true at 120.
    """
    outcomes = [
        _deployed_family_outcome(seed, schedule=STOPPED_FDR_SCHEDULE, absent=DEPLOYED_ABSENT)
        for seed in range(9000, 9000 + STOPPED_FDR_REPS)
    ]
    roster = len(DEPLOYED_ARMS)
    # The retained missing cell never leaves the family, at any look.
    assert {outcome.n_family for outcome in outcomes} == {roster}
    assert {len(outcome.canonical_method) for outcome in outcomes} == {roster}
    # Unselected stopped states report no selected-interval level at all.
    assert all((outcome.fcr_alpha is None) == (not outcome.selected) for outcome in outcomes)
    stopped_fdp = [1.0 if outcome.selected else 0.0 for outcome in outcomes]
    discoveries = int(sum(stopped_fdp))
    upper = binomial_error_upper_bound(discoveries, len(outcomes), STOPPED_FDR_ETA)
    assert upper <= Q + STOPPED_FDR_DELTA, (discoveries, len(outcomes), upper)
