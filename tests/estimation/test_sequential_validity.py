"""False-positive rate under continuous peeking.

Preserves the zero-control absolute Gaussian known-variance reference and
its original draws. Bernoulli cases exercise the deployed raw likelihood.
Gaussian reference results do not certify the relative unknown-variance runtime.
The empirical false-positive criterion counts replications that ever reject.
Passing that criterion is evidence under the specified DGP, not a universal
optional-stopping proof. Tight cases retain their parameter_recovery markers.
Deployed Bernoulli rates are accepted on the exact one-sided Clopper-Pearson
upper bound against the scientific-tolerance policy's absolute excess-error
budget, never on a plug-in ``alpha + k*SE`` band.

``TestDeployedRuntimeNulls`` extends that accounting to the two deployed
runtimes the absolute Gaussian reference never reached: the scalar NIG law
and the bivariate NIW ratio law. Each drives its own registered route, so an
inflated repeated-look rejection rate in either of them now fails a gate
instead of going unobserved.
"""

from fractions import Fraction
from typing import NamedTuple

import numpy as np
import pytest

from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

ALPHA = 0.05
# A .01 family-wise MC decision error splits across each directional bound of
# six null rates: four in ``TestTight`` and two in ``TestDeployedRuntimeNulls``.
# A new gate must raise this count, or it spends the family budget twice.
GATED_RATES = 6
ETA = family_eta(0.01, GATED_RATES)
# Absolute excess-rejection allowance at nominal ALPHA: min(.005, .1*ALPHA).
DELTA = scientific_delta(ALPHA)


def _absolute_gaussian_fpr(reps, looks, batch, seed, *, shift=0.0, directional=False):
    """Known-variance absolute mean difference; preserves the original DGP."""
    from increment.estimation.sequential import GaussianScoreMixture
    from increment.power.sequential import planning_bounds

    rng = np.random.default_rng(seed)
    t = rng.standard_normal((reps, looks)) * np.sqrt(batch) + shift * batch
    c = rng.standard_normal((reps, looks)) * np.sqrt(batch)
    n = batch * np.arange(1, looks + 1)
    difference = (np.cumsum(t, axis=1) - np.cumsum(c, axis=1)) / n - shift
    se = np.sqrt(2.0 / n)
    bounds = np.array(
        planning_bounds(
            GaussianScoreMixture(),
            [k / looks for k in range(1, looks + 1)],
            float(se[-1]),
            ALPHA,
            "upper" if directional else "both",
        )
    )
    statistic = difference / se if directional else np.abs(difference) / se
    return float((statistic > bounds).any(axis=1).mean())


class _RawNullOutcome(NamedTuple):
    """Deployed Bernoulli replication counts under continuous peeking."""

    rejections: int
    public_decisions: int
    reps: int


def _raw_null_rejections(
    reps: int,
    looks: int,
    batch: int,
    seed: int,
    *,
    control_rate: float = 0.3,
    treatment_rate: float = 0.3,
) -> _RawNullOutcome:
    """Count replications whose deployed Bernoulli decision ever rejects.

    Every prefix meets the certified noncrossing screen first. The screen
    bounds the deployed evidence from above, so a prefix it resolves provably
    cannot reach ``1 / alpha`` and needs no runtime call at all -- that is the
    performance win, and it is what keeps the quadratic prefix replay off the
    hot path. Every prefix the screen leaves UNRESOLVED is decided on the
    deployed public route: snapshot capture, ``estimate_sequential``, the
    registration's own alpha allocation, result construction and
    ``stat_sig()``. A recorded rejection is therefore always a deployed
    decision and never a screen artifact; ``public_decisions`` records how
    many prefixes the runtime actually decided.
    """
    from fractions import Fraction
    from functools import cache

    from increment import AlwaysValid, estimate_sequential
    from increment.estimation._sequential_likelihood import BernoulliState, BetaPrior
    from tests.estimation._sequential_events import bernoulli_noncrossing_screen
    from tests.sequential_cases import capture, records, registration

    reg = registration("bernoulli", alpha=ALPHA)
    policy = AlwaysValid(registration=reg)
    prior = BetaPrior(1, 1)
    alpha = reg.roster[0].alpha
    rng = np.random.default_rng(seed)
    # Preserve the original binomial batch-count draws and their ordering.
    treatment = rng.binomial(batch, treatment_rate, size=(reps, looks))
    control = rng.binomial(batch, control_rate, size=(reps, looks))

    @cache
    def proves_noncrossing(n: int, control_successes: int, treatment_successes: int) -> bool:
        """Certified noncrossing for one exact pair of prefix statistics.

        At a fixed prior, ratio, alternative and alpha the screen is a pure
        function of the two Bernoulli sufficient statistics, and a
        repeated-look grid revisits the same triple often: the tight gate's
        100,000 prefixes carry only 42,198 distinct triples. Memoizing them
        is decision-neutral and takes that run from 65.2 s to 57.1 s.
        """
        pooled = Fraction(control_successes + treatment_successes, 2 * n)
        return bernoulli_noncrossing_screen(
            BernoulliState(n, control_successes),
            BernoulliState(n, treatment_successes),
            prior,
            prior,
            ratio=Fraction(1),
            alternative="two-sided",
            feasible_control=pooled,
            feasible_treatment=pooled,
            alpha=alpha,
        ).proves_noncrossing

    rejected = decided = 0
    for trial in range(reps):
        control_successes = treatment_successes = 0
        for look in range(looks):
            control_successes += int(control[trial, look])
            treatment_successes += int(treatment[trial, look])
            n = (look + 1) * batch
            if proves_noncrossing(n, control_successes, treatment_successes):
                continue
            rows = records(
                [1] * control_successes + [0] * (n - control_successes),
                [1] * treatment_successes + [0] * (n - treatment_successes),
            )
            decided += 1
            if estimate_sequential(capture(reg, rows), policy).results[0].stat_sig():
                rejected += 1
                break
    return _RawNullOutcome(rejected, decided, reps)


def _null_fpr_always_valid(reps: int, looks: int, batch: int, seed: int) -> float:
    return _absolute_gaussian_fpr(reps, looks, batch, seed)


def _null_fpr_fixed_horizon(reps: int, looks: int, batch: int, seed: int) -> float:
    """The unprotected baseline: repeated z-tests."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((reps, looks)) * np.sqrt(batch)
    c = rng.standard_normal((reps, looks)) * np.sqrt(batch)
    n = batch * np.arange(1, looks + 1)
    diff = (np.cumsum(t, axis=1) - np.cumsum(c, axis=1)) / n
    z = np.abs(diff) / np.sqrt(2.0 / n)
    return float((z > 1.959964).any(axis=1).mean())


def _null_fpr_always_valid_one_sided_shifted(
    reps: int, looks: int, batch: int, seed: int, shift: float
) -> float:
    return _absolute_gaussian_fpr(reps, looks, batch, seed, shift=shift, directional=True)


# prose: allow-long binary64 quadrature bound and null-support conditions justify these nulls
# --- Deployed runtime nulls: NIG, NIW ratio -------------------------------
# The Gaussian score-mixture reference and the Beta/Bernoulli helper above do
# not cover the registered NIG and NIW ratio runtimes, so each gets its own
# null calibration.
#
# Observations are exact binary64 draws, ``Fraction(float)``, never re-rounded,
# so the only discretization of the continuous registered laws is binary64
# spacing. ``E[Q / sup-null L] <= 1`` then holds up to that lattice's
# midpoint-quadrature error, ``O((h / sigma)**2)`` per coordinate: about
# ``2**-104`` at ``h / sigma ~ 2**-52``, twenty-nine orders inside DELTA.
#
# Both null DGPs satisfy the registered support: the scalar control mean, the
# control ratio and both denominator means are positive. A true parameter
# outside the declared null set would push the sup-null denominator below the
# true likelihood, and the measured rate would calibrate nothing.
RUNTIME_LOOKS = 4
RUNTIME_BATCH = 4
GAUSSIAN_NULL = (1.0, 1.0)  # scalar population mean, sd
RATIO_NULL = (0.25, 1.0, 0.5)  # numerator mean, denominator mean, shared sd


class _RuntimeNullOutcome(NamedTuple):
    """Deployed registered-runtime counts under repeated looks."""

    rejections: int
    runtime_decisions: int
    finite_certificates: int
    max_log_e: float
    reps: int


def _registered_gaussian_null(
    law: str,
    reps: int,
    seed: int,
    *,
    looks: int = RUNTIME_LOOKS,
    batch: int = RUNTIME_BATCH,
    treatment_mean: float | None = None,
) -> _RuntimeNullOutcome:
    """Count replications whose registered Gaussian runtime ever rejects.

    Exercises the route the runtime itself takes for these laws:
    ``_capture_sequential_diagnostic_snapshot`` builds and parent-verifies
    each appended joint prefix, ``_evaluate_sequential_diagnostic`` walks the
    registered roster, constructs the checkpoint, asks the policy for the
    allocated alpha and evaluates it, and ``SequentialResult.rejects()``
    makes the decision. ``treatment_mean`` replaces the treatment location
    (scalar) or numerator location (ratio) to drive the same route under an
    alternative.

    ``runtime_decisions`` and ``finite_certificates`` record how many prefixes
    the runtime actually decided and how many produced finite evidence rather
    than an abstention, so a rate of zero cannot be an artifact of a route
    that abstained throughout.
    """
    from increment import AlwaysValid
    from increment.estimation.sequential_runtime import _evaluate_sequential_diagnostic
    from increment.sequential_state import _capture_sequential_diagnostic_snapshot
    from tests.sequential_cases import records, registration

    reg = registration(law)
    policy = AlwaysValid(registration=reg)
    rng = np.random.default_rng(seed)
    rejections = decisions = finite = 0
    largest = -np.inf

    def draw(location: float, scale: float, size: int) -> list[Fraction]:
        return [Fraction(float(value)) for value in rng.normal(location, scale, size)]

    for _ in range(reps):
        snapshot = None
        offset = 0
        for _look in range(looks):
            if law == "gaussian":
                mean, sd = GAUSSIAN_NULL
                control = draw(mean, sd, batch)
                treatment = draw(mean if treatment_mean is None else treatment_mean, sd, batch)
            else:
                numerator, denominator, sd = RATIO_NULL
                control = list(
                    zip(draw(numerator, sd, batch), draw(denominator, sd, batch), strict=True)
                )
                treatment = list(
                    zip(
                        draw(numerator if treatment_mean is None else treatment_mean, sd, batch),
                        draw(denominator, sd, batch),
                        strict=True,
                    )
                )
            snapshot = _capture_sequential_diagnostic_snapshot(
                reg,
                records(control, treatment, offset=offset),
                source_id=reg.source_id,
                definitions_id=reg.definitions_id,
                finalized=True,
                previous=snapshot,
                append=snapshot is not None,
            )
            offset += batch
            result = _evaluate_sequential_diagnostic(snapshot, policy)[0]
            decisions += 1
            if result.certificate.status == "finite":
                finite += 1
                largest = max(largest, float(result.log_e))
            if result.rejects():
                rejections += 1
                break
    return _RuntimeNullOutcome(rejections, decisions, finite, float(largest), reps)


def test_always_valid_bernoulli_public_route_smoke():
    """Fast-tier deployed witness: the small-N twin of the calibrated
    Bernoulli gate in ``TestTight``.

    The calibrated runs are `slow`/`parameter_recovery`, so without this case
    the default tier carries no deployed statistical witness at all. A tied
    null stream must yield no deployed rejection, and a fully separated
    stream must reject THROUGH the runtime -- snapshot capture,
    ``estimate_sequential``, the registered alpha allocation and
    ``stat_sig()``. Pinning ``public_decisions`` is what stops the certified
    screen from silently becoming the whole decision procedure. Twelve
    replications cannot bound a rate (the exact one-sided bound at zero
    rejections is ~0.43 there); ``TestTight`` owns the calibrated gate.
    """
    null = _raw_null_rejections(reps=12, looks=2, batch=3, seed=5)
    assert null.rejections == 0
    separated = _raw_null_rejections(
        reps=1, looks=1, batch=5, seed=3, control_rate=0.0, treatment_rate=1.0
    )
    assert (separated.public_decisions, separated.rejections) == (1, 1)


@pytest.mark.parametrize("law", ["gaussian", "gaussian_ratio"])
def test_registered_gaussian_runtime_public_facade_refuses_its_own_law(law):
    """The registered NIG and NIW ratio runtimes are not public anytime
    validity, and the calibrated gates below must not be read as if they
    were.

    ``capture_sequential_snapshot`` and the public ``estimate_sequential``
    both refuse these laws, so ``_capture_sequential_diagnostic_snapshot``
    plus ``_evaluate_sequential_diagnostic`` IS the deployed route for them
    -- it is what the certification campaign drives. Pinning the refusal
    keeps that statement true: if the public facade ever admits these laws,
    this fails and the calibrated gates have to be re-pointed at the public
    route rather than silently understating what they cover.
    """
    from increment import AlwaysValid, capture_sequential_snapshot
    from increment import estimate_sequential as public_estimate_sequential
    from increment.errors import CapabilityError
    from increment.sequential_state import _capture_sequential_diagnostic_snapshot
    from tests.sequential_cases import records, registration

    reg = registration(law)
    rows = records([(1, 1)], [(2, 1)]) if law == "gaussian_ratio" else records([1], [2])
    with pytest.raises(CapabilityError) as refused:
        capture_sequential_snapshot(
            reg,
            rows,
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
        )
    assert refused.value.code == "sequential.route.unsupported"
    snapshot = _capture_sequential_diagnostic_snapshot(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
    )
    with pytest.raises(CapabilityError) as refused:
        public_estimate_sequential(snapshot, AlwaysValid(registration=reg))
    assert refused.value.code == "sequential.route.unsupported"


def test_registered_scalar_gaussian_runtime_route_smoke():
    """Fast-tier deployed witness for the NIG runtime.

    Its calibrated gate is `parameter_recovery`, so without this the default
    tier has no witness that the scalar Gaussian runtime decides anything at
    all. A tied null stream must produce finite evidence and no rejection; a
    separated stream must reject THROUGH the same route. Three replications
    bound no rate -- ``TestDeployedRuntimeNulls`` owns the calibrated gate.
    """
    null = _registered_gaussian_null("gaussian", reps=3, seed=5)
    assert null.rejections == 0
    assert null.finite_certificates == null.runtime_decisions == 3 * RUNTIME_LOOKS
    separated = _registered_gaussian_null("gaussian", reps=1, seed=7, treatment_mean=8.0)
    assert separated.rejections == 1
    assert separated.max_log_e > -np.log(ALPHA)


def test_registered_ratio_gaussian_runtime_route_smoke():
    """Fast-tier deployed witness for the NIW ratio runtime.

    The NIW ratio evidence is strongly conservative at every count this
    runtime can be driven to: over the 2,000 prefixes its calibrated gate
    decides, the largest null evidence was ``exp(-6.16)`` against a
    threshold of ``1 / alpha = 20``. A random separated stream does not
    reject at that scale either, so the falsifying witness is a deterministic
    separated stream revealed over four appended looks. It ends at 81 units
    per arm with a treatment-to-control ratio near 5.7 and rejects through
    ``_capture_sequential_diagnostic_snapshot`` ->
    ``_evaluate_sequential_diagnostic`` -> ``rejects()``. The first look
    carries a single unit, so the singular-prefix abstention is crossed on
    the way rather than routed around.
    """
    from increment import AlwaysValid
    from increment.estimation.sequential_runtime import _evaluate_sequential_diagnostic
    from increment.sequential_state import _capture_sequential_diagnostic_snapshot
    from tests.sequential_cases import records, registration

    reg = registration("gaussian_ratio")
    policy = AlwaysValid(registration=reg)
    control_block = [(1, 1), (2, 1), (1, 2), (2, 2)]
    treatment_block = [(8, 1), (9, 1), (8, 2), (9, 2)]
    reveals = [([(1, 1)], [(2, 1)])]
    for repeats in (7, 7, 6):
        reveals.append((control_block * repeats, treatment_block * repeats))
    snapshot = None
    offset = 0
    statuses = []
    for control, treatment in reveals:
        snapshot = _capture_sequential_diagnostic_snapshot(
            reg,
            records(control, treatment, offset=offset),
            source_id=reg.source_id,
            definitions_id=reg.definitions_id,
            finalized=True,
            previous=snapshot,
            append=snapshot is not None,
        )
        offset += len(control)
        result = _evaluate_sequential_diagnostic(snapshot, policy)[0]
        statuses.append((result.certificate.status, result.rejects()))
    assert statuses[0] == ("zero", False)
    assert result.checkpoint.control.n == 81
    assert statuses[-1] == ("finite", True)

    null = _registered_gaussian_null("gaussian_ratio", reps=3, seed=11)
    assert null.rejections == 0
    assert null.finite_certificates == null.runtime_decisions == 3 * RUNTIME_LOOKS


@pytest.mark.slow
class TestSmoke:
    """Original small-N variants with their unchanged tolerances."""

    def test_always_valid_bounds_fpr(self):
        fpr = _null_fpr_always_valid(reps=800, looks=10, batch=50, seed=7)
        assert fpr <= ALPHA + 0.01  # AVI is conservative; MC se ~ 0.008

    def test_fixed_horizon_peeking_inflates(self):
        # Sanity contrast: the unprotected interval must inflate well past alpha.
        fpr = _null_fpr_fixed_horizon(reps=800, looks=10, batch=50, seed=13)
        assert fpr > 0.10

    def test_always_valid_estimated_variance_bernoulli_fpr(self):
        """The repo's other validity sims draw known-unit-variance normals;
        this exercises the actual Beta/Bernoulli likelihood and inversion.
        No estimated SE enters the certified evidence, and the acceptance is
        the exact one-sided Clopper-Pearson upper bound against the
        scientific-tolerance policy's absolute excess-rejection budget rather
        than a plug-in SE band."""
        outcome = _raw_null_rejections(reps=300, looks=10, batch=50, seed=17)
        upper = binomial_error_upper_bound(outcome.rejections, outcome.reps, ETA)
        assert upper <= ALPHA + DELTA, (outcome, upper)

    def test_always_valid_one_sided_shifted_null_bounds_fpr(self):
        """The absolute Gaussian reference is centered at the declared shift."""
        fpr = _null_fpr_always_valid_one_sided_shifted(
            reps=800, looks=10, batch=50, seed=19, shift=-0.02
        )
        assert fpr <= ALPHA

    def test_always_valid_one_sided_fpr_is_invariant_to_the_null_location(self):
        """Retain the original shifted-null rate and cross-location tolerance."""
        rates = [
            _null_fpr_always_valid_one_sided_shifted(reps=800, looks=10, batch=50, seed=19, shift=s)
            for s in (0.0, -0.02, 0.05)
        ]
        assert all(r <= ALPHA for r in rates)
        # Keep the original acceptance tolerance; this needs fresh execution.
        assert max(rates) - min(rates) < 0.005


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestTight:
    def test_always_valid_bounds_fpr_many_looks(self):
        fpr = _null_fpr_always_valid(reps=20_000, looks=40, batch=25, seed=101)
        assert fpr <= ALPHA  # anytime-valid: bounded at alpha, typically well under

    def test_always_valid_estimated_variance_bernoulli_fpr_tight(self):
        """Deployed Bernoulli null calibration at the registered alpha.

        Acceptance is the scientific-tolerance policy's absolute
        excess-rejection budget read off the exact one-sided Clopper-Pearson
        upper bound, ``CP_U(k, reps, ETA) <= ALPHA + DELTA``. The superseded
        ``ALPHA + 3*SE`` band accepted up to .0603, which is neither the
        .055 budget nor any stated confidence statement.

        Sizing caveat, deliberately recorded rather than hidden: at 4,000
        replications the PROSPECTIVE Monte-Carlo margin at the nominal
        acceptance boundary is .01175, well above the scientific-tolerance
        policy's .0025 precision requirement; reaching .00244 prospectively
        needs 82,577 replications (~21 min here, past the 300 s
        parameter_recovery ceiling). The bound asserted below is still exactly
        true at this repetition count -- it is the precision of a
        boundary-valued truth, not the validity of the bound, that remains a
        campaign-sizing decision.
        """
        outcome = _raw_null_rejections(reps=4000, looks=25, batch=50, seed=107)
        # Every recorded rejection came off the deployed route; if the screen
        # ever resolved the whole grid this gate would certify nothing.
        assert outcome.public_decisions > 0
        upper = binomial_error_upper_bound(outcome.rejections, outcome.reps, ETA)
        assert upper <= ALPHA + DELTA, (outcome, upper)

    def test_always_valid_one_sided_shifted_null_fpr_many_looks(self):
        fpr = _null_fpr_always_valid_one_sided_shifted(
            reps=20_000, looks=40, batch=25, seed=109, shift=-0.02
        )
        assert fpr <= ALPHA


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestDeployedRuntimeNulls:
    """Repeated-look null calibration for the two deployed runtimes the
    absolute Gaussian reference never reached.

    Each rate is accepted on the exact one-sided Clopper-Pearson upper bound
    against the scientific-tolerance policy's absolute excess-rejection
    budget, ``CP_U(k, reps, ETA) <= ALPHA + DELTA``, and each shares the one
    .01 family-wise budget declared in ``GATED_RATES``. No rate here is
    accepted on a plug-in SE band, and none of them re-derives a boundary:
    the decision is whatever the registered runtime returns at the alpha the
    registration allocates.
    """

    def test_registered_scalar_gaussian_null_fpr_across_repeated_looks(self):
        """NIG runtime, four appended looks, 4 to 16 units per arm.

        Route: ``_capture_sequential_diagnostic_snapshot`` ->
        ``_evaluate_sequential_diagnostic`` -> ``rejects()``. The null draws
        both arms i.i.d. ``N(1, 1)`` as exact binary64 rationals, so the
        declared positive control population mean holds and the composite
        null contains the truth.

        Sensitivity, measured rather than assumed: across the 2,400 prefixes
        this gate decides, the largest null evidence observed was
        ``exp(-0.023)``, against the rejection threshold ``1 / alpha = 20``.
        The gate therefore fails on any regression that inflates the
        deployed scalar evidence by more than a factor of 21, which is a
        genuinely tight boundary for a null calibration at this cost. A
        boundary-tight PROSPECTIVE bound -- one whose Monte-Carlo margin at
        ``p = ALPHA`` meets the scientific-tolerance policy's .0025 -- would
        need 82,577 replications, ~4.5 h here and far past the 300 s
        ceiling. The bound asserted below is exactly true at 600
        replications; it is its precision at a boundary-valued truth, not its
        validity, that the campaign owns.
        """
        outcome = _registered_gaussian_null("gaussian", reps=600, seed=211)
        # A route that abstained throughout, or that stopped after one look,
        # would report zero rejections without having calibrated anything.
        assert outcome.runtime_decisions == outcome.finite_certificates
        assert outcome.runtime_decisions >= outcome.reps * RUNTIME_LOOKS - outcome.rejections * (
            RUNTIME_LOOKS - 1
        )
        upper = binomial_error_upper_bound(outcome.rejections, outcome.reps, ETA)
        assert upper <= ALPHA + DELTA, (outcome, upper)

    def test_registered_ratio_gaussian_null_fpr_across_repeated_looks(self):
        """NIW ratio runtime, four appended looks, 4 to 16 units per arm.

        Route: ``_capture_sequential_diagnostic_snapshot`` ->
        ``_evaluate_sequential_diagnostic`` -> ``rejects()``. The null draws
        both arms' numerator and denominator i.i.d. as exact binary64
        rationals with positive population denominators and a positive
        control population ratio, so the declared support and the composite
        null both contain the truth.

        Sensitivity, measured: the NIW ratio evidence is far more
        conservative than the scalar law at every reachable count. Across
        the 2,000 prefixes this gate decides, the largest null evidence was
        ``exp(-6.16)`` against the threshold 20, and the evidence falls
        further as the count grows: raising the batch to 16 units per arm,
        so the looks run 16 to 64, drops it to ``exp(-12.96)``. No reachable
        design brings this runtime's null evidence near its own boundary:
        the gate detects inflations above roughly a factor of 9,500, and no
        tighter design exists at any cost. The falsifying witness that the
        route CAN reject at all is
        ``test_registered_ratio_gaussian_runtime_route_smoke``.
        """
        outcome = _registered_gaussian_null("gaussian_ratio", reps=500, seed=223)
        assert outcome.runtime_decisions == outcome.finite_certificates
        assert outcome.runtime_decisions >= outcome.reps * RUNTIME_LOOKS - outcome.rejections * (
            RUNTIME_LOOKS - 1
        )
        upper = binomial_error_upper_bound(outcome.rejections, outcome.reps, ETA)
        assert upper <= ALPHA + DELTA, (outcome, upper)


@pytest.mark.parameter_recovery
class TestPowerRecursionMatchesMC:
    def test_always_valid_crossing_power_matches_simulation(self):
        """Empirically confirms the recursion's crossing math for the
        Gaussian-mixture planning boundary (planning_bounds converts the
        mixture radius to per-look z-scale bounds; the drift recursion is
        shared)."""
        import numpy as np

        from increment.estimation.sequential import GaussianScoreMixture
        from increment.power.sequential import planning_bounds, sequential_power

        looks, reps, drift = 8, 40_000, 2.5
        fractions = [k / looks for k in range(1, looks + 1)]
        bounds = np.array(
            planning_bounds(GaussianScoreMixture(), fractions, se_full=1.0, alpha=0.05)
        )
        analytic = sequential_power(
            GaussianScoreMixture(), drift, 1.0, 0.05, planned_looks=looks, exit_side="both"
        )

        rng = np.random.default_rng(13)
        incr = rng.standard_normal((reps, looks)) / np.sqrt(looks) + drift / looks
        z = np.cumsum(incr, axis=1) / np.sqrt(fractions)
        mc = float((np.abs(z) > bounds).any(axis=1).mean())
        assert analytic == pytest.approx(mc, abs=0.01)
