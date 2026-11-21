"""Compare two joint enrollment forecasts before launching an experiment pair."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

import pyarrow as pa

from increment.overlap import (
    OverlapClaim,
    OverlapFactor,
    OverlapLaw,
    OverlapMetric,
    OverlapPlan,
    simulate_overlap,
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _factor(name: str, on_percent: int) -> OverlapFactor:
    return OverlapFactor(
        target=name,
        experiment=name,
        revision="1",
        off="off",
        on="on",
        off_weight=100 - on_percent,
        on_weight=on_percent,
        arm_draw_id=f"{name}-arm",
        layer=None,
    )


def _plan() -> OverlapPlan:
    return OverlapPlan(
        study_id="checkout-and-promotion",
        unit_namespace="account",
        population_id="eligible-and-enrolled-in-both",
        eligibility_rule_sha256=_digest("accounts eligible for both at entry"),
        enrollment_rule_sha256=_digest("accounts randomized in both experiments"),
        registered_at=datetime(2027, 1, 1, tzinfo=UTC),
        entry_open=datetime(2027, 1, 2, tzinfo=UTC),
        start=datetime(2027, 1, 3, tzinfo=UTC),
        end=datetime(2027, 2, 1, tzinfo=UTC),
        target="jointly_eligible_and_enrolled",
        a=_factor("checkout", on_percent=20),
        b=_factor("promotion", on_percent=10),
        law=OverlapLaw(
            scheme="independent_unit_bernoulli",
            origin="external",
            archive_sha256=(_digest("checkout v1"), _digest("promotion v1")),
            gate_basis="preassigned_roster",
            admission="joint_gate_required",
            rule_source_id="hypothetical-randomizer",
            rule_version="1",
            serving_barrier_id="eligibility-before-assignment",
        ),
        metrics=(
            OverlapMetric(
                name="orders",
                unit="orders per account",
                definition_sha256=_digest("orders per assigned account in the measurement window"),
                kind="mean",
                role="primary",
                better="higher",
            ),
        ),
        claims=(OverlapClaim(metric="orders", contrast="interaction", alternative="difference"),),
        alpha=0.05,
    )


def main() -> None:
    """Show how joint enrollment changes the chance of detecting an interaction."""
    plan = _plan()
    means: dict[Literal["00", "10", "01", "11"], int] = {"00": 10, "10": 11, "01": 13, "11": 12}
    # Each row is an equally likely hypothetical outcome for one account in its group.
    cell_outcomes = {
        cell: pa.table({"orders": [mean + offset for offset in (-6, -2, 2, 6)]})
        for cell, mean in means.items()
    }
    seed, replications = 321, 2_000
    print(f"{replications:,} simulated runs per scenario (seed {seed})")
    for n_target in (1_200, 4_000):
        result = simulate_overlap(
            plan,
            n_target=n_target,
            cell_outcomes=cell_outcomes,
            replications=replications,
            seed=seed,
        )
        interaction = result.per_claim["orders", "interaction"]
        expected = ", ".join(
            f"{cell}: {count:,.0f}"
            for cell, count in zip(means, result.expected_counts, strict=True)
        )
        lower, upper = interaction.mc_interval
        total = result.replications
        print(f"\nJoint enrollment: {result.n_target:,} accounts")
        print(f"  Expected accounts by group: {expected}")
        print(f"  Assumed interaction: {interaction.truth:+.1f} orders per account")
        print(f"  Decision-ready runs: {interaction.n_decision_ready:,}/{total:,}")
        print(
            f"  Interaction detections: {interaction.n_reject:,}/{total:,} "
            f"({interaction.rejection_rate:.1%}; 95% Monte Carlo interval {lower:.1%}–{upper:.1%})"
        )


if __name__ == "__main__":
    main()
