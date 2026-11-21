"""Two independent account experiments: both can ship, but adding B need not help."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from increment.assignment import compile_allocation
from increment.overlap import (
    OverlapCapture,
    OverlapClaim,
    OverlapFeed,
    OverlapMetric,
    OverlapSdkArchive,
    analyze_overlap,
    capture_overlap_frame,
    sdk_overlap_decision,
    sdk_overlap_plan,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _empty_deliveries() -> pa.Table:
    schema = pa.schema(
        [
            (name, pa.string())
            for name in (
                "study_id",
                "event_id",
                "unit_id",
                "target",
                "experiment",
                "revision",
                "delivered_arm",
            )
        ]
        + [("delivered_at", pa.timestamp("us", tz="UTC"))]
    )
    return pa.Table.from_pylist([], schema=schema)


def _capture_receipt(plan, cutoff):
    feeds = (
        ("roster", plan.entry_open, plan.start),
        ("decisions", plan.entry_open, plan.start),
        ("deliveries", plan.entry_open, plan.end),
        ("outcomes", plan.start, plan.end),
    )
    return OverlapCapture(
        generation_id="synthetic-window-v1",
        origin=plan.law.origin,
        law_sha256=plan.law_sha256,
        archive_sha256=plan.law.archive_sha256,
        cutoff=cutoff,
        feeds=tuple(
            OverlapFeed(
                role=role,
                source_id=f"demo-{role}",
                version="snapshot-1",
                recipe_sha256=_digest(f"demo-recipe:{role}"),
                covered_from=first,
                complete_through=last,
            )
            for role, first, last in feeds
        ),
    )


def _make_inputs(plan, raw, receipt, cutoff, n):
    evaluator = compile_allocation(json.loads(raw))
    roster, decisions, deliveries, outcomes = [], [], [], []
    # Fix the entire synthetic cohort before asking the allocator for any arms.
    for i in range(n):
        unit = {"study_id": plan.study_id, "unit_id": f"account-{i}"}
        roster.append(
            {
                **unit,
                "entry_at": plan.entry_open,
                "eligible_a": True,
                "eligible_b": True,
                "eligibility_at_a": plan.entry_open,
                "eligibility_at_b": plan.entry_open,
            }
        )
        on = []
        for j, factor in enumerate((plan.a, plan.b), start=1):
            decision = evaluator.decide(factor.target, unit["unit_id"])
            normalized = sdk_overlap_decision(
                **unit,
                config=receipt,
                decision=decision,
                decided_at=plan.entry_open + timedelta(minutes=j),
            )
            decisions.append(normalized)
            assigned_on = decision["arm"] == factor.on
            on.append(int(assigned_on))
            if assigned_on:
                deliveries.append(
                    {
                        **unit,
                        "event_id": f"{i}:{j}",
                        "target": factor.target,
                        "experiment": factor.experiment,
                        "revision": factor.revision,
                        "delivered_arm": factor.on,
                        "delivered_at": plan.start,
                    }
                )
        a, b = on
        means = (10.0 + 4 * a + 3 * b - 5 * a * b, 1.0)
        for metric, mean in zip(plan.metrics, means, strict=True):
            outcomes.append(
                {
                    **unit,
                    "metric": metric.name,
                    "definition_sha256": metric.definition_sha256,
                    "window_start": plan.start,
                    "window_end": plan.end,
                    "value": mean + (i % 13 - 6) / 40.0,
                    "finalized_at": cutoff,
                }
            )
    return {
        "roster": pa.Table.from_pylist(roster),
        "decisions": pa.Table.from_pylist(decisions),
        "deliveries": pa.Table.from_pylist(deliveries) if deliveries else _empty_deliveries(),
        "outcomes": pa.Table.from_pylist(outcomes),
    }


def run_example(archive_path: Path, *, n: int = 2400):
    """Reproduce the worked synthetic example with the archived allocation rules."""
    raw = archive_path.read_bytes()
    receipt = OverlapSdkArchive.from_bytes(raw)
    entry = datetime(2026, 1, 1, tzinfo=UTC)
    start, end = entry + timedelta(hours=1), entry + timedelta(days=7)
    cutoff = end + timedelta(minutes=1)
    metrics = (
        OverlapMetric(
            name="orders",
            unit="orders per account",
            definition_sha256=_digest("orders"),
            kind="mean",
            role="primary",
            better="higher",
        ),
        OverlapMetric(
            name="cost",
            unit="cost per account",
            definition_sha256=_digest("cost"),
            kind="mean",
            role="guardrail",
            better="lower",
            margin_abs=0.05,
        ),
    )
    claims = (
        OverlapClaim(metric="orders", contrast="b_given_a_on", alternative="benefit"),
        OverlapClaim(metric="orders", contrast="joint_launch", alternative="benefit"),
        OverlapClaim(metric="orders", contrast="interaction", alternative="difference"),
        OverlapClaim(metric="cost", contrast="joint_launch", alternative="noninferiority"),
    )
    plan = sdk_overlap_plan(
        a_config=receipt,
        a_target="A",
        a_experiment="A",
        a_off="off",
        a_on="on",
        b_config=receipt,
        b_target="B",
        b_experiment="B",
        b_off="off",
        b_on="on",
        study_id="checkout-and-promotion",
        unit_namespace="accounts",
        population_id="both-eligible",
        eligibility_rule_sha256=_digest("accounts offered checkout and promotion"),
        enrollment_rule_sha256=_digest("all eligible accounts randomized"),
        serving_barrier_id="one-hour-barrier",
        registered_at=entry - timedelta(days=1),
        entry_open=entry,
        start=start,
        end=end,
        target="jointly_eligible",
        metrics=metrics,
        claims=claims,
        alpha=0.05,
    )
    evidence = capture_overlap_frame(
        plan,
        capture=_capture_receipt(plan, cutoff),
        **_make_inputs(plan, raw, receipt, cutoff, n),
    )
    return analyze_overlap(evidence)


def main() -> None:
    """Show the two launch questions and the nondecision for a tiny window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="original SDK configuration used by the example")
    args = parser.parse_args()
    result = run_example(args.archive)
    for row in result.rows:
        if row.metric == "orders" and row.contrast in {
            "b_given_a_on",
            "joint_launch",
            "interaction",
        }:
            print(f"{row.contrast:15} {row.estimate:+.2f} orders/account  {row.decision}")
    print("joint launch:", result.joint_decision)
    print("too few assignments:", run_example(args.archive, n=3).joint_decision)


if __name__ == "__main__":
    main()
