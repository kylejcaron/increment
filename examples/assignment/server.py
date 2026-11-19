"""Make decisions at service entry and build the corresponding demo responses."""

from __future__ import annotations

import json
from pathlib import Path

from increment.assignment import compile_allocation


def main() -> None:
    """Print demo membership, decisions, responses and the effect of a ramp."""
    here = Path(__file__).resolve().parent
    initial = compile_allocation(json.loads((here / "config.json").read_text(encoding="utf-8")))
    ramped = compile_allocation(
        json.loads((here / "config-ramped.json").read_text(encoding="utf-8"))
    )
    requests = (
        ("acct:1", "checkout"),
        ("acct:2", "checkout"),
        ("acct:3", "checkout"),
        ("acct:1", "search"),
        ("acct:1", "e"),
    )

    # Deduplicate by study and unit at common entry, before rendering any feature.
    memberships = [
        {"unit_id": unit_id, "membership": initial.holdout_membership("shopping", unit_id)}
        for unit_id in dict.fromkeys(unit_id for unit_id, _ in requests)
    ]
    assignments = []
    responses = []
    for unit_id, target in requests:
        decision = initial.decide(target, unit_id)
        assignments.append({"unit_id": unit_id, "decision": decision})
        # These are response bodies, not proof a browser showed them to someone.
        content = "New experience" if decision["arm"] == "treatment" else "Baseline experience"
        responses.append({"unit_id": unit_id, "target": target, "content": content})

    print(
        json.dumps(
            {
                "memberships": memberships,
                "assignments": assignments,
                "responses": responses,
                "ramp": {
                    "before": initial.decide("checkout", "acct:1"),
                    "after": ramped.decide("checkout", "acct:1"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
