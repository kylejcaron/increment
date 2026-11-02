"""Pytest plugin that records collected node IDs and per-tier item counts.

Loaded as a top-level module so the same plugin can collect another checkout
of the repository. It imports only the standard library and pytest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _pytest.mark import MarkMatcher
from _pytest.mark.expression import Expression

OUTPUT_ENV = "INCREMENT_COLLECT_OUTPUT"
TIERS_ENV = "INCREMENT_COLLECT_TIERS"


def pytest_collection_finish(session: pytest.Session) -> None:
    output = os.environ.get(OUTPUT_ENV)
    if not output:
        return
    expressions = {
        tier: Expression.compile(marker)
        for tier, marker in json.loads(os.environ[TIERS_ENV]).items()
    }
    counts = dict.fromkeys(expressions, 0)
    tier_nodeids = {tier: [] for tier in expressions}
    for item in session.items:
        matcher = MarkMatcher.from_markers(item.iter_markers())
        for tier, expression in expressions.items():
            if expression.evaluate(matcher):
                counts[tier] += 1
                tier_nodeids[tier].append(item.nodeid)
    Path(output).write_text(
        json.dumps(
            {
                "tiers": counts,
                "nodeids": [item.nodeid for item in session.items],
                "tier_nodeids": tier_nodeids,
            }
        ),
        encoding="utf-8",
    )
