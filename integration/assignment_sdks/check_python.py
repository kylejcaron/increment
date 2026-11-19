"""Run the protocol corpus against an installed wheel outside the checkout."""

from __future__ import annotations

import inspect
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from increment import assignment, errors


def main() -> None:
    """Check complete records or stable refusals from the wheel's public API."""
    module = Path(inspect.getfile(assignment)).resolve()
    assert module.is_relative_to(Path(sys.prefix).resolve()), module
    vectors = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for case in vectors["cases"]:
        try:
            evaluator = assignment.compile_allocation(vectors["configs"][case["config"]])
            match case["op"]:
                case "decide":
                    result = evaluator.decide(case["target"], case["unit_id"])
                case "membership":
                    result = evaluator.holdout_membership(case["target"], case["unit_id"])
                case other:
                    raise AssertionError(f"Unexpected successful operation {other!r}: {case['id']}")
        except errors.InvalidRequestError as exc:
            assert exc.code == case.get("error"), (case["id"], exc.code, case.get("error"))
        else:
            actual = json.dumps(result, sort_keys=True)
            expected = json.dumps(case["expected"], sort_keys=True)
            assert actual == expected, (case["id"], actual, expected)
    print(
        f"increment {version('increment')}, Python {platform.python_version()}: "
        f"{len(vectors['cases'])} cases"
    )


if __name__ == "__main__":
    main()
