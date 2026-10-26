from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "increment" / "estimators" / "manifest.json"


def _ref_data(ref: Any) -> dict[str, Any]:
    return {
        "estimator_id": ref.estimator_id,
        "module": ref.module,
        "attribute": ref.attribute,
        "contract": ref.contract,
        "estimand": ref.estimand,
        "selector": ref.selector,
        "needs": [
            [{"kind": type(spec).__name__.removesuffix("Spec"), **asdict(spec)} for spec in route]
            for route in ref.needs
        ],
    }


def render(refs=None) -> str:
    from increment.declarations import BUILTIN_ESTIMATORS

    data = [_ref_data(ref) for ref in (BUILTIN_ESTIMATORS if refs is None else refs)]
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def check(root: Path = ROOT) -> list[str]:
    path = root / "increment" / "estimators" / "manifest.json"
    expected = render()
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError:
        return [f"missing generated estimator manifest: {path}"]
    return [] if actual == expected else [f"stale generated estimator manifest: {path}"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        errors = check()
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
