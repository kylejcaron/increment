"""Report fast-test JUnit items above the elapsed-time budget without failing CI."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlowItem:
    nodeid: str
    seconds: float


def over_budget(junit: Path, *, budget_s: float = 5.0) -> list[SlowItem]:
    """Return non-skipped items strictly above the budget, slowest first."""
    items = []
    for case in ET.parse(junit).iter("testcase"):
        if case.find("skipped") is not None:
            continue
        seconds = float(case.get("time", "0"))
        if seconds <= budget_s:
            continue
        classname = case.get("classname", "")
        file = case.get("file")
        if file:
            module = Path(file).with_suffix("").as_posix().replace("/", ".")
            classes = classname.removeprefix(module).lstrip(".").split(".")
            parts = [file, *(part for part in classes if part)]
        else:
            module_parts = classname.split(".")
            test_module = next(
                (i for i, part in enumerate(module_parts) if part.startswith("test_")),
                len(module_parts) - 1,
            )
            parts = ["/".join(module_parts[: test_module + 1]) + ".py"]
            parts.extend(module_parts[test_module + 1 :])
        parts.append(case.get("name", ""))
        items.append(SlowItem("::".join(parts), seconds))
    return sorted(items, key=lambda item: (-item.seconds, item.nodeid))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    parser.add_argument("--budget-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    items = over_budget(args.junit, budget_s=args.budget_s)
    print(f"### Fast items over {args.budget_s:g} s (advisory)\n")
    if items:
        print("| Item | Seconds |\n| --- | ---: |")
        for item in items:
            nodeid = item.nodeid.replace("|", "&#124;").replace("`", "&#96;")
            print(f"| `{nodeid}` | {item.seconds:.3f} |")
    else:
        print("No items exceeded the budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
