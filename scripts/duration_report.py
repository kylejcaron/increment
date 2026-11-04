"""Report test durations against per-layer budgets and committed ratchets."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlowItem:
    nodeid: str
    seconds: float


@dataclass(frozen=True)
class LayerItem:
    layer: str
    nodeid: str
    seconds: float


LAYER_BUDGETS_S = {"unit": 0.1, "integration": 1.0, "slow": 30.0}


def layer_over_budget(junit: Path, collection: Path) -> list[LayerItem]:
    """Return non-skipped tests above their marker-defined layer budget."""
    metadata = json.loads(collection.read_text(encoding="utf-8"))
    node_layers = {
        nodeid: layer
        for layer, nodeids in metadata["tier_nodeids"].items()
        for nodeid in nodeids
        if layer in LAYER_BUDGETS_S
    }
    items = []
    for item in _junit_items(junit):
        layer = node_layers.get(item.nodeid)
        if layer is not None and item.seconds > LAYER_BUDGETS_S[layer]:
            items.append(LayerItem(layer, item.nodeid, item.seconds))
    return sorted(items, key=lambda item: (-item.seconds, item.nodeid))


def over_budget_count_errors(junit: Path, collection: Path, baseline: Path) -> list[str]:
    """Return layers whose over-budget counts exceed their recorded ceiling."""
    reference = json.loads(baseline.read_text(encoding="utf-8"))
    counts = dict.fromkeys(LAYER_BUDGETS_S, 0)
    for item in layer_over_budget(junit, collection):
        counts[item.layer] += 1
    return [
        layer
        for layer, count in counts.items()
        if count > reference["max_over_budget"].get(layer, 0)
    ]


def _junit_items(junit: Path) -> list[SlowItem]:
    """Read all non-skipped JUnit cases."""
    items = []
    for case in ET.parse(junit).iter("testcase"):
        if case.find("skipped") is not None:
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
        items.append(SlowItem("::".join(parts), float(case.get("time", "0"))))
    return items


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


def _read_aggregate_items(
    artifacts_dir: Path, expected_shards: set[str]
) -> tuple[list[LayerItem], list[str]]:
    junit_paths = set(artifacts_dir.rglob("junit.xml"))
    collection_paths = set(artifacts_dir.rglob("junit-collection.json"))
    incomplete = []
    paired_shards = set()
    unique_items: dict[tuple[str, str], LayerItem] = {}
    for junit in sorted(junit_paths):
        collection = junit.with_name("junit-collection.json")
        if collection not in collection_paths:
            incomplete.append(f"JUnit without collection metadata: {junit}")
            continue
        try:
            metadata = json.loads(collection.read_text(encoding="utf-8"))
            shard_id = metadata.get("shard_id")
        except (OSError, UnicodeError, json.JSONDecodeError):
            shard_id = None
        if not shard_id:
            incomplete.append(f"collection metadata without shard ID: {collection}")
            continue
        paired_shards.add(shard_id)
        for item in layer_over_budget(junit, collection):
            key = (item.layer, item.nodeid)
            previous = unique_items.get(key)
            if previous is None or item.seconds > previous.seconds:
                unique_items[key] = item
    for collection in sorted(collection_paths):
        if collection.with_name("junit.xml") not in junit_paths:
            incomplete.append(f"collection metadata without JUnit: {collection}")
    missing_shards = expected_shards - paired_shards
    extra_shards = paired_shards - expected_shards
    if missing_shards:
        incomplete.append(f"missing required shards: {', '.join(sorted(missing_shards))}")
    if extra_shards:
        incomplete.append(f"unexpected shards: {', '.join(sorted(extra_shards))}")
    return list(unique_items.values()), incomplete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path, nargs="?")
    parser.add_argument("--budget-s", type=float, default=5.0)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--expected-shards")
    args = parser.parse_args(argv)
    if args.artifacts_dir:
        if not args.baseline or not args.expected_shards:
            parser.error("--artifacts-dir requires --baseline and --expected-shards")
        reference = json.loads(args.baseline.read_text(encoding="utf-8"))
        mode = reference["mode"]
        if mode not in {"report", "enforce"}:
            parser.error("baseline mode must be 'report' or 'enforce'")
        expected_shards = set(args.expected_shards.split(","))
        items, incomplete = _read_aggregate_items(args.artifacts_dir, expected_shards)
        counts = {layer: sum(item.layer == layer for item in items) for layer in LAYER_BUDGETS_S}
        growth = [
            layer
            for layer, count in counts.items()
            if count > reference["max_over_budget"].get(layer, 0)
        ]
        print("### Per-layer duration ratchet\n")
        for layer, budget in LAYER_BUDGETS_S.items():
            print(f"- {layer}: {counts[layer]} item(s) over {budget:g} s")
        if incomplete:
            command = "error" if mode == "enforce" else "warning"
            print(f"\n::{command}::Incomplete duration evidence: {'; '.join(incomplete)}")
        if growth:
            command = "error" if mode == "enforce" else "warning"
            print(f"\n::{command}::Over-budget counts exceed baseline: {', '.join(growth)}")
        if items:
            print("\n| Layer | Item | Seconds |\n| --- | --- | ---: |")
            for item in sorted(items, key=lambda item: (-item.seconds, item.nodeid)):
                nodeid = item.nodeid.replace("|", "&#124;").replace("`", "&#96;")
                print(f"| {item.layer} | `{nodeid}` | {item.seconds:.3f} |")
        return int(mode == "enforce" and bool(incomplete or growth))

    if args.junit is None:
        parser.error("junit is required unless --artifacts-dir is supplied")
    if args.collection:
        items = layer_over_budget(args.junit, args.collection)
        print("### Per-layer duration ratchet (advisory)\n")
        for layer, budget in LAYER_BUDGETS_S.items():
            count = sum(item.layer == layer for item in items)
            print(f"- {layer}: {count} item(s) over {budget:g} s")
        if args.baseline and over_budget_count_errors(args.junit, args.collection, args.baseline):
            print("\n::warning::Over-budget counts exceed baseline.")
        return 0

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
