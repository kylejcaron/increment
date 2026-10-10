from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tests.doc_evidence import (
    EvidenceReference,
    check_collected,
    check_static,
    collect_nodeids,
    extract_references,
    selector_files,
)


def _documents(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / name for name in ("docs/validation.md", "docs/limitations.md", "docs/llms.txt")
    )


def _local_links(root: Path, documents: tuple[Path, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    raw_pattern = re.compile(
        r"https://raw\.githubusercontent\.com/kylejcaron/increment/main/([^):]+)"
    )
    for document in documents:
        if not document.is_file():
            continue
        text = document.read_text()
        for match in raw_pattern.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if not (root / target).is_file():
                errors.append(
                    f"{document.relative_to(root)}: raw GitHub target does not exist: {target}"
                )
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip().split("#", 1)[0].split("?", 1)[0]
            if not target or re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            candidate = (document.parent / target).resolve()
            if not candidate.is_file():
                errors.append(
                    f"{document.relative_to(root)}: local link target does not exist: {target}"
                )
    return tuple(errors)


def _all_refs(root: Path) -> tuple[EvidenceReference, ...]:
    refs: list[EvidenceReference] = []
    for document in _documents(root):
        if document.is_file():
            refs.extend(
                extract_references(
                    document.read_text(), source=document.relative_to(root).as_posix()
                )
            )
    return tuple(refs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check curated documentation evidence references without running tests or campaigns."
    )
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="also verify pending selectors against one bounded pytest collection",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        refs = _all_refs(root)
    except ValueError as exc:
        print(f"malformed documentation reference: {exc}", file=sys.stderr)
        return 1
    report = check_static(root, refs)
    errors = (*report.errors, *_local_links(root, _documents(root)))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.collect:
        try:
            nodeids = collect_nodeids(root, selector_files(refs))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        unresolved = check_collected(report, nodeids)
        if unresolved:
            print("\n".join(unresolved), file=sys.stderr)
            return 1
        print(
            f"Static references resolved; {len(report.pending_nodeids)} pending selector identities matched collection. No tests were executed and no campaign was run."
        )
        return 0
    print(
        f"Static references resolved; {len(report.pending_nodeids)} selector identities pending collection. This mode does not claim full reference verification."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
