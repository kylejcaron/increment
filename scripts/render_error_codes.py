"""Render the registered refusal-code reference page."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from increment.errors import RETIRED_CODES, RefusalSpec
from tests.refusal_registry import walk_refusal_registries

PAGE = Path("docs/reference/error-codes.md")


@dataclass(frozen=True, slots=True)
class CodeEntry:
    code: str
    error_class: str
    context_keys: tuple[str, ...]
    owner: str


def collect_refusal_specs() -> list[CodeEntry]:
    """Collect each live refusal code once, with its declaring module."""
    entries = []
    for code, registrations in sorted(walk_refusal_registries().items()):
        refusals = sorted(
            ((location, spec) for location, spec in registrations if isinstance(spec, RefusalSpec)),
            key=lambda item: item[0],
        )
        if not refusals:
            continue
        location, spec = refusals[0]
        owner = location.split("[", maxsplit=1)[0].rsplit(".", maxsplit=1)[0]
        entries.append(
            CodeEntry(
                code=code,
                error_class=spec.error_type.__name__,
                context_keys=tuple(sorted(spec.keys)),
                owner=owner,
            )
        )
    return entries


def render(entries: list[CodeEntry]) -> str:
    """Render live refusal declarations and retired-code replacements deterministically."""
    lines = [
        "# Error codes",
        "",
        "Refusal codes are stable identifiers; their human-readable wording may change.",
        "Context keys are part of the refusal contract. Retired codes and their replacements are listed below.",
        "",
        "## Registered refusal codes",
        "",
        "| Code | Error class | Context keys | Owning module |",
        "| --- | --- | --- | --- |",
    ]
    for entry in sorted(entries, key=lambda item: item.code):
        keys = ", ".join(f"`{key}`" for key in entry.context_keys) or "—"
        lines.append(f"| `{entry.code}` | `{entry.error_class}` | {keys} | `{entry.owner}` |")

    lines.extend(
        [
            "",
            "## Retired codes",
            "",
            "| Retired code | Replacement |",
            "| --- | --- |",
        ]
    )
    for code, replacement in sorted(RETIRED_CODES.items()):
        if replacement is None:
            target = "Removed without replacement"
        else:
            replacements = (replacement,) if isinstance(replacement, str) else replacement
            target = ", ".join(f"`{item}`" for item in replacements)
        lines.append(f"| `{code}` | {target} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated page is stale")
    args = parser.parse_args(argv)
    generated = render(collect_refusal_specs())
    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != generated:
            print(f"{PAGE} is stale; run python scripts/render_error_codes.py")
            return 1
        return 0
    PAGE.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
