"""Render the registered refusal-code reference page."""

from __future__ import annotations

import argparse
import html
import inspect
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
    modules: tuple[str, ...]
    message: str | None
    message_rule: str | None


def collect_refusal_specs() -> list[CodeEntry]:
    """Collect each live refusal code once, with every registering module."""
    entries = []
    for code, registrations in sorted(walk_refusal_registries().items()):
        refusals = [
            (location, spec) for location, spec in registrations if isinstance(spec, RefusalSpec)
        ]
        if not refusals:
            continue
        spec = refusals[0][1]
        context_keys = spec.keys | {
            parameter.name
            for parameter in inspect.signature(spec.render).parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and not parameter.name.startswith("_")
        }
        modules = {
            location.split("[", maxsplit=1)[0].rsplit(".", maxsplit=1)[0]
            for location, _ in refusals
        }
        message_rule = None
        if spec.template is None:
            source = inspect.getsourcefile(spec.render)
            if source is None:
                raise ValueError(f"{code}: message renderer has no source file")
            module_path = Path(source).relative_to(Path(__file__).resolve().parents[1])
            _, line = inspect.getsourcelines(spec.render)
            message_rule = f"https://github.com/kylejcaron/increment/blob/main/{module_path.as_posix()}#L{line}"
        entries.append(
            CodeEntry(
                code=code,
                error_class=spec.error_type.__name__,
                context_keys=tuple(sorted(context_keys)),
                modules=tuple(sorted(modules)),
                message=spec.template,
                message_rule=message_rule,
            )
        )
    return entries


def render(entries: list[CodeEntry]) -> str:
    """Render live refusal declarations and retired-code replacements deterministically."""
    ordered = sorted(entries, key=lambda item: item.code)
    families = tuple(dict.fromkeys(entry.code.split(".", 1)[0] for entry in ordered))
    lines = [
        "# Error codes",
        "",
        "Search for the full `CodedError.code`, or choose its family below. "
        "Codes and context keys are stable; message wording may change.",
        "",
        "Message templates show the cause and any corrective guidance declared by the registry. "
        "Braced fields are filled from the error's context at runtime. "
        "For context-dependent messages, read the raised message; the linked rule is its source.",
        "",
        "Context keys include template fields, explicitly declared keys and named renderer "
        "parameters (including optional parameters). Registering modules include the declaration "
        "and every module that reuses it.",
        "",
        "## Registered refusal codes",
        "",
        " · ".join(f"[`{family}.*`](#family-{family})" for family in families),
        "",
    ]
    family = None
    for entry in ordered:
        current_family = entry.code.split(".", 1)[0]
        if current_family != family:
            family = current_family
            lines.extend((f'<span id="family-{family}"></span>', "", f"## `{family}.*`", ""))
        keys = ", ".join(f"`{key}`" for key in entry.context_keys) or "—"
        modules = ", ".join(f"`{module}`" for module in entry.modules)
        cause = (
            f"<p>{html.escape(entry.message)}</p>"
            if entry.message is not None
            else "The cause depends on the supplied context; read the raised message. "
            f"[Message rule]({entry.message_rule})."
        )
        lines.extend(
            (
                f'<span id="{entry.code}"></span>',
                "",
                f"### `{entry.code}`",
                "",
                "**Cause and message:**",
                "",
                cause,
                "",
                "**Action:** Check the named inputs against the message before retrying. "
                "[Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).",
                "",
                f"**Error class:** `{entry.error_class}`",
                "",
                f"**Context keys:** {keys}",
                "",
                f"**Registering modules:** {modules}",
                "",
            )
        )

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
