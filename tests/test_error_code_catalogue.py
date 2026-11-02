from __future__ import annotations

from collections import Counter

from increment.errors import RETIRED_CODES, RefusalSpec
from scripts import render_error_codes
from tests.refusal_registry import walk_refusal_registries


def test_every_registered_code_appears_once() -> None:
    entries = render_error_codes.collect_refusal_specs()
    counts = Counter(entry.code for entry in entries)
    assert counts == Counter(
        {
            code: 1
            for code, registrations in walk_refusal_registries().items()
            if any(isinstance(spec, RefusalSpec) for _, spec in registrations)
        }
    )
    page = render_error_codes.render(entries)
    registered_rows = [
        line for line in page.splitlines() if line.startswith("| `") and line.count("|") == 5
    ]
    assert len(registered_rows) == len(counts)
    assert {line.split("|", maxsplit=2)[1].strip() for line in registered_rows} == {
        f"`{code}`" for code in counts
    }
    metadata = {
        columns[1].strip(): tuple(column.strip() for column in columns[2:-1])
        for line in registered_rows
        if (columns := line.split("|"))
    }
    expected = {
        "`evidence.operation_unavailable`": (
            "`CapabilityError`",
            "`constructor`, `offered`, `request`, `requested`, `route`",
            "`increment._evidence_refusals`, `increment.query._native_refusals`",
        ),
        "`package.builtin_provider_unavailable`": (
            "`CodedError`",
            "`count`, `feature`, `group`, `provider`, `reason`",
            "`increment.errors`, `increment.kernel.ports`, `increment.query.artifact_ports`",
        ),
        "`estimation.diagnostics.alpha`": (
            "`InvalidRequestError`",
            "`alpha`",
            ", ".join(
                f"`{module}`"
                for module in (
                    "increment.breakout.estimates",
                    "increment.estimation.absorption",
                    "increment.estimation.cate",
                    "increment.estimation.diagnostics",
                    "increment.estimation.encouragement",
                    "increment.estimation.inference",
                    "increment.estimation.quantile",
                    "increment.estimation.targeting",
                    "increment.power.core",
                )
            ),
        ),
    }
    assert {code: metadata[code] for code in expected} == expected


def test_retired_codes_name_their_replacement() -> None:
    page = render_error_codes.render(render_error_codes.collect_refusal_specs())
    rows = {
        line.split("|", maxsplit=2)[1].strip(): line
        for line in page.splitlines()
        if line.startswith("| `")
    }
    for code, replacement in RETIRED_CODES.items():
        row = rows[f"`{code}`"]
        if replacement is None:
            assert "Removed without replacement" in row
        else:
            replacements = (replacement,) if isinstance(replacement, str) else replacement
            assert ", ".join(f"`{item}`" for item in replacements) in row


def test_check_mode_fails_on_a_stale_page(tmp_path, monkeypatch) -> None:
    stale = tmp_path / "error-codes.md"
    stale.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(render_error_codes, "PAGE", stale)
    assert render_error_codes.main(["--check"]) == 1
