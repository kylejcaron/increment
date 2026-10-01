from pathlib import Path
from typing import get_args

import pytest

from increment._compatibility_inspector import FamilyStatus, RuntimeStatus
from scripts.render_compatibility import (
    ROOT,
    STATUS_LABELS,
    _detail,
    _display_status,
    _reference_payload,
    check_reference,
    render_reference,
    write_reference,
)


def test_render_is_deterministic_and_contains_static_fallback():
    first = render_reference()
    second = render_reference()
    assert first == second
    assert "<table" in first
    assert "<details" in first
    assert "always-valid-quantile-secondary-family" in first
    assert "Runtime" in first
    assert "Family" in first
    assert "Evidence" in first
    assert "cluster × cuped" in first
    assert "Evidence source: probe" in first


def test_render_surfaces_scenario_finding_details_and_provenance():
    """Finding warning/assumptions/floor/reference and a scenario-level
    contract-provenance summary must be readable without JS -- not just
    present inside the inert embedded JSON payload."""
    output = render_reference()
    readable, _, _ = output.partition('<script id="compatibility-data"')
    assert "Contract provenance" in readable
    assert "arm_moments" in readable
    assert "plan_family" in readable
    assert "arm.metric.quantile_sequential" in readable
    assert "quantile sequential inference is refused" in readable


def test_check_detects_stale_generated_file(tmp_path: Path):
    output = tmp_path / "compatibility.md"
    output.write_text("stale")
    assert check_reference(output) is False
    write_reference(output)
    assert check_reference(output) is True


def test_matrix_cell_buttons_carry_accessible_name():
    """Each matrix cell button must expose an
    accessible name naming the capability, metric, and status -- color/
    uppercase text alone is not an accessible name for assistive tech."""
    from tests.compatibility_catalog import MATRIX

    output = render_reference()
    cell = MATRIX["sequential"]["quantile"]
    assert cell.runtime_status == "refused"
    assert 'aria-label="sequential quantile: REFUSED"' in output, (
        "matrix cell button is missing its aria-label"
    )


def test_matrix_cell_buttons_carry_a_data_status_attribute_for_semantic_coloring():
    """Each matrix cell button must expose its raw internal snake_case
    runtime status (e.g. ``"supported"``) as a ``data-status`` attribute
    so CSS can color it semantically. This is additive: the visible
    status word and accessible name stay the primary signal; color is
    never the only way to tell statuses apart."""
    from tests.compatibility_catalog import MATRIX

    output = render_reference()
    cell = MATRIX["sequential"]["quantile"]
    assert (
        f'data-capability="sequential" data-metric="quantile" '
        f'data-status="{cell.runtime_status}" aria-label='
    ) in output, "matrix cell button is missing its data-status attribute"


def test_display_status_resolves_explicit_status_labels():
    """``_display_status`` is a direct
    ``STATUS_LABELS`` lookup, not an algorithmic snake_case transform --
    ``test_status_labels_covers_every_runtime_and_family_status_value``
    separately verifies the mapping is exhaustive over the real
    ``RuntimeStatus``/``FamilyStatus`` alphabet. This spot-checks the
    consumer-visible labels for two declared entries, including the
    pinned ``not_applicable`` -> ``NOT APPLICABLE`` example."""
    assert _display_status("not_applicable") == "NOT APPLICABLE"
    assert _display_status("limited") == "LIMITED"


def test_status_labels_covers_every_runtime_and_family_status_value():
    """``STATUS_LABELS`` is the single explicit
    mapping both Python and JS render from -- it must have a key for
    every ``RuntimeStatus`` and ``FamilyStatus`` value the type system
    admits, not just the ones a hand-picked example happens to cover."""
    status_alphabet = set(get_args(RuntimeStatus)) | set(get_args(FamilyStatus))
    assert status_alphabet, "RuntimeStatus/FamilyStatus resolved to no literal values"
    assert status_alphabet <= set(STATUS_LABELS)


def test_display_status_is_a_strict_lookup_not_a_reimplemented_transform():
    """``_display_status`` must be a direct ``STATUS_LABELS`` lookup --
    not an algorithmic underscore/uppercase transform that would
    silently "work" for any string, including one no one ever declared
    as a real status. An unmapped value must raise, not guess."""
    assert _display_status("supported") == STATUS_LABELS["supported"]
    with pytest.raises(KeyError):
        _display_status("not_a_real_status")


def test_payload_embeds_the_shared_status_labels_mapping():
    """JS's ``displayStatus`` must consume the exact same mapping Python
    renders from, not a hand-rolled client-side reimplementation --
    ``STATUS_LABELS`` is embedded in the payload verbatim."""
    payload = _reference_payload()
    assert payload["status_labels"] == STATUS_LABELS


def test_not_applicable_renders_as_two_words_everywhere_a_status_appears():
    """Pin ``NOT_APPLICABLE`` rendering as
    ``NOT APPLICABLE`` (space, not underscore) in every surface a status
    is shown as text -- matrix button, aria-label, pair button, static
    ``<details>`` summaries, and scenario Runtime/Family lines. No raw
    ``NOT_APPLICABLE`` or ``not_applicable`` may leak into user-visible
    text; the JSON payload (looked up by JS) is exempt -- it stays
    snake_case for internal lookups, as does the machine-only
    ``data-status`` attribute (used only by CSS for semantic coloring;
    no reader ever sees an HTML attribute value as text)."""
    import re

    from tests.compatibility_catalog import MATRIX

    output = render_reference()
    readable, _, script_and_payload = output.partition('<script id="compatibility-data"')
    cell = MATRIX["estimate"]["total"]
    assert cell.runtime_status == "not_applicable"
    assert 'data-status="not_applicable"' in readable
    assert 'aria-label="estimate total: NOT APPLICABLE"' in readable
    assert ">NOT APPLICABLE<" in readable
    assert "estimate × total: NOT APPLICABLE" in readable
    # No raw snake_case status ever leaks into human-visible text -- strip
    # the machine-only `data-status="..."` attribute values before checking.
    visible_text = re.sub(r'data-status="[^"]*"', "", readable)
    assert "not_applicable" not in visible_text
    assert "NOT_APPLICABLE" not in visible_text
    # The embedded JSON payload is the one place the snake_case form must
    # remain, for JS's own internal lookups/comparisons.
    assert '"runtime":"not_applicable"' in script_and_payload


def test_pair_buttons_carry_a_data_status_attribute_for_semantic_coloring():
    """Each pair button must expose its raw internal snake_case runtime
    status as a ``data-status`` attribute -- the same hook the matrix
    cell buttons carry -- so CSS can color pair badges semantically
    without replacing the visible status word."""
    from tests.compatibility_catalog import PAIRS

    output = render_reference()
    left, right = next(iter(sorted(PAIRS)))
    cell = PAIRS[(left, right)]
    assert (
        f'data-left="{left}" data-right="{right}" data-status="{cell.runtime_status}" aria-label='
    ) in output, "pair button is missing its data-status attribute"


def test_detail_inserts_terminal_punctuation_before_advisory():
    """``_detail`` must not concatenate a
    base note directly onto advisory prose without terminal punctuation --
    that produces an unpunctuated run-on sentence."""
    from tests.compatibility_catalog import PAIRS

    cell = PAIRS[("cluster", "sitewide")]
    assert cell.note and not cell.note.rstrip().endswith((".", "!", "?"))
    detail = _detail(cell)
    assert f"{cell.note}. {cell.advisory}" == detail


def test_generation_fails_when_scenario_evidence_reference_is_stale(monkeypatch):
    """Evidence-reference/marker validation must run
    during renderer generation, not only under pytest -- missing or stale
    evidence must fail direct generation (``render_reference``/
    ``write_reference``/``--check``), not just ``pytest``."""
    import scripts.render_compatibility as render_module
    from increment._compatibility_inspector import CompatibilityReport
    from tests.compatibility_catalog import EvidenceRef, Scenario

    def _unreachable() -> CompatibilityReport:
        raise AssertionError("evaluate should never run: validate_evidence must fail first")

    stale_scenario = Scenario(
        id="stale-evidence-scenario",
        axes=(),
        expected_runtime="unknown",
        expected_family="unknown",
        explanation="synthetic scenario for the stale-evidence generation guard",
        alternative=None,
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/this_file_does_not_exist.py",
                test="test_does_not_exist",
            ),
        ),
        evaluate=_unreachable,
    )
    monkeypatch.setattr(render_module, "SCENARIOS", (stale_scenario,))
    with pytest.raises(ValueError):
        render_module.render_reference()


def test_generation_fails_when_scenario_report_disagrees_with_expected_outcome(monkeypatch):
    """The renderer must compare each observed
    ``CompatibilityReport.overall``/``overall_family`` against the
    scenario's own declared ``expected_runtime``/``expected_family`` and
    fail generation on a mismatch -- publishing whichever one the
    renderer happened to read while the catalog and production behavior
    have drifted apart is exactly the kind of silent optimism the whole
    reference exists to prevent. Uses a real, existing evidence
    reference (this very test) so ``validate_evidence`` does not mask
    the mismatch check."""
    import scripts.render_compatibility as render_module
    from increment._compatibility_inspector import CompatibilityFinding, CompatibilityReport
    from tests.compatibility_catalog import EvidenceRef, Scenario

    def _mismatched_report() -> CompatibilityReport:
        finding = CompatibilityFinding(
            capability="test",
            runtime="supported",
            family="not_applicable",
            refusal_code=None,
            assumptions=(),
            floor=None,
            warning=None,
            reference=None,
            evidence_source="contract",
        )
        return CompatibilityReport.from_findings((finding,))

    mismatched_scenario = Scenario(
        id="mismatched-outcome-scenario",
        axes=(),
        # Declared LIMITED, but the real evaluate() below observes SUPPORTED.
        expected_runtime="limited",
        expected_family="not_applicable",
        explanation="synthetic scenario for the report/expectation mismatch guard",
        alternative=None,
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_render_compatibility.py",
                test="test_generation_fails_when_scenario_report_disagrees_with_expected_outcome",
            ),
        ),
        evaluate=_mismatched_report,
    )
    monkeypatch.setattr(render_module, "SCENARIOS", (mismatched_scenario,))
    with pytest.raises(ValueError):
        render_module.render_reference()


def test_generation_fails_when_scenario_report_disagrees_on_family_only(monkeypatch):
    """Family-only counterpart to the mismatch guard above: a matching
    ``overall`` but mismatched ``overall_family`` must still fail
    generation. Without this case, deleting the
    ``report.overall_family != scenario.expected_family`` disjunct from
    ``validate_scenario_outcome`` would leave every other test green."""
    import scripts.render_compatibility as render_module
    from increment._compatibility_inspector import CompatibilityFinding, CompatibilityReport
    from tests.compatibility_catalog import EvidenceRef, Scenario

    def _mismatched_report() -> CompatibilityReport:
        finding = CompatibilityFinding(
            capability="test",
            runtime="supported",
            family="participates",
            refusal_code=None,
            assumptions=(),
            floor=None,
            warning=None,
            reference=None,
            evidence_source="contract",
        )
        return CompatibilityReport.from_findings((finding,))

    mismatched_scenario = Scenario(
        id="mismatched-family-only-scenario",
        axes=(),
        expected_runtime="supported",
        # Declared NOT_APPLICABLE, but the real evaluate() below observes PARTICIPATES.
        expected_family="not_applicable",
        explanation="synthetic scenario for the family-only mismatch guard",
        alternative=None,
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_render_compatibility.py",
                test="test_generation_fails_when_scenario_report_disagrees_on_family_only",
            ),
        ),
        evaluate=_mismatched_report,
    )
    monkeypatch.setattr(render_module, "SCENARIOS", (mismatched_scenario,))
    with pytest.raises(ValueError):
        render_module.render_reference()


def test_every_base_and_pair_payload_record_exposes_provenance():
    """Every MATRIX cell and PAIRS entry in
    the JSON payload must carry its owning production module and exact
    composition-probe evidence -- a reader must never wonder which module
    decided a cell/pair's status or which test backs the claim."""
    from scripts.render_compatibility import _reference_payload
    from tests.compatibility_catalog import (
        CAPABILITY_PROVENANCE,
        MATRIX,
        PAIR_PROVENANCE,
        PAIRS,
    )

    payload = _reference_payload()
    for capability, column in MATRIX.items():
        for metric in column:
            record = payload["cells"][f"{capability}|{metric}"]
            assert record["owner"] == CAPABILITY_PROVENANCE[capability].owner
            assert record["probe"] == CAPABILITY_PROVENANCE[capability].probe
    for left, right in PAIRS:
        record = payload["pairs"][f"{left}|{right}"]
        assert record["owner"] == PAIR_PROVENANCE[left, right].owner
        assert record["probe"] == PAIR_PROVENANCE[left, right].probe


def test_every_refused_base_and_pair_payload_record_exposes_refusal_metadata():
    """A refused cell's payload record must carry a stable code when one
    is declared, else the explicit ``"unavailable"`` marker, plus the
    exception type and pinned fragment -- a supported/na/silent cell's
    record must carry no ``refusal`` at all."""
    from scripts.render_compatibility import _reference_payload
    from tests.compatibility_catalog import MATRIX, PAIRS

    payload = _reference_payload()
    for capability, column in MATRIX.items():
        for metric, cell in column.items():
            record = payload["cells"][f"{capability}|{metric}"]
            if cell.status == "refused":
                assert cell.raises is not None
                assert set(record["refusal"]) == {
                    "code",
                    "code_display",
                    "exception",
                    "fragment",
                }
                assert record["refusal"]["code"] == cell.code
                assert record["refusal"]["code_display"] == (cell.code or "unavailable")
                assert record["refusal"]["exception"] == cell.raises.__name__
                assert record["refusal"]["fragment"] == cell.fragment
            else:
                assert record["refusal"] is None
    for (left, right), cell in PAIRS.items():
        record = payload["pairs"][f"{left}|{right}"]
        if cell.status == "refused":
            assert cell.raises is not None
            assert set(record["refusal"]) == {
                "code",
                "code_display",
                "exception",
                "fragment",
            }
            assert record["refusal"]["exception"] == cell.raises.__name__
            assert record["refusal"]["fragment"] == cell.fragment
            assert record["refusal"]["code_display"] == (cell.code or "unavailable")
        else:
            assert record["refusal"] is None


def test_static_details_render_owner_probe_and_refusal_for_supported_and_refused_cells():
    """No-JS static ``<details>`` fallback must show the same owner/probe/
    refusal facts the JS drawer renders -- verified for one supported
    matrix cell, one refused matrix cell, and one refused pair."""
    from tests.compatibility_catalog import CAPABILITY_PROVENANCE, PAIR_PROVENANCE

    output = render_reference()
    readable, _, _ = output.partition('<script id="compatibility-data"')
    # Supported cell: estimate x mean.
    assert f"Owner: {CAPABILITY_PROVENANCE['estimate'].owner}" in readable
    assert f"Evidence: {CAPABILITY_PROVENANCE['estimate'].probe}" in readable
    # Refused cell: cluster x quantile -- has a declared stable code.
    assert f"Owner: {CAPABILITY_PROVENANCE['cluster'].owner}" in readable
    assert "Refusal code: source.frame.cluster_capability" in readable
    assert "exception: CapabilityError" in readable
    # Refused pair: cluster x cuped.
    assert f"Owner: {PAIR_PROVENANCE[('cluster', 'cuped')].owner}" in readable
    assert "cannot combine with a CUPED covariate" in readable


def test_status_colors_meet_wcag_aa_contrast_against_the_page_background():
    """Every status color must be readable, not merely present -- each
    light-scheme token must contrast at least 4.5:1 against a white
    page background, and each dark-scheme token at least 4.5:1 against
    mkdocs-material's slate background (``hsl(225deg 15% 14%)``), the
    WCAG 2.1 AA threshold for normal-size text."""
    import re

    def _relative_luminance(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    def _contrast(hex_a: str, hex_b: str) -> float:
        lum_a, lum_b = _relative_luminance(hex_a), _relative_luminance(hex_b)
        lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
        return (lighter + 0.05) / (darker + 0.05)

    css = (ROOT / "docs/stylesheets/compatibility.css").read_text()
    default_scheme, _, slate_scheme = css.partition('[data-md-color-scheme="slate"]')
    status_alphabet = set(get_args(RuntimeStatus)) | set(get_args(FamilyStatus))
    light_bg, dark_bg = "#ffffff", "#1e2129"
    for status in status_alphabet:
        token = f"--compat-status-{status.replace('_', '-')}"
        light_match = re.search(rf"{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}})", default_scheme)
        dark_match = re.search(rf"{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}})", slate_scheme)
        assert light_match, (status, "light token is not a plain hex color")
        assert dark_match, (status, "dark token is not a plain hex color")
        assert _contrast(light_match.group(1), light_bg) >= 4.5, (status, "light contrast")
        assert _contrast(dark_match.group(1), dark_bg) >= 4.5, (status, "dark contrast")


def test_render_has_no_global_filter_form_or_hint():
    """The reference is click-driven per cell, not filtered globally --
    there is no filter form, no per-axis dropdown, and no filter-completeness
    hint anywhere in the page. Regression guard against reintroducing the
    removed global-filter UX."""
    output = render_reference()
    assert 'id="compatibility-filters"' not in output
    assert "data-axis=" not in output
    assert "compatibility-filter-hint" not in output


def test_pair_reason_cell_is_a_plain_static_cell():
    """The pairs table's Reason cell never needs a JS hook -- pair
    baselines never change after page load -- so it carries no
    data-left/data-right identity attributes or dedicated class, unlike
    when it tracked an active filter."""
    import html as html_module

    from tests.compatibility_catalog import PAIRS

    output = render_reference()
    assert "compatibility-pair-reason" not in output
    left, right = next(iter(sorted(PAIRS)))
    cell = PAIRS[(left, right)]
    row = output[output.index(f'data-left="{left}" data-right="{right}"') :]
    reason_cell = row.split("</td><td>", 1)[1].split("</td>", 1)[0]
    assert "data-left=" not in reason_cell, "reason cell tracked a pair identity attribute"
    assert "data-right=" not in reason_cell, "reason cell tracked a pair identity attribute"
    text = (cell.note or cell.reason or cell.fragment or "").rstrip()
    assert text and html_module.escape(text) in reason_cell


def test_base_always_valid_quantile_scenario_declares_explicit_family_role_and_multiplicity():
    """The base ``always-valid-quantile`` scenario previously declared only
    ``inference``, making it ambiguous against the ``...-secondary-family``
    scenario once both are listed together as known interactions for the
    same ``sequential`` x ``quantile`` cell. It must now also declare an
    explicit ``family_role``/``multiplicity`` so its interaction label
    reads as an explicit, distinguishable primary/disabled context rather
    than leaving those two axes implicit."""
    payload = _reference_payload()
    axes = payload["scenarios"]["always-valid-quantile"]["axes"]
    assert axes["family_role"] == "primary"
    assert axes["multiplicity"] == "disabled"
    secondary_axes = payload["scenarios"]["always-valid-quantile-secondary-family"]["axes"]
    assert secondary_axes["family_role"] == "secondary"
    assert secondary_axes["multiplicity"] == "enabled"


def test_static_scenario_details_render_context_axes_with_the_same_labels_as_the_interactive_summary():
    """The no-JS static ``<details>`` fallback for each scenario must show
    the same human-readable context-axis label (e.g. "Family Role:
    primary, Inference: always valid, Multiplicity: disabled") that
    ``compatibility.js``'s ``interactionLabel()`` builds for the
    interactive Known-interactions summary -- a no-JS reader must see the
    same disambiguating context, not just Runtime/Family, or the two
    scenarios sharing the ``sequential`` x ``quantile`` identity would be
    indistinguishable without JavaScript."""
    output = render_reference()
    readable, _, _ = output.partition('<script id="compatibility-data"')
    assert "Family Role: primary, Inference: always valid, Multiplicity: disabled" in readable
    assert "Family Role: secondary, Inference: always valid, Multiplicity: enabled" in readable


def test_quantile_refusal_scenarios_do_not_claim_recovery_evidence():
    """An unsupported likelihood cannot turn refusal evidence into coverage proof."""
    from tests.compatibility_catalog import SCENARIOS

    by_id = {scenario.id: scenario for scenario in SCENARIOS}
    assert "always-valid-quantile" in by_id
    assert "always-valid-quantile-secondary-family" in by_id
    primary = by_id["always-valid-quantile"]
    secondary = by_id["always-valid-quantile-secondary-family"]
    assert dict(primary.axes)["family_role"] == "primary"
    assert dict(primary.axes)["multiplicity"] == "disabled"
    assert dict(secondary.axes)["family_role"] == "secondary"
    assert dict(secondary.axes)["multiplicity"] == "enabled"
    assert not any(ref.kind == "parameter_recovery" for ref in primary.evidence)
    assert not any(ref.kind == "parameter_recovery" for ref in secondary.evidence)
