import re
from collections.abc import Iterator
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

_PINNED_REF = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _is_local_workflow(ref: str) -> bool:
    """Same-repo reusable workflows resolve to the calling commit, so they cannot be pinned."""
    return ref.startswith("./")


def _uses_refs(node: Node) -> Iterator[tuple[int, str]]:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if isinstance(key, ScalarNode) and key.value == "uses":
                ref = value.value if isinstance(value, ScalarNode) else ""
                yield value.start_mark.line + 1, ref
            yield from _uses_refs(value)
    elif isinstance(node, SequenceNode):
        for value in node.value:
            yield from _uses_refs(value)


def _unpinned_uses(path: Path) -> list[tuple[int, str]]:
    root = yaml.compose(path.read_text())
    if root is None:
        return []
    return [
        (line_no, ref)
        for line_no, ref in _uses_refs(root)
        if not _is_local_workflow(ref) and _PINNED_REF.fullmatch(ref) is None
    ]


def test_workflow_actions_are_commit_pinned() -> None:
    bad = []
    workflow_dir = Path(".github/workflows")
    paths = [*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]
    for path in paths:
        for line_no, ref in _unpinned_uses(path):
            bad.append(f"{path}:{line_no}: uses: {ref}")
    assert not bad, "\n".join(bad)


def test_workflow_guard_catches_flow_mapping(tmp_path: Path) -> None:
    workflow = tmp_path / "flow.yaml"
    workflow.write_text("jobs:\n  test:\n    steps:\n      - {uses: actions/checkout@v4}\n")
    assert _unpinned_uses(workflow) == [(4, "actions/checkout@v4")]


def test_local_reusable_workflow_is_exempt_but_third_party_still_guarded(tmp_path: Path) -> None:
    """The local-ref exemption must not become a hole for unpinned third-party actions."""
    workflow = tmp_path / "call.yaml"
    workflow.write_text(
        "jobs:\n"
        "  gate:\n"
        "    uses: ./.github/workflows/ci.yml\n"
        "  smuggled:\n"
        "    steps:\n"
        "      - uses: evil/action@main\n"
    )
    assert [ref for _line_no, ref in _unpinned_uses(workflow)] == ["evil/action@main"]


def test_remote_reusable_workflow_still_requires_a_pin(tmp_path: Path) -> None:
    workflow = tmp_path / "remote.yaml"
    workflow.write_text("jobs:\n  gate:\n    uses: other/repo/.github/workflows/ci.yml@v1\n")
    assert [ref for _line_no, ref in _unpinned_uses(workflow)] == [
        "other/repo/.github/workflows/ci.yml@v1"
    ]
