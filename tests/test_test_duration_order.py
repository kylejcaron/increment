from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from conftest import _group_markdown_items, _order_by_recorded_duration


def _item(nodeid, path):
    item = SimpleNamespace(
        nodeid=nodeid,
        location=(path, 1, nodeid),
        path=Path(path),
        marks=[],
    )
    item.add_marker = lambda marker, marks=item.marks: marks.append(marker)
    item.iter_markers = lambda name: (mark for mark in item.marks if mark.name == name)
    item.get_closest_marker = lambda name: next(
        (marker for marker in item.marks if marker.name == name), None
    )
    return item


def test_recorded_call_durations_sort_longest_first_stably(tmp_path):
    durations = tmp_path / "durations.json"
    durations.write_text('{"first": 3, "slow": 9, "tied": 9}')
    items = [_item(name, "tests/test_sample.py") for name in ("first", "slow", "tied", "unknown")]

    _order_by_recorded_duration(items, durations)

    assert [item.nodeid for item in items] == ["slow", "tied", "first", "unknown"]


def test_missing_duration_file_preserves_collection_order(tmp_path):
    items = [_item("second", "tests/test_sample.py"), _item("first", "tests/test_sample.py")]

    _order_by_recorded_duration(items, tmp_path / "missing.json")

    assert [item.nodeid for item in items] == ["second", "first"]


def test_markdown_items_stay_ordered_and_grouped_by_file(tmp_path):
    first = _item("README.md::example[2]", "README.md")
    second = _item("README.md::example[1]", "README.md")
    other = _item("docs/guide.md::example[1]", "docs/guide.md")
    ordinary = _item("tests/test_sample.py::slow", "tests/test_sample.py")
    durations = tmp_path / "durations.json"
    durations.write_text(
        '{"README.md::example[2]@docs:README.md": 10, "README.md::example[1]": 1, '
        '"tests/test_sample.py::slow": 5}'
    )
    items = [first, second, other, ordinary]

    _group_markdown_items(items)
    _order_by_recorded_duration(items, durations)

    assert items == [ordinary, first, second, other]
    assert first.marks[0].args == ("docs:README.md",)
    assert second.marks[0].args == ("docs:README.md",)
    assert other.marks[0].args == ("docs:docs/guide.md",)


def test_xdist_workers_receive_one_controller_duration_snapshot(tmp_path, monkeypatch):
    import conftest

    duration_path = tmp_path / ".test_durations"
    duration_path.write_text('{"first": 4}')
    monkeypatch.setattr(conftest, "_DURATION_SNAPSHOT", None)

    class Node:
        config = SimpleNamespace(rootpath=tmp_path)

        def __init__(self):
            self.workerinput = {}

    first_worker = Node()
    conftest.pytest_configure_node(first_worker)
    duration_path.write_text('{"second": 9}')
    second_worker = Node()
    conftest.pytest_configure_node(second_worker)

    assert first_worker.workerinput["increment_test_duration_snapshot"] == (("first", 4.0),)
    assert second_worker.workerinput["increment_test_duration_snapshot"] == (("first", 4.0),)


def test_split_runs_do_not_rewrite_shared_duration_weights(tmp_path, monkeypatch):
    import conftest

    path = tmp_path / ".test_durations"
    original = '{"test_old": 12.0}\n'
    path.write_text(original)
    monkeypatch.setattr(conftest, "_CALL_DURATIONS", {"test_new": 0.5})
    config = SimpleNamespace(
        rootpath=tmp_path,
        getoption=lambda name, default=None: 4 if name == "splits" else default,
    )
    session = cast(pytest.Session, SimpleNamespace(config=config))

    conftest.pytest_sessionfinish(session, 0)

    assert path.read_text() == original


def test_testmon_collection_reordering_is_restored_to_longest_first(tmp_path):
    import conftest

    conftest._DURATION_SNAPSHOT = None
    from scripts.run_test_tier_plugin import _restore_duration_order_after_testmon

    durations = tmp_path / ".test_durations"
    durations.write_text('{"slow": 12, "fast": 1}')
    items = [
        _item("fast", "tests/test_sample.py"),
        _item("slow", "tests/test_sample.py"),
    ]
    config = SimpleNamespace(
        rootpath=tmp_path,
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: None,
            hasplugin=lambda name: name == "testmon.pytest_testmon",
        ),
    )
    _restore_duration_order_after_testmon(cast(pytest.Config, config), items)
    assert [item.nodeid for item in items] == ["slow", "fast"]


def test_testmon_restores_markdown_source_order_after_duration_sort(tmp_path, monkeypatch):
    import conftest
    import scripts.run_test_tier_plugin as plugin

    conftest._DURATION_SNAPSHOT = None
    durations = tmp_path / ".test_durations"
    durations.write_text(
        '{"README.md::example[1]": 1, "README.md::example[2]": 10, "ordinary-slow": 20}'
    )
    setup = _item("README.md::example[1]", "README.md")
    dependent = _item("README.md::example[2]", "README.md")
    ordinary_slow = _item("ordinary-slow", "tests/test_sample.py")
    items = [setup, dependent, ordinary_slow]
    config = SimpleNamespace(
        rootpath=tmp_path,
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: None,
            hasplugin=lambda name: name == "testmon.pytest_testmon",
        ),
        getoption=lambda name: False,
        stash={},
    )
    monkeypatch.delenv("INCREMENT_AFFECTED_EVIDENCE", raising=False)
    monkeypatch.delenv("INCREMENT_DISABLE_DURATION_ORDER", raising=False)
    hook = plugin.pytest_collection_modifyitems(cast(pytest.Config, config), items)
    next(hook)
    for item in (setup, dependent):
        item.nodeid += "@docs:README.md"
    items[:] = [dependent, ordinary_slow, setup]

    with pytest.raises(StopIteration):
        next(hook)
    assert [item.nodeid for item in items] == [
        "ordinary-slow",
        "README.md::example[1]@docs:README.md",
        "README.md::example[2]@docs:README.md",
    ]


@pytest.mark.parametrize(
    ("registered_name", "testmon_plugin"),
    [
        ("pytest-testmon", None),
        ("testmon.pytest_testmon", None),
        (None, object()),
    ],
)
def test_testmon_registration_names_are_recognized(registered_name, testmon_plugin):
    from scripts.run_test_tier_plugin import _testmon_is_registered

    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(
            get_plugin=lambda name: testmon_plugin if name == "TestmonSelect" else None,
            hasplugin=lambda name: name == registered_name,
        )
    )

    assert _testmon_is_registered(cast(pytest.Config, config))
