import pytest

from tests._shared_cache import fixture_cache_key


def test_fixture_cache_key_changes_when_fixture_source_changes(tmp_path):
    package = tmp_path / "increment"
    package.mkdir()
    (package / "engine.py").write_text("RESULT = 1\n")
    test_source = tmp_path / "test_fixture.py"
    test_source.write_text("FIXTURE = 1\n")
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("dependency==1\n")

    first = fixture_cache_key(tmp_path, test_source, lockfile)
    test_source.write_text("FIXTURE = 2\n")
    second = fixture_cache_key(tmp_path, test_source, lockfile)
    package_source = fixture_cache_key(tmp_path, test_source, lockfile)
    (package / "engine.py").write_text("RESULT = 2\n")
    third = fixture_cache_key(tmp_path, test_source, lockfile)

    assert first != second
    assert package_source == second
    assert third != second


def test_fixture_cache_key_changes_when_installed_distribution_version_changes(
    tmp_path, monkeypatch
):
    from importlib import metadata

    class Distribution:
        def __init__(self, version):
            self.metadata = {"Name": "example"}
            self.version = version

    versions = iter((Distribution("1.0"), Distribution("2.0")))
    monkeypatch.setattr(metadata, "distributions", lambda: [next(versions)])
    (tmp_path / "uv.lock").write_text("")

    first = fixture_cache_key(tmp_path)
    second = fixture_cache_key(tmp_path)

    assert first != second


def test_get_or_build_reloads_cached_fixture_without_rebuilding(tmp_path):
    from types import MappingProxyType

    from tests._shared_cache import get_or_build

    key = "fixture"
    original = MappingProxyType({"values": (1, 2), "nested": MappingProxyType({"answer": 42})})
    assert get_or_build(tmp_path, key, lambda: original) == original

    def unexpected_build():
        raise AssertionError("cached fixture builder was called")

    restored = get_or_build(tmp_path, key, unexpected_build)
    assert dict(restored) == dict(original)
    assert restored["values"] == (1, 2)
    assert restored["nested"]["answer"] == 42
    with pytest.raises(TypeError):
        restored["nested"]["answer"] = 0
