"""AST gate: package raiser calls resolve against their bound registries."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tests.refusal_registry import (
    resolve_literal_raise_sites,
    validate_raise_sites,
)


def _fixture_module(name: str, source: str) -> None:
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), vars(module))


def test_bound_registry_gate_checks_local_key_and_known_template_context() -> None:
    name = "tests._refusal_registry_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_SHARED = RefusalSpec("shared.code", InvalidRequestError, template="{value}")
_LOCAL = {
    "local.key": RefusalSpec("public.code", InvalidRequestError, template="{value}"),
    "alias_one": _SHARED,
    "alias_two": _SHARED,
}
_ALIAS_REGISTRY = _LOCAL
_LOCAL_KEY = "local.key"
_raise = raiser(_ALIAS_REGISTRY)
_OTHER = RefusalSpec("external.code", InvalidRequestError, template="{value}")
def valid_aliases():
    _raise("alias_one", value=1)
    _raise("alias_two", value=1)
    _raise(_LOCAL_KEY, value=1)
def missing_foreign():
    _raise("external.code", value=1)
def missing_splat():
    _raise("local.key", typo=1)
    _raise("absent.key", **context)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    violations = validate_raise_sites(sites)
    assert len(violations) == 3
    for key in ("external.code", "absent.key", "local.key"):
        assert sum(key in violation for violation in violations) == 1


def test_render_callback_still_requires_local_registration() -> None:
    name = "tests._refusal_render_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"callback": RefusalSpec("callback", InvalidRequestError, lambda *, value, **_: str(value))}
_raise = raiser(_LOCAL)
def produce():
    _raise("absent.callback", **context)
"""
    _fixture_module(name, source)
    violations = validate_raise_sites(resolve_literal_raise_sites(name, source))
    assert len(violations) == 1
    assert "absent.callback" in violations[0]


def test_local_function_named_raise_is_not_a_package_refusal() -> None:
    name = "tests._refusal_shadow_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
_raise = raiser(_LOCAL)
def ordinary():
    def _raise(code, **kwargs):
        return None
    _raise("not.a.refusal", value=1)
def package():
    _raise("local", value=1)
lambda _raise: _raise("not.a.refusal", value=1)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert len(sites) == 1
    assert sites[0].lookup_key == "local"
    assert validate_raise_sites(sites) == ()


def test_global_nonlocal_and_class_scopes_follow_python_name_binding() -> None:
    name = "tests._refusal_scope_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
_raise = raiser(_LOCAL)
class ShadowedClass:
    _raise = lambda code, **kwargs: None
    _raise("class.local", value=1)
    def package_call(self):
        _raise("local", value=1)
def global_call():
    global _raise
    _raise("local", value=1)
def outer():
    _raise = lambda code, **kwargs: None
    def inner():
        nonlocal _raise
        _raise("nonlocal.local", value=1)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [site.lookup_key for site in sites] == ["local", "local"]
    assert validate_raise_sites(sites) == ()


def test_same_line_lambda_scopes_match_their_compiler_bindings() -> None:
    name = "tests._refusal_same_line_scope_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
_raise = raiser(_LOCAL)
def produce():
    return (lambda _raise: _raise("not.package", value=1), lambda: _raise("local", value=1))
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [site.lookup_key for site in sites] == ["local"]
    assert validate_raise_sites(sites) == ()


def test_generic_class_and_function_keep_package_raiser_binding() -> None:
    name = "tests._refusal_generic_scope_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
_raise = raiser(_LOCAL)
class Generic[T]:
    def call(self):
        _raise("local", value=1)
def generic_call[T]():
    _raise("local", value=1)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [site.lookup_key for site in sites] == ["local", "local"]
    assert validate_raise_sites(sites) == ()


def test_dynamic_lookup_key_is_reported_precisely() -> None:
    name = "tests._refusal_dynamic_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
_raise = raiser(_LOCAL)
def produce(code):
    _raise(code, value=1)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [(site.line, site.lookup_key) for site in sites] == [(6, None)]
    assert len(validate_raise_sites(sites)) == 1


def test_unresolved_package_raiser_binding_is_a_gate_error() -> None:
    name = "tests._refusal_unresolved_fixture"
    source = """
from increment.errors import raiser
_LOCAL = None
_raise = raiser(_LOCAL)
def produce():
    _raise("code")
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert len(sites) == 1 and sites[0].binding_error is not None
    assert len(validate_raise_sites(sites)) == 1


def test_global_registry_dispatcher_checks_its_local_spec_context() -> None:
    name = "tests._refusal_dispatch_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, refuse
_LOCAL = {"local": RefusalSpec("public.local", InvalidRequestError, template="{value}")}
def _raise(code, **kwargs):
    refuse(_LOCAL[code], **kwargs)
def produce():
    _raise("local", typo=1)
    _raise("absent", **context)
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [site.lookup_key for site in sites] == ["local", "absent"]
    assert len(validate_raise_sites(sites)) == 2


def test_shadowed_lookup_constants_remain_dynamic() -> None:
    name = "tests._refusal_key_shadow_fixture"
    source = """
from increment.errors import InvalidRequestError, RefusalSpec, raiser
_LOCAL = {"local": RefusalSpec("local", InvalidRequestError, template="{value}")}
KEY = "local"
_raise = raiser(_LOCAL)
def produce(KEY):
    _raise(KEY, value=1)
lambda KEY: _raise(KEY, value=1)
def comprehensions():
    [_raise(KEY, value=1) for KEY in ("absent",)]
    return (_raise(KEY, value=1) for KEY in ("absent",))
"""
    _fixture_module(name, source)
    sites = resolve_literal_raise_sites(name, source)
    assert [site.lookup_key for site in sites] == [None] * 4
    assert len(validate_raise_sites(sites)) == 4


@pytest.mark.slow
def test_every_literal_raise_site_supplies_exactly_its_spec_keys() -> None:
    from pathlib import Path

    import increment

    root = Path(increment.__file__).parent
    violations: list[str] = []
    dynamic: list[tuple[str, int]] = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        module_parts = path.relative_to(root).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(("increment", *module_parts))
        for site in resolve_literal_raise_sites(module_name, source):
            if site.lookup_key is None and site.binding_error is None:
                dynamic.append((str(path.relative_to(root.parent)), site.line))
            else:
                violations.extend(
                    f"{path.relative_to(root.parent)}: {violation}"
                    for violation in validate_raise_sites((site,))
                )
    assert violations == [], "\n".join(violations)
    if dynamic:
        print("Dynamic refusal keys require direct behavior coverage:", dynamic)
