.PHONY: install fmt lint audit-verbosity complexity complexity-diff typecheck typecheck-sync test test-fast test-slow test-parameter-recovery test-all test-versions test-examples test-doc-snippets prek check ci-local sim demo examples

# Loadgroup keeps modules sharing mutable filesystem paths on one worker.
TEST_RUNNER = uv run --extra demo --extra tables --extra dashboard python -m scripts.run_test_tier
PYTEST_ARGS = -n auto --dist loadgroup
install:
	uv sync --group dev --extra demo --extra tables --extra dashboard
	uv run prek install -t pre-commit -t pre-push
fmt:
	uv run ruff format .
	uv run ruff check --fix .
lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ruff check --select PLR0915 --ignore-noqa --config 'lint.pylint.max-statements=100' increment/
	uv run tach check
	uv run vulture increment vulture_whitelist.py --min-confidence 80
	uv run python -m scripts.check_docstring_baseline
	uv run python scripts/audit_verbosity.py --gate
audit-verbosity:
	uv run python scripts/audit_verbosity.py --gate
complexity:
	uv run complexipy increment --plain --sort desc --ignore-complexity=true
complexity-diff:
	@test -n "$(BASE)" || { echo 'usage: make complexity-diff BASE=<gitref>'; exit 2; }
	@git rev-parse --verify --end-of-options "$(BASE)^{commit}"
	uv run complexipy increment --diff-only "$(BASE)" --plain --ignore-complexity=true
typecheck-sync:
	uv sync --group dev --extra demo --extra tables --extra dashboard
typecheck: typecheck-sync
	uv run ty check
test-fast:
	$(TEST_RUNNER) fast -x -q $(PYTEST_ARGS)
# Functional tests exclude Monte-Carlo calibration and notebook subprocesses.
test-slow:
	$(TEST_RUNNER) slow -q $(PYTEST_ARGS)
# One optional shard of the Monte-Carlo tier. Without machine-specific timing
# data, pytest-split distributes collected tests evenly.
# Example: `make test-parameter-recovery SPLITS=4 GROUP=1`.
test-parameter-recovery:
	@if [ -z "$(SPLITS)" ] || [ -z "$(GROUP)" ]; then \
		echo "usage: make test-parameter-recovery SPLITS=<n> GROUP=<1..n>" >&2; \
		exit 1; \
	fi
	$(TEST_RUNNER) parameter-recovery -q $(PYTEST_ARGS) --splits $(SPLITS) --group $(GROUP)
test-all:
	$(TEST_RUNNER) all -q $(PYTEST_ARGS) --evidence-root .test-evidence
test-versions:
	uv run --with nox nox -s tests
test-examples:
	python -m scripts.run_test_entrypoint examples
test-doc-snippets:
	uv run --with nox nox -s docs_snippets
prek:
	uv run prek run --all-files
test:
ifneq ($(strip $(TESTS)),)
	$(TEST_RUNNER) focused $(TESTS) -x -q
else
	$(MAKE) test-fast
endif
sim:
	uv run --extra demo python -m increment.simulate --replications 200
demo:
	uv run --group dev --extra demo --extra tables marimo edit examples/realistic_demo/
examples:
	uv run --group dev --extra demo --extra tables marimo edit examples/ --no-token
check: lint typecheck test-fast test-slow

# Local CI omits interpreter/floor matrices, Windows, notebooks and live warehouses.
# Run notebook acceptance with `python -m scripts.run_test_entrypoint examples`.
ci-local: lint typecheck test-fast test-slow test-doc-snippets
	uv run --with nox nox -s tests_tables
	uv run --with nox nox -s docs
	uv build
	uv run --with nox nox -s wheel_smoke wheel_smoke_demo
