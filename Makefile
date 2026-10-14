.PHONY: install fmt lint audit-verbosity complexity complexity-diff typecheck typecheck-sync test test-fast test-slow test-parameter-recovery test-numerical-certification test-scientific test-all test-versions test-examples test-doc-snippets prek check ci-local sim demo examples test-affected

# Loadgroup keeps modules sharing mutable filesystem paths on one worker.
PYTEST = uv run --extra demo --extra tables --extra dashboard pytest
WORKERS ?= $(shell python3 -c 'import os; print(max(1, (os.cpu_count() or 1) // 3))')
PYTEST_XDIST_ARGS ?= -n $(WORKERS) --dist loadgroup
PYTEST_ARGS ?=
unexport \
	GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG GIT_CONFIG_PARAMETERS GIT_CONFIG_COUNT \
	GIT_OBJECT_DIRECTORY GIT_DIR GIT_WORK_TREE GIT_IMPLICIT_WORK_TREE GIT_GRAFT_FILE \
	GIT_INDEX_FILE GIT_NO_REPLACE_OBJECTS GIT_REPLACE_REF_BASE GIT_PREFIX \
	GIT_INTERNAL_SUPER_PREFIX GIT_SHALLOW_FILE GIT_COMMON_DIR
install:
	uv sync --group dev --extra demo --extra tables --extra dashboard
	uv run python -m scripts.install_dev_hooks --repo "$(CURDIR)"
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
	$(PYTEST) -m 'not slow and not parameter_recovery and not numerical_certification' -x -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
# Functional tests exclude Monte-Carlo calibration and notebook subprocesses.
test-slow:
	$(PYTEST) -m 'slow and not parameter_recovery and not numerical_certification and not examples' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
# One optional shard of the Monte-Carlo tier. Without machine-specific timing
# data, pytest-split distributes collected tests evenly.
# Example: `make test-parameter-recovery SPLITS=4 GROUP=1`.
test-parameter-recovery:
	@if [ -z "$(SPLITS)" ] || [ -z "$(GROUP)" ]; then \
		echo "usage: make test-parameter-recovery SPLITS=<n> GROUP=<1..n>" >&2; \
		exit 1; \
	fi
	$(PYTEST) -m 'parameter_recovery' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS) --splits $(SPLITS) --group $(GROUP)

# Required deterministic numerical certification; not a Monte-Carlo campaign.
test-numerical-certification:
	$(PYTEST) -m 'numerical_certification' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
# Union selector for weekly scientific coverage; split shards retain both purposes.
test-scientific:
	@if [ -z "$(SPLITS)" ] || [ -z "$(GROUP)" ]; then \
		echo "usage: make test-scientific SPLITS=<n> GROUP=<1..n>" >&2; \
		exit 1; \
	fi
	$(PYTEST) -m '(parameter_recovery or numerical_certification)' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS) --splits $(SPLITS) --group $(GROUP)
test-all:
	$(PYTEST) -m 'slow or not slow' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
test-versions:
	uv run --with nox nox -s tests
test-examples:
	$(PYTEST) -m 'examples' -q $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
test-doc-snippets:
	uv run --with nox nox -s docs_snippets
prek:
	uv run prek run --all-files
test:
ifneq ($(strip $(TESTS)),)
	$(PYTEST) $(TESTS) -x -q -m "" $(PYTEST_ARGS)
else
	$(MAKE) test-fast

endif

test-affected:
	@test -n "$(BASE)" || { echo 'usage: make test-affected BASE=<gitref> [TIER=fast|slow]' >&2; exit 2; }
	uv run --extra demo --extra tables --extra dashboard python -m scripts.affected --base $(BASE) --tier $(or $(TIER),fast) -- $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
sim:
	uv run --extra demo python -m increment.simulate --replications 200
demo:
	uv run --group dev --extra demo --extra tables --extra dashboard marimo edit examples/realistic_demo/
examples:
	uv run --group dev --extra demo --extra tables --extra dashboard marimo edit examples/ --no-token
check: lint typecheck test-fast test-slow

# Local CI omits interpreter/floor matrices, Windows, notebooks and live warehouses.
ci-local: lint typecheck test-fast test-slow test-doc-snippets
	uv run --with nox nox -s tests_tables
	uv run --with nox nox -s docs
	uv build
	uv run --with nox nox -s wheel_smoke wheel_smoke_demo
