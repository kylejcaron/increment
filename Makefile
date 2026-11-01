.DEFAULT_GOAL := install
.PHONY: install fmt lint audit-verbosity complexity complexity-diff typecheck typecheck-sync test test-unit test-integration-fast test-tier test-fast test-slow test-parameter-recovery test-numerical-certification test-scientific test-all test-versions test-examples test-doc-snippets prek check ci-local sim demo examples test-changed test-affected check-items mutate selection-map benchmark

BENCHMARK_BASELINE ?=
benchmark:
	uv run python -m scripts.check_benchmark_names tests/fixtures/benchmarks/step4-pre-kernel.json
ifeq ($(strip $(BENCHMARK_BASELINE)),)
	uv run pytest tests/benchmarks/test_analysis_benchmark.py -m benchmark -n 0 --benchmark-only --benchmark-save=local-baseline
else
	uv run pytest tests/benchmarks/test_analysis_benchmark.py -m benchmark -n 0 --benchmark-only --benchmark-compare="$(BENCHMARK_BASELINE)" --benchmark-compare-fail=mean:25%
endif

CHECK_ITEMS_TESTS ?= tests
CHECK_ITEMS_REPO ?= .
UNIT_CPU_ARGS ?= $(if $(filter unit fast,$(TIER)),--unit-cpu-budget,)


# Collected items per tier: merge base of BASE versus the working tree.
check-items:
	$(SLOT) uv run --no-sync python -m scripts.check_collect_count --base $(or $(BASE),origin/main) --repo $(CHECK_ITEMS_REPO) --tests $(CHECK_ITEMS_TESTS)

# Loadgroup keeps modules sharing mutable filesystem paths on one worker.
mutate:
	@test -n "$(CAMPAIGN)" || { echo 'usage: make mutate CAMPAIGN=<name> [KILL_MATRIX=1]'; exit 2; }
	$(SLOT) uv run --no-sync python scripts/mutation_campaign.py $(CAMPAIGN) --output mutation-results $(if $(filter 1,$(KILL_MATRIX)),--kill-matrix,)
SLOT = uv run --no-sync python -m scripts.test_slots --
PYTEST = uv run --extra demo --extra tables --extra dashboard pytest
WORKERS ?=
# @WORKERS@ is replaced by the slot wrapper with the allowance it computes after acquiring a slot.
PYTEST_XDIST_ARGS ?= -n $(if $(strip $(WORKERS)),$(WORKERS),@WORKERS@) --dist loadgroup
PYTEST_ARGS ?=
TIER_MARKER = $(shell uv run python -c "from noxfile import TIER_MARKERS; print(TIER_MARKERS['$(TIER)'])")
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
	uv run ty check tests/typed_builder_calls.py --error-on-warning --warn unused-ignore-comment
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
test-unit:
	$(MAKE) test-tier TIER=unit
test-fast:
	$(MAKE) test-tier TIER=fast
test-integration-fast:
	$(MAKE) test-tier TIER=integration-fast
# Fast tiers stop at the first failure; long tiers report every failure.
test-tier:
	$(SLOT) $(PYTEST) --benchmark-skip -m '$(TIER_MARKER)' $(if $(filter unit fast integration-fast,$(TIER)),-x) -q $(PYTEST_XDIST_ARGS) $(UNIT_CPU_ARGS) $(PYTEST_ARGS) $(SHARD_ARGS)
# Functional tests exclude Monte-Carlo calibration and notebook subprocesses.
test-slow:
	$(MAKE) test-tier TIER=slow
# One optional shard of the Monte-Carlo tier. Without machine-specific timing
# data, pytest-split distributes collected tests evenly.
# Example: `make test-parameter-recovery SPLITS=4 GROUP=1`.
test-parameter-recovery:
	@if [ -z "$(SPLITS)" ] || [ -z "$(GROUP)" ]; then \
		echo "usage: make test-parameter-recovery SPLITS=<n> GROUP=<1..n>" >&2; \
		exit 1; \
	fi
	$(MAKE) test-tier TIER=parameter-recovery SHARD_ARGS='--splits $(SPLITS) --group $(GROUP)'

# Required deterministic numerical certification; not a Monte-Carlo campaign.
test-numerical-certification:
	$(MAKE) test-tier TIER=numerical-certification
# Union selector for nightly scientific coverage; split shards retain both purposes.
test-scientific:
	@if [ -z "$(SPLITS)" ] || [ -z "$(GROUP)" ]; then \
		echo "usage: make test-scientific SPLITS=<n> GROUP=<1..n>" >&2; \
		exit 1; \
	fi
	$(MAKE) test-tier TIER=scientific SHARD_ARGS='--splits $(SPLITS) --group $(GROUP)'
test-all:
	$(MAKE) test-tier TIER=all
test-versions:
	uv run --with nox nox -s tests
test-examples:
	$(MAKE) test-tier TIER=examples
test-doc-snippets:
	uv run --with nox nox -s docs_snippets
prek:
	uv run prek run --all-files
test:
ifneq ($(strip $(TESTS)),)
ifeq ($(words $(TESTS))$(findstring ::,$(TESTS)),1::)
	$(PYTEST) --benchmark-skip $(TESTS) -x -q -m "" $(PYTEST_ARGS)
else
	$(SLOT) $(PYTEST) --benchmark-skip $(TESTS) -x -q -m "" $(PYTEST_XDIST_ARGS) $(PYTEST_ARGS)
endif
else
	$(MAKE) test-unit

endif

test-changed:
	$(SLOT) uv run --no-sync --with pytest-testmon==2.2.0 --extra demo --extra tables --extra dashboard python -m scripts.test_changed

selection-map:
	$(MAKE) test-fast WORKERS=4 PYTEST_ARGS="--cov=increment --cov-context=test --cov-report= --selection-fixtures .selection-fixtures tests"
	uv run --no-sync python scripts/selection_map.py build --coverage .coverage --fixtures .selection-fixtures --tier fast --output tests/selection_map.json

test-affected:
	@test -n "$(BASE)" || { echo 'usage: make test-affected BASE=<gitref> [TIER=fast|slow]' >&2; exit 2; }
	$(SLOT) uv run --extra demo --extra tables --extra dashboard python -m scripts.affected --base $(BASE) --tier $(or $(TIER),fast) -- $(PYTEST_XDIST_ARGS) --benchmark-skip $(PYTEST_ARGS)
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
