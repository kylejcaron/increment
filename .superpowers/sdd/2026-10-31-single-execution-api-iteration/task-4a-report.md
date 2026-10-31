# Task 4A report: source workflows

## Operations and baseline members

- `increment.workflows.materialize(experiment, resolver) -> None` copies the refresh from `Analysis.materialize(self) -> Self`; the new function intentionally returns nothing. Legacy `Analysis.materialize()` still returns `self` after delegating to the same binding-level operation.
- `increment.workflows.export_moments(experiment, resolver, path: str | Path) -> None` copies `Analysis.export(self, path: str | Path) -> None` and calls the existing source export operation.
- `increment.workflows.breakout_summaries(experiment, resolver, *, metrics: Sequence[str | Metric] | None = None) -> dict[str, dict[str, pa.Table]]` copies `Analysis.breakout_summaries(self, *, metrics: Sequence[str | Metric] | None = None)`.
- `increment.workflows.factor_summaries(experiment, resolver) -> dict[str, pa.Table]` copies `Analysis.factor_summaries(self)`.
- `increment.workflows.experiment_definition(experiment, resolver) -> semantics.models.Experiment` copies the semantic `Analysis.experiment` property, not the kernel `Experiment`.
- `increment.workflows.metrics(experiment, resolver) -> list[Metric]`, `compiled_decision_plan(experiment, resolver) -> CompiledDecisionPlan`, and `available_metrics(experiment, resolver) -> list[Metric]` copy the corresponding `Analysis` property values and eligibility.

## Policy, exposure, and inventory

`increment/_source_workflows.py` owns source policy and obtains bindings through `binding_for`: available-metric filtering retains definitions order and excludes declared, report-only (`total`/`active`), and wrong-entity metrics; breakout metric selection, quantile/retention restrictions, and both-population result stamping; factor quantile restrictions and both-population result stamping. Query/materialize/export execution remains delegated to the existing source owners. The workflow owner does not construct SQL or access resolver private fields. Legacy `Analysis` methods/properties delegate to these binding-level owners.

Added `increment/workflows.py` as re-exports only and tach leaves for both workflow modules. `docs/api.md` now names the exact eight promised `increment.workflows` exports. `tests/public_exports.txt` intentionally remains unchanged: it is an exact inventory of root `increment.__all__`, not submodule exports.

## Red/green and verification

- Red: `make test TESTS='tests/test_analysis_materialization.py::test_workflow_materialize_is_a_refresh_then_run'` with a deliberate non-`None` return in `materialize`; observed `AssertionError: assert 'intentional red check' is None` (1 failed).
- Red: `make test TESTS='tests/test_analysis_factor.py::test_workflow_inspection_preserves_source_declarations'` with the semantic experiment helper deliberately returning the plan; observed `assert isinstance(semantic, SemanticExperiment)` failure (1 failed).
- Green smoke: the four operation-journey node IDs for materialize/run, moments export/reopen/run, factor summaries, and breakout summaries passed: `4 passed in 6.52s`. The separate zero-metric workflow check passed: `1 passed in 2.14s`; semantic inspection passed: `1 passed in 2.01s`.
- Final focused verification: `make test TESTS='tests/test_analysis_materialization.py tests/test_analysis_moments.py tests/test_analysis_factor.py tests/test_consumer_standard.py tests/test_test_hygiene.py::test_no_new_private_increment_imports'` → `109 passed in 18.28s`.
- Final checks: `make lint` → all lint, docstring-baseline, and verbosity checks passed; `uv run ty check` → `All checks passed!`; `uv run tach check` → `✅ All modules validated!`.

## Files and test ratchet

Changed: `increment/_source_workflows.py` (new), `increment/workflows.py` (new), `increment/_analysis_execution.py`, `tach.toml`, `docs/api.md`, `tests/test_analysis_materialization.py`, `tests/test_analysis_moments.py`, and `tests/test_analysis_factor.py`; this report is also new. `tests: +5 items — public-path smoke coverage for materialize/run, export/reopen/run, factor summaries, breakout summaries, and source inspection declarations/eligibility.` The zero-metric test was migrated to separate `materialize` and `run` calls and asserts a `None` refresh result plus empty cells.

Concerns: none outstanding. Commit: `refactor: expose source operations in workflows`.
