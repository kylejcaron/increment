# When a request is refused

Increment refuses a request when it cannot interpret or support it honestly.
A refusal is not a result; do not change the request just to make the error
disappear. Check the design and estimand first.

## Start with the code and context

Catch `CodedError` and branch on its stable `code`; use `context` to inspect
the relevant fields. The human-readable message can change, so do not parse
it. See the [API error reference](../api.md#errors-and-refusals) for typed
refusals and examples.

```python
from increment.errors import CodedError


def refusal_record(exc: CodedError) -> tuple[str, dict[str, object]]:
    return exc.code, dict(exc.context)
```

## Find the next step by code family

The first code segment helps you find a family; it is not a separate public
enum. Check the specific refusal's message, context, and linked contract:
similar codes can mean different things.

| Code family | What to check next |
| --- | --- |
| `analysis.*`, `facade.*`, `plan.*`, `power.*`, `arm_planning.*` | Check the declared analysis, population, method, planning inputs, and supported source path in the [capability reference](../reference/capabilities-by-entry-point.md), [power analysis](power-analysis.md), or [method guide](choose-a-method.md). Correct the request or use a supported entry point. |
| `source.*`, `frame.*`, `query.*`, `artifact.*`, `artifact_contract.*`, `definition.*`, `model.*`, `compatibility.*` | Check the reported column, grain, identity, assignment, artifact version, or source capability. Repair the input at its source, then follow the [data model guide](data-model.md), [API reference](../api.md), or [compatibility policy](../api.md#pre-10-compatibility). |
| `evidence.*` | Check `requested`, `offered`, and `route` in the refusal context. Grain, breakout, unit-row, raw-outcome, cluster, covariate, assignment, and operation codes identify which evidence is unavailable; use a source that retains it or choose an operation it supports. `evidence.assignment.unavailable` means the source has no assignment evidence; `evidence.assignment.selection_unavailable` means the requested experiment, version, period, or population is not retained; `evidence.assignment.identity_unavailable` means a replay, whole-ledger exposure, or allocation-history request needs row-level assignment records; supply a row-level ledger through `Analysis.from_definitions` with `experiment.assignment_ledger`, or a `FrameLedger`. `evidence.assignment.invalid` names malformed roles, duplicate identities, inconsistent cluster membership, invalid periods, or invalid replay specifications; `evidence.assignment.replay_mismatch` means replay disagrees with the selected ledger. Compliance declarations distinguish total from as-of evidence; operation codes name unavailable product operations; `evidence.operation_unavailable` covers an operation that the source does not retain. Follow the route in the context rather than fabricating missing evidence. |
| `query.session.warehouse_artifact.verification_unavailable` | The stored relation format cannot be verified by this warehouse route; refusal context names the relation and digest format. Do not read or relabel its hashes. Republish from trusted definitions with warehouse-side integrity enabled. |
| `readout.*`, `readouts.*`, `estimation.*`, `integrity.*`, `breakout.*`, `contrast.*`, `conversion_inference.*`, `arm.*`, `moments.*`, `decision.*`, `design.*`, `identification.*`, `sequential.*`, `absorption.*`, `adjust.*`, `allocation.*`, `impute.*`, `unit_cycle.*`, `sensitivity.*` | Check the requested estimand, inference and multiplicity scope in [reading results](reading-results.md), [multiplicity](multiplicity.md), and the [method guide](choose-a-method.md). For SRM integrity codes, declare the actual target allocation and, for always-valid checks, the assignment law. For `sensitivity.*`, check the row's estimand, declared metric type and scale in the [observational guide](observational.md#how-strong-would-an-omitted-confounder-need-to-be). Use the route named in the refusal; do not treat an omitted or refused cell as a null effect. |
| `cate.*`, `logged_policy.*` | Check identification and support assumptions in [heterogeneity](heterogeneity-and-rollout.md) or [logged policies](logged-policy.md); select a supported estimand rather than weakening a validity gate. |
| `results.*`, `dashboard.*`, `report.*`, `reporting.*`, `tables.*` | Check result-frame projection columns and the dashboard's documented input/view requirements in the [dashboard guide](dashboard.md); use the underlying readout directly if that view is unavailable. `results.operation.alpha_above_allocation` means a reissued interval exceeds its stored declared or selected family allocation; request an equal or smaller alpha, or use an unadjusted row. For `results.readout.*`, pass `readout(scope)` one of the scopes the run executed (listed by hash in `scope_hashes`); a result restored from a store or a pickle keeps its saved cells only, so run the experiment again for its full collections ([reading results](reading-results.md)). |
| `simulate.*` | Check the simulated design and requested operating characteristic in the [power analysis guide](power-analysis.md); simulated behavior is not a guarantee for a different design. |
| `package.*` | Bundled support is missing or inconsistent in the installed package. Reinstall a complete, matching Increment distribution; changing the experiment or data will not repair the installation. |
| `kernel.*` | Check the saved or shared analysis description named in the refusal context (`field`, `kind`). A scope, request or method description is inconsistent or cannot be saved portably. Request identity datetimes must include a timezone; aware offsets are normalized to UTC. For `kernel.definitions.catalog_missing`, bind the definitions with a connection to write the offline catalog; for `kernel.definitions.catalog_stale`, rebind after changing the definitions. Rebuild the description from the analysis rather than editing stored identifiers. |
| `store.*` | The saved run manifest, cell roster, scope map or immutable publication is inconsistent. Keep run IDs unique, publish a complete roster once, and use the pinned `RunHandle` for reads; correct the source run and write a new ID rather than replacing a committed run. |
| `wire.*` | A result cannot be represented in the portable wire format. `wire.procedure.contrast_context` means a switchback contrast run has in-memory cells and decisions but no portable arm decision plan; keep the result in memory, or use legacy `Analysis.run()` when its `ContrastResults` return contract is required. |

`store.run.empty` means an empty metric selection has no result cells to
persist. Select at least one metric before saving the run; an empty result may
still be inspected in memory.

`kernel.request.capture_not_supported` means sequential capture or checkpoint
adoption was placed in saved request options. Call
`increment.workflows.capture_sequential(experiment, resolver, ...)` with the
module-builder pair, then run the experiment, or pass
`capture=SequentialCaptureRequest(...)` to `run`.
`kernel.request.population_scope_only` means population was placed in request
options, or `run(population=...)` differs from a selected scope's population;
set it on `Scope(population=...)` instead.
`kernel.run.provider_execution_unavailable` means `BoundResolver` verified a
custom provider but has no execution adapter. Use the resolver returned by a
module builder to run an experiment.

`kernel.scope.family_mismatch` means a switchback view was requested from an
ordinary arm experiment. Build a switchback experiment with
`from_switchback_frame`, or choose its existing run, daily, asof, or breakout view.

`evidence.joint_grouping_unavailable` means the requested grouping properties
are not all available from one source; declare all grouping properties together on one source
or use a source that retains them.
An `estimation.estimator.adjustment.callback_contract` refusal means a
registered covariate-adjustment callback does not accept the shared
`AdjustmentInputs`, metric, design, and keyword-argument contract. Migrate it
to the current callback shape described in the
[estimator extension guide](../development/adding-an-estimator.md#custom-evidence-providers);
do not work around the check by registering the legacy positional signature.

A missing treatment arm, for example, means an arm-level contrast cannot be
estimated; it does not prevent reading arm counts or a separate allocation
diagnostic. A triggered readout refusal does not license interpreting the
assigned-population result as a triggered-population estimate. See
[reading results](reading-results.md) for interpretation and next actions.

Warehouse-backed CUPED can refuse with `query.cuped.magnitude_range` when an
arm's root-mean-square outcome component (including a ratio denominator) or
continuous covariate is outside `[1e-50, 1e50]`. The refusal context identifies
the metric, arm, component, and bound. Change units to rescale the values, or
use explicitly supplied per-unit data with `Analysis.from_unit_summary` or
`Analysis.from_unit_panel`; local per-unit fitters are unchanged.

## Recover from a missing sequential allocation

Automatic sequential registration fixes its inference boundary using the
assignment probabilities. When a public frame analysis has no allocation, pass
the **actual** allocation used to assign units; do not add a declaration that
does not match the experiment. This runnable example shows the refusal and the
corrected rerun:

```python
import pandas as pd

from increment import Analysis, AnalysisPlan, InferenceSpec, MetricSpec, Randomized
from increment.errors import CapabilityError

frame = pd.DataFrame(
    [
        {"unit": "c1", "arm": "control", "outcome": 2.0, "exposure": 1},
        {"unit": "c2", "arm": "control", "outcome": 3.0, "exposure": 2},
        {"unit": "t1", "arm": "treatment", "outcome": 4.0, "exposure": 3},
        {"unit": "t2", "arm": "treatment", "outcome": 5.0, "exposure": 4},
    ]
)
plan = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))
metrics = [MetricSpec(name="outcome", type="mean")]

try:
    Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=Randomized(control_group="control"),
        metrics=metrics,
        experiment_id="example",
        plan=plan,
        exposure_date="exposure",
    )
except CapabilityError as exc:
    assert exc.code == "sequential.source.invalid"
    assert exc.context["route"] == "design.allocation"
    assert exc.context["mechanism"] == "randomized"
    assert exc.context["required"] == "actual assignment allocation"
else:
    raise AssertionError("the missing allocation should be refused")

# This example's units were assigned 1:1; use the real allocation.
analysis = Analysis.from_unit_summary(
    frame,
    unit="unit",
    group="arm",
    design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    metrics=metrics,
    experiment_id="example",
    plan=plan,
    exposure_date="exposure",
)
results = analysis.run()
assert len(results) == 1
assert results[0].metric == "outcome"
```
