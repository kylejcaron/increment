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
| `readout.*`, `readouts.*`, `estimation.*`, `breakout.*`, `contrast.*`, `conversion_inference.*`, `arm.*`, `moments.*`, `decision.*`, `design.*`, `identification.*`, `sequential.*`, `absorption.*`, `adjust.*`, `allocation.*`, `impute.*`, `unit_cycle.*` | Check the requested estimand, inference and multiplicity scope in [reading results](reading-results.md), [multiplicity](multiplicity.md), and the [method guide](choose-a-method.md). Use the route named in the refusal; do not treat an omitted or refused cell as a null effect. |
| `cate.*`, `logged_policy.*` | Check identification and support assumptions in [heterogeneity](heterogeneity-and-rollout.md) or [logged policies](logged-policy.md); select a supported estimand rather than weakening a validity gate. |
| `results.*`, `dashboard.*`, `report.*`, `reporting.*`, `tables.*` | Check result-frame projection columns and the dashboard's documented input/view requirements in the [dashboard guide](dashboard.md); use the underlying readout directly if that view is unavailable. |
| `simulate.*` | Check the simulated design and requested operating characteristic in the [power analysis guide](power-analysis.md); simulated behavior is not a guarantee for a different design. |

A missing treatment arm prevents an arm contrast, but arm counts and a separate
allocation diagnostic may still be available. A refusal for a triggered
readout does not turn an assigned-population result into a triggered estimate.
See [reading results](reading-results.md) for interpretation and next steps.
