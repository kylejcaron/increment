# When a request is refused

Increment refuses requests it cannot interpret or support honestly. A refusal is
not a result, and changing the request until the error disappears is not a
substitute for checking the design and estimand.

## Start with the code and context

Catch `CodedError` and use its stable `code` to choose handling; use `context`
for the relevant field names and values. The message is for people, not a stable
branching interface. Do not parse its wording. The [API error reference](../api.md#errors-and-refusals)
describes typed refusals and gives examples.

```python
from increment.errors import CodedError


def refusal_record(exc: CodedError) -> tuple[str, dict[str, object]]:
    return exc.code, dict(exc.context)
```

## Find the next step by code family

The first code segment is a navigation hint, not a new public enum. Follow the
specific refusal's message/context and the linked contract; similar-looking
codes can describe different limitations.

| Code family | What to check next |
| --- | --- |
| `analysis.*`, `facade.*`, `plan.*`, `power.*`, `arm_planning.*` | Check the declared analysis, population, method, planning inputs, and supported source path in the [capability reference](../reference/capabilities-by-entry-point.md), [power analysis](power-analysis.md), or [method guide](choose-a-method.md). Correct the request or use a supported entry point. |
| `source.*`, `frame.*`, `query.*`, `artifact.*`, `artifact_contract.*`, `definition.*`, `model.*`, `compatibility.*` | Check the reported column, grain, identity, assignment, artifact version, or source capability. Repair the input at its source, then follow the [data model guide](data-model.md), [API reference](../api.md), or [compatibility policy](../api.md#pre-10-compatibility). |
| `readout.*`, `readouts.*`, `estimation.*`, `breakout.*`, `contrast.*`, `conversion_inference.*`, `arm.*`, `moments.*`, `decision.*`, `design.*`, `identification.*`, `sequential.*`, `absorption.*`, `adjust.*`, `allocation.*`, `impute.*`, `unit_cycle.*` | Check the requested estimand, inference and multiplicity scope in [reading results](reading-results.md), [multiplicity](multiplicity.md), and the [method guide](choose-a-method.md). Use the route named in the refusal; do not treat an omitted or refused cell as a null effect. |
| `cate.*`, `logged_policy.*` | Check identification and support assumptions in [heterogeneity](heterogeneity-and-rollout.md) or [logged policies](logged-policy.md); select a supported estimand rather than weakening a validity gate. |
| `dashboard.*`, `report.*`, `reporting.*`, `tables.*` | Check the dashboard's documented input and view requirements in the [dashboard guide](dashboard.md); use the underlying readout directly if that view is unavailable. |
| `simulate.*` | Check the simulated design and requested operating characteristic in the [power analysis guide](power-analysis.md); simulated behavior is not a guarantee for a different design. |

A missing treatment arm, for example, means an arm-level contrast cannot be
estimated; it does not prevent reading arm counts or a separate allocation
diagnostic. A triggered readout refusal does not license interpreting the
assigned-population result as a triggered-population estimate. See
[reading results](reading-results.md) for interpretation and next actions.
