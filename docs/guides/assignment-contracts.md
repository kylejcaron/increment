# Versioned assignment contracts

An assignment contract declares reproducible fixed-allocation rules for a
parallel randomized experiment. Use the reference assigner or an external
implementation that passes the same conformance vectors. Assignment is an
enrollment decision; log actual exposure separately.

```python
from increment.assignment import (
    AssignmentLayer,
    AssignmentNamespace,
    AssignmentShare,
    AssignmentSpec,
    assign,
    audit_assignments,
    validate_assignment_specs,
)

namespace = AssignmentNamespace(
    name="web",
    version="1",
    salt="web-randomization",
    identity_key="user_id",
    holdout_buckets=1000,
)
layer = AssignmentLayer(
    namespace=namespace,
    name="product",
    version="1",
    salt="product-routing",
    experiments=(
        AssignmentShare(name="checkout", buckets=6000),
        AssignmentShare(name="search", buckets=3000),
    ),
)
contract = AssignmentSpec(
    experiment="checkout",
    version="1",
    salt="checkout-arms",
    eligibility_version="all-users-v1",
    layer=layer,
    control_group="control",
    arms=(
        AssignmentShare(name="control", buckets=8000),
        AssignmentShare(name="treatment", buckets=2000),
    ),
)
validate_assignment_specs([contract])
record = assign(contract, "user-123", eligible=True)
report = audit_assignments(contract, [record])
assert report.replay == "matched"
```

The namespace reserves 10% of units as holdouts across **all** its layers.
Among the remainder, checkout receives 60% and search 30%; the remaining 10%
is outside this layer's experiments. Checkout's conditional arm split is
80% control and 20% treatment. Different layers have distinct routing and arm
hash domains. Experiments sharing one layer use disjoint experiment slots.
Call `validate_assignment_specs()` with all specs deployed together to reject
conflicting namespace, layer, or experiment declarations. Sharing a name and
version requires sharing the complete corresponding rules.

## Identity, eligibility, and versions

`unit_id` is a nonempty UTF-8 string at the randomization grain. Numbers and
booleans are refused rather than silently converted. Choose a stable identity
representation before deployment; Unicode normalization is the caller's
responsibility. A cluster randomized experiment supplies cluster IDs and uses
that column as `identity_key`.

`eligible` is a pre-assignment boolean supplied by your eligibility system.
`eligibility_version` names that system's fixed policy; replay does not verify
that the policy was evaluated correctly. Post-assignment triggers and observed
exposure never determine this input.

Persist the original contract and assignment decision at first enrollment.
Repeated requests for an existing unit reuse that enrollment, even if its
current eligibility or the active allocation has changed. Create a new contract
version for **new enrollments**, and call
`old_contract.validate_successor(new_contract)` before deploying it. Changed
namespace rules require a new namespace version; changed layer rules require
a new layer version; changed experiment rules require a new experiment version.
Versions identify immutable configurations, not rolling dates.

The reference assigner is stateless. The enrollment system owns the persisted
version choice and must pin namespace/layer membership across sibling
experiments as well. In particular, do not re-enroll an old unit into a changed
layer roster while claiming the original mutual exclusion guarantee. Changing
versions or salts can change assignments. Keep historical contracts available
and audit each version's logs separately; a single fixed-allocation analysis
must not silently pool enrollment versions with different laws. Identity
migration requires an explicit enrollment migration outside this fixed contract.

## Three separate integrity readings

`audit_assignments()` consumes **assignment** logs, with one enrollment record
per randomization unit for one contract. Each record includes eligibility,
version, full contract fingerprint, status, and nullable arm.

| Reading | Meaning |
|---|---|
| `replay` | Every supplied recorded decision matches this contract, or mismatches list the unit, fields, logged decision, and expected decision. An empty log is `not_checked`. |
| `exclusions` | No supplied log assigns an ineligible, holdout, or outside-layer unit to an arm. Violations identify the units. An empty log is `not_checked`. |
| `allocation` | The existing always-valid allocation diagnostic compares **logged** arm counts with the declared conditional split. `not_rejected` is a non-rejection, not certification. |

Duplicate units refuse (`assignment.log.duplicate`). Wrong versions/fingerprints,
unknown arms, or inconsistent enrollment statuses make allocation unavailable
(`integrity.counts_missing`) rather than publishing a partial balance result.
Replay can fail while allocation is not rejected; an imbalanced distribution
can also match its hash perfectly. Audit eligibility and identity independently.
A complete exposure log, trustworthy identifiers, and causal validity are not
established by any of these readings.

For cluster assignments pass `randomization_grain="cluster"` and supply one
record per cluster. Do not duplicate a cluster decision for every member.

## Analysis integration

Declare `assignment_contract=contract` on `Experiment` (the same nested fields
can be supplied in YAML). The experiment validates its name, control group,
randomization-grain column, allocation, and independent assignment law before
loading data. An omitted allocation/scheme derives from the contract. An
explicit allocation can use proportional weights but must describe the same
split. Definitions and published unit-day artifacts preserve this declaration.

For `Analysis.from_unit_summary`, `Analysis.from_unit_panel`, and
`Analysis.from_moments`, pass `design=contract.design()`. This carries the same
law to the existing analysis and automatic allocation checks. It does not
carry raw assignment logs or replay evidence. Run `audit_assignments()` on the
separate assignment-grain log; no analysis ingress automatically certifies
replay from exposure facts or aggregated moments.

Existing externally assigned experiments can continue declaring only their
allocation and law, or neither. They acquire no replay claim. Switchback,
encouragement, observational, adaptive, quota, blocked, and fixed-count designs
are outside this fixed parallel contract; use their existing declarations.

The contract changes assignment only. CUPED, ratio metrics, clustering,
sequential inference, winsorization, breakouts, and multiplicity retain their
existing compatibility rules. Its allocation must stay fixed for the analyzed
cohort, and eligibility must precede assignment. This contract does not make
an otherwise unpredictable sequential transform admissible.

## Portable protocol v1

The schema accepts only `schema_version=1` and `algorithm="sha256-json-v1"`.
Arm shares total exactly 10,000 positive integer buckets; layer shares total
at most 10,000. Duplicate share names refuse. Direct assignment-model
`model_validate_json()` calls reject duplicate JSON object keys. When loading
an enclosing Experiment/Definitions document or predecoded data, require its
parser to reject duplicate keys before passing the decoded contract.
Sort shares lexicographically by
Unicode scalar value (equivalently UTF-8 byte order), independently of input
array order. Bucket ranges are half-open; a boundary belongs to the next share.

For each draw, compactly JSON-encode the array below as UTF-8, without Unicode
normalization, a BOM, spaces, or escaped non-ASCII characters. Encode quote and
backslash as `\"` and `\\`. Use the short escapes `\b`, `\t`, `\n`, `\f`, and
`\r` for their respective controls. Encode every other U+0000–U+001F control as
lowercase `\u00xx`; all other Unicode scalars stay verbatim, including `/`.
Alternative JSON escapes are not protocol-equivalent bytes. Hash with
SHA-256, interpret **all 32 digest bytes** as an unsigned big-endian integer,
and take the remainder modulo 10,000.

Let `I` be the five string components
`namespace.name, namespace.version, namespace.salt, namespace.identity_key, unit_id`.

| Draw | JSON array elements, in order |
|---|---|
| Holdout | `"increment.assignment.v1", "holdout", I...` |
| Layer | `"increment.assignment.v1", "layer", I..., layer.name, layer.version, layer.salt` |
| Arm | `"increment.assignment.v1", "arm", I..., layer.name, layer.version, layer.salt, experiment, version, salt, eligibility_version` |

Check eligibility first, then namespace holdout (`bucket < holdout_buckets`),
then the layer slot, then the arm slot. Holdout and layer draws deliberately
exclude the experiment/version, preserving shared exclusions within the same
namespace/layer configuration. The arm domain includes the experiment/version.

`canonical_json()` serializes the full specification with lexicographically
sorted object keys, name-sorted share arrays, the same compact JSON encoding,
and all default fields included. `fingerprint` is the lowercase hex SHA-256 of
those bytes. No aliases, runtime values, or omitted default fields participate.
Retain this fingerprint with every enrollment. Changes in field order or
share-array order cannot change it.

The fixed randomization law assumes the domain-separated hash behaves as
independent uniform draws for the stable input population, and that salts and
rules are chosen before observing outcomes. SHA-256 hashing is deterministic,
not a source of fresh randomness or proof of independent identity. Modulo bias
per bucket is less than `1 / 2**256`; probabilities use the declared bucket
fractions. Do not search salts for favorable assignments.

The language-neutral fixture
[`tests/fixtures/assignment_v1.json`](https://github.com/kylejcaron/increment/blob/main/tests/fixtures/assignment_v1.json)
contains the full spec, canonical bytes, fingerprint, individual buckets, and
logged decisions, including Unicode, control characters, exclusions, and
ineligibility. External implementations should replay these exact vectors.
Language bindings and standards adapters can consume this protocol separately
from Increment's statistical core.
