# Assignment protocol v1

Use this reference to reproduce a decision from an archived configuration or to check a v1 implementation against the [conformance vectors](assignment-v1-vectors.json). This protocol is not the `AssignmentSpec` ledger protocol: its version, byte encoding, and partition rules are separate. Do not use the legacy verifier to certify these decisions.

## Configuration

Pass a decoded JSON object, not JSON text. All fields below are required except root `holdouts`; an omitted `holdouts` is an empty array. Extra fields and wrong shapes are invalid.

| Object | Fields |
|---|---|
| Root | `protocol` (number 1), `revision` (string), `experiments` (array), `layers` (array), `holdouts` (array, optional) |
| Experiment | `key` (string), `enrollment_bps` (integer), `arms` (array) |
| Arm | `key` (string), `weight_bps` (positive integer) |
| Layer | `key` (string), `ranges` (array) |
| Range | `experiment` (experiment key), `start_bps` (integer), `end_bps` (integer) |
| Holdout | `key` (string), `held_out_bps` (integer), `targets` (nonempty array of layer or standalone experiment keys) |

- Names, revisions and unit IDs are nonempty strings of Unicode scalar values. Preserve them exactly (including case, spaces and NUL); do not trim, normalize, replace unpaired surrogates or coerce numbers into strings. Strings used as hash fields must fit in an unsigned 32-bit UTF-8 byte length.
- A basis point is 1/10,000 of allocation. Basis-point values are numbers with integer values from 0 through 10,000; not booleans, strings or nonfinite numbers. Each experiment has at least two distinct arms with positive weights totaling 10,000.
- Experiment keys are unique; layer keys are unique and cannot be experiment keys. A layer can be empty. Its ranges are nonempty, disjoint and half-open: `start_bps` is included, `end_bps` excluded. Every range refers to an existing experiment; an experiment can occur in at most one range across all layers. Ranges in distinct layers may overlap. An experiment in no range is standalone.
- Holdout keys are unique and cannot be layer or standalone experiment keys. Each holdout lists distinct existing decision targets (layers or standalone experiments). A target can be in at most one holdout. Listing a layer covers every experiment in that layer; listing a layered experiment directly is invalid.
- Compile and copy once. Subsequent caller edits to any part of the input must not affect an existing evaluator. The caller archives the original configuration; a revision labels it but does not guarantee uniqueness.

## Buckets

For each draw, concatenate these bytes in order:

1. ASCII `increment.alloc.v1` followed by one zero byte.
2. For each of `stage`, `key`, and `unit_id`, a four-byte unsigned big-endian length of the UTF-8 string, then exactly that many bytes.
3. A four-byte unsigned big-endian attempt counter, starting at zero.

Hash the frame with SHA-256. Interpret the first four digest bytes as an unsigned big-endian integer `x`. If `x < 4294960000`, return `x % 10000`. Otherwise increment the attempt counter and hash again. Counter exhaustion is an internal error; never wrap or turn it into a negative decision. The cutoff leaves exactly the same number of accepted words per bucket.

| `stage` | `key` |
|---|---|
| `holdout` | Holdout key (not the decision target) |
| `layer` | Layer key |
| `enroll` | Experiment key |
| `arm` | Experiment key |

Revision, rates, weights, and array order are not hash inputs. For example, `enroll/e/1285995` first gives `fffffdd3` (4,294,966,739), which must be rejected. Attempt 1 gives `cd1eda00` (3,441,351,168), hence bucket 1168. An enrollment threshold of 5000 includes the unit.

## Decision order and records

Validate the target and unit ID before drawing. A decision target is a layer or a standalone experiment; it is never a layered experiment key. A holdout membership lookup takes a holdout key and uses the same draw as a decision for a covered target.

1. For a covered target, draw `holdout`. A bucket below `held_out_bps` gives `held_out` and stops; otherwise membership is `eligible` and the remaining gates apply. Uncovered targets have no membership.
2. For a layer target, draw `layer` and select the range containing that bucket. A gap (including an empty layer) is `unallocated`. A standalone target goes straight to enrollment in that experiment.
3. Draw `enroll` on the selected experiment. A bucket at or above `enrollment_bps` is `not_enrolled`.
4. Draw `arm`. Sort arms by unsigned UTF-8 key bytes. Starting at zero, lay out cumulative half-open ranges of width `weight_bps` in that order. The arm containing the bucket is `assigned`.

A membership record has exactly `protocol` (1), `revision`, `holdout` (key), `group` (`held_out` or `eligible`), and `group_bps` (the holdout rate for `held_out`, its complement for `eligible`).

A decision has exactly `protocol` (1), `revision`, `target` (key), `membership` (record or null), `status`, and the fields below. Covered decisions carry membership even if a later gate stops them. The probabilities for later gates are conditional, not the overall share of traffic assigned to an arm.

| `status` | `experiment`, `layer_bps`, `enrollment_bps` | `arm`, `arm_bps` |
|---|---|---|
| `held_out`, `unallocated` | All null | Both null |
| `not_enrolled` | Selected key, range width (10,000 if standalone), enrollment rate | Both null |
| `assigned` | Selected key, range width (10,000 if standalone), enrollment rate | Selected key, arm weight |

The record does not include the unit ID or delivery metadata. Assignment is not exposure, and a held-out unit is not assigned to the control arm. Increasing only enrollment leaves prior members' arms unchanged. Changing an occupied range or arm weights can move existing members; mutual exclusion is only within one common configuration and layer.

## Refusals and conformance

| Code | Meaning |
|---|---|
| `allocation.invalid_config` | Input violates a configuration rule above |
| `allocation.invalid_target` | Target is not a valid string of the required kind or is unknown; includes a direct decision request for a layered experiment |
| `allocation.invalid_unit` | Unit ID is not a valid string for this protocol |

Errors must not include raw unit IDs. If several fields are invalid, the order or wording of refusals is not part of the protocol. The [vectors](assignment-v1-vectors.json) include exact frames and digests, expected JSON decisions and membership records, and stable error codes. In a checkout, run `python3 scripts/check_assignment_vectors.py` to verify the successful records against the underlying draws without importing an SDK.
