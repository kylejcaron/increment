# Increment assignment for JavaScript

Use `@increment/assignment` to choose an experiment arm from a stable unit ID
and a versioned allocation configuration. The ESM package uses `TextEncoder`
and Web Crypto SHA-256. It runs in Node.js 20 or later, secure browser contexts,
and edge runtimes that provide those APIs. Consumers do not need TypeScript
to run it.

From this package directory, run `npm ci`, `npm run build`, and `npm pack`.
Install the resulting `increment-assignment-0.0.0.tgz` in your application. The
package is private; these steps do not publish it.

- Import `compileAllocation` from `@increment/assignment`. Pass it a decoded
  configuration (not JSON text). It validates and copies the configuration;
  later edits do not change that evaluator.
- Call `await evaluator.decide(target, unitId)` with a layer or standalone
  experiment key. An `assigned` result names the chosen arm. For `held_out`,
  `unallocated`, or `not_enrolled`, serve your usual behavior; the unit has
  not been assigned to an arm.
- Call `await evaluator.holdoutMembership(holdoutKey, unitId)` to get the unit's
  holdout group before it sees any covered feature. Both `held_out` and
  `eligible` memberships are needed for a holdout comparison. Eligibility alone
  does not mean a unit was assigned to an experiment.

The evaluator does not store IDs, log events, or make network calls. Archive
configurations by revision and record actual delivery separately from decisions.
If a server already makes the decision, send that result to the browser rather
than deciding again. Do not use a browser decision to enforce access.

Invalid configuration throws synchronously with `allocation.invalid_config`.
The async methods reject with `allocation.invalid_target` or
`allocation.invalid_unit` for invalid inputs. Read the error's `code` rather
than matching its message.

See the repository's [Assignment protocol v1](../../docs/reference/assignment-protocol-v1.md)
for configuration fields, probability meanings, and replay rules.
