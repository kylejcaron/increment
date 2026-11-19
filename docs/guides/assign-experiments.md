# Assign units to experiments

Use a stable unit ID to decide whether to serve baseline behavior or an experiment variant. The optional Python and JavaScript evaluators run locally and make the same decision from the same configuration and ID. If you already have an assignment system, keep it: Increment can analyze the data it produces.

## Try the packaged examples

From an Increment checkout with `uv`, Python 3.12 or newer, and Node 22 or newer, build and install the artifacts into a scratch directory:

```bash
demo=$(mktemp -d)
uv build --wheel --out-dir "$demo"
uv venv --python 3.12 "$demo/venv"
uv pip install --python "$demo/venv/bin/python" "$demo"/increment-*.whl
npm --prefix sdks/assignment-js ci
npm --prefix sdks/assignment-js run build
(cd sdks/assignment-js && npm pack --pack-destination "$demo")
cp -R examples/assignment "$demo/example"
cd "$demo"
npm install ./increment-assignment-0.0.0.tgz
./venv/bin/python example/server.py
node example/server.mjs
```

The programs print the same holdout memberships, decisions and example response bodies. They include a treatment assignment, a holdout, a gap, and an enrollment miss. The last pair of decisions uses two archived revisions: only the cart enrollment rate changes, and the previously enrolled unit keeps its holdout group and arm.

Run `node example/server.mjs --serve` and open the two printed localhost URLs. **Server decision** renders the decision from Node without evaluating again. **Browser decision** evaluates locally for a different demo unit. These instructions use a locally packed npm artifact, not a registry release.

## Configure the traffic split

This complete config opts 10% of the `shopping` population out of the `checkout` layer, uses 40% of that layer for `cart`, enrolls 30% of the selected units, and splits enrolled units 25:75. Rates are in basis points: 10,000 means 100%.

```json
{
  "protocol": 1,
  "revision": "checkout-launch",
  "experiments": [{
    "key": "cart",
    "enrollment_bps": 3000,
    "arms": [{"key": "control", "weight_bps": 2500},
             {"key": "treatment", "weight_bps": 7500}]
  }],
  "layers": [{
    "key": "checkout",
    "ranges": [{"experiment": "cart", "start_bps": 6000, "end_bps": 10000}]
  }],
  "holdouts": [{"key": "shopping", "held_out_bps": 1000, "targets": ["checkout"]}]
}
```

Request `checkout`, not `cart`. Each layer picks at most one of its experiments for a unit under a common configuration. A standalone experiment (not listed in any layer) can be requested by its own key. Holdouts cover only their explicit layer and standalone targets; different layers can overlap.

## Decide once, then deliver

In Python, import `compile_allocation` from `increment.assignment`. Compile the decoded config once and call `evaluator.decide("checkout", unit_id)`. In JavaScript, import `compileAllocation` from `@increment/assignment` and call `await evaluator.decide("checkout", unitId)`. Use the same identity string on every request. IDs are case-sensitive; `"007"` is not `"7"`.

| Decision | What to serve |
|---|---|
| `held_out` | Baseline. Do not serve any covered experiment; this is not a control-arm assignment. |
| `unallocated` | Baseline. No experiment covers the unit's slot in the layer. |
| `not_enrolled` | Baseline. An experiment was selected but the unit did not enter it. |
| `assigned` | Serve the returned arm, including the experiment's control behavior if assigned to control. |

Prefer deciding on the server or edge if a request already reaches it. Pass the full decision to the UI instead of recalculating it there. A browser-only app can evaluate locally over HTTPS or localhost using Web Crypto; it must not use a client-side decision to grant access or enforce an entitlement.

Persist the decision with your unit ID and timestamp. Record delivery where it actually happens, as a separate event. A variant can be assigned and never displayed. Use the conditional **arm** weights among enrolled units for the arm-ratio check, not the overall traffic share after multiplying by holdout, layer and enrollment rates.

## Measure the holdout at common entry

Before anyone encounters a scoped feature, define a population, stable ID, holdout key, scope, rate and observation window. At the same point of entry for both groups, call `evaluator.holdout_membership("shopping", unit_id)` in Python or `await evaluator.holdoutMembership("shopping", unitId)` in JavaScript. Save the first membership per study and unit for **both** `held_out` and `eligible`. A 0% or 100% holdout is a valid traffic setting, but cannot yield a two-group comparison.

Keep held-out units on baseline behavior across all covered surfaces. Compare everyone held out with everyone eligible, including eligible units who never enroll or see a variant. This intent-to-treat comparison measures access to the scoped experiment program, not the effect of treatment among units who saw it. Use the same outcome collection and window for both groups. A new scope, rate or population defines a new comparison; don't silently pool it with an old one.

## Ramp without moving existing members

Archive each revision. Increasing only enrollment preserves earlier members' arms. Adding a range in unused capacity leaves existing ranges in place. Changing arm weights, occupied ranges, keys or IDs can move units; in an active layer, use one consistent revision across surfaces. If you must preserve assignments through such changes, serve persisted assignments or end the affected experiments first.

To reproduce a decision, compile the archived SDK config and pass the original target and unit ID. The SDK uses a different protocol from the legacy `AssignmentSpec` ledger verifier. See the [v1 protocol and vectors](../reference/assignment-protocol-v1.md) and [replay boundary](../reference/artifacts-and-replay.md#assignment-evidence-and-replay).
