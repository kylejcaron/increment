import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { compileAllocation } from '@increment/assignment';

const { configs, cases } = JSON.parse(readFileSync(
  new URL('../../../docs/reference/assignment-v1-vectors.json', import.meta.url),
  'utf8',
));
const byId = new Map(cases.map((vector) => [vector.id, vector]));

function decide(evaluator, vector) {
  return evaluator.decide(vector.target, vector.unit_id);
}

const originalCheckout = byId.get('range-start-inclusive');
const originalSearch = byId.get('eligible-miss-other-layer');
const editedCheckout = byId.get('fresh-compile-of-edited-program');
const editedSearch = byId.get('fresh-scope-removes-membership');
const snapshotCases = new Set([
  originalCheckout.id, originalSearch.id, editedCheckout.id, editedSearch.id,
]);

for (const vector of cases) {
  if (snapshotCases.has(vector.id)) continue;
  test(vector.id, async () => {
    const config = configs[vector.config];
    if (vector.op === 'compile') {
      assert.throws(() => compileAllocation(config), { code: vector.error });
      return;
    }
    const evaluator = compileAllocation(config);
    const result = vector.op === 'decide'
      ? decide(evaluator, vector)
      : evaluator.holdoutMembership(vector.target, vector.unit_id);
    if (vector.error) {
      await assert.rejects(result, (error) => {
        assert.equal(error.code, vector.error);
        if (typeof vector.unit_id === 'string' && vector.unit_id.length > 0) {
          const visibleError = JSON.stringify({ ...error, message: error.message });
          const escapedUnit = JSON.stringify(vector.unit_id).slice(1, -1);
          assert.ok(!visibleError.includes(escapedUnit));
          assert.ok(!visibleError.includes('pii-'));
        } else if (typeof vector.unit_id === 'number') {
          assert.ok(!error.message.includes(String(vector.unit_id)));
        }
        return true;
      });
    } else {
      assert.deepEqual(await result, vector.expected);
    }
  });
}

test('compileAllocation snapshots nested edits even while a decision is in flight', async () => {
  const config = structuredClone(configs.program);
  const edited = configs['mutated-program'];
  const evaluator = compileAllocation(config);
  assert.deepEqual(await decide(evaluator, originalCheckout), originalCheckout.expected);

  const pending = decide(evaluator, originalCheckout);
  config.revision = edited.revision;
  config.experiments[0].arms.forEach((arm, i) => {
    arm.weight_bps = edited.experiments[0].arms[i].weight_bps;
  });
  config.layers[0].ranges[0].start_bps = edited.layers[0].ranges[0].start_bps;
  config.holdouts[0].held_out_bps = edited.holdouts[0].held_out_bps;
  config.holdouts[0].targets.splice(0, Infinity, ...edited.holdouts[0].targets);

  assert.deepEqual(await pending, originalCheckout.expected);
  assert.deepEqual(await decide(evaluator, originalCheckout), originalCheckout.expected);
  assert.deepEqual(await decide(evaluator, originalSearch), originalSearch.expected);

  const fresh = compileAllocation(config);
  assert.deepEqual(await decide(fresh, editedCheckout), editedCheckout.expected);
  assert.deepEqual(await decide(fresh, editedSearch), editedSearch.expected);
});

test('editing a returned decision and membership cannot change later results', async () => {
  const evaluator = compileAllocation(configs.program);
  const result = await decide(evaluator, originalCheckout);
  result.revision = 'edited decision';
  result.arm_bps = -1;
  result.membership.group_bps = -1;
  assert.deepEqual(await decide(evaluator, originalCheckout), originalCheckout.expected);

  const { target, unit_id, expected } = byId.get('entry-eligible');
  const membership = await evaluator.holdoutMembership(target, unit_id);
  membership.group = 'held_out';
  assert.deepEqual(await evaluator.holdoutMembership(target, unit_id), expected);
});

test('nonfinite decoded basis points are invalid, not a full enrollment rate', () => {
  const config = structuredClone(configs.full);
  config.experiments[0].enrollment_bps = JSON.parse('1e400');
  assert.throws(() => compileAllocation(config), { code: 'allocation.invalid_config' });
});
