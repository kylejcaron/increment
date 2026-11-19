import { refuse, unitBytes } from './config.js';
import type { CompiledConfig, Holdout } from './config.js';
import { BUCKET_COUNT, bucket } from './hash.js';
import type { AllocationEvaluator, Decision, Membership } from './types.js';

function lookup<T>(entries: ReadonlyMap<string, T>, key: unknown, message: string): T {
  const entry = typeof key === 'string' ? entries.get(key) : undefined;
  if (entry === undefined) refuse('allocation.invalid_target', message);
  return entry;
}

function upperBound<T extends { endBps: number }>(items: readonly T[], draw: number): T | undefined {
  let left = 0;
  let right = items.length;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (items[middle].endBps <= draw) left = middle + 1;
    else right = middle;
  }
  return items[left];
}

async function membershipFor(revision: string, holdout: Holdout, unit: Uint8Array): Promise<Membership> {
  const heldOut = await bucket('holdout', holdout.keyBytes, unit) < holdout.heldOutBps;
  return {
    protocol: 1,
    revision,
    holdout: holdout.key,
    group: heldOut ? 'held_out' : 'eligible',
    group_bps: heldOut ? holdout.heldOutBps : BUCKET_COUNT - holdout.heldOutBps,
  };
}

function emptyDecision(
  revision: string,
  target: string,
  membership: Membership | null,
  status: 'held_out' | 'unallocated',
): Decision {
  return {
    protocol: 1, revision, target, membership, status,
    experiment: null, arm: null, layer_bps: null, enrollment_bps: null, arm_bps: null,
  };
}

export function createEvaluator({ revision, targets, holdouts }: CompiledConfig): AllocationEvaluator {
  return {
    async decide(target, unitId) {
      const selectedTarget = lookup(targets, target, 'Decision target must be an existing layer or standalone experiment.');
      const unit = unitBytes(unitId);
      const membership = selectedTarget.holdout
        ? await membershipFor(revision, selectedTarget.holdout, unit)
        : null;
      if (membership?.group === 'held_out') return emptyDecision(revision, target, membership, 'held_out');

      let experiment;
      let layerBps;
      if (selectedTarget.kind === 'layer') {
        const draw = await bucket('layer', selectedTarget.keyBytes, unit);
        const range = upperBound(selectedTarget.ranges, draw);
        if (!range || draw < range.startBps) return emptyDecision(revision, target, membership, 'unallocated');
        experiment = range.experiment;
        layerBps = range.endBps - range.startBps;
      } else {
        experiment = selectedTarget.experiment;
        layerBps = BUCKET_COUNT;
      }
      const { key, keyBytes, enrollmentBps } = experiment;
      if (await bucket('enroll', keyBytes, unit) >= enrollmentBps) {
        return {
          protocol: 1, revision, target, membership, status: 'not_enrolled',
          experiment: key, arm: null, layer_bps: layerBps, enrollment_bps: enrollmentBps, arm_bps: null,
        };
      }
      const arm = upperBound(experiment.arms, await bucket('arm', keyBytes, unit));
      if (!arm) throw new Error('Assignment arm partition does not cover the bucket.');
      return {
        protocol: 1, revision, target, membership, status: 'assigned',
        experiment: key, arm: arm.key, layer_bps: layerBps, enrollment_bps: enrollmentBps, arm_bps: arm.weightBps,
      };
    },

    async holdoutMembership(holdoutKey, unitId) {
      const holdout = lookup(holdouts, holdoutKey, 'Holdout key must name an existing holdout.');
      return membershipFor(revision, holdout, unitBytes(unitId));
    },
  };
}
