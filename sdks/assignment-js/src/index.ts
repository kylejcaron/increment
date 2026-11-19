import { compileConfig } from './config.js';
import { createEvaluator } from './evaluate.js';
import type { AllocationEvaluator } from './types.js';

export type { AllocationEvaluator, Decision, Membership } from './types.js';

export function compileAllocation(config: unknown): AllocationEvaluator {
  return createEvaluator(compileConfig(config));
}
