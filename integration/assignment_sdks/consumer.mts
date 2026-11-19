import { compileAllocation } from '@increment/assignment';
import type { AllocationEvaluator, Decision, Membership } from '@increment/assignment';

type IsAny<T> = 0 extends (1 & T) ? true : false;
export const factoryIsTyped: IsAny<typeof compileAllocation> = false;
export const evaluatorIsTyped: IsAny<AllocationEvaluator> = false;
export const decisionIsTyped: IsAny<Decision> = false;
export const membershipIsTyped: IsAny<Membership> = false;
export const revisionIsTyped: IsAny<Decision['revision']> = false;
export const groupIsTyped: IsAny<Membership['group']> = false;

export async function decide(config: unknown, unitId: string): Promise<[Decision, Membership]> {
  const evaluator: AllocationEvaluator = compileAllocation(config);
  return Promise.all([
    evaluator.decide('checkout', unitId),
    evaluator.holdoutMembership('shopping', unitId),
  ]);
}
