export interface Membership {
  protocol: 1;
  revision: string;
  holdout: string;
  group: 'held_out' | 'eligible';
  group_bps: number;
}

type DecisionBase = {
  protocol: 1;
  revision: string;
  target: string;
  membership: Membership | null;
};

type SelectedExperiment = {
  experiment: string;
  layer_bps: number;
  enrollment_bps: number;
};

export type Decision = DecisionBase & (
  | {
      status: 'held_out' | 'unallocated';
      experiment: null;
      layer_bps: null;
      enrollment_bps: null;
      arm: null;
      arm_bps: null;
    }
  | (SelectedExperiment & { status: 'not_enrolled'; arm: null; arm_bps: null })
  | (SelectedExperiment & { status: 'assigned'; arm: string; arm_bps: number })
);

export interface AllocationEvaluator {
  decide(target: string, unitId: string): Promise<Decision>;
  holdoutMembership(holdoutKey: string, unitId: string): Promise<Membership>;
}
