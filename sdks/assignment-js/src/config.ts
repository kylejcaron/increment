import { BUCKET_COUNT, fitsHashField, utf8 } from './hash.js';

interface Arm {
  key: string;
  weightBps: number;
  endBps: number;
}

interface Experiment {
  key: string;
  keyBytes: Uint8Array;
  enrollmentBps: number;
  arms: readonly Arm[];
}

interface LayerRange {
  experiment: Experiment;
  startBps: number;
  endBps: number;
}

export interface Holdout {
  key: string;
  keyBytes: Uint8Array;
  heldOutBps: number;
}

type Target = { holdout?: Holdout } & (
  | { kind: 'layer'; keyBytes: Uint8Array; ranges: readonly LayerRange[] }
  | { kind: 'experiment'; experiment: Experiment }
);

export interface CompiledConfig {
  revision: string;
  targets: ReadonlyMap<string, Target>;
  holdouts: ReadonlyMap<string, Holdout>;
}

const INVALID_CONFIG = 'allocation.invalid_config';
const INVALID_UNIT = 'allocation.invalid_unit';
type StringError = typeof INVALID_CONFIG | typeof INVALID_UNIT;

export function refuse(code: StringError | 'allocation.invalid_target', message: string): never {
  throw Object.assign(new Error(message), { code });
}

function record(value: unknown, label: string, fields: readonly string[], optional?: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    refuse(INVALID_CONFIG, `${label} must be an object.`);
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || (key !== optional && !fields.includes(key))) {
      refuse(INVALID_CONFIG, `${label} has an unknown field.`);
    }
  }
  for (const key of fields) {
    if (!Object.hasOwn(value, key)) refuse(INVALID_CONFIG, `${label} is missing ${key}.`);
  }
  return value as Record<string, unknown>;
}

function list(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) refuse(INVALID_CONFIG, `${label} must be an array.`);
  return value;
}

function name(value: unknown, label: string, code: StringError = INVALID_CONFIG): string {
  // In Unicode mode, paired surrogates are one code point outside this range.
  if (typeof value !== 'string' || value.length === 0 || /[\uD800-\uDFFF]/u.test(value)) {
    refuse(code, `${label} must be a nonempty Unicode scalar string.`);
  }
  return value;
}

function keyBytes(key: string, label: string, code: StringError = INVALID_CONFIG): Uint8Array {
  if (!fitsHashField(key)) refuse(code, `${label} exceeds the protocol's UTF-8 length limit.`);
  return utf8(key);
}

export function unitBytes(value: unknown): Uint8Array {
  return keyBytes(name(value, 'Unit ID', INVALID_UNIT), 'Unit ID', INVALID_UNIT);
}

function basisPoints(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > BUCKET_COUNT) {
    refuse(INVALID_CONFIG, `${label} must be an integer from 0 to ${BUCKET_COUNT}.`);
  }
  return value === 0 ? 0 : value;
}

function compareBytes(left: Uint8Array, right: Uint8Array): number {
  const common = Math.min(left.length, right.length);
  for (let i = 0; i < common; i++) {
    if (left[i] !== right[i]) return left[i] - right[i];
  }
  return left.length - right.length;
}

function compileArms(value: unknown): Arm[] {
  const input = list(value, 'Arms');
  if (input.length < 2) refuse(INVALID_CONFIG, 'An experiment must have at least two arms.');
  const names = new Set<string>();
  const arms = [];
  let total = 0;
  for (const entry of input) {
    const arm = record(entry, 'Arm', ['key', 'weight_bps']);
    const key = name(arm.key, 'Arm key');
    if (names.has(key)) refuse(INVALID_CONFIG, 'Arm keys must be distinct within an experiment.');
    const weightBps = basisPoints(arm.weight_bps, 'weight_bps');
    if (weightBps === 0) refuse(INVALID_CONFIG, 'Arm weights must be positive.');
    names.add(key);
    total += weightBps;
    arms.push({ key, bytes: utf8(key), weightBps });
  }
  if (total !== BUCKET_COUNT) refuse(INVALID_CONFIG, 'Arm weights must sum to 10000.');
  arms.sort((a, b) => compareBytes(a.bytes, b.bytes));
  let endBps = 0;
  return arms.map(({ key, weightBps }) => {
    endBps += weightBps;
    return { key, weightBps, endBps };
  });
}

export function compileConfig(config: unknown): CompiledConfig {
  const root = record(config, 'Configuration', ['protocol', 'revision', 'experiments', 'layers'], 'holdouts');
  if (root.protocol !== 1) refuse(INVALID_CONFIG, 'Configuration protocol must be 1.');
  const revision = name(root.revision, 'Revision');
  const experiments = new Map<string, Experiment>();
  for (const entry of list(root.experiments, 'Experiments')) {
    const experiment = record(entry, 'Experiment', ['key', 'enrollment_bps', 'arms']);
    const key = name(experiment.key, 'Experiment key');
    if (experiments.has(key)) refuse(INVALID_CONFIG, 'Experiment keys must be distinct.');
    experiments.set(key, {
      key,
      keyBytes: keyBytes(key, 'Experiment key'),
      enrollmentBps: basisPoints(experiment.enrollment_bps, 'enrollment_bps'),
      arms: compileArms(experiment.arms),
    });
  }

  const targets = new Map<string, Target>();
  const layered = new Set<string>();
  for (const entry of list(root.layers, 'Layers')) {
    const layer = record(entry, 'Layer', ['key', 'ranges']);
    const key = name(layer.key, 'Layer key');
    if (targets.has(key) || experiments.has(key)) {
      refuse(INVALID_CONFIG, 'Layer keys must be distinct from other layer and experiment keys.');
    }
    const ranges: LayerRange[] = [];
    for (const item of list(layer.ranges, 'Layer ranges')) {
      const range = record(item, 'Range', ['experiment', 'start_bps', 'end_bps']);
      const experimentKey = name(range.experiment, 'Range experiment key');
      const experiment = experiments.get(experimentKey);
      if (!experiment) refuse(INVALID_CONFIG, 'A range must refer to an existing experiment.');
      if (layered.has(experimentKey)) refuse(INVALID_CONFIG, 'An experiment can occur in only one range.');
      const startBps = basisPoints(range.start_bps, 'start_bps');
      const endBps = basisPoints(range.end_bps, 'end_bps');
      if (startBps >= endBps) refuse(INVALID_CONFIG, 'A range start must precede its end.');
      layered.add(experimentKey);
      ranges.push({ experiment, startBps, endBps });
    }
    ranges.sort((a, b) => a.startBps - b.startBps);
    for (let i = 1; i < ranges.length; i++) {
      if (ranges[i - 1].endBps > ranges[i].startBps) {
        refuse(INVALID_CONFIG, 'Ranges in a layer must not overlap.');
      }
    }
    targets.set(key, { kind: 'layer', keyBytes: keyBytes(key, 'Layer key'), ranges });
  }
  for (const [key, experiment] of experiments) {
    if (!layered.has(key)) targets.set(key, { kind: 'experiment', experiment });
  }

  const holdouts = new Map<string, Holdout>();
  if (Object.hasOwn(root, 'holdouts')) {
    for (const entry of list(root.holdouts, 'Holdouts')) {
      const item = record(entry, 'Holdout', ['key', 'held_out_bps', 'targets']);
      const key = name(item.key, 'Holdout key');
      if (targets.has(key) || holdouts.has(key)) {
        refuse(INVALID_CONFIG, 'Holdout keys must be distinct from other holdout and decision target keys.');
      }
      const holdout = {
        key,
        keyBytes: keyBytes(key, 'Holdout key'),
        heldOutBps: basisPoints(item.held_out_bps, 'held_out_bps'),
      };
      const scope = list(item.targets, 'Holdout targets');
      if (scope.length === 0) refuse(INVALID_CONFIG, 'A holdout must name at least one target.');
      for (const entry of scope) {
        const target = targets.get(name(entry, 'Holdout target key'));
        if (!target) refuse(INVALID_CONFIG, 'A holdout target must be a layer or standalone experiment.');
        if (target.holdout) refuse(INVALID_CONFIG, 'A target can occur only once across all holdout scopes.');
        target.holdout = holdout;
      }
      holdouts.set(key, holdout);
    }
  }
  return { revision, targets, holdouts };
}
