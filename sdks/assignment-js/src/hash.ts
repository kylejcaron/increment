export const BUCKET_COUNT = 10_000;
const MAX_UINT32 = 0xffff_ffff;
const ACCEPT_BELOW = 4_294_960_000;
const encoder = new TextEncoder();
const prefix = encoder.encode('increment.alloc.v1\0');
const stages = {
  holdout: encoder.encode('holdout'),
  layer: encoder.encode('layer'),
  enroll: encoder.encode('enroll'),
  arm: encoder.encode('arm'),
};

export function utf8(value: string): Uint8Array {
  return encoder.encode(value);
}

export function fitsHashField(value: string): boolean {
  // A UTF-16 code unit contributes at most three UTF-8 bytes.
  if (value.length <= Math.floor(MAX_UINT32 / 3)) return true;
  let length = 0;
  for (const character of value) {
    const code = character.codePointAt(0)!;
    length += code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
    if (length > MAX_UINT32) return false;
  }
  return true;
}

function writeField(frame: Uint8Array, view: DataView, offset: number, field: Uint8Array): number {
  view.setUint32(offset, field.length, false);
  frame.set(field, offset + 4);
  return offset + 4 + field.length;
}

export async function bucket(
  stage: keyof typeof stages,
  key: Uint8Array,
  unit: Uint8Array,
): Promise<number> {
  const stageBytes = stages[stage];
  // The frame has three length words and one attempt word.
  const frame = new Uint8Array(prefix.length + 4 * 4 + stageBytes.length + key.length + unit.length);
  const view = new DataView(frame.buffer);
  frame.set(prefix);
  const keyOffset = writeField(frame, view, prefix.length, stageBytes);
  const unitOffset = writeField(frame, view, keyOffset, key);
  const attemptOffset = writeField(frame, view, unitOffset, unit);

  // Rehash only when the first word falls in the incomplete final group of buckets.
  for (let attempt = 0; attempt <= MAX_UINT32; attempt++) {
    view.setUint32(attemptOffset, attempt, false);
    const digest = await globalThis.crypto.subtle.digest('SHA-256', frame);
    const word = new DataView(digest).getUint32(0, false);
    if (word < ACCEPT_BELOW) return word % BUCKET_COUNT;
  }
  throw new Error('Assignment hash attempts exhausted.');
}
