import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const context = vm.createContext({});
vm.runInContext(readFileSync(new URL("../../increment/dashboard/_native.js", import.meta.url), "utf8"), context);

test("every printed band retains metric and arm despite a full-width caption", () => {
  const head = {};
  const cell = (text, width, span = 1) => ({
    colSpan: span,
    rowSpan: 1,
    textContent: text,
    getBoundingClientRect: () => ({ width }),
    querySelector: () => null,
    hasAttribute: () => false,
  });
  const table = {
    tHead: head,
    rows: [
      { parentElement: head, cells: [cell("Readout evidence", 580, 4)] },
      { parentElement: head, cells: [cell("Metric", 100), cell("Arm", 80), cell("Estimate", 200), cell("Result", 200)] },
    ],
  };
  const host = { querySelector: () => table };
  const adapter = context.createNativeAdapter({});
  const bands = JSON.parse(JSON.stringify(adapter.planBands(host, 360)));
  assert.deepEqual(bands, [[0, 1, 2], [0, 1, 3]]);
});
