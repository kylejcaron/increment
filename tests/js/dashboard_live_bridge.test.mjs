import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../increment/dashboard/_bridge.js", import.meta.url), "utf8");
const { default: bridge } = await import(`data:text/javascript,${encodeURIComponent(source)}`);

test("only the embedded frame can change metrics when its window proxy is retargeted", () => {
  const replies = [];
  const changes = [];
  const frame = { contentWindow: { postMessage: () => { throw new Error("wrong retargeted endpoint"); } } };
  let onMessage;
  const previousWindow = globalThis.window;
  const previousDocument = globalThis.document;
  globalThis.window = {
    addEventListener: (type, handler) => { if (type === "message") onMessage = handler; },
    removeEventListener() {},
  };
  globalThis.document = { createElement: () => frame };
  const model = {
    get: () => "captured page",
    on() {},
    set: (key, value) => changes.push([key, value]),
    save_changes() {},
  };
  try {
    const cleanup = bridge.render({ model, el: { appendChild() {} } });
    const request = { source: "inc-dashboard", type: "set-exploratory-metrics", metrics: ["retention"] };
    onMessage({ source: { frameElement: {} }, data: request });
    onMessage({ source: { get frameElement() { throw new Error("cross-origin"); } }, data: request });
    assert.deepEqual(changes, []);
    const sender = { frameElement: frame, postMessage: message => replies.push(message) };
    onMessage({ source: sender, data: { source: "inc-dashboard", type: "ready" } });
    assert.equal(replies[0]?.type, "bridge");
    onMessage({ source: sender, data: request });
    assert.deepEqual(changes, [["exploratory_metrics", ["retention"]]]);
    cleanup();
  } finally {
    globalThis.window = previousWindow;
    globalThis.document = previousDocument;
  }
});
