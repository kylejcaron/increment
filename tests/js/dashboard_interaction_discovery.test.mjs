import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { loadDashboard } from "./dashboard_population_fixture.mjs";

const template = readFileSync(new URL("../../increment/dashboard/_shell.html", import.meta.url), "utf8");
const bridgeSource = readFileSync(new URL("../../increment/dashboard/_bridge.js", import.meta.url), "utf8");
const { default: bridge } = await import(`data:text/javascript,${encodeURIComponent(bridgeSource)}`);
const matchingDetail = "Preview cards: −5 pp [−7.76, −2.24]";
const relativeDetail = "Preview cards: −16.3 points of relative lift";
const unavailableDetail = "Search tips: nobody was assigned both on";

function page(configure) {
  const assigned = {
    title: "Search layout",
    metrics: [{ key: "save", label: "Save rate", role: "primary" }],
    primary: { label: "Save rate", effect: "+0.5 pp" },
    interactions: {
      status: "",
      metricOptions: [{ key: "save", label: "Save rate" }],
      alert: { metric: "save", pair: "cards-v1" },
      metrics: {
        save: {
          status: "Three concurrent experiments",
          explanation: "18 planned comparisons",
          overview: "Preview cards and other experiments",
          defaultScale: "relative",
          relative: {
            overview: "Change in relative lift across experiments",
            details: { "cards-v1": relativeDetail, "tips-v2": unavailableDetail },
          },
          pairs: [
            { id: "cards-v1", label: "Preview cards" },
            { id: "tips-v2", label: "Search tips" },
          ],
          details: { "cards-v1": matchingDetail, "tips-v2": unavailableDetail },
        },
      },
      reversals: [
        { metric: "save", pair: "cards-v1", text: "On jointly assigned accounts, the effect reversed with Preview cards." },
      ],
      health: [],
    },
  };
  const payload = {
    ...assigned,
    defaultPopulation: "assigned",
    populationOptions: ["assigned", "triggered"],
    populationViews: {
      triggered: {
        ...assigned,
        primary: { label: "Save rate", effect: "+0.7 pp" },
        interactions: { status: "Not checked for the triggered population." },
      },
    },
  };
  return loadDashboard(template.replace("<!-- DASHBOARD_DATA -->", JSON.stringify(payload)), configure);
}

test("rollout context opens exact pair within Explore without replacing the headline", () => {
  const dashboard = page();
  const get = (id) => dashboard.document.getElementById(id);
  dashboard.tabs.explore.dispatch("click");
  dashboard.tabs.readout.dispatch("click");
  assert.equal(dashboard.heroEffect.textContent, "+0.5 pp");
  assert.match(get("rollout-context")?.textContent ?? "", /effect reversed with Preview cards/);
  const button = get("rollout-context").querySelectorAll("button")[0];
  button.dispatch("click");
  assert.equal(dashboard.tabs.explore.getAttribute("aria-selected"), "true");
  assert.equal(get("explore-interactions").hidden, false);
  assert.equal(get("interaction-pair").value, "cards-v1");
  assert.equal(get("interaction-scale").value, "relative");
  assert.equal(get("interaction-overview").innerHTML, "Change in relative lift across experiments");
  assert.equal(get("interaction-detail-body").innerHTML, relativeDetail);
  const scale = get("interaction-scale");
  scale.value = "absolute";
  scale.dispatch("change");
  assert.equal(get("interaction-pair").value, "cards-v1");
  assert.equal(get("interaction-overview").innerHTML, "Preview cards and other experiments");
  assert.equal(get("interaction-detail-body").innerHTML, matchingDetail);
  assert.equal(dashboard.heroEffect.textContent, "+0.5 pp");
});

test("pair selection preserves the full family and switching population hides assigned evidence", () => {
  const dashboard = page();
  const get = (id) => dashboard.document.getElementById(id);
  get("rollout-context")?.querySelectorAll("button")[0].dispatch("click");
  assert.equal(get("interaction-explanation")?.textContent, "18 planned comparisons");
  const choice = get("interaction-pair");
  choice.value = "tips-v2";
  choice.dispatch("change");
  assert.equal(get("interaction-detail-body").innerHTML, unavailableDetail);
  assert.equal(get("interaction-explanation").textContent, "18 planned comparisons");
  dashboard.populationSelect.value = "triggered";
  dashboard.populationSelect.dispatch("change");
  assert.equal(get("interaction-status").textContent, "Not checked for the triggered population.");
  assert.equal(get("interaction-detail-body").innerHTML, "");
  assert.equal(dashboard.heroEffect.textContent, "+0.7 pp");
});

test("header alert opens the exact absolute comparison and clears for the triggered view", () => {
  const dashboard = page();
  const get = (id) => dashboard.document.getElementById(id);
  const badge = get("hero-interaction");
  assert.equal(badge?.hidden, false);
  badge.dispatch("click", { preventDefault() {} });
  assert.equal(dashboard.tabs.explore.getAttribute("aria-selected"), "true");
  assert.equal(get("interaction-scale").value, "absolute");
  assert.equal(get("interaction-detail-body").innerHTML, matchingDetail);
  assert.equal(dashboard.document.activeElement?.id, "interaction-detail-heading");
  dashboard.populationSelect.value = "triggered";
  dashboard.populationSelect.dispatch("change");
  assert.equal(badge.hidden, true);
});

const retentionDetail = "Search tips: retention alongside concurrent assignment";

function refreshInteractionPage(changeScan = () => {}) {
  const configure = (payload) => {
    payload.metrics.push({ key: "retention", label: "Retention", role: "secondary" });
    payload.exploratory = { available: [{ key: "pace", label: "Pace" }], selected: [] };
    const scan = payload.interactions;
    scan.metricOptions.push({ key: "retention", label: "Retention" });
    scan.metrics.retention = {
      ...scan.metrics.save,
      details: { ...scan.metrics.save.details, "tips-v2": retentionDetail },
    };
  };
  const before = page(configure);
  const after = page((payload) => { configure(payload); changeScan(payload.interactions); });
  const frame = {};
  const handlers = {};
  const model = {
    get: (key) => ({ document: "updated document", status: "updating" })[key],
    on: (type, handler) => { handlers[type] = handler; },
    set(_key, _value) {},
    save_changes() {
      handlers["change:status"]();
      handlers["change:document"]();
    },
  };
  const { window, document } = globalThis;
  let receive;
  globalThis.window = {
    addEventListener: (type, handler) => { if (type === "message") receive = handler; },
    removeEventListener() {},
  };
  globalThis.document = { createElement: () => frame };
  try {
    const cleanup = bridge.render({ model, el: { appendChild() {} } });
    const connect = (dashboard) => {
      const peer = { postMessage: (data) => dashboard.dispatchHostMessage(data) };
      frame.contentWindow = peer;
      return () => {
        for (const data of dashboard.hostMessages.splice(0)) receive({ source: peer, data });
      };
    };
    const flush = connect(before);
    flush();
    const get = (id) => before.document.getElementById(id);
    get("rollout-context").querySelector("button").dispatch("click");
    const metric = get("interaction-metric");
    metric.value = "retention";
    metric.dispatch("change");
    const pair = get("interaction-pair");
    pair.value = "tips-v2";
    pair.dispatch("change");
    const scale = get("interaction-scale");
    scale.value = "absolute";
    scale.dispatch("change");
    const heading = get("interaction-detail-heading");
    heading.focus();
    get("explore-interactions").dispatch("focusin", { target: heading });
    get("added-panel").hidden = true;
    get("added-open").dispatch("click");
    const option = get("added-options").querySelector("input");
    option.checked = true;
    option.dispatch("change");
    get("added-apply").dispatch("click");
    flush();
    connect(after)();
    cleanup();
  } finally {
    globalThis.window = window;
    globalThis.document = document;
  }
  return after;
}

test("live metric refresh restores the interaction view, selected metric and pair, and detail focus", () => {
  const dashboard = refreshInteractionPage();
  const get = (id) => dashboard.document.getElementById(id);
  assert.equal(dashboard.tabs.explore.getAttribute("aria-selected"), "true");
  assert.equal(get("explore-interactions").hidden, false);
  assert.equal(get("interaction-metric").value, "retention");
  assert.equal(get("interaction-pair").value, "tips-v2");
  assert.equal(get("interaction-scale").value, "absolute");
  assert.equal(get("interaction-detail-body").innerHTML, retentionDetail);
  assert.equal(dashboard.document.activeElement?.id, "interaction-detail-heading");
});

test("live refresh clears a missing metric's pair even when that pair exists for the fallback metric", () => {
  const dashboard = refreshInteractionPage((scan) => {
    scan.metricOptions.pop();
    delete scan.metrics.retention;
  });
  const get = (id) => dashboard.document.getElementById(id);
  assert.equal(get("interaction-metric").value, "save");
  assert.equal(get("interaction-pair").value, "");
  assert.equal(get("interaction-detail").hidden, true);
  assert.notEqual(dashboard.document.activeElement?.id, "interaction-detail-heading");
});

test("live refresh clears a missing pair without clearing its still-eligible metric", () => {
  const dashboard = refreshInteractionPage((scan) => {
    scan.metrics.retention.pairs = [{ id: "cards-v1", label: "Preview cards" }];
    delete scan.metrics.retention.details["tips-v2"];
  });
  const get = (id) => dashboard.document.getElementById(id);
  assert.equal(get("interaction-metric").value, "retention");
  assert.equal(get("interaction-pair").value, "");
  assert.equal(get("interaction-detail").hidden, true);
  assert.notEqual(dashboard.document.activeElement?.id, "interaction-detail-heading");
});
