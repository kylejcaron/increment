import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import { FakeElement } from "./dom_stub.mjs";
import { loadDashboard } from "./dashboard_population_fixture.mjs";
const SHELL = new URL("../../increment/dashboard/_shell.html", import.meta.url);
const coreMetrics = [
  { key: "conversion", label: "Conversion", role: "primary", detail: "Conversion definition", csv: "conversion groups" },
  { key: "revenue", label: "Revenue", role: "secondary", detail: "Revenue definition", csv: "revenue groups" },
  { key: "guardrail", label: "Guardrail", role: "guardrail", detail: "Guardrail definition", csv: "guardrail groups" },
];
const addedMetric = {
  key: "experiment_tag",
  label: "Experiment tag",
  role: "exploratory",
  detail: "Exploratory definition",
  csv: "exploratory groups",
};
const allMetrics = [...coreMetrics, addedMetric];
const keys = ["cumulative_lift", "cumulative_lift_complete", "cumulative_values", "cumulative_values_complete", "daily_values"];

function packetSet(prefix) {
  return Object.fromEntries(keys.map((key) => [key, {
    html: `<table class="gt_table"><tbody><tr><th>${prefix} ${key}</th><td>Control</td><td>Treatment</td></tr></tbody></table>`,
    caption: `${prefix} ${key}`,
    notes: [`${prefix} note ${key}`],
    pointCount: 2,
  }]));
}

function view(prefix) {
  return {
    title: prefix,
    results: "",
    metrics: allMetrics.map((metric) => ({ ...metric })),
    scopes: {
      overall: { label: "Whole experiment", source: "assigned population" },
      country: { label: "Country", source: "assignment data" },
    },
    overview: {},
    explore: {
      overall: Object.fromEntries(allMetrics.map((metric) => [metric.key, packetSet(`${prefix} ${metric.key}`)])),
      country: Object.fromEntries(allMetrics.map((metric) => [metric.key, packetSet(`${prefix} ${metric.key} country`)])),
    },
    wholeExperiment: packetSet(prefix),
    exploratory: {
      available: [{ key: addedMetric.key, label: addedMetric.label }],
      selected: [addedMetric.key],
    },
  };
}

function makeDashboard({ includeWholeExperiment = true } = {}) {
  const assigned = view("Assigned");
  const triggered = view("Triggered");
  if (!includeWholeExperiment) {
    delete assigned.wholeExperiment;
    delete triggered.wholeExperiment;
  }
  const payload = {
    ...assigned,
    defaultPopulation: "assigned",
    populationOptions: ["assigned", "triggered"],
    populationViews: { assigned, triggered },
  };
  const html = fs.readFileSync(SHELL, "utf8").replace("<!-- DASHBOARD_DATA -->", () => JSON.stringify(payload));
  const dashboard = loadDashboard(html);
  dashboard.tabs.explore.dispatch("click");
  const series = new FakeElement("button");
  series.dataset.exploreView = "series";
  dashboard.panels.explore.dispatch("click", { target: series });
  return dashboard;
}

const byId = (dashboard, id) => dashboard.document.getElementById(id);
const clickData = (dashboard, key, value) => {
  const button = new FakeElement("button");
  button.dataset[key] = value;
  dashboard.panels.explore.dispatch("click", { target: button });
};
const caption = (dashboard) => byId(dashboard, "trajectory-caption").textContent;

test("whole experiment uses its five view packets and restores metric focus on breakouts", () => {
  const dashboard = makeDashboard();
  const layout = byId(dashboard, "explore-layout");
  const focusList = byId(dashboard, "metric-list");
  const inspector = byId(dashboard, "metric-definition");
  const download = byId(dashboard, "group-download");

  assert.equal(byId(dashboard, "explore-metric-name").textContent, "All experiment metrics");
  assert.equal(focusList.hidden, true);
  assert.equal(inspector.hidden, false);
  assert.equal(download.hidden, false);
  assert.ok(dashboard.objectUrls.has(download.href));
  assert.equal(byId(dashboard, "explore-inspector").innerHTML, "Conversion definition");
  assert.match(caption(dashboard), /Assigned cumulative_lift$/);

  byId(dashboard, "complete-only").checked = true;
  byId(dashboard, "complete-only").dispatch("change", { target: byId(dashboard, "complete-only") });
  assert.match(caption(dashboard), /Assigned cumulative_lift_complete$/);

  clickData(dashboard, "mode", "absolute");
  assert.match(caption(dashboard), /Assigned cumulative_values_complete$/);
  clickData(dashboard, "cadence", "daily");
  assert.match(caption(dashboard), /Assigned daily_values$/);
  assert.equal(byId(dashboard, "complete-toggle").hidden, true);
  assert.equal(byId(dashboard, "explore-context").textContent.includes("daily per-arm absolute values"), true);

  clickData(dashboard, "scope", "country");
  assert.equal(focusList.hidden, false);
  assert.equal(inspector.hidden, false);
  assert.equal(download.hidden, false);
  assert.equal(byId(dashboard, "explore-metric-name").textContent, "Conversion");
  assert.match(caption(dashboard), /^Assigned conversion country daily_values$/);
  assert.equal(byId(dashboard, "explore-inspector").innerHTML, "Conversion definition");

  clickData(dashboard, "scope", "overall");
  assert.equal(focusList.hidden, true);
  assert.equal(inspector.hidden, false);
  assert.equal(download.hidden, false);
  assert.ok(dashboard.objectUrls.has(download.href));
  assert.equal(byId(dashboard, "explore-inspector").innerHTML, "Conversion definition");
  assert.match(caption(dashboard), /Assigned daily_values$/);

  dashboard.populationSelect.value = "triggered";
  dashboard.populationSelect.dispatch("change");
  assert.match(caption(dashboard), /Triggered daily_values$/);
  assert.equal(byId(dashboard, "explore-metric-name").textContent, "All experiment metrics");
  assert.equal(layout.hidden, false);
});

test("missing whole-experiment packets stay unavailable instead of showing the focused metric", () => {
  const dashboard = makeDashboard({ includeWholeExperiment: false });
  assert.equal(byId(dashboard, "explore-metric-name").textContent, "All experiment metrics");
  assert.equal(byId(dashboard, "metric-list").hidden, true);
  assert.equal(byId(dashboard, "metric-definition").hidden, false);
  assert.equal(byId(dashboard, "group-download").hidden, false);
  assert.match(caption(dashboard), /Whole-experiment time series were not captured/);
});
