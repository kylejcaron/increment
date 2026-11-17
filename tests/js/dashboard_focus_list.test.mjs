import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import { FakeElement } from "./dom_stub.mjs";
import { loadDashboard } from "./dashboard_population_fixture.mjs";

const SHELL = new URL("../../increment/dashboard/_shell.html", import.meta.url);
const TOPICS = ["revenue", "latency", "retention"];
const ROLES = [
  ...Array(1).fill("primary"),
  ...Array(12).fill("secondary"),
  ...Array(7).fill("guardrail"),
  ...Array(10).fill("exploratory"),
];

/** The packaged shell with a one-population payload of `count` metrics, in declared order. */
function dashboardWith(count, references = {}) {
  const metrics = ROLES.slice(0, count).map((role, index) => {
    const key = `${TOPICS[index % TOPICS.length]}_${String(index + 1).padStart(2, "0")}`;
    return { key, label: key.replace("_", " "), role, detail: "", csv: "" };
  });
  const view = {
    title: "Many metrics",
    results: "",
    metrics,
    references,
    scopes: {
      overall: { label: "Whole experiment", dimension: null, source: null },
      country: { label: "Country", dimension: "country", source: null },
    },
    overview: {},
    explore: {
      overall: Object.fromEntries(
        metrics.map((metric) => [metric.key, { cumulative_lift: { html: "", caption: "", notes: [], pointCount: 1 } }])
      ),
      country: Object.fromEntries(
        metrics.map((metric) => [metric.key, { cumulative_lift: { html: "", caption: "", notes: [], pointCount: 1 } }])
      ),
    },
  };
  const payload = { ...view, defaultPopulation: "assigned", populationOptions: ["assigned"], populationViews: { assigned: view } };
  const html = fs.readFileSync(SHELL, "utf8").replace("<!-- DASHBOARD_DATA -->", () => JSON.stringify(payload));
  const dashboard = loadDashboard(html);
  dashboard.tabs.explore.dispatch("click");
  const series = new FakeElement("button");
  series.dataset.exploreView = "series";
  dashboard.panels.explore.dispatch("click", { target: series });
  const country = new FakeElement("button");
  country.dataset.scope = "country";
  dashboard.panels.explore.dispatch("click", { target: country });
  return dashboard;
}

const byId = (dashboard, id) => dashboard.document.getElementById(id);
const tabs = (dashboard) => byId(dashboard, "metric-options").querySelectorAll(".metric-tab");
const shown = (dashboard) => tabs(dashboard).filter((tab) => !tab.hidden).map((tab) => tab.dataset.metric);
const selected = (dashboard) => tabs(dashboard).filter((tab) => tab.getAttribute("aria-pressed") === "true").map((tab) => tab.dataset.metric);

function filter(dashboard, text) {
  const input = byId(dashboard, "metric-filter");
  input.value = text;
  input.dispatch("input");
}

test("thirty focus metrics keep their role groups and selection under a text filter", () => {
  const dashboard = dashboardWith(30);
  const groups = byId(dashboard, "metric-options").children;
  assert.deepEqual(
    groups.map((group) => group.querySelector(".metric-group-label").textContent),
    ["Primary", "Secondary outcomes", "Guardrails", "Added (exploratory)"]
  );
  assert.deepEqual(groups.map((group) => group.querySelectorAll(".metric-tab").length), [1, 12, 7, 10]);
  assert.equal(byId(dashboard, "metric-filter").hidden, false);
  assert.deepEqual(selected(dashboard), ["revenue_01"]);

  filter(dashboard, "LATENCY");
  assert.equal(shown(dashboard).length, 10);
  assert.ok(shown(dashboard).every((key) => key.startsWith("latency_")));
  assert.deepEqual(groups.map((group) => group.hidden), [true, false, false, false]);
  assert.deepEqual(selected(dashboard), ["revenue_01"]);

  const choice = tabs(dashboard).find((tab) => tab.dataset.metric === "latency_20");
  dashboard.panels.explore.dispatch("click", { target: choice });
  assert.deepEqual(selected(dashboard), ["latency_20"]);
  assert.equal(byId(dashboard, "explore-metric-name").textContent, "latency 20");

  filter(dashboard, "no such metric");
  assert.deepEqual(shown(dashboard), []);
  assert.ok(groups.every((group) => group.hidden));
  assert.equal(byId(dashboard, "metric-filter-empty").hidden, false);
  assert.deepEqual(selected(dashboard), ["latency_20"]);

  filter(dashboard, "guardrail");
  assert.deepEqual(shown(dashboard), tabs(dashboard).slice(13, 20).map((tab) => tab.dataset.metric));

  filter(dashboard, "");
  assert.equal(shown(dashboard).length, 30);
  assert.equal(byId(dashboard, "metric-filter-empty").hidden, true);
});

test("a short focus-metric list shows every metric without a filter", () => {
  const dashboard = dashboardWith(3);
  assert.equal(byId(dashboard, "metric-filter").hidden, true);
  assert.equal(shown(dashboard).length, 3);
});

test("captured-result search decodes timestamp and named-date identities", () => {
  const cell = {
    metric: "revenue_01", analysis_population: "assigned", group_id: "T",
    method: "unadjusted", method_role: "decision", dimension: null,
    dimension_value: null, source: null, ds_basis: "calendar",
  };
  const dashboard = dashboardWith(1, {
    results: {
      timestamp: { cell: { ...cell, ds: { datetime: "2025-01-15T00:00:00Z" } }, sourceId: "capture" },
      named: { cell: { ...cell, ds: { label: "day zero" } }, sourceId: "capture" },
    },
    views: { '["readout"]': ["timestamp", "named"] },
  });
  dashboard.tabs.readout.dispatch("click");
  const row = dashboard.readoutTable.querySelector("tr");
  dashboard.readoutTable.dispatch("click", { target: row.querySelector(".metric-select"), detail: 0 });
  const search = byId(dashboard, "result-filter");
  search.value = "2025-01-15T00:00:00Z";
  search.dispatch("input");
  assert.deepEqual(byId(dashboard, "result-choice").children.map(x => x.getAttribute("value")), ["", "timestamp"]);
  search.value = "day zero";
  search.dispatch("input");
  assert.deepEqual(byId(dashboard, "result-choice").children.map(x => x.getAttribute("value")), ["", "named"]);
});

test("pointer selection brings the responsive inspector into focus", () => {
  const dashboard = dashboardWith(1);
  dashboard.context.window.matchMedia = () => ({ matches: true });
  const button = dashboard.readoutTable.querySelector(".metric-select");
  dashboard.readoutTable.dispatch("click", { target: button, detail: 1 });
  assert.equal(dashboard.document.activeElement?.id, "inspector-title");
});

test("table data stays inert while metric names open result evidence", () => {
  for (const target of ["readout-table", "report-table", "overview-table"]) {
    const dashboard = dashboardWith(1);
    dashboard.tabs[target === "report-table" ? "report" : target === "overview-table" ? "explore" : "readout"].dispatch("click");
    const host = byId(dashboard, target);
    const row = host.querySelector("tr") || new FakeElement("tr");
    row.dataset.metric = "revenue_01";
    row.dataset.incResultRefs = "[]";
    let name = row.querySelector(".metric-select");
    if (!name) {
      name = new FakeElement("button", { className: "metric-select" });
      name.dataset.metric = "revenue_01";
      row.append(name);
    }
    const value = new FakeElement("td");
    value.textContent = "+12.0%";
    row.append(value);
    if (!row.parentElement) host.append(row);
    byId(dashboard, "close-inspector").dispatch("click");
    const events = target === "overview-table" ? dashboard.panels.explore : host;
    events.dispatch("click", { target: value, detail: 1 });
    assert.equal(byId(dashboard, "inspector").hidden, true, target);
    events.dispatch("click", { target: name, detail: 1 });
    assert.equal(byId(dashboard, "inspector").hidden, false, target);
  }
});

test("result choices distinguish original and Overview-corrected readings", () => {
  const cell = { metric: "revenue_01", analysis_population: "assigned", group_id: "T",
    method: "unadjusted", method_role: "decision", dimension: "country",
    dimension_value: "US", source: "events", ds_basis: null, ds: null };
  const dashboard = dashboardWith(1, {
    results: {
      original: { cell, sourceId: "capture", viewLabels: ["Original segment analysis"], selectionLabel: "Original analysis" },
      corrected: { cell, sourceId: "capture", viewLabels: ["Explore Overview"], selectionLabel: "Overview correction" },
    },
    views: { '["readout"]': ["original", "corrected"] },
  });
  dashboard.tabs.readout.dispatch("click");
  dashboard.readoutTable.dispatch("click", {
    target: dashboard.readoutTable.querySelector(".metric-select"), detail: 0,
  });
  const choices = byId(dashboard, "result-choice").children.slice(1);
  assert.notEqual(choices[0].textContent, choices[1].textContent);
  const search = byId(dashboard, "result-filter");
  search.value = "Overview correction";
  search.dispatch("input");
  assert.deepEqual(byId(dashboard, "result-choice").children.map(x => x.getAttribute("value")), ["", "corrected"]);
});
