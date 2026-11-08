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
function dashboardWith(count) {
  const metrics = ROLES.slice(0, count).map((role, index) => {
    const key = `${TOPICS[index % TOPICS.length]}_${String(index + 1).padStart(2, "0")}`;
    return { key, label: key.replace("_", " "), role, detail: "", csv: "" };
  });
  const view = {
    title: "Many metrics",
    results: "",
    metrics,
    scopes: { overall: { label: "Whole experiment", dimension: null, source: null } },
    overview: {},
    explore: {
      overall: Object.fromEntries(
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
