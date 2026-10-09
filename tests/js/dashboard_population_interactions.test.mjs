import assert from "node:assert/strict";
import { test } from "node:test";
import { loadDashboardFromFile } from "./dashboard_population_fixture.mjs";

const population = (dashboard, name) => dashboard.payload.populationViews[name];

function selectPopulation(dashboard, name) {
  dashboard.populationSelect.value = name;
  dashboard.populationSelect.dispatch("change");
}

function assertViewSelected(dashboard, view) {
  assert.equal(dashboard.tabs[view].getAttribute("aria-selected"), "true");
  assert.equal(dashboard.panels[view].hidden, false);
}

test("population switching updates visible readouts and downloads without losing shared selections", async () => {
  const dashboard = loadDashboardFromFile(process.env.INCREMENT_DASHBOARD_DOCUMENT);
  const triggered = population(dashboard, "triggered");
  const assigned = population(dashboard, "assigned");
  const metric = triggered.metrics[0];

  assertViewSelected(dashboard, "readout");
  assert.equal(dashboard.heroEffect.textContent, triggered.primary.effect);
  assert.equal(dashboard.readoutTable.innerHTML, triggered.results);
  assert.equal(dashboard.healthEvidence.innerHTML, "");

  const metricButton = dashboard.readoutTable.querySelector(".metric-select");
  assert.ok(metricButton, "readout metric row is available for selection");
  dashboard.readoutTable.dispatch("click", { target: metricButton, detail: 1 });
  assert.equal(dashboard.document.getElementById("inspector-title").textContent, metric.label);
  assert.equal(
    dashboard.readoutTable.querySelector("tr[data-metric]").getAttribute("data-selected"),
    ""
  );

  selectPopulation(dashboard, "assigned");
  assertViewSelected(dashboard, "readout");
  assert.equal(dashboard.heroEffect.textContent, assigned.primary.effect);
  assert.equal(dashboard.readoutTable.innerHTML, assigned.results);
  assert.equal(dashboard.document.getElementById("inspector-title").textContent, metric.label);
  assert.equal(
    dashboard.readoutTable.querySelector("tr[data-metric]").getAttribute("data-selected"),
    ""
  );

  selectPopulation(dashboard, "triggered");
  assertViewSelected(dashboard, "readout");
  assert.equal(dashboard.heroEffect.textContent, triggered.primary.effect);
  assert.equal(dashboard.readoutTable.innerHTML, triggered.results);
  assert.equal(dashboard.document.getElementById("inspector-title").textContent, metric.label);

  dashboard.tabs.health.dispatch("click");
  assertViewSelected(dashboard, "health");
  assert.equal(dashboard.healthEvidence.innerHTML, triggered.health);
  selectPopulation(dashboard, "assigned");
  assertViewSelected(dashboard, "health");
  assert.equal(dashboard.healthEvidence.innerHTML, assigned.health);
  selectPopulation(dashboard, "triggered");
  assertViewSelected(dashboard, "health");
  assert.equal(dashboard.healthEvidence.innerHTML, triggered.health);

  dashboard.tabs.report.dispatch("click");
  assertViewSelected(dashboard, "report");
  const triggeredUrl = dashboard.reportDownload.href;
  assert.equal(await dashboard.objectUrls.get(triggeredUrl).text(), triggered.readoutCsv);
  selectPopulation(dashboard, "assigned");
  assertViewSelected(dashboard, "report");
  const assignedUrl = dashboard.reportDownload.href;
  assert.notEqual(assignedUrl, triggeredUrl);
  assert.equal(await dashboard.objectUrls.get(assignedUrl).text(), assigned.readoutCsv);
  selectPopulation(dashboard, "triggered");
  assertViewSelected(dashboard, "report");
  assert.equal(await dashboard.objectUrls.get(dashboard.reportDownload.href).text(), triggered.readoutCsv);
});
