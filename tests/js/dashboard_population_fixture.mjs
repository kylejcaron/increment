import vm from "node:vm";
import fs from "node:fs";
import { createDocument, FakeElement } from "./dom_stub.mjs";

function embeddedPayload(html) {
  const match = html.match(/<script type="application\/json" id="dashboard-data">([\s\S]*?)<\/script>/);
  if (!match) throw new Error("built dashboard has no embedded payload");
  return JSON.parse(match[1]);
}

function shellSource(html) {
  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)];
  if (!scripts.length) throw new Error("built dashboard has no shell script");
  return scripts.at(-1)[1];
}

export function loadDashboardFromFile(path) {
  const html = fs.readFileSync(path, "utf8");
  const payload = embeddedPayload(html);
  const document = createDocument();
  const body = new FakeElement("body");
  const shell = new FakeElement("div", { id: "shell" });
  document.append(body);
  body.append(shell);
  document.body = body;
  document.documentElement = document;
  document.title = "";

  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  ids.delete("shell");
  for (const id of ids) shell.append(new FakeElement("div", { id }));
  const dataNode = document.getElementById("dashboard-data");
  dataNode.textContent = JSON.stringify(payload);

  const tabList = new FakeElement("div");
  tabList.setAttribute("role", "tablist");
  shell.append(tabList);
  const tabs = {};
  const panels = {};
  for (const view of ["readout", "explore", "health", "report"]) {
    const tab = document.getElementById(`tab-${view}`);
    tab.setAttribute("role", "tab");
    tab.dataset.viewTarget = view;
    tabList.append(tab);
    tabs[view] = tab;
    const panel = document.getElementById(`panel-${view}`);
    panel.dataset.view = view;
    panels[view] = panel;
  }
  document.getElementById("population-select").value = payload.defaultPopulation;

  const objectUrls = new Map();
  let nextObjectUrl = 0;
  const URL = {
    createObjectURL(blob) {
      const url = `blob:dashboard-${++nextObjectUrl}`;
      objectUrls.set(url, blob);
      return url;
    },
    revokeObjectURL(url) {
      objectUrls.delete(url);
    },
  };
  const windowListeners = {};
  const hostMessages = [];
  const parent = {
    postMessage(message) {
      hostMessages.push(message);
    },
  };
  const window = {
    addEventListener(type, handler) {
      (windowListeners[type] ??= []).push(handler);
    },
    frameElement: null,
    matchMedia: null,
    print() {},
  };
  window.parent = parent;

  function createNativeAdapter() {
    return {
      mountNative(host, markup, options = {}) {
        host.innerHTML = markup;
        if (options.metrics !== "select" || host.id !== "readout-table") return;
        const view = Object.values(payload.populationViews).find((candidate) => candidate.results === markup);
        for (const metric of view?.metrics ?? []) {
          const row = new FakeElement("tr");
          row.dataset.metric = metric.key;
          const button = new FakeElement("button", { className: "metric-select" });
          button.dataset.metric = metric.key;
          button.setAttribute("aria-pressed", "false");
          row.append(button);
          host.append(row);
        }
      },
      planBands() { return null; },
      keepColumns() {},
    };
  }

  const context = vm.createContext({
    document,
    window,
    location: { hash: "" },
    URL,
    Blob,
    setTimeout,
    clearTimeout,
    createNativeAdapter,
  });
  vm.runInContext(shellSource(html), context, { filename: "dashboard-shell.js" });
  function dispatchHostMessage(data) {
    for (const handler of windowListeners.message ?? []) handler({ source: parent, data });
  }

  return {
    payload,
    hostMessages,
    dispatchHostMessage,
    document,
    context,
    objectUrls,
    tabs,
    panels,
    populationSelect: document.getElementById("population-select"),
    heroEffect: document.getElementById("hero-primary-effect"),
    readoutTable: document.getElementById("readout-table"),
    healthEvidence: document.getElementById("health-evidence"),
    reportDownload: document.getElementById("report-download"),
  };
}
