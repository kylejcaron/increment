// Hosts the dashboard page in an iframe and relays the Explore picker to Python.
//
// The page posts {source: "inc-dashboard", type: "ready" | "set-exploratory-metrics"}.
// This widget answers {source: "inc-dashboard-host", type: "bridge" | "status" | "restore"}.
// Python prepares a new snapshot when `exploratory_metrics` changes and replaces `document`;
// the replacement page reopens Explore, where the change was requested.
function render({ model, el }) {
  const frame = document.createElement("iframe");
  frame.className = "inc-dashboard-frame";
  frame.title = "Experiment dashboard";
  let reopen = null;

  const post = (message) => {
    if (frame.contentWindow) {
      frame.contentWindow.postMessage({ source: "inc-dashboard-host", ...message }, "*");
    }
  };

  const onMessage = (event) => {
    if (event.source !== frame.contentWindow) return;
    const message = event.data;
    if (!message || message.source !== "inc-dashboard") return;
    if (message.type === "ready") {
      post({ type: "bridge" });
      if (reopen) post({ type: "restore", view: reopen });
      reopen = null;
    } else if (message.type === "set-exploratory-metrics" && Array.isArray(message.metrics)) {
      model.set("exploratory_metrics", message.metrics.map(String));
      model.save_changes();
    }
  };
  window.addEventListener("message", onMessage);

  model.on("change:status", () => post({ type: "status", status: model.get("status") }));
  model.on("change:document", () => {
    reopen = "explore";
    frame.srcdoc = model.get("document");
  });

  frame.srcdoc = model.get("document");
  el.appendChild(frame);
  return () => window.removeEventListener("message", onMessage);
}

export default { render };
