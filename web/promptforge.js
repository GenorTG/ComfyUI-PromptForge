import { app } from "../../scripts/app.js";

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function widgetValues(node) {
  const values = {};
  for (const widget of node.widgets || []) {
    if (widget.name && widget.value !== undefined) values[widget.name] = widget.value;
  }
  return values;
}

function setWidgets(node, values) {
  for (const [name, value] of Object.entries(values)) {
    const widget = (node.widgets || []).find((item) => item.name === name);
    if (widget) widget.value = value;
  }
  node.setDirtyCanvas(true, true);
}

function showReport(text) {
  if (app.ui?.dialog?.show) {
    app.ui.dialog.show(text);
  } else {
    alert(text);
  }
}

app.registerExtension({
  name: "PromptForge.TestLoad",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["PROMPTFORGE_Model", "PROMPTFORGE_ConfigFinder"].includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const isModel = nodeData.name === "PROMPTFORGE_Model";
      this.addWidget("button", isModel ? "Test load & apply" : "Scan local models", null, async () => {
        try {
          const body = widgetValues(this);
          if (isModel) {
            showReport("Loading the selected model to verify the settings...");
            const result = await post("/promptforge/test", body);
            if (result.ok) {
              setWidgets(this, result.settings || {});
            }
            showReport((result.ok ? "Verified. " : "Failed. ") + (result.message || ""));
          } else {
            const result = await post("/promptforge/scan", body);
            showReport(result.report);
          }
        } catch (error) {
          showReport("PromptForge: " + (error?.message || error));
        }
      });
    };
  },
});

// Workflows saved by older versions keep input slots that a node no longer has (the Enhancer used to take a clip).
app.registerExtension({
  name: "PromptForge.StaleInputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith("PROMPTFORGE_")) return;
    const known = new Set(Object.keys({ ...nodeData.input?.required, ...nodeData.input?.optional }));
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      for (let index = (this.inputs?.length ?? 0) - 1; index >= 0; index--) {
        if (!known.has(this.inputs[index].name)) this.removeInput(index);
      }
      return result;
    };
  },
});

// Nodes with a PIPE input learn from the PromptForge Loader wired to it (family, recommended settings).
app.registerExtension({
  name: "PromptForge.Pipe",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith("PROMPTFORGE_") || !nodeData.input?.optional?.pipe) return;
    const onConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link) {
      onConnections?.apply(this, arguments);
      if (type !== LiteGraph.INPUT || !connected || this.inputs?.[index]?.name !== "pipe" || !link) return;
      const origin = this.graph?.getNodeById(link.origin_id);
      if (origin?.promptforgeFamily) this.promptforgeOnPipe?.(origin);
    };
  },
});
