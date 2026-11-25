import { app } from "../../scripts/app.js";

// ComfyUI's side panel draws a selected node's widgets too and stores its own width on the shared widget object.
// The canvas then sizes the DOM panel from that value (about half the node), so these widgets always follow the node.
app.registerExtension({
  name: "PromptForge.DomWidgetWidth",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith("PROMPTFORGE")) return;
    const add = nodeType.prototype.addDOMWidget;
    nodeType.prototype.addDOMWidget = function () {
      const widget = add.apply(this, arguments);
      Object.defineProperty(widget, "width", { configurable: true, enumerable: true, get: () => undefined, set: () => {} });
      return widget;
    };
  },
});
