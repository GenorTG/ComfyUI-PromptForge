import { app } from "../../scripts/app.js";

const NODE = "PROMPTFORGE_EmptyLatent";

function setHidden(widget, hidden) {
  if (widget.promptforgeSize === undefined) widget.promptforgeSize = widget.computeSize ?? null;
  widget.hidden = hidden;
  widget.computeSize = hidden ? () => [0, -4] : widget.promptforgeSize ?? undefined;
}

app.registerExtension({
  name: "PromptForge.EmptyLatent",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const widget = (name) => node.widgets.find((item) => item.name === name);

      // only the size inputs of the chosen mode, and the frame inputs for video
      node.promptforgeLayout = () => {
        const mode = widget("res_mode").value;
        const video = widget("kind").value === "video";
        setHidden(widget("aspect_ratio"), mode !== "aspect + megapixels");
        setHidden(widget("megapixels"), mode !== "aspect + megapixels");
        setHidden(widget("preset"), mode !== "preset");
        setHidden(widget("width"), mode !== "custom");
        setHidden(widget("height"), mode !== "custom");
        for (const name of ["frames", "seconds", "fps"]) setHidden(widget(name), !video);
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };
      for (const name of ["res_mode", "kind"]) {
        const target = widget(name);
        const original = target.callback;
        target.callback = function () {
          const result = original?.apply(this, arguments);
          node.promptforgeLayout();
          return result;
        };
      }
      node.promptforgeOnPipe = (loader) => {
        const family = widget("family");
        const label = loader.promptforgeFamilyLabel?.();
        if (family.options.values.includes(label)) family.value = label;
      };
      node.promptforgeLayout();
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeLayout?.();
      return result;
    };
  },
});
