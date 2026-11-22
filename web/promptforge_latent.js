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
        const random = widget("randomize_ratio").value;
        const byAspect = mode === "aspect + megapixels" || random;
        setHidden(widget("aspect_ratio"), mode !== "aspect + megapixels" || random);
        setHidden(widget("megapixels"), !byAspect);
        setHidden(widget("ratio_pool"), !random);
        setHidden(widget("orientation"), widget("randomize_orientation").value);
        setHidden(widget("seed"), !random && !widget("randomize_orientation").value);
        setHidden(widget("preset"), mode !== "preset" || random);
        setHidden(widget("width"), mode !== "custom" || random);
        setHidden(widget("height"), mode !== "custom" || random);
        for (const name of ["frames", "seconds", "fps"]) setHidden(widget(name), !video);
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };
      for (const name of ["res_mode", "kind", "randomize_ratio", "randomize_orientation"]) {
        const target = widget(name);
        const original = target.callback;
        target.callback = function () {
          const result = original?.apply(this, arguments);
          node.promptforgeLayout();
          return result;
        };
      }
      node.promptforgeLayout();
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      const width = this.size[0], height = this.size[1];
      this.promptforgeLayout?.();
      // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
      this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
      return result;
    };
  },
});
