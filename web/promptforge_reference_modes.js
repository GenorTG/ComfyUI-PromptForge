import { app } from "../../scripts/app.js";

const NODE = "PROMPTFORGE_V2_H3_REFERENCES";
const KEEP = {
  t2v: [],
  i2v: ["first_frame"],
  l2v: ["last_frame"],
  fl2v: ["first_frame", "last_frame"],
  ref2v: ["reference_images", "reference_video", "reference_audio"],
  continue: ["reference_video", "reference_audio"],
};

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);

function apply(node) {
  const mode = widget(node, "mode")?.value || "t2v";
  const keep = KEEP[mode] || [];
  for (const slot of node.inputs || []) {
    if (!["first_frame", "last_frame", "reference_images", "reference_video", "reference_audio"].includes(slot.name)) continue;
    const want = keep.includes(slot.name);
    if (!slot.link && !want) {
      slot.hidden = true;
      slot.shape = 7;
    } else {
      slot.hidden = false;
      slot.shape = want ? 0 : 7;
    }
  }
  // frame_fit only matters when a frame drives the format
  const fit = widget(node, "frame_fit");
  if (fit) fit.hidden = !["i2v", "l2v", "fl2v"].includes(mode);
  node.setSize?.(node.computeSize());
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "PromptForge.V2.ReferenceModes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const mode = widget(this, "mode");
      const original = mode?.callback;
      if (mode) mode.callback = (...args) => { const result = original?.apply(mode, args); apply(this); return result; };
      requestAnimationFrame(() => apply(this));
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      configured?.apply(this, arguments);
      requestAnimationFrame(() => apply(this));
    };
  },
});
