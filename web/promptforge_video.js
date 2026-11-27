import { app } from "../../scripts/app.js";

// Save Video: ComfyUI shows the saved video itself; here are its facts (size, frames, codec) under it and the show / loop / autoplay switches
const NODES = ["PROMPTFORGE_SaveVideo"];
const ENCODERS = ["PROMPTFORGE_SaveVideo", "PROMPTFORGE_V2_H3_PREVIEW_SAVE"];

// h264 nvenc has no 10-bit: while it is picked, the bit depth list offers 8 only
app.registerExtension({
  name: "PromptForge.VideoBitDepth",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!ENCODERS.includes(nodeData.name)) return;
    const limit = (node) => {
      const codec = node.widgets?.find((widget) => widget.name === "codec");
      const depth = node.widgets?.find((widget) => widget.name === "bit_depth");
      if (!codec || !depth) return;
      const only8 = codec.value === "h264 (nvenc)";
      depth.options.values = only8 ? ["8"] : ["8", "10"];
      if (only8) depth.value = "8";
    };
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const node = this;
      const codec = node.widgets?.find((widget) => widget.name === "codec");
      if (codec) {
        const callback = codec.callback;
        codec.callback = function () {
          const answer = callback?.apply(this, arguments);
          limit(node);
          return answer;
        };
      }
      limit(node);
      return result;
    };
    // loading a workflow sets the values without calling the widget callbacks
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments);
      limit(this);
      return result;
    };
  },
});

// show / loop / autoplay of the video ComfyUI puts in a save node. The frontend builds that player when the file has
// loaded and resets it for every result, so the switches are applied whenever the player or a switch changes.
export function applyPreviewSwitches(node, video, preview) {
  const on = (name) => node.widgets?.find((widget) => widget.name === name)?.value !== false;
  const show = on("show_preview"), loop = on("loop_preview"), autoplay = on("autoplay_preview");
  if (preview && !!preview.hidden === show) {
    preview.promptforgeLayout ||= preview.computeLayoutSize;
    preview.hidden = !show;
    preview.computeLayoutSize = show ? preview.promptforgeLayout : () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    if (preview.element) preview.element.hidden = !show;
    // the node gives up the player's room and takes it back
    if (!show) {
      node.promptforgeHeight = node.size?.[1];
      const needed = node.computeSize?.()[1];
      if (needed < node.promptforgeHeight) node.setSize?.([node.size[0], needed]);
    } else if (node.promptforgeHeight > node.size?.[1]) node.setSize?.([node.size[0], node.promptforgeHeight]);
    node.setDirtyCanvas?.(true, true);
  }
  if (!video) return;
  const state = `${show}|${loop}|${autoplay}|${video.src}`;
  if (video.promptforgeState === state) return;
  video.promptforgeState = state;
  video.hidden = !show;
  video.loop = loop;
  video.autoplay = show && autoplay;
  // a browser only starts a video by itself while it is muted
  if (video.autoplay) video.play?.()?.catch(() => { video.muted = true; video.play?.()?.catch(() => {}); });
  else video.pause?.();
}

app.registerExtension({
  name: "PromptForge.VideoPreviewSwitches",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!ENCODERS.includes(nodeData.name)) return;
    const apply = (node) => {
      // canvas nodes keep the player in videoContainer, Vue nodes inside their own element
      const video = node.videoContainer?.querySelector?.("video") || document.querySelector?.(`[data-node-id="${node.id}"] video`);
      applyPreviewSwitches(node, video, node.widgets?.find((widget) => widget.name === "video-preview"));
    };
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const node = this;
      for (const widget of node.widgets || []) {
        if (!["show_preview", "loop_preview", "autoplay_preview"].includes(widget.name)) continue;
        const callback = widget.callback;
        widget.callback = function () { const answer = callback?.apply(this, arguments); apply(node); return answer; };
      }
      // the player appears some time after the result, with no event of its own
      const timer = setInterval(() => apply(node), 300);
      const removed = node.onRemoved;
      node.onRemoved = function () { clearInterval(timer); return removed?.apply(this, arguments); };
      return result;
    };
  },
});

app.registerExtension({
  name: "PromptForge.SaveVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;

      const label = Object.assign(document.createElement("div"), { textContent: "" });
      Object.assign(label.style, { fontSize: "12px", padding: "3px 6px", color: "var(--input-text, #ddd)", opacity: ".85", boxSizing: "border-box", wordBreak: "break-word" });
      const widget = node.addDOMWidget("video_info", "promptforge_video_info", label, { serialize: false, getMinHeight: () => (label.textContent ? 36 : 0) });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });

      node.promptforgeShowInfo = (text) => {
        label.textContent = text;
        node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
      };
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const info = message?.promptforge_info?.[0];
      if (info) this.promptforgeShowInfo(info);
    };
  },
});
