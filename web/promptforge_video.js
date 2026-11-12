import { app } from "../../scripts/app.js";

// Save Video: ComfyUI shows the saved video itself; here are its facts (size, frames, codec) under it and the show / loop / autoplay switches
const NODES = ["PROMPTFORGE_SaveVideo"];

app.registerExtension({
  name: "PromptForge.SaveVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const option = (name) => node.widgets.find((widget) => widget.name === name);

      const label = Object.assign(document.createElement("div"), { textContent: "" });
      Object.assign(label.style, { fontSize: "12px", padding: "3px 6px", color: "var(--input-text, #ddd)", opacity: ".85", boxSizing: "border-box", wordBreak: "break-word" });
      const widget = node.addDOMWidget("video_info", "promptforge_video_info", label, { serialize: false, getMinHeight: () => (label.textContent ? 36 : 0) });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });

      // the video element belongs to the frontend and appears after the result arrives
      const apply = (tries = 0) => {
        const video = node.videoContainer?.querySelector?.("video");
        if (!video) {
          if (tries < 30) setTimeout(() => apply(tries + 1), 100);
          return;
        }
        video.loop = option("loop_preview")?.value !== false;
        video.autoplay = option("autoplay_preview")?.value !== false;
        node.videoContainer.style.display = option("show_preview")?.value !== false ? "" : "none";
        if (!video.autoplay) video.pause();
        else video.play?.().catch(() => {});
      };
      for (const name of ["show_preview", "loop_preview", "autoplay_preview"]) {
        const toggle = option(name);
        if (!toggle) continue;
        const callback = toggle.callback;
        toggle.callback = function () { callback?.apply(this, arguments); apply(); };
      }
      node.promptforgeShowInfo = (text) => {
        label.textContent = text;
        node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
        apply();
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
