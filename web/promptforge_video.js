import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = ["PROMPTFORGE_SaveVideo", "PROMPTFORGE_H3Endless"];
const HEIGHT = 260;

app.registerExtension({
  name: "PromptForge.SaveVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const option = (name) => node.widgets.find((widget) => widget.name === name);

      const frame = document.createElement("div");
      Object.assign(frame.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden", contain: "layout paint" });
      const video = document.createElement("video");
      Object.assign(video.style, { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", background: "rgba(0,0,0,.25)", borderRadius: "4px" });
      Object.assign(video, { controls: true, muted: true, autoplay: true, playsInline: true });
      frame.append(video);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());

      const visible = () => option("show_preview")?.value !== false && !!video.getAttribute("src");
      const widget = node.addDOMWidget("video_preview", "promptforge_video_preview", frame, { serialize: false, getMinHeight: () => (visible() ? HEIGHT : 0) });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
      const apply = () => {
        video.loop = option("loop_preview")?.value !== false;
        frame.style.display = visible() ? "" : "none";
        video.autoplay = option("autoplay_preview")?.value !== false;
        if (!visible()) video.pause();
        node.setSize([node.size[0], Math.max(node.computeSize()[1], visible() ? node.size[1] : 0)]);
        node.setDirtyCanvas(true, true);
      };
      for (const name of ["show_preview", "loop_preview", "autoplay_preview"]) {
        const toggle = option(name);
        if (!toggle) continue;
        const callback = toggle.callback;
        toggle.callback = function () { callback?.apply(this, arguments); apply(); };
      }
      if (nodeData.name === "PROMPTFORGE_H3Endless") {
        // every part's prompt, added as soon as it has been written
        const box = document.createElement("textarea");
        Object.assign(box.style, { width: "100%", height: "100%", resize: "none", boxSizing: "border-box", padding: "6px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" });
        Object.assign(box, { readOnly: true, placeholder: "The prompt written for each part appears here as it is made." });
        box.addEventListener("wheel", (event) => event.stopPropagation());
        box.addEventListener("pointerdown", (event) => event.stopPropagation());
        const holder = document.createElement("div");
        Object.assign(holder.style, { width: "100%", height: "100%" });
        holder.append(box);
        const prompts = node.addDOMWidget("written_prompts", "promptforge_written_prompts", holder, { serialize: false, getMinHeight: () => 220 });
        Object.defineProperty(prompts, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        node.promptforgeShowPrompt = ({ index, text }) => {
          box.value = (index === 0 ? "" : box.value) + `=== Part ${index + 1} ===\n${text}\n\n`;
          box.scrollTop = box.scrollHeight;
        };
      }
      node.promptforgeShowVideo = (preview) => {
        const query = new URLSearchParams({ filename: preview.filename, subfolder: preview.subfolder, type: preview.type, t: String(Date.now()) });
        video.src = api.apiURL("/view?" + query);
        apply();
        if (video.autoplay) video.play().catch(() => {});
      };
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const preview = message?.promptforge_video?.[0];
      if (preview) this.promptforgeShowVideo(preview);
      for (const entry of message?.promptforge_prompt || []) this.promptforgeShowPrompt?.(entry);
    };
  },
});
