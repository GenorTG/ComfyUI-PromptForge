import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = ["PROMPTFORGE_SaveVideo", "PROMPTFORGE_H3Endless"];
const HEIGHT = 260;

app.registerExtension({
  name: "PromptForge.SaveVideo",
  setup() {
    api.addEventListener("promptforge.preview", (event) => app.graph.getNodeById(Number(event.detail.node))?.promptforgeLive?.(event.detail));
    // a finished or stopped run ends the live preview
    for (const name of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(name, () => app.graph._nodes.forEach((node) => node.promptforgeLiveHide?.()));
    }
  },
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
        // PromptForge's own live preview of the sampling: a few latent frames played as a short loop
        const liveFrame = document.createElement("div");
        Object.assign(liveFrame.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden", display: "none" });
        const liveImage = document.createElement("img");
        Object.assign(liveImage.style, { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", background: "rgba(0,0,0,.25)", borderRadius: "4px" });
        const liveLabel = document.createElement("div");
        Object.assign(liveLabel.style, { position: "absolute", left: "6px", top: "4px", padding: "1px 6px", fontSize: "11px", borderRadius: "3px", background: "rgba(0,0,0,.55)", color: "#fff" });
        liveFrame.append(liveImage, liveLabel);
        let liveTimer = null;
        let liveShown = false;
        const live = node.addDOMWidget("live_preview", "promptforge_live_preview", liveFrame, { serialize: false, getMinHeight: () => (liveShown ? 200 : 0) });
        Object.defineProperty(live, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        const setLive = (shown) => {
          if (shown === liveShown) return;
          liveShown = shown;
          liveFrame.style.display = shown ? "" : "none";
          node.setSize([node.size[0], node.computeSize()[1]]);
          node.setDirtyCanvas(true, true);
        };
        node.promptforgeLive = ({ frames, label, step, total }) => {
          setLive(true);
          liveLabel.textContent = `${label} · step ${step}/${total}`;
          clearInterval(liveTimer);
          let position = 0;
          liveImage.src = frames[0];
          if (frames.length > 1) liveTimer = setInterval(() => { position = (position + 1) % frames.length; liveImage.src = frames[position]; }, 140);
        };
        node.promptforgeLiveHide = () => { clearInterval(liveTimer); setLive(false); };
        node.promptforgeShowPrompt = ({ index, text }) => {
          box.value = (index === 0 ? "" : box.value) + `=== Part ${index + 1} ===\n${text}\n\n`;
          box.scrollTop = box.scrollHeight;
        };
      }
      node.promptforgeShowVideo = (preview) => {
        const query = new URLSearchParams({ filename: preview.filename, subfolder: preview.subfolder, type: preview.type, t: String(Date.now()) });
        node.promptforgeLiveHide?.();
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
