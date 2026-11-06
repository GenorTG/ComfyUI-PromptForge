import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const OVERRIDE = "PROMPTFORGE_PreviewOverride";
const HEIGHT = 240;

// The live preview of a sampling, drawn inside a node: the server sends a short animation (mp4 / WebP) or a still per step.
export function addLivePreview(node) {
  const frame = document.createElement("div");
  Object.assign(frame.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden", display: "none" });
  const media = { background: "rgba(0,0,0,.25)", borderRadius: "4px", position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain" };
  const image = document.createElement("img");
  const video = document.createElement("video");
  Object.assign(image.style, media);
  Object.assign(video.style, media, { display: "none" });
  Object.assign(video, { loop: true, muted: true, autoplay: true, playsInline: true });
  const label = document.createElement("div");
  Object.assign(label.style, { position: "absolute", left: "6px", top: "4px", padding: "1px 6px", fontSize: "11px", borderRadius: "3px", background: "rgba(0,0,0,.55)", color: "#fff", zIndex: 1 });
  frame.append(image, video, label);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());

  let shown = false;
  const widget = node.addDOMWidget("live_preview", "promptforge_live_preview", frame, { serialize: false, getMinHeight: () => (shown ? HEIGHT : 0) });
  Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
  const show = (value) => {
    if (value === shown) return;
    shown = value;
    frame.style.display = value ? "" : "none";
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
  };
  node.promptforgeLive = ({ image: data, mime, label: text, step, total }) => {
    show(true);
    label.textContent = `${text ? text + " · " : ""}step ${step}/${total}`;
    const moving = mime === "video/mp4";
    video.style.display = moving ? "" : "none";
    image.style.display = moving ? "none" : "";
    const source = `data:${mime};base64,${data}`;
    if (moving) { video.src = source; video.play().catch(() => {}); } else image.src = source;
  };
  node.promptforgeLiveHide = () => { video.pause(); show(false); };
}

app.registerExtension({
  name: "PromptForge.Preview",
  setup() {
    api.addEventListener("promptforge.preview", (event) => app.graph.getNodeById(Number(event.detail.node))?.promptforgeLive?.(event.detail));
    // a finished or stopped run ends the live preview
    for (const name of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(name, () => app.graph._nodes.forEach((node) => node.promptforgeLiveHide?.()));
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== OVERRIDE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      addLivePreview(this);
    };
  },
});
