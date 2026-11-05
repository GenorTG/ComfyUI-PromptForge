import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_LoadImage";

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

import { createLibrary, thumb, nsfwShown, setNsfwShown, BLUR } from "./promptforge_library.js";

const HISTORY_KEY = "promptforge.descriptionHistory";
const readHistory = () => { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; } };

// every text typed into the description box is kept (newest first), so nothing typed can be lost for good
function remember(text) {
  text = text.trim();
  if (!text) return;
  const history = readHistory();
  const [latest] = history;
  if (latest && Date.now() - latest.time < 60000 && (text.startsWith(latest.text) || latest.text.startsWith(text) && text.length > latest.text.length / 2)) history.shift();
  else if (history.some((entry) => entry.text === text)) return;
  history.unshift({ text, time: Date.now() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 60)));
}

function showHistory(onRestore) {
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10003, background: "rgba(0,0,0,.6)", display: "flex", alignItems: "center", justifyContent: "center" });
  const card = el("div", { width: "min(640px, 94vw)", maxHeight: "80vh", overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px", padding: "12px", boxSizing: "border-box", background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "8px", fontSize: "13px" });
  const close = () => backdrop.remove();
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(); });
  card.append(el("div", { fontWeight: "bold" }, { textContent: "Description history (click one to put it back in the box)" }));
  const history = readHistory();
  if (!history.length) card.append(el("div", { opacity: ".7" }, { textContent: "Nothing typed yet." }));
  for (const entry of history) {
    const row = el("div", { cursor: "pointer", padding: "6px 8px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: "110px", overflow: "hidden" }, { textContent: entry.text });
    row.prepend(el("div", { opacity: ".55", fontSize: "11px" }, { textContent: new Date(entry.time).toLocaleString() }));
    row.addEventListener("click", () => { onRestore(entry.text); close(); });
    card.append(row);
  }
  backdrop.append(card);
  document.body.append(backdrop);
}

const BUTTON = { cursor: "pointer", padding: "3px 10px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" };

function chooseImage(current) {
  return new Promise((resolve) => {
    const done = (value) => { library.destroy(); backdrop.remove(); document.removeEventListener("keydown", onKey, true); resolve(value); };
    const onKey = (event) => { if (event.key === "Escape" && !document.querySelector("[data-pf-nested]")) { event.stopPropagation(); done(null); } };
    document.addEventListener("keydown", onKey, true);
    const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10001, background: "rgba(0,0,0,.6)", display: "flex", alignItems: "center", justifyContent: "center" });
    const card = el("div", { width: "min(1180px, 95vw)", height: "min(800px, 92vh)", display: "flex", flexDirection: "column", padding: "12px", boxSizing: "border-box", background: "var(--comfy-menu-bg, #222)", border: "1px solid var(--border-color, #555)", borderRadius: "8px" });
    backdrop.append(card);
    document.body.append(backdrop);
    backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) done(null); });

    const upload = el("button", BUTTON, { textContent: "Upload…" });
    const close = el("button", BUTTON, { textContent: "Close" });
    close.addEventListener("click", () => done(null));
    upload.addEventListener("click", () => {
      const picker = el("input", {}, { type: "file", accept: "image/*" });
      picker.addEventListener("change", async () => {
        const file = picker.files[0];
        if (!file) return;
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        const reply = await api.fetchApi("/upload/image", { method: "POST", body });
        if (reply.status === 200) {
          const saved = await reply.json();
          done(`${saved.subfolder ? saved.subfolder + "/" : ""}${saved.name}`);
        }
      });
      picker.click();
    });
    const library = createLibrary(card, { onPick: done, current, extraControls: [upload, close] });
    library.focus();
  });
}

app.registerExtension({
  name: "PromptForge.LoadImage",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const imageWidget = this.widgets.find((widget) => widget.name === "image");
      const textWidget = this.widgets.find((widget) => widget.name === "description");

      const frame = el("div", { position: "relative", width: "100%", height: "100%", minWidth: "0", overflow: "hidden", contain: "layout paint" });
      const root = el("div", { position: "absolute", inset: "0", display: "flex", flexDirection: "column", gap: "4px", boxSizing: "border-box" });
      const picture = el("img", { flex: "1", minHeight: "0", width: "100%", objectFit: "contain", borderRadius: "4px", background: "rgba(0,0,0,.25)", cursor: "pointer" }, { alt: "" });
      const buttons = el("div", { display: "flex", gap: "6px" });
      const choose = el("button", { ...BUTTON, flex: "1" }, { textContent: "Choose image…" });
      const save = el("button", BUTTON, { textContent: "Save description", title: "Store the text below as this image's description. The text box itself is never changed when you pick another image." });
      buttons.append(choose, save);
      const middle = el("div", { flex: "1", minHeight: "0", display: "flex", gap: "6px" });
      const meta = el("textarea", { width: "42%", minWidth: "0", resize: "none", boxSizing: "border-box", padding: "4px 6px", fontSize: "11px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", overflowY: "auto", display: "none" }, { readOnly: true, title: "Caption and keywords from the image library. Click to select, copy with Ctrl+C." });
      meta.addEventListener("wheel", (event) => event.stopPropagation());
      Object.assign(picture.style, { flex: "1", minWidth: "0", height: "100%", width: "auto" });
      middle.append(picture, meta);
      root.append(middle, buttons);
      frame.append(root);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());

      const show = async () => {
        picture.style.visibility = imageWidget.value ? "visible" : "hidden";
        if (!imageWidget.value) return;
        picture.src = thumb(imageWidget.value, 480);
        const info = await (await api.fetchApi("/promptforge/library/info?name=" + encodeURIComponent(imageWidget.value))).json();
        picture.dataset.nsfw = info.nsfw ? "1" : "";
        picture.style.filter = info.nsfw && !nsfwShown() ? BLUR : "none";
        const words = (info.tags || []).join(", ");
        meta.style.display = words || info.caption ? "" : "none";
        meta.value = [info.caption, words && "Tags: " + words].filter(Boolean).join("\n\n");
        const stored = (await (await api.fetchApi("/promptforge/image_description?name=" + encodeURIComponent(imageWidget.value))).json()).description;
        if (textWidget.inputEl) textWidget.inputEl.placeholder = stored ? "Stored description (used when this is empty):\n" + stored : "Description of this image / character. Empty = the stored description.";
      };
      const eye = el("button", BUTTON, { textContent: "👁", title: "Show or blur NSFW images (also in the picker)" });
      eye.addEventListener("click", () => { setNsfwShown(!nsfwShown()); show(); });
      buttons.append(eye);
      const history = el("button", BUTTON, { textContent: "↶", title: "History of everything typed in the description box" });
      history.addEventListener("click", () => showHistory((text) => { textWidget.value = text; node.setDirtyCanvas(true, true); }));
      buttons.append(history);
      let typing;
      const watch = () => {
        if (!textWidget.inputEl || textWidget.inputEl.dataset.pfWatched) return;
        textWidget.inputEl.dataset.pfWatched = "1";
        textWidget.inputEl.addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => remember(textWidget.value), 800); });
        textWidget.inputEl.addEventListener("blur", () => remember(textWidget.value));
      };
      watch();
      setTimeout(watch, 500);
      const pick = async () => {
        const name = await chooseImage(imageWidget.value);
        if (!name) return;
        imageWidget.value = name;
        show();
        node.setDirtyCanvas(true, true);
      };
      choose.addEventListener("click", pick);
      picture.addEventListener("click", pick);
      save.addEventListener("click", async () => {
        if (!imageWidget.value || !textWidget.value.trim()) return;
        const reply = await api.fetchApi("/promptforge/image_description", { method: "POST", body: JSON.stringify({ name: imageWidget.value, description: textWidget.value }) });
        const answer = await reply.json();
        save.textContent = answer.saved ? "Saved ✓" : "Failed";
        setTimeout(() => { save.textContent = "Save description"; }, 1500);
      });

      const widget = this.addDOMWidget("image_picker", "promptforge_image_picker", frame, { serialize: false, getMinHeight: () => 220 });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
      // keep the picker above the text fields
      this.widgets.splice(this.widgets.indexOf(widget), 1);
      this.widgets.splice(0, 0, widget);
      imageWidget.callback = show;
      const configure = this.onConfigure;
      this.onConfigure = function () { const result = configure?.apply(this, arguments); remember(textWidget.value || ""); watch(); show(); return result; };
      this.setSize([Math.max(this.size[0], 340), Math.max(this.size[1], 460)]);
      show();
    };
  },
});
