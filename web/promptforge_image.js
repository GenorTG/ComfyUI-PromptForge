import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_LoadImage";

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

const BUTTON = { cursor: "pointer", padding: "3px 10px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" };
const thumb = (name, size) => api.apiURL(`/promptforge/image_thumb?name=${encodeURIComponent(name)}&size=${size}`);

async function chooseImage(current) {
  const { images } = await (await api.fetchApi("/promptforge/images")).json();
  return new Promise((resolve) => {
    let folder = "";
    let search = "";
    const done = (value) => { backdrop.remove(); document.removeEventListener("keydown", onKey, true); resolve(value); };
    const onKey = (event) => { if (event.key === "Escape") { event.stopPropagation(); done(null); } };
    document.addEventListener("keydown", onKey, true);

    const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10001, background: "rgba(0,0,0,.6)", display: "flex", alignItems: "center", justifyContent: "center" });
    const card = el("div", { width: "min(1100px, 94vw)", height: "min(760px, 90vh)", display: "flex", flexDirection: "column", gap: "8px", padding: "12px", boxSizing: "border-box", background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "8px", fontSize: "13px" });
    const bar = el("div", { display: "flex", gap: "8px", alignItems: "center" });
    const searchBox = el("input", { flex: "1", padding: "5px 8px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" }, { type: "search", placeholder: "Search by name or description..." });
    const folderBox = el("select", { maxWidth: "260px", padding: "4px", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "4px" });
    const upload = el("button", BUTTON, { textContent: "Upload…" });
    const close = el("button", BUTTON, { textContent: "Close" });
    const count = el("span", { opacity: "0.7", whiteSpace: "nowrap" });
    const grid = el("div", { flex: "1", overflowY: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: "8px", alignContent: "start", padding: "2px" });
    bar.append(searchBox, folderBox, count, upload, close);
    card.append(bar, grid);
    backdrop.append(card);
    document.body.append(backdrop);
    backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) done(null); });
    close.addEventListener("click", () => done(null));

    const folderOf = (name) => (name.includes("/") ? name.slice(0, name.lastIndexOf("/")) : "");
    const folders = () => [...new Set(images.map((image) => folderOf(image.name)))].sort();
    const fillFolders = () => {
      folderBox.replaceChildren(new Option("All folders", "*"), ...folders().map((name) => new Option(name || "(input root)", name)));
      folderBox.value = folder === "" && !folders().includes("") ? "*" : folder;
    };
    folder = "*";

    function render() {
      const words = search.toLowerCase().split(/\s+/).filter(Boolean);
      const shown = images
        .filter((image) => (folder === "*" || folderOf(image.name) === folder) && words.every((word) => (image.name + " " + image.description).toLowerCase().includes(word)))
        .sort((a, b) => b.mtime - a.mtime);
      count.textContent = `${shown.length} images`;
      grid.replaceChildren(...shown.map((image) => {
        const tile = el("div", { cursor: "pointer", display: "flex", flexDirection: "column", gap: "3px", padding: "4px", borderRadius: "5px", border: image.name === current ? "2px solid #6af" : "1px solid var(--border-color, #555)", position: "relative" });
        const picture = el("img", { width: "100%", aspectRatio: "1 / 1", objectFit: "contain", borderRadius: "3px", background: "#111" }, { loading: "lazy", src: thumb(image.name, 220) });
        const label = el("div", { fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, { textContent: image.name.split("/").pop() });
        tile.title = image.name + (image.description ? "\n\n" + image.description : "");
        tile.append(picture, label);
        if (image.described) tile.append(el("div", { position: "absolute", top: "8px", right: "8px", padding: "0 5px", borderRadius: "8px", background: "#26a", color: "#fff", fontSize: "10px" }, { textContent: "text" }));
        tile.addEventListener("click", () => done(image.name));
        return tile;
      }));
    }

    let timer;
    searchBox.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { search = searchBox.value; render(); }, 120); });
    folderBox.addEventListener("change", () => { folder = folderBox.value; render(); });
    upload.addEventListener("click", () => {
      const picker = el("input", {}, { type: "file", accept: "image/*" });
      picker.addEventListener("change", async () => {
        const file = picker.files[0];
        if (!file) return;
        const target = folder === "*" ? "" : folder;
        const body = new FormData();
        body.append("image", file);
        body.append("subfolder", target);
        body.append("type", "input");
        const reply = await api.fetchApi("/upload/image", { method: "POST", body });
        if (reply.status === 200) {
          const saved = await reply.json();
          done(`${saved.subfolder ? saved.subfolder + "/" : ""}${saved.name}`);
        }
      });
      picker.click();
    });
    fillFolders();
    render();
    searchBox.focus();
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
      const save = el("button", BUTTON, { textContent: "Save description", title: "Store the text below as this image's description" });
      buttons.append(choose, save);
      root.append(picture, buttons);
      frame.append(root);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());

      const show = () => {
        picture.style.visibility = imageWidget.value ? "visible" : "hidden";
        if (imageWidget.value) picture.src = thumb(imageWidget.value, 480);
      };
      const pick = async () => {
        const name = await chooseImage(imageWidget.value);
        if (!name) return;
        imageWidget.value = name;
        const stored = await (await api.fetchApi("/promptforge/image_description?name=" + encodeURIComponent(name))).json();
        textWidget.value = stored.description || "";
        show();
        node.setDirtyCanvas(true, true);
      };
      choose.addEventListener("click", pick);
      picture.addEventListener("click", pick);
      save.addEventListener("click", async () => {
        if (!imageWidget.value) return;
        const reply = await api.fetchApi("/promptforge/image_description", { method: "POST", body: JSON.stringify({ name: imageWidget.value, description: textWidget.value }) });
        const answer = await reply.json();
        save.textContent = answer.saved ? "Saved ✓" : answer.error ? "Failed" : "Cleared ✓";
        setTimeout(() => { save.textContent = "Save description"; }, 1500);
      });

      const widget = this.addDOMWidget("image_picker", "promptforge_image_picker", frame, { serialize: false, getMinHeight: () => 220 });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
      // keep the picker above the text fields
      this.widgets.splice(this.widgets.indexOf(widget), 1);
      this.widgets.splice(0, 0, widget);
      imageWidget.callback = show;
      const configure = this.onConfigure;
      this.onConfigure = function () { const result = configure?.apply(this, arguments); show(); return result; };
      this.setSize([Math.max(this.size[0], 340), Math.max(this.size[1], 460)]);
      show();
    };
  },
});
