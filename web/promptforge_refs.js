import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_V2_REFERENCES_MANAGER";
const LANES = [["image", "Pictures", ".png,.jpg,.jpeg,.webp", "picture"], ["video", "Videos", "video/*", "video"], ["audio", "Voices", "audio/*", "voice"]];
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const element = (tag, props = {}) => Object.assign(document.createElement(tag), props);

function read(node) {
  try {
    const value = JSON.parse(widget(node, "items")?.value || "[]");
    return Array.isArray(value) ? value.filter((item) => item && item.kind) : [];
  } catch {
    return [];
  }
}

function write(node, items) {
  const store = widget(node, "items");
  if (!store) return;
  store.value = JSON.stringify(items);
  node.graph?.beforeChange();
  node.graph?.afterChange();
  node.setDirtyCanvas?.(true, true);
}

async function upload(file) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("type", "input");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function thumb(item) {
  const url = `/view?filename=${encodeURIComponent(item.file)}&type=input&preview=webp`;
  const box = element("div");
  box.style.cssText = "position:relative;width:100%;aspect-ratio:4/3;border-radius:6px;overflow:hidden;background:#0004";
  if (item.kind === "image") {
    const img = element("img", { src: url, loading: "lazy" });
    img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block";
    box.append(img);
  } else if (item.kind === "video") {
    const video = element("video", { src: url, muted: true, preload: "metadata" });
    video.style.cssText = "width:100%;height:100%;object-fit:cover;display:block";
    video.addEventListener("loadeddata", () => { video.currentTime = 0.1; }, { once: true });
    box.append(video);
  } else {
    const wave = element("div", { textContent: "♪" });
    wave.style.cssText = "display:flex;width:100%;height:100%;align-items:center;justify-content:center;font-size:30px;color:#fff8";
    box.append(wave);
  }
  if (item.enabled === false) box.style.filter = "grayscale(1) brightness(.5)";
  return box;
}

function iconButton(glyph, title, action, danger = false) {
  const button = element("button", { textContent: glyph, type: "button", title });
  button.style.cssText = `padding:0;width:22px;height:22px;line-height:20px;font-size:13px;border-radius:5px;border:1px solid #fff3;background:#0005;color:${danger ? "#ff9a9a" : "#fff"};cursor:pointer`;
  button.addEventListener("click", (event) => { event.stopPropagation(); action(); });
  return button;
}

function card(node, item, index, redraw) {
  const box = element("div");
  box.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:5px;border-radius:8px;background:#0003;border:1px solid #fff2;min-width:0";
  box.draggable = true;
  box.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", String(index)));
  box.addEventListener("dragover", (event) => event.preventDefault());
  box.addEventListener("drop", (event) => {
    event.preventDefault();
    const from = Number(event.dataTransfer.getData("text/plain"));
    if (Number.isNaN(from) || from === index) return;
    const items = read(node);
    const [moved] = items.splice(from, 1);
    items.splice(from < index ? index - 1 : index, 0, moved);
    write(node, items);
    redraw();
  });

  const tools = element("div");
  tools.style.cssText = "display:flex;gap:3px;justify-content:flex-end";
  const toggle = iconButton(item.enabled === false ? "◻" : "◼", "Enabled", () => {
    const items = read(node);
    items[index].enabled = items[index].enabled === false;
    write(node, items);
    redraw();
  });
  tools.append(
    iconButton("↑", "Move up", () => {
      const items = read(node);
      if (index === 0) return;
      [items[index - 1], items[index]] = [items[index], items[index - 1]];
      write(node, items);
      redraw();
    }),
    iconButton("↓", "Move down", () => {
      const items = read(node);
      if (index >= items.length - 1) return;
      [items[index + 1], items[index]] = [items[index], items[index + 1]];
      write(node, items);
      redraw();
    }),
    toggle,
    iconButton("✕", "Remove", () => {
      const items = read(node);
      items.splice(index, 1);
      write(node, items);
      redraw();
    }, true),
  );

  const name = element("input", { value: item.name || "", placeholder: "name" });
  name.style.cssText = "font:600 12px sans-serif;padding:3px 5px;border-radius:4px;border:1px solid #fff2;background:#0004;color:inherit;min-width:0";
  name.addEventListener("change", () => {
    const items = read(node);
    items[index].name = name.value;
    write(node, items);
  });

  const description = element("input", { value: item.description || "", placeholder: "what it shows" });
  description.style.cssText = "font:11px sans-serif;padding:3px 5px;border-radius:4px;border:1px solid #fff2;background:#0004;color:inherit;min-width:0;opacity:.9";
  description.addEventListener("change", () => {
    const items = read(node);
    items[index].description = description.value;
    write(node, items);
  });

  const file = element("div", { textContent: item.file, title: item.file });
  file.style.cssText = "font:10px monospace;opacity:.55;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";

  box.append(thumb(item), tools, name, description, file);
  return box;
}

function build(node) {
  const root = element("div");
  root.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint;border-radius:8px;background:linear-gradient(180deg,#0006,#0002)";
  const list = element("div");
  list.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:8px;color:var(--input-text);font:12px sans-serif";
  root.append(list);
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("wheel", (event) => event.stopPropagation());
  root.addEventListener("dragover", (event) => event.preventDefault());
  root.addEventListener("drop", async (event) => {
    if (event.dataTransfer.getData("text/plain")) return;
    event.preventDefault();
    const items = read(node);
    for (const file of event.dataTransfer.files || []) {
      const kind = file.type.startsWith("video") ? "video" : file.type.startsWith("audio") ? "audio" : "image";
      const path = await upload(file);
      items.push({ kind, file: path, name: `${kind} ${items.filter((item) => item.kind === kind).length + 1}`, description: "", enabled: true });
    }
    write(node, items);
    redraw();
  });
  node.addDOMWidget("reference_lanes", "promptforge_refs", root, { serialize: false, getMinHeight: () => 560 });

  const redraw = () => {
    list.replaceChildren();
    const items = read(node);
    for (const [kind, lane, accept] of LANES) {
      const group = element("div");
      group.style.cssText = "display:flex;flex-direction:column;gap:6px;border:1px solid #fff2;border-radius:9px;padding:6px;background:#ffffff08";
      const head = element("div");
      head.style.cssText = "display:flex;align-items:center;justify-content:space-between;font:600 12px sans-serif;letter-spacing:.02em";
      head.append(element("span", { textContent: `${lane} · ${items.filter((item) => item.kind === kind).length}` }));
      const picker = element("input", { type: "file", accept, multiple: true });
      picker.style.display = "none";
      const add = iconButton("＋ " + lane.slice(0, -1), `Add ${lane}`, () => picker.click());
      add.style.width = "auto";
      add.style.padding = "0 8px";
      add.textContent = `+ ${lane.slice(0, -1)}`;
      picker.addEventListener("change", async () => {
        const current = read(node);
        for (const file of picker.files || []) {
          const path = await upload(file);
          current.push({ kind, file: path, name: `${kind} ${current.filter((item) => item.kind === kind).length + 1}`, description: "", enabled: true });
        }
        write(node, current);
        redraw();
      });
      head.append(add, picker);
      group.append(head);

      const grid = element("div");
      grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px";
      const entries = items.map((item, index) => ({ item, index })).filter(({ item }) => item.kind === kind);
      for (const entry of entries) grid.append(card(node, entry.item, entry.index, redraw));
      const hint = element("div", { textContent: "drop files here" });
      hint.style.cssText = "grid-column:1/-1;text-align:center;font:11px sans-serif;opacity:.45;border:1px dashed #fff3;border-radius:6px;padding:8px";
      group.append(grid, entries.length ? hint : hint);
      list.append(group);
    }
    const note = element("div", { textContent: "Order sets <Picture N>, <Video N> and <Audio N>. Drag a card to reorder." });
    note.style.cssText = "font:11px sans-serif;opacity:.55;text-align:center";
    list.append(note);
  };

  const store = widget(node, "items");
  if (store) store.computeSize = () => [0, -4];
  redraw();
  return redraw;
}

app.registerExtension({
  name: "PromptForge.V2.References",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      this.size = [660, 620];
      build(this);
    };
  },
});
