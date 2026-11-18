import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_V2_REFERENCES_MANAGER";
const LANES = [["image", "Pictures", ".png,.jpg,.jpeg,.webp"], ["video", "Videos", "video/*"], ["audio", "Voices", "audio/*"]];
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const element = (tag, props = {}) => Object.assign(document.createElement(tag), props);

function read(node) {
  const store = widget(node, "items");
  try {
    const value = JSON.parse(store?.value || "[]");
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

function preview(item, size) {
  const url = `/view?filename=${encodeURIComponent(item.file)}&type=input&preview=webp`;
  if (item.kind === "image") return element("img", { src: url, loading: "lazy" });
  if (item.kind === "video") {
    const video = element("video", { src: url, muted: true, preload: "metadata" });
    video.addEventListener("loadeddata", () => { video.currentTime = 0.1; }, { once: true });
    return video;
  }
  const label = element("div", { textContent: "♪" });
  label.style.cssText = `display:flex;align-items:center;justify-content:center;width:${size}px;height:${size}px;background:#0003;border-radius:4px;font-size:20px`;
  return label;
}

function build(node) {
  const root = element("div");
  root.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint";
  const list = element("div");
  list.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:6px;color:var(--input-text);font:12px sans-serif;padding:2px";
  root.append(list);
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("wheel", (event) => event.stopPropagation());
  node.addDOMWidget("reference_lanes", "promptforge_refs", root, { serialize: false, getMinHeight: () => 300 });

  const draw = () => {
    list.replaceChildren();
    const items = read(node);
    for (const [kind, lane] of LANES) {
      const group = element("div");
      group.style.cssText = "display:flex;flex-direction:column;gap:4px;border:1px solid var(--border-color,#0003);border-radius:6px;padding:5px";
      const head = element("div");
      head.style.cssText = "display:flex;align-items:center;justify-content:space-between;font-weight:600";
      const count = items.filter((item) => item.kind === kind).length;
      head.append(element("span", { textContent: `${lane} (${count})` }));
      const picker = element("input", { type: "file", accept: LANES.find(([k]) => k === kind)[2], multiple: true });
      picker.style.display = "none";
      const add = element("button", { textContent: "+ Add", type: "button" });
      add.addEventListener("click", () => picker.click());
      picker.addEventListener("change", async () => {
        for (const file of picker.files || []) {
          const path = await upload(file);
          const current = read(node);
          const index = current.filter((item) => item.kind === kind).length + 1;
          current.push({ kind, file: path, name: `${kind} ${index}`, description: "", enabled: true });
          write(node, current);
        }
        draw();
      });
      head.append(add, picker);
      group.append(head);

      const rows = items.map((item, index) => ({ item, index })).filter(({ item }) => item.kind === kind);
      for (const { item, index } of rows) {
        const row = element("div");
        row.style.cssText = "display:flex;gap:6px;align-items:flex-start;padding:3px;border-radius:4px;background:#0001";
        row.draggable = true;
        row.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", String(index)));
        row.addEventListener("dragover", (event) => event.preventDefault());
        row.addEventListener("drop", (event) => {
          const from = Number(event.dataTransfer.getData("text/plain"));
          if (Number.isNaN(from) || from === index) return;
          const current = read(node);
          const [moved] = current.splice(from, 1);
          current.splice(from < index ? index - 1 : index, 0, moved);
          write(node, current);
          draw();
        });
        row.append(preview(item, 54));

        const fields = element("div");
        fields.style.cssText = "display:flex;flex-direction:column;gap:3px;flex:1;min-width:0";
        const name = element("input", { value: item.name || "", placeholder: "name" });
        name.style.cssText = "font:12px sans-serif;padding:2px 4px";
        name.addEventListener("change", () => {
          const current = read(node);
          current[index].name = name.value;
          write(node, current);
        });
        const description = element("input", { value: item.description || "", placeholder: "what it shows (optional)" });
        description.style.cssText = "font:11px sans-serif;padding:2px 4px;opacity:.9";
        description.addEventListener("change", () => {
          const current = read(node);
          current[index].description = description.value;
          write(node, current);
        });
        const file = element("div", { textContent: item.file, title: item.file });
        file.style.cssText = "font:10px monospace;opacity:.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        fields.append(name, description, file);

        const tools = element("div");
        tools.style.cssText = "display:flex;flex-direction:column;gap:2px";
        const toggle = element("input", { type: "checkbox", title: "Enabled" });
        toggle.checked = item.enabled !== false;
        toggle.addEventListener("change", () => {
          const current = read(node);
          current[index].enabled = toggle.checked;
          write(node, current);
        });
        const up = element("button", { textContent: "↑", type: "button", title: "Move up" });
        up.addEventListener("click", () => {
          const current = read(node);
          if (index === 0) return;
          [current[index - 1], current[index]] = [current[index], current[index - 1]];
          write(node, current);
          draw();
        });
        const down = element("button", { textContent: "↓", type: "button", title: "Move down" });
        down.addEventListener("click", () => {
          const current = read(node);
          if (index >= current.length - 1) return;
          [current[index + 1], current[index]] = [current[index], current[index + 1]];
          write(node, current);
          draw();
        });
        const remove = element("button", { textContent: "✕", type: "button", title: "Remove" });
        remove.addEventListener("click", () => {
          const current = read(node);
          current.splice(index, 1);
          write(node, current);
          draw();
        });
        tools.append(toggle, up, down, remove);

        row.append(fields, tools);
        group.append(row);
      }
      list.append(group);
    }
    const hint = element("div", {
      textContent: "Drag a row to reorder. Order sets the <Picture N>, <Video N> and <Audio N> numbers the prompt uses.",
    });
    hint.style.cssText = "font:11px sans-serif;opacity:.6;padding:2px";
    list.append(hint);
  };

  const store = widget(node, "items");
  if (store) {
    store.computeSize = () => [0, -4];
    for (const type of ["input", "change"]) store[type]?.call?.(store);
  }
  draw();
  return draw;
}

app.registerExtension({
  name: "PromptForge.V2.References",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const draw = build(this);
      this.addEventListener?.("refresh", draw);
    };
  },
});
