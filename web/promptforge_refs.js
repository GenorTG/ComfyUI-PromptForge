import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createLibrary } from "./promptforge_library.js";

const NODE = "PROMPTFORGE_V2_REFERENCES_MANAGER";
const CARD = "application/x-promptforge-reference"; // the drag type of a card being reordered
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

function viewUrl(file) {
  const slash = file.lastIndexOf("/");
  return api.apiURL(`/view?filename=${encodeURIComponent(file.slice(slash + 1))}&subfolder=${encodeURIComponent(slash < 0 ? "" : file.slice(0, slash))}&type=input`);
}

async function waveform(canvas, url) {
  const g = canvas.getContext("2d");
  try {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const data = await context.decodeAudioData(await (await fetch(url)).arrayBuffer());
    context.close();
    const samples = data.getChannelData(0), step = Math.max(1, Math.floor(samples.length / canvas.width)), mid = canvas.height / 2;
    g.fillStyle = "#4fb3c8";
    const peaks = Array.from({ length: canvas.width }, (_, x) => {
      let peak = 0;
      for (let k = 0; k < step; k++) peak = Math.max(peak, Math.abs(samples[x * step + k] || 0));
      return peak;
    });
    const loudest = Math.max(...peaks, 1e-6);
    peaks.forEach((peak, x) => g.fillRect(x, mid - (peak / loudest) * mid, 1, Math.max(1, (peak / loudest) * canvas.height)));
  } catch { g.fillStyle = "#fff8"; g.font = "20px sans-serif"; g.fillText("♪ waveform unavailable", 16, canvas.height / 2); }
}

function thumb(item) {
  const url = viewUrl(item.file);
  const box = element("div");
  box.style.cssText = "position:relative;width:100%;aspect-ratio:16/10;border-radius:7px;overflow:hidden;background:#0d1013";
  if (item.kind === "image") {
    const img = element("img", { src: url, loading: "lazy" });
    img.style.cssText = "width:100%;height:100%;object-fit:contain;display:block";
    box.append(img);
  } else if (item.kind === "video") {
    const video = element("video", { src: url + "#t=0.1", muted: true, preload: "metadata", controls: true, playsInline: true });
    video.style.cssText = "width:100%;height:100%;object-fit:contain;display:block";
    box.append(video);
  } else {
    const wave = element("canvas", { width: 600, height: 150 });
    wave.style.cssText = "width:100%;height:calc(100% - 34px);display:block";
    const audio = element("audio", { src: url, controls: true, preload: "metadata" });
    audio.style.cssText = "width:100%;height:34px;display:block";
    box.append(wave, audio);
    waveform(wave, url);
  }
  if (item.enabled === false) box.style.filter = "grayscale(1) brightness(.5)";
  return box;
}

function iconButton(glyph, title, action, danger = false) {
  const button = element("button", { textContent: glyph, type: "button", title });
  button.style.cssText = `padding:0;width:30px;height:28px;line-height:26px;font-size:15px;border-radius:5px;border:1px solid #fff3;background:#0005;color:${danger ? "#ff9a9a" : "#fff"};cursor:pointer`;
  button.addEventListener("click", (event) => { event.stopPropagation(); action(); });
  return button;
}

const FOCUS = [["", "Whole picture"], ["face, hair, neck", "Face and hair"], ["face", "Face only"], ["person", "Whole person"], ["custom", "Own words…"]];
const FOCUS_NUMBERS = [["focus_padding", "Margin", 0.15, 0.05], ["focus_threshold", "Threshold", 0.5, 0.01], ["focus_grow", "Grow px", 0, 1], ["focus_feather", "Feather px", 2, 1]];
const FOCUS_NAMES = ["focus", "focus_crop", "focus_background", ...FOCUS_NUMBERS.map(([name]) => name)];
const CONTROL = "font:12px sans-serif;padding:4px 6px;border-radius:5px;border:1px solid #fff2;background:#0004;color:inherit;min-width:0";

async function previewFocus(node, item) {
  const backdrop = element("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:10002;background:#000c;display:flex;flex-direction:column;gap:10px;align-items:center;justify-content:center;color:#eee;font:14px sans-serif;cursor:pointer";
  const status = element("div", { textContent: "Masking… the first time loads SAM3." });
  backdrop.append(status);
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  let url = null;
  backdrop.addEventListener("click", () => { if (url) URL.revokeObjectURL(url); backdrop.remove(); });
  document.body.append(backdrop);
  const settings = Object.fromEntries(FOCUS_NAMES.filter((name) => item[name] !== undefined).map((name) => [name, item[name]]));
  try {
    const response = await api.fetchApi("/promptforge/v2/focus-preview", { method: "POST", body: JSON.stringify({ file: item.file, settings, checkpoint: widget(node, "sam3_checkpoint")?.value || "auto" }) });
    if (!response.ok) throw new Error(await response.text());
    url = URL.createObjectURL(await response.blob());
    const picture = element("img", { src: url, alt: "What the model receives" });
    picture.style.cssText = "max-width:92vw;max-height:85vh;object-fit:contain;border:1px solid #fff4";
    status.textContent = "This is what the model receives. Click to close.";
    backdrop.append(picture);
  } catch (error) { status.textContent = `${error.message} (click to close)`; }
}

function focusControls(node, item, index, redraw) {
  const box = element("div");
  box.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const set = (changes) => {
    const items = read(node);
    Object.assign(items[index], changes);
    write(node, items);
    redraw();
  };
  const labelled = (text, control) => {
    const label = element("label");
    label.style.cssText = "display:flex;gap:5px;align-items:center;justify-content:space-between;white-space:nowrap";
    label.append(element("span", { textContent: text }), control);
    return label;
  };
  const preset = element("select");
  preset.style.cssText = `${CONTROL};flex:1`;
  for (const [value, title] of FOCUS) preset.append(new Option(title, value));
  const current = item.focus || "";
  const own = item.focus_own || (current && !FOCUS.some(([value]) => value === current));
  preset.value = own ? "custom" : current;
  preset.addEventListener("change", () => set(preset.value === "custom" ? { focus_own: true } : { focus_own: false, focus: preset.value }));
  box.append(labelled("Focus", preset));
  if (own) {
    const words = element("input", { value: current, placeholder: "what to keep, e.g. torso, red jacket, dog" });
    words.style.cssText = CONTROL;
    words.addEventListener("change", () => set({ focus: words.value }));
    box.append(words);
  }
  if (!current) return box;
  const details = element("details", { open: Boolean(item.focus_open) });
  const summary = element("summary", { textContent: "Crop options" });
  summary.style.cssText = "cursor:pointer;opacity:.8";
  // remembered without a redraw, which would rebuild the card under the pointer
  summary.addEventListener("click", () => { const items = read(node); items[index].focus_open = !details.open; write(node, items); });
  const grid = element("div");
  grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;padding-top:4px";
  const crop = element("input", { type: "checkbox", checked: item.focus_crop !== false });
  crop.addEventListener("change", () => set({ focus_crop: crop.checked }));
  const background = element("select");
  background.style.cssText = CONTROL;
  for (const [value, title] of [["keep", "keep (crop only)"], ["grey", "cut out on grey"], ["white", "cut out on white"], ["black", "cut out on black"]]) background.append(new Option(title, value));
  background.value = item.focus_background || "keep";
  background.addEventListener("change", () => set({ focus_background: background.value }));
  grid.append(labelled("Crop", crop), labelled("Background", background));
  // the mask edge only shows when the subject is cut out
  const numbers = background.value === "keep" ? FOCUS_NUMBERS.filter(([name]) => name !== "focus_grow" && name !== "focus_feather") : FOCUS_NUMBERS;
  for (const [name, title, fallback, step] of numbers) {
    const input = element("input", { type: "number", step, value: item[name] ?? fallback });
    input.style.cssText = `${CONTROL};width:72px`;
    input.addEventListener("change", () => set({ [name]: input.value === "" ? fallback : Number(input.value) }));
    grid.append(labelled(title, input));
  }
  details.append(summary, grid);
  const preview = element("button", { type: "button", textContent: "Preview what the model receives" });
  preview.style.cssText = `${CONTROL};cursor:pointer`;
  preview.addEventListener("click", () => previewFocus(node, read(node)[index]));
  box.append(details, preview);
  return box;
}

function card(node, item, index, redraw) {
  const box = element("div");
  box.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:5px;border-radius:8px;background:#0003;border:1px solid #fff2;min-width:0";
  // a card drags only from its picture or its empty space: on text and controls the mouse selects as usual
  box.addEventListener("pointerdown", (event) => { box.draggable = !event.target.closest("input, select, summary, label, button, video, audio, [data-text]"); });
  box.addEventListener("dragstart", (event) => { if (box.draggable) event.dataTransfer.setData(CARD, String(index)); });
  box.addEventListener("dragend", () => { box.draggable = false; });
  box.addEventListener("pointerup", () => { box.draggable = false; });
  box.addEventListener("dragover", (event) => { if (event.dataTransfer.types.includes(CARD)) event.preventDefault(); });
  box.addEventListener("drop", (event) => {
    const data = event.dataTransfer.getData(CARD);
    if (!data) return;
    event.preventDefault();
    const from = Number(data);
    if (from === index) return;
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
  name.style.cssText = "font:600 13px sans-serif;padding:6px 8px;border-radius:5px;border:1px solid #fff2;background:#0004;color:inherit;min-width:0";
  name.addEventListener("change", () => {
    const items = read(node);
    items[index].name = name.value;
    write(node, items);
  });

  const description = element("input", { value: item.description || "", placeholder: "what it shows" });
  description.style.cssText = "font:12px sans-serif;padding:6px 8px;border-radius:5px;border:1px solid #fff2;background:#0004;color:inherit;min-width:0;opacity:.9";
  description.addEventListener("change", () => {
    const items = read(node);
    items[index].description = description.value;
    write(node, items);
  });

  const file = element("div", { textContent: item.file, title: item.file });
  file.dataset.text = "";
  file.style.cssText = "font:10px monospace;opacity:.55;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:text";

  const preview = thumb(item);
  const swap = iconButton("⇄", "Replace this file; name and description stay", () => replace(node, index, item.kind, redraw));
  swap.style.cssText += ";position:absolute;top:5px;right:5px;background:#000a;z-index:2";
  preview.append(swap);
  box.append(preview, tools, name, description, ...(item.kind === "image" ? [focusControls(node, item, index, redraw)] : []), file);
  return box;
}

function replace(node, index, kind, redraw) {
  const set = (path) => {
    const items = read(node);
    items[index].file = path;
    write(node, items);
    redraw();
  };
  const picker = element("input", { type: "file", accept: LANES.find(([name]) => name === kind)[2] });
  picker.addEventListener("change", async () => { if (picker.files[0]) set(await upload(picker.files[0])); });
  if (kind !== "image") return picker.click();
  const own = element("button", { textContent: "Upload a file…", type: "button" });
  const close = openLibrary("Click the picture to use instead.", (name) => { set(name); close(); }, [own]);
  own.addEventListener("click", () => { picker.addEventListener("change", close, { once: true }); picker.click(); });
}

function openLibrary(hint, onPick, extraControls = []) {
  const backdrop = element("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:10001;background:#0009;display:flex;align-items:center;justify-content:center";
  const card = element("div");
  card.style.cssText = "width:min(1180px,95vw);height:min(800px,92vh);display:flex;flex-direction:column;padding:12px;box-sizing:border-box;background:var(--comfy-menu-bg,#222);border:1px solid #555;border-radius:8px";
  const bar = element("div");
  bar.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:7px";
  const count = element("span", { textContent: hint });
  const close = element("button", { textContent: "Close", type: "button" });
  close.style.cssText = "padding:7px 14px;font:13px sans-serif";
  bar.append(close, count);
  card.append(bar);
  backdrop.append(card);
  let library;
  const cleanup = () => { library?.destroy(); backdrop.remove(); };
  close.onclick = cleanup;
  library = createLibrary(card, { current: "", extraControls, onPick: (name) => onPick(name, count) });
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) cleanup(); });
  document.body.append(backdrop);
  return cleanup;
}

function pickFromLibrary(node, redraw) {
  openLibrary("Click pictures to add them; Close when done.", (name, status) => {
    const items = read(node);
    items.push({ kind: "image", file: name, name: `image ${items.filter((item) => item.kind === "image").length + 1}`, description: "", enabled: true });
    write(node, items);
    redraw();
    status.textContent = `Added ${name}. Click more or Close.`;
  });
}

function build(node) {
  const root = element("div");
  root.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint;border-radius:8px;background:linear-gradient(180deg,#0006,#0002)";
  const list = element("div");
  list.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:8px;color:var(--input-text);font:12px sans-serif;user-select:text";
  root.append(list);
  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  root.addEventListener("wheel", (event) => event.stopPropagation());
  // the browser's own right-click menu (copy, paste) instead of the graph's
  root.addEventListener("contextmenu", (event) => event.stopPropagation());
  root.addEventListener("dragover", (event) => { if (event.dataTransfer.types.includes("Files")) event.preventDefault(); });
  root.addEventListener("drop", async (event) => {
    if (!event.dataTransfer.files.length) return;
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
  node.addDOMWidget("reference_lanes", "promptforge_refs", root, { serialize: false, getMinHeight: () => 760 });

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
      if (kind === "image") {
        const library = iconButton("Library…", "Add pictures from the input library", () => pickFromLibrary(node, redraw));
        library.style.width = "auto"; library.style.padding = "0 10px"; library.style.marginLeft = "6px";
        head.append(library);
      }
      group.append(head);

      const grid = element("div");
      grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px";
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
  if (store) {
    store.computeSize = () => [0, -4];
    store.hidden = true;
    for (const holder of [store.element, store.inputEl]) if (holder) holder.style.display = "none";
  }
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
      this.size = [980, 1180];
      this.promptforgeRefsRedraw = build(this);
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments);
      requestAnimationFrame(() => this.promptforgeRefsRedraw?.());
      return result;
    };
  },
});
