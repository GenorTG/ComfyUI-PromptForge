import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_V2_IMAGE_MODEL";
const AUTO = "auto";
const FOLDERS = { anima: "Anima/", krea2: "Krea 2/", moss_image: "MOSS Image/" };
const FAMILY_ORDER = ["krea2", "moss_image", "anima"];

const CSS = `
.pfmodel-row{display:flex;gap:4px;align-items:center;width:100%;min-width:0;box-sizing:border-box;padding:0 2px}
.pfmodel-select{flex:1;min-width:0;box-sizing:border-box;background:var(--comfy-input-bg,#1b2026);color:var(--input-text,#e7eef5);border:1px solid #46545f;border-radius:4px;padding:3px 4px;font:12px system-ui,sans-serif}
.pfmodel-info{flex:0 0 auto;cursor:pointer;width:21px;height:21px;border-radius:50%;border:1px solid #4b5c69;background:#22303c;color:#bfe0f7;font-style:italic;font-weight:700;line-height:1;padding:0}
.pfmodel-info:disabled{opacity:.4;cursor:default}
.pfmodel-overlay{position:fixed;inset:0;z-index:10003;display:flex;align-items:center;justify-content:center;padding:8px;box-sizing:border-box;background:#000a}
.pfmodel-window{width:560px;height:520px;max-width:100%;max-height:100%;display:flex;flex-direction:column;box-sizing:border-box;padding:12px;background:#141a20;border:1px solid #4a5b69;border-radius:8px;box-shadow:0 18px 50px #000b;color:#e7eef5;font:12px system-ui,sans-serif}
.pfmodel-head{display:flex;gap:8px;align-items:flex-start;flex:0 0 auto}
.pfmodel-title{flex:1;min-width:0;font-size:14px;font-weight:700;overflow-wrap:anywhere}
.pfmodel-tabs{display:flex;gap:4px;margin-top:8px;border-bottom:1px solid #2b3742;padding-bottom:6px;flex:0 0 auto}
.pfmodel-tab{cursor:pointer;padding:3px 10px;border-radius:4px;border:1px solid #46545f;background:#202933;color:#dbe8f1}
.pfmodel-tab.pfmodel-active{background:#2e5f86;border-color:#4a86b8;color:#fff}
.pfmodel-body{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:8px 2px 0}
.pfmodel-body dl{display:grid;grid-template-columns:120px 1fr;gap:6px 10px;margin:0}
.pfmodel-body dt{opacity:.65}
.pfmodel-body dd{margin:0;overflow-wrap:anywhere}
.pfmodel-body a{color:#6cb6f5}
.pfmodel-body img{max-width:100%;max-height:300px;object-fit:contain;background:#0c1115;border-radius:5px}
.pfmodel-text{white-space:pre-wrap;overflow-wrap:anywhere}
.pfmodel-hint{opacity:.65;font-style:italic}
.pfmodel-words{display:flex;flex-wrap:wrap;gap:5px}
.pfmodel-word{padding:1px 8px;border-radius:4px;background:#26343f;color:#dfe9f2}
.pfmodel-shots{display:flex;flex-direction:column;gap:7px}
`;

function el(tag, props = {}) {
  return Object.assign(document.createElement(tag), props);
}
function button(label, action, title) {
  const control = el("button", { type: "button", textContent: label, title: title || label });
  control.className = "pfmodel-tab";
  control.addEventListener("click", action);
  return control;
}
function style() {
  if (document.getElementById("pfmodel-css")) return;
  document.head.append(el("style", { id: "pfmodel-css", textContent: CSS }));
}
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);

function change(node, action) {
  node.graph?.beforeChange();
  try { action(); } finally { node.graph?.afterChange(); }
  node.setDirtyCanvas?.(true, true);
}

const familyOf = (value) => {
  const name = String(value || "").replace(/^\[[a-z]+\]\s+/i, "");
  return Object.keys(FOLDERS).find((key) => name.startsWith(FOLDERS[key])) || "";
};
const nameOf = (value) => String(value || "").replace(/^\[[a-z]+\]\s+/i, "");
const stem = (path) => String(path || "").replace(/\\/g, "/").replace(/\.[^./]+$/, "").toLowerCase();

/* ---------------------------------------------------------------- info window */

function htmlText(html) {
  if (!html) return "";
  const box = el("div", { innerHTML: html });
  for (const br of box.querySelectorAll("br")) br.replaceWith("\n");
  for (const block of box.querySelectorAll("p,div,li,tr,h1,h2,h3,h4")) block.append("\n");
  return box.textContent.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

// Small standalone window: pass a title and a loader returning the record, so a
// LoRA info button can reuse it without knowing anything about checkpoints.
function createInfoWindow() {
  let screen = null;
  const escape = (event) => {
    if (event.key !== "Escape") return;
    event.stopPropagation();
    close();
  };
  function close() {
    if (!screen) return;
    screen.remove();
    screen = null;
    document.removeEventListener("keydown", escape, true);
  }

  function row(list, label, value) {
    if (value == null || value === "" || (Array.isArray(value) && !value.length)) return;
    list.append(el("dt", { textContent: label }));
    const cell = el("dd");
    cell.append(...(Array.isArray(value) ? value : [value instanceof Node ? value : String(value)]));
    list.append(cell);
  }

  function chips(list, label, words) {
    if (!words?.length) return;
    const box = el("div");
    box.className = "pfmodel-words";
    for (const word of words) box.append(el("span", { textContent: word, className: "pfmodel-word" }));
    row(list, label, [box]);
  }

  const tabs = {
    info(body, data) {
      if (!data.entry) {
        body.append(el("div", { className: "pfmodel-hint", textContent: "LoRA Manager has no record for this file yet." }));
        return;
      }
      const list = el("dl");
      const description = htmlText(data.description);
      if (description) row(list, "Description", el("div", { className: "pfmodel-text", textContent: description }));
      const versionNotes = htmlText(data.metadata?.description || data.metadata?.model?.description);
      if (versionNotes && versionNotes !== description) row(list, "Version notes", el("div", { className: "pfmodel-text", textContent: versionNotes }));
      if (data.metadata?.creator?.username) row(list, "Author", data.metadata.creator.username);
      if (data.metadata?.modelId) {
        const link = el("a", { href: `https://civitai.com/models/${data.metadata.modelId}${data.metadata.id ? `?modelVersionId=${data.metadata.id}` : ""}`, target: "_blank", rel: "noopener", textContent: "Open on Civitai" });
        row(list, "Source", [link]);
      }
      const tags = data.metadata?.model?.tags || data.entry.tags;
      chips(list, "Tags", tags);
      chips(list, "Trained words", data.metadata?.trainedWords);
      row(list, "File", nameOf(data.value));
      row(list, "Folder", data.entry.folder || "(root)");
      row(list, "Base model", data.entry.base_model);
      if (data.entry.file_size) row(list, "File size", `${(data.entry.file_size / 1024 ** 2).toFixed(0)} MB`);
      row(list, "Hash (sha256)", data.entry.sha256);
      if (data.entry.notes) row(list, "Notes", el("div", { className: "pfmodel-text", textContent: data.entry.notes }));
      if (!list.children.length) row(list, "Details", "No stored metadata.");
      body.append(list);
    },
    examples(body, data) {
      const previews = [data.preview, ...(data.metadata?.images || []).map((image) => image.url)].filter(Boolean);
      if (!previews.length) {
        body.append(el("div", { className: "pfmodel-hint", textContent: "No stored preview images." }));
        return;
      }
      const list = el("div");
      list.className = "pfmodel-shots";
      previews.forEach((url, index) => {
        const link = el("a", { href: url, target: "_blank", rel: "noopener" });
        link.append(el("img", { src: url, loading: "lazy", alt: index ? `Example ${index}` : "Preview" }));
        list.append(link);
      });
      body.append(list);
    },
    settings(body, data) {
      const names = [["sampler", "Sampler"], ["steps", "Steps"], ["cfgScale", "CFG"], ["Shift", "Shift"]];
      const found = new Map();
      for (const image of data.metadata?.images || []) {
        for (const [key, label] of names) {
          const value = image?.meta?.[key];
          if (value == null || value === "") continue;
          const seen = found.get(key) || { label, values: new Map() };
          seen.values.set(String(value), (seen.values.get(String(value)) || 0) + 1);
          found.set(key, seen);
        }
      }
      const list = el("dl");
      for (const { label, values } of found.values()) {
        const parts = [...values].sort((a, b) => b[1] - a[1]).map(([value, count]) => `${value} ×${count}`);
        row(list, label, parts.join(" · "));
      }
      if (data.metadata?.usage_tips && data.metadata.usage_tips !== "{}") row(list, "Usage tips", el("div", { className: "pfmodel-text", textContent: data.metadata.usage_tips }));
      if (!list.children.length) body.append(el("div", { className: "pfmodel-hint", textContent: "No recommended sampler, steps, CFG or shift in the stored images." }));
      else body.append(list);
    },
  };

  function show({ title, value, load }) {
    close();
    const backdrop = el("div");
    backdrop.className = "pfmodel-overlay";
    const win = el("div");
    win.className = "pfmodel-window";
    const head = el("div");
    head.className = "pfmodel-head";
    const heading = el("div", { textContent: title, title });
    heading.className = "pfmodel-title";
    const closeButton = button("Close", close);
    head.append(heading, closeButton);
    const bar = el("div");
    bar.className = "pfmodel-tabs";
    const body = el("div");
    body.className = "pfmodel-body";
    win.append(head, bar, body);
    backdrop.append(win);
    document.body.append(backdrop);
    screen = backdrop;

    backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(); });
    document.addEventListener("keydown", escape, true);

    let pending;
    const data = () => pending || (pending = load().then((result) => ({ value, ...result })));

    const open = async (key, tab) => {
      for (const other of bar.children) other.classList.toggle("pfmodel-active", other === tab);
      body.replaceChildren(el("div", { className: "pfmodel-hint", textContent: "Loading…" }));
      let payload;
      try { payload = await data(); }
      catch (error) { payload = { error }; }
      body.replaceChildren();
      if (payload.error) {
        body.append(el("div", { className: "pfmodel-hint", textContent: errorText(payload.error) }));
        return;
      }
      tabs[key](body, payload);
    };

    for (const [key, label] of [["info", "Info"], ["examples", "Examples"], ["settings", "Settings"]]) {
      const tab = button(label, () => open(key, tab));
      bar.append(tab);
      if (key === "info") open(key, tab);
    }
  }

  return { show, close };
}

const errorText = (error) => error?.message || String(error || "Unavailable.");

let managerReady = false;
const infoWindow = createInfoWindow();

/* ------------------------------------------------------------ LoRA Manager */

async function findEntry(relative) {
  const base = stem(relative).split("/").pop();
  const response = await api.fetchApi(`/api/lm/checkpoints/list?search=${encodeURIComponent(base)}&search_filename=true&search_modelname=false&page_size=100`);
  if (!response.ok) return null;
  const data = await response.json();
  const wanted = stem(relative);
  return (data.items || []).find((item) => stem(`${item.folder ? `${item.folder}/` : ""}${item.file_name}`).endsWith(wanted)) || null;
}

async function loadInfo(value) {
  const entry = await findEntry(nameOf(value));
  if (!entry) return { entry: null, metadata: {}, description: "" };
  const query = `file_path=${encodeURIComponent(entry.file_path)}`;
  const [metadata, description] = await Promise.all([
    api.fetchApi(`/api/lm/checkpoints/metadata?${query}`),
    api.fetchApi(`/api/lm/checkpoints/model-description?${query}`),
  ]);
  return {
    entry,
    metadata: metadata.ok ? (await metadata.json()).metadata || {} : {},
    description: description.ok ? (await description.json()).description || "" : "",
    preview: entry.preview_url || "",
  };
}

api.fetchApi("/api/lm/checkpoints/folders").then((response) => {
  if (!response.ok) return;
  managerReady = true;
  app.graph?._nodes?.forEach((entry) => entry.promptforgeModelRefresh?.(true));
}).catch(() => {});

/* ------------------------------------------------------------------ node UI */

function reorder(node, rowWidget, afterWidget) {
  node.widgets.splice(node.widgets.indexOf(rowWidget), 1);
  node.widgets.splice(node.widgets.indexOf(afterWidget) + 1, 0, rowWidget);
}

function buildUI(node) {
  if (node.promptforgeModelBound) return;
  node.promptforgeModelBound = true;
  style();

  const family = widget(node, "family");
  const model = widget(node, "model");
  if (!family || !model) return;
  if (!family.options.values.includes(AUTO)) family.options.values.unshift(AUTO);

  const registered = [...model.options.values];
  let all = null;
  model.hidden = true;
  model.computeSize = () => [0, -4];

  const row = el("div");
  row.className = "pfmodel-row";
  const select = el("select", { ariaLabel: "Model file", title: "Model file" });
  select.className = "pfmodel-select";
  const info = el("button", { type: "button", textContent: "i", title: "Model information" });
  info.className = "pfmodel-info";
  row.append(select, info);
  const rowWidget = node.addDOMWidget("model_picker", "promptforge_v2", row, { serialize: false, getMinHeight: () => 48 });
  rowWidget.serialize = false;
  reorder(node, rowWidget, model);

  function paint() {
    select.replaceChildren(...(model.options.values || []).map((value) => new Option(value, value)));
    select.value = model.value;
    info.style.display = managerReady ? "" : "none";
    info.disabled = !managerReady || !model.value || model.value.startsWith("(");
  }

  function refresh(keepValue = false) {
    if (!all) return;
    const familyKey = family.value;
    const scoped = familyKey === AUTO || !FOLDERS[familyKey] ? all : all.filter((value) => familyOf(value) === familyKey);
    if (!keepValue && familyKey !== AUTO && scoped.length && !scoped.includes(model.value)) {
      model.value = scoped[0];
      model.callback?.(model.value);
      node.setDirtyCanvas?.(true, true);
    }
    model.options.values = scoped.includes(model.value) || !model.value ? scoped : [model.value, ...scoped];
    paint();
  }

  select.addEventListener("change", () => {
    if (!select.value) return;
    change(node, () => { model.value = select.value; });
    model.callback?.(model.value);
    paint();
  });

  info.addEventListener("click", () => {
    if (info.disabled) return;
    infoWindow.show({
      title: nameOf(model.value).split("/").pop(),
      value: model.value,
      load: () => loadInfo(model.value),
    });
  });

  const familyCallback = family.callback;
  family.callback = function () {
    const result = familyCallback?.apply(this, arguments);
    refresh();
    return result;
  };
  const modelCallback = model.callback;
  model.callback = function () {
    const result = modelCallback?.apply(this, arguments);
    paint();
    return result;
  };

  // "auto" is a frontend-only choice: the backend rejects it, so resolve it to
  // the selected model's family just before the prompt is built and restore it after
  let queuedAuto = false;
  family.beforeQueued = () => {
    queuedAuto = family.value === AUTO;
    if (queuedAuto) family.value = familyOf(model.value) || FAMILY_ORDER[0];
  };
  family.afterQueued = () => {
    if (!queuedAuto) return;
    queuedAuto = false;
    family.value = AUTO;
    refresh(true);
    node.setDirtyCanvas?.(true, true);
  };

  const removed = node.onRemoved;
  node.onRemoved = function () {
    infoWindow.close();
    return removed?.apply(this, arguments);
  };

  node.promptforgeModelRefresh = refresh;
  paint();
  api.fetchApi(`/object_info/${NODE}`).then(async (response) => {
    if (!response.ok) throw new Error();
    const data = await response.json();
    const spec = data?.[NODE]?.input?.required?.model;
    all = (Array.isArray(spec?.[0]) ? spec[0] : spec?.[1]?.options) || registered;
  }).catch(() => { all = registered; }).finally(() => refresh(true));
}

app.registerExtension({
  name: "PromptForge.V2ImageModel",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      buildUI(this);
      return result;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configure?.apply(this, arguments);
      // the added picker row shifts positional widget values; named values
      // stay correct, so restore those after the base configure
      const named = info?.widgets_values_named;
      if (named && typeof named === "object") {
        for (const [name, value] of Object.entries(named)) {
          const target = this.widgets.find((entry) => entry.name === name);
          if (target) target.value = value;
        }
      }
      this.promptforgeModelRefresh?.(true);
      return result;
    };
  },
});
