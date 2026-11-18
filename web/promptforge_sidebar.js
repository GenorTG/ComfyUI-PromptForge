import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAB_ID = "promptforge-browser";
const NSFW_KEY = "promptforge.browser.nsfw";
const FIELD = { padding: "4px 6px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", font: "12px sans-serif" };
const BUTTON = { ...FIELD, cursor: "pointer", whiteSpace: "nowrap" };
const KIND_LABEL = { all: "Everything", image: "Images", video: "Videos", audio: "Audio", other: "Other" };

const get = async (path) => {
  const response = await api.fetchApi(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status}`);
  return body;
};
const post = async (path, body) => {
  const response = await api.fetchApi(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status}`);
  return data;
};
const fileUrl = (root, path) => api.apiURL(`/promptforge/browser/file?root=${root}&path=${encodeURIComponent(path)}`);
const thumbUrl = (root, path, size = 256) => api.apiURL(`/promptforge/browser/thumb?root=${root}&path=${encodeURIComponent(path)}&size=${size}`);
const size = (bytes) => bytes > 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
const date = (time) => new Date(time * 1000).toLocaleString();

function el(tag, style = {}, props = {}) {
  const node = Object.assign(document.createElement(tag), props);
  Object.assign(node.style, style);
  return node;
}

function button(label, action) {
  const control = el("button", BUTTON, { textContent: label, type: "button" });
  control.addEventListener("click", action);
  return control;
}

function blur(element, meta, state) {
  element.style.filter = meta?.nsfw && !state.nsfw ? "blur(18px) saturate(.7)" : "none";
}

let viewerClose = null;

function preview(root, files, start) {
  viewerClose?.();
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10050, background: "rgba(0,0,0,.82)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "8px", padding: "16px", boxSizing: "border-box", overflow: "hidden" });
  const stage = el("div", { flex: "1", minHeight: "0", width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", overflow: "hidden" });
  const holder = el("div", { flex: "1", minWidth: "0", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" });
  const caption = el("div", { minHeight: "16px", maxWidth: "100%", color: "#ddd", font: "12px sans-serif", textAlign: "center", overflowWrap: "anywhere" });

  let index = Math.max(0, Math.min(files.length - 1, start));
  let media = null;

  const step = (delta) => {
    if (files.length < 2) return;
    index = (index + delta + files.length) % files.length;
    render();
  };
  const close = () => {
    media?.pause?.();
    backdrop.remove();
    document.removeEventListener("keydown", onKey, true);
    if (viewerClose === close) viewerClose = null;
  };
  const onKey = (event) => {
    if (event.key === "Escape") close();
    else if (event.key === "ArrowLeft") { event.preventDefault(); step(-1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); step(1); }
  };

  function render() {
    const file = files[index];
    media?.pause?.();
    holder.replaceChildren();
    if (file.kind === "video") {
      media = el("video", { maxWidth: "100%", maxHeight: "100%", display: "block" }, { src: fileUrl(root, file.path), controls: true, autoplay: true, loop: true });
    } else if (file.kind === "audio") {
      media = el("audio", { width: "min(560px, 80vw)", maxWidth: "100%" }, { src: fileUrl(root, file.path), controls: true, autoplay: true });
    } else if (file.kind === "image") {
      media = el("img", { maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" }, { src: fileUrl(root, file.path), alt: file.name });
    } else {
      media = el("div", { fontSize: "64px" }, { textContent: "📄" });
    }
    holder.append(media);
    caption.textContent = `${file.path} · ${size(file.size)} · ${date(file.mtime)}${file.meta?.nsfw ? " · NSFW" : ""} · ${index + 1} / ${files.length}`;
  }

  stage.append(holder);
  if (files.length > 1) {
    const previous = button("‹", () => step(-1));
    const next = button("›", () => step(1));
    for (const control of [previous, next]) Object.assign(control.style, { flex: "0 0 auto", fontSize: "20px", padding: "12px 10px" });
    stage.prepend(previous);
    stage.append(next);
  }
  const footer = el("div", { display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", maxWidth: "100%" });
  footer.append(caption, button("Close", close));
  backdrop.append(stage, footer);
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop || event.target === stage) close(); });
  document.addEventListener("keydown", onKey, true);
  document.body.append(backdrop);
  viewerClose = close;
  render();
}

async function tagEditor(root, file, state, refresh) {
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10051, background: "rgba(0,0,0,.6)", display: "flex", alignItems: "center", justifyContent: "center" });
  const card = el("div", { width: "min(520px, 92vw)", padding: "14px", display: "flex", flexDirection: "column", gap: "8px", background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "8px", font: "12px sans-serif" });
  const tags = el("input", FIELD, { value: (file.meta?.tags || []).join(", "), placeholder: "tags, comma separated" });
  const description = el("textarea", { ...FIELD, minHeight: "90px" }, { value: file.meta?.description || "", placeholder: "description" });
  const nsfw = el("input", {}, { type: "checkbox", checked: !!file.meta?.nsfw });
  const nsfwLabel = el("label", { display: "flex", gap: "6px", alignItems: "center" });
  nsfwLabel.append(nsfw, "NSFW");
  const row = el("div", { display: "flex", gap: "8px", justifyContent: "flex-end" });
  const status = el("div", { fontSize: "11px", color: "#c88" });
  const close = () => backdrop.remove();
  const save = button("Save", async () => {
    try {
      await post("/promptforge/browser/meta", { root, path: file.path, tags: tags.value, description: description.value, nsfw: nsfw.checked });
      close();
      refresh();
    } catch (error) { status.textContent = error.message; }
  });
  row.append(button("Close", close), save);
  card.append(el("div", { opacity: ".7", overflowWrap: "anywhere" }, { textContent: file.path }), tags, description, nsfwLabel, status, row);
  backdrop.append(card);
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(); });
  document.body.append(backdrop);
  tags.focus();
}

function analyse(root, file, status, refresh) {
  const wait = async () => {
    try {
      const job = await get("/promptforge/browser/analyse");
      if (job.running) { status.textContent = "Analysing…"; setTimeout(wait, 1200); return; }
      if (job.error) { status.textContent = job.error; return; }
      status.textContent = job.entry?.description || "Done";
      refresh();
    } catch (error) { status.textContent = error.message; }
  };
  post("/promptforge/browser/analyse", { root, path: file.path })
    .then(() => wait())
    .catch((error) => { status.textContent = error.message; });
}

export function createPanel() {
  const state = {
    outputs: { root: "output", path: "", search: "", kind: "all" },
    inputs: { root: "input", path: "", search: "", kind: "all", nsfw: localStorage.getItem(NSFW_KEY) === "1" },
    templates: { groups: [] },
    tab: "outputs",
  };
  const panel = el("div", { display: "flex", flexDirection: "column", height: "100%", minHeight: "0", color: "var(--input-text, #ddd)", font: "12px sans-serif", background: "var(--comfy-menu-bg, #202020)" });
  const tabs = el("div", { display: "flex", gap: "2px", padding: "8px 8px 0" });
  const body = el("div", { flex: "1", minHeight: "0", overflow: "auto", padding: "8px", display: "flex", flexDirection: "column", gap: "8px" });
  const views = {};

  const show = (name) => {
    viewerClose?.();
    state.tab = name;
    for (const [key, view] of Object.entries(views)) view.style.display = key === name ? "flex" : "none";
    for (const [key, control] of Object.entries(tabButtons)) control.dataset.active = key === name ? "1" : "";
    if (name === "templates") loadTemplates();
    else load(name);
  };
  const tabButtons = {};
  for (const [key, label] of [["outputs", "Outputs"], ["inputs", "Inputs"], ["templates", "Templates"]]) {
    const control = button(label, () => show(key));
    Object.assign(control.style, { borderBottomLeftRadius: "0", borderBottomRightRadius: "0", borderBottom: "none", padding: "6px 12px" });
    tabButtons[key] = control;
    tabs.append(control);
  }
  panel.append(tabs, body);

  async function load(name) {
    const view = views[name], statePart = state[name];
    if (!view) return;
    view.status.textContent = "Loading…";
    try {
      const data = await get(`/promptforge/browser/list?root=${statePart.root}&path=${encodeURIComponent(statePart.path)}&q=${encodeURIComponent(statePart.search)}&kind=${statePart.kind}`);
      renderList(name, view, data);
      view.status.textContent = "";
    } catch (error) {
      view.status.textContent = error.message;
    }
  }

  function renderList(name, view, data) {
    const statePart = state[name], inputs = name === "inputs";
    view.replaceChildren(view.status);
    const bar = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" });
    const search = el("input", { ...FIELD, flex: "1 1 150px" }, { type: "search", value: statePart.search, placeholder: inputs ? "Search names, tags, descriptions…" : "Search names…" });
    let timer;
    search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { statePart.search = search.value; load(name); }, 250); });
    const kind = el("select", FIELD);
    kind.append(...Object.entries(KIND_LABEL).map(([key, label]) => new Option(label, key)));
    kind.value = statePart.kind;
    kind.addEventListener("change", () => { statePart.kind = kind.value; load(name); });
    bar.append(search, kind);
    if (inputs) {
      const nsfwLabel = el("label", { display: "flex", gap: "4px", alignItems: "center", whiteSpace: "nowrap" }, { title: "Blur files marked NSFW" });
      const nsfw = el("input", {}, { type: "checkbox", checked: statePart.nsfw });
      nsfw.addEventListener("change", () => {
        statePart.nsfw = nsfw.checked;
        localStorage.setItem(NSFW_KEY, nsfw.checked ? "1" : "0");
        view.querySelectorAll("[data-nsfw]").forEach((thumb) => blur(thumb, { nsfw: thumb.dataset.nsfw === "1" }, statePart));
      });
      nsfwLabel.append(nsfw, "Show NSFW");
      bar.append(nsfwLabel);
    }
    view.append(bar);

    const crumbs = el("div", { display: "flex", gap: "4px", flexWrap: "wrap", alignItems: "center", opacity: ".9" });
    crumbs.append(button(statePart.root === "output" ? "output" : "input", () => { statePart.path = ""; load(name); }));
    for (const crumb of data.crumbs) {
      crumbs.append(el("span", {}, { textContent: "/" }));
      crumbs.append(button(crumb.name, () => { statePart.path = crumb.path; load(name); }));
    }
    if (data.crumbs.length) crumbs.append(button("⬆", () => { statePart.path = data.crumbs.slice(0, -1).map((part) => part.name).join("/"); load(name); }));
    view.append(crumbs);

    if (data.dirs.length) {
      const folders = el("div", { display: "flex", gap: "4px", flexWrap: "wrap" });
      for (const folder of data.dirs) folders.append(button(`📁 ${folder.name}${folder.count ? ` (${folder.count})` : ""}`, () => { statePart.path = folder.path; load(name); }));
      view.append(folders);
    }

    const meta = el("div", { opacity: ".6" }, { textContent: `${data.files.length} of ${data.total} files${data.truncated ? " (showing the newest; use search or a folder)" : ""}` });
    view.append(meta);

    if (!data.files.length) {
      view.append(el("div", { opacity: ".6", padding: "16px 0" }, { textContent: "Nothing here." }));
      return;
    }
    const grid = el("div", { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(128px, 1fr))", gap: "6px" });
    data.files.forEach((file, index) => grid.append(tile(name, statePart.root, file, index, data.files, view, load)));
    view.append(grid);
  }

  function tile(name, root, file, index, files, view, refresh) {
    const inputs = name === "inputs";
    const card = el("div", { display: "flex", flexDirection: "column", gap: "3px", minWidth: "0", border: "1px solid var(--border-color, #444)", borderRadius: "5px", padding: "4px", background: "var(--comfy-input-bg, #2a2a2a)" });
    const thumbBox = el("div", { position: "relative", height: "96px", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", borderRadius: "3px", background: "#111", cursor: "pointer" });
    let thumb;
    if (file.kind === "image") {
      thumb = el("img", { width: "100%", height: "100%", objectFit: "cover", display: "block" }, { src: thumbUrl(root, file.path), loading: "lazy", decoding: "async", alt: file.name });
      thumb.addEventListener("error", () => { if (!thumb.dataset.fallback) { thumb.dataset.fallback = "1"; thumb.src = fileUrl(root, file.path); } });
    } else if (file.kind === "video") thumb = el("video", { width: "100%", height: "100%", objectFit: "cover", display: "block" }, { src: `${fileUrl(root, file.path)}#t=0.1`, preload: "metadata", muted: true, playsInline: true });
    else thumb = el("div", { fontSize: "26px" }, { textContent: file.kind === "audio" ? "🎵" : "📄" });
    if (file.meta?.nsfw) thumb.dataset.nsfw = "1";
    blur(thumb, file.meta, state[name]);
    thumbBox.append(thumb);
    thumbBox.addEventListener("click", () => preview(root, files, index));
    if (file.kind === "video") thumbBox.append(el("span", { position: "absolute", inset: "0", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "26px", pointerEvents: "none", textShadow: "0 1px 4px #000" }, { textContent: "▶" }));
    if (file.meta?.nsfw) thumbBox.append(el("span", { position: "absolute", left: "3px", top: "3px", padding: "1px 4px", borderRadius: "3px", background: "#a33", color: "#fff", fontSize: "10px" }, { textContent: "NSFW" }));
    card.append(thumbBox, el("div", { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: "600" }, { textContent: file.name, title: file.path }));
    card.append(el("div", { opacity: ".6", fontSize: "10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, { textContent: `${size(file.size)} · ${date(file.mtime)}` }));
    if (file.meta?.tags?.length) card.append(el("div", { opacity: ".8", fontSize: "10px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#8bc" }, { textContent: file.meta.tags.join(", "), title: file.meta.tags.join(", ") }));
    if (inputs && (file.meta?.description || file.meta?.tags?.length)) {
      const note = el("div", { opacity: ".75", fontSize: "10px", maxHeight: "40px", overflow: "hidden" }, { textContent: file.meta.description || "", title: file.meta.description || "" });
      card.append(note);
    }
    if (inputs) {
      const status = el("div", { fontSize: "10px", opacity: ".75", overflowWrap: "anywhere" });
      const tools = el("div", { display: "flex", gap: "3px", flexWrap: "wrap" });
      tools.append(button("Rename", async () => {
        const nameValue = prompt("New file name", file.name);
        if (!nameValue) return;
        try { await post("/promptforge/browser/rename", { root, path: file.path, name: nameValue }); refresh(name); }
        catch (error) { status.textContent = error.message; }
      }));
      tools.append(button("Classify", () => tagEditor(root, file, state[name], () => refresh(name))));
      if (file.kind === "image") tools.append(button("Analyse", () => analyse(root, file, status, () => refresh(name))));
      card.append(tools, status);
    }
    return card;
  }

  function templateView() {
    const view = el("div", { display: "flex", flexDirection: "column", gap: "8px" });
    const note = el("div", { opacity: ".65" }, { textContent: "Dry starting graphs: pick a model file and enter your own prompt." });
    const status = el("div", { fontSize: "11px", color: "#8bc" });
    const groups = el("div", { display: "flex", flexDirection: "column", gap: "8px" });
    const pick = async (template) => {
      status.textContent = `Loading ${template.title}…`;
      try {
        if (typeof app.loadGraphData !== "function") throw new Error("app.loadGraphData is unavailable.");
        const data = await get(`/promptforge/browser/template?name=${encodeURIComponent(template.name)}`);
        await app.loadGraphData(data);
        status.textContent = `${template.title} loaded, save it as your own workflow.`;
      } catch (error) { status.textContent = error.message; }
    };
    view.append(note, status, groups);
    view.dataset.replace = "1";
    const load = async () => {
      try {
        const data = await get("/promptforge/browser/templates");
        groups.replaceChildren();
        if (!data.groups.length) groups.append(el("div", { opacity: ".6" }, { textContent: "No templates yet." }));
        for (const group of data.groups) {
          groups.append(el("div", { fontWeight: "700", marginTop: "6px", color: "#8bc" }, { textContent: group.family }));
          for (const template of group.templates) groups.append(button(template.title, () => pick(template)));
        }
      } catch (error) { status.textContent = error.message; }
    };
    view.load = load;
    return view;
  }

  views.outputs = el("div", { display: "flex", flexDirection: "column", gap: "8px" });
  views.inputs = el("div", { display: "flex", flexDirection: "column", gap: "8px" });
  views.templates = templateView();
  for (const view of [views.outputs, views.inputs]) {
    view.status = el("div", { fontSize: "11px", color: "#c88" });
    body.append(view);
  }
  body.append(views.templates);

  const loadTemplates = () => views.templates.load?.();
  const refresh = (name) => load(name);

  show("outputs");
  return panel;
}

let panel = null;
function mount(host) {
  if (!panel) panel = createPanel();
  host.replaceChildren(panel);
  host.style.height = "100%";
  host.style.minHeight = "0";
}

function rail() {
  const control = el("button", { ...BUTTON, position: "fixed", left: "8px", bottom: "80px", zIndex: 9998, writingMode: "vertical-rl", padding: "10px 6px", letterSpacing: "1px" }, { textContent: "Prompt Forge", type: "button", title: "Prompt Forge" });
  const drawer = el("div", { position: "fixed", top: "0", right: "0", bottom: "0", width: "min(460px, 94vw)", zIndex: 9999, display: "none", borderLeft: "1px solid var(--border-color, #444)", boxShadow: "-8px 0 24px rgba(0,0,0,.45)" });
  control.addEventListener("click", () => { const visible = drawer.style.display !== "none"; drawer.style.display = visible ? "none" : "block"; if (!visible) mount(drawer); });
  document.body.append(control, drawer);
}

function register() {
  const manager = app.extensionManager;
  const sidebar = manager?.sidebarTab;
  const tab = {
    id: TAB_ID,
    icon: "pi pi-folder-open",
    title: "Prompt Forge",
    tooltip: "Outputs, inputs and workflow templates",
    label: "Prompt Forge",
    type: "custom",
    render: mount,
    destroy: () => { panel?.remove(); },
  };
  if (typeof manager?.registerSidebarTab === "function") {
    manager.registerSidebarTab(tab);
    return;
  }
  if (typeof sidebar?.registerSidebarTab === "function") {
    sidebar.registerSidebarTab(tab);
    return;
  }
  rail();
}

app.registerExtension({
  name: "PromptForge.Sidebar",
  setup() {
    register();
  },
});
