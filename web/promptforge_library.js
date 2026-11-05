import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const THEME_LABELS = { female: "Female", male: "Male", group: "Group", people: "People", nsfw: "NSFW", animals: "Animals", scenery: "Scenery", objects: "Objects", other: "Other" };
const BUTTON = { cursor: "pointer", padding: "3px 10px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" };
const FIELD = { padding: "4px 6px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" };

export const thumb = (name, size) => api.apiURL(`/promptforge/image_thumb?name=${encodeURIComponent(name)}&size=${size}`);
export const nsfwShown = () => localStorage.getItem("promptforge.showNsfw") === "1";
export const setNsfwShown = (value) => localStorage.setItem("promptforge.showNsfw", value ? "1" : "0");
export const BLUR = "blur(22px) saturate(.6)";

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

const post = async (path, body = {}) => (await api.fetchApi(path, { method: "POST", body: JSON.stringify(body) })).json();
const get = async (path) => (await api.fetchApi(path)).json();

async function editImage(image, onChange) {
  const info = await get("/promptforge/library/info?name=" + encodeURIComponent(image.name));
  const note = (await get("/promptforge/image_description?name=" + encodeURIComponent(image.name))).description;
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10002, background: "rgba(0,0,0,.6)", display: "flex", alignItems: "center", justifyContent: "center" });
  const card = el("div", { width: "min(760px, 94vw)", maxHeight: "90vh", overflowY: "auto", display: "flex", flexDirection: "column", gap: "8px", padding: "14px", boxSizing: "border-box", background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "8px", fontSize: "13px" });
  backdrop.dataset.pfNested = "1";
  const close = () => { backdrop.remove(); document.removeEventListener("keydown", onKey, true); };
  const onKey = (event) => { if (event.key === "Escape") { event.stopPropagation(); close(); } };
  document.addEventListener("keydown", onKey, true);
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(); });

  const picture = el("img", { maxWidth: "100%", maxHeight: "300px", objectFit: "contain", alignSelf: "center", borderRadius: "4px", filter: image.nsfw && !nsfwShown() ? BLUR : "none", cursor: "pointer" }, { src: thumb(image.name, 640), title: "Click to blur / show" });
  picture.addEventListener("click", () => { picture.style.filter = picture.style.filter === "none" ? BLUR : "none"; });
  const tagsBox = el("textarea", { ...FIELD, minHeight: "70px" }, { value: (info.tags || []).join(", "), placeholder: "keywords, comma separated" });
  const caption = el("input", FIELD, { value: info.caption || "", placeholder: "caption" });
  const theme = el("select", FIELD);
  theme.append(...Object.entries(THEME_LABELS).map(([key, label]) => new Option(label, key)));
  theme.value = info.theme || "other";
  const nsfw = el("input", {}, { type: "checkbox", checked: !!info.nsfw });
  const nsfwLabel = el("label", { display: "flex", gap: "6px", alignItems: "center" });
  nsfwLabel.append(nsfw, "NSFW");
  const description = el("textarea", { ...FIELD, minHeight: "60px" }, { value: note, placeholder: "description (output by the Load Image node)" });
  const save = el("button", BUTTON, { textContent: "Save" });
  const cancel = el("button", BUTTON, { textContent: "Close" });
  const row = el("div", { display: "flex", gap: "8px", justifyContent: "flex-end" });
  row.append(cancel, save);
  const meta = el("div", { opacity: ".6", fontSize: "11px" }, { textContent: `${image.name}${info.model ? " · tagged by " + info.model : " · not tagged yet"}` });
  const themeRow = el("div", { display: "flex", gap: "12px", alignItems: "center" });
  themeRow.append("Theme", theme, nsfwLabel);
  card.append(picture, meta, themeRow, tagsBox, caption, description, row);
  backdrop.append(card);
  document.body.append(backdrop);
  cancel.addEventListener("click", close);
  save.addEventListener("click", async () => {
    await post("/promptforge/library/edit", { name: image.name, tags: tagsBox.value, caption: caption.value, nsfw: nsfw.checked, theme: theme.value });
    await post("/promptforge/image_description", { name: image.name, description: description.value });
    close();
    onChange?.();
  });
}

/**
 * The image library: browse, search, tag and organise the input folder.
 * `onPick(name)` makes a tile click choose that image (the Load Image picker); without it, clicking opens the editor.
 */
export function createLibrary(container, { onPick = null, current = "", extraControls = [] } = {}) {
  const state = { images: [], search: "", folder: "*", theme: "*", select: false, selected: new Set(), models: [], tools: false };
  let timer = null;
  let polling = null;

  const root = el("div", { display: "flex", flexDirection: "column", gap: "8px", height: "100%", minHeight: "0", color: "var(--input-text, #ddd)", fontSize: "13px" });
  const bar = el("div", { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" });
  const searchBox = el("input", { ...FIELD, flex: "1 1 180px", minWidth: "120px" }, { type: "search", placeholder: "Search names, keywords, descriptions..." });
  const folderBox = el("select", { ...FIELD, maxWidth: "200px" });
  const themeBox = el("select", { ...FIELD, maxWidth: "150px" });
  const nsfwLabel = el("label", { display: "flex", gap: "4px", alignItems: "center", whiteSpace: "nowrap" }, { title: "NSFW images are blurred unless this is on" });
  const nsfwBox = el("input", {}, { type: "checkbox", checked: nsfwShown() });
  nsfwLabel.append(nsfwBox, "Show NSFW");
  const selectButton = el("button", BUTTON, { textContent: "Select", title: "Pick several images for tagging or organising" });
  const toolsButton = el("button", BUTTON, { textContent: "Tools ▾" });
  const count = el("span", { opacity: ".7", whiteSpace: "nowrap" });
  bar.append(searchBox, folderBox, themeBox, nsfwLabel, selectButton, toolsButton, count, ...extraControls);

  const tools = el("div", { display: "none", flexDirection: "column", gap: "6px", padding: "8px", border: "1px solid var(--border-color, #555)", borderRadius: "6px" });
  const modelRow = el("div", { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" });
  const modelBox = el("select", { ...FIELD, flex: "1", minWidth: "180px" });
  const tagUntagged = el("button", BUTTON, { textContent: "Tag untagged", title: "Tag every image that has no tags yet (or changed)" });
  const tagSelected = el("button", BUTTON, { textContent: "Tag selected" });
  const retagAll = el("button", BUTTON, { textContent: "Re-tag all" });
  modelRow.append("Vision model", modelBox, tagUntagged, tagSelected, retagAll);
  const folderRow = el("div", { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" });
  const subfolders = el("input", {}, { type: "checkbox" });
  const subLabel = el("label", { display: "flex", gap: "4px", alignItems: "center" });
  subLabel.append(subfolders, "include images already in subfolders");
  const organize = el("button", BUTTON, { textContent: "Organize into theme folders…", title: "Moves tagged images into female / male / group / nsfw / ... folders. Shows what will move first." });
  const organizeSelected = el("button", BUTTON, { textContent: "Organize selected…" });
  const undo = el("button", BUTTON, { textContent: "Undo last organize" });
  folderRow.append(organize, organizeSelected, undo, subLabel);
  const progress = el("div", { display: "none", flexDirection: "column", gap: "3px" });
  const progressBar = el("div", { height: "8px", borderRadius: "4px", background: "rgba(255,255,255,.12)", overflow: "hidden" });
  const progressFill = el("div", { height: "100%", width: "0%", background: "#4a90e2" });
  progressBar.append(progressFill);
  const progressText = el("div", { fontSize: "12px", opacity: ".85", whiteSpace: "pre-wrap" });
  const cancel = el("button", { ...BUTTON, alignSelf: "flex-start" }, { textContent: "Cancel job" });
  progress.append(progressBar, progressText, cancel);
  const note = el("div", { fontSize: "11px", opacity: ".6" }, { textContent: "Tagging runs a local vision model on this machine and unloads ComfyUI's models while it runs; avoid generating at the same time. Organizing moves files, so workflows that point at the old path need the image picked again (Undo restores them)." });
  tools.append(modelRow, folderRow, progress, note);

  const grid = el("div", { flex: "1", minHeight: "0", overflowY: "auto", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "8px", alignContent: "start", padding: "2px" });
  root.append(bar, tools, grid);
  container.append(root);

  const folderOf = (name) => (name.includes("/") ? name.slice(0, name.lastIndexOf("/")) : "");
  const matches = () => {
    const words = state.search.toLowerCase().split(/\s+/).filter(Boolean);
    return state.images
      .filter((image) => (state.folder === "*" || folderOf(image.name) === state.folder) && (state.theme === "*" || (state.theme === "untagged" ? !image.tagged : image.theme === state.theme)))
      .filter((image) => {
        const haystack = `${image.name} ${image.description} ${(image.tags || []).join(" ")} ${image.caption || ""} ${image.theme || ""}`.toLowerCase();
        return words.every((word) => haystack.includes(word));
      })
      .sort((a, b) => b.mtime - a.mtime);
  };

  function fillFilters() {
    const folders = [...new Set(state.images.map((image) => folderOf(image.name)))].sort();
    folderBox.replaceChildren(new Option("All folders", "*"), ...folders.map((name) => new Option(name || "(input root)", name)));
    folderBox.value = folders.includes(state.folder) ? state.folder : "*";
    state.folder = folderBox.value;
    const counts = {};
    state.images.forEach((image) => { counts[image.theme || "untagged"] = (counts[image.theme || "untagged"] || 0) + 1; });
    themeBox.replaceChildren(new Option("All themes", "*"), ...[...Object.keys(THEME_LABELS), "untagged"].filter((key) => counts[key]).map((key) => new Option(`${THEME_LABELS[key] || "Untagged"} (${counts[key]})`, key)));
    themeBox.value = state.theme === "*" || counts[state.theme] ? state.theme : "*";
    state.theme = themeBox.value;
  }

  function render() {
    const shown = matches();
    count.textContent = `${shown.length} images` + (state.select ? ` · ${state.selected.size} selected` : "");
    selectButton.style.background = state.select ? "#2a5d9a" : BUTTON.background;
    grid.replaceChildren(...shown.map((image) => {
      const hidden = image.nsfw && !nsfwBox.checked;
      const selected = state.selected.has(image.name);
      const tile = el("div", { cursor: "pointer", display: "flex", flexDirection: "column", gap: "3px", padding: "4px", borderRadius: "5px", position: "relative", border: selected ? "2px solid #4a90e2" : image.name === current ? "2px solid #6af" : "1px solid var(--border-color, #555)" });
      const picture = el("img", { width: "100%", aspectRatio: "1 / 1", objectFit: "contain", borderRadius: "3px", background: "#111", filter: hidden ? BLUR : "none" }, { loading: "lazy", src: thumb(image.name, 220) });
      const label = el("div", { fontSize: "11px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, { textContent: image.name.split("/").pop() });
      tile.title = [image.name, image.caption, (image.tags || []).join(", "), image.description].filter(Boolean).join("\n\n");
      tile.append(picture, label);
      const badges = el("div", { position: "absolute", top: "8px", left: "8px", display: "flex", gap: "3px" });
      if (image.theme) badges.append(el("span", { padding: "0 5px", borderRadius: "8px", background: image.nsfw ? "#a33" : "#264", color: "#fff", fontSize: "10px" }, { textContent: THEME_LABELS[image.theme] || image.theme }));
      if (image.described) badges.append(el("span", { padding: "0 5px", borderRadius: "8px", background: "#26a", color: "#fff", fontSize: "10px" }, { textContent: "text" }));
      tile.append(badges);
      const edit = el("button", { ...BUTTON, position: "absolute", top: "6px", right: "6px", padding: "0 6px", borderRadius: "50%", fontStyle: "italic", fontWeight: "bold" }, { textContent: "i", title: "Edit tags, theme and description" });
      edit.addEventListener("click", (event) => { event.stopPropagation(); editImage(image, reload); });
      tile.append(edit);
      tile.addEventListener("click", () => {
        if (state.select) {
          state.selected.has(image.name) ? state.selected.delete(image.name) : state.selected.add(image.name);
          render();
        } else if (onPick) onPick(image.name);
        else editImage(image, reload);
      });
      return tile;
    }));
  }

  async function reload() {
    state.images = (await get("/promptforge/images")).images;
    state.selected = new Set([...state.selected].filter((name) => state.images.some((image) => image.name === name)));
    fillFilters();
    render();
  }

  async function loadModels() {
    state.models = (await get("/promptforge/library/models")).models;
    const saved = localStorage.getItem("promptforge.visionModel");
    modelBox.replaceChildren(...(state.models.length ? state.models.map((model) => new Option(model.label, model.id)) : [new Option("No vision GGUF with an mmproj found", "")]));
    if (state.models.some((model) => model.id === saved)) modelBox.value = saved;
  }

  async function status() {
    const job = await get("/promptforge/library/status");
    progress.style.display = job.running || job.message ? "flex" : "none";
    const fraction = job.total ? job.done / job.total : 0;
    progressFill.style.width = `${Math.round(fraction * 100)}%`;
    progressText.textContent = `${job.message}${job.total ? ` · ${job.done}/${job.total}` : ""}${job.current ? ` · ${job.current}` : ""}${job.failed.length ? `\n${job.failed.length} failed: ${job.failed.slice(-3).join("; ")}` : ""}`;
    cancel.style.display = job.running ? "" : "none";
    for (const button of [tagUntagged, tagSelected, retagAll, organize, organizeSelected, undo]) button.disabled = job.running;
    if (job.running && !polling) polling = setInterval(status, 1000);
    if (!job.running && polling) { clearInterval(polling); polling = null; await reload(); }
    return job;
  }

  async function startTag(names, retag) {
    if (!modelBox.value) return alert("No vision model is available. A vision model is a GGUF with an mmproj file next to it, in a text_encoders or LLM folder.");
    localStorage.setItem("promptforge.visionModel", modelBox.value);
    const answer = await post("/promptforge/library/tag", { model: modelBox.value, names, retag });
    if (!answer.started) alert(answer.message || "Could not start.");
    await status();
  }

  async function startOrganize(names) {
    const body = { names, include_subfolders: subfolders.checked };
    const plan = await post("/promptforge/library/organize", { ...body, dry_run: true });
    if (!plan.total) return alert("Nothing to move. Tag the images first, or tick \"include images already in subfolders\".");
    const lines = Object.entries(plan.counts).map(([theme, number]) => `${number} → ${theme}/`).join("\n");
    if (!confirm(`Move ${plan.total} images into theme folders inside the input folder?\n\n${lines}\n\nWorkflows pointing at the old paths will need the image picked again. "Undo last organize" restores everything.`)) return;
    const done = await post("/promptforge/library/organize", { ...body, dry_run: false });
    progress.style.display = "flex";
    progressText.textContent = `Moved ${done.moved} images.`;
    await reload();
  }

  searchBox.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { state.search = searchBox.value; render(); }, 120); });
  folderBox.addEventListener("change", () => { state.folder = folderBox.value; render(); });
  themeBox.addEventListener("change", () => { state.theme = themeBox.value; render(); });
  nsfwBox.addEventListener("change", () => { setNsfwShown(nsfwBox.checked); render(); });
  selectButton.addEventListener("click", () => { state.select = !state.select; if (!state.select) state.selected.clear(); render(); });
  toolsButton.addEventListener("click", () => { state.tools = !state.tools; tools.style.display = state.tools ? "flex" : "none"; toolsButton.textContent = state.tools ? "Tools ▴" : "Tools ▾"; if (state.tools) { loadModels(); status(); } });
  tagUntagged.addEventListener("click", () => startTag("all", false));
  retagAll.addEventListener("click", () => confirm("Tag every image again? This can take a while.") && startTag("all", true));
  tagSelected.addEventListener("click", () => (state.selected.size ? startTag([...state.selected], true) : alert("Press Select, then click the images to tag.")));
  organize.addEventListener("click", () => startOrganize("all"));
  organizeSelected.addEventListener("click", () => (state.selected.size ? startOrganize([...state.selected]) : alert("Press Select, then click the images to organize.")));
  undo.addEventListener("click", async () => { const answer = await post("/promptforge/library/undo"); progress.style.display = "flex"; progressText.textContent = `Restored ${answer.restored} images.`; await reload(); });
  cancel.addEventListener("click", () => post("/promptforge/library/cancel"));

  reload();
  status();
  return { reload, focus: () => searchBox.focus(), destroy: () => { clearInterval(polling); root.remove(); } };
}

app.registerExtension({
  name: "PromptForge.ImageLibrary",
  async setup() {
    app.extensionManager?.registerSidebarTab?.({
      id: "promptforge-images",
      icon: "pi pi-images",
      title: "Images",
      tooltip: "PromptForge image library: tag, search and organize the input folder",
      type: "custom",
      render: (container) => {
        Object.assign(container.style, { height: "100%", padding: "8px", boxSizing: "border-box" });
        createLibrary(container);
      },
    });
  },
});
