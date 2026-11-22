import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createLibrary, thumb, nsfwShown, setNsfwShown, BLUR } from "./promptforge_library.js";
import { PREFIX, ensureRole, upstreamNodes, parsePresets, capturePreset, applyPreset } from "./promptforge_v2_presets.js";

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
function element(tag, props = {}) { return Object.assign(document.createElement(tag), props); }
function button(label, action) {
  const control = element("button", { textContent: label, type: "button" });
  control.addEventListener("click", action);
  return control;
}
function panel(node, name, minHeight = 100) {
  const frame = element("div");
  frame.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint";
  const content = element("div");
  content.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:5px;color:var(--input-text);font:12px sans-serif";
  frame.append(content);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  frame.addEventListener("wheel", (event) => event.stopPropagation());
  node.addDOMWidget(name, "promptforge_v2", frame, { serialize: false, getMinHeight: () => minHeight });
  return content;
}
function change(node, action) {
  node.graph?.beforeChange();
  try { action(); } finally { node.graph?.afterChange(); }
  node.setDirtyCanvas?.(true, true);
}
// Sampling Settings gained refine_denoise/refine_steps after denoise; arrays saved before that are one value short there.
function migrateSampling(node) {
  const values = node.widgets_values;
  if (!Array.isArray(values) || values.length !== 11) return;
  values.splice(7, 0, 0.25, 4);
  node.widgets_values = values;
}
function download(value) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const link = element("a", { href: url, download: "promptforge-v2-presets.json" });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function presetsUI(node) {
  const root = panel(node, "preset_controls", 210);
  const select = element("select", { ariaLabel: "Saved presets" });
  const status = element("pre");
  status.style.cssText = "white-space:pre-wrap;margin:0";
  const store = widget(node, "presets");
  const name = widget(node, "preset_name");
  const include = () => !!widget(node, "include_text")?.value;
  const read = () => parsePresets(store.value);
  const refresh = () => {
    try {
      select.replaceChildren(...Object.keys(read().presets).map((key) => new Option(key, key)));
      select.value = name.value;
    } catch (error) { status.textContent = error.message; }
  };
  select.onchange = () => change(node, () => { name.value = select.value; });
  const run = (action) => async () => { try { await action(); } catch (error) { status.textContent = error.message; } };
  const save = (data) => { store.value = JSON.stringify(data); refresh(); };
  const bar = element("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  bar.append(
    button("Capture", run(() => {
      const key = name.value.trim();
      if (!key) throw new Error("Enter a preset_name first.");
      const data = read();
      change(node, () => { data.presets = { ...data.presets, [key]: capturePreset(upstreamNodes(node), include()) }; save(data); });
      status.textContent = `Captured ${data.presets[key].nodes.length} upstream V2 nodes as ${key}.`;
    })),
    button("Apply", run(() => {
      const preset = read().presets[name.value];
      if (!preset) throw new Error("Choose a saved preset.");
      status.textContent = applyPreset(node.graph, upstreamNodes(node), preset, include()).join("\n") || "No settings to apply.";
    })),
    button("Delete", run(() => { const data = read(); change(node, () => { delete data.presets[name.value]; save(data); }); status.textContent = "Preset deleted."; })),
    button("Export", run(() => download(read()))),
    button("Import", () => file.click()),
  );
  const file = element("input", { type: "file", accept: ".json,application/json", hidden: true });
  file.onchange = run(async () => {
    if (!file.files[0]) return;
    const data = parsePresets(await file.files[0].text());
    change(node, () => save({ schema_version: 2, presets: { ...read().presets, ...data.presets } }));
    status.textContent = "Imported presets (matching names replaced).";
    file.value = "";
  });
  root.append(select, bar, file, status);
  refresh();
  return refresh;
}

function promptLibraryUI(node) {
  const root = panel(node, "prompt_library_controls", 190);
  const select = element("select", { ariaLabel: "Saved prompt entries" });
  const name = widget(node, "entry_name");
  const text = widget(node, "text");
  const entries = widget(node, "entries");
  const status = element("div");
  const read = () => {
    const data = JSON.parse(entries.value);
    if (!data || typeof data !== "object" || Array.isArray(data) || Object.values(data).some((value) => typeof value !== "string")) throw new Error("Entries must be a JSON dictionary of prompt strings.");
    return data;
  };
  const refresh = () => {
    try {
      select.replaceChildren(new Option("Choose a saved prompt", ""), ...Object.keys(read()).map((key) => new Option(key, key)));
      select.value = name.value;
    } catch (error) { status.textContent = error.message; }
  };
  const run = (action) => () => { try { action(); } catch (error) { status.textContent = error.message; } };
  const load = run(() => {
    const data = read();
    const key = select.value || name.value;
    if (!Object.hasOwn(data, key)) throw new Error("Choose a saved prompt.");
    if (node.inputs?.some((input) => (input.name === "text" || input.widget?.name === "text") && input.link != null)) throw new Error("Text is connected; disconnect it to load a prompt.");
    change(node, () => { name.value = key; text.value = data[key]; text.callback?.(text.value); });
    status.textContent = `Loaded ${key}.`;
  });
  select.onchange = load;
  const bar = element("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  bar.append(button("Save", run(() => {
    const key = name.value.trim();
    if (!key) throw new Error("Enter an entry_name first.");
    const data = { ...read(), [key]: text.value };
    change(node, () => { entries.value = JSON.stringify(data); name.value = key; });
    refresh();
    status.textContent = `Saved ${key} in this workflow.`;
  })), button("Load", load), button("Delete", run(() => {
    const data = read();
    const key = select.value || name.value;
    change(node, () => { delete data[key]; entries.value = JSON.stringify(data); });
    refresh();
    status.textContent = `Deleted ${key}.`;
  })));
  root.append(select, bar, status);
  refresh();
  return refresh;
}

function imagePickerUI(node) {
  const root = panel(node, "reference_file_selector", 260);
  const filename = widget(node, "filename"), description = widget(node, "description");
  const preview = element("div"); preview.style.cssText = "display:flex;gap:8px;min-height:0;flex:1";
  const image = element("img", { alt: "Selected reference image" });
  image.style.cssText = "width:58%;min-height:120px;object-fit:contain;background:#15191d;border-radius:5px;cursor:pointer";
  const metadata = element("textarea", { readOnly: true });
  metadata.style.cssText = "flex:1;min-width:0;resize:none;background:#20262b;color:#d9e2e8;border:1px solid #3e4951;border-radius:5px;padding:6px;font:11px sans-serif";
  preview.append(image, metadata);
  const status = element("div");
  const row = element("div"); row.style.cssText = "display:flex;gap:5px;flex-wrap:wrap";
  const choose = button("Choose / browse images…", () => openPicker());
  const paste = button("Paste clipboard image", async () => {
    try {
      const items = await navigator.clipboard.read();
      const item = items.find((entry) => entry.types.some((type) => type.startsWith("image/")));
      if (!item) throw new Error("Clipboard has no image. Copy an image first.");
      const type = item.types.find((value) => value.startsWith("image/"));
      const blob = await item.getType(type);
      await uploadImage(new File([blob], `clipboard-${Date.now()}.${type.split("/")[1].replace("jpeg", "jpg")}`, { type }));
    } catch (error) { status.textContent = error.message || "Clipboard read was blocked by the browser; use Ctrl+V while the picker is open."; }
  });
  const showNsfw = element("input", { type: "checkbox", checked: nsfwShown() });
  const nsfwLabel = element("label", { textContent: "Show NSFW", title: "NSFW images are blurred unless enabled" });
  nsfwLabel.prepend(showNsfw); nsfwLabel.style.cssText = "display:flex;align-items:center;gap:4px";
  const saveDescription = button("Save description", async () => {
    try {
      if (!filename.value || !description.value.trim()) throw new Error("Choose an image and enter its description first.");
      const reply = await api.fetchApi("/promptforge/image_description", { method: "POST", body: JSON.stringify({ name: filename.value, description: description.value }) });
      if (!reply.ok) throw new Error(`Save failed (${reply.status})`);
      status.textContent = "Description saved to the image library.";
    } catch (error) { status.textContent = error.message; }
  });
  const analyze = button("Analyze / tag", async () => {
    try {
      if (!filename.value) throw new Error("Choose an image first.");
      const models = await (await api.fetchApi("/promptforge/library/models")).json();
      const model = models.models?.[0];
      if (!model) throw new Error("No vision model available for tagging.");
      const reply = await api.fetchApi("/promptforge/library/tag", { method: "POST", body: JSON.stringify({ model, names: [filename.value], retag: true }) });
      if (!reply.ok) throw new Error(`Analyze failed (${reply.status})`);
      status.textContent = "Analyzing in the background; details refresh shortly.";
      setTimeout(refresh, 4000);
    } catch (error) { status.textContent = error.message; }
  });
  row.append(choose, paste, analyze, nsfwLabel, saveDescription);
  root.append(preview, row, status);
  let info = {};
  async function refresh() {
    if (!filename.value) { image.removeAttribute("src"); metadata.value = "Choose an image from the input library, upload one, or paste from clipboard."; return; }
    image.src = thumb(filename.value, 480);
    try {
      const [infoResponse, descriptionResponse] = await Promise.all([
        api.fetchApi(`/promptforge/library/info?name=${encodeURIComponent(filename.value)}`),
        api.fetchApi(`/promptforge/image_description?name=${encodeURIComponent(filename.value)}`),
      ]);
      info = infoResponse.ok ? await infoResponse.json() : {};
      const note = descriptionResponse.ok ? (await descriptionResponse.json()).description : "";
      if (description && !description.value) description.inputEl && (description.inputEl.placeholder = note ? `Saved description: ${note}` : "Image description for prompt enhancement (optional)");
      metadata.value = [info.caption, info.tags?.length ? `Tags: ${info.tags.join(", ")}` : "", note ? `Description: ${note}` : "", info.nsfw ? "NSFW image" : ""].filter(Boolean).join("\n\n") || "No image analysis yet. Open Tools in the picker to tag images with the local vision model.";
      image.style.filter = info.nsfw && !nsfwShown() ? BLUR : "none";
    } catch (error) { metadata.value = `Image loaded. Library details unavailable: ${error.message}`; }
  }
  showNsfw.onchange = () => { setNsfwShown(showNsfw.checked); refresh(); };
  async function uploadImage(file) {
    const body = new FormData(); body.append("image", file); body.append("type", "input");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed (${response.status})`);
    const saved = await response.json();
    change(node, () => { filename.value = `${saved.subfolder ? saved.subfolder + "/" : ""}${saved.name}`; filename.callback?.(filename.value); });
    status.textContent = "Image uploaded to ComfyUI input."; refresh();
  }
  function openPicker() {
    const backdrop = element("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:10001;background:#0009;display:flex;align-items:center;justify-content:center";
    const card = element("div"); card.style.cssText = "width:min(1180px,95vw);height:min(800px,92vh);display:flex;flex-direction:column;padding:12px;box-sizing:border-box;background:var(--comfy-menu-bg,#222);border:1px solid #555;border-radius:8px";
    backdrop.append(card);
    const toolbar = element("div"); toolbar.style.cssText = "display:flex;gap:6px;margin-bottom:7px";
    let library;
    const cleanup = () => { library?.destroy(); document.removeEventListener("paste", onPaste, true); backdrop.remove(); };
    const upload = button("Upload…", () => { const picker = element("input", { type: "file", accept: "image/*" }); picker.onchange = async () => { if (picker.files[0]) try { await uploadImage(picker.files[0]); cleanup(); } catch (error) { status.textContent = error.message; } }; picker.click(); });
    const pasteButton = button("Paste image", () => status.textContent = "Paste the copied image now (Ctrl+V).");
    const close = button("Close", cleanup); toolbar.append(upload, pasteButton, close); card.append(toolbar);
    library = createLibrary(card, { current: filename.value, onPick: (name) => { change(node, () => { filename.value = name; filename.callback?.(name); }); cleanup(); refresh(); } });
    const onPaste = async (event) => { const file = [...(event.clipboardData?.items || [])].find((item) => item.type.startsWith("image/"))?.getAsFile(); if (!file) return; event.preventDefault(); try { await uploadImage(file); cleanup(); } catch (error) { status.textContent = error.message; } };
    document.addEventListener("paste", onPaste, true); backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) cleanup(); });
    document.body.append(backdrop); backdrop.tabIndex = 0; backdrop.focus();
  }
  const callback = filename.callback;
  filename.callback = function () { const result = callback?.apply(this, arguments); refresh(); return result; };
  image.onclick = openPicker;
  refresh();
  return refresh;
}

async function uploadInto(node, filename, file) {
  const body = new FormData(); body.append("image", file); body.append("type", "input");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response.ok) throw new Error(`Upload failed (${response.status})`);
  const saved = await response.json();
  change(node, () => { filename.value = `${saved.subfolder ? saved.subfolder + "/" : ""}${saved.name}`; filename.callback?.(filename.value); });
  return filename.value;
}

function pasteInto(node, filename, status) {
  const apply = async (file) => {
    try { await uploadInto(node, filename, file); status.textContent = `Pasted ${filename.value}`; }
    catch (error) { status.textContent = error.message; }
  };
  const picker = element("input", { type: "file", accept: "image/*,video/*,audio/*" });
  picker.onchange = () => { if (picker.files[0]) apply(picker.files[0]); };
  return [
    button("Upload…", () => picker.click()),
    button("Paste", async () => {
      try {
        const items = await navigator.clipboard.read();
        const item = items.find((entry) => entry.types.some((type) => type.startsWith("image/") || type.startsWith("video/") || type.startsWith("audio/")));
        if (!item) throw new Error("Clipboard has no image, video or audio.");
        const type = item.types.find((value) => value.startsWith("image/") || value.startsWith("video/") || value.startsWith("audio/"));
        await apply(new File([await item.getType(type)], `clipboard-${Date.now()}.${type.split("/")[1].replace("jpeg", "jpg")}`, { type }));
      } catch (error) { status.textContent = error.message || "Clipboard read was blocked; use Ctrl+V."; }
    }),
  ];
}

function mediaUI(node, kind) {
  if (kind === "image") return imagePickerUI(node);
  const root = panel(node, "reference_file_selector", 190);
  const filename = widget(node, "filename");
  const search = element("input", { type: "search", placeholder: `Search input ${kind} files` });
  const select = element("select", { ariaLabel: `Reference ${kind} file` });
  select.style.cssText = "flex:1;min-width:0;max-width:100%";
  const status = element("div");
  const preview = element(kind === "video" ? "video" : "audio");
  preview.controls = kind !== "video";
  if (kind === "video") { preview.muted = true; preview.controls = true; }
  preview.style.cssText = "width:100%;max-height:140px;background:#15191d;border-radius:5px";
  let files = [];
  const show = () => {
    preview.removeAttribute("src");
    if (filename.value) preview.src = api.apiURL(`/view?filename=${encodeURIComponent(filename.value)}&type=input&subfolder=`);
  };
  const refresh = () => {
    const query = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const options = files.filter((name) => query.every((word) => name.toLowerCase().includes(word)));
    if (filename.value && !options.includes(filename.value)) options.unshift(filename.value);
    select.replaceChildren(new Option("Choose an input file", ""), ...options.map((name) => new Option(name, name)));
    select.value = filename.value;
  };
  select.onchange = () => {
    if (!select.value) return;
    change(node, () => { filename.value = select.value; filename.callback?.(select.value); show(); });
  };
  search.oninput = refresh;
  async function reload() {
    try {
      const response = await api.fetchApi(`/promptforge/v2/media?kind=${kind}`);
      if (!response.ok) throw new Error(`File list unavailable (${response.status}); enter filename above.`);
      const data = await response.json();
      if (!Array.isArray(data.files) || data.files.some((name) => typeof name !== "string")) throw new Error("Expected a media files array; enter filename above.");
      files = data.files;
      status.textContent = `${files.length} input ${kind} files`;
      refresh();
    } catch (error) { status.textContent = error.message; }
  }
  const bar = element("div");
  bar.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  bar.append(select, button("Refresh files", reload), ...pasteInto(node, filename, status));
  root.append(preview, search, bar, status);
  show();
  const callback = filename.callback;
  filename.callback = function () {
    const result = callback?.apply(this, arguments);
    refresh();
    show();
    return result;
  };
  refresh();
  show();
  reload();
  return refresh;
}

function encoderPresetUI(node) {
  const preset = widget(node, "encoder_preset");
  if (!preset) return;
  const callback = preset.callback;
  preset.callback = function (value) {
    const values = ENCODER_PRESETS[value];
    if (values) {
      change(node, () => {
        for (const [name, entry] of Object.entries(values)) {
          const target = widget(node, name);
          if (!target) continue;
          target.value = entry;
          target.callback?.(entry);
        }
      });
    }
    return callback?.apply(this, arguments);
  };
}

function savedPreviewUI(node) {
  const root = panel(node, "saved_preview", 240);
  const image = element("img", { alt: "Last saved image" });
  image.style.cssText = "width:100%;height:100%;object-fit:contain;background:#15191d;border-radius:5px";
  const label = element("div", { textContent: "Nothing saved yet" });
  label.style.cssText = "font-size:11px;opacity:.75;overflow-wrap:anywhere";
  root.append(image, label);
  node.promptforgeSaved = (items) => {
    const first = items?.[0];
    if (!first) return;
    image.src = thumb(`${first.subfolder ? first.subfolder + "/" : ""}${first.filename}`, 640);
    label.textContent = first.filename;
  };
}

function livePreviewUI(node) {
  const root = panel(node, "live_preview", 240);
  const image = element("img", { alt: "Live sampling preview" });
  const video = element("video", { loop: true, muted: true, autoplay: true, playsInline: true });
  for (const media of [image, video]) media.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none";
  const label = element("span", { textContent: "Live preview" });
  label.style.cssText = "position:absolute;top:4px;left:6px;padding:2px 5px;background:#222b;color:#fff;font-size:11px";
  root.append(image, video, label);
  node.promptforgeLive = ({ image: data, mime, label: text, step, total }) => {
    if (!["image/jpeg", "image/png", "image/webp", "video/mp4"].includes(mime)) return;
    const moving = mime === "video/mp4";
    label.textContent = `${text ? text + " · " : ""}step ${step}/${total}`;
    image.style.display = moving ? "none" : "";
    video.style.display = moving ? "" : "none";
    if (moving) { video.src = `data:${mime};base64,${data}`; video.play().catch(() => {}); }
    else { video.pause(); image.src = `data:${mime};base64,${data}`; }
  };
  node.promptforgeLiveHide = () => {
    video.pause();
    image.style.display = video.style.display = "none";
    video.removeAttribute("src");
    image.removeAttribute("src");
    label.textContent = "Live preview";
  };
  const removed = node.onRemoved;
  node.onRemoved = function () { this.promptforgeLiveHide(); return removed?.apply(this, arguments); };
}

const SIZES = {
  PROMPTFORGE_V2_LORA_STACK: [560, 620],
  PROMPTFORGE_V2_SINGLE_LORA: [400, 260],
  PROMPTFORGE_V2_PRESETS: [420, 380],
  PROMPTFORGE_V2_PROMPT_LIBRARY: [400, 320],
  PROMPTFORGE_V2_LOAD_IMAGE: [440, 460],
  PROMPTFORGE_V2_LOAD_VIDEO: [420, 300],
  PROMPTFORGE_V2_LOAD_AUDIO: [420, 240],
  PROMPTFORGE_V2_IMAGE_PROMPT: [460, 520],
  PROMPTFORGE_V2_H3_PROMPT: [460, 540],
  PROMPTFORGE_V2_IMAGE_PROMPT_WRITER: [420, 240],
  PROMPTFORGE_V2_H3_PROMPT_WRITER: [420, 260],
  PROMPTFORGE_V2_PROMPT_INSPECT: [440, 260],
  PROMPTFORGE_V2_RUN_INSPECT: [440, 280],
  PROMPTFORGE_V2_PREVIEW_SETTINGS: [420, 320],
  PROMPTFORGE_V2_SAVE: [440, 360],
  PROMPTFORGE_V2_H3_PREVIEW_SAVE: [440, 430],
};

function sizeTo(node, name) {
  const wanted = SIZES[name];
  if (!wanted) return;
  const own = node.computeSize();
  node.setSize([Math.max(wanted[0], own[0]), Math.max(wanted[1], own[1])]);
}

app.registerExtension({
  name: "PromptForge.V2",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name.startsWith(PREFIX)) return;
    const buildUi = (node) => {
      if (node.promptforgeV2Bound) return;
      node.promptforgeV2Bound = true;
      ensureRole(node);
      if (nodeData.name === "PROMPTFORGE_V2_PRESETS") node.promptforgeV2Refresh = presetsUI(node);
      if (nodeData.name === "PROMPTFORGE_V2_PREVIEW_SETTINGS") livePreviewUI(node);
      if (nodeData.name === "PROMPTFORGE_V2_PROMPT_LIBRARY") node.promptforgeV2Refresh = promptLibraryUI(node);
      const media = /^PROMPTFORGE_V2_LOAD_(IMAGE|VIDEO|AUDIO)$/.exec(nodeData.name);
      if (media) node.promptforgeV2Refresh = mediaUI(node, media[1].toLowerCase());
      if (nodeData.name === "PROMPTFORGE_V2_SAVE") savedPreviewUI(node);
      if (nodeData.name === "PROMPTFORGE_V2_H3_PREVIEW_SAVE") encoderPresetUI(node);
      if (/_(PROMPT_INSPECT|RUN_INSPECT|PROMPT_WRITER|SAVE_METADATA)$/.test(nodeData.name)) {
        const output = panel(node, "output_inspector");
        const text = element("pre");
        text.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;margin:0;user-select:text";
        output.append(text);
        node.promptforgeV2Output = text;
      }
    };
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      sizeTo(this, nodeData.name);
      buildUi(this);
      return result;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      if (nodeData.name === "PROMPTFORGE_V2_IMAGE_SAMPLING_SETTINGS") migrateSampling(this);
      const result = configure?.apply(this, arguments);
      // loaded nodes mount their DOM layer after configure; wait for it, then build the panel
      const node = this;
      let tries = 0;
      const mount = () => {
        if (node.domWidgets?.length || tries > 40) { buildUi(node); node.promptforgeV2Refresh?.(); return; }
        tries += 1;
        requestAnimationFrame(mount);
      };
      requestAnimationFrame(mount);
      return result;
    };
    const connections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = connections?.apply(this, arguments);
      this.promptforgeV2Refresh?.();
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = executed?.apply(this, arguments);
      const values = [message?.text, message?.report].flat().filter((value) => value != null);
      if (message?.images) this.promptforgeSaved?.(message.images);
      if (this.promptforgeV2Output) this.promptforgeV2Output.textContent = values.map((value) => typeof value === "string" ? value : JSON.stringify(value, null, 2)).join("\n");
      return result;
    };
  },
});
