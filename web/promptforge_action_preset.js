import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_V2_ACTION_PRESET";
const FAMILIES = [["image", "Image"], ["h3", "H3"]];
const FIELD = "width:100%;box-sizing:border-box;background:var(--comfy-input-bg,#0f1419);color:var(--input-text,#e7eef5);border:1px solid #3b4a56;border-radius:4px;padding:4px 6px;font:12px sans-serif";
const WIDTH = 520;
const HEIGHT = 480;

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const familyKey = (family) => family === "h3" ? "minimax" : "image";
const element = (tag, props = {}) => Object.assign(document.createElement(tag), props);

function button(label, action) {
  const control = element("button", { textContent: label, type: "button" });
  control.style.cssText = "border:1px solid #65717c;border-radius:5px;padding:4px 10px;background:#252b31;color:#eef5fa;cursor:pointer";
  control.addEventListener("click", action);
  return control;
}
function section(title) {
  const label = element("div", { textContent: title });
  label.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.03em;opacity:.7";
  return label;
}
function copyLabel(label) {
  if (!label) return "";
  const match = label.match(/^(.*) copy( \d+)?$/);
  return match ? `${match[1]} copy ${Number(match[2]?.trim() || 1) + 1}` : `${label} copy`;
}

function form(node) {
  const family = widget(node, "family");
  const label = widget(node, "label");
  const category = widget(node, "category");
  const people = widget(node, "people");
  const genders = widget(node, "genders");
  const aspect = widget(node, "aspect");
  const brief = widget(node, "brief");
  const penetration = widget(node, "penetration");
  const saving = widget(node, "save");
  const deleting = widget(node, "delete");

  const frame = element("div");
  frame.style.cssText = "position:relative;width:100%;height:calc(100% - 12px);min-width:0;overflow:auto;contain:layout paint";
  const root = element("div");
  root.style.cssText = "display:flex;flex-direction:column;gap:5px;padding:3px 14px 3px 3px;box-sizing:border-box;min-height:100%;color:var(--input-text);font:12px sans-serif";
  frame.append(root);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  frame.addEventListener("wheel", (event) => event.stopPropagation());
  node.addDOMWidget("action_preset", "promptforge_v2", frame, { serialize: false, getMinHeight: () => 300 });

  const sizes = new Map();
  const hide = (item, hidden) => {
    if (!item) return;
    if (!sizes.has(item)) sizes.set(item, item.computeSize);
    item.hidden = hidden;
    item.computeSize = hidden ? () => [0, -4] : (sizes.get(item) || (() => [0, -4]));
    if (item.inputEl) item.inputEl.style.display = hidden ? "none" : "";
  };
  for (const name of ["family", "label", "category", "people", "genders", "aspect", "brief", "penetration", "save", "delete"]) {
    hide(widget(node, name), true);
  }

  const status = element("div");
  status.style.cssText = "color:#9fc7e4;overflow-wrap:anywhere;min-height:15px";

  function field(name, control) {
    const item = widget(node, name);
    control.value = item?.value ?? "";
    control.addEventListener("input", () => {
      if (!item) return;
      item.value = control.value;
      item.callback?.(control.value);
    });
    return control;
  }
  const familySelect = element("select");
  familySelect.style.cssText = FIELD;
  for (const [value, title] of FAMILIES) familySelect.append(new Option(title, value));
  familySelect.value = family?.value || "image";
  familySelect.addEventListener("change", () => {
    if (family) { family.value = familySelect.value; family.callback?.(family.value); }
    paint();
  });
  const labelInput = field("label", element("input", { type: "text", placeholder: "Preset name, for example Portrait - seated" }));
  labelInput.style.cssText = FIELD;
  const categorySelect = element("select");
  categorySelect.style.cssText = FIELD;
  field("category", categorySelect);
  const peopleSelect = element("select");
  peopleSelect.style.cssText = FIELD;
  for (const count of [1, 2, 3]) peopleSelect.append(new Option(String(count), String(count)));
  field("people", peopleSelect);
  peopleSelect.addEventListener("change", () => {
    if (people) { people.value = Number(peopleSelect.value); people.callback?.(people.value); }
    gendersInput.value = [...Array(peopleSelect.value).keys()].map((index) => gendersInput.value.split(",")[index]?.trim() || (index ? "male" : "female")).join(",");
    gendersInput.dispatchEvent(new Event("input"));
  });
  const gendersInput = field("genders", element("input", { type: "text", placeholder: "female,male" }));
  gendersInput.style.cssText = FIELD;
  const briefArea = field("brief", element("textarea", { placeholder: "{A} and {B} are the people. Where the bodies are, what the hands do, where the camera is.", spellcheck: false }));
  briefArea.style.cssText = `${FIELD};min-height:110px;resize:vertical;line-height:1.35`;
  const aspectSelect = element("select");
  aspectSelect.style.cssText = FIELD;
  for (const value of aspect?.options?.values || ["16:9", "9:16"]) aspectSelect.append(new Option(value, value));
  field("aspect", aspectSelect);
  const penetrationSelect = element("select");
  penetrationSelect.style.cssText = FIELD;
  for (const value of penetration?.options?.values || ["auto", "none", "vaginal", "anal"]) penetrationSelect.append(new Option(value, value));
  field("penetration", penetrationSelect);

  const categoriesFor = (familyName) => (category?.options?.values || [])
    .filter((name) => familyName === "h3" ? !/^NSFW|^SFW/.test(name) : /^NSFW|^SFW/.test(name));
  function paintCategories() {
    const values = categoriesFor(familySelect.value);
    if (categorySelect.value && !values.includes(categorySelect.value)) values.unshift(categorySelect.value);
    categorySelect.replaceChildren(...values.map((value) => new Option(value, value)));
    categorySelect.value = category?.value || values[0] || "";
  }

  const entries = element("div");
  entries.style.cssText = "display:flex;flex-direction:column;gap:3px;max-height:190px;overflow:auto;padding-right:2px";
  function filterRows() {
    const word = search.value.toLowerCase();
    for (const row of entries.children) row.style.display = row.textContent.toLowerCase().includes(word) ? "" : "none";
  }
  function paintEntries() {
    const rows = node.promptforgeActionPreset?.catalog?.[familyKey(familySelect.value)] || {};
    const list = Object.entries(rows).map(([name, entry]) => {
      const row = button("", () => load(name, entry));
      row.setAttribute("aria-label", name);
      row.style.cssText = `display:flex;flex-direction:column;gap:2px;text-align:left;border:1px solid ${name === label?.value ? "#78c9ff" : "#333c44"};border-radius:5px;padding:4px 7px;background:${name === label?.value ? "#24445a" : "#1c2227"};color:#dce8ef`;
      const title = element("div", { textContent: name });
      title.style.cssText = "font-weight:600;color:#eef5fa";
      const meta = element("div", { textContent: [entry.nsfw ? "NSFW" : "", entry.category].filter(Boolean).join(" · ") });
      meta.style.cssText = `font-size:11px;opacity:.75${entry.nsfw ? ";color:#f0b6a0" : ""}`;
      row.append(title, meta);
      return row;
    });
    entries.replaceChildren(...(list.length ? list : [element("div", { textContent: "No presets in this family yet." })]));
    filterRows();
  }
  function paint() {
    // paint is also the resync path after a workflow load
    familySelect.value = family?.value || "image";
    labelInput.value = label?.value ?? "";
    briefArea.value = brief?.value ?? "";
    peopleSelect.value = String(people?.value ?? 1);
    gendersInput.value = genders?.value ?? "female";
    paintCategories();
    paintEntries();
    const h3 = familySelect.value === "h3";
    for (const name of ["aspect", "penetration"]) hide(widget(node, name), !h3);
  }

  function load(name, entry) {
    const set = (item, value) => { if (item) { item.value = value; item.callback?.(value); } };
    set(label, name);
    labelInput.value = name;
    set(category, entry.category);
    categorySelect.value = entry.category;
    const genders = entry.gender || ["female"];
    set(people, genders.length);
    peopleSelect.value = String(genders.length);
    set(genders, genders.join(","));
    gendersInput.value = genders.join(",");
    if (entry.brief) set(brief, entry.brief);
    briefArea.value = entry.brief || "";
    if (entry.aspect) { set(aspect, entry.aspect); aspectSelect.value = entry.aspect; }
    if (entry.penetration) { set(penetration, entry.penetration); penetrationSelect.value = entry.penetration; }
    status.textContent = `Loaded “${name}”.`;
    paintEntries();
  }

  async function store(action, tick) {
    const payload = {
      family: familySelect.value, label: labelInput.value.trim(), category: categorySelect.value,
      people: Number(peopleSelect.value), genders: gendersInput.value,
      aspect: aspectSelect.value, brief: briefArea.value, penetration: penetrationSelect.value,
    };
    if (!payload.label) { status.textContent = "Enter a label first."; return false; }
    const response = await api.fetchApi(`/promptforge/v2/actions/${action}`, { method: "POST", body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) { status.textContent = data.error || `Could not ${action} (${response.status})`; return false; }
    node.promptforgeActionPreset.catalog[familyKey(payload.family)] = data.actions;
    // one pending write only: delete must not be undone by a stale save tick
    for (const item of [saving, deleting]) {
      const wanted = item === tick;
      item.value = wanted;
      item.callback?.(wanted);
    }
    return true;
  }

  root.append(
    section("Family"), familySelect,
    section("Label"), labelInput,
    section("Category"), categorySelect,
    section("People"), peopleSelect,
    section("Genders"), gendersInput,
    section("Brief"), briefArea,
    section("Aspect"), aspectSelect,
    section("Penetration"), penetrationSelect,
    status,
  );
  const tools = element("div");
  tools.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  tools.append(
    button("Save", async () => {
      if (await store("save", saving)) status.textContent = `Saved “${labelInput.value.trim()}”.`;
      paint();
    }),
    button("Duplicate", async () => {
      const source = labelInput.value.trim();
      labelInput.value = copyLabel(source);
      labelInput.dispatchEvent(new Event("input"));
      if (await store("save", saving)) status.textContent = `Duplicated as “${labelInput.value}”.`;
      paint();
    }),
    button("Delete", async () => {
      if (!labelInput.value.trim()) { status.textContent = "Enter a label first."; return; }
      if (await store("delete", deleting)) status.textContent = `Deleted “${labelInput.value.trim()}”.`;
      paint();
    }),
  );
  root.prepend(tools);
  root.append(section("Existing presets"));
  const search = element("input", { type: "search", placeholder: "Filter" });
  search.style.cssText = FIELD;
  search.addEventListener("input", filterRows);
  root.append(search, entries);

  const previousResize = node.onResize;
  node.onResize = function (size) {
    size[0] = Math.max(size[0], WIDTH);
    size[1] = Math.max(size[1], HEIGHT - 220);
    return previousResize?.apply(this, arguments);
  };
  if (family) {
    const callback = family.callback;
    family.callback = function () {
      const result = callback?.apply(this, arguments);
      familySelect.value = family.value;
      paint();
      return result;
    };
  }
  node.promptforgeActionPreset = { catalog: {}, status, paint };
  paint();
  requestAnimationFrame(() => {
    if (!node.size || node.size[0] < WIDTH) node.setSize?.([WIDTH, Math.max(node.size?.[1] || 0, HEIGHT)]);
  });
}

async function loadCatalog(node) {
  try {
    const response = await api.fetchApi("/promptforge/v2/actions");
    if (!response.ok) throw new Error(`Preset list unavailable (${response.status})`);
    node.promptforgeActionPreset.catalog = await response.json();
    node.promptforgeActionPreset.paint?.();
  } catch (error) { node.promptforgeActionPreset.status.textContent = error.message; }
}

app.registerExtension({
  name: "PromptForge.V2ActionPreset",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const node = this;
      node.resizable = true;
      node.flags = { ...node.flags, resizable: true };
      form(node);
      loadCatalog(node);
      const executed = node.onExecuted;
      node.onExecuted = function () {
        const value = executed?.apply(this, arguments);
        for (const name of ["save", "delete"]) {
          const item = widget(node, name);
          if (item?.value) { item.value = false; item.callback?.(false); }
        }
        loadCatalog(node);
        return value;
      };
      return result;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments);
      const node = this;
      requestAnimationFrame(() => node.promptforgeActionPreset?.paint?.());
      return result;
    };
  },
});
