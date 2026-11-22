import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

class Element {
  constructor(tag) { this.tag = tag; this.style = {}; this.dataset = {}; this.children = []; this.events = {}; this.value = ""; this.textContent = ""; }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, callback) { this.events[event] = callback; }
  setAttribute(name, value) { this[name] = value; }
  dispatchEvent(event) { this.events[event.type]?.(event); return true; }
  get lastChild() { return this.children[this.children.length - 1]; }
}

const widget = (name, value, options = []) => ({ name, value, options: { values: options }, computeSize: () => [100, 20], callback() {} });
const fields = () => [
  widget("family", "image", ["image", "h3"]),
  widget("label", ""),
  widget("category", "SFW - solo pose", ["SFW - solo pose", "SFW - couple pose", "NSFW - solo", "Solo female", "Positions"]),
  widget("people", 1),
  widget("genders", "female"),
  widget("aspect", "16:9", ["16:9", "9:16"]),
  widget("brief", ""),
  widget("penetration", "auto", ["auto", "none", "vaginal", "anal"]),
  widget("save", false),
  widget("delete", false),
];

test("action preset form loads, saves, duplicates and deletes through the routes", async () => {
  const calls = [];
  const previous = { document: globalThis.document, app: globalThis.__pfApp, api: globalThis.__pfApi, Option: globalThis.Option, raf: globalThis.requestAnimationFrame };
  const extensions = [];
  globalThis.document = { createElement: (tag) => new Element(tag), head: new Element("head"), getElementById: () => null };
  globalThis.Option = class { constructor(text, value) { this.textContent = text; this.value = value; } };
  globalThis.__pfApp = { registerExtension(value) { extensions.push(value); } };
  globalThis.requestAnimationFrame = (callback) => callback();
  const catalog = {
    image: { "Solo - standing": { category: "SFW - solo pose", description: "She stands.", brief: "{A} stands.", gender: ["female"], aspect: "", nsfw: false } },
    minimax: { "Position - cowgirl": { category: "Positions", description: "{A} rides.", brief: "{A} rides {B}.", gender: ["female", "male"], aspect: "9:16", nsfw: true } },
  };
  globalThis.__pfApi = {
    async fetchApi(path, options = {}) {
      calls.push([path, options]);
      if (path === "/promptforge/v2/actions") return { ok: true, status: 200, async json() { return catalog; } };
      const body = JSON.parse(options.body);
      const result = body.family === "h3" ? { "Position - cowgirl": catalog.minimax["Position - cowgirl"], "Position - cowgirl copy": { ...catalog.minimax["Position - cowgirl"], brief: body.brief } } : catalog.image;
      return { ok: true, status: 200, async json() { return { actions: result }; } };
    },
  };
  try {
    let source = await readFile(new URL("../web/promptforge_action_preset.js", import.meta.url), "utf8");
    source = source.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__pfApp;")
      .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__pfApi;");
    await import(`data:text/javascript;base64,${Buffer.from(source + "\n// " + Math.random()).toString("base64")}`);
    class Node {
      constructor() { this.widgets = fields(); this.panels = []; this.size = [520, 480]; }
      addDOMWidget(name, type, frame) { this.panels.push({ name, frame }); }
      setSize(size) { this.size = size; }
      setDirtyCanvas() {}
    }
    extensions[0].beforeRegisterNodeDef(Node, { name: "PROMPTFORGE_V2_ACTION_PRESET" });
    const node = new Node();
    node.onNodeCreated();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.ok(node.widgets.every((item) => item.hidden));
    const flat = (element) => [element, ...(element.children || []).flatMap(flat)];
    const press = (label) => flat(node.panels[0].frame).find((entry) => (entry.textContent === label || entry["aria-label"] === label) && entry.events.click).events.click();
    const rows = () => flat(node.panels[0].frame).map((entry) => entry["aria-label"]);
    const labelInput = flat(node.panels[0].frame).find((entry) => entry.placeholder?.startsWith("Preset name"));
    const type = (value) => { labelInput.value = value; labelInput.dispatchEvent(new Event("input")); };
    assert.equal(rows().includes("Position - cowgirl"), false);
    const select = flat(node.panels[0].frame).find((entry) => entry.tag === "select");
    select.value = "h3";
    select.events.change();
    assert.equal(node.widgets[0].value, "h3");
    assert.equal(node.widgets[5].hidden, false);
    assert.ok(rows().includes("Position - cowgirl"));
    press("Position - cowgirl");
    assert.equal(node.widgets[1].value, "Position - cowgirl");
    assert.equal(node.widgets[3].value, 2);
    assert.equal(node.widgets[7].value, "auto");
    type("Position - cowgirl");
    press("Save");
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.ok(calls.some(([path, options]) => path.endsWith("/save") && JSON.parse(options.body).family === "h3"));
    press("Duplicate");
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.ok(calls.some(([path, options]) => path.endsWith("/save") && JSON.parse(options.body).label === "Position - cowgirl copy"));
    assert.equal(node.widgets[8].value, true);
    press("Delete");
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.ok(calls.some(([path]) => path.endsWith("/delete")));
    assert.equal(node.widgets[9].value, true);
  } finally {
    globalThis.document = previous.document;
    globalThis.__pfApp = previous.app;
    globalThis.__pfApi = previous.api;
    globalThis.Option = previous.Option;
    globalThis.requestAnimationFrame = previous.raf;
  }
});
