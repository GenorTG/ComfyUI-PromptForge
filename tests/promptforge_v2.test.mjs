import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../web/promptforge_v2_presets.js', import.meta.url), 'utf8');
const { PIPE, upstreamNodes, ensureRole, capturePreset, applyPreset, parsePresets, parseLoras } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const make = (id, type = 'PROMPTFORGE_V2_SETTINGS') => ({ id, type, properties: {}, widgets: [], inputs: [], outputs: [{ type: PIPE }] });
function graphFor(nodes, links = {}) {
  const calls = [];
  const graph = { links, getNodeById: (id) => nodes.find((node) => node.id === id), beforeChange: () => calls.push('before'), afterChange: () => calls.push('after'), calls };
  nodes.forEach((node) => { node.graph = graph; });
  return graph;
}
test('upstream traverses pipe reroutes, stops at outsiders, ignores other sockets and cycles', () => {
  const [start, a, reroute, outside, hidden] = [make(1), make(2), make(3, 'Reroute'), make(4, 'Other'), make(5)];
  start.inputs = [{ type: PIPE, link: 1 }, { type: 'MODEL', link: 5 }];
  reroute.inputs = [{ type: '*', link: 2 }];
  a.inputs = [{ type: PIPE, link: 3 }, { type: PIPE, link: 4 }];
  outside.inputs = [{ type: PIPE, link: 6 }];
  const links = Object.fromEntries([[1, 3], [2, 2], [3, 4], [4, 1], [5, 5], [6, 5]].map(([id, origin_id]) => [id, { origin_id, origin_slot: 0 }]));
  graphFor([start, a, reroute, outside, hidden], links);
  assert.deepEqual(upstreamNodes(start), [a]);
});
test('roles persist and duplicate clone roles are repaired at capture', () => {
  const a = make(1), b = make(2);
  assert.equal(ensureRole(a, () => 'a'), 'a');
  assert.equal(ensureRole(a, () => 'b'), 'a');
  b.properties.promptforge_role = 'a';
  assert.deepEqual(capturePreset([a, b], false, () => 'b').nodes.map((n) => n.role), ['a', 'b']);
});
test('role ids work when crypto.randomUUID is unavailable', () => {
  const original = globalThis.crypto;
  try {
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {} });
    const first = ensureRole(make(30));
    const second = ensureRole(make(31));
    assert.match(first, /^[0-9a-f-]{36}$/);
    assert.match(second, /^[0-9a-f-]{36}$/);
    assert.notEqual(first, second);
  } finally {
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: original });
  }
});
test('capture excludes prose and connections, stores LoRA arrays, supports text opt in', () => {
  const node = make(1);
  node.widgets = Object.entries({ steps: 20, prompt: 'secret', image: 'x.png', seed: 5, loras: '[{"name":"a","on":true}]', presets: '{}' }).map(([name, value]) => ({ name, value }));
  node.inputs = [{ name: 'seed', link: 0 }];
  const saved = capturePreset([node], false, () => 'role');
  assert.deepEqual(saved.nodes[0].widgets, { steps: 20, image: 'x.png', loras: [{ name: 'a', on: true }] });
  assert.equal(capturePreset([node], true).nodes[0].widgets.prompt, 'secret');
  assert.equal(parsePresets(JSON.stringify({ schema_version: 2, presets: { saved } })).presets.saved.nodes[0].role, 'role');
});
test('apply matches role and type, skips connections, uses one undo transaction', () => {
  const node = make(1), wrongType = make(2, 'PROMPTFORGE_V2_OTHER');
  node.properties.promptforge_role = wrongType.properties.promptforge_role = 'r';
  node.widgets = [{ name: 'steps', value: 1 }, { name: 'seed', value: 1 }, { name: 'loras', value: '[]' }];
  wrongType.widgets = [{ name: 'steps', value: 0 }];
  node.inputs = [{ widget: { name: 'seed' }, link: 0 }];
  const graph = graphFor([node, wrongType]);
  const report = applyPreset(graph, [node, wrongType], { nodes: [{ role: 'r', type: node.type, widgets: { steps: 30, seed: 9, loras: [{ name: 'l' }] } }] });
  assert.equal(node.widgets[0].value, 30);
  assert.equal(node.widgets[1].value, 1);
  assert.deepEqual(JSON.parse(node.widgets[2].value), [{ name: 'l' }]);
  assert.equal(wrongType.widgets[0].value, 0);
  assert.ok(report.some((line) => line.includes('connected input')));
  assert.deepEqual(graph.calls, ['before', 'after']);
});
test('invalid imports reject without mutation and ambiguous roles never apply', () => {
  assert.throws(() => parsePresets('{"schema_version":1,"presets":{}}'));
  assert.throws(() => parsePresets({ schema_version: 2, presets: { p: { nodes: [{ role: 'r', type: 'Other', widgets: {} }] } } }));
  assert.throws(() => parseLoras('{}'));
  const a = make(1), b = make(2);
  for (const node of [a, b]) { node.properties.promptforge_role = 'r'; node.widgets = [{ name: 'steps', value: 1 }]; }
  const graph = graphFor([a, b]);
  assert.match(applyPreset(graph, [a, b], { nodes: [{ role: 'r', type: a.type, widgets: { steps: 5 } }] })[0], /ambiguous/);
  assert.equal(a.widgets[0].value, 1);
});
test('undo closes when a widget callback fails', () => {
  const node = make(1);
  node.properties.promptforge_role = 'r';
  node.widgets = [{ name: 'steps', value: 1, callback() { throw new Error('callback'); } }];
  const graph = graphFor([node]);
  assert.throws(() => applyPreset(graph, [node], { nodes: [{ role: 'r', type: node.type, widgets: { steps: 2 } }] }), /callback/);
  assert.deepEqual(graph.calls, ['before', 'after']);
});
test('frontend lifecycle preserves size, displays execution output and never applies presets', async () => {
  class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.dataset = {}; this.children = []; this.value = ''; this.events = {}; }
    append(...children) { this.children.push(...children); }
    prepend(...children) { this.children.unshift(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(event, callback) { this.events[event] = callback; }
    removeAttribute(name) { delete this[name]; }
    setAttribute(name, value) { this[name] = value; }
    pause() {}
    play() { return Promise.resolve(); }
    remove() {}
    classList = { add() {}, remove() {}, toggle() {} };
    getBoundingClientRect() { return { top: 0, bottom: 10, left: 0, right: 10, width: 10, height: 10 }; }
  }
  const previous = { document: globalThis.document, app: globalThis.__pfApp, api: globalThis.__pfApi, Option: globalThis.Option };
  const extensions = [];
  globalThis.document = { createElement: (tag) => new Element(tag), head: new Element('head'), body: new Element('body'), baseURI: 'http://localhost/', getElementById: () => null, addEventListener() {}, removeEventListener() {} };
  globalThis.Option = class extends Element {
    constructor(text, value) { super('option'); this.textContent = text; this.value = value; }
  };
  globalThis.__pfApp = { registerExtension(value) { extensions.push(value); } };
  globalThis.requestAnimationFrame = (callback) => callback();
  const requests = [];
  globalThis.__pfApi = { apiURL: (path) => path, async fetchApi(path, options) {
    requests.push({ path, options });
    return { ok: true, async json() {
      return path.startsWith('/promptforge/v2/media') ? { files: ['references/frame.png', 'references/other.png'] }
        : path.startsWith('/promptforge/v2/loras') ? { loras: [{ name: 'local.safetensors', words: ['trigger'] }] } : { items: [] };
    } };
  } };
  try {
    let frontend = await readFile(new URL('../web/promptforge_v2.js', import.meta.url), 'utf8');
    frontend = frontend.replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.__pfApp;').replace('import { api } from "../../scripts/api.js";', 'const api = globalThis.__pfApi;').replace('"./promptforge_v2_presets.js"', JSON.stringify(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).replace('import { createLibrary, thumb, nsfwShown, setNsfwShown, BLUR } from "./promptforge_library.js";', 'const createLibrary=()=>({destroy(){}}), thumb=()=>"", nsfwShown=()=>false, setNsfwShown=()=>{}, BLUR="";');
    await import(`data:text/javascript;base64,${Buffer.from(frontend).toString('base64')}`);
    let promptFrontend = await readFile(new URL('../web/promptforge_prompt_v2.js', import.meta.url), 'utf8');
    promptFrontend = promptFrontend.replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.__pfApp;').replace('import { api } from "../../scripts/api.js";', 'const api = globalThis.__pfApi;');
    await import(`data:text/javascript;base64,${Buffer.from(promptFrontend + '\n// ' + Math.random()).toString('base64')}`);
    let loraFrontend = await readFile(new URL('../web/promptforge_lora_v2.js', import.meta.url), 'utf8');
    loraFrontend = loraFrontend.replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.__pfApp;').replace('import { api } from "../../scripts/api.js";', 'const api = globalThis.__pfApi;').replace('"./promptforge_v2_presets.js"', JSON.stringify(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`));
    await import(`data:text/javascript;base64,${Buffer.from(loraFrontend + '\n// ' + Math.random()).toString('base64')}`);
    const [extension, promptExtension, loraExtension] = extensions;
    class Node {
      constructor() { this.properties = {}; this.size = [723, 456]; this.widgets = [{ name: 'steps', value: 7 }]; }
      addDOMWidget() {}
      computeSize() { return this.size; }
      setSize(size) { this.size = size; }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'PROMPTFORGE_V2_RUN_INSPECT' });
    const node = new Node();
    node.onNodeCreated();
    node.onConfigure();
    node.onExecuted({ text: ['hello'], report: ['connected input skipped'] });
    assert.equal(node.promptforgeV2Output.textContent, 'hello\nconnected input skipped');
    assert.equal(node.widgets[0].value, 7);
    assert.deepEqual(node.size, [723, 456]);
    const flatten = (element) => [element, ...(element.children || []).flatMap(flatten)];
    const create = (name, widgets = []) => {
      class OtherNode {
        constructor() { this.type = name; this.widgets = widgets; this.properties = {}; this.panels = []; this.size = [500, 400]; }
        addDOMWidget(name, type, frame) { this.panels.push({ name, frame }); }
        computeSize() { return this.size; }
        setSize(size) { this.size = size; }
      }
      extension.beforeRegisterNodeDef(OtherNode, { name });
      if (promptExtension) promptExtension.beforeRegisterNodeDef(OtherNode, { name });
      if (loraExtension) loraExtension.beforeRegisterNodeDef(OtherNode, { name });
      const instance = new OtherNode();
      instance.graph = graphFor([]);
      instance.onNodeCreated();
      return instance;
    };
    for (const type of ['SETTINGS', 'IMAGE_SAMPLE', 'H3_FORMAT', 'WRITER_MODEL', 'IMAGE_LOAD']) {
      const ordinary = create(`PROMPTFORGE_V2_${type}`);
      assert.equal(ordinary.panels.length, 0, type);
    }
    for (const type of ['PROMPT_INSPECT', 'RUN_INSPECT', 'IMAGE_PROMPT_WRITER', 'H3_PROMPT_WRITER', 'SAVE_METADATA']) {
      assert.deepEqual(create(`PROMPTFORGE_V2_${type}`).panels.map(p => p.name), ['output_inspector']);
    }
    const entryName = { name: 'entry_name', value: 'Saved scene' };
    const promptText = { name: 'text', value: 'A quiet scene' };
    const entries = { name: 'entries', value: '{}' };
    const libraryNode = create('PROMPTFORGE_V2_PROMPT_LIBRARY', [entryName, promptText, entries]);
    const libraryElements = () => flatten(libraryNode.panels[0].frame);
    const libraryButton = (name) => libraryElements().find(e => e.textContent === name).events.click();
    libraryButton('Save');
    assert.equal(JSON.parse(entries.value)['Saved scene'], 'A quiet scene');
    promptText.value = 'Changed';
    const librarySelect = libraryElements().find(e => e.tag === 'select');
    librarySelect.value = 'Saved scene';
    librarySelect.onchange();
    assert.equal(promptText.value, 'A quiet scene');
    assert.deepEqual(libraryNode.graph.calls, ['before', 'after', 'before', 'after']);
    libraryNode.inputs = [{ name: 'text', link: 0 }];
    promptText.value = 'Connected';
    libraryButton('Load');
    assert.equal(promptText.value, 'Connected');
    libraryButton('Delete');
    assert.deepEqual(JSON.parse(entries.value), {});
    const preview = create('PROMPTFORGE_V2_PREVIEW_SETTINGS');
    const previewElements = flatten(preview.panels[0].frame);
    const imagePreview = previewElements.find(e => e.tag === 'img');
    const videoPreview = previewElements.find(e => e.tag === 'video');
    preview.promptforgeLive({ image: 'AAAA', mime: 'image/webp', step: 1, total: 2 });
    assert.equal(imagePreview.style.display, '');
    assert.equal(imagePreview.src, 'data:image/webp;base64,AAAA');
    preview.promptforgeLive({ image: 'BBBB', mime: 'video/mp4', step: 2, total: 2 });
    assert.equal(videoPreview.style.display, '');
    assert.equal(videoPreview.src, 'data:video/mp4;base64,BBBB');
    preview.promptforgeLiveHide();
    assert.equal(videoPreview.style.display, 'none');
    assert.equal(imagePreview.src, undefined);
    assert.deepEqual(preview.size, [500, 400]);
    const sourceWidget = { name: 'source', value: 'raw' };
    const actionWidget = { name: 'action', value: 'image: test', inputEl: { style: {} }, computeSize: () => [10, 20] };
    const promptBody = { name: 'text', value: '', inputEl: { style: {} }, computeSize: () => [10, 20] };
    const promptNegative = { name: 'negative', value: '', inputEl: { style: {} }, computeSize: () => [10, 20] };
    const promptNotes = { name: 'notes', value: '', inputEl: { style: {} }, computeSize: () => [10, 20] };
    const promptInput = { name: 'text_in', value: '', inputEl: { style: {} }, computeSize: () => [10, 20] };
    const prompt = create('PROMPTFORGE_V2_IMAGE_PROMPT', [sourceWidget, promptBody, promptInput, promptNegative, actionWidget, promptNotes]);
    for (const item of [sourceWidget, promptBody, promptInput, promptNegative, actionWidget, promptNotes]) assert.equal(item.hidden, true);
    assert.deepEqual(prompt.size, [520, 520]);
    const promptAreas = flatten(prompt.panels[0].frame).filter(e => e.tag === 'textarea');
    const areaFor = (placeholder) => promptAreas.find(e => e.placeholder === placeholder);
    const bodyArea = areaFor('Describe the scene');
    assert.match(bodyArea.style.cssText, /min-height:60px/);
    assert.match(areaFor('What to keep out').style.cssText, /min-height:40px/);
    assert.match(areaFor('Extra direction folded into the action brief').style.cssText, /min-height:40px/);
    sourceWidget.value = 'action';
    sourceWidget.callback();
    assert.equal(actionWidget.hidden, true);
    assert.deepEqual(actionWidget.computeSize(), [0, -4]);
    bodyArea.value = 'Written in the panel';
    bodyArea.events.input();
    assert.equal(promptBody.value, 'Written in the panel');
    assert.deepEqual(prompt.size, [520, 520]);
    const loras = { name: 'loras', value: '{"filter":"all","items":[{"name":"local.safetensors","on":true,"part":"audio"}]}' };
    const layers = { name: 'layer_controls', value: 'standard' };
    const stack = create('PROMPTFORGE_V2_LORA_STACK', [loras, layers]);
    const parts = () => flatten(stack.panels[0].frame).filter(e => e.title === 'Target layers (audiovisual models)');
    assert.equal(parts().length, 0);
    layers.value = 'minimax audiovisual';
    layers.callback();
    assert.equal(parts().length, 1);
    assert.equal(parts()[0].value, 'audio');
    layers.value = 'standard';
    layers.callback();
    assert.equal(parts().length, 0);
    assert.equal(JSON.parse(loras.value).items[0].part, 'audio');
    for (const kind of ['IMAGE', 'VIDEO', 'AUDIO']) {
      const filename = { name: 'filename', value: kind === 'IMAGE' ? '' : 'references/saved.png' };
      const media = create(`PROMPTFORGE_V2_LOAD_${kind}`, [filename, { name: 'description', value: '' }]);
      await new Promise(resolve => setTimeout(resolve, 0));
      assert.deepEqual(media.panels.map(p => p.name), ['reference_file_selector']);
      if (kind === 'IMAGE') {
        const controls = flatten(media.panels[0].frame).map(e => e.textContent);
        assert.ok(controls.includes('Choose / browse images…'));
        assert.ok(controls.includes('Paste clipboard image'));
        assert.ok(controls.includes('Save description'));
        continue;
      }
      const select = flatten(media.panels[0].frame).find(e => e.tag === 'select');
      assert.ok(select.children.some(e => e.value === 'references/frame.png'));
      assert.ok(select.children.some(e => e.value === 'references/saved.png'));
      select.value = 'references/frame.png';
      select.onchange();
      assert.equal(filename.value, 'references/frame.png');
      assert.deepEqual(media.graph.calls, ['before', 'after']);
      const preset = capturePreset([media], false);
      filename.value = 'references/other.png';
      applyPreset(media.graph, [media], preset);
      assert.equal(select.value, 'references/frame.png');
      media.onConfigure();
      assert.deepEqual(media.size, [500, 400]);
      assert.ok(requests.some(r => r.path === `/promptforge/v2/media?kind=${kind.toLowerCase()}`));
    }
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.ok(requests.some(r => r.path.startsWith('/promptforge/v2/loras')));
    flatten(stack.panels[0].frame).find(e => e.textContent === 'Refresh').events.click();
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.ok(requests.some(r => r.path === '/promptforge/v2/loras?refresh=1'));
    stack.promptforgeV2Refresh();
    const infoButton = flatten(stack.panels[0].frame).find(e => e.className === 'pflora-info-btn');
    assert.ok(infoButton, 'LoRA info button missing');
    infoButton.events.click();
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.ok(requests.some(r => r.path.startsWith('/api/lm/loras/list?')));
    assert.ok(requests.every(r => !r.options || !r.options.method || r.options.method === 'GET'));
    // let the panels' fetch chains settle before globals are restored
    await new Promise(resolve => setTimeout(resolve, 50));
  } finally {
    // The prompt test runs concurrently and swaps the same globals; its own finally restores them.
    globalThis.Option = previous.Option;
  }
});
test('LoRA backend envelope is captured as array and preserved on apply', () => {
  const node = make(1);
  node.widgets = [{ name: 'loras', value: '{"filter":"all","items":[{"name":"a","on":false,"part":"audio"}]}' }];
  const preset = capturePreset([node], false, () => 'r');
  assert.deepEqual(preset.nodes[0].widgets.loras, [{ name: 'a', on: false, part: 'audio' }]);
  preset.nodes[0].widgets.loras[0].on = true;
  applyPreset(graphFor([node]), [node], preset);
  const state = JSON.parse(node.widgets[0].value);
  assert.equal(state.filter, 'all');
  assert.equal(state.items[0].on, true);
});

for (const [kind, role] of [['IMAGE', 'first_frame'], ['IMAGE', 'last_frame'], ['IMAGE', 'reference'], ['VIDEO', 'reference'], ['VIDEO', 'continue'], ['AUDIO', 'reference']]) {
  test(`reference ${kind}/${role} filename and selectors round trip without prose`, () => {
    const node = make(1, `PROMPTFORGE_V2_LOAD_${kind}`);
    node.widgets = Object.entries({ filename: `references/local.${kind.toLowerCase()}`, role, index: 3, enabled: true, description: 'Private reference prose' }).map(([name, value]) => ({ name, value }));
    const preset = capturePreset([node], false, () => 'reference-role');
    assert.deepEqual(preset.nodes[0].widgets, { filename: `references/local.${kind.toLowerCase()}`, role, index: 3, enabled: true });
    for (const entry of node.widgets) if (entry.name !== 'description') entry.value = '';
    applyPreset(graphFor([node]), [node], preset);
    assert.equal(node.widgets[0].value, `references/local.${kind.toLowerCase()}`);
    assert.equal(node.widgets[1].value, role);
    assert.equal(node.widgets[2].value, 3);
    assert.equal(node.widgets[3].value, true);
    assert.equal(capturePreset([node], true).nodes[0].widgets.description, 'Private reference prose');
    node.widgets[4].value = 'Keep this prose';
    applyPreset(node.graph, [node], capturePreset([node], true), false);
    assert.equal(node.widgets[4].value, 'Keep this prose');
  });
}
test('selector controls stay included; payloads stay excluded even with text enabled', () => {
  const node = make(1);
  node.widgets = [
    { name: 'prompt_mode', type: 'combo', value: 'raw' },
    { name: 'text_encoder', value: 'encoder.safetensors' },
    { name: 'text', type: 'combo', value: 'choice' },
    { name: 'audio', value: 'references/a.wav' },
    { name: 'image', value: 'data:image/png;base64,AAAA' },
    { name: 'media', value: new Uint8Array([1, 2]) },
    { name: 'positive', value: 'prose' },
  ];
  assert.deepEqual(capturePreset([node], false, () => 'r').nodes[0].widgets, { prompt_mode: 'raw', text_encoder: 'encoder.safetensors', text: 'choice', audio: 'references/a.wav' });
  assert.equal(capturePreset([node], true).nodes[0].widgets.image, undefined);
});
test('owned writer and reference-bank sockets traverse V2 auxiliary chains and reroutes only', () => {
  const start = make(1), writer = make(2, 'PROMPTFORGE_V2_WRITER_MODEL'), bank = make(3, 'PROMPTFORGE_V2_H3_REFERENCE_BANK');
  const bankParent = make(4, 'PROMPTFORGE_V2_H3_REFERENCE_BANK'), outsider = make(5, 'ExternalWriter'), unrelated = make(6), reroute = make(7, 'Reroute');
  const writerSocket = 'PROMPTFORGE_V2_WRITER_MODEL', bankSocket = 'PROMPTFORGE_V2_H3_REFERENCE_BANK';
  start.inputs = [{ type: writerSocket, link: 1 }, { type: bankSocket, link: 2 }, { type: 'MODEL', link: 5 }, { type: writerSocket, link: 6 }];
  writer.outputs = [{ type: writerSocket }];
  bank.outputs = bankParent.outputs = [{ type: bankSocket }];
  bank.inputs = [{ type: bankSocket, link: 3 }];
  bankParent.inputs = [{ type: writerSocket, link: 4 }];
  reroute.inputs = [{ type: '*', link: 7 }];
  outsider.inputs = [{ type: writerSocket, link: 8 }];
  const origins = [writer, bank, bankParent, outsider, unrelated, reroute, writer, unrelated];
  const links = Object.fromEntries(origins.map((node, i) => [i + 1, { origin_id: node.id, origin_slot: 0 }]));
  graphFor([start, writer, bank, bankParent, outsider, unrelated, reroute], links);
  assert.deepEqual(upstreamNodes(start), [writer, bank, bankParent]);
  writer.widgets = [{ name: 'model', value: 'local.gguf' }];
  bank.widgets = [{ name: 'name', value: 'subject' }, { name: 'kind', value: 'image' }, { name: 'payload', value: new Uint8Array([1]) }];
  const preset = capturePreset(upstreamNodes(start), false, () => globalThis.crypto.randomUUID());
  assert.deepEqual(preset.nodes[1].widgets, { name: 'subject', kind: 'image' });
  assert.equal(preset.nodes[0].widgets.model, 'local.gguf');
});
test('IMAGE/MASK dependencies follow only allowed V2 media sources', () => {
  const start = make(1), mask = make(2, 'PROMPTFORGE_V2_MASK_PREPARE'), image = make(3, 'PROMPTFORGE_V2_LOAD_IMAGE');
  const outside = make(4, 'LoadImage'), unrelated = make(5, 'PROMPTFORGE_V2_OTHER');
  start.inputs = [{ type: 'MASK', link: 1 }, { type: 'IMAGE', link: 3 }, { type: 'IMAGE', link: 4 }];
  mask.outputs = [{ type: 'MASK' }];
  mask.inputs = [{ type: 'IMAGE', link: 2 }];
  image.outputs = outside.outputs = unrelated.outputs = [{ type: 'IMAGE' }];
  graphFor([start, mask, image, outside, unrelated], Object.fromEntries([mask, image, outside, unrelated].map((node, i) => [i + 1, { origin_id: node.id, origin_slot: 0 }])));
  assert.deepEqual(upstreamNodes(start), [mask, image]);
});
test('prompt node switches generation, gates audio and fills the action widget from the picker', async () => {
  class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.dataset = {}; this.children = []; this.events = {}; this.value = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(event, callback) { this.events[event] = callback; }
    setAttribute(name, value) { this[name] = value; }
  }
  const previous = { document: globalThis.document, app: globalThis.__pfApp, api: globalThis.__pfApi, Option: globalThis.Option, raf: globalThis.requestAnimationFrame };
  const extensions = [];
  globalThis.document = { createElement: (tag) => new Element(tag), head: new Element('head'), body: new Element('body'), baseURI: 'http://localhost/', getElementById: () => null, addEventListener() {}, removeEventListener() {} };
  globalThis.Option = class extends Element {
    constructor(text, value) { super('option'); this.textContent = text; this.value = value; }
  };
  globalThis.__pfApp = { registerExtension(value) { extensions.push(value); } };
  const catalog = {
    image: { 'Solo - standing': { category: 'Solo', description: 'She stands facing the camera.', gender: ['female'], aspect: '' } },
    minimax: { 'Oral - blowjob, kneeling': { category: 'Oral', description: '{A} kneels in front of {B}.', gender: ['female', 'male'], aspect: '9:16' } },
  };
  globalThis.__pfApi = { async fetchApi(path) { return { ok: true, status: 200, async json() { return path.startsWith('/promptforge/v2/actions') ? catalog : {}; } }; } };
  globalThis.requestAnimationFrame = (callback) => callback();
  try {
    let frontend = await readFile(new URL('../web/promptforge_prompt_v2.js', import.meta.url), 'utf8');
    frontend = frontend.replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.__pfApp;').replace('import { api } from "../../scripts/api.js";', 'const api = globalThis.__pfApi;');
    await import(`data:text/javascript;base64,${Buffer.from(frontend + '\n// ' + Math.random()).toString('base64')}`);
    const fields = ['source', 'generation', 'action', 'text', 'text_in', 'negative', 'subject_1', 'camera', 'notes', 'soundscape', 'music'].map((name) => ({ name, value: name === 'generation' ? 'minimax' : name === 'source' ? 'raw' : '' }));
    const widget = (name) => fields.find((item) => item.name === name);
    class Node {
      constructor() { this.widgets = fields; this.panels = []; this.size = [460, 540]; }
      addDOMWidget(name, type, frame) { this.panels.push({ name, frame }); }
      setDirtyCanvas() {}
    }
    extensions[0].beforeRegisterNodeDef(Node, { name: 'PROMPTFORGE_V2_H3_PROMPT' });
    const node = new Node();
    node.graph = graphFor([]);
    node.onNodeCreated();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(node.panels.map((panel) => panel.name), ['prompt_controls']);
    const flat = (element) => [element, ...(element.children || []).flatMap(flat)];
    const elements = () => flat(node.panels[0].frame);
    for (const name of fields.map((item) => item.name)) assert.equal(widget(name).hidden, true);
    assert.ok(elements().some((entry) => entry.placeholder === 'Room tone, dialogue, effects'));
    const press = (label) => elements().find((entry) => entry.textContent === label || entry['aria-label'] === label).events.click();
    press('Action preset');
    assert.equal(widget('action').hidden, true);
    assert.ok(elements().some((entry) => entry.textContent === 'Oral'));
    assert.equal(elements().filter((entry) => entry.tag === 'button').some((entry) => entry.textContent === 'Position'), false);
    press('Image');
    assert.equal(widget('generation').value, 'image');
    assert.equal(widget('soundscape').hidden, true);
    assert.equal(widget('music').hidden, true);
    assert.ok(elements().some((entry) => entry.textContent === 'Solo'));
    press('Solo - standing');
    assert.equal(widget('action').value, 'image: Solo - standing');
    press('Minimax H3');
    press('Oral - blowjob, kneeling');
    assert.equal(widget('action').value, 'h3: Oral - blowjob, kneeling');
    press('Clear');
    assert.equal(widget('action').value, '');
    const search = elements().find((entry) => entry.tag === 'input' && entry.type === 'search');
    search.value = 'kneel';
    search.oninput();
    assert.ok(elements().some((entry) => entry['aria-label'] === 'Oral - blowjob, kneeling'));
    search.value = 'zzzz';
    search.oninput();
    assert.equal(elements().some((entry) => entry['aria-label'] === 'Oral - blowjob, kneeling'), false);
    assert.equal(widget('source').value, 'action');
  } finally {
    globalThis.document = previous.document;
    globalThis.__pfApp = previous.app;
    globalThis.__pfApi = previous.api;
    globalThis.Option = previous.Option;
    globalThis.requestAnimationFrame = previous.raf;
  }
});
