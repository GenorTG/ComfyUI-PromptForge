#!/usr/bin/env python3
"""Exercise V2 in the isolated ComfyUI server; never run model inference.

Run with /home/genortg/1TB-SAMSUNG/Comfy/comfy-env/bin/python.
The server must use --base-directory /tmp/promptforge-v2-check. Optional
--workflow paths are read only. Reports, downloads and screenshots stay there.
Execution checks use a tiny built-in EmptyImage/PreviewImage backend job and
real frontend executed/preview events; V2 model execution is not attempted.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

URL = os.environ.get("PROMPTFORGE_TEST_URL", "http://127.0.0.1:8199")
ARTIFACTS = Path("/tmp/promptforge-v2-check/browser")
REQUIRED = {
    "PROMPTFORGE_V2_PRESETS", "PROMPTFORGE_V2_LORA_STACK", "PROMPTFORGE_V2_LOAD_IMAGE",
    "PROMPTFORGE_V2_LOAD_VIDEO", "PROMPTFORGE_V2_LOAD_AUDIO", "PROMPTFORGE_V2_IMAGE_MODEL",
    "PROMPTFORGE_V2_IMAGE_FORMAT", "PROMPTFORGE_V2_IMAGE_SAMPLING_SETTINGS",
    "PROMPTFORGE_V2_RUN_INSPECT", "PROMPTFORGE_V2_PROMPT_LIBRARY",
    "PROMPTFORGE_V2_SAVE_METADATA", "PROMPTFORGE_V2_PREVIEW_SETTINGS",
    "PROMPTFORGE_V2_WRITER_MODEL", "PROMPTFORGE_V2_IMAGE_PROMPT_WRITER",
    "PROMPTFORGE_V2_PROMPT_INSPECT",
}


def wait_for_server(timeout):
    deadline = time.monotonic() + timeout
    missing = REQUIRED
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(URL + "/object_info", timeout=3) as response:
                catalogue = json.load(response)
            missing = REQUIRED - catalogue.keys()
            if not missing:
                return catalogue
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Server not ready; missing V2 IDs: {sorted(missing)}")


SETUP = r"""async () => {
  const {app} = await import('/scripts/app.js');
  const {api} = await import('/scripts/api.js');
  const pf = window.pfCheck = {app, api, ids: {}, checks: []};
  pf.assert = (condition, message) => { if (!condition) throw Error(message); };
  pf.node = key => app.graph.getNodeById(pf.ids[key]);
  pf.widget = (key, name) => {
    const result = pf.node(key)?.widgets?.find(w => w.name === name);
    pf.assert(result, `${key}.${name} missing`);
    return result;
  };
  pf.set = (key, name, value) => { const w = pf.widget(key, name); w.value = value; w.callback?.(value); };
  pf.root = (key, name) => {
    const w = pf.widget(key, name);
    const root = w.element;
    pf.assert(root instanceof HTMLElement, `${key}.${name}: no DOM widget element`);
    root.dataset.pfCheck = `${key}/${name}`;
    return root;
  };
  pf.button = (key, panel, label) => {
    const button = [...pf.root(key, panel).querySelectorAll('button')].find(b => b.textContent === label);
    pf.assert(button, `${key}: ${label} button missing`);
    button.click();
  };
  pf.sizes = () => Object.fromEntries(app.graph._nodes.map(n => [n.id, [...n.size]]));
  pf.stable = (before, label) => {
    for (const [id, size] of Object.entries(before)) {
      const node = app.graph.getNodeById(id);
      pf.assert(node && Math.abs(node.size[0] - size[0]) < 0.01 && Math.abs(node.size[1] - size[1]) < 0.01,
        `${label}: node ${id} size ${size} -> ${node?.size}`);
    }
  };
  pf.snapshot = () => Object.fromEntries(app.graph._nodes.map(n => [n.id, {
    type: n.type, role: n.properties.promptforge_role,
    widgets: Object.fromEntries((n.widgets || []).filter(w => w.options?.serialize !== false && ['string','number','boolean'].includes(typeof w.value)).map(w => [w.name,w.value]))
  }]));
  pf.fit = () => {
    const nodes = app.graph._nodes;
    const right = Math.max(...nodes.map(n => n.pos[0] + n.size[0]));
    const bottom = Math.max(...nodes.map(n => n.pos[1] + n.size[1]));
    const canvas = app.canvas;
    canvas.ds.scale = Math.min(1, (canvas.canvas.width - 80) / right, (canvas.canvas.height - 100) / bottom);
    canvas.ds.offset = [35, 50];
    canvas.setDirty(true, true);
  };
  const add = (key, type) => {
    const node = LiteGraph.createNode(type);
    pf.assert(node, `Unregistered ${type}`);
    node.title = `Browser check · ${key}`;
    app.graph.add(node);
    pf.ids[key] = node.id;
    return node;
  };
  const connect = (from, to, type = 'PROMPTFORGE_V2_PIPE') => {
    const a = pf.node(from), b = pf.node(to);
    const output = a.outputs?.findIndex(s => s.type === type);
    const input = b.inputs?.findIndex(s => s.type === type);
    pf.assert(output >= 0 && input >= 0, `Cannot connect ${from} -> ${to} (${type})`);
    pf.assert(a.connect(output, b, input), `Connection failed: ${from} -> ${to}`);
  };
  await app.loadGraphData({last_node_id:0,last_link_id:0,nodes:[],links:[],groups:[],config:{},extra:{},version:0.4});
  const specs = [
    ['model','IMAGE_MODEL'], ['image','LOAD_IMAGE'], ['loras','LORA_STACK'],
    ['format','IMAGE_FORMAT'], ['sample','IMAGE_SAMPLING_SETTINGS'], ['presets','PRESETS'],
    ['library','PROMPT_LIBRARY'], ['preview','PREVIEW_SETTINGS'], ['inspect','RUN_INSPECT'],
    ['prompt','IMAGE_PROMPT'],
    ['writer','WRITER_MODEL'], ['promptWriter','IMAGE_PROMPT_WRITER'], ['promptInspect','PROMPT_INSPECT'],
    ['video','LOAD_VIDEO'], ['audio','LOAD_AUDIO'], ['metadata','SAVE_METADATA'],
  ];
  for (const [key, suffix] of specs) add(key, `PROMPTFORGE_V2_${suffix}`);
  for (const [a,b] of [['model','image'],['image','loras'],['loras','format'],['format','sample'],['sample','prompt'],['prompt','promptWriter'],['promptWriter','promptInspect'],['promptInspect','presets']]) connect(a,b);
  connect('writer','promptWriter','PROMPTFORGE_V2_WRITER_MODEL');
  pf.set('presets','preset_name','browser-baseline');
  pf.set('image','filename','references/browser-first-frame.png');
  pf.set('image','role','first_frame');
  pf.set('image','index',2);
  pf.set('image','description','Private prose excluded by default');
  pf.set('sample','steps',27);
  pf.set('loras','loras',JSON.stringify({filter:'all',items:[{name:'browser-disabled.safetensors',on:false,model:0.7,clip:null,part:'audio'}]}));
  pf.node('loras').promptforgeV2Refresh();
  pf.layout = () => {
    const heights = [0,0,0];
    specs.forEach(([key], index) => {
      const node = pf.node(key), col = index % 3;
      node.pos = [col * 460, heights[col]];
      const min = node.computeSize();
      node.setSize([430, Math.max(min[1] + 50, key === 'loras' ? 350 : 180)]);
      heights[col] += node.size[1] + 55;
    });
    pf.fit();
  };
  pf.layout();
  // Load an actual serialized workflow through the public app entry point.
  const workflow = JSON.parse(JSON.stringify(app.graph.serialize()));
  await app.loadGraphData(workflow);
  pf.fit();
  pf.checks.push('app.loadGraphData with registered V2 nodes');
  return {nodes: specs.length, ids: pf.ids};
}"""

PRESETS = r"""async () => {
  const pf = window.pfCheck;
  pf.button('presets','preset_controls','Capture');
  const store = JSON.parse(pf.widget('presets','presets').value);
  const preset = store.presets['browser-baseline'];
  pf.assert(preset?.nodes.length >= 5, 'Capture did not traverse upstream pipe');
  const byType = suffix => preset.nodes.find(n => n.type === `PROMPTFORGE_V2_${suffix}`);
  pf.assert(byType('LOAD_IMAGE')?.widgets.filename === 'references/browser-first-frame.png', 'Filename missing from preset');
  pf.assert(byType('LOAD_IMAGE').widgets.role === 'first_frame', 'Image role missing');
  pf.assert(byType('LOAD_IMAGE').widgets.index === 2, 'Image index missing');
  pf.assert(!Object.hasOwn(byType('LOAD_IMAGE').widgets,'description'), 'Description captured without include_text');
  pf.assert(Array.isArray(byType('LORA_STACK').widgets.loras), 'LoRA preset value is not an array');
  pf.assert(preset.nodes.every(n => n.role && n.role === pf.app.graph._nodes.find(x => x.properties.promptforge_role === n.role)?.properties.promptforge_role), 'Unstable roles');
  pf.set('sample','steps',3);
  pf.set('image','filename','references/changed.png');
  let before = 0, after = 0;
  const graph = pf.app.graph, beforeChange = graph.beforeChange, afterChange = graph.afterChange;
  graph.beforeChange = function(...args) { before++; return beforeChange.apply(this,args); };
  graph.afterChange = function(...args) { after++; return afterChange.apply(this,args); };
  try { pf.button('presets','preset_controls','Apply'); }
  finally { graph.beforeChange = beforeChange; graph.afterChange = afterChange; }
  pf.assert(before === 1 && after === 1, `Apply undo transactions: ${before}/${after}`);
  pf.assert(pf.widget('sample','steps').value === 27, 'Steps did not restore by named widget');
  pf.assert(pf.widget('image','filename').value === 'references/browser-first-frame.png', 'Reference filename did not restore');
  pf.set('presets','include_text',true);
  pf.button('presets','preset_controls','Capture');
  pf.assert(JSON.parse(pf.widget('presets','presets').value).presets['browser-baseline'].nodes.find(n => n.type === 'PROMPTFORGE_V2_LOAD_IMAGE').widgets.description === 'Private prose excluded by default', 'include_text failed');
  // Capture writer settings through the owned auxiliary socket using the same UI.
  const presetNode = pf.node('presets'), from = pf.node('promptWriter');
  from.connect(from.outputs.findIndex(s => s.type === 'PROMPTFORGE_V2_PIPE'),presetNode,presetNode.inputs.findIndex(s => s.type === 'PROMPTFORGE_V2_PIPE'));
  pf.set('presets','preset_name','writer-chain');
  pf.button('presets','preset_controls','Capture');
  pf.assert(JSON.parse(pf.widget('presets','presets').value).presets['writer-chain'].nodes.some(n => n.type === 'PROMPTFORGE_V2_WRITER_MODEL'), 'Writer auxiliary node not captured');
  // Restore the original upstream graph for the layout and reload checks.
  const sample = pf.node('sample');
  sample.connect(sample.outputs.findIndex(s => s.type === 'PROMPTFORGE_V2_PIPE'),presetNode,presetNode.inputs.findIndex(s => s.type === 'PROMPTFORGE_V2_PIPE'));
  pf.set('presets','preset_name','browser-baseline');
  pf.node('presets').promptforgeV2Refresh();
  pf.checks.push('preset capture/apply, filenames, roles, include_text, LoRA arrays, auxiliary writer, one undo transaction');
  return store;
}"""

CONTROLS = r"""async () => {
  const pf = window.pfCheck;
  for (const key of ['model','sample','format','image','video','audio','loras','library','writer'])
    pf.assert(!pf.node(key).widgets.some(w => w.name === 'output_inspector'), `${key}: unexpected inspector`);
  for (const key of ['inspect','promptWriter','promptInspect','metadata'])
    pf.assert(pf.node(key).widgets.some(w => w.name === 'output_inspector'), `${key}: missing inspector`);
  const parts = () => pf.root('loras','lora_stack_editor').querySelectorAll('select[title="Target layers (audiovisual models)"]');
  pf.set('loras','layer_controls','standard');
  pf.assert(parts().length === 0, 'Part shown in standard mode');
  pf.set('loras','layer_controls','minimax audiovisual');
  pf.assert(parts().length === 1 && parts()[0].value === 'audio', 'Part missing in audiovisual mode');
  pf.set('loras','layer_controls','standard');
  pf.set('library','entry_name','Browser saved prompt');
  pf.set('library','text','A quiet test scene.');
  pf.button('library','prompt_library_controls','Save');
  pf.assert(JSON.parse(pf.widget('library','entries').value)['Browser saved prompt'] === 'A quiet test scene.', 'Library Save failed');
  pf.set('library','text','Changed');
  const dropdown = pf.root('library','prompt_library_controls').querySelector('select');
  dropdown.value = 'Browser saved prompt';
  dropdown.dispatchEvent(new Event('change',{bubbles:true}));
  pf.assert(pf.widget('library','text').value === 'A quiet test scene.', 'Library dropdown did not load text');
  pf.button('library','prompt_library_controls','Delete');
  pf.assert(!Object.hasOwn(JSON.parse(pf.widget('library','entries').value),'Browser saved prompt'), 'Library Delete failed');
  pf.set('prompt','source','idea');
  pf.set('prompt','text','A test idea saved from the V2 prompt node.');
  const ideaRoot = pf.root('prompt','prompt_source_and_saves');
  const ideaName = ideaRoot.querySelector('input[placeholder="Idea name"]');
  pf.assert(ideaName, 'Prompt idea name input missing');
  ideaName.value = 'PromptForge V2 browser check';
  pf.button('prompt','prompt_source_and_saves','Save as');
  await new Promise(resolve => setTimeout(resolve, 200));
  const ideaSelect = ideaRoot.querySelector('select');
  pf.assert([...ideaSelect.options].some(option => option.value === 'PromptForge V2 browser check'), 'Prompt idea did not save');
  pf.set('prompt','text','Changed test idea');
  ideaSelect.value = 'PromptForge V2 browser check';
  ideaSelect.dispatchEvent(new Event('change',{bubbles:true}));
  pf.assert(pf.widget('prompt','text').value === 'A test idea saved from the V2 prompt node.', 'Saved idea did not load');
  pf.button('prompt','prompt_source_and_saves','Delete');
  await new Promise(resolve => setTimeout(resolve, 200));
  pf.assert(![...ideaSelect.options].some(option => option.value === 'PromptForge V2 browser check'), 'Prompt idea did not delete');
  pf.set('prompt','source','raw');
  pf.set('prompt','text','');
  for (const kind of ['image','video','audio']) {
    const response = await pf.api.fetchApi(`/promptforge/v2/media?kind=${kind}`);
    pf.assert(response.ok, `${kind} media route: ${response.status}`);
    pf.assert(Array.isArray((await response.json()).files), `${kind} files response invalid`);
    const picker = pf.root(kind,'reference_file_selector');
    if (kind === 'image') {
      pf.assert([...picker.querySelectorAll('button')].some(button => button.textContent === 'Choose / browse images…'), 'Image library picker missing');
      pf.assert([...picker.querySelectorAll('button')].some(button => button.textContent === 'Paste clipboard image'), 'Clipboard paste control missing');
    } else pf.assert(picker.querySelector('select'), `${kind} picker missing`);
  }
  const response = await pf.api.fetchApi('/promptforge/v2/loras');
  pf.assert(response.ok && Array.isArray((await response.json()).loras), 'LoRA library response must contain loras array');
  pf.checks.push('conditional LoRA parts, diagnostic-only inspectors, workflow prompt library, local media and LoRA endpoints');
  return true;
}"""

RELOAD = r"""async () => {
  const pf = window.pfCheck;
  pf.beforeReload = pf.sizes();
  pf.beforeWidgets = pf.snapshot();
  const saved = JSON.parse(JSON.stringify(pf.app.graph.serialize()));
  await pf.app.loadGraphData(saved);
  pf.fit();
  return true;
}"""

EXECUTION_EVENTS = r"""async () => {
  const pf = window.pfCheck;
  pf.beforeEvents = pf.sizes();
  pf.set('sample','steps',11);
  // Dispatch the real API event consumed by ComfyUI's execution handler.
  for (const key of ['inspect','promptWriter','promptInspect','metadata']) {
    pf.api.dispatchEvent(new CustomEvent('executed',{detail:{node:String(pf.ids[key]), display_node:String(pf.ids[key]), output:{text:['Browser diagnostic'],report:['Browser report']}, merge:false}}));
    pf.assert(pf.node(key).promptforgeV2Output?.textContent.includes('Browser diagnostic'), `${key}: executed text not displayed`);
  }
  pf.assert(pf.widget('sample','steps').value === 11, 'Execution auto-applied a preset');
  pf.api.dispatchEvent(new CustomEvent('promptforge.preview',{detail:{node:String(pf.ids.preview),image:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6N0sAAAAASUVORK5CYII=',mime:'image/png',label:'Browser',step:1,total:2}}));
  pf.assert(pf.root('preview','live_preview').querySelector('img').style.display !== 'none', 'Live image preview not shown');
  pf.stable(pf.beforeEvents,'executed/live preview');
  pf.api.dispatchEvent(new CustomEvent('execution_success',{detail:{}}));
  pf.assert(pf.root('preview','live_preview').querySelector('img').style.display === 'none', 'Live preview not hidden');
  pf.stable(pf.beforeEvents,'preview hide');
  pf.checks.push('real frontend executed/preview event handlers, no preset auto-apply, no preview sizing');
  return true;
}"""


def main():
    global URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", type=float, default=120, help="Seconds to wait for restarted server and uppercase IDs")
    parser.add_argument("--url", help="ComfyUI URL to exercise, including a non-local browser origin if needed")
    parser.add_argument("--workflow", type=Path, action="append", default=[], help="Additionally load an existing UI workflow (read only)")
    parser.add_argument("--skip-queue", action="store_true", help="Skip the tiny built-in backend execution")
    args = parser.parse_args()
    if args.url:
        URL = args.url.rstrip("/")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = {"url": URL, "checks": [], "page_errors": [], "console_errors": [], "http_errors": [], "artifacts": str(ARTIFACTS)}
    page = None
    phase = "browser startup"
    try:
        catalogue = wait_for_server(args.wait)
        report["registered_v2_nodes"] = len([key for key in catalogue if key.startswith("PROMPTFORGE_V2_")])
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(Path.home() / ".cache/ms-playwright")
        with sync_playwright() as p:
            executable = next(iter((Path.home() / ".cache/ms-playwright").glob("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell")), None)
            if not executable:
                raise RuntimeError("No installed Chromium headless shell in ~/.cache/ms-playwright")
            browser = p.chromium.launch(executable_path=str(executable), headless=True, args=["--no-sandbox"])
            context = browser.new_context(viewport={"width": 1800, "height": 1400}, accept_downloads=True)
            # Keep this test local, including frontend startup requests.
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith((URL + "/", "data:", "blob:")) else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: report["page_errors"].append({"phase": phase, "message": str(error), "stack": getattr(error, "stack", "")}))
            page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
            page.on("response", lambda response: report["http_errors"].append({"url": response.url, "status": response.status, "optional": "/api/lm/loras/" in response.url or response.url.endswith("/user.css") or "/api/userdata" in response.url}) if response.status >= 400 else None)
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_function("() => Boolean(window.app?.graph && window.LiteGraph?.registered_node_types?.PROMPTFORGE_V2_PRESETS)", timeout=60000)
            report["browser_crypto"] = page.evaluate("({secure: isSecureContext, randomUUID: typeof globalThis.crypto?.randomUUID})")
            report["startup_page_errors"] = len(report["page_errors"])
            report["startup_http_errors"] = len(report["http_errors"])

            def check(label, javascript):
                nonlocal phase
                phase = label
                print(label, flush=True)
                result = page.evaluate(javascript)
                report["checks"].append(label)
                return result

            try:
                check("load test graph with app.loadGraphData", SETUP)
                page.wait_for_timeout(400)
                check("capture and restore upstream named widgets and roles", PRESETS)
                check("conditional controls, prompt library and local endpoints", CONTROLS)
                with page.expect_download() as download:
                    page.evaluate("pfCheck.button('presets','preset_controls','Export')")
                export_path = ARTIFACTS / "exported-presets.json"
                download.value.save_as(export_path)
                exported = json.loads(export_path.read_text())
                if exported.get("schema_version") != 2 or "browser-baseline" not in exported.get("presets", {}):
                    raise AssertionError("Invalid exported preset JSON")
                page.evaluate("pfCheck.button('presets','preset_controls','Delete')")
                page.evaluate("pfCheck.assert(!Object.hasOwn(JSON.parse(pfCheck.widget('presets','presets').value).presets,'browser-baseline'),'Delete failed')")
                with page.expect_file_chooser() as chooser:
                    page.evaluate("pfCheck.button('presets','preset_controls','Import')")
                chooser.value.set_files(str(export_path))
                page.wait_for_function("JSON.parse(pfCheck.widget('presets','presets').value).presets['browser-baseline']")
                report["checks"].append("real browser preset Export download, Delete, Import file chooser")
                check("serialize and reload saved graph", RELOAD)
                page.wait_for_timeout(400)
                check("size, roles and named widgets stable after reload", """() => {
                  const pf = pfCheck;
                  pf.stable(pf.beforeReload,'reload');
                  pf.assert(JSON.stringify(pf.beforeWidgets) === JSON.stringify(pf.snapshot()),'Roles/named widget values changed on reload');
                  return pf.sizes();
                }""")
                before = page.evaluate("pfCheck.sizes()")
                with page.expect_response(lambda response: response.url.endswith("/promptforge/v2/loras?refresh=1")):
                    page.evaluate("pfCheck.button('loras','lora_stack_editor','Refresh')")
                page.wait_for_timeout(300)
                page.evaluate("before => pfCheck.stable(before,'LoRA Refresh')", before)
                report["checks"].append("actual LoRA Refresh leaves user sizes unchanged")
                check("execution and live-preview event size stability", EXECUTION_EVENTS)
                page.wait_for_timeout(300)
                page.evaluate("pfCheck.stable(pfCheck.beforeEvents,'settled execution events')")
                if not args.skip_queue:
                    queued = check("queue tiny built-in backend job (no V2 model inference)", r"""async () => {
                      const pf = pfCheck;
                      pf.beforeQueue = pf.sizes();
                      const response = await pf.api.fetchApi('/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
                        client_id:pf.api.clientId,
                        prompt:{'900001':{class_type:'EmptyImage',inputs:{width:32,height:32,batch_size:1,color:0}},'900002':{class_type:'PreviewImage',inputs:{images:['900001',0]}}}
                      })});
                      const result = await response.json();
                      pf.assert(response.ok && result.prompt_id,JSON.stringify(result));
                      return result;
                    }""")
                    page.wait_for_function("async id => { const r = await fetch('/history/'+id); const h = await r.json(); return h[id]?.status?.completed; }", arg=queued["prompt_id"], timeout=30000)
                    history = page.evaluate("async id => (await (await fetch('/history/'+id)).json())[id]", queued["prompt_id"])
                    if history["status"]["status_str"] != "success":
                        raise AssertionError(f"Backend execution failed: {history['status']}")
                    page.evaluate("pfCheck.stable(pfCheck.beforeQueue,'backend execution')")
                    report["backend_execution"] = {"prompt_id": queued["prompt_id"], "scope": "32x32 EmptyImage -> PreviewImage; V2 UI events checked separately"}
                page.evaluate("pfCheck.fit()")
                page.wait_for_timeout(300)
                page.screenshot(path=str(ARTIFACTS / "compact-layout.png"))
                report["checks"].append("compact-layout.png captured")
                report["sizes"] = page.evaluate("pfCheck.sizes()")
                for path in args.workflow:
                    phase = f"workflow {path.name}"
                    workflow = json.loads(path.read_text())
                    if not isinstance(workflow.get("nodes"), list):
                        raise ValueError(f"Expected UI workflow: {path}")
                    print(f"load supplied workflow {path.name}", flush=True)
                    workflow_page = context.new_page()
                    workflow_page.add_init_script("""window.__pfBrowserFaults=[];
                      addEventListener('error', e => window.__pfBrowserFaults.push({type:'error',message:e.message,source:e.filename,stack:e.error?.stack}));
                      addEventListener('unhandledrejection', e => window.__pfBrowserFaults.push({type:'rejection',reason:String(e.reason),stack:e.reason?.stack}));""")
                    workflow_page.on("pageerror", lambda error, phase=phase: report["page_errors"].append({"phase": phase, "message": str(error), "stack": getattr(error, "stack", "")}))
                    workflow_page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                    workflow_page.goto(URL, wait_until="domcontentloaded")
                    workflow_page.wait_for_function("() => Boolean(window.app?.graph && window.LiteGraph?.registered_node_types?.PROMPTFORGE_V2_PRESETS)", timeout=60000)
                    error_start = len(report["page_errors"])
                    workflow_page.evaluate("async data => { await app.loadGraphData(data); }", workflow)
                    workflow_page.wait_for_timeout(400)
                    browser_faults = workflow_page.evaluate("window.__pfBrowserFaults")
                    if browser_faults:
                        report.setdefault("workflow_browser_faults", []).append({"workflow": path.name, "faults": browser_faults})
                    workflow_page.evaluate("""saved => {
                      for (const entry of saved.nodes) {
                        const node = app.graph.getNodeById(entry.id);
                        if (!node || node.has_errors) throw Error(`Missing/error node ${entry.id}: ${entry.type}`);
                        if (node.properties.promptforge_role !== entry.properties?.promptforge_role) throw Error(`Role changed: ${entry.id}`);
                        const named = entry.properties?.promptforge_v2_widget_values || {};
                        for (const [name,value] of Object.entries(named)) if (JSON.stringify(node.widgets.find(w=>w.name===name)?.value) !== JSON.stringify(value)) throw Error(`Named widget mismatch ${entry.id}.${name}; actual=${JSON.stringify(node.widgets.map(w=>[w.name,w.value]))}`);
                      }
                      window.workflowBefore = Object.fromEntries(app.graph._nodes.map(n => [n.id, [...n.size]]));
                    }""", workflow)
                    workflow_page.wait_for_timeout(500)
                    workflow_page.evaluate("before => { for (const [id,size] of Object.entries(before)) { const node=app.graph.getNodeById(id); if (!node || Math.abs(node.size[0]-size[0])>.01 || Math.abs(node.size[1]-size[1])>.01) throw Error(`Node auto-resized after load: ${id}`); } }", workflow_page.evaluate("window.workflowBefore"))
                    workflow_page.screenshot(path=str(ARTIFACTS / f"workflow-{path.stem}.png"))
                    if report["page_errors"][error_start:]:
                        raise AssertionError(f"Browser errors opening {path.name}: {report['page_errors'][error_start:]}")
                    workflow_page.close()
                    report["checks"].append(f"supplied workflow {path.name}: named widgets, roles, no auto-resize")
                new_page_errors = report["page_errors"][report["startup_page_errors"]:]
                if new_page_errors:
                    raise AssertionError(f"Browser exceptions during V2 graph exercise: {new_page_errors}")
                # The user's ComfyUI has unrelated custom extensions that issue
                # missing-resource errors during startup. Judge only errors
                # introduced after the V2 workflow exercise begins.
                unexpected_http = [item for item in report["http_errors"][report["startup_http_errors"]:] if not item["optional"]]
                if unexpected_http:
                    raise AssertionError(f"Unexpected HTTP errors: {unexpected_http}")
                report["status"] = "passed"
            except Exception:
                page.screenshot(path=str(ARTIFACTS / "failure.png"))
                raise
            finally:
                browser.close()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
