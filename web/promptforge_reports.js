import { app } from "../../scripts/app.js";

const TYPES = new Set(["PROMPTFORGE_V2_PROMPT_INSPECT", "PROMPTFORGE_V2_RUN_INSPECT", "PROMPTFORGE_V2_SHOW_TEXT"]);
const element = (tag, props = {}) => Object.assign(document.createElement(tag), props);

function badge(text, tone) {
  const span = element("span", { textContent: text });
  span.style.cssText = `padding:1px 7px;border-radius:9px;font:600 11px sans-serif;background:${tone};color:#fff`;
  return span;
}

function block(title, body, mono = false) {
  const wrap = element("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:3px";
  const head = element("div", { textContent: title });
  head.style.cssText = "font:600 11px sans-serif;opacity:.65;text-transform:uppercase;letter-spacing:.04em";
  const text = element("div", { textContent: body });
  text.style.cssText = `font:${mono ? "12px ui-monospace,monospace" : "12px sans-serif"};white-space:pre-wrap;word-break:break-word;line-height:1.45;background:#0004;border-radius:6px;padding:7px`;
  wrap.append(head, text);
  return wrap;
}

function render(node, raw) {
  const view = node.__report;
  if (!view) return;
  view.replaceChildren();
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    view.append(block("report", raw, true));
    return;
  }

  const top = element("div");
  top.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;align-items:center";
  if (data.family) top.append(badge(String(data.family), "#3f789e"));
  if (data.prompt?.source) top.append(badge(`source: ${data.prompt.source}`, "#54502a"));
  if (data.writer?.status) top.append(badge(`writer: ${data.writer.status}`, data.writer.status === "valid" ? "#2c5237" : "#6b3030"));
  const issues = data.validation || [];
  top.append(badge(issues.length ? `${issues.length} issue(s)` : "clean", issues.length ? "#7a2735" : "#245a52"));
  for (const [name, count] of Object.entries(data.token_counts || {})) top.append(badge(`${name}: ${count}`, "#444"));
  view.append(top);

  if (issues.length) view.append(block("validation", issues.join("\n"), true));
  if (data.prompt?.text) view.append(block("prompt settings text", String(data.prompt.text)));
  if (data.prompt?.action) view.append(block("action", String(data.prompt.action)));

  const sections = data.sections && Object.keys(data.sections).length ? data.sections : null;
  if (sections) {
    for (const [name, body] of Object.entries(sections)) view.append(block(name, String(body), true));
  } else if (data.text) {
    view.append(block("generation text", String(data.text), true));
  }

  const copy = element("button", { textContent: "Copy generation text", type: "button" });
  copy.style.cssText = "align-self:flex-start;padding:4px 10px;border-radius:6px;border:1px solid #fff3;background:#0005;color:inherit;font:600 11px sans-serif;cursor:pointer";
  copy.addEventListener("click", () => navigator.clipboard?.writeText(String(data.text || "")));
  view.append(copy);

  if (data.writer?.attempts?.length) {
    const details = element("div");
    details.style.cssText = "font:11px sans-serif;opacity:.6";
    details.textContent = `${data.writer.attempts.length} writer attempt(s)`;
    view.append(details);
  }
}

app.registerExtension({
  name: "PromptForge.V2.Reports",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TYPES.has(nodeData.name)) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const frame = element("div");
      frame.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint";
      const content = element("div");
      content.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:6px;color:var(--input-text)";
      frame.append(content);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());
      frame.addEventListener("wheel", (event) => event.stopPropagation());
      this.addDOMWidget("report_view", "promptforge_report", frame, { serialize: false, getMinHeight: () => 340 });
      this.__report = content;
      this.size = [520, 420];
      const executed = this.onExecuted;
      this.onExecuted = function (message) {
        executed?.apply(this, arguments);
        const raw = message?.text?.[0];
        if (typeof raw === "string") render(this, raw);
      };
    };
  },
});
