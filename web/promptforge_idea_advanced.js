// Idea Advanced: groups that start empty, each thing added with its own choices and text. v2/idea_advanced.py writes them out for the writer.
const pairs = (...values) => values.map((value) => Array.isArray(value) ? value : [value, value[0].toUpperCase() + value.slice(1)]);
const MOVEMENTS = ["static camera", "slow push in", "slow pull out", "pan left", "pan right", "tilt up", "tilt down", "tracking shot", "orbit", "handheld"];

export const GROUPS = [
  { key: "general", title: "General instructions", add: "instruction", placeholder: "Something the whole prompt should follow" },
  { key: "background", title: "Background", add: "background", placeholder: "Where it happens and what is around",
    choices: [{ name: "kind", label: "Kind", options: [["place", "Place"], ["time", "Time & weather"], ["detail", "Detail"], ["foreground", "Foreground"]] }] },
  { key: "characters", title: "Characters", add: "character", named: "Name (optional)", placeholder: "Looks, clothing, expression, pose, where in the frame",
    choices: [{ name: "type", label: "Type", options: pairs("female", "male", "custom") }] },
  { key: "props", title: "Props", add: "prop", named: "What it is", placeholder: "Look, size, where it is, who holds it" },
  { key: "actions", title: "Actions & interactions", add: "action", who: true, placeholder: "What they do, how they touch or relate",
    choices: [{ name: "relation", label: "How", options: [["together", "Together"], ["directed", "First picked → the others"]], when: (item) => (item.who || []).length > 1 },
      { name: "timing", label: "When", video: true, options: [["", "When: auto"], ["first", "First"], ["then", "Then"], ["meanwhile", "At the same time"], ["throughout", "Whole clip"]] }] },
  { key: "camera", title: "Camera", add: "camera", placeholder: "Anything else about the framing or lens",
    choices: [{ name: "shot", label: "Shot", from: "shot" }, { name: "angle", label: "Angle", from: "angle" },
      { name: "movement", label: "Movement", video: true, options: [["", "Movement: auto"], ...MOVEMENTS.map((value) => [value, value])] },
      { name: "focus", label: "Focus", subjects: true, none: "Focus: auto" }] },
  { key: "lighting", title: "Lighting & mood", add: "lighting", placeholder: "Light sources, colours, atmosphere" },
  { key: "style", title: "Style", add: "style", placeholder: "Medium, look, art style, film stock" },
  { key: "sound", title: "Sound & dialogue", add: "sound", video: true, placeholder: "The spoken words, or the sound",
    choices: [{ name: "kind", label: "Kind", options: [["dialogue", "Dialogue"], ["effect", "Sound effect"], ["ambience", "Ambience"], ["music", "Music"]] },
      { name: "speaker", label: "Speaker", subjects: "characters", none: "Speaker: any voice", when: (item) => (item.kind || "dialogue") === "dialogue" }] },
];

export function parseAdvanced(value) {
  let data = {};
  try { data = JSON.parse(value || "{}") || {}; } catch { data = {}; }
  for (const group of GROUPS) data[group.key] = Array.isArray(data[group.key]) ? data[group.key] : [];
  return data;
}

// the same names the request uses (v2/idea_advanced.py labels)
export function subjectLabels(data) {
  const kinds = data.characters.map((item) => item.type === "custom" ? (item.custom || "").trim() || "creature" : item.type === "male" ? "man" : "woman");
  const labels = new Map();
  const counted = {};
  data.characters.forEach((item, index) => {
    const kind = kinds[index];
    counted[kind] = (counted[kind] || 0) + 1;
    labels.set(item.id, (item.name || "").trim() || (kinds.filter((other) => other === kind).length === 1 ? `the ${kind}` : `${kind} ${counted[kind]}`));
  });
  data.props.forEach((item, index) => labels.set(item.id, (item.name || "").trim() || `prop ${index + 1}`));
  return labels;
}

export const newId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;

export function removeItem(data, key, id) {
  data[key] = data[key].filter((item) => item.id !== id);
  for (const item of data.actions) item.who = (item.who || []).filter((other) => other !== id);
  for (const item of data.camera) if (item.focus === id) item.focus = "";
  for (const item of data.sound) if (item.speaker === id) item.speaker = "";
}

const element = (tag, props = {}) => Object.assign(document.createElement(tag), props);
const CONTROL = "background:var(--comfy-input-bg,#0f1419);color:var(--input-text,#e7eef5);border:1px solid #3b4a56;border-radius:4px;padding:3px 5px;font:12px sans-serif";
const BUTTON = "border:1px solid #65717c;border-radius:5px;padding:2px 8px;background:#252b31;color:#eef5fa;cursor:pointer;font:12px sans-serif";
const ROW = "display:flex;gap:4px;align-items:center;flex-wrap:wrap";

// host: read() and write(value) of the widget, commit(action) for one undo step, video(), values(name) of a node combo, box(options) a text box, preview(value)
export function advancedEditor(host) {
  const root = element("div");
  root.style.display = "contents";
  let data = parseAdvanced("");
  let shown;
  let request = "";

  const stamp = () => `${host.video()}|${host.read()}`;
  const store = () => { host.write(JSON.stringify(data)); shown = stamp(); };
  const restructure = (action) => {
    host.commit(() => { action(); request = ""; store(); });
    paint();
  };

  function button(label, title, action) {
    const control = element("button", { textContent: label, title, type: "button" });
    control.style.cssText = BUTTON;
    control.addEventListener("click", action);
    return control;
  }
  function select(label, options, value, pick) {
    const control = element("select", { ariaLabel: label, title: label });
    control.style.cssText = CONTROL;
    for (const [key, text] of options) control.append(new Option(text, key));
    control.value = value || options[0][0];
    control.addEventListener("change", () => restructure(() => pick(control.value)));
    return control;
  }
  function input(placeholder, item, name) {
    const control = element("input", { type: "text", placeholder, ariaLabel: placeholder, value: item[name] || "" });
    control.style.cssText = `${CONTROL};flex:1;min-width:90px`;
    control.addEventListener("input", () => { item[name] = control.value; store(); });
    // other rows show this name once it is finished
    control.addEventListener("change", () => restructure(() => { item[name] = control.value; }));
    return control;
  }
  function optionsOf(choice, labels) {
    if (choice.from) return host.values(choice.from).map((value) => [value, value === "auto" ? `${choice.label}: auto` : value]);
    if (!choice.subjects) return choice.options;
    const people = choice.subjects === "characters" ? data.characters.map((item) => item.id) : [...labels.keys()];
    return [["", choice.none], ...people.map((id) => [id, labels.get(id)])];
  }

  function card(group, item, index, labels) {
    const head = element("div");
    head.style.cssText = `${ROW};border-left:3px solid #4b86ad;padding-left:6px;margin-top:2px`;
    const title = element("span", { textContent: group.who ? `${index + 1}.` : `${group.add[0].toUpperCase()}${group.add.slice(1)} ${index + 1}` });
    title.style.cssText = "font-weight:600;font-size:11px";
    head.append(title);
    for (const choice of group.choices || []) {
      if ((choice.video && !host.video()) || (choice.when && !choice.when(item))) continue;
      head.append(select(choice.label, optionsOf(choice, labels), item[choice.name], (value) => { item[choice.name] = value; }));
    }
    if (item.type === "custom") head.append(input("What is it? (wolf, robot…)", item, "custom"));
    if (group.named) head.append(input(group.named, item, "name"));
    // kept together at the right edge when the row wraps
    const tools = element("span");
    tools.style.cssText = "display:flex;gap:4px;margin-left:auto";
    const move = (by) => restructure(() => {
      const list = data[group.key];
      const other = index + by;
      if (other >= 0 && other < list.length) [list[index], list[other]] = [list[other], list[index]];
    });
    tools.append(button("↑", "Move up", () => move(-1)), button("↓", "Move down", () => move(1)),
      button("✕", `Remove this ${group.add}`, () => restructure(() => removeItem(data, group.key, item.id))));
    head.append(tools);
    const parts = [head];
    if (group.who) {
      const row = element("div");
      row.style.cssText = ROW;
      row.append(element("span", { textContent: labels.size ? "Who:" : "Add characters or props to pick who takes part." }));
      for (const [id, label] of labels) {
        const place = (item.who || []).indexOf(id);
        const chip = button(place < 0 ? label : `${place + 1} · ${label}`, "Click in the order they take part", () => restructure(() => {
          item.who = place < 0 ? [...(item.who || []), id] : item.who.filter((other) => other !== id);
        }));
        chip.setAttribute("aria-pressed", String(place >= 0));
        if (place >= 0) chip.style.cssText = `${BUTTON};border-color:#78c9ff;background:#24445a`;
        row.append(chip);
      }
      parts.push(row);
    }
    parts.push(host.box({ name: `advanced:${item.id}`, value: item.text || "", placeholder: group.placeholder, minHeight: 40,
      onInput: (value) => { item.text = value; store(); } }));
    return parts;
  }

  function paint() {
    shown = stamp();
    const labels = subjectLabels(data);
    const parts = [];
    for (const group of GROUPS) {
      if (group.video && !host.video()) continue;
      const row = element("div");
      row.style.cssText = `${ROW};justify-content:space-between;border-top:1px solid #333c44;padding-top:4px`;
      const title = element("span", { textContent: group.title });
      title.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.03em;opacity:.8";
      row.append(title, button(`+ Add ${group.add}`, `Add a ${group.add}`, () => restructure(() => {
        data[group.key].push({ id: newId(), text: "", ...(group.key === "characters" ? { type: "female" } : {}), ...(group.who ? { who: [] } : {}) });
      })));
      parts.push(row);
      data[group.key].forEach((item, index) => parts.push(...card(group, item, index, labels)));
    }
    const preview = element("pre", { textContent: request });
    preview.style.cssText = `${CONTROL};white-space:pre-wrap;margin:0;max-height:220px;overflow:auto;display:${request ? "block" : "none"}`;
    const show = button(request ? "Hide request" : "Preview request", "The text the prompt writer receives", async () => {
      if (request) request = "";
      else request = await host.preview(host.read());
      paint();
    });
    show.style.cssText = `${BUTTON};align-self:flex-start`;
    parts.push(show, preview);
    root.replaceChildren(...parts);
  }

  // redraws only when the widget or the generation changed from outside, so typing keeps its cursor
  function render() {
    if (shown === stamp()) return;
    data = parseAdvanced(host.read());
    request = "";
    paint();
  }
  return { element: root, render };
}
