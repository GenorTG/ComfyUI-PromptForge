"""Action presets in editable JSON instead of Python.

Shipped presets live in presets/actions/<family>_actions.json and are tracked by git: the SFW
image set and the H3 set. Everything the user writes, and every NSFW image preset, lives in
<family>_actions_extra.json, which is created on the first save and gitignored. A null entry in
the extra file hides a shipped preset, so Delete never has to rewrite a tracked file.

Files are cached by mtime, so hand edits apply without a restart. Read paths repair a broken
entry instead of dropping the whole list; save paths reject it with a reason.
"""
import json
from pathlib import Path
from threading import Lock

from aiohttp import web
from server import PromptServer

FAMILIES = ("image", "h3")
CATEGORIES = {
    "image": ("SFW - portrait and expression", "SFW - solo pose", "SFW - couple pose",
              "NSFW - solo", "NSFW - couple", "NSFW - explicit detail"),
    "h3": ("Solo female", "Solo male", "Oral", "Hands", "Positions", "Anal", "Other"),
}
GENDERS = ("female", "male")
PENETRATIONS = ("auto", "none", "vaginal", "anal")
ASPECTS = ("16:9", "9:16")
PEOPLE = (1, 2, 3)

DIR = Path(__file__).resolve().parent.parent / "presets" / "actions"
_lock = Lock()
_cache = {}


def categories(family):
    _family(family)
    return CATEGORIES[family]


def _family(family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown action family: {family!r}. Use image or h3.")


def _file(family, extra):
    return DIR / f"{family}_actions{'_extra' if extra else ''}.json"


def _read(path):
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    _cache[path] = (key, data)
    return data


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
    _cache.pop(path, None)


def _role(value):
    """One role: a gender string, a [gender, role] pair, or {gender, role}; None when it is neither."""
    if isinstance(value, dict):
        gender, text = value.get("gender"), value.get("role")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        gender, text = value
    else:
        gender, text = value, ""
    if gender not in GENDERS:
        return None
    text = str(text or "").strip()
    return {"gender": gender, "role": text or ("the woman" if gender == "female" else "the man")}


def _normalize(family, label, entry, strict, existing=None):
    """A valid, complete entry. Strict raises for a missing part, lenient repairs it."""
    if not isinstance(entry, dict):
        if strict:
            raise ValueError("An action preset is a JSON object.")
        entry = {}
    label = str(label or "").strip()
    if not label and strict:
        raise ValueError("An action needs a label.")
    category = str(entry.get("category") or "")
    if category not in CATEGORIES[family]:
        if strict:
            raise ValueError(f"Unknown {family} category: {category!r}.")
        category = CATEGORIES[family][0]
    brief = str(entry.get("brief") or "").strip()
    if not brief and strict:
        raise ValueError("An action needs a brief.")
    if family == "image":
        try:
            people = int(entry.get("people") or 1)
        except (TypeError, ValueError):
            people = 1
        people = min(max(people, PEOPLE[0]), PEOPLE[-1])
        return {"label": label, "category": category, "people": people, "brief": brief}

    genders = entry.get("genders")
    if isinstance(genders, str):
        genders = [part.strip() for part in genders.split(",") if part.strip()]
    raw = entry.get("roles") if isinstance(entry.get("roles"), list) else genders
    roles = [role for role in (_role(value) for value in raw or []) if role]
    if not roles:
        if strict:
            raise ValueError("An H3 action needs at least one role (female or male).")
        roles = [{"gender": "female", "role": "the woman"}]
    people = entry.get("people")
    people = len(roles) if not isinstance(people, int) else min(max(people, PEOPLE[0]), PEOPLE[-1])
    if people != len(roles) and strict:
        raise ValueError(f"people is {people} but there are {len(roles)} roles.")
    # keep the written role texts when the genders did not change
    if existing and [role["gender"] for role in existing.get("roles", [])] == [role["gender"] for role in roles]:
        roles = [dict(role) for role in existing["roles"]]
    aspect = str(entry.get("aspect") or "16:9")
    if aspect not in ASPECTS:
        if strict:
            raise ValueError(f"Unknown aspect: {aspect!r}. Use 16:9 or 9:16.")
        aspect = "16:9"
    penetration = str(entry.get("penetration") or "auto")
    if penetration not in PENETRATIONS:
        if strict:
            raise ValueError(f"Unknown penetration: {penetration!r}. Use {', '.join(PENETRATIONS)}.")
        penetration = "auto"
    return {"label": label, "category": category, "roles": roles, "brief": brief,
            "aspect": aspect, "penetration": penetration}


def _shipped(family):
    """The tracked file's entries, keyed by label, repaired."""
    entries = {}
    for label, entry in _read(_file(family, False)).items():
        if isinstance(entry, dict):
            entries[label] = _normalize(family, label, entry, strict=False)
    return entries


def merged(family):
    """Every preset of a family, shipped first, the extra file shadowing it."""
    _family(family)
    entries = _shipped(family)
    for label, entry in _read(_file(family, True)).items():
        if entry is None:
            entries.pop(label, None)
        elif isinstance(entry, dict):
            entries[label] = _normalize(family, label, entry, strict=False, existing=entries.get(label))
    return entries


def labels(family, category=None):
    return [label for label, entry in merged(family).items() if category is None or entry["category"] == category]


def get(family, label):
    _family(family)
    entries = merged(family)
    if label not in entries:
        raise KeyError(f"No {family} action preset called {label!r}.")
    return entries[label]


def has(family, label):
    return label in merged(family)


def nsfw(family, label):
    """H3 is explicit throughout; an image preset is NSFW by its category or by living in the extra file."""
    if family == "h3":
        return True
    return label not in _shipped(family) or get(family, label)["category"].startswith("NSFW")


def save(family, label, entry):
    """Write a preset to the extra file and return the stored entry."""
    _family(family)
    label = str(label or "").strip()
    normalized = _normalize(family, label, entry, strict=True, existing=merged(family).get(label))
    with _lock:
        path = _file(family, True)
        data = _read(path)
        data[label] = normalized
        _write(path, data)
    return normalized


def delete(family, label):
    """Remove a preset. A shipped one is hidden with null; False when there was nothing to remove."""
    _family(family)
    label = str(label or "").strip()
    with _lock:
        path = _file(family, True)
        data = _read(path)
        if data.get(label):
            del data[label]  # an override is removed, the shipped copy shows again
        elif label in _shipped(family) and label not in data:
            data[label] = None  # hide a shipped preset without touching the tracked file
        else:
            return False
        _write(path, data)
    return True


routes = PromptServer.instance.routes


@routes.post("/promptforge/v2/actions/save")
async def _save(request):
    body = await request.json()
    try:
        entry = save(body.get("family"), body.get("label"), body)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"label": entry["label"], "actions": merged(body["family"])})


@routes.post("/promptforge/v2/actions/delete")
async def _delete(request):
    body = await request.json()
    try:
        removed = delete(body.get("family"), body.get("label"))
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    if not removed:
        return web.json_response({"error": "No preset with that label."}, status=404)
    return web.json_response({"actions": merged(body["family"])})
