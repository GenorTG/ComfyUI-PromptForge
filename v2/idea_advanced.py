"""Idea Advanced: the structured things added in the Prompt node, written out as one request for the prompt writer."""
import json
import re

GROUPS = ("general", "background", "characters", "props", "actions", "camera", "lighting", "style", "sound")
PEOPLE = {"female": "woman", "male": "man"}
BACKGROUND = {"place": "Location", "time": "Time and weather", "detail": "Background detail", "foreground": "Foreground"}
SOUND = {"effect": "Sound effect", "ambience": "Ambience", "music": "Music"}
TIMING = {"first": "first", "then": "then", "meanwhile": "at the same time as the step above", "throughout": "for the whole clip"}

# read by the writer only; the words stay clear of the ones that switch the writer's rules on (inside, behind, watching, man)
INSTRUCTION = ("This is a structured request: its headings and list labels organise it and are not part of the prompt. "
               "Each listed character is a separate individual: keep every trait, garment and placement with the character it is listed under, "
               "without swapping or merging them. Names and numbers only tell subjects apart, so describe each one by its look. "
               "Every interaction happens between exactly the subjects named in it.")
VIDEO_INSTRUCTION = (" Numbered steps happen in that order. Every line under Spoken lines is said on screen by the speaker named: "
                     "write each one into the shot description, word for word, at the moment it is said.")


def parse(value):
    try:
        data = json.loads(value or "{}")
    except ValueError as error:
        raise ValueError(f"Idea Advanced holds invalid data: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Idea Advanced holds invalid data: expected an object.")
    return {group: [item for item in data.get(group) or [] if isinstance(item, dict)] for group in GROUPS}


def _text(item, key="text"):
    return " ".join(str(item.get(key) or "").split())


def _choice(item, key):
    value = _text(item, key)
    return "" if value in ("", "auto", "none") else value


def _sentence(text):
    return text if not text or text[-1] in '.!?"”' else text + "."


def _kind(character):
    if character.get("type") != "custom":
        return PEOPLE.get(character.get("type"), PEOPLE["female"])
    return re.sub(r"(?i)^(?:a|an|the)\s+", "", _text(character, "custom")) or "creature"


def labels(data):
    """What each added character and prop is called in the request, keyed by its id."""
    result = {}
    kinds = [_kind(item) for item in data["characters"]]
    counted = {}
    for item, kind in zip(data["characters"], kinds):
        counted[kind] = counted.get(kind, 0) + 1
        result[item.get("id")] = _text(item, "name") or (f"the {kind}" if kinds.count(kind) == 1 else f"{kind} {counted[kind]}")
    for number, item in enumerate(data["props"], 1):
        result[item.get("id")] = _text(item, "name") or f"prop {number}"
    return result


def _capital(text):
    return text[:1].upper() + text[1:]


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def compose(value, video=False, general="", soundscape="", music=""):
    """(request, facts, speech): the writer's request, the user's own words alone for the check of what the prompt left out, and the spoken lines."""
    data = parse(value)
    names = labels(data)
    facts, sections = [], []

    def fact(text):
        if text:
            facts.append(text)
        return text

    def section(title, lines):
        if lines:
            sections.append("\n".join([title + ":", *lines]))

    lines = [f"- {_sentence(fact(_text(item)))}" for item in data["general"] if _text(item)]
    if general.strip():
        lines.append(f"- {_sentence(fact(' '.join(general.split())))}")
    section("General instructions", lines)
    section("Setting", [f"- {BACKGROUND.get(item.get('kind'), BACKGROUND['place'])}: {_sentence(fact(_text(item)))}"
                        for item in data["background"] if _text(item)])

    lines = []
    for item in data["characters"]:
        kind, name, label = _kind(item), _text(item, "name"), names[item.get("id")]
        if item.get("type") == "custom":
            fact(_text(item, "custom"))
        article = "an" if kind[0].lower() in "aeiou" else "a"
        head = f"{name}, {article} {kind}" if name else _capital(label) if label.startswith("the ") else f"{_capital(label)}, {article} {kind}"
        detail = _sentence(fact(_text(item)))
        lines.append(f"- {head}: {detail}" if detail else f"- {head}.")
    section("Characters", lines)

    lines = []
    for item in data["props"]:
        if _text(item) or _text(item, "name"):
            label, detail = _capital(names[item.get("id")]), _sentence(fact(_text(item)))
            fact(_text(item, "name"))
            lines.append(f"- {label}: {detail}" if detail else f"- {label}.")
    section("Props", lines)

    lines = []
    for item in data["actions"]:
        if not _text(item):
            continue
        who = [names[key] for key in item.get("who") or [] if key in names]
        lead = ""
        if len(who) > 1 and item.get("relation") == "directed":
            lead = f"{_capital(who[0])}, acting on {_join(who[1:])}"
        elif len(who) > 1:
            lead = f"{_capital(_join(who))}, together"
        elif who:
            lead = _capital(who[0])
        timing = TIMING.get(item.get("timing"), "") if video else ""
        if timing:
            lead = f"{lead} ({timing})" if lead else _capital(timing)
        detail = _sentence(fact(_text(item)))
        lines.append(f"{len(lines) + 1}. {lead}: {detail}" if lead else f"{len(lines) + 1}. {detail}")
    section("What happens", lines)

    lines = []
    for item in data["camera"]:
        parts = [_choice(item, "shot"), _choice(item, "angle"), _choice(item, "movement") if video else ""]
        if item.get("focus") in names:
            parts.append(f"focus on {names[item['focus']]}")
        line = ". ".join(part for part in (", ".join(part for part in parts if part), fact(_text(item))) if part)
        if line:
            lines.append(f"- {_sentence(_capital(line))}")
    section("Camera", lines)

    for group, title in (("lighting", "Lighting and mood"), ("style", "Style")):
        section(title, [f"- {_sentence(fact(_text(item)))}" for item in data[group] if _text(item)])

    lines, speech = [], []
    sounds = []
    for item in data["sound"] if video else []:
        text = _text(item)
        if not text:
            continue
        if item.get("kind", "dialogue") in SOUND:
            sounds.append(f"- {SOUND[item['kind']]}: {_sentence(fact(text))}")
        else:
            speaker = _capital(names[item["speaker"]]) if item.get("speaker") in names else "A voice"
            speech.append(fact(text.strip('"“” ')))
            lines.append(f'- {speaker} says: "{speech[-1]}"')
    section("Spoken lines", lines)
    section("Sound", sounds)

    if not sections:
        raise ValueError("Idea Advanced is empty: add a character, an action or any other thing and describe it.")
    # the node's own sound fields only add to an idea that holds something
    if video:
        section("Sound direction", [_sentence(fact(soundscape.strip()))] if soundscape.strip() else [])
        section("Music direction", [_sentence(fact(music.strip()))] if music.strip() not in ("", "N/A") else [])
    return "\n\n".join(sections), "\n".join(facts), speech
