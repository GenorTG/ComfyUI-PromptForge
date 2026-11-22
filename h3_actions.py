"""Action and camera presets of the H3 Idea node (adult content): a position or solo act with who is who, and a camera angle, become a brief for the prompt writer.

{A} and {B} stand for the two people (a picture such as <Picture 1> in reference mode, or words). Only the arrangement of the bodies is written here; the
prompt writer's video template turns it into a full video prompt, for one clip or spread over the parts.

The presets live in presets/actions/h3_actions.json and in the gitignored h3_actions_extra.json."""

RULES = """# Coherence rules for this video (mandatory, they override the template where they conflict)
1. One continuous shot from the fixed camera. Write exactly one [Shot 1], no other shot labels, no cuts, and no timestamps at all inside the shot: use "then", "after a moment", "as", "meanwhile" instead of 00:0X.
2. Camera: the framing from the camera line, chosen so every subject is fully in frame and every face is visible, except a POV line: a POV is strict, it is the named person's eyes for the whole clip, only their hands, forearms, chest or thighs enter the frame, their face never appears (except in a mirror), and the other person stays in view. At most one slow, small camera move in the whole clip (a slight drift or pan). Never zoom, push in, pull out, rack focus, whip, or cut to a detail. Never end on a close-up or on an empty part of the room.
3. The act is already under way in the first sentence: the bodies are already in the arrangement, the genitals, hands or mouths are already in contact, and the movement starts immediately. No walking in, no undressing, no arranging limbs, except for a striptease.
4. Layout: state where each subject is in the frame (left, right, foreground, background), which way each faces, and which body parts point to the camera. Keep that layout for the whole clip: nobody turns around, swaps places, changes pose or leaves unless the action says so.
5. Motion and contact: name the contact and the rhythm ("his hips thrust into her at a steady pace, about one stroke per second", "her mouth slides up and down the shaft"), then vary the speed, depth and weight. Describe the motion once, do not repeat it every sentence, and keep it running to the final second. Fill the clip with at least eight changes of movement, reaction, breathing and sound; the detailed description is 650-900 words.
6. Bodies: each person has one head, two arms and two legs. Say where each hand and each leg is (for example "her hands rest on his chest", "his hands hold her hips"). Describe only what the camera can see; parts hidden by the pose are not described. Bodies never pass through each other or through furniture.
7. Contact must be physically possible for the arrangement. Follow the arrangement and subject roles literally; never invent another position or swap the roles. The action names the orifice: keep exactly that one for the whole clip. In a vaginal act the anus is never entered, in an anal act the vagina is never entered; the other hole stays visible and empty. Show the shaft entering and leaving the named hole, the grip of the opening, and the wetness or lube. If a reference picture shows the other act, the written action wins.
8. The penis is erect from the first frame, whatever the reference picture shows.
9. Sound is half the clip: every thrust has an audible skin slap and a wet squelch or suction as the shaft moves, the bed or furniture creaks with the rhythm, and breathing, moans and broken words follow the pace. Voices are breathy, interrupted by breath and gasps, never a flat recital. Overall soundscape names, in order: room tone, body impact, wet penetration, breath and voice, and the small sounds of hands, sheets or lube. No music unless asked.
10. No garment from a reference picture is worn in the act: everyone is naked from the first frame. A striptease is the only exception and starts dressed.
11. Write plain present tense with concrete visible facts: no statements about quality, no lines about what is absent, no stage directions."""
from .v2 import action_store

FAMILY = "h3"
ACTION_CATEGORIES = action_store.categories(FAMILY)


PENETRATIVE_PREFIXES = ("Position",)
NON_PENETRATIVE = ("Position - footjob", "Position - intercrural, between the thighs", "Position - standing 69", "Position - scissoring")
PENETRATION_MODES = ("vaginal", "anal")


def penetration_modes(label):
    forced = action_store.get(FAMILY, label)["penetration"]
    if forced == "none":
        return []
    if forced in PENETRATION_MODES:
        return [forced]
    if label.startswith(PENETRATIVE_PREFIXES) and label not in NON_PENETRATIVE:
        return list(PENETRATION_MODES)
    return []



def with_penetration(text, mode):
    if mode == "anal":
        for old, new in (("vagina", "anus"), ("vaginal", "anal"), ("vulva", "anus")):
            text = text.replace(old, new)
    elif mode == "vaginal":
        for old, new in (("anus", "vagina"), ("anal", "vaginal"), ("anally", "vaginally")):
            text = text.replace(old, new)
    return text


def action_category(label):
    return action_store.get(FAMILY, label)["category"]


CAMERAS = {
    "POV (the viewer is B)": "STRICT POV of {B}: the camera IS {B}'s eyes, at eye height, tilted down along {B}'s own body. {B}'s chest, stomach and hips stay visible across the lower third of the frame, and {B}'s hands, forearms or thighs may enter it. {A} stays at arm's length with the whole torso and hips in frame, the contact visible, and {A}'s whole head and face fully inside the upper part of the frame, with headroom above the crown; never crop the head. {B}'s own genitals stay as they are. Never show {B}'s face (only in a mirror), never cut away from this angle. Keep it medium-wide, not a macro close-up. The viewpoint holder's belly and pubic mound stay low in the foreground and must not hide the join. The shaft is rooted in the viewpoint holder's pubic hair at the bottom edge of the frame and the whole visible length runs straight into {A}'s vulva: the labia part and wrap the shaft at the entry, the base of the shaft touches her perineum, one glans only at the far end. The shaft never floats free of a body, never bends away and never gets a second tip.",
    "POV (the viewer is A)": "STRICT POV of {A}: the camera IS {A}'s eyes, at eye height, tilted down along {A}'s own body. {A}'s chest, stomach and hips stay visible across the lower third of the frame, and {A}'s hands, forearms or thighs may enter it. {B} stays at arm's length with the whole torso and hips in frame, the contact visible, and {B}'s whole head and face fully inside the upper part of the frame, with headroom above the crown; never crop the head. {A}'s own genitals stay as they are. Never show {A}'s face (only in a mirror), never cut away from this angle. Keep it medium-wide, not a macro close-up. The viewpoint holder's belly and pubic mound stay low in the foreground and must not hide the join. The shaft is rooted in the viewpoint holder's pubic hair at the bottom edge of the frame and the whole visible length runs straight into {B}'s vulva: the labia part and wrap the shaft at the entry, the base of the shaft touches her perineum, one glans only at the far end. The shaft never floats free of a body, never bends away and never gets a second tip.",
    "side view, full bodies": "A still camera at chest height, side-on to the people, framing both bodies fully with both faces visible.",
    "from behind": "The camera is behind the people, looking at the action from behind.",
    "front view": "A still camera at chest height facing the person or people from the front, the whole body in frame, the face visible.",
    "close-up on the action": "A close-up on the action and the bodies' contact, with faces visible when the camera moves up.",
    "wide shot": "A wide shot of the whole room and both bodies.",
    "high angle, looking down": "A high camera looking down on the people.",
    "low angle, looking up": "A low camera looking up at the people.",
    "over the shoulder": "Over the shoulder of one person, looking at the other.",
    "handheld": "A handheld camera with slight natural movement.",
    "static, locked off": "A static, locked-off camera that does not move.",
    "slow orbit": "The camera moves slowly around the people.",
}

SOLO_CAMERAS = ("POV (the viewer is B)", "POV (the viewer is A)")
NO_ACTION = "(none)"
ANY_CAMERA = "(writer's choice)"


def roles(label):
    """The role texts of an action, in subject order."""
    return [role["role"] for role in action_store.get(FAMILY, label)["roles"]]


def labels():
    return action_store.labels(FAMILY)


def actions():
    """Every H3 preset, label -> entry."""
    return action_store.merged(FAMILY)


def _role_text(role):
    """The role without the repeated gender prefix, for display next to the gender label."""
    for prefix in ("the first woman", "the second woman", "the woman", "the man", "the person"):
        if role == prefix:
            return "solo"
        if role.startswith(prefix + " "):
            return role[len(prefix) + 1:]
    return role


def role_info(label):
    """The role texts with the gender that decides the subject pickers."""
    return [{"gender": role["gender"], "role": _role_text(role['role'])} for role in action_store.get(FAMILY, label)["roles"]]


def stages_for(label):
    """How many bodies the action needs: 1 (solo) or 2."""
    return len(action_store.get(FAMILY, label)["roles"])


def aspect(label):
    """Recommended frame shape for this arrangement."""
    return action_store.get(FAMILY, label)["aspect"]


def stage(index, total):
    """What this part of a long video shows of the action."""
    if total == 1:
        return ""
    if index == 0:
        return f"This is part {index + 1} of {total} of the action: the lead-in, getting into position and the build-up."
    if index == total - 1:
        return f"This is part {index + 1} of {total} of the action: the peak and the end, then a short calm moment."
    return f"This is part {index + 1} of {total} of the action: the main action, changing the pace and the camera framing from the previous part."


def _role_name(genders, index):
    """A readable name for a subject role, numbered when both roles share a gender."""
    gender = genders[index]
    if gender == "male":
        return "the man"
    if gender == "female":
        same = [i for i, g in enumerate(genders) if g == "female"]
        return f"the woman {same.index(index) + 1}" if len(same) > 1 else "the woman"
    return "the person"


POV_B = "POV (the viewer is B)"
POV_A = "POV (the viewer is A)"
CLOSE = "close-up on the action"
SIDE = "side view, full bodies"
FRONT = "front view"
LOW = "low angle, looking up"
HIGH = "high angle, looking down"
BEHIND = "from behind"
OVER = "over the shoulder"


def cameras(label):
    """Useful camera picks for this arrangement, clearest first; POV still fully supported."""
    if label.startswith("Solo"):
        return [FRONT, SIDE, LOW, HIGH, CLOSE]
    if label.startswith("Oral"):
        return [POV_A, POV_B, CLOSE, SIDE]
    if "anal" in label.lower():
        return [SIDE, CLOSE, BEHIND, POV_B, HIGH]
    if label.startswith("Hands"):
        return [CLOSE, POV_B, SIDE, FRONT]
    if "from behind" in label.lower() or "doggy" in label.lower() or "spooning" in label.lower():
        return [SIDE, CLOSE, BEHIND, POV_B]
    if "on top" in label.lower() or "cowgirl" in label.lower():
        return [SIDE, LOW, POV_A, POV_B, FRONT]
    return [SIDE, CLOSE, POV_A, POV_B, FRONT]


def default_camera(label):
    """The framing used when the camera is left to the writer: both bodies and faces in one still frame."""
    if stages_for(label) == 1:
        return FRONT
    return cameras(label)[0]


def brief(label, camera, subjects, notes="", penetration="vaginal"):
    """The text the prompt writer gets for one chosen action: who is who, the arrangement and the camera. `subjects` are three descriptions (the third may be empty)."""
    entry = action_store.get(FAMILY, label)
    counts = len(entry["roles"])
    genders = [role["gender"] for role in entry["roles"]]
    names = [_role_name(genders, index) for index in range(counts)]
    a, b = "Subject 1", "Subject 2"
    ratio = aspect(label)
    shape = "portrait" if ratio == "9:16" else "landscape"
    lines = [
        f"Action preset: {label}.",
        f"Action: {with_penetration(entry['brief'], penetration).format(A=a, B=b)}",
    ]
    camera = default_camera(label) if camera == ANY_CAMERA else camera
    if camera in CAMERAS and not (counts == 1 and camera in SOLO_CAMERAS):
        lines.append(f"Camera: {CAMERAS[camera].format(A=a, B=b)}")
        if camera.startswith("POV"):
            lines.append("The POV viewpoint holder's face never appears; their chest, stomach and hips stay in the lower third, their belly and pubic mound stay low enough not to hide the join, the framing stays medium-wide, and the other person's whole head and face stay fully inside the upper part of the frame with clear headroom and no crop.")
        if camera.startswith("POV") and penetration_modes(label):
            lines.append("In POV the shaft comes in from the bottom edge: angle the hips and lift the shaft so it enters the named hole; the other hole stays empty and beside it.")
    lines += [
        "This action is exactly what happens, from the first second to the last; the notes only give the setting, the looks, the nudity and the mood.",
        "The act is already in progress in the first sentence: the genitals, hands or mouths are already in contact and moving; no setup.",
        "The join is the sharpest, most legible point of the frame: who goes into which hole, with the surrounding skin tight around it.",
        "Anatomy anchors: the base of the shaft leaves his pubic hair and the balls hang under it as two separate ovals, never a second opening; her vulva stays compact and the labia grip the shaft without swelling around it.",
        f"Recommended aspect ratio: {ratio} ({shape}). Frame every body and face fully inside it and keep that framing for the whole clip.",
        "His hips drive the shaft, they are one piece: part of the length always stays hidden in his pelvis, the stroke is short, and her body takes each push. The shaft keeps one fixed length and never floats or bends like a separate object.",
        "Motion beats: play the clip as three clear beats in that same framing - drawn back until only the glans touches the entrance, one held beat with the glans there, then a push all the way back in to the base; the receiving body shifts toward and away with every push.",
        "Write 650-900 words in the detailed description, one continuous shot covering the whole clip beat by beat.",
    ]
    for index, (role, who) in enumerate(zip(entry["roles"], subjects), start=1):
        lines.append(f"Subject {index} is {names[index - 1]} ({_role_text(role['role'])}): {who}")
    lines.append("Only Subject 1 and Subject 2 appear in the video." if counts == 2 else "Only Subject 1 appears in the video.")
    if counts == 2:
        anatomy = []
        for index, gender in enumerate(genders, start=1):
            if gender == "female":
                anatomy.append(f"Subject {index} is female: vulva, vagina and clitoris, never a penis")
            elif gender == "male":
                anatomy.append(f"Subject {index} is male: erect penis and balls, never a vulva")
        if anatomy:
            lines.append("; ".join(anatomy) + ".")
        if "female" in genders and "male" in genders:
            woman = genders.index("female") + 1
            man = genders.index("male") + 1
            lines.append(f"The penis is Subject {man}\'s and it goes into Subject {woman}; Subject {woman} never grows a penis and Subject {man} never gets a vagina.")
            lines.append(f"The penis base stays attached between Subject {man}\'s hips, so the shaft emerges from his body; it is never a separate object, never floats and is never pasted over the bodies.")
    if counts == 1:
        if subjects[1]:
            lines.append(f"Attached reference only, not in the video: Subject 2 ({subjects[1]}). Use it for anatomy and scale only; it is not a separate object.")
        if subjects[2]:
            lines.append(subjects[2] if subjects[2].startswith("The other attached") else f"Attached reference only, not in the video: Subject 3 ({subjects[2]}).")
        lines.append("Subject 1 is the only body in the scene; only Subject 1's genitals are described. The extra reference pictures are anatomy examples, never a second body.")
        if genders[0] == "male":
            lines.append("Subject 1 is male: use his penis and balls, never a vulva. If the reference picture shows a woman, describe her with a penis.")
        else:
            lines.append("Subject 1 is female in the reference: use her vulva, labia, clitoris and vagina, never a penis.")
    elif counts == 2 and subjects[2]:
        if subjects[2].startswith("The other attached"):
            lines.append(subjects[2])
        else:
            lines.append(f"Subject 3 is also in the video, not part of the action: {subjects[2]}")
    if penetration_modes(label) or "Oral" in label or "Hands" in label:
        lines.append("Audio: audible skin slap on each stroke, wet squelch or suction at the point of contact, breathing and broken breathy moans that follow the rhythm; voices are breathy and interrupted by gasps, never flat.")
    if penetration_modes(label):
        rear = camera in (BEHIND, OVER) or any(word in label.lower() for word in ("from behind", "doggy", "prone", "spooning"))
        if penetration == "anal":
            lines.append("PENETRATION: ANAL SEX. The penis is in the anus, not the vagina, for the whole clip. The anus is the lower opening; the shaft rises from the base at his pelvis and its glans disappears into the anus, the ring gripping the shaft. The vulva, labia and clitoris stay visible above, the vagina stays empty, the strokes are slower and tighter. On each pull-back some shaft and the glans stay outside.")
        elif rear:
            lines.append("PENETRATION: VAGINAL SEX. The penis is in the vagina, not the anus, for the whole clip. The vagina is the upper opening, the anus is the lower one; the shaft visibly enters the upper opening, the labia majora and minora grip the shaft, the clitoris stays visible above the join, the anus stays closed and empty below. On each pull-back some shaft and the glans stay outside, then slide back in. Never move the penis into the anus and never let the shaft disappear into the belly.")
        else:
            lines.append("PENETRATION: VAGINAL SEX. The penis is in the vagina, not the anus, for the whole clip. The shaft rises from the base at his pelvis and disappears into the vaginal opening; the glans is buried inside her, not resting on the outside. The labia majora and minora wrap the shaft at the entry point and the clitoris stays visible above it. The shaft crosses the gap under her mons and enters below the clitoris; it never lies on top of her pubic mound. On each pull-back only the lower shaft stays visible, then slides back in. Describe the anus only if this angle actually shows it; never let the shaft disappear into the belly.")
    lines.append("Use the pictures for identity and anatomy only: ignore their clothing, pose, lighting and background.")
    if "nudity" in notes.lower() or "naked" in notes.lower():
        if "striptease" in label.lower():
            lines.append("This is a striptease: Subject 1 starts dressed and ends naked; no clothing from the pictures remains once it is removed.")
        else:
            lines.append("Both subjects are fully naked from the first frame; no clothing from the pictures remains.")
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")
    return "\n".join(lines)
