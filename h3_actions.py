"""Action and camera presets of the H3 Idea node (adult content): a position or solo act with who is who, and a camera angle, become a brief for the prompt writer.

{A} and {B} stand for the two people (a picture such as <Picture 1> in reference mode, or words). Only the arrangement of the bodies is written here; the
prompt writer's NSFW template turns it into a full video prompt, for one clip or spread over the parts."""

ADULTS = "Everyone in this video is a consenting adult."
RULES = """# Coherence rules for this video (mandatory, they override the template where they conflict)
1. One continuous shot. Write a single [Shot 1] and no cuts, whatever the shot budget above says, unless the user's notes ask for cuts.
2. Camera: one fixed framing from the camera line, chosen so every subject is fully in frame and every face is visible. At most one slow, small camera move in the whole clip (a slight drift or pan). Never zoom, push in, pull out, rack focus, whip, or cut to a detail. Never end on a close-up or on an empty part of the room.
3. Layout first: before the action starts, write where each subject is in the frame (left, right, foreground, background), which way each faces, and which body parts point to the camera. Keep that layout for the whole clip: nobody turns around, swaps places, changes pose or leaves unless the action says so.
4. Bodies: each person has one head, two arms and two legs. Say where each hand and each leg is (for example "her hands rest on his chest", "his hands hold her hips"). Describe only what the camera can see; parts hidden by the pose are not described. Bodies never pass through each other or through furniture.
5. Motion: describe the repeating motion once, with its direction, size and tempo ("a slow steady rhythm, about one movement per second") and what moves with it. Do not list the same motion again and again.
6. Contact must be physically possible for the arrangement. Follow the arrangement and subject roles literally; never invent another position or swap the roles.
7. Expressions and sounds follow the action and what is on screen.
8. Write plain present tense with concrete visible facts: no statements about quality, no lines about what is absent, no stage directions."""
RULES = """# Coherence rules for this video (mandatory, they override the template where they conflict)
1. One continuous shot. Write a single [Shot 1] and no cuts, whatever the shot budget above says, unless the user's notes ask for cuts.
2. Camera: one fixed framing from the camera line, chosen so every subject is fully in frame and every face is visible. At most one slow, small camera move in the whole clip (a slight drift or pan). Never zoom, push in, pull out, rack focus, whip, or cut to a detail. Never end on a close-up or on an empty part of the room.
3. Layout first: before the action starts, write where each subject is in the frame (left, right, foreground, background), which way each faces, and which body parts point to the camera. Keep that layout for the whole clip: nobody turns around, swaps places, changes pose or leaves unless the action says so.
4. Bodies: each person has one head, two arms and two legs. Say where each hand and each leg is (for example "her hands rest on his chest", "his hands hold her hips"). Describe only what the camera can see; parts hidden by the pose are not described. Bodies never pass through each other or through furniture.
5. Motion: describe the repeating motion once, with its direction, size and tempo ("a slow steady rhythm, about one movement per second") and what moves with it. Do not list the same motion again and again.
6. Contact must be physically possible for the arrangement. Follow the arrangement and subject roles literally; never invent another position or swap the roles.
7. Expressions and sounds follow the action and what is on screen.
8. Write plain present tense with concrete visible facts: no statements about quality, no lines about what is absent, no stage directions."""

# label -> (roles of the people in the order A, B; the arrangement)
ACTIONS = {
    "Solo - self touching, lying down": (('the person',), "{A} lies on a bed or sofa and slowly touches their own body: hands over the chest, stomach and thighs, then between the legs."),
    "Solo - self touching, standing": (('the person',), "{A} stands, partly undressed, hands moving over their own body and then between their legs."),
    "Solo - masturbating with a hand": (('the person',), "{A} masturbates with a hand, starting slowly and building the pace."),
    "Solo - masturbating with a toy": (('the person',), "{A} uses a sex toy on themselves, starting slowly and building the pace."),
    "Solo - male, jerking off": (('the person',), "{A} strokes their own penis with one hand, starting slowly and building the pace."),
    "Solo - male, jerking off with a toy": (('the person',), "{A} uses a masturbation toy on their own penis, starting slowly and building the pace."),
    "Solo - striptease": (('the person',), "{A} slowly undresses for the camera, ending naked, posing and touching their own body."),
    "Oral - blowjob, kneeling": (('kneels and gives oral sex', 'stands and receives'), "{A} kneels in front of {B}, who stands; {A} gives {B} oral sex with the mouth and hands."),
    "Oral - blowjob, lying": (('between the legs, gives oral sex', 'lies on the back, receives'), "{B} lies on their back; {A} lies or kneels between {B}'s legs and gives {B} oral sex."),
    "Oral - cunnilingus": (('kneels between the legs, gives oral sex', 'lies back, receives'), "{B} lies back with the legs apart; {A} kneels between {B}'s legs and gives {B} oral sex."),
    "Oral - 69": (('partner', 'partner'), "{A} and {B} lie head to toe, each giving the other oral sex at the same time."),
    "Hands - handjob": (('gives the handjob', 'receives'), "{A} sits or kneels beside or in front of {B} and strokes {B}'s penis with a hand."),
    "Hands - titfuck": (('gives, with the breasts', 'receives'), "{B} lies back or stands; {A} presses their breasts around {B}'s penis and moves."),
    "Hands - mutual masturbation": (('partner', 'partner'), "{A} and {B} sit or lie facing each other, each touching the other between the legs."),
    "Position - missionary": (('lies on the back', 'on top, between the legs'), "{A} lies on their back with the legs open or around {B}; {B} is on top between {A}'s legs, face to face."),
    "Position - cowgirl": (('on top, rides', 'lies on the back'), "{B} lies on their back; {A} straddles {B} facing them and moves on top."),
    "Position - reverse cowgirl": (('on top, facing away', 'lies on the back'), "{B} lies on their back; {A} straddles {B} facing away from them, back and rear toward {B}."),
    "Position - doggy style": (('on hands and knees', 'behind'), "{A} is on hands and knees; {B} kneels or stands behind {A}."),
    "Position - prone bone": (('lies face down', 'on top from behind'), "{A} lies face down, flat on the stomach; {B} lies on top of {A} from behind."),
    "Position - spooning": (('in front', 'behind'), "{A} and {B} lie on their sides, {B} pressed against {A}'s back."),
    "Position - standing, face to face": (('held close, arms around', 'holds'), "{A} and {B} stand facing each other; {B} holds {A} close, {A}'s arms around {B}."),
    "Position - standing, from behind": (('bent forward', 'stands behind'), "{A} stands bent forward, hands on a wall or furniture; {B} stands behind {A}."),
    "Position - one leg up": (('one leg raised', 'holds the leg up'), "{A} and {B} stand facing each other; {A} has one leg raised, held up by {B}."),
    "Position - lap, face to face": (('on the lap, facing', 'sits'), "{B} sits; {A} sits on {B}'s lap facing them, arms around each other."),
    "Position - legs on shoulders": (('on the back, legs raised', 'kneels in front'), "{A} lies on their back with the legs raised onto {B}'s shoulders; {B} kneels in front, leaning forward."),
    "Position - edge of the bed": (('lies at the edge', 'stands in front'), "{A} lies on their back at the edge of the bed, legs open; {B} stands in front of {A}."),
    "Position - against the wall": (('held against the wall', 'holds'), "{A} is held up against a wall by {B}, legs around {B}'s waist."),
    "Position - scissoring": (('partner', 'partner'), "{A} and {B} lie facing each other with the legs interlocked, their hips pressed together."),
}

CAMERAS = {
    "POV (the viewer is B)": "Point of view of {B}: the camera sees {A} and the action from {B}'s eyes, {B}'s hands and body entering the frame.",
    "POV (the viewer is A)": "Point of view of {A}: the camera sees {B} and the action from {A}'s eyes, {A}'s hands and body entering the frame.",
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
    return ACTIONS[label][0]


def stage(index, total):
    """What this part of a long video shows of the action."""
    if total == 1:
        return ""
    if index == 0:
        return f"This is part {index + 1} of {total} of the action: the lead-in, getting into position and the build-up."
    if index == total - 1:
        return f"This is part {index + 1} of {total} of the action: the peak and the end, then a short calm moment."
    return f"This is part {index + 1} of {total} of the action: the main action, changing the pace and the camera framing from the previous part."


def default_camera(label):
    """The framing used when the camera is left to the writer: both bodies and faces in one still frame."""
    return "front view" if len(ACTIONS[label][0]) == 1 else "side view, full bodies"


def brief(label, camera, subjects, notes=""):
    """The text the prompt writer gets for one chosen action: who is who, the arrangement and the camera. `subjects` are three descriptions (the third may be empty)."""
    count = len(ACTIONS[label][0])
    a, b = subjects[0], subjects[1]
    lines = [f"Action: {ACTIONS[label][1].format(A=a, B=b)}", "This action is exactly what happens; the notes only give the setting, the looks and the mood.", ADULTS]
    for number, (role, who) in enumerate(zip(ACTIONS[label][0], subjects), start=1):
        lines.append(f"Subject {number} ({role}): {who}")
    if subjects[2]:
        lines.append(f"Subject 3 is also in the video, not part of the action: {subjects[2]}")
    camera = default_camera(label) if camera == ANY_CAMERA else camera
    if camera in CAMERAS and not (count == 1 and camera in SOLO_CAMERAS):
        lines.append(f"Camera: {CAMERAS[camera].format(A=a, B=b)}")
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")
    return "\n".join(lines)
