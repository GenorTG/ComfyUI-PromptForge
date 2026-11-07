"""Action and camera presets of the H3 Idea node (adult content): a position or solo act with who is who, and a camera angle, become a brief for the prompt writer.

{A} and {B} stand for the two people (a picture such as <Picture 1> in reference mode, or words). Only the arrangement of the bodies is written here; the
prompt writer's NSFW template turns it into a full video prompt, for one clip or spread over the parts."""

ADULTS = "Everyone in this video is a consenting adult."

# label -> (number of people, arrangement)
ACTIONS = {
    "Solo - self touching, lying down": (1, "{A} lies on a bed or sofa and slowly touches their own body: hands over the chest, stomach and thighs, then between the legs."),
    "Solo - self touching, standing": (1, "{A} stands, partly undressed, hands moving over their own body and then between their legs."),
    "Solo - masturbating with a hand": (1, "{A} masturbates with a hand, starting slowly and building the pace."),
    "Solo - masturbating with a toy": (1, "{A} uses a sex toy on themselves, starting slowly and building the pace."),
    "Solo - male, jerking off": (1, "{A} strokes their own penis with one hand, starting slowly and building the pace."),
    "Solo - male, jerking off with a toy": (1, "{A} uses a masturbation toy on their own penis, starting slowly and building the pace."),
    "Solo - striptease": (1, "{A} slowly undresses for the camera, ending naked, posing and touching their own body."),
    "Oral - blowjob, kneeling": (2, "{A} kneels in front of {B}, who stands; {A} gives {B} oral sex with the mouth and hands."),
    "Oral - blowjob, lying": (2, "{B} lies on their back; {A} lies or kneels between {B}'s legs and gives {B} oral sex."),
    "Oral - cunnilingus": (2, "{B} lies back with the legs apart; {A} kneels between {B}'s legs and gives {B} oral sex."),
    "Oral - 69": (2, "{A} and {B} lie head to toe, each giving the other oral sex at the same time."),
    "Hands - handjob": (2, "{A} sits or kneels beside or in front of {B} and strokes {B}'s penis with a hand."),
    "Hands - titfuck": (2, "{B} lies back or stands; {A} presses their breasts around {B}'s penis and moves."),
    "Hands - mutual masturbation": (2, "{A} and {B} sit or lie facing each other, each touching the other between the legs."),
    "Position - missionary": (2, "{A} lies on their back with the legs open or around {B}; {B} is on top between {A}'s legs, face to face."),
    "Position - cowgirl": (2, "{B} lies on their back; {A} straddles {B} facing them and moves on top."),
    "Position - reverse cowgirl": (2, "{B} lies on their back; {A} straddles {B} facing away from them, back and rear toward {B}."),
    "Position - doggy style": (2, "{A} is on hands and knees; {B} kneels or stands behind {A}."),
    "Position - prone bone": (2, "{A} lies face down, flat on the stomach; {B} lies on top of {A} from behind."),
    "Position - spooning": (2, "{A} and {B} lie on their sides, {B} pressed against {A}'s back."),
    "Position - standing, face to face": (2, "{A} and {B} stand facing each other; {B} holds {A} close, {A}'s arms around {B}."),
    "Position - standing, from behind": (2, "{A} stands bent forward, hands on a wall or furniture; {B} stands behind {A}."),
    "Position - one leg up": (2, "{A} and {B} stand facing each other; {A} has one leg raised, held up by {B}."),
    "Position - lap, face to face": (2, "{B} sits; {A} sits on {B}'s lap facing them, arms around each other."),
    "Position - legs on shoulders": (2, "{A} lies on their back with the legs raised onto {B}'s shoulders; {B} kneels in front, leaning forward."),
    "Position - edge of the bed": (2, "{A} lies on their back at the edge of the bed, legs open; {B} stands in front of {A}."),
    "Position - against the wall": (2, "{A} is held up against a wall by {B}, legs around {B}'s waist."),
    "Position - scissoring": (2, "{A} and {B} lie facing each other with the legs interlocked, their hips pressed together."),
}

CAMERAS = {
    "POV (the viewer is B)": "Point of view of {B}: the camera sees {A} and the action from {B}'s eyes, {B}'s hands and body entering the frame.",
    "POV (the viewer is A)": "Point of view of {A}: the camera sees {B} and the action from {A}'s eyes, {A}'s hands and body entering the frame.",
    "side view, full bodies": "A side view that keeps both bodies fully in frame.",
    "from behind": "The camera is behind the people, looking at the action from behind.",
    "front view": "The camera faces the people from the front at chest height.",
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


def people(label):
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


def brief(label, camera, a, b):
    """The text the prompt writer gets for one chosen action: who is who, the arrangement and the camera."""
    count, text = ACTIONS[label]
    lines = [f"Action: {text.format(A=a, B=b)}", "This action is exactly what happens. The idea above only gives the setting, the looks and the mood; if it describes a different act or position, follow this action instead.", ADULTS]
    if camera in CAMERAS and not (count == 1 and camera in SOLO_CAMERAS):
        lines.append(f"Camera: {CAMERAS[camera].format(A=a, B=b)}")
    return "\n".join(lines)
