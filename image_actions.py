"""Pose, place, framing and outfit presets for still-image prompt writing.

Each action is a complete, unambiguous arrangement: where the body is, what the hands and legs do,
where the face points and what the camera sees. Places, cameras, lighting and outfits are separate
so one pose can be reused in any scene. `{A}` and `{B}` are the two subjects; the brief keeps the
subject descriptions from the Action node, or points at the input image for image-to-image work.

The presets themselves live in presets/actions: the SFW set ships in image_actions.json, the NSFW
set and everything the user writes goes to the gitignored image_actions_extra.json.
"""
import re

from .v2 import action_store

CATEGORIES = action_store.categories("image")
FAMILY = "image"

PLACES = {
    "auto": "",
    "studio, seamless backdrop": "A plain photo studio with a seamless backdrop and an even floor.",
    "bedroom, daylight": "A bedroom in daylight, a made bed and a window with thin curtains.",
    "bedroom, night lamp": "A dim bedroom lit by one bedside lamp, dark corners and rumpled sheets.",
    "living room, sofa": "A lived-in living room with a fabric sofa, cushions and a low table.",
    "kitchen": "A domestic kitchen with a counter, cabinets and a window.",
    "bathroom, steam": "A bathroom with tile, a mirror and faint steam in the air.",
    "shower": "A walk-in shower with running water, wet tile and glass.",
    "bathtub": "A freestanding bathtub with water and a tiled wall.",
    "hotel room": "A hotel room with crisp white bedding, a headboard and warm lamps.",
    "poolside": "A pool deck in sunlight, blue water and wet tile.",
    "beach, daylight": "An open beach with sand, surf and a clear horizon.",
    "forest": "A dense forest with trunks, ferns and scattered light.",
    "field at dusk": "A tall grass field at dusk, low warm light and a distant tree line.",
    "city street, day": "A city pavement with shop fronts, parked cars and pedestrians blurred behind.",
    "city street, neon night": "A narrow city street at night, wet asphalt and neon signs.",
    "alley, night": "A brick alley at night with a single overhead lamp.",
    "office": "A clean office with a desk, chair and glass partition.",
    "library": "A library with tall shelves, a wooden table and warm lamps.",
    "gym": "A gym with rubber flooring, racks and mirrored walls.",
    "sauna": "A wooden sauna with slatted benches and low warm light.",
    "car interior": "A car interior with leather seats and light through the windows.",
    "balcony": "A small balcony with a railing and the city behind.",
    "rooftop at sunset": "A flat rooftop at sunset with a low parapet and the skyline behind.",
    "laundromat": "A laundromat with rows of machines and fluorescent light.",
    "art studio": "A paint-spattered art studio with canvases and a tall window.",
    "greenhouse": "A glass greenhouse with leaves, pots and diffused light.",
    "cabin": "A timber cabin with a fireplace and rough walls.",
    "snow, outdoors": "An outdoor snow scene with bare trees and cold blue light.",
}

CAMERAS = {
    "auto": "",
    "full body, straight on": "Full-body framing, camera at chest height, straight on, the whole body and both feet in frame.",
    "full body, low angle": "Full-body framing from a low camera near the floor, looking slightly up, the whole body in frame.",
    "full body, high angle": "Full-body framing from a high camera looking down, the whole body and floor in frame.",
    "cowboy shot": "Framing from mid-thigh to above the head, camera at chest height.",
    "medium shot": "Framing from the waist up, camera at chest height.",
    "close portrait": "Tight head-and-shoulders framing, camera level with the face.",
    "detail close-up": "Very tight framing on one part of the body or the point of contact, the rest cropped.",
    "three-quarter side": "Body at three-quarters to the camera, framing from the knees up.",
    "profile, full body": "The subject exactly side-on, full body in frame.",
    "side view, full bodies": "Still camera at chest height, exactly side-on to the pair, both full bodies in frame with both faces visible.",
    "over the shoulder": "Camera behind one person's shoulder looking at the other, the near shoulder at the edge of frame.",
    "POV": "First-person point of view, the viewer's own body or hands at the bottom of the frame.",
    "from above, lying down": "Camera directly above the subject looking down, the body on the floor or bed filling the frame.",
    "from behind": "Camera behind the subject looking at the back, the face turned away or back over one shoulder.",
    "front view, eye level": "Camera at eye level straight on, the subject facing it, upper body filling the frame.",
    "low angle, from the floor": "Camera on the floor looking up, the body tall against the background and the feet nearest the lens.",
    "high angle, standing": "Camera above head height looking down on the standing subject, the floor around the feet in frame.",
    "top-down": "Camera directly overhead, the body flat against the floor or bed and the limbs spread into the frame.",
    "dutch angle": "Camera tilted fifteen to thirty degrees off level, the horizon or verticals running diagonally.",
    "over the shoulder, tight": "Camera close behind the near person's shoulder, the far person sharp and the near shoulder soft at the edge.",
    "POV (the viewer is A)": "Strict first-person POV of A: the camera sits at A's eye height and tilts down along A's own body, so A's chest, stomach and hips fill the lower third of the frame. B is at arm's length with the whole torso and hips inside the frame, contact visible, B's whole head and face fully inside the upper part of the frame, A's hands may enter; never crop B's head. A's face stays out. This is a medium-wide POV, not a macro close-up.",
    "POV (the viewer is B)": "Strict first-person POV of B: the camera sits at B's eye height and tilts down along B's own body, so B's chest, stomach and hips fill the lower third of the frame. A is at arm's length with the whole torso and hips inside the frame, contact visible, A's whole head and face fully inside the upper part of the frame, B's hands may enter; never crop A's head. B's face stays out. This is a medium-wide POV, not a macro close-up.",
    "handheld, slight tilt": "Handheld framing with a slight roll and off-centre horizon, as if shot at arm's length.",
    "locked off, dead level": "Static camera at chest height, dead level, perfectly centred and symmetrical.",
    "wide environmental": "Wide framing with the subject small in a large, fully visible room.",
    "mirror selfie": "Shot into a mirror, the phone or camera visible at chest height, the reflection centered.",
}

CAMERA_STAGE = {
    "three-quarter side": "The camera stands three-quarters to the pair from the corner of the bed, just above mattress height: the near hip and thigh frame the join and both faces stay open to the lens.",
    "side view, full bodies": "The camera is exactly side-on at bed height, both bodies in profile, the point where the pelvises meet square to the lens.",
    "front view": "The camera is at the foot of the bed facing them: the woman's face and front toward the camera, the man seen from behind above her, his back and shoulders filling the upper frame.",
    "from above, lying down": "The camera is directly above the bed looking down: the man's back on top, the woman's arms, legs and face visible around him.",
    "POV": "The camera is the eyes of the person underneath looking up along their own body at the other person above them.",
    "POV (the viewer is A)": "The camera is A's eyes looking along A's own body at B above: B's face, chest and the contact fill the frame, A's breasts or chest and hips at the lower edge, A's face out of frame.",
    "POV (the viewer is B)": "The camera is B's eyes looking down B's own body at A below: A's face, chest and the contact fill the frame, B's chest or thighs at the lower edge, B's face out of frame.",
    "from behind": "The camera is behind B looking along B's back at A underneath, both bodies in the frame and the join visible from the rear.",
}


LIGHTING = {
    "auto": "",
    "soft window light": "Soft daylight from one window, gentle shadows falling away from it.",
    "hard directional sun": "Hard low sun from one side, long crisp shadows and bright highlights.",
    "golden hour": "Warm low golden-hour light from the side, glow on the skin and long shadows.",
    "overcast": "Flat overcast daylight, soft edges and cool shadows.",
    "neon": "Coloured neon from signs on one side, magenta and blue spill and dark shadows.",
    "single lamp": "One warm practical lamp close to the subject, dark falloff around it.",
    "on-camera flash": "Direct on-camera flash, bright face and hard shadow behind on the wall.",
    "candlelight": "Low warm candlelight from below, soft flicker and deep shadows.",
    "studio softbox": "A large softbox at three-quarters, smooth gradient across the body and a clean shadow.",
    "backlit": "Strong light behind the subject, a bright rim around the body and a darker face.",
    "moonlight": "Cool dim moonlight from one side, blue shadows and low contrast.",
}

OUTFITS = {
    "as in the image": "",
    "nude": "Fully nude.",
    "topless": "Topless, the lower garment still on.",
    "lingerie": "Matching lace lingerie, bra and briefs.",
    "underwear": "Plain cotton underwear.",
    "casual": "A loose T-shirt and jeans.",
    "summer casual": "A cropped top, shorts and sandals.",
    "formal": "A fitted formal dress or a suit.",
    "office wear": "A buttoned shirt, tailored trousers and a belt.",
    "sportswear": "A sports bra and leggings.",
    "swimwear": "A two-piece swimsuit.",
    "raincoat": "An open raincoat over everyday clothes, damp shoulders.",
    "oversized shirt": "An oversized unbuttoned shirt, nothing else.",
    "dress": "A simple short dress.",
    "cosy knit": "A thick knit sweater and socks.",
}

STYLES = {
    "auto": "",
    "photorealistic": "Photorealistic, as a real photograph with natural skin and lens behaviour.",
    "cinematic": "Cinematic film still, anamorphic framing and filmic colour.",
    "editorial fashion": "Editorial fashion photograph, styled and deliberate.",
    "film grain": "Analogue film photograph with visible grain and halation.",
    "anime": "Anime illustration with clean lineart and cel shading.",
    "3D render": "Polished 3D render with soft global illumination.",
    "painterly": "Painterly illustration with visible brush strokes.",
    "black and white": "Black-and-white photograph, strong tonal range.",
}


SUBJECT_SOURCES = ("input image", "text description")


def labels_for(category):
    return action_store.labels(FAMILY, category)


def actions():
    """Every image preset, label -> entry."""
    return action_store.merged(FAMILY)


def has(label):
    return action_store.has(FAMILY, label)


POSE_NOTES = {
    "Sex - missionary": "The woman is BOTTOM, flat on her back, face up. The man is TOP, his chest above hers, hips between her thighs, his body covering hers, his back to the camera. They are on a plain white bed, not the chair from the source. His pelvis flush against hers, pubic bone to pubic bone, the shaft buried to the hilt with only the balls outside and the labia wrapped around its base. Camera three-quarter side so his back and her face are both visible. NOT cowgirl, NOT sitting, NOT from behind, NOT her lying on him.",
    "Sex - cowgirl": "The man is BOTTOM, flat on his back on the bed, face up. The woman is TOP, sitting astride him face to face, both knees planted either side of his hips, her pelvis set down on his. His penis is inside her vagina, the shaft disappearing into her, the labia stretched around its base and his pubic bone against hers. NOT resting between her thighs, NOT her hovering above it, NOT missionary.",
    "Sex - reverse cowgirl": "The man is BOTTOM, flat on his back on the bed, face up. The woman is TOP, sitting astride him with her back to his chest and her rear toward his head, both knees planted either side of his hips, her weight settled down on his pelvis. His penis is inside her vagina, the shaft disappearing into her, the labia stretched around its base and his pubic bone flat under her rear. NOT resting between her thighs, NOT her hovering above it, NOT side by side.",
    "Sex - doggy style": "The woman is BOTTOM on hands and knees. The man kneels behind her. NOT her on her back.",
    "Sex - spooning": "Both lie on their sides, the man behind the woman's back. NOT sitting, NOT her on top.",
    "Oral - cunnilingus": "The woman lies on her back, the man kneels between her legs. NOT standing.",
}


def brief(action, people, camera="auto", place="auto", lighting="auto", outfit="as in the image", style="auto",
          notes="", overrides="", source="input image", instructions=""):
    """The action block appended to the writer's user text."""
    entry = action_store.get(FAMILY, action)
    count, arrangement = entry["people"], entry["brief"]
    labels = ["the woman in the input image" if source == "input image" else "the woman", "the man"]
    action_text = arrangement.format(A=labels[0], B=labels[1])
    action_text = re.sub(r"(?<=[.!?])\s+([a-z])", lambda m: m.group(0)[:-1] + m.group(1).upper(), action_text)
    lines = ["# Image action preset", "Action: " + action_text[:1].upper() + action_text[1:]]
    if action in POSE_NOTES:
        lines.append(f"STAGE LAYOUT (follow exactly): {POSE_NOTES[action]}")
    if people[0].strip():
        lines.append(f"Subject 1 (the woman): {people[0].strip()}")
    if count == 2 and people[1].strip():
        lines.append(f"Subject 2 (the man): {people[1].strip()}")
    if count == 1:
        lines.append("Only this one person appears in the frame." if source != "input image" else
                     "Only this one person appears; the input image supplies the identity.")
    else:
        lines.append("Both people appear in the frame together; the second person is never a background extra.")
        lines.append("Two complete, separate bodies: two heads, two torsos, four arms and four legs. Name what supports each body (bed, floor, wall, chair) and where the free limbs are; the torsos meet only at the named contact.")
    if source == "input image":
        lines.append("The input image supplies identity only: keep the face and body, but replace its pose and framing with the action below; do not keep the source pose.")
        lines.append("Keep the identity, face and body of the person from the input image unchanged.")
    if place != "auto":
        lines.append(f"Place: {PLACES[place]}")
    if count == 2:
        lines.append("Replace the source pose: the arrangement below is authoritative, even if the source is sitting.")
        if not any(word in action.lower() for word in ("standing", "seated", "sits", "sitting", "straddle", "cowgirl", "rides", "lap", "lotus", "squat", "footjob", "crouch", "edge of the bed")):
            lines.append("Both bodies are horizontal on a bed or the floor.")
    if count == 2 and camera == "auto":
        camera = "three-quarter side" if action in POSE_NOTES else "side view, full bodies"
    if camera != "auto":
        lines.append(f"Camera: {CAMERAS[camera]}")
        if camera.startswith("POV"):
            lines.append("POV overrides any 'face to the camera' in the action: the viewpoint holder's face stays out, their chest and stomach stay in the lower third, the other whole torso and their whole head and face stay in frame, and the framing stays medium-wide.")
    if count == 2 and camera in CAMERA_STAGE:
        lines.append(f"Camera stage: {CAMERA_STAGE[camera]}")
    if lighting != "auto":
        lines.append(f"Lighting: {LIGHTING[lighting]}")
    if outfit and outfit != "as in the image":
        lines.append(f"Outfit: {OUTFITS[outfit]}")
    if style != "auto":
        lines.append(f"Style: {STYLES[style]}")
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")
    if overrides.strip():
        lines.append(f"Overrides: {overrides.strip()}")
    if count == 2:
        if action.startswith("Sex - "):
            lines.append("Her body stays whole under him: his hips sit between, behind or above her thighs, never inside her torso. Show the join with exact anatomy: the shaft enters the named opening, the labia grip the shaft, the clitoris stays visible above, and some shaft stays outside her. Never let the shaft merge into the belly or thigh.")
        else:
            lines.append("Show the point of contact between the two bodies clearly.")
    if action in POSE_NOTES:
        lines.append(f"Pose: {POSE_NOTES[action]}")
    if instructions.strip():
        lines.append(f"Extra instructions: {instructions.strip()}")
    if action in POSE_NOTES:
        lines.append(f"Extra instructions, follow these over the source: {POSE_NOTES[action]}")
    lines.append("The pose is exactly this arrangement, held still as one frozen frame. Expand the arrangement with concrete visual detail while preserving every element.")
    return "\n".join(lines)
