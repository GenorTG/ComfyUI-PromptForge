"""Pose, place, framing and outfit presets for still-image prompt writing.

Each action is a complete, unambiguous arrangement: where the body is, what the hands and legs do,
where the face points and what the camera sees. Places, cameras, lighting and outfits are separate
so one pose can be reused in any scene. `{A}` and `{B}` are the two subjects; the brief keeps the
subject descriptions from the Action node, or points at the input image for image-to-image work.
"""

CATEGORIES = (
    "SFW - portrait and expression",
    "SFW - solo pose",
    "SFW - couple pose",
    "NSFW - solo",
    "NSFW - couple",
    "NSFW - explicit detail",
)

# label -> (number of people, category, arrangement)
ACTIONS = {
    # ---------------------------------------------------------------- portrait
    "Portrait - neutral, facing camera": (1, "SFW - portrait and expression", "{A} faces the camera straight on, head level, eyes open and looking into the lens, mouth relaxed, shoulders square to the frame, hands out of frame."),
    "Portrait - soft smile": (1, "SFW - portrait and expression", "{A} faces the camera with the head tilted a few degrees, a soft closed-mouth smile, eyes to the lens, one hand resting at the side of the neck."),
    "Portrait - looking away": (1, "SFW - portrait and expression", "{A} stands with the body at three-quarters to the camera, head turned away to the side, eyes on something outside the frame, hands relaxed at the sides."),
    "Portrait - over the shoulder": (1, "SFW - portrait and expression", "{A} stands with the back three-quarters to the camera and looks back over one shoulder into the lens, chin near the shoulder, one hand loose at the hip."),
    "Portrait - from above, looking up": (1, "SFW - portrait and expression", "{A} sits or crouches low and tilts the face up to the camera, eyes to the lens, shoulders relaxed, hands on the knees."),
    "Portrait - profile": (1, "SFW - portrait and expression", "{A} stands side-on in full profile, face forward, eyes looking ahead out of frame, arms relaxed, no eye contact with the camera."),
    "Portrait - chin on hand": (1, "SFW - portrait and expression", "{A} leans forward with one elbow on a surface, chin resting on the back of the hand, eyes to the lens, the other arm folded under it."),
    "Portrait - hands in hair": (1, "SFW - portrait and expression", "{A} faces the camera with both hands in the hair, elbows out, head tilted back slightly, eyes half closed and lips parted."),
    "Portrait - laughing": (1, "SFW - portrait and expression", "{A} faces the camera mid-laugh, head tipped back, eyes squeezed shut, one hand lifted near the chest, shoulders loose."),
    "Portrait - serious, low angle": (1, "SFW - portrait and expression", "{A} stands tall with the chin slightly down, eyes straight into a low camera, arms at the sides, expression flat and direct."),

    # ------------------------------------------------------------- solo SFW
    "Solo - standing, relaxed": (1, "SFW - solo pose", "{A} stands facing the camera with the weight on one leg, the other knee soft, hips slightly offset, arms hanging loose, face to the lens."),
    "Solo - standing, hand on hip": (1, "SFW - solo pose", "{A} stands three-quarters to the camera, weight on the back leg, one hand planted on the hip with the elbow out, the other arm relaxed, chin lifted."),
    "Solo - sitting on a chair": (1, "SFW - solo pose", "{A} sits on a chair facing the camera, both feet on the floor, knees together, hands resting on the thighs, back away from the backrest."),
    "Solo - sitting on the floor": (1, "SFW - solo pose", "{A} sits on the floor with the knees drawn up and both arms wrapped around them, chin on the knees, eyes to the camera."),
    "Solo - kneeling": (1, "SFW - solo pose", "{A} kneels upright on both knees facing the camera, thighs together, hands resting on the thighs, back straight, face to the lens."),
    "Solo - crouching": (1, "SFW - solo pose", "{A} crouches low on the balls of the feet, knees apart, forearms resting on the knees, face lifted to the camera."),
    "Solo - lying on the back": (1, "SFW - solo pose", "{A} lies on the back along a surface, one knee bent, arms above the head, face turned to the camera, body in a long line."),
    "Solo - lying on the side": (1, "SFW - solo pose", "{A} lies on one side, propped on the lower elbow, the top leg bent over the other, the free hand at the waist, face to the lens."),
    "Solo - lying on the stomach": (1, "SFW - solo pose", "{A} lies face down with the elbows under the shoulders and the chin on the hands, ankles crossed in the air, eyes to the camera."),
    "Solo - walking toward camera": (1, "SFW - solo pose", "{A} walks toward the camera mid-step, one leg forward, arms swinging naturally, chin up, eyes on the lens."),
    "Solo - leaning on a wall": (1, "SFW - solo pose", "{A} leans one shoulder against a wall, the near leg bent with the foot flat on the wall, arms crossed, face to the camera."),
    "Solo - leaning forward on a table": (1, "SFW - solo pose", "{A} leans forward with both forearms on a table, shoulders down, face lifted to the camera, hips back."),
    "Solo - stretching, arms up": (1, "SFW - solo pose", "{A} stands with both arms stretched overhead, fingers linked, back arched, face lifted and eyes closed."),
    "Solo - dancing": (1, "SFW - solo pose", "{A} is caught mid-dance with the hips swung to one side, one arm high and the other across the body, head turned to the camera."),
    "Solo - back to camera, looking back": (1, "SFW - solo pose", "{A} stands with the back to the camera, both arms raised to the hair, head turned so one eye meets the lens over the shoulder."),
    "Solo - sitting on a bed": (1, "SFW - solo pose", "{A} sits on the edge of a bed facing the camera, legs crossed at the knee, hands braced either side of the hips, leaning back slightly."),
    "Solo - on all fours, SFW": (1, "SFW - solo pose", "{A} is on hands and knees side-on to the camera, back flat, head turned to the lens, hips level."),
    "Solo - jumping": (1, "SFW - solo pose", "{A} is airborne with both knees tucked and both arms out, hair and clothing lifted, face to the camera."),
    "Solo - sitting on a windowsill": (1, "SFW - solo pose", "{A} sits sideways on a windowsill with both legs drawn up, one arm around the knees, head against the glass, face to the camera."),

    # ------------------------------------------------------------ couple SFW
    "Couple - standing side by side": (2, "SFW - couple pose", "{A} and {B} stand shoulder to shoulder facing the camera, the near arms around each other's waists, both faces to the lens."),
    "Couple - holding hands, walking": (2, "SFW - couple pose", "{A} and {B} walk toward the camera hand in hand, mid-step, faces to each other and then to the lens, arms linked."),
    "Couple - embracing": (2, "SFW - couple pose", "{A} and {B} stand chest to chest, arms around each other's backs, heads resting together, eyes closed."),
    "Couple - forehead to forehead": (2, "SFW - couple pose", "{A} and {B} stand close with foreheads touching, eyes closed, hands on each other's arms, faces in profile to the camera."),
    "Couple - lifting": (2, "SFW - couple pose", "{B} holds {A} up with both arms under the thighs, {A}'s arms around {B}'s neck and legs around the waist, both faces to the camera."),
    "Couple - dipping": (2, "SFW - couple pose", "{B} dips {A} backward with one arm behind the shoulders, {A} holds {B}'s shoulder, one leg lifted, both faces to each other."),
    "Couple - dancing": (2, "SFW - couple pose", "{A} and {B} hold a dance frame, one pair of hands joined and the other on the waist and shoulder, caught mid-turn, faces close."),
    "Couple - piggyback": (2, "SFW - couple pose", "{A} rides on {B}'s back with arms around the neck, {B}'s hands under the thighs, both faces to the camera."),
    "Couple - sitting together": (2, "SFW - couple pose", "{A} and {B} sit side by side on the floor with shoulders touching, legs out, hands resting together in the laps, faces to the lens."),
    "Couple - lying head to head": (2, "SFW - couple pose", "{A} and {B} lie on their backs head to head, faces turned to the camera, arms loose at the sides, ankles crossed."),
    "Couple - back hug": (2, "SFW - couple pose", "{B} stands behind {A} with arms wrapped around the waist, chin on {A}'s shoulder, {A}'s hands over {B}'s, both faces to the camera."),
    "Couple - kiss": (2, "SFW - couple pose", "{A} and {B} kiss face to face, eyes closed, one hand on the other's jaw, heads tilted slightly, framed from the chest up."),
    "Couple - dancing close": (2, "SFW - couple pose", "{A} and {B} sway together chest to chest, {A}'s arms over {B}'s shoulders and {B}'s hands at {A}'s waist, temples touching."),
    "Couple - lifting on the shoulders": (2, "SFW - couple pose", "{B} stands upright holding {A} up by the thighs so {A} sits on {B}'s shoulders, {A}'s hands on {B}'s head, both faces up to the camera."),
    "Couple - sitting back to back": (2, "SFW - couple pose", "{A} and {B} sit back to back on the floor with knees drawn up, heads tilted back against each other, eyes closed."),

    # ------------------------------------------------------------- solo NSFW
    "Nude - standing, full front": (1, "NSFW - solo", "{A} stands fully nude facing the camera, weight on one leg, arms relaxed at the sides, legs slightly apart, breasts, stomach and pubic area fully visible."),
    "Nude - standing, hands in hair": (1, "NSFW - solo", "{A} stands fully nude facing the camera with both hands in the hair, elbows out, back arched, legs apart, eyes on the lens."),
    "Nude - on the back, arms above head": (1, "NSFW - solo", "{A} lies fully nude on the back with both arms above the head, one knee raised and the other leg straight, face to the camera, vulva visible."),
    "Nude - on the side": (1, "NSFW - solo", "{A} lies fully nude on one side, the top leg lifted over the other to open the hips, one arm along the body, the other under the head, face to the lens."),
    "Nude - on all fours": (1, "NSFW - solo", "{A} is fully nude on hands and knees side-on, back arched, hips high, the head turned to the camera, breasts hanging."),
    "Nude - kneeling, thighs apart": (1, "NSFW - solo", "{A} kneels upright facing the camera, fully nude, knees apart, hands on the thighs, back straight, chest open to the lens."),
    "Nude - sitting, legs open": (1, "NSFW - solo", "{A} sits fully nude facing the camera with both knees up and apart, feet planted, forearms on the knees, eyes on the lens."),
    "Nude - lying face down, hips raised": (1, "NSFW - solo", "{A} lies face down with the chest down and hips raised, knees apart, the face turned to the camera over one arm, rear and soles visible."),
    "Nude - back to camera, looking back": (1, "NSFW - solo", "{A} stands fully nude with the back to the camera, legs apart, both hands in the hair, head turned so the face and one breast are visible."),
    "Nude - sitting on the heels": (1, "NSFW - solo", "{A} kneels and sits back on the heels fully nude, knees apart, hands on the thighs, back straight, chest and hips open to the camera."),
    "Masturbation - lying, fingers": (1, "NSFW - solo", "{A} lies on the back fully nude with the knees up and apart, one hand on the vulva with two fingers inside, the other hand on a breast, face to the camera."),
    "Masturbation - sitting, fingers": (1, "NSFW - solo", "{A} sits fully nude with the back against a headboard, knees up and apart, one hand between the legs rubbing the clitoris, eyes on the lens."),
    "Masturbation - toy": (1, "NSFW - solo", "{A} lies on the back with the knees up and apart, one hand pushing a sex toy into the vagina, the other flat on the stomach, face to the camera."),
    "Masturbation - standing": (1, "NSFW - solo", "{A} stands with the legs apart and one hand braced on a wall, the other between the legs, fingers on the clitoris, face to the lens."),
    "Striptease - top pulled up": (1, "NSFW - solo", "{A} stands facing the camera with the top pulled up over the breasts and held under the arms, skirt or trousers still on, hips shifted, eyes on the lens."),
    "Striptease - bottom pulled down": (1, "NSFW - solo", "{A} stands facing the camera topless with the bottoms pulled down to the thighs, bent slightly forward, one hand on the fabric, face to the camera."),

    # ----------------------------------------------------------- couple NSFW
    "Sex - missionary": (2, "NSFW - couple", "{B} lies on top of {A} between the open legs, face to face, {A}'s hands on {B}'s shoulders and legs around the waist, {B}'s erect penis inside {A}, bodies joined at the pelvis."),
    "Sex - cowgirl": (2, "NSFW - couple", "{A} straddles {B} facing the camera, sitting down on {B}'s erect penis with the shaft buried inside her, hands on {B}'s chest, {B}'s hands on {A}'s hips, both bodies upright and joined."),
    "Sex - reverse cowgirl": (2, "NSFW - couple", "{A} straddles {B} facing away from {B}'s head, sitting down on the erect penis with the shaft buried inside her, hands on {B}'s knees, rear toward {B}, the face turned back over one shoulder."),
    "Sex - doggy style": (2, "NSFW - couple", "{A} is on hands and knees with the back arched, {B} kneels behind holding {A}'s hips, erect penis inside {A}, side-on to the camera."),
    "Sex - prone bone": (2, "NSFW - couple", "{A} lies face down flat on the stomach, {B} lies over the back with the chest on {A}'s back and hips between the legs, penis inside {A}, faces close."),
    "Sex - spooning": (2, "NSFW - couple", "{A} and {B} lie on the same side, {B} pressed against {A}'s back with one arm under the head and the hand on the hip, penis inside {A}."),
    "Sex - standing from behind": (2, "NSFW - couple", "{A} stands bent forward with hands on a wall, {B} stands behind holding the hips, penis inside {A}, both faces turned to the camera."),
    "Sex - standing face to face": (2, "NSFW - couple", "{B} holds {A} up against a wall with both hands under the thighs, {A}'s legs around the waist and arms around the neck, penis inside {A}, faces close."),
    "Sex - one leg up": (2, "NSFW - couple", "{A} and {B} stand facing each other, {B} holds one of {A}'s legs raised at the thigh, penis inside {A}, {A}'s arms around {B}'s neck."),
    "Sex - legs on shoulders": (2, "NSFW - couple", "{A} lies on the back with both legs raised onto {B}'s shoulders, {B} kneels in front and leans over {A}, penis inside {A}, both faces to the camera."),
    "Sex - edge of the bed": (2, "NSFW - couple", "{A} lies on the back at the edge of a bed with the legs hanging open, {B} stands between the legs and leans in, penis inside {A}, side-on to the camera."),
    "Sex - against the wall": (2, "NSFW - couple", "{B} stands and pins {A} against a wall, {A}'s legs around the waist and arms around the neck, penis inside {A}, profile to the camera."),
    "Sex - lap": (2, "NSFW - couple", "{B} sits and {A} sits on the lap facing {B}, arms around each other, penis inside {A}, both seated, side-on to the camera."),
    "Sex - scissors": (2, "NSFW - couple", "{A} and {B} lie facing each other with the legs interlocked and hips pressed together, genitals in contact, both faces to the camera."),
    "Oral - kneeling blowjob": (2, "NSFW - couple", "{A} kneels in front of {B} who stands, {A}'s hands on {B}'s hips, the erect penis in the mouth, cheeks hollowed, eyes up to the camera."),
    "Oral - lying blowjob": (2, "NSFW - couple", "{B} lies on the back with the knees bent, {A} lies between the legs with the penis in the mouth, one hand at the base, eyes to the camera."),
    "Oral - cunnilingus": (2, "NSFW - couple", "{A} lies on the back with the knees up and apart, {B} kneels between the legs with the mouth on the vulva, hands on the thighs, eyes to the camera."),
    "Oral - 69": (2, "NSFW - couple", "{A} and {B} lie head to toe, each mouth on the other's genitals, {A} on the bottom and {B} above, one hand on the other's hip."),
    "Hands - handjob": (2, "NSFW - couple", "{B} sits or stands and {A} kneels beside, one hand wrapped around the erect penis, thumb over the glans, face turned to the camera."),
    "Hands - titfuck": (2, "NSFW - couple", "{B} kneels over {A}'s chest, the erect penis between {A}'s breasts, {A}'s hands pressing the breasts together, both faces to the camera."),
    "Hands - mutual masturbation": (2, "NSFW - couple", "{A} and {B} sit facing each other with legs apart, each hand between the other's legs on the genitals, faces to the camera."),
    "Hands - fingering": (2, "NSFW - couple", "{A} stands or sits with the legs apart, {B} behind or beside with two fingers inside {A} and the other hand on the hip, both faces to the camera."),

    # --------------------------------------------------------- explicit detail
    "Detail - genital close-up": (2, "NSFW - explicit detail", "A tight close-up from the side of the joined genitals, {B}'s erect penis half inside {A}'s vulva, {A}'s labia around the shaft, thighs and pubic hair in frame, faces cropped."),
    "Detail - penetration, from below": (2, "NSFW - explicit detail", "A close-up from below the pelvis looking up, {B}'s erect penis entering {A}'s vagina, {A}'s legs and {B}'s hips filling the frame, faces out of frame."),
    "Detail - oral close-up": (2, "NSFW - explicit detail", "A tight side close-up of {A}'s mouth on the erect penis, lips sealed around the shaft, one hand at the base, {B}'s stomach behind."),
    "Detail - breasts": (1, "NSFW - explicit detail", "A close-up on the bare chest, both breasts filling the frame, nipples hard, one hand lifting and pressing the near breast, face cropped at the chin."),
    "Detail - rear": (1, "NSFW - explicit detail", "A close-up from behind on the bare rear and lower back, {A} bent forward with the legs apart, the tail of the spine and thighs in frame."),
    "Detail - vulva": (1, "NSFW - explicit detail", "A close-up between the open thighs on the vulva, labia parted by two fingers, clitoris visible, stomach and inner thighs in frame."),
    "Detail - erection": (1, "NSFW - explicit detail", "A close-up on the erect penis from the side, shaft and glans in frame, one hand at the base, stomach and thigh behind."),
    "Detail - face during sex": (2, "NSFW - explicit detail", "A tight close-up on {A}'s face during the act, eyes half closed and mouth open, {B}'s jaw or shoulder at the edge of the frame."),
}

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
    "POV (the viewer is A)": "Strict first-person POV of A: the camera is A's eyes, B and the contact fill the frame, only A's hands, forearms, chest or thighs enter at the lower edge, A's face stays out of frame.",
    "POV (the viewer is B)": "Strict first-person POV of B: the camera is B's eyes, A and the contact fill the frame, only B's hands, forearms, chest or thighs enter at the lower edge, B's face stays out of frame.",
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
    return [label for label, (_, group, _) in ACTIONS.items() if group == category]


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
    count, _, arrangement = ACTIONS[action]
    subjects = [people[0].strip() or ("the person in the input image" if source == "input image" else "the woman"),
                people[1].strip() or ("the other person in the input image" if source == "input image" else "the man")]
    lines = ["# Image action preset", f"Action: {arrangement.format(A=subjects[0], B=subjects[1])}"]
    if action in POSE_NOTES:
        lines.append(f"STAGE LAYOUT (follow exactly): {POSE_NOTES[action]}")
    if count == 1:
        lines.append("Only this one person appears in the frame." if source != "input image" else
                     "Only this one person appears; the input image supplies the identity.")
    else:
        lines.append("Both people appear in the frame together; the second person is never a background extra.")
    if source == "input image":
        lines.append("The input image supplies identity only: keep the face and body, but replace its pose and framing with the action below; do not keep the source pose.")
        lines.append("Keep the identity, face and body of the person from the input image unchanged.")
    if place != "auto":
        lines.append(f"Place: {PLACES[place]}")
    if count == 2:
        lines.append("Replace the source pose: the arrangement below is authoritative, even if the source is sitting.")
        if not any(word in action.lower() for word in ("standing", "seated", "sits", "sitting", "straddle", "cowgirl", "rides", "lap", "lotus", "squat", "footjob", "crouch")):
            lines.append("Both bodies are horizontal on a bed or the floor.")
    if count == 2 and camera == "auto":
        camera = "three-quarter side" if action in POSE_NOTES else "side view, full bodies"
    if camera != "auto":
        lines.append(f"Camera: {CAMERAS[camera]}")
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
            lines.append("Show the point where the bodies join, with the penetration visible: where the shaft meets her body, how far it is outside her, and how the labia sit around its base.")
        else:
            lines.append("Show the point of contact between the two bodies clearly.")
    if action in POSE_NOTES:
        lines.append(f"Pose: {POSE_NOTES[action]}")
    if instructions.strip():
        lines.append(f"Extra instructions: {instructions.strip()}")
    if action in POSE_NOTES:
        lines.append(f"Extra instructions, follow these over the source: {POSE_NOTES[action]}")
    lines.append("The pose is exactly this arrangement, held still as one frozen frame; do not change it or add another subject. Expand the arrangement with concrete visual detail while keeping every element exactly.")
    return "\n".join(lines)
