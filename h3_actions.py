"""Action and camera presets of the H3 Idea node (adult content): a position or solo act with who is who, and a camera angle, become a brief for the prompt writer.

{A} and {B} stand for the two people (a picture such as <Picture 1> in reference mode, or words). Only the arrangement of the bodies is written here; the
prompt writer's video template turns it into a full video prompt, for one clip or spread over the parts."""

RULES = """# Coherence rules for this video (mandatory, they override the template where they conflict)
1. One continuous shot from the fixed camera. Write exactly one [Shot 1], no other shot labels, no cuts, and no timestamps at all inside the shot: use "then", "after a moment", "as", "meanwhile" instead of 00:0X.
2. Camera: the framing from the camera line, chosen so every subject is fully in frame and every face is visible, except a POV line: a POV is strict, it is the named person's eyes for the whole clip, only their hands, forearms, chest or thighs enter the frame, their face never appears (except in a mirror), and the other person stays in view. At most one slow, small camera move in the whole clip (a slight drift or pan). Never zoom, push in, pull out, rack focus, whip, or cut to a detail. Never end on a close-up or on an empty part of the room.
3. The act is already under way in the first sentence: the bodies are already in the arrangement, the genitals, hands or mouths are already in contact, and the movement starts immediately. No walking in, no undressing, no arranging limbs, except for a striptease.
4. Layout: state where each subject is in the frame (left, right, foreground, background), which way each faces, and which body parts point to the camera. Keep that layout for the whole clip: nobody turns around, swaps places, changes pose or leaves unless the action says so.
5. Motion and contact: name the contact and the rhythm ("his hips thrust into her at a steady pace, about one stroke per second", "her mouth slides up and down the shaft"), then vary the speed, depth and weight. Describe the motion once, do not repeat it every sentence, and keep it running to the final second. Fill the clip with at least eight changes of movement, reaction, breathing and sound; the detailed description is 500-800 words.
6. Bodies: each person has one head, two arms and two legs. Say where each hand and each leg is (for example "her hands rest on his chest", "his hands hold her hips"). Describe only what the camera can see; parts hidden by the pose are not described. Bodies never pass through each other or through furniture.
7. Contact must be physically possible for the arrangement. Follow the arrangement and subject roles literally; never invent another position or swap the roles. The action names the orifice: keep exactly that one for the whole clip. In a vaginal act the anus is never entered, in an anal act the vagina is never entered; the other hole stays visible and empty. Show the shaft entering and leaving the named hole, the grip of the opening, and the wetness or lube. If a reference picture shows the other act, the written action wins.
8. The penis is erect from the first frame, whatever the reference picture shows.
9. Sound is half the clip: every thrust has an audible skin slap and a wet squelch or suction as the shaft moves, the bed or furniture creaks with the rhythm, and breathing, moans and broken words follow the pace. Voices are breathy, interrupted by breath and gasps, never a flat recital. Overall soundscape names, in order: room tone, body impact, wet penetration, breath and voice, and the small sounds of hands, sheets or lube. No music unless asked.
10. No garment from a reference picture is worn in the act: everyone is naked from the first frame. A striptease is the only exception and starts dressed.
11. Write plain present tense with concrete visible facts: no statements about quality, no lines about what is absent, no stage directions."""
# label -> (roles of the people in the order A, B; the arrangement)
ACTIONS = {
    # solo
    "Solo - self touching, lying down": ((("female", "the woman"),), "{A} lies on a bed or sofa, fully naked, and touches her own body: palms over the breasts and stomach, then fingers on the vulva, stroking the labia and circling the clitoris, one or two fingers sliding inside, hips rocking into the hand. For a male, the hand wraps around the erect penis instead.", "16:9"),
    "Solo - self touching, standing": ((("female", "the woman"),), "{A} stands, fully naked, runs her hands over the breasts, stomach, hips and thighs, then between the legs, rubbing the vulva and clitoris with the fingers, one hand braced on a wall or furniture. For a male, the hand strokes the erect penis instead.", "9:16"),
    "Solo - self touching, sitting": ((("female", "the woman"),), "{A} sits with the back against a headboard or wall, the legs open toward the camera, fully naked, one hand on a breast and the other between the legs, fingers rubbing the clitoris and sliding into the vagina, the hips lifting into the hand.", "9:16"),
    "Solo - masturbating with a hand": ((("female", "the woman"),), "{A} masturbates with a hand: fingers rubbing the clitoris and sliding into the vagina, starting slowly and building the pace, the other hand on a breast or gripping the bedding. For a male, the fist strokes the erect penis from base to glans.", "16:9"),
    "Solo - masturbating with a toy": ((("female", "the woman"),), "{A} pushes a sex toy into the vagina and thrusts it, the other hand working the clitoris, the hips meeting every push, starting slowly and building the pace.", "16:9"),
    "Solo - toy, standing against a wall": ((("female", "the woman"),), "{A} stands with one shoulder against a wall, one leg lifted and braced on a chair, fully naked, and works a toy in and out of the vagina with one hand while the other rubs the clitoris.", "9:16"),
    "Solo - in the shower": ((("female", "the woman"),), "{A} stands under running water in a shower, fully naked, one hand braced on the wet tiles, the other between the legs working the vulva and clitoris, water running over the breasts and back, the hips rolling into the hand.", "9:16"),
    "Solo - in front of a mirror": ((("female", "the woman"),), "{A} stands in front of a mirror, fully naked, facing the reflection, one hand on a breast and the other between the legs, watching herself, the hips pushing into the fingers.", "9:16"),
    "Solo - male, jerking off": ((("male", "the man"),), "{A} strokes his erect penis with one hand, the fist moving from base to glans, hips pushing into the stroke, the other hand resting on his thigh or balls. He starts slowly and builds the pace.", "16:9"),
    "Solo - male, jerking off with a toy": ((("male", "the man"),), "{A} pushes his erect penis into a masturbation sleeve or toy and thrusts into it, one hand holding the toy, starting slowly and building the pace, the hips driving forward.", "16:9"),
    "Solo - striptease": ((("female", "the woman"),), "{A} starts dressed and slowly undresses for the camera: lifts the top over the head and drops it, pushes the bottoms down the legs and steps out of them, until fully naked; then poses and touches her own body, hands over the breasts, stomach, hips and between the legs. She starts clothed and is naked only after the last garment comes off.", "9:16"),
    # oral
    "Oral - blowjob, kneeling": ((("female", "the woman kneels, gives oral"), ("male", "the man stands, receives")), "{A} kneels in front of {B}, who stands with an erect penis; {A} holds it at the base, takes the glans and shaft into the mouth and sucks, the head moving along it. {B}'s hands rest on her head or hips.", "9:16"),
    "Oral - blowjob, lying between the legs": ((("female", "the woman lies down, gives oral"), ("male", "the man lies on his back, receives")), "{B} lies on his back with an erect penis; {A} lies between {B}'s legs on her stomach, takes the penis into the mouth and sucks, the head bobbing along the shaft, one hand at the base.", "16:9"),
    "Oral - blowjob, from the side": ((("female", "the woman kneels, gives oral"), ("male", "the man stands, receives")), "{B} stands; {A} kneels beside him, turned so her profile is to the camera, and takes the erect penis into the mouth, one hand at the base, the head sliding up and down the shaft while {B} looks down at her.", "16:9"),
    "Oral - face sitting": ((("female", "the woman sits on top, receives oral"), ("male", "the man lies below, gives oral")), "{B} lies on his back; {A} straddles his face, facing the camera, her vulva at his mouth, {B}'s hands on her thighs holding her down while his tongue works the clitoris and labia. Her hands rest on the headboard or his stomach.", "9:16"),
    "Oral - cunnilingus": ((("female", "the woman lies back, receives oral"), ("male", "the man kneels between her legs, gives oral")), "{A} lies back with the legs apart; {B} kneels between {A}'s legs, presses the mouth to the vulva, licks and sucks the clitoris and labia, hands holding the thighs open. If {A} is male, the tongue runs along the erect shaft instead.", "16:9"),
    "Oral - 69": ((("female", "the woman on the bottom"), ("male", "the man on top")), "{A} and {B} lie head to toe, each with the other's genitals at the mouth, giving oral sex at the same time: {B} sucks {A}'s vulva while {A} takes {B}'s erect penis into her mouth.", "16:9"),
    "Oral - deepthroat, kneeling": ((("female", "the woman kneels, gives oral"), ("male", "the man stands, receives")), "{A} kneels in front of {B} and takes the erect penis deep into the mouth and throat, nose against his body, one hand at the base, then pulls back to the glans and pushes in again in a steady rhythm, saliva on the shaft.", "9:16"),
    # hands
    "Hands - handjob": ((("female", "the woman gives"), ("male", "the man receives")), "{A} sits or kneels beside {B} and strokes {B}'s erect penis with one hand, the thumb sliding over the glans, building the pace, the other hand on his thigh.", "16:9"),
    "Hands - handjob from behind": ((("female", "the woman reaches around, gives"), ("male", "the man stands in front, receives")), "{B} stands; {A} stands behind him, pressed to his back, reaches around his hips and strokes the erect penis, the fist moving base to glans, her chin at his shoulder.", "16:9"),
    "Hands - fingering from behind": ((("female", "the woman in front, receives"), ("male", "the man behind, gives")), "{A} stands or kneels in front of {B}; {B} reaches around from behind, one hand on her breast and the other between her legs, fingers rubbing the clitoris and sliding into the vagina while she leans back into him.", "16:9"),
    "Hands - titfuck, lying": ((("female", "the woman below, gives with the breasts"), ("male", "the man above, receives")), "{A} lies on her back; {B} straddles her chest with an erect penis; {A} presses her breasts around the shaft and moves her chest up and down while {B} thrusts between them, his hands braced beside her head.", "9:16"),
    "Hands - titfuck, sitting": ((("female", "the woman sits, gives with the breasts"), ("male", "the man kneels above, receives")), "{A} sits leaned back on a sofa; {B} kneels over her chest with an erect penis; {A} presses her breasts around the shaft and moves her chest while {B} thrusts between them, looking down at her.", "9:16"),
    "Hands - mutual masturbation": ((("female", "the woman"), ("male", "the man")), "{A} and {B} sit or lie facing each other, each reaching between the other's legs: {A} strokes {B}'s erect penis, {B} fingers {A}'s vulva and clitoris, both hands moving at the same pace.", "16:9"),
    # intercourse, lying
    "Position - missionary": ((("female", "the woman on her back"), ("male", "the man on top, between her legs")), "{A} lies on her back, legs open and around {B}; {B} lies on top between {A}'s legs, face to face, erect penis in {A}'s vagina, thrusting with a steady rhythm, his forearms beside her shoulders.", "16:9"),
    "Position - missionary, legs on shoulders": ((("female", "the woman on her back, legs raised"), ("male", "the man kneels in front")), "{A} lies on her back with the legs raised and the calves resting on {B}'s shoulders; {B} kneels in front, leans forward and thrusts into {A} at a deep, steady pace, her hips lifted off the bed.", "16:9"),
    "Position - missionary, pillow under the hips": ((("female", "the woman on her back, hips raised"), ("male", "the man above")), "{A} lies on her back with a pillow under the hips and the legs wide and high; {B} lies above her, face to face, erect penis in {A}'s vagina, thrusting down at an angle that presses the clitoris, her hands on his back.", "16:9"),
    "Position - prone bone": ((("female", "the woman face down"), ("male", "the man on top from behind")), "{A} lies face down, flat on the stomach, the legs together; {B} lies on top of {A} from behind, penis in {A}, thrusting in short, firm strokes, his hands beside her shoulders.", "16:9"),
    "Position - spooning": ((("female", "the woman in front"), ("male", "the man behind")), "{A} and {B} lie on their sides, {B} pressed against {A}'s back, penis in {A} from behind, one hand on her breast, thrusting slowly, both bodies curved together.", "16:9"),
    "Position - scissoring": ((("female", "the first woman"), ("female", "the second woman")), "{A} and {B} lie facing each other with the legs interlocked, hips pressed together, grinding their vulvas together in slow circles, hands holding each other's thighs or hips. If one is male, his erect penis slides between the other's thighs and vulva.", "16:9"),
    # intercourse, seated / on top
    "Position - cowgirl": ((("female", "the woman on top, rides"), ("male", "the man lies on his back")), "{B} lies on his back with an erect penis; {A} straddles {B} facing him, already seated with the penis inside her, riding up and down, her hands on his chest, her breasts moving with the bounce.", "9:16"),
    "Position - cowgirl, leaning back": ((("female", "the woman on top, leans back"), ("male", "the man lies on his back")), "{B} lies on his back; {A} sits on him facing away-leaning back with her hands planted on his thighs behind her, the penis inside her, rolling and grinding her hips in slow circles, her back arched.", "9:16"),
    "Position - reverse cowgirl": ((("female", "the woman on top, facing away"), ("male", "the man lies on his back")), "{B} lies on his back with an erect penis; {A} straddles {B} facing away from him, already seated with the penis inside her, riding up and down, her back and rear toward {B}, his hands on her hips.", "9:16"),
    "Position - reverse cowgirl, leaning forward": ((("female", "the woman on top, facing away, leans forward"), ("male", "the man lies on his back")), "{B} lies on his back; {A} sits on him facing away and leans forward with her hands on his shins, the penis inside her, bouncing her hips back onto him in quick strokes, her rear high and toward the camera.", "9:16"),
    "Position - lotus": ((("female", "the woman in his lap, facing"), ("male", "the man sits")), "{B} sits cross-legged or on a chair; {A} sits on his lap facing him, arms around his neck and legs wrapped around his waist, erect penis in her vagina, grinding her hips against him. Their faces are close together.", "9:16"),
    "Position - seated reverse": ((("female", "the woman in his lap, facing away"), ("male", "the man sits")), "{B} sits on a chair or the floor; {A} sits on his lap facing away, the penis inside her, her back against his chest, his hands on her breasts, both moving her hips up and down.", "9:16"),
    "Position - seated on the edge of the bed": ((("female", "the woman sits at the edge"), ("male", "the man kneels in front")), "{A} sits on the edge of the bed, leaning back on her hands, legs open; {B} kneels on the floor in front of her and thrusts into {A}, his hands under her thighs holding them up.", "16:9"),
    "Position - edge of the bed": ((("female", "the woman lies at the edge"), ("male", "the man stands at the foot of the bed")), "{A} lies on her back at the edge of the bed with the legs hanging open; {B} stands at the foot of the bed between {A}'s legs and thrusts into {A}, holding her hips.", "16:9"),
    # intercourse, standing
    "Position - standing, face to face": ((("female", "the woman held up, arms around him"), ("male", "the man holds and lifts her")), "{A} and {B} stand facing each other; {B} holds {A} up by the thighs and ass, {A}'s arms around {B}'s neck and legs around {B}'s waist, erect penis in {A}'s vagina (not her mouth), {B} thrusting up into her.", "9:16"),
    "Position - standing, from behind": ((("female", "the woman bent forward"), ("male", "the man stands behind")), "{A} stands bent forward with hands on a wall or furniture, the legs apart; {B} stands behind, holds {A}'s hips and thrusts into {A} from behind, his body against her rear.", "9:16"),
    "Position - standing, one leg up": ((("female", "the woman with one leg raised"), ("male", "the man holds the leg up")), "{A} and {B} stand facing each other; {B} holds one of {A}'s legs up by the thigh and thrusts into {A} standing, her other hand on his shoulder for balance.", "9:16"),
    "Position - against the wall": ((("female", "the woman pinned against the wall"), ("male", "the man holds her up")), "{A} is pinned between {B} and a wall, {B}'s hands under {A}'s thighs holding her up, {A}'s legs around {B}'s waist and arms around {B}'s neck, erect penis in {A}, {B} thrusting up into her.", "9:16"),
    "Position - standing 69": ((("female", "the woman hangs upside down, receives oral"), ("male", "the man stands, holds her, gives oral")), "{B} stands and holds {A} upside down against his body, her legs over his shoulders and her vulva at his mouth; {B} licks and sucks her clitoris while {A} takes his erect penis into her mouth, both giving oral at once.", "9:16"),
    # intercourse, kneeling / from behind
    "Position - doggy style": ((("female", "the woman on hands and knees"), ("male", "the man kneels behind")), "{A} is on hands and knees, the back arched and the rear toward {B}; {B} kneels behind, holds {A}'s hips and thrusts into {A} from behind in long strokes.", "16:9"),
    "Position - doggy style, facing a mirror": ((("female", "the woman on hands and knees"), ("male", "the man kneels behind")), "{A} is on hands and knees facing a mirror, the rear toward {B}; {B} kneels behind, holds her hips and thrusts into her from behind while both watch the reflection and {A} watches {B}.", "16:9"),
    "Position - kneeling, pulled up by the hips": ((("female", "the woman kneels, pulled up"), ("male", "the man kneels behind")), "{B} kneels behind {A}; {A} is on her knees, the upper body pulled up against {B}'s chest, his arm across her ribs and the other hand on her breast, the penis inside her, short upward thrusts while she leans her head back on his shoulder.", "9:16"),
    "Position - leapfrog": ((("female", "the woman bent over, hands on the floor"), ("male", "the man stands behind")), "{A} stands bent fully forward with her hands on the floor, the legs straight and apart; {B} stands behind her and thrusts down into {A}, his hands on her hips, her breasts hanging.", "16:9"),
    "Position - wheelbarrow": ((("female", "the woman on her hands, legs held"), ("male", "the man behind, holds her legs")), "{A} is on her hands on the floor, {B} holds her legs at his hips like a wheelbarrow and thrusts into her from behind, her upper body low and her arms braced.", "16:9"),
    # anal
    "Position - anal, doggy style": ((("female", "the woman on hands and knees"), ("male", "the man kneels behind")), "{A} is on hands and knees, the back arched, the rear toward {B}; {B} kneels behind, guides the erect penis into her anus and thrusts in short, steady strokes, one thumb pressing her lower back.", "16:9"),
    "Position - anal, missionary": ((("female", "the woman on her back, hips raised"), ("male", "the man above, between her legs")), "{A} lies on her back with a pillow under the hips and the legs up; {B} lies above her and pushes the erect penis into her anus, slow deep strokes, her hands gripping his arms.", "16:9"),
    "Position - anal, spooning": ((("female", "the woman on her side"), ("male", "the man behind")), "{A} lies on her side with the top leg drawn up; {B} lies behind her, penis in her anus, one hand on her hip, thrusting slowly, both bodies close together.", "16:9"),
    "Position - anal, cowgirl": ((("female", "the woman on top, facing away"), ("male", "the man lies on his back")), "{B} lies on his back with an erect penis; {A} straddles him facing away and lowers herself onto the penis anally, then rides up and down in small, controlled strokes, her hands on his thighs.", "9:16"),
    # other
    "Position - intercrural, between the thighs": ((("female", "the woman in front"), ("male", "the man behind")), "{A} stands bent slightly forward or kneels with the legs together; {B} pushes the erect penis between {A}'s closed thighs from behind and thrusts, the shaft sliding under her vulva, his hands on her hips.", "16:9"),
    "Position - footjob": ((("female", "the woman sits, gives with the feet"), ("male", "the man sits or lies, receives")), "{B} sits or lies back with an erect penis; {A} sits facing him, braces her feet around the shaft and strokes it with the soles, the toes curling over the glans.", "16:9"),
    "Position - piledriver": ((("female", "the woman upside down on her shoulders"), ("male", "the man above")), "{A} lies on her back with the hips raised and the legs folded back over her body, shoulders and head on the bed; {B} stands or kneels above and thrusts down into {A}'s vagina, her ankles near his shoulders.", "9:16"),
    "Position - butterfly": ((("female", "the woman on her back, hips raised"), ("male", "the man kneels, holds her hips")), "{A} lies on her back with a pillow under the hips, the legs bent and open; {B} kneels close, lifts her hips with both hands and thrusts into {A} at an upward angle, her feet off the bed.", "16:9"),
    "Position - V": ((("female", "the woman on her back, legs wide"), ("male", "the man kneels in front")), "{A} lies on her back with both legs lifted and spread wide in a V; {B} kneels in front holding her ankles and thrusts into {A}, the camera facing the point where the bodies meet.", "16:9"),
    "Solo - female, sitting on the floor": ((("female", "the woman"),), "{A} sits on the floor with her back against a sofa, fully naked, knees up and apart, one hand on a breast and the other between her legs, fingers working the clitoris, the head tipped back.", "9:16"),
    "Solo - female, kneeling upright": ((("female", "the woman"),), "{A} kneels upright on the bed, fully naked, thighs apart, one hand sliding down her stomach into the pubic hair, the other cupping a breast, eyes on the camera.", "9:16"),
    "Solo - female, on all fours": ((("female", "the woman"),), "{A} is on hands and knees, fully naked, the back arched and the rear toward the camera, one hand reaching back between the legs, fingers on the vulva, the head turned to the lens.", "16:9"),
    "Solo - female, lying with a pillow": ((("female", "the woman"),), "{A} lies on her back with a pillow under the hips, fully naked, legs open, one hand spreading the labia and the other circling the clitoris, the face to the camera.", "16:9"),
    "Solo - female, two fingers inside": ((("female", "the woman"),), "{A} lies on her back, fully naked, two fingers pushed into the vagina and curling, the thumb on the clitoris, hips lifting into the hand, mouth open.", "16:9"),
    "Solo - female, toy on the clitoris": ((("female", "the woman"),), "{A} lies back with a vibrator pressed to the clitoris, fully naked, legs wide, the other hand gripping the bedding, the hips twitching with the vibration.", "16:9"),
    "Solo - female, toy from behind": ((("female", "the woman"),), "{A} is on all fours with a toy in the vagina, reaching back to thrust it, fully naked, the rear high toward the camera, the face turned over one shoulder.", "16:9"),
    "Solo - female, in the bath": ((("female", "the woman"),), "{A} lies back in a bath, water at her waist, breasts above the surface, fully naked, one hand between her legs under the water, the other on the bath edge, steam around her.", "16:9"),
    "Solo - female, sitting on a washing machine": ((("female", "the woman"),), "{A} sits on the edge of a washing machine, fully naked, legs apart, one hand braced behind her and the other between her legs, the vibration of the machine in her body.", "9:16"),
    "Solo - female, mirror, standing": ((("female", "the woman"),), "{A} stands facing a tall mirror, fully naked, one leg forward, watching her own reflection while one hand works between her legs and the other holds a breast.", "9:16"),
    "Solo - male, sitting, jerking off": ((("male", "the man"),), "{A} sits back on a sofa with the legs apart, fully naked, one hand stroking the erect penis base to glans, the other on his thigh, the head tipped back.", "16:9"),
    "Solo - male, lying, jerking off": ((("male", "the man"),), "{A} lies on his back on the bed, fully naked, one hand stroking the erect penis in long strokes, the other behind his head, the hips pushing up into the fist.", "16:9"),
    "Solo - male, standing in the shower": ((("male", "the man"),), "{A} stands under running water, one hand on the tiles, the other stroking the erect penis, water running over his chest and stomach.", "9:16"),
    "Solo - male, toy, on all fours": ((("male", "the man"),), "{A} is on all fours, fully naked, reaching back to push a toy into himself, the erect penis hanging, the head turned to the camera.", "16:9"),
    "Position - standing anal, from behind": ((("female", "the woman bent forward, hands on a wall"), ("male", "the man stands behind")), "{A} stands bent forward with both hands on a wall, the legs apart; {B} stands behind, spreads her rear with one hand and pushes the erect penis into her anus, the other hand on her hip, short slow strokes into the anus.", "16:9"),
    "Position - prone bone anal": ((("female", "the woman face down, hips raised"), ("male", "the man on top behind")), "{A} lies face down with a pillow under the hips, the rear raised; {B} lies over her from behind and thrusts the erect penis into her anus in short deep strokes, his chest on her back.", "16:9"),
    "Position - lotus anal": ((("female", "the woman in his lap, facing"), ("male", "the man sits")), "{B} sits cross-legged; {A} sits on his lap facing him with her arms around his neck, the erect penis in her anus, both grinding slowly, her mouth at his ear.", "9:16"),
    "Position - reverse cowgirl anal, lean forward": ((("female", "the woman on top, facing away, leans forward"), ("male", "the man lies on his back")), "{B} lies on his back; {A} straddles him facing away, lowers onto the penis anally and leans forward with her hands on his shins, pushing her hips back onto him in small strokes, the rear toward the camera.", "9:16"),
    "Position - spooning, top leg raised": ((("female", "the woman on her side, top leg raised"), ("male", "the man behind")), "{A} lies on her side with the top leg pulled up toward her chest; {B} lies along her back, one arm under her leg holding it up, and thrusts into her vagina slowly, both bodies pressed together.", "16:9"),
    "Position - mission, slow and deep": ((("female", "the woman on her back, hips lifted"), ("male", "the man above, chest to chest")), "{A} lies on her back with the hips lifted and the legs wrapped around {B}; {B} lies chest to chest and rolls his hips into her vagina in slow deep strokes, foreheads almost touching, her hands in his hair.", "16:9"),
    "Position - table, bent over": ((("female", "the woman bent over a table"), ("male", "the man stands behind")), "{A} bends over the edge of a table, forearms and chest on the surface, the legs apart; {B} stands behind, pulls her hips back onto his erect penis and thrusts into her vagina, her feet lifted off the floor.", "16:9"),
    "Position - squatting cowgirl": ((("female", "the woman squats over him"), ("male", "the man lies on his back")), "{B} lies on his back; {A} squats over him with her feet flat, lowers herself onto his erect penis and bounces in a deep squat, hands on his chest, thighs working.", "9:16"),
    "Position - sitting on his lap, grinding": ((("female", "the woman in his lap"), ("male", "the man sits")), "{B} sits on a chair; {A} sits astride his lap facing him, the penis inside her vagina, both hands on his shoulders, grinding her hips in slow circles instead of bouncing, faces close.", "9:16"),
    "Position - standing, her back arched": ((("female", "the woman stands, back arched against him"), ("male", "the man behind, holds her")), "{A} stands with her back arched against {B}'s chest, one arm up around his neck; {B} holds her waist and one breast and thrusts up into her vagina, both facing the camera.", "9:16"),
    "Position - edge of the bed, legs up": ((("female", "the woman on her back at the edge, legs up"), ("male", "the man stands between her legs")), "{A} lies at the edge of the bed with both legs raised and held open; {B} stands between her legs and thrusts down into her vagina at a deep angle, her hands gripping the sheets.", "16:9"),
    "Oral - blowjob, from below": ((("female", "the woman below, gives oral"), ("male", "the man above, receives")), "{B} stands over {A}; {A} lies on her back with the head tipped back over the edge of the bed and takes his erect penis into her mouth from below, his hands on her throat and hair.", "9:16"),
    "Oral - cunnilingus, from behind": ((("female", "the woman on all fours, receives oral"), ("male", "the man kneels behind, gives oral")), "{A} is on hands and knees, the rear toward {B}; {B} kneels behind, grips her hips and licks from the clitoris to the anus, tongue flat then pointed, her back arching.", "16:9"),
    "Hands - fingering, standing": ((("female", "the woman stands, receives"), ("male", "the man behind, gives")), "{A} stands with her back to {B}, one hand on a wall, legs apart; {B} presses against her from behind, one hand on her breast and the other between her legs, two fingers inside her vagina, the thumb on the clitoris.", "9:16"),
}
ACTION_CATEGORIES = ("Solo female", "Solo male", "Oral", "Hands", "Positions", "Anal", "Other")


PENETRATIVE_PREFIXES = ("Position",)
NON_PENETRATIVE = ("Position - footjob", "Position - intercrural, between the thighs", "Position - standing 69", "Position - scissoring")
PENETRATION_MODES = ("vaginal", "anal")


def penetration_modes(label):
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
    gender = ACTIONS[label][0][0][0]
    if label.startswith("Solo") and len(ACTIONS[label][0]) == 1:
        return "Solo male" if gender == "male" else "Solo female"
    if label.startswith("Oral"):
        return "Oral"
    if label.startswith("Hands"):
        return "Hands"
    if "anal" in label.lower():
        return "Anal"
    if label.startswith("Position"):
        return "Positions"
    return "Other"


CAMERAS = {
    "POV (the viewer is B)": "STRICT POV of {B}: the camera IS {B}'s eyes, at eye height, tilted down along {B}'s own body. {B}'s chest, stomach and hips stay visible across the lower third of the frame, and {B}'s hands, forearms or thighs may enter it. {A} stays at arm's length with the whole torso and hips in frame and the contact visible. {B}'s own genitals stay as they are. Never show {B}'s face (only in a mirror), never cut away from this angle. Keep it medium-wide, not a macro close-up.",
    "POV (the viewer is A)": "STRICT POV of {A}: the camera IS {A}'s eyes, at eye height, tilted down along {A}'s own body. {A}'s chest, stomach and hips stay visible across the lower third of the frame, and {A}'s hands, forearms or thighs may enter it. {B} stays at arm's length with the whole torso and hips in frame and the contact visible. {A}'s own genitals stay as they are. Never show {A}'s face (only in a mirror), never cut away from this angle. Keep it medium-wide, not a macro close-up.",
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
    return [role for _gender, role in ACTIONS[label][0]]


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
    return [{"gender": gender, "role": _role_text(role)} for gender, role in ACTIONS[label][0]]


def stages_for(label):
    """How many bodies the action needs: 1 (solo) or 2."""
    return len(ACTIONS[label][0])


def aspect(label):
    """Recommended frame shape for this arrangement."""
    return ACTIONS[label][2]


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
    counts = stages_for(label)
    genders = [gender for gender, _role in ACTIONS[label][0]]
    names = [_role_name(genders, index) for index in range(counts)]
    a, b = "Subject 1", "Subject 2"
    ratio = aspect(label)
    shape = "portrait" if ratio == "9:16" else "landscape"
    lines = [
        f"Action preset: {label}.",
        f"Action: {with_penetration(ACTIONS[label][1], penetration).format(A=a, B=b)}",
    ]
    camera = default_camera(label) if camera == ANY_CAMERA else camera
    if camera in CAMERAS and not (counts == 1 and camera in SOLO_CAMERAS):
        lines.append(f"Camera: {CAMERAS[camera].format(A=a, B=b)}")
        if camera.startswith("POV"):
            lines.append("The POV viewpoint holder's face never appears; their chest, stomach and hips stay in the lower third and the framing stays medium-wide, never a macro crop.")
        if camera.startswith("POV") and penetration_modes(label):
            lines.append("In POV the shaft comes in from the bottom edge: angle the hips and lift the shaft so it enters the named hole; the other hole stays empty and beside it.")
    lines += [
        "This action is exactly what happens, from the first second to the last; the notes only give the setting, the looks, the nudity and the mood.",
        "The act is already in progress in the first sentence: the genitals, hands or mouths are already in contact and moving; no setup.",
        f"Recommended aspect ratio: {ratio} ({shape}). Frame every body and face fully inside it and keep that framing for the whole clip.",
        "Write 550-700 words in the detailed description, one continuous shot covering the whole clip beat by beat.",
    ]
    for index, (role, who) in enumerate(zip(ACTIONS[label][0], subjects), start=1):
        lines.append(f"Subject {index} is {names[index - 1]} ({_role_text(role[1])}): {who}")
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
    if counts == 1:
        if subjects[1]:
            lines.append(f"Attached reference only, not in the video: Subject 2 ({subjects[1]}). Use it for anatomy, scale and detail.")
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
        if penetration == "anal":
            lines.append("PENETRATION: ANAL SEX. The penis is in the anus, not the vagina, for the whole clip. The anus is the lower opening and the shaft visibly enters it; the vagina stays empty and visible above, the strokes are slower and tighter.")
        else:
            lines.append("PENETRATION: VAGINAL SEX. The penis is in the vagina, not the anus, for the whole clip. The vagina is the upper opening, the anus is the lower one; the shaft visibly enters the upper opening, the labia stretch around it, the anus stays closed and empty below, and the shaft slides in and out of the vagina. Never move the penis into the anus.")
    lines.append("Use the pictures for identity and anatomy only: ignore their clothing, pose, lighting and background.")
    if "nudity" in notes.lower() or "naked" in notes.lower():
        if "striptease" in label.lower():
            lines.append("This is a striptease: Subject 1 starts dressed and ends naked; no clothing from the pictures remains once it is removed.")
        else:
            lines.append("Both subjects are fully naked from the first frame; no clothing from the pictures remains.")
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")
    return "\n".join(lines)
