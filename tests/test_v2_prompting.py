"""Writer routing and repair checks without loading language-model weights."""
import importlib.util
from pathlib import Path
import re
import unittest
from unittest.mock import patch


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("prompt_bootstrap", Path(__file__).parents[1] / "scripts/check_v2_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.runtime = module.bootstrap()
        cls.prompts = cls.runtime.pack.v2.prompting
        cls.contracts = cls.runtime.pack.v2.contracts
        cls.service = cls.runtime.pack
        import torch
        cls.torch = torch

    @classmethod
    def tearDownClass(cls):
        cls.runtime.loop.close()
        cls.runtime.temporary.cleanup()

    def test_affirmative_lint_and_in_scene_quotes(self):
        lint = self.prompts.validation_issues
        self.assertTrue(lint("A portrait. No cut off face. Never blurry."))
        self.assertTrue(lint("A person without malformed hands. Without blur. No clutter."))
        self.assertTrue(lint("I need to write a lovely scene."))
        self.assertFalse(lint('A sign reads "No entry". A person says "Do not leave."'))
        self.assertFalse(lint("[Shot 1] A person waves.\nnon_diegetic_music: N/A"))

    def test_writer_repair_and_gguf_dispatch(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a garden").args[0]
        outputs = ["I need to describe the garden first, so let me think about it.", "A person stands fully visible in a sunlit garden."]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", side_effect=outputs) as generate:
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={"source": "GGUF", "gguf_model": "local"}, max_tokens=256).args[0]
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args.kwargs["engine"], "gguf")
        self.assertEqual(generate.call_args.kwargs["max_tokens"], 256)
        self.assertEqual(result["artifacts"]["writer"]["status"], "valid")
        self.assertNotIn("text", pipe["artifacts"])

    def test_writer_preset_override_and_mismatch(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a garden").args[0]
        system = []
        def generate(**kwargs):
            system.append(kwargs["system_prompt"])
            return "A person stands fully visible in a sunlit garden."
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", side_effect=generate):
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, writer_preset="Anima Tags").args[0]
        self.assertIn(self.non_explicit("Anima Tags"), system[0])
        report = result["artifacts"]["writer"]
        self.assertEqual(report["preset"], "Anima Tags")
        self.assertIn("anima template", report["preset_mismatch"])
        self.assertEqual(result["settings"]["prompt"], pipe["settings"]["prompt"])

    def test_writer_auto_preset_stays_silent(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("anima"), source="idea", text="garden").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="sunlit garden, flowering plants"):
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={}).args[0]
        report = result["artifacts"]["writer"]
        self.assertEqual(report["preset"], "Anima Tags")
        self.assertNotIn("preset_mismatch", report)

    def test_h3_writer_preset_keeps_h3_layout(self):
        pipe = self.prompts.H3Prompt.execute(self.contracts.new_pipe("minimax_h3"), source="idea", text="a garden").args[0]
        replies = ["integrated_multimodal_description: [Shot 1] A tree sways in a breeze.\noverall_soundscape: Leaves rustle.\nnon_diegetic_music: N/A",
                   "overall_soundscape: Leaves rustle.\nnon_diegetic_music: N/A"]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", side_effect=replies) as generate:
            self.prompts.H3PromptWriter.execute(pipe, writer_model={}, writer_preset="Creative Rewrite")
        system = generate.call_args_list[0].kwargs["system_prompt"]
        self.assertIn(self.non_explicit("Creative Rewrite", prune=False), system)
        self.assertIn("integrated_multimodal_description", system)

    def test_exhausted_repairs_block_output(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("moss_image"), source="idea", text="portrait").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="I need to write a face description, so let me think about it first.") as generate:
            with self.assertRaisesRegex(ValueError, "encoding blocked"):
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="1")
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("text", pipe["artifacts"])

    def test_h3_reference_roles_are_counted_once(self):
        image = self.torch.zeros(1, 16, 16, 3)
        pipe = self.contracts.evolve(self.contracts.new_pipe("minimax_h3"), "references", {"mode": "i2v"},
                                     {"first_frame": image, "source_image": image, "reference_images": [image]})
        images, video, mode, context = self.prompts._writer_context(pipe, True)
        self.assertEqual((len(images), mode, context["picture_count"]), (1, "i2va", 1))

    def test_many_references_and_audio_keep_mapping(self):
        image = self.torch.zeros(1, 16, 16, 3)
        audio = {"waveform": self.torch.zeros(1, 2, 24000), "sample_rate": 24000}
        pipe = self.contracts.evolve(self.contracts.new_pipe("minimax_h3"), "references", {"mode": "ref2v"},
                                     {"reference_images": [image] * 5, "reference_audio": audio})
        images, video, mode, context = self.prompts._writer_context(pipe, True)
        self.assertEqual(len(images), 1)
        self.assertEqual(context["picture_count"], 5)
        self.assertTrue(context["contact_sheet"])
        self.assertEqual(context["audio"][0]["duration"], 1)

    def test_h3_short_structured_prompt_is_valid(self):
        context = {"picture_count": 0, "video_count": 0, "audio": []}
        text = "integrated_multimodal_description: [Shot 1] A tree sways in a breeze.\noverall_soundscape: Leaves rustle.\nnon_diegetic_music: N/A"
        self.assertEqual(self.prompts._h3_issues(text, "t2va", context), [])
        self.assertTrue(self.prompts._h3_issues(text.replace("A tree", "<Picture 7> A tree"), "t2va", context))

    def non_explicit(self, name, prune=True):
        """The preset as the writer sees it for a request that is not sexual."""
        text = self.service.PROMPTFORGE_PRESETS[name]["text"]
        text = self.prompts._SIZE_SECTION.sub("", self.prompts._EXPLICIT_SECTIONS.sub("", text))
        text = text.replace("whether it is worn, open, pulled aside or off", "and how it is worn")
        text = text.replace("and only the two or three props actually in frame", "and only the props the request names")
        text = text.replace("Start with: nsfw, anime style, 2d, flat colors.", "Start with: anime style, 2d, flat colors.").replace(self.prompts._RATING, 'Rating: "safe".')
        text = self.prompts._CAPABLE.sub("", text)
        return self.prompts._drop_unused_camera_rules(self.prompts._fit_to_idea(text, ""), "") if prune else text

    def test_auto_writer_sends_the_family_template_and_no_act_scaffolding_for_a_plain_idea(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a knight on a cliff").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="A knight stands on a cliff at sunset.") as generate:
            self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
        system = generate.call_args.kwargs["system_prompt"]
        self.assertIn(self.non_explicit("Krea 2 T2I"), system)
        self.assertNotIn("which part enters what", system)
        self.assertNotIn("pulled aside or off", system)
        self.assertNotIn("two or three props", system)
        self.assertIn("give them no other position", system)
        self.assertIn("covers a face or body", system)
        self.assertNotIn("turns away from the lens", system)
        self.assertNotIn("NSFW capable", system)
        self.assertIn("they are dressed", system)
        self.assertNotIn("Contact sheet", system)
        self.assertNotIn("excluded_from_scene", system)

    def test_h3_action_brief_keeps_the_arrangement_subject_text_and_role_names(self):
        text = self.prompts._action_text("h3: Position - spooning", ("a tall brunette", "a bearded man", ""), "", "auto")
        self.assertIn("lie on their sides", text)
        self.assertIn("a tall brunette", text)
        self.assertIn("a bearded man", text)
        self.assertIn("Only the woman and the man appear", text)
        self.assertNotIn("Subject 1", text)
        solo = self.prompts._action_text("h3: Solo - male, jerking off", ("a muscular man", "", ""), "", "auto")
        self.assertIn("a muscular man", solo)
        self.assertNotIn("the woman", solo)
        image = self.prompts._action_text("image: Sex - doggy style", ("", "", ""), "", "auto")
        self.assertNotIn("NOT her on her back", image)
        self.assertIn("The woman is BOTTOM on hands and knees. The man kneels behind her.", image)
        self.assertEqual(self.prompts._drop_negations("A woman stands. No clothing, no background. Her hand rests on her hip, not moving. Without props."),
                         "A woman stands. Her hand rests on her hip.")
        self.assertEqual(self.prompts._drop_negations("She remains still, no movement, no shift in posture. The camera frames her face without cropping. Fingers spread but not gripping, palms open."),
                         "She remains still. The camera frames her face. Fingers spread, palms open.")
        self.assertEqual(self.prompts._drop_negations("A bus shelter is visible, with no signs. A woman, age unspecified, stands. She is not moving."),
                         "A bus shelter is visible. A woman stands. She is not moving.")
        hug = self.prompts._action_text("image: Couple - back hug", ("", "", ""), "", "auto")
        self.assertNotIn("Camera stage", hug)
        self.assertNotIn("Camera position", hug)
        self.assertNotIn("point of contact", hug)
        self.assertNotIn("mattress", hug)
        self.assertNotIn("four arms", hug)
        self.assertIn("Write two separate bodies.", hug)
        self.assertNotIn("horizontal on a bed", hug)
        self.assertNotIn("horizontal on a bed", self.prompts._action_text("image: Sex - against the wall", ("", "", ""), "", "auto"))

    def test_anima_nsfw_tag_only_for_explicit_requests(self):
        marker = "Start with: nsfw, anime style, 2d, flat colors."
        for source, text, action, expected in (("idea", "a knight on a cliff", None, False), ("idea", "a nude woman lying on a bed", None, True),
                                               ("action", "", "Portrait - profile", False), ("action", "", "Sex - doggy style", True)):
            kwargs = {"action": "image: " + action} if action else {}
            pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("anima"), source=source, text=text, **kwargs).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="anime style, 2d, 1girl, standing") as generate:
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
            self.assertEqual(marker in generate.call_args.kwargs["system_prompt"], expected, (source, text, action))

    def test_a_reply_that_only_echoes_the_context_fails_the_attempt(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a garden").args[0]
        echo = '{"family": "krea2", "mode": "text_to_image", "references": []}'
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", side_effect=[echo, "A person stands in a sunlit garden holding a basket of flowers."]) as generate:
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="1").args[0]
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["artifacts"]["text"], "A person stands in a sunlit garden holding a basket of flowers.")

    def test_explicit_template_sections_only_reach_explicit_requests(self):
        def system(family, source, text="", action=None, h3=False):
            kwargs = {"action": ("h3: " if h3 else "image: ") + action} if action else {}
            node, writer = (self.prompts.H3Prompt, self.prompts.H3PromptWriter) if h3 else (self.prompts.ImagePrompt, self.prompts.ImagePromptWriter)
            pipe = node.execute(self.contracts.new_pipe(family), source=source, text=text, **kwargs).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
                try:
                    writer.execute(pipe, writer_model={}, retries="off")
                except ValueError:
                    pass
            return generate.call_args_list[0].kwargs["system_prompt"]
        for family, heading in (("krea2", "# Genital anatomy and penetration"), ("anima", "# Genitals")):
            self.assertNotIn(heading, system(family, "idea", "a knight on a cliff"))
            self.assertIn(heading, system(family, "idea", "a nude woman lying on a bed"))
            self.assertIn(heading, system(family, "action", action="Sex - doggy style"))
            self.assertIn("# Age (high priority)", system(family, "idea", "a knight on a cliff"))
        self.assertNotIn("# NSFW audio (mandatory)", system("minimax_h3", "idea", "A woman walks along a rainy street.", h3=True))
        self.assertIn("# NSFW audio (mandatory)", system("minimax_h3", "action", action="Position - spooning", h3=True))

    def test_anima_tags_drop_copied_negations_but_keep_booru_absence_tags(self):
        cleaned = self.prompts._clean_tags("1girl, no humans, no ornamental words, standing, without props, no panties, never blur, solo")
        self.assertEqual(cleaned, "1girl, no humans, standing, no panties, solo")

    def test_rating_and_cast_follow_the_request(self):
        def system(family, source, text="", action=None):
            kwargs = {"action": "image: " + action} if action else {}
            pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe(family), source=source, text=text, **kwargs).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
                try:
                    self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
                except ValueError:
                    pass
            return generate.call_args_list[0].kwargs["system_prompt"]
        self.assertIn('Rating: "safe".', system("anima", "idea", "a knight on a cliff"))
        self.assertIn('Rating: "nsfw" for nudity, "explicit" for sex acts.', system("anima", "action", action="Sex - doggy style"))
        solo = system("krea2", "action", action="Nude - on the back, arms above head")
        self.assertIn("Everyone in this scene is a woman", solo)
        self.assertNotIn("1056", solo)
        self.assertNotIn("glans", solo)
        self.assertIn("glans", system("krea2", "action", action="Sex - doggy style"))
        self.assertIn("show the format", system("krea2", "idea", "a knight on a cliff"))

    def test_body_size_section_only_when_the_request_names_a_size(self):
        def system(text):
            pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text=text).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
                try:
                    self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
                except ValueError:
                    pass
            return generate.call_args_list[0].kwargs["system_prompt"]
        self.assertNotIn("# Body size and scale", system("a nude woman lying on white sheets"))
        self.assertIn("# Body size and scale", system("a nude woman with narrow hips and small breasts"))

    def test_sfw_requests_ask_for_clothing_in_the_user_message_and_explicit_ones_do_not(self):
        def user(family, source, text="", action=None):
            kwargs = {"action": "image: " + action} if action else {}
            pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe(family), source=source, text=text, **kwargs).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
                try:
                    self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
                except ValueError:
                    pass
            return generate.call_args_list[0].kwargs["user_prompt"]
        self.assertIn("what each person wears", user("krea2", "action", action="Couple - back hug"))
        self.assertIn("Write a camera only when the brief gives one", user("krea2", "action", action="Couple - back hug"))
        self.assertIn("what each person wears", user("krea2", "idea", "two friends at a cafe"))
        self.assertIn("Tag every garment", user("anima", "idea", "two friends at a cafe"))
        self.assertIn("never a camera tag", user("anima", "action", action="Couple - back hug"))
        self.assertNotIn("wears", user("krea2", "action", action="Sex - doggy style"))
        self.assertNotIn("wears", user("krea2", "idea", "a nude woman on a bed"))

    def test_anima_tag_lists_drop_a_wrong_solo_and_name_conflicting_shot_sizes(self):
        self.assertEqual(self.prompts._clean_tags("anime style, 1girl, 1boy, solo, full body"), "anime style, 1girl, 1boy, full body")
        self.assertEqual(self.prompts._clean_tags("anime style, 1girl 1boy, solo, full body"), "anime style, 1girl 1boy, full body")
        self.assertEqual(self.prompts._clean_tags("anime style, 2girls, solo"), "anime style, 2girls")
        self.assertEqual(self.prompts._clean_tags("anime style, 1girl, solo, full body"), "anime style, 1girl, solo, full body")
        self.assertEqual(self.prompts._shot_conflict("1girl, full body, from side"), "")
        self.assertTrue(self.prompts._shot_conflict("1girl, close-up, portrait, full body").startswith("conflicting shot sizes (close-up, portrait, full body)"))

    def test_action_output_loses_genitals_and_cameras_the_brief_never_gave(self):
        brief = "Action: The woman lies between the legs with the penis in the mouth.\nCamera: Still camera at chest height."
        text = ("The woman lies between his legs, mouth around the penis. The man's scrotum hangs low. The woman's vulva is visible above the shaft, "
                "clitoris prominent. Camera is chest height. The bed is bare.")
        self.assertEqual(self.prompts._grounded(text, brief), "The woman lies between his legs, mouth around the penis. The man's scrotum hangs low. Camera is chest height. The bed is bare.")
        self.assertEqual(self.prompts._grounded("1girl, fellatio, pussy, labia, penis, glans, from side", brief, tags=True), "1girl, fellatio, penis, glans, from side")
        self.assertEqual(self.prompts._grounded("Camera over the shoulder. She smiles.", "Action: She smiles."), "She smiles.")
        self.assertEqual(self.prompts._grounded("Her vulva shows.", "Action: She stands."), "Her vulva shows.")
        self.assertEqual(self.prompts._grounded("The woman lies between his legs, mouth around the penis shaft, her vulva visible above it. The bed is bare.", brief),
                         "The woman lies between his legs, mouth around the penis shaft. The bed is bare.")
        self.assertEqual(self.prompts._grounded("The vulva, labia majora, and vaginal opening are visible as the shaft enters the mouth. She waits.", brief), "She waits.")
        self.assertEqual(self.prompts._grounded("Her clitoris is visible, and her mouth holds the penis.", brief), "Her mouth holds the penis.")
        self.assertEqual(self.prompts._grounded("Camera is low. She smiles.", "Action: She smiles. Camera from the foot of the bed."), "Camera is low. She smiles.")
        self.assertEqual(self.prompts._grounded("Camera is low. She smiles.", "Action: She stands with her back to the camera."), "She smiles.")

    def test_negative_wording_after_the_last_retry_is_used_with_a_warning_but_leakage_still_blocks(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a cat").args[0]
        blemished = "A grey cat sits on a mat. It does not blink. The tail is not long. The room has no door."
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value=blemished) as generate:
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="1").args[0]
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["artifacts"]["writer"]["status"], "valid")
        self.assertTrue(result["artifacts"]["writer"]["validation"][0].startswith("negative wording"))
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="Final answer: A cat."):
            with self.assertRaises(ValueError):
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="1")

    def test_anima_action_system_text_drops_the_act_lists_that_get_copied_into_the_tags(self):
        def system(source, action=None):
            kwargs = {"action": "image: " + action} if action else {}
            pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("anima"), source=source, text="a nude woman", **kwargs).args[0]
            with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
                try:
                    self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
                except ValueError:
                    pass
            return generate.call_args_list[0].kwargs["system_prompt"]
        acted = system("action", "Oral - lying blowjob")
        for gone in ("creampie", "Choreography", "penis in pussy"):
            self.assertNotIn(gone, acted)
        self.assertIn("The brief names the act", acted)
        idea = system("idea")
        self.assertIn("creampie", idea)
        self.assertIn("Choreography", idea)

    def test_h3_repair_keeps_each_section_once_and_age_echo_goes(self):
        text = ("integrated_multimodal_description:\n[Shot 1] A woman aged exactly as specified in the request walks.\n\n"
                "overall_soundscape:\nwaves from the picture call\n\nnon_diegetic_music:\nN/A\n\n"
                "integrated_multimodal_description:\n[Shot 1] A copy of the scene.\n\n"
                "overall_soundscape:\nsurf and gulls\n\nnon_diegetic_music:\nsoft cello")
        pipe = self.contracts.new_pipe("minimax_h3")
        fixed = self.prompts._h3_repair_sections(text, pipe, "t2v")
        for heading in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
            self.assertEqual(fixed.count(heading + ":"), 1, heading)
        self.assertIn("A woman aged exactly as specified in the request walks.", fixed)
        self.assertNotIn("A copy of the scene", fixed)
        unlabeled = self.prompts._h3_repair_sections("integrated_multimodal_description:\nLive-action, a woman walks.\n\noverall_soundscape:\nsurf", pipe, "t2v")
        self.assertIn("integrated_multimodal_description:\n[Shot 1] Live-action, a woman walks.", unlabeled)
        ref = ("subject_definitions:\n<Subject 1> is a woman.\n\nsummary:\nThe clip, see [Shot 1].\n\nretention_analysis:\n[Shot 1] keeps <Subject 1>.\n\n"
               "detailed_description:\n[Shot 1] She walks. [Shot 2] She stops.\n\noverall_soundscape:\nsurf\n\nnon_diegetic_music:\nN/A")
        fixed_ref = self.prompts._h3_repair_sections(ref, pipe, "ref2v")
        self.assertEqual(re.findall(r"\[Shot \d+\]", fixed_ref), ["[Shot 1]", "[Shot 2]"])
        self.assertTrue(fixed_ref.index("[Shot 1]") > fixed_ref.index("detailed_description:"))
        doubled = self.prompts._h3_repair_sections("integrated_multimodal_description:\n[Shot 1] [Shot 1] She walks.\n\noverall_soundscape:\nsurf", pipe, "t2v")
        self.assertEqual(doubled.count("[Shot 1]"), 1)
        inline = self.prompts._h3_repair_sections("integrated_multimodal_description: [Shot 1] She walks.\n\noverall_soundscape: surf", pipe, "t2v")
        self.assertEqual(re.findall(r"\[Shot \d+\]", inline), ["[Shot 1]"])
        odd = self.prompts._h3_repair_sections("integrated_multimodal_description:\n[SHOT 2: close] She walks. [Shot  3 ] She stops.\n\noverall_soundscape:\n[Shot 1] surf", pipe, "t2v")
        self.assertEqual(re.findall(r"\[Shot \d+\]", odd), ["[Shot 1]", "[Shot 2]"])
        self.assertNotIn("[Shot 1] surf", odd)
        self.assertEqual(self.prompts._H3_AGE_ECHO.sub("", "A woman aged exactly as specified in the request walks."), "A woman walks.")
        self.assertEqual(self.prompts._H3_AGE_ECHO.sub("", "A man, aged as specified in the request, sits."), "A man sits.")

    def test_pov_camera_text_only_mentions_a_vulva_for_penetrative_actions(self):
        library = self.prompts._library("h3_actions")
        oral = library.brief("Oral - blowjob, from the side", library.POV_A, ("a woman", "a man", ""), "")
        self.assertIn("STRICT POV", oral)
        for word in ("labia", "perineum"):
            self.assertNotIn(word, oral)
        penetrative = next(label for label in library.labels() if library.penetration_modes(label))
        vaginal = library.brief(penetrative, library.POV_B, ("a woman", "a man", ""), "", "vaginal")
        self.assertIn("runs straight into Subject 1's vulva: the labia part", vaginal)
        self.assertIn("runs straight into Subject 1's anus", library.brief(penetrative, library.POV_B, ("a woman", "a man", ""), "", "anal"))
        self.assertIn("which hole", vaginal)
        for label in ("Oral - blowjob, from the side", "Hands - handjob", "Solo - standing, still"):
            text = library.brief(label, library.ANY_CAMERA, ("a woman", "a man", ""), "")
            for gone in ("which hole", "hips drive", "drawn back until", "goes into Subject"):
                self.assertNotIn(gone, text, label)
        self.assertEqual(library.default_camera("Oral - blowjob, kneeling"), library.SIDE)
        self.assertIn(library.POV_A, library.cameras("Oral - blowjob, kneeling"))
        self.assertNotIn("already in progress", library.brief("Solo - standing, still", library.ANY_CAMERA, ("a woman", "", ""), ""))
        self.assertIn("one steady rhythm", library.brief("Hands - handjob", library.ANY_CAMERA, ("a woman", "a man", ""), ""))
        self.assertNotIn("vulva, vagina", library.brief("Oral - blowjob, from the side", library.ANY_CAMERA, ("a woman", "a man", ""), ""))
        self.assertIn("vulva, vagina", library.brief("Oral - cunnilingus", library.ANY_CAMERA, ("a woman", "a man", ""), ""))

    def test_h3_description_loses_genitals_the_action_never_involves(self):
        oral = ("Subject 1 kneels and takes the erect penis into the mouth.\nSubject 1 is female: never a penis; Subject 2 is male: erect penis and balls, never a vulva. "
                "Subject 1 never grows a penis and Subject 2 never gets a vagina.")
        text = ("integrated_multimodal_description:\n[Shot 1] She takes the shaft into her mouth, her vulva exposed and labia parted. The man stands still.\n\n"
                "overall_soundscape:\nHer vulva is wet, the sound is soft.")
        fixed = self.prompts._grounded_h3(text, oral)
        self.assertIn("She takes the shaft into her mouth. The man stands still.", fixed)
        self.assertNotIn("labia", fixed)
        self.assertIn("overall_soundscape:\nHer vulva is wet", fixed)
        involved = oral.replace("female: never a penis", "female: vulva, vagina and clitoris, never a penis")
        self.assertEqual(self.prompts._grounded_h3(text, involved).count("vulva"), 2)

    def test_h3_runaway_description_is_flagged_but_never_blocks(self):
        long_text = "integrated_multimodal_description:\n[Shot 1] " + "She moves. " * 800 + "\n\noverall_soundscape:\nsoft\n\nnon_diegetic_music:\nN/A"
        issues = self.prompts._h3_issues(long_text, "t2va", {"picture_count": 0, "video_count": 0, "audio": []})
        self.assertTrue(any(issue.startswith("description too long") for issue in issues))
        short = long_text.replace("She moves. " * 800, "She moves. " * 50)
        self.assertEqual(self.prompts._h3_issues(short, "t2va", {"picture_count": 0, "video_count": 0, "audio": []}), [])

    def test_a_free_hand_is_not_negative_wording(self):
        self.assertEqual(self.prompts.validation_issues("Her left hand rests on the knee, the other hand free, fingers loose, eyes on the lens, hair down.", "krea2"), [])
        self.assertTrue(self.prompts._NEGATIVE.search("a watermark-free picture"))

    def test_penetrative_motion_never_asks_for_the_shaft_to_leave_and_a_pull_out_in_the_text_is_flagged(self):
        library = self.prompts._library("h3_actions")
        penetrative = next(label for label in library.labels() if library.penetration_modes(label))
        text = library.brief(penetrative, library.ANY_CAMERA, ("a woman", "a man", ""), "")
        self.assertNotIn("only the glans touches the entrance", text)
        self.assertIn("never leaves the named hole", text)
        self.assertTrue(self.prompts._LEAVES_THE_BODY.search("He pulls out completely, then pushes back in."))
        self.assertTrue(self.prompts._LEAVES_THE_BODY.search("The shaft slips out of her."))
        self.assertTrue(self.prompts._LEAVES_THE_BODY.search("His penis leaves her vagina for a moment."))
        self.assertFalse(self.prompts._LEAVES_THE_BODY.search("He pushes in and out in short strokes, the glans staying inside her."))

    def test_male_anatomy_anchor_covers_every_action_with_a_man_and_scrotum_as_vulva_is_flagged(self):
        library = self.prompts._library("h3_actions")
        for label in ("Oral - blowjob, from the side", "Hands - handjob", "Solo - male, jerking off", "Position - doggy style"):
            text = library.brief(label, library.ANY_CAMERA, ("a woman", "a man", ""), "")
            self.assertIn("one closed sac with a centre seam", text, label)
        search = self.prompts._SCROTUM_AS_VULVA.search
        self.assertTrue(search("His scrotum has a slit down the middle like a vulva."))
        self.assertTrue(search("the balls parted into labia"))
        self.assertFalse(search("His scrotum slaps against her vulva with every stroke."))
        self.assertFalse(search("His balls swing between her thighs, her vulva spread."))

    def test_a_long_idea_gets_scaled_limits_a_clear_request_frame_and_a_dropped_facts_report(self):
        idea = ("the same exact girl from the provided image but now fully naked, petite uni student young adult girl that is cute with small breasts bent over a desk in a classroom "
                "full of students, she is naked and being taken from behind by her teacher while panting and blushing from embarrassment, her toes barely reaching the ground and curling, "
                "other students staring at the girl and the teacher having sex, class in session, realistic size difference between the girl and the professor with him being older")
        edit = self.service.PROMPTFORGE_PRESETS["Krea 2 Edit"]["text"]
        fitted = self.prompts._fit_to_idea(edit, idea)
        self.assertIn("usually 100-", fitted); self.assertNotIn("100-200 words", fitted)
        self.assertNotIn("two or three props", fitted); self.assertNotIn("four arms, four legs", fitted)
        self.assertIn("small whole figures", fitted)
        short_idea = self.prompts._fit_to_idea(edit, "a girl on a bed")
        self.assertIn("usually 100-280 words, more when the request has more facts", short_idea)
        self.assertIn("four arms, four legs", short_idea)
        short = "The same person as the photo: petite build, long dark hair. She is nude on a desk, a man behind her, soft light, three-quarter view."
        missing = self.prompts._dropped_facts(idea, short)
        for word in ("classroom", "students", "blushing", "toes", "panting"):
            self.assertIn(word, missing)
        self.assertNotIn("naked", missing)
        full = short + " The classroom of students stares, she is blushing and panting, her toes curling, her teacher behind her."
        self.assertEqual([w for w in self.prompts._dropped_facts(idea, full) if w in ("classroom", "students", "blushing", "toes", "panting", "teacher")], [])
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text=idea).args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value=short) as generate:
            result = self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="1").args[0]
        self.assertEqual(generate.call_count, 1)                      # measured, never retried
        self.assertIn("classroom", result["artifacts"]["writer"]["attempts"][0]["dropped_facts"])
        user = generate.call_args_list[0].kwargs["user_prompt"]
        self.assertTrue(user.startswith("Write the prompt for this request."))
        self.assertIn("Request:\n" + idea, user)

    def test_unused_camera_rules_are_dropped_and_a_long_idea_gets_a_facts_first_block(self):
        edit = self.service.PROMPTFORGE_PRESETS["Krea 2 Edit"]["text"]
        plain = self.prompts._drop_unused_camera_rules(edit, "a girl on a bed")
        self.assertNotIn("POV means the camera IS", plain)
        self.assertNotIn("Over-the-shoulder keeps", plain)
        self.assertIn("POV means the camera IS", self.prompts._drop_unused_camera_rules(edit, "a POV shot of a girl"))
        self.assertIn("Over-the-shoulder keeps", self.prompts._drop_unused_camera_rules(edit, "seen from behind"))
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a classroom full of students and a blushing girl at a desk").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="x") as generate:
            try:
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, retries="off")
            except ValueError:
                pass
        system = generate.call_args_list[0].kwargs["system_prompt"]
        self.assertIn("# Priority", system)
        self.assertIn("never the request", system)
        self.assertIn("a guide, not a limit", system)
        self.assertNotIn("Hard limit", system)
        self.assertNotIn("POV means the camera IS", system)
        self.assertIn("100-280", self.service.PROMPTFORGE_PRESETS["Krea 2 Edit"]["text"])
        self.assertEqual(self.service.OUTPUT_CAPS["krea 2 edit"], 340)



if __name__ == "__main__":
    unittest.main()
