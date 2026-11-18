"""Writer routing and repair checks without loading language-model weights."""
import importlib.util
from pathlib import Path
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
        self.assertTrue(lint("A portrait with no cut off face."))
        self.assertTrue(lint("A person without malformed hands."))
        self.assertTrue(lint("I need to write a lovely scene."))
        self.assertFalse(lint('A sign reads "No entry". A person says "Do not leave."'))
        self.assertFalse(lint("[Shot 1] A person waves.\nnon_diegetic_music: N/A"))

    def test_writer_repair_and_gguf_dispatch(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("krea2"), source="idea", text="a garden").args[0]
        outputs = ["A garden without clutter.", "A person stands fully visible in a sunlit garden."]
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
        self.assertIn(self.service.PROMPTFORGE_PRESETS["Anima Tags"]["text"], system[0])
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
        self.assertIn(self.service.PROMPTFORGE_PRESETS["Creative Rewrite"]["text"], system)
        self.assertIn("integrated_multimodal_description", system)

    def test_exhausted_repairs_block_output(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("moss_image"), source="idea", text="portrait").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="A face without distortion.") as generate:
            with self.assertRaisesRegex(ValueError, "encoding blocked"):
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, attempts=2)
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

    def test_word_budget_repairs_instead_of_truncating(self):
        pipe = self.prompts.ImagePrompt.execute(self.contracts.new_pipe("anima"), source="idea", text="garden").args[0]
        with patch.object(self.service.PROMPTFORGE_PromptEnhancer, "_generate", return_value="sunlit garden, flowering plants, stone path"):
            with self.assertRaisesRegex(ValueError, "word budget"):
                self.prompts.ImagePromptWriter.execute(pipe, writer_model={}, max_words=3, attempts=1)


if __name__ == "__main__":
    unittest.main()
