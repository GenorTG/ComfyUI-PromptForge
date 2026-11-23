"""CPU H3 integration fixtures; run in a fresh process with ComfyUI's Python.

python -m unittest discover -s tests -p test_v2_h3.py -v
Real native conditioning is exercised with tiny encoder doubles. Sampling and
VAE decode are replaced only for sequence control-flow tests, not quality claims.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class H3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("h3_runtime_bootstrap", Path(__file__).parents[1] / "scripts/check_v2_runtime.py")
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        cls.runtime = bootstrap.bootstrap()
        cls.h3 = cls.runtime.pack.v2.h3
        cls.contracts = cls.runtime.pack.v2.contracts
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.runtime.loop.close()
        cls.runtime.temporary.cleanup()

    def pipe(self):
        torch = self.torch
        class Clip:
            def tokenize(self, text, **kwargs):
                return text
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.ones(1, 2, 3), {}]]
        class VideoVae:
            def encode(self, images):
                return torch.zeros(1, 24, 2, images.shape[1] // 16, images.shape[2] // 16)
        audio_vae = SimpleNamespace(audio_sample_rate=48000, encode=lambda wave: torch.zeros(1, 32, 2, 8))
        pipe = self.h3.H3Format.execute(self.contracts.new_pipe("minimax_h3"), width=32, height=32, seconds=.2).args[0]
        return self.contracts.evolve(pipe, section="prompt", settings=dict(source="raw", negative=""), artifacts=dict(
            text="subject_definitions: a tree", clip=Clip(), vae=VideoVae(), audio_vae=audio_vae))

    def decoded(self, pipe, value=0):
        n = pipe["settings"]["format"]["frames"]
        return self.contracts.evolve(pipe, artifacts=dict(images=self.torch.full((n, 32, 32, 3), float(value)),
            audio=dict(waveform=self.torch.full((1, 2, round(n / 24 * 48000)), float(value)), sample_rate=48000)))

    def test_format_native_grid_and_branch_isolation(self):
        base = self.contracts.new_pipe("minimax_h3")
        result = self.h3.H3Format.execute(base, seconds=5, aspect="16:9", megapixels=1, orientation="portrait").args[0]
        fmt = result["settings"]["format"]
        self.assertGreater(fmt["height"], fmt["width"])
        self.assertEqual((fmt["width"] % 32, fmt["height"] % 32, fmt["frames"] % 17), (0, 0, 5))
        self.assertGreaterEqual(fmt["seconds"], 5)
        self.assertEqual(base["settings"], {})
        with self.assertRaisesRegex(ValueError, "family"):
            self.h3.H3Format.execute(self.contracts.new_pipe("krea2"))

    def test_all_six_native_conditioning_modes(self):
        image = self.torch.zeros(1, 32, 32, 3)
        for mode in self.h3.MODES:
            with self.subTest(mode=mode):
                pipe = self.h3.H3References.execute(self.pipe(), mode=mode, first_frame=image,
                    last_frame=image, reference_video=image.repeat(5, 1, 1, 1)).args[0]
                encoded = self.h3.H3Encode.execute(pipe).args[0]
                self.assertTrue(self.contracts.require(encoded, "latent")["samples"].is_nested)
                self.assertNotIn("latent", pipe["artifacts"])

    def test_encode_requires_valid_writer_for_enhanced_sources(self):
        for source in ("idea", "action"):
            for status in (None, "raw", "failed"):
                with self.subTest(source=source, status=status):
                    pipe = self.contracts.evolve(self.pipe(), section="prompt", settings={"source": source},
                        artifacts={"writer": {"status": status}})
                    with patch.object(self.h3.core.MiniMaxH3ImageToVideo, "execute") as encode:
                        with self.assertRaisesRegex(ValueError, "prompt writer successfully"):
                            self.h3.H3Encode.execute(pipe)
                        encode.assert_not_called()
            valid = self.contracts.evolve(pipe, artifacts={"writer": {"status": "valid"}})
            self.assertIn("positive", self.h3.H3Encode.execute(valid).args[0]["artifacts"])
        raw = self.contracts.evolve(self.pipe(), artifacts={"writer": {"status": "failed"}})
        self.assertIn("positive", self.h3.H3Encode.execute(raw).args[0]["artifacts"])

    def test_join_removes_overlap_and_crossfades_audio_on_same_clock(self):
        left = self.decoded(self.pipe(), 0)
        right = self.decoded(self.pipe(), 1)
        for transition in ("cut", "crossfade"):
            joined = self.h3.H3Join.execute(right, previous=left, overlap_frames=2, transition=transition, match_levels=False).args[0]
            self.assertEqual(joined["artifacts"]["images"].shape[0], 8)
            self.assertEqual(joined["artifacts"]["audio"]["waveform"].shape[-1], 16000)
            self.assertEqual(left["artifacts"]["images"].shape[0], 5)
            if transition == "crossfade":
                self.assertEqual(joined["artifacts"]["images"][3, 0, 0, 0].item(), 0)
                self.assertEqual(joined["artifacts"]["images"][4, 0, 0, 0].item(), 1)

    def test_join_source_resamples_video_and_supplies_silence(self):
        pipe = self.contracts.evolve(self.decoded(self.pipe()), artifacts=dict(
            continue_video=self.torch.zeros(30, 32, 32, 3), continue_fps=30))
        result = self.h3.H3Join.execute(pipe, use_source=True).args[0]
        self.assertEqual(result["artifacts"]["images"].shape[0], 29)
        self.assertEqual(result["artifacts"]["audio"]["waveform"].shape[-1], 58000)

    def test_sequence_limit_resume_raw_writer_bypass_and_changed_plan_rejection(self):
        pipe = self.h3.H3SamplingSettings.execute(self.pipe()).args[0]
        pipe = self.h3.H3Parts.execute(pipe, '[{"text":"First tree","seconds":0.2},{"text":"Second tree","seconds":0.2}]').args[0]
        self.assertEqual(self.h3.H3Sequence.check_lazy_status(pipe), [])
        with patch.object(self.h3, "sample", side_effect=lambda p, **kw: p), patch.object(self.h3, "decode", side_effect=lambda p, **kw: self.decoded(p)), patch.object(self.runtime.pack.v2.h3_sequence.H3PromptWriter, "execute", side_effect=AssertionError("raw writer ran")):
            first = self.h3.H3Sequence.execute(pipe, max_parts=1).args[0]
            second = self.h3.H3Sequence.execute(pipe, resume=first).args[0]
        self.assertEqual(second["artifacts"]["writer"]["status"], "raw")
        self.assertTrue(all(part["artifacts"]["writer"]["status"] == "raw" for part in second["artifacts"]["sequence_parts"]))
        self.assertEqual(second["artifacts"]["sequence_next"], 2)
        self.assertEqual(len(second["artifacts"]["sequence_parts"]), 2)
        self.assertEqual(second["settings"]["sampling"], pipe["settings"]["sampling"])
        changed = self.h3.H3Parts.execute(pipe, '[{"text":"Different tree"}]').args[0]
        with self.assertRaisesRegex(ValueError, "Resume"):
            self.h3.H3Sequence.execute(changed, resume=first)

    def test_sequence_idea_action_writer_context_and_lazy_dependency(self):
        from comfy_api.latest import io
        pipe = self.h3.H3SamplingSettings.execute(self.pipe()).args[0]
        pipe = self.h3.H3Parts.execute(pipe, '[{"source":"idea","text":"First tree","seconds":0.2},{"source":"action","text":"Tree sways","seconds":0.2}]').args[0]
        self.assertEqual(self.h3.H3Sequence.check_lazy_status(pipe), ["writer_model"])
        seen = []
        def writer(p, **kwargs):
            seen.append(p)
            return io.NodeOutput(self.contracts.evolve(p, artifacts=dict(text="subject_definitions: written tree", writer={"status": "valid"})))
        with patch.object(self.h3, "sample", side_effect=lambda p, **kw: p), patch.object(self.h3, "decode", side_effect=lambda p, **kw: self.decoded(p)), patch.object(self.runtime.pack.v2.h3_sequence.H3PromptWriter, "execute", side_effect=writer):
            result = self.h3.H3Sequence.execute(pipe, writer_model={"local": True}).args[0]
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["settings"]["references"]["mode"], "i2v")
        self.assertIn("first_frame", seen[1]["artifacts"])
        self.assertNotIn("reference_audio", seen[1]["artifacts"])
        self.assertIn("Previous part", seen[1]["settings"]["prompt"]["text"])
        self.assertEqual(result["artifacts"]["sequence_next"], 2)

    def test_refine_reencodes_keyframes_on_upscaled_grid(self):
        pipe = self.h3.H3References.execute(self.pipe(), mode="i2v", frame_fit="crop to workflow aspect", first_frame=self.torch.zeros(1, 32, 32, 3)).args[0]
        pipe = self.h3.H3Encode.execute(pipe).args[0]
        latent = self.h3.core.EmptyMiniMaxH3LatentAV.execute(64, 64, 5).args[0]
        positive = self.h3.refresh_keyframes(pipe, pipe["artifacts"]["positive"], latent)
        self.assertEqual(positive[0][1]["minimax_keyframes"][0]["latent"].shape[-2:], (4, 4))
        self.assertEqual(pipe["artifacts"]["positive"][0][1]["minimax_keyframes"][0]["latent"].shape[-2:], (2, 2))

    def test_quality_refine_schedule_and_zero_shift_semantics(self):
        from comfy_api.latest import io
        model = SimpleNamespace(get_model_object=lambda key: SimpleNamespace(shift=8.0, audio_shift=2.0))
        pipe = self.contracts.evolve(self.pipe(), artifacts=dict(model=model, positive=[], negative=[], latent={"samples": self.torch.ones(1)}))
        with patch.object(self.h3.core.MiniMaxH3SigmaShift, "execute", return_value=io.NodeOutput(model)) as shift, patch.object(self.h3.nodes_custom_sampler.SamplerCustom, "execute", return_value=io.NodeOutput({"samples": self.torch.ones(1)})) as dispatch:
            self.h3.sample(pipe, steps=3, sampler_name="er_sde", scheduler="linear from start", start_sigma=.9035, shift_video=0, shift_audio=0)
            shift.assert_not_called()
            sigmas = dispatch.call_args.args[7]
            self.torch.testing.assert_close(sigmas, self.torch.tensor([.9035, .9035 * 1.0486 * 2 / 3, .9035 * 1.0486 / 3, 0]))
            self.h3.sample(pipe, steps=3, scheduler="linear from start", start_sigma=.9035, shift_video=0, shift_audio=3)
            self.assertEqual(shift.call_args.args[1:], (8.0, 3))

    def test_disabled_stages_do_not_load_or_sample(self):
        pipe = self.contracts.evolve(self.pipe(), artifacts={"latent": {"samples": self.torch.ones(1)}})
        pipe = self.h3.H3SamplingSettings.execute(pipe).args[0]
        with patch.object(self.h3, "sample", side_effect=AssertionError("sample ran")):
            self.assertIs(self.h3.H3Refine.execute(pipe, enabled=False).args[0], pipe)
        with patch.object(self.h3.nodes_hunyuan.LatentUpscaleModelLoader, "execute", side_effect=AssertionError("loader ran")):
            self.assertIs(self.h3.H3LatentUpscale.execute(pipe, enabled=False).args[0], pipe)

    def test_manager_sockets_use_one_native_video_clock_and_names_filter(self):
        frames = self.torch.zeros(30, 32, 32, 3)
        refs = self.runtime.pack.v2.refs.ReferenceManager
        selected = refs.execute(self.pipe(), extra_video_fps=30.0, extra_videos={"item_0": frames},
                                extra_images={"item_0": self.torch.zeros(2, 32, 32, 3)}).args[0]
        artifacts = selected["artifacts"]
        self.assertEqual(artifacts["reference_video_fps"], 24)
        self.assertEqual(artifacts["reference_video"].shape[0], 24)
        self.assertIs(artifacts["reference_video"], artifacts["reference_videos"][0])
        self.assertEqual(len(artifacts["reference_images"]), 2)
        self.assertEqual(selected["settings"]["references"]["selected_names"], ["image 1", "image 2", "video 1"])
        only = refs.execute(self.pipe(), names="video 1", extra_videos={"item_0": frames}, extra_images={"item_0": self.torch.zeros(2, 32, 32, 3)}).args[0]
        self.assertEqual(only["artifacts"]["reference_images"], [])
        with self.assertRaisesRegex(ValueError, "Unknown references"):
            refs.execute(self.pipe(), names="missing")
        selected = self.h3.H3References.execute(selected, mode="continue").args[0]
        self.assertIn("positive", self.h3.H3Encode.execute(selected).args[0]["artifacts"])

    def test_join_levels_and_crossfades_the_seam_instead_of_stepping(self):
        left, right = self.decoded(self.pipe(), .3), self.decoded(self.pipe(), .4)
        matched = self.h3.join(left, right, 4)["artifacts"]["images"]
        stepped = self.h3.join(left, right, 4, "cut", False)["artifacts"]["images"]
        self.assertLess((matched[1:] - matched[:-1]).abs().max(), .02)
        self.assertGreater((stepped[1:] - stepped[:-1]).abs().max(), .09)
        independent = self.h3.join(left, right, 0)["artifacts"]["images"]
        self.assertLess((independent[1:] - independent[:-1]).abs().max(), .02)

    def test_beta57_uses_native_training_resolution_without_registration(self):
        from comfy_api.latest import io
        native = SimpleNamespace(sigmas=self.torch.linspace(0, 1, 10001))
        model = SimpleNamespace(get_model_object=lambda key: native)
        pipe = self.contracts.evolve(self.pipe(), artifacts=dict(model=model, positive=[], negative=[], latent={"samples": self.torch.ones(1)}))
        schedulers = list(self.h3.comfy.samplers.KSampler.SCHEDULERS)
        expected = self.h3.comfy.samplers.beta_scheduler(native, 8, alpha=.5, beta=.7)
        with patch.object(self.h3.nodes_custom_sampler.SamplerCustom, "execute", return_value=io.NodeOutput({"samples": self.torch.ones(1)})) as dispatch:
            self.h3.sample(pipe, steps=8, scheduler="beta57", shift_video=0, shift_audio=0)
            self.torch.testing.assert_close(dispatch.call_args.args[7], expected)
        self.assertEqual(list(self.h3.comfy.samplers.KSampler.SCHEDULERS), schedulers)
        combo = next(item for item in self.h3.H3SamplingSettings.define_schema().inputs if item.id == "scheduler")
        self.assertIn("beta57", combo.options)

    def test_decode_audio_comes_from_the_chosen_pass(self):
        torch = self.torch
        second, first = torch.full((1,), 2.0), torch.full((1,), 1.0)
        latent = {"samples": SimpleNamespace(unbind=lambda: (torch.zeros(1), second))}
        pipe = self.contracts.evolve(self.pipe(), artifacts={"latent": latent, "first_pass_audio": first})
        frames = pipe["settings"]["format"]["frames"]
        seen = []
        def audio(vae, latent):
            seen.append(latent["samples"])
            return {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
        decoder = SimpleNamespace(decode=lambda *args: (torch.zeros(frames, 32, 32, 3),))
        with patch.object(self.h3.minimax_decode, "PROMPTFORGE_MiniMaxDecode", return_value=decoder), patch.object(self.h3.nodes_audio, "vae_decode_audio", side_effect=audio):
            self.h3.decode(pipe)
            self.h3.decode(pipe, audio_from="first pass")
            self.h3.decode(self.contracts.evolve(pipe, invalidate=("first_pass_audio",)), audio_from="first pass")
        self.assertEqual([float(sample) for sample in seen], [2.0, 1.0, 2.0])

    def test_first_pass_preview_is_optional_and_keeps_original_pipe(self):
        from comfy_api.latest import io
        pipe = self.contracts.evolve(self.pipe(), artifacts={"latent": {"samples": self.torch.ones(1)}})
        with patch.object(self.h3, "decode", side_effect=AssertionError("disabled preview decoded")), patch.object(self.h3.H3PreviewSave, "execute", side_effect=AssertionError("disabled preview saved")):
            self.assertIs(self.h3.H3FirstPassPreview.execute(pipe).args[0], pipe)
        decoded = self.decoded(pipe)
        with patch.object(self.h3, "decode", return_value=decoded) as decode, patch.object(self.h3.H3PreviewSave, "execute", return_value=io.NodeOutput(decoded, ui={"preview": True})) as save:
            output = self.h3.H3FirstPassPreview.execute(pipe, enabled=True)
            self.assertIs(output.args[0], pipe)
            self.assertEqual(output.ui, {"preview": True})
            decode.assert_called_once()
            self.assertTrue(save.call_args.kwargs["save"])
            self.assertNotIn("images", pipe["artifacts"])

    def test_real_cpu_preview_encodes_audio_and_video(self):
        result = self.h3.H3PreviewSave.execute(self.decoded(self.pipe()), filename_prefix="h3_fixture").args[0]
        self.assertEqual(self.contracts.require(result, "video").get_dimensions(), (32, 32))


if __name__ == "__main__":
    unittest.main()
