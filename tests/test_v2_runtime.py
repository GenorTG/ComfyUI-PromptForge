"""Run in a fresh process with ComfyUI's Python; see scripts/check_v2_runtime.py."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pf_runtime_bootstrap", Path(__file__).parents[1] / "scripts/check_v2_runtime.py")
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        cls.runtime = bootstrap.bootstrap()
        cls.v2 = cls.runtime.pack.v2
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.runtime.loop.close()
        cls.runtime.temporary.cleanup()

    def pipe(self, family="krea2", **artifacts):
        return self.v2.contracts.evolve(self.v2.contracts.new_pipe(family), artifacts=artifacts)

    def clip(self):
        torch = self.torch
        class Clip:
            def __init__(self):
                self.calls = []
            def tokenize(self, text, **kwargs):
                self.calls.append((text, kwargs))
                return text
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.ones(1, max(1, len(tokens.split())), 64), {}]]
        return Clip()

    def test_no_text_widget_is_rewritten_by_dynamic_prompts(self):
        # the browser strips braces and // comments from such widgets, which destroys JSON state
        for key, node in self.runtime.pack.NODE_CLASS_MAPPINGS.items():
            if key.startswith("PROMPTFORGE_V2_"):
                flagged = [item.id for item in node.define_schema().inputs if getattr(item, "dynamic_prompts", None)]
                self.assertEqual(flagged, [], key)

    def test_h3_templates_sample_the_pipe_model(self):
        # a model socket after Performance / Live Preview would silently replace their patched model
        stages = {"PROMPTFORGE_V2_H3_ENCODE", "PROMPTFORGE_V2_H3_SAMPLE", "PROMPTFORGE_V2_H3_REFINE"}
        templates = sorted((Path(__file__).parents[1] / "example_workflows/v2").glob("h3_*.api.json"))
        self.assertTrue(templates)
        for path in templates:
            wired = [node["class_type"] for node in json.loads(path.read_text()).values() if node["class_type"] in stages and "model" in node["inputs"]]
            self.assertEqual(wired, [], path.name)

    def test_reference_focus_composes_and_crops(self):
        import torch
        focus = self.v2.focus
        image = torch.ones(1, 100, 80, 3) * 0.2
        mask = torch.zeros(1, 100, 80); mask[:, 20:60, 30:50] = 1
        out, cut = focus.compose(image, mask, "grey", True, 0.25)
        self.assertEqual(tuple(out.shape[1:3]), (60, 30))
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.5); self.assertAlmostEqual(float(out[0, 30, 15, 0]), 0.2)
        kept, _ = focus.compose(image, mask, "keep", False, 0.25)
        self.assertTrue(torch.equal(kept, image))

    def test_reference_focus_is_off_unless_a_row_asks(self):
        with patch.object(self.v2.focus, "apply", side_effect=AssertionError("focus ran")):
            self.assertEqual([item.id for item in self.v2.media.LoadImage.define_schema().inputs][6], "focus")
            self.assertEqual(self.v2.media.LoadImage.define_schema().inputs[6].default, "")

    def test_registration_through_real_init(self):
        from comfy_api.latest import io
        mappings = self.v2.NODE_CLASS_MAPPINGS
        self.assertGreaterEqual(len(mappings), 49)
        expected = {"PROMPTFORGE_V2_IMAGE_" + name for name in
                    ("MODEL", "FORMAT", "SAMPLING_SETTINGS", "ENCODE", "SAMPLE", "DECODE", "SAVE")}
        expected.update(("PROMPTFORGE_V2_KREA_EDIT_PREPARE", "PROMPTFORGE_V2_SUBJECT_MASK",
                         "PROMPTFORGE_V2_PERFORMANCE", "PROMPTFORGE_V2_PREVIEW_SETTINGS", "PROMPTFORGE_V2_SAVE_METADATA"))
        self.assertFalse(expected - mappings.keys(), f"Missing stage registrations: {sorted(expected - mappings.keys())}")
        for key, node in mappings.items():
            with self.subTest(node=key):
                self.assertIs(self.runtime.nodes.NODE_CLASS_MAPPINGS[key], node)
                self.assertTrue(issubclass(node, io.ComfyNode))
                self.assertEqual(node.GET_SCHEMA().node_id, key)
                self.assertIsInstance(node.INPUT_TYPES(), dict)
        custom = self.runtime.folder_paths.get_folder_paths("custom_nodes")
        self.assertEqual(len(custom), 1)
        self.assertEqual([p.name for p in Path(custom[0]).iterdir()], ["ComfyUI-PromptForge"])
        self.assertFalse(any("krea2-nag" in str(path) or "comfyui-krea2edit" in str(path)
                             for path in self.runtime.nodes.LOADED_MODULE_DIRS.values()))

    def test_filename_tokens_name_the_action_category_and_family(self):
        shared, contracts, prompts = self.v2.shared, self.v2.contracts, self.v2.prompting
        pipe = prompts.ImagePrompt.execute(contracts.new_pipe("krea2"), source="action", action="image: Sex - doggy style").args[0]
        self.assertEqual(shared.expand_tokens("out/%family%/%category%/%action%", pipe), "out/krea2/NSFW - couple/Sex - doggy style")
        idea = prompts.ImagePrompt.execute(contracts.new_pipe("krea2"), source="idea", text="a knight").args[0]
        self.assertEqual(shared.expand_tokens("%category%/%action%", idea), "idea/idea")
        pipe = contracts.evolve(pipe, section="models", settings={"model": "[unet] Krea 2/concept/krea2_turbo_anime.safetensors"})
        pipe = contracts.evolve(pipe, section="sampling", settings={"seed": 42})
        self.assertEqual(shared.expand_tokens("%model%_%mode%_%source%_seed%seed%", pipe), "krea2_turbo_anime_t2i_action_seed42")
        self.assertEqual(shared.expand_tokens("plain", idea), "plain")

    def test_raw_writer_lazy_bypass(self):
        module = self.v2.prompting
        for prompt, writer, family in ((module.ImagePrompt, module.ImagePromptWriter, "krea2"),
                                       (module.H3Prompt, module.H3PromptWriter, "minimax_h3")):
            with self.subTest(family=family):
                text = 'A sign reads "No entry".'
                pipe = prompt.execute(self.pipe(family), source="raw", text=text).args[0]
                self.assertEqual(writer.check_lazy_status(pipe), [])
                with patch.object(module, "_services", side_effect=AssertionError("raw loaded a writer")):
                    output = writer.execute(pipe).args
                self.assertIn(text, output[1]) if family == "minimax_h3" else self.assertEqual(output[1], text)
                self.assertEqual(output[0]["artifacts"]["writer"]["status"], "raw")
                idea = prompt.execute(self.pipe(family), source="idea", text="a tree").args[0]
                self.assertEqual(writer.check_lazy_status(idea), ["writer_model"])

    def test_image_encode_decode_and_branch_isolation(self):
        torch = self.torch
        clip = self.clip()
        class Vae:
            def decode(self, samples):
                self.samples = samples
                return torch.zeros(samples.shape[0], samples.shape[-2] * 8, samples.shape[-1] * 8, 3)
        vae = Vae()
        model = SimpleNamespace(model=SimpleNamespace(latent_format=SimpleNamespace(spacial_downscale_ratio=8, latent_channels=16)))
        base = self.pipe(clip=clip, vae=vae, model=model)
        fmt = self.v2.image.ImageFormat.execute(base, aspect="16:9", megapixels=.01, orientation="portrait").args[0]
        pipe = self.v2.prompting.ImagePrompt.execute(fmt, text="a tree", negative="blur").args[0]
        output, positive, negative, latent = self.v2.image.ImageEncode.execute(pipe).args
        self.assertEqual([call[0] for call in clip.calls], ["a tree", "blur"])
        decoded, images = self.v2.image.ImageDecode.execute(output).args
        self.assertIs(vae.samples, latent["samples"])
        self.assertIs(decoded["artifacts"]["images"], images)
        self.assertEqual(tuple(images.shape), (1, latent["samples"].shape[-2]*8, latent["samples"].shape[-1]*8, 3))
        self.assertNotIn("images", output["artifacts"] )
        decoded_pipe, images = self.v2.image.ImageDecode.execute(output).args
        self.assertIs(vae.samples, latent["samples"])
        self.assertIs(decoded_pipe["artifacts"]["images"], images)
        self.assertEqual(tuple(images.shape[1:3]), tuple(size * 8 for size in latent["samples"].shape[-2:]))
        self.assertNotIn("images", output["artifacts"])
        self.assertGreater(fmt["settings"]["format"]["height"], fmt["settings"]["format"]["width"])
        self.assertNotIn("latent", pipe["artifacts"])
        self.assertEqual(base["settings"], {})
        decoded, images = self.v2.image.ImageDecode.execute(output).args
        self.assertIs(vae.samples, latent["samples"])
        self.assertIs(decoded["artifacts"]["images"], images)
        self.assertIs(output["artifacts"]["positive"], positive)
        self.assertIs(output["artifacts"]["negative"], negative)
        invalid = self.v2.contracts.evolve(pipe, section="prompt", settings={"source":"idea"}, artifacts={"writer":{"status":"failed"}})
        with self.assertRaisesRegex(ValueError, "writer"):
            self.v2.image.ImageEncode.execute(invalid)

    def test_image_sample_consumes_resolved_settings(self):
        image, torch = self.v2.image, self.torch
        model, positive, negative = object(), object(), object()
        latent = {"samples":torch.zeros(1,16,2,2)}
        pipe = self.pipe(model=model,positive=positive,negative=negative,latent=latent,images=object())
        pipe = image.ImageSamplingSettings.execute(pipe, 17, 8, 1., "euler", "simple", .7).args[0]
        sampled = {"samples":torch.ones(1,16,2,2)}
        with patch.object(image.nodes, "common_ksampler", return_value=(sampled,)) as sampler:
            result, output = image.ImageSample.execute(pipe).args
        self.assertIs(output,sampled)
        self.assertIs(result["artifacts"]["latent"],sampled)
        self.assertIs(pipe["artifacts"]["latent"],latent)
        self.assertNotIn("images",result["artifacts"])
        self.assertEqual(sampler.call_args.args,(model,17,8,1.,"euler","simple",positive,negative,latent))
        self.assertEqual(sampler.call_args.kwargs,{"denoise":.7})

    def test_refine_upscales_before_the_second_pass(self):
        image, torch = self.v2.image, self.torch
        latent = {"samples": torch.zeros(1, 16, 2, 2), "downscale_ratio_spacial": 8}
        pipe = self.pipe(model=object(), positive=object(), negative=object(), latent=latent)
        pipe = image.ImageSamplingSettings.execute(pipe, 17, 8, 1., "euler", "simple", .7, .3, 3).args[0]
        pipe = image.evolve(pipe, "upscale", {"enabled": True, "method": "bicubic", "scale": 2.0})
        sampled = {"samples": torch.ones(1, 16, 4, 4)}
        upscaled = {"samples": torch.zeros(1, 16, 4, 4)}
        with patch.object(image, "upscale_latent", return_value=upscaled) as upscaler, \
                patch.object(image.nodes, "common_ksampler", return_value=(sampled,)) as sampler:
            result = image.refine(pipe, latent)
        upscaler.assert_called_once_with(pipe, latent, "bicubic", 2.0)
        self.assertEqual(sampler.call_args.args[1:3], (18, 3))
        self.assertEqual(sampler.call_args.kwargs, {"denoise": .3})
        self.assertIs(result["artifacts"]["latent"], sampled)

    def test_sampler_seed_comes_from_sampling_settings(self):
        image, torch = self.v2.image, self.torch
        latent = {"samples":torch.zeros(1,16,2,2)}
        pipe = self.pipe(model=object(),positive=object(),negative=object(),latent=latent)
        pipe = image.ImageSamplingSettings.execute(pipe, 17, 8, 1., "euler", "simple", .7).args[0]
        sampled = {"samples":torch.ones(1,16,2,2)}
        with patch.object(image.nodes, "common_ksampler", return_value=(sampled,)) as sampler:
            stored = image.ImageSample.execute(pipe, pipe_settings=False).args[0]
            connected = image.ImageSample.execute(pipe, seed=99, pipe_settings=False).args[0]
        self.assertEqual(sampler.call_args_list[0].args[1], 17)
        self.assertEqual(sampler.call_args_list[1].args[1], 99)
        self.assertEqual(stored["settings"]["sampling"]["seed"], 17)
        self.assertEqual(connected["settings"]["sampling"]["seed"], 99)

    def test_lora_rank_cut_keeps_the_strongest_directions_and_ff_filter_skips_attention(self):
        loader = self.v2.shared.lora_loader
        torch = loader.torch
        generator = torch.Generator().manual_seed(1)
        up, down = torch.randn(40, 8, generator=generator), torch.randn(8, 24, generator=generator)
        adapter = loader.LoRAAdapter(set(), (up, down, 4.0, None, None, None))
        cut = loader.lowrank(adapter, 3)
        new_up, new_down, alpha = cut.weights[0], cut.weights[1], cut.weights[2]
        self.assertEqual((new_up.shape, new_down.shape, alpha), ((40, 3), (3, 24), 4.0 * 3 / 8))
        u, s, vh = torch.linalg.svd(up @ down)
        best = (u[:, :3] * s[:3]) @ vh[:3]
        self.assertTrue(torch.allclose(new_up @ new_down, best, atol=1e-3))
        self.assertIs(loader.lowrank(adapter, 8), adapter)
        for key, kept in (("diffusion_model.blocks.3.mlp.fc1.weight", True), ("diffusion_model.blocks.3.attn.qkv_proj.weight", False)):
            self.assertEqual(bool(loader.MLP_KEY.search(key)), kept)

    def test_model_profiles_ship_for_every_checkpoint_and_user_saves_override_them(self):
        import asyncio, tempfile
        profiles = self.v2.model_profiles
        self.assertEqual(profiles.key("[unet] Krea 2/concept/krea2_turbo_anime_int8_convrot.safetensors"), profiles.key("krea2_turbo_anime.safetensors"))
        for model, steps, sampler in (("[unet] Krea 2/concept/krea2_turbo_anime_int8_convrot.safetensors", 8, "euler"),
                                      ("[unet] Krea 2/characters/Muse By Stable Yogi Krea2 V4.0 Extended CIV INT8-ConvRot C86.safetensors", 16, "er_sde"),
                                      ("[unet] MOSS Image/moss/Dirty-MossV2.safetensors", 25, "er_sde")):
            found = profiles.profile(model)
            self.assertEqual((found["settings"]["steps"], found["settings"]["sampler_name"], found["user"]), (steps, sampler, False))
        eros = profiles.profile("[unet] MiniMax H3/video/10Eros_Max_8stepTurbo_v1.0_hybrid_activation_learned_HQ.safetensors")
        self.assertEqual((eros["settings"]["sampler_name"], eros["settings"]["shift_video"], eros["assumed"]), ("lcm", 12.0, True))
        self.assertEqual(profiles.profile("[ckpt] MiniMax H3/no tags/10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")["settings"]["sampler_name"], "lcm")
        prism = profiles.profile("[unet] MiniMax H3/video/PRISM_DMAD_PDMD_w6a8_g32.safetensors")
        self.assertEqual((prism["settings"]["scheduler"], prism["settings"]["start_sigma"]), ("beta57", 0.9035))
        self.assertIsNone(profiles.profile("[unet] Other/never_heard_of_it.safetensors"))
        with tempfile.TemporaryDirectory() as folder, patch.object(profiles, "USER", Path(folder) / "user.json"):
            class Request:
                query = {"model": "[unet] MOSS Image/moss/Dirty-MossV2.safetensors"}
                async def json(self):
                    return {"model": self.query["model"], "settings": {"steps": 30, "cfg": 3.0, "evil": "ignored"}}
            saved = json.loads(asyncio.run(profiles.save_profile(Request())).text)["profile"]
            self.assertEqual((saved["settings"], saved["user"]), ({"steps": 30, "cfg": 3.0}, True))
            self.assertEqual(profiles.profile(Request.query["model"])["settings"]["steps"], 30)
            forgotten = json.loads(asyncio.run(profiles.delete_profile(Request())).text)["profile"]
            self.assertEqual((forgotten["settings"]["steps"], forgotten["user"]), (25, False))

    def test_every_template_graph_supplies_every_required_input(self):
        # a new required input on an existing node makes every older graph fail validation and silently drop its outputs
        mappings = self.v2.NODE_CLASS_MAPPINGS
        for path in sorted((Path(__file__).parents[1] / "example_workflows" / "v2").glob("*.api.json")):
            for key, node in json.loads(path.read_text()).items():
                schema = mappings[node["class_type"]].define_schema()
                missing = [item.id for item in schema.inputs if not getattr(item, "optional", False) and item.id not in node["inputs"]]
                self.assertEqual(missing, [], f"{path.name} node {key} {node['class_type']}")

    def test_disabled_lora_stack_never_opens_weights(self):
        shared = self.v2.shared
        model, clip = object(), object()
        pipe = self.pipe(model=model, clip=clip)
        with patch.object(shared.lora_loader.comfy.utils, "load_torch_file", side_effect=AssertionError("disabled LoRA read weights")):
            result = shared.LoraStack.execute(pipe, json.dumps({"items":[{"name":"missing.safetensors", "on":False}]})).args[0]
            bypass = shared.SingleLora.execute(pipe, "missing.safetensors", enabled=False).args[0]
        self.assertIs(result["artifacts"]["model"], model)
        self.assertIs(result["artifacts"]["clip"], clip)
        self.assertIs(bypass["artifacts"]["model"], model)
        self.assertEqual(bypass["settings"]["single_loras"]["items"][0]["enabled"], False)
        self.assertNotIn("single_loras", pipe["settings"])

    def test_image_loader_containment_and_pixels(self):
        from PIL import Image
        torch = self.torch
        root = Path(self.runtime.folder_paths.get_input_directory())
        image_path = root / "fixture.png"
        Image.new("RGBA", (4, 3), (255, 0, 0, 128)).save(image_path)
        pipe, image, mask = self.v2.media.LoadImage.execute(self.pipe(), "fixture.png").args
        self.assertEqual(tuple(image.shape), (1,3,4,3))
        torch.testing.assert_close(image[0,0,0], torch.tensor([1.,0.,0.]))
        self.assertAlmostEqual(float(mask[0,0,0]), 1 - 128/255, places=6)
        self.assertIs(pipe["artifacts"]["reference_images"][0], image)
        outside = self.runtime.base / "outside.png"
        Image.new("RGB", (2,2)).save(outside)
        link = root / "escape.png"
        link.symlink_to(outside)
        for filename in ("../outside.png", str(outside), "escape.png"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "input directory"):
                self.v2.media.LoadImage.execute(self.pipe(), filename)

    def test_frame_interpolation_preserves_duration_and_audio(self):
        torch = self.torch
        frames = torch.stack((torch.zeros(2,2,3), torch.ones(2,2,3), torch.full((2,2,3), .5)))
        audio = {"waveform":torch.randn(1,2,120), "sample_rate":960}
        waveform = audio["waveform"].clone()
        pipe = self.v2.contracts.evolve(self.pipe("minimax_h3", images=frames, audio=audio), "format", {"fps":24})
        out, interpolated, fps = self.v2.media.FrameInterpolate.execute(pipe, enabled=True, multiplier=3).args
        self.assertEqual(len(interpolated), len(frames)*3)
        self.assertEqual(fps, 72.)
        self.assertEqual(len(interpolated)/fps, len(frames)/24)
        self.assertIs(out["artifacts"]["audio"], audio)
        torch.testing.assert_close(audio["waveform"], waveform)
        torch.testing.assert_close(interpolated[0], frames[0])
        torch.testing.assert_close(interpolated[-1], frames[-1])
        torch.testing.assert_close(interpolated[1], torch.lerp(frames[0], frames[1], 1/3))
        self.assertEqual(out["settings"]["timing"]["duration"], len(frames)/24)
        self.assertIs(self.v2.media.FrameInterpolate.execute(pipe, enabled=False).args[0], pipe)

    def test_clip_buffers_go_to_disk_when_ram_is_short(self):
        torch = self.torch
        frame_buffer = sys.modules[self.runtime.pack.__name__ + ".frame_buffer"]
        frames = torch.rand(9, 8, 8, 3)
        pipe = self.v2.contracts.evolve(self.pipe("minimax_h3", images=frames), "format", {"fps": 24})
        roomy = self.v2.media.FrameInterpolate.execute(pipe, enabled=True, multiplier=2, batch_size=3).args[1]
        short = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1))
        with patch.object(frame_buffer, "psutil", short):
            tight = self.v2.media.FrameInterpolate.execute(pipe, enabled=True, multiplier=2, batch_size=3).args[1]
        self.assertEqual(tight.dtype, torch.float16)
        self.assertEqual(len(tight), 18)
        torch.testing.assert_close(tight, roomy.half(), atol=1e-3, rtol=0)
        self.assertEqual(list((self.runtime.base / "temp").glob("promptforge_frames_*")), [])

    def test_sam3_list_holds_only_sam3_files(self):
        import struct, tempfile
        masking = importlib.import_module(self.v2.__name__ + ".masking")
        wanted = ["model.detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight", "model.detector.transformer.decoder.query_embed.weight"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, keys in (("segmenter.safetensors", wanted), ("video.safetensors", ["blocks.0.attn.qkv_proj.weight"])):
                header = json.dumps({key: {} for key in keys}).encode()
                (root / name).write_bytes(struct.pack("<Q", len(header)) + header)
            (root / "broken.safetensors").write_bytes(struct.pack("<Q", 4) + b"{no}")
            for name in ("sam3_old.pt", "other.ckpt"):
                (root / name).write_bytes(b"x")
            names = sorted(path.name for path in root.iterdir())
            with patch.object(masking.folder_paths, "get_filename_list", return_value=names), \
                    patch.object(masking.folder_paths, "get_full_path", side_effect=lambda _, name: str(root / name)), \
                    patch.object(masking.folder_paths, "models_dir", folder):
                self.assertEqual(masking.sam3_checkpoints(), ["(select SAM3 checkpoint)", "sam3_old.pt", "segmenter.safetensors"])

    def test_auto_interpolation_streams_pairs_and_blends_what_never_fits(self):
        import torch
        interp = importlib.import_module(self.v2.__name__.rsplit(".", 1)[0] + ".frame_interpolation")

        class Net:
            def extract_features(self, frame):
                return frame

            def memory_used_forward(self, shape, dtype):
                return 0

            def __call__(self, first, second, timestep, cache):
                return first + (second - first) * timestep[:, :, :1, :1]

        patcher = SimpleNamespace(load_device=torch.device("cpu"), model=Net(), model_dtype=lambda: torch.float32)
        frames = torch.rand(4, 8, 8, 3)
        out = torch.empty(10, 8, 8, 3)
        with patch.object(interp.model_management, "load_models_gpu"):
            interp._stream(patcher, frames, 3, out, None)
        torch.testing.assert_close(out, interp._blend(frames, 3))
        out = torch.empty(7, 8, 8, 3)
        with patch.object(interp.model_management, "load_models_gpu", side_effect=torch.cuda.OutOfMemoryError("out of memory")), \
                patch.object(interp.model_management, "unload_all_models"), patch.object(interp.model_management, "soft_empty_cache"):
            interp._stream(patcher, frames, 2, out, (4, 4))
        torch.testing.assert_close(out, interp._blend(frames, 2))
        with patch.object(interp.model_management, "load_models_gpu", side_effect=RuntimeError("broken")), self.assertRaisesRegex(RuntimeError, "broken"):
            interp._stream(patcher, frames, 2, out, None)

    def test_reference_bias_equals_a_dense_mask(self):
        import math
        torch = self.torch
        krea = importlib.import_module(self.v2.__name__ + ".krea")
        torch.manual_seed(0)
        slens, targets, text = [6, 8], 5, 3
        length = text + sum(slens) + targets
        q, k, v = (torch.randn(2, 4, length, 16, dtype=torch.float64) for _ in range(3))
        for boosts, masks in (([1.0, 2.0], [None, None]), ([0.5, 4.0], [None, torch.eye(4)[:2][None]]), ([0.0, 1.0], [None, None])):
            bias = krea._ref_attn_bias(boosts, masks, slens, targets, [(2, 3), (2, 4)], "cpu")
            dense = torch.zeros(length, length, dtype=torch.float64)
            dense[length - targets:, length - len(bias.keys):] = bias.keys.double()
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense).transpose(1, 2).reshape(2, length, 64)
            torch.testing.assert_close(krea.biased_attention(q.clone(), k.clone(), v.clone(), bias), expected, atol=1e-6, rtol=1e-6)
        previous = krea.krea2_model.optimized_attention_masked
        calls = []
        with patch.object(krea.krea2_model, "optimized_attention_masked", lambda *args, **kwargs: calls.append(kwargs.get("mask"))):
            with krea._biased_blocks():
                krea.krea2_model.optimized_attention_masked(None, None, None, 4, mask=None)
        self.assertEqual(calls, [None])
        self.assertIs(krea.krea2_model.optimized_attention_masked, previous)

    def test_subject_mask_is_reused_for_the_same_picture(self):
        import tempfile
        torch = self.torch
        masking = importlib.import_module(self.v2.__name__ + ".masking")
        image = torch.rand(1, 8, 8, 3)
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as file, \
                patch.object(masking, "sam3_path", return_value=Path(file.name)), patch.object(masking, "load_sam3", return_value=(None, None)) as load, \
                patch.object(masking, "segment", side_effect=lambda model, clip, picture, *rest: picture[..., 0].clone()):
            masking._MASKS.clear()
            first = masking.remembered_mask("sam", image, "face", 0.5, 2)
            first[:] = 7
            torch.testing.assert_close(masking.remembered_mask("sam", image.clone(), "face", 0.5, 2), image[..., 0])
            self.assertEqual(load.call_count, 1)
            masking.remembered_mask("sam", image, "hair", 0.5, 2)
            masking.remembered_mask("sam", 1 - image, "face", 0.5, 2)
            self.assertEqual(load.call_count, 3)
            for index in range(masking.KEPT_MASKS + 3):
                masking.remembered_mask("sam", image, str(index), 0.5, 2)
            self.assertEqual(len(masking._MASKS), masking.KEPT_MASKS)

    def test_easycache_switch_wraps_the_model_only_when_on(self):
        torch = self.torch
        image = importlib.import_module(self.v2.__name__ + ".image")
        cached = SimpleNamespace(model_options={"transformer_options": {"easycache": object()}})
        plain = SimpleNamespace(model_options={"transformer_options": {}})
        wrapped = object()
        for model, on, expected, calls in ((plain, False, plain, 0), (plain, True, wrapped, 1), (cached, True, cached, 0)):
            pipe = self.pipe(model=model, positive=[], negative=[], latent={"samples": torch.zeros(1, 4, 2, 2)})
            pipe = image.ImageSamplingSettings.execute(pipe, easycache=on, easycache_threshold=0.3).args[0]
            with patch.object(image.EasyCacheNode, "execute", return_value=SimpleNamespace(args=(wrapped,))) as cache, \
                    patch.object(image.nodes, "common_ksampler", return_value=({"samples": torch.ones(1, 4, 2, 2)},)) as sampler:
                image.ImageSample.execute(pipe)
            self.assertEqual(cache.call_count, calls)
            self.assertIs(sampler.call_args.args[0], expected)
            if calls:
                self.assertEqual(cache.call_args.args, (plain, 0.3, 0.15, 0.95, False))

    def test_browser_deletes_selections_and_whole_views_only_inside_the_root(self):
        import os, tempfile
        browser = importlib.import_module(self.v2.__name__ + ".browser")
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as store, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            def put(rel, base=root):
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_bytes(b"x")
                return base / rel
            with patch.dict(browser.ROOT_NAMES, {"output": lambda: folder}), patch.object(browser, "_store", lambda: Path(store) / "meta.json"):
                for rel in ("a.png", "b.png", "clip.mp4", "sub/c.png", "sub/deep/d.mp4", "sub/graph.json", "only/e.png", ".hidden/f.png"):
                    put(rel)
                secret = put("keep.png", Path(outside))
                os.symlink(secret, root / "link.png")
                deleted, failed = browser._delete_files("output", ["a.png", "../escape.png", "sub/graph.json", "missing.png", str(secret)])
                self.assertEqual((deleted, len(failed)), (["a.png"], 4))
                count = browser._delete_view("output", "sub", "", "video", None, True)
                self.assertEqual((count["files"], count["kept"]), (1, 0))
                self.assertEqual(browser._delete_view("output", "sub", "", "video", count["before"], False)["deleted"], 1)
                self.assertTrue((root / "sub/c.png").exists())
                count = browser._delete_view("output", "", "", "all", None, True)
                self.assertEqual((count["files"], count["kept"]), (5, 1))
                later = put("later.png")
                os.utime(later, (count["before"] + 60, count["before"] + 60))
                result = browser._delete_view("output", "", "", "all", count["before"], False)
                self.assertEqual((result["deleted"], len(result["failed"])), (4, 1))
                self.assertTrue(secret.exists() and later.exists() and (root / "sub/graph.json").exists() and (root / ".hidden/f.png").exists())
                self.assertFalse((root / "only").exists() or (root / "sub/deep").exists())
                self.assertTrue(root.exists())
                with self.assertRaises(ValueError):
                    browser._delete_view("output", "../", "", "all", None, True)
                with self.assertRaises(ValueError):
                    browser._delete_view("user", "", "", "all", None, True)

    def test_out_of_memory_unloads_and_retries_once(self):
        shared = self.v2.shared
        outcomes = [self.torch.cuda.OutOfMemoryError("CUDA out of memory"), "done"]
        def work():
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        with patch.object(shared, "free_vram") as release:
            self.assertEqual(shared.retry_unloaded("testing", work), "done")
        release.assert_called_once()

    def test_h3_encode_mode_dispatch(self):
        h3, torch = self.v2.h3, self.torch
        from comfy_api.latest import io
        first, last = torch.zeros(1,4,4,3), torch.ones(1,4,4,3)
        video = torch.randn(22,4,4,3)
        audio = {"waveform":torch.randn(1,2,960), "sample_rate":960}
        positive, latent = [[torch.ones(1,1,64), {}]], {"samples":torch.zeros(1,4,1,2,2)}
        for mode in h3.MODES:
            with self.subTest(mode=mode):
                clip, vae = self.clip(), object()
                pipe = self.pipe("minimax_h3", clip=clip, vae=vae, text="a tree", first_frame=first,
                                 last_frame=last, reference_video=video, reference_audio=audio, reference_images=[first])
                pipe = self.v2.contracts.evolve(pipe, "format", {"width":32,"height":32,"frames":22,"fps":24})
                pipe = self.v2.contracts.evolve(pipe, "references", {"mode":mode})
                pipe = self.v2.contracts.evolve(pipe, "prompt", {"source":"raw"})
                with patch.object(h3.core.MiniMaxH3ImageToVideo, "execute", return_value=io.NodeOutput(positive,latent)) as still, \
                     patch.object(h3.core.MiniMaxH3ReferenceToVideo, "execute", return_value=io.NodeOutput(positive,latent)) as refs, \
                     patch.object(h3.core.MiniMaxH3AddGuide, "execute", return_value=io.NodeOutput(positive)) as guide:
                    output = h3.H3Encode.execute(pipe).args[0]
                self.assertIs(output["artifacts"]["latent"], latent)
                self.assertEqual(refs.call_count, int(mode == "ref2v"))
                self.assertEqual(still.call_count, int(mode != "ref2v"))
                self.assertEqual(guide.call_count, int(mode == "continue"))
                kwargs = (refs if mode == "ref2v" else still).call_args.kwargs
                self.assertIn("a tree", kwargs["prompt"])
                self.assertIs(kwargs["vae"], vae)
                self.assertEqual("first_frame" in kwargs, mode in ("i2v","fl2v"))
                self.assertEqual("last_frame" in kwargs, mode in ("l2v","fl2v"))
                if mode == "ref2v":
                    self.assertIs(kwargs["ref_images"]["ref_image_1"], first)
                    self.assertIs(kwargs["ref_videos"]["ref_video_1"], video)
                    self.assertIs(kwargs["ref_audios"]["ref_audio_1"], audio)
                if mode == "continue":
                    self.assertEqual(output["artifacts"]["continuation_overlap"], 22)
                    torch.testing.assert_close(guide.call_args.kwargs["image"], video)
                    self.assertIsNot(guide.call_args.kwargs["audio"]["waveform"], audio["waveform"])
                    torch.testing.assert_close(guide.call_args.kwargs["audio"]["waveform"], audio["waveform"][..., -880:])

    def test_save_metadata_redaction_counter_and_opt_in(self):
        shared = self.v2.shared
        pipe = self.pipe(model=object(), text="live text", source_mask=self.torch.ones(1,2,2))
        pipe = self.v2.contracts.evolve(pipe, "prompt", {"source":"raw", "text":"private prompt", "negative":"private negative"})
        pipe = self.v2.contracts.evolve(pipe, "parts", {"plan":[{"text":"private part", "seconds":5}]})
        pipe = self.v2.contracts.evolve(pipe, "media", {"reference:1":{"description":"private caption"}})
        first_pipe, first = shared.SaveRunMetadata.execute(pipe, "runtime/metadata").args
        second_pipe, second = shared.SaveRunMetadata.execute(pipe, "runtime/metadata", include_text=True).args
        self.assertIs(first_pipe, pipe)
        self.assertIs(second_pipe, pipe)
        self.assertTrue(first.endswith("metadata_00001_.json"))
        self.assertTrue(second.endswith("metadata_00002_.json"))
        redacted = json.loads(Path(first).read_text())
        full = json.loads(Path(second).read_text())
        self.assertNotIn("private", json.dumps(redacted))
        self.assertNotIn("artifacts", redacted)
        self.assertEqual(redacted["settings"]["parts"]["plan"][0]["seconds"], 5)
        self.assertEqual(full["settings"], pipe["settings"])
        self.assertEqual(pipe["settings"]["prompt"]["text"], "private prompt")
        self.assertEqual(json.loads(Path(first).read_text()), redacted)
        with self.assertRaises(Exception):
            shared.SaveRunMetadata.execute(pipe, "../metadata-escape")
        self.assertFalse((self.runtime.base / "metadata-escape_00001_.json").exists())

    def test_save_metadata_redacts_subject_mask_query(self):
        pipe = self.v2.contracts.evolve(self.pipe(), "mask", {"checkpoint":"sam.safetensors", "subject":"private mask query", "threshold":.5})
        _, path = self.v2.shared.SaveRunMetadata.execute(pipe, "runtime/mask-metadata", include_text=False).args
        saved = json.loads(Path(path).read_text())
        self.assertNotIn("subject", saved["settings"]["mask"])
        self.assertEqual(saved["settings"]["mask"]["threshold"],.5)
        self.assertEqual(pipe["settings"]["mask"]["subject"],"private mask query")

    def test_single_lora_preserves_stack_metadata(self):
        shared = self.v2.shared
        model, clip, patched_model, patched_clip = object(), object(), object(), object()
        pipe = self.pipe(model=model, clip=clip, lora_report="stack report", lora_triggers="stack trigger", positive=object())
        config = {"loras":{"items":[{"name":"stack.safetensors","on":True}]}, "seed":23,
                  "layer_controls":"minimax audiovisual", "balance":True, "balance_floor":.8}
        pipe = self.v2.contracts.evolve(pipe,"loras",config)
        checkpoint = Path(self.runtime.folder_paths.get_folder_paths("loras")[0]) / "single.safetensors"
        checkpoint.touch()
        with patch.object(shared.lora_loader.PROMPTFORGE_LoraLoader,"load",
                          return_value=(patched_model,patched_clip,"single trigger","single report")) as load:
            out = shared.SingleLora.execute(pipe,checkpoint.name,model_strength=.7,clip_strength=.3).args[0]
        self.assertEqual(out["settings"]["loras"],config)
        self.assertIs(out["artifacts"]["model"],patched_model)
        self.assertIs(out["artifacts"]["clip"],patched_clip)
        self.assertEqual(out["artifacts"]["lora_report"],"stack report\nsingle report")
        self.assertEqual(out["artifacts"]["lora_triggers"],"stack trigger, single trigger")
        self.assertNotIn("positive",out["artifacts"])
        self.assertIn("positive",pipe["artifacts"])
        state = json.loads(load.call_args.args[1])
        self.assertEqual(state["items"],[{"name":checkpoint.name,"on":True,"model":.7,"clip":.3,"part":"all"}])
        bypass = shared.SingleLora.execute(out, "disabled.safetensors", enabled=False).args[0]
        self.assertEqual(len(bypass["settings"]["single_loras"]["items"]),2)
        self.assertEqual(bypass["settings"]["loras"],config)
        self.assertIs(bypass["artifacts"]["model"],patched_model)
        self.assertNotIn("single_loras",pipe["settings"])

    def test_subject_mask_forwards_native_sam_parameters(self):
        module, torch = self.v2.masking, self.torch
        from comfy_api.latest import io
        checkpoint = Path(self.runtime.folder_paths.get_folder_paths("checkpoints")[0]) / "sam-fixture.safetensors"
        checkpoint.touch()
        image, mask = torch.zeros(1,4,6,3), torch.ones(1,4,6)
        base_model = object()
        for family in ("SAM3","SAM31"):
            with self.subTest(family=family):
                clip = self.clip()
                model = SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace(unet_config={"image_model":family})))
                pipe = self.pipe(model=base_model)
                module._MASKS.clear()
                with patch.object(module.comfy.sd,"load_checkpoint_guess_config",return_value=(model,clip,None,None)) as load, \
                     patch.object(module.SAM3_Detect,"execute",return_value=io.NodeOutput(mask)) as detect:
                    out, result = module.SubjectMask.execute(pipe,image,checkpoint.name,"hair",.37,4).args
                load.assert_called_once_with(str(checkpoint),output_vae=False,output_clip=True,
                                            embedding_directory=self.runtime.folder_paths.get_folder_paths("embeddings"))
                self.assertEqual(clip.calls,[("hair",{})])
                self.assertIs(detect.call_args.args[0],model)
                self.assertIs(detect.call_args.args[1],image)
                self.assertEqual(detect.call_args.kwargs["threshold"],.37)
                self.assertEqual(detect.call_args.kwargs["refine_iterations"],4)
                self.assertFalse(detect.call_args.kwargs["individual_masks"])
                self.assertEqual(tuple(detect.call_args.kwargs["conditioning"][0][0].shape),(1,1,64))
                torch.testing.assert_close(result,mask)
                self.assertIs(out["artifacts"]["source_mask"],result)
                self.assertIs(out["artifacts"]["model"],base_model)
                self.assertNotIn("source_mask",pipe["artifacts"])
        wrong = SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace(unet_config={"image_model":"krea2"})))
        module._MASKS.clear()
        with patch.object(module.comfy.sd,"load_checkpoint_guess_config",return_value=(wrong,self.clip(),None,None)), \
             patch.object(module.SAM3_Detect,"execute") as detect:
            with self.assertRaisesRegex(ValueError,"SAM3"):
                module.SubjectMask.execute(self.pipe(),image,checkpoint.name)
            detect.assert_not_called()

    def test_performance_preview_configuration_and_forwarding(self):
        module = self.v2.performance
        from comfy_api.latest import io
        base, attention_model, chunked, sparse = object(), object(), object(), object()
        pipe = self.pipe("minimax_h3",model=base,images=object())
        with patch.object(module.ModelAttentionBackend,"execute",return_value=io.NodeOutput(attention_model)) as attention, \
             patch.object(module,"chunk_feed_forward",return_value=chunked) as chunks, \
             patch.object(module.BlockSparseAttention,"execute",return_value=io.NodeOutput(sparse)) as patch_sparse:
            out = module.Performance.execute(pipe,"pytorch attention",3,True,27.).args[0]
        attention.assert_called_once_with(base,"pytorch attention")
        chunks.assert_called_once_with(attention_model,3)
        patch_sparse.assert_called_once_with(chunked,{"selection":"vsa","keep_percent":27.},0.,1.,dense_blocks="0, 1")
        self.assertIs(out["artifacts"]["model"],sparse)
        self.assertNotIn("images",out["artifacts"])
        self.assertIs(pipe["artifacts"]["model"],base)
        with patch.object(module.preview_override,"apply",side_effect=AssertionError("disabled preview patched model")):
            preview = module.PreviewSettings.execute(pipe,enabled=False,max_resolution=256,preview_frames=2,preview_fps=6).args[0]
        self.assertIs(preview["artifacts"]["model"],base)
        self.assertEqual(preview["settings"]["preview"]["max_resolution"],256)

    def test_native_tiny_krea_edit_nag(self):
        torch = self.torch
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT
        torch.manual_seed(7)
        model = SingleStreamDiT(features=64,tdim=16,txtdim=32,heads=4,kvheads=2,layers=1,patch=2,
                                channels=4,multiplier=2,txtlayers=2,txtheads=4,txtkvheads=2,
                                operations=comfy.ops.disable_weight_init)
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=.02)
        x, context, negative = torch.randn(1,4,4,6), torch.randn(1,3,64), torch.randn(1,2,64)
        sources = [torch.randn(1,4,4,4), torch.randn(1,4,2,4)]
        sampling = dict(use_nag=False,nag_phi=4.,nag_tau=2.5,nag_alpha=0.)
        edit = dict(position="beside",position_a="aligned",ref_boost=1.5,mask=torch.ones(1,32,32))
        import importlib
        forward = importlib.import_module(self.v2.__name__ + ".krea")._forward
        base = forward(model,x,torch.ones(1),context,sources,negative,edit,sampling,{})
        sampling["use_nag"] = True
        zero = forward(model,x,torch.ones(1),context,sources,negative,edit,sampling,{})
        torch.testing.assert_close(base,zero,atol=1e-5,rtol=1e-5)
        sampling["nag_alpha"] = .25
        guided = forward(model,x.unsqueeze(2),torch.ones(1),context,sources,negative,edit,sampling,{})
        self.assertEqual(tuple(guided.shape),(1,4,1,4,6))
        self.assertTrue(torch.isfinite(guided).all())
        self.assertGreater(float((guided.squeeze(2)-base).detach().abs().max()), 0.)


if __name__ == "__main__":
    unittest.main()
