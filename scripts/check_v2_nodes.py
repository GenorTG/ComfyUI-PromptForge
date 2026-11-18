#!/usr/bin/env python3
"""Per-node smoke: every V2 node executes with defaults and stub artifacts.

Success means the node either returns outputs or refuses with a clear ValueError.
A TypeError, AttributeError or KeyError is a defect. Run with ComfyUI's python.
"""
import sys, unittest
from unittest import mock
import torch

sys.path.insert(0, "/home/genortg/1TB-SAMSUNG/Comfy/ComfyUI")
import server  # noqa: E402

class Any:
    def __getattr__(self, name):
        return lambda *a, **k: (lambda f: f)

class Fake:
    routes = Any()
    def send_progress_text(self, *a):
        pass

server.PromptServer.instance = Fake()
spec = __import__("importlib.util").util.spec_from_file_location(
    "pf", "/home/genortg/1TB-SAMSUNG/Comfy/ComfyUI/custom_nodes/ComfyUI-PromptForge/__init__.py",
    submodule_search_locations=["/home/genortg/1TB-SAMSUNG/Comfy/ComfyUI/custom_nodes/ComfyUI-PromptForge"])
pack = __import__("importlib.util").util.module_from_spec(spec)
sys.modules["pf"] = pack
spec.loader.exec_module(pack)

contracts = pack.v2.contracts
LATENT = {"samples": torch.zeros(1, 16, 8, 8), "downscale_ratio_spacial": 8}
IMAGES = torch.zeros(1, 8, 8, 3)
AUDIO = {"waveform": torch.zeros(1, 2, 800), "sample_rate": 44100}
TEXT = "a test prompt"


def stub_model():
    latent_format = mock.Mock(spacial_downscale_ratio=8, latent_channels=16)
    model = mock.Mock()
    model.model.latent_format = latent_format
    model.model.model_config = mock.Mock(vae_ratio_t=4)
    return model


def pipe_for(family):
    pipe = contracts.new_pipe(family)
    pipe["artifacts"] = {"model": stub_model(), "clip": mock.Mock(), "vae": mock.Mock(),
                         "positive": [], "negative": [], "text": TEXT, "latent": dict(LATENT),
                         "images": IMAGES, "audio": AUDIO, "krea_references": [IMAGES],
                         "krea_source_latents": [torch.zeros(1, 16, 8, 8)], "krea_mask": torch.ones(1, 8, 8)}
    pipe["settings"] = {"prompt": {"source": "raw", "text": TEXT, "negative": ""},
                        "sampling": {"seed": 1, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
                                     "scheduler": "simple", "denoise": 1.0, "use_nag": False,
                                     "nag_phi": 4.0, "nag_tau": 2.5, "nag_alpha": 0.25,
                                     "first": {"seed": 1, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
                                               "scheduler": "simple", "shift_video": 0.0, "shift_audio": 0.0},
                                     "refine": {"seed": 2, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
                                                "scheduler": "simple", "shift_video": 0.0, "shift_audio": 0.0}},
                        "format": {"width": 64, "height": 64, "batch_size": 1, "frames": 5, "fps": 24,
                                   "seconds": 5.0},
                        "upscale": {"enabled": False, "model": "x", "width": 128, "height": 128},
                        "writer": {"status": "valid"}, "media": {}}
    return pipe


PATCHES = [
    ("comfy.model_management.soft_empty_cache", mock.Mock()),
    ("comfy_extras.nodes_custom_sampler.KSampler", mock.Mock()),
]


class Smoke(unittest.TestCase):
    def test_every_node(self):
        problems, guarded, ok = [], [], 0
        for key, node in sorted(pack.NODE_CLASS_MAPPINGS.items()):
            if not key.startswith("PROMPTFORGE_V2_"):
                continue
            family = "minimax_h3" if "_H3_" in key else ("krea2" if "_KREA_" in key else "image")
            schema = node.define_schema()
            values = {}
            for item in schema.inputs:
                name, kind = item.id, type(item).__name__
                if name in ("pipe",):
                    values[name] = pipe_for(family)
                elif name in ("model",):
                    values[name] = stub_model()
                elif name in ("clip",):
                    values[name] = mock.Mock()
                elif name in ("vae",):
                    values[name] = mock.Mock()
                elif name in ("latent",):
                    values[name] = dict(LATENT)
                elif name in ("images", "image", "first_frame", "last_frame", "image_b", "destination", "source"):
                    values[name] = IMAGES
                elif name in ("audio",):
                    values[name] = AUDIO
                elif name in ("positive", "negative"):
                    values[name] = []
                elif name in ("video", "writer_model", "resume", "previous", "clip_vision"):
                    continue
                elif name == "checkpoint":
                    values[name] = next((o for o in item.options if o.startswith("sam3::")), item.options[0])
                elif name in ("vae_name", "audio_vae_name"):
                    values[name] = next((o for o in item.options if "minimax_h3_video_vae" in o or "minimax_h3_audio_vae" in o), item.options[0])
                elif name == "model_name":
                    values[name] = next((o for o in item.options if "MiniMax H3" in o), item.options[0])
                elif hasattr(item, "options") and item.options:
                    values[name] = item.options[0]
                elif hasattr(item, "default") and item.default is not None:
                    values[name] = item.default
                elif name == "mask":
                    values[name] = torch.zeros(1, 8, 8)
                elif name == "bank":
                    values[name] = []
                elif name == "parts_json":
                    values[name] = '[{"text":"part","seconds":1}]'
                else:
                    continue
            with mock.patch.object(pack.v2.h3, "sample", lambda pipe, **kw: pipe), \
                    mock.patch.object(pack.v2.h3, "decode", lambda pipe, *a, **k: pipe), \
                    mock.patch.object(pack.v2.h3, "encode", lambda pipe: pipe), \
                    mock.patch.object(pack.v2.image.nodes, "common_ksampler", lambda *a, **k: (dict(LATENT),)), \
                    mock.patch.object(pack.v2.image.nodes, "VAEDecode", mock.Mock(return_value=mock.Mock(
                        decode=lambda *a: (IMAGES,)))), \
                    mock.patch.object(pack.v2.image.nodes, "SaveImage", mock.Mock(return_value=mock.Mock(
                        save_images=lambda *a, **k: {"ui": {"images": []}}))), \
                    mock.patch.object(pack.v2.prompting, "_write", lambda *a, **k: ("written prompt", "")):
                try:
                    node.execute(**values)
                    ok += 1
                except ValueError as exc:
                    guarded.append((key, str(exc)[:70]))
                except (TypeError, AttributeError, KeyError) as exc:
                    problems.append((key, f"{type(exc).__name__}: {exc}"))
                except Exception as exc:  # noqa: BLE001
                    problems.append((key, f"{type(exc).__name__}: {str(exc)[:90]}"))
        print(f"ok={ok} guarded={len(guarded)} problems={len(problems)}")
        for key, note in guarded:
            print("  guard", key, "-", note)
        for key, note in problems:
            print("  PROBLEM", key, "-", note)
        self.assertFalse(problems)


if __name__ == "__main__":
    unittest.main(verbosity=2)
