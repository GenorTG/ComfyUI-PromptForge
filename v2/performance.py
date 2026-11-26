import logging

from comfy.ldm.modules.attention import get_attention_function
from comfy_api.latest import io
from comfy_extras.nodes_model_advanced import ModelAttentionBackend
from comfy_extras.nodes_sparse_attention import BlockSparseAttention
from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

from ..h3_generate import chunk_feed_forward
from .. import preview_decoder, preview_override
from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require

Pipe = io.Custom(PIPE)


class Performance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("PERFORMANCE", "", "Performance", inputs=[Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
                                 io.Combo.Input("attention", options=["as loaded", "pytorch attention", "comfy kitchen attention"]),
                                 io.Int.Input("ffn_chunks", default=2, min=1, max=64, tooltip="MiniMax H3 feed-forward chunks; other families retain their implementation."),
                                 io.Boolean.Input("sparse_attention", default=False),
                                 io.Float.Input("keep_percent", default=10.0, min=0.5, max=100.0, step=0.5),
                                 io.Boolean.Input("use_slg", default=False, tooltip="Skip layer guidance: skips the listed layers on the early steps. Most MOSS checkpoints are tuned with it, layer 22, ending at 0.5."),
                                 io.String.Input("slg_layers", default="22", tooltip="Comma separated layer indices skipped during guidance."),
                                 io.Float.Input("slg_start", default=0.0, min=0.0, max=1.0, step=0.01),
                                 io.Float.Input("slg_end", default=0.5, min=0.0, max=1.0, step=0.01)],
                         outputs=[Pipe.Output(), io.Model.Output("model")])

    @classmethod
    def execute(cls, pipe=None, attention="as loaded", ffn_chunks=2, sparse_attention=False, keep_percent=10.0,
                use_slg=False, slg_layers="22", slg_start=0.0, slg_end=0.5, model=None):
        pipe = evolve(ensure_pipe(pipe, model=model), artifacts={"model": model} if model is not None else None)
        model = require(pipe, "model")
        if attention != "as loaded":
            model = ModelAttentionBackend.execute(model, attention).args[0]
        if pipe["family"] == "minimax_h3":
            if ffn_chunks > 1:
                model = chunk_feed_forward(model, ffn_chunks)
            if sparse_attention:
                model = BlockSparseAttention.execute(model, {"selection": "vsa", "keep_percent": keep_percent},
                                                     0.0, 1.0, dense_blocks="0, 1").args[0]
        elif sparse_attention:
            raise ValueError("This sparse-attention preset is specific to MiniMax H3.")
        if use_slg:
            model = SkipLayerGuidanceDiTSimple.execute(model, slg_start, slg_end, "", slg_layers).result
        return io.NodeOutput(evolve(pipe, "performance", {"attention": attention, "ffn_chunks": ffn_chunks,
                                                          "sparse_attention": sparse_attention, "keep_percent": keep_percent,
                                                          "use_slg": use_slg, "slg_layers": slg_layers,
                                                          "slg_start": slg_start, "slg_end": slg_end},
                                   {"model": model}, invalidate=("images", "audio", "video")), model)


class PreviewSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("PREVIEW", "SETTINGS", "Live Preview", description="Live sampling preview override. For MiniMax H3, select taeh3 from models/vae_approx to watch a decoded video preview while sampling.",
                         inputs=[Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
                                 io.Boolean.Input("enabled", default=False, tooltip="Patch the sampler to send live preview frames to this node."),
                                 io.Int.Input("max_resolution", default=512, min=64, max=2048, step=16, tooltip="Maximum preview width or height."),
                                 io.Int.Input("preview_frames", default=4, min=1, max=4096, tooltip="Frames encoded into each live preview update."),
                                 io.Int.Input("preview_fps", default=12, min=1, max=60, tooltip="Playback rate for the live preview."),
                                 io.Int.Input("preview_quality", default=80, min=10, max=100, tooltip="JPEG quality of the preview frames."),
                                 io.Combo.Input("tiny_vae", options=preview_decoder.decoder_names(), default=preview_decoder.default_decoder(), tooltip="MiniMax H3 video preview: taeh3. Other video models can use latent2rgb or a matching tiny decoder.")],
                         hidden=[io.Hidden.unique_id], outputs=[Pipe.Output()])

    @classmethod
    def execute(cls, pipe=None, enabled=False, max_resolution=512, preview_frames=4, preview_fps=12,
                preview_quality=80, tiny_vae=preview_decoder.LATENT2RGB, model=None):
        pipe = evolve(ensure_pipe(pipe, model=model), artifacts={"model": model} if model is not None else None)
        model = require(pipe, "model")
        if enabled:
            model = preview_override.apply(model, cls.hidden.unique_id, label="V2", max_resolution=max_resolution,
                                           quality=preview_quality, suppress_default=True, frames=preview_frames, fps=preview_fps, tiny_vae=tiny_vae)
        return io.NodeOutput(evolve(pipe, "preview", {"enabled": enabled, "max_resolution": max_resolution,
                                                        "preview_frames": preview_frames, "preview_fps": preview_fps, "tiny_vae": tiny_vae}, {"model": model}))


# label -> the name ComfyUI registers the attention function under; only those installed here are offered
BACKENDS = {"pytorch attention": "pytorch", "comfy kitchen int8": "comfy_kitchen_int8", "sage attention": "sage", "sage attention 3": "sage3"}


class H3Speedup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        backends = ["as loaded", *(label for label, name in BACKENDS.items() if get_attention_function(name, None) is not None)]
        return nodes_schema("H3", "SPEEDUP", "Experimental Speedup", description="Optional MiniMax H3 speedups, all off by default. Place it after Performance and Live Preview and before Encode, with no model wire into Encode, Sample or Refine. Compare the same seed with it bypassed.",
                            inputs=[Pipe.Input("pipe", optional=True), io.Model.Input("model", optional=True),
                                    io.Boolean.Input("sol_attention", default=False, tooltip="Sol-Attn block-sparse attention (ComfyUI's Model Sparse Attention, training-free). Skips low-scoring attention blocks, so the result differs from dense attention."),
                                    io.Float.Input("sol_tau", default=1.3, min=0.0, max=4.0, step=0.05, tooltip="Higher is sparser and faster: 1.0 keeps about 16% of blocks exact, 1.5 about 7%, 2.0 about 2.7%. Lower is closer to dense."),
                                    io.Float.Input("sol_start_percent", default=0.2, min=0.0, max=1.0, step=0.01, tooltip="Attention stays dense before this point of the schedule, where the composition forms."),
                                    io.Float.Input("sol_end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                                    io.Int.Input("sol_extra_tokens", default=256, min=0, max=256, step=64, tooltip="Extra top-scoring tokens kept per block. 256 is closest to dense."),
                                    io.Combo.Input("attention_backend", options=backends, default="as loaded", tooltip="Dense attention implementation, also used wherever sparse attention is inactive. The int8 and Sage backends are approximations."),
                                    io.Boolean.Input("keep_models_for_first_pass", default=False, tooltip="First Pass Preview decodes without unloading the sampling model first, so the refine pass need not load it again. Same output; may be slower or run low on VRAM with a large decode VAE."),
                                    io.Boolean.Input("log_to_console", default=True, tooltip="Print whether each attention shape really ran sparse, and why not.")],
                            outputs=[Pipe.Output(), io.Model.Output("model")])

    @classmethod
    def execute(cls, pipe=None, sol_attention=False, sol_tau=1.3, sol_start_percent=0.2, sol_end_percent=1.0, sol_extra_tokens=256,
                attention_backend="as loaded", keep_models_for_first_pass=False, log_to_console=True, model=None):
        pipe = evolve(ensure_pipe(pipe, family="minimax_h3", model=model), artifacts={"model": model} if model is not None else None)
        if pipe["family"] != "minimax_h3":
            raise ValueError("Experimental Speedup is for MiniMax H3.")
        model = require(pipe, "model")
        if attention_backend != "as loaded":
            function = get_attention_function(BACKENDS.get(attention_backend), None)
            if function is None:
                raise ValueError(f"The attention backend '{attention_backend}' is not available on this machine.")
            model = model.clone()
            model.set_model_optimized_attention(function)
        if sol_attention:
            if pipe["settings"].get("performance", {}).get("sparse_attention"):
                raise ValueError("Turn off sparse_attention on the Performance node to use Sol attention: one sparse method at a time.")
            model = BlockSparseAttention.execute(model, {"selection": "sol-attn", "tau": sol_tau}, sol_start_percent, sol_end_percent,
                                                 extra_tokens=sol_extra_tokens, sink_conditioning="exact_kv_and_rows", verbose=log_to_console).args[0]
        settings = {"sol_attention": sol_attention, "sol_tau": sol_tau, "sol_start_percent": sol_start_percent, "sol_end_percent": sol_end_percent,
                    "sol_extra_tokens": sol_extra_tokens, "attention_backend": attention_backend, "keep_models_for_first_pass": keep_models_for_first_pass}
        if log_to_console:
            logging.info("[PromptForge] experimental speedup: " + ", ".join(f"{key}={value}" for key, value in settings.items()))
        return io.NodeOutput(evolve(pipe, "speedup", settings, {"model": model}, invalidate=("images", "audio", "video")), model)


NODES = [Performance, PreviewSettings, H3Speedup]
NODE_CLASS_MAPPINGS = {node.define_schema().node_id: node for node in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {node.define_schema().node_id: node.define_schema().display_name for node in NODES}
