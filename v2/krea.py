"""Owned Krea edit/NAG adapters, adapted for PromptForge V2.

Modified from comfyui-krea2edit (Apache-2.0) geometry helpers and
krea2-nag (MIT, copyright 2026 iljung1106) guidance math.
No external pack is imported or resolved at runtime.


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

MIT License

Copyright (c) 2026 iljung1106

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import math
from contextlib import contextmanager
from threading import Lock

import torch
import torch.nn.functional as F
import comfy.patcher_extension
import comfy.utils
import comfy.ldm.krea2.model as krea2_model
from comfy.ldm.modules.attention import attention_pytorch
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked
from comfy_api.latest import io

from .contracts import schema as nodes_schema, PIPE, ensure_pipe, evolve, require

Pipe = io.Custom(PIPE)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)

def _place_refs(ref_ids, modes, h_, w_):
    if isinstance(modes, str):
        modes = [modes] * len(ref_ids)
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes, strict=True):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        else:
            raise ValueError(f"unknown ref_position {mode!r}")
        out.append(ids)
    return out

def _fit_reference(image, H, W, fit_mode="crop"):
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + cw2]
        return F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True).movedim(1, -1)
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True).movedim(1, -1)


def _crop_to_mask(image, mask):
    """The masked area with a margin: the identity gets the whole reference grid and the rest of the photo is gone."""
    mask = mask.reshape(-1, *mask.shape[-2:])[0]
    rows, cols = (mask > .5).any(1).nonzero().flatten(), (mask > .5).any(0).nonzero().flatten()
    if not len(rows):
        return image, mask
    height, width = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
    top, bottom = max(0, int(rows[0] - .35 * height)), min(mask.shape[0], int(rows[-1] + 1 + .35 * height))
    left, right = max(0, int(cols[0] - .35 * width)), min(mask.shape[1], int(cols[-1] + 1 + .35 * width))
    return image[:, top:bottom, left:right], mask[None, top:bottom, left:right]


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    key = key + (fit_mode,)
    if key not in cache:
        cache[key] = vae.encode(_fit_reference(image, H, W, fit_mode)[..., :3].clamp(0, 1))
    return cache[key]


def _ref_attn_bias(boosts, boost_masks, txtlen, slens, tgtlen, mask_hw, device, dtype):
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_masks[i] is not None and mask_hw is not None:
            mask = boost_masks[i][:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias

def normalized_attention_guidance(
    positive: torch.Tensor,
    negative: torch.Tensor,
    phi: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """NAG equations 7-10, with norms evaluated in float32.

    ``positive`` and ``negative`` are attention outputs produced from the same
    image query and image K/V. Only their text context is different.
    """
    if positive.shape != negative.shape:
        raise ValueError(
            f"NAG attention shapes must match, got {tuple(positive.shape)} and {tuple(negative.shape)}"
        )

    dtype = positive.dtype
    z_pos = positive.float()
    z_neg = negative.float()

    guided = z_pos + float(phi) * (z_pos - z_neg)
    eps = torch.finfo(torch.float32).eps
    pos_norm = z_pos.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    guided_norm = guided.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    ratio = guided_norm / pos_norm
    normalized = guided * (ratio.clamp_max(float(tau)) / ratio)
    refined = float(alpha) * normalized + (1.0 - float(alpha)) * z_pos
    return refined.to(dtype=dtype)

def encode_grounded(clip, text, images):
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
    template = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
                "texture, quantity, text, spatial relationships of the objects and background:"
                "<|im_end|>\n<|im_start|>user\n" + vision + "{}<|im_end|>\n<|im_start|>assistant\n")
    prepared = []
    for image in images:
        h, w = image.shape[1:3]
        scale = min(1.0, 768 / max(h, w))
        prepared.append(comfy.utils.common_upscale(image.movedim(-1, 1), round(w * scale), round(h * scale),
                                                  "area", "disabled").movedim(1, -1)[..., :3])
    return clip.encode_from_tokens_scheduled(clip.tokenize(text, images=prepared, llama_template=template))


def _raw_attention(attn, x, freqs, options, mask):
    b, length, _ = x.shape
    q = attn.wq(x).reshape(b, length, attn.heads, attn.headdim).transpose(1, 2)
    k = attn.wk(x).reshape(b, length, attn.kvheads, attn.headdim).transpose(1, 2)
    v = attn.wv(x).reshape(b, length, attn.kvheads, attn.headdim).transpose(1, 2)
    q, k = attn.qknorm(q, k)
    q, k = apply_rope(q, k, freqs)
    repeat = attn.heads // attn.kvheads
    if repeat != 1:
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    out = optimized_attention_masked(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                     attn.heads, mask=mask, skip_reshape=True,
                                     preferred_attention=attn.comfy_attention, transformer_options=options)
    return out, attn.gate(x)


def _forward(model, x, timesteps, context, sources, negative, edit, sampling, options):
    """Reference prefix stays positive; NAG shares image Q/K/V and guides target only."""
    temporal_shape = x.shape if x.ndim == 5 else None
    if temporal_shape:
        x = x.movedim(2, 1).flatten(0, 1)
    batch, _, original_h, original_w = x.shape
    target, target_ids, gh, gw = model.process_img(x)
    target = model.first(target)
    source_tokens, reference_ids, source_grids = [], [], []
    for index, source in enumerate(sources, 1):
        source = source.to(x)
        if source.ndim == 5:
            source = source.movedim(2, 1).flatten(0, 1)
        source = comfy.utils.repeat_to_batch_size(source, batch)
        tokens, ids, sh, sw = model.process_img(source, index=index)
        source_tokens.append(model.first(tokens))
        reference_ids.append(_imgids_offset(batch, index, sh, sw, gh, gw, x.device))
        source_grids.append((sh, sw))
    modes = [edit.get("position_a", "aligned")] * max(0, len(sources) - 1) + [edit.get("position", "aligned")]
    reference_ids = _place_refs(reference_ids, modes, gh, gw)
    image_ids = torch.cat(reference_ids + [target_ids], dim=1)
    image = torch.cat(source_tokens + [target], dim=1)
    source_lengths = [item.shape[1] for item in source_tokens]
    offset = sum(source_lengths)
    t = model.tmlp(timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target.dtype))
    vec = model.tproj(t)
    options = options.copy()
    positive_text = model.txtmlp(model.txtfusion(model._unpack_context(context), transformer_options=options))
    use_nag = sampling.get("use_nag", False)
    if use_nag:
        negative = comfy.utils.repeat_to_batch_size(negative.to(context), batch)
        negative_text = model.txtmlp(model.txtfusion(model._unpack_context(negative), transformer_options=options))
    boosts = [edit.get("ref_boost_a", 1.0)] * max(0, len(sources) - 1) + [edit.get("ref_boost", 1.0)]

    def position_and_mask(length):
        ids = torch.cat((torch.zeros(batch, length, 3, device=x.device, dtype=torch.float32), image_ids), dim=1)
        mask = None
        if sources and any(boost != 1.0 for boost in boosts):
            mask = _ref_attn_bias(boosts, edit.get("masks") or [None] * len(sources), length, source_lengths, target.shape[1],
                                  source_grids, image.device, image.dtype)
        return model.pe_embedder(ids), mask

    pos_len = positive_text.shape[1]
    pos_freqs, pos_mask = position_and_mask(pos_len)
    if use_nag:
        neg_len = negative_text.shape[1]
        neg_freqs, neg_mask = position_and_mask(neg_len)
    options.update(total_blocks=len(model.blocks), block_type="single", reference_image_num_tokens=source_lengths,
                   img_slice=[pos_len + offset, pos_len + image.shape[1]])
    combined = torch.cat((positive_text, image), dim=1)
    for index, block in enumerate(model.blocks):
        options["block_index"] = index
        if not use_nag:
            combined = block(combined, vec, pos_freqs, pos_mask, transformer_options=options)
            continue
        prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
        negative_combined = torch.cat((negative_text, combined[:, pos_len:]), dim=1)
        pos_raw, pos_gate = _raw_attention(block.attn, (1 + prescale) * block.prenorm(combined) + preshift,
                                         pos_freqs, options, pos_mask)
        neg_raw, neg_gate = _raw_attention(block.attn, (1 + prescale) * block.prenorm(negative_combined) + preshift,
                                         neg_freqs, options, neg_mask)
        start = pos_len + offset
        guided = normalized_attention_guidance(pos_raw[:, start:], neg_raw[:, neg_len + offset:],
                                               sampling["nag_phi"], sampling["nag_tau"], sampling["nag_alpha"])
        raw = torch.cat((pos_raw[:, :start], guided), dim=1)
        combined = combined + pregate * block.attn.wo(raw * pos_gate.sigmoid())
        negative_text = negative_text + pregate * block.attn.wo(neg_raw[:, :neg_len] * neg_gate[:, :neg_len].sigmoid())
        combined = combined + postgate * block.mlp((1 + postscale) * block.postnorm(combined) + postshift)
        negative_text = negative_text + postgate * block.mlp((1 + postscale) * block.postnorm(negative_text) + postshift)
    output = model.last(combined, t)[:, pos_len + offset:]
    p = model.patch
    output = output.reshape(batch, gh, gw, model.channels, p, p).permute(0, 3, 1, 4, 2, 5)
    output = output.reshape(batch, model.channels, gh * p, gw * p)[..., :original_h, :original_w]
    if temporal_shape:
        b, _, frames, h, w = temporal_shape
        output = output.reshape(b, frames, model.channels, h, w).movedim(1, 2)
    return output


_ATTENTION_LOCK = Lock()


@contextmanager
def _pytorch_attention():
    """Force the torch attention path while a reference bias mask is in use."""
    with _ATTENTION_LOCK:
        previous = krea2_model.optimized_attention_masked
        krea2_model.optimized_attention_masked = attention_pytorch
        try:
            yield
        finally:
            krea2_model.optimized_attention_masked = previous


def patch_model(model, negative, sources, edit, sampling):
    if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
        raise ValueError("Krea edit/NAG needs a native Krea2 SingleStreamDiT model.")
    if sampling.get("use_nag") and sampling["cfg"] != 1.0:
        raise ValueError("Krea NAG requires CFG 1.")
    if sampling.get("use_nag") and len(negative) != 1:
        raise ValueError("Krea NAG needs one scheduled negative conditioning entry.")
    sources = [model.model.process_latent_in(source) for source in sources]
    negative_context = negative[0][0] if sampling.get("use_nag") else None

    boosted = any(boost != 1.0 for boost in (edit.get("ref_boost", 1.0), edit.get("ref_boost_a", 1.0))) or any(mask is not None for mask in edit.get("masks") or ())

    def wrapper(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options=None, **kwargs):
        options = transformer_options or {}
        if sampling.get("use_nag") and any(branch != 0 for branch in options.get("cond_or_uncond", [0])):
            raise ValueError("Krea NAG needs the positive CFG branch only.")
        if attention_mask is not None or ref_latents:
            raise ValueError("Use Krea Edit Prepare for references; this adapter does not consume native reference conditioning or text masks.")
        if not boosted:
            return _forward(executor.class_obj, x, timesteps, context, sources, negative_context, edit, sampling, options)
        # a reference bias is a dense mask, which the xformers kernel rejects at these shapes
        with _pytorch_attention():
            return _forward(executor.class_obj, x, timesteps, context, sources, negative_context, edit, sampling, options)

    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "promptforge_v2_krea", wrapper)
    return patched


class KreaEditPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("KREA", "EDIT_PREPARE", "Edit Prepare", inputs=[Pipe.Input("pipe", optional=True), io.Image.Input("image"),
                io.Image.Input("image_b", optional=True), io.Mask.Input("mask", optional=True, tooltip="Identity area of the first reference (for example face, hair and neck). The text encoder then sees only this area on grey, so clothes and room do not carry over; the reference weight applies to it alone."), io.Mask.Input("mask_b", optional=True, tooltip="The same for the second reference."),
                io.Boolean.Input("crop_to_mask", default=False, tooltip="Crop each masked reference to its mask with a margin before use: the face gets the whole reference grid and nothing else of the photo (clothes, room, arms) is left to copy."),
                io.Combo.Input("fit_mode", options=["fit", "crop"], default="fit", tooltip="How the reference is fitted to the format. fit keeps the whole photo, crop fills the frame."),
                io.Combo.Input("position", options=["aligned", "beside", "bag"], default="aligned", tooltip="Where the main reference sits in the token grid. aligned is the trained position."),
                io.Combo.Input("position_a", options=["aligned", "beside", "bag"], default="aligned", tooltip="Where the second reference sits. aligned is the trained position."),
                io.Float.Input("ref_boost", default=1.0, min=0.0, max=1000.0, tooltip="Strength of the LAST reference (the second one when two are connected, the only one otherwise). With a mask it acts on the masked area only. 1 is neutral, 2 is a good start, higher copies it harder and above about 4 can hide the person."),
                io.Float.Input("ref_boost_a", default=1.0, min=0.0, max=1000.0, tooltip="Strength of the FIRST reference when two are connected. With a mask it acts on the masked area only. 1 is neutral; lower it when that photo should only hint at the outfit or angle.")], outputs=[Pipe.Output("pipe")])

    @classmethod
    def execute(cls, pipe=None, image=None, image_b=None, mask=None, mask_b=None, crop_to_mask=False, fit_mode="fit", position="aligned", position_a="aligned", ref_boost=1.0, ref_boost_a=1.0):
        pipe = ensure_pipe(pipe, family="krea2")
        if pipe["family"] != "krea2":
            raise ValueError("Krea Edit Prepare requires a krea2 pipe and an identity edit LoRA on the model.")
        require(pipe, "vae", "model")
        fmt = pipe["settings"].get("format")
        if fmt is None:
            raise ValueError("Set Image Format before Krea Edit Prepare.")
        references = [image] + ([image_b] if image_b is not None else [])
        masks = [mask, mask_b][:len(references)]
        if crop_to_mask:
            cropped = [reference if reference_mask is None else _crop_to_mask(reference, reference_mask) for reference, reference_mask in zip(references, masks)]
            references = [reference for reference, _ in cropped]
            masks = [None if reference_mask is None else crop_mask for reference_mask, (_, crop_mask) in zip(masks, cropped)]
        source_latents = [_fit_encode_image(ref, pipe["artifacts"]["vae"], fmt["height"] // 8, fmt["width"] // 8,
                                            {}, (), fit_mode) for ref in references]
        settings = {"fit_mode": fit_mode, "position": position, "position_a": position_a,
                        "ref_boost": ref_boost, "ref_boost_a": ref_boost_a}
        text_references = []
        for reference, reference_mask in zip(references, masks):
            if reference_mask is None:
                text_references.append(reference)
                continue
            keep = F.interpolate(reference_mask.reshape(-1, 1, *reference_mask.shape[-2:])[:1].float(), size=reference.shape[1:3], mode="bilinear").movedim(1, -1).clamp(0, 1)
            text_references.append(reference[..., :3] * keep + .5 * (1 - keep))
        masks = [None if reference_mask is None else _fit_reference(reference_mask.reshape(-1, *reference_mask.shape[-2:], 1), fmt["height"] // 8, fmt["width"] // 8, fit_mode)[..., 0]
                 for reference_mask in masks]
        artifacts = {"krea_references": references, "krea_text_references": text_references, "krea_source_latents": source_latents, "krea_masks": masks}
        return io.NodeOutput(evolve(pipe, section="krea_edit", settings=settings, artifacts=artifacts,
                                    invalidate=("positive", "negative", "latent", "images")))


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_V2_KREA_EDIT_PREPARE": KreaEditPrepare}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_V2_KREA_EDIT_PREPARE": "PromptForge V2 · Krea 2 · Edit Prepare"}
