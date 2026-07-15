# TeaCache for the Anima model on Forge Neo.
# License: AGPL-3.0 (see LICENSE)
#
# The cache-decision logic (first-block modulated-input L1 distance, polynomial
# rescale, accumulated threshold) is ported from:
#   https://github.com/CocyNoric/ComfyUI-Anima-TeaCache (nodes.py, Apache-2.0)
# The transformer forward below mirrors backend/nn/anima.py::Anima.forward of
# Forge Neo (https://github.com/Haoming02/sd-webui-forge-classic, AGPL-3.0)
# and must be kept in sync with it.

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from backend.args import dynamic_args
from backend.nn.anima import Anima
from backend.utils import pad_to_patch_size

ANIMA_COEFFICIENTS = (
    16.440769544691957,
    -21.14123745515793,
    13.006701290921894,
    -1.6731977662609427,
    0.08632681059623949,
)

_ORIGINAL_FORWARD_ATTR = "_anima_teacache_original_forward"
_RUNTIME_ATTR = "_anima_teacache_runtime"


def poly1d(coefficients, value):
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


@dataclass
class CacheEntry:
    previous_modulated_input: Optional[torch.Tensor] = None
    previous_residual: Optional[torch.Tensor] = None
    accumulated_distance: float = 0.0
    signature: Optional[tuple] = None


@dataclass
class CacheDecision:
    should_calc: bool
    cacheable: bool
    keys: list
    slices: list


class AnimaTeaCacheRuntime:
    def __init__(self, rel_l1_thresh: float, start_percent: float, end_percent: float, cache_device: Optional[torch.device] = None):
        self.rel_l1_thresh = rel_l1_thresh
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.cache_device = cache_device
        self.entries: dict[Any, CacheEntry] = {}
        self.step_info: Optional[tuple[int, int]] = None
        """(current denoiser call index, expected total denoiser calls) - fed per call by the on_cfg_denoiser callback"""
        self.last_step_index: Optional[int] = None
        self.calls_total = 0
        self.calls_skipped = 0

    def reset_entries(self):
        self.entries.clear()
        self.last_step_index = None

    def set_step(self, index: Optional[int], total: Optional[int]):
        if index is None or not total:
            self.step_info = None
        else:
            self.step_info = (int(index), int(total))

    def _split_layout(self, tensor: torch.Tensor, transformer_options: dict):
        labels = transformer_options.get("cond_or_uncond")
        if not labels:
            return [0], [slice(0, tensor.shape[0])]
        if tensor.shape[0] % len(labels) != 0:
            return None

        chunk_size = tensor.shape[0] // len(labels)
        occurrences = {}
        keys = []
        slices = []
        for index, label in enumerate(labels):
            occurrence = occurrences.get(label, 0)
            occurrences[label] = occurrence + 1
            keys.append((label, occurrence))
            slices.append(slice(index * chunk_size, (index + 1) * chunk_size))
        return keys, slices

    def decide(self, modulated_input: torch.Tensor, transformer_options: dict) -> CacheDecision:
        step_info = self.step_info
        layout = self._split_layout(modulated_input, transformer_options)
        if step_info is None or layout is None:
            self.reset_entries()
            return CacheDecision(True, False, [], [])

        step_index, total_steps = step_info
        if self.last_step_index is not None and step_index < self.last_step_index:
            self.reset_entries()
        self.last_step_index = step_index

        keys, slices = layout
        percent = step_index / max(total_steps, 1)
        in_range = self.start_percent <= percent <= self.end_percent
        boundary = step_index == 0 or step_index >= total_steps - 1
        force_full = not in_range or boundary

        current_inputs = []
        for key, section in zip(keys, slices):
            current = modulated_input[section].detach()
            if self.cache_device is not None:
                current = current.to(self.cache_device)
            current_inputs.append(current)
            entry = self.entries.setdefault(key, CacheEntry())
            signature = (tuple(current.shape), current.dtype)

            if entry.signature != signature:
                entry.previous_modulated_input = None
                entry.previous_residual = None
                entry.accumulated_distance = 0.0
                entry.signature = signature
                force_full = True

            previous = entry.previous_modulated_input
            if previous is None or entry.previous_residual is None:
                force_full = True
            elif not force_full:
                denominator = previous.abs().mean()
                if not torch.isfinite(denominator) or denominator.item() <= 0.0:
                    force_full = True
                else:
                    relative_l1 = ((current - previous).abs().mean() / denominator).float().item()
                    scaled = poly1d(ANIMA_COEFFICIENTS, relative_l1)
                    if not math.isfinite(scaled):
                        force_full = True
                    else:
                        entry.accumulated_distance += scaled
                        if entry.accumulated_distance >= self.rel_l1_thresh:
                            force_full = True

        for key, current in zip(keys, current_inputs):
            self.entries[key].previous_modulated_input = current

        if force_full:
            for key in keys:
                self.entries[key].accumulated_distance = 0.0

        return CacheDecision(force_full, True, keys, slices)

    def commit_residual(self, decision: CacheDecision, residual: torch.Tensor):
        if not decision.cacheable:
            return
        for key, section in zip(decision.keys, decision.slices):
            cached = residual[section].detach()
            if self.cache_device is not None:
                cached = cached.to(self.cache_device)
            self.entries[key].previous_residual = cached

    def reuse_residual(self, decision: CacheDecision, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = []
        for key in decision.keys:
            residual = self.entries[key].previous_residual
            if residual is None:
                raise RuntimeError("Anima TeaCache residual is not initialized")
            residuals.append(residual.to(device=hidden_states.device, dtype=hidden_states.dtype))
        return hidden_states + torch.cat(residuals, dim=0)

    def _first_block_modulated_input(self, dm: Anima, hidden_states, t_embedding, adaln_lora, extra_pos_emb):
        block = dm.blocks[0]
        block_input = hidden_states
        if extra_pos_emb is not None:
            block_input = block_input + extra_pos_emb

        modulation = block.adaln_modulation_self_attn(t_embedding) + adaln_lora
        shift, scale, _gate = modulation.chunk(3, dim=-1)
        shift = shift[:, :, None, None, :]
        scale = scale[:, :, None, None, :]
        return block.layer_norm_self_attn(block_input) * (1 + scale) + shift

    def run(self, dm: Anima, x, timesteps, context, padding_mask=None, **kwargs):
        orig_shape = list(x.shape)

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents
        for ref in ref_latents:
            if x.shape[0] == 2:  # batch_cond_uncond
                ref = torch.cat((ref, ref), dim=0)
            x = torch.cat((x, ref.to(x)), dim=2)

        x = pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
        x_B_C_T_H_W = x
        timesteps_B_T = timesteps
        crossattn_emb = context

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_None = dm.prepare_embedded_sequence(x_B_C_T_H_W, padding_mask=padding_mask)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = dm.t_embedder[1](dm.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
        t_embedding_B_T_D = dm.t_embedding_norm(t_embedding_B_T_D)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_None,
            "transformer_options": kwargs.get("transformer_options", {}),
        }

        # To make fp16 compute_dtype work, we keep the residual stream in fp32 but run attention and MLP modules in fp16.
        if x_B_T_H_W_D.dtype is torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        modulated_input = self._first_block_modulated_input(dm, x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D, extra_pos_emb_None)
        decision = self.decide(modulated_input, block_kwargs["transformer_options"])

        self.calls_total += 1
        if decision.should_calc:
            original_hidden_states = x_B_T_H_W_D
            for block in dm.blocks:
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    **block_kwargs,
                )
            self.commit_residual(decision, x_B_T_H_W_D - original_hidden_states)
        else:
            self.calls_skipped += 1
            x_B_T_H_W_D = self.reuse_residual(decision, x_B_T_H_W_D)

        x_B_T_H_W_O = dm.final_layer(x_B_T_H_W_D.to(crossattn_emb.dtype), t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = dm.unpatchify(x_B_T_H_W_O)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
        return x_B_C_Tt_Hp_Wp


def install_forward_patch():
    if getattr(Anima, _ORIGINAL_FORWARD_ATTR, None) is not None:
        return
    original_forward = Anima.forward

    def forward(self, x, timesteps, context, padding_mask=None, **kwargs):
        runtime = getattr(Anima, _RUNTIME_ATTR, None)
        if runtime is None:
            return original_forward(self, x, timesteps, context, padding_mask=padding_mask, **kwargs)
        return runtime.run(self, x, timesteps, context, padding_mask=padding_mask, **kwargs)

    setattr(Anima, _ORIGINAL_FORWARD_ATTR, original_forward)
    Anima.forward = forward


def get_active_runtime() -> Optional[AnimaTeaCacheRuntime]:
    return getattr(Anima, _RUNTIME_ATTR, None)


def set_active_runtime(runtime: Optional[AnimaTeaCacheRuntime]) -> Optional[AnimaTeaCacheRuntime]:
    previous = getattr(Anima, _RUNTIME_ATTR, None)
    setattr(Anima, _RUNTIME_ATTR, runtime)
    return previous
