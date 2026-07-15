import gradio as gr
import torch

from modules import script_callbacks, scripts
from modules.infotext_utils import PasteField

import anima_teacache_runtime as teacache


def _on_cfg_denoiser(params):
    runtime = teacache.get_active_runtime()
    if runtime is None:
        return
    denoiser = params.denoiser
    if denoiser is not None and getattr(denoiser, "total_steps", None):
        runtime.set_step(denoiser.step, denoiser.total_steps)
    else:
        runtime.set_step(params.sampling_step, params.total_sampling_steps)


script_callbacks.on_cfg_denoiser(_on_cfg_denoiser, name="anima-teacache")


def _report(runtime):
    if runtime is None or runtime.calls_total == 0:
        return
    ratio = 100.0 * runtime.calls_skipped / runtime.calls_total
    print(f"[Anima TeaCache] skipped {runtime.calls_skipped}/{runtime.calls_total} transformer passes ({ratio:.0f}%)")


class AnimaTeaCacheScript(scripts.Script):
    def __init__(self):
        self.warned = False

    def title(self):
        return "Anima TeaCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(
                label="Enabled",
                value=False,
                elem_id=f"{tab}_anima_teacache_enabled",
            )
            rel_l1_thresh = gr.Slider(
                label="rel_l1_thresh",
                info="higher = more skipping = faster but lower quality (typical: 0.05 - 0.20)",
                minimum=0.0,
                maximum=1.0,
                step=0.001,
                value=0.05,
                elem_id=f"{tab}_anima_teacache_thresh",
            )
            with gr.Row():
                start_percent = gr.Slider(
                    label="Start percent",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    elem_id=f"{tab}_anima_teacache_start",
                )
                end_percent = gr.Slider(
                    label="End percent",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    elem_id=f"{tab}_anima_teacache_end",
                )
            cache_device = gr.Radio(
                label="Cache device",
                choices=["GPU", "CPU"],
                value="GPU",
                elem_id=f"{tab}_anima_teacache_device",
            )

        self.infotext_fields = [
            PasteField(enabled, lambda d: "Anima TeaCache thresh" in d),
            PasteField(rel_l1_thresh, "Anima TeaCache thresh"),
            PasteField(start_percent, "Anima TeaCache start"),
            PasteField(end_percent, "Anima TeaCache end"),
            PasteField(cache_device, "Anima TeaCache device"),
        ]

        return [enabled, rel_l1_thresh, start_percent, end_percent, cache_device]

    def process(self, p, *args, **kwargs):
        self.warned = False

    def process_before_every_sampling(self, p, enabled, rel_l1_thresh, start_percent, end_percent, cache_device, **kwargs):
        # a fresh runtime per sampling pass (txt2img / hires fix / img2img each get their own)
        previous = teacache.set_active_runtime(None)
        _report(previous)

        if not enabled or rel_l1_thresh <= 0:
            return
        if start_percent > end_percent:
            print("[Anima TeaCache] start percent > end percent; caching disabled")
            return

        unet = p.sd_model.forge_objects.unet
        diffusion_model = getattr(unet.model, "diffusion_model", None)
        if not isinstance(diffusion_model, teacache.Anima):
            if not self.warned:
                print("[Anima TeaCache] current model is not Anima; caching disabled")
                self.warned = True
            return

        device = torch.device("cpu") if cache_device == "CPU" else None
        runtime = teacache.AnimaTeaCacheRuntime(
            rel_l1_thresh=rel_l1_thresh,
            start_percent=start_percent,
            end_percent=end_percent,
            cache_device=device,
        )
        teacache.install_forward_patch()
        teacache.set_active_runtime(runtime)

        p.extra_generation_params["Anima TeaCache thresh"] = rel_l1_thresh
        p.extra_generation_params["Anima TeaCache start"] = start_percent
        p.extra_generation_params["Anima TeaCache end"] = end_percent
        p.extra_generation_params["Anima TeaCache device"] = cache_device

    def postprocess(self, p, processed, *args):
        previous = teacache.set_active_runtime(None)
        _report(previous)
