import datetime
import json
from pathlib import Path

import gradio as gr
import torch

from modules import processing, script_callbacks, scripts
from modules.infotext_utils import PasteField

import anima_teacache_runtime as teacache

# Auto-mode presets. The sigma bounds are compared against the live noise level
# each step, so the effective skip window follows Shift, scheduler and step
# count automatically.
PROTECTION_PRESETS = {"Light": 0.90, "Standard": 0.80, "Strong": 0.70}
"""Skipping is allowed only once sigma has dropped below this guard value."""
SPEED_PRESETS = {"Safe": 0.10, "Standard": 0.20, "Aggressive": 0.35}
"""rel_l1_thresh used by Auto mode. Calibrated 2026-07-17 from debug logs
(1024x1024, ER SDE/Simple, steps 12-48 x shift 2-6): in-range per-step poly
is ~0.08 at 32 steps and scales ~1/steps, so these give skip-run lengths of
roughly 1 / 2 / 3-4 steps at 32 steps. Below ~20 steps even Safe skips little
or nothing by design (per-step drift already exceeds the cap)."""


def _on_cfg_denoiser(params):
    runtime = teacache.get_active_runtime()
    if runtime is None:
        return
    sigma = params.sigma
    if torch.is_tensor(sigma):
        sigma = float(sigma.flatten()[0].item()) if sigma.numel() > 0 else None
    elif sigma is not None:
        sigma = float(sigma)
    denoiser = params.denoiser
    if denoiser is not None and getattr(denoiser, "total_steps", None):
        runtime.set_step(denoiser.step, denoiser.total_steps, sigma)
    else:
        runtime.set_step(params.sampling_step, params.total_sampling_steps, sigma)


script_callbacks.on_cfg_denoiser(_on_cfg_denoiser, name="anima-teacache")


_debug_dump_counter = 0


def _dump_debug_log(runtime):
    global _debug_dump_counter
    _debug_dump_counter += 1
    logs_dir = Path(teacache.__file__).resolve().parent / "logs"
    try:
        logs_dir.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = logs_dir / f"teacache-{stamp}-{_debug_dump_counter:03d}.json"
        payload = {
            "meta": runtime.debug_meta,
            "summary": {
                "calls_total": runtime.calls_total,
                "calls_skipped": runtime.calls_skipped,
            },
            "records": runtime.log_records,
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"[Anima TeaCache] debug log written to {path}")
    except OSError as e:
        print(f"[Anima TeaCache] failed to write debug log: {e}")


def _report(runtime):
    if runtime is None or runtime.calls_total == 0:
        return
    ratio = 100.0 * runtime.calls_skipped / runtime.calls_total
    print(f"[Anima TeaCache] skipped {runtime.calls_skipped}/{runtime.calls_total} transformer passes ({ratio:.0f}%)")
    if runtime.debug and runtime.log_records:
        call_map = "".join("." if record["action"] == "skip" else "#" for record in runtime.log_records)
        print(f"[Anima TeaCache] call map (#=compute, .=skip): {call_map}")
        _dump_debug_log(runtime)


class AnimaTeaCacheScript(scripts.Script):
    def __init__(self):
        self.warned = False
        self.notified_auto_pass = False

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
            mode = gr.Radio(
                label="Mode",
                choices=["Auto", "Advanced"],
                value="Auto",
                info="Auto tunes the skip window from the live noise level (txt2img first pass only; hires/img2img run uncached). Advanced exposes the raw knobs.",
                elem_id=f"{tab}_anima_teacache_mode",
            )
            with gr.Group(visible=True) as auto_group:
                with gr.Row():
                    protection = gr.Radio(
                        label="Composition protection",
                        choices=list(PROTECTION_PRESETS),
                        value="Standard",
                        info="no skipping while the noise level is above the guard; adapts to Shift / steps / scheduler automatically",
                        elem_id=f"{tab}_anima_teacache_protection",
                    )
                    speed = gr.Radio(
                        label="Speed",
                        choices=list(SPEED_PRESETS),
                        value="Standard",
                        info="higher = more skipping = faster but lower fidelity",
                        elem_id=f"{tab}_anima_teacache_speed",
                    )
            with gr.Group(visible=False) as advanced_group:
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
                        info="trajectory fraction, converted to a noise level via the model schedule",
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
            debug_logging = gr.Checkbox(
                label="Debug logging",
                value=False,
                info="print per-step cache decisions and write a JSON log to the extension's logs/ folder (for calibration)",
                elem_id=f"{tab}_anima_teacache_debug",
            )

            mode.change(
                fn=lambda m: (gr.update(visible=m == "Auto"), gr.update(visible=m == "Advanced")),
                inputs=[mode],
                outputs=[auto_group, advanced_group],
                show_progress="hidden",
            )

        self.infotext_fields = [
            PasteField(enabled, lambda d: "Anima TeaCache thresh" in d),
            PasteField(mode, lambda d: d.get("Anima TeaCache mode", "Advanced" if "Anima TeaCache thresh" in d else "Auto")),
            PasteField(protection, lambda d: d.get("Anima TeaCache protection", "Standard")),
            PasteField(speed, lambda d: d.get("Anima TeaCache speed", "Standard")),
            PasteField(rel_l1_thresh, "Anima TeaCache thresh"),
            PasteField(start_percent, "Anima TeaCache start"),
            PasteField(end_percent, "Anima TeaCache end"),
            PasteField(cache_device, "Anima TeaCache device"),
        ]

        return [enabled, mode, protection, speed, rel_l1_thresh, start_percent, end_percent, cache_device, debug_logging]

    def process(self, p, *args, **kwargs):
        self.warned = False
        self.notified_auto_pass = False

    def process_before_every_sampling(self, p, enabled, mode, protection, speed, rel_l1_thresh, start_percent, end_percent, cache_device, debug_logging, **kwargs):
        # a fresh runtime per sampling pass (txt2img / hires fix / img2img each get their own)
        previous = teacache.set_active_runtime(None)
        _report(previous)

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet
        diffusion_model = getattr(unet.model, "diffusion_model", None)
        if not isinstance(diffusion_model, teacache.Anima):
            if not self.warned:
                print("[Anima TeaCache] current model is not Anima; caching disabled")
                self.warned = True
            return

        if mode == "Auto":
            is_first_txt2img_pass = isinstance(p, processing.StableDiffusionProcessingTxt2Img) and not getattr(p, "is_hr_pass", False)
            if not is_first_txt2img_pass:
                if not self.notified_auto_pass:
                    print("[Anima TeaCache] Auto mode caches only the first txt2img pass; this pass runs uncached (use Advanced mode for manual control)")
                    self.notified_auto_pass = True
                return
            thresh = SPEED_PRESETS.get(speed, SPEED_PRESETS["Standard"])
            start_sigma = PROTECTION_PRESETS.get(protection, PROTECTION_PRESETS["Standard"])
            end_sigma = 0.0
            # only reached if the denoiser callback ever fails to deliver sigma
            fallback_start, fallback_end = 0.4, 1.0
        else:
            thresh = rel_l1_thresh
            if thresh <= 0:
                return
            if start_percent > end_percent:
                print("[Anima TeaCache] start percent > end percent; caching disabled")
                return
            predictor = getattr(unet.model, "predictor", None)
            percent_to_sigma = getattr(predictor, "percent_to_sigma", None)
            if callable(percent_to_sigma):
                # trajectory fraction -> noise level under the live schedule; unlike a
                # step-index ratio this stays correct for Karras-style schedulers and
                # partial img2img trajectories
                start_sigma = float(percent_to_sigma(start_percent))
                end_sigma = float(percent_to_sigma(end_percent))
            else:
                start_sigma = None
                end_sigma = None
            fallback_start, fallback_end = start_percent, end_percent

        device = torch.device("cpu") if cache_device == "CPU" else None
        runtime = teacache.AnimaTeaCacheRuntime(
            rel_l1_thresh=thresh,
            start_percent=fallback_start,
            end_percent=fallback_end,
            cache_device=device,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            debug=bool(debug_logging),
        )
        if debug_logging:
            is_hr_pass = bool(getattr(p, "is_hr_pass", False))
            runtime.debug_meta = {
                "mode": mode,
                "thresh": thresh,
                "sigma_start": start_sigma,
                "sigma_end": end_sigma,
                "protection": protection if mode == "Auto" else None,
                "speed": speed if mode == "Auto" else None,
                "steps": getattr(p, "steps", None),
                "hr_pass": is_hr_pass,
                "sampler": getattr(p, "sampler_name", None),
                "scheduler": getattr(p, "scheduler", None),
                "shift": getattr(p, "hr_distilled_cfg", None) if is_hr_pass else getattr(p, "distilled_cfg_scale", None),
                "width": getattr(p, "width", None),
                "height": getattr(p, "height", None),
                "seed": getattr(p, "seed", None),
            }
        teacache.install_forward_patch()
        teacache.set_active_runtime(runtime)

        p.extra_generation_params["Anima TeaCache mode"] = mode
        p.extra_generation_params["Anima TeaCache thresh"] = thresh
        if mode == "Auto":
            p.extra_generation_params["Anima TeaCache protection"] = protection
            p.extra_generation_params["Anima TeaCache speed"] = speed
        else:
            p.extra_generation_params["Anima TeaCache start"] = start_percent
            p.extra_generation_params["Anima TeaCache end"] = end_percent
        if start_sigma is not None and end_sigma is not None:
            p.extra_generation_params["Anima TeaCache sigma start"] = round(start_sigma, 4)
            p.extra_generation_params["Anima TeaCache sigma end"] = round(end_sigma, 4)
        p.extra_generation_params["Anima TeaCache device"] = cache_device

    def postprocess(self, p, processed, *args):
        previous = teacache.set_active_runtime(None)
        _report(previous)
