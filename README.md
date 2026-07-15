# sd-forge-anima-teacache

TeaCache acceleration for the **Anima** model on **Forge Neo**.

Port of [ComfyUI-Anima-TeaCache](https://github.com/CocyNoric/ComfyUI-Anima-TeaCache)
(cache-decision logic reused as-is; the model hook is reimplemented for Forge Neo's backend).

## What it does

Between adjacent sampling steps, the modulated input of the first transformer
block usually changes very little. When the accumulated relative L1 change stays
under a threshold, the whole transformer block stack is skipped and the residual
cached from the previous full step is reused. Typical speedup: 1.3x - 2.0x,
depending on threshold and step count.

## Usage

1. Place this folder in `ForgeNeo/extensions/`.
2. Load an Anima checkpoint.
3. In txt2img / img2img, open the **Anima TeaCache** accordion, tick **Enabled**.

| Setting | Meaning |
|---|---|
| `rel_l1_thresh` | Higher = more skipping = faster, lower quality. Typical 0.05 - 0.20. 0 disables. |
| `Start/End percent` | Portion of the sampling run where caching is allowed. First and last steps are always fully computed. |
| `Cache device` | `GPU` keeps caches in VRAM (fast). `CPU` offloads them to RAM. |

Works with hires fix and img2img (each sampling pass gets a fresh cache).
The console prints how many transformer passes were skipped after each run.

## Notes

- Only affects Anima checkpoints; silently inactive for any other model.
- `anima_teacache_runtime.py` mirrors `backend/nn/anima.py::Anima.forward` and
  must be kept in sync if Forge Neo changes that file.
- Measured on an RTX 4070 (896x1216, 32 steps, Res Multistep): `rel_l1_thresh=0.2`
  skipped 72% of transformer passes, 24s -> 7s (~3.4x). Note that aggressive
  thresholds with `Start percent = 0` can change the composition even with a
  fixed seed; use `rel_l1_thresh=0.05` + `Start percent = 0.2` to stay close
  to the uncached result.

## License and credits

**AGPL-3.0** (required: the transformer forward in `anima_teacache_runtime.py`
is derived from [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic)'s
`backend/nn/anima.py`, which is AGPL-3.0).

- Cache-decision logic ported from
  [CocyNoric/ComfyUI-Anima-TeaCache](https://github.com/CocyNoric/ComfyUI-Anima-TeaCache) (Apache-2.0).
- TeaCache method: [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache).
- Anima model implementation lineage: NVIDIA Cosmos-Predict2 / ComfyUI / Forge Neo.
