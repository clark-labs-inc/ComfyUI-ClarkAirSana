# ComfyUI-ClarkAirSana

Run **Clark Air Sana 1.6B** — ternary (~1.58-bit) weights on real **GemLite INT2** CUDA kernels —
natively in ComfyUI under the standard **KSampler**. **Self-contained:** the Sana model, the Gemma
text encoder, and the DC-AE VAE are all included, so there is **no other custom node to install**.

## Quick start

1. In **ComfyUI-Manager**, install **Clark Air Sana** and restart ComfyUI. (That's the only node
   pack you need. The Python deps install automatically.)
2. **Download the model** → put it in `ComfyUI/models/clark_air_sana/`:
   **[clark_air_sana_gemlite_comfy.safetensors](https://huggingface.co/clark-labs/clark-air-sana-1.6b-gemlite-2bit/resolve/main/clark_air_sana_gemlite_comfy.safetensors)** (495 MB).
3. **⬇ Download the workflow:** **[example_workflow.json](https://github.com/clark-labs-inc/ComfyUI-ClarkAirSana/raw/main/example_workflow.json)** → drag it onto the ComfyUI canvas → press **Queue**.

The Gemma text encoder and DC-AE VAE download themselves on the first run.

![demo](demo.png)

## Requirements

- **NVIDIA CUDA GPU.** GemLite INT2 runs on **Triton** kernels, so **Linux or WSL2 is the smooth
  path**. On native Windows you need `triton-windows` (the deps file installs it automatically,
  but Triton on native Windows is community-supported and may not work on every setup — WSL2 is
  recommended).
- Python deps (installed for you from `requirements.txt`): `gemlite`, `bitsandbytes`, `triton`,
  `transformers`, `diffusers`, `accelerate`, `timm`, `einops`, `safetensors`, `numpy`.
  If your ComfyUI is the **portable build**, the Manager handles this; to do it by hand use the
  bundled Python: `.\python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI-ClarkAirSana\requirements.txt`.

## Nodes

All provided by this pack (category **ClarkAir/Sana**):

| Node | Output | Role |
|---|---|---|
| Clark Air Sana Loader | MODEL | the GemLite INT2 ternary transformer |
| Clark Air Gemma Loader / Encode | CONDITIONING | Gemma-2 text encoder (4-bit by default) |
| Clark Air DC-AE VAE Loader / Decode | IMAGE | DC-AE latents → pixels |
| Clark Air Sana Empty Latent | LATENT | 32-channel Sana latent |

## Workflow

`example_workflow.json` (drag onto the canvas) wires:

```
Clark Air Gemma Loader ─┬─ Gemma Encode (positive) ─┐
                        └─ Gemma Encode (negative) ─┤
Clark Air Sana Loader ───────── MODEL ──────────────┼─ KSampler ─ Clark Air VAE Decode ─ SaveImage
Clark Air DC-AE VAE Loader ──── VAE ─────────────────┘     │
Clark Air Sana Empty Latent ─── LATENT ───────────────────┘
```

KSampler: **`euler` / `normal`, 20 steps, cfg 4.5, 512×512** (verified end-to-end through ComfyUI's
KSampler). `example_workflow_api.json` is the same graph in API format.

## Footprint (~3.2 GB)

| Component | Download |
|---|---|
| transformer (this repo) | **495 MB** (ternary, GemLite INT2 + FP8 islands) |
| Gemma text encoder | **~2.1 GB** (`unsloth/gemma-2-2b-it-bnb-4bit`, 4-bit; switch to `Efficient-Large-Model/gemma-2-2b-it` for fp16) |
| DC-AE VAE | 1.2 GB (loaded bf16) |

## Notes

- **Keep the transformer GPU-resident.** GemLite holds packed codes as buffers that ComfyUI's
  lowvram weight-streaming can't move; the loader pins the ~0.5 GB trunk on the GPU. Avoid
  `--lowvram` for it (Gemma/VAE are fine).
- The Sana model code under `sana/` is vendored from ComfyUI_ExtraModels (Apache-2.0), which adapts
  NVlabs/Sana — see `NOTICE`. This pack bundles it so it runs with no external node dependency.
