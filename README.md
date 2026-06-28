# ComfyUI-ClarkAirSana

Run the **Clark Air Sana 1.6B** transformer — ternary (~1.58-bit) weights on real **GemLite INT2**
CUDA kernels — natively in ComfyUI under the standard **KSampler**. The `ClarkAirSanaLoader` node
builds ComfyUI_ExtraModels' `SanaMS` skeleton and swaps its attention + GLU-FFN trunk for GemLite
layers (~0.5 GB), leaving the small "island" layers in bf16.

## Requirements

- **NVIDIA CUDA GPU** — GemLite INT2 is CUDA-only.
- [`ComfyUI_ExtraModels`](https://github.com/lawrence-cj/ComfyUI_ExtraModels) (provides Sana,
  the `GemmaLoader`/`GemmaTextEncode` text encoder, and the DC-AE VAE loader). Pin a known-good
  commit.
- `pip install -r requirements.txt` (`gemlite`, `safetensors`, `numpy`).

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/lawrence-cj/ComfyUI_ExtraModels   # if not already present
git clone <this repo> ComfyUI-ClarkAirSana
pip install -r ComfyUI-ClarkAirSana/requirements.txt
```

Download the transformer pack into `ComfyUI/models/clark_air_sana/`:

```
huggingface-cli download clark-labs/clark-air-sana-1.6b-gemlite-2bit \
  clark_air_sana_gemlite_comfy.safetensors --local-dir ComfyUI/models/clark_air_sana
```

`GemmaLoader` fetches the text encoder and `ExtraVAELoader` the DC-AE VAE on first run.

**Footprint (~3.2 GB total).** The example workflow defaults to the small components:

| Component | Download | Note |
|---|---|---|
| transformer (this repo) | **495 MB** | ternary, GemLite INT2 + FP8 islands |
| Gemma text encoder | **~2.1 GB** | `unsloth/gemma-2-2b-it-bnb-4bit` (4-bit). Swap to `Efficient-Large-Model/gemma-2-2b-it` for fp16 (9.8 GB) if you prefer. |
| DC-AE VAE | 1.2 GB (fp32) | loaded in bf16 at runtime; drop a bf16 copy (624 MB) at the VAE path to halve the download. |

## Graph

```
GemmaLoader ─┬─ GemmaTextEncode (positive) ─┐
             └─ GemmaTextEncode (negative) ─┤
ClarkAirSanaLoader ──────── MODEL ──────────┼─ KSampler ─ VAEDecode ─ SaveImage
ExtraVAELoader (dcae-f32c32-sana) ── VAE ────┘     │
EmptySanaLatentImage ───────── LATENT ─────────────┘
```

KSampler: **`euler` / `normal`, 20 steps, cfg 4.5, 512×512** (verified end-to-end). See
`example_workflow.json` (API format). The trunk runs on real GemLite INT2 kernels — identical
math to the standalone render.

![demo](demo.png)

## Notes / limitations

- **Keep the transformer GPU-resident.** GemLite holds packed codes as buffers; ComfyUI's lowvram
  weight-streaming can't move them. The loader pins the trunk on the GPU (`offload_device =
  load_device`); the ~0.5 GB trunk fits easily, and Gemma/VAE still offload normally. Avoid
  `--lowvram` for the transformer.
- The null-caption bank (`y_embedder.y_embedding`) isn't in the diffusers weights; it's unused at
  inference with a real negative prompt (the skeleton's value is left in place).
- Trunk math is identical to the standalone GemLite render; only scheduler conventions differ.
